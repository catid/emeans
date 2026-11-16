"""Baseline k-means solvers for rigorous Phase 1 evaluation against e-means and e-means++.

Provides:
1. `LloydKMeans`: Fast GPU/CPU classical Lloyd's algorithm (random or greedy k-means++ init),
   sharing the exact same Triton/PyTorch/Faiss assignment kernels and empty-cluster donor
   splitting as e-means for apples-to-apples wall-clock & MSE comparison.
2. `SculleyMiniBatchKMeans`: Sculley (WWW 2010) mini-batch k-means with harmonic 1 / V_j
   cumulative-count step size and `reassignment_ratio` dead-cluster resampling, supporting
   both GPU-native Triton/PyTorch execution and scikit-learn's `MiniBatchKMeans`.
3. `NaiveKMeansPP_EMeans`: Full-dataset Greedy k-means++ initialization fed directly into
   vanilla `EMeans` (N_0 = 2^-16, count-only revival), reproducing the Table `init_scale`
   baseline of arXiv:2611.09501.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from .emeans import EMeans, _separate_collapsed_splits
from .kernels import assign_points, prepare_represented_centroids
from .recipe import DEFAULT_INIT_USAGE, SPLIT_EPSILON, epoch_seed
from .seeding import greedy_kmeans_plusplus_torch


@dataclass
class LloydIterStat:
    iteration: int
    mse: float
    n_splits: int
    elapsed_sec: float


class LloydKMeans:
    """Classical full-batch Lloyd's k-means on GPU or CPU (random or greedy k-means++ init)."""

    def __init__(
        self,
        d: int,
        k: int,
        niter: int = 25,
        init: str = "random",
        tol: float = -1.0,
        revive_empty: bool = True,
        split_epsilon: float = SPLIT_EPSILON,
        compute_dtype: torch.dtype | None = None,
        backend: str = "auto",
        seed: int = 1234,
        verbose: bool = False,
    ) -> None:
        if init not in ("random", "k-means++"):
            raise ValueError("init must be 'random' or 'k-means++'")
        if niter < 1:
            raise ValueError("niter must be >= 1")

        self.d = int(d)
        self.k = int(k)
        self.niter = int(niter)
        self.init = init
        self.tol = float(tol)
        self.revive_empty = bool(revive_empty)
        self.split_epsilon = float(split_epsilon)
        self.compute_dtype = compute_dtype
        self.backend = backend
        self.seed = int(seed)
        self.verbose = bool(verbose)

        self.centroids: torch.Tensor | None = None
        self.assignments: torch.Tensor | None = None
        self.iteration_stats: list[LloydIterStat] = []
        self.n_splits: int = 0
        self.init_time_sec: float = 0.0
        self.train_time_sec: float = 0.0
        self.total_time_sec: float = 0.0

    def _resolve_compute_dtype(self, x: torch.Tensor) -> torch.dtype:
        if self.compute_dtype is not None:
            return self.compute_dtype
        if x.is_cuda and x.dtype in (torch.bfloat16, torch.float16):
            return x.dtype
        if x.is_cuda:
            return torch.bfloat16
        return torch.float32

    @torch.no_grad()
    def fit(
        self,
        x: torch.Tensor | np.ndarray,
        init_centroids: torch.Tensor | np.ndarray | None = None,
        capture_assignments: bool = True,
    ) -> tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray | None]:
        input_is_numpy = isinstance(x, np.ndarray)
        x_t = (
            torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
            if input_is_numpy
            else x.detach()
        )
        n, d = x_t.shape
        if d != self.d:
            raise ValueError(f"Expected dimension d={self.d}, got {d}")
        if n < self.k:
            raise ValueError(f"Number of points n={n} must be >= k={self.k}")

        device = x_t.device
        comp_dtype = self._resolve_compute_dtype(x_t)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_init_start = time.perf_counter()

        if init_centroids is not None:
            c_in = (
                torch.from_numpy(np.asarray(init_centroids, dtype=np.float32))
                if isinstance(init_centroids, np.ndarray)
                else init_centroids
            )
            centroids = c_in.to(device=device, dtype=torch.float32).clone().contiguous()
        elif self.init == "random":
            gen = torch.Generator(device=device)
            gen.manual_seed(self.seed)
            perm = torch.randperm(n, generator=gen, device=device)[: self.k]
            centroids = x_t[perm].float().clone().contiguous()
        else:
            centroids = greedy_kmeans_plusplus_torch(x_t, self.k, seed=self.seed)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self.init_time_sec = time.perf_counter() - t_init_start

        t_train_start = time.perf_counter()
        sign = torch.ones((self.d,), dtype=torch.float32, device=device)
        sign[1::2] = -1.0
        split_offset = self.split_epsilon * sign
        split_gen = torch.Generator(device=device)
        split_gen.manual_seed((self.seed + 104729) % (2**32))

        sums = torch.empty((self.k, self.d), dtype=torch.float32, device=device)
        row_chunk = 131072
        labels: torch.Tensor | None = None
        prev_mse = float("inf")
        self.iteration_stats = []
        self.n_splits = 0

        for it in range(self.niter):
            c_rep, c_norm = prepare_represented_centroids(centroids, comp_dtype)
            sq_dists, labels = assign_points(
                x_t,
                centroids,
                compute_dtype=comp_dtype,
                backend=self.backend,
                row_chunk_size=row_chunk,
                c_rep_cached=(c_rep, c_norm),
            )
            it_mse = float(sq_dists.double().sum().item()) / n

            counts = torch.bincount(labels, minlength=self.k).float()
            sums.zero_()
            for s in range(0, n, row_chunk):
                e = min(s + row_chunk, n)
                sums.index_add_(0, labels[s:e], x_t[s:e].float())

            nonempty = counts > 0
            centroids = torch.where(
                nonempty[:, None],
                sums / counts.clamp_min(1.0)[:, None],
                centroids,
            )

            it_splits = 0
            if self.revive_empty:
                empty_mask = ~nonempty
                donor_mask = counts > 1.0
                n_empty = int(empty_mask.sum().item())
                n_donors = int(donor_mask.sum().item())
                nsplit = min(n_empty, n_donors)
                if nsplit > 0:
                    u = torch.rand(
                        (self.k,),
                        generator=split_gen,
                        device=device,
                        dtype=torch.float32,
                    ).clamp_min_(torch.finfo(torch.float32).tiny)
                    keys = torch.where(
                        donor_mask,
                        counts.log() - (-u.log()).log(),
                        torch.full_like(counts, float("-inf")),
                    )
                    source = torch.topk(keys, nsplit).indices
                    target = torch.nonzero(empty_mask, as_tuple=False).squeeze(1)[:nsplit]
                    donors = centroids[source]
                    d_cent = donors * (1.0 - split_offset)
                    c_cent = donors * (1.0 + split_offset)
                    _separate_collapsed_splits(d_cent, c_cent, comp_dtype)
                    centroids[source] = d_cent
                    centroids[target] = c_cent
                    it_splits = nsplit
                    self.n_splits += nsplit

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            stat = LloydIterStat(
                iteration=it + 1,
                mse=it_mse,
                n_splits=it_splits,
                elapsed_sec=time.perf_counter() - t_train_start,
            )
            self.iteration_stats.append(stat)
            if self.verbose:
                print(
                    f"[Lloyd-{self.init}] iter {it + 1}/{self.niter}: "
                    f"mse={it_mse:.6f}, splits={it_splits}, time={stat.elapsed_sec:.3f}s"
                )

            if self.tol >= 0.0 and abs(prev_mse - it_mse) <= self.tol * max(1.0, abs(prev_mse)):
                break
            prev_mse = it_mse

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self.train_time_sec = time.perf_counter() - t_train_start
        self.total_time_sec = self.init_time_sec + self.train_time_sec

        self.centroids = centroids.contiguous()
        self.assignments = labels if capture_assignments else None

        if input_is_numpy:
            return (
                self.centroids.cpu().numpy(),
                self.assignments.cpu().numpy() if self.assignments is not None else None,
            )
        return self.centroids, self.assignments

    @torch.no_grad()
    def assign(
        self,
        x: torch.Tensor | np.ndarray,
    ) -> tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray]:
        if self.centroids is None:
            raise RuntimeError("Model has not been fitted yet.")
        input_is_numpy = isinstance(x, np.ndarray)
        x_t = (
            torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(self.centroids.device)
            if input_is_numpy
            else x.to(self.centroids.device)
        )
        comp_dtype = self._resolve_compute_dtype(x_t)
        dists, labels = assign_points(x_t, self.centroids, compute_dtype=comp_dtype, backend=self.backend)
        if input_is_numpy:
            return dists.cpu().numpy(), labels.cpu().numpy()
        return dists, labels


class SculleyMiniBatchKMeans:
    """Sculley (WWW 2010) Mini-Batch K-Means with harmonic 1/V_j updates.

    Supports both:
      - `engine='torch'` (default): GPU/CPU implementation using the exact same assignment
        kernel as e-means so wall-clock comparisons isolate algorithmic convergence vs hardware.
      - `engine='sklearn'`: Direct wrapper around `sklearn.cluster.MiniBatchKMeans` with
        early stopping disabled (`max_no_improvement=None, tol=0.0`), matching Appendix C.
    """

    def __init__(
        self,
        d: int,
        k: int,
        nepochs: float = 5.0,
        batch_size: int = 4096,
        init: str = "random",
        reassignment_ratio: float = 0.01,
        engine: str = "torch",
        compute_dtype: torch.dtype | None = None,
        backend: str = "auto",
        seed: int = 1234,
    ) -> None:
        if init not in ("random", "k-means++"):
            raise ValueError("init must be 'random' or 'k-means++'")
        if engine not in ("torch", "sklearn"):
            raise ValueError("engine must be 'torch' or 'sklearn'")

        self.d = int(d)
        self.k = int(k)
        self.nepochs = float(nepochs)
        self.batch_size = int(batch_size)
        self.init = init
        self.reassignment_ratio = float(reassignment_ratio)
        self.engine = engine
        self.compute_dtype = compute_dtype
        self.backend = backend
        self.seed = int(seed)

        self.centroids: torch.Tensor | None = None
        self.assignments: torch.Tensor | None = None
        self.init_time_sec: float = 0.0
        self.train_time_sec: float = 0.0
        self.total_time_sec: float = 0.0

    def _resolve_compute_dtype(self, x: torch.Tensor) -> torch.dtype:
        if self.compute_dtype is not None:
            return self.compute_dtype
        if x.is_cuda and x.dtype in (torch.bfloat16, torch.float16):
            return x.dtype
        if x.is_cuda:
            return torch.bfloat16
        return torch.float32

    @torch.no_grad()
    def fit(
        self,
        x: torch.Tensor | np.ndarray,
        capture_assignments: bool = False,
    ) -> tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray | None]:
        input_is_numpy = isinstance(x, np.ndarray)
        if self.engine == "sklearn":
            from sklearn.cluster import MiniBatchKMeans

            x_np = (
                np.ascontiguousarray(x, dtype=np.float32)
                if input_is_numpy
                else np.ascontiguousarray(x.detach().cpu().numpy(), dtype=np.float32)
            )
            n, d = x_np.shape
            if d != self.d:
                raise ValueError(f"Expected d={self.d}, got {d}")
            eff_bsz = max(1, min(self.batch_size, n))
            max_iter = max(1, int(round(self.nepochs)))
            t0 = time.perf_counter()
            mbk = MiniBatchKMeans(
                n_clusters=self.k,
                init=self.init,
                max_iter=max_iter,
                batch_size=eff_bsz,
                compute_labels=capture_assignments,
                random_state=self.seed,
                tol=0.0,
                max_no_improvement=None,
                reassignment_ratio=self.reassignment_ratio,
                n_init=1,
            )
            mbk.fit(x_np)
            elapsed = time.perf_counter() - t0
            self.init_time_sec = 0.0
            self.train_time_sec = elapsed
            self.total_time_sec = elapsed
            c_out = np.ascontiguousarray(mbk.cluster_centers_, dtype=np.float32)
            self.centroids = torch.from_numpy(c_out)
            self.assignments = (
                torch.from_numpy(mbk.labels_.astype(np.int64))
                if capture_assignments and hasattr(mbk, "labels_")
                else None
            )
            if input_is_numpy:
                return c_out, (self.assignments.numpy() if self.assignments is not None else None)
            dev = x.device
            return (
                self.centroids.to(dev),
                self.assignments.to(dev) if self.assignments is not None else None,
            )

        # Native GPU/CPU Sculley (WWW 2010) algorithm
        x_t = (
            torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
            if input_is_numpy
            else x.detach()
        )
        n, d = x_t.shape
        if d != self.d:
            raise ValueError(f"Expected d={self.d}, got {d}")
        if n < self.k:
            raise ValueError(f"n={n} must be >= k={self.k}")

        device = x_t.device
        comp_dtype = self._resolve_compute_dtype(x_t)
        eff_bsz = max(1, min(self.batch_size, n))

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_init = time.perf_counter()

        if self.init == "random":
            gen_init = torch.Generator(device=device)
            gen_init.manual_seed(self.seed)
            perm = torch.randperm(n, generator=gen_init, device=device)[: self.k]
            centroids = x_t[perm].float().clone().contiguous()
        else:
            centroids = greedy_kmeans_plusplus_torch(x_t, self.k, seed=self.seed)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self.init_time_sec = time.perf_counter() - t_init

        t_train = time.perf_counter()
        # Sculley cumulative per-center counts v_j (initialized to 0; first touch sets mean)
        cum_counts = torch.zeros((self.k,), dtype=torch.float32, device=device)
        batch_sums = torch.empty((self.k, self.d), dtype=torch.float32, device=device)
        ones_scalar = torch.ones((1,), dtype=torch.float32, device=device)

        total_rows = max(eff_bsz, int(round(self.nepochs * n)))
        n_full = total_rows // n
        rem = total_rows - n_full * n
        ep_rows = [n] * n_full + ([rem] if rem > 0 else [])

        gen_re = torch.Generator(device=device)
        gen_re.manual_seed((self.seed + 31337) % (2**32))

        step = 0
        for ep_idx, rows_in_ep in enumerate(ep_rows):
            gen_ep = torch.Generator(device=device)
            gen_ep.manual_seed(epoch_seed(self.seed, ep_idx))
            order = torch.randperm(n, generator=gen_ep, device=device)[:rows_in_ep]

            for b_start in range(0, rows_in_ep, eff_bsz):
                b_end = min(b_start + eff_bsz, rows_in_ep)
                x_batch = x_t[order[b_start:b_end]]
                bsz = x_batch.shape[0]

                c_rep, c_norm = prepare_represented_centroids(centroids, comp_dtype)
                _, labels = assign_points(
                    x_batch,
                    centroids,
                    compute_dtype=comp_dtype,
                    backend=self.backend,
                    c_rep_cached=(c_rep, c_norm),
                )
                counts = torch.bincount(labels, minlength=self.k).float()
                batch_sums.zero_()
                batch_sums.index_add_(0, labels, x_batch.float())

                # Exact batched Sculley (2010) harmonic cumulative-count update:
                # v_new = v_old + n_j;  c_new = (v_old * c_old + s_j) / v_new
                touched = counts > 0
                v_new = cum_counts + counts
                safe_v = torch.where(touched, v_new, ones_scalar)
                c_updated = (cum_counts[:, None] * centroids + batch_sums) / safe_v[:, None]
                centroids = torch.where(touched[:, None], c_updated, centroids)
                cum_counts = v_new

                # Sculley / sklearn reassignment of starving centers from current batch
                if self.reassignment_ratio > 0.0 and ((step + 1) % 10 == 0) and bsz > 0:
                    cutoff = self.reassignment_ratio * cum_counts.max()
                    low_mask = cum_counts < cutoff
                    n_low = min(int(low_mask.sum().item()), bsz)
                    if n_low > 0:
                        targets = torch.nonzero(low_mask, as_tuple=False).squeeze(1)[:n_low]
                        pick = torch.randperm(bsz, generator=gen_re, device=device)[:n_low]
                        centroids[targets] = x_batch[pick].float()
                        # Reset reassigned cumulative count to avoid infinite inertia
                        cum_counts[targets] = torch.clamp(cum_counts.median(), min=1.0)

                step += 1

        if capture_assignments:
            _, assignments = assign_points(
                x_t, centroids, compute_dtype=comp_dtype, backend=self.backend
            )
        else:
            assignments = None

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self.train_time_sec = time.perf_counter() - t_train
        self.total_time_sec = self.init_time_sec + self.train_time_sec

        self.centroids = centroids.contiguous()
        self.assignments = assignments

        if input_is_numpy:
            return (
                self.centroids.cpu().numpy(),
                self.assignments.cpu().numpy() if self.assignments is not None else None,
            )
        return self.centroids, self.assignments


class NaiveKMeansPP_EMeans:
    """Naive composition of full-dataset Greedy k-means++ with vanilla e-means.

    Reproduces the Table `init_scale` baseline of arXiv:2611.09501 (Appendix K.1):
      - Full-dataset sequential Greedy k-means++ seeding (O(n * k * d * L)),
      - No Voronoi polish (raw sampled data shell centroids),
      - Vanilla tiny initial EMA mass N_0 = 2^-16 (causing step-0 weight ~ 1.0 erasure),
      - Vanilla count-only revival N_j < 0.1 * mean(N) (assassinating rare clusters).
    """

    def __init__(
        self,
        d: int,
        k: int,
        nepochs: float = 5.0,
        batch_law: str = "auto",
        batch_size: int | None = None,
        alpha0: float | None = None,
        finalization: str = "closing_mean",
        compute_dtype: torch.dtype | None = None,
        backend: str = "auto",
        seed: int = 1234,
        verbose: bool = False,
    ) -> None:
        self.d = int(d)
        self.k = int(k)
        self.nepochs = float(nepochs)
        self.batch_law = batch_law
        self.batch_size = batch_size
        self.alpha0 = alpha0
        self.finalization = finalization
        self.compute_dtype = compute_dtype
        self.backend = backend
        self.seed = int(seed)
        self.verbose = bool(verbose)

        self.emeans = EMeans(
            d=self.d,
            k=self.k,
            nepochs=self.nepochs,
            batch_law=self.batch_law,
            batch_size=self.batch_size,
            alpha0=self.alpha0,
            init_usage=DEFAULT_INIT_USAGE,
            finalization=self.finalization,
            revival_mode="split",
            compute_dtype=self.compute_dtype,
            backend=self.backend,
            seed=self.seed,
            verbose=self.verbose,
        )
        self.centroids: torch.Tensor | None = None
        self.assignments: torch.Tensor | None = None
        self.init_time_sec: float = 0.0
        self.train_time_sec: float = 0.0
        self.total_time_sec: float = 0.0
        self.n_splits: int = 0

    @torch.no_grad()
    def fit(
        self,
        x: torch.Tensor | np.ndarray,
        capture_assignments: bool = True,
    ) -> tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray | None]:
        input_is_numpy = isinstance(x, np.ndarray)
        x_t = (
            torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
            if input_is_numpy
            else x.detach()
        )
        device = x_t.device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        raw_kpp = greedy_kmeans_plusplus_torch(x_t, self.k, seed=self.seed)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        kpp_time = time.perf_counter() - t0

        c_out, a_out = self.emeans.fit(
            x_t, init_centroids=raw_kpp, capture_assignments=capture_assignments
        )
        self.init_time_sec = kpp_time + self.emeans.init_time_sec
        self.train_time_sec = self.emeans.train_time_sec
        self.total_time_sec = self.init_time_sec + self.train_time_sec
        self.n_splits = self.emeans.n_splits
        self.centroids = self.emeans.centroids
        self.assignments = self.emeans.assignments

        if input_is_numpy:
            return (
                c_out.cpu().numpy(),
                a_out.cpu().numpy() if a_out is not None else None,
            )
        return c_out, a_out
