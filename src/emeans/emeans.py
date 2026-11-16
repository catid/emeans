"""From-scratch reproduction of the e-means clustering algorithm.

Implements Algorithm 1, Section 3.1-3.3, and Appendix J of:
  Bessa et al., "e-means: outperforming k-means (in one-tenth the epochs)",
  arXiv:2611.09501 (2026).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from .kernels import assign_points, prepare_represented_centroids
from .recipe import (
    ALPHA_MIN_FACTOR,
    CHECK_UNUSED_EVERY,
    DEFAULT_INIT_USAGE,
    SPLIT_EPSILON,
    EMeansRecipe,
    build_recipe,
    epoch_seed,
    scheduled_alpha,
)


def _separate_collapsed_splits(
    donor: torch.Tensor,
    child: torch.Tensor,
    compute_dtype: torch.dtype,
) -> None:
    """Ensure donor and child differ in represented precision even if donor == 0."""
    rep_dtype = compute_dtype if compute_dtype in (torch.float32, torch.bfloat16, torch.float16) else torch.float32
    dq = donor.to(rep_dtype)
    cq = child.to(rep_dtype)
    collapsed = (dq == cq).all(dim=1)
    val = cq[:, 0]
    up = torch.nextafter(val, torch.full_like(val, float("inf")))
    down = torch.nextafter(val, torch.full_like(val, float("-inf")))
    neighbor = torch.where(torch.isfinite(up), up, down).float()
    child[:, 0] = torch.where(collapsed, neighbor, child[:, 0])


@dataclass
class EpochStat:
    epoch: int
    rows: int
    steps: int
    moving_mse: float
    imbalance_factor: float
    n_splits: int
    elapsed_sec: float


class EMeans:
    """Faithful PyTorch/Triton & CPU reproduction of Meta FAIR's e-means algorithm."""

    def __init__(
        self,
        d: int,
        k: int,
        nepochs: float = 5.0,
        batch_law: str = "auto",
        batch_size: int | None = None,
        alpha0: float | None = None,
        alpha_min_factor: float = ALPHA_MIN_FACTOR,
        unused_threshold: float | None = None,
        check_unused_every: int = CHECK_UNUSED_EVERY,
        split_epsilon: float = SPLIT_EPSILON,
        init_usage: float = DEFAULT_INIT_USAGE,
        finalization: str = "closing_mean",
        slow_ema_mult: float = 0.25,
        revival_mode: str = "split",
        use_cosine_schedule: bool = True,
        compute_dtype: torch.dtype | None = None,
        backend: str = "auto",
        seed: int = 1234,
        verbose: bool = False,
    ) -> None:
        if finalization not in ("closing_mean", "slow_ema", "none"):
            raise ValueError("finalization must be 'closing_mean', 'slow_ema', or 'none'")
        if revival_mode not in ("split", "resample", "off"):
            raise ValueError("revival_mode must be 'split', 'resample', or 'off'")

        self.d = int(d)
        self.k = int(k)
        self.nepochs = float(nepochs)
        self.batch_law = batch_law
        self.batch_size = batch_size
        self.alpha0 = alpha0
        self.alpha_min_factor = float(alpha_min_factor)
        self.unused_threshold = unused_threshold
        self.check_unused_every = int(check_unused_every)
        self.split_epsilon = float(split_epsilon)
        self.init_usage = float(init_usage)
        self.finalization = finalization
        self.slow_ema_mult = float(slow_ema_mult)
        self.revival_mode = revival_mode
        self.use_cosine_schedule = bool(use_cosine_schedule)
        self.compute_dtype = compute_dtype
        self.backend = backend
        self.seed = int(seed)
        self.verbose = bool(verbose)

        self.recipe: EMeansRecipe | None = None
        self.centroids: torch.Tensor | None = None
        self.slow_centroids: torch.Tensor | None = None
        self.assignments: torch.Tensor | None = None
        self.iteration_stats: list[EpochStat] = []
        self.n_splits: int = 0
        self.train_time_sec: float = 0.0
        self.init_time_sec: float = 0.0
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
        """Run e-means clustering on dataset x of shape (N, D)."""
        input_is_numpy = isinstance(x, np.ndarray)
        x_t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)) if input_is_numpy else x.detach()
        n, d = x_t.shape
        if d != self.d:
            raise ValueError(f"Expected dimension d={self.d}, got {d}")

        device = x_t.device
        comp_dtype = self._resolve_compute_dtype(x_t)

        eff_threshold = 0.0 if self.revival_mode == "off" else self.unused_threshold
        recipe = build_recipe(
            n=n,
            d=d,
            k=self.k,
            nepochs=self.nepochs,
            batch_law=self.batch_law,
            batch_size=self.batch_size,
            alpha0=self.alpha0,
            alpha_min_factor=self.alpha_min_factor,
            unused_threshold=eff_threshold,
            check_unused_every=self.check_unused_every,
            split_epsilon=self.split_epsilon,
            init_usage=self.init_usage,
        )
        self.recipe = recipe

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_init_start = time.perf_counter()

        if init_centroids is None:
            gen_init = torch.Generator(device=device)
            gen_init.manual_seed(self.seed)
            perm = torch.randperm(n, generator=gen_init, device=device)[: self.k]
            centroids = x_t[perm].float().clone().contiguous()
        else:
            c_in = torch.from_numpy(np.asarray(init_centroids, dtype=np.float32)) if isinstance(init_centroids, np.ndarray) else init_centroids
            centroids = c_in.to(device=device, dtype=torch.float32).clone().contiguous()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self.init_time_sec = time.perf_counter() - t_init_start

        t_train_start = time.perf_counter()
        usage = torch.full((self.k,), recipe.init_usage, dtype=torch.float32, device=device)

        # Optional slow EMA state (Section 3.2 & Appendix H)
        track_slow = self.finalization == "slow_ema"
        if track_slow:
            slow_centroids = centroids.clone()
            slow_usage = usage.clone()
            alpha_slow = min(1.0, max(1e-6, self.slow_ema_mult * recipe.alpha_min_factor * recipe.alpha0))
        else:
            slow_centroids = None
            slow_usage = None
            alpha_slow = 0.0

        # Precompute checkerboard sign vector [+1, -1, +1, -1, ...]
        sign = torch.ones((self.d,), dtype=torch.float32, device=device)
        sign[1::2] = -1.0
        split_offset = recipe.split_epsilon * sign

        split_gen = torch.Generator(device=device)
        split_gen.manual_seed((self.seed + 7919) % (2**32))

        assignments = (
            torch.full((n,), -1, dtype=torch.int64, device=device)
            if capture_assignments
            else None
        )

        # Workspace buffers for batch accumulation
        ones_buf = torch.ones((recipe.batch_size,), dtype=torch.float32, device=device)
        batch_sums = torch.empty((self.k, self.d), dtype=torch.float32, device=device)

        step = 0
        self.n_splits = 0
        self.iteration_stats = []
        num_epochs = len(recipe.epoch_rows)

        for ep_idx, rows_in_ep in enumerate(recipe.epoch_rows):
            is_last_epoch = ep_idx == (num_epochs - 1)
            gen_ep = torch.Generator(device=device)
            gen_ep.manual_seed(epoch_seed(self.seed, ep_idx))
            order = torch.randperm(n, generator=gen_ep, device=device)[:rows_in_ep]

            pass_counts = torch.zeros((self.k,), dtype=torch.float64, device=device)
            pass_sums = (
                torch.zeros((self.k, self.d), dtype=torch.float32, device=device)
                if (is_last_epoch and self.finalization == "closing_mean")
                else None
            )
            ep_sq_dist_t = torch.zeros((), dtype=torch.float64, device=device)
            splits_before = self.n_splits
            ep_steps = 0

            for b_start in range(0, rows_in_ep, recipe.batch_size):
                b_end = min(b_start + recipe.batch_size, rows_in_ep)
                batch_idx = order[b_start:b_end]
                x_batch = x_t[batch_idx]
                bsz = x_batch.shape[0]

                c_rep, c_norm = prepare_represented_centroids(centroids, comp_dtype)
                sq_dists, labels = assign_points(
                    x_batch,
                    centroids,
                    compute_dtype=comp_dtype,
                    backend=self.backend,
                    c_rep_cached=(c_rep, c_norm),
                )

                if is_last_epoch and assignments is not None:
                    assignments[batch_idx] = labels

                ep_sq_dist_t += sq_dists.double().sum()

                # Batch counts n_j and sums s_j in FP32
                counts = torch.bincount(labels, minlength=self.k).float()
                batch_sums.zero_()
                batch_sums.index_add_(0, labels, x_batch.float())

                pass_counts += counts.double()
                if pass_sums is not None:
                    pass_sums += batch_sums

                # Step size alpha(t)
                if self.use_cosine_schedule:
                    alpha = scheduled_alpha(
                        recipe.alpha0,
                        step,
                        recipe.total_steps,
                        recipe.alpha_min_factor,
                    )
                else:
                    alpha = recipe.alpha0
                decay = 1.0 - alpha

                # EMA update (Equation 1 & 3 in usage/position form)
                touched = counts > 0
                if alpha < 1.0:
                    usage.mul_(decay)
                    retained = usage
                else:
                    retained = torch.zeros_like(usage)

                new_mass = retained + alpha * counts
                safe_counts = torch.where(touched, counts, ones_buf[:1])
                batch_means = batch_sums / safe_counts[:, None]
                weight = (alpha * counts) / torch.where(touched, new_mass, ones_buf[:1])
                updated_pos = centroids + weight[:, None] * (batch_means - centroids)
                updated_pos = torch.where((weight == 1.0)[:, None], batch_means, updated_pos)

                centroids = torch.where(touched[:, None], updated_pos, centroids)
                usage = torch.where(touched, new_mass, usage)

                if track_slow and slow_centroids is not None and slow_usage is not None:
                    if alpha_slow < 1.0:
                        slow_usage.mul_(1.0 - alpha_slow)
                        s_retained = slow_usage
                    else:
                        s_retained = torch.zeros_like(slow_usage)
                    s_new_mass = s_retained + alpha_slow * counts
                    s_weight = (alpha_slow * counts) / torch.where(touched, s_new_mass, ones_buf[:1])
                    s_pos = slow_centroids + s_weight[:, None] * (batch_means - slow_centroids)
                    s_pos = torch.where((s_weight == 1.0)[:, None], batch_means, s_pos)
                    slow_centroids = torch.where(touched[:, None], s_pos, slow_centroids)
                    slow_usage = torch.where(touched, s_new_mass, slow_usage)

                # Dead-cluster revival every `check_unused_every` updates
                if (
                    recipe.unused_threshold > 0.0
                    and self.revival_mode != "off"
                    and ((step + 1) % recipe.check_unused_every == 0)
                ):
                    thresh = usage.mean() * recipe.unused_threshold
                    unused_mask = usage < thresh
                    donor_mask = (~unused_mask) & (usage > 0)
                    n_unused = int(unused_mask.sum().item())
                    n_donors = int(donor_mask.sum().item())

                    if self.revival_mode == "split":
                        nsplit = min(n_unused, n_donors)
                        if nsplit > 0:
                            u = torch.rand(
                                (self.k,),
                                generator=split_gen,
                                device=device,
                                dtype=torch.float32,
                            ).clamp_min_(torch.finfo(torch.float32).tiny)
                            keys = torch.where(
                                donor_mask,
                                usage.log() - (-u.log()).log(),
                                torch.full_like(usage, float("-inf")),
                            )
                            source = torch.topk(keys, nsplit).indices
                            target = torch.nonzero(unused_mask, as_tuple=False).squeeze(1)[:nsplit]

                            donors = centroids[source]
                            d_cent = donors * (1.0 - split_offset)
                            c_cent = donors * (1.0 + split_offset)
                            _separate_collapsed_splits(d_cent, c_cent, comp_dtype)
                            half_u = usage[source] * 0.5

                            centroids[source] = d_cent
                            centroids[target] = c_cent
                            usage[source] = half_u
                            usage[target] = half_u

                            if track_slow and slow_centroids is not None and slow_usage is not None:
                                s_donors = slow_centroids[source]
                                sd_cent = s_donors * (1.0 - split_offset)
                                sc_cent = s_donors * (1.0 + split_offset)
                                _separate_collapsed_splits(sd_cent, sc_cent, comp_dtype)
                                shalf_u = slow_usage[source] * 0.5
                                slow_centroids[source] = sd_cent
                                slow_centroids[target] = sc_cent
                                slow_usage[source] = shalf_u
                                slow_usage[target] = shalf_u

                            self.n_splits += nsplit
                    elif self.revival_mode == "resample" and n_unused > 0 and bsz > 0:
                        nsplit = min(n_unused, bsz)
                        target = torch.nonzero(unused_mask, as_tuple=False).squeeze(1)[:nsplit]
                        r_idx = torch.randperm(bsz, generator=split_gen, device=device)[:nsplit]
                        centroids[target] = x_batch[r_idx].float()
                        usage[target] = usage.mean()
                        self.n_splits += nsplit

                step += 1
                ep_steps += 1

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            ep_sq_dist = float(ep_sq_dist_t.item())
            imbalance = float(
                (self.k * torch.dot(pass_counts, pass_counts) / (float(rows_in_ep) ** 2)).item()
            )
            stat = EpochStat(
                epoch=ep_idx + 1,
                rows=rows_in_ep,
                steps=ep_steps,
                moving_mse=ep_sq_dist / max(1, rows_in_ep),
                imbalance_factor=imbalance,
                n_splits=self.n_splits - splits_before,
                elapsed_sec=time.perf_counter() - t_train_start,
            )
            self.iteration_stats.append(stat)
            if self.verbose:
                print(
                    f"[e-means] pass {ep_idx + 1}/{num_epochs} ({rows_in_ep} rows, {ep_steps} steps): "
                    f"moving_mse={stat.moving_mse:.6f}, splits={stat.n_splits}, time={stat.elapsed_sec:.3f}s"
                )

            # Finalization at the end of the final pass
            if is_last_epoch:
                if self.finalization == "closing_mean" and pass_sums is not None:
                    nonempty = pass_counts > 0
                    centroids = torch.where(
                        nonempty[:, None],
                        pass_sums / pass_counts.clamp_min(1.0).float()[:, None],
                        centroids,
                    )
                elif self.finalization == "slow_ema" and slow_centroids is not None:
                    centroids = slow_centroids

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self.train_time_sec = time.perf_counter() - t_train_start
        self.total_time_sec = self.init_time_sec + self.train_time_sec

        self.centroids = centroids.contiguous()
        self.slow_centroids = slow_centroids
        self.assignments = assignments

        if input_is_numpy:
            return (
                self.centroids.cpu().numpy(),
                self.assignments.cpu().numpy() if self.assignments is not None else None,
            )
        return self.centroids, self.assignments

    @torch.no_grad()
    def assign(self, x: torch.Tensor | np.ndarray) -> tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray]:
        """Assign points x to the learned centroids, returning (sq_distances, labels)."""
        if self.centroids is None:
            raise RuntimeError("Model has not been fitted yet.")
        input_is_numpy = isinstance(x, np.ndarray)
        x_t = (
            torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(self.centroids.device)
            if input_is_numpy
            else x.to(self.centroids.device)
        )
        comp_dtype = self._resolve_compute_dtype(x_t)
        dists, labels = assign_points(
            x_t,
            self.centroids,
            compute_dtype=comp_dtype,
            backend=self.backend,
        )
        if input_is_numpy:
            return dists.cpu().numpy(), labels.cpu().numpy()
        return dists, labels
