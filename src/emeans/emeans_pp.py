"""e-means++: Principled synthesis of sub-epoch k-means++ seeding with EMA mini-batch clustering.

Overcomes BOTH failure modes of naive `k-means++ + e-means` (Table `init_scale` of arXiv:2611.09501):
  1. Sub-epoch seeding latency:
     - O(n*d) Lightweight Coreset (Bachem et al., KDD 2018) with unbiased importance weights,
     - Sync-free GPU Weighted Greedy k-means++ / Geometric Block-Deduplicated k-means++,
     - 1-Step Weighted Coreset Voronoi Polish eliminating the 2x high-D spherical-shell variance
       before Epoch 1 and extracting calibrated cluster mass (p_hat) and distortion (e_hat) priors.
  2. Warm-Started EMA Dynamics & Huygens D^2-Guided Revival:
     - Prior-calibrated initial EMA mass N_j^(0) = prior_mass_scale * B * max(p_hat_j, floor / k)
       so step 0 interpolates smoothly with weight ~ alpha_0 rather than 1.0 (preventing step-0
       erasure of Voronoi-polished k-means++ centroids),
     - Tracked per-cluster EMA distortion E_j <- (1 - alpha) E_j + alpha * sum_{x in B_j} ||x - c_j||^2,
     - Huygens Parallel-Axis Rare-Mode Protection: protects genuine rare/outlier clusters
       (0 < p_j < tau / k) whose Huygens removal SSE cost M_j * min_{m != j} ||c_j - c_m||^2
       exceeds gamma_protect * mean(E), preventing vanilla count-only revival assassination,
     - D^2 Donor Splitting: samples split donors without replacement proportional to EMA distortion
       E_{j'} rather than raw count N_{j'}, targeting high-SSE merged/under-resolved clusters.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from .emeans import _separate_collapsed_splits
from .kernels import assign_points, prepare_represented_centroids
from .recipe import (
    ALPHA_MIN_FACTOR,
    CHECK_UNUSED_EVERY,
    SPLIT_EPSILON,
    EMeansRecipe,
    build_recipe,
    epoch_seed,
    scheduled_alpha,
)
from .seeding import (
    SeedingResult,
    lightweight_coreset,
    seed_emeans_plusplus,
    voronoi_polish_and_priors,
)


@torch.no_grad()
def _nearest_other_centroid_sq_dists(
    centroids: torch.Tensor,
    cand_ids: torch.Tensor,
    chunk_size: int = 512,
) -> torch.Tensor:
    """Compute exact min_{m != j} ||c_j - c_m||^2 for candidate indices cand_ids.

    Strictly masks self-pairs m == j to +inf and centers coordinates to prevent
    FP32 catastrophic cancellation on offset centroids.
    """
    k = centroids.shape[0]
    num_cand = cand_ids.numel()
    device = centroids.device
    if num_cand == 0:
        return torch.empty((0,), dtype=torch.float32, device=device)
    if k <= 1:
        return torch.full((num_cand,), float("inf"), dtype=torch.float32, device=device)

    origin = centroids.float().mean(dim=0, keepdim=True)
    kc = (centroids.float() - origin).contiguous()
    kc_norm = kc.square().sum(dim=1)  # (k,)

    out = torch.empty((num_cand,), dtype=torch.float32, device=device)
    for start in range(0, num_cand, chunk_size):
        end = min(start + chunk_size, num_cand)
        sub_ids = cand_ids[start:end]
        qc = kc[sub_ids]  # (C_sub, d)
        qc_norm = kc_norm[sub_ids]  # (C_sub,)

        d2 = torch.addmm(kc_norm[None, :], qc, kc.t(), alpha=-2.0)
        d2.add_(qc_norm[:, None]).clamp_min_(0.0)
        # Strictly exclude self-distance m == j
        row_arange = torch.arange(end - start, device=device)
        d2[row_arange, sub_ids] = float("inf")
        out[start:end] = d2.min(dim=1).values
    return out


@dataclass
class EMeansPPEpochStat:
    epoch: int
    rows: int
    steps: int
    moving_mse: float
    imbalance_factor: float
    n_splits: int
    n_protected: int
    elapsed_sec: float


class EMeansPP:
    """Principled e-means++ clustering with modular toggles for every algorithmic pillar."""

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
        # --- e-means++ Pillars (each independently toggleable for ablation) ---
        seeding_mode: str = "auto",
        coreset_mult: int = 16,
        min_coreset_size: int = 8192,
        use_voronoi_polish: bool = True,
        use_prior_mass: bool = True,
        prior_mass_scale: float = 0.5,
        prior_floor: float = 1e-3,
        warm_alpha_scale: float = 0.85,
        use_huygens_protection: bool = True,
        gamma_protect: float = 0.5,
        grace_steps: int = 5,
        use_d2_donor_split: bool = True,
        # --- Standard e-means schedule / finalization options ---
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
        if prior_mass_scale <= 0.0:
            raise ValueError("prior_mass_scale must be > 0.0")
        if warm_alpha_scale <= 0.0:
            raise ValueError("warm_alpha_scale must be > 0.0")
        if gamma_protect < 0.0:
            raise ValueError("gamma_protect must be >= 0.0")

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

        self.seeding_mode = seeding_mode
        self.coreset_mult = int(coreset_mult)
        self.min_coreset_size = int(min_coreset_size)
        self.use_voronoi_polish = bool(use_voronoi_polish)
        self.use_prior_mass = bool(use_prior_mass)
        self.prior_mass_scale = float(prior_mass_scale)
        self.prior_floor = float(prior_floor)
        self.warm_alpha_scale = float(warm_alpha_scale)
        self.use_huygens_protection = bool(use_huygens_protection)
        self.gamma_protect = float(gamma_protect)
        self.grace_steps = int(grace_steps)
        self.use_d2_donor_split = bool(use_d2_donor_split)

        self.finalization = finalization
        self.slow_ema_mult = float(slow_ema_mult)
        self.revival_mode = revival_mode
        self.use_cosine_schedule = bool(use_cosine_schedule)
        self.compute_dtype = compute_dtype
        self.backend = backend
        self.seed = int(seed)
        self.verbose = bool(verbose)

        self.recipe: EMeansRecipe | None = None
        self.seeding_result: SeedingResult | None = None
        self.centroids: torch.Tensor | None = None
        self.slow_centroids: torch.Tensor | None = None
        self.assignments: torch.Tensor | None = None
        self.iteration_stats: list[EMeansPPEpochStat] = []
        self.n_splits: int = 0
        self.n_protected: int = 0
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
        """Run e-means++ clustering on dataset x of shape (N, D)."""
        input_is_numpy = isinstance(x, np.ndarray)
        x_t = (
            torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
            if input_is_numpy
            else x.detach()
        )
        n, d = x_t.shape
        if d != self.d:
            raise ValueError(f"Expected dimension d={self.d}, got {d}")

        device = x_t.device
        comp_dtype = self._resolve_compute_dtype(x_t)
        eff_threshold = 0.0 if self.revival_mode == "off" else self.unused_threshold

        base_recipe = build_recipe(
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
        )
        eff_alpha0 = (
            float(min(max(base_recipe.alpha0 * self.warm_alpha_scale, 1e-4), 1.0))
            if self.alpha0 is None
            else float(base_recipe.alpha0)
        )
        recipe = build_recipe(
            n=n,
            d=d,
            k=self.k,
            nepochs=self.nepochs,
            batch_law=self.batch_law,
            batch_size=base_recipe.batch_size,
            alpha0=eff_alpha0,
            alpha_min_factor=self.alpha_min_factor,
            unused_threshold=eff_threshold,
            check_unused_every=self.check_unused_every,
            split_epsilon=self.split_epsilon,
        )
        self.recipe = recipe

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_init_start = time.perf_counter()

        if init_centroids is None:
            seed_res = seed_emeans_plusplus(
                x_t,
                k=self.k,
                seeding_mode=self.seeding_mode,
                coreset_mult=self.coreset_mult,
                min_coreset_size=self.min_coreset_size,
                polish=self.use_voronoi_polish,
                seed=self.seed,
            )
        else:
            c_in = (
                torch.from_numpy(np.asarray(init_centroids, dtype=np.float32))
                if isinstance(init_centroids, np.ndarray)
                else init_centroids
            )
            raw_c = c_in.to(device=device, dtype=torch.float32).clone().contiguous()
            m_sub = min(n, max(self.min_coreset_size, self.coreset_mult * self.k, self.k * 2))
            x_sub, w_sub = lightweight_coreset(x_t, m=m_sub, seed=self.seed)
            if self.use_voronoi_polish:
                seed_res = voronoi_polish_and_priors(
                    x_sub, w_sub, raw_c, compute_dtype=comp_dtype
                )
            else:
                sq_d, lbls = assign_points(x_sub, raw_c, compute_dtype=comp_dtype)
                mass = torch.zeros((self.k,), dtype=torch.float32, device=device)
                mass.index_add_(0, lbls, w_sub)
                dist_m = torch.zeros((self.k,), dtype=torch.float32, device=device)
                dist_m.index_add_(0, lbls, w_sub * sq_d)
                tot_m = mass.sum().clamp_min(1e-30)
                seed_res = SeedingResult(
                    centroids=raw_c,
                    prior_probs=(mass / tot_m).contiguous(),
                    prior_distortion=(dist_m / tot_m).contiguous(),
                )

        self.seeding_result = seed_res
        centroids = seed_res.centroids.to(device=device, dtype=torch.float32).clone().contiguous()
        p_hat = seed_res.prior_probs.to(device=device, dtype=torch.float32)
        e_hat = seed_res.prior_distortion.to(device=device, dtype=torch.float32)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self.init_time_sec = time.perf_counter() - t_init_start

        t_train_start = time.perf_counter()
        bsz_float = float(recipe.batch_size)
        min_prob = max(0.0, self.prior_floor) / float(self.k)
        mean_ehat = e_hat.mean().clamp_min(1e-12)

        if self.use_prior_mass:
            usage = (self.prior_mass_scale * bsz_float * p_hat.clamp_min(min_prob)).float().contiguous()
            distortion = (
                self.prior_mass_scale * bsz_float * e_hat.clamp_min(1e-6 * mean_ehat)
            ).float().contiguous()
        else:
            usage = torch.full((self.k,), recipe.init_usage, dtype=torch.float32, device=device)
            distortion = (bsz_float * e_hat.clamp_min(1e-6 * mean_ehat)).float().contiguous()

        # Epoch-horizon mass tracker prevents mid-epoch Poisson-gap false alarms on rare clusters
        survival_mass = (bsz_float * p_hat.clamp_min(min_prob)).float().contiguous()

        track_slow = self.finalization == "slow_ema"
        if track_slow:
            slow_centroids = centroids.clone()
            slow_usage = usage.clone()
            alpha_slow = min(
                1.0,
                max(1e-6, self.slow_ema_mult * recipe.alpha_min_factor * recipe.alpha0),
            )
        else:
            slow_centroids = None
            slow_usage = None
            alpha_slow = 0.0

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

        ones_scalar = torch.ones((1,), dtype=torch.float32, device=device)
        batch_sums = torch.empty((self.k, self.d), dtype=torch.float32, device=device)
        batch_sse = torch.empty((self.k,), dtype=torch.float32, device=device)

        step = 0
        self.n_splits = 0
        self.n_protected = 0
        self.iteration_stats = []
        num_epochs = len(recipe.epoch_rows)
        eff_grace = (
            min(self.grace_steps, max(0, recipe.total_steps // 3))
            if self.use_huygens_protection
            else 0
        )

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
            prot_before = self.n_protected
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

                counts = torch.bincount(labels, minlength=self.k).float()
                batch_sums.zero_()
                batch_sums.index_add_(0, labels, x_batch.float())
                batch_sse.zero_()
                batch_sse.index_add_(0, labels, sq_dists.float())

                pass_counts += counts.double()
                if pass_sums is not None:
                    pass_sums += batch_sums

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

                touched = counts > 0
                if alpha < 1.0:
                    usage.mul_(decay)
                    retained = usage
                else:
                    retained = torch.zeros_like(usage)

                new_mass = retained + alpha * counts
                safe_counts = torch.where(touched, counts, ones_scalar)
                batch_means = batch_sums / safe_counts[:, None]
                weight = (alpha * counts) / torch.where(touched, new_mass, ones_scalar)
                updated_pos = centroids + weight[:, None] * (batch_means - centroids)
                updated_pos = torch.where((weight == 1.0)[:, None], batch_means, updated_pos)

                centroids = torch.where(touched[:, None], updated_pos, centroids)
                usage = torch.where(touched, new_mass, usage)

                if alpha < 1.0:
                    distortion.mul_(decay).add_(batch_sse, alpha=alpha)
                else:
                    distortion = torch.where(touched, batch_sse, distortion * 0.5)

                alpha_surv = min(alpha, 2.0 * float(bsz) / float(max(1, n)))
                survival_mass.mul_(1.0 - alpha_surv).add_(counts, alpha=alpha_surv)

                if track_slow and slow_centroids is not None and slow_usage is not None:
                    if alpha_slow < 1.0:
                        slow_usage.mul_(1.0 - alpha_slow)
                        s_retained = slow_usage
                    else:
                        s_retained = torch.zeros_like(slow_usage)
                    s_new_mass = s_retained + alpha_slow * counts
                    s_weight = (alpha_slow * counts) / torch.where(touched, s_new_mass, ones_scalar)
                    s_pos = slow_centroids + s_weight[:, None] * (batch_means - slow_centroids)
                    s_pos = torch.where((s_weight == 1.0)[:, None], batch_means, s_pos)
                    slow_centroids = torch.where(touched[:, None], s_pos, slow_centroids)
                    slow_usage = torch.where(touched, s_new_mass, slow_usage)

                # Huygens D^2-guided dead-cluster revival
                if (
                    recipe.unused_threshold > 0.0
                    and self.revival_mode != "off"
                    and ((step + 1) >= max(1, eff_grace))
                    and ((step + 1) % recipe.check_unused_every == 0)
                ):
                    thresh = usage.mean() * recipe.unused_threshold
                    low_mask = usage < thresh
                    n_low = int(low_mask.sum().item())

                    if n_low > 0:
                        if self.use_huygens_protection and self.k >= 2 and self.gamma_protect > 0.0:
                            cand_ids = torch.nonzero(low_mask, as_tuple=False).squeeze(1)
                            d_nn2 = _nearest_other_centroid_sq_dists(centroids, cand_ids)
                            # Cap survival mass at thresh so continuous-manifold clusters
                            # are never artificially shielded, while genuine isolated rare
                            # modes (0 < p_j < tau/k, d_nn^2 >> MSE) stay protected.
                            eff_mass = torch.maximum(
                                usage[cand_ids],
                                torch.minimum(survival_mass[cand_ids], thresh),
                            )
                            removal_cost = eff_mass * d_nn2
                            protect_cutoff = self.gamma_protect * distortion.mean().clamp_min(1e-30)
                            protected_local = removal_cost >= protect_cutoff
                            n_prot = int(protected_local.sum().item())
                            self.n_protected += n_prot

                            unused_mask = low_mask.clone()
                            if n_prot > 0:
                                unused_mask[cand_ids[protected_local]] = False
                        else:
                            unused_mask = low_mask

                        # Protected rare clusters must never be chosen as split donors either
                        donor_mask = (~low_mask) & (usage > 0)
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

                                if self.use_d2_donor_split:
                                    pos_dist_donors = donor_mask & (distortion > 1e-30)
                                    if int(pos_dist_donors.sum().item()) >= nsplit:
                                        donor_scores = distortion
                                        eff_donor_mask = pos_dist_donors
                                    else:
                                        donor_scores = usage
                                        eff_donor_mask = donor_mask
                                else:
                                    donor_scores = usage
                                    eff_donor_mask = donor_mask

                                keys = torch.where(
                                    eff_donor_mask,
                                    donor_scores.clamp_min(1e-30).log() - (-u.log()).log(),
                                    torch.full_like(usage, float("-inf")),
                                )
                                source = torch.topk(keys, nsplit).indices
                                target = torch.nonzero(unused_mask, as_tuple=False).squeeze(1)[:nsplit]

                                donors = centroids[source]
                                d_cent = donors * (1.0 - split_offset)
                                c_cent = donors * (1.0 + split_offset)
                                _separate_collapsed_splits(d_cent, c_cent, comp_dtype)

                                half_u = usage[source] * 0.5
                                half_e = distortion[source] * 0.5
                                half_m = survival_mass[source] * 0.5

                                centroids[source] = d_cent
                                centroids[target] = c_cent
                                usage[source] = half_u
                                usage[target] = half_u
                                distortion[source] = half_e
                                distortion[target] = half_e
                                survival_mass[source] = half_m
                                survival_mass[target] = half_m

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
                            mean_u = usage.mean().clamp_min(recipe.init_usage)
                            mean_e = distortion.mean().clamp_min(1e-12)
                            usage[target] = mean_u
                            distortion[target] = mean_e
                            survival_mass[target] = mean_u
                            self.n_splits += nsplit

                step += 1
                ep_steps += 1

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            ep_sq_dist = float(ep_sq_dist_t.item())
            imbalance = float(
                (self.k * torch.dot(pass_counts, pass_counts) / (float(rows_in_ep) ** 2)).item()
            )
            stat = EMeansPPEpochStat(
                epoch=ep_idx + 1,
                rows=rows_in_ep,
                steps=ep_steps,
                moving_mse=ep_sq_dist / max(1, rows_in_ep),
                imbalance_factor=imbalance,
                n_splits=self.n_splits - splits_before,
                n_protected=self.n_protected - prot_before,
                elapsed_sec=time.perf_counter() - t_train_start,
            )
            self.iteration_stats.append(stat)
            if self.verbose:
                print(
                    f"[e-means++] pass {ep_idx + 1}/{num_epochs} ({rows_in_ep} rows, {ep_steps} steps): "
                    f"moving_mse={stat.moving_mse:.6f}, splits={stat.n_splits}, "
                    f"protected={stat.n_protected}, time={stat.elapsed_sec:.3f}s"
                )

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
    def assign(
        self,
        x: torch.Tensor | np.ndarray,
    ) -> tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray]:
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
