"""Fast GPU/CPU k-means++ initialization, lightweight coresets, and Voronoi pre-conditioning.

Combines:
1. Lightweight Coresets (Bachem, Lucic, Krause, KDD 2018): O(n*d) importance sketch
   that preserves rare/far-away outlier clusters with unbiased importance weights.
2. Sync-free GPU Weighted Greedy k-means++ (Arthur & Vassilvitskii 2007; Celebi et al. 2013):
   evaluates L = 2 + floor(ln k) candidates per step on device with zero CPU-GPU syncs.
3. Geometric Block-Deduplicated k-means++ (`block_greedy_kmeans_plusplus`):
   reaches large k (e.g., k = 16,384+) in ~20 batched GEMM rounds by de-duplicating
   oversampled D^2 candidates locally per round.
4. 1-Step Weighted Voronoi Polish: replaces raw sampled data points (which sit on the
   sqrt(d)*sigma shell at 2x intra-cluster MSE) with weighted coreset Voronoi means
   and returns calibrated cluster priors (p_hat, e_hat) for EMA warm-starting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .kernels import HAS_TRITON, assign_points

if HAS_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _local_greedy_kpp_kernel(
        pair_d2_ptr,
        init_d2_ptr,
        weights_ptr,
        out_idx_ptr,
        P,
        num_select,
        rng_seed,
        BLOCK_P: tl.constexpr,
        L_TRIALS: tl.constexpr,
    ):
        offs_p = tl.arange(0, BLOCK_P)
        mask_p = offs_p < P
        pool_d2 = tl.load(init_d2_ptr + offs_p, mask=mask_p, other=0.0).to(tl.float32)
        w = tl.load(weights_ptr + offs_p, mask=mask_p, other=0.0).to(tl.float32)
        log_w = tl.log(tl.maximum(w, 1e-30))
        avail = mask_p

        for s_idx in range(num_select):
            proposal = tl.where(avail, w * pool_d2, 0.0)
            max_prop = tl.max(proposal, axis=0)
            has_pos = max_prop > 0.0
            log_prop = tl.log(tl.maximum(proposal, 1e-38))

            best_pot = float("inf")
            best_pick = tl.zeros((), dtype=tl.int32)
            best_d2 = pool_d2

            for l in tl.static_range(L_TRIALS):
                if l == L_TRIALS - 1:
                    far_score = tl.where(avail, pool_d2, -1.0)
                    cand = tl.argmax(far_score, axis=0)
                else:
                    rng_offs = (s_idx * L_TRIALS + l) * BLOCK_P + offs_p
                    u = tl.rand(rng_seed, rng_offs)
                    u = tl.minimum(tl.maximum(u, 1e-30), 1.0 - 1e-7)
                    gumbel = -tl.log(-tl.log(u))
                    key_pos = tl.where(avail & (proposal > 0.0), log_prop + gumbel, -float("inf"))
                    key_fb = tl.where(avail, log_w + gumbel, -float("inf"))
                    key = tl.where(has_pos, key_pos, key_fb)
                    cand = tl.argmax(key, axis=0)

                row_d2 = tl.load(
                    pair_d2_ptr + cand.to(tl.int64) * P + offs_p.to(tl.int64),
                    mask=mask_p,
                    other=0.0,
                ).to(tl.float32)
                cand_new_d2 = tl.minimum(pool_d2, row_d2)
                pot = tl.sum(tl.where(mask_p, w * cand_new_d2, 0.0), axis=0)

                better = (l == 0) | (pot < best_pot)
                best_pot = tl.where(better, pot, best_pot)
                best_pick = tl.where(better, cand, best_pick)
                best_d2 = tl.where(better, cand_new_d2, best_d2)

            pool_d2 = tl.where(offs_p == best_pick, 0.0, best_d2)
            avail = avail & (offs_p != best_pick)
            tl.store(out_idx_ptr + s_idx, best_pick.to(tl.int64))


@dataclass
class SeedingResult:
    """Output of k-means++ / coreset initialization."""

    centroids: torch.Tensor        # (K, D) float32 initial centroids
    prior_probs: torch.Tensor      # (K,) float32 empirical probability mass summing to 1
    prior_distortion: torch.Tensor # (K,) float32 empirical mean distortion per cluster


@torch.no_grad()
def lightweight_coreset(
    x: torch.Tensor,
    m: int,
    seed: int = 1234,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct an O(n*d) lightweight coreset (X_sub, weights) of size m <= n.

    Proposal distribution (Bachem et al., KDD 2018):
      q(x_i) = 0.5 / n + 0.5 * ||x_i - mu||^2 / sum_l ||x_l - mu||^2
      w_i    = 1 / (m * q(x_i)), normalized to sum to 1.
    """
    n = x.shape[0]
    if n < 1 or m < 1:
        raise ValueError(f"lightweight_coreset requires n >= 1 and m >= 1, got n={n}, m={m}")
    device = x.device
    if m >= n:
        w = torch.full((n,), 1.0 / n, dtype=torch.float32, device=device)
        return x.float().contiguous(), w

    # Compute mean and squared distance to mean in chunks to keep memory tiny
    chunk = 262144
    mean_acc = torch.zeros((x.shape[1],), dtype=torch.float64, device=device)
    for s in range(0, n, chunk):
        mean_acc += x[s : min(s + chunk, n)].double().sum(dim=0)
    mu = (mean_acc / n).float()

    d2 = torch.empty((n,), dtype=torch.float32, device=device)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        diff = x[s:e].float() - mu[None, :]
        d2[s:e] = diff.square().sum(dim=1)

    raw_total_d2 = d2.double().sum()
    uniform_q = torch.full((n,), 1.0 / n, dtype=torch.float64, device=device)
    q = torch.where(
        raw_total_d2 > 0.0,
        0.5 * (1.0 / n) + 0.5 * (d2.double() / raw_total_d2.clamp_min(1e-30)),
        uniform_q,
    )
    q = q / q.sum().clamp_min(1e-30)

    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed) % (2**32))

    # Draw m samples via inverse-CDF searchsorted (right=True skips zero-mass atoms; zero syncs on CUDA)
    cdf = torch.cumsum(q, dim=0)
    u = torch.rand((m,), generator=gen, device=device, dtype=torch.float64) * cdf[-1]
    idx = torch.searchsorted(cdf, u, right=True).clamp_(0, n - 1)

    x_sub = x[idx].float().contiguous()
    w_sub = (1.0 / (m * q[idx])).float()
    w_sub = w_sub / w_sub.sum().clamp_min(1e-30)
    return x_sub, w_sub


@torch.no_grad()
def greedy_kmeans_plusplus_torch(
    x: torch.Tensor,
    k: int,
    weights: torch.Tensor | None = None,
    n_local_trials: int | None = None,
    include_farthest_candidate: bool = False,
    seed: int = 1234,
) -> torch.Tensor:
    """Sync-free GPU/CPU Weighted Greedy k-means++ seeding.

    Runs all k steps without a single `.item()` or host-device synchronization.
    When `include_farthest_candidate=True` (used in e-means++), replaces 1 of the L
    local trial slots with `argmax(min_d2)` while still arbitrating strictly via the
    global weighted k-means potential `sum_i w_i min(d2_i, cand_d2_i)`: single-point
    noise outliers are rejected by potential minimization, while genuine low-mass
    separated rare clusters are reliably discovered despite high-D bulk noise.
    """
    m, d = x.shape
    device = x.device
    xf = x.float().contiguous()
    if k < 1:
        raise ValueError(f"k={k} must be >= 1")
    if m <= k:
        if m == k:
            return xf.clone()
        raise ValueError(f"Cannot pick k={k} centers from m={m} points")

    w = (
        torch.full((m,), 1.0 / m, dtype=torch.float32, device=device)
        if weights is None
        else weights.float().contiguous()
    )
    w = w / w.sum().clamp_min(1e-30)
    L = int(2 + math.floor(math.log(max(2, k)))) if n_local_trials is None else int(n_local_trials)

    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed) % (2**32))

    # Center coordinates for translation-invariant distance evaluation without cancellation
    origin = xf.mean(dim=0, keepdim=True)
    xc = (xf - origin).contiguous()

    # Use BF16 on CUDA only for large m > 65536 where Tensor Cores pay off; keep exact FP32 on coresets
    use_bf16 = device.type == "cuda" and d >= 32 and m > 65536
    x_rep = xc.to(torch.bfloat16) if use_bf16 else xc
    x_norm = x_rep.float().square().sum(dim=1)

    centroids = torch.empty((k, d), dtype=torch.float32, device=device)

    # First center sampled proportional to weights
    w_double = w.double()
    w_cdf = torch.cumsum(w_double, dim=0)
    u0 = torch.rand((1,), generator=gen, device=device, dtype=torch.float64) * w_cdf[-1]
    idx0 = torch.searchsorted(w_cdf, u0, right=True).clamp_(0, m - 1)
    centroids[0] = xf[idx0].squeeze(0)

    c0_c = xc[idx0]
    c0_rep = c0_c.to(x_rep.dtype)
    c0_norm = c0_rep.float().square().sum(dim=1)
    dots0 = (x_rep @ c0_rep.t()).float().squeeze(1)
    min_d2 = (x_norm + c0_norm - 2.0 * dots0).clamp_min_(0.0)
    min_d2[idx0] = 0.0

    # Pre-generate all random uniforms on device at once to avoid per-step RNG overhead
    all_u = torch.rand((k - 1, L), generator=gen, device=device, dtype=torch.float64)
    w_col = w[:, None]
    use_far = bool(include_farthest_candidate and L >= 2)

    for t in range(1, k):
        weighted_d2 = (w * min_d2).double()
        total_wd2 = weighted_d2.sum()
        # Branchless fallback to uniform weights if total_wd2 <= 0 (all remaining points identical to chosen centers)
        eff_w = torch.where(total_wd2 > 0.0, weighted_d2, w_double)
        cdf = torch.cumsum(eff_w, dim=0)
        u_step = all_u[t - 1] * cdf[-1]
        cand_idx = torch.searchsorted(cdf, u_step, right=True).clamp_(0, m - 1)  # (L,)
        if use_far:
            cand_idx[-1] = min_d2.argmax()

        c_cand_raw = xf[cand_idx]  # (L, d)
        c_cand_c = xc[cand_idx]
        c_cand_rep = c_cand_c.to(x_rep.dtype)
        c_cand_norm = c_cand_rep.float().square().sum(dim=1)  # (L,)

        if use_bf16:
            scores = torch.addmm(
                c_cand_norm[None, :],
                x_rep,
                c_cand_rep.t(),
                alpha=-2.0,
                out_dtype=torch.float32,
            )
        else:
            scores = torch.addmm(c_cand_norm[None, :], x_rep, c_cand_rep.t(), alpha=-2.0)

        cand_d2 = (scores + x_norm[:, None]).clamp_min_(0.0)  # (m, L)
        new_d2 = torch.minimum(min_d2[:, None], cand_d2)      # (m, L)
        potentials = (w_col * new_d2).sum(dim=0)              # (L,)
        best_l = potentials.argmin()                          # () scalar tensor on device

        centroids[t] = c_cand_raw[best_l]
        min_d2 = new_d2[:, best_l]
        min_d2[cand_idx[best_l]] = 0.0

    return centroids


@torch.no_grad()
def _select_from_candidate_pool(
    c_pool: torch.Tensor,
    init_pool_d2: torch.Tensor,
    pool_weights: torch.Tensor,
    num_select: int,
    seed: int,
) -> torch.Tensor:
    """Select `num_select` centers from candidate pool `(c_pool, pool_weights)` via weighted Greedy k-means++."""
    P = c_pool.shape[0]
    device = c_pool.device
    if P <= num_select:
        return c_pool[:num_select]

    c_pool_c = c_pool - c_pool.mean(dim=0, keepdim=True)
    pool_norm = c_pool_c.square().sum(dim=1)
    pair_d2 = (
        pool_norm[:, None] + pool_norm[None, :] - 2.0 * (c_pool_c @ c_pool_c.t())
    ).clamp_min_(0.0).contiguous()
    pair_d2.fill_diagonal_(0.0)

    w_p = pool_weights.float()
    w_p = (w_p / w_p.sum().clamp_min(1e-30)).contiguous()
    pool_d2 = init_pool_d2.float().clone().contiguous()

    if HAS_TRITON and device.type == "cuda" and P <= 1024:
        # Bucket to 2 static power-of-two specializations so seed-dependent P never triggers PTXAS JIT mid-benchmark
        block_p = 256 if P <= 256 else 1024
        selected_local = torch.empty((num_select,), dtype=torch.int64, device=device)
        num_warps = 4 if block_p <= 256 else 8
        _local_greedy_kpp_kernel[(1,)](
            pair_d2,
            pool_d2,
            w_p,
            selected_local,
            P,
            num_select,
            int(seed) % (2**31 - 1),
            BLOCK_P=block_p,
            L_TRIALS=3,
            num_warps=num_warps,
        )
        return c_pool[selected_local]

    # CPU / ultra-wide fallback with identical weighted-potential math
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed) % (2**32))
    avail = torch.ones((P,), dtype=torch.bool, device=device)
    selected_local = torch.empty((num_select,), dtype=torch.int64, device=device)
    u_local = torch.rand((num_select, 4), generator=gen, device=device, dtype=torch.float64)
    w_col = w_p[:, None]

    for s_idx in range(num_select):
        prop_d = torch.where(avail, (w_p * pool_d2).double(), 0.0)
        ltot = prop_d.sum()
        eff_local = torch.where(ltot > 0.0, prop_d, torch.where(avail, w_p.double(), 0.0))
        lcdf = torch.cumsum(eff_local, dim=0)
        lu = u_local[s_idx] * lcdf[-1]
        trials_rnd = torch.searchsorted(lcdf, lu, right=True).clamp_(0, P - 1)
        far_pick = torch.where(avail, pool_d2, -1.0).argmax().unsqueeze(0)
        trials = torch.cat([trials_rnd, far_pick], dim=0)  # (5,)
        trial_d2 = torch.minimum(pool_d2[:, None], pair_d2[:, trials])
        best_t = (w_col * trial_d2).sum(dim=0).argmin()
        pick = trials[best_t]
        selected_local[s_idx] = pick
        pool_d2 = trial_d2[:, best_t]
        pool_d2[pick] = 0.0
        avail[pick] = False

    return c_pool[selected_local]


@torch.no_grad()
def block_greedy_kmeans_plusplus_torch(
    x: torch.Tensor,
    k: int,
    weights: torch.Tensor | None = None,
    oversample: int = 2,
    growth: float = 2.0,
    max_block: int = 384,
    seed: int = 1234,
) -> torch.Tensor:
    """Sub-epoch Geometric Block Greedy k-means++ with Register-Resident Triton Reduction.

    Grows centers geometrically (`16 -> 32 -> 64 -> 128 -> 256 -> 384`) in ~6-15 batched
    rounds. Each round oversamples `P <= 1024` D^2 candidates, weights them by residual
    coreset mass, and selects `b_r` centers inside a single register-resident Triton kernel
    (`BLOCK_P <= 1024`, zero register spilling, zero CPU-GPU syncs).
    """
    m, d = x.shape
    device = x.device
    xf = x.float().contiguous()
    if k < 1:
        raise ValueError(f"k={k} must be >= 1")
    if m <= k:
        if m == k:
            return xf.clone()
        raise ValueError(f"Cannot pick k={k} centers from m={m} points")
    if k <= 16:
        return greedy_kmeans_plusplus_torch(
            xf, k, weights=weights, include_farthest_candidate=True, seed=seed
        )

    w = (
        torch.full((m,), 1.0 / m, dtype=torch.float32, device=device)
        if weights is None
        else weights.float().contiguous()
    )
    w = w / w.sum().clamp_min(1e-30)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed) % (2**32))

    # Center coordinates so BF16 distance evaluation is 100% invariant to global coordinate shifts
    origin = xf.mean(dim=0, keepdim=True)
    xc = (xf - origin).contiguous()
    comp_dtype = torch.bfloat16 if (device.type == "cuda" and d >= 32 and m > 16384) else torch.float32

    b0 = min(16, k)
    centroids = torch.empty((k, d), dtype=torch.float32, device=device)
    c_init = greedy_kmeans_plusplus_torch(
        xf, b0, weights=w, include_farthest_candidate=True, seed=seed
    )
    centroids[:b0] = c_init
    placed = b0
    min_d2, _ = assign_points(xc, (centroids[:placed] - origin).contiguous(), compute_dtype=comp_dtype)

    cur_block = max(32, int(round(b0 * growth)))
    round_idx = 0

    while placed < k:
        b_r = min(k - placed, min(max_block, cur_block))
        cand_pool_size = min(m, min(960, max(b_r + 32, oversample * b_r + 32)))

        weighted_d2 = (w * min_d2).double()
        tot_wd2 = weighted_d2.sum()
        eff_wd2 = torch.where(tot_wd2 > 0.0, weighted_d2, w.double())
        cdf = torch.cumsum(eff_wd2, dim=0)
        u = torch.rand((cand_pool_size,), generator=gen, device=device, dtype=torch.float64) * cdf[-1]
        cand_idx = torch.searchsorted(cdf, u, right=True).clamp_(0, m - 1)
        top_far = torch.topk(min_d2, k=min(64, m)).indices
        cand_idx = torch.unique(torch.cat([cand_idx, top_far]))
        if cand_idx.numel() < b_r:
            extra = torch.randperm(m, generator=gen, device=device)[: min(m, b_r * 2 + 16)]
            cand_idx = torch.unique(torch.cat([cand_idx, extra]))
        if cand_idx.numel() > 1024:
            far_set = top_far[: min(64, top_far.numel())]
            cand_idx = torch.unique(torch.cat([far_set, cand_idx[:960]]))[:1024]

        c_pool = xf[cand_idx].contiguous()
        pool_d2 = min_d2[cand_idx].clone()
        # Exact Bahmani et al. (2012) Voronoi mass on candidate pool (~0.2ms fused Triton call)
        _, pool_lbls = assign_points(xc, (c_pool - origin).contiguous(), compute_dtype=comp_dtype)
        w_pool = torch.zeros((c_pool.shape[0],), dtype=torch.float32, device=device)
        w_pool.index_add_(0, pool_lbls, w)
        w_pool.clamp_min_(1e-12 / max(1, c_pool.shape[0]))
        chosen = _select_from_candidate_pool(
            c_pool, pool_d2, w_pool, b_r, seed=seed + 1000 + round_idx
        )

        actual_b = chosen.shape[0]
        centroids[placed : placed + actual_b] = chosen
        new_d2, _ = assign_points(xc, (chosen - origin).contiguous(), compute_dtype=comp_dtype)
        min_d2 = torch.minimum(min_d2, new_d2)

        placed += actual_b
        cur_block = int(round(cur_block * growth))
        round_idx += 1

    return centroids


@torch.no_grad()
def voronoi_polish_and_priors(
    x_seed: torch.Tensor,
    w_seed: torch.Tensor,
    init_centroids: torch.Tensor,
    compute_dtype: torch.dtype | None = None,
) -> SeedingResult:
    """Perform 1 weighted Voronoi step on the seeding coreset and extract priors.

    Moves centroids from the sqrt(d)*sigma data shell to weighted Voronoi means,
    eliminating the 2x shell-variance penalty before Epoch 1 begins, while
    extracting empirical cluster probabilities p_hat and distortions e_hat.
    """
    k, d = init_centroids.shape
    device = x_seed.device
    origin = x_seed.float().mean(dim=0, keepdim=True)
    _, labels = assign_points(
        (x_seed.float() - origin).contiguous(),
        (init_centroids.float() - origin).contiguous(),
        compute_dtype=compute_dtype,
    )

    w = w_seed.float()
    w = w / w.sum().clamp_min(1e-30)

    mass = torch.zeros((k,), dtype=torch.float32, device=device)
    mass.index_add_(0, labels, w)

    w_x = x_seed.float() * w[:, None]
    sums = torch.zeros((k, d), dtype=torch.float32, device=device)
    sums.index_add_(0, labels, w_x)

    nonempty = mass > 0
    polished = torch.where(
        nonempty[:, None],
        sums / mass.clamp_min(1e-30)[:, None],
        init_centroids.float(),
    )

    # Re-evaluate squared distance to polished centroids along assigned labels (O(m*d), no GEMM!)
    diff = x_seed.float() - polished[labels]
    polished_sq = diff.square().sum(dim=1)
    dist_mass = torch.zeros((k,), dtype=torch.float32, device=device)
    dist_mass.index_add_(0, labels, w * polished_sq)

    # Normalize prior_probs to sum to 1
    prior_probs = mass / mass.sum().clamp_min(1e-30)
    return SeedingResult(
        centroids=polished.contiguous(),
        prior_probs=prior_probs.contiguous(),
        prior_distortion=dist_mass.contiguous(),
    )


@torch.no_grad()
def seed_emeans_plusplus(
    x: torch.Tensor,
    k: int,
    seeding_mode: str = "auto",
    coreset_mult: int = 16,
    min_coreset_size: int = 8192,
    polish: bool = True,
    seed: int = 1234,
) -> SeedingResult:
    """Complete sub-epoch seeding pipeline for e-means++.

    Modes:
      - 'auto': Lightweight coreset (m = min(n, max(min_coreset_size, coreset_mult * k)))
                + exact sync-free Greedy k-means++ for k <= 2048, or Geometric Block
                Greedy k-means++ for k > 2048, followed by 1-step Voronoi polish.
      - 'coreset_exact': Lightweight coreset + exact k-step Weighted Greedy k-means++.
      - 'coreset_block': Lightweight coreset + Geometric Block-Deduplicated k-means++.
      - 'full_greedy': Full-dataset Greedy k-means++ (slow reference baseline).
      - 'random': Uniform random subset without replacement on lightweight coreset (for ablation).
    """
    n, d = x.shape
    device = x.device
    if n < k:
        raise ValueError(f"Number of points n={n} must be >= k={k}")

    if seeding_mode == "full_greedy":
        x_seed = x.float()
        w_seed = torch.full((n,), 1.0 / n, dtype=torch.float32, device=device)
        raw_c = greedy_kmeans_plusplus_torch(x_seed, k, weights=w_seed, seed=seed)
    elif seeding_mode == "random":
        m = min(n, max(min_coreset_size, coreset_mult * k, k * 2))
        x_seed, w_seed = lightweight_coreset(x, m=m, seed=seed)
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed) % (2**32))
        perm = torch.randperm(n, generator=gen, device=device)[:k]
        raw_c = x[perm].float().contiguous()
    else:
        m = min(n, max(min_coreset_size, coreset_mult * k, k * 2))
        x_seed, w_seed = lightweight_coreset(x, m=m, seed=seed)
        if seeding_mode == "coreset_exact" or (seeding_mode == "auto" and k <= 256):
            raw_c = greedy_kmeans_plusplus_torch(
                x_seed, k, weights=w_seed, include_farthest_candidate=True, seed=seed + 17
            )
        elif seeding_mode == "coreset_block" or (seeding_mode == "auto" and k > 256):
            raw_c = block_greedy_kmeans_plusplus_torch(x_seed, k, weights=w_seed, seed=seed + 17)
        else:
            raise ValueError(f"Unknown seeding_mode={seeding_mode!r}")

    w_norm = w_seed.float()
    w_norm = w_norm / w_norm.sum().clamp_min(1e-30)
    comp_dtype = torch.bfloat16 if (device.type == "cuda" and d >= 32 and x_seed.shape[0] > 65536) else torch.float32

    if polish:
        return voronoi_polish_and_priors(x_seed, w_norm, raw_c, compute_dtype=comp_dtype)

    # Without Voronoi polish, still estimate empirical priors on the normalized coreset
    origin = x_seed.float().mean(dim=0, keepdim=True)
    sq_d, labels = assign_points(
        (x_seed.float() - origin).contiguous(),
        (raw_c.float() - origin).contiguous(),
        compute_dtype=comp_dtype,
    )
    mass = torch.zeros((k,), dtype=torch.float32, device=device)
    mass.index_add_(0, labels, w_norm)
    dist_mass = torch.zeros((k,), dtype=torch.float32, device=device)
    dist_mass.index_add_(0, labels, w_norm * sq_d)
    total_m = mass.sum().clamp_min(1e-30)
    return SeedingResult(
        centroids=raw_c.contiguous(),
        prior_probs=(mass / total_m).contiguous(),
        prior_distortion=(dist_mass / total_m).contiguous(),
    )
