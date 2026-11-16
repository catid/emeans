"""Phase 2 Falsifiable Algorithmic Extensions Beyond e-means and e-means++.

Implements five mathematically grounded, modularly toggleable algorithmic hypotheses
extending Bessa et al. (arXiv:2611.09501, 2026) and `e-means++`:

- H-Algo-1 (Principal-Axis Donor Splitting — Su & Dy 2007; Oja 1982):
    Replaces fixed multiplicative checkerboard splitting `c_j * (1 +- 1e-4 * s)`
    (which underflows BF16 mantissa to a single-coordinate ULP nudge and cuts nearly
    orthogonally to high-D cluster axes) with an O(B*d) online Krasulina/Oja rank-1
    principal-eigenvector tracker splitting donors along `+- eta * sigma_1 * v_1(Sigma_j)`.
- H-Algo-2 (Annealed Hartigan Marginal Distortion Assignment — Hartigan & Wong 1979;
    Telgarsky & Vattani 2010):
    Replaces pure Euclidean Voronoi assignment with annealed finite-sample marginal SSE
    assignment `argmin_j (n_hat_j / (n_hat_j + beta(t))) * ||x - c_j||^2`, inducing
    Apollonius ("circlonoi") boundaries that escape unequal-size Voronoi local traps.
- H-Algo-3 (Nesterov Centroid Extrapolation / Coherence-Gated Lookahead EMA —
    Nesterov 1983; Polyak 1964):
    Assigns batches against extrapolated lookahead centroids `c_j + mu(t) * v_j` gated
    by directional drift coherence while keeping internal EMA state `(N_j, S_j)` strictly
    inside the data convex hull.
- H-Algo-4 (Scale-Invariant Top-2 Power-Mean Soft EMA Warmup — Xu & Lange 2019):
    Applies Majorization-Minimization power-mean responsibility sharing across each
    point's top-2 nearest centroids `w_2 / w_1 = ((d_1^2 + eps)/(d_2^2 + eps))^(1 - s(t))`
    during early warmup (`s(t) -> -inf`), crossing Voronoi saddle barriers with zero
    `B x K` matrix allocation and exact coordinate-scale invariance.
- H-Algo-5 (Empirical-Bayes / James-Stein Variance-Shrinkage Finalization —
    James & Stein 1961; Efron & Morris 1973):
    Combines `closing_mean` and `slow_ema` per cluster via precision-weighted Empirical-Bayes
    shrinkage `c_hat_j = lambda_j * mu_close_j + (1 - lambda_j) * mu_slow_j` with
    `lambda_j = n_j / (n_j + kappa_j)`, simultaneously retaining `closing_mean` train MSE
    on populous clusters and `slow_ema` out-of-sample test MSE on sparse clusters.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .emeans import EpochStat, _separate_collapsed_splits
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
from .seeding import (
    SeedingResult,
    lightweight_coreset,
    seed_emeans_plusplus,
    voronoi_polish_and_priors,
)


@dataclass(frozen=True)
class FalsifiableHypothesis:
    """Formal specification of a falsifiable algorithmic hypothesis."""

    id: str
    name: str
    mechanism: str
    equation: str
    predicted_win_regime: str
    predicted_fail_regime: str
    falsification_criterion: str


HYPOTHESES: dict[str, FalsifiableHypothesis] = {
    "H-Algo-1": FalsifiableHypothesis(
        id="H-Algo-1",
        name="Online Oja Principal-Axis Donor Splitting",
        mechanism=(
            "Track per-cluster top covariance eigenvector v_1(Sigma_j) and principal std "
            "sigma_{j,1} via O(B*d) batch Krasulina/Oja power iteration on residuals "
            "(x_i - c_{a(i)}), splitting donors additively along +- eta * sigma_{j,1} * v_1 "
            "rather than fixed multiplicative checkerboard [1 +- 1e-4]."
        ),
        equation=(
            "g_j = (1/n_j) sum_{x_i in B_j} <r_i, p_hat_j> r_i; "
            "c_donor^+- = c_j +- eta_split * sqrt(<g_j, p_hat_j>) * p_hat_j"
        ),
        predicted_win_regime=(
            "Clustered mixtures (GMM, Zipf GMM, Multi-Scale) and anisotropic stretched "
            "Gaussians in moderate-to-high dimension (d >= 64) when dead-cluster revivals "
            "occur (especially random-init e-means and imbalanced regimes), plus GPU BF16 "
            "runs where 1e-4 checkerboard perturbations collapse to coordinate-0 ULP nudges."
        ),
        predicted_fail_regime=(
            "Isotropic uniform/Gaussian clouds (all eigenvalues equal) or warm-started "
            "e-means++ runs on balanced well-separated clusters where zero splits trigger."
        ),
        falsification_criterion=(
            "Falsified if paired multi-seed train/test MSE delta against checkerboard "
            "splitting has Wilcoxon / paired-t p >= 0.05 or mean delta >= 0 across "
            "clustered and anisotropic benchmarks where splits > 0."
        ),
    ),
    "H-Algo-2": FalsifiableHypothesis(
        id="H-Algo-2",
        name="Annealed Hartigan Marginal Distortion Assignment",
        mechanism=(
            "Weight candidate squared distances by Hartigan-Wong's exact 1-point marginal "
            "SSE factor n_hat_j / (n_hat_j + beta(t)) using online usage-calibrated cluster "
            "counts n_hat_j = n * N_j / sum_m N_m, annealing beta(t) -> 0 before closing."
        ),
        equation=(
            "a(x) = argmin_j [ (n_hat_j / (n_hat_j + beta(t))) * ||x - c_j||_2^2 ], "
            "beta(t) = beta_0 * 0.5 * (1 + cos(pi * t / t_stop))"
        ),
        predicted_win_regime=(
            "Small points-per-cluster regimes (n/k in [16, 128]) and unequal cluster size "
            "distributions (Zipf GMM, Multi-Scale) where finite-sample centroid recoil "
            "1/(n_j + 1) creates circlonoi boundaries escaping Voronoi traps."
        ),
        predicted_fail_regime=(
            "Large n/k >= 500 (where n_j/(n_j + 1) -> 1 within <0.2%) or high-dimensional "
            "diffuse noise where pulling boundary points into sparse clusters inflates "
            "small-cluster estimation variance."
        ),
        falsification_criterion=(
            "Falsified if paired multi-seed train or test MSE in n/k in [16, 128] fails to "
            "improve significantly (p >= 0.05) or degrades out-of-sample test MSE."
        ),
    ),
    "H-Algo-3": FalsifiableHypothesis(
        id="H-Algo-3",
        name="Coherence-Gated Nesterov Centroid Lookahead EMA",
        mechanism=(
            "Assign points against Nesterov-extrapolated centroids c_j + mu(t) * v_j where "
            "velocity v_j tracks EMA centroid drift gated by cosine directional coherence "
            "max(0, cos(v_j^(t-1), Delta c_j^(t))), preserving exact convex-hull EMA state."
        ),
        equation=(
            "c_tilde_j^(t) = c_j^(t) + mu(t) * v_j^(t); "
            "v_j^(t) = cos_+(v_j^(t-1), Delta c_j^(t)) * (beta_v v_j^(t-1) + (1 - beta_v) Delta c_j^(t))"
        ),
        predicted_win_regime=(
            "Ultra-low epoch budgets (T in {1, 2}) with trailing EMA lag (random init or "
            "moderate step sizes alpha_0 < 0.6), where centroids undergo coherent ballistic drift."
        ),
        predicted_fail_regime=(
            "Warm-started Voronoi-polished e-means++ at T >= 5 (centroids start near local "
            "means at step 0, leaving little ballistic drift) or alpha_0 ~ 1 regimes where "
            "extrapolation overshoots."
        ),
        falsification_criterion=(
            "Falsified if paired multi-seed train/test MSE at T <= 2 shows no statistically "
            "significant improvement (p >= 0.05) or degrades MSE via overshoot."
        ),
    ),
    "H-Algo-4": FalsifiableHypothesis(
        id="H-Algo-4",
        name="Scale-Invariant Top-2 Power-Mean Soft EMA Warmup",
        mechanism=(
            "Share mini-batch EMA mass across each point's top-2 nearest centroids during "
            "the first rho_warm fraction of steps using Xu & Lange (2019) Power-k-Means MM "
            "weights w_2 / w_1 = ((d_1^2 + eps)/(d_2^2 + eps))^(1 - s(t)) as s(t) -> -inf."
        ),
        equation=(
            "r_i = ((d_{i,1}^2 + eps) / (d_{i,2}^2 + eps))^(1 - s(t)), "
            "w_{i,1} = 1 / (1 + r_i), w_{i,2} = r_i / (1 + r_i)"
        ),
        predicted_win_regime=(
            "Clustered / multimodal / multi-scale geometries with saddle barriers between "
            "modes where nearby centroids benefit from cross-boundary attraction early on."
        ),
        predicted_fail_regime=(
            "High-dimensional continuous or isotropic distributions where boundary sharing "
            "contracts neighboring centroids inward toward each other (soft-quantization bias)."
        ),
        falsification_criterion=(
            "Falsified if top-2 power-mean warmup fails to lower final hard-Voronoi MSE "
            "(p >= 0.05) or increases MSE due to inward centroid contraction."
        ),
    ),
    "H-Algo-5": FalsifiableHypothesis(
        id="H-Algo-5",
        name="Empirical-Bayes / James-Stein Variance-Shrinkage Finalization",
        mechanism=(
            "Interpolate per-cluster closing_mean and slow_ema via Empirical-Bayes precision "
            "weights lambda_j = n_j / (n_j + kappa_j), trusting exact closing_mean on "
            "populous clusters while shrinking noisy small-n_j clusters toward slow_ema."
        ),
        equation=(
            "c_hat_j = lambda_j * mu_close_j + (1 - lambda_j) * mu_slow_j, "
            "lambda_j = n_j / (n_j + kappa_j), kappa_j = ((d-2)/d) * sigma_hat_j^2 / tau_hat^2"
        ),
        predicted_win_regime=(
            "Moderate-to-small n/k (n/k in [16, 128]), high-dimensional embeddings, and "
            "Zipf/imbalanced clusterings where closing_mean suffers 1/n_j test-MSE variance "
            "on tail clusters while slow_ema suffers lag bias on large clusters."
        ),
        predicted_fail_regime=(
            "Very large n/k >= 1000 with uniform cluster sizes where n_j >> kappa_j for all "
            "clusters so lambda_j -> 1 and all finalizers coincide within <0.01%."
        ),
        falsification_criterion=(
            "Falsified if eb_shrinkage fails to simultaneously beat closing_mean on test MSE "
            "(p < 0.05) AND beat slow_ema on train MSE (p < 0.05) in n/k <= 128 regimes."
        ),
    ),
}


# ---------------------------------------------------------------------------
# H-Algo-1: Online Oja / Variance-Directional Principal-Axis Splitting
# ---------------------------------------------------------------------------


def separate_collapsed_directional(
    donor: torch.Tensor,
    child: torch.Tensor,
    direction: torch.Tensor,
    compute_dtype: torch.dtype,
) -> None:
    """Ensure `donor` and `child` differ in `compute_dtype` along their dominant split axis."""
    rep_dtype = (
        compute_dtype
        if compute_dtype in (torch.float32, torch.bfloat16, torch.float16)
        else torch.float32
    )
    dq = donor.to(rep_dtype)
    cq = child.to(rep_dtype)
    collapsed = (dq == cq).all(dim=1)
    if not bool(collapsed.any()):
        return

    dom_coord = direction.abs().argmax(dim=1)  # (M,)
    rows = torch.arange(donor.shape[0], device=donor.device)
    val = cq[rows, dom_coord]
    dir_sign = direction[rows, dom_coord]
    target_inf = torch.where(
        dir_sign >= 0.0,
        torch.full_like(val, float("inf")),
        torch.full_like(val, float("-inf")),
    )
    fallback_inf = -target_inf
    prim = torch.nextafter(val, target_inf)
    alt = torch.nextafter(val, fallback_inf)
    neighbor = torch.where(torch.isfinite(prim), prim, alt).float()
    cur_child_val = child[rows, dom_coord]
    child[rows, dom_coord] = torch.where(collapsed, neighbor, cur_child_val)


@torch.no_grad()
def update_oja_principal_axes(
    prin_vec: torch.Tensor,
    x_batch: torch.Tensor,
    centroids: torch.Tensor,
    labels: torch.Tensor,
    counts: torch.Tensor,
    ema_weight: torch.Tensor,
    g_buf: torch.Tensor | None = None,
) -> torch.Tensor:
    """Update per-cluster principal residual vector `prin_vec` (K, D) in O(B*D) time.

    Uses sign-consistent batch covariance power iteration (Krasulina / Oja):
      r_i = x_i - c_{a(i)}
      g_j = (1 / n_j) * sum_{x_i in B_j} <r_i, p_hat_j> * r_i
    Notice <g_j, p_hat_j> = (1 / n_j) * sum <r_i, p_hat_j>^2 >= 0, so `g_j` never
    undergoes arbitrary +-1 sign flips across mini-batches, and `||prin_vec[j]||_2`
    tracks the principal standard deviation `sqrt(lambda_1(Sigma_j))`.
    """
    k, d = centroids.shape
    if x_batch.shape[0] == 0:
        return prin_vec

    p_norm = prin_vec.norm(dim=1, keepdim=True).clamp_min(1e-12)
    p_hat = prin_vec / p_norm  # (K, D)

    r = x_batch.float() - centroids[labels]  # (B, D)
    proj = (r * p_hat[labels]).sum(dim=1)    # (B,)

    if g_buf is None or g_buf.shape != (k, d):
        g_sums = torch.zeros((k, d), dtype=torch.float32, device=centroids.device)
    else:
        g_sums = g_buf
        g_sums.zero_()
    g_sums.index_add_(0, labels, r * proj[:, None])

    touched = counts > 0
    safe_counts = counts.clamp_min(1.0)
    g_mean = g_sums / safe_counts[:, None]  # (K, D)
    g_norm = g_mean.norm(dim=1)             # (K,)
    rayleigh = (g_mean * p_hat).sum(dim=1).clamp_min(0.0)  # (K,) directional variance

    valid = touched & (g_norm > 1e-12)
    dir_new = g_mean / g_norm.clamp_min(1e-12)[:, None]
    target_vec = dir_new * rayleigh.sqrt()[:, None]

    w = ema_weight.clamp(0.0, 1.0)[:, None]
    updated = (1.0 - w) * prin_vec + w * target_vec
    return torch.where(valid[:, None], updated, prin_vec)


@torch.no_grad()
def update_coord_variance_ema(
    coord_var: torch.Tensor,
    x_batch: torch.Tensor,
    centroids: torch.Tensor,
    labels: torch.Tensor,
    counts: torch.Tensor,
    ema_weight: torch.Tensor,
    v_buf: torch.Tensor | None = None,
) -> torch.Tensor:
    """Update coordinate-wise variance EMA `coord_var` (K, D) for Su & Dy `Var-Part`."""
    k, d = centroids.shape
    if x_batch.shape[0] == 0:
        return coord_var

    r_sq = (x_batch.float() - centroids[labels]).square()
    if v_buf is None or v_buf.shape != (k, d):
        v_sums = torch.zeros((k, d), dtype=torch.float32, device=centroids.device)
    else:
        v_sums = v_buf
        v_sums.zero_()
    v_sums.index_add_(0, labels, r_sq)

    touched = counts > 0
    v_mean = v_sums / counts.clamp_min(1.0)[:, None]
    w = ema_weight.clamp(0.0, 1.0)[:, None]
    updated = (1.0 - w) * coord_var + w * v_mean
    return torch.where(touched[:, None], updated, coord_var)


@torch.no_grad()
def compute_directional_split(
    donors: torch.Tensor,
    source_idx: torch.Tensor,
    split_mode: str,
    split_epsilon: float,
    split_std_frac: float,
    compute_dtype: torch.dtype,
    prin_vec: torch.Tensor | None = None,
    coord_var: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    precomputed_dirs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute `(donor_children, target_children, unit_dirs)` for `source_idx` donors.

    Pass `precomputed_dirs` when applying the same split event to auxiliary `slow_centroids`
    so both centroid tables split along the exact same axis without advancing `generator`.
    """
    m, d = donors.shape
    device = donors.device

    if split_mode == "checkerboard":
        sign = torch.ones((d,), dtype=torch.float32, device=device)
        sign[1::2] = -1.0
        offset = split_epsilon * sign
        d_cent = donors * (1.0 - offset)
        c_cent = donors * (1.0 + offset)
        _separate_collapsed_splits(d_cent, c_cent, compute_dtype)
        unit_dirs = sign.unsqueeze(0).expand(m, d) / math.sqrt(max(1, d))
        return d_cent, c_cent, unit_dirs

    if split_mode == "oja":
        if prin_vec is None:
            raise ValueError("prin_vec is required when split_mode='oja'")
        pv = prin_vec[source_idx]  # (M, D)
        sigma_1 = pv.norm(dim=1, keepdim=True)  # (M, 1)
        if precomputed_dirs is not None:
            u_dir = precomputed_dirs
        else:
            e0 = torch.zeros((m, d), dtype=torch.float32, device=device)
            e0[:, 0] = 1.0
            u_dir = torch.where(sigma_1 > 1e-12, pv / sigma_1.clamp_min(1e-12), e0)
        fallback_scale = split_epsilon * donors.norm(dim=1, keepdim=True).clamp_min(1.0)
        mag = torch.where(sigma_1 > 1e-8, split_std_frac * sigma_1, fallback_scale)
        delta = mag * u_dir
        d_cent = donors - delta
        c_cent = donors + delta
        separate_collapsed_directional(d_cent, c_cent, u_dir, compute_dtype)
        return d_cent, c_cent, u_dir

    if split_mode == "var_axis":
        if coord_var is None:
            raise ValueError("coord_var is required when split_mode='var_axis'")
        cv = coord_var[source_idx].clamp_min(0.0)  # (M, D)
        max_var, max_idx = cv.max(dim=1)           # (M,)
        if precomputed_dirs is not None:
            u_dir = precomputed_dirs
        else:
            u_dir = torch.zeros((m, d), dtype=torch.float32, device=device)
            u_dir[torch.arange(m, device=device), max_idx] = 1.0
        sigma_c = max_var.sqrt().unsqueeze(1)
        fallback_scale = split_epsilon * donors.norm(dim=1, keepdim=True).clamp_min(1.0)
        mag = torch.where(sigma_c > 1e-8, split_std_frac * sigma_c, fallback_scale)
        delta = mag * u_dir
        d_cent = donors - delta
        c_cent = donors + delta
        separate_collapsed_directional(d_cent, c_cent, u_dir, compute_dtype)
        return d_cent, c_cent, u_dir

    if split_mode == "bf16_random":
        # Control arm: same local standard-deviation magnitude as Oja, but isotropic random direction!
        if precomputed_dirs is not None:
            u_dir = precomputed_dirs
        else:
            rnd = torch.randn((m, d), generator=generator, device=device, dtype=torch.float32)
            u_dir = rnd / rnd.norm(dim=1, keepdim=True).clamp_min(1e-12)
        if prin_vec is not None:
            sigma_1 = prin_vec[source_idx].norm(dim=1, keepdim=True)
        else:
            sigma_1 = torch.zeros((m, 1), dtype=torch.float32, device=device)
        fallback_scale = max(split_epsilon, 2.0**-6) * donors.norm(dim=1, keepdim=True).clamp_min(1.0)
        mag = torch.where(sigma_1 > 1e-8, split_std_frac * sigma_1, fallback_scale)
        delta = mag * u_dir
        d_cent = donors - delta
        c_cent = donors + delta
        separate_collapsed_directional(d_cent, c_cent, u_dir, compute_dtype)
        return d_cent, c_cent, u_dir

    raise ValueError(
        f"Unknown split_mode={split_mode!r}; expected 'checkerboard', 'oja', 'var_axis', or 'bf16_random'"
    )


@torch.no_grad()
def reset_post_split_oja_axes(
    prin_vec: torch.Tensor,
    source_idx: torch.Tensor,
    target_idx: torch.Tensor,
    used_unit_dirs: torch.Tensor,
    generator: torch.Generator | None = None,
) -> None:
    """Deflate split axis `used_unit_dirs` after splitting so repeat splits probe orthogonal axes."""
    m, d = used_unit_dirs.shape
    if m == 0:
        return
    old_sigma = prin_vec[source_idx].norm(dim=1, keepdim=True)
    if d <= 1:
        half_p = 0.5 * prin_vec[source_idx]
        prin_vec[source_idx] = half_p
        prin_vec[target_idx] = half_p
        return

    rnd = torch.randn((m, d), generator=generator, device=prin_vec.device, dtype=torch.float32)
    # Gram-Schmidt orthogonalization against `used_unit_dirs`
    proj = (rnd * used_unit_dirs).sum(dim=1, keepdim=True)
    orth = rnd - proj * used_unit_dirs
    orth_norm = orth.norm(dim=1, keepdim=True)
    # Fallback if random vector happened to be collinear
    alt = torch.roll(used_unit_dirs, shifts=1, dims=1)
    alt = alt - (alt * used_unit_dirs).sum(dim=1, keepdim=True) * used_unit_dirs
    orth = torch.where(orth_norm > 1e-8, orth, alt)
    orth_unit = orth / orth.norm(dim=1, keepdim=True).clamp_min(1e-12)
    reset_vec = (0.5 * old_sigma) * orth_unit
    prin_vec[source_idx] = reset_vec
    prin_vec[target_idx] = reset_vec.clone()


# ---------------------------------------------------------------------------
# H-Algo-2: Annealed Hartigan Size-Corrected Marginal Distortion Assignment
# ---------------------------------------------------------------------------


@torch.no_grad()
def hartigan_assign_points(
    x: torch.Tensor,
    centroids: torch.Tensor,
    usage: torch.Tensor,
    n_total: int,
    beta: float,
    compute_dtype: torch.dtype | None = None,
    backend: str = "auto",
    hartigan_n_min: float = 0.5,
    centroid_chunk_size: int = 4096,
    row_chunk_size: int = 16384,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign points by Hartigan marginal SSE `(n_hat_j / (n_hat_j + beta)) * ||x - c_j||^2`.

    Preserves exact translation invariance (`||x||^2` included BEFORE multiplying by
    `w_j = n_hat_j / (n_hat_j + beta)`) and represented-norm consistency (Appendix I.1),
    while returning UNWEIGHTED Euclidean squared distances `||x - c_{a(x)}||^2` in FP32.
    """
    if beta <= 0.0:
        return assign_points(x, centroids, compute_dtype=compute_dtype, backend=backend)

    if compute_dtype is None:
        if x.is_cuda and x.dtype in (torch.bfloat16, torch.float16):
            compute_dtype = x.dtype
        elif x.is_cuda:
            compute_dtype = torch.bfloat16
        else:
            compute_dtype = torch.float32

    N = x.shape[0]
    K = centroids.shape[0]
    device = x.device
    if N == 0:
        return (
            torch.empty((0,), dtype=torch.float32, device=device),
            torch.empty((0,), dtype=torch.int64, device=device),
        )

    c_rep, c_norm = prepare_represented_centroids(centroids, compute_dtype)
    total_u = usage.float().sum().clamp_min(1e-30)
    n_hat = (float(n_total) * (usage.float() / total_u)).clamp_min(float(hartigan_n_min))
    w_hart = n_hat / (n_hat + float(beta))  # (K,) in (0, 1)

    out_dists = torch.empty((N,), dtype=torch.float32, device=device)
    out_labels = torch.empty((N,), dtype=torch.int64, device=device)

    for r0 in range(0, N, row_chunk_size):
        r1 = min(r0 + row_chunk_size, N)
        xb = x[r0:r1]
        B = xb.shape[0]
        x_rep = xb.to(c_rep.dtype)
        x_norm = x_rep.float().square().sum(dim=1)  # (B,)

        best_weighted = torch.full((B,), float("inf"), dtype=torch.float32, device=device)
        best_raw_d2 = torch.zeros((B,), dtype=torch.float32, device=device)
        best_label = torch.zeros((B,), dtype=torch.int64, device=device)

        for k0 in range(0, K, centroid_chunk_size):
            k1 = min(k0 + centroid_chunk_size, K)
            c_block = c_rep[k0:k1]
            cn_block = c_norm[k0:k1]
            w_block = w_hart[k0:k1]

            if x_rep.is_cuda and x_rep.dtype in (torch.bfloat16, torch.float16):
                scores = torch.addmm(
                    cn_block[None, :],
                    x_rep,
                    c_block.t(),
                    alpha=-2.0,
                    out_dtype=torch.float32,
                )
            else:
                scores = torch.addmm(
                    cn_block[None, :],
                    x_rep.float(),
                    c_block.float().t(),
                    alpha=-2.0,
                )

            raw_d2 = (scores + x_norm[:, None]).clamp_min_(0.0)  # (B, K_tile)
            weighted_d2 = raw_d2 * w_block[None, :]              # (B, K_tile)
            tile_min_w, tile_idx = weighted_d2.min(dim=1)
            tile_raw = raw_d2.gather(1, tile_idx[:, None]).squeeze(1)

            better = tile_min_w < best_weighted
            best_weighted = torch.where(better, tile_min_w, best_weighted)
            best_raw_d2 = torch.where(better, tile_raw, best_raw_d2)
            best_label = torch.where(better, tile_idx + k0, best_label)

        out_dists[r0:r1] = best_raw_d2
        out_labels[r0:r1] = best_label

    return out_dists, out_labels


# ---------------------------------------------------------------------------
# H-Algo-3: Coherence-Gated Nesterov Centroid Extrapolation (Lookahead EMA)
# ---------------------------------------------------------------------------


def scheduled_nesterov_mu(
    mu0: float,
    step: int,
    total_steps: int,
    stop_frac: float = 0.85,
) -> float:
    """Cosine-annealed Nesterov lookahead coefficient dropping cleanly to 0 at `stop_frac`."""
    if mu0 <= 0.0:
        return 0.0
    stop_step = max(1, int(round(stop_frac * total_steps)))
    if step >= stop_step:
        return 0.0
    frac = step / float(stop_step)
    return float(mu0 * 0.5 * (1.0 + math.cos(math.pi * frac)))


@torch.no_grad()
def update_nesterov_velocity(
    velocity: torch.Tensor,
    prev_centroids: torch.Tensor,
    new_centroids: torch.Tensor,
    touched: torch.Tensor,
    beta_vel: float = 0.7,
    coherence_gate: bool = True,
) -> torch.Tensor:
    """Update per-centroid velocity `v_j` with optional directional coherence gating.

    When `coherence_gate=True`, oscillatory mini-batch stationary noise (`cos < 0`)
    is automatically damped to zero while consistent directional drift (`cos > 0`)
    accumulates smoothly.
    """
    delta_c = new_centroids - prev_centroids  # (K, D)
    if coherence_gate:
        v_norm = velocity.norm(dim=1)
        d_norm = delta_c.norm(dim=1)
        cos_align = (velocity * delta_c).sum(dim=1) / (v_norm * d_norm).clamp_min(1e-12)
        # On the first move from zero velocity, allow initialization of velocity (gate=1)
        gate = torch.where(v_norm <= 1e-12, torch.ones_like(cos_align), cos_align.clamp(0.0, 1.0))
    else:
        gate = torch.ones((velocity.shape[0],), dtype=torch.float32, device=velocity.device)

    v_touched = gate[:, None] * (beta_vel * velocity + (1.0 - beta_vel) * delta_c)
    v_untouched = beta_vel * velocity
    return torch.where(touched[:, None], v_touched, v_untouched)


# ---------------------------------------------------------------------------
# H-Algo-4: Scale-Invariant Top-2 Power-Mean Soft EMA Warmup
# ---------------------------------------------------------------------------


@torch.no_grad()
def top2_assign_chunked(
    x: torch.Tensor,
    centroids: torch.Tensor,
    compute_dtype: torch.dtype | None = None,
    centroid_chunk_size: int = 4096,
    row_chunk_size: int = 16384,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Streaming top-2 nearest-centroid search without materializing `(N x K)`.

    Returns `(d1_sq, l1, d2_sq, l2)` with `d1_sq <= d2_sq` and `l1 != l2` (when `K >= 2`),
    using exact represented-norm consistency (Appendix I.1).
    """
    if compute_dtype is None:
        if x.is_cuda and x.dtype in (torch.bfloat16, torch.float16):
            compute_dtype = x.dtype
        elif x.is_cuda:
            compute_dtype = torch.bfloat16
        else:
            compute_dtype = torch.float32

    N = x.shape[0]
    K = centroids.shape[0]
    device = x.device
    if N == 0:
        empty_f = torch.empty((0,), dtype=torch.float32, device=device)
        empty_i = torch.empty((0,), dtype=torch.int64, device=device)
        return empty_f, empty_i, empty_f.clone(), empty_i.clone()

    if K == 1:
        d1, l1 = assign_points(x, centroids, compute_dtype=compute_dtype)
        d2 = torch.full_like(d1, float("inf"))
        l2 = torch.zeros_like(l1)
        return d1, l1, d2, l2

    c_rep, c_norm = prepare_represented_centroids(centroids, compute_dtype)

    d1_out = torch.empty((N,), dtype=torch.float32, device=device)
    l1_out = torch.empty((N,), dtype=torch.int64, device=device)
    d2_out = torch.empty((N,), dtype=torch.float32, device=device)
    l2_out = torch.empty((N,), dtype=torch.int64, device=device)

    for r0 in range(0, N, row_chunk_size):
        r1 = min(r0 + row_chunk_size, N)
        xb = x[r0:r1]
        B = xb.shape[0]
        x_rep = xb.to(c_rep.dtype)
        x_norm = x_rep.float().square().sum(dim=1)  # (B,)

        best_scores = torch.full((B, 2), float("inf"), dtype=torch.float32, device=device)
        best_labels = torch.zeros((B, 2), dtype=torch.int64, device=device)
        best_labels[:, 1] = 1

        for k0 in range(0, K, centroid_chunk_size):
            k1 = min(k0 + centroid_chunk_size, K)
            c_block = c_rep[k0:k1]
            cn_block = c_norm[k0:k1]

            if x_rep.is_cuda and x_rep.dtype in (torch.bfloat16, torch.float16):
                scores = torch.addmm(
                    cn_block[None, :],
                    x_rep,
                    c_block.t(),
                    alpha=-2.0,
                    out_dtype=torch.float32,
                )
            else:
                scores = torch.addmm(
                    cn_block[None, :],
                    x_rep.float(),
                    c_block.float().t(),
                    alpha=-2.0,
                )

            q = min(2, k1 - k0)
            tile_vals, tile_idxs = torch.topk(scores, k=q, dim=1, largest=False)
            cand_s = torch.cat([best_scores, tile_vals], dim=1)       # (B, 2 + q)
            cand_l = torch.cat([best_labels, tile_idxs + k0], dim=1)  # (B, 2 + q)
            merged = torch.topk(cand_s, k=2, dim=1, largest=False)
            best_scores = merged.values
            best_labels = cand_l.gather(1, merged.indices)

        d1_out[r0:r1] = (best_scores[:, 0] + x_norm).clamp_min_(0.0)
        d2_out[r0:r1] = (best_scores[:, 1] + x_norm).clamp_min_(0.0)
        l1_out[r0:r1] = best_labels[:, 0]
        l2_out[r0:r1] = best_labels[:, 1]

    return d1_out, l1_out, d2_out, l2_out


@torch.no_grad()
def power_mean_top2_weights(
    d1_sq: torch.Tensor,
    d2_sq: torch.Tensor,
    step: int,
    warmup_steps: int,
    s0: float = -2.0,
    max_weight2: float = 0.5,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Xu & Lange (2019) scale-invariant Power-k-Means MM top-2 responsibilities.

    For `step < warmup_steps` and `s0 < 0`:
      s(t) = s0 / (1 - t / warmup_steps)  -> -inf as t -> warmup_steps
      r_i  = ((d_{i,1}^2 + eps) / (d_{i,2}^2 + eps))^(1 - s(t)) in [0, 1]
      w_2  = min(max_weight2, r_i / (1 + r_i)),  w_1 = 1 - w_2.
    Invariant to positive coordinate scaling `X -> alpha * X` (up to `eps` floor).
    """
    if s0 >= 0.0:
        raise ValueError(f"Power k-means exponent s0 must be < 0, got {s0}")
    if warmup_steps <= 0 or step >= warmup_steps:
        w1 = torch.ones_like(d1_sq, dtype=torch.float32)
        w2 = torch.zeros_like(d1_sq, dtype=torch.float32)
        return w1, w2

    rem = max(1e-4, 1.0 - (float(step) / float(warmup_steps)))
    s_t = float(s0) / rem
    exponent = min(50.0, 1.0 - s_t)  # > 1.0, smoothly growing toward hard argmin

    finite_second = torch.isfinite(d2_sq)
    safe_d2 = torch.where(finite_second, d2_sq, torch.ones_like(d2_sq))
    base_ratio = ((d1_sq + eps) / (safe_d2 + eps)).clamp(0.0, 1.0)
    r = torch.where(finite_second, base_ratio.pow(exponent), torch.zeros_like(d1_sq))
    w2 = (r / (1.0 + r)).clamp(0.0, float(max_weight2))
    w1 = 1.0 - w2
    return w1.float(), w2.float()


# ---------------------------------------------------------------------------
# H-Algo-5: Empirical-Bayes / James-Stein Variance-Shrinkage Finalization
# ---------------------------------------------------------------------------


@dataclass
class ShrinkageDiagnostics:
    """Summary diagnostics of Empirical-Bayes finalization shrinkage."""

    mode: str
    mean_lambda: float
    min_lambda_nonempty: float
    max_lambda_nonempty: float
    median_effective_kappa: float
    tau2_prior: float
    sigma2_mean: float


@torch.no_grad()
def empirical_bayes_finalize(
    closing_means: torch.Tensor,
    slow_centroids: torch.Tensor,
    pass_counts: torch.Tensor,
    pass_sse: torch.Tensor,
    d: int,
    kappa: float | str = "auto",
) -> tuple[torch.Tensor, torch.Tensor, ShrinkageDiagnostics]:
    """Combine `closing_means` and `slow_centroids` via Empirical-Bayes precision weighting.

    Model:
      - `closing_means[j]` has finite-sample estimation risk `E||mu_close_j - mu_j||^2 = sigma_j^2 / n_j`.
      - `slow_centroids[j]` has anchor drift variance `tau^2 = E||mu_slow_j - mu_j||^2`.
      - Optimal MSE convex weight: `lambda_j = n_j / (n_j + kappa_j)`.
      - When `kappa == 'auto'`, `tau^2` is estimated closed-form from `||mu_close_j - mu_slow_j||^2`
        via Empirical-Bayes method of moments with Stein dimension factor `(d - 2) / d` (`d > 2`).
    """
    counts_f = pass_counts.double()
    nonempty = counts_f > 0
    k = closing_means.shape[0]
    device = closing_means.device

    if not bool(nonempty.any()):
        diag = ShrinkageDiagnostics(
            mode=str(kappa),
            mean_lambda=0.0,
            min_lambda_nonempty=0.0,
            max_lambda_nonempty=0.0,
            median_effective_kappa=0.0,
            tau2_prior=0.0,
            sigma2_mean=0.0,
        )
        return slow_centroids.clone(), torch.zeros((k,), dtype=torch.float32, device=device), diag

    diff_sq = (closing_means.double() - slow_centroids.double()).square().sum(dim=1)  # (K,)
    total_n = counts_f.sum().clamp_min(1.0)
    global_sigma2 = float((pass_sse.double().sum() / total_n).clamp_min(0.0).item())

    if isinstance(kappa, (int, float)):
        kappa_val = max(0.0, float(kappa))
        kappa_vec = torch.full((k,), kappa_val, dtype=torch.float64, device=device)
        lam = torch.where(
            nonempty,
            counts_f / (counts_f + kappa_vec).clamp_min(1e-30),
            torch.zeros_like(counts_f),
        )
        tau2_est = float(diff_sq[nonempty].mean().item())
    elif kappa == "auto":
        # Per-cluster Bounded-Influence Positive-Part James-Stein / Empirical-Bayes shrinkage:
        #   sigma_j^2 = empirical within-cluster variance (stabilized with 4 global pseudo-obs)
        #   lambda_j  = max(1 - c_d * (sigma_j^2 / n_j) / (||mu_close_j - mu_slow_j||^2 + eps),
        #                   n_j / (n_j + kappa_max))
        # Provable properties:
        #   1. If slow_ema lagged on a split/moving cluster (||diff||^2 >> sigma_j^2 / n_j),
        #      lambda_j -> 1 automatically, preventing slow_ema lag from hurting train or test MSE!
        #   2. If cluster j is populous (n_j >> 1), sigma_j^2 / n_j -> 0 => lambda_j -> 1.
        #   3. If cluster j is settled & sparse (||diff||^2 <= sigma_j^2 / n_j), shrinks smoothly
        #      toward slow_ema down to floor n_j / (n_j + kappa_max) with kappa_max = 8.0, while
        #      strictly bounding per-cluster train SSE increase by <= c_d * sigma_j^2 * kappa_max / (n_j + kappa_max).
        raw_sigma2 = torch.where(
            counts_f >= 2.0,
            pass_sse.double() / counts_f.clamp_min(1.0),
            torch.full_like(counts_f, global_sigma2),
        ).clamp_min(0.0)
        sigma2_j = (counts_f * raw_sigma2 + 4.0 * global_sigma2) / (counts_f + 4.0)

        if global_sigma2 <= 0.0 and float(diff_sq[nonempty].max().item()) <= 0.0:
            lam = torch.where(nonempty, torch.ones_like(counts_f), torch.zeros_like(counts_f))
            kappa_vec = torch.zeros_like(counts_f)
            tau2_est = 0.0
        else:
            stein_factor = ((d - 2.0) / float(d)) if d > 2 else 1.0
            noise_var_j = stein_factor * sigma2_j / counts_f.clamp_min(1.0)
            js_lam = 1.0 - (noise_var_j / (diff_sq + 1e-30))
            kappa_max = 8.0
            floor_lam = counts_f / (counts_f + kappa_max)
            lam = torch.where(
                nonempty,
                torch.maximum(js_lam, floor_lam).clamp(0.0, 1.0),
                torch.zeros_like(counts_f),
            )
            # Implied effective pseudo-count kappa_j = n_j * (1 - lambda_j) / max(lambda_j, eps)
            kappa_vec = torch.where(
                nonempty & (lam > 1e-12),
                counts_f * (1.0 - lam) / lam.clamp_min(1e-12),
                torch.zeros_like(counts_f),
            )
            tau2_est = float(torch.clamp(diff_sq[nonempty] - noise_var_j[nonempty], min=0.0).mean().item())
    else:
        raise ValueError(f"kappa must be a non-negative float or 'auto', got {kappa!r}")

    lam_f32 = lam.float().clamp(0.0, 1.0)
    final_c = slow_centroids.float() + lam_f32[:, None] * (
        closing_means.float() - slow_centroids.float()
    )
    # Guarantee exact closing_mean bit-identity when lambda == 1 and slow_centroids when lambda == 0
    final_c = torch.where((lam_f32 == 1.0)[:, None], closing_means.float(), final_c)
    final_c = torch.where((~nonempty)[:, None], slow_centroids.float(), final_c)

    lam_ne = lam_f32[nonempty]
    kap_ne = kappa_vec[nonempty].float()
    diag = ShrinkageDiagnostics(
        mode=str(kappa),
        mean_lambda=float(lam_ne.mean().item()),
        min_lambda_nonempty=float(lam_ne.min().item()),
        max_lambda_nonempty=float(lam_ne.max().item()),
        median_effective_kappa=float(kap_ne.median().item()),
        tau2_prior=tau2_est,
        sigma2_mean=global_sigma2,
    )
    return final_c.contiguous(), lam_f32, diag


# ---------------------------------------------------------------------------
# Huygens Rare-Mode Protection Helper (for e-means++ base engine)
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_huygens_nn_sq_dists(
    centroids: torch.Tensor,
    candidate_indices: torch.Tensor,
    tile_size: int = 512,
) -> torch.Tensor:
    """Compute exact `min_{m != j} ||c_j - c_m||^2` for `j in candidate_indices`.

    Strictly masks self-pairs `m == j` to `+inf` and centers coordinates around
    their mean to prevent FP32 catastrophic cancellation on shifted datasets.
    """
    m_cand = candidate_indices.numel()
    k = centroids.shape[0]
    device = centroids.device
    if m_cand == 0:
        return torch.empty((0,), dtype=torch.float32, device=device)
    if k <= 1:
        return torch.full((m_cand,), float("inf"), dtype=torch.float32, device=device)

    origin = centroids.float().mean(dim=0, keepdim=True)
    cf = (centroids.float() - origin).contiguous()
    c_norm = cf.square().sum(dim=1)  # (K,)
    out = torch.empty((m_cand,), dtype=torch.float32, device=device)
    all_cols = torch.arange(k, device=device)

    for s in range(0, m_cand, tile_size):
        e = min(s + tile_size, m_cand)
        idx = candidate_indices[s:e]  # (T,)
        q = cf[idx]                   # (T, D)
        q_norm = c_norm[idx]          # (T,)

        best = torch.full((e - s,), float("inf"), dtype=torch.float32, device=device)
        for k0 in range(0, k, 4096):
            k1 = min(k0 + 4096, k)
            cb = cf[k0:k1]
            cn = c_norm[k0:k1]
            d2 = torch.addmm(cn[None, :], q, cb.t(), alpha=-2.0)
            d2.add_(q_norm[:, None]).clamp_min_(0.0)
            self_mask = idx[:, None] == all_cols[None, k0:k1]
            d2 = torch.where(self_mask, torch.full_like(d2, float("inf")), d2)
            best = torch.minimum(best, d2.min(dim=1).values)
        out[s:e] = best

    return out


# ---------------------------------------------------------------------------
# Unified Phase 2 Modular Estimator (`EMeansAlgoExt`)
# ---------------------------------------------------------------------------


class EMeansAlgoExt:
    """Modular research engine testing `H-Algo-1..5` atop `e-means++` or `e-means`.

    When all five extensions are at their baseline defaults (`split_mode='checkerboard'`,
    `hartigan_beta0=0.0`, `nesterov_mu0=0.0`, `power_warmup_frac=0.0`,
    `finalization='closing_mean'`), this engine executes pure `e-means++`
    (`base_mode='emeans_pp'`) or pure `e-means` (`base_mode='emeans'`).
    """

    def __init__(
        self,
        d: int,
        k: int,
        nepochs: float = 5.0,
        base_mode: str = "emeans_pp",
        # e-means++ seeding & warm-start parameters (aligned 1:1 with EMeansPP defaults)
        seeding_mode: str = "auto",
        coreset_mult: int = 16,
        min_coreset_size: int = 8192,
        use_voronoi_polish: bool = True,
        prior_mass_scale: float = 0.5,
        prior_floor: float = 1e-3,
        warm_alpha_scale: float | None = None,
        huygens_protect: bool = True,
        gamma_protect: float = 0.5,
        grace_steps: int = 5,
        distortion_donors: bool = True,
        # Core e-means recipe parameters
        batch_law: str = "auto",
        batch_size: int | None = None,
        alpha0: float | None = None,
        alpha_min_factor: float = ALPHA_MIN_FACTOR,
        unused_threshold: float | None = None,
        check_unused_every: int = CHECK_UNUSED_EVERY,
        split_epsilon: float = SPLIT_EPSILON,
        init_usage: float = DEFAULT_INIT_USAGE,
        revival_mode: str = "split",
        use_cosine_schedule: bool = True,
        # H-Algo-1: Principal-Axis Donor Splitting
        split_mode: str = "checkerboard",
        split_std_frac: float = 0.08,
        # H-Algo-2: Annealed Hartigan Marginal Distortion Assignment
        hartigan_beta0: float = 0.0,
        hartigan_stop_frac: float = 0.85,
        hartigan_n_min: float = 0.5,
        # H-Algo-3: Coherence-Gated Nesterov Centroid Extrapolation
        nesterov_mu0: float = 0.0,
        nesterov_beta: float = 0.7,
        nesterov_coherence_gate: bool = True,
        nesterov_stop_frac: float = 0.85,
        # H-Algo-4: Scale-Invariant Top-2 Power-Mean Soft EMA Warmup
        power_warmup_frac: float = 0.0,
        power_s0: float = -2.0,
        power_max_w2: float = 0.5,
        # H-Algo-5: Finalization & Empirical-Bayes Variance Shrinkage
        finalization: str = "closing_mean",
        slow_ema_mult: float = 0.25,
        shrinkage_kappa: float | str = "auto",
        # Execution & reproducibility
        compute_dtype: torch.dtype | None = None,
        backend: str = "auto",
        seed: int = 1234,
        verbose: bool = False,
    ) -> None:
        if base_mode not in ("emeans_pp", "emeans"):
            raise ValueError(f"base_mode must be 'emeans_pp' or 'emeans', got {base_mode!r}")
        if finalization not in ("closing_mean", "slow_ema", "eb_shrinkage", "none"):
            raise ValueError(
                f"finalization must be 'closing_mean', 'slow_ema', 'eb_shrinkage', or 'none', got {finalization!r}"
            )
        if revival_mode not in ("split", "resample", "off"):
            raise ValueError(f"revival_mode must be 'split', 'resample', or 'off', got {revival_mode!r}")
        if split_mode not in ("checkerboard", "oja", "var_axis", "bf16_random"):
            raise ValueError(
                f"split_mode must be 'checkerboard', 'oja', 'var_axis', or 'bf16_random', got {split_mode!r}"
            )

        self.d = int(d)
        self.k = int(k)
        self.nepochs = float(nepochs)
        self.base_mode = base_mode

        self.seeding_mode = seeding_mode
        self.coreset_mult = int(coreset_mult)
        self.min_coreset_size = int(min_coreset_size)
        self.use_voronoi_polish = bool(use_voronoi_polish)
        self.prior_mass_scale = float(prior_mass_scale)
        self.prior_floor = float(prior_floor)
        if warm_alpha_scale is None:
            self.warm_alpha_scale = 0.85 if base_mode == "emeans_pp" else 1.0
        else:
            self.warm_alpha_scale = float(warm_alpha_scale)
        self.huygens_protect = bool(huygens_protect) if base_mode == "emeans_pp" else False
        self.gamma_protect = float(gamma_protect)
        self.grace_steps = int(grace_steps) if base_mode == "emeans_pp" else 0
        self.distortion_donors = bool(distortion_donors) if base_mode == "emeans_pp" else False

        self.batch_law = batch_law
        self.batch_size = batch_size
        self.alpha0 = alpha0
        self.alpha_min_factor = float(alpha_min_factor)
        self.unused_threshold = unused_threshold
        self.check_unused_every = int(check_unused_every)
        self.split_epsilon = float(split_epsilon)
        self.init_usage = float(init_usage)
        self.revival_mode = revival_mode
        self.use_cosine_schedule = bool(use_cosine_schedule)

        self.split_mode = split_mode
        self.split_std_frac = float(split_std_frac)

        self.hartigan_beta0 = float(hartigan_beta0)
        self.hartigan_stop_frac = float(hartigan_stop_frac)
        self.hartigan_n_min = float(hartigan_n_min)

        self.nesterov_mu0 = float(nesterov_mu0)
        self.nesterov_beta = float(nesterov_beta)
        self.nesterov_coherence_gate = bool(nesterov_coherence_gate)
        self.nesterov_stop_frac = float(nesterov_stop_frac)

        self.power_warmup_frac = float(power_warmup_frac)
        self.power_s0 = float(power_s0)
        self.power_max_w2 = float(power_max_w2)

        self.finalization = finalization
        self.slow_ema_mult = float(slow_ema_mult)
        self.shrinkage_kappa = shrinkage_kappa

        self.compute_dtype = compute_dtype
        self.backend = backend
        self.seed = int(seed)
        self.verbose = bool(verbose)

        # Fitted state & diagnostics
        self.recipe: EMeansRecipe | None = None
        self.centroids: torch.Tensor | None = None
        self.closing_centroids: torch.Tensor | None = None
        self.slow_centroids: torch.Tensor | None = None
        self.pre_final_centroids: torch.Tensor | None = None
        self.assignments: torch.Tensor | None = None
        self.iteration_stats: list[EpochStat] = []
        self.shrinkage_diagnostics: ShrinkageDiagnostics | None = None
        self.shrinkage_lambdas: torch.Tensor | None = None
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
        seeding_result: SeedingResult | None = None,
        capture_assignments: bool = True,
    ) -> tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray | None]:
        """Fit clustering model on `x` of shape `(N, D)`."""
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
            init_usage=self.init_usage,
        )
        eff_alpha0 = (
            base_recipe.alpha0
            if (self.alpha0 is not None or self.warm_alpha_scale == 1.0)
            else float(min(max(base_recipe.alpha0 * self.warm_alpha_scale, 1e-4), 1.0))
        )
        if eff_alpha0 != base_recipe.alpha0:
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
                init_usage=self.init_usage,
            )
        else:
            recipe = base_recipe
        self.recipe = recipe

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_init_start = time.perf_counter()

        prior_probs: torch.Tensor | None = None
        prior_dist: torch.Tensor | None = None

        if seeding_result is not None:
            centroids = seeding_result.centroids.to(device=device, dtype=torch.float32).clone().contiguous()
            prior_probs = seeding_result.prior_probs.to(device=device, dtype=torch.float32).clone()
            prior_dist = seeding_result.prior_distortion.to(device=device, dtype=torch.float32).clone()
        elif init_centroids is not None:
            c_in = (
                torch.from_numpy(np.asarray(init_centroids, dtype=np.float32))
                if isinstance(init_centroids, np.ndarray)
                else init_centroids
            )
            raw_c = c_in.to(device=device, dtype=torch.float32).clone().contiguous()
            if self.base_mode == "emeans_pp":
                m_sub = min(n, max(self.min_coreset_size, self.coreset_mult * self.k, self.k * 2))
                x_sub, w_sub = lightweight_coreset(x_t, m=m_sub, seed=self.seed)
                if self.use_voronoi_polish:
                    s_res = voronoi_polish_and_priors(x_sub, w_sub, raw_c, compute_dtype=comp_dtype)
                else:
                    sq_d, lbls = assign_points(x_sub, raw_c, compute_dtype=comp_dtype)
                    mass = torch.zeros((self.k,), dtype=torch.float32, device=device)
                    mass.index_add_(0, lbls, w_sub)
                    dist_m = torch.zeros((self.k,), dtype=torch.float32, device=device)
                    dist_m.index_add_(0, lbls, w_sub * sq_d)
                    tot_m = mass.sum().clamp_min(1e-30)
                    s_res = SeedingResult(
                        centroids=raw_c,
                        prior_probs=(mass / tot_m).contiguous(),
                        prior_distortion=(dist_m / tot_m).contiguous(),
                    )
                centroids = s_res.centroids.to(device=device, dtype=torch.float32).clone().contiguous()
                prior_probs = s_res.prior_probs.to(device=device, dtype=torch.float32)
                prior_dist = s_res.prior_distortion.to(device=device, dtype=torch.float32)
            else:
                centroids = raw_c
        elif self.base_mode == "emeans_pp":
            s_res = seed_emeans_plusplus(
                x_t,
                self.k,
                seeding_mode=self.seeding_mode,
                coreset_mult=self.coreset_mult,
                min_coreset_size=self.min_coreset_size,
                polish=self.use_voronoi_polish,
                seed=self.seed,
            )
            centroids = s_res.centroids.clone().contiguous()
            prior_probs = s_res.prior_probs
            prior_dist = s_res.prior_distortion
        else:
            gen_init = torch.Generator(device=device)
            gen_init.manual_seed(self.seed)
            perm = torch.randperm(n, generator=gen_init, device=device)[: self.k]
            centroids = x_t[perm].float().clone().contiguous()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self.init_time_sec = time.perf_counter() - t_init_start

        t_train_start = time.perf_counter()

        # Initialize EMA usage N_j, distortion E_j, and epoch-horizon survival_mass
        bsz_float = float(recipe.batch_size)
        if self.base_mode == "emeans_pp" and prior_probs is not None:
            min_prob = max(0.0, self.prior_floor) / float(self.k)
            p_clamped = prior_probs.clamp_min(min_prob)
            usage = (self.prior_mass_scale * bsz_float * p_clamped).float().contiguous()
            survival_mass: torch.Tensor | None = (bsz_float * p_clamped).float().contiguous()
            if prior_dist is not None:
                mean_ehat = prior_dist.mean().clamp_min(1e-12)
                ema_sse = (
                    self.prior_mass_scale * bsz_float * prior_dist.clamp_min(1e-6 * mean_ehat)
                ).float().contiguous()
            else:
                ema_sse = torch.zeros((self.k,), dtype=torch.float32, device=device)
        else:
            usage = torch.full((self.k,), recipe.init_usage, dtype=torch.float32, device=device)
            ema_sse = torch.zeros((self.k,), dtype=torch.float32, device=device)
            survival_mass = None

        # Track slow EMA whenever needed for 'slow_ema' or 'eb_shrinkage'
        track_slow = self.finalization in ("slow_ema", "eb_shrinkage")
        if track_slow:
            slow_centroids = centroids.clone()
            slow_usage = usage.clone()
            alpha_slow = min(1.0, max(1e-6, self.slow_ema_mult * recipe.alpha_min_factor * eff_alpha0))
        else:
            slow_centroids = None
            slow_usage = None
            alpha_slow = 0.0

        # H-Algo-1 state: online principal axes or coordinate variances
        pca_gen = torch.Generator(device=device)
        pca_gen.manual_seed((self.seed + 31337) % (2**32))
        if self.split_mode in ("oja", "bf16_random"):
            init_dirs = torch.randn((self.k, self.d), generator=pca_gen, device=device, dtype=torch.float32)
            init_dirs = init_dirs / init_dirs.norm(dim=1, keepdim=True).clamp_min(1e-12)
            # Small non-zero initial scale before first batch residual observation
            prin_vec: torch.Tensor | None = 1e-3 * init_dirs
            g_buf: torch.Tensor | None = torch.empty((self.k, self.d), dtype=torch.float32, device=device)
        else:
            prin_vec = None
            g_buf = None

        if self.split_mode == "var_axis":
            coord_var: torch.Tensor | None = torch.full(
                (self.k, self.d), 1e-6, dtype=torch.float32, device=device
            )
            v_buf: torch.Tensor | None = torch.empty((self.k, self.d), dtype=torch.float32, device=device)
        else:
            coord_var = None
            v_buf = None

        # H-Algo-3 state: Nesterov drift velocity
        velocity = (
            torch.zeros((self.k, self.d), dtype=torch.float32, device=device)
            if self.nesterov_mu0 > 0.0
            else None
        )

        # H-Algo-4 state: warmup step horizon (strictly capped before closing pass if num_epochs >= 2)
        num_epochs = len(recipe.epoch_rows)
        non_closing_steps = sum(recipe.epoch_steps[:-1]) if num_epochs >= 2 else max(0, recipe.total_steps - 1)
        power_warmup_steps = (
            min(int(round(self.power_warmup_frac * recipe.total_steps)), max(1, non_closing_steps))
            if self.power_warmup_frac > 0.0 and recipe.total_steps > 1
            else 0
        )

        split_gen = torch.Generator(device=device)
        split_gen.manual_seed((self.seed + 7919) % (2**32))

        assignments = (
            torch.full((n,), -1, dtype=torch.int64, device=device)
            if capture_assignments
            else None
        )

        ones_buf = torch.ones((1,), dtype=torch.float32, device=device)
        batch_sums = torch.empty((self.k, self.d), dtype=torch.float32, device=device)
        batch_sse_buf = torch.empty((self.k,), dtype=torch.float32, device=device)

        step = 0
        self.n_splits = 0
        self.n_protected = 0
        self.iteration_stats = []

        final_pass_counts: torch.Tensor | None = None
        final_pass_sums: torch.Tensor | None = None
        final_pass_sse: torch.Tensor | None = None

        for ep_idx, rows_in_ep in enumerate(recipe.epoch_rows):
            is_last_epoch = ep_idx == (num_epochs - 1)
            gen_ep = torch.Generator(device=device)
            gen_ep.manual_seed(epoch_seed(self.seed, ep_idx))
            order = torch.randperm(n, generator=gen_ep, device=device)[:rows_in_ep]

            pass_counts = torch.zeros((self.k,), dtype=torch.float64, device=device)
            need_closing_stats = is_last_epoch and self.finalization in ("closing_mean", "eb_shrinkage")
            pass_sums = (
                torch.zeros((self.k, self.d), dtype=torch.float32, device=device)
                if need_closing_stats
                else None
            )
            pass_sse = (
                torch.zeros((self.k,), dtype=torch.float64, device=device)
                if (is_last_epoch and self.finalization == "eb_shrinkage")
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

                # H-Algo-3: Nesterov extrapolated query centroids (disabled in closing pass)
                if velocity is not None and not is_last_epoch:
                    mu_t = scheduled_nesterov_mu(
                        self.nesterov_mu0,
                        step,
                        recipe.total_steps,
                        stop_frac=self.nesterov_stop_frac,
                    )
                    query_centroids = centroids + mu_t * velocity if mu_t > 0.0 else centroids
                else:
                    query_centroids = centroids

                # H-Algo-2: Annealed Hartigan size-correction factor beta(t) (disabled in closing pass)
                if self.hartigan_beta0 > 0.0 and not is_last_epoch:
                    h_stop = max(1, int(round(self.hartigan_stop_frac * recipe.total_steps)))
                    if step < h_stop:
                        beta_t = float(
                            self.hartigan_beta0 * 0.5 * (1.0 + math.cos(math.pi * step / float(h_stop)))
                        )
                    else:
                        beta_t = 0.0
                else:
                    beta_t = 0.0

                # H-Algo-4: Top-2 Power-Mean Soft EMA during warmup vs Hard Assignment
                use_soft_top2 = (
                    power_warmup_steps > 0
                    and step < power_warmup_steps
                    and not is_last_epoch
                    and self.k >= 2
                )

                if use_soft_top2:
                    d1_sq, l1, d2_sq, l2 = top2_assign_chunked(
                        x_batch,
                        query_centroids,
                        compute_dtype=comp_dtype,
                    )
                    sq_dists = d1_sq
                    labels = l1
                    w1, w2 = power_mean_top2_weights(
                        d1_sq,
                        d2_sq,
                        step=step,
                        warmup_steps=power_warmup_steps,
                        s0=self.power_s0,
                        max_weight2=self.power_max_w2,
                    )
                    counts = torch.zeros((self.k,), dtype=torch.float32, device=device)
                    counts.index_add_(0, l1, w1)
                    counts.index_add_(0, l2, w2)

                    xf_batch = x_batch.float()
                    batch_sums.zero_()
                    batch_sums.index_add_(0, l1, xf_batch * w1[:, None])
                    batch_sums.index_add_(0, l2, xf_batch * w2[:, None])
                else:
                    if beta_t > 0.0:
                        sq_dists, labels = hartigan_assign_points(
                            x_batch,
                            query_centroids,
                            usage=usage,
                            n_total=n,
                            beta=beta_t,
                            compute_dtype=comp_dtype,
                            backend=self.backend,
                            hartigan_n_min=self.hartigan_n_min,
                        )
                    else:
                        c_rep, c_norm = prepare_represented_centroids(query_centroids, comp_dtype)
                        sq_dists, labels = assign_points(
                            x_batch,
                            query_centroids,
                            compute_dtype=comp_dtype,
                            backend=self.backend,
                            c_rep_cached=(c_rep, c_norm),
                        )

                    counts = torch.bincount(labels, minlength=self.k).float()
                    batch_sums.zero_()
                    batch_sums.index_add_(0, labels, x_batch.float())

                if is_last_epoch and assignments is not None:
                    assignments[batch_idx] = labels

                ep_sq_dist_t += sq_dists.double().sum()

                # Hard counts for exact closing-pass statistics & imbalance factor
                if use_soft_top2:
                    hard_counts = torch.bincount(labels, minlength=self.k).double()
                else:
                    hard_counts = counts.double()
                pass_counts += hard_counts

                if pass_sums is not None:
                    if use_soft_top2:
                        pass_sums.index_add_(0, labels, x_batch.float())
                    else:
                        pass_sums += batch_sums
                if pass_sse is not None:
                    pass_sse.index_add_(0, labels, sq_dists.double())

                # Step size alpha(t)
                if self.use_cosine_schedule:
                    alpha = scheduled_alpha(
                        eff_alpha0,
                        step,
                        recipe.total_steps,
                        recipe.alpha_min_factor,
                    )
                else:
                    alpha = eff_alpha0
                decay = 1.0 - alpha

                # EMA update on (usage, centroids)
                touched = counts > 0
                if alpha < 1.0:
                    usage.mul_(decay)
                    retained = usage
                else:
                    retained = torch.zeros_like(usage)

                new_mass = retained + alpha * counts
                safe_counts = torch.where(touched, counts, ones_buf)
                batch_means = batch_sums / safe_counts[:, None]
                weight = (alpha * counts) / torch.where(touched, new_mass, ones_buf)

                # Update H-Algo-1 covariance / variance statistics BEFORE moving centroids
                # (or using current centroids so residuals r_i = x_i - c_j match the batch assignment)
                if prin_vec is not None:
                    hard_c_for_pca = (
                        torch.bincount(labels, minlength=self.k).float() if use_soft_top2 else counts
                    )
                    prin_vec = update_oja_principal_axes(
                        prin_vec,
                        x_batch,
                        centroids,
                        labels,
                        hard_c_for_pca,
                        weight,
                        g_buf=g_buf,
                    )
                if coord_var is not None:
                    hard_c_for_var = (
                        torch.bincount(labels, minlength=self.k).float() if use_soft_top2 else counts
                    )
                    coord_var = update_coord_variance_ema(
                        coord_var,
                        x_batch,
                        centroids,
                        labels,
                        hard_c_for_var,
                        weight,
                        v_buf=v_buf,
                    )

                prev_centroids = centroids if velocity is not None else None
                updated_pos = centroids + weight[:, None] * (batch_means - centroids)
                updated_pos = torch.where((weight == 1.0)[:, None], batch_means, updated_pos)
                centroids = torch.where(touched[:, None], updated_pos, centroids)
                usage = torch.where(touched, new_mass, usage)

                # Update per-cluster EMA SSE & survival_mass for e-means++ distortion revival
                if self.distortion_donors or self.huygens_protect:
                    batch_sse_buf.zero_()
                    batch_sse_buf.index_add_(0, labels, sq_dists.float())
                    if alpha < 1.0:
                        ema_sse.mul_(decay).add_(batch_sse_buf, alpha=alpha)
                    else:
                        ema_sse = torch.where(touched, batch_sse_buf, ema_sse * 0.5)
                if survival_mass is not None:
                    alpha_surv = min(alpha, 2.0 * float(bsz) / float(max(1, n)))
                    survival_mass.mul_(1.0 - alpha_surv).add_(counts, alpha=alpha_surv)

                # Update H-Algo-3 Nesterov velocity
                if velocity is not None and prev_centroids is not None:
                    velocity = update_nesterov_velocity(
                        velocity,
                        prev_centroids,
                        centroids,
                        touched,
                        beta_vel=self.nesterov_beta,
                        coherence_gate=self.nesterov_coherence_gate,
                    )

                # Update slow EMA for 'slow_ema' / 'eb_shrinkage'
                if track_slow and slow_centroids is not None and slow_usage is not None:
                    if alpha_slow < 1.0:
                        slow_usage.mul_(1.0 - alpha_slow)
                        s_retained = slow_usage
                    else:
                        s_retained = torch.zeros_like(slow_usage)
                    s_new_mass = s_retained + alpha_slow * counts
                    s_weight = (alpha_slow * counts) / torch.where(touched, s_new_mass, ones_buf)
                    s_pos = slow_centroids + s_weight[:, None] * (batch_means - slow_centroids)
                    s_pos = torch.where((s_weight == 1.0)[:, None], batch_means, s_pos)
                    slow_centroids = torch.where(touched[:, None], s_pos, slow_centroids)
                    slow_usage = torch.where(touched, s_new_mass, slow_usage)

                # Dead-cluster revival every `check_unused_every` steps
                eff_grace = (
                    min(self.grace_steps, max(0, recipe.total_steps // 3))
                    if self.huygens_protect
                    else 0
                )
                if (
                    recipe.unused_threshold > 0.0
                    and self.revival_mode != "off"
                    and ((step + 1) >= max(1, eff_grace) if self.huygens_protect else True)
                    and ((step + 1) % recipe.check_unused_every == 0)
                ):
                    thresh = usage.mean() * recipe.unused_threshold
                    low_mask = usage < thresh

                    # e-means++ Huygens Parallel-Axis Rare-Mode Protection
                    if self.huygens_protect and self.k >= 2 and self.gamma_protect > 0.0 and bool(low_mask.any()):
                        cand_idx = torch.nonzero(low_mask, as_tuple=False).reshape(-1)
                        nn_d2 = compute_huygens_nn_sq_dists(centroids, cand_idx)
                        if survival_mass is not None:
                            eff_mass = torch.maximum(
                                usage[cand_idx],
                                torch.minimum(survival_mass[cand_idx], thresh),
                            )
                        else:
                            eff_mass = usage[cand_idx]
                        removal_cost = eff_mass * nn_d2
                        protect_thresh = self.gamma_protect * ema_sse.mean().clamp_min(1e-30)
                        keep_rare = removal_cost >= protect_thresh
                        if bool(keep_rare.any()):
                            self.n_protected += int(keep_rare.sum().item())
                            protected_idx = cand_idx[keep_rare]
                            unused_mask = low_mask.clone()
                            unused_mask[protected_idx] = False
                        else:
                            unused_mask = low_mask
                    else:
                        unused_mask = low_mask

                    # Protected rare clusters must never be chosen as split donors either
                    donor_base = (~low_mask) if self.huygens_protect else (~unused_mask)
                    donor_mask = donor_base & (usage > 0)
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

                            if self.distortion_donors:
                                pos_dist_donors = donor_mask & (ema_sse > 1e-30)
                                if int(pos_dist_donors.sum().item()) >= nsplit:
                                    donor_weights = ema_sse
                                    valid_donor = pos_dist_donors
                                else:
                                    donor_weights = usage
                                    valid_donor = donor_mask
                            else:
                                valid_donor = donor_mask
                                donor_weights = usage

                            safe_dw = torch.where(
                                valid_donor,
                                donor_weights.clamp_min(1e-30) if self.distortion_donors else donor_weights,
                                ones_buf,
                            )
                            keys = torch.where(
                                valid_donor,
                                safe_dw.log() - (-u.log()).log(),
                                torch.full_like(usage, float("-inf")),
                            )
                            source = torch.topk(keys, nsplit).indices
                            target = torch.nonzero(unused_mask, as_tuple=False).reshape(-1)[:nsplit]

                            donors = centroids[source]
                            d_cent, c_cent, split_dirs = compute_directional_split(
                                donors=donors,
                                source_idx=source,
                                split_mode=self.split_mode,
                                split_epsilon=recipe.split_epsilon,
                                split_std_frac=self.split_std_frac,
                                compute_dtype=comp_dtype,
                                prin_vec=prin_vec,
                                coord_var=coord_var,
                                generator=pca_gen,
                            )

                            half_u = usage[source] * 0.5
                            centroids[source] = d_cent
                            centroids[target] = c_cent
                            usage[source] = half_u
                            usage[target] = half_u

                            if self.distortion_donors or self.huygens_protect:
                                half_e = ema_sse[source] * 0.5
                                ema_sse[source] = half_e
                                ema_sse[target] = half_e
                            if survival_mass is not None:
                                half_m = survival_mass[source] * 0.5
                                survival_mass[source] = half_m
                                survival_mass[target] = half_m

                            if track_slow and slow_centroids is not None and slow_usage is not None:
                                s_donors = slow_centroids[source]
                                sd_cent, sc_cent, _ = compute_directional_split(
                                    donors=s_donors,
                                    source_idx=source,
                                    split_mode=self.split_mode,
                                    split_epsilon=recipe.split_epsilon,
                                    split_std_frac=self.split_std_frac,
                                    compute_dtype=comp_dtype,
                                    prin_vec=prin_vec,
                                    coord_var=coord_var,
                                    generator=pca_gen,
                                    precomputed_dirs=split_dirs,
                                )
                                shalf_u = slow_usage[source] * 0.5
                                slow_centroids[source] = sd_cent
                                slow_centroids[target] = sc_cent
                                slow_usage[source] = shalf_u
                                slow_usage[target] = shalf_u

                            if prin_vec is not None:
                                reset_post_split_oja_axes(
                                    prin_vec,
                                    source,
                                    target,
                                    split_dirs,
                                    generator=pca_gen,
                                )
                            if coord_var is not None:
                                cv_src = coord_var[source].clone()
                                max_idx = cv_src.argmax(dim=1)
                                cv_src[torch.arange(nsplit, device=device), max_idx] *= 0.25
                                coord_var[source] = cv_src
                                coord_var[target] = cv_src.clone()

                            if velocity is not None:
                                velocity[source] = 0.0
                                velocity[target] = 0.0

                            self.n_splits += nsplit
                    elif self.revival_mode == "resample" and n_unused > 0 and bsz > 0:
                        nsplit = min(n_unused, bsz)
                        target = torch.nonzero(unused_mask, as_tuple=False).reshape(-1)[:nsplit]
                        r_idx = torch.randperm(bsz, generator=split_gen, device=device)[:nsplit]
                        centroids[target] = x_batch[r_idx].float()
                        mean_u = (
                            usage.mean().clamp_min(recipe.init_usage)
                            if self.base_mode == "emeans_pp"
                            else usage.mean()
                        )
                        usage[target] = mean_u
                        if self.distortion_donors or self.huygens_protect:
                            ema_sse[target] = ema_sse.mean().clamp_min(1e-12)
                        if survival_mass is not None:
                            survival_mass[target] = mean_u
                        if prin_vec is not None:
                            rnd_p = torch.randn(
                                (nsplit, self.d), generator=pca_gen, device=device, dtype=torch.float32
                            )
                            prin_vec[target] = 1e-3 * (rnd_p / rnd_p.norm(dim=1, keepdim=True).clamp_min(1e-12))
                        if coord_var is not None:
                            coord_var[target] = coord_var.mean(dim=0, keepdim=True).clamp_min(1e-6)
                        if velocity is not None:
                            velocity[target] = 0.0
                        self.n_splits += nsplit

                step += 1
                ep_steps += 1

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            ep_sq_dist = float(ep_sq_dist_t.item())
            imbalance = float(
                (self.k * torch.dot(pass_counts, pass_counts) / (float(max(1, rows_in_ep)) ** 2)).item()
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
                    f"[EMeansAlgoExt:{self.base_mode}] pass {ep_idx + 1}/{num_epochs} "
                    f"({rows_in_ep} rows, {ep_steps} steps): moving_mse={stat.moving_mse:.6f}, "
                    f"splits={stat.n_splits}, time={stat.elapsed_sec:.3f}s"
                )

            if is_last_epoch:
                final_pass_counts = pass_counts
                final_pass_sums = pass_sums
                final_pass_sse = pass_sse

        # Save pre-finalization EMA centroids & compute closing_centroids
        self.pre_final_centroids = centroids.clone().contiguous()

        if final_pass_counts is not None and final_pass_sums is not None:
            nonempty = final_pass_counts > 0
            closing_c = torch.where(
                nonempty[:, None],
                final_pass_sums / final_pass_counts.clamp_min(1.0).float()[:, None],
                centroids,
            ).contiguous()
        else:
            closing_c = None
        self.closing_centroids = closing_c

        if self.finalization == "closing_mean" and closing_c is not None:
            centroids = closing_c
        elif self.finalization == "slow_ema" and slow_centroids is not None:
            centroids = slow_centroids.contiguous()
        elif (
            self.finalization == "eb_shrinkage"
            and closing_c is not None
            and slow_centroids is not None
            and final_pass_counts is not None
            and final_pass_sse is not None
        ):
            centroids, lam_w, s_diag = empirical_bayes_finalize(
                closing_means=closing_c,
                slow_centroids=slow_centroids,
                pass_counts=final_pass_counts,
                pass_sse=final_pass_sse,
                d=self.d,
                kappa=self.shrinkage_kappa,
            )
            self.shrinkage_lambdas = lam_w
            self.shrinkage_diagnostics = s_diag

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self.train_time_sec = time.perf_counter() - t_train_start
        self.total_time_sec = self.init_time_sec + self.train_time_sec

        self.centroids = centroids.contiguous()
        self.slow_centroids = slow_centroids.contiguous() if slow_centroids is not None else None
        self.assignments = assignments

        if input_is_numpy:
            return (
                self.centroids.cpu().numpy(),
                self.assignments.cpu().numpy() if self.assignments is not None else None,
            )
        return self.centroids, self.assignments

    @torch.no_grad()
    def assign(
        self, x: torch.Tensor | np.ndarray
    ) -> tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray]:
        """Assign points `x` to fitted centroids, returning `(sq_distances, labels)`."""
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


__all__ = [
    "EMeansAlgoExt",
    "FalsifiableHypothesis",
    "HYPOTHESES",
    "ShrinkageDiagnostics",
    "compute_directional_split",
    "compute_huygens_nn_sq_dists",
    "empirical_bayes_finalize",
    "hartigan_assign_points",
    "power_mean_top2_weights",
    "reset_post_split_oja_axes",
    "scheduled_nesterov_mu",
    "separate_collapsed_directional",
    "top2_assign_chunked",
    "update_coord_variance_ema",
    "update_nesterov_velocity",
    "update_oja_principal_axes",
]
