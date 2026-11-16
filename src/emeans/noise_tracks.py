"""Forward-Only & Perturbation Optimization Tracks (`qlabs-eng/noise`) for `e-means` and `e-means++`.

Translates five signature forward-only learning mechanisms from `https://github.com/qlabs-eng/noise`
(Gülbahar et al., 2026) into modular, GPU-native enhancements on top of `EMeans` and `EMeansPP`:

- `N-Track-1` (`head_edit` Forward-Verified Signed-Hadamard Centroid Transplant):
    Translates `module head_edit` (`train/val/run.py:10272-10497`), `hadamard()`, and exact
    partition-swap scoring (`coordinate_differences` / `frozen_attention`) to clustering:
    splits mini-batches into disjoint Proposal half `P` and Confirmation half `Q`, computes
    exact top-3 Voronoi partition-swap dormancy (shielded by Huygens rare-mode protection),
    evaluates multi-probe signed Sylvester-Hadamard donor splits on `P`, and commits swaps
    iff held-back `Q` confirms a strict forward SSE reduction (`Delta SSE_Q < 0`).
- `N-Track-2` (`quad_merge` In-Run Held-Out Parabolic Trajectory Merge):
    Translates `quad_merge` (`train/speedrun/run.py:3057-3214`, commit `aa67d44`) and
    `soup_parts` (`train/val/run.py:2457-2625`): carves a small disjoint internal slice
    `X_hold` strictly from `x_train` (zero validation/test access), tracks split-consistent
    tail EMA trajectory anchors over the final 70% of updates, fits closed-form 3-point
    parabolic interpolation vertices per cluster-usage stratum on exact hard-Voronoi loss
    over `X_hold` + global shrinkage `QUAD_SHRINKS`, and folds `X_hold` statistics cleanly.
- `N-Track-3` (`anvil_polar` ANVIL-II Dual-Rail Gram-Newton-Schulz Polar Centroid Update):
    Translates `Anvil` / `BatchedAnvil` (`train/val/run.py:2135-2282`): maintains dual
    fast/slow momentum rails (`0.85` / `0.98`) on the `(k, d)` centroid displacement matrix,
    applies the 6-step quintic Gram-Newton-Schulz polar map (`ANVIL_MAPS`) on `min(k, d)` +
    RMS lane energy normalization norm-matched to `||Delta C_ema||_F`, and anneals into
    convex-hull EMA updates (`e-means` Appendix G).
- `N-Track-4` (`dust_antithetic` Zero-Overhead Split-Batch Antithetic Hadamard Voronoi Jitter):
    Translates `Dust` (`train/val/dust.py`) and `EGGROLL` antithetic pairs (`eggroll.py`)
    at `1.00x` assignment FLOPs: routes half-batch `B_+` against `C + sigma(t) (r * E)` and
    half-batch `B_-` against `C - sigma(t) (r * E)` using unit-RMS signed Hadamard frames `E`
    scaled by local cluster RMS spread `r_j`, smoothing non-differentiable Voronoi ridges
    with `O(sigma^2)` antithetic bias cancellation and strict convex-hull EMA preservation.
- `N-Track-5` (`speedrun_strat_wide` Spatial-Quantile Stratified Batching & Wide-Step Schedule):
    Translates `difficulty_order` (`train/speedrun/run.py:4116-4129`) and `wide_rows`
    (`train/speedrun/run.py:148-151`): `O(n*d)` 6-bit octant + radial-energy sorting into
    `B` equal strata so every mini-batch samples 1 vector per stratum (collapsing Binomial
    batch-count variance), optionally paired with late `2x` batch doubling (`wide_from=0.25`)
    and `e-means` step-law coupling (`alpha_0 * 2^0.9003`) along row-progress cosine decay.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .emeans import EpochStat, _separate_collapsed_splits
from .emeans_pp import _nearest_other_centroid_sq_dists
from .kernels import assign_points, prepare_represented_centroids
from .recipe import (
    ALPHA_LAW,
    ALPHA_MIN_FACTOR,
    CHECK_UNUSED_EVERY,
    DEFAULT_INIT_USAGE,
    SPLIT_EPSILON,
    EMeansRecipe,
    build_recipe,
    epoch_seed,
    scheduled_alpha,
)
from .seeding import SeedingResult, lightweight_coreset, seed_emeans_plusplus, voronoi_polish_and_priors


# Six quintic Newton-Schulz coefficients from `qlabs-eng/noise` (`train/val/run.py:2181`, `speedrun/run.py:3228`)
ANVIL_MAPS: tuple[tuple[float, float, float], ...] = (
    (3.923798038567, -6.095026865488, 3.905234618423),
    (3.278126713798, -3.328923386476, 0.989127286973),
    (3.505298394150, -5.137358782410, 1.968325560615),
    (2.815058591845, -3.685181239622, 1.417196497642),
    (2.245503932403, -2.443826979899, 0.963091710461),
    (2.256537145403, -2.166840097229, 0.929501253245),
)

_HADAMARD_CACHE: dict[tuple[int, torch.device], torch.Tensor] = {}


@dataclass(frozen=True)
class NoiseTrackHypothesis:
    """Pre-registered specification of a forward-only clustering hypothesis from `qlabs-eng/noise`."""

    id: str
    name: str
    noise_source: str
    mechanism: str
    equation: str
    complexity_overhead: str
    predicted_win_regime: str
    predicted_fail_regime: str
    falsification_criterion: str


NOISE_HYPOTHESES: dict[str, NoiseTrackHypothesis] = {
    "N-Track-1": NoiseTrackHypothesis(
        id="N-Track-1",
        name="Forward-Verified Signed-Hadamard Centroid Transplant (head_edit)",
        noise_source="train/val/run.py :: module head_edit + hadamard() + coordinate_differences()",
        mechanism=(
            "Split periodic batches into disjoint Proposal half P and Confirmation half Q; "
            "measure exact top-3 Voronoi partition-swap dormancy (shielded by Huygens rare-mode "
            "cost), probe M orthogonal signed-Hadamard splits on high-SSE donors on P, and "
            "commit transplants iff held-back Q confirms strict forward SSE reduction."
        ),
        equation=(
            "p_{i,m} = <x_i - mu_s, h_m>; delta_m^+- = mean_{+- p > 0}(p); "
            "Delta SSE_P(m) = sum_{x in P} [min(d_rem^2(x), d_+^2(x), d_-^2(x)) - d_1^2(x)]; "
            "keep iff Delta SSE_P(m*) < 0 and Delta SSE_Q(m*) < 0"
        ),
        complexity_overhead="< 3% wall-clock (runs every eff_transplant_every <= 8 steps via O(B*K*D) top-3 + 1D half-space Hadamard probes)",
        predicted_win_regime=(
            "Multimodal & imbalanced GMMs (zipf_gmm, low_nk32_gmm, multiscale_rare_outliers) and "
            "random-init e-means where live twin centroids share dense modes (N_d >= 0.1 mean(N)) "
            "while merged donors straddle separated modes."
        ),
        predicted_fail_regime=(
            "Isotropic continuous clouds (isotropic_gaussian) where Voronoi hole creation roughly "
            "cancels donor split gain so Q-confirmation rejects non-dead swaps."
        ),
        falsification_criterion=(
            "Falsified if paired 15-seed test MSE fails to improve significantly (Wilcoxon p >= 0.05 "
            "or mean delta >= 0) on multimodal GMM benchmarks or degrades rare-mode preservation."
        ),
    ),
    "N-Track-2": NoiseTrackHypothesis(
        id="N-Track-2",
        name="In-Run Held-Out Parabolic Trajectory Merge (quad_merge)",
        noise_source="train/speedrun/run.py :: quad_merge() (commit aa67d44) + train/val/run.py :: soup_parts()",
        mechanism=(
            "Hold out a small disjoint internal subset X_hold strictly from x_train (zero val/test "
            "access), track split-consistent tail EMA anchors over the final 70% of updates, fit "
            "closed-form 3-point parabolic vertices per usage-quantile part on exact hard-Voronoi "
            "loss over X_hold + global QUAD_SHRINKS, and fold X_hold statistics into closing centroids."
        ),
        equation=(
            "a_p* = clip(C0 - Delta * (L_+ - L_-) / (2 * max(L_- + L_+ - 2*L_0, 1e-30)), 0, 1); "
            "c_j = (1 - s* a_{p(j)}*) c_j^{close,all} + s* a_{p(j)}* T_{*,j}"
        ),
        complexity_overhead="~32 forward assignments on <= 4096 held-out train rows (~2-4 ms on GPU)",
        predicted_win_regime=(
            "Low-n/k regimes (low_nk32_gmm, cifar10_r18_512, zipf_gmm) on out-of-sample test MSE, "
            "capturing true hard-Voronoi boundary shifts across trajectory states."
        ),
        predicted_fail_regime=(
            "Pure in-sample train MSE on high-n/k clouds (where exact sample mean closing_mean is "
            "already optimal on fixed training assignments)."
        ),
        falsification_criterion=(
            "Falsified if paired 15-seed out-of-sample test MSE fails to beat closing_mean "
            "(Wilcoxon p >= 0.05 or mean delta >= 0) in low-n/k regimes (n/k < 100)."
        ),
    ),
    "N-Track-3": NoiseTrackHypothesis(
        id="N-Track-3",
        name="ANVIL-II Dual-Rail Gram-Newton-Schulz Polar Centroid Update (anvil_polar)",
        noise_source="train/val/run.py :: class Anvil / BatchedAnvil / gram_newton_schulz()",
        mechanism=(
            "Maintain ANVIL-II fast (0.85) and slow (0.98) momentum rails on the (k, d) EMA "
            "displacement matrix, apply 6-step quintic Gram-Newton-Schulz polar orthogonalization "
            "on min(k, d) + per-lane RMS normalization norm-matched to ||Delta C_ema||_F, and "
            "anneal smoothly to 0 so late updates preserve e-means Appendix G convex-hull optimality."
        ),
        equation=(
            "P = GramNS_6((1 - beta_t) Delta C + beta_t (w F + (1 - w) S)); "
            "c_j <- c_j + (1 - lambda(t)) Delta c_j^{ema} + lambda(t) U_j"
        ),
        complexity_overhead="O(k * d * min(k, d)) per active step (< 0.1 ms/step for min(k, d) <= 512)",
        predicted_win_regime=(
            "Early-epoch strongly anisotropic spectra (stretched_a2_t2) where polar singular-value "
            "equalization spreads centroids into secondary variance directions faster than raw EMA."
        ),
        predicted_fail_regime=(
            "Converged or heavy-tailed multimodal densities where isotropic singular-value whitening "
            "conflicts with Zador's p_j ~ rho(c)^{d/(d+2)} quantization density allocation."
        ),
        falsification_criterion=(
            "Falsified if paired 15-seed test MSE shows no statistically significant improvement "
            "(Wilcoxon p >= 0.05) on anisotropic/embedding benchmarks or degrades MSE."
        ),
    ),
    "N-Track-4": NoiseTrackHypothesis(
        id="N-Track-4",
        name="Zero-Overhead Split-Batch Antithetic Hadamard Voronoi Jitter (dust_antithetic)",
        noise_source="train/val/dust.py :: direct_errors() + archive/10M/train/val/eggroll.py + hadamard()",
        mechanism=(
            "Route half-batch B_+ against C + sigma(t)(r * E) and half-batch B_- against "
            "C - sigma(t)(r * E) at 1.00x FLOPs using unit-coordinate-RMS signed Hadamard frames E "
            "scaled by cluster RMS spread r_j, smoothing piecewise-constant Voronoi ridges with "
            "O(sigma^2) antithetic cancellation while keeping (N_j, S_j) strictly in the convex hull."
        ),
        equation=(
            "C^+- = C +- sigma(t) sqrt(E_j / (N_j d)) E_j; "
            "n_j = n_j(B_+, C^+) + n_j(B_-, C^-), s_j = s_j(B_+, C^+) + s_j(B_-, C^-)"
        ),
        complexity_overhead="1.00x assignment FLOPs ((B/2)*K*D + (B/2)*K*D = B*K*D)",
        predicted_win_regime=(
            "Rugged real feature manifolds (cifar10_r18_512, cifar10_patches_128) and random-init "
            "e-means where shallow Voronoi ridge ripples trap deterministic hard routing."
        ),
        predicted_fail_regime=(
            "Cleanly separated warm-started clusters where Voronoi boundaries lie in low-density "
            "gaps so boundary jitter occasionally misroutes interior tail points."
        ),
        falsification_criterion=(
            "Falsified if paired 15-seed test MSE fails to improve significantly (Wilcoxon p >= 0.05) "
            "across real deep embedding and random-init benchmarks."
        ),
    ),
    "N-Track-5": NoiseTrackHypothesis(
        id="N-Track-5",
        name="Spatial-Quantile Stratified Batching & Late Wide-Step Schedule (speedrun_strat_wide)",
        noise_source="train/speedrun/run.py :: difficulty_order() + wide_rows / wide_lr_scale",
        mechanism=(
            "Sort training vectors once in O(n*d) by 6-bit random-projection octant + radial "
            "surprisal into B strata so every mini-batch draws 1 sample per stratum (suppressing "
            "Binomial count variance), optionally doubling batch size to 2*B after wide_from=0.25 "
            "with coupled e-means step scaling alpha_0 * 2^0.9003."
        ),
        equation=(
            "strata = argsort(key).reshape(B_cur, S); batch_s = {strata[b, perm_b[s]]}_{b=1..B_cur}; "
            "B(p >= wide_from) = 2 B, alpha_0^{wide} = min(1, 2^{0.9003} alpha_0)"
        ),
        complexity_overhead="< 0.2 ms one-time O(n*d) sort; wide-step reduces total step launches by ~37.5%",
        predicted_win_regime=(
            "Imbalanced/multiscale distributions (zipf_gmm, multiscale_rare_outliers, cifar10) via "
            "variance-reduced batch counts, plus faster wall-clock time under wide-step doubling."
        ),
        predicted_fail_regime=(
            "Isotropic clouds where i.i.d. batches already have uniform cluster load, or random-init "
            "at T=2 if fewer total steps reduce dead-cluster revival opportunities."
        ),
        falsification_criterion=(
            "Falsified if stratified batching fails to improve paired MSE (Wilcoxon p >= 0.05) and "
            "wide-step fails to reduce wall-clock time at non-inferior MSE (<= +0.05%)."
        ),
    ),
}


@dataclass
class HeadEditDiagnostics:
    """Summary counters for `N-Track-1` (`head_edit`) forward-verified transplants."""

    checks_run: int = 0
    proposals_tried: int = 0
    p_improved: int = 0
    q_confirmed: int = 0


@dataclass
class QuadMergeDiagnostics:
    """Summary diagnostics for `N-Track-2` (`quad_merge`) held-out parabolic trajectory merge."""

    holdout_rows: int
    selected_target: str
    selected_shrink: float
    holdout_mse_plain: float
    holdout_mse_merged: float
    mean_blend_weight: float
    part_vertices: tuple[float, ...]


# ---------------------------------------------------------------------------
# Mathematical Building Blocks for `N-Track-1` through `N-Track-5`
# ---------------------------------------------------------------------------


def _next_power_of_two_int(d: int) -> int:
    p = 1
    while p < d:
        p <<= 1
    return p


def get_sylvester_hadamard(order: int, device: torch.device) -> torch.Tensor:
    """Construct or fetch cached Sylvester Hadamard matrix H_order in float32 (`+-1.0` entries)."""
    if order < 1 or (order & (order - 1)) != 0:
        raise ValueError(f"Sylvester Hadamard order must be a positive power of two, got {order}")
    key = (order, device)
    cached = _HADAMARD_CACHE.get(key)
    if cached is not None:
        return cached
    h = torch.ones((1, 1), dtype=torch.float32, device=device)
    while h.shape[0] < order:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    _HADAMARD_CACHE[key] = h.contiguous()
    return _HADAMARD_CACHE[key]


@torch.no_grad()
def sample_signed_hadamard_rows(
    num_rows: int,
    d: int,
    generator: torch.Generator,
    device: torch.device,
    unit_l2: bool = False,
) -> torch.Tensor:
    """Draw `num_rows` random-signed Walsh-Hadamard probe vectors in R^d for ANY `d >= 1`.

    When `unit_l2=False`, entries are in `{-1.0, +1.0}` so every row has exact unit coordinate
    RMS (`row.square().mean() == 1.0`). When `unit_l2=True`, rows are scaled by `1 / sqrt(d)`
    so every row has exact unit Euclidean norm (`||row||_2 == 1.0`). Within each block of
    `2^ceil(log2(d))` rows, rows are mutually orthogonal when `d` is a power of two.
    """
    if num_rows <= 0:
        return torch.empty((0, d), dtype=torch.float32, device=device)
    if d < 1:
        raise ValueError(f"Dimension d must be >= 1, got {d}")

    order = _next_power_of_two_int(d)
    h_mat = get_sylvester_hadamard(order, device=device)
    n_blocks = (num_rows + order - 1) // order
    rand_u = torch.rand((n_blocks, order), generator=generator, device=device, dtype=torch.float32)
    perms = torch.argsort(rand_u, dim=1)
    signs = (
        torch.randint(0, 2, (n_blocks, 1, order), generator=generator, device=device, dtype=torch.int64)
        .to(dtype=torch.float32)
        .mul_(2.0)
        .sub_(1.0)
    )
    out = (h_mat[perms] * signs).reshape(n_blocks * order, order)[:num_rows, :d]
    if unit_l2:
        out = out * (1.0 / math.sqrt(float(d)))
    return out.contiguous()


@torch.no_grad()
def top3_represented_assign_centered(
    x: torch.Tensor,
    centroids: torch.Tensor,
    compute_dtype: torch.dtype,
    centroid_chunk_size: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute exact translation-invariant top-3 represented squared distances and distinct indices.

    Applies Appendix I.1 represented conversion FIRST (`x.to(compute_dtype)`, `centroids.to(compute_dtype)`),
    then subtracts the batch mean `origin` in FP32 so `||(x_hat - origin) - (c_hat - origin)||^2` is
    both represented-precision consistent AND immune to catastrophic FP32 cancellation at `+1e4` offsets.

    Returns `(top_dists, top_indices, origin)` where:
      - `top_dists`: `(B, 3)` float32 (`+inf` in padded columns if `K < 3`),
      - `top_indices`: `(B, 3)` int64 (`-1` in padded columns if `K < 3`, strictly distinct per row),
      - `origin`: `(1, D)` float32 coordinate origin used for consistent probe evaluation.
    """
    bsz, d = x.shape
    k = centroids.shape[0]
    device = x.device
    if k < 1:
        raise ValueError("centroids must have K >= 1")
    if bsz == 0:
        return (
            torch.empty((0, 3), dtype=torch.float32, device=device),
            torch.empty((0, 3), dtype=torch.int64, device=device),
            torch.zeros((1, d), dtype=torch.float32, device=device),
        )

    if compute_dtype == torch.float64:
        x64 = x.double()
        c64 = centroids.double()
        origin64 = x64.mean(dim=0, keepdim=True)
        xc64 = (x64 - origin64).contiguous()
        cc64 = (c64 - origin64).contiguous()
        x_norm64 = xc64.square().sum(dim=1)
        c_norm64 = cc64.square().sum(dim=1)
        m_keep = min(3, k)
        best_scores64 = torch.full((bsz, m_keep), float("inf"), dtype=torch.float64, device=device)
        best_idx = torch.full((bsz, m_keep), -1, dtype=torch.int64, device=device)
        for k0 in range(0, k, centroid_chunk_size):
            k1 = min(k0 + centroid_chunk_size, k)
            scores64 = torch.addmm(c_norm64[k0:k1][None, :], xc64, cc64[k0:k1].t(), alpha=-2.0)
            m_blk = min(m_keep, k1 - k0)
            blk_sc, blk_id_local = torch.topk(scores64, k=m_blk, dim=1, largest=False, sorted=True)
            blk_id = blk_id_local + k0
            if k0 == 0 and m_blk == m_keep:
                best_scores64 = blk_sc
                best_idx = blk_id
            else:
                comb_sc = torch.cat([best_scores64, blk_sc], dim=1)
                comb_id = torch.cat([best_idx, blk_id], dim=1)
                mer_sc, mer_pos = torch.topk(comb_sc, k=m_keep, dim=1, largest=False, sorted=True)
                best_scores64 = mer_sc
                best_idx = torch.gather(comb_id, 1, mer_pos)
        top_dists = (best_scores64 + x_norm64[:, None]).clamp_min_(0.0).float()
        if m_keep < 3:
            pad_cols = 3 - m_keep
            top_dists = torch.cat(
                [top_dists, torch.full((bsz, pad_cols), float("inf"), dtype=torch.float32, device=device)],
                dim=1,
            )
            best_idx = torch.cat(
                [best_idx, torch.full((bsz, pad_cols), -1, dtype=torch.int64, device=device)],
                dim=1,
            )
        return top_dists, best_idx, origin64.float()

    rep_dtype = (
        compute_dtype
        if compute_dtype in (torch.float32, torch.bfloat16, torch.float16)
        else torch.float32
    )
    x_rep_f = x.to(rep_dtype).float()
    c_rep_f = centroids.to(rep_dtype).float()

    origin = x_rep_f.mean(dim=0, keepdim=True)
    xc = (x_rep_f - origin).contiguous()
    cc = (c_rep_f - origin).contiguous()

    x_norm = xc.square().sum(dim=1)  # (B,)
    c_norm = cc.square().sum(dim=1)  # (K,)

    m_keep = min(3, k)
    best_scores = torch.full((bsz, m_keep), float("inf"), dtype=torch.float32, device=device)
    best_idx = torch.full((bsz, m_keep), -1, dtype=torch.int64, device=device)

    for k0 in range(0, k, centroid_chunk_size):
        k1 = min(k0 + centroid_chunk_size, k)
        c_blk = cc[k0:k1]
        cn_blk = c_norm[k0:k1]
        scores = torch.addmm(cn_blk[None, :], xc, c_blk.t(), alpha=-2.0)
        m_blk = min(m_keep, k1 - k0)
        blk_sc, blk_id_local = torch.topk(scores, k=m_blk, dim=1, largest=False, sorted=True)
        blk_id = blk_id_local + k0

        if k0 == 0 and m_blk == m_keep:
            best_scores = blk_sc
            best_idx = blk_id
        else:
            comb_sc = torch.cat([best_scores, blk_sc], dim=1)
            comb_id = torch.cat([best_idx, blk_id], dim=1)
            mer_sc, mer_pos = torch.topk(comb_sc, k=m_keep, dim=1, largest=False, sorted=True)
            best_scores = mer_sc
            best_idx = torch.gather(comb_id, 1, mer_pos)

    top_dists = (best_scores + x_norm[:, None]).clamp_min_(0.0)
    if m_keep < 3:
        pad_cols = 3 - m_keep
        top_dists = torch.cat(
            [top_dists, torch.full((bsz, pad_cols), float("inf"), dtype=torch.float32, device=device)],
            dim=1,
        )
        best_idx = torch.cat(
            [best_idx, torch.full((bsz, pad_cols), -1, dtype=torch.int64, device=device)],
            dim=1,
        )

    return top_dists, best_idx, origin


def exact_exclusion_sq_dists(
    top_dists: torch.Tensor,
    top_idx: torch.Tensor,
    rem_a: int,
    rem_b: int,
) -> torch.Tensor:
    """Exact `min_{j not in {rem_a, rem_b}} ||x_i - c_j||^2` from top-3 `(top_dists, top_idx)`."""
    a1 = top_idx[:, 0]
    a2 = top_idx[:, 1]
    hit1 = (a1 == rem_a) | (a1 == rem_b)
    hit2 = (a2 == rem_a) | (a2 == rem_b)
    return torch.where(
        ~hit1,
        top_dists[:, 0],
        torch.where(~hit2, top_dists[:, 1], top_dists[:, 2]),
    )


@torch.no_grad()
def run_head_edit_step(
    x_batch: torch.Tensor,
    centroids: torch.Tensor,
    usage: torch.Tensor,
    distortion: torch.Tensor | None,
    survival_mass: torch.Tensor | None,
    slow_centroids: torch.Tensor | None,
    slow_usage: torch.Tensor | None,
    tail_banks: list[torch.Tensor],
    last_edit_step: torch.Tensor,
    step: int,
    compute_dtype: torch.dtype,
    unused_threshold: float,
    use_huygens_protection: bool,
    gamma_protect: float,
    num_probes: int,
    max_pairs: int,
    protect_steps: int,
    generator: torch.Generator,
    diag: HeadEditDiagnostics,
) -> int:
    """Execute one `N-Track-1` (`head_edit`) P/Q forward-verified Hadamard transplant check."""
    bsz, d = x_batch.shape
    k = centroids.shape[0]
    if k < 3 or bsz < 16 or max_pairs < 1 or num_probes < 1:
        return 0

    half = bsz // 2
    if half < 8 or (bsz - half) < 8:
        return 0

    diag.checks_run += 1
    device = x_batch.device
    rep_dtype = (
        compute_dtype
        if compute_dtype in (torch.float32, torch.bfloat16, torch.float16, torch.float64)
        else torch.float32
    )
    all_ids = torch.arange(k, device=device, dtype=torch.int64)
    scale_full = float(bsz) / float(half)
    sse_p = torch.empty((k,), dtype=torch.float32, device=device)
    dormancy_p = torch.empty((k,), dtype=torch.float32, device=device)

    def _compute_partition_state() -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[float],
        list[float],
        list[float],
        list[float],
        list[float],
        list[float],
    ]:
        td, ti, orig = top3_represented_assign_centered(x_batch, centroids, compute_dtype)
        d1p = td[:half, 0]
        d2p = td[:half, 1]
        a1p = ti[:half, 0]
        a1q = ti[half:, 0]

        cp = torch.bincount(a1p, minlength=k).float()
        cq = torch.bincount(a1q, minlength=k).float()
        sse_p.zero_()
        sse_p.index_add_(0, a1p, d1p)

        # Exact forward removal increase on Proposal half P, scaled to full-batch units
        dormancy_p.zero_()
        dormancy_p.index_add_(0, a1p, (d2p - d1p).clamp_min(0.0))
        b_rem = dormancy_p * scale_full

        # Global nearest-neighbor Huygens lower bound shields genuine isolated rare modes
        # (including the surviving twin after a duplicate rare-mode twin is transplanted!)
        d_nn2 = _nearest_other_centroid_sq_dists(centroids, all_ids)
        thresh = usage.mean() * max(unused_threshold, 0.1)
        if survival_mass is not None:
            eff_mass = torch.maximum(usage, torch.minimum(survival_mass, thresh))
        else:
            eff_mass = usage
        h_cost = eff_mass * d_nn2.clamp_max(1e30)
        e_rem = torch.maximum(b_rem, h_cost)

        f_mask = (step - last_edit_step) >= protect_steps
        if use_huygens_protection and distortion is not None and gamma_protect > 0.0:
            prot_cutoff = gamma_protect * distortion.mean().clamp_min(1e-30)
            rare_shield = (usage < thresh) & (h_cost >= prot_cutoff)
            f_mask = f_mask & (~rare_shield)

        don_elig = f_mask & (cp >= 4.0) & (cq >= 2.0) & (sse_p > 1e-12)
        rec_elig = f_mask

        packed = torch.stack(
            [cp, cq, sse_p, e_rem, rec_elig.float(), don_elig.float()], dim=0
        ).tolist()
        return (
            td,
            ti,
            orig,
            d1p,
            a1p,
            packed[0],
            packed[1],
            packed[2],
            packed[3],
            packed[4],
            packed[5],
        )

    (
        top_d,
        top_i,
        origin,
        d1_p,
        a1_p,
        cp_list,
        cq_list,
        ssep_list,
        eff_rem_list,
        rec_elig_list,
        don_elig_list,
    ) = _compute_partition_state()

    don_candidates = [j for j, flg in enumerate(don_elig_list) if flg > 0.5]
    n_rec_avail = sum(1 for flg in rec_elig_list if flg > 0.5)
    if not don_candidates or n_rec_avail < 2:
        return 0

    n_try = min(max_pairs, len(don_candidates), n_rec_avail // 2)
    if n_try < 1:
        return 0

    don_candidates.sort(key=lambda j: ssep_list[j], reverse=True)
    don_order = don_candidates[:n_try]
    rec_order = sorted(
        (j for j, flg in enumerate(rec_elig_list) if flg > 0.5),
        key=lambda j: eff_rem_list[j],
    )

    if rep_dtype == torch.float64:
        origin64 = x_batch.double().mean(dim=0, keepdim=True)
        xc_all = (x_batch.double() - origin64).float().contiguous()
    else:
        origin64 = None
        xc_all = (x_batch.to(rep_dtype).float() - origin).contiguous()
    x_norm_all = xc_all.square().sum(dim=1)
    xc_p = xc_all[:half]
    xn_p = x_norm_all[:half]
    xc_q = xc_all[half:]
    xn_q = x_norm_all[half:]

    used_slots: set[int] = set()
    kept_edits = 0

    for s_idx in don_order:
        if s_idx in used_slots:
            continue
        if cp_list[s_idx] < 4.0 or cq_list[s_idx] < 2.0 or ssep_list[s_idx] <= 1e-12:
            continue
        d_idx = -1
        for cand_d in rec_order:
            if cand_d != s_idx and cand_d not in used_slots and rec_elig_list[cand_d] > 0.5:
                d_idx = cand_d
                break
        if d_idx < 0:
            continue

        # Require recipient effective removal cost to be strictly smaller than 45% of donor SSE on P
        if eff_rem_list[d_idx] >= 0.45 * ssep_list[s_idx]:
            continue

        s_mask = a1_p == s_idx
        xc_s = xc_p[s_mask]
        n_ps = int(xc_s.shape[0])
        if n_ps < 4:
            continue

        diag.proposals_tried += 1
        mu_sc = xc_s.mean(dim=0, keepdim=True)  # (1, D), centered by `origin`
        r_s = xc_s - mu_sc  # (n_ps, D)

        # Build unit probe frame: orthogonal signed-Hadamard directions + Hadamard-seeded forward chords on P
        dirs_had = sample_signed_hadamard_rows(
            num_probes, d, generator=generator, device=device, unit_l2=True
        )
        if num_probes >= 3 and d >= 2:
            v0 = dirs_had[-2:-1].t()
            for _ in range(2):
                v1 = r_s.t() @ (r_s @ v0)
                v1n = torch.linalg.vector_norm(v1)
                v0 = torch.where(v1n > 1e-16, v1 / v1n.clamp_min(1e-16), v0)
            chord_pca = v0.t()
            far_idx = torch.argmax(r_s.square().sum(dim=1))
            v_far = r_s[far_idx : far_idx + 1]
            v_farn = torch.linalg.vector_norm(v_far)
            chord_far = torch.where(
                v_farn > 1e-16, v_far / v_farn.clamp_min(1e-16), dirs_had[-1:]
            )
            dirs = torch.cat([dirs_had[:-2], chord_pca, chord_far], dim=0)
        elif num_probes == 2 and d >= 2:
            v0 = dirs_had[-1:].t()
            for _ in range(2):
                v1 = r_s.t() @ (r_s @ v0)
                v1n = torch.linalg.vector_norm(v1)
                v0 = torch.where(v1n > 1e-16, v1 / v1n.clamp_min(1e-16), v0)
            dirs = torch.cat([dirs_had[:1], v0.t()], dim=0)
        else:
            dirs = dirs_had

        # Exact 1D conditional half-space projections along each probe direction (O(R/sqrt(d)) in any d,
        # zero orthogonal step overshoot, exact +/- Delta on bimodal aligned axes!)
        proj = r_s @ dirs.t()  # (n_ps, M)
        pos_mask = proj > 0.0
        neg_mask = ~pos_mask
        n_pos = pos_mask.float().sum(dim=0)
        n_neg = neg_mask.float().sum(dim=0)
        sum_pos = torch.where(pos_mask, proj, torch.zeros_like(proj)).sum(dim=0)
        sum_neg = torch.where(neg_mask, proj, torch.zeros_like(proj)).sum(dim=0)
        rms_1d = (r_s.square().sum() / float(max(1, n_ps * d))).clamp_min(1e-24).sqrt()
        delta_plus = torch.where(n_pos > 0.0, sum_pos / n_pos.clamp_min(1.0), rms_1d)
        delta_minus = torch.where(n_neg > 0.0, sum_neg / n_neg.clamp_min(1.0), -rms_1d)

        if origin64 is not None:
            base_don64 = origin64 + mu_sc.double()
            cand_minus64 = base_don64 + (delta_minus[:, None] * dirs).double()
            cand_plus64 = base_don64 + (delta_plus[:, None] * dirs).double()
            cm_rep_c = (cand_minus64 - origin64).float().contiguous()
            cp_rep_c = (cand_plus64 - origin64).float().contiguous()
            cand_minus = cand_minus64.to(centroids.dtype)
            cand_plus = cand_plus64.to(centroids.dtype)
        else:
            base_don = origin + mu_sc
            cand_minus_f = base_don + delta_minus[:, None] * dirs
            cand_plus_f = base_don + delta_plus[:, None] * dirs
            cm_rep_c = (cand_minus_f.to(rep_dtype).float() - origin).contiguous()
            cp_rep_c = (cand_plus_f.to(rep_dtype).float() - origin).contiguous()
            cand_minus = cand_minus_f.to(centroids.dtype)
            cand_plus = cand_plus_f.to(centroids.dtype)
        cm_norm = cm_rep_c.square().sum(dim=1)
        cp_norm = cp_rep_c.square().sum(dim=1)

        d_rem_all = exact_exclusion_sq_dists(top_d, top_i, d_idx, s_idx)
        d_rem_p = d_rem_all[:half]

        # Evaluate all M candidate splits simultaneously on Proposal half P: (B_P, M)
        dist_m_p = (torch.addmm(cm_norm[None, :], xc_p, cm_rep_c.t(), alpha=-2.0) + xn_p[:, None]).clamp_min_(0.0)
        dist_p_p = (torch.addmm(cp_norm[None, :], xc_p, cp_rep_c.t(), alpha=-2.0) + xn_p[:, None]).clamp_min_(0.0)
        best_pair_p = torch.minimum(dist_m_p, dist_p_p)
        new_best_p = torch.minimum(d_rem_p[:, None], best_pair_p)
        delta_sse_p = (new_best_p - d1_p[:, None]).sum(dim=0)  # (M,)

        best_dp_t, best_m_t = torch.min(delta_sse_p, dim=0)
        best_pair_vals = torch.stack([best_dp_t, best_m_t.float()]).tolist()
        best_dp = float(best_pair_vals[0])
        best_m = int(best_pair_vals[1])
        if not (best_dp < -1e-6 * max(1.0, ssep_list[s_idx])):
            continue

        diag.p_improved += 1

        # Confirm winning probe `best_m` on disjoint Confirmation half Q
        d_rem_q = d_rem_all[half:]
        d1_q = top_d[half:, 0]
        cm_win = cm_rep_c[best_m : best_m + 1]
        cp_win = cp_rep_c[best_m : best_m + 1]
        dist_m_q = (torch.addmm(cm_norm[best_m : best_m + 1][None, :], xc_q, cm_win.t(), alpha=-2.0).squeeze(1) + xn_q).clamp_min_(0.0)
        dist_p_q = (torch.addmm(cp_norm[best_m : best_m + 1][None, :], xc_q, cp_win.t(), alpha=-2.0).squeeze(1) + xn_q).clamp_min_(0.0)
        new_best_q = torch.minimum(d_rem_q, torch.minimum(dist_m_q, dist_p_q))
        q_vals = torch.stack([(new_best_q - d1_q).sum(), d1_q.sum()]).tolist()
        delta_sse_q = float(q_vals[0])
        mean_sse_q = float(q_vals[1]) / float(max(1, k))

        if not (delta_sse_q < -1e-6 * max(1.0, mean_sse_q)):
            continue

        # Both P and Q strictly improved -> commit transplant!
        diag.q_confirmed += 1
        new_s = cand_minus[best_m : best_m + 1].clone()
        new_d = cand_plus[best_m : best_m + 1].clone()
        _separate_collapsed_splits(new_s, new_d, compute_dtype)

        centroids[s_idx] = new_s[0]
        centroids[d_idx] = new_d[0]

        half_u = usage[s_idx] * 0.5
        usage[s_idx] = half_u
        usage[d_idx] = half_u

        if distortion is not None:
            half_e = distortion[s_idx] * 0.5
            distortion[s_idx] = half_e
            distortion[d_idx] = half_e
        if survival_mass is not None:
            half_sm = survival_mass[s_idx] * 0.5
            survival_mass[s_idx] = half_sm
            survival_mass[d_idx] = half_sm
        if slow_centroids is not None and slow_usage is not None:
            slow_centroids[s_idx] = new_s[0]
            slow_centroids[d_idx] = new_d[0]
            shalf = slow_usage[s_idx] * 0.5
            slow_usage[s_idx] = shalf
            slow_usage[d_idx] = shalf
        for tb in tail_banks:
            tb[s_idx] = new_s[0]
            tb[d_idx] = new_d[0]

        last_edit_step[s_idx] = step
        last_edit_step[d_idx] = step
        used_slots.add(s_idx)
        used_slots.add(d_idx)
        kept_edits += 1

        # Full state refresh (including `d2_p`, `d_nn2`, `huygens_cost`, `rare_shield`, `eff_rem_cost`,
        # and `rec_order`) so surviving rare-mode twins become immediately Huygens-shielded!
        if kept_edits < n_try and len(used_slots) + 2 <= k:
            (
                top_d,
                top_i,
                _,
                d1_p,
                a1_p,
                cp_list,
                cq_list,
                ssep_list,
                eff_rem_list,
                rec_elig_list,
                _,
            ) = _compute_partition_state()
            rec_order = sorted(
                (j for j, flg in enumerate(rec_elig_list) if flg > 0.5),
                key=lambda j: eff_rem_list[j],
            )

    return kept_edits


@torch.no_grad()
def gram_newton_schulz_polar(mat: torch.Tensor, margin: float = 1.1) -> torch.Tensor:
    """6-step quintic Newton-Schulz polar orthogonalization (`ANVIL_MAPS`) on a 2D `(K, D)` matrix.

    Zero host-device `.item()` syncs and fused `addmm` polynomial steps (3 cuBLAS launches/step,
    zero elementwise launches inside the loop): operates on the smaller Gram dimension `min(K, D)`,
    using BF16 Tensor Cores on CUDA when `min(K, D) >= 16` and FP32 otherwise.
    """
    k, d = mat.shape
    if k == 0 or d == 0:
        return mat.clone()

    tall = k > d
    xf = mat.float().t().contiguous() if tall else mat.float().contiguous()  # (r, c) with r <= c
    frob = torch.linalg.vector_norm(xf)  # 0D GPU/CPU tensor (no .item() sync!)
    valid = torch.isfinite(frob) & (frob > 1e-20)
    denom = torch.where(valid, margin * frob + 1e-12, torch.ones_like(frob))
    x = xf / denom

    r = x.shape[0]
    if x.is_cuda and r >= 16:
        xb = x.bfloat16()
        for a, b, c in ANVIL_MAPS:
            s = xb @ xb.t()
            poly = torch.addmm(s, s, s, beta=b, alpha=c)
            xb = torch.addmm(xb, poly, xb, beta=a, alpha=1.0)
        out = xb.float()
    else:
        for a, b, c in ANVIL_MAPS:
            s = x @ x.t()
            poly = torch.addmm(s, s, s, beta=b, alpha=c)
            x = torch.addmm(x, poly, x, beta=a, alpha=1.0)
        out = x

    out = torch.where(valid, out, torch.zeros_like(out))
    return out.t().contiguous() if tall else out.contiguous()


@torch.no_grad()
def anvil_polar_displacement(
    delta_ema: torch.Tensor,
    touched: torch.Tensor,
    fast_rail: torch.Tensor,
    slow_rail: torch.Tensor,
    lane_energy: torch.Tensor,
    progress: float,
    blend0: float,
    stop_frac: float = 0.60,
    fast_beta: float = 0.85,
    slow_beta: float = 0.98,
    lane_beta: float = 0.90,
    fast_weight: float = 0.4385,
    polar_margin: float = 1.1,
) -> torch.Tensor:
    """Compute norm-matched annealed `ANVIL-II` polar centroid displacement (`N-Track-3`, zero `.item()` syncs)."""
    if blend0 <= 0.0 or progress >= stop_frac:
        return delta_ema

    t_mask_f = touched[:, None].float()  # (K, 1)
    fast_rail.lerp_(delta_ema, (1.0 - fast_beta) * t_mask_f)
    slow_rail.lerp_(delta_ema, (1.0 - slow_beta) * t_mask_f)

    rel_p = max(0.0, min(1.0, progress / max(stop_frac, 1e-6)))
    beta_t = 0.85 + 0.08 * math.sin(math.pi * rel_p)
    rail_mix = torch.lerp(slow_rail, fast_rail, fast_weight)
    m_active = torch.lerp(delta_ema, rail_mix, beta_t) * t_mask_f

    p_orth = gram_newton_schulz_polar(m_active, margin=polar_margin) * t_mask_f

    # Per-lane RMS energy normalization (`train/val/run.py:2223-2225`)
    row_sq_mean = p_orth.square().mean(dim=1, keepdim=True)
    lane_energy.lerp_(row_sq_mean, (1.0 - lane_beta) * t_mask_f)
    gained = p_orth * lane_energy.clamp_min(1e-10).rsqrt()

    frob_ema = torch.linalg.vector_norm(delta_ema)
    gained_norm = torch.linalg.vector_norm(gained)
    active_step = (frob_ema > 1e-20) & (gained_norm > 1e-20)
    scale_frob = torch.where(active_step, frob_ema / gained_norm.clamp_min(1e-20), torch.zeros_like(frob_ema))
    u_anvil = gained * scale_frob

    # Row-wise safety trust-region cap (`<= 2.0 * ||delta_ema_j||_2`)
    ema_row_norm = torch.linalg.vector_norm(delta_ema, dim=1, keepdim=True)
    u_row_norm = torch.linalg.vector_norm(u_anvil, dim=1, keepdim=True).clamp_min(1e-12)
    u_anvil.mul_(torch.clamp((2.0 * ema_row_norm) / u_row_norm, max=1.0))

    lam = float(blend0 * 0.5 * (1.0 + math.cos(math.pi * rel_p)))
    blended = torch.lerp(delta_ema, u_anvil, lam)
    return torch.where(active_step, blended, delta_ema)


@torch.no_grad()
def build_spatial_surprisal_order(
    x: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    """Compute `O(n*d)` 16-shell radial-surprisal + PC1 snake sort order (`N-Track-5` `difficulty_order`).

    Partitions vectors into 16 equal-size radial-energy quantile shells (matching `difficulty_order`
    surprisal strata in `qlabs-eng/noise`) and orders continuously within/across shells along the
    leading principal axis via 3-step power iteration with alternating snake parity (`0->1->1->0`),
    eliminating shell-boundary discontinuities and cutting batch radial variance by >20x.
    """
    n, d = x.shape
    device = x.device
    if n <= 1:
        return torch.arange(n, device=device, dtype=torch.int64)

    xf = x.float()
    center = xf.mean(dim=0, keepdim=True)
    xc = xf - center
    rad2 = xc.square().sum(dim=1)  # (n,)

    gen = torch.Generator(device=device)
    gen.manual_seed((int(seed) + 424242) % (2**32))
    v = torch.randn((d, 1), generator=gen, device=device, dtype=torch.float32)
    v = v / torch.linalg.vector_norm(v).clamp_min(1e-12)
    sub_n = min(n, 4096)
    xc_sub = xc[:sub_n]
    for _ in range(3):
        v_next = xc_sub.t() @ (xc_sub @ v)
        vn = torch.linalg.vector_norm(v_next)
        v = torch.where(vn > 1e-20, v_next / vn.clamp_min(1e-20), v)

    pc1 = (xc @ v).reshape(-1)  # (n,)
    pc_span = (pc1.max() - pc1.min()).clamp_min(1e-12)
    pc_norm = ((pc1 - pc1.min()) / pc_span).clamp(0.0, 1.0)

    n_shells = min(16, n)
    rad_order = torch.argsort(rad2, stable=True)
    rad_ranks = torch.empty_like(rad_order)
    rad_ranks[rad_order] = torch.arange(n, device=device, dtype=torch.int64)
    shell = (rad_ranks * n_shells) // n

    snake_pc = torch.where((shell & 1) == 1, 1.0 - pc_norm, pc_norm)
    sort_key = shell.double() + snake_pc.double() * 0.9999
    return torch.argsort(sort_key, stable=True)


@torch.no_grad()
def sample_stratified_epoch_order(
    sorted_order: torch.Tensor,
    rows_in_ep: int,
    batch_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Stratified without-replacement permutation (`train/speedrun/run.py:4126-4129`).

    Reshapes the sorted dataset into `B` strata of `S = n // B` points and permutes each
    stratum independently across the `S` steps so every mini-batch receives 1 point from
    every spatial-radial stratum while visiting each training point at most once!
    """
    n = sorted_order.numel()
    device = sorted_order.device
    b_eff = max(1, min(batch_size, n))
    num_strata_steps = n // b_eff

    if num_strata_steps < 2 or b_eff < 2:
        perm = torch.randperm(n, generator=generator, device=device)
        return sorted_order[perm[:rows_in_ep]]

    main_count = b_eff * num_strata_steps
    rem_count = n - main_count
    if rem_count > 0:
        # Draw 1 remainder element per quantile bucket across `[0, n)` so NEITHER the main strata
        # NOR the remainder batch ever quarantines the outer radial/outlier shell!
        r_ids = torch.arange(rem_count, device=device, dtype=torch.int64)
        starts = (r_ids * n) // rem_count
        ends = ((r_ids + 1) * n) // rem_count
        lengths = (ends - starts).clamp_min(1)
        u_rem = torch.rand((rem_count,), generator=generator, device=device, dtype=torch.float32)
        rem_pos = starts + (u_rem * lengths.float()).to(dtype=torch.int64).minimum(lengths - 1)
        keep_mask = torch.ones((n,), dtype=torch.bool, device=device)
        keep_mask[rem_pos] = False
        main_sorted = sorted_order[keep_mask]
        rem = sorted_order[rem_pos]
    else:
        main_sorted = sorted_order
        rem = None

    strata = main_sorted.reshape(b_eff, num_strata_steps)
    rand_u = torch.rand((b_eff, num_strata_steps), generator=generator, device=device, dtype=torch.float32)
    col_perm = torch.argsort(rand_u, dim=1)
    # Transpose to `(num_strata_steps, b_eff)` so each step `s` draws 1 element from each stratum `0..b_eff-1`
    batches = torch.gather(strata, 1, col_perm).t().contiguous()
    # Shuffle within each mini-batch so first/second halves (for P/Q or Dust split-batch) are identically distributed!
    intra_u = torch.rand((num_strata_steps, b_eff), generator=generator, device=device, dtype=torch.float32)
    batches = torch.gather(batches, 1, torch.argsort(intra_u, dim=1))
    flat_main = batches.reshape(-1)

    if rem is not None:
        rem_perm = rem[torch.randperm(rem_count, generator=generator, device=device)]
        full_order = torch.cat([flat_main, rem_perm], dim=0)
    else:
        full_order = flat_main

    return full_order[:rows_in_ep].contiguous()


@torch.no_grad()
def execute_quad_merge(
    x_hold: torch.Tensor,
    c_close_fit: torch.Tensor,
    pass_counts_fit: torch.Tensor,
    pass_sums_fit: torch.Tensor,
    candidates: list[tuple[str, torch.Tensor]],
    compute_dtype: torch.dtype,
    backend: str,
) -> tuple[torch.Tensor, QuadMergeDiagnostics]:
    """Run `N-Track-2` (`quad_merge`) 3-point parabolic part merge on internal training holdout `x_hold`.

    Pre-centers `x_hold` and all candidate centroid banks in FP64 -> FP32 so parabolic curvature
    estimates are exact under arbitrary coordinate shifts (`+5000`..`+1e6`), free of BF16 ULP
    stair-stepping, precision-bounded (`<= kappa / (n_j + kappa)`) against populous-cluster slope
    noise, and batched into 3 host-device syncs total.
    """
    _ = compute_dtype
    k, d = c_close_fit.shape
    device = c_close_fit.device

    origin64 = x_hold.double().mean(dim=0, keepdim=True)
    x_hold_c = (x_hold.double() - origin64).float().contiguous()
    c_close_c = (c_close_fit.double() - origin64).float().contiguous()
    cand_centered = [
        (t_name, t_mat, (t_mat.double() - origin64).float().contiguous())
        for t_name, t_mat in candidates
    ]

    def _queue_hold_mse_c(cand_c: torch.Tensor) -> torch.Tensor:
        d2, _ = assign_points(x_hold_c, cand_c, compute_dtype=torch.float32, backend=backend)
        return d2.double().mean()

    # Finite-sample precision bound (`kappa / (n_j + kappa)`): prevents noisy holdout slopes on
    # small `x_hold` from overwriting populous Zipf head clusters (`n_j >> 64`) while allowing full
    # parabolic shrinkage on sparse / low-`n/k` clusters!
    kappa_prior = 64.0
    cap_vec = (kappa_prior / (pass_counts_fit.float().clamp_min(0.0) + kappa_prior)).clamp(0.0, 1.0)
    cap_col = cap_vec[:, None]

    # 1. Stage 1 (1 batched sync): plain MSE + coarse anchor probes across `{tail_med, tail_slow, c_ema}`
    anchor_alphas = (0.25, 0.50, 0.75, 1.0)
    stage1_tensors: list[torch.Tensor] = [_queue_hold_mse_c(c_close_c)]
    stage1_meta: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    for t_name, t_mat, t_c in cand_centered:
        diff_c = cap_col * (t_c - c_close_c)
        for alpha_probe in anchor_alphas:
            stage1_tensors.append(_queue_hold_mse_c((c_close_c + alpha_probe * diff_c).contiguous()))
            stage1_meta.append((t_name, t_mat, t_c))

    stage1_vals = torch.stack(stage1_tensors).tolist()
    plain_mse = float(stage1_vals[0])

    best_target_name, best_target_tensor, best_target_c = cand_centered[0]
    best_anchor_mse = float("inf")
    for m_val, (t_name, t_mat, t_c) in zip(stage1_vals[1:], stage1_meta):
        if m_val < best_anchor_mse:
            best_anchor_mse = float(m_val)
            best_target_name = t_name
            best_target_tensor = t_mat
            best_target_c = t_c

    eff_move_c = cap_col * (best_target_c - c_close_c)  # (K, D), centered FP32

    # 2. Stage 2 (1 batched sync): 3-point parabola fit `(0, delta, 2*delta)` per usage-quantile part
    n_parts = min(8, k)
    part_order = torch.argsort(pass_counts_fit, stable=True)
    part_slices = torch.tensor_split(part_order, n_parts)

    delta = 0.45
    stage2_tensors: list[torch.Tensor] = []
    active_parts: list[tuple[int, torch.Tensor]] = []
    for p_idx, p_ids in enumerate(part_slices):
        if p_ids.numel() == 0:
            continue
        trial_mid = c_close_c.clone()
        trial_mid[p_ids] = c_close_c[p_ids] + delta * eff_move_c[p_ids]
        trial_hi = c_close_c.clone()
        trial_hi[p_ids] = c_close_c[p_ids] + (2.0 * delta) * eff_move_c[p_ids]
        stage2_tensors.append(_queue_hold_mse_c(trial_mid))
        stage2_tensors.append(_queue_hold_mse_c(trial_hi))
        active_parts.append((p_idx, p_ids))

    vertices_list = [0.0 for _ in range(n_parts)]
    cluster_alpha = torch.zeros((k,), dtype=torch.float32, device=device)
    if stage2_tensors:
        stage2_vals = torch.stack(stage2_tensors).tolist()
        for idx_pair, (p_idx, p_ids) in enumerate(active_parts):
            l0 = plain_mse
            l1 = float(stage2_vals[2 * idx_pair])
            l2 = float(stage2_vals[2 * idx_pair + 1])
            curve = l0 + l2 - 2.0 * l1
            slope = l2 - l0
            if curve > 1e-15:
                v_cand = max(0.0, min(1.0, delta - delta * slope / (2.0 * curve)))
                dv = v_cand - delta
                q_pred = l1 + (slope / (2.0 * delta)) * dv + (curve / (2.0 * delta * delta)) * (dv * dv)
                v_p = v_cand if q_pred < l0 else 0.0
            else:
                if l2 < l1 and l2 < l0:
                    v_p = min(1.0, 2.0 * delta)
                elif l1 < l0:
                    v_p = delta
                else:
                    v_p = 0.0
            vertices_list[p_idx] = v_p
            if v_p > 0.0:
                cluster_alpha[p_ids] = v_p * cap_vec[p_ids]

    # 3. Stage 3 (1 batched sync): Global shrinkage race (`s in {0.25, 0.5, 0.75, 1.0}` + uniform capped)
    raw_move_c = best_target_c - c_close_c
    stage3_tensors: list[torch.Tensor] = []
    stage3_candidates: list[tuple[float, torch.Tensor]] = []
    for s_val in (0.25, 0.50, 0.75, 1.0):
        a_vec = s_val * cluster_alpha
        stage3_tensors.append(_queue_hold_mse_c((c_close_c + a_vec[:, None] * raw_move_c).contiguous()))
        stage3_candidates.append((s_val, a_vec))
    for u_val in (0.25, 0.50, 0.75, 1.0):
        a_u = u_val * cap_vec
        stage3_tensors.append(_queue_hold_mse_c((c_close_c + a_u[:, None] * raw_move_c).contiguous()))
        stage3_candidates.append((u_val, a_u))

    stage3_vals = torch.stack(stage3_tensors).tolist()
    best_mse = plain_mse
    best_shrink = 0.0
    best_alpha_vec = torch.zeros((k,), dtype=torch.float32, device=device)
    tol = 1e-8 * max(1.0, abs(plain_mse))
    for m_val, (s_cand, a_cand) in zip(stage3_vals, stage3_candidates):
        if m_val < best_mse - tol:
            best_mse = float(m_val)
            best_shrink = float(s_cand)
            best_alpha_vec = a_cand

    # 4. Fold `x_hold` sufficient statistics into `c_close_all` using centered FP32 assignment & FP64 sums
    _, hold_lbls = assign_points(x_hold_c, c_close_c, compute_dtype=torch.float32, backend=backend)
    hold_counts = torch.bincount(hold_lbls, minlength=k).double()
    hold_sums64 = torch.zeros((k, d), dtype=torch.float64, device=device)
    hold_sums64.index_add_(0, hold_lbls, x_hold.double())

    tot_counts = pass_counts_fit + hold_counts
    tot_sums64 = pass_sums_fit.double() + hold_sums64
    nonempty_all = tot_counts > 0
    c_close_all = torch.where(
        nonempty_all[:, None],
        (tot_sums64 / tot_counts.clamp_min(1.0)[:, None]).float(),
        c_close_fit,
    )

    merged_centroids = (
        (1.0 - best_alpha_vec[:, None]) * c_close_all
        + best_alpha_vec[:, None] * best_target_tensor
    ).contiguous()

    diag = QuadMergeDiagnostics(
        holdout_rows=int(x_hold.shape[0]),
        selected_target=best_target_name,
        selected_shrink=float(best_shrink),
        holdout_mse_plain=float(plain_mse),
        holdout_mse_merged=float(best_mse),
        mean_blend_weight=float(best_alpha_vec.mean().item()),
        part_vertices=tuple(float(v) for v in vertices_list),
    )
    return merged_centroids, diag


# ---------------------------------------------------------------------------
# Unified Forward-Only Clustering Engine (`EMeansNoiseTrack`)
# ---------------------------------------------------------------------------


class EMeansNoiseTrack:
    """Modular GPU/CPU clustering solver implementing `N-Track-1..5` on `e-means` & `e-means++`.

    When all five forward-only tracks are disabled (defaults), `EMeansNoiseTrack` executes
    with exact parity against `EMeans` (`base_mode='emeans'`) and `EMeansPP` (`base_mode='emeans_pp'`).
    """

    def __init__(
        self,
        d: int,
        k: int,
        nepochs: float = 5.0,
        base_mode: str = "emeans_pp",
        batch_law: str = "auto",
        batch_size: int | None = None,
        alpha0: float | None = None,
        alpha_min_factor: float = ALPHA_MIN_FACTOR,
        unused_threshold: float | None = None,
        check_unused_every: int = CHECK_UNUSED_EVERY,
        split_epsilon: float = SPLIT_EPSILON,
        init_usage: float = DEFAULT_INIT_USAGE,
        # --- `e-means++` base configuration ---
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
        # --- N-Track-1: `head_edit` Forward-Verified Hadamard Transplant ---
        head_edit: bool = False,
        transplant_every: int = 8,
        transplant_stop_frac: float = 0.75,
        transplant_probes: int = 8,
        transplant_max_pairs: int = 4,
        transplant_protect_steps: int = 10,
        # --- N-Track-2: `quad_merge` Held-Out Parabolic Trajectory Merge ---
        finalization: str = "closing_mean",  # 'closing_mean', 'slow_ema', 'none', or 'quad_merge'
        slow_ema_mult: float = 0.25,
        merge_holdout_frac: float = 0.025,
        merge_from_frac: float = 0.30,
        # --- N-Track-3: `anvil_polar` Dual-Rail Gram-Newton-Schulz Polar Update ---
        anvil_blend0: float = 0.0,
        anvil_stop_frac: float = 0.60,
        anvil_polar_margin: float = 1.1,
        # --- N-Track-4: `dust_antithetic` Split-Batch Signed-Hadamard Voronoi Jitter ---
        dust_sigma0: float = 0.0,
        dust_stop_frac: float = 0.55,
        # --- N-Track-5: `speedrun_strat_wide` Stratified Batching & Wide-Step Schedule ---
        stratified_order: bool = False,
        wide_step_mult: int = 1,
        wide_from_frac: float = 0.25,
        # --- Shared execution options ---
        revival_mode: str = "split",
        use_cosine_schedule: bool = True,
        compute_dtype: torch.dtype | None = None,
        backend: str = "auto",
        seed: int = 1234,
        verbose: bool = False,
    ) -> None:
        if base_mode not in ("emeans", "emeans_pp"):
            raise ValueError("base_mode must be 'emeans' or 'emeans_pp'")
        if finalization not in ("closing_mean", "slow_ema", "none", "quad_merge"):
            raise ValueError("finalization must be 'closing_mean', 'slow_ema', 'none', or 'quad_merge'")
        if revival_mode not in ("split", "resample", "off"):
            raise ValueError("revival_mode must be 'split', 'resample', or 'off'")
        if transplant_every < 1 or transplant_probes < 1 or transplant_max_pairs < 1:
            raise ValueError("transplant_every, transplant_probes, and transplant_max_pairs must be >= 1")
        if not (0.0 <= anvil_blend0 <= 1.0):
            raise ValueError("anvil_blend0 must be in [0.0, 1.0]")
        if dust_sigma0 < 0.0 or not math.isfinite(dust_sigma0):
            raise ValueError("dust_sigma0 must be finite and >= 0.0")
        if wide_step_mult < 1:
            raise ValueError("wide_step_mult must be >= 1")
        if not (0.0 <= merge_holdout_frac < 0.5):
            raise ValueError("merge_holdout_frac must be in [0.0, 0.5)")

        self.d = int(d)
        self.k = int(k)
        self.nepochs = float(nepochs)
        self.base_mode = base_mode
        self.batch_law = batch_law
        self.batch_size = batch_size
        self.alpha0 = alpha0
        self.alpha_min_factor = float(alpha_min_factor)
        self.unused_threshold = unused_threshold
        self.check_unused_every = int(check_unused_every)
        self.split_epsilon = float(split_epsilon)
        self.init_usage = float(init_usage)

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

        self.head_edit = bool(head_edit)
        self.transplant_every = int(transplant_every)
        self.transplant_stop_frac = float(transplant_stop_frac)
        self.transplant_probes = int(transplant_probes)
        self.transplant_max_pairs = int(transplant_max_pairs)
        self.transplant_protect_steps = int(transplant_protect_steps)

        self.finalization = finalization
        self.slow_ema_mult = float(slow_ema_mult)
        self.merge_holdout_frac = float(merge_holdout_frac)
        self.merge_from_frac = float(merge_from_frac)

        self.anvil_blend0 = float(anvil_blend0)
        self.anvil_stop_frac = float(anvil_stop_frac)
        self.anvil_polar_margin = float(anvil_polar_margin)

        self.dust_sigma0 = float(dust_sigma0)
        self.dust_stop_frac = float(dust_stop_frac)

        self.stratified_order = bool(stratified_order)
        self.wide_step_mult = int(wide_step_mult)
        self.wide_from_frac = float(wide_from_frac)

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
        self.iteration_stats: list[EpochStat] = []
        self.n_splits: int = 0
        self.n_protected: int = 0
        self.n_transplants: int = 0
        self.head_edit_diagnostics: HeadEditDiagnostics | None = None
        self.quad_merge_diagnostics: QuadMergeDiagnostics | None = None
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
        """Run forward-only enhanced clustering on dataset `x` of shape `(N, D)`."""
        input_is_numpy = isinstance(x, np.ndarray)
        x_full = (
            torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
            if input_is_numpy
            else x.detach()
        )
        if not x_full.is_cuda and x_full.dtype in (torch.bfloat16, torch.float16):
            # Faiss CPU knn (`kernels.py:295`) converts via `.numpy()`, which rejects CPU bf16
            x_full = x_full.float()
        n_full, d = x_full.shape
        if d != self.d:
            raise ValueError(f"Expected dimension d={self.d}, got {d}")
        if n_full < self.k:
            raise ValueError(f"Number of points n={n_full} must be >= number of clusters k={self.k}")

        device = x_full.device
        comp_dtype = self._resolve_compute_dtype(x_full)
        eff_threshold = 0.0 if self.revival_mode == "off" else self.unused_threshold
        is_pp = self.base_mode == "emeans_pp"

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_init_start = time.perf_counter()

        # Optional internal disjoint holdout carving from `x_full` for `N-Track-2` (`quad_merge`)
        use_quad_merge = self.finalization == "quad_merge" and self.merge_holdout_frac > 0.0
        n_hold = 0
        if use_quad_merge:
            raw_hold = min(4096, max(64, int(round(self.merge_holdout_frac * n_full))))
            # Never starve `x_fit` below `max(k, 64)`
            if n_full - raw_hold >= max(self.k, 64):
                n_hold = raw_hold

        if n_hold >= 16:
            gen_hold = torch.Generator(device=device)
            gen_hold.manual_seed((self.seed + 271828) % (2**32))
            hold_perm = torch.randperm(n_full, generator=gen_hold, device=device)
            hold_idx = hold_perm[:n_hold]
            fit_idx = hold_perm[n_hold:]
            x_t = x_full[fit_idx].contiguous()
            x_hold = x_full[hold_idx].contiguous()
        else:
            fit_idx = None
            x_hold = None
            x_t = x_full

        n = int(x_t.shape[0])

        # Build recipe anchored on `n_full` for batch_size & alpha0 so `quad_merge`'s 2.5%
        # internal holdout never flips batch_size across a power-of-two boundary (`n == n_full` when off)
        if is_pp:
            full_base_recipe = build_recipe(
                n=n_full,
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
                float(min(max(full_base_recipe.alpha0 * self.warm_alpha_scale, 1e-4), 1.0))
                if self.alpha0 is None
                else float(full_base_recipe.alpha0)
            )
            recipe = build_recipe(
                n=n,
                d=d,
                k=self.k,
                nepochs=self.nepochs,
                batch_law=self.batch_law,
                batch_size=min(n, full_base_recipe.batch_size),
                alpha0=eff_alpha0,
                alpha_min_factor=self.alpha_min_factor,
                unused_threshold=eff_threshold,
                check_unused_every=self.check_unused_every,
                split_epsilon=self.split_epsilon,
                init_usage=self.init_usage,
            )
        else:
            full_base_recipe = build_recipe(
                n=n_full,
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
            if n == n_full:
                recipe = full_base_recipe
            else:
                recipe = build_recipe(
                    n=n,
                    d=d,
                    k=self.k,
                    nepochs=self.nepochs,
                    batch_law=self.batch_law,
                    batch_size=min(n, full_base_recipe.batch_size),
                    alpha0=full_base_recipe.alpha0,
                    alpha_min_factor=self.alpha_min_factor,
                    unused_threshold=eff_threshold,
                    check_unused_every=self.check_unused_every,
                    split_epsilon=self.split_epsilon,
                    init_usage=self.init_usage,
                )
        self.recipe = recipe

        # Initialize centroids & priors
        if is_pp:
            if seeding_result is not None:
                seed_res = seeding_result
            elif init_centroids is None:
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
                    seed_res = voronoi_polish_and_priors(x_sub, w_sub, raw_c, compute_dtype=comp_dtype)
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
        else:
            self.seeding_result = None
            if init_centroids is None:
                gen_init = torch.Generator(device=device)
                gen_init.manual_seed(self.seed)
                perm = torch.randperm(n, generator=gen_init, device=device)[: self.k]
                centroids = x_t[perm].float().clone().contiguous()
            else:
                c_in = (
                    torch.from_numpy(np.asarray(init_centroids, dtype=np.float32))
                    if isinstance(init_centroids, np.ndarray)
                    else init_centroids
                )
                centroids = c_in.to(device=device, dtype=torch.float32).clone().contiguous()
            p_hat = None
            e_hat = None

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self.init_time_sec = time.perf_counter() - t_init_start

        # --- Begin timed training phase ---
        t_train_start = time.perf_counter()
        bsz_float = float(recipe.batch_size)

        need_distortion = is_pp or (self.dust_sigma0 > 0.0) or self.head_edit
        if is_pp and p_hat is not None and e_hat is not None:
            min_prob = max(0.0, self.prior_floor) / float(self.k)
            mean_ehat = e_hat.mean().clamp_min(1e-12)
            if self.use_prior_mass:
                usage = (self.prior_mass_scale * bsz_float * p_hat.clamp_min(min_prob)).float().contiguous()
                distortion: torch.Tensor | None = (
                    self.prior_mass_scale * bsz_float * e_hat.clamp_min(1e-6 * mean_ehat)
                ).float().contiguous()
            else:
                usage = torch.full((self.k,), recipe.init_usage, dtype=torch.float32, device=device)
                distortion = (bsz_float * e_hat.clamp_min(1e-6 * mean_ehat)).float().contiguous()
            survival_mass: torch.Tensor | None = (
                bsz_float * p_hat.clamp_min(min_prob)
            ).float().contiguous()
        else:
            usage = torch.full((self.k,), recipe.init_usage, dtype=torch.float32, device=device)
            distortion = (
                torch.zeros((self.k,), dtype=torch.float32, device=device)
                if need_distortion
                else None
            )
            survival_mass = None

        track_slow = self.finalization == "slow_ema"
        if track_slow:
            slow_centroids: torch.Tensor | None = centroids.clone()
            slow_usage: torch.Tensor | None = usage.clone()
            alpha_slow = min(1.0, max(1e-6, self.slow_ema_mult * recipe.alpha_min_factor * recipe.alpha0))
        else:
            slow_centroids = None
            slow_usage = None
            alpha_slow = 0.0

        # `N-Track-2` trajectory tail EMAs (split-consistent!)
        track_tails = self.finalization == "quad_merge"
        if track_tails:
            tail_med = centroids.clone()
            tail_slow = centroids.clone()
            tails_initialized = False
            tail_banks = [tail_med, tail_slow]
        else:
            tail_med = None
            tail_slow = None
            tails_initialized = False
            tail_banks = []

        # `N-Track-3` ANVIL-II dual momentum rails
        if self.anvil_blend0 > 0.0:
            anvil_fast = torch.zeros((self.k, self.d), dtype=torch.float32, device=device)
            anvil_slow = torch.zeros((self.k, self.d), dtype=torch.float32, device=device)
            anvil_energy = torch.zeros((self.k, 1), dtype=torch.float32, device=device)
        else:
            anvil_fast = None
            anvil_slow = None
            anvil_energy = None

        # `N-Track-1` head_edit state & independent RNG
        if self.head_edit:
            last_edit_step = torch.full((self.k,), -10**9, dtype=torch.int64, device=device)
            edit_gen = torch.Generator(device=device)
            edit_gen.manual_seed((self.seed + 918273) % (2**32))
            edit_diag: HeadEditDiagnostics | None = HeadEditDiagnostics()
        else:
            last_edit_step = None
            edit_gen = None
            edit_diag = None

        # `N-Track-4` independent Hadamard perturbation RNG
        if self.dust_sigma0 > 0.0:
            dust_gen = torch.Generator(device=device)
            dust_gen.manual_seed((self.seed + 514229) % (2**32))
        else:
            dust_gen = None

        # `N-Track-5` one-time O(n*d) spatial-surprisal stratification sort
        sorted_order = build_spatial_surprisal_order(x_t, seed=self.seed) if self.stratified_order else None

        sign = torch.ones((self.d,), dtype=torch.float32, device=device)
        sign[1::2] = -1.0
        split_offset = recipe.split_epsilon * sign

        split_gen = torch.Generator(device=device)
        split_gen.manual_seed((self.seed + 7919) % (2**32))

        assignments_fit = (
            torch.full((n,), -1, dtype=torch.int64, device=device)
            if (capture_assignments and fit_idx is None)
            else None
        )

        ones_scalar = torch.ones((1,), dtype=torch.float32, device=device)
        batch_sums = torch.empty((self.k, self.d), dtype=torch.float32, device=device)
        batch_sse = (
            torch.empty((self.k,), dtype=torch.float32, device=device)
            if distortion is not None
            else None
        )
        sub_sums = (
            torch.empty((self.k, self.d), dtype=torch.float32, device=device)
            if self.dust_sigma0 > 0.0
            else None
        )

        step = 0
        rows_processed = 0
        total_planned_rows = max(1, sum(recipe.epoch_rows))
        self.n_splits = 0
        self.n_protected = 0
        self.n_transplants = 0
        self.iteration_stats = []
        self.quad_merge_diagnostics = None
        num_epochs = len(recipe.epoch_rows)
        eff_grace = (
            min(self.grace_steps, max(0, recipe.total_steps // 3))
            if (is_pp and self.use_huygens_protection)
            else 0
        )
        non_closing_rows = sum(recipe.epoch_rows[:-1])
        pre_close_frac = (
            float(non_closing_rows) / float(total_planned_rows)
            if num_epochs >= 2
            else 1.0
        )
        eff_dust_stop = min(self.dust_stop_frac, pre_close_frac)
        eff_anvil_stop = min(self.anvil_stop_frac, pre_close_frac)
        non_closing_steps = sum(
            math.ceil(r / max(1, recipe.batch_size)) for r in recipe.epoch_rows[:-1]
        )
        eff_transplant_every = (
            max(1, min(self.transplant_every, max(2, non_closing_steps // 4)))
            if non_closing_steps > 0
            else self.transplant_every
        )
        eff_transplant_protect = min(self.transplant_protect_steps, max(2, eff_transplant_every * 2))

        # `e-means` step-coupling exponent for `wide_step_mult > 1` (`ALPHA_LAW[2] = 0.9003`)
        wide_alpha_factor = float(self.wide_step_mult ** ALPHA_LAW[2]) if self.wide_step_mult > 1 else 1.0
        in_wide_phase = False

        last_pass_counts: torch.Tensor | None = None
        last_pass_sums: torch.Tensor | None = None

        for ep_idx, rows_in_ep in enumerate(recipe.epoch_rows):
            is_last_epoch = ep_idx == (num_epochs - 1)
            gen_ep = torch.Generator(device=device)
            gen_ep.manual_seed(epoch_seed(self.seed, ep_idx))

            # Determine active batch size at epoch start for stratified shaping
            ep_start_prog = float(rows_processed) / float(total_planned_rows)
            ep_strat_bsz = (
                min(n, recipe.batch_size * self.wide_step_mult)
                if (self.wide_step_mult > 1 and ep_start_prog >= self.wide_from_frac)
                else recipe.batch_size
            )
            if sorted_order is not None:
                order = sample_stratified_epoch_order(
                    sorted_order, rows_in_ep=rows_in_ep, batch_size=ep_strat_bsz, generator=gen_ep
                )
            else:
                order = torch.randperm(n, generator=gen_ep, device=device)[:rows_in_ep]

            pass_counts = torch.zeros((self.k,), dtype=torch.float64, device=device)
            need_closing_sums = is_last_epoch and self.finalization in ("closing_mean", "quad_merge")
            pass_sums = (
                torch.zeros((self.k, self.d), dtype=torch.float32, device=device)
                if need_closing_sums
                else None
            )
            ep_sq_dist_t = torch.zeros((), dtype=torch.float64, device=device)
            splits_before = self.n_splits
            ep_steps = 0

            b_start = 0
            while b_start < rows_in_ep:
                prog = float(rows_processed) / float(total_planned_rows)
                use_wide_now = self.wide_step_mult > 1 and prog >= self.wide_from_frac
                cur_bsz_target = (
                    min(n, recipe.batch_size * self.wide_step_mult)
                    if use_wide_now
                    else recipe.batch_size
                )
                if use_wide_now and not in_wide_phase:
                    # Rescale batch-count-denominated EMA accumulators once so `(usage, distortion,
                    # survival_mass, slow_usage)` match `cur_bsz_target` units without transient jump
                    w_scale = float(cur_bsz_target) / float(max(1, recipe.batch_size))
                    if w_scale != 1.0:
                        usage.mul_(w_scale)
                        if distortion is not None:
                            distortion.mul_(w_scale)
                        if survival_mass is not None:
                            survival_mass.mul_(w_scale)
                        if slow_usage is not None:
                            slow_usage.mul_(w_scale)
                    in_wide_phase = True
                b_end = min(b_start + cur_bsz_target, rows_in_ep)
                batch_idx = order[b_start:b_end]
                x_batch = x_t[batch_idx]
                bsz = int(x_batch.shape[0])

                # --- Forward Routing (`N-Track-4` Antithetic Hadamard Jitter vs Standard Hard Routing) ---
                # Strictly disabled on `is_last_epoch` so closing-pass counts/sums/assignments are 100% clean!
                use_dust_now = (
                    self.dust_sigma0 > 0.0
                    and dust_gen is not None
                    and not is_last_epoch
                    and step >= 1
                    and prog < eff_dust_stop
                    and bsz >= 4
                )
                if use_dust_now and distortion is not None and sub_sums is not None:
                    rel_jit = max(0.0, min(1.0, prog / max(eff_dust_stop, 1e-6)))
                    sigma_t = float(self.dust_sigma0 * 0.5 * (1.0 + math.cos(math.pi * rel_jit)))
                    pos_u = usage > 1e-8
                    fallback_var = (
                        distortion.sum() / (usage.sum().clamp_min(1e-8) * float(self.d))
                    ).clamp_min(0.0)
                    coord_var = torch.where(
                        pos_u,
                        (distortion / (usage.clamp_min(1e-8) * float(self.d))).clamp_min(0.0),
                        fallback_var,
                    )
                    rms_scale = (sigma_t * coord_var.sqrt())[:, None]  # (K, 1)
                    e_frame = sample_signed_hadamard_rows(
                        self.k, self.d, generator=dust_gen, device=device, unit_l2=False
                    )
                    delta_jit = rms_scale * e_frame
                    c_plus = centroids + delta_jit
                    c_minus = centroids - delta_jit

                    mid = bsz // 2
                    xb_p = x_batch[:mid]
                    xb_m = x_batch[mid:]

                    cp_rep, cp_norm = prepare_represented_centroids(c_plus, comp_dtype)
                    sq_p, lbl_p = assign_points(
                        xb_p, c_plus, compute_dtype=comp_dtype, backend=self.backend, c_rep_cached=(cp_rep, cp_norm)
                    )
                    cm_rep, cm_norm = prepare_represented_centroids(c_minus, comp_dtype)
                    sq_m, lbl_m = assign_points(
                        xb_m, c_minus, compute_dtype=comp_dtype, backend=self.backend, c_rep_cached=(cm_rep, cm_norm)
                    )

                    if is_last_epoch and assignments_fit is not None:
                        assignments_fit[batch_idx[:mid]] = lbl_p
                        assignments_fit[batch_idx[mid:]] = lbl_m

                    ep_sq_dist_t += sq_p.double().sum() + sq_m.double().sum()

                    counts = (
                        torch.bincount(lbl_p, minlength=self.k).float()
                        + torch.bincount(lbl_m, minlength=self.k).float()
                    )
                    batch_sums.zero_()
                    batch_sums.index_add_(0, lbl_p, xb_p.float())
                    sub_sums.zero_()
                    sub_sums.index_add_(0, lbl_m, xb_m.float())
                    batch_sums.add_(sub_sums)

                    if batch_sse is not None:
                        batch_sse.zero_()
                        batch_sse.index_add_(0, lbl_p, sq_p.float())
                        batch_sse.index_add_(0, lbl_m, sq_m.float())
                else:
                    c_rep, c_norm = prepare_represented_centroids(centroids, comp_dtype)
                    sq_dists, labels = assign_points(
                        x_batch,
                        centroids,
                        compute_dtype=comp_dtype,
                        backend=self.backend,
                        c_rep_cached=(c_rep, c_norm),
                    )

                    if is_last_epoch and assignments_fit is not None:
                        assignments_fit[batch_idx] = labels

                    ep_sq_dist_t += sq_dists.double().sum()

                    counts = torch.bincount(labels, minlength=self.k).float()
                    batch_sums.zero_()
                    batch_sums.index_add_(0, labels, x_batch.float())
                    if batch_sse is not None:
                        batch_sse.zero_()
                        batch_sse.index_add_(0, labels, sq_dists.float())

                pass_counts += counts.double()
                if pass_sums is not None:
                    pass_sums += batch_sums

                # --- Step Size Schedule ---
                if self.wide_step_mult > 1:
                    eff_peak = min(1.0, recipe.alpha0 * (wide_alpha_factor if use_wide_now else 1.0))
                    if self.use_cosine_schedule:
                        cos_term = 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
                        alpha = float(
                            eff_peak * (recipe.alpha_min_factor + (1.0 - recipe.alpha_min_factor) * cos_term)
                        )
                    else:
                        alpha = float(eff_peak)
                else:
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

                # --- EMA Update (+ `N-Track-3` ANVIL-II Polar Displacement Blend) ---
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

                if (
                    self.anvil_blend0 > 0.0
                    and anvil_fast is not None
                    and anvil_slow is not None
                    and anvil_energy is not None
                    and not is_last_epoch
                    and prog < eff_anvil_stop
                ):
                    raw_delta = torch.where(
                        touched[:, None],
                        weight[:, None] * (batch_means - centroids),
                        torch.zeros_like(centroids),
                    )
                    eff_delta = anvil_polar_displacement(
                        delta_ema=raw_delta,
                        touched=touched,
                        fast_rail=anvil_fast,
                        slow_rail=anvil_slow,
                        lane_energy=anvil_energy,
                        progress=prog,
                        blend0=self.anvil_blend0,
                        stop_frac=eff_anvil_stop,
                        polar_margin=self.anvil_polar_margin,
                    )
                    updated_pos = centroids + eff_delta
                else:
                    updated_pos = centroids + weight[:, None] * (batch_means - centroids)
                    updated_pos = torch.where((weight == 1.0)[:, None], batch_means, updated_pos)

                centroids = torch.where(touched[:, None], updated_pos, centroids)
                usage = torch.where(touched, new_mass, usage)

                if distortion is not None and batch_sse is not None:
                    if alpha < 1.0:
                        distortion.mul_(decay).add_(batch_sse, alpha=alpha)
                    else:
                        distortion = torch.where(touched, batch_sse, distortion * 0.5)

                if survival_mass is not None:
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

                # --- Standard Dead-Cluster Revival ---
                allow_revival = (
                    recipe.unused_threshold > 0.0
                    and self.revival_mode != "off"
                    and ((step + 1) % recipe.check_unused_every == 0)
                    and (not is_pp or ((step + 1) >= max(1, eff_grace)))
                )
                if allow_revival:
                    thresh = usage.mean() * recipe.unused_threshold
                    low_mask = usage < thresh
                    n_low = int(low_mask.sum().item())

                    if n_low > 0:
                        if (
                            is_pp
                            and self.use_huygens_protection
                            and distortion is not None
                            and survival_mass is not None
                            and self.k >= 2
                            and self.gamma_protect > 0.0
                        ):
                            cand_ids = torch.nonzero(low_mask, as_tuple=False).reshape(-1)
                            d_nn2 = _nearest_other_centroid_sq_dists(centroids, cand_ids)
                            eff_mass = torch.maximum(
                                usage[cand_ids],
                                torch.minimum(survival_mass[cand_ids], thresh),
                            )
                            removal_cost = eff_mass * d_nn2
                            protect_cutoff = self.gamma_protect * distortion.mean().clamp_min(1e-30)
                            protected_local = removal_cost >= protect_cutoff
                            unused_mask = low_mask.clone()
                            unused_mask[cand_ids[protected_local]] = False
                        else:
                            protected_local = None
                            unused_mask = low_mask

                        donor_mask = ((~low_mask) if is_pp else (~unused_mask)) & (usage > 0)
                        pos_dist_donors = (
                            (donor_mask & (distortion > 1e-30))
                            if (is_pp and self.use_d2_donor_split and distortion is not None)
                            else None
                        )
                        cnt_tensors = [unused_mask.sum(), donor_mask.sum()]
                        if protected_local is not None:
                            cnt_tensors.append(protected_local.sum())
                        if pos_dist_donors is not None:
                            cnt_tensors.append(pos_dist_donors.sum())
                        cnt_vals = torch.stack(cnt_tensors).tolist()
                        n_unused = int(cnt_vals[0])
                        n_donors = int(cnt_vals[1])
                        c_ptr = 2
                        if protected_local is not None:
                            self.n_protected += int(cnt_vals[c_ptr])
                            c_ptr += 1
                        n_pos_dist = int(cnt_vals[c_ptr]) if pos_dist_donors is not None else 0

                        if self.revival_mode == "split":
                            nsplit = min(n_unused, n_donors)
                            if nsplit > 0:
                                u = torch.rand(
                                    (self.k,),
                                    generator=split_gen,
                                    device=device,
                                    dtype=torch.float32,
                                ).clamp_min_(torch.finfo(torch.float32).tiny)

                                if pos_dist_donors is not None and distortion is not None:
                                    if n_pos_dist >= nsplit:
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
                                    (donor_scores.clamp_min(1e-30) if is_pp else donor_scores).log()
                                    - (-u.log()).log(),
                                    torch.full_like(usage, float("-inf")),
                                )
                                source = torch.topk(keys, nsplit).indices
                                target = torch.nonzero(unused_mask, as_tuple=False).reshape(-1)[:nsplit]

                                donors = centroids[source]
                                d_cent = donors * (1.0 - split_offset)
                                c_cent = donors * (1.0 + split_offset)
                                _separate_collapsed_splits(d_cent, c_cent, comp_dtype)

                                half_u = usage[source] * 0.5
                                centroids[source] = d_cent
                                centroids[target] = c_cent
                                usage[source] = half_u
                                usage[target] = half_u

                                if distortion is not None:
                                    half_e = distortion[source] * 0.5
                                    distortion[source] = half_e
                                    distortion[target] = half_e
                                if survival_mass is not None:
                                    half_m = survival_mass[source] * 0.5
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

                                for tb in tail_banks:
                                    tb_don = tb[source]
                                    tbd = tb_don * (1.0 - split_offset)
                                    tbc = tb_don * (1.0 + split_offset)
                                    _separate_collapsed_splits(tbd, tbc, comp_dtype)
                                    tb[source] = tbd
                                    tb[target] = tbc

                                if anvil_fast is not None and anvil_slow is not None and anvil_energy is not None:
                                    anvil_fast[source] = 0.0
                                    anvil_fast[target] = 0.0
                                    anvil_slow[source] = 0.0
                                    anvil_slow[target] = 0.0
                                    anvil_energy[source] = 0.0
                                    anvil_energy[target] = 0.0

                                self.n_splits += nsplit
                        elif self.revival_mode == "resample" and n_unused > 0 and bsz > 0:
                            nsplit = min(n_unused, bsz)
                            target = torch.nonzero(unused_mask, as_tuple=False).reshape(-1)[:nsplit]
                            r_idx = torch.randperm(bsz, generator=split_gen, device=device)[:nsplit]
                            picked = x_batch[r_idx].float()
                            centroids[target] = picked
                            if is_pp:
                                mean_u = usage.mean().clamp_min(recipe.init_usage)
                                usage[target] = mean_u
                                if distortion is not None:
                                    distortion[target] = distortion.mean().clamp_min(1e-12)
                                if survival_mass is not None:
                                    survival_mass[target] = mean_u
                            else:
                                usage[target] = usage.mean()
                                if distortion is not None:
                                    distortion[target] = distortion.mean()
                            for tb in tail_banks:
                                tb[target] = picked
                            if anvil_fast is not None and anvil_slow is not None and anvil_energy is not None:
                                anvil_fast[target] = 0.0
                                anvil_slow[target] = 0.0
                                anvil_energy[target] = 0.0
                            self.n_splits += nsplit

                # --- `N-Track-1` (`head_edit`): P/Q Forward-Verified Signed-Hadamard Transplant ---
                if (
                    self.head_edit
                    and edit_gen is not None
                    and last_edit_step is not None
                    and edit_diag is not None
                    and not is_last_epoch
                    and ((step + 1) % eff_transplant_every == 0)
                    and (prog <= self.transplant_stop_frac)
                ):
                    n_ed = run_head_edit_step(
                        x_batch=x_batch,
                        centroids=centroids,
                        usage=usage,
                        distortion=distortion,
                        survival_mass=survival_mass,
                        slow_centroids=slow_centroids,
                        slow_usage=slow_usage,
                        tail_banks=tail_banks,
                        last_edit_step=last_edit_step,
                        step=step,
                        compute_dtype=comp_dtype,
                        unused_threshold=recipe.unused_threshold,
                        use_huygens_protection=(is_pp and self.use_huygens_protection),
                        gamma_protect=self.gamma_protect,
                        num_probes=self.transplant_probes,
                        max_pairs=self.transplant_max_pairs,
                        protect_steps=eff_transplant_protect,
                        generator=edit_gen,
                        diag=edit_diag,
                    )
                    if n_ed > 0 and anvil_fast is not None and anvil_slow is not None:
                        edited_now = last_edit_step == step
                        anvil_fast[edited_now] = 0.0
                        anvil_slow[edited_now] = 0.0
                        if anvil_energy is not None:
                            anvil_energy[edited_now] = 0.0
                    self.n_transplants += n_ed
                    self.n_splits += n_ed

                # --- `N-Track-2` (`quad_merge`) Trajectory Tail Updates (`MERGE_FROM=0.3, MERGE_EVERY=5`) ---
                rows_processed += bsz
                next_prog = float(rows_processed) / float(total_planned_rows)
                if track_tails and tail_med is not None and tail_slow is not None:
                    is_final_batch = is_last_epoch and (b_end >= rows_in_ep)
                    if next_prog >= self.merge_from_frac and (((step + 1) % 5 == 0) or is_final_batch):
                        if not tails_initialized:
                            tail_med.copy_(centroids)
                            tail_slow.copy_(centroids)
                            tails_initialized = True
                        else:
                            tail_med.lerp_(centroids, 0.05)
                            tail_slow.lerp_(centroids, 0.015)

                step += 1
                ep_steps += 1
                b_start = b_end

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            imb_t = self.k * torch.dot(pass_counts, pass_counts) / (float(max(1, rows_in_ep)) ** 2)
            ep_vals = torch.stack([ep_sq_dist_t, imb_t]).tolist()
            ep_sq_dist = float(ep_vals[0])
            imbalance = float(ep_vals[1])
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
                    f"[noise-track:{self.base_mode}] pass {ep_idx + 1}/{num_epochs} "
                    f"({rows_in_ep} rows, {ep_steps} steps): moving_mse={stat.moving_mse:.6f}, "
                    f"splits={stat.n_splits}, transplants={self.n_transplants}, time={stat.elapsed_sec:.3f}s"
                )

            if is_last_epoch:
                last_pass_counts = pass_counts
                last_pass_sums = pass_sums

        # --- Finalization (including `N-Track-2` Held-Out Parabolic Trajectory Merge) ---
        c_ema_last = centroids.clone()
        if self.finalization == "closing_mean" and last_pass_counts is not None and last_pass_sums is not None:
            nonempty = last_pass_counts > 0
            centroids = torch.where(
                nonempty[:, None],
                last_pass_sums / last_pass_counts.clamp_min(1.0).float()[:, None],
                centroids,
            )
        elif self.finalization == "slow_ema" and slow_centroids is not None:
            centroids = slow_centroids
        elif self.finalization == "quad_merge" and last_pass_counts is not None and last_pass_sums is not None:
            nonempty = last_pass_counts > 0
            c_close_fit = torch.where(
                nonempty[:, None],
                last_pass_sums / last_pass_counts.clamp_min(1.0).float()[:, None],
                c_ema_last,
            )
            if x_hold is not None and tail_med is not None and tail_slow is not None:
                if not tails_initialized:
                    tail_med.copy_(c_ema_last)
                    tail_slow.copy_(c_ema_last)
                centroids, q_diag = execute_quad_merge(
                    x_hold=x_hold,
                    c_close_fit=c_close_fit,
                    pass_counts_fit=last_pass_counts,
                    pass_sums_fit=last_pass_sums,
                    candidates=[
                        ("tail_med", tail_med),
                        ("tail_slow", tail_slow),
                        ("c_ema_last", c_ema_last),
                    ],
                    compute_dtype=comp_dtype,
                    backend=self.backend,
                )
                self.quad_merge_diagnostics = q_diag
            else:
                centroids = c_close_fit

        if capture_assignments:
            if assignments_fit is not None and self.finalization != "quad_merge":
                assignments_out: torch.Tensor | None = assignments_fit
            else:
                _, assignments_out = assign_points(
                    x_full, centroids, compute_dtype=comp_dtype, backend=self.backend
                )
        else:
            assignments_out = None

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self.train_time_sec = time.perf_counter() - t_train_start
        self.total_time_sec = self.init_time_sec + self.train_time_sec

        self.centroids = centroids.contiguous()
        self.slow_centroids = slow_centroids
        self.assignments = assignments_out
        self.head_edit_diagnostics = edit_diag

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
        """Assign points `x` to the learned centroids, returning `(sq_distances, labels)`."""
        if self.centroids is None:
            raise RuntimeError("Model has not been fitted yet.")
        input_is_numpy = isinstance(x, np.ndarray)
        x_t = (
            torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(self.centroids.device)
            if input_is_numpy
            else x.to(self.centroids.device)
        )
        if not x_t.is_cuda and x_t.dtype in (torch.bfloat16, torch.float16):
            x_t = x_t.float()
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
