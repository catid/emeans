"""Unified Champion (`SuperEMeans`) and `s-means` Cross-Pollination Engine (`HybridSEMeans`).

Synthesizes all verified Phase 1, Phase 2A, and Phase 2B winners in `emeans` into a single
zero-per-step-sync GPU solver (`SuperEMeans`), alongside a modular cross-pollination engine
(`HybridSEMeans` + `HYBRID_ABLATION_ARMS`) testing bidirectional component grafting with
`suayptalha/s-means` (Kocabay, Yuksel, & Yalcin, 2026):

`SuperEMeans` Pillars (`P1..P6`):
  - P1 (`coreset_kpp + voronoi_polish + prior_mass`):
      O(n*d) Bachem lightweight coreset + Fused Triton / Block-Greedy k-means++ + 1-step
      weighted Voronoi polish + prior-calibrated initial EMA mass `N_j^(0)`.
  - P2 (`twin_safe_huygens_protection + d2_donors`):
      Huygens parallel-axis rare-mode shield (`M_j * d_nn^2 >= gamma * mean(E)`) upgraded
      with co-located twin-leader election so simultaneous batch revivals keep 1 guardian
      centroid on twin-seeded rare modes while recycling duplicate twins.
  - P3 (`oja` / `oja_rms` principal-axis donor splitting):
      O(B*d) online Krasulina/Oja rank-1 residual covariance eigenvector tracker with
      fused Triton scatter-projection and BF16/FP16 ULP collapse separation.
  - P4 (`eb_james_stein` positive-part Empirical-Bayes / James-Stein finalizer):
      Bounded-influence per-cluster Stein shrinkage between `closing_mean` and `slow_ema`
      (with exact Kish effective sample size `n_eff = (sum w)^2 / sum(w^2)` when paired
      with weighted closing passes).
  - P5 (`head_edit` forward-verified P/Q signed-Hadamard centroid transplants):
      Periodic P/Q split-batch Voronoi dormancy & Hadamard/chord half-space transplants
      accelerated by fused Triton top-3 search and exact incremental O(K*d) top-3 / NN
      refresh + dynamic Huygens twin-centroid shielding.
  - P6 (`blackwell_triton_assign` + optional `wide_step_mult=2`):
      Blackwell Split-K / FULL_D Tensor-Core assignment with dataset mean-centering
      (100% coordinate-shift invariant even at `+20,000` in BF16/FP16/FP32/FP64),
      pre-encoded BF16/FP16 streaming, and optional 2x batch doubling after 25% budget
      coupled with `2^0.9003 * alpha_0`.

`suayptalha/s-means` Components (`C1..C7`) Supported in `HybridSEMeans`:
  - C1: Multi-round subsampled `(D^2)^1.5 = D^3` seeding (`init_mode="subsampled_d15"`).
  - C2: Power-cosine step schedule (`0.06 + 0.94 * cos^0.92`) + low-d (`d <= 2`) revival.
  - C3: 85% budget revival freeze (`use_revive_stop=True, revive_stop_frac=0.85`).
  - C4: Fixed-decay SSE * usage^0.35 donor scoring (`donor_mode="sse_usage"`).
  - C5: Online O(K*d) inter-cluster feature variance `feat_scale` + RMS Rademacher split.
  - C6: Progressive final-pass linear batch ramp (`w_pass in [1 - 0.35, 1 + 0.35]`).
  - C7: Prior-count Empirical-Bayes shrinkage (`tau = 0.20`) toward live `c_ema_last`.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .algo_extensions import (
    ShrinkageDiagnostics,
    reset_post_split_oja_axes,
    separate_collapsed_directional,
    update_oja_principal_axes,
)
from .emeans import _separate_collapsed_splits
from .fast_ops import (
    HAS_TRITON,
    blackwell_prepare_centroids,
    blackwell_triton_assign,
    fast_seed_emeans_plusplus,
)
from .kernels import assign_points, prepare_represented_centroids, torch_assign_chunked
from .noise_tracks import (
    HeadEditDiagnostics,
    exact_exclusion_sq_dists,
    sample_signed_hadamard_rows,
    top3_represented_assign_centered,
)
from .recipe import (
    ALPHA_LAW,
    ALPHA_MIN_FACTOR,
    CHECK_UNUSED_EVERY,
    DEFAULT_INIT_USAGE,
    DEFAULT_UNUSED_THRESHOLD,
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

if HAS_TRITON:
    import triton
    import triton.language as tl


# ==============================================================================
# 1. Fused Triton Kernels (Scatter + Oja Projection & SRAM Centered Top-3)
# ==============================================================================

if HAS_TRITON:

    @triton.jit(do_not_specialize=["B", "D", "stride_xb", "stride_xd", "stride_sumsk"])
    def _super_scatter_stats_kernel(
        x_ptr,
        row_idx_ptr,
        labels_ptr,
        sq_dists_ptr,
        sums_ptr,
        counts_ptr,
        sse_ptr,
        B,
        D,
        stride_xb,
        stride_xd,
        stride_sumsk,
        HAS_INDICES: tl.constexpr,
        HAS_SSE: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Fused gather + atomic scatter for batch sums, counts, and per-cluster SSE."""
        pid_b = tl.program_id(0).to(tl.int64)
        pid_d = tl.program_id(1).to(tl.int64)

        offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B).to(tl.int64)
        mask_b = offs_b < B
        lbl = tl.load(labels_ptr + offs_b, mask=mask_b, other=0).to(tl.int64)
        src_rows = (
            tl.load(row_idx_ptr + offs_b, mask=mask_b, other=0).to(tl.int64)
            if HAS_INDICES
            else offs_b
        )

        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D).to(tl.int64)
        mask_d = offs_d < D
        mask_bd = mask_b[:, None] & mask_d[None, :]

        vals = tl.load(
            x_ptr + src_rows[:, None] * stride_xb + offs_d[None, :] * stride_xd,
            mask=mask_bd,
            other=0.0,
        ).to(tl.float32)

        tl.atomic_add(
            sums_ptr + lbl[:, None] * stride_sumsk + offs_d[None, :],
            vals,
            mask=mask_bd,
            sem="relaxed",
        )
        if pid_d == 0:
            tl.atomic_add(counts_ptr + lbl, 1.0, mask=mask_b, sem="relaxed")
            if HAS_SSE:
                sqd = tl.load(sq_dists_ptr + offs_b, mask=mask_b, other=0.0).to(tl.float32)
                tl.atomic_add(sse_ptr + lbl, sqd, mask=mask_b, sem="relaxed")

    @triton.jit(do_not_specialize=["B", "D", "stride_xb", "stride_xd", "stride_ck"])
    def _super_scatter_oja_kernel(
        x_ptr,
        row_idx_ptr,
        labels_ptr,
        sq_dists_ptr,
        centroids_ptr,
        p_hat_ptr,
        sums_ptr,
        counts_ptr,
        sse_ptr,
        g_sums_ptr,
        B,
        D,
        stride_xb,
        stride_xd,
        stride_ck,
        HAS_INDICES: tl.constexpr,
        HAS_SSE: tl.constexpr,
        FULL_D: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Single-pass gather + Oja residual covariance projection + scatter without [B, D] temporaries."""
        pid_b = tl.program_id(0).to(tl.int64)
        offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B).to(tl.int64)
        mask_b = offs_b < B

        lbl = tl.load(labels_ptr + offs_b, mask=mask_b, other=0).to(tl.int64)
        src_rows = (
            tl.load(row_idx_ptr + offs_b, mask=mask_b, other=0).to(tl.int64)
            if HAS_INDICES
            else offs_b
        )

        tl.atomic_add(counts_ptr + lbl, 1.0, mask=mask_b, sem="relaxed")
        if HAS_SSE:
            sqd = tl.load(sq_dists_ptr + offs_b, mask=mask_b, other=0.0).to(tl.float32)
            tl.atomic_add(sse_ptr + lbl, sqd, mask=mask_b, sem="relaxed")

        if FULL_D:
            offs_d = tl.arange(0, BLOCK_D).to(tl.int64)
            mask_d = offs_d < D
            mask_bd = mask_b[:, None] & mask_d[None, :]

            x_val = tl.load(
                x_ptr + src_rows[:, None] * stride_xb + offs_d[None, :] * stride_xd,
                mask=mask_bd,
                other=0.0,
            ).to(tl.float32)
            c_val = tl.load(
                centroids_ptr + lbl[:, None] * stride_ck + offs_d[None, :],
                mask=mask_bd,
                other=0.0,
            ).to(tl.float32)
            p_val = tl.load(
                p_hat_ptr + lbl[:, None] * stride_ck + offs_d[None, :],
                mask=mask_bd,
                other=0.0,
            ).to(tl.float32)

            r_val = tl.where(mask_bd, x_val - c_val, 0.0)
            proj = tl.sum(r_val * p_val, axis=1)

            tl.atomic_add(
                sums_ptr + lbl[:, None] * stride_ck + offs_d[None, :],
                x_val,
                mask=mask_bd,
                sem="relaxed",
            )
            tl.atomic_add(
                g_sums_ptr + lbl[:, None] * stride_ck + offs_d[None, :],
                r_val * proj[:, None],
                mask=mask_bd,
                sem="relaxed",
            )
        else:
            proj = tl.zeros((BLOCK_B,), dtype=tl.float32)
            for d0 in range(0, D, BLOCK_D):
                offs_d = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
                mask_d = offs_d < D
                mask_bd = mask_b[:, None] & mask_d[None, :]

                x_tile = tl.load(
                    x_ptr + src_rows[:, None] * stride_xb + offs_d[None, :] * stride_xd,
                    mask=mask_bd,
                    other=0.0,
                ).to(tl.float32)
                c_tile = tl.load(
                    centroids_ptr + lbl[:, None] * stride_ck + offs_d[None, :],
                    mask=mask_bd,
                    other=0.0,
                ).to(tl.float32)
                p_tile = tl.load(
                    p_hat_ptr + lbl[:, None] * stride_ck + offs_d[None, :],
                    mask=mask_bd,
                    other=0.0,
                ).to(tl.float32)

                r_tile = tl.where(mask_bd, x_tile - c_tile, 0.0)
                proj += tl.sum(r_tile * p_tile, axis=1)
                tl.atomic_add(
                    sums_ptr + lbl[:, None] * stride_ck + offs_d[None, :],
                    x_tile,
                    mask=mask_bd,
                    sem="relaxed",
                )

            for d0 in range(0, D, BLOCK_D):
                offs_d = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
                mask_d = offs_d < D
                mask_bd = mask_b[:, None] & mask_d[None, :]

                x_tile = tl.load(
                    x_ptr + src_rows[:, None] * stride_xb + offs_d[None, :] * stride_xd,
                    mask=mask_bd,
                    other=0.0,
                ).to(tl.float32)
                c_tile = tl.load(
                    centroids_ptr + lbl[:, None] * stride_ck + offs_d[None, :],
                    mask=mask_bd,
                    other=0.0,
                ).to(tl.float32)
                r_tile = tl.where(mask_bd, x_tile - c_tile, 0.0)
                tl.atomic_add(
                    g_sums_ptr + lbl[:, None] * stride_ck + offs_d[None, :],
                    r_tile * proj[:, None],
                    mask=mask_bd,
                    sem="relaxed",
                )

    @triton.jit(do_not_specialize=["B", "K", "D", "stride_xb", "stride_xd", "stride_ck", "stride_cd"])
    def _centered_top3_kernel(
        xc_ptr,
        row_idx_ptr,
        cc_ptr,
        xn_ptr,
        cn_ptr,
        out_d_ptr,
        out_i_ptr,
        B,
        K,
        D,
        stride_xb,
        stride_xd,
        stride_ck,
        stride_cd,
        BLOCK_B: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
        DTYPE_MODE: tl.constexpr,  # 0=IEEE_FP32, 1=BF16, 2=FP16
        HAS_INDICES: tl.constexpr,
    ):
        """Exact top-3 nearest centroids in SRAM without [B, K] HBM materialization."""
        pid_b = tl.program_id(0).to(tl.int64)
        offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B).to(tl.int64)
        mask_b = offs_b < B
        src_rows = (
            tl.load(row_idx_ptr + offs_b, mask=mask_b, other=0).to(tl.int64)
            if HAS_INDICES
            else offs_b
        )

        best1_s = tl.full((BLOCK_B,), float("inf"), dtype=tl.float32)
        best2_s = tl.full((BLOCK_B,), float("inf"), dtype=tl.float32)
        best3_s = tl.full((BLOCK_B,), float("inf"), dtype=tl.float32)
        best1_i = tl.full((BLOCK_B,), -1, dtype=tl.int32)
        best2_i = tl.full((BLOCK_B,), -1, dtype=tl.int32)
        best3_i = tl.full((BLOCK_B,), -1, dtype=tl.int32)

        local_k = tl.arange(0, BLOCK_K)

        for k0 in range(0, K, BLOCK_K):
            offs_k = (k0 + local_k).to(tl.int64)
            mask_k = offs_k < K
            dots = tl.zeros((BLOCK_B, BLOCK_K), dtype=tl.float32)

            for d0 in range(0, D, BLOCK_D):
                offs_d = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
                mask_d = offs_d < D
                x_tile = tl.load(
                    xc_ptr + src_rows[:, None] * stride_xb + offs_d[None, :] * stride_xd,
                    mask=mask_b[:, None] & mask_d[None, :],
                    other=0.0,
                )
                c_tile = tl.load(
                    cc_ptr + offs_k[:, None] * stride_ck + offs_d[None, :] * stride_cd,
                    mask=mask_k[:, None] & mask_d[None, :],
                    other=0.0,
                )
                if DTYPE_MODE == 1:
                    dots = tl.dot(
                        x_tile.to(tl.bfloat16),
                        tl.trans(c_tile.to(tl.bfloat16)),
                        acc=dots,
                        out_dtype=tl.float32,
                    )
                elif DTYPE_MODE == 2:
                    dots = tl.dot(
                        x_tile.to(tl.float16),
                        tl.trans(c_tile.to(tl.float16)),
                        acc=dots,
                        out_dtype=tl.float32,
                    )
                else:
                    dots = tl.dot(
                        x_tile.to(tl.float32),
                        tl.trans(c_tile.to(tl.float32)),
                        acc=dots,
                        input_precision="ieee",
                        out_dtype=tl.float32,
                    )

            cn = tl.load(cn_ptr + offs_k, mask=mask_k, other=float("inf")).to(tl.float32)
            scores1 = tl.where(mask_k[None, :], cn[None, :] - 2.0 * dots, float("inf"))

            m1, loc1 = tl.min(
                scores1, axis=1, return_indices=True, return_indices_tie_break_left=True
            )
            scores2 = tl.where(local_k[None, :] == loc1[:, None], float("inf"), scores1)
            m2, loc2 = tl.min(
                scores2, axis=1, return_indices=True, return_indices_tie_break_left=True
            )
            scores3 = tl.where(local_k[None, :] == loc2[:, None], float("inf"), scores2)
            m3, loc3 = tl.min(
                scores3, axis=1, return_indices=True, return_indices_tie_break_left=True
            )

            id1 = k0 + loc1.to(tl.int32)
            id2 = k0 + loc2.to(tl.int32)
            id3 = k0 + loc3.to(tl.int32)

            # Insert candidate 1 into sorted (best1, best2, best3)
            c1_b1 = m1 < best1_s
            c1_b2 = (~c1_b1) & (m1 < best2_s)
            c1_b3 = (~c1_b1) & (~c1_b2) & (m1 < best3_s)
            best3_s = tl.where(c1_b1 | c1_b2, best2_s, tl.where(c1_b3, m1, best3_s))
            best3_i = tl.where(c1_b1 | c1_b2, best2_i, tl.where(c1_b3, id1, best3_i))
            best2_s = tl.where(c1_b1, best1_s, tl.where(c1_b2, m1, best2_s))
            best2_i = tl.where(c1_b1, best1_i, tl.where(c1_b2, id1, best2_i))
            best1_s = tl.where(c1_b1, m1, best1_s)
            best1_i = tl.where(c1_b1, id1, best1_i)

            # Insert candidate 2 (m2 >= m1, so can only enter slot 2 or slot 3)
            valid2 = m2 < float("inf")
            c2_b2 = valid2 & (m2 < best2_s)
            c2_b3 = valid2 & (~c2_b2) & (m2 < best3_s)
            best3_s = tl.where(c2_b2, best2_s, tl.where(c2_b3, m2, best3_s))
            best3_i = tl.where(c2_b2, best2_i, tl.where(c2_b3, id2, best3_i))
            best2_s = tl.where(c2_b2, m2, best2_s)
            best2_i = tl.where(c2_b2, id2, best2_i)

            # Insert candidate 3 (m3 >= m2, so can only enter slot 3)
            c3_b3 = (m3 < float("inf")) & (m3 < best3_s)
            best3_s = tl.where(c3_b3, m3, best3_s)
            best3_i = tl.where(c3_b3, id3, best3_i)

        xn = tl.load(xn_ptr + offs_b, mask=mask_b, other=0.0).to(tl.float32)
        d1 = tl.maximum(best1_s + xn, 0.0)
        d2 = tl.where(best2_i >= 0, tl.maximum(best2_s + xn, 0.0), float("inf"))
        d3 = tl.where(best3_i >= 0, tl.maximum(best3_s + xn, 0.0), float("inf"))

        out_base = offs_b * 3
        tl.store(out_d_ptr + out_base + 0, d1, mask=mask_b)
        tl.store(out_d_ptr + out_base + 1, d2, mask=mask_b)
        tl.store(out_d_ptr + out_base + 2, d3, mask=mask_b)
        tl.store(out_i_ptr + out_base + 0, best1_i.to(tl.int64), mask=mask_b)
        tl.store(out_i_ptr + out_base + 1, best2_i.to(tl.int64), mask=mask_b)
        tl.store(out_i_ptr + out_base + 2, best3_i.to(tl.int64), mask=mask_b)


# ==============================================================================
# 2. Coordinate-Shift-Safe Seeding, Schedules, Twin-Safe Huygens, & Fast Top-3
# ==============================================================================


@torch.no_grad()
def compute_dataset_origin_fp64(x: torch.Tensor, chunk_rows: int = 262144) -> torch.Tensor:
    """Compute exact FP64 per-feature mean `(1, D)` in bounded memory with zero host sync."""
    n, d = x.shape
    if n == 0:
        return torch.zeros((1, d), dtype=torch.float64, device=x.device)
    acc = torch.zeros((1, d), dtype=torch.float64, device=x.device)
    for r0 in range(0, n, chunk_rows):
        r1 = min(r0 + chunk_rows, n)
        acc.add_(x[r0:r1].double().sum(dim=0, keepdim=True))
    return acc / float(n)


@torch.no_grad()
def subsampled_d15_init_centered(
    x: torch.Tensor,
    k: int,
    seed: int,
    rounds: int = 4,
    power: float = 1.5,
) -> torch.Tensor:
    """Shift-invariant multi-round subsampled `(D^2)^power` seeding (`suayptalha/s-means` C1).

    Matches `smeans._recipe.subsampled_d15_init` (`m = min(n, max(4*k, 512))`, `k0 = max(8, k//8)`,
    `multinomial(dmin_sq ** power)`), while centering coordinates around the subsample mean so
    global shifts (`+10^4` / `+20,000`) suffer zero FP32 catastrophic cancellation.
    """
    n = int(x.shape[0])
    if n < k:
        raise ValueError(f"n_samples ({n}) must be >= k ({k})")
    if n == k:
        return x.float().clone().contiguous()

    gen = torch.Generator(device=x.device)
    gen.manual_seed(int(seed) % (2**31 - 1))

    m = min(n, max(k * 4, 512))
    perm = torch.randperm(n, generator=gen, device=x.device)[:m]
    xs_raw = x[perm]
    origin = xs_raw.double().mean(dim=0, keepdim=True)
    xs_c64 = xs_raw.double() - origin
    # Scale-normalize centered coordinates so FP32 square & power never overflow/underflow
    # even at extreme feature scales (1e-30 .. 1e+100)
    scale_max = xs_c64.abs().max().clamp_min(1e-30)
    xs_c = (xs_c64 / scale_max).float().contiguous()
    xsn = xs_c.square().sum(dim=1)

    nr = max(2, int(rounds))
    k0 = min(k, max(8, k // (nr * 2)))
    chosen_idx = torch.empty((k,), dtype=torch.int64, device=x.device)
    chosen_idx[:k0] = torch.arange(k0, device=x.device, dtype=torch.int64)
    cur_k = k0

    c0_c = xs_c[:k0]
    c0n = c0_c.square().sum(dim=1)
    dmin = (xsn[:, None] + c0n[None, :] - 2.0 * (xs_c @ c0_c.t())).clamp_min_(0.0).min(dim=1).values
    dmin[:k0] = 0.0

    step_k = max(1, (k - k0 + nr - 2) // max(1, nr - 1))
    avail = torch.ones((m,), dtype=torch.bool, device=x.device)
    avail[:k0] = False

    for _ in range(nr - 1):
        if cur_k >= k:
            break
        cand_n = min(step_k, k - cur_k)
        dmax = torch.where(avail, dmin, torch.zeros_like(dmin)).max().clamp_min(1e-30)
        d_rel = (dmin / dmax).clamp(0.0, 1.0)
        raw_probs = (d_rel.clamp_min(1e-12) ** float(power)) + 1e-12
        probs = torch.where(avail, raw_probs, torch.zeros_like(raw_probs))
        # Guard degenerate all-zero / fewer-available case safely
        tot_p = probs.sum()
        fallback_p = torch.where(avail, torch.ones_like(raw_probs), torch.zeros_like(raw_probs))
        probs = torch.where(tot_p > 0.0, probs, fallback_p)
        picked = torch.multinomial(probs, cand_n, replacement=False, generator=gen)
        chosen_idx[cur_k : cur_k + cand_n] = picked
        avail[picked] = False

        new_c = xs_c[picked]
        new_n = new_c.square().sum(dim=1)
        new_d = (xsn[:, None] + new_n[None, :] - 2.0 * (xs_c @ new_c.t())).clamp_min_(0.0).min(dim=1).values
        dmin = torch.minimum(dmin, new_d)
        dmin[picked] = 0.0
        cur_k += cand_n

    return xs_raw[chosen_idx[:k]].float().clone().contiguous()


def super_scheduled_alpha(
    alpha0: float,
    frac: float,
    schedule_mode: str = "emeans_cosine",
    alpha_min_factor: float = ALPHA_MIN_FACTOR,
    alpha_floor: float = 0.06,
    cosine_power: float = 0.92,
) -> float:
    """Compute step size `alpha(t)` across `emeans_cosine`, `power_cosine` (`s-means` C2), or `constant`."""
    f = min(1.0, max(0.0, float(frac)))
    if schedule_mode == "constant":
        return float(alpha0)
    cos_half = 0.5 * (1.0 + math.cos(math.pi * f))
    if schedule_mode == "power_cosine":
        return float(alpha0 * (alpha_floor + (1.0 - alpha_floor) * (cos_half ** cosine_power)))
    if schedule_mode == "emeans_cosine":
        return float(alpha0 * (alpha_min_factor + (1.0 - alpha_min_factor) * cos_half))
    raise ValueError(f"Unknown schedule_mode={schedule_mode!r}")


@torch.no_grad()
def _all_pairs_centered_nn(
    centroids: torch.Tensor,
    chunk_size: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute exact `(d_nn2, nn_idx)` over all `K` centroids with `m != j` self-masking (`0` host syncs)."""
    k = centroids.shape[0]
    device = centroids.device
    if k == 0:
        return (
            torch.empty((0,), dtype=torch.float32, device=device),
            torch.empty((0,), dtype=torch.int64, device=device),
        )
    if k == 1:
        return (
            torch.full((1,), float("inf"), dtype=torch.float32, device=device),
            torch.full((1,), -1, dtype=torch.int64, device=device),
        )

    origin = centroids.float().mean(dim=0, keepdim=True)
    kc = (centroids.float() - origin).contiguous()
    kc_norm = kc.square().sum(dim=1)

    d_nn2 = torch.empty((k,), dtype=torch.float32, device=device)
    nn_idx = torch.empty((k,), dtype=torch.int64, device=device)
    all_cols = torch.arange(k, device=device)

    for s in range(0, k, chunk_size):
        e = min(s + chunk_size, k)
        qc = kc[s:e]
        qn = kc_norm[s:e]
        rows = torch.arange(s, e, device=device)
        d2 = torch.addmm(kc_norm[None, :], qc, kc.t(), alpha=-2.0)
        d2.add_(qn[:, None]).clamp_min_(0.0)
        d2 = torch.where(rows[:, None] == all_cols[None, :], float("inf"), d2)
        vals, idxs = d2.min(dim=1)
        d_nn2[s:e] = vals
        nn_idx[s:e] = idxs
    return d_nn2, nn_idx


@torch.no_grad()
def twin_safe_huygens_protection(
    centroids: torch.Tensor,
    low_mask: torch.Tensor,
    usage: torch.Tensor,
    survival_mass: torch.Tensor | None,
    thresh: torch.Tensor,
    ref_distortion_mean: torch.Tensor,
    gamma_protect: float,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Compute boolean `protected_mask` `(K,)` with Twin-Safe Huygens Rare-Mode Leader Election.

    Resolves simultaneous twin-centroid assassination on isolated rare modes:
      - If a single centroid `j` sits on an isolated rare mode (`low_mask[j] == True`), its
        nearest surviving neighbor distance `d2_surv[j]` equals `d_nn2[j]`, matching `EMeansPP`
        100% identically (`eff_mass[j] * d_nn2[j] >= gamma_protect * mean(E)`).
      - If 2+ twin centroids `(j1, j2, ...)` sit on the SAME isolated rare mode and all fall
        below `thresh` on the same revival check, `||c_j1 - c_j2||^2 ~ 0` would trick naive
        `d_nn2` into killing ALL twins simultaneously! Here we measure distance `d2_surv` to the
        nearest NON-low centroid, pool co-located low-mask twins (`d2 <= 0.25 * min(d2_surv)`),
        and protect ONLY the highest-`(usage, -index)` leader per isolated rare mode, so 1 guardian
        centroid always survives while redundant duplicate twins are cleanly recycled!
    """
    k = centroids.shape[0]
    device = centroids.device
    if k < 2 or gamma_protect <= 0.0:
        return torch.zeros((k,), dtype=torch.bool, device=device)

    origin = centroids.float().mean(dim=0, keepdim=True)
    kc = (centroids.float() - origin).contiguous()
    kc_norm = kc.square().sum(dim=1)

    if survival_mass is not None:
        eff_mass = torch.maximum(usage, torch.minimum(survival_mass, thresh))
    else:
        eff_mass = usage

    protect_cutoff = gamma_protect * ref_distortion_mean.clamp_min(1e-30)
    surv_mask = ~low_mask
    has_any_surv = surv_mask.any()
    all_ids = torch.arange(k, device=device)

    # Fast sync-free tiled scan over K when K <= 4096 (no dynamic nonzero sync!)
    if k <= 4096:
        d2 = torch.addmm(kc_norm[None, :], kc, kc.t(), alpha=-2.0)
        d2.add_(kc_norm[:, None]).clamp_min_(0.0)
        diag_mask = all_ids[:, None] == all_ids[None, :]
        d2_no_self = torch.where(diag_mask, float("inf"), d2)
        d_nn2_any = d2_no_self.min(dim=1).values

        # Exact Huygens pair-merge criterion: two low-usage centroids `(a, b)` are co-located
        # redundant twins iff merging one into the other costs strictly less than `protect_cutoff`!
        # Distinct rare modes (`max(m_a, m_b) * d2 >= protect_cutoff`) are NEVER merged together.
        both_low = low_mask[:, None] & low_mask[None, :]
        max_pair_mass = torch.maximum(eff_mass[:, None], eff_mass[None, :])
        co_located = diag_mask | (both_low & ((max_pair_mass * d2) < protect_cutoff))

        # External nearest neighbor outside `co_located[a]` (includes surviving bulk & distinct rare modes!)
        d2_ext_raw = torch.where(co_located, float("inf"), d2_no_self).min(dim=1).values
        d2_ext = torch.where(torch.isfinite(d2_ext_raw), d2_ext_raw, d_nn2_any)

        group_mass = (co_located & both_low).to(torch.float32) @ eff_mass  # (K,)
        higher_peer = (
            co_located
            & both_low
            & (~diag_mask)
            & (
                (usage[None, :] > usage[:, None])
                | ((usage[None, :] == usage[:, None]) & (all_ids[None, :] < all_ids[:, None]))
            )
        )
        is_leader = low_mask & (~higher_peer.any(dim=1))
        removal_cost = group_mass * d2_ext.clamp_max(1e30)
        return is_leader & (removal_cost >= protect_cutoff)

    # Memory-bounded path for ultra-large K > 4096 (e.g., SIFT1M K = 16,384)
    cand_ids = torch.nonzero(low_mask, as_tuple=False).reshape(-1)
    c_num = int(cand_ids.numel())
    if c_num == 0:
        return torch.zeros((k,), dtype=torch.bool, device=device)

    qc = kc[cand_ids]
    qn = kc_norm[cand_ids]
    d2_any = torch.full((c_num,), float("inf"), dtype=torch.float32, device=device)
    d2_surv = torch.full((c_num,), float("inf"), dtype=torch.float32, device=device)
    for k0 in range(0, k, chunk_size):
        k1 = min(k0 + chunk_size, k)
        blk = kc[k0:k1]
        bn = kc_norm[k0:k1]
        cols = all_ids[k0:k1]
        d_blk = torch.addmm(bn[None, :], qc, blk.t(), alpha=-2.0).add_(qn[:, None]).clamp_min_(0.0)
        d_blk = torch.where(cand_ids[:, None] == cols[None, :], float("inf"), d_blk)
        d2_any = torch.minimum(d2_any, d_blk.min(dim=1).values)
        d_blk_s = torch.where(surv_mask[k0:k1][None, :], d_blk, float("inf"))
        d2_surv = torch.minimum(d2_surv, d_blk_s.min(dim=1).values)

    d2_cc = torch.addmm(qn[None, :], qc, qc.t(), alpha=-2.0).add_(qn[:, None]).clamp_min_(0.0)
    c_arange = torch.arange(c_num, device=device)
    c_diag = c_arange[:, None] == c_arange[None, :]
    cand_eff_m = eff_mass[cand_ids]
    max_cand_m = torch.maximum(cand_eff_m[:, None], cand_eff_m[None, :])
    co_loc = c_diag | ((max_cand_m * d2_cc) < protect_cutoff)

    d2_other_low = torch.where(co_loc, float("inf"), d2_cc).min(dim=1).values
    d2_ext_c = torch.minimum(d2_surv, d2_other_low)
    d2_eff = torch.where(torch.isfinite(d2_ext_c), d2_ext_c, d2_any)

    grp_m = co_loc.to(torch.float32) @ cand_eff_m
    cand_u = usage[cand_ids]
    hi_peer = (
        co_loc
        & (~c_diag)
        & (
            (cand_u[None, :] > cand_u[:, None])
            | ((cand_u[None, :] == cand_u[:, None]) & (cand_ids[None, :] < cand_ids[:, None]))
        )
    )
    prot_loc = (~hi_peer.any(dim=1)) & ((grp_m * d2_eff.clamp_max(1e30)) >= protect_cutoff)
    out = torch.zeros((k,), dtype=torch.bool, device=device)
    out[cand_ids] = prot_loc
    return out


@torch.no_grad()
def fast_top3_centered_assign(
    x: torch.Tensor,
    centroids: torch.Tensor,
    compute_dtype: torch.dtype,
    already_centered: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute exact centered top-3 distances & distinct indices using SRAM Triton kernel on CUDA.

    When `already_centered=True` (guaranteed inside `HybridSEMeans.fit()` via Sterbenz mean
    centering) and `compute_dtype` is `bfloat16`/`float16`, executes directly on BF16/FP16
    Tensor Cores (`~7x` faster than IEEE FP32 while matching Appendix I.1 represented norms).
    Returns `(top_dists, top_idx, origin, xc, xn, cc)`.
    """
    bsz, d = x.shape
    k = centroids.shape[0]
    device = x.device
    if k < 3 or bsz == 0 or compute_dtype == torch.float64 or (not x.is_cuda) or (not HAS_TRITON) or d < 16:
        td, ti, orig = top3_represented_assign_centered(x, centroids, compute_dtype)
        if compute_dtype == torch.float64:
            orig64 = x.double().mean(dim=0, keepdim=True)
            xc = (x.double() - orig64).float().contiguous()
            cc = (centroids.double() - orig64).float().contiguous()
        else:
            rep_dt = (
                compute_dtype
                if compute_dtype in (torch.float32, torch.bfloat16, torch.float16)
                else torch.float32
            )
            xc = (x.to(rep_dt).float() - orig).contiguous()
            cc = (centroids.to(rep_dt).float() - orig).contiguous()
        xn = xc.square().sum(dim=1)
        return td, ti, orig, xc, xn, cc

    rep_dtype = (
        compute_dtype
        if compute_dtype in (torch.float32, torch.bfloat16, torch.float16)
        else torch.float32
    )
    use_tc_lowp = already_centered and rep_dtype in (torch.bfloat16, torch.float16)
    if use_tc_lowp:
        origin = torch.zeros((1, d), dtype=torch.float32, device=device)
        x_ker = x if (x.dtype == rep_dtype and x.is_contiguous()) else x.to(rep_dtype).contiguous()
        c_ker = (
            centroids
            if (centroids.dtype == rep_dtype and centroids.is_contiguous())
            else centroids.to(rep_dtype).contiguous()
        )
        xc = x_ker.float().contiguous()
        cc = c_ker.float().contiguous()
        d_mode = 1 if rep_dtype == torch.bfloat16 else 2
        block_b = 64
        block_k = 128 if k >= 128 else 64
        block_d = 64
        n_stages = 3
    else:
        x_rep_f = x.to(rep_dtype).float()
        c_rep_f = centroids.to(rep_dtype).float()
        origin = x_rep_f.mean(dim=0, keepdim=True)
        xc = (x_rep_f - origin).contiguous()
        cc = (c_rep_f - origin).contiguous()
        x_ker = xc
        c_ker = cc
        d_mode = 0
        block_b = 64
        block_k = 64
        block_d = 32
        n_stages = 2

    xn = xc.square().sum(dim=1).contiguous()
    cn = cc.square().sum(dim=1).contiguous()

    top_d = torch.empty((bsz, 3), dtype=torch.float32, device=device)
    top_i = torch.empty((bsz, 3), dtype=torch.int64, device=device)

    with torch.cuda.device(device):
        _centered_top3_kernel[(triton.cdiv(bsz, block_b),)](
            x_ker,
            None,
            c_ker,
            xn,
            cn,
            top_d,
            top_i,
            bsz,
            k,
            d,
            x_ker.stride(0),
            x_ker.stride(1),
            c_ker.stride(0),
            c_ker.stride(1),
            BLOCK_B=block_b,
            BLOCK_K=block_k,
            BLOCK_D=block_d,
            DTYPE_MODE=d_mode,
            HAS_INDICES=False,
            num_warps=4,
            num_stages=n_stages,
        )
    return top_d, top_i, origin, xc, xn, cc


@torch.no_grad()
def incremental_refresh_top3_and_nn(
    xc: torch.Tensor,
    xn: torch.Tensor,
    cc: torch.Tensor,
    cn: torch.Tensor,
    top_d: torch.Tensor,
    top_i: torch.Tensor,
    d_nn2: torch.Tensor,
    nn_idx: torch.Tensor,
    s_idx: int,
    d_idx: int,
    new_s_c: torch.Tensor,
    new_d_c: torch.Tensor,
) -> None:
    """Update `(cc, cn, top_d, top_i, d_nn2, nn_idx)` IN-PLACE after replacing `(s_idx, d_idx)`.

    Mathematically exact:
      1. Rows where neither `s_idx` nor `d_idx` was in old `top_i` (`~99.7%` of rows at K=2048)
         simply merge `[dist(x, new_s), dist(x, new_d)]` into their existing 3 slots in O(B).
      2. Only the tiny subset of affected rows (`(top_i == s_idx) | (top_i == d_idx)`) re-queries
         `cc` (`O(B_affected * K * D)`).
      3. Centroid nearest-neighbor `(d_nn2, nn_idx)` updates in `O(K * D)` plus exact refresh
         solely on centroids whose previous NN was `s_idx` or `d_idx`.
    """
    bsz = xc.shape[0]
    k = cc.shape[0]
    device = xc.device

    cc[s_idx] = new_s_c
    cc[d_idx] = new_d_c
    ns_norm = new_s_c.square().sum()
    nd_norm = new_d_c.square().sum()
    cn[s_idx] = ns_norm
    cn[d_idx] = nd_norm

    # 1. Exact incremental top-3 update on `xc`
    affected_mask = ((top_i == s_idx) | (top_i == d_idx)).any(dim=1)
    pair_c = torch.stack([new_s_c, new_d_c], dim=0)  # (2, D)
    pair_n = torch.stack([ns_norm, nd_norm], dim=0)  # (2,)
    d_pair = (torch.addmm(pair_n[None, :], xc, pair_c.t(), alpha=-2.0) + xn[:, None]).clamp_min_(0.0)

    pair_ids = torch.tensor([s_idx, d_idx], dtype=torch.int64, device=device).expand(bsz, 2)
    comb_d = torch.cat([top_d, d_pair], dim=1)  # (B, 5)
    comb_i = torch.cat([top_i, pair_ids], dim=1)  # (B, 5)
    # Sort by index first so equal-distance ties deterministically prefer smaller centroid index
    idx_order = torch.argsort(comb_i, dim=1)
    comb_i_sorted = torch.gather(comb_i, 1, idx_order)
    comb_d_sorted = torch.gather(comb_d, 1, idx_order)
    mer_d, mer_pos = torch.topk(comb_d_sorted, k=3, dim=1, largest=False, sorted=True)
    mer_i = torch.gather(comb_i_sorted, 1, mer_pos)

    aff_rows = torch.nonzero(affected_mask, as_tuple=False).reshape(-1)
    if aff_rows.numel() > 0:
        xa = xc[aff_rows]
        xna = xn[aff_rows]
        # Chunked exact top-3 for the ~few dozen affected rows
        m_a = int(aff_rows.shape[0])
        best_sc = torch.full((m_a, 3), float("inf"), dtype=torch.float32, device=device)
        best_id = torch.full((m_a, 3), -1, dtype=torch.int64, device=device)
        for k0 in range(0, k, 4096):
            k1 = min(k0 + 4096, k)
            sc_blk = torch.addmm(cn[k0:k1][None, :], xa, cc[k0:k1].t(), alpha=-2.0)
            q_k = min(3, k1 - k0)
            b_sc, b_id_loc = torch.topk(sc_blk, k=q_k, dim=1, largest=False, sorted=True)
            b_id = b_id_loc + k0
            if k0 == 0 and q_k == 3:
                best_sc, best_id = b_sc, b_id
            else:
                cs = torch.cat([best_sc, b_sc], dim=1)
                ci = torch.cat([best_id, b_id], dim=1)
                best_sc, mp = torch.topk(cs, k=3, dim=1, largest=False, sorted=True)
                best_id = torch.gather(ci, 1, mp)
        mer_d[aff_rows] = (best_sc + xna[:, None]).clamp_min_(0.0)
        mer_i[aff_rows] = best_id

    top_d.copy_(mer_d)
    top_i.copy_(mer_i)

    # 2. Exact incremental centroid nearest-neighbor `(d_nn2, nn_idx)` update
    all_ids = torch.arange(k, device=device)
    # Distances from `[s_idx, d_idx]` to all K centroids: (2, K)
    d2_pair_k = (torch.addmm(cn[None, :], pair_c, cc.t(), alpha=-2.0) + pair_n[:, None]).clamp_min_(0.0)
    d2_pair_k[0, s_idx] = float("inf")
    d2_pair_k[1, d_idx] = float("inf")

    # New NN for `s_idx` and `d_idx` themselves
    s_min, s_arg = d2_pair_k[0].min(dim=0)
    d_min, d_arg = d2_pair_k[1].min(dim=0)

    # Centroids whose previous NN was `s_idx` or `d_idx` (excluding `s_idx, d_idx` which we already have)
    stale_nn = (nn_idx == s_idx) | (nn_idx == d_idx)
    stale_nn[s_idx] = False
    stale_nn[d_idx] = False

    # For non-stale centroids, update with distances to `new_s` and `new_d`
    cand_nn_d = torch.stack([d_nn2, d2_pair_k[0], d2_pair_k[1]], dim=1)  # (K, 3)
    cand_nn_i = torch.stack(
        [
            nn_idx,
            torch.full_like(nn_idx, s_idx),
            torch.full_like(nn_idx, d_idx),
        ],
        dim=1,
    )
    # Mask self on rows s_idx, d_idx safely (overwritten below anyway)
    upd_d, upd_pos = cand_nn_d.min(dim=1)
    upd_i = torch.gather(cand_nn_i, 1, upd_pos[:, None]).squeeze(1)

    stale_ids = torch.nonzero(stale_nn, as_tuple=False).reshape(-1)
    if stale_ids.numel() > 0:
        qs = cc[stale_ids]
        qsn = cn[stale_ids]
        d_st = (torch.addmm(cn[None, :], qs, cc.t(), alpha=-2.0) + qsn[:, None]).clamp_min_(0.0)
        d_st = torch.where(stale_ids[:, None] == all_ids[None, :], float("inf"), d_st)
        st_min, st_arg = d_st.min(dim=1)
        upd_d[stale_ids] = st_min
        upd_i[stale_ids] = st_arg

    upd_d[s_idx] = s_min
    upd_i[s_idx] = s_arg
    upd_d[d_idx] = d_min
    upd_i[d_idx] = d_arg

    d_nn2.copy_(upd_d)
    nn_idx.copy_(upd_i)


@torch.no_grad()
def run_fast_head_edit_step(
    x_batch: torch.Tensor,
    centroids: torch.Tensor,
    usage: torch.Tensor,
    distortion: torch.Tensor | None,
    ref_distortion_mean: torch.Tensor | None,
    survival_mass: torch.Tensor | None,
    prin_vec: torch.Tensor | None,
    slow_centroids: torch.Tensor | None,
    slow_usage: torch.Tensor | None,
    pass_counts: torch.Tensor | None,
    pass_counts_sq: torch.Tensor | None,
    pass_sums: torch.Tensor | None,
    pass_sse: torch.Tensor | None,
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
    already_centered: bool = False,
) -> int:
    """Fast, zero-Python-`k`-loop P/Q forward-verified Hadamard transplant (`N-Track-1`)."""
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
    scale_full = float(bsz) / float(half)

    top_d, top_i, origin, xc_all, x_norm_all, cc = fast_top3_centered_assign(
        x_batch, centroids, compute_dtype, already_centered=already_centered
    )
    cn = cc.square().sum(dim=1)
    origin64 = x_batch.double().mean(dim=0, keepdim=True) if rep_dtype == torch.float64 else None

    # Compute all-pairs centered NN `(d_nn2, nn_idx)` ONCE and update incrementally on commits!
    d_nn2, nn_idx = _all_pairs_centered_nn(centroids)

    sse_p = torch.empty((k,), dtype=torch.float32, device=device)
    dormancy_p = torch.empty((k,), dtype=torch.float32, device=device)
    used_mask = torch.zeros((k,), dtype=torch.bool, device=device)

    def _score_donors_and_recipients() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        d1p = top_d[:half, 0]
        d2p = top_d[:half, 1]
        a1p = top_i[:half, 0]
        a1q = top_i[half:, 0]

        cp = torch.bincount(a1p, minlength=k).float()
        cq = torch.bincount(a1q, minlength=k).float()
        sse_p.zero_()
        sse_p.index_add_(0, a1p, d1p)
        dormancy_p.zero_()
        dormancy_p.index_add_(0, a1p, (d2p - d1p).clamp_min(0.0))
        b_rem = dormancy_p * scale_full

        thresh = usage.mean() * max(unused_threshold, 0.1)
        if survival_mass is not None:
            eff_mass = torch.maximum(usage, torch.minimum(survival_mass, thresh))
        else:
            eff_mass = usage
        h_cost = eff_mass * d_nn2.clamp_max(1e30)
        e_rem = torch.maximum(b_rem, h_cost)

        f_mask = ((step - last_edit_step) >= protect_steps) & (~used_mask)
        if use_huygens_protection and ref_distortion_mean is not None and gamma_protect > 0.0:
            prot_cutoff = gamma_protect * ref_distortion_mean.clamp_min(1e-30)
            rare_shield = (usage < thresh) & (h_cost >= prot_cutoff)
            f_mask = f_mask & (~rare_shield)

        don_elig = f_mask & (cp >= 4.0) & (cq >= 2.0) & (sse_p > 1e-12)
        rec_elig = f_mask
        return d1p, a1p, sse_p, e_rem, don_elig, rec_elig

    d1_p, a1_p, sse_p_cur, eff_rem_cur, don_elig_cur, rec_elig_cur = _score_donors_and_recipients()

    # Vectorized GPU top-`max_pairs` donor selection (ZERO Python loops over K!)
    q_try = min(max_pairs, k // 2)
    don_scores = torch.where(don_elig_cur, sse_p_cur, torch.full_like(sse_p_cur, -1.0))
    top_don_vals, top_don_ids = torch.topk(don_scores, k=q_try, largest=True, sorted=True)
    n_rec_avail_t = rec_elig_cur.to(torch.int64).sum()

    # Single tiny transfer of `<= 4` candidate donor IDs + their SSE
    pack_init = torch.cat(
        [top_don_ids.double(), top_don_vals.double(), n_rec_avail_t.view(1).double()]
    ).tolist()
    don_order = [
        int(pack_init[idx])
        for idx in range(q_try)
        if pack_init[q_try + idx] > 1e-12
    ]
    n_rec_avail = int(pack_init[-1])
    if not don_order or n_rec_avail < 2:
        return 0

    n_try = min(max_pairs, len(don_order), n_rec_avail // 2)
    don_order = don_order[:n_try]

    xc_p = xc_all[:half]
    xn_p = x_norm_all[:half]
    xc_q = xc_all[half:]
    xn_q = x_norm_all[half:]

    kept_edits = 0

    for s_idx in don_order:
        # Select best eligible recipient on GPU via masked argmin (ZERO Python loop over K!)
        rec_cand_mask = rec_elig_cur.clone()
        rec_cand_mask[s_idx] = False
        rec_cost_masked = torch.where(
            rec_cand_mask, eff_rem_cur, torch.full_like(eff_rem_cur, float("inf"))
        )
        best_rec_cost_t, d_idx_t = torch.min(rec_cost_masked, dim=0)

        # Pack donor validity + recipient check into 1 tiny 4-scalar read
        pair_meta = torch.stack(
            [
                don_elig_cur[s_idx].float(),
                sse_p_cur[s_idx],
                best_rec_cost_t,
                d_idx_t.float(),
            ]
        ).tolist()
        if pair_meta[0] < 0.5:
            continue
        s_sse_val = float(pair_meta[1])
        rec_cost_val = float(pair_meta[2])
        d_idx = int(pair_meta[3])
        if not math.isfinite(rec_cost_val) or rec_cost_val >= 0.45 * s_sse_val:
            continue

        s_mask = a1_p == s_idx
        xc_s = xc_p[s_mask]
        n_ps = int(xc_s.shape[0])
        if n_ps < 4:
            continue

        diag.proposals_tried += 1
        mu_sc = xc_s.mean(dim=0, keepdim=True)
        r_s = xc_s - mu_sc

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

        proj = r_s @ dirs.t()
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

        dist_m_p = (torch.addmm(cm_norm[None, :], xc_p, cm_rep_c.t(), alpha=-2.0) + xn_p[:, None]).clamp_min_(0.0)
        dist_p_p = (torch.addmm(cp_norm[None, :], xc_p, cp_rep_c.t(), alpha=-2.0) + xn_p[:, None]).clamp_min_(0.0)
        best_pair_p = torch.minimum(dist_m_p, dist_p_p)
        new_best_p = torch.minimum(d_rem_p[:, None], best_pair_p)
        delta_sse_p = (new_best_p - d1_p[:, None]).sum(dim=0)

        best_dp_t, best_m_t = torch.min(delta_sse_p, dim=0)
        best_pair_vals = torch.stack([best_dp_t, best_m_t.float()]).tolist()
        best_dp = float(best_pair_vals[0])
        best_m = int(best_pair_vals[1])
        if not (best_dp < -1e-6 * max(1.0, s_sse_val)):
            continue

        diag.p_improved += 1

        # Confirm winning probe `best_m` on disjoint Confirmation half Q
        d_rem_q = d_rem_all[half:]
        d1_q = top_d[half:, 0]
        cm_win = cm_rep_c[best_m : best_m + 1]
        cp_win = cp_rep_c[best_m : best_m + 1]
        dist_m_q = (
            torch.addmm(cm_norm[best_m : best_m + 1][None, :], xc_q, cm_win.t(), alpha=-2.0).squeeze(1) + xn_q
        ).clamp_min_(0.0)
        dist_p_q = (
            torch.addmm(cp_norm[best_m : best_m + 1][None, :], xc_q, cp_win.t(), alpha=-2.0).squeeze(1) + xn_q
        ).clamp_min_(0.0)
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
        u_win = dirs[best_m : best_m + 1]
        if compute_dtype in (torch.float32, torch.bfloat16, torch.float16):
            separate_collapsed_directional(new_s, new_d, u_win, compute_dtype)
        else:
            _separate_collapsed_splits(new_s, new_d, torch.float32)

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
        if prin_vec is not None:
            s_t = torch.tensor([s_idx], dtype=torch.int64, device=device)
            d_t = torch.tensor([d_idx], dtype=torch.int64, device=device)
            reset_post_split_oja_axes(prin_vec, s_t, d_t, u_win, generator=generator)
        if slow_centroids is not None and slow_usage is not None:
            slow_centroids[s_idx] = new_s[0]
            slow_centroids[d_idx] = new_d[0]
            shalf = slow_usage[s_idx] * 0.5
            slow_usage[s_idx] = shalf
            slow_usage[d_idx] = shalf
        if pass_counts is not None:
            pass_counts[s_idx] = 0.0
            pass_counts[d_idx] = 0.0
        if pass_counts_sq is not None:
            pass_counts_sq[s_idx] = 0.0
            pass_counts_sq[d_idx] = 0.0
        if pass_sums is not None:
            pass_sums[s_idx] = 0.0
            pass_sums[d_idx] = 0.0
        if pass_sse is not None:
            pass_sse[s_idx] = 0.0
            pass_sse[d_idx] = 0.0

        last_edit_step[s_idx] = step
        last_edit_step[d_idx] = step
        used_mask[s_idx] = True
        used_mask[d_idx] = True
        kept_edits += 1

        # Exact O(K*d) incremental top-3 & Huygens NN state refresh so surviving rare-mode
        # twins are immediately Huygens-shielded before checking the next candidate!
        if kept_edits < n_try and int((~used_mask).sum().item()) >= 2:
            if origin64 is not None:
                ns_c = (new_s[0].double() - origin64.squeeze(0)).float()
                nd_c = (new_d[0].double() - origin64.squeeze(0)).float()
            else:
                ns_c = new_s[0].to(rep_dtype).float() - origin.squeeze(0)
                nd_c = new_d[0].to(rep_dtype).float() - origin.squeeze(0)
            incremental_refresh_top3_and_nn(
                xc=xc_all,
                xn=x_norm_all,
                cc=cc,
                cn=cn,
                top_d=top_d,
                top_i=top_i,
                d_nn2=d_nn2,
                nn_idx=nn_idx,
                s_idx=s_idx,
                d_idx=d_idx,
                new_s_c=ns_c,
                new_d_c=nd_c,
            )
            d1_p, a1_p, sse_p_cur, eff_rem_cur, don_elig_cur, rec_elig_cur = _score_donors_and_recipients()

    return kept_edits


# ==============================================================================
# 3. Empirical-Bayes Finalizers (James-Stein w/ Kish n_eff & s-means eb_ema)
# ==============================================================================


@torch.no_grad()
def eb_james_stein_finalize_gpu(
    closing_means: torch.Tensor,
    anchor_centroids: torch.Tensor,
    pass_counts: torch.Tensor,
    pass_counts_sq: torch.Tensor | None,
    pass_sse: torch.Tensor,
    d: int,
    kappa: float | str = "auto",
) -> tuple[torch.Tensor, torch.Tensor, ShrinkageDiagnostics]:
    """Bounded-Influence Positive-Part James-Stein Empirical-Bayes Finalizer (`H-Algo-5`).

    Supports weighted closing passes (`s-means` C6 `pass_w_span > 0`) via exact Kish
    effective sample size `n_eff_j = (sum w_i)^2 / sum(w_i^2)` (which equals `n_j`
    identically when unweighted `w_i == 1`). Packs all diagnostic scalars into a single
    device-to-host transfer.
    """
    counts_w = pass_counts.double()
    nonempty = counts_w > 0.0
    k = closing_means.shape[0]
    device = closing_means.device

    if pass_counts_sq is not None:
        n_eff = torch.where(
            nonempty,
            counts_w.square() / pass_counts_sq.double().clamp_min(1e-30),
            torch.zeros_like(counts_w),
        )
    else:
        n_eff = counts_w

    total_w = counts_w.sum().clamp_min(1e-30)
    global_sigma2_t = (pass_sse.double().sum() / total_w).clamp_min(0.0)
    diff_sq = (closing_means.double() - anchor_centroids.double()).square().sum(dim=1)

    if isinstance(kappa, (int, float)):
        kappa_val = max(0.0, float(kappa))
        kappa_vec = torch.full((k,), kappa_val, dtype=torch.float64, device=device)
        lam = torch.where(
            nonempty,
            n_eff / (n_eff + kappa_vec).clamp_min(1e-30),
            torch.zeros_like(n_eff),
        )
        tau2_t = torch.where(nonempty, diff_sq, torch.zeros_like(diff_sq)).sum() / nonempty.double().sum().clamp_min(1.0)
    elif kappa == "auto":
        raw_sigma2 = torch.where(
            n_eff >= 2.0,
            pass_sse.double() / counts_w.clamp_min(1e-30),
            global_sigma2_t.expand_as(counts_w),
        ).clamp_min(0.0)
        sigma2_j = (n_eff * raw_sigma2 + 4.0 * global_sigma2_t) / (n_eff + 4.0)

        stein_factor = ((d - 2.0) / float(d)) if d > 2 else 1.0
        noise_var_j = stein_factor * sigma2_j / n_eff.clamp_min(1e-30)
        js_lam = 1.0 - (noise_var_j / (diff_sq + 1e-30))
        kappa_max = 8.0
        floor_lam = n_eff / (n_eff + kappa_max)

        zero_all = (global_sigma2_t <= 0.0) & (torch.where(nonempty, diff_sq, 0.0).max() <= 0.0)
        lam_auto = torch.maximum(js_lam, floor_lam).clamp(0.0, 1.0)
        lam = torch.where(
            nonempty,
            torch.where(zero_all, torch.ones_like(n_eff), lam_auto),
            torch.zeros_like(n_eff),
        )
        kappa_vec = torch.where(
            nonempty & (lam > 1e-12),
            n_eff * (1.0 - lam) / lam.clamp_min(1e-12),
            torch.zeros_like(n_eff),
        )
        excess = torch.clamp(diff_sq - noise_var_j, min=0.0)
        tau2_t = torch.where(nonempty, excess, torch.zeros_like(excess)).sum() / nonempty.double().sum().clamp_min(1.0)
    else:
        raise ValueError(f"kappa must be a non-negative float or 'auto', got {kappa!r}")

    has_ne = nonempty.any()
    lam_f32 = lam.float().clamp(0.0, 1.0)
    final_c = anchor_centroids.float() + lam_f32[:, None] * (
        closing_means.float() - anchor_centroids.float()
    )
    final_c = torch.where((lam_f32 == 1.0)[:, None], closing_means.float(), final_c)
    final_c = torch.where((~nonempty)[:, None], anchor_centroids.float(), final_c)

    # Single batched diagnostic summary transfer
    ne_count = nonempty.double().sum().clamp_min(1.0)
    mean_lam_t = torch.where(nonempty, lam, 0.0).sum() / ne_count
    min_lam_t = torch.where(nonempty, lam, float("inf")).min()
    max_lam_t = torch.where(nonempty, lam, float("-inf")).max()
    # Sort kappa_vec with +inf on empty slots to read exact median of nonempty slots without nonzero sync
    sorted_kap = torch.sort(torch.where(nonempty, kappa_vec, float("inf"))).values
    med_idx = (nonempty.to(torch.int64).sum() - 1).clamp_min(0) // 2
    med_kap_t = sorted_kap.gather(0, med_idx.view(1)).squeeze(0)

    vals = torch.stack(
        [
            has_ne.double(),
            mean_lam_t,
            min_lam_t,
            max_lam_t,
            med_kap_t,
            tau2_t,
            global_sigma2_t,
        ]
    ).tolist()

    if vals[0] < 0.5:
        diag = ShrinkageDiagnostics(
            mode=str(kappa),
            mean_lambda=0.0,
            min_lambda_nonempty=0.0,
            max_lambda_nonempty=0.0,
            median_effective_kappa=0.0,
            tau2_prior=0.0,
            sigma2_mean=0.0,
        )
    else:
        diag = ShrinkageDiagnostics(
            mode=str(kappa),
            mean_lambda=float(vals[1]),
            min_lambda_nonempty=float(vals[2]),
            max_lambda_nonempty=float(vals[3]),
            median_effective_kappa=float(vals[4]),
            tau2_prior=float(vals[5]),
            sigma2_mean=float(vals[6]),
        )
    return final_c.contiguous(), lam_f32, diag


# ==============================================================================
# 4. `HybridSEMeans` & `SuperEMeans` Unified Solver
# ==============================================================================


@dataclass
class SuperEMeansEpochStat:
    epoch: int
    rows: int
    steps: int
    moving_mse: float
    imbalance_factor: float
    n_splits: int
    n_protected: int
    n_transplants: int
    elapsed_sec: float


class HybridSEMeans:
    """Modular cross-pollination engine combining `SuperEMeans` (`P1..P6`) and `s-means` (`C1..C7`)."""

    def __init__(
        self,
        d: int,
        k: int,
        nepochs: float = 5.0,
        # Core recipe overrides
        batch_law: str = "auto",
        batch_size: int | None = None,
        alpha0: float | None = None,
        alpha_min_factor: float = ALPHA_MIN_FACTOR,
        unused_threshold: float | None = None,
        check_unused_every: int = CHECK_UNUSED_EVERY,
        init_usage: float = DEFAULT_INIT_USAGE,
        # P1 vs C1: Initialization & Warm-Start Priors
        init_mode: str = "coreset_kpp",
        coreset_mult: int = 16,
        min_coreset_size: int = 8192,
        init_rounds: int = 4,
        init_power: float = 1.5,
        use_voronoi_polish: bool = True,
        use_prior_mass: bool = True,
        prior_mass_scale: float = 0.5,
        prior_floor: float = 1e-3,
        warm_alpha_scale: float | None = None,
        # C2 & P6: Step Schedule, Low-D Revival, & Wide-Step Doubling
        schedule_mode: str = "emeans_cosine",
        alpha_floor: float = 0.06,
        cosine_power: float = 0.92,
        lowd_revival: bool = False,
        wide_step_mult: int = 1,
        wide_from_frac: float = 0.25,
        # P2 vs C3 & C4: Revival Window, Twin-Safe Huygens Shield, & Donor Scoring
        revival_mode: str = "split",
        use_revive_stop: bool = False,
        revive_stop_frac: float = 0.85,
        use_huygens_protection: bool = True,
        gamma_protect: float = 0.5,
        grace_steps: int = 5,
        donor_mode: str = "d2",
        donor_usage_w: float = 0.35,
        couple_sse_to_alpha: bool = True,
        sse_decay: float = 0.85,
        # P3 vs C5: Split Geometry & Perturbation Scale
        split_mode: str = "oja",
        split_std_frac: float = 0.08,
        split_mult: float = 0.55,
        use_feat_scale: bool = False,
        var_beta: float = 0.85,
        split_epsilon: float = SPLIT_EPSILON,
        # P5: Forward-Verified P/Q Signed-Hadamard Centroid Transplants (N-Track-1)
        head_edit: bool = False,
        transplant_every: int = 8,
        transplant_stop_frac: float = 0.85,
        transplant_probes: int = 6,
        transplant_max_pairs: int = 4,
        transplant_protect_steps: int = 10,
        # C6 & P4 vs C7: Final-Pass Weighting & Empirical-Bayes Finalization
        use_pass_weighting: bool = False,
        pass_w_span: float = 0.35,
        adaptive_pass_weighting: bool = False,
        finalization: str = "eb_james_stein",
        shrinkage_kappa: float | str = "auto",
        eb_anchor: str = "slow_ema",
        slow_ema_mult: float = 0.25,
        eb_tau: float = 0.20,
        # Execution, Coordinate-Shift Centering, & Blackwell Fast Path
        center_coordinates: bool = True,
        use_blackwell_assign: bool = True,
        preencode_dataset: bool = True,
        compute_dtype: torch.dtype | None = None,
        backend: str = "auto",
        seed: int = 1234,
        verbose: bool = False,
    ) -> None:
        if init_mode not in ("coreset_kpp", "auto", "coreset_exact", "coreset_block", "subsampled_d15", "random"):
            raise ValueError(f"Unsupported init_mode={init_mode!r}")
        if schedule_mode not in ("emeans_cosine", "power_cosine", "constant"):
            raise ValueError(f"Unsupported schedule_mode={schedule_mode!r}")
        if revival_mode not in ("split", "resample", "off"):
            raise ValueError(f"Unsupported revival_mode={revival_mode!r}")
        if donor_mode not in ("d2", "sse_only", "sse_usage", "usage_only"):
            raise ValueError(f"Unsupported donor_mode={donor_mode!r}")
        if split_mode not in ("oja", "oja_rms", "rms_aniso", "rms_iso", "checkerboard"):
            raise ValueError(f"Unsupported split_mode={split_mode!r}")
        if finalization not in ("eb_james_stein", "eb_shrinkage", "eb_ema", "closing_mean", "slow_ema", "ema_only", "none"):
            raise ValueError(f"Unsupported finalization={finalization!r}")
        if eb_anchor not in ("slow_ema", "ema_last"):
            raise ValueError(f"Unsupported eb_anchor={eb_anchor!r}")
        if prior_mass_scale <= 0.0:
            raise ValueError("prior_mass_scale must be > 0.0")
        if gamma_protect < 0.0:
            raise ValueError("gamma_protect must be >= 0.0")
        if wide_step_mult < 1:
            raise ValueError("wide_step_mult must be >= 1")

        self.d = int(d)
        self.k = int(k)
        self.nepochs = float(nepochs)

        self.batch_law = batch_law
        self.batch_size = batch_size
        self.alpha0 = alpha0
        self.alpha_min_factor = float(alpha_min_factor)
        self.unused_threshold = unused_threshold
        self.check_unused_every = int(check_unused_every)
        self.init_usage = float(init_usage)

        self.init_mode = "auto" if init_mode == "coreset_kpp" else init_mode
        self.coreset_mult = int(coreset_mult)
        self.min_coreset_size = int(min_coreset_size)
        self.init_rounds = int(init_rounds)
        self.init_power = float(init_power)
        self.use_voronoi_polish = bool(use_voronoi_polish)
        self.use_prior_mass = bool(use_prior_mass)
        self.prior_mass_scale = float(prior_mass_scale)
        self.prior_floor = float(prior_floor)
        if warm_alpha_scale is None:
            self.warm_alpha_scale = 0.85 if self.use_prior_mass else 1.0
        else:
            if warm_alpha_scale <= 0.0:
                raise ValueError("warm_alpha_scale must be > 0.0")
            self.warm_alpha_scale = float(warm_alpha_scale)

        self.schedule_mode = schedule_mode
        self.alpha_floor = float(alpha_floor)
        self.cosine_power = float(cosine_power)
        self.lowd_revival = bool(lowd_revival)
        self.wide_step_mult = int(wide_step_mult)
        self.wide_from_frac = float(wide_from_frac)

        self.revival_mode = revival_mode
        self.use_revive_stop = bool(use_revive_stop)
        self.revive_stop_frac = float(revive_stop_frac)
        self.use_huygens_protection = bool(use_huygens_protection)
        self.gamma_protect = float(gamma_protect)
        self.grace_steps = int(grace_steps)
        self.donor_mode = "d2" if donor_mode == "sse_only" else donor_mode
        self.donor_usage_w = float(donor_usage_w)
        self.couple_sse_to_alpha = bool(couple_sse_to_alpha)
        self.sse_decay = float(sse_decay)

        self.split_mode = split_mode
        self.split_std_frac = float(split_std_frac)
        self.split_mult = float(split_mult)
        self.use_feat_scale = bool(use_feat_scale) or (split_mode == "rms_aniso")
        self.var_beta = float(var_beta)
        self.split_epsilon = float(split_epsilon)

        self.head_edit = bool(head_edit)
        self.transplant_every = int(transplant_every)
        self.transplant_stop_frac = float(transplant_stop_frac)
        self.transplant_probes = int(transplant_probes)
        self.transplant_max_pairs = int(transplant_max_pairs)
        self.transplant_protect_steps = int(transplant_protect_steps)

        self.use_pass_weighting = bool(use_pass_weighting)
        self.pass_w_span = float(pass_w_span)
        self.adaptive_pass_weighting = bool(adaptive_pass_weighting)
        self.finalization = "eb_james_stein" if finalization == "eb_shrinkage" else finalization
        self.shrinkage_kappa = shrinkage_kappa
        self.eb_anchor = eb_anchor
        self.slow_ema_mult = float(slow_ema_mult)
        self.eb_tau = float(eb_tau)

        self.center_coordinates = bool(center_coordinates)
        self.use_blackwell_assign = bool(use_blackwell_assign)
        self.preencode_dataset = bool(preencode_dataset)
        self.compute_dtype = compute_dtype
        self.backend = backend
        self.seed = int(seed)
        self.verbose = bool(verbose)

        # Fitted state & diagnostics
        self.recipe: EMeansRecipe | None = None
        self.seeding_result: SeedingResult | None = None
        self.centroids: torch.Tensor | None = None
        self.closing_centroids: torch.Tensor | None = None
        self.slow_centroids: torch.Tensor | None = None
        self.assignments: torch.Tensor | None = None
        self.iteration_stats: list[SuperEMeansEpochStat] = []
        self.shrinkage_diagnostics: ShrinkageDiagnostics | None = None
        self.shrinkage_lambdas: torch.Tensor | None = None
        self.head_edit_diagnostics: HeadEditDiagnostics | None = None
        self.n_splits: int = 0
        self.n_protected: int = 0
        self.n_transplants: int = 0
        self.dead_clusters: int = 0
        self.init_time_sec: float = 0.0
        self.train_time_sec: float = 0.0
        self.total_time_sec: float = 0.0
        self._origin64: torch.Tensor | None = None
        self._fp16_scale: float = 1.0

    def _resolve_compute_dtype(self, x: torch.Tensor) -> torch.dtype:
        if self.compute_dtype is not None:
            if not x.is_cuda and self.compute_dtype in (torch.bfloat16, torch.float16):
                return torch.float32
            return self.compute_dtype
        if x.is_cuda and x.dtype in (torch.bfloat16, torch.float16):
            return x.dtype
        if x.is_cuda:
            return torch.bfloat16
        return torch.float32

    def _assign_internal(
        self,
        x_c: torch.Tensor,
        centroids_c: torch.Tensor,
        comp_dtype: torch.dtype,
        c_rep_cached: tuple[torch.Tensor, torch.Tensor] | None = None,
        x_rep_all: torch.Tensor | None = None,
        row_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dispatch nearest-centroid assignment across Blackwell Triton, standard Triton, or FP64/CPU."""
        use_bw = (
            self.use_blackwell_assign
            and x_c.is_cuda
            and HAS_TRITON
            and comp_dtype in (torch.float32, torch.bfloat16, torch.float16)
            and self.backend in ("auto", "triton", "blackwell")
        )
        if use_bw:
            if c_rep_cached is not None:
                c_rep, c_norm = c_rep_cached
            else:
                c_rep, c_norm = blackwell_prepare_centroids(centroids_c, comp_dtype)
            x_src = x_rep_all if x_rep_all is not None else x_c
            return blackwell_triton_assign(x_src, c_rep, c_norm, row_indices=row_indices)

        x_sub = x_c[row_indices] if row_indices is not None else x_c
        if comp_dtype == torch.float64:
            c64 = centroids_c.double().contiguous()
            c64_norm = c64.square().sum(dim=1).contiguous()
            return torch_assign_chunked(x_sub, c64, c64_norm)

        return assign_points(
            x_sub,
            centroids_c,
            compute_dtype=comp_dtype,
            backend="triton" if self.backend == "blackwell" else self.backend,
            c_rep_cached=c_rep_cached,
        )

    @torch.no_grad()
    def fit(
        self,
        x: torch.Tensor | np.ndarray,
        init_centroids: torch.Tensor | np.ndarray | None = None,
        seeding_result: SeedingResult | None = None,
        capture_assignments: bool = True,
    ) -> tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray | None]:
        """Fit `SuperEMeans` / `HybridSEMeans` on `x` of shape `(N, D)`."""
        input_is_numpy = isinstance(x, np.ndarray)
        if input_is_numpy:
            x_raw = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        else:
            x_raw = x.detach()

        if x_raw.ndim != 2:
            raise ValueError(f"Expected 2D input tensor, got shape {tuple(x_raw.shape)}")
        n, d = int(x_raw.shape[0]), int(x_raw.shape[1])
        if d != self.d:
            raise ValueError(f"Expected dimension d={self.d}, got {d}")
        if n < self.k:
            raise ValueError(f"Number of points n={n} must be >= number of clusters k={self.k}")

        device = x_raw.device
        comp_dtype = self._resolve_compute_dtype(x_raw)

        # Resolve revival threshold (honoring s-means C2 `lowd_revival=True` when `d <= 2`)
        if self.revival_mode == "off":
            eff_threshold: float | None = 0.0
        elif self.unused_threshold is not None:
            eff_threshold = float(self.unused_threshold)
        elif self.lowd_revival:
            eff_threshold = DEFAULT_UNUSED_THRESHOLD
        else:
            eff_threshold = None

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
            float(min(max(base_recipe.alpha0 * self.warm_alpha_scale, 1e-4), 1.0))
            if (self.alpha0 is None and self.warm_alpha_scale != 1.0)
            else float(base_recipe.alpha0)
        )
        recipe = (
            build_recipe(
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
            if eff_alpha0 != base_recipe.alpha0
            else base_recipe
        )
        self.recipe = recipe

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_init_start = time.perf_counter()

        # Exact Sterbenz FP32/FP64 dataset mean centering eliminates cancellation at +20,000 shifts
        # without allocating a 200MB FP64 [N, D] temporary on standard FP32/BF16 runs, plus exact
        # IEEE-754 power-of-two exponent scaling for extreme dynamic ranges (1e-35 .. 1e+35 & FP16)!
        if self.center_coordinates:
            raw_origin64 = compute_dataset_origin_fp64(x_raw)
            if comp_dtype == torch.float64:
                origin64 = raw_origin64
                x_c64 = x_raw.double() - origin64
                max_abs = float(x_c64.abs().max().item()) if n > 0 else 0.0
                if max_abs > 1e12 or (0.0 < max_abs < 1e-12):
                    exp_shift = int(round(math.log2(max_abs)))
                    fp16_scale = float(2.0 ** (-exp_shift))
                    x_c64 = x_c64 * fp16_scale
                else:
                    fp16_scale = 1.0
                x_t = x_c64.contiguous()
            else:
                origin32 = raw_origin64.float()
                origin64 = origin32.double()
                x_c32 = x_raw.float() - origin32
                max_abs = float(x_c32.abs().max().item()) if n > 0 else 0.0
                upper_lim = 16384.0 if comp_dtype == torch.float16 else 1e12
                if comp_dtype == torch.float16 and max_abs > upper_lim:
                    exp_shift = int(math.ceil(math.log2(max_abs / 16384.0)))
                    fp16_scale = float(2.0 ** (-exp_shift))
                    x_c32 = x_c32 * fp16_scale
                elif max_abs > 1e12 or (0.0 < max_abs < 1e-12):
                    exp_shift = int(round(math.log2(max_abs)))
                    fp16_scale = float(2.0 ** (-exp_shift))
                    x_c32 = (x_c32.double() * fp16_scale).float()
                else:
                    fp16_scale = 1.0
                x_t = x_c32.contiguous()
        else:
            origin64 = torch.zeros((1, d), dtype=torch.float64, device=device)
            fp16_scale = 1.0
            if not x_raw.is_cuda and x_raw.dtype in (torch.bfloat16, torch.float16):
                x_t = x_raw.float().contiguous()
            else:
                x_t = x_raw.contiguous()

        self._origin64 = origin64
        self._fp16_scale = fp16_scale

        x_f32 = x_t if x_t.dtype == torch.float32 else x_t.float().contiguous()

        # --- Initialization (`P1` Coreset+Voronoi vs `C1` Subsampled D^1.5 vs `random`) ---
        need_priors = self.use_prior_mass or self.use_voronoi_polish
        if seeding_result is not None:
            c_in64 = (seeding_result.centroids.to(device=device, dtype=torch.float64) - origin64) * fp16_scale
            seed_res = SeedingResult(
                centroids=c_in64.float().contiguous(),
                prior_probs=seeding_result.prior_probs.to(device=device, dtype=torch.float32).contiguous(),
                prior_distortion=(
                    seeding_result.prior_distortion.to(device=device, dtype=torch.float32) * (fp16_scale * fp16_scale)
                ).contiguous(),
            )
        elif init_centroids is not None:
            c_arr = (
                torch.from_numpy(np.asarray(init_centroids, dtype=np.float64)).to(device)
                if isinstance(init_centroids, np.ndarray)
                else init_centroids.detach().to(device=device, dtype=torch.float64)
            )
            raw_c = ((c_arr - origin64) * fp16_scale).float().contiguous()
            if need_priors:
                m_sub = min(n, max(self.min_coreset_size, self.coreset_mult * self.k, self.k * 2))
                x_sub, w_sub = lightweight_coreset(x_f32, m=m_sub, seed=self.seed)
                if self.use_voronoi_polish:
                    seed_res = voronoi_polish_and_priors(
                        x_sub, w_sub, raw_c, compute_dtype=torch.float32 if comp_dtype == torch.float64 else comp_dtype
                    )
                else:
                    sq_d, lbls = self._assign_internal(x_sub, raw_c, comp_dtype)
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
            else:
                seed_res = SeedingResult(
                    centroids=raw_c,
                    prior_probs=torch.full((self.k,), 1.0 / self.k, dtype=torch.float32, device=device),
                    prior_distortion=torch.zeros((self.k,), dtype=torch.float32, device=device),
                )
        elif self.init_mode == "subsampled_d15":
            raw_c = subsampled_d15_init_centered(
                x_f32,
                k=self.k,
                seed=self.seed,
                rounds=self.init_rounds,
                power=self.init_power,
            )
            if need_priors:
                m_sub = min(n, max(self.min_coreset_size, self.coreset_mult * self.k, self.k * 2))
                x_sub, w_sub = lightweight_coreset(x_f32, m=m_sub, seed=self.seed)
                if self.use_voronoi_polish:
                    seed_res = voronoi_polish_and_priors(
                        x_sub, w_sub, raw_c, compute_dtype=torch.float32 if comp_dtype == torch.float64 else comp_dtype
                    )
                else:
                    sq_d, lbls = self._assign_internal(x_sub, raw_c, comp_dtype)
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
            else:
                seed_res = SeedingResult(
                    centroids=raw_c,
                    prior_probs=torch.full((self.k,), 1.0 / self.k, dtype=torch.float32, device=device),
                    prior_distortion=torch.zeros((self.k,), dtype=torch.float32, device=device),
                )
        elif self.init_mode == "random":
            if need_priors:
                seed_res = seed_emeans_plusplus(
                    x_f32,
                    k=self.k,
                    seeding_mode="random",
                    coreset_mult=self.coreset_mult,
                    min_coreset_size=self.min_coreset_size,
                    polish=self.use_voronoi_polish,
                    seed=self.seed,
                )
            else:
                gen_init = torch.Generator(device=device)
                gen_init.manual_seed(self.seed % (2**32))
                perm0 = torch.randperm(n, generator=gen_init, device=device)[: self.k]
                raw_c = x_f32[perm0].clone().contiguous()
                seed_res = SeedingResult(
                    centroids=raw_c,
                    prior_probs=torch.full((self.k,), 1.0 / self.k, dtype=torch.float32, device=device),
                    prior_distortion=torch.zeros((self.k,), dtype=torch.float32, device=device),
                )
        else:
            # P1: Fast Blackwell Coreset + Triton/Block Greedy k-means++ + 1-step Voronoi Polish
            if device.type == "cuda" and HAS_TRITON and not torch.are_deterministic_algorithms_enabled():
                seed_res = fast_seed_emeans_plusplus(
                    x_f32,
                    k=self.k,
                    seeding_mode=self.init_mode,
                    coreset_mult=self.coreset_mult,
                    min_coreset_size=self.min_coreset_size,
                    polish=self.use_voronoi_polish,
                    use_cuda_graph=True,
                    seed=self.seed,
                )
            else:
                seed_res = seed_emeans_plusplus(
                    x_f32,
                    k=self.k,
                    seeding_mode=self.init_mode,
                    coreset_mult=self.coreset_mult,
                    min_coreset_size=self.min_coreset_size,
                    polish=self.use_voronoi_polish,
                    seed=self.seed,
                )

        self.seeding_result = seed_res
        centroids = seed_res.centroids.to(device=device, dtype=torch.float32).clone().contiguous()
        p_hat = seed_res.prior_probs.to(device=device, dtype=torch.float32)
        e_hat = seed_res.prior_distortion.to(device=device, dtype=torch.float32)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self.init_time_sec = time.perf_counter() - t_init_start

        # --- Training Setup ---
        t_train_start = time.perf_counter()
        bsz_float = float(recipe.batch_size)
        min_prob = max(0.0, self.prior_floor) / float(self.k)
        mean_ehat = e_hat.mean().clamp_min(1e-12)
        has_prior_dist = bool((e_hat.max() > 0.0).item()) if need_priors else False

        if self.use_prior_mass:
            usage = (self.prior_mass_scale * bsz_float * p_hat.clamp_min(min_prob)).float().contiguous()
            distortion = (
                self.prior_mass_scale * bsz_float * e_hat.clamp_min(1e-6 * mean_ehat)
            ).float().contiguous()
            sse_bias_mass = 1.0
        else:
            usage = torch.full((self.k,), recipe.init_usage, dtype=torch.float32, device=device)
            if has_prior_dist and self.couple_sse_to_alpha:
                distortion = (bsz_float * e_hat.clamp_min(1e-6 * mean_ehat)).float().contiguous()
                sse_bias_mass = 1.0
            else:
                distortion = torch.zeros((self.k,), dtype=torch.float32, device=device)
                sse_bias_mass = 0.0

        if self.use_huygens_protection:
            if need_priors:
                # Use raw empirical `p_hat` (no `min_prob` floor) so 0-count empty initial
                # centroids stay at `0.0` while real rare modes retain calibrated coreset mass!
                survival_mass: torch.Tensor | None = (bsz_float * p_hat).float().contiguous()
            else:
                # When `need_priors` is False (`p_hat` is unestimated), start `survival_mass` at 0.0
                # so 0-count dead centroids revive immediately while touched rare clusters accumulate!
                survival_mass = torch.zeros((self.k,), dtype=torch.float32, device=device)
        else:
            survival_mass = None

        # Slow EMA anchor (`P4` eb_james_stein with `eb_anchor="slow_ema"`, or `finalization="slow_ema"`)
        track_slow = (self.finalization == "slow_ema") or (
            self.finalization == "eb_james_stein" and self.eb_anchor == "slow_ema"
        )
        if track_slow:
            slow_centroids: torch.Tensor | None = centroids.clone()
            slow_usage: torch.Tensor | None = usage.clone()
            alpha_slow = min(1.0, max(1e-6, self.slow_ema_mult * recipe.alpha_min_factor * eff_alpha0))
        else:
            slow_centroids = None
            slow_usage = None
            alpha_slow = 0.0

        # Online Oja principal-axis state (`P3`)
        track_oja = self.split_mode in ("oja", "oja_rms")
        if track_oja:
            pca_gen = torch.Generator(device=device)
            pca_gen.manual_seed((self.seed + 31337) % (2**32))
            init_dirs = torch.randn((self.k, self.d), generator=pca_gen, device=device, dtype=torch.float32)
            init_dirs = init_dirs / init_dirs.norm(dim=1, keepdim=True).clamp_min(1e-12)
            prin_vec: torch.Tensor | None = (1e-3 * init_dirs).contiguous()
            g_sums_buf: torch.Tensor | None = torch.zeros((self.k, self.d), dtype=torch.float32, device=device)
        else:
            prin_vec = None
            g_sums_buf = None

        # Online O(K*d) inter-cluster feature variance (`C5` `feat_scale`)
        feat_scale: torch.Tensor | None = (
            torch.ones((self.d,), dtype=torch.float32, device=device)
            if self.use_feat_scale
            else None
        )

        # Forward-verified P/Q `head_edit` state (`P5`)
        if self.head_edit:
            last_edit_step = torch.full((self.k,), -(10**9), dtype=torch.int64, device=device)
            edit_gen = torch.Generator(device=device)
            edit_gen.manual_seed((self.seed + 918273) % (2**32))
            edit_diag: HeadEditDiagnostics | None = HeadEditDiagnostics()
        else:
            last_edit_step = None
            edit_gen = None
            edit_diag = None

        # Pre-encode low-precision query representation across passes (50% less HBM bandwidth,
        # 100% Appendix I.1 represented-norm bit-identical!)
        x_rep_all: torch.Tensor | None = None
        if (
            self.preencode_dataset
            and device.type == "cuda"
            and self.nepochs >= 2.0
            and comp_dtype in (torch.bfloat16, torch.float16)
            and x_t.dtype != comp_dtype
        ):
            x_rep_all = x_t.to(comp_dtype).contiguous()

        sign = torch.ones((self.d,), dtype=torch.float32, device=device)
        sign[1::2] = -1.0
        split_offset = recipe.split_epsilon * sign

        split_gen = torch.Generator(device=device)
        split_gen.manual_seed((self.seed + 7919) % (2**32))

        assignments_fit = (
            torch.full((n,), -1, dtype=torch.int64, device=device)
            if capture_assignments
            else None
        )

        ones_scalar = torch.ones((1,), dtype=torch.float32, device=device)
        arange_k = torch.arange(self.k, dtype=torch.int64, device=device)
        batch_sums = torch.zeros((self.k, self.d), dtype=torch.float32, device=device)
        counts_buf = torch.zeros((self.k,), dtype=torch.float32, device=device)
        batch_sse = torch.zeros((self.k,), dtype=torch.float32, device=device)

        use_triton_scatter = (
            device.type == "cuda"
            and HAS_TRITON
            and comp_dtype != torch.float64
            and not torch.are_deterministic_algorithms_enabled()
        )
        can_cache_crep = comp_dtype in (torch.float32, torch.bfloat16, torch.float16)
        if can_cache_crep:
            c_rep_buf, c_norm_buf = (
                blackwell_prepare_centroids(centroids, comp_dtype)
                if (self.use_blackwell_assign and device.type == "cuda")
                else prepare_represented_centroids(centroids, comp_dtype)
            )
        else:
            c_rep_buf, c_norm_buf = None, None

        step = 0
        rows_processed = 0
        total_planned_rows = max(1, sum(recipe.epoch_rows))
        num_epochs = len(recipe.epoch_rows)
        eff_grace = (
            min(self.grace_steps, max(0, recipe.total_steps // 3))
            if (self.use_prior_mass and self.use_huygens_protection)
            else 0
        )

        non_closing_steps = sum(
            math.ceil(r / max(1, recipe.batch_size)) for r in recipe.epoch_rows[:-1]
        )
        eff_transplant_every = (
            max(1, min(self.transplant_every, max(2, non_closing_steps // 4)))
            if non_closing_steps > 0
            else max(1, min(self.transplant_every, max(2, recipe.total_steps // 3)))
        )
        eff_transplant_protect = min(self.transplant_protect_steps, max(2, eff_transplant_every * 2))

        wide_alpha_factor = float(self.wide_step_mult ** ALPHA_LAW[2]) if self.wide_step_mult > 1 else 1.0
        in_wide_phase = False
        eff_pass_weighting = self.use_pass_weighting or (
            self.adaptive_pass_weighting and self.nepochs <= 1.5
        )

        n_splits_gpu = torch.zeros((), dtype=torch.int64, device=device)
        n_protected_gpu = torch.zeros((), dtype=torch.int64, device=device)
        self.n_splits = 0
        self.n_protected = 0
        self.n_transplants = 0
        self.iteration_stats = []
        self.shrinkage_diagnostics = None
        self.shrinkage_lambdas = None
        transplanted_in_last_epoch = False

        last_pass_counts: torch.Tensor | None = None
        last_pass_counts_sq: torch.Tensor | None = None
        last_pass_sums: torch.Tensor | None = None
        last_pass_sse: torch.Tensor | None = None

        need_final_sums = self.finalization in ("eb_james_stein", "eb_ema", "closing_mean")
        need_final_sse = self.finalization == "eb_james_stein"

        for ep_idx, rows_in_ep in enumerate(recipe.epoch_rows):
            is_last_epoch = ep_idx == (num_epochs - 1)
            gen_ep = torch.Generator(device=device)
            gen_ep.manual_seed(epoch_seed(self.seed, ep_idx))
            order = torch.randperm(n, generator=gen_ep, device=device)[:rows_in_ep]

            pass_counts = torch.zeros((self.k,), dtype=torch.float64, device=device)
            pass_counts_sq = (
                torch.zeros((self.k,), dtype=torch.float64, device=device)
                if (is_last_epoch and eff_pass_weighting and need_final_sse)
                else None
            )
            pass_sums = (
                torch.zeros((self.k, self.d), dtype=torch.float32, device=device)
                if (is_last_epoch and need_final_sums)
                else None
            )
            pass_sse = (
                torch.zeros((self.k,), dtype=torch.float64, device=device)
                if (is_last_epoch and need_final_sse)
                else None
            )

            ep_sq_dist_t = torch.zeros((), dtype=torch.float64, device=device)
            splits_before = self.n_splits
            prot_before = self.n_protected
            trans_before = self.n_transplants
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
                    w_scale = float(cur_bsz_target) / float(max(1, recipe.batch_size))
                    if w_scale != 1.0:
                        usage.mul_(w_scale)
                        distortion.mul_(w_scale)
                        if survival_mass is not None:
                            survival_mass.mul_(w_scale)
                        if slow_usage is not None:
                            slow_usage.mul_(w_scale)
                    in_wide_phase = True

                b_end = min(b_start + cur_bsz_target, rows_in_ep)
                batch_idx = order[b_start:b_end]
                bsz = int(b_end - b_start)

                # 1. Nearest-centroid assignment (fused index gather on GPU, zero [B, D] copy!)
                c_cached = (c_rep_buf, c_norm_buf) if (c_rep_buf is not None and c_norm_buf is not None) else None
                sq_dists, labels = self._assign_internal(
                    x_t,
                    centroids,
                    comp_dtype,
                    c_rep_cached=c_cached,
                    x_rep_all=x_rep_all,
                    row_indices=batch_idx,
                )

                if is_last_epoch and assignments_fit is not None:
                    assignments_fit[batch_idx] = labels

                ep_sq_dist_t.add_(sq_dists.sum(dtype=torch.float64))

                # 2. Scatter statistics (+ fused in-register Oja residual projection on GPU!)
                batch_sums.zero_()
                counts_buf.zero_()
                batch_sse.zero_()

                if use_triton_scatter:
                    if track_oja and prin_vec is not None and g_sums_buf is not None:
                        g_sums_buf.zero_()
                        p_norm = prin_vec.norm(dim=1, keepdim=True).clamp_min(1e-12)
                        p_hat_cur = (prin_vec / p_norm).contiguous()
                        full_d_oja = self.d <= 256
                        blk_b_oja = 64 if self.d <= 128 else 32
                        blk_d_oja = (
                            max(16, triton.next_power_of_2(max(1, self.d)))
                            if full_d_oja
                            else 128
                        )
                        with torch.cuda.device(device):
                            _super_scatter_oja_kernel[(triton.cdiv(bsz, blk_b_oja),)](
                                x_f32,
                                batch_idx,
                                labels,
                                sq_dists,
                                centroids,
                                p_hat_cur,
                                batch_sums,
                                counts_buf,
                                batch_sse,
                                g_sums_buf,
                                bsz,
                                self.d,
                                x_f32.stride(0),
                                x_f32.stride(1),
                                centroids.stride(0),
                                HAS_INDICES=True,
                                HAS_SSE=True,
                                FULL_D=full_d_oja,
                                BLOCK_B=blk_b_oja,
                                BLOCK_D=blk_d_oja,
                                num_warps=8 if self.d >= 64 else 4,
                            )
                    else:
                        blk_b_sc = 64
                        blk_d_sc = min(128, max(32, triton.next_power_of_2(max(1, self.d))))
                        grid_sc = (triton.cdiv(bsz, blk_b_sc), triton.cdiv(self.d, blk_d_sc))
                        with torch.cuda.device(device):
                            _super_scatter_stats_kernel[grid_sc](
                                x_f32,
                                batch_idx,
                                labels,
                                sq_dists,
                                batch_sums,
                                counts_buf,
                                batch_sse,
                                bsz,
                                self.d,
                                x_f32.stride(0),
                                x_f32.stride(1),
                                batch_sums.stride(0),
                                HAS_INDICES=True,
                                HAS_SSE=True,
                                BLOCK_B=blk_b_sc,
                                BLOCK_D=blk_d_sc,
                                num_warps=4,
                            )
                    counts = counts_buf
                else:
                    x_batch_f = x_f32[batch_idx]
                    counts = torch.bincount(labels, minlength=self.k).float()
                    batch_sums.index_add_(0, labels, x_batch_f)
                    batch_sse.index_add_(0, labels, sq_dists.float())

                # 3. Closing-pass accumulators (supporting s-means C6 linear ramp `w_pass`)
                if is_last_epoch and eff_pass_weighting:
                    n_sep_est = max(1, math.ceil(rows_in_ep / max(1, cur_bsz_target)))
                    pfrac = min(1.0, (float(b_start) + 0.5 * float(bsz)) / float(max(1, rows_in_ep)))
                    # Match exact `(s_ep + 0.5) / n_sep` when batch size is fixed, or smooth midpoint under wide_step
                    if self.wide_step_mult == 1:
                        pfrac = (ep_steps + 0.5) / float(n_sep_est)
                    w_pass = (1.0 - self.pass_w_span) + 2.0 * self.pass_w_span * pfrac
                    pass_counts.add_(counts.double(), alpha=w_pass)
                    if pass_counts_sq is not None:
                        pass_counts_sq.add_(counts.double(), alpha=w_pass * w_pass)
                    if pass_sums is not None:
                        pass_sums.add_(batch_sums, alpha=w_pass)
                    if pass_sse is not None:
                        pass_sse.add_(batch_sse.double(), alpha=w_pass)
                else:
                    pass_counts.add_(counts.double())
                    if pass_sums is not None:
                        pass_sums.add_(batch_sums)
                    if pass_sse is not None:
                        pass_sse.add_(batch_sse.double())

                # 4. Online O(K*d) inter-cluster feature variance (`s-means` C5 `feat_scale`)
                if feat_scale is not None and bsz > 0:
                    tot_c = counts.sum().clamp_min(1.0)
                    batch_mu = batch_sums.sum(dim=0) / tot_c
                    cur_var = (
                        (batch_sums - counts[:, None] * batch_mu[None, :]).square().sum(dim=0) / tot_c
                    ).clamp_min(1e-12)
                    feat_scale.mul_(self.var_beta).add_(cur_var, alpha=1.0 - self.var_beta)

                # 5. Step-size schedule (`emeans_cosine` vs `power_cosine` + `wide_step_mult`)
                if self.wide_step_mult > 1:
                    eff_peak = min(1.0, recipe.alpha0 * (wide_alpha_factor if use_wide_now else 1.0))
                    sched_frac = prog
                else:
                    eff_peak = recipe.alpha0
                    sched_frac = min(step, recipe.total_steps) / max(1, recipe.total_steps)

                alpha = super_scheduled_alpha(
                    alpha0=eff_peak,
                    frac=sched_frac,
                    schedule_mode=self.schedule_mode,
                    alpha_min_factor=recipe.alpha_min_factor,
                    alpha_floor=self.alpha_floor,
                    cosine_power=self.cosine_power,
                )
                decay = 1.0 - alpha

                # 6. EMA mass, Oja principal axis, centroid, distortion, & slow-EMA updates
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

                if track_oja and prin_vec is not None and g_sums_buf is not None:
                    if use_triton_scatter:
                        g_mean = g_sums_buf / safe_counts[:, None]
                        g_norm = g_mean.norm(dim=1)
                        rayleigh = (g_mean * p_hat_cur).sum(dim=1).clamp_min(0.0)
                        valid_oja = touched & (g_norm > 1e-12)
                        dir_new = g_mean / g_norm.clamp_min(1e-12)[:, None]
                        target_vec = dir_new * rayleigh.sqrt()[:, None]
                        w_oja = weight.clamp(0.0, 1.0)[:, None]
                        prin_vec = torch.where(
                            valid_oja[:, None], (1.0 - w_oja) * prin_vec + w_oja * target_vec, prin_vec
                        )
                    else:
                        prin_vec = update_oja_principal_axes(
                            prin_vec,
                            x_f32[batch_idx],
                            centroids,
                            labels,
                            counts,
                            weight,
                            g_buf=g_sums_buf,
                        )

                updated_pos = centroids + weight[:, None] * (batch_means - centroids)
                updated_pos = torch.where((weight == 1.0)[:, None], batch_means, updated_pos)
                centroids = torch.where(touched[:, None], updated_pos, centroids)
                usage = torch.where(touched, new_mass, usage)

                if self.couple_sse_to_alpha:
                    if alpha < 1.0:
                        distortion.mul_(decay).add_(batch_sse, alpha=alpha)
                        sse_bias_mass = decay * sse_bias_mass + alpha
                    else:
                        distortion = torch.where(touched, batch_sse, distortion * 0.5)
                        sse_bias_mass = 1.0
                else:
                    distortion.mul_(self.sse_decay).add_(batch_sse, alpha=1.0 - self.sse_decay)
                    sse_bias_mass = self.sse_decay * sse_bias_mass + (1.0 - self.sse_decay)

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

                # Bias-corrected mean distortion (keeps Huygens units calibrated even when sse_ema starts at 0!)
                eff_sse_norm = max(sse_bias_mass, 1e-6)
                ref_dist_mean = distortion.mean() / eff_sse_norm

                # 7. Zero-Sync GPU Dead-Cluster Revival (`P2` Twin-Safe Huygens + `C3/C4` + `P3/C5`)
                within_revive_window = True
                if self.use_revive_stop:
                    if self.wide_step_mult > 1:
                        within_revive_window = prog <= self.revive_stop_frac
                    else:
                        within_revive_window = step <= int(self.revive_stop_frac * recipe.total_steps)

                allow_revival = (
                    recipe.unused_threshold > 0.0
                    and self.revival_mode != "off"
                    and within_revive_window
                    and ((step + 1) >= max(1, eff_grace))
                    and ((step + 1) % recipe.check_unused_every == 0)
                    and self.k >= 2
                )
                if allow_revival:
                    thresh = usage.mean() * recipe.unused_threshold
                    low_mask = usage < thresh

                    if self.use_huygens_protection and self.gamma_protect > 0.0:
                        protected_mask = twin_safe_huygens_protection(
                            centroids=centroids,
                            low_mask=low_mask,
                            usage=usage,
                            survival_mass=survival_mass,
                            thresh=thresh,
                            ref_distortion_mean=ref_dist_mean,
                            gamma_protect=self.gamma_protect,
                        )
                        n_protected_gpu.add_(protected_mask.to(torch.int64).sum())
                        unused_mask = low_mask & (~protected_mask)
                        donor_mask = (~low_mask) & (usage > 0.0)
                    else:
                        unused_mask = low_mask
                        donor_mask = (~unused_mask) & (usage > 0.0)

                    n_unused_t = unused_mask.to(torch.int64).sum()
                    n_donors_t = donor_mask.to(torch.int64).sum()

                    if self.revival_mode == "split":
                        nsplit_t = torch.minimum(n_unused_t, n_donors_t)
                        n_splits_gpu.add_(nsplit_t)
                        m_max = self.k // 2

                        u_gumbel = torch.rand(
                            (self.k,), generator=split_gen, device=device, dtype=torch.float32
                        ).clamp_min_(torch.finfo(torch.float32).tiny)
                        g_noise = -(-u_gumbel.log()).log()

                        if self.donor_mode == "usage_only":
                            keys = torch.where(
                                donor_mask,
                                usage.clamp_min(1e-30).log() + g_noise,
                                torch.full_like(usage, float("-inf")),
                            )
                        elif self.donor_mode == "sse_usage":
                            has_sse_t = distortion.max() > 0.0
                            sc_sse_u = distortion.clamp_min(1e-12) * (
                                usage.clamp_min(1e-12) ** self.donor_usage_w
                            )
                            sc_vec = torch.where(has_sse_t, sc_sse_u, usage)
                            keys = torch.where(
                                donor_mask,
                                sc_vec.clamp_min(1e-30).log() + g_noise,
                                torch.full_like(usage, float("-inf")),
                            )
                        else:
                            # `d2` donor scoring (`EMeansPP` / `SuperEMeans`)
                            pos_dist_mask = donor_mask & (distortion > 1e-30)
                            n_pos_t = pos_dist_mask.to(torch.int64).sum()
                            keys_dist = torch.where(
                                pos_dist_mask,
                                distortion.clamp_min(1e-30).log() + g_noise,
                                torch.full_like(usage, float("-inf")),
                            )
                            keys_usage = torch.where(
                                donor_mask,
                                usage.clamp_min(1e-30).log() + g_noise,
                                torch.full_like(usage, float("-inf")),
                            )
                            keys = torch.where(n_pos_t >= nsplit_t, keys_dist, keys_usage)

                        donor_perm = torch.argsort(keys, descending=True)
                        target_perm = torch.argsort(
                            unused_mask.to(torch.int32), descending=True, stable=True
                        )

                        donor_rank = torch.empty((self.k,), dtype=torch.int64, device=device)
                        donor_rank.scatter_(0, donor_perm, arange_k)
                        target_rank = torch.empty((self.k,), dtype=torch.int64, device=device)
                        target_rank.scatter_(0, target_perm, arange_k)

                        is_act_donor = donor_rank < nsplit_t
                        is_act_target = target_rank < nsplit_t
                        src_cand = donor_perm[:m_max]
                        donors_c = centroids[src_cand]

                        # Compute `(d_cent, c_cent, u_dir)` on `(m_max, D)` with zero host sync!
                        if self.split_mode == "checkerboard":
                            d_cent = donors_c * (1.0 - split_offset)
                            c_cent = donors_c * (1.0 + split_offset)
                            _separate_collapsed_splits(d_cent, c_cent, comp_dtype)
                            u_dir_cand = sign.unsqueeze(0).expand(m_max, self.d) / math.sqrt(max(1, self.d))
                        elif self.split_mode in ("oja", "oja_rms") and prin_vec is not None:
                            pv = prin_vec[src_cand]
                            sigma_1 = pv.norm(dim=1, keepdim=True)
                            e0 = torch.zeros((m_max, self.d), dtype=torch.float32, device=device)
                            e0[:, 0] = 1.0
                            u_dir_cand = torch.where(sigma_1 > 1e-12, pv / sigma_1.clamp_min(1e-12), e0)
                            if self.split_mode == "oja_rms":
                                # Combine Oja local eigenvector (when genuinely anisotropic/bimodal)
                                # with s-means C5's full Euclidean chord step `split_mult * sqrt(MSE_s)`!
                                dist_s = distortion[src_cand] / eff_sse_norm
                                u_s = usage[src_cand].clamp_min(1e-6)
                                chord_rms = (dist_s / u_s).clamp_min(1e-10).sqrt()[:, None]
                                iso_coord_std = chord_rms / math.sqrt(float(max(1, self.d)))
                                radem_o = torch.where(
                                    torch.rand((m_max, self.d), generator=split_gen, device=device) < 0.5,
                                    -1.0,
                                    1.0,
                                )
                                if feat_scale is not None:
                                    fvec = feat_scale.clamp_min(1e-10).sqrt()
                                    fvec = fvec / fvec.norm().clamp_min(1e-8) * math.sqrt(float(self.d))
                                    aniso_radem = radem_o * fvec[None, :]
                                else:
                                    aniso_radem = radem_o
                                aniso_radem_u = aniso_radem / aniso_radem.norm(dim=1, keepdim=True).clamp_min(1e-12)
                                u_dir_cand = torch.where(
                                    sigma_1 > (1.20 * iso_coord_std),
                                    u_dir_cand,
                                    aniso_radem_u,
                                )
                                mag = self.split_mult * torch.maximum(sigma_1, chord_rms)
                            else:
                                fallback_sc = self.split_epsilon * donors_c.norm(dim=1, keepdim=True).clamp_min(1.0)
                                mag = torch.where(sigma_1 > 1e-8, self.split_std_frac * sigma_1, fallback_sc)
                            delta = mag * u_dir_cand
                            d_cent = donors_c - delta
                            c_cent = donors_c + delta
                            separate_collapsed_directional(d_cent, c_cent, u_dir_cand, comp_dtype)
                        else:
                            # `rms_aniso` / `rms_iso` (`suayptalha/s-means` C5)
                            dist_s = distortion[src_cand]
                            u_s = usage[src_cand].clamp_min(1e-6)
                            rad = self.split_mult * (
                                dist_s / (u_s * float(max(1, self.d)))
                            ).clamp_min(1e-10).sqrt()[:, None]
                            radem = torch.where(
                                torch.rand((m_max, self.d), generator=split_gen, device=device) < 0.5,
                                -1.0,
                                1.0,
                            )
                            if self.split_mode == "rms_aniso" and feat_scale is not None:
                                fvec = feat_scale.clamp_min(1e-10).sqrt()
                                fvec = fvec / fvec.norm().clamp_min(1e-8) * math.sqrt(float(self.d))
                                step_vec = radem * fvec[None, :]
                            else:
                                step_vec = radem
                            delta = rad * step_vec
                            d_cent = donors_c - delta
                            c_cent = donors_c + delta
                            u_dir_cand = step_vec / step_vec.norm(dim=1, keepdim=True).clamp_min(1e-12)
                            separate_collapsed_directional(d_cent, c_cent, u_dir_cand, comp_dtype)

                        r_don = donor_rank.clamp_max(m_max - 1)
                        r_tgt = target_rank.clamp_max(m_max - 1)

                        centroids = torch.where(
                            is_act_donor[:, None],
                            d_cent[r_don],
                            torch.where(is_act_target[:, None], c_cent[r_tgt], centroids),
                        )

                        half_u_cand = usage[src_cand] * 0.5
                        usage = torch.where(
                            is_act_donor,
                            half_u_cand[r_don],
                            torch.where(is_act_target, half_u_cand[r_tgt], usage),
                        )

                        half_e_cand = distortion[src_cand] * 0.5
                        distortion = torch.where(
                            is_act_donor,
                            half_e_cand[r_don],
                            torch.where(is_act_target, half_e_cand[r_tgt], distortion),
                        )

                        if survival_mass is not None:
                            half_m_cand = survival_mass[src_cand] * 0.5
                            survival_mass = torch.where(
                                is_act_donor,
                                half_m_cand[r_don],
                                torch.where(is_act_target, half_m_cand[r_tgt], survival_mass),
                            )

                        if prin_vec is not None:
                            old_sig = prin_vec[src_cand].norm(dim=1, keepdim=True)
                            if self.d <= 1:
                                rst_vec = 0.5 * prin_vec[src_cand]
                            else:
                                rnd_p = torch.randn(
                                    (m_max, self.d), generator=split_gen, device=device, dtype=torch.float32
                                )
                                orth = rnd_p - (rnd_p * u_dir_cand).sum(dim=1, keepdim=True) * u_dir_cand
                                orth_n = orth.norm(dim=1, keepdim=True)
                                alt = torch.roll(u_dir_cand, shifts=1, dims=1)
                                alt = alt - (alt * u_dir_cand).sum(dim=1, keepdim=True) * u_dir_cand
                                orth = torch.where(orth_n > 1e-8, orth, alt)
                                orth_u = orth / orth.norm(dim=1, keepdim=True).clamp_min(1e-12)
                                rst_vec = (0.5 * old_sig) * orth_u
                            prin_vec = torch.where(
                                is_act_donor[:, None],
                                rst_vec[r_don],
                                torch.where(is_act_target[:, None], rst_vec[r_tgt], prin_vec),
                            )

                        if track_slow and slow_centroids is not None and slow_usage is not None:
                            s_donors = slow_centroids[src_cand]
                            if self.split_mode == "checkerboard":
                                sd_cent = s_donors * (1.0 - split_offset)
                                sc_cent = s_donors * (1.0 + split_offset)
                                _separate_collapsed_splits(sd_cent, sc_cent, comp_dtype)
                            else:
                                s_delta = 0.5 * (c_cent - d_cent)
                                sd_cent = s_donors - s_delta
                                sc_cent = s_donors + s_delta
                                separate_collapsed_directional(sd_cent, sc_cent, u_dir_cand, comp_dtype)
                            shalf_u = slow_usage[src_cand] * 0.5
                            slow_centroids = torch.where(
                                is_act_donor[:, None],
                                sd_cent[r_don],
                                torch.where(is_act_target[:, None], sc_cent[r_tgt], slow_centroids),
                            )
                            slow_usage = torch.where(
                                is_act_donor,
                                shalf_u[r_don],
                                torch.where(is_act_target, shalf_u[r_tgt], slow_usage),
                            )

                    elif self.revival_mode == "resample" and bsz > 0:
                        nsplit_r = torch.minimum(n_unused_t, torch.tensor(bsz, dtype=torch.int64, device=device))
                        n_splits_gpu.add_(nsplit_r)
                        target_perm = torch.argsort(
                            unused_mask.to(torch.int32), descending=True, stable=True
                        )
                        target_rank = torch.empty((self.k,), dtype=torch.int64, device=device)
                        target_rank.scatter_(0, target_perm, arange_k)
                        is_act_target = target_rank < nsplit_r
                        m_res = min(self.k, bsz)
                        r_idx = batch_idx[torch.randperm(bsz, generator=split_gen, device=device)[:m_res]]
                        picked = x_f32[r_idx]
                        r_tgt = target_rank.clamp_max(m_res - 1)
                        centroids = torch.where(is_act_target[:, None], picked[r_tgt], centroids)
                        mean_u = usage.mean().clamp_min(recipe.init_usage)
                        mean_e = distortion.mean().clamp_min(1e-12)
                        usage = torch.where(is_act_target, mean_u, usage)
                        distortion = torch.where(is_act_target, mean_e, distortion)
                        if survival_mass is not None:
                            survival_mass = torch.where(is_act_target, mean_u, survival_mass)

                # 8. Forward-Verified P/Q `head_edit` Centroid Transplants (`P5`)
                # Runs pre-closing for multi-epoch budgets (`not is_last_epoch`), AND on single-epoch
                # (`num_epochs == 1`, e.g. `T = 1.0`) up to `prog <= 0.60` with closing-stat reset!
                allow_head_edit = (
                    self.head_edit
                    and edit_gen is not None
                    and last_edit_step is not None
                    and edit_diag is not None
                    and ((step + 1) % eff_transplant_every == 0)
                    and (
                        (not is_last_epoch and prog <= self.transplant_stop_frac)
                        or (num_epochs == 1 and prog <= min(0.60, self.transplant_stop_frac))
                    )
                )
                if allow_head_edit:
                    xb_edit = x_rep_all[batch_idx] if x_rep_all is not None else x_t[batch_idx]
                    n_ed = run_fast_head_edit_step(
                        x_batch=xb_edit,
                        centroids=centroids,
                        usage=usage,
                        distortion=distortion,
                        ref_distortion_mean=ref_dist_mean,
                        survival_mass=survival_mass,
                        prin_vec=prin_vec,
                        slow_centroids=slow_centroids,
                        slow_usage=slow_usage,
                        pass_counts=pass_counts if is_last_epoch else None,
                        pass_counts_sq=pass_counts_sq if is_last_epoch else None,
                        pass_sums=pass_sums if is_last_epoch else None,
                        pass_sse=pass_sse if is_last_epoch else None,
                        last_edit_step=last_edit_step,
                        step=step,
                        compute_dtype=comp_dtype,
                        unused_threshold=recipe.unused_threshold,
                        use_huygens_protection=self.use_huygens_protection,
                        gamma_protect=self.gamma_protect,
                        num_probes=self.transplant_probes,
                        max_pairs=self.transplant_max_pairs,
                        protect_steps=eff_transplant_protect,
                        generator=edit_gen,
                        diag=edit_diag,
                        already_centered=self.center_coordinates,
                    )
                    if n_ed > 0:
                        self.n_transplants += n_ed
                        n_splits_gpu.add_(n_ed)
                        if is_last_epoch:
                            transplanted_in_last_epoch = True

                # Refresh represented centroids for the next batch step
                if can_cache_crep:
                    c_rep_buf, c_norm_buf = (
                        blackwell_prepare_centroids(centroids, comp_dtype)
                        if (self.use_blackwell_assign and device.type == "cuda")
                        else prepare_represented_centroids(centroids, comp_dtype)
                    )

                rows_processed += bsz
                step += 1
                ep_steps += 1
                b_start = b_end

            if device.type == "cuda":
                torch.cuda.synchronize(device)

            tot_pw = pass_counts.sum().clamp_min(1.0)
            imb_t = self.k * torch.dot(pass_counts, pass_counts) / (tot_pw * tot_pw)
            ep_pack = torch.stack(
                [
                    ep_sq_dist_t,
                    imb_t,
                    n_splits_gpu.double(),
                    n_protected_gpu.double(),
                ]
            ).tolist()
            ep_sq_dist = (
                float(ep_pack[0]) / (fp16_scale * fp16_scale)
                if fp16_scale > 0.0
                else float(ep_pack[0])
            )
            imbalance = float(ep_pack[1])
            self.n_splits = int(ep_pack[2])
            self.n_protected = int(ep_pack[3])

            stat = SuperEMeansEpochStat(
                epoch=ep_idx + 1,
                rows=rows_in_ep,
                steps=ep_steps,
                moving_mse=ep_sq_dist / max(1, rows_in_ep),
                imbalance_factor=imbalance,
                n_splits=self.n_splits - splits_before,
                n_protected=self.n_protected - prot_before,
                n_transplants=self.n_transplants - trans_before,
                elapsed_sec=time.perf_counter() - t_train_start,
            )
            self.iteration_stats.append(stat)
            if self.verbose:
                print(
                    f"[SuperEMeans] pass {ep_idx + 1}/{num_epochs} ({rows_in_ep} rows, {ep_steps} steps): "
                    f"moving_mse={stat.moving_mse:.6f}, splits={stat.n_splits}, "
                    f"protected={stat.n_protected}, transplants={stat.n_transplants}, time={stat.elapsed_sec:.3f}s"
                )

            if is_last_epoch:
                last_pass_counts = pass_counts
                last_pass_counts_sq = pass_counts_sq
                last_pass_sums = pass_sums
                last_pass_sse = pass_sse

        # --- Finalization (`P4` James-Stein EB vs `C7` eb_ema vs `closing_mean` vs `slow_ema`) ---
        c_ema_last = centroids.clone()
        if last_pass_counts is not None:
            self.dead_clusters = int((last_pass_counts <= 0.5).sum().item())
        else:
            self.dead_clusters = 0

        if self.finalization == "closing_mean" and last_pass_counts is not None and last_pass_sums is not None:
            nonempty = last_pass_counts > 0.0
            c_close = torch.where(
                nonempty[:, None],
                last_pass_sums / last_pass_counts.clamp_min(1e-12).float()[:, None],
                c_ema_last,
            )
            self.closing_centroids = c_close
            centroids = c_close
        elif self.finalization == "eb_ema" and last_pass_counts is not None and last_pass_sums is not None:
            # s-means C7: Prior-count Empirical-Bayes shrinkage (`tau = eb_tau`) toward live `c_ema_last`
            nonempty = last_pass_counts > 0.0
            c_close = torch.where(
                nonempty[:, None],
                last_pass_sums / last_pass_counts.clamp_min(1e-12).float()[:, None],
                c_ema_last,
            )
            self.closing_centroids = c_close
            tau = max(0.0, float(self.eb_tau))
            w_fin = (last_pass_counts / (last_pass_counts + tau).clamp_min(1e-30)).float()
            centroids = torch.where(
                nonempty[:, None],
                c_ema_last + w_fin[:, None] * (c_close - c_ema_last),
                c_ema_last,
            )
        elif self.finalization == "eb_james_stein" and last_pass_counts is not None and last_pass_sums is not None and last_pass_sse is not None:
            nonempty = last_pass_counts > 0.0
            c_close = torch.where(
                nonempty[:, None],
                last_pass_sums / last_pass_counts.clamp_min(1e-12).float()[:, None],
                c_ema_last,
            )
            self.closing_centroids = c_close
            anchor_c = (
                slow_centroids
                if (self.eb_anchor == "slow_ema" and slow_centroids is not None)
                else c_ema_last
            )
            centroids, lam_f32, s_diag = eb_james_stein_finalize_gpu(
                closing_means=c_close,
                anchor_centroids=anchor_c,
                pass_counts=last_pass_counts,
                pass_counts_sq=last_pass_counts_sq,
                pass_sse=last_pass_sse,
                d=self.d,
                kappa=self.shrinkage_kappa,
            )
            self.shrinkage_lambdas = lam_f32
            self.shrinkage_diagnostics = s_diag
        elif self.finalization == "slow_ema" and slow_centroids is not None:
            centroids = slow_centroids
        else:
            # `ema_only` / `none`
            centroids = c_ema_last

        # If `nepochs < 1.0` (partial single pass `epoch_rows[-1] < n`) OR a single-pass run
        # executed mid-pass `head_edit` transplants, assign all `n` rows cleanly against finalized centroids.
        if (
            capture_assignments
            and assignments_fit is not None
            and (transplanted_in_last_epoch or recipe.epoch_rows[-1] < n)
        ):
            _, assignments_fit = self._assign_internal(x_t, centroids, comp_dtype, x_rep_all=x_rep_all)

        # Un-center back to original input coordinate frame in FP64 -> FP32
        if self.center_coordinates:
            inv_s = 1.0 / fp16_scale
            centroids_out = (centroids.double() * inv_s + origin64).float().contiguous()
            if self.closing_centroids is not None:
                self.closing_centroids = (
                    self.closing_centroids.double() * inv_s + origin64
                ).float().contiguous()
            if slow_centroids is not None:
                slow_centroids = (slow_centroids.double() * inv_s + origin64).float().contiguous()
        else:
            centroids_out = centroids.contiguous()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self.train_time_sec = time.perf_counter() - t_train_start
        self.total_time_sec = self.init_time_sec + self.train_time_sec

        self.centroids = centroids_out
        self.slow_centroids = slow_centroids
        self.assignments = assignments_fit
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
        dev = self.centroids.device
        x_in = (
            torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(dev)
            if input_is_numpy
            else x.detach().to(dev)
        )
        comp_dtype = self._resolve_compute_dtype(x_in)

        if self.center_coordinates:
            orig64 = (
                self._origin64.to(dev)
                if self._origin64 is not None
                else compute_dataset_origin_fp64(x_in)
            )
            sc = self._fp16_scale
            x_c64 = (x_in.double() - orig64) * sc
            c_c64 = (self.centroids.double() - orig64) * sc
            x_c = (x_c64 if comp_dtype == torch.float64 else x_c64.float()).contiguous()
            c_c = (c_c64 if comp_dtype == torch.float64 else c_c64.float()).contiguous()
            dists, labels = self._assign_internal(x_c, c_c, comp_dtype)
            if sc != 1.0:
                inv_sq = 1.0 / (sc * sc)
                dists = (dists.double() * inv_sq).clamp_max(torch.finfo(torch.float32).max).float()
        else:
            if not x_in.is_cuda and x_in.dtype in (torch.bfloat16, torch.float16):
                x_in = x_in.float()
            dists, labels = self._assign_internal(x_in.contiguous(), self.centroids, comp_dtype)

        if input_is_numpy:
            return dists.cpu().numpy(), labels.cpu().numpy()
        return dists, labels


class SuperEMeans(HybridSEMeans):
    """Unified Champion (`SuperEMeans`) fusing Phase 1 + Phase 2A + Phase 2B winners (`P1..P6`)."""

    def __init__(
        self,
        d: int,
        k: int,
        nepochs: float = 5.0,
        wide_step_mult: int = 1,
        wide_from_frac: float = 0.25,
        head_edit: bool = True,
        split_mode: str = "oja",
        finalization: str = "eb_james_stein",
        shrinkage_kappa: float | str = "auto",
        use_huygens_protection: bool = True,
        gamma_protect: float = 0.5,
        seed: int = 1234,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            d=d,
            k=k,
            nepochs=nepochs,
            init_mode=kwargs.pop("init_mode", "coreset_kpp"),
            use_voronoi_polish=kwargs.pop("use_voronoi_polish", True),
            use_prior_mass=kwargs.pop("use_prior_mass", True),
            schedule_mode=kwargs.pop("schedule_mode", "emeans_cosine"),
            wide_step_mult=wide_step_mult,
            wide_from_frac=wide_from_frac,
            use_huygens_protection=use_huygens_protection,
            gamma_protect=gamma_protect,
            donor_mode=kwargs.pop("donor_mode", "d2"),
            couple_sse_to_alpha=kwargs.pop("couple_sse_to_alpha", True),
            split_mode=split_mode,
            head_edit=head_edit,
            finalization=finalization,
            shrinkage_kappa=shrinkage_kappa,
            use_blackwell_assign=kwargs.pop("use_blackwell_assign", True),
            seed=seed,
            verbose=verbose,
            **kwargs,
        )


# ==============================================================================
# 5. Pre-Registered Bidirectional Ablation & Cross-Pollination Matrix
# ==============================================================================

_EMEANS_PP_ANCHOR: dict[str, Any] = {
    "init_mode": "coreset_kpp",
    "use_voronoi_polish": True,
    "use_prior_mass": True,
    "schedule_mode": "emeans_cosine",
    "lowd_revival": False,
    "wide_step_mult": 1,
    "use_revive_stop": False,
    "use_huygens_protection": True,
    "gamma_protect": 0.5,
    "donor_mode": "d2",
    "couple_sse_to_alpha": True,
    "split_mode": "checkerboard",
    "use_feat_scale": False,
    "head_edit": False,
    "use_pass_weighting": False,
    "finalization": "closing_mean",
}

_SUPER_EMEANS_ANCHOR: dict[str, Any] = {
    **_EMEANS_PP_ANCHOR,
    "split_mode": "oja",
    "split_std_frac": 0.08,
    "head_edit": True,
    "finalization": "eb_james_stein",
    "shrinkage_kappa": "auto",
    "eb_anchor": "slow_ema",
}

_SMEANS_DECONFOUNDED_ANCHOR: dict[str, Any] = {
    "init_mode": "subsampled_d15",
    "init_rounds": 4,
    "init_power": 1.5,
    "use_voronoi_polish": False,
    "use_prior_mass": False,
    "schedule_mode": "power_cosine",
    "alpha_floor": 0.06,
    "cosine_power": 0.92,
    "lowd_revival": True,
    "wide_step_mult": 1,
    "use_revive_stop": True,
    "revive_stop_frac": 0.85,
    "use_huygens_protection": False,
    "donor_mode": "sse_usage",
    "donor_usage_w": 0.35,
    "couple_sse_to_alpha": False,
    "sse_decay": 0.85,
    "split_mode": "rms_aniso",
    "split_mult": 0.55,
    "use_feat_scale": True,
    "var_beta": 0.85,
    "head_edit": False,
    "use_pass_weighting": True,
    "pass_w_span": 0.35,
    "finalization": "eb_ema",
    "eb_tau": 0.20,
}

HYBRID_ABLATION_ARMS: dict[str, dict[str, Any]] = {
    # --- Group 0: Matched-Kernel Baseline & Champion Anchors ---
    "emeans_pp_base": dict(_EMEANS_PP_ANCHOR),
    "super_emeans": dict(_SUPER_EMEANS_ANCHOR),
    "super_emeans_wide": {**_SUPER_EMEANS_ANCHOR, "wide_step_mult": 2, "wide_from_frac": 0.25},
    "smeans_deconfounded": dict(_SMEANS_DECONFOUNDED_ANCHOR),
    # --- Group 1: Grafting Each `s-means` Component `C1..C7` onto `EMeansPP` ---
    "pp_plus_c1_d15_raw": {
        **_EMEANS_PP_ANCHOR,
        "init_mode": "subsampled_d15",
        "use_voronoi_polish": False,
        "use_prior_mass": False,
    },
    "pp_plus_c1_d15_polished": {
        **_EMEANS_PP_ANCHOR,
        "init_mode": "subsampled_d15",
        "use_voronoi_polish": True,
        "use_prior_mass": True,
    },
    "pp_plus_c2_power_cos": {
        **_EMEANS_PP_ANCHOR,
        "schedule_mode": "power_cosine",
        "lowd_revival": True,
    },
    "pp_plus_c3_revive_stop": {
        **_EMEANS_PP_ANCHOR,
        "use_revive_stop": True,
        "revive_stop_frac": 0.85,
    },
    "pp_plus_c4_sse_usage_donor": {
        **_EMEANS_PP_ANCHOR,
        "donor_mode": "sse_usage",
        "donor_usage_w": 0.35,
        "couple_sse_to_alpha": False,
        "sse_decay": 0.85,
    },
    "pp_plus_c5_rms_aniso_split": {
        **_EMEANS_PP_ANCHOR,
        "split_mode": "rms_aniso",
        "split_mult": 0.55,
        "use_feat_scale": True,
    },
    "pp_plus_c6_pass_weight": {
        **_EMEANS_PP_ANCHOR,
        "use_pass_weighting": True,
        "pass_w_span": 0.35,
    },
    "pp_plus_c7_eb_ema": {
        **_EMEANS_PP_ANCHOR,
        "finalization": "eb_ema",
        "eb_tau": 0.20,
    },
    # --- Group 2: Grafting Each `s-means` Component `C1..C7` onto `SuperEMeans` ---
    "super_plus_c1_d15_init": {
        **_SUPER_EMEANS_ANCHOR,
        "init_mode": "subsampled_d15",
    },
    "super_plus_c2_power_cos": {
        **_SUPER_EMEANS_ANCHOR,
        "schedule_mode": "power_cosine",
        "lowd_revival": True,
    },
    "super_plus_c3_revive_stop": {
        **_SUPER_EMEANS_ANCHOR,
        "use_revive_stop": True,
        "revive_stop_frac": 0.85,
    },
    "super_plus_c4_sse_usage_donor": {
        **_SUPER_EMEANS_ANCHOR,
        "donor_mode": "sse_usage",
        "donor_usage_w": 0.35,
        "couple_sse_to_alpha": True,
    },
    "super_plus_c5_rms_aniso_split": {
        **_SUPER_EMEANS_ANCHOR,
        "split_mode": "rms_aniso",
        "split_mult": 0.55,
        "use_feat_scale": True,
    },
    "super_plus_c5b_oja_rms_split": {
        **_SUPER_EMEANS_ANCHOR,
        "split_mode": "oja_rms",
        "split_mult": 0.55,
        "use_feat_scale": True,
    },
    "super_plus_c6_pass_weight": {
        **_SUPER_EMEANS_ANCHOR,
        "use_pass_weighting": True,
        "pass_w_span": 0.35,
    },
    "super_plus_c7_eb_ema": {
        **_SUPER_EMEANS_ANCHOR,
        "finalization": "eb_ema",
        "eb_tau": 0.20,
    },
    # --- Group 3: Grafting Each `SuperEMeans` Pillar `P1..P6` onto `SMeans` ---
    "smeans_plus_p1_coreset_voronoi": {
        **_SMEANS_DECONFOUNDED_ANCHOR,
        "init_mode": "coreset_kpp",
        "use_voronoi_polish": True,
        "use_prior_mass": True,
        "couple_sse_to_alpha": True,
    },
    "smeans_plus_p2_huygens_shield": {
        **_SMEANS_DECONFOUNDED_ANCHOR,
        "use_huygens_protection": True,
        "gamma_protect": 0.5,
        "couple_sse_to_alpha": True,
    },
    "smeans_plus_p3_oja_split": {
        **_SMEANS_DECONFOUNDED_ANCHOR,
        "split_mode": "oja_rms",
        "split_mult": 0.55,
        "use_feat_scale": True,
    },
    "smeans_plus_p3_oja008": {
        **_SMEANS_DECONFOUNDED_ANCHOR,
        "split_mode": "oja",
        "split_std_frac": 0.08,
        "use_feat_scale": False,
    },
    "smeans_plus_p4_eb_js": {
        **_SMEANS_DECONFOUNDED_ANCHOR,
        "finalization": "eb_james_stein",
        "shrinkage_kappa": "auto",
        "eb_anchor": "slow_ema",
    },
    "smeans_plus_p5_head_edit": {
        **_SMEANS_DECONFOUNDED_ANCHOR,
        "head_edit": True,
        "use_huygens_protection": True,
        "gamma_protect": 0.5,
    },
    "smeans_plus_p6_wide_step": {
        **_SMEANS_DECONFOUNDED_ANCHOR,
        "wide_step_mult": 2,
        "wide_from_frac": 0.25,
    },
    # --- Group 4: Synergistic Cross-Pollinated Hybrids (`SuperEMeans + Best SMeans`) ---
    "hybrid_super_smeans": {
        **_SUPER_EMEANS_ANCHOR,
        "lowd_revival": True,
        "use_revive_stop": True,
        "revive_stop_frac": 0.85,
        "donor_mode": "d2",
        "couple_sse_to_alpha": True,
        "split_mode": "oja_rms",
        "split_mult": 0.55,
        "use_feat_scale": True,
        "adaptive_pass_weighting": True,
        "pass_w_span": 0.35,
        "eb_anchor": "ema_last",
    },
    "hybrid_super_smeans_wide": {
        **_SUPER_EMEANS_ANCHOR,
        "lowd_revival": True,
        "use_revive_stop": True,
        "revive_stop_frac": 0.85,
        "donor_mode": "d2",
        "couple_sse_to_alpha": True,
        "split_mode": "oja_rms",
        "split_mult": 0.55,
        "use_feat_scale": True,
        "adaptive_pass_weighting": True,
        "pass_w_span": 0.35,
        "eb_anchor": "ema_last",
        "wide_step_mult": 2,
        "wide_from_frac": 0.25,
    },
}


def build_hybrid_arm(
    arm_name: str,
    d: int,
    k: int,
    nepochs: float = 5.0,
    seed: int = 1234,
    **overrides: Any,
) -> HybridSEMeans:
    """Instantiate any pre-registered ablation or hybrid arm by key."""
    if arm_name not in HYBRID_ABLATION_ARMS:
        valid = ", ".join(sorted(HYBRID_ABLATION_ARMS.keys()))
        raise KeyError(f"Unknown arm_name={arm_name!r}; expected one of: {valid}")
    cfg = {**HYBRID_ABLATION_ARMS[arm_name], **overrides}
    return HybridSEMeans(d=d, k=k, nepochs=nepochs, seed=seed, **cfg)


__all__ = [
    "HYBRID_ABLATION_ARMS",
    "HybridSEMeans",
    "SuperEMeans",
    "SuperEMeansEpochStat",
    "build_hybrid_arm",
    "compute_dataset_origin_fp64",
    "eb_james_stein_finalize_gpu",
    "fast_top3_centered_assign",
    "incremental_refresh_top3_and_nn",
    "run_fast_head_edit_step",
    "subsampled_d15_init_centered",
    "super_scheduled_alpha",
    "twin_safe_huygens_protection",
]
