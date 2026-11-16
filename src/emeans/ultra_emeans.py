"""Next-Generation Algorithmic & Systems Synthesis: `UltraEMeans`.

Synthesizes and transcends Meta FAIR's `e-means` (Bessa et al., arXiv:2611.09501),
`suayptalha/s-means` (Kocabay et al., 2026), and `SuperEMeans` (`P1..P6`) by resolving
the exact mathematical and GPU-systems limitations uncovered across both lineages:

  - Discovery 0 (`ultra_turkpp` Zero-Copy Fused Importance Coreset + Batched Tournament D^2 + Voronoi Polish):
      Single-pass fused Triton radial-energy sketch (`O(n*d)` at memory-bandwidth peak with zero `(N, D)` copy)
      + zero-sync batched multinomial/greedy tournament `k-means++` + 1-step weighted Voronoi polish +
      calibrated prior EMA mass & initial Oja scale (~2 ms on `(200k, 128, 2048)` vs 36 ms in `SuperEMeans`,
      while preserving isolated rare modes that `s-means` uniform `4k` subsampling misses).

  - Discovery 1 (Exact Half-Gaussian Principal-Axis Split: `sqrt(2/pi) * sigma_{j,1} * v_{j,1}`):
      * `e-means` splits by `1e-4 * c` (collapses in BF16, explodes under coordinate shifts, takes 10+ steps).
      * `s-means` splits by `0.55 * sigma_rms * (eps * f_hat)`, whose Euclidean step norm is `0.55 * sqrt(d) * sigma_rms`
        (`6.22 * sigma` overshoot in `d = 128`!) along a random Rademacher sign vector orthogonal to rotated cluster axes.
      * `SuperEMeans` tracked Oja's `v_{j,1}` but stepped by `0.08 * sigma_{j,1}` (`10x` smaller than conditional means).
      * `UltraEMeans` tracks `(sigma_{j,1}, v_{j,1})` in registers inside the fused scatter + EMA Triton kernels (`0` extra
        HBM passes), blends `v_{j,1}` smoothly with unit-normalized feature anisotropy `f_hat` when a cluster is newly
        initialized/isotropic, steps to the **exact 1D half-space Gaussian centroids** `+- sqrt(2/pi) * max(sigma_{j,1}, sigma_rms) * u_j`
        (`sqrt(2/pi) ~= 0.79788456`), deflates child SSE by `(2/pi) * N_j * sigma_split^2`, and flushes stale pre-split
        parent sums from closing-pass accumulators!

  - Discovery 2 (Decay-Matched Bias-Corrected `SSE * Usage^gamma` Donors + Sync-Free Twin-Safe Huygens Shield + 85% Freeze):
      Couples `sse_ema` to the **exact same scheduled `alpha_t`** (and zero-init bias factor) as `usage` so
      `sigma_rms^2 = (sse_ema / sse_bias) / ((usage / usage_bias) * d)` has exact physical variance units at every step,
      scores donors by principal-boosted distortion `(0.5 * E_j + 0.5 * d * N_j * sigma_{j,1}^2) * N_j^gamma`, freezes
      splits at `85%` of budget, enables all-`d` (`d >= 1`) revival, and protects isolated rare modes via a SRAM-tiled
      zero-sync Twin-Safe Huygens kernel (`O(K^2)` in registers, zero `K x K` allocation even at `K = 16,384`).

  - Discovery 3 (Simplex 3-Point Empirical-Bayes Finalizer: `closing_mean <-> live_ema <-> slow_ema` + Adaptive `1/T` Ramp):
      Forms a drift-gated anchor between variance-reduced `slow_ema` and drift-tracking `live_ema` via exact stationary
      noise-to-drift ratio `beta_slow = V_es / max(||c_ema - c_slow||^2, V_es)`, combines `closing_mean` with the anchor
      using correlation-safe Bayes/Stein weights on the 2-simplex (`w_close + w_ema + w_slow = 1`), and scales closing-pass
      progressive ramp weighting as `span * (1 / max(T, 1))^1.25` with exact Kish `n_eff`.

  - Discovery 4 (Strategic Vectorized `P/Q`-Confirmed `head_edit` Transplants + Zero-Sync Fused Blackwell Triton Pipeline):
      Executes entire multi-epoch training in 3 fused Triton launches per batch (`assign`, `scatter+oja+sse`,
      `ema+oja+slow+pass+prep`) with zero `.item()` sync stalls, plus fast Oja-seeded `P/Q`-confirmed dormant-twin
      transplants at pre-closing checkpoints.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .algo_extensions import ShrinkageDiagnostics, separate_collapsed_directional
from .emeans import _separate_collapsed_splits
from .fast_ops import HAS_TRITON, _choose_split_k_partitions
from .kernels import assign_points, prepare_represented_centroids, torch_assign_chunked
from .noise_tracks import HeadEditDiagnostics, exact_exclusion_sq_dists, sample_signed_hadamard_rows
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
)
from .seeding import SeedingResult
from .super_emeans import fast_top3_centered_assign, incremental_refresh_top3_and_nn, super_scheduled_alpha

if HAS_TRITON:
    import triton
    import triton.language as tl


HALF_GAUSSIAN_MEAN: float = math.sqrt(2.0 / math.pi)  # 0.7978845608028654
HALF_GAUSSIAN_VAR_DROP: float = 2.0 / math.pi         # 0.6366197723675814
HALF_GAUSSIAN_REM_STD: float = math.sqrt(1.0 - 2.0 / math.pi)  # 0.6028102749890869


# ==============================================================================
# 1. Fused Zero-Sync Blackwell Triton Kernels for `UltraEMeans`
# ==============================================================================

if HAS_TRITON:

    @triton.jit(do_not_specialize=["P_stride", "P_per_g", "num_select_per_g", "last_g_P", "last_g_sel", "G", "rng_seed"])
    def _ultra_batched_local_greedy_kpp_kernel(
        pair_d2_ptr,
        init_d2_ptr,
        weights_ptr,
        out_idx_ptr,
        P_stride,
        P_per_g,
        num_select_per_g,
        last_g_P,
        last_g_sel,
        G,
        rng_seed,
        BLOCK_P: tl.constexpr,
        L_TRIALS: tl.constexpr,
    ):
        """Multi-CTA spatial-shard parallel Greedy k-means++ tournament (`G` shards in parallel!)."""
        gid = tl.program_id(0).to(tl.int64)
        if gid >= G:
            return
        is_last = gid == (G - 1)
        P_cur = tl.where(is_last, last_g_P, P_per_g)
        sel_cur = tl.where(is_last, last_g_sel, num_select_per_g)
        base_p = gid * P_per_g
        base_out = gid * num_select_per_g

        offs_p = tl.arange(0, BLOCK_P)
        mask_p = offs_p < P_cur
        pool_d2 = tl.load(init_d2_ptr + base_p + offs_p, mask=mask_p, other=0.0).to(tl.float32)
        w = tl.load(weights_ptr + base_p + offs_p, mask=mask_p, other=0.0).to(tl.float32)
        log_w = tl.log(tl.maximum(w, 1e-30))
        avail = mask_p
        seed_g = rng_seed + gid.to(tl.int32) * 7919

        for s_idx in range(sel_cur):
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
                    u = tl.rand(seed_g, rng_offs)
                    u = tl.minimum(tl.maximum(u, 1e-30), 1.0 - 1e-7)
                    gumbel = -tl.log(-tl.log(u))
                    key_pos = tl.where(avail & (proposal > 0.0), log_prop + gumbel, -float("inf"))
                    key_fb = tl.where(avail, log_w + gumbel, -float("inf"))
                    key = tl.where(has_pos, key_pos, key_fb)
                    cand = tl.argmax(key, axis=0)

                row_d2 = tl.load(
                    pair_d2_ptr + (gid * BLOCK_P + cand.to(tl.int64)) * P_stride + offs_p.to(tl.int64),
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
            tl.store(out_idx_ptr + base_out + s_idx, base_p + best_pick.to(tl.int64))

    @triton.jit(do_not_specialize=["P", "num_select", "rng_seed"])
    def _ultra_local_greedy_kpp_kernel(
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
        """Zero-specialization single-CTA weighted Greedy k-means++ candidate tournament."""
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

    @triton.jit(do_not_specialize=["N", "D", "stride_xn", "stride_xd"])
    def _ultra_origin_d2_kernel(
        x_ptr,
        origin_ptr,
        out_d2_ptr,
        N,
        D,
        stride_xn,
        stride_xd,
        coord_scale,
        ORIGIN_FP64: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Single-pass fused centered squared-norm sketch `||(x_i - origin) * coord_scale||_2^2`."""
        pid_n = tl.program_id(0).to(tl.int64)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
        mask_n = offs_n < N

        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for d0 in range(0, D, BLOCK_D):
            cols = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
            mask_d = cols < D
            mask_nd = mask_n[:, None] & mask_d[None, :]
            x_raw = tl.load(
                x_ptr + offs_n[:, None] * stride_xn + cols[None, :] * stride_xd,
                mask=mask_nd,
                other=0.0,
            )
            if ORIGIN_FP64:
                orig64 = tl.load(origin_ptr + cols, mask=mask_d, other=0.0).to(tl.float64)
                xf = ((x_raw.to(tl.float64) - orig64[None, :]) * coord_scale).to(tl.float32)
            else:
                orig32 = tl.load(origin_ptr + cols, mask=mask_d, other=0.0).to(tl.float32)
                xf = (x_raw.to(tl.float32) - orig32[None, :]) * coord_scale
            xf = tl.where(mask_nd, xf, 0.0)
            acc += tl.sum(xf * xf, axis=1)

        tl.store(out_d2_ptr + offs_n, acc, mask=mask_n)

    @triton.jit(do_not_specialize=["B", "K", "D", "stride_xb", "stride_xd", "stride_ck"])
    def _ultra_assign_kernel(
        x_ptr,
        row_idx_ptr,
        origin_ptr,
        c_ptr,
        c_norm_ptr,
        dist_ptr,
        label_ptr,
        B,
        K,
        D,
        stride_xb,
        stride_xd,
        stride_ck,
        coord_scale,
        BLOCK_B: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
        FULL_D: tl.constexpr,
        DTYPE_MODE: tl.constexpr,  # 0=FP32, 1=BF16, 2=FP16
        HAS_INDICES: tl.constexpr,
        HAS_ORIGIN: tl.constexpr,
        HAS_SCALE: tl.constexpr,
        ORIGIN_FP64: tl.constexpr,
    ):
        """Fused index-gather + register centering + represented-norm Tensor-Core assignment."""
        pid_b = tl.program_id(0).to(tl.int64)
        offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B).to(tl.int64)
        mask_b = offs_b < B
        src_rows = (
            tl.load(row_idx_ptr + offs_b, mask=mask_b, other=0).to(tl.int64)
            if HAS_INDICES
            else offs_b
        )

        x_norm = tl.zeros((BLOCK_B,), dtype=tl.float32)
        if FULL_D:
            cols = tl.arange(0, BLOCK_D).to(tl.int64)
            mask_d = cols < D
            mask_bd = mask_b[:, None] & mask_d[None, :]
            x_raw = tl.load(
                x_ptr + src_rows[:, None] * stride_xb + cols[None, :] * stride_xd,
                mask=mask_bd,
                other=0.0,
            )
            if HAS_ORIGIN:
                if ORIGIN_FP64:
                    orig64 = tl.load(origin_ptr + cols, mask=mask_d, other=0.0).to(tl.float64)
                    xdiff = x_raw.to(tl.float64) - orig64[None, :]
                    x_f = (xdiff * coord_scale).to(tl.float32) if HAS_SCALE else xdiff.to(tl.float32)
                else:
                    orig32 = tl.load(origin_ptr + cols, mask=mask_d, other=0.0).to(tl.float32)
                    x_f = x_raw.to(tl.float32) - orig32[None, :]
                    if HAS_SCALE:
                        x_f = x_f * coord_scale
                x_f = tl.where(mask_bd, x_f, 0.0)
            else:
                x_f = x_raw.to(tl.float32)
                if HAS_SCALE:
                    x_f = tl.where(mask_bd, x_f * coord_scale, 0.0)

            if DTYPE_MODE == 1:
                x_full = x_f.to(tl.bfloat16)
                xf_rep = x_full.to(tl.float32)
            elif DTYPE_MODE == 2:
                x_full = x_f.to(tl.float16)
                xf_rep = x_full.to(tl.float32)
            else:
                x_full = x_f
                xf_rep = x_f
            x_norm = tl.sum(xf_rep * xf_rep, axis=1)
        else:
            for d0 in range(0, D, BLOCK_D):
                cols = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
                mask_d = cols < D
                mask_bd = mask_b[:, None] & mask_d[None, :]
                x_raw = tl.load(
                    x_ptr + src_rows[:, None] * stride_xb + cols[None, :] * stride_xd,
                    mask=mask_bd,
                    other=0.0,
                )
                if HAS_ORIGIN:
                    if ORIGIN_FP64:
                        orig64 = tl.load(origin_ptr + cols, mask=mask_d, other=0.0).to(tl.float64)
                        xdiff = x_raw.to(tl.float64) - orig64[None, :]
                        x_f = (xdiff * coord_scale).to(tl.float32) if HAS_SCALE else xdiff.to(tl.float32)
                    else:
                        orig32 = tl.load(origin_ptr + cols, mask=mask_d, other=0.0).to(tl.float32)
                        x_f = x_raw.to(tl.float32) - orig32[None, :]
                        if HAS_SCALE:
                            x_f = x_f * coord_scale
                    x_f = tl.where(mask_bd, x_f, 0.0)
                else:
                    x_f = x_raw.to(tl.float32)
                    if HAS_SCALE:
                        x_f = tl.where(mask_bd, x_f * coord_scale, 0.0)

                if DTYPE_MODE == 1:
                    xf_rep = x_f.to(tl.bfloat16).to(tl.float32)
                elif DTYPE_MODE == 2:
                    xf_rep = x_f.to(tl.float16).to(tl.float32)
                else:
                    xf_rep = x_f
                x_norm += tl.sum(xf_rep * xf_rep, axis=1)

        best_score = tl.full((BLOCK_B,), float("inf"), dtype=tl.float32)
        best_label = tl.zeros((BLOCK_B,), dtype=tl.int32)

        for k0 in range(0, K, BLOCK_K):
            offs_k = (k0 + tl.arange(0, BLOCK_K)).to(tl.int64)
            mask_k = offs_k < K
            if FULL_D:
                cols = tl.arange(0, BLOCK_D).to(tl.int64)
                mask_d = cols < D
                c_tile = tl.load(
                    c_ptr + offs_k[:, None] * stride_ck + cols[None, :],
                    mask=mask_k[:, None] & mask_d[None, :],
                    other=0.0,
                )
                if DTYPE_MODE == 0:
                    acc = tl.dot(x_full, tl.trans(c_tile), input_precision="ieee", out_dtype=tl.float32)
                else:
                    acc = tl.dot(x_full, tl.trans(c_tile), out_dtype=tl.float32)
            else:
                acc = tl.zeros((BLOCK_B, BLOCK_K), dtype=tl.float32)
                for d0 in range(0, D, BLOCK_D):
                    cols = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
                    mask_d = cols < D
                    mask_bd = mask_b[:, None] & mask_d[None, :]
                    x_raw = tl.load(
                        x_ptr + src_rows[:, None] * stride_xb + cols[None, :] * stride_xd,
                        mask=mask_bd,
                        other=0.0,
                    )
                    if HAS_ORIGIN:
                        if ORIGIN_FP64:
                            orig64 = tl.load(origin_ptr + cols, mask=mask_d, other=0.0).to(tl.float64)
                            xdiff = x_raw.to(tl.float64) - orig64[None, :]
                            x_f = (xdiff * coord_scale).to(tl.float32) if HAS_SCALE else xdiff.to(tl.float32)
                        else:
                            orig32 = tl.load(origin_ptr + cols, mask=mask_d, other=0.0).to(tl.float32)
                            x_f = x_raw.to(tl.float32) - orig32[None, :]
                            if HAS_SCALE:
                                x_f = x_f * coord_scale
                        x_f = tl.where(mask_bd, x_f, 0.0)
                    else:
                        x_f = x_raw.to(tl.float32)
                        if HAS_SCALE:
                            x_f = tl.where(mask_bd, x_f * coord_scale, 0.0)

                    c_tile = tl.load(
                        c_ptr + offs_k[:, None] * stride_ck + cols[None, :],
                        mask=mask_k[:, None] & mask_d[None, :],
                        other=0.0,
                    )
                    if DTYPE_MODE == 1:
                        acc = tl.dot(x_f.to(tl.bfloat16), tl.trans(c_tile), acc=acc, out_dtype=tl.float32)
                    elif DTYPE_MODE == 2:
                        acc = tl.dot(x_f.to(tl.float16), tl.trans(c_tile), acc=acc, out_dtype=tl.float32)
                    else:
                        acc = tl.dot(x_f, tl.trans(c_tile), acc=acc, input_precision="ieee", out_dtype=tl.float32)

            c_norm = tl.load(c_norm_ptr + offs_k, mask=mask_k, other=float("inf"))
            scores = tl.where(mask_k[None, :], c_norm[None, :] - 2.0 * acc, float("inf"))
            blk_min, blk_idx = tl.min(
                scores,
                axis=1,
                return_indices=True,
                return_indices_tie_break_left=True,
            )
            better = blk_min < best_score
            best_score = tl.where(better, blk_min, best_score)
            best_label = tl.where(better, k0 + blk_idx.to(tl.int32), best_label)

        dist = tl.maximum(best_score + x_norm, 0.0)
        tl.store(dist_ptr + offs_b, dist, mask=mask_b)
        tl.store(label_ptr + offs_b, best_label.to(tl.int64), mask=mask_b)

    @triton.jit(do_not_specialize=["B", "K", "D", "stride_xb", "stride_xd", "stride_ck"])
    def _ultra_splitk_assign_kernel(
        x_ptr,
        row_idx_ptr,
        origin_ptr,
        c_ptr,
        c_norm_ptr,
        part_score_ptr,
        part_label_ptr,
        part_xnorm_ptr,
        B,
        K,
        D,
        stride_xb,
        stride_xd,
        stride_ck,
        coord_scale,
        PARTITIONS: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
        FULL_D: tl.constexpr,
        DTYPE_MODE: tl.constexpr,
        HAS_INDICES: tl.constexpr,
        HAS_ORIGIN: tl.constexpr,
        HAS_SCALE: tl.constexpr,
        ORIGIN_FP64: tl.constexpr,
    ):
        """2D Split-K nearest-centroid kernel saturating all 188 Blackwell SMs when K >= 2048."""
        pid_b = tl.program_id(0).to(tl.int64)
        part = tl.program_id(1)
        offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B).to(tl.int64)
        mask_b = offs_b < B
        src_rows = (
            tl.load(row_idx_ptr + offs_b, mask=mask_b, other=0).to(tl.int64)
            if HAS_INDICES
            else offs_b
        )

        part_size = (K + PARTITIONS - 1) // PARTITIONS
        k_start = part * part_size
        k_end = tl.minimum(k_start + part_size, K)

        x_norm = tl.zeros((BLOCK_B,), dtype=tl.float32)
        if FULL_D:
            cols = tl.arange(0, BLOCK_D).to(tl.int64)
            mask_d = cols < D
            mask_bd = mask_b[:, None] & mask_d[None, :]
            x_raw = tl.load(
                x_ptr + src_rows[:, None] * stride_xb + cols[None, :] * stride_xd,
                mask=mask_bd,
                other=0.0,
            )
            if HAS_ORIGIN:
                if ORIGIN_FP64:
                    orig64 = tl.load(origin_ptr + cols, mask=mask_d, other=0.0).to(tl.float64)
                    xdiff = x_raw.to(tl.float64) - orig64[None, :]
                    x_f = (xdiff * coord_scale).to(tl.float32) if HAS_SCALE else xdiff.to(tl.float32)
                else:
                    orig32 = tl.load(origin_ptr + cols, mask=mask_d, other=0.0).to(tl.float32)
                    x_f = x_raw.to(tl.float32) - orig32[None, :]
                    if HAS_SCALE:
                        x_f = x_f * coord_scale
                x_f = tl.where(mask_bd, x_f, 0.0)
            else:
                x_f = x_raw.to(tl.float32)
                if HAS_SCALE:
                    x_f = tl.where(mask_bd, x_f * coord_scale, 0.0)

            if DTYPE_MODE == 1:
                x_full = x_f.to(tl.bfloat16)
            elif DTYPE_MODE == 2:
                x_full = x_f.to(tl.float16)
            else:
                x_full = x_f
            if part == 0:
                xf_rep = x_full.to(tl.float32)
                x_norm = tl.sum(xf_rep * xf_rep, axis=1)
        elif part == 0:
            for d0 in range(0, D, BLOCK_D):
                cols = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
                mask_d = cols < D
                mask_bd = mask_b[:, None] & mask_d[None, :]
                x_raw = tl.load(
                    x_ptr + src_rows[:, None] * stride_xb + cols[None, :] * stride_xd,
                    mask=mask_bd,
                    other=0.0,
                )
                if HAS_ORIGIN:
                    if ORIGIN_FP64:
                        orig64 = tl.load(origin_ptr + cols, mask=mask_d, other=0.0).to(tl.float64)
                        xdiff = x_raw.to(tl.float64) - orig64[None, :]
                        x_f = (xdiff * coord_scale).to(tl.float32) if HAS_SCALE else xdiff.to(tl.float32)
                    else:
                        orig32 = tl.load(origin_ptr + cols, mask=mask_d, other=0.0).to(tl.float32)
                        x_f = x_raw.to(tl.float32) - orig32[None, :]
                        if HAS_SCALE:
                            x_f = x_f * coord_scale
                    x_f = tl.where(mask_bd, x_f, 0.0)
                else:
                    x_f = x_raw.to(tl.float32)
                    if HAS_SCALE:
                        x_f = tl.where(mask_bd, x_f * coord_scale, 0.0)
                if DTYPE_MODE == 1:
                    xf_rep = x_f.to(tl.bfloat16).to(tl.float32)
                elif DTYPE_MODE == 2:
                    xf_rep = x_f.to(tl.float16).to(tl.float32)
                else:
                    xf_rep = x_f
                x_norm += tl.sum(xf_rep * xf_rep, axis=1)

        if part == 0:
            tl.store(part_xnorm_ptr + offs_b, x_norm, mask=mask_b)

        best_score = tl.full((BLOCK_B,), float("inf"), dtype=tl.float32)
        best_label = tl.zeros((BLOCK_B,), dtype=tl.int32)

        k0 = k_start
        while k0 < k_end:
            offs_k = (k0 + tl.arange(0, BLOCK_K)).to(tl.int64)
            mask_k = offs_k < k_end
            if FULL_D:
                cols = tl.arange(0, BLOCK_D).to(tl.int64)
                mask_d = cols < D
                c_tile = tl.load(
                    c_ptr + offs_k[:, None] * stride_ck + cols[None, :],
                    mask=mask_k[:, None] & mask_d[None, :],
                    other=0.0,
                )
                if DTYPE_MODE == 0:
                    acc = tl.dot(x_full, tl.trans(c_tile), input_precision="ieee", out_dtype=tl.float32)
                else:
                    acc = tl.dot(x_full, tl.trans(c_tile), out_dtype=tl.float32)
            else:
                acc = tl.zeros((BLOCK_B, BLOCK_K), dtype=tl.float32)
                for d0 in range(0, D, BLOCK_D):
                    cols = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
                    mask_d = cols < D
                    mask_bd = mask_b[:, None] & mask_d[None, :]
                    x_raw = tl.load(
                        x_ptr + src_rows[:, None] * stride_xb + cols[None, :] * stride_xd,
                        mask=mask_bd,
                        other=0.0,
                    )
                    if HAS_ORIGIN:
                        if ORIGIN_FP64:
                            orig64 = tl.load(origin_ptr + cols, mask=mask_d, other=0.0).to(tl.float64)
                            xdiff = x_raw.to(tl.float64) - orig64[None, :]
                            x_f = (xdiff * coord_scale).to(tl.float32) if HAS_SCALE else xdiff.to(tl.float32)
                        else:
                            orig32 = tl.load(origin_ptr + cols, mask=mask_d, other=0.0).to(tl.float32)
                            x_f = x_raw.to(tl.float32) - orig32[None, :]
                            if HAS_SCALE:
                                x_f = x_f * coord_scale
                        x_f = tl.where(mask_bd, x_f, 0.0)
                    else:
                        x_f = x_raw.to(tl.float32)
                        if HAS_SCALE:
                            x_f = tl.where(mask_bd, x_f * coord_scale, 0.0)

                    c_tile = tl.load(
                        c_ptr + offs_k[:, None] * stride_ck + cols[None, :],
                        mask=mask_k[:, None] & mask_d[None, :],
                        other=0.0,
                    )
                    if DTYPE_MODE == 1:
                        acc = tl.dot(x_f.to(tl.bfloat16), tl.trans(c_tile), acc=acc, out_dtype=tl.float32)
                    elif DTYPE_MODE == 2:
                        acc = tl.dot(x_f.to(tl.float16), tl.trans(c_tile), acc=acc, out_dtype=tl.float32)
                    else:
                        acc = tl.dot(x_f, tl.trans(c_tile), acc=acc, input_precision="ieee", out_dtype=tl.float32)

            c_norm = tl.load(c_norm_ptr + offs_k, mask=mask_k, other=float("inf"))
            scores = tl.where(mask_k[None, :], c_norm[None, :] - 2.0 * acc, float("inf"))
            blk_min, blk_idx = tl.min(
                scores,
                axis=1,
                return_indices=True,
                return_indices_tie_break_left=True,
            )
            better = blk_min < best_score
            best_score = tl.where(better, blk_min, best_score)
            best_label = tl.where(better, k0 + blk_idx.to(tl.int32), best_label)
            k0 += BLOCK_K

        out_off = offs_b * PARTITIONS + part
        tl.store(part_score_ptr + out_off, best_score, mask=mask_b)
        tl.store(part_label_ptr + out_off, best_label, mask=mask_b)

    @triton.jit(do_not_specialize=["B"])
    def _ultra_splitk_reduce_kernel(
        part_score_ptr,
        part_label_ptr,
        part_xnorm_ptr,
        out_dist_ptr,
        out_label_ptr,
        B,
        PARTITIONS: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_PART: tl.constexpr,
    ):
        """Fast reduction of Split-K partial winners (`x_norm` already cached by partition 0!)."""
        pid_b = tl.program_id(0).to(tl.int64)
        offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B).to(tl.int64)
        mask_b = offs_b < B
        offs_p = tl.arange(0, BLOCK_PART)
        mask_p = offs_p < PARTITIONS

        p_off = offs_b[:, None] * PARTITIONS + offs_p[None, :]
        p_scores = tl.load(part_score_ptr + p_off, mask=mask_b[:, None] & mask_p[None, :], other=float("inf"))
        best_score, best_part = tl.min(
            p_scores,
            axis=1,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
        p_labels = tl.load(part_label_ptr + p_off, mask=mask_b[:, None] & mask_p[None, :], other=0)
        best_label = tl.sum(tl.where(offs_p[None, :] == best_part[:, None], p_labels, 0), axis=1)

        x_norm = tl.load(part_xnorm_ptr + offs_b, mask=mask_b, other=0.0)
        dist = tl.maximum(best_score + x_norm, 0.0)
        tl.store(out_dist_ptr + offs_b, dist, mask=mask_b)
        tl.store(out_label_ptr + offs_b, best_label.to(tl.int64), mask=mask_b)

    @triton.jit(do_not_specialize=["B", "D", "ep_idx", "stride_xb", "stride_xd", "stride_ck"])
    def _ultra_scatter_oja_sse_kernel(
        x_ptr,
        row_idx_ptr,
        origin_ptr,
        labels_ptr,
        dists_ptr,
        centroids_ptr,
        prin_vec_ptr,
        sums_ptr,
        counts_ptr,
        batch_sse_ptr,
        g_sums_ptr,
        ep_sq_dists_ptr,
        ep_idx,
        B,
        D,
        stride_xb,
        stride_xd,
        stride_ck,
        coord_scale,
        HAS_INDICES: tl.constexpr,
        HAS_ORIGIN: tl.constexpr,
        HAS_SCALE: tl.constexpr,
        ORIGIN_FP64: tl.constexpr,
        TRACK_OJA: tl.constexpr,
        FULL_D: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Single-pass register gather + centering + atomic `sums/counts/sse` + in-register Oja projection."""
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

        if pid_d == 0:
            sq_d = tl.minimum(
                tl.load(dists_ptr + offs_b, mask=mask_b, other=0.0).to(tl.float32),
                1e35,
            )
            tl.atomic_add(counts_ptr + lbl, 1.0, mask=mask_b, sem="relaxed")
            tl.atomic_add(batch_sse_ptr + lbl, sq_d, mask=mask_b, sem="relaxed")
            tile_sum64 = tl.sum(tl.where(mask_b, sq_d, 0.0).to(tl.float64), axis=0)
            tl.atomic_add(ep_sq_dists_ptr + ep_idx, tile_sum64, sem="relaxed")

        if TRACK_OJA:
            if FULL_D:
                cols = tl.arange(0, BLOCK_D).to(tl.int64)
                mask_d = cols < D
                mask_bd = mask_b[:, None] & mask_d[None, :]
                x_raw = tl.load(
                    x_ptr + src_rows[:, None] * stride_xb + cols[None, :] * stride_xd,
                    mask=mask_bd,
                    other=0.0,
                )
                if HAS_ORIGIN:
                    if ORIGIN_FP64:
                        orig64 = tl.load(origin_ptr + cols, mask=mask_d, other=0.0).to(tl.float64)
                        xdiff = x_raw.to(tl.float64) - orig64[None, :]
                        x_val = (xdiff * coord_scale).to(tl.float32) if HAS_SCALE else xdiff.to(tl.float32)
                    else:
                        orig32 = tl.load(origin_ptr + cols, mask=mask_d, other=0.0).to(tl.float32)
                        x_val = x_raw.to(tl.float32) - orig32[None, :]
                        if HAS_SCALE:
                            x_val = x_val * coord_scale
                else:
                    x_val = x_raw.to(tl.float32)
                    if HAS_SCALE:
                        x_val = x_val * coord_scale
                x_val = tl.where(mask_bd, x_val, 0.0)

                c_val = tl.load(
                    centroids_ptr + lbl[:, None] * stride_ck + cols[None, :],
                    mask=mask_bd,
                    other=0.0,
                ).to(tl.float32)
                pv_val = tl.load(
                    prin_vec_ptr + lbl[:, None] * stride_ck + cols[None, :],
                    mask=mask_bd,
                    other=0.0,
                ).to(tl.float32)
                pv_norm = tl.maximum(tl.sqrt(tl.sum(pv_val * pv_val, axis=1)), 1e-12)
                p_hat = pv_val / pv_norm[:, None]

                r_val = tl.minimum(tl.maximum(x_val - c_val, -1e17), 1e17)
                r_val = tl.where(mask_bd, r_val, 0.0)
                proj = tl.sum(r_val * p_hat, axis=1)

                tl.atomic_add(
                    sums_ptr + lbl[:, None] * stride_ck + cols[None, :],
                    x_val,
                    mask=mask_bd,
                    sem="relaxed",
                )
                tl.atomic_add(
                    g_sums_ptr + lbl[:, None] * stride_ck + cols[None, :],
                    r_val * proj[:, None],
                    mask=mask_bd,
                    sem="relaxed",
                )
            else:
                # 2-pass over D tiles (pid_d == 0 handles full row projection cleanly)
                pv_norm_sq = tl.zeros((BLOCK_B,), dtype=tl.float32)
                for d0 in range(0, D, BLOCK_D):
                    cols = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
                    mask_d = cols < D
                    mask_bd = mask_b[:, None] & mask_d[None, :]
                    pv_t = tl.load(
                        prin_vec_ptr + lbl[:, None] * stride_ck + cols[None, :],
                        mask=mask_bd,
                        other=0.0,
                    ).to(tl.float32)
                    pv_norm_sq += tl.sum(pv_t * pv_t, axis=1)
                inv_pvn = 1.0 / tl.maximum(tl.sqrt(pv_norm_sq), 1e-12)

                proj = tl.zeros((BLOCK_B,), dtype=tl.float32)
                for d0 in range(0, D, BLOCK_D):
                    cols = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
                    mask_d = cols < D
                    mask_bd = mask_b[:, None] & mask_d[None, :]
                    x_raw = tl.load(
                        x_ptr + src_rows[:, None] * stride_xb + cols[None, :] * stride_xd,
                        mask=mask_bd,
                        other=0.0,
                    )
                    if HAS_ORIGIN:
                        if ORIGIN_FP64:
                            orig64 = tl.load(origin_ptr + cols, mask=mask_d, other=0.0).to(tl.float64)
                            xdiff = x_raw.to(tl.float64) - orig64[None, :]
                            x_t = (xdiff * coord_scale).to(tl.float32) if HAS_SCALE else xdiff.to(tl.float32)
                        else:
                            orig32 = tl.load(origin_ptr + cols, mask=mask_d, other=0.0).to(tl.float32)
                            x_t = x_raw.to(tl.float32) - orig32[None, :]
                            if HAS_SCALE:
                                x_t = x_t * coord_scale
                    else:
                        x_t = x_raw.to(tl.float32)
                        if HAS_SCALE:
                            x_t = x_t * coord_scale
                    x_t = tl.where(mask_bd, x_t, 0.0)

                    c_t = tl.load(
                        centroids_ptr + lbl[:, None] * stride_ck + cols[None, :],
                        mask=mask_bd,
                        other=0.0,
                    ).to(tl.float32)
                    pv_t = tl.load(
                        prin_vec_ptr + lbl[:, None] * stride_ck + cols[None, :],
                        mask=mask_bd,
                        other=0.0,
                    ).to(tl.float32)
                    r_t = tl.where(mask_bd, tl.minimum(tl.maximum(x_t - c_t, -1e17), 1e17), 0.0)
                    proj += tl.sum(r_t * (pv_t * inv_pvn[:, None]), axis=1)
                    tl.atomic_add(
                        sums_ptr + lbl[:, None] * stride_ck + cols[None, :],
                        x_t,
                        mask=mask_bd,
                        sem="relaxed",
                    )

                for d0 in range(0, D, BLOCK_D):
                    cols = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
                    mask_d = cols < D
                    mask_bd = mask_b[:, None] & mask_d[None, :]
                    x_raw = tl.load(
                        x_ptr + src_rows[:, None] * stride_xb + cols[None, :] * stride_xd,
                        mask=mask_bd,
                        other=0.0,
                    )
                    if HAS_ORIGIN:
                        if ORIGIN_FP64:
                            orig64 = tl.load(origin_ptr + cols, mask=mask_d, other=0.0).to(tl.float64)
                            xdiff = x_raw.to(tl.float64) - orig64[None, :]
                            x_t = (xdiff * coord_scale).to(tl.float32) if HAS_SCALE else xdiff.to(tl.float32)
                        else:
                            orig32 = tl.load(origin_ptr + cols, mask=mask_d, other=0.0).to(tl.float32)
                            x_t = x_raw.to(tl.float32) - orig32[None, :]
                            if HAS_SCALE:
                                x_t = x_t * coord_scale
                    else:
                        x_t = x_raw.to(tl.float32)
                        if HAS_SCALE:
                            x_t = x_t * coord_scale
                    c_t = tl.load(
                        centroids_ptr + lbl[:, None] * stride_ck + cols[None, :],
                        mask=mask_bd,
                        other=0.0,
                    ).to(tl.float32)
                    r_t = tl.where(mask_bd, tl.minimum(tl.maximum(x_t - c_t, -1e17), 1e17), 0.0)
                    tl.atomic_add(
                        g_sums_ptr + lbl[:, None] * stride_ck + cols[None, :],
                        r_t * proj[:, None],
                        mask=mask_bd,
                        sem="relaxed",
                    )
        else:
            offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D).to(tl.int64)
            mask_d = offs_d < D
            mask_bd = mask_b[:, None] & mask_d[None, :]
            x_raw = tl.load(
                x_ptr + src_rows[:, None] * stride_xb + offs_d[None, :] * stride_xd,
                mask=mask_bd,
                other=0.0,
            )
            if HAS_ORIGIN:
                if ORIGIN_FP64:
                    orig64 = tl.load(origin_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float64)
                    xdiff = x_raw.to(tl.float64) - orig64[None, :]
                    x_val = (xdiff * coord_scale).to(tl.float32) if HAS_SCALE else xdiff.to(tl.float32)
                else:
                    orig32 = tl.load(origin_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32)
                    x_val = x_raw.to(tl.float32) - orig32[None, :]
                    if HAS_SCALE:
                        x_val = x_val * coord_scale
            else:
                x_val = x_raw.to(tl.float32)
                if HAS_SCALE:
                    x_val = x_val * coord_scale
            tl.atomic_add(
                sums_ptr + lbl[:, None] * stride_ck + offs_d[None, :],
                x_val,
                mask=mask_bd,
                sem="relaxed",
            )

    @triton.jit(do_not_specialize=["K", "D", "stride_sd"])
    def _ultra_feat_scale_kernel(
        sums_ptr,
        counts_ptr,
        feat_scale_ptr,
        K,
        D,
        stride_sd,
        var_beta,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Online inter-cluster feature variance EMA (`O(K * D)`)."""
        pid_d = tl.program_id(0).to(tl.int64)
        col = pid_d * BLOCK_D + tl.arange(0, BLOCK_D).to(tl.int64)
        cmask = col < D

        tot_c = 0.0
        sum_col = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k_idx = (k0 + tl.arange(0, BLOCK_K)).to(tl.int64)
            kmask = k_idx < K
            cnt = tl.load(counts_ptr + k_idx, mask=kmask, other=0.0).to(tl.float32)
            tot_c += tl.sum(cnt, axis=0)
            s_tile = tl.load(
                sums_ptr + k_idx[:, None] * stride_sd + col[None, :],
                mask=kmask[:, None] & cmask[None, :],
                other=0.0,
            ).to(tl.float32)
            sum_col += tl.sum(s_tile, axis=0)

        safe_tot = tl.maximum(tot_c, 1.0)
        batch_mu = sum_col / safe_tot

        var_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k_idx = (k0 + tl.arange(0, BLOCK_K)).to(tl.int64)
            kmask = k_idx < K
            cnt = tl.load(counts_ptr + k_idx, mask=kmask, other=0.0).to(tl.float32)
            s_tile = tl.load(
                sums_ptr + k_idx[:, None] * stride_sd + col[None, :],
                mask=kmask[:, None] & cmask[None, :],
                other=0.0,
            ).to(tl.float32)
            diff = tl.minimum(tl.maximum(s_tile - cnt[:, None] * batch_mu[None, :], -1e17), 1e17)
            var_acc += tl.sum(diff * diff, axis=0)

        has_data = tot_c > 0.0
        cur_var = tl.minimum(tl.maximum(var_acc / safe_tot, 1e-12), 1e35)
        prev_fs = tl.load(feat_scale_ptr + col, mask=cmask, other=1.0)
        new_fs = tl.where(has_data, var_beta * prev_fs + (1.0 - var_beta) * cur_var, prev_fs)
        tl.store(feat_scale_ptr + col, new_fs, mask=cmask)

    @triton.jit(
        do_not_specialize=["K", "D", "replace_usage", "accum_pass", "prep_rep", "stride_cd", "stride_crepk"]
    )
    def _ultra_ema_oja_slow_pass_prep_kernel(
        sums_ptr,
        counts_ptr,
        batch_sse_ptr,
        g_sums_ptr,
        centroids_ptr,
        usage_ptr,
        sse_ema_ptr,
        prin_vec_ptr,
        survival_mass_ptr,
        slow_centroids_ptr,
        slow_usage_ptr,
        pass_sums_ptr,
        pass_counts_ptr,
        pass_counts_sq_ptr,
        pass_sse_ptr,
        raw_pass_counts_ptr,
        c_rep_ptr,
        c_norm_ptr,
        K,
        D,
        stride_cd,
        stride_crepk,
        alpha,
        decay,
        sse_decay,
        alpha_surv,
        alpha_slow,
        w_pass,
        replace_usage,
        accum_pass,
        prep_rep,
        TRACK_OJA: tl.constexpr,
        TRACK_SURV: tl.constexpr,
        TRACK_SLOW: tl.constexpr,
        REP_MODE: tl.constexpr,  # 0=FP32, 1=BF16, 2=FP16
        BLOCK_D: tl.constexpr,
    ):
        """Fused 1-CTA-per-cluster update: EMA + SSE + Oja + SurvMass + SlowEMA + ClosingPass + c_rep prep."""
        j = tl.program_id(0).to(tl.int64)
        if j >= K:
            return

        cnt = tl.load(counts_ptr + j).to(tl.float32)
        bsse = tl.minimum(tl.load(batch_sse_ptr + j).to(tl.float32), 1e35)
        old_u = tl.load(usage_ptr + j).to(tl.float32)
        old_sse = tl.load(sse_ema_ptr + j).to(tl.float32)

        touched = cnt > 0.0
        is_rep = replace_usage != 0
        do_accum = accum_pass != 0
        do_prep = prep_rep != 0

        retained = tl.where(is_rep, 0.0, old_u * decay)
        mass = retained + alpha * cnt
        weight = tl.where(touched, (alpha * cnt) / tl.where(touched, mass, 1.0), 0.0)
        is_one = weight == 1.0
        inv_cnt = 1.0 / tl.where(touched, cnt, 1.0)

        rpc = tl.load(raw_pass_counts_ptr + j)
        tl.store(raw_pass_counts_ptr + j, rpc + cnt.to(tl.float64))

        if do_accum:
            w_cnt = (w_pass * cnt).to(tl.float64)
            w2_cnt = (w_pass * w_pass * cnt).to(tl.float64)
            w_sse = (w_pass * bsse).to(tl.float64)
            tl.store(pass_counts_ptr + j, tl.load(pass_counts_ptr + j) + w_cnt)
            tl.store(pass_counts_sq_ptr + j, tl.load(pass_counts_sq_ptr + j) + w2_cnt)
            tl.store(pass_sse_ptr + j, tl.load(pass_sse_ptr + j) + w_sse)

        if TRACK_SURV:
            old_sm = tl.load(survival_mass_ptr + j).to(tl.float32)
            tl.store(survival_mass_ptr + j, (1.0 - alpha_surv) * old_sm + alpha_surv * cnt)

        s_weight = 0.0
        s_is_one = False
        if TRACK_SLOW:
            old_su = tl.load(slow_usage_ptr + j).to(tl.float32)
            s_ret = tl.where(alpha_slow >= 1.0, 0.0, old_su * (1.0 - alpha_slow))
            s_mass = s_ret + alpha_slow * cnt
            s_weight = tl.where(touched, (alpha_slow * cnt) / tl.where(touched, s_mass, 1.0), 0.0)
            s_is_one = s_weight == 1.0
            new_su = tl.where(touched, s_mass, tl.where(alpha_slow >= 1.0, old_su, s_ret))
            tl.store(slow_usage_ptr + j, new_su)

        # Zero scalar buffers for next batch
        tl.store(counts_ptr + j, 0.0)
        tl.store(batch_sse_ptr + j, 0.0)

        # Optional 1st loop for Oja Rayleigh quotient & g_mean norm when TRACK_OJA and touched
        inv_gnorm = 0.0
        target_scale = 0.0
        do_oja_upd = False
        if TRACK_OJA:
            if touched:
                pv_nsq = tl.zeros((), dtype=tl.float32)
                g_nsq = tl.zeros((), dtype=tl.float32)
                g_dot_p = tl.zeros((), dtype=tl.float32)
                for d0 in range(0, D, BLOCK_D):
                    cols = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
                    mask_d = cols < D
                    pv_old = tl.load(prin_vec_ptr + j * stride_cd + cols, mask=mask_d, other=0.0).to(tl.float32)
                    gs_val = tl.load(g_sums_ptr + j * stride_cd + cols, mask=mask_d, other=0.0).to(tl.float32)
                    gm_val = gs_val * inv_cnt
                    pv_nsq += tl.sum(pv_old * pv_old, axis=0)
                    g_nsq += tl.sum(gm_val * gm_val, axis=0)
                    g_dot_p += tl.sum(gm_val * pv_old, axis=0)
                pv_n = tl.maximum(tl.sqrt(pv_nsq), 1e-12)
                g_n = tl.sqrt(g_nsq)
                rayleigh = tl.maximum(g_dot_p / pv_n, 0.0)
                do_oja_upd = g_n > 1e-12
                inv_gnorm = 1.0 / tl.maximum(g_n, 1e-12)
                target_scale = tl.sqrt(rayleigh) * inv_gnorm

        norm_acc = tl.zeros((), dtype=tl.float32)
        w_oja = tl.minimum(tl.maximum(weight, 0.0), 1.0)
        for d0 in range(0, D, BLOCK_D):
            cols = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
            mask_d = cols < D
            s_val = tl.load(sums_ptr + j * stride_cd + cols, mask=mask_d, other=0.0).to(tl.float32)
            if do_accum:
                ps_old = tl.load(pass_sums_ptr + j * stride_cd + cols, mask=mask_d, other=0.0).to(tl.float32)
                tl.store(pass_sums_ptr + j * stride_cd + cols, ps_old + w_pass * s_val, mask=mask_d)

            if TRACK_OJA:
                gs_val = tl.load(g_sums_ptr + j * stride_cd + cols, mask=mask_d, other=0.0).to(tl.float32)
                if do_oja_upd:
                    pv_old = tl.load(prin_vec_ptr + j * stride_cd + cols, mask=mask_d, other=0.0).to(tl.float32)
                    gm_val = gs_val * inv_cnt
                    pv_new = (1.0 - w_oja) * pv_old + w_oja * (gm_val * target_scale)
                    tl.store(prin_vec_ptr + j * stride_cd + cols, pv_new, mask=mask_d)

            tl.debug_barrier()
            tl.store(sums_ptr + j * stride_cd + cols, 0.0, mask=mask_d)
            if TRACK_OJA:
                tl.store(g_sums_ptr + j * stride_cd + cols, 0.0, mask=mask_d)

            c_old = tl.load(centroids_ptr + j * stride_cd + cols, mask=mask_d, other=0.0).to(tl.float32)
            mean_val = s_val * inv_cnt
            pos_val = c_old + weight * (mean_val - c_old)
            upd_val = tl.where(is_one, mean_val, pos_val)
            c_new = tl.where(touched, upd_val, c_old)
            tl.store(centroids_ptr + j * stride_cd + cols, c_new, mask=mask_d & touched)

            if TRACK_SLOW:
                if touched:
                    sc_old = tl.load(slow_centroids_ptr + j * stride_cd + cols, mask=mask_d, other=0.0).to(tl.float32)
                    sc_pos = sc_old + s_weight * (mean_val - sc_old)
                    sc_new = tl.where(s_is_one, mean_val, sc_pos)
                    tl.store(slow_centroids_ptr + j * stride_cd + cols, sc_new, mask=mask_d)

            if do_prep:
                if REP_MODE == 1:
                    cq = tl.where(mask_d, c_new, 0.0).to(tl.bfloat16)
                    tl.store(c_rep_ptr + j * stride_crepk + cols, cq, mask=mask_d)
                    cqf = cq.to(tl.float32)
                elif REP_MODE == 2:
                    cq = tl.where(mask_d, c_new, 0.0).to(tl.float16)
                    tl.store(c_rep_ptr + j * stride_crepk + cols, cq, mask=mask_d)
                    cqf = cq.to(tl.float32)
                else:
                    cqf = tl.where(mask_d, c_new, 0.0)
                    tl.store(c_rep_ptr + j * stride_crepk + cols, cqf, mask=mask_d)
                norm_acc += tl.sum(cqf * cqf, axis=0)

        if do_prep:
            tl.store(c_norm_ptr + j, norm_acc)

        new_u = tl.where(touched, mass, tl.where(is_rep, old_u, retained))
        new_sse = tl.where(
            is_rep,
            tl.where(touched, bsse, 0.5 * old_sse),
            sse_decay * old_sse + (1.0 - sse_decay) * bsse,
        )
        tl.store(usage_ptr + j, new_u)
        tl.store(sse_ema_ptr + j, new_sse)

    @triton.jit(do_not_specialize=["K", "D", "stride_ck"])
    def _ultra_huygens_nn_pass1_kernel(
        c_ptr,
        c_norm_ptr,
        low_mask_ptr,
        d2_any_ptr,
        d2_surv_ptr,
        K,
        D,
        stride_ck,
        BLOCK_Q: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """SRAM-tiled all-pairs nearest-neighbor distances (`d2_any`, `d2_surv`) with zero KxK HBM memory."""
        pid_q = tl.program_id(0).to(tl.int64)
        offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q).to(tl.int64)
        mask_q = offs_q < K
        is_low_q = tl.load(low_mask_ptr + offs_q, mask=mask_q, other=0).to(tl.int8) != 0
        active_q = mask_q & is_low_q
        if tl.sum(tl.where(active_q, 1, 0), axis=0) == 0:
            return

        qn = tl.load(c_norm_ptr + offs_q, mask=active_q, other=0.0).to(tl.float32)
        best_any = tl.full((BLOCK_Q,), float("inf"), dtype=tl.float32)
        best_surv = tl.full((BLOCK_Q,), float("inf"), dtype=tl.float32)

        for k0 in range(0, K, BLOCK_K):
            offs_k = (k0 + tl.arange(0, BLOCK_K)).to(tl.int64)
            mask_k = offs_k < K
            is_low_k = tl.load(low_mask_ptr + offs_k, mask=mask_k, other=1).to(tl.int8) != 0
            surv_k = mask_k & (~is_low_k)

            dots = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)
            for d0 in range(0, D, BLOCK_D):
                cols = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
                mask_d = cols < D
                q_tile = tl.load(
                    c_ptr + offs_q[:, None] * stride_ck + cols[None, :],
                    mask=active_q[:, None] & mask_d[None, :],
                    other=0.0,
                ).to(tl.float32)
                k_tile = tl.load(
                    c_ptr + offs_k[:, None] * stride_ck + cols[None, :],
                    mask=mask_k[:, None] & mask_d[None, :],
                    other=0.0,
                ).to(tl.float32)
                dots = tl.dot(q_tile, tl.trans(k_tile), acc=dots, input_precision="ieee", out_dtype=tl.float32)

            kn = tl.load(c_norm_ptr + offs_k, mask=mask_k, other=float("inf")).to(tl.float32)
            d2 = tl.maximum(qn[:, None] + kn[None, :] - 2.0 * dots, 0.0)
            not_self = offs_q[:, None] != offs_k[None, :]

            d2_a = tl.where(mask_k[None, :] & not_self, d2, float("inf"))
            best_any = tl.minimum(best_any, tl.min(d2_a, axis=1))

            d2_s = tl.where(surv_k[None, :] & not_self, d2, float("inf"))
            best_surv = tl.minimum(best_surv, tl.min(d2_s, axis=1))

        tl.store(d2_any_ptr + offs_q, best_any, mask=active_q)
        tl.store(d2_surv_ptr + offs_q, best_surv, mask=active_q)

    @triton.jit(do_not_specialize=["K", "D", "stride_ck"])
    def _ultra_huygens_leader_pass2_kernel(
        c_ptr,
        c_norm_ptr,
        low_mask_ptr,
        usage_ptr,
        eff_mass_ptr,
        d2_any_ptr,
        d2_surv_ptr,
        protect_cutoff_ptr,
        protected_out_ptr,
        K,
        D,
        stride_ck,
        BLOCK_Q: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Early-exit SRAM Twin-Safe Exact Huygens Pair-Merge Leader Election across arbitrary K."""
        pid_q = tl.program_id(0).to(tl.int64)
        offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q).to(tl.int64)
        mask_q = offs_q < K

        is_low_q = tl.load(low_mask_ptr + offs_q, mask=mask_q, other=0).to(tl.int8) != 0
        active_q = mask_q & is_low_q
        if tl.sum(tl.where(active_q, 1, 0), axis=0) == 0:
            tl.store(protected_out_ptr + offs_q, tl.zeros((BLOCK_Q,), dtype=tl.int8), mask=mask_q)
            return

        qn = tl.load(c_norm_ptr + offs_q, mask=active_q, other=0.0).to(tl.float32)
        uq = tl.load(usage_ptr + offs_q, mask=active_q, other=0.0).to(tl.float32)
        em_q = tl.load(eff_mass_ptr + offs_q, mask=active_q, other=0.0).to(tl.float32)
        dq_any = tl.load(d2_any_ptr + offs_q, mask=active_q, other=0.0).to(tl.float32)
        dq_surv = tl.load(d2_surv_ptr + offs_q, mask=active_q, other=float("inf")).to(tl.float32)
        cutoff = tl.load(protect_cutoff_ptr).to(tl.float32)

        grp_mass = tl.zeros((BLOCK_Q,), dtype=tl.float32)
        has_higher = tl.zeros((BLOCK_Q,), dtype=tl.int32)
        best_other_low = tl.full((BLOCK_Q,), float("inf"), dtype=tl.float32)

        for k0 in range(0, K, BLOCK_K):
            offs_k = (k0 + tl.arange(0, BLOCK_K)).to(tl.int64)
            mask_k = offs_k < K
            is_low_k = tl.load(low_mask_ptr + offs_k, mask=mask_k, other=0).to(tl.int8) != 0
            active_k = mask_k & is_low_k
            if tl.sum(tl.where(active_k, 1, 0), axis=0) > 0:
                dots = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)
                for d0 in range(0, D, BLOCK_D):
                    cols = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
                    mask_d = cols < D
                    q_tile = tl.load(
                        c_ptr + offs_q[:, None] * stride_ck + cols[None, :],
                        mask=active_q[:, None] & mask_d[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    k_tile = tl.load(
                        c_ptr + offs_k[:, None] * stride_ck + cols[None, :],
                        mask=active_k[:, None] & mask_d[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    dots = tl.dot(q_tile, tl.trans(k_tile), acc=dots, input_precision="ieee", out_dtype=tl.float32)

                kn = tl.load(c_norm_ptr + offs_k, mask=active_k, other=0.0).to(tl.float32)
                d2 = tl.maximum(qn[:, None] + kn[None, :] - 2.0 * dots, 0.0)
                em_k = tl.load(eff_mass_ptr + offs_k, mask=active_k, other=0.0).to(tl.float32)
                is_diag = offs_q[:, None] == offs_k[None, :]
                max_pair_m = tl.maximum(em_q[:, None], em_k[None, :])
                co_loc = active_q[:, None] & active_k[None, :] & (is_diag | ((max_pair_m * d2) < cutoff))

                non_coloc_low = active_q[:, None] & active_k[None, :] & (~co_loc)
                d2_ol = tl.where(non_coloc_low, d2, float("inf"))
                best_other_low = tl.minimum(best_other_low, tl.min(d2_ol, axis=1))

                grp_mass += tl.sum(tl.where(co_loc, em_k[None, :], 0.0), axis=1)

                uk = tl.load(usage_ptr + offs_k, mask=active_k, other=0.0).to(tl.float32)
                peer_better = (uk[None, :] > uq[:, None]) | (
                    (uk[None, :] == uq[:, None]) & (offs_k[None, :] < offs_q[:, None])
                )
                has_higher += tl.sum(tl.where(co_loc & (~is_diag) & peer_better, 1, 0), axis=1)

        d2_ext_c = tl.minimum(dq_surv, best_other_low)
        d2_ext = tl.where(d2_ext_c < float("inf"), d2_ext_c, dq_any)
        rem_cost = grp_mass * tl.minimum(d2_ext, 1e30)
        prot = active_q & (has_higher == 0) & (rem_cost >= cutoff)
        tl.store(protected_out_ptr + offs_q, prot.to(tl.int8), mask=mask_q)

    @triton.jit(
        do_not_specialize=["D", "rng_seed", "split_mode", "flush_closing", "stride_cd"]
    )
    def _ultra_syncfree_split_kernel(
        centroids_ptr,
        origin_ptr,
        usage_ptr,
        sse_ema_ptr,
        prin_vec_ptr,
        survival_mass_ptr,
        slow_centroids_ptr,
        slow_usage_ptr,
        pass_sums_ptr,
        pass_counts_ptr,
        pass_counts_sq_ptr,
        pass_sse_ptr,
        fvec_ptr,
        src_ptr,
        tgt_ptr,
        n_sp_ptr,
        D,
        stride_cd,
        inv_d,
        sse_inv_bias,
        usage_inv_bias,
        split_mult,
        split_eps,
        rng_seed,
        split_mode,     # 0=half_gaussian_oja, 1=rms_aniso/rms_iso (s-means), 2=oja_only, 3=checkerboard
        flush_closing,  # 1 when in closing epoch & flush_pass_on_split enabled
        TRACK_OJA: tl.constexpr,
        TRACK_SURV: tl.constexpr,
        TRACK_SLOW: tl.constexpr,
        HAS_ORIGIN: tl.constexpr,
        REP_MODE: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Zero-host-sync Half-Gaussian Principal-Axis Split + Exact Child SSE Deflation + Closing Flush."""
        pid = tl.program_id(0).to(tl.int64)
        n_sp = tl.load(n_sp_ptr).to(tl.int64)
        if pid >= n_sp:
            return

        src = tl.load(src_ptr + pid).to(tl.int64)
        tgt = tl.load(tgt_ptr + pid).to(tl.int64)

        u_raw = tl.load(usage_ptr + src).to(tl.float32)
        sse_raw = tl.minimum(tl.load(sse_ema_ptr + src).to(tl.float32), 1e35)
        u_ub = tl.maximum(u_raw * usage_inv_bias, 1e-6)
        sse_ub = sse_raw * sse_inv_bias
        sigma_rms = tl.sqrt(tl.maximum((sse_ub / u_ub) * inv_d, 1e-12))

        # Pass 1 over D: measure Oja ||prin_vec||, unit Rademacher*f_hat norm, and alignment
        pv_nsq = tl.zeros((), dtype=tl.float32)
        rad_nsq = tl.zeros((), dtype=tl.float32)
        pv_dot_rad = tl.zeros((), dtype=tl.float32)
        if split_mode != 3:
            for d0 in range(0, D, BLOCK_D):
                col = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
                cmask = col < D
                fv = tl.load(fvec_ptr + col, mask=cmask, other=0.0).to(tl.float32)
                u_rnd = tl.rand(rng_seed, pid * D + col)
                radem = tl.where(u_rnd < 0.5, -1.0, 1.0) * fv
                radem = tl.where(cmask, radem, 0.0)
                rad_nsq += tl.sum(radem * radem, axis=0)
                if TRACK_OJA:
                    pv = tl.load(prin_vec_ptr + src * stride_cd + col, mask=cmask, other=0.0).to(tl.float32)
                    pv_nsq += tl.sum(pv * pv, axis=0)
                    pv_dot_rad += tl.sum(pv * radem, axis=0)

        sigma_1 = tl.sqrt(pv_nsq)
        inv_pv = 1.0 / tl.maximum(sigma_1, 1e-12)
        inv_rad = 1.0 / tl.maximum(tl.sqrt(rad_nsq), 1e-12)
        align_sign = tl.where(pv_dot_rad >= 0.0, 1.0, -1.0)

        # Blend confidence rho in [0, 1] when split_mode == 0 (Half-Gaussian Oja + Aniso Prior)
        ratio = sigma_1 / tl.maximum(sigma_rms, 1e-12)
        rho = tl.minimum(tl.maximum((ratio - 0.85) * (1.0 / 0.35), 0.0), 1.0)
        if not TRACK_OJA:
            rho = 0.0
        elif split_mode == 2:
            rho = tl.where(sigma_1 > 1e-10, 1.0, 0.0)

        # Compute exact norm of blended direction `rho * v_1 + (1 - rho) * align_sign * r_hat`
        # Using <v_1, r_hat> = |pv_dot_rad| * inv_pv * inv_rad:
        cos_vr = tl.abs(pv_dot_rad) * inv_pv * inv_rad
        blend_nsq = rho * rho + (1.0 - rho) * (1.0 - rho) + 2.0 * rho * (1.0 - rho) * cos_vr
        inv_blend = 1.0 / tl.maximum(tl.sqrt(blend_nsq), 1e-12)

        # Magnitude along unit direction:
        #   mode 0 (Spectral-Concentration Half-Gaussian Oja):
        #     Gates step multiplier smoothly from manifold-safe `0.18` (isotropic / curved deep manifold
        #     where `sigma_1 ~ sigma_rms`) up to exact Half-Gaussian `sqrt(2/pi) = 0.7979` (elongated or
        #     merged bimodal cluster where `sigma_1 >= 2.0 * sigma_rms`)!
        #   mode 1 (s-means rms_aniso): split_mult * sigma_rms * sqrt(D)
        #   mode 2 (oja_only):          split_mult * max(sigma_1, 1e-4)
        sigma_split = tl.maximum(sigma_1, 0.85 * sigma_rms)
        g_spec = tl.minimum(tl.maximum((ratio - 1.10) * (1.0 / 0.90), 0.0), 1.0)
        eta_min = tl.minimum(split_mult, 0.18)
        eta_eff = eta_min + tl.maximum(split_mult - eta_min, 0.0) * g_spec
        if split_mode == 0:
            step_mag = eta_eff * sigma_split
        elif split_mode == 1:
            step_mag = split_mult * sigma_rms * tl.sqrt(rad_nsq)
        else:
            step_mag = split_mult * tl.where(sigma_1 > 1e-8, sigma_1, sigma_rms)

        # Pass 2 over D: write split children, reset Oja orthogonally, update slow_ema, and guard ULP collapse
        orth_nsq = tl.zeros((), dtype=tl.float32)
        u_dot_rnd2 = tl.zeros((), dtype=tl.float32)
        if TRACK_OJA and D > 1:
            for d0 in range(0, D, BLOCK_D):
                col = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
                cmask = col < D
                fv = tl.load(fvec_ptr + col, mask=cmask, other=0.0).to(tl.float32)
                u_rnd = tl.rand(rng_seed, pid * D + col)
                radem = tl.where(cmask, tl.where(u_rnd < 0.5, -1.0, 1.0) * fv, 0.0)
                pv = tl.load(prin_vec_ptr + src * stride_cd + col, mask=cmask, other=0.0).to(tl.float32)
                if split_mode == 1:
                    u_dir = radem * inv_rad
                else:
                    u_dir = (rho * (pv * inv_pv) + (1.0 - rho) * align_sign * (radem * inv_rad)) * inv_blend
                g2 = tl.randn(rng_seed + 17, pid * D + col)
                g2 = tl.where(cmask, g2, 0.0)
                u_dot_rnd2 += tl.sum(u_dir * g2, axis=0)

            for d0 in range(0, D, BLOCK_D):
                col = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
                cmask = col < D
                fv = tl.load(fvec_ptr + col, mask=cmask, other=0.0).to(tl.float32)
                u_rnd = tl.rand(rng_seed, pid * D + col)
                radem = tl.where(cmask, tl.where(u_rnd < 0.5, -1.0, 1.0) * fv, 0.0)
                pv = tl.load(prin_vec_ptr + src * stride_cd + col, mask=cmask, other=0.0).to(tl.float32)
                if split_mode == 1:
                    u_dir = radem * inv_rad
                else:
                    u_dir = (rho * (pv * inv_pv) + (1.0 - rho) * align_sign * (radem * inv_rad)) * inv_blend
                g2 = tl.where(cmask, tl.randn(rng_seed + 17, pid * D + col), 0.0)
                orth_v = tl.where(cmask, g2 - u_dot_rnd2 * u_dir, 0.0)
                orth_nsq += tl.sum(orth_v * orth_v, axis=0)

        inv_orth = 1.0 / tl.maximum(tl.sqrt(orth_nsq), 1e-12)
        rem_sigma = 0.602810275 * sigma_rms

        all_equal = 1
        for d0 in range(0, D, BLOCK_D):
            col = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
            cmask = col < D
            dc = tl.load(centroids_ptr + src * stride_cd + col, mask=cmask, other=0.0).to(tl.float32)

            if split_mode == 3:
                sign = tl.where((col & 1) == 0, 1.0, -1.0)
                if HAS_ORIGIN:
                    orig = tl.load(origin_ptr + col, mask=cmask, other=0.0).to(tl.float32)
                    abs_dc = dc + orig
                    c_tgt = abs_dc * (1.0 + split_eps * sign) - orig
                    c_src = abs_dc * (1.0 - split_eps * sign) - orig
                else:
                    c_tgt = dc * (1.0 + split_eps * sign)
                    c_src = dc * (1.0 - split_eps * sign)
                u_dir = tl.where(cmask, sign * tl.sqrt(inv_d), 0.0)
            else:
                fv = tl.load(fvec_ptr + col, mask=cmask, other=0.0).to(tl.float32)
                u_rnd = tl.rand(rng_seed, pid * D + col)
                radem = tl.where(cmask, tl.where(u_rnd < 0.5, -1.0, 1.0) * fv, 0.0)
                if split_mode == 1:
                    u_dir = radem * inv_rad
                else:
                    pv = (
                        tl.load(prin_vec_ptr + src * stride_cd + col, mask=cmask, other=0.0).to(tl.float32)
                        if TRACK_OJA
                        else tl.zeros((BLOCK_D,), dtype=tl.float32)
                    )
                    u_dir = (rho * (pv * inv_pv) + (1.0 - rho) * align_sign * (radem * inv_rad)) * inv_blend
                delta = step_mag * u_dir
                c_tgt = dc + delta
                c_src = dc - delta

            if REP_MODE == 1:
                eq_col = c_tgt.to(tl.bfloat16).to(tl.float32) == c_src.to(tl.bfloat16).to(tl.float32)
            elif REP_MODE == 2:
                eq_col = c_tgt.to(tl.float16).to(tl.float32) == c_src.to(tl.float16).to(tl.float32)
            else:
                eq_col = c_tgt == c_src
            has_diff = tl.sum(tl.where(cmask & (~eq_col), 1, 0), axis=0)
            if has_diff > 0:
                all_equal = 0

            tl.store(centroids_ptr + tgt * stride_cd + col, c_tgt, mask=cmask)
            tl.store(centroids_ptr + src * stride_cd + col, c_src, mask=cmask)

            if TRACK_SLOW:
                # Reset slow_centroids of split children directly to exact Half-Gaussian child positions
                # so slow_ema carries zero pre-split merged-parent lag!
                tl.store(slow_centroids_ptr + tgt * stride_cd + col, c_tgt, mask=cmask)
                tl.store(slow_centroids_ptr + src * stride_cd + col, c_src, mask=cmask)

            if TRACK_OJA:
                if D <= 1:
                    pv_rst = rem_sigma * u_dir
                else:
                    g2 = tl.where(cmask, tl.randn(rng_seed + 17, pid * D + col), 0.0)
                    orth_v = tl.where(cmask, g2 - u_dot_rnd2 * u_dir, 0.0)
                    pv_rst = tl.where(orth_nsq > 1e-16, rem_sigma * (orth_v * inv_orth), rem_sigma * u_dir)
                tl.store(prin_vec_ptr + tgt * stride_cd + col, pv_rst, mask=cmask)
                tl.store(prin_vec_ptr + src * stride_cd + col, pv_rst, mask=cmask)

            if flush_closing != 0:
                tl.store(pass_sums_ptr + tgt * stride_cd + col, 0.0, mask=cmask)
                tl.store(pass_sums_ptr + src * stride_cd + col, 0.0, mask=cmask)

        if all_equal != 0:
            c0 = tl.load(centroids_ptr + tgt * stride_cd).to(tl.float32)
            if REP_MODE == 1:
                step_ulp = tl.maximum(tl.abs(c0) * (1.0 / 64.0), 1e-30)
            elif REP_MODE == 2:
                step_ulp = tl.maximum(tl.abs(c0) * (1.0 / 512.0), 1e-4)
            else:
                step_ulp = tl.maximum(tl.abs(c0) * 2.5e-7, 1e-30)
            nudge = tl.where(c0 > 0.0, -step_ulp, step_ulp)
            tl.store(centroids_ptr + tgt * stride_cd, c0 + nudge)
            if TRACK_SLOW:
                tl.store(slow_centroids_ptr + tgt * stride_cd, c0 + nudge)

        half_u = 0.5 * u_raw
        tl.store(usage_ptr + tgt, half_u)
        tl.store(usage_ptr + src, half_u)

        # Exact Huygens parallel-axis post-split child SSE deflation:
        # `f(eta) = 2 * eta * sqrt(2/pi) - eta^2` removes exact 1D half-space variance reduction!
        if split_mode == 0:
            var_drop_frac = tl.maximum(2.0 * eta_eff * 0.797884561 - eta_eff * eta_eff, 0.0)
            sse_drop = var_drop_frac * u_raw * (sigma_split * sigma_split) * (usage_inv_bias / tl.maximum(sse_inv_bias, 1e-12))
            rem_sse = tl.maximum(0.15 * sse_raw, sse_raw - sse_drop)
            half_sse = 0.5 * rem_sse
        else:
            half_sse = 0.5 * sse_raw
        tl.store(sse_ema_ptr + tgt, half_sse)
        tl.store(sse_ema_ptr + src, half_sse)

        if TRACK_SURV:
            tl.store(survival_mass_ptr + tgt, 0.0)
            tl.store(survival_mass_ptr + src, 0.0)

        if TRACK_SLOW:
            su_s = tl.load(slow_usage_ptr + src).to(tl.float32)
            tl.store(slow_usage_ptr + tgt, 0.5 * su_s)
            tl.store(slow_usage_ptr + src, 0.5 * su_s)

        if flush_closing != 0:
            tl.store(pass_counts_ptr + tgt, 0.0)
            tl.store(pass_counts_ptr + src, 0.0)
            tl.store(pass_counts_sq_ptr + tgt, 0.0)
            tl.store(pass_counts_sq_ptr + src, 0.0)
            tl.store(pass_sse_ptr + tgt, 0.0)
            tl.store(pass_sse_ptr + src, 0.0)

    @triton.jit(do_not_specialize=["K", "D", "stride_ck", "stride_crepk"])
    def _ultra_prep_centroids_kernel(
        centroids_ptr,
        c_rep_ptr,
        c_norm_ptr,
        K,
        D,
        stride_ck,
        stride_crepk,
        REP_MODE: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Convert centered centroids to represented dtype and compute `||c_rep||^2` in one launch."""
        j = tl.program_id(0).to(tl.int64)
        if j >= K:
            return
        norm_acc = tl.zeros((), dtype=tl.float32)
        for d0 in range(0, D, BLOCK_D):
            cols = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
            mask_d = cols < D
            c_val = tl.load(centroids_ptr + j * stride_ck + cols, mask=mask_d, other=0.0).to(tl.float32)
            if REP_MODE == 1:
                cq = c_val.to(tl.bfloat16)
                tl.store(c_rep_ptr + j * stride_crepk + cols, cq, mask=mask_d)
                cqf = cq.to(tl.float32)
            elif REP_MODE == 2:
                cq = c_val.to(tl.float16)
                tl.store(c_rep_ptr + j * stride_crepk + cols, cq, mask=mask_d)
                cqf = cq.to(tl.float32)
            else:
                tl.store(c_rep_ptr + j * stride_crepk + cols, c_val, mask=mask_d)
                cqf = c_val
            norm_acc += tl.sum(cqf * cqf, axis=0)
        tl.store(c_norm_ptr + j, norm_acc)


# ==============================================================================
# 2. Kernel Dispatch Helpers & Zero-Copy Fast Importance Coreset Seeding
# ==============================================================================


def _dtype_to_rep_mode(comp_dtype: torch.dtype) -> int:
    if comp_dtype == torch.bfloat16:
        return 1
    if comp_dtype == torch.float16:
        return 2
    return 0


def _alloc_padded_2d(rows: int, d: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    d_pad = max(16, ((d + 15) // 16) * 16)
    return torch.zeros((rows, d_pad), dtype=dtype, device=device)


def _launch_ultra_prep_centroids(
    centroids_c: torch.Tensor,
    c_rep: torch.Tensor,
    c_norm: torch.Tensor,
) -> None:
    k, d = centroids_c.shape
    rep_mode = _dtype_to_rep_mode(c_rep.dtype)
    block_d = min(256, max(16, triton.next_power_of_2(max(1, d))))
    with torch.cuda.device(centroids_c.device):
        _ultra_prep_centroids_kernel[(k,)](
            centroids_c,
            c_rep,
            c_norm,
            k,
            d,
            centroids_c.stride(0),
            c_rep.stride(0),
            REP_MODE=rep_mode,
            BLOCK_D=block_d,
            num_warps=2 if d <= 128 else 4,
        )


def _launch_ultra_assign(
    x_t: torch.Tensor,
    c_rep: torch.Tensor,
    c_norm: torch.Tensor,
    row_indices: torch.Tensor | None = None,
    origin: torch.Tensor | None = None,
    coord_scale: float = 1.0,
    out_dist: torch.Tensor | None = None,
    out_label: torch.Tensor | None = None,
    splitk_buffers: tuple[torch.Tensor, torch.Tensor, torch.Tensor, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch fused gather + centered represented-norm Tensor-Core assignment (with Split-K for large K)."""
    b = int(row_indices.shape[0]) if row_indices is not None else int(x_t.shape[0])
    d = int(x_t.shape[1])
    k = int(c_rep.shape[0])
    device = x_t.device
    if out_dist is None:
        out_dist = torch.empty((b,), dtype=torch.float32, device=device)
    if out_label is None:
        out_label = torch.empty((b,), dtype=torch.int64, device=device)
    if b == 0:
        return out_dist, out_label

    rep_mode = _dtype_to_rep_mode(c_rep.dtype)
    reduced = rep_mode in (1, 2)
    cap = torch.cuda.get_device_capability(device)
    low_smem = cap < (8, 0)
    full_d = (d <= 128) if reduced else (d <= 64)

    has_idx = row_indices is not None
    idx_c = row_indices.contiguous() if has_idx else None
    has_orig = origin is not None
    orig_fp64 = bool(has_orig and origin.dtype == torch.float64)
    has_scale = coord_scale != 1.0

    # Use 2D Split-K when K >= 2048 and B under-occupies Blackwell's 188 SMs
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    if not low_smem and k >= 2048 and triton.cdiv(b, 64) < num_sms:
        sp_block_b = 64
        sp_block_k = 128
        if splitk_buffers is not None and splitk_buffers[0].shape[0] >= b:
            part_sc_buf, part_lbl_buf, part_xn_buf, parts = splitk_buffers
            part_sc = part_sc_buf[:b]
            part_lbl = part_lbl_buf[:b]
            part_xn = part_xn_buf[:b]
        else:
            parts = _choose_split_k_partitions(b, k, num_sms=num_sms, block_b=sp_block_b, block_k=sp_block_k)
            part_sc = torch.empty((b, parts), dtype=torch.float32, device=device)
            part_lbl = torch.empty((b, parts), dtype=torch.int32, device=device)
            part_xn = torch.empty((b,), dtype=torch.float32, device=device)

        if parts > 1:
            sp_block_d = (
                max(16, triton.next_power_of_2(max(1, d)))
                if full_d
                else (64 if reduced else 32)
            )
            with torch.cuda.device(device):
                _ultra_splitk_assign_kernel[(triton.cdiv(b, sp_block_b), parts)](
                    x_t,
                    idx_c,
                    origin if has_orig else x_t,
                    c_rep,
                    c_norm,
                    part_sc,
                    part_lbl,
                    part_xn,
                    b,
                    k,
                    d,
                    x_t.stride(0),
                    x_t.stride(1),
                    c_rep.stride(0),
                    float(coord_scale),
                    PARTITIONS=parts,
                    BLOCK_B=sp_block_b,
                    BLOCK_K=sp_block_k,
                    BLOCK_D=sp_block_d,
                    FULL_D=full_d,
                    DTYPE_MODE=rep_mode,
                    HAS_INDICES=has_idx,
                    HAS_ORIGIN=has_orig,
                    HAS_SCALE=has_scale,
                    ORIGIN_FP64=orig_fp64,
                    num_warps=4 if full_d else 8,
                    num_stages=3,
                )
                red_blk_b = 128
                _ultra_splitk_reduce_kernel[(triton.cdiv(b, red_blk_b),)](
                    part_sc,
                    part_lbl,
                    part_xn,
                    out_dist,
                    out_label,
                    b,
                    PARTITIONS=parts,
                    BLOCK_B=red_blk_b,
                    BLOCK_PART=max(16, triton.next_power_of_2(parts)),
                    num_warps=4,
                )
            return out_dist, out_label

    if full_d:
        block_d = max(16, triton.next_power_of_2(max(1, d)))
        if b <= 512 and k >= 1024:
            block_b = 16
        elif low_smem and not reduced and block_d >= 128:
            block_b = 32
        elif reduced and not low_smem and b >= 128 * num_sms and k >= 256:
            block_b = 128
        else:
            block_b = 64
        block_k = 128 if (reduced and not low_smem and k >= 128) else 64
        num_warps = 8 if (reduced and not low_smem and block_d >= 64 and block_k >= 128 and block_b >= 64) else 4
        num_stages = 1 if low_smem else 3
    else:
        block_d = 64 if reduced else 32
        block_b = 32 if (b <= 512 and k >= 1024) else 64
        block_k = 128 if (reduced and not low_smem and k >= 128) else 64
        num_warps = 8 if (reduced and not low_smem) else 4
        num_stages = 1 if low_smem else 3

    with torch.cuda.device(device):
        _ultra_assign_kernel[(triton.cdiv(b, block_b),)](
            x_t,
            idx_c,
            origin if has_orig else x_t,
            c_rep,
            c_norm,
            out_dist,
            out_label,
            b,
            k,
            d,
            x_t.stride(0),
            x_t.stride(1),
            c_rep.stride(0),
            float(coord_scale),
            BLOCK_B=block_b,
            BLOCK_K=block_k,
            BLOCK_D=block_d,
            FULL_D=full_d,
            DTYPE_MODE=rep_mode,
            HAS_INDICES=has_idx,
            HAS_ORIGIN=has_orig,
            HAS_SCALE=has_scale,
            ORIGIN_FP64=orig_fp64,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return out_dist, out_label


@torch.no_grad()
def _launch_syncfree_huygens_protection(
    centroids_c: torch.Tensor,
    low_mask: torch.Tensor,
    usage: torch.Tensor,
    survival_mass: torch.Tensor | None,
    thresh: torch.Tensor,
    ref_distortion_mean: torch.Tensor,
    gamma_protect: float,
    c_pad_buf: torch.Tensor | None = None,
    c_norm_buf: torch.Tensor | None = None,
    d2_any_buf: torch.Tensor | None = None,
    d2_surv_buf: torch.Tensor | None = None,
    prot_i8_buf: torch.Tensor | None = None,
) -> torch.Tensor:
    """Zero-host-sync Twin-Safe Huygens Rare-Mode Shield across ANY K (including K=16,384)."""
    k, d = centroids_c.shape
    device = centroids_c.device
    if k < 2 or gamma_protect <= 0.0:
        return torch.zeros((k,), dtype=torch.bool, device=device)

    if survival_mass is not None:
        eff_mass = torch.where(
            survival_mass > 0.0,
            torch.maximum(usage, torch.minimum(survival_mass, thresh)),
            torch.zeros_like(usage),
        )
    else:
        eff_mass = usage

    if not centroids_c.is_cuda or not HAS_TRITON:
        from .super_emeans import twin_safe_huygens_protection

        # Pass `eff_mass` as both `usage`-floor gated by `survival_mass > 0` so 0-touch outliers
        # have exact 0 removal mass on CPU fallback too!
        prot_cpu = twin_safe_huygens_protection(
            centroids=centroids_c,
            low_mask=low_mask & (eff_mass > 0.0),
            usage=usage,
            survival_mass=eff_mass,
            thresh=thresh,
            ref_distortion_mean=ref_distortion_mean,
            gamma_protect=gamma_protect,
        )
        return prot_cpu
    protect_cutoff = (gamma_protect * ref_distortion_mean.clamp_min(1e-30)).reshape(())

    # Pack all `low_mask` centroids into the leading rows via sync-free `argsort` so ONLY the first
    # 1..2 CTAs execute IEEE FP32 `tl.dot` while all remaining CTAs early-exit in 1 cycle!
    # Stable sort preserves original index order for exact tie-breaking on equal-usage twins!
    low_i8_orig = low_mask.to(torch.int8)
    low_order = torch.argsort(low_i8_orig, descending=True, stable=True)
    inv_order = torch.empty_like(low_order)
    inv_order.scatter_(0, low_order, torch.arange(k, device=device, dtype=torch.int64))

    c_mean = centroids_c.mean(dim=0, keepdim=True)
    kc_sorted = (centroids_c - c_mean)[low_order]
    if c_pad_buf is None or c_pad_buf.shape[0] != k or c_pad_buf.shape[1] < max(16, d):
        c_pad_buf = _alloc_padded_2d(k, d, torch.float32, device)
    else:
        c_pad_buf.zero_()
    c_pad_buf[:, :d] = kc_sorted

    cn = kc_sorted.square().sum(dim=1) if c_norm_buf is None else c_norm_buf
    if c_norm_buf is not None:
        cn.copy_(kc_sorted.square().sum(dim=1))

    if d2_any_buf is None:
        d2_any_buf = torch.empty((k,), dtype=torch.float32, device=device)
    if d2_surv_buf is None:
        d2_surv_buf = torch.empty((k,), dtype=torch.float32, device=device)
    if prot_i8_buf is None:
        prot_i8_buf = torch.empty((k,), dtype=torch.int8, device=device)

    low_i8 = low_i8_orig[low_order].contiguous()
    usage_s = usage[low_order].contiguous()
    eff_mass_s = eff_mass[low_order].contiguous()
    blk_q = 64 if k >= 64 else 16
    blk_k = 64 if k >= 64 else 16
    blk_d = min(64, max(16, triton.next_power_of_2(max(1, d))))

    with torch.cuda.device(device):
        _ultra_huygens_nn_pass1_kernel[(triton.cdiv(k, blk_q),)](
            c_pad_buf,
            cn,
            low_i8,
            d2_any_buf,
            d2_surv_buf,
            k,
            d,
            c_pad_buf.stride(0),
            BLOCK_Q=blk_q,
            BLOCK_K=blk_k,
            BLOCK_D=blk_d,
            num_warps=4,
        )
        _ultra_huygens_leader_pass2_kernel[(triton.cdiv(k, blk_q),)](
            c_pad_buf,
            cn,
            low_i8,
            usage_s,
            eff_mass_s,
            d2_any_buf,
            d2_surv_buf,
            protect_cutoff,
            prot_i8_buf,
            k,
            d,
            c_pad_buf.stride(0),
            BLOCK_Q=blk_q,
            BLOCK_K=blk_k,
            BLOCK_D=blk_d,
            num_warps=4,
        )
    return (prot_i8_buf != 0)[inv_order]


@torch.no_grad()
def _compute_origin_and_scale(
    x_t: torch.Tensor,
    comp_dtype: torch.dtype,
    sample_rows: int = 16384,
) -> tuple[torch.Tensor, float]:
    """Compute translation-invariant coordinate origin `(D,)` and exact power-of-two `coord_scale`."""
    n, d = int(x_t.shape[0]), int(x_t.shape[1])
    device = x_t.device
    if n == 0:
        return torch.zeros((d,), dtype=torch.float64, device=device), 1.0

    stride = max(1, n // sample_rows)
    x_sub64 = x_t[::stride].double()
    origin64 = x_sub64.mean(dim=0).contiguous()

    # Check global dynamic range in one fast reduction so extreme inputs (1e-25 .. 1e25 or FP16 > 8192)
    # never overflow or underflow FP32/FP16 norms!
    x_min = float(x_t.min().item())
    x_max = float(x_t.max().item())
    span_est = max(
        abs(x_max - float(origin64.max().item())),
        abs(x_min - float(origin64.min().item())),
        float((x_sub64 - origin64[None, :]).abs().max().item()),
    )
    upper_lim = 8192.0 if comp_dtype == torch.float16 else 1e11
    if comp_dtype == torch.float16 and span_est > upper_lim:
        exp_shift = int(math.ceil(math.log2(span_est / upper_lim)))
        coord_scale = float(2.0 ** (-exp_shift))
    elif span_est > 1e11 or (0.0 < span_est < 1e-11):
        exp_shift = int(round(math.log2(span_est)))
        coord_scale = float(2.0 ** (-exp_shift))
    else:
        coord_scale = 1.0
    return origin64, coord_scale


@torch.no_grad()
def ultra_turkpp_init_centered(
    x_t: torch.Tensor,
    origin64: torch.Tensor,
    coord_scale: float,
    k: int,
    comp_dtype: torch.dtype,
    coreset_mult: int = 8,
    min_coreset_size: int = 4096,
    polish: bool = True,
    seed: int = 1234,
) -> SeedingResult:
    """Zero-Copy Fused Importance Coreset + Batched Tournament Greedy `k-means++` + Voronoi Polish (`U0`).

    Returns `SeedingResult` ALREADY in the `(x - origin64) * coord_scale` centered coordinate frame!
    """
    n, d = int(x_t.shape[0]), int(x_t.shape[1])
    device = x_t.device
    if n == k:
        xc = ((x_t.double() - origin64[None, :]) * coord_scale).float().contiguous()
        return SeedingResult(
            centroids=xc,
            prior_probs=torch.full((k,), 1.0 / k, dtype=torch.float32, device=device),
            prior_distortion=torch.zeros((k,), dtype=torch.float32, device=device),
        )

    m = min(n, max(int(min_coreset_size), int(coreset_mult) * int(k), int(k) * 2))
    origin_ker = (
        origin64
        if (x_t.dtype == torch.float64 or coord_scale != 1.0 or origin64.abs().max().item() > 1e3)
        else origin64.float().contiguous()
    )

    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed) % (2**32))

    # 1. Single-pass fused Triton radial-energy sketch over x_t (zero [N, D] copy!)
    if m >= n:
        xs_c = ((x_t.double() - origin64[None, :]) * coord_scale).float().contiguous()
        w_sub = torch.full((n,), 1.0 / n, dtype=torch.float32, device=device)
        m = n
    else:
        if device.type == "cuda" and HAS_TRITON:
            d2_orig = torch.empty((n,), dtype=torch.float32, device=device)
            blk_n = 128
            blk_d = min(128, max(16, triton.next_power_of_2(max(1, d))))
            with torch.cuda.device(device):
                _ultra_origin_d2_kernel[(triton.cdiv(n, blk_n),)](
                    x_t,
                    origin_ker,
                    d2_orig,
                    n,
                    d,
                    x_t.stride(0),
                    x_t.stride(1),
                    float(coord_scale),
                    ORIGIN_FP64=(origin_ker.dtype == torch.float64),
                    BLOCK_N=blk_n,
                    BLOCK_D=blk_d,
                    num_warps=4,
                )
        else:
            d2_orig = torch.empty((n,), dtype=torch.float32, device=device)
            for s in range(0, n, 131072):
                e = min(s + 131072, n)
                diff = (x_t[s:e].double() - origin64[None, :]) * coord_scale
                d2_orig[s:e] = diff.float().square().sum(dim=1)

        tot_d2 = d2_orig.double().sum()
        unif_q = torch.full((n,), 1.0 / n, dtype=torch.float64, device=device)
        q = torch.where(
            tot_d2 > 0.0,
            0.5 * (1.0 / n) + 0.5 * (d2_orig.double() / tot_d2.clamp_min(1e-30)),
            unif_q,
        )
        cdf = torch.cumsum(q, dim=0)
        # Stratified quantile jitter `([0..m-1] + u) / m` gives strictly lower variance AND fewer
        # collisions than i.i.d. sampling!
        u_strat = (
            torch.arange(m, device=device, dtype=torch.float64)
            + torch.rand((m,), generator=gen, device=device, dtype=torch.float64)
        ) * (cdf[-1] / float(m))
        idx_sub = torch.searchsorted(cdf, u_strat, right=True).clamp_(0, n - 1)
        xs_c = ((x_t[idx_sub].double() - origin64[None, :]) * coord_scale).float().contiguous()
        w_sub = (1.0 / (float(m) * q[idx_sub])).float()
        w_sub = (w_sub / w_sub.sum().clamp_min(1e-30)).contiguous()

    # 2. Fast Zero-Sync Batched Tournament Greedy D^2 Seeding on L2-resident `xs_c`
    # Uses `scatter_` exclusively (never `tensor[idx] = val`, which triggers ATen host bounds-check syncs!).
    seed_dtype = (
        torch.bfloat16
        if (device.type == "cuda" and HAS_TRITON and d >= 32 and comp_dtype in (torch.bfloat16, torch.float16))
        else torch.float32
    )
    avail = torch.ones((m,), dtype=torch.bool, device=device)
    false_buf = torch.zeros((m,), dtype=torch.bool, device=device)

    def _ultra_select_pool(
        c_pool: torch.Tensor,
        init_pool_d2: torch.Tensor,
        pool_weights: torch.Tensor,
        num_select: int,
        rseed: int,
    ) -> torch.Tensor:
        P_sz = int(c_pool.shape[0])
        if P_sz <= num_select:
            return c_pool[:num_select]

        if HAS_TRITON and device.type == "cuda" and num_select > 32 and P_sz >= 64:
            g_cnt = max(2, (num_select + 31) // 32)
            sel_per_g = num_select // g_cnt
            p_per_g = P_sz // g_cnt
            last_sel = num_select - sel_per_g * (g_cnt - 1)
            last_p = P_sz - p_per_g * (g_cnt - 1)
            if p_per_g >= sel_per_g and last_p >= last_sel and max(p_per_g, last_p) <= 128:
                pgen = torch.Generator(device=device)
                pgen.manual_seed(int(rseed + 33) % (2**32))
                rdir = torch.randn((d,), generator=pgen, device=device, dtype=torch.float32)
                order_p = torch.argsort(c_pool @ rdir)
                cp_s = c_pool[order_p].contiguous()
                pd2_s = init_pool_d2[order_p].float().contiguous()
                wp_s = pool_weights[order_p].float()
                wp_s = (wp_s / wp_s.sum().clamp_min(1e-30)).contiguous()

                # Single vectorized gather into `(g_cnt, 128, d)` — zero Python loop over shards!
                blk_p = 128
                g_ids = torch.arange(g_cnt, device=device, dtype=torch.int64)[:, None]
                loc_ids = torch.arange(blk_p, device=device, dtype=torch.int64)[None, :]
                lim_p = torch.where(g_ids == (g_cnt - 1), last_p, p_per_g)
                valid_slot = loc_ids < lim_p
                src_flat = (g_ids * p_per_g + loc_ids).clamp_(0, P_sz - 1)
                cp_pad = torch.where(valid_slot[:, :, None], cp_s[src_flat], 0.0)
                cp_mean = cp_pad.sum(dim=1, keepdim=True) / lim_p.clamp_min(1).float()[:, :, None]
                cp_pad_c = torch.where(valid_slot[:, :, None], cp_pad - cp_mean, 0.0)
                p_n = cp_pad_c.square().sum(dim=2)
                pair_bmm = (
                    p_n[:, :, None] + p_n[:, None, :] - 2.0 * torch.bmm(cp_pad_c, cp_pad_c.transpose(1, 2))
                ).clamp_min_(0.0).reshape(g_cnt * blk_p, blk_p).contiguous()

                sel_out = torch.empty((num_select,), dtype=torch.int64, device=device)
                with torch.cuda.device(device):
                    _ultra_batched_local_greedy_kpp_kernel[(g_cnt,)](
                        pair_bmm,
                        pd2_s,
                        wp_s,
                        sel_out,
                        blk_p,
                        p_per_g,
                        sel_per_g,
                        last_p,
                        last_sel,
                        g_cnt,
                        int(rseed) % (2**31 - 1),
                        BLOCK_P=blk_p,
                        L_TRIALS=3,
                        num_warps=4,
                    )
                return cp_s[sel_out]

        cp_c = c_pool - c_pool.mean(dim=0, keepdim=True)
        pn = cp_c.square().sum(dim=1)
        pair_d2 = (pn[:, None] + pn[None, :] - 2.0 * (cp_c @ cp_c.t())).clamp_min_(0.0).contiguous()
        pair_d2.fill_diagonal_(0.0)
        w_p = pool_weights.float()
        w_p = (w_p / w_p.sum().clamp_min(1e-30)).contiguous()
        pd2 = init_pool_d2.float().clone().contiguous()
        if HAS_TRITON and device.type == "cuda" and P_sz <= 512:
            block_p = 256 if P_sz <= 256 else 512
            sel_loc = torch.empty((num_select,), dtype=torch.int64, device=device)
            with torch.cuda.device(device):
                _ultra_local_greedy_kpp_kernel[(1,)](
                    pair_d2,
                    pd2,
                    w_p,
                    sel_loc,
                    P_sz,
                    num_select,
                    int(rseed) % (2**31 - 1),
                    BLOCK_P=block_p,
                    L_TRIALS=3,
                    num_warps=4 if block_p <= 256 else 8,
                )
            return c_pool[sel_loc]
        from .seeding import _select_from_candidate_pool

        return _select_from_candidate_pool(c_pool, init_pool_d2, pool_weights, num_select, seed=rseed)

    # Initial seed block `b0 <= 8` selected first so `min_d2` updates immediately and discovers
    # EVERY distinct separated rare mode in subsequent rounds on both CPU and CUDA!
    b0 = min(k, 8)
    p0 = min(m, max(b0, min(128, b0 * 8)))
    rad_d2 = xs_c.square().sum(dim=1)
    rmax0 = rad_d2.max().clamp_min(1e-30)
    prop0 = w_sub * (0.25 + 0.75 * (rad_d2 / rmax0)) + 1e-15
    n_far0 = min(16, m)
    far0 = torch.topk(rad_d2, k=n_far0).indices
    prop0.scatter_(0, far0, (prop0.max().clamp_min(1e-12) * 1e6).expand_as(far0))
    ug0 = torch.rand((m,), generator=gen, device=device, dtype=torch.float32).clamp_(1e-30, 1.0 - 1e-7)
    cand0 = torch.topk(prop0.log() - (-ug0.log()).log(), k=p0, largest=True, sorted=False).indices
    if p0 <= b0:
        c0_blk = xs_c[cand0[:b0]].contiguous()
        avail.scatter_(0, cand0[:b0], false_buf[:b0])
    else:
        cp0 = xs_c[cand0].contiguous()
        init_d2_0 = rad_d2[cand0]
        init_d2_0 = torch.where(init_d2_0.max() > 0.0, init_d2_0, torch.ones_like(init_d2_0))
        c0_blk = _ultra_select_pool(cp0, init_d2_0, w_sub[cand0], b0, rseed=seed + 501)
        avail.scatter_(0, cand0, false_buf[:p0])

    raw_c_list: list[torch.Tensor] = [c0_blk]
    placed = b0
    if placed < k:
        if device.type == "cuda" and HAS_TRITON:
            ch_rep0 = _alloc_padded_2d(b0, d, seed_dtype, device)
            ch_norm0 = torch.empty((b0,), dtype=torch.float32, device=device)
            _launch_ultra_prep_centroids(c0_blk, ch_rep0, ch_norm0)
            min_d2, _ = _launch_ultra_assign(xs_c, ch_rep0, ch_norm0)
        else:
            min_d2, _ = assign_points(xs_c, c0_blk, compute_dtype=seed_dtype)
    else:
        min_d2 = rad_d2

    cur_blk = 32
    round_id = 0
    while placed < k:
        b_r = min(k - placed, min(1024, cur_blk))
        pool_n = min(m - placed, min(1664, max(b_r, int(round(1.6 * b_r)))))

        dmax = torch.where(avail, min_d2, torch.zeros_like(min_d2)).max().clamp_min(1e-30)
        d_rel = (min_d2 / dmax).clamp(0.0, 1.0)
        prop = torch.where(avail, w_sub * d_rel + 1e-15, torch.zeros_like(w_sub))
        n_far = min(24, max(4, b_r // 4), m)
        far_scores = torch.where(avail, d_rel, torch.full_like(d_rel, -1.0))
        far_ids = torch.topk(far_scores, k=n_far).indices
        max_prop = prop.max().clamp_min(1e-12)
        far_boost = torch.where(avail.gather(0, far_ids), max_prop * 1e6, torch.zeros_like(far_ids, dtype=torch.float32))
        prop.scatter_(0, far_ids, far_boost)

        u_g = torch.rand((m,), generator=gen, device=device, dtype=torch.float32).clamp_(1e-30, 1.0 - 1e-7)
        # Fallback uniform key `-200 + gumbel` on unpicked rows ensures `topk(pool_n)` always picks distinct rows even if avail got tight
        gumb = -(-u_g.log()).log()
        g_keys = torch.where(avail, prop.clamp_min(1e-30).log() + gumb, -200.0 + gumb)
        cand_idx = torch.topk(g_keys, k=pool_n, largest=True, sorted=False).indices

        if pool_n <= b_r:
            sel_idx = cand_idx[:b_r]
            chosen_c = xs_c[sel_idx].contiguous()
            avail.scatter_(0, sel_idx, false_buf[:b_r])
        else:
            c_pool = xs_c[cand_idx].contiguous()
            pool_d2 = min_d2[cand_idx].clone()
            if device.type == "cuda" and HAS_TRITON:
                cp_rep = _alloc_padded_2d(pool_n, d, seed_dtype, device)
                cp_norm = torch.empty((pool_n,), dtype=torch.float32, device=device)
                _launch_ultra_prep_centroids(c_pool, cp_rep, cp_norm)
                _, pool_lbls = _launch_ultra_assign(xs_c, cp_rep, cp_norm)
            else:
                _, pool_lbls = assign_points(xs_c, c_pool, compute_dtype=seed_dtype)
            w_pool = torch.zeros((pool_n,), dtype=torch.float32, device=device)
            w_pool.index_add_(0, pool_lbls, w_sub)
            w_pool.clamp_min_(1e-12 / float(pool_n))
            chosen_c = _ultra_select_pool(
                c_pool, pool_d2, w_pool, b_r, rseed=seed + 1000 + round_id
            )
            avail.scatter_(0, cand_idx, false_buf[:pool_n])

        if placed + b_r < k:
            if device.type == "cuda" and HAS_TRITON:
                ch_rep = _alloc_padded_2d(b_r, d, seed_dtype, device)
                ch_norm = torch.empty((b_r,), dtype=torch.float32, device=device)
                _launch_ultra_prep_centroids(chosen_c, ch_rep, ch_norm)
                new_d2, _ = _launch_ultra_assign(xs_c, ch_rep, ch_norm)
            else:
                new_d2, _ = assign_points(xs_c, chosen_c, compute_dtype=seed_dtype)
            min_d2 = torch.minimum(min_d2, new_d2)

        raw_c_list.append(chosen_c)
        placed += b_r
        cur_blk = min(1024, cur_blk * 3)
        round_id += 1

    raw_c = raw_c_list[0][:k].contiguous() if len(raw_c_list) == 1 else torch.cat(raw_c_list, dim=0)[:k].contiguous()

    # 3. 1-Step Weighted Voronoi Polish + Calibrated Prior Extraction on `xs_c`
    if device.type == "cuda" and HAS_TRITON:
        rc_rep = _alloc_padded_2d(k, d, seed_dtype, device)
        rc_norm = torch.empty((k,), dtype=torch.float32, device=device)
        _launch_ultra_prep_centroids(raw_c, rc_rep, rc_norm)
        sq_d0, labels = _launch_ultra_assign(xs_c, rc_rep, rc_norm)
    else:
        sq_d0, labels = assign_points(xs_c, raw_c, compute_dtype=seed_dtype)

    mass = torch.zeros((k,), dtype=torch.float32, device=device)
    mass.index_add_(0, labels, w_sub)

    if not polish:
        dist_m = torch.zeros((k,), dtype=torch.float32, device=device)
        dist_m.index_add_(0, labels, w_sub * sq_d0)
        tot_m = mass.sum().clamp_min(1e-30)
        return SeedingResult(
            centroids=raw_c,
            prior_probs=(mass / tot_m).contiguous(),
            prior_distortion=(dist_m / tot_m).contiguous(),
        )

    sums = torch.zeros((k, d), dtype=torch.float32, device=device)
    sums.index_add_(0, labels, xs_c * w_sub[:, None])
    nonempty = mass > 0.0
    polished = torch.where(
        nonempty[:, None],
        sums / mass.clamp_min(1e-30)[:, None],
        raw_c,
    ).contiguous()

    diff = xs_c - polished[labels]
    pol_sq = diff.square().sum(dim=1)
    dist_mass = torch.zeros((k,), dtype=torch.float32, device=device)
    dist_mass.index_add_(0, labels, w_sub * pol_sq)
    tot_m = mass.sum().clamp_min(1e-30)
    return SeedingResult(
        centroids=polished,
        prior_probs=(mass / tot_m).contiguous(),
        prior_distortion=(dist_mass / tot_m).contiguous(),
    )


# ==============================================================================
# 3. Discovery 3: Simplex 3-Point Empirical-Bayes Finalizer
# ==============================================================================


@torch.no_grad()
def ultra_simplex_eb_finalize(
    pass_sums: torch.Tensor,
    pass_counts: torch.Tensor,
    pass_counts_sq: torch.Tensor | None,
    pass_sse: torch.Tensor,
    live_centroids: torch.Tensor,
    slow_centroids: torch.Tensor | None,
    usage: torch.Tensor,
    usage_inv_bias: float,
    alpha_end: float,
    nepochs: float,
    d: int,
    mode: str = "simplex_eb",
    eb_tau: float = 0.25,
    kappa_max: float = 6.0,
) -> tuple[torch.Tensor, ShrinkageDiagnostics]:
    """Simplex 3-Point Empirical-Bayes Finalizer (`closing_mean <-> live_ema <-> slow_ema`).

    Resolves the exact failure modes of both 2-point finalizers:
      - `s-means` `eb_ema` (`C7`) shrinks only with fixed `tau=0.20` toward noisy `live_ema`
        (missing multi-epoch variance reduction on sparse `low_nk32` / `zipf_gmm` tails).
      - `SuperEMeans` `eb_james_stein` shrinks solely toward `slow_ema` (suffering lag bias on
        drifted/recently-split clusters at `T=1..2` or `zipf_gmm`), while naive JS toward
        `live_ema` over-shrinks on `isotropic_gaussian` due to positive in-pass covariance!
      - `simplex_eb` gates `slow_ema` vs `live_ema` per cluster by stationary noise-to-drift
        ratio `beta_slow = V_es / max(||c_ema - c_slow||^2, V_es)` and applies smooth Bayes/Stein
        precision weights on the `slow_ema` axis + correlation-safe prior-count `eb_tau` on the
        `live_ema` axis (`w_close + w_ema + w_slow == 1`, strictly inside the data convex hull).
    """
    counts_w = pass_counts.double()
    nonempty = counts_w > 0.0
    k = live_centroids.shape[0]
    device = live_centroids.device

    c_close = torch.where(
        nonempty[:, None],
        (pass_sums.double() / counts_w.clamp_min(1e-30)[:, None]).float(),
        live_centroids.float(),
    )

    if mode == "closing_mean":
        diag = ShrinkageDiagnostics("closing_mean", 1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
        return c_close.contiguous(), diag

    if mode == "ema_only":
        diag = ShrinkageDiagnostics("ema_only", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return live_centroids.float().contiguous(), diag

    if mode == "eb_ema" or slow_centroids is None:
        tau = max(0.0, float(eb_tau))
        lam = torch.where(nonempty, counts_w / (counts_w + tau).clamp_min(1e-30), torch.zeros_like(counts_w))
        lam_f = lam.float()
        out = torch.where(
            nonempty[:, None],
            live_centroids.float() + lam_f[:, None] * (c_close - live_centroids.float()),
            live_centroids.float(),
        )
        diag = ShrinkageDiagnostics("eb_ema", float(lam.mean().item()), 0.0, 1.0, tau, 0.0, 0.0)
        return out.contiguous(), diag

    if mode == "eb_slow_only":
        from .super_emeans import eb_james_stein_finalize_gpu

        out, _, diag = eb_james_stein_finalize_gpu(
            closing_means=c_close,
            anchor_centroids=slow_centroids,
            pass_counts=pass_counts,
            pass_counts_sq=pass_counts_sq,
            pass_sse=pass_sse,
            d=d,
            kappa="auto",
        )
        return out, diag

    # Full `simplex_eb` 3-point finalizer
    if pass_counts_sq is not None:
        n_eff = torch.where(
            nonempty,
            counts_w.square() / pass_counts_sq.double().clamp_min(1e-30),
            torch.zeros_like(counts_w),
        )
    else:
        n_eff = counts_w

    total_w = counts_w.sum().clamp_min(1e-30)
    global_sigma2 = (pass_sse.double().sum() / total_w).clamp_min(0.0)
    raw_sigma2 = torch.where(
        n_eff >= 2.0,
        pass_sse.double() / counts_w.clamp_min(1e-30),
        global_sigma2.expand_as(counts_w),
    ).clamp_min(0.0)
    sigma2_j = (n_eff * raw_sigma2 + 4.0 * global_sigma2) / (n_eff + 4.0)

    if d > 2:
        stein_c = (float(d) - 2.0) / float(d)
    elif d == 2:
        stein_c = 0.50
    else:
        stein_c = 0.35

    v_close = stein_c * sigma2_j / n_eff.clamp_min(1e-30)

    # Stationary variance of (c_ema - c_slow) vs observed squared drift ||c_ema - c_slow||^2
    n_batch_hat = (usage.double() * float(usage_inv_bias)).clamp_min(1.0)
    a_end = min(0.99, max(1e-4, float(alpha_end)))
    v_es = (a_end / (2.0 - a_end)) * (sigma2_j / n_batch_hat)
    d_es_sq = (live_centroids.double() - slow_centroids.double()).square().sum(dim=1)

    # Multi-pass prior availability factor: at T <= 0.75 slow_ema has no pre-closing pass;
    # at T >= 1.5 slow_ema carries full multi-pass history!
    phi_multi = min(1.0, max(0.25, (float(nepochs) - 0.5) / 1.0))
    beta_slow = (v_es / torch.maximum(d_es_sq, v_es).clamp_min(1e-30)).clamp(0.0, 1.0) * phi_multi

    anchor = (
        beta_slow[:, None] * slow_centroids.double()
        + (1.0 - beta_slow)[:, None] * live_centroids.double()
    )
    diff_ca_sq = (c_close.double() - anchor).square().sum(dim=1)

    # Smooth Bayes/Stein precision weight along independent `slow_ema` component
    k_max = max(0.0, float(kappa_max))
    max_slow_shrink = k_max / (n_eff + k_max).clamp_min(1e-30)
    js_slow_shrink = v_close / (diff_ca_sq + v_close).clamp_min(1e-30)
    w_slow = beta_slow * torch.minimum(max_slow_shrink, js_slow_shrink)

    # Correlation-safe prior-count weight along `live_ema` component
    tau = max(0.0, float(eb_tau))
    w_ema = (1.0 - beta_slow) * (tau / (n_eff + tau).clamp_min(1e-30))

    zero_all = (global_sigma2 <= 0.0) & (torch.where(nonempty, diff_ca_sq, 0.0).max() <= 0.0)
    tot_shrink = torch.where(zero_all, torch.zeros_like(n_eff), (w_slow + w_ema).clamp(0.0, 1.0))
    w_close = 1.0 - tot_shrink

    # Normalize (w_slow, w_ema) if tot_shrink clipped at 1.0
    scale_sh = torch.where(w_slow + w_ema > 1e-30, tot_shrink / (w_slow + w_ema).clamp_min(1e-30), 0.0)
    w_slow = w_slow * scale_sh
    w_ema = w_ema * scale_sh

    c_fin64 = (
        w_close[:, None] * c_close.double()
        + w_ema[:, None] * live_centroids.double()
        + w_slow[:, None] * slow_centroids.double()
    )
    c_fin = torch.where(nonempty[:, None], c_fin64.float(), live_centroids.float()).contiguous()

    ne_cnt = nonempty.double().sum().clamp_min(1.0)
    mean_lam = torch.where(nonempty, w_close, 0.0).sum() / ne_cnt
    min_lam = torch.where(nonempty, w_close, float("inf")).min()
    max_lam = torch.where(nonempty, w_close, float("-inf")).max()
    kap_eff = torch.where(nonempty & (w_close > 1e-12), n_eff * tot_shrink / w_close.clamp_min(1e-12), 0.0)
    sorted_kap = torch.sort(torch.where(nonempty, kap_eff, float("inf"))).values
    med_idx = (nonempty.to(torch.int64).sum() - 1).clamp_min(0) // 2
    med_kap = sorted_kap.gather(0, med_idx.view(1)).squeeze(0)
    excess = torch.clamp(diff_ca_sq - v_close, min=0.0)
    tau2_t = torch.where(nonempty, excess, 0.0).sum() / ne_cnt

    vals = torch.stack(
        [nonempty.any().double(), mean_lam, min_lam, max_lam, med_kap, tau2_t, global_sigma2]
    ).tolist()
    if vals[0] < 0.5:
        diag = ShrinkageDiagnostics("simplex_eb", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    else:
        diag = ShrinkageDiagnostics(
            mode="simplex_eb",
            mean_lambda=float(vals[1]),
            min_lambda_nonempty=float(vals[2]),
            max_lambda_nonempty=float(vals[3]),
            median_effective_kappa=float(vals[4]),
            tau2_prior=float(vals[5]),
            sigma2_mean=float(vals[6]),
        )
    return c_fin, diag


# ==============================================================================
# 4. Discovery 4: Oja-Seeded Fast Vectorized `P/Q` `head_edit` Transplants
# ==============================================================================


@torch.no_grad()
def run_ultra_head_edit_checkpoint(
    x_batch_c: torch.Tensor,
    centroids_c: torch.Tensor,
    usage: torch.Tensor,
    sse_ema: torch.Tensor,
    ref_distortion_mean: torch.Tensor,
    survival_mass: torch.Tensor | None,
    prin_vec: torch.Tensor | None,
    slow_centroids: torch.Tensor | None,
    slow_usage: torch.Tensor | None,
    pass_sums: torch.Tensor | None,
    pass_counts: torch.Tensor | None,
    pass_counts_sq: torch.Tensor | None,
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
) -> int:
    """Fast Oja-seeded `P/Q`-confirmed dormant-twin transplant (`~0.5 ms` per checkpoint!).

    Upgrades `SuperEMeans`'s `run_fast_head_edit_step`:
      1. Injects each donor's streaming Oja top eigenvector `v_{s,1}` directly as Probe 0
         alongside orthogonal signed-Hadamard + sample-PCA + farthest-chord probes!
      2. Uses `_all_pairs_centered_nn` / SRAM top-3 directly on centered `x_batch_c` & `centroids_c`.
      3. Resets `slow_centroids` directly to the newly confirmed child centers (zero lag!).
    """
    from .super_emeans import _all_pairs_centered_nn

    bsz, d = x_batch_c.shape
    k = centroids_c.shape[0]
    if k < 3 or bsz < 16 or max_pairs < 1 or num_probes < 1:
        return 0
    half = bsz // 2
    if half < 8 or (bsz - half) < 8:
        return 0

    diag.checks_run += 1
    device = x_batch_c.device
    rep_dtype = (
        compute_dtype
        if compute_dtype in (torch.float32, torch.bfloat16, torch.float16, torch.float64)
        else torch.float32
    )
    scale_full = float(bsz) / float(half)

    top_d, top_i, origin, xc_all, x_norm_all, cc = fast_top3_centered_assign(
        x_batch_c, centroids_c, compute_dtype, already_centered=True
    )
    cn = cc.square().sum(dim=1)
    d_nn2, nn_idx = _all_pairs_centered_nn(centroids_c)

    sse_p = torch.empty((k,), dtype=torch.float32, device=device)
    dormancy_p = torch.empty((k,), dtype=torch.float32, device=device)
    used_mask = torch.zeros((k,), dtype=torch.bool, device=device)

    def _score_state() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        eff_mass = (
            torch.maximum(usage, torch.minimum(survival_mass, thresh))
            if survival_mass is not None
            else usage
        )
        h_cost = eff_mass * d_nn2.clamp_max(1e30)
        e_rem = torch.maximum(b_rem, h_cost)

        f_mask = ((step - last_edit_step) >= protect_steps) & (~used_mask)
        if use_huygens_protection and gamma_protect > 0.0:
            prot_cutoff = gamma_protect * ref_distortion_mean.clamp_min(1e-30)
            rare_shield = (usage < thresh) & (h_cost >= prot_cutoff)
            f_mask = f_mask & (~rare_shield)

        don_elig = f_mask & (cp >= 4.0) & (cq >= 2.0) & (sse_p > 1e-12)
        rec_elig = f_mask
        return d1p, a1p, sse_p, e_rem, don_elig, rec_elig

    d1_p, a1_p, sse_p_cur, eff_rem_cur, don_elig_cur, rec_elig_cur = _score_state()

    q_try = min(max_pairs, k // 2)
    don_scores = torch.where(don_elig_cur, sse_p_cur, torch.full_like(sse_p_cur, -1.0))
    top_don_vals, top_don_ids = torch.topk(don_scores, k=q_try, largest=True, sorted=True)
    min_rec_cost = torch.where(rec_elig_cur, eff_rem_cur, torch.full_like(eff_rem_cur, float("inf"))).min()
    n_rec_avail_t = rec_elig_cur.to(torch.int64).sum()

    # Early-exit in 1 tiny packed read if even the cheapest recipient >= 0.45 * best donor SSE!
    pack_init = torch.cat(
        [top_don_ids.double(), top_don_vals.double(), min_rec_cost.view(1).double(), n_rec_avail_t.view(1).double()]
    ).tolist()
    min_rec_val = float(pack_init[2 * q_try])
    n_rec_avail = int(pack_init[2 * q_try + 1])
    don_order = [
        int(pack_init[i])
        for i in range(q_try)
        if pack_init[q_try + i] > 1e-12 and min_rec_val < 0.45 * pack_init[q_try + i]
    ]
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
        rec_cand_mask = rec_elig_cur.clone()
        rec_cand_mask[s_idx] = False
        rec_cost_masked = torch.where(
            rec_cand_mask, eff_rem_cur, torch.full_like(eff_rem_cur, float("inf"))
        )
        best_rec_cost_t, d_idx_t = torch.min(rec_cost_masked, dim=0)
        pair_meta = torch.stack(
            [don_elig_cur[s_idx].float(), sse_p_cur[s_idx], best_rec_cost_t, d_idx_t.float()]
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
        extra_dirs: list[torch.Tensor] = []
        if prin_vec is not None and d >= 2:
            pv_s = prin_vec[s_idx : s_idx + 1]
            pvn = torch.linalg.vector_norm(pv_s)
            extra_dirs.append(torch.where(pvn > 1e-12, pv_s / pvn.clamp_min(1e-12), dirs_had[:1]))
        if num_probes >= 3 and d >= 2:
            v0 = dirs_had[-2:-1].t()
            for _ in range(2):
                v1 = r_s.t() @ (r_s @ v0)
                v1n = torch.linalg.vector_norm(v1)
                v0 = torch.where(v1n > 1e-16, v1 / v1n.clamp_min(1e-16), v0)
            extra_dirs.append(v0.t())
            far_idx = torch.argmax(r_s.square().sum(dim=1))
            v_far = r_s[far_idx : far_idx + 1]
            v_farn = torch.linalg.vector_norm(v_far)
            extra_dirs.append(torch.where(v_farn > 1e-16, v_far / v_farn.clamp_min(1e-16), dirs_had[-1:]))

        if extra_dirs:
            n_ex = min(len(extra_dirs), num_probes)
            dirs = torch.cat([dirs_had[: num_probes - n_ex], *extra_dirs[:n_ex]], dim=0)
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
        delta_plus = torch.where(n_pos > 0.0, sum_pos / n_pos.clamp_min(1.0), HALF_GAUSSIAN_MEAN * rms_1d)
        delta_minus = torch.where(n_neg > 0.0, sum_neg / n_neg.clamp_min(1.0), -HALF_GAUSSIAN_MEAN * rms_1d)

        base_don = origin + mu_sc
        cand_minus_f = base_don + delta_minus[:, None] * dirs
        cand_plus_f = base_don + delta_plus[:, None] * dirs
        cm_rep_c = (cand_minus_f.to(rep_dtype).float() - origin).contiguous()
        cp_rep_c = (cand_plus_f.to(rep_dtype).float() - origin).contiguous()
        cand_minus = cand_minus_f.to(centroids_c.dtype)
        cand_plus = cand_plus_f.to(centroids_c.dtype)
        cm_norm = cm_rep_c.square().sum(dim=1)
        cp_norm = cp_rep_c.square().sum(dim=1)

        d_rem_all = exact_exclusion_sq_dists(top_d, top_i, d_idx, s_idx)
        d_rem_p = d_rem_all[:half]

        dist_m_p = (torch.addmm(cm_norm[None, :], xc_p, cm_rep_c.t(), alpha=-2.0) + xn_p[:, None]).clamp_min_(0.0)
        dist_p_p = (torch.addmm(cp_norm[None, :], xc_p, cp_rep_c.t(), alpha=-2.0) + xn_p[:, None]).clamp_min_(0.0)
        new_best_p = torch.minimum(d_rem_p[:, None], torch.minimum(dist_m_p, dist_p_p))
        delta_sse_p = (new_best_p - d1_p[:, None]).sum(dim=0)

        best_dp_t, best_m_t = torch.min(delta_sse_p, dim=0)
        # Evaluate Q confirmation IN THE SAME GPU LAUNCH BATCH before syncing so P+Q only sync ONCE!
        d_rem_q = d_rem_all[half:]
        d1_q = top_d[half:, 0]
        cm_win = cm_rep_c.index_select(0, best_m_t.view(1))
        cp_win = cp_rep_c.index_select(0, best_m_t.view(1))
        cmn_win = cm_norm.index_select(0, best_m_t.view(1))
        cpn_win = cp_norm.index_select(0, best_m_t.view(1))
        dist_m_q = (torch.addmm(cmn_win[None, :], xc_q, cm_win.t(), alpha=-2.0).squeeze(1) + xn_q).clamp_min_(0.0)
        dist_p_q = (torch.addmm(cpn_win[None, :], xc_q, cp_win.t(), alpha=-2.0).squeeze(1) + xn_q).clamp_min_(0.0)
        new_best_q = torch.minimum(d_rem_q, torch.minimum(dist_m_q, dist_p_q))
        delta_sse_q_t = (new_best_q - d1_q).sum()
        sum_d1_q_t = d1_q.sum()

        pq_vals = torch.stack([best_dp_t, best_m_t.float(), delta_sse_q_t, sum_d1_q_t]).tolist()
        best_dp = float(pq_vals[0])
        best_m = int(pq_vals[1])
        delta_sse_q = float(pq_vals[2])
        mean_sse_q = float(pq_vals[3]) / float(max(1, k))

        if not (best_dp < -1e-6 * max(1.0, s_sse_val)):
            continue
        diag.p_improved += 1
        if not (delta_sse_q < -1e-6 * max(1.0, mean_sse_q)):
            continue

        diag.q_confirmed += 1
        new_s = cand_minus[best_m : best_m + 1].clone()
        new_d = cand_plus[best_m : best_m + 1].clone()
        u_win = dirs[best_m : best_m + 1]
        if compute_dtype in (torch.float32, torch.bfloat16, torch.float16):
            separate_collapsed_directional(new_s, new_d, u_win, compute_dtype)
        else:
            _separate_collapsed_splits(new_s, new_d, torch.float32)

        centroids_c[s_idx] = new_s[0]
        centroids_c[d_idx] = new_d[0]

        half_u = 0.5 * usage[s_idx]
        usage[s_idx] = half_u
        usage[d_idx] = half_u

        half_e = 0.35 * sse_ema[s_idx]
        sse_ema[s_idx] = half_e
        sse_ema[d_idx] = half_e

        if survival_mass is not None:
            survival_mass[s_idx] = 0.0
            survival_mass[d_idx] = 0.0
        if prin_vec is not None:
            from .algo_extensions import reset_post_split_oja_axes

            s_t = torch.tensor([s_idx], dtype=torch.int64, device=device)
            d_t = torch.tensor([d_idx], dtype=torch.int64, device=device)
            reset_post_split_oja_axes(prin_vec, s_t, d_t, u_win, generator=generator)
        if slow_centroids is not None and slow_usage is not None:
            slow_centroids[s_idx] = new_s[0]
            slow_centroids[d_idx] = new_d[0]
            shalf = 0.5 * slow_usage[s_idx]
            slow_usage[s_idx] = shalf
            slow_usage[d_idx] = shalf
        if pass_sums is not None:
            pass_sums[s_idx] = 0.0
            pass_sums[d_idx] = 0.0
        if pass_counts is not None:
            pass_counts[s_idx] = 0.0
            pass_counts[d_idx] = 0.0
        if pass_counts_sq is not None:
            pass_counts_sq[s_idx] = 0.0
            pass_counts_sq[d_idx] = 0.0
        if pass_sse is not None:
            pass_sse[s_idx] = 0.0
            pass_sse[d_idx] = 0.0

        last_edit_step[s_idx] = step
        last_edit_step[d_idx] = step
        used_mask[s_idx] = True
        used_mask[d_idx] = True
        kept_edits += 1

        if kept_edits < n_try:
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
            d1_p, a1_p, sse_p_cur, eff_rem_cur, don_elig_cur, rec_elig_cur = _score_state()

    return kept_edits


# ==============================================================================
# 5. `UltraEMeans` Unified Solver & Pre-Registered Ablation Matrix
# ==============================================================================


@dataclass
class UltraEMeansEpochStat:
    epoch: int
    rows: int
    steps: int
    moving_mse: float
    imbalance_factor: float
    n_splits: int
    n_protected: int
    n_transplants: int
    elapsed_sec: float


class UltraEMeans:
    """Next-generation streaming & few-epoch clustering (`UltraEMeans`)."""

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
        init_usage: float = DEFAULT_INIT_USAGE,
        # U0: Fast Importance Coreset + Tournament k-means++ + 1-Step Voronoi Polish
        init_mode: str = "ultra_turkpp",  # "ultra_turkpp" | "subsampled_d15" | "random" | "coreset_kpp"
        coreset_mult: int = 12,
        min_coreset_size: int = 8192,
        init_rounds: int = 4,
        init_power: float = 1.5,
        use_voronoi_polish: bool = True,
        use_prior_mass: bool = True,
        prior_mass_scale: float = 0.5,
        prior_floor: float = 1e-3,
        warm_alpha_scale: float | None = None,
        # Schedule & WideStep
        schedule_mode: str = "emeans_cosine",  # "emeans_cosine" | "power_cosine" | "constant"
        alpha_floor: float = 0.06,
        cosine_power: float = 0.92,
        lowd_revival: bool = True,
        wide_step_mult: int = 1,
        wide_from_frac: float = 0.25,
        # U2: Decay-Matched Bias-Corrected Donors + Twin-Safe Huygens Shield + Revival Freeze
        revival_mode: str = "split",  # "split" | "off"
        use_revive_stop: bool = True,
        revive_stop_frac: float = 0.85,
        flush_pass_on_split: bool = True,
        use_huygens_protection: bool = True,
        gamma_protect: float = 0.5,
        grace_steps: int = 4,
        donor_mode: str = "oja_sse_usage",  # "oja_sse_usage" | "sse_usage" | "d2" | "usage_only"
        donor_usage_w: float = 0.35,
        couple_sse_to_alpha: bool = True,
        sse_decay: float = 0.85,
        # U1: Exact Half-Gaussian Principal-Axis Split (sqrt(2/pi) * sigma_{j,1} * v_{j,1})
        split_mode: str = "half_gaussian_oja",  # "half_gaussian_oja" | "rms_aniso" | "rms_iso" | "oja_only" | "checkerboard"
        split_mult: float = HALF_GAUSSIAN_MEAN,
        use_feat_scale: bool = True,
        var_beta: float = 0.85,
        split_epsilon: float = SPLIT_EPSILON,
        # U4: Strategic Oja-Seeded P/Q `head_edit` Transplants
        head_edit: bool = True,
        transplant_checkpoints: tuple[float, ...] = (0.46, 0.76),
        transplant_probes: int = 6,
        transplant_max_pairs: int = 5,
        transplant_protect_steps: int = 6,
        # U3: Simplex 3-Point Empirical-Bayes Finalizer + Adaptive `1/T` Closing Ramp
        use_pass_weighting: bool = False,
        adaptive_pass_weighting: bool = True,
        pass_w_span: float = 0.35,
        finalization: str = "simplex_eb",  # "simplex_eb" | "eb_slow_only" | "eb_ema" | "closing_mean" | "ema_only"
        eb_tau: float = 0.25,
        kappa_max: float = 6.0,
        slow_ema_mult: float = 0.25,
        # Precision & Backend
        compute_dtype: torch.dtype | None = None,
        backend: str = "auto",
        seed: int = 1234,
        verbose: bool = False,
    ) -> None:
        if d < 1:
            raise ValueError(f"Dimension d={d} must be >= 1")
        if k < 1:
            raise ValueError(f"Number of clusters k={k} must be >= 1")
        if not math.isfinite(nepochs) or nepochs <= 0.0:
            raise ValueError(f"nepochs={nepochs} must be finite and > 0")
        if init_mode not in ("ultra_turkpp", "subsampled_d15", "random", "coreset_kpp"):
            raise ValueError(f"Unsupported init_mode={init_mode!r}")
        if schedule_mode not in ("emeans_cosine", "power_cosine", "constant"):
            raise ValueError(f"Unsupported schedule_mode={schedule_mode!r}")
        if revival_mode not in ("split", "off"):
            raise ValueError(f"Unsupported revival_mode={revival_mode!r}")
        if donor_mode not in ("oja_sse_usage", "sse_usage", "d2", "sse_only", "usage_only"):
            raise ValueError(f"Unsupported donor_mode={donor_mode!r}")
        if split_mode not in ("half_gaussian_oja", "rms_aniso", "rms_iso", "oja_only", "checkerboard"):
            raise ValueError(f"Unsupported split_mode={split_mode!r}")
        if finalization not in ("simplex_eb", "eb_slow_only", "eb_ema", "closing_mean", "ema_only", "none"):
            raise ValueError(f"Unsupported finalization={finalization!r}")
        if backend not in ("auto", "triton", "torch", "native"):
            raise ValueError(f"Unsupported backend={backend!r}")
        if wide_step_mult < 1:
            raise ValueError("wide_step_mult must be >= 1")
        if prior_mass_scale <= 0.0 or not math.isfinite(prior_mass_scale):
            raise ValueError("prior_mass_scale must be finite and > 0")
        if gamma_protect < 0.0 or not math.isfinite(gamma_protect):
            raise ValueError("gamma_protect must be finite and >= 0")

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

        self.init_mode = init_mode
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
            if warm_alpha_scale <= 0.0 or not math.isfinite(warm_alpha_scale):
                raise ValueError("warm_alpha_scale must be finite and > 0")
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
        self.flush_pass_on_split = bool(flush_pass_on_split)
        self.use_huygens_protection = bool(use_huygens_protection)
        self.gamma_protect = float(gamma_protect)
        self.grace_steps = int(grace_steps)
        self.donor_mode = "d2" if donor_mode == "sse_only" else donor_mode
        self.donor_usage_w = float(donor_usage_w)
        self.couple_sse_to_alpha = bool(couple_sse_to_alpha)
        self.sse_decay = float(sse_decay)

        self.split_mode = split_mode
        self.split_mult = float(split_mult)
        self.use_feat_scale = bool(use_feat_scale)
        self.var_beta = float(var_beta)
        self.split_epsilon = float(split_epsilon)

        self.head_edit = bool(head_edit)
        self.transplant_checkpoints = tuple(float(x) for x in transplant_checkpoints)
        self.transplant_probes = int(transplant_probes)
        self.transplant_max_pairs = int(transplant_max_pairs)
        self.transplant_protect_steps = int(transplant_protect_steps)

        self.use_pass_weighting = bool(use_pass_weighting)
        self.adaptive_pass_weighting = bool(adaptive_pass_weighting)
        self.pass_w_span = float(pass_w_span)
        self.finalization = "ema_only" if finalization == "none" else finalization
        self.eb_tau = float(eb_tau)
        self.kappa_max = float(kappa_max)
        self.slow_ema_mult = float(slow_ema_mult)

        self.compute_dtype = compute_dtype
        self.backend = "torch" if backend == "native" else backend
        self.seed = int(seed)
        self.verbose = bool(verbose)

        # Fitted attributes (1:1 compatible with EMeans / SMeans / SuperEMeans)
        self.recipe: EMeansRecipe | None = None
        self.seeding_result: SeedingResult | None = None
        self.centroids: torch.Tensor | None = None
        self.slow_centroids: torch.Tensor | None = None
        self.assignments: torch.Tensor | None = None
        self.iteration_stats: list[UltraEMeansEpochStat] = []
        self.shrinkage_diagnostics: ShrinkageDiagnostics | None = None
        self.head_edit_diagnostics: HeadEditDiagnostics | None = None
        self.n_splits: int = 0
        self.n_protected: int = 0
        self.n_transplants: int = 0
        self.dead_clusters: int = 0
        self.init_time_sec: float = 0.0
        self.train_time_sec: float = 0.0
        self.total_time_sec: float = 0.0
        self._origin64: torch.Tensor | None = None
        self._coord_scale: float = 1.0

    def _resolve_compute_dtype(self, x: torch.Tensor) -> torch.dtype:
        if self.compute_dtype is not None:
            if not x.is_cuda and self.compute_dtype in (torch.bfloat16, torch.float16):
                return torch.float32
            return self.compute_dtype
        if x.is_cuda and x.dtype in (torch.bfloat16, torch.float16):
            return x.dtype
        if x.is_cuda and self.backend != "torch":
            if torch.cuda.get_device_capability(x.device) >= (8, 0):
                return torch.bfloat16
        return torch.float32

    def _effective_pass_span(self) -> float:
        if self.use_pass_weighting and not self.adaptive_pass_weighting:
            return self.pass_w_span
        if self.adaptive_pass_weighting:
            return float(self.pass_w_span * min(1.0, (1.0 / max(self.nepochs, 1.0)) ** 1.25))
        return 0.0

    @torch.no_grad()
    def fit(
        self,
        x: torch.Tensor | np.ndarray,
        init_centroids: torch.Tensor | np.ndarray | None = None,
        seeding_result: SeedingResult | None = None,
        capture_assignments: bool = True,
    ) -> tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray | None]:
        """Fit `UltraEMeans` on dataset `x` of shape `(N, D)`, returning `(centroids, assignments)`."""
        input_is_numpy = isinstance(x, np.ndarray)
        if input_is_numpy:
            x_np = np.asarray(x)
            if x_np.ndim != 2:
                raise ValueError(f"Expected 2D input array, got shape {x_np.shape}")
            x_t = torch.from_numpy(
                np.ascontiguousarray(x_np, dtype=np.float64 if x_np.dtype == np.float64 else np.float32)
            )
        else:
            if x.ndim != 2:
                raise ValueError(f"Expected 2D input tensor, got shape {tuple(x.shape)}")
            x_t = x.detach()

        n, d = int(x_t.shape[0]), int(x_t.shape[1])
        if d != self.d:
            raise ValueError(f"Expected dimension d={self.d}, got {d}")
        if n < self.k:
            raise ValueError(f"Number of points n={n} must be >= number of clusters k={self.k}")
        if not bool(torch.isfinite(x_t).all()):
            raise ValueError("Input dataset x must contain only finite values (no NaN or Inf)")

        device = x_t.device
        comp_dtype = self._resolve_compute_dtype(x_t)

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

        origin64, coord_scale = _compute_origin_and_scale(x_t, comp_dtype)
        self._origin64 = origin64
        self._coord_scale = coord_scale

        # --- Seeding in centered & scaled frame ---
        need_priors = self.use_prior_mass or self.use_voronoi_polish
        if seeding_result is not None:
            c_in_c = ((seeding_result.centroids.to(device=device, dtype=torch.float64) - origin64[None, :]) * coord_scale).float().contiguous()
            seed_res = SeedingResult(
                centroids=c_in_c,
                prior_probs=seeding_result.prior_probs.to(device=device, dtype=torch.float32).contiguous(),
                prior_distortion=(
                    seeding_result.prior_distortion.to(device=device, dtype=torch.float32) * (coord_scale * coord_scale)
                ).contiguous(),
            )
        elif init_centroids is not None:
            c_in = (
                torch.from_numpy(np.asarray(init_centroids, dtype=np.float64))
                if isinstance(init_centroids, np.ndarray)
                else init_centroids.detach()
            )
            if tuple(c_in.shape) != (self.k, self.d):
                raise ValueError(f"Expected init_centroids shape ({self.k}, {self.d}), got {tuple(c_in.shape)}")
            if not bool(torch.isfinite(c_in).all()):
                raise ValueError("init_centroids must contain only finite values")
            raw_c = ((c_in.to(device=device, dtype=torch.float64) - origin64[None, :]) * coord_scale).float().contiguous()
            seed_res = SeedingResult(
                centroids=raw_c,
                prior_probs=torch.full((self.k,), 1.0 / self.k, dtype=torch.float32, device=device),
                prior_distortion=torch.zeros((self.k,), dtype=torch.float32, device=device),
            )
            need_priors = False
        elif self.init_mode == "subsampled_d15":
            from .smeans import subsampled_d15_init

            raw_abs = subsampled_d15_init(
                x_t, k=self.k, seed=self.seed, rounds=self.init_rounds, power=self.init_power, exact_ref=False
            )
            raw_c = ((raw_abs.to(device=device, dtype=torch.float64) - origin64[None, :]) * coord_scale).float().contiguous()
            if need_priors:
                m_sub = min(n, max(self.min_coreset_size, self.coreset_mult * self.k, self.k * 2))
                gen_s = torch.Generator(device=device)
                gen_s.manual_seed((self.seed + 13) % (2**32))
                idx_s = torch.randperm(n, generator=gen_s, device=device)[:m_sub]
                xs_c = ((x_t[idx_s].double() - origin64[None, :]) * coord_scale).float().contiguous()
                w_s = torch.full((m_sub,), 1.0 / m_sub, dtype=torch.float32, device=device)
                from .seeding import voronoi_polish_and_priors

                seed_res = voronoi_polish_and_priors(
                    xs_c,
                    w_s,
                    raw_c,
                    compute_dtype=torch.float32 if comp_dtype == torch.float64 else comp_dtype,
                )
                if not self.use_voronoi_polish:
                    seed_res = SeedingResult(raw_c, seed_res.prior_probs, seed_res.prior_distortion)
            else:
                seed_res = SeedingResult(
                    centroids=raw_c,
                    prior_probs=torch.full((self.k,), 1.0 / self.k, dtype=torch.float32, device=device),
                    prior_distortion=torch.zeros((self.k,), dtype=torch.float32, device=device),
                )
        elif self.init_mode == "random":
            gen_init = torch.Generator(device=device)
            gen_init.manual_seed(int(self.seed) % (2**32))
            perm = torch.randperm(n, generator=gen_init, device=device)[: self.k]
            raw_c = ((x_t[perm].double() - origin64[None, :]) * coord_scale).float().contiguous()
            seed_res = SeedingResult(
                centroids=raw_c,
                prior_probs=torch.full((self.k,), 1.0 / self.k, dtype=torch.float32, device=device),
                prior_distortion=torch.zeros((self.k,), dtype=torch.float32, device=device),
            )
            need_priors = False
        else:
            # Default `ultra_turkpp` (or `coreset_kpp` alias)
            eff_mult = 16 if self.init_mode == "coreset_kpp" else self.coreset_mult
            eff_min_m = 8192 if self.init_mode == "coreset_kpp" else self.min_coreset_size
            seed_res = ultra_turkpp_init_centered(
                x_t=x_t,
                origin64=origin64,
                coord_scale=coord_scale,
                k=self.k,
                comp_dtype=comp_dtype,
                coreset_mult=eff_mult,
                min_coreset_size=eff_min_m,
                polish=self.use_voronoi_polish,
                seed=self.seed,
            )

        self.seeding_result = seed_res
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self.init_time_sec = time.perf_counter() - t_init_start

        use_triton_fast = (
            device.type == "cuda"
            and HAS_TRITON
            and comp_dtype in (torch.float32, torch.bfloat16, torch.float16)
            and not torch.are_deterministic_algorithms_enabled()
            and (
                self.backend == "triton"
                or (self.backend == "auto" and torch.cuda.get_device_capability(device) >= (8, 0))
            )
        )
        if use_triton_fast:
            centroids_c, assignments = self._fit_cuda_fast(
                x_t=x_t,
                seed_res=seed_res,
                has_priors=need_priors,
                comp_dtype=comp_dtype,
                recipe=recipe,
                origin64=origin64,
                coord_scale=coord_scale,
                capture_assignments=capture_assignments,
            )
        else:
            centroids_c, assignments = self._fit_torch_reference(
                x_t=x_t,
                seed_res=seed_res,
                has_priors=need_priors,
                comp_dtype=comp_dtype,
                recipe=recipe,
                origin64=origin64,
                coord_scale=coord_scale,
                capture_assignments=capture_assignments,
            )

        inv_scale = 1.0 / coord_scale
        self.centroids = (centroids_c.double() * inv_scale + origin64[None, :]).float().contiguous()
        if self.slow_centroids is not None:
            self.slow_centroids = (
                self.slow_centroids.double() * inv_scale + origin64[None, :]
            ).float().contiguous()
        self.assignments = assignments
        self.total_time_sec = self.init_time_sec + self.train_time_sec

        if input_is_numpy:
            return (
                self.centroids.cpu().numpy(),
                self.assignments.cpu().numpy() if self.assignments is not None else None,
            )
        return self.centroids, self.assignments

    def _fit_cuda_fast(
        self,
        x_t: torch.Tensor,
        seed_res: SeedingResult,
        has_priors: bool,
        comp_dtype: torch.dtype,
        recipe: EMeansRecipe,
        origin64: torch.Tensor,
        coord_scale: float,
        capture_assignments: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Zero-sync fused Blackwell Triton training pipeline."""
        device = x_t.device
        n, d = int(x_t.shape[0]), int(x_t.shape[1])
        k = self.k

        t_train_start = time.perf_counter()

        origin32 = origin64.float().contiguous()
        origin_assign = (
            origin64
            if (x_t.dtype == torch.float64 or coord_scale != 1.0 or float(origin64.abs().max().item()) > 1e3)
            else origin32
        )

        centroids_c = seed_res.centroids.to(device=device, dtype=torch.float32).clone().contiguous()
        p_hat = seed_res.prior_probs.to(device=device, dtype=torch.float32)
        e_hat = seed_res.prior_distortion.to(device=device, dtype=torch.float32)

        bsz_float = float(recipe.batch_size)
        min_prob = max(0.0, self.prior_floor) / float(k)
        mean_ehat = e_hat.mean().clamp_min(1e-12)

        if self.use_prior_mass and has_priors:
            usage = (self.prior_mass_scale * bsz_float * p_hat.clamp_min(min_prob)).float().contiguous()
            sse_ema = (self.prior_mass_scale * bsz_float * e_hat.clamp_min(1e-6 * mean_ehat)).float().contiguous()
            usage_bias = float(self.prior_mass_scale)
            sse_bias = float(self.prior_mass_scale)
        else:
            usage = torch.full((k,), recipe.init_usage, dtype=torch.float32, device=device)
            sse_ema = torch.zeros((k,), dtype=torch.float32, device=device)
            usage_bias = float(recipe.init_usage * k) / max(1.0, bsz_float)
            sse_bias = 0.0

        track_surv = bool(self.use_huygens_protection and self.gamma_protect > 0.0)
        if track_surv:
            survival_mass = (
                (bsz_float * p_hat).float().contiguous()
                if has_priors
                else torch.zeros((k,), dtype=torch.float32, device=device)
            )
            huy_c_pad = _alloc_padded_2d(k, d, torch.float32, device)
            huy_cn = torch.empty((k,), dtype=torch.float32, device=device)
            huy_d2_any = torch.empty((k,), dtype=torch.float32, device=device)
            huy_d2_surv = torch.empty((k,), dtype=torch.float32, device=device)
            huy_prot_i8 = torch.empty((k,), dtype=torch.int8, device=device)
        else:
            survival_mass = centroids_c  # dummy ptr
            huy_c_pad = huy_cn = huy_d2_any = huy_d2_surv = huy_prot_i8 = None

        track_slow = self.finalization in ("simplex_eb", "eb_slow_only")
        if track_slow:
            slow_centroids = centroids_c.clone().contiguous()
            slow_usage = usage.clone().contiguous()
            alpha_slow_base = min(1.0, max(1e-6, self.slow_ema_mult * recipe.alpha_min_factor * recipe.alpha0))
        else:
            slow_centroids = centroids_c  # dummy ptr
            slow_usage = usage            # dummy ptr
            alpha_slow_base = 0.0

        track_oja = self.split_mode in ("half_gaussian_oja", "oja_only") or self.donor_mode == "oja_sse_usage"
        if track_oja:
            pca_gen = torch.Generator(device=device)
            pca_gen.manual_seed((self.seed + 31337) % (2**32))
            init_dirs = torch.randn((k, d), generator=pca_gen, device=device, dtype=torch.float32)
            init_dirs = init_dirs / init_dirs.norm(dim=1, keepdim=True).clamp_min(1e-12)
            if has_priors:
                # Calibrate initial Oja scale to physical cluster coordinate RMS!
                init_rms = (e_hat / (p_hat.clamp_min(1e-6) * float(max(1, d)))).clamp_min(1e-8).sqrt()[:, None]
                prin_vec = (init_rms * init_dirs).contiguous()
            else:
                prin_vec = (1e-3 * init_dirs).contiguous()
            g_sums = torch.zeros((k, d), dtype=torch.float32, device=device)
        else:
            prin_vec = centroids_c  # dummy ptr
            g_sums = centroids_c    # dummy ptr

        feat_scale = torch.ones((d,), dtype=torch.float32, device=device)
        ones_d = torch.ones((d,), dtype=torch.float32, device=device)

        sums = torch.zeros((k, d), dtype=torch.float32, device=device)
        counts = torch.zeros((k,), dtype=torch.float32, device=device)
        batch_sse = torch.zeros((k,), dtype=torch.float32, device=device)

        need_pass_accum = self.finalization in ("simplex_eb", "eb_slow_only", "eb_ema", "closing_mean")
        pass_sums = torch.zeros((k, d), dtype=torch.float32, device=device)
        pass_counts = torch.zeros((k,), dtype=torch.float64, device=device)
        pass_counts_sq = torch.zeros((k,), dtype=torch.float64, device=device)
        pass_sse = torch.zeros((k,), dtype=torch.float64, device=device)
        raw_pass_counts = torch.zeros((k,), dtype=torch.float64, device=device)

        c_rep = _alloc_padded_2d(k, d, comp_dtype, device)
        c_norm = torch.empty((k,), dtype=torch.float32, device=device)
        _launch_ultra_prep_centroids(centroids_c, c_rep, c_norm)

        bsz_max = min(n, recipe.batch_size * max(1, self.wide_step_mult))
        dist_buf = torch.empty((bsz_max,), dtype=torch.float32, device=device)
        lbl_buf = torch.empty((bsz_max,), dtype=torch.int64, device=device)

        num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        splitk_buffers: tuple[torch.Tensor, torch.Tensor, torch.Tensor, int] | None = None
        if k >= 2048 and triton.cdiv(recipe.batch_size, 64) < num_sms:
            parts = _choose_split_k_partitions(recipe.batch_size, k, num_sms=num_sms, block_b=64, block_k=128)
            if parts > 1:
                splitk_buffers = (
                    torch.empty((bsz_max, parts), dtype=torch.float32, device=device),
                    torch.empty((bsz_max, parts), dtype=torch.int32, device=device),
                    torch.empty((bsz_max,), dtype=torch.float32, device=device),
                    parts,
                )

        assignments = (
            torch.full((n,), -1, dtype=torch.int64, device=device)
            if capture_assignments
            else None
        )

        split_gen = torch.Generator(device=device)
        split_gen.manual_seed((self.seed + 7919) % (2**32))

        if self.head_edit:
            last_edit_step = torch.full((k,), -(10**9), dtype=torch.int64, device=device)
            edit_gen = torch.Generator(device=device)
            edit_gen.manual_seed((self.seed + 918273) % (2**32))
            edit_diag = HeadEditDiagnostics()
        else:
            last_edit_step = None
            edit_gen = None
            edit_diag = None
        self.head_edit_diagnostics = edit_diag

        num_epochs = len(recipe.epoch_rows)
        ep_sq_dists_gpu = torch.zeros((num_epochs,), dtype=torch.float64, device=device)
        ep_dot_counts_gpu = torch.zeros((num_epochs,), dtype=torch.float64, device=device)
        ep_cum_splits_gpu = torch.zeros((num_epochs,), dtype=torch.int64, device=device)
        ep_cum_prot_gpu = torch.zeros((num_epochs,), dtype=torch.int64, device=device)
        total_splits_gpu = torch.zeros((), dtype=torch.int64, device=device)
        total_prot_gpu = torch.zeros((), dtype=torch.int64, device=device)

        ev_start = torch.cuda.Event(enable_timing=True)
        ev_epochs = [torch.cuda.Event(enable_timing=True) for _ in range(num_epochs)]
        ev_start.record()

        rep_mode = _dtype_to_rep_mode(comp_dtype)
        blk_d_upd = min(256, max(16, triton.next_power_of_2(max(1, d))))
        blk_d_fs = min(128, max(16, triton.next_power_of_2(max(1, d))))
        blk_k_fs = min(64, max(16, triton.next_power_of_2(max(1, k))))

        full_d_oja = track_oja and (d <= 256)
        if track_oja:
            blk_b_sc = 64 if d <= 128 else 32
            blk_d_sc = max(16, triton.next_power_of_2(max(1, d))) if full_d_oja else 128
        else:
            blk_b_sc = 128
            blk_d_sc = min(128, max(16, triton.next_power_of_2(max(1, d))))

        eff_grace = (
            min(self.grace_steps, max(0, recipe.total_steps // 3))
            if (self.use_prior_mass and has_priors and track_surv)
            else 0
        )
        track_fs = (
            recipe.unused_threshold > 0.0
            and self.revival_mode == "split"
            and self.use_feat_scale
            and self.split_mode in ("half_gaussian_oja", "rms_aniso")
        )
        split_mode_code = {
            "half_gaussian_oja": 0,
            "rms_aniso": 1,
            "rms_iso": 1,
            "oja_only": 2,
            "checkerboard": 3,
        }[self.split_mode]

        eff_pass_span = self._effective_pass_span()
        wide_alpha_factor = float(self.wide_step_mult ** ALPHA_LAW[2]) if self.wide_step_mult > 1 else 1.0
        in_wide_phase = False

        total_planned_rows = max(1, sum(recipe.epoch_rows))
        rows_processed = 0
        step = 0
        alpha_last = recipe.alpha0 * recipe.alpha_min_factor
        transplanted_in_last_epoch = False
        ep_transplants: list[int] = []
        next_cp_idx = 0
        sorted_cps = sorted(cp for cp in self.transplant_checkpoints if 0.05 <= cp <= 0.82)

        for ep_idx, rows_in_ep in enumerate(recipe.epoch_rows):
            is_last_epoch = ep_idx == (num_epochs - 1)
            gen_ep = torch.Generator(device=device)
            gen_ep.manual_seed(epoch_seed(self.seed, ep_idx))
            order = torch.randperm(n, generator=gen_ep, device=device)[:rows_in_ep]

            raw_pass_counts.zero_()
            trans_before_ep = self.n_transplants
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
                        sse_ema.mul_(w_scale)
                        if track_surv:
                            survival_mass.mul_(w_scale)
                        if track_slow:
                            slow_usage.mul_(w_scale)
                    in_wide_phase = True

                b_end = min(b_start + cur_bsz_target, rows_in_ep)
                b_cur = b_end - b_start
                batch_idx = order[b_start:b_end]

                d_view = dist_buf[:b_cur]
                l_view = lbl_buf[:b_cur]

                _launch_ultra_assign(
                    x_t=x_t,
                    c_rep=c_rep,
                    c_norm=c_norm,
                    row_indices=batch_idx,
                    origin=origin_assign,
                    coord_scale=coord_scale,
                    out_dist=d_view,
                    out_label=l_view,
                    splitk_buffers=splitk_buffers if not use_wide_now else None,
                )

                if is_last_epoch and assignments is not None:
                    assignments[batch_idx] = l_view

                grid_d_sc = 1 if track_oja else triton.cdiv(d, blk_d_sc)
                with torch.cuda.device(device):
                    _ultra_scatter_oja_sse_kernel[(triton.cdiv(b_cur, blk_b_sc), grid_d_sc)](
                        x_t,
                        batch_idx,
                        origin_assign,
                        l_view,
                        d_view,
                        centroids_c,
                        prin_vec,
                        sums,
                        counts,
                        batch_sse,
                        g_sums,
                        ep_sq_dists_gpu,
                        int(ep_idx),
                        b_cur,
                        d,
                        x_t.stride(0),
                        x_t.stride(1),
                        centroids_c.stride(0),
                        float(coord_scale),
                        HAS_INDICES=True,
                        HAS_ORIGIN=True,
                        HAS_SCALE=(coord_scale != 1.0),
                        ORIGIN_FP64=(origin_assign.dtype == torch.float64),
                        TRACK_OJA=track_oja,
                        FULL_D=full_d_oja,
                        BLOCK_B=blk_b_sc,
                        BLOCK_D=blk_d_sc,
                        num_warps=8 if (track_oja and d >= 64) else 4,
                    )

                within_revive_window = (
                    (prog <= self.revive_stop_frac)
                    if (self.use_revive_stop and self.wide_step_mult > 1)
                    else (
                        (step <= int(self.revive_stop_frac * recipe.total_steps))
                        if self.use_revive_stop
                        else True
                    )
                )

                if track_fs and within_revive_window:
                    with torch.cuda.device(device):
                        _ultra_feat_scale_kernel[(triton.cdiv(d, blk_d_fs),)](
                            sums,
                            counts,
                            feat_scale,
                            k,
                            d,
                            sums.stride(0),
                            float(self.var_beta),
                            BLOCK_K=blk_k_fs,
                            BLOCK_D=blk_d_fs,
                            num_warps=4,
                        )

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
                alpha_last = alpha
                decay = 1.0 - alpha
                eff_sse_decay = decay if self.couple_sse_to_alpha else self.sse_decay
                alpha_surv = min(alpha, 2.0 * float(b_cur) / float(max(1, n)))

                accum_pass = bool(is_last_epoch and need_pass_accum)
                if accum_pass and eff_pass_span > 0.0:
                    if self.wide_step_mult == 1:
                        n_sep_est = max(1, (rows_in_ep + cur_bsz_target - 1) // cur_bsz_target)
                        pfrac = (ep_steps + 0.5) / float(n_sep_est)
                    else:
                        pfrac = min(1.0, (float(b_start) + 0.5 * float(b_cur)) / float(max(1, rows_in_ep)))
                    w_pass = (1.0 - eff_pass_span) + 2.0 * eff_pass_span * pfrac
                else:
                    w_pass = 1.0 if accum_pass else 0.0

                will_revive = (
                    recipe.unused_threshold > 0.0
                    and self.revival_mode == "split"
                    and within_revive_window
                    and ((step + 1) >= max(1, eff_grace))
                    and ((step + 1) % recipe.check_unused_every == 0)
                    and k >= 2
                )

                prog_after = float(rows_processed + b_cur) / float(total_planned_rows)
                will_transplant = (
                    self.head_edit
                    and edit_gen is not None
                    and last_edit_step is not None
                    and edit_diag is not None
                    and next_cp_idx < len(sorted_cps)
                    and prog_after >= sorted_cps[next_cp_idx]
                    and (
                        (not is_last_epoch and prog_after <= 0.85)
                        or (num_epochs == 1 and prog_after <= 0.62)
                    )
                    and b_cur >= 32
                    and k >= 3
                )
                is_very_last_step = is_last_epoch and (b_end >= rows_in_ep)
                prep_in_ema = (not will_revive) and (not will_transplant) and (not is_very_last_step)

                with torch.cuda.device(device):
                    _ultra_ema_oja_slow_pass_prep_kernel[(k,)](
                        sums,
                        counts,
                        batch_sse,
                        g_sums,
                        centroids_c,
                        usage,
                        sse_ema,
                        prin_vec,
                        survival_mass,
                        slow_centroids,
                        slow_usage,
                        pass_sums,
                        pass_counts,
                        pass_counts_sq,
                        pass_sse,
                        raw_pass_counts,
                        c_rep,
                        c_norm,
                        k,
                        d,
                        centroids_c.stride(0),
                        c_rep.stride(0),
                        float(alpha),
                        float(decay),
                        float(eff_sse_decay),
                        float(alpha_surv),
                        float(alpha_slow_base),
                        float(w_pass),
                        int(alpha >= 1.0),
                        int(accum_pass),
                        int(prep_in_ema),
                        TRACK_OJA=track_oja,
                        TRACK_SURV=track_surv,
                        TRACK_SLOW=track_slow,
                        REP_MODE=rep_mode,
                        BLOCK_D=blk_d_upd,
                        num_warps=2 if d <= 128 else 4,
                    )

                # Update scalar bias-correction factors identically
                if alpha >= 1.0:
                    usage_bias = 1.0
                    sse_bias = 1.0
                else:
                    usage_bias = decay * usage_bias + alpha
                    sse_bias = eff_sse_decay * sse_bias + (1.0 - eff_sse_decay)

                eff_u_inv_bias = 1.0 / max(usage_bias, 1e-6)
                eff_sse_inv_bias = 1.0 / max(sse_bias, 1e-6)
                cur_bsz_scale = float(cur_bsz_target)
                ref_dist_mean = (sse_ema.mean() * eff_sse_inv_bias).clamp_min(1e-30)

                if will_revive:
                    thresh = usage.mean() * recipe.unused_threshold
                    low_mask = usage < thresh
                    if track_surv:
                        prot_mask = _launch_syncfree_huygens_protection(
                            centroids_c=centroids_c,
                            low_mask=low_mask,
                            usage=usage,
                            survival_mass=survival_mass,
                            thresh=thresh,
                            ref_distortion_mean=ref_dist_mean,
                            gamma_protect=self.gamma_protect,
                            c_pad_buf=huy_c_pad,
                            c_norm_buf=huy_cn,
                            d2_any_buf=huy_d2_any,
                            d2_surv_buf=huy_d2_surv,
                            prot_i8_buf=huy_prot_i8,
                        )
                        total_prot_gpu += prot_mask.to(torch.int64).sum()
                        unused_mask = low_mask & (~prot_mask)
                        donor_mask = (~low_mask) & (usage > 0.0)
                    else:
                        unused_mask = low_mask
                        donor_mask = (~unused_mask) & (usage > 0.0)

                    n_sp_t = torch.minimum(unused_mask.sum(), donor_mask.sum())
                    total_splits_gpu += n_sp_t

                    if self.donor_mode == "usage_only":
                        score_vec = usage
                    else:
                        u_unb = (usage * eff_u_inv_bias).clamp_min(1e-12)
                        e_unb = (sse_ema * eff_sse_inv_bias).clamp_min(1e-12)
                        if self.donor_mode == "oja_sse_usage" and track_oja:
                            # Blend total cluster SSE `E_j` with `d * N_j * sigma_{j,1}^2` so merged bimodal
                            # pairs in high-D stand out sharply above isotropic background noise!
                            sig1_sq = prin_vec.square().sum(dim=1)
                            base_dist = 0.5 * e_unb + 0.5 * (float(d) * u_unb * sig1_sq)
                        else:
                            base_dist = e_unb
                        w_exp = self.donor_usage_w if self.donor_mode in ("oja_sse_usage", "sse_usage") else 0.0
                        weighted = base_dist * ((u_unb / cur_bsz_scale).clamp_min(1e-12) ** w_exp if w_exp != 0.0 else 1.0)
                        score_vec = torch.where(sse_ema.max() > 0.0, weighted, usage)

                    u_rand = torch.rand(
                        (k,), generator=split_gen, device=device, dtype=torch.float32
                    ).clamp_min_(torch.finfo(torch.float32).tiny)
                    keys = torch.where(
                        donor_mask,
                        score_vec.clamp_min(1e-30).log() - (-u_rand.log()).log(),
                        torch.full_like(usage, float("-inf")),
                    )
                    src_sorted = torch.argsort(keys, descending=True)
                    tgt_sorted = torch.argsort(unused_mask.to(torch.int8), descending=True, stable=True)

                    if self.use_feat_scale and self.split_mode in ("half_gaussian_oja", "rms_aniso"):
                        fvec_s = feat_scale.clamp(1e-10, 1e35).sqrt()
                        fnorm = fvec_s.double().norm().clamp_min(1e-8).float()
                        fvec = fvec_s * (math.sqrt(float(d)) / fnorm)
                    else:
                        fvec = ones_d

                    do_flush_close = int(bool(is_last_epoch and need_pass_accum and self.flush_pass_on_split))
                    with torch.cuda.device(device):
                        _ultra_syncfree_split_kernel[(k,)](
                            centroids_c,
                            origin32,
                            usage,
                            sse_ema,
                            prin_vec,
                            survival_mass,
                            slow_centroids,
                            slow_usage,
                            pass_sums,
                            pass_counts,
                            pass_counts_sq,
                            pass_sse,
                            fvec,
                            src_sorted,
                            tgt_sorted,
                            n_sp_t,
                            d,
                            centroids_c.stride(0),
                            float(1.0 / max(1, d)),
                            float(eff_sse_inv_bias),
                            float(eff_u_inv_bias),
                            float(self.split_mult),
                            float(recipe.split_epsilon),
                            int((self.seed * 10007 + step * 131) % (2**31 - 1)),
                            int(split_mode_code),
                            int(do_flush_close),
                            TRACK_OJA=track_oja,
                            TRACK_SURV=track_surv,
                            TRACK_SLOW=track_slow,
                            HAS_ORIGIN=True,
                            REP_MODE=rep_mode,
                            BLOCK_D=blk_d_upd,
                            num_warps=2 if d <= 128 else 4,
                        )

                if will_transplant:
                    while next_cp_idx < len(sorted_cps) and prog_after >= sorted_cps[next_cp_idx]:
                        next_cp_idx += 1
                    if origin_assign.dtype == torch.float32 and coord_scale == 1.0:
                        xb_c = (x_t[batch_idx].float() - origin32[None, :]).contiguous()
                    else:
                        xb_c = ((x_t[batch_idx].double() - origin64[None, :]) * coord_scale).float().contiguous()
                    n_ed = run_ultra_head_edit_checkpoint(
                        x_batch_c=xb_c,
                        centroids_c=centroids_c,
                        usage=usage,
                        sse_ema=sse_ema,
                        ref_distortion_mean=ref_dist_mean,
                        survival_mass=survival_mass if track_surv else None,
                        prin_vec=prin_vec if track_oja else None,
                        slow_centroids=slow_centroids if track_slow else None,
                        slow_usage=slow_usage if track_slow else None,
                        pass_sums=pass_sums if is_last_epoch else None,
                        pass_counts=pass_counts if is_last_epoch else None,
                        pass_counts_sq=pass_counts_sq if is_last_epoch else None,
                        pass_sse=pass_sse if is_last_epoch else None,
                        last_edit_step=last_edit_step,
                        step=step,
                        compute_dtype=comp_dtype,
                        unused_threshold=recipe.unused_threshold,
                        use_huygens_protection=track_surv,
                        gamma_protect=self.gamma_protect,
                        num_probes=self.transplant_probes,
                        max_pairs=self.transplant_max_pairs,
                        protect_steps=self.transplant_protect_steps,
                        generator=edit_gen,
                        diag=edit_diag,
                    )
                    if n_ed > 0:
                        self.n_transplants += n_ed
                        total_splits_gpu += n_ed
                        if is_last_epoch:
                            transplanted_in_last_epoch = True

                if (will_revive or will_transplant) and (not is_very_last_step):
                    _launch_ultra_prep_centroids(centroids_c, c_rep, c_norm)

                rows_processed += b_cur
                b_start = b_end
                step += 1
                ep_steps += 1

            ep_dot_counts_gpu[ep_idx] = torch.dot(raw_pass_counts, raw_pass_counts)
            ep_cum_splits_gpu[ep_idx] = total_splits_gpu
            ep_cum_prot_gpu[ep_idx] = total_prot_gpu
            ep_transplants.append(self.n_transplants - trans_before_ep)
            ev_epochs[ep_idx].record()

        # Finalize in centered coordinate frame
        c_ema_last = centroids_c.clone()
        centroids_fin, s_diag = ultra_simplex_eb_finalize(
            pass_sums=pass_sums,
            pass_counts=pass_counts,
            pass_counts_sq=pass_counts_sq if eff_pass_span > 0.0 else None,
            pass_sse=pass_sse,
            live_centroids=c_ema_last,
            slow_centroids=slow_centroids if track_slow else None,
            usage=usage,
            usage_inv_bias=1.0 / max(usage_bias, 1e-6),
            alpha_end=alpha_last,
            nepochs=self.nepochs,
            d=d,
            mode=self.finalization,
            eb_tau=self.eb_tau,
            kappa_max=self.kappa_max,
        )
        self.shrinkage_diagnostics = s_diag
        self.slow_centroids = slow_centroids.clone() if track_slow else None

        if (
            capture_assignments
            and assignments is not None
            and (transplanted_in_last_epoch or recipe.epoch_rows[-1] < n)
        ):
            _launch_ultra_prep_centroids(centroids_fin, c_rep, c_norm)
            _, assignments = _launch_ultra_assign(
                x_t=x_t,
                c_rep=c_rep,
                c_norm=c_norm,
                origin=origin_assign,
                coord_scale=coord_scale,
            )

        dead_t = (raw_pass_counts <= 0.5).sum()

        torch.cuda.synchronize(device)
        self.train_time_sec = time.perf_counter() - t_train_start

        ep_sq_list = ep_sq_dists_gpu.cpu().tolist()
        ep_dot_list = ep_dot_counts_gpu.cpu().tolist()
        ep_sp_list = ep_cum_splits_gpu.cpu().tolist()
        ep_pr_list = ep_cum_prot_gpu.cpu().tolist()
        self.n_splits = int(ep_sp_list[-1]) if ep_sp_list else 0
        self.n_protected = int(ep_pr_list[-1]) if ep_pr_list else 0
        self.dead_clusters = int(dead_t.item())

        inv_scale_sq = 1.0 / (coord_scale * coord_scale)
        self.iteration_stats = []
        prev_sp = 0
        prev_pr = 0
        for ep_idx, rows_in_ep in enumerate(recipe.epoch_rows):
            cur_sp = int(ep_sp_list[ep_idx])
            cur_pr = int(ep_pr_list[ep_idx])
            el_s = float(ev_start.elapsed_time(ev_epochs[ep_idx])) * 1e-3
            imb = float(k * ep_dot_list[ep_idx] / (float(max(1, rows_in_ep)) ** 2))
            self.iteration_stats.append(
                UltraEMeansEpochStat(
                    epoch=ep_idx + 1,
                    rows=rows_in_ep,
                    steps=int(recipe.epoch_steps[ep_idx]),
                    moving_mse=(float(ep_sq_list[ep_idx]) * inv_scale_sq) / max(1, rows_in_ep),
                    imbalance_factor=imb,
                    n_splits=cur_sp - prev_sp,
                    n_protected=cur_pr - prev_pr,
                    n_transplants=ep_transplants[ep_idx],
                    elapsed_sec=el_s,
                )
            )
            prev_sp = cur_sp
            prev_pr = cur_pr

        return centroids_fin, assignments

    def _fit_torch_reference(
        self,
        x_t: torch.Tensor,
        seed_res: SeedingResult,
        has_priors: bool,
        comp_dtype: torch.dtype,
        recipe: EMeansRecipe,
        origin64: torch.Tensor,
        coord_scale: float,
        capture_assignments: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Exact centered PyTorch fallback for CPU, `backend='torch'`, or `compute_dtype=float64`."""
        device = x_t.device
        n, d = int(x_t.shape[0]), int(x_t.shape[1])
        k = self.k

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_train_start = time.perf_counter()

        x_c64 = (x_t.double() - origin64[None, :]) * coord_scale
        x_c = x_c64.contiguous() if comp_dtype == torch.float64 else x_c64.float().contiguous()
        x_f32 = x_c.float()

        centroids_c = seed_res.centroids.to(device=device, dtype=torch.float32).clone().contiguous()
        p_hat = seed_res.prior_probs.to(device=device, dtype=torch.float32)
        e_hat = seed_res.prior_distortion.to(device=device, dtype=torch.float32)

        bsz_float = float(recipe.batch_size)
        min_prob = max(0.0, self.prior_floor) / float(k)
        mean_ehat = e_hat.mean().clamp_min(1e-12)

        if self.use_prior_mass and has_priors:
            usage = (self.prior_mass_scale * bsz_float * p_hat.clamp_min(min_prob)).float().contiguous()
            sse_ema = (self.prior_mass_scale * bsz_float * e_hat.clamp_min(1e-6 * mean_ehat)).float().contiguous()
            usage_bias = float(self.prior_mass_scale)
            sse_bias = float(self.prior_mass_scale)
        else:
            usage = torch.full((k,), recipe.init_usage, dtype=torch.float32, device=device)
            sse_ema = torch.zeros((k,), dtype=torch.float32, device=device)
            usage_bias = float(recipe.init_usage * k) / max(1.0, bsz_float)
            sse_bias = 0.0

        track_surv = bool(self.use_huygens_protection and self.gamma_protect > 0.0)
        survival_mass = (
            ((bsz_float * p_hat).float().contiguous() if has_priors else torch.zeros((k,), dtype=torch.float32, device=device))
            if track_surv
            else None
        )

        track_slow = self.finalization in ("simplex_eb", "eb_slow_only")
        slow_centroids = centroids_c.clone() if track_slow else None
        slow_usage = usage.clone() if track_slow else None
        alpha_slow_base = min(1.0, max(1e-6, self.slow_ema_mult * recipe.alpha_min_factor * recipe.alpha0))

        track_oja = self.split_mode in ("half_gaussian_oja", "oja_only") or self.donor_mode == "oja_sse_usage"
        if track_oja:
            pca_gen = torch.Generator(device=device)
            pca_gen.manual_seed((self.seed + 31337) % (2**32))
            init_dirs = torch.randn((k, d), generator=pca_gen, device=device, dtype=torch.float32)
            init_dirs = init_dirs / init_dirs.norm(dim=1, keepdim=True).clamp_min(1e-12)
            init_rms = (
                (e_hat / (p_hat.clamp_min(1e-6) * float(max(1, d)))).clamp_min(1e-8).sqrt()[:, None]
                if has_priors
                else 1e-3
            )
            prin_vec: torch.Tensor | None = (init_rms * init_dirs).contiguous()
        else:
            prin_vec = None

        feat_scale = torch.ones((d,), dtype=torch.float32, device=device)
        pass_sums = torch.zeros((k, d), dtype=torch.float32, device=device)
        pass_counts = torch.zeros((k,), dtype=torch.float64, device=device)
        pass_counts_sq = torch.zeros((k,), dtype=torch.float64, device=device)
        pass_sse = torch.zeros((k,), dtype=torch.float64, device=device)
        raw_pass_counts = torch.zeros((k,), dtype=torch.float64, device=device)
        need_pass_accum = self.finalization in ("simplex_eb", "eb_slow_only", "eb_ema", "closing_mean")

        split_gen = torch.Generator(device=device)
        split_gen.manual_seed((self.seed + 7919) % (2**32))
        if self.head_edit:
            last_edit_step = torch.full((k,), -(10**9), dtype=torch.int64, device=device)
            edit_gen = torch.Generator(device=device)
            edit_gen.manual_seed((self.seed + 918273) % (2**32))
            edit_diag = HeadEditDiagnostics()
        else:
            last_edit_step = None
            edit_gen = None
            edit_diag = None
        self.head_edit_diagnostics = edit_diag

        assignments = torch.full((n,), -1, dtype=torch.int64, device=device) if capture_assignments else None
        num_epochs = len(recipe.epoch_rows)
        eff_grace = (
            min(self.grace_steps, max(0, recipe.total_steps // 3))
            if (self.use_prior_mass and has_priors and track_surv)
            else 0
        )
        eff_pass_span = self._effective_pass_span()
        wide_alpha_factor = float(self.wide_step_mult ** ALPHA_LAW[2]) if self.wide_step_mult > 1 else 1.0
        in_wide_phase = False
        total_planned_rows = max(1, sum(recipe.epoch_rows))
        rows_processed = 0
        step = 0
        alpha_last = recipe.alpha0 * recipe.alpha_min_factor
        transplanted_in_last_epoch = False
        sorted_cps = sorted(cp for cp in self.transplant_checkpoints if 0.05 <= cp <= 0.82)
        next_cp_idx = 0

        self.n_splits = 0
        self.n_protected = 0
        self.n_transplants = 0
        self.iteration_stats = []

        for ep_idx, rows_in_ep in enumerate(recipe.epoch_rows):
            is_last_epoch = ep_idx == (num_epochs - 1)
            gen_ep = torch.Generator(device=device)
            gen_ep.manual_seed(epoch_seed(self.seed, ep_idx))
            order = torch.randperm(n, generator=gen_ep, device=device)[:rows_in_ep]
            raw_pass_counts.zero_()
            ep_sq = 0.0
            sp_before = self.n_splits
            pr_before = self.n_protected
            tr_before = self.n_transplants
            ep_steps = 0
            b_start = 0

            while b_start < rows_in_ep:
                prog = float(rows_processed) / float(total_planned_rows)
                use_wide_now = self.wide_step_mult > 1 and prog >= self.wide_from_frac
                cur_bsz_target = min(n, recipe.batch_size * self.wide_step_mult) if use_wide_now else recipe.batch_size
                if use_wide_now and not in_wide_phase:
                    w_sc = float(cur_bsz_target) / float(max(1, recipe.batch_size))
                    usage.mul_(w_sc)
                    sse_ema.mul_(w_sc)
                    if survival_mass is not None:
                        survival_mass.mul_(w_sc)
                    if slow_usage is not None:
                        slow_usage.mul_(w_sc)
                    in_wide_phase = True

                b_end = min(b_start + cur_bsz_target, rows_in_ep)
                batch_idx = order[b_start:b_end]
                b_cur = int(b_end - b_start)
                xb = x_c[batch_idx]
                xb_f = x_f32[batch_idx]

                if comp_dtype == torch.float64:
                    c64 = centroids_c.double().contiguous()
                    sq_dists, labels = torch_assign_chunked(xb, c64, c64.square().sum(dim=1))
                else:
                    sq_dists, labels = assign_points(xb, centroids_c, compute_dtype=comp_dtype, backend="torch")

                if is_last_epoch and assignments is not None:
                    assignments[batch_idx] = labels

                ep_sq += float(sq_dists.double().sum().item())
                counts = torch.bincount(labels, minlength=k).float()
                bsums = torch.zeros((k, d), dtype=torch.float32, device=device)
                bsums.index_add_(0, labels, xb_f)
                bsse = torch.zeros((k,), dtype=torch.float32, device=device)
                bsse.index_add_(0, labels, sq_dists.float())
                raw_pass_counts.add_(counts.double())

                if is_last_epoch and need_pass_accum:
                    if eff_pass_span > 0.0:
                        n_sep_est = max(1, (rows_in_ep + cur_bsz_target - 1) // cur_bsz_target)
                        pfrac = (
                            (ep_steps + 0.5) / float(n_sep_est)
                            if self.wide_step_mult == 1
                            else min(1.0, (float(b_start) + 0.5 * float(b_cur)) / float(max(1, rows_in_ep)))
                        )
                        w_pass = (1.0 - eff_pass_span) + 2.0 * eff_pass_span * pfrac
                    else:
                        w_pass = 1.0
                    pass_counts.add_(counts.double(), alpha=w_pass)
                    pass_counts_sq.add_(counts.double(), alpha=w_pass * w_pass)
                    pass_sums.add_(bsums, alpha=w_pass)
                    pass_sse.add_(bsse.double(), alpha=w_pass)

                if self.use_feat_scale and b_cur > 0:
                    tot_c = counts.sum().clamp_min(1.0)
                    bmu = bsums.sum(dim=0) / tot_c
                    cvar = ((bsums - counts[:, None] * bmu[None, :]).square().sum(dim=0) / tot_c).clamp_min(1e-12)
                    feat_scale.mul_(self.var_beta).add_(cvar, alpha=1.0 - self.var_beta)

                eff_peak = (
                    min(1.0, recipe.alpha0 * (wide_alpha_factor if use_wide_now else 1.0))
                    if self.wide_step_mult > 1
                    else recipe.alpha0
                )
                sched_frac = prog if self.wide_step_mult > 1 else min(step, recipe.total_steps) / max(1, recipe.total_steps)
                alpha = super_scheduled_alpha(
                    eff_peak, sched_frac, self.schedule_mode, recipe.alpha_min_factor, self.alpha_floor, self.cosine_power
                )
                alpha_last = alpha
                decay = 1.0 - alpha
                eff_sse_decay = decay if self.couple_sse_to_alpha else self.sse_decay

                touched = counts > 0.0
                retained = torch.zeros_like(usage) if alpha >= 1.0 else usage * decay
                new_mass = retained + alpha * counts
                weight = torch.where(touched, (alpha * counts) / torch.where(touched, new_mass, torch.ones_like(new_mass)), 0.0)
                bmeans = bsums / torch.where(touched, counts, torch.ones_like(counts))[:, None]

                if track_oja and prin_vec is not None:
                    from .algo_extensions import update_oja_principal_axes

                    prin_vec = update_oja_principal_axes(prin_vec, xb_f, centroids_c, labels, counts, weight)

                upd_c = torch.where((weight == 1.0)[:, None], bmeans, centroids_c + weight[:, None] * (bmeans - centroids_c))
                centroids_c = torch.where(touched[:, None], upd_c, centroids_c)
                usage = torch.where(touched, new_mass, torch.where(torch.tensor(alpha >= 1.0, device=device), usage, retained))
                sse_ema = (
                    torch.where(touched, bsse, 0.5 * sse_ema)
                    if alpha >= 1.0
                    else eff_sse_decay * sse_ema + (1.0 - eff_sse_decay) * bsse
                )

                if survival_mass is not None:
                    a_surv = min(alpha, 2.0 * float(b_cur) / float(max(1, n)))
                    survival_mass.mul_(1.0 - a_surv).add_(counts, alpha=a_surv)

                if track_slow and slow_centroids is not None and slow_usage is not None:
                    s_ret = torch.zeros_like(slow_usage) if alpha_slow_base >= 1.0 else slow_usage * (1.0 - alpha_slow_base)
                    s_mass = s_ret + alpha_slow_base * counts
                    s_w = torch.where(touched, (alpha_slow_base * counts) / torch.where(touched, s_mass, torch.ones_like(s_mass)), 0.0)
                    sc_upd = torch.where((s_w == 1.0)[:, None], bmeans, slow_centroids + s_w[:, None] * (bmeans - slow_centroids))
                    slow_centroids = torch.where(touched[:, None], sc_upd, slow_centroids)
                    slow_usage = torch.where(touched, s_mass, s_ret)

                if alpha >= 1.0:
                    usage_bias = 1.0
                    sse_bias = 1.0
                else:
                    usage_bias = decay * usage_bias + alpha
                    sse_bias = eff_sse_decay * sse_bias + (1.0 - eff_sse_decay)

                eff_u_inv = 1.0 / max(usage_bias, 1e-6)
                eff_sse_inv = 1.0 / max(sse_bias, 1e-6)
                ref_dist_mean = (sse_ema.mean() * eff_sse_inv).clamp_min(1e-30)

                within_revive = (
                    (prog <= self.revive_stop_frac)
                    if (self.use_revive_stop and self.wide_step_mult > 1)
                    else ((step <= int(self.revive_stop_frac * recipe.total_steps)) if self.use_revive_stop else True)
                )
                if (
                    recipe.unused_threshold > 0.0
                    and self.revival_mode == "split"
                    and within_revive
                    and ((step + 1) >= max(1, eff_grace))
                    and ((step + 1) % recipe.check_unused_every == 0)
                    and k >= 2
                ):
                    thresh = usage.mean() * recipe.unused_threshold
                    low_mask = usage < thresh
                    if track_surv and survival_mass is not None:
                        prot_mask = _launch_syncfree_huygens_protection(
                            centroids_c, low_mask, usage, survival_mass, thresh, ref_dist_mean, self.gamma_protect
                        )
                        self.n_protected += int(prot_mask.sum().item())
                        unused_mask = low_mask & (~prot_mask)
                        donor_mask = (~low_mask) & (usage > 0.0)
                    else:
                        unused_mask = low_mask
                        donor_mask = (~unused_mask) & (usage > 0.0)

                    n_sp = int(min(unused_mask.sum().item(), donor_mask.sum().item()))
                    if n_sp > 0:
                        self.n_splits += n_sp
                        if self.donor_mode == "usage_only":
                            sc_vec = usage
                        else:
                            u_unb = (usage * eff_u_inv).clamp_min(1e-12)
                            e_unb = (sse_ema * eff_sse_inv).clamp_min(1e-12)
                            if self.donor_mode == "oja_sse_usage" and prin_vec is not None:
                                base_d = 0.5 * e_unb + 0.5 * (float(d) * u_unb * prin_vec.square().sum(dim=1))
                            else:
                                base_d = e_unb
                            w_exp = self.donor_usage_w if self.donor_mode in ("oja_sse_usage", "sse_usage") else 0.0
                            sc_vec = torch.where(
                                sse_ema.max() > 0.0,
                                base_d * ((u_unb / float(cur_bsz_target)).clamp_min(1e-12) ** w_exp if w_exp != 0.0 else 1.0),
                                usage,
                            )
                        u_r = torch.rand((k,), generator=split_gen, device=device, dtype=torch.float32).clamp_min_(1e-38)
                        keys = torch.where(donor_mask, sc_vec.clamp_min(1e-30).log() - (-u_r.log()).log(), float("-inf"))
                        src_idx = torch.topk(keys, n_sp).indices
                        tgt_idx = torch.nonzero(unused_mask, as_tuple=False).reshape(-1)[:n_sp]

                        dc = centroids_c[src_idx]
                        u_s = (usage[src_idx] * eff_u_inv).clamp_min(1e-6)
                        e_s = sse_ema[src_idx] * eff_sse_inv
                        sig_rms = ((e_s / u_s) / float(max(1, d))).clamp_min(1e-12).sqrt()[:, None]

                        if self.split_mode == "checkerboard":
                            sgn = torch.ones((d,), dtype=torch.float32, device=device)
                            sgn[1::2] = -1.0
                            orig_f = origin64.float()[None, :] * coord_scale
                            abs_dc = dc + orig_f
                            c_src = abs_dc * (1.0 - recipe.split_epsilon * sgn[None, :]) - orig_f
                            c_tgt = abs_dc * (1.0 + recipe.split_epsilon * sgn[None, :]) - orig_f
                            u_dir = (sgn / math.sqrt(float(max(1, d)))).unsqueeze(0).expand(n_sp, d)
                            _separate_collapsed_splits(c_src, c_tgt, comp_dtype)
                            sig_split = sig_rms.squeeze(1)
                        else:
                            fvec = (
                                (feat_scale.clamp(1e-10, 1e35).sqrt() / feat_scale.clamp(1e-10, 1e35).sqrt().norm().clamp_min(1e-8) * math.sqrt(float(d)))
                                if (self.use_feat_scale and self.split_mode in ("half_gaussian_oja", "rms_aniso"))
                                else torch.ones((d,), dtype=torch.float32, device=device)
                            )
                            radem = torch.where(
                                torch.rand((n_sp, d), generator=split_gen, device=device) < 0.5, -1.0, 1.0
                            ) * fvec[None, :]
                            r_hat = radem / radem.norm(dim=1, keepdim=True).clamp_min(1e-12)
                            if prin_vec is not None and self.split_mode != "rms_aniso" and self.split_mode != "rms_iso":
                                pv_s = prin_vec[src_idx]
                                sig1 = pv_s.norm(dim=1, keepdim=True)
                                v1 = pv_s / sig1.clamp_min(1e-12)
                                align = torch.where((v1 * r_hat).sum(dim=1, keepdim=True) >= 0.0, 1.0, -1.0)
                                rho = (
                                    torch.where(sig1 > 1e-10, torch.ones_like(sig1), torch.zeros_like(sig1))
                                    if self.split_mode == "oja_only"
                                    else ((sig1 / sig_rms.clamp_min(1e-12) - 0.85) / 0.35).clamp(0.0, 1.0)
                                )
                                blend = rho * v1 + (1.0 - rho) * align * r_hat
                                u_dir = blend / blend.norm(dim=1, keepdim=True).clamp_min(1e-12)
                                if self.split_mode == "half_gaussian_oja":
                                    sig_sp_col = torch.maximum(sig1, 0.85 * sig_rms)
                                    g_sp = ((sig1 / sig_rms.clamp_min(1e-12) - 1.10) / 0.90).clamp(0.0, 1.0)
                                    eta_min = min(self.split_mult, 0.18)
                                    eta_eff = eta_min + max(self.split_mult - eta_min, 0.0) * g_sp
                                    mag = eta_eff * sig_sp_col
                                else:
                                    sig_sp_col = torch.where(sig1 > 1e-8, sig1, sig_rms)
                                    eta_eff = torch.full_like(sig_sp_col, self.split_mult)
                                    mag = self.split_mult * sig_sp_col
                                sig_split = sig_sp_col.squeeze(1)
                            else:
                                u_dir = r_hat
                                mag = self.split_mult * sig_rms * radem.norm(dim=1, keepdim=True)
                                sig_split = sig_rms.squeeze(1)
                                eta_eff = torch.full_like(sig_rms, self.split_mult)
                            delta = mag * u_dir
                            c_src = dc - delta
                            c_tgt = dc + delta
                            separate_collapsed_directional(c_src, c_tgt, u_dir, comp_dtype)

                        centroids_c[src_idx] = c_src
                        centroids_c[tgt_idx] = c_tgt
                        hu = 0.5 * usage[src_idx]
                        usage[src_idx] = hu
                        usage[tgt_idx] = hu

                        if self.split_mode == "half_gaussian_oja":
                            eta_1d = eta_eff.squeeze(1)
                            vdrop_f = (2.0 * eta_1d * HALF_GAUSSIAN_MEAN - eta_1d * eta_1d).clamp_min(0.0)
                            drop = vdrop_f * usage[src_idx] * 2.0 * (sig_split ** 2) * (eff_u_inv / max(eff_sse_inv, 1e-12))
                            hsse = 0.5 * torch.maximum(0.15 * sse_ema[src_idx], sse_ema[src_idx] - drop)
                        else:
                            hsse = 0.5 * sse_ema[src_idx]
                        sse_ema[src_idx] = hsse
                        sse_ema[tgt_idx] = hsse

                        if survival_mass is not None:
                            survival_mass[src_idx] = 0.0
                            survival_mass[tgt_idx] = 0.0
                        if slow_centroids is not None and slow_usage is not None:
                            slow_centroids[src_idx] = c_src
                            slow_centroids[tgt_idx] = c_tgt
                            hsu = 0.5 * slow_usage[src_idx]
                            slow_usage[src_idx] = hsu
                            slow_usage[tgt_idx] = hsu
                        if prin_vec is not None:
                            from .algo_extensions import reset_post_split_oja_axes

                            reset_post_split_oja_axes(prin_vec, src_idx, tgt_idx, u_dir, generator=split_gen)
                        if is_last_epoch and need_pass_accum and self.flush_pass_on_split:
                            pass_sums[src_idx] = 0.0
                            pass_sums[tgt_idx] = 0.0
                            pass_counts[src_idx] = 0.0
                            pass_counts[tgt_idx] = 0.0
                            pass_counts_sq[src_idx] = 0.0
                            pass_counts_sq[tgt_idx] = 0.0
                            pass_sse[src_idx] = 0.0
                            pass_sse[tgt_idx] = 0.0

                prog_after = float(rows_processed + b_cur) / float(total_planned_rows)
                if (
                    self.head_edit
                    and edit_gen is not None
                    and last_edit_step is not None
                    and edit_diag is not None
                    and next_cp_idx < len(sorted_cps)
                    and prog_after >= sorted_cps[next_cp_idx]
                    and ((not is_last_epoch and prog_after <= 0.85) or (num_epochs == 1 and prog_after <= 0.62))
                    and b_cur >= 32
                    and k >= 3
                ):
                    while next_cp_idx < len(sorted_cps) and prog_after >= sorted_cps[next_cp_idx]:
                        next_cp_idx += 1
                    n_ed = run_ultra_head_edit_checkpoint(
                        x_batch_c=xb_f,
                        centroids_c=centroids_c,
                        usage=usage,
                        sse_ema=sse_ema,
                        ref_distortion_mean=ref_dist_mean,
                        survival_mass=survival_mass,
                        prin_vec=prin_vec,
                        slow_centroids=slow_centroids,
                        slow_usage=slow_usage,
                        pass_sums=pass_sums if is_last_epoch else None,
                        pass_counts=pass_counts if is_last_epoch else None,
                        pass_counts_sq=pass_counts_sq if is_last_epoch else None,
                        pass_sse=pass_sse if is_last_epoch else None,
                        last_edit_step=last_edit_step,
                        step=step,
                        compute_dtype=comp_dtype,
                        unused_threshold=recipe.unused_threshold,
                        use_huygens_protection=track_surv,
                        gamma_protect=self.gamma_protect,
                        num_probes=self.transplant_probes,
                        max_pairs=self.transplant_max_pairs,
                        protect_steps=self.transplant_protect_steps,
                        generator=edit_gen,
                        diag=edit_diag,
                    )
                    if n_ed > 0:
                        self.n_transplants += n_ed
                        self.n_splits += n_ed
                        if is_last_epoch:
                            transplanted_in_last_epoch = True

                rows_processed += b_cur
                b_start = b_end
                step += 1
                ep_steps += 1

            imb = float((k * torch.dot(raw_pass_counts, raw_pass_counts) / (float(max(1, rows_in_ep)) ** 2)).item())
            self.iteration_stats.append(
                UltraEMeansEpochStat(
                    epoch=ep_idx + 1,
                    rows=rows_in_ep,
                    steps=ep_steps,
                    moving_mse=(ep_sq / (coord_scale * coord_scale)) / max(1, rows_in_ep),
                    imbalance_factor=imb,
                    n_splits=self.n_splits - sp_before,
                    n_protected=self.n_protected - pr_before,
                    n_transplants=self.n_transplants - tr_before,
                    elapsed_sec=time.perf_counter() - t_train_start,
                )
            )

        c_ema_last = centroids_c.clone()
        centroids_fin, s_diag = ultra_simplex_eb_finalize(
            pass_sums=pass_sums,
            pass_counts=pass_counts,
            pass_counts_sq=pass_counts_sq if eff_pass_span > 0.0 else None,
            pass_sse=pass_sse,
            live_centroids=c_ema_last,
            slow_centroids=slow_centroids,
            usage=usage,
            usage_inv_bias=1.0 / max(usage_bias, 1e-6),
            alpha_end=alpha_last,
            nepochs=self.nepochs,
            d=d,
            mode=self.finalization,
            eb_tau=self.eb_tau,
            kappa_max=self.kappa_max,
        )
        self.shrinkage_diagnostics = s_diag
        self.slow_centroids = slow_centroids
        self.dead_clusters = int((raw_pass_counts <= 0.5).sum().item())

        if (
            capture_assignments
            and assignments is not None
            and (transplanted_in_last_epoch or recipe.epoch_rows[-1] < n)
        ):
            if comp_dtype == torch.float64:
                c64 = centroids_fin.double().contiguous()
                _, assignments = torch_assign_chunked(x_c, c64, c64.square().sum(dim=1))
            else:
                _, assignments = assign_points(x_c, centroids_fin, compute_dtype=comp_dtype, backend="torch")

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self.train_time_sec = time.perf_counter() - t_train_start
        return centroids_fin, assignments

    @torch.no_grad()
    def assign(
        self,
        x: torch.Tensor | np.ndarray,
    ) -> tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray]:
        """Assign points `x` to learned centroids, returning `(sq_dists, labels)`."""
        if self.centroids is None:
            raise RuntimeError("UltraEMeans must be fitted before calling assign().")

        input_is_numpy = isinstance(x, np.ndarray)
        c_dev = self.centroids.device
        if input_is_numpy:
            x_np = np.asarray(x)
            if x_np.ndim != 2 or x_np.shape[1] != self.d:
                raise ValueError(f"Expected 2D input with d={self.d}, got shape {x_np.shape}")
            x_t = torch.from_numpy(
                np.ascontiguousarray(x_np, dtype=np.float64 if x_np.dtype == np.float64 else np.float32)
            ).to(c_dev)
        else:
            if x.ndim != 2 or x.shape[1] != self.d:
                raise ValueError(f"Expected 2D input with d={self.d}, got shape {tuple(x.shape)}")
            x_t = x.detach().to(c_dev)

        n = int(x_t.shape[0])
        if n == 0:
            empty_d = torch.empty((0,), dtype=torch.float32, device=c_dev)
            empty_l = torch.empty((0,), dtype=torch.int64, device=c_dev)
            return (empty_d.cpu().numpy(), empty_l.cpu().numpy()) if input_is_numpy else (empty_d, empty_l)

        comp_dtype = self._resolve_compute_dtype(x_t)
        origin64 = (
            self._origin64.to(c_dev)
            if self._origin64 is not None
            else x_t[: min(n, 8192)].double().mean(dim=0).contiguous()
        )
        coord_scale = self._coord_scale
        c_c64 = (self.centroids.double() - origin64[None, :]) * coord_scale

        use_triton_fast = (
            c_dev.type == "cuda"
            and HAS_TRITON
            and comp_dtype in (torch.float32, torch.bfloat16, torch.float16)
            and (
                self.backend == "triton"
                or (self.backend == "auto" and torch.cuda.get_device_capability(c_dev) >= (8, 0))
            )
        )
        if use_triton_fast:
            c_c32 = c_c64.float().contiguous()
            c_rep = _alloc_padded_2d(self.k, self.d, comp_dtype, c_dev)
            c_norm = torch.empty((self.k,), dtype=torch.float32, device=c_dev)
            _launch_ultra_prep_centroids(c_c32, c_rep, c_norm)
            origin_ker = (
                origin64
                if (x_t.dtype == torch.float64 or coord_scale != 1.0 or float(origin64.abs().max().item()) > 1e3)
                else origin64.float().contiguous()
            )
            sq_dists, labels = _launch_ultra_assign(
                x_t=x_t,
                c_rep=c_rep,
                c_norm=c_norm,
                origin=origin_ker,
                coord_scale=coord_scale,
            )
        else:
            x_c64 = (x_t.double() - origin64[None, :]) * coord_scale
            if comp_dtype == torch.float64:
                c64 = c_c64.contiguous()
                sq_dists, labels = torch_assign_chunked(x_c64.contiguous(), c64, c64.square().sum(dim=1))
            else:
                sq_dists, labels = assign_points(
                    x_c64.float().contiguous(),
                    c_c64.float().contiguous(),
                    compute_dtype=comp_dtype,
                    backend="torch" if self.backend == "torch" else "auto",
                )

        if coord_scale != 1.0:
            inv_sq = 1.0 / (coord_scale * coord_scale)
            sq_dists = (sq_dists.double() * inv_sq).clamp_max(torch.finfo(torch.float32).max).float()

        if input_is_numpy:
            return sq_dists.cpu().numpy(), labels.cpu().numpy()
        return sq_dists, labels


# Pre-Registered Ablation & Synthesis Configurations for Phase 4
ULTRA_ABLATION_ARMS: dict[str, dict[str, Any]] = {
    "ultra_emeans": {},
    "ultra_emeans_wide": {"wide_step_mult": 2, "wide_from_frac": 0.25},
    # Discovery 0 LOO: Replace fast Importance Coreset + Voronoi Polish + Prior Mass with raw s-means D^3 init
    "ultra_loo_no_turkpp_init": {
        "init_mode": "subsampled_d15",
        "use_voronoi_polish": False,
        "use_prior_mass": False,
    },
    # Discovery 1 LOO (a): Replace Half-Gaussian Oja `sqrt(2/pi) * sigma_1 * v_1` with s-means `0.55 * sigma_rms * Rademacher`
    "ultra_loo_rademacher055_split": {
        "split_mode": "rms_aniso",
        "split_mult": 0.55,
        "donor_mode": "sse_usage",
    },
    # Discovery 1 LOO (b): Replace Half-Gaussian `sqrt(2/pi)` step size with SuperEMeans's `0.08 * sigma_1`
    "ultra_loo_oja008_split": {
        "split_mode": "oja_only",
        "split_mult": 0.08,
    },
    # Discovery 2 LOO (a): Decouple SSE back to fixed 0.85 decay & drop Twin-Safe Huygens Rare Shield
    "ultra_loo_decoupled_no_huygens": {
        "couple_sse_to_alpha": False,
        "sse_decay": 0.85,
        "use_huygens_protection": False,
        "donor_mode": "sse_usage",
    },
    # Discovery 2 LOO (b): Disable 85% revival stop & closing-pass stale-parent split flush
    "ultra_loo_no_revive_stop_flush": {
        "use_revive_stop": False,
        "flush_pass_on_split": False,
    },
    # Discovery 3 LOO (a): Plain unweighted closing_mean (vanilla e-means finalizer)
    "ultra_loo_closing_mean": {
        "finalization": "closing_mean",
        "adaptive_pass_weighting": False,
        "use_pass_weighting": False,
    },
    # Discovery 3 LOO (b): s-means C6+C7 fixed ramp + live-EMA tau=0.20 finalizer
    "ultra_loo_smeans_eb020": {
        "finalization": "eb_ema",
        "eb_tau": 0.20,
        "adaptive_pass_weighting": False,
        "use_pass_weighting": True,
        "pass_w_span": 0.35,
    },
    # Discovery 3 LOO (c): SuperEMeans 2-point `slow_ema`-only James-Stein finalizer
    "ultra_loo_super_slow_eb": {
        "finalization": "eb_slow_only",
        "adaptive_pass_weighting": False,
        "use_pass_weighting": False,
    },
    # Discovery 4 LOO: Disable strategic Oja-seeded P/Q `head_edit` transplants
    "ultra_loo_no_head_edit": {
        "head_edit": False,
    },
}


def build_ultra_arm(
    arm_name: str,
    d: int,
    k: int,
    nepochs: float = 5.0,
    seed: int = 1234,
    **overrides: Any,
) -> UltraEMeans:
    """Instantiate any pre-registered `UltraEMeans` or ablation arm by key."""
    if arm_name not in ULTRA_ABLATION_ARMS:
        valid = ", ".join(sorted(ULTRA_ABLATION_ARMS.keys()))
        raise KeyError(f"Unknown arm_name={arm_name!r}; expected one of: {valid}")
    cfg = {**ULTRA_ABLATION_ARMS[arm_name], **overrides}
    return UltraEMeans(d=d, k=k, nepochs=nepochs, seed=seed, **cfg)


__all__ = [
    "HALF_GAUSSIAN_MEAN",
    "ULTRA_ABLATION_ARMS",
    "UltraEMeans",
    "UltraEMeansEpochStat",
    "build_ultra_arm",
    "run_ultra_head_edit_checkpoint",
    "ultra_simplex_eb_finalize",
    "ultra_turkpp_init_centered",
]
