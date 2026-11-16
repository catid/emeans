"""Flagship Scale-Adaptive Clustering Engine: `ApexEMeans`.

Unifies and dominates across 4 orders of magnitude (`N = 10^4 -> 10^7+`, `K = 256 -> 65,536`,
`D = 1 -> 1,024+`) across continuous deep embeddings, unit-sphere NLP vectors, heavy-tailed
multimodal distributions, and raw multi-scale tabular sensor data by optimizing every layer:

  1. **Batch (`Two-Gear Nimble -> Wide Schedule + One-Pass Pre-Permuted Coalesced Stream`)**:
     - Single fused Triton pass permutes (`perm0`), origin-centers in FP64, and pre-encodes
       both sequential FP32 scatter rows and FP8/BF16 Tensor-Core query representations with
       precomputed `||x_rep||_2^2` norms (eliminating indirect HBM gather stalls & per-epoch
       norm loops while preserving 100% intra-batch independence across epochs via block-ID
       shuffling of the already-uniform `perm0` stream).
     - Coupled Two-Gear batch widening (`B -> 2B` at `prog >= 0.25` when `total_steps >= 24`,
       scaled via `alpha_0 ~ B^0.9003`) gives nimble early centroid progress AND high-TFLOP/s
       Tensor-Core saturation.

  2. **Fuse & Graph (`True End-to-End Multi-Step CUDAGraph Super-Step Replay`)**:
     - Stores `(step_idx, rows_processed, ep_b_start, usage_bias, sse_bias, alpha_last)` and
       `batch_offset` on GPU in a race-free 3-slot ping-pong ring (`0 -> 1 -> 0 -> 1 -> 2 -> 0`).
     - Computes `alpha(t)` (`emeans_cosine` / `power_cosine` / `constant`), scalar EMA/SSE bias
       correction, closing-pass ramp weights, and step-indexed Philox Gumbel/Rademacher RNG
       **100% on-device inside fused Triton kernels** — ZERO Python scalar updates per step!
     - Captures an entire 5-step super-step (4 fast steps with fused centroid-prep + 1 step with
       on-device SRAM Huygens rare-mode shield + spectral-gated Half-Gaussian Oja split) inside
       a single `torch.cuda.CUDAGraph`, replaying whole epochs in `ceil(steps / 5)` graph launches,
       and switches automatically to a lean Oja-free graph after the `85%` revival freeze!

  3. **Quant (`CroissAttn`-Stabilized Decoupled Per-Row/Per-Centroid FP8 E4M3 + Auto BF16)**:
     - Wires directly into `src/emeans/croiss_fp8.py` (`catid/CroissAttn` saved-payload score
       replay, `FP32_TINY` contraction-axis scale floor, `[-448, 448]` clamp + `rtne` zero-`0x7f`
       NaN guard, `max_num_imprecise_acc=0` FP32 Tensor-Core accumulation, and exact represented
       Euclidean identity `x_norm_i + c_norm_j - 2 * sx_i * dot_fp32(qx_i, qc_j) * sc_j >= 0`),
       while keeping master centroids, sums, SSE, Oja axes, and closing-pass means in exact FP32!

  4. **4-Gear Automatic Scale Adaptation & Algorithmic Precision**:
     - **Gear 1 (`small_bf16_graph`)**: Skips FP8 pre-encode when unamortized; runs BF16 `FULL_D`
       / Split-K + 5-step `CUDAGraph` + global `BLOCK_P <= 1024` tournament `k-means++`.
     - **Gear 2 (`medium_croiss_fp8_graph`)**: `CroissAttn` Decoupled FP8 + Two-Gear Wide-Batch
       + 5-step `CUDAGraph` + fused FP8 centroid prep.
     - **Gear 3 (`huge_k_ivf_fp8_graph`)**: Zero-sync Coarse-to-Fine GPU IVF on exploration
       passes (`K >= ivf_k_threshold`) followed by an Exact `CroissAttn`-FP8 Closing Pass.
     - **Gear 4 (`multi_gpu_nccl`)**: Multi-GPU data-sharded `CroissAttn`-FP8 + coalesced NCCL
       `all_reduce` + replica-identical Huygens/Oja/Simplex-EB.
     - **Pre-Closing Independent Anchor + Positive-Part James-Stein Simplex EB**: Snapshots
       `pre_close_centroids` at closing-pass entry (zero sample-reuse covariance AND zero step-0
       lag bias) for unbiased positive-part Stein shrinkage across all `N/K` and `D`.
"""

from __future__ import annotations

import contextlib
import importlib
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .algo_extensions import ShrinkageDiagnostics, reset_post_split_oja_axes, separate_collapsed_directional
from .emeans import _separate_collapsed_splits
from .fast_ops import FLOAT8_E4M3FN, HAS_TRITON, _choose_split_k_partitions, block_permute_indices
from .kernels import assign_points, torch_assign_chunked
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
    round_power_of_two,
)
from .seeding import SeedingResult
from .super_emeans import fast_top3_centered_assign, incremental_refresh_top3_and_nn, super_scheduled_alpha
from .ultra_emeans import HALF_GAUSSIAN_MEAN

if HAS_TRITON:
    import triton
    import triton.language as tl

FP8_MAX_FINITE: float = 448.0
FP32_TINY: float = 1.1754943508222875e-38

_CROISS_MOD_CACHE: Any = None


def _get_croiss_fp8_module() -> Any:
    """Dynamically load `emeans.croiss_fp8` once available on disk."""
    global _CROISS_MOD_CACHE
    if _CROISS_MOD_CACHE is not None:
        return _CROISS_MOD_CACHE
    try:
        mod = importlib.import_module(".croiss_fp8", package=__package__)
        _CROISS_MOD_CACHE = mod
        return mod
    except Exception:
        return None


def _has_fp8_tensor_cores(device: torch.device) -> bool:
    if device.type != "cuda" or not HAS_TRITON or FLOAT8_E4M3FN is None:
        return False
    cap = torch.cuda.get_device_capability(device)
    return cap >= (8, 9) and hasattr(tl, "float8e4nv")


# ==============================================================================
# 1. Fused Zero-Sync & CUDAGraph-Native Blackwell Triton Kernels
# ==============================================================================

if HAS_TRITON:

    @triton.jit(do_not_specialize=["P_stride", "P_per_g", "num_select_per_g", "last_g_P", "last_g_sel", "G", "rng_seed"])
    def _apex_batched_local_greedy_kpp_kernel(
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
        """Multi-CTA Voronoi-neighborhood parallel Greedy k-means++ tournament."""
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
    def _apex_local_greedy_kpp_kernel(
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
        """Global single-CTA weighted Greedy k-means++ candidate tournament (`BLOCK_P <= 1024`)."""
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
    def _apex_origin_d2_kernel(
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

    @triton.jit(do_not_specialize=["M", "D", "stride_xm", "stride_ck"])
    def _apex_voronoi_polish_scatter_kernel(
        xs_ptr,
        w_ptr,
        labels_ptr,
        sq_d0_ptr,
        mass_ptr,
        dist_m0_ptr,
        sums_ptr,
        M,
        D,
        stride_xm,
        stride_ck,
        DO_SUMS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Fused 1-pass weighted Voronoi polish scatter (`mass`, `w * sq_d0`, `w * xs_c`)."""
        pid_m = tl.program_id(0).to(tl.int64)
        pid_d = tl.program_id(1).to(tl.int64)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
        mask_m = offs_m < M
        lbl = tl.load(labels_ptr + offs_m, mask=mask_m, other=0).to(tl.int64)
        w = tl.load(w_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)

        if pid_d == 0:
            sq_d0 = tl.load(sq_d0_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
            tl.atomic_add(mass_ptr + lbl, w, mask=mask_m, sem="relaxed")
            tl.atomic_add(dist_m0_ptr + lbl, w * sq_d0, mask=mask_m, sem="relaxed")

        if DO_SUMS:
            cols = pid_d * BLOCK_D + tl.arange(0, BLOCK_D).to(tl.int64)
            mask_d = cols < D
            mask_md = mask_m[:, None] & mask_d[None, :]
            x_val = tl.load(
                xs_ptr + offs_m[:, None] * stride_xm + cols[None, :],
                mask=mask_md,
                other=0.0,
            ).to(tl.float32)
            tl.atomic_add(
                sums_ptr + lbl[:, None] * stride_ck + cols[None, :],
                x_val * w[:, None],
                mask=mask_md,
                sem="relaxed",
            )

    @triton.jit(do_not_specialize=["N", "D", "stride_xn", "stride_xd", "stride_f32", "stride_rep"])
    def _apex_prepermute_encode_kernel(
        x_ptr,
        perm_ptr,
        origin_ptr,
        out_f32_ptr,
        out_rep_ptr,
        out_sx_ptr,
        out_xnorm_ptr,
        N,
        D,
        stride_xn,
        stride_xd,
        stride_f32,
        stride_rep,
        coord_scale,
        HAS_PERM: tl.constexpr,
        ORIGIN_FP64: tl.constexpr,
        HAS_SCALE: tl.constexpr,
        WRITE_F32: tl.constexpr,
        REP_MODE: tl.constexpr,  # 0=FP32, 1=BF16, 2=FP16, 3=CROISS_FP8_DECOUPLED
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """One-pass fused permutation + FP64 origin-centering + FP32 stream + CroissAttn-FP8/BF16 encode."""
        pid_n = tl.program_id(0).to(tl.int64)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
        mask_n = offs_n < N
        src_rows = (
            tl.load(perm_ptr + offs_n, mask=mask_n, other=0).to(tl.int64)
            if HAS_PERM
            else offs_n
        )

        if REP_MODE == 3:
            amax_row = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for d0 in range(0, D, BLOCK_D):
                cols = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
                mask_d = cols < D
                mask_nd = mask_n[:, None] & mask_d[None, :]
                x_raw = tl.load(
                    x_ptr + src_rows[:, None] * stride_xn + cols[None, :] * stride_xd,
                    mask=mask_nd,
                    other=0.0,
                )
                if ORIGIN_FP64:
                    orig64 = tl.load(origin_ptr + cols, mask=mask_d, other=0.0).to(tl.float64)
                    xdiff = x_raw.to(tl.float64) - orig64[None, :]
                    x_f = (xdiff * coord_scale).to(tl.float32) if HAS_SCALE else xdiff.to(tl.float32)
                else:
                    orig32 = tl.load(origin_ptr + cols, mask=mask_d, other=0.0).to(tl.float32)
                    x_f = x_raw.to(tl.float32) - orig32[None, :]
                    if HAS_SCALE:
                        x_f = x_f * coord_scale
                x_f = tl.where(mask_nd, x_f, 0.0)
                if WRITE_F32:
                    tl.store(out_f32_ptr + offs_n[:, None] * stride_f32 + cols[None, :], x_f, mask=mask_nd)
                amax_row = tl.maximum(amax_row, tl.max(tl.abs(x_f), axis=1))

            sx = tl.maximum(amax_row * (1.00000024 / 448.0), 1.1754943508222875e-38)
            inv_sx = 1.0 / sx
            q_nsq = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for d0 in range(0, stride_rep, BLOCK_D):
                cols = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
                mask_d = cols < D
                mask_pad = cols < stride_rep
                mask_nd = mask_n[:, None] & mask_d[None, :]
                if WRITE_F32:
                    x_f = tl.load(
                        out_f32_ptr + offs_n[:, None] * stride_f32 + cols[None, :],
                        mask=mask_nd,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    x_raw = tl.load(
                        x_ptr + src_rows[:, None] * stride_xn + cols[None, :] * stride_xd,
                        mask=mask_nd,
                        other=0.0,
                    )
                    if ORIGIN_FP64:
                        orig64 = tl.load(origin_ptr + cols, mask=mask_d, other=0.0).to(tl.float64)
                        xdiff = x_raw.to(tl.float64) - orig64[None, :]
                        x_f = (xdiff * coord_scale).to(tl.float32) if HAS_SCALE else xdiff.to(tl.float32)
                    else:
                        orig32 = tl.load(origin_ptr + cols, mask=mask_d, other=0.0).to(tl.float32)
                        x_f = x_raw.to(tl.float32) - orig32[None, :]
                        if HAS_SCALE:
                            x_f = x_f * coord_scale
                    x_f = tl.where(mask_nd, x_f, 0.0)

                scaled = tl.minimum(tl.maximum(x_f * inv_sx[:, None], -448.0), 448.0)
                scaled = tl.where(mask_nd, scaled, 0.0)
                qx = scaled.to(tl.float8e4nv, fp_downcast_rounding="rtne")
                tl.store(
                    out_rep_ptr + offs_n[:, None] * stride_rep + cols[None, :],
                    qx,
                    mask=mask_n[:, None] & mask_pad[None, :],
                )
                qxf = qx.to(tl.float32)
                q_nsq += tl.sum(tl.where(mask_nd, qxf * qxf, 0.0), axis=1)

            tl.store(out_sx_ptr + offs_n, sx, mask=mask_n)
            tl.store(out_xnorm_ptr + offs_n, (sx * sx) * q_nsq, mask=mask_n)
        else:
            x_nsq = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for d0 in range(0, stride_rep, BLOCK_D):
                cols = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
                mask_d = cols < D
                mask_pad = cols < stride_rep
                mask_nd = mask_n[:, None] & mask_d[None, :]
                x_raw = tl.load(
                    x_ptr + src_rows[:, None] * stride_xn + cols[None, :] * stride_xd,
                    mask=mask_nd,
                    other=0.0,
                )
                if ORIGIN_FP64:
                    orig64 = tl.load(origin_ptr + cols, mask=mask_d, other=0.0).to(tl.float64)
                    xdiff = x_raw.to(tl.float64) - orig64[None, :]
                    x_f = (xdiff * coord_scale).to(tl.float32) if HAS_SCALE else xdiff.to(tl.float32)
                else:
                    orig32 = tl.load(origin_ptr + cols, mask=mask_d, other=0.0).to(tl.float32)
                    x_f = x_raw.to(tl.float32) - orig32[None, :]
                    if HAS_SCALE:
                        x_f = x_f * coord_scale
                x_f = tl.where(mask_nd, x_f, 0.0)
                if WRITE_F32:
                    tl.store(out_f32_ptr + offs_n[:, None] * stride_f32 + cols[None, :], x_f, mask=mask_nd)
                if REP_MODE == 1:
                    xr = x_f.to(tl.bfloat16)
                    xrf = xr.to(tl.float32)
                elif REP_MODE == 2:
                    xr = x_f.to(tl.float16)
                    xrf = xr.to(tl.float32)
                else:
                    xr = x_f
                    xrf = x_f
                tl.store(
                    out_rep_ptr + offs_n[:, None] * stride_rep + cols[None, :],
                    xr,
                    mask=mask_n[:, None] & mask_pad[None, :],
                )
                x_nsq += tl.sum(tl.where(mask_nd, xrf * xrf, 0.0), axis=1)
            tl.store(out_xnorm_ptr + offs_n, x_nsq, mask=mask_n)

    @triton.jit(do_not_specialize=["B", "K", "D", "N_total", "stride_xb", "stride_xd", "stride_ck"])
    def _apex_assign_kernel(
        x_ptr,
        sx_ptr,
        x_norm_ptr,
        row_idx_ptr,
        batch_offset_ptr,
        origin_ptr,
        c_ptr,
        sc_ptr,
        c_norm_ptr,
        dist_ptr,
        label_ptr,
        B,
        K,
        D,
        N_total,
        stride_xb,
        stride_xd,
        stride_ck,
        coord_scale,
        BLOCK_B: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
        FULL_D: tl.constexpr,
        DTYPE_MODE: tl.constexpr,  # 0=FP32, 1=BF16, 2=FP16, 3=CROISS_FP8_DECOUPLED
        PRECOMP_XNORM: tl.constexpr,
        HAS_INDICES: tl.constexpr,
        HAS_BATCH_OFFSET: tl.constexpr,
        WRAP_ROWS: tl.constexpr,
        HAS_ORIGIN: tl.constexpr,
        HAS_SCALE: tl.constexpr,
        ORIGIN_FP64: tl.constexpr,
    ):
        """CUDAGraph-native Tensor-Core assignment (FP32 / BF16 / FP16 / CroissAttn-FP8)."""
        pid_b = tl.program_id(0).to(tl.int64)
        offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B).to(tl.int64)
        mask_b = offs_b < B
        base_off = tl.load(batch_offset_ptr).to(tl.int64) if HAS_BATCH_OFFSET else tl.zeros((), dtype=tl.int64)
        logical_pos = base_off + offs_b
        if WRAP_ROWS:
            logical_pos = logical_pos % N_total

        src_rows = (
            tl.load(row_idx_ptr + logical_pos, mask=mask_b, other=0).to(tl.int64)
            if HAS_INDICES
            else logical_pos
        )

        sx_b = (
            tl.load(sx_ptr + src_rows, mask=mask_b, other=1.0).to(tl.float32)
            if DTYPE_MODE == 3
            else tl.full((BLOCK_B,), 1.0, dtype=tl.float32)
        )

        if PRECOMP_XNORM:
            x_norm = tl.load(x_norm_ptr + src_rows, mask=mask_b, other=0.0).to(tl.float32)
        else:
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
            if DTYPE_MODE == 3:
                x_full = x_raw
                if not PRECOMP_XNORM:
                    xf_rep = x_full.to(tl.float32)
                    x_norm = (sx_b * sx_b) * tl.sum(xf_rep * xf_rep, axis=1)
            else:
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
                    x_f = x_raw.to(tl.float32) if DTYPE_MODE == 0 else x_raw
                    if HAS_SCALE:
                        x_f = tl.where(mask_bd, x_f.to(tl.float32) * coord_scale, 0.0)

                if DTYPE_MODE == 1:
                    x_full = x_f.to(tl.bfloat16)
                elif DTYPE_MODE == 2:
                    x_full = x_f.to(tl.float16)
                else:
                    x_full = x_f.to(tl.float32)
                if not PRECOMP_XNORM:
                    xf_rep = x_full.to(tl.float32)
                    x_norm = tl.sum(xf_rep * xf_rep, axis=1)
        elif not PRECOMP_XNORM:
            for d0 in range(0, D, BLOCK_D):
                cols = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
                mask_d = cols < D
                mask_bd = mask_b[:, None] & mask_d[None, :]
                x_raw = tl.load(
                    x_ptr + src_rows[:, None] * stride_xb + cols[None, :] * stride_xd,
                    mask=mask_bd,
                    other=0.0,
                )
                if DTYPE_MODE == 3:
                    xf_rep = x_raw.to(tl.float32)
                    x_norm += (sx_b * sx_b) * tl.sum(xf_rep * xf_rep, axis=1)
                else:
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
                if DTYPE_MODE == 3:
                    acc = tl.dot(x_full, tl.trans(c_tile), max_num_imprecise_acc=0, out_dtype=tl.float32)
                elif DTYPE_MODE == 0:
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
                    c_tile = tl.load(
                        c_ptr + offs_k[:, None] * stride_ck + cols[None, :],
                        mask=mask_k[:, None] & mask_d[None, :],
                        other=0.0,
                    )
                    if DTYPE_MODE == 3:
                        acc = tl.dot(x_raw, tl.trans(c_tile), acc=acc, max_num_imprecise_acc=0, out_dtype=tl.float32)
                    else:
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
                            x_f = x_raw if ((DTYPE_MODE == 1) or (DTYPE_MODE == 2)) else x_raw.to(tl.float32)
                            if HAS_SCALE:
                                x_f = tl.where(mask_bd, x_f.to(tl.float32) * coord_scale, 0.0)

                        if DTYPE_MODE == 1:
                            acc = tl.dot(x_f.to(tl.bfloat16), tl.trans(c_tile), acc=acc, out_dtype=tl.float32)
                        elif DTYPE_MODE == 2:
                            acc = tl.dot(x_f.to(tl.float16), tl.trans(c_tile), acc=acc, out_dtype=tl.float32)
                        else:
                            acc = tl.dot(x_f.to(tl.float32), tl.trans(c_tile), acc=acc, input_precision="ieee", out_dtype=tl.float32)

            c_norm = tl.load(c_norm_ptr + offs_k, mask=mask_k, other=float("inf"))
            if DTYPE_MODE == 3:
                sc_k = tl.load(sc_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
                dots_scaled = (sx_b[:, None] * acc) * sc_k[None, :]
                scores = tl.where(mask_k[None, :], c_norm[None, :] - 2.0 * dots_scaled, float("inf"))
            else:
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

    @triton.jit(do_not_specialize=["B", "K", "D", "N_total", "stride_xb", "stride_xd", "stride_ck"])
    def _apex_splitk_assign_kernel(
        x_ptr,
        sx_ptr,
        x_norm_ptr,
        row_idx_ptr,
        batch_offset_ptr,
        origin_ptr,
        c_ptr,
        sc_ptr,
        c_norm_ptr,
        part_score_ptr,
        part_label_ptr,
        part_xnorm_ptr,
        B,
        K,
        D,
        N_total,
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
        PRECOMP_XNORM: tl.constexpr,
        HAS_INDICES: tl.constexpr,
        HAS_BATCH_OFFSET: tl.constexpr,
        WRAP_ROWS: tl.constexpr,
        HAS_ORIGIN: tl.constexpr,
        HAS_SCALE: tl.constexpr,
        ORIGIN_FP64: tl.constexpr,
    ):
        """2D Split-K Tensor-Core assignment kernel (FP32 / BF16 / FP16 / CroissAttn-FP8)."""
        pid_b = tl.program_id(0).to(tl.int64)
        part = tl.program_id(1)
        offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B).to(tl.int64)
        mask_b = offs_b < B
        base_off = tl.load(batch_offset_ptr).to(tl.int64) if HAS_BATCH_OFFSET else tl.zeros((), dtype=tl.int64)
        logical_pos = base_off + offs_b
        if WRAP_ROWS:
            logical_pos = logical_pos % N_total
        src_rows = (
            tl.load(row_idx_ptr + logical_pos, mask=mask_b, other=0).to(tl.int64)
            if HAS_INDICES
            else logical_pos
        )

        part_size = (K + PARTITIONS - 1) // PARTITIONS
        k_start = part * part_size
        k_end = tl.minimum(k_start + part_size, K)

        sx_b = (
            tl.load(sx_ptr + src_rows, mask=mask_b, other=1.0).to(tl.float32)
            if DTYPE_MODE == 3
            else tl.full((BLOCK_B,), 1.0, dtype=tl.float32)
        )

        x_norm = tl.zeros((BLOCK_B,), dtype=tl.float32)
        if PRECOMP_XNORM and part == 0:
            x_norm = tl.load(x_norm_ptr + src_rows, mask=mask_b, other=0.0).to(tl.float32)

        if FULL_D:
            cols = tl.arange(0, BLOCK_D).to(tl.int64)
            mask_d = cols < D
            mask_bd = mask_b[:, None] & mask_d[None, :]
            x_raw = tl.load(
                x_ptr + src_rows[:, None] * stride_xb + cols[None, :] * stride_xd,
                mask=mask_bd,
                other=0.0,
            )
            if DTYPE_MODE == 3:
                x_full = x_raw
                if (not PRECOMP_XNORM) and part == 0:
                    xf_rep = x_full.to(tl.float32)
                    x_norm = (sx_b * sx_b) * tl.sum(xf_rep * xf_rep, axis=1)
            else:
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
                    x_f = x_raw.to(tl.float32) if DTYPE_MODE == 0 else x_raw
                    if HAS_SCALE:
                        x_f = tl.where(mask_bd, x_f.to(tl.float32) * coord_scale, 0.0)
                if DTYPE_MODE == 1:
                    x_full = x_f.to(tl.bfloat16)
                elif DTYPE_MODE == 2:
                    x_full = x_f.to(tl.float16)
                else:
                    x_full = x_f.to(tl.float32)
                if (not PRECOMP_XNORM) and part == 0:
                    xf_rep = x_full.to(tl.float32)
                    x_norm = tl.sum(xf_rep * xf_rep, axis=1)
        elif (not PRECOMP_XNORM) and part == 0:
            for d0 in range(0, D, BLOCK_D):
                cols = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
                mask_d = cols < D
                mask_bd = mask_b[:, None] & mask_d[None, :]
                x_raw = tl.load(
                    x_ptr + src_rows[:, None] * stride_xb + cols[None, :] * stride_xd,
                    mask=mask_bd,
                    other=0.0,
                )
                if DTYPE_MODE == 3:
                    xf_rep = x_raw.to(tl.float32)
                    x_norm += (sx_b * sx_b) * tl.sum(xf_rep * xf_rep, axis=1)
                else:
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
                if DTYPE_MODE == 3:
                    acc = tl.dot(x_full, tl.trans(c_tile), max_num_imprecise_acc=0, out_dtype=tl.float32)
                elif DTYPE_MODE == 0:
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
                    c_tile = tl.load(
                        c_ptr + offs_k[:, None] * stride_ck + cols[None, :],
                        mask=mask_k[:, None] & mask_d[None, :],
                        other=0.0,
                    )
                    if DTYPE_MODE == 3:
                        acc = tl.dot(x_raw, tl.trans(c_tile), acc=acc, max_num_imprecise_acc=0, out_dtype=tl.float32)
                    else:
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
                            x_f = x_raw if ((DTYPE_MODE == 1) or (DTYPE_MODE == 2)) else x_raw.to(tl.float32)
                            if HAS_SCALE:
                                x_f = tl.where(mask_bd, x_f.to(tl.float32) * coord_scale, 0.0)
                        if DTYPE_MODE == 1:
                            acc = tl.dot(x_f.to(tl.bfloat16), tl.trans(c_tile), acc=acc, out_dtype=tl.float32)
                        elif DTYPE_MODE == 2:
                            acc = tl.dot(x_f.to(tl.float16), tl.trans(c_tile), acc=acc, out_dtype=tl.float32)
                        else:
                            acc = tl.dot(x_f.to(tl.float32), tl.trans(c_tile), acc=acc, input_precision="ieee", out_dtype=tl.float32)

            c_norm = tl.load(c_norm_ptr + offs_k, mask=mask_k, other=float("inf"))
            if DTYPE_MODE == 3:
                sc_k = tl.load(sc_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
                dots_scaled = (sx_b[:, None] * acc) * sc_k[None, :]
                scores = tl.where(mask_k[None, :], c_norm[None, :] - 2.0 * dots_scaled, float("inf"))
            else:
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
    def _apex_splitk_reduce_kernel(
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

    @triton.jit(do_not_specialize=["B", "D", "N_total", "ep_idx", "stride_xb", "stride_xd", "stride_ck"])
    def _apex_scatter_oja_sse_kernel(
        x_ptr,
        row_idx_ptr,
        batch_offset_ptr,
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
        N_total,
        stride_xb,
        stride_xd,
        stride_ck,
        coord_scale,
        HAS_INDICES: tl.constexpr,
        HAS_BATCH_OFFSET: tl.constexpr,
        WRAP_ROWS: tl.constexpr,
        HAS_ORIGIN: tl.constexpr,
        HAS_SCALE: tl.constexpr,
        ORIGIN_FP64: tl.constexpr,
        TRACK_OJA: tl.constexpr,
        FULL_D: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """CUDAGraph-native fused scatter (`sums`, `counts`, `batch_sse`, `ep_sq_dists`, register Oja)."""
        pid_b = tl.program_id(0).to(tl.int64)
        pid_d = tl.program_id(1).to(tl.int64)

        offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B).to(tl.int64)
        mask_b = offs_b < B
        lbl = tl.load(labels_ptr + offs_b, mask=mask_b, other=0).to(tl.int64)
        base_off = tl.load(batch_offset_ptr).to(tl.int64) if HAS_BATCH_OFFSET else tl.zeros((), dtype=tl.int64)
        logical_pos = base_off + offs_b
        if WRAP_ROWS:
            logical_pos = logical_pos % N_total
        src_rows = (
            tl.load(row_idx_ptr + logical_pos, mask=mask_b, other=0).to(tl.int64)
            if HAS_INDICES
            else logical_pos
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

                r_val = tl.where(mask_bd, tl.minimum(tl.maximum(x_val - c_val, -1e17), 1e17), 0.0)
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
                    x_val = tl.where(mask_bd, x_val * coord_scale, 0.0)
            tl.atomic_add(
                sums_ptr + lbl[:, None] * stride_ck + offs_d[None, :],
                x_val,
                mask=mask_bd,
                sem="relaxed",
            )

    @triton.jit(do_not_specialize=["K", "D", "stride_sd"])
    def _apex_feat_scale_kernel(
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
        do_not_specialize=[
            "K",
            "D",
            "b_cur",
            "N_dataset",
            "rows_in_ep",
            "total_steps",
            "total_planned_rows",
            "accum_pass",
            "prep_rep",
            "stride_cd",
            "stride_crepk",
        ]
    )
    def _apex_ema_oja_slow_pass_prep_kernel(
        state_in_ptr,
        state_out_ptr,
        batch_off_in_ptr,
        batch_off_out_ptr,
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
        sc_ptr,
        c_norm_ptr,
        K,
        D,
        stride_cd,
        stride_crepk,
        b_cur,
        N_dataset,
        rows_in_ep,
        total_steps,
        total_planned_rows,
        eff_peak,
        alpha_min_factor,
        alpha_floor,
        cosine_power,
        fixed_sse_decay,
        alpha_slow,
        eff_pass_span,
        accum_pass,
        prep_rep,
        SCHED_MODE: tl.constexpr,       # 0=emeans_cosine, 1=power_cosine, 2=constant
        USE_WIDE_SCHED: tl.constexpr,   # 1=rows_processed/total_planned_rows, 0=step/total_steps
        COUPLE_SSE: tl.constexpr,
        TRACK_OJA: tl.constexpr,
        TRACK_SURV: tl.constexpr,
        TRACK_SLOW: tl.constexpr,
        REP_MODE: tl.constexpr,         # 0=FP32, 1=BF16, 2=FP16, 3=CROISS_FP8_DECOUPLED
        BLOCK_D: tl.constexpr,
    ):
        """Fused On-Device Schedule + EMA + Oja + Surv + SlowEMA + Closing + Centroid-Prep (incl. FP8)."""
        j = tl.program_id(0).to(tl.int64)
        if j >= K:
            return

        # 1. Compute exact schedule & step weights 100% on-device from ping-pong `state_in_ptr`
        step_f = tl.load(state_in_ptr + 0)
        rows_proc_f = tl.load(state_in_ptr + 1)
        ep_b_start_f = tl.load(state_in_ptr + 2)
        u_bias_in = tl.load(state_in_ptr + 3)
        s_bias_in = tl.load(state_in_ptr + 4)

        if USE_WIDE_SCHED:
            raw_frac = (rows_proc_f / tl.maximum(total_planned_rows.to(tl.float64), 1.0)).to(tl.float32)
        else:
            raw_frac = (tl.minimum(step_f, total_steps.to(tl.float64)) / tl.maximum(total_steps.to(tl.float64), 1.0)).to(tl.float32)
        frac = tl.minimum(tl.maximum(raw_frac, 0.0), 1.0)

        if SCHED_MODE == 2:
            alpha = eff_peak
        else:
            cos_half = 0.5 * (1.0 + tl.cos(3.141592653589793 * frac))
            if SCHED_MODE == 1:
                cos_pow = tl.where(
                    cos_half > 1e-12,
                    tl.exp(cosine_power * tl.log(tl.maximum(cos_half, 1e-30))),
                    0.0,
                )
                alpha = eff_peak * (alpha_floor + (1.0 - alpha_floor) * cos_pow)
            else:
                alpha = eff_peak * (alpha_min_factor + (1.0 - alpha_min_factor) * cos_half)

        decay = 1.0 - alpha
        sse_decay = decay if COUPLE_SSE else fixed_sse_decay
        is_rep = alpha >= 1.0
        alpha_surv = tl.minimum(alpha, (2.0 * b_cur.to(tl.float32)) / tl.maximum(N_dataset.to(tl.float32), 1.0))

        do_accum = accum_pass != 0
        do_prep = prep_rep != 0
        if do_accum:
            if eff_pass_span > 0.0:
                pfrac = tl.minimum(
                    1.0,
                    ((ep_b_start_f + 0.5 * b_cur.to(tl.float64)) / tl.maximum(rows_in_ep.to(tl.float64), 1.0)).to(tl.float32),
                )
                w_pass = (1.0 - eff_pass_span) + 2.0 * eff_pass_span * pfrac
            else:
                w_pass = 1.0
        else:
            w_pass = 0.0

        # CTA 0 updates ping-pong `state_out_ptr != state_in_ptr` (100% race-free across CTAs!)
        if j == 0:
            a64 = alpha.to(tl.float64)
            d64 = decay.to(tl.float64)
            sd64 = sse_decay.to(tl.float64)
            u_bias_out = tl.where(is_rep, 1.0, d64 * u_bias_in + a64)
            s_bias_out = tl.where(is_rep, 1.0, sd64 * s_bias_in + (1.0 - sd64))
            b_f64 = b_cur.to(tl.float64)
            tl.store(state_out_ptr + 0, step_f + 1.0)
            tl.store(state_out_ptr + 1, rows_proc_f + b_f64)
            tl.store(state_out_ptr + 2, ep_b_start_f + b_f64)
            tl.store(state_out_ptr + 3, u_bias_out)
            tl.store(state_out_ptr + 4, s_bias_out)
            tl.store(state_out_ptr + 5, a64)
            bo_in = tl.load(batch_off_in_ptr)
            tl.store(batch_off_out_ptr, bo_in + b_cur.to(tl.int64))

        # 2. Per-cluster scalar & vector EMA updates
        cnt = tl.load(counts_ptr + j).to(tl.float32)
        bsse = tl.minimum(tl.load(batch_sse_ptr + j).to(tl.float32), 1e35)
        old_u = tl.load(usage_ptr + j).to(tl.float32)
        old_sse = tl.load(sse_ema_ptr + j).to(tl.float32)

        touched = cnt > 0.0
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

        tl.store(counts_ptr + j, 0.0)
        tl.store(batch_sse_ptr + j, 0.0)

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
        amax_c = tl.zeros((), dtype=tl.float32)
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
                c_masked = tl.where(mask_d, c_new, 0.0)
                if REP_MODE == 3:
                    amax_c = tl.maximum(amax_c, tl.max(tl.abs(c_masked), axis=0))
                elif REP_MODE == 1:
                    cq = c_masked.to(tl.bfloat16)
                    tl.store(c_rep_ptr + j * stride_crepk + cols, cq, mask=mask_d)
                    cqf = cq.to(tl.float32)
                    norm_acc += tl.sum(cqf * cqf, axis=0)
                elif REP_MODE == 2:
                    cq = c_masked.to(tl.float16)
                    tl.store(c_rep_ptr + j * stride_crepk + cols, cq, mask=mask_d)
                    cqf = cq.to(tl.float32)
                    norm_acc += tl.sum(cqf * cqf, axis=0)
                else:
                    tl.store(c_rep_ptr + j * stride_crepk + cols, c_masked, mask=mask_d)
                    norm_acc += tl.sum(c_masked * c_masked, axis=0)

        if do_prep:
            if REP_MODE == 3:
                sc_j = tl.maximum(amax_c * (1.00000024 / 448.0), 1.1754943508222875e-38)
                inv_sc_j = 1.0 / sc_j
                q_sq = tl.zeros((), dtype=tl.float32)
                for d0 in range(0, stride_crepk, BLOCK_D):
                    cols = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
                    mask_d = cols < D
                    mask_pad = cols < stride_crepk
                    cv = tl.load(centroids_ptr + j * stride_cd + cols, mask=mask_d, other=0.0).to(tl.float32)
                    scaled = tl.minimum(tl.maximum(cv * inv_sc_j, -448.0), 448.0)
                    scaled = tl.where(mask_d, scaled, 0.0)
                    qc = scaled.to(tl.float8e4nv, fp_downcast_rounding="rtne")
                    tl.store(c_rep_ptr + j * stride_crepk + cols, qc, mask=mask_pad)
                    qcf = qc.to(tl.float32)
                    q_sq += tl.sum(tl.where(mask_d, qcf * qcf, 0.0), axis=0)
                tl.store(sc_ptr + j, sc_j)
                tl.store(c_norm_ptr + j, (sc_j * sc_j) * q_sq)
            else:
                tl.store(c_norm_ptr + j, norm_acc)

        new_u = tl.where(touched, mass, tl.where(is_rep, old_u, retained))
        new_sse = tl.where(
            is_rep,
            tl.where(touched, bsse, 0.5 * old_sse),
            sse_decay * old_sse + (1.0 - sse_decay) * bsse,
        )
        tl.store(usage_ptr + j, new_u)
        tl.store(sse_ema_ptr + j, new_sse)

    @triton.jit(do_not_specialize=["K", "D", "stride_ck", "stride_crepk"])
    def _apex_prep_centroids_kernel(
        centroids_ptr,
        c_rep_ptr,
        sc_ptr,
        c_norm_ptr,
        K,
        D,
        stride_ck,
        stride_crepk,
        REP_MODE: tl.constexpr,  # 0=FP32, 1=BF16, 2=FP16, 3=CROISS_FP8_DECOUPLED
        BLOCK_D: tl.constexpr,
    ):
        """Convert centered centroids to represented dtype (incl. CroissAttn-FP8) and compute `||c_rep||^2`."""
        j = tl.program_id(0).to(tl.int64)
        if j >= K:
            return
        if REP_MODE == 3:
            amax_c = tl.zeros((), dtype=tl.float32)
            for d0 in range(0, D, BLOCK_D):
                cols = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
                mask_d = cols < D
                cv = tl.load(centroids_ptr + j * stride_ck + cols, mask=mask_d, other=0.0).to(tl.float32)
                amax_c = tl.maximum(amax_c, tl.max(tl.where(mask_d, tl.abs(cv), 0.0), axis=0))
            sc_j = tl.maximum(amax_c * (1.00000024 / 448.0), 1.1754943508222875e-38)
            inv_sc = 1.0 / sc_j
            q_sq = tl.zeros((), dtype=tl.float32)
            for d0 in range(0, stride_crepk, BLOCK_D):
                cols = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
                mask_d = cols < D
                mask_pad = cols < stride_crepk
                cv = tl.load(centroids_ptr + j * stride_ck + cols, mask=mask_d, other=0.0).to(tl.float32)
                scaled = tl.where(mask_d, tl.minimum(tl.maximum(cv * inv_sc, -448.0), 448.0), 0.0)
                qc = scaled.to(tl.float8e4nv, fp_downcast_rounding="rtne")
                tl.store(c_rep_ptr + j * stride_crepk + cols, qc, mask=mask_pad)
                qcf = qc.to(tl.float32)
                q_sq += tl.sum(tl.where(mask_d, qcf * qcf, 0.0), axis=0)
            tl.store(sc_ptr + j, sc_j)
            tl.store(c_norm_ptr + j, (sc_j * sc_j) * q_sq)
        else:
            norm_acc = tl.zeros((), dtype=tl.float32)
            for d0 in range(0, stride_crepk, BLOCK_D):
                cols = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
                mask_d = cols < D
                mask_pad = cols < stride_crepk
                c_val = tl.load(centroids_ptr + j * stride_ck + cols, mask=mask_d, other=0.0).to(tl.float32)
                c_val = tl.where(mask_d, c_val, 0.0)
                if REP_MODE == 1:
                    cq = c_val.to(tl.bfloat16)
                    tl.store(c_rep_ptr + j * stride_crepk + cols, cq, mask=mask_pad)
                    cqf = cq.to(tl.float32)
                elif REP_MODE == 2:
                    cq = c_val.to(tl.float16)
                    tl.store(c_rep_ptr + j * stride_crepk + cols, cq, mask=mask_pad)
                    cqf = cq.to(tl.float32)
                else:
                    tl.store(c_rep_ptr + j * stride_crepk + cols, c_val, mask=mask_pad)
                    cqf = c_val
                norm_acc += tl.sum(tl.where(mask_d, cqf * cqf, 0.0), axis=0)
            tl.store(c_norm_ptr + j, norm_acc)

    @triton.jit(do_not_specialize=["K", "D", "stride_ck"])
    def _apex_huygens_nn_pass1_kernel(
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
    def _apex_huygens_leader_pass2_kernel(
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
        do_not_specialize=[
            "K",
            "D",
            "b_cur",
            "eff_grace",
            "revive_stop_step",
            "revive_stop_rows",
            "base_seed",
            "stride_cd",
        ]
    )
    def _apex_donor_gumbel_keys_kernel(
        state_ptr,
        usage_ptr,
        sse_ema_ptr,
        prin_vec_ptr,
        low_mask_ptr,
        prot_mask_ptr,
        max_sse_ptr,
        unused_i8_out_ptr,
        donor_keys_out_ptr,
        n_unused_acc_ptr,
        n_donor_acc_ptr,
        n_prot_acc_ptr,
        K,
        D,
        stride_cd,
        b_cur,
        eff_grace,
        revive_stop_step,
        revive_stop_rows,
        donor_usage_w,
        base_seed,
        USE_REVIVE_STOP: tl.constexpr,
        USE_WIDE_STOP: tl.constexpr,
        HAS_PROT: tl.constexpr,
        DONOR_MODE: tl.constexpr,  # 0=oja_sse_usage, 1=sse_usage, 2=d2, 3=usage_only
        TRACK_OJA: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """CUDAGraph-safe step-indexed Philox Gumbel donor scoring + on-device revival window gate."""
        pid_k = tl.program_id(0).to(tl.int64)
        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K).to(tl.int64)
        mask_k = offs_k < K

        # Note: `state_ptr` has ALREADY been incremented by 1 step in `_apex_ema_oja_slow_pass_prep_kernel`!
        step_after = tl.load(state_ptr + 0)
        rows_after = tl.load(state_ptr + 1)
        u_bias = tl.maximum(tl.load(state_ptr + 3).to(tl.float32), 1e-6)
        s_bias = tl.maximum(tl.load(state_ptr + 4).to(tl.float32), 1e-6)

        step_before = step_after - 1.0
        rows_before = rows_after - b_cur.to(tl.float64)

        active_win = step_after >= tl.maximum(eff_grace.to(tl.float64), 1.0)
        if USE_REVIVE_STOP:
            if USE_WIDE_STOP:
                active_win = active_win & (rows_before <= revive_stop_rows.to(tl.float64))
            else:
                active_win = active_win & (step_before <= revive_stop_step.to(tl.float64))

        u_val = tl.load(usage_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        e_val = tl.load(sse_ema_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        is_low = tl.load(low_mask_ptr + offs_k, mask=mask_k, other=0).to(tl.int8) != 0
        if HAS_PROT:
            is_prot = tl.load(prot_mask_ptr + offs_k, mask=mask_k, other=0).to(tl.int8) != 0
            unused = mask_k & active_win & is_low & (~is_prot)
            donor = mask_k & active_win & (~is_low) & (u_val > 0.0)
            n_pr = tl.sum(tl.where(mask_k & active_win & is_prot, 1, 0), axis=0)
            tl.atomic_add(n_prot_acc_ptr, n_pr.to(tl.int64), sem="relaxed")
        else:
            unused = mask_k & active_win & is_low
            donor = mask_k & active_win & (~unused) & (u_val > 0.0)

        tl.store(unused_i8_out_ptr + offs_k, unused.to(tl.int8), mask=mask_k)
        n_un = tl.sum(tl.where(unused, 1, 0), axis=0)
        n_dn = tl.sum(tl.where(donor, 1, 0), axis=0)
        tl.atomic_add(n_unused_acc_ptr, n_un.to(tl.int64), sem="relaxed")
        tl.atomic_add(n_donor_acc_ptr, n_dn.to(tl.int64), sem="relaxed")

        if DONOR_MODE == 3:
            score = u_val
        else:
            u_unb = tl.maximum(u_val / u_bias, 1e-12)
            e_unb = tl.maximum(e_val / s_bias, 1e-12)
            if DONOR_MODE == 0 and TRACK_OJA:
                sig1_sq = tl.zeros((BLOCK_K,), dtype=tl.float32)
                for d0 in range(0, D, BLOCK_D):
                    cols = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
                    mask_d = cols < D
                    pv = tl.load(
                        prin_vec_ptr + offs_k[:, None] * stride_cd + cols[None, :],
                        mask=mask_k[:, None] & mask_d[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    sig1_sq += tl.sum(pv * pv, axis=1)
                base_dist = 0.5 * e_unb + 0.5 * (D.to(tl.float32) * u_unb * sig1_sq)
            else:
                base_dist = e_unb

            if ((DONOR_MODE == 0) or (DONOR_MODE == 1)) and donor_usage_w != 0.0:
                u_rel = tl.maximum(u_unb / tl.maximum(b_cur.to(tl.float32), 1.0), 1e-12)
                weighted = base_dist * tl.exp(donor_usage_w * tl.log(u_rel))
            else:
                weighted = base_dist
            max_e = tl.load(max_sse_ptr).to(tl.float32)
            score = tl.where(max_e > 0.0, weighted, u_val)

        step_seed = base_seed + step_before.to(tl.int32) * 104729
        u_rnd = tl.rand(step_seed, offs_k.to(tl.int32))
        u_rnd = tl.minimum(tl.maximum(u_rnd, 1.17549435e-38), 1.0 - 1e-7)
        gumbel = -tl.log(-tl.log(u_rnd))
        key = tl.where(donor, tl.log(tl.maximum(score, 1e-30)) + gumbel, -float("inf"))
        tl.store(donor_keys_out_ptr + offs_k, key, mask=mask_k)

    @triton.jit(
        do_not_specialize=["D", "base_seed", "split_mode", "flush_closing", "update_pre_close", "stride_cd"]
    )
    def _apex_syncfree_split_kernel(
        state_ptr,
        centroids_ptr,
        origin_ptr,
        usage_ptr,
        sse_ema_ptr,
        prin_vec_ptr,
        survival_mass_ptr,
        slow_centroids_ptr,
        slow_usage_ptr,
        pre_close_centroids_ptr,
        pass_sums_ptr,
        pass_counts_ptr,
        pass_counts_sq_ptr,
        pass_sse_ptr,
        fvec_ptr,
        src_ptr,
        tgt_ptr,
        n_unused_ptr,
        n_donor_ptr,
        total_splits_ptr,
        D,
        stride_cd,
        inv_d,
        split_mult,
        split_eps,
        base_seed,
        split_mode,        # 0=half_gaussian_oja, 1=rms_aniso, 2=oja_only, 3=checkerboard
        flush_closing,
        update_pre_close,
        TRACK_OJA: tl.constexpr,
        TRACK_SURV: tl.constexpr,
        TRACK_SLOW: tl.constexpr,
        HAS_PRE_CLOSE: tl.constexpr,
        HAS_ORIGIN: tl.constexpr,
        REP_MODE: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """CUDAGraph-native Spectral-Concentration Half-Gaussian Oja Split + FP8/BF16 ULP Separation."""
        pid = tl.program_id(0).to(tl.int64)
        n_un = tl.load(n_unused_ptr)
        n_dn = tl.load(n_donor_ptr)
        n_sp = tl.minimum(n_un, n_dn)
        if pid == 0:
            tl.store(total_splits_ptr, tl.load(total_splits_ptr) + n_sp)
        if pid >= n_sp:
            return

        step_before = (tl.load(state_ptr + 0) - 1.0).to(tl.int32)
        u_bias = tl.maximum(tl.load(state_ptr + 3).to(tl.float32), 1e-6)
        s_bias = tl.maximum(tl.load(state_ptr + 4).to(tl.float32), 1e-6)
        usage_inv_bias = 1.0 / u_bias
        sse_inv_bias = 1.0 / s_bias
        rng_seed = base_seed + step_before * 131

        src = tl.load(src_ptr + pid).to(tl.int64)
        tgt = tl.load(tgt_ptr + pid).to(tl.int64)

        u_raw = tl.load(usage_ptr + src).to(tl.float32)
        sse_raw = tl.minimum(tl.load(sse_ema_ptr + src).to(tl.float32), 1e35)
        u_ub = tl.maximum(u_raw * usage_inv_bias, 1e-6)
        sse_ub = sse_raw * sse_inv_bias
        sigma_rms = tl.sqrt(tl.maximum((sse_ub / u_ub) * inv_d, 1e-12))

        pv_nsq = tl.zeros((), dtype=tl.float32)
        rad_nsq = tl.zeros((), dtype=tl.float32)
        pv_dot_rad = tl.zeros((), dtype=tl.float32)
        if split_mode != 3:
            for d0 in range(0, D, BLOCK_D):
                col = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
                cmask = col < D
                fv = tl.load(fvec_ptr + col, mask=cmask, other=0.0).to(tl.float32)
                u_rnd = tl.rand(rng_seed, (pid * D + col).to(tl.int32))
                radem = tl.where(cmask, tl.where(u_rnd < 0.5, -1.0, 1.0) * fv, 0.0)
                rad_nsq += tl.sum(radem * radem, axis=0)
                if TRACK_OJA:
                    pv = tl.load(prin_vec_ptr + src * stride_cd + col, mask=cmask, other=0.0).to(tl.float32)
                    pv_nsq += tl.sum(pv * pv, axis=0)
                    pv_dot_rad += tl.sum(pv * radem, axis=0)

        sigma_1 = tl.sqrt(pv_nsq)
        inv_pv = 1.0 / tl.maximum(sigma_1, 1e-12)
        inv_rad = 1.0 / tl.maximum(tl.sqrt(rad_nsq), 1e-12)
        align_sign = tl.where(pv_dot_rad >= 0.0, 1.0, -1.0)

        ratio = sigma_1 / tl.maximum(sigma_rms, 1e-12)
        rho = tl.minimum(tl.maximum((ratio - 0.85) * (1.0 / 0.35), 0.0), 1.0)
        if not TRACK_OJA:
            rho = 0.0
        elif split_mode == 2:
            rho = tl.where(sigma_1 > 1e-10, 1.0, 0.0)

        cos_vr = tl.abs(pv_dot_rad) * inv_pv * inv_rad
        blend_nsq = rho * rho + (1.0 - rho) * (1.0 - rho) + 2.0 * rho * (1.0 - rho) * cos_vr
        inv_blend = 1.0 / tl.maximum(tl.sqrt(blend_nsq), 1e-12)

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

        orth_nsq = tl.zeros((), dtype=tl.float32)
        u_dot_rnd2 = tl.zeros((), dtype=tl.float32)
        if TRACK_OJA and D > 1:
            for d0 in range(0, D, BLOCK_D):
                col = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
                cmask = col < D
                fv = tl.load(fvec_ptr + col, mask=cmask, other=0.0).to(tl.float32)
                u_rnd = tl.rand(rng_seed, (pid * D + col).to(tl.int32))
                radem = tl.where(cmask, tl.where(u_rnd < 0.5, -1.0, 1.0) * fv, 0.0)
                pv = tl.load(prin_vec_ptr + src * stride_cd + col, mask=cmask, other=0.0).to(tl.float32)
                if split_mode == 1:
                    u_dir = radem * inv_rad
                else:
                    u_dir = (rho * (pv * inv_pv) + (1.0 - rho) * align_sign * (radem * inv_rad)) * inv_blend
                g2 = tl.where(cmask, tl.randn(rng_seed + 17, (pid * D + col).to(tl.int32)), 0.0)
                u_dot_rnd2 += tl.sum(u_dir * g2, axis=0)

            for d0 in range(0, D, BLOCK_D):
                col = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
                cmask = col < D
                fv = tl.load(fvec_ptr + col, mask=cmask, other=0.0).to(tl.float32)
                u_rnd = tl.rand(rng_seed, (pid * D + col).to(tl.int32))
                radem = tl.where(cmask, tl.where(u_rnd < 0.5, -1.0, 1.0) * fv, 0.0)
                pv = tl.load(prin_vec_ptr + src * stride_cd + col, mask=cmask, other=0.0).to(tl.float32)
                if split_mode == 1:
                    u_dir = radem * inv_rad
                else:
                    u_dir = (rho * (pv * inv_pv) + (1.0 - rho) * align_sign * (radem * inv_rad)) * inv_blend
                g2 = tl.where(cmask, tl.randn(rng_seed + 17, (pid * D + col).to(tl.int32)), 0.0)
                orth_v = tl.where(cmask, g2 - u_dot_rnd2 * u_dir, 0.0)
                orth_nsq += tl.sum(orth_v * orth_v, axis=0)

        inv_orth = 1.0 / tl.maximum(tl.sqrt(orth_nsq), 1e-12)
        rem_sigma = 0.602810275 * sigma_rms

        all_equal = 1
        amax_c0 = tl.zeros((), dtype=tl.float32)
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
                u_rnd = tl.rand(rng_seed, (pid * D + col).to(tl.int32))
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

            amax_c0 = tl.maximum(amax_c0, tl.max(tl.where(cmask, tl.abs(c_tgt), 0.0), axis=0))
            if REP_MODE == 1:
                eq_col = c_tgt.to(tl.bfloat16).to(tl.float32) == c_src.to(tl.bfloat16).to(tl.float32)
            elif REP_MODE == 2:
                eq_col = c_tgt.to(tl.float16).to(tl.float32) == c_src.to(tl.float16).to(tl.float32)
            elif REP_MODE == 3:
                # Conservative 1-ULP E4M3 relative separation check (mantissa 3 bits -> 1/8 = 0.125)
                eq_col = tl.abs(c_tgt - c_src) <= (0.14 * tl.maximum(tl.abs(c_tgt), 1e-30))
            else:
                eq_col = c_tgt == c_src
            if tl.sum(tl.where(cmask & (~eq_col), 1, 0), axis=0) > 0:
                all_equal = 0

            tl.store(centroids_ptr + tgt * stride_cd + col, c_tgt, mask=cmask)
            tl.store(centroids_ptr + src * stride_cd + col, c_src, mask=cmask)

            if TRACK_SLOW:
                tl.store(slow_centroids_ptr + tgt * stride_cd + col, c_tgt, mask=cmask)
                tl.store(slow_centroids_ptr + src * stride_cd + col, c_src, mask=cmask)

            if HAS_PRE_CLOSE:
                if update_pre_close != 0:
                    tl.store(pre_close_centroids_ptr + tgt * stride_cd + col, c_tgt, mask=cmask)
                    tl.store(pre_close_centroids_ptr + src * stride_cd + col, c_src, mask=cmask)

            if TRACK_OJA:
                if D <= 1:
                    pv_rst = rem_sigma * u_dir
                else:
                    g2 = tl.where(cmask, tl.randn(rng_seed + 17, (pid * D + col).to(tl.int32)), 0.0)
                    orth_v = tl.where(cmask, g2 - u_dot_rnd2 * u_dir, 0.0)
                    pv_rst = tl.where(orth_nsq > 1e-16, rem_sigma * (orth_v * inv_orth), rem_sigma * u_dir)
                tl.store(prin_vec_ptr + tgt * stride_cd + col, pv_rst, mask=cmask)
                tl.store(prin_vec_ptr + src * stride_cd + col, pv_rst, mask=cmask)

            if flush_closing != 0:
                tl.store(pass_sums_ptr + tgt * stride_cd + col, 0.0, mask=cmask)
                tl.store(pass_sums_ptr + src * stride_cd + col, 0.0, mask=cmask)

        if all_equal != 0:
            c0 = tl.load(centroids_ptr + tgt * stride_cd).to(tl.float32)
            if REP_MODE == 3:
                step_ulp = tl.maximum(amax_c0 * 0.145, 1e-28)
            elif REP_MODE == 1:
                step_ulp = tl.maximum(tl.abs(c0) * (1.0 / 64.0), 1e-30)
            elif REP_MODE == 2:
                step_ulp = tl.maximum(tl.abs(c0) * (1.0 / 512.0), 1e-4)
            else:
                step_ulp = tl.maximum(tl.abs(c0) * 2.5e-7, 1e-30)
            nudge = tl.where(c0 > 0.0, -step_ulp, step_ulp)
            tl.store(centroids_ptr + tgt * stride_cd, c0 + nudge)
            if TRACK_SLOW:
                tl.store(slow_centroids_ptr + tgt * stride_cd, c0 + nudge)
            if HAS_PRE_CLOSE:
                if update_pre_close != 0:
                    tl.store(pre_close_centroids_ptr + tgt * stride_cd, c0 + nudge)

        half_u = 0.5 * u_raw
        tl.store(usage_ptr + tgt, half_u)
        tl.store(usage_ptr + src, half_u)

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

    @triton.jit(do_not_specialize=["K", "D", "stride_ck"])
    def _apex_all_pairs_nn_kernel(
        c_ptr,
        c_norm_ptr,
        out_d2_ptr,
        out_idx_ptr,
        K,
        D,
        stride_ck,
        BLOCK_Q: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """SRAM-tiled all-pairs nearest-neighbor distances `(d_nn2, nn_idx)` with `0` KxK HBM memory."""
        pid_q = tl.program_id(0).to(tl.int64)
        offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q).to(tl.int64)
        mask_q = offs_q < K
        qn = tl.load(c_norm_ptr + offs_q, mask=mask_q, other=0.0).to(tl.float32)
        best_d2 = tl.full((BLOCK_Q,), float("inf"), dtype=tl.float32)
        best_id = tl.zeros((BLOCK_Q,), dtype=tl.int32)

        for k0 in range(0, K, BLOCK_K):
            offs_k = (k0 + tl.arange(0, BLOCK_K)).to(tl.int64)
            mask_k = offs_k < K
            dots = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)
            for d0 in range(0, D, BLOCK_D):
                cols = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
                mask_d = cols < D
                qt = tl.load(
                    c_ptr + offs_q[:, None] * stride_ck + cols[None, :],
                    mask=mask_q[:, None] & mask_d[None, :],
                    other=0.0,
                ).to(tl.float32)
                kt = tl.load(
                    c_ptr + offs_k[:, None] * stride_ck + cols[None, :],
                    mask=mask_k[:, None] & mask_d[None, :],
                    other=0.0,
                ).to(tl.float32)
                dots = tl.dot(qt, tl.trans(kt), acc=dots, input_precision="ieee", out_dtype=tl.float32)
            kn = tl.load(c_norm_ptr + offs_k, mask=mask_k, other=float("inf")).to(tl.float32)
            d2 = tl.maximum(qn[:, None] + kn[None, :] - 2.0 * dots, 0.0)
            valid = mask_k[None, :] & (offs_q[:, None] != offs_k[None, :])
            d2 = tl.where(valid, d2, float("inf"))
            b_min, b_loc = tl.min(d2, axis=1, return_indices=True, return_indices_tie_break_left=True)
            better = b_min < best_d2
            best_d2 = tl.where(better, b_min, best_d2)
            best_id = tl.where(better, k0 + b_loc.to(tl.int32), best_id)

        tl.store(out_d2_ptr + offs_q, best_d2, mask=mask_q)
        tl.store(out_idx_ptr + offs_q, best_id.to(tl.int64), mask=mask_q)


# ==============================================================================
# 2. Kernel Dispatch Helpers, Coalesced Pre-Encoder & On-Device Revival Workspace
# ==============================================================================


def _alloc_padded_2d(rows: int, d: int, dtype: torch.dtype, device: torch.device, align: int = 16) -> torch.Tensor:
    d_pad = max(align, ((d + align - 1) // align) * align)
    return torch.zeros((rows, d_pad), dtype=dtype, device=device)


def _rep_mode_from_precision(prec: str, comp_dtype: torch.dtype) -> int:
    if prec in ("croiss_fp8", "fp8"):
        return 3
    if comp_dtype == torch.bfloat16:
        return 1
    if comp_dtype == torch.float16:
        return 2
    return 0


@dataclass
class _ApexRevivalWorkspace:
    """Static pre-allocated GPU buffers so Huygens + Gumbel-Donor + Oja-Split capture inside CUDAGraph."""

    c_pad: torch.Tensor
    c_norm: torch.Tensor
    d2_any: torch.Tensor
    d2_surv: torch.Tensor
    prot_i8_sorted: torch.Tensor
    prot_i8: torch.Tensor
    inv_order: torch.Tensor
    arange_k: torch.Tensor
    unused_i8: torch.Tensor
    donor_keys: torch.Tensor
    n_unused: torch.Tensor
    n_donor: torch.Tensor
    max_sse: torch.Tensor

    @classmethod
    def allocate(cls, k: int, d: int, device: torch.device) -> _ApexRevivalWorkspace:
        return cls(
            c_pad=_alloc_padded_2d(k, d, torch.float32, device, align=16),
            c_norm=torch.empty((k,), dtype=torch.float32, device=device),
            d2_any=torch.empty((k,), dtype=torch.float32, device=device),
            d2_surv=torch.empty((k,), dtype=torch.float32, device=device),
            prot_i8_sorted=torch.zeros((k,), dtype=torch.int8, device=device),
            prot_i8=torch.zeros((k,), dtype=torch.int8, device=device),
            inv_order=torch.empty((k,), dtype=torch.int64, device=device),
            arange_k=torch.arange(k, dtype=torch.int64, device=device),
            unused_i8=torch.zeros((k,), dtype=torch.int8, device=device),
            donor_keys=torch.empty((k,), dtype=torch.float32, device=device),
            n_unused=torch.zeros((), dtype=torch.int64, device=device),
            n_donor=torch.zeros((), dtype=torch.int64, device=device),
            max_sse=torch.zeros((), dtype=torch.float32, device=device),
        )


def _launch_apex_prep_centroids(
    centroids_c: torch.Tensor,
    c_rep: torch.Tensor,
    sc_buf: torch.Tensor,
    c_norm: torch.Tensor,
    rep_mode: int,
    croiss_ws: Any = None,
    croiss_ds: Any = None,
) -> None:
    if rep_mode == 3 and croiss_ws is not None:
        cmod = _get_croiss_fp8_module()
        if cmod is not None and hasattr(cmod, "croiss_prepare_centroids_inplace"):
            cmod.croiss_prepare_centroids_inplace(
                centroids_c,
                croiss_ds,
                c_rep,
                sc_buf,
                c_norm,
                centroids_already_centered=True,
                scale_mode=getattr(croiss_ws, "scale_mode", "decoupled"),
                _unit_gscale_ptr=getattr(croiss_ws, "unit_gscale_ptr", None),
            )
            return
    k, d = centroids_c.shape
    block_d = min(256, max(16, triton.next_power_of_2(max(1, d))))
    with torch.cuda.device(centroids_c.device):
        _apex_prep_centroids_kernel[(k,)](
            centroids_c,
            c_rep,
            sc_buf,
            c_norm,
            k,
            d,
            centroids_c.stride(0),
            c_rep.stride(0),
            REP_MODE=rep_mode,
            BLOCK_D=block_d,
            num_warps=2 if d <= 128 else 4,
        )


def _launch_apex_assign(
    x_src: torch.Tensor,
    c_rep: torch.Tensor,
    c_norm: torch.Tensor,
    rep_mode: int,
    d: int,
    sx_buf: torch.Tensor | None = None,
    sc_buf: torch.Tensor | None = None,
    x_norm_buf: torch.Tensor | None = None,
    row_indices: torch.Tensor | None = None,
    batch_offset_ptr: torch.Tensor | None = None,
    batch_size: int | None = None,
    wrap_rows: bool = False,
    origin: torch.Tensor | None = None,
    coord_scale: float = 1.0,
    out_dist: torch.Tensor | None = None,
    out_label: torch.Tensor | None = None,
    splitk_buffers: tuple[torch.Tensor, torch.Tensor, torch.Tensor, int] | None = None,
    use_sm120_kernel: bool = True,
    croiss_ds: Any = None,
    croiss_ws: Any = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unified CUDAGraph-safe Tensor-Core assignment across FP32 / BF16 / FP16 / CroissAttn-FP8."""
    if batch_size is not None:
        b = int(batch_size)
    elif row_indices is not None and batch_offset_ptr is None:
        b = int(row_indices.shape[0])
    else:
        b = int(x_src.shape[0])

    k = int(c_rep.shape[0])
    device = x_src.device
    if out_dist is None:
        out_dist = torch.empty((b,), dtype=torch.float32, device=device)
    if out_label is None:
        out_label = torch.empty((b,), dtype=torch.int64, device=device)
    if b == 0:
        return out_dist, out_label

    # Delegate to `@croiss-fp8`'s specialized workspace assigner when available & compatible
    if rep_mode == 3 and croiss_ds is not None and croiss_ws is not None and use_sm120_kernel:
        cmod = _get_croiss_fp8_module()
        if cmod is not None and hasattr(cmod, "croiss_fp8_assign_inplace"):
            return cmod.croiss_fp8_assign_inplace(
                croiss_ds,
                croiss_ws,
                batch_size=b,
                row_offset=0,
                batch_offset_ptr=batch_offset_ptr,
                wrap_rows=wrap_rows,
                row_indices=row_indices,
                out_dists=out_dist,
                out_labels=out_label,
                allow_tma=True,
            )

    n_total = int(row_indices.shape[0]) if row_indices is not None else int(x_src.shape[0])
    has_idx = row_indices is not None
    has_boff = batch_offset_ptr is not None
    has_orig = origin is not None
    orig_fp64 = bool(has_orig and origin.dtype == torch.float64)
    has_scale = coord_scale != 1.0
    precomp_xn = x_norm_buf is not None

    dummy_f32 = c_norm
    sx_arg = sx_buf if sx_buf is not None else dummy_f32
    sc_arg = sc_buf if sc_buf is not None else dummy_f32
    xn_arg = x_norm_buf if x_norm_buf is not None else dummy_f32
    idx_arg = row_indices if has_idx else out_label
    boff_arg = batch_offset_ptr if has_boff else out_label
    orig_arg = origin if has_orig else x_src

    cap = torch.cuda.get_device_capability(device)
    low_smem = (cap < (8, 0)) or (not use_sm120_kernel)
    reduced = rep_mode in (1, 2, 3)

    if not use_sm120_kernel:
        full_d = False
    elif rep_mode == 3:
        full_d = d <= 256
    elif reduced:
        full_d = d <= 128
    else:
        full_d = d <= 64

    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    if use_sm120_kernel and (not low_smem) and k >= 2048 and triton.cdiv(b, 64) < num_sms:
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
                else (128 if rep_mode == 3 else (64 if reduced else 32))
            )
            if rep_mode == 3:
                sp_block_d = max(32, sp_block_d)
            with torch.cuda.device(device):
                _apex_splitk_assign_kernel[(triton.cdiv(b, sp_block_b), parts)](
                    x_src,
                    sx_arg,
                    xn_arg,
                    idx_arg,
                    boff_arg,
                    orig_arg,
                    c_rep,
                    sc_arg,
                    c_norm,
                    part_sc,
                    part_lbl,
                    part_xn,
                    b,
                    k,
                    d,
                    n_total,
                    x_src.stride(0),
                    x_src.stride(1),
                    c_rep.stride(0),
                    float(coord_scale),
                    PARTITIONS=parts,
                    BLOCK_B=sp_block_b,
                    BLOCK_K=sp_block_k,
                    BLOCK_D=sp_block_d,
                    FULL_D=full_d,
                    DTYPE_MODE=rep_mode,
                    PRECOMP_XNORM=precomp_xn,
                    HAS_INDICES=has_idx,
                    HAS_BATCH_OFFSET=has_boff,
                    WRAP_ROWS=wrap_rows,
                    HAS_ORIGIN=has_orig,
                    HAS_SCALE=has_scale,
                    ORIGIN_FP64=orig_fp64,
                    num_warps=4 if full_d else 8,
                    num_stages=3,
                )
                red_blk_b = 128
                _apex_splitk_reduce_kernel[(triton.cdiv(b, red_blk_b),)](
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
        block_d = max(32 if rep_mode == 3 else 16, triton.next_power_of_2(max(1, d)))
        if b <= 512 and k >= 1024:
            block_b = 32 if rep_mode == 3 else 16
        elif reduced and not low_smem and b >= 128 * num_sms and k >= 256 and block_d <= 128:
            block_b = 128
        else:
            block_b = 64
        block_k = 128 if (reduced and not low_smem and k >= 128) else 64
        num_warps = 8 if (reduced and not low_smem and block_d >= 64 and block_k >= 128 and block_b >= 64) else 4
        num_stages = 1 if low_smem else 3
    else:
        block_d = 128 if (rep_mode == 3 and use_sm120_kernel) else (64 if (reduced and use_sm120_kernel) else 32)
        if rep_mode == 3 and block_d < 32:
            block_d = 32
        block_b = 64
        block_k = 128 if (reduced and not low_smem and k >= 128) else 64
        num_warps = 8 if (reduced and not low_smem) else 4
        num_stages = 1 if low_smem else 3

    with torch.cuda.device(device):
        _apex_assign_kernel[(triton.cdiv(b, block_b),)](
            x_src,
            sx_arg,
            xn_arg,
            idx_arg,
            boff_arg,
            orig_arg,
            c_rep,
            sc_arg,
            c_norm,
            out_dist,
            out_label,
            b,
            k,
            d,
            n_total,
            x_src.stride(0),
            x_src.stride(1),
            c_rep.stride(0),
            float(coord_scale),
            BLOCK_B=block_b,
            BLOCK_K=block_k,
            BLOCK_D=block_d,
            FULL_D=full_d,
            DTYPE_MODE=rep_mode,
            PRECOMP_XNORM=precomp_xn,
            HAS_INDICES=has_idx,
            HAS_BATCH_OFFSET=has_boff,
            WRAP_ROWS=wrap_rows,
            HAS_ORIGIN=has_orig,
            HAS_SCALE=has_scale,
            ORIGIN_FP64=orig_fp64,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return out_dist, out_label


@torch.no_grad()
def _launch_apex_cudagraph_revival(
    state_slot: torch.Tensor,
    centroids_c: torch.Tensor,
    origin32: torch.Tensor,
    usage: torch.Tensor,
    sse_ema: torch.Tensor,
    prin_vec: torch.Tensor,
    survival_mass: torch.Tensor,
    slow_centroids: torch.Tensor,
    slow_usage: torch.Tensor,
    pre_close_centroids: torch.Tensor,
    pass_sums: torch.Tensor,
    pass_counts: torch.Tensor,
    pass_counts_sq: torch.Tensor,
    pass_sse: torch.Tensor,
    feat_scale: torch.Tensor,
    ones_d: torch.Tensor,
    ws: _ApexRevivalWorkspace,
    total_splits_gpu: torch.Tensor,
    total_prot_gpu: torch.Tensor,
    b_cur: int,
    unused_threshold: float,
    gamma_protect: float,
    eff_grace: int,
    revive_stop_step: int,
    revive_stop_rows: int,
    donor_usage_w: float,
    split_mult: float,
    split_eps: float,
    base_seed: int,
    use_revive_stop: bool,
    use_wide_stop: bool,
    track_surv: bool,
    track_oja: bool,
    track_slow: bool,
    has_pre_close: bool,
    update_pre_close: bool,
    use_feat_scale: bool,
    flush_closing: bool,
    donor_mode_code: int,
    split_mode_code: int,
    rep_mode: int,
) -> None:
    """100% CUDAGraph-capturable Twin-Safe Huygens + Step-Indexed Gumbel Donor + Half-Gaussian Split."""
    k, d = centroids_c.shape
    device = centroids_c.device

    thresh = usage.mean() * unused_threshold
    low_mask = usage < thresh
    low_i8_orig = low_mask.to(torch.int8)

    if track_surv and gamma_protect > 0.0 and 2 <= k < 16384:
        eff_mass = torch.where(
            survival_mass > 0.0,
            torch.maximum(usage, torch.minimum(survival_mass, thresh)),
            torch.zeros_like(usage),
        )
        s_bias = state_slot[4].clamp_min(1e-6).float()
        ref_dist_mean = (sse_ema.mean() / s_bias).clamp_min(1e-30)
        protect_cutoff = (gamma_protect * ref_dist_mean).reshape(())

        low_order = torch.argsort(low_i8_orig, descending=True, stable=True)
        ws.inv_order.scatter_(0, low_order, ws.arange_k)

        c_mean = centroids_c.mean(dim=0, keepdim=True)
        kc_sorted = (centroids_c - c_mean)[low_order]
        ws.c_pad.zero_()
        ws.c_pad[:, :d] = kc_sorted
        ws.c_norm.copy_(kc_sorted.square().sum(dim=1))

        low_i8_s = low_i8_orig[low_order].contiguous()
        usage_s = usage[low_order].contiguous()
        eff_mass_s = eff_mass[low_order].contiguous()
        blk_q = 64 if k >= 64 else 16
        blk_k = 64 if k >= 64 else 16
        blk_d_h = min(64, max(16, triton.next_power_of_2(max(1, d))))

        with torch.cuda.device(device):
            _apex_huygens_nn_pass1_kernel[(triton.cdiv(k, blk_q),)](
                ws.c_pad,
                ws.c_norm,
                low_i8_s,
                ws.d2_any,
                ws.d2_surv,
                k,
                d,
                ws.c_pad.stride(0),
                BLOCK_Q=blk_q,
                BLOCK_K=blk_k,
                BLOCK_D=blk_d_h,
                num_warps=4,
            )
            _apex_huygens_leader_pass2_kernel[(triton.cdiv(k, blk_q),)](
                ws.c_pad,
                ws.c_norm,
                low_i8_s,
                usage_s,
                eff_mass_s,
                ws.d2_any,
                ws.d2_surv,
                protect_cutoff,
                ws.prot_i8_sorted,
                k,
                d,
                ws.c_pad.stride(0),
                BLOCK_Q=blk_q,
                BLOCK_K=blk_k,
                BLOCK_D=blk_d_h,
                num_warps=4,
            )
        ws.prot_i8.copy_(ws.prot_i8_sorted[ws.inv_order])
        has_prot = True
    else:
        has_prot = False

    ws.n_unused.zero_()
    ws.n_donor.zero_()
    torch.amax(sse_ema, out=ws.max_sse)

    blk_k_don = 128 if k >= 128 else max(16, triton.next_power_of_2(max(1, k)))
    blk_d_don = min(128, max(16, triton.next_power_of_2(max(1, d))))
    with torch.cuda.device(device):
        _apex_donor_gumbel_keys_kernel[(triton.cdiv(k, blk_k_don),)](
            state_slot,
            usage,
            sse_ema,
            prin_vec,
            low_i8_orig,
            ws.prot_i8,
            ws.max_sse,
            ws.unused_i8,
            ws.donor_keys,
            ws.n_unused,
            ws.n_donor,
            total_prot_gpu,
            k,
            d,
            centroids_c.stride(0),
            int(b_cur),
            int(eff_grace),
            int(revive_stop_step),
            int(revive_stop_rows),
            float(donor_usage_w),
            int(base_seed % (2**31 - 1)),
            USE_REVIVE_STOP=use_revive_stop,
            USE_WIDE_STOP=use_wide_stop,
            HAS_PROT=has_prot,
            DONOR_MODE=donor_mode_code,
            TRACK_OJA=track_oja,
            BLOCK_K=blk_k_don,
            BLOCK_D=blk_d_don,
            num_warps=4,
        )

    src_sorted = torch.argsort(ws.donor_keys, descending=True)
    tgt_sorted = torch.argsort(ws.unused_i8, descending=True, stable=True)

    if use_feat_scale and split_mode_code in (0, 1):
        fvec_s = feat_scale.clamp(1e-10, 1e35).sqrt()
        fnorm = fvec_s.double().norm().clamp_min(1e-8).float()
        fvec = fvec_s * (math.sqrt(float(d)) / fnorm)
    else:
        fvec = ones_d

    blk_d_sp = min(256, max(16, triton.next_power_of_2(max(1, d))))
    with torch.cuda.device(device):
        _apex_syncfree_split_kernel[(k,)](
            state_slot,
            centroids_c,
            origin32,
            usage,
            sse_ema,
            prin_vec,
            survival_mass,
            slow_centroids,
            slow_usage,
            pre_close_centroids,
            pass_sums,
            pass_counts,
            pass_counts_sq,
            pass_sse,
            fvec,
            src_sorted,
            tgt_sorted,
            ws.n_unused,
            ws.n_donor,
            total_splits_gpu,
            d,
            centroids_c.stride(0),
            float(1.0 / max(1, d)),
            float(split_mult),
            float(split_eps),
            int((base_seed * 10007 + 97) % (2**31 - 1)),
            int(split_mode_code),
            int(flush_closing),
            int(update_pre_close),
            TRACK_OJA=track_oja,
            TRACK_SURV=track_surv,
            TRACK_SLOW=track_slow,
            HAS_PRE_CLOSE=has_pre_close,
            HAS_ORIGIN=True,
            REP_MODE=rep_mode,
            BLOCK_D=blk_d_sp,
            num_warps=2 if d <= 128 else 4,
        )


# ==============================================================================
# 3. Fast Voronoi-Sharded Global/Parallel `apex_turkpp` Seeding
# ==============================================================================


@torch.no_grad()
def _compute_origin_and_scale(
    x_t: torch.Tensor,
    comp_dtype: torch.dtype,
    sample_rows: int = 32768,
) -> tuple[torch.Tensor, float]:
    """Compute translation-invariant coordinate origin `(D,)` and exact power-of-two `coord_scale`."""
    n, d = int(x_t.shape[0]), int(x_t.shape[1])
    device = x_t.device
    if n == 0:
        return torch.zeros((d,), dtype=torch.float64, device=device), 1.0

    stride = max(1, n // sample_rows)
    x_sub64 = x_t[::stride].double()
    origin64 = x_sub64.mean(dim=0).contiguous()

    # Check global min/max on sampled + boundary rows (or full tensor when N <= 262,144)
    if n <= 262144 or comp_dtype == torch.float16:
        x_min = float(x_t.min().item())
        x_max = float(x_t.max().item())
    else:
        x_min = float(min(x_sub64.min().item(), x_t[:4096].min().item(), x_t[-4096:].min().item()))
        x_max = float(max(x_sub64.max().item(), x_t[:4096].max().item(), x_t[-4096:].max().item()))
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
def apex_turkpp_init_centered(
    x_t: torch.Tensor,
    origin64: torch.Tensor,
    coord_scale: float,
    k: int,
    comp_dtype: torch.dtype,
    coreset_mult: int | str = "auto",
    min_coreset_size: int = 4096,
    polish: bool = True,
    seed: int = 1234,
) -> SeedingResult:
    """Scale-Adaptive Importance Coreset + Global/Voronoi-Sharded Tournament `k-means++` + Huygens Polish.

    - For `P <= 1024` (`K <= 2048`), executes exact single-CTA global `BLOCK_P <= 1024` tournament
      in full `R^D` (eliminating 1D random-projection collisions on high-D `D=512..1024` manifolds!).
    - For `K > 2048` (`K = 4,096 .. 65,536`), partitions candidate pools by nearest already-placed
      `R^D` coarse centroids (true Voronoi neighborhood sharding!) across up to `b_r = 4,096` centers
      per round (`<= 6` rounds at `K = 16,384`), followed by a 1-pass fused scatter + Huygens parallel-axis
      polish (`~2.2 ms` on `200K x 128 @ K=2048`, `~10 ms` on `1M x 128 @ K=16,384`).
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

    if coreset_mult == "auto":
        if n <= 100_000:
            eff_mult = 16
        elif n <= 350_000:
            eff_mult = 8
        elif k >= 16384:
            eff_mult = 2 if k >= 32768 else 3
        else:
            eff_mult = 4 if k >= 8192 else 6
    else:
        eff_mult = int(coreset_mult)

    m = min(n, max(int(min_coreset_size), eff_mult * int(k), int(k) * 2))
    origin_ker = (
        origin64
        if (x_t.dtype == torch.float64 or coord_scale != 1.0 or float(origin64.abs().max().item()) > 1e3)
        else origin64.float().contiguous()
    )

    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed) % (2**32))

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
                _apex_origin_d2_kernel[(triton.cdiv(n, blk_n),)](
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
        u_strat = (
            torch.arange(m, device=device, dtype=torch.float64)
            + torch.rand((m,), generator=gen, device=device, dtype=torch.float64)
        ) * (cdf[-1] / float(m))
        idx_sub = torch.searchsorted(cdf, u_strat, right=True).clamp_(0, n - 1)
        if origin_ker.dtype == torch.float32 and coord_scale == 1.0 and x_t.dtype != torch.float64:
            xs_c = (x_t[idx_sub].float() - origin_ker[None, :]).contiguous()
        else:
            xs_c = ((x_t[idx_sub].double() - origin64[None, :]) * coord_scale).float().contiguous()
        w_sub = (1.0 / (float(m) * q[idx_sub])).float()
        w_sub = (w_sub / w_sub.sum().clamp_min(1e-30)).contiguous()

    use_fp8_seed = (
        device.type == "cuda"
        and _has_fp8_tensor_cores(device)
        and d >= 32
        and m >= 32768
        and k >= 4096
    )
    seed_rep_mode = 3 if use_fp8_seed else (
        1 if (device.type == "cuda" and HAS_TRITON and d >= 32 and comp_dtype in (torch.bfloat16, torch.float16)) else 0
    )
    seed_dtype = FLOAT8_E4M3FN if seed_rep_mode == 3 else (torch.bfloat16 if seed_rep_mode == 1 else torch.float32)

    xs_seed_src = xs_c
    xs_sx: torch.Tensor | None = None
    xs_xnorm: torch.Tensor | None = None
    if device.type == "cuda" and HAS_TRITON:
        pad_align = 64 if seed_rep_mode == 3 else 16
        xs_rep = _alloc_padded_2d(m, d, seed_dtype, device, align=pad_align)
        xs_sx = torch.empty((m,), dtype=torch.float32, device=device) if seed_rep_mode == 3 else None
        xs_xnorm = torch.empty((m,), dtype=torch.float32, device=device)
        blk_n_enc = 64
        blk_d_enc = min(128, max(32 if seed_rep_mode == 3 else 16, triton.next_power_of_2(max(1, d))))
        dummy_orig = torch.zeros((d,), dtype=torch.float32, device=device)
        with torch.cuda.device(device):
            _apex_prepermute_encode_kernel[(triton.cdiv(m, blk_n_enc),)](
                xs_c,
                xs_xnorm,  # dummy perm
                dummy_orig,
                xs_c,
                xs_rep,
                xs_sx if xs_sx is not None else xs_xnorm,
                xs_xnorm,
                m,
                d,
                xs_c.stride(0),
                xs_c.stride(1),
                xs_c.stride(0),
                xs_rep.stride(0),
                1.0,
                HAS_PERM=False,
                ORIGIN_FP64=False,
                HAS_SCALE=False,
                WRITE_F32=False,
                REP_MODE=seed_rep_mode,
                BLOCK_N=blk_n_enc,
                BLOCK_D=blk_d_enc,
                num_warps=4,
            )
        xs_seed_src = xs_rep

    avail = torch.ones((m,), dtype=torch.bool, device=device)
    false_buf = torch.zeros((m,), dtype=torch.bool, device=device)
    dummy_sc = torch.ones((max(16, k),), dtype=torch.float32, device=device)

    def _seed_assign_pool(c_mat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        kc_n = int(c_mat.shape[0])
        if device.type == "cuda" and HAS_TRITON:
            pad_a = 64 if seed_rep_mode == 3 else 16
            c_r = _alloc_padded_2d(kc_n, d, seed_dtype, device, align=pad_a)
            sc_r = torch.empty((kc_n,), dtype=torch.float32, device=device) if seed_rep_mode == 3 else dummy_sc[:kc_n]
            cn_r = torch.empty((kc_n,), dtype=torch.float32, device=device)
            _launch_apex_prep_centroids(c_mat, c_r, sc_r, cn_r, seed_rep_mode)
            return _launch_apex_assign(
                x_src=xs_seed_src,
                c_rep=c_r,
                c_norm=cn_r,
                rep_mode=seed_rep_mode,
                d=d,
                sx_buf=xs_sx,
                sc_buf=sc_r,
                x_norm_buf=xs_xnorm,
            )
        return assign_points(xs_c, c_mat, compute_dtype=torch.bfloat16 if seed_rep_mode == 1 else torch.float32)

    def _apex_select_pool(
        c_pool: torch.Tensor,
        init_pool_d2: torch.Tensor,
        pool_weights: torch.Tensor,
        num_select: int,
        rseed: int,
        coarse_anchors: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        P_sz = int(c_pool.shape[0])
        if P_sz <= num_select:
            return c_pool[:num_select], torch.arange(min(P_sz, num_select), device=device, dtype=torch.int64)

        # Global single-CTA exact tournament when P_sz <= 256 (<0.25ms on 1 SM)
        if HAS_TRITON and device.type == "cuda" and P_sz <= 256:
            cp_c = c_pool - c_pool.mean(dim=0, keepdim=True)
            pn = cp_c.square().sum(dim=1)
            pair_d2 = (pn[:, None] + pn[None, :] - 2.0 * (cp_c @ cp_c.t())).clamp_min_(0.0).contiguous()
            pair_d2.fill_diagonal_(0.0)
            w_p = pool_weights.float()
            w_p = (w_p / w_p.sum().clamp_min(1e-30)).contiguous()
            pd2 = init_pool_d2.float().clone().contiguous()
            sel_loc = torch.empty((num_select,), dtype=torch.int64, device=device)
            with torch.cuda.device(device):
                _apex_local_greedy_kpp_kernel[(1,)](
                    pair_d2,
                    pd2,
                    w_p,
                    sel_loc,
                    P_sz,
                    num_select,
                    int(rseed) % (2**31 - 1),
                    BLOCK_P=256,
                    L_TRIALS=3,
                    num_warps=4,
                )
            return c_pool[sel_loc], sel_loc

        # Multi-CTA Voronoi-neighborhood sharded tournament across SMs when P_sz > 256 (~0.7ms at K=4096!)
        if HAS_TRITON and device.type == "cuda" and P_sz > 256:
            g_cnt = max(2, (P_sz + 127) // 128)
            while (P_sz // g_cnt) < (num_select // g_cnt) or max(P_sz // g_cnt, P_sz - (P_sz // g_cnt) * (g_cnt - 1)) > 128:
                g_cnt += 1
            sel_per_g = num_select // g_cnt
            p_per_g = P_sz // g_cnt
            last_sel = num_select - sel_per_g * (g_cnt - 1)
            last_p = P_sz - p_per_g * (g_cnt - 1)
            if p_per_g >= sel_per_g and last_p >= last_sel and max(p_per_g, last_p) <= 128:
                if coarse_anchors is not None and coarse_anchors.shape[0] >= 16:
                    anc = coarse_anchors[: min(256, int(coarse_anchors.shape[0]))]
                    anc_n = anc.square().sum(dim=1)
                    cp_n = c_pool.square().sum(dim=1)
                    anc_sc = torch.addmm(anc_n[None, :], c_pool, anc.t(), alpha=-2.0) + cp_n[:, None]
                    vor_id = anc_sc.argmin(dim=1)
                    pgen = torch.Generator(device=device)
                    pgen.manual_seed(int(rseed + 33) % (2**32))
                    rdir = torch.randn((d,), generator=pgen, device=device, dtype=torch.float32)
                    proj_tie = (c_pool @ rdir)
                    proj_span = (proj_tie.max() - proj_tie.min()).clamp_min(1.0)
                    sort_key = vor_id.float() * 4.0 + (proj_tie / proj_span)
                    order_p = torch.argsort(sort_key)
                else:
                    pgen = torch.Generator(device=device)
                    pgen.manual_seed(int(rseed + 33) % (2**32))
                    rdir = torch.randn((d,), generator=pgen, device=device, dtype=torch.float32)
                    order_p = torch.argsort(c_pool @ rdir)

                cp_s = c_pool[order_p].contiguous()
                pd2_s = init_pool_d2[order_p].float().contiguous()
                wp_s = pool_weights[order_p].float()
                wp_s = (wp_s / wp_s.sum().clamp_min(1e-30)).contiguous()

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
                    _apex_batched_local_greedy_kpp_kernel[(g_cnt,)](
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
                return cp_s[sel_out], order_p[sel_out]

        from .seeding import _select_from_candidate_pool

        chosen_cpu = _select_from_candidate_pool(c_pool, init_pool_d2, pool_weights, num_select, seed=rseed)
        sel_cpu = torch.cdist(chosen_cpu.float(), c_pool.float()).argmin(dim=1)
        return chosen_cpu, sel_cpu

    arange_m = torch.arange(m, device=device, dtype=torch.int64)
    h_bits_shift = (1 << torch.arange(9, device=device, dtype=torch.int64))[None, :]

    def _diverse_far_leaders(
        d2_vec: torch.Tensor,
        mask_avail: torch.Tensor,
        n_leaders: int,
        h_seed: int,
        fast_mode: bool = False,
    ) -> torch.Tensor:
        """Select up to `n_leaders` spatially deduplicated farthest points via 512-orthant leader reduction (`0` syncs)."""
        if n_leaders <= 0:
            return torch.empty((0,), dtype=torch.int64, device=device)
        masked_d2 = torch.where(mask_avail, d2_vec.clamp_min(0.0), torch.full_like(d2_vec, -1.0))
        if fast_mode or d < 2:
            return torch.topk(masked_d2, k=min(m, n_leaders), largest=True, sorted=False).indices
        n_cand_far = min(m, max(n_leaders * 8, 256))
        top_raw = torch.topk(masked_d2, k=n_cand_far).indices
        if n_cand_far <= n_leaders:
            return top_raw[:n_leaders]
        xc_far = xs_c[top_raw]
        mu_far = xc_far[: min(24, n_cand_far)].mean(dim=0, keepdim=True)
        hgen = torch.Generator(device=device)
        hgen.manual_seed(int(h_seed) % (2**32))
        r_hash = torch.randn((d, 9), generator=hgen, device=device, dtype=torch.float32)
        b_id = (((xc_far - mu_far) @ r_hash > 0.0).to(torch.int64) * h_bits_shift).sum(dim=1)  # (n_cand_far,) in 0..511
        d2_top_bits = d2_vec[top_raw].clamp_min(0.0).float().view(torch.int32).to(torch.int64)
        packed = (d2_top_bits << 32) | top_raw
        valid_top = mask_avail[top_raw]
        packed_masked = torch.where(valid_top, packed, torch.full_like(packed, -1))
        b_max = torch.full((512,), -1, dtype=torch.int64, device=device)
        b_max.scatter_reduce_(0, b_id, packed_masked, reduce="amax", include_self=True)
        k_take = min(n_leaders, 512)
        top_buckets = torch.topk(b_max, k=k_take).values
        # Fall back cleanly to `top_raw[0]` for any empty orthant slots (`top_buckets < 0`)
        leader_idx = torch.where(top_buckets >= 0, top_buckets & 0xFFFFFFFF, top_raw[0])
        return leader_idx

    use_huge_k_fast_seed = bool(k >= 16384 and device.type == "cuda" and HAS_TRITON and m >= k)
    m_coarse_seed = min(256, max(128, round_power_of_two(math.sqrt(float(max(16, k))))))

    if k <= 128 and m <= 16384:
        from .seeding import greedy_kmeans_plusplus_torch

        raw_c = greedy_kmeans_plusplus_torch(
            xs_c, k, weights=w_sub, include_farthest_candidate=True, seed=seed + 17
        ).contiguous()
    elif use_huge_k_fast_seed:
        # Coarse-bucketed (`M = 1*sqrt(K) in [128, 256]`) stratified k-means++ seeding (`#189`: ~5-9ms at K=16K..65K)
        rad_d2 = xs_xnorm if xs_xnorm is not None else xs_c.square().sum(dim=1)
        rmax0 = rad_d2.max().clamp_min(1e-30)
        prop_anc = w_sub * (0.25 + 0.75 * (rad_d2 / rmax0)) + 1e-15
        p_anc = min(m, max(m_coarse_seed * 3, 512))
        ug_a = torch.rand((m,), generator=gen, device=device, dtype=torch.float32).clamp_(1e-30, 1.0 - 1e-7)
        cand_anc = torch.topk(prop_anc.log() - (-ug_a.log()).log(), k=p_anc, largest=True, sorted=False).indices
        cp_anc = xs_c[cand_anc].contiguous()
        init_d2_a = torch.where(rad_d2[cand_anc].max() > 0.0, rad_d2[cand_anc], torch.ones_like(rad_d2[cand_anc]))
        c_coarse, _ = _apex_select_pool(
            cp_anc, init_d2_a, w_sub[cand_anc].clamp_min(1e-12), m_coarse_seed, rseed=seed + 311
        )
        d2_c, cid_c = _seed_assign_pool(c_coarse)
        b_dmax = torch.full((m_coarse_seed,), 1e-30, dtype=torch.float32, device=device)
        b_dmax.scatter_reduce_(0, cid_c, d2_c.clamp_min(0.0), reduce="amax", include_self=True)
        b_mass = torch.zeros((m_coarse_seed,), dtype=torch.float32, device=device)
        b_mass.index_add_(0, cid_c, w_sub)
        rel_in_b = (d2_c.clamp_min(0.0) / b_dmax[cid_c].clamp_min(1e-30)).clamp(0.0, 1.0)
        prop_k = w_sub * (0.20 + 0.80 * rel_in_b) * b_mass[cid_c].clamp_min(1e-12).pow(-0.20) + 1e-15
        far_k = _diverse_far_leaders(d2_c, avail, min(64, m), seed + 307, fast_mode=False)
        prop_k.scatter_(0, far_k, (prop_k.max().clamp_min(1e-12) * 1e6).expand_as(far_k))
        ug_k = torch.rand((m,), generator=gen, device=device, dtype=torch.float32).clamp_(1e-30, 1.0 - 1e-7)
        cand_k = torch.topk(prop_k.clamp_min(1e-30).log() - (-ug_k.log()).log(), k=k, largest=True, sorted=False).indices
        raw_c = xs_c[cand_k].contiguous()
    else:
        b0 = min(k, 16)
        p0 = min(m, max(b0, min(160, b0 * 8)))
        rad_d2 = xs_xnorm if xs_xnorm is not None else xs_c.square().sum(dim=1)
        rmax0 = rad_d2.max().clamp_min(1e-30)
        prop0 = w_sub * (0.25 + 0.75 * (rad_d2 / rmax0)) + 1e-15
        n_far0 = min(32, m)
        far0 = _diverse_far_leaders(rad_d2, avail, n_far0, seed + 307)
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
            _, pool_lbls0 = _seed_assign_pool(cp0)
            w_pool0 = torch.zeros((p0,), dtype=torch.float32, device=device)
            w_pool0.index_add_(0, pool_lbls0, w_sub)
            w_pool0.clamp_min_(1e-12 / float(p0))
            c0_blk, sel_p0 = _apex_select_pool(cp0, init_d2_0, w_pool0, b0, rseed=seed + 501)
            avail.scatter_(0, cand0[sel_p0], false_buf[:b0])

        raw_c_list: list[torch.Tensor] = [c0_blk]
        placed = b0
        if placed < k:
            min_d2, _ = _seed_assign_pool(c0_blk)
        else:
            min_d2 = rad_d2

        # Participation Ratio `d_eff`: on strongly anisotropic multi-scale physical corpora (`d_eff < 6.0` for `d >= 16`),
        # sharpening `d_rel` to `d_rel^1.35` spreads tournament proposals across dominant physical axes.
        if d >= 16 and m >= 32:
            var_d = xs_c.var(dim=0, unbiased=False)
            d_eff_pr = float((var_d.sum().square() / var_d.square().sum().clamp_min(1e-30)).item())
            d_rel_exp = 1.35 if d_eff_pr < 6.0 else 1.0
        else:
            d_rel_exp = 1.0

        max_round_blk = 4096 if k >= 8192 else (2048 if k >= 4096 else 640)
        cur_blk = 64 if k >= 512 else 32
        round_id = 0
        while placed < k:
            b_r = min(k - placed, min(max_round_blk, cur_blk))
            oversamp = 1.45 if b_r > 640 else (1.65 if b_r > 128 else 2.0)
            extra_pad = 0 if b_r > 640 else 32
            max_pool_cap = 6144 if b_r > 640 else 1024
            pool_n = min(m - placed, min(max_pool_cap, max(b_r, int(round(oversamp * b_r)) + extra_pad)))

            dmax = torch.where(avail, min_d2, torch.zeros_like(min_d2)).max().clamp_min(1e-30)
            d_rel = (min_d2 / dmax).clamp(0.0, 1.0)
            d_rel_prop = d_rel.clamp_min(1e-12).pow(d_rel_exp) if d_rel_exp != 1.0 else d_rel
            prop = torch.where(avail, w_sub * d_rel_prop + 1e-15, torch.zeros_like(w_sub))
            n_far = min(64, max(16, b_r // 2), m)
            far_ids = _diverse_far_leaders(
                d_rel, avail, n_far, seed + 701 + round_id * 37, fast_mode=(round_id >= 4)
            )
            max_prop = prop.max().clamp_min(1e-12)
            far_boost = torch.where(avail.gather(0, far_ids), max_prop * 1e6, torch.zeros_like(far_ids, dtype=torch.float32))
            prop.scatter_(0, far_ids, far_boost)

            u_g = torch.rand((m,), generator=gen, device=device, dtype=torch.float32).clamp_(1e-30, 1.0 - 1e-7)
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
                _, pool_lbls = _seed_assign_pool(c_pool)
                w_pool = torch.zeros((pool_n,), dtype=torch.float32, device=device)
                w_pool.index_add_(0, pool_lbls, w_sub)
                w_pool.clamp_min_(1e-12 / float(pool_n))
                coarse_anc = (
                    torch.cat(raw_c_list[: min(3, len(raw_c_list))], dim=0)
                    if pool_n > 256
                    else None
                )
                chosen_c, sel_pr = _apex_select_pool(
                    c_pool, pool_d2, w_pool, b_r, rseed=seed + 1000 + round_id, coarse_anchors=coarse_anc
                )
                avail.scatter_(0, cand_idx[sel_pr], false_buf[:b_r])

            if placed + b_r < k:
                new_d2, _ = _seed_assign_pool(chosen_c)
                min_d2 = torch.minimum(min_d2, new_d2)

            raw_c_list.append(chosen_c)
            placed += b_r
            growth_mult = 2 if cur_blk < 256 else 4
            cur_blk = min(max_round_blk, cur_blk * growth_mult)
            round_id += 1

        raw_c = raw_c_list[0][:k].contiguous() if len(raw_c_list) == 1 else torch.cat(raw_c_list, dim=0)[:k].contiguous()

    # Fused 1-pass Voronoi polish + Huygens parallel-axis exact distortion extraction (`0` [m, D] diff copy!)
    if use_huge_k_fast_seed and m >= 32768:
        ivf_seed = _ApexSyncFreeIVFExplorer(raw_c, nlist=m_coarse_seed, seed=seed + 919)
        sq_d0, labels = ivf_seed.assign_batch(xs_c, nprobe=8)
    else:
        sq_d0, labels = _seed_assign_pool(raw_c)
    mass = torch.zeros((k,), dtype=torch.float32, device=device)
    dist_m0 = torch.zeros((k,), dtype=torch.float32, device=device)
    sums = torch.zeros((k, d), dtype=torch.float32, device=device) if polish else raw_c

    if device.type == "cuda" and HAS_TRITON:
        blk_m = 128
        blk_d = min(128, max(16, triton.next_power_of_2(max(1, d))))
        grid_d = triton.cdiv(d, blk_d) if polish else 1
        with torch.cuda.device(device):
            _apex_voronoi_polish_scatter_kernel[(triton.cdiv(m, blk_m), grid_d)](
                xs_c,
                w_sub,
                labels,
                sq_d0,
                mass,
                dist_m0,
                sums,
                m,
                d,
                xs_c.stride(0),
                sums.stride(0),
                DO_SUMS=polish,
                BLOCK_M=blk_m,
                BLOCK_D=blk_d,
                num_warps=4,
            )
    else:
        mass.index_add_(0, labels, w_sub)
        dist_m0.index_add_(0, labels, w_sub * sq_d0)
        if polish:
            sums.index_add_(0, labels, xs_c * w_sub[:, None])

    tot_m = mass.sum().clamp_min(1e-30)
    if not polish:
        return SeedingResult(
            centroids=raw_c,
            prior_probs=(mass / tot_m).contiguous(),
            prior_distortion=(dist_m0 / tot_m).contiguous(),
        )

    nonempty = mass > 0.0
    polished = torch.where(
        nonempty[:, None],
        sums / mass.clamp_min(1e-30)[:, None],
        raw_c,
    ).contiguous()

    # Exact Huygens parallel-axis identity: sum_{i in C_j} w_i ||x_i - mu_j||^2 = dist_m0[j] - mass[j] * ||mu_j - raw_c[j]||^2
    shift_sq = (polished - raw_c).square().sum(dim=1)
    dist_pol = (dist_m0 - mass * shift_sq).clamp_min_(0.0)
    return SeedingResult(
        centroids=polished,
        prior_probs=(mass / tot_m).contiguous(),
        prior_distortion=(dist_pol / tot_m).contiguous(),
    )


# ==============================================================================
# 4. Pre-Closing Independent Anchor + Positive-Part James-Stein Simplex EB
# ==============================================================================


@torch.no_grad()
def apex_simplex_eb_finalize(
    pass_sums: torch.Tensor,
    pass_counts: torch.Tensor,
    pass_counts_sq: torch.Tensor | None,
    pass_sse: torch.Tensor,
    live_centroids: torch.Tensor,
    pre_close_centroids: torch.Tensor | None,
    slow_centroids: torch.Tensor | None,
    usage: torch.Tensor,
    usage_inv_bias: float,
    alpha_end: float,
    nepochs: float,
    d: int,
    mode: str = "simplex_eb",
    eb_tau: float = 0.25,
    kappa_max: float = 8.0,
) -> tuple[torch.Tensor, ShrinkageDiagnostics]:
    """Pre-Closing Independent Anchor + Positive-Part James-Stein Simplex EB Finalizer.

    Combines:
      1. `pre_close_centroids` (captured right before closing pass, updated on splits/transplants):
         100% statistically independent of closing-pass noise for `T >= 1.5`, with zero step-0 lag!
      2. `slow_centroids`: blended into the independent anchor when stationary (`d_ps_sq <= v_es`).
      3. Exact Positive-Part James-Stein `min(kappa_max / (n_eff + kappa_max), v_close / diff_sq)`
         on the independent pre-closing anchor + correlation-safe `eb_tau` on `live_centroids`.
    """
    counts_w = pass_counts.double()
    nonempty = counts_w > 0.0

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

    if mode == "eb_ema" or (slow_centroids is None and pre_close_centroids is None):
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

    if mode == "eb_slow_only" and slow_centroids is not None:
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
    n_batch_hat = (usage.double() * float(usage_inv_bias)).clamp_min(1.0)
    a_end = min(0.99, max(1e-4, float(alpha_end)))
    v_es = (a_end / (2.0 - a_end)) * (sigma2_j / n_batch_hat)

    # Construct optimal pre-closing / slow anchor
    if pre_close_centroids is not None and nepochs >= 1.25:
        if slow_centroids is not None:
            d_ps_sq = (pre_close_centroids.double() - slow_centroids.double()).square().sum(dim=1)
            w_s_in_anc = 0.5 * (v_es / torch.maximum(d_ps_sq, v_es).clamp_min(1e-30)).clamp(0.0, 1.0)
            indep_anchor = (
                (1.0 - w_s_in_anc)[:, None] * pre_close_centroids.double()
                + w_s_in_anc[:, None] * slow_centroids.double()
            )
        else:
            indep_anchor = pre_close_centroids.double()
        phi_indep = min(1.0, max(0.50, (float(nepochs) - 1.0) / 0.75))
    elif slow_centroids is not None:
        indep_anchor = slow_centroids.double()
        # At T <= 1.0, `pre_close_centroids` (if mid-pass snapshot) or `slow_centroids` only saw first half -> discount covariance
        if pre_close_centroids is not None:
            indep_anchor = 0.65 * pre_close_centroids.double() + 0.35 * slow_centroids.double()
            phi_indep = 0.55
        else:
            phi_indep = 0.35
    else:
        indep_anchor = pre_close_centroids.double()  # type: ignore[union-attr]
        phi_indep = 0.55

    d_ci_sq = (c_close.double() - indep_anchor).square().sum(dim=1)
    k_max = max(0.0, float(kappa_max))
    if nepochs <= 3.5:
        # At T <= 3.5, dense clusters (`n_eff >= 160`) are drift-limited rather than variance-limited,
        # whereas sparse tail clusters (`n_eff < 64`) benefit from full `kappa_max` Stein shrinkage.
        k_max_vec = k_max * ((160.0 - n_eff) / 96.0).clamp(0.0, 1.0)
    else:
        k_max_vec = torch.full_like(n_eff, k_max)
    max_indep_shrink = k_max_vec / (n_eff + k_max_vec).clamp_min(1e-30)
    # Exact Positive-Part James-Stein shrinkage `v_close / d_ci_sq` capped by `k_max_vec / (n_eff + k_max_vec)`
    js_indep_shrink = (v_close / d_ci_sq.clamp_min(1e-30)).clamp(0.0, 1.0)
    w_indep = phi_indep * torch.minimum(max_indep_shrink, js_indep_shrink)

    tau = max(0.0, float(eb_tau))
    w_ema = (1.0 - w_indep).clamp_min(0.0) * (1.0 - phi_indep) * (tau / (n_eff + tau).clamp_min(1e-30))

    zero_all = (global_sigma2 <= 0.0) & (torch.where(nonempty, d_ci_sq, 0.0).max() <= 0.0)
    tot_shrink = torch.where(zero_all, torch.zeros_like(n_eff), (w_indep + w_ema).clamp(0.0, 1.0))
    w_close = 1.0 - tot_shrink

    scale_sh = torch.where(w_indep + w_ema > 1e-30, tot_shrink / (w_indep + w_ema).clamp_min(1e-30), 0.0)
    w_indep = w_indep * scale_sh
    w_ema = w_ema * scale_sh

    c_fin64 = (
        w_close[:, None] * c_close.double()
        + w_ema[:, None] * live_centroids.double()
        + w_indep[:, None] * indep_anchor
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
    excess = torch.clamp(d_ci_sq - v_close, min=0.0)
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
# 5. Fast Dispersion-Gated SRAM `P/Q` `head_edit` Checkpoint & Sync-Free IVF
# ==============================================================================


@torch.no_grad()
def _sram_all_pairs_centered_nn(centroids_c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute `(d_nn2, nn_idx)` in SRAM via Triton (`0` KxK HBM allocation, `~0.12 ms` at K=2048)."""
    k, d = centroids_c.shape
    device = centroids_c.device
    if k == 0:
        return torch.empty((0,), dtype=torch.float32, device=device), torch.empty((0,), dtype=torch.int64, device=device)
    if k == 1:
        return torch.full((1,), float("inf"), dtype=torch.float32, device=device), torch.full((1,), -1, dtype=torch.int64, device=device)
    if not centroids_c.is_cuda or not HAS_TRITON:
        from .super_emeans import _all_pairs_centered_nn

        return _all_pairs_centered_nn(centroids_c)

    c_mean = centroids_c.mean(dim=0, keepdim=True)
    kc = (centroids_c - c_mean).contiguous()
    c_pad = _alloc_padded_2d(k, d, torch.float32, device, align=16)
    c_pad[:, :d] = kc
    cn = kc.square().sum(dim=1).contiguous()
    out_d2 = torch.empty((k,), dtype=torch.float32, device=device)
    out_id = torch.empty((k,), dtype=torch.int64, device=device)
    blk_q = 64 if k >= 64 else 16
    blk_k = 64 if k >= 64 else 16
    blk_d = min(64, max(16, triton.next_power_of_2(max(1, d))))
    with torch.cuda.device(device):
        _apex_all_pairs_nn_kernel[(triton.cdiv(k, blk_q),)](
            c_pad,
            cn,
            out_d2,
            out_id,
            k,
            d,
            c_pad.stride(0),
            BLOCK_Q=blk_q,
            BLOCK_K=blk_k,
            BLOCK_D=blk_d,
            num_warps=4,
        )
    return out_d2, out_id


@torch.no_grad()
def run_apex_head_edit_checkpoint(
    x_batch_c: torch.Tensor,
    centroids_c: torch.Tensor,
    usage: torch.Tensor,
    sse_ema: torch.Tensor,
    ref_distortion_mean: torch.Tensor,
    survival_mass: torch.Tensor | None,
    prin_vec: torch.Tensor | None,
    slow_centroids: torch.Tensor | None,
    slow_usage: torch.Tensor | None,
    pre_close_centroids: torch.Tensor | None,
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
    """Fast Dispersion-Gated Oja-Seeded `P/Q` Transplant (`~0.02 ms` early-exit on isotropic data!)."""
    bsz_in, d = x_batch_c.shape
    k = centroids_c.shape[0]
    if k < 3 or bsz_in < 16 or max_pairs < 1 or num_probes < 1:
        return 0

    # Cap evaluation batch to `max(2048, 3 * K)` so wide batches don't spend unnecessary Top-3 FLOPs
    max_eval_b = max(2048, min(12288, 3 * k))
    xb = x_batch_c[:max_eval_b] if bsz_in > max_eval_b else x_batch_c
    bsz = int(xb.shape[0])
    half = bsz // 2
    if half < 8 or (bsz - half) < 8:
        return 0

    # Fast 2-float dispersion pre-gate: skip O(B*K*D) top-3 when no cluster has excess distortion skew
    if sse_ema.numel() == k:
        gate_pair = torch.stack([sse_ema.max(), sse_ema.mean(), usage.min(), usage.mean()]).tolist()
        if gate_pair[1] > 0.0 and gate_pair[0] < 1.65 * gate_pair[1] and gate_pair[2] > 0.35 * gate_pair[3]:
            return 0

    diag.checks_run += 1
    device = xb.device
    tc_dtype = (
        compute_dtype
        if compute_dtype in (torch.float32, torch.bfloat16, torch.float16, torch.float64)
        else torch.bfloat16
    )
    rep_dtype = tc_dtype if tc_dtype in (torch.float32, torch.bfloat16, torch.float16, torch.float64) else torch.float32
    scale_full = float(bsz) / float(half)

    top_d, top_i, origin, xc_all, x_norm_all, cc = fast_top3_centered_assign(
        xb, centroids_c, tc_dtype, already_centered=True
    )
    cn = cc.square().sum(dim=1)
    d_nn2, nn_idx = _sram_all_pairs_centered_nn(centroids_c)

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
        rec_cost_masked = torch.where(rec_cand_mask, eff_rem_cur, torch.full_like(eff_rem_cur, float("inf")))
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

        dirs_had = sample_signed_hadamard_rows(num_probes, d, generator=generator, device=device, unit_l2=True)
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
        if tc_dtype in (torch.float32, torch.bfloat16, torch.float16):
            separate_collapsed_directional(new_s, new_d, u_win, tc_dtype)
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
            s_t = torch.tensor([s_idx], dtype=torch.int64, device=device)
            d_t = torch.tensor([d_idx], dtype=torch.int64, device=device)
            reset_post_split_oja_axes(prin_vec, s_t, d_t, u_win, generator=generator)
        if slow_centroids is not None and slow_usage is not None:
            slow_centroids[s_idx] = new_s[0]
            slow_centroids[d_idx] = new_d[0]
            shalf = 0.5 * slow_usage[s_idx]
            slow_usage[s_idx] = shalf
            slow_usage[d_idx] = shalf
        if pre_close_centroids is not None:
            pre_close_centroids[s_idx] = new_s[0]
            pre_close_centroids[d_idx] = new_d[0]
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


class _ApexSyncFreeIVFExplorer:
    """Zero-host-sync Coarse-to-Fine GPU IVF Explorer for Huge-K (`K >= 24,576`) exploration passes."""

    def __init__(self, centroids_c: torch.Tensor, nlist: int, seed: int = 1234) -> None:
        from .fast_ops import _dtype_mode, _ivf_inverted_list_kernel, blackwell_prepare_centroids, blackwell_triton_assign

        self._dtype_mode = _dtype_mode
        self._ivf_kernel = _ivf_inverted_list_kernel
        self._bw_prep = blackwell_prepare_centroids
        self._bw_assign = blackwell_triton_assign

        k, d = centroids_c.shape
        device = centroids_c.device
        self.d = int(d)
        self.nlist = min(int(nlist), k)
        gen = torch.Generator(device=device)
        gen.manual_seed((int(seed) + 424242) % (2**32))
        perm = torch.randperm(k, generator=gen, device=device)[: self.nlist]
        meta_c = centroids_c[perm].clone().contiguous()

        # 2 fast Lloyd steps on `centroids_c` itself so every meta-centroid sits inside a populated cluster neighborhood!
        sums = torch.empty((self.nlist, d), dtype=torch.float32, device=device)
        for _ in range(2):
            mc_rep, mc_norm = self._bw_prep(meta_c, torch.bfloat16)
            _, lbl = self._bw_assign(centroids_c, mc_rep, mc_norm)
            cnt = torch.bincount(lbl, minlength=self.nlist).float()
            sums.zero_()
            sums.index_add_(0, lbl, centroids_c)
            meta_c = torch.where((cnt > 0)[:, None], sums / cnt.clamp_min(1.0)[:, None], meta_c)

        self.meta_c = meta_c.contiguous()
        self.meta_c_rep, self.meta_c_norm = self._bw_prep(self.meta_c, torch.bfloat16)
        self.bucket_ids_arange = torch.arange(self.nlist + 1, dtype=torch.int64, device=device)
        self.c_sorted: torch.Tensor | None = None
        self.cn_sorted: torch.Tensor | None = None
        self.c_orig_id: torch.Tensor | None = None
        self.c_offsets: torch.Tensor | None = None
        self.c_rep_bf16: torch.Tensor | None = None
        self.c_norm_bf16: torch.Tensor | None = None
        self.rebuild(centroids_c)

    @torch.no_grad()
    def rebuild(self, centroids_c: torch.Tensor) -> None:
        """Partition `centroids_c` into `nlist` buckets with ZERO `.item()` host syncs."""
        c_rep, c_norm = self._bw_prep(centroids_c, torch.bfloat16)
        self.c_rep_bf16 = c_rep
        self.c_norm_bf16 = c_norm
        _, c_buckets = self._bw_assign(centroids_c, self.meta_c_rep, self.meta_c_norm)
        perm = torch.argsort(c_buckets)
        sorted_b = c_buckets[perm]
        self.c_sorted = c_rep[perm].contiguous()
        self.cn_sorted = c_norm[perm].contiguous()
        self.c_orig_id = perm.to(torch.int32).contiguous()
        self.c_offsets = torch.searchsorted(sorted_b, self.bucket_ids_arange).to(torch.int32).contiguous()

    @torch.no_grad()
    def assign_batch(
        self,
        xb_c: torch.Tensor,
        nprobe: int = 12,
        out_dist: torch.Tensor | None = None,
        out_label: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Assign pre-centered batch `xb_c` via top-`nprobe` IVF buckets (`0` host syncs)."""
        B = int(xb_c.shape[0])
        device = xb_c.device
        if out_dist is None:
            out_dist = torch.empty((B,), dtype=torch.float32, device=device)
        if out_label is None:
            out_label = torch.empty((B,), dtype=torch.int64, device=device)
        if B == 0:
            return out_dist, out_label

        eff_nprobe = max(1, min(int(nprobe), self.nlist))
        x_rep = xb_c.to(torch.bfloat16).contiguous()
        x_norm = x_rep.float().square().sum(dim=1)
        coarse_scores = torch.addmm(
            self.meta_c_norm[None, :],
            x_rep,
            self.meta_c_rep.t(),
            alpha=-2.0,
            out_dtype=torch.float32,
        )
        # Mask empty IVF buckets (`c_offsets[b+1] == c_offsets[b]`) to `+inf` BEFORE `topk` so EVERY probed
        # bucket is guaranteed non-empty (`best_score < inf` 100% of the time with ZERO host sync!)
        assert self.c_offsets is not None and self.c_sorted is not None and self.cn_sorted is not None and self.c_orig_id is not None
        b_counts = self.c_offsets[1:] - self.c_offsets[:-1]
        coarse_scores = torch.where((b_counts > 0)[None, :], coarse_scores, float("inf"))

        _, top_buckets = torch.topk(coarse_scores, k=eff_nprobe, dim=1, largest=False, sorted=False)
        flat_buckets = top_buckets.reshape(-1)
        m_pairs = B * eff_nprobe

        pair_perm = torch.argsort(flat_buckets)
        sorted_buckets = flat_buckets[pair_perm]
        q_offsets = torch.searchsorted(sorted_buckets, self.bucket_ids_arange).to(torch.int32)

        block_b = 64
        block_k = 128
        d_mode = self._dtype_mode(torch.bfloat16)
        full_d = self.d <= 128
        block_d = max(16, triton.next_power_of_2(max(1, self.d))) if full_d else 64
        ivf_stages = 3
        ivf_smem = ((0 if full_d else block_b) + block_k) * block_d * 2
        while ivf_stages > 1 and ivf_smem * ivf_stages > 81920:
            ivf_stages -= 1

        q_counts = q_offsets[1:] - q_offsets[:-1]
        tile_counts = (q_counts + block_b - 1) // block_b
        tile_prefix = torch.cumsum(tile_counts, dim=0).to(torch.int32)
        max_tiles = triton.cdiv(m_pairs, block_b) + self.nlist

        cand_scores = torch.full((m_pairs,), float("inf"), dtype=torch.float32, device=device)
        cand_labels = torch.zeros((m_pairs,), dtype=torch.int32, device=device)
        with torch.cuda.device(device):
            self._ivf_kernel[(max_tiles,)](
                x_rep,
                None,
                pair_perm,
                q_offsets,
                tile_prefix,
                self.c_sorted,
                self.cn_sorted,
                self.c_orig_id,
                self.c_offsets,
                cand_scores,
                cand_labels,
                self.d,
                self.nlist,
                eff_nprobe,
                x_rep.stride(0),
                x_rep.stride(1),
                self.c_sorted.stride(0),
                self.c_sorted.stride(1),
                BLOCK_B=block_b,
                BLOCK_K=block_k,
                BLOCK_D=block_d,
                FULL_D=full_d,
                DTYPE_MODE=d_mode,
                HAS_INDICES=False,
                num_warps=4 if full_d else 8,
                num_stages=ivf_stages,
            )

        scores_2d = cand_scores.view(B, eff_nprobe)
        labels_2d = cand_labels.view(B, eff_nprobe)
        best_score, best_slot = scores_2d.min(dim=1)
        out_label.copy_(labels_2d.gather(1, best_slot[:, None]).squeeze(1).to(torch.int64))
        out_dist.copy_((best_score + x_norm).clamp_min_(0.0))
        return out_dist, out_label


# ==============================================================================
# 6. `ApexEMeans`: Unified 4-Gear Scale-Adaptive Flagship Engine
# ==============================================================================


@dataclass
class ApexEMeansEpochStat:
    epoch: int
    rows: int
    steps: int
    moving_mse: float
    imbalance_factor: float
    n_splits: int
    n_protected: int
    n_transplants: int
    elapsed_sec: float


def _make_superstep_ring_slots(num_substeps: int) -> list[tuple[int, int]]:
    """Construct a race-free ping-pong `(in_slot, out_slot)` sequence over `{0, 1, 2}` starting & ending at `0`."""
    if num_substeps < 2:
        raise ValueError("Super-step CUDAGraph requires num_substeps >= 2")
    if num_substeps % 2 == 0:
        return [(0, 1) if (i % 2 == 0) else (1, 0) for i in range(num_substeps)]
    slots: list[tuple[int, int]] = []
    for i in range(num_substeps - 3):
        slots.append((0, 1) if (i % 2 == 0) else (1, 0))
    # Last 3 sub-steps start at 0 (`num_substeps - 3` is even) and close `0 -> 1 -> 2 -> 0`
    slots.extend([(0, 1), (1, 2), (2, 0)])
    return slots


class ApexEMeans:
    """Unified 4-Gear Scale-Adaptive Clustering Engine (`ApexEMeans`)."""

    def __init__(
        self,
        d: int,
        k: int,
        nepochs: float = 5.0,
        *,
        # 4-Gear Scale Adaptation & Precision
        gear_mode: str = "auto",              # "auto" | "small" | "medium_fp8" | "huge_k_ivf" | "multi_gpu" | "exact"
        precision: str = "auto",              # "auto" | "croiss_fp8" | "fp8" | "bf16" | "fp16" | "fp32" | "fp64"
        devices: list[int | str | torch.device] | None = None,
        # Batch & Streaming Layer
        batch_law: str = "auto",
        batch_size: int | None = None,
        alpha0: float | None = None,
        alpha_min_factor: float = ALPHA_MIN_FACTOR,
        use_wide_batch: bool | str = "auto",  # True | False | "auto"
        wide_step_mult: int = 2,
        wide_from_frac: float = 0.25,
        streaming_mode: str = "auto",         # "auto" | "cyclic_block" | "pre_permuted" | "fused_gather"
        block_permute_size: int = 64,
        preencode_dataset: bool | str = "auto",
        # Kernel, Fusion & CUDA-Graph Layer
        use_fused_pipeline: bool = True,
        use_sm120_kernel: bool = True,
        use_cuda_graph: bool = True,
        graph_super_steps: int = 5,
        # Coarse-to-Fine IVF Exploration Gear (for Huge K)
        ivf_explore: bool | str = "auto",     # "auto" | True | False
        ivf_k_threshold: int = 16384,
        ivf_nlist: int | None = None,
        ivf_nprobe: int = 8,
        ivf_rebuild_every: int = 5,
        # Seeding Layer
        init_mode: str = "apex_turkpp",       # "apex_turkpp" | "ultra_turkpp" | "subsampled_d15" | "random" | "coreset_kpp"
        coreset_mult: int | str = "auto",
        min_coreset_size: int = 4096,
        init_rounds: int = 4,
        init_power: float = 1.5,
        use_voronoi_polish: bool = True,
        use_prior_mass: bool = True,
        prior_mass_scale: float = 0.5,
        prior_floor: float = 1e-3,
        warm_alpha_scale: float | None = None,
        # Schedule, Revival, Donors & Huygens Shield
        schedule_mode: str = "emeans_cosine",  # "emeans_cosine" | "power_cosine" | "constant"
        alpha_floor: float = 0.06,
        cosine_power: float = 0.92,
        unused_threshold: float | None = None,
        check_unused_every: int = CHECK_UNUSED_EVERY,
        init_usage: float = DEFAULT_INIT_USAGE,
        lowd_revival: bool = True,
        revival_mode: str = "split",           # "split" | "off"
        use_revive_stop: bool = True,
        revive_stop_frac: float = 0.85,
        flush_pass_on_split: bool = True,
        use_huygens_protection: bool = True,
        gamma_protect: float = 0.5,
        grace_steps: int = 4,
        donor_mode: str = "oja_sse_usage",     # "oja_sse_usage" | "sse_usage" | "d2" | "usage_only"
        donor_usage_w: float = 0.35,
        couple_sse_to_alpha: bool = True,
        sse_decay: float = 0.85,
        # Spectral-Concentration-Gated Half-Gaussian Oja Split
        split_mode: str = "half_gaussian_oja",  # "half_gaussian_oja" | "rms_aniso" | "rms_iso" | "oja_only" | "checkerboard"
        split_mult: float = HALF_GAUSSIAN_MEAN,
        use_feat_scale: bool = True,
        var_beta: float = 0.85,
        split_epsilon: float = SPLIT_EPSILON,
        # Fast Dispersion-Gated `P/Q` `head_edit` Transplants
        head_edit: bool | str = "auto",
        transplant_checkpoints: tuple[float, ...] = (0.62,),
        transplant_probes: int = 6,
        transplant_max_pairs: int = 4,
        transplant_protect_steps: int = 6,
        # Pre-Closing Independent Anchor + Positive-Part James-Stein Simplex EB
        use_pass_weighting: bool = False,
        adaptive_pass_weighting: bool = True,
        pass_w_span: float = 0.20,
        finalization: str = "simplex_eb",      # "simplex_eb" | "eb_slow_only" | "eb_ema" | "closing_mean" | "ema_only"
        eb_tau: float = 0.35,
        kappa_max: float = 8.0,
        slow_ema_mult: float = 0.35,
        # Backend & Reproducibility
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
        if gear_mode not in ("auto", "small", "medium_fp8", "huge_k_ivf", "multi_gpu", "exact"):
            raise ValueError(f"Unsupported gear_mode={gear_mode!r}")
        if precision not in ("auto", "croiss_fp8", "fp8", "bf16", "fp16", "fp32", "fp64"):
            raise ValueError(f"Unsupported precision={precision!r}")
        if init_mode not in ("apex_turkpp", "ultra_turkpp", "subsampled_d15", "random", "coreset_kpp"):
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
        if streaming_mode not in ("auto", "cyclic_block", "pre_permuted", "fused_gather"):
            raise ValueError(f"Unsupported streaming_mode={streaming_mode!r}")
        if backend not in ("auto", "triton", "torch", "native"):
            raise ValueError(f"Unsupported backend={backend!r}")
        if wide_step_mult < 1:
            raise ValueError("wide_step_mult must be >= 1")
        if graph_super_steps < 1:
            raise ValueError("graph_super_steps must be >= 1")
        if prior_mass_scale <= 0.0 or not math.isfinite(prior_mass_scale):
            raise ValueError("prior_mass_scale must be finite and > 0")
        if gamma_protect < 0.0 or not math.isfinite(gamma_protect):
            raise ValueError("gamma_protect must be finite and >= 0")

        self.d = int(d)
        self.k = int(k)
        self.nepochs = float(nepochs)
        self.gear_mode = gear_mode
        self.precision = precision
        self.devices = [torch.device(dv) for dv in devices] if devices is not None else None

        self.batch_law = batch_law
        self.batch_size = batch_size
        self.alpha0 = alpha0
        self.alpha_min_factor = float(alpha_min_factor)
        self.use_wide_batch = use_wide_batch
        self.wide_step_mult = int(wide_step_mult)
        self.wide_from_frac = float(wide_from_frac)
        self.streaming_mode = streaming_mode
        self.block_permute_size = max(1, int(block_permute_size))
        self.preencode_dataset = preencode_dataset

        self.use_fused_pipeline = bool(use_fused_pipeline)
        self.use_sm120_kernel = bool(use_sm120_kernel)
        self.use_cuda_graph = bool(use_cuda_graph)
        self.graph_super_steps = int(graph_super_steps)

        self.ivf_explore = ivf_explore
        self.ivf_k_threshold = int(ivf_k_threshold)
        self.ivf_nlist = ivf_nlist
        self.ivf_nprobe = int(ivf_nprobe)
        self.ivf_rebuild_every = max(1, int(ivf_rebuild_every))

        self.init_mode = init_mode
        self.coreset_mult = coreset_mult if coreset_mult == "auto" else int(coreset_mult)
        self.min_coreset_size = int(min_coreset_size)
        self.init_rounds = int(init_rounds)
        self.init_power = float(init_power)
        self.use_voronoi_polish = bool(use_voronoi_polish)
        self.use_prior_mass = bool(use_prior_mass)
        self.prior_mass_scale = float(prior_mass_scale)
        self.prior_floor = float(prior_floor)
        if warm_alpha_scale is None:
            self._warm_alpha_auto = True
            self.warm_alpha_scale = 0.85 if self.use_prior_mass else 1.0
        else:
            if warm_alpha_scale <= 0.0 or not math.isfinite(warm_alpha_scale):
                raise ValueError("warm_alpha_scale must be finite and > 0")
            self._warm_alpha_auto = False
            self.warm_alpha_scale = float(warm_alpha_scale)

        self.schedule_mode = schedule_mode
        self.alpha_floor = float(alpha_floor)
        self.cosine_power = float(cosine_power)
        self.unused_threshold = unused_threshold
        self.check_unused_every = int(check_unused_every)
        self.init_usage = float(init_usage)
        self.lowd_revival = bool(lowd_revival)
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

        self.head_edit = head_edit
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

        # Fitted attributes (100% compatible with EMeans / SMeans / SuperEMeans / UltraEMeans)
        self.recipe: EMeansRecipe | None = None
        self.seeding_result: SeedingResult | None = None
        self.centroids: torch.Tensor | None = None
        self.centroids_per_gpu: list[torch.Tensor] = []
        self.slow_centroids: torch.Tensor | None = None
        self.pre_close_centroids: torch.Tensor | None = None
        self.assignments: torch.Tensor | None = None
        self.iteration_stats: list[ApexEMeansEpochStat] = []
        self.shrinkage_diagnostics: ShrinkageDiagnostics | None = None
        self.head_edit_diagnostics: HeadEditDiagnostics | None = None
        self.selected_gear: str = "unfitted"
        self.resolved_precision: str = "unfitted"
        self.n_splits: int = 0
        self.n_protected: int = 0
        self.n_transplants: int = 0
        self.dead_clusters: int = 0
        self.shard_transfer_time_sec: float = 0.0
        self.preencode_time_sec: float = 0.0
        self.init_time_sec: float = 0.0
        self.train_time_sec: float = 0.0
        self.total_time_sec: float = 0.0
        self._origin64: torch.Tensor | None = None
        self._coord_scale: float = 1.0
        self._high_cr_fp32_assign: bool = False
        self._graphs_captured: int = 0
        self._graph_replays: int = 0

    def _resolve_precision_and_dtype(
        self,
        n: int,
        d: int,
        k: int,
        device: torch.device,
        x_dtype: torch.dtype,
    ) -> tuple[str, torch.dtype]:
        """Resolve `(precision_name, torch.dtype)` honoring `compute_dtype`, `precision`, and scale crossover."""
        if self.compute_dtype is not None:
            if self.compute_dtype == torch.float64:
                return "fp64", torch.float64
            if FLOAT8_E4M3FN is not None and self.compute_dtype == FLOAT8_E4M3FN:
                if _has_fp8_tensor_cores(device):
                    return "croiss_fp8", FLOAT8_E4M3FN
                return ("bf16", torch.bfloat16) if device.type == "cuda" else ("fp32", torch.float32)
            if device.type != "cuda" and self.compute_dtype in (torch.bfloat16, torch.float16):
                return "fp32", torch.float32
            if self.compute_dtype == torch.bfloat16:
                return "bf16", torch.bfloat16
            if self.compute_dtype == torch.float16:
                return "fp16", torch.float16
            return "fp32", torch.float32

        if self.precision == "fp64":
            return "fp64", torch.float64
        if self.precision == "fp32" or device.type != "cuda" or self.backend == "torch":
            return "fp32", torch.float32
        if self.precision == "fp16":
            return "fp16", torch.float16
        if self.precision == "bf16":
            return ("bf16", torch.bfloat16) if torch.cuda.get_device_capability(device) >= (8, 0) else ("fp32", torch.float32)
        if self.precision in ("croiss_fp8", "fp8"):
            if _has_fp8_tensor_cores(device):
                return "croiss_fp8", FLOAT8_E4M3FN
            return ("bf16", torch.bfloat16) if torch.cuda.get_device_capability(device) >= (8, 0) else ("fp32", torch.float32)

        # `precision == "auto"`
        if x_dtype == torch.float16:
            return "fp16", torch.float16
        if x_dtype == torch.float64 and self.gear_mode == "exact":
            return "fp64", torch.float64
        if self.gear_mode == "small":
            return ("bf16", torch.bfloat16) if torch.cuda.get_device_capability(device) >= (8, 0) else ("fp32", torch.float32)

        if _has_fp8_tensor_cores(device) and d >= 32:
            if self.gear_mode in ("medium_fp8", "huge_k_ivf", "multi_gpu"):
                return "croiss_fp8", FLOAT8_E4M3FN
            # Empirical SM120 Scale Law (`#189` across 1,327 runs):
            #   - For D > 256 (`FULL_D=False`), tiled indirect-gather FP8 is slower than BF16 -> stay on BF16 in `auto`.
            #   - For D <= 256 (`FULL_D=True`), CroissAttn-FP8 wins >= 1.05x once `N * K * T * (D/128)^0.82 >= 1.852e8`
            #     and `(N >= 65,536 or K >= 2,048)` while avoiding small-N E4M3 quantization noise.
            if self.gear_mode == "auto" and d <= 256:
                eff_t = float(max(self.nepochs, 0.5))
                fp8_law_work = float(n) * float(k) * eff_t * ((float(d) / 128.0) ** 0.820)
                if fp8_law_work >= 1.852e8 and (n >= 65536 or k >= 2048):
                    return "croiss_fp8", FLOAT8_E4M3FN

        if torch.cuda.get_device_capability(device) >= (8, 0):
            return "bf16", torch.bfloat16
        return "fp32", torch.float32

    def _effective_pass_span(self, n: int) -> float:
        """Apply closing-pass ramp when drift dominates sampling noise (`N/K >= 150, T <= 3.5` or `N/K >= 128, T <= 1.25`)."""
        if self.use_pass_weighting and not self.adaptive_pass_weighting:
            return float(self.pass_w_span)
        if self.adaptive_pass_weighting:
            nk_ratio = float(n) / float(max(1, self.k))
            if self.nepochs <= 1.25 and nk_ratio >= 128.0:
                return float(self.pass_w_span)
            if self.nepochs <= 3.5 and nk_ratio >= 150.0:
                return 0.32
        return 0.0

    def _resolve_wide_mult(self, base_recipe: EMeansRecipe) -> int:
        if isinstance(self.use_wide_batch, bool):
            return max(1, self.wide_step_mult) if self.use_wide_batch else 1
        # `use_wide_batch == "auto"` (`#189` Law 3A): widen once total steps >= 24 and batch_size < 32,768
        if base_recipe.total_steps >= 24 and base_recipe.batch_size < 32768 and self.wide_step_mult > 1:
            return self.wide_step_mult
        return 1

    def _resolve_head_edit(self, n: int) -> bool:
        if isinstance(self.head_edit, bool):
            return self.head_edit
        # `head_edit == "auto"` (`#189` Law 3C): transplant pays off when N/K >= 90 and K <= 2,048 and T >= 1.5
        nk_ratio = float(n) / float(max(1, self.k))
        return bool(nk_ratio >= 90.0 and self.k <= 2048 and self.nepochs >= 1.5)

    @torch.no_grad()
    def prepopulate_shards(
        self,
        x: torch.Tensor | np.ndarray,
    ) -> list[torch.Tensor]:
        """Distribute dataset evenly across `self.devices` outside `init_time_sec + train_time_sec`."""
        if not self.devices:
            raise ValueError("prepopulate_shards requires `devices=[...]` on ApexEMeans")
        x_t = (
            torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
            if isinstance(x, np.ndarray)
            else x.detach()
        )
        W = len(self.devices)
        for dv in self.devices:
            torch.cuda.synchronize(dv)
        t0 = time.perf_counter()
        shards = [x_t[w::W].to(device=dv, dtype=torch.float32).contiguous() for w, dv in enumerate(self.devices)]
        for dv in self.devices:
            torch.cuda.synchronize(dv)
        self.shard_transfer_time_sec = time.perf_counter() - t0
        return shards

    @torch.no_grad()
    def fit(
        self,
        x: torch.Tensor | np.ndarray | list[torch.Tensor],
        init_centroids: torch.Tensor | np.ndarray | None = None,
        seeding_result: SeedingResult | None = None,
        capture_assignments: bool = True,
    ) -> tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray | None]:
        """Fit `ApexEMeans` on `(N, D)` tensor/array or pre-resident multi-GPU `list[torch.Tensor]` shards."""
        self.n_splits = 0
        self.n_protected = 0
        self.n_transplants = 0
        self.dead_clusters = 0
        self.preencode_time_sec = 0.0
        self._graphs_captured = 0
        self._graph_replays = 0
        self.iteration_stats = []

        # Gear 4: Multi-GPU Data-Sharded NCCL Engine
        if isinstance(x, list) or (self.devices is not None and len(self.devices) >= 2) or self.gear_mode == "multi_gpu":
            return self._fit_multi_gpu_entry(
                x=x,
                init_centroids=init_centroids,
                seeding_result=seeding_result,
                capture_assignments=capture_assignments,
            )

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

        if self.devices is not None and len(self.devices) == 1 and x_t.device != self.devices[0]:
            x_t = x_t.to(self.devices[0])

        n, d = int(x_t.shape[0]), int(x_t.shape[1])
        if d != self.d:
            raise ValueError(f"Expected dimension d={self.d}, got {d}")
        if n < self.k:
            raise ValueError(f"Number of points n={n} must be >= number of clusters k={self.k}")
        if not bool(torch.isfinite(x_t).all()):
            raise ValueError("Input dataset x must contain only finite values (no NaN or Inf)")

        device = x_t.device
        prec_name, comp_dtype = self._resolve_precision_and_dtype(n, d, self.k, device, x_t.dtype)
        self.resolved_precision = prec_name

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
        eff_warm_scale = (
            0.95
            if (self._warm_alpha_auto and self.use_prior_mass and (float(n) / float(max(1, self.k))) >= 180.0)
            else self.warm_alpha_scale
        )
        eff_alpha0 = (
            float(min(max(base_recipe.alpha0 * eff_warm_scale, 1e-4), 1.0))
            if (self.alpha0 is None and eff_warm_scale != 1.0)
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

        dev_ctx = torch.cuda.device(device) if device.type == "cuda" else contextlib.nullcontext()
        with dev_ctx:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t_init_start = time.perf_counter()

            seed_comp_dtype = torch.bfloat16 if prec_name == "croiss_fp8" else comp_dtype
            origin64, coord_scale = _compute_origin_and_scale(x_t, seed_comp_dtype)
            self._origin64 = origin64
            self._coord_scale = coord_scale

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
                    eff_cm = 8 if self.coreset_mult == "auto" else int(self.coreset_mult)
                    m_sub = min(n, max(self.min_coreset_size, eff_cm * self.k, self.k * 2))
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
                        compute_dtype=torch.float32 if seed_comp_dtype == torch.float64 else seed_comp_dtype,
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
                eff_mult: int | str = 16 if self.init_mode == "coreset_kpp" else self.coreset_mult
                eff_min_m = 8192 if self.init_mode == "coreset_kpp" else self.min_coreset_size
                seed_res = apex_turkpp_init_centered(
                    x_t=x_t,
                    origin64=origin64,
                    coord_scale=coord_scale,
                    k=self.k,
                    comp_dtype=seed_comp_dtype,
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
                and prec_name != "fp64"
                and not torch.are_deterministic_algorithms_enabled()
                and (
                    self.backend == "triton"
                    or (self.backend == "auto" and torch.cuda.get_device_capability(device) >= (8, 0))
                )
            )

            if use_triton_fast and self.use_fused_pipeline:
                centroids_c, assignments = self._fit_cuda_fast(
                    x_t=x_t,
                    seed_res=seed_res,
                    has_priors=need_priors,
                    prec_name=prec_name,
                    comp_dtype=comp_dtype,
                    recipe=recipe,
                    origin64=origin64,
                    coord_scale=coord_scale,
                    capture_assignments=capture_assignments,
                )
            elif use_triton_fast and not self.use_fused_pipeline:
                self.selected_gear = "ladder_unfused_gpu"
                centroids_c, assignments = self._fit_torch_reference(
                    x_t=x_t,
                    seed_res=seed_res,
                    has_priors=need_priors,
                    comp_dtype=torch.bfloat16 if prec_name == "croiss_fp8" else comp_dtype,
                    recipe=recipe,
                    origin64=origin64,
                    coord_scale=coord_scale,
                    capture_assignments=capture_assignments,
                    use_triton_assign_only=True,
                )
            else:
                self.selected_gear = "torch_reference"
                centroids_c, assignments = self._fit_torch_reference(
                    x_t=x_t,
                    seed_res=seed_res,
                    has_priors=need_priors,
                    comp_dtype=torch.float32 if prec_name == "croiss_fp8" else comp_dtype,
                    recipe=recipe,
                    origin64=origin64,
                    coord_scale=coord_scale,
                    capture_assignments=capture_assignments,
                    use_triton_assign_only=False,
                )

            inv_scale = 1.0 / coord_scale
            self.seeding_result = SeedingResult(
                centroids=(seed_res.centroids.double() * inv_scale + origin64[None, :]).float().contiguous(),
                prior_probs=seed_res.prior_probs,
                prior_distortion=(seed_res.prior_distortion.double() * (inv_scale * inv_scale)).float().contiguous(),
            )
            self.centroids = (centroids_c.double() * inv_scale + origin64[None, :]).float().contiguous()
            self.centroids_per_gpu = [self.centroids]
            if self.slow_centroids is not None:
                self.slow_centroids = (self.slow_centroids.double() * inv_scale + origin64[None, :]).float().contiguous()
            if self.pre_close_centroids is not None:
                self.pre_close_centroids = (
                    self.pre_close_centroids.double() * inv_scale + origin64[None, :]
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
        prec_name: str,
        comp_dtype: torch.dtype,
        recipe: EMeansRecipe,
        origin64: torch.Tensor,
        coord_scale: float,
        capture_assignments: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Fused Blackwell Triton + CroissAttn-FP8/BF16 + End-to-End Multi-Step CUDAGraph Engine."""
        device = x_t.device
        n, d = int(x_t.shape[0]), int(x_t.shape[1])
        k = self.k

        t_train_start = time.perf_counter()

        rep_mode = _rep_mode_from_precision(prec_name, comp_dtype)
        eff_wide_mult = self._resolve_wide_mult(recipe)
        eff_head_edit = self._resolve_head_edit(n)
        eff_pass_span = self._effective_pass_span(n)
        num_epochs = len(recipe.epoch_rows)

        use_ivf_gear = (
            (self.ivf_explore is True or self.gear_mode == "huge_k_ivf")
            or (
                self.ivf_explore == "auto"
                and self.gear_mode == "auto"
                and k >= self.ivf_k_threshold
                and d >= 64
                and n >= 30 * k
                and self.nepochs >= 2.0
                and num_epochs >= 2
            )
        )
        if use_ivf_gear:
            self.selected_gear = f"huge_k_ivf_{prec_name}_graph"
        elif prec_name == "croiss_fp8":
            self.selected_gear = "medium_croiss_fp8_graph" if self.use_cuda_graph else "medium_croiss_fp8_eager"
        else:
            self.selected_gear = f"small_{prec_name}_graph" if self.use_cuda_graph else f"small_{prec_name}_eager"

        origin32 = origin64.float().contiguous()
        origin_assign = (
            origin64
            if (x_t.dtype == torch.float64 or coord_scale != 1.0 or float(origin64.abs().max().item()) > 1e3)
            else origin32
        )

        # --- 1. One-Pass Coalesced Pre-Permute & CroissAttn-FP8 / BF16 Pre-Encode ---
        # Pre-permuting up to 5.6GB (`<= 28%` free VRAM, e.g. `10M x 128 = 5.12GB` on 96GB GPUs) makes
        # EVERY epoch's assign AND scatter 100% coalesced!
        free_vram = int(torch.cuda.mem_get_info(device)[0]) if device.type == "cuda" else 0
        max_preperm_bytes = min(5_600_000_000, max(1_100_000_000, int(0.28 * free_vram)))
        if isinstance(self.preencode_dataset, bool):
            do_preperm = self.preencode_dataset and (self.streaming_mode != "fused_gather")
        else:
            do_preperm = (self.streaming_mode != "fused_gather") and (
                (n * d * 4 <= max_preperm_bytes) and (prec_name == "croiss_fp8" or self.nepochs >= 1.0)
            )

        t_pre0 = time.perf_counter()
        perm0: torch.Tensor | None = None
        inv_perm0: torch.Tensor | None = None
        x_scatter_src = x_t
        scatter_has_origin = True
        scatter_has_scale = coord_scale != 1.0
        x_assign_src = x_t
        assign_has_origin = True
        assign_has_scale = coord_scale != 1.0
        sx_buf: torch.Tensor | None = None
        x_norm_buf: torch.Tensor | None = None
        croiss_ds: Any = None
        croiss_ws: Any = None

        cmod_fp8 = _get_croiss_fp8_module() if rep_mode == 3 else None
        fp8_pad_d = (
            int(cmod_fp8.compute_croiss_padded_d(d, "decoupled"))
            if (cmod_fp8 is not None and hasattr(cmod_fp8, "compute_croiss_padded_d"))
            else (((max(1, d) + 63) // 64) * 64)
        )
        bsz_max_ws = min(n, recipe.batch_size * max(1, eff_wide_mult))

        if do_preperm:
            gen_p0 = torch.Generator(device=device)
            gen_p0.manual_seed(epoch_seed(self.seed, 0))
            perm0 = torch.randperm(n, generator=gen_p0, device=device)

            write_f32_stream = n * d * 4 <= max_preperm_bytes
            x_perm_f32 = (
                torch.empty((n, d), dtype=torch.float32, device=device)
                if write_f32_stream
                else x_t
            )
            if rep_mode == 3:
                qx_wrap = torch.empty((n + 256, fp8_pad_d), dtype=FLOAT8_E4M3FN, device=device)
                sx_wrap = torch.empty((n + 256,), dtype=torch.float32, device=device)
                xnorm_wrap = torch.empty((n + 256,), dtype=torch.float32, device=device)
                x_perm_rep = qx_wrap[:n]
                sx_buf = sx_wrap[:n]
                x_norm_buf = xnorm_wrap[:n]
            else:
                qx_wrap = None
                sx_wrap = None
                xnorm_wrap = None
                x_perm_rep = _alloc_padded_2d(n, d, comp_dtype, device, align=16)
                sx_buf = None
                x_norm_buf = torch.empty((n,), dtype=torch.float32, device=device)

            blk_n_enc = 64
            blk_d_enc = min(128, max(32 if rep_mode == 3 else 16, triton.next_power_of_2(max(1, d))))
            with torch.cuda.device(device):
                _apex_prepermute_encode_kernel[(triton.cdiv(n, blk_n_enc),)](
                    x_t,
                    perm0,
                    origin_assign,
                    x_perm_f32,
                    x_perm_rep,
                    sx_buf if sx_buf is not None else x_norm_buf,
                    x_norm_buf,
                    n,
                    d,
                    x_t.stride(0),
                    x_t.stride(1),
                    x_perm_f32.stride(0),
                    x_perm_rep.stride(0),
                    float(coord_scale),
                    HAS_PERM=True,
                    ORIGIN_FP64=(origin_assign.dtype == torch.float64),
                    HAS_SCALE=(coord_scale != 1.0),
                    WRITE_F32=write_f32_stream,
                    REP_MODE=rep_mode,
                    BLOCK_N=blk_n_enc,
                    BLOCK_D=blk_d_enc,
                    num_warps=4,
                )
            x_assign_src = x_perm_rep
            assign_has_origin = False
            assign_has_scale = False
            if write_f32_stream:
                x_scatter_src = x_perm_f32
                scatter_has_origin = False
                scatter_has_scale = False
            else:
                inv_perm0 = perm0  # scatter falls back to indexing x_t via perm0[pos]

            if rep_mode == 3 and qx_wrap is not None and sx_wrap is not None and xnorm_wrap is not None:
                for off_w in range(0, 256, max(1, n)):
                    ln_w = min(n, 256 - off_w)
                    qx_wrap[n + off_w : n + off_w + ln_w].copy_(x_perm_rep[:ln_w])
                    sx_wrap[n + off_w : n + off_w + ln_w].copy_(sx_buf[:ln_w])  # type: ignore[index]
                    xnorm_wrap[n + off_w : n + off_w + ln_w].copy_(x_norm_buf[:ln_w])
                if cmod_fp8 is not None and hasattr(cmod_fp8, "CroissFP8Dataset") and hasattr(cmod_fp8, "CroissFP8Workspace"):
                    try:
                        gs_one = torch.ones((1,), dtype=torch.float64, device=device)
                        inv_gs2_one = torch.ones((1,), dtype=torch.float64, device=device)
                        croiss_ds = cmod_fp8.CroissFP8Dataset(
                            qx=x_perm_rep,
                            sx=sx_buf,
                            x_norm=x_norm_buf,
                            mu=torch.zeros((d,), dtype=torch.float32, device=device),
                            mu_fp64=torch.zeros((d,), dtype=torch.float64, device=device),
                            n=n,
                            d=d,
                            padded_d=fp8_pad_d,
                            scale_mode="decoupled",
                            center=False,
                            coord_scale=1.0,
                            gscale_ptr=gs_one,
                            inv_gscale_sq_ptr=inv_gs2_one,
                            _qx_wrap=qx_wrap,
                            _sx_wrap=sx_wrap,
                            _x_norm_wrap=xnorm_wrap,
                        )
                        croiss_ws = cmod_fp8.CroissFP8Workspace.allocate(
                            max_batch_size=bsz_max_ws,
                            k=k,
                            d=d,
                            padded_d=fp8_pad_d,
                            device=device,
                            scale_mode="decoupled",
                        )
                    except Exception:
                        croiss_ds = None
                        croiss_ws = None
        elif rep_mode == 3:
            # Encode in original order when `streaming_mode == "fused_gather"` or ultra-large N
            qx_wrap = torch.empty((n + 256, fp8_pad_d), dtype=FLOAT8_E4M3FN, device=device)
            sx_wrap = torch.empty((n + 256,), dtype=torch.float32, device=device)
            xnorm_wrap = torch.empty((n + 256,), dtype=torch.float32, device=device)
            x_perm_rep = qx_wrap[:n]
            sx_buf = sx_wrap[:n]
            x_norm_buf = xnorm_wrap[:n]
            blk_n_enc = 64
            blk_d_enc = min(128, max(32, triton.next_power_of_2(max(1, d))))
            with torch.cuda.device(device):
                _apex_prepermute_encode_kernel[(triton.cdiv(n, blk_n_enc),)](
                    x_t,
                    x_norm_buf,
                    origin_assign,
                    x_t,
                    x_perm_rep,
                    sx_buf,
                    x_norm_buf,
                    n,
                    d,
                    x_t.stride(0),
                    x_t.stride(1),
                    x_t.stride(0),
                    x_perm_rep.stride(0),
                    float(coord_scale),
                    HAS_PERM=False,
                    ORIGIN_FP64=(origin_assign.dtype == torch.float64),
                    HAS_SCALE=(coord_scale != 1.0),
                    WRITE_F32=False,
                    REP_MODE=3,
                    BLOCK_N=blk_n_enc,
                    BLOCK_D=blk_d_enc,
                    num_warps=4,
                )
            for off_w in range(0, 256, max(1, n)):
                ln_w = min(n, 256 - off_w)
                qx_wrap[n + off_w : n + off_w + ln_w].copy_(x_perm_rep[:ln_w])
                sx_wrap[n + off_w : n + off_w + ln_w].copy_(sx_buf[:ln_w])
                xnorm_wrap[n + off_w : n + off_w + ln_w].copy_(x_norm_buf[:ln_w])
            x_assign_src = x_perm_rep
            assign_has_origin = False
            assign_has_scale = False
            if cmod_fp8 is not None and hasattr(cmod_fp8, "CroissFP8Dataset") and hasattr(cmod_fp8, "CroissFP8Workspace"):
                try:
                    gs_one = torch.ones((1,), dtype=torch.float64, device=device)
                    inv_gs2_one = torch.ones((1,), dtype=torch.float64, device=device)
                    croiss_ds = cmod_fp8.CroissFP8Dataset(
                        qx=x_perm_rep,
                        sx=sx_buf,
                        x_norm=x_norm_buf,
                        mu=torch.zeros((d,), dtype=torch.float32, device=device),
                        mu_fp64=torch.zeros((d,), dtype=torch.float64, device=device),
                        n=n,
                        d=d,
                        padded_d=fp8_pad_d,
                        scale_mode="decoupled",
                        center=False,
                        coord_scale=1.0,
                        gscale_ptr=gs_one,
                        inv_gscale_sq_ptr=inv_gs2_one,
                        _qx_wrap=qx_wrap,
                        _sx_wrap=sx_wrap,
                        _x_norm_wrap=xnorm_wrap,
                    )
                    croiss_ws = cmod_fp8.CroissFP8Workspace.allocate(
                        max_batch_size=bsz_max_ws,
                        k=k,
                        d=d,
                        padded_d=fp8_pad_d,
                        device=device,
                        scale_mode="decoupled",
                    )
                except Exception:
                    croiss_ds = None
                    croiss_ws = None

        self.preencode_time_sec = time.perf_counter() - t_pre0

        # --- 2. Initialize Master FP32 Cluster State & Ping-Pong GPU State Ring ---
        centroids_c = seed_res.centroids.to(device=device, dtype=torch.float32).clone().contiguous()
        p_hat = seed_res.prior_probs.to(device=device, dtype=torch.float32)
        e_hat = seed_res.prior_distortion.to(device=device, dtype=torch.float32)

        bsz_float = float(recipe.batch_size)
        min_prob = max(0.0, self.prior_floor) / float(k)
        mean_ehat = e_hat.mean().clamp_min(1e-12)

        if self.use_prior_mass and has_priors:
            usage = (self.prior_mass_scale * bsz_float * p_hat.clamp_min(min_prob)).float().contiguous()
            sse_ema = (self.prior_mass_scale * bsz_float * e_hat.clamp_min(1e-6 * mean_ehat)).float().contiguous()
            init_u_bias = float(self.prior_mass_scale)
            init_s_bias = float(self.prior_mass_scale)
        else:
            usage = torch.full((k,), recipe.init_usage, dtype=torch.float32, device=device)
            sse_ema = torch.zeros((k,), dtype=torch.float32, device=device)
            init_u_bias = float(recipe.init_usage * k) / max(1.0, bsz_float)
            init_s_bias = 0.0

        track_surv = bool(self.use_huygens_protection and self.gamma_protect > 0.0)
        survival_mass = (
            ((bsz_float * p_hat).float().contiguous() if has_priors else torch.zeros((k,), dtype=torch.float32, device=device))
            if track_surv
            else centroids_c
        )

        track_slow = self.finalization in ("simplex_eb", "eb_slow_only")
        if track_slow:
            slow_centroids = centroids_c.clone().contiguous()
            slow_usage = usage.clone().contiguous()
            alpha_slow_base = min(1.0, max(1e-6, self.slow_ema_mult * recipe.alpha_min_factor * recipe.alpha0))
        else:
            slow_centroids = centroids_c
            slow_usage = usage
            alpha_slow_base = 0.0

        has_pre_close = self.finalization == "simplex_eb"
        pre_close_centroids = centroids_c.clone().contiguous() if has_pre_close else centroids_c
        pre_close_captured = False

        want_oja = self.split_mode in ("half_gaussian_oja", "oja_only") or self.donor_mode == "oja_sse_usage"
        if want_oja:
            pca_gen = torch.Generator(device=device)
            pca_gen.manual_seed((self.seed + 31337) % (2**32))
            init_dirs = torch.randn((k, d), generator=pca_gen, device=device, dtype=torch.float32)
            init_dirs = init_dirs / init_dirs.norm(dim=1, keepdim=True).clamp_min(1e-12)
            if has_priors:
                init_rms = (e_hat / (p_hat.clamp_min(1e-6) * float(max(1, d)))).clamp_min(1e-8).sqrt()[:, None]
                prin_vec = (init_rms * init_dirs).contiguous()
            else:
                prin_vec = (1e-3 * init_dirs).contiguous()
            g_sums = torch.zeros((k, d), dtype=torch.float32, device=device)
        else:
            prin_vec = centroids_c
            g_sums = centroids_c

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

        if croiss_ws is not None and hasattr(croiss_ws, "qc") and hasattr(croiss_ws, "sc") and hasattr(croiss_ws, "c_norm"):
            c_rep = croiss_ws.qc
            sc_buf = croiss_ws.sc
            c_norm = croiss_ws.c_norm
        else:
            c_pad_align = 64 if rep_mode == 3 else 16
            c_rep_dt = FLOAT8_E4M3FN if rep_mode == 3 else comp_dtype
            c_rep = _alloc_padded_2d(k, d, c_rep_dt, device, align=c_pad_align)
            sc_buf = torch.ones((k,), dtype=torch.float32, device=device)
            c_norm = torch.empty((k,), dtype=torch.float32, device=device)

        _launch_apex_prep_centroids(centroids_c, c_rep, sc_buf, c_norm, rep_mode, croiss_ws=croiss_ws, croiss_ds=croiss_ds)

        # Dimensionless SQNR / Mantissa Resolution Guard (`#169`, `#171`, `#176`):
        # Note `sum_j e_hat_j = MSE`, so `k * mean_ehat = MSE`!
        # When `CR = max_i ||x_i - mu||^2 / MSE > 120.0` (e.g. `covtype_multiscale_54` with 110,761x feature span,
        # `CR ~ 913`, whereas SIFT/GloVe/CIFAR/AG-News all have `CR <= 22`):
        #   - On FP8 (`rep_mode == 3`), upgrade `CroissFP8` from `"decoupled"` to `"ozaki"` (3-lane Bilinear Double-FP8
        #     Ozaki Residual Expansion on top-6 dominant coordinates -> `20.6 ppm = +0.002%` accuracy vs FP64 while
        #     running 100% in SM120 FP8 Tensor Cores + `CUDAGraph`!).
        #   - On BF16/FP16 (`rep_mode in (1, 2)`), run the closing pass (`is_last_epoch`) in exact IEEE-FP32.
        high_cr_closing_fp32 = False
        croiss_use_ozaki = False
        c_rep_f32: torch.Tensor = c_rep
        x_norm_f32: torch.Tensor | None = None
        if rep_mode in (1, 2, 3):
            if x_norm_buf is not None:
                max_x_sq = float(x_norm_buf.max().item())
            else:
                sub_s = x_t[:: max(1, n // 4096)][:4096]
                max_x_sq = float(((sub_s.double() - origin64[None, :]) * coord_scale).float().square().sum(dim=1).max().item())
            if has_priors and float(mean_ehat.item()) > 0.0:
                est_eh = float(mean_ehat.item()) * float(k)
            else:
                sub_e = ((x_t[:: max(1, n // 2048)][:2048].double() - origin64[None, :]) * coord_scale).float().contiguous()
                est_eh = float(assign_points(sub_e, centroids_c, compute_dtype=torch.float32)[0].mean().item())
            if (max_x_sq / max(est_eh, 1e-30)) > 120.0:
                if rep_mode == 3 and cmod_fp8 is not None and hasattr(cmod_fp8, "croiss_encode_dataset"):
                    try:
                        if not scatter_has_origin:
                            croiss_ds = cmod_fp8.croiss_encode_dataset(
                                x_scatter_src, center=False, coord_scale=1.0, scale_mode="ozaki"
                            )
                        else:
                            croiss_ds = cmod_fp8.croiss_encode_dataset(
                                x_t,
                                mu=origin64,
                                row_indices=perm0,
                                center=True,
                                coord_scale=coord_scale,
                                scale_mode="ozaki",
                            )
                        croiss_ws = cmod_fp8.CroissFP8Workspace.allocate(
                            max_batch_size=bsz_max_ws,
                            k=k,
                            d=d,
                            padded_d=croiss_ds.padded_d,
                            device=device,
                            scale_mode="ozaki",
                        )
                        c_rep = croiss_ws.qc
                        sc_buf = croiss_ws.sc
                        c_norm = croiss_ws.c_norm
                        x_assign_src = croiss_ds.qx
                        sx_buf = croiss_ds.sx
                        x_norm_buf = croiss_ds.x_norm
                        croiss_use_ozaki = True
                        _launch_apex_prep_centroids(
                            centroids_c, c_rep, sc_buf, c_norm, 3, croiss_ws=croiss_ws, croiss_ds=croiss_ds
                        )
                    except Exception:
                        croiss_use_ozaki = False
                if not croiss_use_ozaki:
                    high_cr_closing_fp32 = True
                    c_rep_f32 = _alloc_padded_2d(k, d, torch.float32, device, align=16)
                    if not scatter_has_origin:
                        x_norm_f32 = x_scatter_src.square().sum(dim=1).contiguous()
        self._high_cr_fp32_assign = high_cr_closing_fp32 or croiss_use_ozaki

        ivf_explorer: _ApexSyncFreeIVFExplorer | None = None
        eff_ivf_nprobe = (16 if n >= 4_000_000 else 8) if self.ivf_nprobe == 8 else int(self.ivf_nprobe)
        if use_ivf_gear:
            # `#189` Law 2: `1 * sqrt(K)` (`nlist in [128, 256]` for Huge-K) beats `2 * sqrt(K)` on both speed & recall!
            eff_nlist = (
                int(self.ivf_nlist)
                if self.ivf_nlist is not None
                else max(32, min(256, round_power_of_two(math.sqrt(float(max(16, k))))))
            )
            ivf_explorer = _ApexSyncFreeIVFExplorer(centroids_c, nlist=eff_nlist, seed=self.seed)

        bsz_max = min(n, recipe.batch_size * max(1, eff_wide_mult))
        dist_buf = torch.empty((bsz_max,), dtype=torch.float32, device=device)
        lbl_buf = torch.empty((bsz_max,), dtype=torch.int64, device=device)

        num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        splitk_buffers: tuple[torch.Tensor, torch.Tensor, torch.Tensor, int] | None = None
        if self.use_sm120_kernel and k >= 2048 and triton.cdiv(recipe.batch_size, 64) < num_sms:
            parts = _choose_split_k_partitions(recipe.batch_size, k, num_sms=num_sms, block_b=64, block_k=128)
            if parts > 1:
                splitk_buffers = (
                    torch.empty((bsz_max, parts), dtype=torch.float32, device=device),
                    torch.empty((bsz_max, parts), dtype=torch.int32, device=device),
                    torch.empty((bsz_max,), dtype=torch.float32, device=device),
                    parts,
                )

        revival_ws = (
            _ApexRevivalWorkspace.allocate(k, d, device)
            if (recipe.unused_threshold > 0.0 and self.revival_mode == "split" and k >= 2)
            else None
        )

        if eff_head_edit:
            last_edit_step = torch.full((k,), -(10**9), dtype=torch.int64, device=device)
            edit_gen = torch.Generator(device=device)
            edit_gen.manual_seed((self.seed + 918273) % (2**32))
            edit_diag: HeadEditDiagnostics | None = HeadEditDiagnostics()
        else:
            last_edit_step = None
            edit_gen = None
            edit_diag = None
        self.head_edit_diagnostics = edit_diag

        ep_sq_dists_gpu = torch.zeros((num_epochs,), dtype=torch.float64, device=device)
        cur_ep_sq_gpu = torch.zeros((1,), dtype=torch.float64, device=device)
        ep_dot_counts_gpu = torch.zeros((num_epochs,), dtype=torch.float64, device=device)
        ep_cum_splits_gpu = torch.zeros((num_epochs,), dtype=torch.int64, device=device)
        ep_cum_prot_gpu = torch.zeros((num_epochs,), dtype=torch.int64, device=device)
        total_splits_gpu = torch.zeros((), dtype=torch.int64, device=device)
        total_prot_gpu = torch.zeros((), dtype=torch.int64, device=device)

        # 3-slot race-free GPU state ring: `[step_idx, rows_processed, ep_b_start, usage_bias, sse_bias, alpha_last]`
        gpu_state = torch.zeros((3, 8), dtype=torch.float64, device=device)
        gpu_state[0, 3] = init_u_bias
        gpu_state[0, 4] = init_s_bias
        gpu_state[0, 5] = recipe.alpha0 * recipe.alpha_min_factor
        batch_off_buf = torch.zeros((3,), dtype=torch.int64, device=device)
        cur_slot = 0

        ev_start = torch.cuda.Event(enable_timing=True)
        ev_epochs = [torch.cuda.Event(enable_timing=True) for _ in range(num_epochs)]
        ev_start.record()

        blk_d_upd = min(256, max(16, triton.next_power_of_2(max(1, d))))
        blk_d_fs = min(128, max(16, triton.next_power_of_2(max(1, d))))
        blk_k_fs = min(64, max(16, triton.next_power_of_2(max(1, k))))

        sched_code = {"emeans_cosine": 0, "power_cosine": 1, "constant": 2}[self.schedule_mode]
        donor_code = {"oja_sse_usage": 0, "sse_usage": 1, "d2": 2, "usage_only": 3}[self.donor_mode]
        split_code = {
            "half_gaussian_oja": 0,
            "rms_aniso": 1,
            "rms_iso": 1,
            "oja_only": 2,
            "checkerboard": 3,
        }[self.split_mode]

        eff_grace = (
            min(self.grace_steps, max(0, recipe.total_steps // 3))
            if (self.use_prior_mass and has_priors and track_surv)
            else 0
        )
        total_planned_rows = max(1, sum(recipe.epoch_rows))
        eff_rev_frac = (
            max(self.revive_stop_frac, 1.0 - 0.30 / float(max(1, num_epochs)))
            if use_ivf_gear
            else self.revive_stop_frac
        )
        revive_stop_step = int(eff_rev_frac * recipe.total_steps)
        revive_stop_rows = int(eff_rev_frac * total_planned_rows)

        wide_alpha_factor = float(eff_wide_mult ** ALPHA_LAW[2]) if eff_wide_mult > 1 else 1.0
        in_wide_phase = False
        rows_processed_host = 0
        step_host = 0
        next_cp_idx = 0
        sorted_cps = sorted(cp for cp in self.transplant_checkpoints if 0.05 <= cp <= 0.82)
        ep_transplants: list[int] = []
        cap_stream = torch.cuda.Stream(device=device) if self.use_cuda_graph else None
        graph_pool = torch.cuda.graph_pool_handle() if self.use_cuda_graph else None
        graph_cache: dict[tuple[Any, ...], torch.cuda.CUDAGraph] = {}
        order_assign_buf = (
            torch.empty((n,), dtype=torch.int64, device=device)
            if (num_epochs >= 2 or not do_preperm)
            else None
        )
        order_scatter_buf = (
            torch.empty((n,), dtype=torch.int64, device=device)
            if (do_preperm and num_epochs >= 2 and inv_perm0 is not None)
            else None
        )

        def _sync_slot0() -> None:
            nonlocal cur_slot
            if cur_slot != 0:
                gpu_state[0].copy_(gpu_state[cur_slot])
                batch_off_buf[0].copy_(batch_off_buf[cur_slot])
                cur_slot = 0

        for ep_idx, rows_in_ep in enumerate(recipe.epoch_rows):
            is_last_epoch = ep_idx == (num_epochs - 1)
            if is_last_epoch and num_epochs >= 2 and has_pre_close and not pre_close_captured:
                pre_close_centroids.copy_(centroids_c)
                pre_close_captured = True

            # Build per-epoch index stream into fixed-address buffers so CUDAGraphs replay across epochs (`0` re-capture!)
            order_assign: torch.Tensor | None
            order_scatter: torch.Tensor | None
            if do_preperm:
                if ep_idx == 0:
                    order_assign = None
                    order_scatter = inv_perm0
                else:
                    gen_ep = torch.Generator(device=device)
                    gen_ep.manual_seed(epoch_seed(self.seed, ep_idx))
                    if self.streaming_mode == "pre_permuted":
                        order_ep = torch.randperm(n, generator=gen_ep, device=device)[:rows_in_ep]
                    else:
                        order_ep = block_permute_indices(n, rows_in_ep, self.block_permute_size, gen_ep, device)
                    assert order_assign_buf is not None
                    order_assign_buf[:rows_in_ep].copy_(order_ep)
                    order_assign = order_assign_buf[:rows_in_ep]
                    if inv_perm0 is not None and order_scatter_buf is not None:
                        torch.index_select(inv_perm0, 0, order_ep, out=order_scatter_buf[:rows_in_ep])
                        order_scatter = order_scatter_buf[:rows_in_ep]
                    else:
                        order_scatter = order_assign
            else:
                gen_ep = torch.Generator(device=device)
                gen_ep.manual_seed(epoch_seed(self.seed, ep_idx))
                order_ep = (
                    block_permute_indices(n, rows_in_ep, self.block_permute_size, gen_ep, device)
                    if self.streaming_mode == "cyclic_block"
                    else torch.randperm(n, generator=gen_ep, device=device)[:rows_in_ep]
                )
                assert order_assign_buf is not None
                order_assign_buf[:rows_in_ep].copy_(order_ep)
                order_assign = order_assign_buf[:rows_in_ep]
                order_scatter = order_assign

            _sync_slot0()
            gpu_state[0, 2] = 0.0  # reset `ep_b_start`
            batch_off_buf[0] = 0
            cur_ep_sq_gpu.zero_()

            use_fp32_ep = bool(high_cr_closing_fp32 and is_last_epoch)
            ep_rep_mode = 0 if use_fp32_ep else rep_mode
            ep_c_rep = c_rep_f32 if use_fp32_ep else c_rep
            ep_x_src = x_scatter_src if use_fp32_ep else x_assign_src
            ep_order_assign = order_scatter if use_fp32_ep else order_assign
            ep_has_orig = scatter_has_origin if use_fp32_ep else assign_has_origin
            ep_has_scale = scatter_has_scale if use_fp32_ep else assign_has_scale
            ep_x_norm = x_norm_f32 if use_fp32_ep else x_norm_buf
            ep_croiss_ds = None if use_fp32_ep else croiss_ds
            ep_croiss_ws = None if use_fp32_ep else croiss_ws
            if use_fp32_ep:
                _launch_apex_prep_centroids(centroids_c, ep_c_rep, sc_buf, c_norm, 0)

            raw_pass_counts.zero_()
            trans_before_ep = self.n_transplants
            b_start = 0

            def _exec_one_step(
                in_s: int,
                out_s: int,
                b_cur_step: int,
                use_wide_flag: bool,
                track_oja_now: bool,
                do_feat_scale_now: bool,
                do_revival_substep: bool,
                prep_in_ema_flag: bool,
                use_ivf_step: bool,
            ) -> None:
                assert in_s != out_s, "Ping-pong state ring requires disjoint in_s != out_s (`#170` inter-CTA safety)"
                d_view = dist_buf[:b_cur_step]
                l_view = lbl_buf[:b_cur_step]
                eff_peak_now = min(1.0, recipe.alpha0 * (wide_alpha_factor if use_wide_flag else 1.0))

                if use_ivf_step and ivf_explorer is not None:
                    # Gather batch on GPU for IVF explorer (`not CUDAGraph`, exploration passes only)
                    bo_val = int(batch_off_buf[in_s].item())
                    if order_scatter is not None:
                        b_ids = order_scatter[bo_val : bo_val + b_cur_step]
                        xb_ivf = x_scatter_src[b_ids]
                    else:
                        xb_ivf = x_scatter_src[bo_val : bo_val + b_cur_step]
                    if scatter_has_origin:
                        xb_ivf = ((xb_ivf.double() - origin64[None, :]) * coord_scale).float().contiguous()
                    ivf_explorer.assign_batch(xb_ivf, nprobe=eff_ivf_nprobe, out_dist=d_view, out_label=l_view)
                else:
                    _launch_apex_assign(
                        x_src=ep_x_src,
                        c_rep=ep_c_rep,
                        c_norm=c_norm,
                        rep_mode=ep_rep_mode,
                        d=d,
                        sx_buf=None if use_fp32_ep else sx_buf,
                        sc_buf=sc_buf,
                        x_norm_buf=ep_x_norm,
                        row_indices=ep_order_assign,
                        batch_offset_ptr=batch_off_buf[in_s],
                        batch_size=b_cur_step,
                        wrap_rows=False,
                        origin=origin_assign if ep_has_orig else None,
                        coord_scale=coord_scale if ep_has_scale else 1.0,
                        out_dist=d_view,
                        out_label=l_view,
                        splitk_buffers=splitk_buffers if not use_wide_flag else None,
                        use_sm120_kernel=self.use_sm120_kernel,
                        croiss_ds=ep_croiss_ds,
                        croiss_ws=ep_croiss_ws,
                    )

                full_d_oja = track_oja_now and (d <= 256)
                if track_oja_now:
                    blk_b_sc = 64 if d <= 128 else 32
                    blk_d_sc = max(16, triton.next_power_of_2(max(1, d))) if full_d_oja else 128
                else:
                    blk_b_sc = 128
                    blk_d_sc = min(128, max(16, triton.next_power_of_2(max(1, d))))

                sc_n_tot = int(order_scatter.shape[0]) if order_scatter is not None else int(x_scatter_src.shape[0])
                grid_d_sc = 1 if track_oja_now else triton.cdiv(d, blk_d_sc)
                with torch.cuda.device(device):
                    _apex_scatter_oja_sse_kernel[(triton.cdiv(b_cur_step, blk_b_sc), grid_d_sc)](
                        x_scatter_src,
                        order_scatter if order_scatter is not None else l_view,
                        batch_off_buf[in_s],
                        origin_assign,
                        l_view,
                        d_view,
                        centroids_c,
                        prin_vec,
                        sums,
                        counts,
                        batch_sse,
                        g_sums,
                        cur_ep_sq_gpu,
                        0,
                        b_cur_step,
                        d,
                        sc_n_tot,
                        x_scatter_src.stride(0),
                        x_scatter_src.stride(1),
                        centroids_c.stride(0),
                        float(coord_scale),
                        HAS_INDICES=(order_scatter is not None),
                        HAS_BATCH_OFFSET=True,
                        WRAP_ROWS=False,
                        HAS_ORIGIN=scatter_has_origin,
                        HAS_SCALE=scatter_has_scale,
                        ORIGIN_FP64=(origin_assign.dtype == torch.float64),
                        TRACK_OJA=track_oja_now,
                        FULL_D=full_d_oja,
                        BLOCK_B=blk_b_sc,
                        BLOCK_D=blk_d_sc,
                        num_warps=8 if (track_oja_now and d >= 64) else 4,
                    )

                if do_feat_scale_now:
                    with torch.cuda.device(device):
                        _apex_feat_scale_kernel[(triton.cdiv(d, blk_d_fs),)](
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

                ozaki_ep = bool(croiss_use_ozaki and ep_rep_mode == 3 and ep_croiss_ws is not None)
                in_ema_prep_now = int(prep_in_ema_flag and (not ozaki_ep))
                accum_pass = int(bool(is_last_epoch and need_pass_accum))
                with torch.cuda.device(device):
                    _apex_ema_oja_slow_pass_prep_kernel[(k,)](
                        gpu_state[in_s],
                        gpu_state[out_s],
                        batch_off_buf[in_s],
                        batch_off_buf[out_s],
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
                        ep_c_rep,
                        sc_buf,
                        c_norm,
                        k,
                        d,
                        centroids_c.stride(0),
                        ep_c_rep.stride(0),
                        int(b_cur_step),
                        int(n),
                        int(rows_in_ep),
                        int(recipe.total_steps),
                        int(total_planned_rows),
                        float(eff_peak_now),
                        float(recipe.alpha_min_factor),
                        float(self.alpha_floor),
                        float(self.cosine_power),
                        float(self.sse_decay),
                        float(alpha_slow_base),
                        float(eff_pass_span),
                        int(accum_pass),
                        in_ema_prep_now,
                        SCHED_MODE=sched_code,
                        USE_WIDE_SCHED=int(eff_wide_mult > 1),
                        COUPLE_SSE=self.couple_sse_to_alpha,
                        TRACK_OJA=track_oja_now,
                        TRACK_SURV=track_surv,
                        TRACK_SLOW=track_slow,
                        REP_MODE=ep_rep_mode,
                        BLOCK_D=blk_d_upd,
                        num_warps=2 if d <= 128 else 4,
                    )
                if prep_in_ema_flag and ozaki_ep:
                    _launch_apex_prep_centroids(
                        centroids_c, ep_c_rep, sc_buf, c_norm, ep_rep_mode, croiss_ws=ep_croiss_ws, croiss_ds=ep_croiss_ds
                    )

                if do_revival_substep and revival_ws is not None:
                    _launch_apex_cudagraph_revival(
                        state_slot=gpu_state[out_s],
                        centroids_c=centroids_c,
                        origin32=origin32,
                        usage=usage,
                        sse_ema=sse_ema,
                        prin_vec=prin_vec,
                        survival_mass=survival_mass,
                        slow_centroids=slow_centroids,
                        slow_usage=slow_usage,
                        pre_close_centroids=pre_close_centroids,
                        pass_sums=pass_sums,
                        pass_counts=pass_counts,
                        pass_counts_sq=pass_counts_sq,
                        pass_sse=pass_sse,
                        feat_scale=feat_scale,
                        ones_d=ones_d,
                        ws=revival_ws,
                        total_splits_gpu=total_splits_gpu,
                        total_prot_gpu=total_prot_gpu,
                        b_cur=b_cur_step,
                        unused_threshold=recipe.unused_threshold,
                        gamma_protect=self.gamma_protect,
                        eff_grace=eff_grace,
                        revive_stop_step=revive_stop_step,
                        revive_stop_rows=revive_stop_rows,
                        donor_usage_w=self.donor_usage_w,
                        split_mult=self.split_mult,
                        split_eps=recipe.split_epsilon,
                        base_seed=self.seed,
                        use_revive_stop=self.use_revive_stop,
                        use_wide_stop=(eff_wide_mult > 1),
                        track_surv=track_surv,
                        track_oja=track_oja_now,
                        track_slow=track_slow,
                        has_pre_close=has_pre_close,
                        update_pre_close=pre_close_captured,
                        use_feat_scale=self.use_feat_scale,
                        flush_closing=bool(is_last_epoch and need_pass_accum and self.flush_pass_on_split),
                        donor_mode_code=donor_code,
                        split_mode_code=split_code,
                        rep_mode=ep_rep_mode,
                    )
                    _launch_apex_prep_centroids(
                        centroids_c, ep_c_rep, sc_buf, c_norm, ep_rep_mode, croiss_ws=ep_croiss_ws, croiss_ds=ep_croiss_ds
                    )

            while b_start < rows_in_ep:
                # Single-epoch mid-pass snapshot for Pre-Closing Simplex EB when `num_epochs == 1`
                if num_epochs == 1 and has_pre_close and (not pre_close_captured) and b_start >= int(0.55 * rows_in_ep):
                    pre_close_centroids.copy_(centroids_c)
                    pre_close_captured = True

                prog = float(rows_processed_host) / float(total_planned_rows)
                use_wide_now = eff_wide_mult > 1 and prog >= self.wide_from_frac
                cur_bsz_target = min(n, recipe.batch_size * eff_wide_mult) if use_wide_now else recipe.batch_size

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

                use_ivf_now = bool(use_ivf_gear and ivf_explorer is not None and (not is_last_epoch))
                past_all_revivals_and_cps = (
                    (
                        (rows_processed_host > revive_stop_rows)
                        if (self.use_revive_stop and eff_wide_mult > 1)
                        else ((step_host > revive_stop_step) if self.use_revive_stop else False)
                    )
                    and (not eff_head_edit or next_cp_idx >= len(sorted_cps) or prog > 0.85)
                )
                track_oja_cur = want_oja and (not past_all_revivals_and_cps)
                track_fs_cur = (
                    revival_ws is not None
                    and self.use_feat_scale
                    and split_code in (0, 1)
                    and (not past_all_revivals_and_cps)
                )

                # Check if we can execute a multi-step `CUDAGraph` super-step burst right now!
                super_g = self.check_unused_every if (revival_ws is not None and not past_all_revivals_and_cps) else max(2, self.graph_super_steps)
                can_try_graph = (
                    self.use_cuda_graph
                    and cap_stream is not None
                    and (not use_ivf_now)
                    and super_g >= 2
                    and (step_host % super_g == 0 or past_all_revivals_and_cps)
                )

                if can_try_graph:
                    # Bound super-step burst up to next phase transition: wide switch, mid-epoch snapshot, transplant cp, or revive freeze!
                    limit_rows = rows_in_ep
                    if eff_wide_mult > 1 and not use_wide_now:
                        wide_row_ep = b_start + max(0, int(math.ceil(self.wide_from_frac * total_planned_rows)) - rows_processed_host)
                        limit_rows = min(limit_rows, wide_row_ep)
                    if num_epochs == 1 and has_pre_close and not pre_close_captured:
                        limit_rows = min(limit_rows, int(0.55 * rows_in_ep))
                    if eff_head_edit and next_cp_idx < len(sorted_cps):
                        cp_row_ep = b_start + max(0, int(math.floor(sorted_cps[next_cp_idx] * total_planned_rows)) - rows_processed_host)
                        limit_rows = min(limit_rows, cp_row_ep)
                    if not past_all_revivals_and_cps and self.use_revive_stop:
                        if eff_wide_mult > 1:
                            freeze_ep = b_start + max(0, revive_stop_rows - rows_processed_host + cur_bsz_target)
                        else:
                            freeze_ep = b_start + max(0, (revive_stop_step - step_host + 1) * cur_bsz_target)
                        limit_rows = min(limit_rows, freeze_ep)

                    avail_steps = max(0, (limit_rows - b_start) // cur_bsz_target)
                    n_supers = avail_steps // super_g
                    rev_on_last = bool(revival_ws is not None and (not past_all_revivals_and_cps))
                    graph_key = (
                        super_g,
                        cur_bsz_target,
                        int(rows_in_ep),
                        bool(use_wide_now),
                        bool(track_oja_cur),
                        bool(track_fs_cur),
                        bool(rev_on_last),
                        bool(is_last_epoch and need_pass_accum),
                        bool(is_last_epoch and need_pass_accum and self.flush_pass_on_split),
                        bool(pre_close_captured),
                        int(ep_rep_mode),
                        bool(use_fp32_ep),
                        ep_order_assign.data_ptr() if ep_order_assign is not None else 0,
                        order_scatter.data_ptr() if order_scatter is not None else 0,
                    )
                    cached_g = graph_cache.get(graph_key)
                    # Reuse already-captured CUDAGraph at zero capture cost, or capture when launch savings beat instantiation!
                    min_supers = (
                        (3 if cur_bsz_target <= 8192 else 4)
                        if (cur_bsz_target * k) <= 8_500_000
                        else 6
                    )
                    if (cached_g is not None and n_supers >= 1) or (cached_g is None and n_supers >= min_supers):
                        _sync_slot0()
                        if cached_g is not None:
                            for _ in range(n_supers):
                                cached_g.replay()
                            self._graph_replays += n_supers
                        else:
                            ring = _make_superstep_ring_slots(super_g)

                            def _run_super_unit() -> None:
                                for sub_i, (s_in, s_out) in enumerate(ring):
                                    is_rev = rev_on_last and (sub_i == super_g - 1)
                                    _exec_one_step(
                                        in_s=s_in,
                                        out_s=s_out,
                                        b_cur_step=cur_bsz_target,
                                        use_wide_flag=use_wide_now,
                                        track_oja_now=track_oja_cur,
                                        do_feat_scale_now=track_fs_cur,
                                        do_revival_substep=is_rev,
                                        prep_in_ema_flag=(not is_rev),
                                        use_ivf_step=False,
                                    )

                            # Execute 1st super-step on `cap_stream` to warm allocator & compile exact specializations
                            cur_s = torch.cuda.current_stream(device)
                            cap_stream.wait_stream(cur_s)
                            with torch.cuda.stream(cap_stream):
                                _run_super_unit()
                            cur_s.wait_stream(cap_stream)

                            g = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(g, pool=graph_pool, stream=cap_stream):
                                _run_super_unit()
                            graph_cache[graph_key] = g
                            self._graphs_captured += 1

                            for _ in range(n_supers - 1):
                                g.replay()
                            self._graph_replays += max(0, n_supers - 1)

                        steps_done = n_supers * super_g
                        rows_done_burst = steps_done * cur_bsz_target
                        b_start += rows_done_burst
                        rows_processed_host += rows_done_burst
                        step_host += steps_done
                        cur_slot = 0
                        continue

                # Single eager step (alignment warmup, IVF exploration, transplant boundary, or epoch tail)
                b_end = min(b_start + cur_bsz_target, rows_in_ep)
                b_cur = b_end - b_start
                if use_ivf_now and ivf_explorer is not None and step_host > 0 and (step_host % self.ivf_rebuild_every == 0):
                    ivf_explorer.rebuild(centroids_c)

                within_rev_win = (
                    (rows_processed_host <= revive_stop_rows)
                    if (self.use_revive_stop and eff_wide_mult > 1)
                    else ((step_host <= revive_stop_step) if self.use_revive_stop else True)
                )
                will_revive = (
                    revival_ws is not None
                    and within_rev_win
                    and ((step_host + 1) >= max(1, eff_grace))
                    and ((step_host + 1) % self.check_unused_every == 0)
                )
                prog_after = float(rows_processed_host + b_cur) / float(total_planned_rows)
                will_transplant = (
                    eff_head_edit
                    and edit_gen is not None
                    and last_edit_step is not None
                    and edit_diag is not None
                    and next_cp_idx < len(sorted_cps)
                    and prog_after >= sorted_cps[next_cp_idx]
                    and ((not is_last_epoch and prog_after <= 0.85) or (num_epochs == 1 and prog_after <= 0.65))
                    and b_cur >= 32
                    and k >= 3
                )
                is_very_last = is_last_epoch and (b_end >= rows_in_ep)
                prep_in_ema = (not will_revive) and (not will_transplant) and (not is_very_last) and (not use_ivf_now)

                out_slot = 1 if cur_slot == 0 else 0
                _exec_one_step(
                    in_s=cur_slot,
                    out_s=out_slot,
                    b_cur_step=b_cur,
                    use_wide_flag=use_wide_now,
                    track_oja_now=track_oja_cur,
                    do_feat_scale_now=(track_fs_cur and within_rev_win),
                    do_revival_substep=will_revive,
                    prep_in_ema_flag=prep_in_ema,
                    use_ivf_step=use_ivf_now,
                )
                cur_slot = out_slot
                if use_ivf_now and will_revive and ivf_explorer is not None:
                    ivf_explorer.rebuild(centroids_c)

                if will_transplant:
                    while next_cp_idx < len(sorted_cps) and prog_after >= sorted_cps[next_cp_idx]:
                        next_cp_idx += 1
                    bo_prev = b_start
                    if order_scatter is not None:
                        b_ids_t = order_scatter[bo_prev : bo_prev + b_cur]
                        xb_raw_t = x_scatter_src[b_ids_t]
                    else:
                        xb_raw_t = x_scatter_src[bo_prev : bo_prev + b_cur]
                    if scatter_has_origin:
                        xb_c_t = ((xb_raw_t.double() - origin64[None, :]) * coord_scale).float().contiguous()
                    else:
                        xb_c_t = xb_raw_t.float().contiguous()

                    s_bias_cur = gpu_state[cur_slot, 4].clamp_min(1e-6).float()
                    ref_dist_m = (sse_ema.mean() / s_bias_cur).clamp_min(1e-30)
                    n_ed = run_apex_head_edit_checkpoint(
                        x_batch_c=xb_c_t,
                        centroids_c=centroids_c,
                        usage=usage,
                        sse_ema=sse_ema,
                        ref_distortion_mean=ref_dist_m,
                        survival_mass=survival_mass if track_surv else None,
                        prin_vec=prin_vec if want_oja else None,
                        slow_centroids=slow_centroids if track_slow else None,
                        slow_usage=slow_usage if track_slow else None,
                        pre_close_centroids=pre_close_centroids if (has_pre_close and pre_close_captured) else None,
                        pass_sums=pass_sums if is_last_epoch else None,
                        pass_counts=pass_counts if is_last_epoch else None,
                        pass_counts_sq=pass_counts_sq if is_last_epoch else None,
                        pass_sse=pass_sse if is_last_epoch else None,
                        last_edit_step=last_edit_step,
                        step=step_host,
                        compute_dtype=torch.bfloat16 if rep_mode == 3 else comp_dtype,
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
                    if not is_very_last:
                        _launch_apex_prep_centroids(
                            centroids_c, ep_c_rep, sc_buf, c_norm, ep_rep_mode, croiss_ws=ep_croiss_ws, croiss_ds=ep_croiss_ds
                        )

                if use_ivf_now and (b_end >= rows_in_ep) and (ep_idx + 1 == num_epochs - 1):
                    # Prepare exact represented centroids before entering the Exact Closing Pass!
                    next_fp32 = bool(high_cr_closing_fp32)
                    _launch_apex_prep_centroids(
                        centroids_c,
                        c_rep_f32 if next_fp32 else c_rep,
                        sc_buf,
                        c_norm,
                        0 if next_fp32 else rep_mode,
                        croiss_ws=None if next_fp32 else croiss_ws,
                        croiss_ds=None if next_fp32 else croiss_ds,
                    )

                b_start = b_end
                rows_processed_host += b_cur
                step_host += 1

            ep_sq_dists_gpu[ep_idx] = cur_ep_sq_gpu[0]
            ep_dot_counts_gpu[ep_idx] = torch.dot(raw_pass_counts, raw_pass_counts)
            ep_cum_splits_gpu[ep_idx] = total_splits_gpu
            ep_cum_prot_gpu[ep_idx] = total_prot_gpu
            ep_transplants.append(self.n_transplants - trans_before_ep)
            ev_epochs[ep_idx].record()

        # Read final `(usage_bias, alpha_last)` in 1 tiny packed read inside finalizer
        state_tail = gpu_state[cur_slot, 3:6].tolist()
        u_bias_fin = max(float(state_tail[0]), 1e-6)
        alpha_end_fin = max(float(state_tail[2]), 1e-4)

        c_ema_last = centroids_c.clone()
        centroids_fin, s_diag = apex_simplex_eb_finalize(
            pass_sums=pass_sums,
            pass_counts=pass_counts,
            pass_counts_sq=pass_counts_sq if eff_pass_span > 0.0 else None,
            pass_sse=pass_sse,
            live_centroids=c_ema_last,
            pre_close_centroids=pre_close_centroids if (has_pre_close and pre_close_captured) else None,
            slow_centroids=slow_centroids if track_slow else None,
            usage=usage,
            usage_inv_bias=1.0 / u_bias_fin,
            alpha_end=alpha_end_fin,
            nepochs=self.nepochs,
            d=d,
            mode=self.finalization,
            eb_tau=self.eb_tau,
            kappa_max=self.kappa_max,
        )
        self.shrinkage_diagnostics = s_diag
        self.slow_centroids = slow_centroids.clone() if track_slow else None
        self.pre_close_centroids = pre_close_centroids.clone() if (has_pre_close and pre_close_captured) else None

        assignments: torch.Tensor | None = None
        if capture_assignments:
            fin_mode = 0 if high_cr_closing_fp32 else rep_mode
            fin_c_rep = c_rep_f32 if high_cr_closing_fp32 else c_rep
            fin_ds = None if high_cr_closing_fp32 else croiss_ds
            fin_ws = None if high_cr_closing_fp32 else croiss_ws
            _launch_apex_prep_centroids(centroids_fin, fin_c_rep, sc_buf, c_norm, fin_mode, croiss_ws=fin_ws, croiss_ds=fin_ds)
            if do_preperm and perm0 is not None and not (high_cr_closing_fp32 and scatter_has_origin):
                _, perm_lbls = _launch_apex_assign(
                    x_src=x_scatter_src if high_cr_closing_fp32 else x_assign_src,
                    c_rep=fin_c_rep,
                    c_norm=c_norm,
                    rep_mode=fin_mode,
                    d=d,
                    sx_buf=None if high_cr_closing_fp32 else sx_buf,
                    sc_buf=sc_buf,
                    x_norm_buf=x_norm_f32 if high_cr_closing_fp32 else x_norm_buf,
                    use_sm120_kernel=self.use_sm120_kernel,
                    croiss_ds=fin_ds,
                    croiss_ws=fin_ws,
                )
                assignments = torch.empty((n,), dtype=torch.int64, device=device)
                assignments.scatter_(0, perm0, perm_lbls)
            else:
                fin_x = x_t if high_cr_closing_fp32 else x_assign_src
                fin_orig = True if high_cr_closing_fp32 else assign_has_origin
                fin_scale = (coord_scale != 1.0) if high_cr_closing_fp32 else assign_has_scale
                _, assignments = _launch_apex_assign(
                    x_src=fin_x,
                    c_rep=fin_c_rep,
                    c_norm=c_norm,
                    rep_mode=fin_mode,
                    d=d,
                    sx_buf=None if high_cr_closing_fp32 else sx_buf,
                    sc_buf=sc_buf,
                    x_norm_buf=None if high_cr_closing_fp32 else x_norm_buf,
                    origin=origin_assign if fin_orig else None,
                    coord_scale=coord_scale if fin_scale else 1.0,
                    use_sm120_kernel=self.use_sm120_kernel,
                    croiss_ds=fin_ds,
                    croiss_ws=fin_ws,
                )

        dead_t = (raw_pass_counts <= 0.5).sum()

        # Release captured graphs cleanly inside `train_time_sec` (`#190` fair timer accounting)
        for g in graph_cache.values():
            g.reset()
        graph_cache.clear()

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
                ApexEMeansEpochStat(
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

    def _fit_multi_gpu_entry(
        self,
        x: torch.Tensor | np.ndarray | list[torch.Tensor],
        init_centroids: torch.Tensor | np.ndarray | None,
        seeding_result: SeedingResult | None,
        capture_assignments: bool,
    ) -> tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray | None]:
        """Gear 4: Multi-GPU Data-Sharded CroissAttn-FP8/BF16 + Coalesced NCCL All-Reduce Engine."""
        input_is_list = isinstance(x, list)
        input_is_numpy = isinstance(x, np.ndarray)

        if self.devices is None:
            if input_is_list:
                self.devices = [s.device for s in x]  # type: ignore[union-attr]
            else:
                ndev = torch.cuda.device_count()
                self.devices = [torch.device(f"cuda:{i}") for i in range(max(1, ndev))]

        W = len(self.devices)
        if W <= 1 and not input_is_list:
            # Single-GPU fallback when only 1 device configured
            self_devs = self.devices
            self.devices = None
            try:
                x0 = x.to(self_devs[0]) if (isinstance(x, torch.Tensor) and self_devs) else x
                res = self.fit(
                    x0,
                    init_centroids=init_centroids,
                    seeding_result=seeding_result,
                    capture_assignments=capture_assignments,
                )
            finally:
                self.devices = self_devs
            return res

        if input_is_list:
            shards = [s.detach().contiguous() for s in x]  # type: ignore[union-attr]
            if len(shards) == 1:
                self.devices = None
                res = self.fit(
                    shards[0],
                    init_centroids=init_centroids,
                    seeding_result=seeding_result,
                    capture_assignments=capture_assignments,
                )
                self.devices = [shards[0].device]
                return res
            self.devices = [s.device for s in shards]
            W = len(shards)
        else:
            shards = self.prepopulate_shards(x)  # type: ignore[arg-type]

        n = sum(int(s.shape[0]) for s in shards)
        d = int(shards[0].shape[1])
        k = self.k
        if d != self.d:
            raise ValueError(f"Expected dimension d={self.d}, got {d}")
        if n < k:
            raise ValueError(f"Number of points n={n} must be >= k={k}")

        dev0 = self.devices[0]
        prec_name, comp_dtype = self._resolve_precision_and_dtype(n, d, k, dev0, shards[0].dtype)
        self.resolved_precision = prec_name
        self.selected_gear = f"multi_gpu_{W}x_{prec_name}_nccl"
        rep_mode = _rep_mode_from_precision(prec_name, comp_dtype)

        eff_threshold = (
            0.0
            if self.revival_mode == "off"
            else (float(self.unused_threshold) if self.unused_threshold is not None else (DEFAULT_UNUSED_THRESHOLD if self.lowd_revival else None))
        )
        base_recipe = build_recipe(
            n=n,
            d=d,
            k=k,
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
        eff_warm_scale = (
            0.95
            if (self._warm_alpha_auto and self.use_prior_mass and (float(n) / float(max(1, self.k))) >= 180.0)
            else self.warm_alpha_scale
        )
        eff_alpha0 = (
            float(min(max(base_recipe.alpha0 * eff_warm_scale, 1e-4), 1.0))
            if (self.alpha0 is None and eff_warm_scale != 1.0)
            else float(base_recipe.alpha0)
        )
        recipe = (
            build_recipe(
                n=n,
                d=d,
                k=k,
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

        for dv in self.devices:
            torch.cuda.synchronize(dv)
        t_init_start = time.perf_counter()

        dev0_ctx = torch.cuda.device(dev0) if dev0.type == "cuda" else contextlib.nullcontext()
        with dev0_ctx:
            seed_comp_dtype = torch.bfloat16 if rep_mode == 3 else comp_dtype
            origin64_0, coord_scale = _compute_origin_and_scale(shards[0], seed_comp_dtype)
            self._origin64 = origin64_0
            self._coord_scale = coord_scale

            need_priors = self.use_prior_mass or self.use_voronoi_polish
            if seeding_result is not None:
                c_in_c = ((seeding_result.centroids.to(device=dev0, dtype=torch.float64) - origin64_0[None, :]) * coord_scale).float().contiguous()
                seed_res0 = SeedingResult(
                    centroids=c_in_c,
                    prior_probs=seeding_result.prior_probs.to(device=dev0, dtype=torch.float32).contiguous(),
                    prior_distortion=(
                        seeding_result.prior_distortion.to(device=dev0, dtype=torch.float32) * (coord_scale * coord_scale)
                    ).contiguous(),
                )
            elif init_centroids is not None:
                c_in = (
                    torch.from_numpy(np.asarray(init_centroids, dtype=np.float64))
                    if isinstance(init_centroids, np.ndarray)
                    else init_centroids.detach()
                )
                raw_c = ((c_in.to(device=dev0, dtype=torch.float64) - origin64_0[None, :]) * coord_scale).float().contiguous()
                seed_res0 = SeedingResult(
                    centroids=raw_c,
                    prior_probs=torch.full((k,), 1.0 / k, dtype=torch.float32, device=dev0),
                    prior_distortion=torch.zeros((k,), dtype=torch.float32, device=dev0),
                )
                need_priors = False
            else:
                # Pool lightweight subsample across shards onto dev0 if shards[0] < 2*k, else fast shard-0 coreset
                if int(shards[0].shape[0]) >= max(k * 2, 8192):
                    x_seed_pool = shards[0]
                else:
                    per_s = max(k, (max(k * 4, 16384) + W - 1) // W)
                    x_seed_pool = torch.cat([s[: min(int(s.shape[0]), per_s)].to(dev0) for s in shards], dim=0).contiguous()
                seed_res0 = apex_turkpp_init_centered(
                    x_t=x_seed_pool,
                    origin64=origin64_0,
                    coord_scale=coord_scale,
                    k=k,
                    comp_dtype=seed_comp_dtype,
                    coreset_mult=self.coreset_mult,
                    min_coreset_size=self.min_coreset_size,
                    polish=self.use_voronoi_polish,
                    seed=self.seed,
                )
            self.seeding_result = seed_res0

        for dv in self.devices:
            torch.cuda.synchronize(dv)
        self.init_time_sec = time.perf_counter() - t_init_start
        t_train_start = time.perf_counter()

        streams = [torch.cuda.Stream(device=dv) for dv in self.devices]
        return self._run_multi_gpu_aligned_loop(
            shards=shards,
            streams=streams,
            recipe=recipe,
            seed_res0=seed_res0,
            need_priors=need_priors,
            origin64_0=origin64_0,
            coord_scale=coord_scale,
            rep_mode=rep_mode,
            comp_dtype=comp_dtype,
            t_train_start=t_train_start,
            capture_assignments=capture_assignments,
            input_is_list=input_is_list,
            input_is_numpy=input_is_numpy,
        )

    def _run_multi_gpu_aligned_loop(
        self,
        shards: list[torch.Tensor],
        streams: list[torch.cuda.Stream],
        recipe: EMeansRecipe,
        seed_res0: SeedingResult,
        need_priors: bool,
        origin64_0: torch.Tensor,
        coord_scale: float,
        rep_mode: int,
        comp_dtype: torch.dtype,
        t_train_start: float,
        capture_assignments: bool,
        input_is_list: bool,
        input_is_numpy: bool,
    ) -> tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray | None]:
        """Stride-aligned Multi-GPU NCCL training loop (`1` coalesced `all_reduce` per step, `0` step host syncs)."""
        assert self.devices is not None
        W = len(self.devices)
        dev0 = self.devices[0]
        use_hw_nccl = all(dv.type == "cuda" for dv in self.devices) and (len(set(self.devices)) == W)

        def _collective_all_reduce(bufs: list[torch.Tensor]) -> None:
            if use_hw_nccl:
                torch.cuda.nccl.all_reduce(bufs, streams=streams)
            else:
                for w_i in range(1, W):
                    streams[0].wait_stream(streams[w_i])
                with torch.cuda.device(dev0), torch.cuda.stream(streams[0]):
                    for w_i in range(1, W):
                        bufs[0].add_(bufs[w_i].to(dev0))
                for w_i in range(1, W):
                    streams[w_i].wait_stream(streams[0])
                    with torch.cuda.device(self.devices[w_i]), torch.cuda.stream(streams[w_i]):  # type: ignore[index]
                        bufs[w_i].copy_(bufs[0])

        def _collective_broadcast(bufs: list[torch.Tensor]) -> None:
            if use_hw_nccl:
                torch.cuda.nccl.broadcast(bufs, root=0, streams=streams)
            else:
                for w_i in range(1, W):
                    streams[w_i].wait_stream(streams[0])
                    with torch.cuda.device(self.devices[w_i]), torch.cuda.stream(streams[w_i]):  # type: ignore[index]
                        bufs[w_i].copy_(bufs[0])
        n = sum(int(s.shape[0]) for s in shards)
        d = self.d
        k = self.k

        eff_wide_mult = self._resolve_wide_mult(recipe)
        eff_pass_span = self._effective_pass_span(n)
        want_oja = self.split_mode in ("half_gaussian_oja", "oja_only") or self.donor_mode == "oja_sse_usage"
        track_surv = bool(self.use_huygens_protection and self.gamma_protect > 0.0)
        track_slow = self.finalization in ("simplex_eb", "eb_slow_only")
        has_pre_close = self.finalization == "simplex_eb"
        need_pass_accum = self.finalization in ("simplex_eb", "eb_slow_only", "eb_ema", "closing_mean")

        bsz_float = float(recipe.batch_size)
        min_prob = max(0.0, self.prior_floor) / float(k)
        p0_dev0 = seed_res0.prior_probs.to(device=dev0, dtype=torch.float32)
        e0_dev0 = seed_res0.prior_distortion.to(device=dev0, dtype=torch.float32)
        mean_eh0 = e0_dev0.mean().clamp_min(1e-12)
        if self.use_prior_mass and need_priors:
            u0_master = (self.prior_mass_scale * bsz_float * p0_dev0.clamp_min(min_prob)).float().contiguous()
            e0_master = (self.prior_mass_scale * bsz_float * e0_dev0.clamp_min(1e-6 * mean_eh0)).float().contiguous()
            init_u_bias = float(self.prior_mass_scale)
            init_s_bias = float(self.prior_mass_scale)
        else:
            u0_master = torch.full((k,), recipe.init_usage, dtype=torch.float32, device=dev0)
            e0_master = torch.zeros((k,), dtype=torch.float32, device=dev0)
            init_u_bias = float(recipe.init_usage * k) / max(1.0, bsz_float)
            init_s_bias = 0.0

        pca_gen = torch.Generator(device=dev0)
        pca_gen.manual_seed((self.seed + 31337) % (2**32))
        init_dirs0 = torch.randn((k, d), generator=pca_gen, device=dev0, dtype=torch.float32)
        init_dirs0 = init_dirs0 / init_dirs0.norm(dim=1, keepdim=True).clamp_min(1e-12)
        init_rms0 = (
            (e0_dev0 / (p0_dev0.clamp_min(1e-6) * float(max(1, d)))).clamp_min(1e-8).sqrt()[:, None]
            if need_priors
            else 1e-3
        )
        pv0_master = (init_rms0 * init_dirs0).contiguous()

        origin64_w = [origin64_0.to(dv).contiguous() for dv in self.devices]
        origin32_w = [origin64_0.float().to(dv).contiguous() for dv in self.devices]
        centroids_w = [seed_res0.centroids.to(dv, dtype=torch.float32).clone().contiguous() for dv in self.devices]
        usage_w = [u0_master.to(dv).clone().contiguous() for dv in self.devices]
        sse_w = [e0_master.to(dv).clone().contiguous() for dv in self.devices]
        surv_w = [
            ((bsz_float * p0_dev0.to(dv)).float().contiguous() if need_priors else torch.zeros((k,), dtype=torch.float32, device=dv))
            if track_surv
            else centroids_w[w]
            for w, dv in enumerate(self.devices)
        ]
        slow_c_w = [centroids_w[w].clone() if track_slow else centroids_w[w] for w in range(W)]
        slow_u_w = [usage_w[w].clone() if track_slow else usage_w[w] for w in range(W)]
        pre_close_w = [centroids_w[w].clone() if has_pre_close else centroids_w[w] for w in range(W)]
        prin_w = [pv0_master.to(dv).clone().contiguous() if want_oja else centroids_w[w] for w, dv in enumerate(self.devices)]

        # Flat 1D communication buffer: `[sums (k*d), g_sums (k*d if want_oja), counts (k), batch_sse (k)]`
        kd = k * d
        comm_len = (2 * kd + 2 * k) if want_oja else (kd + 2 * k)
        comm_buf_w = [torch.zeros((comm_len,), dtype=torch.float32, device=dv) for dv in self.devices]
        sums_w = [cb[:kd].view(k, d) for cb in comm_buf_w]
        g_sums_w = [cb[kd : 2 * kd].view(k, d) if want_oja else sums_w[w] for w, cb in enumerate(comm_buf_w)]
        counts_w = [cb[-2 * k : -k] for cb in comm_buf_w]
        bsse_w = [cb[-k:] for cb in comm_buf_w]

        feat_scale_w = [torch.ones((d,), dtype=torch.float32, device=dv) for dv in self.devices]
        ones_d_w = [torch.ones((d,), dtype=torch.float32, device=dv) for dv in self.devices]
        pass_sums_w = [torch.zeros((k, d), dtype=torch.float32, device=dv) for dv in self.devices]
        pass_cnt_w = [torch.zeros((k,), dtype=torch.float64, device=dv) for dv in self.devices]
        pass_cnt_sq_w = [torch.zeros((k,), dtype=torch.float64, device=dv) for dv in self.devices]
        pass_sse_w = [torch.zeros((k,), dtype=torch.float64, device=dv) for dv in self.devices]
        raw_cnt_w = [torch.zeros((k,), dtype=torch.float64, device=dv) for dv in self.devices]

        c_pad_align = 64 if rep_mode == 3 else 16
        c_rep_dt = FLOAT8_E4M3FN if rep_mode == 3 else comp_dtype
        c_rep_w = [_alloc_padded_2d(k, d, c_rep_dt, dv, align=c_pad_align) for dv in self.devices]
        sc_w = [torch.ones((k,), dtype=torch.float32, device=dv) for dv in self.devices]
        c_norm_w = [torch.empty((k,), dtype=torch.float32, device=dv) for dv in self.devices]

        gpu_state_w = [torch.zeros((2, 8), dtype=torch.float64, device=dv) for dv in self.devices]
        batch_off_w = [torch.zeros((2,), dtype=torch.int64, device=dv) for dv in self.devices]
        dummy_boff_out_w = [torch.zeros((), dtype=torch.int64, device=dv) for dv in self.devices]
        for w in range(W):
            gpu_state_w[w][0, 3] = init_u_bias
            gpu_state_w[w][0, 4] = init_s_bias
            gpu_state_w[w][0, 5] = recipe.alpha0 * recipe.alpha_min_factor

        rev_ws_w = [
            _ApexRevivalWorkspace.allocate(k, d, dv)
            if (recipe.unused_threshold > 0.0 and self.revival_mode == "split" and k >= 2)
            else None
            for dv in self.devices
        ]

        num_epochs = len(recipe.epoch_rows)
        ep_sq_w = [torch.zeros((num_epochs,), dtype=torch.float64, device=dv) for dv in self.devices]
        tot_sp_w = [torch.zeros((), dtype=torch.int64, device=dv) for dv in self.devices]
        tot_pr_w = [torch.zeros((), dtype=torch.int64, device=dv) for dv in self.devices]

        nw_list = [int(s.shape[0]) for s in shards]
        max_nw = max(nw_list) if nw_list else 1
        bsz_max_global = min(n, recipe.batch_size * max(1, eff_wide_mult))
        bsz_max_local = int(math.ceil(bsz_max_global * float(max_nw) / float(max(1, n)))) + 64
        dist_w = [torch.empty((bsz_max_local,), dtype=torch.float32, device=dv) for dv in self.devices]
        lbl_w = [torch.empty((bsz_max_local,), dtype=torch.int64, device=dv) for dv in self.devices]

        x_assign_w: list[torch.Tensor] = [shards[0]] * W
        x_scatter_w: list[torch.Tensor] = [shards[0]] * W
        sx_shard_w: list[torch.Tensor | None] = [None] * W
        xn_shard_w: list[torch.Tensor | None] = [None] * W
        perm0_w: list[torch.Tensor | None] = [None] * W
        sc_has_orig_w: list[bool] = [True] * W
        croiss_ds_w: list[Any] = [None] * W
        croiss_ws_w: list[Any] = [None] * W

        cmod = _get_croiss_fp8_module() if rep_mode == 3 else None

        t_pre0 = time.perf_counter()
        for w, dv in enumerate(self.devices):
            streams[w].wait_stream(torch.cuda.current_stream(dv))
            with torch.cuda.device(dv), torch.cuda.stream(streams[w]):
                nw = int(shards[w].shape[0])
                free_w = int(torch.cuda.mem_get_info(dv)[0]) if dv.type == "cuda" else 0
                max_preperm_w = min(5_600_000_000, max(1_100_000_000, int(0.28 * free_w)))
                do_perm_w = (nw * d * 4 <= max_preperm_w) and (self.streaming_mode != "fused_gather")
                orig_ker_w = (
                    origin64_w[w]
                    if (shards[w].dtype == torch.float64 or coord_scale != 1.0 or float(origin64_0.abs().max().item()) > 1e3)
                    else origin32_w[w]
                )
                if do_perm_w:
                    gp = torch.Generator(device=dv)
                    gp.manual_seed(epoch_seed(self.seed + 104729 * w, 0))
                    p0w = torch.randperm(nw, generator=gp, device=dv)
                    perm0_w[w] = p0w
                    xf32_w = torch.empty((nw, d), dtype=torch.float32, device=dv)
                    xrep_w = _alloc_padded_2d(nw, d, c_rep_dt, dv, align=c_pad_align)
                    sx_loc = torch.empty((nw,), dtype=torch.float32, device=dv) if rep_mode == 3 else None
                    xn_loc = torch.empty((nw,), dtype=torch.float32, device=dv)
                    blk_d_e = min(128, max(32 if rep_mode == 3 else 16, triton.next_power_of_2(max(1, d))))
                    _apex_prepermute_encode_kernel[(triton.cdiv(nw, 64),)](
                        shards[w],
                        p0w,
                        orig_ker_w,
                        xf32_w,
                        xrep_w,
                        sx_loc if sx_loc is not None else xn_loc,
                        xn_loc,
                        nw,
                        d,
                        shards[w].stride(0),
                        shards[w].stride(1),
                        xf32_w.stride(0),
                        xrep_w.stride(0),
                        float(coord_scale),
                        HAS_PERM=True,
                        ORIGIN_FP64=(orig_ker_w.dtype == torch.float64),
                        HAS_SCALE=(coord_scale != 1.0),
                        WRITE_F32=True,
                        REP_MODE=rep_mode,
                        BLOCK_N=64,
                        BLOCK_D=blk_d_e,
                        num_warps=4,
                    )
                    if rep_mode == 3 and cmod is not None:
                        ds_loc = cmod.croiss_encode_dataset(xf32_w, center=False, coord_scale=1.0)
                        ws_loc = cmod.CroissFP8Workspace.allocate(
                            k=k, d=d, max_batch_size=bsz_max_local, device=dv, padded_d=ds_loc.padded_d
                        )
                        croiss_ds_w[w] = ds_loc
                        croiss_ws_w[w] = ws_loc
                        c_rep_w[w] = ws_loc.qc
                        sc_w[w] = ws_loc.sc
                        c_norm_w[w] = ws_loc.c_norm
                        x_assign_w[w] = ds_loc.qx
                        sx_shard_w[w] = ds_loc.sx
                        xn_shard_w[w] = ds_loc.x_norm
                    else:
                        x_assign_w[w] = xrep_w
                        sx_shard_w[w] = sx_loc
                        xn_shard_w[w] = xn_loc
                    x_scatter_w[w] = xf32_w
                    sc_has_orig_w[w] = False
                elif rep_mode == 3 and cmod is not None:
                    ds_loc = cmod.croiss_encode_dataset(
                        shards[w], mu=origin64_w[w], center=True, coord_scale=coord_scale
                    )
                    ws_loc = cmod.CroissFP8Workspace.allocate(
                        k=k, d=d, max_batch_size=bsz_max_local, device=dv, padded_d=ds_loc.padded_d
                    )
                    croiss_ds_w[w] = ds_loc
                    croiss_ws_w[w] = ws_loc
                    c_rep_w[w] = ws_loc.qc
                    sc_w[w] = ws_loc.sc
                    c_norm_w[w] = ws_loc.c_norm
                    x_assign_w[w] = ds_loc.qx
                    sx_shard_w[w] = ds_loc.sx
                    xn_shard_w[w] = ds_loc.x_norm
                    x_scatter_w[w] = shards[w]
                    sc_has_orig_w[w] = True
                else:
                    x_assign_w[w] = shards[w]
                    x_scatter_w[w] = shards[w]
                    sc_has_orig_w[w] = True

                _launch_apex_prep_centroids(
                    centroids_w[w], c_rep_w[w], sc_w[w], c_norm_w[w], rep_mode, croiss_ws=croiss_ws_w[w]
                )

        self.preencode_time_sec = time.perf_counter() - t_pre0

        sched_code = {"emeans_cosine": 0, "power_cosine": 1, "constant": 2}[self.schedule_mode]
        donor_code = {"oja_sse_usage": 0, "sse_usage": 1, "d2": 2, "usage_only": 3}[self.donor_mode]
        split_code = {
            "half_gaussian_oja": 0,
            "rms_aniso": 1,
            "rms_iso": 1,
            "oja_only": 2,
            "checkerboard": 3,
        }[self.split_mode]
        alpha_slow_base = (
            min(1.0, max(1e-6, self.slow_ema_mult * recipe.alpha_min_factor * recipe.alpha0))
            if track_slow
            else 0.0
        )
        eff_grace = (
            min(self.grace_steps, max(0, recipe.total_steps // 3))
            if (self.use_prior_mass and need_priors and track_surv)
            else 0
        )
        total_planned_rows = max(1, sum(recipe.epoch_rows))
        revive_stop_step = int(self.revive_stop_frac * recipe.total_steps)
        revive_stop_rows = int(self.revive_stop_frac * total_planned_rows)
        wide_alpha_factor = float(eff_wide_mult ** ALPHA_LAW[2]) if eff_wide_mult > 1 else 1.0
        blk_d_upd = min(256, max(16, triton.next_power_of_2(max(1, d))))
        blk_d_fs = min(128, max(16, triton.next_power_of_2(max(1, d))))
        blk_k_fs = min(64, max(16, triton.next_power_of_2(max(1, k))))

        in_wide_phase = False
        pre_close_captured = False
        cur_slot = 0
        rows_processed_host = 0
        step_host = 0
        ep_dot_gpu0 = torch.zeros((num_epochs,), dtype=torch.float64, device=dev0)
        ep_sp_gpu0 = torch.zeros((num_epochs,), dtype=torch.int64, device=dev0)
        ep_pr_gpu0 = torch.zeros((num_epochs,), dtype=torch.int64, device=dev0)

        for ep_idx, rows_in_ep in enumerate(recipe.epoch_rows):
            is_last_epoch = ep_idx == (num_epochs - 1)
            orders_w: list[torch.Tensor | None] = []
            rows_w_ep: list[int] = []
            cum_nw = 0
            for w in range(W):
                next_cum = cum_nw + nw_list[w]
                rw_prop = ((rows_in_ep * next_cum) // max(1, n)) - ((rows_in_ep * cum_nw) // max(1, n))
                cum_nw = next_cum
                rows_w_ep.append(max(0, rw_prop))

            for w, dv in enumerate(self.devices):
                with torch.cuda.device(dv), torch.cuda.stream(streams[w]):
                    if is_last_epoch and num_epochs >= 2 and has_pre_close and not pre_close_captured:
                        pre_close_w[w].copy_(centroids_w[w])
                    if cur_slot != 0:
                        gpu_state_w[w][0].copy_(gpu_state_w[w][cur_slot])
                    gpu_state_w[w][0, 2] = 0.0
                    raw_cnt_w[w].zero_()
                    nw = nw_list[w]
                    rows_w = rows_w_ep[w]
                    if perm0_w[w] is not None and ep_idx == 0 and rows_w <= nw:
                        orders_w.append(None)
                    else:
                        gw = torch.Generator(device=dv)
                        gw.manual_seed(epoch_seed(self.seed + 104729 * w, ep_idx))
                        orders_w.append(block_permute_indices(nw, rows_w, self.block_permute_size, gw, dv))
            if is_last_epoch and num_epochs >= 2 and has_pre_close:
                pre_close_captured = True
            cur_slot = 0

            g0 = 0
            while g0 < rows_in_ep:
                if num_epochs == 1 and has_pre_close and (not pre_close_captured) and g0 >= int(0.55 * rows_in_ep):
                    for w, dv in enumerate(self.devices):
                        with torch.cuda.device(dv), torch.cuda.stream(streams[w]):
                            pre_close_w[w].copy_(centroids_w[w])
                    pre_close_captured = True

                prog = float(rows_processed_host) / float(total_planned_rows)
                use_wide_now = eff_wide_mult > 1 and prog >= self.wide_from_frac
                cur_bsz_target = min(n, recipe.batch_size * eff_wide_mult) if use_wide_now else recipe.batch_size
                if use_wide_now and not in_wide_phase:
                    w_sc = float(cur_bsz_target) / float(max(1, recipe.batch_size))
                    for w, dv in enumerate(self.devices):
                        with torch.cuda.device(dv), torch.cuda.stream(streams[w]):
                            usage_w[w].mul_(w_sc)
                            sse_w[w].mul_(w_sc)
                            if track_surv:
                                surv_w[w].mul_(w_sc)
                            if track_slow:
                                slow_u_w[w].mul_(w_sc)
                    in_wide_phase = True

                g1 = min(g0 + cur_bsz_target, rows_in_ep)
                slice_bounds_w = [
                    ((g0 * rows_w_ep[w]) // max(1, rows_in_ep), (g1 * rows_w_ep[w]) // max(1, rows_in_ep))
                    for w in range(W)
                ]
                b_global = max(1, sum(l1_w - l0_w for l0_w, l1_w in slice_bounds_w))
                within_rev_win = (
                    (rows_processed_host <= revive_stop_rows)
                    if (self.use_revive_stop and eff_wide_mult > 1)
                    else ((step_host <= revive_stop_step) if self.use_revive_stop else True)
                )
                track_oja_cur = want_oja and within_rev_win
                track_fs_cur = rev_ws_w[0] is not None and self.use_feat_scale and split_code in (0, 1) and within_rev_win
                will_revive = (
                    rev_ws_w[0] is not None
                    and within_rev_win
                    and ((step_host + 1) >= max(1, eff_grace))
                    and ((step_host + 1) % self.check_unused_every == 0)
                )
                is_very_last = is_last_epoch and (g1 >= rows_in_ep)
                prep_in_ema = (not will_revive) and (not is_very_last)
                eff_peak_now = min(1.0, recipe.alpha0 * (wide_alpha_factor if use_wide_now else 1.0))
                out_slot = 1 if cur_slot == 0 else 0

                # 1. Parallel local Tensor-Core assign + local scatter on each shard
                for w, dv in enumerate(self.devices):
                    with torch.cuda.device(dv), torch.cuda.stream(streams[w]):
                        l0, l1 = slice_bounds_w[w]
                        b_loc = l1 - l0
                        if b_loc > 0:
                            batch_off_w[w][cur_slot] = l0
                            d_vw = dist_w[w][:b_loc]
                            l_vw = lbl_w[w][:b_loc]
                            has_orig_a = sc_has_orig_w[w] and (rep_mode != 3)
                            _launch_apex_assign(
                                x_src=x_assign_w[w],
                                c_rep=c_rep_w[w],
                                c_norm=c_norm_w[w],
                                rep_mode=rep_mode,
                                d=d,
                                sx_buf=sx_shard_w[w],
                                sc_buf=sc_w[w],
                                x_norm_buf=xn_shard_w[w],
                                row_indices=orders_w[w],
                                batch_offset_ptr=batch_off_w[w][cur_slot],
                                batch_size=b_loc,
                                wrap_rows=False,
                                origin=origin64_w[w] if has_orig_a else None,
                                coord_scale=coord_scale if has_orig_a else 1.0,
                                out_dist=d_vw,
                                out_label=l_vw,
                                use_sm120_kernel=self.use_sm120_kernel,
                                croiss_ds=croiss_ds_w[w],
                                croiss_ws=croiss_ws_w[w],
                            )

                            full_d_oja = track_oja_cur and (d <= 256)
                            blk_b_sc = (64 if d <= 128 else 32) if track_oja_cur else 128
                            blk_d_sc = (
                                max(16, triton.next_power_of_2(max(1, d)))
                                if full_d_oja
                                else (128 if track_oja_cur else min(128, max(16, triton.next_power_of_2(max(1, d)))))
                            )
                            sc_tot_w = int(orders_w[w].shape[0]) if orders_w[w] is not None else int(x_scatter_w[w].shape[0])
                            _apex_scatter_oja_sse_kernel[(triton.cdiv(b_loc, blk_b_sc), 1 if track_oja_cur else triton.cdiv(d, blk_d_sc))](
                                x_scatter_w[w],
                                orders_w[w] if orders_w[w] is not None else l_vw,
                                batch_off_w[w][cur_slot],
                                origin64_w[w],
                                l_vw,
                                d_vw,
                                centroids_w[w],
                                prin_w[w],
                                sums_w[w],
                                counts_w[w],
                                bsse_w[w],
                                g_sums_w[w],
                                ep_sq_w[w],
                                int(ep_idx),
                                b_loc,
                                d,
                                sc_tot_w,
                                x_scatter_w[w].stride(0),
                                x_scatter_w[w].stride(1),
                                centroids_w[w].stride(0),
                                float(coord_scale),
                                HAS_INDICES=(orders_w[w] is not None),
                                HAS_BATCH_OFFSET=True,
                                WRAP_ROWS=False,
                                HAS_ORIGIN=sc_has_orig_w[w],
                                HAS_SCALE=(sc_has_orig_w[w] and coord_scale != 1.0),
                                ORIGIN_FP64=True,
                                TRACK_OJA=track_oja_cur,
                                FULL_D=full_d_oja,
                                BLOCK_B=blk_b_sc,
                                BLOCK_D=blk_d_sc,
                                num_warps=8 if (track_oja_cur and d >= 64) else 4,
                            )

                # 2. Single coalesced hardware NCCL all-reduce across all W GPUs
                _collective_all_reduce(comm_buf_w)

                # 3. Replica-identical EMA + Oja + SlowEMA + ClosingPass + Prep on each GPU
                for w, dv in enumerate(self.devices):
                    with torch.cuda.device(dv), torch.cuda.stream(streams[w]):
                        if track_fs_cur:
                            _apex_feat_scale_kernel[(triton.cdiv(d, blk_d_fs),)](
                                sums_w[w],
                                counts_w[w],
                                feat_scale_w[w],
                                k,
                                d,
                                sums_w[w].stride(0),
                                float(self.var_beta),
                                BLOCK_K=blk_k_fs,
                                BLOCK_D=blk_d_fs,
                                num_warps=4,
                            )
                        _apex_ema_oja_slow_pass_prep_kernel[(k,)](
                            gpu_state_w[w][cur_slot],
                            gpu_state_w[w][out_slot],
                            batch_off_w[w][cur_slot],
                            dummy_boff_out_w[w],
                            sums_w[w],
                            counts_w[w],
                            bsse_w[w],
                            g_sums_w[w],
                            centroids_w[w],
                            usage_w[w],
                            sse_w[w],
                            prin_w[w],
                            surv_w[w],
                            slow_c_w[w],
                            slow_u_w[w],
                            pass_sums_w[w],
                            pass_cnt_w[w],
                            pass_cnt_sq_w[w],
                            pass_sse_w[w],
                            raw_cnt_w[w],
                            c_rep_w[w],
                            sc_w[w],
                            c_norm_w[w],
                            k,
                            d,
                            centroids_w[w].stride(0),
                            c_rep_w[w].stride(0),
                            int(b_global),
                            int(n),
                            int(rows_in_ep),
                            int(recipe.total_steps),
                            int(total_planned_rows),
                            float(eff_peak_now),
                            float(recipe.alpha_min_factor),
                            float(self.alpha_floor),
                            float(self.cosine_power),
                            float(self.sse_decay),
                            float(alpha_slow_base),
                            float(eff_pass_span),
                            int(bool(is_last_epoch and need_pass_accum)),
                            int(prep_in_ema),
                            SCHED_MODE=sched_code,
                            USE_WIDE_SCHED=int(eff_wide_mult > 1),
                            COUPLE_SSE=self.couple_sse_to_alpha,
                            TRACK_OJA=track_oja_cur,
                            TRACK_SURV=track_surv,
                            TRACK_SLOW=track_slow,
                            REP_MODE=rep_mode,
                            BLOCK_D=blk_d_upd,
                            num_warps=2 if d <= 128 else 4,
                        )
                        if will_revive and rev_ws_w[w] is not None:
                            _launch_apex_cudagraph_revival(
                                state_slot=gpu_state_w[w][out_slot],
                                centroids_c=centroids_w[w],
                                origin32=origin32_w[w],
                                usage=usage_w[w],
                                sse_ema=sse_w[w],
                                prin_vec=prin_w[w],
                                survival_mass=surv_w[w],
                                slow_centroids=slow_c_w[w],
                                slow_usage=slow_u_w[w],
                                pre_close_centroids=pre_close_w[w],
                                pass_sums=pass_sums_w[w],
                                pass_counts=pass_cnt_w[w],
                                pass_counts_sq=pass_cnt_sq_w[w],
                                pass_sse=pass_sse_w[w],
                                feat_scale=feat_scale_w[w],
                                ones_d=ones_d_w[w],
                                ws=rev_ws_w[w],
                                total_splits_gpu=tot_sp_w[w],
                                total_prot_gpu=tot_pr_w[w],
                                b_cur=b_global,
                                unused_threshold=recipe.unused_threshold,
                                gamma_protect=self.gamma_protect,
                                eff_grace=eff_grace,
                                revive_stop_step=revive_stop_step,
                                revive_stop_rows=revive_stop_rows,
                                donor_usage_w=self.donor_usage_w,
                                split_mult=self.split_mult,
                                split_eps=recipe.split_epsilon,
                                base_seed=self.seed,
                                use_revive_stop=self.use_revive_stop,
                                use_wide_stop=(eff_wide_mult > 1),
                                track_surv=track_surv,
                                track_oja=track_oja_cur,
                                track_slow=track_slow,
                                has_pre_close=has_pre_close,
                                update_pre_close=pre_close_captured,
                                use_feat_scale=self.use_feat_scale,
                                flush_closing=bool(is_last_epoch and need_pass_accum and self.flush_pass_on_split),
                                donor_mode_code=donor_code,
                                split_mode_code=split_code,
                                rep_mode=rep_mode,
                            )

                if will_revive:
                    # Broadcast rank-0 centroids, usage, SSE & Oja axes after split so replicas stay 100% bit-identical!
                    _collective_broadcast(centroids_w)
                    _collective_broadcast(usage_w)
                    _collective_broadcast(sse_w)
                    if want_oja:
                        _collective_broadcast(prin_w)
                    if not is_very_last:
                        for w, dv in enumerate(self.devices):
                            with torch.cuda.device(dv), torch.cuda.stream(streams[w]):
                                _launch_apex_prep_centroids(
                                    centroids_w[w], c_rep_w[w], sc_w[w], c_norm_w[w], rep_mode, croiss_ws=croiss_ws_w[w]
                                )

                cur_slot = out_slot
                g0 = g1
                rows_processed_host += b_global
                step_host += 1

            with torch.cuda.device(dev0), torch.cuda.stream(streams[0]):
                ep_dot_gpu0[ep_idx] = torch.dot(raw_cnt_w[0], raw_cnt_w[0])
                ep_sp_gpu0[ep_idx] = tot_sp_w[0]
                ep_pr_gpu0[ep_idx] = tot_pr_w[0]

        _collective_all_reduce(ep_sq_w)
        for w, dv in enumerate(self.devices):
            streams[w].synchronize()
            torch.cuda.current_stream(dv).wait_stream(streams[w])

        state_tail0 = gpu_state_w[0][cur_slot, 3:6].tolist()
        u_bias_fin = max(float(state_tail0[0]), 1e-6)
        alpha_end_fin = max(float(state_tail0[2]), 1e-4)

        with torch.cuda.device(dev0):
            centroids_fin0, s_diag = apex_simplex_eb_finalize(
                pass_sums=pass_sums_w[0],
                pass_counts=pass_cnt_w[0],
                pass_counts_sq=pass_cnt_sq_w[0] if eff_pass_span > 0.0 else None,
                pass_sse=pass_sse_w[0],
                live_centroids=centroids_w[0],
                pre_close_centroids=pre_close_w[0] if (has_pre_close and pre_close_captured) else None,
                slow_centroids=slow_c_w[0] if track_slow else None,
                usage=usage_w[0],
                usage_inv_bias=1.0 / u_bias_fin,
                alpha_end=alpha_end_fin,
                nepochs=self.nepochs,
                d=d,
                mode=self.finalization,
                eb_tau=self.eb_tau,
                kappa_max=self.kappa_max,
            )

        self.shrinkage_diagnostics = s_diag
        inv_scale = 1.0 / coord_scale
        c_abs0 = (centroids_fin0.double() * inv_scale + origin64_0[None, :]).float().contiguous()
        self.centroids = c_abs0
        self.centroids_per_gpu = [c_abs0.to(dv).contiguous() for dv in self.devices]
        dead_t0 = (raw_cnt_w[0] <= 0.5).sum()

        assignments_out: torch.Tensor | None = None
        if capture_assignments:
            lbl_parts = [torch.empty((int(s.shape[0]),), dtype=torch.int64, device=s.device) for s in shards]
            for w, dv in enumerate(self.devices):
                with torch.cuda.device(dv), torch.cuda.stream(streams[w]):
                    cw_c = centroids_fin0.to(dv)
                    _launch_apex_prep_centroids(
                        cw_c, c_rep_w[w], sc_w[w], c_norm_w[w], rep_mode, croiss_ws=croiss_ws_w[w]
                    )
                    _, lw_p = _launch_apex_assign(
                        x_src=x_assign_w[w],
                        c_rep=c_rep_w[w],
                        c_norm=c_norm_w[w],
                        rep_mode=rep_mode,
                        d=d,
                        sx_buf=sx_shard_w[w],
                        sc_buf=sc_w[w],
                        x_norm_buf=xn_shard_w[w],
                        origin=origin64_w[w] if (sc_has_orig_w[w] and rep_mode != 3) else None,
                        coord_scale=coord_scale if (sc_has_orig_w[w] and rep_mode != 3) else 1.0,
                        use_sm120_kernel=self.use_sm120_kernel,
                        croiss_ds=croiss_ds_w[w],
                        croiss_ws=croiss_ws_w[w],
                    )
                    if perm0_w[w] is not None:
                        lbl_parts[w].scatter_(0, perm0_w[w], lw_p)
                    else:
                        lbl_parts[w].copy_(lw_p)
            for w, dv in enumerate(self.devices):
                streams[w].synchronize()
                torch.cuda.current_stream(dv).wait_stream(streams[w])
            assignments_out = torch.empty((n,), dtype=torch.int64, device=dev0)
            if input_is_list:
                # Caller passed explicit shard list `[shard_0, shard_1, ...]`: concatenate in shard order
                off_a = 0
                for lp in lbl_parts:
                    m_l = int(lp.shape[0])
                    assignments_out[off_a : off_a + m_l] = lp.to(dev0)
                    off_a += m_l
            else:
                # Caller passed single `x` sharded strided via `prepopulate_shards` (`x[w::W]`): interleave back
                for w, lp in enumerate(lbl_parts):
                    assignments_out[w::W] = lp.to(dev0)

        for dv in self.devices:
            torch.cuda.synchronize(dv)
        self.train_time_sec = time.perf_counter() - t_train_start
        self.total_time_sec = self.init_time_sec + self.train_time_sec
        self.assignments = assignments_out

        ep_sq_list = ep_sq_w[0].cpu().tolist()
        ep_dots = ep_dot_gpu0.cpu().tolist()
        ep_cum_splits = [int(v) for v in ep_sp_gpu0.cpu().tolist()]
        ep_cum_prot = [int(v) for v in ep_pr_gpu0.cpu().tolist()]
        self.dead_clusters = int(dead_t0.item())
        self.n_splits = ep_cum_splits[-1] if ep_cum_splits else 0
        self.n_protected = ep_cum_prot[-1] if ep_cum_prot else 0
        inv_scale_sq = 1.0 / (coord_scale * coord_scale)
        prev_sp = 0
        prev_pr = 0
        for ep_idx, rows_in_ep in enumerate(recipe.epoch_rows):
            cur_sp = ep_cum_splits[ep_idx]
            cur_pr = ep_cum_prot[ep_idx]
            imb = float(k * ep_dots[ep_idx] / (float(max(1, rows_in_ep)) ** 2))
            self.iteration_stats.append(
                ApexEMeansEpochStat(
                    epoch=ep_idx + 1,
                    rows=rows_in_ep,
                    steps=int(recipe.epoch_steps[ep_idx]),
                    moving_mse=(float(ep_sq_list[ep_idx]) * inv_scale_sq) / max(1, rows_in_ep),
                    imbalance_factor=imb,
                    n_splits=cur_sp - prev_sp,
                    n_protected=cur_pr - prev_pr,
                    n_transplants=0,
                    elapsed_sec=self.train_time_sec * float(ep_idx + 1) / float(max(1, num_epochs)),
                )
            )
            prev_sp = cur_sp
            prev_pr = cur_pr

        if input_is_numpy:
            return (
                self.centroids.cpu().numpy(),
                self.assignments.cpu().numpy() if self.assignments is not None else None,
            )
        return self.centroids, self.assignments

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
        use_triton_assign_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Exact centered PyTorch fallback for CPU, `backend='torch'`, `compute_dtype=float64`, and unfused ladder arms."""
        from .ultra_emeans import _launch_syncfree_huygens_protection

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

        has_pre_close = self.finalization == "simplex_eb"
        pre_close_centroids = centroids_c.clone() if has_pre_close else None
        pre_close_captured = False

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
        eff_head_edit = self._resolve_head_edit(n)
        if eff_head_edit:
            last_edit_step = torch.full((k,), -(10**9), dtype=torch.int64, device=device)
            edit_gen = torch.Generator(device=device)
            edit_gen.manual_seed((self.seed + 918273) % (2**32))
            edit_diag: HeadEditDiagnostics | None = HeadEditDiagnostics()
        else:
            last_edit_step = None
            edit_gen = None
            edit_diag = None
        self.head_edit_diagnostics = edit_diag

        num_epochs = len(recipe.epoch_rows)
        eff_grace = (
            min(self.grace_steps, max(0, recipe.total_steps // 3))
            if (self.use_prior_mass and has_priors and track_surv)
            else 0
        )
        eff_wide_mult = self._resolve_wide_mult(recipe)
        eff_pass_span = self._effective_pass_span(n)
        wide_alpha_factor = float(eff_wide_mult ** ALPHA_LAW[2]) if eff_wide_mult > 1 else 1.0
        in_wide_phase = False
        total_planned_rows = max(1, sum(recipe.epoch_rows))
        rows_processed = 0
        step = 0
        alpha_last = recipe.alpha0 * recipe.alpha_min_factor
        sorted_cps = sorted(cp for cp in self.transplant_checkpoints if 0.05 <= cp <= 0.82)
        next_cp_idx = 0

        self.n_splits = 0
        self.n_protected = 0
        self.n_transplants = 0
        self.iteration_stats = []

        for ep_idx, rows_in_ep in enumerate(recipe.epoch_rows):
            is_last_epoch = ep_idx == (num_epochs - 1)
            if is_last_epoch and num_epochs >= 2 and pre_close_centroids is not None and not pre_close_captured:
                pre_close_centroids.copy_(centroids_c)
                pre_close_captured = True

            gen_ep = torch.Generator(device=device)
            gen_ep.manual_seed(epoch_seed(self.seed, ep_idx))
            order = (
                block_permute_indices(n, rows_in_ep, self.block_permute_size, gen_ep, device)
                if self.streaming_mode == "cyclic_block"
                else torch.randperm(n, generator=gen_ep, device=device)[:rows_in_ep]
            )
            raw_pass_counts.zero_()
            ep_sq = 0.0
            sp_before = self.n_splits
            pr_before = self.n_protected
            tr_before = self.n_transplants
            ep_steps = 0
            b_start = 0

            while b_start < rows_in_ep:
                if num_epochs == 1 and pre_close_centroids is not None and (not pre_close_captured) and b_start >= int(0.55 * rows_in_ep):
                    pre_close_centroids.copy_(centroids_c)
                    pre_close_captured = True

                prog = float(rows_processed) / float(total_planned_rows)
                use_wide_now = eff_wide_mult > 1 and prog >= self.wide_from_frac
                cur_bsz_target = min(n, recipe.batch_size * eff_wide_mult) if use_wide_now else recipe.batch_size
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
                elif use_triton_assign_only:
                    sq_dists, labels = assign_points(xb, centroids_c, compute_dtype=comp_dtype, backend="triton")
                else:
                    sq_dists, labels = assign_points(xb, centroids_c, compute_dtype=comp_dtype, backend="torch")

                ep_sq += float(sq_dists.double().sum().item())
                counts = torch.bincount(labels, minlength=k).float()
                bsums = torch.zeros((k, d), dtype=torch.float32, device=device)
                bsums.index_add_(0, labels, xb_f)
                bsse = torch.zeros((k,), dtype=torch.float32, device=device)
                bsse.index_add_(0, labels, sq_dists.float())
                raw_pass_counts.add_(counts.double())

                if is_last_epoch and need_pass_accum:
                    if eff_pass_span > 0.0:
                        pfrac = min(1.0, (float(b_start) + 0.5 * float(b_cur)) / float(max(1, rows_in_ep)))
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
                    if eff_wide_mult > 1
                    else recipe.alpha0
                )
                sched_frac = prog if eff_wide_mult > 1 else min(step, recipe.total_steps) / max(1, recipe.total_steps)
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
                    if (self.use_revive_stop and eff_wide_mult > 1)
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
                            eta_eff = torch.full_like(sig_rms, self.split_mult)
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
                            if prin_vec is not None and self.split_mode not in ("rms_aniso", "rms_iso"):
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
                        if pre_close_centroids is not None and pre_close_captured:
                            pre_close_centroids[src_idx] = c_src
                            pre_close_centroids[tgt_idx] = c_tgt
                        if prin_vec is not None:
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
                    eff_head_edit
                    and edit_gen is not None
                    and last_edit_step is not None
                    and edit_diag is not None
                    and next_cp_idx < len(sorted_cps)
                    and prog_after >= sorted_cps[next_cp_idx]
                    and ((not is_last_epoch and prog_after <= 0.85) or (num_epochs == 1 and prog_after <= 0.65))
                    and b_cur >= 32
                    and k >= 3
                ):
                    while next_cp_idx < len(sorted_cps) and prog_after >= sorted_cps[next_cp_idx]:
                        next_cp_idx += 1
                    n_ed = run_apex_head_edit_checkpoint(
                        x_batch_c=xb_f,
                        centroids_c=centroids_c,
                        usage=usage,
                        sse_ema=sse_ema,
                        ref_distortion_mean=ref_dist_mean,
                        survival_mass=survival_mass,
                        prin_vec=prin_vec,
                        slow_centroids=slow_centroids,
                        slow_usage=slow_usage,
                        pre_close_centroids=pre_close_centroids if pre_close_captured else None,
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

                rows_processed += b_cur
                b_start = b_end
                step += 1
                ep_steps += 1

            imb = float((k * torch.dot(raw_pass_counts, raw_pass_counts) / (float(max(1, rows_in_ep)) ** 2)).item())
            self.iteration_stats.append(
                ApexEMeansEpochStat(
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
        centroids_fin, s_diag = apex_simplex_eb_finalize(
            pass_sums=pass_sums,
            pass_counts=pass_counts,
            pass_counts_sq=pass_counts_sq if eff_pass_span > 0.0 else None,
            pass_sse=pass_sse,
            live_centroids=c_ema_last,
            pre_close_centroids=pre_close_centroids if pre_close_captured else None,
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
        self.pre_close_centroids = pre_close_centroids if pre_close_captured else None
        self.dead_clusters = int((raw_pass_counts <= 0.5).sum().item())

        assignments: torch.Tensor | None = None
        if capture_assignments:
            if comp_dtype == torch.float64:
                c64 = centroids_fin.double().contiguous()
                _, assignments = torch_assign_chunked(x_c, c64, c64.square().sum(dim=1))
            else:
                _, assignments = assign_points(
                    x_c,
                    centroids_fin,
                    compute_dtype=comp_dtype,
                    backend="triton" if use_triton_assign_only else "torch",
                )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self.train_time_sec = time.perf_counter() - t_train_start
        return centroids_fin, assignments

    @torch.no_grad()
    def assign(
        self,
        x: torch.Tensor | np.ndarray | list[torch.Tensor],
    ) -> tuple[torch.Tensor | np.ndarray | list[torch.Tensor], torch.Tensor | np.ndarray | list[torch.Tensor]]:
        """Assign points `x` (or per-GPU shard list) to learned centroids, returning `(sq_dists, labels)`."""
        if self.centroids is None:
            raise RuntimeError("ApexEMeans must be fitted before calling assign().")

        if isinstance(x, list):
            dists_list: list[torch.Tensor] = []
            lbls_list: list[torch.Tensor] = []
            for s in x:
                d_s, l_s = self._assign_single_tensor(s.detach())
                dists_list.append(d_s)
                lbls_list.append(l_s)
            return dists_list, lbls_list

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

        sq_dists, labels = self._assign_single_tensor(x_t)
        if input_is_numpy:
            return sq_dists.cpu().numpy(), labels.cpu().numpy()
        return sq_dists, labels

    def _assign_single_tensor(self, x_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.centroids is not None
        c_dev = x_t.device
        n = int(x_t.shape[0])
        if n == 0:
            return (
                torch.empty((0,), dtype=torch.float32, device=c_dev),
                torch.empty((0,), dtype=torch.int64, device=c_dev),
            )

        assign_ctx = torch.cuda.device(c_dev) if c_dev.type == "cuda" else contextlib.nullcontext()
        with assign_ctx:
            c_on_dev = self.centroids.to(c_dev)
            origin64 = (
                self._origin64.to(c_dev)
                if self._origin64 is not None
                else x_t[: min(n, 8192)].double().mean(dim=0).contiguous()
            )
            coord_scale = self._coord_scale
            c_c64 = (c_on_dev.double() - origin64[None, :]) * coord_scale

            # Use IEEE FP32 or BF16 for exact inference `assign()` unless `compute_dtype=float64`
            if self.compute_dtype == torch.float64 or self.precision == "fp64":
                x_c64 = (x_t.double() - origin64[None, :]) * coord_scale
                c64 = c_c64.contiguous()
                sq_dists, labels = torch_assign_chunked(x_c64.contiguous(), c64, c64.square().sum(dim=1))
            elif (
                c_dev.type == "cuda"
                and HAS_TRITON
                and (self.backend == "triton" or (self.backend == "auto" and torch.cuda.get_device_capability(c_dev) >= (8, 0)))
            ):
                # Default to exact IEEE FP32 when d <= 64 or precision == "fp32", else BF16/FP16
                if self.precision == "fp16" or self.compute_dtype == torch.float16 or x_t.dtype == torch.float16:
                    a_mode = 2
                    a_dt = torch.float16
                elif (
                    self.precision == "fp32"
                    or self.compute_dtype == torch.float32
                    or self.d <= 64
                    or self._high_cr_fp32_assign
                ):
                    a_mode = 0
                    a_dt = torch.float32
                else:
                    a_mode = 1
                    a_dt = torch.bfloat16
                c_c32 = c_c64.float().contiguous()
                c_rep = _alloc_padded_2d(self.k, self.d, a_dt, c_dev, align=16)
                sc_dummy = torch.ones((self.k,), dtype=torch.float32, device=c_dev)
                c_norm = torch.empty((self.k,), dtype=torch.float32, device=c_dev)
                _launch_apex_prep_centroids(c_c32, c_rep, sc_dummy, c_norm, a_mode)
                orig_ker = (
                    origin64
                    if (x_t.dtype == torch.float64 or coord_scale != 1.0 or float(origin64.abs().max().item()) > 1e3)
                    else origin64.float().contiguous()
                )
                sq_dists, labels = _launch_apex_assign(
                    x_src=x_t,
                    c_rep=c_rep,
                    c_norm=c_norm,
                    rep_mode=a_mode,
                    d=self.d,
                    origin=orig_ker,
                    coord_scale=coord_scale,
                    use_sm120_kernel=self.use_sm120_kernel,
                )
            else:
                x_c64 = (x_t.double() - origin64[None, :]) * coord_scale
                a_dt = torch.float16 if (self.precision == "fp16" or self.compute_dtype == torch.float16) else torch.float32
                sq_dists, labels = assign_points(
                    x_c64.float().contiguous(),
                    c_c64.float().contiguous(),
                    compute_dtype=a_dt,
                    backend="torch" if self.backend == "torch" else "auto",
                )

            if coord_scale != 1.0:
                inv_sq = 1.0 / (coord_scale * coord_scale)
                sq_dists = (sq_dists.double() * inv_sq).clamp_max(torch.finfo(torch.float32).max).float()
            return sq_dists, labels


# ==============================================================================
# 7. Pre-Registered Dominance & Optimization Speedup Ladder Arms + JIT Pre-Warmer
# ==============================================================================

APEX_LADDER_ARMS: dict[str, dict[str, Any]] = {
    "ladder_0_baseline": {
        "precision": "bf16",
        "use_wide_batch": False,
        "streaming_mode": "fused_gather",
        "preencode_dataset": False,
        "use_fused_pipeline": False,
        "use_sm120_kernel": False,
        "use_cuda_graph": False,
    },
    "ladder_1_plus_batch": {
        "precision": "bf16",
        "use_wide_batch": True,
        "streaming_mode": "cyclic_block",
        "preencode_dataset": True,
        "use_fused_pipeline": False,
        "use_sm120_kernel": False,
        "use_cuda_graph": False,
    },
    "ladder_2_plus_fuse": {
        "precision": "bf16",
        "use_wide_batch": True,
        "streaming_mode": "cyclic_block",
        "preencode_dataset": True,
        "use_fused_pipeline": True,
        "use_sm120_kernel": False,
        "use_cuda_graph": False,
    },
    "ladder_3_plus_kernel": {
        "precision": "bf16",
        "use_wide_batch": True,
        "streaming_mode": "cyclic_block",
        "preencode_dataset": True,
        "use_fused_pipeline": True,
        "use_sm120_kernel": True,
        "use_cuda_graph": False,
    },
    "ladder_4_plus_cudagraph": {
        "precision": "bf16",
        "use_wide_batch": True,
        "streaming_mode": "cyclic_block",
        "preencode_dataset": True,
        "use_fused_pipeline": True,
        "use_sm120_kernel": True,
        "use_cuda_graph": True,
    },
    "ladder_5_plus_croiss_fp8": {
        "precision": "croiss_fp8",
        "use_wide_batch": True,
        "streaming_mode": "cyclic_block",
        "preencode_dataset": True,
        "use_fused_pipeline": True,
        "use_sm120_kernel": True,
        "use_cuda_graph": True,
    },
}

APEX_ABLATION_ARMS: dict[str, dict[str, Any]] = {
    "apex_emeans": {
        "gear_mode": "auto",
        "precision": "auto",
        "use_cuda_graph": True,
    },
    "apex_emeans_bf16": {
        "gear_mode": "auto",
        "precision": "bf16",
        "use_cuda_graph": True,
    },
    "apex_emeans_fp8": {
        "gear_mode": "auto",
        "precision": "croiss_fp8",
        "use_cuda_graph": True,
    },
    "apex_emeans_eager": {
        "gear_mode": "auto",
        "precision": "auto",
        "use_cuda_graph": False,
    },
    "apex_emeans_ivf": {
        "gear_mode": "huge_k_ivf",
        "precision": "auto",
        "ivf_explore": True,
        "use_cuda_graph": True,
    },
    **APEX_LADDER_ARMS,
}


def build_apex_arm(
    arm_name: str,
    d: int,
    k: int,
    nepochs: float = 5.0,
    seed: int = 1234,
    **overrides: Any,
) -> ApexEMeans:
    """Instantiate any pre-registered `ApexEMeans` dominance or speedup-ladder arm by key."""
    if arm_name not in APEX_ABLATION_ARMS:
        valid = ", ".join(sorted(APEX_ABLATION_ARMS.keys()))
        raise KeyError(f"Unknown arm_name={arm_name!r}; expected one of: {valid}")
    cfg = {**APEX_ABLATION_ARMS[arm_name], **overrides}
    return ApexEMeans(d=d, k=k, nepochs=nepochs, seed=seed, **cfg)


@torch.no_grad()
def warmup_apex_kernels(
    device: torch.device | str = "cuda",
    d: int = 128,
    k: int = 2048,
) -> None:
    """Pre-compile all `ApexEMeans` Triton & CroissFP8 specializations outside benchmark timers (`#111`)."""
    dev = torch.device(device)
    if dev.type != "cuda" or not HAS_TRITON or not torch.cuda.is_available():
        return

    with torch.cuda.device(dev):
        # 1. Pre-warm all global & sharded tournament k-means++ `BLOCK_P` buckets (`256`, `512`, `1024`, batched `128`)
        pair_d2 = torch.zeros((1024, 1024), device=dev, dtype=torch.float32)
        pool_d2 = torch.ones((1024,), device=dev, dtype=torch.float32)
        w_p = torch.full((1024,), 1.0 / 1024.0, device=dev, dtype=torch.float32)
        out_idx = torch.empty((64,), device=dev, dtype=torch.int64)
        for blk_p, n_w in ((256, 4), (512, 4), (1024, 8)):
            _apex_local_greedy_kpp_kernel[(1,)](
                pair_d2, pool_d2, w_p, out_idx, blk_p - 1, 16, 17, BLOCK_P=blk_p, L_TRIALS=3, num_warps=n_w
            )
        pair_b = torch.zeros((4 * 128, 128), device=dev, dtype=torch.float32)
        _apex_batched_local_greedy_kpp_kernel[(4,)](
            pair_b, pool_d2[:512], w_p[:512], out_idx, 128, 120, 16, 120, 16, 4, 17, BLOCK_P=128, L_TRIALS=3, num_warps=4
        )

        cmod = _get_croiss_fp8_module()
        if cmod is not None and hasattr(cmod, "warmup_croiss_fp8_kernels"):
            cmod.warmup_croiss_fp8_kernels(
                dev,
                shapes=[
                    (2048, max(512, min(k, 4096)), d),
                    (4096, max(512, min(k, 4096)), d),
                    (16384, max(512, min(k, 4096)), d),
                ],
            )

        w_k = min(max(k, 64), 2048)
        w_n = max(w_k * 4, 16384)
        x_w = torch.randn((w_n, d), dtype=torch.float32, device=dev)
        # Force at least one dead cluster on synthetic data so revival & Huygens split kernels always JIT-compile!
        c_dead = x_w[:w_k].clone().contiguous()
        if w_k >= 4:
            c_dead[w_k - 2 :] = 1e4

        for prec in ("bf16", "croiss_fp8", "fp32"):
            if prec == "croiss_fp8" and not _has_fp8_tensor_cores(dev):
                continue
            for ep_w in (1.0, 2.0):
                s_w = ApexEMeans(
                    d=d,
                    k=w_k,
                    nepochs=ep_w,
                    precision=prec,
                    use_wide_batch=(ep_w > 1.0),
                    use_cuda_graph=True,
                    seed=7,
                )
                s_w.fit(x_w, capture_assignments=(ep_w > 1.0))
            s_dead = ApexEMeans(d=d, k=w_k, nepochs=1.0, precision=prec, use_cuda_graph=True, seed=7)
            s_dead.fit(x_w[: max(4096, w_k * 2)], init_centroids=c_dead, capture_assignments=False)
            s_dead.assign(x_w[:1024])

        # Warm speedup-ladder (`use_sm120_kernel=False`) & Gear-3 IVF explorer so neither ever cold-JITs
        for l_arm in ("ladder_0_baseline", "ladder_2_plus_fuse"):
            s_l = build_apex_arm(l_arm, d=d, k=min(w_k, 512), nepochs=2.0, seed=7)
            s_l.fit(x_w[: max(4096, min(w_k, 512) * 4)], capture_assignments=True)

        if w_k >= 256:
            s_ivf = build_apex_arm("apex_emeans_ivf", d=d, k=min(w_k, 512), nepochs=2.0, seed=7)
            s_ivf.fit(x_w[: max(4096, min(w_k, 512) * 4)], capture_assignments=False)

        # Also warm High-`CR` Ozaki-FP8 + BF16 closing guard when `d <= 64` (e.g. `covtype_multiscale_54`)
        if d <= 64 and _has_fp8_tensor_cores(dev):
            x_cr = x_w[: max(4096, min(w_k, 512) * 4)].clone()
            x_cr[:, 0] *= 2500.0
            for prec_cr in ("croiss_fp8", "bf16"):
                s_cr = ApexEMeans(d=d, k=min(w_k, 512), nepochs=2.0, precision=prec_cr, use_cuda_graph=True, seed=7)
                s_cr.fit(x_cr, capture_assignments=False)

        torch.cuda.synchronize(dev)


__all__ = [
    "APEX_ABLATION_ARMS",
    "APEX_LADDER_ARMS",
    "ApexEMeans",
    "ApexEMeansEpochStat",
    "apex_simplex_eb_finalize",
    "apex_turkpp_init_centered",
    "build_apex_arm",
    "run_apex_head_edit_checkpoint",
    "warmup_apex_kernels",
]
