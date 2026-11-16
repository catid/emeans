"""Faithful reproduction, modular component decomposition, and zero-sync GPU optimization of s-means.

Dissects and reproduces:
  Kocabay, Yuksel, and Yalcin, "s-means: outperforming e-means (which outperforms k-means)" (2026),
  https://github.com/suayptalha/s-means

Every distinct algorithmic component of `s-means` is exposed as an independent toggle on `SMeans`
(and pre-packaged in `SMEANS_ABLATION_CONFIGS`) so leave-one-out and hybrid cross-pollination
ablations against `EMeans`, `EMeansPP`, and `SuperEMeans` are 100% de-confounded:
  1. `init_mode` ("subsampled_d15" vs "random"): Multi-round uniform-subsampled (D^2)^1.5 = D^3 seeding.
  2. `schedule_mode` ("power_cosine" vs "emeans_cosine" vs "constant"): 0.06 + 0.94 * cos^0.92 decay
     and `lowd_revival=True` (revival enabled for all d >= 1).
  3. `use_revive_stop` (`revive_stop_frac=0.85`): Freezes dead-cluster revival at 85% of total steps.
  4. `donor_mode` ("sse_usage" vs "sse_only" vs "usage_only"): Fixed-decay `sse_ema * usage^0.35`
     Gumbel-top-k donor sampling (plus optional `couple_sse_to_alpha` horizon matching).
  5. `split_mode` ("rms_aniso" vs "rms_iso" vs "checkerboard") & `use_feat_scale`: Additive local-RMS
     distortion radius `0.55 * sqrt(sse_ema / (usage * d))` times online O(K*d) inter-cluster feature
     standard deviation `feat_scale` with Rademacher signs.
  6. `use_pass_weighting` (`pass_w_span=0.35`): Linear final-pass batch ramp weight in [0.65, 1.35].
  7. `finalization` ("eb_ema" vs "closing_mean" vs "ema_only"): Count-adaptive Empirical-Bayes
     shrinkage `n_k / (n_k + eb_tau)` toward the tracked live EMA centroid (`eb_tau=0.20`).

Systems & Numerical Optimizations (`exact_ref=False`, default):
  - Zero host-device `.item()` / `.tolist()` sync stalls across the entire training loop (both per-step
    and on dead-cluster revival checks via device-side `argsort` + masked `n_sp_ptr` Triton split kernel).
  - Zero gathered mini-batch copies (`row_indices` gathered directly in registers).
  - Translation-invariant coordinate centering (`origin`) fused into assignment and scatter kernels
    before low-precision (`bf16`/`fp16`) casts, preserving Appendix I.1 represented-norm consistency
    while eliminating catastrophic cancellation and overflow at `+10^4`..`+10^6` offsets.
  - Fused Triton scatter-stats+SSE and fused EMA+SSE+pass+centroid-prep kernels.
  - Set `exact_ref=True` to reproduce upstream `suayptalha/s-means` RNG streams and uncentered ops
    for direct reference-repository parity verification.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .emeans import EpochStat, _separate_collapsed_splits
from .kernels import assign_points, prepare_represented_centroids, torch_assign_chunked
from .recipe import (
    ALPHA_MIN_FACTOR,
    CHECK_UNUSED_EVERY,
    DEFAULT_INIT_USAGE,
    DEFAULT_UNUSED_THRESHOLD,
    NO_REVIVAL_MAX_D,
    SPLIT_EPSILON,
    EMeansRecipe,
    compute_epoch_rows,
    compute_step_law,
    epoch_seed,
    resolve_batch_law,
    scheduled_alpha,
)

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    HAS_TRITON = False


# Default s-means hyperparameters from suayptalha/s-means (`smeans/_recipe.py`)
SMEANS_ALPHA_FLOOR: float = 0.06
SMEANS_COSINE_POWER: float = 0.92
SMEANS_REVIVE_STOP_FRAC: float = 0.85
SMEANS_SPLIT_MULT: float = 0.55
SMEANS_DONOR_USAGE_W: float = 0.35
SMEANS_EB_TAU: float = 0.20
SMEANS_PASS_W_SPAN: float = 0.35
SMEANS_SSE_DECAY: float = 0.85
SMEANS_VAR_BETA: float = 0.85
SMEANS_INIT_ROUNDS: int = 4
SMEANS_INIT_POWER: float = 1.5


def round_pow2_smeans(val: float) -> int:
    """Upstream s-means power-of-two rounding (`2 ** round(log2(val))`)."""
    val = max(1.0, float(val))
    return max(1, int(2 ** round(math.log2(val))))


def smeans_scheduled_alpha(
    alpha0: float,
    step: int,
    total_steps: int,
    schedule_mode: str = "power_cosine",
    alpha_floor: float = SMEANS_ALPHA_FLOOR,
    cosine_power: float = SMEANS_COSINE_POWER,
) -> float:
    """Compute step size alpha(t) under s-means power-cosine, e-means cosine, or constant schedule."""
    if schedule_mode == "constant":
        return float(alpha0)
    if schedule_mode == "emeans_cosine":
        return scheduled_alpha(alpha0, step, total_steps, ALPHA_MIN_FACTOR)
    if schedule_mode != "power_cosine":
        raise ValueError(
            f"Unknown schedule_mode={schedule_mode!r}; expected 'power_cosine', 'emeans_cosine', or 'constant'"
        )
    frac = min(step, total_steps) / max(1, total_steps)
    cos_half = 0.5 * (1.0 + math.cos(math.pi * frac))
    return float(alpha0 * (alpha_floor + (1.0 - alpha_floor) * (cos_half**cosine_power)))


def smeans_progressive_pass_weight(
    step_in_epoch: int,
    steps_in_epoch: int,
    pass_w_span: float = SMEANS_PASS_W_SPAN,
) -> float:
    """Compute linear final-pass batch ramp weight in [1 - span, 1 + span]."""
    if pass_w_span == 0.0:
        return 1.0
    pfrac = (step_in_epoch + 0.5) / max(1, steps_in_epoch)
    return float((1.0 - pass_w_span) + 2.0 * pass_w_span * pfrac)


def smeans_eb_finalize(
    centroids: torch.Tensor,
    pass_sums: torch.Tensor,
    pass_counts: torch.Tensor,
    eb_tau: float = SMEANS_EB_TAU,
) -> torch.Tensor:
    """Shrink weighted final-pass cluster means toward tracked EMA centroids."""
    pc_f = pass_counts.float()
    has_mass = pc_f > 0.0
    denom = (pc_f + float(eb_tau)).clamp_min(1e-12)
    w_fin = torch.where(has_mass, pc_f / denom, torch.zeros_like(pc_f))
    pmean = pass_sums.float() / pc_f.clamp_min(1.0)[:, None]
    blended = w_fin[:, None] * pmean + (1.0 - w_fin[:, None]) * centroids.float()
    return torch.where(has_mass[:, None], blended, centroids.float())


def _make_ref_generator(device: torch.device, seed: int) -> torch.Generator:
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed) % (2**31 - 1))
    return gen


@torch.no_grad()
def subsampled_d15_init(
    x: torch.Tensor,
    k: int,
    seed: int = 1234,
    rounds: int = SMEANS_INIT_ROUNDS,
    power: float = SMEANS_INIT_POWER,
    subsample_mult: int = 4,
    min_subsample: int = 512,
    exact_ref: bool = False,
    chunk_cols: int = 2048,
) -> torch.Tensor:
    """Multi-round uniform-subsampled distance-weighted seeding (`smeans/_recipe.py`).

    Note: Upstream raises squared Euclidean distance `dmin` to `power=1.5`, so sampling
    probabilities scale as `(||x - c||_2^2)^1.5 = ||x - c||_2^3` (`D^3`).
    When `exact_ref=False` (default), centers coordinates before norm expansion (eliminating
    FP32 catastrophic cancellation under large coordinate shifts), normalizes `dmin` before
    exponentiation (preventing FP32 overflow to `inf` at large scales), and chunks candidate
    products to bound temporary memory on large `k`.
    """
    n = int(x.shape[0])
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if n < k:
        raise ValueError(f"n_samples ({n}) must be >= n_clusters ({k})")
    if n == k:
        return x.detach().float().clone().contiguous()

    device = x.device
    if exact_ref:
        rng = _make_ref_generator(device, seed)
    else:
        rng = torch.Generator(device=device)
        rng.manual_seed(int(seed) % (2**32))

    m = min(n, max(int(k) * int(subsample_mult), int(min_subsample), int(k)))
    perm = torch.randperm(n, generator=rng, device=device)[:m]
    xs_raw = x[perm]

    nr = max(2, int(rounds))
    k0 = min(k, max(8, k // (nr * 2)))
    step_k = max(1, (k - k0 + nr - 2) // max(1, nr - 1))

    if exact_ref:
        xs = xs_raw.float().contiguous()
        xsn = (xs * xs).sum(dim=1)
        centroids = xs[:k0]
        c0n = (centroids * centroids).sum(dim=1)
        dmin = (xsn[:, None] + c0n[None, :] - 2.0 * (xs @ centroids.t())).clamp_min_(0.0).min(dim=1).values

        for _ in range(nr - 1):
            cur_k = int(centroids.shape[0])
            if cur_k >= k:
                break
            cand_n = min(step_k, k - cur_k)
            probs = (dmin.clamp_min(1e-12) ** power) + 1e-12
            picked = torch.multinomial(probs, cand_n, replacement=False, generator=rng)
            new_c = xs[picked]
            centroids = torch.cat([centroids, new_c], dim=0)
            new_n = (new_c * new_c).sum(dim=1)
            new_d = (xsn[:, None] + new_n[None, :] - 2.0 * (xs @ new_c.t())).clamp_min_(0.0).min(dim=1).values
            dmin = torch.minimum(dmin, new_d)
        return centroids[:k].contiguous()

    # Shift-safe, scale-safe, memory-bounded path
    origin64 = xs_raw.double().mean(dim=0, keepdim=True)
    xs_c64 = xs_raw.double() - origin64
    xs_scale = xs_c64.abs().max().clamp_min(1.0)
    xs_c = (xs_c64 / xs_scale).float().contiguous()
    xsn = xs_c.square().sum(dim=1)

    def _min_sq_dists(cands_c: torch.Tensor) -> torch.Tensor:
        cn = cands_c.square().sum(dim=1)
        nc = int(cands_c.shape[0])
        if nc <= chunk_cols:
            scores = torch.addmm(cn[None, :], xs_c, cands_c.t(), alpha=-2.0)
            return (scores.min(dim=1).values + xsn).clamp_min_(0.0)
        best = torch.full((m,), float("inf"), dtype=torch.float32, device=device)
        for j0 in range(0, nc, chunk_cols):
            j1 = min(j0 + chunk_cols, nc)
            scores = torch.addmm(cn[j0:j1][None, :], xs_c, cands_c[j0:j1].t(), alpha=-2.0)
            best = torch.minimum(best, scores.min(dim=1).values)
        return (best + xsn).clamp_min_(0.0)

    picked_indices = [torch.arange(k0, device=device, dtype=torch.int64)]
    dmin = _min_sq_dists(xs_c[:k0])
    chosen_mask = torch.zeros((m,), dtype=torch.bool, device=device)
    chosen_mask[:k0] = True

    for _ in range(nr - 1):
        cur_k = sum(int(t.numel()) for t in picked_indices)
        if cur_k >= k:
            break
        cand_n = min(step_k, k - cur_k)
        # Scale-normalize dmin into [0, 1] before exponentiating so dmin**power never overflows FP32
        d_scale = dmin.max().clamp_min(1e-30)
        if power == 0.0:
            probs = torch.ones_like(dmin)
        else:
            probs = ((dmin / d_scale).clamp_min(0.0) ** power) + 1e-12
        # Zero out already-picked coreset rows when enough distinct unpicked points remain
        probs = torch.where(chosen_mask, torch.full_like(probs, 1e-30), probs)
        picked = torch.multinomial(probs, cand_n, replacement=False, generator=rng)
        chosen_mask[picked] = True
        picked_indices.append(picked)
        new_d = _min_sq_dists(xs_c[picked])
        dmin = torch.minimum(dmin, new_d)

    all_picked = torch.cat(picked_indices, dim=0)[:k]
    return xs_raw[all_picked].float().clone().contiguous()


# ==============================================================================
# Custom Fused & Zero-Sync Triton Kernels for SMeans
# ==============================================================================

if HAS_TRITON:

    @triton.jit(do_not_specialize=["B", "K", "D"])
    def _smeans_fast_assign_kernel(
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
        stride_cd,
        BLOCK_B: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
        FULL_D: tl.constexpr,
        DTYPE_MODE: tl.constexpr,  # 0=FP32, 1=BF16, 2=FP16
        HAS_INDICES: tl.constexpr,
        HAS_ORIGIN: tl.constexpr,
        ORIGIN_FP64: tl.constexpr,
    ):
        """Fused gather + origin-centering + represented-norm L2 Tensor-Core assignment."""
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
            x_raw = tl.load(
                x_ptr + src_rows[:, None] * stride_xb + cols[None, :] * stride_xd,
                mask=mask_b[:, None] & mask_d[None, :],
                other=0.0,
            )
            if HAS_ORIGIN:
                if ORIGIN_FP64:
                    orig64 = tl.load(origin_ptr + cols, mask=mask_d, other=0.0).to(tl.float64)
                    x_f = (x_raw.to(tl.float64) - orig64[None, :]).to(tl.float32)
                else:
                    orig32 = tl.load(origin_ptr + cols, mask=mask_d, other=0.0).to(tl.float32)
                    x_f = x_raw.to(tl.float32) - orig32[None, :]
                x_f = tl.where(mask_b[:, None] & mask_d[None, :], x_f, 0.0)
            else:
                x_f = x_raw.to(tl.float32)

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
                x_raw = tl.load(
                    x_ptr + src_rows[:, None] * stride_xb + cols[None, :] * stride_xd,
                    mask=mask_b[:, None] & mask_d[None, :],
                    other=0.0,
                )
                if HAS_ORIGIN:
                    if ORIGIN_FP64:
                        orig64 = tl.load(origin_ptr + cols, mask=mask_d, other=0.0).to(tl.float64)
                        x_f = (x_raw.to(tl.float64) - orig64[None, :]).to(tl.float32)
                    else:
                        orig32 = tl.load(origin_ptr + cols, mask=mask_d, other=0.0).to(tl.float32)
                        x_f = x_raw.to(tl.float32) - orig32[None, :]
                    x_f = tl.where(mask_b[:, None] & mask_d[None, :], x_f, 0.0)
                else:
                    x_f = x_raw.to(tl.float32)

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
                    c_ptr + offs_k[:, None] * stride_ck + cols[None, :] * stride_cd,
                    mask=mask_k[:, None] & mask_d[None, :],
                    other=0.0,
                )
                if DTYPE_MODE == 0:
                    acc = tl.dot(
                        x_full,
                        tl.trans(c_tile),
                        input_precision="ieee",
                        out_dtype=tl.float32,
                    )
                else:
                    acc = tl.dot(x_full, tl.trans(c_tile), out_dtype=tl.float32)
            else:
                acc = tl.zeros((BLOCK_B, BLOCK_K), dtype=tl.float32)
                for d0 in range(0, D, BLOCK_D):
                    cols = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
                    mask_d = cols < D
                    x_raw = tl.load(
                        x_ptr + src_rows[:, None] * stride_xb + cols[None, :] * stride_xd,
                        mask=mask_b[:, None] & mask_d[None, :],
                        other=0.0,
                    )
                    if HAS_ORIGIN:
                        if ORIGIN_FP64:
                            orig64 = tl.load(origin_ptr + cols, mask=mask_d, other=0.0).to(tl.float64)
                            x_f = (x_raw.to(tl.float64) - orig64[None, :]).to(tl.float32)
                        else:
                            orig32 = tl.load(origin_ptr + cols, mask=mask_d, other=0.0).to(tl.float32)
                            x_f = x_raw.to(tl.float32) - orig32[None, :]
                        x_f = tl.where(mask_b[:, None] & mask_d[None, :], x_f, 0.0)
                    else:
                        x_f = x_raw.to(tl.float32)

                    c_tile = tl.load(
                        c_ptr + offs_k[:, None] * stride_ck + cols[None, :] * stride_cd,
                        mask=mask_k[:, None] & mask_d[None, :],
                        other=0.0,
                    )
                    if DTYPE_MODE == 1:
                        acc = tl.dot(
                            x_f.to(tl.bfloat16),
                            tl.trans(c_tile),
                            acc=acc,
                            out_dtype=tl.float32,
                        )
                    elif DTYPE_MODE == 2:
                        acc = tl.dot(
                            x_f.to(tl.float16),
                            tl.trans(c_tile),
                            acc=acc,
                            out_dtype=tl.float32,
                        )
                    else:
                        acc = tl.dot(
                            x_f,
                            tl.trans(c_tile),
                            acc=acc,
                            input_precision="ieee",
                            out_dtype=tl.float32,
                        )

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

    @triton.jit(do_not_specialize=["B", "D", "ep_idx"])
    def _smeans_scatter_stats_sse_kernel(
        x_ptr,
        row_idx_ptr,
        origin_ptr,
        labels_ptr,
        dists_ptr,
        sums_ptr,
        counts_ptr,
        batch_sse_ptr,
        ep_sq_dists_ptr,
        ep_idx,
        B,
        D,
        stride_xb,
        stride_xd,
        stride_sumsk,
        HAS_INDICES: tl.constexpr,
        HAS_ORIGIN: tl.constexpr,
        ORIGIN_FP64: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Fused gather + centering + atomic scatter into sums, counts, batch_sse, and ep_sq_dist."""
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

        vals_raw = tl.load(
            x_ptr + src_rows[:, None] * stride_xb + offs_d[None, :] * stride_xd,
            mask=mask_bd,
            other=0.0,
        )
        if HAS_ORIGIN:
            if ORIGIN_FP64:
                orig64 = tl.load(origin_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float64)
                vals = (vals_raw.to(tl.float64) - orig64[None, :]).to(tl.float32)
            else:
                orig32 = tl.load(origin_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32)
                vals = vals_raw.to(tl.float32) - orig32[None, :]
        else:
            vals = vals_raw.to(tl.float32)

        tl.atomic_add(
            sums_ptr + lbl[:, None] * stride_sumsk + offs_d[None, :],
            vals,
            mask=mask_bd,
            sem="relaxed",
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

    @triton.jit(do_not_specialize=["K", "D"])
    def _smeans_feat_scale_kernel(
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
        """Online inter-cluster feature variance tracking in O(K * D)."""
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

    @triton.jit(do_not_specialize=["K", "D", "replace_usage", "accum_pass", "prep_rep"])
    def _smeans_ema_sse_pass_prep_kernel(
        sums_ptr,
        counts_ptr,
        batch_sse_ptr,
        centroids_ptr,
        usage_ptr,
        sse_ema_ptr,
        pass_sums_ptr,
        pass_counts_ptr,
        raw_pass_counts_ptr,
        c_rep_ptr,
        c_norm_ptr,
        K,
        D,
        stride_sd,
        stride_cd,
        stride_crepk,
        alpha,
        decay,
        sse_decay,
        w_pass,
        replace_usage,
        accum_pass,
        prep_rep,
        ZERO_STATS: tl.constexpr,
        REP_MODE: tl.constexpr,  # 0=FP32, 1=BF16, 2=FP16
        BLOCK_D: tl.constexpr,
    ):
        """Fused 1-CTA-per-cluster EMA update + SSE EMA + pass accumulation + stat zeroing + c_rep prep."""
        j = tl.program_id(0).to(tl.int64)
        if j >= K:
            return

        cnt = tl.load(counts_ptr + j).to(tl.float32)
        bsse = tl.load(batch_sse_ptr + j).to(tl.float32)
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
            pc_old = tl.load(pass_counts_ptr + j)
            tl.store(pass_counts_ptr + j, pc_old + (w_pass * cnt).to(tl.float64))

        if ZERO_STATS:
            tl.store(counts_ptr + j, 0.0)
            tl.store(batch_sse_ptr + j, 0.0)

        norm_acc = tl.zeros((), dtype=tl.float32)
        for d0 in range(0, D, BLOCK_D):
            cols = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
            mask_d = cols < D
            s_val = tl.load(sums_ptr + j * stride_sd + cols, mask=mask_d, other=0.0).to(tl.float32)
            if do_accum:
                ps_old = tl.load(pass_sums_ptr + j * stride_cd + cols, mask=mask_d, other=0.0).to(tl.float32)
                tl.store(pass_sums_ptr + j * stride_cd + cols, ps_old + w_pass * s_val, mask=mask_d)
            if ZERO_STATS:
                tl.debug_barrier()
                tl.store(sums_ptr + j * stride_sd + cols, 0.0, mask=mask_d)

            c_old = tl.load(centroids_ptr + j * stride_cd + cols, mask=mask_d, other=0.0).to(tl.float32)
            mean_val = s_val * inv_cnt
            pos_val = c_old + weight * (mean_val - c_old)
            upd_val = tl.where(is_one, mean_val, pos_val)
            c_new = tl.where(touched, upd_val, c_old)
            tl.store(centroids_ptr + j * stride_cd + cols, c_new, mask=mask_d & touched)

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
        new_sse = sse_decay * old_sse + (1.0 - sse_decay) * bsse
        tl.store(usage_ptr + j, new_u)
        tl.store(sse_ema_ptr + j, new_sse)

    @triton.jit(do_not_specialize=["D", "rng_seed", "split_mode"])
    def _smeans_syncfree_revive_split_kernel(
        centroids_ptr,
        origin_ptr,
        usage_ptr,
        sse_ema_ptr,
        fvec_ptr,
        src_ptr,
        tgt_ptr,
        n_sp_ptr,
        D,
        stride_cd,
        inv_d,
        split_mult,
        split_eps,
        rng_seed,
        split_mode,  # 0=rms_aniso/rms_iso, 1=checkerboard (runtime int avoids extra JIT compile)
        HAS_ORIGIN: tl.constexpr,
        REP_MODE: tl.constexpr,  # 0=FP32, 1=BF16, 2=FP16
        GUARD_COLLAPSE: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Zero-host-sync centroid split kernel reading dynamic `n_sp` directly from device memory."""
        pid = tl.program_id(0).to(tl.int64)
        n_sp = tl.load(n_sp_ptr).to(tl.int64)
        if pid >= n_sp:
            return

        src = tl.load(src_ptr + pid).to(tl.int64)
        tgt = tl.load(tgt_ptr + pid).to(tl.int64)

        u_s = tl.load(usage_ptr + src).to(tl.float32)
        sse_s = tl.minimum(tl.load(sse_ema_ptr + src).to(tl.float32), 1e35)
        rad = split_mult * tl.sqrt(tl.maximum((sse_s / tl.maximum(u_s, 1e-6)) * inv_d, 1e-10))
        half_u = 0.5 * u_s
        half_sse = 0.5 * sse_s

        is_rms = split_mode == 0
        all_equal = 1
        for d0 in range(0, D, BLOCK_D):
            col = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
            cmask = col < D
            dc = tl.load(centroids_ptr + src * stride_cd + col, mask=cmask, other=0.0).to(tl.float32)
            if is_rms:
                fv = tl.load(fvec_ptr + col, mask=cmask, other=1.0).to(tl.float32)
                u_rnd = tl.rand(rng_seed, pid * D + col)
                radem = tl.where(u_rnd < 0.5, -1.0, 1.0)
                delta = rad * radem * fv
                c_tgt = dc + delta
                c_src = dc - delta
            else:
                sign = tl.where((col & 1) == 0, 1.0, -1.0)
                if HAS_ORIGIN:
                    orig = tl.load(origin_ptr + col, mask=cmask, other=0.0).to(tl.float32)
                    abs_dc = dc + orig
                    c_tgt = abs_dc * (1.0 + split_eps * sign) - orig
                    c_src = abs_dc * (1.0 - split_eps * sign) - orig
                else:
                    c_tgt = dc * (1.0 + split_eps * sign)
                    c_src = dc * (1.0 - split_eps * sign)

            if GUARD_COLLAPSE:
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

        if GUARD_COLLAPSE and (all_equal != 0):
            # Inward signed nudge >= 1 represented-precision ULP prevents overflow near FP16/FP32 max
            c0 = tl.load(centroids_ptr + tgt * stride_cd).to(tl.float32)
            if REP_MODE == 1:
                step_ulp = tl.maximum(tl.abs(c0) * (1.0 / 64.0), 1e-30)
            elif REP_MODE == 2:
                step_ulp = tl.maximum(tl.abs(c0) * (1.0 / 512.0), 1e-4)
            else:
                step_ulp = tl.maximum(tl.abs(c0) * 2.5e-7, 1e-30)
            nudge = tl.where(c0 > 0.0, -step_ulp, step_ulp)
            tl.store(centroids_ptr + tgt * stride_cd, c0 + nudge)

        tl.store(usage_ptr + tgt, half_u)
        tl.store(usage_ptr + src, half_u)
        tl.store(sse_ema_ptr + tgt, half_sse)
        tl.store(sse_ema_ptr + src, half_sse)

    @triton.jit(do_not_specialize=["K", "D"])
    def _smeans_prep_centroids_kernel(
        centroids_ptr,
        c_rep_ptr,
        c_norm_ptr,
        K,
        D,
        stride_ck,
        stride_crepk,
        REP_MODE: tl.constexpr,  # 0=FP32, 1=BF16, 2=FP16
        BLOCK_D: tl.constexpr,
    ):
        """Convert centered centroids to represented dtype and compute ||c_rep||^2 in one launch."""
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


def _dtype_to_rep_mode(comp_dtype: torch.dtype) -> int:
    if comp_dtype == torch.bfloat16:
        return 1
    if comp_dtype == torch.float16:
        return 2
    return 0


def _alloc_c_rep_padded(k: int, d: int, comp_dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Allocate 16-column-aligned represented centroid buffer (zeros in padded columns)."""
    d_pad = max(16, ((d + 15) // 16) * 16)
    return torch.zeros((k, d_pad), dtype=comp_dtype, device=device)


def _launch_prep_centroids(
    centroids_c: torch.Tensor,
    c_rep: torch.Tensor,
    c_norm: torch.Tensor,
) -> None:
    k, d = centroids_c.shape
    rep_mode = _dtype_to_rep_mode(c_rep.dtype)
    block_d = min(256, max(16, triton.next_power_of_2(max(1, d))))
    with torch.cuda.device(centroids_c.device):
        _smeans_prep_centroids_kernel[(k,)](
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


def _launch_fast_assign(
    x_t: torch.Tensor,
    c_rep: torch.Tensor,
    c_norm: torch.Tensor,
    row_indices: torch.Tensor | None = None,
    origin: torch.Tensor | None = None,
    out_dist: torch.Tensor | None = None,
    out_label: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch fused gather + centered represented-norm Triton assignment for arbitrary (B, K, D >= 1)."""
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
    full_d = d <= 128
    if full_d:
        block_d = max(16, triton.next_power_of_2(max(1, d)))
        if b <= 512 and k >= 1024:
            block_b = 16
        elif low_smem and not reduced and block_d >= 128:
            block_b = 32
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

    has_idx = row_indices is not None
    idx_c = row_indices.contiguous() if has_idx else None
    has_orig = origin is not None
    orig_fp64 = bool(has_orig and origin.dtype == torch.float64)

    with torch.cuda.device(device):
        _smeans_fast_assign_kernel[(triton.cdiv(b, block_b),)](
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
            c_rep.stride(1),
            BLOCK_B=block_b,
            BLOCK_K=block_k,
            BLOCK_D=block_d,
            FULL_D=full_d,
            DTYPE_MODE=rep_mode,
            HAS_INDICES=has_idx,
            HAS_ORIGIN=has_orig,
            ORIGIN_FP64=orig_fp64,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return out_dist, out_label


# ==============================================================================
# Pre-Registered Leave-One-Out & Equivalence Ablation Configurations
# ==============================================================================

SMEANS_ABLATION_CONFIGS: dict[str, dict[str, Any]] = {
    "smeans_full": {},
    "loo_no_d15_init": {"init_mode": "random"},
    "loo_no_power_cos": {"schedule_mode": "emeans_cosine"},
    "loo_no_revive_stop": {"use_revive_stop": False},
    "loo_no_sse_donor": {"donor_mode": "usage_only"},
    "loo_no_rms_split": {"split_mode": "checkerboard", "use_feat_scale": False},
    "loo_no_feat_scale": {"split_mode": "rms_iso", "use_feat_scale": False},
    "loo_no_pass_weight": {"use_pass_weighting": False, "pass_w_span": 0.0},
    "loo_no_eb_finalize": {"finalization": "closing_mean", "eb_tau": 0.0},
    "smeans_alpha_coupled_sse": {"couple_sse_to_alpha": True},
    "all_off_emeans_equiv": {
        "init_mode": "random",
        "schedule_mode": "emeans_cosine",
        "lowd_revival": False,
        "use_revive_stop": False,
        "donor_mode": "usage_only",
        "split_mode": "checkerboard",
        "use_feat_scale": False,
        "use_pass_weighting": False,
        "pass_w_span": 0.0,
        "finalization": "closing_mean",
        "eb_tau": 0.0,
    },
}


SMeansEpochStat = EpochStat


class SMeans:
    """Scale- & distortion-aware single/few-epoch clustering (`suayptalha/s-means`) with component toggles."""

    def __init__(
        self,
        d: int,
        k: int,
        nepochs: float = 5.0,
        batch_law: str = "auto",
        batch_size: int | None = None,
        alpha0: float | None = None,
        # --- Independently toggleable s-means components ---
        init_mode: str = "subsampled_d15",
        init_rounds: int = SMEANS_INIT_ROUNDS,
        init_power: float = SMEANS_INIT_POWER,
        schedule_mode: str = "power_cosine",
        alpha_floor: float = SMEANS_ALPHA_FLOOR,
        cosine_power: float = SMEANS_COSINE_POWER,
        revival_mode: str = "split",
        unused_threshold: float | None = DEFAULT_UNUSED_THRESHOLD,
        lowd_revival: bool = True,
        check_unused_every: int = CHECK_UNUSED_EVERY,
        use_revive_stop: bool = True,
        revive_stop_frac: float = SMEANS_REVIVE_STOP_FRAC,
        donor_mode: str = "sse_usage",
        donor_usage_w: float = SMEANS_DONOR_USAGE_W,
        sse_decay: float = SMEANS_SSE_DECAY,
        couple_sse_to_alpha: bool = False,
        split_mode: str = "rms_aniso",
        split_mult: float = SMEANS_SPLIT_MULT,
        use_feat_scale: bool = True,
        var_beta: float = SMEANS_VAR_BETA,
        split_epsilon: float = SPLIT_EPSILON,
        init_usage: float = DEFAULT_INIT_USAGE,
        use_pass_weighting: bool = True,
        pass_w_span: float = SMEANS_PASS_W_SPAN,
        finalization: str = "eb_ema",
        eb_tau: float = SMEANS_EB_TAU,
        # --- Precision, backend, and exact-reference parity options ---
        compute_dtype: torch.dtype | None = None,
        backend: str = "auto",
        exact_ref: bool = False,
        seed: int = 1234,
        verbose: bool = False,
    ) -> None:
        if d < 1:
            raise ValueError(f"Dimension d={d} must be >= 1")
        if k < 1:
            raise ValueError(f"Number of clusters k={k} must be >= 1")
        if not math.isfinite(nepochs) or nepochs <= 0.0:
            raise ValueError(f"nepochs={nepochs} must be finite and > 0")
        if init_mode not in ("subsampled_d15", "random"):
            raise ValueError("init_mode must be 'subsampled_d15' or 'random'")
        if init_rounds < 2:
            raise ValueError("init_rounds must be >= 2")
        if not math.isfinite(init_power) or init_power < 0.0:
            raise ValueError("init_power must be finite and >= 0.0")
        if schedule_mode not in ("power_cosine", "emeans_cosine", "constant"):
            raise ValueError("schedule_mode must be 'power_cosine', 'emeans_cosine', or 'constant'")
        if revival_mode not in ("split", "off"):
            raise ValueError("revival_mode must be 'split' or 'off'")
        if check_unused_every < 1:
            raise ValueError("check_unused_every must be >= 1")
        if not (0.0 <= revive_stop_frac <= 1.0):
            raise ValueError("revive_stop_frac must be in [0.0, 1.0]")
        if donor_mode not in ("sse_usage", "sse_only", "usage_only"):
            raise ValueError("donor_mode must be 'sse_usage', 'sse_only', or 'usage_only'")
        if donor_usage_w < 0.0 or not math.isfinite(donor_usage_w):
            raise ValueError("donor_usage_w must be finite and >= 0.0")
        if not (0.0 < sse_decay < 1.0):
            raise ValueError("sse_decay must be in (0.0, 1.0)")
        if split_mode not in ("rms_aniso", "rms_iso", "checkerboard"):
            raise ValueError("split_mode must be 'rms_aniso', 'rms_iso', or 'checkerboard'")
        if split_mult < 0.0 or not math.isfinite(split_mult):
            raise ValueError("split_mult must be finite and >= 0.0")
        if not (0.0 < var_beta < 1.0):
            raise ValueError("var_beta must be in (0.0, 1.0)")
        if not (0.0 <= pass_w_span <= 1.0):
            raise ValueError("pass_w_span must be in [0.0, 1.0]")
        if finalization not in ("eb_ema", "closing_mean", "ema_only", "none"):
            raise ValueError("finalization must be 'eb_ema', 'closing_mean', 'ema_only', or 'none'")
        if eb_tau < 0.0 or not math.isfinite(eb_tau):
            raise ValueError("eb_tau must be finite and >= 0.0")
        if backend not in ("auto", "triton", "torch", "native"):
            raise ValueError("backend must be 'auto', 'triton', 'torch', or 'native'")
        if unused_threshold is not None and (not math.isfinite(unused_threshold) or unused_threshold < 0.0):
            raise ValueError("unused_threshold must be finite and >= 0.0")
        if not (init_usage > 0.0 and math.isfinite(init_usage)):
            raise ValueError("init_usage must be finite and > 0.0")

        self.d = int(d)
        self.k = int(k)
        self.nepochs = float(nepochs)
        self.batch_law = batch_law
        self.batch_size = batch_size
        self.alpha0 = alpha0

        self.init_mode = init_mode
        self.init_rounds = int(init_rounds)
        self.init_power = float(init_power)
        self.schedule_mode = schedule_mode
        self.alpha_floor = float(alpha_floor)
        self.cosine_power = float(cosine_power)
        self.revival_mode = revival_mode
        self.unused_threshold = unused_threshold
        self.lowd_revival = bool(lowd_revival)
        self.check_unused_every = int(check_unused_every)
        self.use_revive_stop = bool(use_revive_stop)
        self.revive_stop_frac = float(revive_stop_frac)
        self.donor_mode = donor_mode
        self.donor_usage_w = float(donor_usage_w)
        self.sse_decay = float(sse_decay)
        self.couple_sse_to_alpha = bool(couple_sse_to_alpha)
        self.split_mode = split_mode
        self.split_mult = float(split_mult)
        self.use_feat_scale = bool(use_feat_scale)
        self.var_beta = float(var_beta)
        self.split_epsilon = float(split_epsilon)
        self.init_usage = float(init_usage)
        self.use_pass_weighting = bool(use_pass_weighting)
        self.pass_w_span = float(pass_w_span)
        self.finalization = "ema_only" if finalization == "none" else finalization
        self.eb_tau = float(eb_tau)

        self.compute_dtype = compute_dtype
        self.backend = "torch" if backend == "native" else backend
        self.exact_ref = bool(exact_ref)
        self.seed = int(seed)
        self.verbose = bool(verbose)

        self.recipe: EMeansRecipe | None = None
        self.centroids: torch.Tensor | None = None
        self.assignments: torch.Tensor | None = None
        self.iteration_stats: list[EpochStat] = []
        self.n_splits: int = 0
        self.dead_clusters: int = 0
        self.init_time_sec: float = 0.0
        self.train_time_sec: float = 0.0
        self.total_time_sec: float = 0.0
        self._origin64: torch.Tensor | None = None

    def _resolve_compute_dtype(self, x: torch.Tensor) -> torch.dtype:
        if self.compute_dtype is not None:
            return self.compute_dtype
        if x.is_cuda and x.dtype in (torch.bfloat16, torch.float16):
            return x.dtype
        if x.is_cuda and self.backend != "torch":
            cap = torch.cuda.get_device_capability(x.device)
            if cap >= (8, 0):
                return torch.bfloat16
        return torch.float32

    def _build_smeans_recipe(self, n: int, d: int) -> EMeansRecipe:
        resolved_law, reason = resolve_batch_law(self.batch_law, d)
        if self.batch_size is None:
            if resolved_law == "quality":
                c_b, b_k, b_n, b_t = 0.00439175, 0.134749, 0.876473, 0.813344
                cap = max(1, n // 2)
            else:
                c_b, b_k, b_n, b_t = 0.035134, 0.134749, 0.876473, 0.813344
                cap = max(1, round_pow2_smeans(n / 2.0))
            raw_b = c_b * (self.k**b_k) * (n**b_n) * (self.nepochs**b_t)
            b = max(1, min(n, round_pow2_smeans(raw_b), cap))
        else:
            b = max(1, min(int(self.batch_size), n))
            reason = "manual"

        resolved_alpha0 = (
            compute_step_law(n, self.k, self.nepochs, b)
            if self.alpha0 is None
            else float(self.alpha0)
        )
        if self.revival_mode == "off":
            tau = 0.0
        elif self.unused_threshold is None:
            tau = (
                DEFAULT_UNUSED_THRESHOLD
                if (self.lowd_revival or d > NO_REVIVAL_MAX_D)
                else 0.0
            )
        else:
            tau = (
                0.0
                if (not self.lowd_revival and d <= NO_REVIVAL_MAX_D)
                else float(self.unused_threshold)
            )

        ep_rows = tuple(compute_epoch_rows(n, self.nepochs))
        ep_steps = tuple((r + b - 1) // b for r in ep_rows)
        eff_floor = (
            ALPHA_MIN_FACTOR
            if self.schedule_mode == "emeans_cosine"
            else self.alpha_floor
        )
        return EMeansRecipe(
            n=n,
            d=d,
            k=self.k,
            nepochs=self.nepochs,
            batch_law=resolved_law,
            batch_law_reason=reason,
            batch_size=b,
            alpha0=resolved_alpha0,
            alpha_min_factor=eff_floor,
            unused_threshold=tau,
            check_unused_every=self.check_unused_every,
            split_epsilon=self.split_epsilon,
            init_usage=self.init_usage,
            epoch_rows=ep_rows,
            epoch_steps=ep_steps,
            total_steps=sum(ep_steps),
        )

    @torch.no_grad()
    def fit(
        self,
        x: torch.Tensor | np.ndarray,
        init_centroids: torch.Tensor | np.ndarray | None = None,
        capture_assignments: bool = True,
    ) -> tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray | None]:
        """Fit s-means on dataset x of shape (N, D), returning (centroids, assignments)."""
        input_is_numpy = isinstance(x, np.ndarray)
        if input_is_numpy:
            x_np = np.asarray(x)
            if x_np.ndim != 2:
                raise ValueError(f"Expected 2D input array, got shape {x_np.shape}")
            x_t = torch.from_numpy(
                np.ascontiguousarray(
                    x_np,
                    dtype=np.float64 if x_np.dtype == np.float64 else np.float32,
                )
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
        recipe = self._build_smeans_recipe(n=n, d=d)
        self.recipe = recipe

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_init_start = time.perf_counter()

        if init_centroids is None:
            if self.init_mode == "subsampled_d15":
                centroids_init = subsampled_d15_init(
                    x_t,
                    k=self.k,
                    seed=self.seed,
                    rounds=self.init_rounds,
                    power=self.init_power,
                    exact_ref=self.exact_ref,
                )
            else:
                gen_init = torch.Generator(device=device)
                gen_init.manual_seed(int(self.seed) % (2**32))
                perm = torch.randperm(n, generator=gen_init, device=device)[: self.k]
                centroids_init = x_t[perm].float().clone().contiguous()
        else:
            c_in = (
                torch.from_numpy(np.asarray(init_centroids, dtype=np.float64))
                if isinstance(init_centroids, np.ndarray)
                else init_centroids.detach()
            )
            if tuple(c_in.shape) != (self.k, self.d):
                raise ValueError(
                    f"Expected init_centroids shape ({self.k}, {self.d}), got {tuple(c_in.shape)}"
                )
            if not bool(torch.isfinite(c_in).all()):
                raise ValueError("init_centroids must contain only finite values")
            centroids_init = c_in.to(device=device, dtype=torch.float64 if x_t.dtype == torch.float64 else torch.float32).clone().contiguous()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self.init_time_sec = time.perf_counter() - t_init_start

        if self.exact_ref:
            centroids, assignments = self._fit_exact_ref(
                x_t=x_t,
                centroids_init=centroids_init.float().contiguous(),
                comp_dtype=comp_dtype,
                recipe=recipe,
                capture_assignments=capture_assignments,
            )
        else:
            use_triton_fast = (
                device.type == "cuda"
                and HAS_TRITON
                and comp_dtype in (torch.float32, torch.bfloat16, torch.float16)
                and (
                    self.backend == "triton"
                    or (self.backend == "auto" and torch.cuda.get_device_capability(device) >= (8, 0))
                )
            )
            if use_triton_fast:
                centroids, assignments = self._fit_cuda_fast(
                    x_t=x_t,
                    centroids_init=centroids_init,
                    comp_dtype=comp_dtype,
                    recipe=recipe,
                    capture_assignments=capture_assignments,
                )
            else:
                centroids, assignments = self._fit_torch_centered(
                    x_t=x_t,
                    centroids_init=centroids_init,
                    comp_dtype=comp_dtype,
                    recipe=recipe,
                    capture_assignments=capture_assignments,
                )

        self.total_time_sec = self.init_time_sec + self.train_time_sec
        self.centroids = centroids.float().contiguous()
        self.assignments = assignments

        if input_is_numpy:
            return (
                self.centroids.cpu().numpy(),
                self.assignments.cpu().numpy() if self.assignments is not None else None,
            )
        return self.centroids, self.assignments

    def _fit_cuda_fast(
        self,
        x_t: torch.Tensor,
        centroids_init: torch.Tensor,
        comp_dtype: torch.dtype,
        recipe: EMeansRecipe,
        capture_assignments: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Zero-sync, origin-centered, fused-Triton GPU training pipeline."""
        device = x_t.device
        n, d = int(x_t.shape[0]), int(x_t.shape[1])
        k = self.k

        t_train_start = time.perf_counter()

        # Estimate coordinate origin from up to 8192 rows so all BF16/FP16/FP32 norms & sums stay centered
        m_orig = min(n, 8192)
        origin64 = x_t[:m_orig].double().mean(dim=0).contiguous()
        origin32 = origin64.float().contiguous()
        self._origin64 = origin64
        origin_assign = origin64 if x_t.dtype == torch.float64 else origin32

        centroids_c = (centroids_init.to(device=device, dtype=torch.float64) - origin64[None, :]).float().contiguous()
        usage = torch.full((k,), recipe.init_usage, dtype=torch.float32, device=device)
        sse_ema = torch.zeros((k,), dtype=torch.float32, device=device)
        feat_scale = torch.ones((d,), dtype=torch.float32, device=device)
        ones_d = torch.ones((d,), dtype=torch.float32, device=device)

        sums = torch.zeros((k, d), dtype=torch.float32, device=device)
        counts = torch.zeros((k,), dtype=torch.float32, device=device)
        batch_sse = torch.zeros((k,), dtype=torch.float32, device=device)

        need_pass_accum = self.finalization in ("eb_ema", "closing_mean")
        pass_sums = torch.zeros((k, d), dtype=torch.float32, device=device)
        pass_counts = torch.zeros((k,), dtype=torch.float64, device=device)
        raw_pass_counts = torch.zeros((k,), dtype=torch.float64, device=device)

        c_rep = _alloc_c_rep_padded(k, d, comp_dtype, device)
        c_norm = torch.empty((k,), dtype=torch.float32, device=device)
        _launch_prep_centroids(centroids_c, c_rep, c_norm)

        bsz_max = recipe.batch_size
        dist_buf = torch.empty((bsz_max,), dtype=torch.float32, device=device)
        lbl_buf = torch.empty((bsz_max,), dtype=torch.int64, device=device)

        assignments = (
            torch.full((n,), -1, dtype=torch.int64, device=device)
            if capture_assignments
            else None
        )

        split_gen = torch.Generator(device=device)
        split_gen.manual_seed((self.seed + 7919) % (2**32))

        num_epochs = len(recipe.epoch_rows)
        ep_sq_dists_gpu = torch.zeros((num_epochs,), dtype=torch.float64, device=device)
        ep_dot_counts_gpu = torch.zeros((num_epochs,), dtype=torch.float64, device=device)
        ep_cum_splits_gpu = torch.zeros((num_epochs,), dtype=torch.int64, device=device)
        total_splits_gpu = torch.zeros((), dtype=torch.int64, device=device)

        ev_start = torch.cuda.Event(enable_timing=True)
        ev_epochs = [torch.cuda.Event(enable_timing=True) for _ in range(num_epochs)]
        ev_start.record()

        rep_mode = _dtype_to_rep_mode(comp_dtype)
        blk_d_upd = min(256, max(16, triton.next_power_of_2(max(1, d))))
        blk_d_fs = min(128, max(16, triton.next_power_of_2(max(1, d))))
        blk_k_fs = min(64, max(16, triton.next_power_of_2(max(1, k))))
        blk_b_sc = 128
        blk_d_sc = min(128, max(16, triton.next_power_of_2(max(1, d))))

        revive_stop_step = (
            int(self.revive_stop_frac * recipe.total_steps)
            if self.use_revive_stop
            else recipe.total_steps
        )
        track_feat_scale = (
            recipe.unused_threshold > 0.0
            and self.revival_mode == "split"
            and self.split_mode == "rms_aniso"
            and self.use_feat_scale
        )
        split_mode_code = 1 if self.split_mode == "checkerboard" else 0
        eff_pass_span = self.pass_w_span if self.use_pass_weighting else 0.0

        step = 0
        ep_step_counts: list[int] = []

        for ep_idx, rows_in_ep in enumerate(recipe.epoch_rows):
            is_last_epoch = ep_idx == (num_epochs - 1)
            gen_ep = torch.Generator(device=device)
            gen_ep.manual_seed(epoch_seed(self.seed, ep_idx))
            order = torch.randperm(n, generator=gen_ep, device=device)[:rows_in_ep]

            raw_pass_counts.zero_()
            n_sep = (rows_in_ep + bsz_max - 1) // bsz_max
            ep_step_counts.append(n_sep)

            for s_ep in range(n_sep):
                b_start = s_ep * bsz_max
                b_end = min(b_start + bsz_max, rows_in_ep)
                b_cur = b_end - b_start
                batch_idx = order[b_start:b_end]

                d_view = dist_buf[:b_cur]
                l_view = lbl_buf[:b_cur]

                _launch_fast_assign(
                    x_t=x_t,
                    c_rep=c_rep,
                    c_norm=c_norm,
                    row_indices=batch_idx,
                    origin=origin_assign,
                    out_dist=d_view,
                    out_label=l_view,
                )

                if is_last_epoch and assignments is not None:
                    assignments[batch_idx] = l_view

                with torch.cuda.device(device):
                    _smeans_scatter_stats_sse_kernel[(triton.cdiv(b_cur, blk_b_sc), triton.cdiv(d, blk_d_sc))](
                        x_t,
                        batch_idx,
                        origin_assign,
                        l_view,
                        d_view,
                        sums,
                        counts,
                        batch_sse,
                        ep_sq_dists_gpu,
                        int(ep_idx),
                        b_cur,
                        d,
                        x_t.stride(0),
                        x_t.stride(1),
                        sums.stride(0),
                        HAS_INDICES=True,
                        HAS_ORIGIN=True,
                        ORIGIN_FP64=(origin_assign.dtype == torch.float64),
                        BLOCK_B=blk_b_sc,
                        BLOCK_D=blk_d_sc,
                        num_warps=4,
                    )

                if track_feat_scale and (step <= revive_stop_step):
                    with torch.cuda.device(device):
                        _smeans_feat_scale_kernel[(triton.cdiv(d, blk_d_fs),)](
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

                alpha = smeans_scheduled_alpha(
                    recipe.alpha0,
                    step,
                    recipe.total_steps,
                    schedule_mode=self.schedule_mode,
                    alpha_floor=self.alpha_floor,
                    cosine_power=self.cosine_power,
                )
                eff_sse_decay = (1.0 - alpha) if self.couple_sse_to_alpha else self.sse_decay
                accum_pass = bool(is_last_epoch and need_pass_accum)
                w_pass = (
                    smeans_progressive_pass_weight(s_ep, n_sep, eff_pass_span)
                    if accum_pass
                    else 0.0
                )

                will_revive = (
                    recipe.unused_threshold > 0.0
                    and self.revival_mode == "split"
                    and ((step + 1) % recipe.check_unused_every == 0)
                    and (step <= revive_stop_step)
                )
                is_very_last_step = is_last_epoch and (s_ep == n_sep - 1)
                prep_in_ema = (not will_revive) and (not is_very_last_step)

                with torch.cuda.device(device):
                    _smeans_ema_sse_pass_prep_kernel[(k,)](
                        sums,
                        counts,
                        batch_sse,
                        centroids_c,
                        usage,
                        sse_ema,
                        pass_sums,
                        pass_counts,
                        raw_pass_counts,
                        c_rep,
                        c_norm,
                        k,
                        d,
                        sums.stride(0),
                        centroids_c.stride(0),
                        c_rep.stride(0),
                        float(alpha),
                        float(1.0 - alpha),
                        float(eff_sse_decay),
                        float(w_pass),
                        int(alpha >= 1.0),
                        int(accum_pass),
                        int(prep_in_ema),
                        ZERO_STATS=True,
                        REP_MODE=rep_mode,
                        BLOCK_D=blk_d_upd,
                        num_warps=2 if d <= 128 else 4,
                    )

                if will_revive:
                    thresh = usage.mean() * recipe.unused_threshold
                    unused = usage < thresh
                    donors = (~unused) & (usage > 0.0)
                    n_sp_t = torch.minimum(unused.sum(), donors.sum())
                    total_splits_gpu += n_sp_t

                    if self.donor_mode == "usage_only":
                        score_vec = usage
                    else:
                        w_exp = self.donor_usage_w if self.donor_mode == "sse_usage" else 0.0
                        sse_pos = sse_ema.max() > 0.0
                        weighted = sse_ema.clamp_min(1e-12) * (
                            usage.clamp_min(1e-12) ** w_exp if w_exp != 0.0 else 1.0
                        )
                        score_vec = torch.where(sse_pos, weighted, usage)

                    u_rand = torch.rand(
                        (k,),
                        generator=split_gen,
                        device=device,
                        dtype=torch.float32,
                    ).clamp_min_(torch.finfo(torch.float32).tiny)
                    keys = torch.where(
                        donors,
                        score_vec.clamp_min(1e-30).log() - (-u_rand.log()).log(),
                        torch.full_like(usage, float("-inf")),
                    )
                    src_sorted = torch.argsort(keys, descending=True)
                    tgt_sorted = torch.argsort(unused.to(torch.int8), descending=True, stable=True)

                    if self.split_mode == "rms_aniso" and self.use_feat_scale:
                        fvec_s = feat_scale.clamp(1e-10, 1e35).sqrt()
                        fnorm = fvec_s.double().norm().clamp_min(1e-8).float()
                        fvec = fvec_s * (math.sqrt(float(d)) / fnorm)
                    else:
                        fvec = ones_d

                    with torch.cuda.device(device):
                        _smeans_syncfree_revive_split_kernel[(k,)](
                            centroids_c,
                            origin32,
                            usage,
                            sse_ema,
                            fvec,
                            src_sorted,
                            tgt_sorted,
                            n_sp_t,
                            d,
                            centroids_c.stride(0),
                            float(1.0 / max(1, d)),
                            float(self.split_mult),
                            float(recipe.split_epsilon),
                            int((self.seed * 10007 + step * 131) % (2**31 - 1)),
                            int(split_mode_code),
                            HAS_ORIGIN=True,
                            REP_MODE=rep_mode,
                            GUARD_COLLAPSE=True,
                            BLOCK_D=blk_d_upd,
                            num_warps=2 if d <= 128 else 4,
                        )
                    if not is_very_last_step:
                        _launch_prep_centroids(centroids_c, c_rep, c_norm)

                step += 1

            ep_dot_counts_gpu[ep_idx] = torch.dot(raw_pass_counts, raw_pass_counts)
            ep_cum_splits_gpu[ep_idx] = total_splits_gpu
            ev_epochs[ep_idx].record()

        # Finalization in centered frame
        if self.finalization == "eb_ema":
            centroids_c = smeans_eb_finalize(centroids_c, pass_sums, pass_counts, eb_tau=self.eb_tau)
        elif self.finalization == "closing_mean":
            centroids_c = smeans_eb_finalize(centroids_c, pass_sums, pass_counts, eb_tau=0.0)

        centroids_out = (centroids_c.double() + origin64[None, :]).float().contiguous()
        dead_t = (
            (pass_counts <= 0.5).sum()
            if need_pass_accum
            else (raw_pass_counts <= 0.5).sum()
        )

        # Single end-of-training synchronization to stop timer and materialize diagnostics
        torch.cuda.synchronize(device)
        self.train_time_sec = time.perf_counter() - t_train_start

        ep_sq_list = ep_sq_dists_gpu.cpu().tolist()
        ep_dot_list = ep_dot_counts_gpu.cpu().tolist()
        ep_sp_list = ep_cum_splits_gpu.cpu().tolist()
        self.n_splits = int(ep_sp_list[-1]) if ep_sp_list else 0
        self.dead_clusters = int(dead_t.item())

        self.iteration_stats = []
        prev_sp = 0
        for ep_idx, rows_in_ep in enumerate(recipe.epoch_rows):
            cur_sp = int(ep_sp_list[ep_idx])
            el_s = float(ev_start.elapsed_time(ev_epochs[ep_idx])) * 1e-3
            stat = EpochStat(
                epoch=ep_idx + 1,
                rows=rows_in_ep,
                steps=ep_step_counts[ep_idx],
                moving_mse=float(ep_sq_list[ep_idx]) / max(1, rows_in_ep),
                imbalance_factor=float(k * ep_dot_list[ep_idx] / (float(max(1, rows_in_ep)) ** 2)),
                n_splits=cur_sp - prev_sp,
                elapsed_sec=el_s,
            )
            prev_sp = cur_sp
            self.iteration_stats.append(stat)
            if self.verbose:
                print(
                    f"[s-means] pass {ep_idx + 1}/{num_epochs} ({rows_in_ep} rows, {stat.steps} steps): "
                    f"moving_mse={stat.moving_mse:.6f}, splits={stat.n_splits}, time={stat.elapsed_sec:.3f}s"
                )

        return centroids_out, assignments

    def _fit_torch_centered(
        self,
        x_t: torch.Tensor,
        centroids_init: torch.Tensor,
        comp_dtype: torch.dtype,
        recipe: EMeansRecipe,
        capture_assignments: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Shift-safe PyTorch CPU / FP64 / pre-SM80 CUDA execution path."""
        device = x_t.device
        n, d = int(x_t.shape[0]), int(x_t.shape[1])
        k = self.k

        t_train_start = time.perf_counter()

        m_orig = min(n, 8192)
        origin64 = x_t[:m_orig].double().mean(dim=0).contiguous()
        origin32 = origin64.float().contiguous()
        self._origin64 = origin64

        centroids_c = (centroids_init.to(device=device, dtype=torch.float64) - origin64[None, :]).float().contiguous()
        usage = torch.full((k,), recipe.init_usage, dtype=torch.float32, device=device)
        sse_ema = torch.zeros((k,), dtype=torch.float32, device=device)
        feat_scale = torch.ones((d,), dtype=torch.float32, device=device)
        ones_d = torch.ones((d,), dtype=torch.float32, device=device)
        ones_scalar = torch.ones((1,), dtype=torch.float32, device=device)

        sign_chk = torch.ones((d,), dtype=torch.float32, device=device)
        sign_chk[1::2] = -1.0
        split_offset_chk = recipe.split_epsilon * sign_chk

        split_gen = torch.Generator(device=device)
        split_gen.manual_seed((self.seed + 7919) % (2**32))

        assignments = (
            torch.full((n,), -1, dtype=torch.int64, device=device)
            if capture_assignments
            else None
        )

        batch_sums = torch.empty((k, d), dtype=torch.float32, device=device)
        batch_sse = torch.empty((k,), dtype=torch.float32, device=device)
        need_pass_accum = self.finalization in ("eb_ema", "closing_mean")
        pass_sums = torch.zeros((k, d), dtype=torch.float32, device=device)
        pass_counts = torch.zeros((k,), dtype=torch.float64, device=device)

        revive_stop_step = (
            int(self.revive_stop_frac * recipe.total_steps)
            if self.use_revive_stop
            else recipe.total_steps
        )
        eff_pass_span = self.pass_w_span if self.use_pass_weighting else 0.0
        track_feat_scale = (
            recipe.unused_threshold > 0.0
            and self.revival_mode == "split"
            and self.split_mode == "rms_aniso"
            and self.use_feat_scale
        )

        step = 0
        self.n_splits = 0
        self.iteration_stats = []
        num_epochs = len(recipe.epoch_rows)
        last_raw_counts = torch.zeros((k,), dtype=torch.float64, device=device)

        for ep_idx, rows_in_ep in enumerate(recipe.epoch_rows):
            is_last_epoch = ep_idx == (num_epochs - 1)
            gen_ep = torch.Generator(device=device)
            gen_ep.manual_seed(epoch_seed(self.seed, ep_idx))
            order = torch.randperm(n, generator=gen_ep, device=device)[:rows_in_ep]

            raw_pass_counts = torch.zeros((k,), dtype=torch.float64, device=device)
            ep_sq_dist_t = torch.zeros((), dtype=torch.float64, device=device)
            splits_before = self.n_splits
            n_sep = (rows_in_ep + recipe.batch_size - 1) // recipe.batch_size

            for s_ep in range(n_sep):
                b_start = s_ep * recipe.batch_size
                b_end = min(b_start + recipe.batch_size, rows_in_ep)
                batch_idx = order[b_start:b_end]
                x_raw_b = x_t[batch_idx]

                if comp_dtype == torch.float64:
                    xb_c64 = (x_raw_b.double() - origin64[None, :]).contiguous()
                    cb_c64 = centroids_c.double().contiguous()
                    sq_dists, labels = torch_assign_chunked(
                        xb_c64,
                        cb_c64,
                        cb_c64.square().sum(dim=1),
                    )
                    xb_c32 = xb_c64.float()
                else:
                    if x_raw_b.dtype == torch.float64:
                        xb_c32 = (x_raw_b - origin64[None, :]).float().contiguous()
                    else:
                        xb_c32 = (x_raw_b.float() - origin32[None, :]).contiguous()
                    c_rep, c_norm = prepare_represented_centroids(centroids_c, comp_dtype)
                    sq_dists, labels = assign_points(
                        xb_c32,
                        centroids_c,
                        compute_dtype=comp_dtype,
                        backend=self.backend,
                        c_rep_cached=(c_rep, c_norm),
                    )

                if is_last_epoch and assignments is not None:
                    assignments[batch_idx] = labels

                sq_dists_safe = sq_dists.float().clamp(0.0, 1e35)
                ep_sq_dist_t += sq_dists_safe.double().sum()

                counts = torch.bincount(labels, minlength=k).float()
                batch_sums.zero_()
                batch_sums.index_add_(0, labels, xb_c32)
                batch_sse.zero_()
                batch_sse.index_add_(0, labels, sq_dists_safe)

                raw_pass_counts += counts.double()

                alpha = smeans_scheduled_alpha(
                    recipe.alpha0,
                    step,
                    recipe.total_steps,
                    schedule_mode=self.schedule_mode,
                    alpha_floor=self.alpha_floor,
                    cosine_power=self.cosine_power,
                )
                eff_sse_decay = (1.0 - alpha) if self.couple_sse_to_alpha else self.sse_decay

                if is_last_epoch and need_pass_accum:
                    w_pass = smeans_progressive_pass_weight(s_ep, n_sep, eff_pass_span)
                    pass_sums.add_(batch_sums, alpha=w_pass)
                    pass_counts.add_(counts.double(), alpha=w_pass)

                if track_feat_scale and (step <= revive_stop_step):
                    tot_c = counts.sum()
                    safe_tot = tot_c.clamp_min(1.0)
                    batch_mu = batch_sums.sum(dim=0) / safe_tot
                    diff_fs = (batch_sums - counts[:, None] * batch_mu[None, :]).clamp(-1e17, 1e17)
                    cur_var = (diff_fs.square().sum(dim=0) / safe_tot).clamp(1e-12, 1e35)
                    feat_scale = torch.where(
                        tot_c > 0.0,
                        self.var_beta * feat_scale + (1.0 - self.var_beta) * cur_var,
                        feat_scale,
                    )

                touched = counts > 0
                retained = torch.zeros_like(usage) if alpha >= 1.0 else usage * (1.0 - alpha)
                new_mass = retained + alpha * counts
                safe_counts = torch.where(touched, counts, ones_scalar)
                batch_means = batch_sums / safe_counts[:, None]
                weight = (alpha * counts) / torch.where(touched, new_mass, ones_scalar)
                updated_pos = centroids_c + weight[:, None] * (batch_means - centroids_c)
                updated_pos = torch.where((weight == 1.0)[:, None], batch_means, updated_pos)

                centroids_c = torch.where(touched[:, None], updated_pos, centroids_c)
                usage = torch.where(touched, new_mass, usage if alpha >= 1.0 else retained)
                sse_ema.mul_(eff_sse_decay).add_(batch_sse, alpha=1.0 - eff_sse_decay)

                if (
                    recipe.unused_threshold > 0.0
                    and self.revival_mode == "split"
                    and ((step + 1) % recipe.check_unused_every == 0)
                    and (step <= revive_stop_step)
                ):
                    thresh = usage.mean() * recipe.unused_threshold
                    unused_mask = usage < thresh
                    donor_mask = (~unused_mask) & (usage > 0.0)
                    n_unused = int(unused_mask.sum().item())
                    n_donors = int(donor_mask.sum().item())
                    nsplit = min(n_unused, n_donors)
                    if nsplit > 0:
                        if self.donor_mode == "usage_only":
                            score_vec = usage
                        else:
                            w_exp = self.donor_usage_w if self.donor_mode == "sse_usage" else 0.0
                            sse_pos = sse_ema.max() > 0.0
                            weighted = sse_ema.clamp_min(1e-12) * (
                                usage.clamp_min(1e-12) ** w_exp if w_exp != 0.0 else 1.0
                            )
                            score_vec = torch.where(sse_pos, weighted, usage)

                        u_rand = torch.rand(
                            (k,),
                            generator=split_gen,
                            device=device,
                            dtype=torch.float32,
                        ).clamp_min_(torch.finfo(torch.float32).tiny)
                        keys = torch.where(
                            donor_mask,
                            score_vec.clamp_min(1e-30).log() - (-u_rand.log()).log(),
                            torch.full_like(usage, float("-inf")),
                        )
                        src = torch.topk(keys, nsplit).indices
                        tgt = torch.nonzero(unused_mask, as_tuple=False).squeeze(1)[:nsplit]

                        dc = centroids_c[src]
                        if self.split_mode == "checkerboard":
                            abs_dc = dc + origin32[None, :]
                            d_cent = abs_dc * (1.0 - split_offset_chk) - origin32[None, :]
                            c_cent = abs_dc * (1.0 + split_offset_chk) - origin32[None, :]
                        else:
                            if self.split_mode == "rms_aniso" and self.use_feat_scale:
                                fvec_s = feat_scale.clamp(1e-10, 1e35).sqrt()
                                fnorm = fvec_s.double().norm().clamp_min(1e-8).float()
                                fvec = fvec_s * (math.sqrt(float(d)) / fnorm)
                            else:
                                fvec = ones_d
                            rad = (
                                self.split_mult
                                * (
                                    sse_ema[src].clamp_max(1e35)
                                    / (usage[src].clamp_min(1e-6) * float(max(1, d)))
                                )
                                .clamp(1e-10, 1e35)
                                .sqrt()[:, None]
                            )
                            radem = torch.where(
                                torch.rand((nsplit, d), generator=split_gen, device=device) < 0.5,
                                -1.0,
                                1.0,
                            )
                            delta = rad * radem * fvec[None, :]
                            d_cent = dc - delta
                            c_cent = dc + delta

                        _separate_collapsed_splits(d_cent, c_cent, comp_dtype)
                        half_u = 0.5 * usage[src]
                        half_sse = 0.5 * sse_ema[src]
                        centroids_c[src] = d_cent
                        centroids_c[tgt] = c_cent
                        usage[src] = half_u
                        usage[tgt] = half_u
                        sse_ema[src] = half_sse
                        sse_ema[tgt] = half_sse
                        self.n_splits += nsplit

                step += 1

            last_raw_counts = raw_pass_counts
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            ep_sq_dist = float(ep_sq_dist_t.item())
            imbalance = float(
                (k * torch.dot(raw_pass_counts, raw_pass_counts) / (float(max(1, rows_in_ep)) ** 2)).item()
            )
            stat = EpochStat(
                epoch=ep_idx + 1,
                rows=rows_in_ep,
                steps=n_sep,
                moving_mse=ep_sq_dist / max(1, rows_in_ep),
                imbalance_factor=imbalance,
                n_splits=self.n_splits - splits_before,
                elapsed_sec=time.perf_counter() - t_train_start,
            )
            self.iteration_stats.append(stat)
            if self.verbose:
                print(
                    f"[s-means] pass {ep_idx + 1}/{num_epochs} ({rows_in_ep} rows, {n_sep} steps): "
                    f"moving_mse={stat.moving_mse:.6f}, splits={stat.n_splits}, time={stat.elapsed_sec:.3f}s"
                )

        if self.finalization == "eb_ema":
            centroids_c = smeans_eb_finalize(centroids_c, pass_sums, pass_counts, eb_tau=self.eb_tau)
        elif self.finalization == "closing_mean":
            centroids_c = smeans_eb_finalize(centroids_c, pass_sums, pass_counts, eb_tau=0.0)

        centroids_out = (centroids_c.double() + origin64[None, :]).float().contiguous()
        self.dead_clusters = int(
            ((pass_counts if need_pass_accum else last_raw_counts) <= 0.5).sum().item()
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self.train_time_sec = time.perf_counter() - t_train_start
        return centroids_out, assignments

    def _fit_exact_ref(
        self,
        x_t: torch.Tensor,
        centroids_init: torch.Tensor,
        comp_dtype: torch.dtype,
        recipe: EMeansRecipe,
        capture_assignments: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Exact upstream `suayptalha/s-means` reproduction mode (`_cpu.py` & `_gpu.py` RNG/ops)."""
        device = x_t.device
        n, d = int(x_t.shape[0]), int(x_t.shape[1])
        k = self.k
        is_cuda = device.type == "cuda"
        self._origin64 = None

        t_train_start = time.perf_counter()
        centroids = centroids_init.clone().contiguous()
        usage = torch.full((k,), recipe.init_usage, dtype=torch.float32, device=device)
        sse_ema = torch.zeros((k,), dtype=torch.float32, device=device)
        feat_scale = torch.ones((d,), dtype=torch.float32, device=device)

        use_triton_ref = (
            is_cuda
            and HAS_TRITON
            and comp_dtype in (torch.float32, torch.bfloat16, torch.float16)
            and (
                self.backend == "triton"
                or (self.backend == "auto" and torch.cuda.get_device_capability(device) >= (8, 0))
            )
        )

        sums = torch.zeros((k, d), dtype=torch.float32, device=device)
        counts = torch.zeros((k,), dtype=torch.float32, device=device)
        batch_sse = torch.zeros((k,), dtype=torch.float32, device=device)
        pass_dtype = torch.float32 if is_cuda else torch.float64
        pass_sums = torch.zeros((k, d), dtype=pass_dtype, device=device)
        pass_counts = torch.zeros((k,), dtype=pass_dtype, device=device)

        if is_cuda:
            split_rng = torch.Generator(device=device).manual_seed(int(self.seed) * 1009 + 7)
            shuf_rng = torch.Generator(device=device)
        else:
            split_rng = _make_ref_generator(device, self.seed)
            shuf_rng = None

        assignments = (
            torch.full((n,), -1, dtype=torch.int64, device=device)
            if capture_assignments
            else None
        )

        num_epochs = len(recipe.epoch_rows)
        revive_stop_step = (
            int(self.revive_stop_frac * recipe.total_steps)
            if self.use_revive_stop
            else recipe.total_steps
        )
        eff_pass_span = self.pass_w_span if self.use_pass_weighting else 0.0
        need_pass_accum = self.finalization in ("eb_ema", "closing_mean")
        ones_buf = torch.ones((recipe.batch_size,), dtype=torch.float32, device=device)

        step = 0
        self.n_splits = 0
        self.iteration_stats = []

        for ep_idx, rows_in_ep in enumerate(recipe.epoch_rows):
            is_last_epoch = ep_idx == (num_epochs - 1)
            if is_cuda:
                assert shuf_rng is not None
                shuf_rng.manual_seed((int(self.seed) + 100003 * (ep_idx + 1)) % (2**31 - 1))
                perm = torch.randperm(n, generator=shuf_rng, device=device)[:rows_in_ep]
            else:
                gen_ep = _make_ref_generator(device, (int(self.seed) * 1000003 + ep_idx) % (2**32))
                perm = torch.randperm(n, generator=gen_ep, device=device)[:rows_in_ep]

            n_sep = (rows_in_ep + recipe.batch_size - 1) // recipe.batch_size
            ep_sq_dist_t = torch.zeros((), dtype=torch.float64, device=device)
            raw_pass_counts = torch.zeros((k,), dtype=torch.float64, device=device)
            splits_before = self.n_splits

            for s_ep in range(n_sep):
                b_start = s_ep * recipe.batch_size
                b_end = min(b_start + recipe.batch_size, rows_in_ep)
                b_cur = b_end - b_start
                sub_idx = perm[b_start:b_end]
                sub_raw = x_t[sub_idx]

                if use_triton_ref and b_cur >= 16 and k >= 16 and d >= 16:
                    c_rep_ref = _alloc_c_rep_padded(k, d, comp_dtype, device)
                    c_norm_ref = torch.empty((k,), dtype=torch.float32, device=device)
                    _launch_prep_centroids(centroids, c_rep_ref, c_norm_ref)
                    sub_rep = sub_raw if sub_raw.dtype == comp_dtype else sub_raw.to(comp_dtype)
                    sq_dists, labels = _launch_fast_assign(
                        x_t=sub_rep.contiguous(),
                        c_rep=c_rep_ref,
                        c_norm=c_norm_ref,
                        row_indices=None,
                        origin=None,
                    )
                elif is_cuda:
                    c_rep_ref, c_norm_ref = prepare_represented_centroids(centroids, comp_dtype)
                    sq_dists, labels = torch_assign_chunked(sub_raw, c_rep_ref, c_norm_ref)
                else:
                    c_f = centroids.float().contiguous()
                    sq_dists, labels = torch_assign_chunked(
                        sub_raw.float().contiguous(),
                        c_f,
                        c_f.square().sum(dim=1),
                        centroid_chunk_size=256,
                    )

                if is_last_epoch and assignments is not None:
                    assignments[sub_idx] = labels

                ep_sq_dist_t += sq_dists.double().sum()
                sums.zero_()
                counts.zero_()
                batch_sse.zero_()
                sums.index_add_(0, labels, sub_raw.float())
                counts.scatter_add_(0, labels, ones_buf[:b_cur])
                batch_sse.scatter_add_(0, labels, sq_dists.float())
                raw_pass_counts += counts.double()

                alpha = smeans_scheduled_alpha(
                    recipe.alpha0,
                    step,
                    recipe.total_steps,
                    schedule_mode=self.schedule_mode,
                    alpha_floor=self.alpha_floor,
                    cosine_power=self.cosine_power,
                )
                eff_sse_decay = (1.0 - alpha) if self.couple_sse_to_alpha else self.sse_decay
                w_pass = (
                    smeans_progressive_pass_weight(s_ep, n_sep, eff_pass_span)
                    if (is_last_epoch and need_pass_accum)
                    else 0.0
                )

                tot_c = counts.sum()
                if float(tot_c.item()) > 0.0:
                    safe_tot = tot_c.clamp_min(1.0)
                    batch_mu = sums.sum(dim=0) / safe_tot
                    cur_var = ((sums - counts[:, None] * batch_mu[None, :]).pow(2).sum(dim=0) / safe_tot).clamp_min(1e-12)
                    feat_scale.mul_(self.var_beta).add_(cur_var, alpha=1.0 - self.var_beta)

                if is_last_epoch and need_pass_accum:
                    pass_sums.add_(sums.to(pass_dtype), alpha=w_pass)
                    pass_counts.add_(counts.to(pass_dtype), alpha=w_pass)

                touched = counts > 0
                safe_c = torch.where(touched, counts, 1.0)
                mean = sums / safe_c[:, None]
                retained = torch.zeros_like(usage) if alpha >= 1.0 else usage * (1.0 - alpha)
                mass = retained + alpha * counts
                w = alpha * counts / torch.where(touched, mass, 1.0)
                upd = torch.where((w == 1.0)[:, None], mean, centroids + w[:, None] * (mean - centroids))
                centroids = torch.where(touched[:, None], upd, centroids)
                usage = torch.where(touched, mass, usage if alpha >= 1.0 else retained)
                sse_ema.mul_(eff_sse_decay).add_(batch_sse, alpha=1.0 - eff_sse_decay)

                if (
                    recipe.unused_threshold > 0.0
                    and self.revival_mode == "split"
                    and ((step + 1) % recipe.check_unused_every == 0)
                    and (step <= revive_stop_step)
                ):
                    thresh = usage.sum() * recipe.unused_threshold / max(1, k)
                    unused = usage < thresh
                    donors = (~unused) & (usage > 0.0)
                    n_sp = min(int(unused.sum().item()), int(donors.sum().item()))
                    if n_sp > 0:
                        min_u = 1e-38 if is_cuda else torch.finfo(usage.dtype).tiny
                        u_rand = torch.rand(
                            (k,), dtype=torch.float32, device=device, generator=split_rng
                        ).clamp_min(min_u)
                        if self.donor_mode == "usage_only":
                            score_vec = usage
                        elif float(sse_ema.max().item()) > 0.0:
                            w_exp = self.donor_usage_w if self.donor_mode == "sse_usage" else 0.0
                            score_vec = sse_ema.clamp_min(1e-12) * (usage.clamp_min(1e-12) ** w_exp)
                        else:
                            score_vec = usage
                        keys = torch.where(
                            donors,
                            score_vec.clamp_min(1e-30).log() - (-u_rand.log()).log(),
                            float("-inf"),
                        )
                        src = torch.topk(keys, n_sp).indices
                        tgt = (
                            torch.sort(unused.byte(), descending=True, stable=True).indices[:n_sp]
                            if is_cuda
                            else torch.nonzero(unused, as_tuple=True)[0][:n_sp]
                        )

                        fvec = feat_scale.clamp_min(1e-10).sqrt()
                        fvec = fvec / fvec.norm().clamp_min(1e-8) * math.sqrt(float(d))
                        if use_triton_ref:
                            n_sp_t = torch.tensor(n_sp, dtype=torch.int64, device=device)
                            with torch.cuda.device(device):
                                _smeans_syncfree_revive_split_kernel[(n_sp,)](
                                    centroids,
                                    centroids,
                                    usage,
                                    sse_ema,
                                    fvec,
                                    src,
                                    tgt,
                                    n_sp_t,
                                    d,
                                    centroids.stride(0),
                                    float(1.0 / max(1, d)),
                                    float(self.split_mult),
                                    float(recipe.split_epsilon),
                                    int((self.seed * 10007 + step * 131) % (2**31 - 1)),
                                    0,
                                    HAS_ORIGIN=False,
                                    REP_MODE=0,
                                    GUARD_COLLAPSE=False,
                                    BLOCK_D=min(256, max(16, triton.next_power_of_2(max(1, d)))),
                                    num_warps=4,
                                )
                        else:
                            dc = centroids[src]
                            half = 0.5 * usage[src]
                            rad = (
                                self.split_mult
                                * (sse_ema[src] / (usage[src].clamp_min(1e-6) * max(1, d)))
                                .clamp_min(1e-10)
                                .sqrt()[:, None]
                            )
                            radem = torch.where(
                                torch.rand((n_sp, d), device=device, generator=split_rng) < 0.5,
                                -1.0,
                                1.0,
                            )
                            delta = rad * radem * fvec[None, :]
                            centroids[tgt] = dc + delta
                            centroids[src] = dc - delta
                            usage[tgt] = half
                            usage[src] = half
                            hsse = 0.5 * sse_ema[src]
                            sse_ema[tgt] = hsse
                            sse_ema[src] = hsse
                        self.n_splits += n_sp

                step += 1

            if is_cuda:
                torch.cuda.synchronize(device)
            ep_sq_dist = float(ep_sq_dist_t.item())
            imbalance = float(
                (k * torch.dot(raw_pass_counts, raw_pass_counts) / (float(max(1, rows_in_ep)) ** 2)).item()
            )
            self.iteration_stats.append(
                EpochStat(
                    epoch=ep_idx + 1,
                    rows=rows_in_ep,
                    steps=n_sep,
                    moving_mse=ep_sq_dist / max(1, rows_in_ep),
                    imbalance_factor=imbalance,
                    n_splits=self.n_splits - splits_before,
                    elapsed_sec=time.perf_counter() - t_train_start,
                )
            )

        if self.finalization == "eb_ema":
            centroids = smeans_eb_finalize(centroids, pass_sums, pass_counts, eb_tau=self.eb_tau)
        elif self.finalization == "closing_mean":
            centroids = smeans_eb_finalize(centroids, pass_sums, pass_counts, eb_tau=0.0)

        self.dead_clusters = int((pass_counts <= 0.5).sum().item()) if need_pass_accum else 0
        if is_cuda:
            torch.cuda.synchronize(device)
        self.train_time_sec = time.perf_counter() - t_train_start
        return centroids.contiguous(), assignments

    @torch.no_grad()
    def assign(
        self,
        x: torch.Tensor | np.ndarray,
        row_chunk_size: int = 131072,
    ) -> tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray]:
        """Assign rows of x to learned centroids, returning `(sq_distances, labels)`."""
        if self.centroids is None:
            raise RuntimeError("Model has not been fitted yet.")
        input_is_numpy = isinstance(x, np.ndarray)
        dev = self.centroids.device
        if input_is_numpy:
            x_np = np.asarray(x)
            x_t = torch.from_numpy(
                np.ascontiguousarray(
                    x_np, dtype=np.float64 if x_np.dtype == np.float64 else np.float32
                )
            ).to(dev)
        else:
            x_t = x.detach().to(dev)

        n, d = int(x_t.shape[0]), int(x_t.shape[1])
        if d != self.d:
            raise ValueError(f"Expected dimension d={self.d}, got {d}")
        if n == 0:
            empty_d = torch.empty((0,), dtype=torch.float32, device=dev)
            empty_l = torch.empty((0,), dtype=torch.int64, device=dev)
            if input_is_numpy:
                return empty_d.cpu().numpy(), empty_l.cpu().numpy()
            return empty_d, empty_l

        comp_dtype = self._resolve_compute_dtype(x_t)

        if self.exact_ref:
            dists, labels = assign_points(
                x_t,
                self.centroids,
                compute_dtype=comp_dtype,
                backend=self.backend,
                row_chunk_size=row_chunk_size,
            )
        else:
            origin64 = (
                self._origin64.to(dev)
                if self._origin64 is not None
                else self.centroids.double().mean(dim=0).contiguous()
            )
            centroids_c = (self.centroids.double() - origin64[None, :]).float().contiguous()
            use_triton_fast = (
                dev.type == "cuda"
                and HAS_TRITON
                and comp_dtype in (torch.float32, torch.bfloat16, torch.float16)
                and (
                    self.backend == "triton"
                    or (self.backend == "auto" and torch.cuda.get_device_capability(dev) >= (8, 0))
                )
            )
            if use_triton_fast:
                c_rep = _alloc_c_rep_padded(self.k, d, comp_dtype, dev)
                c_norm = torch.empty((self.k,), dtype=torch.float32, device=dev)
                _launch_prep_centroids(centroids_c, c_rep, c_norm)
                orig_in = origin64 if x_t.dtype == torch.float64 else origin64.float().contiguous()
                if n <= row_chunk_size:
                    dists, labels = _launch_fast_assign(
                        x_t=x_t,
                        c_rep=c_rep,
                        c_norm=c_norm,
                        origin=orig_in,
                    )
                else:
                    dists = torch.empty((n,), dtype=torch.float32, device=dev)
                    labels = torch.empty((n,), dtype=torch.int64, device=dev)
                    for i0 in range(0, n, row_chunk_size):
                        i1 = min(i0 + row_chunk_size, n)
                        _launch_fast_assign(
                            x_t=x_t[i0:i1],
                            c_rep=c_rep,
                            c_norm=c_norm,
                            origin=orig_in,
                            out_dist=dists[i0:i1],
                            out_label=labels[i0:i1],
                        )
            elif comp_dtype == torch.float64:
                c64 = (self.centroids.double() - origin64[None, :]).contiguous()
                c64_norm = c64.square().sum(dim=1)
                dists = torch.empty((n,), dtype=torch.float32, device=dev)
                labels = torch.empty((n,), dtype=torch.int64, device=dev)
                for i0 in range(0, n, row_chunk_size):
                    i1 = min(i0 + row_chunk_size, n)
                    xb64 = (x_t[i0:i1].double() - origin64[None, :]).contiguous()
                    d_c, l_c = torch_assign_chunked(xb64, c64, c64_norm)
                    dists[i0:i1] = d_c
                    labels[i0:i1] = l_c
            else:
                orig32 = origin64.float().contiguous()
                c_rep, c_norm = prepare_represented_centroids(centroids_c, comp_dtype)
                dists = torch.empty((n,), dtype=torch.float32, device=dev)
                labels = torch.empty((n,), dtype=torch.int64, device=dev)
                for i0 in range(0, n, row_chunk_size):
                    i1 = min(i0 + row_chunk_size, n)
                    xb_c = (
                        (x_t[i0:i1] - origin64[None, :]).float().contiguous()
                        if x_t.dtype == torch.float64
                        else (x_t[i0:i1].float() - orig32[None, :]).contiguous()
                    )
                    d_c, l_c = assign_points(
                        xb_c,
                        centroids_c,
                        compute_dtype=comp_dtype,
                        backend=self.backend,
                        row_chunk_size=row_chunk_size,
                        c_rep_cached=(c_rep, c_norm),
                    )
                    dists[i0:i1] = d_c
                    labels[i0:i1] = l_c

        if input_is_numpy:
            return dists.cpu().numpy(), labels.cpu().numpy()
        return dists, labels


__all__ = [
    "SMEANS_ABLATION_CONFIGS",
    "SMEANS_ALPHA_FLOOR",
    "SMEANS_COSINE_POWER",
    "SMEANS_DONOR_USAGE_W",
    "SMEANS_EB_TAU",
    "SMEANS_INIT_POWER",
    "SMEANS_INIT_ROUNDS",
    "SMEANS_PASS_W_SPAN",
    "SMEANS_REVIVE_STOP_FRAC",
    "SMEANS_SPLIT_MULT",
    "SMEANS_SSE_DECAY",
    "SMEANS_VAR_BETA",
    "SMeans",
    "SMeansEpochStat",
    "round_pow2_smeans",
    "smeans_eb_finalize",
    "smeans_progressive_pass_weight",
    "smeans_scheduled_alpha",
    "subsampled_d15_init",
]
