"""Systems, Kernel & Hardware Optimizations for e-means and e-means++ on Blackwell GPUs.

Implements five falsifiable systems hypotheses (documented in `docs/optimizations.md`):
  - H-Opt-1: Blackwell-Tuned Split-K / Register-Resident Full-D / TMA Triton Assignment
             (`blackwell_prepare_centroids`, `blackwell_triton_assign`).
  - H-Opt-2: Native FP8 E4M3 Tensor-Core Assignment with Appendix I.1 Centered
             Power-of-Two Scaling & Exact Represented-Norm Matching
             (`FP8Quantizer`, `fp8_assign`).
  - H-Opt-3: (a) Fused Triton Scatter-Stats & Fused EMA + Centroid Preparation Kernels
                 (`fused_scatter_stats`, `fused_ema_and_prepare`), eliminating 15+ per-step
                 elementwise launches and all per-step host-device `.item()` stalls;
             (b) Fused Triton Greedy k-means++ Candidate Evaluation + CUDA Graph Replay
                 (`triton_greedy_kmeans_plusplus`, `fast_seed_emeans_plusplus`).
  - H-Opt-4: Coarse-to-Fine IVF Meta-Centroid Pruned Assignment with Fused Triton
             Inverted-List Tensor-Core Search (`IVFAssigner`) for large k.
  - H-Opt-5: Zero-Copy Fused-Gather / Block-Permuted Contiguous Streaming (`FastEMeans`)
             and Single-Process Multi-GPU Data-Sharded NCCL All-Reduce (`MultiGPUFastEMeans`).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch

from .emeans import EpochStat, _separate_collapsed_splits
from .kernels import assign_points, prepare_represented_centroids, torch_assign_chunked
from .recipe import (
    ALPHA_MIN_FACTOR,
    CHECK_UNUSED_EVERY,
    DEFAULT_INIT_USAGE,
    SPLIT_EPSILON,
    EMeansRecipe,
    build_recipe,
    epoch_seed,
    round_power_of_two,
    scheduled_alpha,
)
from .seeding import (
    SeedingResult,
    block_greedy_kmeans_plusplus_torch,
    lightweight_coreset,
    voronoi_polish_and_priors,
)

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

try:
    from triton.tools.tensor_descriptor import TensorDescriptor

    HAS_TMA_DESC = callable(getattr(TensorDescriptor, "from_tensor", None))
except ImportError:
    TensorDescriptor = None
    HAS_TMA_DESC = False

FLOAT8_E4M3FN = getattr(torch, "float8_e4m3fn", None)
FP8_E4M3_MAX: float = 448.0
FP8_CALIBRATION_TARGET: float = 56.0  # 448 / 8 per Appendix I.1 of arXiv:2611.09501
_INT32_MAX: int = 2**31 - 1


def _pad_columns_aligned(tensor: torch.Tensor, multiple: int) -> torch.Tensor:
    """Pad trailing feature dimension with exact zeros so width and row stride are multiples."""
    d = tensor.shape[1]
    if tensor.stride(1) == 1 and (d % multiple == 0) and (tensor.stride(0) % multiple == 0):
        return tensor
    d_pad = ((d + multiple - 1) // multiple) * multiple
    out = tensor.new_zeros((tensor.shape[0], d_pad))
    out[:, :d] = tensor
    return out


def _is_wide_matrix(rows: int, stride_row: int, cols_padded: int, stride_col: int = 1) -> bool:
    if rows <= 0 or cols_padded <= 0:
        return False
    max_off = (rows - 1) * stride_row + (cols_padded - 1) * stride_col
    return max_off > _INT32_MAX


def _tma_compatible(tensor: torch.Tensor) -> bool:
    if not HAS_TMA_DESC or not tensor.is_cuda or tensor.ndim != 2:
        return False
    cap = torch.cuda.get_device_capability(tensor.device)
    if cap < (9, 0):
        return False
    byte_pitch = tensor.stride(0) * tensor.element_size()
    return (
        tensor.stride(1) == 1
        and (tensor.data_ptr() % 16 == 0)
        and (0 < byte_pitch < 2**40)
        and (byte_pitch % 16 == 0)
        and all(0 < dim <= _INT32_MAX for dim in tensor.shape)
    )


# ==============================================================================
# H-Opt-1 & H-Opt-2: Blackwell-Tuned Triton Assignment Kernels (FP32/BF16/FP16/FP8)
# ==============================================================================

if HAS_TRITON:

    @triton.jit
    def _blackwell_assign_kernel(
        x_ptr,
        row_idx_ptr,
        c_ptr,
        c_norm_ptr,
        out_dist_ptr,
        out_label_ptr,
        B,
        K,
        D,
        stride_xb,
        stride_xd,
        stride_ck,
        stride_cd,
        inv_scale_sq,
        BLOCK_B: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
        FULL_D: tl.constexpr,
        DTYPE_MODE: tl.constexpr,  # 0=FP32, 1=BF16, 2=FP16, 3=FP8_E4M3
        HAS_INDICES: tl.constexpr,
        WIDE_X: tl.constexpr,
        WIDE_C: tl.constexpr,
        USE_TMA: tl.constexpr = False,
        x_desc=None,
        c_desc=None,
    ):
        pid_b = tl.program_id(0).to(tl.int64)
        offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B).to(tl.int64)
        mask_b = offs_b < B

        if HAS_INDICES:
            src_rows = tl.load(row_idx_ptr + offs_b, mask=mask_b, other=0).to(tl.int64)
        else:
            src_rows = offs_b

        row_norm = tl.zeros((BLOCK_B,), dtype=tl.float32)

        if FULL_D:
            offs_d = tl.arange(0, BLOCK_D)
            mask_d = offs_d < D
            if WIDE_X:
                x_off = src_rows[:, None] * stride_xb + offs_d[None, :].to(tl.int64) * stride_xd
            else:
                x_off = src_rows[:, None].to(tl.int32) * stride_xb + offs_d[None, :] * stride_xd
            full_x = tl.load(
                x_ptr + x_off,
                mask=mask_b[:, None] & mask_d[None, :],
                other=0.0,
            )
            if DTYPE_MODE == 0:
                full_x = full_x.to(tl.float32)
            elif DTYPE_MODE == 1:
                full_x = full_x.to(tl.bfloat16)
            elif DTYPE_MODE == 2:
                full_x = full_x.to(tl.float16)
            xf = full_x.to(tl.float32)
            row_norm = tl.sum(xf * xf, axis=1)
        else:
            for d0 in range(0, D, BLOCK_D):
                offs_d = d0 + tl.arange(0, BLOCK_D)
                mask_d = offs_d < D
                if USE_TMA:
                    x_tile = x_desc.load([(pid_b * BLOCK_B).to(tl.int32), d0])
                    x_tile = tl.where(mask_b[:, None] & mask_d[None, :], x_tile, 0.0)
                else:
                    if WIDE_X:
                        x_off = src_rows[:, None] * stride_xb + offs_d[None, :].to(tl.int64) * stride_xd
                    else:
                        x_off = src_rows[:, None].to(tl.int32) * stride_xb + offs_d[None, :] * stride_xd
                    x_tile = tl.load(
                        x_ptr + x_off,
                        mask=mask_b[:, None] & mask_d[None, :],
                        other=0.0,
                    )
                if DTYPE_MODE == 0 and not USE_TMA:
                    x_tile = x_tile.to(tl.float32)
                elif DTYPE_MODE == 1:
                    x_tile = x_tile.to(tl.bfloat16)
                elif DTYPE_MODE == 2:
                    x_tile = x_tile.to(tl.float16)
                xf = x_tile.to(tl.float32)
                row_norm += tl.sum(xf * xf, axis=1)

        best_score = tl.full((BLOCK_B,), float("inf"), dtype=tl.float32)
        best_label = tl.zeros((BLOCK_B,), dtype=tl.int32)

        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K

            if FULL_D:
                offs_d = tl.arange(0, BLOCK_D)
                mask_d = offs_d < D
                if WIDE_C:
                    c_off = offs_k[:, None].to(tl.int64) * stride_ck + offs_d[None, :].to(tl.int64) * stride_cd
                else:
                    c_off = offs_k[:, None] * stride_ck + offs_d[None, :] * stride_cd
                c_tile = tl.load(
                    c_ptr + c_off,
                    mask=mask_k[:, None] & mask_d[None, :],
                    other=0.0,
                )
                if DTYPE_MODE == 0:
                    dots = tl.dot(
                        full_x,
                        tl.trans(c_tile),
                        input_precision="ieee",
                        out_dtype=tl.float32,
                    )
                else:
                    dots = tl.dot(
                        full_x,
                        tl.trans(c_tile),
                        out_dtype=tl.float32,
                    )
            else:
                dots = tl.zeros((BLOCK_B, BLOCK_K), dtype=tl.float32)
                for d0 in range(0, D, BLOCK_D):
                    offs_d = d0 + tl.arange(0, BLOCK_D)
                    mask_d = offs_d < D
                    if USE_TMA:
                        x_tile = x_desc.load([(pid_b * BLOCK_B).to(tl.int32), d0])
                        c_tile = c_desc.load([k0, d0])
                    else:
                        if WIDE_X:
                            x_off = src_rows[:, None] * stride_xb + offs_d[None, :].to(tl.int64) * stride_xd
                        else:
                            x_off = src_rows[:, None].to(tl.int32) * stride_xb + offs_d[None, :] * stride_xd
                        x_tile = tl.load(
                            x_ptr + x_off,
                            mask=mask_b[:, None] & mask_d[None, :],
                            other=0.0,
                        )
                        if WIDE_C:
                            c_off = offs_k[:, None].to(tl.int64) * stride_ck + offs_d[None, :].to(tl.int64) * stride_cd
                        else:
                            c_off = offs_k[:, None] * stride_ck + offs_d[None, :] * stride_cd
                        c_tile = tl.load(
                            c_ptr + c_off,
                            mask=mask_k[:, None] & mask_d[None, :],
                            other=0.0,
                        )
                    if DTYPE_MODE == 0 and not USE_TMA:
                        x_tile = x_tile.to(tl.float32)
                    elif DTYPE_MODE == 1 and not USE_TMA:
                        x_tile = x_tile.to(tl.bfloat16)
                    elif DTYPE_MODE == 2 and not USE_TMA:
                        x_tile = x_tile.to(tl.float16)

                    if DTYPE_MODE == 0:
                        dots = tl.dot(
                            x_tile,
                            tl.trans(c_tile),
                            acc=dots,
                            input_precision="ieee",
                            out_dtype=tl.float32,
                        )
                    else:
                        dots = tl.dot(
                            x_tile,
                            tl.trans(c_tile),
                            acc=dots,
                            out_dtype=tl.float32,
                        )

            c_norm = tl.load(c_norm_ptr + offs_k, mask=mask_k, other=float("inf"))
            scores = c_norm[None, :] - 2.0 * dots
            scores = tl.where(mask_k[None, :], scores, float("inf"))

            tile_min, tile_idx = tl.min(
                scores,
                axis=1,
                return_indices=True,
                return_indices_tie_break_left=True,
            )
            improved = tile_min < best_score
            best_score = tl.where(improved, tile_min, best_score)
            best_label = tl.where(improved, k0 + tile_idx.to(tl.int32), best_label)

        dist = tl.maximum(best_score + row_norm, 0.0)
        if DTYPE_MODE == 3:
            dist = dist * inv_scale_sq
        tl.store(out_dist_ptr + offs_b, dist, mask=mask_b)
        tl.store(out_label_ptr + offs_b, best_label.to(tl.int64), mask=mask_b)

    @triton.jit
    def _blackwell_split_k_kernel(
        x_ptr,
        row_idx_ptr,
        c_ptr,
        c_norm_ptr,
        partial_score_ptr,
        partial_label_ptr,
        B,
        K,
        D,
        stride_xb,
        stride_xd,
        stride_ck,
        stride_cd,
        PARTITIONS: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
        FULL_D: tl.constexpr,
        DTYPE_MODE: tl.constexpr,  # 0=FP32, 1=BF16, 2=FP16, 3=FP8_E4M3
        HAS_INDICES: tl.constexpr,
        WIDE_X: tl.constexpr,
        WIDE_C: tl.constexpr,
    ):
        pid_b = tl.program_id(0).to(tl.int64)
        part = tl.program_id(1)
        offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B).to(tl.int64)
        mask_b = offs_b < B

        if HAS_INDICES:
            src_rows = tl.load(row_idx_ptr + offs_b, mask=mask_b, other=0).to(tl.int64)
        else:
            src_rows = offs_b

        part_size = (K + PARTITIONS - 1) // PARTITIONS
        k_start = part * part_size
        k_end = tl.minimum(k_start + part_size, K)

        best_score = tl.full((BLOCK_B,), float("inf"), dtype=tl.float32)
        best_label = tl.zeros((BLOCK_B,), dtype=tl.int32)

        if FULL_D:
            offs_d = tl.arange(0, BLOCK_D)
            mask_d = offs_d < D
            if WIDE_X:
                x_off = src_rows[:, None] * stride_xb + offs_d[None, :].to(tl.int64) * stride_xd
            else:
                x_off = src_rows[:, None].to(tl.int32) * stride_xb + offs_d[None, :] * stride_xd
            full_x = tl.load(
                x_ptr + x_off,
                mask=mask_b[:, None] & mask_d[None, :],
                other=0.0,
            )
            if DTYPE_MODE == 0:
                full_x = full_x.to(tl.float32)
            elif DTYPE_MODE == 1:
                full_x = full_x.to(tl.bfloat16)
            elif DTYPE_MODE == 2:
                full_x = full_x.to(tl.float16)

        k0 = k_start
        while k0 < k_end:
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < k_end

            if FULL_D:
                offs_d = tl.arange(0, BLOCK_D)
                mask_d = offs_d < D
                if WIDE_C:
                    c_off = offs_k[:, None].to(tl.int64) * stride_ck + offs_d[None, :].to(tl.int64) * stride_cd
                else:
                    c_off = offs_k[:, None] * stride_ck + offs_d[None, :] * stride_cd
                c_tile = tl.load(
                    c_ptr + c_off,
                    mask=mask_k[:, None] & mask_d[None, :],
                    other=0.0,
                )
                if DTYPE_MODE == 0:
                    dots = tl.dot(
                        full_x,
                        tl.trans(c_tile),
                        input_precision="ieee",
                        out_dtype=tl.float32,
                    )
                else:
                    dots = tl.dot(
                        full_x,
                        tl.trans(c_tile),
                        out_dtype=tl.float32,
                    )
            else:
                dots = tl.zeros((BLOCK_B, BLOCK_K), dtype=tl.float32)
                for d0 in range(0, D, BLOCK_D):
                    offs_d = d0 + tl.arange(0, BLOCK_D)
                    mask_d = offs_d < D
                    if WIDE_X:
                        x_off = src_rows[:, None] * stride_xb + offs_d[None, :].to(tl.int64) * stride_xd
                    else:
                        x_off = src_rows[:, None].to(tl.int32) * stride_xb + offs_d[None, :] * stride_xd
                    x_tile = tl.load(
                        x_ptr + x_off,
                        mask=mask_b[:, None] & mask_d[None, :],
                        other=0.0,
                    )
                    if DTYPE_MODE == 0:
                        x_tile = x_tile.to(tl.float32)
                    elif DTYPE_MODE == 1:
                        x_tile = x_tile.to(tl.bfloat16)
                    elif DTYPE_MODE == 2:
                        x_tile = x_tile.to(tl.float16)

                    if WIDE_C:
                        c_off = offs_k[:, None].to(tl.int64) * stride_ck + offs_d[None, :].to(tl.int64) * stride_cd
                    else:
                        c_off = offs_k[:, None] * stride_ck + offs_d[None, :] * stride_cd
                    c_tile = tl.load(
                        c_ptr + c_off,
                        mask=mask_k[:, None] & mask_d[None, :],
                        other=0.0,
                    )
                    if DTYPE_MODE == 0:
                        dots = tl.dot(
                            x_tile,
                            tl.trans(c_tile),
                            acc=dots,
                            input_precision="ieee",
                            out_dtype=tl.float32,
                        )
                    else:
                        dots = tl.dot(
                            x_tile,
                            tl.trans(c_tile),
                            acc=dots,
                            out_dtype=tl.float32,
                        )

            c_norm = tl.load(c_norm_ptr + offs_k, mask=mask_k, other=float("inf"))
            scores = c_norm[None, :] - 2.0 * dots
            scores = tl.where(mask_k[None, :], scores, float("inf"))

            tile_min, tile_idx = tl.min(
                scores,
                axis=1,
                return_indices=True,
                return_indices_tie_break_left=True,
            )
            improved = tile_min < best_score
            best_score = tl.where(improved, tile_min, best_score)
            best_label = tl.where(improved, k0 + tile_idx.to(tl.int32), best_label)
            k0 += BLOCK_K

        out_off = offs_b * PARTITIONS + part
        tl.store(partial_score_ptr + out_off, best_score, mask=mask_b)
        tl.store(partial_label_ptr + out_off, best_label, mask=mask_b)

    @triton.jit
    def _blackwell_reduce_split_k_kernel(
        x_ptr,
        row_idx_ptr,
        partial_score_ptr,
        partial_label_ptr,
        out_dist_ptr,
        out_label_ptr,
        B,
        D,
        stride_xb,
        stride_xd,
        inv_scale_sq,
        PARTITIONS: tl.constexpr,
        BLOCK_PART: tl.constexpr,
        BLOCK_D: tl.constexpr,
        DTYPE_MODE: tl.constexpr,
        HAS_INDICES: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        src_row = tl.load(row_idx_ptr + row).to(tl.int64) if HAS_INDICES else row

        row_norm = tl.zeros((), dtype=tl.float32)
        for d0 in range(0, D, BLOCK_D):
            offs_d = d0 + tl.arange(0, BLOCK_D).to(tl.int64)
            mask_d = offs_d < D
            x_val = tl.load(
                x_ptr + src_row * stride_xb + offs_d * stride_xd,
                mask=mask_d,
                other=0.0,
            )
            if DTYPE_MODE == 1:
                x_val = x_val.to(tl.bfloat16)
            elif DTYPE_MODE == 2:
                x_val = x_val.to(tl.float16)
            xf = x_val.to(tl.float32)
            row_norm += tl.sum(xf * xf, axis=0)

        offs_p = tl.arange(0, BLOCK_PART)
        mask_p = offs_p < PARTITIONS
        p_off = row * PARTITIONS + offs_p
        p_scores = tl.load(partial_score_ptr + p_off, mask=mask_p, other=float("inf"))
        best_score, best_part = tl.min(
            p_scores,
            axis=0,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
        p_labels = tl.load(partial_label_ptr + p_off, mask=mask_p, other=0)
        best_label = tl.sum(tl.where(offs_p == best_part, p_labels, 0), axis=0)

        dist = tl.maximum(best_score + row_norm, 0.0)
        if DTYPE_MODE == 3:
            dist = dist * inv_scale_sq
        tl.store(out_dist_ptr + row, dist)
        tl.store(out_label_ptr + row, best_label.to(tl.int64))


def _dtype_mode(dtype: torch.dtype) -> int:
    if dtype == torch.float32:
        return 0
    if dtype == torch.bfloat16:
        return 1
    if dtype == torch.float16:
        return 2
    if FLOAT8_E4M3FN is not None and dtype == FLOAT8_E4M3FN:
        return 3
    raise TypeError(f"Unsupported assignment dtype: {dtype}")


def _choose_split_k_partitions(
    B: int,
    K: int,
    num_sms: int,
    block_b: int = 64,
    block_k: int = 128,
    min_tiles_per_part: int = 4,
    max_partitions: int = 64,
) -> int:
    """Choose power-of-two Split-K factor targeting ~2.5 waves across Blackwell's 188 SMs."""
    k_blocks = triton.cdiv(K, block_k)
    b_blocks = triton.cdiv(B, block_b)
    target_programs = max(384, 2 * num_sms)
    max_allowed = max(1, min(k_blocks // min_tiles_per_part, max_partitions))
    partitions = 1
    while (2 * partitions <= max_allowed) and (b_blocks * partitions < target_programs):
        partitions *= 2
    return partitions


def blackwell_prepare_centroids(
    centroids: torch.Tensor,
    compute_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare represented centroids and exact represented squared norms (Appendix I.1)."""
    c_rep = centroids.to(compute_dtype).contiguous()
    c_norm = c_rep.float().square().sum(dim=1).contiguous()
    return c_rep, c_norm


def blackwell_triton_assign(
    x: torch.Tensor,
    c_rep: torch.Tensor,
    c_norm: torch.Tensor,
    row_indices: torch.Tensor | None = None,
    inv_scale_sq: float = 1.0,
    force_split_k: bool | None = None,
    allow_tma: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Blackwell-tuned Triton nearest-centroid dispatcher (H-Opt-1).

    Combines:
      1. `FULL_D` single-load register caching of `x` when `D <= 128`.
      2. `Split-K` 2D grid `(B_tiles, K_splits)` when `B_tiles` under-occupies
         Blackwell's 188 SMs and `K >= 2048` (including `FULL_D` inside Split-K).
      3. Wide `BLOCK_B=128, BLOCK_K=256, num_warps=8` Tensor-Core tiles + SM120 TMA
         bulk async loads (`TensorDescriptor`) when `D > 128` and `B, K` are large.
    """
    if not HAS_TRITON or not x.is_cuda:
        x_sub = x if row_indices is None else x[row_indices]
        return torch_assign_chunked(x_sub, c_rep, c_norm)

    B = int(row_indices.shape[0]) if row_indices is not None else int(x.shape[0])
    D = int(x.shape[1])
    K = int(c_rep.shape[0])
    device = x.device

    out_dist = torch.empty((B,), dtype=torch.float32, device=device)
    out_label = torch.empty((B,), dtype=torch.int64, device=device)
    if B == 0:
        return out_dist, out_label
    if K == 0:
        raise ValueError("Centroid table must be non-empty (K >= 1)")

    d_mode = _dtype_mode(c_rep.dtype)
    is_reduced = d_mode != 0
    is_fp8 = d_mode == 3
    if is_fp8 and x.dtype != FLOAT8_E4M3FN:
        raise TypeError("FP8 assignment requires E4M3 encoded x; use fp8_assign()")

    has_indices = row_indices is not None
    if has_indices:
        row_indices_c = row_indices.contiguous()
    else:
        row_indices_c = None
    x_in = x if x.stride(1) == 1 else x.contiguous()

    c_in = c_rep if c_rep.stride(1) == 1 else c_rep.contiguous()
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    elem_bytes = c_in.element_size()

    # FP32 has 4B elements so FULL_D fits registers & SM120 99KB SMEM cleanly up to D <= 64;
    # BF16/FP16 (2B) and FP8 (1B) fit FULL_D cleanly up to D <= 128.
    full_d = (D <= 128) if is_reduced else (D <= 64)
    min_dot_d = 32 if is_fp8 else 16
    max_smem_bytes = 81920  # Safe budget below SM120's 101,376 B per-CTA hardware limit

    split_block_b = 64
    split_block_k = 128
    partitions = _choose_split_k_partitions(
        B, K, num_sms=num_sms, block_b=split_block_b, block_k=split_block_k
    )
    if force_split_k is True:
        partitions = max(2, partitions) if K >= 2 else 1
        use_split_k = partitions > 1
    elif force_split_k is False:
        use_split_k = False
    else:
        use_split_k = partitions > 1

    if use_split_k:
        if full_d:
            block_d = max(min_dot_d, triton.next_power_of_2(max(1, D)))
        else:
            block_d = 128 if is_fp8 else (64 if is_reduced else 32)
        split_stages = 3
        split_smem_stage = ((0 if full_d else split_block_b) + split_block_k) * block_d * elem_bytes
        while split_stages > 1 and split_smem_stage * split_stages > max_smem_bytes:
            split_stages -= 1

        wide_x = _is_wide_matrix(x_in.shape[0] + split_block_b, x_in.stride(0), triton.cdiv(D, block_d) * block_d, x_in.stride(1))
        wide_c = _is_wide_matrix(K + split_block_k, c_in.stride(0), triton.cdiv(D, block_d) * block_d, c_in.stride(1))

        partial_score = torch.empty((B, partitions), dtype=torch.float32, device=device)
        partial_label = torch.empty((B, partitions), dtype=torch.int32, device=device)

        with torch.cuda.device(device):
            _blackwell_split_k_kernel[(triton.cdiv(B, split_block_b), partitions)](
                x_in,
                row_indices_c,
                c_in,
                c_norm,
                partial_score,
                partial_label,
                B,
                K,
                D,
                x_in.stride(0),
                x_in.stride(1),
                c_in.stride(0),
                c_in.stride(1),
                PARTITIONS=partitions,
                BLOCK_B=split_block_b,
                BLOCK_K=split_block_k,
                BLOCK_D=block_d,
                FULL_D=full_d,
                DTYPE_MODE=d_mode,
                HAS_INDICES=has_indices,
                WIDE_X=wide_x,
                WIDE_C=wide_c,
                num_warps=4 if full_d else 8,
                num_stages=split_stages,
            )
            _blackwell_reduce_split_k_kernel[(B,)](
                x_in,
                row_indices_c,
                partial_score,
                partial_label,
                out_dist,
                out_label,
                B,
                D,
                x_in.stride(0),
                x_in.stride(1),
                float(inv_scale_sq),
                PARTITIONS=partitions,
                BLOCK_PART=max(16, triton.next_power_of_2(partitions)),
                BLOCK_D=128,
                DTYPE_MODE=d_mode,
                HAS_INDICES=has_indices,
                num_warps=4,
            )
        return out_dist, out_label

    # Direct single-pass kernel (either FULL_D register-resident or wide tiled + TMA)
    if full_d:
        block_d = max(min_dot_d, triton.next_power_of_2(max(1, D)))
        # Keep >= 188 CTAs across SMs when B allows; widen to 128x256 when B is large
        if is_reduced and B >= 128 * num_sms and K >= 256:
            block_b, block_k, num_warps, num_stages = 128, 256 if is_fp8 else 128, 8, 3
        else:
            block_b = 64
            block_k = 128 if K >= 128 else 64
            num_warps = 4
            num_stages = 3
        use_tma = False
    else:
        # Size BLOCK_D so (block_b + block_k) * block_d * elem_bytes == 24,576 B/stage at 128x256,
        # preserving full num_stages=3 async TMA pipelining below SM120's 99 KB dynamic SMEM limit.
        block_d = 64 if is_fp8 else 32
        if is_reduced and B >= 128 * num_sms and K >= 256:
            block_b, block_k = 128, 256
        else:
            block_b = 64
            block_k = 256 if (is_reduced and K >= 256) else 128
        num_warps = 8
        num_stages = 3
        use_tma = (
            allow_tma
            and is_reduced
            and (not has_indices)
            and (x_in.dtype == c_in.dtype)
            and (D % 16 == 0)
            and _tma_compatible(x_in)
            and _tma_compatible(c_in)
        )

    smem_per_stage = ((0 if full_d else block_b) + block_k) * block_d * elem_bytes
    while num_stages > 1 and smem_per_stage * num_stages > max_smem_bytes:
        num_stages -= 1

    wide_x = _is_wide_matrix(x_in.shape[0] + block_b, x_in.stride(0), triton.cdiv(D, block_d) * block_d, x_in.stride(1))
    wide_c = _is_wide_matrix(K + block_k, c_in.stride(0), triton.cdiv(D, block_d) * block_d, c_in.stride(1))

    tma_kwargs = {}
    if use_tma:
        tma_kwargs = {
            "x_desc": TensorDescriptor.from_tensor(x_in, [block_b, block_d]),
            "c_desc": TensorDescriptor.from_tensor(c_in, [block_k, block_d]),
        }

    with torch.cuda.device(device):
        _blackwell_assign_kernel[(triton.cdiv(B, block_b),)](
            x_in,
            row_indices_c,
            c_in,
            c_norm,
            out_dist,
            out_label,
            B,
            K,
            D,
            x_in.stride(0),
            x_in.stride(1),
            c_in.stride(0),
            c_in.stride(1),
            float(inv_scale_sq),
            BLOCK_B=block_b,
            BLOCK_K=block_k,
            BLOCK_D=block_d,
            FULL_D=full_d,
            DTYPE_MODE=d_mode,
            HAS_INDICES=has_indices,
            WIDE_X=wide_x,
            WIDE_C=wide_c,
            USE_TMA=use_tma,
            **tma_kwargs,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return out_dist, out_label


# ==============================================================================
# H-Opt-2: FP8 E4M3 Quantizer (Appendix I.1 Centered Power-of-Two Scaling)
# ==============================================================================

if HAS_TRITON:

    @triton.jit
    def _encode_fp8_rows_kernel(
        x_ptr,
        row_idx_ptr,
        shift_ptr,
        out_fp8_ptr,
        N,
        D,
        stride_xb,
        stride_xd,
        stride_outb,
        scale,
        HAS_SHIFT: tl.constexpr,
        HAS_INDICES: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        src_row = tl.load(row_idx_ptr + row).to(tl.int64) if HAS_INDICES else row
        for d0 in range(0, stride_outb, BLOCK_D):
            cols = d0 + tl.arange(0, BLOCK_D).to(tl.int64)
            mask_d = cols < D
            val = tl.load(
                x_ptr + src_row * stride_xb + cols * stride_xd,
                mask=mask_d,
                other=0.0,
            ).to(tl.float32)
            if HAS_SHIFT:
                sh = tl.load(shift_ptr + cols, mask=mask_d, other=0.0).to(tl.float32)
                val = (val - sh) * scale
            else:
                val = val * scale
            val = tl.maximum(-448.0, tl.minimum(448.0, val))
            val = tl.where(mask_d, val, 0.0)
            code = val.to(tl.float8e4nv, fp_downcast_rounding="rtne")
            tl.store(out_fp8_ptr + row * stride_outb + cols, code, mask=cols < stride_outb)

    @triton.jit
    def _prepare_fp8_centroids_kernel(
        c_ptr,
        shift_ptr,
        out_fp8_ptr,
        out_norm_ptr,
        K,
        D,
        stride_ck,
        stride_cd,
        stride_outk,
        scale,
        HAS_SHIFT: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        k_idx = tl.program_id(0).to(tl.int64)
        norm_acc = tl.zeros((), dtype=tl.float32)
        for d0 in range(0, stride_outk, BLOCK_D):
            cols = d0 + tl.arange(0, BLOCK_D).to(tl.int64)
            mask_d = cols < D
            val = tl.load(
                c_ptr + k_idx * stride_ck + cols * stride_cd,
                mask=mask_d,
                other=0.0,
            ).to(tl.float32)
            if HAS_SHIFT:
                sh = tl.load(shift_ptr + cols, mask=mask_d, other=0.0).to(tl.float32)
                val = (val - sh) * scale
            else:
                val = val * scale
            val = tl.maximum(-448.0, tl.minimum(448.0, val))
            val = tl.where(mask_d, val, 0.0)
            code = val.to(tl.float8e4nv, fp_downcast_rounding="rtne")
            tl.store(out_fp8_ptr + k_idx * stride_outk + cols, code, mask=cols < stride_outk)
            code_f = code.to(tl.float32)
            norm_acc += tl.sum(code_f * code_f, axis=0)
        tl.store(out_norm_ptr + k_idx, norm_acc)


def _compute_pow2_scale(max_abs: float, target_max: float = FP8_CALIBRATION_TARGET) -> tuple[int, float]:
    """Largest power-of-two multiplier s = 2^e keeping sampled magnitudes <= target_max."""
    if not math.isfinite(max_abs) or max_abs <= 0.0:
        return 0, 1.0
    exp = int(math.floor(math.log2(target_max / max_abs)))
    exp = max(-60, min(60, exp))
    return exp, math.ldexp(1.0, exp)


@dataclass
class FP8Quantizer:
    """Appendix I.1 Centered Power-of-Two FP8 E4M3 Quantizer with Exact Norm Matching."""

    d: int
    scale: float
    exponent: int
    shift: torch.Tensor | None  # (D,) float32 per-coordinate mean if centered, else None
    padded_d: int

    @property
    def inv_scale_sq(self) -> float:
        return math.ldexp(1.0, -2 * self.exponent)

    @classmethod
    @torch.no_grad()
    def calibrate(
        cls,
        x: torch.Tensor,
        center_mode: str = "auto",
        max_samples: int = 262144,
        seed: int = 1234,
    ) -> FP8Quantizer:
        """Fit power-of-two FP8 multiplier and optional per-coordinate centering (Appendix I.1)."""
        if center_mode not in ("auto", "always", "never"):
            raise ValueError(f"center_mode must be 'auto', 'always', or 'never'; got {center_mode!r}")
        n, d = x.shape
        device = x.device
        padded_d = ((d + 63) // 64) * 64

        if n <= max_samples:
            x_cal = x.float()
        else:
            gen = torch.Generator(device=device)
            gen.manual_seed(int(seed) % (2**32))
            idx = torch.randperm(n, generator=gen, device=device)[:max_samples]
            x_cal = x[idx].float()

        raw_max = float(x_cal.abs().max().item()) if n > 0 else 0.0
        exp_raw, scale_raw = _compute_pow2_scale(raw_max)

        if center_mode == "never" or n == 0:
            return cls(d=d, scale=scale_raw, exponent=exp_raw, shift=None, padded_d=padded_d)

        mu = x_cal.double().mean(dim=0).float().contiguous()
        cen_max = float((x_cal - mu[None, :]).abs().max().item())
        exp_cen, scale_cen = _compute_pow2_scale(cen_max)

        if center_mode == "always":
            return cls(d=d, scale=scale_cen, exponent=exp_cen, shift=mu, padded_d=padded_d)

        # Appendix I.1 auto-calibration check on <= 2048 points and <= 1024 candidate centroids:
        # Subtract mean iff uncentered FP8 distortion increase > 0.05% vs FP64 AND centering
        # reduces that excess by at least one third.
        if exp_cen <= exp_raw:
            return cls(d=d, scale=scale_raw, exponent=exp_raw, shift=None, padded_d=padded_d)

        m_cal = x_cal.shape[0]
        n_q = min(2048, m_cal)
        n_c = min(1024, m_cal)
        xq64 = x_cal[:n_q].double()
        xc64 = x_cal[m_cal - n_c :].double()
        d2_64 = (
            xq64.square().sum(dim=1)[:, None]
            + xc64.square().sum(dim=1)[None, :]
            - 2.0 * (xq64 @ xc64.t())
        ).clamp_min_(0.0)
        labels_64 = d2_64.argmin(dim=1)
        mse_64 = float(d2_64[torch.arange(n_q, device=device), labels_64].sum().item())
        if mse_64 <= 1e-30:
            return cls(d=d, scale=scale_raw, exponent=exp_raw, shift=None, padded_d=padded_d)

        def _eval_fp8_mse(shift_t: torch.Tensor | None, sc: float) -> float:
            xq_s = ((x_cal[:n_q] - shift_t) * sc) if shift_t is not None else (x_cal[:n_q] * sc)
            xc_s = ((x_cal[m_cal - n_c :] - shift_t) * sc) if shift_t is not None else (x_cal[m_cal - n_c :] * sc)
            xq_8 = xq_s.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(FLOAT8_E4M3FN).float()
            xc_8 = xc_s.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(FLOAT8_E4M3FN).float()
            scores_8 = xc_8.square().sum(dim=1)[None, :] - 2.0 * (xq_8 @ xc_8.t())
            lbl_8 = scores_8.argmin(dim=1)
            return float(d2_64[torch.arange(n_q, device=device), lbl_8].sum().item())

        mse_raw = _eval_fp8_mse(None, scale_raw)
        rel_excess_raw = (mse_raw - mse_64) / mse_64
        if rel_excess_raw > 5e-4:
            mse_cen = _eval_fp8_mse(mu, scale_cen)
            rel_excess_cen = (mse_cen - mse_64) / mse_64
            if rel_excess_cen <= (2.0 / 3.0) * rel_excess_raw:
                return cls(d=d, scale=scale_cen, exponent=exp_cen, shift=mu, padded_d=padded_d)

        return cls(d=d, scale=scale_raw, exponent=exp_raw, shift=None, padded_d=padded_d)

    def encode_rows(
        self,
        x: torch.Tensor,
        row_indices: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode rows (N, D) into 64-byte-aligned E4M3FN tensor (N, padded_d)."""
        n_rows = int(row_indices.shape[0]) if row_indices is not None else int(x.shape[0])
        if out is None:
            out = torch.empty((n_rows, self.padded_d), dtype=FLOAT8_E4M3FN, device=x.device)
        if n_rows == 0:
            return out

        if HAS_TRITON and x.is_cuda:
            x_c = x if x.stride(1) == 1 else x.contiguous()
            has_idx = row_indices is not None
            idx_c = row_indices.contiguous() if has_idx else None
            has_shift = self.shift is not None
            shift_c = self.shift if has_shift else None
            with torch.cuda.device(x.device):
                _encode_fp8_rows_kernel[(n_rows,)](
                    x_c,
                    idx_c,
                    shift_c,
                    out,
                    n_rows,
                    self.d,
                    x_c.stride(0),
                    x_c.stride(1),
                    out.stride(0),
                    float(self.scale),
                    HAS_SHIFT=has_shift,
                    HAS_INDICES=has_idx,
                    BLOCK_D=min(256, max(64, triton.next_power_of_2(self.padded_d))),
                    num_warps=4,
                )
            return out

        rows = x[row_indices].float() if row_indices is not None else x.float()
        if self.shift is not None:
            rows = (rows - self.shift[None, :]) * self.scale
        else:
            rows = rows * self.scale
        out.zero_()
        out[:, : self.d] = rows.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(FLOAT8_E4M3FN)
        return out

    def prepare_centroids(
        self,
        centroids: torch.Tensor,
        c_fp8_out: torch.Tensor | None = None,
        c_norm_out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode centroids (K, D) to E4M3FN and compute exact represented ||c_fp8||^2."""
        k = int(centroids.shape[0])
        device = centroids.device
        c_fp8 = (
            torch.empty((k, self.padded_d), dtype=FLOAT8_E4M3FN, device=device)
            if c_fp8_out is None
            else c_fp8_out
        )
        c_norm = (
            torch.empty((k,), dtype=torch.float32, device=device)
            if c_norm_out is None
            else c_norm_out
        )
        if k == 0:
            return c_fp8, c_norm

        if HAS_TRITON and centroids.is_cuda:
            c_c = centroids if centroids.stride(1) == 1 else centroids.contiguous()
            has_shift = self.shift is not None
            shift_c = self.shift if has_shift else None
            with torch.cuda.device(device):
                _prepare_fp8_centroids_kernel[(k,)](
                    c_c,
                    shift_c,
                    c_fp8,
                    c_norm,
                    k,
                    self.d,
                    c_c.stride(0),
                    c_c.stride(1),
                    c_fp8.stride(0),
                    float(self.scale),
                    HAS_SHIFT=has_shift,
                    BLOCK_D=min(256, max(64, triton.next_power_of_2(self.padded_d))),
                    num_warps=4,
                )
            return c_fp8, c_norm

        c_fp8 = self.encode_rows(centroids, out=c_fp8)
        c_norm.copy_(c_fp8[:, : self.d].float().square().sum(dim=1))
        return c_fp8, c_norm


def _separate_collapsed_splits_branchless(
    donor: torch.Tensor,
    child: torch.Tensor,
    compute_dtype: torch.dtype,
) -> None:
    """Branchless represented-precision split separation with zero host-device syncs."""
    dq = donor.to(compute_dtype)
    cq = child.to(compute_dtype)
    collapsed = (dq == cq).all(dim=1)
    val = cq[:, 0]
    up = torch.nextafter(val, torch.full_like(val, float("inf")))
    down = torch.nextafter(val, torch.full_like(val, float("-inf")))
    neighbor = torch.where(torch.isfinite(up), up, down).float()
    child[:, 0] = torch.where(collapsed, neighbor, child[:, 0])


def _separate_collapsed_splits_fp8(
    donor: torch.Tensor,
    child: torch.Tensor,
    quantizer: FP8Quantizer,
) -> None:
    """Branchless FP8 split separation (handles -0/+0 and large coordinate shift without host sync)."""
    dq = quantizer.encode_rows(donor)[:, : quantizer.d]
    cq = quantizer.encode_rows(child)[:, : quantizer.d]
    collapsed = (dq.float() == cq.float()).all(dim=1)

    c0_f = cq[:, 0].float()
    step_fp8 = torch.maximum(
        torch.full_like(c0_f, 2.0**-9),
        c0_f.abs() * 0.125,
    )
    inv_s = 1.0 / quantizer.scale
    orig0 = child[:, 0]
    if quantizer.shift is not None:
        shift0 = quantizer.shift[0]
        pos_cand = torch.maximum(
            (c0_f + step_fp8) * inv_s + shift0,
            torch.nextafter(orig0, torch.full_like(orig0, float("inf"))),
        )
        neg_cand = torch.minimum(
            (c0_f - step_fp8) * inv_s + shift0,
            torch.nextafter(orig0, torch.full_like(orig0, float("-inf"))),
        )
    else:
        pos_cand = torch.maximum(
            (c0_f + step_fp8) * inv_s,
            torch.nextafter(orig0, torch.full_like(orig0, float("inf"))),
        )
        neg_cand = torch.minimum(
            (c0_f - step_fp8) * inv_s,
            torch.nextafter(orig0, torch.full_like(orig0, float("-inf"))),
        )

    use_pos = ((c0_f + step_fp8) <= FP8_E4M3_MAX) & torch.isfinite(pos_cand)
    repaired = torch.where(use_pos, pos_cand, neg_cand)
    child[:, 0] = torch.where(collapsed & torch.isfinite(repaired), repaired, orig0)


def fp8_assign(
    x: torch.Tensor,
    centroids: torch.Tensor,
    quantizer: FP8Quantizer | None = None,
    c_fp8_cached: tuple[torch.Tensor, torch.Tensor] | None = None,
    x_fp8_cached: torch.Tensor | None = None,
    row_indices: torch.Tensor | None = None,
    row_chunk_size: int = 262144,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign points using Blackwell FP8 E4M3 Tensor Cores and Appendix I.1 norm matching."""
    if FLOAT8_E4M3FN is None:
        raise RuntimeError("torch.float8_e4m3fn is not available in this PyTorch build")
    if quantizer is None:
        quantizer = FP8Quantizer.calibrate(x, center_mode="auto")
    if c_fp8_cached is not None:
        c_fp8, c_norm = c_fp8_cached
    else:
        c_fp8, c_norm = quantizer.prepare_centroids(centroids)

    total_b = int(row_indices.shape[0]) if row_indices is not None else int(x.shape[0])
    if total_b == 0:
        return (
            torch.empty((0,), dtype=torch.float32, device=x.device),
            torch.empty((0,), dtype=torch.int64, device=x.device),
        )

    if x_fp8_cached is not None:
        return blackwell_triton_assign(
            x_fp8_cached,
            c_fp8,
            c_norm,
            row_indices=row_indices,
            inv_scale_sq=quantizer.inv_scale_sq,
        )

    if total_b <= row_chunk_size:
        x_fp8 = quantizer.encode_rows(x, row_indices=row_indices)
        return blackwell_triton_assign(
            x_fp8,
            c_fp8,
            c_norm,
            inv_scale_sq=quantizer.inv_scale_sq,
        )

    dists = torch.empty((total_b,), dtype=torch.float32, device=x.device)
    labels = torch.empty((total_b,), dtype=torch.int64, device=x.device)
    buf_fp8 = torch.empty((row_chunk_size, quantizer.padded_d), dtype=FLOAT8_E4M3FN, device=x.device)
    for start in range(0, total_b, row_chunk_size):
        end = min(start + row_chunk_size, total_b)
        cur_b = end - start
        view_fp8 = buf_fp8[:cur_b]
        if row_indices is not None:
            quantizer.encode_rows(x, row_indices=row_indices[start:end], out=view_fp8)
        else:
            quantizer.encode_rows(x[start:end], out=view_fp8)
        d_c, l_c = blackwell_triton_assign(
            view_fp8,
            c_fp8,
            c_norm,
            inv_scale_sq=quantizer.inv_scale_sq,
        )
        dists[start:end] = d_c
        labels[start:end] = l_c
    return dists, labels


# ==============================================================================
# H-Opt-3a: Fused Triton Scatter-Stats & Fused EMA + Centroid Prep Kernels
# ==============================================================================

if HAS_TRITON:

    @triton.jit
    def _fused_scatter_stats_kernel(
        x_ptr,
        row_idx_ptr,
        labels_ptr,
        stats_ptr,
        B,
        D,
        stride_xb,
        stride_xd,
        stride_statsk,
        HAS_INDICES: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_b = tl.program_id(0).to(tl.int64)
        pid_d = tl.program_id(1).to(tl.int64)

        offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B).to(tl.int64)
        mask_b = offs_b < B
        lbl = tl.load(labels_ptr + offs_b, mask=mask_b, other=0).to(tl.int64)
        src_rows = tl.load(row_idx_ptr + offs_b, mask=mask_b, other=0).to(tl.int64) if HAS_INDICES else offs_b

        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D).to(tl.int64)
        mask_d = offs_d < D
        mask_bd = mask_b[:, None] & mask_d[None, :]

        vals = tl.load(
            x_ptr + src_rows[:, None] * stride_xb + offs_d[None, :] * stride_xd,
            mask=mask_bd,
            other=0.0,
        ).to(tl.float32)

        tl.atomic_add(
            stats_ptr + lbl[:, None] * stride_statsk + offs_d[None, :],
            vals,
            mask=mask_bd,
            sem="relaxed",
        )
        if pid_d == 0:
            tl.atomic_add(
                stats_ptr + lbl * stride_statsk + D,
                1.0,
                mask=mask_b,
                sem="relaxed",
            )

    @triton.jit
    def _fused_ema_prepare_kernel(
        centroids_ptr,
        usage_ptr,
        sums_ptr,
        counts_ptr,
        pass_counts_ptr,
        pass_sums_ptr,
        c_rep_ptr,
        c_norm_ptr,
        shift_ptr,
        K,
        D,
        stride_sumsk,
        stride_countsk,
        stride_crepk,
        alpha,
        decay,
        fp8_scale,
        DECAY: tl.constexpr,
        REP_MODE: tl.constexpr,  # 0=FP32, 1=BF16, 2=FP16, 3=FP8_E4M3
        HAS_SHIFT: tl.constexpr,
        ACCUM_PASS_COUNTS: tl.constexpr,
        ACCUM_PASS_SUMS: tl.constexpr,
        ZERO_STATS: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        j = tl.program_id(0).to(tl.int64)
        if j >= K:
            return

        cnt = tl.load(counts_ptr + j * stride_countsk).to(tl.float32)
        u_old = tl.load(usage_ptr + j).to(tl.float32)
        touched = cnt > 0.0

        retained = (u_old * decay) if DECAY else 0.0
        new_mass = retained + alpha * cnt
        safe_mass = tl.where(touched, new_mass, 1.0)
        safe_cnt = tl.where(touched, cnt, 1.0)
        weight = (alpha * cnt) / safe_mass

        if ACCUM_PASS_COUNTS:
            pc = tl.load(pass_counts_ptr + j)
            tl.store(pass_counts_ptr + j, pc + cnt.to(tl.float64))

        if ZERO_STATS:
            tl.store(counts_ptr + j * stride_countsk, 0.0)

        norm_acc = tl.zeros((), dtype=tl.float32)
        loop_extent = stride_crepk if REP_MODE == 3 else D

        for d0 in range(0, loop_extent, BLOCK_D):
            cols = d0 + tl.arange(0, BLOCK_D).to(tl.int64)
            mask_d = cols < D

            s_val = tl.load(sums_ptr + j * stride_sumsk + cols, mask=mask_d, other=0.0).to(tl.float32)
            if ACCUM_PASS_SUMS:
                ps_val = tl.load(pass_sums_ptr + j * D + cols, mask=mask_d, other=0.0)
                tl.store(pass_sums_ptr + j * D + cols, ps_val + s_val, mask=mask_d)
            if ZERO_STATS:
                # Synchronize all warps in CTA before zeroing `sums_ptr`, because `c_q8` (FP8)
                # induces a different warp/lane column layout on `s_val` than scalar `0.0` store!
                tl.debug_barrier()
                tl.store(sums_ptr + j * stride_sumsk + cols, 0.0, mask=mask_d)

            c_old = tl.load(centroids_ptr + j * D + cols, mask=mask_d, other=0.0).to(tl.float32)
            mean_val = s_val / safe_cnt
            pos_val = c_old + weight * (mean_val - c_old)
            upd_val = tl.where(weight == 1.0, mean_val, pos_val)
            c_new = tl.where(touched, upd_val, c_old)
            tl.store(centroids_ptr + j * D + cols, c_new, mask=mask_d & touched)

            if REP_MODE == 0:
                c_q = tl.where(mask_d, c_new, 0.0)
                tl.store(c_rep_ptr + j * stride_crepk + cols, c_q, mask=cols < stride_crepk)
                norm_acc += tl.sum(c_q * c_q, axis=0)
            elif REP_MODE == 1:
                c_q16 = tl.where(mask_d, c_new, 0.0).to(tl.bfloat16)
                tl.store(c_rep_ptr + j * stride_crepk + cols, c_q16, mask=cols < stride_crepk)
                c_qf = c_q16.to(tl.float32)
                norm_acc += tl.sum(c_qf * c_qf, axis=0)
            elif REP_MODE == 2:
                c_q16 = tl.where(mask_d, c_new, 0.0).to(tl.float16)
                tl.store(c_rep_ptr + j * stride_crepk + cols, c_q16, mask=cols < stride_crepk)
                c_qf = c_q16.to(tl.float32)
                norm_acc += tl.sum(c_qf * c_qf, axis=0)
            elif REP_MODE == 3:
                if HAS_SHIFT:
                    sh = tl.load(shift_ptr + cols, mask=mask_d, other=0.0).to(tl.float32)
                    scaled = (c_new - sh) * fp8_scale
                else:
                    scaled = c_new * fp8_scale
                clamped = tl.maximum(-448.0, tl.minimum(448.0, scaled))
                c_q8 = tl.where(mask_d, clamped, 0.0).to(tl.float8e4nv, fp_downcast_rounding="rtne")
                tl.store(c_rep_ptr + j * stride_crepk + cols, c_q8, mask=cols < stride_crepk)
                c_qf = c_q8.to(tl.float32)
                norm_acc += tl.sum(c_qf * c_qf, axis=0)

        tl.store(c_norm_ptr + j, norm_acc)
        new_usage = tl.where(touched, new_mass, retained if DECAY else u_old)
        tl.store(usage_ptr + j, new_usage)


def fused_scatter_stats(
    x: torch.Tensor,
    labels: torch.Tensor,
    packed_stats: torch.Tensor,
    row_indices: torch.Tensor | None = None,
) -> None:
    """Accumulate batch sums `packed_stats[:, :D]` and counts `packed_stats[:, D]` in place."""
    B = int(labels.shape[0])
    if B == 0:
        return
    D = int(x.shape[1])
    if HAS_TRITON and x.is_cuda:
        x_c = x if x.stride(1) == 1 else x.contiguous()
        has_idx = row_indices is not None
        idx_c = row_indices.contiguous() if has_idx else None
        block_b = 64
        block_d = min(128, max(32, triton.next_power_of_2(D)))
        grid = (triton.cdiv(B, block_b), triton.cdiv(D, block_d))
        with torch.cuda.device(x.device):
            _fused_scatter_stats_kernel[grid](
                x_c,
                idx_c,
                labels,
                packed_stats,
                B,
                D,
                x_c.stride(0),
                x_c.stride(1),
                packed_stats.stride(0),
                HAS_INDICES=has_idx,
                BLOCK_B=block_b,
                BLOCK_D=block_d,
                num_warps=4,
            )
        return

    rows = x[row_indices].float() if row_indices is not None else x.float()
    packed_stats[:, :D].index_add_(0, labels, rows)
    ones = torch.ones((B,), dtype=torch.float32, device=x.device)
    packed_stats[:, D].index_add_(0, labels, ones)


def fused_ema_and_prepare(
    centroids: torch.Tensor,
    usage: torch.Tensor,
    batch_sums: torch.Tensor,
    counts: torch.Tensor,
    alpha: float,
    compute_dtype: torch.dtype,
    c_rep_out: torch.Tensor | None = None,
    c_norm_out: torch.Tensor | None = None,
    pass_counts: torch.Tensor | None = None,
    pass_sums: torch.Tensor | None = None,
    fp8_quantizer: FP8Quantizer | None = None,
    zero_stats: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-kernel EMA update + pass accumulators + workspace zeroing + centroid prep."""
    K, D = centroids.shape
    device = centroids.device
    rep_mode = _dtype_mode(compute_dtype)
    rep_cols = fp8_quantizer.padded_d if (rep_mode == 3 and fp8_quantizer is not None) else D

    if c_rep_out is None:
        c_rep_out = torch.empty((K, rep_cols), dtype=compute_dtype, device=device)
    if c_norm_out is None:
        c_norm_out = torch.empty((K,), dtype=torch.float32, device=device)
    if K == 0:
        return c_rep_out, c_norm_out

    decay = 1.0 - float(alpha)
    do_decay = float(alpha) < 1.0

    if HAS_TRITON and centroids.is_cuda:
        has_shift = rep_mode == 3 and fp8_quantizer is not None and fp8_quantizer.shift is not None
        shift_ptr = fp8_quantizer.shift if has_shift else None
        fp8_scale = float(fp8_quantizer.scale) if (rep_mode == 3 and fp8_quantizer is not None) else 1.0
        block_d = min(512, max(64, triton.next_power_of_2(rep_cols)))
        with torch.cuda.device(device):
            _fused_ema_prepare_kernel[(K,)](
                centroids,
                usage,
                batch_sums,
                counts,
                pass_counts,
                pass_sums,
                c_rep_out,
                c_norm_out,
                shift_ptr,
                K,
                D,
                batch_sums.stride(0),
                counts.stride(0),
                c_rep_out.stride(0),
                float(alpha),
                float(decay),
                fp8_scale,
                DECAY=do_decay,
                REP_MODE=rep_mode,
                HAS_SHIFT=has_shift,
                ACCUM_PASS_COUNTS=pass_counts is not None,
                ACCUM_PASS_SUMS=pass_sums is not None,
                ZERO_STATS=zero_stats,
                BLOCK_D=block_d,
                num_warps=4,
            )
        return c_rep_out, c_norm_out

    # Exact CPU / PyTorch fallback
    cnt_f = counts.float()
    sums_f = batch_sums.float()
    if pass_counts is not None:
        pass_counts.add_(cnt_f.double())
    if pass_sums is not None:
        pass_sums.add_(sums_f)

    touched = cnt_f > 0
    if do_decay:
        usage.mul_(decay)
        retained = usage
    else:
        retained = torch.zeros_like(usage)

    new_mass = retained + float(alpha) * cnt_f
    safe_cnt = torch.where(touched, cnt_f, torch.ones_like(cnt_f))
    means = sums_f / safe_cnt[:, None]
    weight = (float(alpha) * cnt_f) / torch.where(touched, new_mass, torch.ones_like(new_mass))
    upd = centroids + weight[:, None] * (means - centroids)
    upd = torch.where((weight == 1.0)[:, None], means, upd)
    centroids.copy_(torch.where(touched[:, None], upd, centroids))
    usage.copy_(torch.where(touched, new_mass, usage))

    if zero_stats:
        batch_sums.zero_()
        counts.zero_()

    if rep_mode == 3 and fp8_quantizer is not None:
        return fp8_quantizer.prepare_centroids(centroids, c_fp8_out=c_rep_out, c_norm_out=c_norm_out)
    c_rep, c_norm = prepare_represented_centroids(centroids, compute_dtype)
    c_rep_out.copy_(c_rep)
    c_norm_out.copy_(c_norm)
    return c_rep_out, c_norm_out


# ==============================================================================
# H-Opt-3b: Fused Triton Greedy k-means++ Candidate Kernel + CUDA Graph Replay
# ==============================================================================

if HAS_TRITON:

    @triton.jit
    def _kpp_fused_candidates_kernel(
        x_rep_ptr,
        xf_ptr,
        x_norm_ptr,
        min_d2_ptr,
        w_ptr,
        cand_idx_ptr,
        cand_d2_ptr,
        partial_pot_ptr,
        M,
        D,
        L,
        stride_xm,
        stride_xd,
        stride_xfm,
        stride_xfd,
        stride_cd2m,
        BLOCK_M: tl.constexpr,
        BLOCK_L: tl.constexpr,
        BLOCK_D: tl.constexpr,
        USE_BF16: tl.constexpr,
    ):
        pid_m = tl.program_id(0).to(tl.int64)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
        mask_m = offs_m < M

        offs_l = tl.arange(0, BLOCK_L)
        mask_l = offs_l < L
        c_rows = tl.load(cand_idx_ptr + offs_l, mask=mask_l, other=0).to(tl.int64)

        dots = tl.zeros((BLOCK_M, BLOCK_L), dtype=tl.float32)
        c_norm = tl.zeros((BLOCK_L,), dtype=tl.float32)

        for d0 in range(0, D, BLOCK_D):
            offs_d = d0 + tl.arange(0, BLOCK_D).to(tl.int64)
            mask_d = offs_d < D
            x_tile = tl.load(
                x_rep_ptr + offs_m[:, None] * stride_xm + offs_d[None, :] * stride_xd,
                mask=mask_m[:, None] & mask_d[None, :],
                other=0.0,
            )
            c_tile_f = tl.load(
                xf_ptr + c_rows[:, None] * stride_xfm + offs_d[None, :] * stride_xfd,
                mask=mask_l[:, None] & mask_d[None, :],
                other=0.0,
            )
            if USE_BF16:
                c_tile = c_tile_f.to(tl.bfloat16)
                cf = c_tile.to(tl.float32)
                c_norm += tl.sum(cf * cf, axis=1)
                dots = tl.dot(x_tile, tl.trans(c_tile), acc=dots, out_dtype=tl.float32)
            else:
                c_norm += tl.sum(c_tile_f * c_tile_f, axis=1)
                dots = tl.dot(
                    x_tile,
                    tl.trans(c_tile_f),
                    acc=dots,
                    input_precision="ieee",
                    out_dtype=tl.float32,
                )

        xn = tl.load(x_norm_ptr + offs_m, mask=mask_m, other=0.0)
        cand_d2 = tl.maximum(xn[:, None] + c_norm[None, :] - 2.0 * dots, 0.0)
        tl.store(
            cand_d2_ptr + offs_m[:, None] * stride_cd2m + offs_l[None, :],
            cand_d2,
            mask=mask_m[:, None] & mask_l[None, :],
        )

        cur_min = tl.load(min_d2_ptr + offs_m, mask=mask_m, other=0.0)
        w_val = tl.load(w_ptr + offs_m, mask=mask_m, other=0.0)
        new_d2 = tl.minimum(cur_min[:, None], cand_d2)
        weighted = tl.where(mask_m[:, None] & mask_l[None, :], w_val[:, None] * new_d2, 0.0)
        part_pot = tl.sum(weighted, axis=0)
        tl.store(partial_pot_ptr + pid_m * BLOCK_L + offs_l, part_pot)

    @triton.jit
    def _kpp_select_update_kernel(
        xf_ptr,
        cand_idx_ptr,
        cand_d2_ptr,
        partial_pot_ptr,
        min_d2_ptr,
        w_ptr,
        weighted_d2_ptr,
        centroids_ptr,
        step_ptr,
        M,
        D,
        L,
        num_m_blocks,
        stride_xfm,
        stride_xfd,
        stride_cd2m,
        BLOCK_M: tl.constexpr,
        BLOCK_L: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0).to(tl.int64)
        offs_l = tl.arange(0, BLOCK_L)
        mask_l = offs_l < L

        # Deterministic reduction of partial potentials across M-blocks
        pot_acc = tl.zeros((BLOCK_L,), dtype=tl.float32)
        for b_idx in range(0, num_m_blocks):
            pot_acc += tl.load(partial_pot_ptr + b_idx * BLOCK_L + offs_l)
        pot_acc = tl.where(mask_l, pot_acc, float("inf"))
        _, best_l = tl.min(
            pot_acc,
            axis=0,
            return_indices=True,
            return_indices_tie_break_left=True,
        )

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
        mask_m = offs_m < M
        winner_row = tl.load(cand_idx_ptr + best_l).to(tl.int64)
        cur_min = tl.load(min_d2_ptr + offs_m, mask=mask_m, other=0.0)
        chosen_d2 = tl.load(cand_d2_ptr + offs_m * stride_cd2m + best_l, mask=mask_m, other=0.0)
        updated_d2 = tl.where(offs_m == winner_row, 0.0, tl.minimum(cur_min, chosen_d2))
        w_val = tl.load(w_ptr + offs_m, mask=mask_m, other=0.0)
        tl.store(min_d2_ptr + offs_m, updated_d2, mask=mask_m)
        tl.store(weighted_d2_ptr + offs_m, (w_val * updated_d2).to(tl.float64), mask=mask_m)

        if pid_m == 0:
            step_t = tl.load(step_ptr).to(tl.int64)
            for d0 in range(0, D, BLOCK_D):
                offs_d = d0 + tl.arange(0, BLOCK_D).to(tl.int64)
                mask_d = offs_d < D
                c_val = tl.load(
                    xf_ptr + winner_row * stride_xfm + offs_d * stride_xfd,
                    mask=mask_d,
                    other=0.0,
                )
                tl.store(centroids_ptr + step_t * D + offs_d, c_val, mask=mask_d)
            tl.store(step_ptr, (step_t + 1).to(tl.int32))


@torch.no_grad()
def triton_greedy_kmeans_plusplus(
    x: torch.Tensor,
    k: int,
    weights: torch.Tensor | None = None,
    n_local_trials: int | None = None,
    include_farthest_candidate: bool = False,
    use_bf16: bool | None = None,
    use_cuda_graph: bool = True,
    seed: int = 1234,
) -> torch.Tensor:
    """Fused Triton + CUDA-Graph Sync-Free Weighted Greedy k-means++ (H-Opt-3b)."""
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

    if not HAS_TRITON or not x.is_cuda or L > 32:
        from .seeding import greedy_kmeans_plusplus_torch

        return greedy_kmeans_plusplus_torch(
            xf,
            k,
            weights=w,
            n_local_trials=L,
            include_farthest_candidate=include_farthest_candidate,
            seed=seed,
        )

    origin = xf.mean(dim=0, keepdim=True)
    xc = (xf - origin).contiguous()

    if use_bf16 is None:
        use_bf16 = device.type == "cuda" and d >= 32 and m > 65536
    x_rep = xc.to(torch.bfloat16).contiguous() if use_bf16 else xc
    x_norm = x_rep.float().square().sum(dim=1).contiguous()

    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed) % (2**32))

    centroids = torch.empty((k, d), dtype=torch.float32, device=device)

    w_double = w.double().contiguous()
    w_cdf = torch.cumsum(w_double, dim=0)
    u0 = torch.rand((1,), generator=gen, device=device, dtype=torch.float64) * w_cdf[-1]
    idx0 = torch.searchsorted(w_cdf, u0, right=True).clamp_(0, m - 1)
    centroids[0] = xf[idx0].squeeze(0)
    if k == 1:
        return centroids

    c0_c = xc[idx0]
    c0_rep = c0_c.to(x_rep.dtype)
    c0_norm = c0_rep.float().square().sum(dim=1)
    dots0 = (x_rep @ c0_rep.t()).float().squeeze(1)
    min_d2 = (x_norm + c0_norm - 2.0 * dots0).clamp_min_(0.0).contiguous()
    min_d2[idx0] = 0.0
    weighted_d2 = (w * min_d2).double().contiguous()

    all_u = torch.rand((k - 1, L), generator=gen, device=device, dtype=torch.float64)
    use_far = bool(include_farthest_candidate and L >= 2)

    block_m = 128
    block_l = max(16, triton.next_power_of_2(L))
    block_d = 64 if use_bf16 else 32
    if d < block_d:
        block_d = max(16, triton.next_power_of_2(max(1, d)))
    num_m_blocks = triton.cdiv(m, block_m)

    cand_d2 = torch.empty((m, block_l), dtype=torch.float32, device=device)
    partial_pot = torch.empty((num_m_blocks, block_l), dtype=torch.float32, device=device)
    step_tensor = torch.tensor(1, dtype=torch.int32, device=device)
    u_row_buf = torch.empty((L,), dtype=torch.float64, device=device)
    cdf_buf = torch.empty((m,), dtype=torch.float64, device=device)
    cand_idx_buf = torch.empty((L,), dtype=torch.int64, device=device)

    def _run_one_step() -> None:
        idx_t = (step_tensor - 1).clamp_min(0).to(torch.int64)
        torch.index_select(all_u, 0, idx_t.view(1), out=u_row_buf.view(1, L))
        total_wd2 = weighted_d2.sum()
        eff_w = torch.where(total_wd2 > 0.0, weighted_d2, w_double)
        torch.cumsum(eff_w, dim=0, out=cdf_buf)
        u_scaled = u_row_buf * cdf_buf[-1]
        torch.searchsorted(cdf_buf, u_scaled, right=True, out=cand_idx_buf)
        cand_idx_buf.clamp_(0, m - 1)
        if use_far:
            cand_idx_buf[-1] = min_d2.argmax()

        _kpp_fused_candidates_kernel[(num_m_blocks,)](
            x_rep,
            xc,
            x_norm,
            min_d2,
            w,
            cand_idx_buf,
            cand_d2,
            partial_pot,
            m,
            d,
            L,
            x_rep.stride(0),
            x_rep.stride(1),
            xc.stride(0),
            xc.stride(1),
            cand_d2.stride(0),
            BLOCK_M=block_m,
            BLOCK_L=block_l,
            BLOCK_D=block_d,
            USE_BF16=use_bf16,
            num_warps=4,
            num_stages=2,
        )
        _kpp_select_update_kernel[(num_m_blocks,)](
            xf,
            cand_idx_buf,
            cand_d2,
            partial_pot,
            min_d2,
            w,
            weighted_d2,
            centroids,
            step_tensor,
            m,
            d,
            L,
            num_m_blocks,
            xf.stride(0),
            xf.stride(1),
            cand_d2.stride(0),
            BLOCK_M=block_m,
            BLOCK_L=block_l,
            BLOCK_D=min(256, max(32, triton.next_power_of_2(max(1, d)))),
            num_warps=4,
        )

    with torch.cuda.device(device):
        if use_cuda_graph and (k - 1) >= 8:
            # Prime 2 steps eagerly to compile Triton kernels before graph capture
            _run_one_step()
            _run_one_step()
            remaining = (k - 1) - 2
            if remaining > 0:
                graph = torch.cuda.CUDAGraph()
                cur_stream = torch.cuda.current_stream(device)
                cap_stream = torch.cuda.Stream(device=device)
                cap_stream.wait_stream(cur_stream)
                with torch.cuda.stream(cap_stream):
                    graph.capture_begin(capture_error_mode="thread_local")
                    try:
                        _run_one_step()
                    finally:
                        graph.capture_end()
                cur_stream.wait_stream(cap_stream)
                for _ in range(remaining):
                    graph.replay()
                graph.reset()
        else:
            for _ in range(1, k):
                _run_one_step()

    return centroids


@torch.no_grad()
def fast_seed_emeans_plusplus(
    x: torch.Tensor,
    k: int,
    seeding_mode: str = "auto",
    coreset_mult: int = 16,
    min_coreset_size: int = 8192,
    polish: bool = True,
    use_cuda_graph: bool = True,
    seed: int = 1234,
) -> SeedingResult:
    """Sub-epoch seeding pipeline accelerated by Fused Triton + CUDA-Graph Greedy k-means++."""
    n, d = x.shape
    device = x.device
    if n < k:
        raise ValueError(f"Number of points n={n} must be >= k={k}")
    m = min(n, max(min_coreset_size, coreset_mult * k, k * 2))
    x_seed, w_seed = lightweight_coreset(x, m=m, seed=seed)

    if seeding_mode == "coreset_exact" or (seeding_mode == "auto" and k <= 256):
        raw_c = triton_greedy_kmeans_plusplus(
            x_seed,
            k,
            weights=w_seed,
            include_farthest_candidate=True,
            use_cuda_graph=use_cuda_graph,
            seed=seed + 17,
        )
    elif seeding_mode == "coreset_block" or (seeding_mode == "auto" and k > 256):
        raw_c = block_greedy_kmeans_plusplus_torch(x_seed, k, weights=w_seed, seed=seed + 17)
    else:
        raise ValueError(f"Unknown seeding_mode={seeding_mode!r}")

    w_norm = w_seed.float()
    w_norm = w_norm / w_norm.sum().clamp_min(1e-30)
    comp_dtype = (
        torch.bfloat16 if (device.type == "cuda" and d >= 32 and x_seed.shape[0] > 65536) else torch.float32
    )
    if polish:
        return voronoi_polish_and_priors(x_seed, w_norm, raw_c, compute_dtype=comp_dtype)

    origin = x_seed.float().mean(dim=0, keepdim=True)
    x_seed_c = (x_seed.float() - origin).contiguous()
    raw_c_c = (raw_c.float() - origin).contiguous()
    c_rep, c_norm = blackwell_prepare_centroids(raw_c_c, comp_dtype)
    sq_d, labels = blackwell_triton_assign(x_seed_c, c_rep, c_norm)
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


# ==============================================================================
# H-Opt-4: Coarse-to-Fine IVF Meta-Centroid Pruned Assignment (Big-Batch Triton)
# ==============================================================================

if HAS_TRITON:

    @triton.jit
    def _ivf_inverted_list_kernel(
        x_ptr,
        row_idx_ptr,
        pair_perm_ptr,
        q_offsets_ptr,
        tile_prefix_ptr,
        c_sorted_ptr,
        cn_sorted_ptr,
        c_orig_id_ptr,
        c_offsets_ptr,
        cand_scores_ptr,
        cand_labels_ptr,
        D,
        K_ivf,
        nprobe,
        stride_xb,
        stride_xd,
        stride_ck,
        stride_cd,
        BLOCK_B: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
        FULL_D: tl.constexpr,
        DTYPE_MODE: tl.constexpr,
        HAS_INDICES: tl.constexpr,
    ):
        pid = tl.program_id(0)
        total_tiles = tl.load(tile_prefix_ptr + K_ivf - 1)
        if pid >= total_tiles:
            return

        # Binary search in [0, K_ivf) for smallest bucket b with tile_prefix[b] > pid
        lo = 0
        hi = K_ivf - 1
        for _ in tl.static_range(0, 20):
            mid = (lo + hi) // 2
            pref_mid = tl.load(tile_prefix_ptr + mid)
            hi = tl.where(pref_mid > pid, mid, hi)
            lo = tl.where(pref_mid > pid, lo, mid + 1)
        b = hi
        prev_tiles = tl.load(tile_prefix_ptr + b - 1, mask=b > 0, other=0)
        local_tile = pid - prev_tiles

        q_start = tl.load(q_offsets_ptr + b).to(tl.int64)
        q_end = tl.load(q_offsets_ptr + b + 1).to(tl.int64)
        pair_pos = q_start + local_tile.to(tl.int64) * BLOCK_B + tl.arange(0, BLOCK_B).to(tl.int64)
        mask_p = pair_pos < q_end

        orig_pair = tl.load(pair_perm_ptr + pair_pos, mask=mask_p, other=0).to(tl.int64)
        q_row = orig_pair // nprobe
        src_rows = tl.load(row_idx_ptr + q_row, mask=mask_p, other=0).to(tl.int64) if HAS_INDICES else q_row

        c_start = tl.load(c_offsets_ptr + b)
        c_end = tl.load(c_offsets_ptr + b + 1)

        best_score = tl.full((BLOCK_B,), float("inf"), dtype=tl.float32)
        best_sorted_k = tl.zeros((BLOCK_B,), dtype=tl.int32)

        if FULL_D:
            offs_d = tl.arange(0, BLOCK_D).to(tl.int64)
            mask_d = offs_d < D
            full_x = tl.load(
                x_ptr + src_rows[:, None] * stride_xb + offs_d[None, :] * stride_xd,
                mask=mask_p[:, None] & mask_d[None, :],
                other=0.0,
            )
            if DTYPE_MODE == 0:
                full_x = full_x.to(tl.float32)
            elif DTYPE_MODE == 1:
                full_x = full_x.to(tl.bfloat16)
            elif DTYPE_MODE == 2:
                full_x = full_x.to(tl.float16)

        k0 = c_start
        while k0 < c_end:
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < c_end

            if FULL_D:
                offs_d = tl.arange(0, BLOCK_D).to(tl.int64)
                mask_d = offs_d < D
                c_tile = tl.load(
                    c_sorted_ptr + offs_k[:, None].to(tl.int64) * stride_ck + offs_d[None, :] * stride_cd,
                    mask=mask_k[:, None] & mask_d[None, :],
                    other=0.0,
                )
                if DTYPE_MODE == 0:
                    dots = tl.dot(full_x, tl.trans(c_tile), input_precision="ieee", out_dtype=tl.float32)
                else:
                    dots = tl.dot(full_x, tl.trans(c_tile), out_dtype=tl.float32)
            else:
                dots = tl.zeros((BLOCK_B, BLOCK_K), dtype=tl.float32)
                for d0 in range(0, D, BLOCK_D):
                    offs_d = d0 + tl.arange(0, BLOCK_D).to(tl.int64)
                    mask_d = offs_d < D
                    x_tile = tl.load(
                        x_ptr + src_rows[:, None] * stride_xb + offs_d[None, :] * stride_xd,
                        mask=mask_p[:, None] & mask_d[None, :],
                        other=0.0,
                    )
                    if DTYPE_MODE == 0:
                        x_tile = x_tile.to(tl.float32)
                    elif DTYPE_MODE == 1:
                        x_tile = x_tile.to(tl.bfloat16)
                    elif DTYPE_MODE == 2:
                        x_tile = x_tile.to(tl.float16)
                    c_tile = tl.load(
                        c_sorted_ptr + offs_k[:, None].to(tl.int64) * stride_ck + offs_d[None, :] * stride_cd,
                        mask=mask_k[:, None] & mask_d[None, :],
                        other=0.0,
                    )
                    if DTYPE_MODE == 0:
                        dots = tl.dot(x_tile, tl.trans(c_tile), acc=dots, input_precision="ieee", out_dtype=tl.float32)
                    else:
                        dots = tl.dot(x_tile, tl.trans(c_tile), acc=dots, out_dtype=tl.float32)

            cn = tl.load(cn_sorted_ptr + offs_k, mask=mask_k, other=float("inf"))
            scores = cn[None, :] - 2.0 * dots
            scores = tl.where(mask_k[None, :], scores, float("inf"))

            t_min, t_idx = tl.min(
                scores,
                axis=1,
                return_indices=True,
                return_indices_tie_break_left=True,
            )
            better = t_min < best_score
            best_score = tl.where(better, t_min, best_score)
            best_sorted_k = tl.where(better, k0 + t_idx.to(tl.int32), best_sorted_k)
            k0 += BLOCK_K

        has_match = best_score < float("inf")
        orig_cid = tl.load(c_orig_id_ptr + best_sorted_k, mask=mask_p & has_match, other=0)
        tl.store(cand_scores_ptr + orig_pair, best_score, mask=mask_p)
        tl.store(cand_labels_ptr + orig_pair, orig_cid, mask=mask_p)


class IVFAssigner:
    """Coarse-to-Fine IVF Meta-Centroid Pruned Assigner (Appendix L.2, H-Opt-4)."""

    def __init__(
        self,
        meta_centroids: torch.Tensor,
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if compute_dtype not in (torch.float32, torch.bfloat16, torch.float16):
            raise ValueError(
                f"IVFAssigner compute_dtype must be float32, bfloat16, or float16; got {compute_dtype!r}"
            )
        self.meta_centroids = meta_centroids.float().contiguous()
        self.nlist = int(meta_centroids.shape[0])
        self.d = int(meta_centroids.shape[1])
        self.compute_dtype = compute_dtype
        self.meta_c_rep, self.meta_c_norm = blackwell_prepare_centroids(
            self.meta_centroids, compute_dtype
        )
        self.bucket_ids_arange = torch.arange(
            self.nlist + 1, dtype=torch.int64, device=meta_centroids.device
        )

        self.c_sorted: torch.Tensor | None = None
        self.cn_sorted: torch.Tensor | None = None
        self.c_orig_id: torch.Tensor | None = None
        self.c_offsets: torch.Tensor | None = None
        self.c_rep_ref: torch.Tensor | None = None
        self.c_norm_ref: torch.Tensor | None = None
        self.n_empty_buckets: int = 0

    @classmethod
    @torch.no_grad()
    def build(
        cls,
        x_train: torch.Tensor,
        k: int,
        nlist: int | None = None,
        compute_dtype: torch.dtype = torch.bfloat16,
        seed: int = 1234,
    ) -> IVFAssigner:
        """Train `nlist ~ sqrt(k)` meta-centroids ONCE on a lightweight coreset of `x_train`."""
        n, d = x_train.shape
        device = x_train.device
        if nlist is None:
            nlist = max(16, min(1024, round_power_of_two(math.sqrt(max(16, k)))))
        nlist = min(int(nlist), k, n)

        m_sub = min(n, max(8192, 64 * nlist))
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed + 31337) % (2**32))
        if m_sub < n:
            sub_idx = torch.randperm(n, generator=gen, device=device)[:m_sub]
            x_sub = x_train[sub_idx].float().contiguous()
        else:
            x_sub = x_train.float().contiguous()

        # 3 fast Lloyd iterations on x_sub to converge coarse meta-centroids
        perm0 = torch.randperm(x_sub.shape[0], generator=gen, device=device)[:nlist]
        meta_c = x_sub[perm0].clone().contiguous()
        sums = torch.empty((nlist, d), dtype=torch.float32, device=device)
        for _ in range(3):
            mc_rep, mc_norm = blackwell_prepare_centroids(meta_c, compute_dtype)
            _, lbl = blackwell_triton_assign(x_sub, mc_rep, mc_norm)
            cnt = torch.bincount(lbl, minlength=nlist).float()
            sums.zero_()
            sums.index_add_(0, lbl, x_sub)
            nonempty = cnt > 0
            meta_c = torch.where(
                nonempty[:, None],
                sums / cnt.clamp_min(1.0)[:, None],
                meta_c,
            )
        return cls(meta_c, compute_dtype=compute_dtype)

    @torch.no_grad()
    def rebuild_index(
        self,
        centroids: torch.Tensor,
        c_rep: torch.Tensor | None = None,
        c_norm: torch.Tensor | None = None,
    ) -> None:
        """Partition all K centroids into the fixed `nlist` IVF buckets on GPU."""
        if c_rep is None or c_norm is None:
            c_rep, c_norm = blackwell_prepare_centroids(centroids, self.compute_dtype)
        self.c_rep_ref = c_rep
        self.c_norm_ref = c_norm

        _, c_buckets = blackwell_triton_assign(centroids, self.meta_c_rep, self.meta_c_norm)
        perm = torch.argsort(c_buckets)
        sorted_b = c_buckets[perm]
        self.c_sorted = c_rep[perm].contiguous()
        self.cn_sorted = c_norm[perm].contiguous()
        self.c_orig_id = perm.to(torch.int32).contiguous()
        offsets = torch.searchsorted(sorted_b, self.bucket_ids_arange).to(torch.int32)
        self.c_offsets = offsets.contiguous()
        counts = offsets[1:] - offsets[:-1]
        self.n_empty_buckets = int((counts == 0).sum().item())

    @torch.no_grad()
    def assign(
        self,
        x: torch.Tensor,
        nprobe: int = 8,
        row_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Assign points via coarse meta-centroid top-`nprobe` + Triton inverted-list GEMMs."""
        if self.c_sorted is None or self.cn_sorted is None or self.c_offsets is None:
            raise RuntimeError("IVFAssigner.rebuild_index() must be called before assign()")

        B = int(row_indices.shape[0]) if row_indices is not None else int(x.shape[0])
        device = x.device
        if B == 0:
            return (
                torch.empty((0,), dtype=torch.float32, device=device),
                torch.empty((0,), dtype=torch.int64, device=device),
            )

        eff_nprobe = max(1, min(int(nprobe), self.nlist))
        if eff_nprobe >= self.nlist:
            return blackwell_triton_assign(
                x, self.c_rep_ref, self.c_norm_ref, row_indices=row_indices
            )

        x_sub = x[row_indices] if row_indices is not None else x
        x_rep = x_sub.to(self.compute_dtype).contiguous()
        x_norm = x_rep.float().square().sum(dim=1)

        if self.compute_dtype != torch.float32 and x_rep.is_cuda:
            coarse_scores = torch.addmm(
                self.meta_c_norm[None, :],
                x_rep,
                self.meta_c_rep.t(),
                alpha=-2.0,
                out_dtype=torch.float32,
            )
        else:
            coarse_scores = torch.addmm(
                self.meta_c_norm[None, :],
                x_rep.float(),
                self.meta_c_rep.float().t(),
                alpha=-2.0,
            )

        _, top_buckets = torch.topk(
            coarse_scores, k=eff_nprobe, dim=1, largest=False, sorted=False
        )
        flat_buckets = top_buckets.reshape(-1)
        m_pairs = B * eff_nprobe

        pair_perm = torch.argsort(flat_buckets)
        sorted_buckets = flat_buckets[pair_perm]
        q_offsets = torch.searchsorted(sorted_buckets, self.bucket_ids_arange).to(torch.int32)

        if HAS_TRITON and x.is_cuda:
            block_b = 64
            block_k = 128
            d_mode = _dtype_mode(self.compute_dtype)
            full_d = (self.d <= 128) if (d_mode != 0) else (self.d <= 64)
            block_d = (
                max(16, triton.next_power_of_2(max(1, self.d)))
                if full_d
                else (64 if d_mode != 0 else 32)
            )
            ivf_stages = 3
            ivf_smem = ((0 if full_d else block_b) + block_k) * block_d * x_rep.element_size()
            while ivf_stages > 1 and ivf_smem * ivf_stages > 81920:
                ivf_stages -= 1

            q_counts = q_offsets[1:] - q_offsets[:-1]
            tile_counts = (q_counts + block_b - 1) // block_b
            tile_prefix = torch.cumsum(tile_counts, dim=0).to(torch.int32)
            max_tiles = triton.cdiv(m_pairs, block_b) + self.nlist

            cand_scores = torch.full((m_pairs,), float("inf"), dtype=torch.float32, device=device)
            cand_labels = torch.zeros((m_pairs,), dtype=torch.int32, device=device)

            with torch.cuda.device(device):
                _ivf_inverted_list_kernel[(max_tiles,)](
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
            labels = labels_2d.gather(1, best_slot[:, None]).squeeze(1).to(torch.int64)
        else:
            # CPU / non-Triton fallback
            best_score = torch.full((B,), float("inf"), dtype=torch.float32, device=device)
            labels = torch.zeros((B,), dtype=torch.int64, device=device)
            for b_id in range(self.nlist):
                qs = int(q_offsets[b_id].item())
                qe = int(q_offsets[b_id + 1].item())
                cs = int(self.c_offsets[b_id].item())
                ce = int(self.c_offsets[b_id + 1].item())
                if qs == qe or cs == ce:
                    continue
                pairs_b = pair_perm[qs:qe]
                q_rows = pairs_b // eff_nprobe
                xb = x_rep[q_rows].float()
                cb = self.c_sorted[cs:ce].float()
                cnb = self.cn_sorted[cs:ce]
                sc = cnb[None, :] - 2.0 * (xb @ cb.t())
                b_min, b_idx = sc.min(dim=1)
                orig_ids = self.c_orig_id[cs + b_idx].to(torch.int64)
                for loc_i in range(q_rows.shape[0]):
                    qr = int(q_rows[loc_i].item())
                    if float(b_min[loc_i].item()) < float(best_score[qr].item()):
                        best_score[qr] = b_min[loc_i]
                        labels[qr] = orig_ids[loc_i]

        # Exact fallback if all `eff_nprobe` probed buckets for some row happened to be empty
        if self.n_empty_buckets >= eff_nprobe:
            unmatched = ~torch.isfinite(best_score)
            if bool(unmatched.any()):
                miss_idx = torch.nonzero(unmatched, as_tuple=False).squeeze(1)
                d_fb, l_fb = blackwell_triton_assign(x_rep[miss_idx], self.c_rep_ref, self.c_norm_ref)
                dists = (best_score + x_norm).clamp_min_(0.0)
                dists[miss_idx] = d_fb
                labels[miss_idx] = l_fb
                return dists, labels

        dists = (best_score + x_norm).clamp_min_(0.0)
        return dists, labels


# ==============================================================================
# H-Opt-5a & Unified Engine: FastEMeans (Single-GPU Composable Optimization Stack)
# ==============================================================================

def block_permute_indices(
    n: int,
    rows_needed: int,
    block_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Permute cache-aligned contiguous blocks of size `block_size` (Appendix L.1)."""
    if block_size <= 1 or n <= block_size:
        return torch.randperm(n, generator=generator, device=device)[:rows_needed]
    num_full_blocks = n // block_size
    rem = n - num_full_blocks * block_size
    total_blocks = num_full_blocks + (1 if rem > 0 else 0)
    b_perm = torch.randperm(total_blocks, generator=generator, device=device)

    # Vectorized expansion without host syncs
    grid = b_perm[:, None] * block_size + torch.arange(block_size, device=device)[None, :]
    flat = grid.reshape(-1)
    if rem > 0:
        flat = flat[flat < n]
    return flat[:rows_needed].contiguous()


class FastEMeans:
    """High-performance GPU e-means engine with toggleable H-Opt-1..5 optimizations."""

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
        precision: str = "bf16",  # 'bf16', 'fp8', 'fp32', 'fp16'
        fp8_center_mode: str = "auto",
        use_blackwell_assign: bool = True,
        use_fused_ema: bool = True,
        assign_mode: str = "exact",  # 'exact' or 'ivf'
        ivf_nlist: int | None = None,
        ivf_nprobe: int = 8,
        ivf_rebuild_period: int = 4,
        streaming_mode: str = "fused_gather",  # 'gather_copy', 'fused_gather', 'block_permute'
        block_permute_size: int = 64,
        preencode_dataset: bool = True,
        seed: int = 1234,
        verbose: bool = False,
    ) -> None:
        if finalization not in ("closing_mean", "none"):
            raise ValueError(f"finalization must be 'closing_mean' or 'none'; got {finalization!r}")
        if precision not in ("bf16", "fp8", "fp32", "fp16"):
            raise ValueError(f"Unsupported precision={precision!r}")
        if assign_mode not in ("exact", "ivf"):
            raise ValueError(f"Unsupported assign_mode={assign_mode!r}")
        if streaming_mode not in ("gather_copy", "fused_gather", "block_permute"):
            raise ValueError(f"Unsupported streaming_mode={streaming_mode!r}")

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
        self.precision = precision
        self.fp8_center_mode = fp8_center_mode
        self.use_blackwell_assign = bool(use_blackwell_assign)
        self.use_fused_ema = bool(use_fused_ema)
        self.assign_mode = assign_mode
        self.ivf_nlist = ivf_nlist
        self.ivf_nprobe = int(ivf_nprobe)
        self.ivf_rebuild_period = max(1, int(ivf_rebuild_period))
        self.streaming_mode = streaming_mode
        self.block_permute_size = int(block_permute_size)
        self.preencode_dataset = bool(preencode_dataset)
        self.seed = int(seed)
        self.verbose = bool(verbose)

        self.recipe: EMeansRecipe | None = None
        self.centroids: torch.Tensor | None = None
        self.assignments: torch.Tensor | None = None
        self.fp8_quantizer: FP8Quantizer | None = None
        self.ivf_assigner: IVFAssigner | None = None
        self.iteration_stats: list[EpochStat] = []
        self.n_splits: int = 0
        self.init_time_sec: float = 0.0
        self.train_time_sec: float = 0.0
        self.total_time_sec: float = 0.0

    def _resolve_dtype(self) -> torch.dtype:
        if self.precision == "bf16":
            return torch.bfloat16
        if self.precision == "fp16":
            return torch.float16
        if self.precision == "fp8":
            if FLOAT8_E4M3FN is None:
                raise RuntimeError("torch.float8_e4m3fn not supported")
            return FLOAT8_E4M3FN
        return torch.float32

    @torch.no_grad()
    def fit(
        self,
        x: torch.Tensor | np.ndarray,
        init_centroids: torch.Tensor | np.ndarray | None = None,
        prior_usage: torch.Tensor | None = None,
        capture_assignments: bool = True,
    ) -> tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray | None]:
        """Run optimized e-means clustering on (N, D) dataset."""
        input_is_numpy = isinstance(x, np.ndarray)
        x_t = (
            torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
            if input_is_numpy
            else x.detach()
        )
        if not x_t.is_contiguous():
            x_t = x_t.contiguous()
        n, d = x_t.shape
        if d != self.d:
            raise ValueError(f"Expected d={self.d}, got {d}")

        device = x_t.device
        comp_dtype = self._resolve_dtype() if device.type == "cuda" else torch.float32
        is_fp8 = comp_dtype == FLOAT8_E4M3FN

        recipe = build_recipe(
            n=n,
            d=d,
            k=self.k,
            nepochs=self.nepochs,
            batch_law=self.batch_law,
            batch_size=self.batch_size,
            alpha0=self.alpha0,
            alpha_min_factor=self.alpha_min_factor,
            unused_threshold=self.unused_threshold,
            check_unused_every=self.check_unused_every,
            split_epsilon=self.split_epsilon,
            init_usage=self.init_usage,
        )
        self.recipe = recipe

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_init0 = time.perf_counter()

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

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self.init_time_sec = time.perf_counter() - t_init0

        t_train0 = time.perf_counter()

        if is_fp8:
            self.fp8_quantizer = FP8Quantizer.calibrate(
                x_t, center_mode=self.fp8_center_mode, seed=self.seed
            )
        else:
            self.fp8_quantizer = None

        # Pre-encode low-precision query representation across passes when advantageous
        x_assign_all: torch.Tensor | None = None
        if self.preencode_dataset and device.type == "cuda" and self.nepochs >= 2.0:
            if is_fp8 and self.fp8_quantizer is not None:
                x_assign_all = self.fp8_quantizer.encode_rows(x_t)
            elif comp_dtype in (torch.bfloat16, torch.float16) and x_t.dtype != comp_dtype:
                x_assign_all = x_t.to(comp_dtype).contiguous()

        if self.assign_mode == "ivf":
            ivf_dt = torch.bfloat16 if is_fp8 else comp_dtype
            self.ivf_assigner = IVFAssigner.build(
                x_t, k=self.k, nlist=self.ivf_nlist, compute_dtype=ivf_dt, seed=self.seed
            )
        else:
            self.ivf_assigner = None

        if prior_usage is not None:
            usage = prior_usage.to(device=device, dtype=torch.float32).clone().contiguous()
        else:
            usage = torch.full((self.k,), recipe.init_usage, dtype=torch.float32, device=device)

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

        packed_stats = torch.zeros((self.k, self.d + 1), dtype=torch.float32, device=device)
        rep_cols = self.fp8_quantizer.padded_d if (is_fp8 and self.fp8_quantizer is not None) else self.d
        c_rep_buf = torch.empty((self.k, rep_cols), dtype=comp_dtype, device=device)
        c_norm_buf = torch.empty((self.k,), dtype=torch.float32, device=device)

        # Initial centroid preparation before step 0
        if is_fp8 and self.fp8_quantizer is not None:
            self.fp8_quantizer.prepare_centroids(centroids, c_fp8_out=c_rep_buf, c_norm_out=c_norm_buf)
        else:
            cr0, cn0 = blackwell_prepare_centroids(centroids, comp_dtype)
            c_rep_buf.copy_(cr0)
            c_norm_buf.copy_(cn0)

        if self.ivf_assigner is not None:
            self.ivf_assigner.rebuild_index(centroids)

        step = 0
        self.n_splits = 0
        self.iteration_stats = []
        num_epochs = len(recipe.epoch_rows)

        for ep_idx, rows_in_ep in enumerate(recipe.epoch_rows):
            is_last_epoch = ep_idx == (num_epochs - 1)
            gen_ep = torch.Generator(device=device)
            gen_ep.manual_seed(epoch_seed(self.seed, ep_idx))

            if self.streaming_mode == "block_permute":
                order = block_permute_indices(
                    n, rows_in_ep, self.block_permute_size, gen_ep, device
                )
            else:
                order = torch.randperm(n, generator=gen_ep, device=device)[:rows_in_ep]

            pass_counts = torch.zeros((self.k,), dtype=torch.float64, device=device)
            pass_sums = (
                torch.zeros((self.k, self.d), dtype=torch.float32, device=device)
                if (is_last_epoch and self.finalization == "closing_mean")
                else None
            )
            ep_sq_dist_gpu = torch.zeros((), dtype=torch.float64, device=device)
            splits_before = self.n_splits
            ep_steps = 0

            for b_start in range(0, rows_in_ep, recipe.batch_size):
                b_end = min(b_start + recipe.batch_size, rows_in_ep)
                batch_idx = order[b_start:b_end]

                # 1. Nearest-centroid assignment
                use_indexed = self.streaming_mode != "gather_copy"
                if self.ivf_assigner is not None:
                    if step > 0 and (step % self.ivf_rebuild_period == 0):
                        self.ivf_assigner.rebuild_index(centroids)
                    sq_dists, labels = self.ivf_assigner.assign(
                        x_t if use_indexed else x_t[batch_idx],
                        nprobe=self.ivf_nprobe,
                        row_indices=batch_idx if use_indexed else None,
                    )
                elif is_fp8 and self.fp8_quantizer is not None:
                    if x_assign_all is not None:
                        sq_dists, labels = fp8_assign(
                            x_t,
                            centroids,
                            quantizer=self.fp8_quantizer,
                            c_fp8_cached=(c_rep_buf, c_norm_buf),
                            x_fp8_cached=x_assign_all if use_indexed else x_assign_all[batch_idx],
                            row_indices=batch_idx if use_indexed else None,
                        )
                    else:
                        sq_dists, labels = fp8_assign(
                            x_t if use_indexed else x_t[batch_idx],
                            centroids,
                            quantizer=self.fp8_quantizer,
                            c_fp8_cached=(c_rep_buf, c_norm_buf),
                            row_indices=batch_idx if use_indexed else None,
                        )
                elif self.use_blackwell_assign:
                    x_src = x_assign_all if x_assign_all is not None else x_t
                    sq_dists, labels = blackwell_triton_assign(
                        x_src if use_indexed else x_src[batch_idx],
                        c_rep_buf,
                        c_norm_buf,
                        row_indices=batch_idx if use_indexed else None,
                    )
                else:
                    xb = x_t[batch_idx]
                    sq_dists, labels = assign_points(
                        xb,
                        centroids,
                        compute_dtype=comp_dtype,
                        backend="triton",
                        c_rep_cached=(c_rep_buf, c_norm_buf),
                    )

                if is_last_epoch and assignments is not None:
                    assignments[batch_idx] = labels

                # Device-side FP64 objective accumulation (ZERO per-step .item() sync!)
                ep_sq_dist_gpu.add_(sq_dists.sum(dtype=torch.float64))

                alpha = scheduled_alpha(
                    recipe.alpha0, step, recipe.total_steps, recipe.alpha_min_factor
                )

                # 2. Batch accumulation + EMA update + next-step centroid preparation
                if self.use_fused_ema:
                    if use_indexed:
                        fused_scatter_stats(x_t, labels, packed_stats, row_indices=batch_idx)
                    else:
                        fused_scatter_stats(x_t[batch_idx], labels, packed_stats, row_indices=None)

                    fused_ema_and_prepare(
                        centroids=centroids,
                        usage=usage,
                        batch_sums=packed_stats[:, : self.d],
                        counts=packed_stats[:, self.d],
                        alpha=alpha,
                        compute_dtype=comp_dtype,
                        c_rep_out=c_rep_buf,
                        c_norm_out=c_norm_buf,
                        pass_counts=pass_counts,
                        pass_sums=pass_sums,
                        fp8_quantizer=self.fp8_quantizer,
                        zero_stats=True,
                    )
                else:
                    xb_f = x_t[batch_idx].float()
                    counts = torch.bincount(labels, minlength=self.k).float()
                    b_sums = packed_stats[:, : self.d]
                    b_sums.zero_()
                    b_sums.index_add_(0, labels, xb_f)
                    pass_counts.add_(counts.double())
                    if pass_sums is not None:
                        pass_sums.add_(b_sums)

                    touched = counts > 0
                    if alpha < 1.0:
                        usage.mul_(1.0 - alpha)
                        retained = usage
                    else:
                        retained = torch.zeros_like(usage)
                    new_mass = retained + alpha * counts
                    safe_cnt = torch.where(touched, counts, torch.ones_like(counts))
                    means = b_sums / safe_cnt[:, None]
                    weight = (alpha * counts) / torch.where(touched, new_mass, torch.ones_like(new_mass))
                    upd = centroids + weight[:, None] * (means - centroids)
                    upd = torch.where((weight == 1.0)[:, None], means, upd)
                    centroids = torch.where(touched[:, None], upd, centroids)
                    usage = torch.where(touched, new_mass, usage)
                    if is_fp8 and self.fp8_quantizer is not None:
                        self.fp8_quantizer.prepare_centroids(
                            centroids, c_fp8_out=c_rep_buf, c_norm_out=c_norm_buf
                        )
                    else:
                        cr, cn = blackwell_prepare_centroids(centroids, comp_dtype)
                        c_rep_buf.copy_(cr)
                        c_norm_buf.copy_(cn)

                # 3. Dead-cluster revival every `check_unused_every` steps
                if (
                    recipe.unused_threshold > 0.0
                    and ((step + 1) % recipe.check_unused_every == 0)
                ):
                    thresh = usage.mean() * recipe.unused_threshold
                    unused_mask = usage < thresh
                    donor_mask = (~unused_mask) & (usage > 0)
                    # Single 2-int transfer instead of two separate syncs
                    n_unused, n_donors = torch.stack(
                        (unused_mask.sum(), donor_mask.sum())
                    ).tolist()
                    nsplit = min(int(n_unused), int(n_donors))
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
                        if is_fp8 and self.fp8_quantizer is not None:
                            _separate_collapsed_splits_fp8(d_cent, c_cent, self.fp8_quantizer)
                        else:
                            _separate_collapsed_splits_branchless(d_cent, c_cent, comp_dtype)
                        half_u = usage[source] * 0.5
                        centroids[source] = d_cent
                        centroids[target] = c_cent
                        usage[source] = half_u
                        usage[target] = half_u
                        self.n_splits += nsplit

                        # Refresh prepared centroids after split mutation
                        if is_fp8 and self.fp8_quantizer is not None:
                            self.fp8_quantizer.prepare_centroids(
                                centroids, c_fp8_out=c_rep_buf, c_norm_out=c_norm_buf
                            )
                        else:
                            cr, cn = blackwell_prepare_centroids(centroids, comp_dtype)
                            c_rep_buf.copy_(cr)
                            c_norm_buf.copy_(cn)
                        if self.ivf_assigner is not None:
                            self.ivf_assigner.rebuild_index(centroids)

                step += 1
                ep_steps += 1

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            ep_sq_dist = float(ep_sq_dist_gpu.item())
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
                elapsed_sec=time.perf_counter() - t_train0,
            )
            self.iteration_stats.append(stat)

            if is_last_epoch and self.finalization == "closing_mean" and pass_sums is not None:
                nonempty = pass_counts > 0
                centroids = torch.where(
                    nonempty[:, None],
                    pass_sums / pass_counts.clamp_min(1.0).float()[:, None],
                    centroids,
                )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self.train_time_sec = time.perf_counter() - t_train0
        self.total_time_sec = self.init_time_sec + self.train_time_sec
        self.centroids = centroids.contiguous()
        self.assignments = assignments

        if input_is_numpy:
            return (
                self.centroids.cpu().numpy(),
                self.assignments.cpu().numpy() if self.assignments is not None else None,
            )
        return self.centroids, self.assignments

    @torch.no_grad()
    def assign(self, x: torch.Tensor | np.ndarray) -> tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray]:
        if self.centroids is None:
            raise RuntimeError("FastEMeans has not been fitted yet.")
        input_is_numpy = isinstance(x, np.ndarray)
        dev = self.centroids.device
        x_t = (
            torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(dev)
            if input_is_numpy
            else x.to(dev)
        )
        comp_dtype = self._resolve_dtype() if dev.type == "cuda" else torch.float32
        if comp_dtype == FLOAT8_E4M3FN:
            dists, labels = fp8_assign(x_t, self.centroids, quantizer=self.fp8_quantizer)
        else:
            c_rep, c_norm = blackwell_prepare_centroids(self.centroids, comp_dtype)
            dists, labels = blackwell_triton_assign(x_t, c_rep, c_norm)
        if input_is_numpy:
            return dists.cpu().numpy(), labels.cpu().numpy()
        return dists, labels


# ==============================================================================
# H-Opt-5b: Multi-GPU Data-Sharded NCCL All-Reduce Engine (`MultiGPUFastEMeans`)
# ==============================================================================

class MultiGPUFastEMeans:
    """Single-process multi-GPU data-sharded e-means with hardware NCCL all-reduce (Appendix L.1)."""

    def __init__(
        self,
        d: int,
        k: int,
        devices: list[int | str | torch.device] | None = None,
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
        precision: str = "bf16",
        fp8_center_mode: str = "auto",
        streaming_mode: str = "fused_gather",
        block_permute_size: int = 64,
        preencode_dataset: bool = True,
        seed: int = 1234,
    ) -> None:
        if finalization not in ("closing_mean", "none"):
            raise ValueError(f"finalization must be 'closing_mean' or 'none'; got {finalization!r}")
        if precision not in ("bf16", "fp8", "fp32", "fp16"):
            raise ValueError(f"Unsupported precision={precision!r}")
        if streaming_mode not in ("gather_copy", "fused_gather", "block_permute"):
            raise ValueError(f"Unsupported streaming_mode={streaming_mode!r}")

        self.d = int(d)
        self.k = int(k)
        if devices is None:
            ndev = torch.cuda.device_count()
            self.devices = [torch.device(f"cuda:{i}") for i in range(ndev)]
        else:
            self.devices = [torch.device(dv) for dv in devices]
        if len(self.devices) < 1:
            raise ValueError("MultiGPUFastEMeans requires at least 1 CUDA device")

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
        self.precision = precision
        self.fp8_center_mode = fp8_center_mode
        self.streaming_mode = streaming_mode
        self.block_permute_size = int(block_permute_size)
        self.preencode_dataset = bool(preencode_dataset)
        self.seed = int(seed)

        self.recipe: EMeansRecipe | None = None
        self.centroids: torch.Tensor | None = None
        self.centroids_per_gpu: list[torch.Tensor] = []
        self.iteration_stats: list[EpochStat] = []
        self.n_splits: int = 0
        self.shard_transfer_time_sec: float = 0.0
        self.init_time_sec: float = 0.0
        self.train_time_sec: float = 0.0
        self.total_time_sec: float = 0.0

    def _resolve_dtype(self) -> torch.dtype:
        if self.precision == "bf16":
            return torch.bfloat16
        if self.precision == "fp16":
            return torch.float16
        if self.precision == "fp8":
            return FLOAT8_E4M3FN
        return torch.float32

    @torch.no_grad()
    def prepopulate_shards(
        self,
        x: torch.Tensor | np.ndarray,
    ) -> list[torch.Tensor]:
        """Distribute dataset evenly (strided modulo W) across GPUs so each shard has exact balance."""
        x_t = (
            torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
            if isinstance(x, np.ndarray)
            else x.detach().float()
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
    ) -> torch.Tensor:
        """Fit data-sharded e-means across `self.devices` using concurrent streams + NCCL all-reduce."""
        W = len(self.devices)
        comp_dtype = self._resolve_dtype()
        is_fp8 = comp_dtype == FLOAT8_E4M3FN

        if isinstance(x, list):
            shards = [s.contiguous() for s in x]
            n = sum(int(s.shape[0]) for s in shards)
            d = int(shards[0].shape[1])
        else:
            shards = self.prepopulate_shards(x)
            n = sum(int(s.shape[0]) for s in shards)
            d = int(shards[0].shape[1])

        if d != self.d:
            raise ValueError(f"Expected d={self.d}, got {d}")

        if W == 1:
            single = FastEMeans(
                d=self.d,
                k=self.k,
                nepochs=self.nepochs,
                batch_law=self.batch_law,
                batch_size=self.batch_size,
                alpha0=self.alpha0,
                alpha_min_factor=self.alpha_min_factor,
                unused_threshold=self.unused_threshold,
                check_unused_every=self.check_unused_every,
                split_epsilon=self.split_epsilon,
                init_usage=self.init_usage,
                finalization=self.finalization,
                precision=self.precision,
                fp8_center_mode=self.fp8_center_mode,
                streaming_mode=self.streaming_mode,
                block_permute_size=self.block_permute_size,
                preencode_dataset=self.preencode_dataset,
                seed=self.seed,
            )
            c_out, _ = single.fit(shards[0], init_centroids=init_centroids, capture_assignments=False)
            self.recipe = single.recipe
            self.centroids = c_out
            self.centroids_per_gpu = [c_out]
            self.iteration_stats = single.iteration_stats
            self.n_splits = single.n_splits
            self.init_time_sec = single.init_time_sec
            self.train_time_sec = single.train_time_sec
            self.total_time_sec = single.total_time_sec
            return c_out

        recipe = build_recipe(
            n=n,
            d=d,
            k=self.k,
            nepochs=self.nepochs,
            batch_law=self.batch_law,
            batch_size=self.batch_size,
            alpha0=self.alpha0,
            alpha_min_factor=self.alpha_min_factor,
            unused_threshold=self.unused_threshold,
            check_unused_every=self.check_unused_every,
            split_epsilon=self.split_epsilon,
            init_usage=self.init_usage,
        )
        self.recipe = recipe

        for dv in self.devices:
            torch.cuda.synchronize(dv)
        t_init0 = time.perf_counter()

        # Initial centroids generated deterministically and replicated identically to all GPUs
        dev0 = self.devices[0]
        if init_centroids is None:
            # Draw proportionally across shards
            per_w = [(self.k // W) + (1 if w < (self.k % W) else 0) for w in range(W)]
            parts = []
            for w, kw in enumerate(per_w):
                if kw > 0:
                    gw = torch.Generator(device=self.devices[w])
                    gw.manual_seed((self.seed + 97 * w) % (2**32))
                    pw = torch.randperm(shards[w].shape[0], generator=gw, device=self.devices[w])[:kw]
                    parts.append(shards[w][pw].to(dev0))
            c0_master = torch.cat(parts, dim=0).float().contiguous()
        else:
            c_in = (
                torch.from_numpy(np.asarray(init_centroids, dtype=np.float32))
                if isinstance(init_centroids, np.ndarray)
                else init_centroids
            )
            c0_master = c_in.to(device=dev0, dtype=torch.float32).clone().contiguous()

        for dv in self.devices:
            torch.cuda.synchronize(dv)
        self.init_time_sec = time.perf_counter() - t_init0
        t_train0 = time.perf_counter()

        streams = [torch.cuda.Stream(device=dv) for dv in self.devices]

        # Calibrate FP8 quantizer once on shard 0 and share identical scale/shift across all GPUs
        quantizers: list[FP8Quantizer | None] = [None] * W
        if is_fp8:
            q0 = FP8Quantizer.calibrate(shards[0], center_mode=self.fp8_center_mode, seed=self.seed)
            for w, dv in enumerate(self.devices):
                sh_w = q0.shift.to(dv).contiguous() if q0.shift is not None else None
                quantizers[w] = FP8Quantizer(
                    d=self.d,
                    scale=q0.scale,
                    exponent=q0.exponent,
                    shift=sh_w,
                    padded_d=q0.padded_d,
                )

        rep_cols = quantizers[0].padded_d if (is_fp8 and quantizers[0] is not None) else self.d

        centroids_w = [c0_master.to(dv).clone().contiguous() for dv in self.devices]
        usage_w = [
            torch.full((self.k,), recipe.init_usage, dtype=torch.float32, device=dv)
            for dv in self.devices
        ]
        stats_w = [
            torch.zeros((self.k, self.d + 1), dtype=torch.float32, device=dv)
            for dv in self.devices
        ]
        c_rep_w = [
            torch.empty((self.k, rep_cols), dtype=comp_dtype, device=dv)
            for dv in self.devices
        ]
        c_norm_w = [
            torch.empty((self.k,), dtype=torch.float32, device=dv)
            for dv in self.devices
        ]
        split_offset_w = []
        for dv in self.devices:
            sgn = torch.ones((self.d,), dtype=torch.float32, device=dv)
            sgn[1::2] = -1.0
            split_offset_w.append(recipe.split_epsilon * sgn)

        x_assign_w: list[torch.Tensor | None] = [None] * W
        for w, dv in enumerate(self.devices):
            streams[w].wait_stream(torch.cuda.current_stream(dv))
            with torch.cuda.device(dv), torch.cuda.stream(streams[w]):
                if self.preencode_dataset and self.nepochs >= 2.0:
                    if is_fp8 and quantizers[w] is not None:
                        x_assign_w[w] = quantizers[w].encode_rows(shards[w])
                    elif comp_dtype in (torch.bfloat16, torch.float16):
                        x_assign_w[w] = shards[w].to(comp_dtype).contiguous()
                if is_fp8 and quantizers[w] is not None:
                    quantizers[w].prepare_centroids(
                        centroids_w[w], c_fp8_out=c_rep_w[w], c_norm_out=c_norm_w[w]
                    )
                else:
                    cr, cn = blackwell_prepare_centroids(centroids_w[w], comp_dtype)
                    c_rep_w[w].copy_(cr)
                    c_norm_w[w].copy_(cn)

        split_gen0 = torch.Generator(device=dev0)
        split_gen0.manual_seed((self.seed + 7919) % (2**32))
        u_w = [torch.empty((self.k,), dtype=torch.float32, device=dv) for dv in self.devices]

        step = 0
        self.n_splits = 0
        self.iteration_stats = []
        num_epochs = len(recipe.epoch_rows)

        for ep_idx, rows_in_ep in enumerate(recipe.epoch_rows):
            is_last_epoch = ep_idx == (num_epochs - 1)
            orders_w = []
            pass_counts_w = []
            pass_sums_w: list[torch.Tensor | None] = []
            ep_sq_w = []

            for w, dv in enumerate(self.devices):
                with torch.cuda.device(dv), torch.cuda.stream(streams[w]):
                    rows_w = max(0, (rows_in_ep - w + W - 1) // W)
                    nw = int(shards[w].shape[0])
                    gw = torch.Generator(device=dv)
                    gw.manual_seed(epoch_seed(self.seed + 104729 * w, ep_idx))
                    if self.streaming_mode == "block_permute":
                        ord_w = block_permute_indices(nw, rows_w, self.block_permute_size, gw, dv)
                    else:
                        ord_w = torch.randperm(nw, generator=gw, device=dv)[:rows_w]
                    orders_w.append(ord_w)
                    pass_counts_w.append(torch.zeros((self.k,), dtype=torch.float64, device=dv))
                    pass_sums_w.append(
                        torch.zeros((self.k, self.d), dtype=torch.float32, device=dv)
                        if (is_last_epoch and self.finalization == "closing_mean")
                        else None
                    )
                    ep_sq_w.append(torch.zeros((), dtype=torch.float64, device=dv))

            splits_before = self.n_splits
            ep_steps = 0

            for g0 in range(0, rows_in_ep, recipe.batch_size):
                g1 = min(g0 + recipe.batch_size, rows_in_ep)
                alpha = scheduled_alpha(
                    recipe.alpha0, step, recipe.total_steps, recipe.alpha_min_factor
                )

                # 1. Parallel local assignment + local scatter-stats on each GPU stream
                for w, dv in enumerate(self.devices):
                    with torch.cuda.device(dv), torch.cuda.stream(streams[w]):
                        l0 = max(0, (g0 - w + W - 1) // W)
                        l1 = max(0, (g1 - w + W - 1) // W)
                        if l1 > l0:
                            b_idx_w = orders_w[w][l0:l1]
                            if is_fp8 and quantizers[w] is not None:
                                sq_d, lbl = fp8_assign(
                                    shards[w],
                                    centroids_w[w],
                                    quantizer=quantizers[w],
                                    c_fp8_cached=(c_rep_w[w], c_norm_w[w]),
                                    x_fp8_cached=x_assign_w[w],
                                    row_indices=b_idx_w,
                                )
                            else:
                                x_src_w = x_assign_w[w] if x_assign_w[w] is not None else shards[w]
                                sq_d, lbl = blackwell_triton_assign(
                                    x_src_w, c_rep_w[w], c_norm_w[w], row_indices=b_idx_w
                                )
                            ep_sq_w[w].add_(sq_d.sum(dtype=torch.float64))
                            fused_scatter_stats(
                                shards[w], lbl, stats_w[w], row_indices=b_idx_w
                            )

                # 2. Single coalesced hardware NCCL all-reduce of packed (sums, counts)
                torch.cuda.nccl.all_reduce(stats_w, streams=streams)

                # 3. Deterministic replica-identical fused EMA + centroid preparation
                for w, dv in enumerate(self.devices):
                    with torch.cuda.device(dv), torch.cuda.stream(streams[w]):
                        fused_ema_and_prepare(
                            centroids=centroids_w[w],
                            usage=usage_w[w],
                            batch_sums=stats_w[w][:, : self.d],
                            counts=stats_w[w][:, self.d],
                            alpha=alpha,
                            compute_dtype=comp_dtype,
                            c_rep_out=c_rep_w[w],
                            c_norm_out=c_norm_w[w],
                            pass_counts=pass_counts_w[w],
                            pass_sums=pass_sums_w[w],
                            fp8_quantizer=quantizers[w],
                            zero_stats=True,
                        )

                # 4. Replica-identical dead-cluster revival every `check_unused_every` steps
                if (
                    recipe.unused_threshold > 0.0
                    and ((step + 1) % recipe.check_unused_every == 0)
                ):
                    with torch.cuda.device(dev0), torch.cuda.stream(streams[0]):
                        thresh0 = usage_w[0].mean() * recipe.unused_threshold
                        unused0 = usage_w[0] < thresh0
                        donor0 = (~unused0) & (usage_w[0] > 0)
                        cnt_pair = torch.stack((unused0.sum(), donor0.sum()))
                    streams[0].synchronize()
                    n_unused, n_donors = cnt_pair.tolist()
                    nsplit = min(int(n_unused), int(n_donors))
                    if nsplit > 0:
                        with torch.cuda.device(dev0), torch.cuda.stream(streams[0]):
                            u_w[0].uniform_(0.0, 1.0, generator=split_gen0)
                            u_w[0].clamp_min_(torch.finfo(torch.float32).tiny)
                        # Broadcast tiny (K,) uniform vector via NCCL so all GPUs split identically
                        torch.cuda.nccl.broadcast(u_w, root=0, streams=streams)

                        for w, dv in enumerate(self.devices):
                            with torch.cuda.device(dv), torch.cuda.stream(streams[w]):
                                tw = usage_w[w].mean() * recipe.unused_threshold
                                um = usage_w[w] < tw
                                dm = (~um) & (usage_w[w] > 0)
                                keys = torch.where(
                                    dm,
                                    usage_w[w].log() - (-u_w[w].log()).log(),
                                    torch.full_like(usage_w[w], float("-inf")),
                                )
                                src = torch.topk(keys, nsplit).indices
                                tgt = torch.nonzero(um, as_tuple=False).squeeze(1)[:nsplit]
                                donors = centroids_w[w][src]
                                dc = donors * (1.0 - split_offset_w[w])
                                cc = donors * (1.0 + split_offset_w[w])
                                if is_fp8 and quantizers[w] is not None:
                                    _separate_collapsed_splits_fp8(dc, cc, quantizers[w])
                                else:
                                    _separate_collapsed_splits_branchless(dc, cc, comp_dtype)
                                hu = usage_w[w][src] * 0.5
                                centroids_w[w][src] = dc
                                centroids_w[w][tgt] = cc
                                usage_w[w][src] = hu
                                usage_w[w][tgt] = hu
                                if is_fp8 and quantizers[w] is not None:
                                    quantizers[w].prepare_centroids(
                                        centroids_w[w], c_fp8_out=c_rep_w[w], c_norm_out=c_norm_w[w]
                                    )
                                else:
                                    cr, cn = blackwell_prepare_centroids(centroids_w[w], comp_dtype)
                                    c_rep_w[w].copy_(cr)
                                    c_norm_w[w].copy_(cn)
                        self.n_splits += nsplit

                step += 1
                ep_steps += 1

            # Reduce epoch objective across shards
            torch.cuda.nccl.all_reduce(ep_sq_w, streams=streams)
            for w, dv in enumerate(self.devices):
                streams[w].synchronize()
                torch.cuda.current_stream(dv).wait_stream(streams[w])

            # Note: pass_counts_w[0] already summed ALL global counts because stats_w was all-reduced!
            ep_sq_total = float(ep_sq_w[0].item())
            imbalance = float(
                (self.k * torch.dot(pass_counts_w[0], pass_counts_w[0]) / (float(rows_in_ep) ** 2)).item()
            )
            stat = EpochStat(
                epoch=ep_idx + 1,
                rows=rows_in_ep,
                steps=ep_steps,
                moving_mse=ep_sq_total / max(1, rows_in_ep),
                imbalance_factor=imbalance,
                n_splits=self.n_splits - splits_before,
                elapsed_sec=time.perf_counter() - t_train0,
            )
            self.iteration_stats.append(stat)

            if is_last_epoch and self.finalization == "closing_mean":
                for w, dv in enumerate(self.devices):
                    with torch.cuda.device(dv), torch.cuda.stream(streams[w]):
                        ps_w = pass_sums_w[w]
                        if ps_w is not None:
                            nonempty = pass_counts_w[w] > 0
                            centroids_w[w] = torch.where(
                                nonempty[:, None],
                                ps_w / pass_counts_w[w].clamp_min(1.0).float()[:, None],
                                centroids_w[w],
                            ).contiguous()
                for w, dv in enumerate(self.devices):
                    streams[w].synchronize()
                    torch.cuda.current_stream(dv).wait_stream(streams[w])

        for dv in self.devices:
            torch.cuda.synchronize(dv)
        self.train_time_sec = time.perf_counter() - t_train0
        self.total_time_sec = self.init_time_sec + self.train_time_sec
        self.centroids_per_gpu = centroids_w
        self.centroids = centroids_w[0]
        return self.centroids
