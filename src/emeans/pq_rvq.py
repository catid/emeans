"""Fused Multi-Subspace Product Quantization (`ApexPQ`) & 2-Stage Residual Vector Quantization (`ApexRVQ`).

Demonstrates that `ApexEMeans` + `CroissAttn`-FP8 / SM120 Tensor-Core kernels + `CUDAGraph`
not only dominate raw k-means distortion-vs-time frontiers, but also power **production vector
compression and Asymmetric Distance Computation (ADC) ANN search** orders of magnitude faster
than `Lloyd@25` and `faiss.IndexPQ` while matching or beating Reconstruction MSE / SQNR (dB)
and `Recall@1` / `Recall@10`:

1. **`ApexPQ` (Fused 3D Batched Multi-Subspace Product Quantizer)**:
   - Splits `D` into `M` orthogonal subspaces of dimension `d_sub = D // M` (`Ks = 256` -> `1`
     `uint8` byte per subspace = `M` bytes/vector!).
   - Trains **all `M` codebooks simultaneously in parallel on GPU** (eliminating `M`-way Python
     subspace loops) via:
       * Single-pass per-subspace FP64 origin-centering + power-of-two dynamic-range conditioning
         + coalesced stream pre-permutation (supporting arbitrary non-contiguous `stride_xn, stride_xd`
         and optional fused `CroissAttn` decoupled FP8 per-subspace-vector quantization `qx, sx,
         x_norm_rep` with zero `0x7f` E4M3FN NaNs).
       * Parallel `M`-CTA Tournament Greedy `k-means++` (`L=3`) + 1-step weighted Voronoi polish.
       * Race-free 3D Tensor-Core streaming engine (`_pq_fused_assign_scatter_kernel` + wave-safe
         ping-pong `_pq_ema_oja_prep_kernel` across all `M * Ks` centroids + single-CTA-per-subspace
         `_pq_subspace_revive_kernel` every 5 steps fusing Twin-Safe Huygens shield, Gumbel donor
         tournament, and Half-Gaussian Oja splits) captured inside 5-step `torch.cuda.CUDAGraph`
         super-step replays.
       * Vectorized 3D Pre-Closing Independent Anchor + Positive-Part James-Stein Simplex EB.
   - Blazing-fast GPU LUT-based Asymmetric Distance Computation (`adc_search`) using a Split-N
     L1-resident LUT + IEEE-754 `uint64` packed `(dist_bits << 32) | id` `tl.topk` Triton kernel.
   - Supports pluggable solver backends (`ApexEMeans`, `UltraEMeans`, `HybridSEMeans`, `EMeansPP`,
     `SMeans`, `EMeans`, `Lloyd@25`, `lloyd_25_batched`, and `faiss.IndexPQ` reference) under
     matched `T in {1, 2, 5}` epoch budgets.

2. **`ApexRVQ` (2-Stage Residual Vector Quantizer)**:
   - Stage 1 trains coarse codebook `C_1` (`K_1` centroids) on `X_train`.
   - Fused GPU assignment + FP64-differenced residual kernel computes exact `R = X_train - C_1[labels_1]`
     with 100% `+20,000` coordinate-shift invariance.
   - Stage 2 trains residual codebook `C_2` (`K_2` centroids) on `R` (where `CroissAttn`-FP8's
     per-row scale `sr_i` preserves full mantissa precision across both tiny cell-interior
     residuals and large boundary residuals).
   - Evaluates origin-centered **exact** 2-stage LUT-ADC (`||q - (C_1[c_1] + C_2[c_2])||_2^2 =
     LUT_1[q, c_1] + LUT_2[q, c_2] + 2 <C_1_centered[c_1], C_2[c_2]>`) via both CSR coarse-beam
     pruning (`adc_beam`) and exhaustive Split-N Triton LUT-ADC (`adc_beam=None`), with zero
     train/query split leakage.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import math
import time
from typing import Any, Sequence

import numpy as np
import torch

from .apex_emeans import ApexEMeans, build_apex_arm, warmup_apex_kernels
from .baselines import LloydKMeans
from .emeans import EMeans
from .emeans_pp import EMeansPP
from .fast_ops import FLOAT8_E4M3FN, HAS_TRITON, block_permute_indices
from .kernels import HAS_FAISS, assign_points
from .recipe import (
    ALPHA_MIN_FACTOR,
    CHECK_UNUSED_EVERY,
    DEFAULT_INIT_USAGE,
    DEFAULT_UNUSED_THRESHOLD,
    SPLIT_EPSILON,
    build_recipe,
    compute_step_law,
)
from .smeans import SMeans
from .super_emeans import build_hybrid_arm
from .ultra_emeans import HALF_GAUSSIAN_MEAN, build_ultra_arm

if HAS_FAISS:
    import faiss

if HAS_TRITON:
    import triton
    import triton.language as tl

FP8_MAX_FINITE: float = 448.0
FP32_TINY: float = 1.1754943508222875e-38


# ==============================================================================
# Solver Name Normalization & Pluggable Backend Factory
# ==============================================================================

CANONICAL_SOLVER_ALIASES: dict[str, str] = {
    "apex_emeans": "apex_emeans",
    "apexemeans": "apex_emeans",
    "apex": "apex_emeans",
    "apex_pq": "apex_emeans",
    "apex_rvq": "apex_emeans",
    "apex_emeans_fp8": "apex_emeans_fp8",
    "apex_fp8": "apex_emeans_fp8",
    "croiss_fp8": "apex_emeans_fp8",
    "apex_emeans_bf16": "apex_emeans_bf16",
    "apex_bf16": "apex_emeans_bf16",
    "apex_emeans_subspace": "apex_emeans_subspace",
    "ultra_emeans": "ultra_emeans",
    "ultraemeans": "ultra_emeans",
    "ultra": "ultra_emeans",
    "hybrid_semeans": "hybrid_semeans",
    "hybridsemeans": "hybrid_semeans",
    "hybrid_super_smeans": "hybrid_semeans",
    "emeans_pp": "emeans_pp",
    "emeanspp": "emeans_pp",
    "e-means++": "emeans_pp",
    "emeans++": "emeans_pp",
    "smeans": "smeans",
    "s-means": "smeans",
    "smeans_fast": "smeans",
    "smeans_ref": "smeans_ref",
    "emeans": "emeans",
    "e-means": "emeans",
    "lloyd_25": "lloyd_25",
    "lloyd@25": "lloyd_25",
    "lloydkmeans": "lloyd_25",
    "lloyd": "lloyd_25",
    "lloyd_25_batched": "lloyd_25_batched",
    "faiss_pq": "faiss_pq",
    "faiss_pq_full": "faiss_pq",
    "faiss.indexpq": "faiss_pq",
    "indexpq": "faiss_pq",
    "faiss": "faiss_pq",
    "faiss_pq_sub": "faiss_pq_sub",
    "faiss_pq_subsampled": "faiss_pq_sub",
    "faiss_pq_default": "faiss_pq_sub",
}

PQ_SUPPORTED_SOLVERS: tuple[str, ...] = (
    "apex_emeans",
    "apex_emeans_fp8",
    "apex_emeans_bf16",
    "apex_emeans_subspace",
    "ultra_emeans",
    "hybrid_semeans",
    "emeans_pp",
    "smeans",
    "smeans_ref",
    "emeans",
    "lloyd_25",
    "lloyd_25_batched",
    "faiss_pq",
    "faiss_pq_sub",
)

RVQ_SUPPORTED_SOLVERS: tuple[str, ...] = (
    "apex_emeans",
    "apex_emeans_fp8",
    "apex_emeans_bf16",
    "ultra_emeans",
    "hybrid_semeans",
    "emeans_pp",
    "smeans",
    "smeans_ref",
    "emeans",
    "lloyd_25",
)


def normalize_solver_name(solver: str) -> str:
    """Normalize user-supplied solver identifier to canonical snake_case key."""
    if not isinstance(solver, str):
        raise ValueError(f"solver must be str, got {type(solver).__name__}")
    key = solver.strip().lower().replace(" ", "_")
    if key not in CANONICAL_SOLVER_ALIASES:
        raise ValueError(
            f"Unsupported solver={solver!r}; expected one of {sorted(set(CANONICAL_SOLVER_ALIASES.values()))}"
        )
    return CANONICAL_SOLVER_ALIASES[key]


def instantiate_subproblem_solver(
    solver: str,
    d: int,
    k: int,
    nepochs: float,
    seed: int,
    solver_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Instantiate a single clustering solver `(d, k, nepochs, seed)` for RVQ stages or per-subspace PQ."""
    canon = normalize_solver_name(solver)
    kw = dict(solver_kwargs or {})
    if canon in ("apex_emeans", "apex_emeans_subspace"):
        return ApexEMeans(d=d, k=k, nepochs=nepochs, seed=seed, **kw)
    if canon == "apex_emeans_fp8":
        return build_apex_arm("apex_emeans_fp8", d=d, k=k, nepochs=nepochs, seed=seed, **kw)
    if canon == "apex_emeans_bf16":
        return build_apex_arm("apex_emeans_bf16", d=d, k=k, nepochs=nepochs, seed=seed, **kw)
    if canon == "ultra_emeans":
        return build_ultra_arm("ultra_emeans", d=d, k=k, nepochs=nepochs, seed=seed, **kw)
    if canon == "hybrid_semeans":
        return build_hybrid_arm("hybrid_super_smeans", d=d, k=k, nepochs=nepochs, seed=seed, **kw)
    if canon == "emeans_pp":
        return EMeansPP(d=d, k=k, nepochs=nepochs, seed=seed, **kw)
    if canon == "smeans":
        return SMeans(d=d, k=k, nepochs=nepochs, exact_ref=False, seed=seed, **kw)
    if canon == "smeans_ref":
        return SMeans(d=d, k=k, nepochs=nepochs, exact_ref=True, seed=seed, **kw)
    if canon == "emeans":
        return EMeans(d=d, k=k, nepochs=nepochs, seed=seed, **kw)
    if canon == "lloyd_25":
        niter = int(kw.pop("niter", 25))
        return LloydKMeans(d=d, k=k, niter=niter, init="random", seed=seed, **kw)
    raise ValueError(f"Solver {solver!r} (canonical={canon!r}) is not a standalone single-space k-means solver")


def _has_sm89_fp8(device: torch.device) -> bool:
    if device.type != "cuda" or not HAS_TRITON or FLOAT8_E4M3FN is None:
        return False
    return torch.cuda.get_device_capability(device) >= (8, 9) and hasattr(tl, "float8e4nv")


def _power_of_two_coord_scales_64(max_abs_per_m: torch.Tensor) -> torch.Tensor:
    """Compute exact power-of-two multiplicative conditioning scales `(M,)` in float64.

    Returns `1.0` when `max_abs` is within normal `[1e-5, 1e5]` range so normal-scale inputs
    undergo zero scaling, and `2^k` when `max_abs < 1e-5` or `max_abs > 1e5` so extreme
    `1e-25..1e+25` inputs never underflow/overflow FP32/BF16/FP8 squares.
    """
    m_clamped = max_abs_per_m.double().clamp_min(1e-300)
    log2_v = torch.log2(m_clamped)
    exp_shift = torch.round(-log2_v).clamp(-950.0, 950.0)
    needs_scale = (max_abs_per_m > 1e5) | ((max_abs_per_m > 0.0) & (max_abs_per_m < 1e-5))
    return torch.where(needs_scale, torch.exp2(exp_shift), torch.ones_like(m_clamped))


@dataclass(frozen=True)
class ADCRecallMetrics:
    """Comprehensive Asymmetric Distance Computation (ADC) Recall summary."""

    recall_at_1: float          # Exact ID 1-Recall@1 (`pred_ids[:, 0] == gt_ids[:, 0]`)
    one_recall_at_10: float     # Jegou et al. TPAMI 2011 `R@10` (`gt_ids[:, 0] in pred_ids[:, :10]`)
    recall_at_10: float         # `10-Recall@10` intersection (`|pred[:10] & gt[:10]| / 10`)
    tie_recall_at_1: float      # Tie-aware exact FP64 distance `d2(q, x_pred0) <= gt_d2[0] * (1+1e-11)`
    tie_one_recall_at_10: float # Tie-aware `R@10`
    tie_recall_at_10: float     # Tie-aware `10-Recall@10`
    search_time_sec: float
    qps: float


@dataclass(frozen=True)
class ReconstructionMetrics:
    """Exact IEEE-754 FP64 vector quantization reconstruction fidelity metrics."""

    mse: float                  # Mean squared Euclidean error `E[||x - x_hat||_2^2]`
    per_dim_mse: float          # Per-coordinate MSE `MSE / D`
    rmse: float                 # `sqrt(MSE)`
    sqnr_db: float              # Standard `10 * log10(E[||x||_2^2] / MSE)`
    evr_sqnr_db: float          # Shift-invariant `10 * log10(E[||x - mu_train||_2^2] / MSE)`
    mean_cosine_sim: float      # Mean cosine similarity `<x, x_hat> / (||x|| ||x_hat||)`


# ==============================================================================
# Fused Blackwell SM120 Triton Kernels for `ApexPQ` & `ApexRVQ`
# ==============================================================================

if HAS_TRITON:

    @triton.jit(do_not_specialize=["N", "M", "D_SUB", "D_PAD", "stride_xn", "stride_xd"])
    def _pq_prepermute_encode_kernel(
        x_ptr,
        perm_ptr,
        origin64_ptr,
        scale64_ptr,
        x_stream_ptr,
        qx_ptr,
        sx_ptr,
        x_norm_ptr,
        N,
        M,
        D_SUB,
        D_PAD,
        stride_xn,
        stride_xd,
        WRITE_FP8: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Single-pass fused permutation + FP64 origin-centering + scale + optional CroissAttn-FP8 encode."""
        pid_n = tl.program_id(0).to(tl.int64)
        pid_m = tl.program_id(1).to(tl.int64)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
        mask_n = offs_n < N

        s_xn = stride_xn.to(tl.int64)
        s_xd = stride_xd.to(tl.int64)
        m_i64 = M.to(tl.int64)
        d_sub_i64 = D_SUB.to(tl.int64)
        d_pad_i64 = D_PAD.to(tl.int64)

        src_idx = tl.load(perm_ptr + offs_n, mask=mask_n, other=0).to(tl.int64)
        sc64 = tl.load(scale64_ptr + pid_m).to(tl.float64)

        amax_row = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for d0 in range(0, D_SUB, BLOCK_D):
            offs_d = d0 + tl.arange(0, BLOCK_D).to(tl.int64)
            mask_d = offs_d < D_SUB
            mask_nd = mask_n[:, None] & mask_d[None, :]
            orig64 = tl.load(origin64_ptr + pid_m * d_sub_i64 + offs_d, mask=mask_d, other=0.0).to(tl.float64)
            x_raw = tl.load(
                x_ptr + src_idx[:, None] * s_xn + (pid_m * d_sub_i64 + offs_d[None, :]) * s_xd,
                mask=mask_nd,
                other=0.0,
            ).to(tl.float64)
            x_c = ((x_raw - orig64[None, :]) * sc64).to(tl.float32)
            x_c = tl.where(mask_nd, x_c, 0.0)
            tl.store(
                x_stream_ptr + (offs_n[:, None] * m_i64 + pid_m) * d_sub_i64 + offs_d[None, :],
                x_c,
                mask=mask_nd,
            )
            if WRITE_FP8:
                amax_row = tl.maximum(amax_row, tl.max(tl.abs(x_c), axis=1))

        if WRITE_FP8:
            sx = tl.maximum(amax_row * (1.00000024 / 448.0), 1.1754943508222875e-38)
            inv_sx = 1.0 / sx
            q_sq = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for d0 in range(0, D_PAD, BLOCK_D):
                offs_d = d0 + tl.arange(0, BLOCK_D).to(tl.int64)
                mask_d = offs_d < D_SUB
                mask_pad = offs_d < D_PAD
                x_c = tl.load(
                    x_stream_ptr + (offs_n[:, None] * m_i64 + pid_m) * d_sub_i64 + offs_d[None, :],
                    mask=mask_n[:, None] & mask_d[None, :],
                    other=0.0,
                ).to(tl.float32)
                scaled = tl.where(
                    mask_n[:, None] & mask_d[None, :],
                    tl.minimum(tl.maximum(x_c * inv_sx[:, None], -448.0), 448.0),
                    0.0,
                )
                qx = scaled.to(tl.float8e4nv, fp_downcast_rounding="rtne")
                tl.store(
                    qx_ptr + (offs_n[:, None] * m_i64 + pid_m) * d_pad_i64 + offs_d[None, :],
                    qx,
                    mask=mask_n[:, None] & mask_pad[None, :],
                )
                qxf = qx.to(tl.float32)
                q_sq += tl.sum(tl.where(mask_d[None, :], qxf * qxf, 0.0), axis=1)
            tl.store(sx_ptr + offs_n * m_i64 + pid_m, sx, mask=mask_n)
            tl.store(x_norm_ptr + offs_n * m_i64 + pid_m, (sx * sx) * q_sq, mask=mask_n)

    @triton.jit(do_not_specialize=["P", "num_select", "base_seed"])
    def _pq_batched_greedy_kpp_kernel(
        pair_d2_ptr,
        init_d2_ptr,
        weights_ptr,
        out_idx_ptr,
        P,
        num_select,
        base_seed,
        BLOCK_P: tl.constexpr,
        L_TRIALS: tl.constexpr,
    ):
        """Parallel `M`-CTA weighted Greedy k-means++ candidate tournament (`1` CTA per subspace `m`)."""
        m = tl.program_id(0).to(tl.int64)
        p_i64 = P.to(tl.int64)
        nsel_i64 = num_select.to(tl.int64)
        offs_p = tl.arange(0, BLOCK_P)
        mask_p = offs_p < P

        pool_d2 = tl.load(init_d2_ptr + m * p_i64 + offs_p, mask=mask_p, other=0.0).to(tl.float32)
        w = tl.load(weights_ptr + m * p_i64 + offs_p, mask=mask_p, other=0.0).to(tl.float32)
        log_w = tl.log(tl.maximum(w, 1e-30))
        first_pick = tl.load(out_idx_ptr + m * (nsel_i64 + 1)).to(tl.int32)
        avail = mask_p & (offs_p != first_pick)
        seed_m = base_seed + m.to(tl.int32) * 10007

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
                    u = tl.rand(seed_m, rng_offs)
                    u = tl.minimum(tl.maximum(u, 1e-30), 1.0 - 1e-7)
                    gumbel = -tl.log(-tl.log(u))
                    key_pos = tl.where(avail & (proposal > 0.0), log_prop + gumbel, -float("inf"))
                    key_fb = tl.where(avail, log_w + gumbel, -float("inf"))
                    key = tl.where(has_pos, key_pos, key_fb)
                    cand = tl.argmax(key, axis=0)

                row_d2 = tl.load(
                    pair_d2_ptr + (m * p_i64 + cand.to(tl.int64)) * p_i64 + offs_p.to(tl.int64),
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
            tl.store(out_idx_ptr + m * (nsel_i64 + 1) + 1 + s_idx, best_pick.to(tl.int64))

    @triton.jit(do_not_specialize=["B", "N_total", "M", "KS", "D_SUB", "D_PAD"])
    def _pq_fused_assign_scatter_kernel(
        x_stream_ptr,
        qx_ptr,
        sx_ptr,
        x_norm_fp8_ptr,
        batch_off_ptr,
        c_master_ptr,
        c_rep_ptr,
        sc_ptr,
        c_norm_ptr,
        prin_vec_ptr,
        sums_ptr,
        counts_ptr,
        batch_sse_ptr,
        g_sums_ptr,
        B,
        N_total,
        M,
        KS,
        D_SUB,
        D_PAD,
        REP_MODE: tl.constexpr,     # 0=FP32, 1=BF16, 3=CROISS_FP8_DECOUPLED
        TRACK_OJA: tl.constexpr,
        FULL_D: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Fused 3D SM120 Tensor-Core Assignment + L2 Scatter + Register Oja across `(cdiv(B, BLOCK_B), M)`."""
        pid_b = tl.program_id(0).to(tl.int64)
        pid_m = tl.program_id(1).to(tl.int64)

        m_i64 = M.to(tl.int64)
        ks_i64 = KS.to(tl.int64)
        d_sub_i64 = D_SUB.to(tl.int64)
        d_pad_i64 = D_PAD.to(tl.int64)
        n_tot_i64 = N_total.to(tl.int64)

        offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B).to(tl.int64)
        mask_b = offs_b < B
        base_off = tl.load(batch_off_ptr).to(tl.int64)
        row_pos = (base_off + offs_b) % n_tot_i64

        best_score = tl.full((BLOCK_B,), float("inf"), dtype=tl.float32)
        best_lbl = tl.zeros((BLOCK_B,), dtype=tl.int32)

        if FULL_D:
            cols_pad = tl.arange(0, BLOCK_D).to(tl.int64)
            mask_d = cols_pad < D_SUB
            mask_pad = cols_pad < D_PAD

            x_f32 = tl.load(
                x_stream_ptr + (row_pos[:, None] * m_i64 + pid_m) * d_sub_i64 + cols_pad[None, :],
                mask=mask_b[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)

            if REP_MODE == 3:
                qx_tile = tl.load(
                    qx_ptr + (row_pos[:, None] * m_i64 + pid_m) * d_pad_i64 + cols_pad[None, :],
                    mask=mask_b[:, None] & mask_pad[None, :],
                    other=0.0,
                )
                sx_vec = tl.load(sx_ptr + row_pos * m_i64 + pid_m, mask=mask_b, other=1.0).to(tl.float32)
                x_norm = tl.load(x_norm_fp8_ptr + row_pos * m_i64 + pid_m, mask=mask_b, other=0.0).to(tl.float32)
            elif REP_MODE == 1:
                x_q = x_f32.to(tl.bfloat16)
                x_qf = x_q.to(tl.float32)
                x_norm = tl.sum(x_qf * x_qf, axis=1)
            else:
                x_norm = tl.sum(x_f32 * x_f32, axis=1)

            for k0 in range(0, KS, BLOCK_K):
                offs_k = k0 + tl.arange(0, BLOCK_K).to(tl.int64)
                mask_k = offs_k < KS
                c_tile = tl.load(
                    c_rep_ptr + (pid_m * ks_i64 + offs_k[:, None]) * d_pad_i64 + cols_pad[None, :],
                    mask=mask_k[:, None] & mask_pad[None, :],
                    other=0.0,
                )
                cn = tl.load(c_norm_ptr + pid_m * ks_i64 + offs_k, mask=mask_k, other=float("inf")).to(tl.float32)

                if REP_MODE == 3:
                    sc_vec = tl.load(sc_ptr + pid_m * ks_i64 + offs_k, mask=mask_k, other=0.0).to(tl.float32)
                    raw_dots = tl.dot(
                        qx_tile,
                        tl.trans(c_tile),
                        out_dtype=tl.float32,
                        max_num_imprecise_acc=0,
                    )
                    scores = cn[None, :] - 2.0 * sx_vec[:, None] * (raw_dots * sc_vec[None, :])
                elif REP_MODE == 1:
                    raw_dots = tl.dot(x_q, tl.trans(c_tile.to(tl.bfloat16)), out_dtype=tl.float32)
                    scores = cn[None, :] - 2.0 * raw_dots
                else:
                    raw_dots = tl.dot(
                        x_f32,
                        tl.trans(c_tile.to(tl.float32)),
                        input_precision="ieee",
                        out_dtype=tl.float32,
                    )
                    scores = cn[None, :] - 2.0 * raw_dots

                scores = tl.where(mask_k[None, :], scores, float("inf"))
                t_min, t_idx = tl.min(scores, axis=1, return_indices=True)
                better = t_min < best_score
                best_score = tl.where(better, t_min, best_score)
                best_lbl = tl.where(better, k0 + t_idx.to(tl.int32), best_lbl)

            sq_d = tl.minimum(tl.maximum(x_norm + best_score, 0.0), 1e35)
            lbl_global = pid_m * ks_i64 + best_lbl.to(tl.int64)

            tl.atomic_add(counts_ptr + lbl_global, 1.0, mask=mask_b, sem="relaxed")
            tl.atomic_add(batch_sse_ptr + lbl_global, sq_d, mask=mask_b, sem="relaxed")

            mask_bd = mask_b[:, None] & mask_d[None, :]
            tl.atomic_add(
                sums_ptr + lbl_global[:, None] * d_sub_i64 + cols_pad[None, :],
                x_f32,
                mask=mask_bd,
                sem="relaxed",
            )
            if TRACK_OJA:
                c_val = tl.load(
                    c_master_ptr + lbl_global[:, None] * d_sub_i64 + cols_pad[None, :],
                    mask=mask_bd,
                    other=0.0,
                ).to(tl.float32)
                pv_val = tl.load(
                    prin_vec_ptr + lbl_global[:, None] * d_sub_i64 + cols_pad[None, :],
                    mask=mask_bd,
                    other=0.0,
                ).to(tl.float32)
                pv_norm = tl.maximum(tl.sqrt(tl.sum(pv_val * pv_val, axis=1)), 1e-12)
                r_val = tl.where(mask_bd, tl.minimum(tl.maximum(x_f32 - c_val, -1e17), 1e17), 0.0)
                proj = tl.sum(r_val * (pv_val / pv_norm[:, None]), axis=1)
                tl.atomic_add(
                    g_sums_ptr + lbl_global[:, None] * d_sub_i64 + cols_pad[None, :],
                    r_val * proj[:, None],
                    mask=mask_bd,
                    sem="relaxed",
                )
        else:
            x_norm = tl.zeros((BLOCK_B,), dtype=tl.float32)
            if REP_MODE == 3:
                sx_vec = tl.load(sx_ptr + row_pos * m_i64 + pid_m, mask=mask_b, other=1.0).to(tl.float32)
                x_norm = tl.load(x_norm_fp8_ptr + row_pos * m_i64 + pid_m, mask=mask_b, other=0.0).to(tl.float32)
            else:
                for d0 in range(0, D_SUB, BLOCK_D):
                    cols = d0 + tl.arange(0, BLOCK_D).to(tl.int64)
                    mask_d = cols < D_SUB
                    xt = tl.load(
                        x_stream_ptr + (row_pos[:, None] * m_i64 + pid_m) * d_sub_i64 + cols[None, :],
                        mask=mask_b[:, None] & mask_d[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    if REP_MODE == 1:
                        xt = xt.to(tl.bfloat16).to(tl.float32)
                    x_norm += tl.sum(xt * xt, axis=1)

            for k0 in range(0, KS, BLOCK_K):
                offs_k = k0 + tl.arange(0, BLOCK_K).to(tl.int64)
                mask_k = offs_k < KS
                dots = tl.zeros((BLOCK_B, BLOCK_K), dtype=tl.float32)
                for d0 in range(0, D_PAD, BLOCK_D):
                    cols = d0 + tl.arange(0, BLOCK_D).to(tl.int64)
                    mask_d = cols < D_SUB
                    mask_pad = cols < D_PAD
                    ct = tl.load(
                        c_rep_ptr + (pid_m * ks_i64 + offs_k[:, None]) * d_pad_i64 + cols[None, :],
                        mask=mask_k[:, None] & mask_pad[None, :],
                        other=0.0,
                    )
                    if REP_MODE == 3:
                        qxt = tl.load(
                            qx_ptr + (row_pos[:, None] * m_i64 + pid_m) * d_pad_i64 + cols[None, :],
                            mask=mask_b[:, None] & mask_pad[None, :],
                            other=0.0,
                        )
                        dots = tl.dot(qxt, tl.trans(ct), acc=dots, out_dtype=tl.float32, max_num_imprecise_acc=0)
                    elif REP_MODE == 1:
                        xt = tl.load(
                            x_stream_ptr + (row_pos[:, None] * m_i64 + pid_m) * d_sub_i64 + cols[None, :],
                            mask=mask_b[:, None] & mask_d[None, :],
                            other=0.0,
                        ).to(tl.bfloat16)
                        dots = tl.dot(xt, tl.trans(ct.to(tl.bfloat16)), acc=dots, out_dtype=tl.float32)
                    else:
                        xt = tl.load(
                            x_stream_ptr + (row_pos[:, None] * m_i64 + pid_m) * d_sub_i64 + cols[None, :],
                            mask=mask_b[:, None] & mask_d[None, :],
                            other=0.0,
                        ).to(tl.float32)
                        dots = tl.dot(
                            xt,
                            tl.trans(ct.to(tl.float32)),
                            acc=dots,
                            input_precision="ieee",
                            out_dtype=tl.float32,
                        )
                cn = tl.load(c_norm_ptr + pid_m * ks_i64 + offs_k, mask=mask_k, other=float("inf")).to(tl.float32)
                if REP_MODE == 3:
                    sc_vec = tl.load(sc_ptr + pid_m * ks_i64 + offs_k, mask=mask_k, other=0.0).to(tl.float32)
                    scores = cn[None, :] - 2.0 * sx_vec[:, None] * (dots * sc_vec[None, :])
                else:
                    scores = cn[None, :] - 2.0 * dots
                scores = tl.where(mask_k[None, :], scores, float("inf"))
                t_min, t_idx = tl.min(scores, axis=1, return_indices=True)
                better = t_min < best_score
                best_score = tl.where(better, t_min, best_score)
                best_lbl = tl.where(better, k0 + t_idx.to(tl.int32), best_lbl)

            sq_d = tl.minimum(tl.maximum(x_norm + best_score, 0.0), 1e35)
            lbl_global = pid_m * ks_i64 + best_lbl.to(tl.int64)
            tl.atomic_add(counts_ptr + lbl_global, 1.0, mask=mask_b, sem="relaxed")
            tl.atomic_add(batch_sse_ptr + lbl_global, sq_d, mask=mask_b, sem="relaxed")

            pv_nsq = tl.zeros((BLOCK_B,), dtype=tl.float32)
            if TRACK_OJA:
                for d0 in range(0, D_SUB, BLOCK_D):
                    cols = d0 + tl.arange(0, BLOCK_D).to(tl.int64)
                    mask_d = cols < D_SUB
                    pvt = tl.load(
                        prin_vec_ptr + lbl_global[:, None] * d_sub_i64 + cols[None, :],
                        mask=mask_b[:, None] & mask_d[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    pv_nsq += tl.sum(pvt * pvt, axis=1)
            inv_pvn = 1.0 / tl.maximum(tl.sqrt(pv_nsq), 1e-12)
            proj = tl.zeros((BLOCK_B,), dtype=tl.float32)
            for d0 in range(0, D_SUB, BLOCK_D):
                cols = d0 + tl.arange(0, BLOCK_D).to(tl.int64)
                mask_d = cols < D_SUB
                mask_bd = mask_b[:, None] & mask_d[None, :]
                xt = tl.load(
                    x_stream_ptr + (row_pos[:, None] * m_i64 + pid_m) * d_sub_i64 + cols[None, :],
                    mask=mask_bd,
                    other=0.0,
                ).to(tl.float32)
                tl.atomic_add(sums_ptr + lbl_global[:, None] * d_sub_i64 + cols[None, :], xt, mask=mask_bd, sem="relaxed")
                if TRACK_OJA:
                    ct = tl.load(c_master_ptr + lbl_global[:, None] * d_sub_i64 + cols[None, :], mask=mask_bd, other=0.0).to(tl.float32)
                    pvt = tl.load(prin_vec_ptr + lbl_global[:, None] * d_sub_i64 + cols[None, :], mask=mask_bd, other=0.0).to(tl.float32)
                    rt = tl.where(mask_bd, tl.minimum(tl.maximum(xt - ct, -1e17), 1e17), 0.0)
                    proj += tl.sum(rt * (pvt * inv_pvn[:, None]), axis=1)
            if TRACK_OJA:
                for d0 in range(0, D_SUB, BLOCK_D):
                    cols = d0 + tl.arange(0, BLOCK_D).to(tl.int64)
                    mask_d = cols < D_SUB
                    mask_bd = mask_b[:, None] & mask_d[None, :]
                    xt = tl.load(
                        x_stream_ptr + (row_pos[:, None] * m_i64 + pid_m) * d_sub_i64 + cols[None, :],
                        mask=mask_bd,
                        other=0.0,
                    ).to(tl.float32)
                    ct = tl.load(c_master_ptr + lbl_global[:, None] * d_sub_i64 + cols[None, :], mask=mask_bd, other=0.0).to(tl.float32)
                    rt = tl.where(mask_bd, tl.minimum(tl.maximum(xt - ct, -1e17), 1e17), 0.0)
                    tl.atomic_add(
                        g_sums_ptr + lbl_global[:, None] * d_sub_i64 + cols[None, :],
                        rt * proj[:, None],
                        mask=mask_bd,
                        sem="relaxed",
                    )

    @triton.jit(
        do_not_specialize=[
            "M_KS",
            "D_SUB",
            "D_PAD",
            "b_cur",
            "rows_in_ep",
            "total_steps",
            "total_planned_rows",
            "accum_pass",
        ]
    )
    def _pq_ema_oja_prep_kernel(
        state_in_ptr,       # (6,) float64: [step, rows_proc, ep_b_start, u_bias, s_bias, alpha_last]
        state_out_ptr,      # (6,) float64 ping-pong output (`state_out_ptr != state_in_ptr` -> 0 CTA wave race!)
        batch_off_in_ptr,   # (1,) int64
        batch_off_out_ptr,  # (1,) int64
        sums_ptr,
        counts_ptr,
        batch_sse_ptr,
        g_sums_ptr,
        centroids_ptr,
        usage_ptr,
        sse_ema_ptr,
        prin_vec_ptr,
        slow_centroids_ptr,
        slow_usage_ptr,
        pass_sums_ptr,
        pass_counts_ptr,
        pass_counts_sq_ptr,
        pass_sse_ptr,
        c_rep_ptr,
        sc_ptr,
        c_norm_ptr,
        M_KS,
        D_SUB,
        D_PAD,
        b_cur,
        rows_in_ep,
        total_steps,
        total_planned_rows,
        eff_peak,
        alpha_min_factor,
        alpha_slow,
        eff_pass_span,
        accum_pass,
        USE_WIDE_SCHED: tl.constexpr,
        TRACK_OJA: tl.constexpr,
        REP_MODE: tl.constexpr,   # 0=FP32, 1=BF16, 3=CROISS_FP8_DECOUPLED
        BLOCK_D: tl.constexpr,
    ):
        """Wave-race-free parallel per-centroid (`grid=(M * KS,)`) EMA + Oja + SlowEMA + Closing + Prep."""
        j = tl.program_id(0).to(tl.int64)
        if j >= M_KS:
            return

        d_sub_i64 = D_SUB.to(tl.int64)
        d_pad_i64 = D_PAD.to(tl.int64)

        step_f = tl.load(state_in_ptr + 0)
        rows_proc_f = tl.load(state_in_ptr + 1)
        ep_b_start_f = tl.load(state_in_ptr + 2)
        u_bias_in = tl.load(state_in_ptr + 3)
        s_bias_in = tl.load(state_in_ptr + 4)

        if USE_WIDE_SCHED:
            raw_frac = (rows_proc_f / tl.maximum(total_planned_rows.to(tl.float64), 1.0)).to(tl.float32)
        else:
            raw_frac = (
                tl.minimum(step_f, total_steps.to(tl.float64)) / tl.maximum(total_steps.to(tl.float64), 1.0)
            ).to(tl.float32)
        frac = tl.minimum(tl.maximum(raw_frac, 0.0), 1.0)
        cos_half = 0.5 * (1.0 + tl.cos(3.141592653589793 * frac))
        alpha = eff_peak * (alpha_min_factor + (1.0 - alpha_min_factor) * cos_half)
        decay = 1.0 - alpha
        is_rep = alpha >= 1.0

        do_accum = accum_pass != 0
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

        # CTA 0 writes distinct `state_out_ptr` & `batch_off_out_ptr` (zero read-after-write race across CTA waves)
        if j == 0:
            a64 = alpha.to(tl.float64)
            d64 = decay.to(tl.float64)
            u_bias_out = tl.where(is_rep, 1.0, d64 * u_bias_in + a64)
            s_bias_out = tl.where(is_rep, 1.0, d64 * s_bias_in + a64)
            b_f64 = b_cur.to(tl.float64)
            tl.store(state_out_ptr + 0, step_f + 1.0)
            tl.store(state_out_ptr + 1, rows_proc_f + b_f64)
            tl.store(state_out_ptr + 2, ep_b_start_f + b_f64)
            tl.store(state_out_ptr + 3, u_bias_out)
            tl.store(state_out_ptr + 4, s_bias_out)
            tl.store(state_out_ptr + 5, a64)
            tl.store(batch_off_out_ptr, tl.load(batch_off_in_ptr) + b_cur.to(tl.int64))

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

        if do_accum:
            tl.store(pass_counts_ptr + j, tl.load(pass_counts_ptr + j) + (w_pass * cnt).to(tl.float64))
            tl.store(pass_counts_sq_ptr + j, tl.load(pass_counts_sq_ptr + j) + (w_pass * w_pass * cnt).to(tl.float64))
            tl.store(pass_sse_ptr + j, tl.load(pass_sse_ptr + j) + (w_pass * bsse).to(tl.float64))

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
                for d0 in range(0, D_SUB, BLOCK_D):
                    cols = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
                    mask_d = cols < D_SUB
                    pv_old = tl.load(prin_vec_ptr + j * d_sub_i64 + cols, mask=mask_d, other=0.0).to(tl.float32)
                    gs_val = tl.load(g_sums_ptr + j * d_sub_i64 + cols, mask=mask_d, other=0.0).to(tl.float32)
                    gm_val = gs_val * inv_cnt
                    pv_nsq += tl.sum(pv_old * pv_old, axis=0)
                    g_nsq += tl.sum(gm_val * gm_val, axis=0)
                    g_dot_p += tl.sum(gm_val * pv_old, axis=0)
                pv_n = tl.maximum(tl.sqrt(pv_nsq), 1e-12)
                g_n = tl.sqrt(g_nsq)
                rayleigh = tl.maximum(g_dot_p / pv_n, 0.0)
                do_oja_upd = g_n > 1e-12
                target_scale = tl.sqrt(rayleigh) / tl.maximum(g_n, 1e-12)

        norm_acc = tl.zeros((), dtype=tl.float32)
        amax_c = tl.zeros((), dtype=tl.float32)
        w_oja = tl.minimum(tl.maximum(weight, 0.0), 1.0)

        for d0 in range(0, D_SUB, BLOCK_D):
            cols = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
            mask_d = cols < D_SUB
            s_val = tl.load(sums_ptr + j * d_sub_i64 + cols, mask=mask_d, other=0.0).to(tl.float32)
            if do_accum:
                ps_old = tl.load(pass_sums_ptr + j * d_sub_i64 + cols, mask=mask_d, other=0.0).to(tl.float32)
                tl.store(pass_sums_ptr + j * d_sub_i64 + cols, ps_old + w_pass * s_val, mask=mask_d)

            if TRACK_OJA:
                gs_val = tl.load(g_sums_ptr + j * d_sub_i64 + cols, mask=mask_d, other=0.0).to(tl.float32)
                if do_oja_upd:
                    pv_old = tl.load(prin_vec_ptr + j * d_sub_i64 + cols, mask=mask_d, other=0.0).to(tl.float32)
                    gm_val = gs_val * inv_cnt
                    pv_new = (1.0 - w_oja) * pv_old + w_oja * (gm_val * target_scale)
                    tl.store(prin_vec_ptr + j * d_sub_i64 + cols, pv_new, mask=mask_d)

            tl.debug_barrier()
            tl.store(sums_ptr + j * d_sub_i64 + cols, 0.0, mask=mask_d)
            if TRACK_OJA:
                tl.store(g_sums_ptr + j * d_sub_i64 + cols, 0.0, mask=mask_d)

            c_old = tl.load(centroids_ptr + j * d_sub_i64 + cols, mask=mask_d, other=0.0).to(tl.float32)
            mean_val = s_val * inv_cnt
            pos_val = c_old + weight * (mean_val - c_old)
            upd_val = tl.where(is_one, mean_val, pos_val)
            c_new = tl.where(touched, upd_val, c_old)
            tl.store(centroids_ptr + j * d_sub_i64 + cols, c_new, mask=mask_d & touched)

            if touched:
                sc_old = tl.load(slow_centroids_ptr + j * d_sub_i64 + cols, mask=mask_d, other=0.0).to(tl.float32)
                sc_pos = sc_old + s_weight * (mean_val - sc_old)
                sc_new = tl.where(s_is_one, mean_val, sc_pos)
                tl.store(slow_centroids_ptr + j * d_sub_i64 + cols, sc_new, mask=mask_d)

            c_masked = tl.where(mask_d, c_new, 0.0)
            if REP_MODE == 3:
                amax_c = tl.maximum(amax_c, tl.max(tl.abs(c_masked), axis=0))
            elif REP_MODE == 1:
                cq = c_masked.to(tl.bfloat16)
                tl.store(c_rep_ptr + j * d_pad_i64 + cols, cq, mask=mask_d)
                cqf = cq.to(tl.float32)
                norm_acc += tl.sum(cqf * cqf, axis=0)
            else:
                tl.store(c_rep_ptr + j * d_pad_i64 + cols, c_masked, mask=mask_d)
                norm_acc += tl.sum(c_masked * c_masked, axis=0)

        # Zero any trailing `D_SUB <= col < D_PAD` lanes for BF16/FP32
        if REP_MODE != 3:
            for d0 in range(0, D_PAD, BLOCK_D):
                cols = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
                pad_only = (cols >= D_SUB) & (cols < D_PAD)
                if REP_MODE == 1:
                    tl.store(c_rep_ptr + j * d_pad_i64 + cols, tl.zeros((BLOCK_D,), dtype=tl.bfloat16), mask=pad_only)
                else:
                    tl.store(c_rep_ptr + j * d_pad_i64 + cols, 0.0, mask=pad_only)

        if REP_MODE == 3:
            sc_j = tl.maximum(amax_c * (1.00000024 / 448.0), 1.1754943508222875e-38)
            inv_sc_j = 1.0 / sc_j
            q_sq = tl.zeros((), dtype=tl.float32)
            for d0 in range(0, D_PAD, BLOCK_D):
                cols = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
                mask_d = cols < D_SUB
                mask_pad = cols < D_PAD
                cv = tl.load(centroids_ptr + j * d_sub_i64 + cols, mask=mask_d, other=0.0).to(tl.float32)
                scaled = tl.where(mask_d, tl.minimum(tl.maximum(cv * inv_sc_j, -448.0), 448.0), 0.0)
                qc = scaled.to(tl.float8e4nv, fp_downcast_rounding="rtne")
                tl.store(c_rep_ptr + j * d_pad_i64 + cols, qc, mask=mask_pad)
                qcf = qc.to(tl.float32)
                q_sq += tl.sum(tl.where(mask_d, qcf * qcf, 0.0), axis=0)
            tl.store(sc_ptr + j, sc_j)
            tl.store(c_norm_ptr + j, (sc_j * sc_j) * q_sq)
        else:
            tl.store(c_norm_ptr + j, norm_acc)

        new_u = tl.where(touched, mass, tl.where(is_rep, old_u, retained))
        new_sse = tl.where(is_rep, tl.where(touched, bsse, 0.5 * old_sse), decay * old_sse + alpha * bsse)
        tl.store(usage_ptr + j, new_u)
        tl.store(sse_ema_ptr + j, new_sse)

    @triton.jit(
        do_not_specialize=[
            "M",
            "KS",
            "D_SUB",
            "D_PAD",
            "b_cur",
            "eff_grace",
            "revive_stop_rows",
            "flush_closing",
            "update_pre_close",
            "base_seed",
        ]
    )
    def _pq_subspace_revive_kernel(
        state_ptr,          # (6,) float64 after EMA step
        centroids_ptr,      # (M * KS, D_SUB) float32
        usage_ptr,          # (M * KS,) float32
        sse_ema_ptr,        # (M * KS,) float32
        prin_vec_ptr,       # (M * KS, D_SUB) float32
        slow_centroids_ptr,
        slow_usage_ptr,
        pre_close_ptr,
        pass_sums_ptr,
        pass_counts_ptr,
        pass_counts_sq_ptr,
        pass_sse_ptr,
        c_rep_ptr,
        sc_ptr,
        c_norm_ptr,
        n_splits_ptr,       # (M,) int64
        M,
        KS,
        D_SUB,
        D_PAD,
        b_cur,
        unused_tau,
        gamma_protect,
        donor_usage_w,
        split_mult,
        eff_grace,
        revive_stop_rows,
        flush_closing,
        update_pre_close,
        base_seed,
        TRACK_OJA: tl.constexpr,
        REP_MODE: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Single-CTA-per-subspace (`grid=(M,)`) Twin-Safe Huygens Shield + Gumbel Donor + Half-Gaussian Oja Split."""
        m = tl.program_id(0).to(tl.int64)
        if m >= M:
            return
        if unused_tau <= 0.0 or KS < 2:
            return

        step_after = tl.load(state_ptr + 0)
        rows_after = tl.load(state_ptr + 1)
        u_bias = tl.maximum(tl.load(state_ptr + 3).to(tl.float32), 1e-6)
        s_bias = tl.maximum(tl.load(state_ptr + 4).to(tl.float32), 1e-6)

        step_before = step_after - 1.0
        rows_before = rows_after - b_cur.to(tl.float64)
        active_win = (step_after >= tl.maximum(eff_grace.to(tl.float64), 1.0)) & (
            rows_before <= revive_stop_rows.to(tl.float64)
        )
        if not active_win:
            return

        ks_i64 = KS.to(tl.int64)
        d_sub_i64 = D_SUB.to(tl.int64)
        d_pad_i64 = D_PAD.to(tl.int64)
        offs_k = tl.arange(0, BLOCK_K).to(tl.int64)
        mask_k = offs_k < KS
        k_glob = m * ks_i64 + offs_k

        u_val = tl.load(usage_ptr + k_glob, mask=mask_k, other=0.0).to(tl.float32)
        e_val = tl.load(sse_ema_ptr + k_glob, mask=mask_k, other=0.0).to(tl.float32)
        mean_u = tl.sum(u_val, axis=0) / tl.maximum(KS.to(tl.float32), 1.0)
        is_low = mask_k & (u_val < (unused_tau * mean_u))
        n_low = tl.sum(tl.where(is_low, 1, 0), axis=0)
        if n_low == 0:
            return

        # Huygens rare-mode protection on `is_low & (u_val > 0)`: exact nearest neighbor in subspace `m`
        u_unb = tl.maximum(u_val / u_bias, 0.0)
        e_unb = tl.maximum(e_val / s_bias, 0.0)
        mean_sse = tl.sum(tl.where(mask_k, e_unb, 0.0), axis=0) / tl.maximum(KS.to(tl.float32), 1.0)

        prot_mask = tl.zeros((BLOCK_K,), dtype=tl.int1)
        if gamma_protect > 0.0 and mean_sse > 0.0:
            c_norms_m = tl.zeros((BLOCK_K,), dtype=tl.float32)
            for d0 in range(0, D_SUB, BLOCK_D):
                cols = d0 + tl.arange(0, BLOCK_D).to(tl.int64)
                mask_d = cols < D_SUB
                cv = tl.load(
                    centroids_ptr + k_glob[:, None] * d_sub_i64 + cols[None, :],
                    mask=mask_k[:, None] & mask_d[None, :],
                    other=0.0,
                ).to(tl.float32)
                c_norms_m += tl.sum(cv * cv, axis=1)

            nn_d2 = tl.full((BLOCK_K,), float("inf"), dtype=tl.float32)
            # Tile both `k0_0` and `k1_0` by 64 so `tl.dot` stays `(64, 64)` (`< 16 KB` SMEM even at `KS=512`)!
            for k0_0 in range(0, KS, 64):
                ok0 = k0_0 + tl.arange(0, 64).to(tl.int64)
                mk0 = ok0 < KS
                kg0 = m * ks_i64 + ok0
                c0_n = tl.zeros((64,), dtype=tl.float32)
                for d0 in range(0, D_SUB, BLOCK_D):
                    cols = d0 + tl.arange(0, BLOCK_D).to(tl.int64)
                    mask_d = cols < D_SUB
                    c0v = tl.load(
                        centroids_ptr + kg0[:, None] * d_sub_i64 + cols[None, :],
                        mask=mk0[:, None] & mask_d[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    c0_n += tl.sum(c0v * c0v, axis=1)

                row_nn = tl.full((64,), float("inf"), dtype=tl.float32)
                for k1_0 in range(0, KS, 64):
                    ok1 = k1_0 + tl.arange(0, 64).to(tl.int64)
                    mk1 = ok1 < KS
                    kg1 = m * ks_i64 + ok1
                    dots = tl.zeros((64, 64), dtype=tl.float32)
                    c1_n = tl.zeros((64,), dtype=tl.float32)
                    for d0 in range(0, D_SUB, BLOCK_D):
                        cols = d0 + tl.arange(0, BLOCK_D).to(tl.int64)
                        mask_d = cols < D_SUB
                        c0 = tl.load(
                            centroids_ptr + kg0[:, None] * d_sub_i64 + cols[None, :],
                            mask=mk0[:, None] & mask_d[None, :],
                            other=0.0,
                        ).to(tl.float32)
                        c1 = tl.load(
                            centroids_ptr + kg1[:, None] * d_sub_i64 + cols[None, :],
                            mask=mk1[:, None] & mask_d[None, :],
                            other=0.0,
                        ).to(tl.float32)
                        c1_n += tl.sum(c1 * c1, axis=1)
                        dots = tl.dot(c0, tl.trans(c1), acc=dots, input_precision="ieee", out_dtype=tl.float32)
                    d2_mat = tl.maximum(c0_n[:, None] + c1_n[None, :] - 2.0 * dots, 0.0)
                    same = ok0[:, None] == ok1[None, :]
                    d2_mat = tl.where(mk1[None, :] & (~same), d2_mat, float("inf"))
                    row_nn = tl.minimum(row_nn, tl.min(d2_mat, axis=1))

                in_slice = (offs_k >= k0_0) & (offs_k < (k0_0 + 64))
                # Broadcast `row_nn` into `nn_d2[k0_0 : k0_0 + 64]` via 1D scratch or `c_norm_ptr`?
                # Even simpler: store `row_nn` temporarily into unused `pass_sse_ptr`? NO — never touch live accumulators!
                # Broadcast cleanly via inner scalar/mask or `tl.sum`? Wait: `nn_d2` is only needed to set `prot_mask`:
                # compute `prot_slice` directly on `(64,)` and OR it into `prot_mask` via index match!
                u_slice = tl.load(usage_ptr + kg0, mask=mk0, other=0.0).to(tl.float32) / u_bias
                low_slice = mk0 & ((u_slice * u_bias) < (unused_tau * mean_u))
                h_cost_s = 0.5 * tl.maximum(u_slice, 0.0) * row_nn
                prot_s = low_slice & (u_slice > 1e-6) & (h_cost_s > (gamma_protect * mean_sse))
                # Map `prot_s` `(64,)` onto `prot_mask` `(BLOCK_K,)` via `(offs_k[:, None] == ok0[None, :])` (`BLOCK_K * 64` bools = 8 KB!)
                hit_prot = tl.sum(tl.where((offs_k[:, None] == ok0[None, :]) & prot_s[None, :], 1, 0), axis=1) > 0
                prot_mask = prot_mask | (in_slice & hit_prot)
            _ = c_norms_m
            _ = nn_d2

        unused_cur = is_low & (~prot_mask)
        donor_cur = mask_k & (~is_low) & (u_val > 0.0)
        n_un = tl.sum(tl.where(unused_cur, 1, 0), axis=0)
        n_dn = tl.sum(tl.where(donor_cur, 1, 0), axis=0)
        n_sp = tl.minimum(n_un, n_dn)
        if n_sp <= 0:
            return

        tl.store(n_splits_ptr + m, tl.load(n_splits_ptr + m) + n_sp.to(tl.int64))

        # Principal-boosted `SSE * Usage^gamma` donor Gumbel keys
        sig1_sq = tl.zeros((BLOCK_K,), dtype=tl.float32)
        if TRACK_OJA:
            for d0 in range(0, D_SUB, BLOCK_D):
                cols = d0 + tl.arange(0, BLOCK_D).to(tl.int64)
                mask_d = cols < D_SUB
                pv = tl.load(
                    prin_vec_ptr + k_glob[:, None] * d_sub_i64 + cols[None, :],
                    mask=mask_k[:, None] & mask_d[None, :],
                    other=0.0,
                ).to(tl.float32)
                sig1_sq += tl.sum(pv * pv, axis=1)
            base_dist = 0.5 * tl.maximum(e_unb, 1e-12) + 0.5 * (
                D_SUB.to(tl.float32) * tl.maximum(u_unb, 1e-12) * sig1_sq
            )
        else:
            base_dist = tl.maximum(e_unb, 1e-12)

        u_rel = tl.maximum(u_unb / tl.maximum(b_cur.to(tl.float32), 1.0), 1e-12)
        weighted = base_dist * tl.exp(donor_usage_w * tl.log(u_rel))
        max_e = tl.max(tl.where(mask_k, e_val, 0.0), axis=0)
        score = tl.where(max_e > 0.0, weighted, u_val)

        step_i32 = step_before.to(tl.int32)
        seed_m = base_seed + m.to(tl.int32) * 10007 + step_i32 * 104729
        u_rnd = tl.minimum(tl.maximum(tl.rand(seed_m, offs_k.to(tl.int32)), 1.17549435e-38), 1.0 - 1e-7)
        gumbel = -tl.log(-tl.log(u_rnd))
        donor_keys = tl.where(donor_cur, tl.log(tl.maximum(score, 1e-30)) + gumbel, -float("inf"))

        inv_d = 1.0 / tl.maximum(D_SUB.to(tl.float32), 1.0)
        do_flush = flush_closing != 0
        do_pre = update_pre_close != 0

        for sp_idx in range(n_sp):
            tgt_local = tl.argmin(tl.where(unused_cur, offs_k.to(tl.int32), 999999), axis=0).to(tl.int64)
            src_local = tl.argmax(donor_keys, axis=0).to(tl.int64)
            unused_cur = unused_cur & (offs_k != tgt_local)
            donor_keys = tl.where(offs_k == src_local, -float("inf"), donor_keys)

            src_g = m * ks_i64 + src_local
            tgt_g = m * ks_i64 + tgt_local

            u_src = tl.load(usage_ptr + src_g).to(tl.float32)
            sse_src = tl.minimum(tl.load(sse_ema_ptr + src_g).to(tl.float32), 1e35)
            u_ub = tl.maximum(u_src / u_bias, 1e-6)
            sse_ub = sse_src / s_bias
            sigma_rms = tl.sqrt(tl.maximum((sse_ub / u_ub) * inv_d, 1e-12))

            pv_nsq = tl.zeros((), dtype=tl.float32)
            rad_nsq = tl.zeros((), dtype=tl.float32)
            pv_dot_rad = tl.zeros((), dtype=tl.float32)
            sp_seed = seed_m + sp_idx * 131
            for d0 in range(0, D_SUB, BLOCK_D):
                cols = d0 + tl.arange(0, BLOCK_D).to(tl.int64)
                mask_d = cols < D_SUB
                ur = tl.rand(sp_seed, cols.to(tl.int32))
                radem = tl.where(mask_d, tl.where(ur < 0.5, -1.0, 1.0), 0.0)
                rad_nsq += tl.sum(radem * radem, axis=0)
                if TRACK_OJA:
                    pv = tl.load(prin_vec_ptr + src_g * d_sub_i64 + cols, mask=mask_d, other=0.0).to(tl.float32)
                    pv_nsq += tl.sum(pv * pv, axis=0)
                    pv_dot_rad += tl.sum(pv * radem, axis=0)

            sigma_1 = tl.sqrt(pv_nsq)
            inv_pv = 1.0 / tl.maximum(sigma_1, 1e-12)
            inv_rad = 1.0 / tl.maximum(tl.sqrt(rad_nsq), 1e-12)
            align_sign = tl.where(pv_dot_rad >= 0.0, 1.0, -1.0)
            ratio = sigma_1 / tl.maximum(sigma_rms, 1e-12)
            rho = tl.minimum(tl.maximum((ratio - 0.85) * (1.0 / 0.35), 0.0), 1.0) if TRACK_OJA else 0.0
            cos_vr = tl.abs(pv_dot_rad) * inv_pv * inv_rad
            inv_blend = 1.0 / tl.maximum(
                tl.sqrt(rho * rho + (1.0 - rho) * (1.0 - rho) + 2.0 * rho * (1.0 - rho) * cos_vr),
                1e-12,
            )
            sigma_split = tl.maximum(sigma_1, 0.85 * sigma_rms)
            g_spec = tl.minimum(tl.maximum((ratio - 1.10) * (1.0 / 0.90), 0.0), 1.0)
            eta_min = tl.minimum(split_mult, 0.18)
            eta_eff = eta_min + tl.maximum(split_mult - eta_min, 0.0) * g_spec
            step_mag = tl.maximum(eta_eff * sigma_split, 1e-6)
            rem_sigma = 0.602810275 * sigma_rms

            half_u = 0.5 * u_src
            child_sse = tl.maximum(0.5 * (sse_src - 0.63661977 * (u_ub * step_mag * step_mag) * s_bias), 0.5 * sse_src * 0.36)
            tl.store(usage_ptr + src_g, half_u)
            tl.store(usage_ptr + tgt_g, half_u)
            tl.store(sse_ema_ptr + src_g, child_sse)
            tl.store(sse_ema_ptr + tgt_g, child_sse)
            su_half = 0.5 * tl.load(slow_usage_ptr + src_g).to(tl.float32)
            tl.store(slow_usage_ptr + src_g, su_half)
            tl.store(slow_usage_ptr + tgt_g, su_half)

            if do_flush:
                tl.store(pass_counts_ptr + src_g, 0.0)
                tl.store(pass_counts_ptr + tgt_g, 0.0)
                tl.store(pass_counts_sq_ptr + src_g, 0.0)
                tl.store(pass_counts_sq_ptr + tgt_g, 0.0)
                tl.store(pass_sse_ptr + src_g, 0.0)
                tl.store(pass_sse_ptr + tgt_g, 0.0)

            norm_src = tl.zeros((), dtype=tl.float32)
            norm_tgt = tl.zeros((), dtype=tl.float32)
            amax_src = tl.zeros((), dtype=tl.float32)
            amax_tgt = tl.zeros((), dtype=tl.float32)

            for d0 in range(0, D_SUB, BLOCK_D):
                cols = d0 + tl.arange(0, BLOCK_D).to(tl.int64)
                mask_d = cols < D_SUB
                dc = tl.load(centroids_ptr + src_g * d_sub_i64 + cols, mask=mask_d, other=0.0).to(tl.float32)
                ur = tl.rand(sp_seed, cols.to(tl.int32))
                radem = tl.where(mask_d, tl.where(ur < 0.5, -1.0, 1.0), 0.0)
                if TRACK_OJA:
                    pv = tl.load(prin_vec_ptr + src_g * d_sub_i64 + cols, mask=mask_d, other=0.0).to(tl.float32)
                    u_dir = (rho * (pv * inv_pv) + (1.0 - rho) * align_sign * (radem * inv_rad)) * inv_blend
                else:
                    u_dir = radem * inv_rad
                delta = tl.where(mask_d, step_mag * u_dir, 0.0)
                c_src_new = tl.where(mask_d, dc - delta, 0.0)
                c_tgt_new = tl.where(mask_d, dc + delta, 0.0)

                # Guarantee non-collapsed representation in BF16 / FP8 / FP32
                if REP_MODE == 1:
                    same_bf16 = mask_d & (c_src_new.to(tl.bfloat16).to(tl.float32) == c_tgt_new.to(tl.bfloat16).to(tl.float32))
                    bump = tl.maximum(tl.abs(dc) * 0.015625, 1e-5)
                    c_src_new = tl.where(same_bf16, dc - bump, c_src_new)
                    c_tgt_new = tl.where(same_bf16, dc + bump, c_tgt_new)
                elif REP_MODE == 3:
                    # E4M3FN has 3 mantissa bits (1/8 = 0.125 relative step); ensure distinct quantized codes
                    min_fp8_sep = tl.maximum(tl.abs(dc) * 0.125, 1e-4)
                    too_close = mask_d & (tl.abs(c_tgt_new - c_src_new) < min_fp8_sep)
                    sign_u = tl.where(u_dir >= 0.0, 1.0, -1.0)
                    c_src_new = tl.where(too_close, dc - 0.55 * min_fp8_sep * sign_u, c_src_new)
                    c_tgt_new = tl.where(too_close, dc + 0.55 * min_fp8_sep * sign_u, c_tgt_new)

                tl.store(centroids_ptr + src_g * d_sub_i64 + cols, c_src_new, mask=mask_d)
                tl.store(centroids_ptr + tgt_g * d_sub_i64 + cols, c_tgt_new, mask=mask_d)
                tl.store(slow_centroids_ptr + src_g * d_sub_i64 + cols, c_src_new, mask=mask_d)
                tl.store(slow_centroids_ptr + tgt_g * d_sub_i64 + cols, c_tgt_new, mask=mask_d)
                if do_pre:
                    tl.store(pre_close_ptr + src_g * d_sub_i64 + cols, c_src_new, mask=mask_d)
                    tl.store(pre_close_ptr + tgt_g * d_sub_i64 + cols, c_tgt_new, mask=mask_d)
                if do_flush:
                    tl.store(pass_sums_ptr + src_g * d_sub_i64 + cols, 0.0, mask=mask_d)
                    tl.store(pass_sums_ptr + tgt_g * d_sub_i64 + cols, 0.0, mask=mask_d)
                if TRACK_OJA:
                    ur2 = tl.rand(sp_seed + 19, cols.to(tl.int32))
                    orth = tl.where(mask_d, tl.where(ur2 < 0.5, -1.0, 1.0) * rem_sigma * inv_rad, 0.0)
                    tl.store(prin_vec_ptr + src_g * d_sub_i64 + cols, orth, mask=mask_d)
                    tl.store(prin_vec_ptr + tgt_g * d_sub_i64 + cols, orth, mask=mask_d)

                if REP_MODE == 3:
                    amax_src = tl.maximum(amax_src, tl.max(tl.abs(c_src_new), axis=0))
                    amax_tgt = tl.maximum(amax_tgt, tl.max(tl.abs(c_tgt_new), axis=0))
                elif REP_MODE == 1:
                    qs = c_src_new.to(tl.bfloat16)
                    qt = c_tgt_new.to(tl.bfloat16)
                    tl.store(c_rep_ptr + src_g * d_pad_i64 + cols, qs, mask=mask_d)
                    tl.store(c_rep_ptr + tgt_g * d_pad_i64 + cols, qt, mask=mask_d)
                    qsf = qs.to(tl.float32)
                    qtf = qt.to(tl.float32)
                    norm_src += tl.sum(qsf * qsf, axis=0)
                    norm_tgt += tl.sum(qtf * qtf, axis=0)
                else:
                    tl.store(c_rep_ptr + src_g * d_pad_i64 + cols, c_src_new, mask=mask_d)
                    tl.store(c_rep_ptr + tgt_g * d_pad_i64 + cols, c_tgt_new, mask=mask_d)
                    norm_src += tl.sum(c_src_new * c_src_new, axis=0)
                    norm_tgt += tl.sum(c_tgt_new * c_tgt_new, axis=0)

            if REP_MODE == 3:
                sc_s = tl.maximum(amax_src * (1.00000024 / 448.0), 1.1754943508222875e-38)
                sc_t = tl.maximum(amax_tgt * (1.00000024 / 448.0), 1.1754943508222875e-38)
                qsq_s = tl.zeros((), dtype=tl.float32)
                qsq_t = tl.zeros((), dtype=tl.float32)
                for d0 in range(0, D_PAD, BLOCK_D):
                    cols = d0 + tl.arange(0, BLOCK_D).to(tl.int64)
                    mask_d = cols < D_SUB
                    mask_pad = cols < D_PAD
                    cs = tl.load(centroids_ptr + src_g * d_sub_i64 + cols, mask=mask_d, other=0.0).to(tl.float32)
                    ct = tl.load(centroids_ptr + tgt_g * d_sub_i64 + cols, mask=mask_d, other=0.0).to(tl.float32)
                    qcs = tl.where(mask_d, tl.minimum(tl.maximum(cs / sc_s, -448.0), 448.0), 0.0).to(
                        tl.float8e4nv, fp_downcast_rounding="rtne"
                    )
                    qct = tl.where(mask_d, tl.minimum(tl.maximum(ct / sc_t, -448.0), 448.0), 0.0).to(
                        tl.float8e4nv, fp_downcast_rounding="rtne"
                    )
                    tl.store(c_rep_ptr + src_g * d_pad_i64 + cols, qcs, mask=mask_pad)
                    tl.store(c_rep_ptr + tgt_g * d_pad_i64 + cols, qct, mask=mask_pad)
                    qsf = qcs.to(tl.float32)
                    qtf = qct.to(tl.float32)
                    qsq_s += tl.sum(tl.where(mask_d, qsf * qsf, 0.0), axis=0)
                    qsq_t += tl.sum(tl.where(mask_d, qtf * qtf, 0.0), axis=0)
                tl.store(sc_ptr + src_g, sc_s)
                tl.store(sc_ptr + tgt_g, sc_t)
                tl.store(c_norm_ptr + src_g, (sc_s * sc_s) * qsq_s)
                tl.store(c_norm_ptr + tgt_g, (sc_t * sc_t) * qsq_t)
            else:
                tl.store(c_norm_ptr + src_g, norm_src)
                tl.store(c_norm_ptr + tgt_g, norm_tgt)

    @triton.jit(do_not_specialize=["N", "M", "KS", "D_SUB", "D_PAD", "stride_xn", "stride_xd"])
    def _pq_encode_exact_kernel(
        x_ptr,
        origin64_ptr,
        scales64_ptr,
        c_centered_ptr,
        c_norm_ptr,
        codes_out_ptr,
        N,
        M,
        KS,
        D_SUB,
        D_PAD,
        stride_xn,
        stride_xd,
        OUT_UINT8: tl.constexpr,
        FULL_D: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Exact centered+conditioned IEEE-FP32 subspace encoder `(N, D) -> (N, M)` uint8/int32 codes."""
        pid_n = tl.program_id(0).to(tl.int64)
        pid_m = tl.program_id(1).to(tl.int64)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
        mask_n = offs_n < N

        s_xn = stride_xn.to(tl.int64)
        s_xd = stride_xd.to(tl.int64)
        m_i64 = M.to(tl.int64)
        ks_i64 = KS.to(tl.int64)
        d_sub_i64 = D_SUB.to(tl.int64)
        d_pad_i64 = D_PAD.to(tl.int64)
        sc64 = tl.load(scales64_ptr + pid_m).to(tl.float64)

        best_score = tl.full((BLOCK_N,), float("inf"), dtype=tl.float32)
        best_lbl = tl.zeros((BLOCK_N,), dtype=tl.int32)

        if FULL_D:
            cols = tl.arange(0, BLOCK_D).to(tl.int64)
            mask_d = cols < D_SUB
            orig64 = tl.load(origin64_ptr + pid_m * d_sub_i64 + cols, mask=mask_d, other=0.0).to(tl.float64)
            x_raw = tl.load(
                x_ptr + offs_n[:, None] * s_xn + (pid_m * d_sub_i64 + cols[None, :]) * s_xd,
                mask=mask_n[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float64)
            xc64 = tl.minimum(tl.maximum((x_raw - orig64[None, :]) * sc64, -1e18), 1e18)
            x_c = tl.where(mask_n[:, None] & mask_d[None, :], xc64.to(tl.float32), 0.0)

            for k0 in range(0, KS, BLOCK_K):
                offs_k = k0 + tl.arange(0, BLOCK_K).to(tl.int64)
                mask_k = offs_k < KS
                c_tile = tl.load(
                    c_centered_ptr + (pid_m * ks_i64 + offs_k[:, None]) * d_pad_i64 + cols[None, :],
                    mask=mask_k[:, None] & mask_d[None, :],
                    other=0.0,
                ).to(tl.float32)
                cn = tl.load(c_norm_ptr + pid_m * ks_i64 + offs_k, mask=mask_k, other=float("inf")).to(tl.float32)
                dots = tl.dot(x_c, tl.trans(c_tile), input_precision="ieee", out_dtype=tl.float32)
                scores = tl.where(mask_k[None, :], cn[None, :] - 2.0 * dots, float("inf"))
                t_min, t_idx = tl.min(scores, axis=1, return_indices=True)
                better = t_min < best_score
                best_score = tl.where(better, t_min, best_score)
                best_lbl = tl.where(better, k0 + t_idx.to(tl.int32), best_lbl)
        else:
            for k0 in range(0, KS, BLOCK_K):
                offs_k = k0 + tl.arange(0, BLOCK_K).to(tl.int64)
                mask_k = offs_k < KS
                dots = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
                for d0 in range(0, D_SUB, BLOCK_D):
                    cols = d0 + tl.arange(0, BLOCK_D).to(tl.int64)
                    mask_d = cols < D_SUB
                    orig64 = tl.load(origin64_ptr + pid_m * d_sub_i64 + cols, mask=mask_d, other=0.0).to(tl.float64)
                    x_raw = tl.load(
                        x_ptr + offs_n[:, None] * s_xn + (pid_m * d_sub_i64 + cols[None, :]) * s_xd,
                        mask=mask_n[:, None] & mask_d[None, :],
                        other=0.0,
                    ).to(tl.float64)
                    xc64 = tl.minimum(tl.maximum((x_raw - orig64[None, :]) * sc64, -1e18), 1e18)
                    x_c = tl.where(mask_n[:, None] & mask_d[None, :], xc64.to(tl.float32), 0.0)
                    c_tile = tl.load(
                        c_centered_ptr + (pid_m * ks_i64 + offs_k[:, None]) * d_pad_i64 + cols[None, :],
                        mask=mask_k[:, None] & mask_d[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    dots = tl.dot(x_c, tl.trans(c_tile), acc=dots, input_precision="ieee", out_dtype=tl.float32)
                cn = tl.load(c_norm_ptr + pid_m * ks_i64 + offs_k, mask=mask_k, other=float("inf")).to(tl.float32)
                scores = tl.where(mask_k[None, :], cn[None, :] - 2.0 * dots, float("inf"))
                t_min, t_idx = tl.min(scores, axis=1, return_indices=True)
                better = t_min < best_score
                best_score = tl.where(better, t_min, best_score)
                best_lbl = tl.where(better, k0 + t_idx.to(tl.int32), best_lbl)

        if OUT_UINT8:
            tl.store(codes_out_ptr + offs_n * m_i64 + pid_m, best_lbl.to(tl.uint8), mask=mask_n)
        else:
            tl.store(codes_out_ptr + offs_n * m_i64 + pid_m, best_lbl.to(tl.int32), mask=mask_n)

    @triton.jit(do_not_specialize=["Q", "N", "M", "KS", "SPLIT_N", "chunk_n"])
    def _pq_lut_adc_splitn_kernel(
        lut_ptr,            # (Q, M, KS) float32 contiguous
        codes_ptr,          # (N, M) uint8 or int32 contiguous
        out_keys_ptr,       # (Q, SPLIT_N * BLOCK_CAP) int64/uint64 packed keys
        Q,
        N,
        M,
        KS,
        SPLIT_N,
        chunk_n,
        M_CONST: tl.constexpr,
        BLOCK_CAP: tl.constexpr,
    ):
        """Blazing-fast L1-LUT + L2-Codes Split-N ADC Top-K kernel (`tl.topk` gated by `cur_kth_bits`)."""
        pid_q = tl.program_id(0).to(tl.int64)
        pid_s = tl.program_id(1).to(tl.int64)
        if pid_q >= Q:
            return

        m_i64 = M.to(tl.int64)
        ks_i64 = KS.to(tl.int64)
        split_i64 = SPLIT_N.to(tl.int64)
        chunk_i64 = chunk_n.to(tl.int64)
        n_i64 = N.to(tl.int64)

        n_start = pid_s * chunk_i64
        n_end = tl.minimum(n_start + chunk_i64, n_i64)
        offs_cap = tl.arange(0, BLOCK_CAP).to(tl.int64)
        best_keys = tl.full((BLOCK_CAP,), 0x7F800000FFFFFFFF, dtype=tl.uint64)
        cur_kth_bits = tl.full((), 0x7F800000, dtype=tl.uint32)

        lut_q_base = lut_ptr + pid_q * m_i64 * ks_i64
        for n0 in range(n_start, n_end, BLOCK_CAP):
            offs_n = n0 + offs_cap
            mask_n = offs_n < n_end
            acc = tl.zeros((BLOCK_CAP,), dtype=tl.float32)
            code_base = codes_ptr + offs_n * m_i64
            if M_CONST > 0:
                for m in tl.static_range(M_CONST):
                    cm = tl.load(code_base + m, mask=mask_n, other=0).to(tl.int64)
                    dm = tl.load(lut_q_base + m * ks_i64 + cm, mask=mask_n, other=0.0).to(tl.float32)
                    acc += dm
            else:
                for m in range(M):
                    cm = tl.load(code_base + m, mask=mask_n, other=0).to(tl.int64)
                    dm = tl.load(lut_q_base + m * ks_i64 + cm, mask=mask_n, other=0.0).to(tl.float32)
                    acc += dm
            # Canonical `+0.0` avoids `-0.0` (`0x80000000`) corrupting unsigned bitcast ordering
            acc = tl.where(acc > 0.0, acc, 0.0)
            d_bits32 = tl.where(mask_n, acc.to(tl.uint32, bitcast=True), 0x7F800000)
            if tl.min(d_bits32, axis=0) <= cur_kth_bits:
                d_bits = d_bits32.to(tl.uint64)
                id_u = tl.where(mask_n, offs_n.to(tl.uint32).to(tl.uint64), 0xFFFFFFFF)
                tile_keys = tl.where(mask_n, (d_bits << 32) | id_u, 0x7F800000FFFFFFFF)
                merged = tl.join(best_keys, tile_keys).reshape((2 * BLOCK_CAP,))
                best_keys = tl.topk(merged, BLOCK_CAP, descending=False)
                cur_kth_bits = (tl.max(best_keys, axis=0) >> 32).to(tl.uint32)

        out_base = (pid_q * split_i64 + pid_s) * BLOCK_CAP
        tl.store(out_keys_ptr + out_base + offs_cap, best_keys)

    @triton.jit(do_not_specialize=["N", "K1", "D", "D_PAD", "stride_xn", "stride_xd"])
    def _rvq_fused_assign_residual_kernel(
        x_ptr,              # (N, D) original tensor (arbitrary stride) for exact FP64 top-2 refine & residual
        xc_bf16_ptr,        # (N, D_PAD) pre-centered bfloat16 contiguous
        c1_raw_ptr,         # (K1, D) float32 contiguous
        c1_bf16_ptr,        # (K1, D_PAD) bfloat16 centered contiguous
        c1_norm_ptr,        # (K1,) float32 represented norm
        labels_out_ptr,     # (N,) int32
        res_out_ptr,        # (N, D) float32 contiguous `X - C1[labels_1]` (differenced in FP64!)
        N,
        K1,
        D,
        D_PAD,
        stride_xn,
        stride_xd,
        WRITE_RES: tl.constexpr,
        FULL_D: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """SM120 BF16 Tensor-Core Top-2 Filter + Exact FP64 Top-2 Refinement + Fused FP64 Residual."""
        pid_n = tl.program_id(0).to(tl.int64)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
        mask_n = offs_n < N

        s_xn = stride_xn.to(tl.int64)
        s_xd = stride_xd.to(tl.int64)
        d_i64 = D.to(tl.int64)
        d_pad_i64 = D_PAD.to(tl.int64)

        best0_s = tl.full((BLOCK_N,), float("inf"), dtype=tl.float32)
        best0_l = tl.zeros((BLOCK_N,), dtype=tl.int32)
        best1_s = tl.full((BLOCK_N,), float("inf"), dtype=tl.float32)
        best1_l = tl.zeros((BLOCK_N,), dtype=tl.int32)

        if FULL_D:
            cols_full = tl.arange(0, BLOCK_D).to(tl.int64)
            mask_pad = cols_full < D_PAD
            xc_reg = tl.load(
                xc_bf16_ptr + offs_n[:, None] * d_pad_i64 + cols_full[None, :],
                mask=mask_n[:, None] & mask_pad[None, :],
                other=0.0,
            )
            for k0 in range(0, K1, BLOCK_K):
                offs_k = k0 + tl.arange(0, BLOCK_K).to(tl.int64)
                mask_k = offs_k < K1
                ct = tl.load(
                    c1_bf16_ptr + offs_k[:, None] * d_pad_i64 + cols_full[None, :],
                    mask=mask_k[:, None] & mask_pad[None, :],
                    other=0.0,
                )
                dots = tl.dot(xc_reg, tl.trans(ct), out_dtype=tl.float32)
                cn = tl.load(c1_norm_ptr + offs_k, mask=mask_k, other=float("inf")).to(tl.float32)
                scores = tl.where(mask_k[None, :], cn[None, :] - 2.0 * dots, float("inf"))

                t0_min, t0_idx = tl.min(scores, axis=1, return_indices=True)
                scores_no0 = tl.where(tl.arange(0, BLOCK_K)[None, :] == t0_idx[:, None], float("inf"), scores)
                t1_min, t1_idx = tl.min(scores_no0, axis=1, return_indices=True)
                l0 = k0 + t0_idx.to(tl.int32)
                l1 = k0 + t1_idx.to(tl.int32)

                # Merge `(t0_min, l0)` and `(t1_min, l1)` into `(best0, best1)`
                b0_better = t0_min < best0_s
                new_b1_s = tl.where(b0_better, tl.minimum(best0_s, t1_min), tl.minimum(best1_s, t0_min))
                new_b1_l = tl.where(
                    b0_better,
                    tl.where(t1_min < best0_s, l1, best0_l),
                    tl.where(t0_min < best1_s, l0, best1_l),
                )
                best0_s = tl.where(b0_better, t0_min, best0_s)
                best0_l = tl.where(b0_better, l0, best0_l)
                best1_s = new_b1_s
                best1_l = new_b1_l
        else:
            for k0 in range(0, K1, BLOCK_K):
                offs_k = k0 + tl.arange(0, BLOCK_K).to(tl.int64)
                mask_k = offs_k < K1
                dots = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
                for d0 in range(0, D_PAD, BLOCK_D):
                    cols = d0 + tl.arange(0, BLOCK_D).to(tl.int64)
                    mask_pad = cols < D_PAD
                    xc_t = tl.load(
                        xc_bf16_ptr + offs_n[:, None] * d_pad_i64 + cols[None, :],
                        mask=mask_n[:, None] & mask_pad[None, :],
                        other=0.0,
                    )
                    ct = tl.load(
                        c1_bf16_ptr + offs_k[:, None] * d_pad_i64 + cols[None, :],
                        mask=mask_k[:, None] & mask_pad[None, :],
                        other=0.0,
                    )
                    dots = tl.dot(xc_t, tl.trans(ct), acc=dots, out_dtype=tl.float32)
                cn = tl.load(c1_norm_ptr + offs_k, mask=mask_k, other=float("inf")).to(tl.float32)
                scores = tl.where(mask_k[None, :], cn[None, :] - 2.0 * dots, float("inf"))

                t0_min, t0_idx = tl.min(scores, axis=1, return_indices=True)
                scores_no0 = tl.where(tl.arange(0, BLOCK_K)[None, :] == t0_idx[:, None], float("inf"), scores)
                t1_min, t1_idx = tl.min(scores_no0, axis=1, return_indices=True)
                l0 = k0 + t0_idx.to(tl.int32)
                l1 = k0 + t1_idx.to(tl.int32)

                b0_better = t0_min < best0_s
                new_b1_s = tl.where(b0_better, tl.minimum(best0_s, t1_min), tl.minimum(best1_s, t0_min))
                new_b1_l = tl.where(
                    b0_better,
                    tl.where(t1_min < best0_s, l1, best0_l),
                    tl.where(t0_min < best1_s, l0, best1_l),
                )
                best0_s = tl.where(b0_better, t0_min, best0_s)
                best0_l = tl.where(b0_better, l0, best0_l)
                best1_s = new_b1_s
                best1_l = new_b1_l

        # Exact IEEE-754 FP64 refinement between top-2 candidates `(cand_lo, cand_hi)` sorted ascending by ID
        cand_lo = tl.minimum(best0_l, best1_l).to(tl.int64)
        cand_hi = tl.maximum(best0_l, best1_l).to(tl.int64)
        d2_lo = tl.zeros((BLOCK_N,), dtype=tl.float64)
        d2_hi = tl.zeros((BLOCK_N,), dtype=tl.float64)
        for d0 in range(0, D, BLOCK_D):
            cols = d0 + tl.arange(0, BLOCK_D).to(tl.int64)
            mask_d = cols < D
            mask_nd = mask_n[:, None] & mask_d[None, :]
            xr64 = tl.load(
                x_ptr + offs_n[:, None] * s_xn + cols[None, :] * s_xd,
                mask=mask_nd,
                other=0.0,
            ).to(tl.float64)
            c_lo64 = tl.load(c1_raw_ptr + cand_lo[:, None] * d_i64 + cols[None, :], mask=mask_nd, other=0.0).to(tl.float64)
            c_hi64 = tl.load(c1_raw_ptr + cand_hi[:, None] * d_i64 + cols[None, :], mask=mask_nd, other=0.0).to(tl.float64)
            diff_lo = tl.where(mask_nd, xr64 - c_lo64, 0.0)
            diff_hi = tl.where(mask_nd, xr64 - c_hi64, 0.0)
            d2_lo += tl.sum(diff_lo * diff_lo, axis=1)
            d2_hi += tl.sum(diff_hi * diff_hi, axis=1)

        # Strict `<` preserves smaller ID `cand_lo` on exact ties
        pick_hi = (K1 > 1) & (d2_hi < d2_lo)
        best_lbl64 = tl.where(pick_hi, cand_hi, cand_lo)
        tl.store(labels_out_ptr + offs_n, best_lbl64.to(tl.int32), mask=mask_n)

        if WRITE_RES:
            for d0 in range(0, D, BLOCK_D):
                cols = d0 + tl.arange(0, BLOCK_D).to(tl.int64)
                mask_d = cols < D
                mask_nd = mask_n[:, None] & mask_d[None, :]
                xr64 = tl.load(
                    x_ptr + offs_n[:, None] * s_xn + cols[None, :] * s_xd,
                    mask=mask_nd,
                    other=0.0,
                ).to(tl.float64)
                c1_64 = tl.load(
                    c1_raw_ptr + best_lbl64[:, None] * d_i64 + cols[None, :],
                    mask=mask_nd,
                    other=0.0,
                ).to(tl.float64)
                res_f32 = (xr64 - c1_64).to(tl.float32)
                tl.store(res_out_ptr + offs_n[:, None] * d_i64 + cols[None, :], res_f32, mask=mask_nd)

    @triton.jit(do_not_specialize=["Q", "BEAM", "K1", "K2"])
    def _rvq_csr_beam_adc_kernel(
        lut1_ptr,           # (Q, K1) float32
        lut2_ptr,           # (Q, K2) float32
        g_cross_ptr,        # (K1, K2) float32 `2 * <C1_cent[c1], C2[c2]>`
        probed_c1_ptr,      # (Q, BEAM) int64
        csr_offsets_ptr,    # (K1 + 1,) int64
        csr_ids_ptr,        # (N,) int32
        csr_c2_ptr,         # (N,) int32
        out_keys_ptr,       # (Q, BEAM * BLOCK_CAP) uint64
        Q,
        BEAM,
        K1,
        K2,
        BLOCK_CAP: tl.constexpr,
    ):
        """Zero-sync CSR Coarse-Beam Exact 2-Stage RVQ LUT-ADC Top-K kernel (`tl.topk(uint64)`)."""
        pid_q = tl.program_id(0).to(tl.int64)
        pid_b = tl.program_id(1).to(tl.int64)
        if pid_q >= Q:
            return

        beam_i64 = BEAM.to(tl.int64)
        k1_i64 = K1.to(tl.int64)
        k2_i64 = K2.to(tl.int64)

        c1_id = tl.load(probed_c1_ptr + pid_q * beam_i64 + pid_b).to(tl.int64)
        start = tl.load(csr_offsets_ptr + c1_id).to(tl.int64)
        end = tl.load(csr_offsets_ptr + c1_id + 1).to(tl.int64)
        d1_val = tl.load(lut1_ptr + pid_q * k1_i64 + c1_id).to(tl.float32)

        offs_cap = tl.arange(0, BLOCK_CAP).to(tl.int64)
        best_keys = tl.full((BLOCK_CAP,), 0x7F800000FFFFFFFF, dtype=tl.uint64)
        cur_kth_bits = tl.full((), 0x7F800000, dtype=tl.uint32)

        lut2_q_base = lut2_ptr + pid_q * k2_i64
        g_c1_base = g_cross_ptr + c1_id * k2_i64

        for pos0 in range(start, end, BLOCK_CAP):
            pos = pos0 + offs_cap
            mask_p = pos < end
            c2_id = tl.load(csr_c2_ptr + pos, mask=mask_p, other=0).to(tl.int64)
            d2_val = tl.load(lut2_q_base + c2_id, mask=mask_p, other=0.0).to(tl.float32)
            gc_val = tl.load(g_c1_base + c2_id, mask=mask_p, other=0.0).to(tl.float32)
            acc = d1_val + d2_val + gc_val
            acc = tl.where(acc > 0.0, acc, 0.0)
            d_bits32 = tl.where(mask_p, acc.to(tl.uint32, bitcast=True), 0x7F800000)
            if tl.min(d_bits32, axis=0) <= cur_kth_bits:
                db_id = tl.load(csr_ids_ptr + pos, mask=mask_p, other=-1).to(tl.uint32).to(tl.uint64)
                tile_keys = tl.where(mask_p, (d_bits32.to(tl.uint64) << 32) | db_id, 0x7F800000FFFFFFFF)
                merged = tl.join(best_keys, tile_keys).reshape((2 * BLOCK_CAP,))
                best_keys = tl.topk(merged, BLOCK_CAP, descending=False)
                cur_kth_bits = (tl.max(best_keys, axis=0) >> 32).to(tl.uint32)

        out_base = (pid_q * beam_i64 + pid_b) * BLOCK_CAP
        tl.store(out_keys_ptr + out_base + offs_cap, best_keys)

    @triton.jit(do_not_specialize=["Q", "N", "K1", "K2", "SPLIT_N", "chunk_n"])
    def _rvq_exhaustive_adc_splitn_kernel(
        lut1_ptr,           # (Q, K1) float32
        lut2_ptr,           # (Q, K2) float32
        g_cross_vec_ptr,    # (N,) float32 pre-gathered `G_cross[c1_i, c2_i]` (or computed on the fly)
        codes_ptr,          # (N, 2) int32 contiguous
        out_keys_ptr,       # (Q, SPLIT_N * BLOCK_CAP) uint64
        Q,
        N,
        K1,
        K2,
        SPLIT_N,
        chunk_n,
        BLOCK_CAP: tl.constexpr,
    ):
        """Exhaustive Exact 2-Stage RVQ Split-N LUT-ADC Top-K kernel (`adc_beam=None`)."""
        pid_q = tl.program_id(0).to(tl.int64)
        pid_s = tl.program_id(1).to(tl.int64)
        if pid_q >= Q:
            return

        n_i64 = N.to(tl.int64)
        k1_i64 = K1.to(tl.int64)
        k2_i64 = K2.to(tl.int64)
        split_i64 = SPLIT_N.to(tl.int64)
        chunk_i64 = chunk_n.to(tl.int64)

        n_start = pid_s * chunk_i64
        n_end = tl.minimum(n_start + chunk_i64, n_i64)
        offs_cap = tl.arange(0, BLOCK_CAP).to(tl.int64)
        best_keys = tl.full((BLOCK_CAP,), 0x7F800000FFFFFFFF, dtype=tl.uint64)
        cur_kth_bits = tl.full((), 0x7F800000, dtype=tl.uint32)

        lut1_q_base = lut1_ptr + pid_q * k1_i64
        lut2_q_base = lut2_ptr + pid_q * k2_i64

        for n0 in range(n_start, n_end, BLOCK_CAP):
            offs_n = n0 + offs_cap
            mask_n = offs_n < n_end
            c1_id = tl.load(codes_ptr + offs_n * 2 + 0, mask=mask_n, other=0).to(tl.int64)
            c2_id = tl.load(codes_ptr + offs_n * 2 + 1, mask=mask_n, other=0).to(tl.int64)
            d1_v = tl.load(lut1_q_base + c1_id, mask=mask_n, other=0.0).to(tl.float32)
            d2_v = tl.load(lut2_q_base + c2_id, mask=mask_n, other=0.0).to(tl.float32)
            gc_v = tl.load(g_cross_vec_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
            acc = d1_v + d2_v + gc_v
            acc = tl.where(acc > 0.0, acc, 0.0)
            d_bits32 = tl.where(mask_p := mask_n, acc.to(tl.uint32, bitcast=True), 0x7F800000)
            if tl.min(d_bits32, axis=0) <= cur_kth_bits:
                id_u = tl.where(mask_p, offs_n.to(tl.uint32).to(tl.uint64), 0xFFFFFFFF)
                tile_keys = tl.where(mask_p, (d_bits32.to(tl.uint64) << 32) | id_u, 0x7F800000FFFFFFFF)
                merged = tl.join(best_keys, tile_keys).reshape((2 * BLOCK_CAP,))
                best_keys = tl.topk(merged, BLOCK_CAP, descending=False)
                cur_kth_bits = (tl.max(best_keys, axis=0) >> 32).to(tl.uint32)

        out_base = (pid_q * split_i64 + pid_s) * BLOCK_CAP
        tl.store(out_keys_ptr + out_base + offs_cap, best_keys)


# ==============================================================================
# Unpacking Helper for Packed IEEE-754 `(dist_u32 << 32) | id_u32` Top-K Keys
# ==============================================================================


def _unpack_topk_uint64_keys(
    raw_keys_u64: torch.Tensor,
    k_top: int,
    n_valid_db: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select smallest `k_top` `(dist, id)` pairs across `(Q, W)` `uint64` keys with zero host syncs.

    Note: PyTorch `torch.topk` operates on signed `int64`. Because `acc >= 0.0` (canonical `+0.0`
    up to `+inf = 0x7F800000`), bit-63 (`sign bit`) is **ALWAYS 0** (`0x00000000..0x7F800000` in
    the upper 32 bits!), so signed `int64` ordering is **100% IDENTICAL** to unsigned `uint64`
    ordering across all non-negative IEEE-754 floats and `+inf` sentinels!
    """
    keys_i64 = raw_keys_u64.view(torch.int64)
    k_eff = min(int(k_top), int(n_valid_db), int(keys_i64.shape[1]))
    if k_eff <= 0:
        q = keys_i64.shape[0]
        return (
            torch.empty((q, 0), dtype=torch.float32, device=keys_i64.device),
            torch.empty((q, 0), dtype=torch.int64, device=keys_i64.device),
        )
    if keys_i64.shape[1] == k_eff:
        top_i64, _ = torch.sort(keys_i64, dim=1)
    else:
        top_i64, _ = torch.topk(keys_i64, k=k_eff, dim=1, largest=False, sorted=True)

    low32 = top_i64 & 0xFFFFFFFF
    valid = low32 != 0xFFFFFFFF
    high32_i32 = ((top_i64 >> 32) & 0x7FFFFFFF).to(torch.int32)
    dists = torch.where(valid, high32_i32.view(torch.float32), torch.full_like(high32_i32, float("inf"), dtype=torch.float32))
    ids = torch.where(valid, low32, torch.full_like(low32, -1))
    return dists, ids


# ==============================================================================
# Vectorized 3D Simplex Empirical-Bayes Finalizer for `(M, Ks, d_sub)`
# ==============================================================================


@torch.no_grad()
def _batched_pq_simplex_eb_finalize(
    pass_sums: torch.Tensor,          # (M, Ks, d_sub) float32
    pass_counts: torch.Tensor,        # (M, Ks) float64
    pass_counts_sq: torch.Tensor,     # (M, Ks) float64
    pass_sse: torch.Tensor,           # (M, Ks) float64
    live_centroids: torch.Tensor,     # (M, Ks, d_sub) float32
    pre_close_centroids: torch.Tensor,# (M, Ks, d_sub) float32
    slow_centroids: torch.Tensor,     # (M, Ks, d_sub) float32
    usage: torch.Tensor,              # (M, Ks) float32
    u_bias: torch.Tensor,             # () or (1,) float64
    alpha_end: torch.Tensor,          # () or (1,) float64
    nepochs: float,
    d_sub: int,
    eb_tau: float = 0.25,
    kappa_max: float = 8.0,
) -> torch.Tensor:
    """Zero-sync vectorized 3D Pre-Closing Independent Anchor + Positive-Part James-Stein EB."""
    counts_w = pass_counts.double()
    nonempty = counts_w > 0.0
    c_close = torch.where(
        nonempty[:, :, None],
        (pass_sums.double() / counts_w.clamp_min(1e-30)[:, :, None]).float(),
        live_centroids.float(),
    )

    n_eff = torch.where(
        nonempty,
        counts_w.square() / pass_counts_sq.double().clamp_min(1e-30),
        torch.zeros_like(counts_w),
    )
    total_w = counts_w.sum(dim=1, keepdim=True).clamp_min(1e-30)
    global_sigma2 = (pass_sse.double().sum(dim=1, keepdim=True) / total_w).clamp_min(0.0)
    raw_sigma2 = torch.where(
        n_eff >= 2.0,
        pass_sse.double() / counts_w.clamp_min(1e-30),
        global_sigma2.expand_as(counts_w),
    ).clamp_min(0.0)
    sigma2_j = (n_eff * raw_sigma2 + 4.0 * global_sigma2) / (n_eff + 4.0)

    if d_sub > 2:
        stein_c = (float(d_sub) - 2.0) / float(d_sub)
    elif d_sub == 2:
        stein_c = 0.50
    else:
        stein_c = 0.35

    v_close = stein_c * sigma2_j / n_eff.clamp_min(1e-30)
    u_inv_bias = 1.0 / u_bias.double().clamp_min(1e-6)
    n_batch_hat = (usage.double() * u_inv_bias).clamp_min(1.0)
    a_end = alpha_end.double().clamp(1e-4, 0.99)
    v_es = (a_end / (2.0 - a_end)) * (sigma2_j / n_batch_hat)

    if nepochs >= 1.25:
        d_ps_sq = (pre_close_centroids.double() - slow_centroids.double()).square().sum(dim=2)
        w_s_in_anc = 0.5 * (v_es / torch.maximum(d_ps_sq, v_es).clamp_min(1e-30)).clamp(0.0, 1.0)
        indep_anchor = (
            (1.0 - w_s_in_anc)[:, :, None] * pre_close_centroids.double()
            + w_s_in_anc[:, :, None] * slow_centroids.double()
        )
        phi_indep = min(1.0, max(0.50, (float(nepochs) - 1.0) / 0.75))
    else:
        indep_anchor = 0.65 * pre_close_centroids.double() + 0.35 * slow_centroids.double()
        phi_indep = 0.55

    d_ci_sq = (c_close.double() - indep_anchor).square().sum(dim=2)
    k_max = max(0.0, float(kappa_max))
    if nepochs <= 3.5:
        k_max_vec = k_max * ((160.0 - n_eff) / 96.0).clamp(0.0, 1.0)
    else:
        k_max_vec = torch.full_like(n_eff, k_max)

    max_indep_shrink = k_max_vec / (n_eff + k_max_vec).clamp_min(1e-30)
    js_indep_shrink = (v_close / d_ci_sq.clamp_min(1e-30)).clamp(0.0, 1.0)
    w_indep = phi_indep * torch.minimum(max_indep_shrink, js_indep_shrink)

    tau = max(0.0, float(eb_tau))
    w_ema = (1.0 - w_indep).clamp_min(0.0) * (1.0 - phi_indep) * (tau / (n_eff + tau).clamp_min(1e-30))

    max_d_ci = torch.where(nonempty, d_ci_sq, 0.0).amax(dim=1, keepdim=True)
    zero_all = (global_sigma2 <= 0.0) & (max_d_ci <= 0.0)
    tot_shrink = torch.where(zero_all, torch.zeros_like(n_eff), (w_indep + w_ema).clamp(0.0, 1.0))
    w_close = 1.0 - tot_shrink

    scale_sh = torch.where(w_indep + w_ema > 1e-30, tot_shrink / (w_indep + w_ema).clamp_min(1e-30), 0.0)
    w_indep = w_indep * scale_sh
    w_ema = w_ema * scale_sh

    c_fin64 = (
        w_close[:, :, None] * c_close.double()
        + w_ema[:, :, None] * live_centroids.double()
        + w_indep[:, :, None] * indep_anchor
    )
    return torch.where(nonempty[:, :, None], c_fin64.float(), live_centroids.float()).contiguous()


# ==============================================================================
# 1. `ApexPQ`: Fused Batched Multi-Subspace Product Quantizer + GPU LUT-ADC
# ==============================================================================


class ApexPQ:
    """Fused Batched Multi-Subspace Product Quantizer (`ApexPQ`) with GPU LUT-ADC Search."""

    def __init__(
        self,
        d: int,
        m: int,
        ks: int = 256,
        nepochs: float = 2.0,
        *,
        solver: str = "apex_emeans",
        precision: str = "auto",          # "auto" | "bf16" | "croiss_fp8" | "fp8" | "fp32"
        batched: bool = True,
        use_cuda_graph: bool = True,
        force_cuda_graph: bool = False,
        graph_super_steps: int = 5,
        batch_law: str = "throughput",    # PQ has `N/Ks >> 100` (`Ks=256`); `"throughput"` saturates SMs cleanly
        batch_size: int | None = None,
        seed: int = 1234,
        solver_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if d < 1 or m < 1 or d % m != 0:
            raise ValueError(f"Dimension d={d} must be >= 1 and divisible by subspaces m={m}")
        if ks < 1 or ks > 65536:
            raise ValueError(f"Subspace codebook size ks={ks} must be in [1, 65536]")
        if not math.isfinite(nepochs) or nepochs <= 0.0:
            raise ValueError(f"nepochs={nepochs} must be finite and > 0")
        if precision not in ("auto", "bf16", "croiss_fp8", "fp8", "fp32"):
            raise ValueError(f"Unsupported precision={precision!r}")
        if batch_law not in ("auto", "throughput", "quality"):
            raise ValueError(f"Unsupported batch_law={batch_law!r}")
        if graph_super_steps < 1:
            raise ValueError("graph_super_steps must be >= 1")

        self.d = int(d)
        self.m = int(m)
        self.d_sub = int(d // m)
        self.ks = int(ks)
        self.nbits = max(1, int(math.ceil(math.log2(max(2, self.ks)))))
        self.bytes_per_vec = int(self.m) if self.ks <= 256 else int(math.ceil(self.m * self.nbits / 8.0))
        self.compression_ratio_vs_fp32 = float(4 * self.d) / float(max(1, self.bytes_per_vec))
        self.nepochs = float(nepochs)

        self.solver_name = normalize_solver_name(solver)
        if self.solver_name not in PQ_SUPPORTED_SOLVERS:
            raise ValueError(f"Unsupported PQ solver {solver!r}")
        if self.solver_name == "apex_emeans_fp8" and precision == "auto":
            self.precision = "croiss_fp8"
        elif self.solver_name == "apex_emeans_bf16" and precision == "auto":
            self.precision = "bf16"
        else:
            self.precision = precision

        self.batched = bool(batched) and (self.solver_name != "apex_emeans_subspace")
        self.use_cuda_graph = bool(use_cuda_graph)
        self.force_cuda_graph = bool(force_cuda_graph)
        self.graph_super_steps = int(graph_super_steps)
        self.batch_law = batch_law
        self.batch_size = batch_size
        self.seed = int(seed)
        self.solver_kwargs = dict(solver_kwargs or {})

        # Fitted state (populated strictly from `x_train` inside `.fit()`)
        self.codebooks: torch.Tensor | None = None          # (M, Ks, d_sub) float32
        self._c_centered_pad: torch.Tensor | None = None    # (M * Ks, D_PAD) float32
        self._c_centered_norm: torch.Tensor | None = None   # (M * Ks,) float32
        self._subspace_scales64: torch.Tensor | None = None # (M,) float64
        self._global_scale64: torch.Tensor | None = None    # () float64
        self._inv_global_scale_sq64: torch.Tensor | None = None  # () float64
        self._has_nonunit_global_scale: bool = False
        self.train_origin64: torch.Tensor | None = None     # (M, d_sub) float64
        self.train_global_mean64: torch.Tensor | None = None# (1, D) float64
        self.train_mean_power: float = 0.0
        self.train_var_power: float = 0.0
        self.device: torch.device = torch.device("cpu")

        self.init_time_sec: float = 0.0
        self.preencode_time_sec: float = 0.0
        self.train_time_sec: float = 0.0
        self.total_time_sec: float = 0.0
        self.encode_time_sec: float = 0.0
        self.n_splits: int = 0
        self.empty_clusters: int = 0
        self.resolved_precision: str = "unfitted"
        self.selected_gear: str = "unfitted"
        self._faiss_index: Any = None

    def _resolve_pq_rep_mode(self, device: torch.device, x_dtype: torch.dtype) -> tuple[int, str, int]:
        """Return `(rep_mode, prec_label, d_pad)` where `rep_mode` in `{0: fp32, 1: bf16, 3: croiss_fp8}`."""
        if device.type != "cuda" or not HAS_TRITON or self.precision == "fp32":
            d_pad = max(16, 1 << max(0, (self.d_sub - 1).bit_length()))
            return 0, "fp32", d_pad
        if self.precision in ("croiss_fp8", "fp8"):
            if _has_sm89_fp8(device):
                d_pad = max(32, 1 << max(0, (self.d_sub - 1).bit_length()))
                return 3, "croiss_fp8", d_pad
            d_pad = max(16, 1 << max(0, (self.d_sub - 1).bit_length()))
            return (1, "bf16", d_pad) if torch.cuda.get_device_capability(device) >= (8, 0) else (0, "fp32", d_pad)
        if self.precision == "bf16":
            d_pad = max(16, 1 << max(0, (self.d_sub - 1).bit_length()))
            return (1, "bf16", d_pad) if torch.cuda.get_device_capability(device) >= (8, 0) else (0, "fp32", d_pad)

        # `precision == "auto"`:
        # For `d_sub <= 32, Ks <= 256`, 1-pass BF16 Tensor-Core avoids FP8 pre-encode pass and E4M3 small-d floor;
        # for wide subspaces (`d_sub >= 64`) on large datasets, engage `CroissAttn`-FP8!
        if x_dtype == torch.float64:
            d_pad = max(16, 1 << max(0, (self.d_sub - 1).bit_length()))
            return 0, "fp32", d_pad
        if _has_sm89_fp8(device) and self.d_sub >= 64 and self.ks >= 256:
            d_pad = max(32, 1 << max(0, (self.d_sub - 1).bit_length()))
            return 3, "croiss_fp8", d_pad
        if torch.cuda.get_device_capability(device) >= (8, 0):
            d_pad = max(16, 1 << max(0, (self.d_sub - 1).bit_length()))
            return 1, "bf16", d_pad
        d_pad = max(16, 1 << max(0, (self.d_sub - 1).bit_length()))
        return 0, "fp32", d_pad

    def _cache_centered_codebooks(self, codebooks: torch.Tensor, origin64: torch.Tensor) -> None:
        """Cache centered & conditioned `(M * Ks, D_PAD)` codebooks for fast zero-leakage GPU encode & LUT."""
        dev = codebooks.device
        if self._subspace_scales64 is None:
            self._subspace_scales64 = torch.ones((self.m,), dtype=torch.float64, device=dev)
        elif self._subspace_scales64.device != dev:
            self._subspace_scales64 = self._subspace_scales64.to(dev)
        if self._global_scale64 is None:
            self._global_scale64 = torch.tensor(1.0, dtype=torch.float64, device=dev)
            self._inv_global_scale_sq64 = torch.tensor(1.0, dtype=torch.float64, device=dev)
            self._has_nonunit_global_scale = False
        elif self._global_scale64.device != dev:
            self._global_scale64 = self._global_scale64.to(dev)
            assert self._inv_global_scale_sq64 is not None
            self._inv_global_scale_sq64 = self._inv_global_scale_sq64.to(dev)

        d_pad = max(16, 1 << max(0, (self.d_sub - 1).bit_length()))
        c_cent = (
            (codebooks.double() - origin64[:, None, :]) * self._subspace_scales64[:, None, None]
        ).clamp(-1e18, 1e18).float()
        c_pad = torch.zeros((self.m * self.ks, d_pad), dtype=torch.float32, device=dev)
        c_flat = c_cent.reshape(self.m * self.ks, self.d_sub)
        c_pad[:, : self.d_sub] = c_flat
        self._c_centered_pad = c_pad.contiguous()
        self._c_centered_norm = c_flat.square().sum(dim=1).contiguous()

    def _record_train_signal_stats(self, x_t: torch.Tensor) -> None:
        """Compute exact FP64 training-set origin, all-row scales, & signal/variance power (0 test leakage)."""
        n = int(x_t.shape[0])
        dev = x_t.device
        sum_x = torch.zeros((self.d,), dtype=torch.float64, device=dev)
        sum_sq = torch.zeros((), dtype=torch.float64, device=dev)
        for s in range(0, n, 131072):
            xb = x_t[s : min(s + 131072, n)].double()
            sum_x += xb.sum(dim=0)
            sum_sq += xb.square().sum()
        mean64 = (sum_x / float(n)).reshape(1, self.d)
        orig_m = mean64.reshape(self.m, self.d_sub).contiguous()

        # Scan all `n` training rows for exact per-subspace and global dynamic-range conditioning scales
        max_abs_m = torch.zeros((self.m,), dtype=torch.float64, device=dev)
        for s in range(0, n, 131072):
            e = min(s + 131072, n)
            xb_m = x_t[s:e].double().reshape(e - s, self.m, self.d_sub)
            blk_m = (xb_m - orig_m[None, :, :]).abs().amax(dim=(0, 2))
            max_abs_m = torch.maximum(max_abs_m, blk_m)

        sub_scales = _power_of_two_coord_scales_64(max_abs_m).contiguous()
        glob_scale = _power_of_two_coord_scales_64(max_abs_m.amax().view(1))[0]
        g_val = float(glob_scale.item())

        mean_power = float((sum_sq / float(n)).item())
        mean_norm_sq = float(mean64.square().sum().item())
        var_power = max(mean_power - mean_norm_sq, 0.0)
        self.train_global_mean64 = mean64
        self.train_origin64 = orig_m
        self._subspace_scales64 = sub_scales
        self._global_scale64 = glob_scale
        self._inv_global_scale_sq64 = 1.0 / (glob_scale * glob_scale)
        self._has_nonunit_global_scale = g_val != 1.0
        self.train_mean_power = mean_power
        self.train_var_power = var_power

    @torch.no_grad()
    def fit(
        self,
        x_train: torch.Tensor | np.ndarray,
        *,
        capture_codes: bool = False,
    ) -> tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray | None]:
        """Train all `M` subspace codebooks strictly on `x_train` (`(N, D)`)."""
        input_is_numpy = isinstance(x_train, np.ndarray)
        if input_is_numpy:
            x_np = np.asarray(x_train)
            if x_np.ndim != 2:
                raise ValueError(f"Expected 2D x_train, got shape {x_np.shape}")
            target_dev = (
                torch.device(f"cuda:{torch.cuda.current_device()}")
                if (torch.cuda.is_available() and self.solver_name not in ("faiss_pq", "faiss_pq_sub"))
                else torch.device("cpu")
            )
            x_t = torch.from_numpy(
                np.ascontiguousarray(x_np, dtype=np.float64 if x_np.dtype == np.float64 else np.float32)
            ).to(device=target_dev)
        else:
            if x_train.ndim != 2:
                raise ValueError(f"Expected 2D x_train tensor, got shape {tuple(x_train.shape)}")
            x_t = x_train.detach()

        n, d = int(x_t.shape[0]), int(x_t.shape[1])
        if d != self.d:
            raise ValueError(f"Expected dimension d={self.d}, got {d}")
        if n < self.ks:
            raise ValueError(f"Number of training vectors n={n} must be >= ks={self.ks}")
        if not bool(torch.isfinite(x_t).all()):
            raise ValueError("Input x_train must contain only finite values (no NaN or Inf)")

        self.n_splits = 0
        self.empty_clusters = 0
        self.preencode_time_sec = 0.0
        self._faiss_index = None

        # Reference Path 1: `faiss.IndexPQ` (`faiss_pq` = full N de-confounded; `faiss_pq_sub` = stock 256*Ks subsample)
        if self.solver_name in ("faiss_pq", "faiss_pq_sub"):
            if not HAS_FAISS:
                raise RuntimeError("faiss is not installed in this environment")
            if (1 << self.nbits) != self.ks:
                raise ValueError(f"faiss.IndexPQ requires ks to be an exact power of 2, got ks={self.ks}")
            x_cpu = np.ascontiguousarray(x_t.detach().cpu().float().numpy(), dtype=np.float32)
            if x_t.device.type == "cuda":
                dev_eval = x_t.device
            elif torch.cuda.is_available():
                dev_eval = torch.device(f"cuda:{torch.cuda.current_device()}")
            else:
                dev_eval = torch.device("cpu")
            ctx_ev = torch.cuda.device(dev_eval) if dev_eval.type == "cuda" else contextlib.nullcontext()
            with ctx_ev:
                x_gpu_for_stats = x_t.to(dev_eval) if dev_eval.type == "cuda" else x_t
                self._record_train_signal_stats(x_gpu_for_stats)

            default_max_ppc = 256 if self.solver_name == "faiss_pq_sub" else max(256, (n + self.ks - 1) // self.ks)
            max_ppc = int(self.solver_kwargs.get("max_points_per_centroid", default_max_ppc))

            t0 = time.perf_counter()
            idx_pq = faiss.IndexPQ(self.d, self.m, self.nbits)
            niter = int(self.solver_kwargs.get("niter", 25))
            idx_pq.pq.cp.niter = niter
            idx_pq.pq.cp.max_points_per_centroid = max_ppc
            idx_pq.pq.cp.seed = int(self.seed)
            idx_pq.train(x_cpu)
            t_train = time.perf_counter() - t0

            cb_np = faiss.vector_to_array(idx_pq.pq.centroids).reshape(self.m, self.ks, self.d_sub).copy()
            self._faiss_index = idx_pq
            self.device = dev_eval
            with ctx_ev:
                self.codebooks = torch.from_numpy(cb_np).to(device=dev_eval, dtype=torch.float32).contiguous()
                assert self.train_origin64 is not None
                if self.train_origin64.device != dev_eval:
                    self.train_origin64 = self.train_origin64.to(dev_eval)
                self._cache_centered_codebooks(self.codebooks, self.train_origin64)
            self.init_time_sec = 0.0
            self.train_time_sec = float(t_train)
            self.total_time_sec = float(t_train)
            self.resolved_precision = "fp32_cpu_faiss"
            mode_tag = "sub256" if self.solver_name == "faiss_pq_sub" else "fullN"
            self.selected_gear = f"faiss_indexpq_{mode_tag}_niter{niter}"

            codes_out = self.encode(x_train) if capture_codes else None
            cb_ret: torch.Tensor | np.ndarray = cb_np if input_is_numpy else self.codebooks
            return cb_ret, codes_out

        device = x_t.device
        self.device = device
        self._record_train_signal_stats(x_t)

        # Fast Fused 3D Batched GPU `ApexPQ` Path (`apex_emeans`, `apex_emeans_fp8`, `apex_emeans_bf16`)
        if (
            self.batched
            and self.solver_name in ("apex_emeans", "apex_emeans_fp8", "apex_emeans_bf16")
            and device.type == "cuda"
            and HAS_TRITON
            and self.ks <= 512
        ):
            with torch.cuda.device(device):
                cb = self._fit_batched_apex_gpu(x_t)
            self.codebooks = cb
            assert self.train_origin64 is not None
            self._cache_centered_codebooks(cb, self.train_origin64)
            codes_out = self.encode(x_train) if capture_codes else None
            cb_ret = cb.detach().cpu().numpy() if input_is_numpy else cb
            return cb_ret, codes_out

        # De-confounded Batched 3D GPU `Lloyd@25` Path (`lloyd_25_batched`)
        if self.solver_name == "lloyd_25_batched" and device.type == "cuda" and HAS_TRITON and self.ks <= 512:
            with torch.cuda.device(device):
                cb = self._fit_batched_lloyd_gpu(x_t, niter=int(self.solver_kwargs.get("niter", 25)))
            self.codebooks = cb
            assert self.train_origin64 is not None
            self._cache_centered_codebooks(cb, self.train_origin64)
            codes_out = self.encode(x_train) if capture_codes else None
            cb_ret = cb.detach().cpu().numpy() if input_is_numpy else cb
            return cb_ret, codes_out

        # Pluggable Per-Subspace Solver Path (`ultra_emeans`, `hybrid_semeans`, `emeans_pp`, `smeans`, `emeans`, `lloyd_25`, `ks > 512`, CPU fallback)
        cb = self._fit_pluggable_subspace_solvers(x_t)
        self.codebooks = cb
        assert self.train_origin64 is not None
        self._cache_centered_codebooks(cb, self.train_origin64)
        codes_out = self.encode(x_train) if capture_codes else None
        cb_ret = cb.detach().cpu().numpy() if input_is_numpy else cb
        return cb_ret, codes_out

    @torch.no_grad()
    def _fit_pluggable_subspace_solvers(self, x_t: torch.Tensor) -> torch.Tensor:
        """Fit any pluggable single-space solver across all `M` subspaces with exact timing accounting."""
        device = x_t.device
        assert self.train_origin64 is not None and self._subspace_scales64 is not None
        cb = torch.empty((self.m, self.ks, self.d_sub), dtype=torch.float32, device=device)

        init_acc = 0.0
        train_acc = 0.0
        pre_acc = 0.0
        splits_acc = 0
        target_solver = "lloyd_25" if self.solver_name == "lloyd_25_batched" else self.solver_name
        ctx = torch.cuda.device(device) if device.type == "cuda" else contextlib.nullcontext()

        with ctx:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t_wall0 = time.perf_counter()

            for m_idx in range(self.m):
                sc_m = float(self._subspace_scales64[m_idx].item())
                orig_m = self.train_origin64[m_idx : m_idx + 1]
                x_slice = x_t[:, m_idx * self.d_sub : (m_idx + 1) * self.d_sub]
                if sc_m != 1.0:
                    x_sub = ((x_slice.double() - orig_m) * sc_m).clamp(-1e18, 1e18).float().contiguous()
                else:
                    x_sub = x_slice.float().contiguous()
                sub_seed = int(self.seed) + m_idx * 10007
                sol = instantiate_subproblem_solver(
                    target_solver,
                    d=self.d_sub,
                    k=self.ks,
                    nepochs=self.nepochs,
                    seed=sub_seed,
                    solver_kwargs=self.solver_kwargs,
                )
                c_m, _ = sol.fit(x_sub, capture_assignments=False)
                c_m_t = (
                    torch.from_numpy(c_m).to(device=device, dtype=torch.float32)
                    if isinstance(c_m, np.ndarray)
                    else c_m.to(device=device, dtype=torch.float32)
                )
                if sc_m != 1.0:
                    cb[m_idx] = (c_m_t.double() / sc_m + orig_m).float()
                else:
                    cb[m_idx] = c_m_t
                init_acc += float(getattr(sol, "init_time_sec", 0.0))
                train_acc += float(getattr(sol, "train_time_sec", 0.0))
                pre_acc += float(getattr(sol, "preencode_time_sec", 0.0))
                splits_acc += int(getattr(sol, "n_splits", 0))

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            wall_tot = time.perf_counter() - t_wall0

        self.init_time_sec = init_acc
        self.preencode_time_sec = pre_acc
        self.train_time_sec = max(0.0, wall_tot - init_acc)
        self.total_time_sec = wall_tot
        self.n_splits = splits_acc
        self.resolved_precision = "bf16" if device.type == "cuda" else "fp32"
        self.selected_gear = f"subspace_{target_solver}"
        return cb.contiguous()

    @torch.no_grad()
    def _fit_batched_lloyd_gpu(self, x_t: torch.Tensor, niter: int = 25) -> torch.Tensor:
        """De-confounded 3D Batched GPU `Lloyd@25` across all `M` subspaces simultaneously."""
        device = x_t.device
        n = int(x_t.shape[0])
        assert self.train_origin64 is not None and self._subspace_scales64 is not None

        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        scales64 = self._subspace_scales64

        gen = torch.Generator(device=device)
        gen.manual_seed(int(self.seed) % (2**31 - 1))
        perm = torch.arange(n, device=device, dtype=torch.int64)
        x_stream = torch.empty((n, self.m, self.d_sub), dtype=torch.float32, device=device)
        dummy_fp8 = torch.empty((1,), dtype=torch.uint8, device=device)
        dummy_f32 = torch.empty((1,), dtype=torch.float32, device=device)
        d_pad = max(16, 1 << max(0, (self.d_sub - 1).bit_length()))
        block_d_pre = min(64, d_pad)

        _pq_prepermute_encode_kernel[(triton.cdiv(n, 128), self.m)](
            x_t,
            perm,
            self.train_origin64,
            scales64,
            x_stream,
            dummy_fp8,
            dummy_f32,
            dummy_f32,
            n,
            self.m,
            self.d_sub,
            d_pad,
            x_t.stride(0),
            x_t.stride(1),
            WRITE_FP8=False,
            BLOCK_N=128,
            BLOCK_D=block_d_pre,
            num_warps=4,
        )

        # Random init per subspace (`M, Ks`)
        rand_scores = torch.rand((self.m, n), generator=gen, device=device, dtype=torch.float32)
        init_idx = torch.topk(rand_scores, k=self.ks, dim=1, largest=False, sorted=False).indices  # (M, Ks)
        m_arange = torch.arange(self.m, device=device, dtype=torch.int64)[:, None]
        c_master = x_stream[init_idx, m_arange, :].clone().contiguous().reshape(self.m * self.ks, self.d_sub)

        torch.cuda.synchronize(device)
        t_init_done = time.perf_counter()

        c_rep = torch.zeros((self.m * self.ks, d_pad), dtype=torch.bfloat16, device=device)
        c_rep[:, : self.d_sub] = c_master.to(torch.bfloat16)
        c_norm = c_rep.float().square().sum(dim=1).contiguous()
        sums = torch.zeros((self.m * self.ks, self.d_sub), dtype=torch.float32, device=device)
        counts = torch.zeros((self.m * self.ks,), dtype=torch.float32, device=device)
        batch_sse = torch.zeros((self.m * self.ks,), dtype=torch.float32, device=device)
        zero_off = torch.zeros((1,), dtype=torch.int64, device=device)
        full_d = self.d_sub <= 64
        blk_b_lloyd = 128 if full_d else 64
        blk_k_lloyd = min(64 if full_d else 32, max(16, 1 << max(0, (self.ks - 1).bit_length())))
        blk_d_lloyd = min(64, d_pad) if full_d else 32
        extra_lloyd = {} if full_d else {"num_stages": 1}

        for it in range(niter):
            sums.zero_()
            counts.zero_()
            batch_sse.zero_()
            _pq_fused_assign_scatter_kernel[(triton.cdiv(n, blk_b_lloyd), self.m)](
                x_stream,
                dummy_fp8,
                dummy_f32,
                dummy_f32,
                zero_off,
                c_master,
                c_rep,
                dummy_f32,
                c_norm,
                c_master,
                sums,
                counts,
                batch_sse,
                sums,
                n,
                n,
                self.m,
                self.ks,
                self.d_sub,
                d_pad,
                REP_MODE=1,
                TRACK_OJA=False,
                FULL_D=full_d,
                BLOCK_B=blk_b_lloyd,
                BLOCK_K=blk_k_lloyd,
                BLOCK_D=blk_d_lloyd,
                num_warps=4,
                **extra_lloyd,
            )
            nonempty = counts > 0.0
            c_master = torch.where(nonempty[:, None], sums / counts.clamp_min(1.0)[:, None], c_master)
            # Split biggest cluster into any empty cluster per subspace
            cnt_m = counts.view(self.m, self.ks)
            if bool((cnt_m == 0.0).any()):
                cm_v = c_master.view(self.m, self.ks, self.d_sub)
                empty_m = cnt_m == 0.0
                for m_i in range(self.m):
                    em_idx = torch.nonzero(empty_m[m_i], as_tuple=False).flatten()
                    if em_idx.numel() > 0:
                        n_em = int(em_idx.numel())
                        donors = torch.topk(cnt_m[m_i], k=min(n_em, self.ks), largest=True).indices
                        for s_i in range(n_em):
                            dn = donors[s_i % donors.numel()]
                            tg = em_idx[s_i]
                            if dn != tg:
                                base_c = cm_v[m_i, dn].clone()
                                eps = 1e-3 * base_c.abs().clamp_min(1e-3)
                                cm_v[m_i, dn] = base_c - eps
                                cm_v[m_i, tg] = base_c + eps
                                self.n_splits += 1
            c_rep.zero_()
            c_rep[:, : self.d_sub] = c_master.to(torch.bfloat16)
            c_norm = c_rep.float().square().sum(dim=1).contiguous()
            _ = it

        cb_out = (
            c_master.view(self.m, self.ks, self.d_sub).double() / scales64[:, None, None] + self.train_origin64[:, None, :]
        ).float().contiguous()
        torch.cuda.synchronize(device)
        t_end = time.perf_counter()

        self.init_time_sec = float(t_init_done - t0)
        self.train_time_sec = float(t_end - t_init_done)
        self.total_time_sec = float(t_end - t0)
        self.resolved_precision = "bf16"
        self.selected_gear = "batched_lloyd_25"
        return cb_out

    @torch.no_grad()
    def _fit_batched_apex_gpu(self, x_t: torch.Tensor) -> torch.Tensor:
        """Fused 3D Batched Multi-Subspace `ApexEMeans` + `CUDAGraph` + `CroissAttn`-FP8 / BF16."""
        device = x_t.device
        n = int(x_t.shape[0])
        assert self.train_origin64 is not None

        rep_mode, prec_name, d_pad = self._resolve_pq_rep_mode(device, x_t.dtype)
        self.resolved_precision = prec_name
        write_fp8 = rep_mode == 3

        # Resolve recipe across `n` and `ks` (cap base `b0 <= 8192` when `ks <= 512` so `2x` wide batch is `<= 16384`)
        eff_batch_law = "throughput" if self.batch_law == "auto" and (n // max(1, self.ks)) >= 128 else self.batch_law
        raw_recipe = build_recipe(
            n=n,
            d=self.d_sub,
            k=self.ks,
            nepochs=self.nepochs,
            batch_law=eff_batch_law,
            batch_size=self.batch_size,
            alpha_min_factor=ALPHA_MIN_FACTOR,
            unused_threshold=DEFAULT_UNUSED_THRESHOLD,
            check_unused_every=CHECK_UNUSED_EVERY,
            split_epsilon=SPLIT_EPSILON,
            init_usage=DEFAULT_INIT_USAGE,
        )
        if self.batch_size is None and eff_batch_law == "throughput" and self.ks <= 512 and raw_recipe.batch_size > 8192:
            recipe = build_recipe(
                n=n,
                d=self.d_sub,
                k=self.ks,
                nepochs=self.nepochs,
                batch_law=eff_batch_law,
                batch_size=8192,
                alpha_min_factor=ALPHA_MIN_FACTOR,
                unused_threshold=DEFAULT_UNUSED_THRESHOLD,
                check_unused_every=CHECK_UNUSED_EVERY,
                split_epsilon=SPLIT_EPSILON,
                init_usage=DEFAULT_INIT_USAGE,
            )
        else:
            recipe = raw_recipe
        # Scale-law wide-batch gear (`#189`: `wide_step_mult = 2` iff `total_steps >= 24 and B < 32768`)
        wide_mult = 2 if (recipe.total_steps >= 24 and recipe.batch_size < 32768 and self.batch_size is None) else 1
        nk_ratio = float(n) / float(max(1, self.ks))
        eff_warm_scale = 0.95 if nk_ratio >= 180.0 else 0.85
        alpha0_base = float(min(max(recipe.alpha0 * eff_warm_scale, 1e-4), 1.0))
        eff_pass_span = 0.20 if (self.nepochs <= 1.25 and nk_ratio >= 128.0) else (0.32 if (self.nepochs <= 3.5 and nk_ratio >= 150.0) else 0.0)

        torch.cuda.synchronize(device)
        t_init0 = time.perf_counter()

        # 1. Per-subspace dynamic-range conditioning (already computed over all `N` rows) + fused pre-permutation
        assert self._subspace_scales64 is not None
        scales64 = self._subspace_scales64

        t_pre0 = time.perf_counter()
        gen_perm = torch.Generator(device=device)
        gen_perm.manual_seed((int(self.seed) * 1000003 + 7) % (2**31 - 1))
        perm0 = block_permute_indices(n, n, 64, gen_perm, device)
        x_stream = torch.empty((n, self.m, self.d_sub), dtype=torch.float32, device=device)
        if write_fp8:
            qx_buf = torch.empty((n, self.m, d_pad), dtype=FLOAT8_E4M3FN, device=device)
            sx_buf = torch.empty((n, self.m), dtype=torch.float32, device=device)
            xn_fp8_buf = torch.empty((n, self.m), dtype=torch.float32, device=device)
        else:
            qx_buf = torch.empty((1,), dtype=torch.uint8, device=device)
            sx_buf = torch.empty((1,), dtype=torch.float32, device=device)
            xn_fp8_buf = torch.empty((1,), dtype=torch.float32, device=device)

        block_d_pre = min(64, d_pad)
        _pq_prepermute_encode_kernel[(triton.cdiv(n, 128), self.m)](
            x_t,
            perm0,
            self.train_origin64,
            scales64,
            x_stream,
            qx_buf,
            sx_buf,
            xn_fp8_buf,
            n,
            self.m,
            self.d_sub,
            d_pad,
            x_t.stride(0),
            x_t.stride(1),
            WRITE_FP8=write_fp8,
            BLOCK_N=128,
            BLOCK_D=block_d_pre,
            num_warps=4,
        )
        self.preencode_time_sec = 0.0  # accounted inside init_time_sec without extra host sync

        # 2. Parallel `M`-Subspace Importance Coreset + Batched `L=3` Greedy `k-means++` + Voronoi Polish
        m_cand = min(n, max(4096, 16 * self.ks))
        x_cand = x_stream[:m_cand]  # (m_cand, M, d_sub), already uniformly block-permuted!

        if self.ks == 1:
            c_init = x_cand.mean(dim=0, keepdim=True).permute(1, 0, 2).contiguous()  # (M, 1, d_sub)
            prior_probs = torch.ones((self.m, 1), dtype=torch.float32, device=device)
            prior_sse = (x_cand - c_init.permute(1, 0, 2)).square().sum(dim=(0, 2), keepdim=False)[:, None] / float(m_cand)
        else:
            p_pool = min(m_cand, max(self.ks, min(1024, 4 * self.ks)))
            block_p = 256 if p_pool <= 256 else (512 if p_pool <= 512 else 1024)
            mu_sub = x_cand.mean(dim=0, keepdim=True)  # (1, M, d_sub)
            r2 = (x_cand - mu_sub).square().sum(dim=2)  # (m_cand, M)
            r2_sum = r2.sum(dim=0, keepdim=True).clamp_min(1e-30)
            q_prob = 0.5 / float(m_cand) + 0.5 * (r2 / r2_sum)  # (m_cand, M)

            gen = torch.Generator(device=device)
            gen.manual_seed((int(self.seed) * 2654435761 + 17) % (2**31 - 1))
            u_g = torch.rand((self.m, m_cand), generator=gen, device=device, dtype=torch.float32).clamp(1e-30, 1.0 - 1e-7)
            gumbel_keys = torch.log(q_prob.t().clamp_min(1e-30)) - torch.log(-torch.log(u_g))
            pool_idx = torch.topk(gumbel_keys, k=p_pool, dim=1, largest=True, sorted=False).indices  # (M, P)

            m_idx_grid = torch.arange(self.m, device=device, dtype=torch.int64)[:, None]
            x_pool = x_cand[pool_idx, m_idx_grid, :].contiguous()  # (M, P, d_sub)
            q_picked = q_prob.t().gather(1, pool_idx)
            w_inv = (1.0 / q_picked.clamp_min(1e-30))
            w_pool = (w_inv / w_inv.sum(dim=1, keepdim=True).clamp_min(1e-30)).contiguous()  # (M, P)

            p_norm = x_pool.square().sum(dim=2)  # (M, P)
            pair_d2 = (p_norm[:, :, None] + p_norm[:, None, :] - 2.0 * torch.bmm(x_pool, x_pool.transpose(1, 2))).clamp_min_(0.0).contiguous()

            # Pick first center per subspace via weighted Gumbel-max
            u0 = torch.rand((self.m, p_pool), generator=gen, device=device, dtype=torch.float32).clamp(1e-30, 1.0 - 1e-7)
            c0_idx = (torch.log(w_pool.clamp_min(1e-30)) - torch.log(-torch.log(u0))).argmax(dim=1)  # (M,)
            init_d2 = pair_d2[torch.arange(self.m, device=device), c0_idx, :].contiguous()  # (M, P)

            sel_idx = torch.empty((self.m, self.ks), dtype=torch.int64, device=device)
            sel_idx[:, 0] = c0_idx
            _pq_batched_greedy_kpp_kernel[(self.m,)](
                pair_d2,
                init_d2,
                w_pool,
                sel_idx,
                p_pool,
                self.ks - 1,
                int(self.seed) & 0x7FFFFFFF,
                BLOCK_P=block_p,
                L_TRIALS=3,
                num_warps=4 if block_p <= 256 else 8,
            )
            c_seed = torch.gather(x_pool, 1, sel_idx[:, :, None].expand(-1, -1, self.d_sub)).contiguous()  # (M, Ks, d_sub)

            # 1-Step Batched Voronoi Polish on `x_cand` (`(m_cand, M, d_sub)`) -> `c_init`, `prior_probs`, `prior_sse`
            xc_m = x_cand.permute(1, 0, 2).contiguous()  # (M, m_cand, d_sub)
            cn_s = c_seed.square().sum(dim=2)            # (M, Ks)
            xn_s = xc_m.square().sum(dim=2)              # (M, m_cand)
            d2_pol = (xn_s[:, :, None] + cn_s[:, None, :] - 2.0 * torch.bmm(xc_m, c_seed.transpose(1, 2))).clamp_min_(0.0)
            min_d2_pol, lbl_pol = d2_pol.min(dim=2)      # (M, m_cand)

            pol_counts = torch.zeros((self.m, self.ks), dtype=torch.float32, device=device)
            pol_counts.scatter_add_(1, lbl_pol, torch.ones_like(min_d2_pol))
            pol_sums = torch.zeros((self.m, self.ks, self.d_sub), dtype=torch.float32, device=device)
            pol_sums.scatter_add_(1, lbl_pol[:, :, None].expand(-1, -1, self.d_sub), xc_m)
            pol_sse = torch.zeros((self.m, self.ks), dtype=torch.float32, device=device)
            pol_sse.scatter_add_(1, lbl_pol, min_d2_pol)

            has_pts = pol_counts > 0.0
            c_init = torch.where(has_pts[:, :, None], pol_sums / pol_counts.clamp_min(1.0)[:, :, None], c_seed).contiguous()
            prior_probs = (pol_counts / float(m_cand)).clamp_min(1e-3 / float(self.ks))
            prior_probs = prior_probs / prior_probs.sum(dim=1, keepdim=True)
            prior_sse = pol_sse / float(m_cand)

        torch.cuda.synchronize(device)
        t_init_done = time.perf_counter()
        self.init_time_sec = float(t_init_done - t_init0)
        _ = t_pre0

        # 3. Allocate 3D Batched Streaming Workspace across `M_KS = M * Ks` clusters
        m_ks = self.m * self.ks
        c_master = c_init.reshape(m_ks, self.d_sub).contiguous()
        slow_centroids = c_master.clone()
        pre_close_centroids = c_master.clone()

        b0 = int(recipe.batch_size)
        init_mass = (0.5 * float(b0) * prior_probs).clamp_min(DEFAULT_INIT_USAGE).reshape(m_ks).contiguous()
        usage = init_mass.clone()
        slow_usage = init_mass.clone()
        sse_ema = (0.5 * float(b0) * prior_sse).clamp_min(1e-12).reshape(m_ks).contiguous()

        # Initialize Oja principal vectors with scaled deterministic Rademacher directions
        inv_d = 1.0 / float(max(1, self.d_sub))
        init_sig = torch.sqrt((sse_ema / usage.clamp_min(1e-6)) * inv_d).clamp_min(1e-4)
        gen_oja = torch.Generator(device=device)
        gen_oja.manual_seed((int(self.seed) + 99991) % (2**31 - 1))
        rad_init = torch.where(
            torch.rand((m_ks, self.d_sub), generator=gen_oja, device=device) < 0.5,
            -1.0,
            1.0,
        ) * math.sqrt(inv_d)
        prin_vec = (rad_init * init_sig[:, None]).contiguous()

        sums = torch.zeros((m_ks, self.d_sub), dtype=torch.float32, device=device)
        counts = torch.zeros((m_ks,), dtype=torch.float32, device=device)
        batch_sse = torch.zeros((m_ks,), dtype=torch.float32, device=device)
        g_sums = torch.zeros((m_ks, self.d_sub), dtype=torch.float32, device=device)

        pass_sums = torch.zeros((m_ks, self.d_sub), dtype=torch.float32, device=device)
        pass_counts = torch.zeros((m_ks,), dtype=torch.float64, device=device)
        pass_counts_sq = torch.zeros((m_ks,), dtype=torch.float64, device=device)
        pass_sse = torch.zeros((m_ks,), dtype=torch.float64, device=device)
        n_splits_m = torch.zeros((self.m,), dtype=torch.int64, device=device)

        if rep_mode == 3:
            c_rep = torch.zeros((m_ks, d_pad), dtype=FLOAT8_E4M3FN, device=device)
            sc_vec = torch.ones((m_ks,), dtype=torch.float32, device=device)
            amax_c = c_master.abs().amax(dim=1).clamp_min(0.0)
            sc_init = (amax_c * (1.00000024 / 448.0)).clamp_min(FP32_TINY)
            q_init = (c_master / sc_init[:, None]).clamp(-448.0, 448.0).to(FLOAT8_E4M3FN)
            c_rep[:, : self.d_sub] = q_init
            sc_vec.copy_(sc_init)
            c_norm = (sc_init.square() * q_init.float().square().sum(dim=1)).contiguous()
        elif rep_mode == 1:
            c_rep = torch.zeros((m_ks, d_pad), dtype=torch.bfloat16, device=device)
            c_rep[:, : self.d_sub] = c_master.to(torch.bfloat16)
            sc_vec = torch.empty((1,), dtype=torch.float32, device=device)
            c_norm = c_rep.float().square().sum(dim=1).contiguous()
        else:
            c_rep = torch.zeros((m_ks, d_pad), dtype=torch.float32, device=device)
            c_rep[:, : self.d_sub] = c_master
            sc_vec = torch.empty((1,), dtype=torch.float32, device=device)
            c_norm = c_master.square().sum(dim=1).contiguous()

        # 2-slot ping-pong state ring (`state_ring[0]`, `state_ring[1]`) eliminates multi-wave CTA races!
        # [step, rows_proc, ep_b_start, u_bias, s_bias, alpha_last]
        state_ring = torch.zeros((2, 6), dtype=torch.float64, device=device)
        state_ring[0, 3] = 0.5  # initial `u_bias` matching prior mass scale 0.5
        state_ring[0, 4] = 0.5  # initial `s_bias`
        state_ring[0, 5] = float(alpha0_base)
        batch_off_ring = torch.zeros((2,), dtype=torch.int64, device=device)
        cur_slot = 0

        total_planned_rows = int(sum(recipe.epoch_rows))
        revive_stop_rows = int(round(0.85 * total_planned_rows))
        eff_grace = 4
        full_d = self.d_sub <= 64
        block_b_ass = 128 if full_d else 64
        block_d_ker = min(64, d_pad) if full_d else 32
        block_k_ass = min(64 if full_d else 32, max(16, 1 << max(0, (self.ks - 1).bit_length())))
        block_k_rev = max(16, 1 << max(0, (self.ks - 1).bit_length()))
        extra_ass_kw = {} if full_d else {"num_stages": 1}
        extra_rev_kw = {"num_stages": 1} if block_k_rev >= 512 else {}
        track_oja = self.d_sub > 1
        use_wide_sched = wide_mult > 1

        def _launch_one_step(
            slot_in: int,
            b_step: int,
            rows_ep: int,
            eff_peak_v: float,
            alpha_slow_v: float,
            accum_p: int,
            do_rev: bool,
            upd_pre: int,
        ) -> int:
            slot_out = 1 - slot_in
            _pq_fused_assign_scatter_kernel[(triton.cdiv(b_step, block_b_ass), self.m)](
                x_stream,
                qx_buf,
                sx_buf,
                xn_fp8_buf,
                batch_off_ring[slot_in : slot_in + 1],
                c_master,
                c_rep,
                sc_vec,
                c_norm,
                prin_vec,
                sums,
                counts,
                batch_sse,
                g_sums,
                b_step,
                n,
                self.m,
                self.ks,
                self.d_sub,
                d_pad,
                REP_MODE=rep_mode,
                TRACK_OJA=track_oja,
                FULL_D=full_d,
                BLOCK_B=block_b_ass,
                BLOCK_K=block_k_ass,
                BLOCK_D=block_d_ker,
                num_warps=4,
                **extra_ass_kw,
            )
            _pq_ema_oja_prep_kernel[(m_ks,)](
                state_ring[slot_in],
                state_ring[slot_out],
                batch_off_ring[slot_in : slot_in + 1],
                batch_off_ring[slot_out : slot_out + 1],
                sums,
                counts,
                batch_sse,
                g_sums,
                c_master,
                usage,
                sse_ema,
                prin_vec,
                slow_centroids,
                slow_usage,
                pass_sums,
                pass_counts,
                pass_counts_sq,
                pass_sse,
                c_rep,
                sc_vec,
                c_norm,
                m_ks,
                self.d_sub,
                d_pad,
                b_step,
                rows_ep,
                recipe.total_steps,
                total_planned_rows,
                eff_peak_v,
                ALPHA_MIN_FACTOR,
                alpha_slow_v,
                eff_pass_span,
                accum_p,
                USE_WIDE_SCHED=use_wide_sched,
                TRACK_OJA=track_oja,
                REP_MODE=rep_mode,
                BLOCK_D=block_d_ker,
                num_warps=1 if block_d_ker <= 32 else 2,
            )
            if do_rev:
                _pq_subspace_revive_kernel[(self.m,)](
                    state_ring[slot_out],
                    c_master,
                    usage,
                    sse_ema,
                    prin_vec,
                    slow_centroids,
                    slow_usage,
                    pre_close_centroids,
                    pass_sums,
                    pass_counts,
                    pass_counts_sq,
                    pass_sse,
                    c_rep,
                    sc_vec,
                    c_norm,
                    n_splits_m,
                    self.m,
                    self.ks,
                    self.d_sub,
                    d_pad,
                    b_step,
                    float(recipe.unused_threshold),
                    0.5,
                    0.35,
                    HALF_GAUSSIAN_MEAN,
                    eff_grace,
                    revive_stop_rows,
                    1 if accum_p else 0,
                    upd_pre,
                    int(self.seed) & 0x7FFFFFFF,
                    TRACK_OJA=track_oja,
                    REP_MODE=rep_mode,
                    BLOCK_K=block_k_rev,
                    BLOCK_D=block_d_ker,
                    num_warps=4 if block_k_rev <= 256 else 8,
                    **extra_rev_kw,
                )
            return slot_out

        # 4. Epoch Loop with Scale-Adaptive Wide-Batch Gear & Amortized Even-Step `CUDAGraph` Replay!
        # Note: An even super-step length `S = 2 * CHECK_UNUSED_EVERY = 10` (or `2`, `4`, `6` if forced)
        # starts on `cur_slot` and returns to `cur_slot` (`10` flips of `1 - slot` = identity!), making
        # `CUDAGraph` replay 100% buffer-address-invariant with zero intermediate copies!
        num_epochs = len(recipe.epoch_rows)
        rows_cum = 0
        cur_b_scale = b0
        rev_step_ctr = 0
        used_graph = False
        cap_stream: torch.cuda.Stream | None = None

        for ep_idx, rows_ep in enumerate(recipe.epoch_rows):
            is_closing = ep_idx == (num_epochs - 1)
            if is_closing:
                pre_close_centroids.copy_(c_master)
                pass_sums.zero_()
                pass_counts.zero_()
                pass_counts_sq.zero_()
                pass_sse.zero_()

            # Reset `ep_b_start = 0.0` in active slot and advance cyclic offset with deterministic epoch stride
            state_ring[cur_slot, 2] = 0.0
            if ep_idx > 0:
                ep_shift = (int(self.seed) * 1000003 + ep_idx * 2654435761) % max(1, n)
                batch_off_ring[cur_slot] = (batch_off_ring[cur_slot] + ep_shift) % max(1, n)

            # Split epoch into Phase A (nimble `B`) and Phase B (wide `2B` once `rows_cum >= 0.25 * total_planned_rows`)
            phases: list[tuple[int, int]] = []  # [(b_phase, n_steps_phase), ...]
            rem_rows = int(rows_ep)
            w_thresh = int(round(0.25 * total_planned_rows))
            if wide_mult > 1 and rows_cum < w_thresh and (rows_cum + rem_rows) > w_thresh:
                rows_nimble = w_thresh - rows_cum
                steps_nimble = max(1, (rows_nimble + b0 - 1) // b0)
                covered = min(rem_rows, steps_nimble * b0)
                phases.append((b0, steps_nimble))
                rem_rows -= covered
                rows_cum += covered

            if rem_rows > 0:
                b_ph = b0 * wide_mult if (wide_mult > 1 and rows_cum >= w_thresh) else b0
                steps_ph = max(1, (rem_rows + b_ph - 1) // b_ph)
                phases.append((b_ph, steps_ph))
                rows_cum += rem_rows

            for b_ph, steps_ph in phases:
                if b_ph != cur_b_scale:
                    b_ratio = float(b_ph) / float(cur_b_scale)
                    usage.mul_(b_ratio)
                    slow_usage.mul_(b_ratio)
                    sse_ema.mul_(b_ratio)
                    cur_b_scale = b_ph

                if b_ph == b0:
                    eff_peak_ph = alpha0_base
                else:
                    eff_peak_ph = float(min(max(compute_step_law(n, self.ks, self.nepochs, b_ph) * eff_warm_scale, 1e-4), 1.0))
                alpha_slow_ph = float(min(max(0.35 * eff_peak_ph, 1e-4), 1.0))
                accum_flag = 1 if is_closing else 0
                upd_pre_flag = 1 if is_closing else 0

                # Super-step of 10 steps (`2 * CHECK_UNUSED_EVERY = 10` -> even number of ping-pong flips
                # AND exact multiple of `CHECK_UNUSED_EVERY = 5` so `rev_offset` stays invariant across replays!)
                super_len = 2 * CHECK_UNUSED_EVERY
                n_supers = steps_ph // super_len
                can_graph = self.use_cuda_graph and (
                    n_supers >= 3 or (self.force_cuda_graph and steps_ph >= 2)
                )

                if can_graph and n_supers >= 1:
                    used_graph = True
                    rev_off = rev_step_ctr % CHECK_UNUSED_EVERY
                    if cap_stream is None:
                        cap_stream = torch.cuda.Stream(device=device)
                    g = torch.cuda.CUDAGraph()
                    cap_stream.wait_stream(torch.cuda.current_stream(device))
                    with torch.cuda.stream(cap_stream):
                        with torch.cuda.graph(g, stream=cap_stream):
                            s_tmp = cur_slot
                            for sub_i in range(super_len):
                                do_r = ((rev_off + sub_i) % CHECK_UNUSED_EVERY) == (CHECK_UNUSED_EVERY - 1)
                                s_tmp = _launch_one_step(
                                    s_tmp, b_ph, rows_ep, eff_peak_ph, alpha_slow_ph, accum_flag, do_r, upd_pre_flag
                                )
                            assert s_tmp == cur_slot
                    torch.cuda.current_stream(device).wait_stream(cap_stream)
                    for _ in range(n_supers):
                        g.replay()
                    g.reset()
                    rev_step_ctr += n_supers * super_len
                    rem_steps = steps_ph - n_supers * super_len
                elif can_graph and self.force_cuda_graph and steps_ph >= 2:
                    used_graph = True
                    even_steps = (steps_ph // 2) * 2
                    rev_off = rev_step_ctr % CHECK_UNUSED_EVERY
                    if cap_stream is None:
                        cap_stream = torch.cuda.Stream(device=device)
                    g = torch.cuda.CUDAGraph()
                    cap_stream.wait_stream(torch.cuda.current_stream(device))
                    with torch.cuda.stream(cap_stream):
                        with torch.cuda.graph(g, stream=cap_stream):
                            s_tmp = cur_slot
                            for sub_i in range(even_steps):
                                do_r = ((rev_off + sub_i) % CHECK_UNUSED_EVERY) == (CHECK_UNUSED_EVERY - 1)
                                s_tmp = _launch_one_step(
                                    s_tmp, b_ph, rows_ep, eff_peak_ph, alpha_slow_ph, accum_flag, do_r, upd_pre_flag
                                )
                            assert s_tmp == cur_slot
                    torch.cuda.current_stream(device).wait_stream(cap_stream)
                    g.replay()
                    g.reset()
                    rev_step_ctr += even_steps
                    rem_steps = steps_ph - even_steps
                else:
                    rem_steps = steps_ph

                for _ in range(rem_steps):
                    do_r = (rev_step_ctr % CHECK_UNUSED_EVERY) == (CHECK_UNUSED_EVERY - 1)
                    cur_slot = _launch_one_step(
                        cur_slot, b_ph, rows_ep, eff_peak_ph, alpha_slow_ph, accum_flag, do_r, upd_pre_flag
                    )
                    rev_step_ctr += 1

                # If T == 1.0 (single epoch), snapshot mid-epoch anchor at ~50% if only 1 epoch exists
                if num_epochs == 1 and len(phases) > 1 and b_ph == b0:
                    pre_close_centroids.copy_(c_master)

        # 5. Vectorized 3D Positive-Part James-Stein Simplex EB Finalization + Uncenter/Unscale in FP64
        c_fin_3d = _batched_pq_simplex_eb_finalize(
            pass_sums=pass_sums.view(self.m, self.ks, self.d_sub),
            pass_counts=pass_counts.view(self.m, self.ks),
            pass_counts_sq=pass_counts_sq.view(self.m, self.ks),
            pass_sse=pass_sse.view(self.m, self.ks),
            live_centroids=c_master.view(self.m, self.ks, self.d_sub),
            pre_close_centroids=pre_close_centroids.view(self.m, self.ks, self.d_sub),
            slow_centroids=slow_centroids.view(self.m, self.ks, self.d_sub),
            usage=usage.view(self.m, self.ks),
            u_bias=state_ring[cur_slot, 3],
            alpha_end=state_ring[cur_slot, 5],
            nepochs=self.nepochs,
            d_sub=self.d_sub,
            eb_tau=0.35,
            kappa_max=8.0,
        )
        cb_out = (
            c_fin_3d.double() / scales64[:, None, None] + self.train_origin64[:, None, :]
        ).float().contiguous()

        torch.cuda.synchronize(device)
        t_end = time.perf_counter()

        self.train_time_sec = float(t_end - t_init_done)
        self.total_time_sec = float(t_end - t_init0)
        self.n_splits = int(n_splits_m.sum().item())
        self.empty_clusters = int((pass_counts == 0.0).sum().item())
        self.selected_gear = (
            f"batched3d_{prec_name}_{'graph' if used_graph else 'eager'}"
            f"{'_wide2x' if wide_mult > 1 else ''}"
        )
        return cb_out

    @torch.no_grad()
    def encode(self, x: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """Encode `(N, D)` vectors into compact `(N, M)` `uint8` (for `Ks <= 256`) or `int32` codes."""
        if self.codebooks is None or self.train_origin64 is None or self._c_centered_pad is None or self._c_centered_norm is None:
            raise RuntimeError("ApexPQ must be fitted via .fit(x_train) before calling .encode(x)")

        input_is_numpy = isinstance(x, np.ndarray)
        x_t = (
            torch.from_numpy(np.ascontiguousarray(x, dtype=np.float64 if x.dtype == np.float64 else np.float32))
            if input_is_numpy
            else x.detach()
        )
        if x_t.ndim != 2 or int(x_t.shape[1]) != self.d:
            raise ValueError(f"Expected (N, {self.d}) input to encode, got {tuple(x_t.shape)}")
        if not bool(torch.isfinite(x_t).all()):
            raise ValueError("Input x to encode must contain only finite values")

        n = int(x_t.shape[0])
        out_dtype = torch.uint8 if self.ks <= 256 else torch.int32
        dev = self.codebooks.device
        if n == 0:
            empty = torch.empty((0, self.m), dtype=out_dtype, device=dev)
            return empty.cpu().numpy() if input_is_numpy else empty

        x_dev = x_t.to(device=dev)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        t0 = time.perf_counter()

        if dev.type == "cuda" and HAS_TRITON:
            with torch.cuda.device(dev):
                assert self._subspace_scales64 is not None
                codes = torch.empty((n, self.m), dtype=out_dtype, device=dev)
                d_pad = int(self._c_centered_pad.shape[1])
                full_d = self.d_sub <= 64
                blk_n_enc = 128 if full_d else 64
                blk_k_enc = min(64 if full_d else 32, max(16, 1 << max(0, (self.ks - 1).bit_length())))
                blk_d_enc = min(64, d_pad) if full_d else 32
                extra_enc = {} if full_d else {"num_stages": 1}
                _pq_encode_exact_kernel[(triton.cdiv(n, blk_n_enc), self.m)](
                    x_dev,
                    self.train_origin64,
                    self._subspace_scales64,
                    self._c_centered_pad,
                    self._c_centered_norm,
                    codes,
                    n,
                    self.m,
                    self.ks,
                    self.d_sub,
                    d_pad,
                    x_dev.stride(0),
                    x_dev.stride(1),
                    OUT_UINT8=(self.ks <= 256),
                    FULL_D=full_d,
                    BLOCK_N=blk_n_enc,
                    BLOCK_K=blk_k_enc,
                    BLOCK_D=blk_d_enc,
                    num_warps=4,
                    **extra_enc,
                )
        else:
            # Exact FP64/FP32 CPU/fallback chunked encoder
            assert self._subspace_scales64 is not None
            codes = torch.empty((n, self.m), dtype=out_dtype, device=dev)
            c64 = (self.codebooks.double() - self.train_origin64[:, None, :]) * self._subspace_scales64[:, None, None]
            cn64 = c64.square().sum(dim=2)
            for s in range(0, n, 16384):
                e = min(s + 16384, n)
                xb = (
                    x_dev[s:e].double().reshape(e - s, self.m, self.d_sub) - self.train_origin64[None, :, :]
                ) * self._subspace_scales64[None, :, None]
                dots = torch.einsum("nmd,mkd->nmk", xb, c64)
                scores = cn64[None, :, :] - 2.0 * dots
                codes[s:e] = scores.argmin(dim=2).to(dtype=out_dtype)

        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        self.encode_time_sec = time.perf_counter() - t0
        return codes.detach().cpu().numpy() if input_is_numpy else codes

    @torch.no_grad()
    def decode(self, codes: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """Reconstruct `(N, D)` float32 vectors from `(N, M)` PQ codes."""
        if self.codebooks is None:
            raise RuntimeError("ApexPQ must be fitted before calling .decode(codes)")
        input_is_numpy = isinstance(codes, np.ndarray)
        c_idx = (
            torch.from_numpy(np.ascontiguousarray(codes)).to(device=self.codebooks.device, dtype=torch.int64)
            if input_is_numpy
            else codes.detach().to(device=self.codebooks.device, dtype=torch.int64)
        )
        if c_idx.ndim != 2 or int(c_idx.shape[1]) != self.m:
            raise ValueError(f"Expected (N, {self.m}) codes, got {tuple(c_idx.shape)}")
        n = int(c_idx.shape[0])
        if n == 0:
            empty = torch.empty((0, self.d), dtype=torch.float32, device=self.codebooks.device)
            return empty.cpu().numpy() if input_is_numpy else empty

        m_arange = torch.arange(self.m, device=self.codebooks.device, dtype=torch.int64)[None, :]
        rec = self.codebooks[m_arange, c_idx, :].reshape(n, self.d).contiguous()
        return rec.detach().cpu().numpy() if input_is_numpy else rec

    @torch.no_grad()
    def compute_lut(self, queries: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Compute exact origin-centered & conditioned `(Q, M, Ks)` float32 squared-distance lookup tables."""
        if self.codebooks is None or self.train_origin64 is None or self._global_scale64 is None:
            raise RuntimeError("ApexPQ must be fitted before calling .compute_lut(queries)")
        q_t = (
            torch.from_numpy(np.ascontiguousarray(queries, dtype=np.float64 if queries.dtype == np.float64 else np.float32))
            if isinstance(queries, np.ndarray)
            else queries.detach()
        )
        if q_t.ndim != 2 or int(q_t.shape[1]) != self.d:
            raise ValueError(f"Expected (Q, {self.d}) queries, got {tuple(q_t.shape)}")
        # Avoid host-blocking `.all()` sync on CUDA (`sync_debug_mode("error")` safe)
        if not q_t.is_cuda and not bool(torch.isfinite(q_t).all()):
            raise ValueError("queries must contain only finite values")

        dev = self.codebooks.device
        q_cnt = int(q_t.shape[0])
        if q_cnt == 0:
            return torch.empty((0, self.m, self.ks), dtype=torch.float32, device=dev)

        ctx = torch.cuda.device(dev) if dev.type == "cuda" else contextlib.nullcontext()
        with ctx:
            g_sc = self._global_scale64
            q_c64 = (
                (q_t.to(device=dev, dtype=torch.float64).reshape(q_cnt, self.m, self.d_sub) - self.train_origin64[None, :, :])
                * g_sc
            ).clamp(-1e18, 1e18)
            c_c64 = ((self.codebooks.double() - self.train_origin64[:, None, :]) * g_sc).clamp(-1e18, 1e18)

            q_m64 = q_c64.permute(1, 0, 2).contiguous()  # (M, Q, d_sub)
            qn64 = q_m64.square().sum(dim=2)             # (M, Q)
            cn64 = c_c64.square().sum(dim=2)             # (M, Ks)
            d2_mqk = (qn64[:, :, None] + cn64[:, None, :] - 2.0 * torch.bmm(q_m64, c_c64.transpose(1, 2))).clamp_min_(0.0).float()
            return d2_mqk.permute(1, 0, 2).contiguous()  # (Q, M, Ks)

    @torch.no_grad()
    def adc_search(
        self,
        queries: torch.Tensor | np.ndarray,
        codes: torch.Tensor | np.ndarray,
        k_top: int = 10,
    ) -> tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray]:
        """Asymmetric Distance Computation (ADC) top-`k_top` search: `(dists, ids)` of shape `(Q, k_eff)`."""
        if self.codebooks is None:
            raise RuntimeError("ApexPQ must be fitted before calling .adc_search(queries, codes)")
        if k_top < 1:
            raise ValueError(f"k_top={k_top} must be >= 1")

        ret_numpy = isinstance(queries, np.ndarray) and isinstance(codes, np.ndarray)
        dev = self.codebooks.device
        lut = self.compute_lut(queries)  # (Q, M, Ks) float32 contiguous
        q_cnt = int(lut.shape[0])

        codes_t = (
            torch.from_numpy(np.ascontiguousarray(codes)).to(device=dev)
            if isinstance(codes, np.ndarray)
            else codes.detach().to(device=dev)
        )
        if codes_t.ndim != 2 or int(codes_t.shape[1]) != self.m:
            raise ValueError(f"Expected (N, {self.m}) codes, got {tuple(codes_t.shape)}")
        n_db = int(codes_t.shape[0])
        k_eff = min(int(k_top), n_db)
        if q_cnt == 0 or k_eff == 0:
            d_e = torch.empty((q_cnt, 0), dtype=torch.float32, device=dev)
            i_e = torch.empty((q_cnt, 0), dtype=torch.int64, device=dev)
            return (d_e.cpu().numpy(), i_e.cpu().numpy()) if ret_numpy else (d_e, i_e)

        codes_c = codes_t.contiguous()
        if dev.type == "cuda" and HAS_TRITON and k_eff <= 128 and n_db < (1 << 31):
            with torch.cuda.device(dev):
                block_cap = 32 if k_eff <= 32 else 128
                split_n = min(16, max(1, triton.cdiv(n_db, 16384)))
                chunk_n = triton.cdiv(n_db, split_n)
                q_batch = max(1, min(q_cnt, 16384))
                dists_out = torch.empty((q_cnt, k_eff), dtype=torch.float32, device=dev)
                ids_out = torch.empty((q_cnt, k_eff), dtype=torch.int64, device=dev)
                m_const = self.m if self.m <= 32 else 0

                for q0 in range(0, q_cnt, q_batch):
                    q1 = min(q0 + q_batch, q_cnt)
                    qb = q1 - q0
                    raw_keys = torch.empty((qb, split_n * block_cap), dtype=torch.uint64, device=dev)
                    _pq_lut_adc_splitn_kernel[(qb, split_n)](
                        lut[q0:q1],
                        codes_c,
                        raw_keys,
                        qb,
                        n_db,
                        self.m,
                        self.ks,
                        split_n,
                        chunk_n,
                        M_CONST=m_const,
                        BLOCK_CAP=block_cap,
                        num_warps=2 if block_cap <= 32 else 4,
                    )
                    d_b, i_b = _unpack_topk_uint64_keys(raw_keys, k_top=k_eff, n_valid_db=n_db)
                    dists_out[q0:q1] = d_b
                    ids_out[q0:q1] = i_b
        else:
            # Exact deterministic fallback (handles `k_top > 128` or CPU)
            dists_out = torch.empty((q_cnt, k_eff), dtype=torch.float32, device=dev)
            ids_out = torch.empty((q_cnt, k_eff), dtype=torch.int64, device=dev)
            c_long = codes_c.long()
            id_row = torch.arange(n_db, device=dev, dtype=torch.int64)[None, :]
            for q0 in range(0, q_cnt, 64):
                q1 = min(q0 + 64, q_cnt)
                lut_b = lut[q0:q1]  # (qb, M, Ks)
                acc = torch.zeros((q1 - q0, n_db), dtype=torch.float32, device=dev)
                for m_i in range(self.m):
                    acc += lut_b[:, m_i, c_long[:, m_i]]
                acc = torch.where(acc > 0.0, acc, torch.zeros_like(acc))
                if n_db < (1 << 31):
                    packed = (acc.view(torch.int32).to(torch.int64) << 32) | (id_row & 0xFFFFFFFF)
                    top_p, _ = torch.topk(packed, k=k_eff, dim=1, largest=False, sorted=True)
                    dists_out[q0:q1] = ((top_p >> 32) & 0x7FFFFFFF).to(torch.int32).view(torch.float32)
                    ids_out[q0:q1] = top_p & 0xFFFFFFFF
                else:
                    d_s, i_s = torch.topk(acc, k=k_eff, dim=1, largest=False, sorted=True)
                    dists_out[q0:q1] = d_s
                    ids_out[q0:q1] = i_s

        if self._has_nonunit_global_scale and self._inv_global_scale_sq64 is not None:
            fmax = float(torch.finfo(torch.float32).max)
            dists_out = (dists_out.double() * self._inv_global_scale_sq64).clamp(0.0, fmax).float()

        return (dists_out.detach().cpu().numpy(), ids_out.detach().cpu().numpy()) if ret_numpy else (dists_out, ids_out)

    @torch.no_grad()
    def evaluate_reconstruction(
        self,
        x: torch.Tensor | np.ndarray,
        codes: torch.Tensor | np.ndarray | None = None,
    ) -> ReconstructionMetrics:
        """Compute exact chunked IEEE-754 FP64 Reconstruction MSE, SQNR (dB), EVR-SQNR (dB), and Cosine."""
        if self.codebooks is None or self.train_global_mean64 is None:
            raise RuntimeError("ApexPQ must be fitted before calling .evaluate_reconstruction")
        codes_use = self.encode(x) if codes is None else codes
        return evaluate_vq_reconstruction_fp64(
            x_orig=x,
            x_hat_fn=lambda s, e: self.decode(codes_use[s:e]),
            train_mean64=self.train_global_mean64,
            device=self.codebooks.device,
        )


# ==============================================================================
# 2. `ApexRVQ`: 2-Stage Residual Vector Quantizer + Exact Beam/LUT ADC Search
# ==============================================================================


@dataclass
class RVQInvertedIndex:
    """Compact CSR inverted-list index for fast 2-stage RVQ Beam LUT-ADC search."""

    k1: int
    k2: int
    n_db: int
    csr_offsets: torch.Tensor   # (K1 + 1,) int64
    csr_ids: torch.Tensor       # (N,) int32 database indices
    csr_c2: torch.Tensor        # (N,) int32 Stage-2 residual codes
    codes_raw: torch.Tensor     # (N, 2) int32 `(c1, c2)` cloned snapshot
    source_ptr: int = 0
    source_version: int = -1
    g_cross_vec: torch.Tensor | None = None  # lazily populated for exhaustive `adc_beam=None`


def pack_rvq_codes_uint8(codes: np.ndarray | torch.Tensor, k1: int, k2: int) -> np.ndarray:
    """Bit-pack `(N, 2)` RVQ `(c1, c2)` codes into exact `ceil((b1 + b2) / 8)` `uint8` bytes/vector!

    For example:
      - `K1 = 4096, K2 = 4096` (`12 + 12 = 24` bits) -> `(N, 3)` `uint8` (`3.0 B/vec`)!
      - `K1 = 1024, K2 = 1024` (`10 + 10 = 20` bits) -> `(N, 3)` `uint8` (`20` active bits)!
    """
    c_np = codes.detach().cpu().numpy() if isinstance(codes, torch.Tensor) else np.asarray(codes)
    if c_np.ndim != 2 or c_np.shape[1] != 2:
        raise ValueError(f"Expected (N, 2) codes, got {c_np.shape}")
    b1 = max(1, int(math.ceil(math.log2(max(2, k1)))))
    b2 = max(1, int(math.ceil(math.log2(max(2, k2)))))
    tot_bits = b1 + b2
    if tot_bits > 32:
        raise ValueError(f"pack_rvq_codes_uint8 supports up to 32 total bits, got {tot_bits}")
    n_bytes = (tot_bits + 7) // 8
    c1_u = c_np[:, 0].astype(np.uint64)
    c2_u = c_np[:, 1].astype(np.uint64)
    combined = c1_u | (c2_u << b1)
    out = np.empty((c_np.shape[0], n_bytes), dtype=np.uint8)
    for b_i in range(n_bytes):
        out[:, b_i] = ((combined >> (8 * b_i)) & 0xFF).astype(np.uint8)
    return out


def unpack_rvq_codes_uint8(packed: np.ndarray, k1: int, k2: int) -> np.ndarray:
    """Unpack `pack_rvq_codes_uint8` bytes back into exact `(N, 2)` `int32` `(c1, c2)` codes."""
    if packed.ndim != 2:
        raise ValueError(f"Expected 2D packed byte array, got {packed.shape}")
    b1 = max(1, int(math.ceil(math.log2(max(2, k1)))))
    b2 = max(1, int(math.ceil(math.log2(max(2, k2)))))
    combined = np.zeros((packed.shape[0],), dtype=np.uint64)
    for b_i in range(packed.shape[1]):
        combined |= packed[:, b_i].astype(np.uint64) << (8 * b_i)
    mask1 = (np.uint64(1) << np.uint64(b1)) - np.uint64(1)
    mask2 = (np.uint64(1) << np.uint64(b2)) - np.uint64(1)
    out = np.empty((packed.shape[0], 2), dtype=np.int32)
    out[:, 0] = (combined & mask1).astype(np.int32)
    out[:, 1] = ((combined >> np.uint64(b1)) & mask2).astype(np.int32)
    return out


class ApexRVQ:
    """2-Stage Residual Vector Quantizer (`ApexRVQ`) with Exact Origin-Centered Beam/LUT ADC."""

    def __init__(
        self,
        d: int,
        k1: int = 4096,
        k2: int = 4096,
        nepochs: float = 2.0,
        *,
        solver: str = "apex_emeans",
        encode_beam: int = 1,
        adc_beam: int | None = 128,
        seed: int = 1234,
        solver_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if d < 1:
            raise ValueError(f"Dimension d={d} must be >= 1")
        if k1 < 1 or k2 < 1:
            raise ValueError(f"k1={k1} and k2={k2} must both be >= 1")
        if not math.isfinite(nepochs) or nepochs <= 0.0:
            raise ValueError(f"nepochs={nepochs} must be finite and > 0")
        if encode_beam < 1:
            raise ValueError(f"encode_beam={encode_beam} must be >= 1")
        if adc_beam is not None and adc_beam < 1:
            raise ValueError(f"adc_beam={adc_beam} must be >= 1 or None")

        self.d = int(d)
        self.k1 = int(k1)
        self.k2 = int(k2)
        self.b1 = max(1, int(math.ceil(math.log2(max(2, self.k1)))))
        self.b2 = max(1, int(math.ceil(math.log2(max(2, self.k2)))))
        self.bits_per_vec = self.b1 + self.b2
        self.bytes_per_vec = float(self.bits_per_vec) / 8.0
        self.packed_bytes_per_vec = int(math.ceil(self.bits_per_vec / 8.0))
        self.compression_ratio_vs_fp32 = float(4 * self.d) / max(self.bytes_per_vec, 1e-9)
        self.nepochs = float(nepochs)

        self.solver_name = normalize_solver_name(solver)
        if self.solver_name not in RVQ_SUPPORTED_SOLVERS:
            raise ValueError(f"Unsupported RVQ solver {solver!r}; expected one of {RVQ_SUPPORTED_SOLVERS}")

        self.encode_beam = int(encode_beam)
        self.adc_beam = int(adc_beam) if adc_beam is not None else None
        self.seed = int(seed)
        self.solver_kwargs = dict(solver_kwargs or {})

        # Fitted state (strictly from `x_train`)
        self.c1: torch.Tensor | None = None                 # (K1, D) float32
        self.c2: torch.Tensor | None = None                 # (K2, D) float32
        self.train_origin64: torch.Tensor | None = None     # (1, D) float64
        self._coord_scale64: torch.Tensor | None = None     # () float64 conditioning scale `2^k`
        self._inv_scale_sq64: torch.Tensor | None = None    # () float64 `1 / scale^2`
        self._has_nonunit_scale: bool = False
        self._c1_cond64: torch.Tensor | None = None         # (K1, D) float64 `(C1 - origin) * scale`
        self._c2_cond64: torch.Tensor | None = None         # (K2, D) float64 `C2 * scale`
        self._c1_cond64_norm: torch.Tensor | None = None    # (K1,) float64
        self._c2_cond64_norm: torch.Tensor | None = None    # (K2,) float64
        self._c1_centered: torch.Tensor | None = None       # (K1, D) float32 conditioned
        self._c1_centered_pad: torch.Tensor | None = None   # (K1, D_PAD) float32 conditioned
        self._c1_centered_norm: torch.Tensor | None = None  # (K1,) float32 conditioned
        self._c2_cond_f32: torch.Tensor | None = None       # (K2, D) float32 conditioned
        self._c2_norm: torch.Tensor | None = None           # (K2,) float32 conditioned
        self._g_cross: torch.Tensor | None = None           # (K1, K2) float32 `2 * <C1_cond, C2_cond>`
        self.train_mean_power: float = 0.0
        self.train_var_power: float = 0.0
        self.device: torch.device = torch.device("cpu")

        self.stage1_init_time_sec: float = 0.0
        self.stage1_train_time_sec: float = 0.0
        self.stage1_time_sec: float = 0.0
        self.residual_time_sec: float = 0.0
        self.stage2_init_time_sec: float = 0.0
        self.stage2_train_time_sec: float = 0.0
        self.stage2_time_sec: float = 0.0
        self.init_time_sec: float = 0.0
        self.train_time_sec: float = 0.0
        self.total_time_sec: float = 0.0
        self.encode_time_sec: float = 0.0
        self.stage1_gear: str = "unfitted"
        self.stage2_gear: str = "unfitted"
        self.resolved_precision: str = "unfitted"
        self.n_splits: int = 0
        self._cached_csr: RVQInvertedIndex | None = None

    def _prepare_codebook_tables(self, c1: torch.Tensor, c2: torch.Tensor, origin64: torch.Tensor) -> None:
        """Precompute conditioned `C1_cond`, `C2_cond`, norms, and exact `G_cross = 2 <C1_cond, C2_cond>`."""
        dev = c1.device
        if self._coord_scale64 is None:
            self._coord_scale64 = torch.tensor(1.0, dtype=torch.float64, device=dev)
            self._inv_scale_sq64 = torch.tensor(1.0, dtype=torch.float64, device=dev)
            self._has_nonunit_scale = False
        elif self._coord_scale64.device != dev:
            self._coord_scale64 = self._coord_scale64.to(dev)
            assert self._inv_scale_sq64 is not None
            self._inv_scale_sq64 = self._inv_scale_sq64.to(dev)

        sc64 = self._coord_scale64
        d_pad = max(16, ((self.d + 15) // 16) * 16)
        c1_cond64 = ((c1.double() - origin64) * sc64).clamp(-1e18, 1e18).contiguous()
        c2_cond64 = (c2.double() * sc64).clamp(-1e18, 1e18).contiguous()
        c1_c = c1_cond64.float().contiguous()
        c2_f = c2_cond64.float().contiguous()
        c1_pad = torch.zeros((self.k1, d_pad), dtype=torch.float32, device=dev)
        c1_pad[:, : self.d] = c1_c

        self._c1_cond64 = c1_cond64
        self._c2_cond64 = c2_cond64
        self._c1_cond64_norm = c1_cond64.square().sum(dim=1).contiguous()
        self._c2_cond64_norm = c2_cond64.square().sum(dim=1).contiguous()
        self._c1_centered = c1_c
        self._c1_centered_pad = c1_pad.contiguous()
        self._c1_centered_norm = c1_c.square().sum(dim=1).contiguous()
        self._c2_cond_f32 = c2_f
        self._c2_norm = c2_f.square().sum(dim=1).contiguous()

        g_cross = torch.empty((self.k1, self.k2), dtype=torch.float32, device=dev)
        for k0 in range(0, self.k1, 2048):
            k1_e = min(k0 + 2048, self.k1)
            g_cross[k0:k1_e] = (2.0 * (c1_cond64[k0:k1_e] @ c2_cond64.t())).float()
        self._g_cross = g_cross.contiguous()
        self._cached_csr = None

    @torch.no_grad()
    def _assign_stage1_and_residuals(
        self,
        x_t: torch.Tensor,
        c1: torch.Tensor,
        origin64: torch.Tensor,
        compute_residuals: bool = True,
        residual_scale64: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Exact centered+conditioned Stage-1 nearest-centroid assignment + FP64-differenced residual `R`."""
        dev = x_t.device
        n = int(x_t.shape[0])
        assert self._c1_centered_pad is not None and self._coord_scale64 is not None
        sc64 = self._coord_scale64

        labels1 = torch.empty((n,), dtype=torch.int32, device=dev)
        res_out = torch.empty((n, self.d), dtype=torch.float32, device=dev) if compute_residuals else None

        if dev.type == "cuda" and HAS_TRITON:
            with torch.cuda.device(dev):
                d_pad = max(16, 1 << max(0, (self.d - 1).bit_length())) if self.d <= 128 else ((self.d + 63) // 64) * 64
                c1_bf16 = torch.zeros((self.k1, d_pad), dtype=torch.bfloat16, device=dev)
                c1_bf16[:, : self.d] = ((c1.double() - origin64) * sc64).clamp(-65000.0, 65000.0).to(torch.bfloat16)
                c1_norm_bf16 = c1_bf16.float().square().sum(dim=1).contiguous()

                xc_bf16 = torch.zeros((n, d_pad), dtype=torch.bfloat16, device=dev)
                for s in range(0, n, 262144):
                    e = min(s + 262144, n)
                    xc_bf16[s:e, : self.d] = ((x_t[s:e].double() - origin64) * sc64).clamp(-65000.0, 65000.0).to(torch.bfloat16)

                # When `self._has_nonunit_scale` (e.g. `1e-25` or `1e+25`), pass conditioned `x_cond_f32` & `c1_cond_f32`
                # into `_rvq_fused_assign_residual_kernel` so `(x - c1)^2` never overflows/underflows even in extreme FP64!
                if self._has_nonunit_scale:
                    x_ref_in = torch.empty((n, self.d), dtype=torch.float32, device=dev)
                    for s in range(0, n, 262144):
                        e = min(s + 262144, n)
                        x_ref_in[s:e] = ((x_t[s:e].double() - origin64) * sc64).clamp(-1e18, 1e18).float()
                    c1_ref_in = ((c1.double() - origin64) * sc64).clamp(-1e18, 1e18).float().contiguous()
                else:
                    x_ref_in = x_t
                    c1_ref_in = c1

                full_d = d_pad <= 128
                block_d = d_pad if full_d else 64
                block_k = min(64, max(16, 1 << max(0, (self.k1 - 1).bit_length())))
                _rvq_fused_assign_residual_kernel[(triton.cdiv(n, 128),)](
                    x_ref_in,
                    xc_bf16,
                    c1_ref_in,
                    c1_bf16,
                    c1_norm_bf16,
                    labels1,
                    res_out if res_out is not None else c1_ref_in,
                    n,
                    self.k1,
                    self.d,
                    d_pad,
                    x_ref_in.stride(0),
                    x_ref_in.stride(1),
                    WRITE_RES=compute_residuals,
                    FULL_D=full_d,
                    BLOCK_N=128,
                    BLOCK_K=block_k,
                    BLOCK_D=block_d,
                    num_warps=4,
                )
                # Note: if `self._has_nonunit_scale` and `residual_scale64 is None`, unscale `res_out` back to raw units;
                # if `residual_scale64 is not None`, `res_out` is ALREADY in conditioned `sc64` units!
                if compute_residuals and res_out is not None and self._has_nonunit_scale and residual_scale64 is None:
                    res_out = (res_out.double() / sc64).float().contiguous()
            return labels1, res_out

        # CPU / fallback chunked FP64-centered assignment + residual
        c1_c64 = (c1.double() - origin64) * sc64
        c1_n64 = c1_c64.square().sum(dim=1)
        for s in range(0, n, 16384):
            e = min(s + 16384, n)
            xb64 = x_t[s:e].double()
            xc64 = (xb64 - origin64) * sc64
            best_s = torch.full((e - s,), float("inf"), dtype=torch.float64, device=dev)
            best_i = torch.zeros((e - s,), dtype=torch.int64, device=dev)
            for k0 in range(0, self.k1, 4096):
                k1_e = min(k0 + 4096, self.k1)
                sc = c1_n64[k0:k1_e][None, :] - 2.0 * (xc64 @ c1_c64[k0:k1_e].t())
                mn, mi = sc.min(dim=1)
                bet = mn < best_s
                best_s = torch.where(bet, mn, best_s)
                best_i = torch.where(bet, mi + k0, best_i)
            labels1[s:e] = best_i.to(torch.int32)
            if res_out is not None:
                if residual_scale64 is not None:
                    res_out[s:e] = (xc64 - c1_c64[best_i]).float()
                else:
                    res_out[s:e] = (xb64 - c1.double()[best_i]).float()
        return labels1, res_out

    @torch.no_grad()
    def fit(
        self,
        x_train: torch.Tensor | np.ndarray,
        *,
        capture_codes: bool = False,
    ) -> tuple[tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray], torch.Tensor | np.ndarray | None]:
        """Fit 2-stage RVQ codebooks `(C_1, C_2)` strictly on `x_train`."""
        input_is_numpy = isinstance(x_train, np.ndarray)
        if input_is_numpy:
            x_np = np.asarray(x_train)
            if x_np.ndim != 2:
                raise ValueError(f"Expected 2D x_train, got shape {x_np.shape}")
            target_dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            x_t = torch.from_numpy(
                np.ascontiguousarray(x_np, dtype=np.float64 if x_np.dtype == np.float64 else np.float32)
            ).to(device=target_dev)
        else:
            if x_train.ndim != 2:
                raise ValueError(f"Expected 2D x_train tensor, got shape {tuple(x_train.shape)}")
            x_t = x_train.detach()

        n, d = int(x_t.shape[0]), int(x_t.shape[1])
        if d != self.d:
            raise ValueError(f"Expected dimension d={self.d}, got {d}")
        if n < max(self.k1, self.k2):
            raise ValueError(f"Number of training vectors n={n} must be >= max(k1={self.k1}, k2={self.k2})")
        if not bool(torch.isfinite(x_t).all()):
            raise ValueError("Input x_train must contain only finite values (no NaN or Inf)")

        dev = x_t.device
        self.device = dev
        ctx = torch.cuda.device(dev) if dev.type == "cuda" else contextlib.nullcontext()

        with ctx:
            # Compute training-only FP64 origin, all-row conditioning scale, & signal/variance power
            sum_x = torch.zeros((self.d,), dtype=torch.float64, device=dev)
            sum_sq = torch.zeros((), dtype=torch.float64, device=dev)
            for s in range(0, n, 131072):
                xb = x_t[s : min(s + 131072, n)].double()
                sum_x += xb.sum(dim=0)
                sum_sq += xb.square().sum()
            origin64 = (sum_x / float(n)).reshape(1, self.d).contiguous()

            max_abs = torch.zeros((), dtype=torch.float64, device=dev)
            for s in range(0, n, 131072):
                xb = x_t[s : min(s + 131072, n)].double()
                max_abs = torch.maximum(max_abs, (xb - origin64).abs().amax())

            sc64 = _power_of_two_coord_scales_64(max_abs.view(1))[0]
            sc_val = float(sc64.item())
            self.train_origin64 = origin64
            self._coord_scale64 = sc64
            self._inv_scale_sq64 = 1.0 / (sc64 * sc64)
            self._has_nonunit_scale = sc_val != 1.0
            self.train_mean_power = float((sum_sq / float(n)).item())
            self.train_var_power = max(self.train_mean_power - float(origin64.square().sum().item()), 0.0)

            if self._has_nonunit_scale:
                x_s1_in = torch.empty((n, self.d), dtype=torch.float32, device=dev)
                for s in range(0, n, 131072):
                    e = min(s + 131072, n)
                    x_s1_in[s:e] = ((x_t[s:e].double() - origin64) * sc64).clamp(-1e18, 1e18).float()
            else:
                x_s1_in = x_t if x_t.is_contiguous() else x_t.contiguous()

            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            t_all0 = time.perf_counter()

            # Stage 1: Coarse codebook `C_1` (`K_1` centroids) on `X_train`
            sol1 = instantiate_subproblem_solver(
                self.solver_name,
                d=self.d,
                k=self.k1,
                nepochs=self.nepochs,
                seed=int(self.seed),
                solver_kwargs=self.solver_kwargs,
            )
            c1_raw, _ = sol1.fit(x_s1_in, capture_assignments=False)
            c1_fit = (
                torch.from_numpy(c1_raw).to(device=dev, dtype=torch.float32).contiguous()
                if isinstance(c1_raw, np.ndarray)
                else c1_raw.to(device=dev, dtype=torch.float32).contiguous()
            )
            if self._has_nonunit_scale:
                c1 = (c1_fit.double() / sc64 + origin64).float().contiguous()
            else:
                c1 = c1_fit
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            t_s1_done = time.perf_counter()

            # Exact Stage-1 Assignment + Residual Computation `R = X_train - C_1[labels_1]` on GPU
            d_pad = max(16, ((self.d + 15) // 16) * 16)
            c1_c = ((c1.double() - origin64) * sc64).clamp(-1e18, 1e18).float().contiguous()
            c1_pad = torch.zeros((self.k1, d_pad), dtype=torch.float32, device=dev)
            c1_pad[:, : self.d] = c1_c
            self._c1_centered = c1_c
            self._c1_centered_pad = c1_pad
            self._c1_centered_norm = c1_c.square().sum(dim=1).contiguous()

            labels1_train, residuals_cond = self._assign_stage1_and_residuals(
                x_t,
                c1,
                origin64,
                compute_residuals=True,
                residual_scale64=sc64 if self._has_nonunit_scale else None,
            )
            assert residuals_cond is not None
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            t_res_done = time.perf_counter()

            # Stage 2: Fine residual codebook `C_2` (`K_2` centroids) on conditioned `R`
            stage2_seed = int(self.seed) + 1_000_003
            sol2 = instantiate_subproblem_solver(
                self.solver_name,
                d=self.d,
                k=self.k2,
                nepochs=self.nepochs,
                seed=stage2_seed,
                solver_kwargs=self.solver_kwargs,
            )
            c2_raw, _ = sol2.fit(residuals_cond, capture_assignments=False)
            c2_fit = (
                torch.from_numpy(c2_raw).to(device=dev, dtype=torch.float32).contiguous()
                if isinstance(c2_raw, np.ndarray)
                else c2_raw.to(device=dev, dtype=torch.float32).contiguous()
            )
            if self._has_nonunit_scale:
                c2 = (c2_fit.double() / sc64).float().contiguous()
            else:
                c2 = c2_fit
            self._prepare_codebook_tables(c1, c2, origin64)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            t_all_end = time.perf_counter()

        self.c1 = c1
        self.c2 = c2
        self.stage1_init_time_sec = float(getattr(sol1, "init_time_sec", 0.0))
        self.stage1_train_time_sec = float(getattr(sol1, "train_time_sec", 0.0))
        self.stage1_time_sec = float(t_s1_done - t_all0)
        self.residual_time_sec = float(t_res_done - t_s1_done)
        self.stage2_init_time_sec = float(getattr(sol2, "init_time_sec", 0.0))
        self.stage2_train_time_sec = float(getattr(sol2, "train_time_sec", 0.0))
        self.stage2_time_sec = float(t_all_end - t_res_done)
        self.init_time_sec = self.stage1_init_time_sec + self.stage2_init_time_sec
        self.total_time_sec = float(t_all_end - t_all0)
        self.train_time_sec = max(0.0, self.total_time_sec - self.init_time_sec)
        self.stage1_gear = str(getattr(sol1, "selected_gear", self.solver_name))
        self.stage2_gear = str(getattr(sol2, "selected_gear", self.solver_name))
        self.resolved_precision = str(getattr(sol1, "resolved_precision", "bf16" if dev.type == "cuda" else "fp32"))
        self.n_splits = int(getattr(sol1, "n_splits", 0)) + int(getattr(sol2, "n_splits", 0))
        _ = labels1_train

        codes_out = self.encode(x_train) if capture_codes else None
        if input_is_numpy:
            return (c1.detach().cpu().numpy(), c2.detach().cpu().numpy()), codes_out
        return (c1, c2), codes_out

    @torch.no_grad()
    def encode(
        self,
        x: torch.Tensor | np.ndarray,
        beam: int | None = None,
    ) -> torch.Tensor | np.ndarray:
        """Encode `(N, D)` vectors into `(N, 2)` `int32` `(c_1, c_2)` codes (greedy `beam=1` or multi-beam)."""
        if (
            self.c1 is None
            or self.c2 is None
            or self.train_origin64 is None
            or self._coord_scale64 is None
            or self._c1_centered is None
            or self._c2_cond_f32 is None
            or self._c2_norm is None
            or self._g_cross is None
        ):
            raise RuntimeError("ApexRVQ must be fitted via .fit(x_train) before calling .encode(x)")

        eff_beam = int(self.encode_beam if beam is None else beam)
        if eff_beam < 1:
            raise ValueError(f"beam={eff_beam} must be >= 1")

        input_is_numpy = isinstance(x, np.ndarray)
        x_t = (
            torch.from_numpy(np.ascontiguousarray(x, dtype=np.float64 if x.dtype == np.float64 else np.float32))
            if input_is_numpy
            else x.detach()
        )
        if x_t.ndim != 2 or int(x_t.shape[1]) != self.d:
            raise ValueError(f"Expected (N, {self.d}) input to encode, got {tuple(x_t.shape)}")
        if not bool(torch.isfinite(x_t).all()):
            raise ValueError("Input x to encode must contain only finite values")

        n = int(x_t.shape[0])
        dev = self.c1.device
        if n == 0:
            empty = torch.empty((0, 2), dtype=torch.int32, device=dev)
            return empty.cpu().numpy() if input_is_numpy else empty

        ctx = torch.cuda.device(dev) if dev.type == "cuda" else contextlib.nullcontext()
        with ctx:
            x_dev = x_t.to(device=dev)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            t0 = time.perf_counter()

            codes = torch.empty((n, 2), dtype=torch.int32, device=dev)
            eff_b = min(eff_beam, self.k1)
            sc64 = self._coord_scale64

            if eff_b == 1:
                # Fast Greedy 2-Stage Encoding in conditioned coordinates (`1e-25..1e+25` safe!)
                labels1, residuals_cond = self._assign_stage1_and_residuals(
                    x_dev,
                    self.c1,
                    self.train_origin64,
                    compute_residuals=True,
                    residual_scale64=sc64 if self._has_nonunit_scale else None,
                )
                assert residuals_cond is not None
                _, labels2 = assign_points(residuals_cond, self._c2_cond_f32, compute_dtype=torch.float32)
                codes[:, 0] = labels1
                codes[:, 1] = labels2.to(torch.int32)
            else:
                # Multi-Path M-Beam Encoding in conditioned coordinates
                assert self._c1_centered_norm is not None
                c1_c = self._c1_centered
                c1_n = self._c1_centered_norm
                c2_f = self._c2_cond_f32
                c2_n = self._c2_norm
                g_cr = self._g_cross

                chunk = max(256, min(8192, (64 * 1024 * 1024) // max(1, eff_b * self.k2)))
                for s in range(0, n, chunk):
                    e = min(s + chunk, n)
                    xc = ((x_dev[s:e].double() - self.train_origin64) * sc64).clamp(-1e18, 1e18).float()
                    s1_scores = c1_n[None, :] - 2.0 * (xc @ c1_c.t())
                    top_s1, top_j1 = torch.topk(s1_scores, k=eff_b, dim=1, largest=False, sorted=True)
                    lut2_x = c2_n[None, :] - 2.0 * (xc @ c2_f.t())
                    tot_sc = top_s1[:, :, None] + lut2_x[:, None, :] + g_cr[top_j1]
                    flat_min = tot_sc.reshape(e - s, eff_b * self.k2).argmin(dim=1)
                    best_b = torch.div(flat_min, self.k2, rounding_mode="floor")
                    best_j2 = flat_min % self.k2
                    best_j1 = top_j1.gather(1, best_b[:, None]).squeeze(1)
                    codes[s:e, 0] = best_j1.to(torch.int32)
                    codes[s:e, 1] = best_j2.to(torch.int32)

            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            self.encode_time_sec = time.perf_counter() - t0
        return codes.detach().cpu().numpy() if input_is_numpy else codes

    @torch.no_grad()
    def decode(self, codes: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """Reconstruct `(N, D)` float32 vectors `x_hat = C_1[c_1] + C_2[c_2]`."""
        if self.c1 is None or self.c2 is None:
            raise RuntimeError("ApexRVQ must be fitted before calling .decode(codes)")
        input_is_numpy = isinstance(codes, np.ndarray)
        c_idx = (
            torch.from_numpy(np.ascontiguousarray(codes)).to(device=self.c1.device, dtype=torch.int64)
            if input_is_numpy
            else codes.detach().to(device=self.c1.device, dtype=torch.int64)
        )
        if c_idx.ndim != 2 or int(c_idx.shape[1]) != 2:
            raise ValueError(f"Expected (N, 2) RVQ codes, got {tuple(c_idx.shape)}")
        if c_idx.shape[0] == 0:
            empty = torch.empty((0, self.d), dtype=torch.float32, device=self.c1.device)
            return empty.cpu().numpy() if input_is_numpy else empty

        rec = (self.c1.double()[c_idx[:, 0]] + self.c2.double()[c_idx[:, 1]]).float().contiguous()
        return rec.detach().cpu().numpy() if input_is_numpy else rec

    @torch.no_grad()
    def build_csr_index(self, codes: torch.Tensor | np.ndarray) -> RVQInvertedIndex:
        """Group `(N, 2)` RVQ codes by coarse `c_1` into a GPU CSR index for sub-millisecond Beam LUT-ADC."""
        if self.c1 is None or self._g_cross is None:
            raise RuntimeError("ApexRVQ must be fitted before calling .build_csr_index(codes)")
        dev = self.c1.device
        ctx = torch.cuda.device(dev) if dev.type == "cuda" else contextlib.nullcontext()
        with ctx:
            if isinstance(codes, np.ndarray):
                c_t = torch.from_numpy(np.ascontiguousarray(codes, dtype=np.int32)).to(device=dev)
                src_ptr = 0
                src_ver = -1
            else:
                src_ptr = int(codes.data_ptr()) if codes.device == dev and codes.dtype == torch.int32 else 0
                src_ver = int(codes._version) if src_ptr != 0 else -1
                c_t = codes.detach().to(device=dev, dtype=torch.int32).clone().contiguous()
            if c_t.ndim != 2 or int(c_t.shape[1]) != 2:
                raise ValueError(f"Expected (N, 2) RVQ codes, got {tuple(c_t.shape)}")
            n_db = int(c_t.shape[0])
            c1_long = c_t[:, 0].long()
            sorted_pos = torch.argsort(c1_long, stable=True)
            counts = torch.bincount(c1_long, minlength=self.k1)
            offsets = torch.zeros((self.k1 + 1,), dtype=torch.int64, device=dev)
            torch.cumsum(counts, dim=0, out=offsets[1:])
            csr = RVQInvertedIndex(
                k1=self.k1,
                k2=self.k2,
                n_db=n_db,
                csr_offsets=offsets,
                csr_ids=sorted_pos.to(torch.int32).contiguous(),
                csr_c2=c_t[sorted_pos, 1].contiguous(),
                codes_raw=c_t,
                source_ptr=src_ptr,
                source_version=src_ver,
            )
        self._cached_csr = csr
        return csr

    @torch.no_grad()
    def compute_luts(self, queries: torch.Tensor | np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute exact origin-centered & conditioned `(LUT_1, LUT_2)` of shapes `(Q, K1)` and `(Q, K2)`."""
        if (
            self.c1 is None
            or self.c2 is None
            or self.train_origin64 is None
            or self._coord_scale64 is None
            or self._c1_cond64 is None
            or self._c2_cond64 is None
            or self._c1_cond64_norm is None
            or self._c2_cond64_norm is None
        ):
            raise RuntimeError("ApexRVQ must be fitted before calling .compute_luts(queries)")

        q_t = (
            torch.from_numpy(np.ascontiguousarray(queries, dtype=np.float64 if queries.dtype == np.float64 else np.float32))
            if isinstance(queries, np.ndarray)
            else queries.detach()
        )
        if q_t.ndim != 2 or int(q_t.shape[1]) != self.d:
            raise ValueError(f"Expected (Q, {self.d}) queries, got {tuple(q_t.shape)}")
        if not q_t.is_cuda and not bool(torch.isfinite(q_t).all()):
            raise ValueError("queries must contain only finite values")

        dev = self.c1.device
        q_cnt = int(q_t.shape[0])
        if q_cnt == 0:
            return (
                torch.empty((0, self.k1), dtype=torch.float32, device=dev),
                torch.empty((0, self.k2), dtype=torch.float32, device=dev),
            )

        ctx = torch.cuda.device(dev) if dev.type == "cuda" else contextlib.nullcontext()
        with ctx:
            # Exact FP64 centered+conditioned inner products eliminate TF32/FP32 cancellation drift!
            q_c64 = ((q_t.to(device=dev, dtype=torch.float64) - self.train_origin64) * self._coord_scale64).clamp(-1e18, 1e18)
            q_norm64 = q_c64.square().sum(dim=1, keepdim=True)
            lut1 = (q_norm64 + self._c1_cond64_norm[None, :] - 2.0 * (q_c64 @ self._c1_cond64.t())).clamp_min_(0.0).float().contiguous()
            lut2 = (self._c2_cond64_norm[None, :] - 2.0 * (q_c64 @ self._c2_cond64.t())).float().contiguous()
            return lut1, lut2

    @torch.no_grad()
    def adc_search(
        self,
        queries: torch.Tensor | np.ndarray,
        codes: torch.Tensor | np.ndarray | RVQInvertedIndex,
        k_top: int = 10,
        *,
        adc_beam: int | None = -1,
    ) -> tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray]:
        """2-Stage RVQ Asymmetric Distance Computation (`(dists, ids)` of shape `(Q, k_eff)`)."""
        if self.c1 is None or self.c2 is None or self._g_cross is None:
            raise RuntimeError("ApexRVQ must be fitted before calling .adc_search")
        if k_top < 1:
            raise ValueError(f"k_top={k_top} must be >= 1")

        eff_beam = self.adc_beam if adc_beam == -1 else adc_beam
        ret_numpy = isinstance(queries, np.ndarray) and isinstance(codes, np.ndarray)
        dev = self.c1.device

        lut1, lut2 = self.compute_luts(queries)
        q_cnt = int(lut1.shape[0])

        ctx = torch.cuda.device(dev) if dev.type == "cuda" else contextlib.nullcontext()
        with ctx:
            if isinstance(codes, RVQInvertedIndex):
                csr = codes
            else:
                if isinstance(codes, torch.Tensor):
                    if codes.ndim != 2 or int(codes.shape[1]) != 2:
                        raise ValueError(f"Expected (N, 2) RVQ codes, got {tuple(codes.shape)}")
                    can_reuse = (
                        self._cached_csr is not None
                        and codes.device == dev
                        and codes.dtype == torch.int32
                        and self._cached_csr.n_db == int(codes.shape[0])
                        and self._cached_csr.source_ptr != 0
                        and self._cached_csr.source_ptr == int(codes.data_ptr())
                        and self._cached_csr.source_version == int(codes._version)
                    )
                else:
                    can_reuse = False

                if can_reuse:
                    assert self._cached_csr is not None
                    csr = self._cached_csr
                elif eff_beam is not None and eff_beam < self.k1:
                    csr = self.build_csr_index(codes)
                else:
                    codes_t = (
                        torch.from_numpy(np.ascontiguousarray(codes, dtype=np.int32)).to(device=dev)
                        if isinstance(codes, np.ndarray)
                        else codes.detach().to(device=dev, dtype=torch.int32).contiguous()
                    )
                    if codes_t.ndim != 2 or int(codes_t.shape[1]) != 2:
                        raise ValueError(f"Expected (N, 2) RVQ codes, got {tuple(codes_t.shape)}")
                    csr = RVQInvertedIndex(
                        k1=self.k1,
                        k2=self.k2,
                        n_db=int(codes_t.shape[0]),
                        csr_offsets=torch.empty((0,), dtype=torch.int64, device=dev),
                        csr_ids=torch.empty((0,), dtype=torch.int32, device=dev),
                        csr_c2=torch.empty((0,), dtype=torch.int32, device=dev),
                        codes_raw=codes_t,
                    )

            n_db = csr.n_db
            k_eff = min(int(k_top), n_db)
            if q_cnt == 0 or k_eff == 0:
                d_e = torch.empty((q_cnt, 0), dtype=torch.float32, device=dev)
                i_e = torch.empty((q_cnt, 0), dtype=torch.int64, device=dev)
                return (d_e.cpu().numpy(), i_e.cpu().numpy()) if ret_numpy else (d_e, i_e)

            use_beam = eff_beam is not None and int(eff_beam) < self.k1
            if dev.type == "cuda" and HAS_TRITON and k_eff <= 128 and n_db < (1 << 31):
                block_cap = 32 if k_eff <= 32 else 128
                n_warps_adc = 2 if block_cap <= 32 else 4
                dists_out = torch.empty((q_cnt, k_eff), dtype=torch.float32, device=dev)
                ids_out = torch.empty((q_cnt, k_eff), dtype=torch.int64, device=dev)

                if use_beam:
                    if csr.csr_offsets.numel() == 0:
                        csr = self.build_csr_index(csr.codes_raw)
                    b_width = min(int(eff_beam), self.k1)  # type: ignore[arg-type]
                    probed_c1 = torch.topk(lut1, k=b_width, dim=1, largest=False, sorted=False).indices.contiguous()
                    q_batch = max(1, min(q_cnt, 33_554_432 // max(1, b_width * block_cap)))
                    for q0 in range(0, q_cnt, q_batch):
                        q1 = min(q0 + q_batch, q_cnt)
                        qb = q1 - q0
                        raw_keys = torch.empty((qb, b_width * block_cap), dtype=torch.uint64, device=dev)
                        _rvq_csr_beam_adc_kernel[(qb, b_width)](
                            lut1[q0:q1],
                            lut2[q0:q1],
                            self._g_cross,
                            probed_c1[q0:q1],
                            csr.csr_offsets,
                            csr.csr_ids,
                            csr.csr_c2,
                            raw_keys,
                            qb,
                            b_width,
                            self.k1,
                            self.k2,
                            BLOCK_CAP=block_cap,
                            num_warps=n_warps_adc,
                        )
                        d_b, i_b = _unpack_topk_uint64_keys(raw_keys, k_top=k_eff, n_valid_db=n_db)
                        dists_out[q0:q1] = d_b
                        ids_out[q0:q1] = i_b
                else:
                    if csr.g_cross_vec is None:
                        c1_idx = csr.codes_raw[:, 0].long()
                        c2_idx = csr.codes_raw[:, 1].long()
                        csr.g_cross_vec = self._g_cross[c1_idx, c2_idx].contiguous()
                    split_n = min(16, max(1, triton.cdiv(n_db, 16384)))
                    chunk_n = triton.cdiv(n_db, split_n)
                    q_batch = max(1, min(q_cnt, 16384))
                    for q0 in range(0, q_cnt, q_batch):
                        q1 = min(q0 + q_batch, q_cnt)
                        qb = q1 - q0
                        raw_keys = torch.empty((qb, split_n * block_cap), dtype=torch.uint64, device=dev)
                        _rvq_exhaustive_adc_splitn_kernel[(qb, split_n)](
                            lut1[q0:q1],
                            lut2[q0:q1],
                            csr.g_cross_vec,
                            csr.codes_raw,
                            raw_keys,
                            qb,
                            n_db,
                            self.k1,
                            self.k2,
                            split_n,
                            chunk_n,
                            BLOCK_CAP=block_cap,
                            num_warps=n_warps_adc,
                        )
                        d_b, i_b = _unpack_topk_uint64_keys(raw_keys, k_top=k_eff, n_valid_db=n_db)
                        dists_out[q0:q1] = d_b
                        ids_out[q0:q1] = i_b
            else:
                # Exact deterministic PyTorch fallback (CPU or `k_top > 128`)
                dists_out = torch.empty((q_cnt, k_eff), dtype=torch.float32, device=dev)
                ids_out = torch.empty((q_cnt, k_eff), dtype=torch.int64, device=dev)
                c1_idx = csr.codes_raw[:, 0].long()
                c2_idx = csr.codes_raw[:, 1].long()
                gc_vec = self._g_cross[c1_idx, c2_idx]
                id_row = torch.arange(n_db, device=dev, dtype=torch.int64)[None, :]
                for q0 in range(0, q_cnt, 64):
                    q1 = min(q0 + 64, q_cnt)
                    acc = lut1[q0:q1, c1_idx] + lut2[q0:q1, c2_idx] + gc_vec[None, :]
                    acc = torch.where(acc > 0.0, acc, torch.zeros_like(acc))
                    if use_beam:
                        b_width = min(int(eff_beam), self.k1)  # type: ignore[arg-type]
                        probed = torch.topk(lut1[q0:q1], k=b_width, dim=1, largest=False, sorted=False).indices
                        in_beam = (c1_idx[None, :, None] == probed[:, None, :]).any(dim=2)
                        acc = torch.where(in_beam, acc, torch.full_like(acc, float("inf")))
                    if n_db < (1 << 31):
                        packed = (acc.view(torch.int32).to(torch.int64) << 32) | (id_row & 0xFFFFFFFF)
                        top_p, _ = torch.topk(packed, k=k_eff, dim=1, largest=False, sorted=True)
                        pv_valid = (top_p & 0xFFFFFFFF) != 0xFFFFFFFF
                        d_v = ((top_p >> 32) & 0x7FFFFFFF).to(torch.int32).view(torch.float32)
                        dists_out[q0:q1] = torch.where(pv_valid, d_v, torch.full_like(d_v, float("inf")))
                        ids_out[q0:q1] = torch.where(pv_valid & torch.isfinite(d_v), top_p & 0xFFFFFFFF, torch.full_like(top_p, -1))
                    else:
                        d_s, i_s = torch.topk(acc, k=k_eff, dim=1, largest=False, sorted=True)
                        dists_out[q0:q1] = d_s
                        ids_out[q0:q1] = torch.where(torch.isfinite(d_s), i_s, torch.full_like(i_s, -1))

            if self._has_nonunit_scale and self._inv_scale_sq64 is not None:
                fmax = float(torch.finfo(torch.float32).max)
                dists_out = (dists_out.double() * self._inv_scale_sq64).clamp(0.0, fmax).float()

        return (dists_out.detach().cpu().numpy(), ids_out.detach().cpu().numpy()) if ret_numpy else (dists_out, ids_out)

    @torch.no_grad()
    def evaluate_reconstruction(
        self,
        x: torch.Tensor | np.ndarray,
        codes: torch.Tensor | np.ndarray | None = None,
    ) -> ReconstructionMetrics:
        """Compute exact chunked IEEE-754 FP64 Reconstruction MSE, SQNR (dB), EVR-SQNR (dB), and Cosine."""
        if self.c1 is None or self.c2 is None or self.train_origin64 is None:
            raise RuntimeError("ApexRVQ must be fitted before calling .evaluate_reconstruction")
        codes_use = self.encode(x) if codes is None else codes
        return evaluate_vq_reconstruction_fp64(
            x_orig=x,
            x_hat_fn=lambda s, e: self.decode(codes_use[s:e]),
            train_mean64=self.train_origin64,
            device=self.c1.device,
        )


# ==============================================================================
# Exact FP64 Reconstruction Fidelity & Tie-Aware ADC Recall Evaluators + Warmup
# ==============================================================================


@torch.no_grad()
def evaluate_vq_reconstruction_fp64(
    x_orig: torch.Tensor | np.ndarray,
    x_hat_fn: Any,
    train_mean64: torch.Tensor,
    device: torch.device,
    chunk_size: int = 65536,
) -> ReconstructionMetrics:
    """Chunked exact IEEE-754 FP64 MSE, SQNR (dB), shift-invariant EVR-SQNR (dB), and Cosine similarity."""
    x_t = torch.from_numpy(x_orig) if isinstance(x_orig, np.ndarray) else x_orig.detach()
    n, d = int(x_t.shape[0]), int(x_t.shape[1])
    if n == 0:
        return ReconstructionMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

    mu64 = train_mean64.to(device=device, dtype=torch.float64).reshape(1, d)
    err_sq = torch.zeros((), dtype=torch.float64, device=device)
    raw_sq = torch.zeros((), dtype=torch.float64, device=device)
    var_sq = torch.zeros((), dtype=torch.float64, device=device)
    cos_sum = torch.zeros((), dtype=torch.float64, device=device)

    for s in range(0, n, chunk_size):
        e = min(s + chunk_size, n)
        xb64 = x_t[s:e].to(device=device, dtype=torch.float64)
        xh_raw = x_hat_fn(s, e)
        xh64 = (
            torch.from_numpy(xh_raw).to(device=device, dtype=torch.float64)
            if isinstance(xh_raw, np.ndarray)
            else xh_raw.to(device=device, dtype=torch.float64)
        )
        diff = xb64 - xh64
        err_sq += diff.square().sum()
        raw_sq += xb64.square().sum()
        var_sq += (xb64 - mu64).square().sum()

        nx = xb64.norm(dim=1).clamp_min(1e-30)
        nh = xh64.norm(dim=1).clamp_min(1e-30)
        cos_sum += ((xb64 * xh64).sum(dim=1) / (nx * nh)).sum()

    mse = float((err_sq / float(n)).item())
    p_raw = float((raw_sq / float(n)).item())
    p_var = float((var_sq / float(n)).item())
    mean_cos = float((cos_sum / float(n)).item())

    if mse <= 0.0:
        sqnr_db = 120.0
        evr_sqnr_db = 120.0
    else:
        denom_mse = max(mse, 1e-300)
        sqnr_db = 10.0 * math.log10(p_raw / denom_mse) if p_raw > 0.0 else 0.0
        evr_sqnr_db = 10.0 * math.log10(p_var / denom_mse) if p_var > 0.0 else 0.0

    return ReconstructionMetrics(
        mse=mse,
        per_dim_mse=mse / float(max(1, d)),
        rmse=math.sqrt(max(mse, 0.0)),
        sqnr_db=sqnr_db,
        evr_sqnr_db=evr_sqnr_db,
        mean_cosine_sim=mean_cos,
    )


@torch.no_grad()
def evaluate_adc_recall(
    quantizer: ApexPQ | ApexRVQ,
    x_base: torch.Tensor | np.ndarray,
    codes: torch.Tensor | np.ndarray | RVQInvertedIndex,
    x_query: torch.Tensor | np.ndarray,
    groundtruth_ids: torch.Tensor | np.ndarray,
    groundtruth_sq_dists: torch.Tensor | np.ndarray | None = None,
    *,
    k_top: int = 10,
    adc_beam: int | None = -1,
) -> ADCRecallMetrics:
    """Timed GPU LUT-ADC search + exact ID and tie-aware FP64 Recall@1 / 1-Recall@10 / 10-Recall@10."""
    dev = quantizer.device
    xq_t = (
        torch.from_numpy(np.ascontiguousarray(x_query, dtype=np.float32)).to(device=dev)
        if isinstance(x_query, np.ndarray)
        else x_query.detach().to(device=dev)
    )
    n_q = int(xq_t.shape[0])
    k_req = max(10, int(k_top))

    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    if isinstance(quantizer, ApexRVQ):
        _, pred_ids = quantizer.adc_search(xq_t, codes, k_top=k_req, adc_beam=adc_beam)
    else:
        assert not isinstance(codes, RVQInvertedIndex)
        _, pred_ids = quantizer.adc_search(xq_t, codes, k_top=k_req)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    search_sec = time.perf_counter() - t0

    pred_t = (
        torch.from_numpy(pred_ids).to(device=dev, dtype=torch.int64)
        if isinstance(pred_ids, np.ndarray)
        else pred_ids.to(device=dev, dtype=torch.int64)
    )
    gt_id_t = (
        torch.from_numpy(np.asarray(groundtruth_ids, dtype=np.int64)).to(device=dev)
        if not isinstance(groundtruth_ids, torch.Tensor)
        else groundtruth_ids.detach().to(device=dev, dtype=torch.int64)
    )

    r1_eff = min(1, pred_t.shape[1], gt_id_t.shape[1])
    r10_eff = min(10, pred_t.shape[1], gt_id_t.shape[1])
    if r1_eff == 0 or n_q == 0:
        return ADCRecallMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, search_sec, 0.0)

    valid_p1 = pred_t[:, 0] >= 0
    id_r1 = float(((pred_t[:, 0] == gt_id_t[:, 0]) & valid_p1).double().mean().item())

    valid_p10 = pred_t[:, :r10_eff] >= 0
    one_r10 = float((((pred_t[:, :r10_eff] == gt_id_t[:, 0:1]) & valid_p10).any(dim=1)).double().mean().item())

    inter_10 = (pred_t[:, :r10_eff, None] == gt_id_t[:, None, :r10_eff]).any(dim=2) & valid_p10
    id_r10 = float((inter_10.double().sum(dim=1) / float(r10_eff)).mean().item())

    if groundtruth_sq_dists is not None:
        xb_t = (
            torch.from_numpy(x_base)
            if isinstance(x_base, np.ndarray)
            else x_base.detach()
        )
        gt_d2_t = (
            torch.from_numpy(np.asarray(groundtruth_sq_dists, dtype=np.float64)).to(device=dev)
            if not isinstance(groundtruth_sq_dists, torch.Tensor)
            else groundtruth_sq_dists.detach().to(device=dev, dtype=torch.float64)
        )
        pred_true_d2 = torch.full((n_q, r10_eff), float("inf"), dtype=torch.float64, device=dev)
        for q0 in range(0, n_q, 1024):
            q1 = min(q0 + 1024, n_q)
            pids = pred_t[q0:q1, :r10_eff]
            v_m = pids >= 0
            safe_p = pids.clamp_min(0).cpu() if not xb_t.is_cuda else pids.clamp_min(0)
            xb_sel = xb_t[safe_p].to(device=dev, dtype=torch.float64)
            q64 = xq_t[q0:q1].double()[:, None, :]
            d2_exact = (q64 - xb_sel).square().sum(dim=2)
            pred_true_d2[q0:q1] = torch.where(v_m, d2_exact, torch.full_like(d2_exact, float("inf")))

        thr1 = gt_d2_t[:, 0:1] * (1.0 + 1e-11)
        thr10 = gt_d2_t[:, r10_eff - 1 : r10_eff] * (1.0 + 1e-11)
        tie_r1 = float((pred_true_d2[:, 0:1] <= thr1).double().mean().item())
        tie_one_r10 = float((pred_true_d2.min(dim=1).values[:, None] <= thr1).double().mean().item())
        tie_r10 = float(((pred_true_d2 <= thr10).double().sum(dim=1) / float(r10_eff)).mean().item())
    else:
        tie_r1 = id_r1
        tie_one_r10 = one_r10
        tie_r10 = id_r10

    qps = float(n_q) / max(search_sec, 1e-12)
    return ADCRecallMetrics(
        recall_at_1=id_r1,
        one_recall_at_10=one_r10,
        recall_at_10=id_r10,
        tie_recall_at_1=tie_r1,
        tie_one_recall_at_10=tie_one_r10,
        tie_recall_at_10=tie_r10,
        search_time_sec=search_sec,
        qps=qps,
    )


@torch.no_grad()
def warmup_pq_rvq_kernels(
    device: torch.device | str,
    pq_shapes: Sequence[tuple[int, int, int]] = ((128, 16, 256), (100, 10, 256), (768, 24, 256), (256, 16, 256)),
    rvq_shapes: Sequence[tuple[int, int, int]] = ((128, 512, 512),),
) -> None:
    """Pre-compile all `ApexPQ` & `ApexRVQ` Triton/CUDAGraph specializations (`USE_WIDE_SCHED={False,True}`)."""
    dev = torch.device(device)
    if dev.type != "cuda" or not HAS_TRITON:
        return
    with torch.cuda.device(dev):
        for d, m, ks in pq_shapes:
            n_small = max(2048, ks * 4)
            n_wide = max(65536, ks * 128)
            xw_small = torch.randn((n_small, d), device=dev, dtype=torch.float32)
            xw_wide = torch.randn((n_wide, d), device=dev, dtype=torch.float32)
            qw = torch.randn((32, d), device=dev, dtype=torch.float32)
            for prec in ("bf16", "croiss_fp8"):
                if prec == "croiss_fp8" and not _has_sm89_fp8(dev):
                    continue
                # 1. `USE_WIDE_SCHED=False` specialization
                pq_s = ApexPQ(
                    d=d,
                    m=m,
                    ks=ks,
                    nepochs=1.0,
                    solver="apex_emeans",
                    precision=prec,
                    use_cuda_graph=True,
                    force_cuda_graph=True,
                    seed=777,
                )
                _, codes_s = pq_s.fit(xw_small, capture_codes=True)
                assert codes_s is not None
                _ = pq_s.adc_search(qw, codes_s, k_top=10)
                if n_small >= 64:
                    _ = pq_s.adc_search(qw, codes_s, k_top=64)

                # 2. `USE_WIDE_SCHED=True` + natural 10-step `CUDAGraph` specialization (`#219` item 1)
                pq_w = ApexPQ(
                    d=d,
                    m=m,
                    ks=ks,
                    nepochs=2.0,
                    solver="apex_emeans",
                    precision=prec,
                    use_cuda_graph=True,
                    seed=778,
                )
                _ = pq_w.fit(xw_wide, capture_codes=False)

        for d, k1, k2 in rvq_shapes:
            warmup_apex_kernels(dev, d=d, k=k1)
            if k2 != k1:
                warmup_apex_kernels(dev, d=d, k=k2)
            n_w = max(2048, max(k1, k2) * 2)
            xw = torch.randn((n_w, d), device=dev, dtype=torch.float32)
            qw = torch.randn((32, d), device=dev, dtype=torch.float32)
            rvq_w = ApexRVQ(d=d, k1=k1, k2=k2, nepochs=1.0, solver="apex_emeans", adc_beam=min(32, k1 // 2), seed=779)
            _, codes_r = rvq_w.fit(xw, capture_codes=True)
            assert codes_r is not None
            _ = rvq_w.adc_search(qw, codes_r, k_top=10)
            _ = rvq_w.adc_search(qw, codes_r, k_top=10, adc_beam=None)
    torch.cuda.synchronize(dev)


__all__ = [
    "ADCRecallMetrics",
    "ApexPQ",
    "ApexRVQ",
    "CANONICAL_SOLVER_ALIASES",
    "PQ_SUPPORTED_SOLVERS",
    "RVQInvertedIndex",
    "RVQ_SUPPORTED_SOLVERS",
    "ReconstructionMetrics",
    "evaluate_adc_recall",
    "evaluate_vq_reconstruction_fp64",
    "instantiate_subproblem_solver",
    "normalize_solver_name",
    "pack_rvq_codes_uint8",
    "unpack_rvq_codes_uint8",
    "warmup_pq_rvq_kernels",
]
