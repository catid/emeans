"""CroissAttn-Inspired Ultra-Stable FP8 E4M3 & Native Blackwell MXFP8 Kernels for Clustering.

Translates the core low-precision execution contract from ``catid/CroissAttn``
(`croissattn/nvfp4.py`, `croissattn/attention.py`, `docs/fix_contract.md`) to
Euclidean nearest-centroid assignment and online k-means on NVIDIA Blackwell GPUs:

  1. **Saved-Payload Score Replay & Exact Represented-Norm Matching** (`fix_contract.md` #1–2):
     Every distance score is evaluated strictly on the quantized payloads
     ``x_tilde_i = mu + sx_i * qx_i`` and ``c_tilde_j = mu + sc_j * qc_j`` with
     true FP32 MMA accumulation (`tl.dot(..., max_num_imprecise_acc=0, out_dtype=tl.float32)`):
       ``||x_tilde_i - c_tilde_j||_2^2 = x_norm_i + c_norm_j - 2 * sx_i * (sc_j * <qx_i, qc_j>_fp32) >= 0``
     where ``x_norm_i = sx_i^2 * ||qx_i||_2^2`` is precomputed **once** inside the fused
     dataset pre-encoder (eliminating per-epoch ``row_norm`` reductions and double HBM loads).

  2. **Outlier-Proof Translation-Equivariant Origin + Decoupled Per-Vector Continuous Scales**:
     * Unlike arithmetic-mean centering (which gets dragged by heavy-tailed outlier modes and
       corrupts fine clusters near the origin) or global power-of-two FP8 scaling (which wastes
       up to 3–4 exponent octaves and underflows small clusters to zero), ``croiss_encode_dataset``
       centers by the **odd-sample FP64 coordinate-wise median** (100% translation-equivariant
       ``T(X + S) = T(X) + S``, Sterbenz-exact on shifted FP32 inputs, immune to <50% outlier modes)
       and assigns every row ``i`` and centroid ``j`` its own continuous FP32 scale ``sx_i, sc_j``
       hitting the full ``[-448, +448]`` ``float8_e4m3fn`` dynamic range.
     * Also provides ``scale_mode="mxfp8"`` using native Blackwell SM120 ``tl.dot_scaled``
       with ``CroissAttn``-style hierarchical per-32-feature ``ceil_e8m0`` block scales along
       contraction axis ``D`` for extreme intra-row multi-scale features.

  3. **Zero-`0x7f`-NaN Saturation Contract & Automatic Dynamic-Range Conditioning**:
     Applies ``CroissAttn``'s positive scale floor (`FP32_TINY = 1.17549435e-38`), upward-safe
     reciprocal guard, explicit `[-448.0, 448.0]` saturation clamp, canonical `+0.0` masking on
     zero/padded lanes, `fp_downcast_rounding="rtne"`, and sync-free GPU power-of-two dynamic-range
     conditioning (`1e-300 .. 1e+300`), guaranteeing **zero `0x7f`/`0xff` E4M3FN NaN bytes** and
     **zero `NaN`/`Inf` outputs** across `zeros`, `constants`, `+20,000` shifts, and `+-1e25`.

  4. **100% `torch.cuda.CUDAGraph`-Capturable & SM120 3-Stage TMA + Split-K Execution**:
     * Zero `.item()` / `.tolist()` host syncs and zero dynamic allocations inside step calls.
     * Supports on-device 0D `batch_offset` / `batch_offset_ptr` tensors with branchless cyclic
       wraparound (`_qx_wrap` modular guard tail) directly inside SM120 3-stage TMA hardware loads.
     * Automatic 2D Split-K dispatch on small mini-batches using pre-allocated workspace buffers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    HAS_TRITON = False

try:
    from triton.tools.tensor_descriptor import TensorDescriptor

    HAS_TMA_DESC = callable(getattr(TensorDescriptor, "from_tensor", None))
except ImportError:
    TensorDescriptor = None  # type: ignore[assignment]
    HAS_TMA_DESC = False

FLOAT8_E4M3FN = getattr(torch, "float8_e4m3fn", None)
CROISS_FP8_AVAILABLE: bool = bool(HAS_TRITON and (FLOAT8_E4M3FN is not None))
HAS_DOT_SCALED: bool = bool(HAS_TRITON and hasattr(tl, "dot_scaled"))

FP8_E4M3_MAX: float = 448.0
FP32_SCALE_FLOOR: float = 1.17549435e-38  # torch.finfo(torch.float32).tiny (matches croissattn/nvfp4.py)
FP32_FINITE_CAP: float = 3.4028234e38
MXFP8_BLOCK_SIZE: int = 32
MAX_WRAP_PAD: int = 256
MAX_SPLIT_K: int = 8
_INT32_MAX: int = 2**31 - 1


def _is_mxfp8_device_supported(device: torch.device | None = None) -> bool:
    if not (CROISS_FP8_AVAILABLE and HAS_DOT_SCALED and torch.cuda.is_available()):
        return False
    dev = device if (device is not None and device.type == "cuda") else torch.device("cuda", torch.cuda.current_device())
    cap = torch.cuda.get_device_capability(dev)
    return cap >= (10, 0)


CROISS_MXFP8_AVAILABLE: bool = _is_mxfp8_device_supported()


def compute_croiss_ozaki_count(d: int) -> int:
    """Number of dominant coordinates expanded via 3-lane Double-FP8 Ozaki residuals (`<= 6`)."""
    if d <= 0:
        return 0
    return min(int(d), 6)


def compute_croiss_padded_d(d: int, scale_mode: str = "decoupled") -> int:
    """Choose TMA- & Tensor-Core-aligned feature stride with zero partial-tile waste."""
    if d <= 0:
        raise ValueError(f"Feature dimension d must be positive; got {d}")
    if scale_mode not in ("decoupled", "ozaki", "mxfp8"):
        raise ValueError(f"scale_mode must be 'decoupled', 'ozaki', or 'mxfp8'; got {scale_mode!r}")
    eff_d = d + 3 * compute_croiss_ozaki_count(d) if scale_mode == "ozaki" else d
    if eff_d <= 64:
        return 64
    if eff_d <= 128:
        return 128
    if eff_d <= 256:
        return 256
    return ((eff_d + 127) // 128) * 128


def should_use_croiss_fp8(
    n: int,
    k: int,
    d: int,
    nepochs: float = 1.0,
    device: torch.device | None = None,
) -> bool:
    """Calibrated crossover heuristic where CroissFP8 beats BF16 in total wall-clock (`preencode + train`)."""
    if not CROISS_FP8_AVAILABLE or not torch.cuda.is_available():
        return False
    if device is not None and device.type != "cuda":
        return False
    dev = device if device is not None else torch.device("cuda", torch.cuda.current_device())
    cap = torch.cuda.get_device_capability(dev)
    if cap < (8, 9):
        return False
    if n < 16384 or k < 256 or d < 32:
        return False
    eff_epochs = max(float(nepochs), 0.25)
    work = float(n) * float(k) * eff_epochs
    min_work = 1.6e8 if d < 48 else 8.0e7
    return work >= min_work


def _tma_ready_2d(tensor: torch.Tensor) -> bool:
    if not HAS_TMA_DESC or not tensor.is_cuda or tensor.ndim != 2:
        return False
    cap = torch.cuda.get_device_capability(tensor.device)
    if cap < (9, 0):
        return False
    rows, cols = int(tensor.shape[0]), int(tensor.shape[1])
    if rows <= 0 or cols <= 0 or rows > _INT32_MAX or cols > _INT32_MAX:
        return False
    byte_pitch = int(tensor.stride(0)) * tensor.element_size()
    return (
        int(tensor.stride(1)) == 1
        and (int(tensor.data_ptr()) % 16 == 0)
        and (0 < byte_pitch < 2**40)
        and (byte_pitch % 16 == 0)
    )


def _choose_croiss_split_k(b: int, k: int, block_b: int, num_sms: int) -> int:
    """Select Split-K partitions (`1, 2, 4, 8`) when small mini-batches under-fill Blackwell SMs."""
    if b <= 0 or b > 12288 or k < 1024:
        return 1
    base_ctas = (b + block_b - 1) // block_b
    if base_ctas >= num_sms:
        return 1
    target_parts = max(1, (num_sms + base_ctas - 1) // max(base_ctas, 1))
    max_parts_by_k = max(1, k // 512)
    raw = min(MAX_SPLIT_K, target_parts, max_parts_by_k)
    if raw >= 8:
        return 8
    if raw >= 4:
        return 4
    if raw >= 2:
        return 2
    return 1


# ==============================================================================
# Triton Kernels: Calibration, Fused Encode, Centroid Prep, TMA/Ptr/Split-K/MXFP8 Assign
# ==============================================================================

if HAS_TRITON:
    _TL_FP32_TINY = tl.constexpr(1.17549435e-38)
    _TL_SAFE_COORD = tl.constexpr(1.0e17)
    _TL_INV_448_SAFE = tl.constexpr(1.00000024 / 448.0)
    _TL_FP8_MAX = tl.constexpr(448.0)
    _TL_FP32_CAP = tl.constexpr(3.4028234e38)

    @triton.jit(do_not_specialize=["N", "D", "stride_xn", "stride_xd", "step_n", "num_samples"])
    def _croiss_span_scan_kernel(
        x_ptr,
        mu64_ptr,
        span_max_ptr,
        N,
        D,
        stride_xn,
        stride_xd,
        step_n,
        num_samples,
        HAS_MU: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Fast strided dynamic-range scan computing max |x - mu| in FP64 with zero host syncs."""
        pid = tl.program_id(0)
        offs_s = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_s = offs_s < num_samples
        rows = (offs_s.to(tl.int64) * step_n) % N

        local_max = tl.zeros((), dtype=tl.float64)
        for d0 in range(0, D, BLOCK_D):
            cols = d0 + tl.arange(0, BLOCK_D)
            mask_d = cols < D
            mask_sd = mask_s[:, None] & mask_d[None, :]
            x_raw = tl.load(
                x_ptr + rows[:, None] * stride_xn + cols[None, :].to(tl.int64) * stride_xd,
                mask=mask_sd,
                other=0.0,
            ).to(tl.float64)
            if HAS_MU:
                mu64 = tl.load(mu64_ptr + cols, mask=mask_d, other=0.0).to(tl.float64)
                diff = x_raw - mu64[None, :]
            else:
                diff = x_raw
            # Ignore non-finite entries during span calibration so one rogue Inf/NaN never poisons scale
            finite = (diff == diff) & (tl.abs(diff) < 1.0e300)
            clean_abs = tl.where(mask_sd & finite, tl.abs(diff), 0.0)
            tile_max = tl.max(clean_abs)
            local_max = tl.maximum(local_max, tile_max)

        # Store per-CTA max into float32/float64 scratch slot (`num_ctas <= 256`)
        tl.store(span_max_ptr + pid, local_max)

    @triton.jit(do_not_specialize=["num_ctas"])
    def _croiss_finalize_gscale_kernel(
        span_max_ptr,
        gscale_ptr,
        inv_gscale_sq_ptr,
        coord_scale_val,
        num_ctas,
        FORCE_COORD_SCALE: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        """1-CTA on-device power-of-two conditioning scale computation (`0` host syncs)."""
        if FORCE_COORD_SCALE:
            gs = coord_scale_val.to(tl.float64)
            safe_gs = tl.where(gs > 0.0, gs, 1.0)
            tl.store(gscale_ptr, safe_gs)
            tl.store(inv_gscale_sq_ptr, 1.0)
            return

        offs = tl.arange(0, BLOCK_C)
        vals = tl.load(span_max_ptr + offs, mask=offs < num_ctas, other=0.0).to(tl.float64)
        span = tl.max(vals, axis=0)

        need_rescale = (span > 1.0e10) | ((span > 0.0) & (span < 1.0e-10))
        safe_span = tl.where(span > 0.0, span, 1.0)
        log2_span = tl.math.floor(tl.math.log2(safe_span) + 0.5)
        log2_span = tl.minimum(tl.maximum(log2_span, -480.0), 480.0)
        gs_pow2 = tl.math.exp2(-log2_span)
        inv_gs2_pow2 = tl.math.exp2(2.0 * log2_span)

        gs = tl.where(need_rescale, gs_pow2, 1.0)
        inv_gs2 = tl.where(need_rescale, inv_gs2_pow2, 1.0)
        tl.store(gscale_ptr, gs)
        tl.store(inv_gscale_sq_ptr, inv_gs2)

    @triton.jit(do_not_specialize=["N", "N_phys", "D", "stride_xn", "stride_xd", "stride_qn"])
    def _croiss_encode_decoupled_full_d_kernel(
        x_ptr,
        row_idx_ptr,
        mu_ptr,
        gscale_ptr,
        qx_ptr,
        sx_ptr,
        xnorm_ptr,
        N,
        N_phys,
        D,
        stride_xn,
        stride_xd,
        stride_qn,
        HAS_INDICES: tl.constexpr,
        HAS_MU: tl.constexpr,
        USE_FP64_DIFF: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Single-pass fused centering + per-row continuous FP8 scale + clamp + rtne cast + x_norm + wrap tail."""
        pid_m = tl.program_id(0)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < N_phys
        logical_m = offs_m % N

        if HAS_INDICES:
            src_rows = tl.load(row_idx_ptr + logical_m, mask=mask_m, other=0).to(tl.int64)
        else:
            src_rows = logical_m.to(tl.int64)

        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        mask_md = mask_m[:, None] & mask_d[None, :]

        x_raw = tl.load(
            x_ptr + src_rows[:, None] * stride_xn + offs_d[None, :].to(tl.int64) * stride_xd,
            mask=mask_md,
            other=0.0,
        )

        if USE_FP64_DIFF:
            gs64 = tl.load(gscale_ptr).to(tl.float64)
            x64 = x_raw.to(tl.float64)
            if HAS_MU:
                mu64 = tl.load(mu_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float64)
                diff64 = (x64 - mu64[None, :]) * gs64
            else:
                diff64 = x64 * gs64
            finite64 = (diff64 == diff64) & (tl.abs(diff64) < 1.0e300)
            diff64 = tl.where(mask_md & finite64, diff64, 0.0)
            diff64 = tl.minimum(tl.maximum(diff64, -1.0e17), 1.0e17)
            val = diff64.to(tl.float32)
        else:
            gs32 = tl.load(gscale_ptr).to(tl.float32)
            x32 = x_raw.to(tl.float32)
            if HAS_MU:
                mu32 = tl.load(mu_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32)
                diff32 = (x32 - mu32[None, :]) * gs32
            else:
                diff32 = x32 * gs32
            z32_init = tl.zeros((), dtype=tl.float32)
            # Guard against +-Inf before casting/clamping
            finite32 = (diff32 == diff32) & (tl.abs(diff32) <= _TL_FP32_CAP)
            val = tl.where(mask_md & finite32, diff32, z32_init)

        z32 = tl.zeros((), dtype=tl.float32)
        lim32 = tl.full((), _TL_SAFE_COORD, dtype=tl.float32)
        val = tl.minimum(tl.maximum(val, -lim32), lim32)

        amax = tl.max(tl.abs(val), axis=1).to(tl.float32)
        tiny32 = tl.full((), _TL_FP32_TINY, dtype=tl.float32)
        inv448 = tl.full((), _TL_INV_448_SAFE, dtype=tl.float32)
        one32 = tl.full((), 1.0, dtype=tl.float32)
        max448 = tl.full((), _TL_FP8_MAX, dtype=tl.float32)

        sx = tl.maximum(amax * inv448, tiny32)
        inv_sx = tl.where(amax > z32, one32 / sx, z32)

        scaled = tl.minimum(tl.maximum(val * inv_sx[:, None], -max448), max448)
        scaled = tl.where(mask_md & (amax[:, None] > z32), scaled, z32).to(tl.float32)
        q = scaled.to(tl.float8e4nv, fp_downcast_rounding="rtne")

        qf = q.to(tl.float32)
        u_norm = tl.sum(qf * qf, axis=1)
        x_norm = (sx * sx) * u_norm

        tl.store(
            qx_ptr + offs_m[:, None].to(tl.int64) * stride_qn + offs_d[None, :].to(tl.int64),
            q,
            mask=mask_m[:, None] & (offs_d[None, :] < stride_qn),
        )
        tl.store(sx_ptr + offs_m, sx, mask=mask_m)
        tl.store(xnorm_ptr + offs_m, x_norm, mask=mask_m)

    @triton.jit(do_not_specialize=["N", "N_phys", "D", "stride_xn", "stride_xd", "stride_qn"])
    def _croiss_encode_decoupled_tiled_d_kernel(
        x_ptr,
        row_idx_ptr,
        mu_ptr,
        gscale_ptr,
        qx_ptr,
        sx_ptr,
        xnorm_ptr,
        N,
        N_phys,
        D,
        stride_xn,
        stride_xd,
        stride_qn,
        HAS_INDICES: tl.constexpr,
        HAS_MU: tl.constexpr,
        USE_FP64_DIFF: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Tiled-D (`D > 256`) fused per-row decoupled FP8 encoder with exact represented x_norm."""
        pid_m = tl.program_id(0)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < N_phys
        logical_m = offs_m % N

        if HAS_INDICES:
            src_rows = tl.load(row_idx_ptr + logical_m, mask=mask_m, other=0).to(tl.int64)
        else:
            src_rows = logical_m.to(tl.int64)

        z32 = tl.zeros((), dtype=tl.float32)
        lim32 = tl.full((), _TL_SAFE_COORD, dtype=tl.float32)
        amax = tl.zeros((BLOCK_M,), dtype=tl.float32)

        gs64 = tl.load(gscale_ptr).to(tl.float64)
        gs32 = gs64.to(tl.float32)

        for d0 in range(0, D, BLOCK_D):
            cols = d0 + tl.arange(0, BLOCK_D)
            mask_d = cols < D
            mask_md = mask_m[:, None] & mask_d[None, :]
            x_raw = tl.load(
                x_ptr + src_rows[:, None] * stride_xn + cols[None, :].to(tl.int64) * stride_xd,
                mask=mask_md,
                other=0.0,
            )
            if USE_FP64_DIFF:
                x64 = x_raw.to(tl.float64)
                if HAS_MU:
                    mu64 = tl.load(mu_ptr + cols, mask=mask_d, other=0.0).to(tl.float64)
                    diff64 = (x64 - mu64[None, :]) * gs64
                else:
                    diff64 = x64 * gs64
                fin64 = (diff64 == diff64) & (tl.abs(diff64) < 1.0e300)
                diff64 = tl.where(mask_md & fin64, diff64, 0.0)
                diff64 = tl.minimum(tl.maximum(diff64, -1.0e17), 1.0e17)
                v_tile = diff64.to(tl.float32)
            else:
                x32 = x_raw.to(tl.float32)
                if HAS_MU:
                    mu32 = tl.load(mu_ptr + cols, mask=mask_d, other=0.0).to(tl.float32)
                    diff32 = (x32 - mu32[None, :]) * gs32
                else:
                    diff32 = x32 * gs32
                fin32 = (diff32 == diff32) & (tl.abs(diff32) <= _TL_FP32_CAP)
                v_tile = tl.where(mask_md & fin32, diff32, z32)
                v_tile = tl.minimum(tl.maximum(v_tile, -lim32), lim32)
            amax = tl.maximum(amax, tl.max(tl.abs(v_tile), axis=1).to(tl.float32))

        tiny32 = tl.full((), _TL_FP32_TINY, dtype=tl.float32)
        inv448 = tl.full((), _TL_INV_448_SAFE, dtype=tl.float32)
        one32 = tl.full((), 1.0, dtype=tl.float32)
        max448 = tl.full((), _TL_FP8_MAX, dtype=tl.float32)

        sx = tl.maximum(amax * inv448, tiny32)
        inv_sx = tl.where(amax > z32, one32 / sx, z32)
        u_norm = tl.zeros((BLOCK_M,), dtype=tl.float32)

        for d0 in range(0, stride_qn, BLOCK_D):
            cols = d0 + tl.arange(0, BLOCK_D)
            mask_d = cols < D
            mask_md = mask_m[:, None] & mask_d[None, :]
            x_raw = tl.load(
                x_ptr + src_rows[:, None] * stride_xn + cols[None, :].to(tl.int64) * stride_xd,
                mask=mask_md,
                other=0.0,
            )
            if USE_FP64_DIFF:
                x64 = x_raw.to(tl.float64)
                if HAS_MU:
                    mu64 = tl.load(mu_ptr + cols, mask=mask_d, other=0.0).to(tl.float64)
                    diff64 = (x64 - mu64[None, :]) * gs64
                else:
                    diff64 = x64 * gs64
                fin64 = (diff64 == diff64) & (tl.abs(diff64) < 1.0e300)
                diff64 = tl.where(mask_md & fin64, diff64, 0.0)
                diff64 = tl.minimum(tl.maximum(diff64, -1.0e17), 1.0e17)
                v_tile = diff64.to(tl.float32)
            else:
                x32 = x_raw.to(tl.float32)
                if HAS_MU:
                    mu32 = tl.load(mu_ptr + cols, mask=mask_d, other=0.0).to(tl.float32)
                    diff32 = (x32 - mu32[None, :]) * gs32
                else:
                    diff32 = x32 * gs32
                fin32 = (diff32 == diff32) & (tl.abs(diff32) <= _TL_FP32_CAP)
                v_tile = tl.where(mask_md & fin32, diff32, z32)
                v_tile = tl.minimum(tl.maximum(v_tile, -lim32), lim32)

            scaled = tl.minimum(tl.maximum(v_tile * inv_sx[:, None], -max448), max448)
            scaled = tl.where(mask_md & (amax[:, None] > z32), scaled, z32).to(tl.float32)
            q = scaled.to(tl.float8e4nv, fp_downcast_rounding="rtne")
            tl.store(
                qx_ptr + offs_m[:, None].to(tl.int64) * stride_qn + cols[None, :].to(tl.int64),
                q,
                mask=mask_m[:, None] & (cols[None, :] < stride_qn),
            )
            qf = q.to(tl.float32)
            u_norm += tl.sum(qf * qf, axis=1)

        x_norm = (sx * sx) * u_norm
        tl.store(sx_ptr + offs_m, sx, mask=mask_m)
        tl.store(xnorm_ptr + offs_m, x_norm, mask=mask_m)

    @triton.jit(do_not_specialize=["N", "N_phys", "D", "stride_xn", "stride_xd", "stride_qn", "stride_un"])
    def _croiss_encode_mxfp8_kernel(
        x_ptr,
        row_idx_ptr,
        mu_ptr,
        gscale_ptr,
        qx_u8_ptr,
        ux_u8_ptr,
        sx_ptr,
        xnorm_ptr,
        N,
        N_phys,
        D,
        stride_xn,
        stride_xd,
        stride_qn,
        stride_un,
        HAS_INDICES: tl.constexpr,
        HAS_MU: tl.constexpr,
        USE_FP64_DIFF: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        """CroissAttn-style block-32 ceil-E8M0 + E4M3FN encoder along contraction axis D."""
        pid_m = tl.program_id(0)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < N_phys
        logical_m = offs_m % N

        if HAS_INDICES:
            src_rows = tl.load(row_idx_ptr + logical_m, mask=mask_m, other=0).to(tl.int64)
        else:
            src_rows = logical_m.to(tl.int64)

        gs64 = tl.load(gscale_ptr).to(tl.float64)
        gs32 = gs64.to(tl.float32)
        z32 = tl.zeros((), dtype=tl.float32)
        lim32 = tl.full((), _TL_SAFE_COORD, dtype=tl.float32)
        max448 = tl.full((), _TL_FP8_MAX, dtype=tl.float32)
        inv448 = tl.full((), _TL_INV_448_SAFE, dtype=tl.float32)

        x_norm_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
        row_amax = tl.zeros((BLOCK_M,), dtype=tl.float32)
        num_blocks = stride_qn // 32

        for b_idx in range(0, num_blocks):
            d0 = b_idx * 32
            cols = d0 + tl.arange(0, 32)
            mask_d = cols < D
            mask_md = mask_m[:, None] & mask_d[None, :]
            x_raw = tl.load(
                x_ptr + src_rows[:, None] * stride_xn + cols[None, :].to(tl.int64) * stride_xd,
                mask=mask_md,
                other=0.0,
            )
            if USE_FP64_DIFF:
                x64 = x_raw.to(tl.float64)
                if HAS_MU:
                    mu64 = tl.load(mu_ptr + cols, mask=mask_d, other=0.0).to(tl.float64)
                    diff64 = (x64 - mu64[None, :]) * gs64
                else:
                    diff64 = x64 * gs64
                fin64 = (diff64 == diff64) & (tl.abs(diff64) < 1.0e300)
                diff64 = tl.where(mask_md & fin64, diff64, 0.0)
                diff64 = tl.minimum(tl.maximum(diff64, -1.0e17), 1.0e17)
                val = diff64.to(tl.float32)
            else:
                x32 = x_raw.to(tl.float32)
                if HAS_MU:
                    mu32 = tl.load(mu_ptr + cols, mask=mask_d, other=0.0).to(tl.float32)
                    diff32 = (x32 - mu32[None, :]) * gs32
                else:
                    diff32 = x32 * gs32
                fin32 = (diff32 == diff32) & (tl.abs(diff32) <= _TL_FP32_CAP)
                val = tl.where(mask_md & fin32, diff32, z32)
                val = tl.minimum(tl.maximum(val, -lim32), lim32)

            bamax = tl.max(tl.abs(val), axis=1).to(tl.float32)
            row_amax = tl.maximum(row_amax, bamax)
            pos = bamax > z32
            safe_target = tl.where(pos, bamax * inv448, 1.0).to(tl.float32)
            exp_f = tl.math.ceil(tl.math.log2(safe_target))
            exp_i = tl.minimum(tl.maximum(exp_f.to(tl.int32), -127), 127)
            exp_i = tl.where(pos, exp_i, 0)
            u_byte = (exp_i + 127).to(tl.uint8)
            sb = tl.math.exp2(exp_i.to(tl.float32))
            inv_sb = tl.where(pos, 1.0 / sb, z32).to(tl.float32)

            scaled = tl.minimum(tl.maximum(val * inv_sb[:, None], -max448), max448)
            scaled = tl.where(mask_md & pos[:, None], scaled, z32).to(tl.float32)
            q = scaled.to(tl.float8e4nv, fp_downcast_rounding="rtne")
            qf = q.to(tl.float32)
            x_norm_acc += (sb * sb) * tl.sum(qf * qf, axis=1)

            q_u8 = q.to(tl.int8, bitcast=True).to(tl.uint8)
            tl.store(
                qx_u8_ptr + offs_m[:, None].to(tl.int64) * stride_qn + cols[None, :].to(tl.int64),
                q_u8,
                mask=mask_m[:, None],
            )
            tl.store(ux_u8_ptr + offs_m.to(tl.int64) * stride_un + b_idx, u_byte, mask=mask_m)

        tiny32 = tl.full((), _TL_FP32_TINY, dtype=tl.float32)
        tl.store(sx_ptr + offs_m, tl.maximum(row_amax * inv448, tiny32), mask=mask_m)
        tl.store(xnorm_ptr + offs_m, x_norm_acc, mask=mask_m)

    @triton.jit(do_not_specialize=["N", "N_phys", "D", "stride_xn", "stride_xd", "stride_qn", "M_OZAKI"])
    def _croiss_ozaki_append_kernel(
        x_ptr,
        row_idx_ptr,
        mu_ptr,
        gscale_ptr,
        dom_cols_ptr,
        q_ptr,
        s_ptr,
        norm_ptr,
        N,
        N_phys,
        D,
        stride_xn,
        stride_xd,
        stride_qn,
        M_OZAKI,
        IS_CENTROID: tl.constexpr,
        HAS_INDICES: tl.constexpr,
        HAS_MU: tl.constexpr,
        USE_FP64_DIFF: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        """Append 3-lane Bilinear Double-FP8 Ozaki residuals on top-M dominant variance coordinates."""
        pid_m = tl.program_id(0)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < N_phys
        logical_m = offs_m % N

        if HAS_INDICES:
            src_rows = tl.load(row_idx_ptr + logical_m, mask=mask_m, other=0).to(tl.int64)
        else:
            src_rows = logical_m.to(tl.int64)

        gs64 = tl.load(gscale_ptr).to(tl.float64)
        gs32 = gs64.to(tl.float32)
        s_val = tl.load(s_ptr + offs_m, mask=mask_m, other=1.0).to(tl.float32)
        z32 = tl.zeros((), dtype=tl.float32)
        lim32 = tl.full((), _TL_SAFE_COORD, dtype=tl.float32)
        max448 = tl.full((), _TL_FP8_MAX, dtype=tl.float32)
        tiny_thr = tl.full((), _TL_FP32_TINY * 1.5, dtype=tl.float32)
        active = mask_m & (s_val > tiny_thr)
        inv_s = tl.where(active, 1.0 / s_val, z32).to(tl.float32)

        delta_u = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for m_idx in range(0, M_OZAKI):
            col = tl.load(dom_cols_ptr + m_idx).to(tl.int64)
            x_raw = tl.load(x_ptr + src_rows * stride_xn + col * stride_xd, mask=mask_m, other=0.0)
            if USE_FP64_DIFF:
                x64 = x_raw.to(tl.float64)
                if HAS_MU:
                    mu64 = tl.load(mu_ptr + col).to(tl.float64)
                    diff64 = (x64 - mu64) * gs64
                else:
                    diff64 = x64 * gs64
                fin64 = (diff64 == diff64) & (tl.abs(diff64) < 1.0e300)
                diff64 = tl.where(mask_m & fin64, diff64, 0.0)
                diff64 = tl.minimum(tl.maximum(diff64, -1.0e17), 1.0e17)
                val = diff64.to(tl.float32)
            else:
                x32 = x_raw.to(tl.float32)
                if HAS_MU:
                    mu32 = tl.load(mu_ptr + col).to(tl.float32)
                    diff32 = (x32 - mu32) * gs32
                else:
                    diff32 = x32 * gs32
                fin32 = (diff32 == diff32) & (tl.abs(diff32) <= _TL_FP32_CAP)
                val = tl.where(mask_m & fin32, diff32, z32)
                val = tl.minimum(tl.maximum(val, -lim32), lim32)

            q_hi = tl.load(q_ptr + offs_m.to(tl.int64) * stride_qn + col, mask=mask_m, other=0.0)
            qf_hi = q_hi.to(tl.float32)
            v_scaled = tl.minimum(tl.maximum(val * inv_s, -max448), max448)
            res = tl.minimum(tl.maximum(v_scaled - qf_hi, -max448), max448)
            res = tl.where(active, res, z32).to(tl.float32)
            q_lo = res.to(tl.float8e4nv, fp_downcast_rounding="rtne")
            qf_lo = q_lo.to(tl.float32)

            comb = qf_hi + qf_lo
            delta_u += comb * comb - qf_hi * qf_hi

            aux0 = D + m_idx * 3
            row_base = offs_m.to(tl.int64) * stride_qn
            if IS_CENTROID:
                tl.store(q_ptr + row_base + aux0 + 0, q_lo, mask=mask_m)
                tl.store(q_ptr + row_base + aux0 + 1, q_hi, mask=mask_m)
                tl.store(q_ptr + row_base + aux0 + 2, q_lo, mask=mask_m)
            else:
                tl.store(q_ptr + row_base + aux0 + 0, q_hi, mask=mask_m)
                tl.store(q_ptr + row_base + aux0 + 1, q_lo, mask=mask_m)
                tl.store(q_ptr + row_base + aux0 + 2, q_lo, mask=mask_m)

        old_norm = tl.load(norm_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
        new_norm = tl.maximum(old_norm + (s_val * s_val) * delta_u, z32)
        tl.store(norm_ptr + offs_m, new_norm, mask=mask_m)

    # --------------------------------------------------------------------------
    # Assignment Kernels (SM120 3-Stage TMA, Pointer Int32/Int64, Split-K, MXFP8)
    # --------------------------------------------------------------------------

    @triton.jit(do_not_specialize=["B", "K", "N", "row_offset"])
    def _croiss_assign_tma_full_d_kernel(
        qx_desc,
        sx_ptr,
        xnorm_ptr,
        qc_desc,
        sc_ptr,
        cnorm_ptr,
        batch_off_ptr,
        inv_gs2_ptr,
        dist_ptr,
        lbl_ptr,
        B,
        K,
        N,
        row_offset,
        BLOCK_B: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
        HAS_BATCH_OFF: tl.constexpr,
        WRAP_ROWS: tl.constexpr,
        HAS_INV_GS2: tl.constexpr,
    ):
        """SM120 3-Stage TMA Full-D FP8 Tensor-Core Nearest-Centroid Kernel (`max_num_imprecise_acc=0`)."""
        pid_b = tl.program_id(0)
        b_local = pid_b * BLOCK_B
        offs_b = b_local + tl.arange(0, BLOCK_B)
        mask_b = offs_b < B

        if HAS_BATCH_OFF:
            raw_off = tl.load(batch_off_ptr).to(tl.int64)
        else:
            raw_off = row_offset + tl.zeros((), dtype=tl.int64)

        if WRAP_ROWS:
            n64 = N + tl.zeros((), dtype=tl.int64)
            base_mod = ((raw_off % n64) + n64) % n64
            b_src = ((base_mod + b_local) % n64).to(tl.int32)
        else:
            b_src = (raw_off + b_local).to(tl.int32)

        src_rows = b_src + tl.arange(0, BLOCK_B)

        qx = qx_desc.load([b_src, 0])
        two_sx = 2.0 * tl.load(sx_ptr + src_rows, mask=mask_b, other=0.0)
        x_norm = tl.load(xnorm_ptr + src_rows, mask=mask_b, other=0.0)

        best_score = tl.full((BLOCK_B,), float("inf"), dtype=tl.float32)
        best_lbl = tl.zeros((BLOCK_B,), dtype=tl.int32)

        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K
            qc = qc_desc.load([k0, 0])
            dots = tl.dot(qx, tl.trans(qc), max_num_imprecise_acc=0, out_dtype=tl.float32)
            sc = tl.load(sc_ptr + offs_k, mask=mask_k, other=0.0)
            cn = tl.load(cnorm_ptr + offs_k, mask=mask_k, other=float("inf"))
            scores = cn[None, :] - two_sx[:, None] * (dots * sc[None, :])
            scores = tl.where(mask_k[None, :], scores, float("inf"))
            b_min, b_idx = tl.min(
                scores,
                axis=1,
                return_indices=True,
                return_indices_tie_break_left=True,
            )
            imp = b_min < best_score
            best_score = tl.where(imp, b_min, best_score)
            best_lbl = tl.where(imp, k0 + b_idx.to(tl.int32), best_lbl)

        dist = tl.maximum(best_score + x_norm, 0.0)
        if HAS_INV_GS2:
            inv_gs2 = tl.load(inv_gs2_ptr).to(tl.float64)
            dist = tl.minimum(dist.to(tl.float64) * inv_gs2, _TL_FP32_CAP).to(tl.float32)
        tl.store(dist_ptr + offs_b, dist, mask=mask_b)
        tl.store(lbl_ptr + offs_b, best_lbl.to(tl.int64), mask=mask_b)

    @triton.jit(do_not_specialize=["B", "K", "N", "stride_pb", "row_offset"])
    def _croiss_splitk_tma_full_d_kernel(
        qx_desc,
        sx_ptr,
        qc_desc,
        sc_ptr,
        cnorm_ptr,
        batch_off_ptr,
        part_score_ptr,
        part_lbl_ptr,
        B,
        K,
        N,
        stride_pb,
        row_offset,
        PARTITIONS: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
        HAS_BATCH_OFF: tl.constexpr,
        WRAP_ROWS: tl.constexpr,
    ):
        """SM120 3-Stage TMA 2D Split-K FP8 Tensor-Core Kernel for small mini-batches."""
        pid_b = tl.program_id(0)
        part = tl.program_id(1)
        b_local = pid_b * BLOCK_B
        offs_b = b_local + tl.arange(0, BLOCK_B)
        mask_b = offs_b < B

        if HAS_BATCH_OFF:
            raw_off = tl.load(batch_off_ptr).to(tl.int64)
        else:
            raw_off = row_offset + tl.zeros((), dtype=tl.int64)

        if WRAP_ROWS:
            n64 = N + tl.zeros((), dtype=tl.int64)
            base_mod = ((raw_off % n64) + n64) % n64
            b_src = ((base_mod + b_local) % n64).to(tl.int32)
        else:
            b_src = (raw_off + b_local).to(tl.int32)

        src_rows = b_src + tl.arange(0, BLOCK_B)
        qx = qx_desc.load([b_src, 0])
        two_sx = 2.0 * tl.load(sx_ptr + src_rows, mask=mask_b, other=0.0)

        num_ktiles = (K + BLOCK_K - 1) // BLOCK_K
        tiles_per_part = (num_ktiles + PARTITIONS - 1) // PARTITIONS
        k_start = part * tiles_per_part * BLOCK_K
        k_end = tl.minimum(k_start + tiles_per_part * BLOCK_K, K)

        best_score = tl.full((BLOCK_B,), float("inf"), dtype=tl.float32)
        best_lbl = tl.zeros((BLOCK_B,), dtype=tl.int32)

        k0 = k_start
        while k0 < k_end:
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < k_end
            qc = qc_desc.load([k0, 0])
            dots = tl.dot(qx, tl.trans(qc), max_num_imprecise_acc=0, out_dtype=tl.float32)
            sc = tl.load(sc_ptr + offs_k, mask=mask_k, other=0.0)
            cn = tl.load(cnorm_ptr + offs_k, mask=mask_k, other=float("inf"))
            scores = cn[None, :] - two_sx[:, None] * (dots * sc[None, :])
            scores = tl.where(mask_k[None, :], scores, float("inf"))
            b_min, b_idx = tl.min(
                scores,
                axis=1,
                return_indices=True,
                return_indices_tie_break_left=True,
            )
            imp = b_min < best_score
            best_score = tl.where(imp, b_min, best_score)
            best_lbl = tl.where(imp, k0 + b_idx.to(tl.int32), best_lbl)
            k0 += BLOCK_K

        p_off = offs_b.to(tl.int64) * stride_pb + part
        tl.store(part_score_ptr + p_off, best_score, mask=mask_b)
        tl.store(part_lbl_ptr + p_off, best_lbl, mask=mask_b)

    @triton.jit(do_not_specialize=["B", "K", "N", "row_offset"])
    def _croiss_assign_tma_tiled_d_kernel(
        qx_desc,
        sx_ptr,
        xnorm_ptr,
        qc_desc,
        sc_ptr,
        cnorm_ptr,
        batch_off_ptr,
        inv_gs2_ptr,
        dist_ptr,
        lbl_ptr,
        B,
        K,
        N,
        row_offset,
        D_PAD: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
        HAS_BATCH_OFF: tl.constexpr,
        WRAP_ROWS: tl.constexpr,
        HAS_INV_GS2: tl.constexpr,
    ):
        """SM120 3-Stage TMA Tiled-D (`D > 256`) FP8 Tensor-Core Nearest-Centroid Kernel."""
        pid_b = tl.program_id(0)
        b_local = pid_b * BLOCK_B
        offs_b = b_local + tl.arange(0, BLOCK_B)
        mask_b = offs_b < B

        if HAS_BATCH_OFF:
            raw_off = tl.load(batch_off_ptr).to(tl.int64)
        else:
            raw_off = row_offset + tl.zeros((), dtype=tl.int64)

        if WRAP_ROWS:
            n64 = N + tl.zeros((), dtype=tl.int64)
            base_mod = ((raw_off % n64) + n64) % n64
            b_src = ((base_mod + b_local) % n64).to(tl.int32)
        else:
            b_src = (raw_off + b_local).to(tl.int32)

        src_rows = b_src + tl.arange(0, BLOCK_B)
        two_sx = 2.0 * tl.load(sx_ptr + src_rows, mask=mask_b, other=0.0)
        x_norm = tl.load(xnorm_ptr + src_rows, mask=mask_b, other=0.0)

        best_score = tl.full((BLOCK_B,), float("inf"), dtype=tl.float32)
        best_lbl = tl.zeros((BLOCK_B,), dtype=tl.int32)

        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K
            dots = tl.zeros((BLOCK_B, BLOCK_K), dtype=tl.float32)
            for d0 in range(0, D_PAD, BLOCK_D):
                qx_tile = qx_desc.load([b_src, d0])
                qc_tile = qc_desc.load([k0, d0])
                dots = tl.dot(
                    qx_tile,
                    tl.trans(qc_tile),
                    acc=dots,
                    max_num_imprecise_acc=0,
                    out_dtype=tl.float32,
                )
            sc = tl.load(sc_ptr + offs_k, mask=mask_k, other=0.0)
            cn = tl.load(cnorm_ptr + offs_k, mask=mask_k, other=float("inf"))
            scores = cn[None, :] - two_sx[:, None] * (dots * sc[None, :])
            scores = tl.where(mask_k[None, :], scores, float("inf"))
            b_min, b_idx = tl.min(
                scores,
                axis=1,
                return_indices=True,
                return_indices_tie_break_left=True,
            )
            imp = b_min < best_score
            best_score = tl.where(imp, b_min, best_score)
            best_lbl = tl.where(imp, k0 + b_idx.to(tl.int32), best_lbl)

        dist = tl.maximum(best_score + x_norm, 0.0)
        if HAS_INV_GS2:
            inv_gs2 = tl.load(inv_gs2_ptr).to(tl.float64)
            dist = tl.minimum(dist.to(tl.float64) * inv_gs2, _TL_FP32_CAP).to(tl.float32)
        tl.store(dist_ptr + offs_b, dist, mask=mask_b)
        tl.store(lbl_ptr + offs_b, best_lbl.to(tl.int64), mask=mask_b)

    @triton.jit(do_not_specialize=["B", "K", "D_EFF", "N", "N_IDX", "stride_xb", "stride_ck", "row_offset"])
    def _croiss_assign_ptr_kernel(
        qx_ptr,
        sx_ptr,
        xnorm_ptr,
        qc_ptr,
        sc_ptr,
        cnorm_ptr,
        batch_off_ptr,
        row_idx_ptr,
        inv_gs2_ptr,
        dist_ptr,
        lbl_ptr,
        B,
        K,
        D_EFF,
        N,
        N_IDX,
        stride_xb,
        stride_ck,
        row_offset,
        BLOCK_B: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
        FULL_D: tl.constexpr,
        HAS_INDICES: tl.constexpr,
        HAS_BATCH_OFF: tl.constexpr,
        WRAP_ROWS: tl.constexpr,
        WIDE_X: tl.constexpr,
        WIDE_C: tl.constexpr,
        HAS_INV_GS2: tl.constexpr,
    ):
        """Pointer-gather / arbitrary-stride FP8 Tensor-Core Nearest-Centroid Kernel."""
        pid_b = tl.program_id(0)
        offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        mask_b = offs_b < B

        if HAS_BATCH_OFF:
            raw_off = tl.load(batch_off_ptr).to(tl.int64)
        else:
            raw_off = row_offset + tl.zeros((), dtype=tl.int64)

        if HAS_INDICES:
            pos64 = raw_off + offs_b.to(tl.int64)
            if WRAP_ROWS:
                n_idx64 = N_IDX + tl.zeros((), dtype=tl.int64)
                pos64 = ((pos64 % n_idx64) + n_idx64) % n_idx64
            src64 = tl.load(row_idx_ptr + pos64, mask=mask_b, other=0).to(tl.int64)
        else:
            src64 = raw_off + offs_b.to(tl.int64)
            if WRAP_ROWS:
                n64 = N + tl.zeros((), dtype=tl.int64)
                src64 = ((src64 % n64) + n64) % n64

        two_sx = 2.0 * tl.load(sx_ptr + src64, mask=mask_b, other=0.0)
        x_norm = tl.load(xnorm_ptr + src64, mask=mask_b, other=0.0)

        if FULL_D:
            offs_d = tl.arange(0, BLOCK_D)
            mask_d = offs_d < D_EFF
            if WIDE_X:
                x_off = src64[:, None] * stride_xb + offs_d[None, :].to(tl.int64)
            else:
                x_off = src64[:, None].to(tl.int32) * stride_xb + offs_d[None, :]
            qx_full = tl.load(qx_ptr + x_off, mask=mask_b[:, None] & mask_d[None, :], other=0.0)

        best_score = tl.full((BLOCK_B,), float("inf"), dtype=tl.float32)
        best_lbl = tl.zeros((BLOCK_B,), dtype=tl.int32)

        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K
            if FULL_D:
                offs_d = tl.arange(0, BLOCK_D)
                mask_d = offs_d < D_EFF
                if WIDE_C:
                    c_off = offs_k[:, None].to(tl.int64) * stride_ck + offs_d[None, :].to(tl.int64)
                else:
                    c_off = offs_k[:, None] * stride_ck + offs_d[None, :]
                qc_tile = tl.load(qc_ptr + c_off, mask=mask_k[:, None] & mask_d[None, :], other=0.0)
                dots = tl.dot(qx_full, tl.trans(qc_tile), max_num_imprecise_acc=0, out_dtype=tl.float32)
            else:
                dots = tl.zeros((BLOCK_B, BLOCK_K), dtype=tl.float32)
                for d0 in range(0, D_EFF, BLOCK_D):
                    offs_d = d0 + tl.arange(0, BLOCK_D)
                    mask_d = offs_d < D_EFF
                    if WIDE_X:
                        x_off = src64[:, None] * stride_xb + offs_d[None, :].to(tl.int64)
                    else:
                        x_off = src64[:, None].to(tl.int32) * stride_xb + offs_d[None, :]
                    if WIDE_C:
                        c_off = offs_k[:, None].to(tl.int64) * stride_ck + offs_d[None, :].to(tl.int64)
                    else:
                        c_off = offs_k[:, None] * stride_ck + offs_d[None, :]
                    qx_tile = tl.load(qx_ptr + x_off, mask=mask_b[:, None] & mask_d[None, :], other=0.0)
                    qc_tile = tl.load(qc_ptr + c_off, mask=mask_k[:, None] & mask_d[None, :], other=0.0)
                    dots = tl.dot(
                        qx_tile,
                        tl.trans(qc_tile),
                        acc=dots,
                        max_num_imprecise_acc=0,
                        out_dtype=tl.float32,
                    )

            sc = tl.load(sc_ptr + offs_k, mask=mask_k, other=0.0)
            cn = tl.load(cnorm_ptr + offs_k, mask=mask_k, other=float("inf"))
            scores = cn[None, :] - two_sx[:, None] * (dots * sc[None, :])
            scores = tl.where(mask_k[None, :], scores, float("inf"))
            b_min, b_idx = tl.min(
                scores,
                axis=1,
                return_indices=True,
                return_indices_tie_break_left=True,
            )
            imp = b_min < best_score
            best_score = tl.where(imp, b_min, best_score)
            best_lbl = tl.where(imp, k0 + b_idx.to(tl.int32), best_lbl)

        dist = tl.maximum(best_score + x_norm, 0.0)
        if HAS_INV_GS2:
            inv_gs2 = tl.load(inv_gs2_ptr).to(tl.float64)
            dist = tl.minimum(dist.to(tl.float64) * inv_gs2, _TL_FP32_CAP).to(tl.float32)
        tl.store(dist_ptr + offs_b.to(tl.int64), dist, mask=mask_b)
        tl.store(lbl_ptr + offs_b.to(tl.int64), best_lbl.to(tl.int64), mask=mask_b)

    @triton.jit(do_not_specialize=["B", "K", "D_EFF", "N", "N_IDX", "stride_xb", "stride_ck", "stride_pb", "row_offset"])
    def _croiss_splitk_assign_kernel(
        qx_ptr,
        sx_ptr,
        qc_ptr,
        sc_ptr,
        cnorm_ptr,
        batch_off_ptr,
        row_idx_ptr,
        part_score_ptr,
        part_lbl_ptr,
        B,
        K,
        D_EFF,
        N,
        N_IDX,
        stride_xb,
        stride_ck,
        stride_pb,
        row_offset,
        PARTITIONS: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
        FULL_D: tl.constexpr,
        HAS_INDICES: tl.constexpr,
        HAS_BATCH_OFF: tl.constexpr,
        WRAP_ROWS: tl.constexpr,
        WIDE_X: tl.constexpr,
        WIDE_C: tl.constexpr,
    ):
        """2D Split-K FP8 Tensor-Core Partial Nearest-Centroid Kernel (`0` dynamic allocations)."""
        pid_b = tl.program_id(0)
        part = tl.program_id(1)
        offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        mask_b = offs_b < B

        if HAS_BATCH_OFF:
            raw_off = tl.load(batch_off_ptr).to(tl.int64)
        else:
            raw_off = row_offset + tl.zeros((), dtype=tl.int64)

        if HAS_INDICES:
            pos64 = raw_off + offs_b.to(tl.int64)
            if WRAP_ROWS:
                n_idx64 = N_IDX + tl.zeros((), dtype=tl.int64)
                pos64 = ((pos64 % n_idx64) + n_idx64) % n_idx64
            src64 = tl.load(row_idx_ptr + pos64, mask=mask_b, other=0).to(tl.int64)
        else:
            src64 = raw_off + offs_b.to(tl.int64)
            if WRAP_ROWS:
                n64 = N + tl.zeros((), dtype=tl.int64)
                src64 = ((src64 % n64) + n64) % n64

        part_size = (K + PARTITIONS - 1) // PARTITIONS
        k_start = part * part_size
        k_end = tl.minimum(k_start + part_size, K)

        two_sx = 2.0 * tl.load(sx_ptr + src64, mask=mask_b, other=0.0)

        if FULL_D:
            offs_d = tl.arange(0, BLOCK_D)
            mask_d = offs_d < D_EFF
            if WIDE_X:
                x_off = src64[:, None] * stride_xb + offs_d[None, :].to(tl.int64)
            else:
                x_off = src64[:, None].to(tl.int32) * stride_xb + offs_d[None, :]
            qx_full = tl.load(qx_ptr + x_off, mask=mask_b[:, None] & mask_d[None, :], other=0.0)

        best_score = tl.full((BLOCK_B,), float("inf"), dtype=tl.float32)
        best_lbl = tl.zeros((BLOCK_B,), dtype=tl.int32)

        k0 = k_start
        while k0 < k_end:
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < k_end
            if FULL_D:
                offs_d = tl.arange(0, BLOCK_D)
                mask_d = offs_d < D_EFF
                if WIDE_C:
                    c_off = offs_k[:, None].to(tl.int64) * stride_ck + offs_d[None, :].to(tl.int64)
                else:
                    c_off = offs_k[:, None] * stride_ck + offs_d[None, :]
                qc_tile = tl.load(qc_ptr + c_off, mask=mask_k[:, None] & mask_d[None, :], other=0.0)
                dots = tl.dot(qx_full, tl.trans(qc_tile), max_num_imprecise_acc=0, out_dtype=tl.float32)
            else:
                dots = tl.zeros((BLOCK_B, BLOCK_K), dtype=tl.float32)
                for d0 in range(0, D_EFF, BLOCK_D):
                    offs_d = d0 + tl.arange(0, BLOCK_D)
                    mask_d = offs_d < D_EFF
                    if WIDE_X:
                        x_off = src64[:, None] * stride_xb + offs_d[None, :].to(tl.int64)
                    else:
                        x_off = src64[:, None].to(tl.int32) * stride_xb + offs_d[None, :]
                    if WIDE_C:
                        c_off = offs_k[:, None].to(tl.int64) * stride_ck + offs_d[None, :].to(tl.int64)
                    else:
                        c_off = offs_k[:, None] * stride_ck + offs_d[None, :]
                    qx_tile = tl.load(qx_ptr + x_off, mask=mask_b[:, None] & mask_d[None, :], other=0.0)
                    qc_tile = tl.load(qc_ptr + c_off, mask=mask_k[:, None] & mask_d[None, :], other=0.0)
                    dots = tl.dot(
                        qx_tile,
                        tl.trans(qc_tile),
                        acc=dots,
                        max_num_imprecise_acc=0,
                        out_dtype=tl.float32,
                    )

            sc = tl.load(sc_ptr + offs_k, mask=mask_k, other=0.0)
            cn = tl.load(cnorm_ptr + offs_k, mask=mask_k, other=float("inf"))
            scores = cn[None, :] - two_sx[:, None] * (dots * sc[None, :])
            scores = tl.where(mask_k[None, :], scores, float("inf"))
            b_min, b_idx = tl.min(
                scores,
                axis=1,
                return_indices=True,
                return_indices_tie_break_left=True,
            )
            imp = b_min < best_score
            best_score = tl.where(imp, b_min, best_score)
            best_lbl = tl.where(imp, k0 + b_idx.to(tl.int32), best_lbl)
            k0 += BLOCK_K

        p_off = offs_b.to(tl.int64) * stride_pb + part
        tl.store(part_score_ptr + p_off, best_score, mask=mask_b)
        tl.store(part_lbl_ptr + p_off, best_lbl, mask=mask_b)

    @triton.jit(do_not_specialize=["B", "N", "N_IDX", "stride_pb", "row_offset"])
    def _croiss_splitk_reduce_kernel(
        part_score_ptr,
        part_lbl_ptr,
        xnorm_ptr,
        batch_off_ptr,
        row_idx_ptr,
        inv_gs2_ptr,
        out_dist_ptr,
        out_lbl_ptr,
        B,
        N,
        N_IDX,
        stride_pb,
        row_offset,
        PARTITIONS: tl.constexpr,
        BLOCK_B: tl.constexpr,
        HAS_INDICES: tl.constexpr,
        HAS_BATCH_OFF: tl.constexpr,
        WRAP_ROWS: tl.constexpr,
        HAS_INV_GS2: tl.constexpr,
    ):
        """Fast reduction of Split-K partial winners using precomputed `x_norm`."""
        pid_b = tl.program_id(0)
        offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        mask_b = offs_b < B

        if HAS_BATCH_OFF:
            raw_off = tl.load(batch_off_ptr).to(tl.int64)
        else:
            raw_off = row_offset + tl.zeros((), dtype=tl.int64)

        if HAS_INDICES:
            pos64 = raw_off + offs_b.to(tl.int64)
            if WRAP_ROWS:
                n_idx64 = N_IDX + tl.zeros((), dtype=tl.int64)
                pos64 = ((pos64 % n_idx64) + n_idx64) % n_idx64
            src64 = tl.load(row_idx_ptr + pos64, mask=mask_b, other=0).to(tl.int64)
        else:
            src64 = raw_off + offs_b.to(tl.int64)
            if WRAP_ROWS:
                n64 = N + tl.zeros((), dtype=tl.int64)
                src64 = ((src64 % n64) + n64) % n64

        offs_p = tl.arange(0, PARTITIONS)
        p_off = offs_b[:, None].to(tl.int64) * stride_pb + offs_p[None, :]
        p_scores = tl.load(part_score_ptr + p_off, mask=mask_b[:, None], other=float("inf"))
        best_score, best_part = tl.min(
            p_scores,
            axis=1,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
        p_lbls = tl.load(part_lbl_ptr + p_off, mask=mask_b[:, None], other=0)
        best_lbl = tl.sum(tl.where(offs_p[None, :] == best_part[:, None], p_lbls, 0), axis=1)

        x_norm = tl.load(xnorm_ptr + src64, mask=mask_b, other=0.0)
        dist = tl.maximum(best_score + x_norm, 0.0)
        if HAS_INV_GS2:
            inv_gs2 = tl.load(inv_gs2_ptr).to(tl.float64)
            dist = tl.minimum(dist.to(tl.float64) * inv_gs2, _TL_FP32_CAP).to(tl.float32)
        tl.store(out_dist_ptr + offs_b.to(tl.int64), dist, mask=mask_b)
        tl.store(out_lbl_ptr + offs_b.to(tl.int64), best_lbl.to(tl.int64), mask=mask_b)

    if HAS_DOT_SCALED:

        @triton.jit(do_not_specialize=["B", "K", "N", "N_IDX", "stride_xb", "stride_uxb", "stride_ck", "stride_uck", "row_offset"])
        def _croiss_assign_mxfp8_kernel(
            qx_u8_ptr,
            ux_u8_ptr,
            xnorm_ptr,
            qc_u8_ptr,
            uc_u8_ptr,
            cnorm_ptr,
            batch_off_ptr,
            row_idx_ptr,
            inv_gs2_ptr,
            dist_ptr,
            lbl_ptr,
            B,
            K,
            N,
            N_IDX,
            stride_xb,
            stride_uxb,
            stride_ck,
            stride_uck,
            row_offset,
            D_PAD: tl.constexpr,
            BLOCK_B: tl.constexpr,
            BLOCK_K: tl.constexpr,
            BLOCK_D: tl.constexpr,
            HAS_INDICES: tl.constexpr,
            HAS_BATCH_OFF: tl.constexpr,
            WRAP_ROWS: tl.constexpr,
            HAS_INV_GS2: tl.constexpr,
        ):
            """Native Blackwell SM120 `tl.dot_scaled` MXFP8 (E4M3 + E8M0 block-32) Nearest-Centroid Kernel."""
            pid_b = tl.program_id(0)
            offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
            mask_b = offs_b < B

            if HAS_BATCH_OFF:
                raw_off = tl.load(batch_off_ptr).to(tl.int64)
            else:
                raw_off = row_offset + tl.zeros((), dtype=tl.int64)

            if HAS_INDICES:
                pos64 = raw_off + offs_b.to(tl.int64)
                if WRAP_ROWS:
                    n_idx64 = N_IDX + tl.zeros((), dtype=tl.int64)
                    pos64 = ((pos64 % n_idx64) + n_idx64) % n_idx64
                src64 = tl.load(row_idx_ptr + pos64, mask=mask_b, other=0).to(tl.int64)
            else:
                src64 = raw_off + offs_b.to(tl.int64)
                if WRAP_ROWS:
                    n64 = N + tl.zeros((), dtype=tl.int64)
                    src64 = ((src64 % n64) + n64) % n64

            x_norm = tl.load(xnorm_ptr + src64, mask=mask_b, other=0.0)
            best_score = tl.full((BLOCK_B,), float("inf"), dtype=tl.float32)
            best_lbl = tl.zeros((BLOCK_B,), dtype=tl.int32)
            BLOCK_DS: tl.constexpr = BLOCK_D // 32

            for k0 in range(0, K, BLOCK_K):
                offs_k = k0 + tl.arange(0, BLOCK_K)
                mask_k = offs_k < K
                dots = tl.zeros((BLOCK_B, BLOCK_K), dtype=tl.float32)
                for d0 in range(0, D_PAD, BLOCK_D):
                    offs_d = d0 + tl.arange(0, BLOCK_D)
                    offs_ds = (d0 // 32) + tl.arange(0, BLOCK_DS)
                    qx_t = tl.load(
                        qx_u8_ptr + src64[:, None] * stride_xb + offs_d[None, :].to(tl.int64),
                        mask=mask_b[:, None],
                        other=0,
                    )
                    ux_t = tl.load(
                        ux_u8_ptr + src64[:, None] * stride_uxb + offs_ds[None, :].to(tl.int64),
                        mask=mask_b[:, None],
                        other=127,
                    )
                    qc_t = tl.load(
                        qc_u8_ptr + offs_k[:, None].to(tl.int64) * stride_ck + offs_d[None, :].to(tl.int64),
                        mask=mask_k[:, None],
                        other=0,
                    )
                    uc_t = tl.load(
                        uc_u8_ptr + offs_k[:, None].to(tl.int64) * stride_uck + offs_ds[None, :].to(tl.int64),
                        mask=mask_k[:, None],
                        other=127,
                    )
                    dots = tl.dot_scaled(
                        qx_t,
                        ux_t,
                        "e4m3",
                        tl.trans(qc_t),
                        uc_t,
                        "e4m3",
                        acc=dots,
                        out_dtype=tl.float32,
                    )

                cn = tl.load(cnorm_ptr + offs_k, mask=mask_k, other=float("inf"))
                scores = tl.where(mask_k[None, :], cn[None, :] - 2.0 * dots, float("inf"))
                b_min, b_idx = tl.min(
                    scores,
                    axis=1,
                    return_indices=True,
                    return_indices_tie_break_left=True,
                )
                imp = b_min < best_score
                best_score = tl.where(imp, b_min, best_score)
                best_lbl = tl.where(imp, k0 + b_idx.to(tl.int32), best_lbl)

            dist = tl.maximum(best_score + x_norm, 0.0)
            if HAS_INV_GS2:
                inv_gs2 = tl.load(inv_gs2_ptr).to(tl.float64)
                dist = tl.minimum(dist.to(tl.float64) * inv_gs2, _TL_FP32_CAP).to(tl.float32)
            tl.store(dist_ptr + offs_b.to(tl.int64), dist, mask=mask_b)
            tl.store(lbl_ptr + offs_b.to(tl.int64), best_lbl.to(tl.int64), mask=mask_b)


# ==============================================================================
# Public Data Structures: `CroissFP8Dataset` and `CroissFP8Workspace`
# ==============================================================================


@dataclass
class CroissFP8Dataset:
    """Pre-encoded FP8 E4M3FN dataset with per-row scales, represented norms, and cyclic TMA wrap tail."""

    qx: torch.Tensor               # (N, padded_d) float8_e4m3fn public view
    sx: torch.Tensor               # (N,) float32 positive per-row scale
    x_norm: torch.Tensor           # (N,) float32 exact represented ||x_tilde - mu||^2 (in gscale frame)
    mu: torch.Tensor               # (D,) float32 coordinate origin
    mu_fp64: torch.Tensor          # (D,) float64 coordinate origin
    n: int
    d: int
    padded_d: int
    scale_mode: str = "decoupled"  # "decoupled" | "ozaki" | "mxfp8"
    center: bool = True
    coord_scale: float = 1.0
    gscale_ptr: torch.Tensor = field(repr=False, default=None)          # (1,) float64 GPU/CPU scalar
    inv_gscale_sq_ptr: torch.Tensor = field(repr=False, default=None)   # (1,) float64 GPU/CPU scalar
    ux_blocks: torch.Tensor | None = field(repr=False, default=None)    # (N, padded_d // 32) uint8 E8M0
    dom_cols: torch.Tensor | None = field(repr=False, default=None)     # (M_ozaki,) int32 dominant cols
    _qx_wrap: torch.Tensor = field(repr=False, default=None)            # (N + MAX_WRAP_PAD, padded_d)
    _sx_wrap: torch.Tensor = field(repr=False, default=None)            # (N + MAX_WRAP_PAD,)
    _x_norm_wrap: torch.Tensor = field(repr=False, default=None)        # (N + MAX_WRAP_PAD,)
    _ux_wrap: torch.Tensor | None = field(repr=False, default=None)     # (N + MAX_WRAP_PAD, padded_d // 32)
    _span_scratch: torch.Tensor | None = field(repr=False, default=None)
    _tma_cache: dict[tuple[int, int, bool], Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        dev = self.qx.device
        if self.gscale_ptr is None:
            self.gscale_ptr = torch.ones((1,), dtype=torch.float64, device=dev)
        if self.inv_gscale_sq_ptr is None:
            self.inv_gscale_sq_ptr = torch.ones((1,), dtype=torch.float64, device=dev)

    def get_qx_tma(self, block_b: int, block_d: int, wrap: bool) -> Any:
        key = (block_b, block_d, wrap)
        desc = self._tma_cache.get(key)
        buf = self._qx_wrap if (wrap and self._qx_wrap is not None) else self.qx
        if desc is None:
            desc = TensorDescriptor.from_tensor(buf, [block_b, block_d])
            self._tma_cache[key] = desc
        return desc


@dataclass
class CroissFP8Workspace:
    """Pre-allocated, 100% `torch.cuda.CUDAGraph`-safe workspace for centroid prep & FP8 assignment."""

    qc: torch.Tensor               # (K, padded_d) float8_e4m3fn
    sc: torch.Tensor               # (K,) float32 positive per-centroid scale
    c_norm: torch.Tensor           # (K,) float32 exact represented ||c_tilde - mu||^2
    dists: torch.Tensor            # (max_batch_size,) float32
    labels: torch.Tensor           # (max_batch_size,) int64
    part_scores: torch.Tensor      # (max_batch_size, MAX_SPLIT_K) float32 Split-K scratch
    part_labels: torch.Tensor      # (max_batch_size, MAX_SPLIT_K) int32 Split-K scratch
    k: int
    d: int
    padded_d: int
    max_batch_size: int
    scale_mode: str = "decoupled"
    uc_blocks: torch.Tensor | None = field(repr=False, default=None)    # (K, padded_d // 32) uint8 E8M0
    dom_cols_fallback: torch.Tensor | None = field(repr=False, default=None)
    unit_gscale_ptr: torch.Tensor = field(repr=False, default=None)     # (1,) float64 = 1.0
    _num_sms: int = 188
    _tma_cache: dict[tuple[int, int], Any] = field(default_factory=dict, repr=False)

    @classmethod
    def allocate(
        cls,
        k: int,
        d: int,
        *,
        max_batch_size: int,
        device: torch.device | str,
        scale_mode: str = "decoupled",
        padded_d: int | None = None,
    ) -> CroissFP8Workspace:
        if FLOAT8_E4M3FN is None:
            raise RuntimeError("torch.float8_e4m3fn is required for CroissFP8Workspace")
        if k <= 0 or d <= 0:
            raise ValueError(f"k and d must be positive; got k={k}, d={d}")
        dev = torch.device(device)
        pad_d = int(padded_d) if padded_d is not None else compute_croiss_padded_d(d, scale_mode=scale_mode)
        max_b = max(1, int(max_batch_size))
        qc = torch.empty((k, pad_d), dtype=FLOAT8_E4M3FN, device=dev)
        sc = torch.empty((k,), dtype=torch.float32, device=dev)
        c_norm = torch.empty((k,), dtype=torch.float32, device=dev)
        dists = torch.empty((max_b,), dtype=torch.float32, device=dev)
        labels = torch.empty((max_b,), dtype=torch.int64, device=dev)
        part_scores = torch.empty((max_b, MAX_SPLIT_K), dtype=torch.float32, device=dev)
        part_labels = torch.empty((max_b, MAX_SPLIT_K), dtype=torch.int32, device=dev)
        uc_blocks = (
            torch.empty((k, pad_d // MXFP8_BLOCK_SIZE), dtype=torch.uint8, device=dev)
            if scale_mode == "mxfp8"
            else None
        )
        dom_fb = (
            torch.arange(compute_croiss_ozaki_count(d), dtype=torch.int32, device=dev)
            if scale_mode == "ozaki"
            else None
        )
        unit_gs = torch.ones((1,), dtype=torch.float64, device=dev)
        num_sms = (
            int(torch.cuda.get_device_properties(dev).multi_processor_count)
            if dev.type == "cuda"
            else 1
        )
        return cls(
            qc=qc,
            sc=sc,
            c_norm=c_norm,
            dists=dists,
            labels=labels,
            part_scores=part_scores,
            part_labels=part_labels,
            k=int(k),
            d=int(d),
            padded_d=pad_d,
            max_batch_size=max_b,
            scale_mode=scale_mode,
            uc_blocks=uc_blocks,
            dom_cols_fallback=dom_fb,
            unit_gscale_ptr=unit_gs,
            _num_sms=num_sms,
        )

    def get_qc_tma(self, block_k: int, block_d: int) -> Any:
        key = (block_k, block_d)
        desc = self._tma_cache.get(key)
        if desc is None:
            desc = TensorDescriptor.from_tensor(self.qc, [block_k, block_d])
            self._tma_cache[key] = desc
        return desc

    def prepare_centroids_inplace(
        self,
        centroids: torch.Tensor,
        mu: torch.Tensor | CroissFP8Dataset | None = None,
        *,
        centroids_already_centered: bool = False,
        gscale_ptr: torch.Tensor | None = None,
        dom_cols: torch.Tensor | None = None,
    ) -> None:
        croiss_prepare_centroids_inplace(
            centroids,
            mu,
            self.qc,
            self.sc,
            self.c_norm,
            centroids_already_centered=centroids_already_centered,
            gscale_ptr=gscale_ptr,
            uc_blocks_out=self.uc_blocks,
            dom_cols=dom_cols,
            scale_mode=self.scale_mode,
            _unit_gscale_ptr=self.unit_gscale_ptr,
            _dom_cols_fallback=self.dom_cols_fallback,
        )

    def assign_inplace(
        self,
        dataset: CroissFP8Dataset,
        *,
        batch_size: int | None = None,
        row_offset: int = 0,
        batch_offset_ptr: torch.Tensor | None = None,
        batch_offset: torch.Tensor | None = None,
        wrap_rows: bool = True,
        row_indices: torch.Tensor | None = None,
        out_dists: torch.Tensor | None = None,
        out_labels: torch.Tensor | None = None,
        allow_tma: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return croiss_fp8_assign_inplace(
            dataset,
            self,
            batch_size=batch_size,
            row_offset=row_offset,
            batch_offset_ptr=batch_offset_ptr,
            batch_offset=batch_offset,
            wrap_rows=wrap_rows,
            row_indices=row_indices,
            out_dists=out_dists,
            out_labels=out_labels,
            allow_tma=allow_tma,
        )


# ==============================================================================
# Core Functions: Dataset Pre-Encoding, Centroid Prep, Assignment & Split Guard
# ==============================================================================


def _encode_rows_torch_fallback(
    x: torch.Tensor,
    mu_fp64: torch.Tensor | None,
    gscale_t: torch.Tensor,
    qx_out: torch.Tensor,
    sx_out: torch.Tensor,
    xnorm_out: torch.Tensor,
    ux_out: torch.Tensor | None,
    n_logical: int,
    n_phys: int,
    d: int,
    row_indices: torch.Tensor | None,
    scale_mode: str,
    dom_cols: torch.Tensor | None = None,
    is_centroid: bool = False,
) -> None:
    """Exact CPU / non-Triton reference encoder matching the Triton kernel semantics 1:1."""
    idx_all = torch.arange(n_phys, device=x.device, dtype=torch.int64) % max(n_logical, 1)
    if row_indices is not None:
        src = row_indices.to(torch.int64)[idx_all]
    else:
        src = idx_all
    rows64 = x[src].double()
    gs = float(gscale_t.double().item())
    if mu_fp64 is not None:
        diff64 = (rows64 - mu_fp64.double()[None, :]) * gs
    else:
        diff64 = rows64 * gs
    diff64 = torch.where(torch.isfinite(diff64), diff64, torch.zeros_like(diff64))
    val = diff64.clamp(-1.0e17, 1.0e17).float()
    pad_d = int(qx_out.shape[1])

    if scale_mode == "mxfp8":
        assert ux_out is not None
        padded_val = torch.zeros((n_phys, pad_d), dtype=torch.float32, device=x.device)
        padded_val[:, :d] = val
        blocks = padded_val.reshape(n_phys, pad_d // MXFP8_BLOCK_SIZE, MXFP8_BLOCK_SIZE)
        bamax = blocks.abs().amax(dim=-1)
        pos = bamax > 0.0
        exp = torch.where(
            pos,
            torch.ceil(torch.log2(torch.where(pos, bamax * (1.00000024 / FP8_E4M3_MAX), torch.ones_like(bamax)))),
            torch.zeros_like(bamax),
        ).clamp(-127.0, 127.0).to(torch.int32)
        ux_out.copy_((exp + 127).to(torch.uint8))
        sb = torch.ldexp(torch.ones_like(bamax), exp)
        inv_sb = torch.where(pos, 1.0 / sb, torch.zeros_like(sb))
        q = (blocks * inv_sb.unsqueeze(-1)).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
        q = torch.where(pos.unsqueeze(-1), q, torch.zeros_like(q)).to(FLOAT8_E4M3FN)
        qx_out.copy_(q.reshape(n_phys, pad_d))
        qf = q.float()
        xnorm_out.copy_((sb.square() * qf.square().sum(dim=-1)).sum(dim=-1))
        sx_out.copy_((bamax.amax(dim=-1) * (1.00000024 / FP8_E4M3_MAX)).clamp_min(FP32_SCALE_FLOOR))
        return

    amax = val.abs().amax(dim=1)
    sx = (amax * (1.00000024 / FP8_E4M3_MAX)).clamp_min(FP32_SCALE_FLOOR)
    inv_sx = torch.where(amax > 0.0, 1.0 / sx, torch.zeros_like(sx))
    scaled = (val * inv_sx[:, None]).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    scaled = torch.where((amax > 0.0)[:, None], scaled, torch.zeros_like(scaled))
    qx_out.zero_()
    q = scaled.to(FLOAT8_E4M3FN)
    qx_out[:, :d] = q
    qf_full = q.float().clone()

    if scale_mode == "ozaki" and dom_cols is not None and int(dom_cols.numel()) > 0:
        m_oz = int(dom_cols.numel())
        active = amax > 0.0
        for m_idx in range(m_oz):
            col = int(dom_cols[m_idx].item())
            qf_hi = q[:, col].float()
            res = (scaled[:, col] - qf_hi).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
            res = torch.where(active, res, torch.zeros_like(res))
            q_lo = res.to(FLOAT8_E4M3FN)
            aux = d + 3 * m_idx
            if is_centroid:
                qx_out[:, aux + 0] = q_lo
                qx_out[:, aux + 1] = q[:, col]
                qx_out[:, aux + 2] = q_lo
            else:
                qx_out[:, aux + 0] = q[:, col]
                qx_out[:, aux + 1] = q_lo
                qx_out[:, aux + 2] = q_lo
            qf_full[:, col] = qf_hi + q_lo.float()

    sx_out.copy_(sx)
    xnorm_out.copy_(sx.square() * qf_full.square().sum(dim=1))


@torch.no_grad()
def croiss_encode_dataset(
    x: torch.Tensor,
    *,
    mu: torch.Tensor | None = None,
    row_indices: torch.Tensor | None = None,
    scale_mode: str = "decoupled",
    center: bool = True,
    coord_scale: float | None = None,
    out: CroissFP8Dataset | None = None,
) -> CroissFP8Dataset:
    """Encode `(N, D)` dataset into `CroissFP8Dataset` in a single fused Triton pass (`0` host syncs)."""
    if FLOAT8_E4M3FN is None:
        raise RuntimeError("torch.float8_e4m3fn is not available in this PyTorch build")
    if x.ndim != 2:
        raise ValueError(f"Expected 2D input tensor x of shape (N, D); got {tuple(x.shape)}")
    if scale_mode not in ("decoupled", "ozaki", "mxfp8"):
        raise ValueError(f"scale_mode must be 'decoupled', 'ozaki', or 'mxfp8'; got {scale_mode!r}")

    n_src, d = int(x.shape[0]), int(x.shape[1])
    if d <= 0:
        raise ValueError(f"Feature dimension d must be positive; got {d}")
    n = int(row_indices.shape[0]) if row_indices is not None else n_src
    device = x.device
    padded_d = out.padded_d if out is not None else compute_croiss_padded_d(d, scale_mode=scale_mode)

    if n == 0:
        mu64 = (
            mu.to(device=device, dtype=torch.float64).contiguous()
            if mu is not None
            else torch.zeros((d,), dtype=torch.float64, device=device)
        )
        mu32 = mu64.float().contiguous()
        gs_ptr = torch.ones((1,), dtype=torch.float64, device=device)
        inv_gs2_ptr = torch.ones((1,), dtype=torch.float64, device=device)
        empty_qx = torch.empty((0, padded_d), dtype=FLOAT8_E4M3FN, device=device)
        empty_f = torch.empty((0,), dtype=torch.float32, device=device)
        empty_ux = (
            torch.empty((0, padded_d // MXFP8_BLOCK_SIZE), dtype=torch.uint8, device=device)
            if scale_mode == "mxfp8"
            else None
        )
        return CroissFP8Dataset(
            qx=empty_qx,
            sx=empty_f,
            x_norm=empty_f.clone(),
            mu=mu32,
            mu_fp64=mu64,
            n=0,
            d=d,
            padded_d=padded_d,
            scale_mode=scale_mode,
            center=center,
            coord_scale=float(coord_scale) if coord_scale is not None else 1.0,
            gscale_ptr=gs_ptr,
            inv_gscale_sq_ptr=inv_gs2_ptr,
            ux_blocks=empty_ux,
            _qx_wrap=empty_qx,
            _sx_wrap=empty_f,
            _x_norm_wrap=empty_f,
            _ux_wrap=empty_ux,
        )

    n_phys = n + MAX_WRAP_PAD
    if out is not None and out._qx_wrap is not None and out._qx_wrap.shape == (n_phys, padded_d):
        qx_wrap = out._qx_wrap
        sx_wrap = out._sx_wrap
        xnorm_wrap = out._x_norm_wrap
        ux_wrap = out._ux_wrap
        gs_ptr = out.gscale_ptr
        inv_gs2_ptr = out.inv_gscale_sq_ptr
        span_scratch = out._span_scratch
        out._tma_cache.clear()
    else:
        qx_wrap = torch.empty((n_phys, padded_d), dtype=FLOAT8_E4M3FN, device=device)
        sx_wrap = torch.empty((n_phys,), dtype=torch.float32, device=device)
        xnorm_wrap = torch.empty((n_phys,), dtype=torch.float32, device=device)
        ux_wrap = (
            torch.empty((n_phys, padded_d // MXFP8_BLOCK_SIZE), dtype=torch.uint8, device=device)
            if scale_mode == "mxfp8"
            else None
        )
        gs_ptr = torch.ones((1,), dtype=torch.float64, device=device)
        inv_gs2_ptr = torch.ones((1,), dtype=torch.float64, device=device)
        span_scratch = torch.empty((256,), dtype=torch.float64, device=device)

    # 1. Compute odd-sample translation-equivariant coordinate median (`0` host syncs!)
    # For <=32-bit float inputs, odd-sample median in FP32 is bit-identical to FP64 median and ~6x faster on high-D!
    target_m = min(n_src, 2049)
    if target_m > 1 and (target_m % 2 == 0):
        target_m -= 1  # Odd count guarantees exact order-statistic translation equivariance!
    step_sub = max(1, n_src // max(target_m, 1))
    need_sub = (scale_mode == "ozaki") or (center and mu is None)
    if need_sub:
        sub_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        sub_raw = x[::step_sub][:target_m].to(dtype=sub_dtype)
        sub_clean_f = torch.where(torch.isfinite(sub_raw), sub_raw, torch.zeros_like(sub_raw))
    else:
        sub_clean_f = None

    if not center:
        mu64 = torch.zeros((d,), dtype=torch.float64, device=device)
        mu32 = torch.zeros((d,), dtype=torch.float32, device=device)
    elif mu is not None:
        mu64 = mu.to(device=device, dtype=torch.float64).contiguous()
        mu32 = mu64.float().contiguous()
    else:
        assert sub_clean_f is not None
        med_vals = sub_clean_f.median(dim=0).values
        mu64 = med_vals.double().contiguous()
        mu32 = med_vals.float().contiguous()

    if scale_mode == "ozaki":
        m_oz = compute_croiss_ozaki_count(d)
        assert sub_clean_f is not None
        col_energy = (sub_clean_f.double() - mu64[None, :]).square().sum(dim=0)
        dom_cols: torch.Tensor | None = torch.topk(col_energy, k=m_oz, largest=True, sorted=True).indices.to(torch.int32).contiguous()
    else:
        dom_cols = None

    # 2. Compute sync-free on-device power-of-two conditioning scale (`1.0` for all `[1e-10, 1e+10]` inputs)
    force_cs = coord_scale is not None
    cs_val = float(coord_scale) if force_cs else 1.0

    if HAS_TRITON and x.is_cuda:
        with torch.cuda.device(device):
            if force_cs:
                _croiss_finalize_gscale_kernel[(1,)](
                    span_scratch,
                    gs_ptr,
                    inv_gs2_ptr,
                    cs_val,
                    1,
                    FORCE_COORD_SCALE=True,
                    BLOCK_C=16,
                    num_warps=1,
                )
            else:
                num_scan = min(n_src, 65536)
                step_scan = max(1, n_src // max(num_scan, 1))
                block_m_scan = 256
                num_ctas_scan = min(256, triton.cdiv(num_scan, block_m_scan))
                eff_scan = min(num_scan, num_ctas_scan * block_m_scan)
                block_d_scan = min(256, max(64, triton.next_power_of_2(min(d, 256))))
                _croiss_span_scan_kernel[(num_ctas_scan,)](
                    x,
                    mu64,
                    span_scratch,
                    n_src,
                    d,
                    x.stride(0),
                    x.stride(1),
                    step_scan,
                    eff_scan,
                    HAS_MU=center,
                    BLOCK_M=block_m_scan,
                    BLOCK_D=block_d_scan,
                    num_warps=4,
                )
                block_c = max(16, triton.next_power_of_2(num_ctas_scan))
                _croiss_finalize_gscale_kernel[(1,)](
                    span_scratch,
                    gs_ptr,
                    inv_gs2_ptr,
                    1.0,
                    num_ctas_scan,
                    FORCE_COORD_SCALE=False,
                    BLOCK_C=block_c,
                    num_warps=1,
                )

            has_idx = row_indices is not None
            idx_c = row_indices.contiguous() if (has_idx and not row_indices.is_contiguous()) else row_indices
            use_fp64_diff = x.dtype == torch.float64
            mu_kern = mu64 if use_fp64_diff else mu32

            if scale_mode == "mxfp8":
                assert ux_wrap is not None
                block_m = 16
                _croiss_encode_mxfp8_kernel[(triton.cdiv(n_phys, block_m),)](
                    x,
                    idx_c,
                    mu_kern,
                    gs_ptr,
                    qx_wrap.view(torch.uint8),
                    ux_wrap,
                    sx_wrap,
                    xnorm_wrap,
                    n,
                    n_phys,
                    d,
                    x.stride(0),
                    x.stride(1),
                    qx_wrap.stride(0),
                    ux_wrap.stride(0),
                    HAS_INDICES=has_idx,
                    HAS_MU=center,
                    USE_FP64_DIFF=use_fp64_diff,
                    BLOCK_M=block_m,
                    num_warps=4,
                )
            elif padded_d <= 1024:
                block_m = 16 if padded_d <= 256 else 4
                block_d = max(64, triton.next_power_of_2(padded_d))
                _croiss_encode_decoupled_full_d_kernel[(triton.cdiv(n_phys, block_m),)](
                    x,
                    idx_c,
                    mu_kern,
                    gs_ptr,
                    qx_wrap,
                    sx_wrap,
                    xnorm_wrap,
                    n,
                    n_phys,
                    d,
                    x.stride(0),
                    x.stride(1),
                    qx_wrap.stride(0),
                    HAS_INDICES=has_idx,
                    HAS_MU=center,
                    USE_FP64_DIFF=use_fp64_diff,
                    BLOCK_M=block_m,
                    BLOCK_D=block_d,
                    num_warps=4,
                )
            else:
                block_m = 8
                block_d = 128
                _croiss_encode_decoupled_tiled_d_kernel[(triton.cdiv(n_phys, block_m),)](
                    x,
                    idx_c,
                    mu_kern,
                    gs_ptr,
                    qx_wrap,
                    sx_wrap,
                    xnorm_wrap,
                    n,
                    n_phys,
                    d,
                    x.stride(0),
                    x.stride(1),
                    qx_wrap.stride(0),
                    HAS_INDICES=has_idx,
                    HAS_MU=center,
                    USE_FP64_DIFF=use_fp64_diff,
                    BLOCK_M=block_m,
                    BLOCK_D=block_d,
                    num_warps=4,
                )

            if scale_mode == "ozaki" and dom_cols is not None and int(dom_cols.numel()) > 0:
                m_oz = int(dom_cols.numel())
                _croiss_ozaki_append_kernel[(triton.cdiv(n_phys, 64),)](
                    x,
                    idx_c,
                    mu_kern,
                    gs_ptr,
                    dom_cols,
                    qx_wrap,
                    sx_wrap,
                    xnorm_wrap,
                    n,
                    n_phys,
                    d,
                    x.stride(0),
                    x.stride(1),
                    qx_wrap.stride(0),
                    m_oz,
                    IS_CENTROID=False,
                    HAS_INDICES=has_idx,
                    HAS_MU=center,
                    USE_FP64_DIFF=use_fp64_diff,
                    BLOCK_M=64,
                    num_warps=2,
                )
    else:
        if force_cs:
            gs_ptr.fill_(cs_val if cs_val > 0.0 else 1.0)
            inv_gs2_ptr.fill_(1.0)
        else:
            diff_all = (x.double() - mu64[None, :]) if center else x.double()
            diff_clean = torch.where(torch.isfinite(diff_all), diff_all.abs(), torch.zeros_like(diff_all))
            span = float(diff_clean.max().item()) if diff_clean.numel() > 0 else 0.0
            if span > 1.0e10 or (0.0 < span < 1.0e-10):
                import math

                exp_s = max(-480, min(480, int(round(math.log2(span)))))
                gs_ptr.fill_(math.ldexp(1.0, -exp_s))
                inv_gs2_ptr.fill_(math.ldexp(1.0, 2 * exp_s))
            else:
                gs_ptr.fill_(1.0)
                inv_gs2_ptr.fill_(1.0)

        _encode_rows_torch_fallback(
            x,
            mu64 if center else None,
            gs_ptr,
            qx_wrap,
            sx_wrap,
            xnorm_wrap,
            ux_wrap,
            n,
            n_phys,
            d,
            row_indices,
            scale_mode,
            dom_cols=dom_cols,
            is_centroid=False,
        )

    # Attach metadata onto `mu` and `mu_fp64` so callers passing `dataset.mu` to `croiss_prepare_centroids_inplace`
    # automatically inherit exact FP64 origin, conditioning scale, and Ozaki columns without extra arguments!
    mu32._croiss_mu64 = mu64  # type: ignore[attr-defined]
    mu32._croiss_gscale_ptr = gs_ptr  # type: ignore[attr-defined]
    mu32._croiss_coord_scale_forced = force_cs  # type: ignore[attr-defined]
    mu32._croiss_dom_cols = dom_cols  # type: ignore[attr-defined]
    mu32._croiss_scale_mode = scale_mode  # type: ignore[attr-defined]
    mu64._croiss_mu64 = mu64  # type: ignore[attr-defined]
    mu64._croiss_gscale_ptr = gs_ptr  # type: ignore[attr-defined]
    mu64._croiss_coord_scale_forced = force_cs  # type: ignore[attr-defined]
    mu64._croiss_dom_cols = dom_cols  # type: ignore[attr-defined]
    mu64._croiss_scale_mode = scale_mode  # type: ignore[attr-defined]

    qx_view = qx_wrap[:n]
    sx_view = sx_wrap[:n]
    xnorm_view = xnorm_wrap[:n]
    ux_view = ux_wrap[:n] if ux_wrap is not None else None

    if out is not None:
        out.qx = qx_view
        out.sx = sx_view
        out.x_norm = xnorm_view
        out.mu = mu32
        out.mu_fp64 = mu64
        out.n = n
        out.d = d
        out.padded_d = padded_d
        out.scale_mode = scale_mode
        out.center = center
        out.coord_scale = cs_val
        out.gscale_ptr = gs_ptr
        out.inv_gscale_sq_ptr = inv_gs2_ptr
        out.ux_blocks = ux_view
        out.dom_cols = dom_cols
        out._qx_wrap = qx_wrap
        out._sx_wrap = sx_wrap
        out._x_norm_wrap = xnorm_wrap
        out._ux_wrap = ux_wrap
        out._span_scratch = span_scratch
        return out

    return CroissFP8Dataset(
        qx=qx_view,
        sx=sx_view,
        x_norm=xnorm_view,
        mu=mu32,
        mu_fp64=mu64,
        n=n,
        d=d,
        padded_d=padded_d,
        scale_mode=scale_mode,
        center=center,
        coord_scale=cs_val,
        gscale_ptr=gs_ptr,
        inv_gscale_sq_ptr=inv_gs2_ptr,
        ux_blocks=ux_view,
        dom_cols=dom_cols,
        _qx_wrap=qx_wrap,
        _sx_wrap=sx_wrap,
        _x_norm_wrap=xnorm_wrap,
        _ux_wrap=ux_wrap,
        _span_scratch=span_scratch,
    )


@torch.no_grad()
def croiss_prepare_centroids_inplace(
    centroids: torch.Tensor,
    mu: torch.Tensor | CroissFP8Dataset | None,
    qc_out: torch.Tensor,
    sc_out: torch.Tensor,
    c_norm_out: torch.Tensor,
    *,
    centroids_already_centered: bool = False,
    gscale_ptr: torch.Tensor | None = None,
    uc_blocks_out: torch.Tensor | None = None,
    dom_cols: torch.Tensor | None = None,
    scale_mode: str = "decoupled",
    _unit_gscale_ptr: torch.Tensor | None = None,
    _dom_cols_fallback: torch.Tensor | None = None,
) -> None:
    """In-place, zero-sync, `CUDAGraph`-capturable per-centroid FP8 preparation."""
    k, d = int(centroids.shape[0]), int(centroids.shape[1])
    if k == 0:
        return
    device = centroids.device
    padded_d = int(qc_out.shape[1])

    ds_obj = mu if isinstance(mu, CroissFP8Dataset) else None
    if ds_obj is not None:
        mu_tensor: torch.Tensor | None = None if (centroids_already_centered or not ds_obj.center) else ds_obj.mu_fp64
        if gscale_ptr is None:
            if centroids_already_centered and (ds_obj.coord_scale != 1.0):
                gscale_ptr = _unit_gscale_ptr
            else:
                gscale_ptr = ds_obj.gscale_ptr
        if dom_cols is None:
            dom_cols = ds_obj.dom_cols
        if scale_mode == "decoupled" and ds_obj.scale_mode in ("mxfp8", "ozaki"):
            if ds_obj.scale_mode != "mxfp8" or uc_blocks_out is not None:
                scale_mode = ds_obj.scale_mode
    else:
        mu_tensor = None if centroids_already_centered else mu
        if gscale_ptr is None and mu is not None:
            cs_forced = bool(getattr(mu, "_croiss_coord_scale_forced", False))
            if not (centroids_already_centered and cs_forced):
                gscale_ptr = getattr(mu, "_croiss_gscale_ptr", None)
        if dom_cols is None and mu is not None:
            dom_cols = getattr(mu, "_croiss_dom_cols", None)
        if scale_mode == "decoupled" and mu is not None:
            inh_mode = getattr(mu, "_croiss_scale_mode", "decoupled")
            if inh_mode == "ozaki" or (inh_mode == "mxfp8" and uc_blocks_out is not None):
                scale_mode = inh_mode

    if scale_mode == "ozaki" and dom_cols is None:
        dom_cols = (
            _dom_cols_fallback
            if _dom_cols_fallback is not None
            else torch.arange(compute_croiss_ozaki_count(d), dtype=torch.int32, device=device)
        )

    if gscale_ptr is None:
        gscale_ptr = (
            _unit_gscale_ptr
            if _unit_gscale_ptr is not None
            else torch.ones((1,), dtype=torch.float64, device=device)
        )

    has_mu = (mu_tensor is not None) and (not centroids_already_centered)
    use_fp64_diff = centroids.dtype == torch.float64
    if has_mu and mu_tensor is not None:
        if use_fp64_diff:
            mu_kern = getattr(mu_tensor, "_croiss_mu64", None)
            if mu_kern is None:
                mu_kern = mu_tensor if mu_tensor.dtype == torch.float64 else mu_tensor.double()
        else:
            mu_kern = mu_tensor if mu_tensor.dtype == torch.float32 else mu_tensor.float()
    else:
        mu_kern = None

    if HAS_TRITON and centroids.is_cuda:
        with torch.cuda.device(device):
            if scale_mode == "mxfp8":
                if uc_blocks_out is None:
                    raise ValueError("uc_blocks_out is required when scale_mode='mxfp8'")
                block_m = 16
                _croiss_encode_mxfp8_kernel[(triton.cdiv(k, block_m),)](
                    centroids,
                    None,
                    mu_kern,
                    gscale_ptr,
                    qc_out.view(torch.uint8),
                    uc_blocks_out,
                    sc_out,
                    c_norm_out,
                    k,
                    k,
                    d,
                    centroids.stride(0),
                    centroids.stride(1),
                    qc_out.stride(0),
                    uc_blocks_out.stride(0),
                    HAS_INDICES=False,
                    HAS_MU=has_mu,
                    USE_FP64_DIFF=use_fp64_diff,
                    BLOCK_M=block_m,
                    num_warps=4,
                )
            elif padded_d <= 1024:
                block_m = 16 if padded_d <= 256 else 4
                block_d = max(64, triton.next_power_of_2(padded_d))
                _croiss_encode_decoupled_full_d_kernel[(triton.cdiv(k, block_m),)](
                    centroids,
                    None,
                    mu_kern,
                    gscale_ptr,
                    qc_out,
                    sc_out,
                    c_norm_out,
                    k,
                    k,
                    d,
                    centroids.stride(0),
                    centroids.stride(1),
                    qc_out.stride(0),
                    HAS_INDICES=False,
                    HAS_MU=has_mu,
                    USE_FP64_DIFF=use_fp64_diff,
                    BLOCK_M=block_m,
                    BLOCK_D=block_d,
                    num_warps=4,
                )
            else:
                block_m = 8
                block_d = 128
                _croiss_encode_decoupled_tiled_d_kernel[(triton.cdiv(k, block_m),)](
                    centroids,
                    None,
                    mu_kern,
                    gscale_ptr,
                    qc_out,
                    sc_out,
                    c_norm_out,
                    k,
                    k,
                    d,
                    centroids.stride(0),
                    centroids.stride(1),
                    qc_out.stride(0),
                    HAS_INDICES=False,
                    HAS_MU=has_mu,
                    USE_FP64_DIFF=use_fp64_diff,
                    BLOCK_M=block_m,
                    BLOCK_D=block_d,
                    num_warps=4,
                )

            if scale_mode == "ozaki" and dom_cols is not None and int(dom_cols.numel()) > 0:
                m_oz = int(dom_cols.numel())
                _croiss_ozaki_append_kernel[(triton.cdiv(k, 64),)](
                    centroids,
                    None,
                    mu_kern,
                    gscale_ptr,
                    dom_cols,
                    qc_out,
                    sc_out,
                    c_norm_out,
                    k,
                    k,
                    d,
                    centroids.stride(0),
                    centroids.stride(1),
                    qc_out.stride(0),
                    m_oz,
                    IS_CENTROID=True,
                    HAS_INDICES=False,
                    HAS_MU=has_mu,
                    USE_FP64_DIFF=use_fp64_diff,
                    BLOCK_M=64,
                    num_warps=2,
                )
        return

    mu64_fb = None
    if has_mu and mu_tensor is not None:
        mu64_fb = getattr(mu_tensor, "_croiss_mu64", None)
        if mu64_fb is None:
            mu64_fb = mu_tensor.double()
    _encode_rows_torch_fallback(
        centroids,
        mu64_fb,
        gscale_ptr,
        qc_out,
        sc_out,
        c_norm_out,
        uc_blocks_out,
        k,
        k,
        d,
        None,
        scale_mode,
        dom_cols=dom_cols,
        is_centroid=True,
    )


@torch.no_grad()
def croiss_fp8_assign_inplace(
    dataset: CroissFP8Dataset,
    workspace: CroissFP8Workspace,
    *,
    batch_size: int | None = None,
    row_offset: int = 0,
    batch_offset_ptr: torch.Tensor | None = None,
    batch_offset: torch.Tensor | None = None,
    wrap_rows: bool = True,
    row_indices: torch.Tensor | None = None,
    out_dists: torch.Tensor | None = None,
    out_labels: torch.Tensor | None = None,
    allow_tma: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """100% `torch.cuda.CUDAGraph`-safe nearest-centroid assignment (`0` syncs, `0` dynamic allocations)."""
    if batch_offset_ptr is None and batch_offset is not None:
        batch_offset_ptr = batch_offset

    if batch_size is not None:
        b = int(batch_size)
    elif row_indices is not None:
        b = int(row_indices.shape[0])
    elif out_dists is not None:
        b = int(out_dists.shape[0])
    else:
        b = int(dataset.n)

    dists = out_dists if out_dists is not None else workspace.dists[:b]
    labels = out_labels if out_labels is not None else workspace.labels[:b]

    if b == 0:
        return dists[:0], labels[:0]
    if dataset.n <= 0 or workspace.k <= 0:
        raise ValueError(f"Non-empty assignment requires dataset.n > 0 and workspace.k > 0; got n={dataset.n}, k={workspace.k}")
    if b > int(dists.shape[0]) or b > int(labels.shape[0]):
        raise ValueError(
            f"batch_size={b} exceeds pre-allocated output capacity ({dists.shape[0]}, {labels.shape[0]})"
        )

    k = int(workspace.k)
    pad_d = int(dataset.padded_d)
    device = dataset.qx.device
    has_idx = row_indices is not None
    has_boff = batch_offset_ptr is not None
    do_wrap = bool(wrap_rows) and (has_boff or (row_offset + b > dataset.n))
    n_idx = int(row_indices.shape[0]) if row_indices is not None else 0

    if not (HAS_TRITON and dataset.qx.is_cuda):
        # Exact reference fallback for CPU / non-Triton environments
        base = int(batch_offset_ptr.item()) if has_boff and batch_offset_ptr is not None else int(row_offset)
        pos = torch.arange(b, device=device, dtype=torch.int64) + base
        if has_idx and row_indices is not None:
            if do_wrap:
                pos = pos % n_idx
            src = row_indices.to(torch.int64)[pos]
        else:
            if do_wrap:
                pos = pos % dataset.n
            src = pos

        if dataset.scale_mode == "mxfp8":
            assert dataset.ux_blocks is not None and workspace.uc_blocks is not None
            ux = torch.ldexp(torch.ones_like(dataset.ux_blocks[src], dtype=torch.float32), dataset.ux_blocks[src].to(torch.int32) - 127)
            uc = torch.ldexp(torch.ones_like(workspace.uc_blocks, dtype=torch.float32), workspace.uc_blocks.to(torch.int32) - 127)
            xr = dataset.qx[src].float().reshape(b, pad_d // 32, 32) * ux.unsqueeze(-1)
            cr = workspace.qc.float().reshape(k, pad_d // 32, 32) * uc.unsqueeze(-1)
            dots = xr.reshape(b, pad_d) @ cr.reshape(k, pad_d).t()
            scores = workspace.c_norm[None, :] - 2.0 * dots
        else:
            dots = dataset.qx[src].float() @ workspace.qc.float().t()
            scores = workspace.c_norm[None, :] - (2.0 * dataset.sx[src])[:, None] * (dots * workspace.sc[None, :])

        best_lbl = scores.argmin(dim=1)
        best_sc = scores.gather(1, best_lbl[:, None]).squeeze(1)
        d_val = (best_sc + dataset.x_norm[src]).clamp_min(0.0)
        inv_gs2 = float(dataset.inv_gscale_sq_ptr.item()) if dataset.inv_gscale_sq_ptr is not None else 1.0
        if inv_gs2 != 1.0:
            d_val = (d_val.double() * inv_gs2).clamp_max(FP32_FINITE_CAP).float()
        dists[:b].copy_(d_val)
        labels[:b].copy_(best_lbl)
        return dists[:b], labels[:b]

    with torch.cuda.device(device):
        has_inv_gs2 = dataset.inv_gscale_sq_ptr is not None
        inv_gs2_arg = dataset.inv_gscale_sq_ptr if has_inv_gs2 else workspace.unit_gscale_ptr

        # 1. Native Blackwell SM120 MXFP8 block-scaled path
        if dataset.scale_mode == "mxfp8":
            if not _is_mxfp8_device_supported(device):
                raise RuntimeError("scale_mode='mxfp8' requires Blackwell SM100+ with Triton tl.dot_scaled")
            assert dataset.ux_blocks is not None and workspace.uc_blocks is not None
            idx_c = row_indices.contiguous() if (has_idx and row_indices is not None and not row_indices.is_contiguous()) else row_indices
            block_b = 128
            block_k = 128
            block_d = min(128, pad_d)
            _croiss_assign_mxfp8_kernel[(triton.cdiv(b, block_b),)](
                dataset.qx.view(torch.uint8),
                dataset.ux_blocks,
                dataset.x_norm,
                workspace.qc.view(torch.uint8),
                workspace.uc_blocks,
                workspace.c_norm,
                batch_offset_ptr,
                idx_c,
                inv_gs2_arg,
                dists,
                labels,
                b,
                k,
                dataset.n,
                n_idx,
                dataset.qx.stride(0),
                dataset.ux_blocks.stride(0),
                workspace.qc.stride(0),
                workspace.uc_blocks.stride(0),
                int(row_offset),
                D_PAD=pad_d,
                BLOCK_B=block_b,
                BLOCK_K=block_k,
                BLOCK_D=block_d,
                HAS_INDICES=has_idx,
                HAS_BATCH_OFF=has_boff,
                WRAP_ROWS=do_wrap,
                HAS_INV_GS2=has_inv_gs2,
                num_warps=8,
                num_stages=2,
            )
            return dists[:b], labels[:b]

        # 2. Check SM120 TMA eligibility up front (shared by both Split-K TMA and Full-Batch TMA)
        full_d = pad_d <= 256
        qx_buf = dataset._qx_wrap if (do_wrap and dataset._qx_wrap is not None) else dataset.qx
        sx_buf = dataset._sx_wrap if (do_wrap and dataset._sx_wrap is not None) else dataset.sx
        xn_buf = dataset._x_norm_wrap if (do_wrap and dataset._x_norm_wrap is not None) else dataset.x_norm
        can_use_tma = (
            allow_tma
            and (not has_idx)
            and (not do_wrap or dataset._qx_wrap is not None)
            and _tma_ready_2d(qx_buf)
            and _tma_ready_2d(workspace.qc)
        )

        # 3. Small-batch 2D Split-K dispatch (saturates all 188 Blackwell SMs when B <= 4096!)
        block_b_sk = 64
        parts = (
            _choose_croiss_split_k(b, k, block_b_sk, workspace._num_sms)
            if b <= int(workspace.part_scores.shape[0])
            else 1
        )
        if parts > 1:
            idx_c = row_indices.contiguous() if (has_idx and row_indices is not None and not row_indices.is_contiguous()) else row_indices
            if can_use_tma and full_d and pad_d in (64, 128, 256):
                block_k_tma = 256 if pad_d <= 128 else 128
                qx_desc_sk = dataset.get_qx_tma(block_b_sk, pad_d, do_wrap)
                qc_desc_sk = workspace.get_qc_tma(block_k_tma, pad_d)
                _croiss_splitk_tma_full_d_kernel[(triton.cdiv(b, block_b_sk), parts)](
                    qx_desc_sk,
                    sx_buf,
                    qc_desc_sk,
                    workspace.sc,
                    workspace.c_norm,
                    batch_offset_ptr,
                    workspace.part_scores,
                    workspace.part_labels,
                    b,
                    k,
                    dataset.n,
                    workspace.part_scores.stride(0),
                    int(row_offset),
                    PARTITIONS=parts,
                    BLOCK_B=block_b_sk,
                    BLOCK_K=block_k_tma,
                    BLOCK_D=pad_d,
                    HAS_BATCH_OFF=has_boff,
                    WRAP_ROWS=do_wrap,
                    num_warps=8,
                    num_stages=3 if pad_d <= 128 else 2,
                )
            else:
                block_k_sk = 128
                block_d_sk = max(64, triton.next_power_of_2(pad_d)) if full_d else 128
                wide_x = (dataset.n * int(dataset.qx.stride(0))) > _INT32_MAX
                wide_c = (k * int(workspace.qc.stride(0))) > _INT32_MAX
                grid_sk = (triton.cdiv(b, block_b_sk), parts)
                _croiss_splitk_assign_kernel[grid_sk](
                    dataset.qx,
                    dataset.sx,
                    workspace.qc,
                    workspace.sc,
                    workspace.c_norm,
                    batch_offset_ptr,
                    idx_c,
                    workspace.part_scores,
                    workspace.part_labels,
                    b,
                    k,
                    pad_d,
                    dataset.n,
                    n_idx,
                    dataset.qx.stride(0),
                    workspace.qc.stride(0),
                    workspace.part_scores.stride(0),
                    int(row_offset),
                    PARTITIONS=parts,
                    BLOCK_B=block_b_sk,
                    BLOCK_K=block_k_sk,
                    BLOCK_D=block_d_sk,
                    FULL_D=full_d,
                    HAS_INDICES=has_idx,
                    HAS_BATCH_OFF=has_boff,
                    WRAP_ROWS=do_wrap,
                    WIDE_X=wide_x,
                    WIDE_C=wide_c,
                    num_warps=4,
                    num_stages=2,
                )
            _croiss_splitk_reduce_kernel[(triton.cdiv(b, 256),)](
                workspace.part_scores,
                workspace.part_labels,
                dataset.x_norm,
                batch_offset_ptr,
                idx_c,
                inv_gs2_arg,
                dists,
                labels,
                b,
                dataset.n,
                n_idx,
                workspace.part_scores.stride(0),
                int(row_offset),
                PARTITIONS=parts,
                BLOCK_B=256,
                HAS_INDICES=has_idx,
                HAS_BATCH_OFF=has_boff,
                WRAP_ROWS=do_wrap,
                HAS_INV_GS2=has_inv_gs2,
                num_warps=4,
            )
            return dists[:b], labels[:b]

        # 4. SM120 3-Stage TMA Hardware Path (for contiguous / cyclic-wrapped batches up to 2^31 rows)
        if can_use_tma:
            if full_d and pad_d in (64, 128, 256):
                block_b = 128
                block_k = 256 if pad_d <= 128 else 128
                num_stages = 3 if pad_d <= 128 else 2
                qx_desc = dataset.get_qx_tma(block_b, pad_d, do_wrap)
                qc_desc = workspace.get_qc_tma(block_k, pad_d)
                _croiss_assign_tma_full_d_kernel[(triton.cdiv(b, block_b),)](
                    qx_desc,
                    sx_buf,
                    xn_buf,
                    qc_desc,
                    workspace.sc,
                    workspace.c_norm,
                    batch_offset_ptr,
                    inv_gs2_arg,
                    dists,
                    labels,
                    b,
                    k,
                    dataset.n,
                    int(row_offset),
                    BLOCK_B=block_b,
                    BLOCK_K=block_k,
                    BLOCK_D=pad_d,
                    HAS_BATCH_OFF=has_boff,
                    WRAP_ROWS=do_wrap,
                    HAS_INV_GS2=has_inv_gs2,
                    num_warps=8,
                    num_stages=num_stages,
                )
                return dists[:b], labels[:b]

            if (not full_d) and (pad_d % 128 == 0):
                block_b = 128
                block_k = 256
                block_d = 128
                qx_desc = dataset.get_qx_tma(block_b, block_d, do_wrap)
                qc_desc = workspace.get_qc_tma(block_k, block_d)
                _croiss_assign_tma_tiled_d_kernel[(triton.cdiv(b, block_b),)](
                    qx_desc,
                    sx_buf,
                    xn_buf,
                    qc_desc,
                    workspace.sc,
                    workspace.c_norm,
                    batch_offset_ptr,
                    inv_gs2_arg,
                    dists,
                    labels,
                    b,
                    k,
                    dataset.n,
                    int(row_offset),
                    D_PAD=pad_d,
                    BLOCK_B=block_b,
                    BLOCK_K=block_k,
                    BLOCK_D=block_d,
                    HAS_BATCH_OFF=has_boff,
                    WRAP_ROWS=do_wrap,
                    HAS_INV_GS2=has_inv_gs2,
                    num_warps=8,
                    num_stages=2,
                )
                return dists[:b], labels[:b]

        # 4. General Pointer-Gather / Arbitrary-Padded-D Path
        idx_c = row_indices.contiguous() if (has_idx and row_indices is not None and not row_indices.is_contiguous()) else row_indices
        block_b = 128
        block_k = 128
        block_d = max(64, triton.next_power_of_2(pad_d)) if full_d else 128
        wide_x = (dataset.n * int(dataset.qx.stride(0))) > _INT32_MAX
        wide_c = (k * int(workspace.qc.stride(0))) > _INT32_MAX
        _croiss_assign_ptr_kernel[(triton.cdiv(b, block_b),)](
            dataset.qx,
            dataset.sx,
            dataset.x_norm,
            workspace.qc,
            workspace.sc,
            workspace.c_norm,
            batch_offset_ptr,
            idx_c,
            inv_gs2_arg,
            dists,
            labels,
            b,
            k,
            pad_d,
            dataset.n,
            n_idx,
            dataset.qx.stride(0),
            workspace.qc.stride(0),
            int(row_offset),
            BLOCK_B=block_b,
            BLOCK_K=block_k,
            BLOCK_D=block_d,
            FULL_D=full_d,
            HAS_INDICES=has_idx,
            HAS_BATCH_OFF=has_boff,
            WRAP_ROWS=do_wrap,
            WIDE_X=wide_x,
            WIDE_C=wide_c,
            HAS_INV_GS2=has_inv_gs2,
            num_warps=4 if (full_d and pad_d <= 128) else 8,
            num_stages=3 if pad_d <= 128 else 2,
        )
        return dists[:b], labels[:b]


@torch.no_grad()
def croiss_fp8_assign(
    x_or_dataset: torch.Tensor | CroissFP8Dataset,
    centroids: torch.Tensor,
    *,
    workspace: CroissFP8Workspace | None = None,
    row_indices: torch.Tensor | None = None,
    batch_offset_ptr: torch.Tensor | None = None,
    batch_offset: torch.Tensor | None = None,
    batch_size: int | None = None,
    row_offset: int = 0,
    wrap_rows: bool = True,
    centroids_already_centered: bool = False,
    scale_mode: str = "decoupled",
    allow_tma: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """High-level functional CroissFP8 nearest-centroid assignment."""
    if isinstance(x_or_dataset, CroissFP8Dataset):
        dataset = x_or_dataset
    else:
        dataset = croiss_encode_dataset(x_or_dataset, scale_mode=scale_mode)

    k, d = int(centroids.shape[0]), int(centroids.shape[1])
    if batch_size is not None:
        eff_b = int(batch_size)
    elif row_indices is not None:
        eff_b = int(row_indices.shape[0])
    else:
        eff_b = int(dataset.n)

    if eff_b == 0:
        return (
            torch.empty((0,), dtype=torch.float32, device=centroids.device),
            torch.empty((0,), dtype=torch.int64, device=centroids.device),
        )

    ws = workspace
    if (
        ws is None
        or ws.k != k
        or ws.padded_d != dataset.padded_d
        or ws.max_batch_size < eff_b
        or ws.scale_mode != dataset.scale_mode
    ):
        ws = CroissFP8Workspace.allocate(
            k=k,
            d=d,
            max_batch_size=max(eff_b, 1),
            device=centroids.device,
            scale_mode=dataset.scale_mode,
            padded_d=dataset.padded_d,
        )

    ws.prepare_centroids_inplace(
        centroids,
        dataset,
        centroids_already_centered=centroids_already_centered,
    )
    return ws.assign_inplace(
        dataset,
        batch_size=eff_b,
        row_offset=row_offset,
        batch_offset_ptr=batch_offset_ptr,
        batch_offset=batch_offset,
        wrap_rows=wrap_rows,
        row_indices=row_indices,
        allow_tma=allow_tma,
    )


@torch.no_grad()
def croiss_separate_collapsed_splits_inplace(
    donor: torch.Tensor,
    child: torch.Tensor,
    mu: torch.Tensor | CroissFP8Dataset | None = None,
    *,
    centroids_already_centered: bool = False,
) -> None:
    """Branchless, zero-sync, `CUDAGraph`-safe decoupled-FP8 split-collapse guard.

    Ensures that for every donor/child pair `(m, D)`, their quantized decoupled-FP8
    representations `(sc_m, qc_m)` cannot be identical (even under `+20,000` shifts,
    all-zeros, or tiny perturbations below E4M3's 3-bit mantissa step `0.125 * sc_m`).
    """
    m, d = int(donor.shape[0]), int(donor.shape[1])
    if m == 0 or d == 0:
        return

    ds_obj = mu if isinstance(mu, CroissFP8Dataset) else None
    if centroids_already_centered or mu is None:
        mu_t: torch.Tensor | None = None
    elif ds_obj is not None:
        mu_t = ds_obj.mu if ds_obj.center else None
    else:
        mu_t = mu.float() if mu.dtype != torch.float32 else mu

    pad_d = compute_croiss_padded_d(d, scale_mode="decoupled")
    dq = torch.empty((m, pad_d), dtype=FLOAT8_E4M3FN, device=donor.device)
    cq = torch.empty((m, pad_d), dtype=FLOAT8_E4M3FN, device=child.device)
    ds = torch.empty((m,), dtype=torch.float32, device=donor.device)
    cs = torch.empty((m,), dtype=torch.float32, device=child.device)
    dn = torch.empty((m,), dtype=torch.float32, device=donor.device)
    cn = torch.empty((m,), dtype=torch.float32, device=child.device)

    croiss_prepare_centroids_inplace(donor, mu, dq, ds, dn, centroids_already_centered=centroids_already_centered)
    croiss_prepare_centroids_inplace(child, mu, cq, cs, cn, centroids_already_centered=centroids_already_centered)

    # Decoupled representations `sc * qc` collide iff both `qc` codes match AND scales agree within E4M3 step
    rep_d = ds[:, None] * dq[:, :d].float()
    rep_c = cs[:, None] * cq[:, :d].float()
    collapsed = (rep_d == rep_c).all(dim=1)

    # Step along coordinate 0 by 1 full E4M3 ULP (`max(0.25 * cs, 2^-8 * max(cs, 1e-6))`)
    base0 = child[:, 0].float()
    cen0 = (base0 - mu_t[0]) if mu_t is not None else base0
    ulp_step = torch.maximum(cs * 32.0, torch.full_like(cs, 1e-6))  # 32.0 in [-448,448] = 1 ULP at max octave!
    # Convert `ulp_step` back from gscale frame if dataset scaled coordinates
    gs_ptr = ds_obj.gscale_ptr if ds_obj is not None else getattr(mu, "_croiss_gscale_ptr", None)
    if gs_ptr is not None and not (centroids_already_centered and ds_obj is not None and ds_obj.coord_scale != 1.0):
        gs_f = gs_ptr.to(dtype=torch.float32).clamp_min(FP32_SCALE_FLOOR)
        phys_step = ulp_step / gs_f
    else:
        phys_step = ulp_step

    sign_dir = torch.where(cen0 >= 0.0, -1.0, 1.0)
    cand = base0 + sign_dir * phys_step
    fallback = torch.nextafter(
        base0,
        torch.where(sign_dir > 0.0, torch.full_like(base0, float("inf")), torch.full_like(base0, float("-inf"))),
    )
    repaired = torch.where(cand != base0, cand, fallback).to(dtype=child.dtype)
    child[:, 0] = torch.where(collapsed & torch.isfinite(repaired), repaired, child[:, 0])


@torch.no_grad()
def warmup_croiss_fp8_kernels(
    device: torch.device | str = "cuda",
    shapes: list[tuple[int, int, int]] | None = None,
    *,
    include_mxfp8: bool = False,
    include_ozaki: bool = False,
) -> None:
    """Pre-compile CroissFP8 Triton specializations outside benchmark timers (`#111` compliance)."""
    dev = torch.device(device)
    if not (CROISS_FP8_AVAILABLE and dev.type == "cuda" and torch.cuda.is_available()):
        return

    explicit_shapes = shapes is not None
    if shapes is None:
        shapes = [
            (4096, 2048, 128),   # Split-K + pad_d=128
            (49152, 2048, 128),  # SM120 3-Stage TMA (49152/128 = 384 >= 2*188 SMs!)
        ]

    boff = torch.zeros((), dtype=torch.int64, device=dev)
    seen_pads: set[int] = set()
    for b_sz, k_sz, d_sz in shapes:
        pad_d = compute_croiss_padded_d(d_sz, scale_mode="decoupled")
        x_w = torch.randn((b_sz, d_sz), dtype=torch.float32, device=dev)
        c_w = torch.randn((k_sz, d_sz), dtype=torch.float32, device=dev)

        ds = croiss_encode_dataset(x_w, scale_mode="decoupled")
        ws = CroissFP8Workspace.allocate(k=k_sz, d=d_sz, max_batch_size=max(b_sz, 49152), device=dev)
        ws.prepare_centroids_inplace(c_w, ds, centroids_already_centered=False)
        ws.prepare_centroids_inplace(c_w, ds, centroids_already_centered=True)

        # Warm the exact path `b_sz` hits + TMA wrapped batch_offset
        ws.assign_inplace(ds, batch_size=b_sz, wrap_rows=False, allow_tma=True)
        ws.assign_inplace(ds, batch_size=b_sz, batch_offset_ptr=boff, wrap_rows=True, allow_tma=True)

        if explicit_shapes and pad_d not in seen_pads:
            seen_pads.add(pad_d)
            idx_w = torch.arange(min(b_sz, 4096), dtype=torch.int64, device=dev)
            _ = croiss_encode_dataset(x_w, row_indices=idx_w, scale_mode="decoupled")
            ws.assign_inplace(ds, batch_size=min(b_sz, 4096), row_indices=idx_w, wrap_rows=False, allow_tma=False)

        if include_ozaki:
            ds_oz = croiss_encode_dataset(x_w, scale_mode="ozaki")
            ws_oz = CroissFP8Workspace.allocate(k=k_sz, d=d_sz, max_batch_size=b_sz, device=dev, scale_mode="ozaki")
            ws_oz.prepare_centroids_inplace(c_w, ds_oz)
            ws_oz.assign_inplace(ds_oz, batch_size=b_sz, batch_offset_ptr=boff, wrap_rows=True)

    if include_mxfp8 and _is_mxfp8_device_supported(dev):
        x_mx = torch.randn((4096, 128), dtype=torch.float32, device=dev)
        c_mx = torch.randn((512, 128), dtype=torch.float32, device=dev)
        ds_mx = croiss_encode_dataset(x_mx, scale_mode="mxfp8")
        ws_mx = CroissFP8Workspace.allocate(k=512, d=128, max_batch_size=4096, device=dev, scale_mode="mxfp8")
        ws_mx.prepare_centroids_inplace(c_mx, ds_mx)
        ws_mx.assign_inplace(ds_mx, batch_size=4096, batch_offset_ptr=boff, wrap_rows=True)

    torch.cuda.synchronize(dev)


__all__ = [
    "CROISS_FP8_AVAILABLE",
    "CROISS_MXFP8_AVAILABLE",
    "FP8_E4M3_MAX",
    "FP32_SCALE_FLOOR",
    "MXFP8_BLOCK_SIZE",
    "CroissFP8Dataset",
    "CroissFP8Workspace",
    "compute_croiss_ozaki_count",
    "compute_croiss_padded_d",
    "croiss_encode_dataset",
    "croiss_fp8_assign",
    "croiss_fp8_assign_inplace",
    "croiss_prepare_centroids_inplace",
    "croiss_separate_collapsed_splits_inplace",
    "should_use_croiss_fp8",
    "warmup_croiss_fp8_kernels",
]
