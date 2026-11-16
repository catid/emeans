"""Exhaustive numerical stability, CUDAGraph, edge-shape, and latency tests for `emeans.croiss_fp8`."""

from __future__ import annotations

import time
import pytest
import torch

from emeans.croiss_fp8 import (
    CROISS_FP8_AVAILABLE,
    CROISS_MXFP8_AVAILABLE,
    FP8_E4M3_MAX,
    FP32_SCALE_FLOOR,
    CroissFP8Dataset,
    CroissFP8Workspace,
    compute_croiss_ozaki_count,
    compute_croiss_padded_d,
    croiss_encode_dataset,
    croiss_fp8_assign,
    croiss_fp8_assign_inplace,
    croiss_prepare_centroids_inplace,
    croiss_separate_collapsed_splits_inplace,
    should_use_croiss_fp8,
    warmup_croiss_fp8_kernels,
)
from emeans.fast_ops import FP8Quantizer, fp8_assign

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and CROISS_FP8_AVAILABLE),
    reason="CUDA and torch.float8_e4m3fn + Triton required for CroissFP8 tests",
)


@pytest.fixture(scope="module", autouse=True)
def _prewarm_kernels() -> None:
    if torch.cuda.is_available() and CROISS_FP8_AVAILABLE:
        warmup_croiss_fp8_kernels("cuda", include_mxfp8=CROISS_MXFP8_AVAILABLE, include_ozaki=True)


def _assert_zero_fp8_nan_bytes(t: torch.Tensor) -> None:
    """In E4M3FN, 0x7F (+NaN) and 0xFF (-NaN) have low-7-bits == 0x7F."""
    assert t.dtype == torch.float8_e4m3fn
    raw = t.view(torch.uint8)
    nan_mask = (raw & 0x7F) == 0x7F
    assert int(nan_mask.sum().item()) == 0, f"Found {int(nan_mask.sum().item())} 0x7F/0xFF FP8 NaN bytes!"


def _decoded_represented_vectors(
    ds: CroissFP8Dataset,
    ws: CroissFP8Workspace,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct exact FP64 represented points and centroids `x_tilde, c_tilde` in input frame."""
    d = ds.d
    gs = float(ds.gscale_ptr.double().item())
    inv_gs = 1.0 / gs if gs > 0.0 else 1.0

    if ds.scale_mode == "mxfp8":
        assert ds.ux_blocks is not None and ws.uc_blocks is not None
        pad_d = ds.padded_d
        ux = torch.ldexp(
            torch.ones_like(ds.ux_blocks, dtype=torch.float64),
            ds.ux_blocks.to(torch.int32) - 127,
        )
        uc = torch.ldexp(
            torch.ones_like(ws.uc_blocks, dtype=torch.float64),
            ws.uc_blocks.to(torch.int32) - 127,
        )
        xr = (ds.qx.double().reshape(ds.n, pad_d // 32, 32) * ux.unsqueeze(-1)).reshape(ds.n, pad_d)[:, :d]
        cr = (ws.qc.double().reshape(ws.k, pad_d // 32, 32) * uc.unsqueeze(-1)).reshape(ws.k, pad_d)[:, :d]
    elif ds.scale_mode == "ozaki":
        assert ds.dom_cols is not None
        xr_unit = ds.qx[:, :d].double().clone()
        cr_unit = ws.qc[:, :d].double().clone()
        m_oz = int(ds.dom_cols.numel())
        for m_idx in range(m_oz):
            col = int(ds.dom_cols[m_idx].item())
            aux = d + 3 * m_idx
            xr_unit[:, col] = ds.qx[:, col].double() + ds.qx[:, aux + 1].double()
            cr_unit[:, col] = ws.qc[:, col].double() + ws.qc[:, aux + 0].double()
        xr = ds.sx[:, None].double() * xr_unit
        cr = ws.sc[:, None].double() * cr_unit
    else:
        xr = ds.sx[:, None].double() * ds.qx[:, :d].double()
        cr = ws.sc[:, None].double() * ws.qc[:, :d].double()

    mu64 = ds.mu_fp64 if ds.center else torch.zeros_like(ds.mu_fp64)
    x_tilde = mu64[None, :] + xr * inv_gs
    c_tilde = mu64[None, :] + cr * inv_gs
    return x_tilde, c_tilde


# ==============================================================================
# 1. Exact Saved-Payload Score Replay & Represented-Norm Parity Across Paths
# ==============================================================================


@pytest.mark.parametrize(
    "shape",
    [
        (512, 64, 54),      # pad_d=64 TMA
        (1024, 2048, 128),  # Small-B TMA Split-K
        (16384, 512, 128),  # Full-D 3-Stage TMA
        (4096, 512, 256),   # Full-D D=256 TMA
        (2048, 256, 512),   # Tiled-D D=512 TMA
        (1536, 256, 768),   # Tiled-D D=768 TMA
    ],
)
def test_saved_payload_represented_norm_parity_all_paths(shape: tuple[int, int, int]) -> None:
    n, k, d = shape
    gen = torch.Generator(device="cuda").manual_seed(100 + d + k)
    x = torch.randn((n, d), generator=gen, device="cuda", dtype=torch.float32) * 7.5 + 350.0
    c = torch.randn((k, d), generator=gen, device="cuda", dtype=torch.float32) * 7.5 + 350.0

    ds = croiss_encode_dataset(x, scale_mode="decoupled")
    ws = CroissFP8Workspace.allocate(k=k, d=d, max_batch_size=n, device="cuda")
    ws.prepare_centroids_inplace(c, ds)

    _assert_zero_fp8_nan_bytes(ds._qx_wrap)
    _assert_zero_fp8_nan_bytes(ws.qc)

    dists_tma, lbl_tma = ws.assign_inplace(ds, batch_size=n, wrap_rows=False, allow_tma=True)
    dists_ptr, lbl_ptr = ws.assign_inplace(ds, batch_size=n, wrap_rows=False, allow_tma=False)
    # Note: assign_inplace writes into ws.dists/ws.labels unless separate out buffers passed
    d_tma_c, l_tma_c = dists_tma.clone(), lbl_tma.clone()
    d_ptr_c, l_ptr_c = ws.assign_inplace(
        ds,
        batch_size=n,
        wrap_rows=False,
        allow_tma=False,
        out_dists=torch.empty_like(d_tma_c),
        out_labels=torch.empty_like(l_tma_c),
    )

    x_tilde, c_tilde = _decoded_represented_vectors(ds, ws)
    d2_ref = torch.cdist(x_tilde, c_tilde).square()
    lbl_ref = d2_ref.argmin(dim=1)
    dist_ref = d2_ref.gather(1, lbl_ref[:, None]).squeeze(1).float()

    assert torch.equal(l_tma_c, lbl_ref)
    assert torch.equal(l_ptr_c, lbl_ref)
    torch.testing.assert_close(d_tma_c, dist_ref, rtol=2e-5, atol=1e-4)
    torch.testing.assert_close(d_ptr_c, dist_ref, rtol=2e-5, atol=1e-4)


@pytest.mark.parametrize("scale_mode", ["ozaki", "mxfp8"])
def test_ozaki_and_mxfp8_represented_norm_parity(scale_mode: str) -> None:
    if scale_mode == "mxfp8" and not CROISS_MXFP8_AVAILABLE:
        pytest.skip("SM100+ tl.dot_scaled required for mxfp8")

    n, k, d = 2048, 256, 93
    gen = torch.Generator(device="cuda").manual_seed(777)
    x = torch.randn((n, d), generator=gen, device="cuda", dtype=torch.float32) * 12.0 - 1200.0
    c = torch.randn((k, d), generator=gen, device="cuda", dtype=torch.float32) * 12.0 - 1200.0

    ds = croiss_encode_dataset(x, scale_mode=scale_mode)
    ws = CroissFP8Workspace.allocate(k=k, d=d, max_batch_size=n, device="cuda", scale_mode=scale_mode)
    ws.prepare_centroids_inplace(c, ds)

    _assert_zero_fp8_nan_bytes(ds._qx_wrap)
    _assert_zero_fp8_nan_bytes(ws.qc)

    dists, labels = ws.assign_inplace(ds, batch_size=n, wrap_rows=False)
    x_tilde, c_tilde = _decoded_represented_vectors(ds, ws)
    d2_ref = torch.cdist(x_tilde, c_tilde).square()
    lbl_ref = d2_ref.argmin(dim=1)
    dist_ref = d2_ref.gather(1, lbl_ref[:, None]).squeeze(1).float()

    # Allow tiny FP32 dot tie-breaking if two centroids are nearly equidistant, check MSE parity
    chosen_d2 = d2_ref.gather(1, labels[:, None]).squeeze(1).float()
    torch.testing.assert_close(dists, chosen_d2, rtol=5e-5, atol=1e-3)
    torch.testing.assert_close(chosen_d2.mean(), dist_ref.mean(), rtol=1e-6, atol=1e-5)


# ==============================================================================
# 2. Zero `0x7f` NaN & Extreme Dynamic-Range / Pathological Input Invariants
# ==============================================================================


@pytest.mark.parametrize(
    "case_name",
    [
        "all_zeros",
        "all_constant_pos",
        "all_constant_neg_shift",
        "subnormal_1e_minus_25",
        "extreme_1e_plus_20",
        "extreme_1e_plus_25",
        "fp16_near_max",
        "rogue_nan_and_inf",
    ],
)
def test_zero_fp8_nan_and_finite_outputs_on_extreme_inputs(case_name: str) -> None:
    n, k, d = 1024, 128, 128
    gen = torch.Generator(device="cuda").manual_seed(999)
    z_x = torch.randn((n, d), generator=gen, device="cuda", dtype=torch.float32)
    z_c = torch.randn((k, d), generator=gen, device="cuda", dtype=torch.float32)

    if case_name == "all_zeros":
        x = torch.zeros((n, d), device="cuda", dtype=torch.float32)
        c = torch.zeros((k, d), device="cuda", dtype=torch.float32)
    elif case_name == "all_constant_pos":
        x = torch.full((n, d), 12345.0, device="cuda", dtype=torch.float32)
        c = torch.full((k, d), 12345.0, device="cuda", dtype=torch.float32)
    elif case_name == "all_constant_neg_shift":
        x = torch.full((n, d), -20000.0, device="cuda", dtype=torch.float32)
        c = torch.full((k, d), -20000.0, device="cuda", dtype=torch.float32)
    elif case_name == "subnormal_1e_minus_25":
        x = z_x * 1e-25
        c = z_c * 1e-25
    elif case_name == "extreme_1e_plus_20":
        x = z_x * 1e20
        c = z_c * 1e20
    elif case_name == "extreme_1e_plus_25":
        x = z_x * 1e25
        c = z_c * 1e25
    elif case_name == "fp16_near_max":
        x = (z_x.clamp(-3.0, 3.0) * 20000.0).to(torch.float16)
        c = (z_c.clamp(-3.0, 3.0) * 20000.0).to(torch.float16)
    elif case_name == "rogue_nan_and_inf":
        x = z_x.clone()
        c = z_c.clone()
        x[0, 0] = float("nan")
        x[1, 5] = float("inf")
        x[2, 9] = float("-inf")
        c[0, 3] = float("nan")
        c[1, 7] = float("inf")
    else:
        raise AssertionError(case_name)

    for mode in ("decoupled", "ozaki"):
        ds = croiss_encode_dataset(x, scale_mode=mode)
        ws = CroissFP8Workspace.allocate(k=k, d=d, max_batch_size=n, device="cuda", scale_mode=mode)
        ws.prepare_centroids_inplace(c, ds)

        _assert_zero_fp8_nan_bytes(ds._qx_wrap)
        _assert_zero_fp8_nan_bytes(ws.qc)
        assert torch.isfinite(ds.sx).all() and (ds.sx >= FP32_SCALE_FLOOR).all()
        assert torch.isfinite(ws.sc).all() and (ws.sc >= FP32_SCALE_FLOOR).all()
        assert torch.isfinite(ds.x_norm).all() and (ds.x_norm >= 0.0).all()
        assert torch.isfinite(ws.c_norm).all() and (ws.c_norm >= 0.0).all()

        dists, labels = ws.assign_inplace(ds, batch_size=n, wrap_rows=False)
        assert torch.isfinite(dists).all()
        assert (dists >= 0.0).all()
        assert ((labels >= 0) & (labels < k)).all()

        if case_name in ("subnormal_1e_minus_25", "extreme_1e_plus_20", "extreme_1e_plus_25"):
            # Verify true scale-invariant argmin preservation even at 1e-25 and 1e+25!
            d2_unit = torch.cdist(z_x.double(), z_c.double()).square()
            mse_unit_ref = d2_unit.min(dim=1).values.mean().item()
            mse_unit_got = d2_unit[torch.arange(n, device="cuda"), labels].mean().item()
            rel_excess = (mse_unit_got - mse_unit_ref) / max(mse_unit_ref, 1e-30)
            assert rel_excess < 5e-3, f"{case_name} ({mode}) rel_excess={rel_excess:.6f}"


# ==============================================================================
# 3. Skewed Multi-Scale Coordinates (`1e-6` to `1e+6`) + `+20,000` Shift
# ==============================================================================


def test_multiscale_rows_1e_minus_6_to_1e_plus_6_plus_20k_shift() -> None:
    """70% fine sub-clusters at scale 1e-6 + 30% macro clusters at 1e+6, shifted by +20,000!"""
    d = 128
    k_micro, k_macro = 16, 8
    k = k_micro + k_macro
    n_micro, n_macro = 2800, 1200
    n = n_micro + n_macro

    gen = torch.Generator(device="cuda").manual_seed(2026)
    c_micro = torch.randn((k_micro, d), generator=gen, device="cuda", dtype=torch.float64) * 1e-6
    c_macro = torch.randn((k_macro, d), generator=gen, device="cuda", dtype=torch.float64) * 1e6

    idx_mi = torch.randint(0, k_micro, (n_micro,), generator=gen, device="cuda")
    idx_ma = torch.randint(0, k_macro, (n_macro,), generator=gen, device="cuda")
    x_micro = c_micro[idx_mi] + torch.randn((n_micro, d), generator=gen, device="cuda", dtype=torch.float64) * 1.5e-7
    x_macro = c_macro[idx_ma] + torch.randn((n_macro, d), generator=gen, device="cuda", dtype=torch.float64) * 1.5e5

    shift = 20000.0
    x64 = torch.cat([x_micro, x_macro], dim=0) + shift
    c64 = torch.cat([c_micro, c_macro], dim=0) + shift

    # Exact FP64 ground truth on the 1e-6 micro-cluster slice (avoid uncentered mm expansion at +20,000!)
    d2_micro_64 = torch.cdist(x64[:n_micro], c64, compute_mode="donot_use_mm_for_euclid_dist").square()
    lbl_micro_64 = d2_micro_64.argmin(dim=1)
    mse_micro_64 = d2_micro_64.gather(1, lbl_micro_64[:, None]).mean().item()

    # Global FP8 collapses all 1e-6 points into 0 under 1e+6 outlier span!
    q_glob = FP8Quantizer.calibrate(x64.float(), center_mode="always")
    _, lbl_glob = fp8_assign(x64.float(), c64.float(), quantizer=q_glob)
    mse_micro_glob = d2_micro_64.gather(1, lbl_glob[:n_micro, None]).mean().item()
    assert mse_micro_glob > mse_micro_64 * 5.0  # Global FP8 completely loses micro-clusters

    # CroissFP8 (with FP64 median origin + decoupled per-row/per-centroid scales) resolves BOTH regimes!
    _, lbl_croiss = croiss_fp8_assign(x64, c64, scale_mode="decoupled")
    mse_micro_croiss = d2_micro_64.gather(1, lbl_croiss[:n_micro, None]).mean().item()
    rel_excess_micro = (mse_micro_croiss - mse_micro_64) / mse_micro_64
    agree_micro = (lbl_croiss[:n_micro] == lbl_micro_64).float().mean().item()

    assert rel_excess_micro < 0.005, f"Micro-cluster MSE excess {rel_excess_micro:.6f} >= 0.5%"
    assert agree_micro > 0.98, f"Micro-cluster argmin agreement {agree_micro:.4f} <= 98%"


def test_multiscale_features_and_ozaki_anisotropic_dominance() -> None:
    """Features spanning 1e-6..1e+6 + physical sensor anisotropy where Ozaki beats 1-lane & global FP8."""
    n, k, d = 4096, 512, 64
    gen = torch.Generator(device="cuda").manual_seed(31415)

    # Part A: Log-uniform feature scales spanning 12 orders of magnitude (1e-6 .. 1e+6)
    exps = torch.linspace(-6.0, 6.0, d, device="cuda", dtype=torch.float64)
    f_scales = torch.pow(10.0, exps)
    x_ms = (torch.randn((n, d), generator=gen, device="cuda", dtype=torch.float64) * f_scales[None, :]).float()
    c_ms = x_ms[:k] + (torch.randn((k, d), generator=gen, device="cuda", dtype=torch.float64) * f_scales[None, :] * 0.2).float()

    d2_64 = torch.cdist(x_ms.double(), c_ms.double(), compute_mode="donot_use_mm_for_euclid_dist").square()
    mse_64 = d2_64.min(dim=1).values.mean().item()
    _, lbl_dec = croiss_fp8_assign(x_ms, c_ms, scale_mode="decoupled")
    _, lbl_oz = croiss_fp8_assign(x_ms, c_ms, scale_mode="ozaki")
    mse_dec = d2_64[torch.arange(n, device="cuda"), lbl_dec].mean().item()
    mse_oz = d2_64[torch.arange(n, device="cuda"), lbl_oz].mean().item()
    assert (mse_dec - mse_64) / mse_64 < 5e-3
    assert (mse_oz - mse_64) / mse_64 < 5e-4

    # Part B: High-concentration sensor geometry (2 features in [-4500, 4500], 52 binary {0, 1} flags, fine K=512 Voronoi)
    d_sens = 54
    centers_base = torch.zeros((k, d_sens), device="cuda", dtype=torch.float32)
    centers_base[:, 0] = torch.linspace(1850.0, 3850.0, k, device="cuda")
    centers_base[:, 5] = torch.sin(torch.linspace(0.0, 25.0, k, device="cuda")) * 3500.0 + 2500.0
    centers_base[:, 9] = torch.cos(torch.linspace(0.0, 31.0, k, device="cuda")) * 3200.0 + 2000.0
    centers_base[:, 10:] = (torch.rand((k, d_sens - 10), generator=gen, device="cuda") > 0.85).float()

    src_k = torch.randint(0, k, (n,), generator=gen, device="cuda")
    noise = torch.randn((n, d_sens), generator=gen, device="cuda") * 12.0
    noise[:, 10:] *= 0.02
    x_sens = centers_base[src_k] + noise

    d2_s64 = torch.cdist(x_sens.double(), centers_base.double()).square()
    mse_s64 = d2_s64.min(dim=1).values.mean().item()

    _, l_dec_s = croiss_fp8_assign(x_sens, centers_base, scale_mode="decoupled")
    _, l_oz_s = croiss_fp8_assign(x_sens, centers_base, scale_mode="ozaki")
    ex_dec = (d2_s64[torch.arange(n, device="cuda"), l_dec_s].mean().item() - mse_s64) / mse_s64
    ex_oz = (d2_s64[torch.arange(n, device="cuda"), l_oz_s].mean().item() - mse_s64) / mse_s64

    assert ex_oz < 0.002, f"Ozaki MSE excess {ex_oz:.6f} >= 0.2%"
    assert ex_oz < ex_dec * 0.1, f"Expected Ozaki ({ex_oz:.6f}) << 1-lane ({ex_dec:.6f})"


# ==============================================================================
# 4. Shift Equivariance (`+20,000`) Across Dtypes & Adversarial Prime Shapes
# ==============================================================================


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16, torch.float64])
def test_coordinate_shift_equivariance_across_dtypes(dtype: torch.dtype) -> None:
    n, k, d = 1024, 128, 96
    gen = torch.Generator(device="cuda").manual_seed(555)
    if dtype == torch.bfloat16:
        # In BF16 (7 mantissa bits), ULP at 16384 is 128; scale signal to 512 with +16384 shift so x+shift is non-degenerate
        x_base = (torch.randn((n, d), generator=gen, device="cuda", dtype=torch.float32) * 512.0).to(dtype)
        c_base = (torch.randn((k, d), generator=gen, device="cuda", dtype=torch.float32) * 512.0).to(dtype)
        shift_val = 16384.0
    elif dtype == torch.float16:
        # In FP16 (10 mantissa bits, max 65504), ULP at 16384 is 16; scale signal to 128 with +16384 shift
        x_base = (torch.randn((n, d), generator=gen, device="cuda", dtype=torch.float32) * 128.0).to(dtype)
        c_base = (torch.randn((k, d), generator=gen, device="cuda", dtype=torch.float32) * 128.0).to(dtype)
        shift_val = 16384.0
    else:
        x_base = (torch.randn((n, d), generator=gen, device="cuda", dtype=torch.float32) * 2.0).to(dtype)
        c_base = (torch.randn((k, d), generator=gen, device="cuda", dtype=torch.float32) * 2.0).to(dtype)
        shift_val = 20000.0

    shift = torch.full((1, d), shift_val, device="cuda", dtype=dtype)
    x_sh = x_base + shift
    c_sh = c_base + shift

    d0, l0 = croiss_fp8_assign(x_base, c_base)
    ds, ls = croiss_fp8_assign(x_sh, c_sh)

    if dtype == torch.float64:
        # Exact 53-bit FP64 subtraction must yield 100% bit-identical labels and distances under +20,000!
        assert torch.equal(l0, ls)
        torch.testing.assert_close(d0, ds, rtol=1e-6, atol=1e-6)
    else:
        # Compare shifted assignment against exact FP64 distances of the shifted tensor itself (zero shift blowup!)
        d2_sh_true = torch.cdist(x_sh.double(), c_sh.double(), compute_mode="donot_use_mm_for_euclid_dist").square()
        mse_opt = d2_sh_true.min(dim=1).values.mean().item()
        mses = d2_sh_true[torch.arange(n, device="cuda"), ls].mean().item()
        assert (mses - mse_opt) / max(mse_opt, 1e-12) < 0.005


@pytest.mark.parametrize(
    "n,b,k,d",
    [
        (1, 1, 1, 1),
        (17, 13, 7, 3),
        (129, 127, 19, 31),
        (521, 257, 513, 63),
        (1031, 509, 257, 65),
        (1031, 509, 257, 127),
        (1031, 509, 257, 129),
        (601, 307, 131, 257),
        (409, 199, 67, 513),
        (256, 0, 32, 64),  # Empty batch B=0 edge case
    ],
)
def test_prime_boundary_shapes_and_strided_views(n: int, b: int, k: int, d: int) -> None:
    gen = torch.Generator(device="cuda").manual_seed(n * 31 + d)
    # Create non-contiguous strided column views!
    x_raw = torch.randn((n, d * 2 + 3), generator=gen, device="cuda", dtype=torch.float32)[:, 1 : 1 + 2 * d : 2]
    c_raw = torch.randn((k * 2, d + 5), generator=gen, device="cuda", dtype=torch.float32)[::2, 2 : 2 + d]
    if n > 1 or d > 1:
        assert not x_raw.is_contiguous()
    if k > 1 or d > 1:
        assert not c_raw.is_contiguous()

    ds = croiss_encode_dataset(x_raw)
    ws = CroissFP8Workspace.allocate(k=k, d=d, max_batch_size=max(b, 1), device="cuda")
    ws.prepare_centroids_inplace(c_raw, ds)

    dists, labels = ws.assign_inplace(ds, batch_size=b, row_offset=min(3, max(0, n - 1)), wrap_rows=True)
    assert dists.shape == (b,)
    assert labels.shape == (b,)
    if b > 0:
        assert torch.isfinite(dists).all()
        assert ((labels >= 0) & (labels < k)).all()
        x_tilde, c_tilde = _decoded_represented_vectors(ds, ws)
        rows = (torch.arange(b, device="cuda") + min(3, max(0, n - 1))) % n
        d2_ref = torch.cdist(x_tilde[rows], c_tilde).square()
        lbl_ref = d2_ref.argmin(dim=1)
        assert torch.equal(labels, lbl_ref)


# ==============================================================================
# 5. Zero-Sync Proof & Multi-Step `torch.cuda.CUDAGraph` Replay Bit-Identity
# ==============================================================================


@pytest.mark.parametrize("batch_size", [1024, 16384])  # Tests BOTH TMA Split-K and Full-Batch TMA!
def test_cudagraph_multistep_cyclic_replay_bit_identity_and_zero_sync(batch_size: int) -> None:
    n, k, d = 25000, 1024, 128  # 25000 is intentionally non-divisible by batch_size so batches wrap across N!
    gen = torch.Generator(device="cuda").manual_seed(4242 + batch_size)
    x = torch.randn((n, d), generator=gen, device="cuda", dtype=torch.float32) * 4.0 + 1500.0
    c_init = torch.randn((k, d), generator=gen, device="cuda", dtype=torch.float32) * 4.0 + 1500.0

    ds = croiss_encode_dataset(x)
    ws = CroissFP8Workspace.allocate(k=k, d=d, max_batch_size=batch_size, device="cuda")

    # 1. Verify strictly ZERO host-device syncs inside step calls under sync_debug_mode("error")!
    c_live = c_init.clone()
    boff = torch.tensor(0, device="cuda", dtype=torch.int64)
    ws.prepare_centroids_inplace(c_live, ds)
    ws.assign_inplace(ds, batch_size=batch_size, batch_offset=boff, wrap_rows=True)
    torch.cuda.synchronize()

    prev_mode = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(3):
            ws.prepare_centroids_inplace(c_live, ds)
            ws.assign_inplace(ds, batch_size=batch_size, batch_offset=boff, wrap_rows=True)
            boff.add_(batch_size)
    finally:
        torch.cuda.set_sync_debug_mode(prev_mode)
    torch.cuda.synchronize()

    # 2. Multi-step (5-step super-step wrapping past N=25,000!) CUDAGraph vs Eager bit-identity!
    c_eager = c_init.clone()
    boff_eager = torch.tensor(19000, device="cuda", dtype=torch.int64)  # Immediately straddles N=25,000!
    eager_dists: list[torch.Tensor] = []
    eager_labels: list[torch.Tensor] = []
    for step in range(5):
        ws.prepare_centroids_inplace(c_eager, ds)
        d_s, l_s = ws.assign_inplace(ds, batch_size=batch_size, batch_offset=boff_eager, wrap_rows=True)
        eager_dists.append(d_s.clone())
        eager_labels.append(l_s.clone())
        boff_eager.add_(batch_size)
        c_eager.mul_(0.995).add_(0.005 * float(step + 1))

    c_graph = c_init.clone()
    boff_graph = torch.tensor(19000, device="cuda", dtype=torch.int64)
    graph_dists = [torch.empty((batch_size,), dtype=torch.float32, device="cuda") for _ in range(5)]
    graph_labels = [torch.empty((batch_size,), dtype=torch.int64, device="cuda") for _ in range(5)]

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for step in range(5):
            ws.prepare_centroids_inplace(c_graph, ds)
            ws.assign_inplace(
                ds,
                batch_size=batch_size,
                batch_offset=boff_graph,
                wrap_rows=True,
                out_dists=graph_dists[step],
                out_labels=graph_labels[step],
            )
            boff_graph.add_(batch_size)
            c_graph.mul_(0.995).add_(0.005 * float(step + 1))

    # Mutate state, reset to exact starting point, and replay twice to prove deterministic idempotence!
    for _ in range(2):
        c_graph.copy_(c_init)
        boff_graph.fill_(19000)
        g.replay()
        torch.cuda.synchronize()
        for step in range(5):
            assert torch.equal(graph_labels[step], eager_labels[step]), f"Label mismatch at step {step}"
            assert torch.equal(graph_dists[step], eager_dists[step]), f"Bitwise dist mismatch at step {step}"


# ==============================================================================
# 6. Regression Tests (#166 Direct Dataclass Init, Split Guard, CPU Fallback, Speed)
# ==============================================================================


def test_direct_dataclass_init_none_gscale_regression_166() -> None:
    """Regression test for #166: direct `CroissFP8Dataset(...)` with `inv_gscale_sq_ptr=None`."""
    n, k, d = 1024, 1024, 128
    qx = torch.zeros((n, 128), device="cuda", dtype=torch.float8_e4m3fn)
    sx = torch.ones((n,), device="cuda", dtype=torch.float32)
    xn = torch.zeros((n,), device="cuda", dtype=torch.float32)
    mu = torch.zeros((d,), device="cuda", dtype=torch.float32)
    mu64 = torch.zeros((d,), device="cuda", dtype=torch.float64)

    ds = CroissFP8Dataset(qx=qx, sx=sx, x_norm=xn, mu=mu, mu_fp64=mu64, n=n, d=d, padded_d=128)
    assert ds.inv_gscale_sq_ptr is not None
    ws = CroissFP8Workspace.allocate(k=k, d=d, max_batch_size=n, device="cuda")
    ws.prepare_centroids_inplace(torch.randn((k, d), device="cuda"), None, centroids_already_centered=True)

    # Also test explicitly forcing inv_gscale_sq_ptr = None after construction across Split-K, TMA, and Ptr!
    ds.inv_gscale_sq_ptr = None
    d_sk, l_sk = croiss_fp8_assign_inplace(ds, ws, batch_size=512, wrap_rows=False, allow_tma=True)
    d_ptr, l_ptr = croiss_fp8_assign_inplace(ds, ws, batch_size=n, wrap_rows=False, allow_tma=False)
    assert torch.isfinite(d_sk).all() and torch.isfinite(d_ptr).all()
    assert l_sk.shape == (512,) and l_ptr.shape == (n,)


def test_split_collapse_separation_and_cpu_fallback_parity() -> None:
    m, d = 64, 128
    donor = torch.full((m, d), 20000.0, device="cuda", dtype=torch.float32)
    donor[0] = 0.0
    child = donor.clone() + 1e-7  # Collapses in both FP32@20000 and E4M3!

    ds = croiss_encode_dataset(donor)
    prev = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        croiss_separate_collapsed_splits_inplace(donor, child, ds)
    finally:
        torch.cuda.set_sync_debug_mode(prev)

    dq = torch.empty((m, 128), dtype=torch.float8_e4m3fn, device="cuda")
    cq = torch.empty((m, 128), dtype=torch.float8_e4m3fn, device="cuda")
    sd = torch.empty((m,), dtype=torch.float32, device="cuda")
    sc = torch.empty((m,), dtype=torch.float32, device="cuda")
    nd = torch.empty((m,), dtype=torch.float32, device="cuda")
    nc = torch.empty((m,), dtype=torch.float32, device="cuda")
    croiss_prepare_centroids_inplace(donor, ds, dq, sd, nd)
    croiss_prepare_centroids_inplace(child, ds, cq, sc, nc)
    rep_d = sd[:, None] * dq.float()
    rep_c = sc[:, None] * cq.float()
    assert not bool((rep_d == rep_c).all(dim=1).any().item())

    # CPU vs CUDA parity check
    x_cpu = torch.randn((256, 73), dtype=torch.float32) * 3.0 - 500.0
    c_cpu = torch.randn((32, 73), dtype=torch.float32) * 3.0 - 500.0
    d_cpu, l_cpu = croiss_fp8_assign(x_cpu, c_cpu, scale_mode="decoupled")
    d_gpu, l_gpu = croiss_fp8_assign(x_cpu.cuda(), c_cpu.cuda(), scale_mode="decoupled")
    assert torch.equal(l_cpu, l_gpu.cpu())
    torch.testing.assert_close(d_cpu, d_gpu.cpu(), rtol=1e-4, atol=1e-4)


def test_preencode_latency_budget_and_heuristic() -> None:
    assert not should_use_croiss_fp8(2000, 128, 128, nepochs=1.0)
    assert should_use_croiss_fp8(200000, 2048, 128, nepochs=2.0)
    assert compute_croiss_padded_d(54, "decoupled") == 64
    assert compute_croiss_padded_d(54, "ozaki") == 128
    assert compute_croiss_ozaki_count(54) == 6

    x_200k = torch.randn((200_000, 128), device="cuda", dtype=torch.float32)
    ds_200k = croiss_encode_dataset(x_200k)
    for _ in range(4):
        croiss_encode_dataset(x_200k, out=ds_200k)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(15):
        croiss_encode_dataset(x_200k, out=ds_200k)
    torch.cuda.synchronize()
    ms_200k = (time.perf_counter() - t0) * 1000.0 / 15.0
    assert ms_200k < 1.5, f"200K x 128 pre-encode {ms_200k:.3f} ms >= 1.5 ms budget!"

    x_1m = torch.randn((1_000_000, 128), device="cuda", dtype=torch.float32)
    ds_1m = croiss_encode_dataset(x_1m)
    for _ in range(3):
        croiss_encode_dataset(x_1m, out=ds_1m)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        croiss_encode_dataset(x_1m, out=ds_1m)
    torch.cuda.synchronize()
    ms_1m = (time.perf_counter() - t0) * 1000.0 / 10.0
    assert ms_1m < 6.0, f"1M x 128 pre-encode {ms_1m:.3f} ms >= 6.0 ms budget!"
