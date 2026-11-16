"""Comprehensive unit, parity, ablation, shift-invariance, rare-mode, and adversarial tests for `UltraEMeans`."""

from __future__ import annotations

import math
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pytest
import torch

from emeans.datasets import make_synthetic_dataset
from emeans.emeans import EMeans
from emeans.kernels import exact_mse_fp64
from emeans.super_emeans import twin_safe_huygens_protection
from emeans.ultra_emeans import (
    HALF_GAUSSIAN_MEAN,
    ULTRA_ABLATION_ARMS,
    UltraEMeans,
    _launch_syncfree_huygens_protection,
    build_ultra_arm,
    ultra_simplex_eb_finalize,
)

ALL_DEVICES = ["cpu", "cuda:0"] if torch.cuda.is_available() else ["cpu"]

EXPECTED_ULTRA_ARMS = {
    "ultra_emeans",
    "ultra_emeans_wide",
    "ultra_loo_no_turkpp_init",
    "ultra_loo_rademacher055_split",
    "ultra_loo_oja008_split",
    "ultra_loo_decoupled_no_huygens",
    "ultra_loo_no_revive_stop_flush",
    "ultra_loo_closing_mean",
    "ultra_loo_smeans_eb020",
    "ultra_loo_super_slow_eb",
    "ultra_loo_no_head_edit",
}


# ==============================================================================
# 1. API Contract, Pre-Registered Ablation Matrix, & Input Validation
# ==============================================================================


def test_ultra_ablation_arms_registry_and_validation() -> None:
    """Verify all 11 pre-registered `ULTRA_ABLATION_ARMS` instantiate cleanly and invalid params raise."""
    assert set(ULTRA_ABLATION_ARMS.keys()) == EXPECTED_ULTRA_ARMS
    assert math.isclose(HALF_GAUSSIAN_MEAN, math.sqrt(2.0 / math.pi), rel_tol=1e-12)

    for arm_name in EXPECTED_ULTRA_ARMS:
        model = build_ultra_arm(arm_name, d=16, k=8, nepochs=2.0, seed=7)
        assert isinstance(model, UltraEMeans)

    with pytest.raises(KeyError):
        build_ultra_arm("nonexistent_arm", d=16, k=8)
    with pytest.raises(ValueError, match="d="):
        UltraEMeans(d=0, k=4)
    with pytest.raises(ValueError, match="k="):
        UltraEMeans(d=4, k=0)
    with pytest.raises(ValueError, match="nepochs"):
        UltraEMeans(d=4, k=2, nepochs=0.0)
    with pytest.raises(ValueError, match="nepochs"):
        UltraEMeans(d=4, k=2, nepochs=float("nan"))
    with pytest.raises(ValueError, match="init_mode"):
        UltraEMeans(d=4, k=2, init_mode="bad")
    with pytest.raises(ValueError, match="schedule_mode"):
        UltraEMeans(d=4, k=2, schedule_mode="bad")
    with pytest.raises(ValueError, match="donor_mode"):
        UltraEMeans(d=4, k=2, donor_mode="bad")
    with pytest.raises(ValueError, match="split_mode"):
        UltraEMeans(d=4, k=2, split_mode="bad")
    with pytest.raises(ValueError, match="finalization"):
        UltraEMeans(d=4, k=2, finalization="bad")
    with pytest.raises(ValueError, match="wide_step_mult"):
        UltraEMeans(d=4, k=2, wide_step_mult=0)

    m = UltraEMeans(d=8, k=4, nepochs=1.0)
    with pytest.raises(RuntimeError, match="fitted"):
        m.assign(torch.randn(10, 8))
    with pytest.raises(ValueError, match="2D"):
        m.fit(torch.randn(10))
    with pytest.raises(ValueError, match="Expected dimension"):
        m.fit(torch.randn(10, 6))
    with pytest.raises(ValueError, match="must be >="):
        m.fit(torch.randn(3, 8))

    x_nan = torch.randn(16, 8)
    x_nan[3, 2] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        m.fit(x_nan)

    x_inf = torch.randn(16, 8)
    x_inf[1, 4] = float("-inf")
    with pytest.raises(ValueError, match="finite"):
        m.fit(x_inf)

    c_bad = torch.randn(4, 8)
    c_bad[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        m.fit(torch.randn(16, 8), init_centroids=c_bad)


@pytest.mark.parametrize("dev", ALL_DEVICES)
@pytest.mark.parametrize("arm_name", sorted(EXPECTED_ULTRA_ARMS))
def test_all_ultra_ablation_arms_execute_cleanly(dev: str, arm_name: str) -> None:
    """Verify every `ULTRA_ABLATION_ARMS` configuration runs cleanly on CPU and CUDA."""
    ds = make_synthetic_dataset("zipf_gmm", n_train=1536, n_val=256, n_test=256, d=32, k=32, seed=9)
    x = torch.from_numpy(ds.train).to(dev)

    model = build_ultra_arm(arm_name, d=32, k=32, nepochs=2.0, seed=19)
    c, a = model.fit(x, capture_assignments=True)

    assert c.shape == (32, 32)
    assert c.dtype == torch.float32
    assert torch.isfinite(c).all()
    assert a is not None and a.shape == (1536,) and bool(((a >= 0) & (a < 32)).all())
    assert model.init_time_sec >= 0.0 and model.train_time_sec > 0.0
    assert math.isclose(model.total_time_sec, model.init_time_sec + model.train_time_sec, rel_tol=1e-12)

    sq_d, lbl = model.assign(x[:128])
    assert sq_d.shape == (128,) and sq_d.dtype == torch.float32
    assert lbl.shape == (128,) and lbl.dtype == torch.int64
    assert torch.isfinite(sq_d).all() and bool((sq_d >= 0.0).all())
    assert bool(((lbl >= 0) & (lbl < 32)).all())


@pytest.mark.parametrize("dev", ALL_DEVICES)
def test_numpy_and_extra_modes(dev: str) -> None:
    """Verify NumPy float32/float64 I/O and auxiliary modes (`coreset_kpp`, `random`, `ema_only`, `off`)."""
    rng = np.random.default_rng(42)
    x_np64 = rng.standard_normal((512, 24))
    m_np = UltraEMeans(d=24, k=12, nepochs=1.5, seed=7)
    c_np, a_np = m_np.fit(x_np64, capture_assignments=True)
    assert isinstance(c_np, np.ndarray) and c_np.dtype == np.float32 and c_np.shape == (12, 24)
    assert isinstance(a_np, np.ndarray) and a_np.dtype == np.int64 and a_np.shape == (512,)
    d_np, l_np = m_np.assign(x_np64[:40])
    assert isinstance(d_np, np.ndarray) and d_np.dtype == np.float32 and d_np.shape == (40,)
    assert isinstance(l_np, np.ndarray) and l_np.dtype == np.int64 and l_np.shape == (40,)

    x_t = torch.from_numpy(x_np64.astype(np.float32)).to(dev)
    for kw in [
        {"init_mode": "coreset_kpp"},
        {"init_mode": "random"},
        {"donor_mode": "usage_only"},
        {"donor_mode": "d2"},
        {"split_mode": "checkerboard"},
        {"split_mode": "rms_iso", "use_feat_scale": False},
        {"finalization": "ema_only"},
        {"schedule_mode": "power_cosine"},
        {"schedule_mode": "constant"},
        {"revival_mode": "off"},
    ]:
        m = UltraEMeans(d=24, k=12, nepochs=1.5, seed=3, **kw)
        c, _ = m.fit(x_t, capture_assignments=False)
        assert c.shape == (12, 24) and torch.isfinite(c).all()
        if kw.get("revival_mode") == "off" and not m.head_edit:
            assert m.n_splits == 0


# ==============================================================================
# 2. Mechanistic Proofs of Discoveries 1, 2, 3, and 4
# ==============================================================================


@pytest.mark.parametrize("dev", ALL_DEVICES)
def test_discovery1_half_gaussian_oja_split_beats_rademacher_and_oja008(dev: str) -> None:
    """Verify Half-Gaussian `sqrt(2/pi) * sigma_{j,1} * v_{j,1}` split beats `0.55*Rademacher` and `0.08*Oja`."""
    device = torch.device(dev)
    gen = torch.Generator(device=device).manual_seed(2026)
    d = 64
    # Two separated modes along an off-coordinate diagonal unit axis `u_true` in R^64,
    # seeded with 1 centroid on the merged pair and 1 dead outlier centroid at +100.0!
    u_raw = torch.randn((1, d), generator=gen, device=device)
    u_true = u_raw / u_raw.norm()
    x_left = -4.0 * u_true + 0.25 * torch.randn((1024, d), generator=gen, device=device)
    x_right = 4.0 * u_true + 0.25 * torch.randn((1024, d), generator=gen, device=device)
    x_all = torch.cat([x_left, x_right], dim=0)

    c_init = torch.zeros((2, d), dtype=torch.float32, device=device)
    c_init[0] = 0.0      # Sits at midpoint of merged bimodal donor
    c_init[1] = 100.0    # Dead centroid that triggers revival at step 5!

    # Run 6 steps (`batch_size=340`, `nepochs=1.0`, revival fires at step 5 with only 1 step left!)
    m_ultra = UltraEMeans(
        d=d, k=2, nepochs=1.0, batch_size=340, revive_stop_frac=1.0, head_edit=False, seed=11
    )
    m_radem = UltraEMeans(
        d=d,
        k=2,
        nepochs=1.0,
        batch_size=340,
        revive_stop_frac=1.0,
        head_edit=False,
        split_mode="rms_aniso",
        split_mult=0.55,
        donor_mode="sse_usage",
        seed=11,
    )
    m_oja008 = UltraEMeans(
        d=d,
        k=2,
        nepochs=1.0,
        batch_size=340,
        revive_stop_frac=1.0,
        head_edit=False,
        split_mode="oja_only",
        split_mult=0.08,
        flush_pass_on_split=False,
        seed=11,
    )

    c_u, _ = m_ultra.fit(x_all, init_centroids=c_init)
    c_r, _ = m_radem.fit(x_all, init_centroids=c_init)
    c_o, _ = m_oja008.fit(x_all, init_centroids=c_init)

    mse_u = exact_mse_fp64(x_all, c_u, device=device)
    mse_r = exact_mse_fp64(x_all, c_r, device=device)
    mse_o = exact_mse_fp64(x_all, c_o, device=device)
    assert mse_u < 0.5 * mse_o, f"Expected Half-Gaussian Oja ({mse_u:.3f}) << 0.08 Oja ({mse_o:.3f})"
    assert mse_u < mse_r, f"Expected Half-Gaussian Oja ({mse_u:.3f}) < Rademacher ({mse_r:.3f})"


@pytest.mark.parametrize("dev", ALL_DEVICES)
@pytest.mark.parametrize("K", [16, 193, 4200])
def test_discovery2_syncfree_huygens_exact_parity_and_leader_election(dev: str, K: int) -> None:
    """Verify SRAM `_launch_syncfree_huygens_protection` matches `twin_safe_huygens_protection` across K <= 4096 & K > 4096."""
    if dev == "cpu" and K > 512:
        pytest.skip("Large-K SRAM kernel test targeted at CUDA")

    device = torch.device(dev)
    d = 24
    gen = torch.Generator(device=device).manual_seed(77 + K)
    centroids = torch.randn((K, d), generator=gen, device=device, dtype=torch.float32) * 0.8

    # Place:
    #  - Twin pair (K-4, K-3) on isolated rare mode A at +40.0
    #  - Single guardian (K-2) on distinct rare mode B at -40.0
    #  - Zero-mass empty outlier (K-1) at +80.0 (`#93` non-shielding test!)
    centroids[K - 4] = 40.0
    centroids[K - 3] = 40.0 + 1e-3
    centroids[K - 2] = -40.0
    centroids[K - 1] = 80.0

    usage = torch.full((K,), 20.0, dtype=torch.float32, device=device)
    usage[K - 4] = 0.8
    usage[K - 3] = 0.6
    usage[K - 2] = 0.9
    usage[K - 1] = 1e-5

    surv_mass = usage.clone()
    surv_mass[K - 1] = 0.0

    thresh = usage.mean() * 0.1
    low_mask = usage < thresh
    ref_dist = torch.tensor(10.0, dtype=torch.float32, device=device)

    prot_ultra = _launch_syncfree_huygens_protection(
        centroids_c=centroids,
        low_mask=low_mask,
        usage=usage,
        survival_mass=surv_mass,
        thresh=thresh,
        ref_distortion_mean=ref_dist,
        gamma_protect=0.5,
    )
    prot_ref = twin_safe_huygens_protection(
        centroids=centroids,
        low_mask=low_mask,
        usage=usage,
        survival_mass=surv_mass,
        thresh=thresh,
        ref_distortion_mean=ref_dist,
        gamma_protect=0.5,
    )

    assert torch.equal(prot_ultra, prot_ref)
    assert bool(prot_ultra[K - 4].item()) is True   # Twin leader protected
    assert bool(prot_ultra[K - 3].item()) is False  # Duplicate twin recycled
    assert bool(prot_ultra[K - 2].item()) is True   # Distinct rare mode protected
    assert bool(prot_ultra[K - 1].item()) is False  # Zero-mass empty outlier NOT shielded (#93)


@pytest.mark.parametrize("dev", ALL_DEVICES)
def test_discovery2_rare_mode_survival_and_patrol93_empty_outliers(dev: str) -> None:
    """Verify `UltraEMeans` preserves genuine rare modes AND revives `#93` 0-mass outlier centroids."""
    rng = np.random.default_rng(108)
    d, k = 32, 32
    bulk_centers = rng.standard_normal((28, d)).astype(np.float32) * 2.0
    rare_centers = rng.standard_normal((4, d)).astype(np.float32) * 6.0 + 25.0

    x_bulk = bulk_centers[rng.integers(0, 28, size=3984)] + 0.35 * rng.standard_normal((3984, d)).astype(np.float32)
    x_rare = np.repeat(rare_centers, 4, axis=0) + 0.1 * rng.standard_normal((16, d)).astype(np.float32)
    x_tr = torch.from_numpy(np.vstack([x_bulk, x_rare])).to(dev)

    m_ultra = UltraEMeans(d=d, k=k, nepochs=4.0, batch_size=256, seed=7)
    m_no_huy = build_ultra_arm("ultra_loo_decoupled_no_huygens", d=d, k=k, nepochs=4.0, seed=7, batch_size=256)

    c_ultra, _ = m_ultra.fit(x_tr)
    c_no_huy, _ = m_no_huy.fit(x_tr)

    rc_t = torch.from_numpy(rare_centers).to(dev)
    ultra_hits = int(((rc_t[:, None, :] - c_ultra[None, :, :]).norm(dim=2).min(dim=1).values < 2.0).sum().item())
    assert ultra_hits == 4, f"UltraEMeans should preserve all 4 isolated rare modes, got {ultra_hits}"
    assert exact_mse_fp64(x_tr, c_ultra, device=dev) < exact_mse_fp64(x_tr, c_no_huy, device=dev)

    # Patrol #93 check: 8 completely empty initial centroids at +50.0 must be 100% revived (`dead_clusters == 0`)
    c_init = x_tr[:k].clone()
    c_init[:8] = 50.0
    m93 = UltraEMeans(d=d, k=k, nepochs=1.0, seed=42)
    c93, _ = m93.fit(x_tr, init_centroids=c_init)
    assert torch.isfinite(c93).all()
    assert m93.dead_clusters == 0


@pytest.mark.parametrize("dev", ALL_DEVICES)
def test_discovery3_simplex_eb_finalize_convexity_kish_and_drift_gating(dev: str) -> None:
    """Verify `ultra_simplex_eb_finalize` convex-hull invariance, Kish scaling, and drift gating."""
    gen = torch.Generator(device=dev).manual_seed(99)
    k, d = 32, 24
    c_close = torch.randn((k, d), generator=gen, device=dev, dtype=torch.float32)
    counts = torch.randint(1, 30, (k,), generator=gen, device=dev).double()
    counts[0] = 0.0  # Empty cluster slot
    pass_sums = (c_close.double() * counts[:, None]).float()
    sse = counts * 4.0

    # 1. Stationary regime (`live == slow`): Kish rescaling `w -> 3.0 * w` must produce identical centroids!
    c_live = c_close + 0.1 * torch.randn((k, d), generator=gen, device=dev, dtype=torch.float32)
    c_slow_stat = c_live.clone()
    usage = counts.float().clamp_min(1.0)

    out1, _ = ultra_simplex_eb_finalize(
        pass_sums=pass_sums,
        pass_counts=counts,
        pass_counts_sq=counts,
        pass_sse=sse,
        live_centroids=c_live,
        slow_centroids=c_slow_stat,
        usage=usage,
        usage_inv_bias=1.0,
        alpha_end=0.1,
        nepochs=3.0,
        d=d,
    )
    sc = 3.0
    out_kish, _ = ultra_simplex_eb_finalize(
        pass_sums=pass_sums * sc,
        pass_counts=counts * sc,
        pass_counts_sq=counts * (sc * sc),
        pass_sse=sse * sc,
        live_centroids=c_live,
        slow_centroids=c_slow_stat,
        usage=usage,
        usage_inv_bias=1.0,
        alpha_end=0.1,
        nepochs=3.0,
        d=d,
    )
    torch.testing.assert_close(out1, out_kish, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(out1[0], c_live[0])  # Empty cluster falls back to live_ema

    # 2. Severe slow_ema lag (`||c_live - c_slow|| >> sigma`): drift gate must reject lagged `c_slow`!
    c_slow_lagged = c_live + 500.0
    out_drift, _ = ultra_simplex_eb_finalize(
        pass_sums=pass_sums,
        pass_counts=counts,
        pass_counts_sq=counts,
        pass_sse=sse,
        live_centroids=c_live,
        slow_centroids=c_slow_lagged,
        usage=usage,
        usage_inv_bias=1.0,
        alpha_end=0.1,
        nepochs=3.0,
        d=d,
    )
    dist_to_close = (out_drift[1:] - c_close[1:]).norm(dim=1).max().item()
    assert dist_to_close < 0.5, f"Lag-contaminated slow_ema leaked into final centroids: max diff={dist_to_close}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA backend parity check")
def test_cuda_triton_vs_torch_reference_backend_parity() -> None:
    """Verify fused Triton GPU pipeline and `_fit_torch_reference` achieve matched test MSE."""
    ds = make_synthetic_dataset("zipf_gmm", n_train=8192, n_val=512, n_test=1024, d=64, k=64, seed=21)
    x_tr = torch.from_numpy(ds.train).to("cuda:0")
    x_te = torch.from_numpy(ds.test).to("cuda:0")

    mses_triton, mses_torch = [], []
    for s in (10, 11, 12):
        c_tr, _ = UltraEMeans(d=64, k=64, nepochs=2.0, backend="triton", seed=s).fit(x_tr, capture_assignments=False)
        c_th, _ = UltraEMeans(d=64, k=64, nepochs=2.0, backend="torch", seed=s).fit(x_tr, capture_assignments=False)
        mses_triton.append(exact_mse_fp64(x_te, c_tr))
        mses_torch.append(exact_mse_fp64(x_te, c_th))

    assert float(np.mean(mses_triton)) == pytest.approx(float(np.mean(mses_torch)), rel=0.015)


def test_ultra_emeans_dominates_emeans_at_1_epoch() -> None:
    """Verify `UltraEMeans` achieves dramatically lower 1-epoch MSE than vanilla `EMeans`."""
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(12)
    centers = rng.standard_normal((128, 64), dtype=np.float32) * 5.0
    x_tr = torch.from_numpy(
        centers[rng.integers(0, 128, size=20_000)]
        + 0.4 * rng.standard_normal((20_000, 64), dtype=np.float32)
    ).to(dev)
    x_te = torch.from_numpy(
        centers[rng.integers(0, 128, size=2_000)]
        + 0.4 * rng.standard_normal((2_000, 64), dtype=np.float32)
    ).to(dev)

    c_em, _ = EMeans(d=64, k=128, nepochs=1.0, seed=42).fit(x_tr, capture_assignments=False)
    c_ul, _ = UltraEMeans(d=64, k=128, nepochs=1.0, seed=42).fit(x_tr, capture_assignments=False)

    mse_em = exact_mse_fp64(x_te, c_em)
    mse_ul = exact_mse_fp64(x_te, c_ul)
    assert mse_ul < 0.60 * mse_em, f"Expected UltraEMeans ({mse_ul:.4f}) << EMeans ({mse_em:.4f}) at T=1"


# ==============================================================================
# 3. Adversarial Stress Gate: Shifts, Dtypes, Prime Shapes, Split-K, Strides, Scales
# ==============================================================================


@pytest.mark.parametrize("dev", ALL_DEVICES)
@pytest.mark.parametrize("shift", [10_000.0, 20_000.0])
@pytest.mark.parametrize("comp_dtype", [torch.float32, torch.bfloat16, torch.float16, torch.float64])
def test_coordinate_shift_equivariance_across_dtypes(
    dev: str, shift: float, comp_dtype: torch.dtype
) -> None:
    """Verify `+10,000` and `+20,000` shifts preserve exact MSE across fp32/bf16/fp16/fp64."""
    if dev == "cpu" and comp_dtype in (torch.bfloat16, torch.float16):
        pytest.skip("Half-precision Tensor-Core path targeted at CUDA")

    ds = make_synthetic_dataset("balanced_gmm", n_train=2048, n_val=256, n_test=512, d=48, k=32, seed=31)
    x_base = torch.from_numpy(ds.train).to(device=dev, dtype=torch.float64)
    x_test = torch.from_numpy(ds.test).to(device=dev, dtype=torch.float64)
    sh_vec = torch.linspace(shift, -0.75 * shift, 48, device=dev, dtype=torch.float64)[None, :]

    m0 = UltraEMeans(d=48, k=32, nepochs=2.0, compute_dtype=comp_dtype, seed=55)
    ms = UltraEMeans(d=48, k=32, nepochs=2.0, compute_dtype=comp_dtype, seed=55)
    c0, _ = m0.fit(x_base)
    cs, _ = ms.fit(x_base + sh_vec)

    mse0 = exact_mse_fp64(x_test, c0)
    mses = exact_mse_fp64(x_test + sh_vec, cs)
    assert mses == pytest.approx(mse0, rel=2e-3)

    d_self, _ = ms.assign(cs)
    assert torch.isfinite(d_self).all() and float(d_self.max().item()) < 0.25


@pytest.mark.parametrize("dev", ALL_DEVICES)
@pytest.mark.parametrize(
    "N,D,K,nepochs",
    [
        (129, 1, 7, 1.0),
        (257, 1, 1, 1.5),
        (67, 2, 5, 0.5),
        (131, 3, 2, 2.0),
        (37, 17, 37, 1.0),     # N == K exact boundary
        (521, 127, 31, 1.5),   # Prime N, D, K below D=128
        (509, 129, 29, 2.0),   # Prime D just above FULL_D=128
        (389, 259, 19, 2.0),   # Prime D just above Oja FULL_D=256
    ],
)
def test_adversarial_prime_shapes_and_fractional_budgets(
    dev: str, N: int, D: int, K: int, nepochs: float
) -> None:
    """Stress-test `D=1, 2, 3, 17, 127, 129, 259`, `K=1`, `N=K`, and fractional `nepochs`."""
    gen = torch.Generator(device=dev).manual_seed(N + D * 13 + K * 7)
    x = torch.randn((N, D), generator=gen, device=dev, dtype=torch.float32) * 2.5

    for wide in (1, 2):
        m = UltraEMeans(d=D, k=K, nepochs=nepochs, wide_step_mult=wide, seed=19)
        c, a = m.fit(x, capture_assignments=True)
        assert c.shape == (K, D) and torch.isfinite(c).all()
        assert a is not None and a.shape == (N,) and bool(((a >= 0) & (a < K)).all())
        d0, l0 = m.assign(x[:0])
        assert d0.shape == (0,) and l0.shape == (0,)
        dq, lq = m.assign(x[: min(N, 23)])
        assert torch.isfinite(dq).all() and bool(((lq >= 0) & (lq < K)).all())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA 2D Split-K large-K kernel test")
def test_large_prime_k_splitk_kernel_and_huygens() -> None:
    """Exercise the 2D Split-K Triton assignment kernel and SRAM Huygens kernel at prime `K = 2053`."""
    dev = torch.device("cuda:0")
    gen = torch.Generator(device=dev).manual_seed(5150)
    x = torch.randn((4099, 65), generator=gen, device=dev, dtype=torch.float32)
    m = UltraEMeans(d=65, k=2053, nepochs=1.5, batch_size=512, seed=13)
    c, a = m.fit(x, capture_assignments=True)
    assert c.shape == (2053, 65) and torch.isfinite(c).all()
    assert a is not None and bool(((a >= 0) & (a < 2053)).all())
    dq, lq = m.assign(x[:257])
    assert torch.isfinite(dq).all() and bool(((lq >= 0) & (lq < 2053)).all())


@pytest.mark.parametrize("dev", ALL_DEVICES)
def test_strided_noncontiguous_inputs_across_dtypes(dev: str) -> None:
    """Verify non-contiguous strided prime-`D` inputs across `fp32/bf16/fp16/fp64`."""
    gen = torch.Generator(device=dev).manual_seed(404)
    x_wide = torch.randn((389, 130), generator=gen, device=dev, dtype=torch.float32)
    dtypes = [torch.float32, torch.bfloat16, torch.float64] + ([torch.float16] if dev != "cpu" else [])
    for dt in dtypes:
        x_strided = x_wide.to(dt)[:, ::2]
        assert not x_strided.is_contiguous() and x_strided.shape == (389, 65)
        m = UltraEMeans(d=65, k=23, nepochs=2.0, seed=13)
        c, a = m.fit(x_strided)
        assert c.shape == (23, 65) and torch.isfinite(c).all()
        assert a is not None and bool(((a >= 0) & (a < 23)).all())
        dq, lq = m.assign(x_strided[:31])
        assert torch.isfinite(dq).all() and bool(((lq >= 0) & (lq < 23)).all())


@pytest.mark.parametrize("dev", ALL_DEVICES)
@pytest.mark.parametrize("mode", ["zeros", "constant", "huge_1e25", "tiny_1e_25", "fp16_near_max"])
def test_degenerate_and_extreme_scale_inputs(dev: str, mode: str) -> None:
    """Verify `zeros`, `constant`, `1e25`, `1e-25`, and `fp16_near_max` produce `0` `NaN`/`Inf`."""
    if mode == "fp16_near_max" and dev == "cpu":
        pytest.skip("FP16 Tensor-Core path requires CUDA")

    device = torch.device(dev)
    gen = torch.Generator(device=device).manual_seed(808)
    n, d, k = 512, 32, 16
    comp_dtype = None
    if mode == "zeros":
        x = torch.zeros((n, d), dtype=torch.float32, device=device)
    elif mode == "constant":
        x = torch.full((n, d), -1234.5, dtype=torch.float32, device=device)
    elif mode == "huge_1e25":
        x = torch.randn((n, d), generator=gen, device=device, dtype=torch.float32) * 1e25
    elif mode == "tiny_1e_25":
        x = torch.randn((n, d), generator=gen, device=device, dtype=torch.float32) * 1e-25
    else:
        x = torch.zeros((n, d), dtype=torch.float32, device=device)
        x[:256] = -60_000.0
        x[256:] = 60_000.0
        comp_dtype = torch.float16

    for arm in ("ultra_emeans", "ultra_emeans_wide"):
        m = build_ultra_arm(arm, d=d, k=k, nepochs=2.0, seed=5, compute_dtype=comp_dtype)
        c, a = m.fit(x)
        assert torch.isfinite(c).all(), f"Non-finite centroids in {arm} on {mode}"
        assert a is not None and bool(((a >= 0) & (a < k)).all())
        for st in m.iteration_stats:
            assert math.isfinite(st.moving_mse) and math.isfinite(st.imbalance_factor)
        dq, lq = m.assign(x[:32])
        assert torch.isfinite(dq).all() and bool(((lq >= 0) & (lq < k)).all())
        if mode == "zeros":
            assert exact_mse_fp64(x, c, device=device) < 1e-10
        elif mode == "constant":
            torch.testing.assert_close(c, torch.full_like(c, -1234.5), atol=1e-2, rtol=1e-5)


@pytest.mark.parametrize("dev", ALL_DEVICES)
def test_deterministic_reproducibility(dev: str) -> None:
    """Verify exact bit reproducibility under `torch.use_deterministic_algorithms(True)`."""
    rng = np.random.default_rng(55)
    x_t = torch.from_numpy(rng.standard_normal((1536, 32), dtype=np.float32)).to(dev)

    prev_det = torch.are_deterministic_algorithms_enabled()
    prev_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        c1, a1 = UltraEMeans(d=32, k=48, nepochs=2.0, seed=777, batch_size=256).fit(x_t)
        c2, a2 = UltraEMeans(d=32, k=48, nepochs=2.0, seed=777, batch_size=256).fit(x_t)
    finally:
        torch.use_deterministic_algorithms(prev_det, warn_only=prev_warn)

    assert torch.equal(a1, a2)
    torch.testing.assert_close(c1, c2, rtol=0.0, atol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA latency benchmark")
def test_gpu_production_shape_warm_latency() -> None:
    """Verify warm GPU latency on `(n=200k, d=128, k=2048, T=2.0)` stays blazingly fast (< 35 ms)."""
    dev = torch.device("cuda:0")
    gen = torch.Generator(device=dev).manual_seed(123)
    x = torch.randn((200_000, 128), generator=gen, device=dev, dtype=torch.float32)

    # Untimed warmup
    _ = UltraEMeans(d=128, k=2048, nepochs=2.0, wide_step_mult=2, seed=1).fit(x, capture_assignments=False)
    torch.cuda.synchronize(dev)

    times = []
    for s in (10, 11, 12):
        m = UltraEMeans(d=128, k=2048, nepochs=2.0, wide_step_mult=2, seed=s)
        m.fit(x, capture_assignments=False)
        times.append(m.total_time_sec)
    assert float(np.median(times)) < 0.035, f"Expected warm total latency < 35 ms, got {1000*np.median(times):.2f} ms"
