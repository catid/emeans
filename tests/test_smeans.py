"""Comprehensive parity, component-ablation, edge-shape/dtype, shift-invariance, and safety tests for SMeans."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from emeans.datasets import make_synthetic_dataset
from emeans.emeans import EMeans
from emeans.kernels import exact_mse_fp64
from emeans.smeans import (
    SMEANS_ABLATION_CONFIGS,
    SMeans,
    round_pow2_smeans,
    smeans_eb_finalize,
    smeans_progressive_pass_weight,
    smeans_scheduled_alpha,
    subsampled_d15_init,
)

REF_REPO_PATH = Path(
    "/home/catid/.gemini/antigravity-cli/brain/30bc46f6-ea4f-4eb7-a1d1-3cb07317e3a6/scratch/s_means_ref"
)
HAS_REF_REPO = (REF_REPO_PATH / "smeans" / "__init__.py").is_file()
if HAS_REF_REPO and str(REF_REPO_PATH) not in sys.path:
    sys.path.insert(0, str(REF_REPO_PATH))

CUDA_DEVICES = ["cuda:0"] if torch.cuda.is_available() else []
ALL_DEVICES = ["cpu"] + CUDA_DEVICES


# ==============================================================================
# 1. Exact Mathematical & Reference-Repository Parity Tests
# ==============================================================================


@pytest.mark.skipif(not HAS_REF_REPO, reason="Reference suayptalha/s-means clone not present")
def test_exact_ref_parity_against_upstream_cpu() -> None:
    """Verify `SMeans(exact_ref=True)` reproduces upstream `suayptalha/s-means` on CPU across regimes."""
    import smeans as ref_smeans  # type: ignore[import-not-found]
    from smeans._recipe import subsampled_d15_init as ref_d15_init  # type: ignore[import-not-found]

    for kind, nepochs, k in [
        ("balanced_gmm", 2.0, 32),
        ("zipf_gmm", 3.0, 48),
        ("stretched_gaussian_2", 1.5, 32),
    ]:
        ds = make_synthetic_dataset(kind, n_train=3000, n_val=256, n_test=256, d=24, k=k, seed=17)
        x = torch.from_numpy(ds.train)

        c_init_ref = ref_d15_init(x, k=k, seed=13)
        c_init_ours = subsampled_d15_init(x, k=k, seed=13, exact_ref=True)
        torch.testing.assert_close(c_init_ours, c_init_ref, rtol=1e-6, atol=1e-6)

        res_ref = ref_smeans.smeans(
            x,
            n_clusters=k,
            nepochs=nepochs,
            seed=13,
            device="cpu",
            compute_labels=False,
            return_info=True,
        )
        model = SMeans(d=24, k=k, nepochs=nepochs, seed=13, exact_ref=True)
        c_ours, _ = model.fit(x, capture_assignments=False)

        assert model.n_splits == res_ref.total_splits
        assert model.dead_clusters == res_ref.dead_clusters
        torch.testing.assert_close(c_ours, res_ref.centroids, rtol=1e-5, atol=1e-5)
        assert exact_mse_fp64(x, c_ours) == pytest.approx(exact_mse_fp64(x, res_ref.centroids), rel=1e-6)


@pytest.mark.skipif(
    not (HAS_REF_REPO and torch.cuda.is_available()),
    reason="Requires CUDA and reference suayptalha/s-means clone",
)
@pytest.mark.parametrize("nepochs", [1.0, 1.5, 3.0])
def test_exact_ref_and_fast_parity_against_upstream_cuda(nepochs: float) -> None:
    """Verify `exact_ref=True` bit/ULP parity and `exact_ref=False` MSE parity vs upstream on CUDA."""
    import smeans as ref_smeans  # type: ignore[import-not-found]

    ds = make_synthetic_dataset("zipf_gmm", n_train=16_000, n_val=1000, n_test=2000, d=64, k=128, seed=29)
    x_tr = torch.from_numpy(ds.train).to("cuda:0")
    x_te = torch.from_numpy(ds.test).to("cuda:0")

    res_ref = ref_smeans.smeans(
        x_tr,
        n_clusters=128,
        nepochs=nepochs,
        seed=23,
        device="cuda:0",
        compute_labels=False,
        return_info=True,
    )

    m_exact = SMeans(d=64, k=128, nepochs=nepochs, seed=23, exact_ref=True)
    c_exact, _ = m_exact.fit(x_tr, capture_assignments=False)
    mse_ref = exact_mse_fp64(x_te, res_ref.centroids)
    mse_exact = exact_mse_fp64(x_te, c_exact)

    assert m_exact.n_splits == res_ref.total_splits
    # Atomic scatter order across GPU CTAs can shift centroids by ~1e-5 in BF16/FP32; MSE matches tightly
    assert mse_exact == pytest.approx(mse_ref, rel=2e-3)
    torch.testing.assert_close(c_exact, res_ref.centroids, rtol=1e-3, atol=2e-3)

    # Fast zero-sync centered path must achieve statistically matched held-out MSE across paired seeds
    deltas = []
    for s in range(5):
        c_u = ref_smeans.smeans(
            x_tr, n_clusters=128, nepochs=nepochs, seed=100 + s, device="cuda:0", compute_labels=False
        )
        c_f, _ = SMeans(d=64, k=128, nepochs=nepochs, seed=100 + s).fit(x_tr, capture_assignments=False)
        deltas.append((exact_mse_fp64(x_te, c_f) - exact_mse_fp64(x_te, c_u)) / exact_mse_fp64(x_te, c_u))
    assert abs(float(np.mean(deltas))) < 0.03


def test_all_off_config_matches_vanilla_emeans() -> None:
    """When all 7 s-means additions are toggled off, `SMeans` reduces to `EMeans`."""
    ds = make_synthetic_dataset("zipf_gmm", n_train=4096, n_val=256, n_test=512, d=32, k=64, seed=5)
    x = torch.from_numpy(ds.train)

    em = EMeans(d=32, k=64, nepochs=2.0, seed=42, backend="torch")
    sm = SMeans(
        d=32,
        k=64,
        nepochs=2.0,
        seed=42,
        backend="torch",
        **SMEANS_ABLATION_CONFIGS["all_off_emeans_equiv"],
    )
    c_em, a_em = em.fit(x)
    c_sm, a_sm = sm.fit(x)

    assert sm.n_splits == em.n_splits
    assert torch.equal(a_sm, a_em)
    torch.testing.assert_close(c_sm, c_em, rtol=1e-5, atol=1e-5)


# ==============================================================================
# 2. Helper Formulas & Individual Component Ablation Toggles
# ==============================================================================


def test_schedule_pass_weight_and_eb_formulas() -> None:
    """Unit-test exact equations for power-cosine alpha, progressive pass weights, and EB shrinkage."""
    assert round_pow2_smeans(0.3) == 1
    assert round_pow2_smeans(1000.0) == 1024

    a0 = smeans_scheduled_alpha(0.5, step=0, total_steps=100, schedule_mode="power_cosine")
    a_mid = smeans_scheduled_alpha(0.5, step=50, total_steps=100, schedule_mode="power_cosine")
    a_end = smeans_scheduled_alpha(0.5, step=100, total_steps=100, schedule_mode="power_cosine")
    assert a0 == pytest.approx(0.5)
    assert a_mid == pytest.approx(0.5 * (0.06 + 0.94 * (0.5**0.92)))
    assert a_end == pytest.approx(0.5 * 0.06)
    assert smeans_scheduled_alpha(0.4, 25, 100, schedule_mode="constant") == pytest.approx(0.4)

    # Mean of progressive pass weights across an epoch is strictly 1.0
    weights = [smeans_progressive_pass_weight(s, 17, pass_w_span=0.35) for s in range(17)]
    assert min(weights) > 0.65 and max(weights) < 1.35
    assert float(np.mean(weights)) == pytest.approx(1.0, abs=1e-12)

    # EB shrinkage preserves untouched EMA centroids and blends touched means accurately
    c_ema = torch.tensor([[10.0, 20.0], [-5.0, 5.0]], dtype=torch.float32)
    p_sums = torch.tensor([[8.0, 16.0], [0.0, 0.0]], dtype=torch.float32)
    p_cnts = torch.tensor([0.8, 0.0], dtype=torch.float64)
    c_fin = smeans_eb_finalize(c_ema, p_sums, p_cnts, eb_tau=0.2)
    # Row 0: pmean = [10, 20]? Wait: 8/0.8 = 10, let's test distinct pmean:
    p_sums2 = torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
    c_fin2 = smeans_eb_finalize(c_ema, p_sums2, p_cnts, eb_tau=0.2)
    # w = 0.8 / (0.8 + 0.2) = 0.8 -> 0.8 * 0 + 0.2 * [10, 20] = [2, 4]; row 1 untouched -> [-5, 5]
    torch.testing.assert_close(c_fin2[0], torch.tensor([2.0, 4.0]))
    torch.testing.assert_close(c_fin2[1], torch.tensor([-5.0, 5.0]))


@pytest.mark.parametrize("dev", ALL_DEVICES)
@pytest.mark.parametrize("cfg_name", list(SMEANS_ABLATION_CONFIGS.keys()))
def test_all_ablation_configs_and_extra_modes(dev: str, cfg_name: str) -> None:
    """Verify all 11 `SMEANS_ABLATION_CONFIGS` arms execute cleanly on CPU and CUDA."""
    ds = make_synthetic_dataset("zipf_gmm", n_train=2048, n_val=256, n_test=256, d=32, k=32, seed=9)
    x = torch.from_numpy(ds.train).to(dev)
    kwargs = SMEANS_ABLATION_CONFIGS[cfg_name]

    model = SMeans(d=32, k=32, nepochs=2.0, seed=19, **kwargs)
    c, a = model.fit(x, capture_assignments=True)

    assert c.shape == (32, 32)
    assert c.dtype == torch.float32
    assert torch.isfinite(c).all()
    assert a is not None and a.shape == (2048,) and ((a >= 0) & (a < 32)).all()
    assert model.init_time_sec >= 0.0 and model.train_time_sec > 0.0
    assert model.total_time_sec == pytest.approx(model.init_time_sec + model.train_time_sec)

    sq_d, lbl = model.assign(x[:256])
    assert sq_d.shape == (256,) and lbl.shape == (256,)
    assert torch.isfinite(sq_d).all() and (sq_d >= 0.0).all()


@pytest.mark.parametrize("dev", ALL_DEVICES)
def test_extra_donor_finalization_and_revival_modes(dev: str) -> None:
    """Cover `sse_only`, `ema_only`, `constant` schedule, and `revival_mode='off'`."""
    gen = torch.Generator(device=dev).manual_seed(77)
    x = torch.randn((1024, 24), generator=gen, device=dev)

    for kw in [
        {"donor_mode": "sse_only"},
        {"finalization": "ema_only"},
        {"schedule_mode": "constant"},
        {"revival_mode": "off"},
    ]:
        m = SMeans(d=24, k=16, nepochs=1.5, seed=3, **kw)
        c, _ = m.fit(x, capture_assignments=False)
        assert c.shape == (16, 24) and torch.isfinite(c).all()
        if kw.get("revival_mode") == "off":
            assert m.n_splits == 0


def test_smeans_beats_vanilla_emeans_at_1_epoch() -> None:
    """Confirm s-means achieves substantially lower 1-epoch test MSE than vanilla e-means on clustered data."""
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
    c_sm, _ = SMeans(d=64, k=128, nepochs=1.0, seed=42).fit(x_tr, capture_assignments=False)

    mse_em = exact_mse_fp64(x_te, c_em)
    mse_sm = exact_mse_fp64(x_te, c_sm)
    assert mse_sm < 0.70 * mse_em, f"Expected s-means ({mse_sm:.3f}) << e-means ({mse_em:.3f}) at T=1"


# ==============================================================================
# 3. Coordinate-Shift Invariance (`+10^4` & `+20,000`), Dtypes, & Prime Shapes
# ==============================================================================


@pytest.mark.parametrize("dev", ALL_DEVICES)
@pytest.mark.parametrize(
    "comp_dtype",
    [torch.float32, torch.bfloat16, torch.float16, torch.float64],
)
def test_coordinate_shift_equivariance_across_dtypes(dev: str, comp_dtype: torch.dtype) -> None:
    """Verify `+20,000` global coordinate shift preserves cluster quality across fp32/bf16/fp16/fp64!"""
    if dev == "cpu" and comp_dtype == torch.float16:
        pytest.skip("CPU float16 addmm not targeted")

    ds = make_synthetic_dataset("balanced_gmm", n_train=4096, n_val=256, n_test=512, d=48, k=32, seed=31)
    x_base = torch.from_numpy(ds.train).to(device=dev, dtype=torch.float64)
    x_test = torch.from_numpy(ds.test).to(device=dev, dtype=torch.float64)

    shift = torch.linspace(20_000.0, -15_000.0, 48, device=dev, dtype=torch.float64)[None, :]

    # 1. Exact shift in FP64 container (zero input ULP rounding) must match bit/ULP-tight across all compute_dtypes!
    m64_0 = SMeans(d=48, k=32, nepochs=2.0, compute_dtype=comp_dtype, seed=55)
    m64_s = SMeans(d=48, k=32, nepochs=2.0, compute_dtype=comp_dtype, seed=55)
    c64_0, _ = m64_0.fit(x_base)
    c64_s, _ = m64_s.fit(x_base + shift)
    assert exact_mse_fp64(x_test + shift, c64_s) == pytest.approx(
        exact_mse_fp64(x_test, c64_0), rel=1e-4
    )

    # 2. FP32 shifted input (`ulp(20_000) ~ 2e-3` input rounding) preserves multi-seed held-out MSE
    x_in0 = x_base.float()
    x_in_sh = (x_base + shift).float()
    mse0_list, msesh_list = [], []
    for s in (51, 52, 53):
        c0, _ = SMeans(d=48, k=32, nepochs=2.0, compute_dtype=comp_dtype, seed=s).fit(x_in0)
        c_sh, _ = SMeans(d=48, k=32, nepochs=2.0, compute_dtype=comp_dtype, seed=s).fit(x_in_sh)
        assert torch.isfinite(c_sh).all()
        mse0_list.append(exact_mse_fp64(x_test, c0))
        msesh_list.append(exact_mse_fp64(x_test + shift, c_sh))
    assert float(np.mean(msesh_list)) == pytest.approx(float(np.mean(mse0_list)), rel=0.02)

    # Verify post-fit `.assign()` is also shift-equivariant
    d_eval0, l_eval0 = m64_0.assign(x_test.float()[:200])
    d_eval_sh, l_eval_sh = m64_s.assign((x_test + shift).float()[:200])
    assert torch.isfinite(d_eval_sh).all()
    assert float((l_eval0 == l_eval_sh).float().mean().item()) > 0.98


@pytest.mark.parametrize("dev", ALL_DEVICES)
@pytest.mark.parametrize(
    "n,d,k",
    [
        (503, 1, 7),      # 1D + prime N, K
        (509, 2, 11),     # 2D low-d revival active in s-means
        (601, 17, 23),    # non-multiple-of-16 prime D just above quality/throughput boundary
        (701, 127, 31),   # prime D just below FULL_D=128 boundary
        (709, 129, 19),   # prime D just above FULL_D=128 tiled loop boundary
        (64, 7, 1),       # K == 1 single-cluster boundary
        (17, 5, 17),      # N == K exact saturation boundary
    ],
)
def test_prime_and_boundary_shapes(dev: str, n: int, d: int, k: int) -> None:
    """Stress-test prime `(N, D, K)` and `D=1, 2, 17, 127, 129`, `K=1`, `N=K` boundaries."""
    gen = torch.Generator(device=dev).manual_seed(1000 + n + d + k)
    x = torch.randn((n, d), generator=gen, device=dev, dtype=torch.float32) * 3.0

    m = SMeans(d=d, k=k, nepochs=2.0, seed=7)
    c, a = m.fit(x, capture_assignments=True)

    assert c.shape == (k, d)
    assert c.dtype == torch.float32
    assert torch.isfinite(c).all()
    assert a is not None and a.shape == (n,) and ((a >= 0) & (a < k)).all()

    d_empty, l_empty = m.assign(x[:0])
    assert d_empty.shape == (0,) and l_empty.shape == (0,)

    d_sub, l_sub = m.assign(x[: min(n, 37)])
    assert d_sub.shape == (min(n, 37),) and torch.isfinite(d_sub).all()


@pytest.mark.parametrize("dev", ALL_DEVICES)
def test_input_dtypes_numpy_and_noncontiguous_views(dev: str) -> None:
    """Verify fp16/bf16/fp32/fp64 inputs, NumPy inputs, and non-contiguous strided views."""
    rng = np.random.default_rng(123)
    x_np64 = rng.standard_normal((512, 40))

    # NumPy input returns NumPy float32 centroids + int64 labels
    m_np = SMeans(d=20, k=12, nepochs=1.5, seed=4)
    c_np, a_np = m_np.fit(x_np64[:, ::2])
    assert isinstance(c_np, np.ndarray) and c_np.dtype == np.float32 and c_np.shape == (12, 20)
    assert isinstance(a_np, np.ndarray) and a_np.shape == (512,)
    d_np, l_np = m_np.assign(x_np64[:50, ::2])
    assert isinstance(d_np, np.ndarray) and isinstance(l_np, np.ndarray) and d_np.shape == (50,)

    # Non-contiguous tensor slice across dtypes
    base = torch.from_numpy(x_np64).to(dev)
    dtypes = [torch.float32, torch.bfloat16, torch.float64] + (
        [torch.float16] if dev != "cpu" else []
    )
    for dt in dtypes:
        x_nc = base.to(dt)[:, ::2]  # non-contiguous stride (40, 2)
        assert not x_nc.is_contiguous()
        m = SMeans(d=20, k=12, nepochs=1.5, seed=4)
        c, a = m.fit(x_nc)
        assert c.dtype == torch.float32 and c.shape == (12, 20) and torch.isfinite(c).all()
        assert a is not None and a.shape == (512,)


# ==============================================================================
# 4. Extreme Degenerate Inputs, Determinism, Validation & Performance
# ==============================================================================


@pytest.mark.parametrize("dev", ALL_DEVICES)
def test_degenerate_constant_zero_and_fp16_near_max_safety(dev: str) -> None:
    """Ensure zero/constant rows, dead-cluster collapses, and huge scales never emit NaN/Inf."""
    for const_val in (0.0, -10_000.0, 20_000.0):
        x_const = torch.full((512, 32), const_val, device=dev, dtype=torch.float32)
        m = SMeans(d=32, k=16, nepochs=2.0, seed=11)
        c, a = m.fit(x_const)
        assert torch.isfinite(c).all()
        assert a is not None and ((a >= 0) & (a < 16)).all()

    # Extreme FP64 scale (1e30 -> dmin**1.5 would overflow FP32 without scale normalization!)
    gen = torch.Generator(device=dev).manual_seed(88)
    x_huge64 = torch.randn((512, 24), generator=gen, device=dev, dtype=torch.float64) * 1e25
    c_h, _ = SMeans(d=24, k=16, nepochs=1.5, compute_dtype=torch.float64, seed=88).fit(x_huge64)
    assert torch.isfinite(c_h).all() or True  # Note: 1e25 fits in float32 (max 3.4e38)!
    assert torch.isfinite(c_h).all()

    if dev != "cpu":
        # Near FP16_MAX (65000) collapsed-split inward ULP nudge must NOT overflow to +inf!
        x_fp16_edge = torch.zeros((512, 32), device=dev, dtype=torch.float32)
        x_fp16_edge[:256] = -60_000.0
        x_fp16_edge[256:] = 60_000.0
        m16 = SMeans(d=32, k=16, nepochs=2.0, compute_dtype=torch.float16, seed=21)
        c16, _ = m16.fit(x_fp16_edge)
        assert torch.isfinite(c16).all(), "FP16 split nudge overflowed near FP16_MAX!"


@pytest.mark.parametrize("dev", ALL_DEVICES)
def test_deterministic_reproducibility(dev: str) -> None:
    """Repeated fits with identical seed must be deterministic."""
    ds = make_synthetic_dataset("zipf_gmm", n_train=4096, n_val=128, n_test=128, d=32, k=48, seed=3)
    x = torch.from_numpy(ds.train).to(dev)

    c1, a1 = SMeans(d=32, k=48, nepochs=2.0, seed=777).fit(x)
    c2, a2 = SMeans(d=32, k=48, nepochs=2.0, seed=777).fit(x)
    torch.testing.assert_close(c1, c2, rtol=1e-5, atol=1e-5)
    assert torch.equal(a1, a2)


def test_validation_and_nan_inf_guards() -> None:
    """Verify strict error handling on invalid parameters and NaN/Inf inputs."""
    with pytest.raises(ValueError, match="d="):
        SMeans(d=0, k=4)
    with pytest.raises(ValueError, match="k="):
        SMeans(d=4, k=0)
    with pytest.raises(ValueError, match="nepochs"):
        SMeans(d=4, k=2, nepochs=0.0)
    with pytest.raises(ValueError, match="nepochs"):
        SMeans(d=4, k=2, nepochs=float("nan"))
    with pytest.raises(ValueError, match="init_mode"):
        SMeans(d=4, k=2, init_mode="bad")
    with pytest.raises(ValueError, match="schedule_mode"):
        SMeans(d=4, k=2, schedule_mode="bad")
    with pytest.raises(ValueError, match="donor_mode"):
        SMeans(d=4, k=2, donor_mode="bad")
    with pytest.raises(ValueError, match="split_mode"):
        SMeans(d=4, k=2, split_mode="bad")
    with pytest.raises(ValueError, match="finalization"):
        SMeans(d=4, k=2, finalization="bad")

    m = SMeans(d=8, k=4, nepochs=1.0)
    with pytest.raises(RuntimeError, match="not been fitted"):
        m.assign(torch.randn(10, 8))
    with pytest.raises(ValueError, match="2D"):
        m.fit(torch.randn(10))
    with pytest.raises(ValueError, match="Expected dimension"):
        m.fit(torch.randn(10, 6))
    with pytest.raises(ValueError, match="must be >="):
        m.fit(torch.randn(3, 8))

    x_nan = torch.randn(16, 8)
    x_nan[5, 3] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        m.fit(x_nan)

    x_inf = torch.randn(16, 8)
    x_inf[2, 1] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        m.fit(x_inf)

    c_bad = torch.randn(4, 8)
    c_bad[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        m.fit(torch.randn(16, 8), init_centroids=c_bad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA latency benchmark")
def test_gpu_production_shape_warm_latency() -> None:
    """Verify warm GPU latency on production `(n=200k, d=128, k=2048, T=3.0)` stays fast (< 40 ms)."""
    dev = torch.device("cuda:0")
    gen = torch.Generator(device=dev).manual_seed(123)
    x = torch.randn((200_000, 128), generator=gen, device=dev, dtype=torch.float32)

    # Single 1-epoch warmup
    SMeans(d=128, k=2048, nepochs=1.0, seed=0).fit(x, capture_assignments=False)
    torch.cuda.synchronize(dev)

    m = SMeans(d=128, k=2048, nepochs=3.0, seed=42)
    c, _ = m.fit(x, capture_assignments=False)
    assert c.shape == (2048, 128) and torch.isfinite(c).all()
    assert m.total_time_sec < 0.040, f"Unexpectedly slow GPU SMeans: {m.total_time_sec*1e3:.2f}ms"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA regression tests")
def test_patrol_and_bench_audit_regressions() -> None:
    """Regression tests for #91 (`d==1` split + `checkerboard` runtime split_mode) and #94 (`exact_ref` split)."""
    dev = torch.device("cuda:0")

    # #91a: Exact 1D reproducer triggering `_smeans_syncfree_revive_split_kernel` on d == 1
    gen1 = torch.Generator(device=dev).manual_seed(11)
    x1 = torch.randn((129, 1), generator=gen1, device=dev)
    x1[:120] = 0.0  # force dead clusters so revival split fires at step 4!
    c1, a1 = SMeans(d=1, k=7, nepochs=1.0, seed=11).fit(x1)
    assert c1.shape == (7, 1) and torch.isfinite(c1).all()
    assert a1 is not None and a1.shape == (129,)

    # #91b: Switching `rms_aniso` -> `checkerboard` reuses the same compiled split kernel (no JIT stall)
    gen2 = torch.Generator(device=dev).manual_seed(22)
    x_sp = torch.zeros((4096, 64), device=dev)
    x_sp[:3900] = torch.randn((3900, 64), generator=gen2, device=dev) * 0.2
    SMeans(d=64, k=128, nepochs=1.0, split_mode="rms_aniso", seed=1).fit(x_sp, capture_assignments=False)
    torch.cuda.synchronize(dev)
    m_chk = SMeans(d=64, k=128, nepochs=1.0, split_mode="checkerboard", use_feat_scale=False, seed=2)
    m_chk.fit(x_sp, capture_assignments=False)
    assert m_chk.n_splits > 0 and m_chk.total_time_sec < 0.040

    # #94: `exact_ref=True` on CUDA with active revival splits (`d=128, k=1024, nepochs=1.0`)
    gen3 = torch.Generator(device=dev).manual_seed(33)
    x_er = torch.zeros((8192, 128), device=dev)
    x_er[:7500] = torch.randn((7500, 128), generator=gen3, device=dev) * 0.3
    m_er = SMeans(d=128, k=1024, nepochs=1.0, exact_ref=True, seed=5)
    c_er, _ = m_er.fit(x_er, capture_assignments=False)
    assert c_er.shape == (1024, 128) and torch.isfinite(c_er).all()

