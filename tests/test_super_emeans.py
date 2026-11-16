"""Comprehensive unit, parity, ablation-toggle, shift-equivariance, rare-mode, and adversarial tests for `super_emeans.py`."""

from __future__ import annotations

import math
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pytest
import torch

from emeans.algo_extensions import empirical_bayes_finalize, update_oja_principal_axes
from emeans.emeans_pp import EMeansPP, _nearest_other_centroid_sq_dists
from emeans.kernels import exact_mse_fp64
from emeans.noise_tracks import top3_represented_assign_centered
from emeans.super_emeans import (
    HYBRID_ABLATION_ARMS,
    HybridSEMeans,
    SuperEMeans,
    _all_pairs_centered_nn,
    build_hybrid_arm,
    eb_james_stein_finalize_gpu,
    fast_top3_centered_assign,
    incremental_refresh_top3_and_nn,
    subsampled_d15_init_centered,
    super_scheduled_alpha,
    twin_safe_huygens_protection,
)

CUDA_DEVICES = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]

EXPECTED_29_ARMS = {
    "emeans_pp_base",
    "super_emeans",
    "super_emeans_wide",
    "smeans_deconfounded",
    "pp_plus_c1_d15_raw",
    "pp_plus_c1_d15_polished",
    "pp_plus_c2_power_cos",
    "pp_plus_c3_revive_stop",
    "pp_plus_c4_sse_usage_donor",
    "pp_plus_c5_rms_aniso_split",
    "pp_plus_c6_pass_weight",
    "pp_plus_c7_eb_ema",
    "super_plus_c1_d15_init",
    "super_plus_c2_power_cos",
    "super_plus_c3_revive_stop",
    "super_plus_c4_sse_usage_donor",
    "super_plus_c5_rms_aniso_split",
    "super_plus_c5b_oja_rms_split",
    "super_plus_c6_pass_weight",
    "super_plus_c7_eb_ema",
    "smeans_plus_p1_coreset_voronoi",
    "smeans_plus_p2_huygens_shield",
    "smeans_plus_p3_oja_split",
    "smeans_plus_p3_oja008",
    "smeans_plus_p4_eb_js",
    "smeans_plus_p5_head_edit",
    "smeans_plus_p6_wide_step",
    "hybrid_super_smeans",
    "hybrid_super_smeans_wide",
}


# ---------------------------------------------------------------------------
# 1. Registry, Validation & Estimator API Contract Tests
# ---------------------------------------------------------------------------


def test_hybrid_ablation_arms_registry_and_validation() -> None:
    assert set(HYBRID_ABLATION_ARMS.keys()) == EXPECTED_29_ARMS
    for arm_name in EXPECTED_29_ARMS:
        model = build_hybrid_arm(arm_name, d=16, k=8, nepochs=2.0, seed=7)
        assert isinstance(model, HybridSEMeans)

    with pytest.raises(KeyError):
        build_hybrid_arm("nonexistent_arm", d=16, k=8)
    with pytest.raises(ValueError):
        HybridSEMeans(d=16, k=8, init_mode="bad_init")
    with pytest.raises(ValueError):
        HybridSEMeans(d=16, k=8, schedule_mode="bad_sched")
    with pytest.raises(ValueError):
        HybridSEMeans(d=16, k=8, split_mode="bad_split")
    with pytest.raises(ValueError):
        HybridSEMeans(d=16, k=8, finalization="bad_fin")
    with pytest.raises(ValueError):
        HybridSEMeans(d=16, k=8, wide_step_mult=0)


@pytest.mark.parametrize("dev", CUDA_DEVICES)
def test_numpy_and_tensor_io_and_assign_order(dev: str) -> None:
    rng = np.random.default_rng(42)
    x_np = rng.standard_normal((512, 24), dtype=np.float32)

    m_np = SuperEMeans(d=24, k=16, nepochs=2.0, seed=11)
    with pytest.raises(RuntimeError):
        m_np.assign(x_np[:10])

    c_np, a_np = m_np.fit(x_np, capture_assignments=True)
    assert isinstance(c_np, np.ndarray) and c_np.shape == (16, 24)
    assert isinstance(a_np, np.ndarray) and a_np.shape == (512,)
    d_np, l_np = m_np.assign(x_np[:32])
    assert isinstance(d_np, np.ndarray) and d_np.dtype == np.float32 and d_np.shape == (32,)
    assert isinstance(l_np, np.ndarray) and l_np.dtype == np.int64 and l_np.shape == (32,)
    assert np.all(d_np >= 0.0) and np.all((l_np >= 0) & (l_np < 16))

    x_t = torch.from_numpy(x_np).to(dev)
    m_t = SuperEMeans(d=24, k=16, nepochs=2.0, seed=11)
    c_t, a_t = m_t.fit(x_t, capture_assignments=True)
    assert isinstance(c_t, torch.Tensor) and c_t.device.type == torch.device(dev).type
    assert isinstance(a_t, torch.Tensor) and a_t.dtype == torch.int64
    d_t, l_t = m_t.assign(x_t[:32])
    assert d_t.dtype == torch.float32 and l_t.dtype == torch.int64
    assert m_t.init_time_sec >= 0.0 and m_t.train_time_sec >= 0.0
    assert math.isclose(m_t.total_time_sec, m_t.init_time_sec + m_t.train_time_sec, rel_tol=1e-12)


def test_schedule_modes_exact_values() -> None:
    # At frac=0 -> alpha0; at frac=1 -> floor * alpha0
    assert math.isclose(super_scheduled_alpha(0.8, 0.0, "emeans_cosine"), 0.8, rel_tol=1e-12)
    assert math.isclose(super_scheduled_alpha(0.8, 1.0, "emeans_cosine"), 0.08, rel_tol=1e-12)
    assert math.isclose(super_scheduled_alpha(0.8, 0.0, "power_cosine"), 0.8, rel_tol=1e-12)
    assert math.isclose(super_scheduled_alpha(0.8, 1.0, "power_cosine"), 0.8 * 0.06, rel_tol=1e-12)
    assert math.isclose(super_scheduled_alpha(0.8, 0.5, "constant"), 0.8, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# 2. Exact Kernel, Incremental Top-3/NN, EB-JS Kish n_eff, & EMeansPP Parity
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton top-3 & incremental tests")
@pytest.mark.parametrize("B,K,D", [(129, 37, 17), (513, 193, 128), (257, 67, 259)])
def test_fast_top3_and_incremental_refresh_exact_parity(B: int, K: int, D: int) -> None:
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(101 + B + K + D)
    x = torch.randn((B, D), generator=gen, device=dev, dtype=torch.float32)
    c = torch.randn((K, D), generator=gen, device=dev, dtype=torch.float32)

    # 1. Exact IEEE FP32 + +20,000 coordinate shift parity against reference
    for shift in (0.0, 20_000.0):
        xs = x + shift
        cs = c + shift
        td_fast, ti_fast, orig, xc, xn, cc = fast_top3_centered_assign(
            xs, cs, torch.float32, already_centered=False
        )
        td_ref, ti_ref, _ = top3_represented_assign_centered(xs, cs, torch.float32)
        assert torch.equal(ti_fast, ti_ref)
        torch.testing.assert_close(td_fast, td_ref, rtol=1e-5, atol=1e-4)

    # 2. BF16 Tensor-Core `already_centered=True` parity on centered inputs
    xc_in = (x - x.mean(dim=0, keepdim=True)).contiguous()
    cc_in = (c - x.mean(dim=0, keepdim=True)).contiguous()
    td_tc, ti_tc, _, xc_b, xn_b, cc_b = fast_top3_centered_assign(
        xc_in, cc_in, torch.bfloat16, already_centered=True
    )
    # Direct BF16 represented distance reference
    xr = xc_in.to(torch.bfloat16).float()
    cr = cc_in.to(torch.bfloat16).float()
    d2_full = (xr.square().sum(1)[:, None] + cr.square().sum(1)[None, :] - 2.0 * (xr @ cr.t())).clamp_min(0.0)
    td_bf16_ref, ti_bf16_ref = torch.topk(d2_full, k=3, dim=1, largest=False, sorted=True)
    assert torch.equal(ti_tc, ti_bf16_ref)
    torch.testing.assert_close(td_tc, td_bf16_ref, rtol=1e-5, atol=1e-4)

    # 3. Exact multi-step `incremental_refresh_top3_and_nn` parity against full recomputation!
    cn_b = cc_b.square().sum(dim=1)
    # `_all_pairs_centered_nn` on `cc_b`
    d_nn2, nn_idx = _all_pairs_centered_nn(cc_b)
    c_mut = cc_b.clone()

    for edit_round in range(4):
        s_idx = (edit_round * 7 + 3) % K
        d_idx = (edit_round * 11 + 14) % K
        if d_idx == s_idx:
            d_idx = (d_idx + 1) % K
        new_s = torch.randn((D,), generator=gen, device=dev, dtype=torch.float32).to(torch.bfloat16).float()
        new_d = torch.randn((D,), generator=gen, device=dev, dtype=torch.float32).to(torch.bfloat16).float()
        c_mut[s_idx] = new_s
        c_mut[d_idx] = new_d

        incremental_refresh_top3_and_nn(
            xc=xc_b,
            xn=xn_b,
            cc=cc_b,
            cn=cn_b,
            top_d=td_tc,
            top_i=ti_tc,
            d_nn2=d_nn2,
            nn_idx=nn_idx,
            s_idx=s_idx,
            d_idx=d_idx,
            new_s_c=new_s,
            new_d_c=new_d,
        )

        # Full from-scratch reference check
        d2_after = (
            xc_b.square().sum(1)[:, None] + c_mut.square().sum(1)[None, :] - 2.0 * (xc_b @ c_mut.t())
        ).clamp_min(0.0)
        ref_td, ref_ti = torch.topk(d2_after, k=3, dim=1, largest=False, sorted=True)
        assert torch.equal(ti_tc, ref_ti), f"Top-3 index mismatch at round {edit_round}"
        torch.testing.assert_close(td_tc, ref_td, rtol=1e-5, atol=1e-4)

        ref_nn_d = _nearest_other_centroid_sq_dists(c_mut, torch.arange(K, device=dev))
        torch.testing.assert_close(d_nn2, ref_nn_d, rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("dev", CUDA_DEVICES)
def test_eb_james_stein_parity_and_kish_effective_counts(dev: str) -> None:
    gen = torch.Generator(device=dev).manual_seed(77)
    k, d = 64, 32
    closing = torch.randn((k, d), generator=gen, device=dev, dtype=torch.float32)
    slow = closing + 0.15 * torch.randn((k, d), generator=gen, device=dev, dtype=torch.float32)
    counts = torch.randint(0, 50, (k,), generator=gen, device=dev).double()
    sse = counts * (0.8 + 0.4 * torch.rand((k,), generator=gen, device=dev, dtype=torch.float64))

    # 1. Exact unweighted parity with `algo_extensions.empirical_bayes_finalize`
    c_gpu, lam_gpu, diag_gpu = eb_james_stein_finalize_gpu(
        closing_means=closing,
        anchor_centroids=slow,
        pass_counts=counts,
        pass_counts_sq=None,
        pass_sse=sse,
        d=d,
        kappa="auto",
    )
    c_ref, lam_ref, diag_ref = empirical_bayes_finalize(
        closing_means=closing,
        slow_centroids=slow,
        pass_counts=counts,
        pass_sse=sse,
        d=d,
        kappa="auto",
    )
    torch.testing.assert_close(lam_gpu, lam_ref, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(c_gpu, c_ref, rtol=1e-6, atol=1e-6)
    assert math.isclose(diag_gpu.mean_lambda, diag_ref.mean_lambda, rel_tol=1e-5)

    # 2. Kish effective sample size invariance under uniform weight rescaling `w_i = c`:
    # Scaling weights by a constant `c = 2.5` (`sum w = 2.5 n`, `sum w^2 = 6.25 n`, `sse_w = 2.5 sse`)
    # MUST yield the EXACT SAME `n_eff = n` and identical shrunk centroids!
    scale_w = 2.5
    c_kish, lam_kish, _ = eb_james_stein_finalize_gpu(
        closing_means=closing,
        anchor_centroids=slow,
        pass_counts=counts * scale_w,
        pass_counts_sq=counts * (scale_w * scale_w),
        pass_sse=sse * scale_w,
        d=d,
        kappa="auto",
    )
    torch.testing.assert_close(lam_kish, lam_gpu, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(c_kish, c_gpu, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("dev", CUDA_DEVICES)
def test_emeans_pp_anchor_parity(dev: str) -> None:
    rng = np.random.default_rng(29)
    centers = rng.standard_normal((32, 32)).astype(np.float32) * 3.0
    x_np = (centers[rng.integers(0, 32, size=2048)] + 0.3 * rng.standard_normal((2048, 32))).astype(np.float32)
    x_t = torch.from_numpy(x_np).to(dev)

    ref = EMeansPP(d=32, k=32, nepochs=3.0, batch_size=256, seed=123)
    hyb = build_hybrid_arm(
        "emeans_pp_base",
        d=32,
        k=32,
        nepochs=3.0,
        seed=123,
        batch_size=256,
        center_coordinates=False,
        use_blackwell_assign=False,
    )
    c_ref, _ = ref.fit(x_t)
    c_hyb, _ = hyb.fit(x_t)
    mse_ref = exact_mse_fp64(x_t, c_ref, device=dev)
    mse_hyb = exact_mse_fp64(x_t, c_hyb, device=dev)
    assert math.isclose(mse_hyb, mse_ref, rel_tol=5e-3)


# ---------------------------------------------------------------------------
# 3. Deterministic Reproducibility & Coordinate-Shift Equivariance (+10^4, +20,000)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dev", CUDA_DEVICES)
@pytest.mark.parametrize("arm_name", ["super_emeans", "super_emeans_wide", "hybrid_super_smeans", "smeans_deconfounded"])
def test_deterministic_reproducibility(dev: str, arm_name: str) -> None:
    rng = np.random.default_rng(55)
    x_t = torch.from_numpy(rng.standard_normal((1536, 32), dtype=np.float32)).to(dev)

    prev_det = torch.are_deterministic_algorithms_enabled()
    prev_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        m1 = build_hybrid_arm(arm_name, d=32, k=48, nepochs=2.0, seed=777, batch_size=256)
        m2 = build_hybrid_arm(arm_name, d=32, k=48, nepochs=2.0, seed=777, batch_size=256)
        c1, a1 = m1.fit(x_t)
        c2, a2 = m2.fit(x_t)
    finally:
        torch.use_deterministic_algorithms(prev_det, warn_only=prev_warn)

    assert torch.equal(a1, a2)
    torch.testing.assert_close(c1, c2, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("dev", CUDA_DEVICES)
@pytest.mark.parametrize("shift", [10_000.0, 20_000.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16, torch.float64])
@pytest.mark.parametrize("arm_name", ["super_emeans", "hybrid_super_smeans", "smeans_deconfounded"])
def test_coordinate_shift_invariance_and_represented_norms(
    dev: str, shift: float, dtype: torch.dtype, arm_name: str
) -> None:
    if dev == "cpu" and dtype in (torch.bfloat16, torch.float16):
        pytest.skip("Half precision assignment tested on CUDA")

    rng = np.random.default_rng(91)
    centers = rng.standard_normal((24, 32)).astype(np.float64) * 2.5
    idx = rng.integers(0, 24, size=1536)
    x64 = centers[idx] + 0.35 * rng.standard_normal((1536, 32))

    x_base = torch.from_numpy(x64.astype(np.float32)).to(dev)
    x_shift = torch.from_numpy((x64 + shift).astype(np.float64 if dtype == torch.float64 else np.float32)).to(dev)

    m_base = build_hybrid_arm(
        arm_name, d=32, k=24, nepochs=2.0, seed=2026, batch_size=256, compute_dtype=dtype
    )
    m_shift = build_hybrid_arm(
        arm_name, d=32, k=24, nepochs=2.0, seed=2026, batch_size=256, compute_dtype=dtype
    )
    c_base, _ = m_base.fit(x_base)
    c_shift, _ = m_shift.fit(x_shift)

    mse_base = exact_mse_fp64(x64, c_base.detach().cpu().numpy(), device="cpu")
    mse_shift = exact_mse_fp64(x64 + shift, c_shift.detach().cpu().numpy(), device="cpu")
    rel_diff = abs(mse_shift - mse_base) / max(mse_base, 1e-12)
    assert rel_diff < 5e-3, f"{arm_name} ({dtype}, shift={shift}): mse_base={mse_base:.6f} vs mse_shift={mse_shift:.6f}"

    # Appendix I.1 represented-norm self-assignment check on shifted centroids
    d_self, l_self = m_shift.assign(c_shift)
    assert torch.isfinite(d_self).all()
    assert float(d_self.max().item()) < 0.25


# ---------------------------------------------------------------------------
# 4. Rare-Mode Survival, Twin-Centroid Leader Election, & #93 Empty-Init Regression
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dev", CUDA_DEVICES)
def test_twin_centroid_rare_mode_leader_election(dev: str) -> None:
    """Verify `twin_safe_huygens_protection` keeps 1 guardian on a twin-seeded rare mode while recycling the duplicate!"""
    device = torch.device(dev)
    k, d = 16, 16
    centroids = torch.zeros((k, d), dtype=torch.float32, device=device)
    # 14 bulk centroids near origin
    gen = torch.Generator(device=device).manual_seed(12)
    centroids[:14] = torch.randn((14, d), generator=gen, device=device) * 0.5
    # 2 co-located twin centroids (14 and 15) on a distant isolated rare mode at +40.0!
    centroids[14] = 40.0
    centroids[15] = 40.0 + 1e-3

    usage = torch.full((k,), 20.0, dtype=torch.float32, device=device)
    # Both twins split the rare cluster's small mass (`0.8 + 0.6 = 1.4 < thresh = 1.8`)!
    usage[14] = 0.8
    usage[15] = 0.6
    thresh = usage.mean() * 0.1  # ~1.76
    low_mask = usage < thresh
    assert bool(low_mask[14].item()) and bool(low_mask[15].item())

    # Naive nearest-other-centroid distance is ~1.6e-5 between twins -> naive Huygens would protect NEITHER!
    naive_d_nn2 = _nearest_other_centroid_sq_dists(centroids, torch.tensor([14, 15], device=device))
    assert float(naive_d_nn2.max().item()) < 1e-3

    # Twin-safe Huygens elects centroid 14 (higher usage) as guardian and leaves centroid 15 unprotected for recycling!
    prot = twin_safe_huygens_protection(
        centroids=centroids,
        low_mask=low_mask,
        usage=usage,
        survival_mass=usage.clone(),
        thresh=thresh,
        ref_distortion_mean=torch.tensor(10.0, device=device),
        gamma_protect=0.5,
    )
    assert bool(prot[14].item()) is True
    assert bool(prot[15].item()) is False

    # End-to-end verification in `SuperEMeans.fit` with explicit twin centroids on an isolated rare mode
    n_bulk = 2000
    n_rare = 6  # p_rare = 6 / 2006 < 0.1 / 16 = 0.00625!
    x_bulk = torch.randn((n_bulk, d), generator=gen, device=device) * 0.6
    x_rare = 40.0 + torch.randn((n_rare, d), generator=gen, device=device) * 0.1
    x_all = torch.cat([x_bulk, x_rare], dim=0)

    c_init = x_bulk[:k].clone()
    c_init[14] = x_rare[0]
    c_init[15] = x_rare[1]  # Twin rare centroids!

    model = SuperEMeans(d=d, k=k, nepochs=4.0, batch_size=256, seed=33)
    c_final, _ = model.fit(x_all, init_centroids=c_init)
    dists_to_rare = (c_final - 40.0).norm(dim=1)
    n_guardians = int((dists_to_rare < 2.0).sum().item())
    assert n_guardians == 1, f"Expected exactly 1 surviving rare-mode guardian (duplicate recycled), got {n_guardians}"


@pytest.mark.parametrize("dev", CUDA_DEVICES)
def test_rare_mode_survival_super_emeans_vs_smeans(dev: str) -> None:
    """Verify `SuperEMeans` preserves genuine rare modes while unprotected `smeans_deconfounded` loses them."""
    rng = np.random.default_rng(108)
    d, k = 32, 32
    bulk_centers = rng.standard_normal((28, d)).astype(np.float32) * 2.0
    rare_centers = rng.standard_normal((4, d)).astype(np.float32) * 6.0 + 25.0

    x_bulk = bulk_centers[rng.integers(0, 28, size=3984)] + 0.35 * rng.standard_normal((3984, d)).astype(np.float32)
    # 4 genuine isolated rare modes with only 4 points each (p_j = 0.001 << 0.1 / 32 = 0.0031)
    x_rare = np.repeat(rare_centers, 4, axis=0) + 0.1 * rng.standard_normal((16, d)).astype(np.float32)
    x_tr = torch.from_numpy(np.vstack([x_bulk, x_rare])).to(dev)

    m_super = SuperEMeans(d=d, k=k, nepochs=4.0, batch_size=256, seed=7)
    m_smeans = build_hybrid_arm("smeans_deconfounded", d=d, k=k, nepochs=4.0, seed=7, batch_size=256)

    c_super, _ = m_super.fit(x_tr)
    c_smeans, _ = m_smeans.fit(x_tr)

    rc_t = torch.from_numpy(rare_centers).to(dev)
    super_hits = int(((rc_t[:, None, :] - c_super[None, :, :]).norm(dim=2).min(dim=1).values < 2.0).sum().item())
    mse_super = exact_mse_fp64(x_tr, c_super, device=dev)
    mse_smeans = exact_mse_fp64(x_tr, c_smeans, device=dev)

    assert super_hits == 4, f"SuperEMeans should recover all 4 isolated rare modes, got {super_hits}"
    assert mse_super < mse_smeans


@pytest.mark.parametrize("dev", CUDA_DEVICES)
@pytest.mark.parametrize(
    "arm_name",
    ["smeans_deconfounded", "smeans_plus_p2_huygens_shield", "smeans_plus_p5_head_edit", "pp_plus_c1_d15_raw"],
)
def test_patrol93_empty_outlier_centroids_not_falsely_shielded(dev: str, arm_name: str) -> None:
    """Regression test for `#93`: 0-count empty centroids must NOT receive fake `p_hat = 1/k` survival shielding."""
    gen = torch.Generator(device=dev).manual_seed(17)
    ctrs = torch.randn((32, 32), generator=gen, device=dev) * 2.5
    probs = 1.0 / torch.arange(1, 33, device=dev, dtype=torch.float32)
    probs /= probs.sum()
    idx = torch.multinomial(probs, 2048, replacement=True, generator=gen)
    x = (ctrs[idx] + 0.35 * torch.randn((2048, 32), generator=gen, device=dev)).contiguous()

    c_init = x[:32].clone()
    c_init[:8] = 50.0  # 8 completely empty outlier centroids far from all data
    model = build_hybrid_arm(arm_name, d=32, k=32, nepochs=1.0, seed=42)
    c_out, _ = model.fit(x, init_centroids=c_init)
    assert torch.isfinite(c_out).all()
    assert model.dead_clusters == 0, f"{arm_name} left {model.dead_clusters} dead empty clusters unrevived!"


# ---------------------------------------------------------------------------
# 5. Adversarial Shapes, Fractional Budgets, Extreme Scales, & All-27-Arm Sweep
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dev", CUDA_DEVICES)
@pytest.mark.parametrize(
    "N,D,K,nepochs",
    [
        (129, 1, 7, 1.0),
        (257, 1, 1, 1.5),
        (67, 2, 5, 0.5),
        (131, 3, 2, 2.0),
        (37, 17, 37, 1.0),   # N == K boundary
        (521, 127, 31, 1.5), # Prime N, D, K + fractional epochs
        (509, 129, 29, 2.0), # Just above D=128 FULL_D boundary
        (389, 259, 19, 2.0), # Just above D=256 Oja FULL_D boundary
    ],
)
def test_adversarial_edge_shapes_and_fractional_budgets(
    dev: str, N: int, D: int, K: int, nepochs: float
) -> None:
    gen = torch.Generator(device=dev).manual_seed(N + D * 13 + K * 7)
    x = torch.randn((N, D), generator=gen, device=dev, dtype=torch.float32)

    for arm in ("super_emeans", "super_emeans_wide", "hybrid_super_smeans", "smeans_deconfounded"):
        m = build_hybrid_arm(arm, d=D, k=K, nepochs=nepochs, seed=19)
        c, a = m.fit(x, capture_assignments=True)
        assert c.shape == (K, D) and torch.isfinite(c).all()
        assert a is not None and a.shape == (N,)
        assert bool(((a >= 0) & (a < K)).all())
        dists, lbls = m.assign(x[: min(17, N)])
        assert torch.isfinite(dists).all() and bool(((lbls >= 0) & (lbls < K)).all())


@pytest.mark.parametrize("dev", CUDA_DEVICES)
def test_strided_noncontiguous_prime_d_inputs(dev: str) -> None:
    gen = torch.Generator(device=dev).manual_seed(404)
    # Column-sliced tensor with stride(1) == 2 and prime feature count D = 65
    x_wide = torch.randn((389, 130), generator=gen, device=dev, dtype=torch.float32)
    x_strided = x_wide[:, ::2]
    assert not x_strided.is_contiguous() and x_strided.shape == (389, 65)

    for arm in ("super_emeans", "hybrid_super_smeans", "smeans_deconfounded"):
        m = build_hybrid_arm(arm, d=65, k=23, nepochs=2.0, seed=13)
        c, a = m.fit(x_strided)
        assert c.shape == (23, 65) and torch.isfinite(c).all()
        assert a is not None and bool(((a >= 0) & (a < 23)).all())
        d_q, l_q = m.assign(x_strided[:31])
        assert torch.isfinite(d_q).all() and bool(((l_q >= 0) & (l_q < 23)).all())


@pytest.mark.parametrize("dev", CUDA_DEVICES)
@pytest.mark.parametrize("mode", ["zeros", "constant", "huge_1e25", "tiny_1e_25", "fp16_near_max"])
def test_degenerate_and_extreme_scale_inputs(dev: str, mode: str) -> None:
    if mode == "fp16_near_max" and dev == "cpu":
        pytest.skip("FP16 Tensor-Core path requires CUDA")

    device = torch.device(dev)
    gen = torch.Generator(device=device).manual_seed(808)
    n, d, k = 512, 32, 16

    comp_dtype = None
    if mode == "zeros":
        x = torch.zeros((n, d), dtype=torch.float32, device=device)
    elif mode == "constant":
        x = torch.full((n, d), 1234.5, dtype=torch.float32, device=device)
    elif mode == "huge_1e25":
        x = torch.randn((n, d), generator=gen, device=device, dtype=torch.float32) * 1e25
    elif mode == "tiny_1e_25":
        x = torch.randn((n, d), generator=gen, device=device, dtype=torch.float32) * 1e-25
    else:
        x = torch.randn((n, d), generator=gen, device=device, dtype=torch.float32) * 20_000.0 + 15_000.0
        comp_dtype = torch.float16

    for arm in ("super_emeans", "hybrid_super_smeans", "smeans_deconfounded"):
        m = build_hybrid_arm(arm, d=d, k=k, nepochs=2.0, seed=5, compute_dtype=comp_dtype)
        c, a = m.fit(x)
        assert torch.isfinite(c).all(), f"Non-finite centroids in {arm} on {mode}"
        assert a is not None and bool(((a >= 0) & (a < k)).all())
        for st in m.iteration_stats:
            assert math.isfinite(st.moving_mse) and math.isfinite(st.imbalance_factor)
        if mode == "zeros":
            torch.testing.assert_close(c, torch.zeros_like(c), atol=1e-3, rtol=0.0)
            assert exact_mse_fp64(x, c, device=dev) < 1e-10
        elif mode == "constant":
            torch.testing.assert_close(c, torch.full_like(c, 1234.5), atol=1e-3, rtol=1e-5)


@pytest.mark.parametrize("dev", CUDA_DEVICES)
def test_all_27_hybrid_ablation_arms_smoke(dev: str) -> None:
    rng = np.random.default_rng(314)
    centers = rng.standard_normal((24, 24)).astype(np.float32) * 3.0
    x = torch.from_numpy(
        centers[rng.integers(0, 24, size=768)] + 0.4 * rng.standard_normal((768, 24)).astype(np.float32)
    ).to(dev)

    for arm_name in sorted(HYBRID_ABLATION_ARMS.keys()):
        m = build_hybrid_arm(arm_name, d=24, k=24, nepochs=2.0, seed=12, batch_size=128)
        c, a = m.fit(x)
        assert c.shape == (24, 24) and torch.isfinite(c).all(), f"Failed on {arm_name}"
        assert a is not None and a.shape == (768,)
        mse = exact_mse_fp64(x, c, device=dev)
        assert math.isfinite(mse) and mse < 20.0, f"Unexpected MSE {mse} on {arm_name}"
