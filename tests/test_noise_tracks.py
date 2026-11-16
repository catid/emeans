"""Comprehensive mathematical, numerical, adversarial, and parity unit tests for `noise_tracks.py` & `run_phase2a_noise.py`."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from emeans.datasets import make_synthetic_dataset
from emeans.emeans import EMeans
from emeans.emeans_pp import EMeansPP
from emeans.kernels import exact_mse_fp64
from emeans.noise_tracks import (
    NOISE_HYPOTHESES,
    EMeansNoiseTrack,
    HeadEditDiagnostics,
    anvil_polar_displacement,
    build_spatial_surprisal_order,
    exact_exclusion_sq_dists,
    get_sylvester_hadamard,
    gram_newton_schulz_polar,
    run_head_edit_step,
    sample_signed_hadamard_rows,
    sample_stratified_epoch_order,
    top3_represented_assign_centered,
)
import run_phase2a_noise as phase2a_runner


CUDA_DEVICES = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


# ---------------------------------------------------------------------------
# 1. Pre-Registered `qlabs-eng/noise` Track Specification Completeness
# ---------------------------------------------------------------------------


def test_noise_hypotheses_registry_complete() -> None:
    expected = {"N-Track-1", "N-Track-2", "N-Track-3", "N-Track-4", "N-Track-5"}
    assert set(NOISE_HYPOTHESES.keys()) == expected
    for tid, spec in NOISE_HYPOTHESES.items():
        assert spec.id == tid
        assert len(spec.name) > 15
        assert len(spec.noise_source) > 15
        assert len(spec.mechanism) > 30
        assert len(spec.equation) > 20
        assert len(spec.complexity_overhead) > 10
        assert len(spec.predicted_win_regime) > 20
        assert len(spec.predicted_fail_regime) > 20
        assert "Falsified if" in spec.falsification_criterion


# ---------------------------------------------------------------------------
# 2. Exact Baseline Parity with Core `EMeans` & `EMeansPP`
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dev", CUDA_DEVICES)
@pytest.mark.parametrize("finalization", ["closing_mean", "slow_ema", "none"])
def test_exact_parity_with_base_emeans(dev: str, finalization: str) -> None:
    rng = np.random.default_rng(11)
    x_np = rng.standard_normal((2048, 32), dtype=np.float32)
    x_t = torch.from_numpy(x_np).to(dev)

    ref = EMeans(
        d=32,
        k=64,
        nepochs=3.0,
        batch_size=256,
        finalization=finalization,
        seed=77,
    )
    trk = EMeansNoiseTrack(
        d=32,
        k=64,
        nepochs=3.0,
        base_mode="emeans",
        batch_size=256,
        finalization=finalization,
        seed=77,
    )
    prev_det = torch.are_deterministic_algorithms_enabled()
    prev_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        c_ref, a_ref = ref.fit(x_t)
        c_trk, a_trk = trk.fit(x_t)
    finally:
        torch.use_deterministic_algorithms(prev_det, warn_only=prev_warn)

    assert trk.n_splits == ref.n_splits
    assert trk.n_transplants == 0
    assert torch.equal(a_ref, a_trk)
    torch.testing.assert_close(c_trk, c_ref, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("dev", CUDA_DEVICES)
@pytest.mark.parametrize("finalization", ["closing_mean", "slow_ema"])
@pytest.mark.parametrize("pass_init_centroids", [False, True])
def test_exact_parity_with_emeans_pp(
    dev: str, finalization: str, pass_init_centroids: bool
) -> None:
    rng = np.random.default_rng(23)
    centers = rng.standard_normal((48, 32)).astype(np.float32) * 2.5
    probs = 1.0 / np.arange(1, 49, dtype=np.float64)
    probs /= probs.sum()
    comp = rng.choice(48, size=3000, p=probs)
    x_np = (centers[comp] + 0.35 * rng.standard_normal((3000, 32))).astype(np.float32)
    x_t = torch.from_numpy(x_np).to(dev)
    init_c = x_t[:48].clone() if pass_init_centroids else None

    ref_pp = EMeansPP(
        d=32,
        k=48,
        nepochs=3.0,
        batch_size=256,
        finalization=finalization,
        seed=456,
    )
    trk_pp = EMeansNoiseTrack(
        d=32,
        k=48,
        nepochs=3.0,
        base_mode="emeans_pp",
        batch_size=256,
        finalization=finalization,
        seed=456,
    )
    prev_det = torch.are_deterministic_algorithms_enabled()
    prev_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        c_ref, a_ref = ref_pp.fit(x_t, init_centroids=init_c)
        c_trk, a_trk = trk_pp.fit(x_t, init_centroids=init_c)
    finally:
        torch.use_deterministic_algorithms(prev_det, warn_only=prev_warn)

    assert trk_pp.recipe is not None and ref_pp.recipe is not None
    assert trk_pp.recipe.alpha0 == pytest.approx(ref_pp.recipe.alpha0)
    assert trk_pp.n_splits == ref_pp.n_splits
    assert trk_pp.n_protected == ref_pp.n_protected
    assert torch.equal(a_ref, a_trk)
    torch.testing.assert_close(c_trk, c_ref, rtol=0.0, atol=0.0)


# ---------------------------------------------------------------------------
# 3. Building-Block Mathematical Invariant Tests (`N-Track-1..5`)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dev", CUDA_DEVICES)
@pytest.mark.parametrize("d", [1, 2, 3, 7, 16, 64, 130])
def test_sylvester_hadamard_and_arbitrary_d_probes(dev: str, d: int) -> None:
    device = torch.device(dev)
    if (d & (d - 1)) == 0:
        h = get_sylvester_hadamard(d, device=device)
        gram = h @ h.t()
        torch.testing.assert_close(gram, float(d) * torch.eye(d, device=device), rtol=1e-5, atol=1e-5)

    gen = torch.Generator(device=device).manual_seed(123)
    rows_rms = sample_signed_hadamard_rows(20, d, generator=gen, device=device, unit_l2=False)
    assert rows_rms.shape == (20, d)
    assert torch.all((rows_rms == 1.0) | (rows_rms == -1.0))
    torch.testing.assert_close(rows_rms.square().mean(dim=1), torch.ones(20, device=device))

    rows_u = sample_signed_hadamard_rows(20, d, generator=gen, device=device, unit_l2=True)
    torch.testing.assert_close(torch.linalg.vector_norm(rows_u, dim=1), torch.ones(20, device=device), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dev", CUDA_DEVICES)
@pytest.mark.parametrize("comp_dtype", [torch.float32, torch.bfloat16, torch.float16, torch.float64])
def test_top3_represented_assign_and_exact_exclusion(dev: str, comp_dtype: torch.dtype) -> None:
    device = torch.device(dev)
    gen = torch.Generator(device=device).manual_seed(99)
    # Test with large +2e4 coordinate shift to verify translation-invariant centering
    offset = 20_000.0 if comp_dtype in (torch.float32, torch.float64) else 64.0
    x = torch.randn((64, 24), generator=gen, device=device, dtype=torch.float32) + offset
    c = torch.randn((15, 24), generator=gen, device=device, dtype=torch.float32) + offset

    top_d, top_i, _ = top3_represented_assign_centered(x, c, compute_dtype=comp_dtype, centroid_chunk_size=6)
    assert top_d.shape == (64, 3)
    assert top_i.shape == (64, 3)
    # Strictly distinct indices per row
    assert bool((top_i[:, 0] != top_i[:, 1]).all())
    assert bool((top_i[:, 0] != top_i[:, 2]).all())
    assert bool((top_i[:, 1] != top_i[:, 2]).all())

    # Exact brute-force represented verification
    xr = x.to(comp_dtype).double()
    cr = c.to(comp_dtype).double()
    d_exact = ((xr[:, None, :] - cr[None, :, :]).square().sum(dim=-1)).float()
    for a_rem in (0, 4, 14):
        for b_rem in (1, 7, 13):
            if a_rem == b_rem:
                continue
            keep = [j for j in range(15) if j not in (a_rem, b_rem)]
            bf_min = d_exact[:, keep].min(dim=1).values
            fast_min = exact_exclusion_sq_dists(top_d, top_i, a_rem, b_rem)
            torch.testing.assert_close(fast_min, bf_min, rtol=2e-4, atol=2e-4)

    # Degenerate all-identical centroids & K < 3 safety
    c_zero = torch.zeros((10, 24), device=device, dtype=torch.float32)
    _, idx_zero, _ = top3_represented_assign_centered(x, c_zero, compute_dtype=comp_dtype, centroid_chunk_size=4)
    assert bool((idx_zero[:, 0] != idx_zero[:, 1]).all())
    assert bool((idx_zero[:, 1] != idx_zero[:, 2]).all())

    d_k2, i_k2, _ = top3_represented_assign_centered(x, c[:2], compute_dtype=comp_dtype)
    assert bool(torch.isinf(d_k2[:, 2]).all())
    assert bool((i_k2[:, 2] == -1).all())


@pytest.mark.parametrize("dev", CUDA_DEVICES)
def test_head_edit_escapes_twin_trap_and_protects_rare_modes(dev: str) -> None:
    """Prove `N-Track-1` escapes a local Voronoi twin trap while shielding isolated rare modes."""
    device = torch.device(dev)
    rng = np.random.default_rng(7)
    d = 16
    # Mode A at 0 (500 pts), Mode B1 at +8 e_0 (250 pts), Mode B2 at +14 e_0 (250 pts)
    x_a = 0.15 * rng.standard_normal((500, d)).astype(np.float32)
    c_b1 = np.zeros((1, d), dtype=np.float32); c_b1[0, 0] = 8.0
    c_b2 = np.zeros((1, d), dtype=np.float32); c_b2[0, 0] = 14.0
    x_b1 = c_b1 + 0.15 * rng.standard_normal((250, d)).astype(np.float32)
    x_b2 = c_b2 + 0.15 * rng.standard_normal((250, d)).astype(np.float32)
    x_all = np.concatenate([x_a, x_b1, x_b2], axis=0)
    rng.shuffle(x_all)
    x_t = torch.from_numpy(x_all).to(device)

    # Stuck local optimum for k=3: two centroids in Mode A, one centroid straddling Mode B1 & B2!
    init_c = torch.zeros((3, d), device=device, dtype=torch.float32)
    init_c[0, 0] = -0.08
    init_c[1, 0] = +0.08
    init_c[2, 0] = 11.0  # midpoint between 8 and 14; all 3 centroids get >= 25% of points so revival never triggers!

    m_base = EMeansNoiseTrack(
        d=d, k=3, nepochs=4.0, base_mode="emeans", batch_size=500, head_edit=False, seed=5
    )
    c_base, _ = m_base.fit(x_t, init_centroids=init_c)
    mse_base = exact_mse_fp64(x_t, c_base, device=device)

    m_edit = EMeansNoiseTrack(
        d=d,
        k=3,
        nepochs=4.0,
        base_mode="emeans",
        batch_size=500,
        head_edit=True,
        transplant_every=1,
        transplant_probes=16,
        seed=5,
    )
    c_edit, _ = m_edit.fit(x_t, init_centroids=init_c)
    mse_edit = exact_mse_fp64(x_t, c_edit, device=device)

    assert m_edit.n_transplants >= 1
    assert mse_edit < 0.15 * mse_base  # Huge drop from ~4.5 down to ~0.36!

    # Now verify Huygens rare-mode shield inside `run_head_edit_step` when rare mode has 0 batch samples
    centroids_rare = init_c.clone()
    centroids_rare = torch.cat([centroids_rare, torch.full((1, d), 50.0, device=device)], dim=0)  # k=4, idx=3 is isolated rare mode!
    usage = torch.tensor([125.0, 125.0, 250.0, 1.0], device=device)  # idx 3 has low usage < 0.1 * mean
    distortion = torch.tensor([10.0, 10.0, 2250.0, 0.5], device=device)
    survival_mass = torch.tensor([125.0, 125.0, 250.0, 2.0], device=device)
    last_edit = torch.full((4,), -10**9, dtype=torch.int64, device=device)
    diag = HeadEditDiagnostics()
    gen = torch.Generator(device=device).manual_seed(17)
    rare_before = centroids_rare[3].clone()

    _ = run_head_edit_step(
        x_batch=x_t[:500],
        centroids=centroids_rare,
        usage=usage,
        distortion=distortion,
        survival_mass=survival_mass,
        slow_centroids=None,
        slow_usage=None,
        tail_banks=[],
        last_edit_step=last_edit,
        step=5,
        compute_dtype=torch.float32,
        unused_threshold=0.1,
        use_huygens_protection=True,
        gamma_protect=0.5,
        num_probes=16,
        max_pairs=2,
        protect_steps=5,
        generator=gen,
        diag=diag,
    )
    # Isolated rare mode idx=3 MUST remain untouched (twin idx 0 or 1 is transplanted instead!)
    torch.testing.assert_close(centroids_rare[3], rare_before, rtol=0.0, atol=0.0)

    # Regression test (#54.1 / #55.1): Twin centroids sitting on an isolated rare mode (`0` batch points)
    # with `max_pairs=4` MUST transplant at most ONE twin and immediately shield the surviving twin!
    d_tw = 32
    x_don1_a = torch.randn((128, d_tw), generator=gen, device=device) * 0.2
    x_don1_b = torch.randn((128, d_tw), generator=gen, device=device) * 0.2
    x_don1_a[:, 0] -= 3.0
    x_don1_b[:, 0] += 3.0
    x_don2_a = torch.randn((128, d_tw), generator=gen, device=device) * 0.2
    x_don2_b = torch.randn((128, d_tw), generator=gen, device=device) * 0.2
    x_don2_a[:, 0] += 14.0
    x_don2_b[:, 0] += 20.0
    xb_tw = torch.cat([x_don1_a, x_don1_b, x_don2_a, x_don2_b], dim=0)
    xb_tw = xb_tw[torch.randperm(xb_tw.shape[0], generator=gen, device=device)]

    c_tw = torch.zeros((4, d_tw), device=device, dtype=torch.float32)
    c_tw[0, 0] = 50.000  # Twin #1 on isolated rare mode
    c_tw[1, 0] = 50.001  # Twin #2 on isolated rare mode
    c_tw[2, 0] = 0.0     # Bimodal bulk donor #1 (-3 / +3)
    c_tw[3, 0] = 17.0    # Bimodal bulk donor #2 (+14 / +20)
    rare_target = torch.zeros((1, d_tw), device=device, dtype=torch.float32)
    rare_target[0, 0] = 50.0005

    u_tw = torch.tensor([0.5, 0.5, 256.0, 256.0], device=device)
    dist_tw = torch.tensor([0.1, 0.1, 2400.0, 2400.0], device=device)
    surv_tw = torch.tensor([2.0, 2.0, 256.0, 256.0], device=device)
    last_tw = torch.full((4,), -10**9, dtype=torch.int64, device=device)
    diag_tw = HeadEditDiagnostics()

    n_ed_tw = run_head_edit_step(
        x_batch=xb_tw,
        centroids=c_tw,
        usage=u_tw,
        distortion=dist_tw,
        survival_mass=surv_tw,
        slow_centroids=None,
        slow_usage=None,
        tail_banks=[],
        last_edit_step=last_tw,
        step=10,
        compute_dtype=torch.float32,
        unused_threshold=0.1,
        use_huygens_protection=True,
        gamma_protect=0.5,
        num_probes=8,
        max_pairs=4,
        protect_steps=5,
        generator=gen,
        diag=diag_tw,
    )
    assert n_ed_tw == 1
    min_sq_to_rare = float((c_tw - rare_target).square().sum(dim=1).min().item())
    assert min_sq_to_rare < 1e-4

    # Regression test (#55.2): High-dimensional (`d=128`) oblique bimodal donor split succeeds (`q_confirmed >= 1`)
    d_hi = 128
    dir_oblique = torch.randn((1, d_hi), generator=gen, device=device)
    dir_oblique = dir_oblique / torch.linalg.vector_norm(dir_oblique)
    x_mode0 = 0.25 * torch.randn((256, d_hi), generator=gen, device=device)
    center_hi = torch.zeros((1, d_hi), device=device)
    center_hi[0, 0] = 25.0
    x_m1 = center_hi - 4.0 * dir_oblique + 0.25 * torch.randn((128, d_hi), generator=gen, device=device)
    x_m2 = center_hi + 4.0 * dir_oblique + 0.25 * torch.randn((128, d_hi), generator=gen, device=device)
    x_hi = torch.cat([x_mode0, x_m1, x_m2], dim=0)
    x_hi = x_hi[torch.randperm(x_hi.shape[0], generator=gen, device=device)]
    init_hi = torch.zeros((3, d_hi), device=device)
    init_hi[0, 0] = -0.05
    init_hi[1, 0] = +0.05
    init_hi[2] = center_hi[0]

    m_hi = EMeansNoiseTrack(
        d=d_hi, k=3, nepochs=2.0, base_mode="emeans", batch_size=256, head_edit=True, seed=21
    )
    c_hi, _ = m_hi.fit(x_hi, init_centroids=init_hi)
    assert m_hi.head_edit_diagnostics is not None
    assert m_hi.head_edit_diagnostics.q_confirmed >= 1
    assert exact_mse_fp64(x_hi, c_hi, device=device) < 9.0  # Unsplit MSE is ~8 + 128*0.0625 = ~16.0!


@pytest.mark.parametrize("dev", CUDA_DEVICES)
@pytest.mark.parametrize("shape", [(1, 1), (1, 16), (16, 1), (12, 12), (64, 16), (16, 64)])
def test_gram_newton_schulz_polar_and_anvil_safety(dev: str, shape: tuple[int, int]) -> None:
    device = torch.device(dev)
    k, d = shape
    # Zero input safety
    z = torch.zeros((k, d), device=device, dtype=torch.float32)
    pz = gram_newton_schulz_polar(z)
    assert torch.isfinite(pz).all()
    assert float(pz.abs().max().item()) == 0.0

    # Random full-rank matrix: non-zero singular values map near 1.0
    gen = torch.Generator(device=device).manual_seed(33)
    m = torch.randn((k, d), generator=gen, device=device, dtype=torch.float32) * 5.0
    pm = gram_newton_schulz_polar(m)
    assert pm.shape == (k, d)
    assert torch.isfinite(pm).all()
    svals = torch.linalg.svdvals(pm)
    assert float(svals.max().item()) <= 1.05
    assert float(svals.min().item()) >= 0.85

    # ANVIL row trust-region bound (`<= 2.0 * ||delta_ema_j||_2`)
    touched = torch.ones((k,), dtype=torch.bool, device=device)
    if k >= 2:
        touched[-1] = False
        m[-1] = 0.0
    fast = torch.zeros_like(m)
    slow = torch.zeros_like(m)
    energy = torch.zeros((k, 1), device=device, dtype=torch.float32)
    disp = anvil_polar_displacement(
        delta_ema=m,
        touched=touched,
        fast_rail=fast,
        slow_rail=slow,
        lane_energy=energy,
        progress=0.1,
        blend0=0.5,
    )
    assert torch.isfinite(disp).all()
    if k >= 2:
        assert float(disp[-1].abs().max().item()) == 0.0
    row_norms_disp = torch.linalg.vector_norm(disp, dim=1)
    row_norms_orig = torch.linalg.vector_norm(m, dim=1)
    assert bool((row_norms_disp <= 2.0001 * row_norms_orig + 1e-6).all())


@pytest.mark.parametrize("dev", CUDA_DEVICES)
def test_stratified_batching_reduces_variance_and_preserves_uniqueness(dev: str) -> None:
    device = torch.device(dev)
    ds = make_synthetic_dataset("zipf_gmm", n_train=4096, n_val=64, n_test=64, d=32, k=64, seed=9)
    x_t = torch.from_numpy(ds.train).to(device)
    sorted_ord = build_spatial_surprisal_order(x_t, seed=9)
    assert sorted_ord.shape == (4096,)
    assert int(torch.unique(sorted_ord).numel()) == 4096

    gen_s = torch.Generator(device=device).manual_seed(101)
    ep_ord = sample_stratified_epoch_order(sorted_ord, rows_in_ep=4096, batch_size=256, generator=gen_s)
    assert ep_ord.shape == (4096,)
    # Exact 1-pass permutation without replacement!
    assert int(torch.unique(ep_ord).numel()) == 4096

    # Measure batch mean radial norm variance across the 16 mini-batches vs i.i.d. uniform permutation
    radii = (x_t - x_t.mean(dim=0, keepdim=True)).square().sum(dim=1)
    strat_batch_means = radii[ep_ord].reshape(16, 256).mean(dim=1)
    gen_u = torch.Generator(device=device).manual_seed(101)
    unif_ord = torch.randperm(4096, generator=gen_u, device=device)
    unif_batch_means = radii[unif_ord].reshape(16, 256).mean(dim=1)
    assert float(strat_batch_means.var().item()) < 0.25 * float(unif_batch_means.var().item())

    # Non-divisible `n % batch_size != 0` (`4500 % 256 = 148`) must NOT quarantine outer radial tail
    # into the final remainder batch!
    ds_rem = make_synthetic_dataset("multiscale_rare_outliers", n_train=4500, n_val=64, n_test=64, d=32, k=64, seed=15)
    x_rem = torch.from_numpy(ds_rem.train).to(device)
    s_rem = build_spatial_surprisal_order(x_rem, seed=15)
    ep_rem = sample_stratified_epoch_order(
        s_rem, rows_in_ep=4500, batch_size=256, generator=torch.Generator(device=device).manual_seed(7)
    )
    assert int(torch.unique(ep_rem).numel()) == 4500
    rad_rem = (x_rem - x_rem.mean(dim=0, keepdim=True)).square().sum(dim=1)
    main_mean = float(rad_rem[ep_rem[: 17 * 256]].mean().item())
    tail_mean = float(rad_rem[ep_rem[17 * 256 :]].mean().item())
    assert abs(tail_mean - main_mean) < 0.15 * main_mean

    # Verify `wide_step_mult=2` preserves 100% of rare outlier modes across the B -> 2B transition
    ds_rare = make_synthetic_dataset("multiscale_rare_outliers", n_train=24576, n_val=64, n_test=64, d=32, k=64, seed=14)
    xr = torch.from_numpy(ds_rare.train).to(device)
    m_wide = EMeansNoiseTrack(
        d=32, k=64, nepochs=4.0, base_mode="emeans_pp", batch_size=256,
        stratified_order=True, wide_step_mult=2, wide_from_frac=0.25, seed=14,
    )
    cw, _ = m_wide.fit(xr, capture_assignments=False)
    assert ds_rare.true_centers is not None and ds_rare.rare_mode_indices is not None
    rare_c = torch.from_numpy(ds_rare.true_centers[ds_rare.rare_mode_indices]).to(device=device, dtype=torch.float32)
    d2_rare = torch.cdist(rare_c, cw).square().min(dim=1).values
    assert float((d2_rare < 32.0 * (0.35**2)).float().mean().item()) == 1.0


@pytest.mark.parametrize("dev", CUDA_DEVICES)
def test_quad_merge_monotone_holdout_and_tiny_n_safety(dev: str) -> None:
    device = torch.device(dev)
    ds = make_synthetic_dataset("balanced_gmm", n_train=2048, n_val=256, n_test=512, d=32, k=128, seed=12)
    x_tr = torch.from_numpy(ds.train).to(device)

    m_qm = EMeansNoiseTrack(
        d=32,
        k=128,
        nepochs=3.0,
        base_mode="emeans_pp",
        finalization="quad_merge",
        seed=55,
    )
    c_qm, a_qm = m_qm.fit(x_tr, capture_assignments=True)
    assert c_qm.shape == (128, 32)
    assert a_qm is not None and a_qm.shape == (2048,)
    assert torch.isfinite(c_qm).all()
    assert m_qm.quad_merge_diagnostics is not None
    assert m_qm.quad_merge_diagnostics.holdout_mse_merged <= m_qm.quad_merge_diagnostics.holdout_mse_plain + 1e-6

    # Tiny `n == k` (`n=16, k=16`) graceful fallback without starving `x_fit`
    x_tiny = x_tr[:16]
    m_tiny = EMeansNoiseTrack(
        d=32, k=16, nepochs=2.0, base_mode="emeans", finalization="quad_merge", seed=3
    )
    c_tiny, a_tiny = m_tiny.fit(x_tiny, capture_assignments=True)
    assert c_tiny.shape == (16, 32)
    assert a_tiny is not None and a_tiny.shape == (16,)
    assert torch.isfinite(c_tiny).all()

    # Regression test (#55.3 & #59.2): `+5000.0` coordinate shift must NOT collapse `holdout_mse_plain`
    # to 0.0 under BF16, and `execute_quad_merge` must be exact shift-invariant + protect populous clusters!
    from emeans.noise_tracks import execute_quad_merge

    x_hold_ref = x_tr[:128]
    c_fit_ref = c_qm.clone()
    t_slow_ref = c_qm + 0.15 * torch.randn_like(c_qm)
    counts_zipf = torch.ones((128,), dtype=torch.float64, device=device) * 12.0
    counts_zipf[:8] = 2000.0  # Populous Zipf head clusters (`n_j = 2000 >> kappa = 64`)
    sums_ref = (c_fit_ref.double() * counts_zipf[:, None]).float()

    c_m0, diag0 = execute_quad_merge(
        x_hold=x_hold_ref,
        c_close_fit=c_fit_ref,
        pass_counts_fit=counts_zipf,
        pass_sums_fit=sums_ref,
        candidates=[("tail_slow", t_slow_ref)],
        compute_dtype=torch.bfloat16,
        backend="auto",
    )
    shift = 5000.0
    x_hold_sh = x_hold_ref + shift
    c_fit_sh = c_fit_ref + shift
    t_slow_sh = t_slow_ref + shift
    sums_sh = (c_fit_sh.double() * counts_zipf[:, None]).float()

    c_m_sh, diag_sh = execute_quad_merge(
        x_hold=x_hold_sh,
        c_close_fit=c_fit_sh,
        pass_counts_fit=counts_zipf,
        pass_sums_fit=sums_sh,
        candidates=[("tail_slow", t_slow_sh)],
        compute_dtype=torch.bfloat16,
        backend="auto",
    )
    assert diag_sh.holdout_mse_plain > 1.0
    assert diag_sh.holdout_mse_plain == pytest.approx(diag0.holdout_mse_plain, rel=1e-4, abs=1e-4)
    assert diag_sh.holdout_mse_merged == pytest.approx(diag0.holdout_mse_merged, rel=1e-4, abs=1e-4)
    torch.testing.assert_close(c_m_sh - shift, c_m0, rtol=1e-3, atol=2e-3)


@pytest.mark.parametrize("dev", CUDA_DEVICES)
def test_t2_closing_pass_clean_under_dust_and_anvil(dev: str) -> None:
    """Regression test (#54.2 & #59.1): At `nepochs=2.0`, closing pass (`Epoch 1`) is 100% jitter- & polar-free."""
    device = torch.device(dev)
    ds = make_synthetic_dataset("zipf_gmm", n_train=2048, n_val=64, n_test=64, d=32, k=32, seed=19)
    x_tr = torch.from_numpy(ds.train).to(device)

    m = EMeansNoiseTrack(
        d=32,
        k=32,
        nepochs=2.0,
        base_mode="emeans_pp",
        batch_size=256,
        dust_sigma0=0.35,
        dust_stop_frac=0.80,   # Intentionally > 0.50: `not is_last_epoch` must still keep Epoch 1 clean!
        anvil_blend0=0.50,
        anvil_stop_frac=0.80,  # Intentionally > 0.50: `not is_last_epoch` must still keep Epoch 1 convex-hull!
        revival_mode="off",
        finalization="closing_mean",
        seed=19,
    )
    c_out, a_out = m.fit(x_tr, capture_assignments=True)
    assert a_out is not None
    # Every non-empty cluster's final `closing_mean` centroid must equal the exact sample mean of its
    # Epoch 1 assigned points (which would fail if jitter or polar steps contaminated Epoch 1's pass_sums
    # or if centroids drifted outside the convex hull `[x_tr.min(0), x_tr.max(0)]`)!
    counts = torch.bincount(a_out, minlength=32)
    sums = torch.zeros((32, 32), dtype=torch.float32, device=device)
    sums.index_add_(0, a_out, x_tr.float())
    nonempty = counts > 0
    exact_means = sums[nonempty] / counts[nonempty].float()[:, None]
    torch.testing.assert_close(c_out[nonempty], exact_means, rtol=1e-5, atol=1e-5)
    x_min = x_tr.min(dim=0).values - 1e-4
    x_max = x_tr.max(dim=0).values + 1e-4
    assert bool(((c_out >= x_min) & (c_out <= x_max)).all())


# ---------------------------------------------------------------------------
# 4. Adversarial Edge Cases, Dtypes, Reproducibility, & Runner Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dev", CUDA_DEVICES)
@pytest.mark.parametrize("base_mode", ["emeans", "emeans_pp"])
def test_all_tracks_combined_edge_cases_and_dtypes(dev: str, base_mode: str) -> None:
    device = torch.device(dev)

    # Case A: k=1, d=1
    x_1d = torch.randn((64, 1), device=device, dtype=torch.float32)
    m_1d = EMeansNoiseTrack(
        d=1,
        k=1,
        nepochs=2.0,
        base_mode=base_mode,
        head_edit=True,
        finalization="quad_merge",
        anvil_blend0=0.3,
        dust_sigma0=0.15,
        stratified_order=True,
        wide_step_mult=2,
        seed=1,
    )
    c1, a1 = m_1d.fit(x_1d)
    assert c1.shape == (1, 1)
    assert a1 is not None and bool((a1 == 0).all())
    assert torch.isfinite(c1).all()

    # Case B: Zero-variance constant input & +1e4 shifted input
    for const_val in (0.0, 10_000.0):
        x_const = torch.full((256, 16), const_val, device=device, dtype=torch.float32)
        m_c = EMeansNoiseTrack(
            d=16,
            k=8,
            nepochs=2.0,
            base_mode=base_mode,
            head_edit=True,
            transplant_every=2,
            finalization="quad_merge",
            anvil_blend0=0.3,
            dust_sigma0=0.2,
            stratified_order=True,
            wide_step_mult=2,
            revival_mode="resample",
            seed=2,
        )
        cc, ac = m_c.fit(x_const)
        assert torch.isfinite(cc).all()
        assert ac is not None and ac.shape == (256,)
        assert exact_mse_fp64(x_const, cc, device=device) < 1e-6

    # Case C: Input dtypes (`bfloat16`, `float16`, `float64`, non-contiguous, numpy) + determinism
    gen = torch.Generator(device=device).manual_seed(88)
    x_base = torch.randn((512, 32), generator=gen, device=device, dtype=torch.float32)
    for dt in (torch.float32, torch.bfloat16, torch.float16, torch.float64):
        x_dt = x_base.to(dt)
        m_dt = EMeansNoiseTrack(
            d=32,
            k=24,
            nepochs=2.5,
            base_mode=base_mode,
            head_edit=True,
            transplant_every=2,
            finalization="quad_merge",
            anvil_blend0=0.25,
            dust_sigma0=0.15,
            stratified_order=True,
            wide_step_mult=2,
            seed=42,
        )
        c_dt, a_dt = m_dt.fit(x_dt)
        assert c_dt.dtype == torch.float32
        assert torch.isfinite(c_dt).all()
        assert a_dt is not None and a_dt.dtype == torch.int64
        assert m_dt.total_time_sec == pytest.approx(m_dt.init_time_sec + m_dt.train_time_sec)

    # Explicit `compute_dtype=torch.float64`
    m_fp64 = EMeansNoiseTrack(
        d=32,
        k=16,
        nepochs=2.0,
        base_mode=base_mode,
        head_edit=True,
        transplant_every=2,
        dust_sigma0=0.15,
        compute_dtype=torch.float64,
        seed=42,
    )
    c64, _ = m_fp64.fit(x_base.double())
    assert torch.isfinite(c64).all()

    # Non-contiguous slice & deterministic reproducibility (under deterministic CUDA reductions)
    x_noncontig = torch.randn((300, 64), generator=gen, device=device)[:, ::2]
    assert not x_noncontig.is_contiguous()
    m_r1 = EMeansNoiseTrack(d=32, k=16, nepochs=2.0, base_mode=base_mode, head_edit=True, anvil_blend0=0.2, seed=7)
    m_r2 = EMeansNoiseTrack(d=32, k=16, nepochs=2.0, base_mode=base_mode, head_edit=True, anvil_blend0=0.2, seed=7)
    prev_det = torch.are_deterministic_algorithms_enabled()
    prev_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        cr1, ar1 = m_r1.fit(x_noncontig)
        cr2, ar2 = m_r2.fit(x_noncontig)
    finally:
        torch.use_deterministic_algorithms(prev_det, warn_only=prev_warn)
    torch.testing.assert_close(cr1, cr2, rtol=0.0, atol=0.0)
    assert torch.equal(ar1, ar2)

    # NumPy round-trip + `.assign()`
    x_np = x_base.detach().cpu().numpy()
    m_np = EMeansNoiseTrack(d=32, k=16, nepochs=1.5, base_mode=base_mode, seed=9)
    c_np, a_np = m_np.fit(x_np)
    assert isinstance(c_np, np.ndarray) and isinstance(a_np, np.ndarray)
    d_np, l_np = m_np.assign(x_np[:20])
    assert isinstance(d_np, np.ndarray) and d_np.shape == (20,)
    assert isinstance(l_np, np.ndarray) and l_np.shape == (20,)


def test_phase2a_noise_runner_fleet_and_shard_validation(tmp_path: Path) -> None:
    # Verify 12-GPU / 6-computer fleet safety invariants (Board #23, #40, #47, #54)
    assert len(phase2a_runner.FLEET_SHARDS) == 12
    expected_fleet = {
        ("local", 0),
        ("local", 1),
        ("ripper", 0),
        ("ripper", 1),
        ("ripper", 2),
        ("ripper", 3),
        ("loki", 0),
        ("loki", 1),
        ("cashew.stratabaremetal.com", 0),
        ("cashew.stratabaremetal.com", 1),
        ("acorn.stratabaremetal.com", 1),
        ("work", 1),
    }
    actual_fleet = {(h, g) for h, g, _ in phase2a_runner.FLEET_SHARDS}
    assert actual_fleet == expected_fleet
    for host, _, sess in phase2a_runner.FLEET_SHARDS:
        assert "walnut" not in host
        assert sess.startswith("emeans_p2a_noise_")
        assert sess != "0"

    # Verify atomic JSON write & strict shard loader
    shards_dir = tmp_path / "shards"
    r0 = [{"cell_id": "c0", "seed": 1000, "arm": "emeans_pp_base", "test_mse": 0.5}]
    r1 = [{"cell_id": "c1", "seed": 1000, "arm": "emeans_pp_base", "test_mse": 0.6}]
    phase2a_runner._atomic_write_json(shards_dir / "records_shard_00_of_02.json", r0)
    # Missing shard 01 must raise FileNotFoundError
    with pytest.raises(FileNotFoundError):
        phase2a_runner.load_and_validate_shards(shards_dir, num_shards=2)

    phase2a_runner._atomic_write_json(shards_dir / "records_shard_01_of_02.json", r1)
    # Stale shard from another `num_shards` value must be ignored
    phase2a_runner._atomic_write_json(shards_dir / "records_shard_00_of_05.json", r0)
    merged = phase2a_runner.load_and_validate_shards(shards_dir, num_shards=2)
    assert len(merged) == 2

    # Duplicate `(cell_id, seed, arm)` must raise RuntimeError
    phase2a_runner._atomic_write_json(shards_dir / "records_shard_01_of_02.json", r0)
    with pytest.raises(RuntimeError, match="Duplicate"):
        phase2a_runner.load_and_validate_shards(shards_dir, num_shards=2)
