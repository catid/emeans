"""Comprehensive mathematical, numerical, edge-case, and parity unit tests for Phase 2 (`algo_extensions.py`)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from emeans.algo_extensions import (
    HYPOTHESES,
    EMeansAlgoExt,
    compute_directional_split,
    compute_huygens_nn_sq_dists,
    empirical_bayes_finalize,
    hartigan_assign_points,
    power_mean_top2_weights,
    reset_post_split_oja_axes,
    scheduled_nesterov_mu,
    separate_collapsed_directional,
    top2_assign_chunked,
    update_coord_variance_ema,
    update_nesterov_velocity,
    update_oja_principal_axes,
)
from emeans.emeans import EMeans
from emeans.emeans_pp import EMeansPP
from emeans.kernels import exact_mse_fp64, prepare_represented_centroids


CUDA_DEVICES = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


# ---------------------------------------------------------------------------
# 1. Registry & Hypothesis Specification Completeness
# ---------------------------------------------------------------------------


def test_hypotheses_registry_complete() -> None:
    expected_ids = {"H-Algo-1", "H-Algo-2", "H-Algo-3", "H-Algo-4", "H-Algo-5"}
    assert set(HYPOTHESES.keys()) == expected_ids
    for hid, spec in HYPOTHESES.items():
        assert spec.id == hid
        assert len(spec.name) > 10
        assert len(spec.mechanism) > 30
        assert len(spec.equation) > 15
        assert len(spec.predicted_win_regime) > 20
        assert len(spec.predicted_fail_regime) > 20
        assert "Falsified if" in spec.falsification_criterion


# ---------------------------------------------------------------------------
# 2. Exact Bit-Identity Parity with Core `EMeans` & `EMeansPP`
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dev", CUDA_DEVICES)
@pytest.mark.parametrize("finalization", ["closing_mean", "slow_ema", "none"])
def test_exact_parity_with_base_emeans(dev: str, finalization: str) -> None:
    rng = np.random.default_rng(7)
    x_np = rng.standard_normal((2048, 32), dtype=np.float32)
    x_t = torch.from_numpy(x_np).to(dev)

    ref = EMeans(
        d=32,
        k=64,
        nepochs=3.0,
        batch_size=256,
        finalization=finalization,
        seed=99,
    )
    ext = EMeansAlgoExt(
        d=32,
        k=64,
        nepochs=3.0,
        base_mode="emeans",
        batch_size=256,
        finalization=finalization,
        seed=99,
    )
    c_ref, a_ref = ref.fit(x_t)
    c_ext, a_ext = ext.fit(x_t)

    assert ext.n_splits == ref.n_splits
    assert torch.equal(a_ref, a_ext)
    if dev == "cpu":
        torch.testing.assert_close(c_ext, c_ref, rtol=0.0, atol=0.0)
    else:
        # CUDA `index_add_` uses parallel FP32 `atomicAdd` (~1-2 FP32 ULP scheduling variance)
        torch.testing.assert_close(c_ext, c_ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dev", CUDA_DEVICES)
@pytest.mark.parametrize("finalization", ["closing_mean", "slow_ema"])
def test_exact_parity_with_emeans_pp(dev: str, finalization: str) -> None:
    rng = np.random.default_rng(19)
    # Imbalanced Gaussian mixture with rare modes so revival + Huygens protection both exercise
    centers = rng.standard_normal((48, 32)).astype(np.float32) * 2.5
    probs = 1.0 / np.arange(1, 49, dtype=np.float64)
    probs /= probs.sum()
    comp = rng.choice(48, size=3000, p=probs)
    x_np = (centers[comp] + 0.35 * rng.standard_normal((3000, 32))).astype(np.float32)
    x_t = torch.from_numpy(x_np).to(dev)

    ref_pp = EMeansPP(
        d=32,
        k=48,
        nepochs=3.0,
        batch_size=256,
        finalization=finalization,
        seed=321,
    )
    ext_pp = EMeansAlgoExt(
        d=32,
        k=48,
        nepochs=3.0,
        base_mode="emeans_pp",
        batch_size=256,
        finalization=finalization,
        seed=321,
    )
    c_ref, a_ref = ref_pp.fit(x_t)
    c_ext, a_ext = ext_pp.fit(x_t)

    assert ext_pp.n_splits == ref_pp.n_splits
    assert ext_pp.n_protected == ref_pp.n_protected
    assert torch.equal(a_ref, a_ext)
    if dev == "cpu":
        torch.testing.assert_close(c_ext, c_ref, rtol=0.0, atol=0.0)
    else:
        torch.testing.assert_close(c_ext, c_ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("split_mode", ["checkerboard", "oja", "var_axis", "bf16_random"])
def test_eb_shrinkage_preserves_identical_training_trajectory(split_mode: str) -> None:
    """Verify `finalization='eb_shrinkage'` never perturbs RNG or main-centroid trajectories."""
    rng = np.random.default_rng(101)
    x = torch.from_numpy(rng.standard_normal((1500, 24), dtype=np.float32))

    m_close = EMeansAlgoExt(
        d=24, k=48, nepochs=3.0, base_mode="emeans", batch_size=128,
        split_mode=split_mode, finalization="closing_mean", seed=55,
    )
    m_eb = EMeansAlgoExt(
        d=24, k=48, nepochs=3.0, base_mode="emeans", batch_size=128,
        split_mode=split_mode, finalization="eb_shrinkage", seed=55,
    )
    c_close, a_close = m_close.fit(x)
    _, a_eb = m_eb.fit(x)

    assert m_close.n_splits == m_eb.n_splits
    assert torch.equal(a_close, a_eb)
    assert m_eb.closing_centroids is not None
    torch.testing.assert_close(m_eb.closing_centroids, c_close, rtol=0.0, atol=0.0)


# ---------------------------------------------------------------------------
# 3. H-Algo-1: Online Oja Principal-Axis Donor Splitting Tests
# ---------------------------------------------------------------------------


def test_oja_principal_axis_recovers_rotated_top_eigenvector() -> None:
    """Verify online Oja power iteration converges to the exact non-axis-aligned top eigenvector."""
    gen = torch.Generator()
    gen.manual_seed(42)
    d = 64
    # Arbitrary rotated unit direction u*
    raw_u = torch.randn((d,), generator=gen)
    u_star = raw_u / raw_u.norm()

    # Cluster 0 has rank-1 spike variance 25.0 along u* + isotropic noise 0.04
    # Expected principal std = sqrt(25.0 + 0.04) = 5.004
    centroids = torch.zeros((2, d), dtype=torch.float32)
    prin_vec = 1e-3 * torch.randn((2, d), generator=gen)
    labels = torch.zeros((512,), dtype=torch.int64)
    counts = torch.tensor([512.0, 0.0], dtype=torch.float32)
    ema_w = torch.tensor([0.35, 0.0], dtype=torch.float32)

    p1_before = prin_vec[1].clone()
    for _ in range(25):
        scores_1d = 5.0 * torch.randn((512, 1), generator=gen)
        iso_noise = 0.2 * torch.randn((512, d), generator=gen)
        x_batch = scores_1d * u_star[None, :] + iso_noise
        prin_vec = update_oja_principal_axes(
            prin_vec, x_batch, centroids, labels, counts, ema_w
        )

    # Untouched cluster 1 must remain completely unchanged
    torch.testing.assert_close(prin_vec[1], p1_before, rtol=0.0, atol=0.0)

    # Cluster 0 direction must align with +-u_star (cosine > 0.995) and std ~ 5.0
    est_sigma = float(prin_vec[0].norm().item())
    est_dir = prin_vec[0] / prin_vec[0].norm()
    cos_sim = abs(float(torch.dot(est_dir, u_star).item()))
    assert cos_sim > 0.995, f"Expected cosine > 0.995, got {cos_sim:.5f}"
    assert 4.4 < est_sigma < 5.6, f"Expected principal std ~ 5.0, got {est_sigma:.4f}"


def test_bf16_directional_split_non_collapse_and_zero_safety() -> None:
    """Verify Oja directional split survives BF16 rounding across coordinates and handles zero donors."""
    d = 64
    donors = torch.ones((2, d), dtype=torch.float32)
    donors[1] = 0.0  # Exact zero-origin donor edge case!

    u_dir = torch.zeros((2, d), dtype=torch.float32)
    u_dir[0] = 1.0 / math.sqrt(d)
    u_dir[1, 17] = -1.0  # Dominant coordinate 17 with negative sign
    prin_vec = torch.zeros((2, d), dtype=torch.float32)
    prin_vec[0] = 0.5 * u_dir[0]  # sigma_1 = 0.5
    prin_vec[1] = 0.0             # Degenerate zero-variance donor!

    # Test 1: explicit precomputed_dirs on zero donor nudges exact dominant coordinate 17 negatively
    d_cent, c_cent, out_dirs = compute_directional_split(
        donors=donors,
        source_idx=torch.tensor([0, 1]),
        split_mode="oja",
        split_epsilon=1e-4,
        split_std_frac=0.1,
        compute_dtype=torch.bfloat16,
        prin_vec=prin_vec,
        precomputed_dirs=u_dir,
    )
    assert bool((d_cent[0].bfloat16() != c_cent[0].bfloat16()).all())
    assert not bool((d_cent[1].bfloat16() == c_cent[1].bfloat16()).all())
    assert float(c_cent[1, 17].item()) < 0.0
    assert torch.isfinite(d_cent).all() and torch.isfinite(c_cent).all()
    torch.testing.assert_close(out_dirs.norm(dim=1), torch.tensor([1.0, 1.0]))

    # Test 2: when `precomputed_dirs=None` and `prin_vec[1] == 0`, fallback produces valid unit vector e_0!
    d_fb, c_fb, dirs_fb = compute_directional_split(
        donors=donors,
        source_idx=torch.tensor([0, 1]),
        split_mode="oja",
        split_epsilon=1e-4,
        split_std_frac=0.1,
        compute_dtype=torch.bfloat16,
        prin_vec=prin_vec,
    )
    torch.testing.assert_close(dirs_fb.norm(dim=1), torch.tensor([1.0, 1.0]))
    assert not bool((d_fb[1].bfloat16() == c_fb[1].bfloat16()).all())


def test_reset_post_split_oja_axes_orthogonality() -> None:
    gen = torch.Generator().manual_seed(11)
    prin_vec = torch.randn((6, 16), generator=gen)
    src = torch.tensor([0, 2])
    tgt = torch.tensor([1, 5])
    used = prin_vec[src] / prin_vec[src].norm(dim=1, keepdim=True)
    old_norms = prin_vec[src].norm(dim=1)

    reset_post_split_oja_axes(prin_vec, src, tgt, used, generator=gen)
    dots_src = (prin_vec[src] * used).sum(dim=1).abs()
    dots_tgt = (prin_vec[tgt] * used).sum(dim=1).abs()
    assert float(dots_src.max().item()) < 1e-5
    assert float(dots_tgt.max().item()) < 1e-5
    torch.testing.assert_close(prin_vec[src].norm(dim=1), 0.5 * old_norms, rtol=1e-5, atol=1e-5)


def test_oja_split_beats_checkerboard_on_merged_rotated_bimodal_donor() -> None:
    """Controlled geometric test: 1 live centroid covers 2 merged modes along a rotated axis + 1 dead centroid."""
    rng = np.random.default_rng(2026)
    d = 128
    # Direction supported on coords 4..67 with [1, 1, -1, -1] pattern:
    # strictly orthogonal to BOTH e_0 (BF16 ULP fallback) AND checkerboard [+1, -1, +1, -1, ...]!
    u = np.zeros(d, dtype=np.float32)
    u[4:68] = 1.0
    u[6:68:4] = -1.0
    u[7:68:4] = -1.0
    u /= np.linalg.norm(u)

    # Two tight modes centered at 10 + 2*u and 10 - 2*u
    n = 4096
    half = n // 2
    c_base = np.full((1, d), 10.0, dtype=np.float32)
    x1 = c_base + 2.0 * u[None, :] + 0.15 * rng.standard_normal((half, d)).astype(np.float32)
    x2 = c_base - 2.0 * u[None, :] + 0.15 * rng.standard_normal((half, d)).astype(np.float32)
    x = np.concatenate([x1, x2], axis=0)

    # Initial centroids: centroid 0 at midpoint 10.0 (captures 100% of points), centroid 1 far away at -50.0 (dead!)
    init_c = np.vstack([c_base, np.full((1, d), -50.0, dtype=np.float32)])

    # Run 6 batches (revival fires at step 5 -> 1 post-split step) in BF16 without closing overwrite
    m_chk = EMeansAlgoExt(
        d=d, k=2, nepochs=1.0, base_mode="emeans", batch_size=683,
        split_mode="checkerboard", finalization="none", compute_dtype=torch.bfloat16, seed=1,
    )
    m_rnd = EMeansAlgoExt(
        d=d, k=2, nepochs=1.0, base_mode="emeans", batch_size=683,
        split_mode="bf16_random", finalization="none", compute_dtype=torch.bfloat16, seed=1,
    )
    m_oja = EMeansAlgoExt(
        d=d, k=2, nepochs=1.0, base_mode="emeans", batch_size=683,
        split_mode="oja", finalization="none", compute_dtype=torch.bfloat16, seed=1,
    )
    c_chk, _ = m_chk.fit(x, init_centroids=init_c)
    c_rnd, _ = m_rnd.fit(x, init_centroids=init_c)
    c_oja, _ = m_oja.fit(x, init_centroids=init_c)

    mse_chk = exact_mse_fp64(x, c_chk, device="cpu")
    mse_rnd = exact_mse_fp64(x, c_rnd, device="cpu")
    mse_oja = exact_mse_fp64(x, c_oja, device="cpu")
    assert mse_oja < mse_rnd < mse_chk, (
        f"Expected Oja ({mse_oja:.3f}) < random-dir ({mse_rnd:.3f}) < checkerboard ({mse_chk:.3f})"
    )
    assert mse_oja < mse_chk * 0.80


# ---------------------------------------------------------------------------
# 4. H-Algo-2: Hartigan Size-Corrected Marginal Distortion Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dev", CUDA_DEVICES)
def test_hartigan_assign_matches_bruteforce_and_is_translation_invariant(dev: str) -> None:
    gen = torch.Generator(device=dev).manual_seed(77)
    N, K, D = 97, 23, 32
    x = torch.randn((N, D), generator=gen, device=dev, dtype=torch.float32)
    centroids = torch.randn((K, D), generator=gen, device=dev, dtype=torch.float32)
    # Highly skewed usage across 3 orders of magnitude
    usage = torch.exp(torch.linspace(-3.0, 4.0, K, device=dev))
    n_total = 2000
    beta = 1.5

    dists, labels = hartigan_assign_points(
        x,
        centroids,
        usage=usage,
        n_total=n_total,
        beta=beta,
        compute_dtype=torch.float32,
        centroid_chunk_size=7,
        row_chunk_size=19,
    )

    # Brute-force reference in FP64
    n_hat = (n_total * (usage.double() / usage.double().sum())).clamp_min(0.5)
    w_hart = n_hat / (n_hat + beta)
    raw_d2_ref = torch.cdist(x.double(), centroids.double(), p=2.0).square()  # (N, K)
    weighted_ref = raw_d2_ref * w_hart[None, :]
    ref_labels = weighted_ref.argmin(dim=1)
    ref_unweighted_d2 = raw_d2_ref.gather(1, ref_labels[:, None]).squeeze(1).float()

    assert torch.equal(labels, ref_labels)
    torch.testing.assert_close(dists, ref_unweighted_d2, rtol=1e-4, atol=1e-4)

    # Verify smaller cluster pulls equidistant boundary point!
    c_two = torch.tensor([[-1.0, 0.0], [1.0, 0.0]], device=dev)
    x_mid = torch.tensor([[0.0, 0.0]], device=dev)  # Equidistant (`d^2 = 1.0` to both)
    u_skew = torch.tensor([5.0, 500.0], device=dev)  # Cluster 0 is 100x smaller -> 5/6 < 500/501!
    _, l_mid = hartigan_assign_points(
        x_mid, c_two, usage=u_skew, n_total=505, beta=1.0, compute_dtype=torch.float32
    )
    assert int(l_mid[0].item()) == 0

    # Translation invariance test (`x + shift`, `centroids + shift`)
    shift = torch.linspace(50.0, -50.0, D, device=dev)[None, :]
    _, labels_shifted = hartigan_assign_points(
        x + shift,
        centroids + shift,
        usage=usage,
        n_total=n_total,
        beta=beta,
        compute_dtype=torch.float32,
        centroid_chunk_size=7,
        row_chunk_size=19,
    )
    assert torch.equal(labels, labels_shifted)


# ---------------------------------------------------------------------------
# 5. H-Algo-3: Coherence-Gated Nesterov Centroid Lookahead Tests
# ---------------------------------------------------------------------------


def test_nesterov_coherence_gate_accelerates_drift_and_zeros_oscillation() -> None:
    k, d = 2, 8
    vel = torch.zeros((k, d), dtype=torch.float32)
    c0 = torch.zeros((k, d), dtype=torch.float32)
    c1 = torch.ones((k, d), dtype=torch.float32)
    touched = torch.tensor([True, True])

    # Step 1: initial move from zero velocity -> initialized cleanly for both rows
    vel = update_nesterov_velocity(vel, c0, c1, touched, beta_vel=0.7, coherence_gate=True)
    assert bool((vel > 0.29).all())

    # Step 2: Cluster 0 continues in +1 direction (coherent drift); Cluster 1 reverses in -1 direction (jitter)!
    c2 = c1.clone()
    c2[0] += 1.0   # cos = +1.0 -> gate = 1.0
    c2[1] -= 1.0   # cos = -1.0 -> gate = 0.0!
    vel2 = update_nesterov_velocity(vel, c1, c2, touched, beta_vel=0.7, coherence_gate=True)

    assert float(vel2[0].norm().item()) > float(vel[0].norm().item())
    assert float(vel2[1].norm().item()) == 0.0

    # Verify schedule hits exact 0 at stop_frac
    assert scheduled_nesterov_mu(0.4, step=0, total_steps=100, stop_frac=0.85) == pytest.approx(0.4)
    assert scheduled_nesterov_mu(0.4, step=85, total_steps=100, stop_frac=0.85) == 0.0
    assert scheduled_nesterov_mu(0.4, step=99, total_steps=100, stop_frac=0.85) == 0.0


def test_nesterov_preserves_convex_hull_invariance() -> None:
    rng = np.random.default_rng(12)
    # Points strictly inside [2.0, 5.0]^d
    x = rng.uniform(2.0, 5.0, size=(1024, 16)).astype(np.float32)
    model = EMeansAlgoExt(
        d=16, k=32, nepochs=2.0, base_mode="emeans",
        nesterov_mu0=0.6, revival_mode="off", finalization="none", seed=4,
    )
    c, _ = model.fit(x)
    assert float(c.min()) >= 2.0 - 1e-5
    assert float(c.max()) <= 5.0 + 1e-5


# ---------------------------------------------------------------------------
# 6. H-Algo-4: Scale-Invariant Top-2 Power-Mean Soft EMA Warmup Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dev", CUDA_DEVICES)
def test_top2_assign_chunked_exact_against_bruteforce_and_edge_cases(dev: str) -> None:
    gen = torch.Generator(device=dev).manual_seed(123)
    N, K, D = 53, 17, 24  # Co-prime sizes stressing tile boundaries
    x = torch.randn((N, D), generator=gen, device=dev, dtype=torch.float32)
    centroids = torch.randn((K, D), generator=gen, device=dev, dtype=torch.float32)

    d1, l1, d2, l2 = top2_assign_chunked(
        x, centroids, compute_dtype=torch.float32, centroid_chunk_size=5, row_chunk_size=11
    )
    assert bool((l1 != l2).all())
    assert bool((d1 <= d2 + 1e-6).all())

    c_rep, c_norm = prepare_represented_centroids(centroids, torch.float32)
    full_d2 = (x.square().sum(dim=1, keepdim=True) + c_norm[None, :] - 2.0 * (x @ c_rep.t())).clamp_min(0.0)
    ref_vals, ref_idx = torch.topk(full_d2, k=2, dim=1, largest=False)

    assert torch.equal(l1, ref_idx[:, 0])
    assert torch.equal(l2, ref_idx[:, 1])
    torch.testing.assert_close(d1, ref_vals[:, 0], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(d2, ref_vals[:, 1], rtol=1e-4, atol=1e-4)

    # Edge cases: K=1 and N=0
    d1_k1, l1_k1, d2_k1, _ = top2_assign_chunked(x[:5], centroids[:1], compute_dtype=torch.float32)
    assert bool((l1_k1 == 0).all())
    assert bool(torch.isinf(d2_k1).all())
    w1_k1, w2_k1 = power_mean_top2_weights(d1_k1, d2_k1, step=0, warmup_steps=10, s0=-2.0)
    assert bool((w1_k1 == 1.0).all()) and bool((w2_k1 == 0.0).all())

    d1_e, l1_e, d2_e, l2_e = top2_assign_chunked(x[:0], centroids, compute_dtype=torch.float32)
    assert d1_e.numel() == 0 and l2_e.numel() == 0


def test_power_mean_weights_scale_invariance_and_annealing() -> None:
    d1 = torch.tensor([1.0, 0.25, 0.01, 2.0], dtype=torch.float32)
    d2 = torch.tensor([1.0, 1.00, 4.00, 3.0], dtype=torch.float32)

    w1_a, w2_a = power_mean_top2_weights(d1, d2, step=2, warmup_steps=10, s0=-2.0)
    # Multiply all squared distances by 1,000,000 -> Power k-means weights must be invariant!
    w1_b, w2_b = power_mean_top2_weights(d1 * 1e6, d2 * 1e6, step=2, warmup_steps=10, s0=-2.0)

    torch.testing.assert_close(w1_a, w1_b, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(w2_a, w2_b, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(w1_a + w2_a, torch.ones_like(d1), rtol=1e-6, atol=1e-6)
    # Equidistant boundary point (d1 == d2) gets 50/50 split!
    assert float(w2_a[0].item()) == pytest.approx(0.5, abs=1e-6)
    # Monotonic annealing: later step gives strictly smaller secondary weight on non-tied points
    _, w2_late = power_mean_top2_weights(d1, d2, step=8, warmup_steps=10, s0=-2.0)
    assert float(w2_late[1].item()) < float(w2_a[1].item())
    # Post-warmup gives exact hard (1.0, 0.0)
    w1_end, w2_end = power_mean_top2_weights(d1, d2, step=10, warmup_steps=10, s0=-2.0)
    assert bool((w1_end == 1.0).all()) and bool((w2_end == 0.0).all())


# ---------------------------------------------------------------------------
# 7. H-Algo-5: Empirical-Bayes / James-Stein Shrinkage Finalization Tests
# ---------------------------------------------------------------------------


def test_empirical_bayes_finalize_dominates_both_closing_and_slow_ema_in_quadratic_risk() -> None:
    """Monte-Carlo proof of H-Algo-5: Zipf cluster counts + mixed settled & split-lagged anchors."""
    gen = torch.Generator().manual_seed(2026)
    K, D = 128, 64
    sigma2_true = 1.0  # E||x - mu_j||^2 = 1.0 -> per-coord variance = 1 / D

    true_mu = torch.randn((K, D), generator=gen) * 2.0
    # Heavy-tailed Zipf counts from n_j = 3 up to n_j = 600
    ranks = torch.arange(1, K + 1, dtype=torch.float64)
    counts = (600.0 / ranks).round().clamp_min(3.0)

    # Noisy closing sample means: E||mu_close - mu_true||^2 = sigma2_true / n_j
    noise_std = (sigma2_true / (counts.float() * D)).sqrt()[:, None]
    closing_means = true_mu + torch.randn((K, D), generator=gen) * noise_std

    # Slow EMA anchors:
    # - 85% of clusters are settled (anchor has 5x lower variance than 3-sample closing mean)
    # - 15% of clusters (including populous ones!) suffered severe split/drift lag (bias norm ~ 1.2)
    slow_centroids = true_mu + torch.randn((K, D), generator=gen) * math.sqrt(0.03 / D)
    lagged_idx = torch.arange(0, K, 7)
    slow_centroids[lagged_idx] += 1.2 / math.sqrt(D)

    pass_sse = counts * sigma2_true
    c_eb, lam, diag = empirical_bayes_finalize(
        closing_means=closing_means,
        slow_centroids=slow_centroids,
        pass_counts=counts,
        pass_sse=pass_sse,
        d=D,
        kappa="auto",
    )

    risk_close = float((closing_means - true_mu).square().sum(dim=1).mean().item())
    risk_slow = float((slow_centroids - true_mu).square().sum(dim=1).mean().item())
    risk_eb = float((c_eb - true_mu).square().sum(dim=1).mean().item())

    assert risk_eb < risk_close * 0.75, f"Expected EB risk ({risk_eb:.5f}) << closing ({risk_close:.5f})"
    assert risk_eb < risk_slow * 0.50, f"Expected EB risk ({risk_eb:.5f}) << lagged slow ({risk_slow:.5f})"
    # Lagged clusters must automatically get high lambda (> 0.90) rejecting stale slow_ema!
    assert float(lam[lagged_idx].mean().item()) > 0.92
    # Scale invariance under X -> 100 * X
    c_eb_scaled, lam_scaled, _ = empirical_bayes_finalize(
        closing_means=closing_means * 100.0,
        slow_centroids=slow_centroids * 100.0,
        pass_counts=counts,
        pass_sse=pass_sse * 10000.0,
        d=D,
        kappa="auto",
    )
    torch.testing.assert_close(lam, lam_scaled, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(c_eb * 100.0, c_eb_scaled, rtol=1e-4, atol=1e-4)


def test_empirical_bayes_edge_cases_empty_and_zero_sse() -> None:
    c_close = torch.randn((8, 16))
    c_slow = torch.randn((8, 16))
    # Case 1: all counts == 0 -> returns slow_centroids cleanly
    out0, lam0, _ = empirical_bayes_finalize(
        c_close, c_slow, torch.zeros(8, dtype=torch.float64), torch.zeros(8, dtype=torch.float64), d=16
    )
    torch.testing.assert_close(out0, c_slow)
    assert bool((lam0 == 0.0).all())

    # Case 2: exact zero SSE & zero diff -> returns closing_means with lambda=1 on nonempty, 0 on empty
    counts = torch.tensor([5.0, 0.0, 1.0, 10.0, 0.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    out_exact, lam_exact, _ = empirical_bayes_finalize(
        c_close, c_close.clone(), counts, torch.zeros(8, dtype=torch.float64), d=16
    )
    assert torch.isfinite(out_exact).all()
    assert float(lam_exact[0].item()) == 1.0 and float(lam_exact[1].item()) == 0.0


# ---------------------------------------------------------------------------
# 8. Huygens Self-Distance Exclusion & End-to-End Integration Tests
# ---------------------------------------------------------------------------


def test_huygens_nn_sq_dists_excludes_self_and_handles_large_offset() -> None:
    # Distinct centroids shifted by +1e5 to test catastrophic-cancellation immunity AND m != j masking
    base = torch.tensor([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]], dtype=torch.float32) + 100_000.0
    cand = torch.tensor([0, 1, 2])
    d_nn2 = compute_huygens_nn_sq_dists(base, cand)
    # Nearest neighbor of [0,0] is [3,4] at d^2 = 25; [3,4] -> 25; [6,8] -> [3,4] at d^2 = 25!
    torch.testing.assert_close(d_nn2, torch.tensor([25.0, 25.0, 25.0]), rtol=1e-4, atol=1e-3)
    assert bool((d_nn2 > 0.0).all())


@pytest.mark.parametrize("dev", CUDA_DEVICES)
def test_end_to_end_all_extensions_combined_deterministic_and_finite(dev: str) -> None:
    import os

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    prev_det = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        rng = np.random.default_rng(888)
        x_np = rng.standard_normal((2500, 32), dtype=np.float32)
        x_in = torch.from_numpy(x_np).to(dev)

        kwargs = dict(
            d=32,
            k=64,
            nepochs=2.5,  # Fractional epoch budget!
            base_mode="emeans_pp",
            split_mode="oja",
            hartigan_beta0=1.0,
            nesterov_mu0=0.35,
            power_warmup_frac=0.25,
            finalization="eb_shrinkage",
            seed=42,
        )
        m1 = EMeansAlgoExt(**kwargs)
        m2 = EMeansAlgoExt(**kwargs)
        c1, a1 = m1.fit(x_in)
        c2, a2 = m2.fit(x_in)

        assert torch.isfinite(c1).all()
        assert bool((a1 >= 0).all()) and bool((a1 < 64).all())
        assert torch.equal(a1, a2)
        torch.testing.assert_close(c1, c2, rtol=0.0, atol=0.0)

        d_eval, l_eval = m1.assign(x_in[:100])
        assert torch.isfinite(d_eval).all() and d_eval.shape == (100,) and l_eval.shape == (100,)
    finally:
        torch.use_deterministic_algorithms(prev_det)


# ---------------------------------------------------------------------------
# 9. Benchmark Harness & Parallel Shard / Fleet Dispatch Safety Tests
# ---------------------------------------------------------------------------


def test_phase2_runner_fleet_and_shard_validation(tmp_path: pytest.TempPathFactory) -> None:
    import importlib.util
    import json
    from pathlib import Path

    import sys

    spec = importlib.util.spec_from_file_location(
        "run_phase2_algo",
        Path(__file__).resolve().parents[1] / "scripts" / "run_phase2_algo.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    # Verify output path and 4-computer fleet safety invariants
    assert str(mod.OUTPUT_DIR).endswith("outputs/phase2_algo/classical_tracks")
    assert mod.FLEET_SHARDS == (
        ("local", 1, "emeans_p2_algo_s0"),
        ("loki", 0, "emeans_p2_algo_s1"),
        ("cashew.stratabaremetal.com", 0, "emeans_p2_algo_s2"),
        ("acorn.stratabaremetal.com", 1, "emeans_p2_algo_s3"),
    )
    for host, _, sess in mod.FLEET_SHARDS:
        assert "walnut" not in host
        assert sess.startswith("emeans_") and sess != "0"

    # Verify shard loader ignores stale num_shards files, rejects missing shards, and rejects duplicates
    sdir = Path(tmp_path) / "shards"
    sdir.mkdir(parents=True)
    # Stale 1-shard file must be completely ignored when loading num_shards=2!
    (sdir / "records_shard_00_of_01.json").write_text(
        json.dumps([{"cell_id": "stale", "seed": 0, "arm": "x"}]), encoding="utf-8"
    )
    (sdir / "records_shard_00_of_02.json").write_text(
        json.dumps([{"cell_id": "c0", "seed": 1000, "arm": "a0"}]), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="incomplete shards"):
        mod.load_and_validate_shards(sdir, num_shards=2)

    (sdir / "records_shard_01_of_02.json").write_text(
        json.dumps([{"cell_id": "c1", "seed": 1000, "arm": "a0"}]), encoding="utf-8"
    )
    merged = mod.load_and_validate_shards(sdir, num_shards=2)
    assert len(merged) == 2 and {r["cell_id"] for r in merged} == {"c0", "c1"}

    # Duplicate key detection
    (sdir / "records_shard_01_of_02.json").write_text(
        json.dumps([{"cell_id": "c0", "seed": 1000, "arm": "a0"}]), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="Duplicate"):
        mod.load_and_validate_shards(sdir, num_shards=2)


# ---------------------------------------------------------------------------
# 10. Adversarial Regression Tests (`@bug-hunter` Audit #29)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dev", CUDA_DEVICES)
def test_float64_inputs_and_compute_dtype_regression(dev: str) -> None:
    """Verify `float64` CUDA/CPU inputs & `compute_dtype=torch.float64` never crash in addmm."""
    gen = torch.Generator(device=dev).manual_seed(17)
    x64 = torch.randn((256, 32), generator=gen, device=dev, dtype=torch.float64)

    # Power top-2 warmup + Hartigan on float64 input tensor (compute_dtype=None -> bf16 on CUDA)
    m_auto = EMeansAlgoExt(
        d=32, k=16, nepochs=2.0, power_warmup_frac=0.5, hartigan_beta0=1.0, seed=5
    )
    c_auto, a_auto = m_auto.fit(x64)
    assert c_auto.dtype == torch.float32 and torch.isfinite(c_auto).all()
    assert a_auto.shape == (256,)

    # Explicit `compute_dtype=torch.float64`
    m_f64 = EMeansAlgoExt(
        d=32,
        k=16,
        nepochs=2.0,
        power_warmup_frac=0.5,
        hartigan_beta0=1.0,
        compute_dtype=torch.float64,
        seed=5,
    )
    c_f64, _ = m_f64.fit(x64)
    assert c_f64.dtype == torch.float32 and torch.isfinite(c_f64).all()


@pytest.mark.parametrize("dev", CUDA_DEVICES)
def test_emeans_pp_explicit_init_centroids_and_recipe_alpha0_parity(dev: str) -> None:
    """Verify `EMeansAlgoExt(base_mode='emeans_pp')` with explicit `init_centroids` matches `EMeansPP`."""
    rng = np.random.default_rng(707)
    x_np = rng.standard_normal((1024, 32), dtype=np.float32)
    x = torch.from_numpy(x_np).to(dev)
    init_c = x[:32].clone()

    ref = EMeansPP(d=32, k=32, nepochs=2.0, batch_size=128, seed=7)
    ext = EMeansAlgoExt(d=32, k=32, nepochs=2.0, base_mode="emeans_pp", batch_size=128, seed=7)
    c_ref, a_ref = ref.fit(x, init_centroids=init_c)
    c_ext, a_ext = ext.fit(x, init_centroids=init_c)

    assert ext.recipe is not None and ref.recipe is not None
    assert ext.recipe.alpha0 == pytest.approx(ref.recipe.alpha0, rel=0.0, abs=0.0)
    assert ext.n_splits == ref.n_splits
    assert ext.n_protected == ref.n_protected
    assert torch.equal(a_ref, a_ext)
    if dev == "cpu":
        torch.testing.assert_close(c_ext, c_ref, rtol=0.0, atol=0.0)
    else:
        torch.testing.assert_close(c_ext, c_ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("split_mode", ["oja", "var_axis"])
def test_resample_revival_resets_auxiliary_state_cleanly(split_mode: str) -> None:
    """Verify `revival_mode='resample'` resets `ema_sse`, `survival_mass`, `prin_vec`, and `coord_var`."""
    rng = np.random.default_rng(404)
    x = torch.from_numpy(rng.standard_normal((1024, 24), dtype=np.float32))
    # Force half the initial centroids far out at +1000 so they immediately trigger resample revival
    init_c = x[:32].clone()
    init_c[16:] = 1000.0

    m = EMeansAlgoExt(
        d=24,
        k=32,
        nepochs=2.0,
        base_mode="emeans_pp",
        batch_size=128,
        revival_mode="resample",
        huygens_protect=False,
        split_mode=split_mode,
        nesterov_mu0=0.3,
        seed=13,
    )
    c, a = m.fit(x, init_centroids=init_c)
    assert m.n_splits >= 16
    assert torch.isfinite(c).all()
    assert float(c.abs().max().item()) < 50.0
    assert bool((a >= 0).all())
