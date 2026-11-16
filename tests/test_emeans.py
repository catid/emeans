"""Comprehensive unit, numerical parity, and invariant tests for emeans and emeans++."""

from __future__ import annotations

import math

import faiss.contrib.e_means as faiss_emeans_mod
import numpy as np
import pytest
import torch
from faiss.contrib.e_means import NumpyEmeans
from faiss.contrib.e_means import base as faiss_base

from emeans import (
    ALPHA_LAW,
    ALPHA_MIN_FACTOR,
    BATCH_LAWS,
    CHECK_UNUSED_EVERY,
    DEFAULT_INIT_USAGE,
    DEFAULT_UNUSED_THRESHOLD,
    SPLIT_EPSILON,
    EMeans,
    EMeansPP,
    HAS_TRITON,
    LloydKMeans,
    NaiveKMeansPP_EMeans,
    SculleyMiniBatchKMeans,
    assign_points,
    block_greedy_kmeans_plusplus_torch,
    build_recipe,
    compute_epoch_rows,
    compute_step_law,
    deduplicate_splits,
    epoch_seed,
    exact_mse_fp64,
    greedy_kmeans_plusplus_torch,
    lightweight_coreset,
    load_cifar10_resnet18_embeddings,
    make_synthetic_dataset,
    prepare_represented_centroids,
    resolve_batch_law,
    resolve_unused_threshold,
    round_power_of_two,
    scheduled_alpha,
    seed_emeans_plusplus,
    voronoi_polish_and_priors,
)
from emeans.emeans import _separate_collapsed_splits
from emeans.emeans_pp import _nearest_other_centroid_sq_dists


# ---------------------------------------------------------------------------
# 1. Scaling Laws & Schedule Exact Parity with Official Faiss Reference
# ---------------------------------------------------------------------------
def test_scaling_laws_parity_with_faiss_ref() -> None:
    assert BATCH_LAWS == faiss_base._BATCH_LAWS
    assert ALPHA_LAW == faiss_base._ALPHA_LAW
    assert ALPHA_MIN_FACTOR == faiss_base._ALPHA_MIN_FACTOR
    assert CHECK_UNUSED_EVERY == faiss_base._CHECK_UNUSED_EVERY
    assert DEFAULT_INIT_USAGE == faiss_base._INIT_USAGE
    assert SPLIT_EPSILON == faiss_base._SPLIT_EPSILON
    assert DEFAULT_UNUSED_THRESHOLD == faiss_base._UNUSED_THRESHOLD

    for val in [0.1, 0.9, 1.0, 1.41, 1.42, 2.0, 33.0, 40464.0, 65535.9, 220202.0]:
        assert round_power_of_two(val) == faiss_base._round_power_of_two(val)

    configs = [
        (100_000, 2, 256, 3.0, "auto"),
        (100_000, 16, 512, 5.0, "auto"),
        (100_000, 17, 512, 5.0, "auto"),
        (1_000_000, 96, 16_384, 3.0, "throughput"),
        (1_000_000, 128, 4_096, 1.4, "quality"),
        (10_000_000, 256, 65_536, 2.6, "auto"),
        (50_000, 512, 1_024, 0.5, "auto"),
    ]
    for n, d, k, nepochs, law in configs:
        rec = build_recipe(n=n, d=d, k=k, nepochs=nepochs, batch_law=law)
        ref_law, _ = faiss_base._resolve_batch_law(law, d)
        ref_b, ref_a0 = faiss_base._resolve_recipe(n, k, nepochs, ref_law, None, None)
        ref_rows = tuple(faiss_base._epoch_rows(n, nepochs))
        ref_tau = faiss_base._resolve_unused_threshold(None, d)

        assert rec.batch_law == ref_law
        assert rec.batch_size == ref_b
        assert math.isclose(rec.alpha0, ref_a0, rel_tol=1e-12, abs_tol=1e-15)
        assert rec.epoch_rows == ref_rows
        assert rec.unused_threshold == ref_tau

        for s in [0, 1, rec.total_steps // 2, rec.total_steps - 1, rec.total_steps]:
            assert math.isclose(
                scheduled_alpha(rec.alpha0, s, rec.total_steps),
                faiss_base._scheduled_alpha(ref_a0, s, rec.total_steps),
                rel_tol=1e-12,
            )
        for ep in range(len(rec.epoch_rows)):
            assert epoch_seed(1234, ep) == faiss_base._epoch_seed(1234, ep)

    # Edge-case validation checks
    with pytest.raises(ValueError):
        resolve_unused_threshold(-0.1, 32)
    with pytest.raises(ValueError):
        resolve_unused_threshold(float("nan"), 32)
    with pytest.raises(ValueError):
        compute_epoch_rows(1000, 0.0)
    with pytest.raises(ValueError):
        build_recipe(n=10, d=8, k=20)


# ---------------------------------------------------------------------------
# 2. Assignment Kernels: Triton vs PyTorch vs Faiss vs FP64 & Norm Matching
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("d", [8, 32, 96, 129])
def test_assignment_parity_and_norm_matching(d: int) -> None:
    rng = np.random.default_rng(7)
    B, K = 517, 193
    x_np = rng.standard_normal((B, d), dtype=np.float32)
    c_np = rng.standard_normal((K, d), dtype=np.float32)

    # Exact FP64 reference
    diff64 = x_np.astype(np.float64)[:, None, :] - c_np.astype(np.float64)[None, :, :]
    d2_64 = (diff64 * diff64).sum(axis=2)
    ref_labels = d2_64.argmin(axis=1)
    ref_dists = d2_64.min(axis=1)

    x_cpu = torch.from_numpy(x_np)
    c_cpu = torch.from_numpy(c_np)
    d_faiss, l_faiss = assign_points(x_cpu, c_cpu, compute_dtype=torch.float32, backend="auto")
    d_torch_cpu, l_torch_cpu = assign_points(x_cpu, c_cpu, compute_dtype=torch.float32, backend="torch")
    assert torch.equal(l_faiss, torch.from_numpy(ref_labels))
    assert torch.equal(l_torch_cpu, torch.from_numpy(ref_labels))
    np.testing.assert_allclose(d_faiss.numpy(), ref_dists, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(d_torch_cpu.numpy(), ref_dists, rtol=1e-4, atol=1e-4)

    # Empty batch edge case on CPU and CUDA
    d_empty, l_empty = assign_points(x_cpu[:0], c_cpu)
    assert d_empty.shape == (0,) and l_empty.shape == (0,)

    # Translation invariance of exact_mse_fp64 under huge offset (+1e5 in FP64)
    mse_base = exact_mse_fp64(x_np, c_np, device="cpu")
    mse_shifted = exact_mse_fp64(
        x_np.astype(np.float64) + 1e5, c_np.astype(np.float64) + 1e5, device="cpu"
    )
    assert math.isclose(mse_base, float(ref_dists.mean()), rel_tol=1e-12)
    assert math.isclose(mse_base, mse_shifted, rel_tol=1e-9)

    if torch.cuda.is_available():
        x_gpu = x_cpu.cuda()
        c_gpu = c_cpu.cuda()

        # FP32 Triton vs Torch vs FP64
        d_tri32, l_tri32 = assign_points(x_gpu, c_gpu, compute_dtype=torch.float32, backend="triton")
        d_tor32, l_tor32 = assign_points(x_gpu, c_gpu, compute_dtype=torch.float32, backend="torch")
        assert torch.equal(l_tri32.cpu(), torch.from_numpy(ref_labels))
        assert torch.equal(l_tor32.cpu(), torch.from_numpy(ref_labels))
        np.testing.assert_allclose(d_tri32.cpu().numpy(), ref_dists, rtol=1e-4, atol=1e-4)

        # Appendix I.1 represented-norm consistency in BF16:
        # Assigning converted centroids to themselves MUST yield identity labels and ~0 FP32 residual!
        c_rep, c_norm = prepare_represented_centroids(c_gpu, torch.bfloat16)
        d_self, l_self = assign_points(
            c_rep, c_gpu, compute_dtype=torch.bfloat16, backend="triton", c_rep_cached=(c_rep, c_norm)
        )
        assert torch.equal(l_self, torch.arange(K, device="cuda"))
        assert float(d_self.max().item()) < 1e-4

        # BF16 Triton vs BF16 Torch exact match
        d_tri_bf, l_tri_bf = assign_points(x_gpu, c_gpu, compute_dtype=torch.bfloat16, backend="triton")
        d_tor_bf, l_tor_bf = assign_points(x_gpu, c_gpu, compute_dtype=torch.bfloat16, backend="torch")
        assert float((l_tri_bf == l_tor_bf).float().mean().item()) > 0.999
        assert float((d_tri_bf - d_tor_bf).abs().max().item()) < 1e-3

        # Float64 input tensor on CUDA edge case (both default BF16 and explicit FP64)
        d_f64_auto, l_f64_auto = assign_points(x_gpu.double(), c_gpu.double())
        assert d_f64_auto.dtype == torch.float32 and l_f64_auto.dtype == torch.int64
        d_f64_exact, l_f64_exact = assign_points(
            x_gpu.double(), c_gpu.double(), compute_dtype=torch.float64
        )
        assert torch.equal(l_f64_exact.cpu(), torch.from_numpy(ref_labels))


# ---------------------------------------------------------------------------
# 3. EMA Convex-Hull Invariant & Edge-Case Algebra (counts == 0, alpha == 1)
# ---------------------------------------------------------------------------
def test_ema_convex_hull_and_edge_algebra() -> None:
    rng = np.random.default_rng(99)
    x = rng.uniform(2.0, 5.0, size=(1024, 16)).astype(np.float32)
    init_c = rng.uniform(2.0, 5.0, size=(32, 16)).astype(np.float32)
    # Place one unreachable centroid outside the box; with revival="off", it must stay untouched!
    init_c[0] = 100.0

    model = EMeans(
        d=16,
        k=32,
        nepochs=3.0,
        revival_mode="off",
        finalization="none",
        compute_dtype=torch.float32,
        seed=42,
    )
    c_out, _ = model.fit(x, init_centroids=init_c)
    # Centroid 0 received 0 points -> must equal init_c[0] bit-for-bit
    np.testing.assert_allclose(c_out[0], init_c[0], rtol=0, atol=0)
    # Active centroids 1..31 must lie strictly inside [2.0, 5.0]^16 (Proposition 1 convex hull)
    assert np.all(c_out[1:] >= 2.0 - 1e-6) and np.all(c_out[1:] <= 5.0 + 1e-6)

    # Test alpha0 = 1.0, constant schedule (weight == 1.0 full replacement path) + slow_ema
    for fin in ("closing_mean", "slow_ema", "none"):
        m1 = EMeans(
            d=16,
            k=32,
            nepochs=2.0,
            alpha0=1.0,
            use_cosine_schedule=False,
            revival_mode="off",
            finalization=fin,
            slow_ema_mult=10.0,
            seed=42,
        )
        c1, _ = m1.fit(x, init_centroids=init_c)
        assert np.all(np.isfinite(c1))
        np.testing.assert_allclose(c1[0], init_c[0], rtol=0, atol=0)


# ---------------------------------------------------------------------------
# 4. Revival Donor Splitting & Collapsed Zero-Centroid Separation
# ---------------------------------------------------------------------------
def test_revival_split_separation_and_donor_sampling() -> None:
    for dt in (torch.float32, torch.bfloat16, torch.float16):
        donor = torch.zeros((4, 8), dtype=torch.float32)
        child = torch.zeros((4, 8), dtype=torch.float32)
        _separate_collapsed_splits(donor, child, dt)
        assert not bool((donor.to(dt) == child.to(dt)).all(dim=1).any())
        assert torch.isfinite(child).all()

    # Place half the initial centroids at 1e4 (dead!); split revival must bring them into the data!
    rng = np.random.default_rng(123)
    x = rng.standard_normal((4096, 32), dtype=np.float32)
    init_c = rng.standard_normal((64, 32), dtype=np.float32)
    init_c[32:] = 10_000.0

    m_rev = EMeans(
        d=32,
        k=64,
        nepochs=3.0,
        batch_size=256,
        revival_mode="split",
        seed=7,
    )
    c_rev, labels_rev = m_rev.fit(x, init_centroids=init_c)
    assert m_rev.n_splits >= 32
    # All revived centroids must now sit inside the data cloud, nowhere near 10,000!
    assert float(np.abs(c_rev).max()) < 10.0
    assert labels_rev is not None and (np.bincount(labels_rev, minlength=64) > 0).sum() >= 60


# ---------------------------------------------------------------------------
# 5. Parity Check Against Installed `faiss.contrib.e_means.NumpyEmeans`
# ---------------------------------------------------------------------------
def test_faiss_numpy_emeans_parity() -> None:
    ds = make_synthetic_dataset(
        "balanced_gmm", n_train=12_000, n_val=2_000, n_test=2_000, d=32, k=64, seed=11
    )

    # (a) Exact 1-step deterministic algebraic parity when given identical batch & no shuffle diff
    init_c = ds.train[:64].copy()
    f_one = NumpyEmeans(32, 64, nepochs=1, batch_size=12_000, seed=1234)
    f_one.begin(12_000, init_c)
    f_one.update(f_one.accumulate(ds.train))
    f_one.end_epoch()
    c_faiss_1 = f_one.finalize()

    m_one = EMeans(
        d=32, k=64, nepochs=1.0, batch_size=12_000, revival_mode="off",
        compute_dtype=torch.float32, seed=1234,
    )
    c_ours_1, _ = m_one.fit(ds.train, init_centroids=init_c)
    np.testing.assert_allclose(c_ours_1, c_faiss_1, rtol=1e-5, atol=1e-5)

    # (b) Full multi-epoch run parity across 3 seeds on CPU and CUDA
    faiss_test_mses = []
    ours_cpu_mses = []
    ours_gpu_mses = []
    for seed in (10, 20, 30):
        fm = NumpyEmeans(32, 64, nepochs=5, seed=seed)
        fc, _ = fm.train(ds.train)
        faiss_test_mses.append(exact_mse_fp64(ds.test, fc, device="cpu"))

        om = EMeans(d=32, k=64, nepochs=5.0, seed=seed)
        oc, _ = om.fit(ds.train)
        assert om.recipe is not None
        assert om.recipe.batch_size == fm.recipe["batch_size"]
        assert math.isclose(om.recipe.alpha0, fm.recipe["alpha0"], rel_tol=1e-12)
        ours_cpu_mses.append(exact_mse_fp64(ds.test, oc, device="cpu"))

        if torch.cuda.is_available():
            og = EMeans(d=32, k=64, nepochs=5.0, seed=seed)
            gc, _ = og.fit(torch.from_numpy(ds.train).cuda())
            ours_gpu_mses.append(exact_mse_fp64(ds.test, gc.cpu().numpy(), device="cpu"))

    rel_diff_cpu = abs(np.mean(ours_cpu_mses) - np.mean(faiss_test_mses)) / np.mean(faiss_test_mses)
    assert rel_diff_cpu < 0.02
    if torch.cuda.is_available():
        rel_diff_gpu = abs(np.mean(ours_gpu_mses) - np.mean(faiss_test_mses)) / np.mean(faiss_test_mses)
        assert rel_diff_gpu < 0.02


# ---------------------------------------------------------------------------
# 6. Seeding Invariants: Lightweight Coreset, Greedy/Block k-means++, Voronoi Polish
# ---------------------------------------------------------------------------
def test_seeding_invariants_and_voronoi_polish() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = make_synthetic_dataset(
        "balanced_gmm", n_train=20_000, n_val=2_000, n_test=2_000, d=64, k=64, seed=21
    )
    x_t = torch.from_numpy(ds.train).to(dev)

    # Lightweight coreset normalization & constant-input edge case
    x_sub, w_sub = lightweight_coreset(x_t, m=4096, seed=5)
    assert x_sub.shape == (4096, 64) and w_sub.shape == (4096,)
    assert bool((w_sub > 0).all()) and bool(torch.isfinite(w_sub).all())
    assert math.isclose(float(w_sub.sum().item()), 1.0, rel_tol=1e-5, abs_tol=1e-5)

    # Weighted coreset mean must closely match true dataset mean
    coreset_mean = (x_sub * w_sub[:, None]).sum(dim=0)
    true_mean = x_t.mean(dim=0)
    assert float((coreset_mean - true_mean).norm().item()) < 0.15

    # Degenerate identical-rows dataset must not divide by zero or NaN
    x_const = torch.full((500, 16), 7.5, device=dev)
    xc_sub, wc_sub = lightweight_coreset(x_const, m=100, seed=1)
    assert torch.isfinite(xc_sub).all() and math.isclose(float(wc_sub.sum().item()), 1.0, rel_tol=1e-5)
    c_deg = greedy_kmeans_plusplus_torch(x_const, k=8, seed=1)
    assert c_deg.shape == (8, 16) and torch.isfinite(c_deg).all()

    # Voronoi polish handles empty clusters safely without 0/0 NaNs AND slashes initial shell MSE
    raw_res = seed_emeans_plusplus(x_t, k=64, seeding_mode="coreset_exact", polish=False, seed=12)
    pol_res = seed_emeans_plusplus(x_t, k=64, seeding_mode="coreset_exact", polish=True, seed=12)
    mse_raw = exact_mse_fp64(ds.test, raw_res.centroids.cpu().numpy())
    mse_pol = exact_mse_fp64(ds.test, pol_res.centroids.cpu().numpy())
    # In d=64 Gaussian clusters, raw data points sit at ~2x intra-cluster MSE vs Voronoi means!
    assert mse_pol < 0.68 * mse_raw
    assert math.isclose(float(pol_res.prior_probs.sum().item()), 1.0, rel_tol=1e-5)

    # Empty-cluster safety in voronoi_polish_and_priors
    init_with_empty = pol_res.centroids.clone()
    init_with_empty[0] = 99999.0
    pol_empty = voronoi_polish_and_priors(x_sub, w_sub, init_with_empty)
    assert torch.isfinite(pol_empty.centroids).all()
    assert float(pol_empty.prior_probs[0].item()) == 0.0
    assert torch.allclose(pol_empty.centroids[0], init_with_empty[0])

    # Block greedy k-means++ for k > 128 + shifted coordinates (+5000)
    ds_big = make_synthetic_dataset(
        "balanced_gmm", n_train=16_000, n_val=1_000, n_test=1_000, d=32, k=200, seed=33
    )
    x_big = torch.from_numpy(ds_big.train + 5000.0).to(dev)
    c_blk = block_greedy_kmeans_plusplus_torch(x_big, k=200, seed=33)
    assert c_blk.shape == (200, 32) and torch.isfinite(c_blk).all()
    mse_blk = exact_mse_fp64(ds_big.test + 5000.0, c_blk.cpu().numpy())
    mse_rand = exact_mse_fp64(ds_big.test + 5000.0, ds_big.train[:200] + 5000.0)
    assert mse_blk < mse_rand
    w_big = torch.full((x_big.shape[0],), 1.0 / x_big.shape[0], dtype=torch.float32, device=dev)
    pol_blk = voronoi_polish_and_priors(x_big, w_big, c_blk)
    mse_blk_pol = exact_mse_fp64(ds_big.test + 5000.0, pol_blk.centroids.cpu().numpy())
    assert mse_blk_pol < 0.65 * mse_rand


# ---------------------------------------------------------------------------
# 7. Huygens Rare-Mode Protection & Self-Distance Exclusion (m != j)
# ---------------------------------------------------------------------------
def test_huygens_rare_cluster_protection() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # (a) Strict m != j self-distance exclusion check
    c_test = torch.tensor(
        [[0.0, 0.0], [3.0, 4.0], [10.0, 0.0], [3.0, 4.0]],
        dtype=torch.float32,
        device=dev,
    ) + 1e4  # large offset tests cancellation immunity too
    cand = torch.tensor([0, 1, 2, 3], dtype=torch.int64, device=dev)
    dnn2 = _nearest_other_centroid_sq_dists(c_test, cand).cpu().numpy()
    # c[0] nearest is c[1] at 3^2+4^2 = 25 (NOT 0 from m==0!)
    # c[1] and c[3] coincide (m=3 != 1), so their nearest OTHER centroid distance IS 0.0!
    # c[2] nearest is c[1] at 7^2+4^2 = 65 ( wait: (10-3)^2 + (0-4)^2 = 49 + 16 = 65 )
    np.testing.assert_allclose(dnn2, [25.0, 0.0, 65.0, 0.0], rtol=1e-5, atol=1e-5)

    # (b) Toy & multiscale rare-cluster assassination test:
    # Vanilla revival (N_j < 0.1 * mean(N)) assassinates genuine rare modes (p_r = 0.03 / k),
    # whereas EMeansPP with Huygens protection preserves 100% of them!
    ds = make_synthetic_dataset(
        "multiscale_rare_outliers",
        n_train=40_000,
        n_val=4_000,
        n_test=4_000,
        d=32,
        k=100,
        seed=77,
    )
    assert ds.true_centers is not None and ds.rare_mode_indices is not None
    rare_centers = ds.true_centers[ds.rare_mode_indices]  # (10, 32)
    x_tr = torch.from_numpy(ds.train).to(dev)

    naive = NaiveKMeansPP_EMeans(d=32, k=100, nepochs=5.0, batch_size=2048, seed=77)
    c_naive_t, _ = naive.fit(x_tr)
    c_naive = c_naive_t.cpu().numpy()

    epp = EMeansPP(d=32, k=100, nepochs=5.0, batch_size=2048, seed=77)
    c_epp_t, _ = epp.fit(x_tr)
    c_epp = c_epp_t.cpu().numpy()

    # Check how many true rare modes have a learned centroid within radius 2.0:
    d2_naive = ((rare_centers[:, None, :] - c_naive[None, :, :]) ** 2).sum(axis=2).min(axis=1)
    d2_epp = ((rare_centers[:, None, :] - c_epp[None, :, :]) ** 2).sum(axis=2).min(axis=1)
    recovered_naive = int((d2_naive < 4.0).sum())
    recovered_epp = int((d2_epp < 4.0).sum())

    assert recovered_epp == len(rare_centers), f"Expected all {len(rare_centers)} rare modes preserved, got {recovered_epp}"
    assert recovered_naive < len(rare_centers), "Naive count-only revival should lose rare modes"
    assert epp.n_protected > 0

    mse_oracle = exact_mse_fp64(ds.test, ds.true_centers)
    mse_naive = exact_mse_fp64(ds.test, c_naive)
    mse_epp = exact_mse_fp64(ds.test, c_epp)
    # EMeansPP reaches within ~2% of exact ground-truth oracle centers (~3.91),
    # slashing Naive's excess regret above the irreducible noise floor by >10x!
    assert mse_epp < 0.88 * mse_naive
    assert (mse_epp - mse_oracle) < 0.1 * (mse_naive - mse_oracle)


# ---------------------------------------------------------------------------
# 8. Baselines, All 7 Synthetic Distributions, & Zero Leakage on Real Embeddings
# ---------------------------------------------------------------------------
def test_baselines_and_zero_leakage_datasets() -> None:
    # Verify unit average coordinate variance (~1.0) on all 7 Appendix D.2 synthetic datasets
    for kind in [
        "uniform_cube",
        "isotropic_gaussian",
        "stretched_gaussian_0.5",
        "stretched_gaussian_1",
        "stretched_gaussian_2",
        "balanced_gmm",
        "zipf_gmm",
    ]:
        ds = make_synthetic_dataset(kind, n_train=20_000, n_val=1_000, n_test=1_000, d=32, k=64, seed=5)
        coord_var = float(ds.train.var(axis=0).mean())
        assert 0.85 <= coord_var <= 1.15, f"{kind} mean coordinate variance={coord_var}"

    # Verify deduplicate_splits removes exact & cross-split duplicates
    a = np.array([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    c = np.array([[1.0, 2.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    ac, bc, cc = deduplicate_splits(a, b, c)
    assert len(ac) == 2 and len(bc) == 1 and len(cc) == 1

    # Verify baselines run cleanly on GPU/CPU and decrease MSE
    ds_gmm = make_synthetic_dataset("balanced_gmm", n_train=6_000, n_val=1_000, n_test=1_000, d=32, k=32, seed=9)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_gmm = torch.from_numpy(ds_gmm.train).to(dev)

    lloyd_r = LloydKMeans(d=32, k=32, niter=10, init="random", seed=9)
    cr, _ = lloyd_r.fit(x_gmm)
    lloyd_pp = LloydKMeans(d=32, k=32, niter=10, init="k-means++", seed=9)
    cpp, _ = lloyd_pp.fit(x_gmm)
    assert exact_mse_fp64(ds_gmm.test, cpp.cpu().numpy()) <= exact_mse_fp64(ds_gmm.test, cr.cpu().numpy())

    sc_torch = SculleyMiniBatchKMeans(d=32, k=32, nepochs=3.0, batch_size=512, engine="torch", seed=9)
    cst, _ = sc_torch.fit(x_gmm)
    sc_sk = SculleyMiniBatchKMeans(d=32, k=32, nepochs=3.0, batch_size=512, engine="sklearn", seed=9)
    css, _ = sc_sk.fit(ds_gmm.train)
    assert torch.isfinite(cst).all() and np.all(np.isfinite(css))

    # Real CIFAR-10 ResNet-18 embeddings: verify shape, finiteness, and STRICT ZERO train/val/test leakage!
    real_ds = load_cifar10_resnet18_embeddings(mode="global_512")
    assert real_ds.d == 512
    assert real_ds.train.shape[0] > 45_000
    assert real_ds.val.shape[0] > 4_500
    assert real_ds.test.shape[0] > 4_500
    assert np.all(np.isfinite(real_ds.train))

    def _hash_set(arr: np.ndarray) -> set[bytes]:
        r = np.round(arr, decimals=5)
        r = np.where(r == 0.0, 0.0, r).astype(np.float32)
        return {row.tobytes() for row in r}

    s_tr = _hash_set(real_ds.train)
    s_va = _hash_set(real_ds.val)
    s_te = _hash_set(real_ds.test)
    assert len(s_tr & s_va) == 0, "Train/Val contamination detected!"
    assert len(s_tr & s_te) == 0, "Train/Test contamination detected!"
    assert len(s_va & s_te) == 0, "Val/Test contamination detected!"

    # Also pre-populate and verify zero leakage on `patches_128` real embeddings
    patch_ds = load_cifar10_resnet18_embeddings(mode="patches_128")
    assert patch_ds.d == 128 and patch_ds.train.shape[0] > 150_000
    p_tr = _hash_set(patch_ds.train)
    p_va = _hash_set(patch_ds.val)
    p_te = _hash_set(patch_ds.test)
    assert len(p_tr & p_va) == 0 and len(p_tr & p_te) == 0 and len(p_va & p_te) == 0

    # Deterministic seeding & training check:
    # (1) Raw GPU k-means++ selection & CPU full pipeline are bit-identical;
    # (2) GPU Voronoi polish / EMA `index_add_` match within FP32 atomicAdd rounding (~1e-6).
    s_raw_a = seed_emeans_plusplus(x_gmm, k=32, polish=False, seed=2026)
    s_raw_b = seed_emeans_plusplus(x_gmm, k=32, polish=False, seed=2026)
    assert torch.equal(s_raw_a.centroids, s_raw_b.centroids)

    s_a = seed_emeans_plusplus(x_gmm, k=32, polish=True, seed=2026)
    s_b = seed_emeans_plusplus(x_gmm, k=32, polish=True, seed=2026)
    assert torch.allclose(s_a.centroids, s_b.centroids, rtol=1e-6, atol=1e-6)

    epp_cpu_a = EMeansPP(d=32, k=32, nepochs=2.0, seed=2026)
    epp_cpu_b = EMeansPP(d=32, k=32, nepochs=2.0, seed=2026)
    cca, lca = epp_cpu_a.fit(ds_gmm.train)
    ccb, lcb = epp_cpu_b.fit(ds_gmm.train)
    assert np.array_equal(cca, ccb) and np.array_equal(lca, lcb)

    epp_a = EMeansPP(d=32, k=32, nepochs=2.0, seed=2026)
    epp_b = EMeansPP(d=32, k=32, nepochs=2.0, seed=2026)
    ca, la = epp_a.fit(x_gmm)
    cb, lb = epp_b.fit(x_gmm)
    assert torch.allclose(ca, cb, rtol=1e-5, atol=1e-5)
    assert la is not None and lb is not None and torch.equal(la, lb)

