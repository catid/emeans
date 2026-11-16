"""Comprehensive unit, zero-leakage, Triton parity, shift-invariance, and PSNR tests for `emeans.semdedup_coreset`."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from emeans.apex_emeans import ApexEMeans
from emeans.semdedup_coreset import (
    DEFAULT_SEMDEDUP_DATA_DIR,
    FOUR_LABELED_BENCHMARKS,
    LABELED_SPEC_BY_NAME,
    LabeledVectorDataset,
    MEGAPIXEL_SPEC_BY_NAME,
    MegapixelVQDataset,
    TWO_MEGAPIXEL_VQ_BENCHMARKS,
    _deduplicate_labeled_two_way,
    _semdedup_scores_torch_fallback,
    centered_assign_fp32,
    compute_semdedup_scores_gpu,
    compute_vq_psnr_fp64,
    embed_labeled_text_zero_leakage,
    encode_vq_indices_gpu,
    evaluate_cholesky_ridge_gpu,
    evaluate_multinomial_logistic_gpu,
    evaluate_prototype_knn_gpu,
    evaluate_weighted_knn_gpu,
    load_labeled_benchmark,
    load_megapixel_vq_benchmark,
    reconstruct_patch_mosaic_rgb,
    select_coreset_at_budget,
    select_prototype_coreset,
    select_random_coreset,
    validate_labeled_dataset,
    validate_megapixel_vq_dataset,
    warmup_semdedup_kernels,
)


# ==============================================================================
# 1. Registry, Deduplication & Data-Leakage Rejection Tests
# ==============================================================================


def test_registries_and_error_guards() -> None:
    assert len(FOUR_LABELED_BENCHMARKS) == 4
    assert {s.name for s in FOUR_LABELED_BENCHMARKS} == {
        "agnews_256",
        "cifar10_resnet18_512",
        "cifar10_convnext_768",
        "covtype_54",
    }
    for s in FOUR_LABELED_BENCHMARKS:
        assert LABELED_SPEC_BY_NAME[s.name] == s

    assert len(TWO_MEGAPIXEL_VQ_BENCHMARKS) == 2
    assert {s.name for s in TWO_MEGAPIXEL_VQ_BENCHMARKS} == {
        "megapixel_rgb_3d",
        "megapixel_patch_48d",
    }
    for s in TWO_MEGAPIXEL_VQ_BENCHMARKS:
        assert MEGAPIXEL_SPEC_BY_NAME[s.name] == s
        assert s.k_values == (256, 1024, 4096)

    with pytest.raises(ValueError, match="Unknown labeled benchmark"):
        load_labeled_benchmark("nonexistent_labeled_xyz")
    with pytest.raises(ValueError, match="Unknown Megapixel VQ benchmark"):
        load_megapixel_vq_benchmark("nonexistent_vq_xyz")


def test_deduplicate_labeled_two_way_preserves_labels_and_rejects_leaks() -> None:
    rng = np.random.default_rng(101)
    x_tr = rng.standard_normal((120, 19)).astype(np.float32)
    y_tr = (np.arange(120, dtype=np.int64) % 5)
    x_tr[0, :] = 0.0
    x_tr[15] = x_tr[4]
    y_tr[15] = 4  # duplicate row must drop along with its label, keeping index 4's label!
    x_tr[30] = np.round(x_tr[30], 6)

    x_te = rng.standard_normal((45, 19)).astype(np.float32)
    y_te = (np.arange(45, dtype=np.int64) % 5)
    x_te[0, :] = -0.0  # -0.0 must collide with +0.0 in train
    x_te[7] = x_tr[30] + 1e-7
    x_te[12] = x_tr[50]

    xt_c, yt_c, xe_c, ye_c = _deduplicate_labeled_two_way(x_tr, y_tr, x_te, y_te, decimals=6)
    assert xt_c.shape == (119, 19)
    assert yt_c.shape == (119,)
    assert xe_c.shape == (42, 19)
    assert ye_c.shape == (42,)
    assert yt_c[4] == (4 % 5)

    ds = LabeledVectorDataset("test_clean", "Clean", "Synthetic", xt_c, yt_c, xe_c, ye_c, 19, 5, 16)
    validate_labeled_dataset(ds, check_cross_split_disjoint=True)

    # Injecting a leaked row into x_test must raise DATA LEAKAGE
    xe_bad = xe_c.copy()
    xe_bad[3] = xt_c[10]
    ds_bad = LabeledVectorDataset("bad", "Bad", "Synthetic", xt_c, yt_c, xe_bad, ye_c, 19, 5, 16)
    with pytest.raises(ValueError, match="DATA LEAKAGE"):
        validate_labeled_dataset(ds_bad, check_cross_split_disjoint=True)


def test_embed_labeled_text_zero_train_test_leakage() -> None:
    stems = [
        "superconducting quantum qubit error correction syndrome",
        "monetary policy treasury yield curve central bank",
        "striker goal champions league stadium match",
        "blackwell tensor core triton fp8 kernel optimization",
    ]
    rng = np.random.default_rng(202)
    toks = [f"w{i}" for i in range(150)]

    tr_docs, tr_y = [], []
    for i in range(240):
        c = i % 4
        extra = " ".join(rng.choice(toks, size=7, replace=False))
        tr_docs.append(f"{stems[c]} {extra} trdoc{i}")
        tr_y.append(c)

    te_docs_a, te_y_a = [], []
    for i in range(40):
        c = i % 4
        extra = " ".join(rng.choice(toks, size=7, replace=False))
        te_docs_a.append(f"{stems[c]} {extra} tedoc{i}")
        te_y_a.append(c)

    # Inject case/whitespace leaked train doc into test_a -> must be purged cleanly
    te_docs_leaked = te_docs_a + ["  " + tr_docs[5].upper() + " "]
    te_y_leaked = np.asarray(te_y_a + [1], dtype=np.int64)
    y_tr_arr = np.asarray(tr_y, dtype=np.int64)

    xtr_a, ytr_a, xte_a, yte_a, _ = embed_labeled_text_zero_leakage(
        tr_docs, y_tr_arr, te_docs_leaked, te_y_leaked, d=24, max_features=2000, min_df=2, svd_seed=99
    )
    assert xtr_a.shape == (240, 24)
    assert xte_a.shape == (40, 24)

    # Append wildly skewed test documents -> xtr and original 40 xte rows must be BITWISE IDENTICAL!
    te_docs_b = te_docs_a + [f"{stems[0]} {stems[0]} w1 w2 extra_adv_{j}" for j in range(30)]
    te_y_b = np.asarray(te_y_a + [0] * 30, dtype=np.int64)
    xtr_b, ytr_b, xte_b, _, _ = embed_labeled_text_zero_leakage(
        tr_docs, y_tr_arr, te_docs_b, te_y_b, d=24, max_features=2000, min_df=2, svd_seed=99
    )
    np.testing.assert_array_equal(xtr_a, xtr_b)
    np.testing.assert_array_equal(ytr_a, ytr_b)
    np.testing.assert_array_equal(xte_a, xte_b[:40])


# ==============================================================================
# 2. Triton `_semdedup_margin_kernel` Parity (`FULL_D=True` & `FULL_D=False`) & Edge Cases
# ==============================================================================


@pytest.mark.parametrize("d", [3, 53, 256, 768, 1031])
def test_triton_semdedup_kernel_matches_torch_across_prime_and_wide_dims(d: int) -> None:
    """Verify both `FULL_D=True` (`d <= 1024`) and `FULL_D=False` (`d = 1031`) match PyTorch reference."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(500 + d)
    n, k = 1031, 67  # prime N and prime K!
    x = torch.from_numpy(rng.standard_normal((n, d)).astype(np.float32)).to(dev)
    c = x[:k].clone().contiguous()
    # Force dead clusters at the end of `c` to verify empty-cluster safety
    c[-5:] = 1.0e5

    state = compute_semdedup_scores_gpu(x, c, k=k, window_size=24)
    assert state.empty_clusters >= 4
    assert torch.isfinite(state.scores).all()

    # Compare against exact PyTorch fallback on centered coordinates
    origin64 = x[: min(n, 8192)].double().mean(dim=0, keepdim=True)
    xc = (x.double() - origin64).float().contiguous()
    cc = (c.double() - origin64).float().contiguous()
    sq_d, lbl = centered_assign_fp32(xc, cc, origin=torch.zeros((1, d), device=dev))
    counts = torch.bincount(lbl, minlength=k)
    starts = torch.cumsum(counts, dim=0) - counts
    c_sse = torch.zeros((k,), device=dev, dtype=torch.float64).index_add_(0, lbl, sq_d.double())
    c_rms = torch.sqrt((c_sse / counts.clamp_min(1).double()).clamp_min(1e-24)).float()
    ref_scores = _semdedup_scores_torch_fallback(
        xc, state.sorted_idx, state.sorted_lbl, starts, counts, c_rms, window_size=24
    )
    rel_err = ((state.scores - ref_scores).abs() / ref_scores.abs().clamp_min(1e-6)).max().item()
    assert rel_err < 1e-5

    # Verify Horvitz-Thompson weight conservation & medoid coverage across budgets
    n_active = k - state.empty_clusters
    for frac in (0.10, 0.20, 0.50):
        sel = select_coreset_at_budget(state, retention_frac=frac)
        expected_m = int(round(frac * n))
        assert sel.num_selected == expected_m
        assert sel.indices.numel() == expected_m
        assert torch.unique(sel.indices).numel() == expected_m
        assert abs(float(sel.weights.double().sum().item()) - float(n)) < 1e-2
        if expected_m >= n_active:
            assert sel.active_clusters_covered == n_active

    proto = select_prototype_coreset(state)
    assert proto.num_selected == n_active
    assert abs(float(proto.weights.double().sum().item()) - float(n)) < 1e-3


# ==============================================================================
# 3. Coordinate-Shift Invariance (`+20,000`) & Strict Classifier Zero-Leakage Proofs
# ==============================================================================


def test_coordinate_shift_invariance_and_zero_test_batch_coupling() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(707)
    n_tr, n_te, d, k, num_cls = 1200, 180, 54, 48, 6

    centers = rng.standard_normal((num_cls, d)).astype(np.float32) * 2.2
    y_tr_np = rng.integers(0, num_cls, size=n_tr, dtype=np.int64)
    y_te_np = rng.integers(0, num_cls, size=n_te, dtype=np.int64)
    x_tr_np = centers[y_tr_np] + rng.standard_normal((n_tr, d)).astype(np.float32) * 0.7
    x_te_np = centers[y_te_np] + rng.standard_normal((n_te, d)).astype(np.float32) * 0.7
    # Add Covertype-like binary flags in columns 10..53
    x_tr_np[:, 10:] = (x_tr_np[:, 10:] > 0.0).astype(np.float32)
    x_te_np[:, 10:] = (x_te_np[:, 10:] > 0.0).astype(np.float32)

    # Quantize synthetic coordinates to multiples of 1/256 so `+ 20_000.0` in float32 is lossless (Sterbenz-exact)!
    x_tr = torch.round(torch.from_numpy(x_tr_np).to(dev) * 256.0) / 256.0
    y_tr = torch.from_numpy(y_tr_np).to(dev)
    x_te = torch.round(torch.from_numpy(x_te_np).to(dev) * 256.0) / 256.0
    y_te = torch.from_numpy(y_te_np).to(dev)
    c_base = x_tr[:k].clone().contiguous()

    shift = torch.full((1, d), 20_000.0, device=dev, dtype=torch.float32)
    st_0 = compute_semdedup_scores_gpu(x_tr, c_base, k=k)
    st_s = compute_semdedup_scores_gpu(x_tr + shift, c_base + shift, k=k)
    sel_0 = select_coreset_at_budget(st_0, retention_frac=0.20)
    sel_s = select_coreset_at_budget(st_s, retention_frac=0.20)
    assert torch.equal(sel_0.indices, sel_s.indices)
    assert torch.allclose(sel_0.weights, sel_s.weights, rtol=1e-6, atol=1e-6)

    # Poison ALL non-selected training labels with -999999 to prove 0 discarded-label leakage!
    y_tr_poisoned = torch.full_like(y_tr, -999_999)
    y_tr_poisoned[sel_0.indices] = y_tr[sel_0.indices]
    y_sub = y_tr_poisoned[sel_0.indices]

    x_sub_0 = x_tr[sel_0.indices]
    x_sub_s = (x_tr + shift)[sel_s.indices]

    r0 = evaluate_cholesky_ridge_gpu(x_sub_0, y_sub, sel_0.weights, x_te, y_te, num_cls)
    rs = evaluate_cholesky_ridge_gpu(x_sub_s, y_sub, sel_s.weights, x_te + shift, y_te, num_cls)
    assert abs(r0["accuracy_pct"] - rs["accuracy_pct"]) < 1e-6

    l0 = evaluate_multinomial_logistic_gpu(x_sub_0, y_sub, sel_0.weights, x_te, y_te, num_cls)
    ls = evaluate_multinomial_logistic_gpu(x_sub_s, y_sub, sel_s.weights, x_te + shift, y_te, num_cls)
    assert abs(l0["accuracy_pct"] - ls["accuracy_pct"]) < 1e-5

    k0 = evaluate_weighted_knn_gpu(x_sub_0, y_sub, sel_0.weights, x_te, y_te, num_cls)
    ks = evaluate_weighted_knn_gpu(x_sub_s, y_sub, sel_s.weights, x_te + shift, y_te, num_cls)
    assert abs(k0["accuracy_pct"] - ks["accuracy_pct"]) < 1e-6

    p0 = evaluate_prototype_knn_gpu(x_sub_0, y_sub, sel_0.weights, sel_0.cluster_ids, k, x_te, y_te, num_cls)
    ps = evaluate_prototype_knn_gpu(x_sub_s, y_sub, sel_s.weights, sel_s.cluster_ids, k, x_te + shift, y_te, num_cls)
    assert abs(p0["accuracy_pct"] - ps["accuracy_pct"]) < 1e-6

    # Zero Test-Batch Coupling Proof (#204): corrupting x_te[20:] with +1.0e25 inside a FULL BATCH
    # must leave predictions on x_te[:20] 100% unchanged!
    x_te_corrupt = x_te.clone()
    x_te_corrupt[20:] = 1.0e25
    # Setting y_te_masked[20:] = -999 ensures ONLY indices 0..19 can match in the full-batch call!
    y_te_masked = y_te.clone()
    y_te_masked[20:] = -999
    for fn_eval in (
        lambda xe, ye: evaluate_cholesky_ridge_gpu(x_sub_0, y_sub, sel_0.weights, xe, ye, num_cls)["accuracy_pct"],
        lambda xe, ye: evaluate_multinomial_logistic_gpu(x_sub_0, y_sub, sel_0.weights, xe, ye, num_cls)["accuracy_pct"],
        lambda xe, ye: evaluate_weighted_knn_gpu(x_sub_0, y_sub, sel_0.weights, xe, ye, num_cls)["accuracy_pct"],
        lambda xe, ye: evaluate_prototype_knn_gpu(x_sub_0, y_sub, sel_0.weights, sel_0.cluster_ids, k, xe, ye, num_cls)["accuracy_pct"],
    ):
        acc_slice = fn_eval(x_te[:20], y_te[:20])
        acc_full_corrupt_first20 = fn_eval(x_te_corrupt, y_te_masked) * (float(n_te) / 20.0)
        assert abs(acc_full_corrupt_first20 - acc_slice) < 1e-6
        hits = 0
        for qi in range(20):
            hits += 1 if fn_eval(x_te_corrupt[qi : qi + 1], y_te[qi : qi + 1]) > 50.0 else 0
        assert abs(100.0 * hits / 20.0 - acc_slice) < 1e-6


# ==============================================================================
# 4. Megapixel VQ Exact FP64 PSNR & 2D Image Reconstruction Parity
# ==============================================================================


@pytest.mark.parametrize("patch_hw,d", [((1, 1), 3), ((4, 4), 48)])
def test_megapixel_vq_psnr_matches_full_2d_image_reconstruction(patch_hw: tuple[int, int], d: int) -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(909 + d)
    ph, pw = patch_hw
    h, w = 64, 96
    img_orig = rng.uniform(0.0, 255.0, size=(h, w, 3)).astype(np.float32)

    patches = (
        img_orig.reshape(h // ph, ph, w // pw, pw, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(-1, d)
    )
    patches_t = torch.from_numpy(np.ascontiguousarray(patches)).to(dev)
    solver = ApexEMeans(d=d, k=32, nepochs=2.0, seed=123)
    centroids, _ = solver.fit(patches_t, capture_assignments=False)

    psnr_db, pix_mse, labels = compute_vq_psnr_fp64(patches_t, centroids, peak_val=255.0, device=dev)
    recon_img = reconstruct_patch_mosaic_rgb(labels, centroids, mosaic_hw=(h, w), patch_h=ph, patch_w=pw)
    assert recon_img.shape == (h, w, 3)

    ref_pix_mse = float(np.mean((img_orig.astype(np.float64) - recon_img) ** 2))
    ref_psnr_db = 10.0 * np.log10((255.0 ** 2) / ref_pix_mse)
    assert abs(pix_mse - ref_pix_mse) < 1e-9
    assert abs(psnr_db - ref_psnr_db) < 1e-10

    codes_enc, enc_sec = encode_vq_indices_gpu(patches_t, centroids)
    assert codes_enc.shape == (patches_t.shape[0],)
    assert enc_sec >= 0.0
    # #211 regression guard: a rogue 1e25 row in x_gpu must NEVER alter VQ codes of clean rows!
    patches_rogue = patches_t.clone()
    patches_rogue[-1] = 1.0e25
    codes_rogue, _ = encode_vq_indices_gpu(patches_rogue, centroids)
    assert torch.equal(codes_enc[:-1], codes_rogue[:-1])


# ==============================================================================
# 5. End-to-End Staging & Integrity Verification of All 6 Real Datasets
# ==============================================================================


def test_all_real_labeled_and_megapixel_datasets_load_and_validate() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    warmup_semdedup_kernels(dev, d=64, k=128)

    for spec in FOUR_LABELED_BENCHMARKS:
        ds = load_labeled_benchmark(spec.name, data_dir=DEFAULT_SEMDEDUP_DATA_DIR, device=dev)
        validate_labeled_dataset(ds, check_cross_split_disjoint=True)
        assert ds.d == spec.d
        assert ds.num_classes == spec.num_classes
        assert ds.x_train.shape[0] >= 45_000
        assert ds.x_test.shape[0] >= 7_000

    for vspec in TWO_MEGAPIXEL_VQ_BENCHMARKS:
        vds = load_megapixel_vq_benchmark(vspec.name, data_dir=DEFAULT_SEMDEDUP_DATA_DIR, device=dev)
        validate_megapixel_vq_dataset(vds, check_patch_disjoint=True)
        assert vds.x_train.shape == (vspec.n_train, vspec.d)
        assert vds.x_test.shape == (vspec.n_test, vspec.d)

    # Verify zero leftover `.tmp_*` files in data cache dir and verify `select_random_coreset`
    assert list(DEFAULT_SEMDEDUP_DATA_DIR.glob(".tmp_*")) == []
    rc = select_random_coreset(1000, retention_frac=0.10, seed=17, device=dev)
    assert rc.num_selected == 100
    assert abs(float(rc.weights.sum().item()) - 1000.0) < 1e-3


@pytest.mark.parametrize("case", ["zeros", "constants", "tiny_1e25", "huge_1e25", "k_eq_1", "strided_bf16"])
def test_adversarial_numerical_and_strided_dtype_fuzz(case: str) -> None:
    """Fuzz SemDeDup + all 4 downstream classifiers across degenerate scales, K=1, strided slices & dtypes."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(1337)
    n, n_te, d, num_cls = 257, 41, 37, 4
    base_x = torch.from_numpy(rng.standard_normal((n * 2, d)).astype(np.float32)).to(dev)
    x_te = torch.from_numpy(rng.standard_normal((n_te, d)).astype(np.float32)).to(dev)
    y_tr = torch.from_numpy(rng.integers(0, num_cls, size=n, dtype=np.int64)).to(dev)
    y_te = torch.from_numpy(rng.integers(0, num_cls, size=n_te, dtype=np.int64)).to(dev)

    k = 1 if case == "k_eq_1" else 19
    if case == "zeros":
        x_tr = torch.zeros((n, d), device=dev, dtype=torch.float32)
        x_te = torch.zeros_like(x_te)
    elif case == "constants":
        x_tr = torch.full((n, d), 42.5, device=dev, dtype=torch.float32)
        x_te = torch.full_like(x_te, 42.5)
    elif case == "tiny_1e25":
        x_tr = (base_x[:n].double() * 1.0e-25).to(torch.float64)
        x_te = (x_te.double() * 1.0e-25).to(torch.float64)
    elif case == "huge_1e25":
        x_tr = base_x[:n].clone()
        x_tr[-1] = 1.0e25  # tail-row extreme outlier!
        x_te = x_te * 1.0e25
    elif case == "strided_bf16":
        x_tr = base_x.to(torch.bfloat16)[::2]  # non-contiguous row-strided bf16 view!
        assert not x_tr.is_contiguous()
        x_te = x_te.to(torch.bfloat16)
    else:
        x_tr = base_x[:n]

    c = x_tr[:k].clone()
    st = compute_semdedup_scores_gpu(x_tr, c, k=k)
    assert torch.isfinite(st.scores).all()
    sel = select_coreset_at_budget(st, retention_frac=0.25)
    assert torch.isfinite(sel.weights).all()

    xs, ys, ws = x_tr[sel.indices], y_tr[sel.indices], sel.weights
    for res in (
        evaluate_cholesky_ridge_gpu(xs, ys, ws, x_te, y_te, num_cls),
        evaluate_multinomial_logistic_gpu(xs, ys, ws, x_te, y_te, num_cls, max_iter=15),
        evaluate_weighted_knn_gpu(xs, ys, ws, x_te, y_te, num_cls),
        evaluate_prototype_knn_gpu(xs, ys, ws, sel.cluster_ids, k, x_te, y_te, num_cls),
    ):
        assert math.isfinite(res["accuracy_pct"])
        assert 0.0 <= res["accuracy_pct"] <= 100.0


def test_compute_semdedup_scores_honors_explicit_assign_labels() -> None:
    """Verify `compute_semdedup_scores_gpu` honors explicit `assign_labels` / `assign_sq_dists` and strided origin."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(2026)
    n, d, k = 128, 32, 8
    x = torch.from_numpy(rng.standard_normal((n, d)).astype(np.float32)).to(dev)
    c = x[:k].clone().contiguous()

    forced_lbl = torch.zeros((n,), device=dev, dtype=torch.int64)
    st_forced = compute_semdedup_scores_gpu(x, c, k=k, assign_labels=forced_lbl)
    assert int(st_forced.counts[0].item()) == n
    assert int(st_forced.empty_clusters) == k - 1
    assert torch.isfinite(st_forced.scores).all()

    forced_sq = (x - c[0:1]).square().sum(dim=1)
    st_forced_sq = compute_semdedup_scores_gpu(
        x, c, k=k, assign_labels=forced_lbl, assign_sq_dists=forced_sq
    )
    assert torch.allclose(st_forced.scores, st_forced_sq.scores, rtol=1e-5, atol=1e-5)

    with pytest.raises(ValueError, match="out of bounds"):
        bad_lbl = forced_lbl.clone()
        bad_lbl[0] = k
        compute_semdedup_scores_gpu(x, c, k=k, assign_labels=bad_lbl)

    # Verify strided odd-median origin (`n = 6000 > 4095`) preserves exact +20,000 shift equivariance
    x_big = torch.round(torch.from_numpy(rng.standard_normal((6000, 8)).astype(np.float32)).to(dev) * 256.0) / 256.0
    c_big = x_big[:16].clone().contiguous()
    shift = torch.full((1, 8), 20_000.0, device=dev, dtype=torch.float32)
    st_b0 = compute_semdedup_scores_gpu(x_big, c_big, k=16)
    st_bs = compute_semdedup_scores_gpu(x_big + shift, c_big + shift, k=16)
    assert torch.equal(st_b0.sorted_idx, st_bs.sorted_idx)
    assert torch.allclose(st_b0.scores, st_bs.scores, rtol=1e-5, atol=1e-5)


def test_runner_fleet_safety_and_gitignore_contracts() -> None:
    """Verify `walnut`, `ripper`, `loki`, `work`, and `tmux 0` are hard-blocked and `.npz` caches are git-ignored."""
    import importlib.util
    import subprocess
    import sys

    spec = importlib.util.spec_from_file_location(
        "run_semdedup_benchmarks",
        Path(__file__).resolve().parents[1] / "scripts" / "run_semdedup_benchmarks.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_semdedup_benchmarks"] = mod
    spec.loader.exec_module(mod)

    shards = mod.build_fleet_shards(local_gpus=(1, 3))
    assert len(shards) == 4
    assert all(s.host in {"acorn.stratabaremetal.com", "local"} for s in shards)
    for bad_host in ("walnut.stratabaremetal.com", "walnut", "ripper", "loki", "work"):
        with pytest.raises(RuntimeError, match="SAFETY VIOLATION"):
            mod.assert_safe_fleet_target(bad_host, "emeans_semdedup_s00")
    with pytest.raises(RuntimeError, match="SAFETY VIOLATION"):
        mod.assert_safe_fleet_target("acorn.stratabaremetal.com", "0")
    src_txt = (Path(__file__).resolve().parents[1] / "scripts" / "run_semdedup_benchmarks.py").read_text(encoding="utf-8")
    assert "/tmp/triton_cache" not in src_txt
    assert "/var/tmp" not in src_txt

    # Verify git check-ignore on staged `.npz` caches vs deliverable paths
    repo = Path(__file__).resolve().parents[1]
    chk_cache = subprocess.run(
        ["git", "check-ignore", "-v", "outputs/semdedup/data/semdedup_covtype_54_s20261116_v1.npz"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=False,
    )
    assert chk_cache.returncode == 0  # git-ignored == True
    for tracked_rel in (
        "outputs/semdedup/semdedup_results.json",
        "outputs/semdedup/semdedup_summary.md",
        "outputs/semdedup/semdedup_accuracy_vs_time.png",
        "outputs/semdedup/plots/megapixel_vq_psnr_pareto.png",
        "docs/semdedup_coreset.md",
    ):
        chk_tr = subprocess.run(
            ["git", "check-ignore", "-v", tracked_rel],
            cwd=str(repo),
            capture_output=True,
            text=True,
            check=False,
        )
        assert chk_tr.returncode != 0, f"{tracked_rel} must NOT be git-ignored!"


