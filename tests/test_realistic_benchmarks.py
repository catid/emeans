"""Comprehensive unit, zero-leakage, FP64 parity, and FP8 stability tests for `emeans.realistic_benchmarks`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from emeans.real_corpora import (
    compute_exact_groundtruth_gpu,
    evaluate_ivf_recall_gpu,
    exact_assign_fp64_torch,
    fast_exact_assign_fp64,
    validate_real_corpus,
)
from emeans.realistic_benchmarks import (
    BENCHMARK_SPEC_BY_NAME,
    DEFAULT_PHASE5_DATA_DIR,
    SEVEN_REALISTIC_BENCHMARKS,
    _strict_deduplicate_three_way,
    compute_corpus_dynamic_range_stats,
    embed_text_splits_zero_leakage,
    evaluate_fp8_assignment_stability_probe,
    load_agnews_tfidf_svd_256,
    load_realistic_benchmark,
    validate_covtype_physical_ranges,
)


# ==============================================================================
# 1. Registry, Deduplication & Physical Sensor Validator Tests
# ==============================================================================


def test_seven_realistic_benchmarks_registry_complete() -> None:
    assert len(SEVEN_REALISTIC_BENCHMARKS) == 7
    names = [s.name for s in SEVEN_REALISTIC_BENCHMARKS]
    assert len(set(names)) == 7
    assert set(names) == {
        "sift1m",
        "glove1m_unit",
        "covtype_multiscale_54",
        "cifar10_patches_128",
        "agnews_tfidf_svd_256",
        "cifar10_resnet18_512",
        "cifar10_convnext_768",
    }
    dims = {s.d for s in SEVEN_REALISTIC_BENCHMARKS}
    assert dims == {54, 100, 128, 256, 512, 768}
    geoms = {s.geometry for s in SEVEN_REALISTIC_BENCHMARKS}
    assert geoms == {"euclidean", "unit_sphere"}
    for s in SEVEN_REALISTIC_BENCHMARKS:
        assert BENCHMARK_SPEC_BY_NAME[s.name] == s

    with pytest.raises(ValueError, match="Unknown realistic benchmark"):
        load_realistic_benchmark("nonexistent_corpus_xyz")


def test_strict_deduplicate_three_way_handles_neg_zero_and_near_ties() -> None:
    rng = np.random.default_rng(301)
    base = rng.standard_normal((120, 17)).astype(np.float32)
    base[0, :] = 0.0
    base[10] = base[3]  # exact internal duplicate
    base[25] = np.round(base[25], 6)  # anchor away from half-way 5e-7 decimal boundary

    val = rng.standard_normal((40, 17)).astype(np.float32)
    val[0, :] = -0.0  # -0.0 must collide with +0.0 in base!
    val[5] = base[25] + 1e-7  # guaranteed to round identically at 6 decimals

    query = rng.standard_normal((30, 17)).astype(np.float32)
    query[2] = base[40]
    query[7] = val[12]

    b_c, v_c, q_c = _strict_deduplicate_three_way(base, val, query, decimals=6)
    assert b_c.shape[0] == 119
    assert v_c.shape[0] == 38
    assert q_c.shape[0] == 28

    gt_ids, gt_d2 = compute_exact_groundtruth_gpu(b_c, q_c, top_k=10, device="cpu")
    from emeans.real_corpora import RealVectorCorpus

    validate_real_corpus(
        RealVectorCorpus("clean", b_c, v_c, q_c, gt_ids, gt_d2, 17, "euclidean"),
        check_cross_split_disjoint=True,
    )


def test_validate_covtype_physical_ranges_accepts_raw_and_rejects_normalized() -> None:
    rng = np.random.default_rng(302)
    good = np.zeros((256, 54), dtype=np.float32)
    good[:, 0] = rng.uniform(1860.0, 3855.0, size=256)
    good[:, 1] = rng.uniform(0.0, 360.0, size=256)
    good[:, 2] = rng.uniform(0.0, 65.0, size=256)
    good[:, 3:10] = rng.uniform(0.0, 5000.0, size=(256, 7))
    good[:, 10:] = rng.integers(0, 2, size=(256, 44)).astype(np.float32)
    validate_covtype_physical_ranges(good)

    # Standard-scaled / z-scored features must be rejected (we require raw physical scales!)
    zscored = (good - good.mean(axis=0)) / (good.std(axis=0) + 1e-6)
    with pytest.raises(ValueError, match="elevation"):
        validate_covtype_physical_ranges(zscored)

    # Non-binary soil indicator must be rejected
    bad_soil = good.copy()
    bad_soil[4, 15] = 0.5
    with pytest.raises(ValueError, match="binary"):
        validate_covtype_physical_ranges(bad_soil)


# ==============================================================================
# 2. Zero-Leakage Proof for Train-Only TF-IDF + SVD Document Embedding Pipeline
# ==============================================================================


def test_embed_text_splits_zero_train_val_query_leakage() -> None:
    topics = [
        ("quantum computing superconducting qubits coherence error correction", "physics lab"),
        ("central bank interest rates inflation bond yields treasury auction", "macro finance"),
        ("champions league striker hat trick penalty shootout stadium", "football match"),
        ("spacecraft orbital rendezvous lunar lander propulsion telemetry", "aerospace mission"),
        ("neural network transformer attention kernel triton blackwell", "gpu systems"),
        ("clinical trial monoclonal antibody phase three biomarker", "biotech study"),
    ]
    rng = np.random.default_rng(404)
    vocab_extra = [f"tok{i}" for i in range(180)]

    def _make_docs(n_docs: int, prefix: str) -> list[str]:
        out = []
        for idx in range(n_docs):
            stem, tag = topics[idx % len(topics)]
            picks = " ".join(rng.choice(vocab_extra, size=8, replace=False))
            out.append(f"{stem} {tag} {picks} {prefix}item{idx}")
        return out

    train_docs = _make_docs(260, "tr")
    val_docs_a = _make_docs(50, "va")
    query_docs_a = _make_docs(40, "qu")

    # Inject intentional raw-text leakage (whitespace & case perturbed) into val & query
    val_docs_a_with_leaks = val_docs_a + [train_docs[3].upper(), "  " + train_docs[17] + "  "]
    query_docs_a_with_leaks = query_docs_a + [train_docs[9], val_docs_a[4].upper()]

    zb_a, zv_a, zq_a, meta_a = embed_text_splits_zero_leakage(
        train_docs=train_docs,
        val_docs=val_docs_a_with_leaks,
        query_docs=query_docs_a_with_leaks,
        d=32,
        max_features=2000,
        min_df=2,
        unit_normalize=False,
        svd_seed=77,
    )
    assert zb_a.shape == (260, 32)
    assert zv_a.shape == (50, 32)
    assert zq_a.shape == (40, 32)
    assert meta_a["svd_condition_ratio"] > 1.2

    # Perturb `val_docs` completely and append adversarial documents to `query_docs`:
    # If TF-IDF or SVD leaked across splits, `zb` and the original 40 `zq` rows would change!
    val_docs_b = ["quantum inflation striker spacecraft antibody tok1 tok2 altval" + str(i) for i in range(75)]
    query_docs_b = query_docs_a + [
        "quantum quantum quantum inflation tok3 tok4 extraquery999"
    ]
    zb_b, _, zq_b, _ = embed_text_splits_zero_leakage(
        train_docs=train_docs,
        val_docs=val_docs_b,
        query_docs=query_docs_b,
        d=32,
        max_features=2000,
        min_df=2,
        unit_normalize=False,
        svd_seed=77,
    )
    # Bitwise equality proves 0.0 IDF / vocabulary / SVD leakage from val or query!
    np.testing.assert_array_equal(zb_a, zb_b)
    np.testing.assert_array_equal(zq_a, zq_b[:40])

    # Verify unit_normalize=True produces exact unit-sphere vectors
    zb_u, zv_u, zq_u, _ = embed_text_splits_zero_leakage(
        train_docs=train_docs,
        val_docs=val_docs_a,
        query_docs=query_docs_a,
        d=32,
        max_features=2000,
        min_df=2,
        unit_normalize=True,
        svd_seed=77,
    )
    for split_arr in (zb_u, zv_u, zq_u):
        norms = np.linalg.norm(split_arr.astype(np.float64), axis=1)
        assert np.max(np.abs(norms - 1.0)) < 1e-6


# ==============================================================================
# 3. Exact FP64 Assignment, GPU IVF Recall & CroissAttn-FP8 Multi-Scale Parity
# ==============================================================================


@pytest.mark.parametrize("d", [54, 256, 512, 768])
def test_fast_exact_assign_and_ivf_recall_across_new_benchmark_dims(d: int) -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(1000 + d)
    n_base, n_query, k = 1031, 67, 73  # prime shapes!

    xb = rng.standard_normal((n_base, d)).astype(np.float32)
    xq = rng.standard_normal((n_query, d)).astype(np.float32)
    if d == 54:
        # Simulate Covertype physical scales + +20,000 coordinate shift
        xb[:, 0] = rng.uniform(1859.0, 3858.0, size=n_base) + 20_000.0
        xq[:, 0] = rng.uniform(1859.0, 3858.0, size=n_query) + 20_000.0
        xb[:, 10:] = rng.integers(0, 2, size=(n_base, 44)).astype(np.float32)
        xq[:, 10:] = rng.integers(0, 2, size=(n_query, 44)).astype(np.float32)

    centroids = xb[:k].copy()
    xb_t = torch.from_numpy(xb).to(dev)
    c_t = torch.from_numpy(centroids).to(dev)

    ref_mse, ref_lbl = exact_assign_fp64_torch(xb_t, c_t)
    fast_mse, fast_lbl = fast_exact_assign_fp64(xb_t, c_t, device=dev)
    assert abs(fast_mse - ref_mse) <= 1e-12 * max(1.0, abs(ref_mse))
    assert torch.equal(fast_lbl, ref_lbl)

    gt_ids, gt_d2 = compute_exact_groundtruth_gpu(xb, xq, top_k=20, device=dev)
    ivf_all = evaluate_ivf_recall_gpu(
        x_base=xb,
        centroids=centroids,
        x_query=xq,
        groundtruth_ids=gt_ids,
        groundtruth_sq_dists=gt_d2,
        nprobes=[1, 8, k],
        recall_ks=(1, 10),
        db_labels=fast_lbl,
        device=dev,
    )
    # Monotonicity across nprobe and exact 1.000000 recall when nprobe == K
    assert ivf_all.by_nprobe[1].recall_at_10 <= ivf_all.by_nprobe[8].recall_at_10 + 1e-12
    assert ivf_all.by_nprobe[8].recall_at_10 <= ivf_all.by_nprobe[k].recall_at_10 + 1e-12
    assert abs(ivf_all.by_nprobe[k].recall_at_1 - 1.0) < 1e-12
    assert abs(ivf_all.by_nprobe[k].recall_at_10 - 1.0) < 1e-12
    assert abs(ivf_all.by_nprobe[k].id_recall_at_10 - 1.0) < 1e-12


def test_croiss_fp8_probe_shift_invariance_vs_naive_fp8_collapse() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(505)
    centers = rng.standard_normal((64, 54)).astype(np.float32) * 2.0
    ids = rng.integers(0, 64, size=1024)
    x = centers[ids] + rng.standard_normal((1024, 54)).astype(np.float32) * 0.4

    # Add huge +20,000 coordinate shift on all features
    shift = np.full((1, 54), 20_000.0, dtype=np.float32)
    res_unshifted = evaluate_fp8_assignment_stability_probe(x, centers, device=dev)
    res_shifted = evaluate_fp8_assignment_stability_probe(x + shift, centers + shift, device=dev)

    # Naive uncentered global FP8 collapses catastrophically under +20,000 shift (>50% MSE inflation),
    # whereas CroissAttn-style centered decoupled FP8 is shift-invariant (<0.05% MSE inflation)!
    assert res_shifted["naive_global_fp8_rel_inflate_pct"] > 50.0
    assert abs(res_shifted["croiss_decoupled_fp8_rel_inflate_pct"]) < 0.05
    assert abs(
        res_shifted["croiss_decoupled_fp8_rel_inflate_pct"]
        - res_unshifted["croiss_decoupled_fp8_rel_inflate_pct"]
    ) < 1e-3


# ==============================================================================
# 4. End-to-End Validation of Cached 7 Realistic Benchmarks (When Present on Disk)
# ==============================================================================


def test_all_seven_staged_benchmarks_integrity_and_groundtruth() -> None:
    # Verify no stray atomic temp files in phase5 data dir
    if DEFAULT_PHASE5_DATA_DIR.exists():
        assert list(DEFAULT_PHASE5_DATA_DIR.glob(".tmp_*")) == []

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_count = 0
    for spec in SEVEN_REALISTIC_BENCHMARKS:
        p5_cache = DEFAULT_PHASE5_DATA_DIR / spec.cache_filename
        p4_cache = Path(__file__).resolve().parents[1] / "outputs" / "phase4_million_scale" / "data" / spec.cache_filename
        if not (p5_cache.exists() or p4_cache.exists()):
            continue
        loaded_count += 1
        corpus = load_realistic_benchmark(spec.name, data_dir=DEFAULT_PHASE5_DATA_DIR, device=dev)
        validate_real_corpus(corpus, check_cross_split_disjoint=True)
        assert corpus.d == spec.d
        assert corpus.geometry == spec.geometry
        assert corpus.groundtruth_ids.shape == (corpus.query.shape[0], 100)

        # Spot-verify exact FP64 top-100 ground truth on 16 evenly spaced queries
        q_idx = np.linspace(0, corpus.query.shape[0] - 1, num=16, dtype=np.int64)
        gt_check_ids, gt_check_d2 = compute_exact_groundtruth_gpu(
            corpus.base, corpus.query[q_idx], top_k=100, device=dev
        )
        np.testing.assert_allclose(
            corpus.groundtruth_sq_dists[q_idx], gt_check_d2, rtol=1e-12, atol=1e-9
        )
        np.testing.assert_array_equal(corpus.groundtruth_ids[q_idx], gt_check_ids)

        stats = compute_corpus_dynamic_range_stats(corpus, max_rows=8192)
        assert np.isfinite(list(stats.values())).all()

        if spec.name == "covtype_multiscale_54":
            validate_covtype_physical_ranges(corpus.base)
            assert stats["feature_std_max_over_min"] > 1000.0
        elif spec.name == "cifar10_resnet18_512":
            assert float(corpus.base.min()) >= -1e-6  # ReLU non-negative orthant
        elif spec.name == "cifar10_convnext_768":
            assert float(corpus.base.min()) < -0.5 and float(corpus.base.max()) > 0.5
        elif spec.name == "agnews_tfidf_svd_256":
            assert stats["row_norm_p99_over_p01"] > 3.0

    # Ensure unit-normalized AG-News variant also validates cleanly when staged
    ag_u_cache = DEFAULT_PHASE5_DATA_DIR / "agnews_tfidf_svd_256_unit_s20261116_validated_v1.npz"
    if ag_u_cache.exists():
        ag_u = load_agnews_tfidf_svd_256(data_dir=DEFAULT_PHASE5_DATA_DIR, unit_normalize=True, device=dev)
        validate_real_corpus(ag_u, check_cross_split_disjoint=True)
        assert ag_u.geometry == "unit_sphere"

    assert loaded_count == 7, f"Expected all 7 staged benchmarks on disk, found {loaded_count}"


def test_real_covtype_naive_fp8_vs_croiss_fp8_separation() -> None:
    p5_cache = DEFAULT_PHASE5_DATA_DIR / "covtype_multiscale_54_s20261116_validated_v1.npz"
    if not p5_cache.exists():
        pytest.skip("covtype cache not yet staged")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corpus = load_realistic_benchmark("covtype_multiscale_54", data_dir=DEFAULT_PHASE5_DATA_DIR, device=dev)
    rng = np.random.default_rng(777)
    c_idx = rng.choice(corpus.base.shape[0], size=512, replace=False)
    res = evaluate_fp8_assignment_stability_probe(corpus.query[:2048], corpus.base[c_idx], device=dev)
    # On real Forest Covertype ([1859..3858]m elevation + [0..7173]m road/fire + {0,1} soil),
    # naive global FP8 inflates MSE by >15%, whereas CroissAttn decoupled FP8 cuts error >15x!
    assert res["naive_global_fp8_rel_inflate_pct"] > 15.0
    assert res["croiss_decoupled_fp8_rel_inflate_pct"] < res["naive_global_fp8_rel_inflate_pct"] / 15.0


def test_corrupt_cached_npz_rejected_by_loader(tmp_path: Path) -> None:
    from emeans.real_corpora import atomic_savez
    from emeans.realistic_benchmarks import _load_cached_corpus_npz

    rng = np.random.default_rng(888)
    xb = rng.standard_normal((100, 54)).astype(np.float32)
    xv = rng.standard_normal((20, 54)).astype(np.float32)
    xq = rng.standard_normal((15, 54)).astype(np.float32)
    xq[0] = xb[0]  # leaked vector!
    gt_ids, gt_d2 = compute_exact_groundtruth_gpu(xb, xq, top_k=10, device="cpu")
    bad_path = tmp_path / "leaked.npz"
    atomic_savez(bad_path, base=xb, val=xv, query=xq, groundtruth_ids=gt_ids, groundtruth_sq_dists=gt_d2)
    with pytest.raises(ValueError, match="DATA LEAKAGE"):
        _load_cached_corpus_npz(bad_path, "bad", 54, "euclidean")

