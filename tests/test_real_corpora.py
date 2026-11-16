"""Comprehensive unit, parity, data-hygiene, and adversarial tests for `emeans.real_corpora`."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from emeans.kernels import exact_mse_fp64
from emeans.real_corpora import (
    DEFAULT_MILLION_DATA_DIR,
    RealVectorCorpus,
    atomic_savez,
    compute_exact_groundtruth_gpu,
    compute_sha256,
    evaluate_ivf_recall_gpu,
    exact_assign_fp64_torch,
    exact_cosine_distortion_fp64,
    fast_exact_assign_fp64,
    fast_exact_mse_fp64,
    load_glove1m_unit,
    load_sift1m,
    read_fvecs,
    read_ivecs,
    validate_real_corpus,
    write_fvecs_atomic,
)


def _reference_ivf_flat_numpy(
    xb: np.ndarray,
    centroids: np.ndarray,
    xq: np.ndarray,
    gt_ids: np.ndarray,
    gt_d2: np.ndarray,
    nprobes: list[int],
) -> dict[int, tuple[float, float, float, float, float, float]]:
    """Pure-NumPy FP64 brute-force reference IVF-Flat evaluator for exact parity checks."""
    xb64 = xb.astype(np.float64)
    c64 = centroids.astype(np.float64)
    xq64 = xq.astype(np.float64)

    n_base = xb64.shape[0]
    k = c64.shape[0]
    n_query = xq64.shape[0]
    top_r = gt_ids.shape[1]

    # Exact database assignment (tie-break by smallest cluster ID via np.argmin)
    d2_bc = ((xb64[:, None, :] - c64[None, :, :]) ** 2).sum(axis=2)
    db_labels = np.argmin(d2_bc, axis=1)

    buckets = [np.where(db_labels == cid)[0] for cid in range(k)]

    # Exact coarse query distances (tie-break by smallest cluster ID via stable mergesort)
    d2_qc = ((xq64[:, None, :] - c64[None, :, :]) ** 2).sum(axis=2)
    coarse_order = np.argsort(d2_qc, axis=1, kind="mergesort")

    results: dict[int, tuple[float, float, float, float, float, float]] = {}
    for np_val in nprobes:
        eff_np = max(1, min(int(np_val), k))
        r1_list, r10_list, r100_list = [], [], []
        id1_list, id10_list, id100_list = [], [], []

        for qi in range(n_query):
            cids = coarse_order[qi, :eff_np]
            segs = [buckets[cid] for cid in cids if buckets[cid].size > 0]
            if segs:
                cand_ids = np.concatenate(segs, axis=0)
                cand_ids.sort()  # ascending DB ID before stable distance sort
                cand_d2 = ((xb64[cand_ids] - xq64[qi : qi + 1]) ** 2).sum(axis=1)
                ord_c = np.argsort(cand_d2, kind="mergesort")[:top_r]
                pred_d2 = np.full((top_r,), np.inf, dtype=np.float64)
                pred_id = np.full((top_r,), -1, dtype=np.int64)
                pred_d2[: ord_c.size] = cand_d2[ord_c]
                pred_id[: ord_c.size] = cand_ids[ord_c]
            else:
                pred_d2 = np.full((top_r,), np.inf, dtype=np.float64)
                pred_id = np.full((top_r,), -1, dtype=np.int64)

            for rk, d_acc, id_acc in (
                (1, r1_list, id1_list),
                (10, r10_list, id10_list),
                (100, r100_list, id100_list),
            ):
                reff = min(rk, top_r)
                thresh = gt_d2[qi, reff - 1] * (1.0 + 1e-12) + 1e-15
                d_acc.append(float((pred_d2[:reff] <= thresh).sum()) / reff)
                valid_ids = pred_id[:reff][pred_id[:reff] >= 0]
                id_acc.append(float(np.intersect1d(valid_ids, gt_ids[qi, :reff]).size) / reff)

        results[eff_np] = (
            float(np.mean(r1_list)),
            float(np.mean(r10_list)),
            float(np.mean(r100_list)),
            float(np.mean(id1_list)),
            float(np.mean(id10_list)),
            float(np.mean(id100_list)),
        )
    return results


# ==============================================================================
# 1. Atomic Caching, `.fvecs` / `.ivecs` Readers, and Leak-Guard Validation
# ==============================================================================


def test_atomic_fvecs_and_savez_roundtrip_no_temp_leftovers(tmp_path: Path) -> None:
    rng = np.random.default_rng(101)
    arr = rng.standard_normal((257, 37)).astype(np.float32)
    fvecs_path = tmp_path / "sample.fvecs"
    write_fvecs_atomic(fvecs_path, arr)

    assert fvecs_path.exists()
    assert list(tmp_path.glob(".tmp_*")) == []

    loaded = read_fvecs(fvecs_path)
    assert loaded.shape == (257, 37)
    assert loaded.dtype == np.float32
    np.testing.assert_array_equal(loaded, arr)

    sha1 = compute_sha256(fvecs_path)
    sha2 = compute_sha256(fvecs_path, chunk_bytes=1024)
    assert len(sha1) == 64 and sha1 == sha2

    npz_path = tmp_path / "bundle.npz"
    atomic_savez(npz_path, x=arr, tag=np.int64(7))
    assert npz_path.exists()
    assert list(tmp_path.glob(".tmp_*")) == []
    with np.load(npz_path) as z:
        np.testing.assert_array_equal(z["x"], arr)
        assert int(z["tag"]) == 7


def test_corrupt_fvecs_and_ivecs_rejected(tmp_path: Path) -> None:
    empty_p = tmp_path / "empty.fvecs"
    empty_p.write_bytes(b"")
    with pytest.raises(ValueError, match="Empty"):
        read_fvecs(empty_p)

    # Truncated payload
    trunc_p = tmp_path / "trunc.fvecs"
    np.array([4, 1, 2], dtype="<i4").tofile(trunc_p)
    with pytest.raises(ValueError, match="Corrupt"):
        read_fvecs(trunc_p)

    # NaN payload
    nan_arr = np.ones((4, 8), dtype=np.float32)
    nan_arr[2, 3] = np.nan
    nan_p = tmp_path / "nan.fvecs"
    write_fvecs_atomic(nan_p, nan_arr)
    with pytest.raises(ValueError, match="Non-finite"):
        read_fvecs(nan_p)

    # Corrupt ivecs header mismatch across rows
    bad_iv = tmp_path / "bad.ivecs"
    rows = np.array([[2, 10, 20], [3, 10, 20]], dtype="<i4")
    rows.tofile(bad_iv)
    with pytest.raises(ValueError, match="Inconsistent"):
        read_ivecs(bad_iv)


def test_validate_real_corpus_catches_leakage_and_norm_errors() -> None:
    rng = np.random.default_rng(202)
    xb = rng.standard_normal((500, 16)).astype(np.float32)
    xv = rng.standard_normal((100, 16)).astype(np.float32)
    xq = rng.standard_normal((50, 16)).astype(np.float32)
    gt_ids, gt_d2 = compute_exact_groundtruth_gpu(xb, xq, top_k=20, device="cpu")

    clean = RealVectorCorpus(
        name="toy",
        base=xb,
        val=xv,
        query=xq,
        groundtruth_ids=gt_ids,
        groundtruth_sq_dists=gt_d2,
        d=16,
        geometry="euclidean",
    )
    validate_real_corpus(clean)

    # Inject base -> query leak
    xq_leak = xq.copy()
    xq_leak[7] = xb[123]
    leaked_bq = RealVectorCorpus("toy", xb, xv, xq_leak, gt_ids, gt_d2, 16, "euclidean")
    with pytest.raises(ValueError, match="DATA LEAKAGE"):
        validate_real_corpus(leaked_bq)

    # Inject val -> query leak
    xv_leak = xv.copy()
    xv_leak[3] = xq[12]
    leaked_vq = RealVectorCorpus("toy", xb, xv_leak, xq, gt_ids, gt_d2, 16, "euclidean")
    with pytest.raises(ValueError, match="DATA LEAKAGE"):
        validate_real_corpus(leaked_vq)

    # Non-unit norm on unit_sphere corpus
    bad_unit = RealVectorCorpus("toy_u", xb, xv, xq, gt_ids, gt_d2, 16, "unit_sphere")
    with pytest.raises(ValueError, match="unit sphere"):
        validate_real_corpus(bad_unit)


# ==============================================================================
# 2. Exact Recall@K_all == 1.0 & Brute-Force Parity Tests (CPU & GPU)
# ==============================================================================


@pytest.mark.parametrize("dev_str", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
@pytest.mark.parametrize(
    "n_base,d,k,n_query",
    [
        (600, 17, 16, 40),    # avg ~37/bucket, prime D=17
        (1200, 65, 4, 35),    # avg 300/bucket (> BLOCK_CAP=128 -> exercises streaming tl.topk!)
        (513, 129, 64, 29),   # D > 128 multi-tile path + prime N, D, Q
        (300, 1, 9, 25),      # D == 1 edge case
    ],
)
def test_ivf_recall_full_probe_equals_one_and_partial_matches_reference(
    dev_str: str,
    n_base: int,
    d: int,
    k: int,
    n_query: int,
) -> None:
    rng = np.random.default_rng(303 + d + k)
    xb = rng.standard_normal((n_base, d)).astype(np.float32)
    # Inject exact duplicate database rows (mimicking SIFT1M's 86 duplicate vectors)
    xb[10] = xb[5]
    xb[20] = xb[5]
    xq = rng.standard_normal((n_query, d)).astype(np.float32)
    centroids = xb[:k].copy()
    if k >= 4:
        # Inject an empty cluster far away + duplicate twin centroids
        centroids[k - 1] = np.full((d,), 1e4, dtype=np.float32)
        centroids[1] = centroids[0]

    gt_ids, gt_d2 = compute_exact_groundtruth_gpu(xb, xq, top_k=min(100, n_base), device=dev_str)
    nprobes = sorted({1, max(1, k // 4), max(1, k // 2), k})

    res = evaluate_ivf_recall_gpu(
        x_base=xb,
        centroids=centroids,
        x_query=xq,
        groundtruth_ids=gt_ids,
        groundtruth_sq_dists=gt_d2,
        nprobes=nprobes,
        device=dev_str,
    )

    # Full probe (`nprobe == K`) MUST achieve exact Recall == 1.000000 across all k!
    full = res.by_nprobe[k]
    assert full.recall_at_1 == pytest.approx(1.0, abs=1e-12)
    assert full.recall_at_10 == pytest.approx(1.0, abs=1e-12)
    assert full.recall_at_100 == pytest.approx(1.0, abs=1e-12)
    assert full.id_recall_at_1 == pytest.approx(1.0, abs=1e-12)
    assert full.id_recall_at_10 == pytest.approx(1.0, abs=1e-12)
    assert full.id_recall_at_100 == pytest.approx(1.0, abs=1e-12)
    assert full.actual_scan_frac == pytest.approx(1.0, abs=1e-12)

    # Monotonicity across increasing nprobe
    prev_r1, prev_r10, prev_r100 = -1.0, -1.0, -1.0
    for np_v in nprobes:
        m = res.by_nprobe[np_v]
        assert m.recall_at_1 + 1e-12 >= prev_r1
        assert m.recall_at_10 + 1e-12 >= prev_r10
        assert m.recall_at_100 + 1e-12 >= prev_r100
        prev_r1, prev_r10, prev_r100 = m.recall_at_1, m.recall_at_10, m.recall_at_100

    # Exact agreement against pure-NumPy FP64 brute-force IVF-Flat across ALL nprobe values!
    ref = _reference_ivf_flat_numpy(xb, centroids, xq, gt_ids, gt_d2, nprobes)
    for np_v in nprobes:
        m = res.by_nprobe[np_v]
        rr1, rr10, rr100, rid1, rid10, rid100 = ref[np_v]
        assert m.recall_at_1 == pytest.approx(rr1, abs=1e-12)
        assert m.recall_at_10 == pytest.approx(rr10, abs=1e-12)
        assert m.recall_at_100 == pytest.approx(rr100, abs=1e-12)
        assert m.id_recall_at_1 == pytest.approx(rid1, abs=1e-12)
        assert m.id_recall_at_10 == pytest.approx(rid10, abs=1e-12)
        assert m.id_recall_at_100 == pytest.approx(rid100, abs=1e-12)


# ==============================================================================
# 3. Fast Exact FP64 Assignment, Shift Equivariance & Extreme Scales
# ==============================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("d", [1, 17, 64, 100, 128, 257])
@pytest.mark.parametrize("k", [1, 31, 65, 259])
def test_fast_exact_assign_fp64_parity_vs_reference(d: int, k: int) -> None:
    gen = torch.Generator(device="cuda").manual_seed(404 + d * 13 + k)
    x = torch.randn((2053, d), generator=gen, device="cuda", dtype=torch.float32)
    c = torch.randn((k, d), generator=gen, device="cuda", dtype=torch.float32)

    mse_fast, lbl_fast = fast_exact_assign_fp64(x, c)
    mse_torch, lbl_torch = exact_assign_fp64_torch(x, c)
    mse_ref = exact_mse_fp64(x, c)

    assert math.isfinite(mse_fast)
    assert abs(mse_fast - mse_ref) / max(mse_ref, 1e-12) < 1e-11
    assert abs(mse_fast - mse_torch) / max(mse_torch, 1e-12) < 1e-11
    assert torch.equal(lbl_fast, lbl_torch)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_coordinate_shift_and_extreme_scale_invariance() -> None:
    gen = torch.Generator(device="cuda").manual_seed(505)
    xb = torch.randn((1500, 64), generator=gen, device="cuda", dtype=torch.float64)
    xq = torch.randn((64, 64), generator=gen, device="cuda", dtype=torch.float64)
    c = xb[:129].clone()

    mse0, lbl0 = fast_exact_assign_fp64(xb, c)
    ivf0 = evaluate_ivf_recall_gpu(xb, c, xq, nprobes=[4, 16, 129], db_labels=lbl0)

    # +20,000 coordinate shift in FP64
    shift = 20_000.0
    mse_sh, lbl_sh = fast_exact_assign_fp64(xb + shift, c + shift)
    ivf_sh = evaluate_ivf_recall_gpu(xb + shift, c + shift, xq + shift, nprobes=[4, 16, 129], db_labels=lbl_sh)

    assert abs(mse_sh - mse0) / mse0 < 1e-11
    assert torch.equal(lbl_sh, lbl0)
    for np_v in (4, 16, 129):
        assert ivf_sh.by_nprobe[np_v].recall_at_1 == pytest.approx(ivf0.by_nprobe[np_v].recall_at_1, abs=1e-12)
        assert ivf_sh.by_nprobe[np_v].recall_at_10 == pytest.approx(ivf0.by_nprobe[np_v].recall_at_10, abs=1e-12)

    # Extreme scales: 1e-25, 1e+25, and all-zeros
    for scale in (1e-25, 1e25):
        mse_sc = fast_exact_mse_fp64(xb * scale, c * scale)
        assert math.isfinite(mse_sc)
        assert abs(mse_sc - mse0 * (scale * scale)) / (mse0 * scale * scale) < 1e-10
        ivf_sc = evaluate_ivf_recall_gpu(xb * scale, c * scale, xq * scale, nprobes=[4, 129])
        assert ivf_sc.by_nprobe[4].recall_at_10 == pytest.approx(ivf0.by_nprobe[4].recall_at_10, abs=1e-12)
        assert ivf_sc.by_nprobe[129].recall_at_10 == pytest.approx(1.0, abs=1e-12)

    # Degenerate all-zeros
    z_b = torch.zeros((512, 32), device="cuda", dtype=torch.float32)
    z_c = torch.zeros((64, 32), device="cuda", dtype=torch.float32)
    z_q = torch.zeros((16, 32), device="cuda", dtype=torch.float32)
    mse_z, lbl_z = fast_exact_assign_fp64(z_b, z_c)
    assert mse_z == 0.0
    assert (lbl_z == 0).all()
    ivf_z = evaluate_ivf_recall_gpu(z_b, z_c, z_q, nprobes=[1, 64], db_labels=lbl_z)
    assert ivf_z.by_nprobe[1].recall_at_1 == pytest.approx(1.0, abs=1e-12)
    assert exact_cosine_distortion_fp64(z_b, z_c) == 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16, torch.float64])
def test_mixed_input_dtypes_and_strided_views(dtype: torch.dtype) -> None:
    gen = torch.Generator(device="cuda").manual_seed(606)
    raw = torch.randn((1024, 66), generator=gen, device="cuda", dtype=torch.float32).to(dtype)
    xb = raw[:, ::2]  # non-contiguous strided D=33!
    xq = xb[:32]
    c = xb[:65]
    mse, lbl = fast_exact_assign_fp64(xb, c)
    assert math.isfinite(mse) and mse >= 0.0
    res = evaluate_ivf_recall_gpu(xb, c, xq, nprobes=[8, 65], db_labels=lbl)
    assert res.by_nprobe[65].recall_at_1 == pytest.approx(1.0, abs=1e-12)
    cos_d = exact_cosine_distortion_fp64(xb, c)
    assert math.isfinite(cos_d) and 0.0 <= cos_d <= 2.0


# ==============================================================================
# 4. Cached Real Million-Scale Corpora (`SIFT1M` & `GloVe-1M-Unit`) Verification
# ==============================================================================


@pytest.mark.skipif(
    not (DEFAULT_MILLION_DATA_DIR / "sift1m_validated_v1.npz").exists(),
    reason="SIFT1M cache not yet populated",
)
def test_cached_sift1m_integrity_and_subsecond_gpu_eval() -> None:
    sift = load_sift1m()
    assert sift.base.shape == (1_000_000, 128)
    assert sift.query.shape == (10_000, 128)
    assert sift.groundtruth_ids.shape == (10_000, 100)
    assert sift.val.shape[0] > 80_000 and sift.val.shape[1] == 128
    assert sift.metadata["gt_fp64_max_abs_diff"] < 1e-9

    if torch.cuda.is_available():
        xb_g = torch.from_numpy(sift.base).cuda()
        xq_g = torch.from_numpy(sift.query).cuda()
        c_g = xb_g[:4096].contiguous()
        # Warmup once
        _, _ = fast_exact_assign_fp64(xb_g[:4096], c_g)
        _ = evaluate_ivf_recall_gpu(
            xb_g[:4096], c_g[:64], xq_g[:64], selectivities=(0.05,)
        )
        mse, db_lbl = fast_exact_assign_fp64(xb_g, c_g)
        assert math.isfinite(mse) and mse > 0.0
        ivf = evaluate_ivf_recall_gpu(
            x_base=xb_g,
            centroids=c_g,
            x_query=xq_g,
            groundtruth_ids=sift.groundtruth_ids,
            groundtruth_sq_dists=sift.groundtruth_sq_dists,
            selectivities=(0.001, 0.005, 0.010),
            db_labels=db_lbl,
        )
        assert ivf.eval_time_sec < 1.0
        assert 0.0 < ivf.by_selectivity["0.1%"].recall_at_10 <= ivf.by_selectivity["1%"].recall_at_10 <= 1.0


@pytest.mark.skipif(
    not (DEFAULT_MILLION_DATA_DIR / "glove1m_unit_s20261116_validated_v1.npz").exists(),
    reason="GloVe-1M-Unit cache not yet populated",
)
def test_cached_glove1m_unit_integrity() -> None:
    glove = load_glove1m_unit()
    assert glove.base.shape == (1_000_000, 100)
    assert glove.val.shape == (100_000, 100)
    assert glove.query.shape == (10_000, 100)
    assert glove.groundtruth_ids.shape == (10_000, 100)
    norms = np.linalg.norm(glove.query.astype(np.float64), axis=1)
    assert np.max(np.abs(norms - 1.0)) < 1e-5
