"""Comprehensive Unit, Numerical-Parity, Zero-Leakage & Adversarial Fuzz Tests for `pq_rvq.py`."""

from __future__ import annotations

import os
from pathlib import Path

_HOME_TMP = Path("/home/catid/.cache/emeans_tmp")
_HOME_TMP.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TMPDIR", str(_HOME_TMP))

import numpy as np
import pytest
import torch

from emeans.pq_rvq import (
    PQ_SUPPORTED_SOLVERS,
    RVQ_SUPPORTED_SOLVERS,
    ApexPQ,
    ApexRVQ,
    evaluate_adc_recall,
    normalize_solver_name,
    pack_rvq_codes_uint8,
    unpack_rvq_codes_uint8,
    warmup_pq_rvq_kernels,
)


def _pref_device() -> torch.device:
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def _make_clustered_data(
    n: int = 2048,
    n_q: int = 128,
    d: int = 32,
    n_modes: int = 32,
    seed: int = 123,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate clustered train/query vectors + exact FP64 brute-force top-20 ground truth."""
    dev = _pref_device() if device is None else device
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    centers = torch.randn((n_modes, d), generator=gen, dtype=torch.float32) * 4.0
    assign_b = torch.randint(0, n_modes, (n,), generator=gen)
    assign_q = torch.randint(0, n_modes, (n_q,), generator=gen)
    xb = (centers[assign_b] + 0.35 * torch.randn((n, d), generator=gen)).to(dev)
    xq = (centers[assign_q] + 0.35 * torch.randn((n_q, d), generator=gen)).to(dev)

    d2_64 = torch.cdist(xq.double(), xb.double()).square()
    k_gt = min(20, n)
    gt_d2, gt_ids = torch.topk(d2_64, k=k_gt, dim=1, largest=False, sorted=True)
    return xb, xq, gt_ids.to(torch.int32), gt_d2


def test_solver_alias_normalization_and_input_validation() -> None:
    assert normalize_solver_name("ApexEMeans") == "apex_emeans"
    assert normalize_solver_name("UltraEMeans") == "ultra_emeans"
    assert normalize_solver_name("HybridSEMeans") == "hybrid_semeans"
    assert normalize_solver_name("EMeansPP") == "emeans_pp"
    assert normalize_solver_name("SMeans") == "smeans"
    assert normalize_solver_name("EMeans") == "emeans"
    assert normalize_solver_name("Lloyd@25") == "lloyd_25"
    assert normalize_solver_name("faiss.IndexPQ") == "faiss_pq"

    with pytest.raises(ValueError, match="Unsupported solver"):
        normalize_solver_name("nonexistent_solver")

    with pytest.raises(ValueError, match="divisible"):
        ApexPQ(d=100, m=16, ks=64)

    with pytest.raises(ValueError, match="nepochs"):
        ApexPQ(d=32, m=4, ks=16, nepochs=0.0)

    with pytest.raises(ValueError, match="nepochs"):
        ApexRVQ(d=32, k1=16, k2=16, nepochs=-1.0)

    dev = _pref_device()
    pq = ApexPQ(d=16, m=4, ks=32, nepochs=1.0)
    with pytest.raises(ValueError, match=">= ks"):
        pq.fit(torch.randn((16, 16), device=dev))

    x_nan = torch.randn((128, 16), device=dev)
    x_nan[5, 3] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        pq.fit(x_nan)

    rvq = ApexRVQ(d=16, k1=32, k2=32, nepochs=1.0)
    x_inf = torch.randn((128, 16), device=dev)
    x_inf[7, 1] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        rvq.fit(x_inf)


def test_apex_pq_lut_adc_matches_bruteforce_reconstructed_l2_and_ties() -> None:
    dev = _pref_device()
    xb, xq, _, _ = _make_clustered_data(n=1500, n_q=64, d=32, seed=201, device=dev)
    # Inject exact duplicate rows to exercise deterministic smallest-ID tie-breaking!
    xb[500] = xb[17]
    xb[999] = xb[17]

    pq = ApexPQ(d=32, m=8, ks=64, nepochs=2.0, solver="apex_emeans", seed=11)
    _, codes = pq.fit(xb, capture_codes=True)
    assert codes is not None
    assert codes.dtype == torch.uint8
    assert codes.shape == (1500, 8)

    dists_adc, ids_adc = pq.adc_search(xq, codes, k_top=15)
    x_hat = pq.decode(codes)
    assert isinstance(x_hat, torch.Tensor)
    exact_d2 = (xq.double()[:, None, :] - x_hat.double()[None, :, :]).square().sum(dim=2)

    # Sort exact distances with deterministic smallest-ID tie-breaking
    n_db = xb.shape[0]
    id_grid = torch.arange(n_db, device=dev, dtype=torch.int64)[None, :].expand_as(exact_d2)
    exact_f32 = exact_d2.float()
    exact_f32 = torch.where(exact_f32 > 0.0, exact_f32, torch.zeros_like(exact_f32))
    packed_ref = (exact_f32.view(torch.int32).to(torch.int64) << 32) | id_grid
    top_ref, _ = torch.topk(packed_ref, k=15, dim=1, largest=False, sorted=True)
    ref_ids = top_ref & 0xFFFFFFFF
    ref_d2 = torch.gather(exact_d2, 1, ref_ids).float()

    torch.testing.assert_close(dists_adc, ref_d2, rtol=1e-4, atol=1e-4)
    # Whenever distances are separated by > 1e-4, IDs must match 100%; for exact duplicate codes (17, 500, 999),
    # 17 must always precede 500 and 999!
    for q_i in range(xq.shape[0]):
        row_ids = ids_adc[q_i].tolist()
        for a, b in ((17, 500), (500, 999), (17, 999)):
            if a in row_ids and b in row_ids:
                assert row_ids.index(a) < row_ids.index(b)


def test_apex_rvq_exact_lut_adc_exhaustive_and_beam_match_bruteforce() -> None:
    dev = _pref_device()
    xb, xq, _, _ = _make_clustered_data(n=1600, n_q=48, d=32, seed=303, device=dev)
    xb[777] = xb[12]

    rvq = ApexRVQ(d=32, k1=64, k2=64, nepochs=2.0, solver="apex_emeans", adc_beam=64, seed=21)
    (c1, c2), codes = rvq.fit(xb, capture_codes=True)
    assert codes is not None
    assert c1.shape == (64, 32) and c2.shape == (64, 32)
    assert codes.shape == (1600, 2)

    x_hat = rvq.decode(codes)
    exact_d2 = (xq.double()[:, None, :] - x_hat.double()[None, :, :]).square().sum(dim=2).float()
    ref_d2, _ = torch.topk(exact_d2, k=10, dim=1, largest=False, sorted=True)

    # 1. Exhaustive (`adc_beam=None`)
    d_ex, id_ex = rvq.adc_search(xq, codes, k_top=10, adc_beam=None)
    torch.testing.assert_close(d_ex, ref_d2, rtol=1e-4, atol=1e-4)

    # 2. Full-K1 CSR Beam (`adc_beam=64 == K1`) must also match exhaustive to 1e-4!
    d_bm, id_bm = rvq.adc_search(xq, codes, k_top=10, adc_beam=64)
    torch.testing.assert_close(d_bm, d_ex, rtol=1e-4, atol=1e-4)
    assert torch.equal(id_bm[:, 0], id_ex[:, 0])

    # 3. Pruned CSR Beam (`adc_beam=16 < K1`) returns exact distances for every returned ID
    d_p, id_p = rvq.adc_search(xq, codes, k_top=10, adc_beam=16)
    gathered_true = torch.gather(exact_d2, 1, id_p)
    torch.testing.assert_close(d_p, gathered_true, rtol=1e-4, atol=1e-4)


def test_rvq_multi_beam_encoding_and_uint8_bit_packing() -> None:
    dev = _pref_device()
    xb, xq, _, _ = _make_clustered_data(n=1024, n_q=128, d=24, seed=404, device=dev)
    rvq = ApexRVQ(d=24, k1=64, k2=64, nepochs=2.0, solver="apex_emeans", encode_beam=1, seed=33)
    rvq.fit(xb)

    codes_b1 = rvq.encode(xq, beam=1)
    codes_b4 = rvq.encode(xq, beam=4)
    rec_b1 = rvq.evaluate_reconstruction(xq, codes=codes_b1)
    rec_b4 = rvq.evaluate_reconstruction(xq, codes=codes_b4)
    assert rec_b4.mse <= rec_b1.mse * (1.0 + 1e-6)
    assert rec_b4.sqnr_db >= rec_b1.sqnr_db - 1e-5

    # Test 24-bit (`4096 x 4096` -> 3 B/vec) and 20-bit (`1024 x 1024` -> 3 packed B/vec) round-trip
    rng = np.random.default_rng(99)
    for k1, k2, exp_bytes in ((4096, 4096, 3), (1024, 1024, 3), (64, 64, 2)):
        fake_codes = np.column_stack([
            rng.integers(0, k1, size=513, dtype=np.int32),
            rng.integers(0, k2, size=513, dtype=np.int32),
        ])
        packed = pack_rvq_codes_uint8(fake_codes, k1=k1, k2=k2)
        assert packed.dtype == np.uint8 and packed.shape == (513, exp_bytes)
        unpacked = unpack_rvq_codes_uint8(packed, k1=k1, k2=k2)
        np.testing.assert_array_equal(unpacked, fake_codes)


def test_coordinate_shift_20000_equivariance_pq_and_rvq() -> None:
    dev = _pref_device()
    xb, xq, gt_ids, gt_d2 = _make_clustered_data(n=2048, n_q=96, d=32, seed=505, device=dev)
    shift = torch.linspace(-20000.0, 25000.0, 32, device=dev, dtype=torch.float32)[None, :]
    xb_s = (xb.double() + shift.double()).float()
    xq_s = (xq.double() + shift.double()).float()

    # 1. ApexPQ shift equivariance
    pq0 = ApexPQ(d=32, m=8, ks=64, nepochs=2.0, solver="apex_emeans", seed=77)
    _, c0 = pq0.fit(xb, capture_codes=True)
    rec0 = pq0.evaluate_reconstruction(xq)
    adc0 = evaluate_adc_recall(pq0, xb, c0, xq, gt_ids, gt_d2)

    pq_s = ApexPQ(d=32, m=8, ks=64, nepochs=2.0, solver="apex_emeans", seed=77)
    _, cs = pq_s.fit(xb_s, capture_codes=True)
    rec_s = pq_s.evaluate_reconstruction(xq_s)
    adc_s = evaluate_adc_recall(pq_s, xb_s, cs, xq_s, gt_ids, gt_d2)

    assert abs(rec_s.mse - rec0.mse) / max(rec0.mse, 1e-9) < 0.02
    assert abs(rec_s.evr_sqnr_db - rec0.evr_sqnr_db) < 0.15
    assert abs(adc_s.one_recall_at_10 - adc0.one_recall_at_10) < 0.05

    # 2. ApexRVQ shift equivariance
    rvq0 = ApexRVQ(d=32, k1=64, k2=64, nepochs=2.0, solver="apex_emeans", adc_beam=32, seed=88)
    _, rc0 = rvq0.fit(xb, capture_codes=True)
    rrec0 = rvq0.evaluate_reconstruction(xq)
    radc0 = evaluate_adc_recall(rvq0, xb, rc0, xq, gt_ids, gt_d2)

    rvq_s = ApexRVQ(d=32, k1=64, k2=64, nepochs=2.0, solver="apex_emeans", adc_beam=32, seed=88)
    _, rcs = rvq_s.fit(xb_s, capture_codes=True)
    rrec_s = rvq_s.evaluate_reconstruction(xq_s)
    radc_s = evaluate_adc_recall(rvq_s, xb_s, rcs, xq_s, gt_ids, gt_d2)

    assert abs(rrec_s.mse - rrec0.mse) / max(rrec0.mse, 1e-9) < 0.03
    assert abs(rrec_s.evr_sqnr_db - rrec0.evr_sqnr_db) < 0.20
    assert abs(radc_s.one_recall_at_10 - radc0.one_recall_at_10) < 0.06


def test_query_batch_outlier_isolation_and_strict_zero_leakage() -> None:
    """Verify `#204` defense (`queries[20:] = 1e25` cannot alter `queries[:20]`) and zero query leakage."""
    dev = _pref_device()
    xb, xq, _, _ = _make_clustered_data(n=1024, n_q=40, d=32, seed=606, device=dev)

    pq = ApexPQ(d=32, m=4, ks=64, nepochs=2.0, solver="apex_emeans", seed=19)
    cb_orig, codes_b = pq.fit(xb, capture_codes=True)
    cb_snap = cb_orig.clone()
    orig_snap = pq.train_origin64.clone()

    xq_corrupt = xq.clone()
    xq_corrupt[20:] = 1.0e25

    d_clean, i_clean = pq.adc_search(xq[:20], codes_b, k_top=10)
    d_full, i_full = pq.adc_search(xq_corrupt, codes_b, k_top=10)
    assert torch.isfinite(d_full).all()
    assert (i_full >= 0).all()
    assert torch.equal(i_clean, i_full[:20])
    torch.testing.assert_close(d_clean, d_full[:20], rtol=1e-6, atol=1e-6)
    assert torch.equal(pq.codebooks, cb_snap)
    assert torch.equal(pq.train_origin64, orig_snap)

    rvq = ApexRVQ(d=32, k1=64, k2=64, nepochs=2.0, solver="apex_emeans", adc_beam=32, seed=29)
    (c1_orig, c2_orig), rcodes_b = rvq.fit(xb, capture_codes=True)
    c1_snap, c2_snap = c1_orig.clone(), c2_orig.clone()

    rd_clean, ri_clean = rvq.adc_search(xq[:20], rcodes_b, k_top=10)
    rd_full, ri_full = rvq.adc_search(xq_corrupt, rcodes_b, k_top=10)
    assert torch.isfinite(rd_full).all()
    assert (ri_full >= 0).all()
    assert torch.equal(ri_clean, ri_full[:20])
    torch.testing.assert_close(rd_clean, rd_full[:20], rtol=1e-6, atol=1e-6)
    assert torch.equal(rvq.c1, c1_snap) and torch.equal(rvq.c2, c2_snap)


def test_adversarial_geometries_fp8_prime_shapes_and_duplicates() -> None:
    dev = _pref_device()
    gen = torch.Generator(device="cpu").manual_seed(707)

    # 1. Zeros, constants, ultra-tiny `1e-25`, huge `1e+25`, duplicates (`unique < Ks`), and `Ks=1`
    cases = [
        ("zeros", torch.zeros((512, 32), device=dev)),
        ("constant", torch.full((512, 32), 42.5, device=dev)),
        ("tiny_1e-25", (torch.randn((512, 32), generator=gen) * 1e-25).to(dev)),
        ("huge_1e25", (torch.randn((512, 32), generator=gen) * 1e25).to(dev)),
        ("duplicates", torch.randn((5, 32), generator=gen).repeat(120, 1)[:512].to(dev)),
    ]
    for name, x_case in cases:
        for prec in ("bf16", "croiss_fp8"):
            pq = ApexPQ(d=32, m=4, ks=64, nepochs=1.5, solver="apex_emeans", precision=prec, seed=13)
            cb, codes = pq.fit(x_case, capture_codes=True)
            assert torch.isfinite(cb).all(), f"Non-finite codebooks on {name} ({prec})"
            assert codes is not None and codes.shape == (512, 4)
            d_q, i_q = pq.adc_search(x_case[:16], codes, k_top=5)
            assert torch.isfinite(d_q).all() and (i_q >= 0).all()
            rec_c = pq.evaluate_reconstruction(x_case[:64], codes=codes[:64])
            assert np.isfinite(rec_c.sqnr_db)
            if name in ("tiny_1e-25", "huge_1e25"):
                uniq_per_m = [int(torch.unique(codes[:, m_i]).numel()) for m_i in range(4)]
                assert all(u > 8 for u in uniq_per_m), f"Code collapse on {name} ({prec}): {uniq_per_m}"
                assert 0.5 < rec_c.sqnr_db < 40.0, f"Unphysical SQNR {rec_c.sqnr_db} dB on {name}"

        if name in ("tiny_1e-25", "huge_1e25"):
            rvq_e = ApexRVQ(d=32, k1=32, k2=32, nepochs=1.5, solver="apex_emeans", adc_beam=16, seed=14)
            (rc1, rc2), rcodes = rvq_e.fit(x_case, capture_codes=True)
            assert torch.isfinite(rc1).all() and torch.isfinite(rc2).all()
            assert int(torch.unique(rcodes[:, 0]).numel()) > 8
            assert int(torch.unique(rcodes[:, 1]).numel()) > 8
            lut1, lut2 = rvq_e.compute_luts(x_case[:16])
            assert torch.isfinite(lut1).all() and torch.isfinite(lut2).all()
            rd_q, ri_q = rvq_e.adc_search(x_case[:16], rcodes, k_top=5)
            assert torch.isfinite(rd_q).all() and (ri_q >= 0).all()
            rrec = rvq_e.evaluate_reconstruction(x_case[:64], codes=rcodes[:64])
            assert 0.5 < rrec.sqnr_db < 40.0

    # 2. Prime `(N, D, M, d_sub, Ks)` + `d_sub > 64` FP8 + `ks = 512, 1024` (`#210` items 4 & 5, `#214` item 2)
    for n, d, m, ks, nep in [
        (317, 21, 7, 13, 0.4),     # d_sub = 3, ks = 13 < 16!
        (619, 51, 3, 19, 1.7),     # d_sub = 17, ks = 19
        (257, 16, 4, 1, 1.0),      # ks = 1 edge case
        (521, 192, 2, 64, 1.2),    # d_sub = 96 > 64 (`FULL_D=False` tiled path in BF16 & FP8!)
        (1200, 32, 4, 512, 1.0),   # ks = 512 single-CTA tiled Huygens
        (1200, 32, 4, 1024, 1.0),  # ks = 1024 subspace fallback
    ]:
        x_p = torch.randn((n, d), generator=gen, dtype=torch.float32).to(dev)
        for prec in ("bf16", "croiss_fp8"):
            pq_p = ApexPQ(d=d, m=m, ks=ks, nepochs=nep, precision=prec, seed=55)
            cb_p, codes_p = pq_p.fit(x_p, capture_codes=True)
            assert cb_p.shape == (m, ks, d // m)
            assert torch.isfinite(cb_p).all()
            rec_p = pq_p.evaluate_reconstruction(x_p[:33], codes=codes_p[:33])
            assert np.isfinite(rec_p.mse)
            dq, iq = pq_p.adc_search(x_p[:11], codes_p, k_top=min(7, n))
            assert dq.shape == (11, min(7, n)) and (iq >= 0).all()


def test_non_contiguous_views_mixed_dtypes_and_int32_stride_overflow() -> None:
    dev = _pref_device()
    raw = torch.randn((1024, 64), device=dev, dtype=torch.float32)
    x_strided = raw[::2, ::2]  # non-contiguous in BOTH dim 0 and dim 1 (`stride = (128, 2)`)!
    assert not x_strided.is_contiguous()

    pq = ApexPQ(d=32, m=4, ks=32, nepochs=1.0, seed=91)
    cb_s, codes_s = pq.fit(x_strided, capture_codes=True)
    assert cb_s.shape == (4, 32, 8) and codes_s.shape == (512, 4)

    # Transposed view `.t()`
    raw_t = torch.randn((24, 600), device=dev, dtype=torch.float32).t()  # (600, 24), stride=(1, 600)
    rvq = ApexRVQ(d=24, k1=32, k2=32, nepochs=1.0, adc_beam=16, seed=92)
    (c1, c2), rcodes = rvq.fit(raw_t, capture_codes=True)
    assert c1.shape == (32, 24) and c2.shape == (32, 24) and rcodes.shape == (600, 2)

    # In-place mutation of `codes` must invalidate cached CSR (`#219` item 4)!
    _, ids_before = rvq.adc_search(raw_t[:8], rcodes, k_top=5, adc_beam=8)
    rcodes_flipped = rcodes.flip(0).clone()
    _, ids_expected = rvq.adc_search(raw_t[:8], rcodes_flipped, k_top=5, adc_beam=8)
    # Prime cache with original `rcodes`, then mutate `rcodes` in-place!
    _ = rvq.adc_search(raw_t[:8], rcodes, k_top=5, adc_beam=8)
    rcodes.copy_(rcodes_flipped)
    _, ids_after_mut = rvq.adc_search(raw_t[:8], rcodes, k_top=5, adc_beam=8)
    assert torch.equal(ids_after_mut, ids_expected)
    assert not torch.equal(ids_before, ids_after_mut)

    # Mixed dtypes (`fp16`, `bf16`, `fp64`, numpy `float32`)
    for dt in (torch.float16, torch.bfloat16, torch.float64):
        xd = torch.randn((512, 16), device=dev, dtype=dt)
        pq_d = ApexPQ(d=16, m=4, ks=32, nepochs=1.0, seed=93)
        cbd, cd = pq_d.fit(xd, capture_codes=True)
        assert cbd.dtype == torch.float32 and cd.dtype == torch.uint8

    x_np = np.random.default_rng(94).standard_normal((512, 16)).astype(np.float32)
    pq_np = ApexPQ(d=16, m=4, ks=32, nepochs=1.0, seed=94)
    cb_np, c_np = pq_np.fit(x_np, capture_codes=True)
    assert isinstance(cb_np, np.ndarray) and isinstance(c_np, np.ndarray)
    d_np, i_np = pq_np.adc_search(x_np[:10], c_np, k_top=5)
    assert isinstance(d_np, np.ndarray) and isinstance(i_np, np.ndarray)

    # `> INT32_MAX` element offset strided tensor check on GPU (`256 * big_stride > 2^31`)
    if dev.type == "cuda":
        free_b, _ = torch.cuda.mem_get_info(dev)
        if free_b > 12 * (1024**3):
            big_stride = (1 << 23) + 1024
            n_rows = 257
            assert ((n_rows - 1) * big_stride) > (1 << 31)
            big_1d = torch.zeros(((n_rows - 1) * big_stride + 64,), dtype=torch.float32, device=dev)
            x_huge_off = torch.as_strided(big_1d, size=(n_rows, 32), stride=(big_stride, 1))
            x_huge_off[-1] = 3.5
            pq_h = ApexPQ(d=32, m=4, ks=16, nepochs=1.0, seed=95)
            cb_h, cd_h = pq_h.fit(x_huge_off, capture_codes=True)
            assert torch.isfinite(cb_h).all() and cd_h.shape == (n_rows, 4)
            del big_1d, x_huge_off


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for CUDAGraph & sync_debug_mode")
def test_cudagraph_parity_zero_sync_adc_and_multi_device_safety() -> None:
    dev = torch.device("cuda:0")
    xb, xq, _, _ = _make_clustered_data(n=4096, n_q=64, d=32, seed=808, device=dev)

    pq_eager = ApexPQ(d=32, m=8, ks=64, nepochs=2.0, use_cuda_graph=False, seed=44)
    pq_graph = ApexPQ(d=32, m=8, ks=64, nepochs=2.0, use_cuda_graph=True, force_cuda_graph=True, seed=44)
    _, c_e = pq_eager.fit(xb, capture_codes=True)
    _, c_g = pq_graph.fit(xb, capture_codes=True)
    assert "graph" in pq_graph.selected_gear
    rec_e = pq_eager.evaluate_reconstruction(xq, codes=c_e)
    rec_g = pq_graph.evaluate_reconstruction(xq, codes=c_g)
    assert abs(rec_g.mse - rec_e.mse) / max(rec_e.mse, 1e-9) < 0.03

    # Verify `warmup_pq_rvq_kernels` eliminates Seed-0 cold-JIT spike on wide-batch `N=100K, T=2.0` (`#219` item 1)
    warmup_pq_rvq_kernels(dev, pq_shapes=((64, 8, 256),), rvq_shapes=())
    xw_100k = torch.randn((100_000, 64), device=dev, dtype=torch.float32)
    t_seeds = []
    for s_i in range(3):
        pq_s = ApexPQ(d=64, m=8, ks=256, nepochs=2.0, solver="apex_emeans", seed=500 + s_i)
        pq_s.fit(xw_100k)
        t_seeds.append(pq_s.total_time_sec)
    assert t_seeds[0] <= 2.2 * max(t_seeds[1], 1e-4), f"Cold-JIT spike after warmup: {t_seeds}"
    del xw_100k

    # Verify zero host-device synchronization inside `adc_search`!
    _ = pq_graph.adc_search(xq[:8], c_g, k_top=10)
    rvq = ApexRVQ(d=32, k1=64, k2=64, nepochs=1.0, adc_beam=16, seed=45)
    _, rc = rvq.fit(xb, capture_codes=True)
    csr = rvq.build_csr_index(rc)
    _ = rvq.adc_search(xq[:8], csr, k_top=10)

    torch.cuda.synchronize(dev)
    prev_mode = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        d1, i1 = pq_graph.adc_search(xq, c_g, k_top=10)
        d2, i2 = rvq.adc_search(xq, csr, k_top=10)
        assert d1.shape == (64, 10) and i1.shape == (64, 10)
        assert d2.shape == (64, 10) and i2.shape == (64, 10)
    finally:
        torch.cuda.set_sync_debug_mode(prev_mode)

    if torch.cuda.device_count() >= 2:
        torch.cuda.set_device(0)
        dev1 = torch.device("cuda:1")
        warmup_pq_rvq_kernels(dev1, pq_shapes=((32, 4, 32),), rvq_shapes=((32, 32, 32),))
        xb1 = xb[:1024].to(dev1)
        xq1 = xq[:16].to(dev1)
        pq1 = ApexPQ(d=32, m=4, ks=32, nepochs=1.0, force_cuda_graph=True, seed=46)
        _, c1 = pq1.fit(xb1, capture_codes=True)
        d_out1, _ = pq1.adc_search(xq1, c1, k_top=5)
        assert d_out1.device == dev1

        pq1_sub = ApexPQ(d=32, m=4, ks=32, nepochs=1.0, solver="smeans", seed=47)
        _, c1_sub = pq1_sub.fit(xb1, capture_codes=True)
        assert c1_sub.device == dev1

        pq1_faiss = ApexPQ(d=32, m=4, ks=32, nepochs=1.0, solver="faiss_pq", seed=49, solver_kwargs={"niter": 3})
        cb1_f, c1_f = pq1_faiss.fit(xb1, capture_codes=True)
        assert cb1_f.device == dev1 and c1_f.device == dev1

        rvq1 = ApexRVQ(d=32, k1=32, k2=32, nepochs=1.0, solver="apex_emeans", adc_beam=16, seed=48)
        _, rc1 = rvq1.fit(xb1, capture_codes=True)
        rd1, _ = rvq1.adc_search(xq1, rc1, k_top=5)
        assert rd1.device == dev1


def test_faiss_pq_deconfounded_full_n_vs_subsampled() -> None:
    """Verify `#233` fix: `faiss_pq` trains on all `N` rows (`max_ppc = ceil(N/Ks)`), while `faiss_pq_sub` uses `256`."""
    dev = _pref_device()
    xb, xq, _, _ = _make_clustered_data(n=12_000, n_q=32, d=32, seed=919, device=dev)

    pq_full = ApexPQ(d=32, m=4, ks=32, solver="faiss_pq", seed=71, solver_kwargs={"niter": 5})
    pq_full.fit(xb)
    assert pq_full._faiss_index is not None
    assert int(pq_full._faiss_index.pq.cp.max_points_per_centroid) >= (12_000 + 31) // 32  # 375 > 256
    assert "fullN" in pq_full.selected_gear

    pq_sub = ApexPQ(d=32, m=4, ks=32, solver="faiss_pq_sub", seed=71, solver_kwargs={"niter": 5})
    pq_sub.fit(xb)
    assert pq_sub._faiss_index is not None
    assert int(pq_sub._faiss_index.pq.cp.max_points_per_centroid) == 256
    assert "sub256" in pq_sub.selected_gear
    rec_f = pq_full.evaluate_reconstruction(xq)
    rec_s = pq_sub.evaluate_reconstruction(xq)
    assert np.isfinite(rec_f.mse) and np.isfinite(rec_s.mse)


def test_all_pluggable_solvers_pq_and_rvq_end_to_end() -> None:
    dev = _pref_device()
    xb, xq, gt_ids, gt_d2 = _make_clustered_data(n=1024, n_q=32, d=32, seed=909, device=dev)

    for s_name in PQ_SUPPORTED_SOLVERS:
        pq = ApexPQ(d=32, m=4, ks=32, nepochs=1.0, solver=s_name, seed=61)
        cb, codes = pq.fit(xb, capture_codes=True)
        assert cb.shape == (4, 32, 8)
        rec = pq.evaluate_reconstruction(xq)
        adc = evaluate_adc_recall(pq, xb, codes, xq, gt_ids, gt_d2, k_top=10)
        assert np.isfinite(rec.mse) and np.isfinite(rec.sqnr_db)
        assert 0.0 <= adc.recall_at_1 <= 1.0 and 0.0 <= adc.one_recall_at_10 <= 1.0

    for s_name in RVQ_SUPPORTED_SOLVERS:
        rvq = ApexRVQ(d=32, k1=32, k2=32, nepochs=1.0, solver=s_name, adc_beam=16, seed=62)
        (c1, c2), rcodes = rvq.fit(xb, capture_codes=True)
        assert c1.shape == (32, 32) and c2.shape == (32, 32)
        rec_r = rvq.evaluate_reconstruction(xq)
        adc_r = evaluate_adc_recall(rvq, xb, rcodes, xq, gt_ids, gt_d2, k_top=10)
        assert np.isfinite(rec_r.mse) and np.isfinite(rec_r.sqnr_db)
        assert 0.0 <= adc_r.recall_at_1 <= 1.0 and 0.0 <= adc_r.one_recall_at_10 <= 1.0
