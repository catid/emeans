"""Adversarial unit, numerical-stability, bit-parity, and zero-leakage tests for `scripts/mine_scale_laws.py`."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def _load_miner_module() -> Any:
    script_path = ROOT / "scripts" / "mine_scale_laws.py"
    spec = importlib.util.spec_from_file_location("mine_scale_laws", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mine_scale_laws"] = mod
    spec.loader.exec_module(mod)
    return mod


MINER = _load_miner_module()
CUDA_AVAILABLE = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA GPU required for Blackwell Triton FP8/BF16 kernels")


# ==============================================================================
# 1. Zero Test-Set Leakage & Fleet Safety Guards
# ==============================================================================


def test_validation_problem_exposes_no_test_or_query_attributes() -> None:
    fields = set(MINER.ValidationProblem.__dataclass_fields__.keys())
    assert "x_train" in fields and "x_val" in fields
    assert "test" not in fields
    assert "query" not in fields
    assert "x_test" not in fields
    assert "groundtruth_ids" not in fields


def test_real_corpus_loader_never_reads_poisoned_test_or_query_arrays(tmp_path: Path) -> None:
    """Poison `query`, `test`, and `groundtruth_ids` with NaNs and verify loader never touches them."""
    rng = np.random.default_rng(123)
    base = rng.standard_normal((512, 32), dtype=np.float32)
    val = rng.standard_normal((128, 32), dtype=np.float32)
    poison = np.full((64, 32), np.nan, dtype=np.float32)

    npz_path = tmp_path / "sift1m_validated_v1.npz"
    np.savez(
        npz_path,
        base=base,
        val=val,
        query=poison,
        test=poison,
        groundtruth_ids=np.full((64, 10), -1, dtype=np.int32),
        groundtruth_sq_dists=np.full((64, 10), np.nan, dtype=np.float64),
    )

    dev = torch.device("cuda" if CUDA_AVAILABLE else "cpu")
    prob = MINER.load_real_validation_corpus("sift1m_val", device=dev, million_data_dir=tmp_path)
    assert prob.x_train.shape == (512, 32)
    assert prob.x_val.shape == (128, 32)
    assert prob.source_split == "real_val_only"
    assert torch.isfinite(prob.x_train).all()
    assert torch.isfinite(prob.x_val).all()


def test_approved_fleet_slots_and_safety_invariants() -> None:
    assert len(MINER.APPROVED_SLOTS) == 6
    hosts = {h for h, _ in MINER.APPROVED_SLOTS}
    assert hosts == {"ripper", "loki"}
    assert "walnut" not in " ".join(hosts).lower()
    assert set(MINER.APPROVED_SLOTS) == {
        ("ripper", 0),
        ("ripper", 1),
        ("ripper", 2),
        ("ripper", 3),
        ("loki", 0),
        ("loki", 1),
    }

    all_cells = MINER.build_scale_cells(smoke=False)
    seen_ids: set[str] = set()
    for c in all_cells:
        assert c.cell_id not in seen_ids
        seen_ids.add(c.cell_id)
        assert 0 <= c.shard_idx < 6
        assert c.wave in (1, 2)
        assert c.n >= c.k
        assert c.suite in MINER.SUITE_ARMS


def test_atomic_write_json_rejects_nan_and_inf_without_tmp_leaks(tmp_path: Path) -> None:
    out = tmp_path / "clean.json"
    MINER.atomic_write_json(out, {"ok": 1.25, "nested": [1, 2.0]})
    assert out.exists()
    assert list(tmp_path.glob(".tmp_*")) == []

    for bad_val in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="Non-finite"):
            MINER.atomic_write_json(tmp_path / "bad.json", {"x": [1.0, bad_val]})
    assert not (tmp_path / "bad.json").exists()
    assert list(tmp_path.glob(".tmp_*")) == []


# ==============================================================================
# 2. CroissAttn-FP8 & BF16 Kernel Numerical Stability & Exact Payload Identity
# ==============================================================================


@requires_cuda
@pytest.mark.parametrize(
    "case",
    ["zeros", "constant", "extreme_1e20", "shift_20000", "multiscale_sensor", "prime_shape"],
)
def test_croiss_fp8_encode_zero_nan_across_adversarial_inputs(case: str) -> None:
    dev = torch.device("cuda")
    if case == "zeros":
        x = torch.zeros((257, 73), device=dev, dtype=torch.float32)
    elif case == "constant":
        x = torch.full((257, 73), -42.5, device=dev, dtype=torch.float32)
    elif case == "extreme_1e20":
        gen = torch.Generator(device=dev).manual_seed(70_101)
        x = torch.randn((257, 73), generator=gen, device=dev, dtype=torch.float32) * 1e20
    elif case == "shift_20000":
        gen = torch.Generator(device=dev).manual_seed(70_102)
        x = torch.randn((257, 73), generator=gen, device=dev, dtype=torch.float64) + 20_000.0
    elif case == "multiscale_sensor":
        prob = MINER.make_gpu_validation_problem("multiscale_sensor", 513, 97, 64, seed=70_103, device=dev)
        x = prob.x_train
    else:
        gen = torch.Generator(device=dev).manual_seed(70_104)
        x = torch.randn((1019, 313), generator=gen, device=dev, dtype=torch.float32)

    orig64 = x.double().mean(dim=0).contiguous()
    orig_in = orig64 if (x.dtype == torch.float64 or float(orig64.abs().max().item()) > 1e3) else orig64.float()
    qx, sx, x_norm = MINER.encode_croiss_fp8_dataset(x, orig_in)

    qx_u8 = qx.view(torch.uint8)
    assert int(((qx_u8 & 0x7F) == 0x7F).sum().item()) == 0, "Generated 0x7F/0xFF E4M3FN NaN!"
    assert torch.isfinite(sx).all() and bool((sx > 0).all())
    assert torch.isfinite(x_norm).all() and bool((x_norm >= 0).all())
    # Padded trailing columns must be exact +0.0
    if qx.shape[1] > x.shape[1]:
        assert float(qx[:, x.shape[1] :].float().abs().max().item()) == 0.0


@requires_cuda
@pytest.mark.parametrize(
    "shape",
    [
        (251, 131, 37),     # Prime FULL_D
        (300, 256, 256),    # FP8 FULL_D=True vs BF16 FULL_D=False boundary
        (193, 2053, 128),   # Triggers 2D Split-K (`K >= 2048` and small `B`)!
        (257, 389, 521),    # Prime Wide-D (`FULL_D=False`, 3-stage tiled)
    ],
)
@pytest.mark.parametrize("is_fp8", [False, True])
def test_gear_assign_matches_represented_payload_fp64_reference(
    shape: tuple[int, int, int],
    is_fp8: bool,
) -> None:
    B, K, D = shape
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(70_200 + B + K + D)
    x = torch.randn((B + 64, D), generator=gen, device=dev, dtype=torch.float32) * 3.7 + 12.0
    orig = x.double().mean(dim=0).float().contiguous()
    centroids_c = torch.randn((K, D), generator=gen, device=dev, dtype=torch.float32) * 3.2

    if is_fp8:
        x_enc, sx, x_norm = MINER.encode_croiss_fp8_dataset(x, orig)
        d_pad = x_enc.shape[1]
        c_rep = torch.empty((K, d_pad), dtype=MINER.FLOAT8_E4M3FN, device=dev)
    else:
        x_enc, x_norm = MINER.encode_centered_bf16_dataset(x, orig)
        sx = x_norm
        d_pad = x_enc.shape[1]
        c_rep = torch.empty((K, d_pad), dtype=torch.bfloat16, device=dev)

    sc = torch.ones((K,), dtype=torch.float32, device=dev)
    c_norm = torch.empty((K,), dtype=torch.float32, device=dev)
    blk_d = min(256, max(32 if is_fp8 else 16, MINER.triton.next_power_of_2(D)))
    MINER._gear_prep_centroids_kernel[(K,)](
        centroids_c, c_rep, sc, c_norm, K, D, c_rep.stride(0), IS_FP8=is_fp8, BLOCK_D=blk_d, num_warps=4
    )

    order_buf = torch.randperm(x.shape[0], generator=gen, device=dev)
    batch_off = torch.tensor(17, dtype=torch.int64, device=dev)
    out_d = torch.empty((B,), dtype=torch.float32, device=dev)
    out_l = torch.empty((B,), dtype=torch.int64, device=dev)

    num_sms = torch.cuda.get_device_properties(dev).multi_processor_count
    parts = MINER._choose_split_k_partitions(B, K, num_sms=num_sms, block_b=64, block_k=128) if K >= 2048 else 1
    sk = (
        (
            torch.empty((B, parts), dtype=torch.float32, device=dev),
            torch.empty((B, parts), dtype=torch.int32, device=dev),
            parts,
        )
        if parts > 1
        else None
    )
    MINER._launch_gear_assign(
        x_enc, order_buf, batch_off, sx, x_norm, c_rep, sc, c_norm, out_d, out_l, B, K, D, is_fp8, num_sms, sk
    )

    # Exact FP64 Euclidean distance on the quantized/represented vectors!
    rows = order_buf[17 : 17 + B]
    if is_fp8:
        x_tilde = x_enc[rows, :D].double() * sx[rows, None].double()
        c_tilde = c_rep[:, :D].double() * sc[:, None].double()
    else:
        x_tilde = x_enc[rows, :D].double()
        c_tilde = c_rep[:, :D].double()

    d2_ref = torch.cdist(x_tilde, c_tilde, p=2.0).square()
    min_d2_ref, lbl_ref = d2_ref.min(dim=1)

    torch.testing.assert_close(out_d.double(), min_d2_ref, rtol=2e-4, atol=2e-3)
    # Check label match (allowing near-ties within 1e-4 relative gap)
    chosen_d2 = d2_ref.gather(1, out_l[:, None]).squeeze(1)
    torch.testing.assert_close(chosen_d2, min_d2_ref, rtol=1e-4, atol=1e-4)


# ==============================================================================
# 3. Bit-Exact `CUDAGraph` vs Eager Parity & `+20,000` Coordinate-Shift Equivariance
# ==============================================================================


@requires_cuda
@pytest.mark.parametrize("precision_prefix", ["bf16", "fp8_croiss"])
@pytest.mark.parametrize("nepochs", [1.0, 2.3])  # 2.3 exercises non-divisible epoch-boundary % 5 alignment!
def test_cudagraph_bit_exact_parity_vs_eager(precision_prefix: str, nepochs: float) -> None:
    dev = torch.device("cuda")
    prob = MINER.make_gpu_validation_problem(
        "multiscale_sensor", n_train=4111, d=73, k=131, seed=70_301, device=dev
    )
    orig64, _ = MINER._compute_origin_and_scale(prob.x_train, torch.bfloat16)
    sres = MINER.adaptive_turkpp_init_centered(
        prob.x_train, orig64, 1.0, 131, torch.bfloat16, seed=70_301, huge_k_fast=False
    )

    c_eager, m_eager = MINER.run_gear1_arm(
        f"{precision_prefix}_eager", prob.x_train, orig64, sres, 1.0, 131, nepochs, seed=70_301
    )
    c_graph, m_graph = MINER.run_gear1_arm(
        f"{precision_prefix}_cudagraph", prob.x_train, orig64, sres, 1.0, 131, nepochs, seed=70_301
    )

    assert m_eager["fp8_nan_count"] == 0 and m_graph["fp8_nan_count"] == 0
    assert m_eager["n_splits"] == m_graph["n_splits"]
    assert m_eager["dead_clusters"] == m_graph["dead_clusters"]
    if precision_prefix == "bf16":
        # Concurrent CTA FP32 atomic_add has ~1 ULP order jitter; verify 6-digit centroid equivalence
        torch.testing.assert_close(c_eager, c_graph, rtol=1e-5, atol=5e-4)
    else:
        # E4M3FN (3-bit mantissa) can flip 1 boundary point under 1-ULP atomic_add jitter; verify <0.1% val MSE parity
        mse_e, _ = MINER.evaluate_validation_mse_fp64(prob.x_val, c_eager, prob.geometry, dev)
        mse_g, _ = MINER.evaluate_validation_mse_fp64(prob.x_val, c_graph, prob.geometry, dev)
        assert abs(mse_e - mse_g) / max(mse_e, 1e-12) < 1e-3


@requires_cuda
@pytest.mark.parametrize("arm", ["bf16_cudagraph", "fp8_croiss_cudagraph"])
def test_plus_20000_coordinate_shift_equivariance_fp64(arm: str) -> None:
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(70_401)
    centers = torch.randn((64, 48), generator=gen, device=dev, dtype=torch.float64) * 1.5
    ids = torch.randint(0, 64, (2048,), generator=gen, device=dev)
    x_base = (centers[ids] + 0.3 * torch.randn((2048, 48), generator=gen, device=dev, dtype=torch.float64)).contiguous()
    x_val = (centers[:64].repeat(8, 1) + 0.3 * torch.randn((512, 48), generator=gen, device=dev, dtype=torch.float64)).contiguous()

    shift = 20_000.0
    x_shift = (x_base + shift).contiguous()
    x_val_shift = (x_val + shift).contiguous()

    orig_b, _ = MINER._compute_origin_and_scale(x_base, torch.bfloat16)
    sres_b = MINER.adaptive_turkpp_init_centered(x_base, orig_b, 1.0, 64, torch.bfloat16, seed=70_402)
    c_base, _ = MINER.run_gear1_arm(arm, x_base, orig_b, sres_b, 1.0, 64, 2.0, seed=70_402)

    orig_s, _ = MINER._compute_origin_and_scale(x_shift, torch.bfloat16)
    sres_s = MINER.adaptive_turkpp_init_centered(x_shift, orig_s, 1.0, 64, torch.bfloat16, seed=70_402)
    c_shift, _ = MINER.run_gear1_arm(arm, x_shift, orig_s, sres_s, 1.0, 64, 2.0, seed=70_402)

    mse_b, _ = MINER.evaluate_validation_mse_fp64(x_val, c_base, "euclidean", dev)
    mse_s, _ = MINER.evaluate_validation_mse_fp64(x_val_shift, c_shift, "euclidean", dev)
    if arm == "bf16_cudagraph":
        torch.testing.assert_close(c_shift - shift, c_base, rtol=1e-5, atol=1e-5)
        assert abs(mse_b - mse_s) / max(mse_b, 1e-12) < 1e-7
    else:
        assert abs(mse_b - mse_s) / max(mse_b, 1e-12) < 2e-3


# ==============================================================================
# 4. Gear 2 `IVF -> Exact` & Huge-K Fast Seeding + End-to-End Smoke Verification
# ==============================================================================


@requires_cuda
def test_huge_k_fast_seeding_and_ivf_exact_close_pipeline() -> None:
    dev = torch.device("cuda")
    prob = MINER.make_gpu_validation_problem("balanced_gmm", n_train=24_576, d=64, k=8192, seed=70_501, device=dev)
    orig64, _ = MINER._compute_origin_and_scale(prob.x_train, torch.bfloat16)
    sres = MINER.adaptive_turkpp_init_centered(
        prob.x_train, orig64, 1.0, 8192, torch.bfloat16, seed=70_501, huge_k_fast=True
    )
    assert sres.centroids.shape == (8192, 64)
    assert torch.isfinite(sres.centroids).all()
    assert math.isclose(float(sres.prior_probs.sum().item()), 1.0, rel_tol=1e-4)

    c_ivf_ex, m_ivf_ex = MINER.run_gear2_ivf_arm(
        "ivf2sq_exact_np8", prob.x_train, orig64, sres, 1.0, 8192, 2.0, seed=70_501
    )
    c_pure, _ = MINER.run_gear2_ivf_arm(
        "ivf_pure_np8", prob.x_train, orig64, sres, 1.0, 8192, 2.0, seed=70_501
    )
    assert torch.isfinite(c_ivf_ex).all() and torch.isfinite(c_pure).all()
    assert m_ivf_ex["nlist"] == 256 and m_ivf_ex["nprobe"] == 8
    mse_ex, _ = MINER.evaluate_validation_mse_fp64(prob.x_val, c_ivf_ex, "euclidean", dev)
    mse_pu, _ = MINER.evaluate_validation_mse_fp64(prob.x_val, c_pure, "euclidean", dev)
    assert math.isfinite(mse_ex) and math.isfinite(mse_pu)
    assert mse_ex <= mse_pu * 1.01


@requires_cuda
def test_end_to_end_smoke_pipeline_and_plot_verification(tmp_path: Path) -> None:
    dev = torch.device("cuda")
    cells = MINER.build_scale_cells(smoke=True)
    MINER.warmup_global_kernels(dev)
    recs: list[dict[str, Any]] = []
    for c in cells:
        recs.extend(MINER.execute_cell(c, seed_start=70_600, device=dev))
    MINER.validate_scale_records(recs, cells)
    fitted = MINER.fit_scale_laws(recs)
    MINER.atomic_write_json(tmp_path / "res.json", {"fitted": fitted, "records": recs})
    MINER.write_scale_laws_summary_md(fitted, len(recs), tmp_path / "summary.md")
    plots = MINER.generate_scale_law_plots(fitted, tmp_path)
    assert len(plots) == 4
    for p in plots:
        assert p.exists()
        with Image.open(p) as im:
            im.verify()
        assert (tmp_path / "plots" / p.name).exists()


@requires_cuda
def test_multiwave_k16384_no_inter_cta_step_race() -> None:
    """Regression test for `#170` Bug 1: across K = 16,384 (> 188 SMs, 87 CTA waves),
    ALL 16,384 CTAs must read `step = 0` before `_gear_advance_scalars_kernel` increments `step` to 1!"""
    dev = torch.device("cuda")
    K, D = 16_384, 32
    sums = torch.full((K, D), 10.0, dtype=torch.float32, device=dev)
    counts = torch.ones((K,), dtype=torch.float32, device=dev)
    batch_sse = torch.ones((K,), dtype=torch.float32, device=dev)
    centroids = torch.zeros((K, D), dtype=torch.float32, device=dev)
    usage = torch.ones((K,), dtype=torch.float32, device=dev)
    sse_ema = torch.ones((K,), dtype=torch.float32, device=dev)
    slow_centroids = torch.zeros((K, D), dtype=torch.float32, device=dev)
    pass_sums = torch.zeros((K, D), dtype=torch.float32, device=dev)
    pass_counts = torch.zeros((K,), dtype=torch.float64, device=dev)
    pass_sse = torch.zeros((K,), dtype=torch.float64, device=dev)
    c_rep = torch.empty((K, D), dtype=torch.bfloat16, device=dev)
    sc = torch.ones((K,), dtype=torch.float32, device=dev)
    c_norm = torch.empty((K,), dtype=torch.float32, device=dev)

    step_buf = torch.zeros((), dtype=torch.int32, device=dev)
    batch_off_buf = torch.zeros((), dtype=torch.int64, device=dev)
    accum_pass_buf = torch.zeros((), dtype=torch.int32, device=dev)
    usage_bias_buf = torch.full((), 0.5, dtype=torch.float32, device=dev)
    sse_bias_buf = torch.full((), 0.5, dtype=torch.float32, device=dev)

    # At step=0, total_steps=2, alpha0=0.5, alpha_min_factor=0.1 -> alpha = 0.5, decay = 0.5:
    # u_new = 0.5 * 1.0 + 0.5 * 1.0 = 1.0, w_ema = 0.5 / 1.0 = 0.5 -> c_next = 0 + 0.5 * 10.0 = 5.0!
    MINER._gear_ondevice_ema_prep_kernel[(K,)](
        sums, counts, batch_sse, centroids, usage, sse_ema, slow_centroids,
        pass_sums, pass_counts, pass_sse, c_rep, sc, c_norm,
        step_buf, accum_pass_buf, K, D, c_rep.stride(0),
        2, 0.5, 0.1, 0.05, 1, IS_FP8=False, BLOCK_D=32, num_warps=2,
    )
    MINER._gear_advance_scalars_kernel[(1,)](
        step_buf, batch_off_buf, usage_bias_buf, sse_bias_buf,
        256, 2, 0.5, 0.1, num_warps=1,
    )
    assert int(step_buf.item()) == 1
    assert int(batch_off_buf.item()) == 256
    # Every single one of the 16,384 centroids across all 88 CTA waves must equal exact 5.0!
    assert float((centroids - 5.0).abs().max().item()) == 0.0


@requires_cuda
def test_k16384_cudagraph_vs_eager_multiwave_parity() -> None:
    """End-to-end `K = 16,384` multi-wave `CUDAGraph` vs `Eager` parity check (`#170` Bug 1)."""
    dev = torch.device("cuda")
    prob = MINER.make_gpu_validation_problem("balanced_gmm", n_train=20_480, d=32, k=16_384, seed=70_701, device=dev)
    orig64, _ = MINER._compute_origin_and_scale(prob.x_train, torch.bfloat16)
    sres = MINER.adaptive_turkpp_init_centered(
        prob.x_train, orig64, 1.0, 16_384, torch.bfloat16, seed=70_701, huge_k_fast=True
    )
    c_e, m_e = MINER.run_gear1_arm("bf16_eager", prob.x_train, orig64, sres, 1.0, 16_384, 1.0, seed=70_701)
    c_g, m_g = MINER.run_gear1_arm("bf16_cudagraph", prob.x_train, orig64, sres, 1.0, 16_384, 1.0, seed=70_701)
    assert m_e["n_splits"] == m_g["n_splits"]
    torch.testing.assert_close(c_e, c_g, rtol=1e-5, atol=5e-4)


def test_wave2_merge_automatically_includes_wave1_shards(tmp_path: Path) -> None:
    """Regression test for `#170` Bug 2: `--wave 2` merge preserves all 47 Wave-1 cells when w1 shards exist."""
    all_cells = MINER.build_scale_cells(smoke=False)
    shards_dir = tmp_path / "shards"
    shards_dir.mkdir(parents=True)

    # Without w1 shards, wave="2" loads only w2
    w_tags_only2, cells_only2 = MINER.resolve_waves_to_merge("2", shards_dir, all_cells)
    assert w_tags_only2 == ["w2"]
    assert all(c.wave == 2 for c in cells_only2)

    # Once all 6 w1 shard files exist, wave="2" merges BOTH ["w1", "w2"] across all 77 cells!
    for sidx in range(len(MINER.APPROVED_SLOTS)):
        (shards_dir / f"records_w1_shard_{sidx:02d}.json").write_text("{}", encoding="utf-8")
    w_tags_both, cells_both = MINER.resolve_waves_to_merge("2", shards_dir, all_cells)
    assert w_tags_both == ["w1", "w2"]
    assert len(cells_both) == len(all_cells)


def test_fit_scale_laws_separates_fulld_from_wide_d_and_finds_k16384_crossover() -> None:
    """Regression test for `#185`: `fit_scale_laws` must fit `FULL_D` (`D <= 256`) without pooling
    `D >= 512` indirect-gather fallback rows, and must detect `k_star_vs_exact_fp8 == 16384`."""
    res_path = Path(__file__).resolve().parents[1] / "outputs" / "scale_laws" / "scale_laws_results.json"
    if not res_path.exists():
        pytest.skip("scale_laws_results.json not yet generated")
    import json

    with res_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    fitted = MINER.fit_scale_laws(payload["records"])
    fp8_l = fitted["gear1_fp8_cudagraph_law"]
    ivf_l = fitted["gear2_ivf_exact_law"]
    assert fp8_l["fit_regime"] == "FULL_D_1pass_d_le_256"
    assert 1e7 <= fp8_l["crossover_nkt_at_d128"] <= 5e8
    assert 5e7 <= fp8_l["practical_105x_nkt_at_d128"] <= 5e8
    assert ivf_l["k_star_vs_exact_fp8"] == 16384

    w1_path = res_path.parent / "scale_laws_wave1_results.json"
    if w1_path.exists():
        with w1_path.open("r", encoding="utf-8") as fh:
            w1_payload = json.load(fh)
        w1_fp8 = w1_payload["fitted_laws"]["gear1_fp8_cudagraph_law"]
        w1_ivf = w1_payload["fitted_laws"]["gear2_ivf_exact_law"]
        assert w1_fp8["fit_regime"] == "FULL_D_1pass_d_le_256"
        assert w1_ivf["k_star_vs_exact_fp8"] == 16384

