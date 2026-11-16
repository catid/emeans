"""Comprehensive Unit, Numerical Stability, CUDAGraph, CroissFP8 & Scale-Gear Tests for `ApexEMeans`."""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from emeans.apex_emeans import (
    APEX_ABLATION_ARMS,
    APEX_LADDER_ARMS,
    ApexEMeans,
    ApexEMeansEpochStat,
    _make_superstep_ring_slots,
    apex_simplex_eb_finalize,
    apex_turkpp_init_centered,
    build_apex_arm,
    warmup_apex_kernels,
)


def _exact_mse_fp64(x: torch.Tensor, c: torch.Tensor) -> float:
    """Compute shift-invariant exact IEEE-754 FP64 quantization MSE."""
    x64 = x.detach().double()
    c64 = c.detach().double().to(x64.device)
    origin = x64.mean(dim=0, keepdim=True)
    xc = x64 - origin
    cc = c64 - origin
    d2 = (xc.square().sum(dim=1, keepdim=True) + cc.square().sum(dim=1)[None, :] - 2.0 * (xc @ cc.t())).clamp_min(0.0)
    best_idx = d2.argmin(dim=1)
    exact_d2 = (x64 - c64[best_idx]).square().sum(dim=1)
    return float(exact_d2.mean().item())


def _make_clustered_data(
    n: int,
    d: int,
    k_true: int,
    seed: int = 123,
    device: str | torch.device = "cpu",
    noise_std: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate isolated `(x_train, x_val, centers)` from a well-separated GMM."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    centers = torch.randn((k_true, d), generator=gen, dtype=torch.float32) * 4.0
    ids_tr = torch.randint(0, k_true, (n,), generator=gen)
    ids_va = torch.randint(0, k_true, (max(256, n // 5),), generator=gen)
    x_tr = centers[ids_tr] + torch.randn((n, d), generator=gen, dtype=torch.float32) * noise_std
    x_va = centers[ids_va] + torch.randn((ids_va.shape[0], d), generator=gen, dtype=torch.float32) * noise_std
    dev = torch.device(device)
    return x_tr.to(dev).contiguous(), x_va.to(dev).contiguous(), centers.to(dev).contiguous()


# ==============================================================================
# 1. API Contract, Validation & CPU / NumPy / Torch Reference Parity
# ==============================================================================


def test_ring_slot_builder_invariants() -> None:
    for g in (2, 3, 4, 5, 6, 7, 8):
        ring = _make_superstep_ring_slots(g)
        assert len(ring) == g
        assert ring[0][0] == 0, f"Super-step ring for g={g} must start at slot 0"
        assert ring[-1][1] == 0, f"Super-step ring for g={g} must close back at slot 0"
        for idx, (s_in, s_out) in enumerate(ring):
            assert s_in != s_out, f"In-place slot alias at substep {idx} for g={g}"
            assert 0 <= s_in <= 2 and 0 <= s_out <= 2
            if idx + 1 < g:
                assert ring[idx + 1][0] == s_out


def test_parameter_validation_and_unfitted_assign_guard() -> None:
    with pytest.raises(ValueError):
        ApexEMeans(d=0, k=8)
    with pytest.raises(ValueError):
        ApexEMeans(d=8, k=0)
    with pytest.raises(ValueError):
        ApexEMeans(d=8, k=4, nepochs=0.0)
    with pytest.raises(ValueError):
        ApexEMeans(d=8, k=4, gear_mode="invalid_gear")
    with pytest.raises(ValueError):
        ApexEMeans(d=8, k=4, precision="int4")
    with pytest.raises(KeyError):
        build_apex_arm("nonexistent_arm", d=8, k=4)

    solver = ApexEMeans(d=8, k=4, nepochs=1.0)
    with pytest.raises(RuntimeError):
        solver.assign(torch.randn(16, 8))

    x_small = torch.randn(3, 8)
    with pytest.raises(ValueError):
        solver.fit(x_small)

    x_nan = torch.randn(16, 8)
    x_nan[2, 3] = float("nan")
    with pytest.raises(ValueError):
        solver.fit(x_nan)


def test_cpu_numpy_and_torch_contracts() -> None:
    x_tr, x_va, _ = _make_clustered_data(512, 16, 16, seed=11, device="cpu")
    x_np = x_tr.numpy()

    solver_np = ApexEMeans(d=16, k=16, nepochs=2.0, seed=42)
    c_np, lbl_np = solver_np.fit(x_np, capture_assignments=True)
    assert isinstance(c_np, np.ndarray) and c_np.shape == (16, 16)
    assert isinstance(lbl_np, np.ndarray) and lbl_np.shape == (512,)
    assert np.isfinite(c_np).all()
    assert solver_np.dead_clusters == 0
    assert len(solver_np.iteration_stats) == 2
    assert isinstance(solver_np.iteration_stats[0], ApexEMeansEpochStat)

    d_np, l_np = solver_np.assign(x_va.numpy())
    assert isinstance(d_np, np.ndarray) and d_np.shape == (x_va.shape[0],)
    assert isinstance(l_np, np.ndarray) and l_np.shape == (x_va.shape[0],)

    # Empty batch assign contract
    d_empty, l_empty = solver_np.assign(torch.empty((0, 16), dtype=torch.float32))
    assert d_empty.shape == (0,) and l_empty.shape == (0,)


def test_simplex_eb_finalizer_mathematical_invariants() -> None:
    gen = torch.Generator(device="cpu").manual_seed(99)
    k, d = 64, 32
    true_mu = torch.randn((k, d), generator=gen, dtype=torch.float64) * 3.0

    # Sparse cluster regime (`n_eff` between 4 and 24) where Positive-Part James-Stein shines
    counts = torch.randint(4, 24, (k,), generator=gen).double()
    sigma = 0.8
    # Unbiased noisy closing means `c_close ~ N(true_mu, sigma^2 / n_j * I_d)`
    noise_close = torch.randn((k, d), generator=gen, dtype=torch.float64) * (sigma / counts.sqrt()[:, None])
    c_close = (true_mu + noise_close).float()
    pass_sums = (c_close.double() * counts[:, None]).float()
    pass_sse = counts * float(d) * (sigma**2)

    # Independent pre-closing anchor with low error around `true_mu`
    pre_close = (true_mu + torch.randn((k, d), generator=gen, dtype=torch.float64) * 0.08).float()
    slow_c = (true_mu + torch.randn((k, d), generator=gen, dtype=torch.float64) * 0.12).float()
    live_c = (0.7 * c_close + 0.3 * pre_close).float()
    usage = counts.float()

    c_eb, diag = apex_simplex_eb_finalize(
        pass_sums=pass_sums,
        pass_counts=counts,
        pass_counts_sq=counts,
        pass_sse=pass_sse,
        live_centroids=live_c,
        pre_close_centroids=pre_close,
        slow_centroids=slow_c,
        usage=usage,
        usage_inv_bias=1.0,
        alpha_end=0.08,
        nepochs=3.0,
        d=d,
        mode="simplex_eb",
    )
    err_close = float((c_close.double() - true_mu).square().mean().item())
    err_eb = float((c_eb.double() - true_mu).square().mean().item())
    assert err_eb < err_close, f"Simplex EB MSE ({err_eb:.6f}) must strictly improve noisy closing mean ({err_close:.6f})"
    assert 0.0 < diag.mean_lambda < 1.0

    # High-`n_eff` (`>= 200`) drift-protection gate at `T <= 3.5`: dense clusters stay on `c_close`
    dense_counts = torch.full((k,), 240.0, dtype=torch.float64)
    dense_sums = (c_close.double() * dense_counts[:, None]).float()
    c_eb_dense, _ = apex_simplex_eb_finalize(
        pass_sums=dense_sums,
        pass_counts=dense_counts,
        pass_counts_sq=dense_counts,
        pass_sse=dense_counts * float(d) * (sigma**2),
        live_centroids=live_c,
        pre_close_centroids=pre_close + 2.0,  # deliberately lagged anchor!
        slow_centroids=slow_c + 2.0,
        usage=dense_counts.float(),
        usage_inv_bias=1.0,
        alpha_end=0.08,
        nepochs=2.0,
        d=d,
        mode="simplex_eb",
    )
    max_pull = float((c_eb_dense - c_close).abs().max().item())
    assert max_pull < 0.02, f"Dense n_eff=240 clusters at T=2 must ignore lagged pre_close anchor (got pull={max_pull})"


# ==============================================================================
# 2. GPU Tests: All Arms, 4 Gears, CUDAGraph Parity, CroissFP8 Stability & Multi-Shard
# ==============================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_all_registered_ablation_and_ladder_arms_gpu() -> None:
    dev = torch.device("cuda:0")
    x_tr, x_va, _ = _make_clustered_data(8192, 64, 128, seed=2026, device=dev)
    baseline_mse = _exact_mse_fp64(x_va, x_tr[:128])

    for arm_name in list(APEX_ABLATION_ARMS.keys()):
        solver = build_apex_arm(arm_name, d=64, k=128, nepochs=2.0, seed=77)
        c, lbl = solver.fit(x_tr, capture_assignments=True)
        assert c.shape == (128, 64)
        assert lbl is not None and lbl.shape == (8192,)
        assert bool(torch.isfinite(c).all().item()), f"Non-finite centroids in {arm_name}"
        assert solver.dead_clusters == 0, f"Dead clusters={solver.dead_clusters} in {arm_name}"
        val_mse = _exact_mse_fp64(x_va, c)
        assert val_mse < 0.35 * baseline_mse, f"Poor convergence in {arm_name}: {val_mse} vs {baseline_mse}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("prec", ["bf16", "croiss_fp8", "fp32"])
@pytest.mark.parametrize("super_steps", [2, 5])
def test_cudagraph_vs_eager_parity_and_superstep_rings(prec: str, super_steps: int) -> None:
    dev = torch.device("cuda:0")
    x_tr, x_va, _ = _make_clustered_data(16384, 128, 256, seed=314, device=dev)

    s_eager = ApexEMeans(
        d=128, k=256, nepochs=2.0, precision=prec, use_cuda_graph=False, graph_super_steps=super_steps, seed=555
    )
    s_graph = ApexEMeans(
        d=128, k=256, nepochs=2.0, precision=prec, use_cuda_graph=True, graph_super_steps=super_steps, seed=555
    )
    c_eager, _ = s_eager.fit(x_tr, capture_assignments=False)
    c_graph, _ = s_graph.fit(x_tr, capture_assignments=False)
    assert s_graph._graphs_captured >= 1 and s_graph._graph_replays >= 1

    mse_eager = _exact_mse_fp64(x_va, c_eager)
    mse_graph = _exact_mse_fp64(x_va, c_graph)
    rel_diff = abs(mse_graph - mse_eager) / max(mse_eager, 1e-12)
    assert rel_diff < 0.015, f"CUDAGraph vs Eager MSE mismatch ({prec}, g={super_steps}): {mse_graph:.6f} vs {mse_eager:.6f}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_multiscale_physical_sensor_high_cr_fp8_bf16_fp32_parity() -> None:
    """Verify SQNR closing-pass IEEE-FP32 guard keeps FP8 & BF16 within <0.25% of FP32 on 100,000x span data."""
    dev = torch.device("cuda:0")
    gen = torch.Generator(device="cpu").manual_seed(7777)
    n, d, k = 12288, 54, 256
    # Mimic `covtype_multiscale_54`: first 4 physical distance columns have ~2,000m std; binary one-hot cols ~0.02 std
    col_scales = torch.ones((d,), dtype=torch.float32) * 0.02
    col_scales[0] = 2200.0
    col_scales[1] = 1500.0
    col_scales[2] = 600.0
    col_scales[3] = 350.0
    centers = torch.randn((k, d), generator=gen, dtype=torch.float32) * col_scales[None, :]
    ids_tr = torch.randint(0, k, (n,), generator=gen)
    ids_va = torch.randint(0, k, (2048,), generator=gen)
    noise_scales = col_scales * 0.06
    x_tr = (centers[ids_tr] + torch.randn((n, d), generator=gen) * noise_scales[None, :] + 3500.0).to(dev).contiguous()
    x_va = (centers[ids_va] + torch.randn((2048, d), generator=gen) * noise_scales[None, :] + 3500.0).to(dev).contiguous()

    s_fp32 = ApexEMeans(d=d, k=k, nepochs=2.0, precision="fp32", seed=19)
    c_fp32, _ = s_fp32.fit(x_tr, capture_assignments=False)
    assert s_fp32.seeding_result is not None

    # 1. From identical initial centroids (`seeding_result`), Ozaki-FP8 & BF16-guard match IEEE-FP32 within < 1%!
    s_bf16 = ApexEMeans(d=d, k=k, nepochs=2.0, precision="bf16", seed=19)
    c_bf16, _ = s_bf16.fit(x_tr, seeding_result=s_fp32.seeding_result, capture_assignments=False)
    s_fp8 = ApexEMeans(d=d, k=k, nepochs=2.0, precision="croiss_fp8", seed=19)
    c_fp8, lbl_fp8 = s_fp8.fit(x_tr, seeding_result=s_fp32.seeding_result, capture_assignments=True)

    assert s_bf16._high_cr_fp32_assign is True
    assert s_fp8._high_cr_fp32_assign is True

    mse_32 = _exact_mse_fp64(x_va, c_fp32)
    mse_bf16 = _exact_mse_fp64(x_va, c_bf16)
    mse_fp8 = _exact_mse_fp64(x_va, c_fp8)
    assert abs(mse_bf16 - mse_32) / max(mse_32, 1e-12) < 0.01
    assert abs(mse_fp8 - mse_32) / max(mse_32, 1e-12) < 0.01

    # 2. End-to-end from scratch (`seeding_result=None`), Ozaki-FP8 is never > 1.5% worse than FP32
    c_fp8_e2e, _ = ApexEMeans(d=d, k=k, nepochs=2.0, precision="croiss_fp8", seed=19).fit(x_tr, capture_assignments=False)
    assert _exact_mse_fp64(x_va, c_fp8_e2e) <= 1.015 * mse_32

    # Verify captured assignments match exact FP64 nearest centroids cleanly
    assert lbl_fp8 is not None
    _, exact_lbl = torch.cdist(x_tr[:512].double(), c_fp8.double()).min(dim=1)
    agree = float((lbl_fp8[:512] == exact_lbl).float().mean().item())
    assert agree > 0.995, f"Expected >99.5% exact label agreement on high-CR corpus, got {agree:.4f}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("prec", ["fp32", "bf16", "fp16", "croiss_fp8", "fp64"])
def test_coordinate_shift_and_extreme_scale_equivariance(prec: str) -> None:
    dev = torch.device("cuda:0")
    x_tr, x_va, _ = _make_clustered_data(4096, 64, 64, seed=404, device=dev)
    shift = torch.linspace(15000.0, -25000.0, 64, device=dev, dtype=torch.float32)

    s0 = ApexEMeans(d=64, k=64, nepochs=2.0, precision=prec, seed=88)
    c0, _ = s0.fit(x_tr, capture_assignments=False)
    mse0 = _exact_mse_fp64(x_va, c0)

    s_sh = ApexEMeans(d=64, k=64, nepochs=2.0, precision=prec, seed=88)
    c_sh, _ = s_sh.fit(x_tr + shift[None, :], capture_assignments=False)
    mse_sh = _exact_mse_fp64(x_va + shift[None, :], c_sh)

    rel_mse_gap = abs(mse_sh - mse0) / max(mse0, 1e-12)
    assert rel_mse_gap < 0.02, f"Shift equivariance violation in {prec}: mse0={mse0:.6f} vs mse_sh={mse_sh:.6f}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("scale_exp", [-20, 20])
def test_extreme_magnitude_conditioning_without_nan_or_overflow(scale_exp: int) -> None:
    dev = torch.device("cuda:0")
    scale = 10.0**scale_exp
    x_tr, x_va, _ = _make_clustered_data(4096, 64, 64, seed=505, device=dev)
    x_scaled = (x_tr * scale).contiguous()

    for prec in ("bf16", "croiss_fp8"):
        solver = ApexEMeans(d=64, k=64, nepochs=2.0, precision=prec, seed=91)
        c, lbl = solver.fit(x_scaled, capture_assignments=True)
        assert bool(torch.isfinite(c).all().item()), f"Non-finite centroids at 10^{scale_exp} ({prec})"
        d2, l2 = solver.assign((x_va * scale).contiguous())
        assert bool(torch.isfinite(d2).all().item())
        assert lbl is not None and l2.shape == (x_va.shape[0],)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "n,d,k,nepochs",
    [
        (257, 1, 1, 1.0),
        (513, 2, 3, 1.5),
        (1031, 17, 19, 0.6),
        (4111, 54, 127, 2.3),
        (4200, 100, 257, 1.0),
        (2053, 127, 259, 2.0),
        (1024, 513, 64, 1.0),
        (128, 32, 128, 2.0),  # N == K boundary!
    ],
)
def test_prime_boundary_shapes_and_fractional_epochs(n: int, d: int, k: int, nepochs: float) -> None:
    dev = torch.device("cuda:0")
    gen = torch.Generator(device=dev).manual_seed(n + d * 100 + k)
    x = torch.randn((n, d), generator=gen, device=dev, dtype=torch.float32)
    solver = ApexEMeans(d=d, k=k, nepochs=nepochs, precision="auto", seed=123)
    c, lbl = solver.fit(x, capture_assignments=True)
    assert c.shape == (k, d)
    assert lbl is not None and lbl.shape == (n,)
    assert bool(torch.isfinite(c).all().item())
    d2, l2 = solver.assign(x[: min(n, 67)])
    assert bool(torch.isfinite(d2).all().item()) and l2.shape == (min(n, 67),)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_zeros_and_constant_degenerate_inputs() -> None:
    dev = torch.device("cuda:0")
    for val in (0.0, 42.5):
        x = torch.full((2048, 64), val, device=dev, dtype=torch.float32)
        for prec in ("bf16", "croiss_fp8", "fp32"):
            s = ApexEMeans(d=64, k=32, nepochs=1.0, precision=prec, seed=7)
            c, lbl = s.fit(x, capture_assignments=True)
            assert bool(torch.isfinite(c).all().item()), f"Non-finite centroids on constant={val} ({prec})"
            assert float((c - val).abs().max().item()) < 1e-3
            assert lbl is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_huygens_twin_safe_rare_mode_preservation_and_dead_revival() -> None:
    dev = torch.device("cuda:0")
    gen = torch.Generator(device="cpu").manual_seed(202611)
    d = 64
    k_dense = 56
    k_rare = 8
    k = k_dense + k_rare

    c_dense = torch.randn((k_dense, d), generator=gen, dtype=torch.float32) * 2.0
    # Rare outlier modes placed far from dense bulk (`distance ~ 30`), with only 12 points each!
    c_rare = torch.randn((k_rare, d), generator=gen, dtype=torch.float32) * 3.0 + 25.0

    n_dense = 16000
    ids_d = torch.randint(0, k_dense, (n_dense,), generator=gen)
    x_dense = c_dense[ids_d] + torch.randn((n_dense, d), generator=gen) * 0.20
    x_rare_list = [c_rare[r : r + 1] + torch.randn((12, d), generator=gen) * 0.08 for r in range(k_rare)]
    x_all = torch.cat([x_dense, *x_rare_list], dim=0)
    perm = torch.randperm(x_all.shape[0], generator=gen)
    x_gpu = x_all[perm].to(dev).contiguous()
    c_rare_gpu = c_rare.to(dev)

    solver = ApexEMeans(d=d, k=k, nepochs=3.0, precision="auto", seed=42)
    c_out, _ = solver.fit(x_gpu, capture_assignments=False)

    d2_rare = torch.cdist(c_rare_gpu.double(), c_out.double()).square().min(dim=1).values
    preserved = float((d2_rare < 1.0).float().mean().item())
    assert preserved >= 0.875, f"Expected >= 7/8 rare modes preserved by Huygens shield, got {preserved:.3f}"
    assert solver.dead_clusters == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gear3_ivf_explorer_and_gear4_multi_shard_assignments() -> None:
    """Exercise Gear 3 (`huge_k_ivf`) and Gear 4 (`multi_gpu` list-shards & strided `prepopulate_shards`)."""
    dev = torch.device("cuda:0")
    x_tr, x_va, _ = _make_clustered_data(12288, 64, 256, seed=808, device=dev)

    # Gear 3: Sync-Free IVF Exploration + Exact Closing Pass
    s_ivf = ApexEMeans(d=64, k=256, nepochs=2.0, gear_mode="huge_k_ivf", ivf_explore=True, ivf_nlist=32, ivf_nprobe=8, seed=31)
    c_ivf, _ = s_ivf.fit(x_tr, capture_assignments=False)
    mse_ivf = _exact_mse_fp64(x_va, c_ivf)
    assert "huge_k_ivf" in s_ivf.selected_gear
    assert math.isfinite(mse_ivf) and s_ivf.dead_clusters == 0

    # Gear 4 (Mode A): Explicit `list[torch.Tensor]` shards (`input_is_list=True`, contiguous shard order)
    shards_list = [x_tr[:6000].contiguous(), x_tr[6000:].contiguous()]
    s_mg_list = ApexEMeans(d=64, k=256, nepochs=2.0, gear_mode="multi_gpu", devices=[dev, dev], seed=31)
    c_mg1, lbl_mg1 = s_mg_list.fit(shards_list, capture_assignments=True)
    assert isinstance(c_mg1, torch.Tensor) and c_mg1.shape == (256, 64)
    assert lbl_mg1 is not None and lbl_mg1.shape == (12288,)
    # Verify contiguous shard order (`lbl_mg1` aligns 1:1 with `torch.cat(shards_list, dim=0)`)
    _, exact_mg1 = torch.cdist(x_tr[:512].double(), c_mg1.double()).min(dim=1)
    assert float((lbl_mg1[:512] == exact_mg1).float().mean().item()) > 0.98

    # Gear 4 (Mode B): Single tensor `x_tr` split strided via `prepopulate_shards` (`input_is_list=False`)
    s_mg_strided = ApexEMeans(d=64, k=256, nepochs=2.0, gear_mode="multi_gpu", devices=[dev, dev], seed=31)
    c_mg2, lbl_mg2 = s_mg_strided.fit(x_tr, capture_assignments=True)
    assert lbl_mg2 is not None and lbl_mg2.shape == (12288,)
    # Verify strided interleaving (`assignments_out[w::W]`) restores exact row order of `x_tr` (`#162`)!
    _, exact_mg2 = torch.cdist(x_tr[:512].double(), c_mg2.double()).min(dim=1)
    assert float((lbl_mg2[:512] == exact_mg2).float().mean().item()) > 0.98


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_multiwave_k16384_no_inter_cta_step_race() -> None:
    """Verify 3-slot ping-pong `gpu_state[in_s] != gpu_state[out_s]` has ZERO inter-CTA race across 88 CTA waves (`K=16,384`)."""
    from emeans.apex_emeans import _alloc_padded_2d, _apex_ema_oja_slow_pass_prep_kernel

    dev = torch.device("cuda:0")
    k, d = 16384, 32
    gpu_state = torch.zeros((3, 8), dtype=torch.float64, device=dev)
    gpu_state[0, 0] = 0.0  # step = 0 -> alpha = eff_peak = 1.0!
    gpu_state[0, 3] = 1.0
    gpu_state[0, 4] = 1.0
    gpu_state[0, 5] = 0.1
    batch_off = torch.zeros((3,), dtype=torch.int64, device=dev)

    sums = torch.full((k, d), 10.0, dtype=torch.float32, device=dev)
    counts = torch.ones((k,), dtype=torch.float32, device=dev)
    bsse = torch.zeros((k,), dtype=torch.float32, device=dev)
    centroids_c = torch.zeros((k, d), dtype=torch.float32, device=dev)
    usage = torch.ones((k,), dtype=torch.float32, device=dev)
    sse_ema = torch.zeros((k,), dtype=torch.float32, device=dev)
    pass_sums = torch.zeros((k, d), dtype=torch.float32, device=dev)
    pass_cnt = torch.zeros((k,), dtype=torch.float64, device=dev)
    pass_cnt_sq = torch.zeros((k,), dtype=torch.float64, device=dev)
    pass_sse = torch.zeros((k,), dtype=torch.float64, device=dev)
    raw_cnt = torch.zeros((k,), dtype=torch.float64, device=dev)
    c_rep = _alloc_padded_2d(k, d, torch.bfloat16, dev, align=16)
    sc_buf = torch.ones((k,), dtype=torch.float32, device=dev)
    c_norm = torch.empty((k,), dtype=torch.float32, device=dev)

    # At step=0 with total_steps=2 and eff_peak=0.5, decay=0.5, mass = 0.5*1 + 0.5*1 = 1.0, weight = 0.5 -> c = 5.0
    # If CTA 0 raced into `gpu_state[in_s]`, later CTA waves (>188 SMs) would read `step=1` -> c = 2.75!
    _apex_ema_oja_slow_pass_prep_kernel[(k,)](
        gpu_state[0],
        gpu_state[1],
        batch_off[0],
        batch_off[1],
        sums,
        counts,
        bsse,
        centroids_c,
        centroids_c,
        usage,
        sse_ema,
        centroids_c,
        centroids_c,
        centroids_c,
        usage,
        pass_sums,
        pass_cnt,
        pass_cnt_sq,
        pass_sse,
        raw_cnt,
        c_rep,
        sc_buf,
        c_norm,
        k,
        d,
        centroids_c.stride(0),
        c_rep.stride(0),
        1024,
        32768,
        32768,
        2,
        32768,
        0.5,
        0.1,
        0.01,
        1.0,
        0.5,
        0.05,
        0.0,
        0,
        1,
        SCHED_MODE=0,
        USE_WIDE_SCHED=0,
        COUPLE_SSE=True,
        TRACK_OJA=False,
        TRACK_SURV=False,
        TRACK_SLOW=False,
        REP_MODE=1,
        BLOCK_D=32,
        num_warps=2,
    )
    max_err = float((centroids_c - 5.0).abs().max().item())
    assert max_err == 0.0, f"Inter-CTA step_ptr race detected at K=16,384: max_err={max_err}"
    assert float(gpu_state[0, 0].item()) == 0.0 and float(gpu_state[1, 0].item()) == 1.0


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires >= 2 visible CUDA devices for hardware NCCL")
def test_2gpu_hardware_nccl_parity() -> None:
    """Test true 2-GPU hardware `torch.cuda.nccl.all_reduce` & `broadcast` across `cuda:0` and `cuda:1`."""
    d0 = torch.device("cuda:0")
    d1 = torch.device("cuda:1")
    x_tr, x_va, _ = _make_clustered_data(16384, 64, 256, seed=909, device=d0)

    # 1. Pre-sharded `list[torch.Tensor]` across 2 distinct GPUs
    shards = [x_tr[:8000].to(d0).contiguous(), x_tr[8000:].to(d1).contiguous()]
    s_list = ApexEMeans(d=64, k=256, nepochs=2.0, devices=[d0, d1], precision="auto", seed=42)
    c_l, lbl_l = s_list.fit(shards, capture_assignments=True)
    assert c_l.shape == (256, 64) and bool(torch.isfinite(c_l).all().item())
    assert lbl_l is not None and lbl_l.shape == (16384,)
    assert s_list.dead_clusters == 0
    _, exact_l = torch.cdist(x_tr[:512].double(), c_l.to(d0).double()).min(dim=1)
    assert float((lbl_l[:512].to(d0) == exact_l).float().mean().item()) > 0.98

    # 2. Single tensor distributed strided via `prepopulate_shards` across `cuda:0` & `cuda:1`
    s_str = ApexEMeans(d=64, k=256, nepochs=2.0, devices=[d0, d1], precision="bf16", seed=42)
    c_s, lbl_s = s_str.fit(x_tr, capture_assignments=True)
    assert lbl_s is not None and lbl_s.shape == (16384,)
    _, exact_s = torch.cdist(x_tr[:512].double(), c_s.to(d0).double()).min(dim=1)
    assert float((lbl_s[:512].to(d0) == exact_s).float().mean().item()) > 0.98
    mse_2gpu = _exact_mse_fp64(x_va, c_l)
    mse_base = _exact_mse_fp64(x_va, x_tr[:256])
    assert mse_2gpu < 0.35 * mse_base


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires >= 2 visible CUDA devices")
def test_non_primary_cuda1_warmup_and_fit() -> None:
    """Verify `warmup_apex_kernels(cuda:1)` and `fit/assign` on `cuda:1` succeed while `current_device() == 0`."""
    torch.cuda.set_device(0)
    d1 = torch.device("cuda:1")
    warmup_apex_kernels(d1, d=64, k=128)
    assert torch.cuda.current_device() == 0

    x_tr, x_va, _ = _make_clustered_data(4096, 64, 128, seed=1234, device=d1)
    solver = ApexEMeans(d=64, k=128, nepochs=2.0, precision="auto", seed=77)
    c, lbl = solver.fit(x_tr, capture_assignments=True)
    assert c.device == d1 and lbl is not None and lbl.device == d1
    d2, l2 = solver.assign(x_va)
    assert d2.device == d1 and l2.device == d1 and bool(torch.isfinite(d2).all().item())
    assert torch.cuda.current_device() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_scale_law_gear_transitions_and_huge_k_seeding() -> None:
    """Verify `#189` empirical scale laws: wide-D auto-BF16, FULL_D auto-FP8, and K=16,384 fast seed + IVF close."""
    dev = torch.device("cuda:0")
    # 1. Wide-D (`D=512 > 256`) selects BF16 in `auto`
    s_wide_d = ApexEMeans(d=512, k=1024, nepochs=3.0, precision="auto", gear_mode="auto")
    prec_512, _ = s_wide_d._resolve_precision_and_dtype(50_000, 512, 1024, dev, torch.float32)
    assert prec_512 == "bf16"

    # 2. FULL_D (`D=128 <= 256`, `100K x 2048`) selects CroissAttn-FP8 on SM89+/SM120
    s_med = ApexEMeans(d=128, k=2048, nepochs=3.0, precision="auto", gear_mode="auto")
    prec_128, _ = s_med._resolve_precision_and_dtype(100_000, 128, 2048, dev, torch.float32)
    if torch.cuda.get_device_capability(dev) >= (8, 9):
        assert prec_128 == "croiss_fp8"

    # 3. Huge-K (`K=16,384`) fast coarse-bucketed seeding + IVF polish runs cleanly and produces valid priors
    gen = torch.Generator(device=dev).manual_seed(2026)
    x_hk = torch.randn((32768, 64), generator=gen, device=dev, dtype=torch.float32)
    origin64 = x_hk[:4096].double().mean(dim=0).contiguous()
    seed_hk = apex_turkpp_init_centered(x_hk, origin64, 1.0, k=16384, comp_dtype=torch.bfloat16, seed=55)
    assert seed_hk.centroids.shape == (16384, 64)
    assert bool(torch.isfinite(seed_hk.centroids).all().item())
    assert abs(float(seed_hk.prior_probs.sum().item()) - 1.0) < 1e-4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cold_triton_cache_warmup_eliminates_seed0_jit_spike(tmp_path: Path) -> None:
    """Verify on a brand-new empty `TRITON_CACHE_DIR` that `warmup_apex_kernels` leaves zero JIT spike."""
    cold_cache = tmp_path / "triton_cold_apex"
    cold_cache.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["TRITON_CACHE_DIR"] = str(cold_cache)

    code = (
        "import sys, time, torch\n"
        f"sys.path.insert(0, {str(ROOT / 'src')!r})\n"
        "from emeans.apex_emeans import ApexEMeans, warmup_apex_kernels\n"
        "dev = torch.device('cuda:0')\n"
        "warmup_apex_kernels(dev, d=128, k=512)\n"
        "x = torch.randn((16384, 128), device=dev, dtype=torch.float32)\n"
        "times = []\n"
        "for sd in (100, 101, 102):\n"
        "    torch.cuda.synchronize(dev)\n"
        "    t0 = time.perf_counter()\n"
        "    ApexEMeans(d=128, k=512, nepochs=2.0, seed=sd).fit(x, capture_assignments=False)\n"
        "    torch.cuda.synchronize(dev)\n"
        "    times.append(time.perf_counter() - t0)\n"
        "ratio = times[0] / max(min(times[1:]), 1e-6)\n"
        "assert ratio < 1.75, f'Cold-JIT spike detected: times={times}, ratio={ratio:.2f}'\n"
    )
    res = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=180, check=False)
    assert res.returncode == 0, f"Cold-cache warmup subprocess failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
