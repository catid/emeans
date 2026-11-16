"""Rigorous numerical correctness, invariant, and edge-case tests for `emeans.fast_ops`.

Covers all 5 optimization hypotheses (H-Opt-1 .. H-Opt-5) before any timed benchmarking:
  1. Exact FP32 argmin & distance agreement across prime / non-power-of-two (B, K, D)
     and both Split-K and non-Split-K paths, plus fused `row_indices` gather.
  2. Appendix I.1 mixed-precision represented-norm invariant in BF16 and TMA vs pointer parity.
  3. FP8 E4M3 centered power-of-two calibration, saturating outlier safety, and < 0.15%
     held-out FP64 test MSE parity across d = 96, 128, 512, 1024.
  4. Exact EMA algebra (counts == 0 vs counts > 0, alpha < 1 vs alpha == 1 / weight == 1)
     and deterministic Fused Triton + CUDA-Graph Greedy k-means++ equivalence.
  5. IVF coarse-to-fine monotonicity, empty-bucket safety, streaming modes, and
     Multi-GPU NCCL bit-exact replica consistency across odd/prime shapes.
"""

from __future__ import annotations

import pytest
import torch

from emeans.emeans import EMeans
from emeans.fast_ops import (
    FLOAT8_E4M3FN,
    FP8Quantizer,
    FastEMeans,
    IVFAssigner,
    MultiGPUFastEMeans,
    blackwell_prepare_centroids,
    blackwell_triton_assign,
    fast_seed_emeans_plusplus,
    fp8_assign,
    fused_ema_and_prepare,
    fused_scatter_stats,
    triton_greedy_kmeans_plusplus,
)
from emeans.kernels import exact_mse_fp64, prepare_represented_centroids, torch_assign_chunked
from emeans.seeding import greedy_kmeans_plusplus_torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for Blackwell Triton fast_ops tests"
)


def _make_gaussian_mixture(
    n_train: int,
    n_test: int,
    k_true: int,
    d: int,
    cluster_std: float = 0.35,
    center_scale: float = 3.0,
    offset: float = 0.0,
    seed: int = 42,
    device: str = "cuda:0",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate clean separated train/test Gaussian mixture without train/test leakage."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    centers = torch.randn((k_true, d), generator=gen, device=device, dtype=torch.float32) * center_scale + offset

    idx_tr = torch.randint(0, k_true, (n_train,), generator=gen, device=device)
    noise_tr = torch.randn((n_train, d), generator=gen, device=device, dtype=torch.float32) * cluster_std
    x_train = (centers[idx_tr] + noise_tr).contiguous()

    idx_te = torch.randint(0, k_true, (n_test,), generator=gen, device=device)
    noise_te = torch.randn((n_test, d), generator=gen, device=device, dtype=torch.float32) * cluster_std
    x_test = (centers[idx_te] + noise_te).contiguous()
    return x_train, x_test


# ==============================================================================
# H-Opt-1: Blackwell Split-K, FULL_D, TMA, and Prime Boundary Tests
# ==============================================================================

@pytest.mark.parametrize(
    "B,K,D,force_split_k",
    [
        (0, 64, 32, False),
        (1, 17, 7, False),
        (137, 503, 37, False),
        (137, 503, 37, True),
        (257, 4103, 96, None),    # Auto Split-K + FULL_D (D=96 <= 128)
        (193, 4507, 131, True),   # Explicit Split-K + tiled D > 128 on prime shapes
        (1024, 256, 128, False),  # Direct FULL_D
        (512, 512, 512, False),   # Direct wide-D
    ],
)
def test_h_opt1_fp32_exact_argmin_and_distances(
    B: int, K: int, D: int, force_split_k: bool | None
) -> None:
    dev = torch.device("cuda:0")
    gen = torch.Generator(device=dev).manual_seed(1001 + B + K + D)
    x = torch.randn((max(B, 1), D), generator=gen, device=dev, dtype=torch.float32)[:B]
    c = torch.randn((K, D), generator=gen, device=dev, dtype=torch.float32)
    c_rep, c_norm = blackwell_prepare_centroids(c, torch.float32)

    d_fast, l_fast = blackwell_triton_assign(
        x, c_rep, c_norm, force_split_k=force_split_k
    )
    d_ref, l_ref = torch_assign_chunked(x, c_rep, c_norm)

    assert d_fast.shape == (B,)
    assert l_fast.shape == (B,)
    if B == 0:
        return

    assert torch.isfinite(d_fast).all()
    assert torch.equal(l_fast, l_ref), (
        f"FP32 argmin mismatch: {(l_fast != l_ref).sum().item()} / {B} rows differ"
    )
    torch.testing.assert_close(d_fast, d_ref, rtol=1e-5, atol=1e-4)

    # Also verify fused `row_indices` gather path on permuted subset
    perm = torch.randperm(B, generator=gen, device=dev)[: max(1, B // 2)]
    d_idx, l_idx = blackwell_triton_assign(
        x, c_rep, c_norm, row_indices=perm, force_split_k=force_split_k
    )
    assert torch.equal(l_idx, l_ref[perm])
    torch.testing.assert_close(d_idx, d_ref[perm], rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize(
    "B,K,D",
    [
        (512, 4096, 96),     # Split-K + FULL_D BF16
        (512, 4096, 256),    # Split-K + tiled BF16
        (28000, 512, 128),   # Large-B FULL_D (128x128 tile)
        (28000, 512, 256),   # Large-B TMA descriptor path (128x256 tile)
        (333, 777, 100),     # Non-multiple D=100 BF16
    ],
)
def test_h_opt1_bf16_represented_norm_consistency_and_tma(B: int, K: int, D: int) -> None:
    dev = torch.device("cuda:0")
    gen = torch.Generator(device=dev).manual_seed(2026 + D)
    x = torch.randn((B, D), generator=gen, device=dev, dtype=torch.float32)
    c = torch.randn((K, D), generator=gen, device=dev, dtype=torch.float32)

    c_rep, c_norm = blackwell_prepare_centroids(c, torch.bfloat16)
    # Exact BF16 oracle using represented operands (Appendix I.1 Eq. 10-12)
    d_ref, l_ref = torch_assign_chunked(x, c_rep, c_norm)

    d_tma, l_tma = blackwell_triton_assign(x, c_rep, c_norm, allow_tma=True)
    d_ptr, l_ptr = blackwell_triton_assign(x, c_rep, c_norm, allow_tma=False)

    assert torch.isfinite(d_tma).all()
    assert torch.equal(l_tma, l_ptr)
    torch.testing.assert_close(d_tma, d_ptr, rtol=1e-5, atol=1e-5)

    # Disagreement vs torch BF16 addmm can only occur on exact 1-ULP FP32 dot rounding ties
    disagree = float((l_tma != l_ref).float().mean().item())
    assert disagree < 1e-3
    torch.testing.assert_close(d_tma, d_ref, rtol=2e-4, atol=2e-3)


# ==============================================================================
# H-Opt-2: FP8 E4M3 Centered Power-of-Two Scaling & MSE Parity Across Dimensions
# ==============================================================================

def test_h_opt2_fp8_centering_and_norm_matching() -> None:
    assert FLOAT8_E4M3FN is not None, "FP8 E4M3FN must be supported on Blackwell"
    dev = torch.device("cuda:0")
    # Shifted dataset where coordinates hover around +80.0 +/- 0.4:
    # Without centering, E4M3 wastes dynamic range on the +80 mean offset!
    x_tr, x_te = _make_gaussian_mixture(
        n_train=8192,
        n_test=2048,
        k_true=64,
        d=128,
        cluster_std=0.15,
        center_scale=0.5,
        offset=80.0,
        seed=77,
        device=str(dev),
    )
    c = x_tr[:128].clone()

    q_auto = FP8Quantizer.calibrate(x_tr, center_mode="auto", seed=77)
    q_never = FP8Quantizer.calibrate(x_tr, center_mode="never", seed=77)

    assert q_auto.shift is not None, "Auto calibration must enable centering on high-offset data"
    assert q_auto.exponent > q_never.exponent, "Centering must unlock larger power-of-two scale"

    # Verify exact represented norm invariant: c_norm == sum(float(c_fp8)^2)
    c_fp8, c_norm = q_auto.prepare_centroids(c)
    expected_norm = c_fp8[:, :128].float().square().sum(dim=1)
    torch.testing.assert_close(c_norm, expected_norm, rtol=1e-6, atol=1e-5)

    # Centered FP8 assignment distortion on x_te must beat uncentered FP8 significantly
    _, lbl_auto = fp8_assign(x_te, c, quantizer=q_auto)
    _, lbl_never = fp8_assign(x_te, c, quantizer=q_never)
    mse_auto = float((x_te.double() - c[lbl_auto].double()).square().sum(dim=1).mean().item())
    mse_never = float((x_te.double() - c[lbl_never].double()).square().sum(dim=1).mean().item())
    mse_exact = exact_mse_fp64(x_te, c, device=dev)

    assert mse_auto < mse_never
    assert abs(mse_auto - mse_exact) / mse_exact < 1.5e-3


@pytest.mark.parametrize("d", [96, 128, 512, 1024])
def test_h_opt2_fp8_training_mse_parity_across_dims(d: int) -> None:
    dev = "cuda:0"
    # Appendix D.2 calibrated continuous-manifold GMM (Var(mu)=2/3, Var(noise)=1/3)
    x_train, x_test = _make_gaussian_mixture(
        n_train=64000,
        n_test=10000,
        k_true=512,
        d=d,
        cluster_std=0.57735,
        center_scale=0.81650,
        offset=5.0,
        seed=300 + d,
        device=dev,
    )

    mse_bases, mse_fp8s = [], []
    for seed in range(10, 20):
        base = EMeans(d=d, k=64, nepochs=4.0, seed=seed)
        c_base, _ = base.fit(x_train, capture_assignments=False)
        mse_bases.append(exact_mse_fp64(x_test, c_base, device=dev))

        fast_fp8 = FastEMeans(d=d, k=64, nepochs=4.0, precision="fp8", seed=seed)
        c_fp8, _ = fast_fp8.fit(x_train, capture_assignments=False)
        mse_fp8s.append(exact_mse_fp64(x_test, c_fp8, device=dev))

    mse_base = sum(mse_bases) / len(mse_bases)
    mse_fp8 = sum(mse_fp8s) / len(mse_fp8s)
    rel_diff = abs(mse_fp8 - mse_base) / mse_base
    assert rel_diff < 0.0015, (
        f"d={d}: FP8 held-out MSE {mse_fp8:.6f} vs BF16 {mse_base:.6f} "
        f"exceeded 0.15% tolerance ({rel_diff * 100:.4f}%)"
    )


# ==============================================================================
# H-Opt-3: Fused EMA Update Algebra & Fused Triton + CUDA-Graph Greedy k-means++
# ==============================================================================

@pytest.mark.parametrize("alpha", [0.25, 1.0])
@pytest.mark.parametrize("comp_dtype", [torch.float32, torch.bfloat16, FLOAT8_E4M3FN])
def test_h_opt3a_fused_scatter_and_ema_algebra(alpha: float, comp_dtype: torch.dtype) -> None:
    dev = torch.device("cuda:0")
    K, D, B = 73, 100, 509
    gen = torch.Generator(device=dev).manual_seed(999)

    x = torch.randn((B + 50, D), generator=gen, device=dev, dtype=torch.float32)
    row_idx = torch.randperm(B + 50, generator=gen, device=dev)[:B]
    # Leave clusters 0..9 intentionally untouched (counts == 0) to verify untouched semantics!
    labels = torch.randint(10, K, (B,), generator=gen, device=dev, dtype=torch.int64)
    # Also force one cluster (cluster 10) to start with usage == 0 so weight == 1.0 even for alpha < 1!
    centroids_ref = torch.randn((K, D), generator=gen, device=dev, dtype=torch.float32)
    usage_ref = torch.rand((K,), generator=gen, device=dev, dtype=torch.float32) * 5.0 + 0.1
    usage_ref[10] = 0.0

    centroids_fast = centroids_ref.clone()
    usage_fast = usage_ref.clone()

    packed_stats = torch.zeros((K, D + 1), dtype=torch.float32, device=dev)
    fused_scatter_stats(x, labels, packed_stats, row_indices=row_idx)

    # Reference scatter
    xb = x[row_idx]
    ref_counts = torch.bincount(labels, minlength=K).float()
    ref_sums = torch.zeros((K, D), dtype=torch.float32, device=dev)
    ref_sums.index_add_(0, labels, xb)

    torch.testing.assert_close(packed_stats[:, D], ref_counts, rtol=0.0, atol=0.0)
    torch.testing.assert_close(packed_stats[:, :D], ref_sums, rtol=1e-5, atol=1e-4)

    # Reference EMA update matching `emeans.py` lines 272-287
    pass_counts_fast = torch.ones((K,), dtype=torch.float64, device=dev) * 3.0
    pass_sums_fast = torch.ones((K, D), dtype=torch.float32, device=dev) * 2.0
    pass_counts_ref = pass_counts_fast.clone()
    pass_sums_ref = pass_sums_fast.clone()

    pass_counts_ref += ref_counts.double()
    pass_sums_ref += ref_sums
    touched = ref_counts > 0
    if alpha < 1.0:
        usage_ref.mul_(1.0 - alpha)
        retained = usage_ref
    else:
        retained = torch.zeros_like(usage_ref)
    new_mass = retained + alpha * ref_counts
    means = ref_sums / torch.where(touched, ref_counts, torch.ones_like(ref_counts))[:, None]
    weight = (alpha * ref_counts) / torch.where(touched, new_mass, torch.ones_like(new_mass))
    upd = centroids_ref + weight[:, None] * (means - centroids_ref)
    upd = torch.where((weight == 1.0)[:, None], means, upd)
    centroids_ref = torch.where(touched[:, None], upd, centroids_ref)
    usage_ref = torch.where(touched, new_mass, usage_ref)

    quant = FP8Quantizer.calibrate(x, center_mode="always") if comp_dtype == FLOAT8_E4M3FN else None
    c_rep, c_norm = fused_ema_and_prepare(
        centroids=centroids_fast,
        usage=usage_fast,
        batch_sums=packed_stats[:, :D],
        counts=packed_stats[:, D],
        alpha=alpha,
        compute_dtype=comp_dtype,
        pass_counts=pass_counts_fast,
        pass_sums=pass_sums_fast,
        fp8_quantizer=quant,
        zero_stats=True,
    )

    # Workspace must be zeroed in-place
    assert float(packed_stats.abs().max().item()) == 0.0
    torch.testing.assert_close(centroids_fast, centroids_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(usage_fast, usage_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(pass_counts_fast, pass_counts_ref, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(pass_sums_fast, pass_sums_ref, rtol=1e-5, atol=1e-4)

    # Untouched clusters 0..9 must match exact untouched algebra
    assert torch.equal(centroids_fast[:10], centroids_ref[:10])

    # Prepared norms must match exact represented c_rep norms (Appendix I.1)
    if comp_dtype == FLOAT8_E4M3FN and quant is not None:
        ref_cr, ref_cn = quant.prepare_centroids(centroids_fast)
        assert torch.equal(c_rep.view(torch.uint8), ref_cr.view(torch.uint8))
        torch.testing.assert_close(c_norm, ref_cn, rtol=1e-5, atol=1e-4)
    else:
        ref_cr, ref_cn = prepare_represented_centroids(centroids_fast, comp_dtype)
        assert torch.equal(c_rep, ref_cr)
        torch.testing.assert_close(c_norm, ref_cn, rtol=1e-5, atol=1e-4)


def test_h_opt3b_triton_greedy_kmeans_plusplus_and_cuda_graph() -> None:
    dev = torch.device("cuda:0")
    x_tr, _ = _make_gaussian_mixture(
        n_train=2053, n_test=256, k_true=32, d=96, cluster_std=0.1, center_scale=5.0, seed=55
    )
    gen = torch.Generator(device=dev).manual_seed(55)
    weights = torch.rand((2053,), generator=gen, device=dev, dtype=torch.float32) + 0.1
    weights /= weights.sum()

    # 1. Exact equivalence between eager Triton and CUDA-Graph Triton
    c_eager = triton_greedy_kmeans_plusplus(
        x_tr, k=32, weights=weights, use_bf16=True, use_cuda_graph=False, seed=123
    )
    c_graph = triton_greedy_kmeans_plusplus(
        x_tr, k=32, weights=weights, use_bf16=True, use_cuda_graph=True, seed=123
    )
    assert torch.equal(c_eager, c_graph), "CUDA Graph replay must be bit-identical to eager Triton!"

    # 2. Exact IEEE FP32 match with reference PyTorch greedy_kmeans_plusplus_torch
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        c_fp32_fast = triton_greedy_kmeans_plusplus(
            x_tr, k=24, weights=weights, use_bf16=False, use_cuda_graph=True, seed=777
        )
        c_fp32_ref = greedy_kmeans_plusplus_torch(
            x_tr[:, :16].contiguous(), k=8, weights=weights, seed=777
        )
        c_fp32_fast16 = triton_greedy_kmeans_plusplus(
            x_tr[:, :16].contiguous(), k=8, weights=weights, use_bf16=False, use_cuda_graph=True, seed=777
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32
    torch.testing.assert_close(c_fp32_fast16, c_fp32_ref, rtol=1e-5, atol=1e-5)
    assert torch.isfinite(c_fp32_fast).all()

    # 3. Full fast_seed_emeans_plusplus validity & priors check
    res = fast_seed_emeans_plusplus(x_tr, k=32, polish=True, seed=99)
    assert res.centroids.shape == (32, 96)
    assert torch.isfinite(res.centroids).all()
    assert torch.isfinite(res.prior_probs).all()
    assert abs(float(res.prior_probs.sum().item()) - 1.0) < 1e-5
    assert (res.prior_distortion >= 0.0).all()

    # Must dramatically beat uniform random seeding on well-separated clusters
    mse_kpp = exact_mse_fp64(x_tr, res.centroids, device=dev)
    mse_rnd = exact_mse_fp64(x_tr, x_tr[:32], device=dev)
    assert mse_kpp < 0.25 * mse_rnd


# ==============================================================================
# H-Opt-4: Coarse-to-Fine IVF Assigner & Pathological Empty-Bucket Safety
# ==============================================================================

def test_h_opt4_ivf_assigner_exact_limit_monotonicity_and_empty_buckets() -> None:
    dev = torch.device("cuda:0")
    x_tr, x_te = _make_gaussian_mixture(
        n_train=16384, n_test=2048, k_true=128, d=128, cluster_std=0.35, center_scale=3.0, seed=404
    )
    centroids = x_tr[:1024].clone().contiguous()

    ivf = IVFAssigner.build(x_tr, k=1024, nlist=32, compute_dtype=torch.bfloat16, seed=404)
    ivf.rebuild_index(centroids)

    # 1. When nprobe == nlist, IVF must match exact BF16 assignment
    d_full, l_full = ivf.assign(x_te, nprobe=32)
    c_rep, c_norm = blackwell_prepare_centroids(centroids, torch.bfloat16)
    d_exact, l_exact = blackwell_triton_assign(x_te, c_rep, c_norm)
    assert torch.equal(l_full, l_exact)
    torch.testing.assert_close(d_full, d_exact, rtol=1e-5, atol=1e-4)

    # 2. Monotonic decrease in distortion as nprobe grows (1 -> 4 -> 8 -> 16 -> 32)
    mses = []
    for np_val in (1, 4, 8, 16, 32):
        _, lbl = ivf.assign(x_te, nprobe=np_val)
        mse = float((x_te.double() - centroids[lbl].double()).square().sum(dim=1).mean().item())
        mses.append(mse)
    for i in range(len(mses) - 1):
        assert mses[i + 1] <= mses[i] + 1e-7, f"Non-monotonic IVF MSEs: {mses}"

    # 3. Pathological empty-bucket safety: collapse all centroids around 1 point so 31/32 buckets are empty!
    collapsed_c = centroids[:1].repeat(1024, 1) + 1e-4 * torch.randn_like(centroids)
    ivf.rebuild_index(collapsed_c)
    assert ivf.n_empty_buckets >= 20
    d_patho, l_patho = ivf.assign(x_te, nprobe=1)
    assert torch.isfinite(d_patho).all(), "Empty IVF buckets must fall back cleanly without Inf/NaN!"
    assert ((l_patho >= 0) & (l_patho < 1024)).all()

    # 4. End-to-end FastEMeans(assign_mode='ivf') MSE parity (< 0.15%)
    base = EMeans(d=128, k=256, nepochs=4.0, seed=42)
    c_b, _ = base.fit(x_tr, capture_assignments=False)
    mse_b = exact_mse_fp64(x_te, c_b, device=dev)

    fast_ivf = FastEMeans(
        d=128, k=256, nepochs=4.0, assign_mode="ivf", ivf_nlist=16, ivf_nprobe=6, seed=42
    )
    c_ivf, _ = fast_ivf.fit(x_tr, capture_assignments=False)
    mse_ivf = exact_mse_fp64(x_te, c_ivf, device=dev)
    assert abs(mse_ivf - mse_b) / mse_b < 0.0015


# ==============================================================================
# H-Opt-5: Streaming Modes & Multi-GPU Data-Sharded NCCL Replica Parity
# ==============================================================================

@pytest.mark.parametrize("mode", ["gather_copy", "fused_gather", "block_permute"])
def test_h_opt5a_streaming_modes_mse_parity(mode: str) -> None:
    dev = "cuda:0"
    x_tr, x_te = _make_gaussian_mixture(
        n_train=24000,
        n_test=4000,
        k_true=256,
        d=128,
        cluster_std=0.57735,
        center_scale=0.81650,
        seed=505,
    )
    mse_bases, mse_fasts = [], []
    for seed in range(10, 20):
        base = EMeans(d=128, k=64, nepochs=4.0, seed=seed)
        c_base, _ = base.fit(x_tr, capture_assignments=True)
        mse_bases.append(exact_mse_fp64(x_te, c_base, device=dev))

        fast = FastEMeans(d=128, k=64, nepochs=4.0, streaming_mode=mode, seed=seed)
        c_fast, assign_fast = fast.fit(x_tr, capture_assignments=True)
        assert assign_fast is not None and ((assign_fast >= 0) & (assign_fast < 64)).all()
        mse_fasts.append(exact_mse_fp64(x_te, c_fast, device=dev))

    mse_base = sum(mse_bases) / len(mse_bases)
    mse_fast = sum(mse_fasts) / len(mse_fasts)
    assert abs(mse_fast - mse_base) / mse_base < 0.0015


@pytest.mark.parametrize("precision", ["bf16", "fp8"])
def test_h_opt5b_multigpu_nccl_replica_exactness_and_mse_parity(precision: str) -> None:
    # Use prime N=64007 and odd fractional nepochs=3.5 to exercise arbitrary remainder sharding
    x_tr, x_te = _make_gaussian_mixture(
        n_train=64007,
        n_test=10000,
        k_true=512,
        d=128,
        cluster_std=0.57735,
        center_scale=0.81650,
        seed=606,
    )
    # Include dead initial centroids way out in space so dead-cluster revival splits MUST fire!
    init_c = x_tr[:64].clone()
    init_c[:8] = 1000.0

    ndev = torch.cuda.device_count()
    devices = [f"cuda:{i}" for i in range(min(2, ndev))]

    mse_bases, mse_mgpus = [], []
    for seed in range(10, 20):
        base = EMeans(d=128, k=64, nepochs=3.5, seed=seed)
        c_base, _ = base.fit(x_tr, init_centroids=init_c, capture_assignments=False)
        mse_bases.append(exact_mse_fp64(x_te, c_base, device="cuda:0"))

        mgpu = MultiGPUFastEMeans(
            d=128,
            k=64,
            devices=devices,
            nepochs=3.5,
            precision=precision,
            seed=seed,
        )
        c_mgpu = mgpu.fit(x_tr, init_centroids=init_c)
        assert mgpu.n_splits > 0, "Dead-cluster revival should have fired on outlier init centroids"

        if len(devices) >= 2:
            c0 = mgpu.centroids_per_gpu[0].cpu()
            c1 = mgpu.centroids_per_gpu[1].cpu()
            assert torch.equal(c0, c1), "Multi-GPU centroids must stay bit-for-bit identical across GPUs!"

        mse_mgpus.append(exact_mse_fp64(x_te, c_mgpu, device="cuda:0"))

    mse_base = sum(mse_bases) / len(mse_bases)
    mse_mgpu = sum(mse_mgpus) / len(mse_mgpus)
    assert abs(mse_mgpu - mse_base) / mse_base < 0.0015


# ==============================================================================
# Extreme Inputs Safety: Zero NaN / Zero Inf Invariants
# ==============================================================================

def test_extreme_inputs_zero_nan_inf() -> None:
    dev = torch.device("cuda:0")
    D, K = 128, 32

    # 1. All-zero dataset (triggers collapsed split separation on 0.0 donors!)
    x_zero = torch.zeros((1024, D), device=dev, dtype=torch.float32)
    for prec in ("fp32", "bf16", "fp8"):
        m = FastEMeans(d=D, k=K, nepochs=2.0, precision=prec, seed=11)
        c, lbl = m.fit(x_zero)
        assert torch.isfinite(c).all(), f"NaN/Inf on all-zero input for precision={prec}"
        assert lbl is not None and ((lbl >= 0) & (lbl < K)).all()

    # 2. Large-magnitude coordinates (SimSearchNet++-style ||x||^2 ~ 1e8 >> FP16 max 65504!)
    gen = torch.Generator(device=dev).manual_seed(22)
    x_huge = torch.randn((2048, D), generator=gen, device=dev, dtype=torch.float32) * 1e3 + 2e3
    for prec in ("fp32", "bf16", "fp8"):
        m = FastEMeans(d=D, k=K, nepochs=2.0, precision=prec, seed=22)
        c, lbl = m.fit(x_huge)
        assert torch.isfinite(c).all(), f"NaN/Inf on large-norm input for precision={prec}"
        d_assign, l_assign = m.assign(x_huge[:128])
        assert torch.isfinite(d_assign).all()
        assert ((l_assign >= 0) & (l_assign < K)).all()

    # 3. Extreme test-time outlier spike (1e9) fed to calibrated FP8 assigner
    q = FP8Quantizer.calibrate(x_huge[:512], center_mode="auto")
    x_spike = x_huge[:64].clone()
    x_spike[0, :] = 1e9
    x_spike[1, :] = -1e9
    d_sp, l_sp = fp8_assign(x_spike, x_huge[:K], quantizer=q)
    assert torch.isfinite(d_sp).all(), "Saturating FP8 cast must never emit NaN/Inf on spikes"
    assert ((l_sp >= 0) & (l_sp < K)).all()


# ==============================================================================
# Audit Regression Tests (#32 & #35: Mixed Dtypes, Seeding Parity, Validation)
# ==============================================================================

@pytest.mark.parametrize("in_dtype", [torch.bfloat16, torch.float16, torch.float64])
@pytest.mark.parametrize("d", [32, 128])
def test_mixed_input_dtypes_fp32_precision_and_fp8_guard(
    in_dtype: torch.dtype, d: int
) -> None:
    dev = torch.device("cuda:0")
    gen = torch.Generator(device=dev).manual_seed(911 + d)
    x = torch.randn((300, d), generator=gen, device=dev, dtype=torch.float32).to(in_dtype)
    c = torch.randn((48, d), generator=gen, device=dev, dtype=torch.float32)

    # Direct + Split-K + row_indices in FP32 compute mode on bf16/fp16/fp64 inputs
    c_rep, c_norm = blackwell_prepare_centroids(c, torch.float32)
    for split_k in (False, True):
        d_out, l_out = blackwell_triton_assign(x, c_rep, c_norm, force_split_k=split_k)
        assert torch.isfinite(d_out).all()
        assert ((l_out >= 0) & (l_out < 48)).all()

    # FastEMeans(precision="fp32") and IVFAssigner(compute_dtype=float32) on bf16/fp16/fp64 inputs
    m_fp32 = FastEMeans(d=d, k=16, nepochs=2.0, precision="fp32", seed=1)
    c_fit, _ = m_fp32.fit(x, capture_assignments=False)
    assert torch.isfinite(c_fit).all()
    assert m_fp32.total_time_sec == pytest.approx(m_fp32.init_time_sec + m_fp32.train_time_sec)

    ivf = IVFAssigner.build(x, k=32, nlist=8, compute_dtype=torch.float32, seed=1)
    ivf.rebuild_index(c[:32])
    d_ivf, l_ivf = ivf.assign(x, nprobe=4)
    assert torch.isfinite(d_ivf).all()
    assert ((l_ivf >= 0) & (l_ivf < 32)).all()

    # FP8 raw-input guard must trigger both without AND with `row_indices`
    q = FP8Quantizer.calibrate(c, center_mode="never")
    c_fp8, cn_fp8 = q.prepare_centroids(c)
    with pytest.raises(TypeError, match="FP8 assignment requires E4M3 encoded x"):
        blackwell_triton_assign(x.float(), c_fp8, cn_fp8)
    with pytest.raises(TypeError, match="FP8 assignment requires E4M3 encoded x"):
        blackwell_triton_assign(
            x.float(), c_fp8, cn_fp8, row_indices=torch.arange(16, device=dev)
        )


def test_fast_seed_emeans_plusplus_rare_modes_and_shifted_unpolished() -> None:
    from emeans.datasets import make_synthetic_dataset
    from emeans.seeding import seed_emeans_plusplus

    dev = torch.device("cuda:0")

    # 1. Rare-mode discovery parity (`include_farthest_candidate=True`)
    ds = make_synthetic_dataset("multiscale_rare_outliers", n_train=40000, d=64, k=100, seed=77)
    x_rare = torch.from_numpy(ds.train).to(dev)
    res_ref = seed_emeans_plusplus(x_rare, k=100, seed=77)
    res_fast = fast_seed_emeans_plusplus(x_rare, k=100, seed=77)

    assert ds.true_centers is not None and ds.rare_mode_indices is not None
    rare_centers = torch.from_numpy(ds.true_centers[ds.rare_mode_indices]).to(dev)
    d2_ref = torch.cdist(rare_centers, res_ref.centroids).min(dim=1).values
    d2_fast = torch.cdist(rare_centers, res_fast.centroids).min(dim=1).values
    assert float((d2_fast < 5.0).float().mean().item()) >= float((d2_ref < 5.0).float().mean().item())
    assert float((d2_fast < 5.0).float().mean().item()) == 1.0

    # 2. Shifted dataset + `polish=False` parity (mean-centered priors & normalized distortion)
    ds_shift = make_synthetic_dataset("balanced_gmm", n_train=20000, d=64, k=64, seed=19)
    x_shift = torch.from_numpy(ds_shift.train).to(dev) + 5000.0
    p_ref = seed_emeans_plusplus(x_shift, k=64, polish=False, seed=19)
    p_fast = fast_seed_emeans_plusplus(x_shift, k=64, polish=False, seed=19)

    assert abs(float(p_fast.prior_probs.sum().item()) - 1.0) < 1e-5
    assert float(p_fast.prior_distortion.sum().item()) > 0.0
    assert float((p_fast.prior_probs - p_ref.prior_probs).abs().sum().item()) < 0.05
    rel_dist_err = abs(
        float(p_fast.prior_distortion.sum().item()) - float(p_ref.prior_distortion.sum().item())
    ) / float(p_ref.prior_distortion.sum().item())
    assert rel_dist_err < 0.05


def test_constructor_validation_and_total_time_sec() -> None:
    dev = torch.device("cuda:0")
    x = torch.randn((512, 32), device=dev, dtype=torch.float32)

    with pytest.raises(ValueError, match="finalization"):
        FastEMeans(d=32, k=16, finalization="slow_ema")
    with pytest.raises(ValueError, match="finalization"):
        MultiGPUFastEMeans(d=32, k=16, finalization="slow_ema")
    with pytest.raises(ValueError, match="precision"):
        MultiGPUFastEMeans(d=32, k=16, precision="int4")
    with pytest.raises(ValueError, match="streaming_mode"):
        MultiGPUFastEMeans(d=32, k=16, streaming_mode="invalid")
    with pytest.raises(ValueError, match="IVFAssigner compute_dtype"):
        IVFAssigner(x[:8], compute_dtype=FLOAT8_E4M3FN)

    mg = MultiGPUFastEMeans(d=32, k=16, nepochs=2.0, seed=5)
    mg.fit(x)
    assert mg.total_time_sec > 0.0
    assert mg.total_time_sec == pytest.approx(mg.init_time_sec + mg.train_time_sec)


@pytest.mark.parametrize("d", [144, 256, 300, 512, 1024])
def test_wide_d_3stage_tma_and_fp8_parity(d: int) -> None:
    """Verify 128x256x32 (BF16/FP16) & 128x256x64 (FP8) 3-stage SM120 wide-D paths (#56)."""
    from emeans.kernels import triton_assign

    dev = torch.device("cuda:0")
    gen = torch.Generator(device=dev).manual_seed(2056 + d)
    B, K = 25000, 300  # B >= 128 * 188 triggers 128x256 wide-tile TMA & non-TMA paths
    x_f32 = torch.randn((B, d), generator=gen, device=dev, dtype=torch.float32)
    c_f32 = torch.randn((K, d), generator=gen, device=dev, dtype=torch.float32)
    idx = torch.randperm(B, generator=gen, device=dev)[:4096]

    for dt in (torch.bfloat16, torch.float16):
        x_dt = x_f32.to(dt).contiguous()
        c_rep, c_norm = blackwell_prepare_centroids(c_f32, dt)
        _, lbl_ref = triton_assign(x_dt, c_rep, c_norm)
        d_tma, lbl_tma = blackwell_triton_assign(x_dt, c_rep, c_norm, allow_tma=True, force_split_k=False)
        d_ntma, lbl_ntma = blackwell_triton_assign(x_dt, c_rep, c_norm, allow_tma=False, force_split_k=False)
        d_idx, lbl_idx = blackwell_triton_assign(x_dt, c_rep, c_norm, row_indices=idx, force_split_k=False)

        assert torch.isfinite(d_tma).all() and torch.isfinite(d_idx).all()
        assert torch.equal(lbl_tma, lbl_ntma)
        assert torch.equal(lbl_idx, lbl_tma[idx])
        assert float((lbl_tma != lbl_ref).float().mean().item()) < 1e-3

    q = FP8Quantizer.calibrate(x_f32, center_mode="auto", seed=7)
    x_fp8 = q.encode_rows(x_f32)
    c_fp8, cn_fp8 = q.prepare_centroids(c_f32)
    d8_tma, l8_tma = blackwell_triton_assign(
        x_fp8, c_fp8, cn_fp8, inv_scale_sq=q.inv_scale_sq, allow_tma=True, force_split_k=False
    )
    d8_idx, l8_idx = blackwell_triton_assign(
        x_fp8, c_fp8, cn_fp8, row_indices=idx, inv_scale_sq=q.inv_scale_sq, force_split_k=False
    )
    assert torch.isfinite(d8_tma).all() and torch.isfinite(d8_idx).all()
    assert torch.equal(l8_idx, l8_tma[idx])

