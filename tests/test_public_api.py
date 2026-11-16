"""Public package API, quickstart workflows, backwards-compatibility, and `/home` storage hygiene tests."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

import emeans
from emeans import (
    APEX_ABLATION_ARMS,
    APEX_LADDER_ARMS,
    BENCHMARK_SPEC_BY_NAME,
    CROISS_FP8_AVAILABLE,
    CROISS_MXFP8_AVAILABLE,
    SEVEN_REALISTIC_BENCHMARKS,
    ApexEMeans,
    ApexPQ,
    ApexRVQ,
    CoresetSelection,
    CroissFP8Dataset,
    CroissFP8Workspace,
    EMeans,
    EMeansPP,
    HybridSEMeans,
    RealVectorCorpus,
    RealisticBenchmarkSpec,
    SemDeDupScoreState,
    SMeans,
    SuperEMeans,
    UltraEMeans,
    build_apex_arm,
    build_hybrid_arm,
    build_ultra_arm,
    compute_semdedup_scores_gpu,
    croiss_encode_dataset,
    croiss_fp8_assign,
    croiss_fp8_assign_inplace,
    extract_semdedup_coreset,
    load_agnews_tfidf_svd_256,
    load_cifar10_convnext_768,
    load_cifar10_patches_128,
    load_cifar10_resnet18_512,
    load_covtype_multiscale_54,
    load_glove1m,
    load_glove1m_unit,
    load_glove1m_unit_benchmark,
    load_real_million_corpus,
    load_realistic_benchmark,
    load_sift1m,
    load_sift1m_benchmark,
    select_coreset_at_budget,
    select_prototype_coreset,
    select_random_coreset,
    warmup_apex_kernels,
)

# Complete Phase 1–2 public surface that must remain 100% backwards-compatible
LEGACY_PHASE1_PHASE2_EXPORTS: tuple[str, ...] = (
    "ALPHA_LAW",
    "ALPHA_MIN_FACTOR",
    "ANVIL_MAPS",
    "AUTO_QUALITY_MAX_D",
    "BATCH_LAWS",
    "CHECK_UNUSED_EVERY",
    "DEFAULT_INIT_USAGE",
    "DEFAULT_UNUSED_THRESHOLD",
    "DatasetSplit",
    "EMeans",
    "EMeansAlgoExt",
    "EMeansNoiseTrack",
    "EMeansPP",
    "EMeansPPEpochStat",
    "EMeansRecipe",
    "EpochStat",
    "FLOAT8_E4M3FN",
    "FP8Quantizer",
    "FalsifiableHypothesis",
    "FastEMeans",
    "HAS_FAISS",
    "HAS_TRITON",
    "HYPOTHESES",
    "HeadEditDiagnostics",
    "IVFAssigner",
    "LloydKMeans",
    "MultiGPUFastEMeans",
    "NOISE_HYPOTHESES",
    "NO_REVIVAL_MAX_D",
    "NaiveKMeansPP_EMeans",
    "NoiseTrackHypothesis",
    "QuadMergeDiagnostics",
    "SPLIT_EPSILON",
    "SculleyMiniBatchKMeans",
    "SeedingResult",
    "ShrinkageDiagnostics",
    "anvil_polar_displacement",
    "assign_points",
    "blackwell_prepare_centroids",
    "blackwell_triton_assign",
    "block_greedy_kmeans_plusplus_torch",
    "block_permute_indices",
    "build_recipe",
    "build_spatial_surprisal_order",
    "compute_epoch_rows",
    "compute_step_law",
    "deduplicate_splits",
    "empirical_bayes_finalize",
    "epoch_seed",
    "exact_mse_fp64",
    "execute_quad_merge",
    "fast_seed_emeans_plusplus",
    "fp8_assign",
    "fused_ema_and_prepare",
    "fused_scatter_stats",
    "gram_newton_schulz_polar",
    "greedy_kmeans_plusplus_torch",
    "hartigan_assign_points",
    "lightweight_coreset",
    "load_cifar10_resnet18_embeddings",
    "make_synthetic_dataset",
    "power_mean_top2_weights",
    "prepare_represented_centroids",
    "resolve_batch_law",
    "resolve_unused_threshold",
    "round_power_of_two",
    "run_head_edit_step",
    "scheduled_alpha",
    "seed_emeans_plusplus",
    "top2_assign_chunked",
    "torch_assign_chunked",
    "triton_assign",
    "triton_greedy_kmeans_plusplus",
    "voronoi_polish_and_priors",
)


def _home_scratch_root() -> Path:
    base = (
        Path("/home/catid/emeans_env/tmp")
        if Path("/home/catid/emeans_env").exists()
        else Path.home() / ".cache" / "emeans_tmp"
    )
    base.mkdir(parents=True, exist_ok=True)
    return base


def _suspicious_tmp_entries(pid: int | None = None) -> set[str]:
    found: set[str] = set()
    pid_tok = f"{pid}" if pid is not None else None
    for root_str in ("/tmp", "/var/tmp"):
        root = Path(root_str)
        if not root.exists():
            continue
        for entry in os.listdir(root):
            low = entry.lower()
            if "emeans" in low or "triton" in low or (pid_tok is not None and pid_tok in entry):
                found.add(f"{root_str}/{entry}")
    return found


def test_version_and_public_all_contract() -> None:
    assert emeans.__version__ == "0.5.0"
    assert isinstance(emeans.__all__, list)
    assert len(emeans.__all__) == len(set(emeans.__all__)), "__all__ must contain no duplicates"
    assert emeans.__all__ == sorted(emeans.__all__), "__all__ must be lexicographically sorted"

    for sym in emeans.__all__:
        assert hasattr(emeans, sym), f"Exported symbol {sym!r} missing on emeans"

    star_ns: dict[str, object] = {}
    exec("from emeans import *", star_ns)  # noqa: S102
    for sym in emeans.__all__:
        assert sym in star_ns, f"Wildcard import missed {sym!r}"


def test_backwards_compatibility_and_loader_exports() -> None:
    all_set = set(emeans.__all__)
    for legacy_sym in LEGACY_PHASE1_PHASE2_EXPORTS:
        assert legacy_sym in all_set, f"Legacy export {legacy_sym!r} missing from emeans.__all__"
        assert hasattr(emeans, legacy_sym)

    # Core solvers & builders
    for obj in (
        ApexEMeans,
        UltraEMeans,
        SuperEMeans,
        HybridSEMeans,
        SMeans,
        EMeansPP,
        EMeans,
        build_apex_arm,
        build_ultra_arm,
        build_hybrid_arm,
        warmup_apex_kernels,
    ):
        assert callable(obj)
    assert "apex_emeans" in APEX_ABLATION_ARMS
    assert len(APEX_LADDER_ARMS) == 6

    # CroissAttn-FP8 symbols
    assert isinstance(CROISS_FP8_AVAILABLE, bool)
    assert isinstance(CROISS_MXFP8_AVAILABLE, bool)
    for fp8_obj in (
        CroissFP8Dataset,
        CroissFP8Workspace,
        croiss_encode_dataset,
        croiss_fp8_assign,
        croiss_fp8_assign_inplace,
    ):
        assert fp8_obj is not None

    # Benchmark loaders & metadata
    assert load_glove1m is load_glove1m_unit
    assert len(SEVEN_REALISTIC_BENCHMARKS) == 7
    assert set(BENCHMARK_SPEC_BY_NAME.keys()) == {spec.name for spec in SEVEN_REALISTIC_BENCHMARKS}
    for loader in (
        load_sift1m,
        load_glove1m,
        load_glove1m_unit,
        load_real_million_corpus,
        load_realistic_benchmark,
        load_sift1m_benchmark,
        load_glove1m_unit_benchmark,
        load_cifar10_patches_128,
        load_cifar10_resnet18_512,
        load_cifar10_convnext_768,
        load_agnews_tfidf_svd_256,
        load_covtype_multiscale_54,
    ):
        assert callable(loader)
    assert RealVectorCorpus is not None
    assert RealisticBenchmarkSpec is not None


def test_five_line_quickstart_cpu_and_shift_invariance() -> None:
    gen = torch.Generator().manual_seed(2026)
    centers_true = torch.randn((17, 29), generator=gen, dtype=torch.float32) * 6.0
    x = centers_true[torch.randint(0, 17, (503,), generator=gen)] + 0.25 * torch.randn(
        (503, 29), generator=gen, dtype=torch.float32
    )
    q = centers_true[torch.randint(0, 17, (73,), generator=gen)] + 0.25 * torch.randn(
        (73, 29), generator=gen, dtype=torch.float32
    )

    # 5-line quickstart workflow
    model = ApexEMeans(d=29, k=17, nepochs=2.0, seed=42)
    centroids, train_labels = model.fit(x)
    sq_dists, q_labels = model.assign(q)

    assert isinstance(centroids, torch.Tensor) and centroids.shape == (17, 29)
    assert isinstance(train_labels, torch.Tensor) and train_labels.shape == (503,)
    assert isinstance(sq_dists, torch.Tensor) and sq_dists.shape == (73,)
    assert isinstance(q_labels, torch.Tensor) and q_labels.shape == (73,)
    assert bool(torch.isfinite(centroids).all())
    assert bool(torch.isfinite(sq_dists).all()) and bool((sq_dists >= 0.0).all())
    assert int(q_labels.min().item()) >= 0 and int(q_labels.max().item()) < 17

    # Exact FP64 nearest-centroid agreement on held-out queries
    ref_d2 = ((q.double()[:, None, :] - centroids.double()[None, :, :]) ** 2).sum(dim=2)
    ref_min_d2, ref_lbl = ref_d2.min(dim=1)
    torch.testing.assert_close(sq_dists.double(), ref_min_d2, rtol=1e-5, atol=1e-5)
    assert torch.equal(q_labels, ref_lbl)

    # +20,000 coordinate-shift invariance & NumPy round-trip support
    shift = 20_000.0
    x_np_shifted = (x.double() + shift).numpy().astype(np.float64)
    q_np_shifted = (q.double() + shift).numpy().astype(np.float64)
    model_sh = ApexEMeans(d=29, k=17, nepochs=2.0, seed=42)
    c_sh, _ = model_sh.fit(x_np_shifted)
    d2_sh, lbl_sh = model_sh.assign(q_np_shifted)
    assert isinstance(c_sh, np.ndarray) and c_sh.shape == (17, 29)
    assert isinstance(d2_sh, np.ndarray) and isinstance(lbl_sh, np.ndarray)
    np.testing.assert_allclose(d2_sh.mean(), sq_dists.numpy().mean(), rtol=1e-2, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GPU quickstart test")
def test_five_line_quickstart_and_croiss_fp8_cuda() -> None:
    dev = torch.device("cuda", 0)
    gen = torch.Generator(device=dev).manual_seed(1337)
    centers_true = torch.randn((32, 64), generator=gen, device=dev, dtype=torch.float32) * 4.0
    x = centers_true[torch.randint(0, 32, (2048,), generator=gen, device=dev)] + 0.3 * torch.randn(
        (2048, 64), generator=gen, device=dev, dtype=torch.float32
    )
    q = centers_true[torch.randint(0, 32, (256,), generator=gen, device=dev)] + 0.3 * torch.randn(
        (256, 64), generator=gen, device=dev, dtype=torch.float32
    )

    # Exact 5-line GPU quickstart
    model = ApexEMeans(d=64, k=32, nepochs=2.0, use_cuda_graph=False, seed=99)
    centroids, train_labels = model.fit(x)
    sq_dists, q_labels = model.assign(q)

    assert centroids.is_cuda and centroids.shape == (32, 64)
    assert train_labels is not None and train_labels.shape == (2048,)
    assert sq_dists.shape == (256,) and q_labels.shape == (256,)
    assert bool(torch.isfinite(centroids).all()) and bool(torch.isfinite(sq_dists).all())

    # Out-of-sample quantization MSE strictly beats random K-point baseline
    rand_c = x[:32].clone()
    rand_mse = float(((q.double()[:, None, :] - rand_c.double()[None, :, :]) ** 2).sum(dim=2).min(dim=1).values.mean())
    learned_mse = float(sq_dists.double().mean().item())
    assert learned_mse < 0.65 * rand_mse

    if CROISS_FP8_AVAILABLE:
        ds = croiss_encode_dataset(q)
        fp8_d2, fp8_lbl = croiss_fp8_assign(ds, centroids)
        assert fp8_d2.shape == (256,) and fp8_lbl.shape == (256,)
        assert bool(torch.isfinite(fp8_d2).all()) and bool((fp8_d2 >= 0.0).all())
        assert int(fp8_lbl.min().item()) >= 0 and int(fp8_lbl.max().item()) < 32


def test_top_level_pq_rvq_and_semdedup_workflows() -> None:
    dev = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    gen = torch.Generator(device="cpu").manual_seed(77)
    base_centers = torch.randn((16, 32), generator=gen, dtype=torch.float32) * 3.0
    x_train = (
        base_centers[torch.randint(0, 16, (512,), generator=gen)]
        + 0.2 * torch.randn((512, 32), generator=gen, dtype=torch.float32)
    ).to(dev)
    x_query = (
        base_centers[torch.randint(0, 16, (64,), generator=gen)]
        + 0.2 * torch.randn((64, 32), generator=gen, dtype=torch.float32)
    ).to(dev)

    # 1. ApexPQ top-level workflow
    pq = ApexPQ(d=32, m=4, ks=16, nepochs=2.0, use_cuda_graph=False, seed=7)
    codebooks, codes_tr = pq.fit(x_train, capture_codes=True)
    assert codebooks.shape == (4, 16, 8)
    assert codes_tr is not None and codes_tr.shape == (512, 4) and codes_tr.dtype == torch.uint8
    x_rec_pq = pq.decode(pq.encode(x_query))
    assert x_rec_pq.shape == (64, 32) and bool(torch.isfinite(x_rec_pq).all())
    adc_d, adc_i = pq.adc_search(x_query, codes_tr, k_top=5)
    assert adc_d.shape == (64, 5) and adc_i.shape == (64, 5)
    pq_metrics = pq.evaluate_reconstruction(x_query)
    assert pq_metrics.sqnr_db > 3.0

    # 2. ApexRVQ top-level workflow
    rvq = ApexRVQ(
        d=32,
        k1=16,
        k2=16,
        nepochs=2.0,
        adc_beam=8,
        seed=7,
        solver_kwargs={"use_cuda_graph": False},
    )
    (c1, c2), codes_rvq = rvq.fit(x_train, capture_codes=True)
    assert c1.shape == (16, 32) and c2.shape == (16, 32)
    assert codes_rvq is not None and codes_rvq.shape == (512, 2)
    x_rec_rvq = rvq.decode(rvq.encode(x_query))
    assert x_rec_rvq.shape == (64, 32) and bool(torch.isfinite(x_rec_rvq).all())
    rvq_d, rvq_i = rvq.adc_search(x_query, codes_rvq, k_top=5)
    assert rvq_d.shape == (64, 5) and rvq_i.shape == (64, 5)
    rvq_metrics = rvq.evaluate_reconstruction(x_query)
    assert rvq_metrics.sqnr_db > 5.0

    # 3. extract_semdedup_coreset & low-level SemDeDup exports (+20,000 shift invariance)
    sel = extract_semdedup_coreset(
        x_train,
        k=16,
        retention_frac=0.25,
        nepochs=2.0,
        seed=11,
        use_cuda_graph=False,
    )
    assert isinstance(sel, CoresetSelection)
    assert sel.num_selected == 128
    assert sel.indices.shape == (128,) and sel.weights.shape == (128,)
    assert int(torch.unique(sel.indices).numel()) == 128
    assert float(sel.weights.sum().item()) == pytest.approx(512.0, rel=1e-5)

    # Pre-fitted centroids + exact +20,000 coordinate-shift invariance (Sterbenz-exact FP32 & FP64)
    x_sterbenz = torch.round(x_train * 256.0) / 256.0
    c_fixed = x_sterbenz[:16].clone()
    sel_b = extract_semdedup_coreset(x_sterbenz, centroids=c_fixed, budget_m=80)
    sel_b_shift32 = extract_semdedup_coreset(
        x_sterbenz + 20_000.0,
        centroids=c_fixed + 20_000.0,
        budget_m=80,
    )
    sel_b_shift64 = extract_semdedup_coreset(
        x_sterbenz.double() + 20_000.0,
        centroids=c_fixed.double() + 20_000.0,
        budget_m=80,
    )
    assert sel_b.num_selected == 80
    assert torch.equal(sel_b.indices, sel_b_shift32.indices)
    assert torch.equal(sel_b.indices, sel_b_shift64.indices)
    torch.testing.assert_close(sel_b.weights, sel_b_shift32.weights, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(sel_b.weights, sel_b_shift64.weights, rtol=1e-6, atol=1e-6)

    proto_sel = extract_semdedup_coreset(x_sterbenz, centroids=c_fixed, prototype=True)
    assert 1 <= proto_sel.num_selected <= 16
    assert float(proto_sel.weights.sum().item()) == pytest.approx(512.0, rel=1e-5)

    # Low-level two-stage API parity
    st_low = compute_semdedup_scores_gpu(x_sterbenz, c_fixed, k=16)
    assert isinstance(st_low, SemDeDupScoreState)
    assert torch.equal(select_coreset_at_budget(st_low, budget_m=80).indices, sel_b.indices)
    assert torch.equal(select_prototype_coreset(st_low).indices, proto_sel.indices)
    rnd_sel = select_random_coreset(512, budget_m=64, seed=3, device=dev)
    assert rnd_sel.num_selected == 64

    # Argument validation guards
    with pytest.raises(ValueError):
        extract_semdedup_coreset(x_train, k=None, centroids=None)
    with pytest.raises(ValueError):
        extract_semdedup_coreset(x_train, centroids=c_fixed, prototype=True, retention_frac=0.2)
    with pytest.raises(ValueError):
        extract_semdedup_coreset(x_train, centroids=c_fixed, retention_frac=0.2, budget_m=50)


def test_extract_semdedup_numpy_strides_readonly_and_indexing() -> None:
    import warnings

    rng = np.random.default_rng(2026)
    x_np = rng.standard_normal((256, 16), dtype=np.float32)
    x_neg_rows = x_np[::-1]
    x_neg_cols = x_np[:, ::-1]
    x_ro = x_np.copy()
    x_ro.flags.writeable = False

    c_ro = x_np[:8].copy()[::-1]
    c_ro.flags.writeable = False
    lbl_ro = np.arange(256, dtype=np.int32)[::-1] % 8
    lbl_ro.flags.writeable = False
    sq_ro = np.linspace(0.1, 2.0, 256, dtype=np.float32)[::-1]
    sq_ro.flags.writeable = False

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        sel_neg = extract_semdedup_coreset(x_neg_rows, k=8, nepochs=1.0, use_cuda_graph=False, seed=3)
        sel_col = extract_semdedup_coreset(x_neg_cols, k=8, nepochs=1.0, use_cuda_graph=False, seed=3)
        sel_ro = extract_semdedup_coreset(
            x_ro,
            centroids=c_ro,
            assign_labels=lbl_ro,
            assign_sq_dists=sq_ro,
            budget_m=40,
        )
        if torch.cuda.is_available():
            sel_gpu_np = extract_semdedup_coreset(
                x_ro, k=8, retention_frac=0.25, nepochs=1.0, device="cuda:0", use_cuda_graph=False, seed=3
            )
            assert sel_gpu_np.indices.device.type == "cpu"
            assert x_ro[sel_gpu_np.indices].shape == (64, 16)

    assert sel_neg.indices.device.type == "cpu"
    assert sel_col.indices.device.type == "cpu"
    assert sel_ro.indices.device.type == "cpu"
    assert x_neg_rows[sel_neg.indices].shape == (51, 16)
    assert x_ro[sel_ro.indices].shape == (40, 16)

    x_nan = x_np.copy()
    x_nan[0, 0] = np.nan
    with pytest.raises(ValueError):
        extract_semdedup_coreset(x_nan, k=8, nepochs=1.0)


def test_zero_tmp_side_effects_under_home_scratch() -> None:
    scratch_root = _home_scratch_root() / f"pkg_api_iso_p{os.getpid()}"
    triton_dir = scratch_root / "triton_cache"
    triton_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["TMPDIR"] = str(scratch_root)
    env["TEMP"] = str(scratch_root)
    env["TMP"] = str(scratch_root)
    env["TRITON_CACHE_DIR"] = str(triton_dir)
    env["CUDA_VISIBLE_DEVICES"] = ""

    repo_src = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = f"{repo_src}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else repo_src

    before = _suspicious_tmp_entries()
    probe_code = (
        "import os, tempfile, torch\n"
        "from emeans import ApexEMeans, ApexPQ, ApexRVQ, extract_semdedup_coreset\n"
        "assert tempfile.gettempdir().startswith('/home/'), tempfile.gettempdir()\n"
        "x = torch.randn(192, 16)\n"
        "q = torch.randn(32, 16)\n"
        "m = ApexEMeans(d=16, k=8, nepochs=1.0, seed=5)\n"
        "m.fit(x)\n"
        "_ = m.assign(q)\n"
        "pq = ApexPQ(d=16, m=4, ks=8, nepochs=1.0, seed=5)\n"
        "pq.fit(x)\n"
        "_ = pq.adc_search(q, pq.encode(x), k_top=3)\n"
        "rvq = ApexRVQ(d=16, k1=8, k2=8, nepochs=1.0, seed=5)\n"
        "rvq.fit(x)\n"
        "sel = extract_semdedup_coreset(x, k=8, retention_frac=0.25, nepochs=1.0, seed=5)\n"
        "assert sel.num_selected == 48\n"
        "print('CHILD_PID=' + str(os.getpid()))\n"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", probe_code],
            env=env,
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        child_pid = None
        for line in proc.stdout.splitlines():
            if line.startswith("CHILD_PID="):
                child_pid = int(line.split("=", 1)[1])
        after = _suspicious_tmp_entries(pid=child_pid)
        leaked = after - before
        assert not leaked, f"Unexpected /tmp or /var/tmp artifacts created: {sorted(leaked)}"
    finally:
        import shutil

        shutil.rmtree(scratch_root, ignore_errors=True)
