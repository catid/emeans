"""emeans: Scale-adaptive GPU vector clustering, CroissAttn-FP8 quantization, PQ/RVQ compression, and SemDeDup coreset pruning."""

from typing import Any as _Any

import numpy as _np
import torch as _torch

__version__ = "0.5.0"

from .algo_extensions import (
    HYPOTHESES,
    EMeansAlgoExt,
    FalsifiableHypothesis,
    ShrinkageDiagnostics,
    empirical_bayes_finalize,
    hartigan_assign_points,
    power_mean_top2_weights,
    top2_assign_chunked,
)
from .apex_emeans import (
    APEX_ABLATION_ARMS,
    APEX_LADDER_ARMS,
    ApexEMeans,
    ApexEMeansEpochStat,
    apex_simplex_eb_finalize,
    apex_turkpp_init_centered,
    build_apex_arm,
    run_apex_head_edit_checkpoint,
    warmup_apex_kernels,
)
from .baselines import LloydKMeans, NaiveKMeansPP_EMeans, SculleyMiniBatchKMeans
from .croiss_fp8 import (
    CROISS_FP8_AVAILABLE,
    CROISS_MXFP8_AVAILABLE,
    CroissFP8Dataset,
    CroissFP8Workspace,
    compute_croiss_padded_d,
    croiss_encode_dataset,
    croiss_fp8_assign,
    croiss_fp8_assign_inplace,
    croiss_prepare_centroids_inplace,
    should_use_croiss_fp8,
    warmup_croiss_fp8_kernels,
)
from .datasets import (
    DatasetSplit,
    deduplicate_splits,
    load_cifar10_resnet18_embeddings,
    make_synthetic_dataset,
)
from .emeans import EMeans, EpochStat
from .emeans_pp import EMeansPP, EMeansPPEpochStat
from .fast_ops import (
    FLOAT8_E4M3FN,
    FP8Quantizer,
    FastEMeans,
    IVFAssigner,
    MultiGPUFastEMeans,
    blackwell_prepare_centroids,
    blackwell_triton_assign,
    block_permute_indices,
    fast_seed_emeans_plusplus,
    fp8_assign,
    fused_ema_and_prepare,
    fused_scatter_stats,
    triton_greedy_kmeans_plusplus,
)
from .kernels import (
    HAS_FAISS,
    HAS_TRITON,
    assign_points,
    exact_mse_fp64,
    prepare_represented_centroids,
    torch_assign_chunked,
    triton_assign,
)
from .noise_tracks import (
    ANVIL_MAPS,
    NOISE_HYPOTHESES,
    EMeansNoiseTrack,
    HeadEditDiagnostics,
    NoiseTrackHypothesis,
    QuadMergeDiagnostics,
    anvil_polar_displacement,
    build_spatial_surprisal_order,
    execute_quad_merge,
    gram_newton_schulz_polar,
    run_head_edit_step,
)
from .pq_rvq import (
    ADCRecallMetrics,
    ApexPQ,
    ApexRVQ,
    CANONICAL_SOLVER_ALIASES,
    PQ_SUPPORTED_SOLVERS,
    RVQInvertedIndex,
    RVQ_SUPPORTED_SOLVERS,
    ReconstructionMetrics,
    evaluate_adc_recall,
    evaluate_vq_reconstruction_fp64,
    instantiate_subproblem_solver,
    normalize_solver_name,
    pack_rvq_codes_uint8,
    unpack_rvq_codes_uint8,
    warmup_pq_rvq_kernels,
)
from .real_corpora import (
    DEFAULT_MILLION_DATA_DIR,
    IVFRecallResult,
    IVFSelectivityMetrics,
    RealVectorCorpus,
    compute_exact_groundtruth_gpu,
    evaluate_ivf_recall_gpu,
    exact_assign_fp64_torch,
    exact_cosine_distortion_fp64,
    fast_exact_assign_fp64,
    fast_exact_mse_fp64,
    load_glove1m_unit,
    load_real_million_corpus,
    load_sift1m,
    read_fvecs,
    read_ivecs,
    validate_real_corpus,
    write_fvecs_atomic,
)
from .realistic_benchmarks import (
    BENCHMARK_SPEC_BY_NAME,
    DEFAULT_PHASE5_DATA_DIR,
    RealisticBenchmarkSpec,
    SEVEN_REALISTIC_BENCHMARKS,
    compute_corpus_dynamic_range_stats,
    embed_text_splits_zero_leakage,
    evaluate_fp8_assignment_stability_probe,
    load_agnews_tfidf_svd_256,
    load_cifar10_convnext_768,
    load_cifar10_patches_128,
    load_cifar10_resnet18_512,
    load_covtype_multiscale_54,
    load_glove1m_unit_benchmark,
    load_realistic_benchmark,
    load_sift1m_benchmark,
    stage_all_seven_benchmarks,
    validate_covtype_physical_ranges,
)
from .recipe import (
    ALPHA_LAW,
    ALPHA_MIN_FACTOR,
    AUTO_QUALITY_MAX_D,
    BATCH_LAWS,
    CHECK_UNUSED_EVERY,
    DEFAULT_INIT_USAGE,
    DEFAULT_UNUSED_THRESHOLD,
    NO_REVIVAL_MAX_D,
    SPLIT_EPSILON,
    EMeansRecipe,
    build_recipe,
    compute_epoch_rows,
    compute_step_law,
    epoch_seed,
    resolve_batch_law,
    resolve_unused_threshold,
    round_power_of_two,
    scheduled_alpha,
)
from .seeding import (
    SeedingResult,
    block_greedy_kmeans_plusplus_torch,
    greedy_kmeans_plusplus_torch,
    lightweight_coreset,
    seed_emeans_plusplus,
    voronoi_polish_and_priors,
)
from .semdedup_coreset import (
    CoresetSelection,
    DEFAULT_SEMDEDUP_DATA_DIR,
    FOUR_LABELED_BENCHMARKS,
    LABELED_SPEC_BY_NAME,
    LabeledBenchmarkSpec,
    LabeledVectorDataset,
    MEGAPIXEL_SPEC_BY_NAME,
    MegapixelVQDataset,
    MegapixelVQSpec,
    SemDeDupScoreState,
    TWO_MEGAPIXEL_VQ_BENCHMARKS,
    centered_assign_fp32,
    compute_semdedup_scores_gpu,
    compute_vq_psnr_fp64,
    embed_labeled_text_zero_leakage,
    encode_vq_indices_gpu,
    evaluate_cholesky_ridge_gpu,
    evaluate_multinomial_logistic_gpu,
    evaluate_prototype_knn_gpu,
    evaluate_weighted_knn_gpu,
    load_agnews_256_labeled,
    load_cifar10_convnext_768_labeled,
    load_cifar10_resnet18_512_labeled,
    load_covtype_54_labeled,
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
from .smeans import (
    SMEANS_ABLATION_CONFIGS,
    SMeans,
    SMeansEpochStat,
    smeans_eb_finalize,
    subsampled_d15_init,
)
from .super_emeans import (
    HYBRID_ABLATION_ARMS,
    HybridSEMeans,
    SuperEMeans,
    SuperEMeansEpochStat,
    build_hybrid_arm,
    compute_dataset_origin_fp64,
    eb_james_stein_finalize_gpu,
    subsampled_d15_init_centered,
    twin_safe_huygens_protection,
)
from .ultra_emeans import (
    HALF_GAUSSIAN_MEAN,
    ULTRA_ABLATION_ARMS,
    UltraEMeans,
    UltraEMeansEpochStat,
    build_ultra_arm,
    run_ultra_head_edit_checkpoint,
    ultra_simplex_eb_finalize,
    ultra_turkpp_init_centered,
)

# Ergonomic top-level alias for `load_glove1m_unit`
load_glove1m = load_glove1m_unit


def _normalize_float_tensor(arr: _Any, dev: _torch.device, name: str) -> _torch.Tensor:
    if isinstance(arr, _np.ndarray):
        if arr.dtype.kind not in ("f", "i", "u", "b"):
            raise ValueError(f"{name} must have numeric/real dtype, got {arr.dtype}")
        target_dt = (
            _np.float64
            if arr.dtype == _np.float64
            else (_np.float16 if arr.dtype == _np.float16 else _np.float32)
        )
        if arr.dtype != target_dt or not arr.flags.c_contiguous or not arr.flags.writeable:
            arr = _np.array(arr, dtype=target_dt, copy=True, order="C")
        t = _torch.from_numpy(arr)
    elif isinstance(arr, _torch.Tensor):
        if arr.is_complex():
            raise ValueError(f"{name} must be real-valued, got {arr.dtype}")
        t = arr.detach()
        if not t.is_floating_point():
            t = t.float()
    else:
        raise TypeError(f"Expected torch.Tensor or np.ndarray for {name}, got {type(arr).__name__}")
    t = t.to(device=dev)
    return t if t.is_contiguous() else t.contiguous()


def _normalize_int64_tensor(arr: _Any, dev: _torch.device, name: str) -> _torch.Tensor:
    if isinstance(arr, _np.ndarray):
        if arr.dtype.kind not in ("i", "u"):
            raise ValueError(f"{name} must have integer dtype, got {arr.dtype}")
        if arr.dtype != _np.int64 or not arr.flags.c_contiguous or not arr.flags.writeable:
            arr = _np.array(arr, dtype=_np.int64, copy=True, order="C")
        t = _torch.from_numpy(arr)
    elif isinstance(arr, _torch.Tensor):
        if arr.is_floating_point() or arr.is_complex() or arr.dtype == _torch.bool:
            raise ValueError(f"{name} must have integer dtype, got {arr.dtype}")
        t = arr.detach().to(dtype=_torch.int64)
    else:
        raise TypeError(f"Expected torch.Tensor or np.ndarray for {name}, got {type(arr).__name__}")
    return t.to(device=dev).contiguous()


def extract_semdedup_coreset(
    x_train: _Any,
    k: int | None = None,
    retention_frac: float | None = None,
    *,
    budget_m: int | None = None,
    prototype: bool = False,
    nepochs: float = 2.0,
    centroids: _Any | None = None,
    assign_labels: _Any | None = None,
    assign_sq_dists: _Any | None = None,
    window_size: int = 24,
    device: _Any | None = None,
    seed: int = 1234,
    **solver_kwargs: _Any,
) -> CoresetSelection:
    """One-call SemDeDup + Horvitz-Thompson importance-weighted coreset extraction.

    Clusters ``x_train`` with ``ApexEMeans`` (unless pre-fitted ``centroids`` are supplied),
    computes fused GPU intra-cluster margin & density scores via ``compute_semdedup_scores_gpu``,
    and returns an unbiased Horvitz-Thompson ``CoresetSelection`` at ``retention_frac``
    (default ``0.20``), exact ``budget_m``, or ``prototype=True`` (1 medoid per active cluster).
    """
    if getattr(x_train, "ndim", None) != 2:
        raise ValueError(f"Expected 2D x_train of shape (n, d), got {getattr(x_train, 'shape', type(x_train))}")
    if prototype and (retention_frac is not None or budget_m is not None):
        raise ValueError("prototype=True cannot be combined with retention_frac or budget_m")
    if retention_frac is not None and budget_m is not None:
        raise ValueError("Specify at most one of retention_frac or budget_m")
    if budget_m is not None and int(budget_m) < 1:
        raise ValueError(f"budget_m must be >= 1, got {budget_m}")

    input_is_numpy = isinstance(x_train, _np.ndarray)
    if device is not None:
        exec_dev = _torch.device(device)
    elif isinstance(x_train, _torch.Tensor):
        exec_dev = x_train.device
    elif solver_kwargs.get("devices"):
        exec_dev = _torch.device(solver_kwargs["devices"][0])
    elif isinstance(centroids, _torch.Tensor):
        exec_dev = centroids.device
    else:
        exec_dev = (
            _torch.device("cuda", _torch.cuda.current_device())
            if _torch.cuda.is_available()
            else _torch.device("cpu")
        )

    x_t = _normalize_float_tensor(x_train, exec_dev, "x_train")
    if not bool(_torch.isfinite(x_t).all()):
        raise ValueError("Input x_train must contain only finite values (no NaN or Inf)")

    if centroids is None:
        if k is None:
            raise ValueError("Provide k when centroids is None")
        k_eff = int(k)
        d = int(x_t.shape[1])
        solver = ApexEMeans(d=d, k=k_eff, nepochs=nepochs, seed=seed, **solver_kwargs)
        c_fit, _ = solver.fit(x_t, capture_assignments=False)
        c_t = _normalize_float_tensor(c_fit, exec_dev, "centroids")
    else:
        if getattr(centroids, "ndim", None) != 2:
            raise ValueError(f"Expected 2D centroids of shape (k, d), got {getattr(centroids, 'shape', type(centroids))}")
        c_t = _normalize_float_tensor(centroids, exec_dev, "centroids")
        if not bool(_torch.isfinite(c_t).all()):
            raise ValueError("Input centroids must contain only finite values (no NaN or Inf)")
        k_eff = int(c_t.shape[0]) if k is None else int(k)

    lbl_t = _normalize_int64_tensor(assign_labels, exec_dev, "assign_labels") if assign_labels is not None else None
    sq_t = _normalize_float_tensor(assign_sq_dists, exec_dev, "assign_sq_dists") if assign_sq_dists is not None else None

    state = compute_semdedup_scores_gpu(
        x_train=x_t,
        centroids=c_t,
        k=k_eff,
        assign_labels=lbl_t,
        assign_sq_dists=sq_t,
        window_size=window_size,
    )
    if prototype:
        sel = select_prototype_coreset(state)
    elif budget_m is not None:
        sel = select_coreset_at_budget(state, budget_m=int(budget_m))
    else:
        eff_frac = 0.20 if retention_frac is None else float(retention_frac)
        sel = select_coreset_at_budget(state, retention_frac=eff_frac)

    if input_is_numpy and sel.indices.device.type != "cpu":
        return CoresetSelection(
            indices=sel.indices.cpu(),
            weights=sel.weights.cpu(),
            cluster_ids=sel.cluster_ids.cpu(),
            num_selected=sel.num_selected,
            active_clusters_covered=sel.active_clusters_covered,
            select_time_sec=sel.select_time_sec,
        )
    return sel


__all__ = [
    "ADCRecallMetrics",
    "ALPHA_LAW",
    "ALPHA_MIN_FACTOR",
    "ANVIL_MAPS",
    "APEX_ABLATION_ARMS",
    "APEX_LADDER_ARMS",
    "AUTO_QUALITY_MAX_D",
    "ApexEMeans",
    "ApexEMeansEpochStat",
    "ApexPQ",
    "ApexRVQ",
    "BATCH_LAWS",
    "BENCHMARK_SPEC_BY_NAME",
    "CANONICAL_SOLVER_ALIASES",
    "CHECK_UNUSED_EVERY",
    "CROISS_FP8_AVAILABLE",
    "CROISS_MXFP8_AVAILABLE",
    "CoresetSelection",
    "CroissFP8Dataset",
    "CroissFP8Workspace",
    "DEFAULT_INIT_USAGE",
    "DEFAULT_MILLION_DATA_DIR",
    "DEFAULT_PHASE5_DATA_DIR",
    "DEFAULT_SEMDEDUP_DATA_DIR",
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
    "FOUR_LABELED_BENCHMARKS",
    "FP8Quantizer",
    "FalsifiableHypothesis",
    "FastEMeans",
    "HALF_GAUSSIAN_MEAN",
    "HAS_FAISS",
    "HAS_TRITON",
    "HYBRID_ABLATION_ARMS",
    "HYPOTHESES",
    "HeadEditDiagnostics",
    "HybridSEMeans",
    "IVFAssigner",
    "IVFRecallResult",
    "IVFSelectivityMetrics",
    "LABELED_SPEC_BY_NAME",
    "LabeledBenchmarkSpec",
    "LabeledVectorDataset",
    "LloydKMeans",
    "MEGAPIXEL_SPEC_BY_NAME",
    "MegapixelVQDataset",
    "MegapixelVQSpec",
    "MultiGPUFastEMeans",
    "NOISE_HYPOTHESES",
    "NO_REVIVAL_MAX_D",
    "NaiveKMeansPP_EMeans",
    "NoiseTrackHypothesis",
    "PQ_SUPPORTED_SOLVERS",
    "QuadMergeDiagnostics",
    "RVQInvertedIndex",
    "RVQ_SUPPORTED_SOLVERS",
    "RealVectorCorpus",
    "RealisticBenchmarkSpec",
    "ReconstructionMetrics",
    "SEVEN_REALISTIC_BENCHMARKS",
    "SMEANS_ABLATION_CONFIGS",
    "SMeans",
    "SMeansEpochStat",
    "SPLIT_EPSILON",
    "SculleyMiniBatchKMeans",
    "SeedingResult",
    "SemDeDupScoreState",
    "ShrinkageDiagnostics",
    "SuperEMeans",
    "SuperEMeansEpochStat",
    "TWO_MEGAPIXEL_VQ_BENCHMARKS",
    "ULTRA_ABLATION_ARMS",
    "UltraEMeans",
    "UltraEMeansEpochStat",
    "__version__",
    "anvil_polar_displacement",
    "apex_simplex_eb_finalize",
    "apex_turkpp_init_centered",
    "assign_points",
    "blackwell_prepare_centroids",
    "blackwell_triton_assign",
    "block_greedy_kmeans_plusplus_torch",
    "block_permute_indices",
    "build_apex_arm",
    "build_hybrid_arm",
    "build_recipe",
    "build_spatial_surprisal_order",
    "build_ultra_arm",
    "centered_assign_fp32",
    "compute_corpus_dynamic_range_stats",
    "compute_croiss_padded_d",
    "compute_dataset_origin_fp64",
    "compute_epoch_rows",
    "compute_exact_groundtruth_gpu",
    "compute_semdedup_scores_gpu",
    "compute_step_law",
    "compute_vq_psnr_fp64",
    "croiss_encode_dataset",
    "croiss_fp8_assign",
    "croiss_fp8_assign_inplace",
    "croiss_prepare_centroids_inplace",
    "deduplicate_splits",
    "eb_james_stein_finalize_gpu",
    "embed_labeled_text_zero_leakage",
    "embed_text_splits_zero_leakage",
    "empirical_bayes_finalize",
    "encode_vq_indices_gpu",
    "epoch_seed",
    "evaluate_adc_recall",
    "evaluate_cholesky_ridge_gpu",
    "evaluate_fp8_assignment_stability_probe",
    "evaluate_ivf_recall_gpu",
    "evaluate_multinomial_logistic_gpu",
    "evaluate_prototype_knn_gpu",
    "evaluate_vq_reconstruction_fp64",
    "evaluate_weighted_knn_gpu",
    "exact_assign_fp64_torch",
    "exact_cosine_distortion_fp64",
    "exact_mse_fp64",
    "execute_quad_merge",
    "extract_semdedup_coreset",
    "fast_exact_assign_fp64",
    "fast_exact_mse_fp64",
    "fast_seed_emeans_plusplus",
    "fp8_assign",
    "fused_ema_and_prepare",
    "fused_scatter_stats",
    "gram_newton_schulz_polar",
    "greedy_kmeans_plusplus_torch",
    "hartigan_assign_points",
    "instantiate_subproblem_solver",
    "lightweight_coreset",
    "load_agnews_256_labeled",
    "load_agnews_tfidf_svd_256",
    "load_cifar10_convnext_768",
    "load_cifar10_convnext_768_labeled",
    "load_cifar10_patches_128",
    "load_cifar10_resnet18_512",
    "load_cifar10_resnet18_512_labeled",
    "load_cifar10_resnet18_embeddings",
    "load_covtype_54_labeled",
    "load_covtype_multiscale_54",
    "load_glove1m",
    "load_glove1m_unit",
    "load_glove1m_unit_benchmark",
    "load_labeled_benchmark",
    "load_megapixel_vq_benchmark",
    "load_real_million_corpus",
    "load_realistic_benchmark",
    "load_sift1m",
    "load_sift1m_benchmark",
    "make_synthetic_dataset",
    "normalize_solver_name",
    "pack_rvq_codes_uint8",
    "power_mean_top2_weights",
    "prepare_represented_centroids",
    "read_fvecs",
    "read_ivecs",
    "reconstruct_patch_mosaic_rgb",
    "resolve_batch_law",
    "resolve_unused_threshold",
    "round_power_of_two",
    "run_apex_head_edit_checkpoint",
    "run_head_edit_step",
    "run_ultra_head_edit_checkpoint",
    "scheduled_alpha",
    "seed_emeans_plusplus",
    "select_coreset_at_budget",
    "select_prototype_coreset",
    "select_random_coreset",
    "should_use_croiss_fp8",
    "smeans_eb_finalize",
    "stage_all_seven_benchmarks",
    "subsampled_d15_init",
    "subsampled_d15_init_centered",
    "top2_assign_chunked",
    "torch_assign_chunked",
    "triton_assign",
    "triton_greedy_kmeans_plusplus",
    "twin_safe_huygens_protection",
    "ultra_simplex_eb_finalize",
    "ultra_turkpp_init_centered",
    "unpack_rvq_codes_uint8",
    "validate_covtype_physical_ranges",
    "validate_labeled_dataset",
    "validate_megapixel_vq_dataset",
    "validate_real_corpus",
    "voronoi_polish_and_priors",
    "warmup_apex_kernels",
    "warmup_croiss_fp8_kernels",
    "warmup_pq_rvq_kernels",
    "warmup_semdedup_kernels",
    "write_fvecs_atomic",
]
