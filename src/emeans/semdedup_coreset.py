"""Downstream ML Pipeline Benchmarks: Semantic Data Deduplication (`SemDeDup`) & Megapixel VQ.

Provides zero-leakage dataset loaders, fused GPU `SemDeDup` / prototype coreset selection kernels,
zero-leakage GPU downstream classifiers, and Megapixel RGB & `4x4x3` patch vector quantization
evaluators for proving `ApexEMeans` end-to-end ML pipeline dominance:

  1. **Real Labeled ML Datasets (`FOUR_LABELED_BENCHMARKS`, Zero Train/Test Split Leakage)**:
     - `agnews_256`          (`~110,000` train, `~7,590` test, `4` classes, `256-D` train-only TF-IDF + SVD)
     - `cifar10_resnet18_512`(` ~49,975` train, `~9,986` test, `10` classes, `512-D` ReLU CNN features)
     - `cifar10_convnext_768`(` ~49,975` train, `~9,986` test, `10` classes, `768-D` LayerNorm ConvNeXt-Tiny)
     - `covtype_54`          (` 500,000` train, `50,000` test, `7` classes, `54-D` raw cartographic & soil
                              sensors with `110,761x` feature dynamic range)

  2. **Zero-Loop GPU `SemDeDup` & Horvitz-Thompson Importance-Weighted Coreset Selection**:
     - Unifies Abbas et al. (`SemDeDup`, 2023) intra-cluster near-duplicate elimination, Sorscher et al.
       (*Beyond Neural Scaling Laws*, 2022) prototype-boundary margin ranking, and unbiased
       Horvitz-Thompson cluster-mass weighting `w_i = n_c / m_c` inside a single fused Triton pass
       (`_semdedup_margin_kernel`, `0` Python loops over `K` or `N`, `100%` translation-invariant).

  3. **Zero-Leakage GPU Downstream Classifiers (Trained Strictly on Pruned Coreset Subset)**:
     - Closed-form Golub-Heath-Wahba Leave-One-Out GCV Cholesky Ridge Probe (`evaluate_cholesky_ridge_gpu`),
     - GPU Full-Batch L-BFGS Multinomial Logistic Regression Probe (`evaluate_multinomial_logistic_gpu`),
     - Coreset-Bandwidth Origin-Centered Weighted `$k$-NN` & Prototype `$k$-NN` (`evaluate_weighted_knn_gpu`,
       `evaluate_prototype_knn_gpu`, `100%` independent per test query with `0` test-derived statistics).

  4. **Megapixel High-Resolution Image Color & `4x4x3` Patch Vector Quantization (`Megapixel VQ`)**:
     - `megapixel_rgb_3d`    (`1,000,000 x 3` RGB train mosaic, `262,144 x 3` disjoint test mosaic)
     - `megapixel_patch_48d` (`  500,000 x 48` `4x4x3` train patches = `8.0 MP`, `100,000 x 48` disjoint test)
     - Exact IEEE-754 FP64 per-channel reconstruction PSNR (`10 * log10(255^2 / (VectorMSE / D))`).
"""

from __future__ import annotations

import csv
import math
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .datasets import (
    DEFAULT_CIFAR10_CANDIDATES,
    DEFAULT_CIFAR10_ROOT,
    DEFAULT_RESNET18_CANDIDATES,
    DEFAULT_RESNET18_CKPT,
)
from .fast_ops import HAS_TRITON
from .kernels import assign_points
from .real_corpora import _exact_row_keys, atomic_savez, compute_sha256, fast_exact_assign_fp64
from .realistic_benchmarks import (
    AG_NEWS_TEST_FILENAME,
    AG_NEWS_TEST_SHA256,
    AG_NEWS_TEST_SIZE,
    AG_NEWS_TEST_URLS,
    AG_NEWS_TRAIN_FILENAME,
    AG_NEWS_TRAIN_SHA256,
    AG_NEWS_TRAIN_SIZE,
    AG_NEWS_TRAIN_URLS,
    CONVNEXT_TINY_FILENAME,
    CONVNEXT_TINY_SHA256,
    CONVNEXT_TINY_SIZE,
    CONVNEXT_TINY_URL,
    COVTYPE_FILENAME,
    COVTYPE_SHA256,
    COVTYPE_SIZE,
    COVTYPE_URLS,
    DEFAULT_PHASE5_DATA_DIR,
    _download_atomic_verified,
    _normalize_doc_text,
    validate_covtype_physical_ranges,
)

if HAS_TRITON:
    import triton
    import triton.language as tl


_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEMDEDUP_DATA_DIR = _REPO_ROOT / "outputs" / "semdedup" / "data"


# ==============================================================================
# 1. Benchmark Metadata & Validated Dataset Containers
# ==============================================================================


@dataclass(frozen=True)
class LabeledBenchmarkSpec:
    """Metadata descriptor for one real labeled ML dataset in the SemDeDup suite."""

    name: str
    short_label: str
    modality: str
    d: int
    num_classes: int
    k_default: int
    cache_filename: str
    description: str


FOUR_LABELED_BENCHMARKS: tuple[LabeledBenchmarkSpec, ...] = (
    LabeledBenchmarkSpec(
        name="agnews_256",
        short_label="AG-News SVD (110K×256, 4 cls)",
        modality="Heavy-Tailed News Topic Embeddings",
        d=256,
        num_classes=4,
        k_default=1024,
        cache_filename="semdedup_agnews_256_s20261116_v1.npz",
        description="~110,000 train + ~7,590 official test 4-class news embeddings via train-only TF-IDF + SVD.",
    ),
    LabeledBenchmarkSpec(
        name="cifar10_resnet18_512",
        short_label="CIFAR-10 ResNet-18 (50K×512, 10 cls)",
        modality="Global ReLU CNN Vision Embeddings",
        d=512,
        num_classes=10,
        k_default=1024,
        cache_filename="semdedup_cifar10_resnet18_512_v1.npz",
        description="~49,975 train + ~9,986 official test 10-class ResNet-18 penultimate ReLU features.",
    ),
    LabeledBenchmarkSpec(
        name="cifar10_convnext_768",
        short_label="CIFAR-10 ConvNeXt (50K×768, 10 cls)",
        modality="Modern High-D Vision LayerNorm (768D)",
        d=768,
        num_classes=10,
        k_default=2048,
        cache_filename="semdedup_cifar10_convnext_768_v1.npz",
        description="~49,975 train + ~9,986 official test 10-class ConvNeXt-Tiny 768-D signed embeddings.",
    ),
    LabeledBenchmarkSpec(
        name="covtype_54",
        short_label="Forest Covertype (500K×54, 7 cls)",
        modality="Multi-Scale Physical Sensors (110,761x span)",
        d=54,
        num_classes=7,
        k_default=4096,
        cache_filename="semdedup_covtype_54_s20261116_v1.npz",
        description="500,000 train + 50,000 held-out test 7-class raw USFS cartographic & soil sensors.",
    ),
)

LABELED_SPEC_BY_NAME: dict[str, LabeledBenchmarkSpec] = {s.name: s for s in FOUR_LABELED_BENCHMARKS}


@dataclass
class LabeledVectorDataset:
    """Strictly disjoint `(x_train, y_train)` / `(x_test, y_test)` corpus with ground-truth labels."""

    name: str
    short_label: str
    modality: str
    x_train: np.ndarray  # (N_train, D) float32
    y_train: np.ndarray  # (N_train,) int64 in [0, num_classes - 1]
    x_test: np.ndarray   # (N_test, D) float32
    y_test: np.ndarray   # (N_test,) int64 in [0, num_classes - 1]
    d: int
    num_classes: int
    k_default: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MegapixelVQSpec:
    """Metadata descriptor for one high-resolution Megapixel VQ benchmark."""

    name: str
    short_label: str
    modality: str
    d: int
    patch_h: int
    patch_w: int
    channels: int
    n_train: int
    n_test: int
    train_mosaic_hw: tuple[int, int]
    test_mosaic_hw: tuple[int, int]
    k_values: tuple[int, ...]
    cache_filename: str
    description: str


TWO_MEGAPIXEL_VQ_BENCHMARKS: tuple[MegapixelVQSpec, ...] = (
    MegapixelVQSpec(
        name="megapixel_rgb_3d",
        short_label="Megapixel Color Palette VQ (1M×3 RGB)",
        modality="High-Res RGB Color Quantization",
        d=3,
        patch_h=1,
        patch_w=1,
        channels=3,
        n_train=1_000_000,
        n_test=262_144,
        train_mosaic_hw=(1000, 1000),
        test_mosaic_hw=(512, 512),
        k_values=(256, 1024, 4096),
        cache_filename="megapixel_rgb_3d_s20261116_v1.npz",
        description="1,000,000 x 3 RGB pixels (1000x1000 train mosaic) + 262,144 x 3 disjoint held-out test mosaic.",
    ),
    MegapixelVQSpec(
        name="megapixel_patch_48d",
        short_label="Megapixel 4×4 Patch VQ (500K×48)",
        modality="Natural Image 4×4×3 Patch VQ (8.0 MP)",
        d=48,
        patch_h=4,
        patch_w=4,
        channels=3,
        n_train=500_000,
        n_test=100_000,
        train_mosaic_hw=(2000, 4000),
        test_mosaic_hw=(800, 2000),
        k_values=(256, 1024, 4096),
        cache_filename="megapixel_patch_48d_s20261116_v1.npz",
        description="500,000 x 48 non-overlapping 4x4x3 RGB patches (8.0 MP train mosaic) + 100,000 x 48 disjoint test.",
    ),
)

MEGAPIXEL_SPEC_BY_NAME: dict[str, MegapixelVQSpec] = {s.name: s for s in TWO_MEGAPIXEL_VQ_BENCHMARKS}


@dataclass
class MegapixelVQDataset:
    """Validated Megapixel RGB / `4x4x3` patch dataset built from strictly disjoint source images."""

    name: str
    short_label: str
    modality: str
    x_train: np.ndarray  # (N_train, D) float32 in [0.0, 255.0]
    x_test: np.ndarray   # (N_test, D) float32 in [0.0, 255.0]
    d: int
    patch_h: int
    patch_w: int
    channels: int
    train_mosaic_hw: tuple[int, int]
    test_mosaic_hw: tuple[int, int]
    k_values: tuple[int, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# 2. Cross-Split Deduplication & Zero-Leakage Dataset Validators
# ==============================================================================


def _rounded_row_keys(arr: np.ndarray, decimals: int = 6) -> np.ndarray:
    rounded = np.round(np.ascontiguousarray(arr, dtype=np.float32), decimals=decimals)
    rounded = np.where(rounded == 0.0, 0.0, rounded).astype(np.float32, copy=False)
    return np.ascontiguousarray(rounded).view(
        np.dtype((np.void, rounded.shape[1] * rounded.dtype.itemsize))
    ).ravel()


def _deduplicate_labeled_two_way(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    decimals: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deduplicate within `train`/`test` and purge any `test` row colliding with `train` (rounded + exact bytes)."""
    x_tr = np.ascontiguousarray(x_train, dtype=np.float32)
    y_tr = np.ascontiguousarray(y_train, dtype=np.int64)
    x_te = np.ascontiguousarray(x_test, dtype=np.float32)
    y_te = np.ascontiguousarray(y_test, dtype=np.int64)

    if x_tr.shape[0] != y_tr.shape[0] or x_te.shape[0] != y_te.shape[0]:
        raise ValueError("Feature/label length mismatch before labeled deduplication")

    # Pass 1: rounded decimal keys
    rk_tr = _rounded_row_keys(x_tr, decimals=decimals)
    _, idx_tr1 = np.unique(rk_tr, return_index=True)
    idx_tr1.sort()
    x_tr1 = x_tr[idx_tr1]
    y_tr1 = y_tr[idx_tr1]
    seen_r = rk_tr[idx_tr1]

    rk_te = _rounded_row_keys(x_te, decimals=decimals)
    _, idx_te1 = np.unique(rk_te, return_index=True)
    idx_te1.sort()
    keep_te1 = idx_te1[~np.isin(rk_te[idx_te1], seen_r)]
    x_te1 = x_te[keep_te1]
    y_te1 = y_te[keep_te1]

    # Pass 2: exact float32 byte keys (with -0.0 canonicalized to +0.0)
    ek_tr = _exact_row_keys(x_tr1)
    _, idx_tr2 = np.unique(ek_tr, return_index=True)
    idx_tr2.sort()
    x_tr_clean = np.ascontiguousarray(x_tr1[idx_tr2], dtype=np.float32)
    y_tr_clean = np.ascontiguousarray(y_tr1[idx_tr2], dtype=np.int64)
    seen_e = ek_tr[idx_tr2]

    ek_te = _exact_row_keys(x_te1)
    _, idx_te2 = np.unique(ek_te, return_index=True)
    idx_te2.sort()
    keep_te2 = idx_te2[~np.isin(ek_te[idx_te2], seen_e)]
    x_te_clean = np.ascontiguousarray(x_te1[keep_te2], dtype=np.float32)
    y_te_clean = np.ascontiguousarray(y_te1[keep_te2], dtype=np.int64)

    return x_tr_clean, y_tr_clean, x_te_clean, y_te_clean


def validate_labeled_dataset(
    ds: LabeledVectorDataset,
    check_cross_split_disjoint: bool = True,
) -> None:
    """Enforce finiteness, shape/dtype parity, class-index bounds, and strict train/test disjointness."""
    if ds.x_train.ndim != 2 or ds.x_test.ndim != 2:
        raise ValueError(f"{ds.name}: expected 2D feature arrays, got {ds.x_train.shape}, {ds.x_test.shape}")
    if ds.x_train.shape[1] != ds.d or ds.x_test.shape[1] != ds.d:
        raise ValueError(f"{ds.name}: dimension mismatch vs d={ds.d}: {ds.x_train.shape}, {ds.x_test.shape}")
    if ds.x_train.dtype != np.float32 or ds.x_test.dtype != np.float32:
        raise ValueError(f"{ds.name}: expected float32 features, got {ds.x_train.dtype}, {ds.x_test.dtype}")
    if ds.y_train.shape != (ds.x_train.shape[0],) or ds.y_test.shape != (ds.x_test.shape[0],):
        raise ValueError(f"{ds.name}: label shape mismatch {ds.y_train.shape}, {ds.y_test.shape}")
    if ds.y_train.dtype != np.int64 or ds.y_test.dtype != np.int64:
        raise ValueError(f"{ds.name}: expected int64 labels, got {ds.y_train.dtype}, {ds.y_test.dtype}")
    if not (np.isfinite(ds.x_train).all() and np.isfinite(ds.x_test).all()):
        raise ValueError(f"{ds.name}: NaN/Inf detected in feature matrices")
    if int(ds.y_train.min()) < 0 or int(ds.y_train.max()) >= ds.num_classes:
        raise ValueError(f"{ds.name}: y_train out of range [0, {ds.num_classes - 1}]")
    if int(ds.y_test.min()) < 0 or int(ds.y_test.max()) >= ds.num_classes:
        raise ValueError(f"{ds.name}: y_test out of range [0, {ds.num_classes - 1}]")

    if check_cross_split_disjoint:
        ktr = _exact_row_keys(ds.x_train)
        kte = _exact_row_keys(ds.x_test)
        if np.unique(ktr).size != ktr.size:
            raise ValueError(f"DATA LEAKAGE ({ds.name}): duplicate feature rows inside x_train")
        if np.unique(kte).size != kte.size:
            raise ValueError(f"DATA LEAKAGE ({ds.name}): duplicate feature rows inside x_test")
        if np.intersect1d(ktr, kte).size > 0:
            raise ValueError(f"DATA LEAKAGE ({ds.name}): cross-split collision between x_train and x_test!")


def validate_megapixel_vq_dataset(
    ds: MegapixelVQDataset,
    check_patch_disjoint: bool = True,
) -> None:
    """Enforce `[0, 255]` pixel bounds, shape/dtype parity, and patch-level split disjointness."""
    if ds.x_train.ndim != 2 or ds.x_test.ndim != 2:
        raise ValueError(f"{ds.name}: expected 2D arrays, got {ds.x_train.shape}, {ds.x_test.shape}")
    if ds.x_train.shape[1] != ds.d or ds.x_test.shape[1] != ds.d:
        raise ValueError(f"{ds.name}: dimension mismatch vs d={ds.d}")
    if ds.d != ds.patch_h * ds.patch_w * ds.channels:
        raise ValueError(f"{ds.name}: d={ds.d} != {ds.patch_h}*{ds.patch_w}*{ds.channels}")
    if ds.x_train.dtype != np.float32 or ds.x_test.dtype != np.float32:
        raise ValueError(f"{ds.name}: expected float32 arrays")
    if not (np.isfinite(ds.x_train).all() and np.isfinite(ds.x_test).all()):
        raise ValueError(f"{ds.name}: NaN/Inf detected in VQ pixel/patch arrays")
    for tag, arr in (("x_train", ds.x_train), ("x_test", ds.x_test)):
        lo, hi = float(arr.min()), float(arr.max())
        if lo < -1e-4 or hi > 255.0 + 1e-4:
            raise ValueError(f"{ds.name}.{tag} outside [0, 255] RGB range: [{lo}, {hi}]")

    if check_patch_disjoint and ds.d >= 12:
        ktr = _exact_row_keys(ds.x_train)
        kte = _exact_row_keys(ds.x_test)
        if np.intersect1d(ktr, kte).size > 0:
            raise ValueError(f"DATA LEAKAGE ({ds.name}): exact patch overlap between x_train and x_test!")


# ==============================================================================
# 3. Zero-Leakage Loaders for All 4 Labeled ML Datasets
# ==============================================================================


def _resolve_verified_raw_file(
    filename: str,
    urls: tuple[str, ...],
    expected_size: int,
    expected_sha256: str,
    target_dir: Path,
) -> Path:
    """Reuse SHA-256 verified raw archive from `phase5_dominance/data` if staged, else download atomically."""
    for candidate_dir in (target_dir, DEFAULT_PHASE5_DATA_DIR):
        cand = candidate_dir / filename
        if cand.exists() and cand.stat().st_size == expected_size:
            if compute_sha256(cand) == expected_sha256:
                return cand
    dst = target_dir / filename
    _download_atomic_verified(
        urls=urls,
        dst=dst,
        expected_size=expected_size,
        expected_sha256=expected_sha256,
    )
    return dst


def _load_cached_labeled_npz(
    cache_path: Path,
    spec: LabeledBenchmarkSpec,
    extra_meta: dict[str, Any] | None = None,
) -> LabeledVectorDataset:
    with np.load(cache_path, allow_pickle=False) as z:
        meta = dict(extra_meta or {})
        for k in z.files:
            if k not in ("x_train", "y_train", "x_test", "y_test"):
                arr_k = z[k]
                if arr_k.ndim == 0:
                    meta[k] = arr_k.item()
        ds = LabeledVectorDataset(
            name=spec.name,
            short_label=spec.short_label,
            modality=spec.modality,
            x_train=np.ascontiguousarray(z["x_train"], dtype=np.float32),
            y_train=np.ascontiguousarray(z["y_train"], dtype=np.int64),
            x_test=np.ascontiguousarray(z["x_test"], dtype=np.float32),
            y_test=np.ascontiguousarray(z["y_test"], dtype=np.int64),
            d=spec.d,
            num_classes=spec.num_classes,
            k_default=spec.k_default,
            metadata=meta,
        )
    validate_labeled_dataset(ds, check_cross_split_disjoint=True)
    return ds


def _load_raw_cifar10_with_labels(
    cifar_dir: Path = DEFAULT_CIFAR10_ROOT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load raw CIFAR-10 uint8 `(N, 3, 32, 32)` images AND exact `int64` class labels `(0..9)`."""
    resolved = cifar_dir
    if not resolved.exists():
        for cand in DEFAULT_CIFAR10_CANDIDATES:
            if cand.exists():
                resolved = cand
                break
    if not resolved.exists():
        raise FileNotFoundError(f"CIFAR-10 directory not found at {cifar_dir}")

    tr_imgs, tr_lbls = [], []
    for i in range(1, 6):
        with open(resolved / f"data_batch_{i}", "rb") as f:
            entry = pickle.load(f, encoding="bytes")
        tr_imgs.append(entry[b"data"].reshape(-1, 3, 32, 32))
        tr_lbls.append(np.asarray(entry[b"labels"], dtype=np.int64))
    x_train_raw = np.concatenate(tr_imgs, axis=0)
    y_train_raw = np.concatenate(tr_lbls, axis=0)

    with open(resolved / "test_batch", "rb") as f:
        test_entry = pickle.load(f, encoding="bytes")
    x_test_raw = test_entry[b"data"].reshape(-1, 3, 32, 32)
    y_test_raw = np.asarray(test_entry[b"labels"], dtype=np.int64)
    return x_train_raw, y_train_raw, x_test_raw, y_test_raw


def _deduplicate_raw_labeled_images_two_way(
    train_imgs: np.ndarray,
    train_lbls: np.ndarray,
    test_imgs: np.ndarray,
    test_lbls: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deduplicate raw uint8 `(N, 3, 32, 32)` images within & across `(train -> test)` BEFORE extraction."""
    def _img_keys(imgs: np.ndarray) -> np.ndarray:
        flat = np.ascontiguousarray(imgs.reshape(imgs.shape[0], -1))
        return flat.view(np.dtype((np.void, flat.shape[1]))).ravel()

    k_tr = _img_keys(train_imgs)
    _, idx_tr = np.unique(k_tr, return_index=True)
    idx_tr.sort()
    tr_clean = np.ascontiguousarray(train_imgs[idx_tr])
    y_tr_clean = np.ascontiguousarray(train_lbls[idx_tr], dtype=np.int64)
    seen_tr = k_tr[idx_tr]

    k_te = _img_keys(test_imgs)
    _, idx_te = np.unique(k_te, return_index=True)
    idx_te.sort()
    keep_te = idx_te[~np.isin(k_te[idx_te], seen_tr)]
    te_clean = np.ascontiguousarray(test_imgs[keep_te])
    y_te_clean = np.ascontiguousarray(test_lbls[keep_te], dtype=np.int64)
    return tr_clean, y_tr_clean, te_clean, y_te_clean


def embed_labeled_text_zero_leakage(
    train_docs: list[str],
    y_train: np.ndarray,
    test_docs: list[str],
    y_test: np.ndarray,
    d: int = 256,
    max_features: int = 50_000,
    min_df: int = 3,
    svd_seed: int = 20261116,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Embed labeled `(train, test)` text documents with STRICT train-only TF-IDF and train-only SVD.

    Zero-Leakage Guarantee:
      - Raw text strings are normalized and deduplicated case-insensitively across `train -> test` first.
      - `TfidfVectorizer` vocabulary & IDF weights are fit **exclusively** on `train_docs`.
      - `TruncatedSVD` singular vectors are fit **exclusively** on `X_train_tfidf`.
      - `test_docs` undergo pure out-of-sample `.transform(...)` projections without refitting.
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    if len(train_docs) != len(y_train) or len(test_docs) != len(y_test):
        raise ValueError("Document list and label array lengths must match")

    seen: set[str] = set()
    clean_tr_docs: list[str] = []
    clean_tr_y: list[int] = []
    for doc, lbl in zip(train_docs, y_train):
        norm = _normalize_doc_text(doc)
        key = norm.lower()
        if len(key) >= 8 and key not in seen:
            seen.add(key)
            clean_tr_docs.append(norm)
            clean_tr_y.append(int(lbl))

    clean_te_docs: list[str] = []
    clean_te_y: list[int] = []
    for doc, lbl in zip(test_docs, y_test):
        norm = _normalize_doc_text(doc)
        key = norm.lower()
        if len(key) >= 8 and key not in seen:
            seen.add(key)
            clean_te_docs.append(norm)
            clean_te_y.append(int(lbl))

    if len(clean_tr_docs) <= d:
        raise ValueError(f"Insufficient unique training documents ({len(clean_tr_docs)}) for SVD d={d}")

    vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        max_features=int(max_features),
        min_df=int(min_df),
        max_df=0.90,
        ngram_range=(1, 2),
        norm=None,
        dtype=np.float64,
    )
    x_tr_sp = vectorizer.fit_transform(clean_tr_docs)
    x_te_sp = vectorizer.transform(clean_te_docs)

    n_vocab = int(x_tr_sp.shape[1])
    if n_vocab < d:
        raise ValueError(f"Vocabulary size {n_vocab} smaller than target d={d}")

    svd = TruncatedSVD(n_components=int(d), n_iter=7, random_state=int(svd_seed))
    z_tr64 = svd.fit_transform(x_tr_sp)
    z_te64 = svd.transform(x_te_sp)

    y_tr_arr = np.asarray(clean_tr_y, dtype=np.int64)
    y_te_arr = np.asarray(clean_te_y, dtype=np.int64)

    keep_tr = np.linalg.norm(z_tr64, axis=1) > 1e-6
    keep_te = np.linalg.norm(z_te64, axis=1) > 1e-6

    x_tr32 = np.ascontiguousarray(z_tr64[keep_tr].astype(np.float32), dtype=np.float32)
    y_tr32 = np.ascontiguousarray(y_tr_arr[keep_tr], dtype=np.int64)
    x_te32 = np.ascontiguousarray(z_te64[keep_te].astype(np.float32), dtype=np.float32)
    y_te32 = np.ascontiguousarray(y_te_arr[keep_te], dtype=np.int64)

    x_tr_f, y_tr_f, x_te_f, y_te_f = _deduplicate_labeled_two_way(x_tr32, y_tr32, x_te32, y_te32, decimals=6)
    svals = svd.singular_values_
    meta = {
        "vocab_size": float(n_vocab),
        "svd_sigma_max": float(svals[0]),
        "svd_sigma_min": float(svals[-1]),
        "explained_variance_sum": float(np.sum(svd.explained_variance_ratio_)),
    }
    return x_tr_f, y_tr_f, x_te_f, y_te_f, meta


def load_agnews_256_labeled(
    data_dir: Path | str = DEFAULT_SEMDEDUP_DATA_DIR,
    n_train_limit: int = 110_000,
    split_seed: int = 20261116,
) -> LabeledVectorDataset:
    """Load `AG-News 256-D` (`~110K` train, `~7.6K` official test, `4` classes) with zero leakage."""
    ddir = Path(data_dir)
    spec = LABELED_SPEC_BY_NAME["agnews_256"]
    cache_path = ddir / spec.cache_filename
    if cache_path.exists():
        return _load_cached_labeled_npz(cache_path, spec)

    tr_csv = _resolve_verified_raw_file(
        AG_NEWS_TRAIN_FILENAME, AG_NEWS_TRAIN_URLS, AG_NEWS_TRAIN_SIZE, AG_NEWS_TRAIN_SHA256, ddir
    )
    te_csv = _resolve_verified_raw_file(
        AG_NEWS_TEST_FILENAME, AG_NEWS_TEST_URLS, AG_NEWS_TEST_SIZE, AG_NEWS_TEST_SHA256, ddir
    )

    def _parse_labeled_csv(p: Path) -> tuple[list[str], np.ndarray]:
        docs: list[str] = []
        lbls: list[int] = []
        with open(p, "r", encoding="utf-8", newline="") as f:
            for row in csv.reader(f):
                if len(row) >= 3:
                    c_id = int(row[0].strip()) - 1
                    if 0 <= c_id < 4:
                        docs.append(f"{row[1].strip()}. {row[2].strip()}")
                        lbls.append(c_id)
        return docs, np.asarray(lbls, dtype=np.int64)

    raw_tr_docs, raw_tr_y = _parse_labeled_csv(tr_csv)
    raw_te_docs, raw_te_y = _parse_labeled_csv(te_csv)

    # Deduplicate raw train strings before deterministic 110K train selection
    seen_tr: set[str] = set()
    uniq_docs: list[str] = []
    uniq_y: list[int] = []
    for doc, lbl in zip(raw_tr_docs, raw_tr_y):
        norm = _normalize_doc_text(doc)
        key = norm.lower()
        if len(key) >= 8 and key not in seen_tr:
            seen_tr.add(key)
            uniq_docs.append(norm)
            uniq_y.append(int(lbl))

    rng = np.random.default_rng(int(split_seed))
    perm = rng.permutation(len(uniq_docs))[: min(int(n_train_limit), len(uniq_docs))]
    sel_tr_docs = [uniq_docs[int(i)] for i in perm]
    sel_tr_y = np.asarray([uniq_y[int(i)] for i in perm], dtype=np.int64)

    x_tr, y_tr, x_te, y_te, svd_meta = embed_labeled_text_zero_leakage(
        train_docs=sel_tr_docs,
        y_train=sel_tr_y,
        test_docs=raw_te_docs,
        y_test=raw_te_y,
        d=256,
        max_features=50_000,
        min_df=3,
        svd_seed=split_seed,
    )
    atomic_savez(
        cache_path,
        x_train=x_tr,
        y_train=y_tr,
        x_test=x_te,
        y_test=y_te,
        vocab_size=np.float64(svd_meta["vocab_size"]),
        svd_sigma_max=np.float64(svd_meta["svd_sigma_max"]),
        svd_sigma_min=np.float64(svd_meta["svd_sigma_min"]),
        explained_variance_sum=np.float64(svd_meta["explained_variance_sum"]),
    )
    return _load_cached_labeled_npz(cache_path, spec)


@torch.no_grad()
def load_cifar10_resnet18_512_labeled(
    data_dir: Path | str = DEFAULT_SEMDEDUP_DATA_DIR,
    cifar_dir: Path = DEFAULT_CIFAR10_ROOT,
    ckpt_path: Path = DEFAULT_RESNET18_CKPT,
    device: str | torch.device | None = None,
) -> LabeledVectorDataset:
    """Load `CIFAR-10 ResNet-18 512-D` (`~50K` train, `~10K` official test, `10` classes) with zero leakage."""
    ddir = Path(data_dir)
    spec = LABELED_SPEC_BY_NAME["cifar10_resnet18_512"]
    cache_path = ddir / spec.cache_filename
    if cache_path.exists():
        return _load_cached_labeled_npz(cache_path, spec)

    import torchvision.models as models

    tr_raw, y_tr_raw, te_raw, y_te_raw = _load_raw_cifar10_with_labels(cifar_dir)
    tr_imgs, y_tr_img, te_imgs, y_te_img = _deduplicate_raw_labeled_images_two_way(
        tr_raw, y_tr_raw, te_raw, y_te_raw
    )

    dev = (
        torch.device(device)
        if device is not None
        else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    )
    resolved_ckpt = ckpt_path
    if not resolved_ckpt.exists():
        for cand in DEFAULT_RESNET18_CANDIDATES:
            if cand.exists():
                resolved_ckpt = cand
                break
    if not resolved_ckpt.exists():
        raise FileNotFoundError(f"ResNet-18 checkpoint not found at {ckpt_path}")

    net = models.resnet18(weights=None)
    state = torch.load(resolved_ckpt, map_location="cpu", weights_only=True)
    net.load_state_dict(state)
    net.eval().to(dev)

    mean = torch.tensor([0.485, 0.456, 0.406], device=dev, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev, dtype=torch.float32).view(1, 3, 1, 1)

    def _extract_512(imgs: np.ndarray) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for s in range(0, imgs.shape[0], 512):
            batch = torch.from_numpy(imgs[s : s + 512]).to(dev, dtype=torch.float32) / 255.0
            inp = F.interpolate(batch, size=(96, 96), mode="bilinear", align_corners=False)
            inp = (inp - mean) / std
            h = net.conv1(inp)
            h = net.bn1(h)
            h = net.relu(h)
            h = net.maxpool(h)
            h = net.layer1(h)
            h = net.layer2(h)
            h = net.layer3(h)
            h = net.layer4(h)
            h = net.avgpool(h).flatten(1)
            chunks.append(h.cpu().numpy().astype(np.float32))
        return np.concatenate(chunks, axis=0)

    tr_f = _extract_512(tr_imgs)
    te_f = _extract_512(te_imgs)
    x_tr, y_tr, x_te, y_te = _deduplicate_labeled_two_way(tr_f, y_tr_img, te_f, y_te_img, decimals=5)

    atomic_savez(
        cache_path,
        x_train=x_tr,
        y_train=y_tr,
        x_test=x_te,
        y_test=y_te,
        n_train_rows=np.int64(x_tr.shape[0]),
        n_test_rows=np.int64(x_te.shape[0]),
    )
    return _load_cached_labeled_npz(cache_path, spec)


@torch.no_grad()
def load_cifar10_convnext_768_labeled(
    data_dir: Path | str = DEFAULT_SEMDEDUP_DATA_DIR,
    cifar_dir: Path = DEFAULT_CIFAR10_ROOT,
    device: str | torch.device | None = None,
) -> LabeledVectorDataset:
    """Load `CIFAR-10 ConvNeXt-Tiny 768-D` (`~50K` train, `~10K` official test, `10` classes) with zero leakage."""
    ddir = Path(data_dir)
    spec = LABELED_SPEC_BY_NAME["cifar10_convnext_768"]
    cache_path = ddir / spec.cache_filename
    if cache_path.exists():
        return _load_cached_labeled_npz(cache_path, spec)

    import torchvision.models as models

    ckpt_file = _resolve_verified_raw_file(
        CONVNEXT_TINY_FILENAME,
        (CONVNEXT_TINY_URL,),
        CONVNEXT_TINY_SIZE,
        CONVNEXT_TINY_SHA256,
        ddir,
    )
    tr_raw, y_tr_raw, te_raw, y_te_raw = _load_raw_cifar10_with_labels(cifar_dir)
    tr_imgs, y_tr_img, te_imgs, y_te_img = _deduplicate_raw_labeled_images_two_way(
        tr_raw, y_tr_raw, te_raw, y_te_raw
    )

    dev = (
        torch.device(device)
        if device is not None
        else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    )
    net = models.convnext_tiny(weights=None)
    state = torch.load(ckpt_file, map_location="cpu", weights_only=True)
    net.load_state_dict(state)
    net.eval().to(dev)

    mean = torch.tensor([0.485, 0.456, 0.406], device=dev, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev, dtype=torch.float32).view(1, 3, 1, 1)

    def _extract_768(imgs: np.ndarray) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for s in range(0, imgs.shape[0], 512):
            batch = torch.from_numpy(imgs[s : s + 512]).to(dev, dtype=torch.float32) / 255.0
            inp = F.interpolate(batch, size=(96, 96), mode="bilinear", align_corners=False)
            inp = (inp - mean) / std
            h = net.features(inp)
            h = net.avgpool(h)
            h = net.classifier[0](h)
            h = torch.flatten(h, 1)
            chunks.append(h.cpu().numpy().astype(np.float32))
        return np.concatenate(chunks, axis=0)

    tr_f = _extract_768(tr_imgs)
    te_f = _extract_768(te_imgs)
    x_tr, y_tr, x_te, y_te = _deduplicate_labeled_two_way(tr_f, y_tr_img, te_f, y_te_img, decimals=5)

    atomic_savez(
        cache_path,
        x_train=x_tr,
        y_train=y_tr,
        x_test=x_te,
        y_test=y_te,
        n_train_rows=np.int64(x_tr.shape[0]),
        n_test_rows=np.int64(x_te.shape[0]),
    )
    return _load_cached_labeled_npz(cache_path, spec)


def load_covtype_54_labeled(
    data_dir: Path | str = DEFAULT_SEMDEDUP_DATA_DIR,
    split_seed: int = 20261116,
    n_train: int = 500_000,
    n_test: int = 50_000,
) -> LabeledVectorDataset:
    """Load `Forest Covertype 54-D` (`500K` train, `50K` held-out test, `7` classes, `110,761x` span)."""
    import gzip

    ddir = Path(data_dir)
    spec = LABELED_SPEC_BY_NAME["covtype_54"]
    is_canon = (n_train == 500_000 and n_test == 50_000 and split_seed == 20261116)
    cache_path = (
        ddir / spec.cache_filename
        if is_canon
        else ddir / f"semdedup_covtype_54_s{int(split_seed)}_{n_train}_{n_test}_v1.npz"
    )
    if cache_path.exists():
        ds = _load_cached_labeled_npz(cache_path, spec)
        validate_covtype_physical_ranges(ds.x_train)
        return ds

    gz_path = _resolve_verified_raw_file(
        COVTYPE_FILENAME, COVTYPE_URLS, COVTYPE_SIZE, COVTYPE_SHA256, ddir
    )
    with gzip.open(gz_path, "rt", encoding="ascii") as f:
        raw_text = f.read()
    flat = np.fromstring(raw_text.replace("\n", ","), dtype=np.float64, sep=",")
    if flat.size != 581_012 * 55:
        raise ValueError(f"Unexpected covtype element count {flat.size} != {581_012 * 55}")
    mat = flat.reshape(581_012, 55)
    feats = np.ascontiguousarray(mat[:, :54], dtype=np.float32)
    labels = np.ascontiguousarray(mat[:, 54].astype(np.int64) - 1, dtype=np.int64)
    validate_covtype_physical_ranges(feats)
    if int(labels.min()) < 0 or int(labels.max()) >= 7:
        raise ValueError(f"Covertype class labels out of [0..6]: [{labels.min()}, {labels.max()}]")

    # Deduplicate exact feature rows before seeded train/test partition
    rkeys = _exact_row_keys(feats)
    _, uniq_idx = np.unique(rkeys, return_index=True)
    uniq_idx.sort()
    feats_u = np.ascontiguousarray(feats[uniq_idx], dtype=np.float32)
    labels_u = np.ascontiguousarray(labels[uniq_idx], dtype=np.int64)

    total_needed = int(n_train + n_test)
    if feats_u.shape[0] < total_needed:
        raise ValueError(f"Requested {total_needed} rows from {feats_u.shape[0]} unique Covertype rows")

    rng = np.random.default_rng(int(split_seed))
    perm = rng.permutation(feats_u.shape[0])
    tr_idx = perm[:n_train]
    te_idx = perm[n_train:total_needed]

    x_tr, y_tr, x_te, y_te = _deduplicate_labeled_two_way(
        feats_u[tr_idx], labels_u[tr_idx], feats_u[te_idx], labels_u[te_idx], decimals=6
    )
    if x_tr.shape != (n_train, 54) or x_te.shape != (n_test, 54):
        raise RuntimeError(f"Unexpected post-dedup Covertype shapes: {x_tr.shape}, {x_te.shape}")

    feat_stds = (x_tr.astype(np.float64) - x_tr.astype(np.float64).mean(axis=0, keepdims=True)).std(axis=0)
    nz_stds = feat_stds[feat_stds > 1e-12]
    std_span = float(nz_stds.max() / nz_stds.min())

    atomic_savez(
        cache_path,
        x_train=x_tr,
        y_train=y_tr,
        x_test=x_te,
        y_test=y_te,
        feature_std_span=np.float64(std_span),
        elevation_min=np.float64(x_tr[:, 0].min()),
        elevation_max=np.float64(x_tr[:, 0].max()),
    )
    ds = _load_cached_labeled_npz(cache_path, spec)
    validate_covtype_physical_ranges(ds.x_train)
    return ds


def load_labeled_benchmark(
    name: str,
    data_dir: Path | str = DEFAULT_SEMDEDUP_DATA_DIR,
    device: str | torch.device | None = None,
) -> LabeledVectorDataset:
    """Unified zero-leakage loader for all 4 real labeled ML benchmarks."""
    key = name.strip().lower()
    if key in ("agnews_256", "agnews", "agnews_tfidf_svd_256"):
        return load_agnews_256_labeled(data_dir=data_dir)
    if key in ("cifar10_resnet18_512", "cifar10_r18_512"):
        return load_cifar10_resnet18_512_labeled(data_dir=data_dir, device=device)
    if key == "cifar10_convnext_768":
        return load_cifar10_convnext_768_labeled(data_dir=data_dir, device=device)
    if key in ("covtype_54", "covtype", "covtype_multiscale_54"):
        return load_covtype_54_labeled(data_dir=data_dir)
    raise ValueError(f"Unknown labeled benchmark {name!r}; expected one of {list(LABELED_SPEC_BY_NAME)}")


# ==============================================================================
# 4. Megapixel High-Resolution RGB & `4x4x3` Natural Image Patch VQ Loaders
# ==============================================================================


def _load_cached_megapixel_npz(
    cache_path: Path,
    spec: MegapixelVQSpec,
) -> MegapixelVQDataset:
    with np.load(cache_path, allow_pickle=False) as z:
        meta: dict[str, Any] = {}
        for k in z.files:
            if k not in ("x_train", "x_test"):
                arr_k = z[k]
                if arr_k.ndim == 0:
                    meta[k] = arr_k.item()
        ds = MegapixelVQDataset(
            name=spec.name,
            short_label=spec.short_label,
            modality=spec.modality,
            x_train=np.ascontiguousarray(z["x_train"], dtype=np.float32),
            x_test=np.ascontiguousarray(z["x_test"], dtype=np.float32),
            d=spec.d,
            patch_h=spec.patch_h,
            patch_w=spec.patch_w,
            channels=spec.channels,
            train_mosaic_hw=spec.train_mosaic_hw,
            test_mosaic_hw=spec.test_mosaic_hw,
            k_values=spec.k_values,
            metadata=meta,
        )
    validate_megapixel_vq_dataset(ds, check_patch_disjoint=True)
    return ds


@torch.no_grad()
def _build_disjoint_natural_image_banks(
    cifar_dir: Path = DEFAULT_CIFAR10_ROOT,
    device: str | torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """Return strictly disjoint `(china_hr_train, flower_hr_test, cifar_train_u8, cifar_test_u8)`."""
    import sklearn.datasets as skd

    sample_imgs = skd.load_sample_images().images
    # Make explicit writable copies (`copy=True`) before `torch.from_numpy`
    china_np = np.array(sample_imgs[0], dtype=np.float32, copy=True)   # (427, 640, 3) train-only photo
    flower_np = np.array(sample_imgs[1], dtype=np.float32, copy=True)  # (427, 640, 3) test-only photo

    tr_raw, y_tr_raw, te_raw, y_te_raw = _load_raw_cifar10_with_labels(cifar_dir)
    tr_clean, _, te_clean, _ = _deduplicate_raw_labeled_images_two_way(tr_raw, y_tr_raw, te_raw, y_te_raw)

    dev = (
        torch.device(device)
        if device is not None
        else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    )
    china_t = torch.from_numpy(china_np).permute(2, 0, 1).unsqueeze(0).to(dev)
    flower_t = torch.from_numpy(flower_np).permute(2, 0, 1).unsqueeze(0).to(dev)
    return china_t, flower_t, tr_clean, te_clean


@torch.no_grad()
def load_megapixel_vq_benchmark(
    name: str,
    data_dir: Path | str = DEFAULT_SEMDEDUP_DATA_DIR,
    cifar_dir: Path = DEFAULT_CIFAR10_ROOT,
    split_seed: int = 20261116,
    device: str | torch.device | None = None,
) -> MegapixelVQDataset:
    """Build or load deterministic high-resolution Megapixel RGB (`1M x 3`) or Patch (`500K x 48`) VQ corpus."""
    key = name.strip().lower()
    if key not in MEGAPIXEL_SPEC_BY_NAME:
        raise ValueError(f"Unknown Megapixel VQ benchmark {name!r}; expected {list(MEGAPIXEL_SPEC_BY_NAME)}")

    spec = MEGAPIXEL_SPEC_BY_NAME[key]
    ddir = Path(data_dir)
    cache_path = ddir / spec.cache_filename
    if cache_path.exists():
        return _load_cached_megapixel_npz(cache_path, spec)

    china_t, flower_t, tr_imgs, te_imgs = _build_disjoint_natural_image_banks(cifar_dir=cifar_dir, device=device)
    dev = china_t.device
    rng = np.random.default_rng(int(split_seed))
    tr_perm = rng.permutation(tr_imgs.shape[0])
    te_perm = rng.permutation(te_imgs.shape[0])

    if key == "megapixel_rgb_3d":
        # Train 1,000,000 x 3 RGB mosaic (1000 x 1000): 500K high-res `china.jpg` + 500K upsampled/native CIFAR train
        hr_tr = F.interpolate(china_t, size=(500, 1000), mode="bicubic", align_corners=False).clamp_(0.0, 255.0)[0]
        pix_hr_tr = hr_tr.permute(1, 2, 0).reshape(-1, 3)  # (500_000, 3)
        c10_tr = torch.from_numpy(tr_imgs[tr_perm[:600]]).to(dev, dtype=torch.float32)
        c10_tr_up = F.interpolate(c10_tr[:128], size=(64, 64), mode="bicubic", align_corners=False).clamp_(0.0, 255.0)
        pix_c10_tr = torch.cat(
            [
                c10_tr_up.permute(0, 2, 3, 1).reshape(-1, 3),
                c10_tr[128:].permute(0, 2, 3, 1).reshape(-1, 3),
            ],
            dim=0,
        )[:500_000]
        x_tr = torch.cat([pix_hr_tr, pix_c10_tr], dim=0)[: spec.n_train].cpu().numpy().astype(np.float32)

        # Held-out Test 262,144 x 3 RGB mosaic (512 x 512): strictly from `flower.jpg` + CIFAR official test!
        hr_te = F.interpolate(flower_t, size=(256, 512), mode="bicubic", align_corners=False).clamp_(0.0, 255.0)[0]
        pix_hr_te = hr_te.permute(1, 2, 0).reshape(-1, 3)  # (131_072, 3)
        c10_te = torch.from_numpy(te_imgs[te_perm[:200]]).to(dev, dtype=torch.float32)
        c10_te_up = F.interpolate(c10_te[:32], size=(64, 64), mode="bicubic", align_corners=False).clamp_(0.0, 255.0)
        pix_c10_te = torch.cat(
            [
                c10_te_up.permute(0, 2, 3, 1).reshape(-1, 3),
                c10_te[32:].permute(0, 2, 3, 1).reshape(-1, 3),
            ],
            dim=0,
        )[:131_072]
        x_te = torch.cat([pix_hr_te, pix_c10_te], dim=0)[: spec.n_test].cpu().numpy().astype(np.float32)

    else:
        # `megapixel_patch_48d`: 500,000 x 48 (4x4x3) train patches + 100,000 x 48 disjoint held-out test patches
        def _img_to_4x4_patches(img_chw: torch.Tensor) -> torch.Tensor:
            # img_chw: (3, H, W) with H, W divisible by 4 -> (H//4 * W//4, 48) in (ph, pw, c) layout
            p = img_chw.unfold(1, 4, 4).unfold(2, 4, 4)  # (3, H//4, W//4, 4, 4)
            return p.permute(1, 2, 3, 4, 0).reshape(-1, 48)

        tr_patches_list: list[torch.Tensor] = []
        for h_s, w_s in ((424, 640), (848, 1280), (1272, 1920)):
            im_s = F.interpolate(china_t, size=(h_s, w_s), mode="bicubic", align_corners=False).clamp_(0.0, 255.0)[0]
            tr_patches_list.append(_img_to_4x4_patches(im_s))

        # Add native 32x32 + 64x64 bicubic CIFAR-10 train images (strictly train split!)
        c10_tr_a = torch.from_numpy(tr_imgs[tr_perm[:3000]]).to(dev, dtype=torch.float32)
        c10_tr_a64 = F.interpolate(c10_tr_a[:1200], size=(64, 64), mode="bicubic", align_corners=False).clamp_(0.0, 255.0)
        p_a64 = c10_tr_a64.unfold(2, 4, 4).unfold(3, 4, 4).permute(0, 2, 3, 4, 5, 1).reshape(-1, 48)
        p_a32 = c10_tr_a[1200:].unfold(2, 4, 4).unfold(3, 4, 4).permute(0, 2, 3, 4, 5, 1).reshape(-1, 48)
        tr_patches_list.extend([p_a64, p_a32])
        raw_tr_p = torch.cat(tr_patches_list, dim=0).cpu().numpy().astype(np.float32)

        te_patches_list: list[torch.Tensor] = []
        for h_s, w_s in ((424, 640), (636, 960)):
            im_s = F.interpolate(flower_t, size=(h_s, w_s), mode="bicubic", align_corners=False).clamp_(0.0, 255.0)[0]
            te_patches_list.append(_img_to_4x4_patches(im_s))
        c10_te_a = torch.from_numpy(te_imgs[te_perm[:1500]]).to(dev, dtype=torch.float32)
        c10_te_a64 = F.interpolate(c10_te_a[:300], size=(64, 64), mode="bicubic", align_corners=False).clamp_(0.0, 255.0)
        p_te64 = c10_te_a64.unfold(2, 4, 4).unfold(3, 4, 4).permute(0, 2, 3, 4, 5, 1).reshape(-1, 48)
        p_te32 = c10_te_a[300:].unfold(2, 4, 4).unfold(3, 4, 4).permute(0, 2, 3, 4, 5, 1).reshape(-1, 48)
        te_patches_list.extend([p_te64, p_te32])
        raw_te_p = torch.cat(te_patches_list, dim=0).cpu().numpy().astype(np.float32)

        # Enforce zero exact 48-D patch overlap across train -> test
        ktr = _exact_row_keys(raw_tr_p)
        _, utr = np.unique(ktr, return_index=True)
        utr.sort()
        clean_tr_p = np.ascontiguousarray(raw_tr_p[utr[: spec.n_train]], dtype=np.float32)
        seen_ktr = _exact_row_keys(clean_tr_p)

        kte = _exact_row_keys(raw_te_p)
        _, ute = np.unique(kte, return_index=True)
        ute.sort()
        keep_te = ute[~np.isin(kte[ute], seen_ktr)]
        clean_te_p = np.ascontiguousarray(raw_te_p[keep_te[: spec.n_test]], dtype=np.float32)

        if clean_tr_p.shape != (spec.n_train, 48) or clean_te_p.shape != (spec.n_test, 48):
            raise RuntimeError(
                f"Insufficient unique 4x4 patches: got {clean_tr_p.shape} / {clean_te_p.shape}"
            )
        x_tr, x_te = clean_tr_p, clean_te_p

    atomic_savez(
        cache_path,
        x_train=np.ascontiguousarray(x_tr, dtype=np.float32),
        x_test=np.ascontiguousarray(x_te, dtype=np.float32),
        n_train_rows=np.int64(x_tr.shape[0]),
        n_test_rows=np.int64(x_te.shape[0]),
    )
    return _load_cached_megapixel_npz(cache_path, spec)


# ==============================================================================
# 5. Fused Zero-Loop GPU `SemDeDup` & Coreset Selection Engine
# ==============================================================================

if HAS_TRITON:

    @triton.jit(do_not_specialize=["N", "D", "stride_xn"])
    def _semdedup_margin_kernel(
        x_ptr,
        sorted_idx_ptr,
        sorted_lbl_ptr,
        starts_ptr,
        counts_ptr,
        c_rms_ptr,
        out_score_ptr,
        N,
        D,
        stride_xn,
        W_WIN: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        FULL_D: tl.constexpr,
    ):
        """Single-pass intra-cluster pairwise SemDeDup uniqueness + medoid anchor kernel (`0` atomics, 64-bit ptrs)."""
        N_64 = N.to(tl.int64)
        D_64 = D.to(tl.int64)
        stride_xn_64 = stride_xn.to(tl.int64)
        pid = tl.program_id(0).to(tl.int64)
        offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
        mask_n = offs_n < N_64

        c_id = tl.load(sorted_lbl_ptr + offs_n, mask=mask_n, other=0).to(tl.int64)
        start_c = tl.load(starts_ptr + c_id, mask=mask_n, other=0).to(tl.int64)
        n_c = tl.load(counts_ptr + c_id, mask=mask_n, other=1).to(tl.int64)
        intra = offs_n - start_c
        is_medoid = mask_n & (intra == 0)
        active_nonmed = mask_n & (intra > 0)

        min_d2 = tl.full((BLOCK_N,), float("inf"), dtype=tl.float32)

        if FULL_D:
            cols = tl.arange(0, BLOCK_D).to(tl.int64)
            mask_d = cols < D_64
            self_orig = tl.load(sorted_idx_ptr + offs_n, mask=mask_n, other=0).to(tl.int64)
            x_self = tl.load(
                x_ptr + self_orig[:, None] * stride_xn_64 + cols[None, :],
                mask=mask_n[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)

            med_orig = tl.load(sorted_idx_ptr + start_c, mask=active_nonmed, other=0).to(tl.int64)
            x_med = tl.load(
                x_ptr + med_orig[:, None] * stride_xn_64 + cols[None, :],
                mask=active_nonmed[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)
            diff_m = tl.where(mask_d[None, :], x_self - x_med, 0.0)
            min_d2 = tl.where(active_nonmed, tl.sum(diff_m * diff_m, axis=1), min_d2)

            for lag in tl.static_range(1, W_WIN + 1):
                valid_lag = active_nonmed & (intra >= lag)
                nbr_pos = tl.where(valid_lag, offs_n - lag, tl.zeros((), dtype=tl.int64))
                nbr_orig = tl.load(sorted_idx_ptr + nbr_pos, mask=valid_lag, other=0).to(tl.int64)
                x_nbr = tl.load(
                    x_ptr + nbr_orig[:, None] * stride_xn_64 + cols[None, :],
                    mask=valid_lag[:, None] & mask_d[None, :],
                    other=0.0,
                ).to(tl.float32)
                diff = tl.where(mask_d[None, :], x_self - x_nbr, 0.0)
                min_d2 = tl.where(valid_lag, tl.minimum(min_d2, tl.sum(diff * diff, axis=1)), min_d2)
        else:
            self_orig = tl.load(sorted_idx_ptr + offs_n, mask=mask_n, other=0).to(tl.int64)
            med_orig = tl.load(sorted_idx_ptr + start_c, mask=active_nonmed, other=0).to(tl.int64)
            d2_m = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for d0 in range(0, D_64, BLOCK_D):
                cols = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
                mask_d = cols < D_64
                xs = tl.load(
                    x_ptr + self_orig[:, None] * stride_xn_64 + cols[None, :],
                    mask=active_nonmed[:, None] & mask_d[None, :],
                    other=0.0,
                ).to(tl.float32)
                xm = tl.load(
                    x_ptr + med_orig[:, None] * stride_xn_64 + cols[None, :],
                    mask=active_nonmed[:, None] & mask_d[None, :],
                    other=0.0,
                ).to(tl.float32)
                df = tl.where(mask_d[None, :], xs - xm, 0.0)
                d2_m += tl.sum(df * df, axis=1)
            min_d2 = tl.where(active_nonmed, d2_m, min_d2)

            for lag in range(1, W_WIN + 1):
                valid_lag = active_nonmed & (intra >= lag)
                nbr_pos = tl.where(valid_lag, offs_n - lag, tl.zeros((), dtype=tl.int64))
                nbr_orig = tl.load(sorted_idx_ptr + nbr_pos, mask=valid_lag, other=0).to(tl.int64)
                d2 = tl.zeros((BLOCK_N,), dtype=tl.float32)
                for d0 in range(0, D_64, BLOCK_D):
                    cols = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
                    mask_d = cols < D_64
                    xs = tl.load(
                        x_ptr + self_orig[:, None] * stride_xn_64 + cols[None, :],
                        mask=valid_lag[:, None] & mask_d[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    xn = tl.load(
                        x_ptr + nbr_orig[:, None] * stride_xn_64 + cols[None, :],
                        mask=valid_lag[:, None] & mask_d[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    df = tl.where(mask_d[None, :], xs - xn, 0.0)
                    d2 += tl.sum(df * df, axis=1)
                min_d2 = tl.where(valid_lag, tl.minimum(min_d2, d2), min_d2)

        rms = tl.load(c_rms_ptr + c_id, mask=mask_n, other=1.0).to(tl.float32)
        norm_d = tl.sqrt(tl.maximum(min_d2, 0.0)) / tl.maximum(rms, 1e-12)
        is_tail_outlier = (n_c >= 16) & (intra.to(tl.float32) > 0.97 * n_c.to(tl.float32))
        nonmed_score = tl.where(is_tail_outlier, 0.25 * norm_d, norm_d)
        med_score = 1.0e20 + tl.log(n_c.to(tl.float32) + 1.0)
        score = tl.where(is_medoid, med_score, nonmed_score)
        tl.store(out_score_ptr + offs_n, score, mask=mask_n)


@dataclass
class SemDeDupScoreState:
    """Reusable GPU state computed once per learned clustering for multi-budget coreset extraction."""

    n_train: int
    k: int
    empty_clusters: int
    counts: torch.Tensor       # (K,) int64
    sorted_idx: torch.Tensor   # (N,) int64 indices into x_train
    sorted_lbl: torch.Tensor   # (N,) int64 cluster IDs in sorted order
    intra_rank: torch.Tensor   # (N,) int64 rank within cluster (0 == exact medoid)
    scores: torch.Tensor       # (N,) float32 SemDeDup priority scores in sorted order
    score_time_sec: float


@dataclass
class CoresetSelection:
    """Selected training subset `(indices, weights)` with exact Horvitz-Thompson mass conservation."""

    indices: torch.Tensor       # (M,) int64 indices into x_train
    weights: torch.Tensor       # (M,) float32 importance weights
    cluster_ids: torch.Tensor   # (M,) int64 cluster label per selected point
    num_selected: int
    active_clusters_covered: int
    select_time_sec: float


def _compute_equivariant_origin64(x: torch.Tensor) -> torch.Tensor:
    """Odd-sample strided FP64 coordinate median (`100%` translation-equivariant & `<50%` outlier-immune)."""
    n = int(x.shape[0])
    if n <= 0:
        return torch.zeros((1, int(x.shape[1])), dtype=torch.float64, device=x.device)
    if n <= 4095:
        n_med = n if (n % 2 == 1 or n == 1) else (n - 1)
        sub_raw = x[:n_med]
    else:
        step = max(1, n // 4095)
        sub_raw = x[::step][:4095]
        if int(sub_raw.shape[0]) > 1 and (int(sub_raw.shape[0]) % 2 == 0):
            sub_raw = sub_raw[:-1]
    sub = torch.nan_to_num(sub_raw.double(), nan=0.0, posinf=1.0e150, neginf=-1.0e150)
    return torch.median(sub, dim=0, keepdim=True).values


def _condition_centered_pair(
    x: torch.Tensor,
    y: torch.Tensor,
    origin64: torch.Tensor,
    scale_from_x_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Origin-center `(x, y)` in FP64 and apply power-of-two scale conditioning (`1e-300..1e+300` safe)."""
    xd = torch.nan_to_num(x.double(), nan=0.0, posinf=1.0e150, neginf=-1.0e150)
    yd = torch.nan_to_num(y.double(), nan=0.0, posinf=1.0e150, neginf=-1.0e150)
    orig = torch.nan_to_num(origin64.double(), nan=0.0, posinf=0.0, neginf=0.0)
    xc64 = xd - orig
    yc64 = yd - orig
    ax = xc64.abs().amax() if xc64.numel() > 0 else torch.tensor(0.0, device=x.device, dtype=torch.float64)
    if scale_from_x_only:
        sample_max = float(ax.item())
    else:
        ay = yc64.abs().amax() if yc64.numel() > 0 else torch.tensor(0.0, device=x.device, dtype=torch.float64)
        sample_max = float(torch.maximum(ax, ay).item())
    if not math.isfinite(sample_max) or sample_max <= 0.0:
        coord_scale = 1.0
    elif sample_max > 1.0e4 or sample_max < 1.0e-4:
        exp2 = max(-950, min(950, int(round(math.log2(sample_max)))))
        coord_scale = float(2.0 ** (-exp2))
    else:
        coord_scale = 1.0
    # Clamp at +-1.0e16 so squared L2 norms up to D=2048 stay <= 2.048e35 << float32 max (3.40e38)
    if coord_scale != 1.0:
        xc32 = (xc64 * coord_scale).clamp(-1.0e16, 1.0e16).float().contiguous()
        yc32 = (yc64 * coord_scale).clamp(-1.0e16, 1.0e16).float().contiguous()
    else:
        xc32 = xc64.clamp(-1.0e16, 1.0e16).float().contiguous()
        yc32 = yc64.clamp(-1.0e16, 1.0e16).float().contiguous()
    return xc32, yc32, coord_scale


@torch.no_grad()
def centered_assign_fp32(
    x: torch.Tensor,
    centroids: torch.Tensor,
    origin: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Translation-equivariant IEEE-FP32 Voronoi assignment (`100%` invariant to `+20,000` shifts).

    When `origin` is provided (out-of-sample query/VQ encoding against fixed `centroids`),
    `coord_scale` is derived strictly from `centroids` (`scale_from_x_only=True`) for `0` query coupling.
    """
    if origin is None:
        origin64 = _compute_equivariant_origin64(x)
        xc32, cc32, coord_scale = _condition_centered_pair(x, centroids, origin64, scale_from_x_only=False)
    else:
        origin64 = origin.double().reshape(1, -1)
        cc32, xc32, coord_scale = _condition_centered_pair(centroids, x, origin64, scale_from_x_only=True)
    if xc32.is_cuda:
        with torch.cuda.device(xc32.device):
            sq_d, lbl = assign_points(xc32, cc32, compute_dtype=torch.float32)
    else:
        sq_d, lbl = assign_points(xc32, cc32, compute_dtype=torch.float32)
    if coord_scale != 1.0:
        inv_sq = 1.0 / max(coord_scale * coord_scale, 1.0e-300)
        sq_d = (sq_d.double() * inv_sq).clamp_max(torch.finfo(torch.float32).max).float()
    return sq_d, lbl


def _semdedup_scores_torch_fallback(
    x_c: torch.Tensor,
    sorted_idx: torch.Tensor,
    sorted_lbl: torch.Tensor,
    starts: torch.Tensor,
    counts: torch.Tensor,
    c_rms: torch.Tensor,
    window_size: int = 24,
) -> torch.Tensor:
    """Exact vectorized PyTorch reference for `_semdedup_margin_kernel` (`0` loops over `K` or `N`)."""
    n = x_c.shape[0]
    dev = x_c.device
    pos = torch.arange(n, device=dev, dtype=torch.int64)
    intra = pos - starts[sorted_lbl]
    xs = x_c[sorted_idx]
    x_med = x_c[sorted_idx[starts[sorted_lbl]]]
    min_d2 = (xs - x_med).square().sum(dim=1)
    for lag in range(1, int(window_size) + 1):
        valid = intra >= lag
        prev = x_c[sorted_idx[(pos - lag).clamp_min(0)]]
        d2 = (xs - prev).square().sum(dim=1)
        min_d2 = torch.where(valid, torch.minimum(min_d2, d2), min_d2)
    n_c = counts[sorted_lbl]
    norm_d = torch.sqrt(min_d2.clamp_min(0.0)) / c_rms[sorted_lbl].clamp_min(1e-12)
    is_tail = (n_c >= 16) & (intra.float() > 0.97 * n_c.float())
    nonmed = torch.where(is_tail, 0.25 * norm_d, norm_d)
    med_s = 1.0e20 + torch.log(n_c.float() + 1.0)
    return torch.where(intra == 0, med_s, nonmed)


@torch.no_grad()
def compute_semdedup_scores_gpu(
    x_train: torch.Tensor | np.ndarray,
    centroids: torch.Tensor | np.ndarray,
    k: int,
    assign_labels: torch.Tensor | None = None,
    assign_sq_dists: torch.Tensor | None = None,
    window_size: int = 24,
) -> SemDeDupScoreState:
    """Compute reusable GPU `SemDeDup` intra-cluster priority scores (`0` Python loops over `K` or `N`)."""
    x_t = torch.from_numpy(x_train) if isinstance(x_train, np.ndarray) else x_train
    c_t = torch.from_numpy(centroids) if isinstance(centroids, np.ndarray) else centroids
    n, d = int(x_t.shape[0]), int(x_t.shape[1])
    if n < 1 or k < 1:
        raise ValueError(f"Expected n >= 1 and k >= 1, got n={n}, k={k}")
    if int(c_t.shape[0]) != k or int(c_t.shape[1]) != d:
        raise ValueError(f"Centroids shape {tuple(c_t.shape)} does not match (k={k}, d={d})")

    dev = x_t.device
    c_t = c_t.to(device=dev)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t0 = time.perf_counter()

    origin64 = _compute_equivariant_origin64(x_t)
    x_c, c_c, coord_scale = _condition_centered_pair(x_t, c_t, origin64)

    if assign_labels is not None:
        lbl_in = torch.from_numpy(assign_labels) if isinstance(assign_labels, np.ndarray) else assign_labels
        if lbl_in.ndim != 1 or int(lbl_in.shape[0]) != n:
            raise ValueError(f"Expected assign_labels shape ({n},), got {tuple(lbl_in.shape)}")
        lbl = lbl_in.to(device=dev, dtype=torch.int64).contiguous()
        if bool(((lbl < 0) | (lbl >= k)).any().item()):
            raise ValueError(f"assign_labels out of bounds for k={k}")
        if assign_sq_dists is not None:
            sq_in = torch.from_numpy(assign_sq_dists) if isinstance(assign_sq_dists, np.ndarray) else assign_sq_dists
            if sq_in.ndim != 1 or int(sq_in.shape[0]) != n:
                raise ValueError(f"Expected assign_sq_dists shape ({n},), got {tuple(sq_in.shape)}")
            sq_d = (
                torch.nan_to_num(sq_in.to(device=dev, dtype=torch.float64), nan=0.0, posinf=1.0e300, neginf=0.0)
                .clamp_min(0.0)
                .mul(coord_scale * coord_scale)
                .clamp_max(torch.finfo(torch.float32).max)
                .float()
                .contiguous()
            )
        else:
            sq_d = (x_c - c_c[lbl]).square().sum(dim=1)
    elif assign_sq_dists is not None:
        raise ValueError("assign_sq_dists requires assign_labels to be provided")
    elif dev.type == "cuda":
        with torch.cuda.device(dev):
            sq_d, lbl = assign_points(x_c, c_c, compute_dtype=torch.float32)
    else:
        sq_d, lbl = assign_points(x_c, c_c, compute_dtype=torch.float32)

    counts = torch.bincount(lbl, minlength=k)
    empty_clusters = int((counts == 0).sum().item())

    # Lexicographic GPU radix sort by (cluster_label, sq_dist_to_centroid)
    order_d = torch.argsort(sq_d, stable=True)
    order_cd = torch.argsort(lbl[order_d], stable=True)
    sorted_idx = order_d[order_cd]
    sorted_lbl = lbl[sorted_idx]
    starts = torch.cumsum(counts, dim=0) - counts
    intra_rank = torch.arange(n, device=dev, dtype=torch.int64) - starts[sorted_lbl]

    c_sse = torch.zeros((k,), device=dev, dtype=torch.float64)
    c_sse.index_add_(0, lbl, sq_d.double())
    c_rms = torch.sqrt((c_sse / counts.clamp_min(1).double()).clamp_min(1e-24)).float()

    if dev.type == "cuda" and HAS_TRITON and window_size == 24:
        scores = torch.empty((n,), device=dev, dtype=torch.float32)
        blk_d_pow2 = triton.next_power_of_2(max(d, 16))
        full_d = blk_d_pow2 <= 1024
        blk_d = blk_d_pow2 if full_d else 128
        blk_n = 16 if (full_d and blk_d >= 512) else 64
        grid = (triton.cdiv(n, blk_n),)
        with torch.cuda.device(dev):
            _semdedup_margin_kernel[grid](
                x_c,
                sorted_idx,
                sorted_lbl,
                starts,
                counts,
                c_rms,
                scores,
                n,
                d,
                x_c.stride(0),
                W_WIN=24,
                BLOCK_N=blk_n,
                BLOCK_D=blk_d,
                FULL_D=full_d,
                num_warps=4,
            )
    else:
        scores = _semdedup_scores_torch_fallback(
            x_c, sorted_idx, sorted_lbl, starts, counts, c_rms, window_size=window_size
        )

    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    elapsed = time.perf_counter() - t0

    return SemDeDupScoreState(
        n_train=n,
        k=k,
        empty_clusters=empty_clusters,
        counts=counts,
        sorted_idx=sorted_idx,
        sorted_lbl=sorted_lbl,
        intra_rank=intra_rank,
        scores=scores,
        score_time_sec=elapsed,
    )


@torch.no_grad()
def select_coreset_at_budget(
    state: SemDeDupScoreState,
    retention_frac: float | None = None,
    budget_m: int | None = None,
) -> CoresetSelection:
    """Extract an exact `M`-sample Horvitz-Thompson importance-weighted coreset from `state`."""
    if budget_m is None:
        if retention_frac is None:
            raise ValueError("Provide either retention_frac or budget_m")
        if not (0.0 < float(retention_frac) <= 1.0):
            raise ValueError(f"retention_frac must be in (0, 1], got {retention_frac}")
        m = max(1, min(state.n_train, int(round(float(retention_frac) * state.n_train))))
    else:
        m = max(1, min(state.n_train, int(budget_m)))

    dev = state.scores.device
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t0 = time.perf_counter()

    top_pos = torch.topk(state.scores, k=m, largest=True, sorted=False).indices
    top_pos_sorted, _ = torch.sort(top_pos)  # deterministic ordering
    sel_idx = state.sorted_idx[top_pos_sorted]
    sel_lbl = state.sorted_lbl[top_pos_sorted]

    sel_counts = torch.bincount(sel_lbl, minlength=state.k)
    active_cov = int((sel_counts > 0).sum().item())
    raw_w = state.counts[sel_lbl].double() / sel_counts[sel_lbl].clamp_min(1).double()
    w_sum = float(raw_w.sum().item())
    # Normalize total Horvitz-Thompson weight to exact `n_train` even when `M < K_active`
    norm_w = (raw_w * (float(state.n_train) / max(w_sum, 1e-12))).float()

    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    dt = time.perf_counter() - t0

    return CoresetSelection(
        indices=sel_idx,
        weights=norm_w,
        cluster_ids=sel_lbl,
        num_selected=m,
        active_clusters_covered=active_cov,
        select_time_sec=dt,
    )


@torch.no_grad()
def select_prototype_coreset(state: SemDeDupScoreState) -> CoresetSelection:
    """Select the exact 1-medoid-per-active-cluster prototype coreset (`M <= K`)."""
    dev = state.scores.device
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t0 = time.perf_counter()

    med_mask = (state.counts[state.sorted_lbl] > 0) & (state.intra_rank == 0)
    sel_idx = state.sorted_idx[med_mask]
    sel_lbl = state.sorted_lbl[med_mask]
    sel_w = state.counts[sel_lbl].float()

    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    dt = time.perf_counter() - t0

    return CoresetSelection(
        indices=sel_idx,
        weights=sel_w,
        cluster_ids=sel_lbl,
        num_selected=int(sel_idx.numel()),
        active_clusters_covered=int(sel_idx.numel()),
        select_time_sec=dt,
    )


@torch.no_grad()
def select_random_coreset(
    n_train: int,
    retention_frac: float | None = None,
    budget_m: int | None = None,
    seed: int = 42,
    device: torch.device | str = "cpu",
) -> CoresetSelection:
    """Uniform random pruning baseline (`w_i = N / M`) with isolated `torch.Generator`."""
    dev = torch.device(device)
    if budget_m is None:
        if retention_frac is None:
            raise ValueError("Provide retention_frac or budget_m")
        m = max(1, min(n_train, int(round(float(retention_frac) * n_train))))
    else:
        m = max(1, min(n_train, int(budget_m)))

    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t0 = time.perf_counter()

    gen = torch.Generator(device=dev)
    gen.manual_seed(int(seed) & 0xFFFFFFFF)
    perm = torch.randperm(n_train, generator=gen, device=dev)[:m]
    idx, _ = torch.sort(perm)
    w = torch.full((m,), float(n_train) / float(m), device=dev, dtype=torch.float32)
    c_ids = torch.arange(m, device=dev, dtype=torch.int64)

    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    dt = time.perf_counter() - t0
    return CoresetSelection(
        indices=idx,
        weights=w,
        cluster_ids=c_ids,
        num_selected=m,
        active_clusters_covered=m,
        select_time_sec=dt,
    )


# ==============================================================================
# 6. Zero-Leakage GPU Downstream Classifiers (Strict Coreset-Only Fitting)
# ==============================================================================


def _standardize_and_expand_coreset_features(
    x_sub: torch.Tensor,
    w_sub: torch.Tensor,
    x_eval: torch.Tensor,
    expand_poly_low_d: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standardize strictly using `(x_sub, w_sub)` in IEEE FP64 (`0` test leakage, `+20,000` shift-invariant).

    When `expand_poly_low_d=True` and `d <= 64` (e.g., `54-D` physical sensor tabular data),
    appends deterministic degree-2 interactions on up to the first 10 columns.
    """
    w = torch.nan_to_num(w_sub.double(), nan=1.0, posinf=1.0, neginf=0.0).clamp_min(0.0)
    w_sum = w.sum()
    if float(w_sum.item()) <= 0.0:
        w = torch.ones_like(w)
        w_sum = w.sum()
    w_norm = w / w_sum

    xd_sub = torch.nan_to_num(x_sub.double(), nan=0.0, posinf=1.0e150, neginf=-1.0e150)
    xd_ev = torch.nan_to_num(x_eval.double(), nan=0.0, posinf=1.0e150, neginf=-1.0e150)
    # Odd-sample median anchor first so +20,000 shift is Sterbenz-exact before weighted mean!
    orig = _compute_equivariant_origin64(xd_sub)
    xd_sub_0 = xd_sub - orig
    xd_ev_0 = xd_ev - orig
    amax = float(xd_sub_0.abs().amax().item()) if xd_sub_0.numel() > 0 else 0.0
    if amax > 1.0e6 or (0.0 < amax < 1.0e-6):
        s_pow2 = float(2.0 ** (-max(-950, min(950, int(round(math.log2(amax)))))))
        xd_sub_0 = xd_sub_0 * s_pow2
        xd_ev_0 = xd_ev_0 * s_pow2

    mu = (w_norm[:, None] * xd_sub_0).sum(dim=0, keepdim=True)
    xc_sub = xd_sub_0 - mu
    xc_ev = xd_ev_0 - mu
    var = (w_norm[:, None] * xc_sub.square()).sum(dim=0, keepdim=True)
    valid_col = var > 1e-24
    inv_std = torch.where(valid_col, torch.rsqrt(var.clamp_min(1e-24)), torch.zeros_like(var))
    zs_sub = (xc_sub * inv_std).clamp(-50.0, 50.0)
    zs_ev = (xc_ev * inv_std).clamp(-50.0, 50.0)

    d = int(x_sub.shape[1])
    if expand_poly_low_d and 4 <= d <= 64:
        k_poly = min(d, 10)
        z_p_sub = zs_sub[:, :k_poly].clamp(-8.0, 8.0)
        z_p_ev = zs_ev[:, :k_poly].clamp(-8.0, 8.0)
        tri_i, tri_j = torch.triu_indices(k_poly, k_poly, device=x_sub.device)
        quad_sub = z_p_sub[:, tri_i] * z_p_sub[:, tri_j]
        quad_ev = z_p_ev[:, tri_i] * z_p_ev[:, tri_j]
        q_mu = (w_norm[:, None] * quad_sub).sum(dim=0, keepdim=True)
        q_var = (w_norm[:, None] * (quad_sub - q_mu).square()).sum(dim=0, keepdim=True)
        q_inv = torch.where(q_var > 1e-24, torch.rsqrt(q_var.clamp_min(1e-24)), torch.zeros_like(q_var))
        zs_sub = torch.cat([zs_sub, ((quad_sub - q_mu) * q_inv).clamp(-50.0, 50.0)], dim=1)
        zs_ev = torch.cat([zs_ev, ((quad_ev - q_mu) * q_inv).clamp(-50.0, 50.0)], dim=1)

    return zs_sub, zs_ev, w_norm


def _solve_gcv_cholesky_ridge_fp64(
    zs_sub: torch.Tensor,
    y_sub: torch.Tensor,
    w_norm: torch.Tensor,
    num_classes: int,
    lambdas: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0, 5.0, 20.0),
) -> tuple[torch.Tensor, float]:
    """Closed-form Golub-Heath-Wahba Leave-One-Out GCV Cholesky Ridge solver in IEEE FP64."""
    dev = zs_sub.device
    m = zs_sub.shape[0]
    ones = torch.ones((m, 1), dtype=torch.float64, device=dev)
    phi = torch.cat([zs_sub, ones], dim=1)  # (M, P)
    sw = torch.sqrt(w_norm)[:, None]
    phi_w = phi * sw
    y_oh = F.one_hot(y_sub.to(torch.int64), num_classes=num_classes).double() * sw

    gram = phi_w.T @ phi_w  # (P, P)
    rhs = phi_w.T @ y_oh    # (P, C)
    evals, evecs = torch.linalg.eigh(gram)
    evals = evals.clamp_min(0.0)
    qt_rhs = evecs.T @ rhs
    y_norm_sq = float((y_oh * y_oh).sum().item())
    m_eff = float((1.0 / (w_norm * w_norm).sum().clamp_min(1e-12)).item())

    best_gcv = float("inf")
    best_lam = 1e-2
    for lam in lambdas:
        denom = evals + lam
        fit_f = (evals * (evals + 2.0 * lam)) / (denom * denom)
        rss = max(y_norm_sq - float((fit_f[:, None] * qt_rhs.square()).sum().item()), 1e-15)
        df = float((evals / denom).sum().item())
        if m_eff > 4.0 and df >= 0.96 * m_eff:
            continue
        denom_gcv = max(1.0 - df / max(m_eff, df + 1.0), 1e-3)
        gcv = rss / (denom_gcv * denom_gcv)
        if gcv < best_gcv:
            best_gcv = gcv
            best_lam = float(lam)

    g_reg = gram.clone()
    reg = torch.full((g_reg.shape[0],), best_lam, dtype=torch.float64, device=dev)
    reg[-1] = 1e-9  # minimal intercept regularization for full rank
    g_reg.diagonal().add_(reg)
    L, info = torch.linalg.cholesky_ex(g_reg)
    if int(info.item()) == 0:
        w_mat = torch.cholesky_solve(rhs, L)
    else:
        w_mat = torch.linalg.lstsq(g_reg, rhs).solution
    return w_mat, best_lam


@torch.no_grad()
def evaluate_cholesky_ridge_gpu(
    x_sub: torch.Tensor,
    y_sub: torch.Tensor,
    w_sub: torch.Tensor,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    num_classes: int,
    expand_poly_low_d: bool = True,
) -> dict[str, float]:
    """Train GCV-calibrated Cholesky Ridge strictly on `(x_sub, y_sub, w_sub)` and score `x_eval`."""
    dev = x_sub.device
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t0 = time.perf_counter()

    zs_sub, zs_ev, w_norm = _standardize_and_expand_coreset_features(
        x_sub, w_sub, x_eval, expand_poly_low_d=expand_poly_low_d
    )
    w_mat, best_lam = _solve_gcv_cholesky_ridge_fp64(zs_sub, y_sub, w_norm, num_classes=num_classes)

    phi_ev = torch.cat(
        [zs_ev, torch.ones((zs_ev.shape[0], 1), dtype=torch.float64, device=dev)], dim=1
    )
    preds = (phi_ev @ w_mat).argmax(dim=1)
    acc = 100.0 * float((preds == y_eval).double().mean().item())

    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    dt = time.perf_counter() - t0
    return {"accuracy_pct": acc, "gcv_lambda": best_lam, "eval_time_sec": dt}


def evaluate_multinomial_logistic_gpu(
    x_sub: torch.Tensor,
    y_sub: torch.Tensor,
    w_sub: torch.Tensor,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    num_classes: int,
    max_iter: int = 45,
    expand_poly_low_d: bool = True,
) -> dict[str, float]:
    """Train GPU L-BFGS Multinomial Logistic Regression strictly on `(x_sub, y_sub, w_sub)`."""
    dev = x_sub.device
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t0 = time.perf_counter()

    with torch.no_grad():
        zs_sub64, zs_ev64, w_norm64 = _standardize_and_expand_coreset_features(
            x_sub, w_sub, x_eval, expand_poly_low_d=expand_poly_low_d
        )
        w_ridge, gcv_lam = _solve_gcv_cholesky_ridge_fp64(zs_sub64, y_sub, w_norm64, num_classes=num_classes)
        m_eff = float((1.0 / (w_norm64 * w_norm64).sum().clamp_min(1e-12)).item())
        l2_reg = float(min(max(gcv_lam / max(m_eff, 64.0), 5.0e-5), 5.0e-2))
        xs32 = zs_sub64.clamp(-25.0, 25.0).float().contiguous()
        xev32 = zs_ev64.clamp(-25.0, 25.0).float().contiguous()
        w32 = w_norm64.float().contiguous()
        y_s = y_sub.to(device=dev, dtype=torch.int64)
        # Warm-start strictly convex L-BFGS from zero-mean centered Ridge direction if it lowers initial loss
        w_init = (w_ridge[:-1] - w_ridge[:-1].mean(dim=1, keepdim=True)).clamp(-5.0, 5.0).float() * 2.0
        b_init = (w_ridge[-1] - w_ridge[-1].mean()).clamp(-5.0, 5.0).float() * 2.0
        loss_zero = float(math.log(max(num_classes, 2)))
        ce_ws = F.cross_entropy(xs32 @ w_init + b_init, y_s, reduction="none")
        loss_ws = float(((w32 * ce_ws).sum() + 0.5 * l2_reg * w_init.square().sum()).item())
        use_ws = math.isfinite(loss_ws) and (loss_ws < loss_zero)

    p_dim = int(xs32.shape[1])
    W = (
        w_init.contiguous().clone()
        if use_ws
        else torch.zeros((p_dim, num_classes), device=dev, dtype=torch.float32)
    ).requires_grad_(True)
    b = (
        b_init.contiguous().clone()
        if use_ws
        else torch.zeros((num_classes,), device=dev, dtype=torch.float32)
    ).requires_grad_(True)
    opt = torch.optim.LBFGS(
        [W, b],
        lr=1.0,
        max_iter=int(max_iter),
        history_size=20,
        line_search_fn="strong_wolfe",
    )

    with torch.enable_grad():
        def _closure() -> torch.Tensor:
            opt.zero_grad(set_to_none=True)
            logits = xs32 @ W + b
            ce = F.cross_entropy(logits, y_s, reduction="none")
            loss = (w32 * ce).sum() + 0.5 * l2_reg * W.square().sum()
            loss.backward()
            return loss

        opt.step(_closure)

    with torch.no_grad():
        W_det = torch.nan_to_num(W.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        b_det = torch.nan_to_num(b.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        preds = (xev32 @ W_det + b_det).argmax(dim=1)
        acc = 100.0 * float((preds == y_eval).double().mean().item())

    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    dt = time.perf_counter() - t0
    return {"accuracy_pct": acc, "l2_reg": l2_reg, "eval_time_sec": dt}


@torch.no_grad()
def _estimate_coreset_knn_bandwidth(xs: torch.Tensor) -> torch.Tensor:
    """Estimate positive squared-L2 softmax bandwidth strictly from `xs` (`0` test query coupling!)."""
    m = int(xs.shape[0])
    if m <= 1:
        return torch.tensor(1.0, device=xs.device, dtype=torch.float32)
    n_probe = min(m, 1024)
    step = max(1, m // n_probe)
    q = xs[::step][:n_probe]  # (Q_p, D)
    qn = q.square().sum(dim=1, keepdim=True)
    min_pos_d2 = torch.full((q.shape[0],), float("inf"), device=xs.device, dtype=torch.float32)
    q_global = torch.arange(0, q.shape[0] * step, step, device=xs.device, dtype=torch.int64)[: q.shape[0]]

    for ds in range(0, m, 32768):
        de = min(ds + 32768, m)
        xd = xs[ds:de]
        dn = xd.square().sum(dim=1)
        d2 = (qn + dn[None, :] - 2.0 * (q @ xd.T)).clamp_min_(0.0)
        d_ids = torch.arange(ds, de, device=xs.device, dtype=torch.int64)
        self_mask = q_global[:, None] == d_ids[None, :]
        d2 = torch.where(self_mask, torch.full_like(d2, float("inf")), d2)
        min_pos_d2 = torch.minimum(min_pos_d2, d2.min(dim=1).values)

    finite = min_pos_d2[torch.isfinite(min_pos_d2) & (min_pos_d2 > 0.0)]
    if finite.numel() == 0:
        return torch.tensor(1.0, device=xs.device, dtype=torch.float32)
    return finite.median().clamp_min(1e-6)


@torch.no_grad()
def evaluate_weighted_knn_gpu(
    x_sub: torch.Tensor,
    y_sub: torch.Tensor,
    w_sub: torch.Tensor,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    num_classes: int,
    top_m: int = 7,
) -> dict[str, float]:
    """Origin-centered, scale-conditioned weighted `$k$-NN` (`0` test leakage, `100%` per-query independent)."""
    dev = x_sub.device
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t0 = time.perf_counter()

    w = torch.nan_to_num(w_sub.double(), nan=1.0, posinf=1.0, neginf=0.0).clamp_min(0.0)
    w_sum = w.sum()
    if float(w_sum.item()) <= 0.0:
        w = torch.ones_like(w)
        w_sum = w.sum()
    w_norm = w / w_sum
    origin64 = _compute_equivariant_origin64(x_sub)
    xs, xev, _ = _condition_centered_pair(x_sub, x_eval, origin64, scale_from_x_only=True)

    m = int(xs.shape[0])
    m_eff = max(1, min(int(top_m), m))
    cand_eff = max(m_eff, min(m, max(m_eff * 4, 24)))
    use_bf16_tc = (dev.type == "cuda") and (m >= 8192)
    xs_dot = xs.to(torch.bfloat16) if use_bf16_tc else xs
    s_norm = xs_dot.float().square().sum(dim=1)
    tau = _estimate_coreset_knn_bandwidth(xs)
    oh = F.one_hot(y_sub.to(torch.int64), num_classes=num_classes).float()
    w32 = w_norm.float()

    preds = torch.empty((xev.shape[0],), dtype=torch.int64, device=dev)
    q_chunk = 4096
    db_chunk = 65536
    for qs in range(0, xev.shape[0], q_chunk):
        qe = min(qs + q_chunk, xev.shape[0])
        xq = xev[qs:qe]
        xq_dot = xq.to(torch.bfloat16) if use_bf16_tc else xq
        qn = xq_dot.float().square().sum(dim=1, keepdim=True)
        cand_d2 = torch.full((qe - qs, cand_eff), float("inf"), device=dev, dtype=torch.float32)
        cand_id = torch.zeros((qe - qs, cand_eff), device=dev, dtype=torch.int64)
        for ds in range(0, m, db_chunk):
            de = min(ds + db_chunk, m)
            xd_dot = xs_dot[ds:de]
            dots = (xq_dot @ xd_dot.T).float()
            d2 = (qn + s_norm[ds:de][None, :] - 2.0 * dots).clamp_min_(0.0)
            k_loc = min(cand_eff, de - ds)
            v_loc, i_loc = torch.topk(d2, k=k_loc, dim=1, largest=False)
            comb_d2 = torch.cat([cand_d2, v_loc], dim=1)
            comb_id = torch.cat([cand_id, i_loc + ds], dim=1)
            cand_d2, sel = torch.topk(comb_d2, k=cand_eff, dim=1, largest=False)
            cand_id = torch.gather(comb_id, 1, sel)
        # Exact IEEE FP32 direct-difference refinement over top `cand_eff` candidates
        exact_cand_d2 = (xq[:, None, :] - xs[cand_id]).square().sum(dim=2)
        best_d2, sub_sel = torch.topk(exact_cand_d2, k=m_eff, dim=1, largest=False)
        best_id = torch.gather(cand_id, 1, sub_sel)
        logits_w = torch.nan_to_num(-best_d2 / tau, nan=-1.0e30, neginf=-1.0e30, posinf=0.0)
        logits_w = logits_w - logits_w.max(dim=1, keepdim=True).values
        attn = torch.exp(logits_w) * w32[best_id]
        probs = torch.einsum("nm,nmc->nc", attn, oh[best_id])
        preds[qs:qe] = probs.argmax(dim=1)

    acc = 100.0 * float((preds == y_eval).double().mean().item())
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    dt = time.perf_counter() - t0
    return {"accuracy_pct": acc, "bandwidth_tau": float(tau.item()), "eval_time_sec": dt}


@torch.no_grad()
def evaluate_prototype_knn_gpu(
    x_sub: torch.Tensor,
    y_sub: torch.Tensor,
    w_sub: torch.Tensor,
    cluster_ids_sub: torch.Tensor,
    k_clusters: int,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    num_classes: int,
    top_m: int = 7,
) -> dict[str, float]:
    """Compress retained coreset `(x_sub, y_sub, w_sub)` into `<= K` cluster prototypes & evaluate `$k$-NN`.

    Strictly uses ONLY the retained coreset (`x_sub`, `y_sub`, `w_sub`) — zero discarded labels touched!
    """
    dev = x_sub.device
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t0 = time.perf_counter()

    c_ids = cluster_ids_sub.to(device=dev, dtype=torch.int64)
    w64 = torch.nan_to_num(w_sub.double(), nan=1.0, posinf=1.0, neginf=0.0).clamp_min(1e-12)
    origin64 = _compute_equivariant_origin64(x_sub)
    xs, xev, _ = _condition_centered_pair(x_sub, x_eval, origin64, scale_from_x_only=True)

    mass = torch.zeros((k_clusters,), device=dev, dtype=torch.float64)
    mass.index_add_(0, c_ids, w64)
    active = mass > 0.0
    n_act = int(active.sum().item())
    if n_act == 0:
        raise ValueError("No active clusters in evaluate_prototype_knn_gpu")

    proto_sums = torch.zeros((k_clusters, xs.shape[1]), device=dev, dtype=torch.float64)
    proto_sums.index_add_(0, c_ids, xs.double() * w64[:, None])
    protos = (proto_sums[active] / mass[active][:, None]).float().contiguous()  # (K_act, D)

    oh = F.one_hot(y_sub.to(torch.int64), num_classes=num_classes).double()
    cls_mass = torch.zeros((k_clusters, num_classes), device=dev, dtype=torch.float64)
    cls_mass.index_add_(0, c_ids, oh * w64[:, None])
    # Weighted cluster class distribution (incorporating cluster mass!)
    cls_post = (cls_mass[active] / mass.sum()).float().contiguous()  # (K_act, C)

    tau = _estimate_coreset_knn_bandwidth(protos)
    m_eff = max(1, min(int(top_m), n_act))
    p_norm = protos.square().sum(dim=1)
    preds = torch.empty((xev.shape[0],), dtype=torch.int64, device=dev)
    for qs in range(0, xev.shape[0], 4096):
        qe = min(qs + 4096, xev.shape[0])
        xq = xev[qs:qe]
        qn = xq.square().sum(dim=1, keepdim=True)
        d2 = (qn + p_norm[None, :] - 2.0 * (xq @ protos.T)).clamp_min_(0.0)
        v_top, i_top = torch.topk(d2, k=m_eff, dim=1, largest=False)
        logits_w = -v_top / tau
        logits_w = logits_w - logits_w.max(dim=1, keepdim=True).values
        attn = torch.exp(logits_w)
        probs = torch.einsum("nm,nmc->nc", attn, cls_post[i_top])
        preds[qs:qe] = probs.argmax(dim=1)

    acc = 100.0 * float((preds == y_eval).double().mean().item())
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    dt = time.perf_counter() - t0
    return {"accuracy_pct": acc, "active_prototypes": float(n_act), "eval_time_sec": dt}


# ==============================================================================
# 7. Megapixel Vector Quantization PSNR & Full-Image Reconstruction Helpers
# ==============================================================================


@torch.no_grad()
def compute_vq_psnr_fp64(
    x_pixels_or_patches: torch.Tensor | np.ndarray,
    centroids: torch.Tensor | np.ndarray,
    peak_val: float = 255.0,
    device: torch.device | str | None = None,
) -> tuple[float, float, torch.Tensor]:
    """Compute exact IEEE-754 FP64 `(psnr_db, pixel_mse, labels)` for `[0, 255]` RGB or patch VQ.

    Note: `fast_exact_assign_fp64` returns `VectorMSE = E[||x - c||_2^2]` summed across `D` channels.
    Standard image reconstruction `PixelMSE` divides by `D`: `PixelMSE = VectorMSE / D`.
    """
    x_t = torch.from_numpy(x_pixels_or_patches) if isinstance(x_pixels_or_patches, np.ndarray) else x_pixels_or_patches
    c_t = torch.from_numpy(centroids) if isinstance(centroids, np.ndarray) else centroids
    d = int(x_t.shape[1])
    vec_mse, labels = fast_exact_assign_fp64(x_t, c_t, device=device)
    pixel_mse = float(vec_mse) / float(max(d, 1))
    psnr_db = 10.0 * math.log10((float(peak_val) ** 2) / max(pixel_mse, 1e-12))
    return psnr_db, pixel_mse, labels


def reconstruct_patch_mosaic_rgb(
    codes: np.ndarray | torch.Tensor,
    centroids: np.ndarray | torch.Tensor,
    mosaic_hw: tuple[int, int],
    patch_h: int,
    patch_w: int,
    channels: int = 3,
) -> np.ndarray:
    """Reconstruct a full `(H, W, 3)` RGB image from VQ patch indices and `(K, ph*pw*3)` codebook."""
    idx = codes.detach().cpu().numpy() if isinstance(codes, torch.Tensor) else np.asarray(codes)
    cb = centroids.detach().cpu().numpy() if isinstance(centroids, torch.Tensor) else np.asarray(centroids)
    h, w = int(mosaic_hw[0]), int(mosaic_hw[1])
    gh, gw = h // int(patch_h), w // int(patch_w)
    if idx.size != gh * gw:
        raise ValueError(f"Expected {gh * gw} codes for {mosaic_hw} with {patch_h}x{patch_w} patches, got {idx.size}")
    quant_patches = cb[idx].reshape(gh, gw, int(patch_h), int(patch_w), int(channels))
    img = quant_patches.transpose(0, 2, 1, 3, 4).reshape(h, w, int(channels))
    return np.ascontiguousarray(img, dtype=np.float64)


@torch.no_grad()
def encode_vq_indices_gpu(
    x_gpu: torch.Tensor,
    centroids_gpu: torch.Tensor,
    origin_gpu: torch.Tensor | None = None,
) -> tuple[torch.Tensor, float]:
    """Uniform, de-confounded GPU VQ image encoder returning `(codes, encode_time_sec)` (`0` query coupling)."""
    dev = x_gpu.device
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    eff_origin = _compute_equivariant_origin64(centroids_gpu) if origin_gpu is None else origin_gpu
    _, labels = centered_assign_fp32(x_gpu, centroids_gpu, origin=eff_origin)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    dt = time.perf_counter() - t0
    return labels, dt


@torch.no_grad()
def warmup_semdedup_kernels(
    device: torch.device | str = "cuda",
    d: int = 256,
    k: int = 1024,
) -> None:
    """Pre-compile `_semdedup_margin_kernel` & centered `assign_points` specializations outside timers."""
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        return
    n_w = max(k * 2, 4096)
    x_w = torch.randn((n_w, d), device=dev, dtype=torch.float32)
    c_w = x_w[:k].clone().contiguous()
    st = compute_semdedup_scores_gpu(x_w, c_w, k=k)
    _ = select_coreset_at_budget(st, retention_frac=0.20)
    _ = select_prototype_coreset(st)
    torch.cuda.synchronize(dev)


__all__ = [
    "CoresetSelection",
    "DEFAULT_SEMDEDUP_DATA_DIR",
    "FOUR_LABELED_BENCHMARKS",
    "LABELED_SPEC_BY_NAME",
    "LabeledBenchmarkSpec",
    "LabeledVectorDataset",
    "MEGAPIXEL_SPEC_BY_NAME",
    "MegapixelVQDataset",
    "MegapixelVQSpec",
    "SemDeDupScoreState",
    "TWO_MEGAPIXEL_VQ_BENCHMARKS",
    "centered_assign_fp32",
    "compute_semdedup_scores_gpu",
    "compute_vq_psnr_fp64",
    "embed_labeled_text_zero_leakage",
    "encode_vq_indices_gpu",
    "evaluate_cholesky_ridge_gpu",
    "evaluate_multinomial_logistic_gpu",
    "evaluate_prototype_knn_gpu",
    "evaluate_weighted_knn_gpu",
    "load_agnews_256_labeled",
    "load_cifar10_convnext_768_labeled",
    "load_cifar10_resnet18_512_labeled",
    "load_covtype_54_labeled",
    "load_labeled_benchmark",
    "load_megapixel_vq_benchmark",
    "reconstruct_patch_mosaic_rgb",
    "select_coreset_at_budget",
    "select_prototype_coreset",
    "select_random_coreset",
    "validate_labeled_dataset",
    "validate_megapixel_vq_dataset",
    "warmup_semdedup_kernels",
]
