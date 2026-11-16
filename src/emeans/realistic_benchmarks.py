"""Broad 7-Modality Realistic Benchmark Suite for Phase 5 Dominance Evaluation.

Provides unified, zero-leakage, checksum-verified `RealVectorCorpus` loaders across
**7 genuinely distinct real-world vector modalities and geometries**, each equipped with:
  - Strict train (`base`) / validation (`val`) / test (`query`) split isolation (deduplicated
    at raw input level AND vector level, verified via `validate_real_corpus`),
  - Atomic disk caching (`PID + time.time_ns()` temp file + `os.fsync` + `os.replace`) inside
    `outputs/phase5_dominance/data/` (gitignored via `outputs/**/data/`),
  - Precomputed exact IEEE-754 FP64 top-100 nearest-neighbor query ground truth
    (`groundtruth_ids`, `groundtruth_sq_dists`) for downstream ANN IVF-Flat Recall@10!

The 7 Canonical Realistic Benchmarks (`SEVEN_REALISTIC_BENCHMARKS`):
  1. `sift1m`               (`1,000,000 x 128`, Euclidean `[0..255]` local SIFT visual descriptors)
  2. `glove1m_unit`         (`1,000,000 x 100`, Unit-Sphere `||x||=1` semantic NLP word vectors)
  3. `covtype_multiscale_54`(`  500,000 x  54`, Unnormalized physical sensors: `[1859..3858]`m elevation,
                             `[0..7173]`m distances, `[0..66]` slopes, and 44 binary `{0, 1}` soil flags)
  4. `cifar10_patches_128`  (` ~199,860 x 128`, Mid-level CNN spatial quadrant descriptors on CIFAR-10)
  5. `agnews_tfidf_svd_256` (` ~108,000 x 256`, Heavy-tailed Zipf news topic embeddings via train-only
                             sublinear TF-IDF + randomized SVD; supports both skewed-norm & unit-sphere)
  6. `cifar10_resnet18_512` (`  ~49,970 x 512`, Global ReLU non-negative orthant CNN semantic features)
  7. `cifar10_convnext_768` (`  ~49,970 x 768`, Modern ConvNeXt-Tiny LayerNorm+GELU signed embeddings)
"""

from __future__ import annotations

import csv
import gzip
import math
import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .datasets import (
    DEFAULT_CACHE_DIR,
    DEFAULT_CIFAR10_ROOT,
    DEFAULT_RESNET18_CKPT,
    _deduplicate_raw_images,
    _load_raw_cifar10,
    deduplicate_splits,
    load_cifar10_resnet18_embeddings,
)
from .real_corpora import (
    DEFAULT_MILLION_DATA_DIR,
    RealVectorCorpus,
    _exact_row_keys,
    atomic_savez,
    compute_exact_groundtruth_gpu,
    compute_sha256,
    evaluate_ivf_recall_gpu,
    fast_exact_assign_fp64,
    load_glove1m_unit,
    load_sift1m,
    validate_real_corpus,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHASE5_DATA_DIR = _REPO_ROOT / "outputs" / "phase5_dominance" / "data"

# Pretrained ConvNeXt-Tiny (ImageNet-1K V1, 768-D LayerNorm penultimate embeddings)
CONVNEXT_TINY_URL = "https://download.pytorch.org/models/convnext_tiny-983f1562.pth"
CONVNEXT_TINY_FILENAME = "convnext_tiny-983f1562.pth"
CONVNEXT_TINY_SIZE = 114_419_221
CONVNEXT_TINY_SHA256 = "983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d"

# UCI / USFS Forest Covertype (581,012 x 54 raw cartographic & physical sensor features)
COVTYPE_URLS: tuple[str, ...] = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz",
    "https://ndownloader.figshare.com/files/5976039",
)
COVTYPE_FILENAME = "covtype.data.gz"
COVTYPE_SIZE = 11_240_707
COVTYPE_SHA256 = "614360d0257557dd1792834a85a1cdebfadc3c4f30b011d56afee7ffb5b15771"

# AG's Corpus of News Articles (120,000 train + 7,600 official test articles)
AG_NEWS_TRAIN_URLS: tuple[str, ...] = (
    "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv",
)
AG_NEWS_TRAIN_FILENAME = "ag_news_train.csv"
AG_NEWS_TRAIN_SIZE = 29_470_338
AG_NEWS_TRAIN_SHA256 = "76a0a2d2f92b286371fe4d4044640910a04a803fdd2538e0f3f29a5c6f6b672e"

AG_NEWS_TEST_URLS: tuple[str, ...] = (
    "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv",
)
AG_NEWS_TEST_FILENAME = "ag_news_test.csv"
AG_NEWS_TEST_SIZE = 1_857_427
AG_NEWS_TEST_SHA256 = "521465c2428ed7f02f8d6db6ffdd4b5447c1c701962353eb2c40d548c3c85699"


@dataclass(frozen=True)
class RealisticBenchmarkSpec:
    """Metadata descriptor for one benchmark in the 7-dataset realistic suite."""

    name: str
    short_label: str
    modality: str
    d: int
    k_default: int
    geometry: str  # "euclidean" or "unit_sphere"
    cache_filename: str
    description: str


SEVEN_REALISTIC_BENCHMARKS: tuple[RealisticBenchmarkSpec, ...] = (
    RealisticBenchmarkSpec(
        name="sift1m",
        short_label="SIFT1M (1M×128)",
        modality="Local Visual Descriptors",
        d=128,
        k_default=4096,
        geometry="euclidean",
        cache_filename="sift1m_validated_v1.npz",
        description="1,000,000 x 128 Lowe SIFT descriptors in [0, 255] + 10,000 official queries.",
    ),
    RealisticBenchmarkSpec(
        name="glove1m_unit",
        short_label="GloVe-1M (1M×100, S^99)",
        modality="Unit-Sphere Word Embeddings",
        d=100,
        k_default=4096,
        geometry="unit_sphere",
        cache_filename="glove1m_unit_s20261116_validated_v1.npz",
        description="1,000,000 x 100 L2-unit-normalized GloVe Twitter 27B vectors + 10,000 queries.",
    ),
    RealisticBenchmarkSpec(
        name="covtype_multiscale_54",
        short_label="Covertype (500K×54, Raw)",
        modality="Multi-Scale Physical Sensors",
        d=54,
        k_default=2048,
        geometry="euclidean",
        cache_filename="covtype_multiscale_54_s20261116_validated_v1.npz",
        description="500,000 x 54 raw cartographic sensors: [1859..3858]m elevation, [0..7173]m distances, {0,1} soil.",
    ),
    RealisticBenchmarkSpec(
        name="cifar10_patches_128",
        short_label="CIFAR-10 Patches (200K×128)",
        modality="Mid-Level CNN Patch Features",
        d=128,
        k_default=2048,
        geometry="euclidean",
        cache_filename="cifar10_patches_128_validated_v1.npz",
        description="~199,860 x 128 spatial ResNet-18 layer2 quadrant descriptors with image-level split.",
    ),
    RealisticBenchmarkSpec(
        name="agnews_tfidf_svd_256",
        short_label="AG-News SVD (108K×256)",
        modality="Heavy-Tailed Text Embeddings",
        d=256,
        k_default=1024,
        geometry="euclidean",
        cache_filename="agnews_tfidf_svd_256_raw_s20261116_validated_v1.npz",
        description="~108,000 x 256 unnormalized sublinear TF-IDF + train-only SVD news topic embeddings.",
    ),
    RealisticBenchmarkSpec(
        name="cifar10_resnet18_512",
        short_label="CIFAR-10 ResNet18 (50K×512)",
        modality="Global ReLU CNN Embeddings",
        d=512,
        k_default=1024,
        geometry="euclidean",
        cache_filename="cifar10_resnet18_512_validated_v1.npz",
        description="~49,970 x 512 non-negative orthant ResNet-18 penultimate avgpool features.",
    ),
    RealisticBenchmarkSpec(
        name="cifar10_convnext_768",
        short_label="CIFAR-10 ConvNeXt (50K×768)",
        modality="Modern High-D Vision (768D)",
        d=768,
        k_default=1024,
        geometry="euclidean",
        cache_filename="cifar10_convnext_768_validated_v1.npz",
        description="~49,970 x 768 ConvNeXt-Tiny LayerNorm + GELU signed modern vision representations.",
    ),
)

BENCHMARK_SPEC_BY_NAME: dict[str, RealisticBenchmarkSpec] = {
    spec.name: spec for spec in SEVEN_REALISTIC_BENCHMARKS
}


# ==============================================================================
# Checksummed Atomic Downloader & Cross-Split Exact Deduplication Helpers
# ==============================================================================


def _download_atomic_verified(
    urls: tuple[str, ...],
    dst: Path,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
    timeout_sec: float = 60.0,
) -> None:
    """Atomically download `dst` via `PID+ns` temp file + `os.fsync` + `os.replace` with SHA-256 verification."""
    if dst.exists():
        sz_ok = expected_size is None or dst.stat().st_size == expected_size
        if sz_ok and (expected_sha256 is None or compute_sha256(dst) == expected_sha256):
            return

    dst.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for url in urls:
        tmp = dst.parent / f".tmp_{os.getpid()}_{time.time_ns()}_{dst.name}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "emeans-phase5-bench/1.0"})
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp, open(tmp, "wb") as wf:
                while True:
                    buf = resp.read(4 * 1024 * 1024)
                    if not buf:
                        break
                    wf.write(buf)
                wf.flush()
                os.fsync(wf.fileno())
            if expected_size is not None and tmp.stat().st_size != expected_size:
                raise IOError(
                    f"Byte size mismatch for {dst.name}: got {tmp.stat().st_size}, expected {expected_size}"
                )
            if expected_sha256 is not None:
                actual_sha = compute_sha256(tmp)
                if actual_sha != expected_sha256:
                    raise IOError(
                        f"SHA-256 mismatch for {dst.name}: got {actual_sha}, expected {expected_sha256}"
                    )
            os.replace(tmp, dst)
            return
        except Exception as exc:
            last_err = exc
            if tmp.exists():
                tmp.unlink(missing_ok=True)
    raise RuntimeError(f"Failed to download {dst.name} from {urls!r}: {last_err}") from last_err


def _strict_deduplicate_three_way(
    base: np.ndarray,
    val: np.ndarray,
    query: np.ndarray,
    decimals: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply both rounded-decimal (`deduplicate_splits`) and exact float32 byte-key deduplication."""
    b1, v1, q1 = deduplicate_splits(base, val, query, decimals=decimals)

    kb = _exact_row_keys(b1)
    _, ub_idx = np.unique(kb, return_index=True)
    ub_idx.sort()
    b_clean = np.ascontiguousarray(b1[ub_idx], dtype=np.float32)
    seen_b = kb[ub_idx]

    kv = _exact_row_keys(v1)
    _, uv_idx = np.unique(kv, return_index=True)
    uv_idx.sort()
    v_keep = uv_idx[~np.isin(kv[uv_idx], seen_b)]
    v_clean = np.ascontiguousarray(v1[v_keep], dtype=np.float32)
    seen_bv = np.union1d(seen_b, kv[v_keep])

    kq = _exact_row_keys(q1)
    _, uq_idx = np.unique(kq, return_index=True)
    uq_idx.sort()
    q_keep = uq_idx[~np.isin(kq[uq_idx], seen_bv)]
    q_clean = np.ascontiguousarray(q1[q_keep], dtype=np.float32)

    return b_clean, v_clean, q_clean


def _load_cached_corpus_npz(
    cache_path: Path,
    expected_name: str,
    expected_d: int,
    expected_geometry: str,
    extra_meta: dict[str, Any] | None = None,
) -> RealVectorCorpus:
    """Load a cached `RealVectorCorpus` `.npz` archive and run full `validate_real_corpus` checks."""
    with np.load(cache_path, allow_pickle=False) as z:
        meta = dict(extra_meta or {})
        for k in z.files:
            if k not in ("base", "val", "query", "groundtruth_ids", "groundtruth_sq_dists", "official_groundtruth_ids"):
                arr_k = z[k]
                if arr_k.ndim == 0:
                    meta[k] = arr_k.item()
        corpus = RealVectorCorpus(
            name=expected_name,
            base=np.ascontiguousarray(z["base"], dtype=np.float32),
            val=np.ascontiguousarray(z["val"], dtype=np.float32),
            query=np.ascontiguousarray(z["query"], dtype=np.float32),
            groundtruth_ids=np.ascontiguousarray(z["groundtruth_ids"], dtype=np.int32),
            groundtruth_sq_dists=np.ascontiguousarray(z["groundtruth_sq_dists"], dtype=np.float64),
            d=int(expected_d),
            geometry=expected_geometry,
            metadata=meta,
        )
    validate_real_corpus(corpus, check_cross_split_disjoint=True)
    return corpus


# ==============================================================================
# 1 & 2. Million-Scale Reused Benchmarks (`SIFT1M` & `GloVe-1M-Unit`)
# ==============================================================================


def load_sift1m_benchmark(
    data_dir: Path | str = DEFAULT_PHASE5_DATA_DIR,
    device: str | torch.device | None = None,
) -> RealVectorCorpus:
    """Load SIFT1M (`1,000,000 x 128` base, `89,788` val, `10,000` query), reusing Phase 4 cache if present."""
    ddir = Path(data_dir)
    fname = "sift1m_validated_v1.npz"
    if (ddir / fname).exists():
        return load_sift1m(data_dir=ddir, device=device)
    if (DEFAULT_MILLION_DATA_DIR / fname).exists():
        return load_sift1m(data_dir=DEFAULT_MILLION_DATA_DIR, device=device)
    return load_sift1m(data_dir=ddir, device=device)


def load_glove1m_unit_benchmark(
    data_dir: Path | str = DEFAULT_PHASE5_DATA_DIR,
    split_seed: int = 20261116,
    device: str | torch.device | None = None,
) -> RealVectorCorpus:
    """Load GloVe-Twitter-1M-Unit (`1,000,000 x 100` unit sphere), reusing Phase 4 cache if present."""
    ddir = Path(data_dir)
    fname = f"glove1m_unit_s{int(split_seed)}_validated_v1.npz"
    if (ddir / fname).exists():
        return load_glove1m_unit(data_dir=ddir, split_seed=split_seed, device=device)
    if (DEFAULT_MILLION_DATA_DIR / fname).exists():
        return load_glove1m_unit(data_dir=DEFAULT_MILLION_DATA_DIR, split_seed=split_seed, device=device)
    return load_glove1m_unit(data_dir=ddir, split_seed=split_seed, device=device)


# ==============================================================================
# 3 & 4. CIFAR-10 ResNet-18 Benchmarks (`200K x 128D` Patches & `50K x 512D` Global)
# ==============================================================================


def _load_cifar10_resnet18_corpus(
    mode: str,
    corpus_name: str,
    expected_d: int,
    cache_filename: str,
    data_dir: Path | str = DEFAULT_PHASE5_DATA_DIR,
    cifar_dir: Path = DEFAULT_CIFAR10_ROOT,
    ckpt_path: Path = DEFAULT_RESNET18_CKPT,
    device: str | torch.device | None = None,
) -> RealVectorCorpus:
    """Wrap CIFAR-10 ResNet-18 splits with exact FP64 top-100 ground truth and atomic `.npz` caching."""
    ddir = Path(data_dir)
    cache_path = ddir / cache_filename
    source_label = f"CIFAR-10 + ImageNet ResNet-18 ({mode})"

    if cache_path.exists():
        return _load_cached_corpus_npz(
            cache_path,
            expected_name=corpus_name,
            expected_d=expected_d,
            expected_geometry="euclidean",
            extra_meta={"source": source_label, "mode": mode},
        )

    # Reuse existing Phase 1 extracted feature cache if available, otherwise extract into `ddir`
    p1_cache = DEFAULT_CACHE_DIR if (DEFAULT_CACHE_DIR / f"cifar10_resnet18_{mode}.npz").exists() else ddir
    dev_str = str(device) if device is not None else None
    ds = load_cifar10_resnet18_embeddings(
        mode=mode,
        cifar_dir=cifar_dir,
        ckpt_path=ckpt_path,
        cache_dir=p1_cache,
        device=dev_str,
    )
    xb, xv, xq = _strict_deduplicate_three_way(ds.train, ds.val, ds.test, decimals=5)
    gt_ids, gt_d2 = compute_exact_groundtruth_gpu(xb, xq, top_k=100, device=device)

    atomic_savez(
        cache_path,
        base=xb,
        val=xv,
        query=xq,
        groundtruth_ids=gt_ids,
        groundtruth_sq_dists=gt_d2,
        n_base_rows=np.int64(xb.shape[0]),
        n_val_rows=np.int64(xv.shape[0]),
        n_query_rows=np.int64(xq.shape[0]),
    )
    return _load_cached_corpus_npz(
        cache_path,
        expected_name=corpus_name,
        expected_d=expected_d,
        expected_geometry="euclidean",
        extra_meta={"source": source_label, "mode": mode},
    )


def load_cifar10_patches_128(
    data_dir: Path | str = DEFAULT_PHASE5_DATA_DIR,
    cifar_dir: Path = DEFAULT_CIFAR10_ROOT,
    ckpt_path: Path = DEFAULT_RESNET18_CKPT,
    device: str | torch.device | None = None,
) -> RealVectorCorpus:
    """Load CIFAR-10 ResNet-18 `layer2` local spatial patch descriptors (`~199.9K x 128` + FP64 GT-100)."""
    return _load_cifar10_resnet18_corpus(
        mode="patches_128",
        corpus_name="cifar10_patches_128",
        expected_d=128,
        cache_filename="cifar10_patches_128_validated_v1.npz",
        data_dir=data_dir,
        cifar_dir=cifar_dir,
        ckpt_path=ckpt_path,
        device=device,
    )


def load_cifar10_resnet18_512(
    data_dir: Path | str = DEFAULT_PHASE5_DATA_DIR,
    cifar_dir: Path = DEFAULT_CIFAR10_ROOT,
    ckpt_path: Path = DEFAULT_RESNET18_CKPT,
    device: str | torch.device | None = None,
) -> RealVectorCorpus:
    """Load CIFAR-10 ResNet-18 global semantic embeddings (`~49.97K x 512` + FP64 GT-100)."""
    return _load_cifar10_resnet18_corpus(
        mode="global_512",
        corpus_name="cifar10_resnet18_512",
        expected_d=512,
        cache_filename="cifar10_resnet18_512_validated_v1.npz",
        data_dir=data_dir,
        cifar_dir=cifar_dir,
        ckpt_path=ckpt_path,
        device=device,
    )


# ==============================================================================
# 5. High-D Modern Vision Embeddings (`ConvNeXt-Tiny`, `d = 768`)
# ==============================================================================


@torch.no_grad()
def load_cifar10_convnext_768(
    data_dir: Path | str = DEFAULT_PHASE5_DATA_DIR,
    cifar_dir: Path = DEFAULT_CIFAR10_ROOT,
    device: str | torch.device | None = None,
) -> RealVectorCorpus:
    """Extract & atomically cache 768-D modern ConvNeXt-Tiny LayerNorm embeddings on CIFAR-10.

    Strict Data Hygiene:
      1. Raw uint8 `(3, 32, 32)` CIFAR-10 images are split into `50,000` train / `5,000` val /
         `5,000` query and deduplicated via `_deduplicate_raw_images` BEFORE feature extraction.
      2. Pretrained `convnext_tiny-983f1562.pth` (ImageNet-1K V1) is verified by byte size + SHA-256
         inside `data_dir` (never polluting `$HOME/.cache`).
      3. Penultimate `768-D` LayerNorm representations (`features -> avgpool -> classifier[0] -> flatten`)
         are extracted in IEEE FP32, vector-deduplicated (`decimals=5`), paired with exact FP64 top-100
         nearest-neighbor query ground truth, and saved via `PID+ns` temp + `os.replace`.
    """
    ddir = Path(data_dir)
    cache_path = ddir / "cifar10_convnext_768_validated_v1.npz"
    source_label = "CIFAR-10 + ImageNet-1K ConvNeXt-Tiny (LayerNorm 768-D)"

    if cache_path.exists():
        return _load_cached_corpus_npz(
            cache_path,
            expected_name="cifar10_convnext_768",
            expected_d=768,
            expected_geometry="euclidean",
            extra_meta={"source": source_label, "arch": "convnext_tiny"},
        )

    import torch.nn.functional as F
    import torchvision.models as models

    ckpt_file = ddir / CONVNEXT_TINY_FILENAME
    _download_atomic_verified(
        urls=(CONVNEXT_TINY_URL,),
        dst=ckpt_file,
        expected_size=CONVNEXT_TINY_SIZE,
        expected_sha256=CONVNEXT_TINY_SHA256,
    )

    tr_raw, te_full_raw = _load_raw_cifar10(cifar_dir)
    va_raw = te_full_raw[:5000]
    te_raw = te_full_raw[5000:]
    tr_imgs, va_imgs, te_imgs = _deduplicate_raw_images(tr_raw, va_raw, te_raw)

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
        bsz = 512
        for s in range(0, imgs.shape[0], bsz):
            batch = torch.from_numpy(imgs[s : s + bsz]).to(dev, dtype=torch.float32) / 255.0
            inp = F.interpolate(batch, size=(96, 96), mode="bilinear", align_corners=False)
            inp = (inp - mean) / std
            h = net.features(inp)
            h = net.avgpool(h)
            h = net.classifier[0](h)  # LayerNorm2d((768,), eps=1e-6)
            h = torch.flatten(h, 1)   # (B, 768) signed embeddings
            chunks.append(h.cpu().numpy().astype(np.float32))
        return np.concatenate(chunks, axis=0)

    tr_f = _extract_768(tr_imgs)
    va_f = _extract_768(va_imgs)
    te_f = _extract_768(te_imgs)

    xb, xv, xq = _strict_deduplicate_three_way(tr_f, va_f, te_f, decimals=5)
    gt_ids, gt_d2 = compute_exact_groundtruth_gpu(xb, xq, top_k=100, device=dev)

    atomic_savez(
        cache_path,
        base=xb,
        val=xv,
        query=xq,
        groundtruth_ids=gt_ids,
        groundtruth_sq_dists=gt_d2,
        n_base_rows=np.int64(xb.shape[0]),
        n_val_rows=np.int64(xv.shape[0]),
        n_query_rows=np.int64(xq.shape[0]),
    )
    return _load_cached_corpus_npz(
        cache_path,
        expected_name="cifar10_convnext_768",
        expected_d=768,
        expected_geometry="euclidean",
        extra_meta={"source": source_label, "arch": "convnext_tiny"},
    )


# ==============================================================================
# 6. Real Text / Document Embeddings (`AG-News` & `20Newsgroups` TF-IDF + SVD 256D)
# ==============================================================================


def _normalize_doc_text(text: str) -> str:
    """Canonical whitespace normalization for raw document string deduplication."""
    return " ".join(text.strip().split())


def _deduplicate_raw_documents(
    train_docs: list[str],
    val_docs: list[str],
    query_docs: list[str],
) -> tuple[list[str], list[str], list[str]]:
    """Deduplicate raw text documents case-insensitively within and across `(train, val, query)`."""
    seen: set[str] = set()
    clean_tr: list[str] = []
    for doc in train_docs:
        norm = _normalize_doc_text(doc)
        key = norm.lower()
        if len(key) >= 8 and key not in seen:
            seen.add(key)
            clean_tr.append(norm)

    clean_va: list[str] = []
    for doc in val_docs:
        norm = _normalize_doc_text(doc)
        key = norm.lower()
        if len(key) >= 8 and key not in seen:
            seen.add(key)
            clean_va.append(norm)

    clean_qu: list[str] = []
    for doc in query_docs:
        norm = _normalize_doc_text(doc)
        key = norm.lower()
        if len(key) >= 8 and key not in seen:
            seen.add(key)
            clean_qu.append(norm)

    return clean_tr, clean_va, clean_qu


def embed_text_splits_zero_leakage(
    train_docs: list[str],
    val_docs: list[str],
    query_docs: list[str],
    d: int = 256,
    max_features: int = 50_000,
    min_df: int = 3,
    unit_normalize: bool = False,
    svd_seed: int = 20261116,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Embed `(train, val, query)` text documents with STRICT train-only TF-IDF and train-only SVD.

    Zero Leakage Invariant:
      - `TfidfVectorizer` vocabulary and IDF weights are fitted exclusively on `train_docs`.
      - `TruncatedSVD` singular vectors are fitted exclusively on `X_train_tfidf`.
      - `val_docs` and `query_docs` undergo pure out-of-sample `.transform(...)` projections.
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    clean_tr, clean_va, clean_qu = _deduplicate_raw_documents(train_docs, val_docs, query_docs)
    if len(clean_tr) <= d:
        raise ValueError(f"Insufficient unique training documents ({len(clean_tr)}) for SVD d={d}")

    vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        max_features=int(max_features),
        min_df=int(min_df),
        max_df=0.90,
        ngram_range=(1, 2),
        norm=None,  # Preserve natural document salience & heavy-tailed length skew before SVD
        dtype=np.float64,
    )
    x_tr_sparse = vectorizer.fit_transform(clean_tr)
    x_va_sparse = vectorizer.transform(clean_va)
    x_qu_sparse = vectorizer.transform(clean_qu)

    n_vocab = int(x_tr_sparse.shape[1])
    if n_vocab < d:
        raise ValueError(f"Vocabulary size {n_vocab} smaller than target dimension d={d}")

    svd = TruncatedSVD(n_components=int(d), n_iter=7, random_state=int(svd_seed))
    z_tr64 = svd.fit_transform(x_tr_sparse)
    z_va64 = svd.transform(x_va_sparse)
    z_qu64 = svd.transform(x_qu_sparse)

    def _postprocess(z64: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(z64, axis=1, keepdims=True)
        keep = norms[:, 0] > 1e-6
        z_keep = z64[keep]
        if unit_normalize:
            n_keep = norms[keep]
            u32 = (z_keep / n_keep).astype(np.float32)
            # Second FP64 pass ensures float32 norms equal 1.0 within < 1e-6
            u64 = u32.astype(np.float64)
            u32 = (u64 / np.linalg.norm(u64, axis=1, keepdims=True)).astype(np.float32)
            return np.ascontiguousarray(u32, dtype=np.float32)
        return np.ascontiguousarray(z_keep.astype(np.float32), dtype=np.float32)

    zb_raw = _postprocess(z_tr64)
    zv_raw = _postprocess(z_va64)
    zq_raw = _postprocess(z_qu64)

    zb, zv, zq = _strict_deduplicate_three_way(zb_raw, zv_raw, zq_raw, decimals=6)
    svals = svd.singular_values_
    stats_meta = {
        "vocab_size": float(n_vocab),
        "svd_sigma_max": float(svals[0]),
        "svd_sigma_min": float(svals[-1]),
        "svd_condition_ratio": float(svals[0] / max(float(svals[-1]), 1e-12)),
        "explained_variance_sum": float(np.sum(svd.explained_variance_ratio_)),
    }
    return zb, zv, zq, stats_meta


def _read_ag_news_csv(path: Path) -> list[str]:
    docs: list[str] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 3:
                docs.append(f"{row[1].strip()}. {row[2].strip()}")
    return docs


def load_agnews_tfidf_svd_256(
    data_dir: Path | str = DEFAULT_PHASE5_DATA_DIR,
    unit_normalize: bool = False,
    split_seed: int = 20261116,
    val_size: int = 10_000,
    device: str | torch.device | None = None,
) -> RealVectorCorpus:
    """Load AG-News `d = 256` real document embeddings (`~108K` base, `~10K` val, `~7.5K` query).

    Supports BOTH:
      - `unit_normalize=False` (`agnews_tfidf_svd_256`): Unnormalized skewed-norm heavy-tailed Zipf
        topic representations (`||x||_2` spanning `~4..120+`), testing dynamic range + anisotropy.
      - `unit_normalize=True` (`agnews_tfidf_svd_256_unit`): Exact unit-sphere (`S^255`) L2-normalized
        document embeddings for cosine/angular clustering.
    """
    ddir = Path(data_dir)
    mode_tag = "unit" if unit_normalize else "raw"
    corpus_name = "agnews_tfidf_svd_256_unit" if unit_normalize else "agnews_tfidf_svd_256"
    geom = "unit_sphere" if unit_normalize else "euclidean"
    cache_path = ddir / f"agnews_tfidf_svd_256_{mode_tag}_s{int(split_seed)}_validated_v1.npz"
    source_label = f"AG-News Corpus (train-only sublinear TF-IDF + SVD 256-D, {mode_tag})"

    if cache_path.exists():
        return _load_cached_corpus_npz(
            cache_path,
            expected_name=corpus_name,
            expected_d=256,
            expected_geometry=geom,
            extra_meta={"source": source_label, "unit_normalize": bool(unit_normalize)},
        )

    tr_csv = ddir / AG_NEWS_TRAIN_FILENAME
    te_csv = ddir / AG_NEWS_TEST_FILENAME
    _download_atomic_verified(
        urls=AG_NEWS_TRAIN_URLS,
        dst=tr_csv,
        expected_size=AG_NEWS_TRAIN_SIZE,
        expected_sha256=AG_NEWS_TRAIN_SHA256,
    )
    _download_atomic_verified(
        urls=AG_NEWS_TEST_URLS,
        dst=te_csv,
        expected_size=AG_NEWS_TEST_SIZE,
        expected_sha256=AG_NEWS_TEST_SHA256,
    )

    raw_train_all = _read_ag_news_csv(tr_csv)
    raw_test_all = _read_ag_news_csv(te_csv)

    # Deduplicate raw_train_all first, then deterministically split into train + val
    seen_tr_raw: set[str] = set()
    uniq_train_docs: list[str] = []
    for doc in raw_train_all:
        norm = _normalize_doc_text(doc)
        key = norm.lower()
        if len(key) >= 8 and key not in seen_tr_raw:
            seen_tr_raw.add(key)
            uniq_train_docs.append(norm)

    rng = np.random.default_rng(int(split_seed))
    perm = rng.permutation(len(uniq_train_docs))
    n_val = min(int(val_size), max(1000, len(uniq_train_docs) // 10))
    val_docs = [uniq_train_docs[int(i)] for i in perm[:n_val]]
    train_docs = [uniq_train_docs[int(i)] for i in perm[n_val:]]

    zb, zv, zq, svd_meta = embed_text_splits_zero_leakage(
        train_docs=train_docs,
        val_docs=val_docs,
        query_docs=raw_test_all,
        d=256,
        max_features=50_000,
        min_df=3,
        unit_normalize=unit_normalize,
        svd_seed=split_seed,
    )

    gt_ids, gt_d2 = compute_exact_groundtruth_gpu(zb, zq, top_k=100, device=device)
    atomic_savez(
        cache_path,
        base=zb,
        val=zv,
        query=zq,
        groundtruth_ids=gt_ids,
        groundtruth_sq_dists=gt_d2,
        vocab_size=np.float64(svd_meta["vocab_size"]),
        svd_sigma_max=np.float64(svd_meta["svd_sigma_max"]),
        svd_sigma_min=np.float64(svd_meta["svd_sigma_min"]),
        svd_condition_ratio=np.float64(svd_meta["svd_condition_ratio"]),
        explained_variance_sum=np.float64(svd_meta["explained_variance_sum"]),
    )
    return _load_cached_corpus_npz(
        cache_path,
        expected_name=corpus_name,
        expected_d=256,
        expected_geometry=geom,
        extra_meta={"source": source_label, "unit_normalize": bool(unit_normalize)},
    )


# ==============================================================================
# 7. Real Multi-Scale Physical Sensor Data (`Forest Covertype`, `500K x 54D`)
# ==============================================================================


def _parse_covtype_gz(gz_path: Path) -> np.ndarray:
    """Fast parser for UCI `covtype.data.gz` (`581,012 x 55` CSV -> `581,012 x 54` float32 features)."""
    with gzip.open(gz_path, "rt", encoding="ascii") as f:
        raw_text = f.read()
    # Replace newlines with commas and parse all 581,012 * 55 numbers in C via np.fromstring
    flat = np.fromstring(raw_text.replace("\n", ","), dtype=np.float32, sep=",")
    if flat.size != 581_012 * 55:
        raise ValueError(f"Unexpected covtype element count {flat.size} != {581_012 * 55}")
    mat = flat.reshape(581_012, 55)
    feats = np.ascontiguousarray(mat[:, :54], dtype=np.float32)
    if not np.isfinite(feats).all():
        raise ValueError("Non-finite values in parsed Forest Covertype matrix")
    return feats


def validate_covtype_physical_ranges(feats: np.ndarray) -> None:
    """Verify authentic unnormalized physical sensor dynamic ranges on Forest Covertype."""
    if feats.ndim != 2 or feats.shape[1] != 54:
        raise ValueError(f"Expected (*, 54) Covertype matrix, got {feats.shape}")
    elev_min, elev_max = float(feats[:, 0].min()), float(feats[:, 0].max())
    if not (1800.0 <= elev_min <= 2000.0 and 3700.0 <= elev_max <= 3950.0):
        raise ValueError(f"Covertype col 0 (elevation) outside [1800..3950]m: [{elev_min}, {elev_max}]")
    slope_min, slope_max = float(feats[:, 2].min()), float(feats[:, 2].max())
    if not (0.0 <= slope_min <= 2.0 and 50.0 <= slope_max <= 75.0):
        raise ValueError(f"Covertype col 2 (slope) outside [0..75] deg: [{slope_min}, {slope_max}]")
    bin_vals = np.unique(feats[:, 10:])
    if not np.all(np.isin(bin_vals, np.array([0.0, 1.0], dtype=np.float32))):
        raise ValueError(f"Covertype cols 10..53 must be binary {{0, 1}}, got {bin_vals}")


def load_covtype_multiscale_54(
    data_dir: Path | str = DEFAULT_PHASE5_DATA_DIR,
    split_seed: int = 20261116,
    n_base: int = 500_000,
    n_val: int = 50_000,
    n_query: int = 10_000,
    device: str | torch.device | None = None,
) -> RealVectorCorpus:
    """Load Forest Covertype (`500,000 x 54` base, `50,000` val, `10,000` query, raw physical units).

    Authentic Multi-Scale Dynamic Range Stress Test:
      - Coord `0` (Elevation): `[1859 .. 3858]` meters (`~+2959` mean offset!)
      - Coord `1` (Aspect): `[0 .. 360]` degrees
      - Coord `2` (Slope): `[0 .. 66]` degrees
      - Coords `3..5, 9` (Hydrology / Roadway / Fire distances): `[-173 .. 7173]` meters
      - Coords `6..8` (Hillshade indices): `[0 .. 254]`
      - Coords `10..53` (Wilderness & Soil Type indicators): `44` binary `{0, 1}` flags!
    """
    ddir = Path(data_dir)
    is_canonical = (n_base == 500_000 and n_val == 50_000 and n_query == 10_000)
    suffix = f"s{int(split_seed)}" if is_canonical else f"s{int(split_seed)}_{n_base}_{n_val}_{n_query}"
    cache_path = ddir / f"covtype_multiscale_54_{suffix}_validated_v1.npz"
    source_label = "UCI / USFS Forest Covertype (581,012 x 54 raw cartographic & soil sensors)"

    if cache_path.exists():
        corpus = _load_cached_corpus_npz(
            cache_path,
            expected_name="covtype_multiscale_54",
            expected_d=54,
            expected_geometry="euclidean",
            extra_meta={"source": source_label, "split_seed": int(split_seed)},
        )
        validate_covtype_physical_ranges(corpus.base)
        return corpus

    gz_path = ddir / COVTYPE_FILENAME
    _download_atomic_verified(
        urls=COVTYPE_URLS,
        dst=gz_path,
        expected_size=COVTYPE_SIZE,
        expected_sha256=COVTYPE_SHA256,
    )
    feats = _parse_covtype_gz(gz_path)
    validate_covtype_physical_ranges(feats)

    # Global exact deduplication before deterministic shuffle & split
    rkeys = _exact_row_keys(feats)
    _, uniq_idx = np.unique(rkeys, return_index=True)
    uniq_idx.sort()
    feats_uniq = np.ascontiguousarray(feats[uniq_idx], dtype=np.float32)

    total_needed = int(n_base + n_val + n_query)
    if feats_uniq.shape[0] < total_needed:
        raise ValueError(f"Requested {total_needed} rows from {feats_uniq.shape[0]} unique Covertype rows")

    rng = np.random.default_rng(int(split_seed))
    perm = rng.permutation(feats_uniq.shape[0])
    xb_raw = feats_uniq[perm[:n_base]]
    xv_raw = feats_uniq[perm[n_base : n_base + n_val]]
    xq_raw = feats_uniq[perm[n_base + n_val : total_needed]]

    xb, xv, xq = _strict_deduplicate_three_way(xb_raw, xv_raw, xq_raw, decimals=6)
    if xb.shape != (n_base, 54) or xv.shape != (n_val, 54) or xq.shape != (n_query, 54):
        raise RuntimeError(f"Unexpected post-dedup Covertype shapes: {xb.shape}, {xv.shape}, {xq.shape}")

    gt_ids, gt_d2 = compute_exact_groundtruth_gpu(xb, xq, top_k=100, device=device)
    atomic_savez(
        cache_path,
        base=xb,
        val=xv,
        query=xq,
        groundtruth_ids=gt_ids,
        groundtruth_sq_dists=gt_d2,
        raw_unique_rows=np.int64(feats_uniq.shape[0]),
        elevation_min=np.float64(xb[:, 0].min()),
        elevation_max=np.float64(xb[:, 0].max()),
        max_feature_value=np.float64(xb.max()),
    )
    corpus = _load_cached_corpus_npz(
        cache_path,
        expected_name="covtype_multiscale_54",
        expected_d=54,
        expected_geometry="euclidean",
        extra_meta={"source": source_label, "split_seed": int(split_seed)},
    )
    validate_covtype_physical_ranges(corpus.base)
    return corpus


# ==============================================================================
# Unified 7-Benchmark Dispatcher, Dynamic-Range Diagnostics & Naive FP8 Probe
# ==============================================================================


def load_realistic_benchmark(
    name: str,
    data_dir: Path | str = DEFAULT_PHASE5_DATA_DIR,
    device: str | torch.device | None = None,
) -> RealVectorCorpus:
    """Unified loader for all 7 realistic benchmarks (plus unit-sphere text variant)."""
    key = name.strip().lower()
    if key == "sift1m":
        return load_sift1m_benchmark(data_dir=data_dir, device=device)
    if key in ("glove1m_unit", "glove1m"):
        return load_glove1m_unit_benchmark(data_dir=data_dir, device=device)
    if key in ("covtype_multiscale_54", "covtype", "forest_covertype"):
        return load_covtype_multiscale_54(data_dir=data_dir, device=device)
    if key == "cifar10_patches_128":
        return load_cifar10_patches_128(data_dir=data_dir, device=device)
    if key == "agnews_tfidf_svd_256":
        return load_agnews_tfidf_svd_256(data_dir=data_dir, unit_normalize=False, device=device)
    if key == "agnews_tfidf_svd_256_unit":
        return load_agnews_tfidf_svd_256(data_dir=data_dir, unit_normalize=True, device=device)
    if key in ("cifar10_resnet18_512", "cifar10_r18_512"):
        return load_cifar10_resnet18_512(data_dir=data_dir, device=device)
    if key == "cifar10_convnext_768":
        return load_cifar10_convnext_768(data_dir=data_dir, device=device)
    raise ValueError(
        f"Unknown realistic benchmark {name!r}; expected one of "
        f"{[s.name for s in SEVEN_REALISTIC_BENCHMARKS] + ['agnews_tfidf_svd_256_unit']}"
    )


def stage_all_seven_benchmarks(
    data_dir: Path | str = DEFAULT_PHASE5_DATA_DIR,
    device: str | torch.device | None = None,
) -> dict[str, RealVectorCorpus]:
    """Ensure all 7 realistic benchmarks are validated and cached on disk before fleet dispatch."""
    staged: dict[str, RealVectorCorpus] = {}
    for spec in SEVEN_REALISTIC_BENCHMARKS:
        staged[spec.name] = load_realistic_benchmark(spec.name, data_dir=data_dir, device=device)
    return staged


def compute_corpus_dynamic_range_stats(
    corpus: RealVectorCorpus,
    max_rows: int = 65_536,
) -> dict[str, float]:
    """Compute modality geometry & multi-scale dynamic-range diagnostics on `corpus.base`."""
    n = corpus.base.shape[0]
    if n > max_rows:
        step = max(1, n // max_rows)
        sub = corpus.base[::step][:max_rows].astype(np.float64)
    else:
        sub = corpus.base.astype(np.float64)

    mean_vec = sub.mean(axis=0, keepdims=True)
    centered = sub - mean_vec
    feat_stds = centered.std(axis=0)
    nonzero_stds = feat_stds[feat_stds > 1e-12]
    std_span = (
        float(nonzero_stds.max() / nonzero_stds.min())
        if nonzero_stds.size >= 2
        else 1.0
    )

    row_norms = np.linalg.norm(sub, axis=1)
    p01, p50, p99 = np.quantile(row_norms, [0.01, 0.50, 0.99])
    norm_skew_p99_p01 = float(p99 / max(float(p01), 1e-12))

    rms_centered = math.sqrt(float(np.mean(centered * centered)))
    origin_shift_ratio = float(np.linalg.norm(mean_vec) / max(rms_centered * math.sqrt(corpus.d), 1e-12))

    # Participation ratio (effective intrinsic dimensionality) tr(C)^2 / ||C||_F^2
    cov = (centered.T @ centered) / max(1, sub.shape[0] - 1)
    tr_cov = float(np.trace(cov))
    frob_sq = float(np.sum(cov * cov))
    participation_dim = (tr_cov * tr_cov) / max(frob_sq, 1e-30)

    return {
        "n_base": float(corpus.base.shape[0]),
        "n_val": float(corpus.val.shape[0]),
        "n_query": float(corpus.query.shape[0]),
        "d": float(corpus.d),
        "coord_min": float(corpus.base.min()),
        "coord_max": float(corpus.base.max()),
        "feature_std_max_over_min": std_span,
        "row_norm_median": float(p50),
        "row_norm_p99_over_p01": norm_skew_p99_p01,
        "origin_shift_ratio": origin_shift_ratio,
        "effective_rank_pr": float(participation_dim),
    }


@torch.no_grad()
def evaluate_fp8_assignment_stability_probe(
    x_eval: torch.Tensor | np.ndarray,
    centroids: torch.Tensor | np.ndarray,
    origin: torch.Tensor | np.ndarray | None = None,
    device: str | torch.device | None = None,
) -> dict[str, float]:
    """Compare Naive Uncentered Global-Scale FP8 vs CroissAttn-Style Centered Decoupled FP8 Assignment.

    Given identical `centroids` and held-out `x_eval`, assigns each point under:
      1. Exact FP64 Euclidean assignment (`exact_mse`),
      2. **Naive Global-Scale Uncentered FP8 (`float8_e4m3fn`)**: single tensor-wide scale
         `s_x = max(|X|) / 448`, `s_c = max(|C|) / 448` without origin centering — exposing how
         high-offset / multi-scale coordinates (`Covertype`, `SIFT1M`) suffer mantissa quantization,
      3. **CroissAttn-Style Origin-Centered Per-Row + Per-Centroid Decoupled FP8 (`float8_e4m3fn`)**:
         subtracts `origin` (`mean(X)`), computes per-row `sx_i = clamp(max_d |x_i - mu| / 448, min=TINY)`
         and per-centroid `sc_j = clamp(max_d |c_j - mu| / 448, min=TINY)`, quantizes to `e4m3fn` with
         `[-448, 448]` saturation, and evaluates represented-norm distances in FP32, then scores the
         resulting Voronoi partition in exact IEEE-754 FP64!
    """
    FP8_MAX = 448.0
    FP32_TINY = 1.17549435e-38

    x_t = torch.from_numpy(x_eval) if isinstance(x_eval, np.ndarray) else x_eval.detach()
    c_t = torch.from_numpy(centroids) if isinstance(centroids, np.ndarray) else centroids.detach()
    dev = (
        torch.device(device)
        if device is not None
        else (
            x_t.device
            if x_t.is_cuda
            else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        )
    )
    x_f32 = x_t.to(device=dev, dtype=torch.float32)
    c_f32 = c_t.to(device=dev, dtype=torch.float32)
    x_f64 = x_t.to(device=dev, dtype=torch.float64)
    c_f64 = c_t.to(device=dev, dtype=torch.float64)

    exact_mse, exact_lbl = fast_exact_assign_fp64(x_f32, c_f32, device=dev)

    # 1. Naive Uncentered Global-Scale FP8 E4M3FN
    sx_glob = (x_f32.abs().amax() / FP8_MAX).clamp_min(FP32_TINY)
    sc_glob = (c_f32.abs().amax() / FP8_MAX).clamp_min(FP32_TINY)
    qx_naive = (x_f32 / sx_glob).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float() * sx_glob
    qc_naive = (c_f32 / sc_glob).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float() * sc_glob
    # Uncentered expanded norm ||x||^2 + ||c||^2 - 2 x.c in FP32
    cn_naive = qc_naive.square().sum(dim=1)
    naive_lbl = torch.empty((x_f32.shape[0],), dtype=torch.int64, device=dev)
    for s in range(0, x_f32.shape[0], 8192):
        e = min(s + 8192, x_f32.shape[0])
        xb_n = qx_naive[s:e]
        xn_n = xb_n.square().sum(dim=1, keepdim=True)
        scores_n = xn_n + cn_naive[None, :] - 2.0 * (xb_n @ qc_naive.t())
        naive_lbl[s:e] = scores_n.argmin(dim=1)
    naive_mse = float((x_f64 - c_f64[naive_lbl]).square().sum(dim=1).mean().item())
    naive_disagree = float((naive_lbl != exact_lbl).double().mean().item())

    # 2. CroissAttn-Style Origin-Centered Decoupled Per-Row / Per-Centroid FP8 E4M3FN
    if origin is not None:
        mu = (torch.from_numpy(origin) if isinstance(origin, np.ndarray) else origin.detach()).to(
            device=dev, dtype=torch.float32
        ).reshape(1, -1)
    else:
        mu = x_f64.mean(dim=0, keepdim=True).to(dtype=torch.float32)

    x_cent = x_f32 - mu
    c_cent = c_f32 - mu
    sx_row = (x_cent.abs().amax(dim=1, keepdim=True) / FP8_MAX).clamp_min(FP32_TINY)
    sc_row = (c_cent.abs().amax(dim=1, keepdim=True) / FP8_MAX).clamp_min(FP32_TINY)
    qx_cr = (x_cent / sx_row).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float()
    qc_cr = (c_cent / sc_row).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float()
    # Represented-norm identity: sx_i^2 ||qx_i||^2 + sc_j^2 ||qc_j||^2 - 2 sx_i sc_j (qx_i . qc_j)
    c_tilde = qc_cr * sc_row
    cn_cr = c_tilde.square().sum(dim=1)
    croiss_lbl = torch.empty((x_f32.shape[0],), dtype=torch.int64, device=dev)
    for s in range(0, x_f32.shape[0], 8192):
        e = min(s + 8192, x_f32.shape[0])
        x_tilde = qx_cr[s:e] * sx_row[s:e]
        scores_cr = cn_cr[None, :] - 2.0 * (x_tilde @ c_tilde.t())
        croiss_lbl[s:e] = scores_cr.argmin(dim=1)
    croiss_mse = float((x_f64 - c_f64[croiss_lbl]).square().sum(dim=1).mean().item())
    croiss_disagree = float((croiss_lbl != exact_lbl).double().mean().item())

    denom = max(abs(exact_mse), 1e-30)
    return {
        "exact_fp64_mse": float(exact_mse),
        "naive_global_fp8_mse": naive_mse,
        "naive_global_fp8_rel_inflate_pct": 100.0 * (naive_mse - exact_mse) / denom,
        "naive_global_fp8_label_disagree_pct": 100.0 * naive_disagree,
        "croiss_decoupled_fp8_mse": croiss_mse,
        "croiss_decoupled_fp8_rel_inflate_pct": 100.0 * (croiss_mse - exact_mse) / denom,
        "croiss_decoupled_fp8_label_disagree_pct": 100.0 * croiss_disagree,
    }


__all__ = [
    "AG_NEWS_TEST_SHA256",
    "AG_NEWS_TRAIN_SHA256",
    "BENCHMARK_SPEC_BY_NAME",
    "CONVNEXT_TINY_SHA256",
    "COVTYPE_SHA256",
    "DEFAULT_PHASE5_DATA_DIR",
    "RealisticBenchmarkSpec",
    "SEVEN_REALISTIC_BENCHMARKS",
    "compute_corpus_dynamic_range_stats",
    "embed_text_splits_zero_leakage",
    "evaluate_fp8_assignment_stability_probe",
    "evaluate_ivf_recall_gpu",
    "load_agnews_tfidf_svd_256",
    "load_cifar10_convnext_768",
    "load_cifar10_patches_128",
    "load_cifar10_resnet18_512",
    "load_covtype_multiscale_54",
    "load_glove1m_unit_benchmark",
    "load_realistic_benchmark",
    "load_sift1m_benchmark",
    "stage_all_seven_benchmarks",
    "validate_covtype_physical_ranges",
]
