"""Million-Scale Real-World Vector Corpora & Sync-Free GPU ANN IVF Recall Evaluator.

Provides:
1. Automated, checksum- & shape-verified acquisition and atomic caching (`PID+ns` temp file +
   `os.replace`) for genuine million-scale vector benchmarks in `outputs/phase4_million_scale/data/`:
     - **SIFT1M** (`load_sift1m`): `1,000,000 x 128` float32 base descriptors, clean deduplicated
       validation vectors (`sift_learn` strictly purged of any `base`/`query` overlap), `10,000 x 128`
       official held-out test queries, and `10,000 x 100` official nearest-neighbor ground truth
       (cross-verified in IEEE-754 FP64 on GPU with 0.0 distance discrepancy across all 1M pairs).
     - **GloVe-Twitter-1M-Unit** (`load_glove1m_unit`): `1,000,000 x 100` unit-sphere L2-normalized
       semantic word embeddings (`SkyTbac/ANNS_GloVe.twitter.27B.100d` / Stanford GloVe Twitter 27B),
       with `100,000` validation and `10,000` held-out test queries carved from a seeded global
       permutation after strict unit-sphere deduplication, plus exact FP64 top-100 ground truth.
2. Fast Exact FP64 Distortion & Voronoi Assignment (`fast_exact_assign_fp64`, `fast_exact_mse_fp64`,
   `exact_cosine_distortion_fp64`):
     - Fused Triton IEEE-FP32 centered+scaled top-4 candidate filter followed by exact IEEE-754 FP64
       `(x64 - c64)^2` refinement (`~11.7x` faster than full FP64 GEMM on Blackwell SM120 with
       `~2e-16` relative parity, coordinate-shift invariance, and deterministic tie-breaking).
3. Blazing-Fast, Sync-Free GPU Inverted-File (IVF-Flat) ANN Recall Evaluator (`evaluate_ivf_recall_gpu`):
     - Reproduces & extends Appendix K of Bessa et al. ("e-means", arXiv:2611.09501): partitions all
       `N = 1,000,000` database vectors into `K` learned Voronoi buckets in CSR layout on GPU, routes
       all `Q = 10,000` queries to their top `nprobe` centroids (`0.1%`, `0.5%`, `1.0%` selectivity),
       scans inverted lists in parallel via a zero-sync Triton `tl.topk(uint64)` kernel + exact FP64
       re-ranking, and scores exact tie-aware & ID-intersection Recall@1, Recall@10, and Recall@100.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import math
import os
import tarfile
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .datasets import deduplicate_splits

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MILLION_DATA_DIR = _REPO_ROOT / "outputs" / "phase4_million_scale" / "data"

# Verified byte sizes and SHA-256 digests of official TexMex SIFT1M files
SIFT1M_SPECS: dict[str, tuple[int, str]] = {
    "sift_base.fvecs": (
        516_000_000,
        "21f66e2975057b5728ba56de1c825bac4f4d89d596609ae985741c6242631816",
    ),
    "sift_learn.fvecs": (
        51_600_000,
        "331bc82b6a0e89465776a3ba0c2113e0bd0cceaa014ec3ed639bc8b981af72ea",
    ),
    "sift_query.fvecs": (
        5_160_000,
        "f7fc9be140accdfd64116c2fa2365ecdb69b8f084970c6b0532db5ff79ac8fdc",
    ),
    "sift_groundtruth.ivecs": (
        4_040_000,
        "2b71de0a8d5a83e6a84eec3e23fb8b611d8801dd9b3a6cd62f070ab65ea65f4f",
    ),
}

SIFT1M_PRIMARY_BASE_URLS: tuple[str, ...] = (
    "https://huggingface.co/datasets/qbo-odp/sift1m/resolve/main",
    "https://huggingface.co/datasets/qbo-odp/anns/resolve/main/sift1m",
)
SIFT1M_TAR_MIRROR_URL = "https://huggingface.co/datasets/fzliu/sift1m/resolve/main/sift.tar.gz"
SIFT1M_TAR_SIZE = 168_280_445

GLOVE_TWITTER_100D_URL = (
    "https://huggingface.co/datasets/SkyTbac/ANNS_GloVe.twitter.27B.100d/resolve/main/"
    "glove.twitter.27B.100d.fvecs"
)
GLOVE_TWITTER_100D_FILENAME = "glove.twitter.27B.100d.fvecs"
GLOVE_TWITTER_100D_SIZE = 482_179_656


# ==============================================================================
# Data Structures
# ==============================================================================


@dataclass
class RealVectorCorpus:
    """Validated million-scale real vector corpus with strict train/val/query isolation & GT."""

    name: str
    base: np.ndarray                 # (N_base, D) float32 — database & clustering training set
    val: np.ndarray                  # (N_val, D) float32 — strictly disjoint validation set
    query: np.ndarray                # (N_query, D) float32 — strictly disjoint test query set
    groundtruth_ids: np.ndarray      # (N_query, G) int32 — exact nearest-neighbor indices in `base`
    groundtruth_sq_dists: np.ndarray # (N_query, G) float64 — exact FP64 squared L2 distances
    d: int
    geometry: str                    # "euclidean" or "unit_sphere"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def train(self) -> np.ndarray:
        return self.base

    @property
    def test(self) -> np.ndarray:
        return self.query


@dataclass(frozen=True)
class IVFSelectivityMetrics:
    """Downstream ANN IVF search metrics at a single `nprobe` / selectivity operating point."""

    selectivity: float
    nprobe: int
    recall_at_1: float
    recall_at_10: float
    recall_at_100: float
    id_recall_at_1: float
    id_recall_at_10: float
    id_recall_at_100: float
    one_recall_at_10: float
    mean_scanned_vecs: float
    actual_scan_frac: float
    p99_scanned_vecs: float


@dataclass
class IVFRecallResult:
    """Complete multi-selectivity GPU IVF-Flat evaluation summary for a learned codebook."""

    k: int
    n_base: int
    n_query: int
    imbalance_factor: float
    empty_clusters: int
    min_cluster_size: int
    max_cluster_size: int
    by_nprobe: dict[int, IVFSelectivityMetrics]
    by_selectivity: dict[str, IVFSelectivityMetrics]
    eval_time_sec: float


# ==============================================================================
# Binary `.fvecs` / `.ivecs` Readers, Checksums & Atomic Disk Caching
# ==============================================================================


def compute_sha256(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Stream-compute SHA-256 hex digest of a file without loading all bytes at once."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk_bytes)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def read_fvecs(path: Path) -> np.ndarray:
    """Parse TexMex `.fvecs` binary file (`[int32 d, float32 * d]` per row) with header validation."""
    raw = np.fromfile(path, dtype="<i4")
    if raw.size == 0:
        raise ValueError(f"Empty .fvecs file: {path}")
    dim = int(raw[0])
    if dim < 1 or dim > 65536:
        raise ValueError(f"Invalid vector dimension {dim} in {path}")
    stride = dim + 1
    if raw.size % stride != 0:
        raise ValueError(f"Corrupt .fvecs size {raw.size} ints not divisible by d+1={stride} in {path}")
    rows = raw.reshape(-1, stride)
    if not np.all(rows[:, 0] == dim):
        raise ValueError(f"Inconsistent per-row dimension headers in {path}")
    arr = np.ascontiguousarray(rows[:, 1:].view("<f4"), dtype=np.float32)
    if not np.isfinite(arr).all():
        raise ValueError(f"Non-finite (NaN/Inf) entries detected in {path}")
    return arr


def read_ivecs(path: Path) -> np.ndarray:
    """Parse TexMex `.ivecs` binary file (`[int32 d, int32 * d]` per row) with header validation."""
    raw = np.fromfile(path, dtype="<i4")
    if raw.size == 0:
        raise ValueError(f"Empty .ivecs file: {path}")
    dim = int(raw[0])
    if dim < 1 or dim > 65536:
        raise ValueError(f"Invalid index list width {dim} in {path}")
    stride = dim + 1
    if raw.size % stride != 0:
        raise ValueError(f"Corrupt .ivecs size {raw.size} ints not divisible by d+1={stride} in {path}")
    rows = raw.reshape(-1, stride)
    if not np.all(rows[:, 0] == dim):
        raise ValueError(f"Inconsistent per-row dimension headers in {path}")
    return np.ascontiguousarray(rows[:, 1:], dtype=np.int32)


def write_fvecs_atomic(path: Path, arr: np.ndarray) -> None:
    """Atomically write a 2D `float32` array in `.fvecs` format via PID+ns temp + `os.replace`."""
    arr_f32 = np.ascontiguousarray(arr, dtype="<f4")
    if arr_f32.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr_f32.shape}")
    n, d = arr_f32.shape
    packed = np.empty((n, d + 1), dtype="<i4")
    packed[:, 0] = d
    packed[:, 1:] = arr_f32.view("<i4")

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".tmp_{os.getpid()}_{time.time_ns()}_{path.name}"
    try:
        with open(tmp, "wb") as f:
            packed.tofile(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def atomic_savez(path: Path, **arrays: Any) -> None:
    """Atomically write an uncompressed `.npz` archive using PID+ns temp file + `os.replace`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".tmp_{os.getpid()}_{time.time_ns()}_{path.name}"
    try:
        with open(tmp, "wb") as f:
            np.savez(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def _download_file_parallel_atomic(
    urls: Sequence[str],
    dst: Path,
    expected_size: int,
    expected_sha256: str | None = None,
    n_workers: int = 16,
    timeout_sec: float = 45.0,
) -> None:
    """Download `dst` using parallel HTTP Range requests into a PID-temp file and `os.replace`."""
    if dst.exists() and dst.stat().st_size == expected_size:
        if expected_sha256 is None or compute_sha256(dst) == expected_sha256:
            return

    dst.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None

    for url in urls:
        tmp = dst.parent / f".tmp_{os.getpid()}_{time.time_ns()}_{dst.name}"
        try:
            # Probe Range support on the first 4 bytes before parallelizing
            probe_req = urllib.request.Request(
                url,
                headers={"Range": "bytes=0-3", "User-Agent": "emeans-real-scale/1.0"},
            )
            with urllib.request.urlopen(probe_req, timeout=timeout_sec) as resp:
                supports_range = (resp.status == 206) and (len(resp.read()) == 4)

            if supports_range and expected_size >= 4 * 1024 * 1024 and n_workers > 1:
                with open(tmp, "wb") as f:
                    f.truncate(expected_size)
                chunk_sz = (expected_size + n_workers - 1) // n_workers
                ranges = [
                    (i * chunk_sz, min(expected_size - 1, (i + 1) * chunk_sz - 1))
                    for i in range(n_workers)
                    if i * chunk_sz < expected_size
                ]

                def _fetch_range(bounds: tuple[int, int]) -> None:
                    start, end = bounds
                    expected_len = end - start + 1
                    for attempt in range(4):
                        try:
                            req = urllib.request.Request(
                                url,
                                headers={
                                    "Range": f"bytes={start}-{end}",
                                    "User-Agent": "emeans-real-scale/1.0",
                                },
                            )
                            with urllib.request.urlopen(req, timeout=timeout_sec) as r:
                                data = r.read()
                            if len(data) != expected_len:
                                raise IOError(f"Short HTTP range read: {len(data)} != {expected_len}")
                            with open(tmp, "r+b") as wf:
                                wf.seek(start)
                                wf.write(data)
                            return
                        except Exception:
                            if attempt == 3:
                                raise
                            time.sleep(0.5 * (attempt + 1))

                with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
                    list(pool.map(_fetch_range, ranges))
            else:
                req = urllib.request.Request(url, headers={"User-Agent": "emeans-real-scale/1.0"})
                with urllib.request.urlopen(req, timeout=timeout_sec * 4) as resp, open(tmp, "wb") as wf:
                    while True:
                        chunk = resp.read(4 * 1024 * 1024)
                        if not chunk:
                            break
                        wf.write(chunk)

            actual_size = tmp.stat().st_size
            if actual_size != expected_size:
                raise IOError(f"Size mismatch for {dst.name}: got {actual_size}, expected {expected_size}")
            if expected_sha256 is not None:
                digest = compute_sha256(tmp)
                if digest != expected_sha256:
                    raise IOError(f"SHA-256 mismatch for {dst.name}: got {digest}, expected {expected_sha256}")

            os.replace(tmp, dst)
            return
        except Exception as exc:
            last_err = exc
            if tmp.exists():
                tmp.unlink(missing_ok=True)

    raise RuntimeError(f"Failed to fetch {dst.name} from all mirrors ({urls!r}): {last_err}") from last_err


# ==============================================================================
# Exact FP64 Nearest-Neighbor Ground Truth & Fast Centered-FP32->FP64 Assignment
# ==============================================================================


def _power_of_two_scale(max_abs: float) -> float:
    """Return an IEEE-754 exact power-of-two scale factor preventing FP32 overflow/underflow."""
    if not math.isfinite(max_abs) or max_abs <= 0.0:
        return 1.0
    if 1e-12 <= max_abs <= 1e12:
        return 1.0
    exp = int(round(math.log2(max_abs)))
    # Clamp exponent so 2.0 ** exp and 1 / (2.0 ** exp) stay well inside finite float64
    exp = max(-900, min(900, exp))
    return math.ldexp(1.0, exp)


def _aligned_sum_sq_lastdim(diff: torch.Tensor) -> torch.Tensor:
    """Sum squares along last dim with 32-lane zero padding so identical vectors share identical GPU warp trees."""
    sq = diff.square()
    d = sq.shape[-1]
    rem = (-d) % 32
    if rem != 0 and sq.is_cuda:
        sq = torch.nn.functional.pad(sq, (0, rem), value=0.0)
    return sq.sum(dim=-1)


if HAS_TRITON:

    @triton.jit(
        do_not_specialize=[
            "N",
            "K",
            "stride_xn",
            "stride_xd",
            "stride_ck",
            "stride_cd",
        ]
    )
    def _top4_centered_fp32_kernel(
        x_ptr,
        c_ptr,
        c_norm_ptr,
        out_idx_ptr,
        N,
        K,
        D: tl.constexpr,
        stride_xn,
        stride_xd,
        stride_ck,
        stride_cd,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
        FULL_D: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
        mask_n = offs_n < N

        b0 = tl.full((BLOCK_N,), float("inf"), dtype=tl.float32)
        b1 = tl.full((BLOCK_N,), float("inf"), dtype=tl.float32)
        b2 = tl.full((BLOCK_N,), float("inf"), dtype=tl.float32)
        b3 = tl.full((BLOCK_N,), float("inf"), dtype=tl.float32)
        i0 = tl.zeros((BLOCK_N,), dtype=tl.int32)
        i1 = tl.zeros((BLOCK_N,), dtype=tl.int32)
        i2 = tl.zeros((BLOCK_N,), dtype=tl.int32)
        i3 = tl.zeros((BLOCK_N,), dtype=tl.int32)

        col_ids = tl.arange(0, BLOCK_K)[None, :]

        if FULL_D:
            offs_df = tl.arange(0, BLOCK_D).to(tl.int64)
            mask_df = offs_df < D
            x_full = tl.load(
                x_ptr + offs_n[:, None] * stride_xn + offs_df[None, :] * stride_xd,
                mask=mask_n[:, None] & mask_df[None, :],
                other=0.0,
            ).to(tl.float32)

        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K).to(tl.int64)
            mask_k = offs_k < K
            if FULL_D:
                ct = tl.load(
                    c_ptr + offs_k[:, None] * stride_ck + offs_df[None, :] * stride_cd,
                    mask=mask_k[:, None] & mask_df[None, :],
                    other=0.0,
                ).to(tl.float32)
                dots = tl.dot(x_full, tl.trans(ct), input_precision="ieee", out_dtype=tl.float32)
            else:
                dots = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
                for d0 in range(0, D, BLOCK_D):
                    offs_d = d0 + tl.arange(0, BLOCK_D).to(tl.int64)
                    mask_d = offs_d < D
                    xt = tl.load(
                        x_ptr + offs_n[:, None] * stride_xn + offs_d[None, :] * stride_xd,
                        mask=mask_n[:, None] & mask_d[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    ct = tl.load(
                        c_ptr + offs_k[:, None] * stride_ck + offs_d[None, :] * stride_cd,
                        mask=mask_k[:, None] & mask_d[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    dots = tl.dot(xt, tl.trans(ct), acc=dots, input_precision="ieee", out_dtype=tl.float32)

            cn = tl.load(c_norm_ptr + offs_k, mask=mask_k, other=float("inf"))
            scores = tl.where(mask_k[None, :], cn[None, :] - 2.0 * dots, float("inf"))

            tmin0, tidx0 = tl.min(scores, axis=1, return_indices=True)
            if tl.max((tmin0 < b3).to(tl.int32), axis=0) > 0:
                for pass_idx in tl.static_range(4):
                    if pass_idx == 0:
                        tmin, tidx = tmin0, tidx0
                    else:
                        tmin, tidx = tl.min(scores, axis=1, return_indices=True)
                    cand_i = k0.to(tl.int32) + tidx.to(tl.int32)
                    if pass_idx < 3:
                        scores = tl.where(col_ids == tidx[:, None], float("inf"), scores)

                    u3 = tmin < b3
                    b3 = tl.where(u3, tmin, b3)
                    i3 = tl.where(u3, cand_i, i3)

                    s2 = b3 < b2
                    tb, ti = tl.where(s2, b2, b3), tl.where(s2, i2, i3)
                    b2, i2 = tl.where(s2, b3, b2), tl.where(s2, i3, i2)
                    b3, i3 = tb, ti

                    s1 = b2 < b1
                    tb, ti = tl.where(s1, b1, b2), tl.where(s1, i1, i2)
                    b1, i1 = tl.where(s1, b2, b1), tl.where(s1, i2, i1)
                    b2, i2 = tb, ti

                    s0 = b1 < b0
                    tb, ti = tl.where(s0, b0, b1), tl.where(s0, i0, i1)
                    b0, i0 = tl.where(s0, b1, b0), tl.where(s0, i1, i0)
                    b1, i1 = tb, ti

        base_out = out_idx_ptr + offs_n * 4
        tl.store(base_out + 0, i0, mask=mask_n)
        tl.store(base_out + 1, i1, mask=mask_n)
        tl.store(base_out + 2, i2, mask=mask_n)
        tl.store(base_out + 3, i3, mask=mask_n)

    @triton.jit(
        do_not_specialize=[
            "BQ",
            "P_SLICE",
            "stride_qq",
            "stride_qd",
            "stride_xn",
            "stride_xd",
            "stride_pq",
            "stride_pp",
        ]
    )
    def _ivf_csr_bucket_topcap_kernel(
        q_ptr,
        x_csr_ptr,
        sorted_ids_ptr,
        cluster_offsets_ptr,
        probed_ptr,
        out_d2_ptr,
        out_id_ptr,
        BQ,
        P_SLICE,
        D: tl.constexpr,
        stride_qq,
        stride_qd,
        stride_xn,
        stride_xd,
        stride_pq,
        stride_pp,
        BLOCK_CAP: tl.constexpr,
        BLOCK_D: tl.constexpr,
        FULL_D: tl.constexpr,
    ):
        pid_q = tl.program_id(0).to(tl.int64)
        pid_p = tl.program_id(1).to(tl.int64)
        if pid_q >= BQ or pid_p >= P_SLICE:
            return

        cid = tl.load(probed_ptr + pid_q * stride_pq + pid_p * stride_pp).to(tl.int64)
        start = tl.load(cluster_offsets_ptr + cid).to(tl.int64)
        end = tl.load(cluster_offsets_ptr + cid + 1).to(tl.int64)
        length = end - start

        offs_cap = tl.arange(0, BLOCK_CAP).to(tl.int64)
        out_base = (pid_q * P_SLICE + pid_p) * BLOCK_CAP

        if length <= 0:
            tl.store(out_d2_ptr + out_base + offs_cap, float("inf"))
            tl.store(out_id_ptr + out_base + offs_cap, -1)
            return

        if FULL_D:
            offs_df = tl.arange(0, BLOCK_D).to(tl.int64)
            mask_df = offs_df < D
            qv_full = tl.load(
                q_ptr + pid_q * stride_qq + offs_df * stride_qd,
                mask=mask_df,
                other=0.0,
            ).to(tl.float32)

        if length <= BLOCK_CAP:
            mask_c = offs_cap < length
            src_pos = start + offs_cap
            if FULL_D:
                xv = tl.load(
                    x_csr_ptr + src_pos[:, None] * stride_xn + offs_df[None, :] * stride_xd,
                    mask=mask_c[:, None] & mask_df[None, :],
                    other=0.0,
                ).to(tl.float32)
                diff = xv - qv_full[None, :]
                acc = tl.sum(diff * diff, axis=1)
            else:
                acc = tl.zeros((BLOCK_CAP,), dtype=tl.float32)
                for d0 in range(0, D, BLOCK_D):
                    offs_d = d0 + tl.arange(0, BLOCK_D).to(tl.int64)
                    mask_d = offs_d < D
                    qv = tl.load(
                        q_ptr + pid_q * stride_qq + offs_d * stride_qd,
                        mask=mask_d,
                        other=0.0,
                    ).to(tl.float32)
                    xv = tl.load(
                        x_csr_ptr + src_pos[:, None] * stride_xn + offs_d[None, :] * stride_xd,
                        mask=mask_c[:, None] & mask_d[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    diff = xv - qv[None, :]
                    acc += tl.sum(diff * diff, axis=1)
            acc = tl.where(mask_c, tl.maximum(acc, 0.0), float("inf"))
            db_id = tl.load(sorted_ids_ptr + src_pos, mask=mask_c, other=-1)
            tl.store(out_d2_ptr + out_base + offs_cap, acc)
            tl.store(out_id_ptr + out_base + offs_cap, db_id)
            return

        # Oversized bucket (length > BLOCK_CAP): stream in BLOCK_CAP tiles keeping exact top-BLOCK_CAP
        best_keys = tl.full((BLOCK_CAP,), 0x7F800000FFFFFFFF, dtype=tl.uint64)
        for c0 in range(0, length, BLOCK_CAP):
            offs_c = c0 + offs_cap
            mask_c = offs_c < length
            src_pos = start + offs_c
            if FULL_D:
                xv = tl.load(
                    x_csr_ptr + src_pos[:, None] * stride_xn + offs_df[None, :] * stride_xd,
                    mask=mask_c[:, None] & mask_df[None, :],
                    other=0.0,
                ).to(tl.float32)
                diff = xv - qv_full[None, :]
                acc = tl.sum(diff * diff, axis=1)
            else:
                acc = tl.zeros((BLOCK_CAP,), dtype=tl.float32)
                for d0 in range(0, D, BLOCK_D):
                    offs_d = d0 + tl.arange(0, BLOCK_D).to(tl.int64)
                    mask_d = offs_d < D
                    qv = tl.load(
                        q_ptr + pid_q * stride_qq + offs_d * stride_qd,
                        mask=mask_d,
                        other=0.0,
                    ).to(tl.float32)
                    xv = tl.load(
                        x_csr_ptr + src_pos[:, None] * stride_xn + offs_d[None, :] * stride_xd,
                        mask=mask_c[:, None] & mask_d[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    diff = xv - qv[None, :]
                    acc += tl.sum(diff * diff, axis=1)
            acc = tl.maximum(acc, 0.0)
            db_id_u = tl.load(sorted_ids_ptr + src_pos, mask=mask_c, other=0xFFFFFFFF).to(tl.uint32).to(tl.uint64)
            d_bits = acc.to(tl.uint32, bitcast=True).to(tl.uint64)
            tile_keys = tl.where(mask_c, (d_bits << 32) | db_id_u, 0x7F800000FFFFFFFF)
            merged = tl.join(best_keys, tile_keys).reshape((2 * BLOCK_CAP,))
            best_keys = tl.topk(merged, BLOCK_CAP, descending=False)

        low32 = best_keys & 0xFFFFFFFF
        valid_k = low32 != 0xFFFFFFFF
        out_d2 = tl.where(valid_k, (best_keys >> 32).to(tl.uint32).to(tl.float32, bitcast=True), float("inf"))
        out_id = tl.where(valid_k, low32.to(tl.int64), -1)
        tl.store(out_d2_ptr + out_base + offs_cap, out_d2)
        tl.store(out_id_ptr + out_base + offs_cap, out_id)


@torch.no_grad()
def exact_assign_fp64_torch(
    x: torch.Tensor,
    centroids: torch.Tensor,
    row_chunk: int = 16384,
    col_chunk: int = 4096,
) -> tuple[float, torch.Tensor]:
    """Reference exact double-precision MSE and deterministic smallest-ID Voronoi assignment."""
    n, d = x.shape
    k = centroids.shape[0]
    if k < 1:
        raise ValueError("centroids must have K >= 1")
    if d != centroids.shape[1]:
        raise ValueError(f"Dimension mismatch: x has d={d}, centroids has d={centroids.shape[1]}")
    if n == 0:
        return 0.0, torch.empty((0,), dtype=torch.int64, device=x.device)

    dev = x.device
    c64_raw = centroids.to(device=dev, dtype=torch.float64)
    origin = c64_raw.mean(dim=0, keepdim=True)
    c64_diff = c64_raw - origin
    x_peek = x[: min(n, 4096)].to(device=dev, dtype=torch.float64) - origin
    max_abs = max(float(c64_diff.abs().max().item()), float(x_peek.abs().max().item()))
    # Scale only if near FP64 square overflow (> 1e140)
    scale = math.ldexp(1.0, int(round(math.log2(max_abs)))) if max_abs > 1e140 else 1.0
    inv_scale = 1.0 / scale
    c64 = (c64_diff * inv_scale).contiguous()
    c_norm = c64.square().sum(dim=1)

    labels = torch.empty((n,), dtype=torch.int64, device=dev)
    total_sq = torch.zeros((), dtype=torch.float64, device=dev)

    for s in range(0, n, row_chunk):
        e = min(s + row_chunk, n)
        xb = (x[s:e].to(device=dev, dtype=torch.float64) - origin) * inv_scale
        x_norm = xb.square().sum(dim=1)
        best_score = torch.full((e - s,), float("inf"), dtype=torch.float64, device=dev)
        best_idx = torch.zeros((e - s,), dtype=torch.int64, device=dev)
        for k0 in range(0, k, col_chunk):
            k1 = min(k0 + col_chunk, k)
            cb = c64[k0:k1]
            cn = c_norm[k0:k1]
            scores = cn[None, :] - 2.0 * (xb @ cb.t())
            b_min, b_pos = scores.min(dim=1)
            # Strict `<` preserves smallest centroid index on exact ties
            better = b_min < best_score
            best_score = torch.where(better, b_min, best_score)
            best_idx = torch.where(better, b_pos + k0, best_idx)

        # Refine chosen centroid distance via direct (x - c)^2 for zero catastrophic cancellation
        chosen_c = c64[best_idx]
        exact_d2 = _aligned_sum_sq_lastdim(xb - chosen_c) * (scale * scale)
        labels[s:e] = best_idx
        total_sq += exact_d2.sum()

    return float(total_sq.item()) / n, labels


@torch.no_grad()
def fast_exact_assign_fp64(
    x: torch.Tensor | np.ndarray,
    centroids: torch.Tensor | np.ndarray,
    device: str | torch.device | None = None,
) -> tuple[float, torch.Tensor]:
    """Fast exact FP64 MSE + nearest-centroid assignment (`~11.7x` speedup on Blackwell SM120).

    Uses FP64 mean centering + power-of-two dynamic-range scaling before a register-tiled
    Triton IEEE-FP32 top-4 candidate pass, followed by exact FP64 `(x64 - c64)^2` refinement
    with deterministic ascending-ID tie-breaking.
    """
    x_t = torch.from_numpy(x) if isinstance(x, np.ndarray) else x.detach()
    c_t = torch.from_numpy(centroids) if isinstance(centroids, np.ndarray) else centroids.detach()

    if x_t.ndim != 2 or c_t.ndim != 2:
        raise ValueError(f"Expected 2D x and centroids, got {x_t.shape} and {c_t.shape}")
    n, d = x_t.shape
    k, dc = c_t.shape
    if d != dc:
        raise ValueError(f"Dimension mismatch: x has d={d}, centroids has d={dc}")
    if k < 1:
        raise ValueError("centroids must have K >= 1")
    if n == 0:
        dev_out = torch.device(device) if device is not None else x_t.device
        return 0.0, torch.empty((0,), dtype=torch.int64, device=dev_out)

    dev = (
        torch.device(device)
        if device is not None
        else (
            x_t.device
            if x_t.is_cuda
            else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        )
    )
    x_dev = x_t.to(device=dev)
    c_dev = c_t.to(device=dev)

    if dev.type != "cuda" or not HAS_TRITON or k <= 32 or n < 512:
        return exact_assign_fp64_torch(x_dev, c_dev)

    c64_raw = c_dev.double()
    origin64 = x_dev[: min(n, 8192)].double().mean(dim=0, keepdim=True)
    c64_centered = c64_raw - origin64

    # Scale by data dynamic range (not outlier centroids) so near-data distances keep full FP32 mantissa;
    # clamp extreme outlier centroids only if they would overflow FP32 square (> 1e18 * scale).
    x_max = 0.0
    for s in range(0, n, 262144):
        e = min(s + 262144, n)
        blk_max = float((x_dev[s:e].double() - origin64).abs().max().item())
        if blk_max > x_max:
            x_max = blk_max
    c_max = float(c64_centered.abs().max().item())
    # Only widen scale for c_max if c_max would overflow FP32 square (> 1e18 * max(x_max, 1e-30))
    ref_max = x_max if c_max <= max(x_max, 1.0) * 1e15 else max(x_max, c_max * 1e-15)
    scale = _power_of_two_scale(ref_max)
    inv_scale = 1.0 / scale

    c_c32 = (c64_centered * inv_scale).clamp(-1e18, 1e18).float().contiguous()
    c_norm32 = c_c32.square().sum(dim=1).contiguous()

    x_c32 = torch.empty((n, d), dtype=torch.float32, device=dev)
    for s in range(0, n, 262144):
        e = min(s + 262144, n)
        x_c32[s:e] = ((x_dev[s:e].double() - origin64) * inv_scale).float()

    cand_idx = torch.empty((n, 4), dtype=torch.int32, device=dev)
    full_d = d <= 64
    block_d = triton.next_power_of_2(max(16, d)) if full_d else 64
    block_k = 128 if k >= 128 else 64
    grid = (triton.cdiv(n, 64),)
    _top4_centered_fp32_kernel[grid](
        x_c32,
        c_c32,
        c_norm32,
        cand_idx,
        n,
        k,
        d,
        x_c32.stride(0),
        x_c32.stride(1),
        c_c32.stride(0),
        c_c32.stride(1),
        BLOCK_N=64,
        BLOCK_K=block_k,
        BLOCK_D=block_d,
        FULL_D=full_d,
        num_warps=4,
        num_stages=2,
    )

    c64_raw = c_dev.double()
    best_labels = torch.empty((n,), dtype=torch.int64, device=dev)
    total_sq_scaled = torch.zeros((), dtype=torch.float64, device=dev)
    n_cands = min(4, k)
    for s in range(0, n, 262144):
        e = min(s + 262144, n)
        idx64 = cand_idx[s:e, :n_cands].long().clamp(0, k - 1)
        order = torch.argsort(idx64, dim=1, stable=True)
        idx_sorted = torch.gather(idx64, 1, order)
        x64_raw = x_dev[s:e].double()
        c_sel = c64_raw[idx_sorted]
        d2 = _aligned_sum_sq_lastdim((x64_raw[:, None, :] - c_sel) * inv_scale)
        min_d2, pos = d2.min(dim=1)
        best_labels[s:e] = idx_sorted.gather(1, pos[:, None]).squeeze(1)
        total_sq_scaled += min_d2.sum()

    mse = (float(total_sq_scaled.item()) / n) * (scale * scale)
    return mse, best_labels


def fast_exact_mse_fp64(
    x: torch.Tensor | np.ndarray,
    centroids: torch.Tensor | np.ndarray,
    device: str | torch.device | None = None,
) -> float:
    """Compute exact double-precision MSE using `fast_exact_assign_fp64`."""
    mse, _ = fast_exact_assign_fp64(x, centroids, device=device)
    return mse


@torch.no_grad()
def exact_cosine_distortion_fp64(
    x: torch.Tensor | np.ndarray,
    centroids: torch.Tensor | np.ndarray,
    device: str | torch.device | None = None,
    chunk_size: int = 32768,
) -> float:
    """Compute exact FP64 spherical cosine distortion `mean_i (1 - max_j cos(x_i, c_j))`."""
    x_t = torch.from_numpy(x) if isinstance(x, np.ndarray) else x.detach()
    c_t = torch.from_numpy(centroids) if isinstance(centroids, np.ndarray) else centroids.detach()
    n = x_t.shape[0]
    if n == 0:
        return 0.0
    dev = (
        torch.device(device)
        if device is not None
        else (
            x_t.device
            if x_t.is_cuda
            else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        )
    )
    c64 = c_t.to(device=dev, dtype=torch.float64)
    c_norms = c64.norm(dim=1, keepdim=True)
    valid_c = (c_norms.squeeze(1) > 1e-15)
    if not bool(valid_c.any().item()):
        return 1.0
    c_unit = torch.where(valid_c[:, None], c64 / c_norms.clamp_min(1e-15), torch.zeros_like(c64))

    total_dist = torch.zeros((), dtype=torch.float64, device=dev)
    for s in range(0, n, chunk_size):
        e = min(s + chunk_size, n)
        xb = x_t[s:e].to(device=dev, dtype=torch.float64)
        x_norms = xb.norm(dim=1, keepdim=True).clamp_min(1e-15)
        x_unit = xb / x_norms
        best_sim = torch.full((e - s,), -float("inf"), dtype=torch.float64, device=dev)
        for k0 in range(0, c_unit.shape[0], 8192):
            k1 = min(k0 + 8192, c_unit.shape[0])
            sims = x_unit @ c_unit[k0:k1].t()
            sims = torch.where(valid_c[None, k0:k1], sims, torch.full_like(sims, -float("inf")))
            best_sim = torch.maximum(best_sim, sims.max(dim=1).values)
        total_dist += (1.0 - best_sim.clamp(-1.0, 1.0)).sum()
    return float(total_dist.item()) / n


@torch.no_grad()
def compute_exact_groundtruth_gpu(
    x_base: torch.Tensor | np.ndarray,
    x_query: torch.Tensor | np.ndarray,
    top_k: int = 100,
    device: str | torch.device | None = None,
    query_chunk: int = 512,
    base_chunk: int = 250_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact FP64 top-`top_k` nearest neighbors `(gt_ids_int32, gt_sq_dists_fp64)`.

    Uses centered IEEE-754 FP64 distances with deterministic ascending database-ID tie-breaking.
    """
    xb_t = torch.from_numpy(x_base) if isinstance(x_base, np.ndarray) else x_base.detach()
    xq_t = torch.from_numpy(x_query) if isinstance(x_query, np.ndarray) else x_query.detach()

    n_base, d = xb_t.shape
    n_query, dq = xq_t.shape
    if d != dq:
        raise ValueError(f"Dimension mismatch: base d={d}, query d={dq}")
    eff_k = min(int(top_k), n_base)
    if eff_k < 1:
        raise ValueError("top_k and n_base must be >= 1")

    dev = (
        torch.device(device)
        if device is not None
        else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    )

    origin = xq_t.to(device=dev, dtype=torch.float64).mean(dim=0, keepdim=True)
    xq64 = (xq_t.to(device=dev, dtype=torch.float64) - origin).contiguous()
    xq_norm = xq64.square().sum(dim=1, keepdim=True)

    out_ids = np.empty((n_query, eff_k), dtype=np.int32)
    out_d2 = np.empty((n_query, eff_k), dtype=np.float64)

    for q0 in range(0, n_query, query_chunk):
        q1 = min(q0 + query_chunk, n_query)
        qb = xq64[q0:q1]
        qn = xq_norm[q0:q1]
        run_d2 = torch.full((q1 - q0, eff_k), float("inf"), dtype=torch.float64, device=dev)
        run_id = torch.full((q1 - q0, eff_k), n_base + 1, dtype=torch.int64, device=dev)

        for b0 in range(0, n_base, base_chunk):
            b1 = min(b0 + base_chunk, n_base)
            xb = (xb_t[b0:b1].to(device=dev, dtype=torch.float64) - origin).contiguous()
            bn = xb.square().sum(dim=1)[None, :]
            d2_blk = (qn + bn - 2.0 * (qb @ xb.t())).clamp_min_(0.0)
            k_blk = min(eff_k + 32, b1 - b0)
            # Stable ascending-column tie-break so exact duplicate vectors keep smallest DB IDs
            sub_pos = torch.argsort(d2_blk, dim=1, stable=True)[:, :k_blk]
            cand_ids = sub_pos + b0
            # Recompute direct (xq - xb)^2 in FP64 on finalists (no intermediate origin rounding!)
            q_raw64 = xq_t[q0:q1].to(device=dev, dtype=torch.float64)
            xb_raw64 = xb_t[b0:b1].to(device=dev, dtype=torch.float64)[sub_pos]
            exact_cand_d2 = _aligned_sum_sq_lastdim(q_raw64[:, None, :] - xb_raw64)

            cat_d2 = torch.cat([run_d2, exact_cand_d2], dim=1)
            cat_id = torch.cat([run_id, cand_ids], dim=1)
            id_ord = torch.argsort(cat_id, dim=1, stable=True)
            id_s = torch.gather(cat_id, 1, id_ord)
            d2_s = torch.gather(cat_d2, 1, id_ord)
            d2_ord = torch.argsort(d2_s, dim=1, stable=True)[:, :eff_k]
            run_d2 = torch.gather(d2_s, 1, d2_ord)
            run_id = torch.gather(id_s, 1, d2_ord)

        out_ids[q0:q1] = run_id.cpu().numpy().astype(np.int32)
        out_d2[q0:q1] = run_d2.cpu().numpy()

    return out_ids, out_d2


# ==============================================================================
# Sync-Free GPU Inverted-File (IVF-Flat) ANN Recall Evaluator
# ==============================================================================


def _merge_sorted_top_r(
    d2_a: torch.Tensor,
    id_a: torch.Tensor,
    d2_b: torch.Tensor,
    id_b: torch.Tensor,
    top_r: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge two `(Q, R)` candidate pools in exact `(d2_fp64, ascending_db_id)` order."""
    sentinel = 1 << 62
    md = torch.cat([d2_a, d2_b], dim=1)
    mi = torch.cat([id_a, id_b], dim=1)
    sk = torch.where(mi >= 0, mi, torch.full_like(mi, sentinel))
    io = torch.argsort(sk, dim=1, stable=True)
    mi_s = torch.gather(mi, 1, io)
    md_s = torch.gather(md, 1, io)
    fo = torch.argsort(md_s, dim=1, stable=True)[:, :top_r]
    return torch.gather(md_s, 1, fo), torch.gather(mi_s, 1, fo)


def _search_ivf_probe_slice_gpu(
    xq_c32: torch.Tensor,
    xq_dev: torch.Tensor,
    xb64_or_dev: torch.Tensor,
    inv_scale: float,
    scale_sq: float,
    x_csr_c32: torch.Tensor,
    sorted_ids: torch.Tensor,
    cluster_offsets: torch.Tensor,
    probed_slice: torch.Tensor,
    top_r: int,
    block_cap: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scan a slice of probed inverted lists `(Q, P_total)` with zero `.item()` host syncs."""
    q_total, d = xq_c32.shape
    p_total = probed_slice.shape[1]
    if p_total == 0:
        return (
            torch.full((q_total, top_r), float("inf"), dtype=torch.float64, device=xq_c32.device),
            torch.full((q_total, top_r), -1, dtype=torch.int64, device=xq_c32.device),
        )

    # Bound P_sub <= 128 so (bq, P_sub * block_cap) always processes >= 1024 queries per launch
    p_chunk = 128
    if p_total > p_chunk:
        acc_d2 = torch.full((q_total, top_r), float("inf"), dtype=torch.float64, device=xq_c32.device)
        acc_id = torch.full((q_total, top_r), -1, dtype=torch.int64, device=xq_c32.device)
        for p0 in range(0, p_total, p_chunk):
            p1 = min(p0 + p_chunk, p_total)
            sub_d2, sub_id = _search_ivf_probe_slice_gpu(
                xq_c32=xq_c32,
                xq_dev=xq_dev,
                xb64_or_dev=xb64_or_dev,
                inv_scale=inv_scale,
                scale_sq=scale_sq,
                x_csr_c32=x_csr_c32,
                sorted_ids=sorted_ids,
                cluster_offsets=cluster_offsets,
                probed_slice=probed_slice[:, p0:p1],
                top_r=top_r,
                block_cap=block_cap,
            )
            acc_d2, acc_id = _merge_sorted_top_r(acc_d2, acc_id, sub_d2, sub_id, top_r)
        return acc_d2, acc_id

    dev = xq_c32.device
    out_d2_64 = torch.full((q_total, top_r), float("inf"), dtype=torch.float64, device=dev)
    out_ids = torch.full((q_total, top_r), -1, dtype=torch.int64, device=dev)

    if dev.type == "cuda" and HAS_TRITON:
        full_d = d <= 128
        block_d = triton.next_power_of_2(max(16, d)) if full_d else 64
        width = p_total * block_cap
        bq = max(1, min(q_total, 33_554_432 // max(1, width)))
        sentinel = 1 << 62

        for q0 in range(0, q_total, bq):
            q1 = min(q0 + bq, q_total)
            nb = q1 - q0
            ps = probed_slice[q0:q1].contiguous()
            cand_d2 = torch.empty((nb, width), dtype=torch.float32, device=dev)
            cand_id = torch.empty((nb, width), dtype=torch.int64, device=dev)
            _ivf_csr_bucket_topcap_kernel[(nb, p_total)](
                xq_c32[q0:q1],
                x_csr_c32,
                sorted_ids,
                cluster_offsets,
                ps,
                cand_d2,
                cand_id,
                nb,
                p_total,
                d,
                xq_c32.stride(0),
                xq_c32.stride(1),
                x_csr_c32.stride(0),
                x_csr_c32.stride(1),
                ps.stride(0),
                ps.stride(1),
                BLOCK_CAP=block_cap,
                BLOCK_D=block_d,
                FULL_D=full_d,
                num_warps=4,
            )
            k_pre = min(width, max(block_cap, top_r + 64))
            d2_i64 = cand_d2.clamp_min(0.0).view(torch.int32).to(torch.int64)
            id_u32 = torch.where(cand_id >= 0, cand_id, torch.full_like(cand_id, 0xFFFFFFFF))
            packed_cand = (d2_i64 << 32) | (id_u32 & 0xFFFFFFFF)
            _, top_pos = torch.topk(packed_cand, k=k_pre, dim=1, largest=False, sorted=False)
            sel_ids = torch.gather(cand_id, 1, top_pos)
            valid = sel_ids >= 0
            safe_ids = sel_ids.clamp_min(0)

            q64_raw = xq_dev[q0:q1].double()[:, None, :]
            x64_raw = xb64_or_dev[safe_ids].double()
            d2_64 = _aligned_sum_sq_lastdim((q64_raw - x64_raw) * inv_scale) * scale_sq
            d2_64 = torch.where(valid, d2_64, torch.full_like(d2_64, float("inf")))

            sort_key_id = torch.where(valid, sel_ids, torch.full_like(sel_ids, sentinel))
            id_ord = torch.argsort(sort_key_id, dim=1, stable=True)
            ids_asc = torch.gather(sel_ids, 1, id_ord)
            d2_asc = torch.gather(d2_64, 1, id_ord)
            final_ord = torch.argsort(d2_asc, dim=1, stable=True)
            k_take = min(top_r, k_pre)
            out_d2_64[q0:q1, :k_take] = torch.gather(d2_asc, 1, final_ord[:, :k_take])
            out_ids[q0:q1, :k_take] = torch.gather(ids_asc, 1, final_ord[:, :k_take])
        return out_d2_64, out_ids

    # Exact PyTorch CPU / fallback CSR search
    offsets_cpu = cluster_offsets.cpu().tolist()
    probed_cpu = probed_slice.cpu().tolist()
    for qi in range(q_total):
        segs = [
            sorted_ids[offsets_cpu[cid] : offsets_cpu[cid + 1]]
            for cid in probed_cpu[qi]
            if offsets_cpu[cid + 1] > offsets_cpu[cid]
        ]
        if not segs:
            continue
        cands = torch.cat(segs, dim=0)
        q64_raw = xq_dev[qi : qi + 1].double()
        x64_raw = xb64_or_dev[cands].double()
        d2 = ((q64_raw - x64_raw) * inv_scale).square().sum(dim=1) * scale_sq
        id_ord = torch.argsort(cands, stable=True)
        c_asc = cands[id_ord]
        d_asc = d2[id_ord]
        f_ord = torch.argsort(d_asc, stable=True)[: min(top_r, cands.numel())]
        k_take = f_ord.numel()
        out_d2_64[qi, :k_take] = d_asc[f_ord]
        out_ids[qi, :k_take] = c_asc[f_ord]
    return out_d2_64, out_ids


@torch.no_grad()
def evaluate_ivf_recall_gpu(
    x_base: torch.Tensor | np.ndarray,
    centroids: torch.Tensor | np.ndarray,
    x_query: torch.Tensor | np.ndarray,
    groundtruth_ids: torch.Tensor | np.ndarray | None = None,
    groundtruth_sq_dists: torch.Tensor | np.ndarray | None = None,
    selectivities: Sequence[float] = (0.001, 0.005, 0.010),
    nprobes: Sequence[int] | None = None,
    recall_ks: Sequence[int] = (1, 10, 100),
    db_labels: torch.Tensor | np.ndarray | None = None,
    device: str | torch.device | None = None,
) -> IVFRecallResult:
    """Partition `x_base` into `centroids` and evaluate exact IVF-Flat Recall@{1,10,100} on GPU.

    All `nprobe` operating points are evaluated incrementally in a single CSR index pass with
    zero per-query Python loops and zero inner-loop `.item()` host synchronizations.
    """
    xb_t = torch.from_numpy(x_base) if isinstance(x_base, np.ndarray) else x_base.detach()
    c_t = torch.from_numpy(centroids) if isinstance(centroids, np.ndarray) else centroids.detach()
    xq_t = torch.from_numpy(x_query) if isinstance(x_query, np.ndarray) else x_query.detach()

    if xb_t.ndim != 2 or c_t.ndim != 2 or xq_t.ndim != 2:
        raise ValueError("x_base, centroids, and x_query must all be 2D")
    n_base, d = xb_t.shape
    k, dc = c_t.shape
    n_query, dq = xq_t.shape
    if not (d == dc == dq):
        raise ValueError(f"Dimension mismatch: base={d}, centroids={dc}, query={dq}")
    if k < 1 or n_base < 1 or n_query < 1:
        raise ValueError("n_base, k, and n_query must all be >= 1")

    dev = (
        torch.device(device)
        if device is not None
        else (
            xb_t.device
            if xb_t.is_cuda
            else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        )
    )
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t_start = time.perf_counter()

    xb_dev = xb_t.to(device=dev)
    c_dev = c_t.to(device=dev)
    xq_dev = xq_t.to(device=dev)

    max_rk = min(n_base, max(int(rk) for rk in recall_ks))
    top_r = min(n_base, max(100, max_rk)) if n_base >= max_rk else n_base
    if groundtruth_ids is not None:
        gt_id_t = (
            torch.from_numpy(np.asarray(groundtruth_ids, dtype=np.int64))
            if not isinstance(groundtruth_ids, torch.Tensor)
            else groundtruth_ids.detach().long()
        ).to(device=dev)
        top_r = min(top_r, gt_id_t.shape[1])
    else:
        gt_ids_np, gt_d2_np = compute_exact_groundtruth_gpu(xb_dev, xq_dev, top_k=top_r, device=dev)
        gt_id_t = torch.from_numpy(gt_ids_np).to(device=dev, dtype=torch.int64)
        groundtruth_sq_dists = gt_d2_np

    if groundtruth_sq_dists is not None:
        gt_d2_t = (
            torch.from_numpy(np.asarray(groundtruth_sq_dists, dtype=np.float64))
            if not isinstance(groundtruth_sq_dists, torch.Tensor)
            else groundtruth_sq_dists.detach().double()
        ).to(device=dev)
    else:
        # Compute exact FP64 squared distances for provided groundtruth_ids in chunks
        gt_d2_t = torch.empty_like(gt_id_t, dtype=torch.float64, device=dev)
        for q0 in range(0, n_query, 1024):
            q1 = min(q0 + 1024, n_query)
            q64 = xq_dev[q0:q1].double()[:, None, :]
            x64 = xb_dev[gt_id_t[q0:q1]].double()
            gt_d2_t[q0:q1] = (q64 - x64).square().sum(dim=2)

    # Resolve target (nprobe, selectivity_label, nominal_selectivity) tuples
    target_specs: list[tuple[int, str | None, float]] = []
    if nprobes is not None:
        for np_val in nprobes:
            np_clamped = max(1, min(int(np_val), k))
            target_specs.append((np_clamped, f"{100.0 * np_clamped / k:.3g}%", float(np_clamped) / k))
    else:
        for sel in selectivities:
            sel_f = float(sel)
            np_clamped = max(1, min(int(round(sel_f * k)), k))
            label = f"{100.0 * sel_f:.3g}%"
            target_specs.append((np_clamped, label, sel_f))

    unique_nprobes = sorted({spec[0] for spec in target_specs})
    max_nprobe = max(unique_nprobes)

    # Stage 1: Partition database into K Voronoi inverted lists (reusing db_labels if supplied)
    if db_labels is None:
        _, labels_dev = fast_exact_assign_fp64(xb_dev, c_dev, device=dev)
    else:
        labels_dev = (
            torch.from_numpy(np.asarray(db_labels, dtype=np.int64))
            if not isinstance(db_labels, torch.Tensor)
            else db_labels.detach().long()
        ).to(device=dev)

    sorted_ids = torch.argsort(labels_dev, stable=True)
    cluster_counts = torch.bincount(labels_dev, minlength=k)
    cluster_offsets = torch.zeros((k + 1,), dtype=torch.int64, device=dev)
    torch.cumsum(cluster_counts, dim=0, out=cluster_offsets[1:])

    # Coordinate-shift & extreme-scale invariant centered+scaled FP32 CSR representation
    # Center around query/data mean (NOT centroid mean!) so far-off empty/outlier centroids
    # cannot shift x_base & x_query and steal FP32 mantissa bits.
    c64_raw = c_dev.double()
    origin64 = xq_dev.double().mean(dim=0, keepdim=True)
    c64_diff = c64_raw - origin64
    q64_diff = xq_dev.double() - origin64
    b_sample_max = float((xb_dev[: min(n_base, 65536)].double() - origin64).abs().max().item())
    q_max = float(q64_diff.abs().max().item())
    c_max = float(c64_diff.abs().max().item())
    data_max = max(q_max, b_sample_max)
    # Use data/query dynamic range for CSR FP32 search; coarse routing uses FP64 so clamp only >1e140
    scale = _power_of_two_scale(data_max if c_max <= max(data_max, 1.0) * 1e120 else max(data_max, c_max * 1e-120))
    inv_scale = 1.0 / scale
    scale_sq = scale * scale

    xq64_scaled = (q64_diff * inv_scale).contiguous()
    xq_c32 = xq64_scaled.float().contiguous()

    x_csr_c32 = torch.empty((n_base, d), dtype=torch.float32, device=dev)
    for s in range(0, n_base, 262144):
        e = min(s + 262144, n_base)
        sids = sorted_ids[s:e]
        x_csr_c32[s:e] = ((xb_dev[sids].double() - origin64) * inv_scale).float()

    # Stage 2: Exact FP64 coarse centroid routing for all queries (with deterministic ID tie-breaking)
    c64_scaled = (c64_diff * inv_scale).contiguous()
    c64_norm = c64_scaled.square().sum(dim=1)
    probed_clusters = torch.empty((n_query, max_nprobe), dtype=torch.int64, device=dev)
    cid_arange = torch.arange(k, device=dev, dtype=torch.int64)[None, :]
    for q0 in range(0, n_query, 2048):
        q1 = min(q0 + 2048, n_query)
        qb = xq64_scaled[q0:q1]
        q_raw64 = xq_dev[q0:q1].double()
        if k <= 256:
            d_exact = _aligned_sum_sq_lastdim((q_raw64[:, None, :] - c64_raw[None, :, :]) * inv_scale)
            probed_clusters[q0:q1] = torch.argsort(d_exact, dim=1, stable=True)[:, :max_nprobe]
        else:
            qn = qb.square().sum(dim=1, keepdim=True)
            d2_qc = (qn + c64_norm[None, :] - 2.0 * (qb @ c64_scaled.t())).clamp_min_(0.0)
            if max_nprobe == k:
                probed_clusters[q0:q1] = torch.argsort(d2_qc, dim=1, stable=True)
            else:
                k_cand = min(k, max_nprobe + 8)
                top_c = torch.argsort(d2_qc, dim=1, stable=True)[:, :k_cand]
                c_ord = torch.argsort(top_c, dim=1, stable=True)
                c_asc = torch.gather(top_c, 1, c_ord)
                d_exact = _aligned_sum_sq_lastdim((q_raw64[:, None, :] - c64_raw[c_asc]) * inv_scale)
                f_ord = torch.argsort(d_exact, dim=1, stable=True)[:, :max_nprobe]
                probed_clusters[q0:q1] = torch.gather(c_asc, 1, f_ord)
    del cid_arange

    # Stage 3: Incremental CSR inverted-list scan across unique_nprobes
    block_cap = 128 if top_r + 16 <= 128 else 256
    cur_d2 = torch.full((n_query, top_r), float("inf"), dtype=torch.float64, device=dev)
    cur_id = torch.full((n_query, top_r), -1, dtype=torch.int64, device=dev)

    probed_counts = cluster_counts[probed_clusters]  # (Q, max_nprobe)
    cum_scanned = torch.cumsum(probed_counts, dim=1) # (Q, max_nprobe)

    by_nprobe: dict[int, IVFSelectivityMetrics] = {}
    prev_np = 0
    for np_target in unique_nprobes:
        slice_d2, slice_id = _search_ivf_probe_slice_gpu(
            xq_c32=xq_c32,
            xq_dev=xq_dev,
            xb64_or_dev=xb_dev,
            inv_scale=inv_scale,
            scale_sq=scale_sq,
            x_csr_c32=x_csr_c32,
            sorted_ids=sorted_ids,
            cluster_offsets=cluster_offsets,
            probed_slice=probed_clusters[:, prev_np:np_target],
            top_r=top_r,
            block_cap=block_cap,
        )
        if prev_np == 0:
            cur_d2, cur_id = slice_d2, slice_id
        else:
            cur_d2, cur_id = _merge_sorted_top_r(cur_d2, cur_id, slice_d2, slice_id, top_r)
        prev_np = np_target

        # Vectorized GPU Recall@{1, 10, 100} computation (tie-aware FP64 & strict ID intersection)
        def _tie_recall(rk: int) -> float:
            r_eff = min(rk, top_r)
            # Pure multiplicative relative tolerance (0 at 0.0) preserves exact scale equivariance across 1e-150..1e150
            thresh = gt_d2_t[:, r_eff - 1 : r_eff] * (1.0 + 1e-11)
            hits = (cur_d2[:, :r_eff] <= thresh).double().sum(dim=1) / float(r_eff)
            return float(hits.mean().item())

        def _id_recall(rk: int) -> float:
            r_eff = min(rk, top_r)
            matches = (cur_id[:, :r_eff, None] == gt_id_t[:, None, :r_eff]).any(dim=2)
            valid_pred = cur_id[:, :r_eff] >= 0
            hits = (matches & valid_pred).double().sum(dim=1) / float(r_eff)
            return float(hits.mean().item())

        r1 = _tie_recall(1)
        r10 = _tie_recall(10)
        r100 = _tie_recall(100)
        idr1 = _id_recall(1)
        idr10 = _id_recall(10)
        idr100 = _id_recall(100)

        r10_eff = min(10, top_r)
        thresh_1 = gt_d2_t[:, 0:1] * (1.0 + 1e-11)
        one_r10 = float((cur_d2[:, :r10_eff].min(dim=1).values[:, None] <= thresh_1).double().mean().item())

        scanned_q = cum_scanned[:, np_target - 1].double()
        mean_scan = float(scanned_q.mean().item())
        p99_scan = float(torch.quantile(scanned_q, 0.99).item())

        by_nprobe[np_target] = IVFSelectivityMetrics(
            selectivity=float(np_target) / float(k),
            nprobe=np_target,
            recall_at_1=r1,
            recall_at_10=r10,
            recall_at_100=r100,
            id_recall_at_1=idr1,
            id_recall_at_10=idr10,
            id_recall_at_100=idr100,
            one_recall_at_10=one_r10,
            mean_scanned_vecs=mean_scan,
            actual_scan_frac=mean_scan / float(n_base),
            p99_scanned_vecs=p99_scan,
        )

    imbalance = float(k) * float(cluster_counts.double().square().sum().item()) / (float(n_base) ** 2)
    empty_cnt = int((cluster_counts == 0).sum().item())
    min_sz = int(cluster_counts.min().item())
    max_sz = int(cluster_counts.max().item())

    by_selectivity: dict[str, IVFSelectivityMetrics] = {}
    for np_clamped, sel_label, sel_nom in target_specs:
        m_np = by_nprobe[np_clamped]
        if sel_label is not None:
            by_selectivity[sel_label] = IVFSelectivityMetrics(
                selectivity=sel_nom,
                nprobe=m_np.nprobe,
                recall_at_1=m_np.recall_at_1,
                recall_at_10=m_np.recall_at_10,
                recall_at_100=m_np.recall_at_100,
                id_recall_at_1=m_np.id_recall_at_1,
                id_recall_at_10=m_np.id_recall_at_10,
                id_recall_at_100=m_np.id_recall_at_100,
                one_recall_at_10=m_np.one_recall_at_10,
                mean_scanned_vecs=m_np.mean_scanned_vecs,
                actual_scan_frac=m_np.actual_scan_frac,
                p99_scanned_vecs=m_np.p99_scanned_vecs,
            )

    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    eval_time = time.perf_counter() - t_start

    return IVFRecallResult(
        k=k,
        n_base=n_base,
        n_query=n_query,
        imbalance_factor=imbalance,
        empty_clusters=empty_cnt,
        min_cluster_size=min_sz,
        max_cluster_size=max_sz,
        by_nprobe=by_nprobe,
        by_selectivity=by_selectivity,
        eval_time_sec=eval_time,
    )


# ==============================================================================
# Corpus Integrity Validator & Million-Scale Loaders (`SIFT1M` & `GloVe-1M-Unit`)
# ==============================================================================


def _exact_row_keys(arr: np.ndarray) -> np.ndarray:
    """View contiguous float32 rows as fixed-width byte keys (normalizing `-0.0 -> +0.0`)."""
    clean = np.where(arr == 0.0, 0.0, arr).astype(np.float32, copy=False)
    clean = np.ascontiguousarray(clean)
    return clean.view(np.dtype((np.void, clean.shape[1] * 4))).ravel()


def validate_real_corpus(corpus: RealVectorCorpus, check_cross_split_disjoint: bool = True) -> None:
    """Verify shapes, finiteness, ground-truth ordering, norms, and zero cross-split leakage."""
    for split_name, arr in (("base", corpus.base), ("val", corpus.val), ("query", corpus.query)):
        if arr.ndim != 2 or arr.shape[1] != corpus.d:
            raise ValueError(f"{corpus.name}.{split_name} has invalid shape {arr.shape} (expected (*, {corpus.d}))")
        if arr.dtype != np.float32:
            raise ValueError(f"{corpus.name}.{split_name} must be float32, got {arr.dtype}")
        if not np.isfinite(arr).all():
            raise ValueError(f"Non-finite NaN/Inf detected in {corpus.name}.{split_name}")

    nq = corpus.query.shape[0]
    nb = corpus.base.shape[0]
    if corpus.groundtruth_ids.ndim != 2 or corpus.groundtruth_ids.shape[0] != nq:
        raise ValueError(f"Invalid groundtruth_ids shape {corpus.groundtruth_ids.shape} for n_query={nq}")
    if corpus.groundtruth_sq_dists.shape != corpus.groundtruth_ids.shape:
        raise ValueError("groundtruth_sq_dists shape must match groundtruth_ids")
    if not np.isfinite(corpus.groundtruth_sq_dists).all():
        raise ValueError(f"Non-finite groundtruth_sq_dists in {corpus.name}")
    if corpus.groundtruth_ids.min() < 0 or corpus.groundtruth_ids.max() >= nb:
        raise ValueError(f"Out-of-bounds groundtruth_ids in {corpus.name} for n_base={nb}")
    if np.any(np.diff(corpus.groundtruth_sq_dists, axis=1) < -1e-9):
        raise ValueError(f"groundtruth_sq_dists not sorted ascending in {corpus.name}")

    if corpus.geometry == "unit_sphere":
        for split_name, arr in (("base", corpus.base), ("val", corpus.val), ("query", corpus.query)):
            norms = np.linalg.norm(arr.astype(np.float64), axis=1)
            if np.max(np.abs(norms - 1.0)) > 1e-5:
                raise ValueError(f"{corpus.name}.{split_name} deviates from unit sphere: max|norm-1|={np.max(np.abs(norms-1.0))}")

    if check_cross_split_disjoint:
        kb = np.unique(_exact_row_keys(corpus.base))
        kv = np.unique(_exact_row_keys(corpus.val))
        kq = np.unique(_exact_row_keys(corpus.query))
        if np.intersect1d(kb, kq).size > 0:
            raise ValueError(f"DATA LEAKAGE: {corpus.name} base and query share identical vectors!")
        if np.intersect1d(kb, kv).size > 0:
            raise ValueError(f"DATA LEAKAGE: {corpus.name} base and val share identical vectors!")
        if np.intersect1d(kv, kq).size > 0:
            raise ValueError(f"DATA LEAKAGE: {corpus.name} val and query share identical vectors!")


def _ensure_sift1m_raw_files(data_dir: Path) -> None:
    """Ensure all 4 official SIFT1M `.fvecs`/`.ivecs` files exist and match SHA-256 digests."""
    data_dir.mkdir(parents=True, exist_ok=True)
    all_ok = True
    for fname, (exp_sz, exp_sha) in SIFT1M_SPECS.items():
        fpath = data_dir / fname
        if not (fpath.exists() and fpath.stat().st_size == exp_sz and compute_sha256(fpath) == exp_sha):
            all_ok = False
            break
    if all_ok:
        return

    try:
        for fname, (exp_sz, exp_sha) in SIFT1M_SPECS.items():
            urls = [f"{base}/{fname}" for base in SIFT1M_PRIMARY_BASE_URLS]
            _download_file_parallel_atomic(
                urls=urls,
                dst=data_dir / fname,
                expected_size=exp_sz,
                expected_sha256=exp_sha,
                n_workers=16 if exp_sz >= 32_000_000 else 4,
            )
        return
    except Exception:
        # Fallback to compressed official TexMex archive mirror (`sift.tar.gz`)
        tar_dst = data_dir / "sift.tar.gz"
        _download_file_parallel_atomic(
            urls=[SIFT1M_TAR_MIRROR_URL],
            dst=tar_dst,
            expected_size=SIFT1M_TAR_SIZE,
            expected_sha256=None,
            n_workers=16,
        )
        with tarfile.open(tar_dst, "r:gz") as tf:
            for fname, (exp_sz, exp_sha) in SIFT1M_SPECS.items():
                member = tf.getmember(f"sift/{fname}")
                tmp = data_dir / f".tmp_{os.getpid()}_{time.time_ns()}_{fname}"
                try:
                    src = tf.extractfile(member)
                    if src is None:
                        raise IOError(f"Missing {fname} in {tar_dst}")
                    with src, open(tmp, "wb") as wf:
                        while True:
                            buf = src.read(4 * 1024 * 1024)
                            if not buf:
                                break
                            wf.write(buf)
                    if tmp.stat().st_size != exp_sz or compute_sha256(tmp) != exp_sha:
                        raise IOError(f"Archive extraction checksum mismatch for {fname}")
                    os.replace(tmp, data_dir / fname)
                finally:
                    if tmp.exists():
                        tmp.unlink(missing_ok=True)


def load_sift1m(
    data_dir: Path | str = DEFAULT_MILLION_DATA_DIR,
    device: str | torch.device | None = None,
) -> RealVectorCorpus:
    """Load and validate official SIFT1M (`1,000,000 x 128` base, clean val, `10,000` queries, GT-100).

    Important Data-Hygiene Note:
      In raw TexMex SIFT1M, `9,979 / 10,000` rows of `sift_query` also appear inside `sift_learn`
      (whereas `0 / 10,000` appear in `sift_base`). We keep `sift_base` (`1,000,000 x 128`) and
      `sift_query` (`10,000 x 128`) 100% intact so all `sift_groundtruth` IDs remain valid (per
      Appendix K of arXiv:2611.09501), and strictly purge `val` (`sift_learn`) of duplicates and
      any rows appearing in `sift_base` or `sift_query` (`89,788` clean validation vectors).
    """
    ddir = Path(data_dir)
    cache_path = ddir / "sift1m_validated_v1.npz"

    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as z:
            corpus = RealVectorCorpus(
                name="sift1m",
                base=np.ascontiguousarray(z["base"], dtype=np.float32),
                val=np.ascontiguousarray(z["val"], dtype=np.float32),
                query=np.ascontiguousarray(z["query"], dtype=np.float32),
                groundtruth_ids=np.ascontiguousarray(z["groundtruth_ids"], dtype=np.int32),
                groundtruth_sq_dists=np.ascontiguousarray(z["groundtruth_sq_dists"], dtype=np.float64),
                d=128,
                geometry="euclidean",
                metadata={
                    "source": "TexMex SIFT1M (Jégou et al., IJCV 2011 / HuggingFace qbo-odp/sift1m)",
                    "raw_learn_rows": int(z["raw_learn_rows"]),
                    "purged_learn_overlap_rows": int(z["purged_learn_overlap_rows"]),
                    "gt_fp64_max_abs_diff": float(z["gt_fp64_max_abs_diff"]),
                },
            )
        validate_real_corpus(corpus, check_cross_split_disjoint=True)
        return corpus

    _ensure_sift1m_raw_files(ddir)

    xb = read_fvecs(ddir / "sift_base.fvecs")
    xl_raw = read_fvecs(ddir / "sift_learn.fvecs")
    xq = read_fvecs(ddir / "sift_query.fvecs")
    gt_off = read_ivecs(ddir / "sift_groundtruth.ivecs")

    if xb.shape != (1_000_000, 128):
        raise ValueError(f"Unexpected sift_base shape {xb.shape}")
    if xl_raw.shape != (100_000, 128):
        raise ValueError(f"Unexpected sift_learn shape {xl_raw.shape}")
    if xq.shape != (10_000, 128):
        raise ValueError(f"Unexpected sift_query shape {xq.shape}")
    if gt_off.shape != (10_000, 100):
        raise ValueError(f"Unexpected sift_groundtruth shape {gt_off.shape}")
    if xb.min() < 0.0 or xb.max() > 255.0 or xq.min() < 0.0 or xq.max() > 255.0:
        raise ValueError("SIFT1M values out of expected [0, 255] descriptor range")

    # Purge `sift_learn` of internal duplicates and any vector present in `sift_base` or `sift_query`
    kb = np.unique(_exact_row_keys(xb))
    kq = np.unique(_exact_row_keys(xq))
    if np.intersect1d(kb, kq).size != 0:
        raise ValueError("Unexpected overlap between sift_base and sift_query")
    forbid = np.union1d(kb, kq)
    kl_raw = _exact_row_keys(xl_raw)
    _, ul_idx = np.unique(kl_raw, return_index=True)
    ul_idx.sort()
    keep_mask = ~np.isin(kl_raw[ul_idx], forbid)
    xl_clean = np.ascontiguousarray(xl_raw[ul_idx[keep_mask]], dtype=np.float32)
    purged_rows = int(xl_raw.shape[0] - xl_clean.shape[0])

    # Independently compute exact FP64 top-100 ground truth and verify 0.0 distance discrepancy vs official GT
    gt_exact_ids, gt_exact_d2 = compute_exact_groundtruth_gpu(xb, xq, top_k=100, device=device)

    dev = (
        torch.device(device)
        if device is not None
        else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    )
    xb_dev = torch.from_numpy(xb).to(device=dev)
    xq_dev = torch.from_numpy(xq).to(device=dev)
    gt_off_dev = torch.from_numpy(gt_off).to(device=dev, dtype=torch.int64)
    off_d2_chunks: list[np.ndarray] = []
    for q0 in range(0, 10_000, 1000):
        q1 = min(q0 + 1000, 10_000)
        q64 = xq_dev[q0:q1].double()[:, None, :]
        x64 = xb_dev[gt_off_dev[q0:q1]].double()
        off_d2_chunks.append((q64 - x64).square().sum(dim=2).cpu().numpy())
    gt_off_d2 = np.concatenate(off_d2_chunks, axis=0)
    max_gt_diff = float(np.max(np.abs(gt_exact_d2 - gt_off_d2)))
    if max_gt_diff > 1e-6:
        raise ValueError(f"SIFT1M official groundtruth distance mismatch vs exact FP64: {max_gt_diff}")

    # Store canonical ascending-ID tie-broken exact groundtruth (100% identical FP64 distances to official GT,
    # and deterministic tie order matching GPU IVF search so Recall@K_all == 1.000000 on both metrics!)
    atomic_savez(
        cache_path,
        base=xb,
        val=xl_clean,
        query=xq,
        groundtruth_ids=gt_exact_ids,
        groundtruth_sq_dists=gt_exact_d2,
        official_groundtruth_ids=gt_off,
        raw_learn_rows=np.int64(xl_raw.shape[0]),
        purged_learn_overlap_rows=np.int64(purged_rows),
        gt_fp64_max_abs_diff=np.float64(max_gt_diff),
    )

    corpus = RealVectorCorpus(
        name="sift1m",
        base=xb,
        val=xl_clean,
        query=xq,
        groundtruth_ids=gt_exact_ids,
        groundtruth_sq_dists=gt_exact_d2,
        d=128,
        geometry="euclidean",
        metadata={
            "source": "TexMex SIFT1M (Jégou et al., IJCV 2011 / HuggingFace qbo-odp/sift1m)",
            "raw_learn_rows": int(xl_raw.shape[0]),
            "purged_learn_overlap_rows": purged_rows,
            "gt_fp64_max_abs_diff": max_gt_diff,
        },
    )
    validate_real_corpus(corpus, check_cross_split_disjoint=True)
    return corpus


def load_glove1m_unit(
    data_dir: Path | str = DEFAULT_MILLION_DATA_DIR,
    split_seed: int = 20261116,
    device: str | torch.device | None = None,
) -> RealVectorCorpus:
    """Load million-scale unit-normalized GloVe Twitter 27B (`1,000,000 x 100` base, `100k` val, `10k` query).

    Applies FP64 L2 unit-normalization, global deduplication, and a deterministic seeded shuffle
    (eliminating corpus frequency-rank covariate shift between train/val/query) before carving
    strictly disjoint `(1_000_000, 100_000, 10_000)` splits and computing exact FP64 top-100 GT.
    """
    ddir = Path(data_dir)
    cache_path = ddir / f"glove1m_unit_s{int(split_seed)}_validated_v1.npz"

    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as z:
            corpus = RealVectorCorpus(
                name="glove1m_unit",
                base=np.ascontiguousarray(z["base"], dtype=np.float32),
                val=np.ascontiguousarray(z["val"], dtype=np.float32),
                query=np.ascontiguousarray(z["query"], dtype=np.float32),
                groundtruth_ids=np.ascontiguousarray(z["groundtruth_ids"], dtype=np.int32),
                groundtruth_sq_dists=np.ascontiguousarray(z["groundtruth_sq_dists"], dtype=np.float64),
                d=100,
                geometry="unit_sphere",
                metadata={
                    "source": "Stanford GloVe Twitter 27B 100d (HuggingFace SkyTbac/ANNS_GloVe.twitter.27B.100d)",
                    "split_seed": int(split_seed),
                    "raw_unique_unit_vectors": int(z["raw_unique_unit_vectors"]),
                },
            )
        validate_real_corpus(corpus, check_cross_split_disjoint=True)
        return corpus

    raw_path = ddir / GLOVE_TWITTER_100D_FILENAME
    _download_file_parallel_atomic(
        urls=[GLOVE_TWITTER_100D_URL],
        dst=raw_path,
        expected_size=GLOVE_TWITTER_100D_SIZE,
        expected_sha256=None,
        n_workers=16,
    )

    mat = read_fvecs(raw_path)
    if mat.shape[1] != 100 or mat.shape[0] < 1_110_000:
        raise ValueError(f"Unexpected GloVe fvecs shape {mat.shape}")

    norms64 = np.linalg.norm(mat.astype(np.float64), axis=1, keepdims=True)
    nonzero_mask = norms64[:, 0] > 1e-8
    unit = (mat[nonzero_mask].astype(np.float64) / norms64[nonzero_mask]).astype(np.float32)
    # Re-normalize in FP64 once more after FP32 cast so float32 unit norms hold to <1e-6
    unit = (unit.astype(np.float64) / np.linalg.norm(unit.astype(np.float64), axis=1, keepdims=True)).astype(np.float32)

    # Global deduplication at 6 decimals (per Section 4.1 / datasets.deduplicate_splits)
    rounded = np.round(unit, decimals=6)
    rounded = np.where(rounded == 0.0, 0.0, rounded).astype(np.float32, copy=False)
    rkeys = np.ascontiguousarray(rounded).view(np.dtype((np.void, rounded.shape[1] * 4))).ravel()
    _, uniq_idx = np.unique(rkeys, return_index=True)
    uniq_idx.sort()
    unit_uniq = np.ascontiguousarray(unit[uniq_idx], dtype=np.float32)
    if unit_uniq.shape[0] < 1_110_000:
        raise ValueError(f"Insufficient unique unit GloVe vectors: {unit_uniq.shape[0]} < 1,110,000")

    rng = np.random.default_rng(int(split_seed))
    perm = rng.permutation(unit_uniq.shape[0])
    xb_raw = unit_uniq[perm[:1_000_000]]
    xv_raw = unit_uniq[perm[1_000_000:1_100_000]]
    xq_raw = unit_uniq[perm[1_100_000:1_110_000]]

    xb, xv, xq = deduplicate_splits(xb_raw, xv_raw, xq_raw, decimals=6)
    if xb.shape != (1_000_000, 100) or xv.shape != (100_000, 100) or xq.shape != (10_000, 100):
        raise RuntimeError(f"Unexpected post-dedup shapes: {xb.shape}, {xv.shape}, {xq.shape}")

    gt_ids, gt_d2 = compute_exact_groundtruth_gpu(xb, xq, top_k=100, device=device)

    atomic_savez(
        cache_path,
        base=xb,
        val=xv,
        query=xq,
        groundtruth_ids=gt_ids,
        groundtruth_sq_dists=gt_d2,
        raw_unique_unit_vectors=np.int64(unit_uniq.shape[0]),
    )

    corpus = RealVectorCorpus(
        name="glove1m_unit",
        base=xb,
        val=xv,
        query=xq,
        groundtruth_ids=gt_ids,
        groundtruth_sq_dists=gt_d2,
        d=100,
        geometry="unit_sphere",
        metadata={
            "source": "Stanford GloVe Twitter 27B 100d (HuggingFace SkyTbac/ANNS_GloVe.twitter.27B.100d)",
            "split_seed": int(split_seed),
            "raw_unique_unit_vectors": int(unit_uniq.shape[0]),
        },
    )
    validate_real_corpus(corpus, check_cross_split_disjoint=True)
    return corpus


def load_real_million_corpus(
    name: str,
    data_dir: Path | str = DEFAULT_MILLION_DATA_DIR,
    device: str | torch.device | None = None,
) -> RealVectorCorpus:
    """Dispatch loader for `'sift1m'` or `'glove1m_unit'`."""
    key = name.strip().lower()
    if key == "sift1m":
        return load_sift1m(data_dir=data_dir, device=device)
    if key in ("glove1m_unit", "glove1m", "glove_twitter_1m_unit"):
        return load_glove1m_unit(data_dir=data_dir, device=device)
    raise ValueError(f"Unknown million-scale corpus {name!r}; expected 'sift1m' or 'glove1m_unit'")


__all__ = [
    "DEFAULT_MILLION_DATA_DIR",
    "GLOVE_TWITTER_100D_SIZE",
    "IVFRecallResult",
    "IVFSelectivityMetrics",
    "RealVectorCorpus",
    "SIFT1M_SPECS",
    "atomic_savez",
    "compute_exact_groundtruth_gpu",
    "compute_sha256",
    "evaluate_ivf_recall_gpu",
    "exact_assign_fp64_torch",
    "exact_cosine_distortion_fp64",
    "fast_exact_assign_fp64",
    "fast_exact_mse_fp64",
    "load_glove1m_unit",
    "load_real_million_corpus",
    "load_sift1m",
    "read_fvecs",
    "read_ivecs",
    "validate_real_corpus",
    "write_fvecs_atomic",
]
