#!/usr/bin/env python3
"""Fused Product Quantization (`ApexPQ`) & 2-Stage Residual VQ (`ApexRVQ`) Benchmark Runner.

Evaluates `ApexPQ` (Fused 3D Batched Multi-Subspace `ApexEMeans` + `CUDAGraph` + GPU LUT-ADC)
and `ApexRVQ` (2-Stage Coarse-to-Residual `ApexEMeans` + `CroissAttn`-FP8 + Exact Beam/LUT ADC)
against `faiss.IndexPQ`, `Lloyd@25` (both standard and 3D-batched GPU), `EMeans`, `SMeans`,
`EMeansPP`, `HybridSEMeans`, and `UltraEMeans` across **4 genuine real-world vector corpora**
at matched epoch budgets `T in {1.0, 2.0, 5.0}`:

  1. `sift1m`               (`1,000,000 x 128`): `PQ16x8` (`16 B/vec`, `32x`) + `2-Stage RVQ (4096, 4096)`
  2. `glove1m_unit`         (`1,000,000 x 100`): `PQ10x8` (`10 B/vec`, `40x`) + `2-Stage RVQ (4096, 4096)`
  3. `cifar10_convnext_768` (`   50,000 x 768`): `PQ24x8` (`24 B/vec`, `128x`) + `2-Stage RVQ (1024, 1024)`
  4. `agnews_tfidf_svd_256` (`  109,738 x 256`): `PQ16x8` (`16 B/vec`, `64x`) + `2-Stage RVQ (1024, 1024)`

Strict Hygiene (`AGENTS.md` Rules 1–3):
  - Zero train/query leakage: codebooks, origins, scales, and LUT cross-tables fit ONLY on
    `corpus.base`; Reconstruction MSE / SQNR (dB) and ADC `Recall@1` / `1-Recall@10` / `10-Recall@10`
    evaluated strictly on held-out `corpus.query` against exact `corpus.groundtruth_ids`.
  - Full untimed Triton + CUDAGraph warmup before timed seeds + automated Seed-0 cold-JIT spike gate.
  - 100% single-SKU TDP pinning across `acorn.stratabaremetal.com:0,1 + cashew.stratabaremetal.com:0,1`
    (`4x 600W Server Blackwell`), live `nvidia-smi` idle verification before dispatch, hard block on
    `walnut`, `ripper`, `loki`, `work`, and `tmux` session `0`, and `PID+ns` + `os.fsync` + `os.replace`
    atomic writes (`allow_nan=False`).
"""

from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
import time
from typing import Any

# Isolate Triton & temp caches strictly under `/home` (never `/tmp` on small OS root partitions)
_LOCAL_HOME_TMP = Path("/home/catid/.cache/emeans_tmp")
_LOCAL_HOME_TMP.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TMPDIR", str(_LOCAL_HOME_TMP))
if "TRITON_CACHE_DIR" not in os.environ:
    _vis = os.environ.get("CUDA_VISIBLE_DEVICES", "all").replace(",", "_")
    _tc = _LOCAL_HOME_TMP / f"triton_cache_pq_rvq_u{os.getuid()}_g{_vis}"
    _tc.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(_tc)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import stats
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from emeans.apex_emeans import warmup_apex_kernels
from emeans.kernels import assign_points
from emeans.pq_rvq import (
    ApexPQ,
    ApexRVQ,
    evaluate_adc_recall,
    warmup_pq_rvq_kernels,
)
from emeans.real_corpora import (
    DEFAULT_MILLION_DATA_DIR,
    RealVectorCorpus,
    compute_exact_groundtruth_gpu,
    validate_real_corpus,
)
from emeans.realistic_benchmarks import (
    BENCHMARK_SPEC_BY_NAME,
    DEFAULT_PHASE5_DATA_DIR,
    load_realistic_benchmark,
)
from emeans.seeding import HAS_TRITON as SEED_HAS_TRITON

if SEED_HAS_TRITON:
    from emeans.seeding import _local_greedy_kpp_kernel

try:
    from emeans.croiss_fp8 import warmup_croiss_fp8_kernels
except Exception:
    warmup_croiss_fp8_kernels = None


OUTPUT_DIR = ROOT / "outputs" / "pq_rvq"
REMOTE_WORKDIR = "/home/catid/emeans_runs/pq-rvq-bench"
REMOTE_PYTHON = "/home/catid/emeans_env/.venv/bin/python"
REMOTE_P4_DATA = "/home/catid/emeans_runs/real-scale/outputs/phase4_million_scale/data"
REMOTE_P5_DATA = "/home/catid/emeans_runs/apex-bench/outputs/phase5_dominance/data"

FORBIDDEN_HOSTS = {"walnut", "walnut.stratabaremetal.com", "ripper", "loki", "work"}

EPOCH_BUDGETS: tuple[float, ...] = (1.0, 2.0, 5.0)

# Wave decomposition (`#201`): baselines vs apex
# Note (`#233`): `faiss_pq` sets `max_points_per_centroid = ceil(N / Ks)` (100% de-confounded full-N training);
# `faiss_pq_sub` runs stock FAISS `max_points_per_centroid = 256` (`65,536`-row subsample on `Ks = 256`).
PQ_ANCHOR_BASELINE_ARMS: tuple[str, ...] = ("faiss_pq", "faiss_pq_sub", "lloyd_25", "lloyd_25_batched")
PQ_STREAM_BASELINE_ARMS: tuple[str, ...] = (
    "emeans",
    "smeans",
    "emeans_pp",
    "hybrid_semeans",
    "ultra_emeans",
)
PQ_STREAM_APEX_ARMS: tuple[str, ...] = (
    "apex_emeans_subspace",
    "apex_emeans_fp8",
    "apex_emeans",
)

RVQ_ANCHOR_BASELINE_ARMS: tuple[str, ...] = ("lloyd_25",)
RVQ_STREAM_BASELINE_ARMS: tuple[str, ...] = (
    "emeans",
    "smeans",
    "emeans_pp",
    "hybrid_semeans",
    "ultra_emeans",
)
RVQ_STREAM_APEX_ARMS: tuple[str, ...] = (
    "apex_emeans_bf16",
    "apex_emeans_fp8",
    "apex_emeans",
)

ARM_STYLE: dict[str, tuple[str, str, str, str]] = {
    # arm_id: (Display Label, Color, Marker, Linestyle)
    "faiss_pq": ("FAISS IndexPQ (Full N, 25 it)", "#444444", "P", ":"),
    "faiss_pq_sub": ("FAISS IndexPQ (256/c sub, 25 it)", "#8c564b", "X", ":"),
    "lloyd_25": ("Lloyd @ 25 ep", "#222222", "*", "--"),
    "lloyd_25_batched": ("Batched GPU Lloyd@25", "#7f7f7f", "d", "-."),
    "emeans": ("e-means (Meta FAIR)", "#1f77b4", "o", "-"),
    "smeans": ("s-means (Triton)", "#b8860b", "^", "-."),
    "emeans_pp": ("e-means++", "#ff7f0e", "s", "-"),
    "hybrid_semeans": ("HybridSEMeans", "#2ca02c", "v", "--"),
    "ultra_emeans": ("UltraEMeans", "#17becf", "h", "-"),
    "apex_emeans_subspace": ("ApexEMeans (Per-Subspace)", "#9467bd", "d", "--"),
    "apex_emeans_bf16": ("Apex (BF16+Graph)", "#e377c2", "p", "--"),
    "apex_emeans_fp8": ("Apex (CroissFP8+Graph)", "#d62728", "H", "-"),
    "apex_emeans": ("ApexPQ / ApexRVQ (Flagship)", "#8b0000", "D", "-"),
}


# ==============================================================================
# Cell & 4-GPU Single-SKU 600W Server Blackwell Fleet Specifications
# ==============================================================================


@dataclass(frozen=True)
class CompressionCellSpec:
    """Specification for one PQ or 2-Stage RVQ benchmark cell on a real corpus."""

    cell_id: str
    mode: str           # "pq" or "rvq"
    corpus_name: str
    short_title: str
    codec_label: str
    n_base_limit: int | None
    n_query_limit: int | None
    d: int
    m: int              # PQ subspaces (0 for RVQ)
    ks: int             # PQ codebook size per subspace (0 for RVQ)
    k1: int             # RVQ Stage-1 coarse centroids (0 for PQ)
    k2: int             # RVQ Stage-2 residual centroids (0 for PQ)
    adc_beam: int | None
    geometry: str
    sku_tier: str


@dataclass(frozen=True)
class FleetShardSpec:
    """Single-GPU shard assignment across `acorn:0,1 + cashew:0,1` (`4x 600W Server Blackwell`)."""

    shard_idx: int
    host: str
    gpu_id: int
    session: str
    sku_tier: str
    tasks: tuple[tuple[str, tuple[int, ...]], ...]  # ((cell_id, seed_offsets), ...)


def build_compression_cells(smoke: bool = False) -> list[CompressionCellSpec]:
    """Build the 8 canonical compression cells (4 PQ + 4 RVQ) or fast real-data smoke slices."""
    if smoke:
        return [
            CompressionCellSpec(
                "smoke_pq_sift1m", "pq", "sift1m", "SIFT1M Smoke", "PQ16x6",
                2560, 128, 128, 16, 64, 0, 0, None, "euclidean", "smoke_local",
            ),
            CompressionCellSpec(
                "smoke_pq_glove1m", "pq", "glove1m_unit", "GloVe-1M Smoke", "PQ10x6",
                2560, 128, 100, 10, 64, 0, 0, None, "unit_sphere", "smoke_local",
            ),
            CompressionCellSpec(
                "smoke_pq_convnext", "pq", "cifar10_convnext_768", "ConvNeXt-768 Smoke", "PQ24x6",
                2560, 128, 768, 24, 64, 0, 0, None, "euclidean", "smoke_local",
            ),
            CompressionCellSpec(
                "smoke_pq_agnews", "pq", "agnews_tfidf_svd_256", "AG-News-256 Smoke", "PQ16x6",
                2560, 128, 256, 16, 64, 0, 0, None, "euclidean", "smoke_local",
            ),
            CompressionCellSpec(
                "smoke_rvq_sift1m", "rvq", "sift1m", "SIFT1M Smoke", "RVQ 256×256",
                2048, 128, 128, 0, 0, 256, 256, 32, "euclidean", "smoke_local",
            ),
            CompressionCellSpec(
                "smoke_rvq_glove1m", "rvq", "glove1m_unit", "GloVe-1M Smoke", "RVQ 256×256",
                2048, 128, 100, 0, 0, 256, 256, 32, "unit_sphere", "smoke_local",
            ),
            CompressionCellSpec(
                "smoke_rvq_convnext", "rvq", "cifar10_convnext_768", "ConvNeXt-768 Smoke", "RVQ 256×256",
                2048, 128, 768, 0, 0, 256, 256, 32, "euclidean", "smoke_local",
            ),
            CompressionCellSpec(
                "smoke_rvq_agnews", "rvq", "agnews_tfidf_svd_256", "AG-News-256 Smoke", "RVQ 256×256",
                2048, 128, 256, 0, 0, 256, 256, 32, "euclidean", "smoke_local",
            ),
        ]

    return [
        # 4 Product Quantization (`PQ`) Cells
        CompressionCellSpec(
            "pq_sift1m_16x8", "pq", "sift1m", "SIFT1M (1M×128)", "PQ16×8 (16 B/vec, 32×)",
            None, None, 128, 16, 256, 0, 0, None, "euclidean", "server_600w",
        ),
        CompressionCellSpec(
            "pq_glove1m_10x8", "pq", "glove1m_unit", "GloVe-1M (1M×100)", "PQ10×8 (10 B/vec, 40×)",
            None, None, 100, 10, 256, 0, 0, None, "unit_sphere", "server_600w",
        ),
        CompressionCellSpec(
            "pq_convnext_24x8", "pq", "cifar10_convnext_768", "ConvNeXt-Tiny (50K×768)", "PQ24×8 (24 B/vec, 128×)",
            None, None, 768, 24, 256, 0, 0, None, "euclidean", "server_600w",
        ),
        CompressionCellSpec(
            "pq_agnews_16x8", "pq", "agnews_tfidf_svd_256", "AG-News SVD (110K×256)", "PQ16×8 (16 B/vec, 64×)",
            None, None, 256, 16, 256, 0, 0, None, "euclidean", "server_600w",
        ),
        # 4 Two-Stage Residual Vector Quantization (`RVQ`) Cells
        CompressionCellSpec(
            "rvq_sift1m_4096", "rvq", "sift1m", "SIFT1M (1M×128)", "2-Stage RVQ (4096², 3 B/vec)",
            None, None, 128, 0, 0, 4096, 4096, 128, "euclidean", "server_600w",
        ),
        CompressionCellSpec(
            "rvq_glove1m_4096", "rvq", "glove1m_unit", "GloVe-1M (1M×100)", "2-Stage RVQ (4096², 3 B/vec)",
            None, None, 100, 0, 0, 4096, 4096, 128, "unit_sphere", "server_600w",
        ),
        CompressionCellSpec(
            "rvq_convnext_1024", "rvq", "cifar10_convnext_768", "ConvNeXt-Tiny (50K×768)", "2-Stage RVQ (1024², 2.5 B/vec)",
            None, None, 768, 0, 0, 1024, 1024, 64, "euclidean", "server_600w",
        ),
        CompressionCellSpec(
            "rvq_agnews_1024", "rvq", "agnews_tfidf_svd_256", "AG-News SVD (110K×256)", "2-Stage RVQ (1024², 2.5 B/vec)",
            None, None, 256, 0, 0, 1024, 1024, 64, "euclidean", "server_600w",
        ),
    ]


def build_fleet_shards(num_seeds: int) -> list[FleetShardSpec]:
    """Partition all 8 cells across the 4 identical 600W Server Blackwell GPUs (`acorn:0,1 + cashew:0,1`).

    Every cell runs 100% on `NVIDIA RTX PRO 6000 Blackwell Server Edition (600W)` (`0` cross-SKU cells),
    and every `(cell_id, seed)` executes all arms and all `T in {1, 2, 5}` on ONE physical GPU.
    """
    w2_0 = tuple(i for i in range(num_seeds) if (i % 2) == 0)
    w2_1 = tuple(i for i in range(num_seeds) if (i % 2) == 1)

    shards = [
        FleetShardSpec(
            0,
            "acorn.stratabaremetal.com",
            0,
            "emeans_pqrvq_s00",
            "server_600w",
            (
                ("pq_sift1m_16x8", w2_0),
                ("rvq_sift1m_4096", w2_0),
                ("pq_convnext_24x8", w2_0),
                ("rvq_convnext_1024", w2_0),
            ),
        ),
        FleetShardSpec(
            1,
            "acorn.stratabaremetal.com",
            1,
            "emeans_pqrvq_s01",
            "server_600w",
            (
                ("pq_sift1m_16x8", w2_1),
                ("rvq_sift1m_4096", w2_1),
                ("pq_convnext_24x8", w2_1),
                ("rvq_convnext_1024", w2_1),
            ),
        ),
        FleetShardSpec(
            2,
            "cashew.stratabaremetal.com",
            0,
            "emeans_pqrvq_s02",
            "server_600w",
            (
                ("pq_glove1m_10x8", w2_0),
                ("rvq_glove1m_4096", w2_0),
                ("pq_agnews_16x8", w2_0),
                ("rvq_agnews_1024", w2_0),
            ),
        ),
        FleetShardSpec(
            3,
            "cashew.stratabaremetal.com",
            1,
            "emeans_pqrvq_s03",
            "server_600w",
            (
                ("pq_glove1m_10x8", w2_1),
                ("rvq_glove1m_4096", w2_1),
                ("pq_agnews_16x8", w2_1),
                ("rvq_agnews_1024", w2_1),
            ),
        ),
    ]
    for sp in shards:
        h_low = sp.host.lower()
        if any(bad in h_low for bad in FORBIDDEN_HOSTS):
            raise RuntimeError(f"Forbidden fleet host {sp.host!r} in shard table")
        if sp.session == "0":
            raise RuntimeError("Forbidden tmux session '0'")
    return shards


def get_arm_specs_for_cell(
    cell: CompressionCellSpec,
    budgets: tuple[float, ...],
    wave: str,
) -> list[tuple[str, float]]:
    """Return `[(arm_name, nepochs), ...]` for `cell` under `wave in {'all', 'baselines', 'apex'}`."""
    specs: list[tuple[str, float]] = []
    if cell.mode == "pq":
        if wave in ("all", "baselines"):
            for anch in PQ_ANCHOR_BASELINE_ARMS:
                specs.append((anch, 25.0))
            for b_t in budgets:
                for arm in PQ_STREAM_BASELINE_ARMS:
                    specs.append((arm, float(b_t)))
        if wave in ("all", "apex"):
            for b_t in budgets:
                for arm in PQ_STREAM_APEX_ARMS:
                    specs.append((arm, float(b_t)))
    else:
        if wave in ("all", "baselines"):
            for anch in RVQ_ANCHOR_BASELINE_ARMS:
                specs.append((anch, 25.0))
            for b_t in budgets:
                for arm in RVQ_STREAM_BASELINE_ARMS:
                    specs.append((arm, float(b_t)))
        if wave in ("all", "apex"):
            for b_t in budgets:
                for arm in RVQ_STREAM_APEX_ARMS:
                    specs.append((arm, float(b_t)))
    return specs


# ==============================================================================
# Atomic File I/O, SSH Discovery & GPU SKU Detection
# ==============================================================================


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON atomically via `PID+ns` temp file + `os.fsync` + `os.replace` (`allow_nan=False`)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".tmp_{os.getpid()}_{time.time_ns()}_{path.name}"
    try:
        text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def atomic_write_text(path: Path, text: str) -> None:
    """Write Markdown/text atomically via `PID+ns` temp file + `os.fsync` + `os.replace`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".tmp_{os.getpid()}_{time.time_ns()}_{path.name}"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def atomic_save_png_with_mirror(fig: plt.Figure, out_dir: Path, filename: str, dpi: int = 200) -> Path:
    """Atomically save `fig` to `out_dir/filename` AND byte-identical `out_dir/plots/filename`, verifying via PIL."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    dst_main = out_dir / filename
    dst_mirror = plots_dir / filename
    tmp_main = out_dir / f".tmp_{os.getpid()}_{time.time_ns()}_{filename}"
    tmp_mirror = plots_dir / f".tmp_{os.getpid()}_{time.time_ns()}_{filename}"
    try:
        fig.savefig(tmp_main, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
        with Image.open(tmp_main) as img:
            img.verify()
        shutil.copyfile(tmp_main, tmp_mirror)
        os.replace(tmp_main, dst_main)
        os.replace(tmp_mirror, dst_mirror)
        # Assert byte-for-byte SHA-256 parity
        h1 = hashlib.sha256(dst_main.read_bytes()).hexdigest()
        h2 = hashlib.sha256(dst_mirror.read_bytes()).hexdigest()
        if h1 != h2:
            raise RuntimeError(f"PNG mirror hash mismatch for {filename}")
        return dst_main
    finally:
        plt.close(fig)
        for tp in (tmp_main, tmp_mirror):
            if tp.exists():
                tp.unlink(missing_ok=True)


def discover_live_ssh_auth_sock() -> str | None:
    """Locate a live `ssh-agent` socket (reading `tmux show-environment -t 0 SSH_AUTH_SOCK` without touching session 0)."""
    candidates: list[str] = []
    cur = os.environ.get("SSH_AUTH_SOCK", "")
    if cur:
        candidates.append(cur)
    try:
        res = subprocess.run(
            ["tmux", "show-environment", "-t", "0", "SSH_AUTH_SOCK"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if res.returncode == 0 and res.stdout.startswith("SSH_AUTH_SOCK="):
            candidates.append(res.stdout.strip().split("=", 1)[1])
    except Exception:
        pass
    for sock_p in sorted(Path("/tmp").glob("ssh-*/agent.*"), key=lambda p: p.stat().st_mtime, reverse=True):
        candidates.append(str(sock_p))

    for cand in candidates:
        if cand and os.path.exists(cand):
            env = os.environ.copy()
            env["SSH_AUTH_SOCK"] = cand
            chk = subprocess.run(["ssh-add", "-l"], env=env, capture_output=True, timeout=2, check=False)
            if chk.returncode == 0:
                return cand
    return None


def detect_gpu_sku_info(device: torch.device) -> tuple[str, float]:
    """Identify `(gpu_name (powerW), power_w)` for single-SKU audit logging."""
    if device.type != "cuda":
        return "CPU", 0.0
    name = torch.cuda.get_device_name(device)
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
    power_w = 0.0
    try:
        res = subprocess.run(
            ["nvidia-smi", "-i", vis if vis.isdigit() else "0", "--query-gpu=power.max_limit", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if res.returncode == 0:
            power_w = float(res.stdout.strip().splitlines()[0].strip())
    except Exception:
        power_w = 0.0
    sku_label = f"{name} ({int(round(power_w))}W)" if power_w > 0 else name
    return sku_label, power_w


# ==============================================================================
# Corpus Loading, Comprehensive Untimed Warmup & Cell Execution
# ==============================================================================


@torch.no_grad()
def warmup_global_kernels(device: torch.device) -> None:
    """Pre-warm global coreset alignment buckets + CroissAttn-FP8 helpers outside timers."""
    if device.type != "cuda":
        return
    with torch.cuda.device(device):
        if SEED_HAS_TRITON:
            pair_d2 = torch.zeros((1024, 1024), device=device, dtype=torch.float32)
            pool_d2 = torch.ones((1024,), device=device, dtype=torch.float32)
            w_p = torch.full((1024,), 1.0 / 1024.0, device=device, dtype=torch.float32)
            out_idx = torch.empty((32,), device=device, dtype=torch.int64)
            for block_p, n_warps, p_vals in [(256, 4, (128, 129)), (1024, 8, (512, 513))]:
                for p_v in p_vals:
                    for ns_v in (1, 16, 17):
                        for rs_v in (1, 16, 17):
                            _local_greedy_kpp_kernel[(1,)](
                                pair_d2,
                                pool_d2,
                                w_p,
                                out_idx,
                                p_v,
                                ns_v,
                                rs_v,
                                BLOCK_P=block_p,
                                L_TRIALS=3,
                                num_warps=n_warps,
                            )
        if warmup_croiss_fp8_kernels is not None:
            try:
                warmup_croiss_fp8_kernels(device)
            except Exception:
                pass
    torch.cuda.synchronize(device)


def load_cell_corpus(
    cell: CompressionCellSpec,
    data_dir: Path,
    device: torch.device,
) -> RealVectorCorpus:
    """Load validated `RealVectorCorpus` (computing exact slice GT only in `--smoke` mode)."""
    full = load_realistic_benchmark(cell.corpus_name, data_dir=data_dir, device=device)
    if cell.n_base_limit is None and cell.n_query_limit is None:
        return full

    nb = min(int(cell.n_base_limit or full.base.shape[0]), full.base.shape[0])
    nv = min(512, full.val.shape[0])
    nq = min(int(cell.n_query_limit or full.query.shape[0]), full.query.shape[0])
    idx_b = np.linspace(0, full.base.shape[0] - 1, nb, dtype=np.int64)
    idx_v = np.linspace(0, full.val.shape[0] - 1, nv, dtype=np.int64)
    idx_q = np.linspace(0, full.query.shape[0] - 1, nq, dtype=np.int64)
    xb = np.ascontiguousarray(full.base[idx_b], dtype=np.float32)
    xv = np.ascontiguousarray(full.val[idx_v], dtype=np.float32)
    xq = np.ascontiguousarray(full.query[idx_q], dtype=np.float32)
    gt_ids, gt_d2 = compute_exact_groundtruth_gpu(xb, xq, top_k=min(100, nb), device=device)
    sub = RealVectorCorpus(
        name=cell.cell_id,
        base=xb,
        val=xv,
        query=xq,
        groundtruth_ids=gt_ids,
        groundtruth_sq_dists=gt_d2,
        d=cell.d,
        geometry=cell.geometry,
        metadata=dict(full.metadata),
    )
    validate_real_corpus(sub, check_cross_split_disjoint=True)
    return sub


@torch.no_grad()
def warmup_compression_cell(
    x_base_gpu: torch.Tensor,
    x_query_gpu: torch.Tensor,
    corpus: RealVectorCorpus,
    cell: CompressionCellSpec,
    arm_specs: list[tuple[str, float]],
    device: torch.device,
) -> None:
    """Compile EVERY Triton/CUDAGraph specialization for `cell` outside all timers (`0` cold-JIT spikes)."""
    if device.type != "cuda":
        return
    with torch.cuda.device(device):
        # 1. Warm dedicated PQ/RVQ & Apex kernels
        if cell.mode == "pq":
            warmup_pq_rvq_kernels(device, pq_shapes=((cell.d, cell.m, cell.ks),), rvq_shapes=())
            warmup_apex_kernels(device, d=cell.d // cell.m, k=cell.ks)
            # Warm coreset assign alignment buckets on subspace dimension
            d_sub = cell.d // cell.m
            m_sub = min(int(x_base_gpu.shape[0]), max(4096, 16 * cell.ks))
            x_m = x_base_gpu[:m_sub, :d_sub].float().contiguous()
            for dt in (torch.float32, torch.bfloat16):
                for kc in (1, 16, 17, 64, 65):
                    if kc <= m_sub:
                        assign_points(x_m, x_m[:kc], compute_dtype=dt)
        else:
            warmup_pq_rvq_kernels(device, pq_shapes=(), rvq_shapes=((cell.d, min(512, cell.k1), min(512, cell.k2)),))
            warmup_apex_kernels(device, d=cell.d, k=cell.k1)
            if cell.k2 != cell.k1:
                warmup_apex_kernels(device, d=cell.d, k=cell.k2)
            m_sub = min(int(x_base_gpu.shape[0]), max(8192, 16 * cell.k1))
            x_m = x_base_gpu[:m_sub].float().contiguous()
            for dt in (torch.float32, torch.bfloat16):
                for kc in (1, 16, 17, 64, 65, 128, 129):
                    if kc <= m_sub:
                        assign_points(x_m, x_m[:kc], compute_dtype=dt)

        # 2. Full-shape untimed `.fit(...)` + `.encode(...)` + `.adc_search(...)` across EVERY `(arm, budget)`!
        q_sub = min(128, int(x_query_gpu.shape[0]))
        for arm_name, nep in arm_specs:
            if arm_name in ("faiss_pq", "faiss_pq_sub"):
                # C++ OpenMP CPU — fast 2-iter warm call
                pq_w = ApexPQ(
                    d=cell.d,
                    m=cell.m,
                    ks=cell.ks,
                    nepochs=1.0,
                    solver=arm_name,
                    seed=999_901,
                    solver_kwargs={"niter": 2},
                )
                _, cw = pq_w.fit(x_base_gpu[: max(cell.ks * 4, 4096)], capture_codes=True)
                assert cw is not None
                _ = pq_w.adc_search(x_query_gpu[:q_sub], cw, k_top=10)
            elif cell.mode == "pq":
                kw_extra = {"niter": 2} if arm_name in ("lloyd_25", "lloyd_25_batched") else {}
                pq_w = ApexPQ(
                    d=cell.d,
                    m=cell.m,
                    ks=cell.ks,
                    nepochs=nep,
                    solver=arm_name,
                    seed=999_902,
                    solver_kwargs=kw_extra,
                )
                _, cw = pq_w.fit(x_base_gpu, capture_codes=True)
                assert cw is not None
                _ = pq_w.evaluate_reconstruction(x_query_gpu[:q_sub])
                _ = evaluate_adc_recall(
                    pq_w,
                    x_base_gpu,
                    cw,
                    x_query_gpu[:q_sub],
                    corpus.groundtruth_ids[:q_sub],
                    corpus.groundtruth_sq_dists[:q_sub],
                    k_top=10,
                )
            else:
                kw_extra = {"niter": 2} if arm_name == "lloyd_25" else {}
                rvq_w = ApexRVQ(
                    d=cell.d,
                    k1=cell.k1,
                    k2=cell.k2,
                    nepochs=nep,
                    solver=arm_name,
                    adc_beam=cell.adc_beam,
                    seed=999_903,
                    solver_kwargs=kw_extra,
                )
                _, rw = rvq_w.fit(x_base_gpu, capture_codes=True)
                assert rw is not None
                _ = rvq_w.evaluate_reconstruction(x_query_gpu[:q_sub])
                _ = evaluate_adc_recall(
                    rvq_w,
                    x_base_gpu,
                    rw,
                    x_query_gpu[:q_sub],
                    corpus.groundtruth_ids[:q_sub],
                    corpus.groundtruth_sq_dists[:q_sub],
                    k_top=10,
                )
    torch.cuda.synchronize(device)


@torch.no_grad()
def execute_cell_seeds(
    cell: CompressionCellSpec,
    seeds: list[int],
    budgets: tuple[float, ...],
    wave: str,
    data_dir: Path,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Execute all `(arm, budget)` runs for `cell` across `seeds` with cyclic counterbalancing."""
    if not seeds:
        return []

    arm_specs = get_arm_specs_for_cell(cell, budgets=budgets, wave=wave)
    if not arm_specs:
        return []

    host = socket.gethostname()
    vis_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    gpu_sku, power_w = detect_gpu_sku_info(device)

    corpus = load_cell_corpus(cell, data_dir=data_dir, device=device)
    # Strictly isolate `base` (train) and `query` (held-out test); `val` unused so 0 leakage possible
    x_base_gpu = torch.from_numpy(corpus.base).to(device=device, dtype=torch.float32).contiguous()
    x_query_gpu = torch.from_numpy(corpus.query).to(device=device, dtype=torch.float32).contiguous()

    print(
        f"[PQ-RVQ] Warming {cell.cell_id} ({cell.codec_label}, N={corpus.base.shape[0]:,}, D={cell.d}) "
        f"wave={wave} on {host} GPU={vis_gpu} [{gpu_sku}] ...",
        flush=True,
    )
    t_w0 = time.perf_counter()
    warmup_compression_cell(x_base_gpu, x_query_gpu, corpus, cell, arm_specs, device)
    print(f"[PQ-RVQ] Warmup done for {cell.cell_id} in {time.perf_counter() - t_w0:.2f}s", flush=True)

    records: list[dict[str, Any]] = []
    for s_idx, seed in enumerate(seeds):
        t_s0 = time.perf_counter()
        rot = int(seed) % len(arm_specs)
        ordered_specs = arm_specs[rot:] + arm_specs[:rot]
        seed_map: dict[tuple[str, float], dict[str, Any]] = {}

        for arm_name, nepochs in ordered_specs:
            if cell.mode == "pq":
                q_obj: ApexPQ | ApexRVQ = ApexPQ(
                    d=cell.d,
                    m=cell.m,
                    ks=cell.ks,
                    nepochs=nepochs,
                    solver=arm_name,
                    seed=seed,
                )
            else:
                q_obj = ApexRVQ(
                    d=cell.d,
                    k1=cell.k1,
                    k2=cell.k2,
                    nepochs=nepochs,
                    solver=arm_name,
                    adc_beam=cell.adc_beam,
                    seed=seed,
                )

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            # Fit codebooks strictly on `x_base_gpu` ONLY!
            q_obj.fit(x_base_gpu, capture_codes=False)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

            # Encode database `x_base_gpu` (timed inside `q_obj.encode_time_sec`)
            db_codes = q_obj.encode(x_base_gpu)
            enc_db_sec = float(q_obj.encode_time_sec)

            # Exact FP64 reconstruction fidelity on full `base` (using precomputed `db_codes`) and full held-out `query`
            train_rec = q_obj.evaluate_reconstruction(x_base_gpu, codes=db_codes)
            test_rec = q_obj.evaluate_reconstruction(x_query_gpu)

            # Pre-build CSR outside ADC query timer for RVQ (index build is part of database ingestion)
            if isinstance(q_obj, ApexRVQ):
                t_idx0 = time.perf_counter()
                csr_or_codes: Any = q_obj.build_csr_index(db_codes)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                index_build_sec = time.perf_counter() - t_idx0
            else:
                csr_or_codes = db_codes
                index_build_sec = 0.0

            adc_res = evaluate_adc_recall(
                quantizer=q_obj,
                x_base=x_base_gpu,
                codes=csr_or_codes,
                x_query=x_query_gpu,
                groundtruth_ids=corpus.groundtruth_ids,
                groundtruth_sq_dists=corpus.groundtruth_sq_dists,
                k_top=10,
            )

            rec_entry: dict[str, Any] = {
                "cell_id": cell.cell_id,
                "mode": cell.mode,
                "corpus": cell.corpus_name,
                "short_title": cell.short_title,
                "codec_label": cell.codec_label,
                "geometry": cell.geometry,
                "sku_tier": cell.sku_tier,
                "n_base": int(corpus.base.shape[0]),
                "n_query": int(corpus.query.shape[0]),
                "d": int(cell.d),
                "m": int(cell.m),
                "ks": int(cell.ks),
                "k1": int(cell.k1),
                "k2": int(cell.k2),
                "adc_beam": int(cell.adc_beam) if cell.adc_beam is not None else 0,
                "bytes_per_vec": float(q_obj.bytes_per_vec),
                "compression_ratio_vs_fp32": float(q_obj.compression_ratio_vs_fp32),
                "seed": int(seed),
                "arm": arm_name,
                "nepochs": float(nepochs),
                "host": host,
                "visible_gpu": vis_gpu,
                "gpu_sku": gpu_sku,
                "gpu_power_w": float(power_w),
                "selected_gear": str(
                    q_obj.selected_gear if isinstance(q_obj, ApexPQ) else f"{q_obj.stage1_gear}+{q_obj.stage2_gear}"
                ),
                "resolved_precision": str(q_obj.resolved_precision),
                "init_time_sec": float(q_obj.init_time_sec),
                "train_time_sec": float(q_obj.train_time_sec),
                "total_time_sec": float(q_obj.total_time_sec),
                "stage1_time_sec": float(getattr(q_obj, "stage1_time_sec", q_obj.total_time_sec)),
                "residual_time_sec": float(getattr(q_obj, "residual_time_sec", 0.0)),
                "stage2_time_sec": float(getattr(q_obj, "stage2_time_sec", 0.0)),
                "encode_db_time_sec": float(enc_db_sec + index_build_sec),
                "adc_search_time_sec": float(adc_res.search_time_sec),
                "adc_qps": float(adc_res.qps),
                "n_splits": int(q_obj.n_splits),
                "train_mse": float(train_rec.mse),
                "train_sqnr_db": float(train_rec.sqnr_db),
                "train_evr_sqnr_db": float(train_rec.evr_sqnr_db),
                "test_mse": float(test_rec.mse),
                "test_per_dim_mse": float(test_rec.per_dim_mse),
                "test_rmse": float(test_rec.rmse),
                "test_sqnr_db": float(test_rec.sqnr_db),
                "test_evr_sqnr_db": float(test_rec.evr_sqnr_db),
                "test_cosine_sim": float(test_rec.mean_cosine_sim),
                "recall_at_1": float(adc_res.recall_at_1),
                "one_recall_at_10": float(adc_res.one_recall_at_10),
                "recall_at_10": float(adc_res.recall_at_10),
                "tie_recall_at_1": float(adc_res.tie_recall_at_1),
                "tie_one_recall_at_10": float(adc_res.tie_one_recall_at_10),
                "tie_recall_at_10": float(adc_res.tie_recall_at_10),
            }
            seed_map[(arm_name, float(nepochs))] = rec_entry

        for key in arm_specs:
            records.append(seed_map[key])

        print(
            f"[PQ-RVQ] {cell.cell_id} seed={seed} ({s_idx + 1}/{len(seeds)}) "
            f"done in {time.perf_counter() - t_s0:.2f}s",
            flush=True,
        )

    return records


# ==============================================================================
# Validation, Cold-JIT Spike Gate, Paired Statistics, Markdown & Plots
# ==============================================================================


def validate_records_and_detect_spikes(
    records: list[dict[str, Any]],
    cells: list[CompressionCellSpec],
    expected_seeds: list[int],
    budgets: tuple[float, ...],
    wave: str,
) -> list[dict[str, Any]]:
    """Validate finiteness, uniqueness, single-GPU pairing, single-SKU parity, and Seed-0 cold-JIT spikes."""
    seen: set[tuple[str, str, float, int]] = set()
    cell_skus: dict[str, set[str]] = {}
    seed_gpus: dict[tuple[str, int], set[tuple[str, str]]] = {}

    for r in records:
        key = (str(r["cell_id"]), str(r["arm"]), float(r["nepochs"]), int(r["seed"]))
        if key in seen:
            raise ValueError(f"Duplicate benchmark record key {key}")
        seen.add(key)

        for k_num, val in r.items():
            if isinstance(val, (int, float)) and not math.isfinite(float(val)):
                raise ValueError(f"Non-finite value {k_num}={val!r} in record {key}")

        if not (0.0 <= float(r["recall_at_1"]) <= float(r["one_recall_at_10"]) + 1e-9 <= 1.0 + 1e-9):
            raise ValueError(
                f"Recall monotonicity violation R@1={r['recall_at_1']} > 1-R@10={r['one_recall_at_10']} in {key}"
            )
        cell_skus.setdefault(str(r["cell_id"]), set()).add(str(r["gpu_sku"]))
        seed_gpus.setdefault((str(r["cell_id"]), int(r["seed"])), set()).add(
            (str(r["host"]), str(r["visible_gpu"]))
        )

    for cid, skus in cell_skus.items():
        if len(skus) > 1:
            raise ValueError(f"Cross-SKU confounding detected on {cid}: {skus}")

    for cs_key, gpus in seed_gpus.items():
        if len(gpus) > 1:
            raise ValueError(f"Cross-GPU split within paired (cell_id, seed)={cs_key}: {gpus}")

    # Completeness check
    for cell in cells:
        expected_arms = get_arm_specs_for_cell(cell, budgets=budgets, wave=wave)
        for seed in expected_seeds:
            for arm_name, nep in expected_arms:
                if (cell.cell_id, arm_name, float(nep), int(seed)) not in seen:
                    raise ValueError(f"Missing expected record {(cell.cell_id, arm_name, nep, seed)}")

    # Automated Seed-0 cold-JIT spike detector per `(host, visible_gpu, cell_id, arm, nepochs)`
    spikes: list[dict[str, Any]] = []
    if len(expected_seeds) >= 2:
        by_group: dict[tuple[str, str, str, str, float], list[tuple[int, float]]] = {}
        for r in records:
            gk = (str(r["host"]), str(r["visible_gpu"]), str(r["cell_id"]), str(r["arm"]), float(r["nepochs"]))
            by_group.setdefault(gk, []).append((int(r["seed"]), float(r["total_time_sec"])))
        for gk, pairs in by_group.items():
            if len(pairs) < 2:
                continue
            pairs.sort(key=lambda x: x[0])
            t0 = pairs[0][1]
            rest_med = float(np.median([p[1] for p in pairs[1:]]))
            # Flag genuine Triton compile spikes (> 2.5x AND > +25 ms excess)
            if rest_med > 1e-4 and t0 > max(2.5 * rest_med, rest_med + 0.025):
                spikes.append(
                    {
                        "host": gk[0],
                        "gpu": gk[1],
                        "cell_id": gk[2],
                        "arm": gk[3],
                        "nepochs": gk[4],
                        "first_seed_sec": t0,
                        "rest_median_sec": rest_med,
                        "ratio": t0 / rest_med,
                    }
                )
    return spikes


def compute_paired_summary(
    records: list[dict[str, Any]],
    cells: list[CompressionCellSpec],
    budgets: tuple[float, ...],
) -> list[dict[str, Any]]:
    """Compute paired speedups, ΔMSE, ΔSQNR (dB), ΔRecall@1/10, and paired Student-t / Wilcoxon p-values."""
    by_key: dict[tuple[str, str, float, int], dict[str, Any]] = {
        (str(r["cell_id"]), str(r["arm"]), float(r["nepochs"]), int(r["seed"])): r for r in records
    }
    seeds = sorted({int(r["seed"]) for r in records})
    comparisons: list[dict[str, Any]] = []

    for cell in cells:
        for b_t in budgets:
            apex_rows = [by_key.get((cell.cell_id, "apex_emeans", float(b_t), s)) for s in seeds]
            if any(x is None for x in apex_rows):
                continue
            apex_valid = [x for x in apex_rows if x is not None]

            ref_targets: list[tuple[str, float]] = [("lloyd_25", 25.0)]
            if cell.mode == "pq":
                ref_targets.extend(
                    [
                        ("faiss_pq", 25.0),
                        ("faiss_pq_sub", 25.0),
                        ("lloyd_25_batched", 25.0),
                        ("apex_emeans_subspace", float(b_t)),
                    ]
                )
            for s_arm in ("emeans", "smeans", "emeans_pp", "hybrid_semeans", "ultra_emeans"):
                ref_targets.append((s_arm, float(b_t)))

            for ref_arm, ref_t in ref_targets:
                ref_rows = [by_key.get((cell.cell_id, ref_arm, float(ref_t), s)) for s in seeds]
                if any(x is None for x in ref_rows):
                    continue
                ref_valid = [x for x in ref_rows if x is not None]

                t_ap = np.array([float(x["total_time_sec"]) for x in apex_valid], dtype=np.float64)
                t_rf = np.array([float(x["total_time_sec"]) for x in ref_valid], dtype=np.float64)
                speedups = t_rf / np.maximum(t_ap, 1e-12)

                mse_ap = np.array([float(x["test_mse"]) for x in apex_valid], dtype=np.float64)
                mse_rf = np.array([float(x["test_mse"]) for x in ref_valid], dtype=np.float64)
                rel_mse_pct = 100.0 * (mse_ap - mse_rf) / np.maximum(np.abs(mse_rf), 1e-30)

                sqnr_ap = np.array([float(x["test_sqnr_db"]) for x in apex_valid], dtype=np.float64)
                sqnr_rf = np.array([float(x["test_sqnr_db"]) for x in ref_valid], dtype=np.float64)
                dsqnr = sqnr_ap - sqnr_rf

                r1_ap = np.array([float(x["recall_at_1"]) for x in apex_valid], dtype=np.float64)
                r1_rf = np.array([float(x["recall_at_1"]) for x in ref_valid], dtype=np.float64)
                r10_ap = np.array([float(x["one_recall_at_10"]) for x in apex_valid], dtype=np.float64)
                r10_rf = np.array([float(x["one_recall_at_10"]) for x in ref_valid], dtype=np.float64)

                log_sp = np.log(speedups)
                if len(seeds) >= 2 and float(np.std(log_sp)) > 1e-12:
                    p_speed = float(stats.ttest_1samp(log_sp, 0.0).pvalue)
                else:
                    p_speed = 1.0
                if len(seeds) >= 5 and np.any(np.abs(log_sp) > 1e-12):
                    try:
                        p_speed_w = float(stats.wilcoxon(log_sp).pvalue)
                    except Exception:
                        p_speed_w = p_speed
                else:
                    p_speed_w = p_speed

                mse_diff = mse_ap - mse_rf
                if len(seeds) >= 2 and float(np.std(mse_diff)) > 1e-12:
                    p_mse = float(stats.ttest_rel(mse_ap, mse_rf).pvalue)
                else:
                    p_mse = 1.0
                if len(seeds) >= 5 and np.any(np.abs(mse_diff) > 1e-12):
                    try:
                        p_mse_w = float(stats.wilcoxon(mse_diff).pvalue)
                    except Exception:
                        p_mse_w = p_mse
                else:
                    p_mse_w = p_mse

                comparisons.append(
                    {
                        "cell_id": cell.cell_id,
                        "mode": cell.mode,
                        "corpus": cell.corpus_name,
                        "apex_nepochs": float(b_t),
                        "ref_arm": ref_arm,
                        "ref_nepochs": float(ref_t),
                        "n_seeds": len(seeds),
                        "apex_mean_ms": float(np.mean(t_ap) * 1000.0),
                        "ref_mean_ms": float(np.mean(t_rf) * 1000.0),
                        "geomean_speedup_x": float(np.exp(np.mean(log_sp))),
                        "mean_speedup_x": float(np.mean(speedups)),
                        "speedup_p_value": p_speed,
                        "speedup_wilcoxon_p": p_speed_w,
                        "rel_test_mse_pct": float(np.mean(rel_mse_pct)),
                        "mse_p_value": p_mse,
                        "mse_wilcoxon_p": p_mse_w,
                        "delta_sqnr_db": float(np.mean(dsqnr)),
                        "delta_recall_at_1_pp": float(100.0 * np.mean(r1_ap - r1_rf)),
                        "delta_one_recall_at_10_pp": float(100.0 * np.mean(r10_ap - r10_rf)),
                    }
                )
    return comparisons


def write_summary_markdown(
    records: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    spikes: list[dict[str, Any]],
    cells: list[CompressionCellSpec],
    budgets: tuple[float, ...],
    out_path: Path,
) -> None:
    """Render comprehensive GFM summary tables to `outputs/pq_rvq/pq_rvq_summary.md`."""
    lines: list[str] = [
        "# Fused Product Quantization (`ApexPQ`) & 2-Stage RVQ (`ApexRVQ`) Benchmark Summary",
        "",
        f"- **Total Validated Runs**: `{len(records):,}` (`0` `NaN`/`Inf`, `{len(spikes)}` cold-JIT spikes)",
        f"- **Matched Epoch Budgets**: `{[float(b) for b in budgets]}`",
        "- **Zero Data Leakage**: Codebooks, origins, scales, and cross-tables fit strictly on `corpus.base`; ADC `Recall@1` / `1-Recall@10` / `10-Recall@10` and `Test MSE` / `SQNR (dB)` evaluated on held-out `corpus.query`.",
        "",
        "## 1. Headline Speedup & Recall/SQNR Summary (`ApexPQ` / `ApexRVQ` @ `T = 2` vs Reference Anchors)",
        "",
        "| Cell | Codec | Ref Baseline | Apex Time (ms) | Ref Time (ms) | Geomean Speedup | ΔTest MSE (%) | ΔSQNR (dB) | ΔR@1 (pp) | Δ1-R@10 (pp) |",
        "| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    t_focus = 2.0 if 2.0 in budgets else budgets[-1]
    cell_by_id = {c.cell_id: c for c in cells}
    for cmp in comparisons:
        if abs(float(cmp["apex_nepochs"]) - t_focus) > 1e-9:
            continue
        if cmp["ref_arm"] not in ("faiss_pq", "faiss_pq_sub", "lloyd_25", "lloyd_25_batched", "smeans", "ultra_emeans"):
            continue
        c_spec = cell_by_id[cmp["cell_id"]]
        lines.append(
            f"| `{cmp['cell_id']}` | {c_spec.codec_label} | `{cmp['ref_arm']}` | "
            f"`{cmp['apex_mean_ms']:.2f}` | `{cmp['ref_mean_ms']:.2f}` | "
            f"**`{cmp['geomean_speedup_x']:.2f}x`** | `{cmp['rel_test_mse_pct']:+.2f}%` | "
            f"`{cmp['delta_sqnr_db']:+.3f}` | `{cmp['delta_recall_at_1_pp']:+.2f}` | "
            f"`{cmp['delta_one_recall_at_10_pp']:+.2f}` |"
        )

    lines.extend(["", "## 2. Full Per-Cell Compression, Reconstruction Fidelity & ADC Recall Tables", ""])
    for cell in cells:
        c_recs = [r for r in records if r["cell_id"] == cell.cell_id]
        if not c_recs:
            continue
        gpu_sku = c_recs[0]["gpu_sku"]
        lines.extend(
            [
                f"### `{cell.cell_id}` — {cell.short_title} · {cell.codec_label} (`{gpu_sku}`)",
                "",
                "| Solver Arm | Budget `T` | Train (ms) | Encode DB (ms) | ADC Search (ms) | ADC QPS | Test MSE | SQNR (dB) | EVR-SQNR (dB) | R@1 | 1-R@10 | 10-R@10 |",
                "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        groups: dict[tuple[str, float], list[dict[str, Any]]] = {}
        for r in c_recs:
            groups.setdefault((str(r["arm"]), float(r["nepochs"])), []).append(r)

        def _sort_key(item: tuple[tuple[str, float], list[dict[str, Any]]]) -> tuple[int, float, float]:
            (a_n, nep_v), rows = item
            is_ap = 0 if a_n.startswith("apex") else 1
            return (is_ap, nep_v, float(np.mean([x["total_time_sec"] for x in rows])))

        for (arm_n, nep_v), rows in sorted(groups.items(), key=_sort_key):
            tr_ms = float(np.mean([x["total_time_sec"] for x in rows]) * 1000.0)
            tr_sd = float(np.std([x["total_time_sec"] for x in rows]) * 1000.0)
            en_ms = float(np.mean([x["encode_db_time_sec"] for x in rows]) * 1000.0)
            ad_ms = float(np.mean([x["adc_search_time_sec"] for x in rows]) * 1000.0)
            qps = float(np.mean([x["adc_qps"] for x in rows]))
            mse_m = float(np.mean([x["test_mse"] for x in rows]))
            sqnr_m = float(np.mean([x["test_sqnr_db"] for x in rows]))
            evr_m = float(np.mean([x["test_evr_sqnr_db"] for x in rows]))
            r1_m = float(np.mean([x["recall_at_1"] for x in rows]))
            or10_m = float(np.mean([x["one_recall_at_10"] for x in rows]))
            r10_m = float(np.mean([x["recall_at_10"] for x in rows]))
            bold = "**" if arm_n == "apex_emeans" else ""
            lines.append(
                f"| {bold}`{arm_n}`{bold} | `{nep_v:g}` | {bold}`{tr_ms:.2f} ± {tr_sd:.2f}`{bold} | "
                f"`{en_ms:.2f}` | `{ad_ms:.2f}` | `{qps:,.0f}` | `{mse_m:.4f}` | "
                f"`{sqnr_m:.2f}` | `{evr_m:.2f}` | `{r1_m:.4f}` | {bold}`{or10_m:.4f}`{bold} | `{r10_m:.4f}` |"
            )
        lines.append("")

    atomic_write_text(out_path, "\n".join(lines) + "\n")


def generate_all_plots(
    records: list[dict[str, Any]],
    cells: list[CompressionCellSpec],
    budgets: tuple[float, ...],
    out_dir: Path,
) -> list[Path]:
    """Generate the 3 required crisp publication PNGs (and `plots/` byte-identical mirrors)."""
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )

    pq_cells = [c for c in cells if c.mode == "pq"]
    rvq_cells = [c for c in cells if c.mode == "rvq"]
    saved: list[Path] = []

    # ------------------------------------------------------------------
    # Plot 1: `pq_rvq_pareto_recall.png` (2x4 ADC 1-Recall@10 vs Train Time ms)
    # ------------------------------------------------------------------
    fig1, axes1 = plt.subplots(2, 4, figsize=(19.5, 9.2), constrained_layout=True)
    plot_arms_pq = [
        "faiss_pq",
        "faiss_pq_sub",
        "lloyd_25",
        "lloyd_25_batched",
        "emeans",
        "smeans",
        "emeans_pp",
        "ultra_emeans",
        "apex_emeans_fp8",
        "apex_emeans",
    ]
    plot_arms_rvq = ["lloyd_25", "emeans", "smeans", "emeans_pp", "hybrid_semeans", "ultra_emeans", "apex_emeans_fp8", "apex_emeans"]

    legend_handles: dict[str, Any] = {}
    for row_idx, (row_cells, row_arms, row_tag) in enumerate(
        [(pq_cells, plot_arms_pq, "Product Quantization (PQ)"), (rvq_cells, plot_arms_rvq, "2-Stage Residual VQ (RVQ)")]
    ):
        for col_idx in range(4):
            ax = axes1[row_idx, col_idx]
            if col_idx >= len(row_cells):
                ax.axis("off")
                continue
            cell = row_cells[col_idx]
            crecs = [r for r in records if r["cell_id"] == cell.cell_id]
            for arm_id in row_arms:
                arecs = [r for r in crecs if r["arm"] == arm_id]
                if not arecs:
                    continue
                lbl, color, marker, ls = ARM_STYLE.get(arm_id, (arm_id, "#444444", "o", "-"))
                by_t: dict[float, list[dict[str, Any]]] = {}
                for r in arecs:
                    by_t.setdefault(float(r["nepochs"]), []).append(r)
                ts_sorted = sorted(by_t.keys())
                xs = [float(np.mean([x["total_time_sec"] for x in by_t[t]]) * 1000.0) for t in ts_sorted]
                ys = [float(np.mean([x["one_recall_at_10"] for x in by_t[t]]) * 100.0) for t in ts_sorted]
                lw = 2.4 if arm_id == "apex_emeans" else 1.5
                ms = 7.5 if arm_id in ("apex_emeans", "lloyd_25", "faiss_pq") else 5.8
                zord = 6 if arm_id.startswith("apex") else 3
                (h,) = ax.plot(
                    xs,
                    ys,
                    label=lbl,
                    color=color,
                    marker=marker,
                    linestyle=ls if len(xs) > 1 else "None",
                    linewidth=lw,
                    markersize=ms,
                    zorder=zord,
                )
                if lbl not in legend_handles:
                    legend_handles[lbl] = h

            ax.set_xscale("log")
            ax.grid(True, which="both", linestyle=":", alpha=0.55)
            ax.set_title(f"{cell.short_title}\n{cell.codec_label}", fontweight="bold")
            ax.set_xlabel("Codebook Training Time (ms, log scale)")
            if col_idx == 0:
                ax.set_ylabel(f"{row_tag}\nADC 1-Recall@10 (%) ↑")

    fig1.suptitle(
        "ApexPQ & ApexRVQ Pareto Frontier: ADC Recall@10 vs. GPU Codebook Training Latency (Upper-Left Optimal)",
        fontsize=13,
        fontweight="bold",
    )
    if legend_handles:
        fig1.legend(
            list(legend_handles.values()),
            list(legend_handles.keys()),
            loc="outside lower center",
            ncol=min(6, len(legend_handles)),
            frameon=True,
        )
    saved.append(atomic_save_png_with_mirror(fig1, out_dir, "pq_rvq_pareto_recall.png"))

    # ------------------------------------------------------------------
    # Plot 2: `pq_rvq_speedup_bars.png` (1x2 Grouped Log-Speedup & Latency Bars across 4 Corpora)
    # ------------------------------------------------------------------
    fig2, (ax2_pq, ax2_rvq) = plt.subplots(1, 2, figsize=(17.2, 7.2), constrained_layout=True)
    t_bar = 2.0 if 2.0 in budgets else budgets[-1]

    # Left panel: PQ Training Speedup over `Lloyd@25` (with `FAISS IndexPQ` Full & 256/c sub, `Batched Lloyd@25`, `s-means`, `UltraEMeans`, `ApexPQ T=1,2`)
    pq_bar_arms = [
        ("lloyd_25", 25.0, "Lloyd@25 (1.0×)", "#333333"),
        ("faiss_pq", 25.0, "FAISS IndexPQ (Full N)", "#555555"),
        ("faiss_pq_sub", 25.0, "FAISS IndexPQ (256/c sub)", "#8c564b"),
        ("lloyd_25_batched", 25.0, "Batched GPU Lloyd@25", "#9e9e9e"),
        ("smeans", t_bar, f"s-means (T={t_bar:g})", "#b8860b"),
        ("ultra_emeans", t_bar, f"UltraEMeans (T={t_bar:g})", "#17becf"),
        ("apex_emeans", t_bar, f"ApexPQ (T={t_bar:g})", "#d62728"),
        ("apex_emeans", budgets[0], f"ApexPQ (T={budgets[0]:g})", "#8b0000"),
    ]
    # Deduplicate if `t_bar == budgets[0]`
    seen_bar_pq: list[tuple[str, float, str, str]] = []
    for item in pq_bar_arms:
        if (item[0], item[1]) not in [(x[0], x[1]) for x in seen_bar_pq]:
            seen_bar_pq.append(item)

    x_pos = np.arange(len(pq_cells))
    width_pq = 0.84 / max(1, len(seen_bar_pq))
    any_pos_pq = False
    for b_i, (a_id, nep_v, b_lbl, b_col) in enumerate(seen_bar_pq):
        vals: list[float] = []
        for cell in pq_cells:
            ref_r = [r["total_time_sec"] for r in records if r["cell_id"] == cell.cell_id and r["arm"] == "lloyd_25"]
            if not ref_r:
                # Fallback reference if `--wave apex` ran standalone without `lloyd_25`
                ref_r = [r["total_time_sec"] for r in records if r["cell_id"] == cell.cell_id and r["arm"] == "apex_emeans_subspace"]
            arm_r = [
                r["total_time_sec"]
                for r in records
                if r["cell_id"] == cell.cell_id and r["arm"] == a_id and abs(float(r["nepochs"]) - nep_v) < 1e-9
            ]
            if ref_r and arm_r:
                vals.append(float(np.mean(ref_r)) / max(float(np.mean(arm_r)), 1e-12))
            else:
                vals.append(0.0)
        if any(v > 0.0 for v in vals):
            any_pos_pq = True
            offs = x_pos + (b_i - 0.5 * (len(seen_bar_pq) - 1)) * width_pq
            bars = ax2_pq.bar(offs, vals, width=width_pq * 0.94, label=b_lbl, color=b_col, edgecolor="black", linewidth=0.4)
            if a_id == "apex_emeans":
                for rect, v in zip(bars, vals):
                    if v > 0:
                        ax2_pq.text(
                            rect.get_x() + rect.get_width() * 0.5,
                            v * 1.06,
                            f"{v:.0f}×" if v >= 10 else f"{v:.1f}×",
                            ha="center",
                            va="bottom",
                            fontsize=7.5,
                            fontweight="bold",
                            color="#8b0000",
                        )

    if any_pos_pq:
        ax2_pq.set_yscale("log")
    ax2_pq.set_xticks(x_pos)
    ax2_pq.set_xticklabels([c.short_title.replace(" (", "\n(") for c in pq_cells])
    ax2_pq.set_ylabel("Training Speedup over Lloyd@25 (log scale, ×) ↑")
    ax2_pq.set_title("Product Quantization (PQ): Fused 3D ApexPQ Speedup", fontweight="bold")
    ax2_pq.grid(True, axis="y", which="both", linestyle=":", alpha=0.55)
    ax2_pq.margins(y=0.22)
    if any_pos_pq:
        ax2_pq.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.13),
            ncol=4,
            frameon=True,
            fontsize=8,
        )

    # Right panel: 2-Stage RVQ Speedup over `Lloyd@25`
    rvq_bar_arms = [
        ("lloyd_25", 25.0, "Lloyd@25 (1.0×)", "#333333"),
        ("emeans", t_bar, f"e-means (T={t_bar:g})", "#1f77b4"),
        ("smeans", t_bar, f"s-means (T={t_bar:g})", "#b8860b"),
        ("ultra_emeans", t_bar, f"UltraEMeans (T={t_bar:g})", "#17becf"),
        ("apex_emeans_fp8", t_bar, f"ApexRVQ FP8 (T={t_bar:g})", "#d62728"),
        ("apex_emeans", budgets[0], f"ApexRVQ (T={budgets[0]:g})", "#8b0000"),
    ]
    seen_bar_rvq: list[tuple[str, float, str, str]] = []
    for item in rvq_bar_arms:
        if (item[0], item[1]) not in [(x[0], x[1]) for x in seen_bar_rvq]:
            seen_bar_rvq.append(item)

    x_pos_r = np.arange(len(rvq_cells))
    width_rvq = 0.82 / max(1, len(seen_bar_rvq))
    any_pos_rvq = False
    for b_i, (a_id, nep_v, b_lbl, b_col) in enumerate(seen_bar_rvq):
        vals_r: list[float] = []
        for cell in rvq_cells:
            ref_r = [r["total_time_sec"] for r in records if r["cell_id"] == cell.cell_id and r["arm"] == "lloyd_25"]
            if not ref_r:
                ref_r = [r["total_time_sec"] for r in records if r["cell_id"] == cell.cell_id and r["arm"] == "apex_emeans_bf16"]
            arm_r = [
                r["total_time_sec"]
                for r in records
                if r["cell_id"] == cell.cell_id and r["arm"] == a_id and abs(float(r["nepochs"]) - nep_v) < 1e-9
            ]
            if ref_r and arm_r:
                vals_r.append(float(np.mean(ref_r)) / max(float(np.mean(arm_r)), 1e-12))
            else:
                vals_r.append(0.0)
        if any(v > 0.0 for v in vals_r):
            any_pos_rvq = True
            offs_r = x_pos_r + (b_i - 0.5 * (len(seen_bar_rvq) - 1)) * width_rvq
            bars_r = ax2_rvq.bar(offs_r, vals_r, width=width_rvq * 0.94, label=b_lbl, color=b_col, edgecolor="black", linewidth=0.4)
            if a_id.startswith("apex"):
                for rect, v in zip(bars_r, vals_r):
                    if v > 0:
                        ax2_rvq.text(
                            rect.get_x() + rect.get_width() * 0.5,
                            v * 1.05,
                            f"{v:.1f}×",
                            ha="center",
                            va="bottom",
                            fontsize=7.8,
                            fontweight="bold",
                            color="#8b0000",
                        )

    if any_pos_rvq:
        ax2_rvq.set_yscale("log")
    ax2_rvq.set_xticks(x_pos_r)
    ax2_rvq.set_xticklabels([c.short_title.replace(" (", "\n(") for c in rvq_cells])
    ax2_rvq.set_ylabel("Training Speedup over Lloyd@25 (log scale, ×) ↑")
    ax2_rvq.set_title("2-Stage Residual VQ (RVQ): ApexRVQ + CroissAttn-FP8 Speedup", fontweight="bold")
    ax2_rvq.grid(True, axis="y", which="both", linestyle=":", alpha=0.55)
    ax2_rvq.margins(y=0.22)
    if any_pos_rvq:
        ax2_rvq.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.13),
            ncol=3,
            frameon=True,
            fontsize=8,
        )

    fig2.suptitle(
        "Codebook Training Speedup Across Real Million-Scale & High-D Corpora (Higher is Faster)",
        fontsize=13,
        fontweight="bold",
    )
    saved.append(atomic_save_png_with_mirror(fig2, out_dir, "pq_rvq_speedup_bars.png"))

    # ------------------------------------------------------------------
    # Plot 3: `rvq_sqnr_frontier.png` (2x2 2-Stage RVQ Reconstruction SQNR dB vs Latency ms)
    # ------------------------------------------------------------------
    fig3, axes3 = plt.subplots(2, 2, figsize=(13.5, 9.0), constrained_layout=True)
    flat_ax3 = axes3.ravel()
    sqnr_handles: dict[str, Any] = {}
    for idx, cell in enumerate(rvq_cells[:4]):
        ax = flat_ax3[idx]
        crecs = [r for r in records if r["cell_id"] == cell.cell_id]
        for arm_id in plot_arms_rvq:
            arecs = [r for r in crecs if r["arm"] == arm_id]
            if not arecs:
                continue
            lbl, color, marker, ls = ARM_STYLE.get(arm_id, (arm_id, "#444444", "o", "-"))
            by_t = {}
            for r in arecs:
                by_t.setdefault(float(r["nepochs"]), []).append(r)
            ts_s = sorted(by_t.keys())
            xs = [float(np.mean([x["total_time_sec"] for x in by_t[t]]) * 1000.0) for t in ts_s]
            ys = [float(np.mean([x["test_sqnr_db"] for x in by_t[t]])) for t in ts_s]
            lw = 2.4 if arm_id == "apex_emeans" else 1.5
            ms = 7.8 if arm_id in ("apex_emeans", "lloyd_25") else 6.0
            (h3,) = ax.plot(
                xs,
                ys,
                label=lbl,
                color=color,
                marker=marker,
                linestyle=ls if len(xs) > 1 else "None",
                linewidth=lw,
                markersize=ms,
                zorder=6 if arm_id.startswith("apex") else 3,
            )
            if lbl not in sqnr_handles:
                sqnr_handles[lbl] = h3

        ax.set_xscale("log")
        ax.grid(True, which="both", linestyle=":", alpha=0.55)
        ax.set_title(f"{cell.short_title} — {cell.codec_label}", fontweight="bold")
        ax.set_xlabel("2-Stage Codebook Training Time (ms, log scale)")
        ax.set_ylabel("Held-Out Test SQNR (dB) ↑")

    fig3.suptitle(
        "2-Stage Residual Vector Quantization (ApexRVQ): Held-Out SQNR (dB) vs. Training Time (Upper-Left Optimal)",
        fontsize=12.5,
        fontweight="bold",
    )
    if sqnr_handles:
        fig3.legend(
            list(sqnr_handles.values()),
            list(sqnr_handles.keys()),
            loc="outside lower center",
            ncol=min(4, len(sqnr_handles)),
            frameon=True,
        )
    saved.append(atomic_save_png_with_mirror(fig3, out_dir, "rvq_sqnr_frontier.png"))
    return saved


# ==============================================================================
# 4-GPU 600W Server Blackwell Fleet Dispatcher (`acorn:0,1 + cashew:0,1`)
# ==============================================================================


def assert_fleet_gpus_idle(shards: list[FleetShardSpec], env: dict[str, str]) -> None:
    """Verify via live `nvidia-smi` that every target `(host, gpu_id)` is idle before launching."""
    by_host: dict[str, set[int]] = {}
    for sp in shards:
        if any(len(offs) > 0 for _, offs in sp.tasks):
            by_host.setdefault(sp.host, set()).add(sp.gpu_id)

    for host, gpu_ids in sorted(by_host.items()):
        h_low = host.lower()
        if any(bad in h_low for bad in FORBIDDEN_HOSTS):
            raise RuntimeError(f"Refusing forbidden host {host}")
        res = subprocess.run(
            [
                "ssh", "-A", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                host,
                "nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits",
            ],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        for line in res.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3 and parts[0].isdigit():
                g_idx = int(parts[0])
                util = float(parts[1])
                mem_mib = float(parts[2])
                if g_idx in gpu_ids and (util > 20.0 or mem_mib > 2048.0):
                    raise RuntimeError(
                        f"Target GPU {host}:{g_idx} is NOT idle (util={util}%, mem={mem_mib} MiB) — aborting launch!"
                    )


def dispatch_fleet_4way(
    num_seeds: int,
    seed_start: int,
    wave: str,
    out_dir: Path,
    data_dir: Path,
) -> None:
    """Sync code + staged `.npz` corpora and execute 4-way single-SKU sweep on `acorn:0,1 + cashew:0,1`."""
    shards = build_fleet_shards(num_seeds=num_seeds)
    cells = build_compression_cells(smoke=False)
    cell_by_id = {c.cell_id: c for c in cells}

    env = os.environ.copy()
    live_sock = discover_live_ssh_auth_sock()
    if live_sock:
        env["SSH_AUTH_SOCK"] = live_sock

    assert_fleet_gpus_idle(shards, env)

    host_needed_npz: dict[str, set[str]] = {}
    for sp in shards:
        needed = host_needed_npz.setdefault(sp.host, set())
        for cid, offs in sp.tasks:
            if offs:
                corp_name = cell_by_id[cid].corpus_name
                needed.add(BENCHMARK_SPEC_BY_NAME[corp_name].cache_filename)

    shards_dir = out_dir / "shards"
    logs_dir = out_dir / "logs"
    shards_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    rel_out = out_dir.relative_to(ROOT) if out_dir.is_relative_to(ROOT) else Path("outputs/pq_rvq")
    remote_data_dir = f"{REMOTE_WORKDIR}/outputs/phase5_dominance/data"

    def _resolve_local_npz(npz_name: str) -> Path:
        for d_cand in (data_dir, DEFAULT_PHASE5_DATA_DIR, DEFAULT_MILLION_DATA_DIR):
            p = d_cand / npz_name
            if p.exists():
                return p
        raise FileNotFoundError(f"Missing local validated corpus archive {npz_name}")

    def _sync_host(host: str) -> None:
        print(f"[Fleet-PQRVQ] Syncing code + validated corpora to {host}:{REMOTE_WORKDIR} ...", flush=True)
        subprocess.run(
            [
                "ssh", "-A", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                host,
                f"mkdir -p /home/catid/emeans_env/tmp {REMOTE_WORKDIR}/outputs/logs {remote_data_dir}",
            ],
            env=env,
            check=True,
        )
        subprocess.run(
            [
                "rsync", "-az", "--delete",
                "-e", "ssh -A -o BatchMode=yes -o ConnectTimeout=8",
                "--exclude", ".venv/",
                "--exclude", "outputs/",
                "--exclude", "__pycache__/",
                "--exclude", ".pytest_cache/",
                f"{ROOT}/",
                f"{host}:{REMOTE_WORKDIR}/",
            ],
            env=env,
            check=True,
        )
        for npz_name in sorted(host_needed_npz[host]):
            link_cmd = (
                f"if [ ! -f {remote_data_dir}/{npz_name} ]; then "
                f"  if [ -f {REMOTE_P5_DATA}/{npz_name} ]; then ln -f {REMOTE_P5_DATA}/{npz_name} {remote_data_dir}/{npz_name} || cp -f {REMOTE_P5_DATA}/{npz_name} {remote_data_dir}/{npz_name}; "
                f"  elif [ -f {REMOTE_P4_DATA}/{npz_name} ]; then ln -f {REMOTE_P4_DATA}/{npz_name} {remote_data_dir}/{npz_name} || cp -f {REMOTE_P4_DATA}/{npz_name} {remote_data_dir}/{npz_name}; fi; "
                f"fi; test -s {remote_data_dir}/{npz_name}"
            )
            chk = subprocess.run(
                ["ssh", "-A", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, link_cmd],
                env=env,
                check=False,
            )
            if chk.returncode != 0:
                src_npz = _resolve_local_npz(npz_name)
                subprocess.run(
                    [
                        "rsync", "-az", "--ignore-existing",
                        "-e", "ssh -A -o BatchMode=yes -o ConnectTimeout=12",
                        str(src_npz),
                        f"{host}:{remote_data_dir}/{npz_name}",
                    ],
                    env=env,
                    check=True,
                )
        print(f"[Fleet-PQRVQ] Host {host} ready.", flush=True)

    remote_hosts = sorted(host_needed_npz.keys())
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(remote_hosts)) as pool:
        list(pool.map(_sync_host, remote_hosts))

    active_shards = [sp for sp in shards if any(len(offs) > 0 for _, offs in sp.tasks)]
    num_shards = len(shards)

    for sp in active_shards:
        remote_out = f"{REMOTE_WORKDIR}/{rel_out}"
        shard_fname = f"wave_{wave}_shard_{sp.shard_idx:02d}_of_{num_shards:02d}.json"
        remote_json = f"{remote_out}/shards/{shard_fname}"
        sess = f"{sp.session}_{wave}"
        if sess == "0":
            raise RuntimeError("Forbidden tmux session 0")
        remote_cmd = (
            f"mkdir -p /home/catid/emeans_env/tmp/triton_cache_{sess} {REMOTE_WORKDIR}/outputs/logs {remote_out}/shards && "
            f"rm -f {REMOTE_WORKDIR}/outputs/logs/{sess}.status {remote_json} && "
            f"tmux kill-session -t {sess} 2>/dev/null || true; "
            f"tmux new-session -d -s {sess} "
            + shlex.quote(
                f"export PATH=\"$HOME/.local/bin:/usr/local/cuda/bin:$PATH\"; "
                f"export CUDA_VISIBLE_DEVICES={sp.gpu_id}; "
                f"export TMPDIR=/home/catid/emeans_env/tmp; "
                f"export TRITON_CACHE_DIR=/home/catid/emeans_env/tmp/triton_cache_{sess}; "
                f"export PYTHONPATH={REMOTE_WORKDIR}/src; "
                f"cd {REMOTE_WORKDIR} && "
                f"{REMOTE_PYTHON} scripts/run_pq_rvq_benchmarks.py "
                f"--shard-idx {sp.shard_idx} --num-shards {num_shards} "
                f"--num-seeds {num_seeds} --seed-start {seed_start} --wave {wave} "
                f"--data-dir {remote_data_dir} --out-dir {remote_out} "
                f"> {REMOTE_WORKDIR}/outputs/logs/{sess}.log 2>&1; "
                f"echo EXIT_CODE=$? > {REMOTE_WORKDIR}/outputs/logs/{sess}.status"
            )
        )
        subprocess.run(
            ["ssh", "-A", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", sp.host, remote_cmd],
            env=env,
            check=True,
        )
        print(
            f"[Fleet-PQRVQ] Launched REMOTE shard {sp.shard_idx:02d}/{num_shards} ({wave}) on {sp.host}:{sp.gpu_id} ({sess})",
            flush=True,
        )

    pending = {sp.shard_idx: sp for sp in active_shards}
    t0 = time.perf_counter()
    while pending:
        time.sleep(3.5)
        done_ids: list[int] = []
        for sidx, sp in list(pending.items()):
            sess = f"{sp.session}_{wave}"
            chk = subprocess.run(
                ["ssh", "-A", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", sp.host, f"cat {REMOTE_WORKDIR}/outputs/logs/{sess}.status 2>/dev/null || true"],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            txt = chk.stdout.strip()
            if "EXIT_CODE=" in txt:
                code = int(txt.split("EXIT_CODE=", 1)[1].splitlines()[0].strip())
                log_tail = subprocess.run(
                    ["ssh", "-A", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", sp.host, f"tail -n 60 {REMOTE_WORKDIR}/outputs/logs/{sess}.log 2>/dev/null || true"],
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                ).stdout
                atomic_write_text(logs_dir / f"{sess}.log", log_tail)
                if code != 0:
                    raise RuntimeError(f"Remote shard {sidx:02d} ({sp.host} {sess}) failed (code={code}):\n{log_tail}")
                shard_fname = f"wave_{wave}_shard_{sidx:02d}_of_{num_shards:02d}.json"
                remote_json = f"{REMOTE_WORKDIR}/{rel_out}/shards/{shard_fname}"
                local_json = shards_dir / shard_fname
                subprocess.run(
                    ["rsync", "-az", "-e", "ssh -A -o BatchMode=yes -o ConnectTimeout=8", f"{sp.host}:{remote_json}", str(local_json)],
                    env=env,
                    check=True,
                )
                print(
                    f"[Fleet-PQRVQ] Collected shard {sidx:02d} ({sp.host}:{sp.gpu_id}, wave={wave}) at +{time.perf_counter() - t0:.1f}s",
                    flush=True,
                )
                done_ids.append(sidx)
        for d_i in done_ids:
            pending.pop(d_i, None)


# ==============================================================================
# CLI Entrypoint
# ==============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fused PQ (`ApexPQ`) & 2-Stage RVQ (`ApexRVQ`) Benchmark Runner")
    p.add_argument("--smoke", action="store_true", help="Run fast 2-seed cold-JIT-verified smoke gate")
    p.add_argument("--dispatch-fleet", action="store_true", help="Dispatch 4-GPU 600W Server Blackwell fleet sweep")
    p.add_argument("--merge-only", action="store_true", help="Merge shard JSONs, validate, and render MD + PNGs")
    p.add_argument("--wave", choices=("all", "baselines", "apex"), default="all", help="Execution wave")
    p.add_argument("--shard-idx", type=int, default=None, help="Worker shard index (0..3)")
    p.add_argument("--num-shards", type=int, default=4, help="Total fleet shards (default: 4)")
    p.add_argument("--num-seeds", type=int, default=5, help="Paired seeds per cell (default: 5)")
    p.add_argument("--seed-start", type=int, default=6000, help="Starting RNG seed (default: 6000)")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_PHASE5_DATA_DIR, help="Validated corpus cache dir")
    p.add_argument("--out-dir", type=Path, default=OUTPUT_DIR, help="Output directory for JSON, MD, and PNGs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir: Path = args.out_dir
    data_dir: Path = args.data_dir
    wave: str = args.wave

    if args.smoke:
        seeds = [int(args.seed_start) + i for i in range(2)]
        budgets: tuple[float, ...] = (1.0, 2.0)
        cells = build_compression_cells(smoke=True)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        warmup_global_kernels(dev)

        records: list[dict[str, Any]] = []
        for cell in cells:
            records.extend(execute_cell_seeds(cell, seeds, budgets, wave=wave, data_dir=data_dir, device=dev))

        spikes = validate_records_and_detect_spikes(records, cells, seeds, budgets, wave=wave)
        if spikes:
            raise RuntimeError(f"COLD-JIT SPIKES DETECTED IN SMOKE GATE: {spikes!r}")

        comparisons = compute_paired_summary(records, cells, budgets)
        payload = {
            "phase": "pq_rvq_smoke",
            "wave": wave,
            "num_records": len(records),
            "num_cold_jit_spikes": len(spikes),
            "cold_jit_spikes": spikes,
            "seeds": seeds,
            "budgets": list(budgets),
            "cells": [asdict(c) for c in cells],
            "records": records,
            "comparisons": comparisons,
        }
        atomic_write_json(out_dir / "pq_rvq_results.json", payload)
        write_summary_markdown(records, comparisons, spikes, cells, budgets, out_dir / "pq_rvq_summary.md")
        plots = generate_all_plots(records, cells, budgets, out_dir)
        print(
            f"[PQ-RVQ Smoke] SUCCESS: {len(records)} records, 0 cold-JIT spikes, {len(plots)} PNGs in {out_dir}",
            flush=True,
        )
        return

    cells = build_compression_cells(smoke=False)
    cell_by_id = {c.cell_id: c for c in cells}
    num_seeds = int(args.num_seeds)
    seed_start = int(args.seed_start)
    all_seeds = [seed_start + i for i in range(num_seeds)]
    budgets = EPOCH_BUDGETS

    if args.shard_idx is not None:
        if int(args.num_shards) != 4:
            raise ValueError("Expected --num-shards 4 matching build_fleet_shards")
        shards = build_fleet_shards(num_seeds=num_seeds)
        sp = shards[int(args.shard_idx)]
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        warmup_global_kernels(dev)

        s_records: list[dict[str, Any]] = []
        for cid, offs in sp.tasks:
            c_seeds = [seed_start + o for o in offs]
            s_records.extend(execute_cell_seeds(cell_by_id[cid], c_seeds, budgets, wave=wave, data_dir=data_dir, device=dev))

        shard_path = out_dir / "shards" / f"wave_{wave}_shard_{sp.shard_idx:02d}_of_{int(args.num_shards):02d}.json"
        atomic_write_json(
            shard_path,
            {
                "shard_idx": sp.shard_idx,
                "num_shards": int(args.num_shards),
                "wave": wave,
                "host": sp.host,
                "gpu_id": sp.gpu_id,
                "sku_tier": sp.sku_tier,
                "records": s_records,
            },
        )
        print(f"[PQ-RVQ Shard {sp.shard_idx:02d}] Wrote {len(s_records)} records -> {shard_path}", flush=True)
        return

    if args.dispatch_fleet:
        dispatch_fleet_4way(num_seeds=num_seeds, seed_start=seed_start, wave=wave, out_dir=out_dir, data_dir=data_dir)

    if args.dispatch_fleet or args.merge_only:
        shards_dir = out_dir / "shards"
        shards = build_fleet_shards(num_seeds=num_seeds)
        active_shards = [sp for sp in shards if any(len(offs) > 0 for _, offs in sp.tasks)]

        # Auto-detect whether `wave_all` shards exist or `wave_baselines + wave_apex` shards exist
        have_all = all((shards_dir / f"wave_all_shard_{sp.shard_idx:02d}_of_04.json").exists() for sp in active_shards)
        have_base = all((shards_dir / f"wave_baselines_shard_{sp.shard_idx:02d}_of_04.json").exists() for sp in active_shards)
        have_apex = all((shards_dir / f"wave_apex_shard_{sp.shard_idx:02d}_of_04.json").exists() for sp in active_shards)

        if wave == "all" and have_all:
            waves_to_load = ["all"]
            eff_wave = "all"
        elif wave == "all" and have_base and have_apex:
            waves_to_load = ["baselines", "apex"]
            eff_wave = "all"
        elif wave == "baselines" and have_base:
            waves_to_load = ["baselines"]
            eff_wave = "baselines"
        elif wave == "apex" and have_base and have_apex:
            waves_to_load = ["baselines", "apex"]
            eff_wave = "all"
        elif wave == "apex" and have_apex:
            waves_to_load = ["apex"]
            eff_wave = "apex"
        else:
            raise FileNotFoundError(f"Incomplete shard set in {shards_dir} for wave={wave!r}")

        merged: list[dict[str, Any]] = []
        for w_tag in waves_to_load:
            for sp in active_shards:
                spath = shards_dir / f"wave_{w_tag}_shard_{sp.shard_idx:02d}_of_04.json"
                with open(spath, "r", encoding="utf-8") as f:
                    data = json.load(
                        f,
                        parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"Invalid JSON constant {c}")),
                    )
                merged.extend(data["records"])

        spikes = validate_records_and_detect_spikes(merged, cells, all_seeds, budgets, wave=eff_wave)
        if spikes:
            raise RuntimeError(f"COLD-JIT TIMING SPIKES DETECTED: {spikes!r}")

        comparisons = compute_paired_summary(merged, cells, budgets)
        payload = {
            "phase": "pq_rvq_compression",
            "wave": eff_wave,
            "num_records": len(merged),
            "num_cold_jit_spikes": len(spikes),
            "cold_jit_spikes": spikes,
            "seeds": all_seeds,
            "budgets": list(budgets),
            "cells": [asdict(c) for c in cells],
            "records": merged,
            "comparisons": comparisons,
        }
        atomic_write_json(out_dir / "pq_rvq_results.json", payload)
        if eff_wave == "all":
            write_summary_markdown(merged, comparisons, spikes, cells, budgets, out_dir / "pq_rvq_summary.md")
            plots = generate_all_plots(merged, cells, budgets, out_dir)
            print(
                f"[PQ-RVQ Complete] Validated {len(merged):,} records, 0 cold-JIT spikes, & {len(plots)} PNGs in {out_dir}",
                flush=True,
            )
        else:
            print(
                f"[PQ-RVQ Wave '{eff_wave}' Staged] Validated {len(merged):,} records (0 cold-JIT spikes) in {out_dir}",
                flush=True,
            )
        return

    # Local single-GPU execution fallback
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    warmup_global_kernels(dev)
    loc_records: list[dict[str, Any]] = []
    for cell in cells:
        loc_records.extend(execute_cell_seeds(cell, all_seeds, budgets, wave=wave, data_dir=data_dir, device=dev))
    spikes = validate_records_and_detect_spikes(loc_records, cells, all_seeds, budgets, wave=wave)
    if spikes:
        raise RuntimeError(f"COLD-JIT SPIKES DETECTED: {spikes!r}")
    comparisons = compute_paired_summary(loc_records, cells, budgets)
    atomic_write_json(
        out_dir / "pq_rvq_results.json",
        {
            "phase": "pq_rvq_local",
            "wave": wave,
            "num_records": len(loc_records),
            "num_cold_jit_spikes": len(spikes),
            "cold_jit_spikes": spikes,
            "seeds": all_seeds,
            "budgets": list(budgets),
            "cells": [asdict(c) for c in cells],
            "records": loc_records,
            "comparisons": comparisons,
        },
    )
    write_summary_markdown(loc_records, comparisons, spikes, cells, budgets, out_dir / "pq_rvq_summary.md")
    generate_all_plots(loc_records, cells, budgets, out_dir)


if __name__ == "__main__":
    main()
