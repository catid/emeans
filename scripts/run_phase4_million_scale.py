#!/usr/bin/env python3
"""Phase 4 Multi-Machine Million-Scale Vector Clustering & Downstream ANN IVF Recall Campaign.

Evaluates `Lloyd@25`, `EMeans` (arXiv:2611.09501), `EMeansPP`, `SMeans` (`suayptalha/s-means`),
`SuperEMeans`, `HybridSEMeans` (`wide=1, 2`), and `UltraEMeans` (`wide=1, 2`) at **true paper scale
(`N = 1,000,000`, `K in {4,096, 16,384}`, `T in {1.0, 2.0, 3.0, 5.0}`)** on genuine real-world
vector benchmarks AND downstream **Approximate Nearest Neighbor (ANN) IVF-Flat Recall@{1, 10, 100}**
(reproducing & extending Appendix K of Bessa et al., arXiv:2611.09501):

  1. **SIFT1M (`N = 1,000,000, D = 128`, Euclidean `L_2^2`)**:
     - `1,000,000 x 128` base descriptors, `89,788` clean deduplicated validation vectors,
       `10,000` official held-out test queries, and `10,000 x 100` exact nearest-neighbor ground truth.
  2. **GloVe-Twitter-1M-Unit (`N = 1,000,000, D = 100`, Unit-Sphere Cosine/Angular)**:
     - `1,000,000 x 100` unit-normalized semantic word embeddings, `100,000` validation vectors,
       `10,000` held-out test queries, and `10,000 x 100` exact FP64 ground truth (strictly deduplicated,
       evaluating BOTH exact FP64 Euclidean MSE and exact FP64 spherical cosine distortion).

Strict Benchmarking & De-Confounding Hygiene (`AGENTS.md` Rules 1, 2, 3):
  - **Zero Train/Val/Test Leakage**: Solvers fit strictly on `corpus.base` (`train`); `corpus.val` and
    `corpus.query` (`test`) are never passed to `.fit(...)`.
  - **100% Single-SKU Cell Pinning & Same-GPU Seed Pairing**:
      * `sift1m_k16384`  -> `acorn:0,1` + `cashew:0,1` (100% RTX PRO 6000 Blackwell Server 600W)
      * `sift1m_k4096`   -> `work:0,1`                 (100% RTX PRO 6000 Blackwell Workstation 450W)
      * `glove1m_k16384` -> `ripper:0..3`              (100% RTX PRO 6000 Blackwell Max-Q 300W)
      * `glove1m_k4096`  -> `loki:0,1`                 (100% RTX PRO 6000 Blackwell Max-Q 300W)
    All 33 runs of each `(cell_id, seed)` (`Lloyd@25` + `8` streaming arms x `4` epoch budgets) execute
    on the exact same physical GPU with cyclic per-seed arm-order counterbalancing.
  - **Zero Cold-JIT Spikes**: Untimed full-shape per-cell GPU warmup (including all 36 `_local_greedy_kpp`
    and 14 coreset `_fused_assign` alignment buckets per `#111`) + per-shard `TRITON_CACHE_DIR` isolation.
  - **Safety**: `walnut.stratabaremetal.com` and `tmux` session `0` are strictly forbidden; all JSON/MD/PNG
    writes use PID+ns temp files + `os.replace` with `allow_nan=False`.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import glob
import json
import math
import os
import shlex
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Isolate Triton's disk cache under /home BEFORE importing torch/triton/emeans
if "TRITON_CACHE_DIR" not in os.environ:
    _vis_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "all").replace(",", "_")
    _base_tmp = (
        "/home/catid/emeans_env/tmp"
        if os.path.isdir("/home/catid/emeans_env")
        else os.path.expanduser("~/.cache/emeans_tmp")
    )
    os.makedirs(_base_tmp, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = f"{_base_tmp}/triton_cache_p4_mscale_u{os.getuid()}_g{_vis_gpu}"

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

from emeans.baselines import LloydKMeans
from emeans.emeans import EMeans
from emeans.emeans_pp import EMeansPP
from emeans.kernels import assign_points
from emeans.real_corpora import (
    DEFAULT_MILLION_DATA_DIR,
    RealVectorCorpus,
    compute_exact_groundtruth_gpu,
    evaluate_ivf_recall_gpu,
    exact_cosine_distortion_fp64,
    fast_exact_assign_fp64,
    load_glove1m_unit,
    load_sift1m,
    validate_real_corpus,
)
from emeans.seeding import HAS_TRITON as SEED_HAS_TRITON
from emeans.smeans import SMeans
from emeans.super_emeans import SuperEMeans, build_hybrid_arm
from emeans.ultra_emeans import build_ultra_arm

if SEED_HAS_TRITON:
    from emeans.seeding import _local_greedy_kpp_kernel


OUTPUT_DIR = ROOT / "outputs" / "phase4_million_scale"
REMOTE_WORKDIR = "/home/catid/emeans_runs/real-scale"
REMOTE_PYTHON = "/home/catid/emeans_env/.venv/bin/python"

EPOCH_BUDGETS: tuple[float, ...] = (1.0, 2.0, 3.0, 5.0)
SELECTIVITIES: tuple[float, ...] = (0.001, 0.005, 0.010)
SELECTIVITY_KEYS: tuple[str, ...] = ("0.1%", "0.5%", "1%")

STREAMING_ARMS: tuple[str, ...] = (
    "emeans",
    "emeans_pp",
    "smeans",
    "super_emeans",
    "hybrid_semeans",
    "hybrid_semeans_wide",
    "ultra_emeans",
    "ultra_emeans_wide",
)
ALL_ARMS: tuple[str, ...] = ("lloyd_25",) + STREAMING_ARMS


@dataclass(frozen=True)
class MillionCellSpec:
    """One `(corpus, n_base, d, k)` experimental cell pinned to a single GPU SKU."""

    cell_id: str
    corpus_name: str      # "sift1m", "glove1m_unit", or smoke synthetic IDs
    n_base: int
    n_val: int
    n_query: int
    d: int
    k: int
    geometry: str         # "euclidean" or "unit_sphere"
    sku_tier: str         # expected single-SKU family tag for de-confounding


@dataclass(frozen=True)
class FleetShardSpec:
    """Explicit single-SKU shard assignment `(shard_idx, host, gpu_id, session, cell_id, seed_offsets)`."""

    shard_idx: int
    host: str
    gpu_id: int
    session: str
    sku_tier: str
    cell_id: str
    seed_offsets: tuple[int, ...]


def build_fleet_shards(num_seeds: int) -> list[FleetShardSpec]:
    """Partition `num_seeds` across 12 remote GPUs so every `(corpus, K)` cell stays on 1 exact GPU SKU."""
    def _split_offsets(n_total: int, n_ways: int) -> list[tuple[int, ...]]:
        buckets: list[list[int]] = [[] for _ in range(n_ways)]
        for s_off in range(n_total):
            buckets[s_off % n_ways].append(s_off)
        return [tuple(b) for b in buckets]

    sift16_splits = _split_offsets(num_seeds, 4)   # acorn:0,1 + cashew:0,1 (600W Server)
    sift4_splits = _split_offsets(num_seeds, 2)    # work:0,1 (450W Workstation)
    glove16_splits = _split_offsets(num_seeds, 4)  # ripper:0..3 (300W Max-Q)
    glove4_splits = _split_offsets(num_seeds, 2)   # loki:0,1 (300W Max-Q)

    specs = [
        FleetShardSpec(0, "acorn.stratabaremetal.com", 0, "emeans_p4_mscale_s00", "server_600w", "sift1m_k16384", sift16_splits[0]),
        FleetShardSpec(1, "acorn.stratabaremetal.com", 1, "emeans_p4_mscale_s01", "server_600w", "sift1m_k16384", sift16_splits[1]),
        FleetShardSpec(2, "cashew.stratabaremetal.com", 0, "emeans_p4_mscale_s02", "server_600w", "sift1m_k16384", sift16_splits[2]),
        FleetShardSpec(3, "cashew.stratabaremetal.com", 1, "emeans_p4_mscale_s03", "server_600w", "sift1m_k16384", sift16_splits[3]),
        FleetShardSpec(4, "work", 0, "emeans_p4_mscale_s04", "workstation_450w", "sift1m_k4096", sift4_splits[0]),
        FleetShardSpec(5, "work", 1, "emeans_p4_mscale_s05", "workstation_450w", "sift1m_k4096", sift4_splits[1]),
        FleetShardSpec(6, "ripper", 0, "emeans_p4_mscale_s06", "maxq_300w", "glove1m_k16384", glove16_splits[0]),
        FleetShardSpec(7, "ripper", 1, "emeans_p4_mscale_s07", "maxq_300w", "glove1m_k16384", glove16_splits[1]),
        FleetShardSpec(8, "ripper", 2, "emeans_p4_mscale_s08", "maxq_300w", "glove1m_k16384", glove16_splits[2]),
        FleetShardSpec(9, "ripper", 3, "emeans_p4_mscale_s09", "maxq_300w", "glove1m_k16384", glove16_splits[3]),
        FleetShardSpec(10, "loki", 0, "emeans_p4_mscale_s10", "maxq_300w", "glove1m_k4096", glove4_splits[0]),
        FleetShardSpec(11, "loki", 1, "emeans_p4_mscale_s11", "maxq_300w", "glove1m_k4096", glove4_splits[1]),
    ]
    return specs


def build_cells(smoke: bool = False) -> list[MillionCellSpec]:
    """Return the 4 million-scale real cells (`N=1,000,000`), or 2 fast `k=512` smoke cells."""
    if smoke:
        return [
            MillionCellSpec(
                cell_id="sift1m_k512_smoke",
                corpus_name="smoke_sift_sub",
                n_base=8192,
                n_val=1024,
                n_query=256,
                d=128,
                k=512,  # k=512 > 256 strictly exercises block_greedy_kmeans_plusplus_torch (#111)
                geometry="euclidean",
                sku_tier="smoke_local",
            ),
            MillionCellSpec(
                cell_id="glove1m_k512_smoke",
                corpus_name="smoke_glove_sub",
                n_base=8192,
                n_val=1024,
                n_query=256,
                d=100,
                k=512,
                geometry="unit_sphere",
                sku_tier="smoke_local",
            ),
        ]

    return [
        MillionCellSpec("sift1m_k4096", "sift1m", 1_000_000, 89_788, 10_000, 128, 4096, "euclidean", "workstation_450w"),
        MillionCellSpec("sift1m_k16384", "sift1m", 1_000_000, 89_788, 10_000, 128, 16384, "euclidean", "server_600w"),
        MillionCellSpec("glove1m_k4096", "glove1m_unit", 1_000_000, 100_000, 10_000, 100, 4096, "unit_sphere", "maxq_300w"),
        MillionCellSpec("glove1m_k16384", "glove1m_unit", 1_000_000, 100_000, 10_000, 100, 16384, "unit_sphere", "maxq_300w"),
    ]


# ==============================================================================
# Solver Instantiation & Untimed Full-Shape GPU Warmup (Zero Cold-JIT Spikes)
# ==============================================================================


def instantiate_arm(arm: str, d: int, k: int, nepochs: float, seed: int) -> Any:
    """Construct a clustering solver with identical `(d, k, nepochs, seed)` contract."""
    if arm == "lloyd_25":
        return LloydKMeans(d=d, k=k, niter=25, init="random", seed=seed)
    if arm == "emeans":
        return EMeans(d=d, k=k, nepochs=nepochs, seed=seed)
    if arm == "emeans_pp":
        return EMeansPP(d=d, k=k, nepochs=nepochs, seed=seed)
    if arm == "smeans":
        return SMeans(d=d, k=k, nepochs=nepochs, exact_ref=False, seed=seed)
    if arm == "super_emeans":
        return SuperEMeans(d=d, k=k, nepochs=nepochs, wide_step_mult=1, seed=seed)
    if arm == "hybrid_semeans":
        return build_hybrid_arm("hybrid_super_smeans", d=d, k=k, nepochs=nepochs, seed=seed)
    if arm == "hybrid_semeans_wide":
        return build_hybrid_arm("hybrid_super_smeans_wide", d=d, k=k, nepochs=nepochs, seed=seed)
    if arm == "ultra_emeans":
        return build_ultra_arm("ultra_emeans", d=d, k=k, nepochs=nepochs, seed=seed)
    if arm == "ultra_emeans_wide":
        return build_ultra_arm("ultra_emeans_wide", d=d, k=k, nepochs=nepochs, seed=seed)
    raise ValueError(f"Unknown arm {arm!r}")


@torch.no_grad()
def warmup_global_kernels(device: torch.device) -> None:
    """Pre-warm seed-dependent Triton coreset alignment buckets (`#111`) outside all timers."""
    if device.type != "cuda" or not SEED_HAS_TRITON:
        return
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
    torch.cuda.synchronize(device)


@torch.no_grad()
def warmup_cell_kernels(
    x_base_gpu: torch.Tensor,
    x_val_gpu: torch.Tensor,
    x_query_gpu: torch.Tensor,
    corpus: RealVectorCorpus,
    cell: MillionCellSpec,
    budgets: tuple[float, ...],
    device: torch.device,
) -> None:
    """Compile every Triton kernel specialization on the exact `(N, D, K)` cell tensors outside timers."""
    if device.type != "cuda":
        return

    # 1. Pre-warm `_fused_assign_kernel` on coreset `m_sub` for `% 16 == 0` and `% 16 != 0` (#111)
    m_sub = min(int(x_base_gpu.shape[0]), max(8192, 16 * cell.k, cell.k * 2))
    x_m = x_base_gpu[:m_sub].float().contiguous()
    for dt in (torch.float32, torch.bfloat16):
        for k_cand in (1, 16, 17, 64, 65, 128, 129):
            if k_cand <= m_sub:
                assign_points(x_m, x_m[:k_cand], compute_dtype=dt)

    # 2. Pre-warm Lloyd + all 8 streaming arms across every budget in `budgets`
    #    (compiles exact batch_size B_t and 2*B_t wide_step kernels for every T in {1, 2, 3, 5}!)
    lloyd_warm = LloydKMeans(d=cell.d, k=cell.k, niter=2, init="random", seed=999_991)
    c_warm, _ = lloyd_warm.fit(x_base_gpu, capture_assignments=False)

    for b_t in budgets:
        for arm in STREAMING_ARMS:
            solver = instantiate_arm(arm, d=cell.d, k=cell.k, nepochs=b_t, seed=999_992)
            c_warm, _ = solver.fit(x_base_gpu, capture_assignments=False)

    # 3. Pre-warm exact FP64 evaluators and CSR GPU IVF Recall kernel (both <=128 and >128 bucket paths)
    _, db_lbl = fast_exact_assign_fp64(x_base_gpu, c_warm, device=device)
    _, _ = fast_exact_assign_fp64(x_val_gpu, c_warm, device=device)
    _, _ = fast_exact_assign_fp64(x_query_gpu, c_warm, device=device)
    if cell.geometry == "unit_sphere":
        _ = exact_cosine_distortion_fp64(x_query_gpu, c_warm, device=device)

    _ = evaluate_ivf_recall_gpu(
        x_base=x_base_gpu,
        centroids=c_warm,
        x_query=x_query_gpu[: min(256, x_query_gpu.shape[0])],
        groundtruth_ids=corpus.groundtruth_ids[: min(256, x_query_gpu.shape[0])],
        groundtruth_sq_dists=corpus.groundtruth_sq_dists[: min(256, x_query_gpu.shape[0])],
        selectivities=SELECTIVITIES,
        db_labels=db_lbl,
        device=device,
    )
    torch.cuda.synchronize(device)


# ==============================================================================
# Corpus Loading (Real Million-Scale & Smoke Subsets)
# ==============================================================================


def load_cell_corpus(cell: MillionCellSpec, data_dir: Path, device: torch.device) -> RealVectorCorpus:
    """Load and validate the real corpus for `cell` (or deterministic smoke subset)."""
    if cell.corpus_name == "sift1m":
        return load_sift1m(data_dir=data_dir, device=device)
    if cell.corpus_name == "glove1m_unit":
        return load_glove1m_unit(data_dir=data_dir, device=device)

    if cell.corpus_name == "smoke_sift_sub":
        if (data_dir / "sift1m_validated_v1.npz").exists():
            full = load_sift1m(data_dir=data_dir, device=device)
            xb = full.base[: cell.n_base].copy()
            xv = full.val[: cell.n_val].copy()
            xq = full.query[: cell.n_query].copy()
        else:
            rng = np.random.default_rng(7001)
            xb = rng.uniform(0.0, 200.0, size=(cell.n_base, cell.d)).astype(np.float32)
            xv = rng.uniform(0.0, 200.0, size=(cell.n_val, cell.d)).astype(np.float32)
            xq = rng.uniform(0.0, 200.0, size=(cell.n_query, cell.d)).astype(np.float32)
        gt_ids, gt_d2 = compute_exact_groundtruth_gpu(xb, xq, top_k=100, device=device)
        corp = RealVectorCorpus("sift1m_smoke", xb, xv, xq, gt_ids, gt_d2, cell.d, "euclidean")
        validate_real_corpus(corp)
        return corp

    if cell.corpus_name == "smoke_glove_sub":
        if (data_dir / "glove1m_unit_s20261116_validated_v1.npz").exists():
            full = load_glove1m_unit(data_dir=data_dir, device=device)
            xb = full.base[: cell.n_base].copy()
            xv = full.val[: cell.n_val].copy()
            xq = full.query[: cell.n_query].copy()
        else:
            rng = np.random.default_rng(7002)
            raw = rng.standard_normal((cell.n_base + cell.n_val + cell.n_query, cell.d))
            unit = (raw / np.linalg.norm(raw, axis=1, keepdims=True)).astype(np.float32)
            xb = unit[: cell.n_base].copy()
            xv = unit[cell.n_base : cell.n_base + cell.n_val].copy()
            xq = unit[cell.n_base + cell.n_val :].copy()
        gt_ids, gt_d2 = compute_exact_groundtruth_gpu(xb, xq, top_k=100, device=device)
        corp = RealVectorCorpus("glove1m_smoke", xb, xv, xq, gt_ids, gt_d2, cell.d, "unit_sphere")
        validate_real_corpus(corp)
        return corp

    raise ValueError(f"Unknown cell.corpus_name={cell.corpus_name!r}")


# ==============================================================================
# Single-Cell / Shard Execution
# ==============================================================================


def atomic_write_json(path: Path, payload: Any) -> None:
    """Atomically write JSON with `allow_nan=False` via PID+ns temp file + `os.replace`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".tmp_{os.getpid()}_{time.time_ns()}_{path.name}"
    try:
        text = json.dumps(payload, indent=2, allow_nan=False)
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def detect_gpu_sku_info(device: torch.device) -> tuple[str, float]:
    """Return `(gpu_name, power_limit_watts)` to uniquely identify Blackwell TDP SKU tier."""
    if device.type != "cuda":
        return "CPU", 0.0
    name = torch.cuda.get_device_name(device)
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
    power_w = 0.0
    try:
        cmd = ["nvidia-smi", "-i", vis if vis.isdigit() else "0", "--query-gpu=power.max_limit", "--format=csv,noheader,nounits"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=3, check=False)
        if res.returncode == 0:
            val_s = res.stdout.strip().splitlines()[0].strip()
            power_w = float(val_s)
    except Exception:
        power_w = 0.0
    sku_label = f"{name} ({int(round(power_w))}W)" if power_w > 0 else name
    return sku_label, power_w


@torch.no_grad()
def execute_cell_seeds(
    cell: MillionCellSpec,
    seeds: list[int],
    budgets: tuple[float, ...],
    data_dir: Path,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Run all `seeds` for `cell` on `device` with full warmup and cyclic arm counterbalancing."""
    if not seeds:
        return []

    host = socket.gethostname()
    vis_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    gpu_sku, power_w = detect_gpu_sku_info(device)

    corpus = load_cell_corpus(cell, data_dir=data_dir, device=device)
    x_base_gpu = torch.from_numpy(corpus.base).to(device=device, dtype=torch.float32).contiguous()
    x_val_gpu = torch.from_numpy(corpus.val).to(device=device, dtype=torch.float32).contiguous()
    x_query_gpu = torch.from_numpy(corpus.query).to(device=device, dtype=torch.float32).contiguous()

    print(
        f"[Phase4] Warming kernels for {cell.cell_id} (N={cell.n_base:,}, D={cell.d}, K={cell.k:,}) "
        f"on {host} GPU={vis_gpu} [{gpu_sku}] ...",
        flush=True,
    )
    t_w0 = time.perf_counter()
    warmup_cell_kernels(
        x_base_gpu=x_base_gpu,
        x_val_gpu=x_val_gpu,
        x_query_gpu=x_query_gpu,
        corpus=corpus,
        cell=cell,
        budgets=budgets,
        device=device,
    )
    print(f"[Phase4] Warmup complete for {cell.cell_id} in {time.perf_counter() - t_w0:.2f}s", flush=True)

    # Canonical list of `(arm, nepochs)` run specs per seed (`1` Lloyd@25 + `8 * len(budgets)` streaming)
    base_specs: list[tuple[str, float]] = [("lloyd_25", 25.0)]
    for b_t in budgets:
        for arm in STREAMING_ARMS:
            base_specs.append((arm, float(b_t)))

    records: list[dict[str, Any]] = []
    for s_idx, seed in enumerate(seeds):
        t_seed_start = time.perf_counter()
        # Cyclic counterbalancing across seeds so no arm systematically runs first or last
        rot = int(seed) % len(base_specs)
        ordered_specs = base_specs[rot:] + base_specs[:rot]

        seed_records_by_key: dict[tuple[str, float], dict[str, Any]] = {}
        for arm_name, nepochs in ordered_specs:
            solver = instantiate_arm(arm_name, d=cell.d, k=cell.k, nepochs=nepochs, seed=seed)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            centroids, _ = solver.fit(x_base_gpu, capture_assignments=False)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

            init_s = float(solver.init_time_sec)
            train_s = float(solver.train_time_sec)
            total_s = float(init_s + train_s)

            # Untimed exact FP64 MSE + sync-free GPU IVF-Flat Recall@{1, 10, 100}
            train_mse, db_labels = fast_exact_assign_fp64(x_base_gpu, centroids, device=device)
            val_mse, _ = fast_exact_assign_fp64(x_val_gpu, centroids, device=device)
            test_mse, _ = fast_exact_assign_fp64(x_query_gpu, centroids, device=device)

            if cell.geometry == "unit_sphere":
                val_cos_dist = exact_cosine_distortion_fp64(x_val_gpu, centroids, device=device)
                test_cos_dist = exact_cosine_distortion_fp64(x_query_gpu, centroids, device=device)
            else:
                val_cos_dist = 0.0
                test_cos_dist = 0.0

            ivf_res = evaluate_ivf_recall_gpu(
                x_base=x_base_gpu,
                centroids=centroids,
                x_query=x_query_gpu,
                groundtruth_ids=corpus.groundtruth_ids,
                groundtruth_sq_dists=corpus.groundtruth_sq_dists,
                selectivities=SELECTIVITIES,
                db_labels=db_labels,
                device=device,
            )

            m01 = ivf_res.by_selectivity["0.1%"]
            m05 = ivf_res.by_selectivity["0.5%"]
            m10 = ivf_res.by_selectivity["1%"]

            rec: dict[str, Any] = {
                "cell_id": cell.cell_id,
                "corpus": cell.corpus_name,
                "geometry": cell.geometry,
                "n_base": int(cell.n_base),
                "n_val": int(corpus.val.shape[0]),
                "n_query": int(cell.n_query),
                "d": int(cell.d),
                "k": int(cell.k),
                "seed": int(seed),
                "arm": arm_name,
                "nepochs": float(nepochs),
                "host": host,
                "visible_gpu": vis_gpu,
                "gpu_sku": gpu_sku,
                "gpu_power_w": float(power_w),
                "init_time_sec": init_s,
                "train_time_sec": train_s,
                "total_time_sec": total_s,
                "eval_time_sec": float(ivf_res.eval_time_sec),
                "train_mse": float(train_mse),
                "val_mse": float(val_mse),
                "test_mse": float(test_mse),
                "val_cos_dist": float(val_cos_dist),
                "test_cos_dist": float(test_cos_dist),
                "imbalance_factor": float(ivf_res.imbalance_factor),
                "empty_clusters": int(ivf_res.empty_clusters),
                "n_splits": int(getattr(solver, "n_splits", 0)),
                "n_transplants": int(getattr(solver, "n_transplants", 0)),
                "nprobe_01pct": int(m01.nprobe),
                "nprobe_05pct": int(m05.nprobe),
                "nprobe_10pct": int(m10.nprobe),
                "recall1_01pct": float(m01.recall_at_1),
                "recall10_01pct": float(m01.recall_at_10),
                "recall100_01pct": float(m01.recall_at_100),
                "scan_frac_01pct": float(m01.actual_scan_frac),
                "recall1_05pct": float(m05.recall_at_1),
                "recall10_05pct": float(m05.recall_at_10),
                "recall100_05pct": float(m05.recall_at_100),
                "scan_frac_05pct": float(m05.actual_scan_frac),
                "recall1_10pct": float(m10.recall_at_1),
                "recall10_10pct": float(m10.recall_at_10),
                "recall100_10pct": float(m10.recall_at_100),
                "scan_frac_10pct": float(m10.actual_scan_frac),
            }
            seed_records_by_key[(arm_name, float(nepochs))] = rec

        for key in base_specs:
            records.append(seed_records_by_key[key])

        print(
            f"[Phase4] {cell.cell_id} seed={seed} ({s_idx + 1}/{len(seeds)}) "
            f"done in {time.perf_counter() - t_seed_start:.2f}s",
            flush=True,
        )

    return records


# ==============================================================================
# Validation, Statistical Analysis, Summary Markdown & Publication Plots
# ==============================================================================


def _assert_finite_recursive(obj: Any, path: str = "root") -> None:
    if obj is None:
        raise ValueError(f"Unexpected None value at {path}")
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"Non-finite float {obj!r} at {path}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _assert_finite_recursive(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for idx, v in enumerate(obj):
            _assert_finite_recursive(v, f"{path}[{idx}]")


def validate_records(
    records: list[dict[dict[str, Any], Any]] | list[dict[str, Any]],
    cells: list[MillionCellSpec],
    expected_seeds: list[int],
    budgets: tuple[float, ...],
) -> None:
    """Strictly verify completeness, zero NaN/Inf, zero duplicates, and single-SKU de-confounding."""
    _assert_finite_recursive(records)
    runs_per_seed = 1 + len(STREAMING_ARMS) * len(budgets)
    expected_total = len(cells) * len(expected_seeds) * runs_per_seed
    if len(records) != expected_total:
        raise RuntimeError(f"Record count mismatch: got {len(records)}, expected {expected_total}")

    seen_keys: set[tuple[str, int, str, float]] = set()
    seed_gpu_map: dict[tuple[str, int], set[tuple[str, str]]] = {}
    cell_sku_map: dict[str, set[str]] = {}

    for rec in records:
        rk = (str(rec["cell_id"]), int(rec["seed"]), str(rec["arm"]), float(rec["nepochs"]))
        if rk in seen_keys:
            raise RuntimeError(f"Duplicate record key {rk!r}")
        seen_keys.add(rk)

        if abs(float(rec["total_time_sec"]) - (float(rec["init_time_sec"]) + float(rec["train_time_sec"]))) > 1e-6:
            raise RuntimeError(f"Timing sum mismatch in {rk!r}")

        sg_key = (str(rec["cell_id"]), int(rec["seed"]))
        seed_gpu_map.setdefault(sg_key, set()).add((str(rec["host"]), str(rec["visible_gpu"])))
        cell_sku_map.setdefault(str(rec["cell_id"]), set()).add(str(rec["gpu_sku"]))

    for sg_key, host_gpus in seed_gpu_map.items():
        if len(host_gpus) != 1:
            raise RuntimeError(f"CONFOUNDING VIOLATION: {sg_key} split across multiple GPUs {host_gpus!r}")

    for cell_id, skus in cell_sku_map.items():
        if len(skus) != 1:
            raise RuntimeError(f"CROSS-SKU CONFOUNDING VIOLATION: {cell_id} mixed GPU SKUs {skus!r}")


def compute_paired_comparisons(
    records: list[dict[str, Any]],
    cells: list[MillionCellSpec],
    budgets: tuple[float, ...],
) -> list[dict[str, Any]]:
    """Compute paired Student-t and Wilcoxon signed-rank statistics across seeds."""
    idx: dict[tuple[str, float, str, int], dict[str, Any]] = {}
    seeds_by_cell: dict[str, set[int]] = {}
    for r in records:
        cid = str(r["cell_id"])
        idx[(cid, float(r["nepochs"]), str(r["arm"]), int(r["seed"]))] = r
        seeds_by_cell.setdefault(cid, set()).add(int(r["seed"]))

    comparisons: list[dict[str, Any]] = []
    candidate_arms = ("ultra_emeans", "ultra_emeans_wide", "hybrid_semeans", "super_emeans", "smeans", "emeans_pp")
    baselines = ("emeans", "smeans", "super_emeans", "hybrid_semeans", "lloyd_25")
    metrics = (
        ("test_mse", "lower"),
        ("recall10_01pct", "higher"),
        ("recall10_05pct", "higher"),
        ("recall10_10pct", "higher"),
        ("recall1_01pct", "higher"),
    )

    for cell in cells:
        seeds = sorted(seeds_by_cell.get(cell.cell_id, set()))
        if not seeds:
            continue
        for b_t in budgets:
            for cand in candidate_arms:
                for base in baselines:
                    if cand == base:
                        continue
                    base_t = 25.0 if base == "lloyd_25" else float(b_t)
                    for metric_name, direction in metrics:
                        c_vals = np.array([float(idx[(cell.cell_id, float(b_t), cand, s)][metric_name]) for s in seeds])
                        b_vals = np.array([float(idx[(cell.cell_id, base_t, base, s)][metric_name]) for s in seeds])
                        diffs = c_vals - b_vals
                        rel_pct = float(100.0 * (c_vals.mean() - b_vals.mean()) / max(abs(b_vals.mean()), 1e-12))
                        abs_delta = float(c_vals.mean() - b_vals.mean())

                        tol = 1e-9
                        if direction == "lower":
                            wins = int((diffs < -tol).sum())
                            losses = int((diffs > tol).sum())
                        else:
                            wins = int((diffs > tol).sum())
                            losses = int((diffs < -tol).sum())
                        ties = len(seeds) - wins - losses

                        if len(seeds) >= 2 and float(np.std(diffs)) > 1e-15:
                            p_t = float(stats.ttest_rel(c_vals, b_vals).pvalue)
                            try:
                                p_w = float(stats.wilcoxon(diffs).pvalue)
                            except ValueError:
                                p_w = 1.0
                        else:
                            p_t = 1.0
                            p_w = 1.0

                        comparisons.append(
                            {
                                "cell_id": cell.cell_id,
                                "k": int(cell.k),
                                "nepochs": float(b_t),
                                "arm": cand,
                                "baseline": base,
                                "metric": metric_name,
                                "direction": direction,
                                "arm_mean": float(c_vals.mean()),
                                "arm_std": float(c_vals.std(ddof=1)) if len(seeds) > 1 else 0.0,
                                "base_mean": float(b_vals.mean()),
                                "base_std": float(b_vals.std(ddof=1)) if len(seeds) > 1 else 0.0,
                                "abs_delta": abs_delta,
                                "rel_delta_pct": rel_pct,
                                "wins": wins,
                                "ties": ties,
                                "losses": losses,
                                "p_ttest": p_t if math.isfinite(p_t) else 1.0,
                                "p_wilcoxon": p_w if math.isfinite(p_w) else 1.0,
                            }
                        )
    return comparisons


ARM_DISPLAY: dict[str, tuple[str, str, str, str]] = {
    # arm_id: (Legend Label, Color, Marker, Linestyle)
    "lloyd_25": ("Lloyd @ 25 ep", "#333333", "*", "--"),
    "emeans": ("e-means (Meta FAIR)", "#1f77b4", "o", "-"),
    "emeans_pp": ("e-means++", "#ff7f0e", "s", "-"),
    "smeans": ("s-means (suayptalha)", "#b8860b", "^", "-."),
    "super_emeans": ("SuperEMeans", "#9467bd", "D", "-"),
    "hybrid_semeans": ("HybridSEMeans", "#2ca02c", "P", "-"),
    "hybrid_semeans_wide": ("HybridSEMeans (wide)", "#66a61e", "v", ":"),
    "ultra_emeans": ("UltraEMeans", "#e31a1c", "h", "-"),
    "ultra_emeans_wide": ("UltraEMeans (wide)", "#008b8b", "X", "--"),
}


def generate_plots(
    records: list[dict[str, Any]],
    cells: list[MillionCellSpec],
    budgets: tuple[float, ...],
    out_dir: Path,
) -> list[Path]:
    """Generate publication-grade `sift1m_mse_vs_time.png` & `sift1m_ivf_recall_pareto.png`."""
    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.2,
            "figure.dpi": 220,
        }
    )

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def _save_atomic_png(fig: plt.Figure, targets: list[Path]) -> None:
        for tpath in targets:
            tpath.parent.mkdir(parents=True, exist_ok=True)
            tmp = tpath.parent / f".tmp_{os.getpid()}_{time.time_ns()}_{tpath.name}"
            try:
                fig.savefig(tmp, dpi=220, bbox_inches="tight", format="png")
                with Image.open(tmp) as im:
                    im.verify()
                os.replace(tmp, tpath)
                written.append(tpath)
            finally:
                if tmp.exists():
                    tmp.unlink(missing_ok=True)

    # Group records by (cell_id, arm, nepochs)
    grouped: dict[tuple[str, str, float], list[dict[str, Any]]] = {}
    cell_sku_str: dict[str, str] = {}
    for r in records:
        grouped.setdefault((str(r["cell_id"]), str(r["arm"]), float(r["nepochs"])), []).append(r)
        cell_sku_str[str(r["cell_id"])] = str(r["gpu_sku"])

    # --------------------------------------------------------------------------
    # Plot 1: Million-Scale Held-Out Test Distortion vs Build Time (`sift1m_mse_vs_time.png`)
    # --------------------------------------------------------------------------
    n_panels = len(cells)
    ncols = 2
    nrows = max(1, (n_panels + 1) // 2)
    fig1, axes1 = plt.subplots(nrows, ncols, figsize=(13.6, 5.2 * nrows), squeeze=False)

    for idx_c, cell in enumerate(cells):
        ax = axes1[idx_c // ncols][idx_c % ncols]
        metric_key = "test_mse"
        ylabel = "Held-Out Test MSE (FP64, lower is better)"
        if cell.geometry == "unit_sphere":
            ylabel = "Held-Out Test Euclidean MSE (unit sphere, FP64)"

        # Lloyd@25 reference point + dashed horizontal line
        lloyd_rows = grouped.get((cell.cell_id, "lloyd_25", 25.0), [])
        if lloyd_rows:
            l_ms = np.mean([1000.0 * float(x["total_time_sec"]) for x in lloyd_rows])
            l_val = np.mean([float(x[metric_key]) for x in lloyd_rows])
            lbl, col, mk, _ = ARM_DISPLAY["lloyd_25"]
            ax.axhline(l_val, color=col, linestyle="--", linewidth=1.1, alpha=0.65)
            ax.scatter([l_ms], [l_val], color=col, marker=mk, s=135, zorder=6, label=lbl)

        for arm in STREAMING_ARMS:
            xs, ys, yerrs = [], [], []
            for b_t in budgets:
                rows = grouped.get((cell.cell_id, arm, float(b_t)), [])
                if not rows:
                    continue
                xs.append(float(np.mean([1000.0 * float(r["total_time_sec"]) for r in rows])))
                vals = [float(r[metric_key]) for r in rows]
                ys.append(float(np.mean(vals)))
                yerrs.append(float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0)
            if not xs:
                continue
            lbl, col, mk, ls = ARM_DISPLAY[arm]
            lw = 2.2 if arm.startswith("ultra") else 1.5
            zord = 5 if arm.startswith("ultra") else 3
            ax.errorbar(
                xs,
                ys,
                yerr=yerrs,
                label=lbl,
                color=col,
                marker=mk,
                linestyle=ls,
                linewidth=lw,
                markersize=6.0,
                capsize=2.2,
                zorder=zord,
            )
            if arm in ("ultra_emeans", "smeans", "emeans"):
                for b_t, xp, yp in zip(budgets, xs, ys):
                    ax.annotate(
                        f"T={int(b_t) if b_t.is_integer() else b_t}",
                        (xp, yp),
                        textcoords="offset points",
                        xytext=(3, 4),
                        fontsize=7.0,
                        color=col,
                        alpha=0.85,
                    )

        sku_short = cell_sku_str.get(cell.cell_id, "").replace("NVIDIA RTX PRO 6000 Blackwell ", "Blackwell ")
        ax.set_xscale("log")
        ax.set_title(f"{cell.cell_id}: N={cell.n_base:,}, D={cell.d}, K={cell.k:,}\n[{sku_short}]", fontweight="bold")
        ax.set_xlabel("Total GPU Index Build Time: Init + Train (ms, log scale)")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", linestyle=":", alpha=0.45)

    for extra_i in range(n_panels, nrows * ncols):
        axes1[extra_i // ncols][extra_i % ncols].axis("off")

    handles1, labels1 = axes1[0][0].get_legend_handles_labels()
    leg_y1 = 1.03 if nrows == 1 else 1.015
    sup_y1 = 1.08 if nrows == 1 else 1.045
    n_banner = f"N = {cells[0].n_base:,}" if cells else "Million-Scale"
    fig1.legend(handles1, labels1, loc="upper center", ncol=5, frameon=True, bbox_to_anchor=(0.5, leg_y1))
    fig1.suptitle(
        f"Real-World Vector Quantization ({n_banner}): Held-Out Test MSE vs. GPU Build Time",
        fontsize=12.5,
        fontweight="bold",
        y=sup_y1,
    )
    fig1.tight_layout()
    _save_atomic_png(fig1, [out_dir / "sift1m_mse_vs_time.png", plots_dir / "sift1m_mse_vs_time.png"])
    plt.close(fig1)

    # --------------------------------------------------------------------------
    # Plot 2: Appendix-K Downstream ANN IVF Recall@10 vs Build Time (`sift1m_ivf_recall_pareto.png`)
    # --------------------------------------------------------------------------
    # Top row: SIFT1M K=16,384 (or primary SIFT cell) at 0.1%, 0.5%, 1.0% selectivity
    # Bottom row: GloVe-1M-Unit K=16,384 (or secondary cell) at 0.1%, 0.5%, 1.0% selectivity
    sift_cells = [c for c in cells if "sift" in c.cell_id]
    glove_cells = [c for c in cells if "glove" in c.cell_id]
    row_cells = []
    if sift_cells:
        row_cells.append(max(sift_cells, key=lambda c: c.k))
    if glove_cells:
        row_cells.append(max(glove_cells, key=lambda c: c.k))
    if not row_cells:
        row_cells = cells[:2]

    fig2, axes2 = plt.subplots(
        len(row_cells),
        3,
        figsize=(15.5, 4.9 * len(row_cells)),
        squeeze=False,
    )
    sel_panels = (
        ("recall10_01pct", "0.1% Selectivity", "nprobe_01pct"),
        ("recall10_05pct", "0.5% Selectivity", "nprobe_05pct"),
        ("recall10_10pct", "1.0% Selectivity", "nprobe_10pct"),
    )

    for r_i, cell in enumerate(row_cells):
        sku_short = cell_sku_str.get(cell.cell_id, "").replace("NVIDIA RTX PRO 6000 Blackwell ", "Blackwell ")
        for c_i, (r_field, sel_title, np_field) in enumerate(sel_panels):
            ax = axes2[r_i][c_i]
            lloyd_rows = grouped.get((cell.cell_id, "lloyd_25", 25.0), [])
            np_val = int(lloyd_rows[0][np_field]) if lloyd_rows else max(1, round(SELECTIVITIES[c_i] * cell.k))
            if lloyd_rows:
                l_ms = np.mean([1000.0 * float(x["total_time_sec"]) for x in lloyd_rows])
                l_rec = 100.0 * np.mean([float(x[r_field]) for x in lloyd_rows])
                lbl, col, mk, _ = ARM_DISPLAY["lloyd_25"]
                ax.axhline(l_rec, color=col, linestyle="--", linewidth=1.1, alpha=0.65)
                ax.scatter([l_ms], [l_rec], color=col, marker=mk, s=135, zorder=6, label=lbl)

            for arm in STREAMING_ARMS:
                xs, ys, yerrs = [], [], []
                for b_t in budgets:
                    rows = grouped.get((cell.cell_id, arm, float(b_t)), [])
                    if not rows:
                        continue
                    xs.append(float(np.mean([1000.0 * float(r["total_time_sec"]) for r in rows])))
                    vals = [100.0 * float(r[r_field]) for r in rows]
                    ys.append(float(np.mean(vals)))
                    yerrs.append(float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0)
                if not xs:
                    continue
                lbl, col, mk, ls = ARM_DISPLAY[arm]
                lw = 2.2 if arm.startswith("ultra") else 1.5
                zord = 5 if arm.startswith("ultra") else 3
                ax.errorbar(
                    xs,
                    ys,
                    yerr=yerrs,
                    label=lbl,
                    color=col,
                    marker=mk,
                    linestyle=ls,
                    linewidth=lw,
                    markersize=5.8,
                    capsize=2.0,
                    zorder=zord,
                )

            ax.set_xscale("log")
            ax.set_title(
                f"{cell.cell_id} (K={cell.k:,}) — {sel_title} (nprobe={np_val})\n[{sku_short}]",
                fontweight="bold",
            )
            ax.set_xlabel("Index Build Time: Init + Train (ms, log scale)")
            ax.set_ylabel("Test IVF Recall@10 (%, higher is better)")
            ax.grid(True, which="both", linestyle=":", alpha=0.45)

    handles2, labels2 = axes2[0][0].get_legend_handles_labels()
    n_banner2 = f"N = {row_cells[0].n_base:,}" if row_cells else "Million-Scale"
    leg_y2 = 1.03 if len(row_cells) == 1 else 1.02
    sup_y2 = 1.08 if len(row_cells) == 1 else 1.055
    fig2.legend(handles2, labels2, loc="upper center", ncol=5, frameon=True, bbox_to_anchor=(0.5, leg_y2))
    fig2.suptitle(
        f"Downstream ANN IVF-Flat Recall@10 vs. Index Build Time (Appendix K Protocol, {n_banner2})",
        fontsize=12.5,
        fontweight="bold",
        y=sup_y2,
    )
    fig2.tight_layout()
    _save_atomic_png(fig2, [out_dir / "sift1m_ivf_recall_pareto.png", plots_dir / "sift1m_ivf_recall_pareto.png"])
    plt.close(fig2)

    return written


def write_summary_markdown(
    records: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    cells: list[MillionCellSpec],
    budgets: tuple[float, ...],
    out_path: Path,
) -> None:
    """Write statistical summary tables to `phase4_million_scale_summary.md` via atomic `os.replace`."""
    grouped: dict[tuple[str, str, float], list[dict[str, Any]]] = {}
    for r in records:
        grouped.setdefault((str(r["cell_id"]), str(r["arm"]), float(r["nepochs"])), []).append(r)

    lines: list[str] = [
        "# Phase 4 Million-Scale Real Corpus Quantization & Downstream ANN IVF Recall Summary",
        "",
        f"- **Total Verified Runs**: `{len(records):,}` across `{len(cells)}` cells (`0` `NaN`/`Inf`, `100%` single-SKU per cell).",
        "",
        "## 1. Per-Cell Million-Scale Quantization MSE, IVF Recall@{1, 10}, Imbalance & GPU Build Time",
        "",
    ]

    for cell in cells:
        sku = next((str(r["gpu_sku"]) for r in records if r["cell_id"] == cell.cell_id), "Unknown")
        lines.extend(
            [
                f"### Cell `{cell.cell_id}` (`N={cell.n_base:,}, D={cell.d}, K={cell.k:,}`, `{sku}`)",
                "",
                "| Arm | Budget `T` | Build `ms` (`init + train`) | Train MSE | Test MSE | Cos Dist | IF | R@1 (`0.1%`) | R@10 (`0.1%`) | R@10 (`0.5%`) | R@10 (`1.0%`) |",
                "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        run_order: list[tuple[str, float]] = [("lloyd_25", 25.0)]
        for b_t in budgets:
            for arm in STREAMING_ARMS:
                run_order.append((arm, float(b_t)))

        for arm, b_t in run_order:
            rows = grouped.get((cell.cell_id, arm, b_t), [])
            if not rows:
                continue
            tot_ms = np.mean([1000.0 * float(x["total_time_sec"]) for x in rows])
            ini_ms = np.mean([1000.0 * float(x["init_time_sec"]) for x in rows])
            trn_ms = np.mean([1000.0 * float(x["train_time_sec"]) for x in rows])
            tr_m = np.mean([float(x["train_mse"]) for x in rows])
            te_m = np.mean([float(x["test_mse"]) for x in rows])
            cd_m = np.mean([float(x["test_cos_dist"]) for x in rows])
            if_m = np.mean([float(x["imbalance_factor"]) for x in rows])
            r1_01 = 100.0 * np.mean([float(x["recall1_01pct"]) for x in rows])
            r10_01 = 100.0 * np.mean([float(x["recall10_01pct"]) for x in rows])
            r10_05 = 100.0 * np.mean([float(x["recall10_05pct"]) for x in rows])
            r10_10 = 100.0 * np.mean([float(x["recall10_10pct"]) for x in rows])
            cd_str = f"{cd_m:.6f}" if cell.geometry == "unit_sphere" else "—"
            lines.append(
                f"| `{arm}` | `{b_t:g}` | `{tot_ms:.1f}` (`{ini_ms:.1f}+{trn_ms:.1f}`) | "
                f"`{tr_m:.4f}` | `{te_m:.4f}` | {cd_str} | `{if_m:.3f}` | "
                f"`{r1_01:.2f}%` | `{r10_01:.2f}%` | `{r10_05:.2f}%` | `{r10_10:.2f}%` |"
            )
        lines.append("")

    lines.extend(
        [
            "## 2. Key Paired Head-to-Head Statistical Comparisons (`test_mse` & `recall10_01pct`)",
            "",
            "| Cell | `T` | Candidate | Baseline | Metric | Cand Mean | Base Mean | Rel / Abs Delta | W/T/L | `p_ttest` | `p_wilcox` |",
            "| :--- | ---: | :--- | :--- | :--- | ---: | ---: | ---: | :---: | ---: | ---: |",
        ]
    )
    for comp in comparisons:
        if comp["metric"] not in ("test_mse", "recall10_01pct", "recall10_10pct"):
            continue
        if comp["arm"] not in ("ultra_emeans", "ultra_emeans_wide", "hybrid_semeans", "super_emeans", "smeans"):
            continue
        if comp["baseline"] not in ("emeans", "smeans", "super_emeans", "lloyd_25"):
            continue
        wtl = f"{comp['wins']}/{comp['ties']}/{comp['losses']}"
        if comp["metric"] == "test_mse":
            delta_s = f"{comp['rel_delta_pct']:+.3f}%"
        else:
            delta_s = f"{100.0 * comp['abs_delta']:+.2f} pp"
        lines.append(
            f"| `{comp['cell_id']}` | `{comp['nepochs']:g}` | `{comp['arm']}` | `{comp['baseline']}` | "
            f"`{comp['metric']}` | `{comp['arm_mean']:.5f}` | `{comp['base_mean']:.5f}` | "
            f"{delta_s} | {wtl} | `{comp['p_ttest']:.3e}` | `{comp['p_wilcoxon']:.3e}` |"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.parent / f".tmp_{os.getpid()}_{time.time_ns()}_{out_path.name}"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, out_path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


# ==============================================================================
# 5-Host / 12-GPU Remote Fleet Dispatcher
# ==============================================================================


def discover_live_ssh_auth_sock() -> str | None:
    """Auto-discover live forwarded `SSH_AUTH_SOCK` from `tmux` session 0 or `/tmp/ssh-*/agent.*`."""
    candidates: list[str] = []
    cur = os.environ.get("SSH_AUTH_SOCK")
    if cur:
        candidates.append(cur)
    try:
        res = subprocess.run(
            ["tmux", "show-environment", "-t", "0", "SSH_AUTH_SOCK"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if res.returncode == 0 and "=" in res.stdout:
            candidates.insert(0, res.stdout.strip().split("=", 1)[1])
    except Exception:
        pass
    candidates.extend(sorted(glob.glob("/tmp/ssh-*/agent.*"), key=os.path.getmtime, reverse=True))

    seen: set[str] = set()
    for cand in candidates:
        if not cand or cand in seen or not os.path.exists(cand):
            continue
        seen.add(cand)
        probe_env = os.environ.copy()
        probe_env["SSH_AUTH_SOCK"] = cand
        chk = subprocess.run(["ssh-add", "-l"], env=probe_env, capture_output=True, text=True, timeout=2, check=False)
        if chk.returncode == 0:
            return cand
    return None


def dispatch_fleet_12way(
    num_seeds: int,
    seed_start: int,
    out_dir: Path,
    data_dir: Path,
) -> None:
    """Dispatch all 12 remote GPU shards in parallel across the 5 approved Blackwell hosts."""
    shards = build_fleet_shards(num_seeds)
    num_shards = len(shards)
    assert num_shards == 12

    seen_sessions: set[str] = set()
    seen_host_gpus: set[tuple[str, int]] = set()
    for sp in shards:
        if "walnut" in sp.host.lower():
            raise RuntimeError("SAFETY VIOLATION: walnut.stratabaremetal.com is strictly off-limits!")
        if not sp.session.startswith("emeans_p4_mscale_s") or sp.session == "0":
            raise RuntimeError(f"SAFETY VIOLATION: invalid tmux session {sp.session!r}")
        if sp.session in seen_sessions or (sp.host, sp.gpu_id) in seen_host_gpus:
            raise RuntimeError(f"Duplicate session or GPU assignment in {sp!r}")
        seen_sessions.add(sp.session)
        seen_host_gpus.add((sp.host, sp.gpu_id))

    # Ensure local validated million-scale `.npz` caches exist before syncing to workers
    dev_local = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_sift1m(data_dir=data_dir, device=dev_local)
    load_glove1m_unit(data_dir=data_dir, device=dev_local)

    env = os.environ.copy()
    live_sock = discover_live_ssh_auth_sock()
    if live_sock:
        env["SSH_AUTH_SOCK"] = live_sock

    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    rel_out = out_dir.relative_to(ROOT) if out_dir.is_relative_to(ROOT) else Path("outputs/phase4_million_scale")
    remote_data_dir = f"{REMOTE_WORKDIR}/outputs/phase4_million_scale/data"

    # Determine which validated `.npz` cache each remote host actually needs
    host_needed_npz: dict[str, set[str]] = {}
    for sp in shards:
        npz_name = (
            "sift1m_validated_v1.npz"
            if sp.cell_id.startswith("sift1m")
            else "glove1m_unit_s20261116_validated_v1.npz"
        )
        host_needed_npz.setdefault(sp.host, set()).add(npz_name)

    def _sync_host(host: str) -> None:
        print(f"[Fleet-Phase4] Syncing code + validated corpus cache to {host}:{REMOTE_WORKDIR} ...", flush=True)
        subprocess.run(
            [
                "ssh", "-A", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                host,
                f"mkdir -p {REMOTE_WORKDIR}/outputs/logs {remote_data_dir}",
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
            src_npz = data_dir / npz_name
            if ".stratabaremetal.com" in host and npz_name == "sift1m_validated_v1.npz":
                # On WAN hosts (`acorn`, `cashew`), populate `sift1m_validated_v1.npz` once on GPU 0 directly
                # from HuggingFace CDN (~100 MB/s) if not already cached, avoiding repeated 590MB WAN uploads!
                pop_cmd = (
                    f"mkdir -p /home/catid/emeans_env/tmp && "
                    f"export PATH=\"$HOME/.local/bin:/usr/local/cuda/bin:$PATH\"; "
                    f"export CUDA_VISIBLE_DEVICES=0; "
                    f"export TMPDIR=/home/catid/emeans_env/tmp; "
                    f"export TRITON_CACHE_DIR=/home/catid/emeans_env/tmp/triton_cache_p4_prep; "
                    f"export PYTHONPATH={REMOTE_WORKDIR}/src; "
                    f"cd {REMOTE_WORKDIR} && "
                    f"{REMOTE_PYTHON} -c 'from emeans.real_corpora import load_sift1m; "
                    f"load_sift1m(data_dir=\"{remote_data_dir}\", device=\"cuda\")'"
                )
                subprocess.run(
                    ["ssh", "-A", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, pop_cmd],
                    env=env,
                    check=True,
                )
            else:
                subprocess.run(
                    [
                        "rsync", "-az", "--ignore-existing",
                        "-e", "ssh -A -o BatchMode=yes -o ConnectTimeout=10",
                        str(src_npz),
                        f"{host}:{remote_data_dir}/{npz_name}",
                    ],
                    env=env,
                    check=True,
                )
        print(f"[Fleet-Phase4] Host {host} ready.", flush=True)

    unique_hosts = sorted(host_needed_npz.keys())
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(unique_hosts)) as pool:
        list(pool.map(_sync_host, unique_hosts))

    # Launch all 12 detached `emeans_p4_mscale_s*` sessions simultaneously
    active_shards = [sp for sp in shards if len(sp.seed_offsets) > 0]
    for sp in active_shards:
        remote_out = f"{REMOTE_WORKDIR}/{rel_out}"
        shard_json = f"{remote_out}/shards/records_shard_{sp.shard_idx:02d}_of_{num_shards:02d}.json"
        remote_cmd = (
            f"mkdir -p /home/catid/emeans_env/tmp {REMOTE_WORKDIR}/outputs/logs {remote_out}/shards && "
            f"rm -f {REMOTE_WORKDIR}/outputs/logs/{sp.session}.status {shard_json} && "
            f"tmux kill-session -t {sp.session} 2>/dev/null || true; "
            f"tmux new-session -d -s {sp.session} "
            + shlex.quote(
                f"export PATH=\"$HOME/.local/bin:/usr/local/cuda/bin:$PATH\"; "
                f"export CUDA_VISIBLE_DEVICES={sp.gpu_id}; "
                f"export TMPDIR=/home/catid/emeans_env/tmp; "
                f"export TRITON_CACHE_DIR=/home/catid/emeans_env/tmp/triton_cache_{sp.session}; "
                f"export PYTHONPATH={REMOTE_WORKDIR}/src; "
                f"cd {REMOTE_WORKDIR} && "
                f"{REMOTE_PYTHON} scripts/run_phase4_million_scale.py "
                f"--shard-idx {sp.shard_idx} --num-shards {num_shards} "
                f"--num-seeds {num_seeds} --seed-start {seed_start} "
                f"--data-dir {remote_data_dir} --out-dir {remote_out} "
                f"> {REMOTE_WORKDIR}/outputs/logs/{sp.session}.log 2>&1; "
                f"echo EXIT_CODE=$? > {REMOTE_WORKDIR}/outputs/logs/{sp.session}.status"
            )
        )
        subprocess.run(
            ["ssh", "-A", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", sp.host, remote_cmd],
            env=env,
            check=True,
        )
        print(
            f"[Fleet-Phase4] Launched shard {sp.shard_idx:02d}/{num_shards} on {sp.host} GPU {sp.gpu_id} "
            f"({sp.session}, cell={sp.cell_id}, seeds={len(sp.seed_offsets)})",
            flush=True,
        )

    # Poll until all active shards complete cleanly
    pending = {sp.shard_idx: sp for sp in active_shards}
    t_poll0 = time.perf_counter()
    while pending:
        time.sleep(5.0)
        done_ids: list[int] = []
        for sidx, sp in list(pending.items()):
            chk_cmd = f"cat {REMOTE_WORKDIR}/outputs/logs/{sp.session}.status 2>/dev/null || true"
            res = subprocess.run(
                ["ssh", "-A", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", sp.host, chk_cmd],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            txt = res.stdout.strip()
            if "EXIT_CODE=" in txt:
                code = int(txt.split("EXIT_CODE=", 1)[1].splitlines()[0].strip())
                if code != 0:
                    log_res = subprocess.run(
                        [
                            "ssh", "-A", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                            sp.host,
                            f"tail -n 60 {REMOTE_WORKDIR}/outputs/logs/{sp.session}.log 2>/dev/null || true",
                        ],
                        env=env,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    raise RuntimeError(
                        f"Remote shard {sidx:02d} ({sp.host} {sp.session}) failed with code {code}:\n{log_res.stdout}"
                    )
                remote_json = f"{REMOTE_WORKDIR}/{rel_out}/shards/records_shard_{sidx:02d}_of_{num_shards:02d}.json"
                local_json = shards_dir / f"records_shard_{sidx:02d}_of_{num_shards:02d}.json"
                subprocess.run(
                    [
                        "rsync", "-az",
                        "-e", "ssh -A -o BatchMode=yes -o ConnectTimeout=8",
                        f"{sp.host}:{remote_json}",
                        str(local_json),
                    ],
                    env=env,
                    check=True,
                )
                print(
                    f"[Fleet-Phase4] Collected shard {sidx:02d} ({sp.host} GPU {sp.gpu_id}) "
                    f"at +{time.perf_counter() - t_poll0:.1f}s",
                    flush=True,
                )
                done_ids.append(sidx)
        for did in done_ids:
            pending.pop(did, None)


# ==============================================================================
# CLI Entrypoint
# ==============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 4 Million-Scale Real Corpus & ANN IVF Recall Runner")
    p.add_argument("--smoke", action="store_true", help="Run fast 1-seed k=512 end-to-end smoke test")
    p.add_argument("--dispatch-fleet", action="store_true", help="Dispatch 12-GPU multi-machine campaign & merge")
    p.add_argument("--merge-only", action="store_true", help="Merge existing shard JSONs, validate, and plot")
    p.add_argument("--shard-idx", type=int, default=None, help="Worker shard index (0..num_shards-1)")
    p.add_argument("--num-shards", type=int, default=12, help="Total fleet shards (default: 12)")
    p.add_argument("--num-seeds", type=int, default=10, help="Number of paired seeds per cell (default: 10)")
    p.add_argument("--seed-start", type=int, default=4000, help="Starting RNG seed (default: 4000)")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_MILLION_DATA_DIR, help="Raw/validated corpus cache dir")
    p.add_argument("--out-dir", type=Path, default=OUTPUT_DIR, help="Output directory for JSON, MD, and PNGs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir: Path = args.out_dir
    data_dir: Path = args.data_dir

    if args.smoke:
        num_seeds = 2
        seed_start = int(args.seed_start)
        budgets: tuple[float, ...] = (1.0, 2.0)
        cells = build_cells(smoke=True)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        warmup_global_kernels(dev)
        seeds = [seed_start + i for i in range(num_seeds)]
        all_records: list[dict[str, Any]] = []
        for cell in cells:
            all_records.extend(execute_cell_seeds(cell, seeds, budgets, data_dir=data_dir, device=dev))
        validate_records(all_records, cells, seeds, budgets)
        comparisons = compute_paired_comparisons(all_records, cells, budgets)
        payload = {
            "phase": "phase4_million_scale_smoke",
            "num_records": len(all_records),
            "seeds": seeds,
            "budgets": list(budgets),
            "cells": [asdict(c) for c in cells],
            "records": all_records,
            "comparisons": comparisons,
        }
        atomic_write_json(out_dir / "phase4_million_scale_results.json", payload)
        write_summary_markdown(all_records, comparisons, cells, budgets, out_dir / "phase4_million_scale_summary.md")
        plots = generate_plots(all_records, cells, budgets, out_dir)
        print(f"[Phase4-Smoke] SUCCESS: {len(all_records)} records, {len(plots)} PNGs written to {out_dir}", flush=True)
        return

    cells = build_cells(smoke=False)
    cell_by_id = {c.cell_id: c for c in cells}
    num_seeds = int(args.num_seeds)
    seed_start = int(args.seed_start)
    all_seeds = [seed_start + i for i in range(num_seeds)]
    budgets = EPOCH_BUDGETS

    if args.shard_idx is not None:
        if int(args.num_shards) != 12:
            raise ValueError("Expected --num-shards 12 matching FLEET_SHARDS")
        shards = build_fleet_shards(num_seeds)
        sp = shards[int(args.shard_idx)]
        shard_seeds = [seed_start + off for off in sp.seed_offsets]
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        warmup_global_kernels(dev)
        cell = cell_by_id[sp.cell_id]
        recs = execute_cell_seeds(cell, shard_seeds, budgets, data_dir=data_dir, device=dev)
        shard_path = out_dir / "shards" / f"records_shard_{sp.shard_idx:02d}_of_{int(args.num_shards):02d}.json"
        atomic_write_json(
            shard_path,
            {
                "shard_idx": sp.shard_idx,
                "num_shards": int(args.num_shards),
                "cell_id": sp.cell_id,
                "seeds": shard_seeds,
                "records": recs,
            },
        )
        print(f"[Phase4-Shard {sp.shard_idx:02d}] Wrote {len(recs)} records -> {shard_path}", flush=True)
        return

    if args.dispatch_fleet:
        dispatch_fleet_12way(num_seeds=num_seeds, seed_start=seed_start, out_dir=out_dir, data_dir=data_dir)

    if args.dispatch_fleet or args.merge_only:
        shards_dir = out_dir / "shards"
        merged_records: list[dict[str, Any]] = []
        shards = build_fleet_shards(num_seeds)
        for sp in shards:
            if not sp.seed_offsets:
                continue
            spath = shards_dir / f"records_shard_{sp.shard_idx:02d}_of_12.json"
            if not spath.exists():
                raise FileNotFoundError(f"Missing shard file {spath}")
            with open(spath, "r", encoding="utf-8") as f:
                data = json.load(f, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"Invalid JSON constant {c}")))
            merged_records.extend(data["records"])

        validate_records(merged_records, cells, all_seeds, budgets)
        comparisons = compute_paired_comparisons(merged_records, cells, budgets)
        payload = {
            "phase": "phase4_million_scale",
            "num_records": len(merged_records),
            "seeds": all_seeds,
            "budgets": list(budgets),
            "selectivities": list(SELECTIVITIES),
            "cells": [asdict(c) for c in cells],
            "records": merged_records,
            "comparisons": comparisons,
        }
        atomic_write_json(out_dir / "phase4_million_scale_results.json", payload)
        write_summary_markdown(
            merged_records,
            comparisons,
            cells,
            budgets,
            out_dir / "phase4_million_scale_summary.md",
        )
        plots = generate_plots(merged_records, cells, budgets, out_dir)
        print(
            f"[Phase4-Complete] Validated {len(merged_records):,} million-scale records & {len(plots)} PNGs in {out_dir}",
            flush=True,
        )
        return

    # Local single-GPU sequential fallback when neither --smoke nor --dispatch-fleet is set
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    warmup_global_kernels(dev)
    local_records: list[dict[str, Any]] = []
    for cell in cells:
        local_records.extend(execute_cell_seeds(cell, all_seeds, budgets, data_dir=data_dir, device=dev))
    validate_records(local_records, cells, all_seeds, budgets)
    comparisons = compute_paired_comparisons(local_records, cells, budgets)
    atomic_write_json(
        out_dir / "phase4_million_scale_results.json",
        {
            "phase": "phase4_million_scale_local",
            "num_records": len(local_records),
            "seeds": all_seeds,
            "budgets": list(budgets),
            "cells": [asdict(c) for c in cells],
            "records": local_records,
            "comparisons": comparisons,
        },
    )
    write_summary_markdown(local_records, comparisons, cells, budgets, out_dir / "phase4_million_scale_summary.md")
    generate_plots(local_records, cells, budgets, out_dir)


if __name__ == "__main__":
    main()
