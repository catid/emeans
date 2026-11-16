#!/usr/bin/env python3
"""Phase 5 Multi-Domain Realistic Benchmark Dominance & Optimization Ladder Runner.

Evaluates `ApexEMeans` (`BF16 + CUDAGraph`, `CroissAttn-FP8 + CUDAGraph`, and unified
4-Gear Scale-Adaptive `Auto`) head-to-head against ALL baseline lineages across **7 genuine
real-world benchmark corpora** (`T in {1.0, 2.0, 3.0, 5.0}`, `10` paired seeds) AND isolates
each systems optimization layer on a **6-step Optimization Speedup Ladder**:

Suite 1 — 7-Modality Realistic Benchmark Dominance (`2,870` paired runs):
  1. `sift1m_k4096`            (`1,000,000 x 128`, SIFT visual descriptors, `server_600w`)
  2. `glove1m_k4096`           (`1,000,000 x 100`, GloVe unit-sphere NLP vectors, `maxq_local`)
  3. `covtype_k2048`           (`  500,000 x  54`, Raw Forest Covertype physical sensors, `server_600w`)
  4. `cifar10_patches_k2048`   (` ~200,000 x 128`, ResNet-18 spatial patch features, `maxq_local`)
  5. `agnews_svd256_k1024`     (` ~109,738 x 256`, AG-News train-only TF-IDF+SVD skewed topics, `maxq_local`)
  6. `cifar10_r18_512_k1024`   (`  ~50,000 x 512`, ResNet-18 ReLU global embeddings, `maxq_local`)
  7. `cifar10_convnext_768_k1024` (`~50,000 x 768`, ConvNeXt-Tiny LayerNorm 768D vision, `maxq_local`)

  Compared Solvers (`1` Lloyd anchor + `10` streaming solvers x `4` epoch budgets = `41` runs/seed):
    - `lloyd_25`         (`LloydKMeans`, 25 iterations)
    - `emeans`           (Meta FAIR `arXiv:2611.09501`)
    - `smeans_ref`       (`suayptalha/s-means` exact upstream reference `exact_ref=True`)
    - `smeans_fast`      (`suayptalha/s-means` zero-sync centered Triton path `exact_ref=False`)
    - `emeans_pp`        (`EMeansPP`)
    - `super_emeans`     (`SuperEMeans`)
    - `hybrid_semeans`   (`HybridSEMeans` cross-pollinated champion)
    - `ultra_emeans`     (`UltraEMeans`)
    - `apex_emeans_bf16` (`ApexEMeans (BF16 + CUDAGraph)`)
    - `apex_emeans_fp8`  (`ApexEMeans (CroissAttn-FP8 + CUDAGraph)`)
    - `apex_emeans`      (`ApexEMeans (Auto 4-Gear Scale-Adaptive)`)

Suite 2 — Stepwise Optimization Speedup Ladder (`240` paired runs on `server_600w`):
  - Shapes: `(250,000 x 128, K = 4,096)` and `(1,000,000 x 128, K = 16,384)` real SIFT1M vectors
  - Cumulative Steps (`T in {2.0, 5.0}`, `10` paired seeds):
      `ladder_0_baseline`
      -> `ladder_1_plus_batch`     (`+Batch: Two-Gear / Wide Schedule`)
      -> `ladder_2_plus_fuse`      (`+Fuse: Scatter + EMA + Oja + Revival`)
      -> `ladder_3_plus_kernel`    (`+Kernel: SM120 3-Stage TMA / Split-K`)
      -> `ladder_4_plus_cudagraph` (`+CUDAGraph: On-Device Step / Schedule / Revival`)
      -> `ladder_5_plus_croiss_fp8`(`+Quant: CroissAttn Decoupled FP8`)

Strict Benchmarking & De-Confounding Hygiene (`AGENTS.md` Rules 1, 2, 3):
  - Zero train/val/test leakage (`validate_real_corpus` enforced on every cell).
  - Pre-warmed Triton & CUDAGraph specializations (`#111`) outside all timers + automated
    Seed 0 vs median(Seed 1..N) cold-JIT spike detector.
  - Per-shard isolated `TRITON_CACHE_DIR`, `100%` single-SKU pinning per cell across `T in {1,2,3,5}`,
    `100%` same-GPU cyclic arm counterbalancing per `(cell, seed)`, `walnut` & `tmux` session `0`
    hard-blocked, and PID+ns temp + `os.replace` atomic writes (`allow_nan=False`).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import glob
import json
import math
import os
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Isolate TMPDIR & Triton disk cache on `/home` BEFORE importing torch/triton/emeans (`AGENTS.md` Rule 4)
_DEFAULT_TMP_ROOT = (
    Path("/home/catid/emeans_env/tmp")
    if Path("/home/catid/emeans_env").exists()
    else Path.home() / ".cache" / "emeans_tmp"
)
_DEFAULT_TMP_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TMPDIR", str(_DEFAULT_TMP_ROOT))
if "TRITON_CACHE_DIR" not in os.environ:
    _vis_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "all").replace(",", "_")
    os.environ["TRITON_CACHE_DIR"] = str(_DEFAULT_TMP_ROOT / f"triton_cache_p5_dom_u{os.getuid()}_g{_vis_gpu}")

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

from emeans.apex_emeans import APEX_LADDER_ARMS, build_apex_arm, warmup_apex_kernels
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
    validate_real_corpus,
)
from emeans.realistic_benchmarks import (
    BENCHMARK_SPEC_BY_NAME,
    DEFAULT_PHASE5_DATA_DIR,
    SEVEN_REALISTIC_BENCHMARKS,
    compute_corpus_dynamic_range_stats,
    evaluate_fp8_assignment_stability_probe,
    load_realistic_benchmark,
    stage_all_seven_benchmarks,
)
from emeans.seeding import HAS_TRITON as SEED_HAS_TRITON
from emeans.smeans import SMeans
from emeans.super_emeans import SuperEMeans, build_hybrid_arm
from emeans.ultra_emeans import build_ultra_arm

if SEED_HAS_TRITON:
    from emeans.seeding import _local_greedy_kpp_kernel

try:
    from emeans.croiss_fp8 import warmup_croiss_fp8_kernels
except Exception:
    warmup_croiss_fp8_kernels = None


OUTPUT_DIR = ROOT / "outputs" / "phase5_dominance"
REMOTE_WORKDIR = "/home/catid/emeans_runs/apex-bench"
REMOTE_PYTHON = "/home/catid/emeans_env/.venv/bin/python"
REMOTE_P4_MILLION_DATA = "/home/catid/emeans_runs/real-scale/outputs/phase4_million_scale/data"

EPOCH_BUDGETS: tuple[float, ...] = (1.0, 2.0, 3.0, 5.0)
LADDER_BUDGETS: tuple[float, ...] = (2.0, 5.0)
SELECTIVITIES: tuple[float, ...] = (0.001, 0.005, 0.010)

DOMINANCE_STREAMING_ARMS: tuple[str, ...] = (
    "emeans",
    "smeans_ref",
    "smeans_fast",
    "emeans_pp",
    "super_emeans",
    "hybrid_semeans",
    "ultra_emeans",
    "apex_emeans_bf16",
    "apex_emeans_fp8",
    "apex_emeans",
)
DOMINANCE_ALL_ARMS: tuple[str, ...] = ("lloyd_25",) + DOMINANCE_STREAMING_ARMS

LADDER_ORDERED_ARMS: tuple[str, ...] = (
    "ladder_0_baseline",
    "ladder_1_plus_batch",
    "ladder_2_plus_fuse",
    "ladder_3_plus_kernel",
    "ladder_4_plus_cudagraph",
    "ladder_5_plus_croiss_fp8",
)

LADDER_STEP_LABELS: dict[str, str] = {
    "ladder_0_baseline": "0. Baseline\n(Gather+Unfused)",
    "ladder_1_plus_batch": "1. +Batch\n(2-Gear Wide)",
    "ladder_2_plus_fuse": "2. +Fuse\n(Scatter+EMA+Oja)",
    "ladder_3_plus_kernel": "3. +Kernel\n(SM120 TMA/SplitK)",
    "ladder_4_plus_cudagraph": "4. +CUDAGraph\n(5-Step Graph)",
    "ladder_5_plus_croiss_fp8": "5. +CroissFP8\n(Decoupled E4M3)",
}

ARM_DISPLAY: dict[str, tuple[str, str, str, str]] = {
    # arm_id: (Legend Label, Color, Marker, Linestyle)
    "lloyd_25": ("Lloyd @ 25 ep", "#333333", "*", "--"),
    "emeans": ("e-means (Meta FAIR)", "#1f77b4", "o", "-"),
    "smeans_ref": ("s-means (exact_ref)", "#c49c94", "v", ":"),
    "smeans_fast": ("s-means (Triton)", "#b8860b", "^", "-."),
    "emeans_pp": ("e-means++", "#ff7f0e", "s", "-"),
    "super_emeans": ("SuperEMeans", "#9467bd", "D", "-"),
    "hybrid_semeans": ("HybridSEMeans", "#2ca02c", "P", "--"),
    "ultra_emeans": ("UltraEMeans", "#17becf", "h", "-"),
    "apex_emeans_bf16": ("ApexEMeans (BF16+Graph)", "#e377c2", "X", "--"),
    "apex_emeans_fp8": ("ApexEMeans (CroissFP8+Graph)", "#d62728", "p", "-"),
    "apex_emeans": ("ApexEMeans (Auto Flagship)", "#8b0000", "D", "-"),
}


# ==============================================================================
# Cell & 6-Computer (13-GPU) Single-SKU Fleet Shard Specifications
# ==============================================================================


@dataclass(frozen=True)
class DominanceCellSpec:
    """One realistic benchmark `(corpus_name, d, k)` cell pinned to a single Blackwell SKU."""

    cell_id: str
    corpus_name: str
    short_title: str
    modality: str
    n_base_limit: int | None  # None -> full corpus.base; int -> prefix for smoke/sub-scale
    n_val_limit: int | None
    n_query_limit: int | None
    d: int
    k: int
    geometry: str
    sku_tier: str


@dataclass(frozen=True)
class LadderCellSpec:
    """One stepwise optimization speedup ladder cell on SIFT1M (`250K, K=4096` or `1M, K=16384`)."""

    cell_id: str
    short_title: str
    n_base: int
    n_query: int
    d: int
    k: int
    sku_tier: str


@dataclass(frozen=True)
class FleetShardAssignment:
    """Single-GPU shard assignment across the 6 approved Blackwell hosts (`13` GPUs total)."""

    shard_idx: int
    host: str           # "local" or remote hostname (NEVER walnut!)
    gpu_id: int
    session: str        # tmux session name (NEVER "0"!)
    sku_tier: str
    dominance_tasks: tuple[tuple[str, tuple[int, ...]], ...]  # ((cell_id, seed_offsets), ...)
    ladder_tasks: tuple[tuple[str, tuple[int, ...]], ...]     # ((ladder_cell_id, seed_offsets), ...)


def build_dominance_cells(smoke: bool = False) -> list[DominanceCellSpec]:
    """Construct all 7 realistic benchmark cells (or fast `k=512` real-data slices for `--smoke`)."""
    if smoke:
        return [
            DominanceCellSpec(
                cell_id=f"smoke_{spec.name}",
                corpus_name=spec.name,
                short_title=f"{spec.name} Smoke (4K×{spec.d}, K=512)",
                modality=spec.modality,
                n_base_limit=4096,
                n_val_limit=512,
                n_query_limit=256,
                d=spec.d,
                k=512,  # k=512 > 256 strictly exercises coreset tournament k-means++ (#111)
                geometry=spec.geometry,
                sku_tier="smoke_local",
            )
            for spec in SEVEN_REALISTIC_BENCHMARKS
        ]

    return [
        DominanceCellSpec(
            "sift1m_k4096", "sift1m", "SIFT1M (1M×128, K=4096)",
            "Local Visual Descriptors", None, None, None, 128, 4096, "euclidean", "server_600w",
        ),
        DominanceCellSpec(
            "glove1m_k4096", "glove1m_unit", "GloVe-1M (1M×100, K=4096, S^99)",
            "Unit-Sphere Word Embeddings", None, None, None, 100, 4096, "unit_sphere", "maxq_local",
        ),
        DominanceCellSpec(
            "covtype_k2048", "covtype_multiscale_54", "Forest Covertype (500K×54, K=2048)",
            "Multi-Scale Physical Sensors", None, None, None, 54, 2048, "euclidean", "server_600w",
        ),
        DominanceCellSpec(
            "cifar10_patches_k2048", "cifar10_patches_128", "CIFAR-10 Patches (200K×128, K=2048)",
            "Mid-Level CNN Patch Features", None, None, None, 128, 2048, "euclidean", "maxq_local",
        ),
        DominanceCellSpec(
            "agnews_svd256_k1024", "agnews_tfidf_svd_256", "AG-News TF-IDF+SVD (110K×256, K=1024)",
            "Heavy-Tailed Text Embeddings", None, None, None, 256, 1024, "euclidean", "maxq_local",
        ),
        DominanceCellSpec(
            "cifar10_r18_512_k1024", "cifar10_resnet18_512", "CIFAR-10 ResNet-18 (50K×512, K=1024)",
            "Global ReLU CNN Embeddings", None, None, None, 512, 1024, "euclidean", "maxq_local",
        ),
        DominanceCellSpec(
            "cifar10_convnext_768_k1024", "cifar10_convnext_768", "CIFAR-10 ConvNeXt (50K×768, K=1024)",
            "Modern High-D Vision (768D)", None, None, None, 768, 1024, "euclidean", "maxq_local",
        ),
    ]


def build_ladder_cells(smoke: bool = False) -> list[LadderCellSpec]:
    """Construct the 2 optimization speedup ladder cells (`250K, K=4096` and `1M, K=16384`)."""
    if smoke:
        return [
            LadderCellSpec("smoke_ladder_250k_k512", "SIFT Smoke (8K×128, K=512)", 8192, 256, 128, 512, "smoke_local"),
            LadderCellSpec("smoke_ladder_1m_k1024", "SIFT Smoke (16K×128, K=1024)", 16384, 256, 128, 1024, "smoke_local"),
        ]
    return [
        LadderCellSpec("ladder_sift250k_k4096", "SIFT-250K (250K×128, K=4,096)", 250_000, 10_000, 128, 4096, "server_600w"),
        LadderCellSpec("ladder_sift1m_k16384", "SIFT-1M (1M×128, K=16,384)", 1_000_000, 10_000, 128, 16384, "server_600w"),
    ]


def parse_local_gpus(spec_str: str) -> tuple[int, int, int, int]:
    """Parse `--local-gpus` into 4 local GPU slots on `foureyes` (`a,b,c,d` or 2-GPU wave pair `a,b`)."""
    parts = [int(p.strip()) for p in spec_str.split(",") if p.strip()]
    if len(parts) == 4:
        if len(set(parts)) != 4 or min(parts) < 0:
            raise ValueError(f"--local-gpus with 4 entries must name 4 distinct GPU IDs (e.g. '0,1,2,3'), got {spec_str!r}")
        return parts[0], parts[1], parts[2], parts[3]
    if len(parts) == 2:
        if parts[0] == parts[1] or min(parts) < 0:
            raise ValueError(f"--local-gpus with 2 entries must name 2 distinct GPU IDs (e.g. '1,3'), got {spec_str!r}")
        return parts[0], parts[1], parts[0], parts[1]
    raise ValueError(
        f"--local-gpus must name 4 ('0,1,2,3') or 2 ('1,3' with --shard-filter '4,5' then '6,7') distinct GPU indices, got {spec_str!r}"
    )


def build_fleet_shards(
    num_seeds: int,
    local_gpus: tuple[int, int, int, int] = (0, 1, 2, 3),
) -> list[FleetShardAssignment]:
    """Partition all 7 dominance cells + 2 ladder cells across 8 dedicated GPUs on 3 clean Blackwell hosts.

    Excludes `ripper` & `loki` (external `blender` `#196`/`#212`), `work` (external `openworld_env` `#215`/`#221`),
    and strictly blocks `walnut` & `tmux 0`.
    Strict De-Confounding Guarantees:
      - Every cell executes 100% on a SINGLE Blackwell TDP SKU tier (`0` cross-SKU cells).
      - Every `(cell_id, seed)` executes all arms and all `T in {1, 2, 3, 5}` on ONE physical GPU.
    """
    def _split(n_tot: int, ways: int) -> list[tuple[int, ...]]:
        buckets: list[list[int]] = [[] for _ in range(ways)]
        for i in range(n_tot):
            buckets[i % ways].append(i)
        return [tuple(b) for b in buckets]

    all_off = tuple(range(num_seeds))
    w4 = _split(num_seeds, 4)
    w2 = _split(num_seeds, 2)
    lg0, lg1, lg2, lg3 = (int(local_gpus[0]), int(local_gpus[1]), int(local_gpus[2]), int(local_gpus[3]))
    if lg0 == lg1 or lg2 == lg3:
        raise ValueError(f"local_gpus pairs must be distinct within each wave, got {local_gpus!r}")

    return [
        # 600W Server Blackwell (`acorn:0,1` + `cashew:0,1`): SIFT1M K=4096 + Covertype K=2048 + Both SIFT Ladder Cells
        FleetShardAssignment(
            0, "acorn.stratabaremetal.com", 0, "emeans_p5_dom_s00", "server_600w",
            (("sift1m_k4096", w4[0]), ("covtype_k2048", w4[0])),
            (("ladder_sift250k_k4096", w4[0]), ("ladder_sift1m_k16384", w4[0])),
        ),
        FleetShardAssignment(
            1, "acorn.stratabaremetal.com", 1, "emeans_p5_dom_s01", "server_600w",
            (("sift1m_k4096", w4[1]), ("covtype_k2048", w4[1])),
            (("ladder_sift250k_k4096", w4[1]), ("ladder_sift1m_k16384", w4[1])),
        ),
        FleetShardAssignment(
            2, "cashew.stratabaremetal.com", 0, "emeans_p5_dom_s02", "server_600w",
            (("sift1m_k4096", w4[2]), ("covtype_k2048", w4[2])),
            (("ladder_sift250k_k4096", w4[2]), ("ladder_sift1m_k16384", w4[2])),
        ),
        FleetShardAssignment(
            3, "cashew.stratabaremetal.com", 1, "emeans_p5_dom_s03", "server_600w",
            (("sift1m_k4096", w4[3]), ("covtype_k2048", w4[3])),
            (("ladder_sift250k_k4096", w4[3]), ("ladder_sift1m_k16384", w4[3])),
        ),
        # Local 325W Max-Q Blackwell (`foureyes` GPUs lg0..lg3): GloVe-1M + CIFAR-10 Patches + AG-News + ResNet18 + ConvNeXt
        FleetShardAssignment(
            4, "local", lg0, "emeans_p5_dom_s04", "maxq_local",
            (("glove1m_k4096", w2[0]),), (),
        ),
        FleetShardAssignment(
            5, "local", lg1, "emeans_p5_dom_s05", "maxq_local",
            (("glove1m_k4096", w2[1]),), (),
        ),
        FleetShardAssignment(
            6, "local", lg2, "emeans_p5_dom_s06", "maxq_local",
            (("cifar10_patches_k2048", all_off), ("agnews_svd256_k1024", all_off)), (),
        ),
        FleetShardAssignment(
            7, "local", lg3, "emeans_p5_dom_s07", "maxq_local",
            (("cifar10_r18_512_k1024", all_off), ("cifar10_convnext_768_k1024", all_off)), (),
        ),
    ]


# ==============================================================================
# Solver Instantiation & Comprehensive Untimed Warmup (Zero Cold-JIT Spikes)
# ==============================================================================


def instantiate_dominance_arm(arm: str, d: int, k: int, nepochs: float, seed: int) -> Any:
    """Instantiate one dominance showdown solver with uniform `(d, k, nepochs, seed)` contract."""
    if arm == "lloyd_25":
        return LloydKMeans(d=d, k=k, niter=25, init="random", seed=seed)
    if arm == "emeans":
        return EMeans(d=d, k=k, nepochs=nepochs, seed=seed)
    if arm == "smeans_ref":
        return SMeans(d=d, k=k, nepochs=nepochs, exact_ref=True, seed=seed)
    if arm == "smeans_fast":
        return SMeans(d=d, k=k, nepochs=nepochs, exact_ref=False, seed=seed)
    if arm == "emeans_pp":
        return EMeansPP(d=d, k=k, nepochs=nepochs, seed=seed)
    if arm == "super_emeans":
        return SuperEMeans(d=d, k=k, nepochs=nepochs, wide_step_mult=1, seed=seed)
    if arm == "hybrid_semeans":
        return build_hybrid_arm("hybrid_super_smeans", d=d, k=k, nepochs=nepochs, seed=seed)
    if arm == "ultra_emeans":
        return build_ultra_arm("ultra_emeans", d=d, k=k, nepochs=nepochs, seed=seed)
    if arm in ("apex_emeans", "apex_emeans_bf16", "apex_emeans_fp8"):
        return build_apex_arm(arm, d=d, k=k, nepochs=nepochs, seed=seed)
    raise ValueError(f"Unknown dominance arm {arm!r}")


def instantiate_ladder_arm(arm: str, d: int, k: int, nepochs: float, seed: int) -> Any:
    """Instantiate one stepwise optimization ladder arm from `APEX_LADDER_ARMS`."""
    if arm not in APEX_LADDER_ARMS:
        raise ValueError(f"Unknown ladder arm {arm!r}; expected one of {list(APEX_LADDER_ARMS)}")
    return build_apex_arm(arm, d=d, k=k, nepochs=nepochs, seed=seed)


@torch.no_grad()
def warmup_global_kernels(device: torch.device) -> None:
    """Pre-warm global seed-dependent coreset alignment buckets (`#111`) + FP8/Apex helpers."""
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


@torch.no_grad()
def warmup_dominance_cell(
    x_base_gpu: torch.Tensor,
    x_val_gpu: torch.Tensor,
    x_query_gpu: torch.Tensor,
    corpus: RealVectorCorpus,
    cell: DominanceCellSpec,
    budgets: tuple[float, ...],
    device: torch.device,
) -> None:
    """Compile EVERY Triton/CUDAGraph specialization for `cell` outside all timers."""
    if device.type != "cuda":
        return

    # 1. Coreset `_fused_assign_kernel` `% 16 == 0` and `% 16 != 0` alignment buckets (#111)
    m_sub = min(int(x_base_gpu.shape[0]), max(8192, 16 * cell.k, cell.k * 2))
    x_m = x_base_gpu[:m_sub].float().contiguous()
    for dt in (torch.float32, torch.bfloat16):
        for k_cand in (1, 16, 17, 64, 65, 128, 129):
            if k_cand <= m_sub:
                assign_points(x_m, x_m[:k_cand], compute_dtype=dt)

    # 2. Apex dedicated kernel warmer for `(cell.d, cell.k)`
    warmup_apex_kernels(device, d=cell.d, k=cell.k)

    # 3. Full-shape untimed `.fit(...)` across Lloyd + EVERY `(arm, budget)` pair!
    lloyd_w = LloydKMeans(d=cell.d, k=cell.k, niter=2, init="random", seed=999_991)
    c_warm, _ = lloyd_w.fit(x_base_gpu, capture_assignments=False)

    for b_t in budgets:
        for arm in DOMINANCE_STREAMING_ARMS:
            solver = instantiate_dominance_arm(arm, d=cell.d, k=cell.k, nepochs=b_t, seed=999_992)
            c_warm, _ = solver.fit(x_base_gpu, capture_assignments=False)

    # 4. Pre-warm FP64 MSE, cosine distortion, CSR IVF Recall, and FP8 stability probe
    _, db_lbl = fast_exact_assign_fp64(x_base_gpu, c_warm, device=device)
    _, _ = fast_exact_assign_fp64(x_val_gpu, c_warm, device=device)
    _, _ = fast_exact_assign_fp64(x_query_gpu, c_warm, device=device)
    if cell.geometry == "unit_sphere":
        _ = exact_cosine_distortion_fp64(x_query_gpu, c_warm, device=device)

    q_sub = min(256, int(x_query_gpu.shape[0]))
    _ = evaluate_ivf_recall_gpu(
        x_base=x_base_gpu,
        centroids=c_warm,
        x_query=x_query_gpu[:q_sub],
        groundtruth_ids=corpus.groundtruth_ids[:q_sub],
        groundtruth_sq_dists=corpus.groundtruth_sq_dists[:q_sub],
        selectivities=SELECTIVITIES,
        db_labels=db_lbl,
        device=device,
    )
    origin_warm = x_base_gpu[: min(1024, int(x_base_gpu.shape[0]))].double().mean(dim=0, keepdim=True).float()
    _ = evaluate_fp8_assignment_stability_probe(
        x_query_gpu[:q_sub], c_warm, origin=origin_warm, device=device
    )
    torch.cuda.synchronize(device)


@torch.no_grad()
def warmup_ladder_cell(
    x_base_gpu: torch.Tensor,
    x_query_gpu: torch.Tensor,
    gt_ids: np.ndarray,
    gt_d2: np.ndarray,
    cell: LadderCellSpec,
    budgets: tuple[float, ...],
    device: torch.device,
) -> None:
    """Compile EVERY optimization ladder step across `budgets` outside all timers."""
    if device.type != "cuda":
        return
    m_sub = min(int(x_base_gpu.shape[0]), max(8192, 16 * cell.k, cell.k * 2))
    x_m = x_base_gpu[:m_sub].float().contiguous()
    for dt in (torch.float32, torch.bfloat16):
        for k_cand in (1, 16, 17, 64, 65, 128, 129):
            if k_cand <= m_sub:
                assign_points(x_m, x_m[:k_cand], compute_dtype=dt)

    warmup_apex_kernels(device, d=cell.d, k=cell.k)
    c_warm = x_base_gpu[: cell.k].clone()
    for b_t in budgets:
        for arm in LADDER_ORDERED_ARMS:
            solver = instantiate_ladder_arm(arm, d=cell.d, k=cell.k, nepochs=b_t, seed=999_993)
            c_warm, _ = solver.fit(x_base_gpu, capture_assignments=False)

    _, db_lbl = fast_exact_assign_fp64(x_base_gpu, c_warm, device=device)
    _, _ = fast_exact_assign_fp64(x_query_gpu, c_warm, device=device)
    q_sub = min(256, int(x_query_gpu.shape[0]))
    _ = evaluate_ivf_recall_gpu(
        x_base=x_base_gpu,
        centroids=c_warm,
        x_query=x_query_gpu[:q_sub],
        groundtruth_ids=gt_ids[:q_sub],
        groundtruth_sq_dists=gt_d2[:q_sub],
        selectivities=SELECTIVITIES,
        db_labels=db_lbl,
        device=device,
    )
    torch.cuda.synchronize(device)


# ==============================================================================
# Corpus Loading & Execution of Suite 1 (Dominance) and Suite 2 (Ladder)
# ==============================================================================


def load_dominance_cell_corpus(
    cell: DominanceCellSpec,
    data_dir: Path,
    device: torch.device,
) -> RealVectorCorpus:
    """Load validated `RealVectorCorpus` for `cell` (slicing cleanly with exact GT when `smoke`)."""
    full = load_realistic_benchmark(cell.corpus_name, data_dir=data_dir, device=device)
    if cell.n_base_limit is None and cell.n_val_limit is None and cell.n_query_limit is None:
        return full

    nb = min(int(cell.n_base_limit or full.base.shape[0]), full.base.shape[0])
    nv = min(int(cell.n_val_limit or full.val.shape[0]), full.val.shape[0])
    nq = min(int(cell.n_query_limit or full.query.shape[0]), full.query.shape[0])
    xb = np.ascontiguousarray(full.base[:nb], dtype=np.float32)
    xv = np.ascontiguousarray(full.val[:nv], dtype=np.float32)
    xq = np.ascontiguousarray(full.query[:nq], dtype=np.float32)
    gt_ids, gt_d2 = compute_exact_groundtruth_gpu(xb, xq, top_k=min(100, nb), device=device)
    sub_corp = RealVectorCorpus(
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
    validate_real_corpus(sub_corp, check_cross_split_disjoint=True)
    return sub_corp


def detect_gpu_sku_info(device: torch.device) -> tuple[str, float]:
    """Identify `(gpu_name (powerW), power_w)` so cross-SKU confounding is strictly prevented."""
    if device.type != "cuda":
        return "CPU", 0.0
    name = torch.cuda.get_device_name(device)
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
    power_w = 0.0
    try:
        cmd = [
            "nvidia-smi",
            "-i",
            vis if vis.isdigit() else "0",
            "--query-gpu=power.max_limit",
            "--format=csv,noheader,nounits",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=3, check=False)
        if res.returncode == 0:
            power_w = float(res.stdout.strip().splitlines()[0].strip())
    except Exception:
        power_w = 0.0
    sku_label = f"{name} ({int(round(power_w))}W)" if power_w > 0 else name
    return sku_label, power_w


@torch.no_grad()
def execute_dominance_cell_seeds(
    cell: DominanceCellSpec,
    seeds: list[int],
    budgets: tuple[float, ...],
    data_dir: Path,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Execute Suite 1 runs for `cell` across `seeds` with full warmup & cyclic counterbalancing."""
    if not seeds:
        return [], []

    host = socket.gethostname()
    vis_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    gpu_sku, power_w = detect_gpu_sku_info(device)

    corpus = load_dominance_cell_corpus(cell, data_dir=data_dir, device=device)
    dyn_stats = compute_corpus_dynamic_range_stats(corpus)
    x_base_gpu = torch.from_numpy(corpus.base).to(device=device, dtype=torch.float32).contiguous()
    x_val_gpu = torch.from_numpy(corpus.val).to(device=device, dtype=torch.float32).contiguous()
    x_query_gpu = torch.from_numpy(corpus.query).to(device=device, dtype=torch.float32).contiguous()
    origin_gpu = x_base_gpu.double().mean(dim=0, keepdim=True).float()

    print(
        f"[Phase5-Dom] Warming {cell.cell_id} (N={corpus.base.shape[0]:,}, D={cell.d}, K={cell.k:,}) "
        f"on {host} GPU={vis_gpu} [{gpu_sku}] ...",
        flush=True,
    )
    t_w0 = time.perf_counter()
    warmup_dominance_cell(
        x_base_gpu=x_base_gpu,
        x_val_gpu=x_val_gpu,
        x_query_gpu=x_query_gpu,
        corpus=corpus,
        cell=cell,
        budgets=budgets,
        device=device,
    )
    print(f"[Phase5-Dom] Warmup done for {cell.cell_id} in {time.perf_counter() - t_w0:.2f}s", flush=True)

    base_specs: list[tuple[str, float]] = [("lloyd_25", 25.0)]
    for b_t in budgets:
        for arm in DOMINANCE_STREAMING_ARMS:
            base_specs.append((arm, float(b_t)))

    records: list[dict[str, Any]] = []
    fp8_probes: list[dict[str, Any]] = []
    probe_budget = 2.0 if 2.0 in budgets else budgets[-1]

    for s_idx, seed in enumerate(seeds):
        t_s0 = time.perf_counter()
        rot = int(seed) % len(base_specs)
        ordered_specs = base_specs[rot:] + base_specs[:rot]

        seed_map: dict[tuple[str, float], dict[str, Any]] = {}
        bf16_probe_centroids: torch.Tensor | None = None

        for arm_name, nepochs in ordered_specs:
            solver = instantiate_dominance_arm(arm_name, d=cell.d, k=cell.k, nepochs=nepochs, seed=seed)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            centroids, _ = solver.fit(x_base_gpu, capture_assignments=False)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

            init_s = float(solver.init_time_sec)
            train_s = float(solver.train_time_sec)
            preenc_s = float(getattr(solver, "preencode_time_sec", 0.0))
            total_s = float(init_s + train_s)

            if arm_name == "apex_emeans_bf16" and abs(nepochs - probe_budget) < 1e-9:
                bf16_probe_centroids = centroids.detach().clone()

            train_mse, db_labels = fast_exact_assign_fp64(x_base_gpu, centroids, device=device)
            val_mse, _ = fast_exact_assign_fp64(x_val_gpu, centroids, device=device)
            test_mse, _ = fast_exact_assign_fp64(x_query_gpu, centroids, device=device)

            if cell.geometry == "unit_sphere":
                val_cos = exact_cosine_distortion_fp64(x_val_gpu, centroids, device=device)
                test_cos = exact_cosine_distortion_fp64(x_query_gpu, centroids, device=device)
            else:
                val_cos = 0.0
                test_cos = 0.0

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
                "suite": "dominance",
                "cell_id": cell.cell_id,
                "corpus": cell.corpus_name,
                "short_title": cell.short_title,
                "modality": cell.modality,
                "geometry": cell.geometry,
                "sku_tier": cell.sku_tier,
                "n_base": int(corpus.base.shape[0]),
                "n_val": int(corpus.val.shape[0]),
                "n_query": int(corpus.query.shape[0]),
                "d": int(cell.d),
                "k": int(cell.k),
                "seed": int(seed),
                "arm": arm_name,
                "nepochs": float(nepochs),
                "host": host,
                "visible_gpu": vis_gpu,
                "gpu_sku": gpu_sku,
                "gpu_power_w": float(power_w),
                "selected_gear": str(getattr(solver, "selected_gear", "n/a")),
                "resolved_precision": str(getattr(solver, "resolved_precision", "n/a")),
                "init_time_sec": init_s,
                "preencode_time_sec": preenc_s,
                "train_time_sec": train_s,
                "total_time_sec": total_s,
                "eval_time_sec": float(ivf_res.eval_time_sec),
                "train_mse": float(train_mse),
                "val_mse": float(val_mse),
                "test_mse": float(test_mse),
                "val_cos_dist": float(val_cos),
                "test_cos_dist": float(test_cos),
                "fp8_vs_bf16_test_mse_rel_pct": 0.0,
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
            seed_map[(arm_name, float(nepochs))] = rec

        # Populate exact paired FP8-vs-BF16 Test MSE delta for each budget `T`
        for b_t in budgets:
            r_bf16 = seed_map[("apex_emeans_bf16", float(b_t))]
            bf16_mse = max(abs(float(r_bf16["test_mse"])), 1e-30)
            for target_arm in ("apex_emeans_fp8", "apex_emeans"):
                r_tgt = seed_map[(target_arm, float(b_t))]
                r_tgt["fp8_vs_bf16_test_mse_rel_pct"] = float(
                    100.0 * (float(r_tgt["test_mse"]) - float(r_bf16["test_mse"])) / bf16_mse
                )

        if bf16_probe_centroids is not None:
            p_res = evaluate_fp8_assignment_stability_probe(
                x_eval=x_query_gpu,
                centroids=bf16_probe_centroids,
                origin=origin_gpu,
                device=device,
            )
            fp8_probes.append(
                {
                    "cell_id": cell.cell_id,
                    "corpus": cell.corpus_name,
                    "short_title": cell.short_title,
                    "modality": cell.modality,
                    "seed": int(seed),
                    "probe_nepochs": float(probe_budget),
                    "feature_std_max_over_min": float(dyn_stats["feature_std_max_over_min"]),
                    "row_norm_p99_over_p01": float(dyn_stats["row_norm_p99_over_p01"]),
                    "origin_shift_ratio": float(dyn_stats["origin_shift_ratio"]),
                    **p_res,
                }
            )

        for key in base_specs:
            records.append(seed_map[key])

        print(
            f"[Phase5-Dom] {cell.cell_id} seed={seed} ({s_idx + 1}/{len(seeds)}) "
            f"done in {time.perf_counter() - t_s0:.2f}s",
            flush=True,
        )

    return records, fp8_probes


@torch.no_grad()
def execute_ladder_cell_seeds(
    cell: LadderCellSpec,
    seeds: list[int],
    budgets: tuple[float, ...],
    data_dir: Path,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Execute Suite 2 (6-step Optimization Speedup Ladder) with cyclic counterbalancing."""
    if not seeds:
        return []

    host = socket.gethostname()
    vis_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    gpu_sku, power_w = detect_gpu_sku_info(device)

    sift_full = load_realistic_benchmark("sift1m", data_dir=data_dir, device=device)
    nb = min(int(cell.n_base), sift_full.base.shape[0])
    nq = min(int(cell.n_query), sift_full.query.shape[0])
    xb_np = np.ascontiguousarray(sift_full.base[:nb], dtype=np.float32)
    xq_np = np.ascontiguousarray(sift_full.query[:nq], dtype=np.float32)

    if nb == sift_full.base.shape[0] and nq == sift_full.query.shape[0]:
        gt_ids = sift_full.groundtruth_ids
        gt_d2 = sift_full.groundtruth_sq_dists
    else:
        gt_ids, gt_d2 = compute_exact_groundtruth_gpu(xb_np, xq_np, top_k=min(100, nb), device=device)

    x_base_gpu = torch.from_numpy(xb_np).to(device=device, dtype=torch.float32).contiguous()
    x_query_gpu = torch.from_numpy(xq_np).to(device=device, dtype=torch.float32).contiguous()

    print(
        f"[Phase5-Ladder] Warming {cell.cell_id} (N={nb:,}, D={cell.d}, K={cell.k:,}) "
        f"on {host} GPU={vis_gpu} [{gpu_sku}] ...",
        flush=True,
    )
    t_w0 = time.perf_counter()
    warmup_ladder_cell(
        x_base_gpu=x_base_gpu,
        x_query_gpu=x_query_gpu,
        gt_ids=gt_ids,
        gt_d2=gt_d2,
        cell=cell,
        budgets=budgets,
        device=device,
    )
    print(f"[Phase5-Ladder] Warmup done for {cell.cell_id} in {time.perf_counter() - t_w0:.2f}s", flush=True)

    base_specs: list[tuple[str, float]] = [
        (arm, float(b_t)) for b_t in budgets for arm in LADDER_ORDERED_ARMS
    ]

    records: list[dict[str, Any]] = []
    for s_idx, seed in enumerate(seeds):
        t_s0 = time.perf_counter()
        rot = int(seed) % len(base_specs)
        ordered_specs = base_specs[rot:] + base_specs[:rot]

        seed_map: dict[tuple[str, float], dict[str, Any]] = {}
        for arm_name, nepochs in ordered_specs:
            solver = instantiate_ladder_arm(arm_name, d=cell.d, k=cell.k, nepochs=nepochs, seed=seed)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            centroids, _ = solver.fit(x_base_gpu, capture_assignments=False)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

            init_s = float(solver.init_time_sec)
            train_s = float(solver.train_time_sec)
            preenc_s = float(getattr(solver, "preencode_time_sec", 0.0))
            total_s = float(init_s + train_s)

            train_mse, db_labels = fast_exact_assign_fp64(x_base_gpu, centroids, device=device)
            test_mse, _ = fast_exact_assign_fp64(x_query_gpu, centroids, device=device)
            ivf_res = evaluate_ivf_recall_gpu(
                x_base=x_base_gpu,
                centroids=centroids,
                x_query=x_query_gpu,
                groundtruth_ids=gt_ids,
                groundtruth_sq_dists=gt_d2,
                selectivities=SELECTIVITIES,
                db_labels=db_labels,
                device=device,
            )
            m10 = ivf_res.by_selectivity["1%"]

            seed_map[(arm_name, float(nepochs))] = {
                "suite": "ladder",
                "cell_id": cell.cell_id,
                "short_title": cell.short_title,
                "sku_tier": cell.sku_tier,
                "n_base": int(nb),
                "n_query": int(nq),
                "d": int(cell.d),
                "k": int(cell.k),
                "seed": int(seed),
                "arm": arm_name,
                "nepochs": float(nepochs),
                "host": host,
                "visible_gpu": vis_gpu,
                "gpu_sku": gpu_sku,
                "gpu_power_w": float(power_w),
                "selected_gear": str(getattr(solver, "selected_gear", "n/a")),
                "resolved_precision": str(getattr(solver, "resolved_precision", "n/a")),
                "init_time_sec": init_s,
                "preencode_time_sec": preenc_s,
                "train_time_sec": train_s,
                "total_time_sec": total_s,
                "train_mse": float(train_mse),
                "test_mse": float(test_mse),
                "recall10_10pct": float(m10.recall_at_10),
                "empty_clusters": int(ivf_res.empty_clusters),
            }

        for key in base_specs:
            records.append(seed_map[key])

        print(
            f"[Phase5-Ladder] {cell.cell_id} seed={seed} ({s_idx + 1}/{len(seeds)}) "
            f"done in {time.perf_counter() - t_s0:.2f}s",
            flush=True,
        )

    return records


# ==============================================================================
# Forensic Integrity Checks, Cold-JIT Spike Detector & Paired Statistics
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


def check_cold_jit_spikes(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Detect any `(cell_id, host, gpu, arm, nepochs)` group where earliest seed spiked from cold JIT.

    Flags a spike if the first executed seed on a shard exceeds `max(2.5 * median_rest, median_rest + 0.080s)`.
    """
    by_shard_group: dict[tuple[str, str, str, str, float], list[dict[str, Any]]] = {}
    for r in records:
        key = (
            str(r["cell_id"]),
            str(r["host"]),
            str(r["visible_gpu"]),
            str(r["arm"]),
            float(r["nepochs"]),
        )
        by_shard_group.setdefault(key, []).append(r)

    spikes: list[dict[str, Any]] = []
    for key, rows in by_shard_group.items():
        if len(rows) < 2:
            continue
        rows_sorted = sorted(rows, key=lambda x: int(x["seed"]))
        t0 = float(rows_sorted[0]["total_time_sec"])
        rest = [float(x["total_time_sec"]) for x in rows_sorted[1:]]
        med_rest = float(np.median(rest))
        i0 = float(rows_sorted[0]["init_time_sec"])
        i_rest = [float(x["init_time_sec"]) for x in rows_sorted[1:]]
        med_i_rest = float(np.median(i_rest))
        if t0 > max(2.2 * med_rest, med_rest + 0.050) or i0 > max(2.5 * med_i_rest, med_i_rest + 0.040):
            spikes.append(
                {
                    "cell_id": key[0],
                    "host": key[1],
                    "visible_gpu": key[2],
                    "arm": key[3],
                    "nepochs": key[4],
                    "first_seed": int(rows_sorted[0]["seed"]),
                    "first_time_ms": 1000.0 * t0,
                    "median_rest_ms": 1000.0 * med_rest,
                    "first_init_ms": 1000.0 * i0,
                    "median_init_rest_ms": 1000.0 * med_i_rest,
                }
            )
    return spikes


def validate_all_records(
    dom_records: list[dict[str, Any]],
    ladder_records: list[dict[str, Any]],
    fp8_probes: list[dict[str, Any]],
    dom_cells: list[DominanceCellSpec],
    ladder_cells: list[LadderCellSpec],
    expected_seeds: list[int],
    dom_budgets: tuple[float, ...],
    ladder_budgets: tuple[float, ...],
) -> list[dict[str, Any]]:
    """Verify completeness, finiteness, selectivity monotonicity, single-SKU pinning, and cold-JIT hygiene."""
    _assert_finite_recursive(dom_records, "dom_records")
    _assert_finite_recursive(ladder_records, "ladder_records")
    _assert_finite_recursive(fp8_probes, "fp8_probes")

    exp_dom = len(dom_cells) * len(expected_seeds) * (1 + len(DOMINANCE_STREAMING_ARMS) * len(dom_budgets))
    if len(dom_records) != exp_dom:
        raise RuntimeError(f"Dominance record count mismatch: got {len(dom_records)}, expected {exp_dom}")

    exp_lad = len(ladder_cells) * len(expected_seeds) * len(LADDER_ORDERED_ARMS) * len(ladder_budgets)
    if len(ladder_records) != exp_lad:
        raise RuntimeError(f"Ladder record count mismatch: got {len(ladder_records)}, expected {exp_lad}")

    exp_probes = len(dom_cells) * len(expected_seeds)
    if len(fp8_probes) != exp_probes:
        raise RuntimeError(f"FP8 probe count mismatch: got {len(fp8_probes)}, expected {exp_probes}")

    for suite_label, recs in (("dominance", dom_records), ("ladder", ladder_records)):
        seen_keys: set[tuple[str, int, str, float]] = set()
        seed_gpu_map: dict[tuple[str, int], set[tuple[str, str]]] = {}
        cell_sku_map: dict[str, set[str]] = {}

        for r in recs:
            rk = (str(r["cell_id"]), int(r["seed"]), str(r["arm"]), float(r["nepochs"]))
            if rk in seen_keys:
                raise RuntimeError(f"Duplicate {suite_label} record key {rk!r}")
            seen_keys.add(rk)

            if abs(float(r["total_time_sec"]) - (float(r["init_time_sec"]) + float(r["train_time_sec"]))) > 1e-6:
                raise RuntimeError(f"Timing sum mismatch in {rk!r}")

            if suite_label == "dominance":
                # Verify IVF Recall selectivity monotonicity: 0.1% <= 0.5% <= 1.0%
                for rk_tag in ("recall1", "recall10", "recall100"):
                    v01 = float(r[f"{rk_tag}_01pct"])
                    v05 = float(r[f"{rk_tag}_05pct"])
                    v10 = float(r[f"{rk_tag}_10pct"])
                    if not (v01 <= v05 + 1e-9 and v05 <= v10 + 1e-9):
                        raise RuntimeError(
                            f"Selectivity monotonicity violation in {rk!r} {rk_tag}: {v01} -> {v05} -> {v10}"
                        )

            sg_key = (str(r["cell_id"]), int(r["seed"]))
            seed_gpu_map.setdefault(sg_key, set()).add((str(r["host"]), str(r["visible_gpu"])))
            cell_sku_map.setdefault(str(r["cell_id"]), set()).add(str(r["gpu_sku"]))

        for sg_key, gpus in seed_gpu_map.items():
            if len(gpus) != 1:
                raise RuntimeError(f"CONFOUNDING VIOLATION ({suite_label}): {sg_key} split across GPUs {gpus!r}")

        for cid, skus in cell_sku_map.items():
            if len(skus) != 1:
                raise RuntimeError(f"CROSS-SKU VIOLATION ({suite_label}): {cid} mixed GPU SKUs {skus!r}")

    spikes = check_cold_jit_spikes(dom_records + ladder_records)
    return spikes


def compute_paired_comparisons(
    dom_records: list[dict[str, Any]],
    dom_cells: list[DominanceCellSpec],
    budgets: tuple[float, ...],
) -> list[dict[str, Any]]:
    """Compute paired Student-t and Wilcoxon signed-rank statistics across all 7 realistic benchmarks."""
    idx: dict[tuple[str, float, str, int], dict[str, Any]] = {}
    seeds_by_cell: dict[str, set[int]] = {}
    for r in dom_records:
        cid = str(r["cell_id"])
        idx[(cid, float(r["nepochs"]), str(r["arm"]), int(r["seed"]))] = r
        seeds_by_cell.setdefault(cid, set()).add(int(r["seed"]))

    candidates = ("apex_emeans", "apex_emeans_fp8", "apex_emeans_bf16")
    baselines = (
        "lloyd_25",
        "emeans",
        "smeans_ref",
        "smeans_fast",
        "emeans_pp",
        "super_emeans",
        "hybrid_semeans",
        "ultra_emeans",
    )
    metrics = (
        ("test_mse", "lower"),
        ("recall10_10pct", "higher"),
        ("recall10_01pct", "higher"),
        ("total_time_sec", "lower"),
    )

    out: list[dict[str, Any]] = []
    for cell in dom_cells:
        seeds = sorted(seeds_by_cell.get(cell.cell_id, set()))
        if not seeds:
            continue
        for b_t in budgets:
            for cand in candidates:
                for base in baselines:
                    base_t = 25.0 if base == "lloyd_25" else float(b_t)
                    for m_name, direction in metrics:
                        c_vals = np.array([float(idx[(cell.cell_id, float(b_t), cand, s)][m_name]) for s in seeds])
                        b_vals = np.array([float(idx[(cell.cell_id, base_t, base, s)][m_name]) for s in seeds])
                        diffs = c_vals - b_vals
                        rel_pct = float(100.0 * (c_vals.mean() - b_vals.mean()) / max(abs(b_vals.mean()), 1e-12))
                        abs_delta = float(c_vals.mean() - b_vals.mean())
                        speedup_x = (
                            float(b_vals.mean() / max(c_vals.mean(), 1e-9))
                            if m_name == "total_time_sec"
                            else 1.0
                        )

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

                        out.append(
                            {
                                "cell_id": cell.cell_id,
                                "corpus": cell.corpus_name,
                                "k": int(cell.k),
                                "nepochs": float(b_t),
                                "arm": cand,
                                "baseline": base,
                                "metric": m_name,
                                "direction": direction,
                                "arm_mean": float(c_vals.mean()),
                                "arm_std": float(c_vals.std(ddof=1)) if len(seeds) > 1 else 0.0,
                                "base_mean": float(b_vals.mean()),
                                "base_std": float(b_vals.std(ddof=1)) if len(seeds) > 1 else 0.0,
                                "abs_delta": abs_delta,
                                "rel_delta_pct": rel_pct,
                                "speedup_x": speedup_x,
                                "wins": wins,
                                "ties": ties,
                                "losses": losses,
                                "p_ttest": p_t if math.isfinite(p_t) else 1.0,
                                "p_wilcoxon": p_w if math.isfinite(p_w) else 1.0,
                            }
                        )
    return out


# ==============================================================================
# Publication-Grade 4-Figure Suite (`outputs/phase5_dominance/` & `plots/`)
# ==============================================================================


def generate_all_plots(
    dom_records: list[dict[str, Any]],
    ladder_records: list[dict[str, Any]],
    fp8_probes: list[dict[str, Any]],
    dom_cells: list[DominanceCellSpec],
    ladder_cells: list[LadderCellSpec],
    dom_budgets: tuple[float, ...],
    ladder_budgets: tuple[float, ...],
    out_dir: Path,
) -> list[Path]:
    """Generate all 4 required headline plots with atomic PID+ns `os.replace` & `PIL.Image.verify()`."""
    plt.rcParams.update(
        {
            "font.size": 9.0,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.0,
            "legend.fontsize": 8.0,
            "figure.dpi": 220,
        }
    )

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def _save_atomic(fig: plt.Figure, filename: str) -> None:
        for dst in (out_dir / filename, plots_dir / filename):
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.parent / f".tmp_{os.getpid()}_{time.time_ns()}_{dst.name}"
            try:
                fig.savefig(tmp, dpi=220, bbox_inches="tight", format="png")
                with Image.open(tmp) as im:
                    im.verify()
                os.replace(tmp, dst)
                written.append(dst)
            finally:
                if tmp.exists():
                    tmp.unlink(missing_ok=True)

    dom_grouped: dict[tuple[str, str, float], list[dict[str, Any]]] = {}
    cell_sku_short: dict[str, str] = {}
    for r in dom_records:
        dom_grouped.setdefault((str(r["cell_id"]), str(r["arm"]), float(r["nepochs"])), []).append(r)
        cell_sku_short[str(r["cell_id"])] = str(r["gpu_sku"]).replace(
            "NVIDIA RTX PRO 6000 Blackwell ", "Blackwell "
        )

    plot_arms = (
        "emeans",
        "smeans_ref",
        "smeans_fast",
        "emeans_pp",
        "super_emeans",
        "hybrid_semeans",
        "ultra_emeans",
        "apex_emeans_bf16",
        "apex_emeans_fp8",
        "apex_emeans",
    )

    # --------------------------------------------------------------------------
    # Figure 1: `apex_7benchmark_dominance_pareto.png` (2x4 Grid: 7 Benchmarks + Summary Scorecard)
    # --------------------------------------------------------------------------
    fig1, axes1 = plt.subplots(2, 4, figsize=(21.0, 9.8), squeeze=False)
    for idx_c, cell in enumerate(dom_cells[:7]):
        ax = axes1[idx_c // 4][idx_c % 4]
        lloyd_rows = dom_grouped.get((cell.cell_id, "lloyd_25", 25.0), [])
        if lloyd_rows:
            l_ms = float(np.mean([1000.0 * float(x["total_time_sec"]) for x in lloyd_rows]))
            l_mse = float(np.mean([float(x["test_mse"]) for x in lloyd_rows]))
            lbl, col, mk, _ = ARM_DISPLAY["lloyd_25"]
            ax.axhline(l_mse, color=col, linestyle="--", linewidth=1.0, alpha=0.6)
            ax.scatter([l_ms], [l_mse], color=col, marker=mk, s=130, zorder=6, label=lbl)

        for arm in plot_arms:
            xs, ys, yerrs = [], [], []
            for b_t in dom_budgets:
                rows = dom_grouped.get((cell.cell_id, arm, float(b_t)), [])
                if not rows:
                    continue
                xs.append(float(np.mean([1000.0 * float(x["total_time_sec"]) for x in rows])))
                vals = [float(x["test_mse"]) for x in rows]
                ys.append(float(np.mean(vals)))
                yerrs.append(float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0)
            if not xs:
                continue
            lbl, col, mk, ls = ARM_DISPLAY[arm]
            is_apex = arm.startswith("apex_")
            ax.errorbar(
                xs,
                ys,
                yerr=yerrs,
                label=lbl,
                color=col,
                marker=mk,
                linestyle=ls,
                linewidth=2.3 if is_apex else 1.35,
                markersize=6.2 if is_apex else 4.8,
                capsize=2.0,
                zorder=7 if arm == "apex_emeans" else (6 if is_apex else 3),
            )

        ax.set_xscale("log")
        ax.set_title(f"{cell.short_title}\n[{cell_sku_short.get(cell.cell_id, '')}]", fontweight="bold")
        ax.set_xlabel("Total GPU Time: Init + Train (ms, log)")
        ax.set_ylabel("Held-Out FP64 Test MSE (lower is better)")
        ax.grid(True, which="both", linestyle=":", alpha=0.45)

    # Panel 8 (axes1[1][3]): Geometric-Mean Speedup & Mean Test MSE Gain of ApexEMeans Across All 7 Benchmarks (T=2)
    ax_sum = axes1[1][3]
    ref_b = 2.0 if 2.0 in dom_budgets else dom_budgets[-1]
    comp_targets = [
        ("lloyd_25", "vs Lloyd@25"),
        ("emeans", "vs e-means"),
        ("smeans_fast", "vs s-means"),
        ("emeans_pp", "vs e-means++"),
        ("super_emeans", "vs SuperEMeans"),
        ("ultra_emeans", "vs UltraEMeans"),
    ]
    mse_gains: list[float] = []
    speedups: list[float] = []
    bar_labels: list[str] = []
    for base_arm, blbl in comp_targets:
        base_t = 25.0 if base_arm == "lloyd_25" else ref_b
        rel_reds = []
        spds = []
        for cell in dom_cells:
            a_rows = dom_grouped.get((cell.cell_id, "apex_emeans", ref_b), [])
            b_rows = dom_grouped.get((cell.cell_id, base_arm, base_t), [])
            if a_rows and b_rows:
                a_m = float(np.mean([float(x["test_mse"]) for x in a_rows]))
                b_m = float(np.mean([float(x["test_mse"]) for x in b_rows]))
                a_t = float(np.mean([float(x["total_time_sec"]) for x in a_rows]))
                b_t_s = float(np.mean([float(x["total_time_sec"]) for x in b_rows]))
                rel_reds.append(100.0 * (b_m - a_m) / max(abs(b_m), 1e-12))
                spds.append(b_t_s / max(a_t, 1e-9))
        if rel_reds:
            mse_gains.append(float(np.mean(rel_reds)))
            speedups.append(float(np.exp(np.mean(np.log(np.maximum(spds, 1e-6))))))
            bar_labels.append(blbl)

    y_pos = np.arange(len(bar_labels))
    bars = ax_sum.barh(y_pos, mse_gains, color="#8b0000", alpha=0.88, height=0.58)
    ax_sum.axvline(0.0, color="#333333", linewidth=1.0)
    ax_sum.set_yticks(y_pos)
    ax_sum.set_yticklabels(bar_labels, fontweight="bold")
    ax_sum.invert_yaxis()
    ax_sum.set_xlabel(f"Mean Test MSE Reduction % at T={ref_b:g} (higher = Apex better)")
    ax_sum.set_title(f"7-Benchmark Summary: ApexEMeans Dominance (T={ref_b:g})", fontweight="bold")
    ax_sum.grid(True, axis="x", linestyle=":", alpha=0.45)
    for bar_obj, mg, sp in zip(bars, mse_gains, speedups):
        ax_sum.annotate(
            f" {mg:+.2f}% MSE | {sp:.2f}x speed",
            xy=(max(mg, 0.0), bar_obj.get_y() + bar_obj.get_height() / 2.0),
            va="center",
            ha="left",
            fontsize=8.0,
            fontweight="bold",
            color="#222222",
        )
    if mse_gains:
        ax_sum.set_xlim(min(0.0, min(mse_gains) * 1.30), max(0.5, max(mse_gains)) * 1.85 + 0.8)

    handles1, labels1 = axes1[0][0].get_legend_handles_labels()
    fig1.legend(handles1, labels1, loc="upper center", ncol=6, frameon=True, bbox_to_anchor=(0.5, 1.03))
    fig1.suptitle(
        "Phase 5 Multi-Domain Realistic Benchmark Suite (7 Modalities): Held-Out FP64 Test MSE vs. GPU Build Time",
        fontsize=13.0,
        fontweight="bold",
        y=1.065,
    )
    fig1.tight_layout()
    _save_atomic(fig1, "apex_7benchmark_dominance_pareto.png")
    plt.close(fig1)

    # --------------------------------------------------------------------------
    # Figure 2: `apex_optimization_ladder.png` (Stepwise Systems Speedup Waterfall)
    # --------------------------------------------------------------------------
    lad_grouped: dict[tuple[str, str, float], list[dict[str, Any]]] = {}
    lad_sku: dict[str, str] = {}
    for r in ladder_records:
        lad_grouped.setdefault((str(r["cell_id"]), str(r["arm"]), float(r["nepochs"])), []).append(r)
        lad_sku[str(r["cell_id"])] = str(r["gpu_sku"]).replace("NVIDIA RTX PRO 6000 Blackwell ", "Blackwell ")

    target_lad_t = 5.0 if 5.0 in ladder_budgets else ladder_budgets[-1]
    fig2, axes2 = plt.subplots(1, len(ladder_cells), figsize=(9.2 * len(ladder_cells), 6.2), squeeze=False)
    last_ax2 = None
    for idx_l, lcell in enumerate(ladder_cells):
        ax = axes2[0][idx_l]
        init_ms, pre_ms, core_train_ms, tot_ms, test_mses = [], [], [], [], []
        xlabels = []
        for arm in LADDER_ORDERED_ARMS:
            rows = lad_grouped.get((lcell.cell_id, arm, target_lad_t), [])
            if not rows:
                continue
            im = float(np.mean([1000.0 * float(x["init_time_sec"]) for x in rows]))
            pm = float(np.mean([1000.0 * float(x["preencode_time_sec"]) for x in rows]))
            tm = float(np.mean([1000.0 * float(x["train_time_sec"]) for x in rows]))
            init_ms.append(im)
            pre_ms.append(min(pm, tm))
            core_train_ms.append(max(0.0, tm - pm))
            tot_ms.append(im + tm)
            test_mses.append(float(np.mean([float(x["test_mse"]) for x in rows])))
            xlabels.append(LADDER_STEP_LABELS[arm])

        xpos = np.arange(len(xlabels))
        ax.bar(xpos, init_ms, width=0.52, color="#4e79a7", label="Seeding / Init (ms)")
        ax.bar(xpos, pre_ms, bottom=init_ms, width=0.52, color="#f28e2b", label="Pre-Permute / FP8 Encode (ms)")
        bottoms = [a + b for a, b in zip(init_ms, pre_ms)]
        ax.bar(xpos, core_train_ms, bottom=bottoms, width=0.52, color="#d62728", alpha=0.88, label="Streaming Epoch Loop (ms)")

        base0 = tot_ms[0] if tot_ms else 1.0
        for xi, t_cur in enumerate(tot_ms):
            cum_sp = base0 / max(t_cur, 1e-6)
            prev_t = tot_ms[xi - 1] if xi > 0 else t_cur
            delta_ms = t_cur - prev_t
            badge = f"{t_cur:.1f} ms\n(1.00x)" if xi == 0 else f"{t_cur:.1f} ms\n({cum_sp:.2f}x | {delta_ms:+.1f}ms)"
            ax.annotate(
                badge,
                xy=(xi, t_cur),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.6,
                fontweight="bold",
            )

        if tot_ms:
            # Leave upper 25% of axis strictly clear for the Held-Out Test MSE curve
            ax.set_ylim(0.0, max(tot_ms) * 1.48)

        ax2 = ax.twinx()
        last_ax2 = ax2
        ax2.plot(xpos, test_mses, color="#111111", marker="o", linewidth=1.9, linestyle="--", label="Held-Out FP64 Test MSE")
        ax2.set_ylabel("Held-Out FP64 Test MSE (dashed line)")
        if test_mses:
            m_min, m_max = min(test_mses), max(test_mses)
            m_mean = float(np.mean(test_mses))
            span = max(m_max - m_min, 0.008 * abs(m_mean), 1e-6)
            # Position dashed MSE curve cleanly in upper ~80-92% band above bar annotations
            ax2.set_ylim(m_min - 9.0 * span, m_max + 1.4 * span)

        ax.set_xticks(xpos)
        ax.set_xticklabels(xlabels, fontsize=8.0)
        ax.set_ylabel(f"Total GPU Wall-Clock Time at T={target_lad_t:g} (ms)")
        ax.set_title(
            f"{lcell.short_title} (T={target_lad_t:g})\n[{lad_sku.get(lcell.cell_id, '')}]",
            fontweight="bold",
        )
        ax.grid(True, axis="y", linestyle=":", alpha=0.45)

    h_bar, l_bar = axes2[0][0].get_legend_handles_labels()
    if last_ax2 is not None:
        h_l2, l_l2 = last_ax2.get_legend_handles_labels()
        h_bar = list(h_bar) + list(h_l2)
        l_bar = list(l_bar) + list(l_l2)
    fig2.legend(h_bar, l_bar, loc="upper center", ncol=4, frameon=True, bbox_to_anchor=(0.5, 1.03))
    fig2.suptitle(
        "ApexEMeans Stepwise Optimization Speedup Ladder: Batch -> Fuse -> SM120 Kernel -> CUDAGraph -> CroissAttn-FP8",
        fontsize=12.2,
        fontweight="bold",
        y=1.08,
    )
    fig2.tight_layout()
    _save_atomic(fig2, "apex_optimization_ladder.png")
    plt.close(fig2)

    # --------------------------------------------------------------------------
    # Figure 3: `apex_fp8_stability_across_modalities.png` (7-Modality FP8 Stability Proof)
    # --------------------------------------------------------------------------
    fig3, (ax3_l, ax3_r) = plt.subplots(1, 2, figsize=(16.8, 5.9))

    probe_by_cell: dict[str, list[dict[str, Any]]] = {}
    for p in fp8_probes:
        probe_by_cell.setdefault(str(p["cell_id"]), []).append(p)

    mod_labels: list[str] = []
    naive_infl: list[float] = []
    croiss_infl: list[float] = []
    trained_delta_t2: list[float] = []
    trained_ci_t2: list[float] = []
    trained_speedup_t2: list[float] = []

    ref_t_fp8 = 2.0 if 2.0 in dom_budgets else dom_budgets[-1]
    for cell in dom_cells:
        p_rows = probe_by_cell.get(cell.cell_id, [])
        std_sp = float(p_rows[0]["feature_std_max_over_min"]) if p_rows else 1.0
        mod_labels.append(f"{cell.corpus_name}\n(d={cell.d}, span={std_sp:.0f}x)")
        naive_infl.append(float(np.mean([float(x["naive_global_fp8_rel_inflate_pct"]) for x in p_rows])) if p_rows else 0.0)
        croiss_infl.append(float(np.mean([float(x["croiss_decoupled_fp8_rel_inflate_pct"]) for x in p_rows])) if p_rows else 0.0)

        fp8_rows = dom_grouped.get((cell.cell_id, "apex_emeans_fp8", ref_t_fp8), [])
        bf16_rows = dom_grouped.get((cell.cell_id, "apex_emeans_bf16", ref_t_fp8), [])
        if fp8_rows and bf16_rows:
            deltas = [float(x["fp8_vs_bf16_test_mse_rel_pct"]) for x in fp8_rows]
            trained_delta_t2.append(float(np.mean(deltas)))
            trained_ci_t2.append(
                1.96 * float(np.std(deltas, ddof=1) / math.sqrt(len(deltas))) if len(deltas) > 1 else 0.0
            )
            t_fp8 = float(np.mean([float(x["total_time_sec"]) for x in fp8_rows]))
            t_bf16 = float(np.mean([float(x["total_time_sec"]) for x in bf16_rows]))
            trained_speedup_t2.append(t_bf16 / max(t_fp8, 1e-9))
        else:
            trained_delta_t2.append(0.0)
            trained_ci_t2.append(0.0)
            trained_speedup_t2.append(1.0)

    y_m = np.arange(len(mod_labels))
    h_w = 0.36
    ax3_l.barh(y_m - h_w / 2, np.maximum(naive_infl, 1e-4), height=h_w, color="#7f7f7f", label="Naive Global Uncentered FP8")
    ax3_l.barh(y_m + h_w / 2, np.maximum(croiss_infl, 1e-4), height=h_w, color="#d62728", label="CroissAttn Origin-Centered Decoupled FP8")
    ax3_l.set_xscale("log")
    ax3_l.set_yticks(y_m)
    ax3_l.set_yticklabels(mod_labels, fontsize=8.2)
    ax3_l.invert_yaxis()
    ax3_l.set_xlabel("Single-Pass Assignment FP64 MSE Inflation % (log scale, lower is better)")
    ax3_l.set_title("Quantization Stress Test: Naive FP8 vs. CroissAttn Decoupled FP8", fontweight="bold")
    ax3_l.legend(loc="lower right")
    ax3_l.grid(True, which="both", axis="x", linestyle=":", alpha=0.45)

    ax3_r.axvline(0.0, color="#222222", linewidth=1.1)
    ax3_r.axvspan(-0.15, 0.15, color="#2ca02c", alpha=0.12, label="±0.15% Parity Band")
    bars_r = ax3_r.barh(
        y_m,
        trained_delta_t2,
        xerr=trained_ci_t2,
        height=0.52,
        color="#d62728",
        alpha=0.88,
        capsize=2.5,
        label=f"Trained ApexEMeans (CroissFP8 vs BF16, T={ref_t_fp8:g})",
    )
    ax3_r.set_yticks(y_m)
    ax3_r.set_yticklabels(mod_labels, fontsize=8.2)
    ax3_r.invert_yaxis()
    ax3_r.set_xlabel(f"End-to-End Held-Out Test MSE Delta % vs. BF16 at T={ref_t_fp8:g} (mean ± 95% CI)")
    ax3_r.set_title("End-to-End Training Stability: ApexEMeans (CroissAttn-FP8) vs. BF16", fontweight="bold")
    ax3_r.legend(loc="lower right")
    ax3_r.grid(True, axis="x", linestyle=":", alpha=0.45)
    max_abs_d = max([abs(v) + c for v, c in zip(trained_delta_t2, trained_ci_t2)] + [0.25])
    for b_obj, d_val, ci_v, sp_v in zip(bars_r, trained_delta_t2, trained_ci_t2, trained_speedup_t2):
        x_anchor = max(d_val + ci_v, 0.16) + 0.03 * max_abs_d
        ax3_r.annotate(
            f"{d_val:+.3f}% ({sp_v:.2f}x)",
            xy=(x_anchor, b_obj.get_y() + b_obj.get_height() / 2.0),
            va="center",
            ha="left",
            fontsize=7.8,
            fontweight="bold",
        )
    ax3_r.set_xlim(-max_abs_d * 1.35, max_abs_d * 2.15 + 0.35)

    fig3.suptitle(
        "CroissAttn-Inspired FP8 Numerical Stability Across All 7 Real-World Modalities (Up to 110,761x Feature Span)",
        fontsize=12.2,
        fontweight="bold",
        y=1.03,
    )
    fig3.tight_layout()
    _save_atomic(fig3, "apex_fp8_stability_across_modalities.png")
    plt.close(fig3)

    # --------------------------------------------------------------------------
    # Figure 4: `apex_ivf_recall_across_modalities.png` (2x4 Grid: 7 Modalities IVF R@10 @ 1% + Summary)
    # --------------------------------------------------------------------------
    fig4, axes4 = plt.subplots(2, 4, figsize=(21.0, 9.8), squeeze=False)
    for idx_c, cell in enumerate(dom_cells[:7]):
        ax = axes4[idx_c // 4][idx_c % 4]
        lloyd_rows = dom_grouped.get((cell.cell_id, "lloyd_25", 25.0), [])
        np_1pct = int(lloyd_rows[0]["nprobe_10pct"]) if lloyd_rows else max(1, round(0.01 * cell.k))
        if lloyd_rows:
            l_ms = float(np.mean([1000.0 * float(x["total_time_sec"]) for x in lloyd_rows]))
            l_r10 = float(100.0 * np.mean([float(x["recall10_10pct"]) for x in lloyd_rows]))
            lbl, col, mk, _ = ARM_DISPLAY["lloyd_25"]
            ax.axhline(l_r10, color=col, linestyle="--", linewidth=1.0, alpha=0.6)
            ax.scatter([l_ms], [l_r10], color=col, marker=mk, s=130, zorder=6, label=lbl)

        for arm in plot_arms:
            xs, ys, yerrs = [], [], []
            for b_t in dom_budgets:
                rows = dom_grouped.get((cell.cell_id, arm, float(b_t)), [])
                if not rows:
                    continue
                xs.append(float(np.mean([1000.0 * float(x["total_time_sec"]) for x in rows])))
                vals = [100.0 * float(x["recall10_10pct"]) for x in rows]
                ys.append(float(np.mean(vals)))
                yerrs.append(float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0)
            if not xs:
                continue
            lbl, col, mk, ls = ARM_DISPLAY[arm]
            is_apex = arm.startswith("apex_")
            ax.errorbar(
                xs,
                ys,
                yerr=yerrs,
                label=lbl,
                color=col,
                marker=mk,
                linestyle=ls,
                linewidth=2.3 if is_apex else 1.35,
                markersize=6.2 if is_apex else 4.8,
                capsize=2.0,
                zorder=7 if arm == "apex_emeans" else (6 if is_apex else 3),
            )

        ax.set_xscale("log")
        ax.set_title(
            f"{cell.short_title} (nprobe={np_1pct}, 1%)\n[{cell_sku_short.get(cell.cell_id, '')}]",
            fontweight="bold",
        )
        ax.set_xlabel("Index Build Time: Init + Train (ms, log)")
        ax.set_ylabel("Test ANN IVF Recall@10 at 1% Selectivity (%)")
        ax.grid(True, which="both", linestyle=":", alpha=0.45)

    # Panel 8: Cross-modality IVF Recall@10 (1% selectivity) gain at T=ref_b
    ax4_sum = axes4[1][3]
    rec_gains: list[float] = []
    rec_labels: list[str] = []
    for base_arm, blbl in comp_targets:
        base_t = 25.0 if base_arm == "lloyd_25" else ref_b
        pp_diffs = []
        for cell in dom_cells:
            a_rows = dom_grouped.get((cell.cell_id, "apex_emeans", ref_b), [])
            b_rows = dom_grouped.get((cell.cell_id, base_arm, base_t), [])
            if a_rows and b_rows:
                a_r = 100.0 * float(np.mean([float(x["recall10_10pct"]) for x in a_rows]))
                b_r = 100.0 * float(np.mean([float(x["recall10_10pct"]) for x in b_rows]))
                pp_diffs.append(a_r - b_r)
        if pp_diffs:
            rec_gains.append(float(np.mean(pp_diffs)))
            rec_labels.append(blbl)

    y_r = np.arange(len(rec_labels))
    bars4 = ax4_sum.barh(y_r, rec_gains, color="#1b7837", alpha=0.88, height=0.58)
    ax4_sum.axvline(0.0, color="#333333", linewidth=1.0)
    ax4_sum.set_yticks(y_r)
    ax4_sum.set_yticklabels(rec_labels, fontweight="bold")
    ax4_sum.invert_yaxis()
    ax4_sum.set_xlabel(f"Mean Recall@10 Gain (percentage points) at T={ref_b:g}, 1% Selectivity")
    ax4_sum.set_title(f"7-Modality Mean IVF Recall@10 Advantage (T={ref_b:g})", fontweight="bold")
    ax4_sum.grid(True, axis="x", linestyle=":", alpha=0.45)
    for b_obj, rg in zip(bars4, rec_gains):
        ax4_sum.annotate(
            f" {rg:+.2f} pp",
            xy=(max(rg, 0.0), b_obj.get_y() + b_obj.get_height() / 2.0),
            va="center",
            ha="left",
            fontsize=8.2,
            fontweight="bold",
        )
    if rec_gains:
        ax4_sum.set_xlim(min(-0.2, min(rec_gains) * 1.3), max(rec_gains) * 1.55 + 0.35)

    handles4, labels4 = axes4[0][0].get_legend_handles_labels()
    fig4.legend(handles4, labels4, loc="upper center", ncol=6, frameon=True, bbox_to_anchor=(0.5, 1.03))
    fig4.suptitle(
        "Downstream Approximate Nearest Neighbor (IVF-Flat) Recall@10 at 1% Selectivity Across All 7 Modalities",
        fontsize=13.0,
        fontweight="bold",
        y=1.065,
    )
    fig4.tight_layout()
    _save_atomic(fig4, "apex_ivf_recall_across_modalities.png")
    plt.close(fig4)

    return written


# ==============================================================================
# Markdown Summary Writer
# ==============================================================================


def write_summary_markdown(
    dom_records: list[dict[str, Any]],
    ladder_records: list[dict[str, Any]],
    fp8_probes: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    cold_jit_spikes: list[dict[str, Any]],
    dom_cells: list[DominanceCellSpec],
    ladder_cells: list[LadderCellSpec],
    dom_budgets: tuple[float, ...],
    ladder_budgets: tuple[float, ...],
    out_path: Path,
) -> None:
    """Atomically write comprehensive statistical Markdown report to `phase5_dominance_summary.md`."""
    dom_grouped: dict[tuple[str, str, float], list[dict[str, Any]]] = {}
    for r in dom_records:
        dom_grouped.setdefault((str(r["cell_id"]), str(r["arm"]), float(r["nepochs"])), []).append(r)

    lad_grouped: dict[tuple[str, str, float], list[dict[str, Any]]] = {}
    for r in ladder_records:
        lad_grouped.setdefault((str(r["cell_id"]), str(r["arm"]), float(r["nepochs"])), []).append(r)

    lines: list[str] = [
        "# Phase 5 Multi-Domain Realistic Benchmark Dominance & Optimization Speedup Summary",
        "",
        f"- **Dominance Suite Runs**: `{len(dom_records):,}` across `{len(dom_cells)}` real-world modalities (`0` `NaN`/`Inf`, `100%` single-SKU per cell).",
        f"- **Optimization Speedup Ladder Runs**: `{len(ladder_records):,}` across `{len(ladder_cells)}` shapes (`250K x 128, K=4,096` and `1M x 128, K=16,384`).",
        f"- **Cold-JIT Timing Spikes Detected**: `{len(cold_jit_spikes)}`.",
        "",
        "## 1. Per-Modality Quantization MSE, IVF Recall@10 (`1%` Selectivity), FP8 Stability & GPU Build Time",
        "",
    ]

    for cell in dom_cells:
        sku = next((str(r["gpu_sku"]) for r in dom_records if r["cell_id"] == cell.cell_id), "Unknown")
        n_b = next((int(r["n_base"]) for r in dom_records if r["cell_id"] == cell.cell_id), 0)
        lines.extend(
            [
                f"### `{cell.cell_id}` — {cell.modality} (`N={n_b:,}, D={cell.d}, K={cell.k:,}`, `{sku}`)",
                "",
                "| Arm | `T` | Build `ms` (`init + train`) | Train MSE | Test MSE | FP8 vs BF16 | Empty | R@10 (`0.1%`) | R@10 (`1.0%`) |",
                "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        run_order: list[tuple[str, float]] = [("lloyd_25", 25.0)]
        for b_t in dom_budgets:
            for arm in DOMINANCE_STREAMING_ARMS:
                run_order.append((arm, float(b_t)))

        for arm, b_t in run_order:
            rows = dom_grouped.get((cell.cell_id, arm, b_t), [])
            if not rows:
                continue
            tot_ms = float(np.mean([1000.0 * float(x["total_time_sec"]) for x in rows]))
            ini_ms = float(np.mean([1000.0 * float(x["init_time_sec"]) for x in rows]))
            trn_ms = float(np.mean([1000.0 * float(x["train_time_sec"]) for x in rows]))
            tr_m = float(np.mean([float(x["train_mse"]) for x in rows]))
            te_m = float(np.mean([float(x["test_mse"]) for x in rows]))
            fp8_d = float(np.mean([float(x["fp8_vs_bf16_test_mse_rel_pct"]) for x in rows]))
            emp = float(np.mean([float(x["empty_clusters"]) for x in rows]))
            r10_01 = 100.0 * float(np.mean([float(x["recall10_01pct"]) for x in rows]))
            r10_10 = 100.0 * float(np.mean([float(x["recall10_10pct"]) for x in rows]))
            fp8_s = f"`{fp8_d:+.3f}%`" if arm in ("apex_emeans_fp8", "apex_emeans") else "—"
            lines.append(
                f"| `{arm}` | `{b_t:g}` | `{tot_ms:.2f}` (`{ini_ms:.2f}+{trn_ms:.2f}`) | "
                f"`{tr_m:.4f}` | `{te_m:.4f}` | {fp8_s} | `{emp:.1f}` | `{r10_01:.2f}%` | `{r10_10:.2f}%` |"
            )
        lines.append("")

    lines.extend(
        [
            "## 2. Stepwise Optimization Speedup Ladder (`+Batch -> +Fuse -> +Kernel -> +CUDAGraph -> +CroissAttn-FP8`)",
            "",
        ]
    )
    for lcell in ladder_cells:
        sku = next((str(r["gpu_sku"]) for r in ladder_records if r["cell_id"] == lcell.cell_id), "Unknown")
        lines.extend(
            [
                f"### `{lcell.cell_id}` (`N={lcell.n_base:,}, D={lcell.d}, K={lcell.k:,}`, `{sku}`)",
                "",
                "| `T` | Ladder Step | Total `ms` | Init `ms` | PreEncode `ms` | Train `ms` | Step Delta `ms` | Cumul Speedup | Test MSE | R@10 (`1%`) |",
                "| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for b_t in ladder_budgets:
            base0_ms = None
            prev_ms = None
            for arm in LADDER_ORDERED_ARMS:
                rows = lad_grouped.get((lcell.cell_id, arm, float(b_t)), [])
                if not rows:
                    continue
                tot_ms = float(np.mean([1000.0 * float(x["total_time_sec"]) for x in rows]))
                ini_ms = float(np.mean([1000.0 * float(x["init_time_sec"]) for x in rows]))
                pre_ms = float(np.mean([1000.0 * float(x["preencode_time_sec"]) for x in rows]))
                trn_ms = float(np.mean([1000.0 * float(x["train_time_sec"]) for x in rows]))
                te_m = float(np.mean([float(x["test_mse"]) for x in rows]))
                r10 = 100.0 * float(np.mean([float(x["recall10_10pct"]) for x in rows]))
                if base0_ms is None:
                    base0_ms = tot_ms
                d_ms = 0.0 if prev_ms is None else (tot_ms - prev_ms)
                prev_ms = tot_ms
                cum_x = base0_ms / max(tot_ms, 1e-6)
                lines.append(
                    f"| `{b_t:g}` | `{arm}` | `{tot_ms:.2f}` | `{ini_ms:.2f}` | `{pre_ms:.2f}` | `{trn_ms:.2f}` | "
                    f"`{d_ms:+.2f}` | **`{cum_x:.2f}x`** | `{te_m:.4f}` | `{r10:.2f}%` |"
                )
        lines.append("")

    lines.extend(
        [
            "## 3. CroissAttn-FP8 vs. Naive Uncentered Global FP8 Multi-Scale Stress Test",
            "",
            "| Cell | Modality | Feature Std Span | Naive Global FP8 Inflate | CroissAttn Decoupled FP8 Inflate | Error Reduction Factor |",
            "| :--- | :--- | ---: | ---: | ---: | ---: |",
        ]
    )
    probe_by_cell: dict[str, list[dict[str, Any]]] = {}
    for p in fp8_probes:
        probe_by_cell.setdefault(str(p["cell_id"]), []).append(p)
    for cell in dom_cells:
        p_rows = probe_by_cell.get(cell.cell_id, [])
        if not p_rows:
            continue
        sp = float(p_rows[0]["feature_std_max_over_min"])
        n_inf = float(np.mean([float(x["naive_global_fp8_rel_inflate_pct"]) for x in p_rows]))
        c_inf = float(np.mean([float(x["croiss_decoupled_fp8_rel_inflate_pct"]) for x in p_rows]))
        ratio = abs(n_inf) / max(abs(c_inf), 1e-6)
        lines.append(
            f"| `{cell.cell_id}` | {cell.modality} | `{sp:,.1f}x` | `{n_inf:+.3f}%` | `{c_inf:+.4f}%` | **`{ratio:.1f}x`** |"
        )
    lines.append("")

    lines.extend(
        [
            "## 4. Key Paired Statistical Comparisons (`apex_emeans` & `apex_emeans_fp8` vs. Baselines)",
            "",
            "| Cell | `T` | Candidate | Baseline | Metric | Cand Mean | Base Mean | Delta / Speedup | W/T/L | `p_ttest` | `p_wilcox` |",
            "| :--- | ---: | :--- | :--- | :--- | ---: | ---: | ---: | :---: | ---: | ---: |",
        ]
    )
    for comp in comparisons:
        if comp["arm"] not in ("apex_emeans", "apex_emeans_fp8"):
            continue
        if comp["metric"] not in ("test_mse", "recall10_10pct", "total_time_sec"):
            continue
        if comp["nepochs"] not in (1.0, 2.0, 5.0):
            continue
        wtl = f"{comp['wins']}/{comp['ties']}/{comp['losses']}"
        if comp["metric"] == "test_mse":
            d_str = f"{comp['rel_delta_pct']:+.3f}%"
        elif comp["metric"] == "recall10_10pct":
            d_str = f"{100.0 * comp['abs_delta']:+.2f} pp"
        else:
            d_str = f"{comp['speedup_x']:.2f}x ({comp['rel_delta_pct']:+.1f}%)"
        lines.append(
            f"| `{comp['cell_id']}` | `{comp['nepochs']:g}` | `{comp['arm']}` | `{comp['baseline']}` | "
            f"`{comp['metric']}` | `{comp['arm_mean']:.5f}` | `{comp['base_mean']:.5f}` | "
            f"{d_str} | {wtl} | `{comp['p_ttest']:.3e}` | `{comp['p_wilcoxon']:.3e}` |"
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
# 6-Machine / 13-GPU Parallel Fleet Dispatcher (Zero WAN Re-Upload)
# ==============================================================================


def discover_live_ssh_auth_sock() -> str | None:
    """Discover active forwarded `SSH_AUTH_SOCK` without touching any `tmux` session `0` state."""
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


def assert_fleet_gpus_idle(shards: list[FleetShardAssignment], env: dict[str, str]) -> None:
    """Verify via `nvidia-smi` that every target GPU (remote & local) is idle before launching."""
    by_host: dict[str, list[int]] = {}
    for sp in shards:
        has_work = any(len(o) > 0 for _, o in sp.dominance_tasks) or any(len(o) > 0 for _, o in sp.ladder_tasks)
        if has_work:
            by_host.setdefault(sp.host, []).append(int(sp.gpu_id))

    for host, gpu_ids in sorted(by_host.items()):
        q_cmd = "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits"
        if host == "local":
            res = subprocess.run(shlex.split(q_cmd), capture_output=True, text=True, timeout=5, check=True)
        else:
            res = subprocess.run(
                ["ssh", "-A", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, q_cmd],
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
        status_by_idx: dict[int, tuple[float, float]] = {}
        for line in res.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                status_by_idx[int(parts[0])] = (float(parts[1]), float(parts[2]))
        for gid in gpu_ids:
            if gid not in status_by_idx:
                raise RuntimeError(f"GPU {gid} not found in nvidia-smi on {host}")
            mem_mib, util_pct = status_by_idx[gid]
            if util_pct > 20.0 or mem_mib > 2048.0:
                raise RuntimeError(
                    f"CONTENTION GUARD BLOCKED LAUNCH: {host} GPU {gid} busy "
                    f"(util={util_pct:.0f}%, mem={mem_mib:.0f} MiB)!"
                )


def dispatch_fleet_10way(
    num_seeds: int,
    seed_start: int,
    local_gpus: tuple[int, int, int, int],
    out_dir: Path,
    data_dir: Path,
    shard_filter: set[int] | None = None,
) -> None:
    """Dispatch idle GPU shards across clean Blackwell machines (`ripper`, `loki`, `work` & `walnut` excluded)."""
    shards = build_fleet_shards(num_seeds=num_seeds, local_gpus=local_gpus)
    num_shards = len(shards)
    assert num_shards == 8

    seen_sessions: set[str] = set()
    for sp in shards:
        h_low = sp.host.lower()
        if "walnut" in h_low or "ripper" in h_low or "loki" in h_low or h_low == "work" or h_low.startswith("work."):
            raise RuntimeError(f"SAFETY VIOLATION: forbidden/busy host {sp.host!r} in fleet shards!")
        if not sp.session.startswith("emeans_p5_dom_s") or sp.session == "0":
            raise RuntimeError(f"SAFETY VIOLATION: invalid tmux session {sp.session!r}")
        if sp.session in seen_sessions:
            raise RuntimeError(f"Duplicate session in {sp!r}")
        seen_sessions.add(sp.session)

    selected_shards = [
        sp for sp in shards
        if (shard_filter is None or sp.shard_idx in shard_filter)
        and (any(len(o) > 0 for _, o in sp.dominance_tasks) or any(len(o) > 0 for _, o in sp.ladder_tasks))
    ]
    if not selected_shards:
        raise ValueError(f"No active shards matched shard_filter={shard_filter!r}")

    seen_host_gpus: set[tuple[str, int]] = set()
    for sp in selected_shards:
        if (sp.host, sp.gpu_id) in seen_host_gpus:
            raise RuntimeError(
                f"Duplicate GPU assignment {(sp.host, sp.gpu_id)!r} in selected shards; "
                f"pass 4 distinct --local-gpus (e.g. '0,1,2,3') or split --shard-filter '4,5' then '6,7'."
            )
        seen_host_gpus.add((sp.host, sp.gpu_id))

    env = os.environ.copy()
    live_sock = discover_live_ssh_auth_sock()
    if live_sock:
        env["SSH_AUTH_SOCK"] = live_sock

    # Pre-flight idle-GPU check across all target remote & local GPUs before staging/launching
    assert_fleet_gpus_idle(selected_shards, env)

    # Ensure all 7 local validated `.npz` benchmark caches exist before syncing
    if torch.cuda.is_available():
        ord_idx = int(local_gpus[0]) if int(local_gpus[0]) < torch.cuda.device_count() else 0
        dev_local = torch.device(f"cuda:{ord_idx}")
    else:
        dev_local = torch.device("cpu")
    stage_all_seven_benchmarks(data_dir=data_dir, device=dev_local)

    dom_cell_by_id = {c.cell_id: c for c in build_dominance_cells(smoke=False)}
    host_needed_npz: dict[str, set[str]] = {}
    for sp in selected_shards:
        if sp.host == "local":
            continue
        needed = host_needed_npz.setdefault(sp.host, set())
        for cid, offs in sp.dominance_tasks:
            if offs:
                corp_name = dom_cell_by_id[cid].corpus_name
                needed.add(BENCHMARK_SPEC_BY_NAME[corp_name].cache_filename)
        for _, offs in sp.ladder_tasks:
            if offs:
                needed.add("sift1m_validated_v1.npz")

    shards_dir = out_dir / "shards"
    logs_dir = out_dir / "logs"
    shards_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    rel_out = out_dir.relative_to(ROOT) if out_dir.is_relative_to(ROOT) else Path("outputs/phase5_dominance")
    remote_data_dir = f"{REMOTE_WORKDIR}/outputs/phase5_dominance/data"
    local_gpus_str = (
        f"{local_gpus[0]},{local_gpus[1]}"
        if local_gpus[:2] == local_gpus[2:]
        else f"{local_gpus[0]},{local_gpus[1]},{local_gpus[2]},{local_gpus[3]}"
    )

    def _resolve_local_npz(npz_name: str) -> Path:
        p5 = data_dir / npz_name
        if p5.exists():
            return p5
        p4 = DEFAULT_MILLION_DATA_DIR / npz_name
        if p4.exists():
            return p4
        raise FileNotFoundError(f"Missing local validated cache {npz_name}")

    def _sync_remote_host(host: str) -> None:
        print(f"[Fleet-Phase5] Syncing code + required corpus caches to {host}:{REMOTE_WORKDIR} ...", flush=True)
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
            link_cmd = (
                f"if [ ! -f {remote_data_dir}/{npz_name} ] && [ -f {REMOTE_P4_MILLION_DATA}/{npz_name} ]; then "
                f"ln -f {REMOTE_P4_MILLION_DATA}/{npz_name} {remote_data_dir}/{npz_name} || "
                f"cp -f {REMOTE_P4_MILLION_DATA}/{npz_name} {remote_data_dir}/{npz_name}; "
                f"fi; "
                f"test -s {remote_data_dir}/{npz_name}"
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
        print(f"[Fleet-Phase5] Host {host} ready.", flush=True)

    remote_hosts = sorted(host_needed_npz.keys())
    if remote_hosts:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(remote_hosts)) as pool:
            list(pool.map(_sync_remote_host, remote_hosts))

    active_shards = selected_shards

    local_procs: dict[int, tuple[FleetShardAssignment, subprocess.Popen[str], Any]] = {}

    for sp in active_shards:
        if sp.host == "local":
            local_shard_json = shards_dir / f"records_shard_{sp.shard_idx:02d}_of_{num_shards:02d}.json"
            local_shard_json.unlink(missing_ok=True)
            local_tmp_root = Path.home() / ".cache" / "emeans_tmp"
            local_tmp_root.mkdir(parents=True, exist_ok=True)
            local_env = os.environ.copy()
            local_env["CUDA_VISIBLE_DEVICES"] = str(sp.gpu_id)
            local_env["TMPDIR"] = str(local_tmp_root)
            local_env["TRITON_CACHE_DIR"] = str(local_tmp_root / f"triton_cache_{sp.session}")
            local_env["PYTHONPATH"] = str(ROOT / "src")
            local_log_path = logs_dir / f"{sp.session}.log"
            lfh = open(local_log_path, "w", encoding="utf-8")
            proc = subprocess.Popen(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_phase5_dominance.py"),
                    "--shard-idx", str(sp.shard_idx),
                    "--num-shards", str(num_shards),
                    "--num-seeds", str(num_seeds),
                    "--seed-start", str(seed_start),
                    "--local-gpus", local_gpus_str,
                    "--data-dir", str(data_dir),
                    "--out-dir", str(out_dir),
                ],
                cwd=str(ROOT),
                env=local_env,
                stdout=lfh,
                stderr=subprocess.STDOUT,
                text=True,
            )
            local_procs[sp.shard_idx] = (sp, proc, lfh)
            print(
                f"[Fleet-Phase5] Launched LOCAL shard {sp.shard_idx:02d}/{num_shards} on foureyes GPU {sp.gpu_id} "
                f"({sp.session})",
                flush=True,
            )
            continue

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
                f"{REMOTE_PYTHON} scripts/run_phase5_dominance.py "
                f"--shard-idx {sp.shard_idx} --num-shards {num_shards} "
                f"--num-seeds {num_seeds} --seed-start {seed_start} "
                f"--local-gpus {local_gpus_str} "
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
            f"[Fleet-Phase5] Launched REMOTE shard {sp.shard_idx:02d}/{num_shards} on {sp.host} GPU {sp.gpu_id} "
            f"({sp.session})",
            flush=True,
        )

    pending_remote = {sp.shard_idx: sp for sp in active_shards if sp.host != "local"}
    t_poll0 = time.perf_counter()

    try:
        while pending_remote or local_procs:
            time.sleep(4.0)
            done_local: list[int] = []
            for sidx, (sp_l, proc_l, lfh_l) in list(local_procs.items()):
                rc = proc_l.poll()
                if rc is not None:
                    lfh_l.close()
                    if rc != 0:
                        tail_txt = (logs_dir / f"{sp_l.session}.log").read_text(encoding="utf-8")[-3000:]
                        raise RuntimeError(f"Local shard {sidx:02d} failed (rc={rc}):\n{tail_txt}")
                    print(
                        f"[Fleet-Phase5] Completed LOCAL shard {sidx:02d} "
                        f"(foureyes GPU {sp_l.gpu_id}) at +{time.perf_counter() - t_poll0:.1f}s",
                        flush=True,
                    )
                    done_local.append(sidx)
            for dl in done_local:
                local_procs.pop(dl, None)

            done_ids: list[int] = []
            for sidx, sp in list(pending_remote.items()):
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
                            f"Remote shard {sidx:02d} ({sp.host} {sp.session}) failed (code={code}):\n{log_res.stdout}"
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
                        f"[Fleet-Phase5] Collected REMOTE shard {sidx:02d} ({sp.host} GPU {sp.gpu_id}) "
                        f"at +{time.perf_counter() - t_poll0:.1f}s",
                        flush=True,
                    )
                    done_ids.append(sidx)
            for did in done_ids:
                pending_remote.pop(did, None)
    finally:
        for _, (_, _, lfh_l) in local_procs.items():
            try:
                lfh_l.close()
            except Exception:
                pass


# ==============================================================================
# CLI Entrypoint
# ==============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 5 Multi-Domain Realistic Dominance & Optimization Ladder Runner")
    p.add_argument("--smoke", action="store_true", help="Run fast 2-seed cold-JIT-checked end-to-end smoke gate")
    p.add_argument("--dispatch-fleet", action="store_true", help="Dispatch 8-GPU clean Blackwell campaign & merge")
    p.add_argument("--merge-only", action="store_true", help="Merge existing shard JSONs, validate, and plot")
    p.add_argument("--shard-idx", type=int, default=None, help="Worker shard index (0..7)")
    p.add_argument("--num-shards", type=int, default=8, help="Total fleet shards (default: 8)")
    p.add_argument("--num-seeds", type=int, default=10, help="Paired seeds per cell (default: 10)")
    p.add_argument("--seed-start", type=int, default=5000, help="Starting RNG seed (default: 5000)")
    p.add_argument(
        "--local-gpus",
        type=str,
        default="0,1,2,3",
        help="4 distinct local foureyes GPU IDs '0,1,2,3' (or 2 IDs '1,3' when splitting --shard-filter '4,5' then '6,7')",
    )
    p.add_argument(
        "--shard-filter",
        type=str,
        default=None,
        help="Optional comma-separated subset of shard indices to dispatch (e.g. '0,1,2,3' and '4,5,6,7')",
    )
    p.add_argument("--data-dir", type=Path, default=DEFAULT_PHASE5_DATA_DIR, help="Validated corpus cache dir")
    p.add_argument("--out-dir", type=Path, default=OUTPUT_DIR, help="Output directory for JSON, MD, and PNGs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir: Path = args.out_dir
    data_dir: Path = args.data_dir

    if args.smoke:
        num_seeds = 2
        seed_start = int(args.seed_start)
        seeds = [seed_start + i for i in range(num_seeds)]
        dom_budgets: tuple[float, ...] = (1.0, 2.0)
        lad_budgets: tuple[float, ...] = (2.0,)
        dom_cells = build_dominance_cells(smoke=True)
        lad_cells = build_ladder_cells(smoke=True)

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        warmup_global_kernels(dev)

        dom_records: list[dict[str, Any]] = []
        fp8_probes: list[dict[str, Any]] = []
        for cell in dom_cells:
            recs, prbs = execute_dominance_cell_seeds(cell, seeds, dom_budgets, data_dir=data_dir, device=dev)
            dom_records.extend(recs)
            fp8_probes.extend(prbs)

        lad_records: list[dict[str, Any]] = []
        for lcell in lad_cells:
            lad_records.extend(execute_ladder_cell_seeds(lcell, seeds, lad_budgets, data_dir=data_dir, device=dev))

        spikes = validate_all_records(
            dom_records=dom_records,
            ladder_records=lad_records,
            fp8_probes=fp8_probes,
            dom_cells=dom_cells,
            ladder_cells=lad_cells,
            expected_seeds=seeds,
            dom_budgets=dom_budgets,
            ladder_budgets=lad_budgets,
        )
        if spikes:
            raise RuntimeError(f"COLD-JIT SPIKES DETECTED IN SMOKE GATE: {spikes!r}")

        comparisons = compute_paired_comparisons(dom_records, dom_cells, dom_budgets)
        payload = {
            "phase": "phase5_dominance_smoke",
            "num_dominance_records": len(dom_records),
            "num_ladder_records": len(lad_records),
            "num_fp8_probes": len(fp8_probes),
            "num_cold_jit_spikes": len(spikes),
            "cold_jit_spikes": spikes,
            "seeds": seeds,
            "dominance_budgets": list(dom_budgets),
            "ladder_budgets": list(lad_budgets),
            "dominance_cells": [asdict(c) for c in dom_cells],
            "ladder_cells": [asdict(c) for c in lad_cells],
            "dominance_records": dom_records,
            "ladder_records": lad_records,
            "fp8_probes": fp8_probes,
            "comparisons": comparisons,
        }
        atomic_write_json(out_dir / "phase5_dominance_results.json", payload)
        write_summary_markdown(
            dom_records=dom_records,
            ladder_records=lad_records,
            fp8_probes=fp8_probes,
            comparisons=comparisons,
            cold_jit_spikes=spikes,
            dom_cells=dom_cells,
            ladder_cells=lad_cells,
            dom_budgets=dom_budgets,
            ladder_budgets=lad_budgets,
            out_path=out_dir / "phase5_dominance_summary.md",
        )
        plots = generate_all_plots(
            dom_records=dom_records,
            ladder_records=lad_records,
            fp8_probes=fp8_probes,
            dom_cells=dom_cells,
            ladder_cells=lad_cells,
            dom_budgets=dom_budgets,
            ladder_budgets=lad_budgets,
            out_dir=out_dir,
        )
        print(
            f"[Phase5-Smoke] SUCCESS: {len(dom_records)} dom + {len(lad_records)} ladder records, "
            f"0 cold-JIT spikes, {len(plots)} PNGs written to {out_dir}",
            flush=True,
        )
        return

    dom_cells = build_dominance_cells(smoke=False)
    lad_cells = build_ladder_cells(smoke=False)
    dom_by_id = {c.cell_id: c for c in dom_cells}
    lad_by_id = {c.cell_id: c for c in lad_cells}
    num_seeds = int(args.num_seeds)
    seed_start = int(args.seed_start)
    all_seeds = [seed_start + i for i in range(num_seeds)]
    dom_budgets = EPOCH_BUDGETS
    lad_budgets = LADDER_BUDGETS

    local_gpus = parse_local_gpus(args.local_gpus)

    if args.shard_idx is not None:
        shards = build_fleet_shards(num_seeds=num_seeds, local_gpus=local_gpus)
        if int(args.num_shards) != len(shards):
            raise ValueError(f"Expected --num-shards {len(shards)} matching build_fleet_shards, got {args.num_shards}")
        sp = shards[int(args.shard_idx)]
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        warmup_global_kernels(dev)

        s_dom_records: list[dict[str, Any]] = []
        s_probes: list[dict[str, Any]] = []
        for cid, offs in sp.dominance_tasks:
            c_seeds = [seed_start + o for o in offs]
            recs, prbs = execute_dominance_cell_seeds(dom_by_id[cid], c_seeds, dom_budgets, data_dir=data_dir, device=dev)
            s_dom_records.extend(recs)
            s_probes.extend(prbs)

        s_lad_records: list[dict[str, Any]] = []
        for lid, offs in sp.ladder_tasks:
            l_seeds = [seed_start + o for o in offs]
            s_lad_records.extend(execute_ladder_cell_seeds(lad_by_id[lid], l_seeds, lad_budgets, data_dir=data_dir, device=dev))

        shard_path = out_dir / "shards" / f"records_shard_{sp.shard_idx:02d}_of_{int(args.num_shards):02d}.json"
        atomic_write_json(
            shard_path,
            {
                "shard_idx": sp.shard_idx,
                "num_shards": int(args.num_shards),
                "host": sp.host,
                "gpu_id": sp.gpu_id,
                "sku_tier": sp.sku_tier,
                "dominance_records": s_dom_records,
                "ladder_records": s_lad_records,
                "fp8_probes": s_probes,
            },
        )
        print(
            f"[Phase5-Shard {sp.shard_idx:02d}] Wrote {len(s_dom_records)} dom + {len(s_lad_records)} ladder -> {shard_path}",
            flush=True,
        )
        return

    shard_filter_set: set[int] | None = None
    if args.shard_filter:
        shard_filter_set = {int(x.strip()) for x in args.shard_filter.split(",") if x.strip()}

    if args.dispatch_fleet:
        dispatch_fleet_10way(
            num_seeds=num_seeds,
            seed_start=seed_start,
            local_gpus=local_gpus,
            out_dir=out_dir,
            data_dir=data_dir,
            shard_filter=shard_filter_set,
        )

    if args.dispatch_fleet or args.merge_only:
        shards_dir = out_dir / "shards"
        shards = build_fleet_shards(num_seeds=num_seeds, local_gpus=local_gpus)
        n_sh = len(shards)
        required_paths = [
            shards_dir / f"records_shard_{sp.shard_idx:02d}_of_{n_sh:02d}.json"
            for sp in shards
            if any(len(o) > 0 for _, o in sp.dominance_tasks) or any(len(o) > 0 for _, o in sp.ladder_tasks)
        ]
        if shard_filter_set is not None and not args.merge_only and not all(p.exists() for p in required_paths):
            done_cnt = sum(1 for p in required_paths if p.exists())
            print(
                f"[Phase5-PartialWave] Collected filtered shards {sorted(shard_filter_set)} "
                f"({done_cnt}/{len(required_paths)} total shards on disk); deferring final merge until remaining wave completes.",
                flush=True,
            )
            return

        merged_dom: list[dict[str, Any]] = []
        merged_lad: list[dict[str, Any]] = []
        merged_prb: list[dict[str, Any]] = []
        for sp in shards:
            has_work = any(len(o) > 0 for _, o in sp.dominance_tasks) or any(len(o) > 0 for _, o in sp.ladder_tasks)
            if not has_work:
                continue
            spath = shards_dir / f"records_shard_{sp.shard_idx:02d}_of_{n_sh:02d}.json"
            if not spath.exists():
                raise FileNotFoundError(f"Missing shard output {spath}")
            with open(spath, "r", encoding="utf-8") as f:
                data = json.load(
                    f,
                    parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"Invalid JSON constant {c}")),
                )
            merged_dom.extend(data["dominance_records"])
            merged_lad.extend(data["ladder_records"])
            merged_prb.extend(data["fp8_probes"])

        spikes = validate_all_records(
            dom_records=merged_dom,
            ladder_records=merged_lad,
            fp8_probes=merged_prb,
            dom_cells=dom_cells,
            ladder_cells=lad_cells,
            expected_seeds=all_seeds,
            dom_budgets=dom_budgets,
            ladder_budgets=lad_budgets,
        )
        if spikes:
            raise RuntimeError(f"COLD-JIT TIMING SPIKES DETECTED: {spikes!r}")

        comparisons = compute_paired_comparisons(merged_dom, dom_cells, dom_budgets)
        payload = {
            "phase": "phase5_dominance",
            "num_dominance_records": len(merged_dom),
            "num_ladder_records": len(merged_lad),
            "num_fp8_probes": len(merged_prb),
            "num_cold_jit_spikes": len(spikes),
            "cold_jit_spikes": spikes,
            "seeds": all_seeds,
            "dominance_budgets": list(dom_budgets),
            "ladder_budgets": list(lad_budgets),
            "selectivities": list(SELECTIVITIES),
            "dominance_cells": [asdict(c) for c in dom_cells],
            "ladder_cells": [asdict(c) for c in lad_cells],
            "dominance_records": merged_dom,
            "ladder_records": merged_lad,
            "fp8_probes": merged_prb,
            "comparisons": comparisons,
        }
        atomic_write_json(out_dir / "phase5_dominance_results.json", payload)
        write_summary_markdown(
            dom_records=merged_dom,
            ladder_records=merged_lad,
            fp8_probes=merged_prb,
            comparisons=comparisons,
            cold_jit_spikes=spikes,
            dom_cells=dom_cells,
            ladder_cells=lad_cells,
            dom_budgets=dom_budgets,
            ladder_budgets=lad_budgets,
            out_path=out_dir / "phase5_dominance_summary.md",
        )
        plots = generate_all_plots(
            dom_records=merged_dom,
            ladder_records=merged_lad,
            fp8_probes=merged_prb,
            dom_cells=dom_cells,
            ladder_cells=lad_cells,
            dom_budgets=dom_budgets,
            ladder_budgets=lad_budgets,
            out_dir=out_dir,
        )
        print(
            f"[Phase5-Complete] Validated {len(merged_dom):,} dom + {len(merged_lad):,} ladder records, "
            f"0 cold-JIT spikes, & {len(plots)} PNGs in {out_dir}",
            flush=True,
        )
        return

    # Local sequential single-GPU fallback
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    warmup_global_kernels(dev)
    loc_dom: list[dict[str, Any]] = []
    loc_prb: list[dict[str, Any]] = []
    for cell in dom_cells:
        r_d, r_p = execute_dominance_cell_seeds(cell, all_seeds, dom_budgets, data_dir=data_dir, device=dev)
        loc_dom.extend(r_d)
        loc_prb.extend(r_p)
    loc_lad: list[dict[str, Any]] = []
    for lcell in lad_cells:
        loc_lad.extend(execute_ladder_cell_seeds(lcell, all_seeds, lad_budgets, data_dir=data_dir, device=dev))
    spikes = validate_all_records(
        loc_dom, loc_lad, loc_prb, dom_cells, lad_cells, all_seeds, dom_budgets, lad_budgets
    )
    comparisons = compute_paired_comparisons(loc_dom, dom_cells, dom_budgets)
    atomic_write_json(
        out_dir / "phase5_dominance_results.json",
        {
            "phase": "phase5_dominance_local",
            "num_dominance_records": len(loc_dom),
            "num_ladder_records": len(loc_lad),
            "num_cold_jit_spikes": len(spikes),
            "seeds": all_seeds,
            "dominance_records": loc_dom,
            "ladder_records": loc_lad,
            "fp8_probes": loc_prb,
            "comparisons": comparisons,
        },
    )
    write_summary_markdown(
        loc_dom, loc_lad, loc_prb, comparisons, spikes, dom_cells, lad_cells, dom_budgets, lad_budgets,
        out_dir / "phase5_dominance_summary.md",
    )
    generate_all_plots(loc_dom, loc_lad, loc_prb, dom_cells, lad_cells, dom_budgets, lad_budgets, out_dir)


if __name__ == "__main__":
    main()
