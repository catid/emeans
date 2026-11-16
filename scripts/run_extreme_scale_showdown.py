#!/usr/bin/env python3
"""Multi-Decade Extreme-Scale Showdown (`N = 20,000 -> 10,000,000`, `K = 512 -> 65,536`).

Evaluates clustering algorithms across **4 orders of magnitude in dataset scale (`N`)** and
**more than 2 orders of magnitude in codebook size (`K`)**, spanning both 1-GPU and 2-GPU NCCL
execution on dedicated 96 GB Blackwell GPUs (`work:0,1`, `acorn:0,1`, `cashew:0,1`):

  - **Tier 1 (Tiny / High-Variance Regime)**:
      * `c1a_tiny_zipf_20k_k512`:         `N = 20,000,     D = 128, K = 512`    (`zipf_gmm`, `N/K = 39.1`)
      * `c1b_tiny_cifar512_50k_k1024`:    `N ~ 50,000,     D = 512, K = 1,024`  (Real CIFAR-10 ResNet-18 `global_512`)
  - **Tier 2 (Medium Regime)**:
      * `c2a_med_cifar_patches_200k_k2048`: `N ~ 200,000,  D = 128, K = 2,048`  (Real CIFAR-10 ResNet-18 `patches_128`)
      * `c2b_med_rare_outliers_200k_k2048`: `N = 200,000,  D = 128, K = 2,048`  (`multiscale_rare_outliers`)
  - **Tier 3 (Million-Scale High-K Regime + Downstream ANN IVF Recall)**:
      * `c3a_mil_sift1m_1m_k16384`:       `N = 1,000,000,  D = 128, K = 16,384` (Real SIFT1M + IVF Recall@{1,10,100})
      * `c3b_mil_glove1m_1m_k16384`:      `N = 1,000,000,  D = 100, K = 16,384` (Real GloVe-1M Unit + Cosine + IVF)
  - **Tier 4 (Extreme Huge-K `65,536` & 10-Million Vector Regime — 1-GPU & 2-GPU NCCL)**:
      * `c4a_ext_hugek_2m_k65536_1gpu`:   `N = 2,000,000,  D = 128, K = 65,536` (1-GPU 600W Server Blackwell)
      * `c4b_ext_10m_k16384_1gpu`:        `N = 10,000,000, D = 128, K = 16,384` (1-GPU 600W Server Blackwell)
      * `c4c_ext_hugek_2m_k65536_2gpu`:   `N = 2,000,000,  D = 128, K = 65,536` (2-GPU NCCL `acorn:0,1` / `cashew:0,1`)
      * `c4d_ext_10m_k16384_2gpu`:        `N = 10,000,000, D = 128, K = 16,384` (2-GPU NCCL `acorn:0,1` / `cashew:0,1`)

Two-Wave Atomic Caching Architecture:
  - **Wave A (`--wave baselines`)**: Benchmarks all competitor baselines (`lloyd_25`,
    `faiss.contrib.e_means` `faiss_emeans`, `emeans`, `smeans`, `emeans_pp`, `super_emeans`,
    `ultra_emeans`, `ultra_emeans_wide` on 1-GPU cells; `multigpu_emeans_bf16` and
    `multigpu_emeans_fp8` on 2-GPU cells) and caches per-shard JSONs atomically.
  - **Wave B (`--wave apex`)**: Runs `ApexEMeans` (`apex_emeans` / `apex_emeans_2gpu`, single
    unified 4-Gear Scale-Adaptive Engine) on the EXACT SAME physical `(host, gpu)` and paired
    seeds, then merges Wave A + Wave B without re-running cached baselines.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import glob
import importlib
import json
import math
import os
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Isolate Triton's disk cache under /home BEFORE importing torch / triton / emeans
if "TRITON_CACHE_DIR" not in os.environ:
    _vis_tag = os.environ.get("CUDA_VISIBLE_DEVICES", "all").replace(",", "_")
    _base_tmp = (
        "/home/catid/emeans_env/tmp"
        if os.path.isdir("/home/catid/emeans_env")
        else os.path.expanduser("~/.cache/emeans_tmp")
    )
    os.makedirs(_base_tmp, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = f"{_base_tmp}/triton_cache_ext_scale_u{os.getuid()}_g{_vis_tag}"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import stats
import torch

from faiss.contrib.e_means import QuantizedEmeans

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from emeans import seeding as emeans_seeding
from emeans.baselines import LloydKMeans
from emeans.datasets import (
    DEFAULT_CACHE_DIR,
    DEFAULT_CIFAR10_ROOT,
    deduplicate_splits,
    load_cifar10_resnet18_embeddings,
    make_synthetic_dataset,
)
from emeans.emeans import EMeans
from emeans.emeans_pp import EMeansPP
from emeans.fast_ops import MultiGPUFastEMeans, blackwell_prepare_centroids, blackwell_triton_assign
from emeans.kernels import HAS_TRITON, assign_points
from emeans.real_corpora import (
    DEFAULT_MILLION_DATA_DIR,
    compute_exact_groundtruth_gpu,
    evaluate_ivf_recall_gpu,
    exact_cosine_distortion_fp64,
    fast_exact_assign_fp64,
    load_glove1m_unit,
    load_sift1m,
)
from emeans.smeans import SMeans
from emeans.super_emeans import SuperEMeans
from emeans.ultra_emeans import UltraEMeans


OUTPUT_DIR = ROOT / "outputs" / "extreme_scale"
DOCS_OUT_PATH = ROOT / "docs" / "extreme_scale_results.md"
REMOTE_WORKDIR = "/home/catid/emeans_runs/extreme-scale"
REMOTE_PYTHON = "/home/catid/emeans_env/.venv/bin/python"
REMOTE_CIFAR_FALLBACK = Path("/home/catid/emeans_env/data/cifar10/cifar-10-batches-py")

EPOCH_BUDGETS: tuple[float, ...] = (1.0, 2.0, 5.0)
IVF_SELECTIVITIES: tuple[float, ...] = (0.001, 0.005, 0.010)

WAVE_A_1GPU_STREAMING_ARMS: tuple[str, ...] = (
    "faiss_emeans",
    "emeans",
    "smeans",
    "emeans_pp",
    "super_emeans",
    "ultra_emeans",
    "ultra_emeans_wide",
)
WAVE_B_1GPU_STREAMING_ARMS: tuple[str, ...] = ("apex_emeans",)

WAVE_A_2GPU_STREAMING_ARMS: tuple[str, ...] = (
    "multigpu_emeans_bf16",
    "multigpu_emeans_fp8",
)
WAVE_B_2GPU_STREAMING_ARMS: tuple[str, ...] = ("apex_emeans_2gpu",)

HUMAN_ARM_NAMES: dict[str, str] = {
    "lloyd_25": "Lloyd k-means (25 iter)",
    "faiss_emeans": "faiss.contrib.e_means",
    "emeans": "e-means (arXiv:2611.09501)",
    "smeans": "s-means (suayptalha)",
    "emeans_pp": "e-means++",
    "super_emeans": "SuperEMeans",
    "ultra_emeans": "UltraEMeans",
    "ultra_emeans_wide": "UltraEMeans (fast)",
    "apex_emeans": "ApexEMeans (4-Gear Auto)",
    "multigpu_emeans_bf16": "MultiGPU e-means (BF16, 2xGPU)",
    "multigpu_emeans_fp8": "MultiGPU e-means (FP8, 2xGPU)",
    "apex_emeans_2gpu": "ApexEMeans (2xGPU NCCL)",
}


# ==============================================================================
# 1. Scale-Ladder Cell & Dedicated 6-GPU Shard Specifications
# ==============================================================================


@dataclass(frozen=True)
class ScaleCellSpec:
    """Experimental cell in the 4-decade scale ladder (`N = 20K -> 10M`, `K = 512 -> 65,536`)."""

    cell_id: str
    tier: int                 # 1 (Tiny), 2 (Medium), 3 (Million), 4 (Extreme 65K / 10M)
    tier_label: str
    corpus_kind: str
    n_train: int
    n_val: int
    n_test: int
    d: int
    k: int
    num_gpus: int             # 1 or 2
    geometry: str             # "euclidean" or "unit_sphere"
    has_ivf_eval: bool
    has_rare_modes: bool
    corpus_seed: int
    sku_tier: str             # "workstation_blackwell" or "server_600w_blackwell"
    num_seeds: int


@dataclass(frozen=True)
class DedicatedShardSpec:
    """Explicit single-SKU shard assignment on `work:0,1`, `acorn:0,1`, and `cashew:0,1`."""

    shard_idx: int
    stage: int                # 1 = 6x parallel 1-GPU shards; 2 = 2x parallel 2-GPU NCCL shards
    host: str
    visible_gpus: str         # "0", "1", or "0,1"
    session: str
    sku_tier: str
    cell_seed_map: dict[str, tuple[int, ...]]


@dataclass
class LoadedScaleCorpus:
    """In-memory / GPU-resident validated dataset split for one scale cell."""

    name: str
    x_train: torch.Tensor                  # (N_train, D) float32 on primary device
    x_val: torch.Tensor                    # (N_val, D) float32 on primary device
    x_test: torch.Tensor                   # (N_test, D) float32 on primary device
    d: int
    geometry: str
    rare_centers: torch.Tensor | None = None
    rare_cutoff_d2: float = 0.0
    gt_ids: np.ndarray | None = None
    gt_sq_dists: np.ndarray | None = None
    shards_2gpu: list[torch.Tensor] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def assert_safe_fleet_target(host: str, session: str) -> None:
    """Enforce workspace safety invariants: block `walnut` and `tmux` session `0`."""
    h_low = host.strip().lower()
    if "walnut" in h_low:
        raise RuntimeError(f"SAFETY VIOLATION: host {host!r} is strictly off-limits!")
    allowed_hosts = {"work", "acorn.stratabaremetal.com", "cashew.stratabaremetal.com"}
    if host not in allowed_hosts:
        raise RuntimeError(
            f"OWNERSHIP VIOLATION: host {host!r} outside extreme-scale-bench dedicated hosts {sorted(allowed_hosts)}"
        )
    if session == "0" or not session.startswith("emeans_ext_s"):
        raise RuntimeError(f"SAFETY VIOLATION: invalid or forbidden tmux session {session!r}")


def build_scale_cells(smoke: bool = False) -> list[ScaleCellSpec]:
    """Return the 10-cell 4-decade scale ladder, or a fast 4-cell smoke ladder covering all paths."""
    if smoke:
        return [
            ScaleCellSpec(
                cell_id="smoke_c1_tiny_zipf",
                tier=1,
                tier_label="Tier 1: Tiny",
                corpus_kind="zipf_gmm",
                n_train=4096,
                n_val=512,
                n_test=512,
                d=64,
                k=256,
                num_gpus=1,
                geometry="euclidean",
                has_ivf_eval=False,
                has_rare_modes=False,
                corpus_seed=20261101,
                sku_tier="smoke_local",
                num_seeds=2,
            ),
            ScaleCellSpec(
                cell_id="smoke_c2_rare_outliers",
                tier=2,
                tier_label="Tier 2: Medium Rare",
                corpus_kind="multiscale_rare_outliers",
                n_train=6144,
                n_val=512,
                n_test=512,
                d=128,
                k=512,  # k=512 > 256 exercises block_greedy_kmeans_plusplus_torch (#111)
                num_gpus=1,
                geometry="euclidean",
                has_ivf_eval=False,
                has_rare_modes=True,
                corpus_seed=20261102,
                sku_tier="smoke_local",
                num_seeds=2,
            ),
            ScaleCellSpec(
                cell_id="smoke_c3_unit_ivf",
                tier=3,
                tier_label="Tier 3: Unit IVF",
                corpus_kind="smoke_glove_unit",
                n_train=6144,
                n_val=512,
                n_test=256,
                d=100,  # non-power-of-two d=100 exercises masked D kernels
                k=512,
                num_gpus=1,
                geometry="unit_sphere",
                has_ivf_eval=True,
                has_rare_modes=False,
                corpus_seed=20261103,
                sku_tier="smoke_local",
                num_seeds=2,
            ),
            ScaleCellSpec(
                cell_id="smoke_c4_ext_2gpu",
                tier=4,
                tier_label="Tier 4: Multi-GPU",
                corpus_kind="extreme_zipf_rare_manifold",
                n_train=8192,
                n_val=512,
                n_test=512,
                d=128,
                k=512,
                num_gpus=2,
                geometry="euclidean",
                has_ivf_eval=False,
                has_rare_modes=True,
                corpus_seed=20261104,
                sku_tier="smoke_local",
                num_seeds=2,
            ),
        ]

    return [
        # Tier 1: Tiny / High-Variance Regime (`work:0,1`)
        ScaleCellSpec(
            cell_id="c1a_tiny_zipf_20k_k512",
            tier=1,
            tier_label="Tier 1: Tiny (20K, K=512)",
            corpus_kind="zipf_gmm",
            n_train=20_000,
            n_val=5_000,
            n_test=10_000,
            d=128,
            k=512,
            num_gpus=1,
            geometry="euclidean",
            has_ivf_eval=False,
            has_rare_modes=False,
            corpus_seed=20261101,
            sku_tier="workstation_blackwell",
            num_seeds=10,
        ),
        ScaleCellSpec(
            cell_id="c1b_tiny_cifar512_50k_k1024",
            tier=1,
            tier_label="Tier 1: CIFAR-10 512D (50K, K=1K)",
            corpus_kind="cifar10_global_512",
            n_train=50_000,
            n_val=5_000,
            n_test=5_000,
            d=512,
            k=1024,
            num_gpus=1,
            geometry="euclidean",
            has_ivf_eval=False,
            has_rare_modes=False,
            corpus_seed=20261101,
            sku_tier="workstation_blackwell",
            num_seeds=10,
        ),
        # Tier 2: Medium Regime (`work:0,1`)
        ScaleCellSpec(
            cell_id="c2a_med_cifar_patches_200k_k2048",
            tier=2,
            tier_label="Tier 2: CIFAR Patches (200K, K=2K)",
            corpus_kind="cifar10_patches_128",
            n_train=200_000,
            n_val=20_000,
            n_test=20_000,
            d=128,
            k=2048,
            num_gpus=1,
            geometry="euclidean",
            has_ivf_eval=False,
            has_rare_modes=False,
            corpus_seed=20261102,
            sku_tier="workstation_blackwell",
            num_seeds=10,
        ),
        ScaleCellSpec(
            cell_id="c2b_med_rare_outliers_200k_k2048",
            tier=2,
            tier_label="Tier 2: Rare Outliers (200K, K=2K)",
            corpus_kind="multiscale_rare_outliers",
            n_train=200_000,
            n_val=20_000,
            n_test=20_000,
            d=128,
            k=2048,
            num_gpus=1,
            geometry="euclidean",
            has_ivf_eval=False,
            has_rare_modes=True,
            corpus_seed=20261102,
            sku_tier="workstation_blackwell",
            num_seeds=10,
        ),
        # Tier 3: Million-Scale High-K Regime (`acorn:0,1` + `cashew:0,1` 600W Server)
        ScaleCellSpec(
            cell_id="c3a_mil_sift1m_1m_k16384",
            tier=3,
            tier_label="Tier 3: SIFT1M (1M, K=16K)",
            corpus_kind="sift1m",
            n_train=1_000_000,
            n_val=89_788,
            n_test=10_000,
            d=128,
            k=16_384,
            num_gpus=1,
            geometry="euclidean",
            has_ivf_eval=True,
            has_rare_modes=False,
            corpus_seed=20261103,
            sku_tier="server_600w_blackwell",
            num_seeds=8,
        ),
        ScaleCellSpec(
            cell_id="c3b_mil_glove1m_1m_k16384",
            tier=3,
            tier_label="Tier 3: GloVe-1M (1M, K=16K)",
            corpus_kind="glove1m_unit",
            n_train=1_000_000,
            n_val=100_000,
            n_test=10_000,
            d=100,
            k=16_384,
            num_gpus=1,
            geometry="unit_sphere",
            has_ivf_eval=True,
            has_rare_modes=False,
            corpus_seed=20261103,
            sku_tier="server_600w_blackwell",
            num_seeds=8,
        ),
        # Tier 4: Extreme Huge-K (`65,536`) & 10-Million Vector Regime — 1-GPU & 2-GPU (`acorn` + `cashew`)
        ScaleCellSpec(
            cell_id="c4a_ext_hugek_2m_k65536_1gpu",
            tier=4,
            tier_label="Tier 4: Huge-K (2M, K=65,536, 1xGPU)",
            corpus_kind="extreme_zipf_rare_manifold",
            n_train=2_000_000,
            n_val=20_000,
            n_test=20_000,
            d=128,
            k=65_536,
            num_gpus=1,
            geometry="euclidean",
            has_ivf_eval=False,
            has_rare_modes=True,
            corpus_seed=20261104,
            sku_tier="server_600w_blackwell",
            num_seeds=4,
        ),
        ScaleCellSpec(
            cell_id="c4b_ext_10m_k16384_1gpu",
            tier=4,
            tier_label="Tier 4: 10M Vectors (10M, K=16K, 1xGPU)",
            corpus_kind="extreme_zipf_rare_manifold",
            n_train=10_000_000,
            n_val=20_000,
            n_test=20_000,
            d=128,
            k=16_384,
            num_gpus=1,
            geometry="euclidean",
            has_ivf_eval=False,
            has_rare_modes=True,
            corpus_seed=20261105,
            sku_tier="server_600w_blackwell",
            num_seeds=4,
        ),
        ScaleCellSpec(
            cell_id="c4c_ext_hugek_2m_k65536_2gpu",
            tier=4,
            tier_label="Tier 4: Huge-K (2M, K=65,536, 2xGPU)",
            corpus_kind="extreme_zipf_rare_manifold",
            n_train=2_000_000,
            n_val=20_000,
            n_test=20_000,
            d=128,
            k=65_536,
            num_gpus=2,
            geometry="euclidean",
            has_ivf_eval=False,
            has_rare_modes=True,
            corpus_seed=20261104,
            sku_tier="server_600w_blackwell",
            num_seeds=4,
        ),
        ScaleCellSpec(
            cell_id="c4d_ext_10m_k16384_2gpu",
            tier=4,
            tier_label="Tier 4: 10M Vectors (10M, K=16K, 2xGPU)",
            corpus_kind="extreme_zipf_rare_manifold",
            n_train=10_000_000,
            n_val=20_000,
            n_test=20_000,
            d=128,
            k=16_384,
            num_gpus=2,
            geometry="euclidean",
            has_ivf_eval=False,
            has_rare_modes=True,
            corpus_seed=20261105,
            sku_tier="server_600w_blackwell",
            num_seeds=4,
        ),
    ]


def build_dedicated_shards(seed_start: int = 5000) -> list[DedicatedShardSpec]:
    """Construct the 8 deterministic shards (6x Stage-1 1-GPU + 2x Stage-2 2-GPU) on our 6 GPUs."""
    s10_a = tuple(seed_start + i for i in range(0, 5))   # 5000..5004 on work:0
    s10_b = tuple(seed_start + i for i in range(5, 10))  # 5005..5009 on work:1

    # 8 seeds for Tier-3 Million-Scale split 4-way across acorn:0,1 + cashew:0,1
    s8_0 = (seed_start + 0, seed_start + 4)
    s8_1 = (seed_start + 1, seed_start + 5)
    s8_2 = (seed_start + 2, seed_start + 6)
    s8_3 = (seed_start + 3, seed_start + 7)

    # 4 seeds for Tier-4 Extreme-Scale (1 seed per Server GPU in 1-GPU mode; 2 seeds per Server host in 2-GPU mode)
    shards = [
        DedicatedShardSpec(
            shard_idx=0,
            stage=1,
            host="work",
            visible_gpus="0",
            session="emeans_ext_s00",
            sku_tier="workstation_blackwell",
            cell_seed_map={
                "c1a_tiny_zipf_20k_k512": s10_a,
                "c1b_tiny_cifar512_50k_k1024": s10_a,
                "c2a_med_cifar_patches_200k_k2048": s10_a,
                "c2b_med_rare_outliers_200k_k2048": s10_a,
            },
        ),
        DedicatedShardSpec(
            shard_idx=1,
            stage=1,
            host="work",
            visible_gpus="1",
            session="emeans_ext_s01",
            sku_tier="workstation_blackwell",
            cell_seed_map={
                "c1a_tiny_zipf_20k_k512": s10_b,
                "c1b_tiny_cifar512_50k_k1024": s10_b,
                "c2a_med_cifar_patches_200k_k2048": s10_b,
                "c2b_med_rare_outliers_200k_k2048": s10_b,
            },
        ),
        DedicatedShardSpec(
            shard_idx=2,
            stage=1,
            host="acorn.stratabaremetal.com",
            visible_gpus="0",
            session="emeans_ext_s02",
            sku_tier="server_600w_blackwell",
            cell_seed_map={
                "c3a_mil_sift1m_1m_k16384": s8_0,
                "c3b_mil_glove1m_1m_k16384": s8_0,
                "c4a_ext_hugek_2m_k65536_1gpu": (seed_start + 0,),
                "c4b_ext_10m_k16384_1gpu": (seed_start + 0,),
            },
        ),
        DedicatedShardSpec(
            shard_idx=3,
            stage=1,
            host="acorn.stratabaremetal.com",
            visible_gpus="1",
            session="emeans_ext_s03",
            sku_tier="server_600w_blackwell",
            cell_seed_map={
                "c3a_mil_sift1m_1m_k16384": s8_1,
                "c3b_mil_glove1m_1m_k16384": s8_1,
                "c4a_ext_hugek_2m_k65536_1gpu": (seed_start + 1,),
                "c4b_ext_10m_k16384_1gpu": (seed_start + 1,),
            },
        ),
        DedicatedShardSpec(
            shard_idx=4,
            stage=1,
            host="cashew.stratabaremetal.com",
            visible_gpus="0",
            session="emeans_ext_s04",
            sku_tier="server_600w_blackwell",
            cell_seed_map={
                "c3a_mil_sift1m_1m_k16384": s8_2,
                "c3b_mil_glove1m_1m_k16384": s8_2,
                "c4a_ext_hugek_2m_k65536_1gpu": (seed_start + 2,),
                "c4b_ext_10m_k16384_1gpu": (seed_start + 2,),
            },
        ),
        DedicatedShardSpec(
            shard_idx=5,
            stage=1,
            host="cashew.stratabaremetal.com",
            visible_gpus="1",
            session="emeans_ext_s05",
            sku_tier="server_600w_blackwell",
            cell_seed_map={
                "c3a_mil_sift1m_1m_k16384": s8_3,
                "c3b_mil_glove1m_1m_k16384": s8_3,
                "c4a_ext_hugek_2m_k65536_1gpu": (seed_start + 3,),
                "c4b_ext_10m_k16384_1gpu": (seed_start + 3,),
            },
        ),
        # Stage 2: 2-GPU NCCL shards on acorn:0,1 and cashew:0,1 (paired on exact same server host!)
        DedicatedShardSpec(
            shard_idx=6,
            stage=2,
            host="acorn.stratabaremetal.com",
            visible_gpus="0,1",
            session="emeans_ext_s06",
            sku_tier="server_600w_blackwell",
            cell_seed_map={
                "c4c_ext_hugek_2m_k65536_2gpu": (seed_start + 0, seed_start + 1),
                "c4d_ext_10m_k16384_2gpu": (seed_start + 0, seed_start + 1),
            },
        ),
        DedicatedShardSpec(
            shard_idx=7,
            stage=2,
            host="cashew.stratabaremetal.com",
            visible_gpus="0,1",
            session="emeans_ext_s07",
            sku_tier="server_600w_blackwell",
            cell_seed_map={
                "c4c_ext_hugek_2m_k65536_2gpu": (seed_start + 2, seed_start + 3),
                "c4d_ext_10m_k16384_2gpu": (seed_start + 2, seed_start + 3),
            },
        ),
    ]
    for sp in shards:
        assert_safe_fleet_target(sp.host, sp.session)
    return shards


# ==============================================================================
# 2. Extreme-Scale Multi-Scale Corpus Generator & Real-World Loaders
# ==============================================================================


@torch.no_grad()
def _verify_zero_split_hash_leakage(
    x_train: torch.Tensor,
    x_val: torch.Tensor,
    x_test: torch.Tensor,
) -> None:
    """Verify in O(N*D) on GPU (~3 ms at 10M x 128) that val/test share 0 exact rows with train or each other."""
    dev = x_train.device
    d = x_train.shape[1]
    g_h = torch.Generator(device="cpu").manual_seed(0x5EED_CAFE)
    weights = (torch.randint(1, 1 << 30, (d,), generator=g_h, dtype=torch.int64) * 2 + 1).to(dev)

    def _hash_rows(arr: torch.Tensor) -> torch.Tensor:
        clean = torch.where(arr == 0.0, torch.zeros_like(arr), arr).contiguous()
        bits = clean.view(torch.int32).to(torch.int64)
        h = torch.zeros((arr.shape[0],), dtype=torch.int64, device=dev)
        chunk = 1_000_000
        for s in range(0, arr.shape[0], chunk):
            e = min(s + chunk, arr.shape[0])
            h[s:e] = (bits[s:e] * weights[None, :]).sum(dim=1)
        return h

    h_tr = _hash_rows(x_train)
    h_va = _hash_rows(x_val)
    h_te = _hash_rows(x_test)

    # Check exact row equality only for any hash collision candidates
    for split_name, x_other, h_other in (("val", x_val, h_va), ("test", x_test, h_te)):
        coll_mask = torch.isin(h_other, h_tr)
        if bool(coll_mask.any().item()):
            cand_rows = x_other[coll_mask]
            # Exact row comparison against matching-hash rows in x_train
            cand_hashes = h_other[coll_mask]
            tr_match_mask = torch.isin(h_tr, cand_hashes)
            tr_cands = x_train[tr_match_mask]
            for r_i in range(cand_rows.shape[0]):
                if bool((tr_cands == cand_rows[r_i : r_i + 1]).all(dim=1).any().item()):
                    raise ValueError(f"DATA LEAKAGE: exact duplicate row between train and {split_name}!")

    coll_vt = torch.isin(h_te, h_va)
    if bool(coll_vt.any().item()):
        te_cands = x_test[coll_vt]
        va_cands = x_val[torch.isin(h_va, h_te[coll_vt])]
        for r_i in range(te_cands.shape[0]):
            if bool((va_cands == te_cands[r_i : r_i + 1]).all(dim=1).any().item()):
                raise ValueError("DATA LEAKAGE: exact duplicate row between val and test!")


@torch.no_grad()
def generate_extreme_multiscale_corpus_gpu(
    n_train: int,
    n_val: int,
    n_test: int,
    d: int,
    k: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Deterministically synthesize a hierarchical Zipf + anisotropic + rare-outlier manifold on GPU.

    Properties:
      1. Full `G = K` latent modes (`95%` hierarchical anisotropic Zipfian bulk + `5%` separated
         sub-threshold rare outlier modes with `p_r = 0.035 / K < 0.1 / K`).
      2. Strictly disjoint chunked RNG seed namespaces for `train`, `val`, and `test` (`~50 ms` for
         `N = 10,000,000 x 128` on Blackwell GPU, 100% bit-identical across 1-GPU and 2-GPU runs).
      3. Verified zero `train`/`val`/`test` duplicate vector leakage.
    """
    if n_train < 1 or n_val < 1 or n_test < 1 or d < 1 or k < 2:
        raise ValueError(f"Invalid extreme corpus dimensions: n=({n_train},{n_val},{n_test}), d={d}, k={k}")

    g_cpu = torch.Generator(device="cpu").manual_seed(int(seed))
    n_rare = max(4, k // 20)
    n_bulk = max(1, k - n_rare)

    j_idx = torch.arange(1, d + 1, dtype=torch.float64)
    coord_var = float(d) * (j_idx ** -0.8) / (j_idx ** -0.8).sum()
    coord_std = coord_var.sqrt().to(dtype=torch.float32).to(device)

    n_macro = max(8, int(round(math.sqrt(float(n_bulk)))))
    macro_c = torch.randn((n_macro, d), generator=g_cpu, dtype=torch.float32) * 1.4
    macro_assign = torch.randint(0, n_macro, (n_bulk,), generator=g_cpu)
    bulk_offsets = 0.55 * torch.randn((n_bulk, d), generator=g_cpu, dtype=torch.float32)
    bulk_c = (macro_c[macro_assign] + bulk_offsets).to(device) * coord_std[None, :]

    raw_dir = torch.randn((n_rare, d), generator=g_cpu, dtype=torch.float32)
    raw_dir = raw_dir / raw_dir.norm(dim=1, keepdim=True).clamp_min(1e-12)
    rare_radius = 3.2 * math.sqrt(float(d) * 1.8)
    rare_c = (raw_dir * rare_radius).to(device).contiguous()

    centers = torch.cat([bulk_c, rare_c], dim=0).contiguous()

    ranks = torch.arange(1, n_bulk + 1, dtype=torch.float64)
    bulk_w = (ranks + 8.0) ** -0.85
    rare_total = min(0.15, 0.035 * float(n_rare) / float(k))
    bulk_p = (1.0 - rare_total) * (bulk_w / bulk_w.sum())
    rare_p = torch.full((n_rare,), rare_total / float(n_rare), dtype=torch.float64)
    probs = torch.cat([bulk_p, rare_p], dim=0).to(device)
    probs = probs / probs.sum()
    cdf = torch.cumsum(probs, dim=0)
    cdf[-1] = 1.0

    sigma_bulk = 0.28
    sigma_rare = 0.12
    sigmas = torch.cat(
        [
            torch.full((n_bulk,), sigma_bulk, device=device, dtype=torch.float32),
            torch.full((n_rare,), sigma_rare, device=device, dtype=torch.float32),
        ],
        dim=0,
    )

    def _draw_split(n_rows: int, stream_offset: int) -> torch.Tensor:
        out = torch.empty((n_rows, d), device=device, dtype=torch.float32)
        chunk = 500_000
        for cid, s in enumerate(range(0, n_rows, chunk)):
            e = min(s + chunk, n_rows)
            m = e - s
            g_dev = torch.Generator(device=device).manual_seed(
                (int(seed) + stream_offset + cid * 10_007) % (2**62)
            )
            u = torch.rand((m,), generator=g_dev, device=device, dtype=torch.float64)
            comp = torch.searchsorted(cdf, u, right=True).clamp_(0, k - 1)
            z = torch.randn((m, d), generator=g_dev, device=device, dtype=torch.float32) * coord_std[None, :]
            out[s:e] = centers[comp] + sigmas[comp, None] * z
        return out.contiguous()

    x_tr = _draw_split(n_train, 1_000_000)
    x_va = _draw_split(n_val, 5_000_000)
    x_te = _draw_split(n_test, 9_000_000)

    _verify_zero_split_hash_leakage(x_tr, x_va, x_te)
    rare_cutoff_d2 = float(d) * (0.35**2)
    return x_tr, x_va, x_te, rare_c, rare_cutoff_d2


def _resolve_cifar_dir() -> Path:
    if DEFAULT_CIFAR10_ROOT.exists():
        return DEFAULT_CIFAR10_ROOT
    if REMOTE_CIFAR_FALLBACK.exists():
        return REMOTE_CIFAR_FALLBACK
    return DEFAULT_CIFAR10_ROOT


@torch.no_grad()
def load_scale_cell_data(
    cell: ScaleCellSpec,
    data_dir: Path,
    primary_device: torch.device,
    multi_devices: list[torch.device] | None = None,
) -> LoadedScaleCorpus:
    """Load or deterministically construct strictly isolated `(train, val, test)` tensors for `cell`."""
    local_cache = ROOT / "outputs" / "phase1" / "cache"
    cifar_cache_dir = local_cache if local_cache.exists() else DEFAULT_CACHE_DIR

    rare_c: torch.Tensor | None = None
    rare_cutoff = float(cell.d) * (0.35**2)
    gt_ids: np.ndarray | None = None
    gt_d2: np.ndarray | None = None
    meta: dict[str, Any] = {"corpus_kind": cell.corpus_kind}

    if cell.corpus_kind in ("zipf_gmm", "multiscale_rare_outliers"):
        ds = make_synthetic_dataset(
            kind=cell.corpus_kind,
            n_train=cell.n_train,
            n_val=cell.n_val,
            n_test=cell.n_test,
            d=cell.d,
            k=cell.k,
            seed=cell.corpus_seed,
        )
        tr_np, va_np, te_np = deduplicate_splits(ds.train, ds.val, ds.test)
        x_tr = torch.from_numpy(tr_np).to(device=primary_device, dtype=torch.float32).contiguous()
        x_va = torch.from_numpy(va_np).to(device=primary_device, dtype=torch.float32).contiguous()
        x_te = torch.from_numpy(te_np).to(device=primary_device, dtype=torch.float32).contiguous()
        if ds.true_centers is not None and ds.rare_mode_indices is not None and len(ds.rare_mode_indices) > 0:
            rare_c = torch.from_numpy(ds.true_centers[ds.rare_mode_indices]).to(
                device=primary_device, dtype=torch.float32
            ).contiguous()

    elif cell.corpus_kind in ("cifar10_global_512", "cifar10_patches_128"):
        mode = "global_512" if cell.corpus_kind == "cifar10_global_512" else "patches_128"
        ds = load_cifar10_resnet18_embeddings(
            mode=mode,
            k_recommended=cell.k,
            cifar_dir=_resolve_cifar_dir(),
            cache_dir=cifar_cache_dir,
            device=str(primary_device),
        )
        x_tr = torch.from_numpy(ds.train).to(device=primary_device, dtype=torch.float32).contiguous()
        x_va = torch.from_numpy(ds.val).to(device=primary_device, dtype=torch.float32).contiguous()
        x_te = torch.from_numpy(ds.test).to(device=primary_device, dtype=torch.float32).contiguous()

    elif cell.corpus_kind == "sift1m":
        corp = load_sift1m(data_dir=data_dir, device=primary_device)
        x_tr = torch.from_numpy(corp.base).to(device=primary_device, dtype=torch.float32).contiguous()
        x_va = torch.from_numpy(corp.val).to(device=primary_device, dtype=torch.float32).contiguous()
        x_te = torch.from_numpy(corp.query).to(device=primary_device, dtype=torch.float32).contiguous()
        gt_ids = corp.groundtruth_ids
        gt_d2 = corp.groundtruth_sq_dists

    elif cell.corpus_kind == "glove1m_unit":
        corp = load_glove1m_unit(data_dir=data_dir, device=primary_device)
        x_tr = torch.from_numpy(corp.base).to(device=primary_device, dtype=torch.float32).contiguous()
        x_va = torch.from_numpy(corp.val).to(device=primary_device, dtype=torch.float32).contiguous()
        x_te = torch.from_numpy(corp.query).to(device=primary_device, dtype=torch.float32).contiguous()
        gt_ids = corp.groundtruth_ids
        gt_d2 = corp.groundtruth_sq_dists

    elif cell.corpus_kind == "smoke_glove_unit":
        rng = np.random.default_rng(cell.corpus_seed)
        tot = cell.n_train + cell.n_val + cell.n_test
        raw = rng.standard_normal((tot, cell.d))
        unit = (raw / np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)).astype(np.float32)
        tr_np, va_np, te_np = deduplicate_splits(
            unit[: cell.n_train],
            unit[cell.n_train : cell.n_train + cell.n_val],
            unit[cell.n_train + cell.n_val :],
        )
        gt_ids, gt_d2 = compute_exact_groundtruth_gpu(tr_np, te_np, top_k=100, device=primary_device)
        x_tr = torch.from_numpy(tr_np).to(device=primary_device, dtype=torch.float32).contiguous()
        x_va = torch.from_numpy(va_np).to(device=primary_device, dtype=torch.float32).contiguous()
        x_te = torch.from_numpy(te_np).to(device=primary_device, dtype=torch.float32).contiguous()

    elif cell.corpus_kind == "extreme_zipf_rare_manifold":
        x_tr, x_va, x_te, rare_c, rare_cutoff = generate_extreme_multiscale_corpus_gpu(
            n_train=cell.n_train,
            n_val=cell.n_val,
            n_test=cell.n_test,
            d=cell.d,
            k=cell.k,
            seed=cell.corpus_seed,
            device=primary_device,
        )
    else:
        raise ValueError(f"Unknown corpus_kind={cell.corpus_kind!r}")

    if x_tr.shape[1] != cell.d or x_va.shape[1] != cell.d or x_te.shape[1] != cell.d:
        raise ValueError(f"Dimension mismatch in {cell.cell_id}: expected d={cell.d}")
    for split_lbl, ten in (("train", x_tr), ("val", x_va), ("test", x_te)):
        if not bool(torch.isfinite(ten).all().item()):
            raise ValueError(f"Non-finite entries in {cell.cell_id}.{split_lbl}")

    shards_2gpu: list[torch.Tensor] | None = None
    if cell.num_gpus == 2:
        devs = multi_devices if multi_devices else [primary_device]
        w_cnt = len(devs)
        shards_2gpu = [
            x_tr[w::w_cnt].to(device=dv, dtype=torch.float32).contiguous()
            for w, dv in enumerate(devs)
        ]
        for dv in devs:
            if dv.type == "cuda":
                torch.cuda.synchronize(dv)

    return LoadedScaleCorpus(
        name=cell.corpus_kind,
        x_train=x_tr,
        x_val=x_va,
        x_test=x_te,
        d=cell.d,
        geometry=cell.geometry,
        rare_centers=rare_c,
        rare_cutoff_d2=rare_cutoff,
        gt_ids=gt_ids,
        gt_sq_dists=gt_d2,
        shards_2gpu=shards_2gpu,
        metadata=meta,
    )


# ==============================================================================
# 3. Dynamic `ApexEMeans` Loader, Arm Instantiation & Full-Shape GPU Warmup
# ==============================================================================


_APEX_CLASS_OVERRIDE: Any = None


def set_apex_class_for_testing(cls_or_none: Any) -> None:
    """Allow unit tests to inject a contract-faithful `ApexEMeans` stub before Wave B lands."""
    global _APEX_CLASS_OVERRIDE
    _APEX_CLASS_OVERRIDE = cls_or_none


def resolve_apex_class(required: bool = False) -> Any:
    """Lazily resolve `emeans.apex_emeans.ApexEMeans` (or injected test class)."""
    if _APEX_CLASS_OVERRIDE is not None:
        return _APEX_CLASS_OVERRIDE
    try:
        mod = importlib.import_module("emeans.apex_emeans")
        return getattr(mod, "ApexEMeans")
    except Exception as exc:
        if required:
            raise RuntimeError(
                "ApexEMeans (`src/emeans/apex_emeans.py`) is required for Wave B (`--wave apex`) "
                f"but could not be imported: {exc}"
            ) from exc
        return None


def get_wave_run_specs(
    cell: ScaleCellSpec,
    budgets: tuple[float, ...],
    wave: str,
    apex_available: bool,
) -> list[tuple[str, float]]:
    """Return ordered `(arm_name, nepochs)` specs for `(cell, wave)`."""
    specs: list[tuple[str, float]] = []
    include_a = wave in ("baselines", "all")
    include_b = wave == "apex" or (wave == "all" and apex_available)

    if cell.num_gpus == 1:
        if include_a:
            specs.append(("lloyd_25", 25.0))
            for b_t in budgets:
                for arm in WAVE_A_1GPU_STREAMING_ARMS:
                    specs.append((arm, float(b_t)))
        if include_b:
            for b_t in budgets:
                for arm in WAVE_B_1GPU_STREAMING_ARMS:
                    specs.append((arm, float(b_t)))
    else:
        if include_a:
            for b_t in budgets:
                for arm in WAVE_A_2GPU_STREAMING_ARMS:
                    specs.append((arm, float(b_t)))
        if include_b:
            for b_t in budgets:
                for arm in WAVE_B_2GPU_STREAMING_ARMS:
                    specs.append((arm, float(b_t)))
    return specs


def instantiate_scale_arm(
    arm: str,
    d: int,
    k: int,
    nepochs: float,
    seed: int,
    multi_devices: list[torch.device] | None = None,
) -> tuple[str, Any]:
    """Construct `(api_kind, solver)` for `arm` with identical `(d, k, nepochs, seed)` contract."""
    if arm == "lloyd_25":
        return "emeans", LloydKMeans(d=d, k=k, niter=int(round(nepochs)), init="random", seed=seed)
    if arm == "faiss_emeans":
        ne_arg: int | float = int(nepochs) if float(nepochs).is_integer() else float(nepochs)
        return "faiss", QuantizedEmeans(d, k, nepochs=ne_arg, seed=seed, capture_assignments=False)
    if arm == "emeans":
        return "emeans", EMeans(d=d, k=k, nepochs=nepochs, seed=seed)
    if arm == "smeans":
        return "emeans", SMeans(d=d, k=k, nepochs=nepochs, exact_ref=False, seed=seed)
    if arm == "emeans_pp":
        return "emeans", EMeansPP(d=d, k=k, nepochs=nepochs, seed=seed)
    if arm == "super_emeans":
        return "emeans", SuperEMeans(d=d, k=k, nepochs=nepochs, wide_step_mult=1, seed=seed)
    if arm == "ultra_emeans":
        return "emeans", UltraEMeans(d=d, k=k, nepochs=nepochs, wide_step_mult=1, seed=seed)
    if arm == "ultra_emeans_wide":
        return "emeans", UltraEMeans(
            d=d, k=k, nepochs=nepochs, wide_step_mult=2, wide_from_frac=0.25, seed=seed
        )
    if arm == "apex_emeans":
        apex_cls = resolve_apex_class(required=True)
        return "apex_1gpu", apex_cls(d=d, k=k, nepochs=nepochs, gear_mode="auto", precision="auto", seed=seed)
    if arm == "multigpu_emeans_bf16":
        return "multigpu_fast", MultiGPUFastEMeans(
            d=d,
            k=k,
            devices=multi_devices,
            nepochs=nepochs,
            precision="bf16",
            streaming_mode="block_permute",
            seed=seed,
        )
    if arm == "multigpu_emeans_fp8":
        return "multigpu_fast", MultiGPUFastEMeans(
            d=d,
            k=k,
            devices=multi_devices,
            nepochs=nepochs,
            precision="fp8",
            streaming_mode="block_permute",
            seed=seed,
        )
    if arm == "apex_emeans_2gpu":
        apex_cls = resolve_apex_class(required=True)
        return "apex_2gpu", apex_cls(
            d=d,
            k=k,
            nepochs=nepochs,
            gear_mode="auto",
            precision="auto",
            devices=multi_devices,
            seed=seed,
        )
    raise ValueError(f"Unknown arm {arm!r}")


@torch.no_grad()
def fit_arm_timed(
    api_kind: str,
    solver: Any,
    corpus: LoadedScaleCorpus,
    sync_devices: list[torch.device],
) -> tuple[torch.Tensor, float, float, float, float]:
    """Execute `.fit(...)` / `.train(...)` bracketed by `torch.cuda.synchronize()` across all active GPUs."""
    for dv in sync_devices:
        if dv.type == "cuda":
            torch.cuda.synchronize(dv)
    t0 = time.perf_counter()

    used_reported_subtimer = False
    if api_kind == "faiss":
        c_out, _ = solver.train(corpus.x_train)
    elif api_kind in ("emeans", "apex_1gpu"):
        c_out, _ = solver.fit(corpus.x_train, capture_assignments=False)
    elif api_kind == "multigpu_fast":
        assert corpus.shards_2gpu is not None
        c_out = solver.fit(corpus.shards_2gpu)
    elif api_kind == "apex_2gpu":
        assert corpus.shards_2gpu is not None
        try:
            res = solver.fit(corpus.shards_2gpu, capture_assignments=False)
        except (TypeError, AttributeError, ValueError):
            # Fallback if ApexEMeans expects full x_train tensor and scatters internally
            res = solver.fit(corpus.x_train, capture_assignments=False)
            used_reported_subtimer = True
        c_out = res[0] if isinstance(res, tuple) else res
    else:
        raise ValueError(f"Unsupported api_kind={api_kind!r}")

    for dv in sync_devices:
        if dv.type == "cuda":
            torch.cuda.synchronize(dv)
    wall_sec = time.perf_counter() - t0

    if not isinstance(c_out, torch.Tensor):
        c_out = torch.as_tensor(c_out, device=corpus.x_train.device, dtype=torch.float32)
    else:
        c_out = c_out.to(device=corpus.x_train.device, dtype=torch.float32).contiguous()

    rep_init = float(getattr(solver, "init_time_sec", 0.0))
    rep_train = float(getattr(solver, "train_time_sec", max(0.0, wall_sec - rep_init)))
    if used_reported_subtimer and (rep_init + rep_train) > 0.0:
        init_sec = rep_init
        train_sec = rep_train
        total_sec = init_sec + train_sec
    else:
        init_sec = min(max(0.0, rep_init), wall_sec)
        train_sec = max(0.0, wall_sec - init_sec)
        total_sec = init_sec + train_sec

    return c_out, init_sec, train_sec, total_sec, wall_sec


@torch.no_grad()
def warmup_global_kernels(device: torch.device) -> None:
    """Pre-warm global seed-dependent Triton coreset tournament buckets (`#111`) outside timers."""
    if device.type != "cuda" or not HAS_TRITON:
        return
    with torch.cuda.device(device):
        local_kpp_fn = getattr(emeans_seeding, "_local_greedy_kpp_kernel", None)
        if local_kpp_fn is not None:
            pair_d2 = torch.zeros((1024, 1024), device=device, dtype=torch.float32)
            pool_d2 = torch.ones((1024,), device=device, dtype=torch.float32)
            w_p = torch.full((1024,), 1.0 / 1024.0, device=device, dtype=torch.float32)
            out_idx = torch.empty((32,), device=device, dtype=torch.int64)
            for block_p, n_warps, p_vals in [(256, 4, (128, 129)), (1024, 8, (512, 513))]:
                for p_v in p_vals:
                    for ns_v in (1, 16, 17):
                        for rs_v in (1, 16, 17):
                            local_kpp_fn[(1,)](
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
        try:
            apex_mod = importlib.import_module("emeans.apex_emeans")
            warm_fn = getattr(apex_mod, "warmup_apex_kernels", None)
            if callable(warm_fn):
                warm_fn(device)
        except Exception:
            pass
    torch.cuda.synchronize(device)


@torch.no_grad()
def warmup_scale_cell(
    cell: ScaleCellSpec,
    corpus: LoadedScaleCorpus,
    run_specs: list[tuple[str, float]],
    primary_device: torch.device,
    sync_devices: list[torch.device],
) -> None:
    """Compile every Triton / cuBLAS / CUDAGraph / NCCL specialization on exact cell shapes outside timers."""
    if primary_device.type != "cuda":
        return

    # 1. Warm `_fused_assign_kernel` on coreset `m_sub` for `%16==0` and `%16!=0` (#111)
    m_sub = min(int(corpus.x_train.shape[0]), max(8192, 16 * cell.k, cell.k * 2))
    x_m = corpus.x_train[:m_sub].float().contiguous()
    for dt in (torch.float32, torch.bfloat16):
        for k_cand in (1, 16, 17, 64, 65, 128, 129):
            if k_cand <= m_sub:
                _ = assign_points(x_m, x_m[:k_cand], compute_dtype=dt)

    # 2. Force-compile dead-cluster split / Huygens kernels on `cell.d` & `cell.k` even on uniform spheres
    if cell.k >= 4 and int(corpus.x_train.shape[0]) >= cell.k:
        n_mini = min(int(corpus.x_train.shape[0]), max(4096, cell.k * 2))
        c_dead = corpus.x_train[: cell.k].float().clone().contiguous()
        c_dead[cell.k // 2 :] = 1e5
        if any(a == "smeans" for a, _ in run_specs):
            _ = SMeans(d=cell.d, k=cell.k, nepochs=1.0, exact_ref=False, seed=999_801).fit(
                corpus.x_train[:n_mini], init_centroids=c_dead, capture_assignments=False
            )
        if any(a in ("ultra_emeans", "ultra_emeans_wide") for a, _ in run_specs):
            _ = UltraEMeans(d=cell.d, k=cell.k, nepochs=1.0, seed=999_802).fit(
                corpus.x_train[:n_mini], init_centroids=c_dead, capture_assignments=False
            )
        if any(a == "apex_emeans" for a, _ in run_specs):
            apex_cls = resolve_apex_class(required=False)
            if apex_cls is not None:
                try:
                    _ = apex_cls(d=cell.d, k=cell.k, nepochs=1.0, seed=999_803).fit(
                        corpus.x_train[:n_mini], init_centroids=c_dead, capture_assignments=False
                    )
                except Exception:
                    pass

    # 3. Full-shape untimed fit of EVERY `(arm, budget)` in `run_specs` (`niter=2` for `lloyd_25`)
    c_warm: torch.Tensor | None = None
    for idx_s, (arm_name, nepochs) in enumerate(run_specs):
        warm_ep = 2.0 if arm_name == "lloyd_25" else float(nepochs)
        api_kind, solver = instantiate_scale_arm(
            arm_name,
            d=cell.d,
            k=cell.k,
            nepochs=warm_ep,
            seed=999_900 + idx_s,
            multi_devices=sync_devices if cell.num_gpus == 2 else None,
        )
        c_warm, _, _, _, _ = fit_arm_timed(api_kind, solver, corpus, sync_devices)

    # 4. Warm exact FP64 evaluators, full-N count evaluator, and CSR IVF Recall evaluator
    if c_warm is not None:
        _ = _evaluate_fitted_centroids(cell, corpus, c_warm, primary_device)

    for dv in sync_devices:
        if dv.type == "cuda":
            torch.cuda.synchronize(dv)


# ==============================================================================
# 4. Untimed Exact FP64 Distortion, Imbalance, Rare-Mode & IVF Recall Evaluator
# ==============================================================================


@torch.no_grad()
def _evaluate_fitted_centroids(
    cell: ScaleCellSpec,
    corpus: LoadedScaleCorpus,
    centroids: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    """Compute exact IEEE-754 FP64 Test/Val/Train MSE, Imbalance, Empty Clusters, Rare Survival & IVF Recall."""
    n_tr = int(corpus.x_train.shape[0])
    val_mse, _ = fast_exact_assign_fp64(corpus.x_val, centroids, device=device)
    test_mse, _ = fast_exact_assign_fp64(corpus.x_test, centroids, device=device)

    if n_tr <= 1_000_000:
        train_mse, db_labels = fast_exact_assign_fp64(corpus.x_train, centroids, device=device)
        train_eval_rows = n_tr
    else:
        # Exact IEEE-754 FP64 MSE on deterministic 250K strided probe + full-N centered assignment for exact counts
        stride = max(1, n_tr // 250_000)
        x_tr_probe = corpus.x_train[::stride].contiguous()
        train_mse, _ = fast_exact_assign_fp64(x_tr_probe, centroids, device=device)
        train_eval_rows = int(x_tr_probe.shape[0])

        origin = corpus.x_train[: min(n_tr, 8192)].double().mean(dim=0, keepdim=True).float()
        c_c = (centroids.float() - origin).contiguous()
        db_labels = torch.empty((n_tr,), dtype=torch.int64, device=device)
        comp_dt = torch.bfloat16 if device.type == "cuda" else torch.float32
        if device.type == "cuda" and HAS_TRITON:
            c_rep, c_norm = blackwell_prepare_centroids(c_c, comp_dt)
            chunk = 500_000
            for s in range(0, n_tr, chunk):
                e = min(s + chunk, n_tr)
                xb_c = (corpus.x_train[s:e] - origin).contiguous()
                _, db_labels[s:e] = blackwell_triton_assign(xb_c, c_rep, c_norm)
        else:
            _, db_labels = assign_points(corpus.x_train - origin, c_c, compute_dtype=comp_dt)

    if cell.geometry == "unit_sphere":
        val_cos_dist = exact_cosine_distortion_fp64(corpus.x_val, centroids, device=device)
        test_cos_dist = exact_cosine_distortion_fp64(corpus.x_test, centroids, device=device)
    else:
        val_cos_dist = 0.0
        test_cos_dist = 0.0

    if cell.has_ivf_eval and corpus.gt_ids is not None and corpus.gt_sq_dists is not None:
        ivf_res = evaluate_ivf_recall_gpu(
            x_base=corpus.x_train,
            centroids=centroids,
            x_query=corpus.x_test,
            groundtruth_ids=corpus.gt_ids,
            groundtruth_sq_dists=corpus.gt_sq_dists,
            selectivities=IVF_SELECTIVITIES,
            db_labels=db_labels,
            device=device,
        )
        imbalance = float(ivf_res.imbalance_factor)
        empty_cnt = int(ivf_res.empty_clusters)
        m01 = ivf_res.by_selectivity["0.1%"]
        m05 = ivf_res.by_selectivity["0.5%"]
        m10 = ivf_res.by_selectivity["1%"]
        r10_01 = float(m01.recall_at_10)
        r10_05 = float(m05.recall_at_10)
        r10_10 = float(m10.recall_at_10)
        r1_01 = float(m01.recall_at_1)
    else:
        counts = torch.bincount(db_labels, minlength=cell.k).double()
        imbalance = float(cell.k) * float(counts.square().sum().item()) / (float(n_tr) ** 2)
        empty_cnt = int((counts == 0).sum().item())
        r10_01 = 0.0
        r10_05 = 0.0
        r10_10 = 0.0
        r1_01 = 0.0

    if cell.has_rare_modes and corpus.rare_centers is not None:
        # Exact shift-invariant FP64 squared distance from each true rare mode center to nearest centroid
        rc64 = corpus.rare_centers.double()
        c64 = centroids.double()
        origin_r = rc64.mean(dim=0, keepdim=True)
        rc_c = rc64 - origin_r
        cc_c = c64 - origin_r
        cc_norm = cc_c.square().sum(dim=1)
        min_d2_rare = torch.empty((rc64.shape[0],), dtype=torch.float64, device=device)
        for r0 in range(0, rc64.shape[0], 512):
            r1 = min(r0 + 512, rc64.shape[0])
            sub_r = rc_c[r0:r1]
            best_approx = torch.full((r1 - r0,), float("inf"), dtype=torch.float64, device=device)
            best_idx = torch.zeros((r1 - r0,), dtype=torch.int64, device=device)
            for k0 in range(0, c64.shape[0], 8192):
                k1 = min(k0 + 8192, c64.shape[0])
                d2_blk = (
                    sub_r.square().sum(dim=1, keepdim=True)
                    + cc_norm[None, k0:k1]
                    - 2.0 * (sub_r @ cc_c[k0:k1].t())
                )
                b_vals, b_pos = d2_blk.min(dim=1)
                better = b_vals < best_approx
                best_approx = torch.where(better, b_vals, best_approx)
                best_idx = torch.where(better, b_pos + k0, best_idx)
            min_d2_rare[r0:r1] = (rc64[r0:r1] - c64[best_idx]).square().sum(dim=1)
        rare_preserved = float((min_d2_rare <= corpus.rare_cutoff_d2).double().mean().item())
    else:
        rare_preserved = 0.0

    return {
        "train_mse": float(train_mse),
        "train_eval_rows": int(train_eval_rows),
        "val_mse": float(val_mse),
        "test_mse": float(test_mse),
        "val_cos_dist": float(val_cos_dist),
        "test_cos_dist": float(test_cos_dist),
        "imbalance_factor": float(imbalance),
        "empty_clusters": int(empty_cnt),
        "rare_preserved_frac": float(rare_preserved),
        "recall1_01pct": float(r1_01),
        "recall10_01pct": float(r10_01),
        "recall10_05pct": float(r10_05),
        "recall10_10pct": float(r10_10),
    }


# ==============================================================================
# 5. Per-Cell / Per-Shard Execution Engine
# ==============================================================================


def detect_gpu_sku_info(device: torch.device) -> tuple[str, float]:
    """Return `(gpu_sku_label, max_power_watts)` for single-SKU de-confounding checks."""
    if device.type != "cuda":
        return "CPU", 0.0
    name = torch.cuda.get_device_name(device)
    vis0 = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
    power_w = 0.0
    try:
        cmd = [
            "nvidia-smi",
            "-i",
            vis0 if vis0.isdigit() else "0",
            "--query-gpu=power.max_limit",
            "--format=csv,noheader,nounits",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=3, check=False)
        if res.returncode == 0 and res.stdout.strip():
            power_w = float(res.stdout.strip().splitlines()[0].strip())
    except Exception:
        power_w = 0.0
    sku_label = f"{name} ({int(round(power_w))}W)" if power_w > 0 else name
    return sku_label, power_w


@torch.no_grad()
def execute_scale_cell(
    cell: ScaleCellSpec,
    seeds: list[int],
    budgets: tuple[float, ...],
    wave: str,
    data_dir: Path,
    smoke: bool = False,
) -> list[dict[str, Any]]:
    """Run all requested `(arm, budget)` pairs across `seeds` for `cell` with full warmup."""
    if not seeds:
        return []

    apex_avail = resolve_apex_class(required=(wave == "apex")) is not None
    run_specs = get_wave_run_specs(cell, budgets, wave=wave, apex_available=apex_avail)
    if not run_specs:
        return []

    if torch.cuda.is_available():
        primary_dev = torch.device("cuda:0")
        if cell.num_gpus == 2:
            n_vis = torch.cuda.device_count()
            if n_vis < 2 and not smoke:
                raise RuntimeError(
                    f"2-GPU cell {cell.cell_id} requires >= 2 visible CUDA devices, found {n_vis}"
                )
            sync_devs = (
                [torch.device("cuda:0"), torch.device("cuda:1")]
                if n_vis >= 2
                else [torch.device("cuda:0")]
            )
        else:
            sync_devs = [primary_dev]
    else:
        if cell.num_gpus == 2 and not smoke:
            raise RuntimeError(f"2-GPU cell {cell.cell_id} requires CUDA devices")
        primary_dev = torch.device("cpu")
        sync_devs = [primary_dev]

    for dv in sync_devs:
        warmup_global_kernels(dv)

    host = socket.gethostname()
    vis_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    gpu_sku, power_w = detect_gpu_sku_info(primary_dev)

    t_load0 = time.perf_counter()
    corpus = load_scale_cell_data(
        cell,
        data_dir=data_dir,
        primary_device=primary_dev,
        multi_devices=sync_devs if cell.num_gpus == 2 else None,
    )
    print(
        f"[ExtScale] Loaded {cell.cell_id} (N={corpus.x_train.shape[0]:,}, D={cell.d}, K={cell.k:,}, "
        f"gpus={len(sync_devs)}) on {host} GPU={vis_gpu} [{gpu_sku}] in {time.perf_counter()-t_load0:.2f}s",
        flush=True,
    )

    t_warm0 = time.perf_counter()
    warmup_scale_cell(
        cell=cell,
        corpus=corpus,
        run_specs=run_specs,
        primary_device=primary_dev,
        sync_devices=sync_devs,
    )
    print(
        f"[ExtScale] Warmup complete for {cell.cell_id} ({len(run_specs)} specs) in {time.perf_counter()-t_warm0:.2f}s",
        flush=True,
    )

    records: list[dict[str, Any]] = []
    for s_idx, seed in enumerate(seeds):
        t_seed0 = time.perf_counter()
        rot = int(seed) % len(run_specs)
        ordered_specs = run_specs[rot:] + run_specs[:rot]
        seed_map: dict[tuple[str, float], dict[str, Any]] = {}

        for exec_pos, (arm_name, nepochs) in enumerate(ordered_specs):
            np.random.seed(int(seed) % (2**31 - 1))
            torch.manual_seed(int(seed))
            if primary_dev.type == "cuda":
                torch.cuda.manual_seed_all(int(seed))

            api_kind, solver = instantiate_scale_arm(
                arm_name,
                d=cell.d,
                k=cell.k,
                nepochs=nepochs,
                seed=int(seed),
                multi_devices=sync_devs if cell.num_gpus == 2 else None,
            )
            t_eval_guard = time.perf_counter()
            centroids, init_s, train_s, total_s, wall_s = fit_arm_timed(
                api_kind=api_kind,
                solver=solver,
                corpus=corpus,
                sync_devices=sync_devs,
            )
            if not bool(torch.isfinite(centroids).all().item()):
                raise RuntimeError(
                    f"Non-finite centroids in cell={cell.cell_id}, seed={seed}, arm={arm_name}, T={nepochs}"
                )

            t_ev0 = time.perf_counter()
            metrics = _evaluate_fitted_centroids(cell, corpus, centroids, device=primary_dev)
            eval_s = time.perf_counter() - t_ev0
            _ = t_eval_guard

            rec: dict[str, Any] = {
                "cell_id": cell.cell_id,
                "tier": int(cell.tier),
                "tier_label": cell.tier_label,
                "corpus": cell.corpus_kind,
                "geometry": cell.geometry,
                "num_gpus": int(cell.num_gpus),
                "active_gpus": int(len(sync_devs)),
                "n_train": int(corpus.x_train.shape[0]),
                "n_val": int(corpus.x_val.shape[0]),
                "n_test": int(corpus.x_test.shape[0]),
                "d": int(cell.d),
                "k": int(cell.k),
                "n_times_k": int(corpus.x_train.shape[0]) * int(cell.k),
                "n_over_k": float(corpus.x_train.shape[0]) / float(cell.k),
                "seed": int(seed),
                "seed_order_idx": int(s_idx),
                "exec_order_idx": int(exec_pos),
                "arm": arm_name,
                "wave": "apex" if arm_name.startswith("apex_") else "baselines",
                "nepochs": float(nepochs),
                "host": host,
                "visible_gpu": str(vis_gpu),
                "gpu_sku": gpu_sku,
                "gpu_power_w": float(power_w),
                "selected_gear": str(getattr(solver, "selected_gear", "baseline")),
                "resolved_precision": str(getattr(solver, "resolved_precision", "default")),
                "init_time_sec": float(init_s),
                "train_time_sec": float(train_s),
                "total_time_sec": float(total_s),
                "wall_fit_time_sec": float(wall_s),
                "eval_time_sec": float(eval_s),
                "n_splits": int(getattr(solver, "n_splits", 0)),
                "n_protected": int(getattr(solver, "n_protected", 0)),
                "n_transplants": int(getattr(solver, "n_transplants", 0)),
                **metrics,
            }
            seed_map[(arm_name, float(nepochs))] = rec

        for key in run_specs:
            records.append(seed_map[key])

        print(
            f"[ExtScale] {cell.cell_id} seed={seed} ({s_idx + 1}/{len(seeds)}) "
            f"done in {time.perf_counter() - t_seed0:.2f}s",
            flush=True,
        )

    del corpus
    if primary_dev.type == "cuda":
        torch.cuda.empty_cache()
    return records


# ==============================================================================
# 6. Atomic I/O, Integrity Validation, Cold-JIT Detector & Paired Statistics
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


def atomic_write_text(path: Path, content: str) -> None:
    """Atomically write UTF-8 text via PID+ns temp file + `os.replace`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".tmp_{os.getpid()}_{time.time_ns()}_{path.name}"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def _assert_finite_recursive(obj: Any, path: str = "root") -> None:
    if obj is None:
        raise ValueError(f"Unexpected None at {path}")
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"Non-finite float {obj!r} at {path}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _assert_finite_recursive(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _assert_finite_recursive(v, f"{path}[{i}]")


def detect_cold_jit_spikes(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flag any `(cell_id, host, visible_gpu, arm, nepochs)` where seed_order_idx==0 spiked >2.5x + 15ms."""
    grouped: dict[tuple[str, str, str, str, float], list[dict[str, Any]]] = {}
    for r in records:
        gk = (
            str(r["cell_id"]),
            str(r["host"]),
            str(r["visible_gpu"]),
            str(r["arm"]),
            float(r["nepochs"]),
        )
        grouped.setdefault(gk, []).append(r)

    spikes: list[dict[str, Any]] = []
    for gk, rows in grouped.items():
        if len(rows) < 2:
            continue
        rows_sorted = sorted(rows, key=lambda x: (int(x.get("seed_order_idx", 0)), int(x["seed"])))
        first_t = float(rows_sorted[0]["total_time_sec"])
        rest_med = float(np.median([float(x["total_time_sec"]) for x in rows_sorted[1:]]))
        if first_t > max(rest_med * 2.5 + 0.015, rest_med + 0.250):
            spikes.append(
                {
                    "cell_id": gk[0],
                    "host": gk[1],
                    "visible_gpu": gk[2],
                    "arm": gk[3],
                    "nepochs": gk[4],
                    "first_sec": first_t,
                    "rest_median_sec": rest_med,
                }
            )
    return spikes


def validate_wave_records(
    records: list[dict[str, Any]],
    cells: list[ScaleCellSpec],
    budgets: tuple[float, ...],
    wave: str,
    seed_start: int,
    apex_available: bool,
    smoke: bool = False,
) -> None:
    """Strictly verify completeness, finiteness, zero duplicates, IVF monotonicity, and GPU pairing."""
    _assert_finite_recursive(records)

    expected_keys: set[tuple[str, int, str, float]] = set()
    for cell in cells:
        n_s = 2 if smoke else cell.num_seeds
        c_seeds = [seed_start + i for i in range(n_s)]
        specs = get_wave_run_specs(cell, budgets, wave=wave, apex_available=apex_available)
        for sd in c_seeds:
            for arm_n, b_t in specs:
                expected_keys.add((cell.cell_id, int(sd), arm_n, float(b_t)))

    seen_keys: set[tuple[str, int, str, float]] = set()
    seed_gpu_map: dict[tuple[str, int], set[tuple[str, str]]] = {}
    cell_sku_map: dict[str, set[str]] = {}
    cell_by_id = {c.cell_id: c for c in cells}

    for rec in records:
        rk = (str(rec["cell_id"]), int(rec["seed"]), str(rec["arm"]), float(rec["nepochs"]))
        if rk in seen_keys:
            raise RuntimeError(f"Duplicate record key detected: {rk!r}")
        seen_keys.add(rk)

        if abs(float(rec["total_time_sec"]) - (float(rec["init_time_sec"]) + float(rec["train_time_sec"]))) > 1e-6:
            raise RuntimeError(f"Timing sum mismatch in {rk!r}")
        if float(rec["total_time_sec"]) <= 0.0 or float(rec["test_mse"]) < 0.0:
            raise RuntimeError(f"Invalid non-positive time or negative MSE in {rk!r}")

        c_spec = cell_by_id[str(rec["cell_id"])]
        if c_spec.has_ivf_eval:
            r01 = float(rec["recall10_01pct"])
            r05 = float(rec["recall10_05pct"])
            r10 = float(rec["recall10_10pct"])
            if not (r01 <= r05 + 1e-9 and r05 <= r10 + 1e-9):
                raise RuntimeError(f"IVF Recall@10 monotonicity violation in {rk!r}: {r01}, {r05}, {r10}")

        sg_key = (str(rec["cell_id"]), int(rec["seed"]))
        seed_gpu_map.setdefault(sg_key, set()).add((str(rec["host"]), str(rec["visible_gpu"])))
        cell_sku_map.setdefault(str(rec["cell_id"]), set()).add(str(rec["gpu_sku"]))

    missing = expected_keys - seen_keys
    extra = seen_keys - expected_keys
    if missing or extra:
        raise RuntimeError(
            f"Record completeness mismatch for wave={wave!r}: missing={len(missing)} "
            f"(e.g. {sorted(missing)[:3]}), extra={len(extra)} (e.g. {sorted(extra)[:3]})"
        )

    for sg_key, gpus in seed_gpu_map.items():
        if len(gpus) != 1:
            raise RuntimeError(f"CONFOUNDING VIOLATION: {sg_key} ran on multiple GPUs across arms/waves: {gpus}")

    for cid, skus in cell_sku_map.items():
        if len(skus) != 1:
            raise RuntimeError(f"CROSS-SKU VIOLATION: cell {cid} mixed GPU SKUs: {skus}")


def compute_paired_statistics(
    records: list[dict[str, Any]],
    cells: list[ScaleCellSpec],
    budgets: tuple[float, ...],
) -> list[dict[str, Any]]:
    """Compute paired Student-t and Wilcoxon signed-rank comparisons across seeds for every cell & T."""
    idx: dict[tuple[str, float, str, int], dict[str, Any]] = {}
    seeds_by_cell: dict[str, set[int]] = {}
    arms_by_cell: dict[str, set[str]] = {}
    for r in records:
        cid = str(r["cell_id"])
        idx[(cid, float(r["nepochs"]), str(r["arm"]), int(r["seed"]))] = r
        seeds_by_cell.setdefault(cid, set()).add(int(r["seed"]))
        arms_by_cell.setdefault(cid, set()).add(str(r["arm"]))

    comparisons: list[dict[str, Any]] = []
    for cell in cells:
        seeds = sorted(seeds_by_cell.get(cell.cell_id, set()))
        if not seeds:
            continue
        present_arms = arms_by_cell.get(cell.cell_id, set())
        if cell.num_gpus == 1:
            cands = [a for a in ("apex_emeans", "ultra_emeans", "ultra_emeans_wide") if a in present_arms]
            bases = [
                a
                for a in (
                    "faiss_emeans",
                    "emeans",
                    "smeans",
                    "emeans_pp",
                    "super_emeans",
                    "ultra_emeans",
                    "ultra_emeans_wide",
                    "lloyd_25",
                )
                if a in present_arms
            ]
        else:
            cands = [a for a in ("apex_emeans_2gpu", "multigpu_emeans_fp8") if a in present_arms]
            bases = [a for a in ("multigpu_emeans_bf16", "multigpu_emeans_fp8") if a in present_arms]

        metrics: list[tuple[str, str]] = [
            ("test_mse", "lower"),
            ("total_time_sec", "lower"),
        ]
        if cell.has_ivf_eval:
            metrics.append(("recall10_01pct", "higher"))
            metrics.append(("recall10_10pct", "higher"))
        if cell.has_rare_modes:
            metrics.append(("rare_preserved_frac", "higher"))

        for b_t in budgets:
            for cand in cands:
                for base in bases:
                    if cand == base:
                        continue
                    base_t = 25.0 if base == "lloyd_25" else float(b_t)
                    if not all(
                        (cell.cell_id, float(b_t), cand, s) in idx and (cell.cell_id, base_t, base, s) in idx
                        for s in seeds
                    ):
                        continue
                    for metric_name, direction in metrics:
                        c_vals = np.array(
                            [float(idx[(cell.cell_id, float(b_t), cand, s)][metric_name]) for s in seeds],
                            dtype=np.float64,
                        )
                        b_vals = np.array(
                            [float(idx[(cell.cell_id, base_t, base, s)][metric_name]) for s in seeds],
                            dtype=np.float64,
                        )
                        diffs = c_vals - b_vals
                        rel_pct = 100.0 * diffs / np.maximum(np.abs(b_vals), 1e-30)
                        speedup_ratios = b_vals / np.maximum(np.abs(c_vals), 1e-30)

                        if direction == "lower":
                            wins = int(np.sum(diffs < -1e-12))
                            losses = int(np.sum(diffs > 1e-12))
                        else:
                            wins = int(np.sum(diffs > 1e-12))
                            losses = int(np.sum(diffs < -1e-12))
                        ties = len(seeds) - wins - losses

                        if len(seeds) >= 2 and float(np.std(diffs, ddof=1)) > 1e-15:
                            _, p_t = stats.ttest_rel(c_vals, b_vals)
                            p_ttest = float(p_t)
                            t_crit = float(stats.t.ppf(0.975, df=len(seeds) - 1))
                            ci95_rel = t_crit * float(np.std(rel_pct, ddof=1)) / math.sqrt(len(seeds))
                        else:
                            p_ttest = 1.0
                            ci95_rel = 0.0

                        nz = diffs[np.abs(diffs) > 1e-15]
                        if len(nz) >= 5:
                            try:
                                _, p_w = stats.wilcoxon(nz)
                                p_wilcoxon = float(p_w)
                            except ValueError:
                                p_wilcoxon = 1.0
                        else:
                            p_wilcoxon = p_ttest

                        comparisons.append(
                            {
                                "cell_id": cell.cell_id,
                                "tier": int(cell.tier),
                                "nepochs": float(b_t),
                                "arm": cand,
                                "baseline": base,
                                "metric": metric_name,
                                "direction": direction,
                                "n_seeds": len(seeds),
                                "arm_mean": float(np.mean(c_vals)),
                                "base_mean": float(np.mean(b_vals)),
                                "abs_delta": float(np.mean(diffs)),
                                "rel_delta_pct": float(np.mean(rel_pct)),
                                "ci95_rel_delta_pct": float(ci95_rel),
                                "speedup_x": float(np.mean(speedup_ratios)),
                                "wins": wins,
                                "ties": ties,
                                "losses": losses,
                                "p_ttest": p_ttest,
                                "p_wilcoxon": p_wilcoxon,
                            }
                        )
    return comparisons


# ==============================================================================
# 7. Publication-Grade Plots (`apex_4decade_scale_dominance.png` & Pareto Grid)
# ==============================================================================


def _save_verified_figure(fig: plt.Figure, out_dir: Path, filename: str) -> list[Path]:
    """Atomically write PNG to both `out_dir/filename` and `out_dir/plots/filename` and verify via PIL."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for dst in (out_dir / filename, plots_dir / filename):
        tmp = dst.parent / f".tmp_{os.getpid()}_{time.time_ns()}_{dst.name}"
        try:
            fig.savefig(tmp, format="png", dpi=200, bbox_inches="tight")
            with Image.open(tmp) as im:
                im.verify()
            os.replace(tmp, dst)
            written.append(dst)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
    plt.close(fig)
    return written


def _compute_panel_a_series(
    records: list[dict[str, Any]],
    cells_1gpu: list[ScaleCellSpec],
    arm_id: str,
    max_t: float,
) -> tuple[list[float], list[float], list[str]]:
    """Aggregate 1-GPU cells by `round(log10(N * K * D), 2)` into strictly increasing X buckets."""
    buckets: dict[float, list[tuple[float, float, str]]] = {}
    t_pick = 25.0 if arm_id == "lloyd_25" else max_t
    for c in cells_1gpu:
        sub = [r for r in records if r["cell_id"] == c.cell_id and r["arm"] == arm_id and r["nepochs"] == t_pick]
        if not sub:
            continue
        nkd = float(np.mean([float(r["n_train"]) * float(r["k"]) * float(r["d"]) for r in sub]))
        ms = 1000.0 * float(np.median([float(r["total_time_sec"]) for r in sub]))
        gear = str(sub[0].get("selected_gear", ""))
        b_key = round(math.log10(max(nkd, 1.0)), 2)
        buckets.setdefault(b_key, []).append((nkd, ms, gear))

    xs_nkd: list[float] = []
    ys_ms: list[float] = []
    gears_lbl: list[str] = []
    for b_key in sorted(buckets.keys()):
        items = buckets[b_key]
        x_geo = float(np.exp(np.mean(np.log([max(it[0], 1.0) for it in items]))))
        y_geo = float(np.exp(np.mean(np.log([max(it[1], 1e-6) for it in items]))))
        gear_unique = sorted({it[2] for it in items if it[2] and it[2] not in ("baseline", "auto")})
        xs_nkd.append(x_geo)
        ys_ms.append(y_geo)
        gears_lbl.append("/".join(gear_unique) if gear_unique else "")
    return xs_nkd, ys_ms, gears_lbl


def _compute_panel_c_pareto_series(
    records: list[dict[str, Any]],
    cells_1gpu: list[ScaleCellSpec],
    budgets: tuple[float, ...],
    arm_id: str,
) -> tuple[list[float], list[float]]:
    """Compute `(geomean_speedup_vs_lloyd25, mean_rel_mse_delta_vs_lloyd25_pct)` across `budgets`."""
    if arm_id == "lloyd_25":
        return [1.0], [0.0]
    pts_x: list[float] = []
    pts_y: list[float] = []
    for b_t in sorted(budgets):
        sp_list: list[float] = []
        mse_rel_list: list[float] = []
        for c in cells_1gpu:
            ll_sub = [r for r in records if r["cell_id"] == c.cell_id and r["arm"] == "lloyd_25"]
            ar_sub = [
                r for r in records if r["cell_id"] == c.cell_id and r["arm"] == arm_id and r["nepochs"] == b_t
            ]
            if ll_sub and ar_sub:
                ll_t = float(np.median([float(r["total_time_sec"]) for r in ll_sub]))
                ar_t = float(np.median([float(r["total_time_sec"]) for r in ar_sub]))
                ll_m = float(np.mean([float(r["test_mse"]) for r in ll_sub]))
                ar_m = float(np.mean([float(r["test_mse"]) for r in ar_sub]))
                sp_list.append(ll_t / max(ar_t, 1e-9))
                mse_rel_list.append(100.0 * (ar_m - ll_m) / max(ll_m, 1e-30))
        if sp_list:
            pts_x.append(float(np.exp(np.mean(np.log(np.maximum(sp_list, 1e-6))))))
            pts_y.append(float(np.mean(mse_rel_list)))
    return pts_x, pts_y


def generate_extreme_scale_plots(
    records: list[dict[str, Any]],
    cells: list[ScaleCellSpec],
    budgets: tuple[float, ...],
    out_dir: Path,
) -> list[Path]:
    """Render `apex_4decade_scale_dominance.png` and `extreme_scale_pareto_8cells.png`."""
    all_written: list[Path] = []
    has_apex = any(str(r["arm"]).startswith("apex_") for r in records)
    max_t = max(budgets)
    b_str = ", ".join(f"{b:g}" for b in sorted(budgets))

    cells_1gpu = [c for c in cells if c.num_gpus == 1]
    cells_2gpu = [c for c in cells if c.num_gpus == 2]

    arm_palette: list[tuple[str, str, str, str, str]] = [
        ("lloyd_25", "Lloyd@25", "#222222", "h", ":"),
        ("faiss_emeans", "faiss.contrib.e_means", "#7f7f7f", "s", "--"),
        ("emeans", "e-means", "#999999", "^", "--"),
        ("smeans", "s-means", "#ff7f0e", "v", "-."),
        ("emeans_pp", "e-means++", "#1f77b4", "o", "-."),
        ("super_emeans", "SuperEMeans", "#2ca02c", "D", "--"),
        ("ultra_emeans", "UltraEMeans", "#9467bd", "P", "-"),
        ("ultra_emeans_wide", "UltraEMeans (fast)", "#17becf", "X", "-"),
        ("apex_emeans", "ApexEMeans (4-Gear Auto)", "#d62728", "*", "-"),
    ]

    # ------------------------------------------------------------------
    # Plot 1: Flagship 4-Panel `apex_4decade_scale_dominance.png`
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 11.2))
    fig.suptitle(
        "4-Decade Scale Ladder Showdown (N = 20,000 -> 10,000,000, K = 512 -> 65,536)",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )

    # Panel A: Build Latency (ms) vs Per-Pass FLOP Scale N * K * D at canonical T = max_t
    ax_a = axes[0, 0]
    for arm_id, label, color, marker, ls in arm_palette:
        xs_nkd, ys_ms, gears_lbl = _compute_panel_a_series(records, cells_1gpu, arm_id, max_t)
        if xs_nkd:
            lw = 2.8 if arm_id == "apex_emeans" else 1.6
            msz = 11 if arm_id == "apex_emeans" else 6.5
            ax_a.plot(xs_nkd, ys_ms, label=label, color=color, marker=marker, linestyle=ls, linewidth=lw, markersize=msz)
            if arm_id == "apex_emeans":
                prev_gear = None
                ann_idx = 0
                for x_pt, y_pt, g_txt in zip(xs_nkd, ys_ms, gears_lbl):
                    if g_txt and g_txt != prev_gear:
                        y_off = -14 if (ann_idx % 2 == 0) else 10
                        ax_a.annotate(
                            g_txt,
                            (x_pt, y_pt),
                            textcoords="offset points",
                            xytext=(0, y_off),
                            ha="center",
                            fontsize=7.5,
                            color="#b30000",
                            fontweight="bold",
                        )
                        prev_gear = g_txt
                        ann_idx += 1
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlabel("Per-Pass Compute Scale N * K * D (Point-Centroid-Coord Ops)", fontsize=10.5)
    ax_a.set_ylabel(f"GPU Build Time (ms, log scale) at T={max_t:g}", fontsize=10.5)
    ax_a.set_title("A. Speed Scaling Across 4 Orders of Magnitude", fontsize=11.5, fontweight="bold")
    ax_a.grid(True, which="both", alpha=0.28)
    ax_a.legend(fontsize=8, loc="upper left", ncol=2)

    # Panel B: Held-Out FP64 Test MSE Change (%) vs `faiss_emeans` across all 1-GPU cells at T = max_t
    ax_b = axes[0, 1]
    compare_arms = [
        a
        for a in ("smeans", "super_emeans", "ultra_emeans", "apex_emeans")
        if any(r["arm"] == a for r in records)
    ]
    bar_colors = {
        "smeans": "#ff7f0e",
        "super_emeans": "#2ca02c",
        "ultra_emeans": "#9467bd",
        "apex_emeans": "#d62728",
    }
    if cells_1gpu and compare_arms:
        x_pos = np.arange(len(cells_1gpu), dtype=np.float64)
        width = 0.8 / max(1, len(compare_arms))
        for idx_a, arm_id in enumerate(compare_arms):
            means_pct: list[float] = []
            errs_pct: list[float] = []
            for c in cells_1gpu:
                base_map = {
                    int(r["seed"]): float(r["test_mse"])
                    for r in records
                    if r["cell_id"] == c.cell_id and r["arm"] == "faiss_emeans" and r["nepochs"] == max_t
                }
                arm_map = {
                    int(r["seed"]): float(r["test_mse"])
                    for r in records
                    if r["cell_id"] == c.cell_id and r["arm"] == arm_id and r["nepochs"] == max_t
                }
                common = sorted(set(base_map) & set(arm_map))
                if common:
                    rel = np.array([100.0 * (arm_map[s] - base_map[s]) / max(base_map[s], 1e-30) for s in common])
                    means_pct.append(float(np.mean(rel)))
                    errs_pct.append(float(np.std(rel, ddof=1) / math.sqrt(len(common))) if len(common) >= 2 else 0.0)
                else:
                    means_pct.append(0.0)
                    errs_pct.append(0.0)
            offset = (idx_a - 0.5 * (len(compare_arms) - 1)) * width
            ax_b.bar(
                x_pos + offset,
                means_pct,
                width=width * 0.92,
                yerr=errs_pct,
                capsize=2.5,
                color=bar_colors.get(arm_id, "#333333"),
                label=HUMAN_ARM_NAMES.get(arm_id, arm_id),
                alpha=0.9,
            )
        ax_b.axhline(0.0, color="#333333", linewidth=1.0, linestyle="--")
        short_lbls = [c.cell_id.replace("_1gpu", "").split("_", 1)[-1][:16] for c in cells_1gpu]
        ax_b.set_xticks(x_pos)
        ax_b.set_xticklabels(short_lbls, rotation=24, ha="right", fontsize=8.5)
    ax_b.set_ylabel(f"Held-Out FP64 Test MSE Delta vs faiss.e_means (%, T={max_t:g})", fontsize=10)
    ax_b.set_title("B. Accuracy Gain Across Scales (More Negative = Better)", fontsize=11.5, fontweight="bold")
    ax_b.grid(True, axis="y", alpha=0.28)
    ax_b.legend(fontsize=8.5, loc="lower left")

    # Panel C: Global Speedup vs Accuracy Pareto Across Budgets T (Stationary Anchor = Lloyd@25)
    ax_c = axes[1, 0]
    for arm_id, label, color, marker, ls in arm_palette:
        pts_x, pts_y = _compute_panel_c_pareto_series(records, cells_1gpu, budgets, arm_id)
        if arm_id == "lloyd_25":
            ax_c.scatter(
                pts_x,
                pts_y,
                color=color,
                marker=marker,
                s=90,
                label="Lloyd@25 (1.0x speed, 0.0% MSE ref)",
                zorder=5,
            )
            continue
        if pts_x:
            lw = 2.8 if arm_id == "apex_emeans" else 1.6
            msz = 11 if arm_id == "apex_emeans" else 6.5
            ax_c.plot(pts_x, pts_y, label=label, color=color, marker=marker, linestyle=ls, linewidth=lw, markersize=msz)
    ax_c.axhline(0.0, color="#7f7f7f", linewidth=0.9, linestyle=":")
    ax_c.axvline(1.0, color="#7f7f7f", linewidth=0.8, linestyle=":")
    ax_c.set_xscale("log")
    ax_c.set_xlabel("Geomean Speedup over Lloyd@25 Across All Scales (x, Higher = Faster)", fontsize=10)
    ax_c.set_ylabel("Mean Held-Out Test MSE vs Lloyd@25 (%, Lower = Better)", fontsize=10)
    ax_c.set_title(f"C. Multi-Decade Speed-Accuracy Pareto (T = {b_str})", fontsize=11.5, fontweight="bold")
    ax_c.grid(True, which="both", alpha=0.28)
    ax_c.legend(fontsize=8, loc="best")

    # Panel D: Extreme Tier-4 1-GPU & 2-GPU NCCL Scaling (`2M x K=65,536` and `10M x K=16,384`)
    ax_d = axes[1, 1]
    t4_bars: list[tuple[str, float, float, str]] = []
    for cid_1, cid_2, tag in [
        ("c4a_ext_hugek_2m_k65536_1gpu", "c4c_ext_hugek_2m_k65536_2gpu", "2M, K=65K"),
        ("c4b_ext_10m_k16384_1gpu", "c4d_ext_10m_k16384_2gpu", "10M, K=16K"),
        ("smoke_c4_ext_2gpu", "smoke_c4_ext_2gpu", "Smoke 2xGPU"),
    ]:
        for arm_id, col, short in [
            ("lloyd_25", "#444444", "Lloyd@25 (1G)"),
            ("ultra_emeans", "#9467bd", "Ultra (1G)"),
            ("faiss_emeans", "#7f7f7f", "faiss.em (1G)"),
            ("apex_emeans", "#e34a33", "Apex (1G)"),
        ]:
            t_q = 25.0 if arm_id == "lloyd_25" else max_t
            sub = [r for r in records if r["cell_id"] == cid_1 and r["arm"] == arm_id and r["nepochs"] == t_q]
            if sub:
                t4_bars.append(
                    (
                        f"{tag} | {short}",
                        1000.0 * float(np.median([r["total_time_sec"] for r in sub])),
                        float(np.mean([r["test_mse"] for r in sub])),
                        col,
                    )
                )
        for arm_id, col, short in [
            ("multigpu_emeans_bf16", "#6baed6", "MG-BF16 (2G)"),
            ("multigpu_emeans_fp8", "#2171b5", "MG-FP8 (2G)"),
            ("apex_emeans_2gpu", "#b30000", "Apex (2G)"),
        ]:
            sub2 = [r for r in records if r["cell_id"] == cid_2 and r["arm"] == arm_id and r["nepochs"] == max_t]
            if sub2:
                t4_bars.append(
                    (
                        f"{tag} | {short}",
                        1000.0 * float(np.median([r["total_time_sec"] for r in sub2])),
                        float(np.mean([r["test_mse"] for r in sub2])),
                        col,
                    )
                )

    if t4_bars:
        y_idx = np.arange(len(t4_bars))
        vals_ms = [v for _, v, _, _ in t4_bars]
        ax_d.barh(y_idx, vals_ms, color=[c for _, _, _, c in t4_bars], alpha=0.9)
        ax_d.set_yticks(y_idx)
        ax_d.set_yticklabels([lbl for lbl, _, _, _ in t4_bars], fontsize=8)
        ax_d.invert_yaxis()
        ax_d.set_xscale("log")
        min_v = max(0.2, min(vals_ms) * 0.45)
        max_v = max(vals_ms) * 2.1
        ax_d.set_xlim(min_v, max_v)
        for yi, (_, v_ms, v_mse, _) in enumerate(t4_bars):
            ax_d.text(
                v_ms * 1.04,
                yi,
                f"{v_ms:.1f} ms (MSE {v_mse:.2f})",
                va="center",
                ha="left",
                fontsize=7.3,
            )
        ax_d.set_xlabel(f"Wall-Clock Build Latency (ms, log scale, T={max_t:g})", fontsize=10)
    ax_d.set_title("D. Extreme 65K-Centroid & 10M-Vector (1-GPU -> 2-GPU NCCL)", fontsize=11.5, fontweight="bold")
    ax_d.grid(True, which="both", axis="x", alpha=0.28)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    flagship_name = "apex_4decade_scale_dominance.png" if has_apex else "wave_a_baselines_scale_ladder.png"
    all_written.extend(_save_verified_figure(fig, out_dir, flagship_name))

    # ------------------------------------------------------------------
    # Plot 2: Per-Cell Pareto Frontier Grid (`extreme_scale_pareto_8cells.png`)
    # ------------------------------------------------------------------
    if cells_1gpu:
        ncols = min(4, len(cells_1gpu))
        nrows = math.ceil(len(cells_1gpu) / ncols)
        fig2, axes2 = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.9 * nrows), squeeze=False)
        for idx_c, cell in enumerate(cells_1gpu):
            ax = axes2[idx_c // ncols][idx_c % ncols]
            c_recs = [r for r in records if r["cell_id"] == cell.cell_id]
            for arm_id, label, color, marker, ls in arm_palette:
                if arm_id == "lloyd_25":
                    l_sub = [r for r in c_recs if r["arm"] == "lloyd_25"]
                    if l_sub:
                        ax.scatter(
                            [1000.0 * float(np.median([r["total_time_sec"] for r in l_sub]))],
                            [float(np.mean([r["test_mse"] for r in l_sub]))],
                            label=label,
                            color=color,
                            marker=marker,
                            s=75,
                            zorder=5,
                        )
                    continue
                pts = []
                for b_t in sorted(budgets):
                    sub_b = [r for r in c_recs if r["arm"] == arm_id and r["nepochs"] == b_t]
                    if sub_b:
                        pts.append(
                            (
                                1000.0 * float(np.median([r["total_time_sec"] for r in sub_b])),
                                float(np.mean([r["test_mse"] for r in sub_b])),
                            )
                        )
                if pts:
                    lw = 2.5 if arm_id == "apex_emeans" else 1.4
                    msz = 9.5 if arm_id == "apex_emeans" else 5.5
                    ax.plot(
                        [p[0] for p in pts],
                        [p[1] for p in pts],
                        label=label,
                        color=color,
                        marker=marker,
                        linestyle=ls,
                        linewidth=lw,
                        markersize=msz,
                    )
            ax.set_xscale("log")
            ax.set_title(cell.tier_label, fontsize=9.5, fontweight="bold")
            ax.set_xlabel("Build Time (ms, log)", fontsize=8.5)
            ax.set_ylabel("Held-Out FP64 Test MSE", fontsize=8.5)
            ax.grid(True, which="both", alpha=0.28)
            if idx_c == 0:
                ax.legend(fontsize=6.8, loc="best")
        for empty_i in range(len(cells_1gpu), nrows * ncols):
            axes2[empty_i // ncols][empty_i % ncols].axis("off")
        fig2.suptitle(
            f"Per-Cell Speed vs Held-Out FP64 Test MSE Pareto Frontiers (T = {b_str})",
            fontsize=13.5,
            fontweight="bold",
            y=1.01,
        )
        fig2.tight_layout()
        grid_name = "extreme_scale_pareto_8cells.png" if has_apex else "wave_a_pareto_8cells.png"
        all_written.extend(_save_verified_figure(fig2, out_dir, grid_name))

    _ = cells_2gpu
    return all_written


# ==============================================================================
# 8. Markdown Summary & `docs/extreme_scale_results.md` Synthesis
# ==============================================================================


def build_markdown_report(
    records: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    cells: list[ScaleCellSpec],
    budgets: tuple[float, ...],
    cold_jit_spikes: list[dict[str, Any]],
    for_docs: bool = False,
) -> str:
    """Format a complete, reproducible statistical report for `outputs/extreme_scale/` or `docs/`."""
    has_apex = any(str(r["arm"]).startswith("apex_") for r in records)
    grouped: dict[tuple[str, str, float], list[dict[str, Any]]] = {}
    for r in records:
        grouped.setdefault((str(r["cell_id"]), str(r["arm"]), float(r["nepochs"])), []).append(r)

    img_prefix = "../outputs/extreme_scale/" if for_docs else ""
    flagship_img = "apex_4decade_scale_dominance.png" if has_apex else "wave_a_baselines_scale_ladder.png"
    grid_img = "extreme_scale_pareto_8cells.png" if has_apex else "wave_a_pareto_8cells.png"

    lines: list[str] = [
        "# 4-Decade Extreme-Scale Showdown (`N = 20,000 -> 10,000,000`, `K = 512 -> 65,536`)",
        "",
        f"- **Total Verified Runs**: `{len(records):,}` across `{len(cells)}` scale cells (`0` `NaN`/`Inf`, `0` duplicates, `{len(cold_jit_spikes)}` cold-JIT spikes).",
        "- **Dedicated Hardware Isolation**: `work:0,1` (Workstation Blackwell) for Tiers 1–2 (`20K..200K`); `acorn:0,1` + `cashew:0,1` (600W Server Blackwell) for Tiers 3–4 (`1M..10M`, `K=16K..65,536`, 1-GPU & 2-GPU NCCL).",
        "",
        f"![4-Decade Scale Dominance]({img_prefix}{flagship_img})",
        "",
        f"![Per-Cell Pareto Frontiers]({img_prefix}{grid_img})",
        "",
        "## 1. Per-Cell Speed, Held-Out FP64 Test MSE, Rare-Mode Survival & IVF Recall",
        "",
    ]

    for cell in cells:
        sku = next((str(r["gpu_sku"]) for r in records if r["cell_id"] == cell.cell_id), "Unknown")
        n_tr_actual = next((int(r["n_train"]) for r in records if r["cell_id"] == cell.cell_id), cell.n_train)
        lines.extend(
            [
                f"### `{cell.cell_id}` — {cell.tier_label} (`N={n_tr_actual:,}, D={cell.d}, K={cell.k:,}, GPUs={cell.num_gpus}`, `{sku}`)",
                "",
                "| Arm | `T` | Gear / Prec | Build `ms` (`init+train`) | Train MSE | Val MSE | Held-Out Test MSE | IF | Empty | Extra (`Rare%` / `R@10@0.1%` / `Cos`) |",
                "| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |",
            ]
        )
        arm_order = (
            [("lloyd_25", 25.0)]
            + [
                (a, b_t)
                for b_t in budgets
                for a in (WAVE_A_1GPU_STREAMING_ARMS + WAVE_B_1GPU_STREAMING_ARMS)
            ]
            if cell.num_gpus == 1
            else [
                (a, b_t)
                for b_t in budgets
                for a in (WAVE_A_2GPU_STREAMING_ARMS + WAVE_B_2GPU_STREAMING_ARMS)
            ]
        )
        for arm_id, b_t in arm_order:
            rows = grouped.get((cell.cell_id, arm_id, float(b_t)), [])
            if not rows:
                continue
            tot_ms = float(np.median([1000.0 * float(x["total_time_sec"]) for x in rows]))
            ini_ms = float(np.median([1000.0 * float(x["init_time_sec"]) for x in rows]))
            trn_ms = float(np.median([1000.0 * float(x["train_time_sec"]) for x in rows]))
            tr_m = float(np.mean([float(x["train_mse"]) for x in rows]))
            va_m = float(np.mean([float(x["val_mse"]) for x in rows]))
            te_m = float(np.mean([float(x["test_mse"]) for x in rows]))
            if_m = float(np.mean([float(x["imbalance_factor"]) for x in rows]))
            emp_m = float(np.mean([float(x["empty_clusters"]) for x in rows]))
            gear_s = f"{rows[0]['selected_gear']}/{rows[0]['resolved_precision']}"

            extras: list[str] = []
            if cell.has_rare_modes:
                rp = 100.0 * float(np.mean([float(x["rare_preserved_frac"]) for x in rows]))
                extras.append(f"Rare={rp:.1f}%")
            if cell.has_ivf_eval:
                r01 = 100.0 * float(np.mean([float(x["recall10_01pct"]) for x in rows]))
                r10 = 100.0 * float(np.mean([float(x["recall10_10pct"]) for x in rows]))
                extras.append(f"R@10(0.1%/1%)={r01:.1f}%/{r10:.1f}%")
            if cell.geometry == "unit_sphere":
                cd = float(np.mean([float(x["test_cos_dist"]) for x in rows]))
                extras.append(f"Cos={cd:.5f}")
            extra_str = ", ".join(extras) if extras else "—"

            lines.append(
                f"| `{arm_id}` | `{b_t:g}` | `{gear_s}` | `{tot_ms:.1f}` (`{ini_ms:.1f}+{trn_ms:.1f}`) | "
                f"`{tr_m:.4f}` | `{va_m:.4f}` | **`{te_m:.4f}`** | `{if_m:.3f}` | `{emp_m:.1f}` | {extra_str} |"
            )
        lines.append("")

    lines.extend(
        [
            "## 2. Paired Head-to-Head Statistical Comparisons (`test_mse` & `total_time_sec`)",
            "",
            "| Cell | `T` | Candidate | Baseline | Metric | Cand Mean | Base Mean | Rel Delta / Speedup | W/T/L | `p_ttest` | `p_wilcox` |",
            "| :--- | ---: | :--- | :--- | :--- | ---: | ---: | ---: | :---: | ---: | ---: |",
        ]
    )
    focus_cands = ("apex_emeans", "apex_emeans_2gpu") if has_apex else ("ultra_emeans", "multigpu_emeans_fp8")
    for comp in comparisons:
        if comp["arm"] not in focus_cands:
            continue
        if comp["metric"] not in ("test_mse", "total_time_sec", "recall10_01pct", "rare_preserved_frac"):
            continue
        wtl = f"{comp['wins']}/{comp['ties']}/{comp['losses']}"
        if comp["metric"] == "total_time_sec":
            d_str = f"`{comp['speedup_x']:.2f}x` faster (`{comp['rel_delta_pct']:+.1f}%`)"
        else:
            d_str = f"`{comp['rel_delta_pct']:+.3f}%`"
        lines.append(
            f"| `{comp['cell_id']}` | `{comp['nepochs']:g}` | `{comp['arm']}` | `{comp['baseline']}` | "
            f"`{comp['metric']}` | `{comp['arm_mean']:.5f}` | `{comp['base_mean']:.5f}` | "
            f"{d_str} | {wtl} | `{comp['p_ttest']:.3e}` | `{comp['p_wilcoxon']:.3e}` |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


# ==============================================================================
# 9. Dedicated 6-GPU Remote Fleet Dispatcher (Wave A & Wave B)
# ==============================================================================


def discover_live_ssh_auth_sock() -> str | None:
    """Auto-discover live forwarded `SSH_AUTH_SOCK` read-only from `tmux` or `/tmp/ssh-*/agent.*`."""
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


def shard_json_filename(wave: str, shard_idx: int, num_shards: int) -> str:
    prefix = {"baselines": "wave_a", "apex": "wave_b", "all": "wave_all"}[wave]
    return f"{prefix}_shard_{shard_idx:02d}_of_{num_shards:02d}.json"


def dispatch_dedicated_fleet(
    wave: str,
    seed_start: int,
    out_dir: Path,
    data_dir: Path,
) -> None:
    """Dispatch Stage 1 (6x 1-GPU shards) then Stage 2 (2x 2-GPU NCCL shards) across `work`, `acorn`, `cashew`."""
    if wave not in ("baselines", "apex", "all"):
        raise ValueError(f"Unsupported wave={wave!r}")
    if wave in ("apex", "all"):
        resolve_apex_class(required=True)

    shards = build_dedicated_shards(seed_start=seed_start)
    num_shards = len(shards)
    assert num_shards == 8

    # Ensure local CIFAR-10 and SIFT1M/GloVe1M validated `.npz` caches exist before syncing
    dev_local = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_cifar10_resnet18_embeddings(mode="global_512", device=str(dev_local))
    load_cifar10_resnet18_embeddings(mode="patches_128", device=str(dev_local))
    load_sift1m(data_dir=data_dir, device=dev_local)
    load_glove1m_unit(data_dir=data_dir, device=dev_local)

    env = os.environ.copy()
    live_sock = discover_live_ssh_auth_sock()
    if live_sock:
        env["SSH_AUTH_SOCK"] = live_sock

    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    rel_out = out_dir.relative_to(ROOT) if out_dir.is_relative_to(ROOT) else Path("outputs/extreme_scale")
    remote_mscale_data = f"{REMOTE_WORKDIR}/outputs/phase4_million_scale/data"

    def _sync_host(host: str) -> None:
        assert_safe_fleet_target(host, "emeans_ext_s99")
        print(f"[ExtScale-Fleet] Syncing code & caches to {host}:{REMOTE_WORKDIR} ...", flush=True)
        prep_cmd = (
            f"mkdir -p {REMOTE_WORKDIR}/outputs/logs {REMOTE_WORKDIR}/outputs/phase1/cache {remote_mscale_data} && "
            f"for d in /home/catid/emeans_runs/real-scale/outputs/phase4_million_scale/data "
            f"/home/catid/emeans_runs/ultra-synth/outputs/phase1/cache "
            f"/home/catid/emeans_runs/smeans-bench/outputs/phase1/cache; do "
            f"  if [ -d \"$d\" ]; then "
            f"    case \"$d\" in *phase4_million_scale*) cp -lf \"$d\"/*.npz {remote_mscale_data}/ 2>/dev/null || true ;; "
            f"    *phase1/cache*) cp -lf \"$d\"/*.npz {REMOTE_WORKDIR}/outputs/phase1/cache/ 2>/dev/null || true ;; esac; "
            f"  fi; "
            f"done"
        )
        subprocess.run(
            ["ssh", "-A", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, prep_cmd],
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
        if host == "work":
            subprocess.run(
                [
                    "rsync", "-az", "--ignore-existing",
                    "-e", "ssh -A -o BatchMode=yes -o ConnectTimeout=10",
                    f"{ROOT}/outputs/phase1/cache/",
                    f"{host}:{REMOTE_WORKDIR}/outputs/phase1/cache/",
                ],
                env=env,
                check=True,
            )
        else:
            # Ensure both SIFT1M and GloVe-1M validated caches exist on `acorn` and `cashew`
            for npz_name, loader_fn in [
                ("sift1m_validated_v1.npz", "load_sift1m"),
                ("glove1m_unit_s20261116_validated_v1.npz", "load_glove1m_unit"),
            ]:
                chk = subprocess.run(
                    [
                        "ssh", "-A", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                        host,
                        f"test -f {remote_mscale_data}/{npz_name} && echo PRESENT || echo MISSING",
                    ],
                    env=env,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                if "MISSING" in chk.stdout:
                    pop_cmd = (
                        f"mkdir -p /home/catid/emeans_env/tmp && "
                        f"export PATH=\"$HOME/.local/bin:/usr/local/cuda/bin:$PATH\"; "
                        f"export CUDA_VISIBLE_DEVICES=0; "
                        f"export TMPDIR=/home/catid/emeans_env/tmp; "
                        f"export TRITON_CACHE_DIR=/home/catid/emeans_env/tmp/triton_cache_ext_prep; "
                        f"export PYTHONPATH={REMOTE_WORKDIR}/src; "
                        f"cd {REMOTE_WORKDIR} && "
                        f"{REMOTE_PYTHON} -c 'from emeans.real_corpora import {loader_fn}; "
                        f"{loader_fn}(data_dir=\"{remote_mscale_data}\", device=\"cuda\")'"
                    )
                    subprocess.run(
                        ["ssh", "-A", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, pop_cmd],
                        env=env,
                        check=True,
                    )
        print(f"[ExtScale-Fleet] Host {host} ready.", flush=True)

    unique_hosts = sorted({sp.host for sp in shards})
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(unique_hosts)) as pool:
        list(pool.map(_sync_host, unique_hosts))

    def _run_stage(stage_shards: list[DedicatedShardSpec], stage_num: int) -> None:
        t_st0 = time.perf_counter()
        for sp in stage_shards:
            assert_safe_fleet_target(sp.host, sp.session)
            remote_out = f"{REMOTE_WORKDIR}/{rel_out}"
            s_fname = shard_json_filename(wave, sp.shard_idx, num_shards)
            remote_json = f"{remote_out}/shards/{s_fname}"
            remote_cmd = (
                f"mkdir -p /home/catid/emeans_env/tmp {REMOTE_WORKDIR}/outputs/logs {remote_out}/shards && "
                f"rm -f {REMOTE_WORKDIR}/outputs/logs/{sp.session}.status {remote_json} && "
                f"tmux kill-session -t {sp.session} 2>/dev/null || true; "
                f"tmux new-session -d -s {sp.session} "
                + shlex.quote(
                    f"export PATH=\"$HOME/.local/bin:/usr/local/cuda/bin:$PATH\"; "
                    f"export CUDA_VISIBLE_DEVICES={sp.visible_gpus}; "
                    f"export TMPDIR=/home/catid/emeans_env/tmp; "
                    f"export TRITON_CACHE_DIR=/home/catid/emeans_env/tmp/triton_cache_{sp.session}; "
                    f"export PYTHONPATH={REMOTE_WORKDIR}/src; "
                    f"cd {REMOTE_WORKDIR} && "
                    f"{REMOTE_PYTHON} scripts/run_extreme_scale_showdown.py "
                    f"--wave {wave} --shard-idx {sp.shard_idx} --num-shards {num_shards} "
                    f"--seed-start {seed_start} --data-dir {remote_mscale_data} --out-dir {remote_out} "
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
                f"[ExtScale-Stage{stage_num}] Launched shard {sp.shard_idx:02d}/{num_shards} on "
                f"{sp.host} GPUs={sp.visible_gpus} ({sp.session}, wave={wave})",
                flush=True,
            )

        pending = {sp.shard_idx: sp for sp in stage_shards}
        while pending:
            time.sleep(4.0)
            done_ids: list[int] = []
            for sidx, sp in list(pending.items()):
                res = subprocess.run(
                    [
                        "ssh", "-A", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                        sp.host,
                        f"cat {REMOTE_WORKDIR}/outputs/logs/{sp.session}.status 2>/dev/null || true",
                    ],
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
                    s_fname = shard_json_filename(wave, sidx, num_shards)
                    remote_json = f"{REMOTE_WORKDIR}/{rel_out}/shards/{s_fname}"
                    local_json = shards_dir / s_fname
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
                        f"[ExtScale-Stage{stage_num}] Collected shard {sidx:02d} ({sp.host} GPUs={sp.visible_gpus}) "
                        f"at +{time.perf_counter() - t_st0:.1f}s",
                        flush=True,
                    )
                    done_ids.append(sidx)
            for did in done_ids:
                pending.pop(did, None)

    stage1 = [sp for sp in shards if sp.stage == 1]
    stage2 = [sp for sp in shards if sp.stage == 2]
    _run_stage(stage1, stage_num=1)
    _run_stage(stage2, stage_num=2)


# ==============================================================================
# 10. Merge & CLI Entrypoint
# ==============================================================================


def merge_and_synthesize(
    out_dir: Path,
    cells: list[ScaleCellSpec],
    budgets: tuple[float, ...],
    seed_start: int,
    num_shards: int = 8,
    write_docs: bool = False,
) -> dict[str, Any]:
    """Load cached Wave A (+ Wave B if present) shards, validate, compute stats, and write plots/reports."""
    shards_dir = out_dir / "shards"
    have_wave_a = all((shards_dir / shard_json_filename("baselines", i, num_shards)).exists() for i in range(num_shards))
    have_wave_b = all((shards_dir / shard_json_filename("apex", i, num_shards)).exists() for i in range(num_shards))
    have_wave_all = all((shards_dir / shard_json_filename("all", i, num_shards)).exists() for i in range(num_shards))

    def _strict_load_json(p: Path) -> dict[str, Any]:
        return json.loads(
            p.read_text(encoding="utf-8"),
            parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"Invalid JSON constant {c} in {p}")),
        )

    merged_records: list[dict[str, Any]] = []
    if have_wave_all:
        effective_wave = "all"
        apex_present = True
        for i in range(num_shards):
            p = shards_dir / shard_json_filename("all", i, num_shards)
            data = _strict_load_json(p)
            merged_records.extend(data["records"])
    elif have_wave_a and have_wave_b:
        effective_wave = "all"
        apex_present = True
        for i in range(num_shards):
            pa = shards_dir / shard_json_filename("baselines", i, num_shards)
            pb = shards_dir / shard_json_filename("apex", i, num_shards)
            da = _strict_load_json(pa)
            db = _strict_load_json(pb)
            merged_records.extend(da["records"])
            merged_records.extend(db["records"])
    elif have_wave_a:
        effective_wave = "baselines"
        apex_present = False
        for i in range(num_shards):
            pa = shards_dir / shard_json_filename("baselines", i, num_shards)
            da = _strict_load_json(pa)
            merged_records.extend(da["records"])
    else:
        raise FileNotFoundError(f"Incomplete Wave A shard files in {shards_dir}")

    validate_wave_records(
        merged_records,
        cells=cells,
        budgets=budgets,
        wave=effective_wave,
        seed_start=seed_start,
        apex_available=apex_present,
        smoke=False,
    )
    spikes = detect_cold_jit_spikes(merged_records)
    comparisons = compute_paired_statistics(merged_records, cells, budgets)

    payload = {
        "phase": "phase5_extreme_scale_showdown" if apex_present else "phase5_extreme_scale_wave_a_baselines",
        "wave_state": effective_wave,
        "apex_included": apex_present,
        "num_records": len(merged_records),
        "cold_jit_spikes": spikes,
        "budgets": list(budgets),
        "seed_start": int(seed_start),
        "cells": [asdict(c) for c in cells],
        "records": merged_records,
        "comparisons": comparisons,
    }

    json_name = "extreme_scale_results.json" if apex_present else "wave_a_baselines_results.json"
    md_name = "extreme_scale_summary.md" if apex_present else "wave_a_baselines_summary.md"
    atomic_write_json(out_dir / json_name, payload)
    md_text = build_markdown_report(merged_records, comparisons, cells, budgets, spikes, for_docs=False)
    atomic_write_text(out_dir / md_name, md_text)
    plots = generate_extreme_scale_plots(merged_records, cells, budgets, out_dir)

    if write_docs and apex_present:
        docs_text = build_markdown_report(merged_records, comparisons, cells, budgets, spikes, for_docs=True)
        atomic_write_text(DOCS_OUT_PATH, docs_text)

    print(
        f"[ExtScale-Merge] Validated {len(merged_records):,} records (apex={apex_present}, "
        f"cold_jit_spikes={len(spikes)}, plots={len(plots)}) -> {out_dir / json_name}",
        flush=True,
    )
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="4-Decade Extreme-Scale Showdown Runner (`20K -> 10M`, `K = 512 -> 65,536`)")
    p.add_argument(
        "--wave",
        type=str,
        default="all",
        choices=("baselines", "apex", "all"),
        help="Run Wave A ('baselines'), Wave B ('apex'), or both ('all').",
    )
    p.add_argument("--smoke", action="store_true", help="Run fast 4-cell, 2-seed end-to-end smoke check.")
    p.add_argument("--dispatch-fleet", action="store_true", help="Dispatch dedicated 6-GPU fleet (`work`, `acorn`, `cashew`).")
    p.add_argument("--merge-only", action="store_true", help="Merge cached Wave A (+ Wave B) shards & plot.")
    p.add_argument("--write-docs", action="store_true", help="Also write `docs/extreme_scale_results.md` when Wave B is merged.")
    p.add_argument("--shard-idx", type=int, default=None, help="Worker shard index (0..7).")
    p.add_argument("--num-shards", type=int, default=8, help="Total dedicated shards (default: 8).")
    p.add_argument("--seed-start", type=int, default=5000, help="Starting paired RNG seed (default: 5000).")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_MILLION_DATA_DIR, help="Million-scale validated corpus dir.")
    p.add_argument("--out-dir", type=Path, default=OUTPUT_DIR, help="Output directory for JSON, Markdown, and PNGs.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir: Path = args.out_dir.resolve()
    data_dir: Path = args.data_dir.resolve()

    if args.smoke:
        budgets: tuple[float, ...] = (1.0, 2.0)
        cells = build_scale_cells(smoke=True)
        apex_avail = resolve_apex_class(required=(args.wave == "apex")) is not None
        seeds = [int(args.seed_start), int(args.seed_start) + 1]
        smoke_records: list[dict[str, Any]] = []
        for cell in cells:
            smoke_records.extend(
                execute_scale_cell(cell, seeds, budgets, wave=args.wave, data_dir=data_dir, smoke=True)
            )
        validate_wave_records(
            smoke_records,
            cells=cells,
            budgets=budgets,
            wave=args.wave,
            seed_start=int(args.seed_start),
            apex_available=apex_avail,
            smoke=True,
        )
        spikes = detect_cold_jit_spikes(smoke_records)
        comparisons = compute_paired_statistics(smoke_records, cells, budgets)
        payload = {
            "phase": "phase5_extreme_scale_smoke",
            "wave_state": args.wave,
            "apex_included": apex_avail and args.wave in ("apex", "all"),
            "num_records": len(smoke_records),
            "cold_jit_spikes": spikes,
            "budgets": list(budgets),
            "cells": [asdict(c) for c in cells],
            "records": smoke_records,
            "comparisons": comparisons,
        }
        atomic_write_json(out_dir / "extreme_scale_smoke_results.json", payload)
        md_text = build_markdown_report(smoke_records, comparisons, cells, budgets, spikes, for_docs=False)
        atomic_write_text(out_dir / "extreme_scale_smoke_summary.md", md_text)
        plots = generate_extreme_scale_plots(smoke_records, cells, budgets, out_dir)
        print(
            f"[ExtScale-Smoke] SUCCESS: {len(smoke_records)} records, spikes={len(spikes)}, "
            f"plots={len(plots)} written to {out_dir}",
            flush=True,
        )
        return

    cells = build_scale_cells(smoke=False)
    cell_by_id = {c.cell_id: c for c in cells}
    budgets = EPOCH_BUDGETS
    seed_start = int(args.seed_start)

    if args.shard_idx is not None:
        if int(args.num_shards) != 8:
            raise ValueError("Expected --num-shards 8 matching build_dedicated_shards()")
        shards = build_dedicated_shards(seed_start=seed_start)
        sp = shards[int(args.shard_idx)]
        shard_recs: list[dict[str, Any]] = []
        for cid, c_seeds in sp.cell_seed_map.items():
            shard_recs.extend(
                execute_scale_cell(
                    cell_by_id[cid],
                    seeds=list(c_seeds),
                    budgets=budgets,
                    wave=args.wave,
                    data_dir=data_dir,
                    smoke=False,
                )
            )
        s_path = out_dir / "shards" / shard_json_filename(args.wave, sp.shard_idx, int(args.num_shards))
        atomic_write_json(
            s_path,
            {
                "shard_idx": sp.shard_idx,
                "stage": sp.stage,
                "wave": args.wave,
                "host": sp.host,
                "visible_gpus": sp.visible_gpus,
                "records": shard_recs,
            },
        )
        print(f"[ExtScale-Shard {sp.shard_idx:02d}] Wrote {len(shard_recs)} records -> {s_path}", flush=True)
        return

    if args.dispatch_fleet:
        dispatch_dedicated_fleet(wave=args.wave, seed_start=seed_start, out_dir=out_dir, data_dir=data_dir)
        merge_and_synthesize(
            out_dir=out_dir,
            cells=cells,
            budgets=budgets,
            seed_start=seed_start,
            num_shards=8,
            write_docs=args.write_docs,
        )
        return

    if args.merge_only:
        merge_and_synthesize(
            out_dir=out_dir,
            cells=cells,
            budgets=budgets,
            seed_start=seed_start,
            num_shards=int(args.num_shards),
            write_docs=args.write_docs,
        )
        return

    raise ValueError("Specify one of: --smoke, --shard-idx, --dispatch-fleet, or --merge-only")


if __name__ == "__main__":
    main()
