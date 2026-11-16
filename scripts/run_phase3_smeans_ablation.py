#!/usr/bin/env python3
"""Phase 3 Multi-Machine Parallel Benchmark & Component Ablation Campaign: `s-means` vs `SuperEMeans` & Hybrids.

Answers the core research question across 6 Blackwell machines (14 GPUs in parallel):
  "Look for improvements to our algorithm by taking the components of `https://github.com/suayptalha/s-means`
   and running ablations. Determine if `s-means` is better than our best ideas (`SuperEMeans`) or if a
   combination (`HybridSEMeans`) is best."

Experimental Design (Two De-Confounded, 15-Seed Paired Suites):
  1. Suite 1 — Head-to-Head Showdown Across Diverse Regimes & Epoch Budgets (`T in {1, 2, 5}` + Pareto):
     Compares `Lloyd@25`, `Vanilla EMeans`, `EMeansPP`, `SMeans (exact_ref)`, `SMeans (fast)`,
     `SuperEMeans`, `SuperEMeans (wide)`, `HybridSEMeans`, and `HybridSEMeans (wide)` across 10 diverse
     regimes (30 cells x 15 seeds x 9 arms = 4,050 runs):
       - `isotropic_ctrl` (`isotropic_gaussian`, n=100k, d=128, k=512)
       - `stretched_a2` (`stretched_gaussian_2`, n=100k, d=128, k=1024)
       - `balanced_gmm` (`balanced_gmm`, n=100k, d=128, k=1024)
       - `zipf_gmm` (`zipf_gmm`, n=100k, d=128, k=1024)
       - `multiscale_rare` (`multiscale_rare_outliers`, n=100k, d=128, k=512)
       - `low_nk32_gmm` (`balanced_gmm`, n=65,536, d=128, k=2048)
       - `lowd_zipf_d8` (`zipf_gmm`, n=100k, d=8, k=512 — `s-means` d<=16 quality-law & low-d revival target)
       - `cifar10_patches_128` (real 200K 128-D ResNet-18 patch descriptors, k=2048)
       - `cifar10_r18_512` (real 50K 512-D ResNet-18 global embeddings, k=1024)
       - `cifar10_r18_512_unit` (unit-sphere L2-normalized real 512-D embeddings, k=1024; reports BOTH
         exact FP64 Euclidean MSE and exact FP64 spherical cosine distortion, de-confounded from unnormalized cells)
  2. Suite 2 — Full Directional Component Ablation Matrix (12 cells x 15 seeds x 40 arms = 7,200 runs):
     (a) Leave-one-out ablation of every `s-means` component (`C1..C7` + `couple_sse_to_alpha`) inside `SMeans`.
     (b) Additive grafting of every `s-means` component (`C1..C7`) onto `EMeansPP` AND onto `SuperEMeans`.
     (c) Additive grafting of every `SuperEMeans` pillar (`P1..P6`: Coreset+Voronoi+PriorMass, Huygens Shield,
         Oja Split, EB-JS Finalizer, P/Q `head_edit`, WideStep) onto `SMeans`, plus cross-pollinated hybrids.

Scientific & Benchmarking Hygiene (`AGENTS.md`):
  - Zero train/val/test leakage: `val` and `test` splits are strictly isolated; `test` is touched ONLY for
    post-fit `exact_mse_fp64` (and `exact_cosine_distortion_fp64` on unit-sphere cells).
  - Exact full-shape per-cell untimed GPU/Triton warmup (`nepochs=cell.nepochs` on full `x_tr`) outside all
    timers so zero cold Triton JIT compilations ever leak into Seed 0 or any arm.
  - `torch.cuda.synchronize()` around every timer (`init_time_sec`, `train_time_sec`, `total_time_sec`).
  - Single-GPU / single-SKU cell pairing + per-seed cyclic arm-order counterbalancing so paired MSE deltas
    and paired wall-clock speedups never mix heterogeneous TDP SKUs across arms of the same cell.
  - Atomic PID+nanosecond temp-file + `os.replace` JSON writes with `allow_nan=False` (`NaN`/`Inf` impossible).
  - Safe 14-GPU parallel dispatch across all 6 approved machines (`foureyes`, `ripper`, `work`, `loki`,
    `acorn.stratabaremetal.com`, `cashew.stratabaremetal.com` — `walnut` strictly excluded, `tmux` session `0` untouched).
"""

from __future__ import annotations

import argparse
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

# Isolate Triton's disk cache per (uid, visible_gpu) BEFORE importing torch/triton/emeans
# so concurrent multi-GPU processes never collide on shared `$HOME/.triton/cache`.
if "TRITON_CACHE_DIR" not in os.environ:
    _vis_gpu_tag = os.environ.get("CUDA_VISIBLE_DEVICES", "all").replace(",", "_")
    os.environ["TRITON_CACHE_DIR"] = f"/tmp/triton_cache_emeans_u{os.getuid()}_g{_vis_gpu_tag}"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from emeans.baselines import LloydKMeans
from emeans.datasets import (
    DEFAULT_CACHE_DIR,
    DEFAULT_CIFAR10_ROOT,
    DatasetSplit,
    load_cifar10_resnet18_embeddings,
    make_synthetic_dataset,
)
from emeans.emeans import EMeans
from emeans.emeans_pp import EMeansPP
from emeans.kernels import HAS_TRITON, assign_points, exact_mse_fp64
from emeans import seeding as emeans_seeding
from emeans.smeans import SMEANS_ABLATION_CONFIGS, SMeans
from emeans.super_emeans import (
    HYBRID_ABLATION_ARMS,
    HybridSEMeans,
    SuperEMeans,
    build_hybrid_arm,
)


OUTPUT_DIR = ROOT / "outputs" / "phase3_smeans"

# 13 non-overlapping GPUs across ALL 6 approved machines:
#   - acorn.stratabaremetal.com: GPUs 0, 1 (600W Server Edition)
#   - cashew.stratabaremetal.com: GPUs 0, 1 (600W Server Edition)
#   - work: GPUs 0, 1 (450W Workstation Edition)
#   - ripper: GPUs 0, 1, 2, 3 (300W Max-Q)
#   - loki: GPUs 0, 1 (300W Max-Q)
#   - local (foureyes): GPU 3 (300W Max-Q; GPUs 0, 1, 2 left untouched for real-scale, ultra-synth, & patrol-3!)
# NEVER includes `walnut.stratabaremetal.com` and NEVER touches any `tmux` session `0`.
FLEET_SHARDS: tuple[tuple[str, int, str, float], ...] = (
    ("acorn.stratabaremetal.com", 0, "emeans_p3_smeans_s00", 1.35),
    ("acorn.stratabaremetal.com", 1, "emeans_p3_smeans_s01", 1.35),
    ("cashew.stratabaremetal.com", 0, "emeans_p3_smeans_s02", 1.35),
    ("cashew.stratabaremetal.com", 1, "emeans_p3_smeans_s03", 1.35),
    ("work", 0, "emeans_p3_smeans_s04", 1.18),
    ("work", 1, "emeans_p3_smeans_s05", 1.18),
    ("ripper", 0, "emeans_p3_smeans_s06", 1.00),
    ("ripper", 1, "emeans_p3_smeans_s07", 1.00),
    ("ripper", 2, "emeans_p3_smeans_s08", 1.00),
    ("ripper", 3, "emeans_p3_smeans_s09", 1.00),
    ("loki", 0, "emeans_p3_smeans_s10", 1.00),
    ("loki", 1, "emeans_p3_smeans_s11", 1.00),
    ("local", 3, "emeans_p3_smeans_s12", 1.00),
)

REMOTE_WORKDIR = "/home/catid/emeans_runs/smeans-bench"
REMOTE_PYTHON = "/home/catid/emeans_env/.venv/bin/python"
REMOTE_CIFAR_FALLBACK = Path("/home/catid/emeans_env/data/cifar10/cifar-10-batches-py")


@dataclass(frozen=True)
class CellSpec:
    """One experimental `(suite, dataset, n_train, d, k, nepochs)` evaluation cell."""

    cell_id: str
    suite: str             # "suite1" (head-to-head + Pareto) or "suite2" (directional component ablation)
    regime_name: str       # canonical regime stem (e.g. "zipf_gmm", "cifar10_r18_512")
    dataset_kind: str
    n_train: int
    d: int
    k: int
    nepochs: float
    geometry: str          # "euclidean" or "unit_sphere"


SUITE1_ARMS: tuple[str, ...] = (
    "lloyd_25",
    "vanilla_emeans",
    "emeans_pp",
    "smeans_exact_ref",
    "smeans_fast",
    "super_emeans",
    "super_emeans_wide",
    "hybrid_super_smeans",
    "hybrid_super_smeans_wide",
)

# Canonical ordered LOO keys expected in `SMEANS_ABLATION_CONFIGS`
EXPECTED_SMEANS_LOO_KEYS: tuple[str, ...] = (
    "smeans_full",
    "loo_no_d15_init",
    "loo_no_power_cos",
    "loo_no_revive_stop",
    "loo_no_sse_donor",
    "loo_no_rms_split",
    "loo_no_feat_scale",
    "loo_no_pass_weight",
    "loo_no_eb_finalize",
    "smeans_alpha_coupled_sse",
    "all_off_emeans_equiv",
)

# Canonical ordered keys expected in `HYBRID_ABLATION_ARMS`
EXPECTED_HYBRID_ARM_KEYS: tuple[str, ...] = (
    # Matched-kernel anchors
    "emeans_pp_base",
    "super_emeans",
    "super_emeans_wide",
    "smeans_deconfounded",
    # Suite 2(b): s-means C1..C7 grafted onto EMeansPP
    "pp_plus_c1_d15_raw",
    "pp_plus_c1_d15_polished",
    "pp_plus_c2_power_cos",
    "pp_plus_c3_revive_stop",
    "pp_plus_c4_sse_usage_donor",
    "pp_plus_c5_rms_aniso_split",
    "pp_plus_c6_pass_weight",
    "pp_plus_c7_eb_ema",
    # Suite 2(b): s-means C1..C7 grafted onto SuperEMeans
    "super_plus_c1_d15_init",
    "super_plus_c2_power_cos",
    "super_plus_c3_revive_stop",
    "super_plus_c4_sse_usage_donor",
    "super_plus_c5_rms_aniso_split",
    "super_plus_c5b_oja_rms_split",
    "super_plus_c6_pass_weight",
    "super_plus_c7_eb_ema",
    # Suite 2(c): SuperEMeans pillars P1..P6 grafted onto SMeans
    "smeans_plus_p1_coreset_voronoi",
    "smeans_plus_p2_huygens_shield",
    "smeans_plus_p3_oja_split",
    "smeans_plus_p3_oja008",
    "smeans_plus_p4_eb_js",
    "smeans_plus_p5_head_edit",
    "smeans_plus_p6_wide_step",
    # Combined cross-pollinated hybrids
    "hybrid_super_smeans",
    "hybrid_super_smeans_wide",
)


def get_suite2_arm_specs() -> list[tuple[str, str, str]]:
    """Return ordered `(arm_id, family, config_key)` specs for Suite 2 directional ablations."""
    specs: list[tuple[str, str, str]] = []

    # 2(a): Leave-one-out ablation inside SMeans (`smeans.py`)
    seen_sm: set[str] = set()
    for k_sm in EXPECTED_SMEANS_LOO_KEYS:
        if k_sm in SMEANS_ABLATION_CONFIGS:
            specs.append((f"s2a_{k_sm}", "smeans_loo", k_sm))
            seen_sm.add(k_sm)
    for k_sm in sorted(SMEANS_ABLATION_CONFIGS.keys()):
        if k_sm not in seen_sm:
            specs.append((f"s2a_{k_sm}", "smeans_loo", k_sm))

    # 2(b) & 2(c): Bidirectional grafting in `HybridSEMeans` (`super_emeans.py`)
    seen_hyb: set[str] = set()
    for k_hyb in EXPECTED_HYBRID_ARM_KEYS:
        if k_hyb in HYBRID_ABLATION_ARMS:
            specs.append((f"s2bc_{k_hyb}", "hybrid_graft", k_hyb))
            seen_hyb.add(k_hyb)
    for k_hyb in sorted(HYBRID_ABLATION_ARMS.keys()):
        if k_hyb not in seen_hyb:
            specs.append((f"s2bc_{k_hyb}", "hybrid_graft", k_hyb))

    return specs


def build_campaign_cells(smoke: bool = False, suite_filter: str = "all") -> list[CellSpec]:
    """Construct Suite 1 (30 cells) and Suite 2 (12 cells), or fast smoke cells."""
    if smoke:
        cells = [
            CellSpec("smoke_s1_zipf_t2", "suite1", "zipf_gmm", "zipf_gmm", 8192, 64, 512, 2.0, "euclidean"),
            CellSpec("smoke_s1_unit_t1", "suite1", "unit_sphere", "isotropic_gaussian", 8192, 64, 256, 1.0, "unit_sphere"),
            CellSpec("smoke_s2_rare_t2", "suite2", "multiscale_rare", "multiscale_rare_outliers", 8192, 64, 512, 2.0, "euclidean"),
        ]
        if suite_filter != "all":
            cells = [c for c in cells if c.suite == suite_filter]
        return cells

    s1_regimes: list[tuple[str, str, int, int, int, str]] = [
        ("isotropic_ctrl", "isotropic_gaussian", 100_000, 128, 512, "euclidean"),
        ("stretched_a2", "stretched_gaussian_2", 100_000, 128, 1024, "euclidean"),
        ("balanced_gmm", "balanced_gmm", 100_000, 128, 1024, "euclidean"),
        ("zipf_gmm", "zipf_gmm", 100_000, 128, 1024, "euclidean"),
        ("multiscale_rare", "multiscale_rare_outliers", 100_000, 128, 512, "euclidean"),
        ("low_nk32_gmm", "balanced_gmm", 65_536, 128, 2048, "euclidean"),
        ("lowd_zipf_d8", "zipf_gmm", 100_000, 8, 512, "euclidean"),
        ("cifar10_patches_128", "cifar10_patches_128", 200_000, 128, 2048, "euclidean"),
        ("cifar10_r18_512", "cifar10_global_512", 50_000, 512, 1024, "euclidean"),
        ("cifar10_r18_512_unit", "cifar10_global_512_unit", 50_000, 512, 1024, "unit_sphere"),
    ]
    suite1_cells: list[CellSpec] = []
    for reg_name, ds_kind, n_tr, d, k, geom in s1_regimes:
        for t_ep in (1.0, 2.0, 5.0):
            t_str = f"t{int(t_ep)}"
            suite1_cells.append(
                CellSpec(
                    cell_id=f"s1_{reg_name}_{t_str}",
                    suite="suite1",
                    regime_name=reg_name,
                    dataset_kind=ds_kind,
                    n_train=n_tr,
                    d=d,
                    k=k,
                    nepochs=t_ep,
                    geometry=geom,
                )
            )

    suite2_cells: list[CellSpec] = [
        CellSpec("s2_zipf_gmm_t1", "suite2", "zipf_gmm", "zipf_gmm", 100_000, 128, 1024, 1.0, "euclidean"),
        CellSpec("s2_zipf_gmm_t2", "suite2", "zipf_gmm", "zipf_gmm", 100_000, 128, 1024, 2.0, "euclidean"),
        CellSpec("s2_zipf_gmm_t5", "suite2", "zipf_gmm", "zipf_gmm", 100_000, 128, 1024, 5.0, "euclidean"),
        CellSpec("s2_stretched_a2_t2", "suite2", "stretched_a2", "stretched_gaussian_2", 100_000, 128, 1024, 2.0, "euclidean"),
        CellSpec("s2_stretched_a2_t5", "suite2", "stretched_a2", "stretched_gaussian_2", 100_000, 128, 1024, 5.0, "euclidean"),
        CellSpec("s2_multiscale_rare_t2", "suite2", "multiscale_rare", "multiscale_rare_outliers", 100_000, 128, 512, 2.0, "euclidean"),
        CellSpec("s2_multiscale_rare_t5", "suite2", "multiscale_rare", "multiscale_rare_outliers", 100_000, 128, 512, 5.0, "euclidean"),
        CellSpec("s2_low_nk32_gmm_t2", "suite2", "low_nk32_gmm", "balanced_gmm", 65_536, 128, 2048, 2.0, "euclidean"),
        CellSpec("s2_lowd_zipf_d8_t2", "suite2", "lowd_zipf_d8", "zipf_gmm", 100_000, 8, 512, 2.0, "euclidean"),
        CellSpec("s2_cifar10_r18_512_t2", "suite2", "cifar10_r18_512", "cifar10_global_512", 50_000, 512, 1024, 2.0, "euclidean"),
        CellSpec("s2_cifar10_r18_512_t5", "suite2", "cifar10_r18_512", "cifar10_global_512", 50_000, 512, 1024, 5.0, "euclidean"),
        CellSpec("s2_cifar10_patches_128_t2", "suite2", "cifar10_patches_128", "cifar10_patches_128", 200_000, 128, 2048, 2.0, "euclidean"),
    ]

    all_cells = suite1_cells + suite2_cells
    if suite_filter != "all":
        all_cells = [c for c in all_cells if c.suite == suite_filter]
    return all_cells


def _estimate_cell_work(cell: CellSpec, n_s2_arms: int) -> float:
    """Deterministic static cost heuristic for balanced multi-GPU shard partitioning."""
    n_arms = len(SUITE1_ARMS) if cell.suite == "suite1" else n_s2_arms
    eff_epochs = cell.nepochs + (2.8 if cell.suite == "suite1" else 0.0)
    low_d_mult = 3.5 if cell.d <= 16 else 1.0
    return (
        float(n_arms)
        * (float(cell.n_train) / 100_000.0)
        * (float(max(cell.d, 64)) / 128.0)
        * (float(cell.k) / 1024.0)
        * eff_epochs
        * low_d_mult
    )


def partition_cells_across_shards(cells: list[CellSpec], num_shards: int) -> list[list[CellSpec]]:
    """Deterministically partition `cells` into `num_shards` disjoint load-balanced buckets.

    Crucial de-confounding invariant: all `(T=1, 2, 5)` Suite 1 cells sharing a `regime_name`
    are constrained to shards within the SAME Blackwell TDP SKU tier (`speed_weight`), so every
    `T=1 -> 2 -> 5` Pareto curve and `Lloyd@25` reference in `smeans_vs_super_pareto.png` is
    100% single-SKU matched while still parallelizing `T=1, 2, 5` across GPUs of that tier.
    """
    if num_shards <= 1:
        return [list(cells)]

    n_s2 = len(get_suite2_arm_specs())
    speed_weights = (
        [w for _, _, _, w in FLEET_SHARDS]
        if num_shards == len(FLEET_SHARDS)
        else [1.0] * num_shards
    )
    buckets: list[list[CellSpec]] = [[] for _ in range(num_shards)]
    loads = [0.0] * num_shards

    # Group shards by identical hardware TDP tier (`speed_weight`)
    tiers: dict[float, list[int]] = {}
    for s_idx, sw in enumerate(speed_weights):
        tiers.setdefault(sw, []).append(s_idx)

    # 1. Assign Suite 1 regimes so all `T` budgets of a regime stay on one TDP SKU tier
    s1_by_regime: dict[str, list[CellSpec]] = {}
    s2_cells: list[CellSpec] = []
    for c in cells:
        if c.suite == "suite1":
            s1_by_regime.setdefault(c.regime_name, []).append(c)
        else:
            s2_cells.append(c)

    ranked_regimes = sorted(
        s1_by_regime.items(),
        key=lambda kv: (-sum(_estimate_cell_work(x, n_s2) for x in kv[1]), kv[0]),
    )
    for _, reg_cells in ranked_regimes:
        reg_sorted = sorted(reg_cells, key=lambda c: (-_estimate_cell_work(c, n_s2), c.cell_id))
        best_tier_key: float | None = None
        best_sim_peak = float("inf")
        best_sim_mean = float("inf")

        for sw_key, tier_shards in sorted(tiers.items(), key=lambda kv: -kv[0]):
            sim_loads = {s: loads[s] for s in tier_shards}
            for rc in reg_sorted:
                w = _estimate_cell_work(rc, n_s2)
                s_pick = min(tier_shards, key=lambda s: ((sim_loads[s] + w) / speed_weights[s], s))
                sim_loads[s_pick] += w
            peak_norm = max(sim_loads[s] / speed_weights[s] for s in tier_shards)
            mean_norm = sum(sim_loads[s] for s in tier_shards) / sum(speed_weights[s] for s in tier_shards)
            if (peak_norm, mean_norm) < (best_sim_peak, best_sim_mean):
                best_sim_peak = peak_norm
                best_sim_mean = mean_norm
                best_tier_key = sw_key

        assert best_tier_key is not None
        chosen_shards = tiers[best_tier_key]
        for rc in reg_sorted:
            w = _estimate_cell_work(rc, n_s2)
            s_pick = min(chosen_shards, key=lambda s: ((loads[s] + w) / speed_weights[s], s))
            buckets[s_pick].append(rc)
            loads[s_pick] += w

    # 2. Pack Suite 2 cells greedily across all shards to equalize completion time
    ranked_s2 = sorted(s2_cells, key=lambda c: (-_estimate_cell_work(c, n_s2), c.cell_id))
    for cell in ranked_s2:
        w = _estimate_cell_work(cell, n_s2)
        best_s = min(range(num_shards), key=lambda s: ((loads[s] + w) / speed_weights[s], s))
        buckets[best_s].append(cell)
        loads[best_s] += w

    return buckets


# ---------------------------------------------------------------------------
# Dataset Loading & Exact FP64 Metrics (Euclidean MSE + Spherical Cosine)
# ---------------------------------------------------------------------------


def _resolve_cifar_dir() -> Path:
    if DEFAULT_CIFAR10_ROOT.exists():
        return DEFAULT_CIFAR10_ROOT
    if REMOTE_CIFAR_FALLBACK.exists():
        return REMOTE_CIFAR_FALLBACK
    return DEFAULT_CIFAR10_ROOT


def _l2_normalize_rows_independent(arr: np.ndarray) -> np.ndarray:
    """Normalize each row independently to unit L2 norm in FP64 (zero cross-row/split leakage)."""
    arr64 = np.asarray(arr, dtype=np.float64)
    norms = np.maximum(np.linalg.norm(arr64, axis=1, keepdims=True), 1e-12)
    return np.ascontiguousarray(arr64 / norms, dtype=np.float32)


def load_cell_dataset(cell: CellSpec, device: torch.device) -> DatasetSplit:
    """Load strictly isolated train / val / test split for `cell`."""
    local_cache = ROOT / "outputs" / "phase1" / "cache"
    cache_dir = local_cache if local_cache.exists() else DEFAULT_CACHE_DIR

    if cell.dataset_kind in ("cifar10_global_512", "cifar10_global_512_unit"):
        ds = load_cifar10_resnet18_embeddings(
            mode="global_512",
            k_recommended=cell.k,
            cifar_dir=_resolve_cifar_dir(),
            cache_dir=cache_dir,
            device=str(device),
        )
    elif cell.dataset_kind == "cifar10_patches_128":
        ds = load_cifar10_resnet18_embeddings(
            mode="patches_128",
            k_recommended=cell.k,
            cifar_dir=_resolve_cifar_dir(),
            cache_dir=cache_dir,
            device=str(device),
        )
    else:
        ds = make_synthetic_dataset(
            kind=cell.dataset_kind,
            n_train=cell.n_train,
            n_val=10_000,
            n_test=10_000,
            d=cell.d,
            k=cell.k,
            seed=42,
        )

    if cell.geometry == "unit_sphere":
        ds = DatasetSplit(
            name=f"{ds.name}_unit",
            train=_l2_normalize_rows_independent(ds.train),
            val=_l2_normalize_rows_independent(ds.val),
            test=_l2_normalize_rows_independent(ds.test),
            d=ds.d,
            k_recommended=ds.k_recommended,
            true_centers=(
                _l2_normalize_rows_independent(ds.true_centers)
                if ds.true_centers is not None
                else None
            ),
            true_probs=ds.true_probs,
            rare_mode_indices=ds.rare_mode_indices,
            metadata={**ds.metadata, "geometry": "unit_sphere"},
        )
    return ds


@torch.no_grad()
def exact_cosine_distortion_fp64(
    x: torch.Tensor,
    centroids: torch.Tensor,
    chunk_rows: int = 8192,
) -> float:
    """Compute exact FP64 spherical cosine distortion `mean_i (1 - max_j cos(x_i, c_j))`."""
    n = int(x.shape[0])
    if n == 0:
        return 0.0
    c64 = centroids.detach().to(device=x.device, dtype=torch.float64)
    c_norms = c64.norm(dim=1, keepdim=True)
    c_unit = torch.where(c_norms > 1e-12, c64 / c_norms.clamp_min(1e-12), torch.zeros_like(c64))
    c_unit_t = c_unit.t().contiguous()

    total_dist = torch.zeros((), dtype=torch.float64, device=x.device)
    for start in range(0, n, chunk_rows):
        end = min(start + chunk_rows, n)
        xb = x[start:end].to(dtype=torch.float64)
        xb_norms = xb.norm(dim=1, keepdim=True)
        xb_unit = torch.where(xb_norms > 1e-12, xb / xb_norms.clamp_min(1e-12), torch.zeros_like(xb))
        sims = xb_unit @ c_unit_t
        max_sim = sims.max(dim=1).values.clamp(-1.0, 1.0)
        total_dist += (1.0 - max_sim).sum()

    return float((total_dist / float(n)).item())


def infer_gpu_sku_tag(host_hint: str, gpu_name: str) -> str:
    """Classify Blackwell TDP SKU cleanly for stratified runtime reporting."""
    gn = gpu_name.lower()
    if "server" in gn or "acorn" in host_hint or "cashew" in host_hint:
        return "Blackwell_Server_600W"
    if "max-q" in gn or host_hint in ("foureyes", "ripper", "loki"):
        return "Blackwell_MaxQ_300W"
    if "workstation" in gn or host_hint == "work":
        return "Blackwell_Workstation_450W"
    return gpu_name


# ---------------------------------------------------------------------------
# Arm Instantiation & Exact-Shape Untimed GPU/Triton Warmup
# ---------------------------------------------------------------------------


def instantiate_arm_model(arm_id: str, suite: str, cell: CellSpec, seed: int) -> Any:
    """Instantiate the exact solver corresponding to `(suite, arm_id)` for `(cell, seed)`."""
    d, k, nepochs = cell.d, cell.k, cell.nepochs
    if suite == "suite1":
        if arm_id == "lloyd_25":
            return LloydKMeans(d=d, k=k, niter=25, init="random", seed=seed)
        if arm_id == "vanilla_emeans":
            return EMeans(d=d, k=k, nepochs=nepochs, seed=seed)
        if arm_id == "emeans_pp":
            return EMeansPP(d=d, k=k, nepochs=nepochs, seed=seed)
        if arm_id == "smeans_exact_ref":
            return SMeans(d=d, k=k, nepochs=nepochs, exact_ref=True, seed=seed)
        if arm_id == "smeans_fast":
            return SMeans(d=d, k=k, nepochs=nepochs, exact_ref=False, seed=seed)
        if arm_id == "super_emeans":
            return SuperEMeans(d=d, k=k, nepochs=nepochs, wide_step_mult=1, seed=seed)
        if arm_id == "super_emeans_wide":
            return SuperEMeans(d=d, k=k, nepochs=nepochs, wide_step_mult=2, seed=seed)
        if arm_id == "hybrid_super_smeans":
            return build_hybrid_arm("hybrid_super_smeans", d=d, k=k, nepochs=nepochs, seed=seed)
        if arm_id == "hybrid_super_smeans_wide":
            return build_hybrid_arm("hybrid_super_smeans_wide", d=d, k=k, nepochs=nepochs, seed=seed)
        raise ValueError(f"Unknown Suite 1 arm_id={arm_id!r}")

    if arm_id.startswith("s2a_"):
        cfg_key = arm_id[len("s2a_"):]
        cfg_kwargs = dict(SMEANS_ABLATION_CONFIGS[cfg_key])
        return SMeans(d=d, k=k, nepochs=nepochs, exact_ref=False, seed=seed, **cfg_kwargs)
    if arm_id.startswith("s2bc_"):
        hyb_key = arm_id[len("s2bc_"):]
        return build_hybrid_arm(hyb_key, d=d, k=k, nepochs=nepochs, seed=seed)
    raise ValueError(f"Unknown Suite 2 arm_id={arm_id!r}")


@torch.no_grad()
def warmup_global_kernels(device: torch.device) -> None:
    """Warm up global Triton / cuBLAS / PyTorch specializations outside all benchmark timers."""
    if device.type != "cuda":
        return

    # Pre-warm all seed-dependent integer-alignment buckets (`==1`, `%16==0`, `%16!=0`) of
    # `seeding._local_greedy_kpp_kernel` so `torch.unique` pool sizes `P` never trigger PTXAS JIT mid-sweep.
    local_kpp_fn = getattr(emeans_seeding, "_local_greedy_kpp_kernel", None)
    if HAS_TRITON and local_kpp_fn is not None:
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

    x_w = torch.randn((32_768, 128), device=device, dtype=torch.float32)
    _ = EMeans(d=128, k=1024, nepochs=1.0, seed=0).fit(x_w, capture_assignments=False)
    _ = EMeansPP(d=128, k=1024, nepochs=1.0, seed=0).fit(x_w, capture_assignments=False)
    _ = SMeans(d=128, k=1024, nepochs=1.0, exact_ref=True, seed=0).fit(x_w, capture_assignments=False)
    _ = SMeans(d=128, k=128, nepochs=1.0, exact_ref=False, seed=0).fit(x_w[:8192], capture_assignments=False)
    c_w, _ = SuperEMeans(d=128, k=1024, nepochs=2.0, wide_step_mult=2, seed=0).fit(
        x_w, capture_assignments=False
    )
    _ = assign_points(x_w[:1024], c_w)
    _ = exact_mse_fp64(x_w[:1024], c_w, device=device)
    torch.cuda.synchronize(device)


@torch.no_grad()
def warmup_cell_kernels(
    x_tr: torch.Tensor,
    x_te: torch.Tensor,
    cell: CellSpec,
    device: torch.device,
) -> None:
    """Warm up exact `(N, d, k, nepochs)` kernels on full `x_tr` so Seed 0 never pays cold JIT."""
    if device.type != "cuda":
        return

    # Pre-warm `_fused_assign_kernel` on coreset shape `m_sub x cell.d` across `K==1`, `%16==0`, and `%16!=0`
    # for both `BLOCK_K=64` and `BLOCK_K=128` (`fp32` & `bf16`) used inside `block_greedy_kmeans_plusplus_torch`.
    m_sub = min(int(x_tr.shape[0]), max(8192, 16 * cell.k, cell.k * 2))
    x_m = x_tr[:m_sub].float().contiguous()
    for dt in (torch.float32, torch.bfloat16):
        for k_cand in (1, 16, 17, 64, 65, 128, 129):
            if k_cand <= m_sub:
                _ = assign_points(x_m, x_m[:k_cand], compute_dtype=dt)

    warm_models: list[Any] = [
        EMeans(d=cell.d, k=cell.k, nepochs=cell.nepochs, seed=999_991),
        EMeansPP(d=cell.d, k=cell.k, nepochs=cell.nepochs, seed=999_992),
        SMeans(d=cell.d, k=cell.k, nepochs=cell.nepochs, exact_ref=False, seed=999_993),
        SuperEMeans(d=cell.d, k=cell.k, nepochs=cell.nepochs, wide_step_mult=1, seed=999_994),
        SuperEMeans(d=cell.d, k=cell.k, nepochs=cell.nepochs, wide_step_mult=2, seed=999_995),
    ]
    if cell.suite == "suite1":
        warm_models.append(LloydKMeans(d=cell.d, k=cell.k, niter=2, init="random", seed=999_996))
        warm_models.append(SMeans(d=cell.d, k=cell.k, nepochs=cell.nepochs, exact_ref=True, seed=999_997))
        if "hybrid_super_smeans_wide" in HYBRID_ABLATION_ARMS:
            warm_models.append(
                build_hybrid_arm(
                    "hybrid_super_smeans_wide",
                    d=cell.d,
                    k=cell.k,
                    nepochs=cell.nepochs,
                    seed=999_998,
                )
            )
    else:
        # Warm SPLIT_MODE=1 (`checkerboard`) and `rms_iso` (`use_feat_scale=False`) so zero JIT leaks into LOO arms
        warm_models.append(
            SMeans(
                d=cell.d,
                k=cell.k,
                nepochs=cell.nepochs,
                exact_ref=False,
                split_mode="checkerboard",
                use_feat_scale=False,
                seed=999_989,
            )
        )
        warm_models.append(
            SMeans(
                d=cell.d,
                k=cell.k,
                nepochs=cell.nepochs,
                exact_ref=False,
                split_mode="rms_iso",
                use_feat_scale=False,
                seed=999_990,
            )
        )
        for hk in (
            "smeans_deconfounded",
            "smeans_plus_p6_wide_step",
            "pp_plus_c5_rms_aniso_split",
            "hybrid_super_smeans_wide",
        ):
            if hk in HYBRID_ABLATION_ARMS:
                warm_models.append(
                    build_hybrid_arm(hk, d=cell.d, k=cell.k, nepochs=cell.nepochs, seed=999_998)
                )

    c_last: torch.Tensor | None = None
    for wm in warm_models:
        c_last, _ = wm.fit(x_tr, capture_assignments=False)

    if c_last is not None:
        _ = assign_points(x_tr, c_last)
        _ = exact_mse_fp64(x_tr, c_last, device=device)
        _ = exact_mse_fp64(x_te, c_last, device=device)
        if cell.geometry == "unit_sphere":
            _ = exact_cosine_distortion_fp64(x_te, c_last)
    torch.cuda.synchronize(device)


# ---------------------------------------------------------------------------
# Per-Shard Benchmark Runner
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_cell_shard(
    cells: list[CellSpec],
    seeds: list[int],
    device: torch.device,
) -> list[dict[str, Any]]:
    """Execute all `(cell, seed, arm)` runs assigned to this shard on a single physical GPU."""
    if not cells:
        return []

    warmup_global_kernels(device)
    hostname = socket.gethostname()
    vis_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    sku_tag = infer_gpu_sku_tag(hostname, gpu_name)
    s2_specs = get_suite2_arm_specs()
    records: list[dict[str, Any]] = []

    for cell in cells:
        t_cell0 = time.perf_counter()
        ds = load_cell_dataset(cell, device=device)
        x_tr = torch.from_numpy(ds.train).to(device=device, dtype=torch.float32).contiguous()
        x_te = torch.from_numpy(ds.test).to(device=device, dtype=torch.float32).contiguous()
        n_tr_total = int(x_tr.shape[0])
        n_te_total = int(x_te.shape[0])

        if x_tr.shape[1] != cell.d or x_te.shape[1] != cell.d:
            raise ValueError(
                f"Dimension mismatch in cell {cell.cell_id}: expected d={cell.d}, got {x_tr.shape}, {x_te.shape}"
            )
        if not (bool(torch.isfinite(x_tr).all()) and bool(torch.isfinite(x_te).all())):
            raise ValueError(f"Non-finite input values detected in cell {cell.cell_id}")

        # Full-shape untimed warmup outside all benchmark timers
        warmup_cell_kernels(x_tr, x_te, cell, device=device)

        arm_ids: list[str] = (
            list(SUITE1_ARMS)
            if cell.suite == "suite1"
            else [aid for aid, _, _ in s2_specs]
        )

        rare_c: torch.Tensor | None = None
        rare_cutoff: float = float(cell.d) * (0.35**2)
        if ds.true_centers is not None and ds.rare_mode_indices is not None and len(ds.rare_mode_indices) > 0:
            rare_c = torch.from_numpy(ds.true_centers[ds.rare_mode_indices]).to(
                device=device, dtype=torch.float32
            )

        print(
            f"[Phase3-SMeans] Starting {cell.cell_id} ({cell.suite}, n={n_tr_total}, d={cell.d}, "
            f"k={cell.k}, T={cell.nepochs}, seeds={len(seeds)}, arms={len(arm_ids)}) on "
            f"{hostname}:GPU{vis_gpu} ({sku_tag})",
            flush=True,
        )

        for s_idx, seed in enumerate(seeds):
            # Cyclic arm-order counterbalancing across seeds eliminates within-seed thermal drift bias
            rot = s_idx % len(arm_ids)
            rotated_arms = arm_ids[rot:] + arm_ids[:rot]
            seed_out_by_arm: dict[str, dict[str, Any]] = {}

            for arm_id in rotated_arms:
                np.random.seed(seed % (2**31 - 1))
                torch.manual_seed(seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(seed)

                model = instantiate_arm_model(arm_id, cell.suite, cell, seed=seed)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                t_fit0 = time.perf_counter()
                centroids, _ = model.fit(x_tr, capture_assignments=False)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                wall_fit_sec = time.perf_counter() - t_fit0

                if not isinstance(centroids, torch.Tensor):
                    centroids = torch.as_tensor(centroids, device=device, dtype=torch.float32)
                if not bool(torch.isfinite(centroids).all()):
                    raise RuntimeError(
                        f"Non-finite centroids produced by arm={arm_id} on cell={cell.cell_id} seed={seed}"
                    )

                init_sec = float(getattr(model, "init_time_sec", 0.0))
                train_sec = float(getattr(model, "train_time_sec", wall_fit_sec))
                total_sec = float(getattr(model, "total_time_sec", init_sec + train_sec))
                if total_sec <= 0.0:
                    total_sec = wall_fit_sec

                train_mse = float(exact_mse_fp64(x_tr, centroids, device=device))
                test_mse = float(exact_mse_fp64(x_te, centroids, device=device))

                _, eval_labels = assign_points(x_tr, centroids)
                eval_counts = torch.bincount(eval_labels, minlength=cell.k).double()
                imbalance = float(
                    (cell.k * torch.dot(eval_counts, eval_counts) / (float(n_tr_total) ** 2)).item()
                )
                dead_lt5 = int((eval_counts < 5).sum().item())
                empty_exact = int((eval_counts == 0).sum().item())

                rec: dict[str, Any] = {
                    "cell_id": cell.cell_id,
                    "suite": cell.suite,
                    "regime_name": cell.regime_name,
                    "dataset": cell.dataset_kind,
                    "geometry": cell.geometry,
                    "host": hostname,
                    "visible_gpu": str(vis_gpu),
                    "gpu_name": gpu_name,
                    "gpu_sku": sku_tag,
                    "n_train": n_tr_total,
                    "n_test": n_te_total,
                    "d": cell.d,
                    "k": cell.k,
                    "n_over_k": float(n_tr_total) / float(cell.k),
                    "nepochs": cell.nepochs,
                    "seed": int(seed),
                    "arm": arm_id,
                    "train_mse": train_mse,
                    "test_mse": test_mse,
                    "init_time_sec": init_sec,
                    "train_time_sec": train_sec,
                    "total_time_sec": total_sec,
                    "wall_fit_time_sec": float(wall_fit_sec),
                    "n_splits": int(getattr(model, "n_splits", 0)),
                    "n_protected": int(getattr(model, "n_protected", 0)),
                    "n_transplants": int(getattr(model, "n_transplants", 0)),
                    "imbalance_factor": imbalance,
                    "dead_clusters_lt5": dead_lt5,
                    "empty_clusters": empty_exact,
                }

                if cell.geometry == "unit_sphere":
                    rec["train_cos_dist"] = exact_cosine_distortion_fp64(x_tr, centroids)
                    rec["test_cos_dist"] = exact_cosine_distortion_fp64(x_te, centroids)

                if rare_c is not None:
                    d_rare, _ = assign_points(rare_c, centroids)
                    rec["rare_preserved_frac"] = float((d_rare <= rare_cutoff).float().mean().item())

                seed_out_by_arm[arm_id] = rec

            # Emit in canonical arm order for clean deterministic JSON output
            for arm_id in arm_ids:
                records.append(seed_out_by_arm[arm_id])

        print(
            f"[Phase3-SMeans] Finished {cell.cell_id} in {time.perf_counter() - t_cell0:.2f}s",
            flush=True,
        )

    return records


# ---------------------------------------------------------------------------
# Atomic JSON I/O & Strict Shard Completeness / Deduplication Checks
# ---------------------------------------------------------------------------


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON atomically via a PID+ns temp file + `os.replace`, forbidding NaN/Inf."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".tmp_{os.getpid()}_{time.time_ns()}_{path.name}"
    try:
        serialized = json.dumps(payload, indent=2, allow_nan=False)
        tmp_path.write_text(serialized, encoding="utf-8")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def load_and_validate_shards(
    shards_dir: Path,
    num_shards: int,
    expected_cells: list[CellSpec],
    expected_seeds: list[int],
) -> list[dict[str, Any]]:
    """Load exact `records_shard_XX_of_{num_shards:02d}.json` files and verify 100% completeness."""
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")

    s2_arm_ids = [aid for aid, _, _ in get_suite2_arm_specs()]
    expected_keys: set[tuple[str, int, str]] = set()
    for cell in expected_cells:
        arms = list(SUITE1_ARMS) if cell.suite == "suite1" else s2_arm_ids
        for sd in expected_seeds:
            for a in arms:
                expected_keys.add((cell.cell_id, int(sd), a))

    merged: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, int, str]] = set()
    cell_sku_check: dict[str, set[tuple[str, str]]] = {}

    for idx in range(num_shards):
        sf = shards_dir / f"records_shard_{idx:02d}_of_{num_shards:02d}.json"
        if not sf.exists():
            raise FileNotFoundError(f"Missing expected shard file: {sf}")
        part = json.loads(sf.read_text(encoding="utf-8"))
        for rec in part:
            key = (str(rec["cell_id"]), int(rec["seed"]), str(rec["arm"]))
            if key in seen_keys:
                raise RuntimeError(f"Duplicate `(cell_id, seed, arm)` record detected across shards: {key}")
            for num_k in ("train_mse", "test_mse", "init_time_sec", "train_time_sec", "total_time_sec"):
                val = float(rec[num_k])
                if not math.isfinite(val):
                    raise RuntimeError(f"Non-finite {num_k}={val} in record {key}")
            cell_sku_check.setdefault(key[0], set()).add((str(rec["host"]), str(rec["visible_gpu"])))
            seen_keys.add(key)
            merged.append(rec)

    # Verify strict single-GPU / single-SKU isolation per cell AND single TDP SKU across T=1,2,5 per Suite 1 regime
    for cid, host_gpus in cell_sku_check.items():
        if len(host_gpus) != 1:
            raise RuntimeError(f"Cell {cid} mixed multiple GPUs/hosts across arms/seeds: {host_gpus}")

    s1_regime_skus: dict[str, set[str]] = {}
    for rec in merged:
        if rec["suite"] == "suite1":
            s1_regime_skus.setdefault(str(rec["regime_name"]), set()).add(str(rec["gpu_sku"]))
    for reg_name, skus_found in s1_regime_skus.items():
        if len(skus_found) != 1:
            raise RuntimeError(
                f"Suite 1 regime {reg_name!r} mixed heterogeneous GPU TDP SKUs across T budgets: {skus_found}"
            )

    missing_keys = expected_keys - seen_keys
    extra_keys = seen_keys - expected_keys
    if missing_keys or extra_keys:
        raise RuntimeError(
            f"Shard record mismatch: missing={len(missing_keys)} (e.g. {sorted(missing_keys)[:3]}), "
            f"unexpected={len(extra_keys)} (e.g. {sorted(extra_keys)[:3]})"
        )

    return merged


# ---------------------------------------------------------------------------
# Paired Statistical Comparisons, Summary Synthesis & Publication-Grade Plots
# ---------------------------------------------------------------------------


def compute_paired_comparison(
    records: list[dict[str, Any]],
    cell_id: str,
    arm: str,
    baseline_arm: str,
    metric: str = "test_mse",
) -> dict[str, Any]:
    """Compute paired per-seed relative delta (%), speedup ratio, Student-t p, and Wilcoxon p."""
    arm_by_seed = {
        int(r["seed"]): float(r[metric])
        for r in records
        if r["cell_id"] == cell_id and r["arm"] == arm and metric in r
    }
    base_by_seed = {
        int(r["seed"]): float(r[metric])
        for r in records
        if r["cell_id"] == cell_id and r["arm"] == baseline_arm and metric in r
    }
    common_seeds = sorted(set(arm_by_seed.keys()) & set(base_by_seed.keys()))
    if not common_seeds:
        return {}

    a_vals = np.array([arm_by_seed[s] for s in common_seeds], dtype=np.float64)
    b_vals = np.array([base_by_seed[s] for s in common_seeds], dtype=np.float64)
    rel_pct = 100.0 * (a_vals - b_vals) / np.maximum(np.abs(b_vals), 1e-30)
    ratios = a_vals / np.maximum(np.abs(b_vals), 1e-30)
    diffs = a_vals - b_vals

    n_seeds = len(common_seeds)
    wins = int(np.sum(diffs < -1e-12))
    ties = int(np.sum(np.abs(diffs) <= 1e-12))
    losses = int(np.sum(diffs > 1e-12))

    if n_seeds >= 2 and float(np.std(diffs, ddof=1)) > 1e-15:
        _, p_ttest = stats.ttest_rel(a_vals, b_vals)
        p_ttest = float(p_ttest)
    else:
        p_ttest = 1.0

    nonzero_diffs = diffs[np.abs(diffs) > 1e-15]
    if len(nonzero_diffs) >= 5:
        try:
            _, p_wilcoxon = stats.wilcoxon(nonzero_diffs)
            p_wilcoxon = float(p_wilcoxon)
        except ValueError:
            p_wilcoxon = 1.0
    else:
        p_wilcoxon = p_ttest

    return {
        "cell_id": cell_id,
        "arm": arm,
        "baseline": baseline_arm,
        "metric": metric,
        "n_seeds": n_seeds,
        "arm_mean": float(np.mean(a_vals)),
        "base_mean": float(np.mean(b_vals)),
        "mean_delta_pct": float(np.mean(rel_pct)),
        "median_delta_pct": float(np.median(rel_pct)),
        "std_delta_pct": float(np.std(rel_pct, ddof=1)) if n_seeds >= 2 else 0.0,
        "mean_ratio": float(np.mean(ratios)),
        "median_ratio": float(np.median(ratios)),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "p_ttest": p_ttest,
        "p_wilcoxon": p_wilcoxon,
    }


def aggregate_across_cells(
    paired_rows: list[dict[str, Any]],
    arm: str,
    baseline: str,
    metric: str = "test_mse",
    cell_prefix: str | None = None,
    nepochs_filter: float | None = None,
    cell_nepochs_map: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Summarize cell-level paired deltas, win/tie/loss counts, and meta-Wilcoxon across cells."""
    sub = [
        r for r in paired_rows
        if r["arm"] == arm
        and r["baseline"] == baseline
        and r["metric"] == metric
        and (cell_prefix is None or str(r["cell_id"]).startswith(cell_prefix))
        and (
            nepochs_filter is None
            or (cell_nepochs_map is not None and abs(cell_nepochs_map.get(r["cell_id"], -1.0) - nepochs_filter) < 1e-6)
        )
    ]
    if not sub:
        return {}
    deltas = np.array([r["mean_delta_pct"] for r in sub], dtype=np.float64)
    ratios = np.array([r["mean_ratio"] for r in sub], dtype=np.float64)
    cell_wins = int(np.sum(deltas < -1e-9))
    cell_ties = int(np.sum(np.abs(deltas) <= 1e-9))
    cell_losses = int(np.sum(deltas > 1e-9))
    sig_wins = int(sum(1 for r in sub if r["mean_delta_pct"] < 0 and r["p_wilcoxon"] < 0.05))
    sig_losses = int(sum(1 for r in sub if r["mean_delta_pct"] > 0 and r["p_wilcoxon"] < 0.05))

    if len(deltas) >= 2 and float(np.std(deltas, ddof=1)) > 1e-15:
        _, p_meta_t = stats.ttest_1samp(deltas, 0.0)
        p_meta_t = float(p_meta_t)
    else:
        p_meta_t = 1.0

    nz = deltas[np.abs(deltas) > 1e-15]
    if len(nz) >= 5:
        try:
            _, p_meta_w = stats.wilcoxon(nz)
            p_meta_w = float(p_meta_w)
        except ValueError:
            p_meta_w = 1.0
    else:
        p_meta_w = p_meta_t

    return {
        "arm": arm,
        "baseline": baseline,
        "metric": metric,
        "n_cells": len(sub),
        "mean_delta_pct": float(np.mean(deltas)),
        "median_delta_pct": float(np.median(deltas)),
        "min_delta_pct": float(np.min(deltas)),
        "max_delta_pct": float(np.max(deltas)),
        "mean_ratio": float(np.mean(ratios)),
        "median_ratio": float(np.median(ratios)),
        "cell_wins": cell_wins,
        "cell_ties": cell_ties,
        "cell_losses": cell_losses,
        "sig_cell_wins_p05": sig_wins,
        "sig_cell_losses_p05": sig_losses,
        "seed_wins": int(sum(r["wins"] for r in sub)),
        "seed_ties": int(sum(r["ties"] for r in sub)),
        "seed_losses": int(sum(r["losses"] for r in sub)),
        "p_cell_ttest": p_meta_t,
        "p_cell_wilcoxon": p_meta_w,
    }


def _save_fig_dual(fig: plt.Figure, out_dir: Path, filename: str) -> None:
    """Save publication-grade figure directly under `out_dir/filename` and `out_dir/plots/filename`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / filename, dpi=200, bbox_inches="tight")
    fig.savefig(plots_dir / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_plots(
    records: list[dict[str, Any]],
    s1_paired: list[dict[str, Any]],
    s2_paired: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    """Generate the 3 publication-quality Phase 3 PNG plots."""
    s1_recs = [r for r in records if r["suite"] == "suite1"]
    s2_recs = [r for r in records if r["suite"] == "suite2"]

    # ------------------------------------------------------------------
    # Plot 1: `smeans_vs_super_pareto.png` — Speed vs Test MSE Pareto (`T=1, 2, 5`)
    # ------------------------------------------------------------------
    s1_regimes_present = []
    for reg in (
        "zipf_gmm",
        "multiscale_rare",
        "low_nk32_gmm",
        "stretched_a2",
        "cifar10_r18_512",
        "cifar10_patches_128",
    ):
        if any(r["regime_name"] == reg for r in s1_recs):
            s1_regimes_present.append(reg)
    if not s1_regimes_present:
        s1_regimes_present = sorted({r["regime_name"] for r in s1_recs})[:6]

    if s1_regimes_present:
        ncols = min(3, len(s1_regimes_present))
        nrows = math.ceil(len(s1_regimes_present) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.1 * nrows), squeeze=False)
        arm_styles = [
            ("vanilla_emeans", "Vanilla e-means", "#7f7f7f", "s", "--"),
            ("emeans_pp", "EMeansPP", "#1f77b4", "o", "-."),
            ("smeans_exact_ref", "s-means (upstream ref)", "#ff7f0e", "^", ":"),
            ("smeans_fast", "s-means (fast GPU)", "#d62728", "v", "-"),
            ("super_emeans", "SuperEMeans (ours)", "#2ca02c", "D", "-"),
            ("super_emeans_wide", "SuperEMeans + WideStep", "#17becf", "P", "--"),
            ("hybrid_super_smeans", "Hybrid (s-means + Super)", "#9467bd", "*", "-"),
            ("hybrid_super_smeans_wide", "Hybrid + WideStep", "#e377c2", "X", "--"),
        ]
        for idx, reg in enumerate(s1_regimes_present):
            ax = axes[idx // ncols][idx % ncols]
            reg_sub = [r for r in s1_recs if r["regime_name"] == reg]
            reg_skus = sorted({str(r["gpu_sku"]).replace("Blackwell_", "") for r in reg_sub})
            sku_lbl = "/".join(reg_skus)

            # Plot Lloyd@25 reference point (matched single-SKU across all cells of this regime)
            lloyd_sub = [r for r in reg_sub if r["arm"] == "lloyd_25"]
            if lloyd_sub:
                ax.scatter(
                    [1000.0 * float(np.mean([r["total_time_sec"] for r in lloyd_sub]))],
                    [float(np.mean([r["test_mse"] for r in lloyd_sub]))],
                    label="Lloyd@25 (random)",
                    color="#111111",
                    marker="h",
                    s=85,
                    zorder=5,
                )

            for arm_k, lbl, col, mk, ls in arm_styles:
                pts = []
                for t_val in sorted({r["nepochs"] for r in reg_sub if r["arm"] == arm_k}):
                    sub_t = [r for r in reg_sub if r["arm"] == arm_k and r["nepochs"] == t_val]
                    if sub_t:
                        pts.append(
                            (
                                1000.0 * float(np.mean([r["total_time_sec"] for r in sub_t])),
                                float(np.mean([r["test_mse"] for r in sub_t])),
                                t_val,
                            )
                        )
                if pts:
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    ax.plot(xs, ys, color=col, linestyle=ls, linewidth=1.6, alpha=0.88, zorder=4)
                    ax.scatter(
                        xs,
                        ys,
                        label=lbl,
                        color=col,
                        marker=mk,
                        s=130 if mk == "*" else 68,
                        zorder=6,
                    )

            ax.set_xscale("log")
            ax.set_title(f"{reg} [{sku_lbl}]", fontsize=10.2, fontweight="bold")
            ax.set_xlabel("Total End-to-End GPU Latency (ms, log scale)")
            ax.set_ylabel("Exact FP64 Test MSE (lower is better)")
            ax.grid(True, which="both", alpha=0.28)

        for extra_i in range(len(s1_regimes_present), nrows * ncols):
            axes[extra_i // ncols][extra_i % ncols].axis("off")

        handles, labels = axes[0][0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                ncol=min(5, len(labels)),
                fontsize=8.5,
                frameon=True,
                bbox_to_anchor=(0.5, -0.02),
            )
        fig.suptitle(
            "Phase 3 Showdown: s-means vs. SuperEMeans vs. Hybrids Speed–Test MSE Pareto Frontier",
            fontsize=12.5,
            fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0.05, 1, 0.96])
        _save_fig_dual(fig, out_dir, "smeans_vs_super_pareto.png")

    # ------------------------------------------------------------------
    # Plot 2: `smeans_ablation_heatmap.png` — Suite 2(a) LOO & Suite 2(b,c) Grafting Heatmap
    # ------------------------------------------------------------------
    s2_cells = sorted({r["cell_id"] for r in s2_recs})
    if s2_cells:
        heatmap_rows: list[tuple[str, str, str]] = [
            # 2(a) LOO vs s2a_smeans_full (positive = component helped SMeans; negative = removing improved!)
            ("s2a_loo_no_d15_init", "s2a_smeans_full", "LOO SMeans: -C1 D^3 Init"),
            ("s2a_loo_no_power_cos", "s2a_smeans_full", "LOO SMeans: -C2 PowerCos"),
            ("s2a_loo_no_revive_stop", "s2a_smeans_full", "LOO SMeans: -C3 Stop@85%"),
            ("s2a_loo_no_sse_donor", "s2a_smeans_full", "LOO SMeans: -C4 SSE*N^0.35"),
            ("s2a_loo_no_rms_split", "s2a_smeans_full", "LOO SMeans: -C5 RMS Split"),
            ("s2a_loo_no_feat_scale", "s2a_smeans_full", "LOO SMeans: -C5b FeatScale"),
            ("s2a_loo_no_pass_weight", "s2a_smeans_full", "LOO SMeans: -C6 PassRamp"),
            ("s2a_loo_no_eb_finalize", "s2a_smeans_full", "LOO SMeans: -C7 EB(0.2)"),
            # 2(b) Grafting s-means C_i onto SuperEMeans (negative = beats pure SuperEMeans!)
            ("s2bc_super_plus_c1_d15_init", "s2bc_super_emeans", "Super + C1(D^3 Init)"),
            ("s2bc_super_plus_c2_power_cos", "s2bc_super_emeans", "Super + C2(PowerCos)"),
            ("s2bc_super_plus_c3_revive_stop", "s2bc_super_emeans", "Super + C3(Stop@85%)"),
            ("s2bc_super_plus_c4_sse_usage_donor", "s2bc_super_emeans", "Super + C4(SSE*N^0.35)"),
            ("s2bc_super_plus_c5_rms_aniso_split", "s2bc_super_emeans", "Super + C5(Rademacher RMS)"),
            ("s2bc_super_plus_c5b_oja_rms_split", "s2bc_super_emeans", "Super + C5b(Oja 0.55*RMS)"),
            ("s2bc_super_plus_c6_pass_weight", "s2bc_super_emeans", "Super + C6(PassRamp)"),
            ("s2bc_super_plus_c7_eb_ema", "s2bc_super_emeans", "Super + C7(EB tau=0.2)"),
            # 2(c) Grafting SuperEMeans P_i onto SMeans (negative = improves SMeans!)
            ("s2bc_smeans_plus_p1_coreset_voronoi", "s2bc_smeans_deconfounded", "SMeans + P1(Coreset+Voronoi)"),
            ("s2bc_smeans_plus_p2_huygens_shield", "s2bc_smeans_deconfounded", "SMeans + P2(Huygens Shield)"),
            ("s2bc_smeans_plus_p3_oja_split", "s2bc_smeans_deconfounded", "SMeans + P3(Oja Axis)"),
            ("s2bc_smeans_plus_p4_eb_js", "s2bc_smeans_deconfounded", "SMeans + P4(EB-JS auto)"),
            ("s2bc_smeans_plus_p5_head_edit", "s2bc_smeans_deconfounded", "SMeans + P5(P/Q head_edit)"),
            ("s2bc_smeans_plus_p6_wide_step", "s2bc_smeans_deconfounded", "SMeans + P6(WideStep 2x)"),
            ("s2bc_hybrid_super_smeans", "s2bc_super_emeans", "Hybrid vs SuperEMeans"),
            ("s2bc_hybrid_super_smeans", "s2bc_smeans_deconfounded", "Hybrid vs SMeans"),
        ]
        present_rows = [
            row for row in heatmap_rows
            if any(s["arm"] == row[0] and s["baseline"] == row[1] for s in s2_paired)
        ]
        if present_rows:
            mat = np.zeros((len(present_rows), len(s2_cells)), dtype=np.float64)
            pmat = np.ones((len(present_rows), len(s2_cells)), dtype=np.float64)
            lookup = {
                (s["cell_id"], s["arm"], s["baseline"], s["metric"]): s
                for s in s2_paired
            }
            for i, (ak, bk, _) in enumerate(present_rows):
                for j, cid in enumerate(s2_cells):
                    st = lookup.get((cid, ak, bk, "test_mse"))
                    if st is not None:
                        mat[i, j] = st["mean_delta_pct"]
                        pmat[i, j] = st["p_wilcoxon"]

            fig, ax = plt.subplots(
                figsize=(max(10.5, 0.95 * len(s2_cells)), max(7.5, 0.36 * len(present_rows)))
            )
            vmax = max(1.0, min(15.0, float(np.percentile(np.abs(mat), 92))))
            im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
            ax.set_xticks(range(len(s2_cells)))
            ax.set_xticklabels([c.replace("s2_", "") for c in s2_cells], rotation=32, ha="right", fontsize=8.5)
            ax.set_yticks(range(len(present_rows)))
            ax.set_yticklabels([lbl for _, _, lbl in present_rows], fontsize=8.5)
            for i in range(len(present_rows)):
                for j in range(len(s2_cells)):
                    sig = "*" if pmat[i, j] < 0.05 else ""
                    val = mat[i, j]
                    txt = f"{val:+.1f}%{sig}" if abs(val) >= 10.0 else f"{val:+.2f}%{sig}"
                    ax.text(
                        j,
                        i,
                        txt,
                        ha="center",
                        va="center",
                        fontsize=6.8,
                        color="white" if abs(val) > 0.58 * vmax else "black",
                    )
            fig.colorbar(im, ax=ax, label="Paired Test MSE Delta (%) [Blue < 0 = Lower Error, * p<0.05]")
            ax.set_title(
                "Phase 3 Directional Component Ablation Matrix: LOO SMeans & Bidirectional Grafting",
                fontsize=11.5,
                fontweight="bold",
            )
            fig.tight_layout()
            _save_fig_dual(fig, out_dir, "smeans_ablation_heatmap.png")

    # ------------------------------------------------------------------
    # Plot 3: `component_grafting_bars.png` — 3-Panel Mean Effect Across Suite 2 Regimes
    # ------------------------------------------------------------------
    if s2_paired:
        panels = [
            (
                "A. Leave-One-Out from SMeans\n(>0 means component helped SMeans)",
                [
                    ("s2a_loo_no_rms_split", "s2a_smeans_full", "-C5 RMS Split"),
                    ("s2a_loo_no_sse_donor", "s2a_smeans_full", "-C4 SSE*N^0.35"),
                    ("s2a_loo_no_d15_init", "s2a_smeans_full", "-C1 D^3 Init"),
                    ("s2a_loo_no_revive_stop", "s2a_smeans_full", "-C3 Stop@85%"),
                    ("s2a_loo_no_feat_scale", "s2a_smeans_full", "-C5b FeatScale"),
                    ("s2a_loo_no_power_cos", "s2a_smeans_full", "-C2 PowerCos"),
                    ("s2a_loo_no_pass_weight", "s2a_smeans_full", "-C6 PassRamp"),
                    ("s2a_loo_no_eb_finalize", "s2a_smeans_full", "-C7 EB(0.2)"),
                ],
            ),
            (
                "B. Grafting s-means C_i onto SuperEMeans\n(<0 means improves SuperEMeans!)",
                [
                    ("s2bc_super_plus_c1_d15_init", "s2bc_super_emeans", "+C1 D^3 Init"),
                    ("s2bc_super_plus_c2_power_cos", "s2bc_super_emeans", "+C2 PowerCos"),
                    ("s2bc_super_plus_c3_revive_stop", "s2bc_super_emeans", "+C3 Stop@85%"),
                    ("s2bc_super_plus_c4_sse_usage_donor", "s2bc_super_emeans", "+C4 SSE*N^0.35"),
                    ("s2bc_super_plus_c5_rms_aniso_split", "s2bc_super_emeans", "+C5 Radem RMS"),
                    ("s2bc_super_plus_c5b_oja_rms_split", "s2bc_super_emeans", "+C5b Oja+RMS"),
                    ("s2bc_super_plus_c6_pass_weight", "s2bc_super_emeans", "+C6 PassRamp"),
                    ("s2bc_super_plus_c7_eb_ema", "s2bc_super_emeans", "+C7 EB(0.2)"),
                    ("s2bc_hybrid_super_smeans", "s2bc_super_emeans", "Full Hybrid"),
                ],
            ),
            (
                "C. Grafting SuperEMeans P_i onto SMeans\n(<0 means improves SMeans!)",
                [
                    ("s2bc_smeans_plus_p1_coreset_voronoi", "s2bc_smeans_deconfounded", "+P1 Coreset+Voronoi"),
                    ("s2bc_smeans_plus_p2_huygens_shield", "s2bc_smeans_deconfounded", "+P2 Huygens Shield"),
                    ("s2bc_smeans_plus_p3_oja_split", "s2bc_smeans_deconfounded", "+P3 Oja Axis"),
                    ("s2bc_smeans_plus_p4_eb_js", "s2bc_smeans_deconfounded", "+P4 EB-JS (auto)"),
                    ("s2bc_smeans_plus_p5_head_edit", "s2bc_smeans_deconfounded", "+P5 P/Q head_edit"),
                    ("s2bc_smeans_plus_p6_wide_step", "s2bc_smeans_deconfounded", "+P6 WideStep 2x"),
                    ("s2bc_super_emeans", "s2bc_smeans_deconfounded", "Pure SuperEMeans"),
                    ("s2bc_hybrid_super_smeans", "s2bc_smeans_deconfounded", "Full Hybrid"),
                ],
            ),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.2))
        for ax, (title, items) in zip(axes, panels):
            labels_p: list[str] = []
            means_p: list[float] = []
            errs_p: list[float] = []
            for ak, bk, lbl in items:
                cell_vals = [
                    s["mean_delta_pct"]
                    for s in s2_paired
                    if s["arm"] == ak and s["baseline"] == bk and s["metric"] == "test_mse"
                ]
                if not cell_vals:
                    continue
                arr = np.array(cell_vals, dtype=np.float64)
                labels_p.append(lbl)
                means_p.append(float(np.mean(arr)))
                sem = float(np.std(arr, ddof=1) / math.sqrt(len(arr))) if len(arr) >= 2 else 0.0
                errs_p.append(1.96 * sem)

            if labels_p:
                y_pos = np.arange(len(labels_p))
                colors = ["#2ca02c" if m < 0 else "#d62728" for m in means_p]
                if title.startswith("A."):
                    # In LOO, >0 means removing hurt SMeans (i.e., component was beneficial inside SMeans)
                    colors = ["#1f77b4" if m > 0 else "#ff7f0e" for m in means_p]
                ax.barh(y_pos, means_p, xerr=errs_p, color=colors, alpha=0.86, capsize=3)
                ax.axvline(0.0, color="black", linewidth=0.9, linestyle="--")
                ax.set_yticks(y_pos)
                ax.set_yticklabels(labels_p, fontsize=8.8)
                ax.invert_yaxis()
                ax.set_xlabel("Mean Paired Test MSE Delta (%) ± 95% CI")
                ax.set_title(title, fontsize=10, fontweight="bold")
                ax.grid(True, axis="x", alpha=0.3)

        fig.suptitle(
            "Directional Component Attribution Across Regimes: SMeans vs. SuperEMeans Pillars",
            fontsize=12,
            fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        _save_fig_dual(fig, out_dir, "component_grafting_bars.png")


def analyze_and_plot(records: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    """Compute all Suite 1 & Suite 2 paired statistical tests, synthesize verdict, and generate plots."""
    s1_cells = sorted({r["cell_id"] for r in records if r["suite"] == "suite1"})
    s2_cells = sorted({r["cell_id"] for r in records if r["suite"] == "suite2"})
    cell_nepochs_map = {r["cell_id"]: float(r["nepochs"]) for r in records}

    # Suite 1 key head-to-head paired comparisons
    s1_pairs = [
        ("smeans_fast", "vanilla_emeans"),
        ("smeans_exact_ref", "vanilla_emeans"),
        ("smeans_fast", "smeans_exact_ref"),
        ("emeans_pp", "vanilla_emeans"),
        ("emeans_pp", "smeans_fast"),
        ("super_emeans", "vanilla_emeans"),
        ("super_emeans", "emeans_pp"),
        ("super_emeans", "smeans_fast"),
        ("super_emeans", "smeans_exact_ref"),
        ("super_emeans_wide", "smeans_fast"),
        ("super_emeans_wide", "super_emeans"),
        ("hybrid_super_smeans", "smeans_fast"),
        ("hybrid_super_smeans", "super_emeans"),
        ("hybrid_super_smeans", "emeans_pp"),
        ("hybrid_super_smeans_wide", "smeans_fast"),
        ("hybrid_super_smeans_wide", "super_emeans_wide"),
        ("hybrid_super_smeans_wide", "super_emeans"),
        ("super_emeans", "lloyd_25"),
        ("hybrid_super_smeans", "lloyd_25"),
        ("smeans_fast", "lloyd_25"),
    ]
    s1_paired: list[dict[str, Any]] = []
    for cid in s1_cells:
        metrics_for_cell = ["train_mse", "test_mse", "total_time_sec"]
        if any(r["cell_id"] == cid and "test_cos_dist" in r for r in records):
            metrics_for_cell.extend(["train_cos_dist", "test_cos_dist"])
        for arm, base in s1_pairs:
            for m in metrics_for_cell:
                st = compute_paired_comparison(records, cid, arm, base, metric=m)
                if st:
                    s1_paired.append(st)

    # Suite 2 directional component comparisons
    s2_arm_ids = [aid for aid, _, _ in get_suite2_arm_specs()]
    s2_pairs: list[tuple[str, str]] = []
    for aid in s2_arm_ids:
        if aid.startswith("s2a_") and aid != "s2a_smeans_full":
            s2_pairs.append((aid, "s2a_smeans_full"))
        elif aid.startswith("s2bc_pp_plus_"):
            s2_pairs.append((aid, "s2bc_emeans_pp_base"))
        elif aid.startswith("s2bc_super_plus_"):
            s2_pairs.append((aid, "s2bc_super_emeans"))
        elif aid.startswith("s2bc_smeans_plus_"):
            s2_pairs.append((aid, "s2bc_smeans_deconfounded"))
    for extra_pair in [
        ("s2bc_super_emeans", "s2bc_emeans_pp_base"),
        ("s2bc_super_emeans", "s2bc_smeans_deconfounded"),
        ("s2bc_super_emeans_wide", "s2bc_super_emeans"),
        ("s2bc_super_emeans_wide", "s2bc_smeans_deconfounded"),
        ("s2bc_hybrid_super_smeans", "s2bc_super_emeans"),
        ("s2bc_hybrid_super_smeans", "s2bc_emeans_pp_base"),
        ("s2bc_hybrid_super_smeans", "s2bc_smeans_deconfounded"),
        ("s2bc_hybrid_super_smeans_wide", "s2bc_super_emeans_wide"),
        ("s2bc_hybrid_super_smeans_wide", "s2bc_super_emeans"),
        ("s2bc_hybrid_super_smeans_wide", "s2bc_smeans_deconfounded"),
    ]:
        if extra_pair not in s2_pairs:
            s2_pairs.append(extra_pair)

    s2_paired: list[dict[str, Any]] = []
    for cid in s2_cells:
        for arm, base in s2_pairs:
            for m in ("train_mse", "test_mse", "total_time_sec"):
                st = compute_paired_comparison(records, cid, arm, base, metric=m)
                if st:
                    s2_paired.append(st)

    # Aggregate Suite 1 & Suite 2 headline summaries
    s1_aggregates: list[dict[str, Any]] = []
    for arm, base in s1_pairs:
        for m in ("test_mse", "train_mse", "total_time_sec"):
            agg_all = aggregate_across_cells(s1_paired, arm, base, metric=m)
            if agg_all:
                agg_all["scope"] = "all_T"
                s1_aggregates.append(agg_all)
            for t_ep in (1.0, 2.0, 5.0):
                agg_t = aggregate_across_cells(
                    s1_paired,
                    arm,
                    base,
                    metric=m,
                    nepochs_filter=t_ep,
                    cell_nepochs_map=cell_nepochs_map,
                )
                if agg_t:
                    agg_t["scope"] = f"T={int(t_ep)}"
                    s1_aggregates.append(agg_t)

    s2_aggregates: list[dict[str, Any]] = []
    for arm, base in s2_pairs:
        for m in ("test_mse", "train_mse", "total_time_sec"):
            agg = aggregate_across_cells(s2_paired, arm, base, metric=m)
            if agg:
                s2_aggregates.append(agg)

    # SKU-stratified absolute latency table for Suite 1
    sku_runtime: list[dict[str, Any]] = []
    skus = sorted({r["gpu_sku"] for r in records if r["suite"] == "suite1"})
    for sku in skus:
        for arm in SUITE1_ARMS:
            sub = [r for r in records if r["suite"] == "suite1" and r["gpu_sku"] == sku and r["arm"] == arm]
            if sub:
                sku_runtime.append(
                    {
                        "gpu_sku": sku,
                        "arm": arm,
                        "n_runs": len(sub),
                        "mean_init_ms": 1000.0 * float(np.mean([r["init_time_sec"] for r in sub])),
                        "mean_train_ms": 1000.0 * float(np.mean([r["train_time_sec"] for r in sub])),
                        "mean_total_ms": 1000.0 * float(np.mean([r["total_time_sec"] for r in sub])),
                        "median_total_ms": 1000.0 * float(np.median([r["total_time_sec"] for r in sub])),
                    }
                )

    # Rare-mode survival summary (`multiscale_rare`)
    rare_summary: list[dict[str, Any]] = []
    for cid in sorted({r["cell_id"] for r in records if "rare_preserved_frac" in r}):
        for arm in sorted({r["arm"] for r in records if r["cell_id"] == cid and "rare_preserved_frac" in r}):
            sub = [r["rare_preserved_frac"] for r in records if r["cell_id"] == cid and r["arm"] == arm]
            if sub:
                rare_summary.append(
                    {
                        "cell_id": cid,
                        "arm": arm,
                        "mean_rare_preserved_pct": 100.0 * float(np.mean(sub)),
                        "std_rare_preserved_pct": 100.0 * float(np.std(sub, ddof=1)) if len(sub) >= 2 else 0.0,
                    }
                )

    generate_plots(records, s1_paired, s2_paired, out_dir)

    return {
        "num_records": len(records),
        "suite1_cells": s1_cells,
        "suite2_cells": s2_cells,
        "suite1_aggregates": s1_aggregates,
        "suite2_aggregates": s2_aggregates,
        "suite1_paired_comparisons": s1_paired,
        "suite2_paired_comparisons": s2_paired,
        "sku_stratified_runtime": sku_runtime,
        "rare_mode_survival": rare_summary,
    }


def write_markdown_report(summary: dict[str, Any], out_path: Path) -> None:
    """Write comprehensive Phase 3 statistical report (`phase3_smeans_summary.md`)."""
    lines: list[str] = [
        "# Phase 3 Multi-Machine Showdown & Component Ablation: `s-means` vs. `SuperEMeans` & Hybrids",
        "",
        f"- **Total Validated Paired Runs**: `{summary['num_records']:,}` (`0` `NaN`/`Inf`, `0` duplicates, single-GPU/single-SKU paired within every cell)",
        "- **Evaluation Protocol**: Exact `FP64` full-dataset `train_mse` and held-out `test_mse` (`exact_mse_fp64`), plus exact `FP64` spherical cosine distortion on `unit_sphere` cells, and synchronized warm GPU `init_time_sec` / `train_time_sec` / `total_time_sec`.",
        "- **Sign Convention**: Negative `Mean Delta (%)` means **lower MSE (better accuracy)** or **faster wall-clock latency** relative to the baseline.",
        "",
        "## 1. Suite 1 Executive Showdown Aggregates (`Test MSE` & `Total Time` Across Regimes)",
        "",
        "| Scope | Arm | Baseline | Metric | Cells | Mean Delta (%) | Median Delta (%) | Cell W/T/L | Seed W/T/L | `p_ttest` | `p_wilcox` |",
        "| :--- | :--- | :--- | :--- | ---: | ---: | ---: | :---: | :---: | ---: | ---: |",
    ]
    for row in summary["suite1_aggregates"]:
        if row["metric"] not in ("test_mse", "total_time_sec"):
            continue
        if row["scope"] not in ("all_T", "T=1", "T=2", "T=5"):
            continue
        sig = "**" if row["p_cell_wilcoxon"] < 0.05 and row["mean_delta_pct"] < 0 else ""
        cw = f"{row['cell_wins']}/{row['cell_ties']}/{row['cell_losses']}"
        sw = f"{row['seed_wins']}/{row['seed_ties']}/{row['seed_losses']}"
        lines.append(
            f"| `{row['scope']}` | `{row['arm']}` | `{row['baseline']}` | `{row['metric']}` | {row['n_cells']} | "
            f"{sig}{row['mean_delta_pct']:+.4f}%{sig} | {row['median_delta_pct']:+.4f}% | "
            f"{cw} | {sw} | {row['p_cell_ttest']:.3e} | {row['p_cell_wilcoxon']:.3e} |"
        )

    lines.extend(
        [
            "",
            "## 2. Suite 2 Directional Component Attribution Aggregates (`Test MSE` & `Total Time`)",
            "",
            "| Arm | Baseline | Metric | Cells | Mean Delta (%) | Median Delta (%) | Cell W/T/L | Seed W/T/L | `p_ttest` | `p_wilcox` |",
            "| :--- | :--- | :--- | ---: | ---: | ---: | :---: | :---: | ---: | ---: |",
        ]
    )
    for row in summary["suite2_aggregates"]:
        if row["metric"] not in ("test_mse", "total_time_sec"):
            continue
        sig = "**" if row["p_cell_wilcoxon"] < 0.05 and row["mean_delta_pct"] < 0 else ""
        cw = f"{row['cell_wins']}/{row['cell_ties']}/{row['cell_losses']}"
        sw = f"{row['seed_wins']}/{row['seed_ties']}/{row['seed_losses']}"
        lines.append(
            f"| `{row['arm']}` | `{row['baseline']}` | `{row['metric']}` | {row['n_cells']} | "
            f"{sig}{row['mean_delta_pct']:+.4f}%{sig} | {row['median_delta_pct']:+.4f}% | "
            f"{cw} | {sw} | {row['p_cell_ttest']:.3e} | {row['p_cell_wilcoxon']:.3e} |"
        )

    if summary.get("rare_mode_survival"):
        lines.extend(
            [
                "",
                "## 3. Rare-Mode Survival (`multiscale_rare_outliers`, Huygens Shield vs. Revival Assassination)",
                "",
                "| Cell | Arm | Mean Rare Modes Preserved (%) | Std (%) |",
                "| :--- | :--- | ---: | ---: |",
            ]
        )
        for rrow in summary["rare_mode_survival"]:
            if rrow["cell_id"].startswith("s1_"):
                lines.append(
                    f"| `{rrow['cell_id']}` | `{rrow['arm']}` | {rrow['mean_rare_preserved_pct']:.2f}% | "
                    f"{rrow['std_rare_preserved_pct']:.2f}% |"
                )

    if summary.get("sku_stratified_runtime"):
        lines.extend(
            [
                "",
                "## 4. SKU-Stratified End-to-End GPU Latency Table (Suite 1)",
                "",
                "| Blackwell TDP SKU | Arm | Runs | Mean Init (ms) | Mean Train (ms) | Mean Total (ms) | Median Total (ms) |",
                "| :--- | :--- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for srow in summary["sku_stratified_runtime"]:
            lines.append(
                f"| `{srow['gpu_sku']}` | `{srow['arm']}` | {srow['n_runs']} | "
                f"{srow['mean_init_ms']:.2f} | {srow['mean_train_ms']:.2f} | "
                f"{srow['mean_total_ms']:.2f} | {srow['median_total_ms']:.2f} |"
            )

    lines.extend(
        [
            "",
            "## 5. Per-Cell Suite 1 Head-to-Head Paired `Test MSE` & Spherical Cosine Distortion Table",
            "",
            "| Cell | Arm | Baseline | Metric | Arm Mean | Base Mean | Mean Delta (%) | W/T/L | `p_ttest` | `p_wilcox` |",
            "| :--- | :--- | :--- | :--- | ---: | ---: | ---: | :---: | ---: | ---: |",
        ]
    )
    key_s1_pairs = {
        ("smeans_fast", "vanilla_emeans"),
        ("smeans_fast", "smeans_exact_ref"),
        ("super_emeans", "smeans_fast"),
        ("hybrid_super_smeans", "smeans_fast"),
        ("hybrid_super_smeans", "super_emeans"),
        ("hybrid_super_smeans_wide", "super_emeans_wide"),
    }
    for row in summary["suite1_paired_comparisons"]:
        if (row["arm"], row["baseline"]) not in key_s1_pairs:
            continue
        if row["metric"] not in ("test_mse", "test_cos_dist"):
            continue
        sig = "**" if row["p_wilcoxon"] < 0.05 and row["mean_delta_pct"] < 0 else ""
        w_str = f"{row['wins']}/{row['ties']}/{row['losses']}"
        lines.append(
            f"| `{row['cell_id']}` | `{row['arm']}` | `{row['baseline']}` | `{row['metric']}` | "
            f"{row['arm_mean']:.6f} | {row['base_mean']:.6f} | {sig}{row['mean_delta_pct']:+.4f}%{sig} | "
            f"{w_str} | {row['p_ttest']:.3e} | {row['p_wilcoxon']:.3e} |"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_md = out_path.parent / f".tmp_{os.getpid()}_{time.time_ns()}_{out_path.name}"
    try:
        tmp_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(tmp_md, out_path)
    finally:
        if tmp_md.exists():
            tmp_md.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 6-Machine / 14-GPU Parallel Fleet Dispatcher
# ---------------------------------------------------------------------------


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
        chk = subprocess.run(
            ["ssh-add", "-l"],
            env=probe_env,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if chk.returncode == 0:
            return cand
    return None


def ensure_local_cifar_caches_populated() -> None:
    """Ensure local CIFAR-10 ResNet-18 `.npz` caches exist before syncing to remote workers."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    load_cifar10_resnet18_embeddings(mode="global_512", device=dev)
    load_cifar10_resnet18_embeddings(mode="patches_128", device=dev)


def dispatch_fleet_14way(
    num_seeds: int,
    seed_start: int,
    suite_filter: str,
    out_dir: Path,
) -> None:
    """Dispatch all 13 fleet shards in parallel across all 6 approved machines and collect shard JSONs."""
    num_shards = len(FLEET_SHARDS)
    assert num_shards == 13

    seen_sessions: set[str] = set()
    seen_host_gpus: set[tuple[str, int]] = set()
    for host, gpu_id, sess, _ in FLEET_SHARDS:
        if "walnut" in host.lower():
            raise RuntimeError("SAFETY VIOLATION: walnut.stratabaremetal.com is strictly off-limits!")
        if not sess.startswith("emeans_p3_smeans_s") or sess == "0":
            raise RuntimeError(f"SAFETY VIOLATION: invalid tmux session name {sess!r}")
        if sess in seen_sessions:
            raise RuntimeError(f"Duplicate tmux session name {sess!r}")
        if (host, gpu_id) in seen_host_gpus:
            raise RuntimeError(f"Duplicate (host, gpu_id) assignment {(host, gpu_id)!r}")
        seen_sessions.add(sess)
        seen_host_gpus.add((host, gpu_id))

    ensure_local_cifar_caches_populated()

    env = os.environ.copy()
    live_sock = discover_live_ssh_auth_sock()
    if live_sock:
        env["SSH_AUTH_SOCK"] = live_sock

    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    rel_out = out_dir.relative_to(ROOT) if out_dir.is_relative_to(ROOT) else Path("outputs/phase3_smeans")

    # 1. Sync audited repo snapshot + ensure CIFAR cache on all 5 unique remote machines
    unique_hosts = sorted({h for h, _, _, _ in FLEET_SHARDS if h != "local"})
    for host in unique_hosts:
        print(f"[Fleet-Phase3] Syncing audited code snapshot to {host}:{REMOTE_WORKDIR} ...", flush=True)
        prep_cmd = (
            f"mkdir -p {REMOTE_WORKDIR}/outputs/phase1/cache {REMOTE_WORKDIR}/outputs/logs && "
            f"if [ -d /home/catid/emeans_runs/noise-fwd/outputs/phase1/cache ]; then "
            f"cp -lf /home/catid/emeans_runs/noise-fwd/outputs/phase1/cache/*.npz "
            f"{REMOTE_WORKDIR}/outputs/phase1/cache/ 2>/dev/null || "
            f"cp -f /home/catid/emeans_runs/noise-fwd/outputs/phase1/cache/*.npz "
            f"{REMOTE_WORKDIR}/outputs/phase1/cache/ 2>/dev/null || true; fi"
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
        subprocess.run(
            [
                "rsync", "-az", "--ignore-existing",
                "-e", "ssh -A -o BatchMode=yes -o ConnectTimeout=8",
                f"{ROOT}/outputs/phase1/cache/",
                f"{host}:{REMOTE_WORKDIR}/outputs/phase1/cache/",
            ],
            env=env,
            check=True,
        )

    # 2. Launch detached `emeans_p3_smeans_s*` sessions simultaneously across all 12 remote GPUs
    remote_entries = [(idx, h, g, s) for idx, (h, g, s, _) in enumerate(FLEET_SHARDS) if h != "local"]
    for idx, host, gpu_id, session in remote_entries:
        remote_out = f"{REMOTE_WORKDIR}/{rel_out}"
        remote_cmd = (
            f"mkdir -p {REMOTE_WORKDIR}/outputs/logs {remote_out}/shards && "
            f"rm -f {REMOTE_WORKDIR}/outputs/logs/{session}.status "
            f"{remote_out}/shards/records_shard_{idx:02d}_of_{num_shards:02d}.json && "
            f"tmux kill-session -t {session} 2>/dev/null || true; "
            f"tmux new-session -d -s {session} "
            + shlex.quote(
                f"export PATH=\"$HOME/.local/bin:/usr/local/cuda/bin:$PATH\"; "
                f"export CUDA_VISIBLE_DEVICES={gpu_id}; "
                f"export TRITON_CACHE_DIR=/tmp/triton_cache_{session}; "
                f"export PYTHONPATH={REMOTE_WORKDIR}/src; "
                f"cd {REMOTE_WORKDIR} && "
                f"{REMOTE_PYTHON} scripts/run_phase3_smeans_ablation.py "
                f"--shard-idx {idx} --num-shards {num_shards} "
                f"--num-seeds {num_seeds} --seed-start {seed_start} "
                f"--suite {suite_filter} --out-dir {remote_out} "
                f"> {REMOTE_WORKDIR}/outputs/logs/{session}.log 2>&1; "
                f"echo EXIT_CODE=$? > {REMOTE_WORKDIR}/outputs/logs/{session}.status"
            )
        )
        subprocess.run(
            ["ssh", "-A", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, remote_cmd],
            env=env,
            check=True,
        )
        print(f"[Fleet-Phase3] Launched shard {idx:02d}/{num_shards} on {host} GPU {gpu_id} ({session})", flush=True)

    # 3. Run local shards on `foureyes` GPUs 0 & 1 in parallel while remote machines execute!
    local_entries = [(idx, h, g, s) for idx, (h, g, s, _) in enumerate(FLEET_SHARDS) if h == "local"]
    local_procs: list[tuple[int, int, subprocess.Popen[Any], Any]] = []
    local_logs_dir = out_dir / "logs"
    local_logs_dir.mkdir(parents=True, exist_ok=True)
    try:
        for idx, _, gpu_id, session in local_entries:
            local_shard_json = shards_dir / f"records_shard_{idx:02d}_of_{num_shards:02d}.json"
            local_shard_json.unlink(missing_ok=True)
            local_env = env.copy()
            local_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            local_env["TRITON_CACHE_DIR"] = f"/tmp/triton_cache_{session}"
            local_env["PYTHONPATH"] = str(ROOT / "src")
            log_f = open(local_logs_dir / f"{session}.log", "w", encoding="utf-8")
            print(f"[Fleet-Phase3] Launching local shard {idx:02d}/{num_shards} on foureyes GPU {gpu_id} ...", flush=True)
            proc = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--shard-idx", str(idx),
                    "--num-shards", str(num_shards),
                    "--num-seeds", str(num_seeds),
                    "--seed-start", str(seed_start),
                    "--suite", suite_filter,
                    "--out-dir", str(out_dir),
                ],
                env=local_env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
            )
            local_procs.append((idx, gpu_id, proc, log_f))

        for idx, gpu_id, proc, log_f in local_procs:
            rc = proc.wait()
            log_f.close()
            if rc != 0:
                tail_txt = (local_logs_dir / f"emeans_p3_smeans_s{idx:02d}.log").read_text(encoding="utf-8")[-2000:]
                raise RuntimeError(
                    f"Local shard {idx} on foureyes GPU {gpu_id} failed with exit code {rc}:\n{tail_txt}"
                )
            print(f"[Fleet-Phase3] Completed local shard {idx:02d}/{num_shards} on foureyes GPU {gpu_id}!", flush=True)
    finally:
        for _, _, proc, log_f in local_procs:
            if proc.poll() is None:
                proc.terminate()
            if not log_f.closed:
                log_f.close()

    # 4. Poll remote shards and pull completed shard JSONs atomically
    pending = list(remote_entries)
    while pending:
        still_running = []
        for idx, host, gpu_id, session in pending:
            probe = subprocess.run(
                [
                    "ssh", "-A", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host,
                    f"cat {REMOTE_WORKDIR}/outputs/logs/{session}.status 2>/dev/null || echo RUNNING",
                ],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            status_txt = probe.stdout.strip()
            if status_txt.startswith("EXIT_CODE="):
                code = int(status_txt.split("=", 1)[1])
                if code != 0:
                    tail_res = subprocess.run(
                        [
                            "ssh", "-A", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host,
                            f"tail -n 60 {REMOTE_WORKDIR}/outputs/logs/{session}.log",
                        ],
                        env=env,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    raise RuntimeError(
                        f"Remote shard {idx} failed on {host} GPU {gpu_id} with {status_txt}:\n{tail_res.stdout}"
                    )
                remote_json = (
                    f"{host}:{REMOTE_WORKDIR}/{rel_out}/shards/records_shard_{idx:02d}_of_{num_shards:02d}.json"
                )
                subprocess.run(
                    [
                        "rsync", "-az",
                        "-e", "ssh -A -o BatchMode=yes -o ConnectTimeout=8",
                        remote_json,
                        f"{shards_dir}/",
                    ],
                    env=env,
                    check=True,
                )
                print(
                    f"[Fleet-Phase3] Collected finished shard {idx:02d}/{num_shards} from {host} GPU {gpu_id}!",
                    flush=True,
                )
            else:
                still_running.append((idx, host, gpu_id, session))
        pending = still_running
        if pending:
            time.sleep(4.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 3 s-means vs SuperEMeans & Hybrid Ablation Orchestrator.")
    p.add_argument("--num-seeds", type=int, default=15, help="Number of paired seeds per cell (default: 15).")
    p.add_argument("--seed-start", type=int, default=2000, help="Starting deterministic seed index.")
    p.add_argument("--shard-idx", type=int, default=0, help="0-based shard index.")
    p.add_argument("--num-shards", type=int, default=1, help="Total number of parallel shards.")
    p.add_argument(
        "--suite",
        type=str,
        default="all",
        choices=("all", "suite1", "suite2"),
        help="Run both suites ('all') or isolate 'suite1' / 'suite2'.",
    )
    p.add_argument("--smoke", action="store_true", help="Fast 3-cell, 2-seed end-to-end smoke verification.")
    p.add_argument("--merge-only", action="store_true", help="Merge existing shard JSONs, analyze, and plot.")
    p.add_argument(
        "--dispatch-fleet",
        action="store_true",
        help="Dispatch 14 shards in parallel across all 6 approved machines and merge into out-dir.",
    )
    p.add_argument("--out-dir", type=Path, default=OUTPUT_DIR, help="Output directory for JSON, Markdown, and PNGs.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir: Path = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    if args.dispatch_fleet:
        dispatch_fleet_14way(
            num_seeds=args.num_seeds,
            seed_start=args.seed_start,
            suite_filter=args.suite,
            out_dir=out_dir,
        )
        args.num_shards = len(FLEET_SHARDS)
        args.merge_only = True

    all_cells = build_campaign_cells(smoke=args.smoke, suite_filter=args.suite)
    n_seeds = 2 if args.smoke else args.num_seeds
    seeds = [args.seed_start + i for i in range(n_seeds)]

    if not args.merge_only:
        if not (0 <= args.shard_idx < args.num_shards):
            raise ValueError(f"Invalid shard_idx={args.shard_idx} for num_shards={args.num_shards}")
        shard_buckets = partition_cells_across_shards(all_cells, num_shards=args.num_shards)
        my_cells = shard_buckets[args.shard_idx]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        t0 = time.perf_counter()
        records = run_cell_shard(my_cells, seeds=seeds, device=device)
        elapsed = time.perf_counter() - t0

        shard_tag = "smoke" if args.smoke else f"shard_{args.shard_idx:02d}_of_{args.num_shards:02d}"
        shard_path = shards_dir / f"records_{shard_tag}.json"
        _atomic_write_json(shard_path, records)
        print(
            f"[Phase3-SMeans] Saved {len(records)} records to {shard_path} in {elapsed:.2f}s",
            flush=True,
        )
        if args.smoke:
            summary = analyze_and_plot(records, out_dir)
            _atomic_write_json(
                out_dir / "phase3_smeans_results.json",
                {
                    "metadata": {
                        "smoke": True,
                        "num_seeds": n_seeds,
                        "seed_start": args.seed_start,
                        "suite": args.suite,
                        "cells": [asdict(c) for c in all_cells],
                    },
                    "summary": summary,
                    "records": records,
                },
            )
            write_markdown_report(summary, out_dir / "phase3_smeans_summary.md")
            return

    if args.num_shards == 1 or args.merge_only:
        merged_records = load_and_validate_shards(
            shards_dir=shards_dir,
            num_shards=args.num_shards,
            expected_cells=all_cells,
            expected_seeds=seeds,
        )
        summary = analyze_and_plot(merged_records, out_dir)
        payload = {
            "metadata": {
                "smoke": False,
                "num_seeds": len(seeds),
                "seed_start": args.seed_start,
                "suite": args.suite,
                "num_shards": args.num_shards,
                "fleet_shards": [
                    {"shard_idx": i, "host": h, "gpu_id": g, "session": s, "speed_weight": w}
                    for i, (h, g, s, w) in enumerate(FLEET_SHARDS)
                ],
                "cells": [asdict(c) for c in all_cells],
                "suite1_arms": list(SUITE1_ARMS),
                "suite2_arms": [aid for aid, _, _ in get_suite2_arm_specs()],
            },
            "summary": summary,
            "records": merged_records,
        }
        _atomic_write_json(out_dir / "phase3_smeans_results.json", payload)
        write_markdown_report(summary, out_dir / "phase3_smeans_summary.md")
        print(
            f"[Phase3-SMeans] Successfully merged & validated {len(merged_records):,} records -> "
            f"{out_dir / 'phase3_smeans_results.json'} & {out_dir / 'phase3_smeans_summary.md'}",
            flush=True,
        )


if __name__ == "__main__":
    main()
