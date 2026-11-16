#!/usr/bin/env python3
"""Phase 2A Multi-Seed Falsification Campaign for Forward-Only `qlabs-eng/noise` Tracks (`N-Track-1..5`).

Evaluates five pre-registered forward-only / perturbation optimization tracks (`src/emeans/noise_tracks.py`)
translated from `https://github.com/qlabs-eng/noise` against `e-means++` (`EMeansPP`) and `e-means`
(`EMeans`) on the exact unified Phase 2A benchmark cells, holdout splits, epoch budgets, seeds, and
double-precision `exact_mse_fp64` protocol, writing artifacts to `outputs/phase2_algo/noise_tracks/`.

Scientific & Benchmarking Hygiene:
  - Strict train / val / test split isolation (`val` and `test` never touched during training or
    `quad_merge`; `N-Track-2` carves its 2.5% holdout strictly inside `x_train` per `Noise` commit `aa67d44`).
  - Exact FP64 MSE evaluation (`exact_mse_fp64`) over 100% of `train` and 100% of `test`.
  - Shared-seed paired design: on each `(cell, seed)`, all `e-means++` arms receive the exact same
    `SeedingResult` and deterministic seed so differences isolate the forward-only mechanism cleanly;
    every `(cell, seed, arm)` comparison runs on the exact same physical GPU.
  - Global (`N=32,768, k=1,024`) + per-cell untimed GPU/Triton warmup outside all timers and
    `torch.cuda.synchronize()` before starting AND before stopping every `time.perf_counter()` region.
  - Reports BOTH confirmed wins AND falsified/negative tracks honestly with paired Student-t and
    Wilcoxon signed-rank p-values across 15 seeds.

Parallel Sharding & 12-GPU / 6-Computer Fleet Dispatch (Board #23, #40, #47, & #54):
  - Single shard: `CUDA_VISIBLE_DEVICES=0 python scripts/run_phase2a_noise.py --shard-idx 0 --num-shards 12`
  - Merge shards: `python scripts/run_phase2a_noise.py --num-shards 12 --merge-only`
  - Audited 12-GPU fleet orchestrator (`foureyes:0,1`, `ripper:0,1,2,3`, `loki:0,1`,
    `cashew.stratabaremetal.com:0,1`, `acorn.stratabaremetal.com:1`, `work:1` — 1 cell per GPU!):
    `python scripts/run_phase2a_noise.py --dispatch-fleet`
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from emeans.datasets import (
    DEFAULT_CACHE_DIR,
    DEFAULT_CIFAR10_ROOT,
    DatasetSplit,
    load_cifar10_resnet18_embeddings,
    make_synthetic_dataset,
)
from emeans.kernels import assign_points, exact_mse_fp64
from emeans.noise_tracks import EMeansNoiseTrack, NOISE_HYPOTHESES
from emeans.seeding import SeedingResult, seed_emeans_plusplus


OUTPUT_DIR = ROOT / "outputs" / "phase2_algo" / "noise_tracks"

# Strict non-overlapping 12-GPU assignment across 6 computers for `noise-fwd` (Board #23, #40, #47, #54).
# 1 cell per GPU across all 12 Phase 2A cells! NEVER includes `walnut` and NEVER touches any tmux session `0`.
FLEET_SHARDS: tuple[tuple[str, int, str], ...] = (
    ("local", 0, "emeans_p2a_noise_s00"),
    ("local", 1, "emeans_p2a_noise_s01"),
    ("ripper", 0, "emeans_p2a_noise_s02"),
    ("ripper", 1, "emeans_p2a_noise_s03"),
    ("ripper", 2, "emeans_p2a_noise_s04"),
    ("ripper", 3, "emeans_p2a_noise_s05"),
    ("loki", 0, "emeans_p2a_noise_s06"),
    ("loki", 1, "emeans_p2a_noise_s07"),
    ("cashew.stratabaremetal.com", 0, "emeans_p2a_noise_s08"),
    ("cashew.stratabaremetal.com", 1, "emeans_p2a_noise_s09"),
    ("acorn.stratabaremetal.com", 1, "emeans_p2a_noise_s10"),
    ("work", 1, "emeans_p2a_noise_s11"),
)
REMOTE_WORKDIR = "/home/catid/emeans_runs/noise-fwd"
REMOTE_PYTHON = "/home/catid/emeans_env/.venv/bin/python"
REMOTE_CIFAR_FALLBACK = Path("/home/catid/emeans_env/data/cifar10/cifar-10-batches-py")


@dataclass(frozen=True)
class CellSpec:
    """One experimental `(dataset, n_train, d, k, nepochs)` evaluation cell."""

    cell_id: str
    dataset_kind: str
    n_train: int
    d: int
    k: int
    nepochs: float
    regime_tag: str


def build_campaign_cells(smoke: bool = False) -> list[CellSpec]:
    """Construct the unified Phase 2A evaluation cells (identical to `scripts/run_phase2_algo.py`)."""
    if smoke:
        return [
            CellSpec("smoke_zipf_t2", "zipf_gmm", 8192, 64, 256, 2.0, "imbalanced_low_nk"),
            CellSpec("smoke_stretch_t2", "stretched_gaussian_2", 8192, 64, 256, 2.0, "anisotropic"),
        ]

    return [
        # 1. Heavy-tailed imbalanced clusters (n/k ~ 97.7): tests N1, N2, N4, N5
        CellSpec("zipf_gmm_t2", "zipf_gmm", 100_000, 128, 1024, 2.0, "imbalanced_low_nk_early"),
        CellSpec("zipf_gmm_t5", "zipf_gmm", 100_000, 128, 1024, 5.0, "imbalanced_low_nk_conv"),
        # 2. Strong anisotropic eigenvalue spectrum (lambda_i ~ i^-2): tests N3 (ANVIL polar), N4, N5
        CellSpec("stretched_a2_t2", "stretched_gaussian_2", 100_000, 128, 1024, 2.0, "anisotropic_early"),
        CellSpec("stretched_a2_t5", "stretched_gaussian_2", 100_000, 128, 1024, 5.0, "anisotropic_conv"),
        # 3. Multi-scale + rare isolated modes: tests N1 rare-mode safety, N2, N5
        CellSpec("multiscale_rare_t2", "multiscale_rare_outliers", 100_000, 128, 512, 2.0, "multiscale_early"),
        CellSpec("multiscale_rare_t5", "multiscale_rare_outliers", 100_000, 128, 512, 5.0, "multiscale_conv"),
        # 4. Tight low-n/k balanced multimodal GMM (n/k = 32, k = 2048): tests N1, N2, N5
        CellSpec("low_nk32_gmm_t3", "balanced_gmm", 65_536, 128, 2048, 3.0, "low_nk32_clustered"),
        # 5. Negative / Neutral Control: Isotropic Gaussian (n/k ~ 195): tests predicted neutrality!
        CellSpec("isotropic_ctrl_t5", "isotropic_gaussian", 100_000, 128, 512, 5.0, "isotropic_control"),
        # 6. Real Deep Embeddings: CIFAR-10 ResNet-18 512-D (n ~ 50k, k = 1024 -> n/k ~ 48.8)
        CellSpec("cifar10_r18_512_t2", "cifar10_global_512", 50_000, 512, 1024, 2.0, "real_d512_low_nk_early"),
        CellSpec("cifar10_r18_512_t5", "cifar10_global_512", 50_000, 512, 1024, 5.0, "real_d512_low_nk_conv"),
        # 7. Real Local Visual Patch Descriptors: CIFAR-10 ResNet-18 128-D (n ~ 200k, k = 2048 -> n/k ~ 97.7)
        CellSpec("cifar10_patches_128_t2", "cifar10_patches_128", 200_000, 128, 2048, 2.0, "real_d128_patches_early"),
        CellSpec("cifar10_patches_128_t5", "cifar10_patches_128", 200_000, 128, 2048, 5.0, "real_d128_patches_conv"),
    ]


def _resolve_cifar_dir() -> Path:
    if DEFAULT_CIFAR10_ROOT.exists():
        return DEFAULT_CIFAR10_ROOT
    if REMOTE_CIFAR_FALLBACK.exists():
        return REMOTE_CIFAR_FALLBACK
    return DEFAULT_CIFAR10_ROOT


def load_cell_dataset(cell: CellSpec, device: torch.device) -> DatasetSplit:
    """Load synthetic or real deduplicated train/val/test dataset for `cell`."""
    local_cache = ROOT / "outputs" / "phase1" / "cache"
    cache_dir = local_cache if local_cache.exists() else DEFAULT_CACHE_DIR
    if cell.dataset_kind == "cifar10_global_512":
        return load_cifar10_resnet18_embeddings(
            mode="global_512",
            k_recommended=cell.k,
            cifar_dir=_resolve_cifar_dir(),
            cache_dir=cache_dir,
            device=str(device),
        )
    if cell.dataset_kind == "cifar10_patches_128":
        return load_cifar10_resnet18_embeddings(
            mode="patches_128",
            k_recommended=cell.k,
            cifar_dir=_resolve_cifar_dir(),
            cache_dir=cache_dir,
            device=str(device),
        )
    return make_synthetic_dataset(
        kind=cell.dataset_kind,
        n_train=cell.n_train,
        n_val=10_000,
        n_test=10_000,
        d=cell.d,
        k=cell.k,
        seed=42,
    )


def build_arms_for_cell(cell: CellSpec) -> list[tuple[str, dict[str, Any]]]:
    """Return `(arm_name, EMeansNoiseTrack kwargs)` testing `N-Track-1..5` + ablations cleanly."""
    common = dict(d=cell.d, k=cell.k, nepochs=cell.nepochs)

    arms: list[tuple[str, dict[str, Any]]] = [
        # --- Track A: Forward-Only `Noise` Tracks on Warm-Started `e-means++` ---
        ("emeans_pp_base", {**common, "base_mode": "emeans_pp", "finalization": "closing_mean"}),
        ("emeans_pp_slow_ema", {**common, "base_mode": "emeans_pp", "finalization": "slow_ema"}),
        ("N1_pp_head_edit", {**common, "base_mode": "emeans_pp", "head_edit": True}),
        ("N2_pp_quad_merge", {**common, "base_mode": "emeans_pp", "finalization": "quad_merge"}),
        ("N3_pp_anvil_b025", {**common, "base_mode": "emeans_pp", "anvil_blend0": 0.25}),
        ("N4_pp_dust_s015", {**common, "base_mode": "emeans_pp", "dust_sigma0": 0.15}),
        ("N5a_pp_strat_only", {**common, "base_mode": "emeans_pp", "stratified_order": True, "wide_step_mult": 1}),
        ("N5b_pp_wide_only", {**common, "base_mode": "emeans_pp", "stratified_order": False, "wide_step_mult": 2}),
        ("N5_pp_strat_wide", {**common, "base_mode": "emeans_pp", "stratified_order": True, "wide_step_mult": 2}),
        (
            "Combo_pp_N1_N2_N5a",
            {
                **common,
                "base_mode": "emeans_pp",
                "head_edit": True,
                "finalization": "quad_merge",
                "stratified_order": True,
            },
        ),
        # --- Track B: Forward-Only `Noise` Tracks on Random-Init `e-means` ---
        ("emeans_rand_base", {**common, "base_mode": "emeans", "finalization": "closing_mean"}),
        ("emeans_rand_slow_ema", {**common, "base_mode": "emeans", "finalization": "slow_ema"}),
        ("N1_rand_head_edit", {**common, "base_mode": "emeans", "head_edit": True}),
        ("N2_rand_quad_merge", {**common, "base_mode": "emeans", "finalization": "quad_merge"}),
        ("N3_rand_anvil_b025", {**common, "base_mode": "emeans", "anvil_blend0": 0.25}),
        ("N4_rand_dust_s015", {**common, "base_mode": "emeans", "dust_sigma0": 0.15}),
        ("N5a_rand_strat_only", {**common, "base_mode": "emeans", "stratified_order": True, "wide_step_mult": 1}),
        ("N5b_rand_wide_only", {**common, "base_mode": "emeans", "stratified_order": False, "wide_step_mult": 2}),
        ("N5_rand_strat_wide", {**common, "base_mode": "emeans", "stratified_order": True, "wide_step_mult": 2}),
        (
            "Combo_rand_N1_N2_N5a",
            {
                **common,
                "base_mode": "emeans",
                "head_edit": True,
                "finalization": "quad_merge",
                "stratified_order": True,
            },
        ),
    ]
    return arms


@torch.no_grad()
def warmup_gpu_kernels(device: torch.device) -> None:
    """Warm up global Triton / cuBLAS / PyTorch specializations outside all timed regions.

    Exercises `N=32,768 > 16,384` and `k=1024 > 256` across all five forward-only mechanisms
    so zero JIT compilation leaks into benchmark timers.
    """
    if device.type != "cuda":
        return
    x_w = torch.randn((32_768, 128), device=device, dtype=torch.float32)
    m_w = EMeansNoiseTrack(
        d=128,
        k=1024,
        nepochs=2.0,
        base_mode="emeans_pp",
        head_edit=True,
        transplant_every=5,
        finalization="quad_merge",
        anvil_blend0=0.25,
        dust_sigma0=0.15,
        stratified_order=True,
        wide_step_mult=2,
        seed=0,
    )
    m_w.fit(x_w, capture_assignments=False)
    _ = assign_points(x_w[:1024], m_w.centroids)
    _ = exact_mse_fp64(x_w[:1024], m_w.centroids, device=device)
    torch.cuda.synchronize(device)


@torch.no_grad()
def warmup_cell_kernels(x_tr: torch.Tensor, cell: CellSpec, device: torch.device) -> None:
    """Warm up exact `(N, d, k)` Triton/cuBLAS specializations for `cell` outside timers.

    Covers BOTH full-`x_tr` (`closing_mean`, exact `n_tr % B` remainder tile) AND `quad_merge` +
    `wide_step_mult=2` + `dust` (`(n_tr - n_hold) % B`, `2*B`, and `B // 2` tiles) so Seed 0 Arm 0
    (`emeans_pp_base`) never pays cold JIT compilation!
    """
    if device.type != "cuda":
        return
    w_seed = seed_emeans_plusplus(
        x_tr,
        k=cell.k,
        seeding_mode="auto",
        coreset_mult=16,
        min_coreset_size=8192,
        polish=True,
        seed=999_999,
    )
    m_base_w = EMeansNoiseTrack(
        d=cell.d,
        k=cell.k,
        nepochs=2.0,
        base_mode="emeans_pp",
        finalization="closing_mean",
        seed=999_998,
    )
    _ = m_base_w.fit(x_tr, seeding_result=w_seed, capture_assignments=False)
    m_cell = EMeansNoiseTrack(
        d=cell.d,
        k=cell.k,
        nepochs=2.0,
        base_mode="emeans_pp",
        head_edit=True,
        transplant_every=5,
        finalization="quad_merge",
        anvil_blend0=0.25,
        dust_sigma0=0.15,
        stratified_order=True,
        wide_step_mult=2,
        seed=999_999,
    )
    c_w, _ = m_cell.fit(x_tr, seeding_result=w_seed, capture_assignments=False)
    _ = assign_points(x_tr[: min(2048, x_tr.shape[0])], c_w)
    _ = exact_mse_fp64(x_tr[: min(2048, x_tr.shape[0])], c_w, device=device)
    torch.cuda.synchronize(device)


@torch.no_grad()
def run_cell_shard(
    cells: list[CellSpec],
    seeds: list[int],
    device: torch.device,
) -> list[dict[str, Any]]:
    """Run all `(cell, seed, arm)` evaluations assigned to this shard."""
    warmup_gpu_kernels(device)
    hostname = socket.gethostname()
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    records: list[dict[str, Any]] = []

    for cell in cells:
        t_cell0 = time.perf_counter()
        ds = load_cell_dataset(cell, device=device)
        x_tr = torch.from_numpy(ds.train).to(device=device, dtype=torch.float32).contiguous()
        x_te = torch.from_numpy(ds.test).to(device=device, dtype=torch.float32).contiguous()
        n_tr_total = int(x_tr.shape[0])
        if x_tr.shape[1] != cell.d or x_te.shape[1] != cell.d:
            raise ValueError(f"Dimension mismatch in cell {cell.cell_id}: got {x_tr.shape}, {x_te.shape}")
        if not (bool(torch.isfinite(x_tr).all()) and bool(torch.isfinite(x_te).all())):
            raise ValueError(f"Non-finite input values detected in cell {cell.cell_id}")

        # Untimed per-cell warmup so Seed 0 never pays Triton JIT compilation inside timers
        warmup_cell_kernels(x_tr, cell, device=device)

        arms = build_arms_for_cell(cell)
        print(
            f"[Phase2A-Noise] Running cell={cell.cell_id} (n={n_tr_total}, d={cell.d}, "
            f"k={cell.k}, T={cell.nepochs}, seeds={len(seeds)}, arms={len(arms)}) on {hostname}:{device} ({gpu_name})",
            flush=True,
        )

        for seed in seeds:
            # Precompute shared e-means++ SeedingResult ONCE per (cell, seed) with synchronized timing
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t_s0 = time.perf_counter()
            shared_seed_res: SeedingResult = seed_emeans_plusplus(
                x_tr,
                k=cell.k,
                seeding_mode="auto",
                coreset_mult=16,
                min_coreset_size=8192,
                polish=True,
                seed=seed,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            shared_init_sec = time.perf_counter() - t_s0

            for arm_name, arm_kwargs in arms:
                model = EMeansNoiseTrack(**arm_kwargs, seed=seed)
                if arm_kwargs["base_mode"] == "emeans_pp":
                    centroids, _ = model.fit(
                        x_tr,
                        seeding_result=shared_seed_res,
                        capture_assignments=False,
                    )
                    # Include both shared seeding latency AND any arm-specific internal init (e.g. quad_merge split)
                    init_sec = shared_init_sec + model.init_time_sec
                else:
                    centroids, _ = model.fit(
                        x_tr,
                        capture_assignments=False,
                    )
                    init_sec = model.init_time_sec

                train_sec = model.train_time_sec
                total_sec = init_sec + train_sec

                train_mse = exact_mse_fp64(x_tr, centroids, device=device)
                test_mse = exact_mse_fp64(x_te, centroids, device=device)

                _, eval_labels = assign_points(x_tr, centroids)
                eval_counts = torch.bincount(eval_labels, minlength=cell.k).double()
                imbalance = float(
                    (cell.k * torch.dot(eval_counts, eval_counts) / (float(n_tr_total) ** 2)).item()
                )
                dead_lt5 = int((eval_counts < 5).sum().item())

                # Rare-mode recovery check when ground-truth rare modes exist (`multiscale_rare_outliers`)
                rare_preserved_frac: float | None = None
                if ds.true_centers is not None and ds.rare_mode_indices is not None and len(ds.rare_mode_indices) > 0:
                    rare_c = torch.from_numpy(ds.true_centers[ds.rare_mode_indices]).to(
                        device=device, dtype=torch.float32
                    )
                    d_rare, _ = assign_points(rare_c, centroids)
                    # A rare mode at radius 3.5*sqrt(2d/3) with std 0.15 has true E[||mu_r - c||^2] << 2.0 * d * 0.15^2
                    hit_cutoff = 1.0 * float(cell.d) * (0.35**2)
                    rare_preserved_frac = float((d_rare <= hit_cutoff).float().mean().item())

                rec: dict[str, Any] = {
                    "cell_id": cell.cell_id,
                    "dataset": cell.dataset_kind,
                    "regime_tag": cell.regime_tag,
                    "host": hostname,
                    "gpu_name": gpu_name,
                    "n_train": n_tr_total,
                    "n_test": int(x_te.shape[0]),
                    "d": cell.d,
                    "k": cell.k,
                    "n_over_k": float(n_tr_total) / float(cell.k),
                    "nepochs": cell.nepochs,
                    "seed": seed,
                    "arm": arm_name,
                    "base_mode": arm_kwargs["base_mode"],
                    "train_mse": float(train_mse),
                    "test_mse": float(test_mse),
                    "init_time_sec": float(init_sec),
                    "train_time_sec": float(train_sec),
                    "total_time_sec": float(total_sec),
                    "n_splits": int(model.n_splits),
                    "n_protected": int(model.n_protected),
                    "n_transplants": int(model.n_transplants),
                    "imbalance_factor": imbalance,
                    "dead_clusters_lt5": dead_lt5,
                }
                if rare_preserved_frac is not None:
                    rec["rare_preserved_frac"] = rare_preserved_frac
                if model.head_edit_diagnostics is not None:
                    rec["he_proposals"] = model.head_edit_diagnostics.proposals_tried
                    rec["he_p_improved"] = model.head_edit_diagnostics.p_improved
                    rec["he_q_confirmed"] = model.head_edit_diagnostics.q_confirmed
                if model.quad_merge_diagnostics is not None:
                    rec["qm_target"] = model.quad_merge_diagnostics.selected_target
                    rec["qm_shrink"] = model.quad_merge_diagnostics.selected_shrink
                    rec["qm_mean_blend"] = model.quad_merge_diagnostics.mean_blend_weight
                records.append(rec)

        print(
            f"[Phase2A-Noise] Finished cell={cell.cell_id} in {time.perf_counter() - t_cell0:.2f}s",
            flush=True,
        )

    return records


def compute_paired_comparison(
    records: list[dict[str, Any]],
    cell_id: str,
    arm: str,
    baseline_arm: str,
    metric: str = "test_mse",
) -> dict[str, Any]:
    """Compute paired per-seed relative delta (%), Student-t p-value, and Wilcoxon p-value."""
    arm_by_seed = {
        r["seed"]: r[metric]
        for r in records
        if r["cell_id"] == cell_id and r["arm"] == arm
    }
    base_by_seed = {
        r["seed"]: r[metric]
        for r in records
        if r["cell_id"] == cell_id and r["arm"] == baseline_arm
    }
    common_seeds = sorted(set(arm_by_seed.keys()) & set(base_by_seed.keys()))
    if not common_seeds:
        return {}

    a_vals = np.array([arm_by_seed[s] for s in common_seeds], dtype=np.float64)
    b_vals = np.array([base_by_seed[s] for s in common_seeds], dtype=np.float64)
    rel_pct = 100.0 * (a_vals - b_vals) / np.maximum(b_vals, 1e-30)
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
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "p_ttest": p_ttest,
        "p_wilcoxon": p_wilcoxon,
    }


def analyze_and_plot(records: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    """Aggregate paired statistical tests, wall-clock ratios, and generate plots + summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    cell_ids = sorted({r["cell_id"] for r in records})

    comparisons_to_run = [
        # Track A (`e-means++` warm-started base)
        ("N1_pp_head_edit", "emeans_pp_base"),
        ("N2_pp_quad_merge", "emeans_pp_base"),
        ("N2_pp_quad_merge", "emeans_pp_slow_ema"),
        ("emeans_pp_slow_ema", "emeans_pp_base"),
        ("N3_pp_anvil_b025", "emeans_pp_base"),
        ("N4_pp_dust_s015", "emeans_pp_base"),
        ("N5a_pp_strat_only", "emeans_pp_base"),
        ("N5b_pp_wide_only", "emeans_pp_base"),
        ("N5_pp_strat_wide", "emeans_pp_base"),
        ("Combo_pp_N1_N2_N5a", "emeans_pp_base"),
        # Track B (`e-means` random-init base)
        ("N1_rand_head_edit", "emeans_rand_base"),
        ("N2_rand_quad_merge", "emeans_rand_base"),
        ("N2_rand_quad_merge", "emeans_rand_slow_ema"),
        ("emeans_rand_slow_ema", "emeans_rand_base"),
        ("N3_rand_anvil_b025", "emeans_rand_base"),
        ("N4_rand_dust_s015", "emeans_rand_base"),
        ("N5a_rand_strat_only", "emeans_rand_base"),
        ("N5b_rand_wide_only", "emeans_rand_base"),
        ("N5_rand_strat_wide", "emeans_rand_base"),
        ("Combo_rand_N1_N2_N5a", "emeans_rand_base"),
    ]

    paired_stats: list[dict[str, Any]] = []
    for cid in cell_ids:
        for arm, base in comparisons_to_run:
            for metric in ("train_mse", "test_mse", "total_time_sec"):
                res = compute_paired_comparison(records, cid, arm, base, metric=metric)
                if res:
                    paired_stats.append(res)

    # Plot 1: N2 (`quad_merge`) Train-vs-Test MSE Pareto Frontier across low-n/k & real datasets
    pareto_cells = [
        c for c in [
            "zipf_gmm_t5",
            "low_nk32_gmm_t3",
            "cifar10_r18_512_t5",
            "cifar10_patches_128_t5",
        ]
        if c in cell_ids
    ]
    if pareto_cells:
        fig, axes = plt.subplots(1, len(pareto_cells), figsize=(4.2 * len(pareto_cells), 3.8), squeeze=False)
        arms_pareto = [
            ("emeans_pp_base", "closing_mean (base)", "#1f77b4", "o"),
            ("emeans_pp_slow_ema", "slow_ema", "#ff7f0e", "s"),
            ("N1_pp_head_edit", "N1: head_edit Transplant", "#d62728", "^"),
            ("N2_pp_quad_merge", "N2: quad_merge Parabolic", "#2ca02c", "*"),
            ("Combo_pp_N1_N2_N5a", "Combo: N1+N2+N5a", "#9467bd", "P"),
        ]
        for ax, cid in zip(axes[0], pareto_cells):
            base_tr = float(np.mean([r["train_mse"] for r in records if r["cell_id"] == cid and r["arm"] == "emeans_pp_base"]))
            base_te = float(np.mean([r["test_mse"] for r in records if r["cell_id"] == cid and r["arm"] == "emeans_pp_base"]))
            for arm_key, label, color, marker in arms_pareto:
                sub = [r for r in records if r["cell_id"] == cid and r["arm"] == arm_key]
                if not sub:
                    continue
                d_tr = 100.0 * (float(np.mean([r["train_mse"] for r in sub])) - base_tr) / base_tr
                d_te = 100.0 * (float(np.mean([r["test_mse"] for r in sub])) - base_te) / base_te
                ax.scatter(d_tr, d_te, label=label, color=color, marker=marker, s=95 if marker != "*" else 160, zorder=4)
            ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
            ax.axvline(0.0, color="gray", linestyle="--", linewidth=0.8)
            ax.set_title(cid, fontsize=10, fontweight="bold")
            ax.set_xlabel("Train MSE Delta vs closing_mean (%)")
            ax.set_ylabel("Test MSE Delta vs closing_mean (%)")
            ax.grid(True, alpha=0.3)
        axes[0][-1].legend(fontsize=7.5, loc="best")
        fig.suptitle("qlabs-eng/noise Forward-Only Tracks: Train vs. Out-of-Sample Test MSE (Lower-Left Dominates)", fontsize=11, fontweight="bold")
        fig.tight_layout()
        fig.savefig(plots_dir / "noise_train_test_pareto.png", dpi=180)
        plt.close(fig)

    # Plot 2: Falsification Matrix Heatmap Across All 5 `Noise` Tracks (`e-means++` & `e-means`)
    focus_arms = [
        ("N1_pp_head_edit", "emeans_pp_base", "N1: head_edit (e-means++)"),
        ("N1_rand_head_edit", "emeans_rand_base", "N1: head_edit (e-means)"),
        ("N2_pp_quad_merge", "emeans_pp_base", "N2: quad_merge (e-means++)"),
        ("N3_pp_anvil_b025", "emeans_pp_base", "N3: ANVIL Polar (e-means++)"),
        ("N4_pp_dust_s015", "emeans_pp_base", "N4: Dust Jitter (e-means++)"),
        ("N4_rand_dust_s015", "emeans_rand_base", "N4: Dust Jitter (e-means)"),
        ("N5a_pp_strat_only", "emeans_pp_base", "N5a: Stratified (e-means++)"),
        ("N5_pp_strat_wide", "emeans_pp_base", "N5: Strat+Wide (e-means++)"),
        ("Combo_pp_N1_N2_N5a", "emeans_pp_base", "Combo N1+N2+N5a (e-means++)"),
        ("Combo_rand_N1_N2_N5a", "emeans_rand_base", "Combo N1+N2+N5a (e-means)"),
    ]
    if cell_ids:
        mat = np.zeros((len(focus_arms), len(cell_ids)), dtype=np.float64)
        pmat = np.ones((len(focus_arms), len(cell_ids)), dtype=np.float64)
        for i, (arm_k, base_k, _) in enumerate(focus_arms):
            for j, cid in enumerate(cell_ids):
                matches = [
                    s for s in paired_stats
                    if s["cell_id"] == cid and s["arm"] == arm_k and s["baseline"] == base_k and s["metric"] == "test_mse"
                ]
                if matches:
                    mat[i, j] = matches[0]["mean_delta_pct"]
                    pmat[i, j] = matches[0]["p_wilcoxon"]

        fig, ax = plt.subplots(figsize=(max(9.5, 0.98 * len(cell_ids)), 5.8))
        vmax = max(0.25, min(2.5, float(np.percentile(np.abs(mat), 95))))
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(cell_ids)))
        ax.set_xticklabels(cell_ids, rotation=35, ha="right", fontsize=8.5)
        ax.set_yticks(range(len(focus_arms)))
        ax.set_yticklabels([lbl for _, _, lbl in focus_arms], fontsize=9)
        for i in range(len(focus_arms)):
            for j in range(len(cell_ids)):
                sig = "*" if pmat[i, j] < 0.05 else ""
                ax.text(
                    j, i, f"{mat[i, j]:+.2f}%{sig}",
                    ha="center", va="center", fontsize=7,
                    color="white" if abs(mat[i, j]) > 0.6 * vmax else "black",
                )
        fig.colorbar(im, ax=ax, label="Paired Test MSE Delta (%) [Blue < 0 = Improvement, * p<0.05]")
        ax.set_title("Phase 2A qlabs-eng/noise Falsification Matrix: Paired Test MSE Delta", fontsize=11, fontweight="bold")
        fig.tight_layout()
        fig.savefig(plots_dir / "noise_falsification_heatmap.png", dpi=180)
        plt.close(fig)

    summary_payload = {
        "hypotheses": {k: asdict(v) for k, v in NOISE_HYPOTHESES.items()},
        "num_records": len(records),
        "cells": cell_ids,
        "paired_comparisons": paired_stats,
    }
    return summary_payload


def write_markdown_report(summary: dict[str, Any], out_path: Path) -> None:
    """Write human-readable scientific falsification report with p-values & win/loss counts."""
    lines = [
        "# Phase 2A Forward-Only (`qlabs-eng/noise`) Falsification Report (`N-Track-1` through `N-Track-5`)",
        "",
        "All MSE metrics are exact double-precision (`FP64`) over 100% of `train` and held-out `test` across paired seeds.",
        "Negative `Delta %` indicates lower MSE or faster wall-clock time; `p_wilcox` and `p_ttest` are two-sided paired tests.",
        "",
        "## Paired Statistical Summary Table (`Test MSE`, `Train MSE`, & `Total Time`)",
        "",
        "| Cell | Arm | Baseline | Metric | Mean Delta (%) | Median Delta (%) | Wins/Ties/Losses | `p_ttest` | `p_wilcox` |",
        "| :--- | :--- | :--- | :--- | ---: | ---: | :---: | ---: | ---: |",
    ]
    for row in summary["paired_comparisons"]:
        if row["metric"] not in ("test_mse", "train_mse", "total_time_sec"):
            continue
        sig = "**" if row["p_wilcoxon"] < 0.05 and row["mean_delta_pct"] < 0 else ""
        w_str = f"{row['wins']}/{row['ties']}/{row['losses']}"
        lines.append(
            f"| `{row['cell_id']}` | `{row['arm']}` | `{row['baseline']}` | `{row['metric']}` | "
            f"{sig}{row['mean_delta_pct']:+.4f}%{sig} | {row['median_delta_pct']:+.4f}% | "
            f"{w_str} | {row['p_ttest']:.3e} | {row['p_wilcoxon']:.3e} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Strict Shard Validation & Audited 5-GPU Fleet Dispatcher
# ---------------------------------------------------------------------------


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON atomically via a PID-unique temp file + `os.replace`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".tmp_{os.getpid()}_{path.name}"
    try:
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def load_and_validate_shards(shards_dir: Path, num_shards: int) -> list[dict[str, Any]]:
    """Load exact `records_shard_XX_of_{num_shards:02d}.json` files and verify zero duplicates."""
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")

    merged: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, int, str]] = set()
    for idx in range(num_shards):
        sf = shards_dir / f"records_shard_{idx:02d}_of_{num_shards:02d}.json"
        if not sf.exists():
            raise FileNotFoundError(f"Missing expected shard file: {sf}")
        part = json.loads(sf.read_text(encoding="utf-8"))
        for rec in part:
            key = (str(rec["cell_id"]), int(rec["seed"]), str(rec["arm"]))
            if key in seen_keys:
                raise RuntimeError(f"Duplicate `(cell_id, seed, arm)` record detected across shards: {key}")
            seen_keys.add(key)
            merged.append(rec)
    return merged


def discover_live_ssh_auth_sock() -> str | None:
    """Discover live SSH_AUTH_SOCK from `tmux show-environment` or `/tmp/ssh-*/agent.*`."""
    try:
        res = subprocess.run(
            ["tmux", "show-environment", "-t", "0", "SSH_AUTH_SOCK"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if res.returncode == 0 and "=" in res.stdout:
            sock = res.stdout.strip().split("=", 1)[1]
            if os.path.exists(sock):
                return sock
    except Exception:
        pass

    cur = os.environ.get("SSH_AUTH_SOCK")
    if cur and os.path.exists(cur):
        return cur
    return None


def ensure_local_cifar_caches_populated() -> None:
    """Ensure both `global_512` and `patches_128` `.npz` caches exist locally before syncing."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    load_cifar10_resnet18_embeddings(mode="global_512", device=dev)
    load_cifar10_resnet18_embeddings(mode="patches_128", device=dev)


def dispatch_fleet_12way(num_seeds: int, seed_start: int, out_dir: Path) -> None:
    """Execute the 12-cell campaign across `FLEET_SHARDS` (12 GPUs across 6 computers, 1 cell/GPU) and merge."""
    num_shards = len(FLEET_SHARDS)
    assert num_shards == 12
    seen_sessions: set[str] = set()
    seen_host_gpus: set[tuple[str, int]] = set()
    for host, gpu_id, sess in FLEET_SHARDS:
        if "walnut" in host:
            raise RuntimeError("SAFETY VIOLATION: walnut is strictly off-limits!")
        if not sess.startswith("emeans_") or sess == "0":
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
    rel_out = out_dir.relative_to(ROOT) if out_dir.is_relative_to(ROOT) else Path("outputs/phase2_algo/noise_tracks")

    # 1. Sync audited repo + CIFAR `.npz` caches ONCE per unique remote host BEFORE launching sessions!
    unique_hosts = sorted({h for h, _, _ in FLEET_SHARDS if h != "local"})
    for host in unique_hosts:
        print(f"[Fleet-Noise] Syncing audited snapshot + CIFAR cache to {host}:{REMOTE_WORKDIR} ...", flush=True)
        subprocess.run(
            [
                "ssh", "-A", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host,
                f"mkdir -p {REMOTE_WORKDIR}/outputs/phase1/cache {REMOTE_WORKDIR}/outputs/logs",
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
        subprocess.run(
            [
                "rsync", "-az",
                "-e", "ssh -A -o BatchMode=yes -o ConnectTimeout=8",
                f"{ROOT}/outputs/phase1/cache/",
                f"{host}:{REMOTE_WORKDIR}/outputs/phase1/cache/",
            ],
            env=env,
            check=True,
        )

    # 2. Launch remote detached `emeans_p2a_noise_s*` sessions simultaneously across all 10 remote GPUs
    remote_entries = [(idx, h, g, s) for idx, (h, g, s) in enumerate(FLEET_SHARDS) if h != "local"]
    for idx, host, gpu_id, session in remote_entries:
        remote_out = f"{REMOTE_WORKDIR}/{rel_out}"
        remote_cmd = (
            f"mkdir -p {REMOTE_WORKDIR}/outputs/logs {remote_out}/shards && "
            f"rm -f {REMOTE_WORKDIR}/outputs/logs/{session}.status && "
            f"tmux kill-session -t {session} 2>/dev/null || true; "
            f"tmux new-session -d -s {session} "
            + shlex.quote(
                f"export CUDA_VISIBLE_DEVICES={gpu_id}; "
                f"export PYTHONPATH={REMOTE_WORKDIR}/src; "
                f"cd {REMOTE_WORKDIR} && "
                f"{REMOTE_PYTHON} scripts/run_phase2a_noise.py "
                f"--shard-idx {idx} --num-shards {num_shards} "
                f"--num-seeds {num_seeds} --seed-start {seed_start} "
                f"--out-dir {remote_out} "
                f"> {REMOTE_WORKDIR}/outputs/logs/{session}.log 2>&1; "
                f"echo EXIT_CODE=$? > {REMOTE_WORKDIR}/outputs/logs/{session}.status"
            )
        )
        subprocess.run(
            ["ssh", "-A", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, remote_cmd],
            env=env,
            check=True,
        )
        print(f"[Fleet-Noise] Launched shard {idx}/{num_shards} on {host} GPU {gpu_id} (tmux={session})", flush=True)

    # 3. Launch local shard(s) on `foureyes` GPUs `0,1` IN PARALLEL while remote shards run!
    local_entries = [(idx, h, g, s) for idx, (h, g, s) in enumerate(FLEET_SHARDS) if h == "local"]
    local_procs: list[tuple[int, int, subprocess.Popen[Any]]] = []
    try:
        for idx, _, gpu_id, _ in local_entries:
            local_env = env.copy()
            local_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            print(f"[Fleet-Noise] Launching local shard {idx}/{num_shards} on foureyes GPU {gpu_id} ...", flush=True)
            proc = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--shard-idx", str(idx),
                    "--num-shards", str(num_shards),
                    "--num-seeds", str(num_seeds),
                    "--seed-start", str(seed_start),
                    "--out-dir", str(out_dir),
                ],
                env=local_env,
            )
            local_procs.append((idx, gpu_id, proc))

        for idx, gpu_id, proc in local_procs:
            rc = proc.wait()
            if rc != 0:
                raise RuntimeError(f"Local shard {idx} on foureyes GPU {gpu_id} failed with exit code {rc}")
            print(f"[Fleet-Noise] Completed local shard {idx}/{num_shards} on foureyes GPU {gpu_id}!", flush=True)
    finally:
        for _, _, proc in local_procs:
            if proc.poll() is None:
                proc.terminate()

    # 4. Poll remote shards cleanly until all complete and rsync shard JSONs back
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
                            f"tail -n 50 {REMOTE_WORKDIR}/outputs/logs/{session}.log",
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
                print(f"[Fleet-Noise] Collected finished shard {idx}/{num_shards} from {host} GPU {gpu_id}!", flush=True)
            else:
                still_running.append((idx, host, gpu_id, session))
        pending = still_running
        if pending:
            time.sleep(5.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Phase 2A forward-only (qlabs-eng/noise) falsification suite.")
    p.add_argument("--num-seeds", type=int, default=15, help="Number of paired seeds per cell.")
    p.add_argument("--seed-start", type=int, default=1000, help="Starting seed index.")
    p.add_argument("--shard-idx", type=int, default=0, help="0-based shard index.")
    p.add_argument("--num-shards", type=int, default=1, help="Total number of parallel shards.")
    p.add_argument("--smoke", action="store_true", help="Quick 2-cell, 2-seed smoke verification.")
    p.add_argument("--merge-only", action="store_true", help="Merge existing shard JSONs and plot.")
    p.add_argument(
        "--dispatch-fleet",
        action="store_true",
        help="Run 12 shards in parallel across foureyes:0,1 + ripper:0..3 + loki:0,1 + cashew:0,1 + acorn:1 + work:1.",
    )
    p.add_argument("--out-dir", type=Path, default=OUTPUT_DIR, help="Output artifact directory.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir: Path = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    if args.dispatch_fleet:
        dispatch_fleet_12way(num_seeds=args.num_seeds, seed_start=args.seed_start, out_dir=out_dir)
        args.num_shards = len(FLEET_SHARDS)
        args.merge_only = True

    if not args.merge_only:
        if not (0 <= args.shard_idx < args.num_shards):
            raise ValueError(f"Invalid shard_idx={args.shard_idx} for num_shards={args.num_shards}")
        all_cells = build_campaign_cells(smoke=args.smoke)
        n_seeds = 2 if args.smoke else args.num_seeds
        seeds = [args.seed_start + i for i in range(n_seeds)]

        my_cells = [c for idx, c in enumerate(all_cells) if (idx % args.num_shards) == args.shard_idx]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        t0 = time.perf_counter()
        records = run_cell_shard(my_cells, seeds=seeds, device=device)
        elapsed = time.perf_counter() - t0

        shard_tag = "smoke" if args.smoke else f"shard_{args.shard_idx:02d}_of_{args.num_shards:02d}"
        shard_path = shards_dir / f"records_{shard_tag}.json"
        _atomic_write_json(shard_path, records)
        print(
            f"[Phase2A-Noise] Saved {len(records)} records to {shard_path} in {elapsed:.2f}s",
            flush=True,
        )
        if args.smoke:
            summary = analyze_and_plot(records, out_dir / "smoke")
            write_markdown_report(summary, out_dir / "smoke" / "smoke_summary.md")
            return

    if args.num_shards == 1 or args.merge_only:
        merged_records = load_and_validate_shards(shards_dir, num_shards=args.num_shards)
        expected_cells = build_campaign_cells(smoke=False)
        expected_total = len(expected_cells) * args.num_seeds * len(build_arms_for_cell(expected_cells[0]))
        if len(merged_records) != expected_total:
            raise RuntimeError(
                f"Merged record count {len(merged_records)} != expected {expected_total} "
                f"({len(expected_cells)} cells x {args.num_seeds} seeds x 20 arms)"
            )

        summary = analyze_and_plot(merged_records, out_dir)
        full_json = {
            "summary": summary,
            "records": merged_records,
        }
        _atomic_write_json(out_dir / "phase2a_noise_results.json", full_json)
        write_markdown_report(summary, out_dir / "phase2a_noise_summary.md")
        print(
            f"[Phase2A-Noise] Merged {len(merged_records)} total records -> "
            f"{out_dir / 'phase2a_noise_results.json'} & {out_dir / 'phase2a_noise_summary.md'}",
            flush=True,
        )


if __name__ == "__main__":
    main()
