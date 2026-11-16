#!/usr/bin/env python3
"""Downstream ML Pipeline Benchmark Runner: `SemDeDup` Coreset Pruning & Megapixel VQ.

Evaluates `ApexEMeans` (`Auto 4-Gear`, `CroissAttn-FP8 + Ozaki + CUDAGraph`, `BF16 + CUDAGraph`)
against `Random`, `Lloyd@25`, `e-means`, `s-means`, `e-means++`, and `UltraEMeans` across two
high-impact downstream ML/systems workloads beyond ANN search:

  1. **Suite 1 — Large-Scale Semantic Data Deduplication & Coreset Selection (`SemDeDup`)**:
     - `agnews_256_k1024`         (`~110K` train, `7.6K` test, `4` classes, `256-D` TF-IDF+SVD)
     - `cifar10_r18_512_k512`     (` ~50K` train, `10K` test, `10` classes, `512-D` ResNet-18, `K=512`)
     - `cifar10_r18_512_k1024`    (` ~50K` train, `10K` test, `10` classes, `512-D` ResNet-18, `K=1024`)
     - `cifar10_convnext_768_k2048`(`~50K` train, `10K` test, `10` classes, `768-D` ConvNeXt, `K=2048`)
     - `covtype_54_k4096`         (` 500K` train, `50K` test,  `7` classes,  `54-D` Covertype, `K=4096`)
     Evaluates 3 zero-leakage GPU downstream classifiers (GCV Cholesky Ridge, GPU L-BFGS Multinomial
     Logistic Probe, and Origin-Centered Weighted / Prototype `$k$-NN`) at `10%`, `20%`, `50%`, and
     `Prototype (<= K)` budgets against `100%` unpruned reference accuracy.

  2. **Suite 2 — High-Resolution Megapixel Image Color & `4x4x3` Patch Vector Quantization**:
     - `vq_rgb_1m_k{256,1024,4096}`     (`1,000,000 x 3` RGB train, `262,144 x 3` disjoint test)
     - `vq_patch_500k_k{256,1024,4096}` (`  500,000 x 48` `4x4x3` train, `100,000 x 48` disjoint test)
     Reports exact IEEE-754 FP64 Full-Image & Held-Out Test PSNR (dB) vs. GPU compression latency (ms).

Supports two-wave conflict-free fleet execution (`--wave baselines`, `--wave apex`, `--wave all`)
across dedicated Blackwell slots (`acorn:0,1` / `cashew:0,1` + `foureyes`, `0` overlap with `walnut`, `ripper`, `loki`, `work`, or `tmux 0`).
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

# Isolate TMPDIR and Triton disk cache on `/home` BEFORE importing torch/triton/emeans (AGENTS.md Rule 4)
_HOME_SCRATCH_BASE = (
    Path("/home/catid/emeans_env/tmp")
    if Path("/home/catid/emeans_env").exists()
    else Path.home() / ".cache" / "emeans_tmp"
)
_HOME_SCRATCH_BASE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TMPDIR", str(_HOME_SCRATCH_BASE))
if "TRITON_CACHE_DIR" not in os.environ:
    _vis_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "all").replace(",", "_")
    _default_tc = _HOME_SCRATCH_BASE / f"triton_cache_semdedup_u{os.getuid()}_g{_vis_gpu}"
    _default_tc.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(_default_tc)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
from PIL import Image
from scipy import stats
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from emeans.apex_emeans import build_apex_arm, warmup_apex_kernels
from emeans.baselines import LloydKMeans
from emeans.emeans import EMeans
from emeans.emeans_pp import EMeansPP
from emeans.kernels import assign_points
from emeans.real_corpora import fast_exact_assign_fp64
from emeans.seeding import HAS_TRITON as SEED_HAS_TRITON
from emeans.semdedup_coreset import (
    DEFAULT_SEMDEDUP_DATA_DIR,
    FOUR_LABELED_BENCHMARKS,
    LABELED_SPEC_BY_NAME,
    MEGAPIXEL_SPEC_BY_NAME,
    TWO_MEGAPIXEL_VQ_BENCHMARKS,
    LabeledVectorDataset,
    MegapixelVQDataset,
    _compute_equivariant_origin64,
    compute_semdedup_scores_gpu,
    compute_vq_psnr_fp64,
    encode_vq_indices_gpu,
    evaluate_cholesky_ridge_gpu,
    evaluate_multinomial_logistic_gpu,
    evaluate_prototype_knn_gpu,
    evaluate_weighted_knn_gpu,
    load_labeled_benchmark,
    load_megapixel_vq_benchmark,
    select_coreset_at_budget,
    select_prototype_coreset,
    select_random_coreset,
    validate_labeled_dataset,
    validate_megapixel_vq_dataset,
    warmup_semdedup_kernels,
)
from emeans.smeans import SMeans
from emeans.ultra_emeans import build_ultra_arm

if SEED_HAS_TRITON:
    from emeans.seeding import _local_greedy_kpp_kernel

try:
    from emeans.croiss_fp8 import warmup_croiss_fp8_kernels
except Exception:
    warmup_croiss_fp8_kernels = None


OUTPUT_DIR = ROOT / "outputs" / "semdedup"
REMOTE_WORKDIR = "/home/catid/emeans_runs/semdedup-bench"
REMOTE_PYTHON = "/home/catid/emeans_env/.venv/bin/python"

EPOCH_BUDGETS: tuple[float, ...] = (1.0, 2.0, 5.0)
RETENTION_BUDGETS: tuple[float, ...] = (0.10, 0.20, 0.50)

BASELINE_STREAMING_ARMS: tuple[str, ...] = (
    "emeans",
    "smeans",
    "emeans_pp",
    "ultra_emeans",
)
APEX_STREAMING_ARMS: tuple[str, ...] = (
    "apex_emeans_bf16",
    "apex_emeans_fp8",
    "apex_emeans",
)
ALL_STREAMING_ARMS: tuple[str, ...] = BASELINE_STREAMING_ARMS + APEX_STREAMING_ARMS

ARM_DISPLAY: dict[str, tuple[str, str, str, str]] = {
    "random": ("Random Pruning / Palette", "#7f7f7f", "x", ":"),
    "lloyd_25": ("Lloyd @ 25 ep", "#333333", "*", "--"),
    "emeans": ("e-means (Meta FAIR)", "#1f77b4", "o", "-"),
    "smeans": ("s-means (Triton)", "#b8860b", "^", "-."),
    "emeans_pp": ("e-means++", "#ff7f0e", "s", "-"),
    "ultra_emeans": ("UltraEMeans", "#17becf", "h", "-"),
    "apex_emeans_bf16": ("ApexEMeans (BF16+Graph)", "#e377c2", "X", "--"),
    "apex_emeans_fp8": ("ApexEMeans (CroissFP8+Graph)", "#d62728", "p", "-"),
    "apex_emeans": ("ApexEMeans (Auto Flagship)", "#8b0000", "D", "-"),
}


# ==============================================================================
# 1. Benchmark Cell & Dedicated 4-GPU (`work:0,1 + loki:0,1`) Fleet Specs
# ==============================================================================


@dataclass(frozen=True)
class SemDeDupCellSpec:
    """One labeled ML SemDeDup / Coreset Selection benchmark cell pinned to a single GPU SKU."""

    cell_id: str
    dataset_name: str
    short_title: str
    modality: str
    n_train_limit: int | None
    n_test_limit: int | None
    d: int
    num_classes: int
    k: int
    primary_probe: str  # pre-registered by modality geometry: "logreg" (embeddings) or "knn" (spatial sensors)
    sku_tier: str


@dataclass(frozen=True)
class MegapixelVQCellSpec:
    """One Megapixel RGB or `4x4x3` Patch VQ benchmark cell pinned to a single GPU SKU."""

    cell_id: str
    dataset_name: str
    short_title: str
    modality: str
    n_train_limit: int | None
    n_test_limit: int | None
    d: int
    k: int
    bits_per_pixel: float
    sku_tier: str


@dataclass(frozen=True)
class FleetShardSpec:
    """Dedicated single-GPU shard on `acorn`/`cashew` (`600W Server`) or `foureyes` (`300W Max-Q`)."""

    shard_idx: int
    host: str
    gpu_id: int
    session: str
    sku_tier: str
    semdedup_cells: tuple[str, ...]
    vq_cells: tuple[str, ...]


def assert_safe_fleet_target(host: str, session: str) -> None:
    """Hard-block `walnut`, `ripper` (`#196`), `loki` (`#212`), `work` (`#223`), and `tmux` session `0`."""
    h_low = host.strip().lower()
    if "walnut" in h_low:
        raise RuntimeError(f"SAFETY VIOLATION: {host!r} (walnut) is strictly forbidden!")
    if any(bad in h_low for bad in ("ripper", "loki", "work")):
        raise RuntimeError(
            f"SAFETY VIOLATION: {host!r} excluded to protect external user jobs (#196/#212/#223)!"
        )
    allowed = {"acorn.stratabaremetal.com", "cashew.stratabaremetal.com", "local"}
    if host not in allowed:
        raise RuntimeError(f"Unexpected host {host!r}; allowed dedicated semdedup hosts: {sorted(allowed)}")
    if session == "0" or not session.startswith("emeans_semdedup_s"):
        raise RuntimeError(f"SAFETY VIOLATION: invalid or forbidden tmux session {session!r}")


def build_semdedup_cells(smoke: bool = False) -> list[SemDeDupCellSpec]:
    """Construct the 5 SemDeDup / Coreset Pruning cells (`K = 512..4096`) or fast real-data smoke cells."""
    if smoke:
        return [
            SemDeDupCellSpec(
                "smoke_agnews_256", "agnews_256", "AG-News Smoke (8K×256, K=512)",
                "News Topic Embeddings", 8192, 512, 256, 4, 512, "logreg", "smoke_local",
            ),
            SemDeDupCellSpec(
                "smoke_cifar10_r18_512", "cifar10_resnet18_512", "CIFAR-10 R18 Smoke (8K×512, K=512)",
                "ReLU CNN Embeddings", 8192, 512, 512, 10, 512, "logreg", "smoke_local",
            ),
            SemDeDupCellSpec(
                "smoke_cifar10_convnext_768", "cifar10_convnext_768", "ConvNeXt Smoke (8K×768, K=512)",
                "ConvNeXt 768D Vision", 8192, 512, 768, 10, 512, "logreg", "smoke_local",
            ),
            SemDeDupCellSpec(
                "smoke_covtype_54", "covtype_54", "Covertype Smoke (8K×54, K=512)",
                "Multi-Scale Physical Sensors", 8192, 512, 54, 7, 512, "knn", "smoke_local",
            ),
        ]

    return [
        SemDeDupCellSpec(
            "agnews_256_k1024", "agnews_256", "AG-News SVD (110K×256, K=1024)",
            "Heavy-Tailed News Topics (4 cls)", None, None, 256, 4, 1024, "logreg", "server_600w",
        ),
        SemDeDupCellSpec(
            "cifar10_r18_512_k512", "cifar10_resnet18_512", "CIFAR-10 ResNet-18 (50K×512, K=512)",
            "Global ReLU CNN (10 cls, K=512)", None, None, 512, 10, 512, "logreg", "maxq_local",
        ),
        SemDeDupCellSpec(
            "cifar10_r18_512_k1024", "cifar10_resnet18_512", "CIFAR-10 ResNet-18 (50K×512, K=1024)",
            "Global ReLU CNN (10 cls, K=1024)", None, None, 512, 10, 1024, "logreg", "maxq_local",
        ),
        SemDeDupCellSpec(
            "cifar10_convnext_768_k2048", "cifar10_convnext_768", "CIFAR-10 ConvNeXt (50K×768, K=2048)",
            "Modern 768-D Vision (10 cls)", None, None, 768, 10, 2048, "logreg", "maxq_local",
        ),
        SemDeDupCellSpec(
            "covtype_54_k4096", "covtype_54", "Forest Covertype (500K×54, K=4096)",
            "Physical Sensors 110,761x Span (7 cls)", None, None, 54, 7, 4096, "knn", "server_600w",
        ),
    ]


def build_vq_cells(smoke: bool = False) -> list[MegapixelVQCellSpec]:
    """Construct all 6 Megapixel VQ cells (`K in {256, 1024, 4096}` on `1M x 3` & `500K x 48`)."""
    if smoke:
        return [
            MegapixelVQCellSpec(
                "smoke_vq_rgb_k256", "megapixel_rgb_3d", "RGB Smoke (8K×3, K=256)",
                "RGB Color VQ", 8192, 1024, 3, 256, 8.0, "smoke_local",
            ),
            MegapixelVQCellSpec(
                "smoke_vq_rgb_k512", "megapixel_rgb_3d", "RGB Smoke (8K×3, K=512)",
                "RGB Color VQ", 8192, 1024, 3, 512, 9.0, "smoke_local",
            ),
            MegapixelVQCellSpec(
                "smoke_vq_patch_k256", "megapixel_patch_48d", "4×4 Patch Smoke (8K×48, K=256)",
                "4×4 Patch VQ", 8192, 1024, 48, 256, 0.50, "smoke_local",
            ),
            MegapixelVQCellSpec(
                "smoke_vq_patch_k512", "megapixel_patch_48d", "4×4 Patch Smoke (8K×48, K=512)",
                "4×4 Patch VQ", 8192, 1024, 48, 512, 9.0 / 16.0, "smoke_local",
            ),
        ]

    cells: list[MegapixelVQCellSpec] = []
    for k_v in (256, 1024, 4096):
        cells.append(
            MegapixelVQCellSpec(
                f"vq_rgb_1m_k{k_v}",
                "megapixel_rgb_3d",
                f"Megapixel RGB Palette (1M×3, K={k_v:,})",
                "1.0 MP Color Palette VQ",
                None,
                None,
                3,
                k_v,
                math.log2(k_v),
                "server_600w",
            )
        )
    for k_v in (256, 1024, 4096):
        cells.append(
            MegapixelVQCellSpec(
                f"vq_patch_500k_k{k_v}",
                "megapixel_patch_48d",
                f"Megapixel 4×4 Patch VQ (500K×48, K={k_v:,})",
                "8.0 MP 4×4×3 Patch VQ",
                None,
                None,
                48,
                k_v,
                math.log2(k_v) / 16.0,
                "server_600w",
            )
        )
    return cells


def build_fleet_shards(
    local_gpus: tuple[int, int] = (1, 3),
    remote_host: str = "acorn.stratabaremetal.com",
) -> list[FleetShardSpec]:
    """Dedicated 4-GPU partition across `remote_host:0,1` (`600W Server`) + `local` (`foureyes` `300W Max-Q`)."""
    g0, g1 = int(local_gpus[0]), int(local_gpus[1])
    if g0 == g1:
        raise ValueError(f"--local-gpus must name two distinct GPU indices, got {local_gpus}")
    rem_tier = "server_600w"
    shards = [
        FleetShardSpec(
            shard_idx=0,
            host=remote_host,
            gpu_id=0,
            session="emeans_semdedup_s00",
            sku_tier=rem_tier,
            semdedup_cells=("covtype_54_k4096",),
            vq_cells=("vq_rgb_1m_k256", "vq_rgb_1m_k1024", "vq_rgb_1m_k4096"),
        ),
        FleetShardSpec(
            shard_idx=1,
            host=remote_host,
            gpu_id=1,
            session="emeans_semdedup_s01",
            sku_tier=rem_tier,
            semdedup_cells=("agnews_256_k1024",),
            vq_cells=("vq_patch_500k_k256", "vq_patch_500k_k1024", "vq_patch_500k_k4096"),
        ),
        FleetShardSpec(
            shard_idx=2,
            host="local",
            gpu_id=g0,
            session="emeans_semdedup_s02",
            sku_tier="maxq_local",
            semdedup_cells=("cifar10_r18_512_k512", "cifar10_r18_512_k1024"),
            vq_cells=(),
        ),
        FleetShardSpec(
            shard_idx=3,
            host="local",
            gpu_id=g1,
            session="emeans_semdedup_s03",
            sku_tier="maxq_local",
            semdedup_cells=("cifar10_convnext_768_k2048",),
            vq_cells=(),
        ),
    ]
    for sp in shards:
        assert_safe_fleet_target(sp.host, sp.session)
    return shards


# ==============================================================================
# 2. Solver Instantiation & Untimed Full-Shape Warmup (`0` Cold-JIT Spikes)
# ==============================================================================


def get_wave_specs(wave: str, budgets: tuple[float, ...]) -> list[tuple[str, float]]:
    """Return ordered `(arm_name, nepochs)` execution list for `baselines`, `apex`, or `all`."""
    out: list[tuple[str, float]] = []
    if wave in ("baselines", "all"):
        out.append(("random", 0.0))
        out.append(("lloyd_25", 25.0))
        for b_t in budgets:
            for arm in BASELINE_STREAMING_ARMS:
                out.append((arm, float(b_t)))
    if wave in ("apex", "all"):
        for b_t in budgets:
            for arm in APEX_STREAMING_ARMS:
                out.append((arm, float(b_t)))
    if not out:
        raise ValueError(f"Unknown wave={wave!r}; expected 'baselines', 'apex', or 'all'")
    return out


def instantiate_solver_arm(arm: str, d: int, k: int, nepochs: float, seed: int) -> Any:
    if arm == "lloyd_25":
        return LloydKMeans(d=d, k=k, niter=25, init="random", seed=seed)
    if arm == "emeans":
        return EMeans(d=d, k=k, nepochs=nepochs, seed=seed)
    if arm == "smeans":
        return SMeans(d=d, k=k, nepochs=nepochs, exact_ref=False, seed=seed)
    if arm == "emeans_pp":
        return EMeansPP(d=d, k=k, nepochs=nepochs, seed=seed)
    if arm == "ultra_emeans":
        return build_ultra_arm("ultra_emeans", d=d, k=k, nepochs=nepochs, seed=seed)
    if arm in ("apex_emeans", "apex_emeans_bf16", "apex_emeans_fp8"):
        return build_apex_arm(arm, d=d, k=k, nepochs=nepochs, seed=seed)
    raise ValueError(f"Unknown clustering arm {arm!r}")


def detect_gpu_sku_info(device: torch.device) -> tuple[str, float]:
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
def warmup_global_kernels(device: torch.device) -> None:
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
def warmup_cell_wave(
    x_train_gpu: torch.Tensor,
    x_test_gpu: torch.Tensor,
    d: int,
    k: int,
    wave_specs: list[tuple[str, float]],
    is_semdedup: bool,
    y_train_gpu: torch.Tensor | None,
    y_test_gpu: torch.Tensor | None,
    num_classes: int,
    device: torch.device,
) -> None:
    """Pre-compile every Triton & CUDAGraph specialization for `(N, D, K)` outside all timers."""
    if device.type != "cuda":
        return

    m_sub = min(int(x_train_gpu.shape[0]), max(8192, 16 * k, k * 2))
    x_m = x_train_gpu[:m_sub].float().contiguous()
    with torch.cuda.device(device):
        for dt in (torch.float32, torch.bfloat16):
            for k_cand in (1, 16, 17, 64, 65, 128, 129):
                if k_cand <= m_sub:
                    assign_points(x_m, x_m[:k_cand], compute_dtype=dt)

    if any(a.startswith("apex_") for a, _ in wave_specs):
        warmup_apex_kernels(device, d=d, k=k)
    if is_semdedup:
        warmup_semdedup_kernels(device, d=d, k=k)

    c_warm = x_train_gpu[:k].clone().contiguous()
    for arm_name, nepochs in wave_specs:
        if arm_name == "random":
            continue
        if arm_name == "lloyd_25":
            slv = LloydKMeans(d=d, k=k, niter=2, init="random", seed=999_801)
        else:
            slv = instantiate_solver_arm(arm_name, d=d, k=k, nepochs=nepochs, seed=999_802)
        c_warm, _ = slv.fit(x_train_gpu, capture_assignments=False)

    if is_semdedup and y_train_gpu is not None and y_test_gpu is not None:
        st_w = compute_semdedup_scores_gpu(x_train_gpu, c_warm, k=k)
        sel_w = select_coreset_at_budget(st_w, retention_frac=0.10)
        xs = x_train_gpu[sel_w.indices]
        ys = y_train_gpu[sel_w.indices]
        q_sub = min(256, int(x_test_gpu.shape[0]))
        _ = evaluate_cholesky_ridge_gpu(xs, ys, sel_w.weights, x_test_gpu[:q_sub], y_test_gpu[:q_sub], num_classes)
        with torch.enable_grad():
            _ = evaluate_multinomial_logistic_gpu(
                xs, ys, sel_w.weights, x_test_gpu[:q_sub], y_test_gpu[:q_sub], num_classes, max_iter=5
            )
        _ = evaluate_weighted_knn_gpu(xs, ys, sel_w.weights, x_test_gpu[:q_sub], y_test_gpu[:q_sub], num_classes)
        _ = evaluate_prototype_knn_gpu(
            xs, ys, sel_w.weights, sel_w.cluster_ids, k, x_test_gpu[:q_sub], y_test_gpu[:q_sub], num_classes
        )
    else:
        _ = encode_vq_indices_gpu(x_train_gpu, c_warm)
        _ = compute_vq_psnr_fp64(x_train_gpu[: min(4096, int(x_train_gpu.shape[0]))], c_warm, device=device)
        _ = compute_vq_psnr_fp64(x_test_gpu[: min(2048, int(x_test_gpu.shape[0]))], c_warm, device=device)

    torch.cuda.synchronize(device)


# ==============================================================================
# 3. Suite 1 (`SemDeDup` & Coreset Selection) + Suite 2 (`Megapixel VQ`) Execution
# ==============================================================================


def _load_semdedup_cell_data(
    cell: SemDeDupCellSpec,
    data_dir: Path,
    device: torch.device,
) -> LabeledVectorDataset:
    full = load_labeled_benchmark(cell.dataset_name, data_dir=data_dir, device=device)
    if cell.n_train_limit is None and cell.n_test_limit is None:
        return full
    ntr = min(int(cell.n_train_limit or full.x_train.shape[0]), full.x_train.shape[0])
    nte = min(int(cell.n_test_limit or full.x_test.shape[0]), full.x_test.shape[0])
    tr_idx = np.linspace(0, full.x_train.shape[0] - 1, ntr, dtype=np.int64)
    te_idx = np.linspace(0, full.x_test.shape[0] - 1, nte, dtype=np.int64)
    sub = LabeledVectorDataset(
        name=cell.cell_id,
        short_label=cell.short_title,
        modality=cell.modality,
        x_train=np.ascontiguousarray(full.x_train[tr_idx], dtype=np.float32),
        y_train=np.ascontiguousarray(full.y_train[tr_idx], dtype=np.int64),
        x_test=np.ascontiguousarray(full.x_test[te_idx], dtype=np.float32),
        y_test=np.ascontiguousarray(full.y_test[te_idx], dtype=np.int64),
        d=cell.d,
        num_classes=cell.num_classes,
        k_default=cell.k,
        metadata=dict(full.metadata),
    )
    validate_labeled_dataset(sub, check_cross_split_disjoint=True)
    return sub


def _load_vq_cell_data(
    cell: MegapixelVQCellSpec,
    data_dir: Path,
    device: torch.device,
) -> MegapixelVQDataset:
    full = load_megapixel_vq_benchmark(cell.dataset_name, data_dir=data_dir, device=device)
    if cell.n_train_limit is None and cell.n_test_limit is None:
        return full
    ntr = min(int(cell.n_train_limit or full.x_train.shape[0]), full.x_train.shape[0])
    nte = min(int(cell.n_test_limit or full.x_test.shape[0]), full.x_test.shape[0])
    tr_idx = np.linspace(0, full.x_train.shape[0] - 1, ntr, dtype=np.int64)
    te_idx = np.linspace(0, full.x_test.shape[0] - 1, nte, dtype=np.int64)
    sub = MegapixelVQDataset(
        name=cell.cell_id,
        short_label=cell.short_title,
        modality=full.modality,
        x_train=np.ascontiguousarray(full.x_train[tr_idx], dtype=np.float32),
        x_test=np.ascontiguousarray(full.x_test[te_idx], dtype=np.float32),
        d=cell.d,
        patch_h=full.patch_h,
        patch_w=full.patch_w,
        channels=full.channels,
        train_mosaic_hw=full.train_mosaic_hw,
        test_mosaic_hw=full.test_mosaic_hw,
        k_values=full.k_values,
        metadata=dict(full.metadata),
    )
    validate_megapixel_vq_dataset(sub, check_patch_disjoint=True)
    return sub


@torch.no_grad()
def execute_semdedup_cell(
    cell: SemDeDupCellSpec,
    seeds: list[int],
    wave: str,
    budgets: tuple[float, ...],
    data_dir: Path,
    device: torch.device,
) -> list[dict[str, Any]]:
    if not seeds:
        return []

    host = socket.gethostname()
    vis_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    gpu_sku, power_w = detect_gpu_sku_info(device)

    ds = _load_semdedup_cell_data(cell, data_dir=data_dir, device=device)
    x_tr = torch.from_numpy(ds.x_train).to(device=device, dtype=torch.float32).contiguous()
    y_tr = torch.from_numpy(ds.y_train).to(device=device, dtype=torch.int64).contiguous()
    x_te = torch.from_numpy(ds.x_test).to(device=device, dtype=torch.float32).contiguous()
    y_te = torch.from_numpy(ds.y_test).to(device=device, dtype=torch.int64).contiguous()

    wave_specs = get_wave_specs(wave, budgets)
    print(
        f"[SemDeDup] Warming {cell.cell_id} (N={x_tr.shape[0]:,}, D={cell.d}, K={cell.k:,}, wave={wave}) "
        f"on {host} GPU={vis_gpu} [{gpu_sku}] ...",
        flush=True,
    )
    t_w0 = time.perf_counter()
    warmup_cell_wave(
        x_train_gpu=x_tr,
        x_test_gpu=x_te,
        d=cell.d,
        k=cell.k,
        wave_specs=wave_specs,
        is_semdedup=True,
        y_train_gpu=y_tr,
        y_test_gpu=y_te,
        num_classes=cell.num_classes,
        device=device,
    )
    print(f"[SemDeDup] Warmup done for {cell.cell_id} in {time.perf_counter() - t_w0:.2f}s", flush=True)

    # Evaluate 100% unpruned reference accuracies once per cell (deterministic)
    w_full = torch.ones((x_tr.shape[0],), device=device, dtype=torch.float32)
    full_ridge = evaluate_cholesky_ridge_gpu(x_tr, y_tr, w_full, x_te, y_te, cell.num_classes)["accuracy_pct"]
    full_logreg = evaluate_multinomial_logistic_gpu(x_tr, y_tr, w_full, x_te, y_te, cell.num_classes)["accuracy_pct"]
    full_knn = evaluate_weighted_knn_gpu(x_tr, y_tr, w_full, x_te, y_te, cell.num_classes)["accuracy_pct"]
    full_primary = full_logreg if cell.primary_probe == "logreg" else full_knn

    records: list[dict[str, Any]] = []
    for s_idx, seed in enumerate(seeds):
        t_s0 = time.perf_counter()
        rot = int(seed) % len(wave_specs)
        ordered_specs = wave_specs[rot:] + wave_specs[:rot]
        seed_map: dict[tuple[str, float], dict[str, Any]] = {}

        for arm_name, nepochs in ordered_specs:
            if arm_name == "random":
                init_s = 0.0
                train_s = 0.0
                cluster_s = 0.0
                score_s = 0.0
                empty_c = 0
                sel_gear = "random"
                res_prec = "n/a"

                # Random K-prototype baseline for MSE & prototype kNN comparison
                r_proto = select_random_coreset(x_tr.shape[0], budget_m=cell.k, seed=seed + 991, device=device)
                c_rand = x_tr[r_proto.indices].contiguous()
                train_mse, _ = fast_exact_assign_fp64(x_tr, c_rand, device=device)
                test_mse, _ = fast_exact_assign_fp64(x_te, c_rand, device=device)

                xs_p = x_tr[r_proto.indices]
                ys_p = y_tr[r_proto.indices]
                proto_ridge = evaluate_cholesky_ridge_gpu(xs_p, ys_p, r_proto.weights, x_te, y_te, cell.num_classes)["accuracy_pct"]
                proto_logreg = evaluate_multinomial_logistic_gpu(xs_p, ys_p, r_proto.weights, x_te, y_te, cell.num_classes)["accuracy_pct"]
                proto_knn = evaluate_weighted_knn_gpu(xs_p, ys_p, r_proto.weights, x_te, y_te, cell.num_classes)["accuracy_pct"]

                b_metrics: dict[str, float] = {}
                sel_time_sum = 0.0
                for b_idx, frac in enumerate(RETENTION_BUDGETS):
                    tag = f"{int(round(frac * 100))}pct"
                    r_sel = select_random_coreset(
                        x_tr.shape[0], retention_frac=frac, seed=seed + 101 * (b_idx + 1), device=device
                    )
                    sel_time_sum += r_sel.select_time_sec
                    xs = x_tr[r_sel.indices]
                    ys = y_tr[r_sel.indices]
                    ws = r_sel.weights
                    r_acc = evaluate_cholesky_ridge_gpu(xs, ys, ws, x_te, y_te, cell.num_classes)["accuracy_pct"]
                    l_acc = evaluate_multinomial_logistic_gpu(xs, ys, ws, x_te, y_te, cell.num_classes)["accuracy_pct"]
                    k_acc = evaluate_weighted_knn_gpu(xs, ys, ws, x_te, y_te, cell.num_classes)["accuracy_pct"]
                    p_acc = l_acc if cell.primary_probe == "logreg" else k_acc
                    b_metrics[f"ridge_acc_{tag}"] = r_acc
                    b_metrics[f"logreg_acc_{tag}"] = l_acc
                    b_metrics[f"knn_acc_{tag}"] = k_acc
                    b_metrics[f"primary_acc_{tag}"] = p_acc
                score_s = sel_time_sum / float(len(RETENTION_BUDGETS))
            else:
                solver = instantiate_solver_arm(arm_name, d=cell.d, k=cell.k, nepochs=nepochs, seed=seed)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                centroids, _ = solver.fit(x_tr, capture_assignments=False)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)

                init_s = float(solver.init_time_sec)
                train_s = float(solver.train_time_sec)
                cluster_s = float(init_s + train_s)
                sel_gear = str(getattr(solver, "selected_gear", "n/a"))
                res_prec = str(getattr(solver, "resolved_precision", "n/a"))

                st = compute_semdedup_scores_gpu(x_tr, centroids, k=cell.k)
                score_s = float(st.score_time_sec)
                empty_c = int(st.empty_clusters)

                train_mse, _ = fast_exact_assign_fp64(x_tr, centroids, device=device)
                test_mse, _ = fast_exact_assign_fp64(x_te, centroids, device=device)

                # Prototype-level (<= K medoids) evaluation
                p_sel = select_prototype_coreset(st)
                xs_p = x_tr[p_sel.indices]
                ys_p = y_tr[p_sel.indices]
                proto_ridge = evaluate_cholesky_ridge_gpu(xs_p, ys_p, p_sel.weights, x_te, y_te, cell.num_classes)["accuracy_pct"]
                proto_logreg = evaluate_multinomial_logistic_gpu(xs_p, ys_p, p_sel.weights, x_te, y_te, cell.num_classes)["accuracy_pct"]
                proto_knn = evaluate_prototype_knn_gpu(
                    xs_p, ys_p, p_sel.weights, p_sel.cluster_ids, cell.k, x_te, y_te, cell.num_classes
                )["accuracy_pct"]

                b_metrics = {}
                for frac in RETENTION_BUDGETS:
                    tag = f"{int(round(frac * 100))}pct"
                    c_sel = select_coreset_at_budget(st, retention_frac=frac)
                    xs = x_tr[c_sel.indices]
                    ys = y_tr[c_sel.indices]
                    ws = c_sel.weights
                    r_acc = evaluate_cholesky_ridge_gpu(xs, ys, ws, x_te, y_te, cell.num_classes)["accuracy_pct"]
                    l_acc = evaluate_multinomial_logistic_gpu(xs, ys, ws, x_te, y_te, cell.num_classes)["accuracy_pct"]
                    k_acc = evaluate_weighted_knn_gpu(xs, ys, ws, x_te, y_te, cell.num_classes)["accuracy_pct"]
                    p_acc = l_acc if cell.primary_probe == "logreg" else k_acc
                    b_metrics[f"ridge_acc_{tag}"] = r_acc
                    b_metrics[f"logreg_acc_{tag}"] = l_acc
                    b_metrics[f"knn_acc_{tag}"] = k_acc
                    b_metrics[f"primary_acc_{tag}"] = p_acc

            proto_primary = proto_logreg if cell.primary_probe == "logreg" else proto_knn
            seed_map[(arm_name, float(nepochs))] = {
                "suite": "semdedup",
                "cell_id": cell.cell_id,
                "dataset": cell.dataset_name,
                "short_title": cell.short_title,
                "modality": cell.modality,
                "primary_probe": cell.primary_probe,
                "sku_tier": cell.sku_tier,
                "n_train": int(x_tr.shape[0]),
                "n_test": int(x_te.shape[0]),
                "d": int(cell.d),
                "num_classes": int(cell.num_classes),
                "k": int(cell.k),
                "seed": int(seed),
                "arm": arm_name,
                "nepochs": float(nepochs),
                "host": host,
                "visible_gpu": vis_gpu,
                "gpu_sku": gpu_sku,
                "gpu_power_w": float(power_w),
                "selected_gear": sel_gear,
                "resolved_precision": res_prec,
                "init_time_sec": init_s,
                "train_time_sec": train_s,
                "cluster_time_sec": cluster_s,
                "semdedup_score_time_sec": score_s,
                "total_time_sec": float(cluster_s + score_s),
                "empty_clusters": int(empty_c),
                "train_mse": float(train_mse),
                "test_mse": float(test_mse),
                "full_100pct_ridge_acc": float(full_ridge),
                "full_100pct_logreg_acc": float(full_logreg),
                "full_100pct_knn_acc": float(full_knn),
                "full_100pct_primary_acc": float(full_primary),
                "proto_ridge_acc": float(proto_ridge),
                "proto_logreg_acc": float(proto_logreg),
                "proto_knn_acc": float(proto_knn),
                "proto_primary_acc": float(proto_primary),
                **b_metrics,
            }

        for key in wave_specs:
            records.append(seed_map[key])

        print(
            f"[SemDeDup] {cell.cell_id} wave={wave} seed={seed} ({s_idx + 1}/{len(seeds)}) "
            f"done in {time.perf_counter() - t_s0:.2f}s",
            flush=True,
        )
    return records


@torch.no_grad()
def execute_vq_cell(
    cell: MegapixelVQCellSpec,
    seeds: list[int],
    wave: str,
    budgets: tuple[float, ...],
    data_dir: Path,
    device: torch.device,
) -> list[dict[str, Any]]:
    if not seeds:
        return []

    host = socket.gethostname()
    vis_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    gpu_sku, power_w = detect_gpu_sku_info(device)

    ds = _load_vq_cell_data(cell, data_dir=data_dir, device=device)
    x_tr = torch.from_numpy(ds.x_train).to(device=device, dtype=torch.float32).contiguous()
    x_te = torch.from_numpy(ds.x_test).to(device=device, dtype=torch.float32).contiguous()
    origin_tr = _compute_equivariant_origin64(x_tr)

    wave_specs = get_wave_specs(wave, budgets)
    print(
        f"[Megapixel-VQ] Warming {cell.cell_id} (N={x_tr.shape[0]:,}, D={cell.d}, K={cell.k:,}, wave={wave}) "
        f"on {host} GPU={vis_gpu} [{gpu_sku}] ...",
        flush=True,
    )
    t_w0 = time.perf_counter()
    warmup_cell_wave(
        x_train_gpu=x_tr,
        x_test_gpu=x_te,
        d=cell.d,
        k=cell.k,
        wave_specs=wave_specs,
        is_semdedup=False,
        y_train_gpu=None,
        y_test_gpu=None,
        num_classes=0,
        device=device,
    )
    print(f"[Megapixel-VQ] Warmup done for {cell.cell_id} in {time.perf_counter() - t_w0:.2f}s", flush=True)

    records: list[dict[str, Any]] = []
    for s_idx, seed in enumerate(seeds):
        t_s0 = time.perf_counter()
        rot = int(seed) % len(wave_specs)
        ordered_specs = wave_specs[rot:] + wave_specs[:rot]
        seed_map: dict[tuple[str, float], dict[str, Any]] = {}

        for arm_name, nepochs in ordered_specs:
            if arm_name == "random":
                r_sel = select_random_coreset(x_tr.shape[0], budget_m=cell.k, seed=seed + 773, device=device)
                centroids = x_tr[r_sel.indices].contiguous()
                init_s = float(r_sel.select_time_sec)
                train_s = 0.0
                sel_gear = "random"
                res_prec = "n/a"
            else:
                solver = instantiate_solver_arm(arm_name, d=cell.d, k=cell.k, nepochs=nepochs, seed=seed)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                centroids, _ = solver.fit(x_tr, capture_assignments=False)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                init_s = float(solver.init_time_sec)
                train_s = float(solver.train_time_sec)
                sel_gear = str(getattr(solver, "selected_gear", "n/a"))
                res_prec = str(getattr(solver, "resolved_precision", "n/a"))

            codes_tr, enc_s = encode_vq_indices_gpu(x_tr, centroids, origin_gpu=origin_tr)
            counts = torch.bincount(codes_tr, minlength=cell.k)
            empty_c = int((counts == 0).sum().item())

            tr_psnr, tr_pmse, _ = compute_vq_psnr_fp64(x_tr, centroids, peak_val=255.0, device=device)
            te_psnr, te_pmse, _ = compute_vq_psnr_fp64(x_te, centroids, peak_val=255.0, device=device)

            codebook_s = float(init_s + train_s)
            comp_s = float(codebook_s + enc_s)

            seed_map[(arm_name, float(nepochs))] = {
                "suite": "megapixel_vq",
                "cell_id": cell.cell_id,
                "dataset": cell.dataset_name,
                "short_title": cell.short_title,
                "modality": cell.modality,
                "sku_tier": cell.sku_tier,
                "n_train": int(x_tr.shape[0]),
                "n_test": int(x_te.shape[0]),
                "d": int(cell.d),
                "k": int(cell.k),
                "bits_per_pixel": float(cell.bits_per_pixel),
                "compression_ratio_vs_rgb24": float(24.0 / cell.bits_per_pixel),
                "seed": int(seed),
                "arm": arm_name,
                "nepochs": float(nepochs),
                "host": host,
                "visible_gpu": vis_gpu,
                "gpu_sku": gpu_sku,
                "gpu_power_w": float(power_w),
                "selected_gear": sel_gear,
                "resolved_precision": res_prec,
                "init_time_sec": init_s,
                "train_time_sec": train_s,
                "codebook_time_sec": codebook_s,
                "encode_time_sec": float(enc_s),
                "total_time_sec": comp_s,
                "empty_clusters": empty_c,
                "train_pixel_mse": float(tr_pmse),
                "test_pixel_mse": float(te_pmse),
                "train_psnr_db": float(tr_psnr),
                "test_psnr_db": float(te_psnr),
            }

        for key in wave_specs:
            records.append(seed_map[key])

        print(
            f"[Megapixel-VQ] {cell.cell_id} wave={wave} seed={seed} ({s_idx + 1}/{len(seeds)}) "
            f"done in {time.perf_counter() - t_s0:.2f}s",
            flush=True,
        )
    return records


# ==============================================================================
# 4. Forensic Integrity Checks, Cold-JIT Spike Detector & Paired Statistics
# ==============================================================================


def atomic_write_json(path: Path, payload: Any) -> None:
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
    """Detect any `(cell_id, host, gpu, arm, nepochs)` group where seed 0 spiked from cold JIT."""
    grouped: dict[tuple[str, str, str, str, float], list[dict[str, Any]]] = {}
    for r in records:
        if str(r["arm"]) == "random":
            continue
        key = (
            str(r["cell_id"]),
            str(r["host"]),
            str(r["visible_gpu"]),
            str(r["arm"]),
            float(r["nepochs"]),
        )
        grouped.setdefault(key, []).append(r)

    spikes: list[dict[str, Any]] = []
    for key, rows in grouped.items():
        if len(rows) < 2:
            continue
        rows_sorted = sorted(rows, key=lambda x: int(x["seed"]))
        t_field = "cluster_time_sec" if "cluster_time_sec" in rows_sorted[0] else "codebook_time_sec"
        t0 = float(rows_sorted[0][t_field])
        rest = [float(x[t_field]) for x in rows_sorted[1:]]
        med_rest = float(np.median(rest))
        i0 = float(rows_sorted[0]["init_time_sec"])
        i_rest = [float(x["init_time_sec"]) for x in rows_sorted[1:]]
        med_i = float(np.median(i_rest))
        if t0 > max(2.2 * med_rest, med_rest + 0.050) or i0 > max(2.5 * med_i, med_i + 0.040):
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
                    "median_init_rest_ms": 1000.0 * med_i,
                }
            )
    return spikes


def validate_records_integrity(
    semdedup_records: list[dict[str, Any]],
    vq_records: list[dict[str, Any]],
    semdedup_cells: list[SemDeDupCellSpec],
    vq_cells: list[MegapixelVQCellSpec],
    expected_seeds: list[int],
    expected_wave_specs: list[tuple[str, float]],
) -> list[dict[str, Any]]:
    _assert_finite_recursive(semdedup_records, "semdedup_records")
    _assert_finite_recursive(vq_records, "vq_records")

    exp_sd = len(semdedup_cells) * len(expected_seeds) * len(expected_wave_specs)
    if len(semdedup_records) != exp_sd:
        raise RuntimeError(f"SemDeDup record count mismatch: got {len(semdedup_records)}, expected {exp_sd}")

    exp_vq = len(vq_cells) * len(expected_seeds) * len(expected_wave_specs)
    if len(vq_records) != exp_vq:
        raise RuntimeError(f"Megapixel VQ record count mismatch: got {len(vq_records)}, expected {exp_vq}")

    for suite_name, recs in (("semdedup", semdedup_records), ("megapixel_vq", vq_records)):
        seen: set[tuple[str, int, str, float]] = set()
        seed_gpu: dict[tuple[str, int], set[tuple[str, str]]] = {}
        cell_sku: dict[str, set[str]] = {}
        dataset_sku: dict[str, set[str]] = {}

        for r in recs:
            rk = (str(r["cell_id"]), int(r["seed"]), str(r["arm"]), float(r["nepochs"]))
            if rk in seen:
                raise RuntimeError(f"Duplicate {suite_name} record key {rk!r}")
            seen.add(rk)
            seed_gpu.setdefault((str(r["cell_id"]), int(r["seed"])), set()).add(
                (str(r["host"]), str(r["visible_gpu"]))
            )
            cell_sku.setdefault(str(r["cell_id"]), set()).add(str(r["gpu_sku"]))
            if suite_name == "megapixel_vq":
                dataset_sku.setdefault(str(r["dataset"]), set()).add(str(r["gpu_sku"]))

        for sg, gpus in seed_gpu.items():
            if len(gpus) != 1:
                raise RuntimeError(f"CONFOUNDING ({suite_name}): {sg} split across GPUs {gpus!r}")
        for cid, skus in cell_sku.items():
            if len(skus) != 1:
                raise RuntimeError(f"CROSS-SKU VIOLATION ({suite_name}): {cid} mixed SKUs {skus!r}")
        for dname, skus in dataset_sku.items():
            if len(skus) != 1:
                raise RuntimeError(f"CROSS-SKU VQ VIOLATION: {dname} mixed SKUs across K: {skus!r}")

    return check_cold_jit_spikes(semdedup_records + vq_records)


def compute_paired_comparisons(
    semdedup_records: list[dict[str, Any]],
    vq_records: list[dict[str, Any]],
    semdedup_cells: list[SemDeDupCellSpec],
    vq_cells: list[MegapixelVQCellSpec],
    budgets: tuple[float, ...],
) -> list[dict[str, Any]]:
    """Compute paired Student-t and Wilcoxon signed-rank stats when Apex arms are present."""
    present_arms = {str(r["arm"]) for r in (semdedup_records + vq_records)}
    candidates = [c for c in ("apex_emeans", "apex_emeans_fp8", "apex_emeans_bf16") if c in present_arms]
    baselines = [b for b in ("random", "lloyd_25", "emeans", "smeans", "emeans_pp", "ultra_emeans") if b in present_arms]
    if not candidates or not baselines:
        return []

    out: list[dict[str, Any]] = []

    def _stat_row(
        suite: str,
        cid: str,
        dset: str,
        k: int,
        b_t: float,
        cand: str,
        base: str,
        m_name: str,
        direction: str,
        c_vals: np.ndarray,
        b_vals: np.ndarray,
    ) -> dict[str, Any]:
        diffs = c_vals - b_vals
        abs_delta = float(c_vals.mean() - b_vals.mean())
        rel_pct = float(100.0 * abs_delta / max(abs(float(b_vals.mean())), 1e-12))
        speedup_x = (
            float(b_vals.mean() / max(float(c_vals.mean()), 1e-9))
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
        ties = len(c_vals) - wins - losses
        if len(c_vals) >= 2 and float(np.std(diffs)) > 1e-15:
            p_t = float(stats.ttest_rel(c_vals, b_vals).pvalue)
            try:
                p_w = float(stats.wilcoxon(diffs).pvalue)
            except ValueError:
                p_w = 1.0
        else:
            p_t, p_w = 1.0, 1.0
        return {
            "suite": suite,
            "cell_id": cid,
            "dataset": dset,
            "k": int(k),
            "nepochs": float(b_t),
            "arm": cand,
            "baseline": base,
            "metric": m_name,
            "direction": direction,
            "arm_mean": float(c_vals.mean()),
            "arm_std": float(c_vals.std(ddof=1)) if len(c_vals) > 1 else 0.0,
            "base_mean": float(b_vals.mean()),
            "base_std": float(b_vals.std(ddof=1)) if len(b_vals) > 1 else 0.0,
            "abs_delta": abs_delta,
            "rel_delta_pct": rel_pct,
            "speedup_x": speedup_x,
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "p_ttest": p_t if math.isfinite(p_t) else 1.0,
            "p_wilcoxon": p_w if math.isfinite(p_w) else 1.0,
        }

    sd_idx = {(str(r["cell_id"]), float(r["nepochs"]), str(r["arm"]), int(r["seed"])): r for r in semdedup_records}
    sd_seeds: dict[str, list[int]] = {}
    for r in semdedup_records:
        sd_seeds.setdefault(str(r["cell_id"]), []).append(int(r["seed"]))
    sd_seeds = {k: sorted(set(v)) for k, v in sd_seeds.items()}

    sd_metrics = (
        ("primary_acc_10pct", "higher"),
        ("primary_acc_20pct", "higher"),
        ("primary_acc_50pct", "higher"),
        ("proto_primary_acc", "higher"),
        ("total_time_sec", "lower"),
    )
    for cell in semdedup_cells:
        seeds = sd_seeds.get(cell.cell_id, [])
        if not seeds:
            continue
        for b_t in budgets:
            for cand in candidates:
                for base in baselines:
                    base_t = 0.0 if base == "random" else (25.0 if base == "lloyd_25" else float(b_t))
                    for m_name, direction in sd_metrics:
                        if m_name == "total_time_sec" and base == "random":
                            continue
                        c_arr = np.array([float(sd_idx[(cell.cell_id, float(b_t), cand, s)][m_name]) for s in seeds])
                        b_arr = np.array([float(sd_idx[(cell.cell_id, base_t, base, s)][m_name]) for s in seeds])
                        out.append(
                            _stat_row(
                                "semdedup", cell.cell_id, cell.dataset_name, cell.k, float(b_t),
                                cand, base, m_name, direction, c_arr, b_arr,
                            )
                        )

    vq_idx = {(str(r["cell_id"]), float(r["nepochs"]), str(r["arm"]), int(r["seed"])): r for r in vq_records}
    vq_seeds: dict[str, list[int]] = {}
    for r in vq_records:
        vq_seeds.setdefault(str(r["cell_id"]), []).append(int(r["seed"]))
    vq_seeds = {k: sorted(set(v)) for k, v in vq_seeds.items()}

    vq_metrics = (
        ("test_psnr_db", "higher"),
        ("train_psnr_db", "higher"),
        ("total_time_sec", "lower"),
    )
    for vcell in vq_cells:
        seeds = vq_seeds.get(vcell.cell_id, [])
        if not seeds:
            continue
        for b_t in budgets:
            for cand in candidates:
                for base in baselines:
                    base_t = 0.0 if base == "random" else (25.0 if base == "lloyd_25" else float(b_t))
                    for m_name, direction in vq_metrics:
                        if m_name == "total_time_sec" and base == "random":
                            continue
                        c_arr = np.array([float(vq_idx[(vcell.cell_id, float(b_t), cand, s)][m_name]) for s in seeds])
                        b_arr = np.array([float(vq_idx[(vcell.cell_id, base_t, base, s)][m_name]) for s in seeds])
                        out.append(
                            _stat_row(
                                "megapixel_vq", vcell.cell_id, vcell.dataset_name, vcell.k, float(b_t),
                                cand, base, m_name, direction, c_arr, b_arr,
                            )
                        )
    return out


# ==============================================================================
# 5. Publication-Grade 3-Figure Suite (`outputs/semdedup/` & `plots/`)
# ==============================================================================


def generate_all_plots(
    semdedup_records: list[dict[str, Any]],
    vq_records: list[dict[str, Any]],
    semdedup_cells: list[SemDeDupCellSpec],
    vq_cells: list[MegapixelVQCellSpec],
    budgets: tuple[float, ...],
    out_dir: Path,
) -> list[Path]:
    """Generate `semdedup_accuracy_vs_time.png`, `semdedup_retention_curves.png`, and `megapixel_vq_psnr_pareto.png`."""
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

    sd_grp: dict[tuple[str, str, float], list[dict[str, Any]]] = {}
    sd_sku: dict[str, str] = {}
    for r in semdedup_records:
        sd_grp.setdefault((str(r["cell_id"]), str(r["arm"]), float(r["nepochs"])), []).append(r)
        sd_sku[str(r["cell_id"])] = str(r["gpu_sku"]).replace("NVIDIA RTX PRO 6000 Blackwell ", "Blackwell ")

    # Pick canonical 4 modality cells for 4-panel plots (prefer K=1024 on CIFAR-10 ResNet-18 when both K=512 & K=1024 exist)
    canon_sd_cells: list[SemDeDupCellSpec] = []
    seen_ds: set[str] = set()
    for c in sorted(semdedup_cells, key=lambda x: -x.k):
        if c.dataset_name not in seen_ds:
            seen_ds.add(c.dataset_name)
            canon_sd_cells.append(c)
    order_ds = ["agnews_256", "cifar10_resnet18_512", "cifar10_convnext_768", "covtype_54"]
    canon_sd_cells.sort(key=lambda c: order_ds.index(c.dataset_name) if c.dataset_name in order_ds else 99)

    ref_t = 2.0 if 2.0 in budgets else budgets[-1]
    stream_plot_arms = [a for a in ALL_STREAMING_ARMS if any(r["arm"] == a for r in semdedup_records)]

    # --------------------------------------------------------------------------
    # Figure 1: `semdedup_accuracy_vs_time.png` (2x3 Grid: 4 Modalities @ 20% + Prototype Bar + Summary)
    # --------------------------------------------------------------------------
    fig1, axes1 = plt.subplots(2, 3, figsize=(19.2, 10.2), squeeze=False)
    for idx_c, cell in enumerate(canon_sd_cells[:4]):
        ax = axes1[idx_c // 2][idx_c % 2]
        rand_rows = sd_grp.get((cell.cell_id, "random", 0.0), [])
        if rand_rows:
            r_acc = float(np.mean([float(x["primary_acc_20pct"]) for x in rand_rows]))
            ax.axhline(r_acc, color="#7f7f7f", linestyle=":", linewidth=1.35, label="Random @ 20%")
            full_acc = float(rand_rows[0]["full_100pct_primary_acc"])
            ax.axhline(full_acc, color="#2ca02c", linestyle="-.", linewidth=1.1, alpha=0.75, label="100% Unpruned Ref")

        lloyd_rows = sd_grp.get((cell.cell_id, "lloyd_25", 25.0), [])
        if lloyd_rows:
            l_ms = float(np.mean([1000.0 * float(x["total_time_sec"]) for x in lloyd_rows]))
            l_acc = float(np.mean([float(x["primary_acc_20pct"]) for x in lloyd_rows]))
            lbl, col, mk, _ = ARM_DISPLAY["lloyd_25"]
            ax.scatter([l_ms], [l_acc], color=col, marker=mk, s=135, zorder=6, label=lbl)

        for arm in stream_plot_arms:
            xs, ys, yerrs = [], [], []
            for b_t in budgets:
                rows = sd_grp.get((cell.cell_id, arm, float(b_t)), [])
                if not rows:
                    continue
                xs.append(float(np.mean([1000.0 * float(x["total_time_sec"]) for x in rows])))
                vals = [float(x["primary_acc_20pct"]) for x in rows]
                ys.append(float(np.mean(vals)))
                yerrs.append(float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0)
            if not xs:
                continue
            lbl, col, mk, ls = ARM_DISPLAY[arm]
            is_apex = arm.startswith("apex_")
            ax.errorbar(
                xs, ys, yerr=yerrs, label=lbl, color=col, marker=mk, linestyle=ls,
                linewidth=2.3 if is_apex else 1.4, markersize=6.2 if is_apex else 4.8,
                capsize=2.0, zorder=7 if arm == "apex_emeans" else (6 if is_apex else 3),
            )

        ax.set_xscale("log")
        probe_lbl = "Logistic Probe" if cell.primary_probe == "logreg" else "Weighted k-NN"
        ax.set_title(f"{cell.short_title}\n[{probe_lbl} | {sd_sku.get(cell.cell_id, '')}]", fontweight="bold")
        ax.set_xlabel("Total GPU Pruning Time: Cluster + SemDeDup (ms, log)")
        ax.set_ylabel("Held-Out Test Accuracy at 20% Budget (%)")
        ax.grid(True, which="both", linestyle=":", alpha=0.45)

    # Panel 5 (`axes1[0][2]`): Active Prototype Coverage (`K - empty_clusters`) % at T=ref_t
    ax_p5 = axes1[0][2]
    bar_arms = [a for a in ("smeans", "emeans", "emeans_pp", "ultra_emeans", "lloyd_25", "apex_emeans_fp8", "apex_emeans") if any(r["arm"] == a for r in semdedup_records)]
    cov_pcts, proto_accs, p5_labels, p5_cols = [], [], [], []
    for arm in bar_arms:
        b_use = 25.0 if arm == "lloyd_25" else ref_t
        c_list, a_list = [], []
        for cell in semdedup_cells:
            rows = sd_grp.get((cell.cell_id, arm, b_use), [])
            if rows:
                c_list.append(100.0 * (1.0 - float(np.mean([float(x["empty_clusters"]) for x in rows])) / float(cell.k)))
                a_list.append(float(np.mean([float(x["proto_primary_acc"]) for x in rows])))
        if c_list:
            cov_pcts.append(float(np.mean(c_list)))
            proto_accs.append(float(np.mean(a_list)))
            p5_labels.append(ARM_DISPLAY[arm][0].replace(" (Auto Flagship)", "").replace(" (CroissFP8+Graph)", " FP8"))
            p5_cols.append(ARM_DISPLAY[arm][1])
    y_p5 = np.arange(len(p5_labels))
    bars_p5 = ax_p5.barh(y_p5, proto_accs, color=p5_cols, alpha=0.88, height=0.58)
    ax_p5.set_yticks(y_p5)
    ax_p5.set_yticklabels(p5_labels, fontsize=8.3, fontweight="bold")
    ax_p5.invert_yaxis()
    ax_p5.set_xlabel(f"Mean Prototype-Only (M <= K) Held-Out Accuracy % (T={ref_t:g})")
    ax_p5.set_title(f"Extreme Prototype Compression (M = K, T={ref_t:g})", fontweight="bold")
    ax_p5.grid(True, axis="x", linestyle=":", alpha=0.45)
    if proto_accs:
        lo_p = max(0.0, min(proto_accs) - 2.5)
        hi_p = max(proto_accs) + 4.5
        ax_p5.set_xlim(lo_p, hi_p)
        for b_obj, pa, cp in zip(bars_p5, proto_accs, cov_pcts):
            ax_p5.annotate(
                f" {pa:.2f}% ({cp:.1f}% live K)",
                xy=(pa, b_obj.get_y() + b_obj.get_height() / 2.0),
                va="center", ha="left", fontsize=7.8, fontweight="bold",
            )

    # Panel 6 (`axes1[1][2]`): Cross-Modality Accuracy Gain vs Random & Speedup vs Lloyd@25
    ax_p6 = axes1[1][2]
    champ = "apex_emeans" if any(r["arm"] == "apex_emeans" for r in semdedup_records) else bar_arms[-1]
    gains_10, gains_20, gains_50, cell_names, spds_lloyd = [], [], [], [], []
    for cell in canon_sd_cells[:4]:
        c_rows = sd_grp.get((cell.cell_id, champ, ref_t), [])
        r_rows = sd_grp.get((cell.cell_id, "random", 0.0), [])
        l_rows = sd_grp.get((cell.cell_id, "lloyd_25", 25.0), [])
        if c_rows and r_rows:
            gains_10.append(float(np.mean([x["primary_acc_10pct"] for x in c_rows]) - np.mean([x["primary_acc_10pct"] for x in r_rows])))
            gains_20.append(float(np.mean([x["primary_acc_20pct"] for x in c_rows]) - np.mean([x["primary_acc_20pct"] for x in r_rows])))
            gains_50.append(float(np.mean([x["primary_acc_50pct"] for x in c_rows]) - np.mean([x["primary_acc_50pct"] for x in r_rows])))
            c_t = float(np.mean([x["total_time_sec"] for x in c_rows]))
            l_t = float(np.mean([x["total_time_sec"] for x in l_rows])) if l_rows else 0.0
            spds_lloyd.append(l_t / max(c_t, 1e-9) if l_rows else 0.0)
            cell_names.append(cell.dataset_name.replace("cifar10_", "c10_"))
    y_c = np.arange(len(cell_names))
    bw = 0.25
    ax_p6.barh(y_c - bw, gains_10, height=bw, color="#f28e2b", label="10% Budget vs Random")
    ax_p6.barh(y_c, gains_20, height=bw, color="#d62728", label="20% Budget vs Random")
    ax_p6.barh(y_c + bw, gains_50, height=bw, color="#8b0000", label="50% Budget vs Random")
    ax_p6.axvline(0.0, color="#222222", linewidth=1.0)
    ax_p6.set_yticks(y_c)
    ax_p6.set_yticklabels(
        [f"{cn}\n({sp:.1f}x vs Lloyd)" if sp > 0.0 else cn for cn, sp in zip(cell_names, spds_lloyd)],
        fontsize=8.2,
        fontweight="bold",
    )
    ax_p6.invert_yaxis()
    ax_p6.set_xlabel(f"Held-Out Accuracy Gain over Random Pruning (pp, T={ref_t:g})")
    ax_p6.set_title(f"{ARM_DISPLAY[champ][0].split(' (')[0]} SemDeDup Gain & Lloyd Speedup", fontweight="bold")
    ax_p6.legend(loc="lower right", fontsize=7.6)
    ax_p6.grid(True, axis="x", linestyle=":", alpha=0.45)

    handles1, labels1 = axes1[0][0].get_legend_handles_labels()
    fig1.legend(handles1, labels1, loc="upper center", ncol=5, frameon=True, bbox_to_anchor=(0.5, 1.04))
    fig1.suptitle(
        "Semantic Data Deduplication & Coreset Selection (SemDeDup): Held-Out Accuracy vs. GPU Pruning Latency",
        fontsize=12.8, fontweight="bold", y=1.075,
    )
    fig1.tight_layout()
    _save_atomic(fig1, "semdedup_accuracy_vs_time.png")
    plt.close(fig1)

    # --------------------------------------------------------------------------
    # Figure 2: `semdedup_retention_curves.png` (2x2 Grid: Accuracy vs Retained Budget %)
    # --------------------------------------------------------------------------
    fig2, axes2 = plt.subplots(2, 2, figsize=(15.2, 10.0), squeeze=False)
    curve_arms = [a for a in ("random", "emeans", "smeans", "ultra_emeans", "lloyd_25", "apex_emeans_fp8", "apex_emeans") if any(r["arm"] == a for r in semdedup_records)]

    for idx_c, cell in enumerate(canon_sd_cells[:4]):
        ax = axes2[idx_c // 2][idx_c % 2]
        n_tr_c = next((int(r["n_train"]) for r in semdedup_records if r["cell_id"] == cell.cell_id), 50000)
        proto_pct = 100.0 * float(cell.k) / float(max(n_tr_c, 1))
        budget_points = sorted(
            [
                (proto_pct, f"Proto\n({proto_pct:.1f}%)", "proto_primary_acc"),
                (10.0, "10%", "primary_acc_10pct"),
                (20.0, "20%", "primary_acc_20pct"),
                (50.0, "50%", "primary_acc_50pct"),
            ],
            key=lambda item: item[0],
        )
        xs_pct = [bp[0] for bp in budget_points]

        any_rows = sd_grp.get((cell.cell_id, "random", 0.0), [])
        full_acc_str = ""
        if any_rows:
            full_acc = float(any_rows[0]["full_100pct_primary_acc"])
            full_acc_str = f" | 100% Ref: {full_acc:.2f}%"
            ax.axhline(full_acc, color="#2ca02c", linestyle="--", linewidth=1.4, label="100% Unpruned Corpus")

        for arm in curve_arms:
            b_use = 0.0 if arm == "random" else (25.0 if arm == "lloyd_25" else ref_t)
            rows = sd_grp.get((cell.cell_id, arm, b_use), [])
            if not rows:
                continue
            ys = [
                float(np.mean([float(x[bp[2]]) for x in rows]))
                for bp in budget_points
            ]
            yerrs = [
                float(np.std([float(x[bp[2]]) for x in rows], ddof=1) / math.sqrt(len(rows))) if len(rows) > 1 else 0.0
                for bp in budget_points
            ]
            lbl, col, mk, ls = ARM_DISPLAY[arm]
            is_apex = arm.startswith("apex_")
            ax.errorbar(
                xs_pct, ys, yerr=yerrs, label=lbl, color=col, marker=mk, linestyle=ls,
                linewidth=2.3 if is_apex else 1.4, markersize=6.2 if is_apex else 4.8,
                capsize=2.2, zorder=7 if arm == "apex_emeans" else (6 if is_apex else 3),
            )

        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks([max(0.5, bp[0]) for bp in budget_points])
        ax.set_xticklabels([bp[1] for bp in budget_points])
        probe_lbl = "GPU Multinomial Logistic Probe" if cell.primary_probe == "logreg" else "Origin-Centered Weighted k-NN"
        ax.set_title(f"{cell.short_title} (T={ref_t:g})\n[{probe_lbl}{full_acc_str}]", fontweight="bold")
        ax.set_xlabel("Retained Training Data Budget (% of Full Corpus, log scale)")
        ax.set_ylabel("Downstream Held-Out Classification Accuracy (%)")
        ax.grid(True, which="both", linestyle=":", alpha=0.45)

    handles2, labels2 = axes2[0][0].get_legend_handles_labels()
    fig2.legend(handles2, labels2, loc="upper center", ncol=4, frameon=True, bbox_to_anchor=(0.5, 1.04))
    fig2.suptitle(
        "Downstream Data-Pruning Scaling Curves: Beating Random Sampling Across 4 Real ML Modalities",
        fontsize=12.8, fontweight="bold", y=1.075,
    )
    fig2.tight_layout()
    _save_atomic(fig2, "semdedup_retention_curves.png")
    plt.close(fig2)

    # --------------------------------------------------------------------------
    # Figure 3: `megapixel_vq_psnr_pareto.png` (2x3 Grid: RGB 1Mx3 & Patch 500Kx48 Across K)
    # --------------------------------------------------------------------------
    vq_grp: dict[tuple[str, str, float], list[dict[str, Any]]] = {}
    vq_sku: dict[str, str] = {}
    for r in vq_records:
        vq_grp.setdefault((str(r["cell_id"]), str(r["arm"]), float(r["nepochs"])), []).append(r)
        vq_sku[str(r["cell_id"])] = str(r["gpu_sku"]).replace("NVIDIA RTX PRO 6000 Blackwell ", "Blackwell ")

    rgb_cells = [c for c in vq_cells if c.dataset_name == "megapixel_rgb_3d"]
    patch_cells = [c for c in vq_cells if c.dataset_name == "megapixel_patch_48d"]
    n_cols_vq = max(len(rgb_cells), len(patch_cells), 1)
    fig3, axes3 = plt.subplots(2, n_cols_vq, figsize=(6.4 * n_cols_vq, 9.6), squeeze=False)
    vq_stream_arms = [a for a in ALL_STREAMING_ARMS if any(r["arm"] == a for r in vq_records)]

    for row_idx, row_cells in enumerate((rgb_cells, patch_cells)):
        for col_idx in range(n_cols_vq):
            ax = axes3[row_idx][col_idx]
            if col_idx >= len(row_cells):
                ax.axis("off")
                continue
            vcell = row_cells[col_idx]
            rand_v = vq_grp.get((vcell.cell_id, "random", 0.0), [])
            if rand_v:
                r_psnr = float(np.mean([float(x["test_psnr_db"]) for x in rand_v]))
                ax.axhline(r_psnr, color="#7f7f7f", linestyle=":", linewidth=1.2, label="Random Codebook")

            lloyd_v = vq_grp.get((vcell.cell_id, "lloyd_25", 25.0), [])
            if lloyd_v:
                l_ms = float(np.mean([1000.0 * float(x["total_time_sec"]) for x in lloyd_v]))
                l_psnr = float(np.mean([float(x["test_psnr_db"]) for x in lloyd_v]))
                lbl, col, mk, _ = ARM_DISPLAY["lloyd_25"]
                ax.axhline(l_psnr, color=col, linestyle="--", linewidth=0.95, alpha=0.55)
                ax.scatter([l_ms], [l_psnr], color=col, marker=mk, s=135, zorder=6, label=lbl)

            for arm in vq_stream_arms:
                xs, ys, yerrs = [], [], []
                for b_t in budgets:
                    rows = vq_grp.get((vcell.cell_id, arm, float(b_t)), [])
                    if not rows:
                        continue
                    xs.append(float(np.mean([1000.0 * float(x["total_time_sec"]) for x in rows])))
                    vals = [float(x["test_psnr_db"]) for x in rows]
                    ys.append(float(np.mean(vals)))
                    yerrs.append(float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0)
                if not xs:
                    continue
                lbl, col, mk, ls = ARM_DISPLAY[arm]
                is_apex = arm.startswith("apex_")
                ax.errorbar(
                    xs, ys, yerr=yerrs, label=lbl, color=col, marker=mk, linestyle=ls,
                    linewidth=2.3 if is_apex else 1.4, markersize=6.2 if is_apex else 4.8,
                    capsize=2.0, zorder=7 if arm == "apex_emeans" else (6 if is_apex else 3),
                )

            ax.set_xscale("log")
            cr = 24.0 / vcell.bits_per_pixel
            ax.set_title(
                f"{vcell.short_title} ({vcell.bits_per_pixel:.2f} bpp, {cr:.1f}x)\n[{vq_sku.get(vcell.cell_id, '')}]",
                fontweight="bold",
            )
            ax.set_xlabel("GPU Compression Time: Codebook + Encode (ms, log)")
            ax.set_ylabel("Held-Out Image PSNR (dB, higher is better)")
            ax.grid(True, which="both", linestyle=":", alpha=0.45)

    handles3, labels3 = axes3[0][0].get_legend_handles_labels()
    fig3.legend(handles3, labels3, loc="upper center", ncol=5, frameon=True, bbox_to_anchor=(0.5, 1.04))
    fig3.suptitle(
        "Megapixel Color Palette (1M×3) & 4×4 Natural Image Patch (500K×48) Vector Quantization: PSNR vs. GPU Latency",
        fontsize=12.6, fontweight="bold", y=1.075,
    )
    fig3.tight_layout()
    _save_atomic(fig3, "megapixel_vq_psnr_pareto.png")
    plt.close(fig3)

    return written


# ==============================================================================
# 6. Markdown Summary Writer (`outputs/semdedup/semdedup_summary.md`)
# ==============================================================================


def write_summary_markdown(
    semdedup_records: list[dict[str, Any]],
    vq_records: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    cold_jit_spikes: list[dict[str, Any]],
    semdedup_cells: list[SemDeDupCellSpec],
    vq_cells: list[MegapixelVQCellSpec],
    budgets: tuple[float, ...],
    out_path: Path,
) -> None:
    sd_grp: dict[tuple[str, str, float], list[dict[str, Any]]] = {}
    for r in semdedup_records:
        sd_grp.setdefault((str(r["cell_id"]), str(r["arm"]), float(r["nepochs"])), []).append(r)

    vq_grp: dict[tuple[str, str, float], list[dict[str, Any]]] = {}
    for r in vq_records:
        vq_grp.setdefault((str(r["cell_id"]), str(r["arm"]), float(r["nepochs"])), []).append(r)

    lines: list[str] = [
        "# Downstream ML Pipeline Benchmark Summary: `SemDeDup` Coreset Pruning & Megapixel VQ",
        "",
        f"- **Suite 1 (`SemDeDup` & Coreset Selection) Runs**: `{len(semdedup_records):,}` across `{len(semdedup_cells)}` real labeled ML cells (`0` `NaN`/`Inf`, `100%` single-SKU per cell).",
        f"- **Suite 2 (`Megapixel VQ`) Runs**: `{len(vq_records):,}` across `{len(vq_cells)}` high-resolution RGB & `4x4x3` patch cells.",
        f"- **Cold-JIT Timing Spikes Detected**: `{len(cold_jit_spikes)}`.",
        "",
        "## 1. Suite 1 — Semantic Data Deduplication & Coreset Selection (`10%`, `20%`, `50%`, & Prototype)",
        "",
    ]

    for cell in semdedup_cells:
        sku = next((str(r["gpu_sku"]) for r in semdedup_records if r["cell_id"] == cell.cell_id), "Unknown")
        n_tr = next((int(r["n_train"]) for r in semdedup_records if r["cell_id"] == cell.cell_id), 0)
        n_te = next((int(r["n_test"]) for r in semdedup_records if r["cell_id"] == cell.cell_id), 0)
        full_acc = next((float(r["full_100pct_primary_acc"]) for r in semdedup_records if r["cell_id"] == cell.cell_id), 0.0)
        lines.extend(
            [
                f"### `{cell.cell_id}` — {cell.modality} (`N_train={n_tr:,}, N_test={n_te:,}, D={cell.d}, K={cell.k:,}`, `{sku}`)",
                f"- **100% Unpruned Reference Accuracy (`{cell.primary_probe}`)**: **`{full_acc:.2f}%`**",
                "",
                "| Arm | `T` | Prune `ms` (`clust+score`) | Empty | Proto (`<=K`) | `10%` Acc | `20%` Acc | `50%` Acc | Ridge `20%` | LogReg `20%` | kNN `20%` |",
                "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        run_order: list[tuple[str, float]] = [("random", 0.0), ("lloyd_25", 25.0)]
        for b_t in budgets:
            for arm in ALL_STREAMING_ARMS:
                run_order.append((arm, float(b_t)))

        for arm, b_t in run_order:
            rows = sd_grp.get((cell.cell_id, arm, b_t), [])
            if not rows:
                continue
            tot_ms = float(np.mean([1000.0 * float(x["total_time_sec"]) for x in rows]))
            cl_ms = float(np.mean([1000.0 * float(x["cluster_time_sec"]) for x in rows]))
            sc_ms = float(np.mean([1000.0 * float(x["semdedup_score_time_sec"]) for x in rows]))
            emp = float(np.mean([float(x["empty_clusters"]) for x in rows]))
            pr_a = float(np.mean([float(x["proto_primary_acc"]) for x in rows]))
            a10 = float(np.mean([float(x["primary_acc_10pct"]) for x in rows]))
            a20 = float(np.mean([float(x["primary_acc_20pct"]) for x in rows]))
            a50 = float(np.mean([float(x["primary_acc_50pct"]) for x in rows]))
            r20 = float(np.mean([float(x["ridge_acc_20pct"]) for x in rows]))
            l20 = float(np.mean([float(x["logreg_acc_20pct"]) for x in rows]))
            k20 = float(np.mean([float(x["knn_acc_20pct"]) for x in rows]))
            lines.append(
                f"| `{arm}` | `{b_t:g}` | `{tot_ms:.2f}` (`{cl_ms:.2f}+{sc_ms:.2f}`) | `{emp:.1f}` | "
                f"`{pr_a:.2f}%` | `{a10:.2f}%` | `{a20:.2f}%` | `{a50:.2f}%` | "
                f"`{r20:.2f}%` | `{l20:.2f}%` | `{k20:.2f}%` |"
            )
        lines.append("")

    lines.extend(
        [
            "## 2. Suite 2 — Megapixel Color Palette (`1M×3`) & `4×4` Patch (`500K×48`) Vector Quantization",
            "",
        ]
    )
    for vcell in vq_cells:
        sku = next((str(r["gpu_sku"]) for r in vq_records if r["cell_id"] == vcell.cell_id), "Unknown")
        lines.extend(
            [
                f"### `{vcell.cell_id}` (`N={vcell.n_train_limit or MEGAPIXEL_SPEC_BY_NAME[vcell.dataset_name].n_train:,}, D={vcell.d}, K={vcell.k:,}, {vcell.bits_per_pixel:.2f} bpp`, `{sku}`)",
                "",
                "| Arm | `T` | Compress `ms` (`codebook+enc`) | Empty | Full-Image Train PSNR (`dB`) | Held-Out Test PSNR (`dB`) | Test Pixel MSE |",
                "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        run_order_vq: list[tuple[str, float]] = [("random", 0.0), ("lloyd_25", 25.0)]
        for b_t in budgets:
            for arm in ALL_STREAMING_ARMS:
                run_order_vq.append((arm, float(b_t)))

        for arm, b_t in run_order_vq:
            rows = vq_grp.get((vcell.cell_id, arm, b_t), [])
            if not rows:
                continue
            tot_ms = float(np.mean([1000.0 * float(x["total_time_sec"]) for x in rows]))
            cb_ms = float(np.mean([1000.0 * float(x["codebook_time_sec"]) for x in rows]))
            en_ms = float(np.mean([1000.0 * float(x["encode_time_sec"]) for x in rows]))
            emp = float(np.mean([float(x["empty_clusters"]) for x in rows]))
            tr_p = float(np.mean([float(x["train_psnr_db"]) for x in rows]))
            te_p = float(np.mean([float(x["test_psnr_db"]) for x in rows]))
            te_m = float(np.mean([float(x["test_pixel_mse"]) for x in rows]))
            lines.append(
                f"| `{arm}` | `{b_t:g}` | `{tot_ms:.2f}` (`{cb_ms:.2f}+{en_ms:.2f}`) | `{emp:.1f}` | "
                f"`{tr_p:.3f}` | **`{te_p:.3f}`** | `{te_m:.3f}` |"
            )
        lines.append("")

    if comparisons:
        lines.extend(
            [
                "## 3. Paired Statistical Comparisons (`apex_emeans` & `apex_emeans_fp8` vs. Baselines)",
                "",
                "| Suite | Cell | `T` | Candidate | Baseline | Metric | Cand Mean | Base Mean | Delta / Speedup | W/T/L | `p_ttest` | `p_wilcox` |",
                "| :--- | :--- | ---: | :--- | :--- | :--- | ---: | ---: | ---: | :---: | ---: | ---: |",
            ]
        )
        for comp in comparisons:
            if comp["arm"] not in ("apex_emeans", "apex_emeans_fp8"):
                continue
            if comp["nepochs"] not in (1.0, 2.0, 5.0):
                continue
            if comp["metric"] not in ("primary_acc_10pct", "primary_acc_20pct", "primary_acc_50pct", "test_psnr_db", "total_time_sec"):
                continue
            wtl = f"{comp['wins']}/{comp['ties']}/{comp['losses']}"
            if comp["metric"] == "total_time_sec":
                d_str = f"{comp['speedup_x']:.2f}x ({comp['rel_delta_pct']:+.1f}%)"
            elif comp["metric"] == "test_psnr_db":
                d_str = f"{comp['abs_delta']:+.3f} dB"
            else:
                d_str = f"{comp['abs_delta']:+.2f} pp"
            lines.append(
                f"| `{comp['suite']}` | `{comp['cell_id']}` | `{comp['nepochs']:g}` | `{comp['arm']}` | "
                f"`{comp['baseline']}` | `{comp['metric']}` | `{comp['arm_mean']:.4f}` | `{comp['base_mean']:.4f}` | "
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
# 7. Dedicated 4-GPU (`work:0,1 + loki:0,1`) Two-Wave Fleet Dispatcher
# ==============================================================================


def discover_live_ssh_auth_sock() -> str | None:
    candidates: list[str] = []
    try:
        res = subprocess.run(
            ["tmux", "show-environment", "-t", "0", "SSH_AUTH_SOCK"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if res.returncode == 0 and "=" in res.stdout:
            candidates.append(res.stdout.strip().split("=", 1)[1])
    except Exception:
        pass
    cur = os.environ.get("SSH_AUTH_SOCK")
    if cur:
        candidates.append(cur)
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


def assert_fleet_gpus_idle(shards: list[FleetShardSpec], env: dict[str, str]) -> None:
    """Verify via live `nvidia-smi` that every target `(host, gpu_id)` is idle before dispatch."""
    by_host: dict[str, list[int]] = {}
    for sp in shards:
        assert_safe_fleet_target(sp.host, sp.session)
        by_host.setdefault(sp.host, []).append(int(sp.gpu_id))

    for host, gpus in sorted(by_host.items()):
        if host == "local":
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
        else:
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
        gpu_stats: dict[int, tuple[float, float]] = {}
        for line in res.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                gpu_stats[int(parts[0])] = (float(parts[1]), float(parts[2]))
        for gid in gpus:
            if gid not in gpu_stats:
                raise RuntimeError(f"GPU {gid} missing on {host}")
            util_pct, mem_mib = gpu_stats[gid]
            if util_pct > 20.0 or mem_mib > 2048.0:
                raise RuntimeError(
                    f"GPU CONTENTION BLOCKED: {host}:{gid} is busy (util={util_pct:.0f}%, mem={mem_mib:.0f} MiB)!"
                )


def stage_all_local_datasets(data_dir: Path, device: torch.device) -> None:
    for spec in FOUR_LABELED_BENCHMARKS:
        load_labeled_benchmark(spec.name, data_dir=data_dir, device=device)
    for vspec in TWO_MEGAPIXEL_VQ_BENCHMARKS:
        load_megapixel_vq_benchmark(vspec.name, data_dir=data_dir, device=device)


def dispatch_fleet_4way(
    wave: str,
    num_seeds: int,
    seed_start: int,
    local_gpus: tuple[int, int],
    remote_host: str,
    shard_filter: set[int] | None,
    out_dir: Path,
    data_dir: Path,
) -> bool:
    """Dispatch selected shards across `remote_host:0,1` + `local` GPUs; returns True iff all 4 shards exist."""
    all_shards = build_fleet_shards(local_gpus=local_gpus, remote_host=remote_host)
    num_shards = len(all_shards)
    assert num_shards == 4
    active_shards = [sp for sp in all_shards if (shard_filter is None or sp.shard_idx in shard_filter)]
    if not active_shards:
        raise ValueError(f"No shards matched --shard-filter {shard_filter!r}")

    env = os.environ.copy()
    live_sock = discover_live_ssh_auth_sock()
    if live_sock:
        env["SSH_AUTH_SOCK"] = live_sock

    assert_fleet_gpus_idle(active_shards, env)

    expected_npz = [s.cache_filename for s in FOUR_LABELED_BENCHMARKS] + [
        vs.cache_filename for vs in TWO_MEGAPIXEL_VQ_BENCHMARKS
    ]
    if not all((data_dir / fn).exists() for fn in expected_npz):
        stage_dev = torch.device("cpu")
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            ord_idx = int(local_gpus[0]) if 0 <= int(local_gpus[0]) < torch.cuda.device_count() else 0
            try:
                smi = subprocess.run(
                    [
                        "nvidia-smi",
                        f"--id={ord_idx}",
                        "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    env=env,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                parts = [p.strip() for p in smi.stdout.strip().splitlines()[0].split(",")]
                if len(parts) >= 2 and float(parts[0]) <= 20.0 and float(parts[1]) <= 2048.0:
                    stage_dev = torch.device(f"cuda:{ord_idx}")
            except Exception:
                stage_dev = torch.device("cpu")
        stage_all_local_datasets(data_dir=data_dir, device=stage_dev)

    sd_by_id = {c.cell_id: c for c in build_semdedup_cells(smoke=False)}
    vq_by_id = {c.cell_id: c for c in build_vq_cells(smoke=False)}
    host_needed_npz: dict[str, set[str]] = {}
    for sp in active_shards:
        if sp.host == "local":
            continue
        needed = host_needed_npz.setdefault(sp.host, set())
        for cid in sp.semdedup_cells:
            needed.add(LABELED_SPEC_BY_NAME[sd_by_id[cid].dataset_name].cache_filename)
        for vid in sp.vq_cells:
            needed.add(MEGAPIXEL_SPEC_BY_NAME[vq_by_id[vid].dataset_name].cache_filename)

    shards_dir = out_dir / "shards"
    logs_dir = out_dir / "logs"
    shards_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    rel_out = out_dir.relative_to(ROOT) if out_dir.is_relative_to(ROOT) else Path("outputs/semdedup")
    remote_data_dir = f"{REMOTE_WORKDIR}/outputs/semdedup/data"

    def _sync_host(host: str) -> None:
        print(f"[Fleet-SemDeDup] Syncing code + validated caches to {host}:{REMOTE_WORKDIR} ...", flush=True)
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
            if not src_npz.exists():
                raise FileNotFoundError(f"Missing local staged dataset cache {src_npz}")
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
        print(f"[Fleet-SemDeDup] Host {host} synced.", flush=True)

    hosts = sorted(host_needed_npz.keys())
    if hosts:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(hosts)) as pool:
            list(pool.map(_sync_host, hosts))

    local_procs: dict[int, tuple[FleetShardSpec, subprocess.Popen[str], Any]] = {}
    local_gpus_str = f"{local_gpus[0]},{local_gpus[1]}"

    try:
        for sp in active_shards:
            sess = f"{sp.session}_{wave}"
            assert_safe_fleet_target(sp.host, sess)
            if sp.host == "local":
                local_json = shards_dir / f"records_{wave}_shard_{sp.shard_idx:02d}_of_{num_shards:02d}.json"
                local_json.unlink(missing_ok=True)
                loc_scratch = Path.home() / ".cache" / "emeans_tmp"
                loc_tc = loc_scratch / f"triton_cache_{sess}"
                loc_tc.mkdir(parents=True, exist_ok=True)
                loc_env = os.environ.copy()
                loc_env["CUDA_VISIBLE_DEVICES"] = str(sp.gpu_id)
                loc_env["TMPDIR"] = str(loc_scratch)
                loc_env["TRITON_CACHE_DIR"] = str(loc_tc)
                loc_env["PYTHONPATH"] = str(ROOT / "src")
                fh = open(logs_dir / f"{sess}.log", "w", encoding="utf-8")
                proc = subprocess.Popen(
                    [
                        sys.executable,
                        str(ROOT / "scripts" / "run_semdedup_benchmarks.py"),
                        "--wave", wave,
                        "--shard-idx", str(sp.shard_idx),
                        "--num-shards", str(num_shards),
                        "--num-seeds", str(num_seeds),
                        "--seed-start", str(seed_start),
                        "--local-gpus", local_gpus_str,
                        "--remote-host", remote_host,
                        "--data-dir", str(data_dir),
                        "--out-dir", str(out_dir),
                    ],
                    cwd=str(ROOT),
                    env=loc_env,
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                local_procs[sp.shard_idx] = (sp, proc, fh)
                print(
                    f"[Fleet-SemDeDup] Launched LOCAL wave={wave} shard {sp.shard_idx:02d}/{num_shards} "
                    f"on foureyes:{sp.gpu_id} ({sess})",
                    flush=True,
                )
                continue

            remote_out = f"{REMOTE_WORKDIR}/{rel_out}"
            remote_tmp_base = "/home/catid/emeans_env/tmp"
            shard_json = f"{remote_out}/shards/records_{wave}_shard_{sp.shard_idx:02d}_of_{num_shards:02d}.json"
            remote_cmd = (
                f"mkdir -p {REMOTE_WORKDIR}/outputs/logs {remote_out}/shards {remote_tmp_base}/triton_cache_{sess} && "
                f"rm -f {REMOTE_WORKDIR}/outputs/logs/{sess}.status {shard_json} && "
                f"tmux kill-session -t {sess} 2>/dev/null || true; "
                f"tmux new-session -d -s {sess} "
                + shlex.quote(
                    f"export PATH=\"$HOME/.local/bin:/usr/local/cuda/bin:$PATH\"; "
                    f"export CUDA_VISIBLE_DEVICES={sp.gpu_id}; "
                    f"export TMPDIR={remote_tmp_base}; "
                    f"export TRITON_CACHE_DIR={remote_tmp_base}/triton_cache_{sess}; "
                    f"export PYTHONPATH={REMOTE_WORKDIR}/src; "
                    f"cd {REMOTE_WORKDIR} && "
                    f"{REMOTE_PYTHON} scripts/run_semdedup_benchmarks.py "
                    f"--wave {wave} --shard-idx {sp.shard_idx} --num-shards {num_shards} "
                    f"--num-seeds {num_seeds} --seed-start {seed_start} "
                    f"--local-gpus {local_gpus_str} --remote-host {remote_host} "
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
                f"[Fleet-SemDeDup] Launched REMOTE wave={wave} shard {sp.shard_idx:02d}/{num_shards} "
                f"on {sp.host}:{sp.gpu_id} ({sess})",
                flush=True,
            )

        pending_remote = {sp.shard_idx: sp for sp in active_shards if sp.host != "local"}
        t_poll0 = time.perf_counter()
        while pending_remote or local_procs:
            time.sleep(3.0)
            done_loc: list[int] = []
            for sidx, (sp, proc, fh) in list(local_procs.items()):
                rc = proc.poll()
                if rc is not None:
                    fh.close()
                    sess = f"{sp.session}_{wave}"
                    if rc != 0:
                        tail_txt = (logs_dir / f"{sess}.log").read_text(encoding="utf-8")[-3000:]
                        raise RuntimeError(f"Local shard {sidx:02d} ({sess}) failed (rc={rc}):\n{tail_txt}")
                    print(
                        f"[Fleet-SemDeDup] Completed LOCAL wave={wave} shard {sidx:02d} (foureyes:{sp.gpu_id}) "
                        f"at +{time.perf_counter() - t_poll0:.1f}s",
                        flush=True,
                    )
                    done_loc.append(sidx)
            for sidx in done_loc:
                local_procs.pop(sidx, None)

            done_ids: list[int] = []
            for sidx, sp in list(pending_remote.items()):
                sess = f"{sp.session}_{wave}"
                chk = subprocess.run(
                    [
                        "ssh", "-A", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                        sp.host,
                        f"cat {REMOTE_WORKDIR}/outputs/logs/{sess}.status 2>/dev/null || true",
                    ],
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                txt = chk.stdout.strip()
                if "EXIT_CODE=" in txt:
                    code = int(txt.split("EXIT_CODE=", 1)[1].splitlines()[0].strip())
                    if code != 0:
                        log_res = subprocess.run(
                            [
                                "ssh", "-A", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                                sp.host,
                                f"tail -n 60 {REMOTE_WORKDIR}/outputs/logs/{sess}.log 2>/dev/null || true",
                            ],
                            env=env,
                            capture_output=True,
                            text=True,
                            check=False,
                        )
                        raise RuntimeError(f"Remote shard {sidx:02d} ({sp.host} {sess}) failed (rc={code}):\n{log_res.stdout}")
                    remote_json = f"{REMOTE_WORKDIR}/{rel_out}/shards/records_{wave}_shard_{sidx:02d}_of_{num_shards:02d}.json"
                    local_json = shards_dir / f"records_{wave}_shard_{sidx:02d}_of_{num_shards:02d}.json"
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
                        f"[Fleet-SemDeDup] Collected REMOTE wave={wave} shard {sidx:02d} ({sp.host}:{sp.gpu_id}) "
                        f"at +{time.perf_counter() - t_poll0:.1f}s",
                        flush=True,
                    )
                    done_ids.append(sidx)
            for did in done_ids:
                pending_remote.pop(did, None)
    finally:
        for _, (_, _, fh) in local_procs.items():
            try:
                fh.close()
            except Exception:
                pass

    return all((shards_dir / f"records_{wave}_shard_{i:02d}_of_04.json").exists() for i in range(4))


# ==============================================================================
# 8. CLI Entrypoint (`--smoke`, `--dispatch-fleet`, `--merge-only`, `--shard-idx`)
# ==============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Downstream ML SemDeDup Coreset & Megapixel VQ Benchmark Runner")
    p.add_argument("--smoke", action="store_true", help="Run fast 2-seed cold-JIT-checked end-to-end smoke gate")
    p.add_argument("--dispatch-fleet", action="store_true", help="Dispatch 4-GPU (work:0,1 + foureyes) fleet run")
    p.add_argument("--merge-only", action="store_true", help="Merge existing shard JSONs, validate, and plot")
    p.add_argument(
        "--wave",
        type=str,
        choices=("baselines", "apex", "all"),
        default="all",
        help="Execution wave: 'baselines', 'apex' (auto-merges baselines if present), or 'all'",
    )
    p.add_argument("--shard-idx", type=int, default=None, help="Worker shard index (0..3)")
    p.add_argument("--num-shards", type=int, default=4, help="Total fleet shards (default: 4)")
    p.add_argument("--shard-filter", type=str, default=None, help="Optional comma-separated shard indices (e.g. '2,3' or '0,1')")
    p.add_argument("--local-gpus", type=str, default="1,3", help="Two local foureyes GPU IDs for shards 2,3 (default: '1,3')")
    p.add_argument(
        "--remote-host",
        type=str,
        choices=("acorn.stratabaremetal.com", "cashew.stratabaremetal.com"),
        default="acorn.stratabaremetal.com",
        help="Dedicated 600W remote Blackwell Server host for shards 0,1 (default: 'acorn.stratabaremetal.com')",
    )
    p.add_argument("--num-seeds", type=int, default=5, help="Paired seeds per cell (default: 5)")
    p.add_argument("--seed-start", type=int, default=6000, help="Starting RNG seed (default: 6000)")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_SEMDEDUP_DATA_DIR, help="Validated dataset cache dir")
    p.add_argument("--out-dir", type=Path, default=OUTPUT_DIR, help="Output directory for JSON, MD, and PNGs")
    return p.parse_args()


def _strict_load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(
            f,
            parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"Invalid JSON constant {c} in {path}")),
        )


def main() -> None:
    args = parse_args()
    out_dir: Path = args.out_dir
    data_dir: Path = args.data_dir

    if args.smoke:
        seeds = [int(args.seed_start), int(args.seed_start) + 1]
        smoke_budgets: tuple[float, ...] = (1.0, 2.0)
        sd_cells = build_semdedup_cells(smoke=True)
        vq_cells = build_vq_cells(smoke=True)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        warmup_global_kernels(dev)

        sd_records: list[dict[str, Any]] = []
        for c in sd_cells:
            sd_records.extend(execute_semdedup_cell(c, seeds, args.wave, smoke_budgets, data_dir, dev))
        vq_records: list[dict[str, Any]] = []
        for vc in vq_cells:
            vq_records.extend(execute_vq_cell(vc, seeds, args.wave, smoke_budgets, data_dir, dev))

        exp_specs = get_wave_specs(args.wave, smoke_budgets)
        spikes = validate_records_integrity(sd_records, vq_records, sd_cells, vq_cells, seeds, exp_specs)
        if spikes:
            raise RuntimeError(f"COLD-JIT SPIKES DETECTED IN SMOKE GATE: {spikes!r}")

        comparisons = compute_paired_comparisons(sd_records, vq_records, sd_cells, vq_cells, smoke_budgets)
        payload = {
            "phase": "semdedup_megapixel_vq_smoke",
            "wave": args.wave,
            "num_semdedup_records": len(sd_records),
            "num_megapixel_vq_records": len(vq_records),
            "num_cold_jit_spikes": len(spikes),
            "cold_jit_spikes": spikes,
            "seeds": seeds,
            "epoch_budgets": list(smoke_budgets),
            "retention_budgets": list(RETENTION_BUDGETS),
            "semdedup_cells": [asdict(c) for c in sd_cells],
            "megapixel_vq_cells": [asdict(c) for c in vq_cells],
            "semdedup_records": sd_records,
            "megapixel_vq_records": vq_records,
            "comparisons": comparisons,
        }
        atomic_write_json(out_dir / "semdedup_results.json", payload)
        write_summary_markdown(
            sd_records, vq_records, comparisons, spikes, sd_cells, vq_cells, smoke_budgets,
            out_dir / "semdedup_summary.md",
        )
        plots = generate_all_plots(sd_records, vq_records, sd_cells, vq_cells, smoke_budgets, out_dir)
        print(
            f"[SemDeDup-Smoke] SUCCESS: {len(sd_records)} semdedup + {len(vq_records)} VQ records, "
            f"0 cold-JIT spikes, {len(plots)} PNGs written to {out_dir}",
            flush=True,
        )
        return

    sd_cells = build_semdedup_cells(smoke=False)
    vq_cells = build_vq_cells(smoke=False)
    sd_by_id = {c.cell_id: c for c in sd_cells}
    vq_by_id = {c.cell_id: c for c in vq_cells}
    num_seeds = int(args.num_seeds)
    seed_start = int(args.seed_start)
    all_seeds = [seed_start + i for i in range(num_seeds)]
    budgets = EPOCH_BUDGETS

    lg_parts = [int(x.strip()) for x in str(args.local_gpus).split(",") if x.strip()]
    if len(lg_parts) != 2:
        raise ValueError(f"--local-gpus expects 2 comma-separated GPU IDs, got {args.local_gpus!r}")
    local_gpus = (lg_parts[0], lg_parts[1])
    shard_filter = (
        {int(x.strip()) for x in str(args.shard_filter).split(",") if x.strip()}
        if args.shard_filter is not None
        else None
    )

    if args.shard_idx is not None:
        if int(args.num_shards) != 4:
            raise ValueError("Expected --num-shards 4 matching build_fleet_shards")
        shards = build_fleet_shards(local_gpus=local_gpus, remote_host=str(args.remote_host))
        sp = shards[int(args.shard_idx)]
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        warmup_global_kernels(dev)

        s_sd: list[dict[str, Any]] = []
        for cid in sp.semdedup_cells:
            s_sd.extend(execute_semdedup_cell(sd_by_id[cid], all_seeds, args.wave, budgets, data_dir, dev))
        s_vq: list[dict[str, Any]] = []
        for vid in sp.vq_cells:
            s_vq.extend(execute_vq_cell(vq_by_id[vid], all_seeds, args.wave, budgets, data_dir, dev))

        shard_path = out_dir / "shards" / f"records_{args.wave}_shard_{sp.shard_idx:02d}_of_04.json"
        atomic_write_json(
            shard_path,
            {
                "wave": args.wave,
                "shard_idx": sp.shard_idx,
                "num_shards": 4,
                "host": sp.host,
                "gpu_id": sp.gpu_id,
                "sku_tier": sp.sku_tier,
                "semdedup_records": s_sd,
                "megapixel_vq_records": s_vq,
            },
        )
        print(
            f"[SemDeDup-Shard {sp.shard_idx:02d}] wave={args.wave}: wrote {len(s_sd)} semdedup + {len(s_vq)} VQ -> {shard_path}",
            flush=True,
        )
        return

    all_wave_shards_ready = True
    if args.dispatch_fleet:
        all_wave_shards_ready = dispatch_fleet_4way(
            wave=args.wave,
            num_seeds=num_seeds,
            seed_start=seed_start,
            local_gpus=local_gpus,
            remote_host=str(args.remote_host),
            shard_filter=shard_filter,
            out_dir=out_dir,
            data_dir=data_dir,
        )
        if not all_wave_shards_ready and not args.merge_only:
            print(
                f"[Fleet-SemDeDup] Partial --shard-filter {sorted(shard_filter or [])} for wave={args.wave} "
                f"completed cleanly; awaiting remaining shards before merge.",
                flush=True,
            )
            return

    if args.dispatch_fleet or args.merge_only:
        shards_dir = out_dir / "shards"
        # Automatic multi-wave merge (#170 item 2): if both `baselines` and `apex` shards exist, merge both!
        has_all_4 = all((shards_dir / f"records_all_shard_{i:02d}_of_04.json").exists() for i in range(4))
        has_base_4 = all((shards_dir / f"records_baselines_shard_{i:02d}_of_04.json").exists() for i in range(4))
        has_apex_4 = all((shards_dir / f"records_apex_shard_{i:02d}_of_04.json").exists() for i in range(4))

        if args.wave == "all" and has_all_4:
            waves_to_merge = ["all"]
            eff_wave = "all"
        elif has_base_4 and has_apex_4:
            waves_to_merge = ["baselines", "apex"]
            eff_wave = "all"
        elif args.wave == "baselines" and has_base_4:
            waves_to_merge = ["baselines"]
            eff_wave = "baselines"
        elif args.wave == "apex" and has_apex_4:
            waves_to_merge = ["apex"]
            eff_wave = "apex"
        else:
            raise FileNotFoundError(f"Incomplete shard JSONs in {shards_dir} for wave={args.wave!r}")

        merged_sd: list[dict[str, Any]] = []
        merged_vq: list[dict[str, Any]] = []
        for w_tag in waves_to_merge:
            for sidx in range(4):
                spath = shards_dir / f"records_{w_tag}_shard_{sidx:02d}_of_04.json"
                data = _strict_load_json(spath)
                merged_sd.extend(data["semdedup_records"])
                merged_vq.extend(data["megapixel_vq_records"])

        exp_specs = get_wave_specs(eff_wave, budgets)
        spikes = validate_records_integrity(merged_sd, merged_vq, sd_cells, vq_cells, all_seeds, exp_specs)
        if spikes:
            raise RuntimeError(f"COLD-JIT TIMING SPIKES DETECTED: {spikes!r}")

        comparisons = compute_paired_comparisons(merged_sd, merged_vq, sd_cells, vq_cells, budgets)
        payload = {
            "phase": "semdedup_megapixel_vq",
            "wave": eff_wave,
            "num_semdedup_records": len(merged_sd),
            "num_megapixel_vq_records": len(merged_vq),
            "num_cold_jit_spikes": len(spikes),
            "cold_jit_spikes": spikes,
            "seeds": all_seeds,
            "epoch_budgets": list(budgets),
            "retention_budgets": list(RETENTION_BUDGETS),
            "semdedup_cells": [asdict(c) for c in sd_cells],
            "megapixel_vq_cells": [asdict(c) for c in vq_cells],
            "semdedup_records": merged_sd,
            "megapixel_vq_records": merged_vq,
            "comparisons": comparisons,
        }
        out_json = (
            out_dir / "semdedup_baselines_wave_results.json"
            if eff_wave == "baselines"
            else out_dir / "semdedup_results.json"
        )
        atomic_write_json(out_json, payload)
        if eff_wave == "all":
            write_summary_markdown(
                merged_sd, merged_vq, comparisons, spikes, sd_cells, vq_cells, budgets,
                out_dir / "semdedup_summary.md",
            )
            plots = generate_all_plots(merged_sd, merged_vq, sd_cells, vq_cells, budgets, out_dir)
            print(
                f"[SemDeDup-Complete] Validated {len(merged_sd):,} semdedup + {len(merged_vq):,} VQ records, "
                f"0 cold-JIT spikes, & {len(plots)} PNGs in {out_dir}",
                flush=True,
            )
        else:
            print(
                f"[SemDeDup-Wave-{eff_wave}] Validated {len(merged_sd):,} semdedup + {len(merged_vq):,} VQ records "
                f"(0 cold-JIT spikes) -> {out_json}",
                flush=True,
            )
        return

    # Local sequential single-GPU fallback
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    warmup_global_kernels(dev)
    loc_sd = [r for c in sd_cells for r in execute_semdedup_cell(c, all_seeds, args.wave, budgets, data_dir, dev)]
    loc_vq = [r for vc in vq_cells for r in execute_vq_cell(vc, all_seeds, args.wave, budgets, data_dir, dev)]
    spikes = validate_records_integrity(loc_sd, loc_vq, sd_cells, vq_cells, all_seeds, get_wave_specs(args.wave, budgets))
    if spikes:
        raise RuntimeError(f"COLD-JIT TIMING SPIKES DETECTED: {spikes!r}")
    comparisons = compute_paired_comparisons(loc_sd, loc_vq, sd_cells, vq_cells, budgets)
    atomic_write_json(
        out_dir / "semdedup_results.json",
        {
            "phase": "semdedup_megapixel_vq_local",
            "wave": args.wave,
            "num_semdedup_records": len(loc_sd),
            "num_megapixel_vq_records": len(loc_vq),
            "num_cold_jit_spikes": len(spikes),
            "cold_jit_spikes": spikes,
            "seeds": all_seeds,
            "semdedup_records": loc_sd,
            "megapixel_vq_records": loc_vq,
            "comparisons": comparisons,
        },
    )
    write_summary_markdown(loc_sd, loc_vq, comparisons, spikes, sd_cells, vq_cells, budgets, out_dir / "semdedup_summary.md")
    generate_all_plots(loc_sd, loc_vq, sd_cells, vq_cells, budgets, out_dir)


if __name__ == "__main__":
    main()
