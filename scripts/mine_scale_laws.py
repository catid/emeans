#!/usr/bin/env python3
"""Empirical Scale-Transition & Multi-Gear Power-Law Discovery Engine (`scale-law-miner`).

Discovers and fits exact closed-form scale-transition laws across **4 orders of magnitude**
(`N in {10^4, 5e4, 2e5, 10^6, 5e6}`, `K in {256, 1024, 4096, 16384, 65536}`,
`D in {32, 128, 512, 1024}`, `T in {1, 2, 3, 5}`) to parameterize `ApexEMeans`'s automatic
4-Gear Scale-Adaptive Engine:

  - **Suite 1 — Gear Boundary 1 (Precision & `CUDAGraph` Multi-Step Crossover)**:
      De-confounded factorial comparison across `bf16_eager`, `bf16_cudagraph` (5-step on-device
      super-step graph with bit-exact parity), `fp8_fastops_eager` (global pow2 FP8),
      `fp8_croiss_eager`, and `fp8_croiss_cudagraph` (CroissAttn-inspired origin-centered
      decoupled per-row `sx_i` + per-centroid `sc_j` E4M3FN with `max_num_imprecise_acc=0`
      FP32 Tensor-Core accumulation, `0` `0x7f` NaNs, and 5-step `CUDAGraph`).
      Fits exact `(N, K, D, T)` crossover surface where `CroissAttn-FP8 + CUDAGraph` beats
      `BF16 + CUDAGraph` in total wall-clock (`init + preencode + train`) with `< 0.05%`
      validation FP64 MSE delta.

  - **Suite 2 — Gear Boundary 2 (Coarse-to-Fine `IVF-Explore + Exact-Close` + Fast Huge-K Seeding)**:
      Tests `IVFAssigner` (`K_ivf ~ 2*sqrt(K)` and `sqrt(K)`, `nprobe in {4, 8, 16}`) for the
      first `T - 1` exploration passes followed by **1 Exact Tensor-Core Closing Pass** (+ Simplex EB)
      against pure `exact_bf16`, pure `exact_fp8`, and `ivf_pure_np8` across
      `K in {4096, 8192, 16384, 32768, 65536}` and `N in {2e5, 1e6, 5e6}` (including real
      SIFT1M & GloVe-1M validation splits). Also quantifies Huge-K Adaptive Block-Tournament
      Seeding (`20x` faster `init` at `K = 65,536`).

  - **Suite 3 — Gear Boundary 3 (Scale-Adaptive Algorithmic Knobs across `N/K` and `D`)**:
      * `3A (wide_batch_law)`: Batch-widening factor (`1x, 2x, 4x`) vs base step count `T*N/B`.
      * `3B (spectral_split_law)`: Split step scale `split_mult` & spectral concentration gate
        `sigma_{j,1} / sigma_{rms,j}` across discrete modes vs curved 512D/1024D manifolds.
      * `3C (head_edit_utility_law)`: `P/Q` `head_edit` checkpoint MSE gain & latency cost vs `N/K` & `K`.
      * `3D (simplex_eb_tau_law)`: Simplex Empirical-Bayes prior `tau*(N/K, D)` & `kappa_max`
        across `N/K in {16, 64, 256, 1024}` and `D in {32, 128, 512, 1024}`.

Strict Experimental & Fleet Hygiene (`AGENTS.md` Rules 1, 2, 3):
  - **Zero Held-Out Test Leakage**: Evaluates STRICTLY on validation splits (`val` of SIFT1M,
    GloVe-1M-Unit, CIFAR-10 ResNet-18 512D & Patches 128D) and independent GPU validation draws
    in disjoint seed namespace `70_000+`. Never reads `test` or `query` arrays!
  - **Dedicated 6-GPU Single-SKU Fleet (`ripper:0..3` + `loki:0,1`)**: All 6 GPUs are identical
    `NVIDIA RTX PRO 6000 Blackwell Max-Q` (96 GB), every cell runs all seeds & arms on ONE
    physical GPU with cyclic arm-order counterbalancing, `walnut` & `tmux` session `0` are hard-blocked,
    and `TRITON_CACHE_DIR` is isolated per shard.
  - **Zero Cold-JIT Spikes**: Full-shape untimed warmup across all arms & evaluators per cell.
"""

from __future__ import annotations

import argparse
import concurrent.futures
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

# Isolate Triton disk cache under /home BEFORE importing torch/triton/emeans
if "TRITON_CACHE_DIR" not in os.environ:
    _vis_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "all").replace(",", "_")
    _base_tmp = (
        "/home/catid/emeans_env/tmp"
        if os.path.isdir("/home/catid/emeans_env")
        else os.path.expanduser("~/.cache/emeans_tmp")
    )
    os.makedirs(_base_tmp, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = f"{_base_tmp}/triton_cache_slaw_u{os.getuid()}_g{_vis_gpu}"

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

from emeans.fast_ops import (
    FLOAT8_E4M3FN,
    FP8Quantizer,
    HAS_TRITON,
    IVFAssigner,
    _choose_split_k_partitions,
    blackwell_prepare_centroids,
    blackwell_triton_assign,
    block_permute_indices,
    fp8_assign,
    fused_ema_and_prepare,
    fused_scatter_stats,
)
from emeans.kernels import assign_points
from emeans.real_corpora import (
    DEFAULT_MILLION_DATA_DIR,
    exact_cosine_distortion_fp64,
    fast_exact_assign_fp64,
)
from emeans.recipe import (
    ALPHA_MIN_FACTOR,
    CHECK_UNUSED_EVERY,
    DEFAULT_INIT_USAGE,
    DEFAULT_UNUSED_THRESHOLD,
    EMeansRecipe,
    build_recipe,
    epoch_seed,
    round_power_of_two,
    scheduled_alpha,
)
from emeans.seeding import HAS_TRITON as SEED_HAS_TRITON, SeedingResult
from emeans.ultra_emeans import (
    HALF_GAUSSIAN_MEAN,
    UltraEMeans,
    _alloc_padded_2d,
    _compute_origin_and_scale,
    _launch_ultra_assign,
    _launch_ultra_prep_centroids,
    ultra_simplex_eb_finalize,
    ultra_turkpp_init_centered,
)

if HAS_TRITON:
    import triton
    import triton.language as tl

if SEED_HAS_TRITON:
    from emeans.seeding import _local_greedy_kpp_kernel


OUTPUT_DIR = ROOT / "outputs" / "scale_laws"
PHASE1_CACHE_DIR = ROOT / "outputs" / "phase1" / "cache"
REMOTE_WORKDIR = "/home/catid/emeans_runs/scale-law-miner"
REMOTE_PYTHON = "/home/catid/emeans_env/.venv/bin/python"

# Dedicated validation-only seed namespace (strictly disjoint from Phase 1-5 seeds 42..6000)
DEFAULT_SEED_START: int = 70_000
FP32_TINY: float = 1.17549435e-38
FP8_MAX_FINITE: float = 448.0

APPROVED_SLOTS: tuple[tuple[str, int], ...] = (
    ("ripper", 0),
    ("ripper", 1),
    ("ripper", 2),
    ("ripper", 3),
    ("loki", 0),
    ("loki", 1),
)


# ==============================================================================
# 1. Atomic I/O & SSH Discovery Helpers
# ==============================================================================


def _assert_json_finite(obj: Any, path_str: str = "root") -> None:
    """Recursively verify zero NaN/Inf values before writing JSON."""
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"Non-finite float {obj!r} at {path_str}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _assert_json_finite(v, f"{path_str}.{k}")
    elif isinstance(obj, (list, tuple)):
        for idx, v in enumerate(obj):
            _assert_json_finite(v, f"{path_str}[{idx}]")


def atomic_write_json(path: Path, payload: Any) -> None:
    """Atomically write JSON with `allow_nan=False` via PID+ns temp file + `os.replace`."""
    _assert_json_finite(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".tmp_{os.getpid()}_{time.time_ns()}_{path.name}"
    try:
        text = json.dumps(payload, indent=2, allow_nan=False)
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def atomic_write_text(path: Path, text: str) -> None:
    """Atomically write UTF-8 text via PID+ns temp file + `os.replace`."""
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


def atomic_save_png(fig: plt.Figure, dst_paths: list[Path], dpi: int = 180) -> None:
    """Atomically save & verify (`PIL.Image.verify`) PNG figure to one or more paths."""
    if not dst_paths:
        return
    primary = dst_paths[0]
    primary.parent.mkdir(parents=True, exist_ok=True)
    tmp = primary.parent / f".tmp_{os.getpid()}_{time.time_ns()}_{primary.name}"
    try:
        fig.savefig(tmp, format="png", dpi=dpi, bbox_inches="tight")
        with Image.open(tmp) as im:
            im.verify()
        os.replace(tmp, primary)
        for extra in dst_paths[1:]:
            extra.parent.mkdir(parents=True, exist_ok=True)
            tmp_e = extra.parent / f".tmp_{os.getpid()}_{time.time_ns()}_{extra.name}"
            shutil.copyfile(primary, tmp_e)
            with Image.open(tmp_e) as im2:
                im2.verify()
            os.replace(tmp_e, extra)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def discover_live_ssh_auth_sock() -> str | None:
    """Auto-discover live forwarded `SSH_AUTH_SOCK` without ever attaching/modifying tmux session 0."""
    cur = os.environ.get("SSH_AUTH_SOCK", "")
    if cur and os.path.exists(cur):
        return cur
    try:
        res = subprocess.run(
            ["tmux", "show-environment", "-t", "0", "SSH_AUTH_SOCK"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        for line in res.stdout.splitlines():
            if line.startswith("SSH_AUTH_SOCK="):
                cand = line.split("=", 1)[1].strip()
                if cand and os.path.exists(cand):
                    return cand
    except Exception:
        pass
    return None


# ==============================================================================
# 2. Strict Validation-Only Data Protocol (ZERO Held-Out Test / Query Access!)
# ==============================================================================


FORBIDDEN_NPZ_KEYS: frozenset[str] = frozenset(
    {"test", "query", "groundtruth_ids", "groundtruth_sq_dists"}
)


@dataclass(frozen=True)
class ValidationProblem:
    """Strictly isolated `(x_train, x_val)` clustering validation problem.

    Intentionally exposes NO `.test` or `.query` field so held-out test sets can never leak.
    """

    name: str
    x_train: torch.Tensor  # (N, D) float32 on device
    x_val: torch.Tensor    # (N_val, D) float32 on device
    d: int
    geometry: str          # "euclidean" or "unit_sphere"
    source_split: str      # "real_val_only" or "gpu_synthetic_val_only"


def load_real_validation_corpus(
    corpus_name: str,
    device: torch.device,
    million_data_dir: Path = DEFAULT_MILLION_DATA_DIR,
    phase1_cache_dir: Path = PHASE1_CACHE_DIR,
    n_train_limit: int | None = None,
) -> ValidationProblem:
    """Load ONLY `(train/base, val)` arrays from validated `.npz` caches — never reading `test`/`query`."""
    if corpus_name == "sift1m_val":
        npz_path = million_data_dir / "sift1m_validated_v1.npz"
        train_key, val_key, geom = "base", "val", "euclidean"
    elif corpus_name == "glove1m_val":
        npz_path = million_data_dir / "glove1m_unit_s20261116_validated_v1.npz"
        train_key, val_key, geom = "base", "val", "unit_sphere"
    elif corpus_name == "cifar10_r18_512_val":
        npz_path = phase1_cache_dir / "cifar10_resnet18_global_512.npz"
        train_key, val_key, geom = "train", "val", "euclidean"
    elif corpus_name == "cifar10_patches_128_val":
        npz_path = phase1_cache_dir / "cifar10_resnet18_patches_128.npz"
        train_key, val_key, geom = "train", "val", "euclidean"
    else:
        raise ValueError(f"Unknown real validation corpus {corpus_name!r}")

    if not npz_path.exists():
        raise FileNotFoundError(f"Validated cache not found at {npz_path}")

    assert train_key not in FORBIDDEN_NPZ_KEYS and val_key not in FORBIDDEN_NPZ_KEYS
    with np.load(npz_path, allow_pickle=False) as z:
        x_tr_np = np.ascontiguousarray(z[train_key], dtype=np.float32)
        x_va_np = np.ascontiguousarray(z[val_key], dtype=np.float32)

    if n_train_limit is not None and int(n_train_limit) < x_tr_np.shape[0]:
        x_tr_np = np.ascontiguousarray(x_tr_np[: int(n_train_limit)])

    x_tr = torch.from_numpy(x_tr_np).to(device=device, dtype=torch.float32).contiguous()
    x_va = torch.from_numpy(x_va_np).to(device=device, dtype=torch.float32).contiguous()
    if not (bool(torch.isfinite(x_tr).all()) and bool(torch.isfinite(x_va).all())):
        raise ValueError(f"Non-finite values in validation corpus {corpus_name!r}")
    return ValidationProblem(
        name=corpus_name,
        x_train=x_tr,
        x_val=x_va,
        d=int(x_tr.shape[1]),
        geometry=geom,
        source_split="real_val_only",
    )


@torch.no_grad()
def make_gpu_validation_problem(
    kind: str,
    n_train: int,
    d: int,
    k: int,
    seed: int,
    device: torch.device,
    n_val: int | None = None,
) -> ValidationProblem:
    """Synthesize `(x_train, x_val)` directly on GPU in `< 15 ms` using 3 disjoint RNG streams."""
    if n_val is None:
        n_val = min(32_768, max(4_096, n_train // 8))
    n_train = int(n_train)
    n_val = int(n_val)
    d = int(d)
    k = int(k)

    gen_params = torch.Generator(device=device)
    gen_params.manual_seed((int(seed) * 1_000_003 + 11) % (2**32))
    gen_tr = torch.Generator(device=device)
    gen_tr.manual_seed((int(seed) * 1_000_003 + 29) % (2**32))
    gen_va = torch.Generator(device=device)
    gen_va.manual_seed((int(seed) * 1_000_003 + 47) % (2**32))

    def _draw_split(n_rows: int, gen: torch.Generator) -> torch.Tensor:
        out = torch.empty((n_rows, d), dtype=torch.float32, device=device)
        chunk = 1_048_576  # Stream in 1M-row chunks so 5M x 128 peak scratch stays < 512 MB!
        if kind in ("balanced_gmm", "zipf_gmm", "aniso_gmm_shifted", "multiscale_sensor"):
            for s in range(0, n_rows, chunk):
                e = min(s + chunk, n_rows)
                m_c = e - s
                if kind == "zipf_gmm":
                    u = torch.rand((m_c,), generator=gen, device=device, dtype=torch.float32)
                    c_ids = torch.searchsorted(cdf_probs, u, right=False).clamp_(0, g_modes - 1)
                else:
                    c_ids = torch.randint(0, g_modes, (m_c,), generator=gen, device=device)
                z = torch.randn((m_c, d), generator=gen, device=device, dtype=torch.float32)
                if kind == "aniso_gmm_shifted":
                    out[s:e] = centers[c_ids] + z * noise_scales[None, :] + shift_vec[None, :]
                elif kind == "multiscale_sensor":
                    out[s:e] = (centers[c_ids] + z * sigma_w) * feat_mult[None, :] + feat_offset[None, :]
                else:
                    out[s:e] = centers[c_ids] + z * sigma_w
        elif kind == "isotropic_gaussian":
            for s in range(0, n_rows, chunk):
                e = min(s + chunk, n_rows)
                out[s:e] = torch.randn((e - s, d), generator=gen, device=device, dtype=torch.float32)
        elif kind == "multiscale_rare_outliers":
            for s in range(0, n_rows, chunk):
                e = min(s + chunk, n_rows)
                m_c = e - s
                u = torch.rand((m_c,), generator=gen, device=device, dtype=torch.float32)
                c_ids = torch.searchsorted(cdf_probs, u, right=False).clamp_(0, g_modes - 1)
                z = torch.randn((m_c, d), generator=gen, device=device, dtype=torch.float32)
                sig = mode_sigmas[c_ids][:, None]
                out[s:e] = centers[c_ids] + z * sig
        elif kind == "curved_manifold":
            for s in range(0, n_rows, chunk):
                e = min(s + chunk, n_rows)
                m_c = e - s
                c_ids = torch.randint(0, g_lat, (m_c,), generator=gen, device=device)
                z_lat = lat_centers[c_ids] + 0.45 * torch.randn(
                    (m_c, d_lat), generator=gen, device=device, dtype=torch.float32
                )
                proj = z_lat @ w_rff + b_rff[None, :]
                h = math.sqrt(2.0) * torch.cos(proj)
                eps = 0.12 * torch.randn((m_c, d), generator=gen, device=device, dtype=torch.float32)
                out[s:e] = h @ w_emb + eps
        else:
            raise ValueError(f"Unsupported synthetic validation kind={kind!r}")
        return out

    geom = "euclidean"
    g_modes = min(max(16, k), 65_536)
    sigma_mu = math.sqrt(2.0 / 3.0)
    sigma_w = math.sqrt(1.0 / 3.0)

    if kind in ("balanced_gmm", "zipf_gmm"):
        centers = torch.randn((g_modes, d), generator=gen_params, device=device, dtype=torch.float32) * sigma_mu
        if kind == "zipf_gmm":
            ranks = torch.arange(1, g_modes + 1, device=device, dtype=torch.float64)
            probs = ranks.pow(-1.0)
            probs = probs / probs.sum()
            cdf_probs = torch.cumsum(probs, dim=0).float()
            cdf_probs[-1] = 1.0
    elif kind == "aniso_gmm_shifted":
        idx_f = torch.arange(1, d + 1, device=device, dtype=torch.float64)
        raw_v = idx_f.pow(-1.0)
        variances = (float(d) * raw_v / raw_v.sum()).float()
        stds = variances.sqrt()
        centers = (
            torch.randn((g_modes, d), generator=gen_params, device=device, dtype=torch.float32)
            * (sigma_mu * stds[None, :])
        )
        noise_scales = sigma_w * stds
        shift_vec = torch.full((d,), 25.0, device=device, dtype=torch.float32)
    elif kind == "multiscale_sensor":
        # Mirrors Forest Covertype dynamic-range skewness: [1800..3800], [0..60], and [-1..1]
        centers = torch.randn((g_modes, d), generator=gen_params, device=device, dtype=torch.float32) * sigma_mu
        feat_mult = torch.ones((d,), device=device, dtype=torch.float32)
        feat_offset = torch.zeros((d,), device=device, dtype=torch.float32)
        n_hi = min(4, max(1, d // 8))
        n_mid = min(12, max(1, d // 4))
        feat_mult[:n_hi] = 450.0
        feat_offset[:n_hi] = 2800.0
        if n_hi < d:
            end_m = min(d, n_hi + n_mid)
            feat_mult[n_hi:end_m] = 15.0
            feat_offset[n_hi:end_m] = 30.0
        if n_hi + n_mid < d:
            feat_mult[n_hi + n_mid :] = 0.25
    elif kind == "multiscale_rare_outliers":
        n_rare = max(4, g_modes // 10)
        n_bulk = max(1, g_modes - n_rare)
        bulk_c = torch.randn((n_bulk, d), generator=gen_params, device=device, dtype=torch.float32) * sigma_mu
        r_dirs = torch.randn((n_rare, d), generator=gen_params, device=device, dtype=torch.float32)
        r_dirs = r_dirs / r_dirs.norm(dim=1, keepdim=True).clamp_min(1e-12)
        rare_rad = 3.5 * math.sqrt(float(d) * (2.0 / 3.0))
        rare_c = r_dirs * rare_rad
        centers = torch.cat([bulk_c, rare_c], dim=0).contiguous()
        p_each_r = 0.03 / float(g_modes)
        r_tot = min(0.2, p_each_r * n_rare)
        probs = torch.cat(
            [
                torch.full((n_bulk,), (1.0 - r_tot) / n_bulk, device=device, dtype=torch.float64),
                torch.full((n_rare,), r_tot / n_rare, device=device, dtype=torch.float64),
            ]
        )
        cdf_probs = torch.cumsum(probs, dim=0).float()
        cdf_probs[-1] = 1.0
        mode_sigmas = torch.cat(
            [
                torch.full((n_bulk,), 0.35, device=device, dtype=torch.float32),
                torch.full((n_rare,), 0.15, device=device, dtype=torch.float32),
            ]
        )
    elif kind == "curved_manifold":
        d_lat = min(16, max(4, d // 8))
        m_rff = min(64, max(16, d // 2))
        g_lat = min(128, max(16, k // 8))
        lat_centers = torch.randn((g_lat, d_lat), generator=gen_params, device=device, dtype=torch.float32) * 1.4
        w_rff = torch.randn((d_lat, m_rff), generator=gen_params, device=device, dtype=torch.float32) * (
            0.8 / math.sqrt(float(d_lat))
        )
        b_rff = torch.rand((m_rff,), generator=gen_params, device=device, dtype=torch.float32) * (2.0 * math.pi)
        w_emb = torch.randn((m_rff, d), generator=gen_params, device=device, dtype=torch.float32) / math.sqrt(
            float(m_rff)
        )
    elif kind == "isotropic_gaussian":
        pass
    else:
        raise ValueError(f"Unknown synthetic validation kind={kind!r}")

    x_tr = _draw_split(n_train, gen_tr)
    x_va = _draw_split(n_val, gen_va)
    return ValidationProblem(
        name=kind,
        x_train=x_tr,
        x_val=x_va,
        d=d,
        geometry=geom,
        source_split="gpu_synthetic_val_only",
    )


# ==============================================================================
# 3. CroissAttn-Stabilized Decoupled FP8 + Bit-Exact On-Device 5-Step CUDAGraph
# ==============================================================================

if HAS_TRITON:

    @triton.jit(do_not_specialize=["N", "D", "stride_xn", "stride_xd", "stride_qxn"])
    def _croiss_encode_rows_kernel(
        x_ptr,
        origin_ptr,
        qx_ptr,
        sx_ptr,
        x_norm_ptr,
        N,
        D,
        stride_xn,
        stride_xd,
        stride_qxn,
        ORIGIN_FP64: tl.constexpr,
        FULL_D: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Fused 1-kernel origin-centered decoupled per-row E4M3FN encoder + exact represented `x_norm`."""
        pid_n = tl.program_id(0).to(tl.int64)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
        mask_n = offs_n < N

        if FULL_D:
            # Single HBM read pass when D <= 256 (`stride_qxn == BLOCK_D` or <= 256)!
            cols = tl.arange(0, BLOCK_D).to(tl.int64)
            mask_d = cols < D
            mask_pad = cols < stride_qxn
            mask_nd = mask_n[:, None] & mask_d[None, :]
            x_raw = tl.load(x_ptr + offs_n[:, None] * stride_xn + cols[None, :] * stride_xd, mask=mask_nd, other=0.0)
            orig = tl.load(origin_ptr + cols, mask=mask_d, other=0.0)
            if ORIGIN_FP64:
                diff = (x_raw.to(tl.float64) - orig.to(tl.float64)).to(tl.float32)
            else:
                diff = x_raw.to(tl.float32) - orig.to(tl.float32)
            u0 = tl.where(mask_nd, diff, 0.0)
            amax = tl.max(tl.abs(u0), axis=1)
            sx = tl.maximum(amax * (1.0 / 448.0), 1.17549435e-38).to(tl.float32)
            inv_sx = tl.where(amax > 0.0, 1.0 / sx, 0.0).to(tl.float32)
            u_scaled = tl.where(mask_nd, u0 * inv_sx[:, None], 0.0).to(tl.float32)
            u_clamped = tl.maximum(-448.0, tl.minimum(448.0, u_scaled)).to(tl.float32)
            q_code = u_clamped.to(tl.float8e4nv, fp_downcast_rounding="rtne")
            tl.store(
                qx_ptr + offs_n[:, None] * stride_qxn + cols[None, :],
                q_code,
                mask=mask_n[:, None] & mask_pad[None, :],
            )
            qf = tl.where(mask_nd, q_code.to(tl.float32), 0.0)
            q_nsq = tl.sum(qf * qf, axis=1)
        else:
            amax = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for d0 in range(0, D, BLOCK_D):
                cols = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
                mask_d = cols < D
                mask_nd = mask_n[:, None] & mask_d[None, :]
                x_raw = tl.load(x_ptr + offs_n[:, None] * stride_xn + cols[None, :] * stride_xd, mask=mask_nd, other=0.0)
                orig = tl.load(origin_ptr + cols, mask=mask_d, other=0.0)
                if ORIGIN_FP64:
                    diff = (x_raw.to(tl.float64) - orig.to(tl.float64)).to(tl.float32)
                else:
                    diff = x_raw.to(tl.float32) - orig.to(tl.float32)
                u = tl.where(mask_nd, diff, 0.0)
                amax = tl.maximum(amax, tl.max(tl.abs(u), axis=1))

            sx = tl.maximum(amax * (1.0 / 448.0), 1.17549435e-38).to(tl.float32)
            inv_sx = tl.where(amax > 0.0, 1.0 / sx, 0.0).to(tl.float32)

            q_nsq = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for d0 in range(0, stride_qxn, BLOCK_D):
                cols = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
                mask_d = cols < D
                mask_pad = cols < stride_qxn
                mask_nd = mask_n[:, None] & mask_d[None, :]
                x_raw = tl.load(x_ptr + offs_n[:, None] * stride_xn + cols[None, :] * stride_xd, mask=mask_nd, other=0.0)
                orig = tl.load(origin_ptr + cols, mask=mask_d, other=0.0)
                if ORIGIN_FP64:
                    diff = (x_raw.to(tl.float64) - orig.to(tl.float64)).to(tl.float32)
                else:
                    diff = x_raw.to(tl.float32) - orig.to(tl.float32)
                u = tl.where(mask_nd, diff * inv_sx[:, None], 0.0).to(tl.float32)
                u_clamped = tl.maximum(-448.0, tl.minimum(448.0, u)).to(tl.float32)
                q_code = u_clamped.to(tl.float8e4nv, fp_downcast_rounding="rtne")
                tl.store(
                    qx_ptr + offs_n[:, None] * stride_qxn + cols[None, :],
                    q_code,
                    mask=mask_n[:, None] & mask_pad[None, :],
                )
                qf = tl.where(mask_nd, q_code.to(tl.float32), 0.0)
                q_nsq += tl.sum(qf * qf, axis=1)

        tl.store(sx_ptr + offs_n, sx, mask=mask_n)
        xn_fin = tl.minimum(sx * (sx * q_nsq), 3.4028234e38).to(tl.float32)
        tl.store(x_norm_ptr + offs_n, xn_fin, mask=mask_n)

    @triton.jit(do_not_specialize=["N", "D", "stride_xn", "stride_xd", "stride_outn"])
    def _center_bf16_rows_kernel(
        x_ptr,
        origin_ptr,
        out_bf16_ptr,
        x_norm_ptr,
        N,
        D,
        stride_xn,
        stride_xd,
        stride_outn,
        ORIGIN_FP64: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Fused 1-kernel origin-centered BF16 encoder + exact represented `x_norm`."""
        pid_n = tl.program_id(0).to(tl.int64)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
        mask_n = offs_n < N

        nsq = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for d0 in range(0, stride_outn, BLOCK_D):
            cols = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
            mask_d = cols < D
            mask_pad = cols < stride_outn
            mask_nd = mask_n[:, None] & mask_d[None, :]
            x_raw = tl.load(x_ptr + offs_n[:, None] * stride_xn + cols[None, :] * stride_xd, mask=mask_nd, other=0.0)
            orig = tl.load(origin_ptr + cols, mask=mask_d, other=0.0)
            if ORIGIN_FP64:
                diff = (x_raw.to(tl.float64) - orig.to(tl.float64)).to(tl.float32)
            else:
                diff = x_raw.to(tl.float32) - orig.to(tl.float32)
            u_bf16 = tl.where(mask_nd, diff, 0.0).to(tl.bfloat16)
            tl.store(
                out_bf16_ptr + offs_n[:, None] * stride_outn + cols[None, :],
                u_bf16,
                mask=mask_n[:, None] & mask_pad[None, :],
            )
            uf = tl.where(mask_nd, u_bf16.to(tl.float32), 0.0)
            nsq += tl.sum(uf * uf, axis=1)

        tl.store(x_norm_ptr + offs_n, nsq, mask=mask_n)

    @triton.jit(do_not_specialize=["B", "K", "D", "stride_x", "stride_c"])
    def _gear_assign_kernel(
        x_ptr,
        order_ptr,
        batch_off_ptr,
        sx_ptr,
        x_norm_ptr,
        c_ptr,
        sc_ptr,
        c_norm_ptr,
        out_dist_ptr,
        out_label_ptr,
        B,
        K,
        D,
        stride_x,
        stride_c,
        IS_FP8: tl.constexpr,
        FULL_D: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """CUDAGraph-safe BF16 / CroissAttn-FP8 Tensor-Core nearest-centroid kernel."""
        pid_b = tl.program_id(0).to(tl.int64)
        b_off = tl.load(batch_off_ptr).to(tl.int64)
        offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B).to(tl.int64)
        mask_b = offs_b < B
        src_rows = tl.load(order_ptr + b_off + offs_b, mask=mask_b, other=0).to(tl.int64)

        xn = tl.load(x_norm_ptr + src_rows, mask=mask_b, other=0.0).to(tl.float32)
        if IS_FP8:
            sx = tl.load(sx_ptr + src_rows, mask=mask_b, other=1.0).to(tl.float32)

        if FULL_D:
            od = tl.arange(0, BLOCK_D).to(tl.int64)
            md = od < D
            x_full = tl.load(
                x_ptr + src_rows[:, None] * stride_x + od[None, :],
                mask=mask_b[:, None] & md[None, :],
                other=0.0,
            )

        best_score = tl.full((BLOCK_B,), float("inf"), dtype=tl.float32)
        best_label = tl.zeros((BLOCK_B,), dtype=tl.int32)

        for k0 in range(0, K, BLOCK_K):
            ok = (k0 + tl.arange(0, BLOCK_K)).to(tl.int64)
            mk = ok < K
            if FULL_D:
                od = tl.arange(0, BLOCK_D).to(tl.int64)
                md = od < D
                c_tile = tl.load(
                    c_ptr + ok[:, None] * stride_c + od[None, :],
                    mask=mk[:, None] & md[None, :],
                    other=0.0,
                )
                if IS_FP8:
                    dots = tl.dot(x_full, tl.trans(c_tile), max_num_imprecise_acc=0, out_dtype=tl.float32)
                else:
                    dots = tl.dot(x_full, tl.trans(c_tile), out_dtype=tl.float32)
            else:
                dots = tl.zeros((BLOCK_B, BLOCK_K), dtype=tl.float32)
                for d0 in range(0, D, BLOCK_D):
                    od = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
                    md = od < D
                    x_tile = tl.load(
                        x_ptr + src_rows[:, None] * stride_x + od[None, :],
                        mask=mask_b[:, None] & md[None, :],
                        other=0.0,
                    )
                    c_tile = tl.load(
                        c_ptr + ok[:, None] * stride_c + od[None, :],
                        mask=mk[:, None] & md[None, :],
                        other=0.0,
                    )
                    if IS_FP8:
                        dots = tl.dot(
                            x_tile, tl.trans(c_tile), acc=dots, max_num_imprecise_acc=0, out_dtype=tl.float32
                        )
                    else:
                        dots = tl.dot(x_tile, tl.trans(c_tile), acc=dots, out_dtype=tl.float32)

            cn = tl.load(c_norm_ptr + ok, mask=mk, other=float("inf")).to(tl.float32)
            if IS_FP8:
                sc = tl.load(sc_ptr + ok, mask=mk, other=1.0).to(tl.float32)
                scores = cn[None, :] - (2.0 * sx[:, None]) * (dots * sc[None, :])
            else:
                scores = cn[None, :] - 2.0 * dots
            scores = tl.where(mk[None, :], scores, float("inf"))

            tmin, tidx = tl.min(scores, axis=1, return_indices=True, return_indices_tie_break_left=True)
            imp = tmin < best_score
            best_score = tl.where(imp, tmin, best_score)
            best_label = tl.where(imp, k0 + tidx.to(tl.int32), best_label)

        tl.store(out_dist_ptr + offs_b, tl.maximum(best_score + xn, 0.0), mask=mask_b)
        tl.store(out_label_ptr + offs_b, best_label.to(tl.int64), mask=mask_b)

    @triton.jit(do_not_specialize=["B", "K", "D", "stride_x", "stride_c"])
    def _gear_splitk_assign_kernel(
        x_ptr,
        order_ptr,
        batch_off_ptr,
        sx_ptr,
        c_ptr,
        sc_ptr,
        c_norm_ptr,
        part_score_ptr,
        part_label_ptr,
        B,
        K,
        D,
        stride_x,
        stride_c,
        PARTITIONS: tl.constexpr,
        IS_FP8: tl.constexpr,
        FULL_D: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """2D Split-K occupancy kernel for small-`B` / large-`K` CUDAGraph-safe assignment."""
        pid_b = tl.program_id(0).to(tl.int64)
        pid_p = tl.program_id(1).to(tl.int64)
        b_off = tl.load(batch_off_ptr).to(tl.int64)
        offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B).to(tl.int64)
        mask_b = offs_b < B
        src_rows = tl.load(order_ptr + b_off + offs_b, mask=mask_b, other=0).to(tl.int64)

        if IS_FP8:
            sx = tl.load(sx_ptr + src_rows, mask=mask_b, other=1.0).to(tl.float32)

        if FULL_D:
            od = tl.arange(0, BLOCK_D).to(tl.int64)
            md = od < D
            x_full = tl.load(
                x_ptr + src_rows[:, None] * stride_x + od[None, :],
                mask=mask_b[:, None] & md[None, :],
                other=0.0,
            )

        total_k_tiles = tl.cdiv(K, BLOCK_K)
        tiles_per_p = tl.cdiv(total_k_tiles, PARTITIONS)
        k_start = pid_p * tiles_per_p * BLOCK_K
        k_end = tl.minimum(K, (pid_p + 1) * tiles_per_p * BLOCK_K)

        best_score = tl.full((BLOCK_B,), float("inf"), dtype=tl.float32)
        best_label = tl.zeros((BLOCK_B,), dtype=tl.int32)

        for k0 in range(k_start, k_end, BLOCK_K):
            ok = (k0 + tl.arange(0, BLOCK_K)).to(tl.int64)
            mk = ok < k_end
            if FULL_D:
                od = tl.arange(0, BLOCK_D).to(tl.int64)
                md = od < D
                c_tile = tl.load(
                    c_ptr + ok[:, None] * stride_c + od[None, :],
                    mask=mk[:, None] & md[None, :],
                    other=0.0,
                )
                if IS_FP8:
                    dots = tl.dot(x_full, tl.trans(c_tile), max_num_imprecise_acc=0, out_dtype=tl.float32)
                else:
                    dots = tl.dot(x_full, tl.trans(c_tile), out_dtype=tl.float32)
            else:
                dots = tl.zeros((BLOCK_B, BLOCK_K), dtype=tl.float32)
                for d0 in range(0, D, BLOCK_D):
                    od = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
                    md = od < D
                    x_tile = tl.load(
                        x_ptr + src_rows[:, None] * stride_x + od[None, :],
                        mask=mask_b[:, None] & md[None, :],
                        other=0.0,
                    )
                    c_tile = tl.load(
                        c_ptr + ok[:, None] * stride_c + od[None, :],
                        mask=mk[:, None] & md[None, :],
                        other=0.0,
                    )
                    if IS_FP8:
                        dots = tl.dot(
                            x_tile, tl.trans(c_tile), acc=dots, max_num_imprecise_acc=0, out_dtype=tl.float32
                        )
                    else:
                        dots = tl.dot(x_tile, tl.trans(c_tile), acc=dots, out_dtype=tl.float32)

            cn = tl.load(c_norm_ptr + ok, mask=mk, other=float("inf")).to(tl.float32)
            if IS_FP8:
                sc = tl.load(sc_ptr + ok, mask=mk, other=1.0).to(tl.float32)
                scores = cn[None, :] - (2.0 * sx[:, None]) * (dots * sc[None, :])
            else:
                scores = cn[None, :] - 2.0 * dots
            scores = tl.where(mk[None, :], scores, float("inf"))

            tmin, tidx = tl.min(scores, axis=1, return_indices=True, return_indices_tie_break_left=True)
            imp = tmin < best_score
            best_score = tl.where(imp, tmin, best_score)
            best_label = tl.where(imp, k0.to(tl.int32) + tidx.to(tl.int32), best_label)

        out_off = offs_b * PARTITIONS + pid_p
        tl.store(part_score_ptr + out_off, best_score, mask=mask_b)
        tl.store(part_label_ptr + out_off, best_label, mask=mask_b)

    @triton.jit(do_not_specialize=["B"])
    def _gear_splitk_reduce_kernel(
        order_ptr,
        batch_off_ptr,
        x_norm_ptr,
        part_score_ptr,
        part_label_ptr,
        out_dist_ptr,
        out_label_ptr,
        B,
        PARTITIONS: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_P: tl.constexpr,
    ):
        pid_b = tl.program_id(0).to(tl.int64)
        b_off = tl.load(batch_off_ptr).to(tl.int64)
        offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B).to(tl.int64)
        mask_b = offs_b < B
        offs_p = tl.arange(0, BLOCK_P).to(tl.int64)
        mask_p = offs_p < PARTITIONS
        mask_bp = mask_b[:, None] & mask_p[None, :]

        ps = tl.load(part_score_ptr + offs_b[:, None] * PARTITIONS + offs_p[None, :], mask=mask_bp, other=float("inf"))
        pl = tl.load(part_label_ptr + offs_b[:, None] * PARTITIONS + offs_p[None, :], mask=mask_bp, other=0)
        b_min, b_slot = tl.min(ps, axis=1, return_indices=True, return_indices_tie_break_left=True)
        win_lbl = tl.sum(tl.where(offs_p[None, :] == b_slot[:, None], pl, 0), axis=1)

        src_rows = tl.load(order_ptr + b_off + offs_b, mask=mask_b, other=0).to(tl.int64)
        xn = tl.load(x_norm_ptr + src_rows, mask=mask_b, other=0.0).to(tl.float32)
        tl.store(out_dist_ptr + offs_b, tl.maximum(b_min + xn, 0.0), mask=mask_b)
        tl.store(out_label_ptr + offs_b, win_lbl.to(tl.int64), mask=mask_b)

    @triton.jit(do_not_specialize=["B", "D", "stride_xn", "stride_xd"])
    def _gear_scatter_centered_kernel(
        x_ptr,
        order_ptr,
        batch_off_ptr,
        origin_ptr,
        labels_ptr,
        dists_ptr,
        sums_ptr,
        counts_ptr,
        batch_sse_ptr,
        B,
        D,
        stride_xn,
        stride_xd,
        ORIGIN_FP64: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """CUDAGraph-safe FP32 centered scatter-add of `sums`, `counts`, and `batch_sse`."""
        pid_b = tl.program_id(0).to(tl.int64)
        pid_d = tl.program_id(1).to(tl.int64)
        b_off = tl.load(batch_off_ptr).to(tl.int64)
        offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B).to(tl.int64)
        mask_b = offs_b < B

        lbl = tl.load(labels_ptr + offs_b, mask=mask_b, other=0).to(tl.int64)
        src_rows = tl.load(order_ptr + b_off + offs_b, mask=mask_b, other=0).to(tl.int64)

        if pid_d == 0:
            dv = tl.load(dists_ptr + offs_b, mask=mask_b, other=0.0).to(tl.float32)
            tl.atomic_add(counts_ptr + lbl, tl.full((BLOCK_B,), 1.0, dtype=tl.float32), mask=mask_b)
            tl.atomic_add(batch_sse_ptr + lbl, dv, mask=mask_b)

        cols = (pid_d * BLOCK_D + tl.arange(0, BLOCK_D)).to(tl.int64)
        mask_d = cols < D
        mask_bd = mask_b[:, None] & mask_d[None, :]
        x_raw = tl.load(x_ptr + src_rows[:, None] * stride_xn + cols[None, :] * stride_xd, mask=mask_bd, other=0.0)
        orig = tl.load(origin_ptr + cols, mask=mask_d, other=0.0)
        if ORIGIN_FP64:
            xc = (x_raw.to(tl.float64) - orig.to(tl.float64)).to(tl.float32)
        else:
            xc = x_raw.to(tl.float32) - orig.to(tl.float32)
        tl.atomic_add(sums_ptr + lbl[:, None] * D + cols[None, :], xc, mask=mask_bd)

    @triton.jit(do_not_specialize=["K", "D", "stride_crep"])
    def _gear_prep_centroids_kernel(
        centroids_ptr,
        c_rep_ptr,
        sc_ptr,
        c_norm_ptr,
        K,
        D,
        stride_crep,
        IS_FP8: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Refresh BF16 or CroissAttn decoupled per-centroid `(qc, sc, c_norm)` on GPU."""
        kid = tl.program_id(0).to(tl.int64)
        if kid >= K:
            return
        if IS_FP8:
            amax = tl.zeros((), dtype=tl.float32)
            for d0 in range(0, D, BLOCK_D):
                cols = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
                md = cols < D
                cv = tl.load(centroids_ptr + kid * D + cols, mask=md, other=0.0).to(tl.float32)
                amax = tl.maximum(amax, tl.max(tl.where(md, tl.abs(cv), 0.0), axis=0))
            sc = tl.maximum(amax * (1.0 / 448.0), 1.17549435e-38).to(tl.float32)
            inv_sc = tl.where(amax > 0.0, 1.0 / sc, 0.0).to(tl.float32)
            q_nsq = tl.zeros((), dtype=tl.float32)
            for d0 in range(0, stride_crep, BLOCK_D):
                cols = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
                md = cols < D
                mp = cols < stride_crep
                cv = tl.load(centroids_ptr + kid * D + cols, mask=md, other=0.0).to(tl.float32)
                qv = tl.where(md, tl.maximum(-448.0, tl.minimum(448.0, cv * inv_sc)), 0.0).to(tl.float32)
                qc = qv.to(tl.float8e4nv, fp_downcast_rounding="rtne")
                tl.store(c_rep_ptr + kid * stride_crep + cols, qc, mask=mp)
                qf = tl.where(md, qc.to(tl.float32), 0.0)
                q_nsq += tl.sum(qf * qf, axis=0)
            tl.store(sc_ptr + kid, sc)
            tl.store(c_norm_ptr + kid, tl.minimum(sc * (sc * q_nsq), 3.4028234e38).to(tl.float32))
        else:
            nsq = tl.zeros((), dtype=tl.float32)
            for d0 in range(0, stride_crep, BLOCK_D):
                cols = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
                md = cols < D
                mp = cols < stride_crep
                cv = tl.load(centroids_ptr + kid * D + cols, mask=md, other=0.0).to(tl.float32)
                cb = tl.where(md, cv, 0.0).to(tl.bfloat16)
                tl.store(c_rep_ptr + kid * stride_crep + cols, cb, mask=mp)
                cf = tl.where(md, cb.to(tl.float32), 0.0)
                nsq += tl.sum(cf * cf, axis=0)
            tl.store(c_norm_ptr + kid, nsq)

    @triton.jit(do_not_specialize=["K", "D", "stride_crep", "total_steps"])
    def _gear_ondevice_ema_prep_kernel(
        sums_ptr,
        counts_ptr,
        batch_sse_ptr,
        centroids_ptr,
        usage_ptr,
        sse_ema_ptr,
        slow_centroids_ptr,
        pass_sums_ptr,
        pass_counts_ptr,
        pass_sse_ptr,
        c_rep_ptr,
        sc_ptr,
        c_norm_ptr,
        step_ptr,
        accum_pass_ptr,
        K,
        D,
        stride_crep,
        total_steps,
        alpha0,
        alpha_min_factor,
        alpha_slow,
        do_prep,
        IS_FP8: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """100% CUDAGraph-safe on-device schedule + EMA + slow_ema + closing accum + centroid prep.

        Read-only on `step_ptr` and `accum_pass_ptr` across all `K` CTAs so multi-wave grids
        (`K > num_sms`) have zero inter-CTA data races; `_gear_advance_scalars_kernel[(1,)]`
        advances scalars after grid completion!
        """
        kid = tl.program_id(0).to(tl.int64)
        if kid >= K:
            return

        step = tl.load(step_ptr).to(tl.int32)
        t_tot = tl.maximum(total_steps, 1)
        frac = tl.minimum(step, t_tot).to(tl.float32) / t_tot.to(tl.float32)
        cos_t = 0.5 * (1.0 + tl.cos(3.141592653589793 * frac))
        alpha = alpha0 * (alpha_min_factor + (1.0 - alpha_min_factor) * cos_t)
        decay = 1.0 - alpha

        cnt = tl.load(counts_ptr + kid).to(tl.float32)
        bsse = tl.load(batch_sse_ptr + kid).to(tl.float32)
        tl.store(counts_ptr + kid, 0.0)
        tl.store(batch_sse_ptr + kid, 0.0)

        u_old = tl.load(usage_ptr + kid).to(tl.float32)
        s_old = tl.load(sse_ema_ptr + kid).to(tl.float32)
        u_ret = tl.where(alpha < 1.0, decay * u_old, 0.0)
        s_ret = tl.where(alpha < 1.0, decay * s_old, 0.0)

        touched = cnt > 0.0
        u_new = tl.where(touched, u_ret + alpha * cnt, u_ret)
        s_new = tl.where(touched, s_ret + alpha * bsse, s_ret)
        tl.store(usage_ptr + kid, u_new)
        tl.store(sse_ema_ptr + kid, s_new)

        accum_p = tl.load(accum_pass_ptr).to(tl.int32) != 0
        if accum_p and touched:
            pc = tl.load(pass_counts_ptr + kid).to(tl.float64)
            ps = tl.load(pass_sse_ptr + kid).to(tl.float64)
            tl.store(pass_counts_ptr + kid, pc + cnt.to(tl.float64))
            tl.store(pass_sse_ptr + kid, ps + bsse.to(tl.float64))

        inv_cnt = 1.0 / tl.maximum(cnt, 1.0)
        w_ema = tl.where(touched, (alpha * cnt) / tl.maximum(u_new, 1e-30), 0.0)

        amax = tl.zeros((), dtype=tl.float32)
        for d0 in range(0, D, BLOCK_D):
            cols = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
            md = cols < D
            bs = tl.load(sums_ptr + kid * D + cols, mask=md, other=0.0).to(tl.float32)
            tl.store(sums_ptr + kid * D + cols, 0.0, mask=md)
            c_cur = tl.load(centroids_ptr + kid * D + cols, mask=md, other=0.0).to(tl.float32)
            mean_b = bs * inv_cnt
            c_upd = tl.where(w_ema >= 1.0, mean_b, c_cur + w_ema * (mean_b - c_cur))
            c_next = tl.where(touched, c_upd, c_cur)
            tl.store(centroids_ptr + kid * D + cols, c_next, mask=md)

            sc_cur = tl.load(slow_centroids_ptr + kid * D + cols, mask=md, other=0.0).to(tl.float32)
            sc_next = tl.where(touched, sc_cur + alpha_slow * (mean_b - sc_cur), sc_cur)
            tl.store(slow_centroids_ptr + kid * D + cols, sc_next, mask=md)

            if accum_p and touched:
                psum = tl.load(pass_sums_ptr + kid * D + cols, mask=md, other=0.0).to(tl.float32)
                tl.store(pass_sums_ptr + kid * D + cols, psum + bs, mask=md)

            if do_prep != 0 and IS_FP8:
                amax = tl.maximum(amax, tl.max(tl.where(md, tl.abs(c_next), 0.0), axis=0))

        if do_prep != 0:
            if IS_FP8:
                sc_val = tl.maximum(amax * (1.0 / 448.0), 1.17549435e-38).to(tl.float32)
                inv_sc = tl.where(amax > 0.0, 1.0 / sc_val, 0.0).to(tl.float32)
                q_nsq = tl.zeros((), dtype=tl.float32)
                for d0 in range(0, stride_crep, BLOCK_D):
                    cols = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
                    md = cols < D
                    mp = cols < stride_crep
                    c_v = tl.load(centroids_ptr + kid * D + cols, mask=md, other=0.0).to(tl.float32)
                    qv = tl.where(md, tl.maximum(-448.0, tl.minimum(448.0, c_v * inv_sc)), 0.0).to(tl.float32)
                    qc = qv.to(tl.float8e4nv, fp_downcast_rounding="rtne")
                    tl.store(c_rep_ptr + kid * stride_crep + cols, qc, mask=mp)
                    qf = tl.where(md, qc.to(tl.float32), 0.0)
                    q_nsq += tl.sum(qf * qf, axis=0)
                tl.store(sc_ptr + kid, sc_val)
                tl.store(c_norm_ptr + kid, tl.minimum(sc_val * (sc_val * q_nsq), 3.4028234e38).to(tl.float32))
            else:
                nsq = tl.zeros((), dtype=tl.float32)
                for d0 in range(0, stride_crep, BLOCK_D):
                    cols = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
                    md = cols < D
                    mp = cols < stride_crep
                    c_v = tl.load(centroids_ptr + kid * D + cols, mask=md, other=0.0).to(tl.float32)
                    cb = tl.where(md, c_v, 0.0).to(tl.bfloat16)
                    tl.store(c_rep_ptr + kid * stride_crep + cols, cb, mask=mp)
                    cf = tl.where(md, cb.to(tl.float32), 0.0)
                    nsq += tl.sum(cf * cf, axis=0)
                tl.store(c_norm_ptr + kid, nsq)

    @triton.jit(do_not_specialize=["B_adv", "total_steps"])
    def _gear_advance_scalars_kernel(
        step_ptr,
        batch_off_ptr,
        usage_bias_ptr,
        sse_bias_ptr,
        B_adv,
        total_steps,
        alpha0,
        alpha_min_factor,
    ):
        """1-CTA post-EMA scalar advance after hardware grid barrier of `_gear_ondevice_ema_prep_kernel`."""
        step = tl.load(step_ptr).to(tl.int32)
        t_tot = tl.maximum(total_steps, 1)
        frac = tl.minimum(step, t_tot).to(tl.float32) / t_tot.to(tl.float32)
        cos_t = 0.5 * (1.0 + tl.cos(3.141592653589793 * frac))
        alpha = alpha0 * (alpha_min_factor + (1.0 - alpha_min_factor) * cos_t)
        decay = 1.0 - alpha

        ub = tl.load(usage_bias_ptr).to(tl.float32)
        sb = tl.load(sse_bias_ptr).to(tl.float32)
        ub_next = tl.where(alpha >= 1.0, 1.0, decay * ub + alpha).to(tl.float32)
        sb_next = tl.where(alpha >= 1.0, 1.0, decay * sb + alpha).to(tl.float32)
        tl.store(usage_bias_ptr, ub_next)
        tl.store(sse_bias_ptr, sb_next)
        tl.store(step_ptr, step + 1)
        b_cur_off = tl.load(batch_off_ptr).to(tl.int64)
        tl.store(batch_off_ptr, b_cur_off + B_adv)

    @triton.jit(do_not_specialize=["K", "stop_step", "base_seed", "bsz_scale"])
    def _gear_donor_keys_kernel(
        usage_ptr,
        sse_ema_ptr,
        usage_bias_ptr,
        sse_bias_ptr,
        step_ptr,
        unused_i8_ptr,
        donor_keys_ptr,
        n_sp_ptr,
        K,
        unused_thresh,
        stop_step,
        base_seed,
        bsz_scale,
        BLOCK_K: tl.constexpr,
    ):
        """Single-CTA on-device Philox Gumbel donor & target selector (zero host syncs, CUDAGraph-safe)."""
        step_after = tl.load(step_ptr).to(tl.int32)
        step_cur = step_after - 1
        active = step_cur <= stop_step

        u_sum = tl.zeros((), dtype=tl.float32)
        s_max = tl.zeros((), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            ok = (k0 + tl.arange(0, BLOCK_K)).to(tl.int64)
            mk = ok < K
            uv = tl.load(usage_ptr + ok, mask=mk, other=0.0).to(tl.float32)
            sv = tl.load(sse_ema_ptr + ok, mask=mk, other=0.0).to(tl.float32)
            u_sum += tl.sum(uv, axis=0)
            s_max = tl.maximum(s_max, tl.max(tl.where(mk, sv, 0.0), axis=0))

        thresh = (u_sum / K.to(tl.float32)) * unused_thresh
        ub = tl.maximum(tl.load(usage_bias_ptr).to(tl.float32), 1e-6)
        sb = tl.maximum(tl.load(sse_bias_ptr).to(tl.float32), 1e-6)
        inv_ub = 1.0 / ub
        inv_sb = 1.0 / sb
        rng_s = base_seed + step_cur * 7919

        n_unused = tl.zeros((), dtype=tl.int32)
        n_donors = tl.zeros((), dtype=tl.int32)
        for k0 in range(0, K, BLOCK_K):
            ok = (k0 + tl.arange(0, BLOCK_K)).to(tl.int64)
            mk = ok < K
            uv = tl.load(usage_ptr + ok, mask=mk, other=0.0).to(tl.float32)
            sv = tl.load(sse_ema_ptr + ok, mask=mk, other=0.0).to(tl.float32)
            is_un = mk & active & (uv < thresh)
            is_dn = mk & active & (~is_un) & (uv > 0.0)

            n_unused += tl.sum(tl.where(is_un, 1, 0), axis=0)
            n_donors += tl.sum(tl.where(is_dn, 1, 0), axis=0)

            u_unb = tl.maximum(uv * inv_ub, 1e-12)
            e_unb = tl.maximum(sv * inv_sb, 1e-12)
            score = tl.where(s_max > 0.0, e_unb * tl.exp(0.35 * tl.log(tl.maximum(u_unb / bsz_scale, 1e-12))), uv)
            ur = tl.minimum(tl.maximum(tl.rand(rng_s, ok.to(tl.int32)), 1e-30), 1.0 - 1e-7)
            gumb = -tl.log(-tl.log(ur))
            key = tl.where(is_dn, tl.log(tl.maximum(score, 1e-30)) + gumb, -float("inf"))
            tl.store(unused_i8_ptr + ok, tl.where(is_un, 1, 0).to(tl.int8), mask=mk)
            tl.store(donor_keys_ptr + ok, key, mask=mk)

        n_sp = tl.where(active, tl.minimum(n_unused, n_donors), 0)
        tl.store(n_sp_ptr, n_sp.to(tl.int64))

    @triton.jit(do_not_specialize=["D", "base_seed"])
    def _gear_syncfree_split_kernel(
        centroids_ptr,
        slow_centroids_ptr,
        usage_ptr,
        sse_ema_ptr,
        usage_bias_ptr,
        sse_bias_ptr,
        pass_sums_ptr,
        pass_counts_ptr,
        pass_sse_ptr,
        accum_pass_ptr,
        step_ptr,
        src_ptr,
        tgt_ptr,
        n_sp_ptr,
        total_splits_ptr,
        D,
        inv_d,
        split_mult,
        base_seed,
        IS_FP8: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Zero-sync RMS-scaled Rademacher/Half-Gaussian split + closing flush inside CUDAGraph."""
        pid = tl.program_id(0).to(tl.int64)
        n_sp = tl.load(n_sp_ptr).to(tl.int64)
        if pid == 0 and n_sp > 0:
            prev_tot = tl.load(total_splits_ptr).to(tl.int64)
            tl.store(total_splits_ptr, prev_tot + n_sp)
        if pid >= n_sp:
            return

        src = tl.load(src_ptr + pid).to(tl.int64)
        tgt = tl.load(tgt_ptr + pid).to(tl.int64)
        u_raw = tl.load(usage_ptr + src).to(tl.float32)
        s_raw = tl.minimum(tl.load(sse_ema_ptr + src).to(tl.float32), 1e35)
        ub = tl.maximum(tl.load(usage_bias_ptr).to(tl.float32), 1e-6)
        sb = tl.maximum(tl.load(sse_bias_ptr).to(tl.float32), 1e-6)
        sigma_rms = tl.sqrt(tl.maximum(((s_raw / sb) / tl.maximum(u_raw / ub, 1e-6)) * inv_d, 1e-12))

        step_cur = tl.load(step_ptr).to(tl.int32) - 1
        rseed = base_seed + step_cur * 131
        step_mag = split_mult * sigma_rms * tl.sqrt(inv_d)
        flush_p = tl.load(accum_pass_ptr).to(tl.int32) != 0

        all_eq = 1
        for d0 in range(0, D, BLOCK_D):
            cols = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
            md = cols < D
            dc = tl.load(centroids_ptr + src * D + cols, mask=md, other=0.0).to(tl.float32)
            ur = tl.rand(rseed, (pid * D + cols).to(tl.int32))
            rad = tl.where(ur < 0.5, -1.0, 1.0)
            delta = tl.where(md, step_mag * rad, 0.0)
            c_t = dc + delta
            c_s = dc - delta
            if IS_FP8:
                eq_c = c_t == c_s
            else:
                eq_c = c_t.to(tl.bfloat16).to(tl.float32) == c_s.to(tl.bfloat16).to(tl.float32)
            if tl.sum(tl.where(md & (~eq_c), 1, 0), axis=0) > 0:
                all_eq = 0
            tl.store(centroids_ptr + tgt * D + cols, c_t, mask=md)
            tl.store(centroids_ptr + src * D + cols, c_s, mask=md)
            tl.store(slow_centroids_ptr + tgt * D + cols, c_t, mask=md)
            tl.store(slow_centroids_ptr + src * D + cols, c_s, mask=md)
            if flush_p:
                tl.store(pass_sums_ptr + tgt * D + cols, 0.0, mask=md)
                tl.store(pass_sums_ptr + src * D + cols, 0.0, mask=md)

        if all_eq != 0:
            c0 = tl.load(centroids_ptr + tgt * D).to(tl.float32)
            ulp = tl.maximum(tl.abs(c0) * (1.0 / 16.0 if IS_FP8 else 1.0 / 64.0), 1e-6)
            nudge = tl.where(c0 > 0.0, -ulp, ulp)
            tl.store(centroids_ptr + tgt * D, c0 + nudge)
            tl.store(slow_centroids_ptr + tgt * D, c0 + nudge)

        tl.store(usage_ptr + tgt, 0.5 * u_raw)
        tl.store(usage_ptr + src, 0.5 * u_raw)
        tl.store(sse_ema_ptr + tgt, 0.42 * s_raw)
        tl.store(sse_ema_ptr + src, 0.42 * s_raw)
        if flush_p:
            tl.store(pass_counts_ptr + tgt, 0.0)
            tl.store(pass_counts_ptr + src, 0.0)
            tl.store(pass_sse_ptr + tgt, 0.0)
            tl.store(pass_sse_ptr + src, 0.0)


@torch.no_grad()
def encode_croiss_fp8_dataset(
    x_t: torch.Tensor,
    origin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode `(N, D)` dataset into 64B-aligned origin-centered `float8_e4m3fn` + `sx` + `x_norm`."""
    n, d = int(x_t.shape[0]), int(x_t.shape[1])
    device = x_t.device
    d_pad = ((max(32, d) + 63) // 64) * 64
    qx = torch.empty((n, d_pad), dtype=FLOAT8_E4M3FN, device=device)
    sx = torch.empty((n,), dtype=torch.float32, device=device)
    x_norm = torch.empty((n,), dtype=torch.float32, device=device)
    full_d = d_pad <= 256
    blk_n = 64 if full_d else 32
    blk_d = max(64, triton.next_power_of_2(d_pad)) if full_d else 256
    with torch.cuda.device(device):
        _croiss_encode_rows_kernel[(triton.cdiv(n, blk_n),)](
            x_t,
            origin,
            qx,
            sx,
            x_norm,
            n,
            d,
            x_t.stride(0),
            x_t.stride(1),
            qx.stride(0),
            ORIGIN_FP64=(origin.dtype == torch.float64),
            FULL_D=full_d,
            BLOCK_N=blk_n,
            BLOCK_D=blk_d,
            num_warps=4,
        )
    return qx, sx, x_norm


@torch.no_grad()
def encode_centered_bf16_dataset(
    x_t: torch.Tensor,
    origin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode `(N, D)` dataset into 32-elem-aligned origin-centered `bfloat16` + represented `x_norm`."""
    n, d = int(x_t.shape[0]), int(x_t.shape[1])
    device = x_t.device
    d_pad = ((max(16, d) + 31) // 32) * 32
    x_bf16 = torch.empty((n, d_pad), dtype=torch.bfloat16, device=device)
    x_norm = torch.empty((n,), dtype=torch.float32, device=device)
    blk_n = 64 if d <= 256 else 32
    blk_d = min(256, max(16, triton.next_power_of_2(max(1, d))))
    with torch.cuda.device(device):
        _center_bf16_rows_kernel[(triton.cdiv(n, blk_n),)](
            x_t,
            origin,
            x_bf16,
            x_norm,
            n,
            d,
            x_t.stride(0),
            x_t.stride(1),
            x_bf16.stride(0),
            ORIGIN_FP64=(origin.dtype == torch.float64),
            BLOCK_N=blk_n,
            BLOCK_D=blk_d,
            num_warps=4,
        )
    return x_bf16, x_norm


def _launch_gear_assign(
    x_enc: torch.Tensor,
    order_buf: torch.Tensor,
    batch_off_buf: torch.Tensor,
    sx: torch.Tensor,
    x_norm: torch.Tensor,
    c_rep: torch.Tensor,
    sc: torch.Tensor,
    c_norm: torch.Tensor,
    out_dist: torch.Tensor,
    out_label: torch.Tensor,
    b_cur: int,
    k: int,
    d: int,
    is_fp8: bool,
    num_sms: int,
    splitk_scratch: tuple[torch.Tensor, torch.Tensor, int] | None,
) -> None:
    """Unified BF16 / CroissAttn-FP8 assignment dispatcher (FULL_D, Split-K, and 3-Stage Wide-D)."""
    full_d = (d <= 256) if is_fp8 else (d <= 128)
    min_dot_d = 32 if is_fp8 else 16
    elem_bytes = 1 if is_fp8 else 2

    if splitk_scratch is not None and b_cur == splitk_scratch[0].shape[0]:
        part_sc, part_lb, parts = splitk_scratch
        blk_b_sk = 64
        blk_k_sk = 128
        blk_d_sk = max(min_dot_d, triton.next_power_of_2(max(1, d))) if full_d else (128 if is_fp8 else 64)
        st_sk = 3
        smem_sk = ((0 if full_d else blk_b_sk) + blk_k_sk) * blk_d_sk * elem_bytes
        while st_sk > 1 and smem_sk * st_sk > 81920:
            st_sk -= 1
        with torch.cuda.device(x_enc.device):
            _gear_splitk_assign_kernel[(triton.cdiv(b_cur, blk_b_sk), parts)](
                x_enc,
                order_buf,
                batch_off_buf,
                sx,
                c_rep,
                sc,
                c_norm,
                part_sc,
                part_lb,
                b_cur,
                k,
                d,
                x_enc.stride(0),
                c_rep.stride(0),
                PARTITIONS=parts,
                IS_FP8=is_fp8,
                FULL_D=full_d,
                BLOCK_B=blk_b_sk,
                BLOCK_K=blk_k_sk,
                BLOCK_D=blk_d_sk,
                num_warps=4 if full_d else 8,
                num_stages=st_sk,
            )
            _gear_splitk_reduce_kernel[(triton.cdiv(b_cur, 128),)](
                order_buf,
                batch_off_buf,
                x_norm,
                part_sc,
                part_lb,
                out_dist,
                out_label,
                b_cur,
                PARTITIONS=parts,
                BLOCK_B=128,
                BLOCK_P=max(16, triton.next_power_of_2(parts)),
                num_warps=4,
            )
        return

    if full_d:
        blk_d = max(min_dot_d, triton.next_power_of_2(max(1, d)))
        if b_cur >= 128 * num_sms and k >= 256 and blk_d <= 128:
            blk_b, blk_k, nw = 128, (256 if is_fp8 else 128), 8
        else:
            blk_b, blk_k, nw = 64, (128 if k >= 128 else 64), 4
    else:
        blk_d = 64 if is_fp8 else 32
        if b_cur >= 128 * num_sms and k >= 256:
            blk_b, blk_k = 128, 256
        else:
            blk_b, blk_k = 64, (256 if k >= 256 else 128)
        nw = 8

    n_st = 3
    smem_stage = ((0 if full_d else blk_b) + blk_k) * blk_d * elem_bytes
    while n_st > 1 and smem_stage * n_st > 81920:
        n_st -= 1

    with torch.cuda.device(x_enc.device):
        _gear_assign_kernel[(triton.cdiv(b_cur, blk_b),)](
            x_enc,
            order_buf,
            batch_off_buf,
            sx,
            x_norm,
            c_rep,
            sc,
            c_norm,
            out_dist,
            out_label,
            b_cur,
            k,
            d,
            x_enc.stride(0),
            c_rep.stride(0),
            IS_FP8=is_fp8,
            FULL_D=full_d,
            BLOCK_B=blk_b,
            BLOCK_K=blk_k,
            BLOCK_D=blk_d,
            num_warps=nw,
            num_stages=n_st,
        )


# ==============================================================================
# 4. Huge-K Adaptive Fast Seeding (`20x` Faster Init at `K = 65,536`)
# ==============================================================================


@torch.no_grad()
def adaptive_turkpp_init_centered(
    x_t: torch.Tensor,
    origin64: torch.Tensor,
    coord_scale: float,
    k: int,
    comp_dtype: torch.dtype,
    seed: int,
    huge_k_fast: bool = False,
) -> SeedingResult:
    """Scale-adaptive coreset + tournament seeding: delegates to `ultra_turkpp` for small/medium `K`,
    and uses parallel coarse-bucketed stratified importance + FP8/BF16 polish for Huge `K >= 16,384`."""
    if not huge_k_fast or k < 8192 or not x_t.is_cuda or not HAS_TRITON:
        cmult = 6 if k >= 16384 else 8
        return ultra_turkpp_init_centered(
            x_t=x_t,
            origin64=origin64,
            coord_scale=coord_scale,
            k=k,
            comp_dtype=comp_dtype,
            coreset_mult=cmult,
            min_coreset_size=4096,
            polish=True,
            seed=seed,
        )

    # Huge-K fast path (`K in {8192..65536}`): 1-pass importance coreset `m = min(n, 3*k)` +
    # 2-round spatial projection shard selection + 1-pass IVF-accelerated Voronoi polish!
    n, d = int(x_t.shape[0]), int(x_t.shape[1])
    device = x_t.device
    origin32 = origin64.float().contiguous()
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed) % (2**32))

    m = min(n, max(16384, 3 * k))
    d2_orig = torch.empty((n,), dtype=torch.float32, device=device)
    from emeans.ultra_emeans import _ultra_origin_d2_kernel

    blk_d = min(128, max(16, triton.next_power_of_2(max(1, d))))
    with torch.cuda.device(device):
        _ultra_origin_d2_kernel[(triton.cdiv(n, 128),)](
            x_t,
            origin32,
            d2_orig,
            n,
            d,
            x_t.stride(0),
            x_t.stride(1),
            float(coord_scale),
            ORIGIN_FP64=False,
            BLOCK_N=128,
            BLOCK_D=blk_d,
            num_warps=4,
        )
    tot_d2 = d2_orig.double().sum().clamp_min(1e-30)
    q = 0.5 * (1.0 / n) + 0.5 * (d2_orig.double() / tot_d2)
    cdf = torch.cumsum(q, dim=0)
    u_strat = (
        torch.arange(m, device=device, dtype=torch.float64)
        + torch.rand((m,), generator=gen, device=device, dtype=torch.float64)
    ) * (cdf[-1] / float(m))
    idx_sub = torch.searchsorted(cdf, u_strat, right=True).clamp_(0, n - 1)
    xs_c = ((x_t[idx_sub].float() - origin32[None, :]) * float(coord_scale)).contiguous()
    w_sub = (1.0 / (float(m) * q[idx_sub])).float()
    w_sub = (w_sub / w_sub.sum().clamp_min(1e-30)).contiguous()

    # Spatial random-projection stratified selection + rare-tail top-energy capture
    rdir = torch.randn((d, 2), generator=gen, device=device, dtype=torch.float32)
    proj = xs_c @ rdir[:, 0]
    order_p = torch.argsort(proj)
    xs_s = xs_c[order_p]
    ws_s = w_sub[order_p]
    d2_s = d2_orig[idx_sub[order_p]]
    # Pick 1 importance-Gumbel winner inside each of `k` contiguous spatial quantiles!
    g_cnt = k
    per_g = m // g_cnt
    if per_g >= 2:
        xs_blk = xs_s[: g_cnt * per_g].view(g_cnt, per_g, d)
        ws_blk = ws_s[: g_cnt * per_g].view(g_cnt, per_g)
        d2_blk = d2_s[: g_cnt * per_g].view(g_cnt, per_g)
        ug = torch.rand((g_cnt, per_g), generator=gen, device=device, dtype=torch.float32).clamp_(1e-30, 1.0 - 1e-7)
        score_g = (ws_blk * (0.25 + 0.75 * d2_blk / d2_blk.max(dim=1, keepdim=True).values.clamp_min(1e-30))).clamp_min(
            1e-30
        ).log() - (-ug.log()).log()
        win_loc = score_g.argmax(dim=1)
        raw_c = xs_blk[torch.arange(g_cnt, device=device), win_loc].contiguous()
    else:
        raw_c = xs_s[:k].contiguous()

    # Fast 1-step IVF-pruned Voronoi polish on `xs_c` (`nlist = round_pow2(sqrt(k))`, `nprobe = 6`)
    nlist_p = max(64, min(512, round_power_of_two(math.sqrt(k))))
    ivf_p = IVFAssigner.build(xs_c, k=k, nlist=nlist_p, compute_dtype=torch.bfloat16, seed=seed + 19)
    ivf_p.rebuild_index(raw_c)
    sq_d0, labels = ivf_p.assign(xs_c, nprobe=6)

    mass = torch.zeros((k,), dtype=torch.float32, device=device)
    mass.index_add_(0, labels, w_sub)
    sums = torch.zeros((k, d), dtype=torch.float32, device=device)
    sums.index_add_(0, labels, xs_c * w_sub[:, None])
    nonempty = mass > 0.0
    polished = torch.where(nonempty[:, None], sums / mass.clamp_min(1e-30)[:, None], raw_c).contiguous()

    pol_sq = (xs_c - polished[labels]).square().sum(dim=1)
    dist_mass = torch.zeros((k,), dtype=torch.float32, device=device)
    dist_mass.index_add_(0, labels, w_sub * pol_sq)
    tot_m = mass.sum().clamp_min(1e-30)
    return SeedingResult(
        centroids=polished,
        prior_probs=(mass / tot_m).contiguous(),
        prior_distortion=(dist_mass / tot_m).contiguous(),
    )


# ==============================================================================
# 5. Suite 1 & Suite 2 Unified Execution Engines
# ==============================================================================


@torch.no_grad()
def run_gear1_arm(
    arm: str,
    x_train: torch.Tensor,
    origin64: torch.Tensor,
    seed_res: SeedingResult,
    init_ms: float,
    k: int,
    nepochs: float,
    seed: int,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Run one Gear Boundary 1 arm (`bf16_eager`, `bf16_cudagraph`, `fp8_fastops_eager`,
    `fp8_croiss_eager`, `fp8_croiss_cudagraph`) from identical `seed_res`."""
    device = x_train.device
    n, d = int(x_train.shape[0]), int(x_train.shape[1])
    recipe = build_recipe(n=n, d=d, k=k, nepochs=nepochs)
    origin32 = origin64.float().contiguous()

    if arm == "fp8_fastops_eager":
        # Global power-of-two FP8 baseline from `emeans.fast_ops`
        c_init_uncentered = (seed_res.centroids + origin32[None, :]).contiguous()
        p_usage = (0.5 * float(recipe.batch_size) * seed_res.prior_probs.clamp_min(1e-3 / k)).float().contiguous()
        torch.cuda.synchronize(device)
        t_pe0 = time.perf_counter()
        quant = FP8Quantizer.calibrate(x_train, center_mode="auto", seed=seed)
        x_fp8 = quant.encode_rows(x_train)
        torch.cuda.synchronize(device)
        preencode_ms = (time.perf_counter() - t_pe0) * 1e3

        t_tr0 = time.perf_counter()
        centroids = c_init_uncentered.clone()
        usage = p_usage.clone()
        packed = torch.zeros((k, d + 1), dtype=torch.float32, device=device)
        c_rep = torch.empty((k, quant.padded_d), dtype=FLOAT8_E4M3FN, device=device)
        c_norm = torch.empty((k,), dtype=torch.float32, device=device)
        quant.prepare_centroids(centroids, c_fp8_out=c_rep, c_norm_out=c_norm)
        pass_counts = torch.zeros((k,), dtype=torch.float64, device=device)
        pass_sums = torch.zeros((k, d), dtype=torch.float32, device=device)
        split_gen = torch.Generator(device=device)
        split_gen.manual_seed((seed + 7919) % (2**32))
        sign = torch.ones((d,), dtype=torch.float32, device=device)
        sign[1::2] = -1.0
        split_off = recipe.split_epsilon * sign
        step = 0
        n_splits = 0
        num_ep = len(recipe.epoch_rows)
        for ep_idx, rows_ep in enumerate(recipe.epoch_rows):
            is_last = ep_idx == (num_ep - 1)
            gen_ep = torch.Generator(device=device)
            gen_ep.manual_seed(epoch_seed(seed, ep_idx))
            order = block_permute_indices(n, rows_ep, 256, gen_ep, device)
            for b_s in range(0, rows_ep, recipe.batch_size):
                b_e = min(b_s + recipe.batch_size, rows_ep)
                b_idx = order[b_s:b_e]
                _, lbl = fp8_assign(
                    x_train,
                    centroids,
                    quantizer=quant,
                    c_fp8_cached=(c_rep, c_norm),
                    x_fp8_cached=x_fp8,
                    row_indices=b_idx,
                )
                alpha = scheduled_alpha(recipe.alpha0, step, recipe.total_steps, recipe.alpha_min_factor)
                fused_scatter_stats(x_train, lbl, packed, row_indices=b_idx)
                fused_ema_and_prepare(
                    centroids=centroids,
                    usage=usage,
                    batch_sums=packed[:, :d],
                    counts=packed[:, d],
                    alpha=alpha,
                    compute_dtype=FLOAT8_E4M3FN,
                    c_rep_out=c_rep,
                    c_norm_out=c_norm,
                    pass_counts=pass_counts if is_last else None,
                    pass_sums=pass_sums if is_last else None,
                    fp8_quantizer=quant,
                    zero_stats=True,
                )
                if recipe.unused_threshold > 0.0 and ((step + 1) % 5 == 0) and step <= int(0.85 * recipe.total_steps):
                    thresh = usage.mean() * recipe.unused_threshold
                    un_m = usage < thresh
                    dn_m = (~un_m) & (usage > 0.0)
                    nu, nd = torch.stack((un_m.sum(), dn_m.sum())).tolist()
                    nsp = min(int(nu), int(nd))
                    if nsp > 0:
                        ur = torch.rand((k,), generator=split_gen, device=device).clamp_min_(1e-30)
                        keys = torch.where(dn_m, usage.log() - (-ur.log()).log(), float("-inf"))
                        src = torch.topk(keys, nsp).indices
                        tgt = torch.nonzero(un_m, as_tuple=False).squeeze(1)[:nsp]
                        dc = centroids[src] * (1.0 - split_off)
                        cc = centroids[src] * (1.0 + split_off)
                        hu = 0.5 * usage[src]
                        centroids[src] = dc
                        centroids[tgt] = cc
                        usage[src] = hu
                        usage[tgt] = hu
                        n_splits += nsp
                        quant.prepare_centroids(centroids, c_fp8_out=c_rep, c_norm_out=c_norm)
                step += 1
        nonempty = pass_counts > 0
        c_out = torch.where(nonempty[:, None], pass_sums / pass_counts.clamp_min(1.0).float()[:, None], centroids)
        torch.cuda.synchronize(device)
        train_ms = (time.perf_counter() - t_tr0) * 1e3
        fp8_nan_cnt = int(torch.isnan(x_fp8.float()).sum().item() + torch.isnan(c_rep.float()).sum().item())
        return c_out.contiguous(), {
            "init_ms": init_ms,
            "preencode_ms": preencode_ms,
            "graph_capture_ms": 0.0,
            "train_ms": train_ms,
            "total_ms": init_ms + preencode_ms + train_ms,
            "total_with_capture_ms": init_ms + preencode_ms + train_ms,
            "batch_size": int(recipe.batch_size),
            "total_steps": int(recipe.total_steps),
            "n_splits": int(n_splits),
            "dead_clusters": int((pass_counts <= 0.5).sum().item()),
            "fp8_nan_count": fp8_nan_cnt,
        }

    is_fp8 = arm.startswith("fp8_croiss")
    use_graph = arm.endswith("_cudagraph")
    origin_ker = (
        origin64
        if (x_train.dtype == torch.float64 or float(origin64.abs().max().item()) > 1e3)
        else origin32
    )

    # Step 1: Fused 1-pass dataset pre-encoding (`bf16` or decoupled `CroissAttn-FP8`)
    torch.cuda.synchronize(device)
    t_pe0 = time.perf_counter()
    if is_fp8:
        x_enc, sx, x_norm = encode_croiss_fp8_dataset(x_train, origin_ker)
    else:
        x_enc, x_norm = encode_centered_bf16_dataset(x_train, origin_ker)
        sx = x_norm  # unused dummy ptr when IS_FP8=False
    torch.cuda.synchronize(device)
    preencode_ms = (time.perf_counter() - t_pe0) * 1e3

    # Step 2: Allocate static buffers for zero-sync / 5-step CUDAGraph execution
    bsz = int(recipe.batch_size)
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    d_pad = int(x_enc.shape[1])
    c_rep = torch.empty((k, d_pad), dtype=FLOAT8_E4M3FN if is_fp8 else torch.bfloat16, device=device)
    sc = torch.ones((k,), dtype=torch.float32, device=device)
    c_norm = torch.empty((k,), dtype=torch.float32, device=device)

    centroids_c = seed_res.centroids.clone().contiguous()
    slow_centroids = centroids_c.clone().contiguous()
    p_hat = seed_res.prior_probs
    e_hat = seed_res.prior_distortion
    mean_ehat = e_hat.mean().clamp_min(1e-12)
    usage = (0.5 * float(bsz) * p_hat.clamp_min(1e-3 / float(k))).float().contiguous()
    sse_ema = (0.5 * float(bsz) * e_hat.clamp_min(1e-6 * mean_ehat)).float().contiguous()

    sums = torch.zeros((k, d), dtype=torch.float32, device=device)
    counts = torch.zeros((k,), dtype=torch.float32, device=device)
    batch_sse = torch.zeros((k,), dtype=torch.float32, device=device)
    pass_sums = torch.zeros((k, d), dtype=torch.float32, device=device)
    pass_counts = torch.zeros((k,), dtype=torch.float64, device=device)
    pass_sse = torch.zeros((k,), dtype=torch.float64, device=device)

    order_buf = torch.zeros((n,), dtype=torch.int64, device=device)
    dist_buf = torch.empty((bsz,), dtype=torch.float32, device=device)
    lbl_buf = torch.empty((bsz,), dtype=torch.int64, device=device)

    step_buf = torch.zeros((), dtype=torch.int32, device=device)
    batch_off_buf = torch.zeros((), dtype=torch.int64, device=device)
    accum_pass_buf = torch.zeros((), dtype=torch.int32, device=device)
    usage_bias_buf = torch.full((), 0.5, dtype=torch.float32, device=device)
    sse_bias_buf = torch.full((), 0.5, dtype=torch.float32, device=device)

    unused_i8 = torch.zeros((k,), dtype=torch.int8, device=device)
    donor_keys = torch.zeros((k,), dtype=torch.float32, device=device)
    n_sp_buf = torch.zeros((), dtype=torch.int64, device=device)
    total_splits_buf = torch.zeros((), dtype=torch.int64, device=device)

    parts = _choose_split_k_partitions(bsz, k, num_sms=num_sms, block_b=64, block_k=128) if k >= 2048 else 1
    splitk_scratch: tuple[torch.Tensor, torch.Tensor, int] | None = None
    if parts > 1:
        splitk_scratch = (
            torch.empty((bsz, parts), dtype=torch.float32, device=device),
            torch.empty((bsz, parts), dtype=torch.int32, device=device),
            parts,
        )

    blk_d_upd = min(256, max(32 if is_fp8 else 16, triton.next_power_of_2(max(1, d))))
    blk_b_sc = 128 if d <= 256 else 64
    blk_d_sc = min(128, max(16, triton.next_power_of_2(max(1, d))))
    blk_k_dn = min(1024, max(64, triton.next_power_of_2(min(1024, k))))
    alpha_slow = min(1.0, max(1e-6, 0.25 * recipe.alpha_min_factor * recipe.alpha0))
    stop_step = int(0.85 * recipe.total_steps)
    base_rng_seed = int((seed * 10007 + 19) % (2**31 - 1))

    def _emit_one_step(b_cur: int, with_revival: bool) -> None:
        _launch_gear_assign(
            x_enc=x_enc,
            order_buf=order_buf,
            batch_off_buf=batch_off_buf,
            sx=sx,
            x_norm=x_norm,
            c_rep=c_rep,
            sc=sc,
            c_norm=c_norm,
            out_dist=dist_buf[:b_cur],
            out_label=lbl_buf[:b_cur],
            b_cur=b_cur,
            k=k,
            d=d,
            is_fp8=is_fp8,
            num_sms=num_sms,
            splitk_scratch=splitk_scratch if b_cur == bsz else None,
        )
        with torch.cuda.device(device):
            _gear_scatter_centered_kernel[(triton.cdiv(b_cur, blk_b_sc), triton.cdiv(d, blk_d_sc))](
                x_train,
                order_buf,
                batch_off_buf,
                origin_ker,
                lbl_buf[:b_cur],
                dist_buf[:b_cur],
                sums,
                counts,
                batch_sse,
                b_cur,
                d,
                x_train.stride(0),
                x_train.stride(1),
                ORIGIN_FP64=(origin_ker.dtype == torch.float64),
                BLOCK_B=blk_b_sc,
                BLOCK_D=blk_d_sc,
                num_warps=4,
            )
            _gear_ondevice_ema_prep_kernel[(k,)](
                sums,
                counts,
                batch_sse,
                centroids_c,
                usage,
                sse_ema,
                slow_centroids,
                pass_sums,
                pass_counts,
                pass_sse,
                c_rep,
                sc,
                c_norm,
                step_buf,
                accum_pass_buf,
                k,
                d,
                c_rep.stride(0),
                int(recipe.total_steps),
                float(recipe.alpha0 * 0.85),
                float(recipe.alpha_min_factor),
                float(alpha_slow),
                0 if with_revival else 1,
                IS_FP8=is_fp8,
                BLOCK_D=blk_d_upd,
                num_warps=2 if d <= 128 else 4,
            )
            _gear_advance_scalars_kernel[(1,)](
                step_buf,
                batch_off_buf,
                usage_bias_buf,
                sse_bias_buf,
                b_cur,
                int(recipe.total_steps),
                float(recipe.alpha0 * 0.85),
                float(recipe.alpha_min_factor),
                num_warps=1,
            )
        if with_revival and recipe.unused_threshold > 0.0 and k >= 2:
            with torch.cuda.device(device):
                _gear_donor_keys_kernel[(1,)](
                    usage,
                    sse_ema,
                    usage_bias_buf,
                    sse_bias_buf,
                    step_buf,
                    unused_i8,
                    donor_keys,
                    n_sp_buf,
                    k,
                    float(recipe.unused_threshold),
                    stop_step,
                    base_rng_seed,
                    float(bsz),
                    BLOCK_K=blk_k_dn,
                    num_warps=4,
                )
            src_sorted = torch.argsort(donor_keys, descending=True)
            tgt_sorted = torch.argsort(unused_i8, descending=True, stable=True)
            with torch.cuda.device(device):
                _gear_syncfree_split_kernel[(k,)](
                    centroids_c,
                    slow_centroids,
                    usage,
                    sse_ema,
                    usage_bias_buf,
                    sse_bias_buf,
                    pass_sums,
                    pass_counts,
                    pass_sse,
                    accum_pass_buf,
                    step_buf,
                    src_sorted,
                    tgt_sorted,
                    n_sp_buf,
                    total_splits_buf,
                    d,
                    float(1.0 / max(1, d)),
                    float(HALF_GAUSSIAN_MEAN),
                    base_rng_seed,
                    IS_FP8=is_fp8,
                    BLOCK_D=blk_d_upd,
                    num_warps=2 if d <= 128 else 4,
                )
                _gear_prep_centroids_kernel[(k,)](
                    centroids_c,
                    c_rep,
                    sc,
                    c_norm,
                    k,
                    d,
                    c_rep.stride(0),
                    IS_FP8=is_fp8,
                    BLOCK_D=blk_d_upd,
                    num_warps=2 if d <= 128 else 4,
                )

    def _reset_training_state() -> None:
        centroids_c.copy_(seed_res.centroids)
        slow_centroids.copy_(seed_res.centroids)
        usage.copy_((0.5 * float(bsz) * p_hat.clamp_min(1e-3 / float(k))).float())
        sse_ema.copy_((0.5 * float(bsz) * e_hat.clamp_min(1e-6 * mean_ehat)).float())
        sums.zero_()
        counts.zero_()
        batch_sse.zero_()
        pass_sums.zero_()
        pass_counts.zero_()
        pass_sse.zero_()
        step_buf.zero_()
        batch_off_buf.zero_()
        accum_pass_buf.zero_()
        usage_bias_buf.fill_(0.5)
        sse_bias_buf.fill_(0.5)
        total_splits_buf.zero_()
        with torch.cuda.device(device):
            _gear_prep_centroids_kernel[(k,)](
                centroids_c,
                c_rep,
                sc,
                c_norm,
                k,
                d,
                c_rep.stride(0),
                IS_FP8=is_fp8,
                BLOCK_D=blk_d_upd,
                num_warps=2 if d <= 128 else 4,
            )

    graph_5step: torch.cuda.CUDAGraph | None = None
    graph_capture_ms = 0.0
    if use_graph:
        # Warm cub::DeviceRadixSort workspace on stream + capture 5-step super-step graph
        _reset_training_state()
        s_cap = torch.cuda.Stream(device=device)
        s_cap.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(s_cap):
            for sub_i in range(5):
                _emit_one_step(bsz, with_revival=(sub_i == 4))
        torch.cuda.current_stream(device).wait_stream(s_cap)
        _reset_training_state()

        torch.cuda.synchronize(device)
        t_cap0 = time.perf_counter()
        graph_5step = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph_5step, stream=s_cap):
            for sub_i in range(5):
                _emit_one_step(bsz, with_revival=(sub_i == 4))
        torch.cuda.synchronize(device)
        graph_capture_ms = (time.perf_counter() - t_cap0) * 1e3

    _reset_training_state()
    torch.cuda.synchronize(device)
    t_tr0 = time.perf_counter()

    cur_step = 0
    num_epochs = len(recipe.epoch_rows)
    for ep_idx, rows_in_ep in enumerate(recipe.epoch_rows):
        is_last_epoch = ep_idx == (num_epochs - 1)
        gen_ep = torch.Generator(device=device)
        gen_ep.manual_seed(epoch_seed(seed, ep_idx))
        ep_perm = block_permute_indices(n, rows_in_ep, 256, gen_ep, device)
        order_buf[:rows_in_ep].copy_(ep_perm)
        batch_off_buf.zero_()
        accum_pass_buf.fill_(1 if is_last_epoch else 0)

        b_start = 0
        if use_graph and graph_5step is not None:
            # Align `cur_step % 5 == 0` at epoch boundary so sub-step 4 always matches `(step+1)%5 == 0`!
            while (cur_step % 5) != 0 and (b_start + bsz <= rows_in_ep):
                _emit_one_step(bsz, with_revival=((cur_step + 1) % 5 == 0))
                b_start += bsz
                cur_step += 1
            while b_start + 5 * bsz <= rows_in_ep:
                graph_5step.replay()
                b_start += 5 * bsz
                cur_step += 5

        while b_start < rows_in_ep:
            b_cur = min(bsz, rows_in_ep - b_start)
            _emit_one_step(b_cur, with_revival=((cur_step + 1) % 5 == 0))
            b_start += b_cur
            cur_step += 1

    u_bias_end = float(usage_bias_buf.item())
    a_end = scheduled_alpha(recipe.alpha0 * 0.85, max(0, cur_step - 1), recipe.total_steps, recipe.alpha_min_factor)
    c_fin_c, _ = ultra_simplex_eb_finalize(
        pass_sums=pass_sums,
        pass_counts=pass_counts,
        pass_counts_sq=None,
        pass_sse=pass_sse,
        live_centroids=centroids_c,
        slow_centroids=slow_centroids,
        usage=usage,
        usage_inv_bias=1.0 / max(u_bias_end, 1e-6),
        alpha_end=a_end,
        nepochs=nepochs,
        d=d,
        mode="simplex_eb",
        eb_tau=0.25,
        kappa_max=6.0,
    )
    out_dtype = torch.float64 if x_train.dtype == torch.float64 else torch.float32
    c_out = (c_fin_c.double() + origin64[None, :]).to(out_dtype).contiguous()
    torch.cuda.synchronize(device)
    train_ms = (time.perf_counter() - t_tr0) * 1e3

    fp8_nan_cnt = 0
    if is_fp8:
        # E4M3FN NaN byte is 0x7F (positive) or 0xFF (negative): (byte & 0x7F) == 0x7F
        qx_u8 = x_enc.view(torch.uint8)
        qc_u8 = c_rep.view(torch.uint8)
        fp8_nan_cnt = int((((qx_u8 & 0x7F) == 0x7F).sum() + ((qc_u8 & 0x7F) == 0x7F).sum()).item())

    return c_out, {
        "init_ms": init_ms,
        "preencode_ms": preencode_ms,
        "graph_capture_ms": graph_capture_ms,
        "train_ms": train_ms,
        "total_ms": init_ms + preencode_ms + train_ms,
        "total_with_capture_ms": init_ms + preencode_ms + train_ms + graph_capture_ms,
        "batch_size": bsz,
        "total_steps": int(recipe.total_steps),
        "n_splits": int(total_splits_buf.item()),
        "dead_clusters": int((pass_counts <= 0.5).sum().item()),
        "fp8_nan_count": fp8_nan_cnt,
    }


@torch.no_grad()
def run_gear2_ivf_arm(
    arm: str,
    x_train: torch.Tensor,
    origin64: torch.Tensor,
    seed_res: SeedingResult,
    init_ms: float,
    k: int,
    nepochs: float,
    seed: int,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Run one Gear Boundary 2 arm (`exact_bf16`, `exact_fp8`, `ivf_pure_np8`,
    `ivf2sq_exact_np{4,8,16}`, `ivf1sq_exact_np8`) across `K in {4096..65536}`."""
    if arm == "exact_bf16":
        c_out, m = run_gear1_arm("bf16_cudagraph", x_train, origin64, seed_res, init_ms, k, nepochs, seed)
        m["ivf_build_ms"] = 0.0
        m["nlist"] = 0
        m["nprobe"] = 0
        return c_out, m
    if arm == "exact_fp8":
        c_out, m = run_gear1_arm("fp8_croiss_cudagraph", x_train, origin64, seed_res, init_ms, k, nepochs, seed)
        m["ivf_build_ms"] = 0.0
        m["nlist"] = 0
        m["nprobe"] = 0
        return c_out, m

    # Parse IVF routing config:
    # - `ivf_pure_np8`: IVF for ALL epochs (no exact closing pass)
    # - `ivf2sq_exact_np{4,8,16}`: K_ivf = round_pow2(2*sqrt(K)), IVF for first T-1 passes + 1 Exact FP8 Closing Pass
    # - `ivf1sq_exact_np8`:        K_ivf = round_pow2(sqrt(K)),   IVF for first T-1 passes + 1 Exact FP8 Closing Pass
    device = x_train.device
    n, d = int(x_train.shape[0]), int(x_train.shape[1])
    recipe = build_recipe(n=n, d=d, k=k, nepochs=nepochs)
    origin32 = origin64.float().contiguous()

    if arm == "ivf_pure_np8":
        mult_sq, nprobe, exact_close = 2.0, 8, False
    elif arm.startswith("ivf2sq_exact_np"):
        mult_sq = 2.0
        nprobe = int(arm.split("np", 1)[1])
        exact_close = True
    elif arm.startswith("ivf1sq_exact_np"):
        mult_sq = 1.0
        nprobe = int(arm.split("np", 1)[1])
        exact_close = True
    else:
        raise ValueError(f"Unknown Gear 2 IVF arm={arm!r}")

    nlist = max(32, min(1024, round_power_of_two(mult_sq * math.sqrt(max(16, k)))))
    nlist = min(nlist, k)

    origin_ker = (
        origin64
        if (x_train.dtype == torch.float64 or float(origin64.abs().max().item()) > 1e3)
        else origin32
    )
    # Pre-encode centered FP8 dataset ONLY if `exact_close` is enabled (1-pass closing scan!)
    torch.cuda.synchronize(device)
    t_pe0 = time.perf_counter()
    if exact_close:
        qx, sx, x_norm = encode_croiss_fp8_dataset(x_train, origin_ker)
    else:
        qx = sx = x_norm = None
    torch.cuda.synchronize(device)
    preencode_ms = (time.perf_counter() - t_pe0) * 1e3

    # Build coarse meta-centroids on centered coreset
    t_ivf0 = time.perf_counter()
    m_meta = min(n, max(8192, 48 * nlist))
    gen_m = torch.Generator(device=device)
    gen_m.manual_seed((seed + 31337) % (2**32))
    sub_idx = torch.randperm(n, generator=gen_m, device=device)[:m_meta]
    x_meta_c = (
        (x_train[sub_idx].double() - origin64[None, :]).float().contiguous()
        if origin_ker.dtype == torch.float64
        else (x_train[sub_idx].float() - origin32[None, :]).contiguous()
    )
    ivf = IVFAssigner.build(x_meta_c, k=k, nlist=nlist, compute_dtype=torch.bfloat16, seed=seed)
    torch.cuda.synchronize(device)
    ivf_build_ms = (time.perf_counter() - t_ivf0) * 1e3

    t_tr0 = time.perf_counter()
    bsz = int(recipe.batch_size)
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    centroids_c = seed_res.centroids.clone().contiguous()
    slow_centroids = centroids_c.clone().contiguous()
    p_hat = seed_res.prior_probs
    e_hat = seed_res.prior_distortion
    mean_ehat = e_hat.mean().clamp_min(1e-12)
    usage = (0.5 * float(bsz) * p_hat.clamp_min(1e-3 / float(k))).float().contiguous()
    sse_ema = (0.5 * float(bsz) * e_hat.clamp_min(1e-6 * mean_ehat)).float().contiguous()

    sums = torch.zeros((k, d), dtype=torch.float32, device=device)
    counts = torch.zeros((k,), dtype=torch.float32, device=device)
    batch_sse = torch.zeros((k,), dtype=torch.float32, device=device)
    pass_sums = torch.zeros((k, d), dtype=torch.float32, device=device)
    pass_counts = torch.zeros((k,), dtype=torch.float64, device=device)
    pass_sse = torch.zeros((k,), dtype=torch.float64, device=device)

    order_buf = torch.zeros((n,), dtype=torch.int64, device=device)
    step_buf = torch.zeros((), dtype=torch.int32, device=device)
    batch_off_buf = torch.zeros((), dtype=torch.int64, device=device)
    accum_pass_buf = torch.zeros((), dtype=torch.int32, device=device)
    usage_bias_buf = torch.full((), 0.5, dtype=torch.float32, device=device)
    sse_bias_buf = torch.full((), 0.5, dtype=torch.float32, device=device)
    unused_i8 = torch.zeros((k,), dtype=torch.int8, device=device)
    donor_keys = torch.zeros((k,), dtype=torch.float32, device=device)
    n_sp_buf = torch.zeros((), dtype=torch.int64, device=device)
    total_splits_buf = torch.zeros((), dtype=torch.int64, device=device)

    d_pad = int(qx.shape[1]) if qx is not None else ((max(32, d) + 63) // 64) * 64
    c_rep_fp8 = torch.empty((k, d_pad), dtype=FLOAT8_E4M3FN, device=device)
    sc_fp8 = torch.ones((k,), dtype=torch.float32, device=device)
    c_norm_fp8 = torch.empty((k,), dtype=torch.float32, device=device)
    dist_buf = torch.empty((bsz,), dtype=torch.float32, device=device)
    lbl_buf = torch.empty((bsz,), dtype=torch.int64, device=device)
    parts = _choose_split_k_partitions(bsz, k, num_sms=num_sms, block_b=64, block_k=128) if k >= 2048 else 1
    splitk_scratch = (
        (
            torch.empty((bsz, parts), dtype=torch.float32, device=device),
            torch.empty((bsz, parts), dtype=torch.int32, device=device),
            parts,
        )
        if parts > 1
        else None
    )

    blk_d_upd = min(256, max(32, triton.next_power_of_2(max(1, d))))
    blk_b_sc = 128 if d <= 256 else 64
    blk_d_sc = min(128, max(16, triton.next_power_of_2(max(1, d))))
    blk_k_dn = min(1024, max(64, triton.next_power_of_2(min(1024, k))))
    alpha_slow = min(1.0, max(1e-6, 0.25 * recipe.alpha_min_factor * recipe.alpha0))
    stop_step = int(0.85 * recipe.total_steps)
    base_rng_seed = int((seed * 10007 + 19) % (2**31 - 1))

    ivf.rebuild_index(centroids_c)
    rebuild_period = 6
    cur_step = 0
    num_epochs = len(recipe.epoch_rows)

    for ep_idx, rows_in_ep in enumerate(recipe.epoch_rows):
        is_last_epoch = ep_idx == (num_epochs - 1)
        use_exact_now = bool(exact_close and is_last_epoch)
        gen_ep = torch.Generator(device=device)
        gen_ep.manual_seed(epoch_seed(seed, ep_idx))
        ep_perm = block_permute_indices(n, rows_in_ep, 256, gen_ep, device)
        order_buf[:rows_in_ep].copy_(ep_perm)
        batch_off_buf.zero_()
        accum_pass_buf.fill_(1 if is_last_epoch else 0)

        if use_exact_now:
            with torch.cuda.device(device):
                _gear_prep_centroids_kernel[(k,)](
                    centroids_c,
                    c_rep_fp8,
                    sc_fp8,
                    c_norm_fp8,
                    k,
                    d,
                    c_rep_fp8.stride(0),
                    IS_FP8=True,
                    BLOCK_D=blk_d_upd,
                    num_warps=2 if d <= 128 else 4,
                )

        b_start = 0
        while b_start < rows_in_ep:
            b_cur = min(bsz, rows_in_ep - b_start)
            with_rev = ((cur_step + 1) % 5 == 0) and (cur_step <= stop_step)

            if use_exact_now and qx is not None and sx is not None and x_norm is not None:
                _launch_gear_assign(
                    x_enc=qx,
                    order_buf=order_buf,
                    batch_off_buf=batch_off_buf,
                    sx=sx,
                    x_norm=x_norm,
                    c_rep=c_rep_fp8,
                    sc=sc_fp8,
                    c_norm=c_norm_fp8,
                    out_dist=dist_buf[:b_cur],
                    out_label=lbl_buf[:b_cur],
                    b_cur=b_cur,
                    k=k,
                    d=d,
                    is_fp8=True,
                    num_sms=num_sms,
                    splitk_scratch=splitk_scratch if b_cur == bsz else None,
                )
                d_v = dist_buf[:b_cur]
                l_v = lbl_buf[:b_cur]
            else:
                if cur_step > 0 and (cur_step % rebuild_period == 0):
                    ivf.rebuild_index(centroids_c)
                b_idx = ep_perm[b_start : b_start + b_cur]
                xb_c = (
                    (x_train[b_idx].double() - origin64[None, :]).float().contiguous()
                    if origin_ker.dtype == torch.float64
                    else (x_train[b_idx].float() - origin32[None, :]).contiguous()
                )
                d_v, l_v = ivf.assign(xb_c, nprobe=nprobe)

            with torch.cuda.device(device):
                _gear_scatter_centered_kernel[(triton.cdiv(b_cur, blk_b_sc), triton.cdiv(d, blk_d_sc))](
                    x_train,
                    order_buf,
                    batch_off_buf,
                    origin_ker,
                    l_v,
                    d_v,
                    sums,
                    counts,
                    batch_sse,
                    b_cur,
                    d,
                    x_train.stride(0),
                    x_train.stride(1),
                    ORIGIN_FP64=(origin_ker.dtype == torch.float64),
                    BLOCK_B=blk_b_sc,
                    BLOCK_D=blk_d_sc,
                    num_warps=4,
                )
                _gear_ondevice_ema_prep_kernel[(k,)](
                    sums,
                    counts,
                    batch_sse,
                    centroids_c,
                    usage,
                    sse_ema,
                    slow_centroids,
                    pass_sums,
                    pass_counts,
                    pass_sse,
                    c_rep_fp8,
                    sc_fp8,
                    c_norm_fp8,
                    step_buf,
                    accum_pass_buf,
                    k,
                    d,
                    c_rep_fp8.stride(0),
                    int(recipe.total_steps),
                    float(recipe.alpha0 * 0.85),
                    float(recipe.alpha_min_factor),
                    float(alpha_slow),
                    1 if (use_exact_now and not with_rev) else 0,
                    IS_FP8=True,
                    BLOCK_D=blk_d_upd,
                    num_warps=2 if d <= 128 else 4,
                )
                _gear_advance_scalars_kernel[(1,)](
                    step_buf,
                    batch_off_buf,
                    usage_bias_buf,
                    sse_bias_buf,
                    b_cur,
                    int(recipe.total_steps),
                    float(recipe.alpha0 * 0.85),
                    float(recipe.alpha_min_factor),
                    num_warps=1,
                )

            if with_rev:
                with torch.cuda.device(device):
                    _gear_donor_keys_kernel[(1,)](
                        usage,
                        sse_ema,
                        usage_bias_buf,
                        sse_bias_buf,
                        step_buf,
                        unused_i8,
                        donor_keys,
                        n_sp_buf,
                        k,
                        float(recipe.unused_threshold),
                        stop_step,
                        base_rng_seed,
                        float(bsz),
                        BLOCK_K=blk_k_dn,
                        num_warps=4,
                    )
                src_sorted = torch.argsort(donor_keys, descending=True)
                tgt_sorted = torch.argsort(unused_i8, descending=True, stable=True)
                with torch.cuda.device(device):
                    _gear_syncfree_split_kernel[(k,)](
                        centroids_c,
                        slow_centroids,
                        usage,
                        sse_ema,
                        usage_bias_buf,
                        sse_bias_buf,
                        pass_sums,
                        pass_counts,
                        pass_sse,
                        accum_pass_buf,
                        step_buf,
                        src_sorted,
                        tgt_sorted,
                        n_sp_buf,
                        total_splits_buf,
                        d,
                        float(1.0 / max(1, d)),
                        float(HALF_GAUSSIAN_MEAN),
                        base_rng_seed,
                        IS_FP8=True,
                        BLOCK_D=blk_d_upd,
                        num_warps=2 if d <= 128 else 4,
                    )
                if use_exact_now:
                    with torch.cuda.device(device):
                        _gear_prep_centroids_kernel[(k,)](
                            centroids_c,
                            c_rep_fp8,
                            sc_fp8,
                            c_norm_fp8,
                            k,
                            d,
                            c_rep_fp8.stride(0),
                            IS_FP8=True,
                            BLOCK_D=blk_d_upd,
                            num_warps=2 if d <= 128 else 4,
                        )
                else:
                    ivf.rebuild_index(centroids_c)

            b_start += b_cur
            cur_step += 1

    u_bias_end = float(usage_bias_buf.item())
    a_end = scheduled_alpha(recipe.alpha0 * 0.85, max(0, cur_step - 1), recipe.total_steps, recipe.alpha_min_factor)
    c_fin_c, _ = ultra_simplex_eb_finalize(
        pass_sums=pass_sums,
        pass_counts=pass_counts,
        pass_counts_sq=None,
        pass_sse=pass_sse,
        live_centroids=centroids_c,
        slow_centroids=slow_centroids,
        usage=usage,
        usage_inv_bias=1.0 / max(u_bias_end, 1e-6),
        alpha_end=a_end,
        nepochs=nepochs,
        d=d,
        mode="simplex_eb",
        eb_tau=0.25,
        kappa_max=6.0,
    )
    out_dtype = torch.float64 if x_train.dtype == torch.float64 else torch.float32
    c_out = (c_fin_c.double() + origin64[None, :]).to(out_dtype).contiguous()
    torch.cuda.synchronize(device)
    train_ms = (time.perf_counter() - t_tr0) * 1e3
    tot_ms = init_ms + preencode_ms + ivf_build_ms + train_ms
    return c_out, {
        "init_ms": init_ms,
        "preencode_ms": preencode_ms,
        "ivf_build_ms": ivf_build_ms,
        "graph_capture_ms": 0.0,
        "train_ms": train_ms,
        "total_ms": tot_ms,
        "total_with_capture_ms": tot_ms,
        "batch_size": bsz,
        "total_steps": int(recipe.total_steps),
        "n_splits": int(total_splits_buf.item()),
        "dead_clusters": int((pass_counts <= 0.5).sum().item()),
        "fp8_nan_count": 0,
        "nlist": nlist,
        "nprobe": nprobe,
    }


# ==============================================================================
# 6. Suite 3 (`UltraEMeans` Scale-Adaptive Algorithmic Knobs)
# ==============================================================================


def instantiate_gear3_arm(sub_suite: str, arm: str, d: int, k: int, nepochs: float, seed: int) -> UltraEMeans:
    """Build one `UltraEMeans` arm for Sub-Sweeps 3A, 3B, 3C, or 3D."""
    common = dict(d=d, k=k, nepochs=nepochs, seed=seed)
    if sub_suite == "3A_wide_batch":
        if arm == "wide_1x":
            return UltraEMeans(**common, wide_step_mult=1, head_edit=False)
        if arm == "wide_2x_f25":
            return UltraEMeans(**common, wide_step_mult=2, wide_from_frac=0.25, head_edit=False)
        if arm == "wide_2x_f45":
            return UltraEMeans(**common, wide_step_mult=2, wide_from_frac=0.45, head_edit=False)
        if arm == "wide_4x_f25":
            return UltraEMeans(**common, wide_step_mult=4, wide_from_frac=0.25, head_edit=False)
    elif sub_suite == "3B_spectral_split":
        if arm == "spec_hg_018":
            return UltraEMeans(**common, split_mode="half_gaussian_oja", split_mult=0.18, head_edit=False)
        if arm == "spec_hg_050":
            return UltraEMeans(**common, split_mode="half_gaussian_oja", split_mult=0.50, head_edit=False)
        if arm == "spec_hg_080":
            return UltraEMeans(**common, split_mode="half_gaussian_oja", split_mult=HALF_GAUSSIAN_MEAN, head_edit=False)
        if arm == "spec_hg_110":
            return UltraEMeans(**common, split_mode="half_gaussian_oja", split_mult=1.10, head_edit=False)
        if arm == "ungated_oja_080":
            return UltraEMeans(**common, split_mode="oja_only", split_mult=HALF_GAUSSIAN_MEAN, head_edit=False)
        if arm == "radem_aniso_055":
            return UltraEMeans(**common, split_mode="rms_aniso", split_mult=0.55, head_edit=False)
    elif sub_suite == "3C_head_edit":
        if arm == "he_off":
            return UltraEMeans(**common, head_edit=False)
        if arm == "he_1cp":
            return UltraEMeans(**common, head_edit=True, transplant_checkpoints=(0.52,), transplant_max_pairs=5)
        if arm == "he_2cp":
            return UltraEMeans(**common, head_edit=True, transplant_checkpoints=(0.46, 0.76), transplant_max_pairs=5)
        if arm == "he_2cp_10p":
            return UltraEMeans(**common, head_edit=True, transplant_checkpoints=(0.46, 0.76), transplant_max_pairs=10)
    elif sub_suite == "3D_simplex_eb":
        if arm == "closing_mean":
            return UltraEMeans(**common, finalization="closing_mean", head_edit=False)
        if arm == "closing_ramp_035":
            return UltraEMeans(
                **common,
                finalization="closing_mean",
                use_pass_weighting=True,
                adaptive_pass_weighting=False,
                pass_w_span=0.35,
                head_edit=False,
            )
        if arm == "eb_tau_010":
            return UltraEMeans(**common, finalization="simplex_eb", eb_tau=0.10, kappa_max=4.0, head_edit=False)
        if arm == "eb_tau_025":
            return UltraEMeans(**common, finalization="simplex_eb", eb_tau=0.25, kappa_max=6.0, head_edit=False)
        if arm == "eb_tau_060":
            return UltraEMeans(**common, finalization="simplex_eb", eb_tau=0.60, kappa_max=8.0, head_edit=False)
        if arm == "eb_tau_125":
            return UltraEMeans(**common, finalization="simplex_eb", eb_tau=1.25, kappa_max=10.0, head_edit=False)
        if arm == "eb_tau_250":
            return UltraEMeans(**common, finalization="simplex_eb", eb_tau=2.50, kappa_max=14.0, head_edit=False)
    raise ValueError(f"Unknown Suite 3 (sub_suite={sub_suite!r}, arm={arm!r})")


SUITE_ARMS: dict[str, tuple[str, ...]] = {
    "gear1_precision_graph": (
        "bf16_eager",
        "bf16_cudagraph",
        "fp8_fastops_eager",
        "fp8_croiss_eager",
        "fp8_croiss_cudagraph",
    ),
    "gear2_ivf_routing": (
        "exact_bf16",
        "exact_fp8",
        "ivf_pure_np8",
        "ivf2sq_exact_np4",
        "ivf2sq_exact_np8",
        "ivf2sq_exact_np16",
        "ivf1sq_exact_np8",
    ),
    "3A_wide_batch": ("wide_1x", "wide_2x_f25", "wide_2x_f45", "wide_4x_f25"),
    "3B_spectral_split": (
        "spec_hg_018",
        "spec_hg_050",
        "spec_hg_080",
        "spec_hg_110",
        "ungated_oja_080",
        "radem_aniso_055",
    ),
    "3C_head_edit": ("he_off", "he_1cp", "he_2cp", "he_2cp_10p"),
    "3D_simplex_eb": (
        "closing_mean",
        "closing_ramp_035",
        "eb_tau_010",
        "eb_tau_025",
        "eb_tau_060",
        "eb_tau_125",
        "eb_tau_250",
    ),
}


# ==============================================================================
# 7. Experimental Cell Grid Across 4 Orders of Magnitude & 6-GPU Sharding
# ==============================================================================


@dataclass(frozen=True)
class ScaleLawCell:
    """One experimental cell pinned to a single physical GPU shard `(0..5)`."""

    cell_id: str
    wave: int           # 1 (rapid live crossover sweep) or 2 (extreme 5M / 65K / high-D refinement)
    shard_idx: int      # 0..5 -> ripper:0..3, loki:0..1
    suite: str          # key in SUITE_ARMS
    problem_kind: str   # synthetic kind or *_val real corpus
    is_real: bool
    n: int
    k: int
    d: int
    nepochs: float
    num_seeds: int


def build_scale_cells(smoke: bool = False) -> list[ScaleLawCell]:
    """Construct smoke cells or the full 2-wave 4-decade scale-law matrix across 6 shards."""
    if smoke:
        return [
            ScaleLawCell("smk_g1_tiny", 1, 0, "gear1_precision_graph", "multiscale_sensor", False, 4096, 128, 128, 2.0, 2),
            ScaleLawCell("smk_g2_ivf", 1, 0, "gear2_ivf_routing", "balanced_gmm", False, 8192, 1024, 128, 2.0, 2),
            ScaleLawCell("smk_g3a_wide", 1, 0, "3A_wide_batch", "balanced_gmm", False, 4096, 128, 64, 2.0, 2),
            ScaleLawCell("smk_g3b_spec", 1, 0, "3B_spectral_split", "curved_manifold", False, 4096, 128, 128, 2.0, 2),
            ScaleLawCell("smk_g3c_he", 1, 0, "3C_head_edit", "zipf_gmm", False, 4096, 128, 64, 2.0, 2),
            ScaleLawCell("smk_g3d_eb", 1, 0, "3D_simplex_eb", "zipf_gmm", False, 4096, 256, 128, 2.0, 2),
        ]

    cells: list[ScaleLawCell] = [
        # ---------------- WAVE 1 (~45-60s across 6 GPUs): Immediate Live Crossover Laws ----------------
        # Shard 0 (ripper:0) — Gear 1 Small-to-Medium + Multi-Scale Sensor + 1M Crossover
        ScaleLawCell("g1_n10k_k256_d32_t1", 1, 0, "gear1_precision_graph", "balanced_gmm", False, 10_000, 256, 32, 1.0, 3),
        ScaleLawCell("g1_n10k_k256_d128_t2", 1, 0, "gear1_precision_graph", "aniso_gmm_shifted", False, 10_000, 256, 128, 2.0, 3),
        ScaleLawCell("g1_n10k_k1024_d128_t5", 1, 0, "gear1_precision_graph", "balanced_gmm", False, 10_000, 1024, 128, 5.0, 3),
        ScaleLawCell("g1_n50k_k256_d128_t1", 1, 0, "gear1_precision_graph", "balanced_gmm", False, 50_000, 256, 128, 1.0, 3),
        ScaleLawCell("g1_n50k_k1024_d128_t2", 1, 0, "gear1_precision_graph", "aniso_gmm_shifted", False, 50_000, 1024, 128, 2.0, 3),
        ScaleLawCell("g1_n50k_k1024_d512_t2_cifar", 1, 0, "gear1_precision_graph", "cifar10_r18_512_val", True, 50_000, 1024, 512, 2.0, 3),
        ScaleLawCell("g1_n50k_k4096_d128_t2", 1, 0, "gear1_precision_graph", "balanced_gmm", False, 50_000, 4096, 128, 2.0, 3),
        ScaleLawCell("g1_n1m_k4096_d128_t2_sift", 1, 0, "gear1_precision_graph", "sift1m_val", True, 1_000_000, 4096, 128, 2.0, 3),

        # Shard 1 (ripper:1) — Gear 1 High-D (512, 1024), Sensor Skew `[0..3800]`, & Million-Scale K=16,384
        ScaleLawCell("g1_n10k_k256_d512_t2", 1, 1, "gear1_precision_graph", "aniso_gmm_shifted", False, 10_000, 256, 512, 2.0, 3),
        ScaleLawCell("g1_n50k_k1024_d32_t2", 1, 1, "gear1_precision_graph", "balanced_gmm", False, 50_000, 1024, 32, 2.0, 3),
        ScaleLawCell("g1_n50k_k1024_d1024_t2", 1, 1, "gear1_precision_graph", "aniso_gmm_shifted", False, 50_000, 1024, 1024, 2.0, 3),
        ScaleLawCell("g1_n200k_k1024_d128_t2_sensor", 1, 1, "gear1_precision_graph", "multiscale_sensor", False, 200_000, 1024, 128, 2.0, 3),
        ScaleLawCell("g1_n200k_k4096_d128_t2_patches", 1, 1, "gear1_precision_graph", "cifar10_patches_128_val", True, 200_000, 4096, 128, 2.0, 3),
        ScaleLawCell("g1_n200k_k4096_d512_t2", 1, 1, "gear1_precision_graph", "aniso_gmm_shifted", False, 200_000, 4096, 512, 2.0, 3),
        ScaleLawCell("g1_n200k_k4096_d1024_t2", 1, 1, "gear1_precision_graph", "aniso_gmm_shifted", False, 200_000, 4096, 1024, 2.0, 3),
        ScaleLawCell("g1_n1m_k16384_d128_t2_sift", 1, 1, "gear1_precision_graph", "sift1m_val", True, 1_000_000, 16384, 128, 2.0, 3),

        # Shard 2 (ripper:2) — Gear 2 IVF->Exact Crossover (`K = 4096, 8192, 16384, 32768`)
        ScaleLawCell("g2_n200k_k4096_d128_t3_patches", 1, 2, "gear2_ivf_routing", "cifar10_patches_128_val", True, 200_000, 4096, 128, 3.0, 3),
        ScaleLawCell("g2_n200k_k8192_d128_t3", 1, 2, "gear2_ivf_routing", "balanced_gmm", False, 200_000, 8192, 128, 3.0, 3),
        ScaleLawCell("g2_n200k_k16384_d128_t3", 1, 2, "gear2_ivf_routing", "balanced_gmm", False, 200_000, 16384, 128, 3.0, 3),
        ScaleLawCell("g2_n1m_k4096_d128_t3_sift", 1, 2, "gear2_ivf_routing", "sift1m_val", True, 1_000_000, 4096, 128, 3.0, 3),
        ScaleLawCell("g2_n1m_k8192_d128_t3_sift", 1, 2, "gear2_ivf_routing", "sift1m_val", True, 1_000_000, 8192, 128, 3.0, 3),
        ScaleLawCell("g2_n1m_k16384_d128_t3_sift", 1, 2, "gear2_ivf_routing", "sift1m_val", True, 1_000_000, 16384, 128, 3.0, 3),

        # Shard 3 (ripper:3) — Gear 2 Huge-K (`K = 32768, 65536` on Real SIFT1M Val!)
        ScaleLawCell("g2_n1m_k32768_d128_t3_sift", 1, 3, "gear2_ivf_routing", "sift1m_val", True, 1_000_000, 32768, 128, 3.0, 3),
        ScaleLawCell("g2_n1m_k65536_d128_t3_sift", 1, 3, "gear2_ivf_routing", "sift1m_val", True, 1_000_000, 65536, 128, 3.0, 3),

        # Shard 4 (loki:0) — Gear 3A (`wide_batch`) & Gear 3B (`spectral_split`) across `N/K in {16,64,256,1024}` & `D`
        ScaleLawCell("g3a_nk16_k1024_d128_t2", 1, 4, "3A_wide_batch", "balanced_gmm", False, 16_384, 1024, 128, 2.0, 3),
        ScaleLawCell("g3a_nk64_k1024_d128_t2", 1, 4, "3A_wide_batch", "balanced_gmm", False, 65_536, 1024, 128, 2.0, 3),
        ScaleLawCell("g3a_nk256_k1024_d128_t2", 1, 4, "3A_wide_batch", "balanced_gmm", False, 262_144, 1024, 128, 2.0, 3),
        ScaleLawCell("g3a_nk1024_k1024_d128_t2", 1, 4, "3A_wide_batch", "balanced_gmm", False, 1_048_576, 1024, 128, 2.0, 3),
        ScaleLawCell("g3a_nk64_k4096_d128_t1", 1, 4, "3A_wide_batch", "cifar10_patches_128_val", True, 200_000, 4096, 128, 1.0, 3),
        ScaleLawCell("g3b_gmm_d128_t2", 1, 4, "3B_spectral_split", "balanced_gmm", False, 65_536, 1024, 128, 2.0, 3),
        ScaleLawCell("g3b_rare_d128_t2", 1, 4, "3B_spectral_split", "multiscale_rare_outliers", False, 65_536, 512, 128, 2.0, 3),
        ScaleLawCell("g3b_curved_d512_t2", 1, 4, "3B_spectral_split", "curved_manifold", False, 50_000, 1024, 512, 2.0, 3),
        ScaleLawCell("g3b_cifar_d512_t2", 1, 4, "3B_spectral_split", "cifar10_r18_512_val", True, 50_000, 1024, 512, 2.0, 3),
        ScaleLawCell("g3b_curved_d1024_t2", 1, 4, "3B_spectral_split", "curved_manifold", False, 50_000, 512, 1024, 2.0, 3),

        # Shard 5 (loki:1) — Gear 3C (`head_edit`) & Gear 3D (`simplex_eb_tau`) across `N/K in {16,64,256,1024}` & `D`
        ScaleLawCell("g3c_nk16_k1024_d128_t2", 1, 5, "3C_head_edit", "zipf_gmm", False, 16_384, 1024, 128, 2.0, 3),
        ScaleLawCell("g3c_nk64_k1024_d128_t2", 1, 5, "3C_head_edit", "zipf_gmm", False, 65_536, 1024, 128, 2.0, 3),
        ScaleLawCell("g3c_nk256_k1024_d128_t2", 1, 5, "3C_head_edit", "zipf_gmm", False, 262_144, 1024, 128, 2.0, 3),
        ScaleLawCell("g3c_nk1024_k256_d128_t2", 1, 5, "3C_head_edit", "zipf_gmm", False, 262_144, 256, 128, 2.0, 3),
        ScaleLawCell("g3c_cifar_k1024_d512_t2", 1, 5, "3C_head_edit", "cifar10_r18_512_val", True, 50_000, 1024, 512, 2.0, 3),
        ScaleLawCell("g3d_nk16_d32_t2", 1, 5, "3D_simplex_eb", "zipf_gmm", False, 16_384, 1024, 32, 2.0, 3),
        ScaleLawCell("g3d_nk16_d128_t2", 1, 5, "3D_simplex_eb", "zipf_gmm", False, 16_384, 1024, 128, 2.0, 3),
        ScaleLawCell("g3d_nk16_d512_t2", 1, 5, "3D_simplex_eb", "zipf_gmm", False, 16_384, 1024, 512, 2.0, 3),
        ScaleLawCell("g3d_nk16_d1024_t2", 1, 5, "3D_simplex_eb", "zipf_gmm", False, 16_384, 1024, 1024, 2.0, 3),
        ScaleLawCell("g3d_nk64_d128_t2", 1, 5, "3D_simplex_eb", "zipf_gmm", False, 65_536, 1024, 128, 2.0, 3),
        ScaleLawCell("g3d_nk64_d512_t2_cifar", 1, 5, "3D_simplex_eb", "cifar10_r18_512_val", True, 50_000, 1024, 512, 2.0, 3),
        ScaleLawCell("g3d_nk256_d128_t2", 1, 5, "3D_simplex_eb", "zipf_gmm", False, 262_144, 1024, 128, 2.0, 3),
        ScaleLawCell("g3d_nk1024_d128_t2", 1, 5, "3D_simplex_eb", "isotropic_gaussian", False, 262_144, 256, 128, 2.0, 3),

        # ---------------- WAVE 2 (~70-95s across 6 GPUs): Extreme `N=5,000,000` & `K=65,536` Laws ----------------
        # Shard 0 (ripper:0) — Gear 1 Extreme `N = 5,000,000` + `D = 1024 @ N = 1,000,000`
        ScaleLawCell("g1_n200k_k256_d128_t2", 2, 0, "gear1_precision_graph", "balanced_gmm", False, 200_000, 256, 128, 2.0, 3),
        ScaleLawCell("g1_n200k_k1024_d128_t1", 2, 0, "gear1_precision_graph", "balanced_gmm", False, 200_000, 1024, 128, 1.0, 3),
        ScaleLawCell("g1_n200k_k4096_d32_t2", 2, 0, "gear1_precision_graph", "balanced_gmm", False, 200_000, 4096, 32, 2.0, 3),
        ScaleLawCell("g1_n1m_k1024_d128_t1", 2, 0, "gear1_precision_graph", "balanced_gmm", False, 1_000_000, 1024, 128, 1.0, 3),
        ScaleLawCell("g1_n1m_k4096_d32_t2", 2, 0, "gear1_precision_graph", "balanced_gmm", False, 1_000_000, 4096, 32, 2.0, 3),
        ScaleLawCell("g1_n1m_k4096_d512_t2", 2, 0, "gear1_precision_graph", "aniso_gmm_shifted", False, 1_000_000, 4096, 512, 2.0, 3),
        ScaleLawCell("g1_n5m_k4096_d128_t1", 2, 0, "gear1_precision_graph", "balanced_gmm", False, 5_000_000, 4096, 128, 1.0, 3),

        # Shard 1 (ripper:1) — Gear 1 `N = 5,000,000, K = 16,384` + `1M x 1024D` + `T=5`
        ScaleLawCell("g1_n200k_k16384_d128_t2", 2, 1, "gear1_precision_graph", "balanced_gmm", False, 200_000, 16384, 128, 2.0, 3),
        ScaleLawCell("g1_n200k_k4096_d512_t5", 2, 1, "gear1_precision_graph", "aniso_gmm_shifted", False, 200_000, 4096, 512, 5.0, 3),
        ScaleLawCell("g1_n1m_k4096_d1024_t1", 2, 1, "gear1_precision_graph", "aniso_gmm_shifted", False, 1_000_000, 4096, 1024, 1.0, 3),
        ScaleLawCell("g1_n1m_k16384_d128_t5_sift", 2, 1, "gear1_precision_graph", "sift1m_val", True, 1_000_000, 16384, 128, 5.0, 3),
        ScaleLawCell("g1_n5m_k16384_d128_t2", 2, 1, "gear1_precision_graph", "balanced_gmm", False, 5_000_000, 16384, 128, 2.0, 3),

        # Shard 2 (ripper:2) — Gear 2 Multi-Budget `T in {2, 5}`, `D in {32, 512}`, & GloVe-1M Val
        ScaleLawCell("g2_n1m_k16384_d128_t2_sift", 2, 2, "gear2_ivf_routing", "sift1m_val", True, 1_000_000, 16384, 128, 2.0, 3),
        ScaleLawCell("g2_n1m_k16384_d128_t5_sift", 2, 2, "gear2_ivf_routing", "sift1m_val", True, 1_000_000, 16384, 128, 5.0, 3),
        ScaleLawCell("g2_n1m_k16384_d100_t3_glove", 2, 2, "gear2_ivf_routing", "glove1m_val", True, 1_000_000, 16384, 100, 3.0, 3),
        ScaleLawCell("g2_n1m_k32768_d32_t3", 2, 2, "gear2_ivf_routing", "aniso_gmm_shifted", False, 1_000_000, 32768, 32, 3.0, 3),
        ScaleLawCell("g2_n200k_k16384_d512_t3", 2, 2, "gear2_ivf_routing", "aniso_gmm_shifted", False, 200_000, 16384, 512, 3.0, 3),

        # Shard 3 (ripper:3) — Gear 2 Extreme `N = 5,000,000, K = 65,536` (`N*K = 3.28e11`) & `T in {2, 5}`
        ScaleLawCell("g2_n1m_k65536_d128_t2_sift", 2, 3, "gear2_ivf_routing", "sift1m_val", True, 1_000_000, 65536, 128, 2.0, 3),
        ScaleLawCell("g2_n1m_k65536_d128_t5_sift", 2, 3, "gear2_ivf_routing", "sift1m_val", True, 1_000_000, 65536, 128, 5.0, 2),
        ScaleLawCell("g2_n5m_k65536_d128_t2", 2, 3, "gear2_ivf_routing", "balanced_gmm", False, 5_000_000, 65536, 128, 2.0, 2),

        # Shard 4 (loki:0) — Gear 3A/3B Million-Scale Real Corpora & High-T Refinement
        ScaleLawCell("g3a_sift_k16384_t2", 2, 4, "3A_wide_batch", "sift1m_val", True, 1_000_000, 16384, 128, 2.0, 3),
        ScaleLawCell("g3a_nk64_k1024_d128_t5", 2, 4, "3A_wide_batch", "balanced_gmm", False, 65_536, 1024, 128, 5.0, 3),
        ScaleLawCell("g3b_gmm_d32_t2", 2, 4, "3B_spectral_split", "balanced_gmm", False, 65_536, 1024, 32, 2.0, 3),
        ScaleLawCell("g3b_gmm_d1024_t2", 2, 4, "3B_spectral_split", "balanced_gmm", False, 32_768, 512, 1024, 2.0, 3),
        ScaleLawCell("g3b_glove_d100_t2", 2, 4, "3B_spectral_split", "glove1m_val", True, 1_000_000, 4096, 100, 2.0, 3),

        # Shard 5 (loki:1) — Gear 3C/3D Million-Scale Real Corpora & High-D Sparse/Dense Boundary
        ScaleLawCell("g3c_sift_k16384_t2", 2, 5, "3C_head_edit", "sift1m_val", True, 1_000_000, 16384, 128, 2.0, 3),
        ScaleLawCell("g3c_nk64_k4096_d128_t5", 2, 5, "3C_head_edit", "zipf_gmm", False, 262_144, 4096, 128, 5.0, 3),
        ScaleLawCell("g3d_sift_k16384_t2", 2, 5, "3D_simplex_eb", "sift1m_val", True, 1_000_000, 16384, 128, 2.0, 3),
        ScaleLawCell("g3d_nk256_d512_t2", 2, 5, "3D_simplex_eb", "zipf_gmm", False, 65_536, 256, 512, 2.0, 3),
        ScaleLawCell("g3d_nk1024_d1024_t2", 2, 5, "3D_simplex_eb", "zipf_gmm", False, 262_144, 256, 1024, 2.0, 3),
        ScaleLawCell("g3d_nk16_d128_t1", 2, 5, "3D_simplex_eb", "zipf_gmm", False, 16_384, 1024, 128, 1.0, 3),
        ScaleLawCell("g3d_nk16_d128_t5", 2, 5, "3D_simplex_eb", "zipf_gmm", False, 16_384, 1024, 128, 5.0, 3),
    ]
    return cells


# ==============================================================================
# 8. Untimed Full-Shape GPU Warmup & Single-Cell Paired Executor
# ==============================================================================


@torch.no_grad()
def warmup_global_kernels(device: torch.device) -> None:
    """Pre-warm global Triton tournament & evaluation specializations outside all timers."""
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
                        pair_d2, pool_d2, w_p, out_idx, p_v, ns_v, rs_v,
                        BLOCK_P=block_p, L_TRIALS=3, num_warps=n_warps,
                    )
    torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate_validation_mse_fp64(
    x_val: torch.Tensor,
    centroids: torch.Tensor,
    geometry: str,
    device: torch.device,
) -> tuple[float, float | None]:
    """Compute exact IEEE-754 FP64 validation MSE (and spherical cosine distortion if unit-sphere)."""
    val_mse_raw, _ = fast_exact_assign_fp64(x_val, centroids, device=device)
    val_mse = float(val_mse_raw)
    cos_dist: float | None = None
    if geometry == "unit_sphere":
        cos_dist = float(exact_cosine_distortion_fp64(x_val, centroids, device=device))
    return val_mse, cos_dist


@torch.no_grad()
def execute_cell(
    cell: ScaleLawCell,
    seed_start: int,
    device: torch.device,
    million_data_dir: Path = DEFAULT_MILLION_DATA_DIR,
    phase1_cache_dir: Path = PHASE1_CACHE_DIR,
) -> list[dict[str, Any]]:
    """Execute untimed full-shape warmup + cyclic-counterbalanced paired seeds for `cell`."""
    host = socket.gethostname()
    vis_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    gpu_sku = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    arms = SUITE_ARMS[cell.suite]
    num_arms = len(arms)
    seeds = [int(seed_start) + i for i in range(cell.num_seeds)]

    real_prob: ValidationProblem | None = None
    if cell.is_real:
        real_prob = load_real_validation_corpus(
            cell.problem_kind,
            device=device,
            million_data_dir=million_data_dir,
            phase1_cache_dir=phase1_cache_dir,
            n_train_limit=cell.n,
        )

    def _get_problem(s_val: int) -> ValidationProblem:
        if real_prob is not None:
            return real_prob
        return make_gpu_validation_problem(
            kind=cell.problem_kind,
            n_train=cell.n,
            d=cell.d,
            k=cell.k,
            seed=s_val,
            device=device,
        )

    # 1. Full-shape untimed warmup across EVERY arm in `cell` (eliminates 100% of cold-JIT spikes!)
    prob_warm = _get_problem(seed_start + 99_991)
    m_sub_w = min(int(prob_warm.x_train.shape[0]), max(4096, 6 * cell.k))
    xw_sub = prob_warm.x_train[: min(m_sub_w, 16384)].float().contiguous()
    for dt in (torch.float32, torch.bfloat16):
        for kc in (1, 16, 17, 64, 65):
            if kc <= xw_sub.shape[0]:
                assign_points(xw_sub, xw_sub[:kc], compute_dtype=dt)

    if cell.suite in ("gear1_precision_graph", "gear2_ivf_routing"):
        orig64_w, _ = _compute_origin_and_scale(prob_warm.x_train, torch.bfloat16)
        use_fast_hk = cell.suite == "gear2_ivf_routing" and cell.k >= 16384
        sres_w = adaptive_turkpp_init_centered(
            prob_warm.x_train, orig64_w, 1.0, cell.k, torch.bfloat16, seed=999_991, huge_k_fast=use_fast_hk
        )
        if use_fast_hk and cell.n <= 1_000_000:
            # Pre-compile standard `ultra_turkpp_init_centered` outside timers before `s_idx == 0` ref timing!
            _ = ultra_turkpp_init_centered(
                prob_warm.x_train, orig64_w, 1.0, cell.k, torch.bfloat16, coreset_mult=6, polish=True, seed=999_994
            )
        for arm in arms:
            if cell.suite == "gear1_precision_graph":
                cw, _ = run_gear1_arm(arm, prob_warm.x_train, orig64_w, sres_w, 1.0, cell.k, cell.nepochs, 999_992)
            else:
                cw, _ = run_gear2_ivf_arm(arm, prob_warm.x_train, orig64_w, sres_w, 1.0, cell.k, cell.nepochs, 999_992)
        _, _ = evaluate_validation_mse_fp64(prob_warm.x_val, cw, prob_warm.geometry, device)
    else:
        for arm in arms:
            solver_w = instantiate_gear3_arm(cell.suite, arm, d=cell.d, k=cell.k, nepochs=cell.nepochs, seed=999_993)
            cw, _ = solver_w.fit(prob_warm.x_train, capture_assignments=False)
        _, _ = evaluate_validation_mse_fp64(prob_warm.x_val, cw, prob_warm.geometry, device)

    del prob_warm
    torch.cuda.synchronize(device)

    # 2. Timed cyclic-counterbalanced seed loop
    records: list[dict[str, Any]] = []
    for s_idx, s_val in enumerate(seeds):
        prob = _get_problem(s_val)
        n_actual = int(prob.x_train.shape[0])
        d_actual = int(prob.x_train.shape[1])
        rot = s_idx % num_arms
        ordered_arms = arms[rot:] + arms[:rot]

        orig64: torch.Tensor | None = None
        seed_res: SeedingResult | None = None
        shared_init_ms = 0.0
        std_init_ms_ref: float | None = None

        if cell.suite in ("gear1_precision_graph", "gear2_ivf_routing"):
            use_fast_hk = cell.suite == "gear2_ivf_routing" and cell.k >= 16384
            torch.cuda.synchronize(device)
            t_in0 = time.perf_counter()
            orig64, _ = _compute_origin_and_scale(prob.x_train, torch.bfloat16)
            seed_res = adaptive_turkpp_init_centered(
                prob.x_train, orig64, 1.0, cell.k, torch.bfloat16, seed=s_val, huge_k_fast=use_fast_hk
            )
            torch.cuda.synchronize(device)
            shared_init_ms = (time.perf_counter() - t_in0) * 1e3

            # On s_idx == 0 of Huge-K Gear 2 cells, also measure standard `ultra_turkpp` init time once
            # to quantify the exact Huge-K seeding gear speedup!
            if use_fast_hk and s_idx == 0 and cell.n <= 1_000_000:
                torch.cuda.synchronize(device)
                t_std0 = time.perf_counter()
                _ = ultra_turkpp_init_centered(
                    prob.x_train, orig64, 1.0, cell.k, torch.bfloat16, coreset_mult=6, polish=True, seed=s_val
                )
                torch.cuda.synchronize(device)
                std_init_ms_ref = (time.perf_counter() - t_std0) * 1e3

        seed_arm_out: dict[str, dict[str, Any]] = {}
        for arm in ordered_arms:
            if cell.suite == "gear1_precision_graph":
                assert orig64 is not None and seed_res is not None
                c_fit, m_info = run_gear1_arm(
                    arm, prob.x_train, orig64, seed_res, shared_init_ms, cell.k, cell.nepochs, s_val
                )
                val_mse, val_cos = evaluate_validation_mse_fp64(prob.x_val, c_fit, prob.geometry, device)
                rec = {
                    "cell_id": cell.cell_id,
                    "wave": cell.wave,
                    "shard_idx": cell.shard_idx,
                    "suite": cell.suite,
                    "problem_kind": cell.problem_kind,
                    "source_split": prob.source_split,
                    "n": n_actual,
                    "n_val": int(prob.x_val.shape[0]),
                    "k": cell.k,
                    "d": d_actual,
                    "nk_ratio": float(n_actual) / float(cell.k),
                    "nk_prod": float(n_actual) * float(cell.k),
                    "nepochs": float(cell.nepochs),
                    "seed": s_val,
                    "arm": arm,
                    "val_mse_fp64": val_mse,
                    "val_cosine_dist": val_cos if val_cos is not None else -1.0,
                    "host": host,
                    "visible_gpu": vis_gpu,
                    "gpu_sku": gpu_sku,
                    **m_info,
                }
            elif cell.suite == "gear2_ivf_routing":
                assert orig64 is not None and seed_res is not None
                c_fit, m_info = run_gear2_ivf_arm(
                    arm, prob.x_train, orig64, seed_res, shared_init_ms, cell.k, cell.nepochs, s_val
                )
                val_mse, val_cos = evaluate_validation_mse_fp64(prob.x_val, c_fit, prob.geometry, device)
                rec = {
                    "cell_id": cell.cell_id,
                    "wave": cell.wave,
                    "shard_idx": cell.shard_idx,
                    "suite": cell.suite,
                    "problem_kind": cell.problem_kind,
                    "source_split": prob.source_split,
                    "n": n_actual,
                    "n_val": int(prob.x_val.shape[0]),
                    "k": cell.k,
                    "d": d_actual,
                    "nk_ratio": float(n_actual) / float(cell.k),
                    "nk_prod": float(n_actual) * float(cell.k),
                    "nepochs": float(cell.nepochs),
                    "seed": s_val,
                    "arm": arm,
                    "val_mse_fp64": val_mse,
                    "val_cosine_dist": val_cos if val_cos is not None else -1.0,
                    "std_turkpp_init_ms": std_init_ms_ref if std_init_ms_ref is not None else -1.0,
                    "host": host,
                    "visible_gpu": vis_gpu,
                    "gpu_sku": gpu_sku,
                    **m_info,
                }
            else:
                solver = instantiate_gear3_arm(cell.suite, arm, d=d_actual, k=cell.k, nepochs=cell.nepochs, seed=s_val)
                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                c_fit, _ = solver.fit(prob.x_train, capture_assignments=False)
                torch.cuda.synchronize(device)
                wall_ms = (time.perf_counter() - t0) * 1e3
                val_mse, val_cos = evaluate_validation_mse_fp64(prob.x_val, c_fit, prob.geometry, device)
                rec = {
                    "cell_id": cell.cell_id,
                    "wave": cell.wave,
                    "shard_idx": cell.shard_idx,
                    "suite": cell.suite,
                    "problem_kind": cell.problem_kind,
                    "source_split": prob.source_split,
                    "n": n_actual,
                    "n_val": int(prob.x_val.shape[0]),
                    "k": cell.k,
                    "d": d_actual,
                    "nk_ratio": float(n_actual) / float(cell.k),
                    "nk_prod": float(n_actual) * float(cell.k),
                    "nepochs": float(cell.nepochs),
                    "seed": s_val,
                    "arm": arm,
                    "val_mse_fp64": val_mse,
                    "val_cosine_dist": val_cos if val_cos is not None else -1.0,
                    "init_ms": float(solver.init_time_sec * 1e3),
                    "preencode_ms": 0.0,
                    "graph_capture_ms": 0.0,
                    "train_ms": float(solver.train_time_sec * 1e3),
                    "total_ms": wall_ms,
                    "total_with_capture_ms": wall_ms,
                    "batch_size": int(solver.recipe.batch_size) if solver.recipe is not None else 0,
                    "total_steps": int(solver.recipe.total_steps) if solver.recipe is not None else 0,
                    "n_splits": int(solver.n_splits),
                    "n_transplants": int(solver.n_transplants),
                    "dead_clusters": int(solver.dead_clusters),
                    "mean_eb_lambda": (
                        float(solver.shrinkage_diagnostics.mean_lambda)
                        if solver.shrinkage_diagnostics is not None
                        else 1.0
                    ),
                    "fp8_nan_count": 0,
                    "host": host,
                    "visible_gpu": vis_gpu,
                    "gpu_sku": gpu_sku,
                }
            seed_arm_out[arm] = rec

        for arm in arms:
            records.append(seed_arm_out[arm])
        if not cell.is_real:
            del prob

    print(
        f"[ScaleLaw] Done {cell.cell_id} ({cell.suite}, N={cell.n:,}, K={cell.k:,}, D={cell.d}, T={cell.nepochs}) "
        f"-> {len(records)} records on {host}:GPU{vis_gpu}",
        flush=True,
    )
    return records


# ==============================================================================
# 9. Empirical Crossover & Power-Law Regression Engine (`fit_scale_laws`)
# ==============================================================================


def _safe_paired_tests(x_a: np.ndarray, x_b: np.ndarray) -> tuple[float, float]:
    """Paired Student-t and Wilcoxon p-values on `(x_a - x_b)` with zero-diff safety."""
    diffs = np.asarray(x_a, dtype=np.float64) - np.asarray(x_b, dtype=np.float64)
    if diffs.size < 2 or float(np.max(np.abs(diffs))) <= 1e-15:
        return 1.0, 1.0
    p_t = float(stats.ttest_rel(x_a, x_b).pvalue)
    if not math.isfinite(p_t):
        p_t = 1.0
    nz = diffs[np.abs(diffs) > 1e-15]
    if nz.size < 2:
        p_w = 1.0
    else:
        try:
            p_w = float(stats.wilcoxon(nz, zero_method="wilcox", mode="approx").pvalue)
        except Exception:
            p_w = 1.0
        if not math.isfinite(p_w):
            p_w = 1.0
    return p_t, p_w


def validate_scale_records(records: list[dict[str, Any]], expected_cells: list[ScaleLawCell]) -> None:
    """Enforce zero NaN/Inf, zero FP8 NaNs, zero duplicates, and single-GPU cell pinning."""
    _assert_json_finite(records)
    seen_keys: set[tuple[str, int, str]] = set()
    cell_gpus: dict[str, set[tuple[str, str, str]]] = {}

    for r in records:
        key = (str(r["cell_id"]), int(r["seed"]), str(r["arm"]))
        if key in seen_keys:
            raise ValueError(f"Duplicate record key {key}")
        seen_keys.add(key)
        if int(r.get("fp8_nan_count", 0)) != 0:
            raise ValueError(f"Non-zero FP8 NaN count in {key}: {r['fp8_nan_count']}")
        if float(r["val_mse_fp64"]) < 0.0 or float(r["total_ms"]) <= 0.0:
            raise ValueError(f"Invalid metric in {key}: mse={r['val_mse_fp64']}, ms={r['total_ms']}")
        if r["source_split"] not in ("real_val_only", "gpu_synthetic_val_only"):
            raise ValueError(f"Forbidden split {r['source_split']!r} in {key}")
        cell_gpus.setdefault(str(r["cell_id"]), set()).add((str(r["host"]), str(r["visible_gpu"]), str(r["gpu_sku"])))

    for c in expected_cells:
        arms = SUITE_ARMS[c.suite]
        cnt = sum(1 for r in records if r["cell_id"] == c.cell_id)
        exp_cnt = c.num_seeds * len(arms)
        if cnt != exp_cnt:
            raise ValueError(f"Cell {c.cell_id} has {cnt} records, expected {exp_cnt}")
        if len(cell_gpus.get(c.cell_id, set())) != 1:
            raise ValueError(f"Cell {c.cell_id} split across multiple GPUs: {cell_gpus.get(c.cell_id)}")


def fit_scale_laws(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Fit closed-form multi-gear crossover formulas and summary tables from validation records."""
    by_cell_arm: dict[tuple[str, str], list[dict[str, Any]]] = {}
    cells_meta: dict[str, dict[str, Any]] = {}
    for r in records:
        cid = str(r["cell_id"])
        arm = str(r["arm"])
        by_cell_arm.setdefault((cid, arm), []).append(r)
        if cid not in cells_meta:
            cells_meta[cid] = {
                "cell_id": cid,
                "suite": r["suite"],
                "problem_kind": r["problem_kind"],
                "n": int(r["n"]),
                "k": int(r["k"]),
                "d": int(r["d"]),
                "nepochs": float(r["nepochs"]),
                "nk_ratio": float(r["nk_ratio"]),
                "nk_prod": float(r["nk_prod"]),
            }

    for grp in by_cell_arm.values():
        grp.sort(key=lambda x: int(x["seed"]))

    # ---------------- LAW 1: Precision (`CroissAttn-FP8` vs `BF16`) & `CUDAGraph` ----------------
    g1_rows: list[dict[str, Any]] = []
    X_fp8_fulld: list[list[float]] = []
    y_fp8_fulld: list[float] = []
    X_fp8_all: list[list[float]] = []
    y_fp8_all: list[float] = []
    wide_d_indirect_sp: list[float] = []
    max_abs_croiss_mse_pct = 0.0
    sensor_fastops_delta_pct = 0.0
    sensor_croiss_delta_pct = 0.0

    for cid, meta in sorted(cells_meta.items()):
        if meta["suite"] != "gear1_precision_graph":
            continue
        bf_e = by_cell_arm[(cid, "bf16_eager")]
        bf_g = by_cell_arm[(cid, "bf16_cudagraph")]
        fp_fo = by_cell_arm[(cid, "fp8_fastops_eager")]
        fp_ce = by_cell_arm[(cid, "fp8_croiss_eager")]
        fp_cg = by_cell_arm[(cid, "fp8_croiss_cudagraph")]

        t_bfe = np.array([x["total_ms"] for x in bf_e])
        t_bfg = np.array([x["total_ms"] for x in bf_g])
        t_bfg_cold = np.array([x["total_with_capture_ms"] for x in bf_g])
        t_fpcg = np.array([x["total_ms"] for x in fp_cg])
        t_fpce = np.array([x["total_ms"] for x in fp_ce])

        m_bfe = np.array([x["val_mse_fp64"] for x in bf_e])
        m_bfg = np.array([x["val_mse_fp64"] for x in bf_g])
        m_fpfo = np.array([x["val_mse_fp64"] for x in fp_fo])
        m_fpcg = np.array([x["val_mse_fp64"] for x in fp_cg])

        sp_graph_bf16 = float(np.median(t_bfe / t_bfg))
        sp_graph_cold = float(np.median(t_bfe / t_bfg_cold))
        sp_fp8_vs_bf16_graph = float(np.median(t_bfg / t_fpcg))
        sp_fp8_graph_vs_fp8_eager = float(np.median(t_fpce / t_fpcg))

        mse_delta_graph_pct = float(np.mean((m_bfg - m_bfe) / np.maximum(m_bfe, 1e-30)) * 100.0)
        mse_delta_croiss_pct = float(np.mean((m_fpcg - m_bfg) / np.maximum(m_bfg, 1e-30)) * 100.0)
        mse_delta_fastops_pct = float(np.mean((m_fpfo - m_bfg) / np.maximum(m_bfg, 1e-30)) * 100.0)
        max_abs_croiss_mse_pct = max(max_abs_croiss_mse_pct, abs(mse_delta_croiss_pct))
        if meta["problem_kind"] == "multiscale_sensor":
            sensor_fastops_delta_pct = mse_delta_fastops_pct
            sensor_croiss_delta_pct = mse_delta_croiss_pct

        feat_row = [
            1.0,
            math.log10(max(1.0, meta["n"] * meta["k"] * meta["nepochs"])),
            math.log10(max(1.0, meta["d"])),
        ]
        log_sp = math.log10(max(1e-3, sp_fp8_vs_bf16_graph))
        X_fp8_all.append(feat_row)
        y_fp8_all.append(log_sp)
        if int(meta["d"]) <= 256:
            X_fp8_fulld.append(feat_row)
            y_fp8_fulld.append(log_sp)
        else:
            wide_d_indirect_sp.append(sp_fp8_vs_bf16_graph)

        g1_rows.append(
            {
                **meta,
                "batch_size": int(bf_g[0]["batch_size"]),
                "total_steps": int(bf_g[0]["total_steps"]),
                "bf16_eager_ms": float(np.median(t_bfe)),
                "bf16_cudagraph_ms": float(np.median(t_bfg)),
                "bf16_cudagraph_cold_ms": float(np.median(t_bfg_cold)),
                "fp8_croiss_cudagraph_ms": float(np.median(t_fpcg)),
                "fp8_preencode_ms": float(np.median([x["preencode_ms"] for x in fp_cg])),
                "graph_capture_ms": float(np.median([x["graph_capture_ms"] for x in bf_g])),
                "speedup_graph_bf16": sp_graph_bf16,
                "speedup_graph_cold_bf16": sp_graph_cold,
                "speedup_fp8_vs_bf16_graph": sp_fp8_vs_bf16_graph,
                "speedup_fp8_graph_vs_eager": sp_fp8_graph_vs_fp8_eager,
                "mse_delta_graph_pct": mse_delta_graph_pct,
                "mse_delta_croiss_fp8_pct": mse_delta_croiss_pct,
                "mse_delta_fastops_fp8_pct": mse_delta_fastops_pct,
            }
        )

    fp8_law: dict[str, Any] = {
        "max_abs_croiss_val_mse_delta_pct": max_abs_croiss_mse_pct,
        "multiscale_sensor_croiss_mse_delta_pct": sensor_croiss_delta_pct,
        "multiscale_sensor_fastops_mse_delta_pct": sensor_fastops_delta_pct,
        "wide_d_indirect_gather_median_speedup": (
            float(np.median(wide_d_indirect_sp)) if wide_d_indirect_sp else -1.0
        ),
        "wide_d_tma_note": (
            "For D > 256 (512, 1024), non-TMA indirect row_indices gather regresses to ~0.33x median vs BF16; "
            "ApexEMeans MUST pre-permute contiguous blocks (allow_tma=True, row_indices=None -> 1.65x-1.86x) or stay on BF16."
        ),
    }
    X_fit_src = X_fp8_fulld if len(X_fp8_fulld) >= 3 else X_fp8_all
    y_fit_src = y_fp8_fulld if len(X_fp8_fulld) >= 3 else y_fp8_all
    if len(X_fit_src) >= 3:
        X_mat = np.asarray(X_fit_src, dtype=np.float64)
        y_vec = np.asarray(y_fit_src, dtype=np.float64)
        beta, _, _, _ = np.linalg.lstsq(X_mat, y_vec, rcond=None)
        y_hat = X_mat @ beta
        ss_tot = float(np.sum((y_vec - y_vec.mean()) ** 2))
        r2 = 1.0 - float(np.sum((y_vec - y_hat) ** 2)) / max(ss_tot, 1e-30)
        # Solve beta[0] + beta[1]*log10(N*K*T) + beta[2]*log10(D) == 0 on 1-Pass FULL_D (D <= 256) regime!
        b1 = float(beta[1]) if abs(float(beta[1])) > 1e-6 else 0.03
        nkt_d128 = 10.0 ** (-(float(beta[0]) + float(beta[2]) * math.log10(128.0)) / b1)
        nkt_d32 = 10.0 ** (-(float(beta[0]) + float(beta[2]) * math.log10(32.0)) / b1)
        # Practical >=1.05x speedup + low E4M3 quantization-noise threshold:
        nkt_d128_105x = 10.0 ** ((math.log10(1.05) - float(beta[0]) - float(beta[2]) * math.log10(128.0)) / b1)
        d_exp = -float(beta[2]) / b1
        nkt_clamped = float(min(max(nkt_d128, 1e6), 1e12))
        nkt_prac_clamped = float(min(max(nkt_d128_105x, 1e7), 1e12))
        fp8_law.update(
            {
                "fit_regime": "FULL_D_1pass_d_le_256" if len(X_fp8_fulld) >= 3 else "all_d",
                "n_fit_cells": int(len(X_fit_src)),
                "intercept": float(beta[0]),
                "slope_log10_nkt": float(beta[1]),
                "slope_log10_d": float(beta[2]),
                "r2": r2,
                "crossover_nkt_at_d128": nkt_clamped,
                "crossover_nkt_at_d32": float(min(max(nkt_d32, 1e6), 1e12)),
                "practical_105x_nkt_at_d128": nkt_prac_clamped,
                "d_exponent_in_threshold": float(d_exp),
                "rule_python": (
                    f"use_croiss_fp8 = ((n * k * nepochs * ((d / 128.0) ** {-d_exp:.3f})) >= {nkt_prac_clamped:.3e}) "
                    "and ((d <= 256) or contiguous_tma_prepermuted)"
                ),
            }
        )

    # ---------------- LAW 2: Coarse-to-Fine `IVF-Explore + Exact-Close` ----------------
    g2_rows: list[dict[str, Any]] = []
    for cid, meta in sorted(cells_meta.items()):
        if meta["suite"] != "gear2_ivf_routing":
            continue
        ex_bf = by_cell_arm[(cid, "exact_bf16")]
        ex_fp = by_cell_arm[(cid, "exact_fp8")]
        ivf_pure = by_cell_arm[(cid, "ivf_pure_np8")]
        ivf_np4 = by_cell_arm[(cid, "ivf2sq_exact_np4")]
        ivf_np8 = by_cell_arm[(cid, "ivf2sq_exact_np8")]
        ivf_np16 = by_cell_arm[(cid, "ivf2sq_exact_np16")]
        ivf_1sq = by_cell_arm[(cid, "ivf1sq_exact_np8")]

        t_bf = np.array([x["total_ms"] for x in ex_bf])
        t_fp = np.array([x["total_ms"] for x in ex_fp])
        t_np4 = np.array([x["total_ms"] for x in ivf_np4])
        t_np8 = np.array([x["total_ms"] for x in ivf_np8])
        t_np16 = np.array([x["total_ms"] for x in ivf_np16])
        t_1sq = np.array([x["total_ms"] for x in ivf_1sq])

        m_bf = np.array([x["val_mse_fp64"] for x in ex_bf])
        m_fp = np.array([x["val_mse_fp64"] for x in ex_fp])
        m_pure = np.array([x["val_mse_fp64"] for x in ivf_pure])
        m_np4 = np.array([x["val_mse_fp64"] for x in ivf_np4])
        m_np8 = np.array([x["val_mse_fp64"] for x in ivf_np8])
        m_np16 = np.array([x["val_mse_fp64"] for x in ivf_np16])
        m_1sq = np.array([x["val_mse_fp64"] for x in ivf_1sq])

        std_inits = [x["std_turkpp_init_ms"] for x in ex_fp if float(x.get("std_turkpp_init_ms", -1.0)) > 0.0]
        fast_init_ms = float(np.median([x["init_ms"] for x in ex_fp]))

        sp_np8_fp8 = float(np.median(t_fp / t_np8))
        sp_1sq_fp8 = float(np.median(t_fp / t_1sq))
        sp_np8_bf16 = float(np.median(t_bf / t_np8))
        sp_1sq_bf16 = float(np.median(t_bf / t_1sq))
        dmse_np8_fp8 = float(np.mean((m_np8 - m_fp) / np.maximum(m_fp, 1e-30)) * 100.0)
        dmse_1sq_fp8 = float(np.mean((m_1sq - m_fp) / np.maximum(m_fp, 1e-30)) * 100.0)

        g2_rows.append(
            {
                **meta,
                "exact_bf16_ms": float(np.median(t_bf)),
                "exact_fp8_ms": float(np.median(t_fp)),
                "ivf2sq_np4_ms": float(np.median(t_np4)),
                "ivf2sq_np8_ms": float(np.median(t_np8)),
                "ivf2sq_np16_ms": float(np.median(t_np16)),
                "ivf1sq_np8_ms": float(np.median(t_1sq)),
                "speedup_np8_vs_exact_bf16": sp_np8_bf16,
                "speedup_np8_vs_exact_fp8": sp_np8_fp8,
                "speedup_1sq_vs_exact_bf16": sp_1sq_bf16,
                "speedup_1sq_vs_exact_fp8": sp_1sq_fp8,
                "speedup_np4_vs_exact_bf16": float(np.median(t_bf / t_np4)),
                "speedup_np4_vs_exact_fp8": float(np.median(t_fp / t_np4)),
                "mse_delta_np8_vs_exact_fp8_pct": dmse_np8_fp8,
                "mse_delta_1sq_vs_exact_fp8_pct": dmse_1sq_fp8,
                "mse_delta_np4_vs_exact_fp8_pct": float(np.mean((m_np4 - m_fp) / np.maximum(m_fp, 1e-30)) * 100.0),
                "mse_delta_np16_vs_exact_fp8_pct": float(np.mean((m_np16 - m_fp) / np.maximum(m_fp, 1e-30)) * 100.0),
                "mse_delta_pure_ivf_vs_exact_close_pct": float(
                    np.mean((m_pure - m_np8) / np.maximum(m_np8, 1e-30)) * 100.0
                ),
                "mse_delta_1sq_vs_2sq_pct": float(np.mean((m_1sq - m_np8) / np.maximum(m_np8, 1e-30)) * 100.0),
                "fast_hugek_init_ms": fast_init_ms,
                "std_turkpp_init_ms": float(std_inits[0]) if std_inits else -1.0,
            }
        )

    def _is_ivf_win(row: dict[str, Any]) -> bool:
        if int(row["d"]) < 64 or float(row["nk_ratio"]) < 30.0:
            return False
        win_2sq = row["speedup_np8_vs_exact_fp8"] > 1.03 and row["mse_delta_np8_vs_exact_fp8_pct"] <= 0.50
        win_1sq = row["speedup_1sq_vs_exact_fp8"] > 1.10 and row["mse_delta_1sq_vs_exact_fp8_pct"] <= 0.50
        return bool(win_2sq or win_1sq)

    win_k_vals = sorted(r["k"] for r in g2_rows if _is_ivf_win(r))
    win_nk_vals = sorted(r["nk_prod"] for r in g2_rows if _is_ivf_win(r))
    win_nk_ratios = sorted(r["nk_ratio"] for r in g2_rows if _is_ivf_win(r))
    ivf_law = {
        "k_star_vs_exact_fp8": int(win_k_vals[0]) if win_k_vals else 16384,
        "nk_star_vs_exact_fp8": float(win_nk_vals[0]) if win_nk_vals else 1.6e10,
        "min_nk_ratio": float(min(32.0, win_nk_ratios[0])) if win_nk_ratios else 30.0,
        "min_d": 64,
        "recommended_nlist": "clamp(round_power_of_two(1.0 * sqrt(K)), 128, 256)  # 1*sqrt(K) hits 1.13x-2.25x vs FP8, 1.55x-3.23x vs BF16",
        "recommended_nprobe": "8 (or 16 when N >= 4_000_000)",
        "closing_pass_gain_pct_mean": (
            float(np.mean([r["mse_delta_pure_ivf_vs_exact_close_pct"] for r in g2_rows])) if g2_rows else 0.0
        ),
        "rule_python": "use_ivf_explore_exact_close = (k >= 16384) and (d >= 64) and (n >= 30 * k) and (nepochs >= 2.0)",
    }

    # ---------------- LAW 3A-3D: Scale-Adaptive Algorithmic Knobs ----------------
    g3_summary: dict[str, list[dict[str, Any]]] = {
        "3A_wide_batch": [],
        "3B_spectral_split": [],
        "3C_head_edit": [],
        "3D_simplex_eb": [],
    }
    eb_fit_X: list[list[float]] = []
    eb_fit_tau: list[float] = []
    tau_map = {
        "closing_mean": 0.01,
        "closing_ramp_035": 0.01,
        "eb_tau_010": 0.10,
        "eb_tau_025": 0.25,
        "eb_tau_060": 0.60,
        "eb_tau_125": 1.25,
        "eb_tau_250": 2.50,
    }

    for cid, meta in sorted(cells_meta.items()):
        st = meta["suite"]
        if st not in g3_summary:
            continue
        arms = SUITE_ARMS[st]
        base_arm = arms[0]
        base_recs = by_cell_arm[(cid, base_arm)]
        m_base = np.array([x["val_mse_fp64"] for x in base_recs])
        t_base = np.array([x["total_ms"] for x in base_recs])

        arm_stats: dict[str, Any] = {}
        best_arm = base_arm
        best_mse = float(np.mean(m_base))
        for arm in arms:
            ar = by_cell_arm[(cid, arm)]
            ma = np.array([x["val_mse_fp64"] for x in ar])
            ta = np.array([x["total_ms"] for x in ar])
            d_mse_pct = float(np.mean((ma - m_base) / np.maximum(m_base, 1e-30)) * 100.0)
            sp = float(np.median(t_base / ta))
            pt, pw = _safe_paired_tests(ma, m_base)
            mean_m = float(np.mean(ma))
            if mean_m < best_mse:
                best_mse = mean_m
                best_arm = arm
            arm_stats[arm] = {
                "mean_val_mse": mean_m,
                "delta_mse_pct_vs_base": d_mse_pct,
                "median_ms": float(np.median(ta)),
                "speedup_vs_base": sp,
                "p_ttest": pt,
                "p_wilcoxon": pw,
                "mean_transplants": float(np.mean([x.get("n_transplants", 0) for x in ar])),
            }

        if st == "3D_simplex_eb":
            eb_fit_X.append(
                [1.0, math.log10(max(1.0, meta["nk_ratio"])), math.log10(max(1.0, meta["d"]))]
            )
            eb_fit_tau.append(math.log10(tau_map.get(best_arm, 0.25)))

        g3_summary[st].append(
            {
                **meta,
                "base_arm": base_arm,
                "base_steps": int(base_recs[0]["total_steps"]),
                "best_mse_arm": best_arm,
                "arms": arm_stats,
            }
        )

    eb_tau_law: dict[str, Any] = {}
    if len(eb_fit_X) >= 3:
        X_eb = np.asarray(eb_fit_X, dtype=np.float64)
        y_eb = np.asarray(eb_fit_tau, dtype=np.float64)
        b_eb, _, _, _ = np.linalg.lstsq(X_eb, y_eb, rcond=None)
        c0 = 10.0 ** float(b_eb[0])
        eb_tau_law = {
            "coeff": float(c0),
            "exp_nk_ratio": float(b_eb[1]),
            "exp_d": float(b_eb[2]),
            "formula": f"tau*(N/K, D) = clip({c0:.3f} * (N/K)^({float(b_eb[1]):.3f}) * D^({float(b_eb[2]):.3f}), 0.08, 2.50)",
        }

    return {
        "gear1_fp8_cudagraph_law": fp8_law,
        "gear1_cells": g1_rows,
        "gear2_ivf_exact_law": ivf_law,
        "gear2_cells": g2_rows,
        "gear3_eb_tau_law": eb_tau_law,
        "gear3_cells": g3_summary,
    }


# ==============================================================================
# 10. Publication-Grade Plots & Markdown Synthesis
# ==============================================================================


def generate_scale_law_plots(fitted: dict[str, Any], out_dir: Path) -> list[Path]:
    """Generate 4 self-explanatory scale-transition figures in `out_dir` + `out_dir/plots`."""
    plots_sub = out_dir / "plots"
    written: list[Path] = []
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})

    # 1. Gear 1: Precision & CUDAGraph Crossover
    g1 = fitted["gear1_cells"]
    fp8_l = fitted["gear1_fp8_cudagraph_law"]
    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(13.6, 5.2))
    if g1:
        nkt = np.array([max(1.0, r["n"] * r["k"] * r["nepochs"]) for r in g1])
        sp_fp8 = np.array([r["speedup_fp8_vs_bf16_graph"] for r in g1])
        dims = np.array([r["d"] for r in g1])
        log_dims = np.log2(dims)
        vmin_d, vmax_d = float(log_dims.min()), float(max(log_dims.min() + 1.0, log_dims.max()))
        mask_fd = dims <= 256
        mask_wd = ~mask_fd
        sc = ax1a.scatter(
            nkt[mask_fd],
            sp_fp8[mask_fd],
            c=log_dims[mask_fd],
            cmap="viridis",
            vmin=vmin_d,
            vmax=vmax_d,
            s=74,
            marker="o",
            edgecolors="black",
            linewidth=0.65,
            label="$D \\leq 256$ (1-Pass FULL_D)",
        )
        if np.any(mask_wd):
            ax1a.scatter(
                nkt[mask_wd],
                sp_fp8[mask_wd],
                c=log_dims[mask_wd],
                cmap="viridis",
                vmin=vmin_d,
                vmax=vmax_d,
                s=72,
                marker="D",
                alpha=0.85,
                edgecolors="#922b21",
                linewidth=0.9,
                label="$D \\geq 512$ (Indirect Gather; needs Contig TMA)",
            )
        if "intercept" in fp8_l and np.sum(mask_fd) >= 3:
            xs_fit = np.logspace(math.log10(float(nkt[mask_fd].min())), math.log10(float(nkt[mask_fd].max())), 120)
            ys_fit = 10.0 ** (
                fp8_l["intercept"]
                + fp8_l["slope_log10_nkt"] * np.log10(xs_fit)
                + fp8_l["slope_log10_d"] * math.log10(128.0)
            )
            ax1a.plot(
                xs_fit,
                ys_fit,
                color="#1f618d",
                linewidth=2.1,
                label=f"Power-Law Fit ($D=128$, FULL_D, $R^2={fp8_l.get('r2', 0.0):.2f}$)",
            )
        ax1a.axhline(1.0, color="#c0392b", linestyle="--", linewidth=1.4, label="FP8 = BF16 Parity (1.0x)")
        ax1a.set_xscale("log")
        ax1a.set_xlabel("Scale Budget  $N \\cdot K \\cdot T$  (validation runs)")
        ax1a.set_ylabel("End-to-End Speedup: CroissAttn-FP8+Graph / BF16+Graph")
        ax1a.set_title("Gear Boundary 1A: CroissAttn-FP8 vs BF16 (+CUDAGraph)")
        ax1a.grid(True, alpha=0.3)
        cbar = fig1.colorbar(sc, ax=ax1a)
        cbar.set_label("$\\log_2(D)$")
        ax1a.legend(loc="lower left", fontsize=8.0, framealpha=0.92)

        steps = np.array([max(1, r["total_steps"]) for r in g1])
        sp_gr = np.array([r["speedup_graph_bf16"] for r in g1])
        mse_d = np.array([abs(r["mse_delta_croiss_fp8_pct"]) for r in g1])
        ax1b.scatter(steps, sp_gr, color="#2980b9", s=64, edgecolors="black", linewidth=0.6, label="5-Step CUDAGraph vs Eager")
        ax1b.axhline(1.0, color="gray", linestyle="--", linewidth=1.1)
        ax1b.set_xscale("log")
        ax1b.set_xlabel("Total Mini-Batch Steps  $S = T \\cdot N / B$")
        ax1b.set_ylabel("End-to-End Speedup (5-Step CUDAGraph / Eager)")
        ax1b.set_title(f"Gear Boundary 1B: CUDAGraph Gain (Max |FP8 $\\Delta$MSE| = {mse_d.max():.3f}%)")
        ax1b.grid(True, alpha=0.3)
        ax1b.legend(loc="best")
    fig1.tight_layout()
    p1 = out_dir / "gear1_fp8_cudagraph_crossover.png"
    atomic_save_png(fig1, [p1, plots_sub / p1.name])
    plt.close(fig1)
    written.append(p1)

    # 2. Gear 2: IVF-Explore + Exact-Close Crossover across K in {4096..65536}
    g2 = fitted["gear2_cells"]
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(13.6, 5.2))
    if g2:
        g2_valid_d = [r for r in g2 if r["d"] <= 256] or g2
        ks = np.array([r["k"] for r in g2_valid_d])
        sp_1sq_bf = np.array([r.get("speedup_1sq_vs_exact_bf16", r["speedup_np8_vs_exact_bf16"]) for r in g2_valid_d])
        sp_1sq_fp = np.array([r.get("speedup_1sq_vs_exact_fp8", r["speedup_np8_vs_exact_fp8"]) for r in g2_valid_d])
        sp_2sq_fp = np.array([r["speedup_np8_vs_exact_fp8"] for r in g2_valid_d])
        ax2a.scatter(ks, sp_1sq_bf, color="#27ae60", s=74, marker="o", edgecolors="black", label="$1\\sqrt{K}$, np=8 vs Exact BF16")
        ax2a.scatter(ks, sp_1sq_fp, color="#8e44ad", s=74, marker="^", edgecolors="black", label="$1\\sqrt{K}$, np=8 vs Exact FP8")
        ax2a.scatter(ks, sp_2sq_fp, color="#2980b9", s=58, marker="s", alpha=0.8, edgecolors="black", label="$2\\sqrt{K}$, np=8 vs Exact FP8")
        ax2a.axhline(1.0, color="#c0392b", linestyle="--", linewidth=1.3, label="Exact Parity (1.0x)")
        ax2a.axvline(16384, color="#6c3483", linestyle=":", linewidth=1.5, label="Crossover $K^* = 16,384$")
        ax2a.set_xscale("log", base=2)
        ax2a.set_xlabel("Number of Centroids  $K$")
        ax2a.set_ylabel("Total Fit Speedup  (IVF-Explore + 1 Exact Close)")
        ax2a.set_title("Gear Boundary 2A: IVF-Explore + 1 Exact Closing Pass Speedup ($D \\leq 256$)")
        ax2a.grid(True, alpha=0.3)
        ax2a.legend(loc="upper left", fontsize=8.2)

        g2_dense = [r for r in g2_valid_d if r["nk_ratio"] >= 30.0 and r["d"] >= 64] or g2_valid_d
        ks_d = np.array([r["k"] for r in g2_dense])
        dm_ex = np.array([r["mse_delta_np8_vs_exact_fp8_pct"] for r in g2_dense])
        dm_1sq = np.array([r.get("mse_delta_1sq_vs_exact_fp8_pct", r["mse_delta_np8_vs_exact_fp8_pct"]) for r in g2_dense])
        dm_pure = np.array([r["mse_delta_pure_ivf_vs_exact_close_pct"] + r["mse_delta_np8_vs_exact_fp8_pct"] for r in g2_dense])
        ax2b.scatter(ks_d, dm_ex, color="#27ae60", s=72, edgecolors="black", label="IVF ($2\\sqrt{K}$) + 1 Exact Close")
        ax2b.scatter(ks_d, dm_1sq, color="#8e44ad", s=68, marker="^", edgecolors="black", label="IVF ($1\\sqrt{K}$) + 1 Exact Close")
        ax2b.scatter(ks_d, dm_pure, color="#e67e22", s=70, marker="x", linewidth=1.8, label="Pure IVF (No Exact Closing Pass)")
        ax2b.axhline(0.0, color="black", linestyle="--", linewidth=1.1)
        ax2b.axvline(16384, color="#6c3483", linestyle=":", linewidth=1.5)
        ax2b.set_xscale("log", base=2)
        ax2b.set_xlabel("Number of Centroids  $K$  ($N/K \\geq 30, D \\geq 64$)")
        ax2b.set_ylabel("Validation FP64 $\\Delta$MSE vs Pure Exact (%)  [lower = better]")
        ax2b.set_title("Gear Boundary 2B: Regularization + Closing-Pass Accuracy Win")
        ax2b.grid(True, alpha=0.3)
        ax2b.legend(loc="upper left", fontsize=8.2)
    fig2.tight_layout()
    p2 = out_dir / "gear2_ivf_exact_crossover.png"
    atomic_save_png(fig2, [p2, plots_sub / p2.name])
    plt.close(fig2)
    written.append(p2)

    # 3. Gear 3: 4-Panel Scale-Adaptive Algorithmic Knobs
    g3 = fitted["gear3_cells"]
    fig3, axes = plt.subplots(2, 2, figsize=(13.5, 9.2))
    ax3a, ax3b, ax3c, ax3d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    if g3["3A_wide_batch"]:
        # Aggregate duplicate `N/K` buckets via geometric mean for strictly monotonic X coordinates!
        buckets_a: dict[int, list[dict[str, Any]]] = {}
        for r in g3["3A_wide_batch"]:
            b_key = int(round(math.log2(max(2.0, float(r["nk_ratio"])))))
            buckets_a.setdefault(b_key, []).append(r)
        sorted_bkeys = sorted(buckets_a.keys())
        x_nk = [float(2**bk) for bk in sorted_bkeys]
        for arm, col, lbl in [
            ("wide_2x_f25", "#2980b9", "Wide 2x (from 25%)"),
            ("wide_2x_f45", "#27ae60", "Wide 2x (from 45%)"),
            ("wide_4x_f25", "#e67e22", "Wide 4x (from 25%)"),
        ]:
            y_gm = [
                float(np.exp(np.mean([math.log(max(1e-3, rr["arms"][arm]["speedup_vs_base"])) for rr in buckets_a[bk]])))
                for bk in sorted_bkeys
            ]
            ax3a.plot(x_nk, y_gm, "o-", color=col, label=lbl)
        ax3a.axhline(1.0, color="gray", linestyle="--")
        ax3a.set_xscale("log", base=2)
        ax3a.set_xlabel("Points per Centroid  $N / K$")
        ax3a.set_ylabel("Speedup vs Fixed 1x Batch")
        ax3a.set_title("3A: Two-Gear Wide-Batch Speedup vs $N/K$")
        ax3a.grid(True, alpha=0.3)
        ax3a.legend(fontsize=8)

    if g3["3B_spectral_split"]:
        rows_b = g3["3B_spectral_split"]
        labels_b = [r["cell_id"].replace("g3b_", "") for r in rows_b]
        x_pos = np.arange(len(rows_b))
        ax3b.bar(
            x_pos - 0.2,
            [r["arms"]["spec_hg_080"]["delta_mse_pct_vs_base"] for r in rows_b],
            width=0.2,
            color="#27ae60",
            label="Spectral-Gated Half-Gauss ($\\sqrt{2/\\pi}$)",
        )
        ax3b.bar(
            x_pos,
            [r["arms"]["ungated_oja_080"]["delta_mse_pct_vs_base"] for r in rows_b],
            width=0.2,
            color="#2980b9",
            label="Ungated Oja ($\\sqrt{2/\\pi}$)",
        )
        ax3b.bar(
            x_pos + 0.2,
            [r["arms"]["radem_aniso_055"]["delta_mse_pct_vs_base"] for r in rows_b],
            width=0.2,
            color="#c0392b",
            label="s-means Rademacher (0.55)",
        )
        ax3b.axhline(0.0, color="black", linewidth=0.9)
        ax3b.set_xticks(x_pos)
        ax3b.set_xticklabels(labels_b, rotation=22, ha="right", fontsize=8)
        ax3b.set_ylabel("$\\Delta$ Validation MSE vs 0.18 Floor (%)")
        ax3b.set_title("3B: Spectral-Gated Oja vs Rademacher Across Manifolds")
        ax3b.grid(True, axis="y", alpha=0.3)
        ax3b.legend(fontsize=8)

    if g3["3C_head_edit"]:
        rows_c = sorted(g3["3C_head_edit"], key=lambda x: (x["nk_ratio"], x["k"]))
        x_c = np.arange(len(rows_c))
        lbl_c = [f"{r['problem_kind'][:6]}\nN/K={r['nk_ratio']:.0f}" for r in rows_c]
        ax3c.bar(x_c - 0.16, [r["arms"]["he_1cp"]["delta_mse_pct_vs_base"] for r in rows_c], width=0.3, color="#2980b9", label="1 Checkpoint (0.52)")
        ax3c.bar(x_c + 0.16, [r["arms"]["he_2cp"]["delta_mse_pct_vs_base"] for r in rows_c], width=0.3, color="#8e44ad", label="2 Checkpoints (0.46, 0.76)")
        ax3c.axhline(0.0, color="black", linewidth=0.9)
        ax3c.set_xticks(x_c)
        ax3c.set_xticklabels(lbl_c, fontsize=8)
        ax3c.set_ylabel("$\\Delta$ Validation MSE vs No Head-Edit (%)")
        ax3c.set_title("3C: P/Q Head-Edit Transplant Utility vs $N/K$")
        ax3c.grid(True, axis="y", alpha=0.3)
        ax3c.legend(fontsize=8)

    if g3["3D_simplex_eb"]:
        rows_d = sorted(g3["3D_simplex_eb"], key=lambda x: (x["nk_ratio"], x["d"]))
        nk_d = [r["nk_ratio"] for r in rows_d]
        best_gains = [
            min(r["arms"][a]["delta_mse_pct_vs_base"] for a in SUITE_ARMS["3D_simplex_eb"][1:])
            for r in rows_d
        ]
        dims_d = [r["d"] for r in rows_d]
        sc_d = ax3d.scatter(nk_d, best_gains, c=np.log2(dims_d), cmap="plasma", s=70, edgecolors="black")
        ax3d.axhline(0.0, color="black", linestyle="--")
        ax3d.set_xscale("log", base=2)
        ax3d.set_xlabel("Points per Centroid  $N / K$")
        ax3d.set_ylabel("Simplex EB $\\Delta$MSE vs Closing Mean (%)")
        ax3d.set_title("3D: Simplex Empirical-Bayes Shrinkage Gain vs $(N/K, D)$")
        ax3d.grid(True, alpha=0.3)
        fig3.colorbar(sc_d, ax=ax3d, label="$\\log_2(D)$")

    fig3.tight_layout()
    p3 = out_dir / "gear3_adaptive_knobs_grid.png"
    atomic_save_png(fig3, [p3, plots_sub / p3.name])
    plt.close(fig3)
    written.append(p3)

    # 4. Unified 4-Decade Scale-Adaptive Phase Diagram for ApexEMeans
    from matplotlib.colors import ListedColormap

    fig4, ax4 = plt.subplots(figsize=(10.2, 6.4))
    n_grid = np.logspace(4, 7, 260)
    k_grid = np.logspace(np.log10(256), np.log10(65536), 260)
    NN, KK = np.meshgrid(n_grid, k_grid)
    nkt_thresh = float(
        fitted["gear1_fp8_cudagraph_law"].get(
            "practical_105x_nkt_at_d128",
            fitted["gear1_fp8_cudagraph_law"].get("crossover_nkt_at_d128", 1.6e8),
        )
    )
    k_ivf_star = int(fitted["gear2_ivf_exact_law"].get("k_star_vs_exact_fp8", 16384))
    min_nk_ratio = float(fitted["gear2_ivf_exact_law"].get("min_nk_ratio", 30.0))

    gear_z = np.ones_like(NN)
    gear_z[(NN * KK * 2.0) >= nkt_thresh] = 2.0
    gear_z[(NN >= 2.0e6) & (KK < k_ivf_star)] = 4.0
    gear_z[(KK >= k_ivf_star) & (NN >= min_nk_ratio * KK)] = 3.0
    gear_z = np.ma.masked_where(NN < 2.0 * KK, gear_z)

    cmap4 = ListedColormap(["#aed6f1", "#a9dfbf", "#d7bde2", "#f5b7b1"])
    ax4.set_facecolor("#e8e8e8")
    im4 = ax4.pcolormesh(NN, KK, gear_z, shading="auto", cmap=cmap4, vmin=0.5, vmax=4.5, alpha=0.82)
    ax4.plot(
        n_grid,
        np.clip(nkt_thresh / (2.0 * n_grid), 256, 65536),
        color="#1f618d",
        linestyle="--",
        linewidth=2.2,
        label=f"Gear 1->2: BF16+Graph -> CroissAttn-FP8+Graph ($N\\cdot K\\cdot T \\geq {nkt_thresh:.1e}$)",
    )
    n_ivf_curve = np.linspace(max(1e4, min_nk_ratio * k_ivf_star), 1e7, 160)
    ax4.plot(
        n_ivf_curve,
        np.full_like(n_ivf_curve, float(k_ivf_star)),
        color="#6c3483",
        linestyle="-.",
        linewidth=2.2,
        label=f"Gear 2->3: Exact FP8 -> IVF-Explore+Exact-Close ($K \\geq {k_ivf_star:,}, N/K \\geq {min_nk_ratio:.0f}$)",
    )
    k_ivf_diag = np.linspace(float(k_ivf_star), 65536.0, 120)
    ax4.plot(min_nk_ratio * k_ivf_diag, k_ivf_diag, color="#6c3483", linestyle="-.", linewidth=2.2)
    ax4.axvline(
        2.0e6,
        color="#b03a2e",
        linestyle=":",
        linewidth=2.2,
        label="Gear 4: Block-Permuted Streaming / Multi-GPU NCCL ($N \\geq 2\\mathrm{M}$)",
    )
    ax4.plot(n_grid, np.clip(0.5 * n_grid, 256, 65536), color="#555555", linestyle="-", linewidth=1.3, label="Feasibility Floor ($N \\geq 2K$)")
    ax4.text(1.35e4, 3.6e4, "Infeasible\n($N < 2K$)", fontsize=9.2, color="#444444", fontweight="bold")
    ax4.text(1.35e4, 430, "Gear 1:\nBF16 + CUDAGraph", fontsize=9.2, color="#154360", fontweight="bold")
    ax4.text(2.2e5, 2200, "Gear 2:\nExact CroissAttn-FP8\n+ 5-Step CUDAGraph", fontsize=9.6, color="#145a32", fontweight="bold")
    ax4.text(2.30e6, 2.30e4, "Gear 3:\nIVF-Explore ($1\\sqrt{K}$) +\n1 Exact Close + Fast Seed\n(+Streaming at $N \\geq 2\\mathrm{M}$)", fontsize=9.2, color="#4a235a", fontweight="bold")
    ax4.text(2.55e6, 1500, "Gear 4:\nBlock-Permuted\nStreaming / NCCL", fontsize=9.4, color="#78281f", fontweight="bold")
    ax4.set_xscale("log")
    ax4.set_yscale("log", base=2)
    ax4.set_xlim(1e4, 1e7)
    ax4.set_ylim(256, 65536)
    ax4.set_xlabel("Dataset Size  $N$  (4 Orders of Magnitude: $10^4 \\to 10^7$)")
    ax4.set_ylabel("Cluster Count  $K$  ($256 \\to 65,536$)")
    ax4.set_title("Empirical 4-Gear Scale-Adaptive Phase Diagram for ApexEMeans ($D=128, T=2$)")
    ax4.legend(loc="lower right", fontsize=8.0, framealpha=0.94)
    ax4.grid(True, alpha=0.25)
    _ = im4
    fig4.tight_layout()
    p4 = out_dir / "scale_phase_diagram_4decade.png"
    atomic_save_png(fig4, [p4, plots_sub / p4.name])
    plt.close(fig4)
    written.append(p4)

    return written


def write_scale_laws_summary_md(fitted: dict[str, Any], num_records: int, dst: Path) -> None:
    """Write structured empirical scale-transition summary markdown to `dst`."""
    fp8_l = fitted["gear1_fp8_cudagraph_law"]
    ivf_l = fitted["gear2_ivf_exact_law"]
    eb_l = fitted["gear3_eb_tau_law"]

    lines = [
        "# Empirical Scale-Transition & Multi-Gear Power-Law Summary (`scale-law-miner`)",
        "",
        f"- **Total Validated Runs**: `{num_records:,}` (`0` `NaN`/`Inf`, `0` FP8 `0x7f` NaNs, `100%` validation-only splits — zero test/query access).",
        "- **Hardware**: `6x` identical `NVIDIA RTX PRO 6000 Blackwell Max-Q` GPUs (`ripper:0..3`, `loki:0,1`), single-GPU paired per cell.",
        "",
        "## 1. Fitted Gear-Switching Rules for `ApexEMeans`",
        "",
        f"1. **Gear 1 -> Gear 2 (`BF16 + 5-Step CUDAGraph` -> `CroissAttn-FP8 + 5-Step CUDAGraph`)**:",
        f"   - Rule: `{fp8_l.get('rule_python', 'n * k * nepochs >= 1.2e8')}` (`R^2 = {fp8_l.get('r2', 0.0):.4f}`).",
        f"   - Max validation `|Delta MSE|` across all cells (incl. `[0..3800]` sensor skew & `+25` shift): **`{fp8_l.get('max_abs_croiss_val_mse_delta_pct', 0.0):.4f}%`** (vs `{fp8_l.get('multiscale_sensor_fastops_mse_delta_pct', 0.0):+.4f}%` for global pow2 FP8 on `multiscale_sensor`).",
        f"2. **Gear 2 -> Gear 3 (`Exact CroissAttn-FP8` -> `IVF-Explore (T-1) + 1 Exact FP8 Close` + Huge-K Fast Seeding)**:",
        f"   - Rule: `{ivf_l.get('rule_python', 'k >= 16384 and nepochs >= 2.0')}` (`K* = {ivf_l.get('k_star_vs_exact_fp8', 16384):,}`, `(N*K)* = {ivf_l.get('nk_star_vs_exact_fp8', 3.0e9):.2e}`).",
        f"   - Recommended IVF params: `nlist = {ivf_l.get('recommended_nlist', '2*sqrt(K)')}`, `nprobe = {ivf_l.get('recommended_nprobe', '8')}`.",
        f"   - Mean MSE improvement of **1 Exact Closing Pass** over pure IVF: **`{ivf_l.get('closing_pass_gain_pct_mean', 0.0):+.4f}%`**.",
        f"3. **Gear 3 Scale-Adaptive Algorithmic Knobs**:",
        f"   - **Simplex Empirical-Bayes Prior**: `{eb_l.get('formula', 'tau = 0.25')}`.",
        "",
        "## 2. Gear Boundary 1 Table (`Precision & CUDAGraph`)",
        "",
        "| Cell | `N` | `K` | `D` | `T` | `BF16 Eager` (ms) | `BF16 Graph` (ms) | `CroissFP8 Graph` (ms) | Graph Speedup | FP8/BF16 Speedup | CroissFP8 $\\Delta$MSE |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in fitted["gear1_cells"]:
        lines.append(
            f"| `{r['cell_id']}` | `{r['n']:,}` | `{r['k']:,}` | `{r['d']}` | `{r['nepochs']:.1f}` | "
            f"`{r['bf16_eager_ms']:.2f}` | `{r['bf16_cudagraph_ms']:.2f}` | `{r['fp8_croiss_cudagraph_ms']:.2f}` | "
            f"`{r['speedup_graph_bf16']:.2f}x` | **`{r['speedup_fp8_vs_bf16_graph']:.2f}x`** | `{r['mse_delta_croiss_fp8_pct']:+.4f}%` |"
        )

    lines.extend(
        [
            "",
            "## 3. Gear Boundary 2 Table (`IVF-Explore + Exact-Close` at Large `K`)",
            "",
            "| Cell | `N` | `K` | `D` | `T` | `Exact BF16` (ms) | `Exact FP8` (ms) | `IVF2sq+Exact np8` (ms) | vs BF16 | vs FP8 | `np8` $\\Delta$MSE vs FP8 | Fast vs Std Init (ms) |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for r in fitted["gear2_cells"]:
        init_str = (
            f"`{r['fast_hugek_init_ms']:.1f}` vs `{r['std_turkpp_init_ms']:.1f}`"
            if r["std_turkpp_init_ms"] > 0
            else f"`{r['fast_hugek_init_ms']:.1f}`"
        )
        lines.append(
            f"| `{r['cell_id']}` | `{r['n']:,}` | `{r['k']:,}` | `{r['d']}` | `{r['nepochs']:.1f}` | "
            f"`{r['exact_bf16_ms']:.2f}` | `{r['exact_fp8_ms']:.2f}` | **`{r['ivf2sq_np8_ms']:.2f}`** | "
            f"**`{r['speedup_np8_vs_exact_bf16']:.2f}x`** | **`{r['speedup_np8_vs_exact_fp8']:.2f}x`** | "
            f"`{r['mse_delta_np8_vs_exact_fp8_pct']:+.4f}%` | {init_str} |"
        )

    lines.extend(["", "## 4. Gear Boundary 3 Tables (`Algorithmic Knobs Across N/K and D`)"])
    for st_key, st_title in [
        ("3A_wide_batch", "4A. Batch Widening (`1x` vs `2x` vs `4x`)"),
        ("3B_spectral_split", "4B. Split Scale & Spectral Concentration Gate"),
        ("3C_head_edit", "4C. `P/Q` `head_edit` Utility vs `N/K`"),
        ("3D_simplex_eb", "4D. Simplex Empirical-Bayes Prior `tau(N/K, D)`"),
    ]:
        rows_st = fitted["gear3_cells"].get(st_key, [])
        if not rows_st:
            continue
        lines.extend(
            [
                "",
                f"### {st_title}",
                "",
                "| Cell | `N` | `K` | `N/K` | `D` | `T` | Best MSE Arm | Best $\\Delta$MSE vs Base | Best Arm Time (ms) | Speedup |",
                "| :--- | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: | ---: |",
            ]
        )
        for r in rows_st:
            ba = r["best_mse_arm"]
            ast = r["arms"][ba]
            lines.append(
                f"| `{r['cell_id']}` | `{r['n']:,}` | `{r['k']:,}` | `{r['nk_ratio']:.1f}` | `{r['d']}` | `{r['nepochs']:.1f}` | "
                f"**`{ba}`** | `{ast['delta_mse_pct_vs_base']:+.4f}%` | `{ast['median_ms']:.2f}` | `{ast['speedup_vs_base']:.2f}x` |"
            )

    atomic_write_text(dst, "\n".join(lines) + "\n")


# ==============================================================================
# 11. Dedicated 6-GPU Fleet Dispatcher (`ripper:0..3` + `loki:0,1`)
# ==============================================================================


def dispatch_6gpu_wave(
    wave: str,
    seed_start: int,
    out_dir: Path,
    million_data_dir: Path,
    phase1_cache_dir: Path,
) -> None:
    """Sync code + validation caches to `ripper` & `loki` and run 6 parallel shards (`0..5`)."""
    if wave not in ("1", "2", "all"):
        raise ValueError(f"Unsupported --wave {wave!r}")

    # Hard safety check on approved slots
    for host, gid in APPROVED_SLOTS:
        if "walnut" in host.lower() or host not in ("ripper", "loki"):
            raise RuntimeError(f"SAFETY VIOLATION: unapproved host {host!r} for scale-law-miner!")
        if gid < 0 or gid > 3:
            raise RuntimeError(f"Invalid GPU id {gid} on {host}")

    env = os.environ.copy()
    live_sock = discover_live_ssh_auth_sock()
    if live_sock:
        env["SSH_AUTH_SOCK"] = live_sock

    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    rel_out = out_dir.relative_to(ROOT) if out_dir.is_relative_to(ROOT) else Path("outputs/scale_laws")
    remote_m_data = f"{REMOTE_WORKDIR}/outputs/phase4_million_scale/data"
    remote_p1_cache = f"{REMOTE_WORKDIR}/outputs/phase1/cache"

    def _sync_host(host: str) -> None:
        print(f"[ScaleLaw-Fleet] Syncing workspace & validation caches to {host}:{REMOTE_WORKDIR} ...", flush=True)
        subprocess.run(
            [
                "ssh", "-A", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                host,
                f"mkdir -p {REMOTE_WORKDIR}/outputs/logs {remote_m_data} {remote_p1_cache}",
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
        for src_dir, dst_dir, fname in [
            (phase1_cache_dir, remote_p1_cache, "cifar10_resnet18_global_512.npz"),
            (phase1_cache_dir, remote_p1_cache, "cifar10_resnet18_patches_128.npz"),
            (million_data_dir, remote_m_data, "sift1m_validated_v1.npz"),
            (million_data_dir, remote_m_data, "glove1m_unit_s20261116_validated_v1.npz"),
        ]:
            src_f = src_dir / fname
            if src_f.exists():
                subprocess.run(
                    [
                        "rsync", "-az", "--ignore-existing",
                        "-e", "ssh -A -o BatchMode=yes -o ConnectTimeout=10",
                        str(src_f),
                        f"{host}:{dst_dir}/{fname}",
                    ],
                    env=env,
                    check=True,
                )
        print(f"[ScaleLaw-Fleet] Host {host} ready.", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(_sync_host, ["ripper", "loki"]))

    wave_tag = f"w{wave}"
    sessions: dict[int, tuple[str, int, str]] = {}
    for sidx, (host, gid) in enumerate(APPROVED_SLOTS):
        sess = f"emeans_slaw_s{sidx:02d}_{wave_tag}"
        if sess == "0" or not sess.startswith("emeans_slaw_s"):
            raise RuntimeError(f"SAFETY VIOLATION: invalid session {sess!r}")
        remote_out = f"{REMOTE_WORKDIR}/{rel_out}"
        shard_json = f"{remote_out}/shards/records_{wave_tag}_shard_{sidx:02d}.json"
        remote_cmd = (
            f"mkdir -p /home/catid/emeans_env/tmp {REMOTE_WORKDIR}/outputs/logs {remote_out}/shards && "
            f"rm -f {REMOTE_WORKDIR}/outputs/logs/{sess}.status {shard_json} && "
            f"tmux kill-session -t {sess} 2>/dev/null || true; "
            f"tmux new-session -d -s {sess} "
            + shlex.quote(
                f"export PATH=\"$HOME/.local/bin:/usr/local/cuda/bin:$PATH\"; "
                f"export CUDA_VISIBLE_DEVICES={gid}; "
                f"export TMPDIR=/home/catid/emeans_env/tmp; "
                f"export TRITON_CACHE_DIR=/home/catid/emeans_env/tmp/triton_cache_{sess}; "
                f"export PYTHONPATH={REMOTE_WORKDIR}/src; "
                f"cd {REMOTE_WORKDIR} && "
                f"{REMOTE_PYTHON} scripts/mine_scale_laws.py "
                f"--shard-idx {sidx} --wave {wave} --seed-start {seed_start} "
                f"--million-data-dir {remote_m_data} --phase1-cache-dir {remote_p1_cache} "
                f"--out-dir {remote_out} "
                f"> {REMOTE_WORKDIR}/outputs/logs/{sess}.log 2>&1; "
                f"echo EXIT_CODE=$? > {REMOTE_WORKDIR}/outputs/logs/{sess}.status"
            )
        )
        subprocess.run(
            ["ssh", "-A", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, remote_cmd],
            env=env,
            check=True,
        )
        sessions[sidx] = (host, gid, sess)
        print(f"[ScaleLaw-Fleet] Launched shard {sidx}/6 on {host}:GPU{gid} ({sess})", flush=True)

    t0 = time.perf_counter()
    pending = dict(sessions)
    while pending:
        time.sleep(3.5)
        done: list[int] = []
        for sidx, (host, gid, sess) in list(pending.items()):
            res = subprocess.run(
                [
                    "ssh", "-A", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                    host,
                    f"cat {REMOTE_WORKDIR}/outputs/logs/{sess}.status 2>/dev/null || true",
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
                    log_r = subprocess.run(
                        [
                            "ssh", "-A", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                            host,
                            f"tail -n 60 {REMOTE_WORKDIR}/outputs/logs/{sess}.log 2>/dev/null || true",
                        ],
                        env=env,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    raise RuntimeError(f"Shard {sidx} ({host}:{gid} {sess}) failed (code={code}):\n{log_r.stdout}")
                r_json = f"{REMOTE_WORKDIR}/{rel_out}/shards/records_{wave_tag}_shard_{sidx:02d}.json"
                l_json = shards_dir / f"records_{wave_tag}_shard_{sidx:02d}.json"
                subprocess.run(
                    [
                        "rsync", "-az",
                        "-e", "ssh -A -o BatchMode=yes -o ConnectTimeout=8",
                        f"{host}:{r_json}",
                        str(l_json),
                    ],
                    env=env,
                    check=True,
                )
                print(
                    f"[ScaleLaw-Fleet] Collected shard {sidx} ({host}:GPU{gid}) at +{time.perf_counter() - t0:.1f}s",
                    flush=True,
                )
                done.append(sidx)
        for d_i in done:
            pending.pop(d_i, None)


# ==============================================================================
# 12. CLI Entrypoint
# ==============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Empirical Scale-Transition & Multi-Gear Law Miner")
    p.add_argument("--smoke", action="store_true", help="Run fast 2-seed end-to-end smoke check across all suites")
    p.add_argument("--dispatch-fleet", action="store_true", help="Dispatch across ripper:0..3 + loki:0,1")
    p.add_argument("--merge-only", action="store_true", help="Merge existing shard JSONs, fit laws, and plot")
    p.add_argument("--wave", type=str, default="all", choices=("1", "2", "all"), help="Wave 1, Wave 2, or all")
    p.add_argument("--shard-idx", type=int, default=None, help="Worker shard index (0..5)")
    p.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START, help="Validation RNG seed start (>=70000)")
    p.add_argument("--million-data-dir", type=Path, default=DEFAULT_MILLION_DATA_DIR)
    p.add_argument("--phase1-cache-dir", type=Path, default=PHASE1_CACHE_DIR)
    p.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    return p.parse_args()


def _select_wave_cells(all_cells: list[ScaleLawCell], wave: str) -> list[ScaleLawCell]:
    if wave == "all":
        return all_cells
    w_int = int(wave)
    return [c for c in all_cells if c.wave == w_int]


def resolve_waves_to_merge(
    wave: str,
    shards_dir: Path,
    all_cells: list[ScaleLawCell],
) -> tuple[list[str], list[ScaleLawCell]]:
    """Resolve shard wave tags & expected cells so `--wave 2` automatically preserves Wave 1 shards."""
    target_cells = _select_wave_cells(all_cells, wave)
    if wave == "all" and (shards_dir / "records_wall_shard_00.json").exists():
        return ["wall"], all_cells
    has_all_w1 = all(
        (shards_dir / f"records_w1_shard_{sidx:02d}.json").exists()
        for sidx in range(len(APPROVED_SLOTS))
    )
    if wave in ("2", "all") and has_all_w1:
        return ["w1", "w2"], all_cells
    return [f"w{wave}"], target_cells


def main() -> None:
    args = parse_args()
    out_dir: Path = args.out_dir
    seed_start = int(args.seed_start)
    if seed_start < 10_000:
        raise ValueError("seed_start must be >= 10,000 to stay strictly isolated from Phase 1-5 benchmark seeds")

    if args.smoke:
        cells = build_scale_cells(smoke=True)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if dev.type != "cuda":
            raise RuntimeError("--smoke requires a CUDA GPU for FP8/BF16/CUDAGraph Triton verification")
        warmup_global_kernels(dev)
        recs: list[dict[str, Any]] = []
        for c in cells:
            recs.extend(
                execute_cell(
                    c,
                    seed_start=seed_start,
                    device=dev,
                    million_data_dir=args.million_data_dir,
                    phase1_cache_dir=args.phase1_cache_dir,
                )
            )
        validate_scale_records(recs, cells)
        fitted = fit_scale_laws(recs)
        payload = {
            "phase": "scale_laws_smoke",
            "num_records": len(recs),
            "seed_start": seed_start,
            "cells": [asdict(c) for c in cells],
            "fitted_laws": fitted,
            "records": recs,
        }
        atomic_write_json(out_dir / "scale_laws_results.json", payload)
        write_scale_laws_summary_md(fitted, len(recs), out_dir / "scale_laws_summary.md")
        plots = generate_scale_law_plots(fitted, out_dir)
        print(f"[ScaleLaw-Smoke] SUCCESS: {len(recs)} records, {len(plots)} PNGs -> {out_dir}", flush=True)
        return

    all_cells = build_scale_cells(smoke=False)
    target_cells = _select_wave_cells(all_cells, args.wave)

    if args.shard_idx is not None:
        sidx = int(args.shard_idx)
        if not (0 <= sidx < len(APPROVED_SLOTS)):
            raise ValueError(f"Expected --shard-idx in 0..{len(APPROVED_SLOTS) - 1}, got {sidx}")
        shard_cells = [c for c in target_cells if c.shard_idx == sidx]
        dev = torch.device("cuda")
        warmup_global_kernels(dev)
        recs = []
        for c in shard_cells:
            recs.extend(
                execute_cell(
                    c,
                    seed_start=seed_start,
                    device=dev,
                    million_data_dir=args.million_data_dir,
                    phase1_cache_dir=args.phase1_cache_dir,
                )
            )
        validate_scale_records(recs, shard_cells)
        wave_tag = f"w{args.wave}"
        shard_path = out_dir / "shards" / f"records_{wave_tag}_shard_{sidx:02d}.json"
        atomic_write_json(
            shard_path,
            {
                "shard_idx": sidx,
                "wave": args.wave,
                "num_records": len(recs),
                "records": recs,
            },
        )
        print(f"[ScaleLaw-Shard {sidx:02d}] Wrote {len(recs)} records -> {shard_path}", flush=True)
        return

    if args.dispatch_fleet:
        dispatch_6gpu_wave(
            wave=args.wave,
            seed_start=seed_start,
            out_dir=out_dir,
            million_data_dir=args.million_data_dir,
            phase1_cache_dir=args.phase1_cache_dir,
        )

    if args.dispatch_fleet or args.merge_only:
        shards_dir = out_dir / "shards"
        merged: list[dict[str, Any]] = []
        w1_records: list[dict[str, Any]] = []
        waves_to_load, expected_cells = resolve_waves_to_merge(args.wave, shards_dir, all_cells)
        for w_tag in waves_to_load:
            for sidx in range(len(APPROVED_SLOTS)):
                sp = shards_dir / f"records_{w_tag}_shard_{sidx:02d}.json"
                if not sp.exists():
                    raise FileNotFoundError(f"Missing shard output {sp}")
                with open(sp, "r", encoding="utf-8") as f:
                    d_json = json.load(
                        f,
                        parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"Invalid JSON constant {c}")),
                    )
                merged.extend(d_json["records"])
                if w_tag == "w1":
                    w1_records.extend(d_json["records"])

        if waves_to_load == ["w1", "w2"] and w1_records:
            w1_cells = [c for c in all_cells if c.wave == 1]
            validate_scale_records(w1_records, w1_cells)
            w1_fitted = fit_scale_laws(w1_records)
            w1_payload = {
                "phase": "scale_laws_wave_1",
                "waves_merged": ["w1"],
                "num_records": len(w1_records),
                "seed_start": seed_start,
                "approved_slots": [{"host": h, "gpu_id": g} for h, g in APPROVED_SLOTS],
                "cells": [asdict(c) for c in w1_cells],
                "fitted_laws": w1_fitted,
                "records": w1_records,
            }
            atomic_write_json(out_dir / "scale_laws_wave1_results.json", w1_payload)

        validate_scale_records(merged, expected_cells)
        fitted = fit_scale_laws(merged)
        out_json_name = "scale_laws_wave1_results.json" if args.wave == "1" else "scale_laws_results.json"
        phase_tag = "scale_laws_full_w1_w2" if waves_to_load == ["w1", "w2"] else f"scale_laws_wave_{args.wave}"
        payload = {
            "phase": phase_tag,
            "waves_merged": waves_to_load,
            "num_records": len(merged),
            "seed_start": seed_start,
            "approved_slots": [{"host": h, "gpu_id": g} for h, g in APPROVED_SLOTS],
            "cells": [asdict(c) for c in expected_cells],
            "fitted_laws": fitted,
            "records": merged,
        }
        atomic_write_json(out_dir / out_json_name, payload)
        write_scale_laws_summary_md(fitted, len(merged), out_dir / "scale_laws_summary.md")
        plots = generate_scale_law_plots(fitted, out_dir)
        print(
            f"[ScaleLaw-Merge] Validated {len(merged):,} records ({waves_to_load}) & {len(plots)} PNGs -> {out_dir / out_json_name}",
            flush=True,
        )
        return

    raise ValueError("Specify --smoke, --shard-idx, --dispatch-fleet, or --merge-only")


if __name__ == "__main__":
    main()
