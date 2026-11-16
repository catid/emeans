#!/usr/bin/env python3
"""Phase 2 Systems, Kernel & Hardware Optimization Benchmark Suite (`sys-opt`).

Executes paired interleaved AB/BA benchmarks with strict GPU synchronization and
zero train/test leakage across all 5 falsifiable hypotheses (H-Opt-1 .. H-Opt-5):
  - Suite 0: Empirical Blackwell Roofline (HBM GB/s, FP32 / BF16 / FP8 TFLOP/s).
  - Suite 1: H-Opt-1 Blackwell Split-K + Register-Resident FULL_D + TMA Assignment.
  - Suite 2: H-Opt-2 FP8 E4M3 Centered Power-of-Two Assignment & Training Parity.
  - Suite 3: H-Opt-3 Fused Triton EMA+Prepare & Fused Triton + CUDA-Graph k-means++.
  - Suite 4: H-Opt-4 Coarse-to-Fine IVF Assignment Speed-vs-MSE Pareto Frontier.
  - Suite 5: H-Opt-5 Contiguous/Fused Streaming & Multi-GPU Data-Sharded Scaling
             across n = 10^5 .. 10^7, k = 1024 .. 65536, d = 96 .. 1024.

Usage:
  Single-GPU suite:
    CUDA_VISIBLE_DEVICES=2 .venv/bin/python scripts/run_phase2_opt.py --mode single
  Multi-GPU scaling suite:
    CUDA_VISIBLE_DEVICES=2,3 .venv/bin/python scripts/run_phase2_opt.py --mode multigpu
  Full campaign (both):
    CUDA_VISIBLE_DEVICES=2,3 .venv/bin/python scripts/run_phase2_opt.py --mode all
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from emeans.emeans import EMeans
from emeans.fast_ops import (
    FLOAT8_E4M3FN,
    FP8Quantizer,
    FastEMeans,
    IVFAssigner,
    MultiGPUFastEMeans,
    blackwell_prepare_centroids,
    blackwell_triton_assign,
    fast_seed_emeans_plusplus,
    fp8_assign,
    triton_greedy_kmeans_plusplus,
)
from emeans.kernels import exact_mse_fp64, prepare_represented_centroids, triton_assign
from emeans.seeding import greedy_kmeans_plusplus_torch, seed_emeans_plusplus


OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "phase2_opt"


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON atomically via PID-unique temporary file + `os.replace`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".tmp_{os.getpid()}_{path.name}"
    try:
        tmp_path.write_text(json.dumps(payload, indent=2))
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def sync_all(devices: list[torch.device] | None = None) -> None:
    if devices is not None:
        for dv in devices:
            if dv.type == "cuda":
                torch.cuda.synchronize(dv)
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


def paired_abba_timer(
    fn_a: Callable[[], Any],
    fn_b: Callable[[], Any],
    n_warmup: int = 3,
    n_pairs: int = 6,
    devices: list[torch.device] | None = None,
) -> dict[str, Any]:
    """Run interleaved AB / BA paired warm measurements with strict CUDA sync."""
    for _ in range(n_warmup):
        fn_a()
        fn_b()
    sync_all(devices)

    times_a: list[float] = []
    times_b: list[float] = []

    for p in range(n_pairs):
        if p % 2 == 0:
            sync_all(devices)
            t0 = time.perf_counter()
            fn_a()
            sync_all(devices)
            times_a.append(time.perf_counter() - t0)

            sync_all(devices)
            t0 = time.perf_counter()
            fn_b()
            sync_all(devices)
            times_b.append(time.perf_counter() - t0)
        else:
            sync_all(devices)
            t0 = time.perf_counter()
            fn_b()
            sync_all(devices)
            times_b.append(time.perf_counter() - t0)

            sync_all(devices)
            t0 = time.perf_counter()
            fn_a()
            sync_all(devices)
            times_a.append(time.perf_counter() - t0)

    arr_a = np.array(times_a, dtype=np.float64)
    arr_b = np.array(times_b, dtype=np.float64)
    ratios = arr_a / np.maximum(arr_b, 1e-12)
    return {
        "mean_a_ms": float(arr_a.mean() * 1e3),
        "std_a_ms": float(arr_a.std(ddof=1) * 1e3) if len(arr_a) > 1 else 0.0,
        "median_a_ms": float(np.median(arr_a) * 1e3),
        "mean_b_ms": float(arr_b.mean() * 1e3),
        "std_b_ms": float(arr_b.std(ddof=1) * 1e3) if len(arr_b) > 1 else 0.0,
        "median_b_ms": float(np.median(arr_b) * 1e3),
        "speedup_mean": float(arr_a.mean() / max(arr_b.mean(), 1e-12)),
        "speedup_median": float(np.median(ratios)),
        "speedup_ci95": float(1.96 * ratios.std(ddof=1) / math.sqrt(len(ratios))) if len(ratios) > 1 else 0.0,
        "n_pairs": n_pairs,
    }


@torch.no_grad()
def make_clustered_dataset(
    n_train: int,
    n_test: int,
    k_true: int,
    d: int,
    cluster_std: float = 0.57735,
    center_scale: float = 0.81650,
    offset: float = 0.0,
    seed: int = 1234,
    device: torch.device = torch.device("cuda:0"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthesize realistic multi-scale Gaussian mixture on GPU without train/test leakage."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    centers = (
        torch.randn((k_true, d), generator=gen, device=device, dtype=torch.float32) * center_scale
        + offset
    )

    # Generate in chunks so even n_train = 10^7 stays fast and low-overhead in VRAM
    x_train = torch.empty((n_train, d), dtype=torch.float32, device=device)
    chunk = 524288
    for s in range(0, n_train, chunk):
        e = min(s + chunk, n_train)
        m = e - s
        idx = torch.randint(0, k_true, (m,), generator=gen, device=device)
        x_train[s:e] = centers[idx]
        noise = torch.randn((m, d), generator=gen, device=device, dtype=torch.float32)
        x_train[s:e].add_(noise, alpha=cluster_std)

    idx_te = torch.randint(0, k_true, (n_test,), generator=gen, device=device)
    noise_te = torch.randn((n_test, d), generator=gen, device=device, dtype=torch.float32) * cluster_std
    x_test = (centers[idx_te] + noise_te).contiguous()
    return x_train, x_test


@torch.no_grad()
def measure_roofline(dev: torch.device) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(dev)
    buf = torch.randn(256 * 1024 * 1024, device=dev, dtype=torch.float32)
    dst = torch.empty_like(buf)
    for _ in range(5):
        dst.copy_(buf)
    sync_all([dev])
    t0 = time.perf_counter()
    for _ in range(20):
        dst.copy_(buf)
    sync_all([dev])
    dt_bw = (time.perf_counter() - t0) / 20.0
    hbm_gbps = float(2 * buf.numel() * 4 / dt_bw / 1e9)
    del buf, dst

    M = N = K = 8192
    a32 = torch.randn((M, K), device=dev, dtype=torch.float32)
    b32 = torch.randn((N, K), device=dev, dtype=torch.float32)
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for _ in range(3):
            _ = a32 @ b32.t()
        sync_all([dev])
        t0 = time.perf_counter()
        for _ in range(10):
            _ = a32 @ b32.t()
        sync_all([dev])
        fp32_tflops = float(2 * M * N * K / ((time.perf_counter() - t0) / 10.0) / 1e12)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32
    del a32, b32

    a16 = torch.randn((M, K), device=dev, dtype=torch.bfloat16)
    b16 = torch.randn((N, K), device=dev, dtype=torch.float16).to(torch.bfloat16)
    for _ in range(5):
        _ = a16 @ b16.t()
    sync_all([dev])
    t0 = time.perf_counter()
    for _ in range(20):
        _ = a16 @ b16.t()
    sync_all([dev])
    bf16_tflops = float(2 * M * N * K / ((time.perf_counter() - t0) / 20.0) / 1e12)

    a8 = a16.to(FLOAT8_E4M3FN)
    b8 = b16.to(FLOAT8_E4M3FN)
    one = torch.ones((1,), device=dev, dtype=torch.float32)
    for _ in range(5):
        _ = torch._scaled_mm(a8, b8.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16)
    sync_all([dev])
    t0 = time.perf_counter()
    for _ in range(20):
        _ = torch._scaled_mm(a8, b8.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16)
    sync_all([dev])
    fp8_tflops = float(2 * M * N * K / ((time.perf_counter() - t0) / 20.0) / 1e12)
    del a16, b16, a8, b8
    torch.cuda.empty_cache()

    return {
        "gpu_name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "num_sms": props.multi_processor_count,
        "l2_cache_mb": round(props.L2_cache_size / (1024 * 1024), 1),
        "vram_gb": round(props.total_memory / (1024**3), 2),
        "empirical_hbm_gbps": round(hbm_gbps, 1),
        "empirical_fp32_ieee_tflops": round(fp32_tflops, 2),
        "empirical_bf16_tflops": round(bf16_tflops, 2),
        "empirical_fp8_e4m3_tflops": round(fp8_tflops, 2),
    }


@torch.no_grad()
def run_suite1_h_opt1(dev: torch.device, roofline: dict[str, Any]) -> list[dict[str, Any]]:
    """H-Opt-1: Baseline `kernels.triton_assign` vs `blackwell_triton_assign`."""
    configs = [
        # Small-B / Large-K Split-K + FULL_D regime
        {"tag": "SplitK+FullD (B=512, K=65536, D=128)", "B": 512, "K": 65536, "D": 128},
        {"tag": "SplitK+FullD (B=2048, K=65536, D=96)", "B": 2048, "K": 65536, "D": 96},
        {"tag": "SplitK+FullD (B=4096, K=16384, D=128)", "B": 4096, "K": 16384, "D": 128},
        {"tag": "SplitK+WideD (B=2048, K=32768, D=512)", "B": 2048, "K": 32768, "D": 512},
        # Large-B / FULL_D register-resident regime
        {"tag": "FullD-Large (B=65536, K=16384, D=96)", "B": 65536, "K": 16384, "D": 96},
        {"tag": "FullD-Large (B=65536, K=16384, D=128)", "B": 65536, "K": 16384, "D": 128},
        {"tag": "FullD-HugeK (B=32768, K=65536, D=128)", "B": 32768, "K": 65536, "D": 128},
        # Large-B / Large-D TMA 128x256 Tensor-Core regime
        {"tag": "TMA-WideD (B=65536, K=16384, D=256)", "B": 65536, "K": 16384, "D": 256},
        {"tag": "TMA-WideD (B=32768, K=16384, D=512)", "B": 32768, "K": 16384, "D": 512},
        {"tag": "TMA-WideD (B=32768, K=16384, D=1024)", "B": 32768, "K": 16384, "D": 1024},
    ]
    results = []
    gen = torch.Generator(device=dev).manual_seed(2026)
    for cfg in configs:
        B, K, D = cfg["B"], cfg["K"], cfg["D"]
        x_bf16 = torch.randn((B, D), generator=gen, device=dev, dtype=torch.float32).to(torch.bfloat16).contiguous()
        c_f32 = torch.randn((K, D), generator=gen, device=dev, dtype=torch.float32)
        c_rep, c_norm = prepare_represented_centroids(c_f32, torch.bfloat16)

        d_base, l_base = triton_assign(x_bf16, c_rep, c_norm)
        d_opt, l_opt = blackwell_triton_assign(x_bf16, c_rep, c_norm)
        disagree = float((l_base != l_opt).float().mean().item())

        stats = paired_abba_timer(
            lambda: triton_assign(x_bf16, c_rep, c_norm),
            lambda: blackwell_triton_assign(x_bf16, c_rep, c_norm),
            n_warmup=4,
            n_pairs=8,
            devices=[dev],
        )
        flops = 2.0 * B * K * D
        tflops_base = flops / (stats["mean_a_ms"] * 1e-3) / 1e12
        tflops_opt = flops / (stats["mean_b_ms"] * 1e-3) / 1e12
        row = {
            **cfg,
            **stats,
            "tflops_baseline": round(tflops_base, 2),
            "tflops_blackwell": round(tflops_opt, 2),
            "pct_bf16_roofline": round(100.0 * tflops_opt / roofline["empirical_bf16_tflops"], 1),
            "label_disagree_vs_base": round(disagree, 6),
        }
        print(
            f"[H-Opt-1] {cfg['tag']}: {stats['mean_a_ms']:.3f}ms -> {stats['mean_b_ms']:.3f}ms "
            f"({stats['speedup_mean']:.2f}x, {tflops_opt:.1f} TFLOP/s)"
        )
        results.append(row)
        del x_bf16, c_f32, c_rep, c_norm
    torch.cuda.empty_cache()
    return results


@torch.no_grad()
def run_suite2_h_opt2(dev: torch.device, roofline: dict[str, Any]) -> dict[str, Any]:
    """H-Opt-2: FP8 E4M3 Tensor-Core Assignment & Multi-D End-to-End Training Parity."""
    kernel_rows = []
    gen = torch.Generator(device=dev).manual_seed(3030)
    for D in (96, 128, 512, 1024):
        B, K = 65536, 16384
        x_f32 = torch.randn((B, D), generator=gen, device=dev, dtype=torch.float32) + 8.0
        c_f32 = torch.randn((K, D), generator=gen, device=dev, dtype=torch.float32) + 8.0

        c_bf16, cn_bf16 = blackwell_prepare_centroids(c_f32, torch.bfloat16)
        x_bf16 = x_f32.to(torch.bfloat16).contiguous()

        quant = FP8Quantizer.calibrate(x_f32, center_mode="auto", seed=42)
        c_fp8, cn_fp8 = quant.prepare_centroids(c_f32)
        x_fp8 = quant.encode_rows(x_f32)

        # Measure both pre-encoded FP8 kernel and dynamic encode+assign vs BF16 Blackwell kernel
        st_pre = paired_abba_timer(
            lambda: blackwell_triton_assign(x_bf16, c_bf16, cn_bf16),
            lambda: blackwell_triton_assign(x_fp8, c_fp8, cn_fp8, inv_scale_sq=quant.inv_scale_sq),
            n_warmup=4,
            n_pairs=8,
            devices=[dev],
        )
        st_dyn = paired_abba_timer(
            lambda: blackwell_triton_assign(x_bf16, c_bf16, cn_bf16),
            lambda: fp8_assign(x_f32, c_f32, quantizer=quant, c_fp8_cached=(c_fp8, cn_fp8)),
            n_warmup=3,
            n_pairs=6,
            devices=[dev],
        )

        # Mean-centered FP64 oracle disagreement check on first 4096 rows
        mu64 = x_f32[:4096].double().mean(dim=0, keepdim=True)
        xq64 = x_f32[:4096].double() - mu64
        c64 = c_f32.double() - mu64
        lbl_64 = (c64.square().sum(dim=1)[None, :] - 2.0 * (xq64 @ c64.t())).argmin(dim=1)
        _, lbl_bf16 = blackwell_triton_assign(x_bf16[:4096], c_bf16, cn_bf16)
        _, lbl_fp8 = blackwell_triton_assign(
            x_fp8[:4096], c_fp8, cn_fp8, inv_scale_sq=quant.inv_scale_sq
        )
        dis_bf16 = float((lbl_bf16 != lbl_64).float().mean().item()) * 100.0
        dis_fp8 = float((lbl_fp8 != lbl_64).float().mean().item()) * 100.0

        flops = 2.0 * B * K * D
        tf_bf16 = flops / (st_pre["mean_a_ms"] * 1e-3) / 1e12
        tf_fp8 = flops / (st_pre["mean_b_ms"] * 1e-3) / 1e12
        row = {
            "B": B,
            "K": K,
            "D": D,
            "bf16_ms": round(st_pre["mean_a_ms"], 3),
            "fp8_preencoded_ms": round(st_pre["mean_b_ms"], 3),
            "fp8_dynamic_ms": round(st_dyn["mean_b_ms"], 3),
            "speedup_preencoded": round(st_pre["speedup_mean"], 3),
            "speedup_dynamic": round(st_dyn["speedup_mean"], 3),
            "tflops_bf16": round(tf_bf16, 2),
            "tflops_fp8": round(tf_fp8, 2),
            "pct_fp8_roofline": round(100.0 * tf_fp8 / roofline["empirical_fp8_e4m3_tflops"], 1),
            "disagree_bf16_pct": round(dis_bf16, 3),
            "disagree_fp8_pct": round(dis_fp8, 3),
            "fp8_centered": quant.shift is not None,
            "fp8_scale_exp": quant.exponent,
        }
        print(
            f"[H-Opt-2 Kernel] D={D}: BF16 {row['bf16_ms']}ms ({tf_bf16:.1f} TF/s) -> "
            f"FP8 {row['fp8_preencoded_ms']}ms ({tf_fp8:.1f} TF/s, {row['speedup_preencoded']:.2f}x), "
            f"disagree FP8={dis_fp8:.2f}%"
        )
        kernel_rows.append(row)
        del x_f32, c_f32, c_bf16, cn_bf16, c_fp8, cn_fp8, x_bf16, x_fp8

    # End-to-end training parity across d = 96, 128, 512, 1024 (paired AB/BA over 3 seeds)
    train_rows = []
    for D in (96, 128, 512, 1024):
        n_tr = 250000 if D <= 512 else 150000
        k_cl = 2048
        x_tr, x_te = make_clustered_dataset(
            n_train=n_tr,
            n_test=15000,
            k_true=1024,
            d=D,
            cluster_std=0.57735,
            center_scale=0.81650,
            offset=25.0,  # Realistic positive mean shift to test centered FP8!
            seed=4000 + D,
            device=dev,
        )

        # Compare FastEMeans(precision='bf16') vs FastEMeans(precision='fp8') AND baseline EMeans
        mse_base_list, mse_bf16_list, mse_fp8_list, mse_fp8_uncen_list = [], [], [], []
        t_base_list, t_bf16_list, t_fp8_list = [], [], []

        # Explicit JIT warmup outside timer with identical (n_tr, k_cl, nepochs=4.0) specialization
        # so preencode_dataset, HAS_INDICES=True/False, ACCUM_PASS_SUMS=False/True, batch_size, and
        # the n_tr % batch_size remainder tile are 100% compiled before timing.
        _ = EMeans(d=D, k=k_cl, nepochs=4.0, seed=0).fit(x_tr, capture_assignments=False)
        _ = FastEMeans(d=D, k=k_cl, nepochs=4.0, precision="bf16", seed=0).fit(
            x_tr, capture_assignments=False
        )
        _ = FastEMeans(
            d=D, k=k_cl, nepochs=4.0, precision="fp8", fp8_center_mode="auto", seed=0
        ).fit(x_tr, capture_assignments=False)
        sync_all([dev])

        for s_idx, seed in enumerate(range(10, 16)):
            m_base = EMeans(d=D, k=k_cl, nepochs=4.0, seed=seed)
            m_bf16 = FastEMeans(d=D, k=k_cl, nepochs=4.0, precision="bf16", seed=seed)
            m_fp8 = FastEMeans(d=D, k=k_cl, nepochs=4.0, precision="fp8", fp8_center_mode="auto", seed=seed)
            m_fp8_u = FastEMeans(d=D, k=k_cl, nepochs=4.0, precision="fp8", fp8_center_mode="never", seed=seed)

            if s_idx % 2 == 0:
                sync_all([dev]); t0 = time.perf_counter()
                c_b, _ = m_base.fit(x_tr, capture_assignments=False)
                sync_all([dev]); t_base_list.append((time.perf_counter() - t0) * 1e3)

                sync_all([dev]); t0 = time.perf_counter()
                c_16, _ = m_bf16.fit(x_tr, capture_assignments=False)
                sync_all([dev]); t_bf16_list.append((time.perf_counter() - t0) * 1e3)

                sync_all([dev]); t0 = time.perf_counter()
                c_8, _ = m_fp8.fit(x_tr, capture_assignments=False)
                sync_all([dev]); t_fp8_list.append((time.perf_counter() - t0) * 1e3)
            else:
                sync_all([dev]); t0 = time.perf_counter()
                c_8, _ = m_fp8.fit(x_tr, capture_assignments=False)
                sync_all([dev]); t_fp8_list.append((time.perf_counter() - t0) * 1e3)

                sync_all([dev]); t0 = time.perf_counter()
                c_16, _ = m_bf16.fit(x_tr, capture_assignments=False)
                sync_all([dev]); t_bf16_list.append((time.perf_counter() - t0) * 1e3)

                sync_all([dev]); t0 = time.perf_counter()
                c_b, _ = m_base.fit(x_tr, capture_assignments=False)
                sync_all([dev]); t_base_list.append((time.perf_counter() - t0) * 1e3)

            c_8u, _ = m_fp8_u.fit(x_tr, capture_assignments=False)

            mse_base_list.append(exact_mse_fp64(x_te, c_b, device=dev))
            mse_bf16_list.append(exact_mse_fp64(x_te, c_16, device=dev))
            mse_fp8_list.append(exact_mse_fp64(x_te, c_8, device=dev))
            mse_fp8_uncen_list.append(exact_mse_fp64(x_te, c_8u, device=dev))

        mb = float(np.mean(mse_base_list))
        m16 = float(np.mean(mse_bf16_list))
        m8 = float(np.mean(mse_fp8_list))
        m8u = float(np.mean(mse_fp8_uncen_list))
        tb = float(np.mean(t_base_list))
        t16 = float(np.mean(t_bf16_list))
        t8 = float(np.mean(t_fp8_list))

        trow = {
            "n_train": n_tr,
            "k": k_cl,
            "d": D,
            "baseline_emeans_ms": round(tb, 2),
            "fast_bf16_ms": round(t16, 2),
            "fast_fp8_ms": round(t8, 2),
            "speedup_fp8_vs_baseline": round(tb / t8, 3),
            "speedup_fp8_vs_fast_bf16": round(t16 / t8, 3),
            "test_mse_baseline": round(mb, 6),
            "test_mse_fast_bf16": round(m16, 6),
            "test_mse_fast_fp8_centered": round(m8, 6),
            "test_mse_fast_fp8_uncentered": round(m8u, 6),
            "rel_mse_delta_fp8_vs_base_pct": round(100.0 * (m8 - mb) / mb, 4),
            "rel_mse_delta_uncentered_pct": round(100.0 * (m8u - mb) / mb, 4),
        }
        print(
            f"[H-Opt-2 Train] d={D}: Base {tb:.1f}ms -> FastBF16 {t16:.1f}ms -> FastFP8 {t8:.1f}ms "
            f"({tb / t8:.2f}x vs Base, {t16 / t8:.2f}x vs FastBF16), "
            f"MSE delta={trow['rel_mse_delta_fp8_vs_base_pct']:+.4f}% (uncentered={trow['rel_mse_delta_uncentered_pct']:+.4f}%)"
        )
        train_rows.append(trow)
        del x_tr, x_te
        torch.cuda.empty_cache()

    return {"kernel": kernel_rows, "training": train_rows}


@torch.no_grad()
def run_suite3_h_opt3(dev: torch.device) -> dict[str, Any]:
    """H-Opt-3: (a) Fused Triton Scatter + EMA + Prepare; (b) Fused Triton + CUDA-Graph k-means++."""
    # Part 3A: Isolate Fused EMA + Sync-Free loop speedup across batch regimes
    ema_rows = []
    for cfg in [
        {"tag": "Quality-Law SmallBatch (n=200k, k=1024, d=128, B=1024)", "n": 200000, "k": 1024, "d": 128, "B": 1024},
        {"tag": "Medium-Batch (n=500k, k=2048, d=128, B=4096)", "n": 500000, "k": 2048, "d": 128, "B": 4096},
        {"tag": "Throughput-Law LargeBatch (n=500k, k=4096, d=128, B=32768)", "n": 500000, "k": 4096, "d": 128, "B": 32768},
        {"tag": "High-D LargeK (n=250k, k=4096, d=512, B=4096)", "n": 250000, "k": 4096, "d": 512, "B": 4096},
    ]:
        x_tr, x_te = make_clustered_dataset(
            cfg["n"], 10000, k_true=256, d=cfg["d"], seed=5000 + cfg["B"], device=dev
        )
        # Compare Baseline EMeans vs Unfused-Blackwell vs Fused-Blackwell
        def _run_base():
            m = EMeans(d=cfg["d"], k=cfg["k"], nepochs=3.0, batch_size=cfg["B"], seed=42)
            c, _ = m.fit(x_tr, capture_assignments=False)
            return c

        def _run_unfused():
            m = FastEMeans(
                d=cfg["d"],
                k=cfg["k"],
                nepochs=3.0,
                batch_size=cfg["B"],
                use_blackwell_assign=True,
                use_fused_ema=False,
                streaming_mode="gather_copy",
                preencode_dataset=False,
                seed=42,
            )
            c, _ = m.fit(x_tr, capture_assignments=False)
            return c

        def _run_fused():
            m = FastEMeans(
                d=cfg["d"],
                k=cfg["k"],
                nepochs=3.0,
                batch_size=cfg["B"],
                use_blackwell_assign=True,
                use_fused_ema=True,
                streaming_mode="gather_copy",
                preencode_dataset=False,
                seed=42,
            )
            c, _ = m.fit(x_tr, capture_assignments=False)
            return c

        st_vs_unfused = paired_abba_timer(_run_unfused, _run_fused, n_warmup=1, n_pairs=4, devices=[dev])
        st_vs_base = paired_abba_timer(_run_base, _run_fused, n_warmup=1, n_pairs=4, devices=[dev])

        c_b = _run_base()
        c_f = _run_fused()
        mse_b = exact_mse_fp64(x_te, c_b, device=dev)
        mse_f = exact_mse_fp64(x_te, c_f, device=dev)

        erow = {
            **cfg,
            "baseline_emeans_ms": round(st_vs_base["mean_a_ms"], 2),
            "unfused_blackwell_ms": round(st_vs_unfused["mean_a_ms"], 2),
            "fused_ema_ms": round(st_vs_unfused["mean_b_ms"], 2),
            "speedup_vs_unfused": round(st_vs_unfused["speedup_mean"], 3),
            "speedup_vs_baseline": round(st_vs_base["speedup_mean"], 3),
            "test_mse_baseline": round(mse_b, 6),
            "test_mse_fused": round(mse_f, 6),
            "rel_mse_delta_pct": round(100.0 * (mse_f - mse_b) / mse_b, 4),
        }
        print(
            f"[H-Opt-3a] {cfg['tag']}: Base {erow['baseline_emeans_ms']}ms -> "
            f"Unfused {erow['unfused_blackwell_ms']}ms -> Fused {erow['fused_ema_ms']}ms "
            f"({erow['speedup_vs_unfused']:.2f}x from EMA fusion, {erow['speedup_vs_baseline']:.2f}x total)"
        )
        ema_rows.append(erow)
        del x_tr, x_te
        torch.cuda.empty_cache()

    # Part 3B: Greedy k-means++ Seeding Benchmark
    # De-confounds (i) pure FP32 Triton fusion, (ii) FP32 Triton + CUDA Graph, and (iii) BF16 TensorCore + CUDA Graph!
    kpp_rows = []
    for (m_sz, k_sz, d_sz) in [
        (16384, 512, 128),
        (32768, 1024, 96),
        (32768, 1024, 128),
        (65536, 2048, 128),
        (32768, 1024, 512),
    ]:
        x_coreset, x_te = make_clustered_dataset(
            m_sz, 8000, k_true=256, d=d_sz, seed=6000 + k_sz + d_sz, device=dev
        )
        w = torch.full((m_sz,), 1.0 / m_sz, dtype=torch.float32, device=dev)

        st_fp32_eager = paired_abba_timer(
            lambda: greedy_kmeans_plusplus_torch(x_coreset, k_sz, weights=w, seed=77),
            lambda: triton_greedy_kmeans_plusplus(
                x_coreset, k_sz, weights=w, use_bf16=False, use_cuda_graph=False, seed=77
            ),
            n_warmup=2,
            n_pairs=4,
            devices=[dev],
        )
        st_fp32_graph = paired_abba_timer(
            lambda: greedy_kmeans_plusplus_torch(x_coreset, k_sz, weights=w, seed=77),
            lambda: triton_greedy_kmeans_plusplus(
                x_coreset, k_sz, weights=w, use_bf16=False, use_cuda_graph=True, seed=77
            ),
            n_warmup=2,
            n_pairs=4,
            devices=[dev],
        )
        st_bf16_graph = paired_abba_timer(
            lambda: greedy_kmeans_plusplus_torch(x_coreset, k_sz, weights=w, seed=77),
            lambda: triton_greedy_kmeans_plusplus(
                x_coreset, k_sz, weights=w, use_bf16=True, use_cuda_graph=True, seed=77
            ),
            n_warmup=2,
            n_pairs=4,
            devices=[dev],
        )

        c_ref = greedy_kmeans_plusplus_torch(x_coreset, k_sz, weights=w, seed=77)
        c_fp32 = triton_greedy_kmeans_plusplus(
            x_coreset, k_sz, weights=w, use_bf16=False, use_cuda_graph=True, seed=77
        )
        c_bf16 = triton_greedy_kmeans_plusplus(
            x_coreset, k_sz, weights=w, use_bf16=True, use_cuda_graph=True, seed=77
        )
        mse_ref = exact_mse_fp64(x_te, c_ref, device=dev)
        mse_fp32 = exact_mse_fp64(x_te, c_fp32, device=dev)
        mse_bf16 = exact_mse_fp64(x_te, c_bf16, device=dev)

        # Also measure full end-to-end sub-epoch seeding pipeline (coreset + exact k-means++ + Voronoi polish)
        x_full, _ = make_clustered_dataset(
            max(200000, m_sz * 4), 1024, k_true=256, d=d_sz, seed=6500 + k_sz, device=dev
        )
        st_pipeline = paired_abba_timer(
            lambda: seed_emeans_plusplus(x_full, k_sz, seeding_mode="coreset_exact", seed=77),
            lambda: fast_seed_emeans_plusplus(
                x_full, k_sz, seeding_mode="coreset_exact", use_cuda_graph=True, seed=77
            ),
            n_warmup=1,
            n_pairs=4,
            devices=[dev],
        )
        del x_full

        krow = {
            "m_coreset": m_sz,
            "k": k_sz,
            "d": d_sz,
            "torch_fp32_kpp_ms": round(st_fp32_graph["mean_a_ms"], 2),
            "triton_fp32_eager_ms": round(st_fp32_eager["mean_b_ms"], 2),
            "triton_fp32_cudagraph_ms": round(st_fp32_graph["mean_b_ms"], 2),
            "triton_bf16_cudagraph_ms": round(st_bf16_graph["mean_b_ms"], 2),
            "speedup_fp32_eager": round(st_fp32_eager["speedup_mean"], 3),
            "speedup_fp32_cudagraph": round(st_fp32_graph["speedup_mean"], 3),
            "speedup_bf16_cudagraph": round(st_bf16_graph["speedup_mean"], 3),
            "full_seed_pipeline_ref_ms": round(st_pipeline["mean_a_ms"], 2),
            "full_seed_pipeline_fast_ms": round(st_pipeline["mean_b_ms"], 2),
            "full_seed_pipeline_speedup": round(st_pipeline["speedup_mean"], 3),
            "seed_mse_ref": round(mse_ref, 6),
            "seed_mse_fp32_opt": round(mse_fp32, 6),
            "seed_mse_bf16_opt": round(mse_bf16, 6),
            "rel_mse_fp32_delta_pct": round(100.0 * (mse_fp32 - mse_ref) / mse_ref, 4),
            "rel_mse_bf16_delta_pct": round(100.0 * (mse_bf16 - mse_ref) / mse_ref, 4),
        }
        print(
            f"[H-Opt-3b k-means++] (m={m_sz}, k={k_sz}, d={d_sz}): TorchFP32 {krow['torch_fp32_kpp_ms']}ms -> "
            f"TritonFP32 {krow['triton_fp32_eager_ms']}ms ({krow['speedup_fp32_eager']:.2f}x) -> "
            f"TritonFP32+Graph {krow['triton_fp32_cudagraph_ms']}ms ({krow['speedup_fp32_cudagraph']:.2f}x) -> "
            f"TritonBF16+Graph {krow['triton_bf16_cudagraph_ms']}ms ({krow['speedup_bf16_cudagraph']:.2f}x)"
        )
        kpp_rows.append(krow)
        del x_coreset, x_te
        torch.cuda.empty_cache()

    return {"ema_fusion": ema_rows, "kmeans_plusplus": kpp_rows}


@torch.no_grad()
def run_suite4_h_opt4(dev: torch.device) -> list[dict[str, Any]]:
    """H-Opt-4: Coarse-to-Fine IVF Assignment Speed-vs-MSE Pareto Frontier for Large k."""
    pareto_rows = []
    for (n_tr, k_cl, nlist) in [
        (500000, 16384, 128),
        (1000000, 65536, 256),
    ]:
        d = 128
        x_tr, x_te = make_clustered_dataset(
            n_tr, 20000, k_true=1024, d=d, cluster_std=0.57735, center_scale=0.81650, seed=7000 + k_cl, device=dev
        )

        # Explicit JIT warmup outside timer for Baseline EMeans, FastEMeans, and FastEMeans(IVF)
        _ = EMeans(d=d, k=1024, nepochs=2.0, seed=0).fit(x_tr[:32768], capture_assignments=False)
        _ = FastEMeans(d=d, k=1024, nepochs=2.0, seed=0).fit(x_tr[:32768], capture_assignments=False)
        _ = FastEMeans(
            d=d, k=1024, nepochs=2.0, assign_mode="ivf", ivf_nlist=32, ivf_nprobe=4, seed=0
        ).fit(x_tr[:32768], capture_assignments=False)
        sync_all([dev])

        st_exact = paired_abba_timer(
            lambda: EMeans(d=d, k=k_cl, nepochs=3.0, seed=42).fit(x_tr, capture_assignments=False),
            lambda: FastEMeans(d=d, k=k_cl, nepochs=3.0, precision="bf16", seed=42).fit(
                x_tr, capture_assignments=False
            ),
            n_warmup=1,
            n_pairs=2,
            devices=[dev],
        )
        base_ms = st_exact["mean_a_ms"]
        fex_ms = st_exact["mean_b_ms"]

        c_base, _ = EMeans(d=d, k=k_cl, nepochs=3.0, seed=42).fit(x_tr, capture_assignments=False)
        mse_base_te = exact_mse_fp64(x_te, c_base, device=dev)
        mse_base_tr = exact_mse_fp64(x_tr, c_base, device=dev)

        c_fex, _ = FastEMeans(d=d, k=k_cl, nepochs=3.0, precision="bf16", seed=42).fit(
            x_tr, capture_assignments=False
        )
        mse_fex_te = exact_mse_fp64(x_te, c_fex, device=dev)
        mse_fex_tr = exact_mse_fp64(x_tr, c_fex, device=dev)

        pareto_rows.append({
            "n_train": n_tr,
            "k": k_cl,
            "d": d,
            "method": "Baseline EMeans (Exact BF16)",
            "nlist": 0,
            "nprobe": k_cl,
            "rebuild_period": 0,
            "train_ms": round(base_ms, 2),
            "speedup_vs_baseline": 1.0,
            "speedup_vs_fast_exact": round(fex_ms / base_ms, 3),
            "train_mse_fp64": round(mse_base_tr, 6),
            "test_mse_fp64": round(mse_base_te, 6),
            "rel_test_mse_delta_pct": 0.0,
        })
        pareto_rows.append({
            "n_train": n_tr,
            "k": k_cl,
            "d": d,
            "method": "FastEMeans (Exact Blackwell BF16)",
            "nlist": 0,
            "nprobe": k_cl,
            "rebuild_period": 0,
            "train_ms": round(fex_ms, 2),
            "speedup_vs_baseline": round(base_ms / fex_ms, 3),
            "speedup_vs_fast_exact": 1.0,
            "train_mse_fp64": round(mse_fex_tr, 6),
            "test_mse_fp64": round(mse_fex_te, 6),
            "rel_test_mse_delta_pct": round(100.0 * (mse_fex_te - mse_base_te) / mse_base_te, 4),
        })

        for nprobe in (2, 4, 8, 16, 32):
            st_ivf = paired_abba_timer(
                lambda: FastEMeans(d=d, k=k_cl, nepochs=3.0, precision="bf16", seed=42).fit(
                    x_tr, capture_assignments=False
                ),
                lambda np_cur=nprobe: FastEMeans(
                    d=d,
                    k=k_cl,
                    nepochs=3.0,
                    precision="bf16",
                    assign_mode="ivf",
                    ivf_nlist=nlist,
                    ivf_nprobe=np_cur,
                    ivf_rebuild_period=4,
                    seed=42,
                ).fit(x_tr, capture_assignments=False),
                n_warmup=1,
                n_pairs=2,
                devices=[dev],
            )
            ivf_ms = st_ivf["mean_b_ms"]
            c_ivf, _ = FastEMeans(
                d=d,
                k=k_cl,
                nepochs=3.0,
                precision="bf16",
                assign_mode="ivf",
                ivf_nlist=nlist,
                ivf_nprobe=nprobe,
                ivf_rebuild_period=4,
                seed=42,
            ).fit(x_tr, capture_assignments=False)
            mse_te = exact_mse_fp64(x_te, c_ivf, device=dev)
            mse_tr = exact_mse_fp64(x_tr, c_ivf, device=dev)
            prow = {
                "n_train": n_tr,
                "k": k_cl,
                "d": d,
                "method": f"IVF{nlist}.np{nprobe} (p=4)",
                "nlist": nlist,
                "nprobe": nprobe,
                "rebuild_period": 4,
                "train_ms": round(ivf_ms, 2),
                "speedup_vs_baseline": round(base_ms / ivf_ms, 3),
                "speedup_vs_fast_exact": round(st_ivf["speedup_mean"], 3),
                "train_mse_fp64": round(mse_tr, 6),
                "test_mse_fp64": round(mse_te, 6),
                "rel_test_mse_delta_pct": round(100.0 * (mse_te - mse_base_te) / mse_base_te, 4),
            }
            print(
                f"[H-Opt-4] k={k_cl} {prow['method']}: {prow['train_ms']}ms "
                f"({prow['speedup_vs_baseline']:.2f}x vs Base, {prow['speedup_vs_fast_exact']:.2f}x vs FastExact), "
                f"Test MSE delta={prow['rel_test_mse_delta_pct']:+.4f}%"
            )
            pareto_rows.append(prow)

        del x_tr, x_te, c_base, c_fex
        gc.collect()
        torch.cuda.empty_cache()

    return pareto_rows


@torch.no_grad()
def run_suite5_single_streaming(dev: torch.device) -> list[dict[str, Any]]:
    """H-Opt-5a: End-to-End Dataset Scale Sweep (n = 10^5 .. 10^7) & Streaming Ablations."""
    rows = []
    for (n_tr, k_cl, d_sz) in [
        (100000, 1024, 128),
        (1000000, 4096, 128),
        (5000000, 8192, 128),
        (10000000, 16384, 128),
    ]:
        x_tr, x_te = make_clustered_dataset(
            n_tr, 20000, k_true=512, d=d_sz, seed=8000 + k_cl, device=dev
        )

        # Warmup
        _ = EMeans(d=d_sz, k=256, nepochs=2.0, seed=1).fit(x_tr[:16384], capture_assignments=False)
        _ = FastEMeans(d=d_sz, k=256, nepochs=2.0, precision="fp8", seed=1).fit(
            x_tr[:16384], capture_assignments=False
        )
        sync_all([dev])

        n_pairs = 3 if n_tr <= 1000000 else 2

        def _fit_base():
            m = EMeans(d=d_sz, k=k_cl, nepochs=4.0, seed=42)
            return m.fit(x_tr, capture_assignments=False)[0], m.train_time_sec

        def _fit_gather_copy():
            m = FastEMeans(
                d=d_sz,
                k=k_cl,
                nepochs=4.0,
                precision="bf16",
                streaming_mode="gather_copy",
                preencode_dataset=False,
                seed=42,
            )
            return m.fit(x_tr, capture_assignments=False)[0], m.train_time_sec

        def _fit_fused_stream():
            m = FastEMeans(
                d=d_sz,
                k=k_cl,
                nepochs=4.0,
                precision="bf16",
                streaming_mode="fused_gather",
                preencode_dataset=True,
                seed=42,
            )
            return m.fit(x_tr, capture_assignments=False)[0], m.train_time_sec

        def _fit_block_perm_fp8():
            m = FastEMeans(
                d=d_sz,
                k=k_cl,
                nepochs=4.0,
                precision="fp8",
                streaming_mode="block_permute",
                block_permute_size=64,
                preencode_dataset=True,
                seed=42,
            )
            return m.fit(x_tr, capture_assignments=False)[0], m.train_time_sec

        st_bf16 = paired_abba_timer(
            lambda: _fit_base(),
            lambda: _fit_fused_stream(),
            n_warmup=1,
            n_pairs=n_pairs,
            devices=[dev],
        )
        st_stream_iso = paired_abba_timer(
            lambda: _fit_gather_copy(),
            lambda: _fit_fused_stream(),
            n_warmup=1,
            n_pairs=n_pairs,
            devices=[dev],
        )
        st_fp8_stack = paired_abba_timer(
            lambda: _fit_base(),
            lambda: _fit_block_perm_fp8(),
            n_warmup=1,
            n_pairs=n_pairs,
            devices=[dev],
        )

        c_base, _ = _fit_base()
        c_bf16, _ = _fit_fused_stream()
        c_fp8, _ = _fit_block_perm_fp8()

        mse_base = exact_mse_fp64(x_te, c_base, device=dev)
        mse_bf16 = exact_mse_fp64(x_te, c_bf16, device=dev)
        mse_fp8 = exact_mse_fp64(x_te, c_fp8, device=dev)

        srow = {
            "n_train": n_tr,
            "k": k_cl,
            "d": d_sz,
            "baseline_ms": round(st_bf16["mean_a_ms"], 2),
            "gather_copy_ms": round(st_stream_iso["mean_a_ms"], 2),
            "fast_bf16_fused_stream_ms": round(st_bf16["mean_b_ms"], 2),
            "fast_fp8_block_stream_ms": round(st_fp8_stack["mean_b_ms"], 2),
            "streaming_isolated_speedup": round(st_stream_iso["speedup_mean"], 3),
            "speedup_fast_bf16_vs_base": round(st_bf16["speedup_mean"], 3),
            "speedup_fast_fp8_vs_base": round(st_fp8_stack["speedup_mean"], 3),
            "test_mse_base": round(mse_base, 6),
            "test_mse_fast_bf16": round(mse_bf16, 6),
            "test_mse_fast_fp8": round(mse_fp8, 6),
            "rel_mse_bf16_pct": round(100.0 * (mse_bf16 - mse_base) / mse_base, 4),
            "rel_mse_fp8_pct": round(100.0 * (mse_fp8 - mse_base) / mse_base, 4),
        }
        print(
            f"[H-Opt-5a Scale] n={n_tr:,}, k={k_cl}: Base {srow['baseline_ms']}ms -> "
            f"FastBF16 {srow['fast_bf16_fused_stream_ms']}ms ({srow['speedup_fast_bf16_vs_base']:.2f}x) -> "
            f"FastFP8 {srow['fast_fp8_block_stream_ms']}ms ({srow['speedup_fast_fp8_vs_base']:.2f}x)"
        )
        rows.append(srow)
        del x_tr, x_te, c_base, c_bf16, c_fp8
        gc.collect()
        torch.cuda.empty_cache()

    return rows


@torch.no_grad()
def run_suite5_multigpu() -> list[dict[str, Any]]:
    """H-Opt-5b: 1-GPU vs 2-GPU Data-Sharded NCCL All-Reduce Scaling (`CUDA_VISIBLE_DEVICES=2,3`)."""
    ndev = torch.cuda.device_count()
    if ndev < 2:
        print("[H-Opt-5b] Skipping 2-GPU benchmark because visible GPU count < 2")
        return []

    dev0 = torch.device("cuda:0")
    dev1 = torch.device("cuda:1")
    devs = [dev0, dev1]
    rows = []

    configs = [
        # Include small-scale regime to falsify/expose PCIe NCCL overhead when B*K*D is tiny!
        {"tag": "Small-CommBound (n=200k, k=1024, d=128)", "n": 200000, "k": 1024, "d": 128},
        {"tag": "Medium (n=2M, k=4096, d=128)", "n": 2000000, "k": 4096, "d": 128},
        {"tag": "Large-K (n=2M, k=16384, d=128)", "n": 2000000, "k": 16384, "d": 128},
        {"tag": "High-D (n=2M, k=8192, d=512)", "n": 2000000, "k": 8192, "d": 512},
        {"tag": "TenMillion (n=10M, k=16384, d=128)", "n": 10000000, "k": 16384, "d": 128},
        {"tag": "Huge-K (n=4M, k=65536, d=128)", "n": 4000000, "k": 65536, "d": 128},
    ]

    # Warmup NCCL communicator outside timer
    w_tr, _ = make_clustered_dataset(32768, 1024, 64, 128, seed=1, device=dev0)
    w_mg = MultiGPUFastEMeans(d=128, k=256, devices=devs, nepochs=2.0, seed=1)
    w_shards = w_mg.prepopulate_shards(w_tr)
    w_mg.fit(w_shards)
    del w_tr, w_shards
    sync_all(devs)

    for cfg in configs:
        n_tr, k_cl, d_sz = cfg["n"], cfg["k"], cfg["d"]
        x_tr, x_te = make_clustered_dataset(
            n_tr, 20000, k_true=512, d=d_sz, seed=9000 + k_cl, device=dev0
        )
        mg_helper = MultiGPUFastEMeans(d=d_sz, k=k_cl, devices=devs, nepochs=4.0, seed=42)
        shards_2gpu = mg_helper.prepopulate_shards(x_tr)
        shard_xfer_ms = mg_helper.shard_transfer_time_sec * 1e3

        def _run_1gpu_base():
            m = EMeans(d=d_sz, k=k_cl, nepochs=4.0, seed=42)
            c, _ = m.fit(x_tr, capture_assignments=False)
            return c

        def _run_1gpu_fast():
            m = FastEMeans(d=d_sz, k=k_cl, nepochs=4.0, precision="bf16", seed=42)
            c, _ = m.fit(x_tr, capture_assignments=False)
            return c

        def _run_2gpu_bf16():
            m = MultiGPUFastEMeans(d=d_sz, k=k_cl, devices=devs, nepochs=4.0, precision="bf16", seed=42)
            return m.fit(shards_2gpu)

        def _run_2gpu_fp8():
            m = MultiGPUFastEMeans(d=d_sz, k=k_cl, devices=devs, nepochs=4.0, precision="fp8", seed=42)
            return m.fit(shards_2gpu)

        n_pairs = 3 if n_tr <= 2000000 else 2
        st_1v2_bf16 = paired_abba_timer(_run_1gpu_fast, _run_2gpu_bf16, n_warmup=1, n_pairs=n_pairs, devices=devs)
        st_base_v_2fp8 = paired_abba_timer(_run_1gpu_base, _run_2gpu_fp8, n_warmup=1, n_pairs=n_pairs, devices=devs)

        c_base = _run_1gpu_base()
        c_1fast = _run_1gpu_fast()
        c_2bf16 = _run_2gpu_bf16()
        c_2fp8 = _run_2gpu_fp8()

        mse_base = exact_mse_fp64(x_te, c_base, device=dev0)
        mse_1fast = exact_mse_fp64(x_te, c_1fast, device=dev0)
        mse_2bf16 = exact_mse_fp64(x_te, c_2bf16, device=dev0)
        mse_2fp8 = exact_mse_fp64(x_te, c_2fp8, device=dev0)

        mrow = {
            **cfg,
            "shard_transfer_ms": round(shard_xfer_ms, 2),
            "base_1gpu_ms": round(st_base_v_2fp8["mean_a_ms"], 2),
            "fast_1gpu_bf16_ms": round(st_1v2_bf16["mean_a_ms"], 2),
            "fast_2gpu_bf16_ms": round(st_1v2_bf16["mean_b_ms"], 2),
            "fast_2gpu_bf16_incl_shard_xfer_ms": round(st_1v2_bf16["mean_b_ms"] + shard_xfer_ms, 2),
            "fast_2gpu_fp8_ms": round(st_base_v_2fp8["mean_b_ms"], 2),
            "fast_2gpu_fp8_incl_shard_xfer_ms": round(st_base_v_2fp8["mean_b_ms"] + shard_xfer_ms, 2),
            "scaling_speedup_2gpu_vs_1gpu_fast": round(st_1v2_bf16["speedup_mean"], 3),
            "scaling_speedup_incl_shard_xfer": round(
                st_1v2_bf16["mean_a_ms"] / max(st_1v2_bf16["mean_b_ms"] + shard_xfer_ms, 1e-9), 3
            ),
            "parallel_efficiency_pct": round(100.0 * st_1v2_bf16["speedup_mean"] / 2.0, 1),
            "total_speedup_2gpu_fp8_vs_base": round(st_base_v_2fp8["speedup_mean"], 3),
            "test_mse_base": round(mse_base, 6),
            "test_mse_1gpu_fast": round(mse_1fast, 6),
            "test_mse_2gpu_bf16": round(mse_2bf16, 6),
            "test_mse_2gpu_fp8": round(mse_2fp8, 6),
            "rel_mse_2gpu_bf16_pct": round(100.0 * (mse_2bf16 - mse_base) / mse_base, 4),
            "rel_mse_2gpu_fp8_pct": round(100.0 * (mse_2fp8 - mse_base) / mse_base, 4),
        }
        print(
            f"[H-Opt-5b 2-GPU] {cfg['tag']}: Base1G {mrow['base_1gpu_ms']}ms -> "
            f"Fast1G {mrow['fast_1gpu_bf16_ms']}ms -> Fast2G-BF16 {mrow['fast_2gpu_bf16_ms']}ms "
            f"({mrow['scaling_speedup_2gpu_vs_1gpu_fast']:.2f}x, {mrow['parallel_efficiency_pct']:.1f}% eff) -> "
            f"Fast2G-FP8 {mrow['fast_2gpu_fp8_ms']}ms ({mrow['total_speedup_2gpu_fp8_vs_base']:.2f}x total)"
        )
        rows.append(mrow)
        del x_tr, x_te, shards_2gpu, c_base, c_1fast, c_2bf16, c_2fp8
        gc.collect()
        for dv in devs:
            with torch.cuda.device(dv):
                torch.cuda.empty_cache()

    return rows


def generate_plots(results: dict[str, Any], out_dir: Path, tag: str = "local_foureyes") -> None:
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})
    suffix = "" if tag == "local_foureyes" else f"_{tag}"

    # Plot 1: H-Opt-1 & H-Opt-2 Kernel TFLOP/s vs Roofline
    if "h_opt1" in results and "h_opt2" in results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
        h1 = results["h_opt1"]
        labels1 = [r["tag"].split(" (")[0] + "\n" + r["tag"].split("(")[1].rstrip(")") for r in h1[:6]]
        x_pos = np.arange(len(labels1))
        w = 0.36
        ax1.bar(x_pos - w / 2, [r["tflops_baseline"] for r in h1[:6]], w, label="Baseline Triton (64x128)", color="#7f8c8d")
        ax1.bar(x_pos + w / 2, [r["tflops_blackwell"] for r in h1[:6]], w, label="H-Opt-1 Blackwell (SplitK+FullD)", color="#2980b9")
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(labels1, fontsize=8)
        ax1.set_ylabel("Effective TFLOP/s")
        ax1.set_title("H-Opt-1: Assignment TFLOP/s Across Regimes")
        ax1.legend()

        h2k = results["h_opt2"]["kernel"]
        dims = [f"D={r['D']}" for r in h2k]
        x2 = np.arange(len(dims))
        ax2.bar(x2 - w / 2, [r["tflops_bf16"] for r in h2k], w, label="Blackwell BF16", color="#2980b9")
        ax2.bar(x2 + w / 2, [r["tflops_fp8"] for r in h2k], w, label="H-Opt-2 Blackwell FP8 E4M3", color="#e67e22")
        ax2.axhline(results["roofline"]["empirical_bf16_tflops"], linestyle="--", color="#2980b9", alpha=0.6, label="BF16 GEMM Peak")
        ax2.axhline(results["roofline"]["empirical_fp8_e4m3_tflops"], linestyle="--", color="#e67e22", alpha=0.6, label="FP8 GEMM Peak")
        ax2.set_xticks(x2)
        ax2.set_xticklabels(dims)
        ax2.set_ylabel("Effective TFLOP/s (B=65536, K=16384)")
        ax2.set_title("H-Opt-2: Native FP8 E4M3 vs BF16 Assignment")
        ax2.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"h_opt1_opt2_tflops{suffix}.png", dpi=180)
        plt.close(fig)

    # Plot 2: H-Opt-4 IVF Speed vs Test MSE Pareto Frontier
    if "h_opt4" in results and results["h_opt4"]:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, k_target in zip(axes, [16384, 65536]):
            sub = [r for r in results["h_opt4"] if r["k"] == k_target]
            ivf_pts = [r for r in sub if r["nlist"] > 0]
            base_pt = next(r for r in sub if "Baseline" in r["method"])
            fex_pt = next(r for r in sub if "FastEMeans" in r["method"])

            ax.scatter([base_pt["train_ms"] / 1e3], [base_pt["rel_test_mse_delta_pct"]], s=100, marker="s", color="#c0392b", label="Baseline EMeans (Exact)")
            ax.scatter([fex_pt["train_ms"] / 1e3], [fex_pt["rel_test_mse_delta_pct"]], s=100, marker="^", color="#2980b9", label="FastEMeans (Exact)")
            ax.plot(
                [r["train_ms"] / 1e3 for r in ivf_pts],
                [r["rel_test_mse_delta_pct"] for r in ivf_pts],
                "o-",
                color="#27ae60",
                linewidth=2,
                label=f"H-Opt-4 IVF{ivf_pts[0]['nlist']} (nprobe=2..32)",
            )
            for r in ivf_pts:
                ax.annotate(f"np={r['nprobe']}", (r["train_ms"] / 1e3, r["rel_test_mse_delta_pct"]), textcoords="offset points", xytext=(4, 4), fontsize=8)
            ax.axhline(0.15, linestyle="--", color="gray", label="+0.15% MSE Parity Bound")
            ax.axhline(-0.15, linestyle="--", color="gray")
            ax.set_xlabel("Training Wall-Clock Time (seconds)")
            ax.set_ylabel("Relative Held-Out FP64 Test MSE Delta (%)")
            ax.set_title(f"H-Opt-4 Pareto Frontier (n={sub[0]['n_train']:,}, k={k_target:,})")
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"h_opt4_ivf_pareto{suffix}.png", dpi=180)
        plt.close(fig)

    # Plot 3: End-to-End Dataset Scale & Multi-GPU Speedup
    if "h_opt5_single" in results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5))
        s5 = results["h_opt5_single"]
        x_lbl = [f"n={r['n_train']//1000}k\nk={r['k']}" for r in s5]
        xp = np.arange(len(x_lbl))
        ax1.plot(xp, [r["speedup_fast_bf16_vs_base"] for r in s5], "o-", linewidth=2, color="#2980b9", label="FastEMeans BF16 (1 GPU)")
        ax1.plot(xp, [r["speedup_fast_fp8_vs_base"] for r in s5], "s-", linewidth=2, color="#e67e22", label="FastEMeans FP8 + BlockStream (1 GPU)")
        ax1.axhline(1.0, linestyle="--", color="gray")
        ax1.set_xticks(xp)
        ax1.set_xticklabels(x_lbl)
        ax1.set_ylabel("Speedup over Baseline EMeans (x)")
        ax1.set_title("Single-GPU End-to-End Speedup (n = 10^5 .. 10^7)")
        ax1.legend()

        if results.get("h_opt5_multigpu"):
            m5 = results["h_opt5_multigpu"]
            mlbl = [r["tag"].split(" (")[0] for r in m5]
            xm = np.arange(len(mlbl))
            w = 0.35
            ax2.bar(xm - w / 2, [r["scaling_speedup_2gpu_vs_1gpu_fast"] for r in m5], w, color="#16a085", label="2-GPU vs 1-GPU FastEMeans (BF16)")
            ax2.bar(xm + w / 2, [r["total_speedup_2gpu_fp8_vs_base"] for r in m5], w, color="#8e44ad", label="2-GPU FastFP8 vs 1-GPU Baseline")
            ax2.axhline(2.0, linestyle="--", color="#16a085", alpha=0.6, label="Ideal 2.0x Linear Scaling")
            ax2.axhline(1.0, linestyle=":", color="gray")
            ax2.set_xticks(xm)
            ax2.set_xticklabels(mlbl, rotation=18, ha="right", fontsize=8)
            ax2.set_ylabel("Speedup Factor (x)")
            ax2.set_title("H-Opt-5b: 2x Blackwell Data-Sharded NCCL Scaling")
            ax2.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"h_opt5_scaling{suffix}.png", dpi=180)
        plt.close(fig)


@torch.no_grad()
def run_smoke_check(dev: torch.device, out_dir: Path) -> dict[str, Any]:
    """Fast 5-second end-to-end harness & JSON schema sanity check (writes only to smoke/)."""
    smoke_dir = out_dir / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    x_tr, x_te = make_clustered_dataset(8192, 1024, k_true=32, d=128, seed=123, device=dev)

    st = paired_abba_timer(
        lambda: EMeans(d=128, k=64, nepochs=2.0, seed=7).fit(x_tr, capture_assignments=False),
        lambda: FastEMeans(d=128, k=64, nepochs=2.0, precision="fp8", seed=7).fit(
            x_tr, capture_assignments=False
        ),
        n_warmup=1,
        n_pairs=2,
        devices=[dev],
    )
    c_b, _ = EMeans(d=128, k=64, nepochs=2.0, seed=7).fit(x_tr, capture_assignments=False)
    c_f, _ = FastEMeans(d=128, k=64, nepochs=2.0, precision="fp8", seed=7).fit(
        x_tr, capture_assignments=False
    )
    c_ivf, _ = FastEMeans(
        d=128, k=64, nepochs=2.0, assign_mode="ivf", ivf_nlist=16, ivf_nprobe=4, seed=7
    ).fit(x_tr, capture_assignments=False)
    s_kpp = fast_seed_emeans_plusplus(x_tr, k=32, seed=7)

    payload = {
        "smoke_abba": st,
        "mse_base": exact_mse_fp64(x_te, c_b, device=dev),
        "mse_fp8": exact_mse_fp64(x_te, c_f, device=dev),
        "mse_ivf": exact_mse_fp64(x_te, c_ivf, device=dev),
        "mse_kpp_seed": exact_mse_fp64(x_te, s_kpp.centroids, device=dev),
    }
    assert all(math.isfinite(float(v)) for k, v in payload.items() if k.startswith("mse_"))
    _atomic_write_json(smoke_dir / "smoke_results.json", payload)
    print(f"[Smoke OK] {json.dumps(payload, indent=2)}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 2 Systems Optimization Benchmarks")
    parser.add_argument("--mode", choices=["single", "multigpu", "all"], default="all")
    parser.add_argument(
        "--suites",
        type=str,
        default="",
        help="Optional comma-separated subset of: h_opt1,h_opt2,h_opt3,h_opt4,h_opt5_single,h_opt5_multigpu",
    )
    parser.add_argument("--tag", type=str, default="local_foureyes")
    parser.add_argument("--shard-idx", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.num_shards < 1 or not (0 <= args.shard_idx < args.num_shards):
        raise ValueError(f"Invalid shard {args.shard_idx}/{args.num_shards}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev0 = torch.device("cuda:0")

    if args.smoke:
        run_smoke_check(dev0, OUT_DIR)
        return

    merged_path = OUT_DIR / f"phase2_opt_results_{args.tag}.json"
    if args.merge_only:
        merged: dict[str, Any] = {}
        pattern = f"phase2_opt_results_{args.tag}_shard_*_of_{args.num_shards:02d}.json"
        shard_files = sorted(OUT_DIR.glob(pattern))
        if not shard_files:
            raise FileNotFoundError(f"No shard files matching {pattern} in {OUT_DIR}")
        for sf in shard_files:
            part = json.loads(sf.read_text())
            for k, v in part.items():
                if k == "roofline" and "roofline" in merged:
                    continue
                merged[k] = v
        _atomic_write_json(merged_path, merged)
        generate_plots(merged, OUT_DIR, tag=args.tag)
        print(f"[Merged] {len(shard_files)} shards -> {merged_path}")
        return

    out_path = (
        OUT_DIR / f"phase2_opt_results_{args.tag}_shard_{args.shard_idx:02d}_of_{args.num_shards:02d}.json"
        if args.num_shards > 1
        else merged_path
    )
    results: dict[str, Any] = {}
    if out_path.exists():
        try:
            results = json.loads(out_path.read_text())
        except Exception:
            results = {}

    if args.suites.strip():
        requested = [s.strip() for s in args.suites.split(",") if s.strip()]
    else:
        requested = []
        if args.mode in ("single", "all"):
            requested.extend(["h_opt1", "h_opt2", "h_opt3", "h_opt4", "h_opt5_single"])
        if args.mode in ("multigpu", "all"):
            requested.append("h_opt5_multigpu")

    active_suites = [
        s for idx, s in enumerate(requested) if (idx % args.num_shards) == args.shard_idx
    ]
    print(f"[Runner] tag={args.tag} shard={args.shard_idx}/{args.num_shards} suites={active_suites}")

    if "roofline" not in results or any(s in active_suites for s in ("h_opt1", "h_opt2")):
        results["roofline"] = measure_roofline(dev0)
    print(f"[Roofline] {json.dumps(results['roofline'], indent=2)}")

    suite_runners: dict[str, Callable[[], Any]] = {
        "h_opt1": lambda: run_suite1_h_opt1(dev0, results["roofline"]),
        "h_opt2": lambda: run_suite2_h_opt2(dev0, results["roofline"]),
        "h_opt3": lambda: run_suite3_h_opt3(dev0),
        "h_opt4": lambda: run_suite4_h_opt4(dev0),
        "h_opt5_single": lambda: run_suite5_single_streaming(dev0),
        "h_opt5_multigpu": lambda: run_suite5_multigpu(),
    }

    for s_name in active_suites:
        if s_name not in suite_runners:
            raise ValueError(f"Unknown suite name: {s_name!r}")
        results[s_name] = suite_runners[s_name]()
        _atomic_write_json(out_path, results)

    if args.num_shards == 1:
        generate_plots(results, OUT_DIR, tag=args.tag)
    print(f"[Done] Saved Phase 2 optimization artifacts to {out_path}")


if __name__ == "__main__":
    main()
