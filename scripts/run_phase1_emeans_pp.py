#!/usr/bin/env python3
"""Phase 1 Experimental Campaign: Reproducing e-means & Proving e-means++.

Produces rigorous multi-seed JSON results and publication-quality matplotlib plots
in `outputs/phase1/`:
  1. `exp1_repro_emeans`:
     - Epochs-to-Lloyd@25 & warm GPU wall-clock speedup (>10x) on real & synthetic data.
     - Paper Table 1 component ablation (`our recipe`, validation-tuned `no cosine schedule`,
       `no revival`, `revival by resampling`, `no finalization`, `slow EMA`, `naive k-means++`).
  2. `exp2_pareto_tradeoff`:
     - Warm wall-clock time (charging BOTH seeding + training!) vs exact FP64 train & held-out
       test MSE across `Lloyd (random)`, `Lloyd (greedy k-means++)`, validation-tuned
       `Sculley MiniBatch`, `Vanilla e-means`, `Naive k-means++ + e-means`, and `e-means++`.
  3. `exp3_appendix_d2_suite`:
     - Full benchmark across all 7 synthetic distributions of Appendix D.2 of arXiv:2611.09501
       plus `multiscale_rare_outliers` at T in {1, 3, 5}.
  4. `exp4_mechanistic_traps`:
     - (a) Poisson 36.8% Coupon-Collector Empty-Mode Trap on `balanced_gmm` (mode coverage + MSE).
     - (b) Revival Assassination Trap on `multiscale_rare_outliers` & `zipf_gmm` (rare-mode
           survival rate + MSE across epochs).
  5. `exp5_emeans_pp_ablation`:
     - Pillar-by-pillar ablation of `e-means++` (coreset acceleration, Voronoi polish,
       prior-calibrated EMA mass, Huygens rare-mode protection, D^2 donor splitting).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats as sp_stats
from tabulate import tabulate

from emeans import (
    DatasetSplit,
    EMeans,
    EMeansPP,
    LloydKMeans,
    NaiveKMeansPP_EMeans,
    SculleyMiniBatchKMeans,
    build_recipe,
    exact_mse_fp64,
    load_cifar10_resnet18_embeddings,
    make_synthetic_dataset,
    seed_emeans_plusplus,
)

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "phase1"
ALL_EXPS = ("exp1_repro", "exp2_pareto", "exp3_suite", "exp4_mechanistic", "exp5_ablation")


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def warmup_gpu(dev: torch.device) -> None:
    """Run discarded warmup calls across all Triton/CUDA specializations before any timed measurements."""
    if dev.type != "cuda":
        return
    for d_w in (64, 128, 512):
        x_w = torch.randn(32_768, d_w, device=dev, dtype=torch.float32)
        # Warm exact coreset (k <= 256), block_p=256 (k=300), and block_p=1024 + m > 16384 BF16 (k=1024)
        for k_w in (64, 300, 1024):
            _ = EMeansPP(d=d_w, k=k_w, nepochs=1.0, seed=0).fit(x_w)
        _ = EMeans(d=d_w, k=1024, nepochs=1.0, seed=0).fit(x_w)
        _ = LloydKMeans(d=d_w, k=256, niter=2, init="random", seed=0).fit(x_w[:8192])
        _ = LloydKMeans(d=d_w, k=256, niter=2, init="k-means++", seed=0).fit(x_w[:8192])
        _ = SculleyMiniBatchKMeans(d=d_w, k=256, nepochs=1.0, batch_size=1024, seed=0).fit(x_w[:8192])
        _ = exact_mse_fp64(x_w[:1024], x_w[:256], device=dev)
    _sync(dev)


def eval_centroids(
    ds: DatasetSplit,
    centroids: torch.Tensor | np.ndarray,
    dev: torch.device,
    train_eval_cap: int = 100_000,
) -> dict[str, float]:
    """Compute exact double-precision train (fixed prefix subset <= 100K, matching paper) & test MSE."""
    tr_sub = ds.train[: min(len(ds.train), train_eval_cap)]
    train_mse = exact_mse_fp64(tr_sub, centroids, device=dev)
    test_mse = exact_mse_fp64(ds.test, centroids, device=dev)
    return {"train_mse": float(train_mse), "test_mse": float(test_mse)}


def mode_recovery_rate(
    true_centers: np.ndarray,
    learned_centroids: torch.Tensor | np.ndarray,
    sigma_w: float,
) -> float:
    """Fraction of ground-truth modes that have a learned centroid within 1.5 * sqrt(d) * sigma_w."""
    c_np = (
        learned_centroids.detach().cpu().numpy()
        if isinstance(learned_centroids, torch.Tensor)
        else np.asarray(learned_centroids)
    )
    d = true_centers.shape[1]
    # Squared distance threshold: 0.5 * inter-mode typical spacing (well inside Voronoi cell)
    thresh_d2 = (1.15 * math.sqrt(float(d)) * sigma_w) ** 2
    tc_t = torch.from_numpy(true_centers).double()
    lc_t = torch.from_numpy(c_np).double()
    origin = tc_t.mean(dim=0, keepdim=True)
    tc_c = tc_t - origin
    lc_c = lc_t - origin
    d2 = (
        tc_c.square().sum(dim=1, keepdim=True)
        + lc_c.square().sum(dim=1).unsqueeze(0)
        - 2.0 * (tc_c @ lc_c.t())
    ).clamp_min_(0.0)
    min_d2 = d2.min(dim=1).values.numpy()
    return float((min_d2 <= thresh_d2).mean())


def tune_sculley_on_val(
    ds: DatasetSplit,
    k: int,
    nepochs: float,
    dev: torch.device,
) -> tuple[int, float]:
    """Tune Sculley (batch_size, reassignment_ratio) strictly on the VALIDATION split (Appendix C)."""
    x_tr = torch.from_numpy(ds.train).to(dev)
    best_val = float("inf")
    best_cfg = (4096, 0.01)
    for bsz in (1024, 4096, 16384):
        if bsz > len(ds.train):
            continue
        for rr in (0.0, 0.01, 0.05):
            m = SculleyMiniBatchKMeans(
                d=ds.d, k=k, nepochs=nepochs, batch_size=bsz, reassignment_ratio=rr, seed=999
            )
            c, _ = m.fit(x_tr)
            v_mse = exact_mse_fp64(ds.val, c, device=dev)
            if v_mse < best_val:
                best_val = v_mse
                best_cfg = (bsz, rr)
    return best_cfg


def tune_constant_alpha_on_val(
    ds: DatasetSplit,
    k: int,
    nepochs: float,
    dev: torch.device,
) -> float:
    """Tune constant step alpha on validation split for fair 'no cosine schedule' ablation (Appendix K.3)."""
    x_tr = torch.from_numpy(ds.train).to(dev)
    rec = build_recipe(n=len(ds.train), d=ds.d, k=k, nepochs=nepochs)
    best_val = float("inf")
    best_a = 0.5 * rec.alpha0
    for mult in (0.125, 0.25, 0.35, 0.5, 0.7, 1.0):
        cand_a = min(1.0, max(1e-4, rec.alpha0 * mult))
        m = EMeans(
            d=ds.d, k=k, nepochs=nepochs, alpha0=cand_a, use_cosine_schedule=False, seed=999
        )
        c, _ = m.fit(x_tr)
        v_mse = exact_mse_fp64(ds.val, c, device=dev)
        if v_mse < best_val:
            best_val = v_mse
            best_a = cand_a
    return best_a


def summarize_seeds(rows: list[dict[str, float]], keys: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {"n_seeds": len(rows)}
    for key in keys:
        vals = np.array([r[key] for r in rows], dtype=np.float64)
        out[f"{key}_mean"] = float(vals.mean())
        out[f"{key}_median"] = float(np.median(vals))
        out[f"{key}_std"] = float(vals.std(ddof=1) if len(vals) > 1 else 0.0)
        out[f"{key}_per_seed"] = [float(v) for v in vals]
    return out


# ===========================================================================
# Experiment 1: Reproduce Meta FAIR's e-means (>10x Speedup & Table 1 Ablation)
# ===========================================================================
def run_exp1_repro_emeans(dev: torch.device, seeds: list[int], smoke: bool = False) -> dict[str, Any]:
    print("\n=== [Exp 1] Reproducing e-means: Epochs-to-Lloyd@25 & Component Ablation ===")
    # Use paper's k ~ 8*sqrt(N)..16*sqrt(N) compute-bound regime (k=4096 at N=200K..300K)
    n_syn = 40_000 if smoke else 300_000
    k_syn = 512 if smoke else 4096

    datasets: list[tuple[DatasetSplit, int]] = [
        (load_cifar10_resnet18_embeddings("patches_128"), 512 if smoke else 4096),
        (load_cifar10_resnet18_embeddings("global_512"), 512 if smoke else 2048),
        (
            make_synthetic_dataset(
                "isotropic_gaussian", n_train=n_syn, n_val=10_000, n_test=10_000, d=128, k=k_syn, seed=101
            ),
            k_syn,
        ),
        (
            make_synthetic_dataset(
                "stretched_gaussian_1", n_train=n_syn, n_val=10_000, n_test=10_000, d=128, k=k_syn, seed=102
            ),
            k_syn,
        ),
    ]
    if smoke:
        datasets = datasets[:2]

    epoch_ladder = [0.6, 1.0, 1.6, 2.0, 2.6, 3.0, 5.0] if not smoke else [1.0, 2.0, 5.0]
    speedup_results: dict[str, Any] = {}

    for ds, k in datasets:
        x_tr = torch.from_numpy(ds.train).to(dev)
        _ = LloydKMeans(d=ds.d, k=k, niter=2, init="random", seed=0).fit(x_tr)
        _ = EMeans(d=ds.d, k=k, nepochs=1.0, seed=0).fit(x_tr)
        _ = EMeansPP(d=ds.d, k=k, nepochs=1.0, seed=0).fit(x_tr)
        _sync(dev)

        lloyd_rows = []
        for s in seeds:
            ll = LloydKMeans(d=ds.d, k=k, niter=25, init="random", seed=s)
            c_ll, _ = ll.fit(x_tr)
            metrics = eval_centroids(ds, c_ll, dev)
            metrics["total_time_sec"] = ll.total_time_sec
            lloyd_rows.append(metrics)
        lloyd_sum = summarize_seeds(lloyd_rows, ("train_mse", "test_mse", "total_time_sec"))

        ladder_sum: dict[str, Any] = {}
        crossing_epoch = None
        crossing_speedup = None

        for T in epoch_ladder:
            em_rows = []
            for s in seeds:
                em = EMeans(d=ds.d, k=k, nepochs=T, seed=s)
                c_em, _ = em.fit(x_tr)
                m = eval_centroids(ds, c_em, dev)
                m["total_time_sec"] = em.total_time_sec
                em_rows.append(m)
            s_em = summarize_seeds(em_rows, ("train_mse", "test_mse", "total_time_sec"))
            s_em["rel_test_vs_lloyd25_pct"] = (
                (s_em["test_mse_median"] - lloyd_sum["test_mse_median"])
                / lloyd_sum["test_mse_median"]
                * 100.0
            )
            s_em["rel_train_vs_lloyd25_pct"] = (
                (s_em["train_mse_median"] - lloyd_sum["train_mse_median"])
                / lloyd_sum["train_mse_median"]
                * 100.0
            )
            paired_speedups = [
                lloyd_rows[i]["total_time_sec"] / max(1e-9, em_rows[i]["total_time_sec"])
                for i in range(len(seeds))
            ]
            s_em["speedup_vs_lloyd25_median"] = float(np.median(paired_speedups))
            ladder_sum[str(T)] = s_em

            if crossing_epoch is None and s_em["test_mse_median"] <= lloyd_sum["test_mse_median"]:
                crossing_epoch = T
                crossing_speedup = s_em["speedup_vs_lloyd25_median"]

        print(
            f"  {ds.name} (N={len(ds.train)}, d={ds.d}, k={k}): "
            f"Lloyd@25 test={lloyd_sum['test_mse_median']:.5f} ({lloyd_sum['total_time_sec_median']*1e3:.1f}ms) | "
            f"crossing T*={crossing_epoch} ({crossing_speedup:.1f}x speedup) | "
            f"e-means@5 rel_test={ladder_sum[str(epoch_ladder[-1])]['rel_test_vs_lloyd25_pct']:+.3f}%"
        )
        speedup_results[ds.name] = {
            "n_train": len(ds.train),
            "d": ds.d,
            "k": k,
            "lloyd25": lloyd_sum,
            "emeans_ladder": ladder_sum,
            "crossing_epoch_test": crossing_epoch,
            "crossing_speedup_median": crossing_speedup,
        }

    # Component ablation (Table 1 of paper: T = 3 epochs on CIFAR-10 ResNet18 Patches d=128)
    abl_ds, abl_k = datasets[0]
    x_abl = torch.from_numpy(abl_ds.train).to(dev)
    T_abl = 3.0
    const_alpha = tune_constant_alpha_on_val(abl_ds, abl_k, T_abl, dev)
    _ = EMeansPP(d=abl_ds.d, k=abl_k, nepochs=1.0, seed=0).fit(x_abl)
    _ = NaiveKMeansPP_EMeans(d=abl_ds.d, k=abl_k, nepochs=1.0, seed=0).fit(x_abl[: min(len(abl_ds.train), max(abl_k * 2, 8192))])
    _sync(dev)

    ablation_arms = [
        ("our_recipe", lambda s: EMeans(d=abl_ds.d, k=abl_k, nepochs=T_abl, seed=s)),
        (
            "no_cosine_schedule_tuned",
            lambda s: EMeans(
                d=abl_ds.d, k=abl_k, nepochs=T_abl, alpha0=const_alpha, use_cosine_schedule=False, seed=s
            ),
        ),
        ("no_revival", lambda s: EMeans(d=abl_ds.d, k=abl_k, nepochs=T_abl, revival_mode="off", seed=s)),
        (
            "revival_by_resampling",
            lambda s: EMeans(d=abl_ds.d, k=abl_k, nepochs=T_abl, revival_mode="resample", seed=s),
        ),
        (
            "no_finalization",
            lambda s: EMeans(d=abl_ds.d, k=abl_k, nepochs=T_abl, finalization="none", seed=s),
        ),
        (
            "slow_ema",
            lambda s: EMeans(
                d=abl_ds.d, k=abl_k, nepochs=T_abl, finalization="slow_ema", slow_ema_mult=0.25, seed=s
            ),
        ),
        ("naive_kmeans_pp_init", lambda s: NaiveKMeansPP_EMeans(d=abl_ds.d, k=abl_k, nepochs=T_abl, seed=s)),
        ("emeans_pp_ours", lambda s: EMeansPP(d=abl_ds.d, k=abl_k, nepochs=T_abl, seed=s)),
    ]

    ablation_summary: dict[str, Any] = {}
    base_train = base_test = None
    for name, factory in ablation_arms:
        rows = []
        for s in seeds:
            m = factory(s)
            c, _ = m.fit(x_abl)
            ev = eval_centroids(abl_ds, c, dev)
            ev["init_time_sec"] = m.init_time_sec
            ev["train_time_sec"] = m.train_time_sec
            ev["total_time_sec"] = m.total_time_sec
            rows.append(ev)
        summ = summarize_seeds(
            rows, ("train_mse", "test_mse", "init_time_sec", "train_time_sec", "total_time_sec")
        )
        if name == "our_recipe":
            base_train = summ["train_mse_median"]
            base_test = summ["test_mse_median"]
        summ["delta_train_pct"] = (
            0.0 if base_train is None else (summ["train_mse_median"] - base_train) / base_train * 100.0
        )
        summ["delta_test_pct"] = (
            0.0 if base_test is None else (summ["test_mse_median"] - base_test) / base_test * 100.0
        )
        ablation_summary[name] = summ
        print(
            f"  [Table 1 Ablation] {name:26s}: train={summ['train_mse_median']:.5f} ({summ['delta_train_pct']:+.2f}%) | "
            f"test={summ['test_mse_median']:.5f} ({summ['delta_test_pct']:+.2f}%) | "
            f"init={summ['init_time_sec_median']*1e3:.1f}ms tot={summ['total_time_sec_median']*1e3:.1f}ms"
        )

    return {"speedup_vs_lloyd25": speedup_results, "recipe_ablation_t3": ablation_summary}


# ===========================================================================
# Experiment 2: Speed-vs-MSE Pareto Frontier (All 6 Solvers, Charging Init!)
# ===========================================================================
def run_exp2_pareto_tradeoff(dev: torch.device, seeds: list[int], smoke: bool = False) -> dict[str, Any]:
    print("\n=== [Exp 2] Speed-vs-MSE Pareto Frontier Across 4 Benchmark Regimes ===")
    n_syn = 40_000 if smoke else 200_000

    benchmarks: list[tuple[DatasetSplit, int]] = [
        (load_cifar10_resnet18_embeddings("patches_128"), 512 if smoke else 2048),
        (load_cifar10_resnet18_embeddings("global_512"), 512 if smoke else 1024),
        (
            make_synthetic_dataset(
                "balanced_gmm", n_train=n_syn, n_val=10_000, n_test=10_000, d=128, k=512 if smoke else 2048, seed=201
            ),
            512 if smoke else 2048,
        ),
        (
            make_synthetic_dataset(
                "multiscale_rare_outliers",
                n_train=n_syn,
                n_val=10_000,
                n_test=10_000,
                d=128,
                k=256 if smoke else 1024,
                seed=202,
            ),
            256 if smoke else 1024,
        ),
    ]
    if smoke:
        benchmarks = benchmarks[2:]

    t_emeans = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0] if not smoke else [1.0, 3.0, 5.0]
    t_lloyd = [3, 6, 12, 25] if not smoke else [5, 25]
    t_sculley = [1.0, 3.0, 5.0, 10.0] if not smoke else [1.0, 5.0]
    t_naive = [1.0, 3.0, 5.0] if not smoke else [3.0]

    pareto_out: dict[str, Any] = {}
    for ds, k in benchmarks:
        print(f"  Running Pareto sweep on {ds.name} (N={len(ds.train)}, d={ds.d}, k={k})...")
        x_tr = torch.from_numpy(ds.train).to(dev)
        # Warmup at exact problem shape (N, K, D) outside timer!
        _ = EMeansPP(d=ds.d, k=k, nepochs=1.0, seed=0).fit(x_tr)
        _ = EMeans(d=ds.d, k=k, nepochs=1.0, seed=0).fit(x_tr)
        _ = LloydKMeans(d=ds.d, k=k, niter=2, init="random", seed=0).fit(x_tr)
        _sync(dev)

        sc_bsz, sc_rr = tune_sculley_on_val(ds, k, 5.0, dev)
        ds_res: dict[str, list[dict[str, Any]]] = {
            "lloyd_random": [],
            "lloyd_kmeans_pp": [],
            "sculley_minibatch": [],
            "emeans": [],
            "naive_kpp_emeans": [],
            "emeans_pp": [],
        }

        for niter in t_lloyd:
            rows_r, rows_p = [], []
            for s in seeds:
                mr = LloydKMeans(d=ds.d, k=k, niter=niter, init="random", seed=s)
                cr, _ = mr.fit(x_tr)
                er = eval_centroids(ds, cr, dev)
                er.update(init_time_sec=mr.init_time_sec, train_time_sec=mr.train_time_sec, total_time_sec=mr.total_time_sec)
                rows_r.append(er)

                mp = LloydKMeans(d=ds.d, k=k, niter=niter, init="k-means++", seed=s)
                cp, _ = mp.fit(x_tr)
                ep = eval_centroids(ds, cp, dev)
                ep.update(init_time_sec=mp.init_time_sec, train_time_sec=mp.train_time_sec, total_time_sec=mp.total_time_sec)
                rows_p.append(ep)

            sr = summarize_seeds(rows_r, ("train_mse", "test_mse", "init_time_sec", "train_time_sec", "total_time_sec"))
            sr["budget"] = niter
            ds_res["lloyd_random"].append(sr)

            sp = summarize_seeds(rows_p, ("train_mse", "test_mse", "init_time_sec", "train_time_sec", "total_time_sec"))
            sp["budget"] = niter
            ds_res["lloyd_kmeans_pp"].append(sp)

        for T in t_sculley:
            rows_sc = []
            for s in seeds:
                msc = SculleyMiniBatchKMeans(
                    d=ds.d, k=k, nepochs=T, batch_size=sc_bsz, reassignment_ratio=sc_rr, seed=s
                )
                csc, _ = msc.fit(x_tr)
                esc = eval_centroids(ds, csc, dev)
                esc.update(init_time_sec=msc.init_time_sec, train_time_sec=msc.train_time_sec, total_time_sec=msc.total_time_sec)
                rows_sc.append(esc)
            ssc = summarize_seeds(rows_sc, ("train_mse", "test_mse", "init_time_sec", "train_time_sec", "total_time_sec"))
            ssc["budget"] = T
            ssc["tuned_batch_size"] = sc_bsz
            ssc["tuned_reassignment_ratio"] = sc_rr
            ds_res["sculley_minibatch"].append(ssc)

        # Paired interleaved AB/BA runs for e-means vs e-means++
        for T in t_emeans:
            rows_em, rows_epp = [], []
            for idx_s, s in enumerate(seeds):
                if idx_s % 2 == 0:
                    mem = EMeans(d=ds.d, k=k, nepochs=T, seed=s)
                    cem, _ = mem.fit(x_tr)
                    mepp = EMeansPP(d=ds.d, k=k, nepochs=T, seed=s)
                    cepp, _ = mepp.fit(x_tr)
                else:
                    mepp = EMeansPP(d=ds.d, k=k, nepochs=T, seed=s)
                    cepp, _ = mepp.fit(x_tr)
                    mem = EMeans(d=ds.d, k=k, nepochs=T, seed=s)
                    cem, _ = mem.fit(x_tr)

                eem = eval_centroids(ds, cem, dev)
                eem.update(init_time_sec=mem.init_time_sec, train_time_sec=mem.train_time_sec, total_time_sec=mem.total_time_sec)
                rows_em.append(eem)

                eepp = eval_centroids(ds, cepp, dev)
                eepp.update(init_time_sec=mepp.init_time_sec, train_time_sec=mepp.train_time_sec, total_time_sec=mepp.total_time_sec)
                rows_epp.append(eepp)

            sem = summarize_seeds(rows_em, ("train_mse", "test_mse", "init_time_sec", "train_time_sec", "total_time_sec"))
            sem["budget"] = T
            ds_res["emeans"].append(sem)

            sepp = summarize_seeds(rows_epp, ("train_mse", "test_mse", "init_time_sec", "train_time_sec", "total_time_sec"))
            sepp["budget"] = T
            ds_res["emeans_pp"].append(sepp)

        for T in t_naive:
            rows_nv = []
            for s in seeds:
                mnv = NaiveKMeansPP_EMeans(d=ds.d, k=k, nepochs=T, seed=s)
                cnv, _ = mnv.fit(x_tr)
                env = eval_centroids(ds, cnv, dev)
                env.update(init_time_sec=mnv.init_time_sec, train_time_sec=mnv.train_time_sec, total_time_sec=mnv.total_time_sec)
                rows_nv.append(env)
            snv = summarize_seeds(rows_nv, ("train_mse", "test_mse", "init_time_sec", "train_time_sec", "total_time_sec"))
            snv["budget"] = T
            ds_res["naive_kpp_emeans"].append(snv)

        pareto_out[ds.name] = {"n_train": len(ds.train), "d": ds.d, "k": k, "curves": ds_res}

    return pareto_out


# ===========================================================================
# Experiment 3: Full Appendix D.2 Synthetic Distribution Suite + Rare Outliers
# ===========================================================================
def run_exp3_appendix_d2_suite(dev: torch.device, seeds: list[int], smoke: bool = False) -> dict[str, Any]:
    print("\n=== [Exp 3] Full Appendix D.2 Synthetic Suite + Rare Outliers ===")
    kinds = [
        "uniform_cube",
        "isotropic_gaussian",
        "stretched_gaussian_0.5",
        "stretched_gaussian_1",
        "stretched_gaussian_2",
        "balanced_gmm",
        "zipf_gmm",
        "multiscale_rare_outliers",
    ]
    if smoke:
        kinds = ["isotropic_gaussian", "balanced_gmm", "multiscale_rare_outliers"]

    n_tr = 30_000 if smoke else 100_000
    k = 256 if smoke else 1024
    d = 128
    suite_out: dict[str, Any] = {}

    for idx_k, kind in enumerate(kinds):
        ds = make_synthetic_dataset(
            kind, n_train=n_tr, n_val=10_000, n_test=10_000, d=d, k=k, seed=300 + idx_k
        )
        x_tr = torch.from_numpy(ds.train).to(dev)
        if idx_k == 0:
            _ = EMeansPP(d=d, k=k, nepochs=1.0, seed=0).fit(x_tr)
            _ = EMeans(d=d, k=k, nepochs=1.0, seed=0).fit(x_tr)
            _ = LloydKMeans(d=d, k=k, niter=2, init="k-means++", seed=0).fit(x_tr)
            _sync(dev)
        sc_bsz, sc_rr = tune_sculley_on_val(ds, k, 5.0, dev)

        methods = [
            ("lloyd_rand_25", lambda s: LloydKMeans(d=d, k=k, niter=25, init="random", seed=s)),
            ("lloyd_kpp_25", lambda s: LloydKMeans(d=d, k=k, niter=25, init="k-means++", seed=s)),
            (
                "sculley_5",
                lambda s: SculleyMiniBatchKMeans(
                    d=d, k=k, nepochs=5.0, batch_size=sc_bsz, reassignment_ratio=sc_rr, seed=s
                ),
            ),
            ("emeans_1", lambda s: EMeans(d=d, k=k, nepochs=1.0, seed=s)),
            ("emeans_3", lambda s: EMeans(d=d, k=k, nepochs=3.0, seed=s)),
            ("emeans_5", lambda s: EMeans(d=d, k=k, nepochs=5.0, seed=s)),
            ("naive_kpp_emeans_5", lambda s: NaiveKMeansPP_EMeans(d=d, k=k, nepochs=5.0, seed=s)),
            ("emeans_pp_1", lambda s: EMeansPP(d=d, k=k, nepochs=1.0, seed=s)),
            ("emeans_pp_3", lambda s: EMeansPP(d=d, k=k, nepochs=3.0, seed=s)),
            ("emeans_pp_5", lambda s: EMeansPP(d=d, k=k, nepochs=5.0, seed=s)),
        ]

        kind_res: dict[str, Any] = {}
        for m_name, fn in methods:
            rows = []
            for s in seeds:
                m = fn(s)
                c, _ = m.fit(x_tr)
                ev = eval_centroids(ds, c, dev)
                ev["total_time_sec"] = m.total_time_sec
                ev["init_time_sec"] = m.init_time_sec
                rows.append(ev)
            kind_res[m_name] = summarize_seeds(
                rows, ("train_mse", "test_mse", "init_time_sec", "total_time_sec")
            )

        # Paired t-test / Wilcoxon p-value comparing e-means++@5 vs e-means@5 and Lloyd@25
        epp5_te = kind_res["emeans_pp_5"]["test_mse_per_seed"]
        em5_te = kind_res["emeans_5"]["test_mse_per_seed"]
        p_val = float(sp_stats.ttest_rel(epp5_te, em5_te).pvalue) if len(seeds) >= 3 else 1.0
        kind_res["paired_ttest_p_epp5_vs_em5"] = p_val

        print(
            f"  {kind:25s}: Lloyd@25={kind_res['lloyd_rand_25']['test_mse_median']:.4f} | "
            f"e-means@1={kind_res['emeans_1']['test_mse_median']:.4f} -> @5={kind_res['emeans_5']['test_mse_median']:.4f} | "
            f"NaiveKPP@5={kind_res['naive_kpp_emeans_5']['test_mse_median']:.4f} | "
            f"e-means++@1={kind_res['emeans_pp_1']['test_mse_median']:.4f} -> @5={kind_res['emeans_pp_5']['test_mse_median']:.4f} (p={p_val:.3e})"
        )
        suite_out[kind] = kind_res

    return suite_out


# ===========================================================================
# Experiment 4: Mechanistic Proof — Coupon-Collector & Revival Assassination
# ===========================================================================
def run_exp4_mechanistic_traps(dev: torch.device, seeds: list[int], smoke: bool = False) -> dict[str, Any]:
    print("\n=== [Exp 4] Mechanistic Proof: Coupon-Collector & Revival Assassination ===")
    n_tr = 40_000 if smoke else 100_000
    k_gmm = 256 if smoke else 512
    d = 64

    # 4A: Coupon-Collector Empty-Mode Trap on Balanced Separated GMM
    ds_gmm = make_synthetic_dataset(
        "balanced_gmm", n_train=n_tr, n_val=5_000, n_test=10_000, d=d, k=k_gmm, seed=401
    )
    assert ds_gmm.true_centers is not None
    x_gmm = torch.from_numpy(ds_gmm.train).to(dev)
    sigma_w_gmm = math.sqrt(1.0 / 3.0)

    coupon_res: dict[str, Any] = {"oracle_test_mse": exact_mse_fp64(ds_gmm.test, ds_gmm.true_centers, device=dev)}
    budgets = [1.0, 2.0, 3.0, 5.0] if not smoke else [1.0, 5.0]

    # Measure T=0 initial centroids first!
    init_arms = {
        "random_init_t0": [],
        "raw_kpp_t0": [],
        "polished_coreset_kpp_t0": [],
    }
    for s in seeds:
        gen = torch.Generator(device=dev)
        gen.manual_seed(s)
        c_rand = x_gmm[torch.randperm(len(ds_gmm.train), generator=gen, device=dev)[:k_gmm]]
        init_arms["random_init_t0"].append({
            "test_mse": exact_mse_fp64(ds_gmm.test, c_rand, device=dev),
            "mode_coverage": mode_recovery_rate(ds_gmm.true_centers, c_rand, sigma_w_gmm),
        })
        sr_raw = seed_emeans_plusplus(x_gmm, k=k_gmm, polish=False, seed=s)
        init_arms["raw_kpp_t0"].append({
            "test_mse": exact_mse_fp64(ds_gmm.test, sr_raw.centroids, device=dev),
            "mode_coverage": mode_recovery_rate(ds_gmm.true_centers, sr_raw.centroids, sigma_w_gmm),
        })
        sr_pol = seed_emeans_plusplus(x_gmm, k=k_gmm, polish=True, seed=s)
        init_arms["polished_coreset_kpp_t0"].append({
            "test_mse": exact_mse_fp64(ds_gmm.test, sr_pol.centroids, device=dev),
            "mode_coverage": mode_recovery_rate(ds_gmm.true_centers, sr_pol.centroids, sigma_w_gmm),
        })
    for k_init, r_list in init_arms.items():
        coupon_res[k_init] = summarize_seeds(r_list, ("test_mse", "mode_coverage"))

    coupon_curves: dict[str, list[dict[str, Any]]] = {"lloyd_random": [], "emeans": [], "emeans_pp": []}
    for T in budgets:
        r_ll, r_em, r_epp = [], [], []
        for s in seeds:
            cll, _ = LloydKMeans(d=d, k=k_gmm, niter=int(round(T)), init="random", seed=s).fit(x_gmm)
            r_ll.append({
                "test_mse": exact_mse_fp64(ds_gmm.test, cll, device=dev),
                "mode_coverage": mode_recovery_rate(ds_gmm.true_centers, cll, sigma_w_gmm),
            })
            cem, _ = EMeans(d=d, k=k_gmm, nepochs=T, seed=s).fit(x_gmm)
            r_em.append({
                "test_mse": exact_mse_fp64(ds_gmm.test, cem, device=dev),
                "mode_coverage": mode_recovery_rate(ds_gmm.true_centers, cem, sigma_w_gmm),
            })
            cepp, _ = EMeansPP(d=d, k=k_gmm, nepochs=T, seed=s).fit(x_gmm)
            r_epp.append({
                "test_mse": exact_mse_fp64(ds_gmm.test, cepp, device=dev),
                "mode_coverage": mode_recovery_rate(ds_gmm.true_centers, cepp, sigma_w_gmm),
            })
        for arm_key, rows_arm in (("lloyd_random", r_ll), ("emeans", r_em), ("emeans_pp", r_epp)):
            sm = summarize_seeds(rows_arm, ("test_mse", "mode_coverage"))
            sm["budget"] = T
            coupon_curves[arm_key].append(sm)
    coupon_res["curves"] = coupon_curves

    print(
        f"  [Coupon-Collector] T=0 coverage: Random={coupon_res['random_init_t0']['mode_coverage_mean']*100:.1f}% "
        f"(theory 63.2%), Polished e-means++={coupon_res['polished_coreset_kpp_t0']['mode_coverage_mean']*100:.1f}% "
        f"(MSE {coupon_res['raw_kpp_t0']['test_mse_mean']:.2f} -> {coupon_res['polished_coreset_kpp_t0']['test_mse_mean']:.2f} via Voronoi Polish!)"
    )

    # 4B: Revival Assassination Trap on Multi-Scale Rare Outliers
    k_rare = 128 if smoke else 256
    ds_rare = make_synthetic_dataset(
        "multiscale_rare_outliers", n_train=n_tr, n_val=5_000, n_test=10_000, d=d, k=k_rare, seed=402
    )
    assert ds_rare.true_centers is not None and ds_rare.rare_mode_indices is not None
    rare_true = ds_rare.true_centers[ds_rare.rare_mode_indices]
    x_rare = torch.from_numpy(ds_rare.train).to(dev)

    rare_curves: dict[str, list[dict[str, Any]]] = {
        "emeans": [],
        "naive_kpp_emeans": [],
        "emeans_pp_no_huygens": [],
        "emeans_pp": [],
    }
    for T in budgets:
        for arm_name, fn in [
            ("emeans", lambda s: EMeans(d=d, k=k_rare, nepochs=T, seed=s)),
            ("naive_kpp_emeans", lambda s: NaiveKMeansPP_EMeans(d=d, k=k_rare, nepochs=T, seed=s)),
            (
                "emeans_pp_no_huygens",
                lambda s: EMeansPP(d=d, k=k_rare, nepochs=T, use_huygens_protection=False, seed=s),
            ),
            ("emeans_pp", lambda s: EMeansPP(d=d, k=k_rare, nepochs=T, seed=s)),
        ]:
            rows = []
            for s in seeds:
                m = fn(s)
                c, _ = m.fit(x_rare)
                c_np = c.cpu().numpy()
                d2_r = ((rare_true[:, None, :] - c_np[None, :, :]) ** 2).sum(axis=2).min(axis=1)
                surv = float((d2_r < 4.0).mean())
                ev = eval_centroids(ds_rare, c, dev)
                ev["rare_survival_rate"] = surv
                rows.append(ev)
            sm = summarize_seeds(rows, ("train_mse", "test_mse", "rare_survival_rate"))
            sm["budget"] = T
            rare_curves[arm_name].append(sm)

    print(
        f"  [Rare-Mode Survival @ T={budgets[-1]}] "
        f"e-means={rare_curves['emeans'][-1]['rare_survival_rate_mean']*100:.1f}% | "
        f"Naive k-m+++e-means={rare_curves['naive_kpp_emeans'][-1]['rare_survival_rate_mean']*100:.1f}% | "
        f"e-means++ (no Huygens)={rare_curves['emeans_pp_no_huygens'][-1]['rare_survival_rate_mean']*100:.1f}% | "
        f"e-means++ (Full)={rare_curves['emeans_pp'][-1]['rare_survival_rate_mean']*100:.1f}%"
    )
    return {
        "coupon_collector_balanced_gmm": coupon_res,
        "rare_mode_assassination": {
            "oracle_test_mse": exact_mse_fp64(ds_rare.test, ds_rare.true_centers, device=dev),
            "n_rare_modes": len(rare_true),
            "curves": rare_curves,
        },
    }


# ===========================================================================
# Experiment 5: Pillar-by-Pillar Ablation of e-means++
# ===========================================================================
def run_exp5_emeans_pp_ablation(dev: torch.device, seeds: list[int], smoke: bool = False) -> dict[str, Any]:
    print("\n=== [Exp 5] Pillar-by-Pillar Ablation of e-means++ ===")
    n_tr = 30_000 if smoke else 100_000
    k = 256 if smoke else 1024

    abl_datasets: list[tuple[DatasetSplit, int, float]] = [
        (
            make_synthetic_dataset(
                "balanced_gmm", n_train=n_tr, n_val=5_000, n_test=10_000, d=128, k=k, seed=501
            ),
            k,
            0.5,
        ),
        (
            make_synthetic_dataset(
                "multiscale_rare_outliers", n_train=n_tr, n_val=5_000, n_test=10_000, d=128, k=k // 2, seed=502
            ),
            k // 2,
            5.0,
        ),
        (
            make_synthetic_dataset(
                "zipf_gmm", n_train=n_tr, n_val=5_000, n_test=10_000, d=128, k=k, seed=503
            ),
            k,
            3.0,
        ),
        (load_cifar10_resnet18_embeddings("patches_128"), 512 if smoke else 2048, 1.0),
    ]
    if smoke:
        abl_datasets = abl_datasets[:2]

    out: dict[str, Any] = {}
    for ds, k_use, T in abl_datasets:
        x_tr = torch.from_numpy(ds.train).to(dev)
        _ = EMeansPP(d=ds.d, k=k_use, nepochs=1.0, seed=0).fit(x_tr)
        _sync(dev)

        variants = [
            ("full_emeans_pp", lambda s: EMeansPP(d=ds.d, k=k_use, nepochs=T, seed=s)),
            (
                "minus_coreset_full_greedy",
                lambda s: EMeansPP(d=ds.d, k=k_use, nepochs=T, seeding_mode="full_greedy", seed=s),
            ),
            (
                "minus_voronoi_polish",
                lambda s: EMeansPP(d=ds.d, k=k_use, nepochs=T, use_voronoi_polish=False, seed=s),
            ),
            (
                "minus_prior_mass",
                lambda s: EMeansPP(
                    d=ds.d, k=k_use, nepochs=T, use_prior_mass=False, warm_alpha_scale=1.0, seed=s
                ),
            ),
            (
                "minus_huygens_protection",
                lambda s: EMeansPP(d=ds.d, k=k_use, nepochs=T, use_huygens_protection=False, seed=s),
            ),
            (
                "minus_d2_donor_split",
                lambda s: EMeansPP(d=ds.d, k=k_use, nepochs=T, use_d2_donor_split=False, seed=s),
            ),
            ("naive_kpp_emeans", lambda s: NaiveKMeansPP_EMeans(d=ds.d, k=k_use, nepochs=T, seed=s)),
            ("vanilla_emeans", lambda s: EMeans(d=ds.d, k=k_use, nepochs=T, seed=s)),
        ]

        ds_abl: dict[str, Any] = {"T": T, "k": k_use, "variants": {}}
        ref_test = None
        for v_name, fn in variants:
            rows = []
            for s in seeds:
                m = fn(s)
                c, _ = m.fit(x_tr)
                ev = eval_centroids(ds, c, dev)
                ev["init_time_sec"] = m.init_time_sec
                ev["train_time_sec"] = m.train_time_sec
                ev["total_time_sec"] = m.total_time_sec
                rows.append(ev)
            sm = summarize_seeds(
                rows, ("train_mse", "test_mse", "init_time_sec", "train_time_sec", "total_time_sec")
            )
            if v_name == "full_emeans_pp":
                ref_test = sm["test_mse_median"]
            sm["delta_test_vs_full_pct"] = (
                0.0 if ref_test is None else (sm["test_mse_median"] - ref_test) / ref_test * 100.0
            )
            ds_abl["variants"][v_name] = sm
            print(
                f"  [{ds.name} T={T}] {v_name:27s}: test={sm['test_mse_median']:.5f} "
                f"({sm['delta_test_vs_full_pct']:+.2f}%) | init={sm['init_time_sec_median']*1e3:.1f}ms "
                f"tot={sm['total_time_sec_median']*1e3:.1f}ms"
            )
        out[ds.name] = ds_abl

    return out


# ===========================================================================
# Publication-Quality Matplotlib Plotting
# ===========================================================================
def generate_plots(results: dict[str, Any], out_dir: Path) -> list[str]:
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "figure.dpi": 160,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })
    saved: list[str] = []

    # 1. Pareto Speed-vs-MSE 2x2 Grid (`pareto_speed_vs_mse.png`)
    pareto = results.get("exp2_pareto", {})
    if pareto:
        names = list(pareto.keys())
        ncols = min(2, len(names))
        nrows = math.ceil(len(names) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 4.6 * nrows), squeeze=False)
        style_map = {
            "lloyd_random": ("Lloyd (random)", "#d62728", "s", "--"),
            "lloyd_kmeans_pp": ("Lloyd (greedy k-means++)", "#ff7f0e", "D", "--"),
            "sculley_minibatch": ("Sculley MiniBatch (tuned)", "#8c564b", "^", "-."),
            "naive_kpp_emeans": ("Naive k-means++ + e-means", "#9467bd", "v", ":"),
            "emeans": ("e-means (Meta FAIR)", "#1f77b4", "o", "-"),
            "emeans_pp": ("e-means++ (Ours)", "#2ca02c", "*", "-"),
        }
        for idx, dname in enumerate(names):
            ax = axes[idx // ncols][idx % ncols]
            entry = pareto[dname]
            curves = entry["curves"]
            ll25_mse = curves["lloyd_random"][-1]["test_mse_median"]
            ax.axhline(0.0, color="#d62728", linestyle=":", alpha=0.6, linewidth=1.2, label="Lloyd@25 ref (0%)")
            for mkey, (label, color, marker, ls) in style_map.items():
                pts = curves.get(mkey, [])
                if not pts:
                    continue
                xs = [p["total_time_sec_median"] * 1e3 for p in pts]
                ys = [(p["test_mse_median"] - ll25_mse) / ll25_mse * 100.0 for p in pts]
                lw = 2.4 if mkey == "emeans_pp" else 1.7
                ms = 8.5 if mkey == "emeans_pp" else 5.5
                ax.plot(xs, ys, label=label, color=color, marker=marker, linestyle=ls, linewidth=lw, markersize=ms)
            ax.set_xscale("log")
            ax.set_xlabel("Total Warm Wall-Clock Time incl. Seeding (ms, log scale)")
            ax.set_ylabel("Held-Out Test MSE vs. Lloyd@25 (%)")
            ax.set_title(f"{dname} (N={entry['n_train']:,}, d={entry['d']}, k={entry['k']})")
            if idx == 0:
                ax.legend(loc="best", framealpha=0.9)
        fig.suptitle("Phase 1 Pareto Frontier: Held-Out Test Distortion vs. Total Wall-Clock Time", fontweight="bold", y=1.01)
        fig.tight_layout()
        p1 = out_dir / "fig1_pareto_speed_vs_mse.png"
        fig.savefig(p1, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(p1))

    # 2. Mechanistic Proof 2-Panel Plot (`fig2_mechanistic_traps.png`)
    mech = results.get("exp4_mechanistic", {})
    if mech:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.6))
        cc = mech["coupon_collector_balanced_gmm"]
        t0_map = {
            "lloyd_random": cc["random_init_t0"]["mode_coverage_mean"] * 100.0,
            "emeans": cc["random_init_t0"]["mode_coverage_mean"] * 100.0,
            "emeans_pp": cc["polished_coreset_kpp_t0"]["mode_coverage_mean"] * 100.0,
        }
        for key, label, col, mk in [
            ("lloyd_random", "Lloyd (random init)", "#d62728", "s"),
            ("emeans", "Vanilla e-means", "#1f77b4", "o"),
            ("emeans_pp", "e-means++ (Ours)", "#2ca02c", "*"),
        ]:
            pts = cc["curves"][key]
            xs = [0.0] + [p["budget"] for p in pts]
            ys = [t0_map[key]] + [p["mode_coverage_mean"] * 100.0 for p in pts]
            ax1.plot(xs, ys, label=label, color=col, marker=mk, linewidth=2.2, markersize=7)
        ax1.axhline(100.0 * (1.0 - math.exp(-1.0)), color="gray", linestyle="--", alpha=0.7, label="Poisson 1 - 1/e = 63.2% trap")
        ax1.set_xlabel("Training Budget T (Epochs)")
        ax1.set_ylabel("Ground-Truth Gaussian Modes Recovered (%)")
        ax1.set_ylim(55, 102)
        ax1.set_title("(a) Overcoming the 36.8% Empty-Mode Trap (Balanced GMM)")
        ax1.legend(loc="lower right")

        ra = mech["rare_mode_assassination"]["curves"]
        for key, label, col, mk in [
            ("emeans", "Vanilla e-means", "#1f77b4", "o"),
            ("naive_kpp_emeans", "Naive k-means++ + e-means", "#9467bd", "v"),
            ("emeans_pp_no_huygens", "e-means++ w/o Huygens protection", "#ff7f0e", "D"),
            ("emeans_pp", "e-means++ (Huygens Protected)", "#2ca02c", "*"),
        ]:
            pts = ra[key]
            xs = [p["budget"] for p in pts]
            ys = [p["rare_survival_rate_mean"] * 100.0 for p in pts]
            ax2.plot(xs, ys, label=label, color=col, marker=mk, linewidth=2.2, markersize=7)
        ax2.set_xlabel("Training Budget T (Epochs)")
        ax2.set_ylabel("Rare Outlier Modes Preserved (%)")
        ax2.set_ylim(-5, 105)
        ax2.set_title("(b) Preventing Revival Assassination of Rare Modes (p_r = 0.03/k)")
        ax2.legend(loc="center right")

        fig.tight_layout()
        p2 = out_dir / "fig2_mechanistic_traps.png"
        fig.savefig(p2, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(p2))

    # 3. Synthetic Suite + e-means++ Pillar Ablation (`fig3_suite_and_ablation.png`)
    suite = results.get("exp3_suite", {})
    abl = results.get("exp5_ablation", {})
    if suite and abl:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.0))
        skinds = list(suite.keys())
        x_pos = np.arange(len(skinds))
        width = 0.25
        for offset, mkey, lbl, col in [
            (-width, "emeans_5", "e-means (T=5)", "#1f77b4"),
            (0.0, "naive_kpp_emeans_5", "Naive k-m++ + e-means (T=5)", "#9467bd"),
            (width, "emeans_pp_5", "e-means++ (T=5, Ours)", "#2ca02c"),
        ]:
            deltas = []
            for kd in skinds:
                ll_ref = suite[kd]["lloyd_rand_25"]["test_mse_median"]
                val = suite[kd][mkey]["test_mse_median"]
                deltas.append((val - ll_ref) / ll_ref * 100.0)
            ax1.bar(x_pos + offset, deltas, width=width, label=lbl, color=col, alpha=0.88)
        ax1.axhline(0.0, color="#d62728", linestyle="--", linewidth=1.1, label="Lloyd@25 (0%)")
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([k.replace("_gaussian", "\ngauss").replace("multiscale_", "") for k in skinds], rotation=28, ha="right", fontsize=8)
        ax1.set_ylabel("Test MSE Delta vs. Lloyd@25 (%, lower is better)")
        ax1.set_title("(a) Appendix D.2 + Rare-Outlier Suite (d=128, k=1024)")
        ax1.legend(loc="best")

        # Panel b: Pillar ablation on multiscale_rare_outliers (or first available)
        target_abl_key = "multiscale_rare_outliers" if "multiscale_rare_outliers" in abl else list(abl.keys())[0]
        vdict = abl[target_abl_key]["variants"]
        vkeys = list(vdict.keys())
        y_pos = np.arange(len(vkeys))
        vals = [vdict[vk]["delta_test_vs_full_pct"] for vk in vkeys]
        colors = ["#2ca02c" if vk == "full_emeans_pp" else ("#d62728" if val > 1.0 else "#ff7f0e") for vk, val in zip(vkeys, vals)]
        ax2.barh(y_pos, vals, color=colors, alpha=0.88)
        ax2.axvline(0.0, color="black", linewidth=1.0)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([vk.replace("_", " ") for vk in vkeys], fontsize=8.5)
        ax2.invert_yaxis()
        ax2.set_xlabel("Excess Held-Out Test MSE vs. Full e-means++ (%)")
        ax2.set_title(f"(b) e-means++ Pillar Ablation ({target_abl_key})")

        fig.tight_layout()
        p3 = out_dir / "fig3_suite_and_ablation.png"
        fig.savefig(p3, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(p3))

    return saved


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".tmp_{os.getpid()}_{path.name}"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 1 e-means & e-means++ benchmarks.")
    parser.add_argument("--seeds", type=int, default=8, help="Number of random seeds per configuration.")
    parser.add_argument("--smoke", action="store_true", help="Fast smoke check mode.")
    parser.add_argument(
        "--only",
        type=str,
        default="all",
        help="Comma-separated subset of experiments to run (exp1_repro,exp2_pareto,exp3_suite,exp4_mechanistic,exp5_ablation) or 'all'.",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Merge per-experiment JSON shard files (`part_<exp>.json`) into `phase1_results.json` and regenerate plots.",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = "smoke_" if args.smoke else ""
    dev = torch.device(args.device)
    n_seeds = 2 if args.smoke else int(args.seeds)
    seeds = [100 + 17 * i for i in range(n_seeds)]
    seeds_mech = [200 + 13 * i for i in range(2 if args.smoke else max(n_seeds, 10))]

    if args.merge_only:
        merged: dict[str, Any] = {
            "metadata": {
                "device": str(dev),
                "gpu_name": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
                "torch_version": torch.__version__,
                "seeds_main": seeds,
                "seeds_mechanistic": seeds_mech,
                "smoke": bool(args.smoke),
                "merged_from_parts": [],
            }
        }
        total_wall = 0.0
        for exp_key in ALL_EXPS:
            part_path = out_dir / f"{prefix}part_{exp_key}.json"
            if not part_path.exists():
                raise FileNotFoundError(f"Missing expected part file for merge: {part_path}")
            with open(part_path, "r", encoding="utf-8") as f:
                part_data = json.load(f)
            merged[exp_key] = part_data["data"]
            total_wall += float(part_data.get("wall_sec", 0.0))
            merged["metadata"]["merged_from_parts"].append(
                {
                    "exp": exp_key,
                    "host": part_data.get("host", "unknown"),
                    "gpu_name": part_data.get("gpu_name", "unknown"),
                    "wall_sec": part_data.get("wall_sec", 0.0),
                }
            )
        merged["metadata"]["sum_shard_wall_sec"] = total_wall
        plots = generate_plots(merged, out_dir)
        merged["metadata"]["plots"] = plots
        json_path = out_dir / ("phase1_smoke_results.json" if args.smoke else "phase1_results.json")
        _atomic_write_json(json_path, merged)
        print(f"\nMerged all {len(ALL_EXPS)} experiment parts to: {json_path}")
        for p in plots:
            print(f"Saved figure to: {p}")
        return

    selected = list(ALL_EXPS) if args.only == "all" else [s.strip() for s in args.only.split(",") if s.strip()]
    for s_exp in selected:
        if s_exp not in ALL_EXPS:
            raise ValueError(f"Unknown experiment {s_exp!r}; valid choices: {ALL_EXPS}")

    gpu_name = torch.cuda.get_device_name(dev) if dev.type == "cuda" else "CPU"
    host_name = os.uname().nodename
    print(f"Phase 1 Benchmark Suite | host={host_name} | device={dev} ({gpu_name}) | selected={selected} | seeds={seeds}")
    t_all = time.perf_counter()
    warmup_gpu(dev)

    runners = {
        "exp1_repro": lambda: run_exp1_repro_emeans(dev, seeds, smoke=args.smoke),
        "exp2_pareto": lambda: run_exp2_pareto_tradeoff(dev, seeds, smoke=args.smoke),
        "exp3_suite": lambda: run_exp3_appendix_d2_suite(dev, seeds, smoke=args.smoke),
        "exp4_mechanistic": lambda: run_exp4_mechanistic_traps(dev, seeds_mech, smoke=args.smoke),
        "exp5_ablation": lambda: run_exp5_emeans_pp_ablation(dev, seeds, smoke=args.smoke),
    }

    results: dict[str, Any] = {
        "metadata": {
            "host": host_name,
            "device": str(dev),
            "gpu_name": gpu_name,
            "torch_version": torch.__version__,
            "seeds_main": seeds,
            "seeds_mechanistic": seeds_mech,
            "smoke": bool(args.smoke),
            "selected": selected,
        }
    }

    for exp_key in selected:
        t_e0 = time.perf_counter()
        exp_data = runners[exp_key]()
        t_el = time.perf_counter() - t_e0
        results[exp_key] = exp_data
        part_path = out_dir / f"{prefix}part_{exp_key}.json"
        _atomic_write_json(
            part_path,
            {
                "exp": exp_key,
                "host": host_name,
                "gpu_name": gpu_name,
                "seeds": seeds_mech if exp_key == "exp4_mechanistic" else seeds,
                "wall_sec": t_el,
                "data": exp_data,
            },
        )
        print(f"  -> Wrote shard artifact: {part_path} ({t_el:.1f}s)")

    results["metadata"]["total_campaign_wall_sec"] = time.perf_counter() - t_all
    if set(selected) == set(ALL_EXPS):
        plots = generate_plots(results, out_dir)
        results["metadata"]["plots"] = plots
        json_path = out_dir / ("phase1_smoke_results.json" if args.smoke else "phase1_results.json")
        _atomic_write_json(json_path, results)
        print(f"\nSaved JSON results to: {json_path}")
        for p in plots:
            print(f"Saved figure to:       {p}")


if __name__ == "__main__":
    main()
