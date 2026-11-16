# `emeans`: Principled `e-means++`, Forward-Only Voronoi Search, and Blackwell SM120 Clustering

High-performance PyTorch + Triton library reproducing **Meta FAIR's `e-means`** (`arXiv:2611.09501`), introducing **`e-means++`** to overcome both the $O(nkd)$ seeding bottleneck and count-only rare-mode assassination of naive `k-means++ + e-means`, evaluating **10 pre-registered algorithmic & forward-only (`qlabs-eng/noise`) hypotheses across 7,200 paired GPU runs**, and delivering **Blackwell `SM120` BF16 / Centered-FP8 / Sync-Free IVF / Multi-GPU NCCL kernels** capable of clustering **10 million 128-D vectors into 16,384 centroids in 400.7 ms on 1 GPU and 297.1 ms on 2 GPUs (`95.7%` parallel efficiency)**.

---

## Key Highlights

- **Exact `e-means` Reproduction (`EMeans`)**: Bit-structural parity with `faiss.contrib.e_means.base` universal power laws ($B(k, n, T)$, $\alpha_0(k, B, n, T)$), cosine schedule floor ($0.1\alpha_0$), Gumbel-max donor revival, and Appendix I.1 represented-norm matching ($\|x - \hat{c}_j\|_2^2 = \|x\|_2^2 + \|\hat{c}_j\|_2^2 - 2\langle x, \hat{c}_j\rangle$). Matches `Lloyd@25` held-out FP64 test MSE in **`0.6–2.0` epochs** (`1.5x–4.9x` single-GPU speedup) and reproduces all Table 1 ablations.
- **Principled `e-means++` (`EMeansPP`)**: Combines $O(nd)$ lightweight coresets (Bachem et al., KDD 2018), GPU-fused Block-Greedy $k$-means++, **1-Step Weighted Coreset Voronoi Polish** (eliminating the $2\times$ high-$d$ shell-variance penalty at $T=0$: `-39.2%` MSE), **Prior-Calibrated EMA Mass $N_j^{(0)}$**, **Huygens Parallel-Axis Rare-Mode Protection** ($M_j^{\text{eff}} \min_{m \neq j}\|c_j - c_m\|_2^2 > \gamma_{\text{prot}}\bar{E}$), and **$D^2$ EMA-Distortion Donor Splitting**:
  - **`4.6x–10.3x` faster seeding** than full Greedy $k$-means++.
  - **`-38.42%` lower test MSE** than `Lloyd@25` in **`48.7 ms`** on `multiscale_rare_outliers` (beating `Vanilla e-means`, `Sculley`, and `3.7x`-slower `Naive k-means++ + e-means`).
  - **`86.8%` mean (`90.0%` median) rare-mode preservation** at $T=5$ vs. **`0.0%`** without Huygens protection.
- **Phase 2A Classical & Forward-Only Extensions (`EMeansAlgoExt`, `EMeansNoiseTrack` — `7,200` Paired Runs)**:
  - **`H-Algo-5` (`H5_pp_eb_auto`, Bounded-Influence Empirical-Bayes / James-Stein Shrinkage)**: Eliminates warm-started `slow_ema` lag (`+1.20%..+10.17%` train MSE penalty) and beats `emeans_pp_base` out-of-sample test MSE on **`10/12` cells** ($p_W < 0.01$, up to **`-0.3205%`**, $p_t = 6.54 \times 10^{-13}$). Paired with **`H-Algo-1` (Online Oja Principal-Axis Split + BF16 `1e-4` Rounding-Collapse Fix)**, **`Combo_pp_H1_H5`** wins **all `12/12` cells** (`7/12` at $p_W < 0.05$).
  - **`N-Track-1` (`head_edit` $P/Q$ Forward-Verified Signed-Hadamard Transplants)**: Wins **`5/5` multimodal GMM cells** on `e-means++` (up to **`-5.810%`** test MSE, rare-mode recall `49.80% -> 57.78%`) while held-back $Q$-confirmation yields **`0 / 360` (`0.0%`) false-positive transplants** on continuous isotropic controls.
  - **`N-Track-5` (`wide_step` Batch Doubling)**: Cuts `train_time_sec` by **`-21.4%` to `-35.2%`** across **`24/24` pairs** via late $2\times$ batch widening coupled with $\alpha_0 \cdot 2^{0.9003}$.
- **Phase 2B Blackwell `SM120` Kernels & Multi-GPU Systems (`FastEMeans`, `MultiGPUFastEMeans`)**:
  - **`H-Opt-1` (`blackwell_triton_assign`)**: **`16.52x–19.05x`** 2D Split-$K$ + `FULL_D` speedup, **`317.8 TFLOP/s`** 3-stage SM120 TMA BF16 (`24 KB/stage`, `num_stages=3` under SM120's `99 KB` SMEM limit), `0.000%` argmin disagreement across `30/30` `(SKU x shape)` cells.
  - **`H-Opt-2` (`FP8Quantizer` + `fp8_assign`)**: **`636.4 TFLOP/s` (`81.8%–84.2%` of FP8 GEMM peak)** across 300W Max-Q, 450W Workstation, and 600W Server SKUs, with Appendix I.1 auto-centering holding shifted-manifold (`+25.0`) held-out FP64 test MSE parity within **`-0.014%` to `+0.078%` (`< 0.08%`)**.
  - **`H-Opt-3a/b`**: **`1.85x–2.04x`** fused Triton Scatter+EMA+Prep step speedup (`B <= 4096`) and **`1.75x–4.54x`** de-confounded Triton + `CUDAGraph` Greedy $k$-means++.
  - **`H-Opt-4` (`IVFAssigner`)**: **`6.44x–7.44x`** sync-free inverted-list Tensor-Core speedup at $k = 65{,}536$ (`53.9 ms` on 450W) while **lowering** held-out test MSE by **`-0.184%` to `-0.269%`**.
  - **`H-Opt-5a/b`**: Zero-copy streaming + 2-GPU NCCL `all_reduce` clusters **`n = 10M, k = 16,384, d = 128` in `400.7 ms` (1 GPU, `3.39x`) and `297.1 ms` (2 GPUs, `1.914x` / `95.7%` parallel efficiency, `5.48x` total speedup)**.

---

## Installation & Verification

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e ".[dev]"

# Run the full 156-test CPU + CUDA verification suite (~18.5s on a Blackwell GPU)
CUDA_VISIBLE_DEVICES=3 .venv/bin/pytest -v
```

---

## Unified Python API Quickstart

All six clustering engines and helper primitives are exported directly from `emeans`:

```python
import torch
from emeans import (
    EMeans,
    EMeansPP,
    EMeansAlgoExt,
    EMeansNoiseTrack,
    FastEMeans,
    MultiGPUFastEMeans,
    exact_mse_fp64,
)

x_train = torch.randn(200_000, 128, device="cuda", dtype=torch.float32)
x_test = torch.randn(20_000, 128, device="cuda", dtype=torch.float32)

# 1. Meta FAIR e-means (exact arXiv:2611.09501 recipe)
c_em, _ = EMeans(d=128, k=2048, nepochs=5.0, seed=42).fit(x_train)

# 2. Principled e-means++ (Coreset + Voronoi Polish + Prior Mass + Huygens Shield + D^2 Split)
c_pp, _ = EMeansPP(d=128, k=2048, nepochs=5.0, seed=42).fit(x_train)

# 3. Phase 2A Classical Winner: Combo H1 (Oja Principal-Axis Split) + H5 (Bounded-Influence EB-JS)
c_h15, _ = EMeansAlgoExt(
    d=128,
    k=2048,
    nepochs=5.0,
    base_mode="emeans_pp",
    split_mode="oja",
    finalization="eb_shrinkage",
    shrinkage_kappa="auto",
    seed=42,
).fit(x_train)

# 4. Phase 2A Forward-Only (qlabs-eng/noise): P/Q-Confirmed head_edit + 2x Wide-Step Speedrun
c_noise, _ = EMeansNoiseTrack(
    d=128,
    k=2048,
    nepochs=5.0,
    base_mode="emeans_pp",
    head_edit=True,
    wide_step_mult=2,
    seed=42,
).fit(x_train)

# 5. Phase 2B Single-GPU Blackwell SM120 Centered-FP8 (or assign_mode="ivf" for k >= 16,384)
c_fast, _ = FastEMeans(d=128, k=2048, nepochs=5.0, precision="fp8", seed=42).fit(x_train)

# 6. Phase 2B Multi-GPU Data-Sharded NCCL All-Reduce (bit-identical centroid replicas)
if torch.cuda.device_count() >= 2:
    c_mgpu = MultiGPUFastEMeans(
        d=128, k=2048, devices=[0, 1], nepochs=5.0, precision="fp8", seed=42
    ).fit(x_train)

print("Held-out FP64 Test MSE (Combo H1+H5):", exact_mse_fp64(x_test, c_h15))
```

---

## Repository Structure

```text
emeans/
├── src/emeans/
│   ├── __init__.py            # Unified public exports (EMeans, EMeansPP, EMeansAlgoExt, EMeansNoiseTrack, FastEMeans, MultiGPUFastEMeans)
│   ├── recipe.py              # Universal batch/step power laws, cosine schedule, power-of-two rounding
│   ├── kernels.py             # Represented-norm Triton / tiled PyTorch / Faiss assignment & exact_mse_fp64
│   ├── emeans.py              # Sync-free GPU/CPU EMeans (closing_mean / slow_ema / none + Gumbel-max revival)
│   ├── seeding.py             # O(nd) lightweight coreset, Block-Greedy k-means++, 1-step Voronoi polish
│   ├── emeans_pp.py           # Principled EMeansPP with toggleable pillars & Huygens rare-mode shield
│   ├── baselines.py           # LloydKMeans, SculleyMiniBatchKMeans, NaiveKMeansPP_EMeans
│   ├── datasets.py            # 8 synthetic distributions + CIFAR-10 ResNet-18 embeddings (atomic .npz cache)
│   ├── algo_extensions.py     # Phase 2A Classical H-Algo-1..5 (Oja, Hartigan, Nesterov, Power-Top2, EB-JS)
│   ├── noise_tracks.py        # Phase 2A Forward-Only N-Track-1..5 (head_edit, quad_merge, ANVIL, Dust, Wide-Step)
│   └── fast_ops.py            # Phase 2B Blackwell SM120 Split-K/FULL_D/TMA, FP8Quantizer, IVFAssigner, FastEMeans, MultiGPUFastEMeans
├── scripts/
│   ├── run_phase1_emeans_pp.py  # Phase 1 reproduction, Pareto, 8-dist suite, mechanistic traps, ablations
│   ├── run_phase2_algo.py       # Phase 2A 3,600-run multi-GPU classical falsification harness
│   ├── run_phase2a_noise.py     # Phase 2A 3,600-run 12-GPU / 6-host forward-only falsification harness
│   ├── run_phase2_opt.py        # Phase 2B paired AB/BA roofline, BF16/FP8, IVF, and NCCL scaling benchmark
│   └── check_fleet.sh           # Non-hanging SSH & GPU readiness probe across the 6-host Blackwell fleet
├── tests/
│   ├── test_emeans.py           # 11 core reproduction, norm-matching, coreset & Huygens tests
│   ├── test_algo_extensions.py  # 38 parity, Oja, Hartigan, Nesterov, Power-Top2 & EB-JS tests
│   ├── test_fast_ops.py         # 45 SM120 TMA, FP8 centering, fused EMA, CUDA-Graph, IVF & 2-GPU NCCL tests
│   └── test_noise_tracks.py     # 62 Hadamard, top-3 swap, twin-shield head_edit, quad_merge & ANVIL tests
├── docs/
│   ├── literature_survey.md          # 48-citation mathematical survey & H-Algo-1..5 derivations
│   ├── noise_forward_only_survey.md  # qlabs-eng/noise catalog, clustering isomorphism & N-Track-1..5 results
│   ├── optimizations.md              # SM120 3-SKU roofline model & H-Opt-1..5 AB/BA benchmark synthesis
│   └── fleet_status.md               # 6-host / 16-GPU RTX PRO 6000 Blackwell cluster topology
└── outputs/
    ├── phase1/                       # phase1_results.json + fig1..fig3 PNG plots
    ├── phase2_algo/classical_tracks/ # phase2_algo_results.json (3,600 runs), summary & plots
    ├── phase2_algo/noise_tracks/     # phase2a_noise_results.json (3,600 runs), summary & plots
    └── phase2_opt/                   # 300W Max-Q, 450W Workstation & 600W Server JSONs + 9 PNG plots
```

---

## Summary of Falsifiable Hypotheses & Empirical Verdicts

### Phase 2A: Classical (`H-Algo-1..5`) & Forward-Only `qlabs-eng/noise` (`N-Track-1..5`)

| ID | Mechanism | Empirical Verdict (`15` Paired Seeds $\times$ `12` Cells) | Headline Metric (`FP64` Held-Out Test MSE / Wall-Clock) |
| :--- | :--- | :--- | :--- |
| **`H-Algo-5`** | Bounded-Influence Empirical-Bayes / James-Stein Finalization (`H5_pp_eb_auto`) | **CONFIRMED WINNER** (`10/12` `e-means++` cells $p_W < 0.01$; `12/12` vs `slow_ema`) | Up to **`-0.3205%`** test MSE ($p_t = 6.54\times 10^{-13}$); eliminates `+1.2%..+10.2%` `slow_ema` lag (`+4.3 ms`). |
| **`H-Algo-1` + `H5`** | Online Oja Principal-Axis Split + BF16 `1e-4` Fix (`Combo_pp_H1_H5`) | **CONFIRMED SYNERGY** (**`12/12` `e-means++` cells $< 0$**, `7/12` $p_W < 0.05$) | Up to **`-0.3367%`** test MSE (`cifar10_r18_512_t2`); `-0.8361%` train MSE on `multiscale_rare_t5`. |
| **`H-Algo-2`** | Annealed Hartigan Marginal Distortion (`H2_*_hartigan_b1`) | **REGIME-BIFURCATED** (Wins balanced low-$n/k$; falsified on Zipf / high-$d$) | **`-1.38%` / `-8.82%`** test MSE on `low_nk32_gmm_t3`, vs. `+0.81%` on `zipf_gmm_t5` & `+0.12%` boundary overfit on `r18_512_t5`. |
| **`H-Algo-4`** | Scale-Invariant Top-2 Power-Mean Soft EMA Warmup (`H4_*_power_top2`) | **MANIFOLD-VS-DISCRETE BIFURCATION** | **`-0.06%` to `-0.14%`** on real `cifar10` vision manifolds, vs. `+0.93%` to `+4.05%` on discrete GMMs. |
| **`H-Algo-3`** | Coherence-Gated Nesterov Centroid Lookahead (`H3_*_nesterov_m035`) | **FALSIFIED** (`0/12` significant test-MSE wins) | $p_W \ge 0.0554$ across all `24` `(cell, mode)` pairs. |
| **`N-Track-1`** | `head_edit` $P/Q$-Confirmed Signed-Hadamard Transplant (`N1_*_head_edit`) | **CONFIRMED** (`5/5` multimodal GMMs; **`0/360` false transplants** on isotropic) | **`-5.810%`** `e-means++` / **`-6.090%`** `e-means` (`multiscale_rare_t5`), rare-mode recall `49.80% -> 57.78%`. |
| **`N-Track-5`** | `speedrun` 16-Shell Stratification + $2\times$ `wide_step` (`N5b` / `N5`) | **SPEEDUP CONFIRMED (`24/24` pairs)**; `strat_only` MSE falsified | **`-21.4%` to `-35.2%`** `train_time_sec` (`N5b_wide_only`), non-inferior `test_mse` on `10/12` `e-means++` cells. |
| **`N-Track-2`** | `quad_merge` $2.5\%$ Internal `x_train` Parabolic Merge (`N2_*_quad_merge`) | **PARTIALLY CONFIRMED** (`12/12` vs `slow_ema`; `2.5%` holdout tax on low-$n/k$) | `-1.96%` to **`-8.38%`** vs `slow_ema`; `-0.05%` to `-0.12%` on `e-means` `cifar10`; falsified on low-$n/k$ discrete `e-means++`. |
| **`N-Track-3`** | `ANVIL II` Quintic Gram-Newton-Schulz Polar Step (`N3_*_anvil_b025`) | **FALSIFIED on anisotropic (`stretched_a2`); SURPRISE `4/4` `zipf_gmm` WIN** | `+0.01%..+0.03%` on `stretched_a2`; **`-0.25%..-0.56%`** across all `4/4` `zipf_gmm` arms. |
| **`N-Track-4`** | `Dust` Split-Batch Antithetic Signed-Hadamard Jitter (`N4_*_dust_s015`) | **FALSIFIED** (`0/24` significant `test_mse` wins) | Neutral across `22/24` pairs (`-0.21%` to `+0.17%`, $p \ge 0.18$). |

### Phase 2B: Blackwell `SM120` Kernels & Multi-GPU Scaling (`300W Max-Q` / `450W Workstation` / `600W Server`)

| ID | Kernel / System Optimization | Confirmed Peak Speedup & Throughput | Honest Saturation / Falsified Boundary | Held-Out FP64 Parity |
| :--- | :--- | :--- | :--- | :--- |
| **`H-Opt-1`** | Split-$K$ + `FULL_D` + 3-Stage SM120 TMA (`blackwell_triton_assign`) | **`16.5x–19.0x`** Split-$K$, **`1.66x–1.73x`** `FullD`, **`317.8 TF/s`** TMA BF16 | `0.95x–1.07x` at `B=32768, D>=512` (baseline already SM-saturated); naive `BLOCK_D=64` (`num_stages=1` under `99 KB` SMEM) `0.61x–0.76x` | **`0.000%`** argmin disagree (`30/30`) |
| **`H-Opt-2`** | Native `float8_e4m3fn` + Appendix I.1 Auto-Centering (`fp8_assign`) | **`636.4 TF/s` (`81.8%–84.2%` peak)**; **`1.65x–1.86x`** vs BF16; **`2.22x–3.39x`** 1-GPU `n=1M..10M` | Uncentered FP8 fails (`-0.22%` MSE at `d=96`); `0.76x–0.85x` vs `FastBF16` at small `n<=250k` (`~2.5–6 ms` calibrate/encode cost) | **`-0.014%` to `+0.078%`** |
| **`H-Opt-3`** | Fused Scatter+EMA+Prep & CUDA-Graph Greedy $k$-means++ | **`1.85x–2.04x`** EMA step (`B<=4096`); **`1.75x–3.98x`** FP32 & **`2.07x–4.54x`** BF16 `k-means++` | EMA tapers to `1.45x–1.63x` at `B=32768`; `CUDAGraph` tapers to `1.00x` vs eager Triton at HBM-bound `m=65536` | **`0.000%` to `+0.049%`** |
| **`H-Opt-4`** | Sync-Free Inverted-List Tensor-Core `IVFAssigner` | **`6.44x–7.44x`** vs Base (`3.71x–4.54x` vs `FastExact`) at `k=65536` (`53.9 ms`) | `0.83x–0.88x` vs `FastExact` at `k=16384, nprobe=32` (`25%` buckets probed) | **`-0.184%` to `-0.269%`** *(net gain)* |
| **`H-Opt-5`** | Zero-Copy Streaming & 2-GPU NCCL `MultiGPUFastEMeans` | **`400.7 ms`** 1-GPU & **`297.1 ms`** 2-GPU at `n=10M, k=16384` (**`1.914x` = `95.7%` eff.**, **`5.48x` total**) | `0.42x` (`21.0%` eff.) at tiny comm-bound `n=200k, k=1024` (`~0.6 ms` PCIe NCCL > `<0.4 ms` GEMM) | **`-0.053%` to `+0.024%`** |

---

## Reproducing Benchmarks

Every runner supports a fast `--smoke` gate in addition to full multi-seed / multi-GPU execution:

```bash
# Phase 1: e-means reproduction & e-means++ experiments
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_phase1_emeans_pp.py --smoke

# Phase 2A Classical (H-Algo-1..5) & Forward-Only Noise (N-Track-1..5)
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/run_phase2_algo.py --smoke --out-dir /tmp/p2_algo_smoke
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/run_phase2a_noise.py --smoke --out-dir /tmp/p2_noise_smoke

# Phase 2B Blackwell SM120 Roofline, Kernel & Multi-GPU Scaling Suite
CUDA_VISIBLE_DEVICES=2,3 .venv/bin/python scripts/run_phase2_opt.py --smoke --out /tmp/p2_opt_smoke.json --plot-dir /tmp
```
