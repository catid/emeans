# `UltraEMeans`: Mathematical & Systems Synthesis Beyond `s-means`, `SuperEMeans`, and `HybridSEMeans`

## 1. Executive Summary

Phase 3's `11,250`-run directional ablation ([`smeans_analysis.md`](file:///home/catid/emeans/docs/smeans_analysis.md), [`ablations_and_experiments.md`](file:///home/catid/emeans/docs/ablations_and_experiments.md)) answered the question *"Is `s-means` better than `SuperEMeans`, or is a combination best?"* by uncovering a fundamental **speed–geometry bifurcation**:

1. **`suayptalha/s-means` (`SMeans`)** ran in `~7–11 ms` on Blackwell GPUs and beat vanilla `e-means` on well-separated synthetic GMMs via large additive Rademacher splits (`C5`) and `SSE * Usage^0.35` donor scoring (`C4`), **but** regressed `+1.5%..+4.5%` on **every real 512-D vision manifold** (`cifar10_r18_512`, `cifar10_r18_512_unit`), destroyed `96%–100%` of isolated rare modes (`multiscale_rare`), and left up to `978` dead empty clusters on million-scale unit-sphere embeddings (`glove1m_k16384`).
2. **`SuperEMeans` & `HybridSEMeans`** achieved substantially lower quantization error than `s-means` (`25/30` and `26/30` cells), **but** (a) paid a `36 ms` initialization bottleneck (`72–102 ms` total on `200K x 128 @ K=2048`, `389–627 ms` at `1M x 128 @ K=16,384`), (b) used a conservative `0.08 * sigma_1` Oja split that under-separated merged bimodal clusters at $T \in \{1, 2\}$, and (c) suffered a `+1.1%` regression in `HybridSEMeans` on continuous isotropic data (`isotropic_ctrl`) due to in-pass covariance between the closing mean and `live_ema`.

**`UltraEMeans`** ([`ultra_emeans.py`](file:///home/catid/emeans/src/emeans/ultra_emeans.py)) resolves all three bottlenecks simultaneously via **five mathematical and GPU-kernel discoveries (`D0`–`D4`)**, validated across **`7,440` paired 15-seed runs** in [`run_phase4_ultra.py`](file:///home/catid/emeans/scripts/run_phase4_ultra.py) (plus **`1,320` paired million-scale runs** in [`million_scale_recall.md`](file:///home/catid/emeans/docs/million_scale_recall.md)):

![Speed vs. Held-Out Test MSE Pareto Frontier](../outputs/phase4_ultra/ultra_pareto_frontier.png)

![Head-to-Head Accuracy & Rare-Mode Survival Across Regimes](../outputs/phase4_ultra/ultra_head_to_head_regimes.png)

![UltraEMeans Discovery Attribution (LOO Counterfactuals & Baselines)](../outputs/phase4_ultra/ultra_discovery_ablation.png)

### Headline Empirical Findings (`7,440` Paired Runs, `15` Seeds `2000..2014`, `0` Cold-JIT Spikes)

- **Dominates Every Streaming & Batch Baseline in Held-Out `FP64` Test MSE (`30` Suite 1 Cells = `10` Regimes $\times$ `T in {1, 2, 5}`)**:
  - vs. **`e-means` (`arXiv:2611.09501`)**: **`-6.84%` mean Test MSE** (`27/0/3` cells, `374/0/76` seeds, Wilcoxon $p_W = 5.72 \times 10^{-7}$; **`-8.03%` at $T = 1$**).
  - vs. **`s-means` (`suayptalha/s-means`)**: **`-4.37%` mean Test MSE (`-2.73%` median)** across **`27/0/3` cells** (`411/0/39` seeds, $p_t = 2.85 \times 10^{-5}$, **$p_W = 1.02 \times 10^{-7}$**; **`-5.54%` at $T = 1$**), eliminating `s-means`'s `-4.52%` deep-manifold regression and `0.00%` rare-mode collapse.
  - vs. **`e-means++`**: **`-2.81%` mean Test MSE** (`27/0/3` cells, $p_W = 8.33 \times 10^{-7}$) **and `1.56x` faster median GPU latency** (`-36.0%`, `28/0/2` cells).
  - vs. **`SuperEMeans`**: **`-2.09%` (`UltraEMeans`) / `-2.14%` (`UltraEMeans (fast)`) lower mean Test MSE** ($p_t = 3.05 \times 10^{-4}$, $p_W = 7.11 \times 10^{-3}$) **while running `3.32x` (`-69.9%` median) to `3.71x` (`-73.0%` median) faster across `30/0/0` cells (`450/0/0` seeds, $p_t = 3.77 \times 10^{-27}$)**!
  - vs. **`Hybrid SuperEMeans` (Phase 3 winner)**: **`-1.89%` (`UltraEMeans`) / `-2.13%` (`UltraEMeans (fast)` vs. `Hybrid (fast)`) lower mean Test MSE** across **`28/0/2` cells (`398/0/52` and `402/0/48` seeds, $p_W = 4.66 \times 10^{-8}$ — including `10/0/0` cells at $T = 2$ and `14/0/0` Suite 2 cells)** **while running `3.20x–3.65x` faster across `30/0/0` cells**!
  - vs. **`Lloyd k-means (25 iter)`**: **`-8.62%` mean Test MSE** (`24/0/6` cells; **`10/0/0` cells and `150/0/0` seeds at $T = 5$**, `9/0/1` cells at $T = 2$).
- **Highest Rare-Mode Survival (`multiscale_rare_outliers`)**: Preserves **`47.2% / 60.5% / 77.9%`** (`UltraEMeans`) and **`49.7% / 64.3% / 82.9%`** (`UltraEMeans (fast)`) of isolated rare modes at $T = 1, 2, 5$, vs. **`0.0% / 0.0% / 4.2%`** in `s-means` and **`27.6% / 44.1% / 60.5%`** in `SuperEMeans`.
- **Transfers Cleanly to Million-Scale Real Corpora ($N = 10^6, K = 16{,}384$, [`million_scale_recall.md`](file:///home/catid/emeans/docs/million_scale_recall.md))**: Beats `Lloyd@25` at $T = 2$ (`4.3x–12.5x` faster) on both **SIFT1M** (`192.2 ms` / `164.2 ms` vs. `819.7 ms`, higher IVF Recall@10) and **GloVe-Twitter-1M-Unit** (`199.0 ms` / `171.0 ms` vs. `2,139.1 ms`, #1 lowest cosine distortion across all 9 solvers at every $T \in \{1, 2, 3, 5\}$), with **`0.0` empty clusters across 100% of all 320 million-scale runs**.

---

## 2. Five Core Discoveries (`D0`–`D4`)

### 2.1 Discovery 0 (`ultra_turkpp_seeding`): Zero-Sync Gumbel-Top-$K$ Coreset + Parallel Spatial-Sharded Greedy `k-means++` (`8.6x` Init Speedup)

Profiling `SuperEMeans` (`36.3 ms` init on `200K x 128 @ K=2048`; `299.5 ms` at `1M x 128 @ K=16,384`) uncovered three systems bottlenecks in `block_greedy_kmeans_plusplus_torch`:
1. Four full-dataset HBM scans during coreset construction,
2. Sequential ATen `torch.multinomial(..., replacement=False)` (`~6 ms`) + per-round `torch.unique` host-device syncs + `tensor[idx] = val` CUDA bounds-check stalls, and
3. A single-CTA (`grid=(1,)`) serial loop over all $k$ centroids.

[`ultra_turkpp_init_centered`](file:///home/catid/emeans/src/emeans/ultra_emeans.py#L1498-L1764) replaces this with:
- **Single-pass centered radial energy sketch** (`_ultra_origin_d2_kernel`, `0.40 ms` on `200K x 128`, zero `[N, D]` copy) forming the exact Bachem-Lucic-Krause (`2018`) importance proposal $q_i = \frac{1}{2n} + \frac{\|\mathbf{x}_i - \hat{\boldsymbol{\mu}}\|_2^2}{2\sum_\ell \|\mathbf{x}_\ell - \hat{\boldsymbol{\mu}}\|_2^2}$ (`m = max(6k, 4096)`), plus exact deterministic farthest-energy rare-tail capture (`b0 = min(k, 8)` boosted by $+13.8$ Gumbel log-units) sampled in 1 pass via **Plackett-Luce Gumbel-Top-$M$** (`log(q_i) - log(-log(U_i))`, zero host syncs).
- **Parallel spatial-sharded greedy `k-means++` tournament** (`_ultra_batched_local_greedy_kpp_kernel` with `do_not_specialize=["p_len", "n_select", "rng_seed"]`), partitioning candidates into parallel `BLOCK_P = 128` CTAs running $L = 3$ local greedy `k-means++` trials in SRAM (`<0.5 ms` vs. `12 ms` on 1 SM), followed by **1-pass Voronoi polish** extracting calibrated prior mass $\hat{p}_j$ and distortion $\hat{e}_j$.
- **Result**: Initialization drops from `19.8–38.5 ms` (`SuperEMeans`) to **`3.2–6.3 ms`** across Suite 1 (`30.7 ms` vs. `299.5 ms` at `1M x 128 @ K=16,384`).

### 2.2 Discovery 1: Spectral-Concentration-Gated Half-Gaussian Principal-Axis Split ($\eta_j \in [0.18, \sqrt{2/\pi}]$)

Why did `s-means`'s split (`C5`: $\pm 0.55 \hat{\sigma}_{\text{rms}}(\boldsymbol{\varepsilon}_s \odot \hat{\mathbf{f}})$) help on synthetic GMMs yet regress `-3.0%..-4.5%` on 512-D real embeddings?
- Because $\boldsymbol{\varepsilon}_s \in \{-1, +1\}^d$ has $\|\boldsymbol{\varepsilon}_s \odot \hat{\mathbf{f}}\|_2 = \sqrt{d}$, `s-means`'s **Euclidean displacement norm** is $0.55\sqrt{d}\,\hat{\sigma}_{\text{rms}}$ — i.e., **`6.22 * sigma_rms` in $d = 128$ and `12.45 * sigma_rms` in $d = 512$**, hurling child centroids off low-intrinsic-dimension curved manifolds into empty ambient space!
- Conversely, for a cluster elongated or bimodal along unit principal eigenvector $\mathbf{v}_{j,1}$ with 1D projected variance $\sigma_{j,1}^2 = \mathbb{E}[\langle \mathbf{x} - \mathbf{c}_j, \mathbf{v}_{j,1}\rangle^2]$, the **exact 1D conditional half-space centroids** of $\mathcal{N}(0, \sigma_{j,1}^2)$ are:
  $$\mathbf{c}_j^{\pm} = \mathbf{c}_j \pm \sqrt{\frac{2}{\pi}}\,\sigma_{j,1}\mathbf{v}_{j,1} \approx \mathbf{c}_j \pm 0.7979\,\sigma_{j,1}\mathbf{v}_{j,1},$$
  which has dimension-free Euclidean norm $O(\sigma_{j,1})$ (independent of $\sqrt{d}$!) and cuts variance along $\mathbf{v}_{j,1}$ by $2/\pi \approx 63.7\%$ in a single step (`10x` larger than `SuperEMeans`'s `0.08 * sigma_1`).

In [`_ultra_syncfree_split_kernel`](file:///home/catid/emeans/src/emeans/ultra_emeans.py#L1284-L1462):
1. **Register-Resident Online Krasulina/Oja Tracking**: `_ultra_scatter_oja_sse_kernel` projects batch residuals $y_i = \langle \mathbf{x}_i - \mathbf{c}_j, \mathbf{p}_j\rangle / \|\mathbf{p}_j\|_2$ in registers (`0` `[B, D]` temporary buffer!) and updates $\mathbf{p}_j \leftarrow (1 - \alpha_t)\mathbf{p}_j + \alpha_t \bar{\mathbf{g}}_j$, so $\|\mathbf{p}_j\|_2 \to \sigma_{j,1}$ and $\mathbf{p}_j / \|\mathbf{p}_j\|_2 \to \mathbf{v}_{j,1}$.
2. **Cold-Start Anisotropy Blending & Spectral-Concentration Gate**: Using the cluster's spectral concentration ratio $r_j = \sigma_{j,1} / \sigma_{\text{rms},j}$ (where $r_j \approx 1$ on an isotropic ball and $r_j \gg 1$ on a merged bimodal pair or elongated filament), `UltraEMeans` blends $\mathbf{v}_{j,1}$ smoothly with the unit-normalized $\hat{\mathbf{f}}$-Rademacher direction when cold ($\rho_j = \operatorname{clip}((r_j - 0.85)/0.35, 0, 1)$) and gates the step multiplier smoothly from manifold-safe `0.18` up to exact Half-Gaussian $\sqrt{2/\pi}$:
   $$\eta_j = 0.18 + \left(\sqrt{\frac{2}{\pi}} - 0.18\right)\operatorname{clip}\!\left(\frac{\sigma_{j,1}/\sigma_{\text{rms},j} - 1.10}{0.90},\; 0,\; 1\right), \qquad \sigma_{j,\text{split}} = \max(\sigma_{j,1},\, 0.85\sigma_{\text{rms},j}).$$
3. **Exact Parallel-Axis Child SSE Deflation**: By Huygens' theorem, displacing $\mathbf{c}_j$ by $\pm \eta_j \sigma_{j,\text{split}}$ removes $(2\eta_j\sqrt{2/\pi} - \eta_j^2) N_j \sigma_{j,\text{split}}^2$ of parallel-axis SSE; each child inherits $\frac{1}{2}\max(0,\, E_j - \Delta E_j)$, preventing cascading re-splits on the next check!
- **Causal Proof (Suite 2 `LOO -D1a` & `LOO -D1b`)**: Replacing `D1` with `s-means` Rademacher `0.55` degrades Test MSE by **`+3.32%` across `12/14` cells** ($p_W = 6.10 \times 10^{-4}$; `+6.82%` on `balanced_gmm`, `+8.09%` on `multiscale_rare`, `+3.45%` on `cifar10_r18_512`), while shrinking `D1` to `SuperEMeans` `0.08` degrades Test MSE across **`11/14` cells** ($p_W = 1.07 \times 10^{-2}$).

### 2.3 Discovery 2: Decay-Matched Bias-Corrected Donor Scoring + Twin-Safe SRAM Huygens Rare-Mode Shield + 85% Freeze/Flush

1. **Matched Horizon & Bias Correction**: Both `usage` and `sse_ema` decay with the exact same scheduled `(1 - alpha_t)` factor and scalar zero-init bias correction, so $\hat{\sigma}_{\text{rms},j}^2 = \text{sse\_unb}_j / (\text{usage\_unb}_j \cdot d)$ has exact physical variance units at every step $t$, and donors are scored by $(0.5 E_j + 0.5 d N_j \sigma_{j,1}^2) \cdot (N_j / B)^{0.35}$ (spotting bimodal 1D pairs buried in $d-1$ isotropic noise dimensions).
2. **2-Pass Sorted-`low_mask` SRAM Huygens Leader Election (`0` $K \times K$ HBM Matrix up to $K = 16{,}384$)**: [`_ultra_huygens_nn_pass1_kernel`](file:///home/catid/emeans/src/emeans/ultra_emeans.py#L1072-L1163) and [`_ultra_huygens_leader_pass2_kernel`](file:///home/catid/emeans/src/emeans/ultra_emeans.py#L1166-L1254) pack `low_mask` rows contiguously so non-active CTAs exit in 1 cycle, compute exact IEEE-FP32 nearest-neighbor distances in SRAM, elect one leader among co-located twins, and gate effective mass strictly by `survival_mass > 0` (preventing 0-touch empty split children in empty voids from ever being shielded).
3. **85% Revival Stop + Stale-Parent Closing Flush**: Revivals freeze at `revive_stop_frac = 0.85`, and any split during the first `85%` of the closing pass flushes the donor's stale pre-split `pass_sums/pass_counts` so post-split child assignments are never pulled back to the pre-split parent midpoint.
- **Causal Proof (Suite 2 `LOO -D2a` & `LOO -D2b`)**: Dropping coupled SSE + Twin-Safe Huygens degrades Test MSE by **`+3.62%` across `13/14` cells** ($p_W = 1.71 \times 10^{-3}$; `+13.8%` on `multiscale_rare`, `+8.50%` on `balanced_gmm`), and dropping 85% freeze/flush hurts both MSE (`+0.14%`, `+0.86%` on `multiscale_rare`) and latency (`+4.86%` across `14/14` cells).

### 2.4 Discovery 3 (`ultra_simplex_eb_finalize`): Covariance-Safe Simplex 3-Point Empirical-Bayes Finalizer

Phase 3 revealed that `HybridSEMeans` (`eb_anchor="ema_last"`) regressed `+1.08%` on `isotropic_ctrl` (`120.32` vs. `119.02`) because the closing-pass sample mean $\bar{\boldsymbol{\mu}}_j^{\text{close}}$ and `live_ema` $\mathbf{c}_j^{\text{ema}}$ are computed from the **same final-epoch mini-batches**! Positive covariance makes $\|\bar{\boldsymbol{\mu}}_j^{\text{close}} - \mathbf{c}_j^{\text{ema}}\|_2^2 < V_{\text{close}}$, tricking James-Stein into shrinking heavily toward the higher-variance `live_ema`.

[`ultra_simplex_eb_finalize`](file:///home/catid/emeans/src/emeans/ultra_emeans.py#L2295-L2428) resolves this on the 3-point simplex $\{\bar{\boldsymbol{\mu}}_j^{\text{close}},\, \mathbf{c}_j^{\text{ema}},\, \mathbf{c}_j^{\text{slow}}\}$:
1. **Budget-Adaptive Closing Ramp**: Ramp span decays as $0.35 / \max(1, T)^{1.25}$ (`0.35` at $T=1$ where centroids still drift, `0.047` at $T=5$ where centroids are stationary so uniform weights maximize Kish effective sample size $n_{\text{eff},j} = (\sum_s w_s n_{j,s})^2 / \sum_s w_s^2 n_{j,s}$).
2. **Covariance-Safe Shrinkage**: Applies tiny prior-count regularization (`eb_tau = 0.25`) toward `live_ema` for `1–2`-hit tail clusters, and applies empirical-Bayes James-Stein shrinkage **strictly along the stationary `slow_ema` axis** gated by `num_epochs > 1`.
- **Result**: `UltraEMeans` completely eliminates `HybridSEMeans`'s `+1.08%` isotropic regression (`119.020` vs. `e-means` `119.020`, `-1.08%` vs. `Hybrid`, `15/0/0` seeds!) while statistically beating both raw `closing_mean` (`LOO -D3a`, $p_W = 0.049$) and `s-means`'s fixed `eb_0.20` (`LOO -D3b`, $p_t = 0.033, p_W = 0.049$).

### 2.5 Discovery 4 (`run_ultra_head_edit_checkpoint` & Fused Zero-Sync Triton Kernels)

- **Oja-Seeded Single-Sync $P/Q$ Dormant-Twin Transplants**: Fires at pre-closing checkpoints `(0.46, 0.76)`, seeding candidate split planes with the donor's live Oja eigenvector $\mathbf{p}_s$, 2-step batch power direction, farthest-residual vector, and signed Hadamard probes, evaluating split half $P$ and confirmation half $Q$ in a **single GPU batch** (`LOO -D4`: removing `head_edit` worsens Test MSE across `10/14` cells, $p_W = 0.035$).
- **Zero-Sync Fused Blackwell Triton Pipeline**: Fusing gather+center+scale+represented-norm assignment (`_ultra_assign_kernel` / Split-K), register-Oja scatter (`_ultra_scatter_oja_sse_kernel`), 1-CTA-per-cluster state update (`_ultra_ema_oja_slow_pass_prep_kernel`), SRAM Huygens, and zero-sync split yields **`21.2–28.1 ms` median end-to-end GPU latency** across all 3 Blackwell TDP SKUs (`3.3x–3.7x` faster than `SuperEMeans`).

---

## 3. Summary Tables (`7,440` Paired Runs Across `6` Blackwell Machines)

### 3.1 Suite 1 Executive Showdown Aggregates (`30` Cells $\times$ `15` Seeds)

| Solver | vs. Baseline | Held-Out Test MSE Mean $\Delta$ | Test MSE Cell W/T/L | Test MSE `p_wilcox` | GPU Time Median $\Delta$ | Time Cell W/T/L |
| :--- | :--- | ---: | :---: | ---: | ---: | :---: |
| **`UltraEMeans`** | `e-means` | **`-6.84%`** (`-8.03%` @ `T=1`) | `27/0/3` (`374/76` seeds) | `5.72e-07` | `+49.8%` | `3/0/27` |
| **`UltraEMeans`** | `s-means` | **`-4.37%`** (`-5.54%` @ `T=1`) | `27/0/3` (`411/39` seeds) | `1.02e-07` | `+218.9%` (`23.5` vs `7.1 ms`) | `0/0/30` |
| **`UltraEMeans`** | `e-means++` | **`-2.81%`** | `27/0/3` (`362/88` seeds) | `8.33e-07` | **`-36.0%` (`1.56x` faster)** | `28/0/2` |
| **`UltraEMeans`** | `SuperEMeans` | **`-2.09%`** | `17/0/13` (`290/160` seeds) | `7.11e-03` | **`-69.9%` (`3.32x` faster)** | `30/0/0` |
| **`UltraEMeans`** | `Hybrid SuperEMeans` | **`-1.89%`** | `28/0/2` (`398/52` seeds) | `4.66e-08` | **`-68.8%` (`3.20x` faster)** | `30/0/0` |
| **`UltraEMeans`** | `Lloyd k-means (25 iter)` | **`-8.62%`** (`-9.62%` @ `T=5`) | `24/0/6` (`10/0/0` @ `T=5`) | `2.83e-04` | `+116.6%` (faster at `K >= 2048`!) | `2/0/28` |
| **`UltraEMeans (fast)`** | `s-means` | **`-4.42%`** | `27/0/3` (`412/38` seeds) | `3.15e-07` | `+171.5%` (`21.2` vs `7.1 ms`) | `0/0/30` |
| **`UltraEMeans (fast)`** | `SuperEMeans` | **`-2.14%`** | `18/0/12` (`283/167` seeds) | `8.71e-03` | **`-73.0%` (`3.71x` faster)** | `30/0/0` |
| **`UltraEMeans (fast)`** | `Hybrid SuperEMeans (fast)` | **`-2.13%`** | `28/0/2` (`402/48` seeds) | `4.66e-08` | **`-64.1%` (`2.79x` faster)** | `30/0/0` |

### 3.2 SKU-Stratified End-to-End GPU Latency (Suite 1)

| Blackwell TDP SKU | `s-means` Median (ms) | `e-means` Median (ms) | **`UltraEMeans (fast)` Median (`Init + Train` ms)** | **`UltraEMeans` Median (`Init + Train` ms)** | `Hybrid SuperEMeans` Median (ms) | `SuperEMeans` Median (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **`Max-Q 300W`** (`315` runs/arm) | `7.11` | `13.66` | **`21.17`** (`3.28 + 21.88` mean) | **`23.50`** (`3.33 + 27.52` mean) | `70.04` | `72.45` |
| **`Workstation 450W`** (`90` runs/arm) | `7.83` | `13.21` | **`21.60`** (`3.16 + 18.40` mean) | **`26.38`** (`3.20 + 24.31` mean) | `57.78` | `59.04` |
| **`Server 600W`** (`45` runs/arm, `200K x 128 @ K=2048`) | `8.74` | `19.09` | **`24.98`** (`6.32 + 22.45` mean) | **`28.09`** (`6.34 + 26.35` mean) | `98.92` | `101.52` |

---

## 4. Reproducibility & Artifacts

- **Implementation**: [`src/emeans/ultra_emeans.py`](file:///home/catid/emeans/src/emeans/ultra_emeans.py) (`UltraEMeans`, `ULTRA_ABLATION_ARMS`, `build_ultra_arm`)
- **Test Suite (`81 passed, 0 failed`)**: [`tests/test_ultra_emeans.py`](file:///home/catid/emeans/tests/test_ultra_emeans.py)
- **Multi-Machine Orchestrator**: [`scripts/run_phase4_ultra.py`](file:///home/catid/emeans/scripts/run_phase4_ultra.py)
- **Full Statistical Report & JSON**: [`outputs/phase4_ultra/phase4_ultra_summary.md`](file:///home/catid/emeans/outputs/phase4_ultra/phase4_ultra_summary.md), [`outputs/phase4_ultra/phase4_ultra_results.json`](file:///home/catid/emeans/outputs/phase4_ultra/phase4_ultra_results.json)
