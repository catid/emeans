# Mathematical & Systems Dissection of `s-means` vs. `e-means` & `e-means++` / `SuperEMeans`

## 1. Executive Overview

[`suayptalha/s-means`](https://github.com/suayptalha/s-means) (*"s-means: Scale- & Distortion-Aware Single/Few-Epoch Clustering"*, Kocabay et al., 2026) is an Euclidean squared-$L_2$ streaming mini-batch clustering algorithm built directly on top of Meta FAIR's [`e-means`](https://arxiv.org/abs/2611.09501) (Bessa et al., 2026). Note that the `"s"` in `s-means` stands for **Scale- & distortion-aware** (not spherical/cosine): all distances, updates, and evaluations in `s-means` optimize exact Euclidean quantization distortion $\frac{1}{n}\sum_{i=1}^n \min_{j \in [k]} \|\mathbf{x}_i - \mathbf{c}_j\|_2^2$.

`s-means` retains `e-means`'s dimension-aware mini-batch scaling law (`quality` for $d \le 16$, `throughput` for $d > 16$), Robbins-Monro step-size power law $\alpha_0(n, k, B, T)$, represented-norm BF16 Tensor-Core distance evaluation, and count-weighted EMA centroid update, while modifying **seven algorithmic mechanisms** (`C1`–`C7`).

This document provides:
1. Complete mathematical equations for every component of `suayptalha/s-means` (verified line-by-line against `smeans/{_recipe,_cpu,_gpu,_triton_kernels}.py`),
2. Direct side-by-side comparison against `EMeans`, `EMeansPP`, and `SuperEMeans`,
3. Four subtle mathematical/numerical failure modes uncovered in upstream `s-means` source code, and
4. Systems/kernel optimizations implemented in [`smeans.py`](file:///home/catid/emeans/src/emeans/smeans.py) (`1.47x–1.51x` faster warm GPU execution on Blackwell SM120 `(200k, 128, 2048)`, zero host-device `.item()` sync stalls, zero cold-JIT specialization leaks, and exact coordinate-shift equivariance).

---

## 2. Component-by-Component Mathematical Comparison

| Pillar / Stage | Vanilla `e-means` (`EMeans`) | `suayptalha/s-means` (`SMeans`) | `e-means++` / `SuperEMeans` |
| :--- | :--- | :--- | :--- |
| **Geometry** | Euclidean $L_2^2$ (represented-norm BF16) | Euclidean $L_2^2$ (represented-norm BF16) | Euclidean $L_2^2$ (centered represented-norm BF16/FP8) |
| **Batch & $\alpha_0$ Law** | `quality` ($d \le 16$) / `throughput` ($d > 16$) power law | Identical power-law constants (`_recipe.py:54-67`) | Identical power-law constants + warm $\alpha_0$ scale (`0.85`) + optional `2x` `wide_step` |
| **C1: Seeding** | Uniform random rows `randperm(n)[:k]`; $N_j^{(0)} = 2^{-16}$ | Uniform subsample $m=\max(4k, 512)$, 4 batched multinomial rounds $\propto (d_{\min}^2)^{1.5} = d_{\min}^3$; unpolished; $N_j^{(0)} = 2^{-16}$ | $O(nd)$ $D^2$ Lightweight Coreset ($m=\max(16k, 8192)$) + Block-Greedy `k-means++` + **1-step Voronoi Polish** + **Prior-Calibrated Mass** $N_j^{(0)} = 0.5 B \hat{p}_j$ |
| **C2: Step Schedule & Low-$d$ Revival** | $\alpha_t = \alpha_0 [0.10 + 0.90 \frac{1+\cos(\pi f_t)}{2}]$; revival off for $d \le 2$ | **Power-cosine**: $\alpha_t = \alpha_0 [0.06 + 0.94 (\frac{1+\cos(\pi f_t)}{2})^{0.92}]$; revival enabled for **all $d \ge 1$** | Cosine schedule (`0.10` floor); revival gated by Huygens removal cost |
| **C3: Revival Window & Rare-Mode Shield** | Every 5 steps through $100\%$ of budget; **no rare-mode shield** | Every 5 steps up to **$85\%$ of budget** (`revive_stop_frac=0.85`); **no rare-mode shield** | Every 5 steps after grace window; **Huygens Parallel-Axis Rare-Mode Shield** protects isolated rare clusters ($M_j \min_{m \ne j}\|\mathbf{c}_j - \mathbf{c}_m\|_2^2 \ge \gamma \bar{E}$) + $P/Q$ `head_edit` transplants |
| **C4: Donor Selection** | Gumbel-top-$k$ weighted by $\text{usage}_j$ | Gumbel-top-$k$ weighted by $\text{sse\_ema}_j \cdot \text{usage}_j^{0.35}$, with fixed $\text{sse\_ema} \leftarrow 0.85\,\text{sse\_ema} + 0.15\,\text{SSE}_B$ | Gumbel-top-$k$ weighted by $\alpha_t$-coupled distortion $E_j \leftarrow (1-\alpha_t)E_j + \alpha_t\,\text{SSE}_B$ |
| **C5: Split Perturbation** | Multiplicative checkerboard $\mathbf{c}_s (1 \pm 10^{-4}\mathbf{s})$ (breaks at $\mathbf{0}$ and explodes under shifts!) | **Additive local-RMS Rademacher**: $\pm 0.55 \sqrt{\frac{\text{sse\_ema}_s}{\text{usage}_s \cdot d}} (\boldsymbol{\varepsilon}_s \odot \hat{\mathbf{f}})$ using online $O(Kd)$ inter-cluster feature std $\hat{\mathbf{f}}$ | **Online Oja Principal-Axis Split** along within-cluster top eigenvector $\pm \eta \hat{\sigma}_{s,1} \mathbf{v}_{s,1}$ + represented-ULP collapse separation |
| **C6: Closing-Pass Batch Weights** | Uniform $w_{\text{pass}} = 1$ across final epoch | **Progressive linear ramp** $w_{\text{pass}}(s) = 0.65 + 0.70 \frac{s + 0.5}{S_{\text{last}}} \in [0.65, 1.35]$ | Uniform $w_{\text{pass}} = 1$ across final epoch |
| **C7: Finalization** | Pure closing-pass mean $\bar{\boldsymbol{\mu}}_k = \mathbf{S}_k / n_k$ (or `slow_ema`) | **Scalar-prior EB toward live EMA**: $\frac{\tilde{n}_k}{\tilde{n}_k + 0.20}\bar{\boldsymbol{\mu}}_k + \frac{0.20}{\tilde{n}_k + 0.20}\mathbf{c}_k^{\text{EMA}}$ | **Positive-Part James-Stein / Empirical-Bayes** (`H-Algo-5`) calibrating $\kappa_j$ from empirical drift $\|\bar{\boldsymbol{\mu}}_j - \mathbf{c}_j^{\text{slow}}\|_2^2$ toward `slow_ema` |

---

## 3. Exact Equations of `s-means` & Subtle Source-Code Findings

### 3.1 C1 — Multi-Round Subsampled $(D^2)^{1.5} = D^3$ Seeding (`smeans/_recipe.py:99-139`)
Given $X \in \mathbb{R}^{n \times d}$, `s-means` draws a uniform random subset $X_S$ of size $m = \min(n, \max(4k, 512))$ without replacement, seeds $k_0 = \min(k, \max(8, \lfloor k / 8 \rfloor))$ initial centers uniformly, and runs $R - 1 = 3$ batched rounds (`step_k = ceil((k - k0) / 3)`):
$$d_{\min}^{(r)}(i) = \min_{\mathbf{c} \in C^{(r)}} \|\mathbf{x}_i - \mathbf{c}\|_2^2, \qquad p_i^{(r)} \propto \max\!\big(d_{\min}^{(r)}(i),\, 10^{-12}\big)^{1.5} + 10^{-12}.$$

**Critical Source-Code Findings:**
1. **$D^3$ vs. $D^{1.5}$ Exponent**: Because `_recipe.py:123` computes *squared* Euclidean distance $d_{\min} = \|\mathbf{x}_i - \mathbf{c}\|_2^2$ and `_recipe.py:131` raises `dmin ** 1.5`, the sampling distribution is actually proportional to $\|\mathbf{x}_i - \mathbf{c}\|_2^{3.0}$ ($D^3$), *not* $D^{1.5}$ as described in their README prose. Higher exponents ($D^3 > D^2$) aggressively favor extreme outliers/boundary points, amplifying high-dimensional spherical-shell initialization bias when no Voronoi polish follows.
2. **Step-0 Centroid Erasure**: Despite paying for multi-round seeding, `s-means` initializes `usage` to $2^{-16}$ (`_gpu.py:135`). On step $t = 0$, the EMA interpolation weight is $w_k = \frac{\alpha_0 n_k}{(1-\alpha_0)2^{-16} + \alpha_0 n_k} \approx 1.0$, immediately overwriting every touched seed with noisy single-mini-batch means (the exact Table `init_scale` pathology diagnosed by `EMeansPP`'s prior-mass calibration).
3. **Uniform Subsample Blindness on Rare Modes**: An unweighted uniform subsample of size $4k$ has probability $(1 - p_{\text{rare}})^{4k} \approx e^{-4 k p_{\text{rare}}}$ of missing a rare cluster completely (e.g., $67\%$ miss rate when $p_{\text{rare}} = 0.1/k$), whereas `EMeansPP`'s $D^2$-sensitive lightweight coreset ($m = \max(16k, 8192)$) samples isolated rare modes with high probability.
4. **Uncentered FP32 Cancellation & Scale Overflow**: Upstream computes `xsn[:, None] + c0n[None, :] - 2.0 * (xs @ c.t())` in uncentered FP32 (cancelling catastrophically under global shifts `x + 10^4`) and computes `dmin ** 1.5` without scale normalization (overflowing FP32 `3.4e38` to `inf` when feature scales exceed $10^{13}$). In [`smeans.py`](file:///home/catid/emeans/src/emeans/smeans.py), default mode (`exact_ref=False`) centers `xs` in FP64 and normalizes `dmin / max(dmin)` before exponentiation.

### 3.2 C2 & C3 — Power-Cosine Schedule & 85% Revival Freeze (`smeans/_gpu.py:155,207`)
At step $t \in \{0, \dots, T_{\text{steps}} - 1\}$ with $f_t = \min(t, T_{\text{steps}}) / \max(1, T_{\text{steps}})$:
$$\alpha(t) = \alpha_0 \left[0.06 + 0.94 \left(\frac{1 + \cos(\pi f_t)}{2}\right)^{0.92}\right].$$
Dead-cluster revival checks run every 5 steps **only while** $t \le \lfloor 0.85\,T_{\text{steps}} \rfloor$. Freezing revival over the final $15\%$ of steps is genuinely beneficial: in vanilla `e-means`, splits firing near $t \approx T_{\text{steps}} - 1$ perturb converged centroids right at the end of the closing pass with zero steps left to separate and equilibrate. However, because `s-means` lacks `EMeansPP`'s Huygens removal-cost shield, any genuine isolated rare mode with mass $p_j < 0.1 / k$ is destroyed during the first $85\%$ of training.

### 3.3 C4 & C5 — SSE $\times$ Usage$^{0.35}$ Donors & Local-RMS Anisotropic Rademacher Split (`smeans/_gpu.py:212-313`)
Every batch, `s-means` updates per-cluster quantization SSE with a **fixed** decay $\beta_{\text{sse}} = 0.85$:
$$\text{sse\_ema}_j \leftarrow 0.85\,\text{sse\_ema}_j + 0.15\,\sum_{i \in B_j} \|\mathbf{x}_i - \mathbf{c}_j\|_2^2,$$
and updates an $O(Kd)$ inter-cluster feature variance proxy ($\beta_{\text{var}} = 0.85$):
$$\hat{\boldsymbol{\mu}}_B = \frac{\sum_j \mathbf{s}_j}{\max(1, \sum_j n_j)}, \qquad \mathbf{v}_{\text{cur}} = \max\!\left(10^{-12},\; \frac{\sum_{j=1}^k (\mathbf{s}_j - n_j \hat{\boldsymbol{\mu}}_B)^2}{\max(1, \sum_j n_j)}\right), \qquad \text{feat\_scale} \leftarrow 0.85\,\text{feat\_scale} + 0.15\,\mathbf{v}_{\text{cur}}.$$
Note that $\sum_j (\mathbf{s}_j - n_j \hat{\boldsymbol{\mu}}_B)^2 / B = \frac{1}{B}\sum_j n_j^2 (\bar{\mathbf{x}}_j - \hat{\boldsymbol{\mu}}_B)^2$ is strictly **translation-invariant** (cancelling global coordinate shifts), though quadratically weighted by cluster batch count $n_j^2$.

On revival steps, donors are sampled via Gumbel-top-$k$ with score $\text{sse\_ema}_j \cdot \text{usage}_j^{0.35}$ ($\approx N_j^{1.35}\sigma_j^2$), and split additively along normalized feature scale $\hat{\mathbf{f}} = \sqrt{d}\,\sqrt{\text{feat\_scale}} / \|\sqrt{\text{feat\_scale}}\|_2$ with Rademacher signs $\boldsymbol{\varepsilon}_s \in \{-1, +1\}^d$:
$$\boldsymbol{\Delta}_s = 0.55 \sqrt{\max\!\left(10^{-10},\; \frac{\text{sse\_ema}_s}{\max(\text{usage}_s, 10^{-6}) \cdot d}\right)} \, (\boldsymbol{\varepsilon}_s \odot \hat{\mathbf{f}}), \qquad \mathbf{c}_{\text{tgt}}, \mathbf{c}_s \leftarrow \mathbf{c}_s \pm \boldsymbol{\Delta}_s.$$

**Why C5 Dramatically Beats Vanilla `e-means` — and How It Compares to Oja (`H-Algo-1`):**
- **Major win over vanilla `e-means`**: Vanilla `e-means` splits via $\mathbf{c}_s (1 \pm 10^{-4}\mathbf{s})$, which (a) collapses at $\mathbf{c}_s = \mathbf{0}$, (b) underflows BF16 mantissa (`2^-7 = 7.8e-3 >> 1e-4`), (c) explodes under coordinate shifts, and (d) leaves donor and child virtually superimposed so points take many mini-batches to separate. By contrast, `s-means` displaces donor and child by $\pm 0.55\,\hat{\sigma}_{\text{rms}}$, immediately bisecting the Voronoi cell on the **very next step** — explaining the bulk of `s-means`'s 1-epoch advantage over vanilla `e-means`.
- **Two theoretical limitations vs. `SuperEMeans`**:
  1. *Horizon mismatch*: `sse_ema` uses fixed rate $1 - 0.85 = 0.15$, whereas `usage` uses scheduled rate $\alpha(t)$! When $\alpha(t) \ll 0.15$ (or $\alpha(t) \gg 0.15$), $\text{sse\_ema}_s / \text{usage}_s$ divides two EMAs with mismatched effective sample sizes (toggleable via `couple_sse_to_alpha=True` in `SMeans`).
  2. *Random Rademacher orientation vs. within-cluster Oja eigenvector*: `feat_scale` captures only *global coordinate-axis* between-cluster variance, and random Rademacher signs $\boldsymbol{\varepsilon}_s$ have expected squared cosine $\frac{1}{d}$ with a cluster's true principal axis $\mathbf{v}_{s,1}$ when clusters are rotated off the coordinate axes — whereas `H-Algo-1` Oja power iteration tracks each cluster's actual top covariance eigenvector $\mathbf{v}_{s,1}$ directly. Combining **Oja's direction $\mathbf{v}_{s,1}$** with **`s-means`'s $0.55\,\hat{\sigma}$ chord scale** is a natural hybrid candidate tested in Phase 3.

### 3.4 C6 & C7 — Progressive Final-Pass Weighting & Empirical-Bayes EMA Shrinkage (`smeans/_gpu.py:208-209,318-334`)
In the closing epoch ($S_{\text{last}}$ steps), step $s \in \{0, \dots, S_{\text{last}} - 1\}$ receives weight:
$$w_{\text{pass}}(s) = (1 - 0.35) + 0.70\,\frac{s + 0.5}{S_{\text{last}}} \in [0.65,\, 1.35],$$
accumulating $\tilde{\mathbf{S}}_k = \sum_s w_{\text{pass}}(s)\mathbf{s}_{k,s}$ and $\tilde{n}_k = \sum_s w_{\text{pass}}(s)n_{k,s}$. At termination, clusters with $\tilde{n}_k > 0$ blend $\bar{\boldsymbol{\mu}}_k = \tilde{\mathbf{S}}_k / \tilde{n}_k$ with the live EMA centroid $\mathbf{c}_k^{\text{EMA}}$:
$$\mathbf{c}_k^{\text{final}} = \frac{\tilde{n}_k}{\tilde{n}_k + 0.20}\,\bar{\boldsymbol{\mu}}_k + \frac{0.20}{\tilde{n}_k + 0.20}\,\mathbf{c}_k^{\text{EMA}}.$$
Note that $\tau_{\text{eb}} = 0.20$ is quite small (for $\tilde{n}_k = 5$, shrinkage weight on $\mathbf{c}_k^{\text{EMA}}$ is $0.20 / 5.20 = 3.8\%$), and shrinks toward the *fast* live EMA rather than a variance-reduced `slow_ema` anchor with data-driven James-Stein $\hat{\kappa}_j$ (`H-Algo-5`).

---

## 4. GPU Systems Bottlenecks Fixed in `src/emeans/smeans.py`

Auditing `suayptalha/s-means/smeans/_gpu.py` revealed five GPU bottlenecks and numerical hazards fixed in our default fast pipeline (`exact_ref=False`):

1. **Eliminated Host-Device Synchronization Stalls**: Upstream `_gpu.py:271,275` calls `.tolist()` and `.item()` inside the training loop every 5 steps (`torch.stack((unused.sum(), donors.sum())).tolist()` and `sse_ema.max().item()`), draining the CUDA stream. [`smeans.py`](file:///home/catid/emeans/src/emeans/smeans.py) computes donor/target rankings entirely on device via `torch.argsort` and passes the 0-D GPU tensor `n_sp_t` directly into `_smeans_syncfree_revive_split_kernel`, achieving **zero host-device syncs across the entire training run**.
2. **Eliminated Gathered Mini-Batch & Transpose Allocations**: Upstream `_gpu.py:179,189-201` allocates `c_rep.t().contiguous().t()` every step and materializes `x_in[sub_idx]` in BF16 and again in FP32. Our `_smeans_fast_assign_kernel` and `_smeans_scatter_stats_sse_kernel` gather `row_indices` directly in registers and fuse `sums`, `counts`, `batch_sse`, and `ep_sq_dist` atomic accumulation into one pass.
3. **Fused EMA + SSE + Pass + Centroid-Prep Kernel**: `_smeans_ema_sse_pass_prep_kernel` updates `centroids`, `usage`, `sse_ema`, `pass_sums`, `pass_counts`, zeroes `sums`/`counts`/`batch_sse` in-place, and writes represented `c_rep` and `c_norm` for the next step in a single launch.
4. **Zero Cold-JIT Specialization Leaks**: Decorated Triton kernels with `do_not_specialize` on dynamic batch counts, epoch indices, RNG seeds, and pass flags so a single 1-epoch warmup compiles the exact kernel binaries used across all fractional and integer epoch budgets (`T in {1.0, 1.5, 2.0, 3.0, 5.0}`).
5. **Fused Zero-Copy Coordinate Centering & Inward ULP Split Guard**: Subtracts sampled dataset mean `origin` in registers *before* casting to `bf16`/`fp16` while keeping centroids in the centered frame, preserving exact Appendix I.1 represented-norm consistency and full precision even under `+20,000` coordinate shifts or near `FP16_MAX`.

### Warm GPU Latency on NVIDIA RTX PRO 6000 Blackwell (`n = 200,000, d = 128, k = 2048`, 5-Run Median)

| Epoch Budget $T$ | Upstream `suayptalha/s-means` (`triton`) | `emeans.smeans.SMeans` (`exact_ref=False`) | Speedup |
| :---: | :---: | :---: | :---: |
| $T = 1.0$ | `9.76 ms` | **`6.62 ms`** | **`1.47x`** |
| $T = 2.0$ | `10.19 ms` | **`6.93 ms`** | **`1.47x`** |
| $T = 3.0$ | `14.92 ms` | **`9.85 ms`** | **`1.51x`** |
| $T = 5.0$ | `16.11 ms` | **`10.99 ms`** | **`1.47x`** |
