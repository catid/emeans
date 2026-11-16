# Research History, Mathematical Failure Modes & Engineering Challenges

This document records the chronological research progression of `emeans`, the mathematical and geometric failure modes uncovered in standard and streaming vector quantization, mixed-precision floating-point traps (`FP8` / `BF16` / `FP16`), NVIDIA Blackwell `SM120` hardware/compiler subtleties, and all pre-experiment correctness/timing bugs caught during independent dual-verifier and patrol audits.

- **Headline results & plots**: [`../README.md`](../README.md)
- **Full component ablations & statistical tables**: [`ablations_and_experiments.md`](ablations_and_experiments.md)

---

## Contents

1. [Research History & Phase Progression](#1-research-history--phase-progression)
2. [Algorithmic & Mathematical Failure Modes Uncovered](#2-algorithmic--mathematical-failure-modes-uncovered)
3. [Mixed-Precision & Numerical Stability Traps (`FP8`, `BF16`, `FP16`, `FP32`)](#3-mixed-precision--numerical-stability-traps-fp8-bf16-fp16-fp32)
4. [Blackwell `SM120` Hardware, Triton JIT & GPU Synchronization Challenges](#4-blackwell-sm120-hardware-triton-jit--gpu-synchronization-challenges)
5. [Pre-Experiment Bug Audit Log & Distributed Fleet Hygiene](#5-pre-experiment-bug-audit-log--distributed-fleet-hygiene)

---

## 1. Research History & Phase Progression

1. **Phase 1 — Reproducing Meta FAIR's `e-means` & Designing `e-means++`**:
   - Reproduced `faiss.contrib.e_means.base` (`arXiv:2611.09501`) with exact universal batch/step scaling laws, cosine annealing floor (`0.1\alpha_0`), Gumbel-max donor revival, and Appendix I.1 represented-norm matching.
   - Diagnosed why naive `k-means++ + e-means` both stalls for hundreds of milliseconds/seconds on $O(nkd)$ seeding **and** destroys isolated rare clusters during EMA updates. Built `EMeansPP` around five coupled pillars (`lightweight_coreset`, Block-Greedy `k-means++`, 1-step Voronoi polish, prior-calibrated EMA mass $N_j^{(0)}$, and Huygens parallel-axis rare-mode protection + $D^2$ donor splitting).
2. **Phase 2A — Pre-Registered Classical (`H-Algo-1`..`5`) & Forward-Only (`qlabs-eng/noise` `N-Track-1`..`5`) Falsification (`7,200` Runs)**:
   - Tested 10 pre-registered hypotheses across 12 diverse synthetic and real vision-embedding cells (`15` paired seeds $\times$ `20` arms per campaign).
   - Identified 4 clear winners (`H-Algo-5` Bounded-Influence Empirical-Bayes / James-Stein finalization, `H-Algo-1` Online Oja Principal-Axis splitting + BF16 rounding fix, `N-Track-1` $P/Q$ forward-verified `head_edit` transplants, and `N-Track-5` late $2\times$ `wide_step` batch doubling) while cleanly falsifying Nesterov lookahead (`H-Algo-3`) and `Dust` Voronoi jitter (`N-Track-4`) and mapping sharp manifold-vs-discrete bifurcations for Hartigan (`H-Algo-2`), Power-Top2 (`H-Algo-4`), `quad_merge` (`N-Track-2`), and `ANVIL II` (`N-Track-3`).
3. **Phase 2B — NVIDIA Blackwell `SM120` Roofline, Centered-FP8, Sync-Free IVF & Multi-GPU NCCL**:
   - Measured empirical compute/bandwidth rooflines across three RTX PRO 6000 Blackwell TDP SKUs (`300W Max-Q`, `450W Workstation`, `600W Server`).
   - Engineered 2D Split-$K$ + register-resident `FULL_D` + 3-stage SM120 TMA kernels (`317.8 TFLOP/s` BF16), Appendix I.1 auto-centered `float8_e4m3fn` assignment (`636.4 TFLOP/s`, `84.2%` of GEMM peak), sync-free inverted-list Tensor-Core `IVFAssigner` (`7.44x` at $k=65{,}536$), and 2-GPU NCCL `MultiGPUFastEMeans` (`297.1 ms` on $n=10^7, k=16{,}384, d=128$, `95.7%` parallel efficiency).
4. **Phase 3 — Source-Level Dissection of `suayptalha/s-means` & Synthesis of `SuperEMeans` / `HybridSEMeans` (`11,250` Runs)**:
   - Dissected `suayptalha/s-means` (`C1`–`C7`), fixed its GPU `.tolist()`/`.item()` sync stalls for a `2.44x–2.65x` speedup (`smeans_fast`), fused all Phase 1–2 winners into zero-sync `SuperEMeans` (`P1`–`P6`), and ran a 40-arm bidirectional component grafting campaign across 6 Blackwell hosts.
   - Discovered the exact geometric split law separating discrete multimodal clusters (`HybridSEMeans` `0.55\sqrt{\text{MSE}_s}` chord along Oja $\mathbf{v}_{s,1}$ wins `15/15` GMM cells) from curved high-dimensional deep manifolds (where total-SSE chord steps overshoot manifold geometry and pure `SuperEMeans` `0.08\sigma_1` wins `12/12` real/continuous cells).

---

## 2. Algorithmic & Mathematical Failure Modes Uncovered

### 2.1 The $1 - e^{-1} \approx 63.2\%$ Poisson Coupon-Collector Seeding Trap
Drawing $k$ initial centroids uniformly at random from $k$ balanced ground-truth clusters places `Poisson(1)` seeds per mode, leaving $e^{-1} \approx \mathbf{36.8\%}$ of modes completely empty (`61.4%` empirical $T=0$ coverage on `balanced_gmm` at $k=1024$) while crowding multiple redundant twins into dense modes (`53.93` MSE vs. `21.33` oracle MSE). Full Lloyd (`25` passes) stalls permanently in twin-centroid local minima unless dead-cluster revival or $D^2$ coreset seeding intervenes.

![Phase 1 Mechanistic Traps](../outputs/phase1/fig2_mechanistic_traps.png)

### 2.2 The $2\times$ High-Dimensional Spherical-Shell Initialization Penalty
Standard `k-means++` (and `s-means`'s subsampled $D^3$ seeding) chooses **raw data vectors** $\mathbf{x}_{i_j} \in X$ as initial centroids. Inside a cluster with mean $\boldsymbol{\mu}_j$ and covariance $\Sigma_j$, a random point sits on the thin high-dimensional shell of radius $\sqrt{\operatorname{tr}(\Sigma_j)} \approx \sqrt{d}\sigma$, yielding expected initial distortion:
$$\mathbb{E}_{\mathbf{x}, \mathbf{x}' \sim P_j}\!\left[\|\mathbf{x} - \mathbf{x}'\|_2^2\right] = \mathbb{E}\|\mathbf{x} - \boldsymbol{\mu}_j\|_2^2 + \mathbb{E}\|\mathbf{x}' - \boldsymbol{\mu}_j\|_2^2 = 2\operatorname{tr}(\Sigma_j).$$
Running a **single weighted Voronoi averaging step** on the $O(nd)$ lightweight coreset pulls raw seeds from the shell to interior means before step 0, cutting $T=0$ test MSE on `balanced_gmm` by **`-39.2%`** (`47.43 -> 28.83`) in `< 2 ms`.

### 2.3 Step-0 Warm-Start Erasure (`N_0 = 2^-16`)
Both vanilla `e-means` and `suayptalha/s-means` initialize EMA cluster usage to $N_j^{(0)} = 2^{-16}$. On mini-batch step $t=0$, the convex-hull EMA interpolation weight is:
$$w_j^{(0)} = \frac{\alpha_0 n_j^{(0)}}{(1 - \alpha_0) 2^{-16} + \alpha_0 n_j^{(0)}} \approx 1.0,$$
which **completely overwrites** every carefully seeded centroid with a noisy mini-batch average of $B/k \approx 4\text{–}16$ points. Calibrating $N_j^{(0)} = 0.5 B \hat{p}_j$ from the coreset Voronoi partition preserves warm-started geometry seamlessly into step 0.

### 2.4 Count-Only Revival Rare-Mode Assassination
Vanilla `e-means` and `suayptalha/s-means` recycle any centroid with usage $N_j < 0.1 \bar{N}$ (`0.1 B / k`). Any genuine isolated rare/outlier mode with population share $p_r < 0.1 / k$ (e.g., $p_r = 0.03 / k$) falls below this count threshold and is **100% assassinated** (`0.0%` preserved in Phase 1 `Vanilla e-means` and `Naive k-means++ + e-means`; `0.00%` at $T=1,2$ and `4.18%` at $T=5$ in Phase 3 `s-means`). By Fritzke's utility theorem and Huygens' parallel-axis identity, removing cluster $j$ increases global SSE by at least $M_j^{\text{eff}} \min_{m \neq j}\|\mathbf{c}_j - \mathbf{c}_m\|_2^2$. Protecting clusters where this removal cost exceeds $\gamma_{\text{prot}} \bar{E}$ preserves **`86.8%` (`90.0%` median)** of rare modes in Phase 1 and **`60.5%–63.5%`** under aggressive Phase 3 hybrid splitting.

### 2.5 Warm-Started `slow_ema` Trajectory Lag
While `slow_ema` reduces out-of-sample variance by `-0.23%` on random-init `e-means`, applying `slow_ema` on top of warm-started `e-means++` caused **universal regression across `12/12` Phase 2A cells** (`+1.20%` to `+10.17%` worse train MSE; `+0.02%` to `+9.13%` worse test MSE) because `slow_usage` lags behind rapid fine-scale Voronoi equilibration and split transitions. `H-Algo-5` (`empirical_bayes_finalize`) solved this via positive-part James-Stein bounded influence: when empirical drift $\|\bar{\boldsymbol{\mu}}_j^{\text{close}} - \hat{\boldsymbol{\mu}}_j^{\text{slow}}\|_2^2 \gg \hat{\sigma}_j^2 / n_j$, shrinkage weight $\lambda_j \to 1$ automatically rejects the lagged anchor.

### 2.6 High-Dimensional Split Geometry: $O(1/\sqrt{d})$ Orthogonality & Total-SSE Manifold Overshoot
Three distinct geometric traps emerged in donor splitting across Phases 2A and 3:
1. **Arbitrary/Rademacher Split Orthogonality**: In dimension $d \in \{128, 512, 1024\}$, a fixed `[+1, -1, ...]` or random Rademacher vector has expected projection $O(1/\sqrt{d})$ onto the inter-cluster axis $\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2$ of two merged clusters ($\approx 87.5^\circ\text{–}88.2^\circ$ nearly orthogonal).
2. **Naive $O(R)$ Hadamard Probe Failure (`#55`)**: Early in Phase 2A `N-Track-1`, stepping $\pm 0.35 R$ along random unit Hadamard probes in $d \in \{128, 512\}$ added orthogonal variance $+\delta^2$ across $d-1$ coordinates exceeding the 1D projection gain $-2\delta R/\sqrt{d}$, yielding **`0` confirmed transplants on `11/12` cells**. Replacing fixed steps with **exact 1D conditional half-space centroids** ($\delta_m^\pm = \operatorname{mean}_{\pm p_{i,m} > 0}(p_{i,m})$) automatically scaled steps as $O(R/\sqrt{d})$ on isotropic noise and $\pm \Delta$ on bimodal separated modes, unlocking `5/5` multimodal GMM wins with `0/360` false positives on isotropic data.
3. **Total-SSE Chord Overshoot on Curved Deep Manifolds (Phase 3)**: `s-means` displaces split children by $0.55\sqrt{\text{SSE}_s / (N_s d)}\,\sqrt{d} = 0.55\sqrt{\text{MSE}_s}$, using **total $d$-dimensional cluster radius** $\sqrt{\text{MSE}_s}$. On 512-D ResNet-18 embeddings (`cifar10_r18_512`), total $\text{MSE}_s$ sums small variance across 512 curved dimensions; stepping $0.55\sqrt{\text{MSE}_s}$ along a single chord shoots centroids **off the manifold**, causing `s-means` to lose to *vanilla* `e-means` by **`+1.46%` to `+4.32%`** on all `9/9` ResNet-18 cells (`15/15` seeds each)!

### 2.7 Upstream `suayptalha/s-means` Source-Code Discrepancies
Line-by-line inspection of `suayptalha/s-means` uncovered three undocumented mathematical behaviors (preserved in `SMeans(..., exact_ref=True)` for bit-faithful parity and toggleable in `SMeans`):
- **$D^3$ vs. $D^{1.5}$ Seeding Exponent**: `smeans/_recipe.py:123,131` computes *squared* distance `dmin = ||x - c||^2` and raises `dmin ** 1.5`, sampling proportional to $\|\mathbf{x} - \mathbf{c}\|_2^{3.0}$ ($D^3$) rather than $D^{1.5}$, amplifying outlier shell bias without Voronoi polish.
- **Uniform `4k` Subsample Blindness**: Drawing $m = 4k$ unweighted rows misses rare clusters of mass $p_{\text{rare}} = 0.1/k$ with probability $(1 - p_{\text{rare}})^{4k} \approx e^{-0.4} = 67\%$.
- **Fixed `0.85` SSE Decay vs. Scheduled $\alpha(t)$ Horizon Mismatch**: `sse_ema` decays at fixed rate $0.15$ while `usage` decays at scheduled $\alpha(t)$, dividing two EMAs with mismatched effective sample horizons inside $\text{sse\_ema}_s / \text{usage}_s$.

---

## 3. Mixed-Precision & Numerical Stability Traps (`FP8`, `BF16`, `FP16`, `FP32`)

1. **Appendix I.1 Represented-Norm Mismatch**:
   Expanding $\|\mathbf{x} - \hat{\mathbf{c}}_j\|_2^2 = \|\mathbf{x}\|_2^2 + \|\hat{\mathbf{c}}_j\|_2^2 - 2\langle \mathbf{x}, \hat{\mathbf{c}}_j\rangle$ requires computing $\|\hat{\mathbf{c}}_j\|_2^2$ from the **exact quantized `BF16` / `FP16` / `FP8` centroid values** fed into `tl.dot`. Using pre-cast `FP32` $\|\mathbf{c}_j\|_2^2$ injects norm rounding bias $\|\mathbf{c}_j\|_2^2 - \|\hat{\mathbf{c}}_j\|_2^2$ exceeding inter-centroid differences on dense cells.
2. **BF16 `1e-4` Multiplicative Split Rounding Collapse**:
   Vanilla `e-means` splits donors via $\mathbf{c}_s \odot (1 \pm 10^{-4}\mathbf{s})$. Because `bfloat16` has only 7 mantissa bits (half-ULP $= 2^{-8} \approx 3.91 \times 10^{-3} \gg 10^{-4}$), both children round to the **exact same `bfloat16` vector** across all $d$ coordinates, forcing the fallback guard to nudge only coordinate `0` by 1 ULP (and collapsing to $\mathbf{0}$ near the origin). `H-Algo-1` and `SuperEMeans` apply directional multi-coordinate ULP separation along $\mathbf{v}_{s,1}$.
3. **Uncentered Coordinate-Shift Cancellation (`+25` in FP8, `+5,000`–`+20,000` in BF16/FP32)**:
   - On shifted embeddings (`offset = +25.0`), uncentered `float8_e4m3fn` (`max = 448`, 3 mantissa bits) wastes exponent range on the global mean and fails `<0.15%` parity (`-0.2207%` MSE drift at $d=96$). `FP8Quantizer` (`auto_center=True`, lossless power-of-two scale $s = 2^e$) restores held-out FP64 parity to `< 0.079%`.
   - Under `+5,000` to `+20,000` global shifts, uncentered `xsn + c0n - 2 * (xs @ c.T)` cancels catastrophically in FP32/BF16, and `BF16` ULP stair-steps (`0.78%` of $\|\mathbf{c}\|$) destroy 3-point parabolic curvature in `quad_merge` (`#55`). Register-level Sterbenz/FP64 origin subtraction (`x - origin`, `c - origin`) makes `EMeans`, `SMeans`, `SuperEMeans`, and `quad_merge` shift-equivariant to `< 10^{-4}` across `fp32/bf16/fp16/fp64`.
4. **Extreme Dynamic Range (`|x| = 10^{25}`) & Near-`FP16_MAX` Split Overflow (`#93`, `#95`)**:
   - Raising unnormalized squared distances `dmin ** 1.5` overflows IEEE FP32 (`3.4e38`) when feature magnitudes exceed $10^{13}$, crashing `torch.multinomial` with `inf`/`NaN`. Normalizing `xs_c / max(|xs_c|, 1)` and `(dmin / max(dmin)) ** power` makes seeding scale-invariant from `10^{-25}` to `10^{+25}`.
   - Nudging collapsed `float16` split children outward with `nextafter(+inf)` overflows values at `FP16_MAX = 65,504` to `+inf`; inward ULP nudging when $|c| \ge \text{FP16\_MAX}/2$ eliminates overflow completely.

---

## 4. Blackwell `SM120` Hardware, Triton JIT & GPU Synchronization Challenges

1. **Blackwell `SM120` `99 KB` Per-CTA Shared-Memory Limit (`H-Opt-1` / `H-Opt-2`)**:
   Unlike Hopper `SM90` (`227 KB` dynamic SMEM/CTA), NVIDIA RTX PRO 6000 Blackwell (`SM120`) enforces a strict **`101,376`-byte (`99 KB`)** dynamic shared-memory ceiling per CTA. A naive `128 x 256 x 64` BF16 tile requires `49,152 B/stage`, overflowing `99 KB` at `num_stages=3` (`147 KB`) and forcing Triton to clamp `num_stages=1`, which **regressed kernel throughput to `0.61x–0.76x`** due to stalled GDDR7 loads. Sizing wide-$D$ tiles to **`BLOCK_D = 32` for BF16 (`2 B`) and `BLOCK_D = 64` for FP8 (`1 B`)** yields an identical **`24,576 B/stage` (`73,728 B <= 99 KB` at `num_stages=3`)**, unlocking async 3-stage SM120 TMA pipelining at **`317.8 TFLOP/s` (BF16)** and **`636.4 TFLOP/s` (FP8)**.
2. **Triton 3.x `% 16` Pointer Alignment & Seed-Dependent Pool Specialization Spikes (`#20`, `#21`, `#91`, `#94`, `#111`)**:
   - Triton 3.8 specializes kernel binaries on whether integer arguments `(B, K, D, P, num_select, rng_seed)` are divisible by `16` (`% 16 == 0` vs `!= 0` vs `== 1`) unless `do_not_specialize` is declared.
   - In `seeding.py`, `torch.unique` produces seed-dependent candidate pool sizes `P <= 1024`. Without static `BLOCK_P in {256, 1024}` bucketing (`#20`) and pre-warming all `36` `(BLOCK_P, P%16, num_select%16, rng%16)` combinations (`#111`), fresh `TRITON_CACHE_DIR` workers suffered **`235–310 ms` mid-sweep PTXAS compilation spikes** on the first seed hitting `P % 16 == 0` (completely invisible to `k=256` `--smoke` runs because `block_greedy_kmeans_plusplus_torch` only activates at `k > 256`!).
   - In Triton 3.x, scalar `D = 1` is auto-promoted to a Python `constexpr int`, causing `D.to(tl.float32)` inside `_smeans_syncfree_revive_split_kernel` to fail with `CompilationError: 'int' object has no attribute 'to'` (`#91`), fixed by passing precomputed `float(1.0 / max(1, d))` and `do_not_specialize`.
3. **Eliminating Per-Step `.item()` / `.tolist()` Host-Device Sync Stalls (`#54`, `#55`, `smeans.py`)**:
   - Upstream `suayptalha/s-means` calls `.tolist()` and `.item()` every 5 steps (`unused.sum()`, `donors.sum()`, `sse_ema.max().item()`), draining the CUDA stream; replacing them with device-resident `torch.argsort` + 0-D tensor bounds cut end-to-end latency by **`-58.96%` mean / `-62.22%` median (`2.44x–2.65x` speedup)**.
   - Removing per-step `.item()` calls in `EMeans` (`ep_sq_dist`), `ANVIL II` (`3 -> 0` syncs/step), `Dust` (`fallback_var`), and `quad_merge` (`37 -> 3` batched `.tolist()` syncs) eliminated `46%–98%` artificial launch overhead prior to Phase 2A/2B sweeps.

---

## 5. Pre-Experiment Bug Audit Log & Distributed Fleet Hygiene

Every source module and benchmark script was gated by independent pre-experiment audits (`@verifier`, `@bug-hunter`, `@verifier-2`, `@patrol-3`) before launching multi-GPU sweeps:

| Audit Ref | Module / Script | Defect Caught Before Sweep | Resolution & Regression Test |
| :--- | :--- | :--- | :--- |
| **`#8–#12`** | `docs/fleet_status.md`, `scripts/check_fleet.sh` | Reattached `tmux` left shells with dead `/tmp/ssh-*/agent.*` `SSH_AUTH_SOCK` (`Permission denied` on `acorn`/`cashew`); `acorn`/`cashew` defaulted to Python `3.14.4`; `walnut` marked off-limits | Auto-discover live socket via `tmux show-environment -t 0 SSH_AUTH_SOCK`; standardize all 6 hosts on `uv venv --python 3.12`; hard-block `walnut` in code & scripts |
| **`#20`** | `src/emeans/datasets.py`, `src/emeans/seeding.py` | Non-atomic `np.savez` cache race (`BadZipFile` under concurrent workers) + 7-power `BLOCK_P` Triton JIT specialization | Atomic PID-temp `.npz` + `os.replace`; static `BLOCK_P in {256, 1024}` bucketing + full-shape `N=32768, k=1024` warmup |
| **`#29`** | `src/emeans/algo_extensions.py` | `float64` CUDA inputs crashed `torch.addmm(..., out_dtype=torch.float32)`; passing `init_centroids` to `EMeansAlgoExt(base_mode="emeans_pp")` skipped coreset priors (`2^-16` step-0 erasure); stale `prin_vec`/`ema_sse` on `resample` | Matched `bf16` default compute dtype + `fp64` fallback; auto-extract coreset priors when `seeding_result is None`; full state reset on `resample` |
| **`#32, #35`** | `src/emeans/fast_ops.py` | `DTYPE_MODE == 0` (`FP32`) left `bf16`/`fp16`/`fp64` `x_tile` uncast before `tl.dot`; unnormalized `prior_distortion` when `sum(w_seed) != 1`; missing `finalization` / `IVFAssigner` FP8 constructor guards | Explicit `.to(tl.float32)` cast in `DTYPE_MODE == 0`; unit-sum normalized `w_norm` & `dist_mass / total_m`; strict `ValueError`/`TypeError` validation (`test_fast_ops.py`) |
| **`#54, #55`** | `src/emeans/noise_tracks.py` | **Twin-centroid rare-mode assassination**: multi-pair `run_head_edit_step` did not refresh `d_nn2` / `rare_shield` after transplanting Twin #1, immediately transplanting Twin #2 away from the rare mode; `dust_antithetic` leaked into `T=2` closing pass; $O(R)$ Hadamard overshoot; BF16 `quad_merge` shift/ULP collapse | Full incremental `d_nn2` + `huygens_cost` + `rare_shield` refresh between accepted pairs; `not is_last_epoch` guard; 1D half-space conditional projections; FP32 centered `quad_merge` (`test_noise_tracks.py`) |
| **`#91, #94`** | `src/emeans/smeans.py`, `run_phase3_smeans_ablation.py` | `D=1` Triton `constexpr int` `.to(tl.float32)` crash; `_fit_exact_ref` call-site signature mismatch after making `split_mode` a runtime `int`; Suite 1 `(T=1,2,5)` cells split across `300W`/`450W`/`600W` SKUs on same Pareto curve | Passed scalar `inv_d` + `do_not_specialize` across all 6 kernels; pinned all `(T=1,2,5)` cells of every Suite 1 regime to a single Blackwell TDP SKU (`assert len(skus_for_regime) == 1`) |
| **`#93`** | `src/emeans/super_emeans.py` | When `need_priors == False` (`smeans_plus_p2_huygens_shield` / `p5_head_edit`), dummy uniform `p_hat = 1/k` initialized `survival_mass = B/k`, falsely shielding 0-count dead clusters from revival; `d15` `1e25` FP32 overflow | Initialize `survival_mass = 0` when `not need_priors` so only clusters with $\ge 1$ real batch hit earn protection; max-normalized scale-invariant `subsampled_d15_init_centered` (`test_super_emeans.py`) |
| **`#111`** | `scripts/run_phase3_smeans_ablation.py` | Fresh `TRITON_CACHE_DIR` workers hit `235–310 ms` cold-JIT spikes on `k > 256` when `torch.unique` pool `P` first hit `P % 16 == 0` | Pre-warmed all 36 `(BLOCK_P, P%16, num_select%16, rng%16)` buckets + coreset `K % 16` buckets outside timers; bumped `--smoke` to `k=512`; verified `0/885` outliers across 11,250 runs |
