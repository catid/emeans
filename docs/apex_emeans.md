# `ApexEMeans`: Unified 4-Gear Scale-Adaptive GPU Clustering Engine

## 1. Overview & Design Goals

[`ApexEMeans`](../src/emeans/apex_emeans.py) is a unified streaming vector quantization and $k$-means engine designed to dominate existing streaming and batch clustering baselines simultaneously on **wall-clock latency** and **held-out quantization distortion / ANN IVF recall** across four orders of magnitude in dataset scale ($N = 10^4 \to 10^7+$), codebook cardinality ($K = 256 \to 65{,}536$), and embedding dimensionality ($D = 1 \to 1{,}024+$).

All system, quantization, scale-law, and verification components are organized across:
- Flagship 4-gear implementation: [`src/emeans/apex_emeans.py`](../src/emeans/apex_emeans.py)
- `CroissAttn`-stabilized FP8 + Bilinear Double-FP8 Ozaki kernels: [`src/emeans/croiss_fp8.py`](../src/emeans/croiss_fp8.py) & [`docs/croissattn_fp8_study.md`](./croissattn_fp8_study.md)
- Empirical 4-decade gear-transition power laws (`1,327` runs across `79` cells): [`docs/scale_adaptive_laws.md`](./scale_adaptive_laws.md)
- Automated verification suite (`35` unit, parity, CUDAGraph, Ozaki-FP8, 88-wave race, 2-GPU NCCL & cold-cache tests): [`tests/test_apex_emeans.py`](../tests/test_apex_emeans.py)

---

## 2. Four-Gear Scale-Adaptive Architecture

`ApexEMeans(..., gear_mode="auto", precision="auto")` automatically routes each `(N, D, K, nepochs)` workload to the Pareto-optimal execution gear calibrated across `1,327` validation-only empirical scale-law runs ([`docs/scale_adaptive_laws.md`](./scale_adaptive_laws.md)):

1. **Gear 1 — Small-Scale & Wide-$D$ Regime (`small_{bf16|fp32}_graph`)**:
   - Selected when $N \cdot K \cdot T \cdot (D/128)^{0.820} < 1.852 \times 10^8$ (or $N < 65{,}536 \land K < 2{,}048$), or when $D > 256$ (`FULL_D = False`, avoiding tiled indirect-gather overhead on 512D–1024D embeddings).
   - Executes BF16 Tensor-Core assignment (with 2D Split-$K$ SM saturation on nimble batches) fused into a 5-step on-device `torch.cuda.CUDAGraph` replay loop.
2. **Gear 2 — Medium-to-Million Dense Tensor-Core Regime (`medium_croiss_fp8_graph`, $D \le 256$)**:
   - Pre-permutes and encodes centered coordinates once in a single coalesced HBM pass (up to $5.6\,\text{GB}$ / $28\%$ free VRAM, covering $10\text{M} \times 128$ in FP32 + FP8) into decoupled per-row `sx_i` + per-centroid `sc_j` FP8 `float8_e4m3fn` (`tl.dot(..., max_num_imprecise_acc=0, out_dtype=tl.float32)` via SM120 3-Stage TMA / Split-$K$ in [`src/emeans/croiss_fp8.py`](../src/emeans/croiss_fp8.py)).
   - Automatically upgrades to **3-lane Bilinear Double-FP8 Ozaki Residual Expansion (`scale_mode="ozaki"`)** whenever the dimensionless coordinate-to-Voronoi dynamic range ratio satisfies:
     $$\text{CR} = \frac{\max_i \|x_i - \mu\|_2^2}{\text{MSE}_{\text{init}}} > 120$$
     (e.g., 54D multi-scale physical sensor features with $110{,}761\times$ column standard deviation span, $\text{CR} \approx 913$), achieving $20.6\,\text{ppm}$ assignment fidelity inside FP8 Tensor Cores + `CUDAGraph`.
3. **Gear 3 — Huge-$K$ Coarse-to-Fine IVF + Exact Closing Gear (`huge_k_ivf_croiss_fp8_graph`, $K \ge 16{,}384, D \ge 64, N \ge 30 K, T \ge 2$)**:
   - Initializes via **Coarse-Bucketed Stratified `k-means++` + Sync-Free IVF Voronoi Polish** ($M_{\text{coarse}} = \operatorname{clamp}(\operatorname{pow2}(\sqrt{K}), 128, 256)$, cutting $K = 65{,}536$ seeding from $\sim 102\,\text{ms} \to 10.3\,\text{ms}$).
   - Routes exploration epochs ($0 \dots T - 2$) through `_ApexSyncFreeIVFExplorer` (`nlist = 1 * sqrt(K) in [128, 256]`, `nprobe = 8..16`, masking empty buckets to $+\infty$ prior to `topk` with zero host syncs and rebuilding immediately on revivals), then executes the final closing pass in **100% Exact `CroissAttn`-FP8 + `CUDAGraph`** so final Voronoi partitions and Empirical-Bayes statistics are exact.
4. **Gear 4 — Multi-GPU Data-Sharded Collective Engine (`multi_gpu_{W}x_{prec}_nccl`)**:
   - Distributes shards across $W$ GPUs via `prepopulate_shards(x)` (isolated outside `init_time_sec + train_time_sec`) or accepts pre-resident `list[torch.Tensor]` shards directly (supporting both equal and unequal shard lengths via proportional `slice_bounds_w`).
   - Packs `[sums, g_sums, counts, batch_sse]` into a single contiguous 1D communication buffer per GPU (`stride == d`), reducing all statistics in **one coalesced `torch.cuda.nccl.all_reduce` per step** with zero host-device synchronizations inside the epoch loop.

---

## 3. Systems Optimizations (6-Step Speedup Ladder)

`APEX_LADDER_ARMS` in [`src/emeans/apex_emeans.py`](../src/emeans/apex_emeans.py) isolates each systems optimization layer cumulatively:

| Ladder Key | Layer Added | Key Mechanism |
| :--- | :--- | :--- |
| `ladder_0_baseline` | Baseline | Random-gather `x[order]` + unfused assign / scatter / EMA |
| `ladder_1_plus_batch` | **+Batch** | 1-pass coalesced pre-permutation + cyclic block streaming + Two-Gear $2\times$ wide batch after $25\%$ progress ($B < 32{,}768$, $\alpha_0 \propto B^{0.9003}$) |
| `ladder_2_plus_fuse` | **+Fuse** | Fused `_apex_scatter_oja_sse_kernel` + `_apex_ema_oja_slow_pass_prep_kernel` + SRAM Twin-Safe Huygens revival |
| `ladder_3_plus_kernel` | **+Kernel** | Blackwell SM120 register/SRAM-tiled `FULL_D` + 2D Split-$K$ Tensor-Core kernels |
| `ladder_4_plus_cudagraph` | **+CUDAGraph** | 3-slot ping-pong on-device state ring (`gpu_state[3, 8]`) + fixed-address `order_assign_buf` cross-epoch graph cache capturing 5-step super-steps (`g.reset()` accounted inside `train_time_sec`) |
| `ladder_5_plus_croiss_fp8` | **+CroissFP8** | Origin-centered decoupled E4M3FN (`1.00000024 / 448.0`, `FP32_TINY` floor, exact represented norm $\|q_c\|^2 s_c^2$, `0` `0x7F` NaNs) |

### Race-Free 3-Slot Ping-Pong State Ring (Multi-Wave CTA Safety)
In `_apex_ema_oja_slow_pass_prep_kernel[(K,)]`, when $K = 16{,}384$ exceeds the 188 SMs on Blackwell (`88` CTA waves), mutating `step_idx` in-place from CTA `0` would race against later CTA waves reading `step_idx`. `ApexEMeans` eliminates this structurally via `_make_superstep_ring_slots(G)` over `gpu_state[0..2]`:
- For even $G$: `(0->1), (1->0), ...`
- For odd $G \ge 3$ (e.g., $G = 5$): `(0->1), (1->0), (0->1), (1->2), (2->0)`

Every substep satisfies `in_s != out_s` (`assert in_s != out_s`), and every super-step begins and ends at slot `0`, making `g.replay()` 100% composable across epochs and race-free across arbitrary $K$ (`max_err == 0.0` verified in `test_multiwave_k16384_no_inter_cta_step_race`).

---

## 4. Mathematical & Algorithmic Foundations

1. **Voronoi-Anchored Multi-CTA Seeding + Orthant-Deduplicated Outlier Leaders + 1-Pass Huygens Polish (`apex_turkpp_init_centered`)**:
   - Deduplicates extreme tail candidates across 512 directional orthants centered on the outlier cloud (`scatter_reduce_("amax")` on packed `(d2_bits << 32) | idx`) so distinct rare modes never collapse onto a single mode's points.
   - Shards candidate pools ($P > 256$) by nearest $\mathbb{R}^D$ coarse Voronoi anchors across SMs (`_apex_batched_local_greedy_kpp_kernel`, $\sim 0.73\,\text{ms}$ at $K=4{,}096$) and switches to Coarse-Bucketed Stratified `k-means++` + IVF Voronoi polish at $K \ge 16{,}384$.
   - Computes exact post-polish cluster distortions in **one** pass without a second GEMM or `[m, D]` residual copy via the Huygens parallel-axis identity:
     $$\sum_{i \in C_j} w_i \|x_i - \mu_j\|_2^2 = \sum_{i \in C_j} w_i \|x_i - c_j^{(0)}\|_2^2 - M_j \|\mu_j - c_j^{(0)}\|_2^2$$
2. **Spectral-Concentration Half-Gaussian Oja Split + Twin-Safe SRAM Huygens Rare-Mode Shield**:
   - Tracks leading cluster principal vector $v_{j,1}$ (`||v_{j,1}|| = \sigma_{j,1}`) online inside `_apex_scatter_oja_sse_kernel` (automatically disabled after the revival window to cut closing-pass atomic traffic).
   - Modulates split step size smoothly by spectral concentration $r_j = \sigma_{j,1} / \sigma_{\text{rms},j}$: $\eta_{\text{eff}}(r_j) \in [0.18, \sqrt{2/\pi}]$, gently separating curved high-$D$ manifolds while bisecting merged bimodal clusters at the exact $1\text{D}$ Gaussian conditional mean $\sqrt{2/\pi}\sigma_1$.
   - Protects genuine isolated rare modes ($K < 16{,}384$) whose removal cost $M_{\text{group}} \cdot d_{\text{ext}}^2 \ge \gamma_{\text{protect}} \bar{e}$ while splitting zero-distance duplicate twins, and gates $P/Q$-confirmed `head_edit` transplants strictly to the positive-ROI regime ($N/K \ge 90, K \le 2{,}048, T \ge 1.5$).
3. **Pre-Closing Independent-Anchor Positive-Part James-Stein Simplex Empirical-Bayes (`apex_simplex_eb_finalize`)**:
   - Snapshots `pre_close_centroids` immediately before the closing pass (updated coherently on splits/transplants), providing a zero-lag anchor statistically independent of closing-pass sampling noise for $T \ge 1.5$.
   - Applies Positive-Part James-Stein shrinkage:
     $$w_{\text{indep},j} = \phi_{\text{indep}} \min\!\left(\frac{\kappa_{\max}(n_{\text{eff},j})}{n_{\text{eff},j} + \kappa_{\max}(n_{\text{eff},j})},\; \frac{c_{\text{Stein}}(D)\,\hat{\sigma}_j^2 / n_{\text{eff},j}}{\|\bar{x}_{\text{close},j} - \mu_{\text{anc},j}\|_2^2}\right)$$
     where $\kappa_{\max}(n_{\text{eff},j}) = \kappa_{\max}\operatorname{clamp}((160 - n_{\text{eff},j})/96, 0, 1)$ at $T \le 3.5$ protects dense high-$N/K$ clusters from drift lag while giving sparse tail clusters full Stein variance reduction.

---

## 5. Validation-Split Latency & Accuracy Profile (`325W` Blackwell Max-Q)

All pre-experiment profiling strictly used `.base` / `.train` and `.val` splits (`0` held-out `.query` or `.test` access):

| Workload (`Val Split Only`) | Shape `(N, D, K, T)` | Selected Gear | Total Latency | Key Baseline Comparison |
| :--- | :--- | :--- | ---: | :--- |
| `covtype_multiscale_54` | `(500K, 54, 2048, 3)` | `medium_croiss_fp8_graph` (`ozaki`) | **`19.7 ms`** | `27,354.4` val MSE vs `UltraEMeans` `26.2 ms / 27,357.4` & `smeans_fast` `27,364.0` |
| `agnews_tfidf_svd_256` | `(110K, 256, 1024, 3)` | `medium_croiss_fp8_graph` | **`27.9 ms`** | `172.18` val MSE vs `UltraEMeans` `39.4 ms` (`1.42x`), `SuperEMeans` `337.2 ms / 172.36` |
| `cifar10_resnet18_512` | `(50K, 512, 1024, 3)` | `small_bf16_graph` | **`35.9 ms`** | `800.46` val MSE vs `SuperEMeans` `92.6 ms` (`2.58x`), `UltraEMeans` `56.0 ms / 801.47` |
| `cifar10_convnext_768` | `(50K, 768, 1024, 3)` | `small_bf16_graph` | **`46.4 ms`** | `151.60` val MSE vs `SuperEMeans` `96.0 ms / 151.62` (`2.07x`), `smeans_fast` `156.44` |
| `sift1m` | `(1M, 128, 4096, 2)` | `medium_croiss_fp8_graph` | **`35.2–38.1 ms`** | `1.57x–1.75x` faster than `UltraEMeans` (`55.2–66.6 ms`) with matched val MSE |
| `glove1m_unit` | `(1M, 100, 4096, 2)` | `medium_croiss_fp8_graph` | **`39.3 ms`** | `0.60612` val MSE vs `UltraEMeans` `57.2 ms / 0.60630`, `smeans_fast` `0.60936` |
| `c2b` Rare Outliers | `(200K, 128, 2048, 5)` | `medium_croiss_fp8_graph` | **`26.5 ms`** | `init = 6.7 ms`, `56.9%` rare modes preserved |
| `c4a` Huge-$K$ (1-GPU) | `(2M, 128, 65536, 5)` | `huge_k_ivf_croiss_fp8_graph` | **`550.1 ms`** | `init = 10.3 ms` (`9.9x` faster seed), `3.8x` faster than `UltraEMeans` (`2,083 ms`) |
| `c4c` Huge-$K$ (2-GPU) | `(2M, 128, 65536, 5)` | `multi_gpu_2x_croiss_fp8_nccl` | **`494.7 ms`** | `init = 8.2 ms`, hardware `torch.cuda.nccl` `all_reduce` across `cuda:0 + cuda:1` |

---

## 6. Reproduction & Verification Commands

```bash
# Run full 35-test unit, CUDAGraph, CroissFP8/Ozaki, 88-wave K=16384 race, non-primary cuda:1, 2-GPU NCCL & cold-cache suite:
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/pytest -v tests/test_apex_emeans.py
```
