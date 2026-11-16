# Phase 3 Multi-Machine Showdown & Component Ablation: `s-means` vs. `SuperEMeans` & Hybrids

- **Total Validated Paired Runs**: `11,250` (`4,050` Suite 1 + `7,200` Suite 2 across `13` Blackwell GPUs on all `6` approved machines; `0` `NaN`/`Inf`, `0` duplicates, single-GPU/single-SKU paired within every cell and across `T in {1, 2, 5}` per Suite 1 regime).
- **Evaluation Protocol**: Exact `FP64` full-dataset `train_mse` and held-out `test_mse` (`exact_mse_fp64`), plus exact `FP64` spherical cosine distortion on `unit_sphere` cells, and synchronized warm GPU `init_time_sec` / `train_time_sec` / `total_time_sec`.
- **Sign Convention**: Negative `Mean Delta (%)` means **lower MSE (better accuracy)** or **faster wall-clock latency** relative to the baseline.

## 0. Definitive Verdict (`s-means` vs. `SuperEMeans` vs. `HybridSEMeans`)

1. **`SuperEMeans` and `HybridSEMeans` decisively outperform `suayptalha/s-means` on quantization accuracy across diverse synthetic and real deep-embedding regimes**:
   - **`HybridSEMeans` (`hybrid_super_smeans`) vs. `SMeans` (`smeans_fast`)**: **`-2.5683%` mean / `-1.3608%` median Test MSE reduction** across all 30 Suite 1 cells (`26/0/4` cell W/T/L, `372/0/78` seed W/T/L, paired $t$-test $p = 1.631\times 10^{-4}$, Wilcoxon $p = 1.061\times 10^{-5}$), winning **`10/10` cells at `T=1`** (`-3.7061%`), **`8/10` at `T=2`** (`-2.4216%`), and **`8/10` at `T=5`** (`-1.5773%`), plus **`-2.8569%` mean / `-1.3954%` median** on matched-kernel Suite 2 (`10/2` cells, `157/23` seeds, $p = 0.0049$).
   - **`SuperEMeans` (`super_emeans`) vs. `SMeans` (`smeans_fast`)**: **`-2.3636%` mean / `-1.3521%` median Test MSE reduction** across all 30 Suite 1 cells (`25/0/5` cells, `371/0/79` seeds, $p_{\text{ttest}} = 1.731\times 10^{-4}$, $p_{\text{wilcox}} = 9.220\times 10^{-6}$), sweeping **`15/15` seeds on every real visual embedding cell** (`cifar10_r18_512`: `-4.71% / -3.17% / -1.78%`; `cifar10_r18_512_unit` cosine distortion: `-4.77% / -3.36% / -1.78%`; `cifar10_patches_128`: `-1.28% / -0.39% / -0.16%`) and preserving **`60.52%`–`63.53%`** of isolated rare outlier modes at `T=5` where `s-means` preserves **`4.18%` (`0.00%` at `T=1, 2`)** due to lack of Huygens rare-mode protection.
2. **When a combination (`HybridSEMeans`) wins vs. when pure `SuperEMeans` wins (Exact Geometric Complementarity)**:
   - Across all 30 Suite 1 cells, **`hybrid_super_smeans` achieves the lowest overall mean Test MSE (`-0.2135%` vs. `SuperEMeans` in Suite 1; `-0.4139%` in Suite 2) AND `-2.32%` faster runtime (`28/30` cells, $p = 3.25\times 10^{-6}$)** (`hybrid_super_smeans_wide` adds `-24.55%` speedup over `SuperEMeans` across `30/30` cells).
   - Cell wins split `15/15` strictly along **cluster separation vs. continuous manifold geometry**:
     * **Multi-Modal / Heavy-Tailed Clustered Regimes (`balanced_gmm`, `zipf_gmm`, `low_nk32_gmm`, `multiscale_rare`, `lowd_zipf_d8`)**: **`HybridSEMeans` wins (`-0.56%` to `-2.54%` over `SuperEMeans`, $p < 10^{-3}..10^{-11}$)** because fusing `SuperEMeans`'s local **Oja principal eigenvector direction (`P3`)** with `s-means`'s **`0.55\sqrt{\text{MSE}_s}` chord step (`C5b: oja_rms`) + `SSE * N^{0.35}` donor scoring (`C4`) + `85%` revival freeze (`C3`)** bifurcates merged clusters in 1 step while retaining `P1` Coreset+Voronoi, `P2` Huygens Shield, `P4` EB-JS, and `P5` $P/Q$ `head_edit`.
     * **Continuous Manifolds & Real Deep Embeddings (`cifar10_r18_512`, `cifar10_r18_512_unit`, `cifar10_patches_128`, `isotropic_ctrl`)**: **Pure `SuperEMeans` (`0.08\sigma_1` Oja split + pure $D^2$ donors) wins `12/12` cells (`-0.14%` to `-1.10%` over `HybridSEMeans`)**, whereas `s-means` regresses even against *vanilla* `e-means` (`+1.46%` to `+4.32%` on `cifar10_r18_512`) because `0.55\sigma_{\text{rms}}` Rademacher/chord jumps overshoot curved deep feature manifolds; grafting **only `C3` (`revive_stop=0.85`) + `C4` (`coupled sse_usage`)** onto `SuperEMeans` saves **`-3.07%` runtime (`12/12` cells)** with zero manifold overshoot (`-0.004%` / `-0.18%` Test MSE).
3. **Speed–Accuracy Pareto Frontier & Zero-Sync `smeans_fast` Kernel Win**:
   - Our zero-sync centered-Triton `smeans_fast` matches upstream `smeans_exact_ref` accuracy (`-0.031%`, $p = 0.335$) while running **`2.44x–2.65x` faster (`-58.96%` mean / `-62.22%` median latency, `30/30` cells, $p = 1.29\times 10^{-27}$)** and **`2.12x` faster than `vanilla_emeans` (`-48.55%` mean / `-52.91%` median, `7.2–8.8 ms` median)** — making **`smeans_fast` the sub-`10ms` ultra-low-latency Pareto point** (when `~14–38ms` coreset initialization cannot be afforded) and **`hybrid_super_smeans_wide` / `super_emeans_wide` (`43–83 ms`) + `hybrid_super_smeans` / `super_emeans` (`57–106 ms`) the global high-accuracy Pareto frontier**.

## 1. Suite 1 Executive Showdown Aggregates (`Test MSE` & `Total Time` Across Regimes)

| Scope | Arm | Baseline | Metric | Cells | Mean Delta (%) | Median Delta (%) | Cell W/T/L | Seed W/T/L | `p_ttest` | `p_wilcox` |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | :---: | :---: | ---: | ---: |
| `all_T` | `smeans_fast` | `vanilla_emeans` | `test_mse` | 30 | -2.8201% | -0.0915% | 18/0/12 | 241/0/209 | 2.061e-02 | 1.241e-01 |
| `T=1` | `smeans_fast` | `vanilla_emeans` | `test_mse` | 10 | -2.9400% | -0.2269% | 6/0/4 | 84/0/66 | 2.615e-01 | 4.922e-01 |
| `T=2` | `smeans_fast` | `vanilla_emeans` | `test_mse` | 10 | -2.8297% | -0.1159% | 6/0/4 | 80/0/70 | 2.119e-01 | 4.922e-01 |
| `T=5` | `smeans_fast` | `vanilla_emeans` | `test_mse` | 10 | -2.6906% | -0.0745% | 6/0/4 | 77/0/73 | 1.137e-01 | 3.750e-01 |
| `all_T` | `smeans_fast` | `vanilla_emeans` | `total_time_sec` | 30 | **-48.5511%** | -52.9084% | 30/0/0 | 450/0/0 | 3.369e-20 | 1.863e-09 |
| `T=1` | `smeans_fast` | `vanilla_emeans` | `total_time_sec` | 10 | **-48.4909%** | -53.9121% | 10/0/0 | 150/0/0 | 6.979e-07 | 1.953e-03 |
| `T=2` | `smeans_fast` | `vanilla_emeans` | `total_time_sec` | 10 | **-49.4166%** | -53.4168% | 10/0/0 | 150/0/0 | 2.997e-07 | 1.953e-03 |
| `T=5` | `smeans_fast` | `vanilla_emeans` | `total_time_sec` | 10 | **-47.7457%** | -51.3324% | 10/0/0 | 150/0/0 | 3.480e-07 | 1.953e-03 |
| `all_T` | `smeans_exact_ref` | `vanilla_emeans` | `test_mse` | 30 | -2.7909% | -0.0813% | 17/0/13 | 243/0/207 | 2.083e-02 | 1.294e-01 |
| `T=1` | `smeans_exact_ref` | `vanilla_emeans` | `test_mse` | 10 | -2.8225% | -0.2405% | 6/0/4 | 82/0/68 | 2.716e-01 | 4.922e-01 |
| `T=2` | `smeans_exact_ref` | `vanilla_emeans` | `test_mse` | 10 | -2.8353% | -0.0920% | 5/0/5 | 81/0/69 | 2.112e-01 | 5.566e-01 |
| `T=5` | `smeans_exact_ref` | `vanilla_emeans` | `test_mse` | 10 | -2.7149% | -0.0490% | 6/0/4 | 80/0/70 | 1.110e-01 | 3.750e-01 |
| `all_T` | `smeans_exact_ref` | `vanilla_emeans` | `total_time_sec` | 30 | +24.6747% | +23.3662% | 0/0/30 | 0/0/450 | 8.002e-21 | 1.863e-09 |
| `T=1` | `smeans_exact_ref` | `vanilla_emeans` | `total_time_sec` | 10 | +25.6961% | +23.4824% | 0/0/10 | 0/0/150 | 8.550e-08 | 1.953e-03 |
| `T=2` | `smeans_exact_ref` | `vanilla_emeans` | `total_time_sec` | 10 | +23.1377% | +21.1205% | 0/0/10 | 0/0/150 | 2.639e-07 | 1.953e-03 |
| `T=5` | `smeans_exact_ref` | `vanilla_emeans` | `total_time_sec` | 10 | +25.1902% | +24.0488% | 0/0/10 | 0/0/150 | 4.672e-07 | 1.953e-03 |
| `all_T` | `smeans_fast` | `smeans_exact_ref` | `test_mse` | 30 | -0.0309% | -0.0164% | 18/0/12 | 239/0/211 | 3.351e-01 | 1.909e-01 |
| `T=1` | `smeans_fast` | `smeans_exact_ref` | `test_mse` | 10 | **-0.1256%** | -0.0425% | 8/0/2 | 88/0/62 | 7.985e-02 | 1.953e-02 |
| `T=2` | `smeans_fast` | `smeans_exact_ref` | `test_mse` | 10 | +0.0067% | -0.0017% | 5/0/5 | 77/0/73 | 9.009e-01 | 1.000e+00 |
| `T=5` | `smeans_fast` | `smeans_exact_ref` | `test_mse` | 10 | +0.0262% | -0.0067% | 5/0/5 | 74/0/76 | 4.836e-01 | 9.219e-01 |
| `all_T` | `smeans_fast` | `smeans_exact_ref` | `total_time_sec` | 30 | **-58.9597%** | -62.2165% | 30/0/0 | 450/0/0 | 1.293e-27 | 1.863e-09 |
| `T=1` | `smeans_fast` | `smeans_exact_ref` | `total_time_sec` | 10 | **-59.2626%** | -62.9674% | 10/0/0 | 150/0/0 | 3.389e-09 | 1.953e-03 |
| `T=2` | `smeans_fast` | `smeans_exact_ref` | `total_time_sec` | 10 | **-59.1506%** | -62.2989% | 10/0/0 | 150/0/0 | 1.819e-09 | 1.953e-03 |
| `T=5` | `smeans_fast` | `smeans_exact_ref` | `total_time_sec` | 10 | **-58.4658%** | -61.5497% | 10/0/0 | 150/0/0 | 1.439e-09 | 1.953e-03 |
| `all_T` | `emeans_pp` | `vanilla_emeans` | `test_mse` | 30 | **-4.3770%** | -0.3195% | 21/0/9 | 339/0/111 | 1.642e-03 | 1.529e-04 |
| `T=1` | `emeans_pp` | `vanilla_emeans` | `test_mse` | 10 | **-5.7679%** | -0.4509% | 9/0/1 | 136/0/14 | 6.111e-02 | 5.859e-03 |
| `T=2` | `emeans_pp` | `vanilla_emeans` | `test_mse` | 10 | -4.3844% | -0.0999% | 6/0/4 | 105/0/45 | 8.180e-02 | 1.055e-01 |
| `T=5` | `emeans_pp` | `vanilla_emeans` | `test_mse` | 10 | -2.9787% | -0.0347% | 6/0/4 | 98/0/52 | 1.007e-01 | 1.309e-01 |
| `all_T` | `emeans_pp` | `vanilla_emeans` | `total_time_sec` | 30 | +156.2412% | +162.8344% | 0/0/30 | 0/0/450 | 1.840e-11 | 1.863e-09 |
| `T=1` | `emeans_pp` | `vanilla_emeans` | `total_time_sec` | 10 | +179.6150% | +183.1874% | 0/0/10 | 0/0/150 | 3.389e-04 | 1.953e-03 |
| `T=2` | `emeans_pp` | `vanilla_emeans` | `total_time_sec` | 10 | +153.4387% | +173.6432% | 0/0/10 | 0/0/150 | 2.253e-05 | 1.953e-03 |
| `T=5` | `emeans_pp` | `vanilla_emeans` | `total_time_sec` | 10 | +135.6698% | +137.4700% | 0/0/10 | 0/0/150 | 3.625e-04 | 1.953e-03 |
| `all_T` | `emeans_pp` | `smeans_fast` | `test_mse` | 30 | **-1.6196%** | -0.6517% | 22/0/8 | 341/0/109 | 1.399e-03 | 6.084e-04 |
| `T=1` | `emeans_pp` | `smeans_fast` | `test_mse` | 10 | **-2.9516%** | -2.8747% | 9/0/1 | 135/0/15 | 1.179e-02 | 5.859e-03 |
| `T=2` | `emeans_pp` | `smeans_fast` | `test_mse` | 10 | **-1.6036%** | -0.5275% | 8/0/2 | 117/0/33 | 6.452e-02 | 1.953e-02 |
| `T=5` | `emeans_pp` | `smeans_fast` | `test_mse` | 10 | -0.3036% | -0.0333% | 5/0/5 | 89/0/61 | 4.954e-01 | 5.566e-01 |
| `all_T` | `emeans_pp` | `smeans_fast` | `total_time_sec` | 30 | +404.1724% | +385.9889% | 0/0/30 | 0/0/450 | 1.864e-15 | 1.863e-09 |
| `T=1` | `emeans_pp` | `smeans_fast` | `total_time_sec` | 10 | +447.6759% | +444.3726% | 0/0/10 | 0/0/150 | 1.660e-05 | 1.953e-03 |
| `T=2` | `emeans_pp` | `smeans_fast` | `total_time_sec` | 10 | +409.5865% | +426.7510% | 0/0/10 | 0/0/150 | 2.947e-06 | 1.953e-03 |
| `T=5` | `emeans_pp` | `smeans_fast` | `total_time_sec` | 10 | +355.2547% | +356.6364% | 0/0/10 | 0/0/150 | 1.249e-05 | 1.953e-03 |
| `all_T` | `super_emeans` | `vanilla_emeans` | `test_mse` | 30 | **-5.0206%** | -0.3595% | 27/0/3 | 393/0/57 | 1.637e-03 | 1.684e-06 |
| `T=1` | `super_emeans` | `vanilla_emeans` | `test_mse` | 10 | **-6.3201%** | -0.5760% | 9/0/1 | 135/0/15 | 6.915e-02 | 5.859e-03 |
| `T=2` | `super_emeans` | `vanilla_emeans` | `test_mse` | 10 | **-4.9291%** | -0.2773% | 9/0/1 | 130/0/20 | 8.089e-02 | 9.766e-03 |
| `T=5` | `super_emeans` | `vanilla_emeans` | `test_mse` | 10 | **-3.8126%** | -0.2621% | 9/0/1 | 128/0/22 | 9.507e-02 | 1.367e-02 |
| `all_T` | `super_emeans` | `vanilla_emeans` | `total_time_sec` | 30 | +375.6915% | +383.1760% | 0/0/30 | 0/0/450 | 9.263e-20 | 1.863e-09 |
| `T=1` | `super_emeans` | `vanilla_emeans` | `total_time_sec` | 10 | +367.0403% | +380.6950% | 0/0/10 | 0/0/150 | 2.565e-06 | 1.953e-03 |
| `T=2` | `super_emeans` | `vanilla_emeans` | `total_time_sec` | 10 | +368.1146% | +390.9886% | 0/0/10 | 0/0/150 | 2.426e-07 | 1.953e-03 |
| `T=5` | `super_emeans` | `vanilla_emeans` | `total_time_sec` | 10 | +391.9197% | +382.5665% | 0/0/10 | 0/0/150 | 1.756e-07 | 1.953e-03 |
| `all_T` | `super_emeans` | `emeans_pp` | `test_mse` | 30 | **-0.7565%** | -0.1128% | 24/0/6 | 339/0/111 | 1.070e-02 | 1.886e-04 |
| `T=1` | `super_emeans` | `emeans_pp` | `test_mse` | 10 | -0.7038% | -0.0498% | 6/0/4 | 99/0/51 | 2.543e-01 | 1.934e-01 |
| `T=2` | `super_emeans` | `emeans_pp` | `test_mse` | 10 | **-0.6365%** | -0.1399% | 9/0/1 | 118/0/32 | 1.212e-01 | 1.953e-02 |
| `T=5` | `super_emeans` | `emeans_pp` | `test_mse` | 10 | **-0.9293%** | -0.2062% | 9/0/1 | 122/0/28 | 1.049e-01 | 2.734e-02 |
| `all_T` | `super_emeans` | `emeans_pp` | `total_time_sec` | 30 | +94.6796% | +87.2357% | 0/0/30 | 0/0/450 | 1.241e-13 | 1.863e-09 |
| `T=1` | `super_emeans` | `emeans_pp` | `total_time_sec` | 10 | +74.5157% | +70.7667% | 0/0/10 | 0/0/150 | 1.567e-05 | 1.953e-03 |
| `T=2` | `super_emeans` | `emeans_pp` | `total_time_sec` | 10 | +89.1584% | +84.8039% | 0/0/10 | 0/0/150 | 1.969e-06 | 1.953e-03 |
| `T=5` | `super_emeans` | `emeans_pp` | `total_time_sec` | 10 | +120.3646% | +108.3681% | 0/0/10 | 0/0/150 | 2.867e-05 | 1.953e-03 |
| `all_T` | `super_emeans` | `smeans_fast` | `test_mse` | 30 | **-2.3636%** | -1.3521% | 25/0/5 | 371/0/79 | 1.731e-04 | 9.220e-06 |
| `T=1` | `super_emeans` | `smeans_fast` | `test_mse` | 10 | **-3.6348%** | -3.4214% | 8/0/2 | 129/0/21 | 9.980e-03 | 9.766e-03 |
| `T=2` | `super_emeans` | `smeans_fast` | `test_mse` | 10 | **-2.2314%** | -1.0946% | 9/0/1 | 127/0/23 | 2.807e-02 | 5.859e-03 |
| `T=5` | `super_emeans` | `smeans_fast` | `test_mse` | 10 | -1.2245% | -0.3277% | 8/0/2 | 115/0/35 | 1.507e-01 | 8.398e-02 |
| `all_T` | `super_emeans` | `smeans_fast` | `total_time_sec` | 30 | +854.4781% | +955.4909% | 0/0/30 | 0/0/450 | 4.793e-19 | 1.863e-09 |
| `T=1` | `super_emeans` | `smeans_fast` | `total_time_sec` | 10 | +833.8731% | +955.4909% | 0/0/10 | 0/0/150 | 1.303e-06 | 1.953e-03 |
| `T=2` | `super_emeans` | `smeans_fast` | `total_time_sec` | 10 | +855.0737% | +900.6386% | 0/0/10 | 0/0/150 | 1.033e-06 | 1.953e-03 |
| `T=5` | `super_emeans` | `smeans_fast` | `total_time_sec` | 10 | +874.4875% | +914.9344% | 0/0/10 | 0/0/150 | 6.581e-07 | 1.953e-03 |
| `all_T` | `super_emeans` | `smeans_exact_ref` | `test_mse` | 30 | **-2.3929%** | -1.2249% | 25/0/5 | 369/0/81 | 2.421e-04 | 1.061e-05 |
| `T=1` | `super_emeans` | `smeans_exact_ref` | `test_mse` | 10 | **-3.7528%** | -3.5009% | 8/0/2 | 132/0/18 | 1.082e-02 | 9.766e-03 |
| `T=2` | `super_emeans` | `smeans_exact_ref` | `test_mse` | 10 | **-2.2260%** | -0.8528% | 9/0/1 | 126/0/24 | 3.308e-02 | 5.859e-03 |
| `T=5` | `super_emeans` | `smeans_exact_ref` | `test_mse` | 10 | -1.1999% | -0.3235% | 8/0/2 | 111/0/39 | 1.667e-01 | 8.398e-02 |
| `all_T` | `super_emeans` | `smeans_exact_ref` | `total_time_sec` | 30 | +282.7436% | +296.3887% | 0/0/30 | 0/0/450 | 4.263e-18 | 1.863e-09 |
| `T=1` | `super_emeans` | `smeans_exact_ref` | `total_time_sec` | 10 | +272.4708% | +291.7820% | 0/0/10 | 0/0/150 | 6.948e-06 | 1.953e-03 |
| `T=2` | `super_emeans` | `smeans_exact_ref` | `total_time_sec` | 10 | +280.8282% | +311.2611% | 0/0/10 | 0/0/150 | 6.645e-07 | 1.953e-03 |
| `T=5` | `super_emeans` | `smeans_exact_ref` | `total_time_sec` | 10 | +294.9317% | +284.5123% | 0/0/10 | 0/0/150 | 1.011e-06 | 1.953e-03 |
| `all_T` | `super_emeans_wide` | `smeans_fast` | `test_mse` | 30 | **-2.1738%** | -0.8876% | 24/0/6 | 363/0/87 | 1.880e-04 | 1.598e-05 |
| `T=1` | `super_emeans_wide` | `smeans_fast` | `test_mse` | 10 | **-3.2959%** | -3.2197% | 8/0/2 | 127/0/23 | 8.558e-03 | 9.766e-03 |
| `T=2` | `super_emeans_wide` | `smeans_fast` | `test_mse` | 10 | **-2.1087%** | -0.8863% | 9/0/1 | 126/0/24 | 3.026e-02 | 5.859e-03 |
| `T=5` | `super_emeans_wide` | `smeans_fast` | `test_mse` | 10 | -1.1170% | -0.3188% | 7/0/3 | 110/0/40 | 1.799e-01 | 1.309e-01 |
| `all_T` | `super_emeans_wide` | `smeans_fast` | `total_time_sec` | 30 | +630.7912% | +673.9785% | 0/0/30 | 0/0/450 | 4.397e-18 | 1.863e-09 |
| `T=1` | `super_emeans_wide` | `smeans_fast` | `total_time_sec` | 10 | +620.3564% | +694.0103% | 0/0/10 | 0/0/150 | 3.714e-06 | 1.953e-03 |
| `T=2` | `super_emeans_wide` | `smeans_fast` | `total_time_sec` | 10 | +647.8757% | +686.5216% | 0/0/10 | 0/0/150 | 1.681e-06 | 1.953e-03 |
| `T=5` | `super_emeans_wide` | `smeans_fast` | `total_time_sec` | 10 | +624.1415% | +644.8886% | 0/0/10 | 0/0/150 | 1.043e-06 | 1.953e-03 |
| `all_T` | `super_emeans_wide` | `super_emeans` | `test_mse` | 30 | +0.2043% | +0.0279% | 9/0/21 | 155/0/295 | 3.881e-02 | 2.367e-03 |
| `T=1` | `super_emeans_wide` | `super_emeans` | `test_mse` | 10 | +0.3699% | +0.0684% | 1/0/9 | 34/0/116 | 1.913e-01 | 5.859e-03 |
| `T=2` | `super_emeans_wide` | `super_emeans` | `test_mse` | 10 | +0.1316% | +0.0564% | 2/0/8 | 54/0/96 | 4.348e-02 | 2.734e-02 |
| `T=5` | `super_emeans_wide` | `super_emeans` | `test_mse` | 10 | +0.1113% | -0.0033% | 6/0/4 | 67/0/83 | 2.912e-01 | 1.000e+00 |
| `all_T` | `super_emeans_wide` | `super_emeans` | `total_time_sec` | 30 | **-23.4776%** | -23.0819% | 30/0/0 | 450/0/0 | 1.948e-24 | 1.863e-09 |
| `T=1` | `super_emeans_wide` | `super_emeans` | `total_time_sec` | 10 | **-23.0694%** | -21.8860% | 10/0/0 | 150/0/0 | 1.682e-07 | 1.953e-03 |
| `T=2` | `super_emeans_wide` | `super_emeans` | `total_time_sec` | 10 | **-21.7398%** | -21.3398% | 10/0/0 | 150/0/0 | 1.325e-09 | 1.953e-03 |
| `T=5` | `super_emeans_wide` | `super_emeans` | `total_time_sec` | 10 | **-25.6236%** | -24.9264% | 10/0/0 | 150/0/0 | 2.995e-10 | 1.953e-03 |
| `all_T` | `hybrid_super_smeans` | `smeans_fast` | `test_mse` | 30 | **-2.5683%** | -1.3608% | 26/0/4 | 372/0/78 | 1.631e-04 | 1.061e-05 |
| `T=1` | `hybrid_super_smeans` | `smeans_fast` | `test_mse` | 10 | **-3.7061%** | -3.2620% | 10/0/0 | 142/0/8 | 1.340e-02 | 1.953e-03 |
| `T=2` | `hybrid_super_smeans` | `smeans_fast` | `test_mse` | 10 | **-2.4216%** | -1.8953% | 8/0/2 | 117/0/33 | 3.982e-02 | 2.734e-02 |
| `T=5` | `hybrid_super_smeans` | `smeans_fast` | `test_mse` | 10 | **-1.5773%** | -1.2563% | 8/0/2 | 113/0/37 | 8.654e-02 | 3.711e-02 |
| `all_T` | `hybrid_super_smeans` | `smeans_fast` | `total_time_sec` | 30 | +831.7124% | +928.7736% | 0/0/30 | 0/0/450 | 4.527e-19 | 1.863e-09 |
| `T=1` | `hybrid_super_smeans` | `smeans_fast` | `total_time_sec` | 10 | +813.0866% | +933.4049% | 0/0/10 | 0/0/150 | 1.344e-06 | 1.953e-03 |
| `T=2` | `hybrid_super_smeans` | `smeans_fast` | `total_time_sec` | 10 | +824.6713% | +880.2358% | 0/0/10 | 0/0/150 | 1.004e-06 | 1.953e-03 |
| `T=5` | `hybrid_super_smeans` | `smeans_fast` | `total_time_sec` | 10 | +857.3795% | +885.0875% | 0/0/10 | 0/0/150 | 6.060e-07 | 1.953e-03 |
| `all_T` | `hybrid_super_smeans` | `super_emeans` | `test_mse` | 30 | -0.2135% | +0.0118% | 15/0/15 | 214/0/236 | 1.586e-01 | 3.387e-01 |
| `T=1` | `hybrid_super_smeans` | `super_emeans` | `test_mse` | 10 | -0.0822% | +0.0344% | 5/0/5 | 75/0/75 | 6.256e-01 | 6.250e-01 |
| `T=2` | `hybrid_super_smeans` | `super_emeans` | `test_mse` | 10 | -0.2020% | -0.0408% | 5/0/5 | 72/0/78 | 4.908e-01 | 6.953e-01 |
| `T=5` | `hybrid_super_smeans` | `super_emeans` | `test_mse` | 10 | -0.3562% | +0.0118% | 5/0/5 | 67/0/83 | 2.906e-01 | 5.566e-01 |
| `all_T` | `hybrid_super_smeans` | `super_emeans` | `total_time_sec` | 30 | **-2.3172%** | -2.2888% | 28/0/2 | 376/0/74 | 3.251e-06 | 3.790e-06 |
| `T=1` | `hybrid_super_smeans` | `super_emeans` | `total_time_sec` | 10 | **-2.1650%** | -1.8208% | 10/0/0 | 127/0/23 | 2.435e-03 | 1.953e-03 |
| `T=2` | `hybrid_super_smeans` | `super_emeans` | `total_time_sec` | 10 | **-3.1103%** | -2.5785% | 9/0/1 | 126/0/24 | 2.521e-03 | 3.906e-03 |
| `T=5` | `hybrid_super_smeans` | `super_emeans` | `total_time_sec` | 10 | -1.6764% | -2.3176% | 9/0/1 | 123/0/27 | 6.249e-02 | 8.398e-02 |
| `all_T` | `hybrid_super_smeans` | `emeans_pp` | `test_mse` | 30 | -0.9634% | -0.0207% | 16/0/14 | 241/0/209 | 1.336e-02 | 1.840e-01 |
| `T=1` | `hybrid_super_smeans` | `emeans_pp` | `test_mse` | 10 | -0.7815% | +0.0152% | 5/0/5 | 74/0/76 | 2.734e-01 | 6.953e-01 |
| `T=2` | `hybrid_super_smeans` | `emeans_pp` | `test_mse` | 10 | -0.8302% | +0.0003% | 5/0/5 | 71/0/79 | 2.042e-01 | 5.566e-01 |
| `T=5` | `hybrid_super_smeans` | `emeans_pp` | `test_mse` | 10 | -1.2785% | -0.0366% | 6/0/4 | 96/0/54 | 9.166e-02 | 2.754e-01 |
| `all_T` | `hybrid_super_smeans` | `emeans_pp` | `total_time_sec` | 30 | +90.0703% | +83.2966% | 0/0/30 | 0/0/450 | 2.587e-13 | 1.863e-09 |
| `T=1` | `hybrid_super_smeans` | `emeans_pp` | `total_time_sec` | 10 | +70.4581% | +65.3364% | 0/0/10 | 0/0/150 | 1.301e-05 | 1.953e-03 |
| `T=2` | `hybrid_super_smeans` | `emeans_pp` | `total_time_sec` | 10 | +83.1490% | +76.1700% | 0/0/10 | 0/0/150 | 2.687e-06 | 1.953e-03 |
| `T=5` | `hybrid_super_smeans` | `emeans_pp` | `total_time_sec` | 10 | +116.6039% | +102.1828% | 0/0/10 | 0/0/150 | 3.184e-05 | 1.953e-03 |
| `all_T` | `hybrid_super_smeans_wide` | `smeans_fast` | `test_mse` | 30 | **-2.3866%** | -1.3971% | 26/0/4 | 369/0/81 | 1.315e-04 | 7.994e-06 |
| `T=1` | `hybrid_super_smeans_wide` | `smeans_fast` | `test_mse` | 10 | **-3.4046%** | -3.2709% | 10/0/0 | 139/0/11 | 1.089e-02 | 1.953e-03 |
| `T=2` | `hybrid_super_smeans_wide` | `smeans_fast` | `test_mse` | 10 | **-2.3476%** | -1.7256% | 8/0/2 | 118/0/32 | 3.308e-02 | 1.953e-02 |
| `T=5` | `hybrid_super_smeans_wide` | `smeans_fast` | `test_mse` | 10 | **-1.4078%** | -0.9421% | 8/0/2 | 112/0/38 | 1.045e-01 | 3.711e-02 |
| `all_T` | `hybrid_super_smeans_wide` | `smeans_fast` | `total_time_sec` | 30 | +620.1155% | +665.6763% | 0/0/30 | 0/0/450 | 3.362e-18 | 1.863e-09 |
| `T=1` | `hybrid_super_smeans_wide` | `smeans_fast` | `total_time_sec` | 10 | +606.9157% | +677.9669% | 0/0/10 | 0/0/150 | 3.644e-06 | 1.953e-03 |
| `T=2` | `hybrid_super_smeans_wide` | `smeans_fast` | `total_time_sec` | 10 | +633.9770% | +665.3515% | 0/0/10 | 0/0/150 | 1.503e-06 | 1.953e-03 |
| `T=5` | `hybrid_super_smeans_wide` | `smeans_fast` | `total_time_sec` | 10 | +619.4540% | +634.5122% | 0/0/10 | 0/0/150 | 9.371e-07 | 1.953e-03 |
| `all_T` | `hybrid_super_smeans_wide` | `super_emeans_wide` | `test_mse` | 30 | -0.2192% | +0.0267% | 14/0/16 | 208/0/242 | 1.069e-01 | 2.621e-01 |
| `T=1` | `hybrid_super_smeans_wide` | `super_emeans_wide` | `test_mse` | 10 | -0.1180% | -0.0241% | 5/0/5 | 72/0/78 | 4.501e-01 | 5.566e-01 |
| `T=2` | `hybrid_super_smeans_wide` | `super_emeans_wide` | `test_mse` | 10 | -0.2497% | -0.0327% | 5/0/5 | 70/0/80 | 2.868e-01 | 4.316e-01 |
| `T=5` | `hybrid_super_smeans_wide` | `super_emeans_wide` | `test_mse` | 10 | -0.2898% | +0.0489% | 4/0/6 | 66/0/84 | 3.716e-01 | 6.953e-01 |
| `all_T` | `hybrid_super_smeans_wide` | `super_emeans_wide` | `total_time_sec` | 30 | **-1.3600%** | -1.6061% | 26/0/4 | 348/0/102 | 7.479e-03 | 1.886e-04 |
| `T=1` | `hybrid_super_smeans_wide` | `super_emeans_wide` | `total_time_sec` | 10 | **-1.8089%** | -1.5129% | 8/0/2 | 121/0/29 | 1.937e-02 | 9.766e-03 |
| `T=2` | `hybrid_super_smeans_wide` | `super_emeans_wide` | `total_time_sec` | 10 | -1.7209% | -1.8753% | 9/0/1 | 119/0/31 | 9.053e-02 | 8.398e-02 |
| `T=5` | `hybrid_super_smeans_wide` | `super_emeans_wide` | `total_time_sec` | 10 | -0.5501% | -1.4760% | 9/0/1 | 108/0/42 | 5.591e-01 | 8.398e-02 |
| `all_T` | `hybrid_super_smeans_wide` | `super_emeans` | `test_mse` | 30 | -0.0170% | +0.0421% | 13/0/17 | 180/0/270 | 9.038e-01 | 8.712e-01 |
| `T=1` | `hybrid_super_smeans_wide` | `super_emeans` | `test_mse` | 10 | +0.2496% | +0.1847% | 4/0/6 | 59/0/91 | 2.964e-01 | 4.316e-01 |
| `T=2` | `hybrid_super_smeans_wide` | `super_emeans` | `test_mse` | 10 | -0.1197% | -0.0343% | 5/0/5 | 68/0/82 | 5.129e-01 | 4.922e-01 |
| `T=5` | `hybrid_super_smeans_wide` | `super_emeans` | `test_mse` | 10 | -0.1808% | +0.0421% | 4/0/6 | 53/0/97 | 5.700e-01 | 8.457e-01 |
| `all_T` | `hybrid_super_smeans_wide` | `super_emeans` | `total_time_sec` | 30 | **-24.5456%** | -24.0871% | 30/0/0 | 450/0/0 | 1.093e-24 | 1.863e-09 |
| `T=1` | `hybrid_super_smeans_wide` | `super_emeans` | `total_time_sec` | 10 | **-24.4452%** | -23.6894% | 10/0/0 | 150/0/0 | 1.972e-07 | 1.953e-03 |
| `T=2` | `hybrid_super_smeans_wide` | `super_emeans` | `total_time_sec` | 10 | **-23.1202%** | -22.7050% | 10/0/0 | 150/0/0 | 1.253e-09 | 1.953e-03 |
| `T=5` | `hybrid_super_smeans_wide` | `super_emeans` | `total_time_sec` | 10 | **-26.0714%** | -25.5216% | 10/0/0 | 150/0/0 | 4.658e-10 | 1.953e-03 |
| `all_T` | `super_emeans` | `lloyd_25` | `test_mse` | 30 | **-6.8906%** | -0.5410% | 24/0/6 | 357/0/93 | 2.459e-03 | 2.091e-04 |
| `T=1` | `super_emeans` | `lloyd_25` | `test_mse` | 10 | -5.8496% | -0.0067% | 5/0/5 | 78/0/72 | 1.344e-01 | 4.316e-01 |
| `T=2` | `super_emeans` | `lloyd_25` | `test_mse` | 10 | **-6.8621%** | -0.5395% | 9/0/1 | 131/0/19 | 8.964e-02 | 9.766e-03 |
| `T=5` | `super_emeans` | `lloyd_25` | `test_mse` | 10 | **-7.9600%** | -0.9104% | 10/0/0 | 148/0/2 | 7.636e-02 | 1.953e-03 |
| `all_T` | `super_emeans` | `lloyd_25` | `total_time_sec` | 30 | +592.4732% | +514.4098% | 0/0/30 | 0/0/450 | 1.581e-09 | 1.863e-09 |
| `T=1` | `super_emeans` | `lloyd_25` | `total_time_sec` | 10 | +474.9166% | +418.1949% | 0/0/10 | 0/0/150 | 1.945e-04 | 1.953e-03 |
| `T=2` | `super_emeans` | `lloyd_25` | `total_time_sec` | 10 | +534.2020% | +509.2576% | 0/0/10 | 0/0/150 | 4.523e-05 | 1.953e-03 |
| `T=5` | `super_emeans` | `lloyd_25` | `total_time_sec` | 10 | +768.3009% | +619.7344% | 0/0/10 | 0/0/150 | 1.374e-03 | 1.953e-03 |
| `all_T` | `hybrid_super_smeans` | `lloyd_25` | `test_mse` | 30 | **-7.0353%** | -0.3570% | 19/0/11 | 285/0/165 | 2.763e-03 | 1.745e-02 |
| `T=1` | `hybrid_super_smeans` | `lloyd_25` | `test_mse` | 10 | -5.8914% | +0.2877% | 4/0/6 | 61/0/89 | 1.407e-01 | 5.566e-01 |
| `T=2` | `hybrid_super_smeans` | `lloyd_25` | `test_mse` | 10 | -6.9738% | -0.0601% | 6/0/4 | 92/0/58 | 9.920e-02 | 2.754e-01 |
| `T=5` | `hybrid_super_smeans` | `lloyd_25` | `test_mse` | 10 | **-8.2408%** | -0.5594% | 9/0/1 | 132/0/18 | 7.457e-02 | 2.734e-02 |
| `all_T` | `hybrid_super_smeans` | `lloyd_25` | `total_time_sec` | 30 | +574.6219% | +501.7782% | 0/0/30 | 0/0/450 | 1.305e-09 | 1.863e-09 |
| `T=1` | `hybrid_super_smeans` | `lloyd_25` | `total_time_sec` | 10 | +459.2736% | +401.4494% | 0/0/10 | 0/0/150 | 1.394e-04 | 1.953e-03 |
| `T=2` | `hybrid_super_smeans` | `lloyd_25` | `total_time_sec` | 10 | +512.4127% | +483.7697% | 0/0/10 | 0/0/150 | 3.455e-05 | 1.953e-03 |
| `T=5` | `hybrid_super_smeans` | `lloyd_25` | `total_time_sec` | 10 | +752.1795% | +597.8653% | 0/0/10 | 0/0/150 | 1.265e-03 | 1.953e-03 |
| `all_T` | `smeans_fast` | `lloyd_25` | `test_mse` | 30 | -4.7689% | -0.1088% | 17/0/13 | 238/0/212 | 1.799e-02 | 2.449e-01 |
| `T=1` | `smeans_fast` | `lloyd_25` | `test_mse` | 10 | -2.4584% | +0.3918% | 4/0/6 | 53/0/97 | 4.511e-01 | 7.695e-01 |
| `T=2` | `smeans_fast` | `lloyd_25` | `test_mse` | 10 | -4.8370% | -0.0959% | 6/0/4 | 81/0/69 | 1.826e-01 | 3.750e-01 |
| `T=5` | `smeans_fast` | `lloyd_25` | `test_mse` | 10 | -7.0112% | -0.4324% | 7/0/3 | 104/0/46 | 8.210e-02 | 1.934e-01 |
| `all_T` | `smeans_fast` | `lloyd_25` | `total_time_sec` | 30 | **-23.0593%** | -33.8891% | 27/0/3 | 403/0/47 | 9.595e-03 | 2.020e-03 |
| `T=1` | `smeans_fast` | `lloyd_25` | `total_time_sec` | 10 | -32.7157% | -45.6176% | 9/0/1 | 135/0/15 | 3.468e-02 | 8.398e-02 |
| `T=2` | `smeans_fast` | `lloyd_25` | `total_time_sec` | 10 | -28.0579% | -33.2489% | 9/0/1 | 135/0/15 | 5.055e-02 | 8.398e-02 |
| `T=5` | `smeans_fast` | `lloyd_25` | `total_time_sec` | 10 | -8.4043% | -25.0203% | 9/0/1 | 133/0/17 | 6.409e-01 | 8.398e-02 |

## 2. Suite 2 Directional Component Attribution Aggregates (`Test MSE` & `Total Time`)

| Arm | Baseline | Metric | Cells | Mean Delta (%) | Median Delta (%) | Cell W/T/L | Seed W/T/L | `p_ttest` | `p_wilcox` |
| :--- | :--- | :--- | ---: | ---: | ---: | :---: | :---: | ---: | ---: |
| `s2a_loo_no_d15_init` | `s2a_smeans_full` | `test_mse` | 12 | +0.7819% | +0.1253% | 3/0/9 | 69/0/111 | 7.509e-02 | 5.225e-02 |
| `s2a_loo_no_d15_init` | `s2a_smeans_full` | `total_time_sec` | 12 | **-10.0362%** | -10.7778% | 12/0/0 | 180/0/0 | 5.614e-07 | 4.883e-04 |
| `s2a_loo_no_power_cos` | `s2a_smeans_full` | `test_mse` | 12 | +0.0385% | +0.0075% | 6/0/6 | 88/0/92 | 2.985e-01 | 4.697e-01 |
| `s2a_loo_no_power_cos` | `s2a_smeans_full` | `total_time_sec` | 12 | **-0.9856%** | -0.9824% | 10/0/2 | 158/0/22 | 2.387e-03 | 2.441e-03 |
| `s2a_loo_no_revive_stop` | `s2a_smeans_full` | `test_mse` | 12 | +0.0063% | -0.0001% | 9/0/3 | 107/0/73 | 4.784e-01 | 5.693e-01 |
| `s2a_loo_no_revive_stop` | `s2a_smeans_full` | `total_time_sec` | 12 | +6.0358% | +6.3325% | 0/0/12 | 5/0/175 | 1.764e-06 | 4.883e-04 |
| `s2a_loo_no_sse_donor` | `s2a_smeans_full` | `test_mse` | 12 | +2.6180% | +0.7624% | 2/0/10 | 32/0/148 | 5.193e-02 | 6.836e-03 |
| `s2a_loo_no_sse_donor` | `s2a_smeans_full` | `total_time_sec` | 12 | **-4.6667%** | -4.8747% | 12/0/0 | 178/0/2 | 3.718e-05 | 4.883e-04 |
| `s2a_loo_no_rms_split` | `s2a_smeans_full` | `test_mse` | 12 | -0.4416% | -0.1737% | 6/0/6 | 109/0/71 | 3.899e-01 | 5.186e-01 |
| `s2a_loo_no_rms_split` | `s2a_smeans_full` | `total_time_sec` | 12 | **-16.8196%** | -14.8627% | 12/0/0 | 180/0/0 | 4.592e-06 | 4.883e-04 |
| `s2a_loo_no_feat_scale` | `s2a_smeans_full` | `test_mse` | 12 | +0.0693% | +0.0549% | 3/0/9 | 69/0/111 | 1.045e-01 | 1.514e-01 |
| `s2a_loo_no_feat_scale` | `s2a_smeans_full` | `total_time_sec` | 12 | **-16.8518%** | -14.7981% | 12/0/0 | 180/0/0 | 4.325e-06 | 4.883e-04 |
| `s2a_loo_no_pass_weight` | `s2a_smeans_full` | `test_mse` | 12 | +0.0373% | +0.0241% | 3/0/9 | 62/0/118 | 6.232e-02 | 3.418e-02 |
| `s2a_loo_no_pass_weight` | `s2a_smeans_full` | `total_time_sec` | 12 | **-0.9848%** | -0.9205% | 11/0/1 | 152/0/28 | 1.956e-02 | 1.221e-02 |
| `s2a_loo_no_eb_finalize` | `s2a_smeans_full` | `test_mse` | 12 | +0.0104% | +0.0011% | 4/0/8 | 53/0/127 | 1.380e-01 | 1.514e-01 |
| `s2a_loo_no_eb_finalize` | `s2a_smeans_full` | `total_time_sec` | 12 | **-0.9749%** | -0.7408% | 12/0/0 | 154/0/26 | 3.089e-03 | 4.883e-04 |
| `s2a_smeans_alpha_coupled_sse` | `s2a_smeans_full` | `test_mse` | 12 | +0.1413% | +0.1360% | 4/0/8 | 64/0/116 | 2.047e-01 | 2.661e-01 |
| `s2a_smeans_alpha_coupled_sse` | `s2a_smeans_full` | `total_time_sec` | 12 | -0.7888% | -0.7502% | 9/0/3 | 147/0/33 | 7.476e-02 | 6.396e-02 |
| `s2a_all_off_emeans_equiv` | `s2a_smeans_full` | `test_mse` | 12 | +4.1361% | +1.3260% | 4/0/8 | 62/0/118 | 9.861e-02 | 1.099e-01 |
| `s2a_all_off_emeans_equiv` | `s2a_smeans_full` | `total_time_sec` | 12 | **-26.5376%** | -25.7350% | 12/0/0 | 180/0/0 | 5.406e-08 | 4.883e-04 |
| `s2bc_pp_plus_c1_d15_raw` | `s2bc_emeans_pp_base` | `test_mse` | 12 | +1.1848% | +0.0581% | 2/0/10 | 67/0/113 | 1.008e-01 | 6.396e-02 |
| `s2bc_pp_plus_c1_d15_raw` | `s2bc_emeans_pp_base` | `total_time_sec` | 12 | **-42.2030%** | -44.5733% | 12/0/0 | 180/0/0 | 4.477e-07 | 4.883e-04 |
| `s2bc_pp_plus_c1_d15_polished` | `s2bc_emeans_pp_base` | `test_mse` | 12 | +0.9115% | +0.0290% | 4/0/8 | 81/0/99 | 1.912e-01 | 5.186e-01 |
| `s2bc_pp_plus_c1_d15_polished` | `s2bc_emeans_pp_base` | `total_time_sec` | 12 | **-40.5820%** | -42.6578% | 12/0/0 | 179/0/1 | 5.802e-07 | 4.883e-04 |
| `s2bc_pp_plus_c2_power_cos` | `s2bc_emeans_pp_base` | `test_mse` | 12 | -0.0191% | +0.0137% | 4/0/8 | 83/0/97 | 5.572e-01 | 7.334e-01 |
| `s2bc_pp_plus_c2_power_cos` | `s2bc_emeans_pp_base` | `total_time_sec` | 12 | **-0.5199%** | -0.2806% | 9/0/3 | 134/0/46 | 2.011e-02 | 6.836e-03 |
| `s2bc_pp_plus_c3_revive_stop` | `s2bc_emeans_pp_base` | `test_mse` | 12 | -0.0104% | -0.0010% | 7/0/5 | 105/1/74 | 1.232e-01 | 1.514e-01 |
| `s2bc_pp_plus_c3_revive_stop` | `s2bc_emeans_pp_base` | `total_time_sec` | 12 | **-4.5040%** | -4.1215% | 12/0/0 | 178/0/2 | 1.597e-06 | 4.883e-04 |
| `s2bc_pp_plus_c4_sse_usage_donor` | `s2bc_emeans_pp_base` | `test_mse` | 12 | -0.2096% | -0.0167% | 9/0/3 | 113/0/67 | 1.305e-01 | 1.294e-01 |
| `s2bc_pp_plus_c4_sse_usage_donor` | `s2bc_emeans_pp_base` | `total_time_sec` | 12 | **-0.7193%** | -0.3913% | 12/0/0 | 165/0/15 | 6.989e-03 | 4.883e-04 |
| `s2bc_pp_plus_c5_rms_aniso_split` | `s2bc_emeans_pp_base` | `test_mse` | 12 | +0.1146% | +0.0003% | 6/0/6 | 82/0/98 | 8.047e-01 | 8.501e-01 |
| `s2bc_pp_plus_c5_rms_aniso_split` | `s2bc_emeans_pp_base` | `total_time_sec` | 12 | +8.4535% | +8.0510% | 0/0/12 | 1/0/179 | 8.760e-09 | 4.883e-04 |
| `s2bc_pp_plus_c6_pass_weight` | `s2bc_emeans_pp_base` | `test_mse` | 12 | -0.0158% | -0.0101% | 7/0/5 | 92/0/88 | 4.906e-01 | 4.238e-01 |
| `s2bc_pp_plus_c6_pass_weight` | `s2bc_emeans_pp_base` | `total_time_sec` | 12 | **-0.4146%** | -0.1963% | 11/0/1 | 121/0/59 | 3.725e-02 | 6.836e-03 |
| `s2bc_pp_plus_c7_eb_ema` | `s2bc_emeans_pp_base` | `test_mse` | 12 | -0.0012% | -0.0004% | 10/0/2 | 148/0/32 | 2.821e-01 | 9.229e-02 |
| `s2bc_pp_plus_c7_eb_ema` | `s2bc_emeans_pp_base` | `total_time_sec` | 12 | **-0.4174%** | -0.2280% | 8/0/4 | 111/0/69 | 6.142e-02 | 3.418e-02 |
| `s2bc_super_plus_c1_d15_init` | `s2bc_super_emeans` | `test_mse` | 12 | +0.7751% | +0.0103% | 4/0/8 | 92/0/88 | 2.196e-01 | 5.186e-01 |
| `s2bc_super_plus_c1_d15_init` | `s2bc_super_emeans` | `total_time_sec` | 12 | **-22.2887%** | -22.9972% | 12/0/0 | 178/0/2 | 9.913e-06 | 4.883e-04 |
| `s2bc_super_plus_c2_power_cos` | `s2bc_super_emeans` | `test_mse` | 12 | -0.0321% | -0.0065% | 6/0/6 | 89/0/91 | 1.162e-01 | 3.013e-01 |
| `s2bc_super_plus_c2_power_cos` | `s2bc_super_emeans` | `total_time_sec` | 12 | -0.0960% | -0.1167% | 7/0/5 | 92/0/88 | 6.108e-01 | 6.221e-01 |
| `s2bc_super_plus_c3_revive_stop` | `s2bc_super_emeans` | `test_mse` | 12 | -0.0041% | -0.0003% | 8/0/4 | 93/1/86 | 3.484e-01 | 5.186e-01 |
| `s2bc_super_plus_c3_revive_stop` | `s2bc_super_emeans` | `total_time_sec` | 12 | **-3.0693%** | -2.8173% | 12/0/0 | 170/0/10 | 2.472e-06 | 4.883e-04 |
| `s2bc_super_plus_c4_sse_usage_donor` | `s2bc_super_emeans` | `test_mse` | 12 | -0.1766% | -0.0193% | 8/0/4 | 112/0/68 | 1.331e-01 | 6.396e-02 |
| `s2bc_super_plus_c4_sse_usage_donor` | `s2bc_super_emeans` | `total_time_sec` | 12 | **-0.7990%** | -0.3557% | 10/0/2 | 126/0/54 | 6.927e-02 | 4.248e-02 |
| `s2bc_super_plus_c5_rms_aniso_split` | `s2bc_super_emeans` | `test_mse` | 12 | +0.3636% | +0.0192% | 5/0/7 | 73/0/107 | 4.066e-01 | 5.186e-01 |
| `s2bc_super_plus_c5_rms_aniso_split` | `s2bc_super_emeans` | `total_time_sec` | 12 | **-7.1160%** | -5.8589% | 12/0/0 | 179/0/1 | 8.616e-07 | 4.883e-04 |
| `s2bc_super_plus_c5b_oja_rms_split` | `s2bc_super_emeans` | `test_mse` | 12 | -0.3935% | -0.0467% | 6/0/6 | 87/0/93 | 1.858e-01 | 3.394e-01 |
| `s2bc_super_plus_c5b_oja_rms_split` | `s2bc_super_emeans` | `total_time_sec` | 12 | +4.8457% | +5.3869% | 0/0/12 | 18/0/162 | 2.082e-05 | 4.883e-04 |
| `s2bc_super_plus_c6_pass_weight` | `s2bc_super_emeans` | `test_mse` | 12 | -0.0142% | -0.0046% | 7/0/5 | 89/0/91 | 5.120e-01 | 9.697e-01 |
| `s2bc_super_plus_c6_pass_weight` | `s2bc_super_emeans` | `total_time_sec` | 12 | -0.1523% | +0.0434% | 6/0/6 | 67/0/113 | 4.225e-01 | 7.910e-01 |
| `s2bc_super_plus_c7_eb_ema` | `s2bc_super_emeans` | `test_mse` | 12 | +0.0620% | +0.0242% | 5/0/7 | 61/0/119 | 1.285e-01 | 1.763e-01 |
| `s2bc_super_plus_c7_eb_ema` | `s2bc_super_emeans` | `total_time_sec` | 12 | **-6.8611%** | -6.4938% | 12/0/0 | 178/0/2 | 8.434e-08 | 4.883e-04 |
| `s2bc_smeans_plus_p1_coreset_voronoi` | `s2bc_smeans_deconfounded` | `test_mse` | 12 | -1.0707% | -0.0596% | 7/0/5 | 105/0/75 | 1.916e-01 | 3.013e-01 |
| `s2bc_smeans_plus_p1_coreset_voronoi` | `s2bc_smeans_deconfounded` | `total_time_sec` | 12 | +87.7102% | +87.6933% | 0/0/12 | 0/0/180 | 2.176e-05 | 4.883e-04 |
| `s2bc_smeans_plus_p2_huygens_shield` | `s2bc_smeans_deconfounded` | `test_mse` | 12 | -0.4892% | +0.0182% | 5/0/7 | 79/0/101 | 2.865e-01 | 9.097e-01 |
| `s2bc_smeans_plus_p2_huygens_shield` | `s2bc_smeans_deconfounded` | `total_time_sec` | 12 | +19.1895% | +18.9145% | 0/0/12 | 0/0/180 | 8.764e-14 | 4.883e-04 |
| `s2bc_smeans_plus_p3_oja_split` | `s2bc_smeans_deconfounded` | `test_mse` | 12 | **-0.6794%** | -0.5382% | 10/0/2 | 133/0/47 | 1.109e-02 | 2.100e-02 |
| `s2bc_smeans_plus_p3_oja_split` | `s2bc_smeans_deconfounded` | `total_time_sec` | 12 | +36.3940% | +35.6338% | 0/0/12 | 0/0/180 | 1.946e-14 | 4.883e-04 |
| `s2bc_smeans_plus_p3_oja008` | `s2bc_smeans_deconfounded` | `test_mse` | 12 | -0.6232% | -0.2526% | 8/0/4 | 118/0/62 | 1.862e-01 | 2.661e-01 |
| `s2bc_smeans_plus_p3_oja008` | `s2bc_smeans_deconfounded` | `total_time_sec` | 12 | +17.2426% | +16.6251% | 0/0/12 | 0/0/180 | 6.061e-12 | 4.883e-04 |
| `s2bc_smeans_plus_p4_eb_js` | `s2bc_smeans_deconfounded` | `test_mse` | 12 | +0.0172% | +0.0104% | 4/0/8 | 56/0/124 | 2.289e-01 | 2.036e-01 |
| `s2bc_smeans_plus_p4_eb_js` | `s2bc_smeans_deconfounded` | `total_time_sec` | 12 | +21.7291% | +21.8368% | 0/0/12 | 0/0/180 | 3.674e-14 | 4.883e-04 |
| `s2bc_smeans_plus_p5_head_edit` | `s2bc_smeans_deconfounded` | `test_mse` | 12 | -1.0245% | -0.0080% | 7/0/5 | 105/0/75 | 1.610e-01 | 2.036e-01 |
| `s2bc_smeans_plus_p5_head_edit` | `s2bc_smeans_deconfounded` | `total_time_sec` | 12 | +114.5604% | +111.8168% | 0/0/12 | 0/0/180 | 2.390e-07 | 4.883e-04 |
| `s2bc_smeans_plus_p6_wide_step` | `s2bc_smeans_deconfounded` | `test_mse` | 12 | -0.0948% | -0.1008% | 7/0/5 | 109/0/71 | 5.008e-01 | 4.697e-01 |
| `s2bc_smeans_plus_p6_wide_step` | `s2bc_smeans_deconfounded` | `total_time_sec` | 12 | **-31.8726%** | -31.5676% | 12/0/0 | 180/0/0 | 3.796e-15 | 4.883e-04 |
| `s2bc_super_emeans` | `s2bc_emeans_pp_base` | `test_mse` | 12 | **-0.8244%** | -0.1002% | 11/0/1 | 142/0/38 | 1.059e-01 | 9.766e-04 |
| `s2bc_super_emeans` | `s2bc_emeans_pp_base` | `total_time_sec` | 12 | +82.6275% | +71.1466% | 0/0/12 | 0/0/180 | 3.752e-07 | 4.883e-04 |
| `s2bc_super_emeans` | `s2bc_smeans_deconfounded` | `test_mse` | 12 | **-2.4559%** | -1.2689% | 10/0/2 | 143/0/37 | 2.108e-02 | 1.611e-02 |
| `s2bc_super_emeans` | `s2bc_smeans_deconfounded` | `total_time_sec` | 12 | +255.5875% | +263.7966% | 0/0/12 | 0/0/180 | 4.226e-09 | 4.883e-04 |
| `s2bc_super_emeans_wide` | `s2bc_super_emeans` | `test_mse` | 12 | +0.0931% | +0.0565% | 4/0/8 | 70/0/110 | 7.513e-02 | 9.229e-02 |
| `s2bc_super_emeans_wide` | `s2bc_super_emeans` | `total_time_sec` | 12 | **-23.0304%** | -22.7573% | 12/0/0 | 180/0/0 | 8.064e-11 | 4.883e-04 |
| `s2bc_super_emeans_wide` | `s2bc_smeans_deconfounded` | `test_mse` | 12 | **-2.3718%** | -1.2172% | 10/0/2 | 143/0/37 | 2.093e-02 | 1.611e-02 |
| `s2bc_super_emeans_wide` | `s2bc_smeans_deconfounded` | `total_time_sec` | 12 | +174.4683% | +186.1689% | 0/0/12 | 0/0/180 | 7.531e-08 | 4.883e-04 |
| `s2bc_hybrid_super_smeans` | `s2bc_super_emeans` | `test_mse` | 12 | -0.4139% | -0.0407% | 6/0/6 | 89/0/91 | 1.630e-01 | 3.394e-01 |
| `s2bc_hybrid_super_smeans` | `s2bc_super_emeans` | `total_time_sec` | 12 | **-3.5861%** | -3.1599% | 12/0/0 | 159/0/21 | 4.457e-05 | 4.883e-04 |
| `s2bc_hybrid_super_smeans` | `s2bc_emeans_pp_base` | `test_mse` | 12 | -1.2300% | -0.0512% | 6/0/6 | 101/0/79 | 7.197e-02 | 3.013e-01 |
| `s2bc_hybrid_super_smeans` | `s2bc_emeans_pp_base` | `total_time_sec` | 12 | +76.0310% | +62.9734% | 0/0/12 | 0/0/180 | 7.031e-07 | 4.883e-04 |
| `s2bc_hybrid_super_smeans` | `s2bc_smeans_deconfounded` | `test_mse` | 12 | **-2.8569%** | -1.3954% | 10/0/2 | 157/0/23 | 1.451e-02 | 4.883e-03 |
| `s2bc_hybrid_super_smeans` | `s2bc_smeans_deconfounded` | `total_time_sec` | 12 | +243.1124% | +253.4739% | 0/0/12 | 0/0/180 | 7.596e-09 | 4.883e-04 |
| `s2bc_hybrid_super_smeans_wide` | `s2bc_super_emeans_wide` | `test_mse` | 12 | -0.4176% | -0.0327% | 6/0/6 | 91/0/89 | 1.380e-01 | 3.394e-01 |
| `s2bc_hybrid_super_smeans_wide` | `s2bc_super_emeans_wide` | `total_time_sec` | 12 | **-2.7820%** | -2.3471% | 12/0/0 | 152/0/28 | 1.443e-04 | 4.883e-04 |
| `s2bc_hybrid_super_smeans_wide` | `s2bc_super_emeans` | `test_mse` | 12 | -0.3265% | -0.0342% | 6/0/6 | 87/0/93 | 2.311e-01 | 3.804e-01 |
| `s2bc_hybrid_super_smeans_wide` | `s2bc_super_emeans` | `total_time_sec` | 12 | **-25.2142%** | -25.9188% | 12/0/0 | 180/0/0 | 1.807e-11 | 4.883e-04 |
| `s2bc_hybrid_super_smeans_wide` | `s2bc_smeans_deconfounded` | `test_mse` | 12 | **-2.7800%** | -1.5529% | 11/0/1 | 159/0/21 | 1.213e-02 | 2.441e-03 |
| `s2bc_hybrid_super_smeans_wide` | `s2bc_smeans_deconfounded` | `total_time_sec` | 12 | +166.9036% | +179.5309% | 0/0/12 | 0/0/180 | 1.045e-07 | 4.883e-04 |

## 3. Rare-Mode Survival (`multiscale_rare_outliers`, Huygens Shield vs. Revival Assassination)

| Cell | Arm | Mean Rare Modes Preserved (%) | Std (%) |
| :--- | :--- | ---: | ---: |
| `s1_multiscale_rare_t1` | `emeans_pp` | 26.54% | 4.56% |
| `s1_multiscale_rare_t1` | `hybrid_super_smeans` | 29.02% | 4.70% |
| `s1_multiscale_rare_t1` | `hybrid_super_smeans_wide` | 30.33% | 4.25% |
| `s1_multiscale_rare_t1` | `lloyd_25` | 37.12% | 8.75% |
| `s1_multiscale_rare_t1` | `smeans_exact_ref` | 0.00% | 0.00% |
| `s1_multiscale_rare_t1` | `smeans_fast` | 0.00% | 0.00% |
| `s1_multiscale_rare_t1` | `super_emeans` | 27.58% | 4.60% |
| `s1_multiscale_rare_t1` | `super_emeans_wide` | 29.28% | 4.02% |
| `s1_multiscale_rare_t1` | `vanilla_emeans` | 0.00% | 0.00% |
| `s1_multiscale_rare_t2` | `emeans_pp` | 39.87% | 4.61% |
| `s1_multiscale_rare_t2` | `hybrid_super_smeans` | 46.80% | 3.30% |
| `s1_multiscale_rare_t2` | `hybrid_super_smeans_wide` | 46.14% | 4.44% |
| `s1_multiscale_rare_t2` | `lloyd_25` | 37.12% | 8.75% |
| `s1_multiscale_rare_t2` | `smeans_exact_ref` | 0.13% | 0.51% |
| `s1_multiscale_rare_t2` | `smeans_fast` | 0.00% | 0.00% |
| `s1_multiscale_rare_t2` | `super_emeans` | 44.05% | 3.62% |
| `s1_multiscale_rare_t2` | `super_emeans_wide` | 44.44% | 5.38% |
| `s1_multiscale_rare_t2` | `vanilla_emeans` | 0.00% | 0.00% |
| `s1_multiscale_rare_t5` | `emeans_pp` | 49.54% | 5.05% |
| `s1_multiscale_rare_t5` | `hybrid_super_smeans` | 63.53% | 6.28% |
| `s1_multiscale_rare_t5` | `hybrid_super_smeans_wide` | 60.00% | 6.06% |
| `s1_multiscale_rare_t5` | `lloyd_25` | 37.12% | 8.75% |
| `s1_multiscale_rare_t5` | `smeans_exact_ref` | 4.97% | 2.44% |
| `s1_multiscale_rare_t5` | `smeans_fast` | 4.18% | 3.47% |
| `s1_multiscale_rare_t5` | `super_emeans` | 60.52% | 5.78% |
| `s1_multiscale_rare_t5` | `super_emeans_wide` | 60.00% | 4.73% |
| `s1_multiscale_rare_t5` | `vanilla_emeans` | 1.05% | 1.46% |

## 4. SKU-Stratified End-to-End GPU Latency Table (Suite 1)

| Blackwell TDP SKU | Arm | Runs | Mean Init (ms) | Mean Train (ms) | Mean Total (ms) | Median Total (ms) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `Blackwell_MaxQ_300W` | `lloyd_25` | 315 | 0.10 | 12.25 | 12.35 | 11.19 |
| `Blackwell_MaxQ_300W` | `vanilla_emeans` | 315 | 0.10 | 25.20 | 25.30 | 13.59 |
| `Blackwell_MaxQ_300W` | `emeans_pp` | 315 | 19.53 | 28.47 | 48.00 | 39.28 |
| `Blackwell_MaxQ_300W` | `smeans_exact_ref` | 315 | 0.56 | 30.13 | 30.69 | 18.43 |
| `Blackwell_MaxQ_300W` | `smeans_fast` | 315 | 0.78 | 10.50 | 11.29 | 7.18 |
| `Blackwell_MaxQ_300W` | `super_emeans` | 315 | 19.81 | 80.78 | 100.59 | 72.29 |
| `Blackwell_MaxQ_300W` | `super_emeans_wide` | 315 | 19.80 | 54.79 | 74.59 | 56.02 |
| `Blackwell_MaxQ_300W` | `hybrid_super_smeans` | 315 | 19.79 | 77.32 | 97.11 | 69.25 |
| `Blackwell_MaxQ_300W` | `hybrid_super_smeans_wide` | 315 | 19.80 | 52.98 | 72.78 | 54.78 |
| `Blackwell_Server_600W` | `lloyd_25` | 45 | 0.13 | 29.15 | 29.27 | 29.27 |
| `Blackwell_Server_600W` | `vanilla_emeans` | 45 | 0.13 | 21.34 | 21.46 | 19.29 |
| `Blackwell_Server_600W` | `emeans_pp` | 45 | 37.46 | 25.57 | 63.02 | 60.66 |
| `Blackwell_Server_600W` | `smeans_exact_ref` | 45 | 0.80 | 26.28 | 27.08 | 23.65 |
| `Blackwell_Server_600W` | `smeans_fast` | 45 | 1.13 | 9.28 | 10.42 | 8.78 |
| `Blackwell_Server_600W` | `super_emeans` | 45 | 38.26 | 69.50 | 107.76 | 101.96 |
| `Blackwell_Server_600W` | `super_emeans_wide` | 45 | 38.26 | 46.77 | 85.03 | 81.69 |
| `Blackwell_Server_600W` | `hybrid_super_smeans` | 45 | 38.24 | 67.82 | 106.06 | 99.46 |
| `Blackwell_Server_600W` | `hybrid_super_smeans_wide` | 45 | 38.25 | 45.39 | 83.64 | 80.38 |
| `Blackwell_Workstation_450W` | `lloyd_25` | 90 | 0.09 | 9.62 | 9.70 | 9.78 |
| `Blackwell_Workstation_450W` | `vanilla_emeans` | 90 | 0.09 | 13.24 | 13.33 | 13.31 |
| `Blackwell_Workstation_450W` | `emeans_pp` | 90 | 13.87 | 16.41 | 30.28 | 30.53 |
| `Blackwell_Workstation_450W` | `smeans_exact_ref` | 90 | 0.52 | 16.81 | 17.33 | 17.52 |
| `Blackwell_Workstation_450W` | `smeans_fast` | 90 | 0.73 | 7.00 | 7.74 | 7.96 |
| `Blackwell_Workstation_450W` | `super_emeans` | 90 | 14.31 | 43.52 | 57.82 | 59.02 |
| `Blackwell_Workstation_450W` | `super_emeans_wide` | 90 | 14.30 | 29.09 | 43.38 | 44.04 |
| `Blackwell_Workstation_450W` | `hybrid_super_smeans` | 90 | 14.29 | 43.32 | 57.61 | 57.62 |
| `Blackwell_Workstation_450W` | `hybrid_super_smeans_wide` | 90 | 14.29 | 29.59 | 43.88 | 43.40 |

## 5. Per-Cell Suite 1 Head-to-Head Paired `Test MSE` & Spherical Cosine Distortion Table

| Cell | Arm | Baseline | Metric | Arm Mean | Base Mean | Mean Delta (%) | W/T/L | `p_ttest` | `p_wilcox` |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | :---: | ---: | ---: |
| `s1_balanced_gmm_t1` | `smeans_fast` | `vanilla_emeans` | `test_mse` | 50.663366 | 55.466108 | **-8.6511%** | 15/0/0 | 3.415e-13 | 6.104e-05 |
| `s1_balanced_gmm_t1` | `smeans_fast` | `smeans_exact_ref` | `test_mse` | 50.663366 | 50.724430 | -0.1147% | 8/0/7 | 7.306e-01 | 7.615e-01 |
| `s1_balanced_gmm_t1` | `super_emeans` | `smeans_fast` | `test_mse` | 48.519268 | 50.663366 | **-4.2241%** | 15/0/0 | 1.676e-09 | 6.104e-05 |
| `s1_balanced_gmm_t1` | `hybrid_super_smeans` | `smeans_fast` | `test_mse` | 48.151394 | 50.663366 | **-4.9478%** | 15/0/0 | 3.900e-10 | 6.104e-05 |
| `s1_balanced_gmm_t1` | `hybrid_super_smeans` | `super_emeans` | `test_mse` | 48.151394 | 48.519268 | **-0.7548%** | 14/0/1 | 1.599e-04 | 1.831e-04 |
| `s1_balanced_gmm_t1` | `hybrid_super_smeans_wide` | `super_emeans_wide` | `test_mse` | 48.296708 | 48.669720 | **-0.7626%** | 13/0/2 | 1.247e-03 | 1.160e-03 |
| `s1_balanced_gmm_t2` | `smeans_fast` | `vanilla_emeans` | `test_mse` | 48.207148 | 52.218158 | **-7.6674%** | 15/0/0 | 3.325e-11 | 6.104e-05 |
| `s1_balanced_gmm_t2` | `smeans_fast` | `smeans_exact_ref` | `test_mse` | 48.207148 | 48.070336 | +0.2916% | 7/0/8 | 3.946e-01 | 5.245e-01 |
| `s1_balanced_gmm_t2` | `super_emeans` | `smeans_fast` | `test_mse` | 47.517359 | 48.207148 | **-1.4246%** | 14/0/1 | 1.042e-04 | 3.052e-04 |
| `s1_balanced_gmm_t2` | `hybrid_super_smeans` | `smeans_fast` | `test_mse` | 46.998296 | 48.207148 | **-2.5021%** | 15/0/0 | 1.179e-07 | 6.104e-05 |
| `s1_balanced_gmm_t2` | `hybrid_super_smeans` | `super_emeans` | `test_mse` | 46.998296 | 47.517359 | **-1.0919%** | 15/0/0 | 9.072e-08 | 6.104e-05 |
| `s1_balanced_gmm_t2` | `hybrid_super_smeans_wide` | `super_emeans_wide` | `test_mse` | 47.224405 | 47.668648 | **-0.9296%** | 15/0/0 | 6.438e-06 | 6.104e-05 |
| `s1_balanced_gmm_t5` | `smeans_fast` | `vanilla_emeans` | `test_mse` | 46.579432 | 49.724290 | **-6.3158%** | 15/0/0 | 9.001e-12 | 6.104e-05 |
| `s1_balanced_gmm_t5` | `smeans_fast` | `smeans_exact_ref` | `test_mse` | 46.579432 | 46.494265 | +0.1893% | 6/0/9 | 5.101e-01 | 4.543e-01 |
| `s1_balanced_gmm_t5` | `super_emeans` | `smeans_fast` | `test_mse` | 45.861310 | 46.579432 | **-1.5367%** | 15/0/0 | 2.628e-05 | 6.104e-05 |
| `s1_balanced_gmm_t5` | `hybrid_super_smeans` | `smeans_fast` | `test_mse` | 45.384178 | 46.579432 | **-2.5608%** | 15/0/0 | 1.495e-07 | 6.104e-05 |
| `s1_balanced_gmm_t5` | `hybrid_super_smeans` | `super_emeans` | `test_mse` | 45.384178 | 45.861310 | **-1.0402%** | 15/0/0 | 2.621e-09 | 6.104e-05 |
| `s1_balanced_gmm_t5` | `hybrid_super_smeans_wide` | `super_emeans_wide` | `test_mse` | 45.875576 | 46.296404 | **-0.9081%** | 15/0/0 | 1.424e-06 | 6.104e-05 |
| `s1_cifar10_patches_128_t1` | `smeans_fast` | `vanilla_emeans` | `test_mse` | 1.464139 | 1.448478 | +1.0812% | 0/0/15 | 7.078e-16 | 6.104e-05 |
| `s1_cifar10_patches_128_t1` | `smeans_fast` | `smeans_exact_ref` | `test_mse` | 1.464139 | 1.464551 | -0.0281% | 10/0/5 | 2.683e-01 | 2.769e-01 |
| `s1_cifar10_patches_128_t1` | `super_emeans` | `smeans_fast` | `test_mse` | 1.445404 | 1.464139 | **-1.2796%** | 15/0/0 | 2.623e-19 | 6.104e-05 |
| `s1_cifar10_patches_128_t1` | `hybrid_super_smeans` | `smeans_fast` | `test_mse` | 1.448451 | 1.464139 | **-1.0714%** | 15/0/0 | 6.241e-18 | 6.104e-05 |
| `s1_cifar10_patches_128_t1` | `hybrid_super_smeans` | `super_emeans` | `test_mse` | 1.448451 | 1.445404 | +0.2109% | 0/0/15 | 9.449e-12 | 6.104e-05 |
| `s1_cifar10_patches_128_t1` | `hybrid_super_smeans_wide` | `super_emeans_wide` | `test_mse` | 1.447955 | 1.445930 | +0.1400% | 0/0/15 | 6.278e-08 | 6.104e-05 |
| `s1_cifar10_patches_128_t2` | `smeans_fast` | `vanilla_emeans` | `test_mse` | 1.441322 | 1.436147 | +0.3604% | 0/0/15 | 1.970e-11 | 6.104e-05 |
| `s1_cifar10_patches_128_t2` | `smeans_fast` | `smeans_exact_ref` | `test_mse` | 1.441322 | 1.441514 | -0.0133% | 8/0/7 | 4.373e-01 | 5.614e-01 |
| `s1_cifar10_patches_128_t2` | `super_emeans` | `smeans_fast` | `test_mse` | 1.435669 | 1.441322 | **-0.3922%** | 15/0/0 | 8.796e-12 | 6.104e-05 |
| `s1_cifar10_patches_128_t2` | `hybrid_super_smeans` | `smeans_fast` | `test_mse` | 1.438467 | 1.441322 | **-0.1980%** | 15/0/0 | 2.119e-08 | 6.104e-05 |
| `s1_cifar10_patches_128_t2` | `hybrid_super_smeans` | `super_emeans` | `test_mse` | 1.438467 | 1.435669 | +0.1950% | 0/0/15 | 1.582e-08 | 6.104e-05 |
| `s1_cifar10_patches_128_t2` | `hybrid_super_smeans_wide` | `super_emeans_wide` | `test_mse` | 1.437703 | 1.435763 | +0.1351% | 0/0/15 | 1.291e-07 | 6.104e-05 |
| `s1_cifar10_patches_128_t5` | `smeans_fast` | `vanilla_emeans` | `test_mse` | 1.432368 | 1.430677 | +0.1182% | 1/0/14 | 2.666e-06 | 1.221e-04 |
| `s1_cifar10_patches_128_t5` | `smeans_fast` | `smeans_exact_ref` | `test_mse` | 1.432368 | 1.432356 | +0.0009% | 6/0/9 | 9.707e-01 | 6.387e-01 |
| `s1_cifar10_patches_128_t5` | `super_emeans` | `smeans_fast` | `test_mse` | 1.430147 | 1.432368 | **-0.1551%** | 15/0/0 | 3.075e-08 | 6.104e-05 |
| `s1_cifar10_patches_128_t5` | `hybrid_super_smeans` | `smeans_fast` | `test_mse` | 1.430648 | 1.432368 | **-0.1201%** | 15/0/0 | 2.135e-06 | 6.104e-05 |
| `s1_cifar10_patches_128_t5` | `hybrid_super_smeans` | `super_emeans` | `test_mse` | 1.430648 | 1.430147 | +0.0351% | 4/0/11 | 5.801e-02 | 6.372e-02 |
| `s1_cifar10_patches_128_t5` | `hybrid_super_smeans_wide` | `super_emeans_wide` | `test_mse` | 1.431331 | 1.429917 | +0.0989% | 0/0/15 | 4.303e-06 | 6.104e-05 |
| `s1_cifar10_r18_512_t1` | `smeans_fast` | `vanilla_emeans` | `test_mse` | 862.979682 | 827.264972 | +4.3173% | 0/0/15 | 2.090e-20 | 6.104e-05 |
| `s1_cifar10_r18_512_t1` | `smeans_fast` | `smeans_exact_ref` | `test_mse` | 862.979682 | 863.361821 | -0.0439% | 7/0/8 | 4.895e-01 | 6.387e-01 |
| `s1_cifar10_r18_512_t1` | `super_emeans` | `smeans_fast` | `test_mse` | 822.348410 | 862.979682 | **-4.7080%** | 15/0/0 | 1.382e-21 | 6.104e-05 |
| `s1_cifar10_r18_512_t1` | `hybrid_super_smeans` | `smeans_fast` | `test_mse` | 826.939966 | 862.979682 | **-4.1760%** | 15/0/0 | 1.114e-20 | 6.104e-05 |
| `s1_cifar10_r18_512_t1` | `hybrid_super_smeans` | `super_emeans` | `test_mse` | 826.939966 | 822.348410 | +0.5584% | 0/0/15 | 3.225e-13 | 6.104e-05 |
| `s1_cifar10_r18_512_t1` | `hybrid_super_smeans_wide` | `super_emeans_wide` | `test_mse` | 826.474571 | 822.979052 | +0.4248% | 0/0/15 | 4.740e-10 | 6.104e-05 |
| `s1_cifar10_r18_512_t2` | `smeans_fast` | `vanilla_emeans` | `test_mse` | 841.840015 | 817.901893 | +2.9269% | 0/0/15 | 2.235e-20 | 6.104e-05 |
| `s1_cifar10_r18_512_t2` | `smeans_fast` | `smeans_exact_ref` | `test_mse` | 841.840015 | 841.606810 | +0.0278% | 6/0/9 | 4.772e-01 | 4.543e-01 |
| `s1_cifar10_r18_512_t2` | `super_emeans` | `smeans_fast` | `test_mse` | 815.164162 | 841.840015 | **-3.1687%** | 15/0/0 | 1.138e-21 | 6.104e-05 |
| `s1_cifar10_r18_512_t2` | `hybrid_super_smeans` | `smeans_fast` | `test_mse` | 821.082462 | 841.840015 | **-2.4656%** | 15/0/0 | 2.623e-17 | 6.104e-05 |
| `s1_cifar10_r18_512_t2` | `hybrid_super_smeans` | `super_emeans` | `test_mse` | 821.082462 | 815.164162 | +0.7261% | 0/0/15 | 4.686e-11 | 6.104e-05 |
| `s1_cifar10_r18_512_t2` | `hybrid_super_smeans_wide` | `super_emeans_wide` | `test_mse` | 820.305956 | 815.959049 | +0.5328% | 0/0/15 | 3.988e-11 | 6.104e-05 |
| `s1_cifar10_r18_512_t5` | `smeans_fast` | `vanilla_emeans` | `test_mse` | 827.078374 | 815.162736 | +1.4618% | 0/0/15 | 1.881e-15 | 6.104e-05 |
| `s1_cifar10_r18_512_t5` | `smeans_fast` | `smeans_exact_ref` | `test_mse` | 827.078374 | 827.232055 | -0.0185% | 8/0/7 | 6.268e-01 | 6.387e-01 |
| `s1_cifar10_r18_512_t5` | `super_emeans` | `smeans_fast` | `test_mse` | 812.373911 | 827.078374 | **-1.7778%** | 15/0/0 | 1.484e-16 | 6.104e-05 |
| `s1_cifar10_r18_512_t5` | `hybrid_super_smeans` | `smeans_fast` | `test_mse` | 816.129980 | 827.078374 | **-1.3236%** | 15/0/0 | 1.282e-15 | 6.104e-05 |
| `s1_cifar10_r18_512_t5` | `hybrid_super_smeans` | `super_emeans` | `test_mse` | 816.129980 | 812.373911 | +0.4625% | 0/0/15 | 1.802e-09 | 6.104e-05 |
| `s1_cifar10_r18_512_t5` | `hybrid_super_smeans_wide` | `super_emeans_wide` | `test_mse` | 816.128869 | 812.356300 | +0.4645% | 0/0/15 | 1.092e-08 | 6.104e-05 |
| `s1_cifar10_r18_512_unit_t1` | `smeans_fast` | `vanilla_emeans` | `test_mse` | 0.368044 | 0.354182 | +3.9139% | 0/0/15 | 1.285e-18 | 6.104e-05 |
| `s1_cifar10_r18_512_unit_t1` | `smeans_fast` | `vanilla_emeans` | `test_cos_dist` | 0.205582 | 0.197050 | +4.3302% | 0/0/15 | 1.269e-18 | 6.104e-05 |
| `s1_cifar10_r18_512_unit_t1` | `smeans_fast` | `smeans_exact_ref` | `test_mse` | 0.368044 | 0.368303 | -0.0704% | 10/0/5 | 2.121e-01 | 1.876e-01 |
| `s1_cifar10_r18_512_unit_t1` | `smeans_fast` | `smeans_exact_ref` | `test_cos_dist` | 0.205582 | 0.205730 | -0.0716% | 10/0/5 | 2.525e-01 | 2.293e-01 |
| `s1_cifar10_r18_512_unit_t1` | `super_emeans` | `smeans_fast` | `test_mse` | 0.352206 | 0.368044 | **-4.3029%** | 15/0/0 | 8.696e-20 | 6.104e-05 |
| `s1_cifar10_r18_512_unit_t1` | `super_emeans` | `smeans_fast` | `test_cos_dist` | 0.195778 | 0.205582 | **-4.7687%** | 15/0/0 | 9.309e-20 | 6.104e-05 |
| `s1_cifar10_r18_512_unit_t1` | `hybrid_super_smeans` | `smeans_fast` | `test_mse` | 0.352929 | 0.368044 | **-4.1065%** | 15/0/0 | 3.112e-20 | 6.104e-05 |
| `s1_cifar10_r18_512_unit_t1` | `hybrid_super_smeans` | `smeans_fast` | `test_cos_dist` | 0.196184 | 0.205582 | **-4.5712%** | 15/0/0 | 3.482e-20 | 6.104e-05 |
| `s1_cifar10_r18_512_unit_t1` | `hybrid_super_smeans` | `super_emeans` | `test_mse` | 0.352929 | 0.352206 | +0.2053% | 0/0/15 | 2.324e-07 | 6.104e-05 |
| `s1_cifar10_r18_512_unit_t1` | `hybrid_super_smeans` | `super_emeans` | `test_cos_dist` | 0.196184 | 0.195778 | +0.2075% | 0/0/15 | 1.245e-06 | 6.104e-05 |
| `s1_cifar10_r18_512_unit_t1` | `hybrid_super_smeans_wide` | `super_emeans_wide` | `test_mse` | 0.352916 | 0.352417 | +0.1415% | 1/0/14 | 1.208e-05 | 1.831e-04 |
| `s1_cifar10_r18_512_unit_t1` | `hybrid_super_smeans_wide` | `super_emeans_wide` | `test_cos_dist` | 0.196161 | 0.195908 | +0.1290% | 1/0/14 | 8.516e-05 | 4.272e-04 |
| `s1_cifar10_r18_512_unit_t2` | `smeans_fast` | `vanilla_emeans` | `test_mse` | 0.360360 | 0.350252 | +2.8861% | 0/0/15 | 2.352e-19 | 6.104e-05 |
| `s1_cifar10_r18_512_unit_t2` | `smeans_fast` | `vanilla_emeans` | `test_cos_dist` | 0.200836 | 0.194530 | +3.2422% | 0/0/15 | 1.442e-19 | 6.104e-05 |
| `s1_cifar10_r18_512_unit_t2` | `smeans_fast` | `smeans_exact_ref` | `test_mse` | 0.360360 | 0.360476 | -0.0320% | 10/0/5 | 4.742e-01 | 3.894e-01 |
| `s1_cifar10_r18_512_unit_t2` | `smeans_fast` | `smeans_exact_ref` | `test_cos_dist` | 0.200836 | 0.200901 | -0.0321% | 10/0/5 | 5.134e-01 | 3.894e-01 |
| `s1_cifar10_r18_512_unit_t2` | `super_emeans` | `smeans_fast` | `test_mse` | 0.349501 | 0.360360 | **-3.0133%** | 15/0/0 | 2.950e-19 | 6.104e-05 |
| `s1_cifar10_r18_512_unit_t2` | `super_emeans` | `smeans_fast` | `test_cos_dist` | 0.194098 | 0.200836 | **-3.3551%** | 15/0/0 | 3.090e-19 | 6.104e-05 |
| `s1_cifar10_r18_512_unit_t2` | `hybrid_super_smeans` | `smeans_fast` | `test_mse` | 0.350866 | 0.360360 | **-2.6345%** | 15/0/0 | 2.507e-19 | 6.104e-05 |
| `s1_cifar10_r18_512_unit_t2` | `hybrid_super_smeans` | `smeans_fast` | `test_cos_dist` | 0.194857 | 0.200836 | **-2.9771%** | 15/0/0 | 3.238e-19 | 6.104e-05 |
| `s1_cifar10_r18_512_unit_t2` | `hybrid_super_smeans` | `super_emeans` | `test_mse` | 0.350866 | 0.349501 | +0.3907% | 0/0/15 | 1.136e-09 | 6.104e-05 |
| `s1_cifar10_r18_512_unit_t2` | `hybrid_super_smeans` | `super_emeans` | `test_cos_dist` | 0.194857 | 0.194098 | +0.3913% | 0/0/15 | 7.155e-09 | 6.104e-05 |
| `s1_cifar10_r18_512_unit_t2` | `hybrid_super_smeans_wide` | `super_emeans_wide` | `test_mse` | 0.350791 | 0.349541 | +0.3576% | 0/0/15 | 5.734e-10 | 6.104e-05 |
| `s1_cifar10_r18_512_unit_t2` | `hybrid_super_smeans_wide` | `super_emeans_wide` | `test_cos_dist` | 0.194801 | 0.194128 | +0.3463% | 0/0/15 | 2.608e-09 | 6.104e-05 |
| `s1_cifar10_r18_512_unit_t5` | `smeans_fast` | `vanilla_emeans` | `test_mse` | 0.353885 | 0.348720 | +1.4813% | 0/0/15 | 9.206e-17 | 6.104e-05 |
| `s1_cifar10_r18_512_unit_t5` | `smeans_fast` | `vanilla_emeans` | `test_cos_dist` | 0.196726 | 0.193552 | +1.6399% | 0/0/15 | 1.008e-16 | 6.104e-05 |
| `s1_cifar10_r18_512_unit_t5` | `smeans_fast` | `smeans_exact_ref` | `test_mse` | 0.353885 | 0.353973 | -0.0245% | 10/0/5 | 6.456e-01 | 5.995e-01 |
| `s1_cifar10_r18_512_unit_t5` | `smeans_fast` | `smeans_exact_ref` | `test_cos_dist` | 0.196726 | 0.196790 | -0.0323% | 10/0/5 | 5.925e-01 | 5.614e-01 |
| `s1_cifar10_r18_512_unit_t5` | `super_emeans` | `smeans_fast` | `test_mse` | 0.348085 | 0.353885 | **-1.6389%** | 15/0/0 | 2.885e-15 | 6.104e-05 |
| `s1_cifar10_r18_512_unit_t5` | `super_emeans` | `smeans_fast` | `test_cos_dist` | 0.193219 | 0.196726 | **-1.7827%** | 15/0/0 | 4.647e-15 | 6.104e-05 |
| `s1_cifar10_r18_512_unit_t5` | `hybrid_super_smeans` | `smeans_fast` | `test_mse` | 0.348943 | 0.353885 | **-1.3966%** | 15/0/0 | 5.196e-16 | 6.104e-05 |
| `s1_cifar10_r18_512_unit_t5` | `hybrid_super_smeans` | `smeans_fast` | `test_cos_dist` | 0.193665 | 0.196726 | **-1.5558%** | 15/0/0 | 4.031e-16 | 6.104e-05 |
| `s1_cifar10_r18_512_unit_t5` | `hybrid_super_smeans` | `super_emeans` | `test_mse` | 0.348943 | 0.348085 | +0.2465% | 0/0/15 | 3.255e-06 | 6.104e-05 |
| `s1_cifar10_r18_512_unit_t5` | `hybrid_super_smeans` | `super_emeans` | `test_cos_dist` | 0.193665 | 0.193219 | +0.2312% | 0/0/15 | 1.996e-05 | 6.104e-05 |
| `s1_cifar10_r18_512_unit_t5` | `hybrid_super_smeans_wide` | `super_emeans_wide` | `test_mse` | 0.349014 | 0.348029 | +0.2829% | 0/0/15 | 1.134e-07 | 6.104e-05 |
| `s1_cifar10_r18_512_unit_t5` | `hybrid_super_smeans_wide` | `super_emeans_wide` | `test_cos_dist` | 0.193709 | 0.193190 | +0.2689% | 0/0/15 | 6.220e-07 | 6.104e-05 |
| `s1_isotropic_ctrl_t1` | `smeans_fast` | `vanilla_emeans` | `test_mse` | 119.759933 | 119.009122 | +0.6309% | 0/0/15 | 7.411e-20 | 6.104e-05 |
| `s1_isotropic_ctrl_t1` | `smeans_fast` | `smeans_exact_ref` | `test_mse` | 119.759933 | 119.748290 | +0.0097% | 7/0/8 | 3.382e-01 | 5.995e-01 |
| `s1_isotropic_ctrl_t1` | `super_emeans` | `smeans_fast` | `test_mse` | 118.982512 | 119.759933 | **-0.6491%** | 15/0/0 | 3.316e-21 | 6.104e-05 |
| `s1_isotropic_ctrl_t1` | `hybrid_super_smeans` | `smeans_fast` | `test_mse` | 119.652501 | 119.759933 | **-0.0897%** | 15/0/0 | 3.162e-07 | 6.104e-05 |
| `s1_isotropic_ctrl_t1` | `hybrid_super_smeans` | `super_emeans` | `test_mse` | 119.652501 | 118.982512 | +0.5631% | 0/0/15 | 1.717e-19 | 6.104e-05 |
| `s1_isotropic_ctrl_t1` | `hybrid_super_smeans_wide` | `super_emeans_wide` | `test_mse` | 119.611564 | 118.964296 | +0.5441% | 0/0/15 | 3.374e-18 | 6.104e-05 |
| `s1_isotropic_ctrl_t2` | `smeans_fast` | `vanilla_emeans` | `test_mse` | 119.507289 | 119.019138 | +0.4101% | 0/0/15 | 5.231e-14 | 6.104e-05 |
| `s1_isotropic_ctrl_t2` | `smeans_fast` | `smeans_exact_ref` | `test_mse` | 119.507289 | 119.495541 | +0.0098% | 8/0/7 | 5.388e-01 | 9.341e-01 |
| `s1_isotropic_ctrl_t2` | `super_emeans` | `smeans_fast` | `test_mse` | 118.992523 | 119.507289 | **-0.4307%** | 15/0/0 | 4.677e-15 | 6.104e-05 |
| `s1_isotropic_ctrl_t2` | `hybrid_super_smeans` | `smeans_fast` | `test_mse` | 120.319987 | 119.507289 | +0.6801% | 0/0/15 | 1.088e-11 | 6.104e-05 |
| `s1_isotropic_ctrl_t2` | `hybrid_super_smeans` | `super_emeans` | `test_mse` | 120.319987 | 118.992523 | +1.1156% | 0/0/15 | 6.800e-15 | 6.104e-05 |
| `s1_isotropic_ctrl_t2` | `hybrid_super_smeans_wide` | `super_emeans_wide` | `test_mse` | 119.662382 | 118.986173 | +0.5683% | 0/0/15 | 6.324e-14 | 6.104e-05 |
| `s1_isotropic_ctrl_t5` | `smeans_fast` | `vanilla_emeans` | `test_mse` | 119.590357 | 119.061670 | +0.4441% | 0/0/15 | 1.159e-11 | 6.104e-05 |
| `s1_isotropic_ctrl_t5` | `smeans_fast` | `smeans_exact_ref` | `test_mse` | 119.590357 | 119.560899 | +0.0247% | 4/0/11 | 2.212e-01 | 2.078e-01 |
| `s1_isotropic_ctrl_t5` | `super_emeans` | `smeans_fast` | `test_mse` | 119.031907 | 119.590357 | **-0.4669%** | 15/0/0 | 4.273e-13 | 6.104e-05 |
| `s1_isotropic_ctrl_t5` | `hybrid_super_smeans` | `smeans_fast` | `test_mse` | 120.185565 | 119.590357 | +0.4978% | 0/0/15 | 5.966e-07 | 6.104e-05 |
| `s1_isotropic_ctrl_t5` | `hybrid_super_smeans` | `super_emeans` | `test_mse` | 120.185565 | 119.031907 | +0.9692% | 0/0/15 | 1.719e-11 | 6.104e-05 |
| `s1_isotropic_ctrl_t5` | `hybrid_super_smeans_wide` | `super_emeans_wide` | `test_mse` | 120.004724 | 119.032326 | +0.8169% | 0/0/15 | 1.195e-07 | 6.104e-05 |
| `s1_low_nk32_gmm_t1` | `smeans_fast` | `vanilla_emeans` | `test_mse` | 57.329431 | 60.953052 | **-5.9425%** | 15/0/0 | 5.380e-11 | 6.104e-05 |
| `s1_low_nk32_gmm_t1` | `smeans_fast` | `smeans_exact_ref` | `test_mse` | 57.329431 | 57.600268 | -0.4692% | 11/0/4 | 1.170e-01 | 1.205e-01 |
| `s1_low_nk32_gmm_t1` | `super_emeans` | `smeans_fast` | `test_mse` | 51.645805 | 57.329431 | **-9.9002%** | 15/0/0 | 1.493e-13 | 6.104e-05 |
| `s1_low_nk32_gmm_t1` | `hybrid_super_smeans` | `smeans_fast` | `test_mse` | 51.270918 | 57.329431 | **-10.5532%** | 15/0/0 | 1.004e-13 | 6.104e-05 |
| `s1_low_nk32_gmm_t1` | `hybrid_super_smeans` | `super_emeans` | `test_mse` | 51.270918 | 51.645805 | **-0.7251%** | 15/0/0 | 2.752e-07 | 6.104e-05 |
| `s1_low_nk32_gmm_t1` | `hybrid_super_smeans_wide` | `super_emeans_wide` | `test_mse` | 51.461068 | 51.851988 | **-0.7532%** | 15/0/0 | 4.520e-08 | 6.104e-05 |
| `s1_low_nk32_gmm_t2` | `smeans_fast` | `vanilla_emeans` | `test_mse` | 54.737641 | 57.728023 | **-5.1730%** | 15/0/0 | 3.167e-09 | 6.104e-05 |
| `s1_low_nk32_gmm_t2` | `smeans_fast` | `smeans_exact_ref` | `test_mse` | 54.737641 | 54.912942 | -0.3190% | 9/0/6 | 2.867e-01 | 3.303e-01 |
| `s1_low_nk32_gmm_t2` | `super_emeans` | `smeans_fast` | `test_mse` | 50.283575 | 54.737641 | **-8.1234%** | 15/0/0 | 1.723e-12 | 6.104e-05 |
| `s1_low_nk32_gmm_t2` | `hybrid_super_smeans` | `smeans_fast` | `test_mse` | 49.815416 | 54.737641 | **-8.9753%** | 15/0/0 | 4.152e-12 | 6.104e-05 |
| `s1_low_nk32_gmm_t2` | `hybrid_super_smeans` | `super_emeans` | `test_mse` | 49.815416 | 50.283575 | **-0.9290%** | 15/0/0 | 8.290e-05 | 6.104e-05 |
| `s1_low_nk32_gmm_t2` | `hybrid_super_smeans_wide` | `super_emeans_wide` | `test_mse` | 49.931325 | 50.425720 | **-0.9780%** | 14/0/1 | 5.202e-05 | 1.831e-04 |
| `s1_low_nk32_gmm_t5` | `smeans_fast` | `vanilla_emeans` | `test_mse` | 49.443620 | 53.212643 | **-7.0811%** | 15/0/0 | 5.182e-16 | 6.104e-05 |
| `s1_low_nk32_gmm_t5` | `smeans_fast` | `smeans_exact_ref` | `test_mse` | 49.443620 | 49.451261 | -0.0143% | 9/0/6 | 9.080e-01 | 1.000e+00 |
| `s1_low_nk32_gmm_t5` | `super_emeans` | `smeans_fast` | `test_mse` | 49.348708 | 49.443620 | -0.1884% | 9/0/6 | 4.689e-01 | 4.212e-01 |
| `s1_low_nk32_gmm_t5` | `hybrid_super_smeans` | `smeans_fast` | `test_mse` | 48.854310 | 49.443620 | **-1.1889%** | 14/0/1 | 1.561e-04 | 3.052e-04 |
| `s1_low_nk32_gmm_t5` | `hybrid_super_smeans` | `super_emeans` | `test_mse` | 48.854310 | 49.348708 | **-0.9994%** | 14/0/1 | 6.019e-05 | 3.052e-04 |
| `s1_low_nk32_gmm_t5` | `hybrid_super_smeans_wide` | `super_emeans_wide` | `test_mse` | 49.165198 | 49.477477 | **-0.6303%** | 13/0/2 | 1.451e-03 | 2.625e-03 |
| `s1_lowd_zipf_d8_t1` | `smeans_fast` | `vanilla_emeans` | `test_mse` | 1.662060 | 1.663033 | -0.0580% | 9/0/6 | 5.000e-01 | 4.887e-01 |
| `s1_lowd_zipf_d8_t1` | `smeans_fast` | `smeans_exact_ref` | `test_mse` | 1.662060 | 1.661411 | +0.0401% | 9/0/6 | 7.452e-01 | 9.341e-01 |
| `s1_lowd_zipf_d8_t1` | `super_emeans` | `smeans_fast` | `test_mse` | 1.665550 | 1.662060 | +0.2103% | 1/0/14 | 1.275e-03 | 3.052e-04 |
| `s1_lowd_zipf_d8_t1` | `hybrid_super_smeans` | `smeans_fast` | `test_mse` | 1.661795 | 1.662060 | -0.0154% | 10/0/5 | 8.534e-01 | 6.387e-01 |
| `s1_lowd_zipf_d8_t1` | `hybrid_super_smeans` | `super_emeans` | `test_mse` | 1.661795 | 1.665550 | **-0.2253%** | 14/0/1 | 6.609e-04 | 4.272e-04 |
| `s1_lowd_zipf_d8_t1` | `hybrid_super_smeans_wide` | `super_emeans_wide` | `test_mse` | 1.661919 | 1.665702 | **-0.2270%** | 15/0/0 | 4.957e-05 | 6.104e-05 |
| `s1_lowd_zipf_d8_t2` | `smeans_fast` | `vanilla_emeans` | `test_mse` | 1.646472 | 1.647821 | -0.0817% | 9/0/6 | 2.397e-01 | 2.524e-01 |
| `s1_lowd_zipf_d8_t2` | `smeans_fast` | `smeans_exact_ref` | `test_mse` | 1.646472 | 1.648215 | -0.1054% | 9/0/6 | 1.479e-01 | 1.514e-01 |
| `s1_lowd_zipf_d8_t2` | `super_emeans` | `smeans_fast` | `test_mse` | 1.651219 | 1.646472 | +0.2887% | 3/0/12 | 1.131e-03 | 2.625e-03 |
| `s1_lowd_zipf_d8_t2` | `hybrid_super_smeans` | `smeans_fast` | `test_mse` | 1.649049 | 1.646472 | +0.1568% | 5/0/10 | 4.516e-02 | 4.791e-02 |
| `s1_lowd_zipf_d8_t2` | `hybrid_super_smeans` | `super_emeans` | `test_mse` | 1.649049 | 1.651219 | **-0.1314%** | 12/0/3 | 8.475e-03 | 1.508e-02 |
| `s1_lowd_zipf_d8_t2` | `hybrid_super_smeans_wide` | `super_emeans_wide` | `test_mse` | 1.649622 | 1.650915 | -0.0780% | 9/0/6 | 2.776e-01 | 2.524e-01 |
| `s1_lowd_zipf_d8_t5` | `smeans_fast` | `vanilla_emeans` | `test_mse` | 1.638241 | 1.639911 | -0.1014% | 9/0/6 | 2.484e-01 | 1.514e-01 |
| `s1_lowd_zipf_d8_t5` | `smeans_fast` | `smeans_exact_ref` | `test_mse` | 1.638241 | 1.639371 | -0.0687% | 9/0/6 | 3.362e-01 | 4.212e-01 |
| `s1_lowd_zipf_d8_t5` | `super_emeans` | `smeans_fast` | `test_mse` | 1.642891 | 1.638241 | +0.2845% | 4/0/11 | 9.764e-03 | 1.025e-02 |
| `s1_lowd_zipf_d8_t5` | `hybrid_super_smeans` | `smeans_fast` | `test_mse` | 1.642696 | 1.638241 | +0.2726% | 1/0/14 | 4.278e-03 | 3.052e-04 |
| `s1_lowd_zipf_d8_t5` | `hybrid_super_smeans` | `super_emeans` | `test_mse` | 1.642696 | 1.642891 | -0.0114% | 7/0/8 | 8.505e-01 | 8.469e-01 |
| `s1_lowd_zipf_d8_t5` | `hybrid_super_smeans_wide` | `super_emeans_wide` | `test_mse` | 1.642153 | 1.641494 | +0.0407% | 8/0/7 | 5.782e-01 | 7.197e-01 |
| `s1_multiscale_rare_t1` | `smeans_fast` | `vanilla_emeans` | `test_mse` | 20.104336 | 25.763136 | **-21.8293%** | 15/0/0 | 1.084e-11 | 6.104e-05 |
| `s1_multiscale_rare_t1` | `smeans_fast` | `smeans_exact_ref` | `test_mse` | 20.104336 | 20.215952 | -0.5273% | 10/0/5 | 2.848e-01 | 1.876e-01 |
| `s1_multiscale_rare_t1` | `super_emeans` | `smeans_fast` | `test_mse` | 18.313394 | 20.104336 | **-8.8948%** | 15/0/0 | 1.504e-10 | 6.104e-05 |
| `s1_multiscale_rare_t1` | `hybrid_super_smeans` | `smeans_fast` | `test_mse` | 18.179289 | 20.104336 | **-9.5662%** | 15/0/0 | 1.129e-11 | 6.104e-05 |
| `s1_multiscale_rare_t1` | `hybrid_super_smeans` | `super_emeans` | `test_mse` | 18.179289 | 18.313394 | **-0.7257%** | 13/0/2 | 3.067e-02 | 3.534e-02 |
| `s1_multiscale_rare_t1` | `hybrid_super_smeans_wide` | `super_emeans_wide` | `test_mse` | 18.682631 | 18.805579 | -0.6394% | 8/0/7 | 1.530e-01 | 3.028e-01 |
| `s1_multiscale_rare_t2` | `smeans_fast` | `vanilla_emeans` | `test_mse` | 19.124537 | 23.712704 | **-19.2280%** | 15/0/0 | 4.033e-11 | 6.104e-05 |
| `s1_multiscale_rare_t2` | `smeans_fast` | `smeans_exact_ref` | `test_mse` | 19.124537 | 19.137391 | -0.0545% | 9/0/6 | 8.705e-01 | 8.469e-01 |
| `s1_multiscale_rare_t2` | `super_emeans` | `smeans_fast` | `test_mse` | 18.124290 | 19.124537 | **-5.2159%** | 14/0/1 | 6.764e-06 | 1.831e-04 |
| `s1_multiscale_rare_t2` | `hybrid_super_smeans` | `smeans_fast` | `test_mse` | 17.797952 | 19.124537 | **-6.9322%** | 15/0/0 | 1.864e-08 | 6.104e-05 |
| `s1_multiscale_rare_t2` | `hybrid_super_smeans` | `super_emeans` | `test_mse` | 17.797952 | 18.124290 | **-1.7812%** | 13/0/2 | 2.667e-03 | 2.014e-03 |
| `s1_multiscale_rare_t2` | `hybrid_super_smeans_wide` | `super_emeans_wide` | `test_mse` | 17.961374 | 18.211018 | **-1.3585%** | 11/0/4 | 2.811e-03 | 5.371e-03 |
| `s1_multiscale_rare_t5` | `smeans_fast` | `vanilla_emeans` | `test_mse` | 18.582627 | 21.458470 | **-13.3251%** | 15/0/0 | 2.001e-10 | 6.104e-05 |
| `s1_multiscale_rare_t5` | `smeans_fast` | `smeans_exact_ref` | `test_mse` | 18.582627 | 18.603746 | -0.1069% | 9/0/6 | 7.533e-01 | 7.615e-01 |
| `s1_multiscale_rare_t5` | `super_emeans` | `smeans_fast` | `test_mse` | 17.142057 | 18.582627 | **-7.7446%** | 15/0/0 | 4.689e-11 | 6.104e-05 |
| `s1_multiscale_rare_t5` | `hybrid_super_smeans` | `smeans_fast` | `test_mse` | 17.012390 | 18.582627 | **-8.4335%** | 15/0/0 | 2.214e-09 | 6.104e-05 |
| `s1_multiscale_rare_t5` | `hybrid_super_smeans` | `super_emeans` | `test_mse` | 17.012390 | 17.142057 | -0.7545% | 10/0/5 | 5.078e-02 | 8.325e-02 |
| `s1_multiscale_rare_t5` | `hybrid_super_smeans_wide` | `super_emeans_wide` | `test_mse` | 17.078779 | 17.163835 | -0.4834% | 9/0/6 | 3.062e-01 | 3.894e-01 |
| `s1_stretched_a2_t1` | `smeans_fast` | `vanilla_emeans` | `test_mse` | 18.991828 | 19.067330 | **-0.3959%** | 15/0/0 | 2.160e-07 | 6.104e-05 |
| `s1_stretched_a2_t1` | `smeans_fast` | `smeans_exact_ref` | `test_mse` | 18.991828 | 18.994096 | -0.0117% | 8/0/7 | 8.461e-01 | 9.341e-01 |
| `s1_stretched_a2_t1` | `super_emeans` | `smeans_fast` | `test_mse` | 18.995447 | 18.991828 | +0.0192% | 8/0/7 | 6.964e-01 | 8.469e-01 |
| `s1_stretched_a2_t1` | `hybrid_super_smeans` | `smeans_fast` | `test_mse` | 18.969526 | 18.991828 | -0.1172% | 12/0/3 | 6.495e-02 | 5.536e-02 |
| `s1_stretched_a2_t1` | `hybrid_super_smeans` | `super_emeans` | `test_mse` | 18.969526 | 18.995447 | **-0.1364%** | 14/0/1 | 1.171e-05 | 1.221e-04 |
| `s1_stretched_a2_t1` | `hybrid_super_smeans_wide` | `super_emeans_wide` | `test_mse` | 18.974020 | 18.999125 | **-0.1321%** | 15/0/0 | 2.951e-07 | 6.104e-05 |
| `s1_stretched_a2_t2` | `smeans_fast` | `vanilla_emeans` | `test_mse` | 18.910416 | 18.938919 | **-0.1502%** | 11/0/4 | 1.699e-02 | 2.155e-02 |
| `s1_stretched_a2_t2` | `smeans_fast` | `smeans_exact_ref` | `test_mse` | 18.910416 | 18.899502 | +0.0580% | 7/0/8 | 3.081e-01 | 5.614e-01 |
| `s1_stretched_a2_t2` | `super_emeans` | `smeans_fast` | `test_mse` | 18.897206 | 18.910416 | -0.0698% | 9/0/6 | 1.642e-01 | 1.688e-01 |
| `s1_stretched_a2_t2` | `hybrid_super_smeans` | `smeans_fast` | `test_mse` | 18.906594 | 18.910416 | -0.0201% | 7/0/8 | 6.166e-01 | 6.387e-01 |
| `s1_stretched_a2_t2` | `hybrid_super_smeans` | `super_emeans` | `test_mse` | 18.906594 | 18.897206 | +0.0498% | 3/0/12 | 6.587e-02 | 5.536e-02 |
| `s1_stretched_a2_t2` | `hybrid_super_smeans_wide` | `super_emeans_wide` | `test_mse` | 18.902468 | 18.900068 | +0.0127% | 6/0/9 | 4.505e-01 | 5.614e-01 |
| `s1_stretched_a2_t5` | `smeans_fast` | `vanilla_emeans` | `test_mse` | 18.885355 | 18.894363 | -0.0475% | 7/0/8 | 2.918e-01 | 4.887e-01 |
| `s1_stretched_a2_t5` | `smeans_fast` | `smeans_exact_ref` | `test_mse` | 18.885355 | 18.881983 | +0.0181% | 7/0/8 | 6.922e-01 | 9.341e-01 |
| `s1_stretched_a2_t5` | `super_emeans` | `smeans_fast` | `test_mse` | 18.868158 | 18.885355 | **-0.0910%** | 10/0/5 | 2.111e-02 | 3.534e-02 |
| `s1_stretched_a2_t5` | `hybrid_super_smeans` | `smeans_fast` | `test_mse` | 18.881582 | 18.885355 | -0.0199% | 8/0/7 | 5.445e-01 | 4.212e-01 |
| `s1_stretched_a2_t5` | `hybrid_super_smeans` | `super_emeans` | `test_mse` | 18.881582 | 18.868158 | +0.0712% | 2/0/13 | 4.563e-03 | 4.272e-03 |
| `s1_stretched_a2_t5` | `hybrid_super_smeans_wide` | `super_emeans_wide` | `test_mse` | 18.878083 | 18.867307 | +0.0572% | 6/0/9 | 6.322e-02 | 1.070e-01 |
| `s1_zipf_gmm_t1` | `smeans_fast` | `vanilla_emeans` | `test_mse` | 50.875044 | 52.163012 | **-2.4666%** | 15/0/0 | 7.359e-08 | 6.104e-05 |
| `s1_zipf_gmm_t1` | `smeans_fast` | `smeans_exact_ref` | `test_mse` | 50.875044 | 50.897117 | -0.0411% | 8/0/7 | 8.452e-01 | 8.469e-01 |
| `s1_zipf_gmm_t1` | `super_emeans` | `smeans_fast` | `test_mse` | 49.539610 | 50.875044 | **-2.6187%** | 15/0/0 | 5.747e-08 | 6.104e-05 |
| `s1_zipf_gmm_t1` | `hybrid_super_smeans` | `smeans_fast` | `test_mse` | 49.642110 | 50.875044 | **-2.4175%** | 15/0/0 | 2.137e-07 | 6.104e-05 |
| `s1_zipf_gmm_t1` | `hybrid_super_smeans` | `super_emeans` | `test_mse` | 49.642110 | 49.539610 | +0.2079% | 5/0/10 | 1.829e-01 | 2.769e-01 |
| `s1_zipf_gmm_t1` | `hybrid_super_smeans_wide` | `super_emeans_wide` | `test_mse` | 49.634744 | 49.593385 | +0.0839% | 5/0/10 | 3.351e-01 | 3.591e-01 |
| `s1_zipf_gmm_t2` | `smeans_fast` | `vanilla_emeans` | `test_mse` | 49.119047 | 50.420832 | **-2.5798%** | 15/0/0 | 1.075e-10 | 6.104e-05 |
| `s1_zipf_gmm_t2` | `smeans_fast` | `smeans_exact_ref` | `test_mse` | 49.119047 | 49.020076 | +0.2043% | 4/0/11 | 2.603e-01 | 1.514e-01 |
| `s1_zipf_gmm_t2` | `super_emeans` | `smeans_fast` | `test_mse` | 48.742663 | 49.119047 | **-0.7646%** | 12/0/3 | 2.691e-04 | 1.160e-03 |
| `s1_zipf_gmm_t2` | `hybrid_super_smeans` | `smeans_fast` | `test_mse` | 48.467202 | 49.119047 | **-1.3250%** | 15/0/0 | 3.964e-06 | 6.104e-05 |
| `s1_zipf_gmm_t2` | `hybrid_super_smeans` | `super_emeans` | `test_mse` | 48.467202 | 48.742663 | **-0.5641%** | 14/0/1 | 8.118e-04 | 3.357e-03 |
| `s1_zipf_gmm_t2` | `hybrid_super_smeans_wide` | `super_emeans_wide` | `test_mse` | 48.421997 | 48.793266 | **-0.7599%** | 15/0/0 | 2.824e-06 | 6.104e-05 |
| `s1_zipf_gmm_t5` | `smeans_fast` | `vanilla_emeans` | `test_mse` | 47.668281 | 49.419161 | **-3.5406%** | 15/0/0 | 2.599e-11 | 6.104e-05 |
| `s1_zipf_gmm_t5` | `smeans_fast` | `smeans_exact_ref` | `test_mse` | 47.668281 | 47.544600 | +0.2622% | 6/0/9 | 1.824e-01 | 1.876e-01 |
| `s1_zipf_gmm_t5` | `super_emeans` | `smeans_fast` | `test_mse` | 48.176734 | 47.668281 | +1.0693% | 2/0/13 | 9.570e-05 | 4.272e-04 |
| `s1_zipf_gmm_t5` | `hybrid_super_smeans` | `smeans_fast` | `test_mse` | 46.951805 | 47.668281 | **-1.5002%** | 15/0/0 | 1.276e-06 | 6.104e-05 |
| `s1_zipf_gmm_t5` | `hybrid_super_smeans` | `super_emeans` | `test_mse` | 46.951805 | 48.176734 | **-2.5410%** | 15/0/0 | 1.087e-11 | 6.104e-05 |
| `s1_zipf_gmm_t5` | `hybrid_super_smeans_wide` | `super_emeans_wide` | `test_mse` | 46.851389 | 48.120872 | **-2.6369%** | 15/0/0 | 1.491e-14 | 6.104e-05 |
