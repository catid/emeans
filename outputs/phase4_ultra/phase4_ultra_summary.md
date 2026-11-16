# Phase 4 Multi-Machine Showdown & Discovery Attribution: `UltraEMeans`

- **Total Validated Paired Runs**: `7,440` (`0` `NaN`/`Inf`, `0` duplicates, single-GPU/single-SKU paired within every cell)
- **Evaluation Protocol**: Exact `FP64` full-dataset `train_mse` and held-out `test_mse` (`exact_mse_fp64`), exact `FP64` spherical cosine distortion on `unit_sphere` cells, rare-mode survival on `multiscale_rare`, and synchronized warm GPU `init_time_sec` / `train_time_sec` / `total_time_sec`.
- **Sign Convention**: In Suite 1, negative `Mean Delta (%)` means **lower MSE (better accuracy)** or **faster wall-clock latency** relative to the baseline. In Suite 2 LOO counterfactuals (`arm` vs `ultra_emeans`), positive `Mean Delta (%)` means removing that discovery **worsened (increased) Test MSE**.

## 1. Suite 1 Executive Showdown Aggregates (`Test MSE` & `Total Time` Across Regimes)

| Scope | Solver | Baseline | Metric | Cells | Mean Delta (%) | Median Delta (%) | Cell W/T/L | Seed W/T/L | `p_ttest` | `p_wilcox` |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | :---: | :---: | ---: | ---: |
| `all_T` | UltraEMeans | e-means | `test_mse` | 30 | **-6.8422%** | -0.3624% | 27/0/3 | 374/0/76 | 6.403e-04 | 5.718e-07 |
| `T=1` | UltraEMeans | e-means | `test_mse` | 10 | **-8.0320%** | -0.4957% | 9/0/1 | 132/0/18 | 5.595e-02 | 5.859e-03 |
| `T=2` | UltraEMeans | e-means | `test_mse` | 10 | **-6.8484%** | -0.1702% | 9/0/1 | 124/0/26 | 6.235e-02 | 9.766e-03 |
| `T=5` | UltraEMeans | e-means | `test_mse` | 10 | **-5.6460%** | -0.1243% | 9/0/1 | 118/0/32 | 5.887e-02 | 5.859e-03 |
| `all_T` | UltraEMeans | e-means | `total_time_sec` | 30 | +68.7411% | +49.8448% | 3/0/27 | 45/0/405 | 1.529e-06 | 4.657e-08 |
| `T=1` | UltraEMeans | e-means | `total_time_sec` | 10 | +62.7236% | +44.0541% | 1/0/9 | 15/0/135 | 8.865e-03 | 3.906e-03 |
| `T=2` | UltraEMeans | e-means | `total_time_sec` | 10 | +58.6539% | +39.0884% | 1/0/9 | 15/0/135 | 1.674e-02 | 5.859e-03 |
| `T=5` | UltraEMeans | e-means | `total_time_sec` | 10 | +84.8458% | +56.2219% | 1/0/9 | 15/0/135 | 3.369e-03 | 3.906e-03 |
| `all_T` | UltraEMeans | e-means++ | `test_mse` | 30 | **-2.8135%** | -0.1232% | 27/0/3 | 362/0/88 | 3.276e-04 | 8.326e-07 |
| `T=1` | UltraEMeans | e-means++ | `test_mse` | 10 | **-2.7051%** | -0.0868% | 8/0/2 | 110/0/40 | 5.847e-02 | 1.953e-02 |
| `T=2` | UltraEMeans | e-means++ | `test_mse` | 10 | **-2.8184%** | -0.1268% | 10/0/0 | 126/0/24 | 4.740e-02 | 1.953e-03 |
| `T=5` | UltraEMeans | e-means++ | `test_mse` | 10 | **-2.9170%** | -0.1364% | 9/0/1 | 126/0/24 | 4.333e-02 | 9.766e-03 |
| `all_T` | UltraEMeans | e-means++ | `total_time_sec` | 30 | **-32.2248%** | -35.9785% | 28/0/2 | 410/0/40 | 4.050e-09 | 1.304e-07 |
| `T=1` | UltraEMeans | e-means++ | `total_time_sec` | 10 | **-39.2221%** | -44.4047% | 10/0/0 | 150/0/0 | 1.292e-04 | 1.953e-03 |
| `T=2` | UltraEMeans | e-means++ | `total_time_sec` | 10 | **-37.0655%** | -42.5281% | 10/0/0 | 140/0/10 | 2.285e-04 | 1.953e-03 |
| `T=5` | UltraEMeans | e-means++ | `total_time_sec` | 10 | **-20.3870%** | -28.3941% | 8/0/2 | 120/0/30 | 1.466e-02 | 1.367e-02 |
| `all_T` | UltraEMeans | s-means | `test_mse` | 30 | **-4.3740%** | -2.7262% | 27/0/3 | 411/0/39 | 2.846e-05 | 1.024e-07 |
| `T=1` | UltraEMeans | s-means | `test_mse` | 10 | **-5.5443%** | -4.3690% | 9/0/1 | 139/0/11 | 1.180e-02 | 3.906e-03 |
| `T=2` | UltraEMeans | s-means | `test_mse` | 10 | **-4.3591%** | -2.9635% | 9/0/1 | 134/0/16 | 2.288e-02 | 5.859e-03 |
| `T=5` | UltraEMeans | s-means | `test_mse` | 10 | **-3.2185%** | -1.5560% | 9/0/1 | 138/0/12 | 3.081e-02 | 3.906e-03 |
| `all_T` | UltraEMeans | s-means | `total_time_sec` | 30 | +220.7288% | +218.9420% | 0/0/30 | 0/0/450 | 1.010e-17 | 1.863e-09 |
| `T=1` | UltraEMeans | s-means | `total_time_sec` | 10 | +209.1052% | +210.4514% | 0/0/10 | 0/0/150 | 3.687e-07 | 1.953e-03 |
| `T=2` | UltraEMeans | s-means | `total_time_sec` | 10 | +204.8156% | +210.5485% | 0/0/10 | 0/0/150 | 2.571e-06 | 1.953e-03 |
| `T=5` | UltraEMeans | s-means | `total_time_sec` | 10 | +248.2655% | +237.6613% | 0/0/10 | 0/0/150 | 2.709e-06 | 1.953e-03 |
| `all_T` | UltraEMeans | SuperEMeans | `test_mse` | 30 | **-2.0934%** | -0.1529% | 17/0/13 | 290/0/160 | 3.050e-04 | 7.111e-03 |
| `T=1` | UltraEMeans | SuperEMeans | `test_mse` | 10 | -2.0391% | -0.1529% | 7/0/3 | 103/0/47 | 4.390e-02 | 6.445e-02 |
| `T=2` | UltraEMeans | SuperEMeans | `test_mse` | 10 | -2.2169% | -0.0844% | 5/0/5 | 96/0/54 | 4.710e-02 | 2.754e-01 |
| `T=5` | UltraEMeans | SuperEMeans | `test_mse` | 10 | -2.0242% | -0.0994% | 5/0/5 | 91/0/59 | 5.333e-02 | 2.754e-01 |
| `all_T` | UltraEMeans | SuperEMeans | `total_time_sec` | 30 | **-64.1601%** | -69.8777% | 30/0/0 | 450/0/0 | 1.047e-21 | 1.863e-09 |
| `T=1` | UltraEMeans | SuperEMeans | `total_time_sec` | 10 | **-64.2459%** | -70.5781% | 10/0/0 | 150/0/0 | 1.689e-07 | 1.953e-03 |
| `T=2` | UltraEMeans | SuperEMeans | `total_time_sec` | 10 | **-66.0738%** | -72.1494% | 10/0/0 | 150/0/0 | 6.169e-08 | 1.953e-03 |
| `T=5` | UltraEMeans | SuperEMeans | `total_time_sec` | 10 | **-62.1605%** | -67.9463% | 10/0/0 | 150/0/0 | 2.373e-07 | 1.953e-03 |
| `all_T` | UltraEMeans | Hybrid SuperEMeans | `test_mse` | 30 | **-1.8924%** | -0.5144% | 28/0/2 | 398/0/52 | 7.961e-05 | 4.657e-08 |
| `T=1` | UltraEMeans | Hybrid SuperEMeans | `test_mse` | 10 | **-1.9667%** | -0.4341% | 9/0/1 | 132/0/18 | 2.741e-02 | 5.859e-03 |
| `T=2` | UltraEMeans | Hybrid SuperEMeans | `test_mse` | 10 | **-2.0304%** | -0.8005% | 10/0/0 | 137/0/13 | 2.213e-02 | 1.953e-03 |
| `T=5` | UltraEMeans | Hybrid SuperEMeans | `test_mse` | 10 | **-1.6802%** | -0.5831% | 9/0/1 | 129/0/21 | 4.737e-02 | 3.906e-03 |
| `all_T` | UltraEMeans | Hybrid SuperEMeans | `total_time_sec` | 30 | **-63.3293%** | -68.7763% | 30/0/0 | 450/0/0 | 2.069e-21 | 1.863e-09 |
| `T=1` | UltraEMeans | Hybrid SuperEMeans | `total_time_sec` | 10 | **-63.6624%** | -70.2525% | 10/0/0 | 150/0/0 | 1.624e-07 | 1.953e-03 |
| `T=2` | UltraEMeans | Hybrid SuperEMeans | `total_time_sec` | 10 | **-64.9469%** | -70.9680% | 10/0/0 | 150/0/0 | 9.194e-08 | 1.953e-03 |
| `T=5` | UltraEMeans | Hybrid SuperEMeans | `total_time_sec` | 10 | **-61.3785%** | -66.7689% | 10/0/0 | 150/0/0 | 3.268e-07 | 1.953e-03 |
| `all_T` | UltraEMeans | Lloyd k-means (25 iter) | `test_mse` | 30 | **-8.6161%** | -0.5004% | 24/0/6 | 357/0/93 | 1.067e-03 | 2.833e-04 |
| `T=1` | UltraEMeans | Lloyd k-means (25 iter) | `test_mse` | 10 | -7.5594% | +0.0277% | 5/0/5 | 77/0/73 | 9.669e-02 | 4.922e-01 |
| `T=2` | UltraEMeans | Lloyd k-means (25 iter) | `test_mse` | 10 | **-8.6739%** | -0.4096% | 9/0/1 | 130/0/20 | 6.959e-02 | 5.859e-03 |
| `T=5` | UltraEMeans | Lloyd k-means (25 iter) | `test_mse` | 10 | **-9.6150%** | -0.7576% | 10/0/0 | 150/0/0 | 5.852e-02 | 1.953e-03 |
| `all_T` | UltraEMeans | Lloyd k-means (25 iter) | `total_time_sec` | 30 | +130.9835% | +116.5538% | 2/0/28 | 28/0/422 | 6.282e-09 | 9.313e-09 |
| `T=1` | UltraEMeans | Lloyd k-means (25 iter) | `total_time_sec` | 10 | +93.8046% | +73.2316% | 1/0/9 | 15/0/135 | 2.550e-03 | 3.906e-03 |
| `T=2` | UltraEMeans | Lloyd k-means (25 iter) | `total_time_sec` | 10 | +105.2834% | +86.6910% | 1/0/9 | 13/0/137 | 8.523e-04 | 3.906e-03 |
| `T=5` | UltraEMeans | Lloyd k-means (25 iter) | `total_time_sec` | 10 | +193.8626% | +176.3305% | 0/0/10 | 0/0/150 | 1.098e-04 | 1.953e-03 |
| `all_T` | UltraEMeans (fast) | e-means | `test_mse` | 30 | **-6.8748%** | -0.3316% | 23/0/7 | 365/0/85 | 6.864e-04 | 5.145e-06 |
| `T=1` | UltraEMeans (fast) | e-means | `test_mse` | 10 | **-7.9853%** | -0.4590% | 8/0/2 | 131/0/19 | 5.794e-02 | 9.766e-03 |
| `T=2` | UltraEMeans (fast) | e-means | `test_mse` | 10 | **-6.8979%** | -0.1921% | 8/0/2 | 118/0/32 | 6.320e-02 | 1.367e-02 |
| `T=5` | UltraEMeans (fast) | e-means | `test_mse` | 10 | **-5.7411%** | -0.1391% | 7/0/3 | 116/0/34 | 6.159e-02 | 2.734e-02 |
| `all_T` | UltraEMeans (fast) | e-means | `total_time_sec` | 30 | +42.2056% | +33.5736% | 3/0/27 | 46/0/404 | 6.606e-05 | 1.699e-04 |
| `T=1` | UltraEMeans (fast) | e-means | `total_time_sec` | 10 | +36.7962% | +23.3939% | 1/0/9 | 15/0/135 | 3.531e-02 | 3.711e-02 |
| `T=2` | UltraEMeans (fast) | e-means | `total_time_sec` | 10 | +29.3279% | +20.0526% | 1/0/9 | 16/0/134 | 5.967e-02 | 4.883e-02 |
| `T=5` | UltraEMeans (fast) | e-means | `total_time_sec` | 10 | +60.4927% | +40.2555% | 1/0/9 | 15/0/135 | 8.740e-03 | 2.734e-02 |
| `all_T` | UltraEMeans (fast) | s-means | `test_mse` | 30 | **-4.4172%** | -2.6889% | 27/0/3 | 412/0/38 | 3.219e-05 | 3.148e-07 |
| `T=1` | UltraEMeans (fast) | s-means | `test_mse` | 10 | **-5.5054%** | -4.3367% | 9/0/1 | 137/0/13 | 1.221e-02 | 5.859e-03 |
| `T=2` | UltraEMeans (fast) | s-means | `test_mse` | 10 | **-4.4188%** | -2.9665% | 9/0/1 | 136/0/14 | 2.352e-02 | 5.859e-03 |
| `T=5` | UltraEMeans (fast) | s-means | `test_mse` | 10 | **-3.3272%** | -1.5481% | 9/0/1 | 139/0/11 | 3.531e-02 | 5.859e-03 |
| `all_T` | UltraEMeans (fast) | s-means | `total_time_sec` | 30 | +172.3471% | +171.4679% | 0/0/30 | 0/0/450 | 7.281e-15 | 1.863e-09 |
| `T=1` | UltraEMeans (fast) | s-means | `total_time_sec` | 10 | +161.0344% | +169.7854% | 0/0/10 | 0/0/150 | 4.286e-06 | 1.953e-03 |
| `T=2` | UltraEMeans (fast) | s-means | `total_time_sec` | 10 | +151.1677% | +165.0052% | 0/0/10 | 0/0/150 | 5.015e-06 | 1.953e-03 |
| `T=5` | UltraEMeans (fast) | s-means | `total_time_sec` | 10 | +204.8393% | +197.7151% | 0/0/10 | 0/0/150 | 2.208e-05 | 1.953e-03 |
| `all_T` | UltraEMeans (fast) | SuperEMeans | `test_mse` | 30 | **-2.1413%** | -0.1137% | 18/0/12 | 283/0/167 | 3.102e-04 | 8.705e-03 |
| `T=1` | UltraEMeans (fast) | SuperEMeans | `test_mse` | 10 | -1.9994% | -0.1137% | 7/0/3 | 100/0/50 | 4.735e-02 | 1.309e-01 |
| `T=2` | UltraEMeans (fast) | SuperEMeans | `test_mse` | 10 | -2.2809% | -0.0696% | 5/0/5 | 92/0/58 | 4.758e-02 | 2.324e-01 |
| `T=5` | UltraEMeans (fast) | SuperEMeans | `test_mse` | 10 | -2.1436% | -0.1014% | 6/0/4 | 91/0/59 | 4.940e-02 | 2.324e-01 |
| `all_T` | UltraEMeans (fast) | SuperEMeans | `total_time_sec` | 30 | **-70.2006%** | -73.0128% | 30/0/0 | 450/0/0 | 3.773e-27 | 1.863e-09 |
| `T=1` | UltraEMeans (fast) | SuperEMeans | `total_time_sec` | 10 | **-70.5307%** | -73.7650% | 10/0/0 | 150/0/0 | 2.686e-09 | 1.953e-03 |
| `T=2` | UltraEMeans (fast) | SuperEMeans | `total_time_sec` | 10 | **-72.5562%** | -75.3941% | 10/0/0 | 150/0/0 | 4.931e-10 | 1.953e-03 |
| `T=5` | UltraEMeans (fast) | SuperEMeans | `total_time_sec` | 10 | **-67.5149%** | -70.1127% | 10/0/0 | 150/0/0 | 7.809e-09 | 1.953e-03 |
| `all_T` | UltraEMeans (fast) | SuperEMeans (fast) | `test_mse` | 30 | **-2.3314%** | -0.1197% | 18/0/12 | 275/0/175 | 3.332e-04 | 5.013e-03 |
| `T=1` | UltraEMeans (fast) | SuperEMeans (fast) | `test_mse` | 10 | -2.3424% | -0.1272% | 7/0/3 | 102/0/48 | 5.056e-02 | 6.445e-02 |
| `T=2` | UltraEMeans (fast) | SuperEMeans (fast) | `test_mse` | 10 | -2.4052% | -0.0680% | 5/0/5 | 89/0/61 | 4.654e-02 | 2.324e-01 |
| `T=5` | UltraEMeans (fast) | SuperEMeans (fast) | `test_mse` | 10 | -2.2466% | -0.0571% | 6/0/4 | 84/0/66 | 5.206e-02 | 2.754e-01 |
| `all_T` | UltraEMeans (fast) | SuperEMeans (fast) | `total_time_sec` | 30 | **-61.1292%** | -64.4792% | 30/0/0 | 450/0/0 | 1.108e-22 | 1.863e-09 |
| `T=1` | UltraEMeans (fast) | SuperEMeans (fast) | `total_time_sec` | 10 | **-61.9651%** | -66.3654% | 10/0/0 | 150/0/0 | 2.962e-08 | 1.953e-03 |
| `T=2` | UltraEMeans (fast) | SuperEMeans (fast) | `total_time_sec` | 10 | **-64.9771%** | -69.0222% | 10/0/0 | 150/0/0 | 9.552e-09 | 1.953e-03 |
| `T=5` | UltraEMeans (fast) | SuperEMeans (fast) | `total_time_sec` | 10 | **-56.4455%** | -59.1084% | 10/0/0 | 150/0/0 | 2.919e-07 | 1.953e-03 |
| `all_T` | UltraEMeans (fast) | Hybrid SuperEMeans | `test_mse` | 30 | **-1.9406%** | -0.5331% | 27/0/3 | 400/0/50 | 8.166e-05 | 1.304e-07 |
| `T=1` | UltraEMeans (fast) | Hybrid SuperEMeans | `test_mse` | 10 | **-1.9269%** | -0.4136% | 8/0/2 | 130/0/20 | 3.012e-02 | 1.367e-02 |
| `T=2` | UltraEMeans (fast) | Hybrid SuperEMeans | `test_mse` | 10 | **-2.0948%** | -0.8174% | 10/0/0 | 138/0/12 | 2.250e-02 | 1.953e-03 |
| `T=5` | UltraEMeans (fast) | Hybrid SuperEMeans | `test_mse` | 10 | **-1.8000%** | -0.5724% | 9/0/1 | 132/0/18 | 4.369e-02 | 3.906e-03 |
| `all_T` | UltraEMeans (fast) | Hybrid SuperEMeans | `total_time_sec` | 30 | **-69.5153%** | -72.6348% | 30/0/0 | 450/0/0 | 6.114e-27 | 1.863e-09 |
| `T=1` | UltraEMeans (fast) | Hybrid SuperEMeans | `total_time_sec` | 10 | **-70.0472%** | -73.1471% | 10/0/0 | 150/0/0 | 2.377e-09 | 1.953e-03 |
| `T=2` | UltraEMeans (fast) | Hybrid SuperEMeans | `total_time_sec` | 10 | **-71.6585%** | -74.6170% | 10/0/0 | 150/0/0 | 6.769e-10 | 1.953e-03 |
| `T=5` | UltraEMeans (fast) | Hybrid SuperEMeans | `total_time_sec` | 10 | **-66.8404%** | -69.5238% | 10/0/0 | 150/0/0 | 1.064e-08 | 1.953e-03 |
| `all_T` | UltraEMeans (fast) | Hybrid SuperEMeans (fast) | `test_mse` | 30 | **-2.1261%** | -0.4713% | 28/0/2 | 402/0/48 | 1.674e-04 | 4.657e-08 |
| `T=1` | UltraEMeans (fast) | Hybrid SuperEMeans (fast) | `test_mse` | 10 | **-2.2347%** | -0.3718% | 8/0/2 | 129/0/21 | 3.925e-02 | 1.367e-02 |
| `T=2` | UltraEMeans (fast) | Hybrid SuperEMeans (fast) | `test_mse` | 10 | **-2.1752%** | -0.5008% | 10/0/0 | 138/0/12 | 3.054e-02 | 1.953e-03 |
| `T=5` | UltraEMeans (fast) | Hybrid SuperEMeans (fast) | `test_mse` | 10 | **-1.9683%** | -0.4897% | 10/0/0 | 135/0/15 | 5.016e-02 | 1.953e-03 |
| `all_T` | UltraEMeans (fast) | Hybrid SuperEMeans (fast) | `total_time_sec` | 30 | **-60.5359%** | -64.1040% | 30/0/0 | 450/0/0 | 1.445e-22 | 1.863e-09 |
| `T=1` | UltraEMeans (fast) | Hybrid SuperEMeans (fast) | `total_time_sec` | 10 | **-61.2301%** | -65.3153% | 10/0/0 | 150/0/0 | 2.951e-08 | 1.953e-03 |
| `T=2` | UltraEMeans (fast) | Hybrid SuperEMeans (fast) | `total_time_sec` | 10 | **-64.3234%** | -68.3130% | 10/0/0 | 150/0/0 | 9.295e-09 | 1.953e-03 |
| `T=5` | UltraEMeans (fast) | Hybrid SuperEMeans (fast) | `total_time_sec` | 10 | **-56.0541%** | -60.2935% | 10/0/0 | 150/0/0 | 3.785e-07 | 1.953e-03 |
| `all_T` | UltraEMeans (fast) | UltraEMeans | `test_mse` | 30 | -0.0488% | +0.0155% | 9/0/21 | 205/0/245 | 4.143e-01 | 3.184e-01 |
| `T=1` | UltraEMeans (fast) | UltraEMeans | `test_mse` | 10 | +0.0426% | +0.0362% | 1/0/9 | 54/0/96 | 6.717e-01 | 8.398e-02 |
| `T=2` | UltraEMeans (fast) | UltraEMeans | `test_mse` | 10 | -0.0667% | +0.0014% | 5/0/5 | 83/0/67 | 3.294e-01 | 6.250e-01 |
| `T=5` | UltraEMeans (fast) | UltraEMeans | `test_mse` | 10 | -0.1223% | +0.0101% | 3/0/7 | 68/0/82 | 3.905e-01 | 6.953e-01 |
| `all_T` | UltraEMeans (fast) | UltraEMeans | `total_time_sec` | 30 | **-15.3280%** | -14.7685% | 29/0/1 | 426/0/24 | 1.722e-10 | 1.304e-08 |
| `T=1` | UltraEMeans (fast) | UltraEMeans | `total_time_sec` | 10 | **-15.5853%** | -15.1400% | 9/0/1 | 141/0/9 | 1.024e-03 | 3.906e-03 |
| `T=2` | UltraEMeans (fast) | UltraEMeans | `total_time_sec` | 10 | **-17.2602%** | -16.1603% | 10/0/0 | 147/0/3 | 5.860e-05 | 1.953e-03 |
| `T=5` | UltraEMeans (fast) | UltraEMeans | `total_time_sec` | 10 | **-13.1386%** | -10.3549% | 10/0/0 | 138/0/12 | 8.191e-04 | 1.953e-03 |
| `all_T` | s-means | e-means | `test_mse` | 30 | -2.8146% | -0.0915% | 18/0/12 | 241/0/209 | 2.071e-02 | 1.241e-01 |
| `T=1` | s-means | e-means | `test_mse` | 10 | -2.9407% | -0.2274% | 6/0/4 | 84/0/66 | 2.614e-01 | 4.922e-01 |
| `T=2` | s-means | e-means | `test_mse` | 10 | -2.8252% | -0.1142% | 6/0/4 | 80/0/70 | 2.121e-01 | 4.922e-01 |
| `T=5` | s-means | e-means | `test_mse` | 10 | -2.6777% | -0.0725% | 6/0/4 | 77/0/73 | 1.139e-01 | 3.750e-01 |
| `all_T` | s-means | e-means | `total_time_sec` | 30 | **-48.6106%** | -52.9757% | 30/0/0 | 450/0/0 | 3.022e-20 | 1.863e-09 |
| `T=1` | s-means | e-means | `total_time_sec` | 10 | **-48.5687%** | -53.6562% | 10/0/0 | 150/0/0 | 6.393e-07 | 1.953e-03 |
| `T=2` | s-means | e-means | `total_time_sec` | 10 | **-49.2026%** | -52.9035% | 10/0/0 | 150/0/0 | 3.791e-07 | 1.953e-03 |
| `T=5` | s-means | e-means | `total_time_sec` | 10 | **-48.0603%** | -51.7083% | 10/0/0 | 150/0/0 | 2.769e-07 | 1.953e-03 |
| `all_T` | SuperEMeans | e-means | `test_mse` | 30 | **-5.0156%** | -0.3535% | 27/0/3 | 393/0/57 | 1.642e-03 | 1.684e-06 |
| `T=1` | SuperEMeans | e-means | `test_mse` | 10 | **-6.3207%** | -0.5810% | 9/0/1 | 136/0/14 | 6.916e-02 | 5.859e-03 |
| `T=2` | SuperEMeans | e-means | `test_mse` | 10 | **-4.9252%** | -0.2766% | 9/0/1 | 128/0/22 | 8.090e-02 | 9.766e-03 |
| `T=5` | SuperEMeans | e-means | `test_mse` | 10 | **-3.8009%** | -0.2665% | 9/0/1 | 129/0/21 | 9.513e-02 | 1.367e-02 |
| `all_T` | SuperEMeans | e-means | `total_time_sec` | 30 | +374.9537% | +379.6686% | 0/0/30 | 0/0/450 | 8.725e-20 | 1.863e-09 |
| `T=1` | SuperEMeans | e-means | `total_time_sec` | 10 | +365.0687% | +377.1949% | 0/0/10 | 0/0/150 | 2.529e-06 | 1.953e-03 |
| `T=2` | SuperEMeans | e-means | `total_time_sec` | 10 | +369.1996% | +387.4808% | 0/0/10 | 0/0/150 | 2.500e-07 | 1.953e-03 |
| `T=5` | SuperEMeans | e-means | `total_time_sec` | 10 | +390.5927% | +378.7508% | 0/0/10 | 0/0/150 | 1.684e-07 | 1.953e-03 |
| `all_T` | SuperEMeans | s-means | `test_mse` | 30 | **-2.3640%** | -1.3521% | 25/0/5 | 371/0/79 | 1.726e-04 | 9.220e-06 |
| `T=1` | SuperEMeans | s-means | `test_mse` | 10 | **-3.6347%** | -3.4214% | 8/0/2 | 129/0/21 | 9.992e-03 | 9.766e-03 |
| `T=2` | SuperEMeans | s-means | `test_mse` | 10 | **-2.2319%** | -1.0946% | 9/0/1 | 127/0/23 | 2.803e-02 | 5.859e-03 |
| `T=5` | SuperEMeans | s-means | `test_mse` | 10 | -1.2253% | -0.3272% | 8/0/2 | 115/0/35 | 1.503e-01 | 8.398e-02 |
| `all_T` | SuperEMeans | s-means | `total_time_sec` | 30 | +853.6817% | +943.7631% | 0/0/30 | 0/0/450 | 3.813e-19 | 1.863e-09 |
| `T=1` | SuperEMeans | s-means | `total_time_sec` | 10 | +830.8735% | +943.7631% | 0/0/10 | 0/0/150 | 1.256e-06 | 1.953e-03 |
| `T=2` | SuperEMeans | s-means | `total_time_sec` | 10 | +853.5411% | +893.9623% | 0/0/10 | 0/0/150 | 1.023e-06 | 1.953e-03 |
| `T=5` | SuperEMeans | s-means | `total_time_sec` | 10 | +876.6303% | +916.5804% | 0/0/10 | 0/0/150 | 5.413e-07 | 1.953e-03 |
| `all_T` | Hybrid SuperEMeans | e-means | `test_mse` | 30 | **-5.1844%** | -0.1206% | 20/0/10 | 279/0/171 | 1.949e-03 | 1.546e-02 |
| `T=1` | Hybrid SuperEMeans | e-means | `test_mse` | 10 | **-6.3643%** | -0.4344% | 9/0/1 | 115/0/35 | 7.556e-02 | 2.734e-02 |
| `T=2` | Hybrid SuperEMeans | e-means | `test_mse` | 10 | -5.0662% | -0.0462% | 5/0/5 | 77/0/73 | 9.493e-02 | 3.750e-01 |
| `T=5` | Hybrid SuperEMeans | e-means | `test_mse` | 10 | -4.1229% | -0.0350% | 6/0/4 | 87/0/63 | 8.955e-02 | 3.223e-01 |
| `all_T` | Hybrid SuperEMeans | e-means | `total_time_sec` | 30 | +364.1211% | +360.4563% | 0/0/30 | 0/0/450 | 1.905e-19 | 1.863e-09 |
| `T=1` | Hybrid SuperEMeans | e-means | `total_time_sec` | 10 | +357.0683% | +359.1947% | 0/0/10 | 0/0/150 | 3.161e-06 | 1.953e-03 |
| `T=2` | Hybrid SuperEMeans | e-means | `total_time_sec` | 10 | +354.6295% | +360.7172% | 0/0/10 | 0/0/150 | 3.476e-07 | 1.953e-03 |
| `T=5` | Hybrid SuperEMeans | e-means | `total_time_sec` | 10 | +380.6654% | +360.4563% | 0/0/10 | 0/0/150 | 1.903e-07 | 1.953e-03 |
| `all_T` | Hybrid SuperEMeans | s-means | `test_mse` | 30 | **-2.5698%** | -1.3680% | 26/0/4 | 371/0/79 | 1.618e-04 | 1.061e-05 |
| `T=1` | Hybrid SuperEMeans | s-means | `test_mse` | 10 | **-3.7061%** | -3.2620% | 10/0/0 | 142/0/8 | 1.340e-02 | 1.953e-03 |
| `T=2` | Hybrid SuperEMeans | s-means | `test_mse` | 10 | **-2.4235%** | -1.9032% | 8/0/2 | 117/0/33 | 3.965e-02 | 2.734e-02 |
| `T=5` | Hybrid SuperEMeans | s-means | `test_mse` | 10 | **-1.5797%** | -1.2565% | 8/0/2 | 112/0/38 | 8.624e-02 | 3.711e-02 |
| `all_T` | Hybrid SuperEMeans | s-means | `total_time_sec` | 30 | +830.6962% | +922.4379% | 0/0/30 | 0/0/450 | 3.856e-19 | 1.863e-09 |
| `T=1` | Hybrid SuperEMeans | s-means | `total_time_sec` | 10 | +813.0785% | +923.9604% | 0/0/10 | 0/0/150 | 1.296e-06 | 1.953e-03 |
| `T=2` | Hybrid SuperEMeans | s-means | `total_time_sec` | 10 | +822.7500% | +885.5116% | 0/0/10 | 0/0/150 | 1.065e-06 | 1.953e-03 |
| `T=5` | Hybrid SuperEMeans | s-means | `total_time_sec` | 10 | +856.2601% | +888.5170% | 0/0/10 | 0/0/150 | 5.055e-07 | 1.953e-03 |
| `all_T` | Hybrid SuperEMeans | SuperEMeans | `test_mse` | 30 | -0.2146% | +0.0127% | 15/0/15 | 214/0/236 | 1.570e-01 | 3.184e-01 |
| `T=1` | Hybrid SuperEMeans | SuperEMeans | `test_mse` | 10 | -0.0823% | +0.0320% | 5/0/5 | 75/0/75 | 6.255e-01 | 6.250e-01 |
| `T=2` | Hybrid SuperEMeans | SuperEMeans | `test_mse` | 10 | -0.2035% | -0.0409% | 5/0/5 | 72/0/78 | 4.880e-01 | 6.953e-01 |
| `T=5` | Hybrid SuperEMeans | SuperEMeans | `test_mse` | 10 | -0.3579% | +0.0127% | 5/0/5 | 67/0/83 | 2.889e-01 | 5.566e-01 |
| `all_T` | Hybrid SuperEMeans | SuperEMeans | `total_time_sec` | 30 | **-2.3432%** | -2.4889% | 27/0/3 | 372/0/78 | 3.117e-06 | 6.918e-06 |
| `T=1` | Hybrid SuperEMeans | SuperEMeans | `total_time_sec` | 10 | **-1.8395%** | -1.2656% | 9/0/1 | 121/0/29 | 9.480e-03 | 3.906e-03 |
| `T=2` | Hybrid SuperEMeans | SuperEMeans | `total_time_sec` | 10 | **-3.2244%** | -3.1170% | 9/0/1 | 130/0/20 | 2.268e-03 | 3.906e-03 |
| `T=5` | Hybrid SuperEMeans | SuperEMeans | `total_time_sec` | 10 | -1.9657% | -2.5390% | 9/0/1 | 121/0/29 | 2.873e-02 | 6.445e-02 |

## 2. Suite 2 Discovery Attribution (Counterfactual / Anchor vs. Full `UltraEMeans`)

| Counterfactual / Anchor | Baseline | Metric | Cells | Mean Delta vs Ultra (%) | Median Delta (%) | Cell W/T/L | `p_ttest` | `p_wilcox` |
| :--- | :--- | :--- | ---: | ---: | ---: | :---: | ---: | ---: |
| UltraEMeans (fast) | UltraEMeans | `test_mse` | 14 | -0.1393% | -0.0012% | 7/0/7 | 2.013e-01 | 5.016e-01 |
| UltraEMeans (fast) | UltraEMeans | `total_time_sec` | 14 | -15.0092% | -14.7403% | 14/0/0 | 1.300e-06 | 1.221e-04 |
| Ultra - D0 (s-means Raw D^3 Init) | UltraEMeans | `test_mse` | 14 | +0.7511% | +0.0523% | 5/0/9 | 1.390e-01 | 1.726e-01 |
| Ultra - D0 (s-means Raw D^3 Init) | UltraEMeans | `total_time_sec` | 14 | -9.3999% | -10.0312% | 13/0/1 | 8.082e-06 | 2.441e-04 |
| Ultra - D1a (s-means Rademacher 0.55) | UltraEMeans | `test_mse` | 14 | **+3.3196%** | +3.1309% | 2/0/12 | 5.440e-04 | 6.104e-04 |
| Ultra - D1a (s-means Rademacher 0.55) | UltraEMeans | `total_time_sec` | 14 | +0.2573% | -1.4635% | 8/0/6 | 8.826e-01 | 8.552e-01 |
| Ultra - D1b (Super Oja 0.08) | UltraEMeans | `test_mse` | 14 | **+0.1209%** | +0.0419% | 3/0/11 | 7.142e-02 | 1.074e-02 |
| Ultra - D1b (Super Oja 0.08) | UltraEMeans | `total_time_sec` | 14 | -6.3162% | -5.4474% | 14/0/0 | 3.856e-06 | 1.221e-04 |
| Ultra - D2a (Decoupled SSE, No Huygens) | UltraEMeans | `test_mse` | 14 | **+3.6155%** | +1.1484% | 1/0/13 | 1.642e-02 | 1.709e-03 |
| Ultra - D2a (Decoupled SSE, No Huygens) | UltraEMeans | `total_time_sec` | 14 | -13.4715% | -11.8110% | 13/0/1 | 1.012e-05 | 2.441e-04 |
| Ultra - D2b (No 85% Freeze/Flush) | UltraEMeans | `test_mse` | 14 | +0.1440% | +0.0008% | 5/0/9 | 8.420e-02 | 1.531e-01 |
| Ultra - D2b (No 85% Freeze/Flush) | UltraEMeans | `total_time_sec` | 14 | +4.8609% | +4.9610% | 0/0/14 | 7.371e-07 | 1.221e-04 |
| Ultra - D3a (Raw Closing Mean) | UltraEMeans | `test_mse` | 14 | **+0.0131%** | +0.0069% | 4/0/10 | 8.025e-02 | 4.944e-02 |
| Ultra - D3a (Raw Closing Mean) | UltraEMeans | `total_time_sec` | 14 | -2.6857% | -2.8349% | 14/0/0 | 5.576e-06 | 1.221e-04 |
| Ultra - D3b (s-means Fixed EB 0.20) | UltraEMeans | `test_mse` | 14 | **+0.0279%** | +0.0177% | 5/0/9 | 3.252e-02 | 4.944e-02 |
| Ultra - D3b (s-means Fixed EB 0.20) | UltraEMeans | `total_time_sec` | 14 | -2.3437% | -2.8014% | 13/0/1 | 1.714e-05 | 2.441e-04 |
| Ultra - D3c (Super Slow-EMA Only) | UltraEMeans | `test_mse` | 14 | -0.0328% | -0.0010% | 7/0/7 | 2.157e-01 | 4.263e-01 |
| Ultra - D3c (Super Slow-EMA Only) | UltraEMeans | `total_time_sec` | 14 | -0.4320% | -0.5004% | 11/0/3 | 2.431e-01 | 7.849e-02 |
| Ultra - D4 (No P/Q Transplants) | UltraEMeans | `test_mse` | 14 | **+0.0397%** | +0.0100% | 4/0/10 | 5.164e-02 | 3.528e-02 |
| Ultra - D4 (No P/Q Transplants) | UltraEMeans | `total_time_sec` | 14 | -28.4113% | -30.4178% | 14/0/0 | 8.806e-07 | 1.221e-04 |
| s-means (anchor) | UltraEMeans | `test_mse` | 14 | **+4.8723%** | +2.8153% | 1/0/13 | 5.873e-03 | 6.104e-04 |
| s-means (anchor) | UltraEMeans | `total_time_sec` | 14 | -66.4475% | -68.2183% | 14/0/0 | 7.945e-14 | 1.221e-04 |
| SuperEMeans (anchor) | UltraEMeans | `test_mse` | 14 | +2.4512% | +1.6930% | 6/0/8 | 7.051e-03 | 6.763e-02 |
| SuperEMeans (anchor) | UltraEMeans | `total_time_sec` | 14 | +219.9726% | +240.6134% | 0/0/14 | 1.147e-08 | 1.221e-04 |
| Hybrid SuperEMeans (anchor) | UltraEMeans | `test_mse` | 14 | **+2.0647%** | +1.0586% | 0/0/14 | 3.970e-03 | 1.221e-04 |
| Hybrid SuperEMeans (anchor) | UltraEMeans | `total_time_sec` | 14 | +209.0163% | +226.0369% | 0/0/14 | 1.224e-08 | 1.221e-04 |

## 3. Isolated Rare-Mode Survival (`multiscale_rare_outliers`)

| Cell | Solver | Mean Rare Modes Preserved (%) | Std (%) |
| :--- | :--- | ---: | ---: |
| `s1_multiscale_rare_t1` | e-means++ | 26.54% | 4.56% |
| `s1_multiscale_rare_t1` | Hybrid SuperEMeans | 29.02% | 4.70% |
| `s1_multiscale_rare_t1` | Hybrid SuperEMeans (fast) | 30.33% | 4.25% |
| `s1_multiscale_rare_t1` | Lloyd k-means (25 iter) | 37.12% | 8.75% |
| `s1_multiscale_rare_t1` | s-means | 0.00% | 0.00% |
| `s1_multiscale_rare_t1` | SuperEMeans | 27.58% | 4.60% |
| `s1_multiscale_rare_t1` | SuperEMeans (fast) | 29.28% | 4.02% |
| `s1_multiscale_rare_t1` | UltraEMeans | 47.19% | 6.65% |
| `s1_multiscale_rare_t1` | UltraEMeans (fast) | 49.67% | 6.32% |
| `s1_multiscale_rare_t1` | e-means | 0.00% | 0.00% |
| `s1_multiscale_rare_t2` | e-means++ | 39.87% | 4.61% |
| `s1_multiscale_rare_t2` | Hybrid SuperEMeans | 46.80% | 3.30% |
| `s1_multiscale_rare_t2` | Hybrid SuperEMeans (fast) | 46.14% | 4.44% |
| `s1_multiscale_rare_t2` | Lloyd k-means (25 iter) | 37.12% | 8.75% |
| `s1_multiscale_rare_t2` | s-means | 0.00% | 0.00% |
| `s1_multiscale_rare_t2` | SuperEMeans | 44.05% | 3.62% |
| `s1_multiscale_rare_t2` | SuperEMeans (fast) | 44.44% | 5.38% |
| `s1_multiscale_rare_t2` | UltraEMeans | 60.52% | 6.83% |
| `s1_multiscale_rare_t2` | UltraEMeans (fast) | 64.31% | 4.70% |
| `s1_multiscale_rare_t2` | e-means | 0.00% | 0.00% |
| `s1_multiscale_rare_t5` | e-means++ | 49.28% | 5.49% |
| `s1_multiscale_rare_t5` | Hybrid SuperEMeans | 63.53% | 6.28% |
| `s1_multiscale_rare_t5` | Hybrid SuperEMeans (fast) | 60.00% | 6.06% |
| `s1_multiscale_rare_t5` | Lloyd k-means (25 iter) | 37.12% | 8.75% |
| `s1_multiscale_rare_t5` | s-means | 4.18% | 3.47% |
| `s1_multiscale_rare_t5` | SuperEMeans | 60.52% | 5.78% |
| `s1_multiscale_rare_t5` | SuperEMeans (fast) | 60.00% | 4.73% |
| `s1_multiscale_rare_t5` | UltraEMeans | 77.91% | 2.82% |
| `s1_multiscale_rare_t5` | UltraEMeans (fast) | 82.88% | 2.40% |
| `s1_multiscale_rare_t5` | e-means | 1.05% | 1.46% |

## 4. SKU-Stratified End-to-End GPU Latency Table (Suite 1)

| Blackwell TDP SKU | Solver | Runs | Mean Init (ms) | Mean Train (ms) | Mean Total (ms) | Median Total (ms) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `Blackwell_MaxQ_300W` | Lloyd k-means (25 iter) | 315 | 0.10 | 12.27 | 12.37 | 11.20 |
| `Blackwell_MaxQ_300W` | e-means | 315 | 0.10 | 25.32 | 25.42 | 13.66 |
| `Blackwell_MaxQ_300W` | e-means++ | 315 | 19.51 | 28.66 | 48.17 | 39.37 |
| `Blackwell_MaxQ_300W` | s-means | 315 | 0.80 | 10.51 | 11.32 | 7.11 |
| `Blackwell_MaxQ_300W` | SuperEMeans | 315 | 19.79 | 80.99 | 100.78 | 72.45 |
| `Blackwell_MaxQ_300W` | SuperEMeans (fast) | 315 | 19.77 | 54.92 | 74.69 | 56.17 |
| `Blackwell_MaxQ_300W` | Hybrid SuperEMeans | 315 | 19.77 | 77.65 | 97.42 | 70.04 |
| `Blackwell_MaxQ_300W` | Hybrid SuperEMeans (fast) | 315 | 19.77 | 53.06 | 72.83 | 54.85 |
| `Blackwell_MaxQ_300W` | UltraEMeans | 315 | 3.33 | 27.52 | 30.85 | 23.50 |
| `Blackwell_MaxQ_300W` | UltraEMeans (fast) | 315 | 3.28 | 21.88 | 25.16 | 21.17 |
| `Blackwell_Server_600W` | Lloyd k-means (25 iter) | 45 | 0.13 | 29.35 | 29.48 | 29.49 |
| `Blackwell_Server_600W` | e-means | 45 | 0.13 | 21.82 | 21.96 | 19.09 |
| `Blackwell_Server_600W` | e-means++ | 45 | 37.70 | 26.32 | 64.02 | 60.79 |
| `Blackwell_Server_600W` | s-means | 45 | 1.18 | 9.43 | 10.61 | 8.74 |
| `Blackwell_Server_600W` | SuperEMeans | 45 | 38.47 | 71.98 | 110.45 | 101.52 |
| `Blackwell_Server_600W` | SuperEMeans (fast) | 45 | 38.43 | 47.98 | 86.41 | 81.51 |
| `Blackwell_Server_600W` | Hybrid SuperEMeans | 45 | 38.40 | 69.57 | 107.97 | 98.92 |
| `Blackwell_Server_600W` | Hybrid SuperEMeans (fast) | 45 | 38.34 | 46.24 | 84.58 | 80.09 |
| `Blackwell_Server_600W` | UltraEMeans | 45 | 6.34 | 26.35 | 32.68 | 28.09 |
| `Blackwell_Server_600W` | UltraEMeans (fast) | 45 | 6.32 | 22.45 | 28.77 | 24.98 |
| `Blackwell_Workstation_450W` | Lloyd k-means (25 iter) | 90 | 0.09 | 9.63 | 9.72 | 9.78 |
| `Blackwell_Workstation_450W` | e-means | 90 | 0.09 | 13.28 | 13.36 | 13.21 |
| `Blackwell_Workstation_450W` | e-means++ | 90 | 13.90 | 16.53 | 30.43 | 30.60 |
| `Blackwell_Workstation_450W` | s-means | 90 | 0.74 | 6.99 | 7.73 | 7.83 |
| `Blackwell_Workstation_450W` | SuperEMeans | 90 | 14.32 | 43.75 | 58.07 | 59.04 |
| `Blackwell_Workstation_450W` | SuperEMeans (fast) | 90 | 14.33 | 29.36 | 43.69 | 44.01 |
| `Blackwell_Workstation_450W` | Hybrid SuperEMeans | 90 | 14.32 | 43.50 | 57.81 | 57.78 |
| `Blackwell_Workstation_450W` | Hybrid SuperEMeans (fast) | 90 | 14.31 | 29.64 | 43.95 | 43.74 |
| `Blackwell_Workstation_450W` | UltraEMeans | 90 | 3.20 | 24.31 | 27.51 | 26.38 |
| `Blackwell_Workstation_450W` | UltraEMeans (fast) | 90 | 3.16 | 18.40 | 21.56 | 21.60 |

## 5. Per-Cell Suite 1 Head-to-Head Paired `Test MSE` & Spherical Cosine Distortion Table

| Cell | Solver | Baseline | Metric | Solver Mean | Base Mean | Mean Delta (%) | ±95% CI | W/T/L | `p_wilcox` |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | :---: | ---: |
| `s1_balanced_gmm_t1` | UltraEMeans | e-means | `test_mse` | 45.252484 | 55.476910 | **-18.4192%** | ±0.6423% | 15/0/0 | 6.104e-05 |
| `s1_balanced_gmm_t1` | UltraEMeans | s-means | `test_mse` | 45.252484 | 50.663366 | **-10.6699%** | ±0.6199% | 15/0/0 | 6.104e-05 |
| `s1_balanced_gmm_t1` | UltraEMeans | SuperEMeans | `test_mse` | 45.252484 | 48.519268 | **-6.7257%** | ±0.5450% | 15/0/0 | 6.104e-05 |
| `s1_balanced_gmm_t1` | UltraEMeans | Hybrid SuperEMeans | `test_mse` | 45.252484 | 48.151394 | **-6.0159%** | ±0.4893% | 15/0/0 | 6.104e-05 |
| `s1_balanced_gmm_t1` | UltraEMeans (fast) | s-means | `test_mse` | 45.360519 | 50.663366 | **-10.4575%** | ±0.5742% | 15/0/0 | 6.104e-05 |
| `s1_balanced_gmm_t1` | UltraEMeans (fast) | SuperEMeans (fast) | `test_mse` | 45.360519 | 48.669720 | **-6.7934%** | ±0.4709% | 15/0/0 | 6.104e-05 |
| `s1_balanced_gmm_t1` | UltraEMeans (fast) | Hybrid SuperEMeans (fast) | `test_mse` | 45.360519 | 48.296708 | **-6.0747%** | ±0.4744% | 15/0/0 | 6.104e-05 |
| `s1_balanced_gmm_t2` | UltraEMeans | e-means | `test_mse` | 44.431549 | 52.223186 | **-14.9086%** | ±0.6017% | 15/0/0 | 6.104e-05 |
| `s1_balanced_gmm_t2` | UltraEMeans | s-means | `test_mse` | 44.431549 | 48.207148 | **-7.8241%** | ±0.5311% | 15/0/0 | 6.104e-05 |
| `s1_balanced_gmm_t2` | UltraEMeans | SuperEMeans | `test_mse` | 44.431549 | 47.517359 | **-6.4893%** | ±0.4112% | 15/0/0 | 6.104e-05 |
| `s1_balanced_gmm_t2` | UltraEMeans | Hybrid SuperEMeans | `test_mse` | 44.431549 | 46.998296 | **-5.4565%** | ±0.3905% | 15/0/0 | 6.104e-05 |
| `s1_balanced_gmm_t2` | UltraEMeans (fast) | s-means | `test_mse` | 44.470242 | 48.207148 | **-7.7454%** | ±0.4313% | 15/0/0 | 6.104e-05 |
| `s1_balanced_gmm_t2` | UltraEMeans (fast) | SuperEMeans (fast) | `test_mse` | 44.470242 | 47.668648 | **-6.7057%** | ±0.3176% | 15/0/0 | 6.104e-05 |
| `s1_balanced_gmm_t2` | UltraEMeans (fast) | Hybrid SuperEMeans (fast) | `test_mse` | 44.470242 | 47.224404 | **-5.8297%** | ±0.2596% | 15/0/0 | 6.104e-05 |
| `s1_balanced_gmm_t5` | UltraEMeans | e-means | `test_mse` | 43.259050 | 49.710903 | **-12.9715%** | ±0.4773% | 15/0/0 | 6.104e-05 |
| `s1_balanced_gmm_t5` | UltraEMeans | s-means | `test_mse` | 43.259050 | 46.579432 | **-7.1245%** | ±0.3734% | 15/0/0 | 6.104e-05 |
| `s1_balanced_gmm_t5` | UltraEMeans | SuperEMeans | `test_mse` | 43.259050 | 45.863707 | **-5.6772%** | ±0.2368% | 15/0/0 | 6.104e-05 |
| `s1_balanced_gmm_t5` | UltraEMeans | Hybrid SuperEMeans | `test_mse` | 43.259050 | 45.375650 | **-4.6616%** | ±0.3019% | 15/0/0 | 6.104e-05 |
| `s1_balanced_gmm_t5` | UltraEMeans (fast) | s-means | `test_mse` | 43.292427 | 46.579432 | **-7.0534%** | ±0.3476% | 15/0/0 | 6.104e-05 |
| `s1_balanced_gmm_t5` | UltraEMeans (fast) | SuperEMeans (fast) | `test_mse` | 43.292427 | 46.296404 | **-6.4858%** | ±0.3261% | 15/0/0 | 6.104e-05 |
| `s1_balanced_gmm_t5` | UltraEMeans (fast) | Hybrid SuperEMeans (fast) | `test_mse` | 43.292427 | 45.875453 | **-5.6278%** | ±0.3208% | 15/0/0 | 6.104e-05 |
| `s1_cifar10_patches_128_t1` | UltraEMeans | e-means | `test_mse` | 1.443916 | 1.448522 | **-0.3179%** | ±0.0389% | 15/0/0 | 6.104e-05 |
| `s1_cifar10_patches_128_t1` | UltraEMeans | s-means | `test_mse` | 1.443916 | 1.464139 | **-1.3811%** | ±0.0428% | 15/0/0 | 6.104e-05 |
| `s1_cifar10_patches_128_t1` | UltraEMeans | SuperEMeans | `test_mse` | 1.443916 | 1.445403 | **-0.1029%** | ±0.0269% | 15/0/0 | 6.104e-05 |
| `s1_cifar10_patches_128_t1` | UltraEMeans | Hybrid SuperEMeans | `test_mse` | 1.443916 | 1.448451 | **-0.3131%** | ±0.0304% | 15/0/0 | 6.104e-05 |
| `s1_cifar10_patches_128_t1` | UltraEMeans (fast) | s-means | `test_mse` | 1.444276 | 1.464139 | **-1.3566%** | ±0.0481% | 15/0/0 | 6.104e-05 |
| `s1_cifar10_patches_128_t1` | UltraEMeans (fast) | SuperEMeans (fast) | `test_mse` | 1.444276 | 1.445931 | **-0.1144%** | ±0.0377% | 14/0/1 | 1.831e-04 |
| `s1_cifar10_patches_128_t1` | UltraEMeans (fast) | Hybrid SuperEMeans (fast) | `test_mse` | 1.444276 | 1.447968 | **-0.2549%** | ±0.0367% | 15/0/0 | 6.104e-05 |
| `s1_cifar10_patches_128_t2` | UltraEMeans | e-means | `test_mse` | 1.435832 | 1.436142 | -0.0216% | ±0.0451% | 11/0/4 | 2.078e-01 |
| `s1_cifar10_patches_128_t2` | UltraEMeans | s-means | `test_mse` | 1.435832 | 1.441322 | **-0.3808%** | ±0.0374% | 15/0/0 | 6.104e-05 |
| `s1_cifar10_patches_128_t2` | UltraEMeans | SuperEMeans | `test_mse` | 1.435832 | 1.435676 | +0.0109% | ±0.0380% | 8/0/7 | 7.197e-01 |
| `s1_cifar10_patches_128_t2` | UltraEMeans | Hybrid SuperEMeans | `test_mse` | 1.435832 | 1.438478 | **-0.1839%** | ±0.0269% | 15/0/0 | 6.104e-05 |
| `s1_cifar10_patches_128_t2` | UltraEMeans (fast) | s-means | `test_mse` | 1.436034 | 1.441322 | **-0.3669%** | ±0.0387% | 15/0/0 | 6.104e-05 |
| `s1_cifar10_patches_128_t2` | UltraEMeans (fast) | SuperEMeans (fast) | `test_mse` | 1.436034 | 1.435762 | +0.0189% | ±0.0370% | 5/0/10 | 3.303e-01 |
| `s1_cifar10_patches_128_t2` | UltraEMeans (fast) | Hybrid SuperEMeans (fast) | `test_mse` | 1.436034 | 1.437674 | **-0.1141%** | ±0.0367% | 14/0/1 | 1.221e-04 |
| `s1_cifar10_patches_128_t5` | UltraEMeans | e-means | `test_mse` | 1.430840 | 1.430733 | +0.0075% | ±0.0384% | 7/0/8 | 1.000e+00 |
| `s1_cifar10_patches_128_t5` | UltraEMeans | s-means | `test_mse` | 1.430840 | 1.432361 | **-0.1062%** | ±0.0305% | 15/0/0 | 6.104e-05 |
| `s1_cifar10_patches_128_t5` | UltraEMeans | SuperEMeans | `test_mse` | 1.430840 | 1.430139 | +0.0490% | ±0.0383% | 3/0/12 | 2.155e-02 |
| `s1_cifar10_patches_128_t5` | UltraEMeans | Hybrid SuperEMeans | `test_mse` | 1.430840 | 1.430665 | +0.0122% | ±0.0357% | 6/0/9 | 6.788e-01 |
| `s1_cifar10_patches_128_t5` | UltraEMeans (fast) | s-means | `test_mse` | 1.431050 | 1.432361 | **-0.0915%** | ±0.0242% | 15/0/0 | 6.104e-05 |
| `s1_cifar10_patches_128_t5` | UltraEMeans (fast) | SuperEMeans (fast) | `test_mse` | 1.431050 | 1.429960 | +0.0763% | ±0.0443% | 2/0/13 | 1.526e-03 |
| `s1_cifar10_patches_128_t5` | UltraEMeans (fast) | Hybrid SuperEMeans (fast) | `test_mse` | 1.431050 | 1.431331 | -0.0196% | ±0.0286% | 9/0/6 | 1.514e-01 |
| `s1_cifar10_r18_512_t1` | UltraEMeans | e-means | `test_mse` | 823.938588 | 827.305419 | **-0.4069%** | ±0.0386% | 15/0/0 | 6.104e-05 |
| `s1_cifar10_r18_512_t1` | UltraEMeans | s-means | `test_mse` | 823.938588 | 862.979682 | **-4.5237%** | ±0.1054% | 15/0/0 | 6.104e-05 |
| `s1_cifar10_r18_512_t1` | UltraEMeans | SuperEMeans | `test_mse` | 823.938588 | 822.348232 | +0.1935% | ±0.0732% | 1/0/14 | 3.052e-04 |
| `s1_cifar10_r18_512_t1` | UltraEMeans | Hybrid SuperEMeans | `test_mse` | 823.938588 | 826.939966 | **-0.3628%** | ±0.0758% | 15/0/0 | 6.104e-05 |
| `s1_cifar10_r18_512_t1` | UltraEMeans (fast) | s-means | `test_mse` | 824.243862 | 862.979682 | **-4.4883%** | ±0.1206% | 15/0/0 | 6.104e-05 |
| `s1_cifar10_r18_512_t1` | UltraEMeans (fast) | SuperEMeans (fast) | `test_mse` | 824.243862 | 822.979052 | +0.1539% | ±0.0945% | 3/0/12 | 6.714e-03 |
| `s1_cifar10_r18_512_t1` | UltraEMeans (fast) | Hybrid SuperEMeans (fast) | `test_mse` | 824.243862 | 826.517589 | **-0.2749%** | ±0.0977% | 15/0/0 | 6.104e-05 |
| `s1_cifar10_r18_512_t2` | UltraEMeans | e-means | `test_mse` | 816.757592 | 817.919390 | **-0.1419%** | ±0.1026% | 12/0/3 | 1.245e-02 |
| `s1_cifar10_r18_512_t2` | UltraEMeans | s-means | `test_mse` | 816.757592 | 841.840015 | **-2.9794%** | ±0.0869% | 15/0/0 | 6.104e-05 |
| `s1_cifar10_r18_512_t2` | UltraEMeans | SuperEMeans | `test_mse` | 816.757592 | 815.164162 | +0.1956% | ±0.1014% | 2/0/13 | 2.014e-03 |
| `s1_cifar10_r18_512_t2` | UltraEMeans | Hybrid SuperEMeans | `test_mse` | 816.757592 | 821.057575 | **-0.5234%** | ±0.1264% | 15/0/0 | 6.104e-05 |
| `s1_cifar10_r18_512_t2` | UltraEMeans (fast) | s-means | `test_mse` | 816.418116 | 841.840015 | **-3.0197%** | ±0.0866% | 15/0/0 | 6.104e-05 |
| `s1_cifar10_r18_512_t2` | UltraEMeans (fast) | SuperEMeans (fast) | `test_mse` | 816.418116 | 815.973262 | +0.0546% | ±0.0843% | 4/0/11 | 2.078e-01 |
| `s1_cifar10_r18_512_t2` | UltraEMeans (fast) | Hybrid SuperEMeans (fast) | `test_mse` | 816.418116 | 820.305956 | **-0.4739%** | ±0.0769% | 15/0/0 | 6.104e-05 |
| `s1_cifar10_r18_512_t5` | UltraEMeans | e-means | `test_mse` | 814.241519 | 815.083338 | **-0.1032%** | ±0.0483% | 14/0/1 | 1.831e-04 |
| `s1_cifar10_r18_512_t5` | UltraEMeans | s-means | `test_mse` | 814.241519 | 827.078374 | **-1.5519%** | ±0.0774% | 15/0/0 | 6.104e-05 |
| `s1_cifar10_r18_512_t5` | UltraEMeans | SuperEMeans | `test_mse` | 814.241519 | 812.373911 | +0.2300% | ±0.0515% | 0/0/15 | 6.104e-05 |
| `s1_cifar10_r18_512_t5` | UltraEMeans | Hybrid SuperEMeans | `test_mse` | 814.241519 | 816.125821 | **-0.2309%** | ±0.0385% | 15/0/0 | 6.104e-05 |
| `s1_cifar10_r18_512_t5` | UltraEMeans (fast) | s-means | `test_mse` | 814.373118 | 827.078374 | **-1.5361%** | ±0.0458% | 15/0/0 | 6.104e-05 |
| `s1_cifar10_r18_512_t5` | UltraEMeans (fast) | SuperEMeans (fast) | `test_mse` | 814.373118 | 812.342710 | +0.2500% | ±0.0656% | 0/0/15 | 6.104e-05 |
| `s1_cifar10_r18_512_t5` | UltraEMeans (fast) | Hybrid SuperEMeans (fast) | `test_mse` | 814.373118 | 816.128869 | **-0.2150%** | ±0.0725% | 15/0/0 | 6.104e-05 |
| `s1_cifar10_r18_512_unit_t1` | UltraEMeans | e-means | `test_mse` | 0.352532 | 0.354187 | **-0.4671%** | ±0.0937% | 15/0/0 | 6.104e-05 |
| `s1_cifar10_r18_512_unit_t1` | UltraEMeans | e-means | `test_cos_dist` | 0.195891 | 0.197055 | **-0.5904%** | ±0.1008% | 15/0/0 | 6.104e-05 |
| `s1_cifar10_r18_512_unit_t1` | UltraEMeans | s-means | `test_mse` | 0.352532 | 0.368044 | **-4.2143%** | ±0.0844% | 15/0/0 | 6.104e-05 |
| `s1_cifar10_r18_512_unit_t1` | UltraEMeans | s-means | `test_cos_dist` | 0.195891 | 0.205582 | **-4.7137%** | ±0.0963% | 15/0/0 | 6.104e-05 |
| `s1_cifar10_r18_512_unit_t1` | UltraEMeans | SuperEMeans | `test_mse` | 0.352532 | 0.352193 | +0.0963% | ±0.0691% | 3/0/12 | 1.508e-02 |
| `s1_cifar10_r18_512_unit_t1` | UltraEMeans | SuperEMeans | `test_cos_dist` | 0.195891 | 0.195770 | +0.0620% | ±0.0760% | 4/0/11 | 1.205e-01 |
| `s1_cifar10_r18_512_unit_t1` | UltraEMeans | Hybrid SuperEMeans | `test_mse` | 0.352532 | 0.352929 | **-0.1124%** | ±0.0576% | 12/0/3 | 8.545e-04 |
| `s1_cifar10_r18_512_unit_t1` | UltraEMeans | Hybrid SuperEMeans | `test_cos_dist` | 0.195891 | 0.196184 | **-0.1491%** | ±0.0636% | 15/0/0 | 6.104e-05 |
| `s1_cifar10_r18_512_unit_t1` | UltraEMeans (fast) | s-means | `test_mse` | 0.352640 | 0.368044 | **-4.1850%** | ±0.1014% | 15/0/0 | 6.104e-05 |
| `s1_cifar10_r18_512_unit_t1` | UltraEMeans (fast) | s-means | `test_cos_dist` | 0.195956 | 0.205582 | **-4.6822%** | ±0.1138% | 15/0/0 | 6.104e-05 |
| `s1_cifar10_r18_512_unit_t1` | UltraEMeans (fast) | SuperEMeans (fast) | `test_mse` | 0.352640 | 0.352416 | +0.0637% | ±0.0665% | 5/0/10 | 6.372e-02 |
| `s1_cifar10_r18_512_unit_t1` | UltraEMeans (fast) | SuperEMeans (fast) | `test_cos_dist` | 0.195956 | 0.195907 | +0.0249% | ±0.0735% | 6/0/9 | 5.614e-01 |
| `s1_cifar10_r18_512_unit_t1` | UltraEMeans (fast) | Hybrid SuperEMeans (fast) | `test_mse` | 0.352640 | 0.352916 | **-0.0781%** | ±0.0622% | 10/0/5 | 2.155e-02 |
| `s1_cifar10_r18_512_unit_t1` | UltraEMeans (fast) | Hybrid SuperEMeans (fast) | `test_cos_dist` | 0.195956 | 0.196161 | **-0.1044%** | ±0.0691% | 11/0/4 | 1.025e-02 |
| `s1_cifar10_r18_512_unit_t2` | UltraEMeans | e-means | `test_mse` | 0.349738 | 0.350245 | **-0.1445%** | ±0.0680% | 13/0/2 | 8.545e-04 |
| `s1_cifar10_r18_512_unit_t2` | UltraEMeans | e-means | `test_cos_dist` | 0.194184 | 0.194525 | **-0.1751%** | ±0.0760% | 13/0/2 | 6.104e-04 |
| `s1_cifar10_r18_512_unit_t2` | UltraEMeans | s-means | `test_mse` | 0.349738 | 0.360360 | **-2.9475%** | ±0.0613% | 15/0/0 | 6.104e-05 |
| `s1_cifar10_r18_512_unit_t2` | UltraEMeans | s-means | `test_cos_dist` | 0.194184 | 0.200836 | **-3.3120%** | ±0.0689% | 15/0/0 | 6.104e-05 |
| `s1_cifar10_r18_512_unit_t2` | UltraEMeans | SuperEMeans | `test_mse` | 0.349738 | 0.349498 | +0.0689% | ±0.0885% | 4/0/11 | 1.205e-01 |
| `s1_cifar10_r18_512_unit_t2` | UltraEMeans | SuperEMeans | `test_cos_dist` | 0.194184 | 0.194095 | +0.0463% | ±0.1010% | 5/0/10 | 3.303e-01 |
| `s1_cifar10_r18_512_unit_t2` | UltraEMeans | Hybrid SuperEMeans | `test_mse` | 0.349738 | 0.350866 | **-0.3214%** | ±0.0737% | 15/0/0 | 6.104e-05 |
| `s1_cifar10_r18_512_unit_t2` | UltraEMeans | Hybrid SuperEMeans | `test_cos_dist` | 0.194184 | 0.194857 | **-0.3451%** | ±0.0835% | 15/0/0 | 6.104e-05 |
| `s1_cifar10_r18_512_unit_t2` | UltraEMeans (fast) | s-means | `test_mse` | 0.349861 | 0.360360 | **-2.9134%** | ±0.0625% | 15/0/0 | 6.104e-05 |
| `s1_cifar10_r18_512_unit_t2` | UltraEMeans (fast) | s-means | `test_cos_dist` | 0.194259 | 0.200836 | **-3.2750%** | ±0.0688% | 15/0/0 | 6.104e-05 |
| `s1_cifar10_r18_512_unit_t2` | UltraEMeans (fast) | SuperEMeans (fast) | `test_mse` | 0.349861 | 0.349541 | +0.0916% | ±0.0426% | 2/0/13 | 8.545e-04 |
| `s1_cifar10_r18_512_unit_t2` | UltraEMeans (fast) | SuperEMeans (fast) | `test_cos_dist` | 0.194259 | 0.194128 | +0.0672% | ±0.0475% | 5/0/10 | 1.508e-02 |
| `s1_cifar10_r18_512_unit_t2` | UltraEMeans (fast) | Hybrid SuperEMeans (fast) | `test_mse` | 0.349861 | 0.350791 | **-0.2651%** | ±0.0417% | 15/0/0 | 6.104e-05 |
| `s1_cifar10_r18_512_unit_t2` | UltraEMeans (fast) | Hybrid SuperEMeans (fast) | `test_cos_dist` | 0.194259 | 0.194801 | **-0.2781%** | ±0.0470% | 15/0/0 | 6.104e-05 |
| `s1_cifar10_r18_512_unit_t5` | UltraEMeans | e-means | `test_mse` | 0.348369 | 0.348785 | **-0.1191%** | ±0.0683% | 11/0/4 | 2.625e-03 |
| `s1_cifar10_r18_512_unit_t5` | UltraEMeans | e-means | `test_cos_dist` | 0.193333 | 0.193592 | **-0.1334%** | ±0.0749% | 11/0/4 | 4.272e-03 |
| `s1_cifar10_r18_512_unit_t5` | UltraEMeans | s-means | `test_mse` | 0.348369 | 0.353891 | **-1.5601%** | ±0.0790% | 15/0/0 | 6.104e-05 |
| `s1_cifar10_r18_512_unit_t5` | UltraEMeans | s-means | `test_cos_dist` | 0.193333 | 0.196729 | **-1.7261%** | ±0.0889% | 15/0/0 | 6.104e-05 |
| `s1_cifar10_r18_512_unit_t5` | UltraEMeans | SuperEMeans | `test_mse` | 0.348369 | 0.348085 | +0.0818% | ±0.0615% | 3/0/12 | 1.807e-02 |
| `s1_cifar10_r18_512_unit_t5` | UltraEMeans | SuperEMeans | `test_cos_dist` | 0.193333 | 0.193219 | +0.0593% | ±0.0667% | 5/0/10 | 1.205e-01 |
| `s1_cifar10_r18_512_unit_t5` | UltraEMeans | Hybrid SuperEMeans | `test_mse` | 0.348369 | 0.348943 | **-0.1642%** | ±0.0673% | 13/0/2 | 6.104e-04 |
| `s1_cifar10_r18_512_unit_t5` | UltraEMeans | Hybrid SuperEMeans | `test_cos_dist` | 0.193333 | 0.193665 | **-0.1714%** | ±0.0718% | 13/0/2 | 6.104e-04 |
| `s1_cifar10_r18_512_unit_t5` | UltraEMeans (fast) | s-means | `test_mse` | 0.348370 | 0.353891 | **-1.5601%** | ±0.0619% | 15/0/0 | 6.104e-05 |
| `s1_cifar10_r18_512_unit_t5` | UltraEMeans (fast) | s-means | `test_cos_dist` | 0.193331 | 0.196729 | **-1.7272%** | ±0.0709% | 15/0/0 | 6.104e-05 |
| `s1_cifar10_r18_512_unit_t5` | UltraEMeans (fast) | SuperEMeans (fast) | `test_mse` | 0.348370 | 0.348029 | +0.0979% | ±0.0593% | 1/0/14 | 1.526e-03 |
| `s1_cifar10_r18_512_unit_t5` | UltraEMeans (fast) | SuperEMeans (fast) | `test_cos_dist` | 0.193331 | 0.193190 | +0.0732% | ±0.0650% | 5/0/10 | 4.126e-02 |
| `s1_cifar10_r18_512_unit_t5` | UltraEMeans (fast) | Hybrid SuperEMeans (fast) | `test_mse` | 0.348370 | 0.349014 | **-0.1844%** | ±0.0792% | 13/0/2 | 4.272e-04 |
| `s1_cifar10_r18_512_unit_t5` | UltraEMeans (fast) | Hybrid SuperEMeans (fast) | `test_cos_dist` | 0.193331 | 0.193709 | **-0.1951%** | ±0.0855% | 13/0/2 | 6.104e-04 |
| `s1_isotropic_ctrl_t1` | UltraEMeans | e-means | `test_mse` | 119.047321 | 119.013465 | +0.0285% | ±0.0200% | 2/0/13 | 1.025e-02 |
| `s1_isotropic_ctrl_t1` | UltraEMeans | s-means | `test_mse` | 119.047321 | 119.759933 | **-0.5950%** | ±0.0161% | 15/0/0 | 6.104e-05 |
| `s1_isotropic_ctrl_t1` | UltraEMeans | SuperEMeans | `test_mse` | 119.047321 | 118.979185 | +0.0573% | ±0.0126% | 0/0/15 | 6.104e-05 |
| `s1_isotropic_ctrl_t1` | UltraEMeans | Hybrid SuperEMeans | `test_mse` | 119.047321 | 119.651989 | **-0.5053%** | ±0.0180% | 15/0/0 | 6.104e-05 |
| `s1_isotropic_ctrl_t1` | UltraEMeans (fast) | s-means | `test_mse` | 119.052082 | 119.759933 | **-0.5911%** | ±0.0153% | 15/0/0 | 6.104e-05 |
| `s1_isotropic_ctrl_t1` | UltraEMeans (fast) | SuperEMeans (fast) | `test_mse` | 119.052082 | 118.967630 | +0.0710% | ±0.0136% | 0/0/15 | 6.104e-05 |
| `s1_isotropic_ctrl_t1` | UltraEMeans (fast) | Hybrid SuperEMeans (fast) | `test_mse` | 119.052082 | 119.612614 | **-0.4686%** | ±0.0202% | 15/0/0 | 6.104e-05 |
| `s1_isotropic_ctrl_t2` | UltraEMeans | e-means | `test_mse` | 119.019855 | 119.020473 | -0.0005% | ±0.0160% | 9/0/6 | 8.904e-01 |
| `s1_isotropic_ctrl_t2` | UltraEMeans | s-means | `test_mse` | 119.019855 | 119.508519 | **-0.4089%** | ±0.0277% | 15/0/0 | 6.104e-05 |
| `s1_isotropic_ctrl_t2` | UltraEMeans | SuperEMeans | `test_mse` | 119.019855 | 118.988228 | +0.0266% | ±0.0128% | 1/0/14 | 1.831e-04 |
| `s1_isotropic_ctrl_t2` | UltraEMeans | Hybrid SuperEMeans | `test_mse` | 119.019855 | 120.316405 | **-1.0775%** | ±0.0669% | 15/0/0 | 6.104e-05 |
| `s1_isotropic_ctrl_t2` | UltraEMeans (fast) | s-means | `test_mse` | 119.028921 | 119.508519 | **-0.4013%** | ±0.0259% | 15/0/0 | 6.104e-05 |
| `s1_isotropic_ctrl_t2` | UltraEMeans (fast) | SuperEMeans (fast) | `test_mse` | 119.028921 | 118.986266 | +0.0359% | ±0.0114% | 0/0/15 | 6.104e-05 |
| `s1_isotropic_ctrl_t2` | UltraEMeans (fast) | Hybrid SuperEMeans (fast) | `test_mse` | 119.028921 | 119.660523 | **-0.5278%** | ±0.0406% | 15/0/0 | 6.104e-05 |
| `s1_isotropic_ctrl_t5` | UltraEMeans | e-means | `test_mse` | 119.061919 | 119.064059 | -0.0018% | ±0.0120% | 6/0/9 | 9.341e-01 |
| `s1_isotropic_ctrl_t5` | UltraEMeans | s-means | `test_mse` | 119.061919 | 119.587861 | **-0.4397%** | ±0.0435% | 15/0/0 | 6.104e-05 |
| `s1_isotropic_ctrl_t5` | UltraEMeans | SuperEMeans | `test_mse` | 119.061919 | 119.030501 | +0.0264% | ±0.0156% | 4/0/11 | 5.371e-03 |
| `s1_isotropic_ctrl_t5` | UltraEMeans | Hybrid SuperEMeans | `test_mse` | 119.061919 | 120.186527 | **-0.9354%** | ±0.1051% | 15/0/0 | 6.104e-05 |
| `s1_isotropic_ctrl_t5` | UltraEMeans (fast) | s-means | `test_mse` | 119.068398 | 119.587861 | **-0.4343%** | ±0.0377% | 15/0/0 | 6.104e-05 |
| `s1_isotropic_ctrl_t5` | UltraEMeans (fast) | SuperEMeans (fast) | `test_mse` | 119.068398 | 119.025755 | +0.0358% | ±0.0170% | 1/0/14 | 3.052e-04 |
| `s1_isotropic_ctrl_t5` | UltraEMeans (fast) | Hybrid SuperEMeans (fast) | `test_mse` | 119.068398 | 120.041895 | **-0.8101%** | ±0.1735% | 15/0/0 | 6.104e-05 |
| `s1_low_nk32_gmm_t1` | UltraEMeans | e-means | `test_mse` | 49.040864 | 60.953052 | **-19.5404%** | ±0.4148% | 15/0/0 | 6.104e-05 |
| `s1_low_nk32_gmm_t1` | UltraEMeans | s-means | `test_mse` | 49.040864 | 57.329431 | **-14.4412%** | ±0.8154% | 15/0/0 | 6.104e-05 |
| `s1_low_nk32_gmm_t1` | UltraEMeans | SuperEMeans | `test_mse` | 49.040864 | 51.645805 | **-5.0402%** | ±0.4797% | 15/0/0 | 6.104e-05 |
| `s1_low_nk32_gmm_t1` | UltraEMeans | Hybrid SuperEMeans | `test_mse` | 49.040864 | 51.270918 | **-4.3461%** | ±0.4945% | 15/0/0 | 6.104e-05 |
| `s1_low_nk32_gmm_t1` | UltraEMeans (fast) | s-means | `test_mse` | 49.339821 | 57.329431 | **-13.9184%** | ±0.8596% | 15/0/0 | 6.104e-05 |
| `s1_low_nk32_gmm_t1` | UltraEMeans (fast) | SuperEMeans (fast) | `test_mse` | 49.339821 | 51.851988 | **-4.8404%** | ±0.5277% | 15/0/0 | 6.104e-05 |
| `s1_low_nk32_gmm_t1` | UltraEMeans (fast) | Hybrid SuperEMeans (fast) | `test_mse` | 49.339821 | 51.461068 | **-4.1182%** | ±0.5106% | 15/0/0 | 6.104e-05 |
| `s1_low_nk32_gmm_t2` | UltraEMeans | e-means | `test_mse` | 46.961700 | 57.736256 | **-18.6560%** | ±0.4660% | 15/0/0 | 6.104e-05 |
| `s1_low_nk32_gmm_t2` | UltraEMeans | s-means | `test_mse` | 46.961700 | 54.737641 | **-14.1911%** | ±0.7173% | 15/0/0 | 6.104e-05 |
| `s1_low_nk32_gmm_t2` | UltraEMeans | SuperEMeans | `test_mse` | 46.961700 | 50.283575 | **-6.6039%** | ±0.3323% | 15/0/0 | 6.104e-05 |
| `s1_low_nk32_gmm_t2` | UltraEMeans | Hybrid SuperEMeans | `test_mse` | 46.961700 | 49.815416 | **-5.7247%** | ±0.4493% | 15/0/0 | 6.104e-05 |
| `s1_low_nk32_gmm_t2` | UltraEMeans (fast) | s-means | `test_mse` | 46.917791 | 54.737641 | **-14.2708%** | ±0.7523% | 15/0/0 | 6.104e-05 |
| `s1_low_nk32_gmm_t2` | UltraEMeans (fast) | SuperEMeans (fast) | `test_mse` | 46.917791 | 50.425720 | **-6.9530%** | ±0.4452% | 15/0/0 | 6.104e-05 |
| `s1_low_nk32_gmm_t2` | UltraEMeans (fast) | Hybrid SuperEMeans (fast) | `test_mse` | 46.917791 | 49.931325 | **-6.0316%** | ±0.4835% | 15/0/0 | 6.104e-05 |
| `s1_low_nk32_gmm_t5` | UltraEMeans | e-means | `test_mse` | 45.599602 | 53.212677 | **-14.3043%** | ±0.3968% | 15/0/0 | 6.104e-05 |
| `s1_low_nk32_gmm_t5` | UltraEMeans | s-means | `test_mse` | 45.599602 | 49.443620 | **-7.7714%** | ±0.4379% | 15/0/0 | 6.104e-05 |
| `s1_low_nk32_gmm_t5` | UltraEMeans | SuperEMeans | `test_mse` | 45.599602 | 49.348708 | **-7.5918%** | ±0.5134% | 15/0/0 | 6.104e-05 |
| `s1_low_nk32_gmm_t5` | UltraEMeans | Hybrid SuperEMeans | `test_mse` | 45.599602 | 48.854310 | **-6.6578%** | ±0.4590% | 15/0/0 | 6.104e-05 |
| `s1_low_nk32_gmm_t5` | UltraEMeans (fast) | s-means | `test_mse` | 45.582667 | 49.443620 | **-7.8058%** | ±0.4114% | 15/0/0 | 6.104e-05 |
| `s1_low_nk32_gmm_t5` | UltraEMeans (fast) | SuperEMeans (fast) | `test_mse` | 45.582667 | 49.477477 | **-7.8691%** | ±0.3963% | 15/0/0 | 6.104e-05 |
| `s1_low_nk32_gmm_t5` | UltraEMeans (fast) | Hybrid SuperEMeans (fast) | `test_mse` | 45.582667 | 49.165198 | **-7.2825%** | ±0.4372% | 15/0/0 | 6.104e-05 |
| `s1_lowd_zipf_d8_t1` | UltraEMeans | e-means | `test_mse` | 1.663022 | 1.663033 | -0.0001% | ±0.1883% | 10/0/5 | 5.245e-01 |
| `s1_lowd_zipf_d8_t1` | UltraEMeans | s-means | `test_mse` | 1.663022 | 1.662060 | +0.0584% | ±0.1796% | 6/0/9 | 5.614e-01 |
| `s1_lowd_zipf_d8_t1` | UltraEMeans | SuperEMeans | `test_mse` | 1.663022 | 1.665550 | -0.1514% | ±0.1743% | 11/0/4 | 8.325e-02 |
| `s1_lowd_zipf_d8_t1` | UltraEMeans | Hybrid SuperEMeans | `test_mse` | 1.663022 | 1.661795 | +0.0746% | ±0.2276% | 8/0/7 | 5.995e-01 |
| `s1_lowd_zipf_d8_t1` | UltraEMeans (fast) | s-means | `test_mse` | 1.663611 | 1.662060 | +0.0938% | ±0.1838% | 7/0/8 | 3.303e-01 |
| `s1_lowd_zipf_d8_t1` | UltraEMeans (fast) | SuperEMeans (fast) | `test_mse` | 1.663611 | 1.665702 | -0.1249% | ±0.2146% | 8/0/7 | 2.524e-01 |
| `s1_lowd_zipf_d8_t1` | UltraEMeans (fast) | Hybrid SuperEMeans (fast) | `test_mse` | 1.663611 | 1.661873 | +0.1055% | ±0.2448% | 7/0/8 | 5.245e-01 |
| `s1_lowd_zipf_d8_t2` | UltraEMeans | e-means | `test_mse` | 1.648248 | 1.647821 | +0.0260% | ±0.1188% | 6/0/9 | 5.995e-01 |
| `s1_lowd_zipf_d8_t2` | UltraEMeans | s-means | `test_mse` | 1.648248 | 1.646472 | +0.1081% | ±0.1455% | 4/0/11 | 9.460e-02 |
| `s1_lowd_zipf_d8_t2` | UltraEMeans | SuperEMeans | `test_mse` | 1.648248 | 1.651219 | **-0.1797%** | ±0.1339% | 13/0/2 | 1.025e-02 |
| `s1_lowd_zipf_d8_t2` | UltraEMeans | Hybrid SuperEMeans | `test_mse` | 1.648248 | 1.649049 | -0.0482% | ±0.1435% | 10/0/5 | 4.212e-01 |
| `s1_lowd_zipf_d8_t2` | UltraEMeans (fast) | s-means | `test_mse` | 1.648655 | 1.646472 | +0.1330% | ±0.1727% | 5/0/10 | 1.514e-01 |
| `s1_lowd_zipf_d8_t2` | UltraEMeans (fast) | SuperEMeans (fast) | `test_mse` | 1.648655 | 1.650919 | -0.1366% | ±0.1757% | 10/0/5 | 2.078e-01 |
| `s1_lowd_zipf_d8_t2` | UltraEMeans (fast) | Hybrid SuperEMeans (fast) | `test_mse` | 1.648655 | 1.649622 | -0.0583% | ±0.1394% | 10/0/5 | 3.894e-01 |
| `s1_lowd_zipf_d8_t5` | UltraEMeans | e-means | `test_mse` | 1.639515 | 1.639911 | -0.0238% | ±0.1370% | 9/0/6 | 5.245e-01 |
| `s1_lowd_zipf_d8_t5` | UltraEMeans | s-means | `test_mse` | 1.639515 | 1.638241 | +0.0782% | ±0.1606% | 6/0/9 | 2.524e-01 |
| `s1_lowd_zipf_d8_t5` | UltraEMeans | SuperEMeans | `test_mse` | 1.639515 | 1.642891 | **-0.2051%** | ±0.1422% | 12/0/3 | 1.508e-02 |
| `s1_lowd_zipf_d8_t5` | UltraEMeans | Hybrid SuperEMeans | `test_mse` | 1.639515 | 1.642696 | **-0.1934%** | ±0.1414% | 12/0/3 | 2.155e-02 |
| `s1_lowd_zipf_d8_t5` | UltraEMeans (fast) | s-means | `test_mse` | 1.639938 | 1.638241 | +0.1040% | ±0.1466% | 6/0/9 | 1.354e-01 |
| `s1_lowd_zipf_d8_t5` | UltraEMeans (fast) | SuperEMeans (fast) | `test_mse` | 1.639938 | 1.641495 | -0.0942% | ±0.1828% | 9/0/6 | 2.524e-01 |
| `s1_lowd_zipf_d8_t5` | UltraEMeans (fast) | Hybrid SuperEMeans (fast) | `test_mse` | 1.639938 | 1.641976 | -0.1238% | ±0.1404% | 11/0/4 | 8.325e-02 |
| `s1_multiscale_rare_t1` | UltraEMeans | e-means | `test_mse` | 17.322856 | 25.763620 | **-32.6467%** | ±1.6694% | 15/0/0 | 6.104e-05 |
| `s1_multiscale_rare_t1` | UltraEMeans | s-means | `test_mse` | 17.322856 | 20.104336 | **-13.8280%** | ±0.9915% | 15/0/0 | 6.104e-05 |
| `s1_multiscale_rare_t1` | UltraEMeans | SuperEMeans | `test_mse` | 17.322856 | 18.313394 | **-5.3806%** | ±1.4091% | 15/0/0 | 6.104e-05 |
| `s1_multiscale_rare_t1` | UltraEMeans | Hybrid SuperEMeans | `test_mse` | 17.322856 | 18.179289 | **-4.6947%** | ±1.1111% | 15/0/0 | 6.104e-05 |
| `s1_multiscale_rare_t1` | UltraEMeans (fast) | s-means | `test_mse` | 17.208018 | 20.104336 | **-14.3973%** | ±0.9989% | 15/0/0 | 6.104e-05 |
| `s1_multiscale_rare_t1` | UltraEMeans (fast) | SuperEMeans (fast) | `test_mse` | 17.208018 | 18.805579 | **-8.4753%** | ±0.9937% | 15/0/0 | 6.104e-05 |
| `s1_multiscale_rare_t1` | UltraEMeans (fast) | Hybrid SuperEMeans (fast) | `test_mse` | 17.208018 | 18.682631 | **-7.8827%** | ±0.6961% | 15/0/0 | 6.104e-05 |
| `s1_multiscale_rare_t2` | UltraEMeans | e-means | `test_mse` | 17.004289 | 23.702015 | **-28.1466%** | ±1.7951% | 15/0/0 | 6.104e-05 |
| `s1_multiscale_rare_t2` | UltraEMeans | s-means | `test_mse` | 17.004289 | 19.124537 | **-11.0778%** | ±1.0069% | 15/0/0 | 6.104e-05 |
| `s1_multiscale_rare_t2` | UltraEMeans | SuperEMeans | `test_mse` | 17.004289 | 18.124290 | **-6.1207%** | ±1.5987% | 15/0/0 | 6.104e-05 |
| `s1_multiscale_rare_t2` | UltraEMeans | Hybrid SuperEMeans | `test_mse` | 17.004289 | 17.797952 | **-4.4051%** | ±1.5750% | 14/0/1 | 3.052e-04 |
| `s1_multiscale_rare_t2` | UltraEMeans (fast) | s-means | `test_mse` | 16.894893 | 19.124537 | **-11.6445%** | ±1.2075% | 15/0/0 | 6.104e-05 |
| `s1_multiscale_rare_t2` | UltraEMeans (fast) | SuperEMeans (fast) | `test_mse` | 16.894893 | 18.211018 | **-7.1951%** | ±1.3819% | 15/0/0 | 6.104e-05 |
| `s1_multiscale_rare_t2` | UltraEMeans (fast) | Hybrid SuperEMeans (fast) | `test_mse` | 16.894893 | 17.961374 | **-5.9009%** | ±1.5352% | 15/0/0 | 6.104e-05 |
| `s1_multiscale_rare_t5` | UltraEMeans | e-means | `test_mse` | 16.514806 | 21.442195 | **-22.9062%** | ±1.5082% | 15/0/0 | 6.104e-05 |
| `s1_multiscale_rare_t5` | UltraEMeans | s-means | `test_mse` | 16.514806 | 18.582627 | **-11.1183%** | ±0.7248% | 15/0/0 | 6.104e-05 |
| `s1_multiscale_rare_t5` | UltraEMeans | SuperEMeans | `test_mse` | 16.514806 | 17.142057 | **-3.6378%** | ±0.9698% | 15/0/0 | 6.104e-05 |
| `s1_multiscale_rare_t5` | UltraEMeans | Hybrid SuperEMeans | `test_mse` | 16.514806 | 17.012390 | **-2.8943%** | ±1.0456% | 13/0/2 | 3.052e-04 |
| `s1_multiscale_rare_t5` | UltraEMeans (fast) | s-means | `test_mse` | 16.292434 | 18.582627 | **-12.3153%** | ±0.8038% | 15/0/0 | 6.104e-05 |
| `s1_multiscale_rare_t5` | UltraEMeans (fast) | SuperEMeans (fast) | `test_mse` | 16.292434 | 17.163835 | **-5.0586%** | ±0.8720% | 15/0/0 | 6.104e-05 |
| `s1_multiscale_rare_t5` | UltraEMeans (fast) | Hybrid SuperEMeans (fast) | `test_mse` | 16.292434 | 17.078779 | **-4.5760%** | ±1.0867% | 15/0/0 | 6.104e-05 |
| `s1_stretched_a2_t1` | UltraEMeans | e-means | `test_mse` | 18.967513 | 19.067492 | **-0.5242%** | ±0.0784% | 15/0/0 | 6.104e-05 |
| `s1_stretched_a2_t1` | UltraEMeans | s-means | `test_mse` | 18.967513 | 18.991828 | **-0.1278%** | ±0.0988% | 13/0/2 | 3.534e-02 |
| `s1_stretched_a2_t1` | UltraEMeans | SuperEMeans | `test_mse` | 18.967513 | 18.996867 | **-0.1544%** | ±0.0853% | 13/0/2 | 3.357e-03 |
| `s1_stretched_a2_t1` | UltraEMeans | Hybrid SuperEMeans | `test_mse` | 18.967513 | 18.969526 | -0.0104% | ±0.0958% | 7/0/8 | 9.341e-01 |
| `s1_stretched_a2_t1` | UltraEMeans (fast) | s-means | `test_mse` | 18.975697 | 18.991828 | -0.0848% | ±0.0924% | 10/0/5 | 6.372e-02 |
| `s1_stretched_a2_t1` | UltraEMeans (fast) | SuperEMeans (fast) | `test_mse` | 18.975697 | 19.000306 | **-0.1294%** | ±0.0688% | 12/0/3 | 1.160e-03 |
| `s1_stretched_a2_t1` | UltraEMeans (fast) | Hybrid SuperEMeans (fast) | `test_mse` | 18.975697 | 18.974020 | +0.0090% | ±0.0757% | 7/0/8 | 7.615e-01 |
| `s1_stretched_a2_t2` | UltraEMeans | e-means | `test_mse` | 18.901091 | 18.938259 | **-0.1959%** | ±0.1324% | 13/0/2 | 4.272e-03 |
| `s1_stretched_a2_t2` | UltraEMeans | s-means | `test_mse` | 18.901091 | 18.910416 | -0.0492% | ±0.0755% | 10/0/5 | 2.524e-01 |
| `s1_stretched_a2_t2` | UltraEMeans | SuperEMeans | `test_mse` | 18.901091 | 18.897231 | +0.0208% | ±0.1198% | 8/0/7 | 8.469e-01 |
| `s1_stretched_a2_t2` | UltraEMeans | Hybrid SuperEMeans | `test_mse` | 18.901091 | 18.906594 | -0.0289% | ±0.1011% | 8/0/7 | 7.197e-01 |
| `s1_stretched_a2_t2` | UltraEMeans (fast) | s-means | `test_mse` | 18.900165 | 18.910416 | -0.0541% | ±0.0721% | 11/0/4 | 1.070e-01 |
| `s1_stretched_a2_t2` | UltraEMeans (fast) | SuperEMeans (fast) | `test_mse` | 18.900165 | 18.900068 | +0.0007% | ±0.0959% | 8/0/7 | 9.780e-01 |
| `s1_stretched_a2_t2` | UltraEMeans (fast) | Hybrid SuperEMeans (fast) | `test_mse` | 18.900165 | 18.905545 | -0.0282% | ±0.1047% | 9/0/6 | 5.245e-01 |
| `s1_stretched_a2_t5` | UltraEMeans | e-means | `test_mse` | 18.869138 | 18.893607 | **-0.1294%** | ±0.0741% | 11/0/4 | 6.714e-03 |
| `s1_stretched_a2_t5` | UltraEMeans | s-means | `test_mse` | 18.869138 | 18.885355 | **-0.0858%** | ±0.0645% | 12/0/3 | 1.025e-02 |
| `s1_stretched_a2_t5` | UltraEMeans | SuperEMeans | `test_mse` | 18.869138 | 18.867981 | +0.0062% | ±0.0795% | 9/0/6 | 9.780e-01 |
| `s1_stretched_a2_t5` | UltraEMeans | Hybrid SuperEMeans | `test_mse` | 18.869138 | 18.881243 | -0.0640% | ±0.0712% | 10/0/5 | 9.460e-02 |
| `s1_stretched_a2_t5` | UltraEMeans (fast) | s-means | `test_mse` | 18.863511 | 18.885355 | **-0.1156%** | ±0.0610% | 13/0/2 | 4.272e-03 |
| `s1_stretched_a2_t5` | UltraEMeans (fast) | SuperEMeans (fast) | `test_mse` | 18.863511 | 18.867307 | -0.0200% | ±0.0736% | 11/0/4 | 3.591e-01 |
| `s1_stretched_a2_t5` | UltraEMeans (fast) | Hybrid SuperEMeans (fast) | `test_mse` | 18.863511 | 18.878446 | -0.0790% | ±0.0851% | 12/0/3 | 5.536e-02 |
| `s1_zipf_gmm_t1` | UltraEMeans | e-means | `test_mse` | 47.961059 | 52.149017 | **-8.0260%** | ±0.4798% | 15/0/0 | 6.104e-05 |
| `s1_zipf_gmm_t1` | UltraEMeans | s-means | `test_mse` | 47.961059 | 50.875044 | **-5.7208%** | ±0.5409% | 15/0/0 | 6.104e-05 |
| `s1_zipf_gmm_t1` | UltraEMeans | SuperEMeans | `test_mse` | 47.961059 | 49.539610 | **-3.1831%** | ±0.4341% | 15/0/0 | 6.104e-05 |
| `s1_zipf_gmm_t1` | UltraEMeans | Hybrid SuperEMeans | `test_mse` | 47.961059 | 49.642110 | **-3.3809%** | ±0.5360% | 15/0/0 | 6.104e-05 |
| `s1_zipf_gmm_t1` | UltraEMeans (fast) | s-means | `test_mse` | 47.987738 | 50.875044 | **-5.6692%** | ±0.4866% | 15/0/0 | 6.104e-05 |
| `s1_zipf_gmm_t1` | UltraEMeans (fast) | SuperEMeans (fast) | `test_mse` | 47.987738 | 49.593385 | **-3.2346%** | ±0.3849% | 15/0/0 | 6.104e-05 |
| `s1_zipf_gmm_t1` | UltraEMeans (fast) | Hybrid SuperEMeans (fast) | `test_mse` | 47.987738 | 49.631908 | **-3.3093%** | ±0.4239% | 15/0/0 | 6.104e-05 |
| `s1_zipf_gmm_t2` | UltraEMeans | e-means | `test_mse` | 47.231566 | 50.405946 | **-6.2946%** | ±0.3886% | 15/0/0 | 6.104e-05 |
| `s1_zipf_gmm_t2` | UltraEMeans | s-means | `test_mse` | 47.231566 | 49.119045 | **-3.8406%** | ±0.3538% | 15/0/0 | 6.104e-05 |
| `s1_zipf_gmm_t2` | UltraEMeans | SuperEMeans | `test_mse` | 47.231566 | 48.742648 | **-3.0982%** | ±0.3349% | 15/0/0 | 6.104e-05 |
| `s1_zipf_gmm_t2` | UltraEMeans | Hybrid SuperEMeans | `test_mse` | 47.231566 | 48.460933 | **-2.5344%** | ±0.3825% | 15/0/0 | 6.104e-05 |
| `s1_zipf_gmm_t2` | UltraEMeans (fast) | s-means | `test_mse` | 47.199822 | 49.119045 | **-3.9051%** | ±0.3490% | 15/0/0 | 6.104e-05 |
| `s1_zipf_gmm_t2` | UltraEMeans (fast) | SuperEMeans (fast) | `test_mse` | 47.199822 | 48.793266 | **-3.2636%** | ±0.3300% | 15/0/0 | 6.104e-05 |
| `s1_zipf_gmm_t2` | UltraEMeans (fast) | Hybrid SuperEMeans (fast) | `test_mse` | 47.199822 | 48.421997 | **-2.5225%** | ±0.3041% | 15/0/0 | 6.104e-05 |
| `s1_zipf_gmm_t5` | UltraEMeans | e-means | `test_mse` | 46.475486 | 49.394907 | **-5.9086%** | ±0.2950% | 15/0/0 | 6.104e-05 |
| `s1_zipf_gmm_t5` | UltraEMeans | s-means | `test_mse` | 46.475486 | 47.670893 | **-2.5048%** | ±0.3637% | 15/0/0 | 6.104e-05 |
| `s1_zipf_gmm_t5` | UltraEMeans | SuperEMeans | `test_mse` | 46.475486 | 48.174096 | **-3.5236%** | ±0.3170% | 15/0/0 | 6.104e-05 |
| `s1_zipf_gmm_t5` | UltraEMeans | Hybrid SuperEMeans | `test_mse` | 46.475486 | 46.951487 | **-1.0122%** | ±0.2897% | 15/0/0 | 6.104e-05 |
| `s1_zipf_gmm_t5` | UltraEMeans (fast) | s-means | `test_mse` | 46.494984 | 47.670893 | **-2.4645%** | ±0.3012% | 15/0/0 | 6.104e-05 |
| `s1_zipf_gmm_t5` | UltraEMeans (fast) | SuperEMeans (fast) | `test_mse` | 46.494984 | 48.131536 | **-3.3986%** | ±0.2135% | 15/0/0 | 6.104e-05 |
| `s1_zipf_gmm_t5` | UltraEMeans (fast) | Hybrid SuperEMeans (fast) | `test_mse` | 46.494984 | 46.853531 | **-0.7644%** | ±0.2011% | 15/0/0 | 6.104e-05 |
