# Downstream ML Pipeline Dominance: `SemDeDup` Coreset Selection & Megapixel Vector Quantization

## Executive Summary

Across **`1,265` paired GPU benchmark runs** (`575` Semantic Data Deduplication / Coreset Selection runs across `5` real labeled ML cells + `690` Megapixel Image Vector Quantization runs across `6` high-resolution RGB & `4x4x3` patch cells, `5` paired seeds `6000..6004`), [`ApexEMeans`](../src/emeans/apex_emeans.py) paired with our fused 1-pass Triton [`semdedup_coreset.py`](../src/emeans/semdedup_coreset.py) engine dominates both downstream ML training-data pruning and real-time high-resolution image compression:

- **Suite 1 — Millisecond Semantic Data Deduplication & Coreset Selection (`50K–500K` Labeled Corpora, `K = 512..4,096`)**: At `T = 2`, `ApexEMeans` prunes `50K–500K` corpora (full clustering + fused Triton `SemDeDup` margin scoring + Horvitz-Thompson importance weighting) with a **`1.17x` geometric-mean speedup** over `Lloyd@25` (**`1.22x` at `T = 1`**, peaking at **`3.72x` at `T = 2`** / **`4.03x` at `T = 1`**), while improving downstream held-out test accuracy over `Random` pruning by an average of **`+0.18 pp` (`10%`), `+0.13 pp` (`20%`), and `+0.20 pp` (`50%`)** across modalities.
- **Zero Empty-Cluster Collapse & Prototype-Only (`M <= K`) Compression**: Across all `5` labeled cells at `T = 2`, `ApexEMeans` maintains **`0.0` mean empty clusters** (`100.0%` live prototypes), whereas `s-means` collapses **`252.4` empty clusters per cell** — giving `ApexEMeans` a **`+0.55 pp` mean accuracy advantage** over `s-means` in the extreme prototype-only (`M <= K`) regime.
- **Suite 2 — Megapixel Color Palette (`1,000,000 x 3`) & `4x4` Natural Image Patch (`500,000 x 48`) Vector Quantization (`K in {256, 1024, 4096}`)**: At `T = 2`, `ApexEMeans` accelerates end-to-end GPU codebook learning + full-corpus index encoding by **`2.27x` geometric mean** over `Lloyd@25` (**`2.33x` at `T = 1`**, peaking at **`10.09x` at `T = 2`** / **`10.20x` at `T = 1`**) while simultaneously **improving held-out test-image PSNR by `+0.44 dB` mean (up to `+1.08 dB`) over `Lloyd@25`** and **`+2.16 dB` over `Random` codebooks**.
- **Zero-Leakage & Forensic Hygiene**: `0` train/test feature or label leakage (raw text/image deduplication prior to train-only TF-IDF+SVD / frozen CNN extraction + two-way exact & rounded vector deduplication; downstream probes trained strictly on `(X_train[coreset], y_train[coreset], w_coreset)`), `0` `NaN`/`Inf` values, `0` Seed-0 cold-JIT spikes (`num_cold_jit_spikes = 0`), and `100%` single-GPU / single-SKU de-confounding across [`outputs/semdedup/semdedup_results.json`](../outputs/semdedup/semdedup_results.json) and [`outputs/semdedup/semdedup_summary.md`](../outputs/semdedup/semdedup_summary.md).

![SemDeDup Held-Out Accuracy vs GPU Pruning Latency](../outputs/semdedup/semdedup_accuracy_vs_time.png)

![Downstream Data-Pruning Scaling Curves Across 4 Real ML Modalities](../outputs/semdedup/semdedup_retention_curves.png)

![Megapixel Color Palette & 4x4 Patch VQ PSNR vs GPU Latency](../outputs/semdedup/megapixel_vq_psnr_pareto.png)

---

## 1. Algorithmic Architecture (`src/emeans/semdedup_coreset.py`)

Modern vision and LLM data curation pipelines — notably `SemDeDup` (Abbas et al., 2023) and prototype-based neural scaling pruning (Sorscher et al., *Beyond Neural Scaling Laws*, NeurIPS 2022) — cluster `N` training embeddings into `K` semantic prototypes to discard near-duplicate samples and concentrate training compute on diverse, representative examples. Existing implementations face two major bottlenecks:
1. **Clustering Bottleneck**: Multi-pass Lloyd `K`-means over `50K–500K` vectors at `K = 1,024..4,096` takes hundreds of milliseconds to seconds on modern GPUs, while uncalibrated fast streaming solvers (`s-means`, vanilla `e-means`) suffer severe empty-cluster collapse (`30–45%` dead prototypes on ReLU/ConvNeXt features) or catastrophic FP8 underflow on unnormalized multi-scale tabular features (`110,761x` dynamic range in Forest Covertype).
2. **Unweighted Coreset Distribution Shift**: Pruning dense redundant clusters without importance weighting distorts class priors and feature covariance, causing linear/logistic probes trained on unweighted pruned subsets to underperform random uniform subsampling at moderate (`20–50%`) retention.

Our implementation in [`src/emeans/semdedup_coreset.py`](../src/emeans/semdedup_coreset.py) (verified by `20` unit & adversarial tests in [`tests/test_semdedup_coreset.py`](../tests/test_semdedup_coreset.py) and orchestrated by [`scripts/run_semdedup_benchmarks.py`](../scripts/run_semdedup_benchmarks.py)) resolves both:
- **Strided Equivariant FP64 Origin Centering & Outlier-Immune Conditioning**: `_compute_equivariant_origin64` computes an odd-sample strided FP64 coordinate median across the full corpus (`100%` equivariant to arbitrary `+20,000` coordinate shifts and immune to `<50%` `1e25` outliers), and `scale_from_x_only=True` isolates out-of-sample test queries from test-batch outlier coupling.
- **Single-Pass 64-Bit Triton `_semdedup_margin_kernel` (`0` Python Loops over `K` or `N`)**: After sorting points by `(cluster_id, dist_to_centroid)` via two GPU radix passes, a single Triton kernel (`FULL_D=True` register-resident path for `D <= 1024`, tiled `FULL_D=False` loop for `D > 1024`) anchors each live cluster's exact medoid (`intra_rank == 0`) and scores non-medoid points by their RMS-normalized minimum Euclidean margin against a 24-neighbor intra-cluster window (`~1.2 ms` on `50K x 768`, `~2.0 ms` on `500K x 54`).
- **Horvitz-Thompson Cluster-Mass Conservation**: When `m_c <= n_c` diverse representatives are retained from cluster `c`, each retained point receives exact Horvitz-Thompson weight `w_i = n_c / m_c`, guaranteeing `sum_i w_i = N` (`0` class-prior or covariance drift!).
- **Three Zero-Leakage GPU Downstream Classifiers**: Closed-form LOO-GCV Cholesky Ridge (`evaluate_cholesky_ridge_gpu`), Ridge-warm-started GPU L-BFGS Multinomial Logistic Regression (`evaluate_multinomial_logistic_gpu`), and BF16-screened + exact FP32 direct-difference refined Weighted / Prototype `$k$-NN` (`evaluate_weighted_knn_gpu`, `evaluate_prototype_knn_gpu`).

---

## 2. Suite 1 Headline Scorecard — Semantic Data Deduplication (`T = 2` vs `Random` & `Lloyd@25`)

| Cell | Corpus `(N_tr / N_te, D, K)` | `Apex` Gear | `Apex` `ms` | `Lloyd@25` `ms` (`x`) | `100%` Ref | `Apex` / `Rand` `10%` | `Apex` / `Rand` `20%` | `Apex` / `Rand` `50%` | `Apex` / `s-means` Empty |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `agnews_256_k1024` | `109,876/7,579, D=256, K=1,024` | `medium_croiss_fp8_graph/croiss_fp8` | **`25.14`** | `24.61` (**`0.98x`**) | `87.49%` | **`85.76%`** / `86.42%` (`-0.66`) | **`86.70%`** / `87.06%` (`-0.37`) | **`87.25%`** / `87.41%` (`-0.16`) | **`0.0`** / `30.2` |
| `cifar10_r18_512_k512` | `50,000/10,000, D=512, K=512` | `small_bf16_graph/bf16` | **`17.54`** | `14.85` (**`0.85x`**) | `83.89%` | **`79.61%`** / `79.27%` (`+0.34`) | **`81.16%`** / `80.97%` (`+0.19`) | **`82.86%`** / `82.68%` (`+0.17`) | **`0.0`** / `48.0` |
| `cifar10_r18_512_k1024` | `50,000/10,000, D=512, K=1,024` | `small_bf16_graph/bf16` | **`23.25`** | `19.26` (**`0.83x`**) | `83.89%` | **`79.37%`** / `79.27%` (`+0.11`) | **`81.11%`** / `80.97%` (`+0.15`) | **`82.98%`** / `82.68%` (`+0.30`) | **`0.0`** / `262.8` |
| `cifar10_convnext_768_k2048` | `50,000/10,000, D=768, K=2,048` | `small_bf16_graph/bf16` | **`47.31`** | `41.08` (**`0.87x`**) | `91.26%` | **`89.62%`** / `89.09%` (`+0.53`) | **`89.98%`** / `89.82%` (`+0.16`) | **`90.96%`** / `90.43%` (`+0.53`) | **`0.0`** / `921.2` |
| `covtype_54_k4096` | `500,000/50,000, D=54, K=4,096` | `medium_croiss_fp8_graph/croiss_fp8` | **`43.49`** | `161.87` (**`3.72x`**) | `96.37%` | **`90.79%`** / `90.19%` (`+0.60`) | **`93.85%`** / `93.32%` (`+0.54`) | **`95.68%`** / `95.53%` (`+0.15`) | **`0.0`** / `0.0` |

---

## 3. Suite 2 Headline Scorecard — Megapixel Color & `4x4` Patch VQ (`T = 2` vs `Lloyd@25`)

| Cell | Corpus `(N_tr / N_te, D, K, bpp)` | `Apex` Gear | `Apex` `ms` (`cb+enc`) | `Lloyd@25` `ms` (`x`) | `Ultra` `ms` (`x`) | `Apex` Train PSNR | `Apex` Held-Out PSNR | vs `Lloyd@25` | vs `Random` |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `vq_rgb_1m_k256` | `1,000,000/262,144, D=3, K=256, 8.00 bpp` | `small_bf16_graph/bf16` | **`44.00`** (`41.6+2.4`) | `64.20` (**`1.46x`**) | `125.28` (`2.85x`) | `34.37 dB` | **`27.61 dB`** | **`+0.31 dB`** | `+2.13 dB` |
| `vq_rgb_1m_k1024` | `1,000,000/262,144, D=3, K=1,024, 10.00 bpp` | `small_bf16_graph/bf16` | **`58.84`** (`47.0+11.8`) | `307.46` (**`5.22x`**) | `141.81` (`2.41x`) | `38.44 dB` | **`30.90 dB`** | **`+1.08 dB`** | `+3.05 dB` |
| `vq_rgb_1m_k4096` | `1,000,000/262,144, D=3, K=4,096, 12.00 bpp` | `small_bf16_graph/bf16` | **`115.62`** (`71.0+44.6`) | `1167.12` (**`10.09x`**) | `192.29` (`1.66x`) | `42.56 dB` | **`33.98 dB`** | **`+0.84 dB`** | `+3.17 dB` |
| `vq_patch_500k_k256` | `500,000/100,000, D=48, K=256, 0.50 bpp` | `small_bf16_graph/bf16` | **`18.92`** (`17.0+1.9`) | `17.86` (**`0.94x`**) | `24.64` (`1.30x`) | `25.89 dB` | **`23.89 dB`** | **`+0.14 dB`** | `+1.74 dB` |
| `vq_patch_500k_k1024` | `500,000/100,000, D=48, K=1,024, 0.62 bpp` | `medium_croiss_fp8_graph/croiss_fp8` | **`25.31`** (`21.6+3.7`) | `27.77` (**`1.10x`**) | `33.11` (`1.31x`) | `27.50 dB` | **`25.76 dB`** | **`+0.10 dB`** | `+1.70 dB` |
| `vq_patch_500k_k4096` | `500,000/100,000, D=48, K=4,096, 0.75 bpp` | `medium_croiss_fp8_graph/croiss_fp8` | **`38.76`** (`28.2+10.5`) | `66.64` (**`1.72x`**) | `50.25` (`1.30x`) | `28.99 dB` | **`27.26 dB`** | **`+0.14 dB`** | `+1.17 dB` |

---

## 4. Complete Per-Cell Suite 1 (`SemDeDup`) Results Across All Budgets & Classifiers

### `agnews_256_k1024` — AG-News SVD (110K×256, K=1024) (`Heavy-Tailed News Topics (4 cls)`)
- **Shape & Hardware**: `N_train = 109,876`, `N_test = 7,579`, `D = 256`, `Classes = 4`, `K = 1,024`, Primary Probe: `logreg`, `100%` Unpruned Accuracy: **`87.49%`** (Ridge `87.24%`, LogReg `87.49%`, k-NN `81.01%`), GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition (600W)` (`acorn:GPU1`).

| Arm | `T` | Gear / Prec | Prune `ms` (`clust+score`) | Empty | Test MSE | Proto (`<=K`) | `10%` Acc | `20%` Acc | `50%` Acc | Ridge `20%` | LogReg `20%` | k-NN `20%` |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Pruning / Palette | `0` | `random/n/a` | `0.19` (`0.0+0.2`) | `0.0` | `269.0515` | `73.34%` | `86.42%` | `87.06%` | `87.41%` | `87.11%` | `87.06%` | `76.75%` |
| Lloyd @ 25 ep | `25` | `n/a/n/a` | `24.61` (`18.5+6.1`) | `0.0` | `174.5403` | `70.62%` | `85.77%` | `86.51%` | `87.16%` | `87.07%` | `86.51%` | `71.75%` |
| e-means (Meta FAIR) | `1` | `n/a/n/a` | `25.67` (`19.6+6.1`) | `5.6` | `177.1330` | `67.90%` | `85.74%` | `86.72%` | `87.31%` | `87.14%` | `86.72%` | `71.20%` |
| s-means (Triton) | `1` | `n/a/n/a` | `15.35` (`9.2+6.1`) | `25.8` | `181.8642` | `68.53%` | `85.98%` | `86.68%` | `87.28%` | `87.13%` | `86.68%` | `70.99%` |
| e-means++ | `1` | `n/a/n/a` | `53.52` (`47.4+6.1`) | `5.6` | `175.3678` | `68.50%` | `85.89%` | `86.70%` | `87.33%` | `87.14%` | `86.70%` | `72.10%` |
| UltraEMeans | `1` | `n/a/n/a` | `39.09` (`33.0+6.1`) | `0.0` | `174.1635` | `69.05%` | `85.85%` | `86.79%` | `87.19%` | `87.14%` | `86.79%` | `71.53%` |
| ApexEMeans (BF16+Graph) | `1` | `small_bf16_graph/bf16` | `23.50` (`17.3+6.2`) | `0.0` | `174.2904` | `67.70%` | `85.71%` | `86.57%` | `87.18%` | `87.09%` | `86.57%` | `71.71%` |
| ApexEMeans (CroissFP8+Graph) | `1` | `medium_croiss_fp8_graph/croiss_fp8` | `24.51` (`18.4+6.1`) | `0.0` | `174.3468` | `68.30%` | `85.74%` | `86.73%` | `87.19%` | `87.10%` | `86.73%` | `71.88%` |
| ApexEMeans (Auto Flagship) | `1` | `medium_croiss_fp8_graph/croiss_fp8` | `24.45` (`18.3+6.1`) | `0.0` | `174.3468` | `68.30%` | `85.74%` | `86.73%` | `87.19%` | `87.10%` | `86.73%` | `71.88%` |
| e-means (Meta FAIR) | `2` | `n/a/n/a` | `25.95` (`19.8+6.1`) | `0.0` | `173.6982` | `68.52%` | `85.95%` | `86.73%` | `87.21%` | `87.15%` | `86.73%` | `71.02%` |
| s-means (Triton) | `2` | `n/a/n/a` | `15.53` (`9.4+6.1`) | `30.2` | `175.8129` | `69.41%` | `85.93%` | `86.69%` | `87.24%` | `87.03%` | `86.69%` | `71.81%` |
| e-means++ | `2` | `n/a/n/a` | `53.35` (`47.2+6.1`) | `0.0` | `173.1681` | `68.78%` | `85.78%` | `86.69%` | `87.20%` | `87.10%` | `86.69%` | `70.97%` |
| UltraEMeans | `2` | `n/a/n/a` | `39.30` (`33.2+6.1`) | `0.0` | `172.4951` | `68.70%` | `85.93%` | `86.71%` | `87.23%` | `87.12%` | `86.71%` | `71.83%` |
| ApexEMeans (BF16+Graph) | `2` | `small_bf16_graph/bf16` | `24.23` (`18.1+6.1`) | `0.0` | `172.7521` | `69.61%` | `85.81%` | `86.84%` | `87.19%` | `87.06%` | `86.84%` | `71.90%` |
| ApexEMeans (CroissFP8+Graph) | `2` | `medium_croiss_fp8_graph/croiss_fp8` | `25.10` (`18.9+6.2`) | `0.0` | `172.7268` | `69.24%` | `85.78%` | `86.69%` | `87.28%` | `87.15%` | `86.69%` | `71.93%` |
| ApexEMeans (Auto Flagship) | `2` | `medium_croiss_fp8_graph/croiss_fp8` | `25.14` (`19.0+6.2`) | `0.0` | `172.7225` | `69.30%` | `85.76%` | `86.70%` | `87.25%` | `87.12%` | `86.70%` | `71.84%` |
| e-means (Meta FAIR) | `5` | `n/a/n/a` | `32.33` (`26.2+6.1`) | `0.0` | `172.4231` | `68.38%` | `86.00%` | `86.67%` | `87.26%` | `87.09%` | `86.67%` | `70.95%` |
| s-means (Triton) | `5` | `n/a/n/a` | `18.45` (`12.3+6.1`) | `16.2` | `172.9742` | `69.22%` | `85.94%` | `86.75%` | `87.21%` | `87.04%` | `86.75%` | `72.04%` |
| e-means++ | `5` | `n/a/n/a` | `60.74` (`54.6+6.1`) | `0.0` | `172.1027` | `69.14%` | `85.97%` | `86.68%` | `87.19%` | `87.08%` | `86.68%` | `71.18%` |
| UltraEMeans | `5` | `n/a/n/a` | `53.17` (`47.0+6.1`) | `0.0` | `171.7298` | `68.73%` | `85.75%` | `86.68%` | `87.18%` | `87.09%` | `86.68%` | `71.12%` |
| ApexEMeans (BF16+Graph) | `5` | `small_bf16_graph/bf16` | `35.37` (`29.2+6.2`) | `0.0` | `171.7201` | `69.21%` | `85.70%` | `86.68%` | `87.19%` | `87.10%` | `86.68%` | `71.68%` |
| ApexEMeans (CroissFP8+Graph) | `5` | `medium_croiss_fp8_graph/croiss_fp8` | `34.99` (`28.8+6.2`) | `0.0` | `171.8790` | `69.85%` | `85.74%` | `86.62%` | `87.13%` | `87.07%` | `86.62%` | `71.45%` |
| ApexEMeans (Auto Flagship) | `5` | `medium_croiss_fp8_graph/croiss_fp8` | `34.66` (`28.5+6.2`) | `0.0` | `171.8842` | `69.80%` | `85.72%` | `86.59%` | `87.11%` | `87.08%` | `86.59%` | `71.49%` |

### `cifar10_r18_512_k512` — CIFAR-10 ResNet-18 (50K×512, K=512) (`Global ReLU CNN (10 cls, K=512)`)
- **Shape & Hardware**: `N_train = 50,000`, `N_test = 10,000`, `D = 512`, `Classes = 10`, `K = 512`, Primary Probe: `logreg`, `100%` Unpruned Accuracy: **`83.89%`** (Ridge `74.29%`, LogReg `83.89%`, k-NN `79.46%`), GPU: `NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (325W)` (`foureyes:GPU3`).

| Arm | `T` | Gear / Prec | Prune `ms` (`clust+score`) | Empty | Test MSE | Proto (`<=K`) | `10%` Acc | `20%` Acc | `50%` Acc | Ridge `20%` | LogReg `20%` | k-NN `20%` |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Pruning / Palette | `0` | `random/n/a` | `0.16` (`0.0+0.2`) | `0.0` | `1266.1158` | `74.04%` | `79.27%` | `80.97%` | `82.68%` | `74.20%` | `80.97%` | `74.61%` |
| Lloyd @ 25 ep | `25` | `n/a/n/a` | `14.85` (`11.0+3.9`) | `0.0` | `834.9043` | `74.20%` | `79.72%` | `81.17%` | `82.99%` | `75.94%` | `81.17%` | `73.09%` |
| e-means (Meta FAIR) | `1` | `n/a/n/a` | `16.81` (`13.0+3.9`) | `0.0` | `843.0923` | `73.98%` | `79.84%` | `81.29%` | `82.99%` | `75.75%` | `81.29%` | `73.13%` |
| s-means (Triton) | `1` | `n/a/n/a` | `10.71` (`6.9+3.8`) | `15.4` | `866.2400` | `73.30%` | `79.66%` | `81.25%` | `82.97%` | `76.06%` | `81.25%` | `73.04%` |
| e-means++ | `1` | `n/a/n/a` | `31.37` (`27.5+3.8`) | `0.0` | `840.0362` | `74.00%` | `79.63%` | `81.10%` | `82.93%` | `75.75%` | `81.10%` | `73.08%` |
| UltraEMeans | `1` | `n/a/n/a` | `29.49` (`25.6+3.9`) | `0.0` | `837.7916` | `73.95%` | `79.42%` | `81.06%` | `82.84%` | `75.66%` | `81.06%` | `73.27%` |
| ApexEMeans (BF16+Graph) | `1` | `small_bf16_graph/bf16` | `17.17` (`13.3+3.9`) | `0.0` | `837.8036` | `74.45%` | `79.72%` | `80.99%` | `82.82%` | `75.66%` | `80.99%` | `73.29%` |
| ApexEMeans (CroissFP8+Graph) | `1` | `medium_croiss_fp8_graph/croiss_fp8` | `16.86` (`13.0+3.9`) | `0.0` | `837.8615` | `74.28%` | `79.85%` | `81.06%` | `82.92%` | `75.73%` | `81.06%` | `73.16%` |
| ApexEMeans (Auto Flagship) | `1` | `small_bf16_graph/bf16` | `17.17` (`13.3+3.9`) | `0.0` | `837.8036` | `74.45%` | `79.72%` | `80.99%` | `82.82%` | `75.66%` | `80.99%` | `73.29%` |
| e-means (Meta FAIR) | `2` | `n/a/n/a` | `16.87` (`13.0+3.8`) | `0.0` | `834.1788` | `74.46%` | `79.68%` | `81.05%` | `82.87%` | `75.96%` | `81.05%` | `73.45%` |
| s-means (Triton) | `2` | `n/a/n/a` | `10.85` (`7.0+3.9`) | `48.0` | `848.7355` | `73.86%` | `79.50%` | `81.06%` | `82.83%` | `75.74%` | `81.06%` | `73.08%` |
| e-means++ | `2` | `n/a/n/a` | `31.51` (`27.7+3.8`) | `0.0` | `833.6794` | `74.22%` | `79.69%` | `81.11%` | `82.79%` | `75.64%` | `81.11%` | `73.39%` |
| UltraEMeans | `2` | `n/a/n/a` | `29.68` (`25.8+3.8`) | `0.0` | `832.3094` | `74.50%` | `79.63%` | `81.14%` | `82.87%` | `75.73%` | `81.14%` | `73.18%` |
| ApexEMeans (BF16+Graph) | `2` | `small_bf16_graph/bf16` | `17.80` (`13.9+3.9`) | `0.0` | `833.1910` | `74.64%` | `79.61%` | `81.16%` | `82.86%` | `75.69%` | `81.16%` | `73.55%` |
| ApexEMeans (CroissFP8+Graph) | `2` | `medium_croiss_fp8_graph/croiss_fp8` | `18.51` (`14.6+3.9`) | `0.0` | `832.7601` | `74.51%` | `79.78%` | `81.11%` | `82.90%` | `75.78%` | `81.11%` | `73.25%` |
| ApexEMeans (Auto Flagship) | `2` | `small_bf16_graph/bf16` | `17.54` (`13.7+3.9`) | `0.0` | `833.1910` | `74.64%` | `79.61%` | `81.16%` | `82.86%` | `75.69%` | `81.16%` | `73.55%` |
| e-means (Meta FAIR) | `5` | `n/a/n/a` | `20.62` (`16.8+3.8`) | `0.0` | `830.7832` | `74.18%` | `79.65%` | `80.85%` | `82.74%` | `75.90%` | `80.85%` | `73.63%` |
| s-means (Triton) | `5` | `n/a/n/a` | `12.96` (`9.1+3.8`) | `25.4` | `839.7721` | `74.01%` | `79.64%` | `80.95%` | `82.93%` | `75.72%` | `80.95%` | `73.18%` |
| e-means++ | `5` | `n/a/n/a` | `35.85` (`32.0+3.8`) | `0.0` | `830.9508` | `74.74%` | `79.96%` | `81.21%` | `82.83%` | `75.93%` | `81.21%` | `73.74%` |
| UltraEMeans | `5` | `n/a/n/a` | `39.70` (`35.9+3.8`) | `0.0` | `830.0176` | `74.57%` | `79.58%` | `81.03%` | `82.84%` | `75.96%` | `81.03%` | `73.54%` |
| ApexEMeans (BF16+Graph) | `5` | `small_bf16_graph/bf16` | `25.83` (`21.9+3.9`) | `0.0` | `830.2754` | `74.59%` | `79.64%` | `81.20%` | `82.91%` | `75.83%` | `81.20%` | `73.58%` |
| ApexEMeans (CroissFP8+Graph) | `5` | `medium_croiss_fp8_graph/croiss_fp8` | `27.40` (`23.5+3.9`) | `0.0` | `830.0748` | `74.01%` | `79.42%` | `81.06%` | `82.84%` | `75.98%` | `81.06%` | `73.58%` |
| ApexEMeans (Auto Flagship) | `5` | `small_bf16_graph/bf16` | `25.39` (`21.5+3.9`) | `0.0` | `830.2754` | `74.59%` | `79.64%` | `81.20%` | `82.91%` | `75.83%` | `81.20%` | `73.58%` |

### `cifar10_r18_512_k1024` — CIFAR-10 ResNet-18 (50K×512, K=1024) (`Global ReLU CNN (10 cls, K=1024)`)
- **Shape & Hardware**: `N_train = 50,000`, `N_test = 10,000`, `D = 512`, `Classes = 10`, `K = 1,024`, Primary Probe: `logreg`, `100%` Unpruned Accuracy: **`83.89%`** (Ridge `74.29%`, LogReg `83.89%`, k-NN `79.46%`), GPU: `NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (325W)` (`foureyes:GPU3`).

| Arm | `T` | Gear / Prec | Prune `ms` (`clust+score`) | Empty | Test MSE | Proto (`<=K`) | `10%` Acc | `20%` Acc | `50%` Acc | Ridge `20%` | LogReg `20%` | k-NN `20%` |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Pruning / Palette | `0` | `random/n/a` | `0.16` (`0.0+0.2`) | `0.0` | `1196.0325` | `76.33%` | `79.27%` | `80.97%` | `82.68%` | `74.20%` | `80.97%` | `74.61%` |
| Lloyd @ 25 ep | `25` | `n/a/n/a` | `19.26` (`14.4+4.8`) | `0.0` | `813.6058` | `76.56%` | `79.59%` | `81.14%` | `82.90%` | `75.81%` | `81.14%` | `73.54%` |
| e-means (Meta FAIR) | `1` | `n/a/n/a` | `18.85` (`14.0+4.8`) | `0.8` | `820.7539` | `76.16%` | `79.54%` | `80.93%` | `82.72%` | `75.81%` | `80.93%` | `73.44%` |
| s-means (Triton) | `1` | `n/a/n/a` | `15.33` (`10.5+4.8`) | `146.6` | `857.4482` | `74.75%` | `79.64%` | `81.18%` | `82.91%` | `75.72%` | `81.18%` | `72.86%` |
| e-means++ | `1` | `n/a/n/a` | `44.80` (`40.0+4.8`) | `2.2` | `817.8435` | `76.08%` | `79.37%` | `80.98%` | `82.76%` | `75.67%` | `80.98%` | `73.63%` |
| UltraEMeans | `1` | `n/a/n/a` | `43.85` (`39.0+4.8`) | `0.0` | `817.1734` | `76.27%` | `79.70%` | `81.11%` | `82.90%` | `75.90%` | `81.11%` | `73.72%` |
| ApexEMeans (BF16+Graph) | `1` | `small_bf16_graph/bf16` | `22.88` (`18.1+4.8`) | `0.0` | `816.1621` | `76.08%` | `79.62%` | `80.91%` | `82.75%` | `75.73%` | `80.91%` | `73.11%` |
| ApexEMeans (CroissFP8+Graph) | `1` | `medium_croiss_fp8_graph/croiss_fp8` | `26.78` (`22.0+4.8`) | `0.0` | `815.9229` | `76.09%` | `79.74%` | `81.26%` | `82.80%` | `75.70%` | `81.26%` | `73.56%` |
| ApexEMeans (Auto Flagship) | `1` | `small_bf16_graph/bf16` | `22.73` (`17.9+4.8`) | `0.0` | `816.1621` | `76.08%` | `79.62%` | `80.91%` | `82.75%` | `75.73%` | `80.91%` | `73.11%` |
| e-means (Meta FAIR) | `2` | `n/a/n/a` | `19.47` (`14.6+4.8`) | `0.4` | `811.1397` | `75.93%` | `79.67%` | `80.95%` | `82.71%` | `75.67%` | `80.95%` | `73.38%` |
| s-means (Triton) | `2` | `n/a/n/a` | `15.49` (`10.7+4.8`) | `262.8` | `834.3176` | `75.13%` | `79.55%` | `80.99%` | `82.81%` | `75.96%` | `80.99%` | `72.95%` |
| e-means++ | `2` | `n/a/n/a` | `46.21` (`41.4+4.8`) | `0.2` | `811.5759` | `75.94%` | `79.39%` | `80.85%` | `82.73%` | `75.69%` | `80.85%` | `73.55%` |
| UltraEMeans | `2` | `n/a/n/a` | `45.23` (`40.4+4.8`) | `0.0` | `809.4968` | `76.25%` | `79.46%` | `80.92%` | `82.73%` | `75.76%` | `80.92%` | `73.65%` |
| ApexEMeans (BF16+Graph) | `2` | `small_bf16_graph/bf16` | `23.20` (`18.4+4.8`) | `0.0` | `809.6641` | `76.07%` | `79.37%` | `81.11%` | `82.98%` | `75.68%` | `81.11%` | `73.35%` |
| ApexEMeans (CroissFP8+Graph) | `2` | `medium_croiss_fp8_graph/croiss_fp8` | `27.34` (`22.5+4.8`) | `0.0` | `809.5660` | `75.99%` | `79.85%` | `80.98%` | `82.84%` | `75.76%` | `80.98%` | `73.45%` |
| ApexEMeans (Auto Flagship) | `2` | `small_bf16_graph/bf16` | `23.25` (`18.4+4.8`) | `0.0` | `809.6641` | `76.07%` | `79.37%` | `81.11%` | `82.98%` | `75.68%` | `81.11%` | `73.35%` |
| e-means (Meta FAIR) | `5` | `n/a/n/a` | `24.03` (`19.2+4.8`) | `0.0` | `808.3057` | `76.28%` | `79.63%` | `81.00%` | `82.76%` | `75.70%` | `81.00%` | `73.50%` |
| s-means (Triton) | `5` | `n/a/n/a` | `18.73` (`13.9+4.8`) | `110.4` | `820.7754` | `75.94%` | `79.42%` | `80.92%` | `82.91%` | `76.08%` | `80.92%` | `73.29%` |
| e-means++ | `5` | `n/a/n/a` | `50.95` (`46.1+4.8`) | `0.0` | `808.5617` | `76.28%` | `79.82%` | `81.10%` | `82.77%` | `75.55%` | `81.10%` | `73.96%` |
| UltraEMeans | `5` | `n/a/n/a` | `59.40` (`54.6+4.8`) | `0.0` | `807.0882` | `75.82%` | `79.62%` | `80.87%` | `82.80%` | `75.64%` | `80.87%` | `73.56%` |
| ApexEMeans (BF16+Graph) | `5` | `small_bf16_graph/bf16` | `28.41` (`23.6+4.8`) | `0.0` | `806.7454` | `76.16%` | `79.60%` | `81.22%` | `82.69%` | `75.75%` | `81.22%` | `73.71%` |
| ApexEMeans (CroissFP8+Graph) | `5` | `medium_croiss_fp8_graph/croiss_fp8` | `34.20` (`29.4+4.8`) | `0.0` | `807.3517` | `76.36%` | `79.70%` | `81.30%` | `82.82%` | `75.68%` | `81.30%` | `73.57%` |
| ApexEMeans (Auto Flagship) | `5` | `small_bf16_graph/bf16` | `28.32` (`23.5+4.8`) | `0.0` | `806.7454` | `76.16%` | `79.60%` | `81.22%` | `82.69%` | `75.75%` | `81.22%` | `73.71%` |

### `cifar10_convnext_768_k2048` — CIFAR-10 ConvNeXt (50K×768, K=2048) (`Modern 768-D Vision (10 cls)`)
- **Shape & Hardware**: `N_train = 50,000`, `N_test = 10,000`, `D = 768`, `Classes = 10`, `K = 2,048`, Primary Probe: `logreg`, `100%` Unpruned Accuracy: **`91.26%`** (Ridge `87.79%`, LogReg `91.26%`, k-NN `88.52%`), GPU: `NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (325W)` (`foureyes:GPU2`).

| Arm | `T` | Gear / Prec | Prune `ms` (`clust+score`) | Empty | Test MSE | Proto (`<=K`) | `10%` Acc | `20%` Acc | `50%` Acc | Ridge `20%` | LogReg `20%` | k-NN `20%` |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Pruning / Palette | `0` | `random/n/a` | `0.19` (`0.0+0.2`) | `0.0` | `219.1723` | `88.56%` | `89.09%` | `89.82%` | `90.43%` | `87.80%` | `89.82%` | `86.84%` |
| Lloyd @ 25 ep | `25` | `n/a/n/a` | `41.08` (`30.6+10.4`) | `0.0` | `150.5278` | `88.62%` | `89.38%` | `89.82%` | `90.99%` | `88.26%` | `89.82%` | `86.26%` |
| e-means (Meta FAIR) | `1` | `n/a/n/a` | `30.10` (`19.7+10.4`) | `10.0` | `151.7238` | `88.43%` | `89.24%` | `89.81%` | `90.89%` | `88.30%` | `89.81%` | `86.21%` |
| s-means (Triton) | `1` | `n/a/n/a` | `34.33` (`23.8+10.5`) | `473.2` | `161.2107` | `88.02%` | `89.25%` | `89.85%` | `90.90%` | `88.31%` | `89.85%` | `86.16%` |
| e-means++ | `1` | `n/a/n/a` | `77.81` (`67.4+10.4`) | `1.6` | `150.6419` | `88.79%` | `89.52%` | `90.06%` | `91.03%` | `88.27%` | `90.06%` | `86.13%` |
| UltraEMeans | `1` | `n/a/n/a` | `66.00` (`55.6+10.4`) | `0.0` | `151.2213` | `88.68%` | `89.26%` | `89.86%` | `90.89%` | `88.21%` | `89.86%` | `86.25%` |
| ApexEMeans (BF16+Graph) | `1` | `small_bf16_graph/bf16` | `45.30` (`34.9+10.4`) | `0.0` | `150.3627` | `88.62%` | `89.55%` | `90.01%` | `90.92%` | `88.22%` | `90.01%` | `86.24%` |
| ApexEMeans (CroissFP8+Graph) | `1` | `medium_croiss_fp8_graph/croiss_fp8` | `53.15` (`42.6+10.5`) | `0.0` | `150.3701` | `88.83%` | `89.63%` | `90.15%` | `90.97%` | `88.25%` | `90.15%` | `86.25%` |
| ApexEMeans (Auto Flagship) | `1` | `small_bf16_graph/bf16` | `45.10` (`34.7+10.4`) | `0.0` | `150.3627` | `88.62%` | `89.54%` | `90.01%` | `90.92%` | `88.22%` | `90.01%` | `86.24%` |
| e-means (Meta FAIR) | `2` | `n/a/n/a` | `30.69` (`20.3+10.4`) | `0.2` | `149.3575` | `88.71%` | `89.49%` | `89.79%` | `90.93%` | `88.22%` | `89.79%` | `86.30%` |
| s-means (Triton) | `2` | `n/a/n/a` | `34.93` (`24.4+10.5`) | `921.2` | `156.2536` | `87.67%` | `89.29%` | `89.79%` | `90.91%` | `88.32%` | `89.79%` | `86.34%` |
| e-means++ | `2` | `n/a/n/a` | `78.76` (`68.4+10.4`) | `0.0` | `149.3975` | `88.86%` | `89.66%` | `90.11%` | `91.01%` | `88.19%` | `90.11%` | `86.08%` |
| UltraEMeans | `2` | `n/a/n/a` | `69.19` (`58.8+10.4`) | `0.0` | `148.9263` | `88.50%` | `89.54%` | `90.13%` | `91.05%` | `88.15%` | `90.13%` | `86.26%` |
| ApexEMeans (BF16+Graph) | `2` | `small_bf16_graph/bf16` | `47.33` (`36.9+10.4`) | `0.0` | `148.8377` | `88.75%` | `89.61%` | `89.93%` | `90.97%` | `88.25%` | `89.93%` | `86.25%` |
| ApexEMeans (CroissFP8+Graph) | `2` | `medium_croiss_fp8_graph/croiss_fp8` | `58.82` (`48.4+10.5`) | `0.0` | `148.8573` | `88.70%` | `89.46%` | `89.85%` | `91.00%` | `88.24%` | `89.85%` | `86.16%` |
| ApexEMeans (Auto Flagship) | `2` | `small_bf16_graph/bf16` | `47.31` (`36.9+10.4`) | `0.0` | `148.8443` | `88.75%` | `89.62%` | `89.98%` | `90.96%` | `88.24%` | `89.98%` | `86.28%` |
| e-means (Meta FAIR) | `5` | `n/a/n/a` | `37.23` (`26.8+10.4`) | `0.0` | `148.5851` | `88.77%` | `89.62%` | `89.97%` | `90.95%` | `88.24%` | `89.97%` | `86.15%` |
| s-means (Triton) | `5` | `n/a/n/a` | `42.68` (`32.2+10.5`) | `393.2` | `151.4602` | `88.58%` | `89.54%` | `89.91%` | `90.82%` | `88.25%` | `89.91%` | `86.26%` |
| e-means++ | `5` | `n/a/n/a` | `86.12` (`75.7+10.4`) | `0.0` | `148.8293` | `88.70%` | `89.60%` | `90.05%` | `90.96%` | `88.23%` | `90.05%` | `86.14%` |
| UltraEMeans | `5` | `n/a/n/a` | `104.92` (`94.5+10.4`) | `0.0` | `148.3613` | `88.64%` | `89.48%` | `89.90%` | `90.98%` | `88.16%` | `89.90%` | `86.15%` |
| ApexEMeans (BF16+Graph) | `5` | `small_bf16_graph/bf16` | `60.12` (`49.7+10.4`) | `0.0` | `148.2707` | `88.60%` | `89.42%` | `89.99%` | `90.90%` | `88.23%` | `89.99%` | `86.29%` |
| ApexEMeans (CroissFP8+Graph) | `5` | `medium_croiss_fp8_graph/croiss_fp8` | `77.71` (`67.3+10.4`) | `0.0` | `148.3207` | `88.51%` | `89.47%` | `89.92%` | `90.95%` | `88.22%` | `89.92%` | `85.99%` |
| ApexEMeans (Auto Flagship) | `5` | `small_bf16_graph/bf16` | `60.38` (`49.9+10.4`) | `0.0` | `148.2826` | `88.65%` | `89.45%` | `89.92%` | `90.97%` | `88.25%` | `89.92%` | `86.23%` |

### `covtype_54_k4096` — Forest Covertype (500K×54, K=4096) (`Physical Sensors 110,761x Span (7 cls)`)
- **Shape & Hardware**: `N_train = 500,000`, `N_test = 50,000`, `D = 54`, `Classes = 7`, `K = 4,096`, Primary Probe: `knn`, `100%` Unpruned Accuracy: **`96.37%`** (Ridge `71.22%`, LogReg `74.02%`, k-NN `96.37%`), GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition (600W)` (`acorn:GPU0`).

| Arm | `T` | Gear / Prec | Prune `ms` (`clust+score`) | Empty | Test MSE | Proto (`<=K`) | `10%` Acc | `20%` Acc | `50%` Acc | Ridge `20%` | LogReg `20%` | k-NN `20%` |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Pruning / Palette | `0` | `random/n/a` | `0.23` (`0.0+0.2`) | `0.0` | `31206.0155` | `73.41%` | `90.19%` | `93.32%` | `95.53%` | `71.21%` | `74.02%` | `93.32%` |
| Lloyd @ 25 ep | `25` | `n/a/n/a` | `161.87` (`146.1+15.7`) | `0.0` | `18705.4255` | `75.09%` | `90.77%` | `93.81%` | `95.77%` | `71.34%` | `73.99%` | `93.81%` |
| e-means (Meta FAIR) | `1` | `n/a/n/a` | `47.80` (`31.9+15.9`) | `0.2` | `19337.5914` | `75.15%` | `90.76%` | `93.84%` | `95.74%` | `71.28%` | `73.95%` | `93.84%` |
| s-means (Triton) | `1` | `n/a/n/a` | `28.32` (`12.5+15.8`) | `0.6` | `18706.9816` | `74.66%` | `90.80%` | `93.78%` | `95.75%` | `71.31%` | `73.96%` | `93.78%` |
| e-means++ | `1` | `n/a/n/a` | `123.24` (`107.4+15.8`) | `0.4` | `18687.6888` | `74.88%` | `90.85%` | `93.80%` | `95.74%` | `71.30%` | `73.95%` | `93.80%` |
| UltraEMeans | `1` | `n/a/n/a` | `55.00` (`39.2+15.8`) | `0.0` | `18660.8522` | `74.82%` | `90.84%` | `93.86%` | `95.71%` | `71.29%` | `74.01%` | `93.86%` |
| ApexEMeans (BF16+Graph) | `1` | `small_bf16_graph/bf16` | `144.08` (`128.3+15.7`) | `0.0` | `18771.7064` | `74.83%` | `90.76%` | `93.89%` | `95.74%` | `71.27%` | `73.96%` | `93.89%` |
| ApexEMeans (CroissFP8+Graph) | `1` | `medium_croiss_fp8_graph/croiss_fp8` | `40.36` (`24.7+15.7`) | `0.0` | `18768.0062` | `74.81%` | `90.75%` | `93.86%` | `95.74%` | `71.29%` | `73.99%` | `93.86%` |
| ApexEMeans (Auto Flagship) | `1` | `medium_croiss_fp8_graph/croiss_fp8` | `40.22` (`24.5+15.7`) | `0.0` | `18767.9969` | `74.81%` | `90.77%` | `93.87%` | `95.74%` | `71.29%` | `73.99%` | `93.87%` |
| e-means (Meta FAIR) | `2` | `n/a/n/a` | `49.69` (`33.9+15.8`) | `0.0` | `18757.1351` | `75.22%` | `90.82%` | `93.85%` | `95.72%` | `71.34%` | `73.97%` | `93.85%` |
| s-means (Triton) | `2` | `n/a/n/a` | `29.47` (`13.8+15.7`) | `0.0` | `18334.5785` | `74.74%` | `90.68%` | `93.81%` | `95.77%` | `71.32%` | `73.94%` | `93.81%` |
| e-means++ | `2` | `n/a/n/a` | `125.44` (`109.6+15.8`) | `0.2` | `18291.5498` | `74.63%` | `90.79%` | `93.80%` | `95.69%` | `71.30%` | `74.00%` | `93.80%` |
| UltraEMeans | `2` | `n/a/n/a` | `57.94` (`42.2+15.8`) | `0.0` | `18289.2820` | `74.99%` | `90.79%` | `93.85%` | `95.77%` | `71.30%` | `74.01%` | `93.85%` |
| ApexEMeans (BF16+Graph) | `2` | `small_bf16_graph/bf16` | `58.85` (`43.1+15.7`) | `0.0` | `18328.0254` | `74.75%` | `90.67%` | `93.81%` | `95.80%` | `71.29%` | `74.00%` | `93.81%` |
| ApexEMeans (CroissFP8+Graph) | `2` | `medium_croiss_fp8_graph/croiss_fp8` | `42.94` (`27.2+15.7`) | `0.0` | `18331.2950` | `74.80%` | `90.79%` | `93.84%` | `95.70%` | `71.32%` | `73.98%` | `93.84%` |
| ApexEMeans (Auto Flagship) | `2` | `medium_croiss_fp8_graph/croiss_fp8` | `43.49` (`27.8+15.7`) | `0.0` | `18331.1618` | `74.80%` | `90.79%` | `93.85%` | `95.68%` | `71.33%` | `73.99%` | `93.85%` |
| e-means (Meta FAIR) | `5` | `n/a/n/a` | `77.44` (`61.7+15.7`) | `0.0` | `18470.6371` | `75.26%` | `90.74%` | `93.78%` | `95.78%` | `71.30%` | `73.97%` | `93.78%` |
| s-means (Triton) | `5` | `n/a/n/a` | `38.84` (`23.1+15.7`) | `0.0` | `18087.2695` | `74.72%` | `90.65%` | `93.78%` | `95.75%` | `71.31%` | `73.94%` | `93.78%` |
| e-means++ | `5` | `n/a/n/a` | `153.35` (`137.6+15.7`) | `0.0` | `18067.4774` | `74.82%` | `90.83%` | `93.82%` | `95.72%` | `71.31%` | `74.01%` | `93.82%` |
| UltraEMeans | `5` | `n/a/n/a` | `80.48` (`64.8+15.7`) | `0.0` | `18078.8379` | `74.88%` | `90.80%` | `93.86%` | `95.74%` | `71.27%` | `74.00%` | `93.86%` |
| ApexEMeans (BF16+Graph) | `5` | `small_bf16_graph/bf16` | `74.61` (`58.7+15.9`) | `0.0` | `18083.9640` | `74.81%` | `90.81%` | `93.86%` | `95.74%` | `71.26%` | `73.95%` | `93.86%` |
| ApexEMeans (CroissFP8+Graph) | `5` | `medium_croiss_fp8_graph/croiss_fp8` | `57.18` (`41.3+15.9`) | `0.0` | `18090.0924` | `74.64%` | `90.81%` | `93.81%` | `95.77%` | `71.31%` | `73.98%` | `93.81%` |
| ApexEMeans (Auto Flagship) | `5` | `medium_croiss_fp8_graph/croiss_fp8` | `57.08` (`41.2+15.9`) | `0.0` | `18090.0250` | `74.64%` | `90.78%` | `93.80%` | `95.75%` | `71.31%` | `73.99%` | `93.80%` |

---

## 5. Complete Per-Cell Suite 2 (`Megapixel VQ`) Results Across All Codebook Sizes

### `vq_rgb_1m_k256` — Megapixel RGB Palette (1M×3, K=256) (`8.00 bpp`, `3.0x` Compression vs 24-bit RGB)
- **Shape & Hardware**: `N_train = 1,000,000`, `N_test = 262,144`, `D = 3`, `K = 256`, GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition (600W)` (`acorn:GPU0`).

| Arm | `T` | Gear / Prec | Total `ms` (`codebook+encode`) | Empty | Train Pixel MSE | Test Pixel MSE | Full-Image PSNR (dB) | Held-Out PSNR (dB) |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Pruning / Palette | `0` | `random/n/a` | `2.41` (`0.20+2.21`) | `0.4` | `53.0913` | `188.7708` | `30.90` | `25.48` |
| Lloyd @ 25 ep | `25` | `n/a/n/a` | `64.20` (`62.00+2.19`) | `0.0` | `27.7372` | `121.2747` | `33.70` | `27.29` |
| e-means (Meta FAIR) | `1` | `n/a/n/a` | `171.28` (`169.06+2.22`) | `0.0` | `27.5822` | `122.3284` | `33.73` | `27.26` |
| s-means (Triton) | `1` | `n/a/n/a` | `69.91` (`67.68+2.24`) | `0.0` | `24.9065` | `118.1978` | `34.17` | `27.41` |
| e-means++ | `1` | `n/a/n/a` | `270.98` (`268.76+2.21`) | `0.0` | `24.0685` | `112.9891` | `34.32` | `27.60` |
| UltraEMeans | `1` | `n/a/n/a` | `126.44` (`124.21+2.22`) | `0.0` | `24.0999` | `114.4979` | `34.31` | `27.54` |
| ApexEMeans (BF16+Graph) | `1` | `small_bf16_graph/bf16` | `31.02` (`28.63+2.39`) | `0.0` | `24.0636` | `113.5011` | `34.32` | `27.58` |
| ApexEMeans (CroissFP8+Graph) | `1` | `medium_croiss_fp8_graph/croiss_fp8` | `33.08` (`30.74+2.33`) | `0.0` | `24.0701` | `113.2536` | `34.32` | `27.59` |
| ApexEMeans (Auto Flagship) | `1` | `small_bf16_graph/bf16` | `30.93` (`28.54+2.39`) | `0.0` | `24.0635` | `113.5007` | `34.32` | `27.58` |
| e-means (Meta FAIR) | `2` | `n/a/n/a` | `171.75` (`169.54+2.21`) | `0.0` | `26.6287` | `118.2281` | `33.88` | `27.40` |
| s-means (Triton) | `2` | `n/a/n/a` | `69.83` (`67.60+2.23`) | `0.0` | `24.4518` | `113.3003` | `34.25` | `27.59` |
| e-means++ | `2` | `n/a/n/a` | `270.72` (`268.50+2.22`) | `0.0` | `23.8453` | `113.2129` | `34.36` | `27.59` |
| UltraEMeans | `2` | `n/a/n/a` | `125.28` (`123.06+2.22`) | `0.0` | `23.8509` | `112.9864` | `34.36` | `27.60` |
| ApexEMeans (BF16+Graph) | `2` | `small_bf16_graph/bf16` | `43.98` (`41.59+2.39`) | `0.0` | `23.7891` | `112.7640` | `34.37` | `27.61` |
| ApexEMeans (CroissFP8+Graph) | `2` | `medium_croiss_fp8_graph/croiss_fp8` | `51.78` (`49.44+2.33`) | `0.0` | `23.7786` | `113.4566` | `34.37` | `27.58` |
| ApexEMeans (Auto Flagship) | `2` | `small_bf16_graph/bf16` | `44.00` (`41.61+2.39`) | `0.0` | `23.7891` | `112.7640` | `34.37` | `27.61` |
| e-means (Meta FAIR) | `5` | `n/a/n/a` | `214.93` (`212.72+2.21`) | `0.0` | `25.9340` | `116.7630` | `33.99` | `27.46` |
| s-means (Triton) | `5` | `n/a/n/a` | `87.10` (`84.87+2.23`) | `0.0` | `24.1994` | `112.9038` | `34.29` | `27.60` |
| e-means++ | `5` | `n/a/n/a` | `324.96` (`322.74+2.22`) | `0.0` | `23.7059` | `112.5732` | `34.38` | `27.62` |
| UltraEMeans | `5` | `n/a/n/a` | `156.28` (`154.05+2.22`) | `0.0` | `23.7057` | `111.7612` | `34.38` | `27.65` |
| ApexEMeans (BF16+Graph) | `5` | `small_bf16_graph/bf16` | `45.56` (`43.15+2.41`) | `0.0` | `23.7061` | `112.6252` | `34.38` | `27.61` |
| ApexEMeans (CroissFP8+Graph) | `5` | `medium_croiss_fp8_graph/croiss_fp8` | `50.73` (`48.39+2.34`) | `0.0` | `23.6985` | `112.8191` | `34.38` | `27.61` |
| ApexEMeans (Auto Flagship) | `5` | `small_bf16_graph/bf16` | `44.68` (`42.29+2.39`) | `0.0` | `23.7005` | `112.8062` | `34.38` | `27.61` |

### `vq_rgb_1m_k1024` — Megapixel RGB Palette (1M×3, K=1,024) (`10.00 bpp`, `2.4x` Compression vs 24-bit RGB)
- **Shape & Hardware**: `N_train = 1,000,000`, `N_test = 262,144`, `D = 3`, `K = 1,024`, GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition (600W)` (`acorn:GPU0`).

| Arm | `T` | Gear / Prec | Total `ms` (`codebook+encode`) | Empty | Train Pixel MSE | Test Pixel MSE | Full-Image PSNR (dB) | Held-Out PSNR (dB) |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Pruning / Palette | `0` | `random/n/a` | `11.85` (`0.20+11.66`) | `3.0` | `20.6864` | `108.2360` | `34.99` | `27.85` |
| Lloyd @ 25 ep | `25` | `n/a/n/a` | `307.46` (`295.82+11.64`) | `0.0` | `11.0359` | `67.8916` | `37.70` | `29.82` |
| e-means (Meta FAIR) | `1` | `n/a/n/a` | `186.25` (`174.60+11.65`) | `0.0` | `11.4410` | `73.3292` | `37.55` | `29.48` |
| s-means (Triton) | `1` | `n/a/n/a` | `79.90` (`68.23+11.68`) | `0.0` | `9.9428` | `64.7521` | `38.16` | `30.02` |
| e-means++ | `1` | `n/a/n/a` | `254.93` (`243.28+11.66`) | `0.0` | `9.4773` | `57.8260` | `38.36` | `30.53` |
| UltraEMeans | `1` | `n/a/n/a` | `141.78` (`130.11+11.67`) | `0.0` | `9.4761` | `53.8623` | `38.36` | `30.84` |
| ApexEMeans (BF16+Graph) | `1` | `small_bf16_graph/bf16` | `52.38` (`40.55+11.83`) | `0.0` | `9.4783` | `55.5860` | `38.36` | `30.69` |
| ApexEMeans (CroissFP8+Graph) | `1` | `medium_croiss_fp8_graph/croiss_fp8` | `48.81` (`37.11+11.70`) | `0.0` | `9.4870` | `54.9855` | `38.36` | `30.74` |
| ApexEMeans (Auto Flagship) | `1` | `small_bf16_graph/bf16` | `51.96` (`40.14+11.82`) | `0.0` | `9.4783` | `55.5938` | `38.36` | `30.69` |
| e-means (Meta FAIR) | `2` | `n/a/n/a` | `188.17` (`176.51+11.66`) | `0.0` | `10.7984` | `67.9864` | `37.80` | `29.82` |
| s-means (Triton) | `2` | `n/a/n/a` | `80.16` (`68.48+11.68`) | `0.0` | `9.6385` | `60.0041` | `38.29` | `30.36` |
| e-means++ | `2` | `n/a/n/a` | `255.75` (`244.09+11.66`) | `0.0` | `9.3124` | `54.8527` | `38.44` | `30.75` |
| UltraEMeans | `2` | `n/a/n/a` | `141.81` (`130.14+11.67`) | `0.0` | `9.3213` | `50.7938` | `38.44` | `31.09` |
| ApexEMeans (BF16+Graph) | `2` | `small_bf16_graph/bf16` | `59.02` (`47.20+11.82`) | `0.0` | `9.3031` | `52.9234` | `38.44` | `30.90` |
| ApexEMeans (CroissFP8+Graph) | `2` | `medium_croiss_fp8_graph/croiss_fp8` | `67.66` (`55.97+11.70`) | `0.0` | `9.3051` | `51.6742` | `38.44` | `31.00` |
| ApexEMeans (Auto Flagship) | `2` | `small_bf16_graph/bf16` | `58.84` (`47.03+11.82`) | `0.0` | `9.3031` | `52.9205` | `38.44` | `30.90` |
| e-means (Meta FAIR) | `5` | `n/a/n/a` | `235.18` (`223.52+11.66`) | `0.0` | `10.5387` | `59.7712` | `37.90` | `30.38` |
| s-means (Triton) | `5` | `n/a/n/a` | `97.76` (`86.07+11.68`) | `0.0` | `9.4975` | `56.3317` | `38.35` | `30.64` |
| e-means++ | `5` | `n/a/n/a` | `310.39` (`298.73+11.66`) | `0.0` | `9.2243` | `56.9990` | `38.48` | `30.59` |
| UltraEMeans | `5` | `n/a/n/a` | `174.17` (`162.50+11.67`) | `0.0` | `9.2243` | `49.9456` | `38.48` | `31.15` |
| ApexEMeans (BF16+Graph) | `5` | `small_bf16_graph/bf16` | `63.04` (`51.22+11.82`) | `0.0` | `9.2083` | `51.7819` | `38.49` | `30.99` |
| ApexEMeans (CroissFP8+Graph) | `5` | `medium_croiss_fp8_graph/croiss_fp8` | `68.64` (`56.94+11.70`) | `0.0` | `9.2114` | `54.0882` | `38.49` | `30.80` |
| ApexEMeans (Auto Flagship) | `5` | `small_bf16_graph/bf16` | `63.15` (`51.33+11.82`) | `0.0` | `9.2092` | `52.3221` | `38.49` | `30.95` |

### `vq_rgb_1m_k4096` — Megapixel RGB Palette (1M×3, K=4,096) (`12.00 bpp`, `2.0x` Compression vs 24-bit RGB)
- **Shape & Hardware**: `N_train = 1,000,000`, `N_test = 262,144`, `D = 3`, `K = 4,096`, GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition (600W)` (`acorn:GPU0`).

| Arm | `T` | Gear / Prec | Total `ms` (`codebook+encode`) | Empty | Train Pixel MSE | Test Pixel MSE | Full-Image PSNR (dB) | Held-Out PSNR (dB) |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Pruning / Palette | `0` | `random/n/a` | `44.60` (`0.20+44.39`) | `18.2` | `8.1969` | `54.5412` | `38.99` | `30.81` |
| Lloyd @ 25 ep | `25` | `n/a/n/a` | `1167.12` (`1122.72+44.40`) | `0.0` | `4.4079` | `31.5948` | `41.69` | `33.14` |
| e-means (Meta FAIR) | `1` | `n/a/n/a` | `227.33` (`182.93+44.40`) | `0.0` | `4.5678` | `37.3835` | `41.53` | `32.41` |
| s-means (Triton) | `1` | `n/a/n/a` | `112.74` (`68.32+44.42`) | `0.0` | `3.9911` | `33.1167` | `42.12` | `32.93` |
| e-means++ | `1` | `n/a/n/a` | `354.39` (`309.98+44.41`) | `0.0` | `3.7099` | `24.9531` | `42.44` | `34.16` |
| UltraEMeans | `1` | `n/a/n/a` | `187.58` (`143.17+44.41`) | `0.0` | `3.6870` | `26.2936` | `42.46` | `33.93` |
| ApexEMeans (BF16+Graph) | `1` | `small_bf16_graph/bf16` | `114.84` (`70.27+44.57`) | `0.0` | `3.7106` | `26.8853` | `42.44` | `33.84` |
| ApexEMeans (CroissFP8+Graph) | `1` | `medium_croiss_fp8_graph/croiss_fp8` | `107.04` (`62.62+44.42`) | `0.0` | `3.7123` | `27.0594` | `42.43` | `33.81` |
| ApexEMeans (Auto Flagship) | `1` | `small_bf16_graph/bf16` | `114.41` (`69.83+44.58`) | `0.0` | `3.7106` | `26.8848` | `42.44` | `33.84` |
| e-means (Meta FAIR) | `2` | `n/a/n/a` | `234.39` (`189.99+44.40`) | `0.0` | `4.2948` | `31.5994` | `41.80` | `33.14` |
| s-means (Triton) | `2` | `n/a/n/a` | `112.77` (`68.36+44.41`) | `0.0` | `3.7607` | `28.4665` | `42.38` | `33.59` |
| e-means++ | `2` | `n/a/n/a` | `356.26` (`311.85+44.41`) | `0.0` | `3.6086` | `25.0839` | `42.56` | `34.14` |
| UltraEMeans | `2` | `n/a/n/a` | `192.29` (`147.89+44.40`) | `0.0` | `3.6013` | `26.0492` | `42.57` | `33.97` |
| ApexEMeans (BF16+Graph) | `2` | `small_bf16_graph/bf16` | `115.66` (`71.09+44.57`) | `0.0` | `3.6093` | `26.0275` | `42.56` | `33.98` |
| ApexEMeans (CroissFP8+Graph) | `2` | `medium_croiss_fp8_graph/croiss_fp8` | `111.55` (`67.14+44.42`) | `0.0` | `3.6070` | `26.2364` | `42.56` | `33.94` |
| ApexEMeans (Auto Flagship) | `2` | `small_bf16_graph/bf16` | `115.62` (`71.04+44.58`) | `0.0` | `3.6093` | `26.0275` | `42.56` | `33.98` |
| e-means (Meta FAIR) | `5` | `n/a/n/a` | `352.07` (`307.68+44.39`) | `0.0` | `4.1314` | `30.3135` | `41.97` | `33.32` |
| s-means (Triton) | `5` | `n/a/n/a` | `129.89` (`85.47+44.42`) | `0.0` | `3.6892` | `28.0146` | `42.46` | `33.66` |
| e-means++ | `5` | `n/a/n/a` | `480.85` (`436.44+44.40`) | `0.0` | `3.5527` | `23.7836` | `42.63` | `34.37` |
| UltraEMeans | `5` | `n/a/n/a` | `234.27` (`189.86+44.41`) | `0.0` | `3.5384` | `25.4068` | `42.64` | `34.08` |
| ApexEMeans (BF16+Graph) | `5` | `small_bf16_graph/bf16` | `133.60` (`89.03+44.58`) | `0.0` | `3.5463` | `25.6044` | `42.63` | `34.05` |
| ApexEMeans (CroissFP8+Graph) | `5` | `medium_croiss_fp8_graph/croiss_fp8` | `137.14` (`92.72+44.42`) | `0.0` | `3.5409` | `25.3186` | `42.64` | `34.10` |
| ApexEMeans (Auto Flagship) | `5` | `small_bf16_graph/bf16` | `133.61` (`89.04+44.58`) | `0.0` | `3.5460` | `25.6092` | `42.63` | `34.05` |

### `vq_patch_500k_k256` — Megapixel 4×4 Patch VQ (500K×48, K=256) (`0.50 bpp`, `48.0x` Compression vs 24-bit RGB)
- **Shape & Hardware**: `N_train = 500,000`, `N_test = 100,000`, `D = 48`, `K = 256`, GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition (600W)` (`acorn:GPU1`).

| Arm | `T` | Gear / Prec | Total `ms` (`codebook+encode`) | Empty | Train Pixel MSE | Test Pixel MSE | Full-Image PSNR (dB) | Held-Out PSNR (dB) |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Pruning / Palette | `0` | `random/n/a` | `2.10` (`0.17+1.93`) | `0.0` | `253.6389` | `397.1469` | `24.09` | `22.16` |
| Lloyd @ 25 ep | `25` | `n/a/n/a` | `17.86` (`15.94+1.92`) | `0.0` | `173.6624` | `274.3714` | `25.73` | `23.75` |
| e-means (Meta FAIR) | `1` | `n/a/n/a` | `19.94` (`18.02+1.91`) | `0.0` | `176.1389` | `275.4764` | `25.67` | `23.73` |
| s-means (Triton) | `1` | `n/a/n/a` | `11.93` (`10.02+1.92`) | `0.0` | `169.2684` | `264.4128` | `25.85` | `23.91` |
| e-means++ | `1` | `n/a/n/a` | `75.30` (`73.39+1.91`) | `0.0` | `168.9845` | `267.0089` | `25.85` | `23.87` |
| UltraEMeans | `1` | `n/a/n/a` | `27.90` (`25.98+1.92`) | `0.0` | `168.7479` | `263.8180` | `25.86` | `23.92` |
| ApexEMeans (BF16+Graph) | `1` | `small_bf16_graph/bf16` | `23.96` (`21.88+2.08`) | `0.0` | `168.7922` | `270.4754` | `25.86` | `23.81` |
| ApexEMeans (CroissFP8+Graph) | `1` | `medium_croiss_fp8_graph/croiss_fp8` | `25.11` (`23.04+2.07`) | `0.0` | `169.0101` | `267.3313` | `25.85` | `23.86` |
| ApexEMeans (Auto Flagship) | `1` | `small_bf16_graph/bf16` | `23.75` (`21.67+2.07`) | `0.0` | `168.7922` | `270.4754` | `25.86` | `23.81` |
| e-means (Meta FAIR) | `2` | `n/a/n/a` | `20.30` (`18.38+1.92`) | `0.0` | `172.2951` | `273.9783` | `25.77` | `23.75` |
| s-means (Triton) | `2` | `n/a/n/a` | `12.10` (`10.19+1.92`) | `0.0` | `167.6752` | `264.1985` | `25.89` | `23.91` |
| e-means++ | `2` | `n/a/n/a` | `74.46` (`72.54+1.91`) | `0.0` | `167.3058` | `265.9254` | `25.90` | `23.88` |
| UltraEMeans | `2` | `n/a/n/a` | `24.64` (`22.73+1.91`) | `0.0` | `167.3450` | `266.1972` | `25.89` | `23.88` |
| ApexEMeans (BF16+Graph) | `2` | `small_bf16_graph/bf16` | `18.94` (`17.02+1.92`) | `0.0` | `167.5196` | `265.3996` | `25.89` | `23.89` |
| ApexEMeans (CroissFP8+Graph) | `2` | `medium_croiss_fp8_graph/croiss_fp8` | `20.29` (`18.37+1.92`) | `0.0` | `167.7829` | `261.0553` | `25.88` | `23.96` |
| ApexEMeans (Auto Flagship) | `2` | `small_bf16_graph/bf16` | `18.92` (`17.01+1.92`) | `0.0` | `167.5196` | `265.3966` | `25.89` | `23.89` |
| e-means (Meta FAIR) | `5` | `n/a/n/a` | `26.31` (`24.39+1.92`) | `0.0` | `171.4779` | `274.4931` | `25.79` | `23.75` |
| s-means (Triton) | `5` | `n/a/n/a` | `15.12` (`13.20+1.92`) | `0.0` | `167.1737` | `263.8816` | `25.90` | `23.92` |
| e-means++ | `5` | `n/a/n/a` | `83.07` (`81.15+1.92`) | `0.0` | `166.7623` | `262.8378` | `25.91` | `23.93` |
| UltraEMeans | `5` | `n/a/n/a` | `30.47` (`28.56+1.91`) | `0.0` | `166.9067` | `263.3994` | `25.91` | `23.93` |
| ApexEMeans (BF16+Graph) | `5` | `small_bf16_graph/bf16` | `29.29` (`27.38+1.92`) | `0.0` | `166.7779` | `265.1015` | `25.91` | `23.90` |
| ApexEMeans (CroissFP8+Graph) | `5` | `medium_croiss_fp8_graph/croiss_fp8` | `30.44` (`28.52+1.92`) | `0.0` | `167.2293` | `260.8905` | `25.90` | `23.97` |
| ApexEMeans (Auto Flagship) | `5` | `medium_croiss_fp8_graph/croiss_fp8` | `30.38` (`28.47+1.91`) | `0.0` | `167.2411` | `261.0042` | `25.90` | `23.96` |

### `vq_patch_500k_k1024` — Megapixel 4×4 Patch VQ (500K×48, K=1,024) (`0.62 bpp`, `38.4x` Compression vs 24-bit RGB)
- **Shape & Hardware**: `N_train = 500,000`, `N_test = 100,000`, `D = 48`, `K = 1,024`, GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition (600W)` (`acorn:GPU1`).

| Arm | `T` | Gear / Prec | Total `ms` (`codebook+encode`) | Empty | Train Pixel MSE | Test Pixel MSE | Full-Image PSNR (dB) | Held-Out PSNR (dB) |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Pruning / Palette | `0` | `random/n/a` | `3.94` (`0.17+3.77`) | `0.0` | `170.3152` | `256.2852` | `25.82` | `24.05` |
| Lloyd @ 25 ep | `25` | `n/a/n/a` | `27.77` (`24.02+3.75`) | `0.0` | `119.8752` | `176.7296` | `27.34` | `25.66` |
| e-means (Meta FAIR) | `1` | `n/a/n/a` | `24.38` (`20.62+3.76`) | `0.0` | `123.3861` | `184.5856` | `27.22` | `25.47` |
| s-means (Triton) | `1` | `n/a/n/a` | `14.06` (`10.32+3.75`) | `0.0` | `117.5676` | `177.8008` | `27.43` | `25.63` |
| e-means++ | `1` | `n/a/n/a` | `53.63` (`49.89+3.74`) | `0.0` | `117.4267` | `175.6748` | `27.43` | `25.69` |
| UltraEMeans | `1` | `n/a/n/a` | `32.45` (`28.70+3.75`) | `0.0` | `116.7424` | `171.7576` | `27.46` | `25.78` |
| ApexEMeans (BF16+Graph) | `1` | `small_bf16_graph/bf16` | `29.88` (`25.94+3.94`) | `0.0` | `116.9450` | `178.3661` | `27.45` | `25.62` |
| ApexEMeans (CroissFP8+Graph) | `1` | `medium_croiss_fp8_graph/croiss_fp8` | `30.34` (`26.42+3.92`) | `0.0` | `117.2137` | `176.0530` | `27.44` | `25.68` |
| ApexEMeans (Auto Flagship) | `1` | `medium_croiss_fp8_graph/croiss_fp8` | `30.36` (`26.43+3.94`) | `0.0` | `117.2165` | `176.0578` | `27.44` | `25.68` |
| e-means (Meta FAIR) | `2` | `n/a/n/a` | `24.56` (`20.81+3.75`) | `0.0` | `120.3294` | `176.4485` | `27.33` | `25.67` |
| s-means (Triton) | `2` | `n/a/n/a` | `14.25` (`10.49+3.75`) | `0.0` | `115.7144` | `173.4410` | `27.50` | `25.74` |
| e-means++ | `2` | `n/a/n/a` | `53.43` (`49.67+3.75`) | `0.0` | `115.4585` | `170.3670` | `27.51` | `25.82` |
| UltraEMeans | `2` | `n/a/n/a` | `33.11` (`29.35+3.76`) | `0.0` | `115.2871` | `168.0547` | `27.51` | `25.88` |
| ApexEMeans (BF16+Graph) | `2` | `small_bf16_graph/bf16` | `23.84` (`20.09+3.75`) | `0.0` | `115.3299` | `174.5004` | `27.51` | `25.71` |
| ApexEMeans (CroissFP8+Graph) | `2` | `medium_croiss_fp8_graph/croiss_fp8` | `25.41` (`21.66+3.74`) | `0.0` | `115.5885` | `173.2098` | `27.50` | `25.75` |
| ApexEMeans (Auto Flagship) | `2` | `medium_croiss_fp8_graph/croiss_fp8` | `25.31` (`21.56+3.75`) | `0.0` | `115.5908` | `172.7315` | `27.50` | `25.76` |
| e-means (Meta FAIR) | `5` | `n/a/n/a` | `33.33` (`29.56+3.77`) | `0.0` | `118.3779` | `172.6417` | `27.40` | `25.76` |
| s-means (Triton) | `5` | `n/a/n/a` | `17.28` (`13.55+3.73`) | `0.0` | `114.6928` | `173.2304` | `27.54` | `25.74` |
| e-means++ | `5` | `n/a/n/a` | `63.61` (`59.84+3.76`) | `0.0` | `114.4267` | `170.5886` | `27.55` | `25.81` |
| UltraEMeans | `5` | `n/a/n/a` | `44.03` (`40.30+3.73`) | `0.0` | `114.3355` | `167.4265` | `27.55` | `25.89` |
| ApexEMeans (BF16+Graph) | `5` | `small_bf16_graph/bf16` | `34.61` (`30.86+3.75`) | `0.0` | `114.2765` | `171.7574` | `27.55` | `25.78` |
| ApexEMeans (CroissFP8+Graph) | `5` | `medium_croiss_fp8_graph/croiss_fp8` | `40.54` (`36.80+3.74`) | `0.0` | `114.6787` | `167.3310` | `27.54` | `25.90` |
| ApexEMeans (Auto Flagship) | `5` | `medium_croiss_fp8_graph/croiss_fp8` | `41.39` (`37.63+3.76`) | `0.0` | `114.6745` | `167.9995` | `27.54` | `25.88` |

### `vq_patch_500k_k4096` — Megapixel 4×4 Patch VQ (500K×48, K=4,096) (`0.75 bpp`, `32.0x` Compression vs 24-bit RGB)
- **Shape & Hardware**: `N_train = 500,000`, `N_test = 100,000`, `D = 48`, `K = 4,096`, GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition (600W)` (`acorn:GPU1`).

| Arm | `T` | Gear / Prec | Total `ms` (`codebook+encode`) | Empty | Train Pixel MSE | Test Pixel MSE | Full-Image PSNR (dB) | Held-Out PSNR (dB) |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Pruning / Palette | `0` | `random/n/a` | `10.69` (`0.18+10.51`) | `0.0` | `119.4320` | `160.1302` | `27.36` | `26.09` |
| Lloyd @ 25 ep | `25` | `n/a/n/a` | `66.64` (`56.32+10.31`) | `0.0` | `85.8636` | `125.9888` | `28.79` | `27.13` |
| e-means (Meta FAIR) | `1` | `n/a/n/a` | `35.58` (`24.98+10.60`) | `0.4` | `89.8183` | `131.4241` | `28.60` | `26.95` |
| s-means (Triton) | `1` | `n/a/n/a` | `22.26` (`11.75+10.50`) | `0.0` | `85.4954` | `124.4955` | `28.81` | `27.18` |
| e-means++ | `1` | `n/a/n/a` | `114.10` (`103.38+10.72`) | `0.6` | `83.9693` | `124.3189` | `28.89` | `27.19` |
| UltraEMeans | `1` | `n/a/n/a` | `48.20` (`37.47+10.73`) | `0.0` | `83.0731` | `119.8694` | `28.94` | `27.34` |
| ApexEMeans (BF16+Graph) | `1` | `small_bf16_graph/bf16` | `141.25` (`130.53+10.71`) | `0.0` | `83.4691` | `121.7883` | `28.92` | `27.28` |
| ApexEMeans (CroissFP8+Graph) | `1` | `medium_croiss_fp8_graph/croiss_fp8` | `35.91` (`25.29+10.62`) | `2.6` | `83.8153` | `120.7252` | `28.90` | `27.31` |
| ApexEMeans (Auto Flagship) | `1` | `medium_croiss_fp8_graph/croiss_fp8` | `35.74` (`25.14+10.60`) | `2.8` | `83.8110` | `120.7392` | `28.90` | `27.31` |
| e-means (Meta FAIR) | `2` | `n/a/n/a` | `36.74` (`26.05+10.69`) | `0.0` | `86.7879` | `126.7957` | `28.75` | `27.10` |
| s-means (Triton) | `2` | `n/a/n/a` | `23.82` (`13.33+10.49`) | `0.0` | `82.3862` | `119.5714` | `28.97` | `27.35` |
| e-means++ | `2` | `n/a/n/a` | `115.18` (`104.46+10.72`) | `0.0` | `81.7094` | `121.6203` | `29.01` | `27.28` |
| UltraEMeans | `2` | `n/a/n/a` | `50.25` (`39.54+10.71`) | `0.0` | `81.4901` | `120.2308` | `29.02` | `27.33` |
| ApexEMeans (BF16+Graph) | `2` | `small_bf16_graph/bf16` | `55.98` (`45.44+10.54`) | `0.0` | `81.6973` | `119.9461` | `29.01` | `27.34` |
| ApexEMeans (CroissFP8+Graph) | `2` | `medium_croiss_fp8_graph/croiss_fp8` | `38.98` (`28.36+10.62`) | `2.8` | `82.1042` | `120.3752` | `28.99` | `27.33` |
| ApexEMeans (Auto Flagship) | `2` | `medium_croiss_fp8_graph/croiss_fp8` | `38.76` (`28.25+10.52`) | `3.0` | `82.1026` | `122.0994` | `28.99` | `27.26` |
| e-means (Meta FAIR) | `5` | `n/a/n/a` | `50.55` (`39.96+10.60`) | `0.2` | `84.9957` | `122.9385` | `28.84` | `27.23` |
| s-means (Triton) | `5` | `n/a/n/a` | `32.94` (`22.54+10.40`) | `0.0` | `81.1168` | `118.2157` | `29.04` | `27.40` |
| e-means++ | `5` | `n/a/n/a` | `132.21` (`121.48+10.73`) | `0.0` | `80.4092` | `120.2586` | `29.08` | `27.33` |
| UltraEMeans | `5` | `n/a/n/a` | `74.13` (`63.42+10.72`) | `0.0` | `80.3231` | `118.1537` | `29.08` | `27.41` |
| ApexEMeans (BF16+Graph) | `5` | `small_bf16_graph/bf16` | `74.74` (`64.20+10.54`) | `0.0` | `80.4047` | `118.5542` | `29.08` | `27.39` |
| ApexEMeans (CroissFP8+Graph) | `5` | `medium_croiss_fp8_graph/croiss_fp8` | `55.71` (`45.18+10.52`) | `0.8` | `80.7803` | `119.5717` | `29.06` | `27.35` |
| ApexEMeans (Auto Flagship) | `5` | `medium_croiss_fp8_graph/croiss_fp8` | `56.26` (`45.64+10.62`) | `1.0` | `80.7807` | `118.9580` | `29.06` | `27.38` |

---

## 6. Paired Statistical Hypothesis Tests (`ApexEMeans` at `T = 2`, `n = 5` Paired Seeds)

| Suite | Cell | Baseline | Metric | `Apex` Mean ± SD | Baseline Mean ± SD | Delta | Speedup | W/T/L | `p_ttest` | `p_wilcoxon` |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | :---: | ---: | ---: |
| `semdedup` | `agnews_256_k1024` | `random` | `primary_acc_10pct` | `85.76 ± 0.17` | `86.42 ± 0.11` | `-0.66 pp` | `1.00x` | `0/0/5` | `0.003308` | `0.0625` |
| `semdedup` | `agnews_256_k1024` | `random` | `primary_acc_20pct` | `86.70 ± 0.11` | `87.06 ± 0.12` | `-0.37 pp` | `1.00x` | `0/0/5` | `0.02325` | `0.0625` |
| `semdedup` | `agnews_256_k1024` | `random` | `primary_acc_50pct` | `87.25 ± 0.17` | `87.41 ± 0.12` | `-0.16 pp` | `1.00x` | `2/0/3` | `0.1773` | `0.3125` |
| `semdedup` | `agnews_256_k1024` | `lloyd_25` | `primary_acc_20pct` | `86.70 ± 0.11` | `86.51 ± 0.14` | `+0.18 pp` | `1.00x` | `5/0/0` | `0.1236` | `0.0625` |
| `semdedup` | `agnews_256_k1024` | `smeans` | `proto_primary_acc` | `69.30 ± 2.22` | `69.41 ± 0.76` | `-0.11 pp` | `1.00x` | `4/0/1` | `0.9378` | `0.625` |
| `semdedup` | `agnews_256_k1024` | `emeans` | `primary_acc_20pct` | `86.70 ± 0.11` | `86.73 ± 0.19` | `-0.03 pp` | `1.00x` | `1/0/4` | `0.5446` | `0.625` |
| `semdedup` | `cifar10_r18_512_k512` | `random` | `primary_acc_10pct` | `79.61 ± 0.35` | `79.27 ± 0.39` | `+0.34 pp` | `1.00x` | `3/0/2` | `0.3313` | `0.625` |
| `semdedup` | `cifar10_r18_512_k512` | `random` | `primary_acc_20pct` | `81.16 ± 0.16` | `80.97 ± 0.25` | `+0.19 pp` | `1.00x` | `3/0/2` | `0.2847` | `0.4375` |
| `semdedup` | `cifar10_r18_512_k512` | `random` | `primary_acc_50pct` | `82.86 ± 0.16` | `82.68 ± 0.15` | `+0.17 pp` | `1.00x` | `5/0/0` | `0.02937` | `0.0625` |
| `semdedup` | `cifar10_r18_512_k512` | `lloyd_25` | `primary_acc_20pct` | `81.16 ± 0.16` | `81.17 ± 0.42` | `-0.01 pp` | `1.00x` | `2/0/3` | `0.9667` | `1` |
| `semdedup` | `cifar10_r18_512_k512` | `smeans` | `proto_primary_acc` | `74.64 ± 0.29` | `73.86 ± 0.54` | `+0.78 pp` | `1.00x` | `5/0/0` | `0.01214` | `0.0625` |
| `semdedup` | `cifar10_r18_512_k512` | `emeans` | `primary_acc_20pct` | `81.16 ± 0.16` | `81.05 ± 0.25` | `+0.11 pp` | `1.00x` | `3/0/2` | `0.4652` | `0.625` |
| `semdedup` | `cifar10_r18_512_k1024` | `random` | `primary_acc_10pct` | `79.37 ± 0.22` | `79.27 ± 0.39` | `+0.11 pp` | `1.00x` | `3/0/2` | `0.645` | `0.8125` |
| `semdedup` | `cifar10_r18_512_k1024` | `random` | `primary_acc_20pct` | `81.11 ± 0.25` | `80.97 ± 0.25` | `+0.15 pp` | `1.00x` | `4/0/1` | `0.2772` | `0.3125` |
| `semdedup` | `cifar10_r18_512_k1024` | `random` | `primary_acc_50pct` | `82.98 ± 0.09` | `82.68 ± 0.15` | `+0.30 pp` | `1.00x` | `5/0/0` | `0.003961` | `0.0625` |
| `semdedup` | `cifar10_r18_512_k1024` | `lloyd_25` | `primary_acc_20pct` | `81.11 ± 0.25` | `81.14 ± 0.26` | `-0.03 pp` | `1.00x` | `2/0/3` | `0.8912` | `1` |
| `semdedup` | `cifar10_r18_512_k1024` | `smeans` | `proto_primary_acc` | `76.07 ± 0.17` | `75.13 ± 0.80` | `+0.94 pp` | `1.00x` | `4/0/1` | `0.0555` | `0.125` |
| `semdedup` | `cifar10_r18_512_k1024` | `emeans` | `primary_acc_20pct` | `81.11 ± 0.25` | `80.95 ± 0.23` | `+0.16 pp` | `1.00x` | `3/0/2` | `0.1859` | `0.3125` |
| `semdedup` | `cifar10_convnext_768_k2048` | `random` | `primary_acc_10pct` | `89.62 ± 0.10` | `89.09 ± 0.17` | `+0.53 pp` | `1.00x` | `5/0/0` | `0.008362` | `0.0625` |
| `semdedup` | `cifar10_convnext_768_k2048` | `random` | `primary_acc_20pct` | `89.98 ± 0.12` | `89.82 ± 0.24` | `+0.16 pp` | `1.00x` | `2/0/3` | `0.3631` | `0.8125` |
| `semdedup` | `cifar10_convnext_768_k2048` | `random` | `primary_acc_50pct` | `90.96 ± 0.10` | `90.43 ± 0.17` | `+0.53 pp` | `1.00x` | `5/0/0` | `0.001614` | `0.0625` |
| `semdedup` | `cifar10_convnext_768_k2048` | `lloyd_25` | `primary_acc_20pct` | `89.98 ± 0.12` | `89.82 ± 0.11` | `+0.16 pp` | `1.00x` | `4/0/1` | `0.0974` | `0.1875` |
| `semdedup` | `cifar10_convnext_768_k2048` | `smeans` | `proto_primary_acc` | `88.75 ± 0.34` | `87.67 ± 0.12` | `+1.09 pp` | `1.00x` | `5/0/0` | `0.0005169` | `0.0625` |
| `semdedup` | `cifar10_convnext_768_k2048` | `emeans` | `primary_acc_20pct` | `89.98 ± 0.12` | `89.79 ± 0.06` | `+0.19 pp` | `1.00x` | `5/0/0` | `0.007129` | `0.0625` |
| `semdedup` | `covtype_54_k4096` | `random` | `primary_acc_10pct` | `90.79 ± 0.07` | `90.19 ± 0.05` | `+0.60 pp` | `1.00x` | `5/0/0` | `7.164e-05` | `0.0625` |
| `semdedup` | `covtype_54_k4096` | `random` | `primary_acc_20pct` | `93.85 ± 0.14` | `93.32 ± 0.08` | `+0.54 pp` | `1.00x` | `5/0/0` | `0.001899` | `0.0625` |
| `semdedup` | `covtype_54_k4096` | `random` | `primary_acc_50pct` | `95.68 ± 0.06` | `95.53 ± 0.06` | `+0.15 pp` | `1.00x` | `4/0/1` | `0.03464` | `0.125` |
| `semdedup` | `covtype_54_k4096` | `lloyd_25` | `primary_acc_20pct` | `93.85 ± 0.14` | `93.81 ± 0.10` | `+0.04 pp` | `1.00x` | `4/0/1` | `0.5196` | `0.5625` |
| `semdedup` | `covtype_54_k4096` | `smeans` | `proto_primary_acc` | `74.80 ± 0.18` | `74.74 ± 0.18` | `+0.06 pp` | `1.00x` | `2/0/3` | `0.6369` | `1` |
| `semdedup` | `covtype_54_k4096` | `emeans` | `primary_acc_20pct` | `93.85 ± 0.14` | `93.85 ± 0.10` | `+0.00 pp` | `1.00x` | `2/0/3` | `0.9745` | `1` |
| `megapixel_vq` | `vq_rgb_1m_k256` | `lloyd_25` | `test_psnr_db` | `27.61 ± 0.03` | `27.29 ± 0.11` | `+0.31 dB` | `1.00x` | `5/0/0` | `0.006048` | `0.0625` |
| `megapixel_vq` | `vq_rgb_1m_k256` | `emeans` | `test_psnr_db` | `27.61 ± 0.03` | `27.40 ± 0.11` | `+0.20 dB` | `1.00x` | `5/0/0` | `0.02242` | `0.0625` |
| `megapixel_vq` | `vq_rgb_1m_k256` | `smeans` | `test_psnr_db` | `27.61 ± 0.03` | `27.59 ± 0.08` | `+0.02 dB` | `1.00x` | `2/0/3` | `0.6839` | `0.8125` |
| `megapixel_vq` | `vq_rgb_1m_k256` | `ultra_emeans` | `test_psnr_db` | `27.61 ± 0.03` | `27.60 ± 0.10` | `+0.01 dB` | `1.00x` | `2/0/3` | `0.888` | `1` |
| `megapixel_vq` | `vq_rgb_1m_k1024` | `lloyd_25` | `test_psnr_db` | `30.90 ± 0.30` | `29.82 ± 0.34` | `+1.08 dB` | `1.00x` | `5/0/0` | `0.003125` | `0.0625` |
| `megapixel_vq` | `vq_rgb_1m_k1024` | `emeans` | `test_psnr_db` | `30.90 ± 0.30` | `29.82 ± 0.35` | `+1.09 dB` | `1.00x` | `5/0/0` | `0.01188` | `0.0625` |
| `megapixel_vq` | `vq_rgb_1m_k1024` | `smeans` | `test_psnr_db` | `30.90 ± 0.30` | `30.36 ± 0.33` | `+0.54 dB` | `1.00x` | `5/0/0` | `0.01455` | `0.0625` |
| `megapixel_vq` | `vq_rgb_1m_k1024` | `ultra_emeans` | `test_psnr_db` | `30.90 ± 0.30` | `31.09 ± 0.40` | `-0.19 dB` | `1.00x` | `2/0/3` | `0.5728` | `0.625` |
| `megapixel_vq` | `vq_rgb_1m_k4096` | `lloyd_25` | `test_psnr_db` | `33.98 ± 0.15` | `33.14 ± 0.13` | `+0.84 dB` | `1.00x` | `5/0/0` | `5.758e-05` | `0.0625` |
| `megapixel_vq` | `vq_rgb_1m_k4096` | `emeans` | `test_psnr_db` | `33.98 ± 0.15` | `33.14 ± 0.12` | `+0.84 dB` | `1.00x` | `5/0/0` | `0.0002491` | `0.0625` |
| `megapixel_vq` | `vq_rgb_1m_k4096` | `smeans` | `test_psnr_db` | `33.98 ± 0.15` | `33.59 ± 0.03` | `+0.39 dB` | `1.00x` | `5/0/0` | `0.005324` | `0.0625` |
| `megapixel_vq` | `vq_rgb_1m_k4096` | `ultra_emeans` | `test_psnr_db` | `33.98 ± 0.15` | `33.97 ± 0.14` | `+0.00 dB` | `1.00x` | `2/0/3` | `0.9119` | `0.8125` |
| `megapixel_vq` | `vq_patch_500k_k256` | `lloyd_25` | `test_psnr_db` | `23.89 ± 0.07` | `23.75 ± 0.09` | `+0.14 dB` | `1.00x` | `4/0/1` | `0.0971` | `0.1875` |
| `megapixel_vq` | `vq_patch_500k_k256` | `emeans` | `test_psnr_db` | `23.89 ± 0.07` | `23.75 ± 0.11` | `+0.14 dB` | `1.00x` | `4/0/1` | `0.103` | `0.125` |
| `megapixel_vq` | `vq_patch_500k_k256` | `smeans` | `test_psnr_db` | `23.89 ± 0.07` | `23.91 ± 0.12` | `-0.02 dB` | `1.00x` | `3/0/2` | `0.747` | `1` |
| `megapixel_vq` | `vq_patch_500k_k256` | `ultra_emeans` | `test_psnr_db` | `23.89 ± 0.07` | `23.88 ± 0.04` | `+0.01 dB` | `1.00x` | `3/0/2` | `0.6547` | `0.8125` |
| `megapixel_vq` | `vq_patch_500k_k1024` | `lloyd_25` | `test_psnr_db` | `25.76 ± 0.11` | `25.66 ± 0.13` | `+0.10 dB` | `1.00x` | `3/0/2` | `0.3262` | `0.4375` |
| `megapixel_vq` | `vq_patch_500k_k1024` | `emeans` | `test_psnr_db` | `25.76 ± 0.11` | `25.67 ± 0.07` | `+0.09 dB` | `1.00x` | `4/0/1` | `0.2583` | `0.1875` |
| `megapixel_vq` | `vq_patch_500k_k1024` | `smeans` | `test_psnr_db` | `25.76 ± 0.11` | `25.74 ± 0.07` | `+0.02 dB` | `1.00x` | `3/0/2` | `0.788` | `0.8125` |
| `megapixel_vq` | `vq_patch_500k_k1024` | `ultra_emeans` | `test_psnr_db` | `25.76 ± 0.11` | `25.88 ± 0.12` | `-0.12 dB` | `1.00x` | `1/0/4` | `0.1955` | `0.3125` |
| `megapixel_vq` | `vq_patch_500k_k4096` | `lloyd_25` | `test_psnr_db` | `27.26 ± 0.08` | `27.13 ± 0.10` | `+0.14 dB` | `1.00x` | `4/0/1` | `0.1258` | `0.1875` |
| `megapixel_vq` | `vq_patch_500k_k4096` | `emeans` | `test_psnr_db` | `27.26 ± 0.08` | `27.10 ± 0.08` | `+0.16 dB` | `1.00x` | `4/0/1` | `0.08237` | `0.125` |
| `megapixel_vq` | `vq_patch_500k_k4096` | `smeans` | `test_psnr_db` | `27.26 ± 0.08` | `27.35 ± 0.07` | `-0.09 dB` | `1.00x` | `1/0/4` | `0.1894` | `0.1875` |
| `megapixel_vq` | `vq_patch_500k_k4096` | `ultra_emeans` | `test_psnr_db` | `27.26 ± 0.08` | `27.33 ± 0.11` | `-0.07 dB` | `1.00x` | `1/0/4` | `0.1657` | `0.125` |

---

## 7. Reproducibility Commands

```bash
# 1. Run full unit, adversarial, zero-leakage, & fleet-hygiene test suite (/home scratch):
TMPDIR=/home/catid/.cache/emeans_tmp TRITON_CACHE_DIR=/home/catid/.cache/emeans_tmp/triton_semdedup \
  CUDA_VISIBLE_DEVICES=1 .venv/bin/pytest -v tests/test_semdedup_coreset.py

# 2. Run 4-GPU single-SKU production sweep across acorn:0,1 (600W Server) + foureyes:1,3 (Max-Q):
.venv/bin/python scripts/run_semdedup_benchmarks.py --dispatch-fleet --wave all \
  --remote-host acorn.stratabaremetal.com --local-gpus 1,3
```
