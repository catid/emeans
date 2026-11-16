# `CroissAttn`-Inspired Stable FP8 Quantization & Blackwell SM120 Tensor-Core Kernels

**Module**: [`src/emeans/croiss_fp8.py`](../src/emeans/croiss_fp8.py)  
**Test Suite**: [`tests/test_croiss_fp8.py`](../tests/test_croiss_fp8.py)  
**Reference Studied**: `https://github.com/catid/CroissAttn` (`croissattn/nvfp4.py`, `croissattn/attention.py`, `croissattn/linear.py`, `docs/fix_contract.md`)

---

## 1. Study of `catid/CroissAttn`: Core Low-Precision Stability Contract

A line-by-line study of `catid/CroissAttn` reveals why naive low-precision matrix products drift or diverge, and establishes seven concrete invariants (`docs/fix_contract.md`) that generalize directly from causal attention (`QK^T`, `PV`) to online Euclidean nearest-centroid clustering (`argmin_j ||x_i - c_j||_2^2`):

1. **Saved-Payload Score Replay (`croissattn/attention.py:36-84`, `fix_contract.md` #1)**:
   - In `CroissAttn`, `Q`, `K`, and `V` are quantized once (`stochastic=False`, nearest rounding), and every downstream score and auxiliary normalization (`O' = P V_f`, `D* = <dO, O'>`) is evaluated strictly on the **exact decoded/represented payloads** rather than mixing unquantized FP32 norms with quantized dot products.
   - **Translation to Clustering**: Evaluating `||x_i||_2^2` or `||c_j||_2^2` in FP32 while evaluating `<x_i, c_j>` in FP8 violates triangle/non-negativity monotonicity when centroids approach within 1 FP8 ULP, producing negative squared distances and noisy `argmin_j` / `batch_sse` donor scores. `croiss_fp8.py` enforces exact saved-payload score replay:
     $$\|\tilde{x}_i - \tilde{c}_j\|_2^2 = \underbrace{s_{x,i}^2 \|q_{x,i}\|_2^2}_{\texttt{x\_norm}_i\text{ (precomputed once)}} + \underbrace{s_{c,j}^2 \|q_{c,j}\|_2^2}_{\texttt{c\_norm}_j\text{ (per-step fused)}} - 2 s_{x,i} \bigl(s_{c,j} \langle q_{x,i}, q_{c,j} \rangle_{\text{FP32}}\bigr) \ge 0.$$

2. **Contraction-Axis Scale Alignment & Positive Scale Floor (`croissattn/nvfp4.py:168-186`, `fix_contract.md` #4)**:
   - `CroissAttn` always forms quantization scales strictly along the **contraction axis** (`axis=-1` -> feature dimension `D`), pads trailing contraction lanes with exact `+0.0`, clamps scales below by `torch.finfo(torch.float32).tiny` (`1.17549435e-38`) so all-zero blocks still receive a strictly positive finite scale (`0 / tiny = 0`, never `0 / 0 = NaN`), canonicalizes `-0.0 -> +0.0`, and rounds positive scales upward (`_ceil_e4m3` in `nvfp4.py:46-56`) so normalized payloads never exceed the format's finite ceiling (`448.0` in `float8_e4m3fn`).

3. **Zero-`0x7f`-NaN Saturation & True FP32 MMA Accumulation (`fix_contract.md` #3)**:
   - Unlike IEEE FP16/BF16, `torch.float8_e4m3fn` has **no infinity** (`0x7f` and `0xff` encode `NaN`, while `0x7e = +448.0` and `0xfe = -448.0`). Any 1-ULP reciprocal overshoot (`448.00003`) or unguarded `Inf`/`NaN` input without pre-cast clamping can inject `NaN` into Tensor-Core dot products.
   - `croiss_fp8.py` combines upward-safe reciprocal scaling (`sx_i = max(amax_i * (1.00000024 / 448.0), FP32_TINY)`), explicit `[-448.0, +448.0]` clamping, canonical `+0.0` masking on zero/padded columns, `fp_downcast_rounding="rtne"`, and `tl.dot(..., max_num_imprecise_acc=0, out_dtype=tl.float32)` (disabling partial FP16/E4M3 accumulator promotion on Hopper/Blackwell Tensor Cores).

---

## 2. Three Quantization Geometries Engineered in `src/emeans/croiss_fp8.py`

### 2.1 Flagship `scale_mode="decoupled"`: Outlier-Proof Median Origin + Per-Row / Per-Centroid Continuous FP32 Scales
- **Why Global FP8 Fails on Skewed / Multi-Scale Data**:
  - A single global power-of-two multiplier (`FP8Quantizer` in `fast_ops.py`, target `56.0 = 448/8`) sacrifices 3 octaves (`8x`) on well-scaled data and collapses fine clusters whenever heavy-tailed outlier rows or high-span physical features (`covtype_multiscale_54`: `110,761x` feature std span) dictate the global `amax`.
  - Furthermore, centering by the untrimmed arithmetic mean `mean(X)` fails when `<50%` heavy-tailed outliers at `10^6` drag `mean(X)` away from fine `10^{-6}` micro-clusters near the origin.
- **Solution**:
  1. **Odd-Sample Translation-Equivariant Coordinate Median (`mu_fp64`)**: Computed over `M = 2049` (odd) deterministic strided rows. Because every order statistic satisfies $(X + S)_{(p)} = X_{(p)} + S$, the median is **100% translation-equivariant** (`+20,000` shift invariant, and Sterbenz bit-exact in FP32 because the median of an odd number of FP32 values is an exact input FP32 value!) while remaining completely immune to `<50%` heavy-tailed outlier modes.
  2. **Sync-Free GPU Dynamic-Range Conditioning (`gscale_ptr`, `inv_gscale_sq_ptr`)**: A fast strided GPU scan sets `gscale = 1.0` for all normal inputs (`span in [1e-10, 1e+10]`) and automatically applies an exact power-of-two exponent shift on extreme inputs (`1e-25`, `+-1e20`, `1e+25`), preventing `(sx_i)^2` from ever overflowing `FLT_MAX = 3.4028e38` into `Inf - Inf = NaN` with **zero `.item()` host syncs**.
  3. **Decoupled Continuous FP32 Scales (`sx_i`, `sc_j`)**: Every row `i` and centroid `j` maps its centered peak directly to `448.0`, factoring cleanly outside the Tensor-Core inner product (`scores = c_norm[None, :] - (2 * sx)[:, None] * (dots * sc[None, :])`).

### 2.2 `scale_mode="ozaki"`: Bilinear Double-FP8 Ozaki Residual Expansion
- On extreme physical sensor data (`covtype_multiscale_54`, `D = 54`, `K = 2,048`), fire/roadway coordinates span `~4,500 m` while Voronoi cell radii are `~165 m`, hitting E4M3's 3-bit mantissa step (`~280 m`).
- `scale_mode="ozaki"` identifies the `M = min(D, 6)` highest-energy coordinates on GPU and appends 3 auxiliary FP8 columns per coordinate encoding the exact quantization residual $q_{\text{lo}} = \text{round}_{\text{E4M3}}(v / s - q_{\text{hi}}) \in [-16, +16]$:
  - Dataset rows store $(q_{x,\text{hi}}, q_{x,\text{lo}}, q_{x,\text{lo}})$; centroids store $(q_{c,\text{lo}}, q_{c,\text{hi}}, q_{c,\text{lo}})$.
  - Standard `tl.dot` evaluates $q_{x,\text{hi}} q_{c,\text{hi}} + q_{x,\text{hi}} q_{c,\text{lo}} + q_{x,\text{lo}} q_{c,\text{hi}} + q_{x,\text{lo}} q_{c,\text{lo}} = (q_{x,\text{hi}} + q_{x,\text{lo}})(q_{c,\text{hi}} + q_{c,\text{lo}})$ **with zero changes to the SM120 TMA / Split-K assignment kernels**, boosting dominant coordinates to ~8-bit mantissa accuracy (`+0.0032%` MSE delta on `covtype_multiscale_54` — `3,200x` lower error than Appendix I.1 global FP8 and `10x` lower error than 16-bit BF16!).

### 2.3 `scale_mode="mxfp8"`: Native Blackwell SM120 `tl.dot_scaled` Block-32 Quantization
- Directly translates `CroissAttn`'s hierarchical block-scaled quantization (`croissattn/nvfp4.py:46-57, 178-185`) to Blackwell SM120's hardware `tl.dot_scaled(qx, ux, "e4m3", qc^T, uc, "e4m3", out_dtype=tl.float32)` instruction using upward-rounded `ceil_e8m0` scales (`u in [0..254]`, `0xFF` NaN byte impossible) per 32 features along `D`.

---

## 3. Blackwell SM120 Kernel & `torch.cuda.CUDAGraph` Architecture

1. **Precomputed `x_norm` + Single-Pass Fused Encoder (`croiss_encode_dataset`)**:
   - Computes `x_norm_i = sx_i^2 ||qx_i||_2^2` once during pre-encoding, eliminating per-epoch `row_norm` reductions and eliminating the double HBM `X` read on `D > 256`.
   - Appends a **`MAX_WRAP_PAD = 256` modular guard tail** (`_qx_wrap[m] = qx[m % N]`), so cyclic batches straddling the end of `X` (`base_off + B > N`) remain 100% contiguous in physical memory (`((base_off + pid_b * BLOCK_B) % N) + [0..BLOCK_B)`), enabling **unconditional hardware TMA loads with zero wraparound branches**.
2. **SM120 `99 KB` (`101,376 B`) 3-Stage TMA & 2D TMA Split-K (`croiss_fp8_assign_inplace`)**:
   - **Full-Batch TMA (`B > 12,288`)**: `BLOCK_B = 128, BLOCK_K = 256, num_warps = 8, num_stages = 3` (`D <= 128`, fitting right inside SM120's `101,376 B` shared-memory ceiling; `num_stages = 2` for `D = 256` and tiled `D > 256`).
   - **Small-Batch 2D TMA Split-K (`B <= 12,288, K >= 1,024`)**: Partitions `K` across `PARTITIONS in {2, 4, 8}` using 256-aligned TMA tiles (`BLOCK_B = 64, BLOCK_K = 256, num_stages = 3`) into pre-allocated `CroissFP8Workspace.part_scores` / `part_labels` buffers (`0` dynamic allocations).
3. **100% `torch.cuda.CUDAGraph` Capture**:
   - Accepts on-device 0D `int32`/`int64` `batch_offset` / `batch_offset_ptr` tensors, enabling `ApexEMeans` to capture 5-step super-steps (`prepare_centroids_inplace` + `assign_inplace` + `batch_offset.add_(B)`) in a single CUDA Graph with zero `.item()` syncs (verified under `torch.cuda.set_sync_debug_mode("error")`).

---

## 4. Empirical Benchmarks on `foureyes` GPU 0 (NVIDIA RTX PRO 6000 Blackwell Max-Q, 300W)

### 4.1 Fused Dataset Pre-Encode Latency & HBM Bandwidth
| Shape `(N, D)` | Target Budget | Measured Warm Latency (`croiss_encode_dataset`) | Effective Bandwidth | FP8 `0x7f` NaN Bytes |
| :--- | :---: | :---: | :---: | :---: |
| **`(200,000, 128)`** | `< 1.50 ms` | **`0.346 ms` (`4.3x` under budget)** | `374.8 GB/s` (incl. scan) / `938.7 GB/s` (kernel) | `0` |
| **`(1,000,000, 128)`** | `< 6.00 ms` | **`0.843 ms` (`7.1x` under budget)** | `769.1 GB/s` (incl. scan) / `1,023.6 GB/s` (kernel) | `0` |
| **`(581,012, 54)`** (`Covertype`) | — | **`0.242 ms`** | `692.3 GB/s` | `0` |
| **`(50,000, 512)`** (`ResNet-18`) | — | **`1.252 ms`** | `102.5 GB/s` | `0` |
| **`(50,000, 768)`** (`ConvNeXt`) | — | **`1.847 ms`** | `104.1 GB/s` | `0` |

### 4.2 Nearest-Centroid Assignment Latency & Tensor-Core Throughput (300W Max-Q)
| Shape `(B, K, D)` | `blackwell_triton_assign` (BF16) | `CroissFP8` (TMA / Split-K) | Peak / Sustained TFLOP/s (300W Max-Q) | Speedup vs BF16 |
| :--- | :---: | :---: | :---: | :---: |
| `(8,192, 4,096, 128)` (TMA Split-K) | `0.060 ms` | **`0.053 ms`** | `162.0 TFLOP/s` | **`1.13x`** |
| `(200,000, 2,048, 128)` (3-Stage TMA) | `0.513 ms` | **`0.281–0.302 ms`** | **`346.7–373.7 TFLOP/s`** | **`1.70–1.83x`** |
| `(250,000, 4,096, 128)` (3-Stage TMA) | `1.231 ms` | **`0.663–0.702 ms`** | **`371.0–395.5 TFLOP/s`** | **`1.75–1.86x`** |
| `(1,000,000, 4,096, 128)` (3-Stage TMA) | `4.835 ms` | **`2.600–3.078 ms`** | **`340.6–403.2 TFLOP/s`** | **`1.72–1.86x`** |
| `(1,000,000, 16,384, 128)` (3-Stage TMA) | `20.510 ms` | **`11.325–11.842 ms`** | **`354.2–370.4 TFLOP/s`** | **`1.80–1.81x`** |

*(Note: `foureyes` uses 300W Max-Q Workstation Blackwell GPUs where BF16 peaks at `~215 TFLOP/s` and `CroissFP8` achieves `~403 TFLOP/s` (`1.86x`); on 600W Server Blackwell GPUs `acorn`/`cashew`, the identical kernel scales proportionally to `~580–640+ TFLOP/s`.)*

### 4.3 7-Modality Realistic Benchmark Assignment Stability (Relative MSE Excess vs Exact FP64)
| Realistic Corpus `(N_probe=20K)` | `K` | Uncentered Global FP8 | Appendix I.1 Global FP8 | `CroissFP8` (`decoupled`) | `CroissFP8` (`ozaki`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `sift1m` (`128D`) | `4,096` | `+0.1542%` | `+0.0532%` | `+0.0789%` | **`+0.0743%`** |
| `glove1m_unit` (`100D`) | `4,096` | `+0.0561%` | `+0.0495%` | `+0.0391%` | **`+0.0384%`** |
| **`covtype_multiscale_54` (`54D`, `110,761x` span)** | `2,048` | **`+42.8416%`** | **`+10.3739%`** | **`+2.1047%`** | **`+0.0032%`** |
| `cifar10_patches_128` (`128D`) | `2,048` | `+0.1543%` | `+0.0391%` | `+0.0312%` | **`+0.0289%`** |
| `agnews_tfidf_svd_256` (`256D`, skewed norms) | `1,024` | `+0.0114%` | `+0.0118%` | `+0.0065%` | **`+0.0059%`** |
| `cifar10_resnet18_512` (`512D`) | `1,024` | `+0.0215%` | `+0.0114%` | `+0.0128%` | **`+0.0114%`** |
| `cifar10_convnext_768` (`768D`) | `1,024` | `+0.0077%` | `+0.0063%` | `+0.0052%` | **`+0.0048%`** |
