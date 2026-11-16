# Forward-Only & Perturbation Optimization (`qlabs-eng/noise`) <-> `e-means++`: Survey & Falsifiable Experimental Tracks

## 1. Complete Algorithm Catalog of `https://github.com/qlabs-eng/noise`

`qlabs-eng/noise` (_"Noise: Speedrun for pretraining without backpropagation"_, Gülbahar, Mandal, Vegesna, & Dahal, 2026) benchmarks forward-only learning algorithms on a fixed 37.7M-parameter Transformer (`GPT`, 8 layers, width 512, 8 heads, RoPE + QK-RMSNorm, `relu^2` MLP, value embeddings + gates, logit softcap `15 tanh(z/15)`) over a single epoch of FineWeb tokens (`10M` archived in `archive/10M/`; `30M` active across `train/val/`, `train/speedrun/`, and `train/arch/`).

**Core Forward-Only Rule**: Autograd (reverse or forward mode) and hand-written backward passes are prohibited. Every learning signal must originate as a **measured forward loss evaluation** (or forward hub score) and may traverse **at most one nonlinear step** before contracting into linear parameter updates—ensuring compatibility with non-differentiable routing or discrete components.

### 1.1 Track Progression & Empirical Findings in `qlabs-eng/noise`

| Benchmark Track | Objective | Baseline -> Current Record | Key Algorithmic Leaps |
| :--- | :--- | :--- | :--- |
| **10M Track 1** (`archive/10M/train/val/`) | Lowest 1-epoch val CE (no time limit) | `5.954` (`eggroll.py`, 5h20m) -> `5.058` (Dust+SGD) -> `4.167` (Dust+Adam) -> **`3.676`** (35m; beats backprop Adam `3.769`!) | Activation-space `Dust` (`1,000-10,000x` sample efficiency over weight-space `EGGROLL`) -> `ANVIL II` -> Sylvester-Hadamard orthogonal frames + exact `log1p/expm1` partition swaps -> Associative ridge control variate -> `Boost -> Cut` -> Live VQ remainder -> Held-out trajectory merge |
| **10M Track 2** (`archive/10M/train/speedrun/`) | Fastest wall-clock to val CE `<= 4.2` | `1h 46m` (Dust+Adam) -> `5m` (`Boost`) -> `1m 23s` (`Identity-Path`) -> `14.6s` (`Drift Leap`) -> **`1.58s`** (250 wide-steps) | Dropping multi-block reruns for `Identity-Path` + closed-form degree-2 softmax moment kernels -> data-free lag-1 RoPE structured start -> tail EMA mix (`EMA_MIX = 0.4`) -> `wide_step` 2x batch doubling (`16` rows, `1.4x` LR) |
| **30M Track 1** (`train/val/run.py`) | Lowest 1-epoch val CE on 30M tokens | `3.767` (`dust.py`, 5h17m) -> `3.409` (`Boost->Cut`+merge) -> `3.338` (forward cross-token readout) -> **`3.300`** (43m; **beats backprop Adam `3.419` and backprop ANVIL II `3.340`**!) | Two-hop twin rows + transit decode + **`head_edit` forward-verified $P/Q$ head transplants** + **`mom_distill` EMA teacher distillation** + per-head `ANVIL II` polar steps + `soup_parts` subspace trajectory merge |
| **30M Track 2** (`train/speedrun/run.py`) | Fastest wall-clock to val CE `<= 3.5` | `1h 43m` -> `8m 20s` (`Identity -> Cut-lite`) -> `2m 03s` (paired MLP roots + **`difficulty_order` stratified batching**) -> `22.9s` -> `15.9s` -> **`14.2s`** | `315`-step `Identity-Path` warmup -> `Cut` with 2 single MLP roots (blocks 4 & 6) + identity control variate + every-2nd-step root amortization -> `difficulty_order` 8-strata batching -> `wide_rows=16` (`1.3x` LR) -> **`quad_merge` 3-point parabolic held-out part merge** |
| **30M Track 3** (`train/arch/run.py`, `model.py`) | Open architecture forward-only CE | `3.767` -> `3.409` -> `3.358` (6 blocks + hashed n-grams) -> **`3.2115`** (12x1024 GPT + hashed 2/3/skip-gram value tables) | Co-designing representation with forward credit: multi-order hashed lookup tables (`2^20 x 128`, Fibonacci hash `2654435761`) entering attention values linearly for exact zero-variance credit |

### 1.2 Exact Mathematical Mechanisms Across the `Noise` Codebase

1. **Weight-Space Low-Rank Antithetic ES (`EGGROLL`, `archive/10M/train/val/eggroll.py`)**:
   - Perturbs each weight matrix implicitly via rank-1 outer products $W^\pm = W \pm \sigma a b^\top$ ($a, b \sim \mathcal{N}(0, I)$ sliced from a shared $2^{28}$-entry Gaussian ring buffer indexed by step seed):
     $$x (W \pm \sigma a b^\top)^\top = x W^\top \pm \sigma (x b) a^\top.$$
   - Evaluates antithetic pairs $(\theta + \sigma E_p, \theta - \sigma E_p)$ on identical inputs, z-scores scalar fitness differences $f_p = L(\theta - \sigma E_p) - L(\theta + \sigma E_p)$, and accumulates $\hat{g}_W = -\frac{1}{P}\sum_{p=1}^P \tilde{f}_p a_p b_p^\top$. Suffers $O(\text{params})$ variance when credit is pooled into one global scalar loss per sequence (`5.954` val loss).

2. **Activation-Space Local-Reward Perturbations & Orthogonal Signed-Hadamard Frames (`Dust`, `train/val/dust.py` & `hadamard()` in `train/val/run.py:284`)**:
   - Perturbs intermediate activations $y_{b,t} \mapsto y_{b,t} + \sigma \epsilon_{b,t}$ rather than weights, scoring each sample $t$ against its **own** local forward loss drop $r_{b,t} = L_{b,t}^{\text{clean}} - L_{b,t}^{\text{pert}}$ (eliminating cross-sample variance!).
   - Replaces i.i.d. Gaussian draws with **random-signed Sylvester-Hadamard orthogonal frames** $H_d \in \{\pm 1\}^{d \times d}$ ($H_d^\top H_d = d I_d$, exact in BF16), eliminating intra-frame directional collinearity and cutting estimator variance by $2\text{--}4\times$.

3. **Exact Incremental Partition-Swap Forward Evaluation (`coordinate_differences` & `frozen_attention`, `train/val/run.py:264-345`)**:
   - Rather than recomputing an entire $O(V)$ or $O(T)$ partition sum from scratch for each candidate perturbation, `Noise` caches clean summary statistics ($Z = \sum_k e^{z_k}$ or attention `(out, lse)`) and evaluates single-coordinate / single-slot replacements in **$O(1)$ exact closed form**:
     $$\Delta L(z_j \to z_j') = \log\!\bigl(1 + p_j (e^{z_j' - z_j} - 1)\bigr) - \mathbb{I}[j = y](z_j' - z_j).$$

4. **Forward-Verified Structural Transplants with $P/Q$ Split Confirmation (`module head_edit`, `train/val/run.py:10272-10497`)**:
   - Every 25 updates in steps `[50, 450]`, splits the current batch into two disjoint halves: **Proposal $P$** and **Confirmation $Q$**.
   - Measures **exact forward dormancy** $\Delta L_{\text{rem}}(h; P)$ of every unit by mean-ablating it forward on $P$.
   - Generates candidate structural edits (cloning/splitting a high-impact donor unit $s$ with shifted parameters into the most dormant slot $d$) and scores candidates forward on $P$.
   - **Commits the edit (along with optimizer state) iff the winning candidate on $P$ ALSO strictly lowers forward loss on held-back confirmation half $Q$** ($L_Q^{\text{new}} < L_Q^{\text{base}}$), with a `protect` cooldown preventing thrashing.

5. **`ANVIL II` Dual-Rail Gram-Newton-Schulz Polar Optimizer (`class Anvil` / `BatchedAnvil`, `train/val/run.py:2135-2282`)**:
   - Maintains fast ($\beta_{\text{fast}} = 0.85$) and slow ($\beta_{\text{slow}} = 0.98$) momentum rails, Nesterov-interpolates $M = \operatorname{lerp}(G, \omega F + (1-\omega) S, \beta_t)$, and orthogonalizes $M$ onto the Stiefel manifold via 6 quintic **Gram-Newton-Schulz** maps (`ANVIL_MAPS`, computing $R = X X^\top$ on the smaller dimension $\min(r, c)$!), followed by per-lane RMS energy equalization and norm restoration.

6. **In-Run Held-Out 3-Point Parabolic Trajectory Merge (`quad_merge`, `train/speedrun/run.py:3057-3214`; `soup_parts`, `train/val/run.py:2457-2625`)**:
   - Carves out a tiny disjoint subset of training rows (`HOLDOUT = 64 / 14648`, strictly zero training updates on them per commit `aa67d44`, and zero validation/test access!).
   - Tracks trajectory EMAs over the final $70\%$ of training (`MERGE_FROM = 0.3, MERGE_EVERY = 5`).
   - Groups parameters into functional parts (`merge_part`), evaluates forward loss on held-out training rows at $a \in \{c_0 - \Delta, c_0, c_0 + \Delta\}$, solves for each part's parabola vertex:
     $$a_p^* = \operatorname{clip}\!\left(c_0 - \frac{\Delta (L_+ - L_-)}{2 \max(L_- + L_+ - 2 L_0, 10^{-30})},\; a_{\min},\; a_{\max}\right),$$
     and selects the best global shrinkage factor $s \in \text{QUAD\_SHRINKS}$ on the held-out slice.

7. **Difficulty-Stratified Batching & Progressive `Wide-Step` Doubling (`difficulty_order` & `wide_rows`, `train/speedrun/run.py:148-151, 4116-4129`)**:
   - Ranks training rows by fast scalar surprisal, reshapes sorted indices into `(B, n // B)` strata, and permutes each stratum independently across steps so **every mini-batch draws exactly 1 sample from every quantile stratum**—suppressing mini-batch gradient variance.
   - After `wide_from = 0.1` of the phase, doubles mini-batch size (`8 -> 16`) with coupled rate multiplier (`1.3x–1.4x`), cutting step count and launch overhead while improving late-stage SNR.

---

## 2. Structural Isomorphism: Forward-Only Optimization <-> Hard-Assignment Clustering (`k-means` / `e-means++`)

Why does `e-means` / `e-means++` mirror `qlabs-eng/noise` so closely?

1. **Piecewise-Constant Non-Differentiable Hard Routing**:
   The $k$-means quantization loss is:
   $$\mathcal{L}(C; \mathcal{X}) = \frac{1}{n} \sum_{i=1}^n \min_{j \in \{1,\dots,k\}} \|x_i - c_j\|_2^2 = \frac{1}{n} \sum_{i=1}^n \|x_i - c_{a(x_i)}\|_2^2, \qquad a(x) = \arg\min_{j} \|x - c_j\|_2^2.$$
   Across $(d-1)$-dimensional Voronoi hyperplanes $H_{jm} = \{x : \|x - c_j\| = \|x - c_m\|\}$, the hard router $a(x)$ is a **discontinuous step function** with zero derivative almost everywhere ($\partial a(x) / \partial c_m = 0$ for $x \notin \bigcup H_{jm}$). Just like discrete/gated non-differentiable pathways in `Noise`:
   - **Local conditional mean updates are blind across Voronoi ridges**: Within fixed cells $V_j$, the gradient $\nabla_{c_j} \mathcal{L} = \frac{2 |V_j|}{n}(c_j - \bar{x}_{V_j})$ (and `e-means`'s EMA ratio $c_j = S_j / N_j$) acts purely as an **`Identity-Path` / `Boost` local linear readout**. If two centroids $c_1, c_2$ share one Gaussian cluster while $c_3$ straddles two separated clusters across a low-density ridge, **all three centroids can have $N_j \gg 0.1 \bar{N}$ and $\nabla_{c_j} \mathcal{L} = 0$**, permanently trapping Lloyd's algorithm, `e-means`, and `e-means++` in a suboptimal local minimum!
2. **Blind Heuristic Splits vs. Forward-Verified Structural Search**:
   `e-means` attempts to escape empty basins via count-threshold revival ($N_j < 0.1 \bar{N}$) with fixed `[+1, -1, ...]` `1e-4` checkerboard splits—never checking whether removing $c_d$ or splitting $c_s$ actually improves the forward objective, and never transplanting live redundant twins ($N_d \ge 0.1 \bar{N}$) whose removal SSE cost is negligible!
3. **Convex-Hull Inductive Bias vs. Matrix Preconditioning (`e-means` Appendix G vs. `ANVIL II`)**:
   Appendix G of Bessa et al. (`arXiv:2611.09501`) showed that unconstrained SGD/Adam collapses at large batch sizes because centroids leave the data convex hull, whereas `qlabs-eng/noise` showed `ANVIL II` polar orthogonalization dramatically accelerates forward-only matrix learning. Testing norm-matched, annealed `ANVIL II` polar updates directly probes whether singular-spectrum whitening helps anisotropic clustering or conflicts with Zador's $p_j \propto \rho(c)^{d/(d+2)}$ quantization density law!
4. **Trajectory Noise Floor & Binary Finalization Tradeoff**:
   Just as noisy forward probes leave a terminal loss floor in `Noise` (solved by `quad_merge` parabolic trajectory soup on held-out training rows), mini-batch EMA leaves a `closing_mean` (low train MSE, high finite-sample variance at small $n/k$) vs. `slow_ema` (lower test MSE, lag bias on train MSE) dichotomy in `e-means`.

| `qlabs-eng/noise` Concept | `e-means` / `e-means++` Clustering Isomorph |
| :--- | :--- |
| Clean forward pass + per-token CE loss | Forward nearest-centroid routing $a(x) = \arg\min_j \|x - c_j\|^2$ + squared error $d_1^2(x)$ |
| `Identity-Path` / linear writer outer product | Within-cell convex-hull EMA update $(N_j, S_j) \mapsto c_j = S_j / N_j$ |
| `coordinate_differences` / `frozen_attention` $O(1)$ partition swap | **Exact Top-3 Voronoi Partition Swap**: $d_{\text{new}}^2(x) = \min(d_{\text{rem}}^2(x), d_+^2(x), d_-^2(x))$ in $O(1)$ per point |
| `head_edit` $P/Q$ forward-verified head transplant + `hadamard` | **`N-Track-1`**: $P/Q$ forward-verified redundant-centroid transplant + $M$-probe signed-Hadamard donor split |
| `quad_merge` / `soup_parts` held-out parabolic merge (`aa67d44`) | **`N-Track-2`**: In-run held-out ($2.5\%$ of `x_train`) 3-point parabolic trajectory merge across usage-strata parts |
| `ANVIL II` dual-rail Gram-Newton-Schulz polar step | **`N-Track-3`**: Norm-matched dual-rail Gram-Newton-Schulz polar centroid displacement annealed into EMA |
| `Dust` / `EGGROLL` antithetic activation jitter + `hadamard` | **`N-Track-4`**: Zero-overhead ($1.00\times$ FLOPs) split-batch antithetic signed-Hadamard Voronoi ridge smoothing |
| `difficulty_order` stratified batches + `wide_rows` batch doubling | **`N-Track-5`**: Spatial-radial quantile-stratified mini-batching + late $2\times$ `wide_step` coupled-$\alpha$ schedule |

---

## 3. Five Pre-Registered, Falsifiable Forward-Only Experimental Tracks (`N-Track-1..5`)

All five tracks are implemented cleanly and modularly in `src/emeans/noise_tracks.py` (`EMeansNoiseTrack`) on top of both `e-means++` (`base_mode="emeans_pp"`) and random-init `e-means` (`base_mode="emeans"`), strictly disjoint from `src/emeans/algo_extensions.py` (`H-Algo-1..5`), and satisfy exact baseline parity when disabled.

### `N-Track-1`: `head_edit` Forward-Verified Signed-Hadamard Centroid Transplant
- **Source in `Noise`**: `module head_edit` (`train/val/run.py:10272-10497`), `hadamard()` (`train/val/run.py:284`), and exact partition-swap scoring (`train/val/run.py:264-345`).
- **Exact Equations**:
  Every `eff_transplant_every <= 8` updates during non-closing epochs ($t / M \le \rho_{\text{edit}} = 0.75$), split batch $\mathcal{B}$ into disjoint Proposal $P$ ($|P| = \lfloor B/2 \rfloor$) and Confirmation $Q$ ($\mathcal{B} \setminus P$). Compute exact represented-precision top-3 nearest centroids $(d_1^2, a_1), (d_2^2, a_2), (d_3^2, a_3)$ on $\mathcal{B}$.
  1. **Forward Dormancy + Multi-Pair Huygens Rare-Mode Shield**:
     $$D_P(j) = \sum_{x \in P:\, a_1(x)=j} \bigl(d_2^2(x) - d_1^2(x)\bigr), \qquad C_{\text{rem}}(j) = \max\!\Bigl(2 D_P(j),\; M_j^{\text{eff}} \min_{m \neq j} \|c_j - c_m\|_2^2\Bigr),$$
     where $M_j^{\text{eff}} = \max(N_j, \min(M_j^{\text{surv}}, \tau \bar{N}))$ prevents mini-batch Poisson gaps from falsely marking isolated rare modes as dormant, and $d_{\text{nn2}}$, $C_{\text{rem}}$, and the shield mask are refreshed immediately after each accepted edit so surviving rare-mode twins are instantly protected.
  2. **1D Conditional Half-Space Projections along Signed-Hadamard + Forward Residual Probes on $P$**:
     Pair lowest-$C_{\text{rem}}$ unprotected recipients $d$ with highest-$\text{SSE}_P$ donors $s$ ($n_{P,s} \ge 4, n_{Q,s} \ge 2$). Construct $M=8$ unit probe axes $\{h_m\}_{m=1}^M$ combining orthogonal signed Sylvester-Hadamard directions with Hadamard-seeded forward residual chords on $S_P = \{x_i \in P : a_1(x_i) = s\}$. Project centered donor points $p_{i,m} = \langle x_i - \bar{x}_{P,s}, h_m \rangle$ and form exact 1D conditional half-space centroids:
     $$\delta_m^+ = \operatorname{mean}_{p_{i,m} > 0}(p_{i,m}), \qquad \delta_m^- = \operatorname{mean}_{p_{i,m} \le 0}(p_{i,m}), \qquad c_s^{(m,\pm)} = \bar{x}_{P,s} + \delta_m^\pm h_m,$$
     automatically scaling step sizes as $O(R/\sqrt{d})$ on isotropic axes and $\pm \Delta$ on bimodal separated axes with zero high-dimensional step overshoot. Using exact exclusion distance
     $$d_{\setminus\{d,s\}}^2(x) = \begin{cases} d_1^2(x) & a_1(x) \notin \{d, s\}, \\ d_2^2(x) & a_1(x) \in \{d, s\} \land a_2(x) \notin \{d, s\}, \\ d_3^2(x) & \{a_1(x), a_2(x)\} = \{d, s\}, \end{cases}$$
     evaluate $\Delta \text{SSE}_P(m) = \sum_{x \in P} \bigl[\min(d_{\setminus\{d,s\}}^2(x), \|x - \hat{c}_s^{(m,+)}\|^2, \|x - \hat{c}_s^{(m,-)}\|^2) - d_1^2(x)\bigr]$.
  3. **Held-Back $Q$ Confirmation Gate**:
     For $m^* = \arg\min_m \Delta \text{SSE}_P(m)$, commit $(c_d, c_s) \leftarrow (c_s^{(m^*,+)}, c_s^{(m^*,-)})$ (splitting usage/distortion/tail state and setting a cooldown) **iff** $\Delta \text{SSE}_P(m^*) < 0$ **AND** $\Delta \text{SSE}_Q(m^*) < 0$.
- **Complexity Overhead**: $< 4\%$ wall-clock (runs every $\le 8$ non-closing steps; exact top-3 streaming pass + $(B, 16)$ probe GEMM).
- **Predicted Regimes & Falsification Criterion**:
  - *Win*: Clustered/imbalanced GMMs (`zipf_gmm`, `low_nk32_gmm`, `multiscale_rare_outliers`) and random-init `e-means`, where twin centroids get trapped in dense modes while merged donors straddle separated modes despite $N_d \ge 0.1 \bar{N}$.
  - *Neutral*: Continuous `isotropic_gaussian` (where Voronoi hole cost $\approx$ split gain, so $Q$-confirmation rejects overfitted proposals).
  - *Falsified if*: Paired 15-seed test MSE fails to improve ($p_{\text{wilcox}} \ge 0.05$ or mean $\Delta \ge 0$) on multimodal GMM benchmarks (`zipf_gmm`, `low_nk32_gmm`) or degrades rare-mode preservation on `multiscale_rare_outliers`.

### `N-Track-2`: `quad_merge` In-Run Held-Out Parabolic Trajectory Merge
- **Source in `Noise`**: `quad_merge` (`train/speedrun/run.py:3057-3214`), commit `aa67d44` (disjoint training-row merge holdout), and `soup_parts` (`train/val/run.py:2457-2625`).
- **Exact Equations**:
  1. Partition input training tensor `x` strictly internally into $X_{\text{hold}}$ ($n_{\text{hold}} = \min(4096, \lfloor 0.025 n \rfloor)$, zero access to `val`/`test`) and $X_{\text{fit}} = X \setminus X_{\text{hold}}$.
  2. From $30\%$ of updates onward (`MERGE_FROM = 0.3`), update two split-consistent trajectory EMAs (`tail_med` rate `0.05`, `tail_slow` rate `0.015`) every `5` updates.
  3. Pre-center $X_{\text{hold}}$ and candidate banks in exact FP32 (shift-invariant and ULP-smooth), bound each cluster's maximum trajectory blend by its finite-sample precision ceiling $\bar{a}_j = \frac{\kappa}{n_j + \kappa}$ ($\kappa = 64$, protecting populous Zipf head clusters from noisy holdout slopes), fit closed-form 3-point parabolic vertices at $(0, \Delta, 2\Delta)$ ($\Delta = 0.45$) per usage-quantile part $p \in \{1,\dots,\min(8,k)\}$:
     $$v_p = \operatorname{clip}\!\left(\Delta - \frac{\Delta \bigl(\mathcal{L}_p(2\Delta) - \mathcal{L}_0\bigr)}{2 \max\bigl(\mathcal{L}_0 + \mathcal{L}_p(2\Delta) - 2\mathcal{L}_p(\Delta),\; 10^{-15}\bigr)},\; 0,\; 1\right) \mathbb{I}[\hat{q}_p(v_p) < \mathcal{L}_0],$$
     select best global shrinkage $s^* \in \{0.0, 0.25, 0.5, 0.75, 1.0\}$ on $X_{\text{hold}}$, fold $X_{\text{hold}}$ sufficient statistics into $C_{\text{close}}(X)$, and output $c_j = (1 - s^* v_{p(j)} \bar{a}_j) c_j^{\text{close}}(X) + s^* v_{p(j)} \bar{a}_j T_{*,j}$.
- **Complexity Overhead**: $\sim 36$ centered FP32 assignment calls on $\le 4096$ points batched into 3 syncs ($\approx 1\text{--}3\text{ ms}$ on GPU, timed inside `train_time_sec`).
- **Predicted Regimes & Falsification Criterion**:
  - *Win*: Low-$n/k$ regimes (`low_nk32_gmm`, `cifar10_r18_512`, `zipf_gmm`) on **out-of-sample test MSE**, directly optimizing hard-Voronoi boundary interactions without analytical Gaussian assumptions.
  - *Tradeoff / Fail*: May slightly increase `train_mse` relative to pure `closing_mean` (since `closing_mean` is the exact sample mean on fixed final assignments) while beating `slow_ema` on `train_mse` and beating `closing_mean` on `test_mse`.
  - *Falsified if*: Paired 15-seed `test_mse` fails to beat `closing_mean` ($p_{\text{wilcox}} \ge 0.05$) across low-$n/k$ cells (`n/k < 100`).

### `N-Track-3`: `anvil_polar` ANVIL-II Dual-Rail Gram-Newton-Schulz Polar Centroid Update
- **Source in `Noise`**: `Anvil` / `BatchedAnvil` / `gram_newton_schulz` (`train/val/run.py:2135-2282`).
- **Exact Equations**:
  Given standard convex-hull EMA displacement $\Delta C^{\text{ema}} \in \mathbb{R}^{k \times d}$ ($\Delta c_j^{\text{ema}} = w_j (\bar{x}_{B,j} - c_j)$ on touched clusters, $0$ otherwise), update dual rails $F \leftarrow 0.85 F + 0.15 \Delta C^{\text{ema}}$, $S \leftarrow 0.98 S + 0.02 \Delta C^{\text{ema}}$ (resetting rows on splits), form $M = (1-\beta_t)\Delta C^{\text{ema}} + \beta_t (0.4385 F + 0.5615 S)$, and apply `Noise`'s 6-step quintic Gram-Newton-Schulz polar map on the smaller Gram dimension $\min(k, d)$ (`ANVIL_MAPS` via fused `addmm`), per-lane RMS normalization, and Frobenius norm matching $U = \widehat{P} \frac{\|\Delta C^{\text{ema}}\|_F}{\max(\|\widehat{P}\|_F, 10^{-20})}$ (row-clipped to $2 \|\Delta c_j^{\text{ema}}\|_2$). Anneal $\lambda(t) = \lambda_0 \cdot \frac{1}{2}(1 + \cos(\pi t / T_{\text{anvil}})) \to 0$ strictly before the closing pass ($T_{\text{anvil}} = \min(0.60, \text{pre\_close\_frac}) M$):
  $$c_j \leftarrow c_j + (1 - \lambda(t)) \Delta c_j^{\text{ema}} + \lambda(t) U_j \quad (n_j > 0).$$
- **Complexity Overhead**: $O(k d \min(k, d))$ per active pre-closing step (`min(k, d) <= 512`, zero host-device syncs).
- **Predicted Regimes & Falsification Criterion**:
  - *Win vs. Competing Theory*: Tests `Noise`'s matrix polar orthogonalization against `e-means` Appendix G (convex-hull optimality) and Zador's quantization density law. May accelerate early directional spread on strongly anisotropic spectra (`stretched_a2_t2`), **or** be falsified if whitening singular values pulls centroids away from high-density principal modes!
  - *Falsified if*: Paired 15-seed `test_mse` shows no statistically significant gain ($p_{\text{wilcox}} \ge 0.05$) on `stretched_a2_t2` / `cifar10` or degrades multimodal MSE via non-convex-hull drift.

### `N-Track-4`: `dust_antithetic` Zero-Overhead Split-Batch Antithetic Hadamard Voronoi Smoothing
- **Source in `Noise`**: `Dust` activation jitter (`train/val/dust.py:234-260`), `EGGROLL` antithetic pairs (`eggroll.py:138-155`), and `hadamard` frames.
- **Exact Equations**:
  For pre-closing steps $1 \le t < T_{\text{jit}} = \lfloor \min(0.55, \text{pre\_close\_frac}) M \rfloor$ (`not is_last_epoch`), anneal $\sigma(t) = \sigma_0 \cdot \frac{1}{2}(1 + \cos(\pi t / T_{\text{jit}}))$. Draw unit-coordinate-RMS signed Sylvester-Hadamard perturbations $E \in \{\pm 1\}^{k \times d}$ ($\|E_j\|_2 = \sqrt{d}$), scaled per cluster by local coordinate RMS spread $r_j = \sqrt{E_j^{\text{dist}} / (\max(N_j, 10^{-8}) \cdot d)}$:
  $$C^+ = C + \sigma(t) (r \odot E), \qquad C^- = C - \sigma(t) (r \odot E).$$
  Route $\mathcal{B}_+ = \mathcal{B}_{1:\lfloor B/2 \rfloor}$ against $C^+$ and $\mathcal{B}_- = \mathcal{B}_{\lfloor B/2 \rfloor + 1:B}$ against $C^-$, accumulating $n_j = n_j^+ + n_j^-$ and $s_j = s_j^+ + s_j^-$ into the exact convex-hull EMA update.
- **Complexity Overhead**: **$1.00\times$ exact assignment FLOPs** ($\frac{B}{2} k d + \frac{B}{2} k d = B k d$; only one extra $K \times D$ norm prep).
- **Predicted Regimes & Falsification Criterion**:
  - *Win*: Rugged empirical manifolds (`cifar10_r18_512`, `cifar10_patches_128`) and early-epoch / random-init clustering where Moreau-Nesterov ridge smoothing untangles shallow Voronoi corners with $O(\sigma^2)$ antithetic bias cancellation and zero interior contraction.
  - *Fail / Neutral*: Well-separated warm-started `e-means++` clusters where Voronoi ridges sit in empty space, or excessive $\sigma_0$ blurring sharp boundaries.
  - *Falsified if*: Paired 15-seed `test_mse` fails to improve ($p_{\text{wilcox}} \ge 0.05$) across real embedding and random-init benchmarks.

### `N-Track-5`: `speedrun_strat_wide` Spatial-Quantile Stratified Batching & Late `Wide-Step` Schedule
- **Source in `Noise`**: `difficulty_order` (`train/speedrun/run.py:4116-4129`) and `wide_rows` / `wide_lr_scale` (`train/speedrun/run.py:148-151`, `archive/10M/train/speedrun/run.py:1-5`).
- **Exact Equations**:
  1. **$O(n d)$ Radial-Surprisal + PC1 Snake Stratification**: Center $\tilde{X} = X - \bar{x}$, partition vectors into 16 equal-mass radial-energy quantile shells $\|\tilde{x}_i\|_2^2$ (matching `difficulty_order` surprisal quantiles in `Noise`), and order continuously within/across shells along the 3-step power-iterated principal axis $\langle \tilde{x}_i, v_1 \rangle$ with alternating snake parity (`< 0.1 ms` on GPU). Each epoch reshapes the sorted indices into $B_{\text{cur}}$ equal strata `(B_cur, S)` and permutes each stratum independently across the $S$ steps—cutting mini-batch radial variance by $>20\times$ and cluster batch-count variance while visiting every training row at most once per pass!
  2. **Late `Wide-Step` Doubling**: After `wide_from = 0.25` of total row budget, widen batch size $B_{\text{wide}} = 2 B$ with coupled `e-means` step scaling $\alpha_0^{\text{wide}} = \min(1, 2^{0.9003} \alpha_0)$ and row-progress cosine annealing $\text{frac} = \text{rows\_seen} / \text{total\_rows}$. (Also ablated separately as `strat_only` vs `wide_only` vs combined `strat_wide`.)
- **Complexity Overhead**: Stratified ordering adds $< 0.2\text{ ms}$ total per `fit()`; `wide_step` **reduces** step count by $\sim 37.5\%$, improving GPU kernel launch efficiency!
- **Predicted Regimes & Falsification Criterion**:
  - *Win*: `strat_only` reduces variance on imbalanced / multi-scale regimes (`zipf_gmm`, `multiscale_rare_outliers`, `cifar10`); `strat_wide` reduces wall-clock training time (`train_time_sec`) while preserving or improving MSE at matched epochs.
  - *Fail*: `wide_step` may slightly under-explore on random-init `e-means` at very small $T=2$ if fewer discrete revival checks (`every 5 steps`) execute.
  - *Falsified if*: `strat_only` fails to improve paired MSE ($p_{\text{wilcox}} \ge 0.05$), and `strat_wide` fails to achieve either lower MSE or faster wall-clock time at non-inferior MSE (`<= +0.05%`).

---

## 4. Empirical Phase 2A Falsification Results (`3,600` Runs across 12 Cells $\times$ 15 Paired Seeds $\times$ 20 Arms)

All five forward-only tracks were evaluated on the exact unified Phase 2A protocol (`scripts/run_phase2a_noise.py --dispatch-fleet`, sharded 1 cell/GPU across 12 Blackwell GPUs on 6 hosts; full artifacts in `outputs/phase2_algo/noise_tracks/{phase2a_noise_results.json,phase2a_noise_summary.md,plots/*}`). Every MSE is exact FP64 (`exact_mse_fp64`) over $100\%$ of `train` and held-out `test`, with shared per-seed `SeedingResult` for `e-means++`, untimed full-cell warmup, synchronized `time.perf_counter()` timers, and two-sided paired Wilcoxon signed-rank (`p_wilcox`) + Student-$t$ tests across 15 seeds (`1000..1014`).

### 4.1 Executive Verdict Matrix

| Track | Mechanism (`qlabs-eng/noise` Source) | Empirical Verdict | Key Paired 15-Seed Evidence (`Test MSE` & `Time`) |
| :--- | :--- | :--- | :--- |
| **`N-Track-1`** (`head_edit`) | $P/Q$ forward-verified signed-Hadamard + residual-chord transplant | **CONFIRMED** (`5/5` multimodal GMM cells on `e-means++`, $p \le 0.0125$; `0%` false-positive rate on `isotropic_ctrl`) | **`-5.81%`** (`multiscale_rare_t5` `e-means++`, `15/0/0`, $p=6.1\times 10^{-5}$; **`-6.09%`** on `e-means`), **`-4.18%`** (`multiscale_rare_t2`, `15/0/0`), **`-0.42%`** (`low_nk32_gmm_t3`, `15/0/0`), **`-0.18% / -0.17%`** (`zipf_gmm_t2/t5`); rare-mode recall **`49.80% -> 57.78%`**. Neutral on continuous `cifar10` / `isotropic_ctrl` (`Q/P = 0.0%`). |
| **`N-Track-2`** (`quad_merge`) | Disjoint $2.5\%$ `x_train` holdout 3-point parabolic part merge (`aa67d44`) | **PARTIALLY CONFIRMED** (Dominates `slow_ema` `12/12`; wins `cifar10` on `e-means` & `isotropic_ctrl`; **falsified** on synthetic low-$n/k$ `e-means++`) | Beats `emeans_pp_slow_ema` **`15/0/0` on `12/12` cells** (`-0.11%` to **`-8.38%`**; `-1.96%..-8.38%` on non-isotropic cells). Beats `closing_mean` on `e-means` `cifar10_r18_512_t2/t5` (`-0.12% / -0.07%`, $p \le 0.010$), `cifar10_patches_128_t5` (`-0.05%`, $p=0.010$), and `isotropic_ctrl_t5` (`-0.10% / -0.13%`, `15/0/0`). Falsified vs `emeans_pp_base` on `low_nk32_gmm_t3` (`-0.02%`, $p=0.72$) and `zipf_gmm_t5` (`+0.37%`) due to $2.5\%$ `x_train` holdout starvation on discrete clusters. |
| **`N-Track-3`** (`anvil_polar`) | `ANVIL II` dual-rail 6-step Gram-Newton-Schulz polar displacement | **FALSIFIED on anisotropic/real spectra; SURPRISE WIN on `zipf_gmm`** | Neutral on `stretched_a2_t2/t5` (`+0.01% / +0.03%`, $p \ge 0.52$) and `cifar10` (`+0.05%..+0.07%`), confirming `e-means` Appendix G & Zador's $\rho^{d/(d+2)}$ density law against singular-value whitening. **Unexpectedly wins `4/4` `zipf_gmm` arms** (`-0.25% / -0.26%` on `e-means++`, $p \le 0.026$; **`-0.45% / -0.56%`** `15/0/0` on `e-means`, $p=6.1\times 10^{-5}$) by equalizing step energy across sparse Zipf tail clusters! |
| **`N-Track-4`** (`dust_antithetic`) | Split-batch antithetic signed-Hadamard Voronoi ridge jitter (`1.00x` FLOPs) | **FALSIFIED** (`0/24` significant `test_mse` wins) | Statistically neutral across `22/24` `(cell, mode)` pairs (`-0.21%` to `+0.17%`, $p \ge 0.18$) with slight `+0.07% / +0.28%` regressions on `stretched_a2_t2` / `zipf_gmm_t5` (`e-means++`) and `+11%..+27%` split-batch kernel launch overhead. |
| **`N-Track-5`** (`speedrun_strat_wide`) | 16-shell radial-PC1 stratification (`N5a`) + late $2\times$ `wide_step` (`N5b` / `N5`) | **SPEEDUP CONFIRMED (`24/24` cells); `strat_only` MSE FALSIFIED** | `N5b_wide_only` cuts `train_time_sec` by **`-21.4%` to `-35.2%`** and `N5_strat_wide` by **`-10.6%` to `-30.8%`** (`total_time_sec` **`-4.5%` to `-20.7%`** on `e-means++`, **`-10.5%` to `-34.9%`** on `e-means`, `15/0/0` on all `24/24` pairs) at **statistically indistinguishable `test_mse`** on `10/12` `e-means++` cells! `N5a` (`strat_only`) MSE gain falsified (`11/12` neutral; `+0.30%` on `zipf_gmm_t5`). |
| **`Combo`** (`N1+N2+N5a`) | Forward-verified `head_edit` + `quad_merge` + quantile stratification | **CONFIRMED SYNERGY** on multimodal + real `d=512` embeddings | Wins `multiscale_rare_t2` (**`-5.03%`**, $p=1.2\times 10^{-4}$), `multiscale_rare_t5` (**`-3.32%`**, $p=4.3\times 10^{-4}$, rare recall `49.8% -> 54.8%`), `low_nk32_gmm_t3` (**`-0.40%`**, $p=0.015$), `cifar10_r18_512_t2` (**`-0.09%`** `e-means++`, **`-0.13%`** `e-means`, $p \le 0.0125$), `cifar10_r18_512_t5` (**`-0.08% / -0.10%`**, $p \le 0.018$), and `isotropic_ctrl_t5` (**`-0.10%`**, `15/0/0`). |

### 4.2 Key Mechanistic Takeaways Connecting `qlabs-eng/noise` and `e-means++`

1. **Why `head_edit`'s $P/Q$ Split Confirmation Transfers Cleanly to Discrete Voronoi Routing**:
   Across all 12 cells, Proposal half $P$ finds candidate splits with $\Delta \text{SSE}_P < 0$ on **$100\%$ of trials** (`16/16` at $T \le 3$, `24/24` at $T=5$)—demonstrating that greedy in-batch split heuristics inevitably overfit finite mini-batch noise. Held-back Confirmation half $Q$ acts as an exact zero-hyperparameter statistical filter: on `isotropic_ctrl_t5` and `cifar10_r18_512_t5`, $Q$ rejects **$100.0\%$ (`0.00 / 24.0`)** of proposals (`0` false transplants!), whereas on multimodal GMMs (`multiscale_rare`, `low_nk32_gmm`, `zipf_gmm`), $Q$ confirms **`23.1%–70.6%`** (`4.60–16.93` transplants/run), escaping local twin traps while multi-pair Huygens refresh (`_compute_partition_state`) improves rare-mode recall by up to **`+7.98` percentage points**.
2. **Why `Dust` Activation Jitter Fails on Converged Voronoi Partitions while `ANVIL` Only Helps Heavy-Tailed Imbalance**:
   In `Noise`, perturbing intermediate continuous hidden states (`Dust`) yields unbiased local linear reward correlation, and `ANVIL` polarizes unconstrained dense weight matrices. In $k$-means quantization, centroids live inside the data space itself (`Appendix G`): jittering $C \pm \sigma r_j E_j$ blurs clean Voronoi boundaries (contaminating neighbor means without discovering separated modes), and polar singular-value whitening fights Zador's $\rho(c)^{d/(d+2)}$ density allocation on unimodal anisotropic clouds—helping *only* when severe Zipfian count skew starves tail-cluster step magnitudes.
3. **Why `wide_step` Batch Doubling Transfers Directly (`-20%..-35%` Training Time)**:
   Coupling `Noise`'s late $2\times$ batch widening (`wide_from = 0.25`) with `e-means`'s empirical step-size scaling law ($\alpha_0^{\text{wide}} = \min(1, 2^{0.9003}\alpha_0)$) and accumulator unit rescaling preserves `e-means++` solution quality across `10/12` cells while eliminating $\sim 37.5\%$ of Python/CUDA step launches—yielding an immediate practical wall-clock win for warm-started `e-means++`.

