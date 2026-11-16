# Literature Survey: From Classical Vector Quantization to `e-means`, `e-means++`, and Beyond

_Comprehensive mathematical and algorithmic synthesis supporting `e-means` (Bessa et al., arXiv:2611.09501, 2026), `e-means++`, and Phase 2 falsifiable algorithmic extensions._

---

## Executive Synthesis

For nearly six decades, practitioners solving the Euclidean $k$-means quantization problem—finding centroids $\mathcal{C} = \{c_1, \dots, c_k\} \subset \mathbb{R}^d$ that minimize empirical distortion
$$
\mathrm{MSE}(X, \mathcal{C}) \;=\; \frac{1}{n}\sum_{i=1}^{n} \min_{j \in \{1,\dots,k\}} \|x_i - c_j\|_2^2
$$
on a static dataset $X = \{x_1, \dots, x_n\} \subset \mathbb{R}^d$—faced a stark dichotomy:

1. **Full-batch Lloyd's algorithm** (Lloyd, 1957/1982; Forgy, 1965) requires zero learning-rate tuning and decreases training MSE monotonically via exact block-Newton steps, but converges slowly (typically $25+$ full-dataset passes) and gets trapped in poor Voronoi local minima where merged mega-clusters coexist alongside starved or dead centroids.
2. **Stochastic & mini-batch $k$-means** (MacQueen, 1967; Bottou & Bengio, 1995; Sculley, 2010; Zhu et al., 2024) slash per-step latency by updating centroids on mini-batches of size $B \ll n$, yet historically converged to **strictly higher MSE** than Lloyd due to vanishingharmonic step schedules ($\eta_t = 1/v_j^{(t)}$), centroid staleness at large $B$, lack of dead-cluster revival, and absence of problem-adaptive hyperparameter rules.

**`e-means` (Bessa et al., 2026, arXiv:2611.09501)** resolved this 50-year speed-versus-quality dilemma by repurposing the **exponential moving average (EMA)** codebook update of Vector-Quantized Variational Autoencoders (van den Oord et al., 2017) for stationary clustering, augmenting it with four structural pillars:
- **Convex-hull-preserving EMA statistics** $(N_j, S_j)$ with near-zero initial mass ($N_j^{(0)} = 2^{-16}$ or $10^{-10}$), remaining unconditionally stable at large GPU-friendly batch sizes where SGD and Adam diverge;
- **Cosine step-size annealing** from peak $\alpha_0$ down to a non-zero floor $\varepsilon \alpha_0$ ($\varepsilon = 0.1$);
- **Periodic donor-splitting dead-cluster revival** every 5 updates whenever $N_j < \tau \bar{N}$ ($\tau = 0.1$);
- **Zero-tuning empirical power laws** coupling optimal step size $\alpha_0(k, B, n, T)$ and knee batch size $B(k, n, T)$, paired with task-specific finalization (`closing_mean` for train MSE, `slow_ema` for out-of-sample test MSE).

Yet deep examination of the surrounding literature reveals five concrete structural blind spots in both vanilla `e-means` and naive `k-means++` composition:
1. **Arbitrary split geometry**: `e-means` splits donor centroids along a fixed coordinate-alternating checkerboard vector $[+1, -1, +1, -1, \dots]$ rather than the cluster's principal variance axis (Su & Dy, 2007).
2. **Voronoi boundary bias under finite/unequal cluster sizes**: Nearest-centroid assignment ignores the finite-sample $n_j / (n_j + 1)$ marginal distortion correction of Hartigan & Wong (1979) and Telgarsky & Vattani (2010).
3. **Unextrapolated centroid drift in ultra-low-epoch budgets ($T \le 2$)**: Standard EMA tracks trailing weighted averages without Nesterov/Polyak lookahead or variance-adaptive step modulation.
4. **Hard Voronoi saddle barriers**: Hard top-1 assignment prevents centroids from feeling cross-cell attraction across empty or low-density saddle regions that Power $k$-means (Xu & Lange, 2019) crosses via smooth power-mean homotopy.
5. **Dichotomous train-vs-test finalization**: `closing_mean` overfits small clusters ($n_j \ll n/k$) while `slow_ema` incurs estimation lag on populous clusters ($n_j \gg n/k$), missing the Empirical-Bayes / James-Stein optimal shrinkage interpolant.

This survey reviews the four foundational pillars of the literature in full mathematical detail and maps their open intersections directly onto `e-means++` (Phase 1) and our five falsifiable algorithmic hypotheses (`H-Algo-1` through `H-Algo-5`, Phase 2).

---

## 1. Mini-Batch, Online, and EMA $k$-Means Dynamics

### 1.1 Sequential Stochastic Approximation: MacQueen (1967) and Robbins-Monro (1951)
MacQueen (1967) introduced the name "$k$-means" not for Lloyd's batch iteration, but for an **online sequential update**: drawing one sample $x_t$ at step $t$, finding $j^* = \arg\min_j \|x_t - c_j^{(t-1)}\|_2^2$, incrementing the cumulative cluster count $v_{j^*}^{(t)} = v_{j^*}^{(t-1)} + 1$, and updating:
$$
c_{j^*}^{(t)} \;=\; c_{j^*}^{(t-1)} + \frac{1}{v_{j^*}^{(t)}}\bigl(x_t - c_{j^*}^{(t-1)}\bigr).
$$
Every centroid $c_j^{(t)}$ is the exact unweighted arithmetic mean of all points ever assigned to it—including points assigned early in training when Voronoi boundaries were wildly mispositioned. Viewed as Robbins-Monro (1951) stochastic approximation on the piecewise-quadratic quantization potential $\Phi(\mathcal{C}) = \frac{1}{2}\mathbb{E}_x[\min_j \|x - c_j\|_2^2]$, the per-cluster learning rate $\eta_j^{(t)} = 1 / v_j^{(t)}$ satisfies the classic almost-sure convergence conditions:
$$
\sum_{t=1}^{\infty} \eta_j^{(t)} = \infty, \qquad \sum_{t=1}^{\infty} \bigl(\eta_j^{(t)}\bigr)^2 < \infty.
$$
However, harmonic decay $1/v_j^{(t)}$ suffers from **catastrophic early lock-in** ("historical inertia"): after the first pass over $n$ points, $v_j \approx n/k$, so subsequent passes move centroids by $O(k/n)$, freezing misplaced centers before global Voronoi equilibrium is reached.

### 1.2 Newton vs. Gradient Geometry: Bottou & Bengio (1995) and So, Mahajan, & Dasgupta (2022)
Bottou & Bengio (1995, NIPS 1994) provided the foundational second-order analysis comparing batch Lloyd $k$-means against online $k$-means on the quantization objective $E(\mathcal{C}) = \frac{1}{2}\sum_{i=1}^n \|x_i - c_{a(i)}\|_2^2$:
- **Gradient and Hessian structure**: Within a fixed assignment region where assignments $a(i)$ do not change, $E(\mathcal{C})$ is smooth and decoupled across clusters, with gradient and diagonal Hessian:
  $$
  \frac{\partial E}{\partial c_j} = \sum_{i : a(i)=j} (c_j - x_i) = n_j (c_j - \bar{x}_j), \qquad \frac{\partial^2 E}{\partial c_j \partial c_m^\top} = \delta_{jm} n_j I_d.
  $$
- **Batch Lloyd is exact Newton's method**: Multiplying the negative gradient by the inverse Hessian $(n_j I_d)^{-1}$ moves $c_j \leftarrow c_j - n_j^{-1} n_j(c_j - \bar{x}_j) = \bar{x}_j$ in a single step! Because partition boundaries shift when centroids move, global convergence is linear rather than quadratic, but intra-basin conditioning with respect to heterogeneous cluster masses $n_j$ is automatically normalized.
- **Online/SGD conditioning pathology**: Plain gradient descent $c_j \leftarrow c_j - \eta \sum_{x \in \mathcal{B}_j} (c_j - x) = c_j + \eta n_j (\bar{x}_j - c_j)$ multiplies the step by $n_j$. If $\eta n_j > 2$, the update overshoots the reflection of $c_j$ across $\bar{x}_j$ and diverges!
- **Finite-sample online convergence**: So, Mahajan, & Dasgupta (2022, AISTATS) proved rigorous non-asymptotic convergence rates for online $k$-means, showing that stochastic fluctuations decay at $O(1/t)$ locally within a Voronoi basin, while boundary-crossing discontinuities create flat plateaus and local traps unless step schedules balance exploration and variance decay.

### 1.3 Sculley (2010), Béjar (2013), Nested & Staleness-Reduced Mini-Batch $k$-Means
Sculley (2010, WWW) vectorized MacQueen's sequential rule over mini-batches $\mathcal{B}_t$ of size $B$: caching assignments across the batch first, then applying cumulative harmonic updates $v_j \leftarrow v_j + 1$, $c_j \leftarrow (1 - 1/v_j)c_j + (1/v_j)x$ for each $x \in \mathcal{B}_t$ assigned to $j$. While Sculley achieved orders-of-magnitude speedups on sparse web data, Béjar (2013) demonstrated systematically that Sculley's `MiniBatchKMeans` converges to noticeably worse MSE than Lloyd across cluster structures, especially as $k$ grows or clusters have unequal density.

Subsequent works attempted to narrow this gap within the cumulative-average paradigm:
- **Nested Mini-Batch $k$-Means (Newling & Fleuret, 2016, NeurIPS)** combined triangle-inequality distance bounds (Elkan, 2003) with nested subset progressions ($B_1 \subset B_2 \subset \dots \subset X$), dynamically expanding batch size when stochastic sampling variance exceeded centroid movement.
- **Streaming $k$-Means with Forgetfulness (Apache Spark, 2014; Bidaurrazaga et al., 2021, ICDM)** introduced a discount factor $\alpha_{\mathrm{forget}}$ on historical counts $N_j \leftarrow \gamma N_j + n_j$ to track non-stationary streams, but provided no scaling law for stationary datasets.
- **Staleness-Reduction Mini-Batch $k$-Means (`SRMBatch`, Zhu et al., 2024, IEEE TNNLS)** diagnosed **centroid staleness**: when $B$ is small, centroids unobserved in a mini-batch retain outdated positions relative to updated neighbors, whereas when $B$ is large, intra-batch assignments use a stale snapshot of $\mathcal{C}$. Zhu et al. introduced staleness-compensated weighting hybridized with periodic full/bounded refresh steps. However, as Bessa et al. (2026, Figure 2 & Appendix C) verified empirically, even validation-tuned Sculley and `SRMBatch` beat Lloyd at $T=1$ epoch only to plateau **above** Lloyd's converged MSE by $T=5\text{--}25$.

### 1.4 VQ-VAE EMA Codebooks and the Łańcucki et al. (2020) EMA-SGD Equivalence
In deep representation learning, van den Oord, Vinyals, & Kavukcuoglu (2017, NeurIPS, Appendix A.1)—building on soft/hard incremental EM (Nowlan, 1991; Neal & Hinton, 1998)—maintained discrete codebook vectors via decoupled exponential moving averages of cluster counts and vector sums:
$$
N_j^{(t)} \;=\; \gamma N_j^{(t-1)} + (1 - \gamma) n_j^{(t)}, \qquad S_j^{(t)} \;=\; \gamma S_j^{(t-1)} + (1 - \gamma) s_j^{(t)}, \qquad c_j^{(t)} \;=\; \frac{S_j^{(t)}}{N_j^{(t)}}.
$$
Neural audio and speech codecs—**Jukebox** (Dhariwal et al., 2020), **SoundStream** (Zeghidour et al., 2021), **EnCodec** (Défossez et al., 2022), **SpeechTokenizer** (Zhang et al., 2024), and **WavTokenizer** (Ji et al., 2025)—hardened this rule against codebook collapse by periodically replacing low-usage codes ($N_j < \tau$). Conversely, image/audio VQ-GANs such as ViT-VQGAN (Yu et al., 2022), RVQGAN (Kumar et al., 2023), and LlamaGen (Sun et al., 2024) abandoned EMA in favor of Adam/AdamW with $\ell_2$-normalized low-dimensional codes.

**Łańcucki et al. (2020, IJCNN, *"Robust Training of Vector Quantized Bottleneck Models"*)** established the exact algebraic bridge between EMA codebook updates and SGD. Rewriting $c_j^{(t)} = S_j^{(t)} / N_j^{(t)}$ in position-and-usage form with $\alpha = 1 - \gamma$ and batch cluster mean $\bar{x}_j^{(t)} = s_j^{(t)} / n_j^{(t)}$ (for $n_j^{(t)} > 0$):
$$
c_j^{(t)} \;=\; \frac{\gamma N_j^{(t-1)} c_j^{(t-1)} + \alpha n_j^{(t)} \bar{x}_j^{(t)}}{\gamma N_j^{(t-1)} + \alpha n_j^{(t)}} \;=\; c_j^{(t-1)} + w_j^{(t)} \bigl(\bar{x}_j^{(t)} - c_j^{(t-1)}\bigr), \qquad w_j^{(t)} \;=\; \frac{\alpha n_j^{(t)}}{(1 - \alpha) N_j^{(t-1)} + \alpha n_j^{(t)}}.
$$
Comparing this against mini-batch SGD on $\mathcal{L}(\mathcal{C}) = \frac{1}{B}\sum_{x \in \mathcal{B}} \|x - c_{a(x)}\|_2^2$, whose step is $c_j^{(t)} = c_j^{(t-1)} + 2 \eta_{\mathrm{SGD}} \frac{n_j^{(t)}}{B}(\bar{x}_j^{(t)} - c_j^{(t-1)})$, reveals three profound properties:
1. **Adaptive per-cluster preconditioning**: Near stationarity ($N_j^{(t-1)} \approx \mathbb{E}[n_j] = B p_j$ and $n_j^{(t)} \approx B p_j$), the effective interpolation weight simplifies to $w_j^{(t)} \approx \alpha$, independent of cluster probability $p_j$! Rare clusters ($p_j \ll 1/k$) and massive clusters ($p_j \gg 1/k$) advance at the **same normalized rate** $\alpha$ toward their conditional means, functioning as an online stochastic Newton method (matching Bottou & Bengio's inverse-Hessian $n_j^{-1}$ scaling without matrix inversion), whereas plain SGD steps scale as $p_j$, starving rare clusters while exploding on popular ones.
2. **Unconditional Convex-Hull Stability ($w_j^{(t)} \in [0, 1]$)**: Because $N_j^{(0)} > 0$ and $\alpha \in (0, 1]$, we strictly have $0 \le w_j^{(t)} \le 1$ for every batch count $n_j^{(t)}$ and every $\alpha \in (0, 1]$. By induction:
   $$
   c_j^{(t)} \;\in\; \mathrm{conv}\Bigl(\{c_j^{(0)}\} \cup \bigcup_{s \le t} \{x \in \mathcal{B}_s : a(x) = j\}\Bigr).
   $$
   As Bessa et al. (2026, Appendix E) proved empirically, when batch size $B$ increases to saturate GPU Tensor Cores ($B \in [2^{14}, 2^{21}]$), optimal learning rates must scale up steeply; SGD and Adam exceed their stability threshold $2 \eta n_j / B > 1$, ejecting centroids out of the data convex hull and causing catastrophic loss collapse, whereas EMA smoothly interpolates all the way to $\alpha = 1$ (exact batch Lloyd!).
3. **Scaling-Law Coupling (`e-means`, Bessa et al., 2026)**: Rather than treating $\gamma = 0.99$ as an inherited magic constant, `e-means` discovered across 74,629 runs that optimal initial step size $\alpha_0$ and $+0.2\%$-knee batch size $B$ follow universal power laws in $(k, n, T, B)$:
   $$
   B_{\mathrm{throughput}} = \mathrm{pow2}\!\left(0.035134 \cdot k^{0.134749} \, n^{0.876473} \, T^{0.813344}\right), \qquad \alpha_0 = \min\!\left(0.8244 \cdot k^{-0.4617} \, B^{0.9003} \, n^{-0.4498} \, T^{-0.1057},\; 1\right).
   $$
   Notice the physical dimensionlessness hidden inside $\alpha_0$: the exponents on $(B/n)$ and $k$ are approximately $+0.90$ vs $-0.45$ and $-0.46$, meaning $\alpha_0 \propto \frac{B}{\sqrt{n k}} T^{-0.11}$! When $B$ increases by $2\times$, each batch delivers $2\times$ less noisy cluster means and $2\times$ fewer updates per epoch, so $\alpha_0$ scales up nearly linearly ($B^{0.9003}$).

---

## 2. Seeding Theory, Greedy $D^2$-Sampling, and Lightweight Coresets

### 2.1 Arthur & Vassilvitskii (2007): $k$-Means++ and the High-Dimensional Shell Penalty
Arthur & Vassilvitskii (2007, SODA) showed that choosing $c_1 \sim \mathrm{Unif}(X)$ and iteratively sampling $c_t = x_i$ with probability proportional to $D^2(x_i) = \min_{1 \le m < t} \|x_i - c_m\|_2^2$ achieves expected potential:
$$
\mathbb{E}[\Phi(X, \mathcal{C}_{++})] \;\le\; 8(\ln k + 2)\,\Phi_{\mathrm{OPT}}(X).
$$
Despite its strong theoretical guarantee, composing standard $k$-means++ directly with fast GPU clustering exposes **three fundamental bottlenecks**:
1. **Sequential $O(n k d)$ pass complexity**: Standard $k$-means++ requires $k - 1$ sequential passes over all $n$ vectors. In Bessa et al. (2026, Table `init_scale`), full-dataset $k$-means++ at $n=10^6, k=16{,}384$ took **2,852–6,080 seconds** on 64 CPU cores—over **$100\times$ slower** than the entire 5-epoch `e-means` training run ($22\text{--}38\text{ s}$)!
2. **The $2\times$ High-Dimensional Spherical-Shell Penalty**: Because $k$-means++ selects **raw data vectors** $x_{i_j} \in X$ as initial centroids, each chosen seed inside a true cluster of mean $\mu_j$ and covariance $\Sigma_j$ satisfies:
   $$
   \mathbb{E}_{x, x' \sim P_j}\bigl[\|x - x'\|_2^2\bigr] \;=\; \mathbb{E}\|x - \mu_j\|_2^2 + \mathbb{E}\|x' - \mu_j\|_2^2 \;=\; 2\,\mathrm{tr}(\Sigma_j).
   $$
   In high dimension $d$, Gaussian/embedding mass concentrates tightly on the thin spherical shell of radius $\sqrt{\mathrm{tr}(\Sigma_j)} \approx \sqrt{d}\sigma$ around $\mu_j$. Raw $k$-means++ seeds therefore start at **$2\times$ the optimal within-cluster MSE** until at least one Voronoi averaging step pulls them from the shell to the interior mean $\mu_j$.
3. **Destructive Interaction with Vanilla `e-means` (`N_0 = 2^-16` + Count-Only Revival)**: In vanilla `e-means`, initial mass $N_j^{(0)} = 2^{-16}$ gives step 0 an effective weight $w_j^{(1)} = \frac{\alpha_0 n_j}{(1-\alpha_0)2^{-16} + \alpha_0 n_j} \approx 1.0$, immediately overwriting carefully seeded centroids with noisy mini-batch samples of size $B/k \approx 4\text{--}16$. Worse, count-only revival ($N_j < 0.1 \bar{N}$) **assassinates genuine rare/outlier clusters** ($0 < p_j < 0.1/k$) that $D^2$-sampling specifically discovered!

### 2.2 Bahmani et al. (2012): Scalable $k$-Means++ (`k-means||`)
Bahmani, Moseley, Vattani, Kumar, & Vassilvitskii (2012, PVLDB) parallelized $D^2$-sampling by running $O(\log \psi)$ rounds (typically $5\text{--}8$), independently oversampling $\ell = \Theta(k)$ points per round with probability $p_x = \min(1, \ell D^2(x) / \Phi_X(\mathcal{C}))$, accumulating $O(k \log \psi)$ weighted intermediate centers, and reclustering them into $k$ centers via weighted $k$-means++. While `k-means||` achieves an $O(\log k)$ approximation in a logarithmic number of passes, at large $k$ ($k \in [16{,}384, 65{,}536]$) maintaining distances against $O(k \log \psi)$ intermediate centers over 5–8 full passes over $n$ points costs **5–40 full-dataset assignment equivalents** ($51\text{--}369\text{ s}$ on GPU in cuVS/cuML, Bessa et al., 2026), still an order of magnitude slower than `e-means` itself.

### 2.3 Celebi et al. (2013) vs. Grunau et al. (2023): The Greedy $k$-Means++ Paradox
In each step of **Greedy $k$-means++** (proposed heuristically in Arthur & Vassilvitskii, 2007; popularized empirically by **Celebi, Kingravi, & Vela, 2013**, *Expert Systems with Applications*; adopted as the default in `scikit-learn` with $L = 2 + \lfloor \ln k \rfloor$ local candidates), the algorithm draws $L$ independent candidates from the $D^2$ distribution and greedily commits to whichever candidate minimizes the updated global potential $\Phi(X, \mathcal{C}_{t-1} \cup \{c_{\mathrm{cand}}\})$.

For a decade, whether Greedy $k$-means++ improved or degraded the worst-case $O(\log k)$ bound was open:
- **Bhattacharya, Eube, Röglin, & Schmidt (2020, ESA)** proved a surprising $\Omega(\ell \log k)$ lower bound showing greedy selection can be *worse* in adversarial configurations.
- **Grunau, Özüdoğru, Rozhoň, & Tětek (2023, SODA)** resolved the problem with a nearly tight bound: Greedy $k$-means++ is an **$O(\ell^3 \log^3 k)$-approximation** with a matching **$\Omega\!\left(\frac{\ell^3 \log^3 k}{\log^2(\ell \log k)}\right)$** adversarial lower bound!
- **Why Greedy $k$-means++ Dominates in Practice Despite Worse Adversarial Bounds**: In Grunau et al.'s adversarial lower-bound gadget, picking a center standing symmetrically *between* two unpicked clusters temporarily lowers global SSE slightly more than picking the true center of one cluster, starving later steps. On real high-dimensional embeddings, however, two failure modes of plain ($L=1$) $k$-means++ dominate instead: (i) accidentally picking an extreme outlier far out in the tail of an already-covered cluster, or (ii) picking a redundant point in an already-covered cluster. Evaluating $L = 2 + \lfloor \ln k \rfloor$ candidates acts as an **order-statistic filter** that rejects tail/redundant draws and picks interior cluster representatives. Bessa et al. (2026, Appendix G.2, Figure 13) confirmed this starkly across 640 runs: plain ($L=1$) Faiss $k$-means++ had *null-to-negative* benefit on Lloyd's converged MSE, whereas Greedy $k$-means++ markedly lowered converged distortion on every corpus!

### 2.4 Bachem et al. (2016, 2018): MCMC Seeding and Lightweight Coresets
To decouple seeding complexity from dataset size $n$:
- **AFK-$\mathrm{MC}^2$ (Bachem, Lucic, Hassani, & Krause, 2016, NeurIPS)** replaced sequential full-dataset $D^2$ passes with short Metropolis-Hastings Markov chains of length $m_{\mathrm{chain}}$ initialized from a uniform/importance proposal $q(x)$ computed in one preprocessing pass:
  $$
  q(x_i) \;=\; \frac{1}{2n} + \frac{\|x_i - \mu\|_2^2}{2 \sum_{l=1}^{n} \|x_l - \mu\|_2^2}, \qquad \mu = \frac{1}{n}\sum_{l=1}^{n} x_l.
  $$
  After one $O(n d)$ scan to compute $q$, seeding takes $O(m_{\mathrm{chain}} k^2 d)$ time independent of $n$. However, for large $k \ge 16{,}384$, the $O(k^2 d)$ term remains quadratic in $k$.
- **Lightweight Coresets (Bachem, Lucic, & Krause, 2018, KDD)** proved that drawing $m$ points i.i.d. from the **exact same $O(n d)$ proposal $q(x_i)$** and assigning unbiased importance weights $w_i = \frac{1}{m q(x_i)}$ yields an $(\varepsilon, \delta)$ **lightweight coreset** $(X_{\mathrm{sub}}, w)$ satisfying, uniformly over all $k$-center sets $\mathcal{C} \subset \mathbb{R}^d$:
  $$
  \left|\frac{1}{n}\sum_{i=1}^n d^2(x_i, \mathcal{C}) - \sum_{r=1}^m \tilde{w}_r d^2(x_{i_r}, \mathcal{C})\right| \;\le\; \frac{\varepsilon}{2}\,\frac{1}{n}\sum_{i=1}^n d^2(x_i, \mathcal{C}) + \frac{\varepsilon}{2}\,\frac{1}{n}\sum_{i=1}^n \|x_i - \mu\|_2^2
  $$
  whenever $m = \tilde{O}(d k / \varepsilon^2)$. Crucially, the $\frac{1}{2}\frac{\|x_i - \mu\|_2^2}{\sum_l \|x_l - \mu\|_2^2}$ term oversamples far-away rare outlier clusters proportional to their squared norm from the global mean, while the $\frac{1}{2n}$ uniform component protects dense central modes!

> **Direct Bridge to `e-means++` (Phase 1)**: Combining a Lightweight Coreset ($m = \Theta(k)$) with GPU Block-Deduplicated / Greedy $k$-means++, **1-Step Coreset Voronoi Polish** (eliminating the $2\times$ $\sqrt{d}\sigma$ shell penalty at $O(m k d)$ sub-epoch cost), prior-calibrated initial EMA mass $N_j^{(0)}$, and **Huygens $D^2$-protected revival** resolves every incompatibility between $k$-means++ and `e-means`.

---

## 3. Escaping Local Minima in $k$-Means: Exchanges, Splits, and Landscape Smoothing

Even with $k$-means++ seeding, standard Voronoi descent frequently stalls in suboptimal local minima characterized by **macro-structural imbalances**: two centroids trapped sharing one true cluster ("merged/twin centroids") while another centroid spans two distant clusters ("under-segmented mega-cluster") (Fränti & Sieranoja, 2018, 2019). Five distinct algorithmic families address this trap:

### 3.1 Hartigan & Wong (1979) and Telgarsky & Vattani (2010): Exact Finite-Sample Marginal Distortion
Lloyd's algorithm assigns point $x$ to $\arg\min_j \|x - c_j\|_2^2$, treating centroids $c_j$ as **fixed** during the assignment step. **Hartigan (1975)** and **Hartigan & Wong (1979, JRSS-C Algorithm AS 136)** asked a deeper question: *what is the exact change in total Sum of Squared Errors (SSE) when a single point $x$ is removed from cluster $i$ (of size $n_i > 1$) and inserted into cluster $j$ (of size $n_j$), simultaneously recomputing both cluster means?*

By elementary algebra (parallel-axis / Huygens identity):
- **Adding $x$ to cluster $j$** shifts its mean from $c_j$ to $c_j^+ = \frac{n_j c_j + x}{n_j + 1}$, increasing cluster $j$'s SSE by:
  $$
  \Delta \mathrm{SSE}_+(x \to j) \;=\; \frac{n_j}{n_j + 1}\,\|x - c_j\|_2^2.
  $$
- **Removing $x$ from cluster $i$** ($n_i > 1$) shifts its mean from $c_i$ to $c_i^- = \frac{n_i c_i - x}{n_i - 1}$, decreasing cluster $i$'s SSE by:
  $$
  \Delta \mathrm{SSE}_-(x \leftarrow i) \;=\; \frac{n_i}{n_i - 1}\,\|x - c_i\|_2^2.
  $$
Thus, transferring $x$ from $i$ to $j$ strictly decreases global SSE if and only if:
$$
\frac{n_j}{n_j + 1}\,\|x - c_j\|_2^2 \;<\; \frac{n_i}{n_i - 1}\,\|x - c_i\|_2^2.
$$

**Telgarsky & Vattani (2010, AISTATS, *"Hartigan's Method: k-means Clustering without Voronoi"*)** proved that:
1. Every Hartigan local optimum is a strict Voronoi partition, **and** the set of Hartigan local optima is a **strict subset** of Lloyd local optima—Hartigan's rule escapes degenerate/asymmetric Lloyd fixed points!
2. Geometrically, the boundary between cluster $i$ (small $n_i$) and cluster $j$ (large $n_j$) is no longer the Euclidean perpendicular bisector ($\|x - c_j\|^2 = \|x - c_i\|^2$), but an **Apollonius sphere ("circlonoi" cell)** defined by $\frac{n_j}{n_j + 1}\|x - c_j\|^2 = \frac{n_i}{n_i - 1}\|x - c_i\|^2$. Because $\frac{n_i}{n_i + 1} < \frac{n_j}{n_j + 1}$ whenever $n_i < n_j$, **smaller clusters exert stronger per-point pull near boundaries** because absorbing $x$ pulls a small cluster's centroid toward $x$ by $\frac{1}{n_i + 1}$, absorbing part of $\|x - c_i\|^2$!
3. When $n_j \to \infty$ ($n/k \gg 10^3$), $\frac{n_j}{n_j + 1} \to 1$ and Hartigan collapses onto Lloyd; when $n/k$ is small-to-moderate ($n/k \in [8, 128]$, e.g., $n=10^6, k=16{,}384 \implies n/k = 61$), the factor $\frac{n_j}{n_j + 1}$ varies substantially between sparse clusters ($n_j = 5 \implies 0.833$) and dense clusters ($n_j = 200 \implies 0.995$).

> **Direct Bridge to `H-Algo-2`**: In mini-batch EMA clustering, the effective sample size of cluster $j$ is tracked online via usage $N_j$ (scaled to dataset counts $\hat{n}_j = \frac{n}{\sum_m N_m} N_j$). Assigning via annealed Hartigan-corrected score $\frac{\hat{n}_j}{\hat{n}_j + \beta(t)} \|x - c_j\|_2^2$ directly tests whether circlonoi marginal distortion improves quantization when $n/k$ is small!

### 3.2 Utility-Guided Splitting & Removal: Fritzke's LBG-U (1997) and Breathing $k$-Means (2020)
**Fritzke (1997, *Neural Processing Letters*, "The LBG-U method for vector quantization")** identified why dead/starved-cluster heuristics that only trigger at $n_j = 0$ fail to escape non-empty twin-centroid traps: a redundant centroid sharing a cluster with a twin still captures $n_j > 0$ points, so count-zero revival never fires! Fritzke defined the **utility $U_j$** of centroid $c_j$ as the exact increase in SSE if $c_j$ were deleted and its points reassigned to their second-nearest centroids:
$$
U_j \;=\; \sum_{x \in A_j} \Bigl(\min_{m \neq j} \|x - c_m\|_2^2 - \|x - c_j\|_2^2\Bigr).
$$
Approximating $x \approx c_j$ via Huygens' parallel-axis theorem yields the zero-data-scan lower/leading approximation:
$$
U_j \;\approx\; |A_j| \cdot \min_{m \neq j} \|c_j - c_m\|_2^2,
$$
which is the exact theoretical justification for `e-means++`'s **Huygens rare-mode protection** ($N_j d_{\mathrm{nn}}^2(c_j)$): a genuine rare cluster ($N_j$ small) sitting far from all other centroids ($d_{\mathrm{nn}}^2(c_j)$ huge) has **high utility $U_j$** and must be protected, whereas a redundant/dead centroid near another center has tiny $U_j$ and should be relocated to split the cluster with highest distortion $E_{j'}$!

**Fritzke (2020, *"Breathing k-Means"*)** generalized LBG-U into synchronous "breathe-in / breathe-out" cycles: inserting $m$ new centroids alongside the $m$ highest-SSE clusters, running Lloyd to relax, and deleting the $m$ lowest-utility centroids via mutual-nearest-neighbor freeze protection.

### 3.3 Principal-Axis vs. Arbitrary Donor Splitting: Su & Dy (2007) `PCA-Part` and `Var-Part`
When splitting a high-distortion or high-mass donor cluster $j'$ into two children $(c_{j'}^-, c_{j'}^+)$, *along which unit direction $v \in \mathbb{S}^{d-1}$ should the split occur?*

- **What Faiss and `e-means` currently do (Appendix F.1 of Bessa et al., 2026)**: Perturb multiplicatively along a fixed coordinate-alternating checkerboard vector $s = [+1, -1, +1, -1, \dots]^\top$:
  $$
  c_{j'}^\pm \;=\; c_{j'} \odot (1 \pm \delta s), \qquad \delta = 10^{-4}.
  $$
  Notice TWO geometric flaws in $c_{j'} \odot (1 \pm \delta s)$:
  1. **Origin-dependent scaling & zero-coordinate blindness**: Because the perturbation is *multiplicative* in $c_{j'}$, if $c_{j'}$ lies near the origin ($c_{j'} \approx 0$, e.g., centered cluster!), $\|c_{j'}^+ - c_{j'}^-\|_2 \approx 0$; if $c_{j'}$ lies far from the origin along direction $u$, the perturbation vector $\delta (c_{j'} \odot s)$ points in a coordinate-reflected copy of $c_{j'}$'s global position rather than the cluster's internal spread!
  2. **Orthogonality in high dimension $d$**: Even for an isotropic additive random/checkerboard unit vector $\tilde{s}$, if donor $j'$ actually covers **two merged clusters** separated along inter-cluster axis $u^* \in \mathbb{S}^{d-1}$, then in dimension $d$, $|\langle \tilde{s}, u^* \rangle| \sim O(1/\sqrt{d})$! At $d = 1024$, an arbitrary split hyperplane cuts nearly **perpendicular** to the true inter-cluster axis ($\approx 88.2^\circ$ angle!), slicing both sub-clusters in half so the next mini-batch pulls both children almost identically unless tiny $O(1/\sqrt{d})$ asymmetry survives BF16 quantization!
- **Su & Dy (2007, *Intelligent Data Analysis*, `PCA-Part` and `Var-Part`)**: Su & Dy proved that for a cluster with covariance $\Sigma_{j'}$, splitting by a hyperplane orthogonal to unit vector $v$ reduces within-cluster SSE in proportion to the projected variance $v^\top \Sigma_{j'} v$ (specifically $\frac{2}{\pi} v^\top \Sigma_{j'} v$ for Gaussian sub-clusters, or $\| \mu_1 - \mu_2 \|_2^2 / 4$ when $v \parallel (\mu_1 - \mu_2)$ for two merged clusters!).
  - **`PCA-Part`**: Splits along the top eigenvector $v_1(\Sigma_{j'})$, maximizing $v^\top \Sigma_{j'} v = \lambda_{\max}(\Sigma_{j'})$. If a donor covers two merged sub-clusters with centers $\mu_1, \mu_2$, $\Sigma_{j'} = \Sigma_w + \frac{1}{4}(\mu_1 - \mu_2)(\mu_1 - \mu_2)^\top$, so $v_1(\Sigma_{j'})$ aligns directly with the inter-cluster vector $\mu_1 - \mu_2$!
  - **`Var-Part`**: Approximates $\Sigma_{j'}$ by its diagonal $\mathrm{diag}(\sigma_{j',1}^2, \dots, \sigma_{j',d}^2)$ at $O(d)$ memory, splitting along the highest-variance coordinate(s).

> **Direct Bridge to `H-Algo-1`**: Without forming $O(k d^2)$ covariance matrices, we can maintain an **$O(k d)$ online Oja / Krasulina rank-1 principal residual vector** $p_j \in \mathbb{R}^d$ per cluster updated purely from mini-batch centered residuals $r_i = x_i - c_{a(i)}$! Splitting donors additively along $\pm \epsilon_{\mathrm{split}} \hat{p}_{j'}$ aligns the split hyperplane directly with $\lambda_{\max}(\Sigma_{j'})$ in $O(B d)$ time!

### 3.4 Random Swap (Fränti, 2018) and PNN-Smoothing (Baldassi, 2022)
- **Random Swap (Fränti, 2018, *Applied Intelligence*; Fränti & Sieranoja, 2019, *Pattern Recognition*)**: Repeatedly deletes a uniformly random centroid, inserts it at a random data point, and runs 2–5 local Lloyd iterations, accepting the swap only if global SSE strictly decreases. Over thousands of swaps, Random Swap finds near-global optima on low-$d$ clustered benchmarks, at the cost of hundreds/thousands of passes.
- **PNN-Smoothing (`pnns`, Baldassi, 2022, *TMLR*)**: Partitions $X$ into $J$ disjoint random subsets, clusters each into $k$ centers, pools the $J k$ centers weighted by cluster sizes, and hierarchically merges nearest pairs via **Pairwise-Nearest-Neighbor (PNN)** optimal merge cost $\Delta_{ab} = \frac{n_a n_b}{n_a + n_b}\|c_a - c_b\|_2^2$ down to $k$ centers. Notice Bessa et al.'s commented benchmark table (`experiments.tex`, lines 573–585): across 8 million-to-ten-million vector benchmarks, `e-means` beat both `PNN-smoothing` and `Lloyd + k-means++` on every single dataset!

### 3.5 Power $k$-Means: Homotopy Smoothing via Power Means (Xu & Lange, 2019)
**Xu & Lange (2019, ICML, PMLR 97)** embedded the non-smooth $\min_j \|x_i - c_j\|_2^2$ objective into a smooth homotopy family indexed by power exponent $s < 0$ using generalized Hölder power means $M_s(y_1, \dots, y_k) = \left(\frac{1}{k}\sum_{j=1}^k y_j^s\right)^{1/s}$:
$$
F_s(\mathcal{C}) \;=\; \sum_{i=1}^{n} M_s\bigl(\|x_i - c_1\|_2^2, \dots, \|x_i - c_k\|_2^2\bigr), \qquad \lim_{s \to -\infty} F_s(\mathcal{C}) \;=\; \sum_{i=1}^{n} \min_{j} \|x_i - c_j\|_2^2.
$$
Because $M_s(\mathbf{y})$ is concave in $\mathbf{y} \in \mathbb{R}_+^k$ for all $s \le 1$, linearizing $M_s$ at the current squared distances $d_{ij}^2 = \|x_i - c_j^{(m)}\|_2^2$ yields a strict **Majorization-Minimization (MM)** quadratic surrogate whose exact minimizer is a soft weighted mean:
$$
w_{ij}^{(m)} \;=\; \frac{\bigl(\|x_i - c_j^{(m)}\|_2^2 + \varepsilon\bigr)^{s - 1}}{\left(\frac{1}{k}\sum_{l=1}^k \bigl(\|x_i - c_l^{(m)}\|_2^2 + \varepsilon\bigr)^s\right)^{1 - 1/s}}, \qquad c_j^{(m+1)} \;=\; \frac{\sum_{i=1}^n w_{ij}^{(m)} x_i}{\sum_{i=1}^n w_{ij}^{(m)}}.
$$
(For $s = -1$, $M_{-1}$ is the harmonic mean, recovering Zhang, Hsu, & Dayal's 1999 $k$-Harmonic Means!) By initializing at moderate negative $s_0$ (e.g., $s_0 = -1$ or $-2$) and annealing $s_t \to -\infty$, centroids trapped in empty pockets still receive inverse-polynomial attraction $w_{ij} \propto (d_{ij}^2)^{s-1}$ from nearby clusters and smoothly cross Voronoi saddle barriers.

**Why full Power $k$-Means is $40\times$ too slow at scale—and how to make it $O(1)$-overhead (`H-Algo-4`)**: Full Power $k$-means materializes dense $B \times k$ weights and updates every centroid from every point. However, for $s \ll -1$ (or softmin temperature), $>99\%$ of a point's weight concentrates on its **top-2 nearest centroids** $(j_1, j_2)$! Tracking top-2 distances inside a single streaming pass over centroid blocks requires zero $B \times k$ memory, and annealing top-2 power-mean / softmin responsibility $(w_1, w_2)$ to $(1, 0)$ over the early warmup steps tests whether smooth cross-boundary attraction improves `e-means++` without sacrificing throughput!

### 3.6 Finite-Sample Estimation Variance & Empirical-Bayes Finalization (James & Stein, 1961; Efron & Morris, 1973)
Consider the finalization stage of `e-means` (Section 3.2 & Appendix H of Bessa et al., 2026) when $n/k$ is moderate or cluster sizes follow a heavy-tailed Zipf distribution ($n_j \in [2, 10^4]$):
- For cluster $j$ with true population Voronoi mean $\mu_j = \mathbb{E}_{x \sim P_j}[x]$, covariance $\Sigma_j$ ($\mathrm{tr}(\Sigma_j) = \sigma_j^2 d$), and $n_j$ training vectors assigned in the final epoch:
  - The **`closing_mean`** estimator $\hat{\mu}_j^{\mathrm{close}} = \frac{1}{n_j}\sum_{i \in A_j} x_i$ is the exact empirical risk minimizer on the fixed training partition $A_j$, achieving minimum training SSE $\sum_{i \in A_j} \|x_i - \hat{\mu}_j^{\mathrm{close}}\|_2^2$.
  - However, on **out-of-sample test queries** $x_{\mathrm{test}} \sim P_j$, expected squared error decomposes via bias-variance:
    $$
    \mathbb{E}\bigl[\|x_{\mathrm{test}} - \hat{c}_j\|_2^2\bigr] \;=\; \mathrm{tr}(\Sigma_j) + \mathbb{E}\bigl[\|\hat{c}_j - \mu_j\|_2^2\bigr].
    $$
    Because $\hat{\mu}_j^{\mathrm{close}}$ averages only $n_j$ points (and actually slightly overfits Voronoi boundary noise in high $d$), its estimation variance is $\mathbb{E}\|\hat{\mu}_j^{\mathrm{close}} - \mu_j\|_2^2 \approx \frac{\mathrm{tr}(\Sigma_j)}{n_j}$. For sparse tail clusters ($n_j = 4$), this adds **+25% excess test MSE**!
  - Conversely, **`slow_ema`** $\hat{\mu}_j^{\mathrm{slow}}$ integrates multi-epoch trajectory history with lower variance $\tau_{\mathrm{slow}}^2$, reducing test error on small clusters ($-0.082\%$ at $n/k=76.3$ in Table 7 of Bessa et al., 2026), **at the price of $+0.698\%$ worse train MSE** because `slow_ema` lags behind the exact closing mean on populous clusters ($n_j \gg 100$)!

By **Empirical-Bayes / James-Stein precision weighting** (James & Stein, 1961; Efron & Morris, 1973), shrinking the noisy closing mean $\hat{\mu}_j^{\mathrm{close}}$ (variance $\propto 1/n_j$) toward the smooth anchor $\hat{\mu}_j^{\mathrm{slow}}$ (prior effective sample size $\kappa > 0$):
$$
\hat{c}_j^{\mathrm{EB}}(\kappa) \;=\; \lambda_j \hat{\mu}_j^{\mathrm{close}} + (1 - \lambda_j) \hat{\mu}_j^{\mathrm{slow}}, \qquad \lambda_j \;=\; \frac{n_j}{n_j + \kappa},
$$
automatically gives $\lambda_j \to 1$ on populous clusters ($n_j \gg \kappa$, eliminating `slow_ema`'s lag and preserving exact train MSE!) while shrinking noisy tail clusters ($n_j \ll \kappa$, $\lambda_j \to 0$) toward $\hat{\mu}_j^{\mathrm{slow}}$ to suppress $1/n_j$ variance inflation on test data! Even better, $\kappa$ can be calibrated closed-form from the empirical discrepancy $\|\hat{\mu}_j^{\mathrm{close}} - \hat{\mu}_j^{\mathrm{slow}}\|_2^2$ via empirical Bayes!

---

## 4. Auxiliary-Loss-Free MoE Load Balancing & Capacity-Constrained Dual Routing vs. Distortion Minimization

### 4.1 Balanced Clustering & Inverted-File (IVF) Indexing Requirements
In Inverted File (IVF) approximate nearest neighbor search (Sivic & Zisserman, 2003; Jégou, Tavenard, Douze, & Amsaleg, 2011; Douze et al., 2024) and distributed expert sharding, skewed cluster sizes inflate tail query latency: if cluster sizes are $(n_1, \dots, n_k)$ with mean $\bar{n} = n/k$, scanning a random point's bucket touches an expected number of vectors proportional to the **Imbalance Factor**:
$$
\mathrm{IF} \;=\; \frac{1}{k}\sum_{j=1}^{k} \left(\frac{n_j}{\bar{n}}\right)^2 \;\ge\; 1.0,
$$
where $\mathrm{IF} = 1.0$ denotes perfectly equal cluster sizes (and multinomial uniform random routing achieves $\mathbb{E}[\mathrm{IF}] = 1 + \frac{k-1}{n}$).

Classical balanced $k$-means methods suffer severe scalability or quality bottlenecks:
- **Exact Balanced $k$-Means (Malinen & Fränti, 2014)**: Solves a linear assignment / minimum-cost bipartite matching problem via the Hungarian algorithm in $O(n^3)$ time per iteration—completely intractable beyond $n \approx 10^4$.
- **Penalty / Soft-Regularized Balanced $k$-Means (Tavenard et al., 2011; Liu et al., 2018; Li et al., 2018; De Maeyer et al., 2023)**: Adds a quadratic load penalty $\frac{\lambda}{2}\sum_{j=1}^k n_j^2$ directly to the clustering loss. Under sequential or soft updates, penalty gradients push popular centroids *away* from dense regions of data space to shed load, distorting centroid geometry.

### 4.2 DeepSeek's Auxiliary-Loss-Free MoE Routing (Wang et al., 2024) & Primal-Dual Theory (Han et al., 2025)
In Sparse Mixture-of-Experts (MoE) LLMs, **Wang, Gao, Zhao, Sun, & Dai (2024, arXiv:2408.15664, *"Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts"*)** showed that adding an auxiliary load-balancing loss $\mathcal{L}_{\mathrm{aux}}$ injects **interference gradients** into expert/router representations, degrading language modeling perplexity. Instead, Wang et al. left the training loss *completely untouched* and injected an expert-wise scalar bias $b_j$ **solely into the forward argmax/argmin routing decision**, updating $b_j$ online after each batch according to whether expert $j$ was under- or over-loaded:
$$
b_j \;\leftarrow\; b_j + u \cdot \mathrm{sign}(\bar{c} - n_j) \qquad \text{(Wang et al., 2024 sign controller)}.
$$

**Han et al. (2025, *"A Theoretical Framework for Auxiliary-Loss-Free Load Balancing of Sparse Mixture-of-Experts"*)**—and the capacity-constrained optimal transport formulation noted in `method.tex` of Bessa et al. (2026)—established *why* additive routing bias is not merely a heuristic, but the **exact Lagrangian dual** of balanced $k$-means! Consider the capacity-constrained assignment problem for fixed centroids $\mathcal{C}$:
$$
\min_{z_{ij} \in [0,1]} \sum_{i=1}^{n} \sum_{j=1}^{k} z_{ij} \|x_i - c_j\|_2^2 \qquad \text{s.t.} \qquad \sum_{j=1}^{k} z_{ij} = 1 \;\; \forall i, \qquad \sum_{i=1}^{n} z_{ij} = \frac{n}{k} \;\; \forall j.
$$
Introducing Lagrange dual multipliers (Kantorovich potentials) $b = (b_1, \dots, b_k) \in \mathbb{R}^k$ for the capacity constraints $\sum_i z_{ij} = n/k$ yields the dual-decoupled Lagrangian:
$$
\mathcal{L}(z, b) \;=\; \sum_{i=1}^{n} \sum_{j=1}^{k} z_{ij}\bigl(\|x_i - c_j\|_2^2 - b_j\bigr) + \frac{n}{k}\sum_{j=1}^{k} b_j.
$$
Minimizing over $z_{ij}$ pointwise decouples completely across data points into **Laguerre / Power-Diagram ("additively weighted Voronoi") routing**:
$$
a_b(x_i) \;=\; \arg\min_{j \in \{1,\dots,k\}} \Bigl(\|x_i - c_j\|_2^2 - b_j\Bigr),
$$
while dual ascent on $b_j$ updates each multiplier in proportion to constraint violation $(\bar{c} - n_j)$!

### 4.3 Fundamental Tradeoff: Dual-Biased Routing vs. Pure Distortion Minimization
Comparing capacity-constrained dual routing against pure distortion minimization (`e-means` / `e-means++`) highlights four sharp mathematical principles (also corroborated by the balanced-clustering notes in `method.tex` of Bessa et al., 2026):
1. **Zero Centroid Corruption**: Because $c_j$ is still updated as the exact conditional EMA mean $S_j / N_j$ of the points routed to cell $j$, each centroid remains optimal for its *Laguerre cell* $L_j(b) = \{x : \|x - c_j\|_2^2 - b_j \le \|x - c_m\|_2^2 - b_m \;\forall m\}$—eliminating the interference bias of penalty methods.
2. **Proportional Zero-Sum Dual Step vs. Sign Step**: Replacing Wang et al.'s $\mathrm{sign}(\bar{c} - n_j)$ with the scale-invariant proportional step $b_j \leftarrow b_j + u(t)\,\bar{D}\,\frac{\bar{c} - n_j}{\bar{c}}$ (where $\bar{c} = B/k$ and $\bar{D}$ is the running mean assigned squared distance) makes $u$ dimensionless across datasets of arbitrary coordinate scale and satisfies $\sum_{j=1}^k \Delta b_j = 0$, preventing gauge drift in $b$.
3. **Why Dual Balancing ALWAYS Costs Pure Euclidean MSE on Non-Uniform Data**: By Zador's theorem (Zador, 1982; Gersho, 1979) for high-dimensional vector quantization on density $p(x)$, the **MSE-optimal** point density of centroids scales as $\rho_{\mathrm{MSE}}^*(x) \propto p(x)^{\frac{d}{d+2}}$, implying cluster probability masses $p_j^* \propto p(c_j)^{\frac{2}{d+2}}$. As $d \to \infty$, $p(c_j)^{\frac{2}{d+2}} \to 1$ only on continuous densities, whereas on multimodal/Zipfian mixtures or manifolds of low intrinsic dimension $d_{\mathrm{int}} \ll d$ (Amsaleg et al., 2015), MSE-optimal Voronoi cells have unequal masses! Enforcing $n_j = n/k$ via nonzero $b_j$ shifts boundaries away from Euclidean bisectors, trading higher Euclidean distortion for $\mathrm{IF} \to 1.0$. Furthermore, `b_j` **must ship with the centroids at deployment** (`argmin_j (||x - c_j||^2 - b_j)`), because discarding $b_j$ at inference reverts to Euclidean Voronoi cells and destroys the learned load balance!
4. **Contrasting Hartigan's Bias (`H-Algo-2`) vs. Balanced Dual Bias**: Notice the striking sign contrast!
   - **Balanced MoE dual bias** *penalizes* large clusters ($n_j > \bar{c} \implies b_j$ decreases $\implies \|x - c_j\|^2 - b_j$ increases) linearly without bound until $n_j = \bar{c}$, sacrificing MSE to force $\mathrm{IF} \to 1$.
   - **Hartigan's exact SSE criterion** weights *multiplicatively* by $\frac{n_j}{n_j + 1} \in [\frac{1}{2}, 1)$, which differs from $1$ only by $O(1/n_j)$—exact for **minimizing Euclidean SSE** rather than equalizing cluster counts!

---

## 5. Summary Matrix & Phase 2 Falsifiable Algorithmic Roadmap

| Literature Thread | Key References | Core Mathematical Mechanism | Blind Spot in Vanilla `e-means` / `e-means++` | Phase 2 Falsifiable Hypothesis |
| :--- | :--- | :--- | :--- | :--- |
| **Principal-Axis Divisive Splitting** | Su & Dy (2007) `PCA-Part` / `Var-Part`; Fritzke (1997) `LBG-U` | Split donor along top eigenvector $v_1(\Sigma_j)$ maximizing $\Delta \mathrm{SSE} \propto v^\top \Sigma_j v$ | Multiplicative checkerboard $c_j \odot (1 \pm \delta s)$ is nearly orthogonal ($O(1/\sqrt{d})$) to merged inter-cluster axes in high $d$ and collapses near $c_j \approx 0$ | **`H-Algo-1`**: Online Oja rank-1 principal-axis donor splitting vs. checkerboard split |
| **Circlonoi / Marginal SSE Exchange** | Hartigan & Wong (1979); Telgarsky & Vattani (2010) | Exact 1-point transfer reduces SSE iff $\frac{n_j}{n_j+1}\|x-c_j\|^2 < \frac{n_i}{n_i-1}\|x-c_i\|^2$ | Standard Euclidean Voronoi assignment ignores finite-$n_j$ centroid recoil when $n/k$ is small ($n/k \in [16, 128]$) | **`H-Algo-2`**: Annealed Hartigan size-corrected marginal distortion assignment |
| **Stochastic Momentum & Preconditioning** | Polyak (1964); Nesterov (1983); Bottou & Bengio (1995); Łańcucki et al. (2020) | Lookahead extrapolation $c_j + \beta(c_j^{(t)} - c_j^{(t-1)})$ accelerates ballistic drift in early passes | Trailing EMA lags behind moving Voronoi barycenters during rapid early-pass drift ($T \le 2$) | **`H-Algo-3`**: Annealed Nesterov centroid lookahead / drift-adaptive EMA step |
| **Power-Mean Homotopy Smoothing** | Xu & Lange (2019) Power $k$-Means; Zhang et al. (1999) KHM | MM weights $w_{ij} \propto (d_{ij}^2)^{s-1}$ ($s \to -\infty$) smooth non-convex Voronoi saddle barriers | Hard top-1 assignment gives zero cross-cell pull across saddle barriers; full $B \times k$ Power $k$-means is $40\times$ slower | **`H-Algo-4`**: Memory-free Top-2 Power-Mean soft EMA warmup annealing |
| **Empirical-Bayes Shrinkage Estimation** | James & Stein (1961); Efron & Morris (1973); Bessa et al. (2026) App. H | Optimal MSE estimator shrinks noisy sample mean ($\mathrm{Var} \propto 1/n_j$) toward low-variance prior anchor | `closing_mean` overfits small clusters on test MSE; `slow_ema` lags large clusters on train MSE | **`H-Algo-5`**: Empirical-Bayes / James-Stein variance-shrinkage finalization $\lambda_j = \frac{n_j}{n_j + \kappa^*}$ |

---

## Bibliography

1. **Amsaleg, L., Chelly, O., Furon, T., Girard, S., Houle, M. E., Kawarabayashi, K., & Nett, M.** (2015). Estimating local intrinsic dimensionality. *ACM SIGKDD*, 29–38.
2. **Apache Spark** (2014). `StreamingKMeans`, Apache Spark MLlib (SPARK-3254).
3. **Arthur, D., & Vassilvitskii, S.** (2007). `k-means++`: The advantages of careful seeding. *ACM-SIAM SODA*, 1027–1035.
4. **Bachem, O., Lucic, M., Hassani, H., & Krause, A.** (2016). Fast and provably good seedings for $k$-means (`AFK-MC^2`). *NeurIPS*, 29.
5. **Bachem, O., Lucic, M., & Krause, A.** (2018). Scalable $k$-means clustering via lightweight coresets. *ACM SIGKDD* / *arXiv:1702.08248*.
6. **Bahmani, B., Moseley, B., Vattani, A., Kumar, R., & Vassilvitskii, S.** (2012). Scalable $k$-means++ (`k-means||`). *PVLDB*, 5(7), 622–633.
7. **Baldassi, C.** (2022). Systematically and efficiently improving $k$-means initialization by pairwise-nearest-neighbor smoothing (`PNN-smoothing`). *Transactions on Machine Learning Research (TMLR)*.
8. **Béjar, J.** (2013). $K$-means vs Mini Batch $K$-means: A comparison. Technical Report LSI-13-8-R, UPC BarcelonaTech.
9. **Bessa, S., Mazaré, P.-E., Szilvasy, G., Touvron, H., Picard, D., Douze, M., & Jégou, H.** (2026). `e-means`: Outperforming $k$-means (in one-tenth the epochs). *arXiv:2611.09501*.
10. **Bhattacharya, A., Eube, J., Röglin, H., & Schmidt, M.** (2020). Noisy, greedy and not so greedy $k$-means++. *ESA*, LIPIcs 173.
11. **Bidaurrazaga, A., Pérez, A., & Capó, M.** (2021). $K$-means for evolving data streams. *IEEE ICDM*.
12. **Bottou, L., & Bengio, Y.** (1995). Convergence properties of the $K$-means algorithms. *Advances in Neural Information Processing Systems (NIPS 1994)*, 7, 585–592.
13. **Celebi, M. E., Kingravi, H. A., & Vela, P. A.** (2013). A comparative study of efficient initialization methods for the $k$-means clustering algorithm. *Expert Systems with Applications*, 40(1), 200–210.
14. **De Maeyer, R., et al.** (2023). Balanced $k$-means revisited. *Applied Sciences* / *arXiv*.
15. **Défossez, A., Copet, J., Synnaeve, G., & Adi, Y.** (2022). High fidelity neural audio compression (`EnCodec`). *arXiv:2210.13438* / *TMLR 2023*.
16. **Dhariwal, P., Jun, H., Payne, C., Kim, J. W., Radford, A., & Sutskever, I.** (2020). Jukebox: A generative model for music. *arXiv:2005.00341*.
17. **Ding, Y., Zhao, Y., Shen, X., Musuvathi, M., & Mytkowicz, T.** (2015). Yinyang $K$-Means: A drop-in replacement of the classic $K$-means with consistent speedup. *ICML*, PMLR 37, 579–587.
18. **Douze, M., Guzhva, A., Deng, C., Johnson, J., Szilvasy, G., Mazaré, P.-E., Lomeli, M., Hosseini, L., & Jégou, H.** (2024). The Faiss library. *IEEE Transactions on Big Data*, 12, 346–361.
19. **Efron, B., & Morris, C.** (1973). Stein's estimation rule and its competitors—an empirical Bayes approach. *Journal of the American Statistical Association*, 68(341), 117–130.
20. **Elkan, C.** (2003). Using the triangle inequality to accelerate $K$-means. *ICML*, 147–153.
21. **Fränti, P., & Sieranoja, S.** (2018). $K$-means properties on six clustering benchmark datasets (Random Swap). *Applied Intelligence*, 48(12), 4743–4759.
22. **Fränti, P., & Sieranoja, S.** (2019). How much can $k$-means be improved by using better initialization and repeats? *Pattern Recognition*, 93, 95–112.
23. **Fritzke, B.** (1997). The LBG-U method for vector quantization — an improvement over LBG inspired from neural networks. *Neural Processing Letters*, 5(1), 35–45.
24. **Fritzke, B.** (2020). Breathing $k$-Means. *arXiv:2006.15666*.
25. **Grunau, C., Özüdoğru, A. A., Rozhoň, V., & Tětek, J.** (2023). A nearly tight analysis of greedy $k$-means++. *ACM-SIAM SODA*, 1012–1070.
26. **Han, X., et al.** (2025). A theoretical framework for auxiliary-loss-free load balancing of sparse mixture-of-experts in large-scale AI models. *arXiv*.
27. **Hartigan, J. A., & Wong, M. A.** (1979). Algorithm AS 136: A $K$-means clustering algorithm. *JRSS Series C (Applied Statistics)*, 28(1), 100–108.
28. **James, W., & Stein, C.** (1961). Estimation with quadratic loss. *Proceedings of the Fourth Berkeley Symposium on Mathematical Statistics and Probability*, 1, 361–379.
29. **Kuffo, L., Hepkema, S., & Boncz, P.** (2026). Super $k$-means: A super-fast clustering library for high-dimensional vector embeddings.
30. **Łańcucki, A., Chorowski, J., Sanchez, G., Marxer, R., Chen, N., Dolfing, H. J., Khurana, S., Alumäe, T., & Laurent, A.** (2020). Robust training of vector quantized bottleneck models. *IJCNN*, IEEE.
31. **Lloyd, S. P.** (1982; Bell Labs memo 1957). Least squares quantization in PCM. *IEEE Transactions on Information Theory*, 28(2), 129–137.
32. **MacQueen, J.** (1967). Some methods for classification and analysis of multivariate observations. *Fifth Berkeley Symposium on Mathematical Statistics and Probability*, 1, 281–297.
33. **Malinen, M. I., & Fränti, P.** (2014). Balanced $k$-means for clustering. *Joint IAPR International Workshop on Structural, Syntactic, and Statistical Pattern Recognition*, 32–41.
34. **Neal, R. M., & Hinton, G. E.** (1998). A view of the EM algorithm that justifies incremental, sparse, and other variants. *Learning in Graphical Models*, 355–368.
35. **Newling, J., & Fleuret, F.** (2016). Nested mini-batch $K$-means. *NeurIPS*, 29.
36. **Nesterov, Y.** (1983). A method for solving the convex programming problem with convergence rate $O(1/k^2)$. *Soviet Mathematics Doklady*, 27, 372–376.
37. **Robbins, H., & Monro, S.** (1951). A stochastic approximation method. *Annals of Mathematical Statistics*, 22(3), 400–407.
38. **Sculley, D.** (2010). Web-scale $K$-means clustering. *WWW*, 1177–1178.
39. **So, G., Mahajan, G., & Dasgupta, S.** (2022). Convergence of online $k$-means. *AISTATS*, PMLR 151.
40. **Su, T., & Dy, J. G.** (2007). In search of deterministic methods for initializing $K$-means and Gaussian mixture clustering (`PCA-Part` & `Var-Part`). *Intelligent Data Analysis*, 11(4), 319–338.
41. **Telgarsky, M., & Vattani, A.** (2010). Hartigan's method: $k$-means clustering without Voronoi. *AISTATS*, JMLR W&CP 9, 820–827.
42. **van den Oord, A., Vinyals, O., & Kavukcuoglu, K.** (2017). Neural discrete representation learning (`VQ-VAE`). *NeurIPS*, 30.
43. **Wang, L., Gao, H., Zhao, C., Sun, X., & Dai, D.** (2024). Auxiliary-loss-free load balancing strategy for mixture-of-experts. *arXiv:2408.15664*.
44. **Xu, J., & Lange, K.** (2019). Power $k$-means clustering. *ICML*, PMLR 97, 6921–6931.
45. **Yang, S., et al.** (2026). Flash-KMeans: Fast and memory-efficient exact $K$-means. *arXiv:2603.09229*.
46. **Zador, P. L.** (1982). Asymptotic quantization error of continuous signals and the quantization dimension. *IEEE Transactions on Information Theory*, 28(2), 139–149.
47. **Zeghidour, N., Luebs, A., Omran, A., Skoglund, J., & Tagliasacchi, M.** (2021). SoundStream: An end-to-end neural audio codec. *IEEE/ACM TASLP*, 30, 495–507.
48. **Zhu, X., Sun, J., He, Z., Jiang, J., & Wang, Z.** (2024). Staleness-reduction mini-batch $k$-means. *IEEE Transactions on Neural Networks and Learning Systems*, 35(10), 14424–14436.
