"""Empirical scaling laws and cosine step schedule for e-means.

Reproduces Section 3.3 (Equations 4, 6, 7) of:
  Swann Bessa, Pierre-Emmanuel Mazare, Gergely Szilvasy, Hugo Touvron,
  David Picard, Matthijs Douze, Herve Jegou,
  "e-means: outperforming k-means (in one-tenth the epochs)",
  arXiv:2611.09501 (2026).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Cosine schedule floor factor (epsilon in Eq. 4 of the paper)
ALPHA_MIN_FACTOR: float = 0.1

# Number of mini-batch updates between dead-cluster revival checks (Algorithm 1, line 9)
CHECK_UNUSED_EVERY: int = 5

# Initial EMA usage mass N_0 for random initialization (2**-16 in reference; 1e-10 in paper prose)
DEFAULT_INIT_USAGE: float = 2.0**-16

# Checkerboard coordinate split perturbation delta (Appendix J, Eq. 8)
SPLIT_EPSILON: float = 1e-4

# Revival threshold tau: revive clusters with N_j < tau * mean(N)
DEFAULT_UNUSED_THRESHOLD: float = 0.1

# Dimension thresholds for auto mode
NO_REVIVAL_MAX_D: int = 2
AUTO_QUALITY_MAX_D: int = 16

# Empirical power-law coefficients:
# B = C_B * k^b_k * n^b_n * T^b_T
BATCH_LAWS: dict[str, tuple[float, float, float, float]] = {
    "throughput": (0.035134, 0.134749, 0.876473, 0.813344),
    "quality": (0.00439175, 0.134749, 0.876473, 0.813344),
}

# Step-size coupling law:
# alpha_0 = C_a * k^a_k * B^a_B * n^a_n * T^a_T
ALPHA_LAW: tuple[float, float, float, float, float] = (
    0.8244,
    -0.4617,
    0.9003,
    -0.4498,
    -0.1057,
)


def round_power_of_two(value: float) -> int:
    """Round a positive value to the nearest power of two in log-space."""
    if value <= 1.0:
        return 1
    low = 1 << int(math.floor(math.log2(value)))
    high = low << 1
    return low if (value / low) < (high / value) else high


def compute_step_law(n: int, k: int, nepochs: float, batch_size: int) -> float:
    """Compute initial EMA step size alpha_0 from (n, k, T, B) via Equation 6."""
    c_a, a_k, a_b, a_n, a_t = ALPHA_LAW
    alpha0 = (
        c_a
        * (k**a_k)
        * (batch_size**a_b)
        * (n**a_n)
        * (nepochs**a_t)
    )
    return float(min(max(alpha0, 1e-4), 1.0))


def resolve_batch_law(batch_law: str, d: int) -> tuple[str, str]:
    """Resolve 'auto' batch law based on feature dimension d."""
    if batch_law == "auto":
        if d <= AUTO_QUALITY_MAX_D:
            return "quality", f"d <= {AUTO_QUALITY_MAX_D}"
        return "throughput", f"d > {AUTO_QUALITY_MAX_D}"
    if batch_law not in BATCH_LAWS:
        raise ValueError(f"Unknown batch_law={batch_law!r}; expected 'auto', 'throughput', or 'quality'")
    return batch_law, "explicit"


def resolve_unused_threshold(unused_threshold: float | None, d: int) -> float:
    """Resolve dead-cluster revival threshold tau based on dimension d."""
    if unused_threshold is None:
        return 0.0 if d <= NO_REVIVAL_MAX_D else DEFAULT_UNUSED_THRESHOLD
    tau = float(unused_threshold)
    if not (0.0 <= tau < float("inf")):
        raise ValueError("unused_threshold must be finite and >= 0.0")
    return tau


def compute_epoch_rows(n: int, nepochs: float) -> list[int]:
    """Partition total budget T*n into whole passes, optional fractional pass, and full closing pass."""
    if not math.isfinite(nepochs) or nepochs <= 0:
        raise ValueError("nepochs must be finite and > 0")
    whole = math.floor(nepochs)
    remainder = int(round((nepochs - whole) * n))
    if whole == 0:
        return [max(1, remainder)]
    rows = [n] * (whole - 1)
    if remainder > 0:
        rows.append(remainder)
    rows.append(n)
    return rows


def scheduled_alpha(
    alpha0: float,
    step: int,
    total_steps: int,
    alpha_min_factor: float = ALPHA_MIN_FACTOR,
) -> float:
    """Cosine-annealed step size alpha(t) from alpha_0 to epsilon * alpha_0 (Equation 4)."""
    frac = min(step, total_steps) / max(1, total_steps)
    cos_term = 0.5 * (1.0 + math.cos(math.pi * frac))
    return float(alpha0 * (alpha_min_factor + (1.0 - alpha_min_factor) * cos_term))


def epoch_seed(seed: int, epoch_idx: int) -> int:
    """Deterministic per-epoch shuffle seed."""
    return (int(seed) * 1000003 + int(epoch_idx)) % (2**32)


@dataclass(frozen=True)
class EMeansRecipe:
    """Resolved execution plan and scaling-law hyperparameters for an e-means run."""

    n: int
    d: int
    k: int
    nepochs: float
    batch_law: str
    batch_law_reason: str
    batch_size: int
    alpha0: float
    alpha_min_factor: float
    unused_threshold: float
    check_unused_every: int
    split_epsilon: float
    init_usage: float
    epoch_rows: tuple[int, ...]
    epoch_steps: tuple[int, ...]
    total_steps: int


def build_recipe(
    n: int,
    d: int,
    k: int,
    nepochs: float = 5.0,
    batch_law: str = "auto",
    batch_size: int | None = None,
    alpha0: float | None = None,
    alpha_min_factor: float = ALPHA_MIN_FACTOR,
    unused_threshold: float | None = None,
    check_unused_every: int = CHECK_UNUSED_EVERY,
    split_epsilon: float = SPLIT_EPSILON,
    init_usage: float = DEFAULT_INIT_USAGE,
) -> EMeansRecipe:
    """Resolve scaling-law batch size B, initial step alpha_0, and pass schedule."""
    if d < 1:
        raise ValueError(f"Dimension d={d} must be >= 1")
    if k < 1:
        raise ValueError(f"Number of clusters k={k} must be >= 1")
    if n < k:
        raise ValueError(f"Number of points n={n} must be >= number of clusters k={k}")
    if check_unused_every < 1:
        raise ValueError(f"check_unused_every={check_unused_every} must be >= 1")
    if not (init_usage > 0.0 and math.isfinite(init_usage)):
        raise ValueError(f"init_usage={init_usage} must be finite and > 0")
    resolved_law, reason = resolve_batch_law(batch_law, d)
    if batch_size is None:
        c_b, b_k, b_n, b_t = BATCH_LAWS[resolved_law]
        raw_b = c_b * (k**b_k) * (n**b_n) * (nepochs**b_t)
        b = round_power_of_two(raw_b)
        if resolved_law == "throughput":
            b = min(b, round_power_of_two(n / 2.0))
        else:
            b = min(b, max(1, n // 2))
        b = max(1, min(b, n))
    else:
        b = max(1, min(int(batch_size), n))
        reason = "manual"

    resolved_alpha0 = (
        compute_step_law(n, k, nepochs, b) if alpha0 is None else float(alpha0)
    )
    tau = resolve_unused_threshold(unused_threshold, d)
    ep_rows = tuple(compute_epoch_rows(n, nepochs))
    ep_steps = tuple((r + b - 1) // b for r in ep_rows)
    return EMeansRecipe(
        n=n,
        d=d,
        k=k,
        nepochs=float(nepochs),
        batch_law=resolved_law,
        batch_law_reason=reason,
        batch_size=b,
        alpha0=resolved_alpha0,
        alpha_min_factor=float(alpha_min_factor),
        unused_threshold=tau,
        check_unused_every=int(check_unused_every),
        split_epsilon=float(split_epsilon),
        init_usage=float(init_usage),
        epoch_rows=ep_rows,
        epoch_steps=ep_steps,
        total_steps=sum(ep_steps),
    )
