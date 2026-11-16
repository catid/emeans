---
trigger: always_on
description: "Always optimize, validate, and double-check code and runner scripts for bugs before running experiments, and periodically run bug checks in parallel."
---

# Pre-Experiment Optimization, Bug Checking, and Parallel Verification

Follow these standing rules across all `emeans` modules, kernels, tests, and experiment scripts:

## 1. Optimize and Validate Before Long-Running Experiments
> It is always worth optimizing code and validating it before long-running experiments because skipping this step wastes massive compute and wall-clock time.

- **Profile before sweeping**: Measure per-arm / hot-path latency on target `(n, d, k)` production shapes and estimate total sweep runtime before launching multi-seed or multi-GPU experiments.
- **Eliminate bottlenecks first**: Remove Python loops over clusters/candidates/steps, per-iteration `.item()` / `.cpu()` host-device sync stalls, redundant allocations, and unvectorized PyTorch/Triton operations first.
- **Validate after every optimization**: Confirm numerical parity, pass `.venv/bin/pytest` plus randomized/adversarial shape & dtype checks, and run a 1-seed `--smoke` pass *after* optimizing and *before* launching any sweep.

## 2. Always Double-Check for Bugs Before Running Experiments
Never launch a benchmark or experiment sweep without a thorough pre-flight bug audit of both library code and runner scripts:

- **Math, shapes, dtypes, and edge cases**:
  - Verify objective/distortion math, normalization weights, sign conventions, and metric parity.
  - Check broadcasting shapes, reduction axes, index bounds, and non-multiple / prime `(B, K, D)` shapes.
  - Check mixed-precision casts (`fp32`, `bf16`, `fp16`, `fp8`, `fp64`), FP32 accumulation, coordinate-shift invariance, empty-cluster handling, zero-division guards, and `NaN`/`Inf` edge cases.
- **Data hygiene, RNG, timers, and atomic I/O**:
  - Verify strict train/val/test split isolation (zero leakage) and independent RNG seeds across runs/workers.
  - Isolate warmup outside timed regions, bracket GPU timers with `torch.cuda.synchronize()`, keep precision/CUDA-graph ablations de-confounded, and use PID-temp + `os.replace` atomic JSON writes.
- **Re-check after every edit**:
  - Run `.venv/bin/pytest` and a smoke check before every sweep, and re-audit + re-test after any post-review edit or optimization.

## 3. Periodically Run Bug and Regression Checks in Parallel
Run ongoing verification concurrently with development and active sweeps:

- **Parallel bug & regression sweeps**: Periodically run independent code audits, `.venv/bin/pytest` suites, and adversarial shape/dtype probes in parallel (via subagents, swarm verifiers/patrols, or scheduled tasks) while development and experiments proceed.
- **Parallel artifact & log checks**: Periodically inspect live experiment logs and generated JSON outputs (`outputs/`) for tracebacks, stalls, `NaN`/`Inf` entries, duplicate keys, or metric inconsistencies so issues are caught early.
