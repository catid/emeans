# Workspace Agent Guidelines (`emeans`)

## Standing Engineering Rules

### 1. Optimize and Validate Before Long-Running Experiments
It is always worth optimizing code and validating it before long-running experiments because skipping this step is a huge waste of time:

- **Profile hot paths first**: Benchmark per-arm / kernel latency on real production `(n, d, k)` shapes and project total runtime before launching any multi-seed sweep.
- **Eliminate bottlenecks before scaling**: Remove Python loops over `k`/steps, per-step `.item()` / `.cpu()` host-device sync stalls, redundant tensor allocations, and unvectorized operations first.
- **Validate numerical parity after optimizing**: Run `.venv/bin/pytest`, adversarial edge-shape/dtype checks, and a 1-seed end-to-end `--smoke` run *after* every optimization and *before* dispatching long-running experiments.

### 2. Always Double-Check for Bugs Before Running Experiments
Always double-check both library implementations (`src/emeans/`) and benchmark/runner scripts (`scripts/`) before running experiments:

- **Audit math, shapes, dtypes, and stability**: Verify mathematical formulas, normalization factors, tensor broadcasting shapes, reduction axes, mixed-precision casts (`fp32`/`bf16`/`fp16`/`fp8`/`fp64`), coordinate-shift stability, empty-slice guards, and `NaN`/`Inf` handling.
- **Audit experiment & timing hygiene**: Verify zero train/test data-split leakage, independent RNG seeding, isolated warmup passes, `torch.cuda.synchronize()` around GPU timers, de-confounded benchmark comparisons, and PID-temp + `os.replace` atomic JSON writes.
- **Re-verify after any edit**: Run tests + smoke checks before launching sweeps, and always re-check after any post-review edit or script change.

### 3. Periodically Run Bug and Regression Checks in Parallel
Do not rely solely on a single pre-launch check:

- **Parallel periodic audits**: Continuously run independent bug checks, regression tests, and adversarial randomized shape/dtype probes in parallel alongside ongoing development and long-running experiments.
- **Live output & log validation**: Periodically audit active experiment logs and newly written `outputs/**/*.json` artifacts in parallel for Python tracebacks, stalled shards, `NaN`/`Inf` values, schema/key errors, or metric anomalies.

### 4. Store All Data, Caches, and Scratch Files on `/home`
Root `/` disks on workstations and bare-metal servers (including `/tmp` and `/var/tmp`) are OS partitions not meant for datasets, environments, smoke outputs, or Triton/compiler caches:

- **Local `foureyes`**: Keep all datasets, `.venv`, and outputs under `/home/catid/emeans/...` and place all scratch files, smoke/fuzz runs, `TMPDIR`, and isolated `TRITON_CACHE_DIR` directories strictly under `/home/catid/.cache/emeans_tmp/...` (never `/tmp` or `/var/tmp`).
- **Remote fleet servers (`ripper`, `loki`, `work`, `acorn.stratabaremetal.com`, `cashew.stratabaremetal.com`)**: Keep all virtualenvs, datasets, logs, `TMPDIR`, and per-shard `TRITON_CACHE_DIR` directories strictly under `/home/catid/emeans_env/...` (e.g., `TMPDIR=/home/catid/emeans_env/tmp` and `TRITON_CACHE_DIR=/home/catid/emeans_env/tmp/triton_cache_${SESSION}`) and `/home/catid/emeans_runs/...`.
- **Socket & host safety**: Never delete `/tmp/ssh-*` (forwarded `SSH_AUTH_SOCK` sockets), `/tmp/tmux-*`, or non-`emeans` files; never modify `tmux` session `0`; and never access off-limits `walnut.stratabaremetal.com`.
