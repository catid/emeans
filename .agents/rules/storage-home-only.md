---
trigger: always_on
description: "Store all emeans datasets, virtualenvs, logs, TMPDIR, smoke/fuzz runs, and TRITON_CACHE_DIR strictly under /home/catid across all 6 approved servers — never on /, /tmp, or /var/tmp."
---

# Server Storage Hygiene: `/home` Only Across All Fleet Hosts (`emeans`)

Root `/` partitions on fleet servers (which back `/tmp` and `/var/tmp`) are OS disks not meant for datasets, virtualenvs, smoke artifacts, or Triton/compiler caches. Enforce these rules across all **6 approved hosts** (`foureyes`, `ripper`, `loki`, `work`, `acorn.stratabaremetal.com`, `cashew.stratabaremetal.com`; note `walnut.stratabaremetal.com` is strictly off-limits):

## 1. Approved `/home/catid` Storage Locations
- **Localhost (`foureyes`)**:
  - Workspace, `.venv`, & outputs: `/home/catid/emeans/`, `/home/catid/emeans/.venv`, `/home/catid/emeans/outputs/`.
  - Scratch, smoke/fuzz runs, `TMPDIR`, & isolated `TRITON_CACHE_DIR`: **`/home/catid/.cache/emeans_tmp/...`** (create with `mkdir -p /home/catid/.cache/emeans_tmp`) or the Antigravity conversation scratch directory (`/home/catid/.gemini/antigravity-cli/brain/<conversation-id>/scratch/`).
- **Remote Fleet Hosts (`ripper`, `loki`, `work`, `acorn.stratabaremetal.com`, `cashew.stratabaremetal.com`)**:
  - Environment & datasets: **`/home/catid/emeans_env/.venv`** and **`/home/catid/emeans_env/data/...`**.
  - Synced workspaces & shard outputs: **`/home/catid/emeans_runs/...`** (or `~/emeans/`).
  - Scratch, `TMPDIR`, & per-shard `TRITON_CACHE_DIR`: **`/home/catid/emeans_env/tmp`** and **`/home/catid/emeans_env/tmp/triton_cache_${SESSION}`** (always run `mkdir -p /home/catid/emeans_env/tmp` in remote `ssh`/`tmux` launch commands).

## 2. Zero Writes to `/`, `/tmp`, or `/var/tmp`
- **Never** use `/tmp` or `/var/tmp` for `TRITON_CACHE_DIR`, `TMPDIR`, `--out-dir`, `--data-dir`, smoke/fuzz tests, or shell stderr/log redirection in `src/emeans/`, `scripts/`, `tests/`, or ad-hoc commands.
- In standalone runner scripts, default `TRITON_CACHE_DIR` to `/home/catid/emeans_env/tmp/...` when `/home/catid/emeans_env` exists (remote hosts) and `/home/catid/.cache/emeans_tmp/...` otherwise (local `foureyes`).

## 3. Strict `/tmp` Socket & Multi-Project Safety
- Reading `/tmp/ssh-*/agent.*` to discover a live forwarded `SSH_AUTH_SOCK` is allowed, **never** delete `/tmp/ssh-*`, `/tmp/tmux-*`, X11 sockets, or any non-`emeans` files/directories in `/tmp` or `/var/tmp`.
- Never touch `tmux` session `0` on any host, and never connect to `walnut.stratabaremetal.com`.
