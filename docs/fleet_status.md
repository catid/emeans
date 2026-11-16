# Fleet & Remote-Compute Status Manifest (`emeans`)

_Last verified: 2026-11-16 UTC via non-hanging read-only `ssh -A -o BatchMode=yes -o ConnectTimeout=5` diagnostics (`scripts/check_fleet.sh`)._

## 1. Executive Summary & Critical Warnings

- **ALL 5 APPROVED REMOTE MACHINES ARE REACHABLE (`work`, `ripper`, `loki`, `acorn.stratabaremetal.com`, `cashew.stratabaremetal.com`)!**
  - **Key Unlock (`SSH_AUTH_SOCK` after `tmux` reattach)**: Existing shells on `foureyes` may inherit a stale `SSH_AUTH_SOCK`, while `tmux` session `0` holds the live forwarded agent socket (`export SSH_AUTH_SOCK="$(tmux show-environment -t 0 SSH_AUTH_SOCK | cut -d= -f2-)"` -> `/tmp/ssh-TSqv7B3fFq/agent.1208827`, key `mrcat@kuang3` ED25519). Exporting that live `SSH_AUTH_SOCK` enables **direct passwordless `ssh -A` + live agent forwarding (`ssh-add -l`) + `tmux` on all 5 approved remote machines**.
- **STRICT OFF-LIMITS HOST — DO NOT USE `walnut` (`walnut.stratabaremetal.com`)**:
  - Per explicit user instruction, **`walnut` (`walnut.stratabaremetal.com`) is STRICTLY OFF-LIMITS** and has been removed from the checked/available fleet list. **Never SSH to or run any process on `walnut.stratabaremetal.com` — use `cashew.stratabaremetal.com` instead!**
- **Localhost Identity (`foureyes`)**:
  - Current local machine is **`foureyes`** (`10.10.0.216/24` on `.lan`). None of `acorn`, `cashew`, `work`, `ripper`, or `loki` aliases to `foureyes`. Never SSH into `foureyes` or `10.10.0.216` thinking it is remote.
  - Local `foureyes` GPUs (`0..3`, 4x RTX PRO 6000 Blackwell Max-Q 96 GB) are assigned to local swarm workers (`GPU 0`: `core-impl`, `GPU 1`: `lit-algo`, `GPU 2`: `sys-opt`, `GPU 3`: `verifier`).
- **12 Remote 96 GB Blackwell GPUs Ready Immediately (+ 4 Local on `foureyes` = 16 Total)**:
  - **`ripper`**: **4x** RTX PRO 6000 Blackwell Max-Q (96 GB each) — **IDLE**
  - **`work`**: **2x** RTX PRO 6000 Blackwell Workstation Edition (96 GB each, **full non-Max-Q Workstation TDP**) — **IDLE**
  - **`loki`**: **2x** RTX PRO 6000 Blackwell Max-Q (96 GB each) — **IDLE**
  - **`acorn.stratabaremetal.com`**: **2x** RTX PRO 6000 Blackwell **Server Edition** (96 GB each, **600W Server Edition**) — **IDLE**
  - **`cashew.stratabaremetal.com`**: **2x** RTX PRO 6000 Blackwell **Server Edition** (96 GB each, **600W Server Edition**) — **0% util, 0 MiB used (COMPLETELY IDLE)**

---

## 2. Approved Fleet Reachability & Hardware Matrix

| Host / SSH Target | IPv4 (Public / Internal) | SSH Status | Auth Used | `ssh -A` Agent Fwd | `tmux` | CPU & Available RAM | GPUs & Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`foureyes` (LOCAL)** | `10.10.0.216` (`.lan`) | **LOCALHOST** | Local shell | Live sock in `tmux -t 0` | `3.4` | Threadripper PRO 9985WX (64C/128T), `989 GiB` / `1.0 TiB` | **4x RTX PRO 6000 Blackwell Max-Q (96 GB)** — local swarm GPUs `0..3` |
| **`ripper`** | `10.10.0.161` (`ripper.lan`) | **REACHABLE** | `id_ed25519_dora_study` | **OK** (`ssh-add -l` live)* | `3.4` | Threadripper PRO 9985WX (64C/128T), `991 GiB` / `1.0 TiB` | **4x RTX PRO 6000 Blackwell Max-Q (96 GB)** — **IDLE** |
| **`work`** | `10.10.0.236` (`work.lan`) | **REACHABLE** | `id_ed25519_dora_study` | **OK** (`ssh-add -l` live)* | `3.4` | Threadripper 9980X (64C/128T), `493 GiB` / `503 GiB` | **2x RTX PRO 6000 Blackwell Workstation (96 GB, full TDP)** — **IDLE** |
| **`loki`** | `10.10.0.113` (`loki.lan`) | **REACHABLE** | `id_ed25519_dora_study` | **OK** (`ssh-add -l` live)* | `3.4` | Threadripper PRO 9985WX (64C/128T), `994 GiB` / `1.0 TiB` | **2x RTX PRO 6000 Blackwell Max-Q (96 GB)** — **IDLE** |
| **`acorn.stratabaremetal.com`** | `67.208.250.102` / `10.100.0.171` | **REACHABLE*** | Forwarded `mrcat@kuang3` | **OK** (`ssh-add -l` live)* | `3.6` | Xeon 6731P (32C/64T), `492 GiB` / `499 GiB` | **2x RTX PRO 6000 Blackwell Server Ed. (96 GB, 600W)** — **0% util (IDLE)** |
| **`cashew.stratabaremetal.com`** | `67.208.250.100` / `10.100.0.177` | **REACHABLE*** | Forwarded `mrcat@kuang3` | **OK** (`ssh-add -l` live)* | `3.6` | Xeon 6731P (32C/64T), `493 GiB` / `499 GiB` | **2x RTX PRO 6000 Blackwell Server Ed. (96 GB, 600W)** — **0% util (IDLE)** |
| `acorn`, `cashew` (bare) | _NXDOMAIN (`search lan`)_ | Use `.stratabaremetal.com` | — | — | — | — | Always use FQDNs `acorn.stratabaremetal.com` & `cashew.stratabaremetal.com` |
| `{work,ripper,loki}.stratabaremetal.com` | _NXDOMAIN_ | Use short names | — | — | — | — | Use `work`, `ripper`, `loki` (resolve via `.lan`) |
| ~~`walnut.stratabaremetal.com`~~ | — | **OFF-LIMITS** | — | — | — | — | **STRICTLY OFF-LIMITS per user — NEVER connect or run jobs; use `cashew.stratabaremetal.com` instead** |

_\*Requires exporting the live `SSH_AUTH_SOCK` from `tmux` session `0` (one-liner helper in Section 3.2 below; handled automatically in `scripts/check_fleet.sh`)._

---

## 3. SSH, DNS, `~/.ssh/config`, & Live `SSH_AUTH_SOCK` Discovery

### 3.1 DNS, `/etc/hosts`, and Subnet Topology
- `/etc/hosts` on `foureyes` defines only `127.0.0.1 localhost` and `127.0.1.1 foureyes`.
- `/etc/resolv.conf` uses `systemd-resolved` (`127.0.0.53`) with `search lan`:
  - **LAN Cluster (`10.10.0.0/24`)**: `foureyes` (`.216`), `ripper` (`.161`), `work` (`.236`), `loki` (`.113`). Resolve by bare name (`work`, `ripper`, `loki`).
  - **Strata Bare Metal Cluster (`67.208.250.0/24` public, `10.100.0.0/24` internal `bond-data`)**: `acorn.stratabaremetal.com` (`67.208.250.102` / `10.100.0.171`) and `cashew.stratabaremetal.com` (`67.208.250.100` / `10.100.0.177`, `10.100.0.178/32`). Must use FQDNs from `foureyes`.
  - **Direct Connectivity for `cashew.stratabaremetal.com`**: Direct `ssh -A -o BatchMode=yes -o ConnectTimeout=5 cashew.stratabaremetal.com` connects in `<1s` (ED25519 host key verified in `~/.ssh/known_hosts`) — no `-J` jump host needed (`acorn` -> `10.100.0.177` is also reachable over internal `bond-data`).

### 3.2 `SSH_AUTH_SOCK` Refresh Helper (Mandatory for `acorn`/`cashew` & Agent Forwarding)
When reattaching to `tmux` over `ssh -A`, `tmux` updates its session environment (`tmux show-environment -t 0 SSH_AUTH_SOCK`), leaving previously spawned processes with an old `/tmp/ssh-*/agent.*` path. Always include this snippet before calling `ssh -A` or `rsync` to `acorn.stratabaremetal.com` / `cashew.stratabaremetal.com` (or when agent forwarding is needed on `work`/`ripper`/`loki`):

```bash
# Auto-discover live forwarded SSH_AUTH_SOCK from tmux session 0 or /tmp/ssh-*/agent.*
if [ ! -S "${SSH_AUTH_SOCK:-}" ]; then
  export SSH_AUTH_SOCK="$(tmux show-environment -t 0 SSH_AUTH_SOCK 2>/dev/null | sed -n 's/^SSH_AUTH_SOCK=//p')"
  if [ ! -S "${SSH_AUTH_SOCK:-}" ]; then
    for _s in /tmp/ssh-*/agent.*; do
      if [ -S "$_s" ] && [ -O "$_s" ] && SSH_AUTH_SOCK="$_s" timeout 2 ssh-add -l >/dev/null 2>&1; then
        export SSH_AUTH_SOCK="$_s"; break
      fi
    done
  fi
fi
```

### 3.3 `~/.ssh/config` Precedence Note
`~/.ssh/config` on `foureyes` sets `IdentityFile ~/.ssh/id_ed25519_dora_study`, `IdentitiesOnly yes`, and `ForwardAgent no` for `Host ripper work loki`.
- Passing `-A` on the CLI (`ssh -A work ...`) **overrides** `ForwardAgent no` in `~/.ssh/config` (empirically verified: `ssh-add -l` succeeds remotely on `work`, `ripper`, `loki`, `acorn.stratabaremetal.com`, and `cashew.stratabaremetal.com` once the live `SSH_AUTH_SOCK` above is exported).

---

## 4. Detailed Per-Node Specifications

### 4.1 `foureyes` — **LOCALHOST (`10.10.0.216`)**
- **OS / Kernel**: Ubuntu 24.04.5 LTS, Linux `6.8.0-147-generic` x86_64
- **CPU / RAM**: AMD Ryzen Threadripper PRO 9985WX (64C/128T), `1.0 TiB` RAM (`989 GiB` avail)
- **GPUs**: 4x NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (`97,887 MiB` each), Driver `620.70.06`
- **Disk**: `/` `2.3T` avail; `/home` `451G` avail (88% full — keep massive scratch files off `/home` on `foureyes`)
- **Env**: `/home/catid/emeans/.venv/bin/python` (`Python 3.12.3`, `torch 2.14.1+cu130`, `triton 3.8.0`, `faiss-cpu 1.16.0`), `uv 0.13.5`, `tmux 3.4`

### 4.2 `ripper` — **REMOTE LAN (`10.10.0.161`)**
- **OS / Kernel**: Ubuntu 24.04.5 LTS, Linux `6.8.0-147-generic` x86_64
- **CPU / RAM**: AMD Ryzen Threadripper PRO 9985WX (64C/128T), `1.0 TiB` RAM (`991 GiB` avail)
- **GPUs**: **4x NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition** (`97,887 MiB` each, **IDLE**), Driver `620.70.06`, `nvcc` CUDA `13.5`
- **Disk**: `/` `2.5T` avail; `/home` `1.8T` avail
- **Tools**: `/usr/bin/python3` (`3.12.3`), `~/.local/bin/uv` (`0.13.5`), `tmux 3.4` (session `0` attached — **do not touch**)

### 4.3 `work` — **REMOTE LAN (`10.10.0.236`)**
- **OS / Kernel**: Ubuntu 24.04.5 LTS, Linux `6.8.0-147-generic` x86_64
- **CPU / RAM**: AMD Ryzen Threadripper 9980X (64C/128T), `503 GiB` RAM (`493 GiB` avail)
- **GPUs**: **2x NVIDIA RTX PRO 6000 Blackwell Workstation Edition** (`97,887 MiB` each, **full non-Max-Q Workstation TDP, IDLE**), Driver `620.70.06`, `nvcc` CUDA `13.5`
- **Disk**: `/` `3.3T` avail; `/home` `4.5T` avail
- **Tools**: `/usr/bin/python3` (`3.12.3`), `~/.local/bin/uv` (`0.13.5`), `tmux 3.4` (session `0` attached — **do not touch**)

### 4.4 `loki` — **REMOTE LAN (`10.10.0.113`)**
- **OS / Kernel**: Ubuntu 24.04.5 LTS, Linux `6.8.0-147-generic` x86_64
- **CPU / RAM**: AMD Ryzen Threadripper PRO 9985WX (64C/128T), `1.0 TiB` RAM (`994 GiB` avail)
- **GPUs**: **2x NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition** (`97,887 MiB` each, **IDLE**), Driver `620.70.06`, `nvcc` CUDA `13.5`
- **Disk**: `/` `3.4T` avail; `/home` **`9.2T` avail**
- **Tools**: `/usr/bin/python3` (`3.12.3`), `~/.local/bin/uv` (`0.13.5`), `tmux 3.4` (session `0` attached — **do not touch**)

### 4.5 `acorn.stratabaremetal.com` — **REMOTE STRATA BARE METAL (`67.208.250.102` / `10.100.0.171`)**
- **OS / Kernel**: Ubuntu 26.04.1 LTS (Resolute Raccoon), Linux `7.0.0-43-generic` x86_64
- **CPU / RAM**: Intel Xeon 6731P (32C/64T, 1 NUMA node), `499 GiB` RAM (`492 GiB` avail)
- **GPUs**: **2x NVIDIA RTX PRO 6000 Blackwell Server Edition** (`97,887 MiB` each, **600W Server Edition, both IDLE at 0% util / 0 MiB used**), Driver `620.70.06`, `nvcc` CUDA `13.4`
- **Disk**: `/` `805G` avail; `/home` **`3.3T` avail** (1% used)
- **Tools**: `~/.local/bin/uv` (`0.13.5`), `tmux 3.6` (session `0` attached — **do not touch**). System `/usr/bin/python3` is `3.14.4` -> pass `--python 3.12` to `uv venv` so `uv` fetches CPython 3.12 automatically for full wheel parity!

### 4.6 `cashew.stratabaremetal.com` — **REMOTE STRATA BARE METAL (`67.208.250.100` / `10.100.0.177`)**
- **OS / Kernel**: Ubuntu 26.04.1 LTS (Resolute Raccoon), Linux `7.0.0-43-generic` x86_64
- **CPU / RAM**: Single-Socket **Intel Xeon 6731P** (32C/64T, 1 NUMA node), **`499 GiB` RAM** (`493 GiB` avail, `5.9 GiB` used)
- **GPUs**: **2x NVIDIA RTX PRO 6000 Blackwell Server Edition** (`97,887 MiB` / 96 GB GDDR7 each, **600W Server Edition**), Driver `620.70.06` (CUDA UMD `13.5`), `/usr/local/cuda/bin/nvcc` CUDA `13.4`
  - **STATUS**: **COMPLETELY IDLE** — both GPU 0 and GPU 1 at **`0%` util, `0 MiB / 97,887 MiB` VRAM used, `28°C`, `35W / 600W` (`P8`), zero GPU compute processes**.
- **Disk**: `/` `805G` avail (of `879G`, 4% used); `/home` **`3.3T` avail** (of `3.5T`, 1% used)
- **Tools & Python Envs**:
  - `tmux 3.6` installed (`tmux` session `0` is attached — **do not touch**; use detached `emeans_*` sessions only).
  - `~/.local/bin/uv` (`uv 0.13.5`; prepend `export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"` in non-login SSH commands).
  - System `/usr/bin/python3` is `Python 3.14.4` (`cpython-3.14.4`); **no pre-existing CUDA `.venv`s** exist on `cashew`, so bootstrap `~/emeans/.venv` with `uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -e ".[dev]"` (Step 1 below).

> **Reminder on `walnut.stratabaremetal.com`**: Strictly **OFF-LIMITS** per user instruction. Do NOT connect to or run anything on `walnut.stratabaremetal.com`.

---

## 5. Exact `ssh -A` + Detached `tmux` (`emeans_*`) Recipes (incl. `cashew.stratabaremetal.com`)

> **Safety Rules Across All Remote Nodes**:
> 1. **NEVER** attach to, send keys to, or kill existing user `tmux` sessions (`0`, etc.). Always use a unique detached session name prefixed with `emeans_` (e.g., `emeans_cashew_bench`, `emeans_opt`, `emeans_scale`).
> 2. **NEVER** touch `walnut.stratabaremetal.com` (strictly off-limits).
> 3. Always export the live `SSH_AUTH_SOCK` before connecting (Section 3.2), and prepend `export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"` inside remote commands so `uv` and `nvcc` are found.
> 4. Use `uv venv .venv --python 3.12` so Ubuntu 24.04 (`3.12.3`) and Ubuntu 26.04 (`3.14.4` system default on `acorn`/`cashew`) both standardize on Python 3.12 wheels (`torch 2.14.1+cu130`, `triton 3.8.0`, `faiss-cpu 1.16.0`).

### Step 1: Sync Workspace & Bootstrap `.venv` on `cashew.stratabaremetal.com` (or Any Approved Host)
```bash
# Approved hosts: cashew.stratabaremetal.com, acorn.stratabaremetal.com, ripper, work, loki
HOST=cashew.stratabaremetal.com

# Ensure live SSH_AUTH_SOCK from tmux session 0
[ -S "${SSH_AUTH_SOCK:-}" ] || export SSH_AUTH_SOCK="$(tmux show-environment -t 0 SSH_AUTH_SOCK | cut -d= -f2-)"

# 1. Sync source/scripts/tests cleanly (excluding local .venv and outputs)
rsync -az --delete \
  -e "ssh -A -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5" \
  --exclude '.venv/' \
  --exclude 'outputs/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  /home/catid/emeans/ "${HOST}:~/emeans/"

# 2. Create/update Python 3.12 .venv via uv
ssh -A -o BatchMode=yes -o ConnectTimeout=5 "${HOST}" '
  set -euo pipefail
  export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
  cd ~/emeans
  if [ ! -x .venv/bin/python ]; then
    uv venv .venv --python 3.12
  fi
  uv pip install --python .venv/bin/python -e ".[dev]"
  .venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.device_count(), torch.cuda.get_device_name(0))"
'
```

### Step 2: Launch a Detached `tmux` (`emeans_*`) Benchmark/Experiment Session on `cashew.stratabaremetal.com`
```bash
HOST=cashew.stratabaremetal.com  # or: acorn.stratabaremetal.com, ripper, work, loki
SESSION=emeans_cashew_bench      # MUST start with emeans_ (never touch session 0!)
GPU_IDS=0                        # 0..1 on cashew, acorn, work, loki; 0..3 on ripper

[ -S "${SSH_AUTH_SOCK:-}" ] || export SSH_AUTH_SOCK="$(tmux show-environment -t 0 SSH_AUTH_SOCK | cut -d= -f2-)"

ssh -A -o BatchMode=yes -o ConnectTimeout=5 "${HOST}" "
  set -euo pipefail
  mkdir -p ~/emeans/outputs/logs
  tmux kill-session -t ${SESSION} 2>/dev/null || true
  tmux new-session -d -s ${SESSION} \"bash -lc '
    export PATH=\\\"\\\$HOME/.local/bin:/usr/local/cuda/bin:\\\$PATH\\\"
    export CUDA_VISIBLE_DEVICES=${GPU_IDS}
    cd ~/emeans
    .venv/bin/python scripts/run_phase2_opt.py 2>&1 | tee outputs/logs/${SESSION}.log
    echo EXIT_CODE=\\\${PIPESTATUS[0]} >> outputs/logs/${SESSION}.status
  '\"
  tmux list-sessions | grep ${SESSION}
"
```

### Step 3: Non-Blocking Status Poll & Artifact Retrieval
```bash
HOST=cashew.stratabaremetal.com
SESSION=emeans_cashew_bench

[ -S "${SSH_AUTH_SOCK:-}" ] || export SSH_AUTH_SOCK="$(tmux show-environment -t 0 SSH_AUTH_SOCK | cut -d= -f2-)"

ssh -A -o BatchMode=yes -o ConnectTimeout=5 "${HOST}" "
  tmux has-session -t ${SESSION} 2>/dev/null && echo 'STATUS: RUNNING' || echo 'STATUS: FINISHED'
  cat ~/emeans/outputs/logs/${SESSION}.status 2>/dev/null || true
  tail -n 20 ~/emeans/outputs/logs/${SESSION}.log 2>/dev/null || true
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
"

rsync -az -e "ssh -A -o BatchMode=yes -o ConnectTimeout=5" \
  "${HOST}:~/emeans/outputs/" /home/catid/emeans/outputs/
```

### Step 4: One-Command Fleet Health Check
Run `bash /home/catid/emeans/scripts/check_fleet.sh` anytime — it auto-discovers the live `SSH_AUTH_SOCK` and reports SSH, agent-forwarding, `tmux`, RAM/disk, and GPU utilization across `foureyes`, `work`, `ripper`, `loki`, `acorn.stratabaremetal.com`, and `cashew.stratabaremetal.com` (excluding off-limits `walnut.stratabaremetal.com`).
