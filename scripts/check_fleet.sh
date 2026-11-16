#!/usr/bin/env bash
# Read-only, non-hanging fleet connectivity & GPU status check for /home/catid/emeans.
# Automatically discovers a live SSH_AUTH_SOCK from tmux session environments or /tmp/ssh-*/agent.*
# when the current shell's $SSH_AUTH_SOCK has gone stale after tmux reattachment.
set -euo pipefail

discover_live_ssh_sock() {
  local cand=""
  # 1. Check current SSH_AUTH_SOCK first
  if [ -n "${SSH_AUTH_SOCK:-}" ] && [ -S "${SSH_AUTH_SOCK}" ]; then
    if SSH_AUTH_SOCK="${SSH_AUTH_SOCK}" timeout 2 ssh-add -l >/dev/null 2>&1; then
      echo "${SSH_AUTH_SOCK}"
      return 0
    fi
  fi
  # 2. Check active tmux session environments (updated on ssh -A reattach)
  if command -v tmux >/dev/null 2>&1; then
    for sess in $(tmux list-sessions -F '#{session_name}' 2>/dev/null || true); do
      cand="$(tmux show-environment -t "${sess}" SSH_AUTH_SOCK 2>/dev/null | sed -n 's/^SSH_AUTH_SOCK=//p' || true)"
      if [ -n "${cand}" ] && [ -S "${cand}" ]; then
        if SSH_AUTH_SOCK="${cand}" timeout 2 ssh-add -l >/dev/null 2>&1; then
          echo "${cand}"
          return 0
        fi
      fi
    done
  fi
  # 3. Scan /tmp/ssh-*/agent.* owned by current user
  for cand in /tmp/ssh-*/agent.*; do
    [ -S "${cand}" ] || continue
    [ -O "${cand}" ] || continue
    if SSH_AUTH_SOCK="${cand}" timeout 2 ssh-add -l >/dev/null 2>&1; then
      echo "${cand}"
      return 0
    fi
  done
  return 1
}

ORIG_SOCK="${SSH_AUTH_SOCK:-<unset>}"
if LIVE_SOCK="$(discover_live_ssh_sock)"; then
  export SSH_AUTH_SOCK="${LIVE_SOCK}"
  AGENT_STATUS="LIVE (${SSH_AUTH_SOCK}; $(timeout 2 ssh-add -l 2>&1 | head -n 1))"
else
  AGENT_STATUS="MISSING/STALE (orig=${ORIG_SOCK})"
fi

SSH_OPTS=(-A -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5)

echo "========================================================================"
echo "LOCAL HOST:     $(hostname) (FQDN: $(hostname -f 2>/dev/null || hostname))"
echo "IPv4:           $(ip -4 -o addr show scope global | awk '{print $2":"$4}' | tr '\n' ' ')"
echo "Orig AUTH_SOCK: ${ORIG_SOCK}"
echo "Live ssh-agent: ${AGENT_STATUS}"
echo "OFF-LIMITS:     walnut.stratabaremetal.com (excluded per user — do NOT use)"
echo "Local GPUs:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | sed 's/^/  /' || echo "  no nvidia-smi"
echo "========================================================================"

# NOTE: walnut.stratabaremetal.com is strictly OFF-LIMITS per user instruction.
# Use cashew.stratabaremetal.com instead.
CANDIDATES=(
  "work"
  "ripper"
  "loki"
  "acorn.stratabaremetal.com"
  "cashew.stratabaremetal.com"
)

for host in "${CANDIDATES[@]}"; do
  echo ""
  echo ">>> Checking ${host} <<<"
  resolved="$(getent ahosts "${host}" 2>/dev/null | awk '{print $1}' | grep -v ':' | sort -u | tr '\n' ' ' || true)"
  echo "  DNS IPv4: ${resolved:-<unresolved>}"

  if ssh "${SSH_OPTS[@]}" "${host}" '
    echo "  SSH:      OK (hostname=$(hostname), kernel=$(uname -r))"
    if [ -n "${SSH_AUTH_SOCK:-}" ] && [ -S "${SSH_AUTH_SOCK}" ] && timeout 2 ssh-add -l >/dev/null 2>&1; then
      echo "  AgentFwd: OK ($(timeout 2 ssh-add -l | head -n 1))"
    else
      echo "  AgentFwd: INACTIVE (SSH_AUTH_SOCK=${SSH_AUTH_SOCK:-unset})"
    fi
    echo "  Tmux:     $(tmux -V 2>/dev/null || echo missing)"
    echo "  RAM/Disk: $(free -h | awk "/^Mem:/ {print \$7 \" avail / \" \$2 \" total\"}"), /home $(df -h /home | awk "NR==2 {print \$4 \" avail\"}")"
    echo "  GPUs:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | sed "s/^/    /" || echo "    none"
  ' 2>/tmp/emeans_ssh_err.$$; then
    rm -f /tmp/emeans_ssh_err.$$
  else
    err_msg="$(tail -n 2 /tmp/emeans_ssh_err.$$ 2>/dev/null | tr '\r\n' ' ' || true)"
    rm -f /tmp/emeans_ssh_err.$$
    echo "  SSH:      UNREACHABLE (${err_msg})"
  fi
done
