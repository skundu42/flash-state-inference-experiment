#!/usr/bin/env bash
# One-time setup on a Linux x86_64 machine with NVIDIA GPU(s):
#   1. clone FlashStates at the pinned commit (with its vLLM fork)
#   2. three Python 3.12 environments: official vLLM 0.33.0, FlashStates (0.33.0 + the fork's files), the replay client
#   3. download Qwen3.8-27B (MODEL_VARIANT=fp8 by default, or bf16)
#   4. download the coding-agent traces (skip with SKIP_TRACES=1; only scripts/replay.sh needs them)
# Re-running is safe: finished steps are skipped.
set -euo pipefail
source "$(dirname "$0")/lib.sh"

for tool in git curl python3 nvidia-smi; do command -v "$tool" > /dev/null || die "$tool not found"; done
echo "GPUs:"; nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

# Preflight (warnings only). On RunPod, clone this repo into /workspace: the container disk is small and is wiped
# when the pod stops, the volume is not.
driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)
if (( ${driver%%.*} < 580 )); then
  echo "WARNING: driver $driver. vLLM $VLLM_VERSION is built for CUDA 13, which needs driver 580 or newer;" \
       "the servers will probably fail to start. On RunPod, pick a pod with CUDA 13 in the deploy filter."
fi
free_gb=$(df -Pk "$ROOT" | awk 'NR == 2 { print int($4 / 1048576) }')
need_gb=$([[ $MODEL_VARIANT == bf16 ]] && echo 110 || echo 85)
if (( free_gb < need_gb )); then
  echo "WARNING: ${free_gb} GB free under $ROOT; setup needs about ${need_gb} GB (weights, three environments, traces)."
fi
if [[ -d /workspace && $ROOT != /workspace/* ]]; then
  echo "WARNING: this looks like a RunPod pod but the repo is not under /workspace; a pod stop will lose it."
fi
export UV_LINK_MODE=copy   # network volumes do not support the hardlinks uv tries first

if ! command -v uv > /dev/null; then
  echo "installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# 1. upstream at the pinned commit
if [[ ! -d $UP/.git ]]; then
  mkdir -p "$(dirname "$UP")"
  git clone "$UPSTREAM_URL" "$UP"
fi
git -C "$UP" fetch -q origin
git -C "$UP" checkout -q "$UPSTREAM_SHA"
git -C "$UP" submodule update --init --depth 1 third_party/vllm

# 2. environments
venv() { [[ -x $VENVS/$1/bin/python ]] || uv venv --python 3.12 "$VENVS/$1"; }
venv official
VIRTUAL_ENV=$VENVS/official uv pip install "vllm==$VLLM_VERSION"

venv flashstate
VIRTUAL_ENV=$VENVS/flashstate uv pip install "vllm==$VLLM_VERSION"
site=$("$VENVS/flashstate/bin/python" -c 'import os, vllm; print(os.path.dirname(vllm.__file__))')
# The fork changes only Python files of v0.33.0: copy them over the installed package (upstream docs/install.md).
while read -r status path; do
  [[ -n $path ]] || continue
  case $status in
    D) rm -f "$site/${path#vllm/}" ;;
    *) install -D -m 644 "$UP/third_party/vllm/$path" "$site/${path#vllm/}" ;;
  esac
done < "$UP/docker/vllm_overlay.txt"
grep -rqE 'mamba[-_]frontier[-_]checkpoint' "$site/engine/arg_utils.py" "$site/config/cache.py" \
  || die "the FlashStates options are missing from $site after the overlay"
official_site=$("$VENVS/official/bin/python" -c 'import os, vllm; print(os.path.dirname(vllm.__file__))')
if grep -rqE 'mamba[-_]frontier[-_]checkpoint' "$official_site/engine/arg_utils.py" "$official_site/config/cache.py"; then
  die "the official environment has FlashStates options; it must be a plain vllm==$VLLM_VERSION"
fi

venv aiperf
VIRTUAL_ENV=$VENVS/aiperf uv pip install "$UP/benchmarks/agentx_harness"

# 3. model
if [[ ${SKIP_MODEL:-0} != 1 ]]; then
  echo "downloading $MODEL_REPO@$MODEL_REVISION to $MODEL_DIR"
  "$VENVS/official/bin/python" - "$MODEL_REPO" "$MODEL_REVISION" "$MODEL_DIR" <<'PY'
import sys

from huggingface_hub import snapshot_download

repo, revision, local_dir = sys.argv[1:]
snapshot_download(repo, revision=revision, local_dir=local_dir)
PY
fi

# 4. traces for the replay benchmark
if [[ ${SKIP_TRACES:-0} != 1 ]]; then
  PYTHON=$VENVS/aiperf/bin/python bash "$UP/benchmarks/tools/fetch_dataset.sh"
fi

echo
echo "setup done. Next: scripts/ab.sh mechanism"
