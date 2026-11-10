# Shared settings and helpers. Sourced by setup.sh and scripts/*.sh.
# Override any variable below from the environment or from ./fs.env (KEY=value lines).

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
[[ -f $ROOT/fs.env ]] && { set -a; source "$ROOT/fs.env"; set +a; }

# --- upstream FlashStates, pinned -------------------------------------------------------------------------------
UPSTREAM_URL=${UPSTREAM_URL:-https://github.com/FutureMLS-Lab/FlashStates.git}
UPSTREAM_SHA=${UPSTREAM_SHA:-e2ea9a7864b41081a0147a3968e73bd5c6875a7d}   # main on 2026-11-09 (vLLM v0.33.0 base)
UP=$ROOT/third_party/FlashStates
VLLM_VERSION=0.33.0
VENVS=${VENVS:-$ROOT/.venvs}

# --- model ------------------------------------------------------------------------------------------------------
MODEL_VARIANT=${MODEL_VARIANT:-fp8}      # fp8 (about 31 GB of weights) or bf16 (about 56 GB)
case $MODEL_VARIANT in
  fp8)  : "${MODEL_REPO:=Qwen/Qwen3.8-27B-FP8}"; : "${MODEL_REVISION:=c3a462cc3d38cf351f3b26cbd763469eeb74ef81}" ;;
  bf16) : "${MODEL_REPO:=Qwen/Qwen3.8-27B}";     : "${MODEL_REVISION:=37561816e66b5b9153b1b64c10968c18c1e6f4a0}" ;;
  *) echo "MODEL_VARIANT is fp8 or bf16" >&2; exit 2 ;;
esac
MODEL_DIR=${MODEL_DIR:-$ROOT/models/qwen38-27b-$MODEL_VARIANT}
SERVED=${SERVED:-qwen38-27b}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144}   # the model's native context; lower it if the KV pool is smaller than this
HF_HOME=${HF_HOME:-$ROOT/.hf}
export HF_HOME

# --- serving ----------------------------------------------------------------------------------------------------
PORT=${PORT:-8000}
URL=http://127.0.0.1:$PORT
KV_HEADROOM_GIB=${KV_HEADROOM_GIB:-6}    # per-GPU memory left for activations and CUDA graphs when sizing the KV pool
CUDAGRAPH=${CUDAGRAPH:-default}          # default (vLLM's capture sizes) or upstream (FlashStates' 91 sizes up to 4,608)
EXTRA_SERVE_ARGS=${EXTRA_SERVE_ARGS:-}   # appended to both engines
# Prefill cost coefficients for GreedyDual-Size eviction (only their ratio matters). The default is upstream's
# Qwen3.8-Flash-Next value; scripts/calibrate.sh measures this model on this GPU and writes results/coeffs.env.
# shellcheck disable=SC1091
[[ -z ${PREFILL_COEFFS:-} && -f $ROOT/results/coeffs.env ]] && source "$ROOT/results/coeffs.env"
PREFILL_COEFFS=${PREFILL_COEFFS:-2.230e-5 4.34e-6}

die() { echo "error: $*" >&2; exit 1; }

require_setup() {
  [[ -f $UP/benchmarks/configs/common.sh ]] || die "run ./setup.sh first (missing $UP)"
  [[ -x $VENVS/flashstate/bin/vllm && -x $VENVS/official/bin/vllm ]] || die "run ./setup.sh first (missing venvs)"
  [[ -f $MODEL_DIR/config.json ]] || die "model not found in $MODEL_DIR (run ./setup.sh)"
}

# Sets GPUS (comma-separated indices), TP (their count) and GPU_MIB (the smallest one's memory).
detect_gpus() {
  command -v nvidia-smi > /dev/null || die "nvidia-smi not found"
  local idx mem want=",${GPUS:-},"
  local picked=()
  GPU_MIB=0
  while IFS=', ' read -r idx mem; do
    [[ -n ${GPUS:-} && $want != *",$idx,"* ]] && continue
    picked+=("$idx")
    if (( GPU_MIB == 0 || mem < GPU_MIB )); then GPU_MIB=$mem; fi
  done < <(nvidia-smi --query-gpu=index,memory.total --format=csv,noheader,nounits)
  (( ${#picked[@]} > 0 )) || die "no GPUs selected (GPUS=${GPUS:-all})"
  GPUS=$(IFS=,; echo "${picked[*]}")
  TP=${#picked[@]}
}

# Sets KV_BYTES (per GPU, the same for both engines) unless already given.
size_kv_pool() {
  [[ -n ${KV_BYTES:-} ]] && return
  local weights
  weights=$(python3 - "$MODEL_DIR" <<'PY'
import os, sys
print(sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(sys.argv[1]) for f in fs
          if f.endswith(".safetensors")))
PY
)
  KV_BYTES=$(( GPU_MIB * 1048576 * 92 / 100 - weights / TP - KV_HEADROOM_GIB * 1073741824 ))
  (( KV_BYTES >= 2 * 1073741824 )) || die "only $(( KV_BYTES / 1048576 )) MiB per GPU left for the KV pool \
(${TP} x ${GPU_MIB} MiB, weights $(( weights / 1073741824 )) GiB). Use more or larger GPUs, MODEL_VARIANT=fp8, \
or set KV_BYTES yourself."
}

# build_serve ENGINE: sets SERVE_ARGS, SERVE_ENV and VLLM_BIN. ENGINE is official or flashstate.
build_serve() {
  local engine=$1
  # Upstream's shared settings: CAPTURE, FLASHSTATE_SWITCHES, ENV_COMMON, ENV_FLASHSTATE, CLIENT_BASE, DATASET_*.
  # shellcheck disable=SC1091
  source "$UP/benchmarks/configs/common.sh"
  SERVE_ARGS=(--served-model-name "$SERVED" --tensor-parallel-size "$TP" --enable-prefix-caching
              --max-num-batched-tokens 16384 --seed 0 --prefix-cache-retention-interval 0
              --enable-prompt-tokens-details --max-model-len "$MAX_MODEL_LEN"
              --kv-cache-memory-bytes "$KV_BYTES" --kv-cache-dtype fp8 --long-prefill-token-threshold 6272)
  [[ $CUDAGRAPH == upstream ]] && SERVE_ARGS+=(--compilation-config "$CAPTURE")
  SERVE_ENV=("${ENV_COMMON[@]}" "CUDA_VISIBLE_DEVICES=$GPUS")
  case $engine in
    official) ;;
    flashstate)
      local flags coeffs
      # FLASHSTATE_FLAGS replaces the switches, for ablations (for example "--mamba-frontier-checkpoint" alone).
      if [[ -n ${FLASHSTATE_FLAGS:-} ]]; then read -r -a flags <<< "$FLASHSTATE_FLAGS"
      else
        read -r -a coeffs <<< "$PREFILL_COEFFS"
        flags=("${FLASHSTATE_SWITCHES[@]}" --prefill-cost-coefficients "${coeffs[@]}")
      fi
      SERVE_ARGS+=("${flags[@]}")
      SERVE_ENV+=("${ENV_FLASHSTATE[@]}") ;;
    *) die "ENGINE is official or flashstate" ;;
  esac
  local extra
  read -r -a extra <<< "$EXTRA_SERVE_ARGS"
  SERVE_ARGS+=(${extra[@]+"${extra[@]}"})
  VLLM_BIN=$VENVS/$engine/bin/vllm
}

# start_server ENGINE OUT_DIR: a fresh server in the background, healthy on return. Sets SERVER_PID and SERVER_LOG.
start_server() {
  local engine=$1 out=$2
  require_setup; detect_gpus; size_kv_pool; build_serve "$engine"
  mkdir -p "$out"
  SERVER_LOG=$out/server.log
  curl -sf "$URL/health" > /dev/null && die "something is already serving on port $PORT"
  printf '%s\n' "${SERVE_ENV[@]}" > "$out/env.txt"
  printf '%q ' "$VLLM_BIN" serve "$MODEL_DIR" "${SERVE_ARGS[@]}" --host 127.0.0.1 --port "$PORT" > "$out/command.txt"
  echo "[$engine] starting: ${TP} GPU(s) [$GPUS], KV pool $(( KV_BYTES / 1048576 )) MiB per GPU, log $SERVER_LOG" >&2
  setsid env "${SERVE_ENV[@]}" "$VLLM_BIN" serve "$MODEL_DIR" "${SERVE_ARGS[@]}" --host 127.0.0.1 --port "$PORT" \
    > "$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  trap stop_server EXIT
  local waited=0
  until curl -sf "$URL/health" > /dev/null; do
    if ! kill -0 "$SERVER_PID" 2> /dev/null; then
      tail -n 30 "$SERVER_LOG" >&2
      die "[$engine] server exited during startup (full log: $SERVER_LOG). If the log says the KV cache is too \
small for max_model_len, lower MAX_MODEL_LEN; if it is out of memory, raise KV_HEADROOM_GIB."
    fi
    (( waited < ${STARTUP_TIMEOUT:-3600} )) || die "[$engine] server not healthy after ${waited}s"
    sleep 10; waited=$(( waited + 10 ))
  done
  sleep "${SETTLE_SECONDS:-15}"
  echo "[$engine] healthy after ${waited}s" >&2
}

stop_server() {
  [[ -n ${SERVER_PID:-} ]] || return 0
  curl -sf "$URL/metrics" > "$(dirname "$SERVER_LOG")/metrics.prom" 2> /dev/null || true
  kill -- "-$SERVER_PID" 2> /dev/null || kill "$SERVER_PID" 2> /dev/null || true
  local i
  for i in $(seq 60); do kill -0 "$SERVER_PID" 2> /dev/null || break; sleep 1; done
  kill -9 -- "-$SERVER_PID" 2> /dev/null || true
  SERVER_PID=
  sleep 5   # let the driver release GPU memory before the next engine starts
}

# The pool's capacity in tokens, as vLLM logs it at startup (empty if the line is not found).
kv_pool_tokens() { grep -o 'GPU KV cache size: [0-9,]*' "$1" | head -n 1 | tr -dc '0-9'; }
