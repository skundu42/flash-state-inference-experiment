#!/usr/bin/env bash
# One point of the paper's benchmark, on Qwen3.8-27B: a fresh server of one engine, then a closed-loop replay of the
# public coding-agent traces at one session count, then upstream's readings. Follows upstream's benchmarks/run_local.sh.
#
#   scripts/replay.sh ENGINE SESSIONS        ENGINE is official or flashstate
#
# DURATION (default 2700 s) and WINDOW (default "900 2700": the readings skip the first 15 minutes, while the cache
# fills). For a quick check: DURATION=600 WINDOW="120 600". The client needs about 70 GiB of host memory.
set -euo pipefail
source "$(dirname "$0")/../lib.sh"
engine=${1:?ENGINE (official or flashstate)} sessions=${2:?SESSIONS}
DURATION=${DURATION:-2700}
read -r -a window <<< "${WINDOW:-900 2700}"
out=$ROOT/results/replay/$engine/$sessions
[[ -x $VENVS/aiperf/bin/aiperf ]] || die "the replay client is missing (run ./setup.sh)"
rm -rf "$out/client"
ram_gb=$(awk '/MemAvailable/ { print int($2 / 1048576) }' /proc/meminfo)
(( ram_gb >= 70 )) || echo "WARNING: ${ram_gb} GiB of host memory available; the replay client needs about 70 GiB" \
  "while it builds the sessions and may be killed." >&2

start_server "$engine" "$out"   # also loads upstream's CLIENT_BASE
echo "[$engine] replaying at $sessions sessions for ${DURATION}s (client log: $out/client.log)" >&2
# The dataset loads offline from its cache. HF_HUB_OFFLINE stays unset: with it, the client looks the tokenizer up as
# a Hub repository instead of loading the model directory (upstream's note).
env HF_DATASETS_OFFLINE=1 "$VENVS/aiperf/bin/aiperf" profile "${CLIENT_BASE[@]}" \
  --max-context-length "$MAX_MODEL_LEN" --url "$URL" --model "$SERVED" --tokenizer "$MODEL_DIR" \
  --concurrency "$sessions" --benchmark-duration "$DURATION" --output-artifact-dir "$out/client" \
  > "$out/client.log" 2>&1 || { tail -n 20 "$out/client.log" >&2; die "the replay client failed"; }
stop_server
python3 "$UP/analysis/readings.py" "$out/client" --gpus "$TP" --window "${window[@]}" | tee "$out/readings.json"
