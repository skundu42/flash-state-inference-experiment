#!/usr/bin/env bash
# A/B on synthetic agent sessions: official vLLM, then FlashState, same GPUs, same KV pool, same workload.
#
#   scripts/ab.sh mechanism     a few sessions that fit in the cache: isolates where the next turn resumes
#   scripts/ab.sh pressure      more session context than the KV pool holds (about 1.5x): exercises eviction
#
# Workload overrides: SESSIONS TURNS CONCURRENCY CONTEXT_TOKENS TOOL_TOKENS OUTPUT_TOKENS.
# ENGINES="official flashstate" by default; results go to results/<preset>/<engine>/ and results/<preset>/report.md.
set -euo pipefail
source "$(dirname "$0")/../lib.sh"
preset=${1:-mechanism}
out=$ROOT/results/$preset
mkdir -p "$out"

TURNS=${TURNS:-8}
TOOL_TOKENS=${TOOL_TOKENS:-300}
OUTPUT_TOKENS=${OUTPUT_TOKENS:-200}
case $preset in
  mechanism) SESSIONS=${SESSIONS:-4}; CONTEXT_TOKENS=${CONTEXT_TOKENS:-20000} ;;
  pressure)  CONTEXT_TOKENS=${CONTEXT_TOKENS:-30000} ;;
  *) die "preset is mechanism or pressure" ;;
esac

for engine in ${ENGINES:-official flashstate}; do
  start_server "$engine" "$out/$engine"
  if [[ $preset == pressure && -z ${SESSIONS:-} ]]; then
    # Size the workload once, from the first engine's pool, so both engines get the same one.
    if [[ -f $out/sessions.txt ]]; then SESSIONS=$(< "$out/sessions.txt")
    else
      pool=$(kv_pool_tokens "$SERVER_LOG")
      [[ -n $pool ]] || die "could not read the KV pool size from $SERVER_LOG; set SESSIONS yourself"
      per_session=$(( 2000 + CONTEXT_TOKENS + TURNS * (TOOL_TOKENS + OUTPUT_TOKENS) ))
      SESSIONS=$(( (pool * 3 / 2 + per_session - 1) / per_session ))
      (( SESSIONS > 256 )) && SESSIONS=256
      echo "$SESSIONS" > "$out/sessions.txt"
      echo "KV pool: $pool tokens -> $SESSIONS sessions of about $per_session tokens" >&2
    fi
  fi
  python3 "$ROOT/bench/turns.py" --url "$URL" --model "$SERVED" --out "$out/$engine" \
    --sessions "$SESSIONS" --turns "$TURNS" --concurrency "${CONCURRENCY:-$SESSIONS}" \
    --context-tokens "$CONTEXT_TOKENS" --tool-tokens "$TOOL_TOKENS" --output-tokens "$OUTPUT_TOKENS" \
    || echo "[$engine] the client reported failed requests (see $out/$engine/records.jsonl)" >&2
  stop_server
done

if [[ -f $out/official/summary.json && -f $out/flashstate/summary.json ]]; then
  python3 "$ROOT/bench/compare.py" "$out"
fi
