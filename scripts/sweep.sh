#!/usr/bin/env bash
# The replay benchmark for both engines over several session counts, then the report.
#
#   scripts/sweep.sh 8 16 24 32
#
# Pick counts that bracket the point where official vLLM's hit rate drops; that point scales with the KV pool
# (upstream's 2 x B200 pool held up to about 40 sessions). Finished points are skipped; delete their directory to redo.
set -euo pipefail
source "$(dirname "$0")/../lib.sh"
(( $# > 0 )) || die "give the session counts, for example: scripts/sweep.sh 8 16 24 32"
for sessions in "$@"; do
  for engine in ${ENGINES:-official flashstate}; do
    if [[ -s $ROOT/results/replay/$engine/$sessions/readings.json ]]; then
      echo "skipping $engine at $sessions sessions (done)"; continue
    fi
    "$ROOT/scripts/replay.sh" "$engine" "$sessions" || echo "FAILED: $engine at $sessions sessions" >&2
  done
done
python3 "$ROOT/bench/report.py" "$ROOT/results/replay"
