#!/usr/bin/env bash
# Start one engine and keep it running, to poke at it by hand: scripts/serve.sh official|flashstate
# The exact command and environment are in results/serve/<engine>/command.txt and env.txt.
set -euo pipefail
source "$(dirname "$0")/../lib.sh"
engine=${1:?ENGINE (official or flashstate)}
start_server "$engine" "$ROOT/results/serve/$engine"
echo "serving $SERVED at $URL (Ctrl-C to stop)"
tail -f "$SERVER_LOG"
