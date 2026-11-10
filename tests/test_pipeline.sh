#!/usr/bin/env bash
# Checks the client and report scripts end to end against tests/mock_server.py. Needs only python3, no GPU.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
tmp=$(mktemp -d)
pids=()
trap 'kill "${pids[@]}" 2> /dev/null || true; rm -rf "$tmp"' EXIT

port=18731
for engine in official flashstate; do
  python3 "$root/tests/mock_server.py" "$port" "$engine" &
  pids+=($!)
  until curl -sf "http://127.0.0.1:$port/health" > /dev/null; do sleep 0.2; done
  python3 "$root/bench/turns.py" --url "http://127.0.0.1:$port" --model mock --out "$tmp/ab/$engine" \
    --sessions 3 --turns 5 --concurrency 3 --context-tokens 5000 --tool-tokens 300 --output-tokens 50 > /dev/null
  port=$(( port + 1 ))
done
python3 "$root/bench/compare.py" "$tmp/ab" > "$tmp/report.txt"
grep -q 'H1 resume point.*-> SUPPORTED' "$tmp/report.txt"
grep -q 'H3 same answers.*15/15 -> SUPPORTED' "$tmp/report.txt"

# The fit recovers known coefficients.
python3 - "$root" <<'PY'
import sys
sys.path.insert(0, sys.argv[1] + "/bench")
from calibrate_coeffs import fit
a, d, e = 0.05, 2.2e-5, 4.3e-6
got = fit([(n, a + d * n + e * n * n / 200000) for n in (4000, 8000, 16000, 32000, 64000, 128000)])
assert all(abs(g - w) / w < 1e-6 for g, w in zip(got, (a, d, e))), got
PY

# The replay report reads upstream-style readings.
for engine in official flashstate; do
  for sessions in 8 16; do
    mkdir -p "$tmp/replay/$engine/$sessions"
    tpm=40000
    [[ $engine == official ]] && tpm=$(( 40000 / (sessions / 8) ))
    printf '{"output_tpm": %s, "output_tpm_per_gpu": %s, "prompt_tpm": 3000000, "prompt_tpm_per_gpu": 3000000,
      "ttft_p50_s": 0.5, "ttft_p90_s": 1.5, "decode_p50_tok_s": 80, "hit_rate": 0.9, "uncached_p50": 1200,
      "requests_in_window": 100, "errors": 0, "window_s": [900, 2700], "gpus": 1}\n' "$tpm" "$tpm" \
      > "$tmp/replay/$engine/$sessions/readings.json"
  done
done
python3 "$root/bench/report.py" "$tmp/replay" > "$tmp/replay.txt"
grep -q '| 16 | 2.00x' "$tmp/replay.txt"
grep -q 'Peak to peak.*1.00x' "$tmp/replay.txt"

for f in "$root"/lib.sh "$root"/setup.sh "$root"/scripts/*.sh; do bash -n "$f"; done
echo "ok"
