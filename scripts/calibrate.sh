#!/usr/bin/env bash
# Measure Qwen3.8-27B's prefill cost coefficients on this machine (official vLLM, cold prompts) and write
# results/coeffs.env, which later FlashState runs pick up. Optional: without it, upstream's Qwen3.8-Flash-Next
# coefficients are used.
set -euo pipefail
source "$(dirname "$0")/../lib.sh"
mkdir -p "$ROOT/results"
start_server official "$ROOT/results/calibrate"
python3 "$ROOT/bench/calibrate_coeffs.py" --url "$URL" --model "$SERVED" --max-tokens-len "$MAX_MODEL_LEN" \
  --out "$ROOT/results/coeffs.env"
stop_server
