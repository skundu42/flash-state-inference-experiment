#!/usr/bin/env python3
"""Table and verdict for the trace replay sweep.

    python3 bench/report.py results/replay     # holds <engine>/<sessions>/readings.json from scripts/replay.sh

Two comparisons, because they answer different questions:
- same load: both engines at the same session count. The published multiples (for example 2.75x) are of this kind,
  taken at a load where vLLM's cache has already collapsed.
- peak to peak: each engine at its own best session count. This is the capacity a provider actually gains.
Writes report.md next to the results.
"""
import json
import sys
from pathlib import Path


def fmt(v, digits=2):
    if v is None:
        return "n/a"
    return f"{v:,.0f}" if abs(v) >= 100 else f"{v:.{digits}f}"


def main():
    root = Path(sys.argv[1])
    runs = {}
    for path in sorted(root.glob("*/*/readings.json")):
        try:
            runs[(path.parent.parent.name, int(path.parent.name))] = json.loads(path.read_text())
        except (ValueError, json.JSONDecodeError):
            print(f"skipping {path}: not a readings file", file=sys.stderr)
    if not runs:
        raise SystemExit(f"no readings.json under {root}")

    out = ["| Sessions | Engine | Output TPM per GPU | Prompt TPM per GPU | Uncached p50 | TTFT p50 (s) | TTFT p90 (s) "
           "| Hit rate | Errors |", "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for sessions in sorted({s for _, s in runs}):
        for engine in ("official", "flashstate"):
            r = runs.get((engine, sessions))
            if r:
                out.append(f"| {sessions} | {engine} | {fmt(r['output_tpm_per_gpu'])} | {fmt(r['prompt_tpm_per_gpu'])} "
                           f"| {fmt(r['uncached_p50'])} | {fmt(r['ttft_p50_s'])} | {fmt(r['ttft_p90_s'])} "
                           f"| {fmt(r['hit_rate'], 3)} | {r['errors']} |")

    out += ["", "**Same load** (FlashState / vLLM at each session count):", "",
            "| Sessions | Output TPM | Prompt TPM | Uncached p50 | TTFT p90 |", "| ---: | ---: | ---: | ---: | ---: |"]
    common = sorted(s for e, s in runs if e == "official" and ("flashstate", s) in runs)
    for s in common:
        o, f = runs[("official", s)], runs[("flashstate", s)]

        def x(key):
            return fmt(f[key] / o[key]) + "x" if o.get(key) and f.get(key) is not None else "n/a"
        out.append(f"| {s} | {x('output_tpm')} | {x('prompt_tpm')} | {x('uncached_p50')} | {x('ttft_p90_s')} |")

    best = {e: max((r["output_tpm"], s) for (eng, s), r in runs.items() if eng == e)
            for e in ("official", "flashstate") if any(eng == e for eng, _ in runs)}
    if len(best) == 2 and best["official"][0]:
        (o_tpm, o_s), (f_tpm, f_s) = best["official"], best["flashstate"]
        out += ["", f"**Peak to peak**: FlashState {fmt(f_tpm)} output TPM at {f_s} sessions, vLLM {fmt(o_tpm)} at "
                    f"{o_s} sessions: {f_tpm / o_tpm:.2f}x.",
                "", "If either engine's best point is the largest session count you ran, its peak is not found yet: "
                    "add higher session counts."]
    else:
        out += ["", "Run both engines to get the comparison."]
    text = "\n".join(out) + "\n"
    (root / "report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
