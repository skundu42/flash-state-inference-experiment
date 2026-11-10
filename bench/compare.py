#!/usr/bin/env python3
"""Compare two runs of bench/turns.py (official vLLM and FlashState) and judge the hypotheses.

    python3 bench/compare.py RESULTS_DIR      # holds official/ and flashstate/, each with summary.json and records.jsonl

Prints a Markdown report and writes it to RESULTS_DIR/report.md. The thresholds are fixed here, before any run:

- H1 (resume point): FlashState's median uncached prompt tokens on later turns are at most 0.6x vLLM's.
- H2 (latency): FlashState's median later-turn TTFT is at most 0.9x vLLM's.
- H3 (same answers): at temperature 0, at least 98% of the requests that received identical prompts on both engines
  produce identical output tokens.
"""
import json
import sys
from pathlib import Path

UNCACHED_MAX_RATIO, TTFT_MAX_RATIO, PARITY_MIN = 0.6, 0.9, 0.98


def load(d: Path):
    summary = json.loads((d / "summary.json").read_text())
    records = [json.loads(line) for line in (d / "records.jsonl").open() if line.strip()]
    return summary, {(r["session"], r["turn"]): r for r in records if "error" not in r}


def parity(a, b):
    """Requests with identical prompts on both engines, and how many of them gave identical output."""
    comparable = identical = 0
    for s in sorted({k[0] for k in a} & {k[0] for k in b}):
        t = 1
        while (s, t) in a and (s, t) in b:
            comparable += 1
            if a[(s, t)]["output_ids"] != b[(s, t)]["output_ids"]:
                break  # the next prompts differ, so later turns are not comparable
            identical += 1
            t += 1
    return comparable, identical


def fmt(v, digits=2):
    if v is None:
        return "n/a"
    if isinstance(v, int):
        return f"{v:,}"
    return f"{v:,.0f}" if abs(v) >= 100 else f"{v:.{digits}f}"


def ratio(fs, off):
    return fs / off if fs is not None and off else None


def verdict(ok):
    return "n/a" if ok is None else ("SUPPORTED" if ok else "NOT SUPPORTED")


def main():
    root = Path(sys.argv[1])
    (off_s, off_r), (fs_s, fs_r) = load(root / "official"), load(root / "flashstate")
    o, f = off_s["later_turns"], fs_s["later_turns"]
    rows = [
        ("Uncached prompt tokens, p50 (later turns)", o["uncached_p50"], f["uncached_p50"]),
        ("Uncached prompt tokens, p90 (later turns)", o["uncached_p90"], f["uncached_p90"]),
        ("Tokens appended per turn, p50", o["appended_p50"], f["appended_p50"]),
        ("Cache hit rate (later turns)", o["hit_rate"], f["hit_rate"]),
        ("TTFT p50, later turns (s)", o["ttft_p50_s"], f["ttft_p50_s"]),
        ("TTFT p90, later turns (s)", o["ttft_p90_s"], f["ttft_p90_s"]),
        ("TTFT p50, first turn (s)", off_s["first_turn"]["ttft_p50_s"], fs_s["first_turn"]["ttft_p50_s"]),
        ("Output tokens per second (whole run)", off_s["output_tokens_per_s"], fs_s["output_tokens_per_s"]),
        ("Wall time (s)", off_s["wall_s"], fs_s["wall_s"]),
        ("Failed requests", off_s["errors"], fs_s["errors"]),
    ]
    out = ["| Metric | vLLM | FlashState | FlashState / vLLM |", "| --- | ---: | ---: | ---: |"]
    for name, a, b in rows:
        r = ratio(b, a)
        out.append(f"| {name} | {fmt(a)} | {fmt(b)} | {fmt(r) + 'x' if r is not None else 'n/a'} |")

    comparable, identical = parity(off_r, fs_r)
    match = identical / comparable if comparable else None
    r_unc, r_ttft = ratio(f["uncached_p50"], o["uncached_p50"]), ratio(f["ttft_p50_s"], o["ttft_p50_s"])
    if o["uncached_p50"] == 0 and f["uncached_p50"] == 0:
        r_unc = 1.0
    out += [
        "",
        f"- **H1 resume point** (uncached p50 ratio <= {UNCACHED_MAX_RATIO}): "
        f"{fmt(r_unc)} -> {verdict(None if r_unc is None else r_unc <= UNCACHED_MAX_RATIO)}",
        f"- **H2 latency** (later-turn TTFT p50 ratio <= {TTFT_MAX_RATIO}): "
        f"{fmt(r_ttft)} -> {verdict(None if r_ttft is None else r_ttft <= TTFT_MAX_RATIO)}",
        f"- **H3 same answers** (>= {PARITY_MIN:.0%} identical outputs for identical prompts): "
        f"{identical}/{comparable} -> {verdict(None if match is None else match >= PARITY_MIN)}",
    ]
    notes = []
    if off_s["errors"] or fs_s["errors"]:
        notes.append("Some requests failed; the run is not a clean comparison.")
    if off_s["retokenized"] or fs_s["retokenized"]:
        notes.append("The server did not return token IDs for some requests; the client re-tokenized the text, so "
                     "prompts may not extend the previous ones exactly.")
    def workload(summary):
        return {k: v for k, v in summary["settings"].items() if k not in ("out", "url", "timeout")}
    if workload(off_s) != workload(fs_s):
        notes.append("The two runs used different workload settings.")
    out += [""] + [f"> {n}" for n in notes]
    text = "\n".join(out).rstrip() + "\n"
    (root / "report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
