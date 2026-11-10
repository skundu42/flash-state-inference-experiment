#!/usr/bin/env python3
"""Synthetic multi-turn agent sessions against an OpenAI-compatible /v1/completions server.

Each session starts from a long prompt and then repeats: the server generates OUTPUT tokens, the client appends that
output and TOOL new tokens (a stand-in for a tool result) and asks again. So every prompt begins, token for token,
with the previous prompt and the server's own output, which is the pattern FlashState claims to serve from cache.

Prompts are random token IDs, so no tokenizer is needed and no two sessions share more than the shared prefix.
Sessions take their turns in rounds: all sessions do turn 1 (at most CONCURRENCY at a time), then all do turn 2, ...

Writes OUT_DIR/records.jsonl (one line per request) and OUT_DIR/summary.json. Standard library only.
"""
import argparse
import json
import random
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ID_LO, ID_HI = 1000, 100000  # ordinary vocabulary entries of Qwen's tokenizer, away from the special tokens


def percentile(values, q):
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def post_json(url, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def complete(base, model, prompt_ids, max_tokens, timeout):
    """One streamed completion. Returns ttft, e2e (seconds), output token IDs, usage and whether IDs were re-derived."""
    body = {"model": model, "prompt": prompt_ids, "max_tokens": max_tokens, "min_tokens": max_tokens,
            "ignore_eos": True, "temperature": 0, "seed": 0, "stream": True,
            "stream_options": {"include_usage": True}, "return_token_ids": True}
    req = urllib.request.Request(base + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0, ttft, ids, text, usage = time.perf_counter(), None, [], [], None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("error"):
                    raise RuntimeError(json.dumps(chunk["error"]))
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices") or []:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    text.append(choice.get("text") or "")
                    ids.extend(choice.get("token_ids") or [])
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:500]}") from None
    e2e = time.perf_counter() - t0
    if usage is None or ttft is None:
        raise RuntimeError("the stream ended without usage or without any token")
    retokenized = False
    if len(ids) != usage["completion_tokens"]:
        # The server did not return token IDs: recover them from the text.
        ids = post_json(base + "/tokenize", {"model": model, "prompt": "".join(text), "add_special_tokens": False},
                        timeout)["tokens"]
        retokenized = True
    return ttft, e2e, ids, usage, retokenized


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--model", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sessions", type=int, default=4)
    p.add_argument("--turns", type=int, default=8)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--context-tokens", type=int, default=20000, help="unique tokens in each session's first prompt")
    p.add_argument("--shared-prefix-tokens", type=int, default=2000, help="tokens every session starts with")
    p.add_argument("--tool-tokens", type=int, default=300, help="new tokens appended after each output")
    p.add_argument("--output-tokens", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout", type=float, default=1800)
    a = p.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    shared_rng = random.Random(a.seed)
    shared = [shared_rng.randrange(ID_LO, ID_HI) for _ in range(a.shared_prefix_tokens)]
    rngs = [random.Random(a.seed * 1000003 + i + 1) for i in range(a.sessions)]
    prompts = [shared + [r.randrange(ID_LO, ID_HI) for _ in range(a.context_tokens)] for r in rngs]
    previous = [0] * a.sessions  # the previous prompt's length, per session
    records = []

    def turn(i, t):
        prompt = prompts[i]
        rec = {"session": i, "turn": t, "sent_tokens": len(prompt), "appended": len(prompt) - previous[i]}
        try:
            ttft, e2e, out_ids, usage, retok = complete(a.url, a.model, prompt, a.output_tokens, a.timeout)
        except Exception as e:  # noqa: BLE001 - one failed request must not stop the run
            rec["error"] = str(e)
            return rec
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        rec.update(prompt_tokens=usage["prompt_tokens"], cached=cached, uncached=usage["prompt_tokens"] - cached,
                   completion_tokens=usage["completion_tokens"], ttft=ttft, e2e=e2e, retokenized=retok,
                   output_ids=out_ids)
        previous[i] = len(prompt)
        prompts[i] = prompt + out_ids + [rngs[i].randrange(ID_LO, ID_HI) for _ in range(a.tool_tokens)]
        return rec

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=a.concurrency) as pool, (a.out / "records.jsonl").open("w") as f:
        for t in range(1, a.turns + 1):
            alive = [i for i in range(a.sessions) if not any(r["session"] == i and "error" in r for r in records)]
            round_start = time.perf_counter()
            for rec in pool.map(lambda i: turn(i, t), alive):
                records.append(rec)
                f.write(json.dumps(rec) + "\n")
            f.flush()
            ok = [r for r in records if r["turn"] == t and "error" not in r]
            if ok:
                print(f"turn {t}: {len(ok)}/{len(alive)} ok in {time.perf_counter() - round_start:.1f}s, "
                      f"uncached p50 {statistics.median(r['uncached'] for r in ok):.0f}, "
                      f"TTFT p50 {statistics.median(r['ttft'] for r in ok):.2f}s", flush=True)
    wall = time.perf_counter() - start

    ok = [r for r in records if "error" not in r]
    first = [r for r in ok if r["turn"] == 1]
    later = [r for r in ok if r["turn"] > 1]
    prompt_total = sum(r["prompt_tokens"] for r in later)
    summary = {
        "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
        "requests": len(records), "errors": len(records) - len(ok),
        "retokenized": sum(r["retokenized"] for r in ok),
        "wall_s": wall,
        "output_tokens_per_s": sum(r["completion_tokens"] for r in ok) / wall if wall else None,
        "first_turn": {"ttft_p50_s": percentile([r["ttft"] for r in first], 0.5)},
        # Turns after the first are the ones a prefix cache can serve.
        "later_turns": {
            "requests": len(later),
            "appended_p50": percentile([r["appended"] for r in later], 0.5),
            "uncached_p50": percentile([r["uncached"] for r in later], 0.5),
            "uncached_p90": percentile([r["uncached"] for r in later], 0.9),
            "hit_rate": sum(r["cached"] for r in later) / prompt_total if prompt_total else None,
            "ttft_p50_s": percentile([r["ttft"] for r in later], 0.5),
            "ttft_p90_s": percentile([r["ttft"] for r in later], 0.9),
        },
    }
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary["later_turns"], indent=1))
    if summary["errors"]:
        print(f"{summary['errors']} request(s) failed; first error: "
              f"{next(r['error'] for r in records if 'error' in r)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
