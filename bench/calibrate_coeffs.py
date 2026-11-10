#!/usr/bin/env python3
"""Measure this model's prefill cost coefficients on this machine, for FlashState's GreedyDual-Size eviction.

Upstream describes the two numbers of --prefill-cost-coefficients as "seconds per token d, and e per token per
100,000 tokens of context; only their ratio matters". Read that as: a token at context position c costs
d + e * c / 100000 seconds to prefill, so a cold prompt of n tokens takes

    T(n) = a + d * n + e * n^2 / 200000        (a: fixed overhead per request)

This sends cold prompts (random token IDs, one output token) of growing length, one at a time, and fits a, d, e by
least squares to the measured time to first token. That reading of the formula is this repo's, not confirmed by
upstream; since only the ratio matters and it only orders evictions, an approximate fit is enough.

    python3 bench/calibrate_coeffs.py --model NAME --max-tokens-len 262144 --out results/coeffs.env
"""
import argparse
import json
import random
import time
import urllib.request

ID_LO, ID_HI = 1000, 100000


def ttft(base, model, n, rng, timeout):
    body = {"model": model, "prompt": [rng.randrange(ID_LO, ID_HI) for _ in range(n)], "max_tokens": 1,
            "temperature": 0, "stream": True}
    req = urllib.request.Request(base + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if line.startswith("data:") and '"choices"' in line:
                first = time.perf_counter() - t0
                resp.read()
                return first
    raise RuntimeError("no token in the response")


def solve3(m, v):
    """Gaussian elimination with partial pivoting for a 3x3 system."""
    m = [row[:] + [v[i]] for i, row in enumerate(m)]
    for c in range(3):
        pivot = max(range(c, 3), key=lambda r: abs(m[r][c]))
        m[c], m[pivot] = m[pivot], m[c]
        for r in range(c + 1, 3):
            k = m[r][c] / m[c][c]
            m[r] = [x - k * y for x, y in zip(m[r], m[c])]
    x = [0.0] * 3
    for r in (2, 1, 0):
        x[r] = (m[r][3] - sum(m[r][c] * x[c] for c in range(r + 1, 3))) / m[r][r]
    return x


def fit(points):
    """Least squares for T = a + d*n + e*n^2/200000 over (n, T) points. Returns (a, d, e)."""
    feats = [(1.0, n, n * n / 200000.0) for n, _ in points]
    gram = [[sum(f[i] * f[j] for f in feats) for j in range(3)] for i in range(3)]
    rhs = [sum(f[i] * t for f, (_, t) in zip(feats, points)) for i in range(3)]
    return solve3(gram, rhs)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--model", required=True)
    p.add_argument("--max-tokens-len", type=int, required=True, help="the server's --max-model-len")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--out", required=True)
    p.add_argument("--timeout", type=float, default=1800)
    a = p.parse_args()

    limit = a.max_tokens_len - 64
    lengths = [n for n in (4000, 8000, 16000, 32000, 64000, 96000, 128000, 192000, 250000) if n <= limit]
    if len(lengths) < 4:
        raise SystemExit("the context is too short to fit the coefficients; keep the default")
    rng = random.Random(12345)
    ttft(a.url, a.model, 2000, rng, a.timeout)  # warm-up, not measured
    points = []
    for n in lengths:
        best = min(ttft(a.url, a.model, n, rng, a.timeout) for _ in range(a.repeats))
        points.append((n, best))
        print(f"{n:>7} tokens: {best:.3f}s cold TTFT", flush=True)
    base, d, e = fit(points)
    print(f"fit: overhead {base:.3f}s, d {d:.3e} s/token, e {e:.3e} s/token per 100K tokens of context")
    if d <= 0 or e <= 0:
        raise SystemExit("the fit is not usable (a coefficient is not positive); keeping the default coefficients")
    with open(a.out, "w") as f:
        f.write(f'PREFILL_COEFFS="{d:.3e} {e:.3e}"\n')
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
