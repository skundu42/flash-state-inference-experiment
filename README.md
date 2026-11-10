# flash-state: does FlashState beat vLLM on Qwen3.8-27B?

An A/B harness for the claim in [this post](https://x.com/_weilix/status/2119837671946400165): FlashState, a fork of
vLLM v0.33.0, serves agent workloads on hybrid models much faster than vLLM by managing the prefix cache better.
The authors measured GLM-5.3-Flash, Qwen3.8-Flash-Next, Nemotron 3 Super and Kimi-K3.1-Pro on B200/GB300.
**Qwen3.8-27B is not among them**, so this repo tests it: official vLLM 0.33.0 against
[FlashStates](https://github.com/FutureMLS-Lab/FlashStates) (pinned commit), same GPUs, same KV pool, same workload.

> Status: the client and report scripts are tested against a mock server (`tests/test_pipeline.sh`). Nothing here
> has run on a GPU yet. Expect to fix a flag or two on the first real run; see [If something fails](#if-something-fails).

## The hypothesis

Qwen3.8-27B has 16 attention layers (a KV cache that grows with context) and 48 linear-attention layers (one
fixed-size recurrent state). To resume from a cached prefix the server needs the KV blocks *and* a recurrent state
saved at exactly the resume position. vLLM saves those states at block boundaries of the prompt, so after an agent
turn it recomputes the tail of the prompt plus its own previous output. FlashState adds:

| Switch | Claim |
| --- | --- |
| `--mamba-frontier-checkpoint` | keep a finished request's state at its exact end, so the next turn resumes there |
| `--mamba-prefill-export` | save states inside a prefill chunk, not only at chunk ends |
| `--prefix-cache-eviction gds` + `--prefill-cost-coefficients` | evict by prefill seconds saved per block freed |
| `--prefix-cache-leaf-last` | evict older states of a session before its newest one |

What that should show up as, and where this repo measures it:

| | Prediction | Experiment |
| --- | --- | --- |
| H1 | fewer uncached prompt tokens per turn | `ab.sh mechanism`, `ab.sh pressure`, replay |
| H2 | lower time to first token on later turns | same |
| H3 | same output tokens as vLLM at temperature 0 | `ab.sh mechanism` |
| H4 | when sessions outgrow the KV pool, the hit rate and throughput hold up instead of collapsing | `ab.sh pressure`, replay sweep |

## Run it on RunPod

Pod: **CUDA 13 / driver 580+** (use the CUDA filter when deploying), a volume of **150 GB or more mounted at
`/workspace`**, any PyTorch template. Pods are containers, so nothing here uses Docker.

| GPUs | Verdict |
| --- | --- |
| 1 x 48 GB | FP8 weights fit, but only about 7 GB is left for the cache. `MAX_MODEL_LEN=65536 scripts/ab.sh mechanism` only. |
| 1 x 80 GB (H100, A100) | Minimum for a meaningful run: about 40 GB of KV pool with FP8. |
| 1 x H200 / B200, or 2 x 80 GB | Recommended. Needed for `MODEL_VARIANT=bf16`. |

The trace replay also needs about 70 GiB of host RAM for its client. GPU counts must be 1, 2, 4 or 8 (tensor parallel).

```bash
cd /workspace
git clone <this repo> flash-state && cd flash-state
tmux new -s fs                      # runs are long; survive SSH drops

./setup.sh                          # 20-40 min: FlashStates, 3 environments, FP8 weights (31 GB), traces
scripts/ab.sh mechanism             # ~10 min: 4 sessions x 8 turns on each engine, then the verdict
scripts/ab.sh pressure              # ~20-40 min: 1.5x more session context than the KV pool holds
scripts/calibrate.sh                # optional, ~5 min: this model's eviction cost coefficients
scripts/sweep.sh 4 8 12 16          # the paper's benchmark; about 50 min per engine per session count
```

Reports land in `results/mechanism/report.md`, `results/pressure/report.md` and `results/replay/report.md`. Every
run directory keeps the exact server command, environment, server log and a `/metrics` snapshot.

## The three experiments

**1. `scripts/ab.sh mechanism`** — synthetic agent sessions (`bench/turns.py`): a 22K-token prompt, then 8 turns
of "server writes 200 tokens, client appends them plus 300 new tokens, asks again". Prompts are random token IDs,
sent as IDs, so each prompt extends the previous one exactly. Everything fits in the cache, so this isolates the
resume point. If FlashState works, its uncached tokens per turn should be close to the 300 new ones, and vLLM's
visibly higher. The report applies thresholds fixed in `bench/compare.py`: uncached p50 at most 0.6x vLLM's (H1),
later-turn TTFT p50 at most 0.9x (H2), at least 98% identical outputs for identical prompts (H3).

**2. `scripts/ab.sh pressure`** — the same sessions, but enough of them to need 1.5x the KV pool (sized from
the pool vLLM reports at startup, identically for both engines). Turns go round-robin, so every session's state is
an eviction candidate while the others run. This is where the eviction policy matters.

**3. `scripts/sweep.sh N...`** — the paper's method: upstream's replay client and its
[public coding-agent traces](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126), closed loop
at N concurrent sessions, 45 minutes per point, readings over minutes 15 to 45 by upstream's own
`analysis/readings.py`. For a first look: `DURATION=600 WINDOW="120 600" scripts/sweep.sh 4 8`.

## Reading the result honestly

The published multiples compare both engines **at the same session count**, at a load where vLLM's cache has
already collapsed. In upstream's own Qwen3.8-Flash-Next table, FlashState is 2.75x vLLM at 56 sessions but 0.98x
at 32, and its best point is 1.13x vLLM's best point (38.9K vs 34.4K output tokens per minute per GPU).
`bench/report.py` prints both views. So:

- Run enough session counts to find where vLLM's hit rate drops. That point scales with the KV pool: upstream's
  pool was 72 GB per GPU on two GPUs; an 80 GB card here has about 40 GB in total, so expect it at far fewer sessions.
- "Peak to peak" is the capacity gain. "Same load" past vLLM's peak is the overload-tolerance gain.
- Below the pressure point, expect roughly equal throughput and judge on uncached tokens and TTFT.

Differences from the paper's setup, any of which can change the numbers: a different model (27B, 262K native
context, no YaRN), your GPU instead of B200 (upstream says Qwen's in-chunk export falls back from a B200 kernel
to FlashInfer's stride checkpoints elsewhere), a KV pool sized by this repo's heuristic, vLLM's default CUDA graph
sizes (`CUDAGRAPH=upstream` restores theirs), and Flash-Next's eviction coefficients unless you run
`scripts/calibrate.sh`. Traces longer than `MAX_MODEL_LEN` are left out.

## Settings

Set in the environment or in `./fs.env`. Defaults are in `lib.sh`.

| Variable | Default | Meaning |
| --- | --- | --- |
| `MODEL_VARIANT` | `fp8` | `fp8` (Qwen/Qwen3.8-27B-FP8) or `bf16` (Qwen/Qwen3.8-27B), both pinned to a revision |
| `GPUS` | all | comma-separated GPU indices; their count is the tensor-parallel size |
| `KV_BYTES` | computed | KV pool per GPU, both engines: 92% of the card, minus weights, minus `KV_HEADROOM_GIB` (6) |
| `MAX_MODEL_LEN` | `262144` | lower it if the server says the pool cannot hold one full context |
| `PREFILL_COEFFS` | `2.230e-5 4.34e-6` | eviction cost coefficients; `results/coeffs.env` from `calibrate.sh` overrides the default |
| `FLASHSTATE_FLAGS` | all four switches | replace them for ablations, e.g. `FLASHSTATE_FLAGS=--mamba-frontier-checkpoint` |
| `EXTRA_SERVE_ARGS` | none | appended to both engines |
| `ENGINES` | `official flashstate` | run only one |

`scripts/serve.sh official|flashstate` starts one server and leaves it up for manual requests.

## If something fails

- **Server exits at startup**: the last 30 log lines are printed; the full log is `results/.../server.log`.
  "KV cache too small for max_model_len": lower `MAX_MODEL_LEN`. Out of memory: raise `KV_HEADROOM_GIB`.
  CUDA or driver errors: the pod's driver is older than 580.
- **`unrecognized arguments`** on the official engine: a flag in `build_serve` (`lib.sh`) is not in vLLM 0.33.0 as
  I assumed. The flag set is upstream's `benchmarks/configs/common.sh` plus its Qwen config; compare against those.
- **Report says the client re-tokenized**: the server ignored `return_token_ids`; prompts may then not extend
  exactly, which lowers hit rates on both engines.
- **`pressure` cannot read the pool size**: vLLM's startup line changed; set `SESSIONS` yourself.

## Layout

```
setup.sh              one-time install (upstream clone, environments, weights, traces)
lib.sh                settings; GPU detection; KV sizing; server start/stop
scripts/ab.sh         experiments 1 and 2          bench/turns.py, bench/compare.py
scripts/calibrate.sh  eviction coefficients        bench/calibrate_coeffs.py
scripts/replay.sh     one replay point             upstream aiperf + analysis/readings.py
scripts/sweep.sh      replay over session counts   bench/report.py
tests/                mock server and pipeline test (no GPU): bash tests/test_pipeline.sh
```

FlashStates, its vLLM fork and the replay client are Apache-2.0 and are fetched by `setup.sh`, not copied here;
`scripts/replay.sh` and the serve flags follow upstream's `benchmarks/run_local.sh` and `benchmarks/configs/`.
