# FlashState vs. vLLM on Qwen3.8-27B

An independent, controlled A/B test of [FlashStates](https://github.com/FutureMLS-Lab/FlashStates), a fork of
vLLM 0.33.0 that improves prefix caching for hybrid (attention + linear-attention) models in multi-turn agent
workloads. The [original claim](https://x.com/_weilix/status/2119837671946400165) was measured on GLM-5.3-Flash,
Qwen3.8-Flash-Next, Nemotron 3 Super and Kimi-K3.1-Pro using B200 and GB300 GPUs. This repository tests a model
the authors did not cover, **Qwen3.8-27B**, against unmodified vLLM 0.33.0. Both engines run on the same GPU with the
same KV memory and the same workload.

## Results at a glance

*One NVIDIA H200 · Qwen3.8-27B-FP8 · FP8 KV cache · one run per engine · 2026-11-10.*
*Full methodology, per-turn data and caveats are in **[report.md](report.md)**.*

**When all sessions fit in cache (mechanism test: 4 sessions × 8 turns)**

| Later turns (2–8) | vLLM 0.33.0 | FlashState | Change |
| --- | ---: | ---: | ---: |
| Uncached prompt tokens, p50 | 1,500 | **301** | −80% |
| Time to first token, p50 | 0.554 s | **0.282 s** | −49% |
| Time to first token, p90 | 0.788 s | **0.303 s** | −62% |
| Output throughput | 215 tok/s | **230 tok/s** | +7% |

FlashState resumes each turn from the exact end of the previous one. Only the 300 new tool-result tokens (plus one)
need computing. vLLM restores state only at block boundaries, so it recomputes a tail of 600 to 2,100 tokens on every turn.

**When sessions overflow the cache (pressure test: 123 sessions, about 1.5× the KV pool)**

| Later turns (2–8) | vLLM 0.33.0 | FlashState | Change |
| --- | ---: | ---: | ---: |
| Cache hit rate | 4.6% | **31.4%** | 6.8× |
| Total uncached prompt tokens | 27.9 M | **20.1 M** | −28% |
| Time to first token, p50 | 141.9 s | **98.8 s** | −30% |
| Output throughput | 83.8 tok/s | **106.3 tok/s** | +27% |
| Benchmark wall time | 39.2 min | **30.9 min** | −21% |

**Output parity did not meet the bar.** At temperature 0, FlashState produced token-identical outputs for 71% (10/14)
and 81% (403/498) of comparable prompts, short of the pre-registered 98% threshold. Divergence already appears on
some first turns, before any resume path is involved. FP8 numerics and batch scheduling are plausible causes, but
this experiment does not establish the cause, and no quality evaluation was run.

## Hypotheses

The thresholds were fixed in [`bench/compare.py`](bench/compare.py) before the run.

| | Prediction | Mechanism | Pressure |
| --- | --- | --- | --- |
| H1 | Later-turn uncached tokens p50 ≤ 0.6× vLLM | ✅ 0.20× | ❌ 0.99× (but total uncached −28%) |
| H2 | Later-turn TTFT p50 ≤ 0.9× vLLM | ✅ 0.51× | ✅ 0.70× |
| H3 | ≥ 98% identical outputs at temperature 0 | ❌ 71% | ❌ 81% |
| H4 | Cache behaviour holds up under overload | — | ✅ at one pressure point; no curve measured |

**Verdict:** FlashState's mechanism transfers to Qwen3.8-27B. It eliminates almost all recomputation between turns
and gives clear latency and throughput gains under cache pressure. Its outputs are not bit-identical to vLLM's,
and these runs do not reproduce or test the paper's headline multiples, which come from a trace-replay sweep.

## How the experiment works

Qwen3.8-27B has 16 full-attention layers, whose KV cache grows with context, and 48 linear-attention layers, which each
keep a fixed-size recurrent state. To resume from a cached prefix, a server needs both the KV blocks and a recurrent
state saved at exactly that position. FlashState adds four switches to make that possible more often:

| Switch | Purpose |
| --- | --- |
| `--mamba-frontier-checkpoint` | Keep a finished request's state at its exact end, so the next turn resumes there |
| `--mamba-prefill-export` | Save states inside a prefill chunk, not only at chunk ends |
| `--prefix-cache-eviction gds` | Evict by prefill seconds saved per block freed (GreedyDual-Size) |
| `--prefix-cache-leaf-last` | Evict a session's older states before its newest one |

The client ([`bench/turns.py`](bench/turns.py)) simulates agent sessions. Each session sends a long prompt, receives
200 output tokens, then appends those tokens plus 300 new "tool result" tokens and asks again, for 8 turns. Prompts
are sent as token IDs, so each turn extends the previous one exactly. The two engines get identical settings: the
same GPU, model files, KV allocation (94.4 GiB), scheduler limits, seeds and request stream.

| Test | Sessions | Initial context | What it isolates |
| --- | ---: | ---: | --- |
| `mechanism` | 4 | 22K tokens | Where the next turn resumes; everything fits in cache |
| `pressure` | 123 | 32K tokens | Eviction policy; total context ≈ 1.5× the KV pool |
| `sweep` (not run) | 4–N | Real traces | The paper's method: closed-loop replay of coding-agent traces |

## Reproduce it

**Hardware:** one GPU with at least 80 GB (H100, H200, or B200 recommended) and an NVIDIA driver of 580 or newer.
You also need about 100 GB of disk.
**Toolchain:** `nvcc` 12.9 or newer must be on `PATH` with `CUDA_HOME` set, because vLLM's DeepGEMM kernels compile at
startup. Images that ship CUDA 12.8 fail here. See [report.md](report.md#setup-fixes-and-reproducibility) for the
pip-installed CUDA 13 toolkit used in this run.

```bash
git clone https://github.com/skundu42/flash-state-inference-experiment.git
cd flash-state-inference-experiment
tmux new -s fs                   # runs are long; survive SSH drops

SKIP_TRACES=1 ./setup.sh         # 20–40 min: FlashStates, 3 Python envs, FP8 weights (31 GB)
scripts/calibrate.sh             # optional, ~5 min: eviction cost coefficients for this model and GPU
scripts/ab.sh mechanism          # ~10 min
scripts/ab.sh pressure           # ~1.5 h on an H200
```

Each run writes `results/<test>/report.md` plus, for each engine, the raw records, a summary, the server log, the
exact command, the environment and a `/metrics` snapshot.

To run the paper's trace-replay sweep (about 50 minutes per engine per session count), drop `SKIP_TRACES=1` and run
`scripts/sweep.sh 4 8 12 16`.

### Settings

Set these in the environment or in `fs.env`, which is git-ignored. The defaults are in [`lib.sh`](lib.sh).

| Variable | Default | Meaning |
| --- | --- | --- |
| `MODEL_VARIANT` | `fp8` | `fp8` (Qwen3.8-27B-FP8) or `bf16`; both pinned to a revision |
| `GPUS` | all | GPU indices to use; their count sets the tensor-parallel size |
| `MAX_MODEL_LEN` | `262144` | Lower this if the KV pool cannot hold one full context |
| `KV_HEADROOM_GIB` | `6` | Memory kept back for activations and CUDA graphs |
| `PREFILL_COEFFS` | upstream's | Eviction cost coefficients; `calibrate.sh` overrides them |
| `FLASHSTATE_FLAGS` | all four | Replaces the switches, for ablations |
| `ENGINES` | `official flashstate` | Runs only one engine |
| `HF_TOKEN` | — | Optional; avoids Hugging Face download rate limits |

## Limitations

- This is a single run per engine on a single H200, with one model variant and one seed. There are no repeats and no confidence intervals.
- The mechanism test is small: 28 later-turn requests per engine.
- The workload is synthetic random tokens, not natural language. No quality evaluation was run.
- The eviction coefficients were not calibrated for this model and GPU.
- The pressure test measures one overload point. It does not show where either engine peaks or collapses.
- The paper used B200 and GB300. On an H200, FlashState's in-chunk state export falls back from the B200 kernel to
  FlashInfer's stride checkpoints.

## Next steps

1. `scripts/calibrate.sh`, then rerun `pressure` for FlashState only (`ENGINES=flashstate`).
2. Repeat both tests 3 or more times to get variance.
3. Run a trace-replay sweep to find each engine's peak and compare peak-to-peak.
4. Ablate the four switches individually to attribute the gains.
5. Investigate output divergence: compare against vLLM run against itself, and test with BF16 weights and KV cache.

## Repository layout

```
setup.sh              one-time install: upstream clone, environments, weights, traces
lib.sh                settings, GPU detection, KV sizing, server start/stop
scripts/ab.sh         mechanism and pressure tests       bench/turns.py, bench/compare.py
scripts/calibrate.sh  eviction cost coefficients         bench/calibrate_coeffs.py
scripts/sweep.sh      trace-replay sweep                 scripts/replay.sh, bench/report.py
tests/                mock server and pipeline test (no GPU needed): bash tests/test_pipeline.sh
report.md             full write-up of the H200 run
```

FlashStates, its vLLM fork and the replay client are Apache-2.0. `setup.sh` fetches them rather than copying them here.
