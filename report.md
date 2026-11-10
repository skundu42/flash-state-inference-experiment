# FlashState versus vLLM: core A/B experiment

Run date: 2026-11-10 UTC. Both experiments completed on one NVIDIA H200. Only the mechanism and pressure presets ran; no calibration, trace replay, or session-count sweep ran.

## Findings

FlashState substantially improved prefix reuse when all sessions fit in cache. In the mechanism test, median later-turn uncached tokens fell from 1,500 to 301 (79.9% fewer), median time to first token fell from 0.554 to 0.282 seconds (49.1% lower), and whole-run output throughput increased by 6.7%.

At the tested pressure point, FlashState increased whole-run output throughput by 26.9%, from 83.78 to 106.30 tokens/s. Median later-turn time to first token fell by 30.4%, from 141.926 to 98.785 seconds. The later-turn token-weighted cache hit rate improved from 4.62% to 31.38%. However, median uncached tokens barely changed, from 32,400 to 31,900; the preset's H1 threshold was not met under pressure.

Exact output parity failed the predefined 98% threshold in both experiments. These runs demonstrate measured performance gains on this synthetic workload, but do not establish equivalent output tokens or equivalent model quality. Every request succeeded: 2,032 requests across both engines and both experiments, with zero re-tokenization fallbacks.

## Hardware, software, and controls

| Setting | Value |
| --- | --- |
| GPU | 1 × NVIDIA H200, 143,771 MiB reported memory |
| Tensor parallelism | 1; GPU index 0 |
| NVIDIA driver | 595.91.07; `nvidia-smi` reports CUDA compatibility 13.2 |
| Local CUDA compiler | CUDA 13.0, NVCC 13.0.88 |
| Python / PyTorch | Python 3.12 / PyTorch 2.14.0 |
| Official engine | Unmodified vLLM 0.33.0 |
| FlashState engine | vLLM 0.33.0 plus the pinned FlashStates Python overlay |
| FlashStates commit | `e2ea9a7864b41081a0147a3968e73bd5c6875a7d` |
| Harness commit before report edits | `1ec6a51d8123e3b591d3e3db4ab2240b89f876a0` |
| Model | `Qwen/Qwen3.8-27B-FP8` |
| Model revision configured for setup | `c3a462cc3d38cf351f3b26cbd763469eeb74ef81` |
| Local model directory | `models/qwen38-27b-fp8` |
| Maximum model length | 262,144 tokens |
| KV cache dtype | FP8 |
| KV allocation, each engine | 101,385,116,616 bytes (94.42 GiB) |
| Reported KV capacity, each engine | 2,929,042 tokens |
| Attention block size | 1,600 tokens, chosen by vLLM for hybrid state alignment |
| Maximum batched tokens | 16,384 |
| Long prefill threshold | 6,272 tokens |
| CUDA graphs | vLLM defaults; no upstream capture-size override |
| Sampling | Temperature 0; seed 0; exactly 200 output tokens per request; EOS ignored |
| Run order | Mechanism: official → FlashState; pressure: official → FlashState |

Both engines used prefix caching, the same GPU, model files, KV memory allocation, request settings, and deterministic prompt-generation seeds. The official environment was checked for the absence of FlashState checkpoint options. All files in the FlashState overlay were checked against the pinned source, with zero mismatches.

The common environment included `VLLM_HTTP_TIMEOUT_KEEP_ALIVE=900` and `CUDA_VISIBLE_DEVICES=0`. FlashState also used `VLLM_DETOKENIZE_PRIME_SUFFIX=1`, as specified by upstream. Its additional serving flags were:

```text
--mamba-frontier-checkpoint
--mamba-prefill-export
--prefix-cache-eviction gds
--prefix-cache-leaf-last
--prefix-cache-leaf-idle-seconds 60
--prefill-cost-coefficients 2.230e-5 4.34e-6
```

The eviction coefficients are the harness defaults derived from upstream's different Qwen configuration, not measurements calibrated for this model and H200.

## Workload and measurement

The client sends synthetic random vocabulary token IDs directly to `/v1/completions`. Sessions share a 2,000-token prefix and have independently generated session-specific context. After every response, the client appends its 200 output token IDs plus 300 new synthetic tool-result tokens. Each next prompt therefore extends that engine's previous prompt and output exactly.

Requests run in synchronized rounds: all sessions complete one turn before any session advances to the next. The client concurrency equals the session count. This is a synthetic agent-session workload, not a natural-language quality benchmark or the paper's trace replay.

| Setting | Mechanism | Pressure |
| --- | ---: | ---: |
| Sessions / client concurrency | 4 | 123 |
| Turns per session | 8 | 8 |
| Shared initial prefix | 2,000 tokens | 2,000 tokens |
| Unique initial context per session | 20,000 tokens | 30,000 tokens |
| Total first prompt per session | 22,000 tokens | 32,000 tokens |
| Prompt growth per subsequent turn | 500 tokens | 500 tokens |
| Final prompt per session | 25,500 tokens | 35,500 tokens |
| Output tokens per request | 200 | 200 |
| Requests per engine | 32 | 984 |
| Later-turn requests per engine | 28 | 861 |
| Generated tokens per engine | 6,400 | 196,800 |

Pressure session count was computed once from official vLLM's reported pool and reused for FlashState:

```text
per_session_budget = 2,000 + 30,000 + 8 × (300 + 200) = 36,000 tokens
sessions = ceil(1.5 × 2,929,042 / 36,000) = 123
```

This is the harness's approximate sizing heuristic, not an exact accounting of physical cache blocks: shared prefixes, state checkpoints, alignment, and scheduling affect actual occupancy.

Time to first token (TTFT) is measured by the client from request start to the first streamed choice. It includes queuing and HTTP overhead. Later-turn measurements pool turns 2–8. Cache hit rate is total cached prompt tokens divided by total prompt tokens over those turns. Uncached counts come from the server's prompt-token usage details. Whole-run throughput is total generated output tokens divided by client benchmark wall time.

Server launch, model loading, and pre-request warmup are excluded from benchmark wall time. First-request compilation that happens after requests begin is included; FlashState's mechanism log recorded inference-time checkpoint-kernel compilation. The clients received output IDs directly on every request, so neither engine used the re-tokenization fallback.

## Mechanism results

| Metric | vLLM | FlashState | FlashState / vLLM |
| --- | ---: | ---: | ---: |
| Uncached prompt tokens, later-turn p50 | 1,500 | 301 | 0.201× |
| Uncached prompt tokens, later-turn p90 | 2,100 | 301 | 0.143× |
| Cache hit rate, later turns | 94.29% | 98.75% | 1.047× |
| Time to first token, later-turn p50 (s) | 0.554 | 0.282 | 0.509× |
| Time to first token, later-turn p90 (s) | 0.788 | 0.303 | 0.384× |
| Time to first token, first-turn p50 (s) | 5.661 | 5.699 | 1.007× |
| Output throughput, whole run (tokens/s) | 215.39 | 229.84 | 1.067× |
| Benchmark wall time (s) | 29.71 | 27.85 | 0.937× |
| Successful requests / total | 32/32 | 32/32 | — |
| Requests re-tokenized | 0 | 0 | — |
| Exact output parity for comparable prompts | Reference | 10/14 (71.43%) | Fails ≥98% threshold |

The 301 uncached tokens closely track the 300 newly supplied tool tokens. FlashState reuses nearly all of the previous prompt and its generated continuation. vLLM repeatedly recomputes a larger tail, consistent with block-aligned hybrid-state checkpoints. This interpretation is supported by the cache counters; the experiment does not independently trace the internal checkpoint selected for every request.

## Pressure results

| Metric | vLLM | FlashState | FlashState / vLLM |
| --- | ---: | ---: | ---: |
| Uncached prompt tokens, later-turn p50 | 32,400 | 31,900 | 0.985× |
| Uncached prompt tokens, later-turn p90 | 33,900 | 33,900 | 1.000× |
| Cache hit rate, later turns | 4.62% | 31.38% | 6.787× |
| Time to first token, later-turn p50 (s) | 141.926 | 98.785 | 0.696× |
| Time to first token, later-turn p90 (s) | 259.699 | 191.845 | 0.739× |
| Time to first token, first-turn p50 (s) | 139.662 | 138.579 | 0.992× |
| Output throughput, whole run (tokens/s) | 83.78 | 106.30 | 1.269× |
| Benchmark wall time (s) | 2,349.10 | 1,851.40 | 0.788× |
| Successful requests / total | 984/984 | 984/984 | — |
| Requests re-tokenized | 0 | 0 | — |
| Exact output parity for comparable prompts | Reference | 403/498 (80.92%) | Fails ≥98% threshold |

vLLM's client benchmark took 39.15 minutes; FlashState's took 30.86 minutes, a reduction of 8.30 minutes (21.2%). Startup and warmup took additional time.

The near-equal median uncached-token count does not contradict the improved aggregate hit rate. A majority of requests can still miss most of their context while a subset achieves much better reuse. Across all later-turn requests, vLLM processed 27,920,400 uncached prompt tokens versus FlashState's 20,087,545, a reduction of 28.1%. Mean uncached tokens per request fell from 32,427.9 to 23,330.5, even though the median only fell by 1.5%.

This supports better aggregate cache retention and throughput at this particular overload point. It does not establish the session count at which either engine peaks or collapses, a peak-to-peak capacity gain, or a performance curve; those require the sweep that was intentionally excluded.

## Per-turn measurements

Each row includes all sessions for that turn. TTFT values are medians in seconds. All rows completed without request errors.

### Mechanism

| Turn | vLLM uncached p50 | FlashState uncached p50 | vLLM TTFT p50 (s) | FlashState TTFT p50 (s) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 21,200 | 21,200 | 5.661 | 5.699 |
| 2 | 1,700 | 301 | 0.701 | 0.310 |
| 3 | 600 | 301 | 0.449 | 0.280 |
| 4 | 1,100 | 301 | 0.540 | 0.277 |
| 5 | 1,600 | 301 | 0.678 | 0.291 |
| 6 | 2,100 | 301 | 0.845 | 0.290 |
| 7 | 1,000 | 301 | 0.547 | 0.293 |
| 8 | 1,500 | 301 | 0.681 | 0.259 |

### Pressure

| Turn | vLLM uncached p50 | FlashState uncached p50 | vLLM TTFT p50 (s) | FlashState TTFT p50 (s) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 30,400 | 30,400 | 139.662 | 138.579 |
| 2 | 30,900 | 30,900 | 133.454 | 55.180 |
| 3 | 31,400 | 31,400 | 136.425 | 92.516 |
| 4 | 31,900 | 31,900 | 139.687 | 88.983 |
| 5 | 32,400 | 32,400 | 142.197 | 108.323 |
| 6 | 32,900 | 32,900 | 144.773 | 101.612 |
| 7 | 33,400 | 33,400 | 147.121 | 102.413 |
| 8 | 33,900 | 33,900 | 149.646 | 105.384 |

## Predefined hypotheses

Thresholds come from `bench/compare.py` and were not changed after observing the results.

| Hypothesis / criterion | Mechanism | Pressure |
| --- | --- | --- |
| H1: later-turn uncached-token p50 ratio ≤0.60 | Supported: 0.201 | Not supported: 0.985 |
| H2: later-turn TTFT p50 ratio ≤0.90 | Supported: 0.509 | Supported: 0.696 |
| H3: ≥98% exact output matches for comparable prompts | Not supported: 10/14, 71.43% | Not supported: 403/498, 80.92% |
| H4: retain useful cache behavior under overload | Not tested by this preset | Improved aggregate hit rate and throughput at one pressure point; no collapse curve measured |

## Exact output parity

The comparison checks outputs while a session's prompts remain identical across engines. After the first differing output, later prompts in that session differ because each client feeds back its own engine's response. Those later requests are excluded from parity, rather than being treated as independently comparable.

For mechanism, all four first-turn responses matched. First divergence occurred at turns 4, 6, 2, and 2 for sessions 0–3 respectively. There were 10 identical outputs among 14 comparable requests; none of the four sessions matched through all eight turns.

For pressure, 107 of 123 first-turn responses matched. Twenty-eight sessions matched through all eight turns. Overall, 403 of 498 comparable requests matched. The 984 requests per engine are therefore not 984 matched-prompt parity comparisons.

The experiment does not identify the cause of divergence. FP8 numerical sensitivity, different batching and scheduling, and different recurrent-state resume paths are plausible contributors, not established explanations. Differences already occur on the first pressure turn, so they cannot all be attributed to resuming a previous turn's checkpoint. Server logs also warn that the FP8 KV cache uses default/uncalibrated scaling factors. Temperature zero does not by itself establish bitwise-equivalent outputs across implementations and execution paths.

No natural-language or task-quality evaluation was performed. Once sessions diverge, performance comparisons retain the same lengths, seeds, and synthetic workload construction, but subsequent prompt token contents are no longer identical across engines.

## Setup fixes and reproducibility

The host's original CUDA toolkit was 12.8. The first server attempts failed because this vLLM/DeepGEMM build requires NVCC ≥12.9. A local CUDA 13.0 toolkit was installed under `.cuda13/` without changing either engine's model or benchmark source.

The local toolchain uses `nvidia-cuda-nvcc==13.0.88`, `nvidia-cuda-crt==13.0.88`, `nvidia-nvvm==13.0.88`, `nvidia-cuda-runtime==13.0.96`, `nvidia-cuda-cccl==13.0.85`, and Ninja 1.13.2. Missing CUDA library headers were supplied from the official environment's installed NVIDIA CUDA 13 packages; `lib64` and `libcudart.so` symlinks supplied the layout expected by FlashInfer. These settings apply to both engines through the ignored local `fs.env`:

```bash
CUDA_HOME=/root/flash-state-inference-experiment/.cuda13/nvidia/cu13
PATH=/root/flash-state-inference-experiment/.cuda13/nvidia/cu13/bin:/root/flash-state-inference-experiment/.cuda13/bin:$PATH
```

With the completed local setup, the executed experiment commands were:

```bash
bash scripts/ab.sh mechanism
bash scripts/ab.sh pressure
```

Use fresh output directories or preserve existing results before rerunning: the scripts write to the same result paths. On a fresh host, `SKIP_TRACES=1 ./setup.sh` prepares the repository without downloading the replay traces; a suitable CUDA toolkit is still required.

## Limits of the result

- One run per preset and engine, one GPU, one model variant, and one seed. No repetition, confidence intervals, randomized execution order, or cross-hardware validation.
- Small mechanism sample: only 28 later-turn requests per engine. Pressure has 861 later-turn requests per engine but still uses one synthetic run.
- Fixed run order and shared compiler caches can affect cold-start work. Reported benchmark times exclude server startup, but include any compilation triggered during requests.
- No calibrated model-specific eviction coefficients, feature ablations, trace replay, session-count sweep, or quality evaluation.
- Cache hit rates and uncached-token counts depend on the server's reported usage; they were not independently reconstructed from kernel traces.
- The H200 and Qwen3.8-27B FP8 workload differ from the upstream paper's hardware and model configurations. These results do not validate its published multiples.

## Saved artifacts

| Artifact | Location |
| --- | --- |
| Minimal comparison table | [README.md](README.md) |
| Mechanism generated report | [results/mechanism/report.md](results/mechanism/report.md) |
| Pressure generated report | [results/pressure/report.md](results/pressure/report.md) |
| Mechanism raw records and summaries | [official](results/mechanism/official/) · [FlashState](results/mechanism/flashstate/) |
| Pressure raw records and summaries | [official](results/pressure/official/) · [FlashState](results/pressure/flashstate/) |
| Mechanism client log | [results/mechanism-run.log](results/mechanism-run.log) |
| Pressure client log | [results/pressure-run.log](results/pressure-run.log) |
| Pressure session count | [results/pressure/sessions.txt](results/pressure/sessions.txt) |
| Hardware/version metadata and local settings | [results/run-metadata.json](results/run-metadata.json) |
| Earlier startup failures from the preexisting checkout | [results/prior-startup-failures/](results/prior-startup-failures/) |

Every engine directory contains `records.jsonl`, `summary.json`, `server.log`, `command.txt`, `env.txt`, and `metrics.prom`. `env.txt` records the explicitly added engine environment; inherited local toolkit settings are recorded separately in the run metadata. Raw results, model files, virtual environments, and `fs.env` are local ignored artifacts, so these links require this checkout's retained results directory; they are not automatically included in a Git commit.

Both servers were stopped by the harness after completion. The GPU reported 0 MiB allocated when checked afterward.
