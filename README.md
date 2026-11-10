| Test | Metric | vLLM 0.33.0 | FlashState | FlashState / vLLM |
| --- | --- | ---: | ---: | ---: |
| Mechanism | Uncached prompt tokens, later-turn p50 | 1,500 | 301 | 0.201× |
| Mechanism | Uncached prompt tokens, later-turn p90 | 2,100 | 301 | 0.143× |
| Mechanism | Cache hit rate, later turns | 94.29% | 98.75% | 1.047× |
| Mechanism | Time to first token, later-turn p50 (s) | 0.554 | 0.282 | 0.509× |
| Mechanism | Time to first token, later-turn p90 (s) | 0.788 | 0.303 | 0.384× |
| Mechanism | Time to first token, first-turn p50 (s) | 5.661 | 5.699 | 1.007× |
| Mechanism | Output throughput, whole run (tokens/s) | 215.39 | 229.84 | 1.067× |
| Mechanism | Benchmark wall time (s) | 29.71 | 27.85 | 0.937× |
| Mechanism | Successful requests / total | 32/32 | 32/32 | — |
| Mechanism | Requests re-tokenized | 0 | 0 | — |
| Mechanism | Exact output parity for comparable prompts | Reference | 10/14 (71.43%) | Fails ≥98% threshold |
| Pressure | Uncached prompt tokens, later-turn p50 | 32,400 | 31,900 | 0.985× |
| Pressure | Uncached prompt tokens, later-turn p90 | 33,900 | 33,900 | 1.000× |
| Pressure | Cache hit rate, later turns | 4.62% | 31.38% | 6.787× |
| Pressure | Time to first token, later-turn p50 (s) | 141.926 | 98.785 | 0.696× |
| Pressure | Time to first token, later-turn p90 (s) | 259.699 | 191.845 | 0.739× |
| Pressure | Time to first token, first-turn p50 (s) | 139.662 | 138.579 | 0.992× |
| Pressure | Output throughput, whole run (tokens/s) | 83.78 | 106.30 | 1.269× |
| Pressure | Benchmark wall time (s) | 2,349.10 | 1,851.40 | 0.788× |
| Pressure | Successful requests / total | 984/984 | 984/984 | — |
| Pressure | Requests re-tokenized | 0 | 0 | — |
| Pressure | Exact output parity for comparable prompts | Reference | 403/498 (80.92%) | Fails ≥98% threshold |
