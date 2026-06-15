# Final MoE Offload MVP Closeout Progress (2026-06-15)

Plan: `paper/docs/implementation_plan_final_mvp_closeout_20260615.md`.

## Scope Update

This closeout is now `llama-cli` first. The goal is for humans to experience
the MoE offload speedup in interactive chat, not only through
`llama-moe-bench` summaries.

Out of scope for this closeout:

- Post-MVP cache-management research in
  `post_mvp_plan_cache_improve_20260615.md`.

## Initial Plan Changes

- Removed non-CLI phases, validation, deliverables, and closeout criteria from
  the final MVP closeout plan.
- Prioritized `llama-cli` performance parity with `llama-moe-bench`.
- Added explicit README and known-issues documentation requirements for the
  eventual `llama-cli` fix.
- Added a requirement to preserve a conservative ubatch-1 fallback even if the
  fast interactive path is promoted.

## Current Implementation Status

Implemented the CLI-first closeout path.

Code changes in `C:\code\llama.cpp.offload`:

- Added `--moe-fast-paths` for `llama-cli`, with `LLAMA_MOE_FAST_PATHS=1` as
  the environment equivalent.
- `--moe-fast-paths` applies the accepted fast guard stack:
  `LLAMA_MOE_SLOT_MMVQ=1`, `LLAMA_MOE_SLOT_GRAPHS=1`,
  `LLAMA_MOE_SLOT_GLU_FUSION=1`, `LLAMA_MOE_PREFILL_MMVQ=0`, and
  `LLAMA_MOE_TOPK_FUSION_DIAG=0`.
- The default `llama-cli --moe-offload` path still forces `n_ubatch=1` when no
  streaming ubatch override is set.
- The fast profile no longer applies the CLI ubatch-1 clamp, so the runtime
  auto-sizes streaming ubatch from the cache budget.
- `llama-cli` now prints the effective MoE `n_ubatch` after model load.
- `llama-cli --moe-profile-summary` now writes the bench-style MoE summary at
  process exit, using the existing `llama_moe::format_summary()` helper and
  aggregating CLI turn timings.
- The shared summary formatter now reports VRAM as unavailable when a caller
  does not provide memory samples, instead of printing a misleading 0 GB peak.

Documentation changes:

- Updated `docs/moe-offload/README.md` with the human-facing fast
  `llama-cli` command, profiling command, fallback behavior, and fast-profile
  guard explanation.
- Updated `docs/moe-offload/known-issues.md` so the old ubatch-1 caveat now
  reflects the explicit fast profile and conservative fallback.

## Validation

Builds:

```powershell
cmake --build build-moe-static --config Release --target llama-cli -j 8
cmake --build build-moe\tools\cli --config Release --target llama-cli -j 8
```

Both passed.

Focused model-dependent fast CLI run:

```powershell
.\build-moe-static\bin\Release\llama-cli.exe `
  --model C:/AI/models/qwen/Qwen3.5-35B-A3B-Q4_K_M.moe.gguf `
  --moe-offload --moe-cache-vram-mb 12000 --moe-predictor lru `
  --moe-fast-paths `
  --moe-profile-csv tests\moe-offload\_out\cli-closeout-fast.csv `
  --moe-profile-summary tests\moe-offload\_out\cli-closeout-fast.summary.txt `
  --jinja --reasoning off --temp 0 --seed 42 --simple-io --no-warmup `
  -c 4096 -n 32 -sys "You are a helpful assistant."
```

Result:

- Answered `what is the capital of France?` with Paris.
- Reported `llama-cli: MoE effective n_ubatch=16 (fast paths)`.
- Summary reported `ubatch: requested=512 effective=16  slots=145/256`.
- Summary emitted bench-style TTFT/TPOT and prefill/decode I/O breakdowns.

Correctness and smoke gates:

```powershell
ctest --test-dir build-moe-static -C Release -L moe-offload --output-on-failure
```

Passed: 8/8 tests.

```powershell
$env:LLAMA_MOE_FAST_PATHS='1'
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\tests\moe-offload\test-llama-cli-chat.ps1 `
  -Bin "$PWD\build-moe-static\bin\Release\llama-cli.exe" `
  -CacheMb 12000 -Predictor lru -NPredict 64 -StreamingUBatch 0
```

Passed: formatted `llama-cli` chat smoke.

Earlier in this closeout run, the accepted guard stack also passed the raw
golden-logit gate with `max|d| = 0` for the short `Hello` prompt, 4000 MiB
streaming cache, ubatch 8, and 8 predicted tokens.

## Residual Notes

- The fast CLI summary is now comparable in structure to `llama-moe-bench`, but
  CLI prompt shape, chat loop, console I/O, and cache state still differ from
  synthetic bench runs. Use matched cache/predictor/guard/cache-state settings
  when comparing numbers.
- The short one-turn CLI timing is noisy. The important closeout behavior is
  that humans can run one documented `llama-cli` command, see effective
  auto-sized ubatch, get coherent chat output, and collect the same category of
  MoE breakdown as the bench report.

## CLI Decode Gap Investigation

The current saved `llama-cli` fast-path run does not prove a decode kernel
regression. It is not comparable to the Phase K `llama-moe-bench` result:

| Field | `llama-cli` fast smoke | Phase K `llama-moe-bench` |
| --- | ---: | ---: |
| Predictor | LRU | EAMC |
| Prompt tokens | 31 | 256 |
| Generated tokens | 9 | 128 x 3 repeats |
| Effective ubatch | 16 | 16 |
| Decode hit rate | 83.7% | 92.4% |
| Decode SSD read | 65.39 ms/token | 8.03 ms/token |
| Decode H2D | 12.62 ms/token | 6.64 ms/token |
| Decode GPU compute | 12.13 ms/token | 11.16 ms/token |
| Decode TPOT | 89.15 ms/token | 26.78 ms/token |

Interpretation:

- The `gpu_compute` bucket is close: 12.13 ms/token in CLI versus
  11.16 ms/token in bench.
- The large TPOT gap is currently dominated by cache and I/O behavior, not by
  the guarded decode MMVQ/graph/GLU compute path.
- The CLI smoke generated only 9 tokens, so per-token wall time is very noisy.
- The CLI prompt is a Jinja chat prompt plus frontend/server/streaming console
  path; the bench prompt is a synthetic repeated-text prompt and direct
  decode loop.

### Matched Results

After rerunning matched 12000 MiB cases with the accepted guard stack and a
128-token generation budget, the CLI path was close enough to bench to close
the remaining decode concern:

| Predictor | `llama-moe-bench` TPOT | `llama-cli` TPOT | Notes |
| --- | ---: | ---: | --- |
| LRU | 27.26 ms/token | 28.36 ms/token | close match |
| EAMC | 31.27 ms/token | 34.59 ms/token | still close, but a bit slower |

For the EAMC comparison, the CLI prompt still began from a short interactive
chat exchange, so some residual difference remains in prompt/cache-route
behavior. The important closeout result is that the CLI fast profile does reach
bench-like decode speed when measured on a matched workload.

### Phase G Diagnostic Plan

Run matched workloads before changing more code:

```powershell
$env:LLAMA_MOE_SLOT_MMVQ='1'
$env:LLAMA_MOE_SLOT_GRAPHS='1'
$env:LLAMA_MOE_SLOT_GLU_FUSION='1'
$env:LLAMA_MOE_PREFILL_MMVQ='0'
$env:LLAMA_MOE_TOPK_FUSION_DIAG='0'

.\build-moe-static\bin\Release\llama-moe-bench.exe `
  --model C:/AI/models/qwen/Qwen3.5-35B-A3B-Q4_K_M.moe.gguf `
  --pp 256 --tg 128 --repeat 3 `
  --moe-cache-vram-mb 12000 --moe-predictor lru `
  --moe-reset-cache-between-repeats `
  --moe-profile-csv tests\moe-offload\_out\cli-gap-bench-lru.csv `
  --moe-profile-summary tests\moe-offload\_out\cli-gap-bench-lru.summary.txt
```

```powershell
.\build-moe-static\bin\Release\llama-cli.exe `
  --model C:/AI/models/qwen/Qwen3.5-35B-A3B-Q4_K_M.moe.gguf `
  --moe-offload --moe-cache-vram-mb 12000 --moe-predictor lru `
  --moe-fast-paths `
  --moe-profile-csv tests\moe-offload\_out\cli-gap-cli-lru.csv `
  --moe-profile-summary tests\moe-offload\_out\cli-gap-cli-lru.summary.txt `
  --jinja --reasoning off --temp 0 --seed 42 --simple-io --no-warmup `
  -c 4096 -n 128 -sys "You are a helpful assistant."
```

Feed a prompt that is likely to produce the full 128-token budget, then `/exit`.
Repeat the pair with `--moe-predictor eamc`.

Acceptance for the investigation:

- Compare only runs with the same cache budget, predictor, guard stack, cache
  reset/warm policy, and comparable generated-token count.
- If CLI `gpu_compute`, H2D, stall, and predictor are close but TPOT is still
  worse, treat the remaining gap as frontend, cache-route, or prompt-shape
  overhead and document the bucket.
- If CLI MoE profile buckets are worse under matched settings, open the next
  implementation phase against the specific bucket rather than treating the
  whole CLI as slow.

### Phase G Outcome

No additional code fix was required beyond the existing fast CLI profile and
profile-summary plumbing. The closeout now treats the remaining difference as a
workload/cache-route effect rather than a CLI performance bug.

## CLI Prefill Gap Investigation

The same workload-matching issue affected prefill. The apparent CLI prefill
gap came from comparing a short chat prompt against the synthetic repeated
bench prompt.

Measured 12000 MiB LRU cases with the accepted guard stack:

| Case | Prompt tokens | Prefill TPOT | Prefill hit rate | Artifact |
| --- | ---: | ---: | ---: | --- |
| Short raw bench prompt | 27 | 76.95 ms/token | 24.0% | `cli-prefill-bench-lru-shortprompt.summary.txt` |
| Short CLI chat prompt | 51 | 71.70 ms/token | 41.7% | `cli-gap-cli-lru.summary.txt` |
| Long repeated CLI chat prompt | 269 | 21.87 ms/token | 80.3% | `cli-prefill-cli-lru-longhello.summary.txt` |
| Single-repeat repeated bench prompt | 256 | 28.30 ms/token | 75.1% | `cli-prefill-bench-lru-pp256.summary.txt` |

Interpretation:

- Short prompts are slow per token in both tools because cold slot loads are
  not amortized and the prompt has lower expert locality.
- The long repeated CLI prompt reaches the same class of prefill speed as the
  repeated bench prompt.
- The faster repeat-averaged benchmark prefill number should not be used as
  the expected speed for a first-turn short chat prompt.

Experimental prefill MMVQ check:

- `LLAMA_MOE_PREFILL_MMVQ=1` reduced the short CLI TTFT from 3656.9 ms to
  3534.9 ms, but decode TPOT worsened from 28.36 to 30.50 ms/token.
- This is not a safe final-MVP default, so `--moe-fast-paths` continues to
  force `LLAMA_MOE_PREFILL_MMVQ=0`.

Outcome:

- No CLI code fix was needed for prefill parity under matched workload.
- Documentation now tells users to compare prefill with matched prompt length,
  prompt locality, cache state, and repeat policy.

Final validation after Phase G:

```powershell
ctest --test-dir build-moe-static -C Release -L moe-offload --output-on-failure
```

Passed: 8/8 tests.

```powershell
$env:LLAMA_MOE_FAST_PATHS='1'
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\tests\moe-offload\test-llama-cli-chat.ps1 `
  -Bin "$PWD\build-moe-static\bin\Release\llama-cli.exe" `
  -CacheMb 12000 -Predictor lru -NPredict 64 -StreamingUBatch 0
```

Passed: formatted `llama-cli` chat smoke.

Documentation updated:

- `docs/moe-offload/README.md` now reports the matched Phase G CLI-vs-bench
  decode numbers and explains how to compare TPOT and prefill fairly.
- `docs/moe-offload/known-issues.md` now records that the fast profile reaches
  bench-like decode speed under matched settings, and records the prefill
  workload/locality caveat instead of treating it as a CLI-specific regression.
