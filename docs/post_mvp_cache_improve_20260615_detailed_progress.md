# Post-MVP MoE Cache Improvement Progress (2026-06-15)

Plan: `paper/docs/post_mvp_cache_improve_20260615_detailed.md`.

Implementation repo: `C:\code\llama.cpp.offload`.

## Scope

This progress entry covers Phase 0 through Phase 2 implementation and
validation:

- Phase 0 diagnostics and offline CSV analysis.
- Phase 1 EAMC fidelity audit and request-level `eamc-r` support.
- Phase 2 eviction-only predictor experiments.

It does not implement true expert prefetch, FineMoE ExpertMaps, or CPU DRAM
tiering.

## Code Changes

### Phase 0 Diagnostics

Added profiler fields for cache-policy diagnostics:

- `k_cold_miss`
- `k_evicted`
- `ssd_bytes`
- `ssd_reads`
- `eamc_score_fallbacks`
- `eamc_corpus_rows`
- `eamc_effective_rows`
- `victim_score_min`
- `victim_score_max`
- `victim_score_sum`
- `victim_score_count`

These are written in the normal MoE profile CSV and summarized by the
bench-style profile summary where applicable.

Implemented runtime wiring in `src/moe-offload/slot_pool.cpp`:

- Counts misses that use a free/cold slot separately from misses that require
  eviction.
- Aggregates victim scores selected by the existing predictor-based eviction
  path.
- Carries EAMC fallback and corpus-row gauges from predictor scoring into
  profiler rows.

Added offline analyzer:

```text
tests/moe-offload/analyze-profile.py
```

The analyzer consumes one or more MoE profiler CSV files and reports:

- phase-level hit/miss rates,
- miss, cold-slot, and eviction counts,
- SSD bytes/token when `ssd_bytes` is present,
- SSD/H2D/predictor/callback ms/token,
- EAMC row/fallback diagnostics,
- miss-heavy layers,
- deltas versus the first run.

It handles older CSV files too, but older files do not contain `ssd_bytes`,
`ssd_reads`, cold-miss, eviction, or victim-score columns, so those values show
as zero for old artifacts.

### Phase 1 EAMC Fidelity Audit

Added explicit predictor trace metadata:

- `predictor_trace_stats`
- `predictor::trace_stats()`

The compatibility EAMC predictor reports:

```text
trace_scope = iteration-batch-eam-v1
sidecar_magic = EAM1
sidecar_phase_tagged = false
request_level_rows = false
```

This documents the current `eamc` behavior as MVP EAMC-lite rather than
MoE-Infinity request-level EAMC. `eamc-lite` is now an explicit alias for this
compatibility behavior, and `eamc` remains accepted for existing scripts and
sidecars.

Added a request-level predictor mode:

```text
--moe-predictor eamc-r
```

`eamc-r` uses the new predictor sequence/iteration API:

```cpp
begin_sequence_or_turn();
begin_iteration(phase, token_count);
observe(layer, experts_used);
score(layer, expert);
end_iteration();
end_sequence_or_turn();
```

Runtime wiring now opens predictor iterations from the actual MoE callback phase
(`prefill` or `decode`). A prefill batch starts a new request-level sequence;
decode batches accumulate into that sequence until the next prefill, cache reset,
or predictor flush. This keeps `eamc`/`eamc-lite` as iteration-batch EAM1 while
allowing `eamc-r` to store request-level rows.

Added an `EAM2` sidecar format for `eamc-r`:

- Keeps the existing shape/capacity/top-k header.
- Adds replacement-policy metadata.
- Stores a per-row phase tag before each dense row.
- Loads existing `EAM1` sidecars as unknown-phase history.
- Saves `eamc-r` corpora as `EAM2`.

Replacement policy status:

- `fifo` remains the default for compatibility.
- `dedupe-nearest` is available behind diagnostic env var
  `LLAMA_MOE_EAMC_REPLACE=dedupe-nearest`.
- No default replacement behavior was changed for Phase 1.

Extended `tests/moe-offload/test-eamc-cosine.cpp`:

- Asserts the current EAMC metadata reports `EAM1` and
  `iteration-batch-eam-v1`.
- Asserts `EAM1` sidecars are not phase-tagged.
- Asserts current rows are not request-level rEAM rows.
- Verifies `eamc-lite` parses as an alias.
- Adds an audit test showing two `begin_request()/end_request()` intervals
  produce two `EAM1` rows, not one accumulated request-level row.
- Verifies `eamc-r` reports request-level `EAM2` metadata.
- Verifies one request with multiple decode iterations produces one decode rEAM
  row.
- Verifies separate request sequences produce separate rows.
- Verifies prefill/decode observations are stored as separate phase-tagged
  rows.
- Verifies partial current decode iEAM scores against historical request-level
  rEAM rows.
- Verifies `EAM2` round-trip preserves version, phases, row count, capacity,
  top-k, and dense values.
- Verifies `eamc-r` loads legacy `EAM1` sidecars and upgrades saved output to
  `EAM2`.
- Verifies diagnostic `dedupe-nearest` collapses near-identical request rows.

### Phase 2 Eviction Policy Experiments

Added opt-in predictor names:

```text
--moe-predictor lfu
--moe-predictor eamc-lfu
```

`lru` remains the default. No prefetch behavior was added.

Implemented pure LFU:

- Maintains a per-layer visit frequency for `(layer, expert)`.
- Uses frequency as the keep score.
- Uses a small LRU recency component only as a tie-break for equal frequency.

Implemented `eamc-lfu`:

- Reuses EAMC-lite prediction and `EAM1` sidecars.
- Maintains the same per-layer visit frequency.
- Scores resident experts as:

```text
keep_score = max(predicted_probability, eps) * max(freq, 1) + lru_tie_break
```

This keeps the Phase 2 hybrid eviction-only and opt-in. Existing `eamc`,
`eamc-lite`, and `eamc-r` behavior remains unchanged.

Extended tests:

- Verifies `lfu` parsing and trace metadata.
- Verifies LFU prefers higher-frequency experts.
- Verifies LFU uses LRU as the tie-break for equal frequencies.
- Verifies `eamc-lfu` parsing and trace metadata.
- Verifies `eamc-lfu` loads `EAM1` sidecars.
- Verifies `eamc-lfu` multiplies equal EAMC predictions by visit frequency.

## Files Changed In `llama.cpp.offload`

- `src/moe-offload/predictor.h`
- `src/moe-offload/predictor.cpp`
- `src/moe-offload/loader.cpp`
- `src/moe-offload/profiler.h`
- `src/moe-offload/profiler.cpp`
- `src/moe-offload/slot_pool.cpp`
- `common/arg.cpp`
- `tools/moe-bench/main.cpp`
- `tools/llama-bench/llama-bench.cpp`
- `tests/moe-offload/test-eamc-cosine.cpp`
- `tests/moe-offload/analyze-profile.py`
- `tests/.gitignore`

## Validation

Python analyzer syntax:

```powershell
python -m py_compile tests\moe-offload\analyze-profile.py
```

Passed.

Analyzer smoke on existing saved CSV artifacts:

```powershell
python tests\moe-offload\analyze-profile.py `
  --tokens prefill=256,decode=128 --top 2 `
  lru=tests\moe-offload\_out\cli-gap-bench-lru.csv `
  eamc=tests\moe-offload\_out\cli-gap-bench-eamc.csv
```

Passed. On those old artifacts, the analyzer reproduced the already-known
trend:

- EAMC decode hit rate was 0.8 percentage points lower than LRU.
- EAMC decode misses were 1013 higher than LRU.
- EAMC decode predictor cost was 1.63 ms/token higher than LRU.

The old artifacts do not include `ssd_bytes`, so byte/token fields are zero
for that smoke test.

Focused build:

```powershell
cmake --build build-moe-static --config Release --target `
  test-eamc-cosine -j 8

cmake --build build-moe-static --config Release --target `
  test-lru-eviction llama-moe-bench llama-bench llama-cli -j 8
```

Passed. Existing MSVC link warnings were unchanged:

- `LNK4098` static CRT conflict on `llama-moe-bench`, `llama-bench`, and
  `llama-cli`

Focused tests:

```powershell
.\build-moe-static\bin\Release\test-eamc-cosine.exe
.\build-moe-static\bin\Release\test-lru-eviction.exe
```

Passed.

Full MoE-offload CTest label:

```powershell
ctest --test-dir build-moe-static -C Release -L moe-offload --output-on-failure
```

Passed: 8/8 tests.

CLI smoke:

```powershell
.\build-moe-static\bin\Release\llama-bench.exe --help |
  Select-String -Pattern "moe-predictor"

.\build-moe-static\bin\Release\llama-moe-bench.exe --help
```

The help output includes:

```text
--moe-predictor <lru|lfu|eamc|eamc-lite|eamc-r|eamc-lfu>
```

Phase 1 `eamc-r` model smoke:

```powershell
.\build-moe-static\bin\Release\llama-moe-bench.exe `
  --model C:\AI\models\qwen\Qwen3.5-35B-A3B-Q4_K_M.moe.gguf `
  --pp 32 --tg 8 --repeat 1 `
  --moe-cache-vram-mb 8000 `
  --moe-predictor eamc-r `
  --moe-eamc-path tests\moe-offload\_out\phase1-eamc-r-smoke.eamc `
  --moe-profile-csv tests\moe-offload\_out\phase1-eamc-r-smoke.csv `
  --moe-profile-summary tests\moe-offload\_out\phase1-eamc-r-smoke.summary.txt `
  -ub 8
```

Passed. Artifacts:

- `tests/moe-offload/_out/phase1-eamc-r-smoke.eamc`
- `tests/moe-offload/_out/phase1-eamc-r-smoke.csv`
- `tests/moe-offload/_out/phase1-eamc-r-smoke.summary.txt`

Smoke result:

- `predictor: eamc-r`
- prefill predictor cost: `0.01 ms/token`
- decode predictor cost: `0.01 ms/token`
- sidecar magic: `EAM2`
- sidecar rows: `2`
- sidecar phases: `1` (`prefill`) and `2` (`decode`)
- sidecar shape: `40` layers x `256` experts, `capacity=1024`, `top_k=8`

This smoke is not a Phase 2 performance comparison; it only validates that
`eamc-r` runs in the real offload runtime, writes `EAM2`, and stays under the
Phase 1 predictor-overhead gate on the short model run.

Phase 2 analyzer:

```powershell
python tests\moe-offload\analyze-profile.py `
  --tokens prefill=256,decode=256 --top 4 `
  8000-lru=tests\moe-offload\_out\post-cache-p2-20260615-8000-lru-cold-reset.csv `
  8000-lfu=tests\moe-offload\_out\post-cache-p2-20260615-8000-lfu-cold-reset.csv `
  8000-eamc-lite=tests\moe-offload\_out\post-cache-p2-20260615-8000-eamc-lite-cold-reset.csv `
  8000-eamc-r=tests\moe-offload\_out\post-cache-p2-20260615-8000-eamc-r-cold-reset.csv `
  8000-eamc-lfu=tests\moe-offload\_out\post-cache-p2-20260615-8000-eamc-lfu-cold-reset.csv `
  12000-lru=tests\moe-offload\_out\post-cache-p2-20260615-12000-lru-cold-reset.csv `
  12000-lfu=tests\moe-offload\_out\post-cache-p2-20260615-12000-lfu-cold-reset.csv `
  12000-eamc-r=tests\moe-offload\_out\post-cache-p2-20260615-12000-eamc-r-cold-reset.csv `
  12000-eamc-lfu=tests\moe-offload\_out\post-cache-p2-20260615-12000-eamc-lfu-cold-reset.csv `
  > tests\moe-offload\_out\post-cache-p2-20260615-analysis.md
```

Passed.

## Phase 0 Benchmark Matrix

The required Phase 0 matrix was run on the model-bearing environment with the
final MVP guard stack:

```powershell
$env:LLAMA_MOE_SLOT_MMVQ='1'
$env:LLAMA_MOE_PREFILL_MMVQ='0'
$env:LLAMA_MOE_SLOT_GRAPHS='1'
$env:LLAMA_MOE_SLOT_GLU_FUSION='1'
$env:LLAMA_MOE_TOPK_FUSION_DIAG='0'
```

Common benchmark shape:

```text
--pp 256 --tg 256 --repeat 3
```

Artifacts:

- `tests/moe-offload/_out/post-cache-p0-20260615-8000-lru-cold-reset.*`
- `tests/moe-offload/_out/post-cache-p0-20260615-8000-lru-warm-hot.*`
- `tests/moe-offload/_out/post-cache-p0-20260615-8000-eamc-lite-cold-reset.*`
- `tests/moe-offload/_out/post-cache-p0-20260615-8000-eamc-lite-warm-hot.*`
- `tests/moe-offload/_out/post-cache-p0-20260615-12000-lru-cold-reset.*`
- `tests/moe-offload/_out/post-cache-p0-20260615-12000-eamc-lite-cold-reset.*`
- `tests/moe-offload/_out/post-cache-p0-20260615-analysis.md`

The warm/hot-start cases used a copied EAMC seed sidecar:

```text
tests/moe-offload/_out/post-cache-p0-20260615-seed.eamc
```

This avoided mutating the model-side EAMC sidecar during the benchmark.

### Matrix Results

| Case | Eff. ubatch | Prefill ms/tok | Decode TPOT | Prefill hit | Decode hit | Decode SSD GB | Decode predictor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8000 LRU cold/reset | 8 | 22.91 | 29.73 | 80.1% | 91.1% | 38.83 | 0.12 ms/tok |
| 8000 LRU warm/hot | 8 | 17.22 | 27.01 | 87.0% | 91.1% | 38.82 | 0.13 ms/tok |
| 8000 EAMC-lite cold/reset | 8 | 20.10 | 31.13 | 81.0% | 88.2% | 51.54 | 0.62 ms/tok |
| 8000 EAMC-lite warm/hot | 8 | 22.25 | 32.64 | 89.0% | 88.3% | 51.10 | 1.57 ms/tok |
| 12000 LRU cold/reset | 16 | 16.19 | 22.30 | 75.1% | 95.3% | 20.24 | 0.07 ms/tok |
| 12000 EAMC-lite cold/reset | 16 | 15.88 | 22.98 | 75.6% | 94.4% | 24.16 | 0.32 ms/tok |

### Interpretation

Phase 0 confirms that the current EAMC implementation should not be promoted as
the default cache policy.

At 8000 MiB:

- EAMC-lite slightly improved cold/reset prefill hit rate versus LRU
  (`81.0%` vs `80.1%`) and reduced prefill SSD bytes.
- EAMC-lite lost decode hit rate versus LRU (`88.2%` vs `91.1%`) and read
  substantially more decode SSD data (`51.54 GB` vs `38.83 GB`).
- Warm/hot-start made EAMC-lite prefill hit rate higher (`89.0%`) but did not
  help decode; decode stayed below LRU and predictor overhead rose to
  `1.57 ms/token`.
- LRU warm/hot-start was the best 8000 MiB case overall: `27.01 ms/token`
  decode TPOT.

At 12000 MiB:

- EAMC-lite slightly improved prefill per-token time (`15.88` vs
  `16.19 ms/token`) and prefill hit rate (`75.6%` vs `75.1%`).
- EAMC-lite still lost decode hit rate (`94.4%` vs `95.3%`) and read more SSD
  data (`24.16 GB` vs `20.24 GB`).
- LRU was still faster for decode: `22.30 ms/token` vs `22.98 ms/token`.

The new diagnostics also separate cold-slot misses from eviction misses:

- 8000 MiB cold/reset prefill had many cold-slot misses, while decode was
  almost entirely eviction-driven.
- Warm/hot-start removed cold-slot misses but did not improve 8000 MiB decode
  hit rate, which means the decode problem is eviction quality, not cold fill.
- EAMC-lite had many prefill fallback events in cold/reset because the corpus is
  still warming up, reinforcing that current EAMC-lite is not a strong
  first-request policy.

The Phase 0 conclusion is therefore:

1. Keep LRU as the default and baseline.
2. Treat current EAMC as `eamc-lite`.
3. Do not implement true prefetch on top of EAMC-lite without first improving
   eviction scoring or adding a paper-faithful request-level predictor.
4. Phase 2 therefore tested eviction-only alternatives, especially LFU and
   EAMC/LFU hybrids, before any speculative SSD prefetch.

## Phase 2 Benchmark Matrix

The required Phase 2 no-prefetch matrix was run on the same model-bearing
environment:

```powershell
$env:LLAMA_MOE_SLOT_MMVQ='1'
$env:LLAMA_MOE_PREFILL_MMVQ='0'
$env:LLAMA_MOE_SLOT_GRAPHS='1'
$env:LLAMA_MOE_SLOT_GLU_FUSION='1'
$env:LLAMA_MOE_TOPK_FUSION_DIAG='0'
```

Common benchmark shape:

```text
--pp 256 --tg 256 --repeat 3 --moe-reset-cache-between-repeats
```

Artifacts:

- `tests/moe-offload/_out/post-cache-p2-20260615-8000-lru-cold-reset.*`
- `tests/moe-offload/_out/post-cache-p2-20260615-8000-lfu-cold-reset.*`
- `tests/moe-offload/_out/post-cache-p2-20260615-8000-eamc-lite-cold-reset.*`
- `tests/moe-offload/_out/post-cache-p2-20260615-8000-eamc-r-cold-reset.*`
- `tests/moe-offload/_out/post-cache-p2-20260615-8000-eamc-lfu-cold-reset.*`
- `tests/moe-offload/_out/post-cache-p2-20260615-12000-lru-cold-reset.*`
- `tests/moe-offload/_out/post-cache-p2-20260615-12000-lfu-cold-reset.*`
- `tests/moe-offload/_out/post-cache-p2-20260615-12000-eamc-r-cold-reset.*`
- `tests/moe-offload/_out/post-cache-p2-20260615-12000-eamc-lfu-cold-reset.*`
- `tests/moe-offload/_out/post-cache-p2-20260615-analysis.md`

EAMC-family runs used per-run sidecars under `tests/moe-offload/_out/` to avoid
mutating the model-side sidecar.

### Matrix Results

| Cache | Predictor | Prefill ms/tok | Decode TPOT | Prefill hit | Decode hit | Decode SSD GB | Decode predictor |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8000 MiB | LRU | 24.89 | 31.77 | 80.1% | 91.1% | 38.83 | 0.12 ms/tok |
| 8000 MiB | LFU | 20.06 | 28.41 | 80.5% | 89.2% | 47.18 | 0.13 ms/tok |
| 8000 MiB | EAMC-lite | 20.16 | 30.85 | 81.0% | 88.2% | 51.54 | 0.58 ms/tok |
| 8000 MiB | EAMC-r | 20.03 | 25.76 | 81.0% | 92.1% | 34.54 | 0.15 ms/tok |
| 8000 MiB | EAMC-LFU | 20.07 | 26.85 | 80.9% | 91.5% | 37.01 | 0.48 ms/tok |
| 12000 MiB | LRU | 16.09 | 22.42 | 75.1% | 95.3% | 20.24 | 0.07 ms/tok |
| 12000 MiB | LFU | 15.80 | 21.46 | 75.5% | 95.5% | 19.77 | 0.06 ms/tok |
| 12000 MiB | EAMC-r | 15.75 | 21.00 | 75.6% | 95.9% | 17.85 | 0.08 ms/tok |
| 12000 MiB | EAMC-LFU | 15.77 | 21.57 | 75.6% | 95.4% | 19.90 | 0.24 ms/tok |

### Interpretation

Phase 2 acceptance is met by `eamc-r`.

At 8000 MiB:

- `eamc-r` improved decode hit rate over LRU from `91.1%` to `92.1%`.
- `eamc-r` improved decode TPOT from `31.77 ms/token` to `25.76 ms/token`.
- `eamc-r` reduced decode SSD reads from `38.83 GB` to `34.54 GB`.
- `eamc-r` predictor overhead was `0.15 ms/token`, only `0.03 ms/token` above
  LRU and far below the Phase 1 overhead gate.
- `eamc-lfu` also beat LRU on decode hit rate and TPOT, but was slower than
  `eamc-r` and cost more predictor time.
- LFU was faster than LRU on this run but had worse decode hit rate
  (`89.2%`), so it does not satisfy the Phase 2 acceptance rule.

At 12000 MiB:

- `eamc-r` improved decode hit rate over LRU from `95.3%` to `95.9%`.
- `eamc-r` improved decode TPOT from `22.42 ms/token` to `21.00 ms/token`.
- `eamc-r` reduced decode SSD reads from `20.24 GB` to `17.85 GB`.
- LFU also beat LRU at 12000 MiB, but `eamc-r` was best on decode hit rate,
  TPOT, and SSD bytes.
- `eamc-lfu` remained a valid opt-in experiment, but did not beat `eamc-r`.

The Phase 2 conclusion is therefore:

1. Keep LRU as the default until a broader workload set confirms the new policy.
2. Use `eamc-r` as the best measured experimental eviction policy and the
   preferred Phase 3 prefetch base.
3. Keep `lfu` as a low-overhead baseline.
4. Keep `eamc-lfu` opt-in, but do not build Phase 3 prefetch around it unless a
   later corpus shows it beating `eamc-r`.

## Current Phase 0 Status

Implemented:

- profiler diagnostics,
- analyzer,
- alias/help updates,
- validation on saved CSVs,
- fresh six-case model benchmark matrix,
- analyzer report on the fresh matrix.

Phase 0 is complete.

## Current Phase 1 Status

Implemented:

- explicit EAMC-lite semantics in code metadata,
- `eamc-lite` compatibility alias,
- `eamc-r` request-level predictor mode,
- sequence/turn and iteration-level predictor API,
- live runtime phase-aware predictor iteration wiring,
- `EAM2` request-level sidecar with phase tags and replacement-policy
  metadata,
- `EAM1` load compatibility for `eamc` and `eamc-r`,
- FIFO default replacement preserved,
- diagnostic `dedupe-nearest` replacement policy,
- tests locking down both EAMC-lite and request-level `eamc-r` semantics,
- model smoke validating `eamc-r` runtime execution and `EAM2` output.

Phase 1 is complete.

## Current Phase 2 Status

Implemented:

- per-layer visit-frequency tracking in the predictor layer,
- pure `lfu` predictor,
- `eamc-lfu` hybrid predictor,
- CLI/help wiring for `lfu` and `eamc-lfu`,
- focused predictor tests for LFU ordering and EAMC/LFU scoring,
- required no-prefetch benchmark matrix at 8000 MiB and 12000 MiB,
- analyzer report for the Phase 2 matrix.

Phase 2 is complete.

## Next Step

Proceed to Phase 3 true prefetch. Use `eamc-r` as the first prefetch base
because it is the only Phase 2 predictor that clearly beat LRU on decode hit
rate, TPOT, and SSD bytes at both tested cache sizes. Keep `lru` as the default
fallback and keep `lfu` / `eamc-lfu` as experimental comparator policies.
