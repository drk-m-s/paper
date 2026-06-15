# Post-MVP MoE Cache Improvement Progress (2026-06-15)

Plan: `paper/docs/post_mvp_cache_improve_20260615_detailed.md`.

Implementation repo: `C:\code\llama.cpp.offload`.

## Scope

This progress entry covers Phase 0 and Phase 1 implementation scaffolding:

- Phase 0 diagnostics and offline CSV analysis.
- Phase 1 EAMC fidelity audit foundations.

It does not implement new eviction policies, true expert prefetch, FineMoE
ExpertMaps, or CPU DRAM tiering.

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

The current EAMC predictor now reports:

```text
trace_scope = iteration-batch-eam-v1
sidecar_magic = EAM1
sidecar_phase_tagged = false
request_level_rows = false
```

This intentionally documents the current behavior as MVP EAMC-lite rather than
paper-complete MoE-Infinity request-level EAMC.

Added an explicit predictor alias:

```text
--moe-predictor eamc-lite
```

`eamc` remains accepted for compatibility. The alias is accepted by common args,
`llama-moe-bench`, and `llama-bench`.

Extended `tests/moe-offload/test-eamc-cosine.cpp`:

- Asserts the current EAMC metadata reports `EAM1` and
  `iteration-batch-eam-v1`.
- Asserts `EAM1` sidecars are not phase-tagged.
- Asserts current rows are not request-level rEAM rows.
- Verifies `eamc-lite` parses as an alias.
- Adds an audit test showing two `begin_request()/end_request()` intervals
  produce two `EAM1` rows, not one accumulated request-level row.

No `EAM2` sidecar was added because runtime row semantics were not changed in
this phase. The next implementation phase can add `EAM2` when request-level
rEAM storage is actually introduced.

## Files Changed In `llama.cpp.offload`

- `src/moe-offload/predictor.h`
- `src/moe-offload/predictor.cpp`
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
  test-eamc-cosine test-lru-eviction llama-moe-bench -j 8
```

Passed. Existing MSVC/link warnings were unchanged:

- `C4244` in `llama-graph.cpp`
- `LNK4098` static CRT conflict on `llama-moe-bench`

Focused tests:

```powershell
.\build-moe-static\bin\Release\test-eamc-cosine.exe
.\build-moe-static\bin\Release\test-lru-eviction.exe
ctest --test-dir build-moe-static -C Release -R "test-eamc-cosine|test-lru-eviction" --output-on-failure
```

Passed.

Full MoE-offload CTest label:

```powershell
ctest --test-dir build-moe-static -C Release -L moe-offload --output-on-failure
```

Passed: 8/8 tests.

CLI rebuild after common-arg changes:

```powershell
cmake --build build-moe-static --config Release --target llama-cli -j 8
```

Passed, with the existing `LNK4098` static CRT warning.

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
4. Phase 2 should start with eviction-only alternatives, especially LFU and
   EAMC/LFU hybrids, before any speculative SSD prefetch.

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
- `eamc-lite` alias,
- tests locking down that current `EAM1` rows are not request-level rEAM rows,
- compatibility preserved for existing `eamc` and `EAM1` sidecars.

Deferred:

- `EAM2` request-level rEAM sidecar,
- sequence/turn-level predictor API in live runtime,
- paper-faithful `eamc-r` predictor mode,
- dedupe/diverse EAMC replacement policy.

## Next Step

Proceed to Phase 2 eviction-only policies. Based on Phase 0, start with LFU and
EAMC/LFU hybrids rather than true SSD prefetch. The first Phase 2 acceptance
gate should be beating LRU decode hit rate and TPOT under the same 8000 MiB and
12000 MiB matrix.
