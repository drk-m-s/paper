# MoE Offload Performance Progress (2026-06-11)

Repository under work: `C:\code\llama.cpp.offload`.

This records the Phase A, Phase B, Phase C, Phase D, guarded Phase E, guarded
Phase F, and guarded Phase G implementation from
`implementation_plan_mvp_perf_20260611.md`.

Phase B keeps EAMC online during normal inference, with updates kept in DRAM.
The sidecar is saved only at benchmark end or context/session teardown, not
after every internal `llama_decode()` batch.

## Scope Completed

Implemented Phase A profiler and benchmark calibration infrastructure:

- request-level timing rows,
- expanded per-layer CSV fields,
- decode wall/profile reconciliation in the summary,
- split predictor observe vs eviction score timing,
- callback wall timing,
- top-k D2H timing,
- slot-id and slot-table H2D host-call timing,
- predictor finalization timing,
- EAMC sidecar save timing and sidecar byte estimate,
- profile flush timing,
- request/repeat/batch identifiers,
- `llama-moe-bench` calibration flags for cold/warm-cache measurements.

Implemented Phase B EAMC lifecycle and persistence changes:

- per-`llama_decode()` EAMC current-row reset,
- online in-memory EAMC row append,
- FIFO/ring full-capacity replacement instead of quadratic redundancy pruning,
- explicit `llama_moe::flush_predictor()` API,
- one EAMC flush at `llama-moe-bench` end,
- context/session teardown EAMC flush,
- no EAMC sidecar save from per-token `slot_pool_end_request()`.

Implemented Phase C EAMC scoring optimization:

- sparse in-memory EAMC rows,
- dense sidecar load/save compatibility,
- indexed sparse cosine scoring over the uncapped corpus,
- lazy per-callback `score_for_expert` materialization,
- EAMC score-profile counters,
- diagnostic-only `LLAMA_MOE_EAMC_ROWS=N` row cap.

Implemented Phase D H2D buffer lifetime changes:

- nonblocking CUDA event query API,
- async pinned-buffer lifetime tracking for H2D staging buffers,
- preserved `cudaStreamWaitEvent()` compute-stream ordering,
- request-end/reset/shutdown drains for outstanding H2D buffers before profile
  events are released,
- miss-blob submission sorted by file offset,
- larger CUDA event-pool retention cap to avoid per-token event churn.

Implemented the guarded Phase E `.slot` MMVQ decode fast path:

- quantized single-token `.slot` `MUL_MAT_ID` can use CUDA MMVQ when
  `LLAMA_MOE_SLOT_MMVQ=1`,
- default behavior remains the generic sorted CUDA path,
- `.slot` fusion is disabled only under the guard so the benchmark isolates
  unfused MMVQ,
- CUDA graph capture remains disabled for `.slot` `MUL_MAT_ID`,
- MMQ/MMF and multi-token prefill fast paths remain separate validation steps.

## Code Changes

### Profiler Schema

Changed `src/moe-offload/profiler.{h,cpp}`:

- Added `profile_request_row`.
- Added request aggregation fields to `profile_phase_stats`.
- Added new per-layer fields:
  - `request_idx`
  - `repeat_idx`
  - `batch_idx`
  - `pred_observe_us`
  - `pred_score_us`
  - `callback_wall_us`
  - `topk_d2h_us`
  - `slot_ids_h2d_us`
  - `slot_table_h2d_us`
  - `eamc_rows_scored`
  - `eamc_cosine_us`
  - `eamc_score_materialize_us`
  - `eamc_score_cache_hits`
  - `eamc_score_cache_misses`
- Added request-only fields:
  - `request_wall_us`
  - `request_end_us`
  - `predictor_end_us`
  - `predictor_save_us`
  - `profile_flush_us`
  - `sidecar_write_bytes`
- CSV now has a `row_type` column:
  - `layer` for per-layer callback rows,
  - `request` for one row per `llama_decode()` request.
- Added `profiler::reset()` for benchmark warmup handling.

The new CSV header is:

```text
row_type,request_idx,repeat_idx,batch_idx,token_idx,phase,layer,k_required,k_hit,k_miss,ssd_read_us,h2d_us,compute_us,stall_us,pred_us,pred_observe_us,pred_score_us,callback_wall_us,topk_d2h_us,slot_ids_h2d_us,slot_table_h2d_us,eamc_rows_scored,eamc_cosine_us,eamc_score_materialize_us,eamc_score_cache_hits,eamc_score_cache_misses,cache_resident_experts,predictor,request_wall_us,request_end_us,predictor_end_us,predictor_save_us,profile_flush_us,sidecar_write_bytes
```

### Runtime Request Context

Changed `src/moe-offload/runtime.{h,cpp}`:

- Added `set_profile_request_context(repeat_idx, batch_idx, phase)`.
- Added active request tracking around `llama_moe::begin_request()` /
  `llama_moe::end_request()`.
- Added `current_profile_request_row()` so slot-pool layer rows inherit the
  request/repeat/batch labels.
- Added `add_current_request_timing()` so slot-pool can report predictor
  finalization, save, and profile-flush timing.
- Added `reset_profile()` for benchmark warm-cache runs.
- Added `flush_predictor()` for explicit predictor persistence at logical
  request/session boundaries.
- Calls the slot-pool predictor begin hook from `llama_moe::begin_request()`
  so EAMC's current row is reset for each internal decode batch.

### Slot-Pool Timing

Changed `src/moe-offload/slot_pool.{h,cpp}`:

- Added `slot_pool_reset_cache()` for benchmark calibration only.
- Added top-k D2H timing around `ggml_backend_tensor_get()`.
- Split predictor timing:
  - `pred_observe_us` around `s.pred->observe()`,
  - `pred_score_us` around eviction candidate scoring.
- Added host-call timing for slot-id and slot-table H2D writes.
- Added `callback_wall_us` around the eval-callback body.
- Added request-end timing for:
  - `s.pred->end_request()`,
  - `prof->flush()`.
- Removed `s.pred->save()` from `slot_pool_end_request()`.
- Added `slot_pool_begin_request()` to reset predictor observations before
  each internal decode batch.
- Added `slot_pool_flush_predictor()` to save dirty EAMC state only at
  explicit flush/session end.

### Phase D H2D Lifetime

Changed `src/moe-offload/slot_pool.cpp`:

- Added `inflight_h2d` records for pinned staging buffers whose async H2D copy
  has been submitted but whose end event has not completed.
- Kept `io_compute_wait(s.compute_backend, h2d_event)` on every async H2D
  completion so the CUDA compute stream still waits before consuming the slot.
- Removed the normal per-completion `io_event_sync(h2d_event)` host block.
- Recycled pinned buffers through `io_event_query(h2d_event)` polling at
  callback start, after draining worker completions, and while waiting for free
  pinned buffers.
- Added request-end, reset, reconfigure, cache-reset, and shutdown drains so
  no profile row releases a borrowed H2D event before its pinned buffer is safe.
- Kept a blocking fallback if compute-stream wait insertion fails.
- Sorted miss blobs by file offset before worker submission.

Changed `src/moe-offload/io.{h,cpp}` and
`ggml/src/ggml-cuda/moe_offload_io.cu`:

- Added `io_event_query()` / `moe_io_cuda_event_query()`.
- Raised the CUDA event pool retention cap from 128 to 4096 events.

### Phase E Slot MMVQ

Changed `ggml/src/ggml-cuda/ggml-cuda.cu`:

- Added `.slot` tensor detection helper for MoE slot-backed tensors.
- Added `LLAMA_MOE_SLOT_MMVQ` guard parsing.
- Re-enabled only the quantized single-token `.slot` MMVQ branch for
  `GGML_OP_MUL_MAT_ID` when `LLAMA_MOE_SLOT_MMVQ=1`.
- Kept `.slot` MMQ/MMF, multi-token prefill, CUDA graph capture, and fusion
  out of this step.
- Disabled `.slot` fusion decisions only while the MMVQ guard is enabled, so
  default behavior remains unchanged.

Added `tests/moe-offload/test-slot-mmvq.cpp` and registered it with CTest when
the CUDA backend target is available:

- synthetic Q4_0 `.slot` `MUL_MAT_ID` graph,
- decode-shaped generic sorted path check with the guard off,
- decode-shaped guarded MMVQ check with the guard on,
- prefill-shaped guarded fallback check.

### EAMC Predictor

Changed `src/moe-offload/predictor.cpp`:

- Replaced full-capacity `evict_redundant()` with FIFO/ring replacement.
- Preserved the existing dense EAMC sidecar format.
- Preserved online EAMC row append semantics.
- Converted EAMC corpus rows from dense vectors to sparse per-layer
  expert/count vectors.
- Precomputed per-row prefix norms for dense-equivalent cosine scoring.
- Added an inverted `(layer, expert) -> corpus row` index for uncapped sparse
  cosine scoring.
- Materialized one per-layer score vector on first eviction score request in a
  callback, then served later candidate scores in O(1).
- Kept `LLAMA_MOE_EAMC_ROWS=N` as a diagnostic-only row cap; default behavior
  remains uncapped.

### Benchmark Calibration

Changed `tools/moe-bench/main.cpp`:

- Added `--moe-reset-cache-between-repeats`.
  - Clears only MoE slot residency and counters between repeats.
  - Does not reload the model or change slot tensors.
- Added `--moe-warm-cache`.
  - Runs one unmeasured prefill before measured repeats.
  - Resets profiler output after warmup so CSV/summary contain measured rows
    only.
- Sets request context labels before each benchmark `llama_decode()`:
  - prefill request: `repeat_idx=rep`, `batch_idx=0`, `phase=prefill`,
  - decode request: `repeat_idx=rep`, `batch_idx=gen+1`, `phase=decode`.
- Summary now prints calibration flags when either is active.
- Calls `llama_moe::flush_predictor()` once after all measured repeats and
  before final summary output.

### Context Teardown

Changed `src/llama-context.cpp`:

- Calls `llama_moe::flush_predictor()` before shutting down MoE I/O during
  context destruction.

## Summary Output Additions

The MoE summary now includes:

```text
Profiler breakdown (decode, mean per token):
  predictor_observe
  predictor_score
  callback_wall
  topk_d2h
  slot_ids_h2d
  slot_table_h2d
  eamc_cosine
  eamc_materialize
  eamc_rows_scored
  eamc_cache_hits
  eamc_cache_misses
  request_end
  predictor_end
  predictor_save
  profile_flush
  sidecar_written

Wall/profile reconciliation (decode):
  wall_decode_us
  profiled_decode_us
  unattributed_decode_us
```

This is intended to confirm whether the current seconds-per-token decode gap is
inside EAMC finalization/save, profile flush, or still outside the MoE profiler.

## Validation

Focused build:

```powershell
cmake --build build-moe --config Release --target llama-moe-bench llama-bench test-eamc-cosine
```

Result: passed.

Phase D focused build:

```powershell
cmake --build build-moe --config Release --target llama-moe-bench llama-bench llama-completion
```

Result: passed.

Phase E focused builds:

```powershell
cmake --build build-moe --config Release --target test-slot-mmvq -j 8
cmake --build build-moe --config Release --target llama-moe-bench -j 8
```

Result: passed. Building `llama-completion` also passed; the local CMake
invocation for `llama-cli` looked for a root-level `llama-cli.vcxproj`, but the
existing `build-moe\bin\Release\llama-cli.exe` was present and used for the chat
smoke.

MoE CTest label:

```powershell
ctest --test-dir build-moe -C Release -L moe-offload --output-on-failure
```

Phase A-C result: 6/6 passed.

Phase D result on 2026-06-12: 5/6 passed. `test-cuda-stream` did not start
because Windows Application Control blocked the freshly rebuilt unsigned
`test-cuda-stream.exe`; the other MoE tests passed:

Passing tests:

- `test-eamc-cosine`
- `test-lru-eviction`
- `test-manifest-roundtrip`
- `test-repack-slices`
- `test-moe-oracle-failfast`

Additional Phase D environment note:

- Rebuilding `ggml-cuda.dll` with the default compressed CUDA fatbin was also
  blocked by Windows Application Control (`0xc0e90002`) on this dev box.
- Reconfiguring with `-DGGML_CUDA_COMPRESSION_MODE=none` produced a loadable
  CUDA backend and allowed `llama-moe-bench` to run.

Phase E focused CTest:

```powershell
.\build-moe\bin\Release\test-slot-mmvq.exe
ctest --test-dir build-moe -C Release -R test-slot-mmvq --output-on-failure
```

Result: passed. The standalone run reported:

- decode generic sorted: `max_abs=0.05145073`, `rms=0.01440378`,
- decode guarded MMVQ: `max_abs=0.05145073`, `rms=0.01440378`,
- prefill guard fallback: `max_abs=0.05145073`, `rms=0.01363423`.

Phase E guarded correctness gates:

```powershell
$env:LLAMA_MOE_SLOT_MMVQ='1'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tests\moe-offload\test-golden-logits.ps1 `
  -StreamCacheMb 8000 -NPredict 32 -UBatch 8

$env:LLAMA_MOE_SLOT_MMVQ='1'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tests\moe-offload\test-llama-cli-chat.ps1 `
  -CacheMb 8000 -Predictor eamc
```

Result: both passed. Golden logits reported `max|d| = 0`.

## Not Implemented

The following Phase C+ items were intentionally not implemented:

- Default corpus row caps or approximate EAMC scoring.
- CUDA `.slot` MMQ/MMF, multi-token prefill fast paths, CUDA graph capture,
  and top-k/fusion restoration.
- Any change to `llama-cli` default ubatch or chat correctness policy.

## Phase B Performance Validation

Old baseline files:

- `C:\code\llama.cpp.offload\moe-summary.txt`
- `C:\code\llama.cpp.offload\moe-profile.csv`

New Phase B files:

- `tests\moe-offload\_out\phase-b-eamc-8gb.csv`
- `tests\moe-offload\_out\phase-b-eamc-8gb.summary.txt`
- `tests\moe-offload\_out\phase-b-eamc-8gb.eamc`

The benchmark used a copy of the original sidecar so the run did not mutate
`C:\AI\models\qwen\Qwen3.5-35B-A3B-Q4_K_M.eamc`.

```powershell
.\build-moe\bin\Release\llama-moe-bench.exe `
  --model C:/AI/models/qwen/Qwen3.5-35B-A3B-Q4_K_M.moe.gguf `
  --pp 256 --tg 256 --repeat 3 `
  --moe-cache-vram-mb 8000 `
  --moe-predictor eamc `
  --moe-eamc-path tests/moe-offload/_out/phase-b-eamc-8gb.eamc `
  --moe-profile-csv tests\moe-offload\_out\phase-b-eamc-8gb.csv `
  --moe-profile-summary tests\moe-offload\_out\phase-b-eamc-8gb.summary.txt `
  -ngl 99 -c 4096 -ub 512
```

Result:

| Metric | Old MVP | Phase B | Change |
| --- | ---: | ---: | ---: |
| TTFT / prefill | 33646.2 ms | 25825.6 ms | 1.30x faster |
| TPOT / decode | 7184.84 ms/token | 162.66 ms/token | 44.17x faster |
| Total | 1872964.8 ms | 67466.9 ms | 27.76x faster |
| Prefill hit rate | 80.1% | 81.9% | +1.8 pp |
| Decode hit rate | 88.8% | 90.3% | +1.5 pp |
| Decode SSD read | 17.53 ms/token | 18.07 ms/token | about flat |
| Decode H2D | 43.59 ms/token | 9.02 ms/token | 4.83x faster |
| Decode compute | 122.98 ms/token | 42.26 ms/token | 2.91x faster |
| Decode stall | 54.22 ms/token | 16.91 ms/token | 3.21x faster |
| Decode predictor | 133.61 ms/token | 90.54 ms/token | 1.48x faster |

Phase B hot-path persistence checks from the new CSV:

- decode request rows: 768
- rows with nonzero `predictor_save_us`: 0
- rows with nonzero `sidecar_write_bytes`: 0
- total decode `predictor_end_us`: 1784 us
- explicit benchmark-end sidecar save: one write of 41943076 bytes

Remaining bottleneck after Phase B:

- EAMC scoring still costs about 90.54 ms/token in decode.
- That makes Phase C, EAMC scoring optimization, the next highest-value
  predictor-side task.

## Phase C Performance Validation

New Phase C files:

- `tests\moe-offload\_out\phase-c-eamc-8gb.csv`
- `tests\moe-offload\_out\phase-c-eamc-8gb.summary.txt`
- `tests\moe-offload\_out\phase-c-eamc-8gb.eamc`

The benchmark used a copy of the model sidecar so the run did not mutate
`C:\AI\models\qwen\Qwen3.5-35B-A3B-Q4_K_M.eamc`.

```powershell
.\build-moe\bin\Release\llama-moe-bench.exe `
  --model C:/AI/models/qwen/Qwen3.5-35B-A3B-Q4_K_M.moe.gguf `
  --pp 256 --tg 256 --repeat 3 `
  --moe-cache-vram-mb 8000 `
  --moe-predictor eamc `
  --moe-eamc-path tests/moe-offload/_out/phase-c-eamc-8gb.eamc `
  --moe-profile-csv tests\moe-offload\_out\phase-c-eamc-8gb.csv `
  --moe-profile-summary tests\moe-offload\_out\phase-c-eamc-8gb.summary.txt `
  -ngl 99 -c 4096 -ub 512
```

The benchmark wrote complete CSV/summary artifacts but returned exit code 1
because `llama-moe-bench` logged the existing `get_logits_ith: invalid logits
id 0` warning during repeated summary writes.

Result:

| Metric | Phase B | Phase C | Change |
| --- | ---: | ---: | ---: |
| TTFT / prefill | 25825.6 ms | 21496.1 ms | 1.20x faster |
| TPOT / decode | 162.66 ms/token | 57.78 ms/token | 2.82x faster |
| Total | 67466.9 ms | 36287.1 ms | 1.86x faster |
| Prefill hit rate | 81.9% | 81.9% | about flat |
| Decode hit rate | 90.3% | 89.6% | -0.7 pp |
| Decode SSD read | 18.07 ms/token | 10.95 ms/token | 1.65x faster |
| Decode H2D | 9.02 ms/token | 8.99 ms/token | about flat |
| Decode compute | 42.26 ms/token | 33.92 ms/token | 1.25x faster |
| Decode stall | 16.91 ms/token | 16.77 ms/token | about flat |
| Decode predictor | 90.54 ms/token | 2.52 ms/token | 35.9x faster |

Phase C EAMC counters from the CSV:

- decode `eamc_cosine_us`: 1757640 us total, 2.29 ms/token,
- decode `eamc_score_materialize_us`: 12145 us total, 0.02 ms/token,
- decode `eamc_rows_scored`: 15864832 rows total, 20657.3 rows/token,
- decode `eamc_score_cache_hits`: 2954.50 hits/token,
- decode `eamc_score_cache_misses`: 20.17 misses/token.

Hit-rate note:

- Dense-equivalence unit tests pass for partial prefixes, repeated
  observations, prefill-like repeated layer waves, FIFO replacement, and dense
  sidecar round-trip.
- The recorded Phase B and Phase C benchmark sidecars did not provide a clean
  same-start-sidecar comparison. The final saved Phase B and Phase C sidecars
  have the same hash, but the model sidecar copied at Phase C start has a
  different hash from the saved Phase B artifact. Treat the measured -0.7 pp
  decode hit-rate delta as needing a clean same-start-sidecar rerun before
  changing EAMC defaults.

Phase C acceptance status:

- Predictor-time target: passed. Decode predictor time is 2.52 ms/token, below
  the 10 ms/token target.
- Prefill hit-rate gate: passed against the recorded Phase B profile.
- Decode hit-rate gate: inconclusive against the recorded Phase B profile
  because the sidecar baseline was not identical; the measured delta is
  slightly outside the 0.5 pp gate.
- Row caps remain diagnostic-only and are not enabled by default.

## Phase D Performance Validation

New Phase D files:

- `tests\moe-offload\_out\phase-d-eamc-8gb.csv`
- `tests\moe-offload\_out\phase-d-eamc-8gb.summary.txt`
- `tests\moe-offload\_out\phase-d-eamc-8gb.eamc`

The benchmark used a copy of the model sidecar so the run did not mutate
`C:\AI\models\qwen\Qwen3.5-35B-A3B-Q4_K_M.eamc`.

The command was run from `build-moe\bin\Release` after configuring
`build-moe` with `-DGGML_CUDA_COMPRESSION_MODE=none`:

```powershell
.\llama-moe-bench.exe `
  --model C:/AI/models/qwen/Qwen3.5-35B-A3B-Q4_K_M.moe.gguf `
  --pp 256 --tg 256 --repeat 3 `
  --moe-cache-vram-mb 8000 `
  --moe-predictor eamc `
  --moe-eamc-path C:/code/llama.cpp.offload/tests/moe-offload/_out/phase-d-eamc-8gb.eamc `
  --moe-profile-csv C:/code/llama.cpp.offload/tests/moe-offload/_out/phase-d-eamc-8gb.csv `
  --moe-profile-summary C:/code/llama.cpp.offload/tests/moe-offload/_out/phase-d-eamc-8gb.summary.txt `
  -ngl 99 -c 4096 -ub 512
```

Result versus the most recent pre-Phase-D `moe-summary.txt`:

| Metric | Pre-Phase-D latest | Phase D | Change |
| --- | ---: | ---: | ---: |
| TTFT / prefill | 16741.3 ms | 19704.8 ms | 0.85x as fast |
| TPOT / decode | 57.92 ms/token | 62.13 ms/token | 0.93x as fast |
| Total | 31568.9 ms | 35609.0 ms | 0.89x as fast |
| Prefill hit rate | 80.6% | 81.9% | +1.3 pp |
| Decode hit rate | 89.4% | 89.2% | -0.2 pp |
| Decode SSD read | 13.51 ms/token | 15.57 ms/token | 15.2% slower |
| Decode H2D | 9.20 ms/token | 9.42 ms/token | 2.4% slower |
| Decode compute | 35.17 ms/token | 34.86 ms/token | about flat |
| Decode stall | 16.84 ms/token | 0.41 ms/token | 41.1x lower |
| Decode predictor | 0.87 ms/token | 2.10 ms/token | 2.4x slower |
| Decode callback wall | 22.42 ms/token | 24.33 ms/token | 8.5% slower |

Result versus the recorded Phase C artifact:

| Metric | Phase C | Phase D | Change |
| --- | ---: | ---: | ---: |
| TPOT / decode | 57.78 ms/token | 62.13 ms/token | 0.93x as fast |
| Prefill hit rate | 81.9% | 81.9% | flat |
| Decode hit rate | 89.6% | 89.2% | -0.4 pp |
| Decode stall | 16.77 ms/token | 0.41 ms/token | 40.9x lower |
| Decode SSD read | 10.95 ms/token | 15.57 ms/token | 42.2% slower |
| Decode callback wall | 21.28 ms/token | 24.33 ms/token | 14.3% slower |

Phase D acceptance status:

- `stall_us` reduction: passed. The per-completion host event sync was removed
  from the normal path, and decode stall fell from about 16.8 ms/token to
  0.41 ms/token.
- Hit-rate preservation: passed against the latest root summary and within the
  Phase C 0.5 pp gate for decode.
- End-to-end speed: not passed on this run. TPOT regressed from 57.92 to
  62.13 ms/token versus the latest root summary, and from 57.78 to
  62.13 ms/token versus Phase C.
- New bottleneck: the removed stall did not translate into wall-clock speedup
  because worker `ssd_read_us`, callback wall, and EAMC predictor time were
  higher in the Phase D run. The wall/profile reconciliation improved sharply
  (`unattributed_decode_us` moved from about -13.9 s to -0.6 s), which confirms
  the profiler is now less dominated by overlapping buckets, but the next
  optimization should focus on worker read latency/queueing and callback wall
  rather than more host event-sync removal.

## Phase E Performance Validation

New Phase E files:

- `tests\moe-offload\_out\phase-e-slot-mmvq-8gb.csv`
- `tests\moe-offload\_out\phase-e-slot-mmvq-8gb.summary.txt`
- `tests\moe-offload\_out\phase-e-slot-mmvq-8gb.eamc`

The benchmark used a copy of the Phase D sidecar and enabled only the guarded
MMVQ decode path:

```powershell
$env:LLAMA_MOE_SLOT_MMVQ='1'
.\build-moe\bin\Release\llama-moe-bench.exe `
  --model C:/AI/models/qwen/Qwen3.5-35B-A3B-Q4_K_M.moe.gguf `
  --pp 256 --tg 256 --repeat 3 `
  --moe-cache-vram-mb 8000 `
  --moe-predictor eamc `
  --moe-eamc-path tests/moe-offload/_out/phase-e-slot-mmvq-8gb.eamc `
  --moe-profile-csv tests\moe-offload\_out\phase-e-slot-mmvq-8gb.csv `
  --moe-profile-summary tests\moe-offload\_out\phase-e-slot-mmvq-8gb.summary.txt `
  -ngl 99 -c 4096 -ub 512
```

Result versus Phase D:

| Metric | Phase D | Phase E | Change |
| --- | ---: | ---: | ---: |
| TTFT / prefill | 19704.8 ms | 18481.0 ms | 1.07x faster |
| TPOT / decode | 62.13 ms/token | 47.20 ms/token | 1.32x faster |
| Total | 35609.0 ms | 30564.0 ms | 1.16x faster |
| Prefill hit rate | 81.9% | 81.9% | flat |
| Decode hit rate | 89.2% | 89.2% | flat |
| Decode SSD read | 15.57 ms/token | 13.08 ms/token | 1.19x faster |
| Decode H2D | 9.42 ms/token | 9.37 ms/token | about flat |
| Decode compute | 34.86 ms/token | 24.34 ms/token | 1.43x faster |
| Decode stall | 0.41 ms/token | 0.12 ms/token | 3.4x lower |
| Decode predictor | 2.10 ms/token | 1.32 ms/token | 1.59x faster |
| Decode callback wall | 24.33 ms/token | 19.80 ms/token | 1.23x faster |

Phase E acceptance status:

- Golden-logit gate with `LLAMA_MOE_SLOT_MMVQ=1`: passed, `max|d| = 0`.
- Chat smoke with `LLAMA_MOE_SLOT_MMVQ=1`: passed with default `llama-cli`
  ubatch 1.
- End-to-end speed: passed. TTFT improved from 19704.8 ms to 18481.0 ms and
  TPOT improved from 62.13 ms/token to 47.20 ms/token versus Phase D.
- Compute gate: passed. Decode `compute_us` fell from 34.86 ms/token to
  24.34 ms/token.
- Secondary metrics gate: passed on this run. H2D, stall, predictor time, SSD
  read time, callback wall time, hit rate, and misses/token were flat or better,
  so the wall-clock gain was not offset by hidden regressions.
- Remaining Phase E work: MMQ/MMF, multi-token prefill fast paths, CUDA graph
  capture, and fusion remain disabled for `.slot` tensors until separately
  validated.

## Phase F Implementation

Implemented guarded CUDA graph capture for the Phase E `.slot` MMVQ decode
path:

- added `LLAMA_MOE_SLOT_GRAPHS=1` as a second guard, independent from
  `LLAMA_MOE_SLOT_MMVQ=1`,
- added a `.slot` `MUL_MAT_ID` graph compatibility predicate that requires:
  - `LLAMA_MOE_SLOT_MMVQ=1`,
  - `LLAMA_MOE_SLOT_GRAPHS=1`,
  - quantized `.slot` `src0`,
  - F32 activation and F32 output tensors,
  - I32 slot-id tensor,
  - single-token decode shape,
  - MMVQ-supported batch size,
- left default `.slot` behavior unchanged,
- kept generic sorted, MMQ/MMF, and multi-token prefill shapes graph-disabled.

Scheduler/callback ordering was checked before enabling the guard. The ggml
scheduler computes the graph view ending at the observed top-k tensor,
synchronizes the split backend, runs `moe_eval_callback()`, and only then
computes the next graph view. The callback writes the slot-id tensor and inserts
compute-stream waits for async expert H2D completions before the `.slot`
`MUL_MAT_ID` consumer graph view is launched. CUDA graph replay therefore reads
updated device tensor contents from stable tensor addresses rather than captured
slot-id values.

Changed `tests/moe-offload/test-slot-mmvq.cpp`:

- kept the generic sorted decode case,
- kept the guarded MMVQ decode case,
- added a guarded MMVQ graph replay case that runs the same decode graph five
  times with changing slot ids, changing slot tensor contents, and changing
  activations,
- kept the guarded prefill fallback case with graph guard enabled, proving the
  prefill shape still declines graph capture and uses the safe path.

## Phase F Validation

Focused build and test:

```powershell
cmake --build build-moe --config Release --target test-slot-mmvq -j 8
```

The shared `build-moe` CUDA DLL was then blocked by Windows Smart App Control
after the rebuild:

```text
0xc0e90002
Code Integrity ... test-slot-mmvq.exe attempted to load ... ggml-cuda.dll
that did not meet the Enterprise signing level requirements
```

To continue validation without changing machine policy, a separate static CUDA
validation tree was configured and used:

```powershell
cmake -S . -B build-moe-static -G "Visual Studio 18 2026" `
  -DLLAMA_MOE_OFFLOAD=ON `
  -DGGML_CUDA=ON `
  -DGGML_CUDA_GRAPHS=ON `
  -DGGML_CUDA_COMPRESSION_MODE=none `
  -DBUILD_SHARED_LIBS=OFF `
  -DGGML_STATIC=ON `
  -DLLAMA_BUILD_TESTS=ON `
  -DLLAMA_BUILD_TOOLS=ON `
  -DLLAMA_BUILD_EXAMPLES=OFF `
  -DLLAMA_BUILD_APP=OFF `
  -DLLAMA_BUILD_SERVER=ON `
  -DLLAMA_BUILD_UI=OFF `
  -DLLAMA_CURL=OFF `
  -DLLAMA_OPENSSL=OFF

cmake --build build-moe-static --config Release --target test-slot-mmvq -j 8
.\build-moe-static\bin\Release\test-slot-mmvq.exe
ctest --test-dir build-moe-static -C Release -R test-slot-mmvq --output-on-failure
```

Standalone test output:

```text
[test-slot-mmvq] decode generic sorted OK max_abs=0.05145073 rms=0.01440378
[test-slot-mmvq] decode guarded MMVQ OK max_abs=0.05145073 rms=0.01440378
[test-slot-mmvq] decode guarded MMVQ graphs iter=0 OK max_abs=0.05145073 rms=0.01440378
[test-slot-mmvq] decode guarded MMVQ graphs iter=1 OK max_abs=0.04878712 rms=0.01495539
[test-slot-mmvq] decode guarded MMVQ graphs iter=2 OK max_abs=0.05170250 rms=0.01552438
[test-slot-mmvq] decode guarded MMVQ graphs iter=3 OK max_abs=0.03503227 rms=0.01436052
[test-slot-mmvq] decode guarded MMVQ graphs iter=4 OK max_abs=0.03183556 rms=0.01001032
[test-slot-mmvq] prefill guard fallback OK max_abs=0.05145073 rms=0.01363423
ggml_backend_cuda_graph_compute: CUDA graph warmup complete
```

Static CTest passed:

```text
1/1 Test #53: test-slot-mmvq ...................   Passed
```

Model-level gates were run with static `llama-completion` and `llama-cli`
binaries because the shared `build-moe` CUDA DLL remained blocked by Smart App
Control:

```powershell
$env:LLAMA_MOE_SLOT_MMVQ='1'
$env:LLAMA_MOE_SLOT_GRAPHS='1'

powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\tests\moe-offload\test-golden-logits.ps1 `
  -Bin .\build-moe-static\bin\Release\llama-completion.exe `
  -StreamCacheMb 8000 -NPredict 32 -UBatch 8

powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\tests\moe-offload\test-llama-cli-chat.ps1 `
  -Bin .\build-moe-static\bin\Release\llama-cli.exe `
  -CacheMb 8000 -Predictor eamc
```

Results:

- Golden logits: passed, `max|d| = 0`.
- Chat smoke: passed.

## Phase F Performance Validation

New Phase F files:

- `tests\moe-offload\_out\phase-f-slot-graphs-8gb.csv`
- `tests\moe-offload\_out\phase-f-slot-graphs-8gb.summary.txt`
- `tests\moe-offload\_out\phase-f-slot-graphs-8gb.eamc`
- `tests\moe-offload\_out\phase-f-slot-graphs-8gb.stdout.txt`

The benchmark used a copy of the Phase E sidecar and enabled both guarded
decode flags:

```powershell
Copy-Item -Force `
  tests\moe-offload\_out\phase-e-slot-mmvq-8gb.eamc `
  tests\moe-offload\_out\phase-f-slot-graphs-8gb.eamc

$env:LLAMA_MOE_SLOT_MMVQ='1'
$env:LLAMA_MOE_SLOT_GRAPHS='1'

.\build-moe-static\bin\Release\llama-moe-bench.exe `
  --model C:/AI/models/qwen/Qwen3.5-35B-A3B-Q4_K_M.moe.gguf `
  --pp 256 --tg 256 --repeat 3 `
  --moe-cache-vram-mb 8000 `
  --moe-predictor eamc `
  --moe-eamc-path tests/moe-offload/_out/phase-f-slot-graphs-8gb.eamc `
  --moe-profile-csv tests\moe-offload\_out\phase-f-slot-graphs-8gb.csv `
  --moe-profile-summary tests\moe-offload\_out\phase-f-slot-graphs-8gb.summary.txt `
  -ngl 99 -c 4096 -ub 512
```

Result versus Phase E:

| Metric | Phase E | Phase F | Change |
| --- | ---: | ---: | ---: |
| TTFT / prefill | 18481.0 ms | 16026.2 ms | 1.15x faster |
| TPOT / decode | 47.20 ms/token | 30.63 ms/token | 1.54x faster |
| Total | 30564.0 ms | 23866.8 ms | 1.28x faster |
| Prefill hit rate | 81.9% | 81.9% | flat |
| Decode hit rate | 89.2% | 89.2% | flat |
| Decode SSD read | 13.08 ms/token | 11.04 ms/token | 1.18x faster |
| Decode H2D | 9.37 ms/token | 9.34 ms/token | about flat |
| Decode compute | 24.34 ms/token | 10.46 ms/token | 2.33x faster |
| Decode stall | 0.12 ms/token | 0.13 ms/token | about flat |
| Decode predictor | 1.32 ms/token | 1.24 ms/token | 1.06x faster |
| Decode callback wall | 19.80 ms/token | 16.90 ms/token | 1.17x faster |

Phase F acceptance status:

- Golden-logit gate with `LLAMA_MOE_SLOT_MMVQ=1` and
  `LLAMA_MOE_SLOT_GRAPHS=1`: passed, `max|d| = 0`.
- Chat smoke with both guards enabled: passed with default `llama-cli` ubatch 1.
- End-to-end speed: passed. TPOT improved from 47.20 ms/token to
  30.63 ms/token versus Phase E.
- Compute gate: passed. Decode `compute_us` fell from 24.34 ms/token to
  10.46 ms/token.
- Secondary metrics gate: passed on this run. H2D, stall, predictor time, SSD
  read time, callback wall time, hit rate, and misses/token were flat or better,
  so the decode wall-clock gain was not erased by hidden regressions.
- Caveat: benchmark and model-level gates were run from the static validation
  build because Windows Smart App Control blocked the rebuilt shared
  `build-moe\bin\Release\ggml-cuda.dll`.

## Phase F Shared-Build Tool Repair

After Phase F, the normal shared `build-moe` tool paths failed immediately:

```text
LASTEXIT=-1058471934  # 0xc0e90002
Code Integrity ... llama-cli.exe attempted to load ... ggml-cuda.dll
that did not meet the Enterprise signing level requirements
```

This was the same Smart App Control block observed during Phase F validation,
not a MoE runtime/logit failure. To restore the commands developers already
run from `build-moe\bin\Release`, the static validation binaries were copied
over the tiny shared launchers:

- `build-moe\bin\Release\llama-moe-bench.exe`
- `build-moe\bin\Release\llama-cli.exe`
- `build-moe\bin\Release\llama-completion.exe`

The original shared launchers were backed up under:

```text
build-moe\bin\Release\shared-launcher-backup-phase-f-20260612-170112
```

Post-repair smoke results:

- `build-moe\bin\Release\llama-moe-bench.exe` with `--pp 8 --tg 2 --repeat 1`
  exited 0 and wrote nonempty profile/summary artifacts.
- `build-moe\bin\Release\llama-cli.exe` answered `what is 1 + 1?` with `2`
  and exited 0.

## Phase G Implementation

Phase G was implemented as a split result:

- accepted guarded decode-only `.slot` quantized `MUL_MAT_ID + GLU` fusion via
  `LLAMA_MOE_SLOT_GLU_FUSION=1`,
- kept top-k MoE fusion disabled in `LLAMA_MOE_OFFLOAD` builds after a
  revalidation failure,
- kept defaults correctness-first.

Code changes:

- `ggml/src/ggml-cuda/ggml-cuda.cu`
  - Added `LLAMA_MOE_SLOT_GLU_FUSION=1`.
  - Allows `.slot` GLU fusion only when all of these are true:
    `LLAMA_MOE_SLOT_MMVQ=1`, `LLAMA_MOE_SLOT_GLU_FUSION=1`, quantized
    `.slot` weights, F32 activations/output, I32 ids, and single-token decode.
  - Keeps `.slot` MMVF fusion disabled.
  - Keeps standalone `.slot` add/bias fusion disabled; only the GLU subgraph
    call sites pass the new allow flag.
  - Leaves `LLAMA_MOE_TOPK_FUSION=1` ignored in `LLAMA_MOE_OFFLOAD` builds
    after golden-logit revalidation failed.

- `tests/moe-offload/test-slot-mmvq.cpp`
  - Added a synthetic two-matrix `.slot` GLU graph:
    `MUL_MAT_ID(gate.slot)`, `MUL_MAT_ID(up.slot)`, `SWIGLU`.
  - Compares guarded GLU fusion output against the existing unfused CUDA
    baseline.
  - Replays the fused GLU graph while changing slot contents and slot ids.
  - Confirms multi-token/prefill-shaped GLU input still falls back cleanly.

## Phase G Validation

Focused build and CTest:

```powershell
cmake --build build-moe-static --config Release --target test-slot-mmvq llama-completion llama-cli llama-moe-bench -j 8
ctest --test-dir build-moe-static -C Release -R test-slot-mmvq --output-on-failure
```

Result:

```text
1/1 Test #53: test-slot-mmvq ...................   Passed
```

Golden logits and chat smoke used the static validation build because the
shared CUDA DLL path had previously been blocked by Windows Smart App Control:

```powershell
$env:LLAMA_MOE_SLOT_MMVQ='1'
$env:LLAMA_MOE_SLOT_GRAPHS='1'
$env:LLAMA_MOE_SLOT_GLU_FUSION='1'
$env:LLAMA_MOE_TOPK_FUSION='1' # ignored in LLAMA_MOE_OFFLOAD builds

powershell -NoProfile -ExecutionPolicy Bypass `
  -File tests\moe-offload\test-golden-logits.ps1 `
  -Bin .\build-moe-static\bin\Release\llama-completion.exe `
  -Model "C:\AI\models\qwen\Qwen3.5-35B-A3B-Q4_K_M.moe.gguf" `
  -Tol 1e-3 -NPredict 8 -StreamCacheMb 4000 `
  -Prompt "Hello" -Seed 42 -Context 4096 -UBatch 8

powershell -NoProfile -ExecutionPolicy Bypass `
  -File tests\moe-offload\test-llama-cli-chat.ps1 `
  -Bin .\build-moe-static\bin\Release\llama-cli.exe `
  -Model "C:\AI\models\qwen\Qwen3.5-35B-A3B-Q4_K_M.moe.gguf" `
  -CacheMb 8000 -Predictor lru -NPredict 64
```

Results:

- Golden logits: passed, `max|d| = 0`.
- Chat smoke: passed.

Experimental top-k MoE fusion was also re-enabled during Phase G before being
disabled again. With `LLAMA_MOE_TOPK_FUSION=1` actually active, the same golden
gate failed:

```text
n_steps  = 8
n_vocab  = 248320
max|d|   = 4.639292e-01
mean|d|  = 3.669774e-02
tol      = 1.000000e-03
FAIL: max|d| at step 3
```

That result keeps top-k MoE fusion open as a correctness issue rather than an
accepted Phase G optimization.

## Phase G Performance Validation

Benchmark artifacts:

- `tests\moe-offload\_out\phase-g-baseline-8gb.csv`
- `tests\moe-offload\_out\phase-g-baseline-8gb.summary.txt`
- `tests\moe-offload\_out\phase-g-baseline-8gb.eamc`
- `tests\moe-offload\_out\phase-g-baseline-8gb.stdout.txt`
- `tests\moe-offload\_out\phase-g-slot-glu-8gb.csv`
- `tests\moe-offload\_out\phase-g-slot-glu-8gb.summary.txt`
- `tests\moe-offload\_out\phase-g-slot-glu-8gb.eamc`
- `tests\moe-offload\_out\phase-g-slot-glu-8gb.stdout.txt`

Both runs used copies of the Phase F sidecar. Baseline enabled Phase F guards;
the Phase G run additionally enabled `LLAMA_MOE_SLOT_GLU_FUSION=1`.

```powershell
Copy-Item -Force `
  tests\moe-offload\_out\phase-f-slot-graphs-8gb.eamc `
  tests\moe-offload\_out\phase-g-baseline-8gb.eamc
Copy-Item -Force `
  tests\moe-offload\_out\phase-f-slot-graphs-8gb.eamc `
  tests\moe-offload\_out\phase-g-slot-glu-8gb.eamc
```

Result versus same-build Phase F guard baseline:

| Metric | Phase F guards | Phase G GLU | Change |
| --- | ---: | ---: | ---: |
| TTFT / prefill | 14719.8 ms | 14021.9 ms | 1.05x faster |
| TPOT / decode | 31.56 ms/token | 30.67 ms/token | 1.03x faster |
| Total | 22798.5 ms | 21873.7 ms | 1.04x faster |
| Prefill hit rate | 81.9% | 81.9% | flat |
| Decode hit rate | 89.2% | 89.2% | flat |
| Decode SSD read | 11.82 ms/token | 11.03 ms/token | 1.07x faster |
| Decode H2D | 9.35 ms/token | 9.34 ms/token | flat |
| Decode compute | 10.44 ms/token | 10.37 ms/token | flat |
| Decode stall | 0.12 ms/token | 0.14 ms/token | about flat |
| Decode predictor | 1.27 ms/token | 1.31 ms/token | about flat |
| Decode callback wall | 17.78 ms/token | 17.06 ms/token | 1.04x faster |
| Decode top-k D2H | 1.57 ms/token | 1.57 ms/token | flat |
| Decode misses/token | 20.78 | 20.78 | flat |

Phase G acceptance status:

- GLU golden-logit gate: passed, `max|d| = 0`.
- GLU chat smoke: passed with default `llama-cli` ubatch 1.
- GLU synthetic CUDA replay: passed with changing slot ids and contents.
- End-to-end speed: passed modestly. TPOT improved from 31.56 ms/token to
  30.67 ms/token in the same static build.
- Compute/callback gate: callback wall improved from 17.78 ms/token to
  17.06 ms/token; `gpu_compute` stayed effectively flat.
- Secondary metrics gate: passed. H2D, top-k D2H, hit rate, misses/token, and
  predictor time stayed within noise.
- Top-k fusion: failed golden logits and remains disabled/open.
- Caveat: both `llama-moe-bench` runs wrote complete summaries but returned a
  nonzero process code after repeated `get_logits_ith: invalid logits id 0`
  warnings. The summaries were still emitted successfully and are usable for
  timing comparison, but this bench exit-code issue remains separate cleanup.
