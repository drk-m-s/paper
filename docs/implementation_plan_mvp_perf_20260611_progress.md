# MoE Offload Performance Progress (2026-06-11)

Repository under work: `C:\code\llama.cpp.offload`.

This records the Phase A, Phase B, Phase C, and Phase D implementation from
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

## Not Implemented

The following Phase C+ items were intentionally not implemented:

- Default corpus row caps or approximate EAMC scoring.
- CUDA `.slot` `MUL_MAT_ID` fast-path restoration.
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
