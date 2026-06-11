# MoE Offload Performance Progress (2026-06-11)

Repository under work: `C:\code\llama.cpp.offload`.

This records the Phase A and Phase B implementation from
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
row_type,request_idx,repeat_idx,batch_idx,token_idx,phase,layer,k_required,k_hit,k_miss,ssd_read_us,h2d_us,compute_us,stall_us,pred_us,pred_observe_us,pred_score_us,callback_wall_us,topk_d2h_us,slot_ids_h2d_us,slot_table_h2d_us,cache_resident_experts,predictor,request_wall_us,request_end_us,predictor_end_us,predictor_save_us,profile_flush_us,sidecar_write_bytes
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

### EAMC Predictor

Changed `src/moe-offload/predictor.cpp`:

- Replaced full-capacity `evict_redundant()` with FIFO/ring replacement.
- Preserved the existing dense EAMC sidecar format.
- Preserved online EAMC row append semantics.

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

MoE CTest label:

```powershell
ctest --test-dir build-moe -C Release -L moe-offload --output-on-failure
```

Result: 6/6 passed.

Passing tests:

- `test-cuda-stream`
- `test-eamc-cosine`
- `test-lru-eviction`
- `test-manifest-roundtrip`
- `test-repack-slices`
- `test-moe-oracle-failfast`

## Not Implemented

The following Phase C+ items were intentionally not implemented:

- EAMC scoring optimization.
- H2D buffer lifetime changes.
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
