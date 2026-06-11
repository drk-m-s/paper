# MoE Offload Performance Phase A Progress (2026-06-11)

Repository under work: `C:\code\llama.cpp.offload`.

This records the Phase A implementation from
`implementation_plan_mvp_perf_20260611.md`. No Phase B behavior changes were
made: EAMC still updates and saves exactly as before, now with timing around
that path.

Phase B direction was clarified after Phase A: EAMC should remain an online
predictor during normal inference, with updates kept in DRAM. The sidecar
should be saved only at logical user request end or session/context end, not
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
  - `s.pred->save()`,
  - `prof->flush()`,
  - sidecar file size after save.
- Kept the existing EAMC save/update behavior unchanged.

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
cmake --build build-moe --config Release --target llama-moe-bench llama-bench
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

The following Phase B+ items were intentionally not implemented:

- Online EAMC update with deferred sidecar save.
- Removing per-token EAMC sidecar saves.
- EAMC flush at logical user request end or session/context end.
- EAMC scoring optimization.
- H2D buffer lifetime changes.
- CUDA `.slot` `MUL_MAT_ID` fast-path restoration.
- Any change to `llama-cli` default ubatch or chat correctness policy.

## Next Validation Run

Run a short profiling job with EAMC and inspect the new `request` rows:

```powershell
.\build-moe\bin\Release\llama-moe-bench.exe `
  --model C:/AI/models/qwen/Qwen3.5-35B-A3B-Q4_K_M.moe.gguf `
  --pp 256 --tg 16 --repeat 1 `
  --moe-cache-vram-mb 8000 `
  --moe-predictor eamc `
  --moe-eamc-path C:/AI/models/qwen/Qwen3.5-35B-A3B-Q4_K_M.eamc `
  --moe-profile-csv tests\moe-offload\_out\phase-a-eamc.csv `
  --moe-profile-summary tests\moe-offload\_out\phase-a-eamc.summary.txt `
  -ngl 99 -c 4096 -ub 512
```

Expected diagnostic signal:

- `request` rows should show nonzero `predictor_end_us` and
  `predictor_save_us` when using EAMC with a sidecar.
- Summary `unattributed_decode_us` should shrink relative to the old profile
  if the hidden cost is mostly predictor finalization/save.
