# MoE Offload Adaptive UBatch Progress (2026-06-09)

Repository under work: `C:\code\llama.cpp.offload`.

This records the implementation pass against
`implementation_plan_mvp_20260609.md`, including the follow-up small-cache
runtime fix from the 2026-06-09 smoke test.

## Summary

Implemented the runtime side of removing the fixed streaming `n_ubatch <= 8`
cap. Streaming mode now auto-sizes the effective ubatch from the configured MoE
slot budget and model top-k. Very small cache budgets reserve at least one
token's top-k working set per layer, so the default path shrinks throughput
instead of aborting when the budget is below the requested prefill shape.

The real small-cache failure reproduced on the dev model during warmup:
`n_uniq=256 exceeds n_slots=96`. The root cause was llama.cpp warmup expanding
MoE routing to all experts. Streaming MoE offload now keeps warmup routing at
the model top-k, because the all-expert warmup shape is incompatible with a
slot cache that is intentionally smaller than the expert count.

## Code Changes

### Adaptive UBatch Policy

Changed `src/llama-context.cpp`:

- Removed the unconditional `kMaxStreamingUbatch = 8` rewrite.
- Added streaming-only auto-sizing:
  - requested ubatch comes from normal context params,
  - slot budget comes from `llama_moe::n_slots_per_layer()`,
  - model top-k comes from `hparams.n_expert_used`,
  - recommendation is computed by `llama_moe::recommended_ubatch()`.
- Added diagnostic environment controls:
  - `LLAMA_MOE_STREAMING_UBATCH=N` forces a fixed effective streaming ubatch.
  - `LLAMA_MOE_STREAMING_UBATCH=off` disables auto-sizing.
  - `LLAMA_MOE_UBATCH_SAFETY=F` adjusts the slot-budget safety factor.
- Added startup logging for requested/effective ubatch, slot count, top-k, and
  safety factor.

Changed `src/moe-offload/slot_pool.{h,cpp}`:

- Added read-only accessors:
  - `n_slots_per_layer()`
  - `n_experts_per_layer()`
  - `streaming_mode()`
- Exported the accessors needed by `llama-moe-bench`.
- Added `recommended_ubatch(requested, n_expert_used, safety)`.
- Added a minimum viable slot reservation based on model top-k, with
  `LLAMA_MOE_MIN_SLOTS=N` retained as a diagnostic override.

Changed `src/moe-offload/loader.{h,cpp}`:

- Added `manifest::n_expert_used`.
- Reads the architecture GGUF key ending in `.expert_used_count` so the slot
  pool can size the minimum working set from model metadata before the context
  exists.

Changed `src/llama-graph.cpp`:

- When `LLAMA_MOE_OFFLOAD` streaming mode is active, warmup keeps
  `n_expert_used = hparams.n_expert_used` instead of expanding to
  `hparams.n_expert`.
- This fixes the warmup-only `n_uniq=256 exceeds n_slots=96` abort observed at
  `--moe-cache-vram-mb 8000`.

The default safety factor is `1.0`, which corresponds to the worst-case
capacity bound `n_slots / top_k` before rounding to common ubatch sizes. Lower
values are available for conservative testing.

### Chunked Miss I/O

Changed `src/moe-offload/io.{h,cpp}`:

- Added non-blocking `io_try_acquire_buffer()`.
- Fixed `io_submit()` so a queue-full failure does not leave the outstanding
  counter inflated.

Changed `src/moe-offload/slot_pool.cpp`:

- Reworked the streaming miss path to build a miss-blob list, then submit and
  drain in chunks.
- The callback now drains completions when pinned buffers or queue capacity are
  tight instead of requiring the pool to cover all layer misses.
- Updated `slot_pool_init_io()` buffer sizing comments and minimum; the pool no
  longer assumes `8 * top_k * 3 = 192` in-flight blobs.
- For async H2D completions, the callback now inserts the compute-stream wait
  and synchronizes the H2D event before recycling the pinned staging buffer.
  This avoids reusing host memory while CUDA may still be reading it.

### Bench Support

Changed `tools/moe-bench/main.cpp`:

- Added `-ub N`, `--ubatch N`, and `--ubatch-size N`.
- Plumbs the requested value into `ctx_params.n_ubatch`.
- Reports requested/effective ubatch through the profiler summary.

Changed `src/moe-offload/profiler.{h,cpp}`:

- Added optional summary fields for requested ubatch, effective ubatch,
  slot count, expert count, and streaming/full-residency mode.

## Documentation Changes

Updated:

- `docs/moe-offload/README.md`
- `docs/moe-offload/known-issues.md`
- `tests/moe-offload/README.md`

The docs no longer list the fixed streaming ubatch cap as an open limitation.
They now describe adaptive sizing, the minimum top-k slot reservation, the
remaining diagnostic-only `n_uniq <= n_slots` capacity rule, and the
environment controls.

## Validation Performed

Targeted build:

```powershell
cmake --build build-moe --config Release --target llama-moe-bench llama-completion
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

Small-cache smoke:

```powershell
.\build-moe\bin\Release\llama-completion.exe `
  --model C:/AI/models/qwen/Qwen3.5-35B-A3B-Q4_K_M.moe.gguf `
  --moe-offload `
  --moe-cache-vram-mb 8000 `
  --moe-predictor lru `
  -p "Hello" -n 32
```

Result after the warmup fix: exit code 0. The command generated 32 tokens and
did not reproduce the earlier `n_uniq=256 exceeds n_slots=96` failure.

Explicit-ubatch bench smoke:

```powershell
.\build-moe\bin\Release\llama-moe-bench.exe `
  --model C:/AI/models/qwen/Qwen3.5-35B-A3B-Q4_K_M.moe.gguf `
  --pp 64 --tg 1 --repeat 1 `
  --moe-cache-vram-mb 8000 `
  --moe-predictor lru `
  -ub 16
```

Result: exit code 0. Summary reported `ubatch: requested=16 effective=8`,
`slots=96/256`, and `mode=streaming`, confirming that explicit user ubatch is
auto-downshifted under the 8000 MiB slot budget instead of failing.

## Validation Still Pending

The following still need a dev-box model run:

- Golden-logit regression gate at the old small-cache setting.
- Golden-logit larger-than-8 streaming gates.
- LRU vs EAMC deterministic parity at the largest passing adaptive ubatch.
- `llama-moe-bench` prefill comparison across `-ub 8`, adaptive, `-ub 16`,
  and larger cache budgets.
- Compute-sanitizer run for at least one larger-than-8 streaming ubatch.

## Current Risk

The implementation now passes compile, CTest, and the reported small-cache
completion smoke. The main remaining acceptance bar is still the large-model
golden-logit and prefill-performance matrix; until that passes, the adaptive
path should be treated as functionally fixed for the reported abort but not yet
fully performance-characterized.
