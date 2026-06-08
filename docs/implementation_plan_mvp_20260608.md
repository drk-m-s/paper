# MoE Offload Post-MVP Plan: Remove Streaming `n_ubatch <= 8` Cap (2026-06-08)

Repository under work: `C:\code\llama.cpp.offload`.

## Summary

Remove the fixed streaming `n_ubatch <= 8` workaround without regressing the
June 5 MVP correctness gates. The work should prove that larger streaming
microbatches are safe with the current slot-id remap path, then remove or
replace the hard cap in `src/llama-context.cpp`.

This is post-MVP performance hardening. Correctness remains the acceptance bar:
larger streaming ubatches must match full-residency logits or fail clearly
before unsafe execution.

## Current State

The cap is currently enforced in `src/llama-context.cpp` after
`cparams.n_ubatch` is derived:

```cpp
if (llama_moe::runtime_enabled() && llama_moe::streaming_mode()) {
    constexpr uint32_t kMaxStreamingUbatch = 8;
    if (cparams.n_ubatch > kMaxStreamingUbatch) {
        LLAMA_LOG_WARN("%s: capping n_ubatch %u -> %u for MoE streaming safety\n",
                __func__, cparams.n_ubatch, kMaxStreamingUbatch);
        cparams.n_ubatch = kMaxStreamingUbatch;
    }
}
```

The comment above that cap is stale. It refers to the older
`ggml_get_rows` -> `mm_ids_helper` path. The current June 5 implementation no
longer uses `ggml_get_rows` for streaming remap:

- `remap_selected_experts()` returns a graph-private
  `moe.slot_ids.<layer>` I32 tensor in streaming mode.
- The eval callback fills that tensor directly after the registered top-k node
  executes.
- CUDA `.slot` `MUL_MAT_ID` bypasses specialized MMVQ/MMQ/MMF paths and uses
  the generic sorted CUDA path.
- CUDA top-k MoE fusion remains disabled under `LLAMA_MOE_OFFLOAD`.

The fixed cap is therefore probably no longer needed for the original CUDA
kernel crash, but two remaining constraints must be handled before removing it:

1. **Per-layer cache capacity**: every unique expert selected by a callback must
   fit in the layer's slot cache. The runtime already aborts if
   `uniq.size() > n_slots`.
2. **I/O buffering and queue depth**: the async I/O path was sized around the
   old cap. `slot_pool_init_io()` assumes worst case `8 * top_k * 3 = 192`
   in-flight blob requests, while larger ubatches can require many more missed
   expert blobs before the callback drains completions.

## Non-Goals

- Do not re-enable CUDA top-k MoE fusion in this plan.
- Do not re-enable specialized CUDA `.slot` `MUL_MAT_ID` MMVQ/MMQ/MMF paths in
  this plan.
- Do not implement router-aware token splitting inside a single MoE layer in
  this plan unless plain cap removal proves impossible.
- Do not add direct I/O, multi-GPU, CPU DRAM expert tier, KV offload, learned
  predictors, speculative prefetch/decoding, or FineMoE-style splitting.

## Core Hypothesis

With the June 5 slot-id remap and generic `.slot` `MUL_MAT_ID` path, the
original large-ubatch CUDA crash should be gone. If a larger ubatch fails now,
the likely causes are:

- `uniq.size() > n_slots` for the selected experts in a layer.
- Async I/O buffer/queue starvation because too many miss requests are submitted
  before any completions are drained.
- A remaining ordering bug in callback-filled `moe.slot_ids.<layer>` tensors or
  async H2D waits.

## Implementation Plan

### Phase A - Baseline and Visibility

1. Add or confirm logging for the effective context microbatch:
   - Requested `params.n_ubatch`.
   - Final `cparams.n_ubatch`.
   - Whether MoE streaming mode is active.
   - `n_slots` per layer.

2. Add temporary diagnostic override for the cap:
   - Environment variable: `LLAMA_MOE_DISABLE_UBATCH_CAP=1`.
   - Scope: debug/development only.
   - Purpose: reproduce and classify failures before deleting the cap.
   - Remove this override or leave it undocumented once the final behavior is
     decided.

3. Add a dev-box matrix script:
   - Suggested path: `tests/moe-offload/test-streaming-ubatch-matrix.ps1`.
   - Inputs: model path, cache MB, ubatch list, prompt, seed, tolerance.
   - For each ubatch, run the golden-logit harness and collect:
     - exit code,
     - `max|d|`,
     - first failing step if any,
     - whether `n_uniq > n_slots` occurred,
     - whether CUDA illegal access occurred,
     - whether the cap warning appeared.

4. Run initial diagnostic matrix with the cap disabled:
   - Cache: `4000`, `8000`, `12000`, and full residency.
   - UBatch: `8`, `16`, `32`, `64`, `128`.
   - Decode: `-n 8`, prompt `"Hello"`, `--temp 0 --seed 42`.

Expected Phase A output:

- Identify the largest passing streaming ubatch per cache size.
- Separate capacity failures from CUDA/kernel failures.
- Confirm whether the original `>8` CUDA crash is still reproducible.

### Phase B - Fix Async I/O Scaling

The current miss loop can submit all missed expert blobs for a layer before
draining completions. That was acceptable with the old cap and a 192-buffer
floor, but it is fragile for larger ubatches.

Implement bounded miss submission:

1. Convert the per-callback miss handling into a submit/drain loop:
   - Build the list of missing `(expert, kind, slot)` blobs first.
   - Submit requests while buffers and queue capacity are available.
   - Drain completed requests periodically.
   - For each completion, insert the compute-stream wait, record H2D timing
     events, release the pinned buffer, and continue.
   - After all requests are submitted, drain the remainder before filling
     `slot_ids`.

2. Avoid requiring the buffer pool to cover the entire layer miss set:
   - Keep a reasonable minimum buffer pool.
   - Do not deadlock if required blobs exceed `n_buffers`.
   - Do not require `work_queue::kCapacity` to exceed the entire layer miss set.

3. Update comments in `slot_pool_init_io()`:
   - Remove the hard-coded `8 * top_k * 3` assumption.
   - Document that the callback supports chunked miss submission.

4. Keep existing safety checks:
   - `rec.size <= stride`.
   - slot tensor OOB write protection.
   - `uniq.size() <= n_slots`.
   - missing slot mapping abort after residency.

Acceptance for Phase B:

- No deadlock when ubatch produces more than 192 blob requests.
- No queue overflow for large-but-cache-feasible ubatches.
- H2D timing events are still released at `slot_pool_end_request()`.

### Phase C - Remove the Fixed Cap

Once Phase A/B show the larger ubatch path is stable, replace the hard cap in
`src/llama-context.cpp`.

Preferred final behavior:

- No automatic `n_ubatch -> 8` rewrite in streaming mode.
- Keep a warning only if the user requests a large ubatch with an obviously
  small cache, but do not silently change the value.
- Let the runtime's `uniq.size() > n_slots` check fail clearly if a specific
  routed layer cannot fit in cache.

Alternative if a hard safety guard is still needed:

- Replace the fixed `8` with an opt-in guard, for example:
  `LLAMA_MOE_STREAMING_UBATCH_CAP=N`.
- Default should be uncapped once golden-logit gates pass.

Update stale comments:

- Remove references to `ggml_get_rows` as the active remap path.
- Explain that the old cap was a workaround for the pre-June-5 remap path.
- Cross-reference the current slot-id remap and generic `.slot`
  `MUL_MAT_ID` guard.

### Phase D - Bench and Test Knobs

`llama-moe-bench` currently does not parse `-ub` or `--ubatch`, so it cannot
measure the cap-removal result directly.

Add:

- `bench_params::n_ubatch`.
- Parse `-ub N`, `--ubatch N`, and optionally `--ubatch=N`.
- Set `ctx_params.n_ubatch = p.n_ubatch` when provided.
- Include effective/requested ubatch in the benchmark summary.

Extend test scripts:

- `test-golden-logits.ps1` already has `-UBatch`; keep it as the numeric
  correctness harness.
- Add matrix wrapper for repeated ubatch/cache sweeps.
- Preserve outputs under `tests/moe-offload/_out/`.

### Phase E - Documentation

Update:

- `docs/moe-offload/README.md`
- `docs/moe-offload/known-issues.md`
- `tests/moe-offload/README.md`
- `paper/docs/implementation_plan_mvp_20260605_progress.md`

Required documentation changes:

- Remove "streaming mode caps ubatch to 8" from open known issues once the
  validation gates pass.
- Replace it with a capacity limitation:
  `n_uniq` selected experts in a callback must fit in `n_slots`.
- Document recommended cache/ubatch combinations observed on the dev box.
- Document that very large ubatches with small streaming caches can still fail
  clearly due to slot capacity.

## Validation Plan

### Build and CTest

```powershell
cmake --build build-moe --config Release --target `
  llama-completion llama-moe-bench `
  test-cuda-stream test-eamc-cosine test-lru-eviction `
  test-manifest-roundtrip test-repack-slices

ctest --test-dir build-moe -C Release -L moe-offload --output-on-failure
```

Expected:

- Build passes.
- CTest label `moe-offload` passes 6/6.

### Golden-Logit Gates

Baseline regression gate:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tests\moe-offload\test-golden-logits.ps1 `
  -Model "C:\AI\models\qwen\Qwen3.5-35B-A3B-Q4_K_M.moe.gguf" `
  -Tol 1e-3 -NPredict 8 -StreamCacheMb 4000 `
  -Prompt "Hello" -Seed 42 -Context 4096 -UBatch 8
```

New large-ubatch gates:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tests\moe-offload\test-golden-logits.ps1 `
  -Model "C:\AI\models\qwen\Qwen3.5-35B-A3B-Q4_K_M.moe.gguf" `
  -Tol 1e-3 -NPredict 8 -StreamCacheMb 8000 `
  -Prompt "Hello" -Seed 42 -Context 4096 -UBatch 16

powershell -NoProfile -ExecutionPolicy Bypass -File tests\moe-offload\test-golden-logits.ps1 `
  -Model "C:\AI\models\qwen\Qwen3.5-35B-A3B-Q4_K_M.moe.gguf" `
  -Tol 1e-3 -NPredict 8 -StreamCacheMb 12000 `
  -Prompt "Hello" -Seed 42 -Context 4096 -UBatch 32
```

Stretch gates, if cache capacity allows:

- `StreamCacheMb=12000`, `UBatch=64`.
- `StreamCacheMb=16000` or full residency, `UBatch=128`.

Expected:

- Passing runs report `max|d| < 1e-3`.
- If a cache is too small, failure is a clear `n_uniq > n_slots` error, not a
  hang, queue overflow, CUDA illegal access, or silent logit drift.
- No log line says `capping n_ubatch ... -> 8`.

### LRU vs EAMC Parity

For the largest passing streaming ubatch:

- Run deterministic LRU logit dump.
- Run deterministic EAMC logit dump.
- Compare with `compare_logits.py --tol 1e-7`.

Expected:

- `max|d| = 0` or within the same tolerance used by the June 5 parity gate.

### Bench Gate

After adding `-ub` support to `llama-moe-bench`:

```powershell
build-moe\bin\Release\llama-moe-bench.exe `
  --model "C:\AI\models\qwen\Qwen3.5-35B-A3B-Q4_K_M.moe.gguf" `
  --pp 1024 --tg 256 --repeat 3 `
  --moe-cache-vram-mb 8000 `
  --moe-predictor eamc `
  --moe-eamc-path tests\moe-offload\_out\moe-bench-ubatch.eamc `
  --moe-profile-csv tests\moe-offload\_out\moe-bench-ubatch.csv `
  --moe-profile-summary tests\moe-offload\_out\moe-bench-ubatch.summary.txt `
  -ngl 99 -c 4096 -ub 16
```

Expected:

- Completes without hang, CUDA illegal access, queue overflow, or use-after
  evict.
- Emits CSV and summary.
- Summary records the requested/effective ubatch.
- No fixed cap warning appears.

### Compute-Sanitizer Gate

Run at least one larger-than-8 streaming decode under compute-sanitizer:

- Use the smallest prompt/decode that reproduces the larger ubatch path.
- Prefer `UBatch=16` first, then `UBatch=32`.

Expected:

- No CUDA illegal memory access.
- No invalid global read/write.

## Acceptance Criteria

The cap-removal work is complete when:

1. `src/llama-context.cpp` no longer silently rewrites streaming
   `n_ubatch` to 8 by default.
2. At least one streaming ubatch greater than 8 passes the full-residency
   golden-logit comparison with `max|d| < 1e-3`.
3. The old `UBatch=8`, `StreamCacheMb=4000` golden gate still passes.
4. LRU and EAMC still produce identical deterministic logits at the largest
   passing ubatch.
5. The 1024 prefill + 256 decode bench gate completes with explicit
   `-ub > 8`.
6. If a requested ubatch cannot fit the available slot cache, the failure is
   clear and early enough to diagnose, not a deadlock or CUDA crash.
7. MoE CTest label passes.
8. README, known issues, and test docs no longer claim an unconditional
   streaming cap of 8.

## Risks

### Large ubatch may exceed slot capacity

This is not the same as the old CUDA kernel crash. With small streaming caches,
large prefill ubatches can route to more unique experts than available slots.
The current single-pass slot cache cannot safely serve that layer. The correct
behavior is a clear capacity error unless a future router-aware split is added.

### I/O queue and buffer assumptions may deadlock

The current worker queue and buffer pool were sized with the old cap in mind.
Larger ubatches can require chunked miss submission and completion draining.

### Generic `.slot` path may still have hidden large-shape bugs

The generic sorted CUDA `MUL_MAT_ID` path performs host/device synchronization
and should be safer than MMQ/MMVQ/MMF, but it still needs explicit large-ubatch
validation.

### Documentation may overpromise

Removing the fixed cap does not mean every cache size supports every ubatch.
Docs must distinguish "no hard cap of 8" from "slot capacity is unlimited."

## Rollback Plan

If larger-than-8 streaming ubatches show CUDA illegal access or logit drift that
cannot be fixed locally:

1. Keep the default cap in place.
2. Update comments to describe the current slot-id path accurately.
3. Keep an opt-in diagnostic override for continued investigation.
4. Document the latest failing ubatch/cache matrix in `known-issues.md`.

If only small-cache capacity failures occur:

1. Remove the fixed cap anyway.
2. Keep the `n_uniq > n_slots` hard error.
3. Document required cache sizing or recommend lower ubatch for that cache.

## Suggested Work Order

1. Add diagnostic cap override and effective-ubatch logging.
2. Add `-ub` support to `llama-moe-bench`.
3. Run the uncapped diagnostic matrix and identify the first real blocker.
4. Implement chunked miss submission/draining if buffer or queue pressure is
   observed.
5. Remove the default hard cap.
6. Run golden-logit, parity, bench, CTest, and compute-sanitizer gates.
7. Update README, known issues, test docs, and progress notes.
