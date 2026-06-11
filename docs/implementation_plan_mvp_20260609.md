# MoE Offload Plan: Adaptive Streaming UBatch From VRAM Budget (2026-06-09)

Repository under work: `C:\code\llama.cpp.offload`.

## Summary

This plan starts the post-MVP fix for the remaining open item:

- Remove the streaming `n_ubatch <= 8` cap.
- Recover prefill throughput by allowing larger streaming microbatches.
- Make the effective `n_ubatch` automatically adjust from the configured MoE
  VRAM budget (`--moe-cache-vram-mb` or `--moe-cache-vram-frac`) instead of
  using one fixed safe value.

The key change from the June 8 plan is that cap removal should not mean
"always trust the user-provided ubatch." Streaming mode still has a real
capacity constraint: every unique routed expert used by one MoE callback must
fit in the layer's slot cache. The new default should therefore be adaptive:
use as large an `n_ubatch` as the current slot budget can plausibly support,
fall back to clear runtime errors when a specific routed batch still exceeds
capacity, and avoid the old unconditional clamp to 8.

## Current State

The fixed cap is still in `src/llama-context.cpp`:

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

The comment above it is stale for the June 5 runtime:

- Streaming remap now uses graph-private `moe.slot_ids.<layer>` tensors.
- The eval callback fills slot ids after the top-k node.
- CUDA `.slot` `MUL_MAT_ID` bypasses specialized MMVQ/MMQ/MMF paths.
- CUDA top-k MoE fusion remains disabled under `LLAMA_MOE_OFFLOAD`.

The remaining real constraints are now slot capacity and I/O scaling, not the
old `ggml_get_rows` -> `mm_ids_helper` crash path.

The slot budget is currently computed in `src/moe-offload/slot_pool.cpp`:

- `compute_n_slots()` converts `runtime_options::cache_vram_mb` into
  `n_slots_per_layer`.
- `streaming_mode()` is true when `n_slots < n_experts_per_layer`.
- `moe_eval_callback()` aborts clearly when `uniq.size() > n_slots`.
- `slot_pool_init_io()` still sizes its pinned-buffer pool around the old
  ubatch=8 assumption.

## Goal

Replace the fixed streaming ubatch cap with a budget-aware default that:

1. Preserves correctness against full-residency golden logits.
2. Raises prefill throughput for streaming caches larger than the MVP minimum.
3. Uses the user's VRAM budget to choose an effective ubatch automatically.
4. Never hides a capacity issue as a CUDA crash, deadlock, or silent logit
   drift.

## Non-Goals

- Do not re-enable CUDA top-k MoE fusion in this plan.
- Do not re-enable specialized CUDA `.slot` `MUL_MAT_ID` fast paths in this
  plan.
- Do not implement router-aware pre-splitting inside a single MoE layer unless
  the adaptive ubatch path proves insufficient.
- Do not add direct I/O, multi-GPU, CPU DRAM expert tier, KV offload, learned
  predictors, or speculative prefetch.

## Design Direction

### Effective UBatch Policy

Introduce a MoE-specific effective ubatch calculation after the slot pool is
configured and before `llama_context` finalizes `cparams.n_ubatch`.

Preferred behavior:

- If MoE offload is disabled: keep upstream `n_ubatch` behavior unchanged.
- If MoE offload is full-residency: keep upstream `n_ubatch` behavior unchanged.
- If MoE offload is streaming:
  - derive `n_slots` from the VRAM budget as today,
  - estimate a safe prefill ubatch from `n_slots` and model top-k
    (`hparams.n_expert_used`),
  - clamp only to that adaptive value when auto mode is active,
  - let explicit user override remain possible for diagnostics.

Proposed first heuristic:

```text
safe_ubatch = floor(n_slots * occupancy_factor / n_expert_used)
```

Initial constants:

- `occupancy_factor = 0.50` for default auto mode.
- minimum adaptive ubatch: `8`.
- maximum adaptive ubatch: requested `n_ubatch`.
- round down to a power of two or a small allowed set: `8, 16, 32, 64, 128,
  256, 512, 1024, 2048`.

Rationale:

- Worst case unique experts is bounded by `n_tokens * n_expert_used`, but real
  routing repeats experts heavily.
- A 50% occupancy factor leaves room for routing diversity without forcing the
  old fixed cap.
- The existing `uniq.size() > n_slots` check remains the hard correctness
  guard for pathological prompts or models.

Example for Qwen3.5-style top-k=8:

| `n_slots` | auto `n_ubatch` target |
| ---: | ---: |
| 48 | 8 |
| 96 | 8 |
| 145 | 8 or 16 after validation |
| 192 | 16 |
| 256 | 16 |

These values are intentionally conservative until the golden-logit matrix
proves larger settings.

### CLI/API Shape

Keep the existing `-ub` / `--ubatch-size` meaning for users who explicitly set
it. Add MoE-specific auto behavior without changing non-MoE defaults.

Preferred knobs:

- `--moe-ubatch auto` as the eventual user-facing default for MoE streaming.
- `--moe-ubatch N` to request a fixed MoE streaming ubatch.
- `--moe-ubatch-safety F` for diagnostics, default `0.50`.

Minimal first implementation if CLI churn should be avoided:

- Treat `params.n_ubatch == 0` as auto.
- Add environment diagnostics:
  - `LLAMA_MOE_STREAMING_UBATCH=auto|N`
  - `LLAMA_MOE_UBATCH_SAFETY=0.50`
  - `LLAMA_MOE_DISABLE_UBATCH_CAP=1` only while validating.

Final preference is a real CLI option, because prefill performance is a user
visible tuning concern and should not require environment variables.

### Logging

Add one startup line when MoE streaming is active:

```text
moe ubatch: requested=2048 effective=16 mode=auto n_slots=192 top_k=8 safety=0.50 cache=8000 MiB
```

If the user explicitly requests a fixed `n_ubatch` above the adaptive estimate,
log a warning but do not silently change it unless the mode is auto.

## Implementation Phases

### Phase A - Confirm Inputs and Add Visibility

1. Add read-only helpers in the MoE runtime/slot pool:
   - `n_slots_per_layer()`
   - `n_experts_per_layer()`
   - `streaming_mode()`
   - optional `cache_vram_mb()` / `cache_vram_frac()` accessors for logging.

2. Add an adaptive ubatch helper:

```cpp
uint32_t llama_moe::recommended_ubatch(
        uint32_t requested,
        uint32_t n_expert_used,
        float safety);
```

3. Move the cap decision out of inline `llama-context.cpp` logic where
   possible, so the policy is testable and documented near the slot-budget
   code.

4. Add logs for requested/effective ubatch, slot count, top-k, and cache budget.

Acceptance:

- Existing `-UBatch 8`, `StreamCacheMb=4000` gate still passes.
- Logs clearly show whether auto sizing or explicit sizing was used.

### Phase B - Fix I/O Scaling Before Larger UBatch

The current I/O loop submits all miss blobs before draining completions. The
buffer-pool comment still assumes `8 * top_k * 3 = 192` requests. That must be
removed before larger prefill ubatches are enabled by default.

Implement chunked submit/drain:

1. Build the list of missing `(expert, kind, slot)` blob loads.
2. Submit while a pinned buffer and worker queue entry are available.
3. Drain completions periodically.
4. For each completion:
   - accumulate SSD timing,
   - insert compute-stream wait for async H2D,
   - preserve H2D event handles for profiler rows,
   - release the pinned buffer.
5. Drain the remainder before writing `slot_ids`.

Acceptance:

- No queue overflow when a large ubatch causes more misses than the buffer
  count.
- No deadlock if misses exceed `slot_pool_init_io()` buffer count.
- `h2d_us`, `ssd_read_us`, and `stall_us` remain populated.

### Phase C - Adaptive UBatch Policy

Replace the fixed cap with adaptive behavior:

1. Remove the unconditional `kMaxStreamingUbatch = 8`.
2. In streaming mode, compute:

```text
adaptive = floor(n_slots * safety / n_expert_used)
effective = min(requested, round_allowed(max(8, adaptive)))
```

3. Only auto-clamp when MoE ubatch mode is auto.
4. Preserve explicit user requests for diagnostic and benchmark runs.
5. Keep `uniq.size() > n_slots` as the hard runtime safety check.

Acceptance:

- No startup log says `capping n_ubatch ... -> 8`.
- Streaming mode can run with effective `n_ubatch > 8` when the cache budget is
  large enough.
- Small-cache streaming remains stable and either uses ubatch 8 or fails with a
  clear `n_uniq > n_slots` message.

### Phase D - Bench Tool Support

Update `tools/moe-bench/main.cpp` so prefill speed can be measured directly:

- Parse `-ub N`, `--ubatch N`, and `--ubatch-size N`.
- Set `ctx_params.n_ubatch`.
- Include requested/effective ubatch in stdout and summary file.
- Include cache budget, `n_slots`, and streaming/full-residency mode in the
  summary if accessors are available.

Acceptance:

- `llama-moe-bench ... --pp 1024 --tg 256 -ub 16` records ubatch in the
  summary.
- Prefill throughput can be compared across `-ub 8`, auto, `-ub 16`, `-ub 32`.

### Phase E - Validation Matrix

Run the matrix after Phase B and Phase C.

Golden-logit gates:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tests\moe-offload\test-golden-logits.ps1 `
  -Model "C:\AI\models\qwen\Qwen3.5-35B-A3B-Q4_K_M.moe.gguf" `
  -Tol 1e-3 -NPredict 8 -StreamCacheMb 4000 `
  -Prompt "Hello" -Seed 42 -Context 4096 -UBatch 8
```

Then test larger caches and ubatches:

| Cache MiB | UBatch | Expected |
| ---: | ---: | --- |
| 4000 | 8 | pass, regression gate |
| 8000 | auto | pass or clear capacity error |
| 8000 | 16 | pass target |
| 12000 | auto | pass target |
| 12000 | 32 | pass stretch |
| 16000 | 64 | stretch |
| full residency | 128+ | pass, reference behavior |

Bench gates:

```powershell
build-moe\bin\Release\llama-moe-bench.exe `
  --model "C:\AI\models\qwen\Qwen3.5-35B-A3B-Q4_K_M.moe.gguf" `
  --pp 1024 --tg 256 --repeat 3 `
  --moe-cache-vram-mb 8000 `
  --moe-predictor eamc `
  --moe-eamc-path tests\moe-offload\_out\moe-bench-adaptive.eamc `
  --moe-profile-csv tests\moe-offload\_out\moe-bench-adaptive.csv `
  --moe-profile-summary tests\moe-offload\_out\moe-bench-adaptive.summary.txt `
  -ngl 99 -c 4096 -ub 16
```

Compare:

- prefill tok/s,
- TTFT,
- total run time,
- prefill hit rate,
- SSD reads during prefill,
- `n_uniq > n_slots` frequency,
- peak VRAM.

### Phase F - Documentation

Update after validation:

- `docs/moe-offload/README.md`
- `docs/moe-offload/known-issues.md`
- `tests/moe-offload/README.md`
- `paper/docs/implementation_plan_mvp_20260605_progress.md`

Replace "streaming mode caps ubatch to 8" with:

- MoE streaming uses adaptive ubatch sizing from the VRAM budget.
- Larger cache budgets allow larger prefill ubatches.
- Very large ubatches can still exceed per-layer unique-expert capacity.
- Explicit `-ub` remains available for benchmarking and diagnostics.

## Acceptance Criteria

The work is done when:

1. Streaming mode no longer silently rewrites all `n_ubatch > 8` requests to 8.
2. Auto mode derives effective ubatch from `--moe-cache-vram-mb` or
   `--moe-cache-vram-frac` through `n_slots`.
3. At least one streaming run with effective `n_ubatch > 8` passes the
   full-residency golden-logit comparison with `max|d| < 1e-3`.
4. The old `StreamCacheMb=4000`, `UBatch=8` gate still passes.
5. `llama-moe-bench` demonstrates improved prefill throughput versus the fixed
   ubatch=8 path on the dev box.
6. Large but cache-feasible ubatches do not deadlock or overflow the I/O queue.
7. Cache-infeasible ubatches fail with a clear `n_uniq > n_slots` diagnostic.
8. MoE CTest label still passes.
9. Docs no longer list the fixed ubatch=8 cap as an open limitation.

## Risks

### Adaptive Estimate May Be Too Optimistic

Router distributions vary by model and prompt. The heuristic can still choose
an ubatch whose actual unique expert count exceeds `n_slots`. Keep the runtime
capacity check and tune the default safety factor from the validation matrix.

### Adaptive Estimate May Be Too Conservative

If auto mode remains at 8 for common 8-12 GB caches, prefill speed will not
improve. Record matrix data and raise the safety factor only when golden logits
and I/O stability support it.

### I/O Chunking Can Distort Profiler Timing

Chunked miss handling changes where host waits happen. Keep the existing
per-row metrics but document that `stall_us` remains approximate.

### User-Provided Fixed UBatch Can Still Fail

This is acceptable if failure is clear. The adaptive default should be safe,
while explicit fixed ubatch remains a power-user override.

## Rollback

If large streaming ubatches still produce CUDA illegal access or logit drift:

1. Keep adaptive auto mode disabled by default.
2. Preserve the old ubatch=8 behavior behind the auto policy.
3. Leave explicit larger ubatch available only under a diagnostic environment
   variable.
4. Record the failing cache/ubatch matrix in `known-issues.md`.

If failures are only capacity errors:

1. Keep adaptive ubatch enabled.
2. Lower the safety factor or allowed ubatch table.
3. Document required cache budgets for larger prefill ubatches.

## Suggested Work Order

1. Add ubatch visibility logs and MoE budget accessors.
2. Implement chunked submit/drain in `moe_eval_callback()`.
3. Add `llama-moe-bench` `-ub` support.
4. Add adaptive ubatch helper and replace the fixed cap.
5. Run golden-logit matrix.
6. Run bench matrix and choose the default safety factor.
7. Update README, known issues, tests docs, and progress notes.
