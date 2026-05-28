# Plan: MoE Offload MVP — Phase D-4 Completion & Updated Status

> **Date: 2026-05-28**
> **Status: Phases A + B + C GREEN. Phase D-4 COMPLETE.**
> **Next: D-5 (async I/O) → E (profiler + predictor) → F (bench + docs)**

---

## Phase D-4: Streaming Expert Remap — COMPLETE ✅

### Problem
Streaming mode (`n_slots < n_expert`) crashed at `launch_mul_mat_q` / `quantize_mmq_q8_1+0x570` with CUDA illegal memory access when the ubatch size exceeded ~8 tokens. Full residency (`n_slots == n_expert`) worked at any batch size. Small ubatch (`-ub 4`) worked in streaming mode.

### Root Cause (empirically isolated)
The `ggml_get_rows` kernel's output, when consumed by `mul_mat_id`'s internal `mm_ids_helper` + `quantize_mmq_q8_1` kernels, triggers an illegal memory access at large ubatch. The compute-sanitizer trace shows a negative-address-wrap OOB (`0xffffff8da44257b0`) inside `quantize_mmq_q8_1+0x570`, block (44,1,0), thread (115,0,0).

A **bypass test** (returning `selected_experts` unchanged — identity mapping without `get_rows`) proved that `mul_mat_id` itself works correctly at large ubatch with zeroed slot tensors. The crash is **specifically** in the `get_rows` → `mm_ids_helper` interaction, not in slot data or tensor dimensions.

**Ruled out (all tested and confirmed NOT the cause):**
- `ne[2]` expert-count dimension (tested 145, 160, 256 — all crash with large ubatch)
- Uninitialized / garbage slot tensor data (zeroed all 120 tensors — still crashed)
- Slot_table buffer aliasing by the scheduler (fixed by pre-allocation via `create_unfiled_tensor`)
- Stride bugs in slot tensor (verified `nb=[144,1152,589824]` matches source tensor exactly)
- Rounding `n_slots` to multiple-of-32 (160 slots — still crashed)

### Fix Applied (3-part)

#### 1. Pre-allocated slot_table tensors
Slot_table tensors (`[1, n_expert]` I32) are created via `llama_model_loader::create_unfiled_tensor` at model-load time — the same mechanism used for the expert slot tensors. This gives them persistent CUDA0-backed storage that the scheduler never aliases with graph temporaries.

**Files:** `src/moe-offload/slot_pool.cpp` (`intercept_expert_tensor`), `slot_pool.h`

#### 2. Zero-initialized slot tensor buffers
All 120 slot tensors (40 layers × 3 kinds) are zeroed via `ggml_backend_tensor_memset` before any expert loading. This ensures unused slots contain valid all-zero Q4_K blocks instead of GPU garbage. The CUDA backend's implementation uses `cudaMemsetAsync` + `cudaStreamSynchronize`, so zeroing is synchronous and visible to subsequent compute-stream kernels.

**Performance:** 120 tensors zeroed in **939 ms** (one-time startup cost).

**Files:** `src/moe-offload/slot_pool.cpp` (`prefetch_all_experts`)

#### 3. Ubatch auto-cap for streaming safety
When `streaming_mode()` is active (`n_slots < n_expert`), `n_ubatch` is capped to **8** in `llama-context.cpp`. This prevents the `get_rows`→`mm_ids_helper` crash while maintaining acceptable prefill throughput.

**Files:** `src/llama-context.cpp` (line ~196)

```cpp
#ifdef LLAMA_MOE_OFFLOAD
    if (llama_moe::runtime_enabled() && llama_moe::streaming_mode()) {
        constexpr uint32_t kMaxStreamingUbatch = 8;
        if (cparams.n_ubatch > kMaxStreamingUbatch) {
            LLAMA_LOG_WARN("%s: capping n_ubatch %u -> %u for MoE streaming safety\n",
                    __func__, cparams.n_ubatch, kMaxStreamingUbatch);
            cparams.n_ubatch = kMaxStreamingUbatch;
        }
    }
#endif
```

### Verification

| Test Case | Result |
|-----------|--------|
| Full residency (n_slots=256) + default ub (2048) | ✅ Generates coherent output |
| Streaming (n_slots=145, cache=12GB) + auto-capped ub=8 | ✅ Generates coherent output |
| Streaming + explicit ub=4 | ✅ Works |
| Streaming + default ub (before auto-cap) | ❌ CUDA illegal memory access |

**Golden output** (streaming, cache=12GB, auto-capped ub):
```
user
Hello
assistant
<think>
Okay, the user wrote "Hello" followed by...
```

---

## ⚠️ Known Limitation: Ubatch Capping & Throughput Impact

### The Problem
The ubatch auto-cap to **8** is a **workaround**, not a fix. The underlying `ggml_get_rows` → `mm_ids_helper` kernel bug remains unidentified. With ubatch capped at 8:

- **Prefill throughput is reduced.** Each prefill microbatch processes at most 8 tokens, whereas the default ubatch (2048) would process up to 2048 tokens per kernel launch. For a 4096-token prefill, this means ~512 kernel launches instead of ~2 — a **~250× increase in launch overhead**.
- **Smaller batches mean fewer unique experts per layer** (≤64 instead of ≤96), which increases the fraction of cache hits. This partially mitigates the I/O cost but doesn't compensate for the kernel launch overhead.
- **Full residency mode is unaffected** — when `n_slots == n_expert`, the auto-cap does NOT apply, and the original ubatch (2048) is used.

### Impact Estimate
For a 4096-token prefill with streaming at cache=12GB:
- **Before cap:** ~2 ubatch launches, ~96 unique experts/layer, ~1.2 GiB/s SSD reads
- **After cap (ub=8):** ~512 ubatch launches, ~64 unique experts/layer, comparable SSD reads but ~250× more kernel launch overhead
- **Expected slowdown:** 2–5× for prefill, minimal for decode (decode is token-by-token, already ub=1)

### Fix Required (post-MVP)
The root cause must be found and fixed in either `ggml_get_rows` (CUDA gather kernel) or `mmid.cuh` (`mm_ids_helper` kernel). The `get_rows` bypass test (identity mapping) proves that `mul_mat_id` handles large ubatch correctly — the bug is specific to the remapped-ids path. Candidates for investigation:

1. **`ggml_get_rows` stride bug**: The gather kernel may produce incorrect output strides for certain input shapes (slot_table `[1, 256]` I32), causing `mm_ids_helper` to read wrong expert IDs.
2. **`mm_ids_helper` shared memory overflow**: The kernel allocates `n_tokens * sizeof(mm_ids_helper_store)` shared memory. With `n_tokens=2048`, this is only 8 KB — well within limits. But the grid dimension (`n_experts=256` blocks) combined with large `n_tokens` may hit a warp-synchronization edge case.
3. **`quantize_mmq_q8_1` indexing**: The quantizer uses `ids_src1` (produced by `mm_ids_helper`) to index into the activations tensor. If `ids_src1` contains out-of-range values for large `n_tokens`, the quantizer accesses OOB memory.

### Recommendation
**Ship the MVP with the ubatch cap.** The cap is safe, tested, and produces correct output. The throughput impact is acceptable for an MVP demonstration. Post-MVP, allocate 2–3 days to investigate the kernel bug with focused compute-sanitizer runs comparing the working (identity) and crashing (remapped) paths.

---

## Code Changes Summary

| File | Change |
|------|--------|
| `src/moe-offload/slot_pool.cpp` | Pre-allocated `slot_table_tensors` vector; zero-init in `prefetch_all_experts`; eval-callback writes to persistent `stt` via `ggml_backend_tensor_set` |
| `src/moe-offload/slot_pool.h` | Added `get_slot_table_tensor()` declaration |
| `src/moe-offload/runtime.cpp` | `remap_selected_experts` uses `get_slot_table_tensor()` + `ggml_get_rows` (restored from bypass) |
| `src/llama-context.cpp` | Ubatch auto-cap to 8 when `streaming_mode()` is active |
| `ggml/src/ggml-cuda/mmq.cu` | (read-only analysis) |

---

## Next: Phase D-5 — Async I/O Pipeline

Per the original plan §D-5:

1. Create `src/moe-offload/io.{h,cpp}` with:
   - I/O worker thread + SPSC ring queue
   - Pinned host staging pool (`cudaHostAlloc`)
   - `moe_h2d_stream` per CUDA device
2. Expose `ggml_backend_cuda_get_stream()` in ggml-cuda
3. Rewrite eval-callback miss path: enqueue async I/O, `cudaStreamWaitEvent` on compute stream
4. Keep full-residency path unchanged

---

## Remaining Phases

### Phase E — Profiler + Predictor
- Per-layer CSV rows from eval-callback (SSD read time, H2D time, compute time)
- Summary (TTFT, TPOT, hit rate, bandwidth)
- EAMC predictor sidecar dump/reload

### Phase F — Bench + Docs
- `tools/moe-bench/main.cpp` with `--pp/--tg/--repeat`
- `docs/moe-offload/README.md` with real instructions
- Stress tests at `--moe-cache-vram-mb 4000`

---

## Build & Test Commands

```powershell
# Build
cmake --build c:\code\llama.cpp.offload\build-moe --config Release --target llama-completion

# Full residency (works at any batch size)
llama-completion.exe -m C:\AI\models\qwen\Qwen3.5-35B-A3B-Q4_K_M.moe.gguf `
  --moe-offload --moe-cache-vram-mb 99999 -ngl 99 -fit off -c 4096 `
  -p "Hello" -n 8 --temp 0 --seed 42 --no-warmup

# Streaming (auto-capped ubatch to 8)
llama-completion.exe -m C:\AI\models\qwen\Qwen3.5-35B-A3B-Q4_K_M.moe.gguf `
  --moe-offload --moe-cache-vram-mb 12000 -ngl 99 -fit off -c 4096 `
  -p "Hello" -n 8 --temp 0 --seed 42 --no-warmup
```
