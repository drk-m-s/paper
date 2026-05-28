# MoE Offload MVP Closeout — Execution Progress

Source plan: [docs/implementation_plan_mvp_20260529.md](implementation_plan_mvp_20260529.md)

This file mirrors the current session-memory plan and records what landed
each session. One phase per session.

---

## Phase order

- **G — DONE (this session, 2026-05-28)** — CUDA H2D stream + pinned staging
  plumbing + throwaway micro-test
- H — Wire eval-callback to async H2D + remove the double-read
- I — Real `compute_us` via CUDA events
- J — Streaming golden-logits gate
- K — `tests/moe-offload/` test set (Phase G's micro-test absorbed/replaced)
- L — README truth-up

---

## Phase G — DONE (2026-05-28)

Goal: Land the CUDA-side plumbing needed for async H2D in Phase H, gated
behind `LLAMA_MOE_OFFLOAD`, without changing any runtime path that
`llama-completion` actually exercises today.

### Files added

- `ggml/src/ggml-cuda/moe_offload_io.cu`
  - `extern "C"` symbols, all exported via `GGML_BACKEND_API`, gated by
    `#ifdef LLAMA_MOE_OFFLOAD`:
    - `moe_io_cuda_pinned_alloc(size_t)` → `cudaHostAlloc(..., cudaHostAllocPortable)`
    - `moe_io_cuda_pinned_free(void*)` → `cudaFreeHost`
    - `moe_io_cuda_h2d_async(dst_dev, src_pinned, bytes, void** out_ev)`
      lazily creates a single non-blocking H2D stream
      (`cudaStreamCreateWithFlags(..., cudaStreamNonBlocking)`),
      issues `cudaMemcpyAsync` on it, acquires an event from a
      mutex-protected pool (cap 128, `cudaEventDisableTiming`), records,
      returns the event as `void *`.
    - `moe_io_cuda_event_release(void*)` returns the event to the pool.
    - `moe_io_cuda_compute_wait(ggml_backend_t, void*)` casts
      `backend->context` to `ggml_backend_cuda_context*` and calls
      `cudaStreamWaitEvent(ctx->stream(), ev, 0)` on the backend's
      compute stream.
    - `moe_io_cuda_event_sync(void*)` (host wait — test helper).
    - `moe_io_cuda_events_in_use()` (diagnostic).
  - Single-device MVP. Multi-GPU is post-MVP.

- `tests/moe-offload/test-cuda-stream.cpp`
  - Allocates 4 KiB pinned, 4 KiB device, fills pinned with a pattern,
    issues `io_h2d_async`, `io_event_sync`, `cudaMemcpy` D→H,
    byte-compares, releases everything, asserts
    `io_events_in_use() == 0`.
  - Compiles to a no-op (prints skip / returns 0) when `GGML_USE_CUDA`
    is not defined or no CUDA device is present at runtime.

### Files modified

- `src/moe-offload/io.h`
  - Added `#include "llama.h"` for `LLAMA_API`.
  - Forward-declared `ggml_backend_t`.
  - Added the Phase-G public surface (all marked `LLAMA_API` so they
    are exported from `llama.dll`):
    `io_pinned_alloc`, `io_pinned_free`, `io_h2d_async`,
    `io_event_release`, `io_compute_wait`, `io_event_sync`,
    `io_events_in_use`.

- `src/moe-offload/io.cpp`
  - Added `#include "ggml-backend.h"`.
  - Under `#if defined(GGML_USE_CUDA)`: declared the seven
    `moe_io_cuda_*` symbols `extern "C"` with `GGML_BACKEND_API`
    (so they are pulled from `ggml-cuda.dll` as dllimport), and added
    trivial forwarders for the `llama_moe::io_*` wrappers.
  - Under `#else`: CPU-only stubs that return `nullptr` / `false` / `0`
    so callers can probe at runtime.

- `tests/CMakeLists.txt`
  - Appended an `if (LLAMA_MOE_OFFLOAD)` block that registers
    `moe-offload/test-cuda-stream.cpp` via `llama_build_and_test` with
    label `moe-offload`, adds `src/` to the include path so the test
    can `#include "moe-offload/io.h"`, and links `CUDA::cudart` when
    `find_package(CUDAToolkit)` succeeds.

### Deviations from the plan

1. **Placed the `.cu` file in `ggml/src/ggml-cuda/` instead of
   `src/moe-offload/io_cuda.cu`.** Reason: the ggml-cuda
   `CMakeLists.txt` already enables CUDA-as-language and auto-globs
   `*.cu` from that directory, and `LLAMA_MOE_OFFLOAD` is already
   propagated via top-level `add_compile_definitions`. Putting it in
   `src/moe-offload/` would have required enabling the CUDA language
   for the `llama` target, which was the more invasive change.
2. **No new public header `ggml-cuda-moe.h`.** The accessor is purely
   internal between `io.cpp` ↔ `moe_offload_io.cu`. Both sides declare
   the `moe_io_cuda_*` symbols `extern "C"` with `GGML_BACKEND_API`;
   the dllexport/dllimport split is handled by `GGML_BACKEND_BUILD` /
   `GGML_BACKEND_SHARED` exactly like the other backend exports.
3. **Compute-stream accessor is inlined inside `moe_io_cuda_compute_wait`,
   not exposed as `ggml_backend_cuda_get_stream_moe`.** Same internal
   reason as (2). Phase H will not need the stream as a separate handle
   either — it only needs `cudaStreamWaitEvent` on it.

### Build verification

Vanilla (LLAMA_MOE_OFFLOAD=OFF):
```
cmake --build c:\code\llama.cpp.offload\build-vanilla-check --config Release --target llama
```
→ all four `ggml*.dll` and `llama.dll` linked clean. No regression.

MoE (LLAMA_MOE_OFFLOAD=ON, LLAMA_BUILD_TESTS=ON):
```
cmake -S c:\code\llama.cpp.offload -B c:\code\llama.cpp.offload\build-moe ^
      -DLLAMA_MOE_OFFLOAD=ON -DLLAMA_BUILD_TESTS=ON ^
      -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF
cmake --build c:\code\llama.cpp.offload\build-moe --config Release --target test-cuda-stream
```
→ `test-cuda-stream.exe` linked.

Test run:
```
c:\code\llama.cpp.offload\build-moe\bin\Release\test-cuda-stream.exe
[test-cuda-stream] OK
exit code: 0
```

`llama-completion` runtime path was not touched in this phase
(slot_pool eval-callback unchanged), so no smoke regression is
expected; deferred to Phase H verification.

### Status tracker (Phase G)

- [x] G.1 stream accessor (inlined in `moe_io_cuda_compute_wait`)
- [x] G.2 `.cu` file (`ggml/src/ggml-cuda/moe_offload_io.cu`)
- [x] G.3 `io.h` surface (with `LLAMA_API` exports)
- [x] G.4 CMake wiring (`tests/CMakeLists.txt`)
- [x] G.5 micro-test (`tests/moe-offload/test-cuda-stream.cpp`)
- [x] G.6 builds + runs clean (vanilla + moe + test)
- [x] G.7 this extend.md updated

---

## Next session — Phase H

Wire the eval-callback in `src/moe-offload/slot_pool.cpp` to use
`io_h2d_async` + `io_compute_wait` instead of the current
synchronous `ggml_backend_tensor_set`, and remove the double-read
that the original plan flagged. Re-use the pinned buffers acquired
from the I/O worker pool. The micro-test from Phase G is left in
place as a smoke test until Phase K replaces it with the proper
test set.
