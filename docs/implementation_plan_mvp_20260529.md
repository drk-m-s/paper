# Plan: MoE Offload MVP — Closeout (2026-05-29)

> **Status going in (2026-05-28 EOD)**: Phases A, B, C, D-1…D-4, E, and F are
> functionally landed. The build is green, `llama-completion` and
> `llama-moe-bench` run end-to-end on `Qwen3.5-35B-A3B-Q4_K_M.moe.gguf`, the
> profiler/summary writer emits the §4.7 format, and full-residency mode
> reproduces vanilla argmax. **However**, three of the original MVP deliverables
> are still not honestly satisfied:
>
> 1. **No true async I/O / overlap.** `src/moe-offload/io.cpp` worker only
>    does `_fseeki64`+`fread` into heap buffers; the eval-callback calls
>    `io_wait_all()` and then re-reads the same blob synchronously through
>    `load_expert_into_slot` to do `ggml_backend_tensor_set` (the “FIXME” path
>    documented in 20260528_extend.md). There is no `moe_h2d_stream`, no
>    `cudaMemcpyAsync`, no `cudaStreamWaitEvent`, no `cudaHostAlloc`, and no
>    Windows direct I/O. As a result:
>    - `stall_us` is `ssd_read_us + h2d_us` (sync miss cost), not
>      “compute-stream wait loss”.
>    - `compute_us` is hard-wired to `0` in every CSV row because no CUDA
>      event timing is wrapped around the per-layer MoE compute.
> 2. **Streaming numeric correctness gate not formally recorded.** Phase C
>    proved full-residency argmax equality; Phase D-4 only reported
>    “generates coherent output” at `--moe-cache-vram-mb 12000`. The
>    `max|Δlogit| < 1e-3` test required by Plan §7 / §11 Deliverable 3 against a
>    forced-eviction cache (≤ 4 GB) has not been executed and logged.
> 3. **No tests under `tests/moe-offload/`.** Plan §9 lists eight tests; zero
>    exist on disk. Even the smoke harness from §9 item 7 is missing.
>
> Two additional items the original plan promised are still open but already
> documented as deferred / accepted limitations:
>
> - **Streaming ubatch is capped at 8** (`src/llama-context.cpp`) because of an
>   unresolved `ggml_get_rows` → `mm_ids_helper` interaction. Plan
>   20260528_extend.md explicitly accepts this for the MVP; we keep that
>   acceptance.
> - **`--moe-oracle`** mode (Plan §8) was explicitly deferred at sign-off
>   (Plan §12 “Decisions captured” / 20260527_extend.md). We keep it deferred.
>
> This document is the closeout plan: do the smallest set of changes that turn
> the three gaps above into honest, measured MVP deliverables, without
> attempting any new performance feature.

---

## 1. Closeout scope

In scope (must land for MVP sign-off):

1. **Real async H2D pipeline** — pinned host staging, dedicated CUDA H2D
   stream, `cudaMemcpyAsync`, `cudaEventRecord`, `cudaStreamWaitEvent` on the
   compute stream. Eliminates the “read twice” FIXME and makes `stall_us`
   mean what the plan says it means.
2. **CUDA event timing for per-layer MoE compute** — `compute_us` becomes a
   real number in the CSV and summary.
3. **Streaming numeric correctness gate** — automated comparison vs. the
   Phase C golden run at `--moe-cache-vram-mb 4000`, with the result checked
   into `tests/moe-offload/` and the measured `max|Δlogit|` recorded in the
   docs.
4. **Minimum viable test set** — three tests under `tests/moe-offload/`
   sufficient to cover Plan §9 items 1, 2, 5, 6, 7 at a level that can run on
   the dev box. Unit-test items 3 and 4 (cache / io low-level) are folded into
   the integration tests rather than carried as separate GoogleTest binaries.
5. **README truth-up** — `docs/moe-offload/README.md` records the measured
   numbers, the ubatch=8 streaming cap, and what `compute_us` / `stall_us` now
   actually mean.

Out of scope (explicitly deferred, do not touch in this milestone):

- Removing the streaming `n_ubatch <= 8` cap. The root cause is in upstream
  CUDA kernels and is post-MVP work.
- `--moe-oracle` mode.
- Pinned-buffer pool tuning, multi-thread I/O worker, NUMA, multi-GPU.
- Any new predictor, new model architecture, or graph-shape change.
- Windows `FILE_FLAG_NO_BUFFERING` direct I/O. The buffered `fread` path is
  fast enough (1.16 GiB/s measured in Phase C) and is not a correctness gate.
  We will note this as a deferred optimization in the README; we will not
  block MVP on it.

---

## 2. Architecture deltas vs. the 2026-05-27/28 state

Only two subsystems change behavior:

### 2.1 I/O subsystem (`src/moe-offload/io.{h,cpp}` + new `.cu`)

Today: pure C++, heap buffers, blocking `fread`. Eval-callback calls
`io_wait_all()` then `load_expert_into_slot` re-reads + does sync
`ggml_backend_tensor_set`.

Closeout: split the worker’s job into **read** (CPU, fread) and **upload**
(GPU, `cudaMemcpyAsync` on `moe_h2d_stream` + `cudaEventRecord`). Eval-callback
collects the per-miss event list and calls `cudaStreamWaitEvent` on the
compute stream once, then returns. No double-read.

Because llama.cpp’s build system does not expose CUDA headers to `.cpp` files
(see 20260528_extend.md “Design Decision”), the CUDA-touching pieces move into
a tiny new `src/moe-offload/io_cuda.cu` compiled by nvcc and linked when
`LLAMA_MOE_OFFLOAD` and `GGML_CUDA` are both on. `io.cpp` keeps the worker
thread and queue; it calls into `io_cuda.cu` for pinned alloc, async copy,
event record/wait.

Public surface added to `io.h` (still pure C++):

```cpp
// returns nullptr if CUDA unavailable; caller falls back to malloc + sync H2D
void * io_pinned_alloc(size_t bytes);
void   io_pinned_free (void * p);

// queue a host->device copy on moe_h2d_stream; record event into *out_ev
// on completion. Returns false if CUDA unavailable.
bool   io_h2d_async(void * dst_dev, const void * src_pinned, size_t bytes,
                    void ** out_ev);

// release event back to pool
void   io_event_release(void * ev);

// make the backend's compute stream wait for *ev before next kernel
bool   io_compute_wait(void * ev);
```

The single accessor needed inside ggml-cuda is a guarded
`ggml_backend_cuda_get_stream(int device)` already called out in the original
plan and in 20260527_extend.md “Further considerations” #2. We add it once,
under `#ifdef LLAMA_MOE_OFFLOAD`, in `ggml/src/ggml-cuda/ggml-cuda.cu`.

### 2.2 Profiler — real `compute_us`

Today: `row.compute_us = 0`. The profiler already aggregates everything else.

Closeout: wrap each MoE layer’s `mul_mat_id` dispatch in a pair of
`cudaEvent_t` (`compute_begin[il]`, `compute_end[il]`) recorded on the
backend’s compute stream. The eval-callback already runs once per MoE layer
both *before* and *after* the relevant graph nodes, so the hook points exist;
we record `compute_begin` on the post-topk callback (right after the
slot_table write) and `compute_end` on the next-layer entry (or on
`end_request` for the last layer). `cudaEventElapsedTime` is computed lazily
in `end_request` to avoid stalling the compute stream mid-decode.

Events are pooled per layer with double-buffering so a decode token doesn’t
wait on the previous token’s elapsed-time computation.

No other profiler change; the snapshot/exports from Phase E stay byte-stable
except `compute_us` ≠ 0.

---

## 3. Phases

Each phase is independently testable and reverts cleanly.

### Phase G — CUDA stream accessor + pinned staging plumbing

Goal: get the smallest possible CUDA surface in place without changing
behavior yet.

Edits:
- `ggml/src/ggml-cuda/ggml-cuda.cu`: add
  ```cpp
  #ifdef LLAMA_MOE_OFFLOAD
  extern "C" cudaStream_t ggml_backend_cuda_get_stream(int device);
  #endif
  ```
  implemented by returning `ctx->stream(device, 0)` from the active
  `ggml_backend_cuda_context`. Declared in a new header
  `ggml/include/ggml-cuda-moe.h` that is only installed when
  `LLAMA_MOE_OFFLOAD` is on.
- New `src/moe-offload/io_cuda.cu` implementing the five functions in §2.1.
  `cudaHostAlloc(cudaHostAllocPortable)` for staging; one `cudaStream_t
  moe_h2d_stream` per CUDA device; lock-free event pool (`std::vector<cudaEvent_t>`
  guarded by a small spinlock — events are cheap, contention is per-layer).
- `src/CMakeLists.txt` and `src/moe-offload/` CMake: add `io_cuda.cu` to the
  `llama` target when both `LLAMA_MOE_OFFLOAD` and `GGML_CUDA` are on; keep a
  no-op C++ shim when CUDA is off so the rest of `io.cpp` compiles.

Verification:
- `cmake -B build-vanilla-check` (offload OFF) still configures, builds, and
  runs `test-backend-ops` clean → no behavioral regression.
- `cmake -B build-moe -DLLAMA_MOE_OFFLOAD=ON -DGGML_CUDA=ON` builds; symbol
  `ggml_backend_cuda_get_stream` is exported from `ggml.dll` (verify with
  `dumpbin /EXPORTS`).
- A throwaway micro-test in `tests/moe-offload/test-cuda-stream.cpp` calls
  `io_pinned_alloc(4096)` and `io_h2d_async` into a small device buffer and
  asserts the event signals — exists only to prove the plumbing works; deleted
  after Phase H.

### Phase H — Wire the eval-callback to async H2D + remove the double-read

Edits in `src/moe-offload/slot_pool.cpp`:

- Replace today’s “submit-then-`load_expert_into_slot`” code path with:
  1. Acquire pinned buffer via `io_pinned_alloc` at startup
     (`expert_blob_size_max` rounded up to 4 KiB, `n_io_inflight` of them,
     default 32). Fall back to `malloc` if `io_pinned_alloc` returns null
     (CPU-only build).
  2. `io_submit` the read as today (worker does `_fseeki64` + `fread` into
     pinned buf).
  3. On worker completion, instead of pushing only to `completed_list`, the
     worker also calls `io_h2d_async(slot_dst, pinned_buf, size, &ev)` and
     stores `ev` in the completion record. The H2D runs on `moe_h2d_stream`
     entirely off the compute thread.
  4. The eval-callback drains completions for the current layer, collects the
     event list, calls `io_compute_wait(ev)` for each, then releases the
     events and recycles the pinned buffers.
- Delete the fallback that re-reads through `load_expert_into_slot` after
  `io_wait_all()`. Keep `load_expert_into_slot` itself — it is still used by
  `prefetch_all_experts` (Phase C startup path) and by full-residency mode.
- Update profiler row emission: `h2d_us` is now read from CUDA event timing
  (paired `cudaEvent_t h2d_begin/h2d_end` per miss, recorded on
  `moe_h2d_stream`). `ssd_read_us` stays CPU-timed. `stall_us` is recomputed
  as `compute_wait_us` — the time the compute stream spent blocked on the
  outstanding miss events — measured via the gap between “compute_begin event
  recorded” and “first kernel post-wait actually starts”. If that precise
  measurement is too costly, fall back to
  `max(0, h2d_end_time - compute_begin_time)` aggregated per layer; document
  the approximation in the CSV column header comment.

Verification:
- A/B vs. previous build at `--moe-cache-vram-mb 99999` (full residency):
  identical output, `stall_us ≈ 0` for every row.
- A/B at `--moe-cache-vram-mb 12000` (streaming, n_slots=145): same generated
  text as the Phase D-4 golden (`Hello\n<think>\nOkay, the user wrote
  "Hello"…`) byte-for-byte for the first 8 tokens at `--temp 0 --seed 42`.
  This is now run as `tests/moe-offload/test-streaming-bytes.ps1`.
- Profiler CSV rows show `compute_us == 0` still (fixed in Phase I) but
  `h2d_us`, `ssd_read_us`, and `stall_us` are all non-zero and internally
  consistent (`h2d_us ≤ ssd_read_us + h2d_us`).

### Phase I — Real `compute_us` via CUDA events

Edits in `src/moe-offload/slot_pool.cpp` and `src/moe-offload/profiler.{h,cpp}`:

- Allocate `2 * n_moe_layers` `cudaEvent_t` pairs (double-buffered) at runtime
  init.
- In the post-topk eval-callback for layer `il`, after the slot_table write
  and the `cudaStreamWaitEvent` block, record `compute_begin[il][parity]`.
- On the *next* MoE layer’s pre-topk callback (`ask=true` for layer `il+1`),
  record `compute_end[il][parity]` first. For the final layer, record
  `compute_end[last]` at the start of `end_request`.
- In `end_request`, drain all pending events with
  `cudaEventElapsedTime` and write `compute_us` back into the matching
  buffered `profile_row`s. Profiler rows are buffered in memory until
  `end_request` so we can patch in the elapsed time before flushing them to
  CSV.

Verification:
- Smoke run at `--pp 8 --tg 2`: every emitted CSV row has `compute_us > 0`,
  and per-layer `sum(compute_us)` over decode tokens is within ±5 % of the
  total decode wall time minus host overhead (sanity check, computed in the
  summary writer).
- Decode tok/s does not regress more than 2 % vs. Phase H (CUDA events are
  effectively free; this is a regression guard, not a feature).

### Phase J — Streaming numeric correctness gate (Plan §7 / §11.3)

Goal: turn “coherent output” into a measured `max|Δlogit|` number.

Edits / new files:
- `tests/moe-offload/test-golden-logits.ps1`: PowerShell harness that
  1. Runs `llama-completion --moe-offload --moe-cache-vram-mb 99999 -ngl 99 …`
     on `Qwen3.5-35B-A3B-Q4_K_M.moe.gguf` with `--temp 0 --seed 42 -n 32` on
     prompt `"The quick brown fox"`, dumping per-step logits via a new
     `--logit-dump PATH` switch on `llama-completion` (small addition: read
     `ctx->logits` after each decode and append `float32` rows to a binary
     file).
  2. Runs the same command at `--moe-cache-vram-mb 4000` (forces eviction)
     into a second dump.
  3. Compares dumps in Python (`tests/moe-offload/compare_logits.py`), prints
     `max|Δlogit|`, `mean|Δlogit|`, exits non-zero if `max|Δlogit| ≥ 1e-3`.
- `tests/moe-offload/README.md`: documents how to run the harness against the
  user’s local model path; CI does **not** run this test (model is too big).

Verification: harness exit code 0 on the dev box. The measured
`max|Δlogit|` is recorded in `docs/moe-offload/README.md` under a new
“Correctness” section.

If the test fails (i.e., streaming logits drift above 1e-3), the failure is
the gating bug, not the harness. Most likely root cause given the rest of the
pipeline is correct: a slot index leaks from a previous batch because
`reset_graph_state()` (already added 2026-05-27 per 20260527_extend.md) is not
called on every rebuild path. Fix in `src/llama-context.cpp`.

### Phase K — Minimal `tests/moe-offload/` test set

New files (all small, all dev-box only — no CI):

- `tests/moe-offload/CMakeLists.txt`: gated by `LLAMA_MOE_OFFLOAD`; registers
  the C++ unit tests below with CTest.
- `tests/moe-offload/test-eamc-cosine.cpp`: synthetic rEAMs, asserts cosine
  ordering, redundancy-replacement on full insert, sidecar round-trip
  (dump → reload → identical state). Covers Plan §9 items 1 and partially the
  EAMC sidecar gap noted in §1.
- `tests/moe-offload/test-lru-eviction.cpp`: deterministic sequence, asserts
  the victim chosen by `lru_predictor::score` matches a hand-computed
  expected order. Covers Plan §9 item 2.
- `tests/moe-offload/test-manifest-roundtrip.cpp`: opens the user’s
  `.moe.gguf`, asserts `moe_offload.version == 2`,
  `expert_blob.table.size() == n_layers * n_experts * 3`, and that every
  `(rel_offset, size)` pair points inside the source tensor’s on-disk byte
  range. Covers Plan §9 item 5 at the level the MVP needs.
- `tests/moe-offload/test-streaming-bytes.ps1` (from Phase H) and
  `tests/moe-offload/test-golden-logits.ps1` (from Phase J): cover Plan §9
  items 6 and 7.

Verification: `ctest -R moe-offload` runs the three C++ tests and they pass
in both Release and Debug builds. The two PowerShell harnesses are documented
in `tests/moe-offload/README.md` but not registered with CTest (they require
the full model).

### Phase L — Documentation truth-up

Edits in `docs/moe-offload/README.md`:

- Replace the “Remaining post-MVP caveats” bullet list with the actual numbers
  measured in Phases H / I / J: TTFT, TPOT, hit rate at 4 GB cache, full
  residency tok/s, measured `max|Δlogit|`, and the SSD bandwidth observed.
- Update the troubleshooting section: `compute_us` is now a real number;
  `stall_us` semantics change (overlap loss, not sync wait) — describe both
  the precise and the fallback formula.
- Keep the streaming `n_ubatch ≤ 8` warning verbatim; cross-link it to a new
  `docs/moe-offload/known-issues.md` that captures the upstream
  `ggml_get_rows` / `mm_ids_helper` interaction in a form that a future
  contributor (per `AGENTS.md` policy: human-authored) can pick up.

---

## 4. Relevant files

- `ggml/src/ggml-cuda/ggml-cuda.cu`,
  `ggml/include/ggml-cuda-moe.h` (new) — stream accessor.
- `src/moe-offload/io.{h,cpp}` — worker queue, no CUDA includes.
- `src/moe-offload/io_cuda.cu` (new) — pinned alloc, async H2D, event pool,
  `cudaStreamWaitEvent`.
- `src/moe-offload/slot_pool.{h,cpp}` — eval-callback rewrite (remove
  double-read), compute-event recording.
- `src/moe-offload/profiler.{h,cpp}` — buffered row patching for `compute_us`;
  no schema change.
- `src/llama-context.cpp` — confirm `reset_graph_state()` on all rebuild paths
  (Phase J fallback).
- `tools/main/llama-completion.cpp` or equivalent — `--logit-dump PATH`
  switch.
- `tests/moe-offload/` (new directory).
- `docs/moe-offload/README.md`, `docs/moe-offload/known-issues.md` (new).

---

## 5. Final acceptance (MVP definition of done)

A run on the dev box (RTX 5070 Ti, NVMe, Qwen3.5-35B-A3B-Q4_K_M) must
demonstrate, in order:

1. `cmake -B build-vanilla-check` (offload OFF) configures, builds, and
   `ctest -R '^(test-arg-parser|test-llama-archs)$'` passes.
2. `cmake -B build-moe -DLLAMA_MOE_OFFLOAD=ON -DGGML_CUDA=ON` builds; no new
   compiler warnings beyond the pre-existing `C4244` in `llama-graph.cpp`.
3. `ctest -R moe-offload` passes the three C++ unit tests.
4. `tests/moe-offload/test-streaming-bytes.ps1` reports identical first-8
   tokens between full-residency and `--moe-cache-vram-mb 12000`.
5. `tests/moe-offload/test-golden-logits.ps1` reports `max|Δlogit| < 1e-3`
   between full-residency and `--moe-cache-vram-mb 4000`, both on
   `Qwen3.5-35B-A3B-Q4_K_M.moe.gguf` with `--temp 0 --seed 42 -n 32` on
   `"The quick brown fox"`. Number is logged into the README.
6. `llama-moe-bench --pp 1024 --tg 256 --repeat 3 --moe-offload
   --moe-cache-vram-mb 8000 --moe-predictor eamc --moe-profile-csv … 
   --moe-profile-summary …`:
   - Produces a stable summary across reps (per-token wall-clock std-dev
     < 5 % of mean).
   - The CSV has non-zero values in every column listed in Plan §4.7(A),
     including `compute_us` and `stall_us`.
   - The summary contains TTFT, TPOT, hit rate (prefill + decode), SSD
     bytes/reads, I/O breakdown, VRAM/DRAM peak — all non-zero — and matches
     the `format_summary(...)` template from
     `src/moe-offload/profiler.cpp`.
7. EAMC sidecar dump appears next to the model as `*.eamc` after a bench run;
   re-running with the same flags reloads it and skips the cold-cache
   warm-up phase (visible as higher decode hit rate on the first repeat
   relative to the previous run's first repeat).
8. `--moe-predictor lru` and `--moe-predictor eamc` produce identical
   generated text at `--temp 0 --seed 42` on the same prompt (predictor
   changes eviction, not outputs — already true today; guarded by the
   PowerShell harness).
9. `docs/moe-offload/README.md` describes how to perform every item above
   from a fresh clone.

If any of 1–8 fails, that failure is the MVP gate. There is no perf floor —
per Plan §11 / §12, MVP acceptance is correctness + measurement, not speed.

---

## 6. Explicitly deferred (do not implement in this milestone)

These were in the 2026-05-27 plan but are deliberately left for a follow-up
milestone so the MVP can land:

- Removing the streaming `n_ubatch <= 8` cap (upstream CUDA work).
- Windows `FILE_FLAG_NO_BUFFERING` direct I/O (buffered fread is sufficient).
- `--moe-oracle` mode (Plan §8) — diagnostic, not on the MVP critical path.
- Multi-thread I/O worker, queue-depth tuning, NUMA pinning.
- Tests §9 items 3 and 4 as standalone GoogleTest binaries (folded into
  integration tests for the MVP).
- CPU DRAM tier, KV-cache offload, FineMoE-style sub-expert splits,
  learned predictors, speculative prefetch — all already deferred to
  Phase 2+ in the original plan.

---

## 7. Risks specific to this closeout

| Risk | Mitigation |
|------|------------|
| `ggml_backend_cuda_get_stream` accessor races with backend stream rotation | The accessor returns `ctx->stream(device, 0)` which is the same stream `ggml-cuda` itself uses for compute; if upstream rotates streams in the future, the accessor returns the active one. Guarded by `LLAMA_MOE_OFFLOAD` so upstream merges remain unaffected. |
| `cudaHostAlloc` failure on Windows under VRAM pressure | Fall back to `malloc` and log once; pipeline still works, just with extra memcpy. |
| Event-pool growth under high miss-rate at small cache | Cap pool at 4 × `n_io_inflight`; recycle aggressively; if exhausted, fall back to sync wait for one miss (rare). |
| `max|Δlogit|` exceeds 1e-3 at 4 GB cache | Almost certainly a `reset_graph_state()` leak on a rebuild path (see Phase J). Fix is local to `llama-context.cpp`. |
| `compute_us` recording adds host-side latency to the eval-callback | Events are recorded but **not** queried in the hot path; `cudaEventElapsedTime` is deferred to `end_request`. Phase I includes the < 2 % regression guard. |

---

## 8. Decisions captured (2026-05-29)

| Decision | Choice |
|----------|--------|
| Closeout strategy | Smallest set of changes that make Plan §11 deliverables 3, 5, 7 honest |
| Async H2D | Real `cudaMemcpyAsync` + `cudaStreamWaitEvent`; new `io_cuda.cu` |
| Pinned staging | `cudaHostAllocPortable`, capacity `n_io_inflight = 32` × `expert_blob_size_max` rounded to 4 KiB |
| `compute_us` | CUDA-event timed, double-buffered, computed in `end_request` |
| Streaming correctness | Automated harness at `--moe-cache-vram-mb 4000`, tolerance `1e-3` |
| Test surface | 3 C++ unit tests + 2 PowerShell harnesses under `tests/moe-offload/` |
| Direct I/O on Windows | Deferred — buffered fread is sufficient |
| Streaming ubatch cap | Kept at 8 — root cause is post-MVP |
| `--moe-oracle` | Deferred per original Plan §12 |
