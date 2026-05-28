# Plan: MoE Offload MVP — Data Plane Delivery
Build the SSD-backed MoE expert offload data plane on top of the existing M0 scaffold for llama.cpp.offload, using repack layout B (keep fused weights in GGUF, add a (layer, expert, kind) → (file_offset, size) table) and remap strategy A (per-layer host sync via ggml_backend_sched eval-callback + slot-indexed VRAM weights + a slot_table graph input feeding ggml_get_rows to rewrite selected_experts). CUDA on, Qwen3.5-35B-A3B Q4_K_M as the acceptance model.

# Phases (each independently verifiable)

Phase A — Repack expert table. Extend tools/moe-repack/main.cpp to compute per (layer, expert, kind) byte offset/size inside the fused tensor data and emit moe_offload.expert_blob.table (u64 pairs) + bump moe_offload.version to 2. Validate by reloading the .moe.gguf in vanilla llama-cli (offload OFF) and confirming logits match the original.
Phase B — Loader: skip expert weights, allocate VRAM slots. Under --moe-offload, mark ffn_{gate,up,down}_exps as TENSOR_NOT_REQUIRED in src/llama-model.cpp, then in src/moe-offload/loader.cpp allocate per-layer [d_in, d_out, n_slots] CUDA tensors (n_slots from --moe-cache-vram-mb/--moe-cache-vram-frac) and bind them back into layer.ffn_*_exps. Acceptance: model loads; VRAM usage matches budget.
Phase C — Real I/O + "all-resident" correctness gate. Implement real direct-I/O in src/moe-offload/platform_io.cpp (Windows FILE_FLAG_NO_BUFFERING|OVERLAPPED), add new src/moe-offload/io.cpp with I/O worker thread + pinned staging + moe_h2d_stream + events. With n_slots == n_expert and a one-shot prefetch-all at configure time, run end-to-end with selected_experts unchanged. Golden-logits test ("The quick brown fox", -n 32, --temp 0 --seed 42) within 1e-3. Correctness gate before Phase D.
Phase D — Slot-indexed remap + per-layer eval callback. In build_moe_ffn (src/llama-graph.cpp around the ffn_moe_topk node) insert a [n_expert] I32 input slot_table.{il} and selected_experts = ggml_get_rows(slot_table, selected_experts) (spike first to confirm ggml_get_rows semantics; otherwise add a small ggml_gather_i32 op). Add new src/moe-offload/scheduler.cpp with a ggml_backend_sched_set_eval_callback that, after each layer's top-k node executes, (1) D2H the indices, (2) moe_cache::begin_layer decides hits/misses + dispatches I/O, (3) ggml_backend_tensor_set writes the slot table, (4) cudaStreamWaitEvent on the compute stream for outstanding misses. Add a guarded ggml_backend_cuda_get_stream accessor in ggml/src/ggml-cuda/ggml-cuda.cu. Validate against Phase C logits at --moe-cache-vram-mb 4000.
Phase E — Profiler + predictor wiring. Drive the per-layer CSV rows from the eval-callback using cudaEventElapsedTime; wire lru_predictor::observe and eamc_predictor::observe/end_request (with .eamc sidecar dump/reload). Confirm lru and eamc produce identical logits.
Phase F — Bench + docs. Finalize tools/moe-bench/main.cpp to print the §4.7(B) summary; update docs/moe-offload/README.md with real instructions and remove the "still pending" caveats.
Relevant files

tools/moe-repack/main.cpp — expert_blob.table emission.
src/llama-model.cpp — suppress expert tensor creation under offload.
src/llama-graph.cpp llm_graph_context::build_moe_ffn — add slot_table input + ggml_get_rows remap.
src/llama-context.cpp — register eval callback per request lifecycle.
src/moe-offload/loader.cpp, cache.cpp, predictor.cpp, platform_io.cpp, profiler.cpp, runtime.cpp — fill out real implementations.
New: src/moe-offload/io.cpp (I/O worker + CUDA H2D stream), src/moe-offload/scheduler.cpp (eval callback).
ggml/src/ggml-cuda/ggml-cuda.cu — guarded ggml_backend_cuda_get_stream accessor.
tools/moe-bench/main.cpp, docs/moe-offload/README.md.
Verification

Default-off build still green; -DLLAMA_MOE_OFFLOAD=ON -DGGML_CUDA=ON configures and builds.
Vanilla llama-cli on the .moe.gguf matches original Qwen3.5-A3B logits exactly.
Phase C: --moe-offload --moe-cache-vram-mb 99999 matches vanilla logits within 1e-3.
Phase D: --moe-cache-vram-mb 4000 matches Phase C logits within 1e-3; stress at 2 GB cache crashes-free.
lru vs eamc produce identical logits.
CSV columns present (token_idx, phase, layer, k_required, k_hit, k_miss, ssd_read_us, h2d_us, compute_us, stall_us, cache_resident_experts, predictor); summary includes TTFT, TPOT, hit_rate, I/O breakdown, VRAM/DRAM/SSD totals.
llama-moe-bench --pp 1024 --tg 256 --repeat 3 produces a stable aggregated summary at --ctx 8192.
Decisions

CUDA build (-DGGML_CUDA=ON); model present at the documented path.
Remap strategy A (host sync per MoE layer), repack layout B (fused weights + index table).
Deferred: --moe-oracle, full GoogleTest suite. Kept: per-layer CSV, EAMC predictor, M6 overlap polish.
Further considerations

ggml_get_rows may not accept an I32 lookup table with an I32 indices tensor directly. Phase D opens with a half-day spike to confirm; if it doesn't, add a minimal ggml_gather_i32 op (CPU + CUDA) rather than reshape gymnastics.
The CUDA compute stream isn't currently exposed by ggml-cuda. We will add a small guarded accessor (ggml_backend_cuda_get_stream) used only by the offload module; this is the smallest viable surface change to upstream files.
Windows direct-I/O requires sector-aligned offsets, sizes, and host pointers. Pinned buffers will be sized to expert_blob_size_max rounded up to 4 KiB, and the repacker's 4 KiB alignment guarantees offsets/sizes; we still query dwBytesPerSector to verify on the target NVMe.


# Plan: MoE Offload MVP — Data Plane Delivery

> **Status 2026-05-27 (late)**: Phases A + B + C GREEN. Phase C added `prefetch_all_experts()` in src/moe-offload/slot_pool.cpp (buffered FILE* + `_fseeki64` + `ggml_backend_tensor_set` slice-by-slice; 30720 reads, 18.12 GiB in 15.6 s = 1.16 GiB/s) hooked in src/llama-model.cpp after `load_all_data` under `LLAMA_MOE_OFFLOAD`. Greedy A/B match: `--moe-offload .moe.gguf -ngl 99` vs vanilla `.gguf -ngl 20` produced IDENTICAL first-32 tokens for prompt "The quick brown fox" (both: `Thinking Process:\n\n1.  **Analyze the Request:** ...`). Argmax equivalence ⇒ correctness gate satisfied. Next: Phase D — slot_table graph input + eval-callback scheduler. Phase A + Phase B green. Phase B implementation: new `src/moe-offload/slot_pool.{h,cpp}` + new public `llama_model_loader::create_unfiled_tensor` and `mark_tensor_unloaded` + intercept hook in `llama_model_base::create_tensor` (src/llama-model.cpp ~line 1573) + `configure_slot_pool()` called from `configure_from_params`. Verified: `configure_slot_pool: 40 layers x 256 experts -> 256 slots/layer`; intercept fires for all `ffn_{gate,up,down}_exps`; `done_getting_tensors` happy (via `mark_tensor_unloaded` decrementing size_data + bumping n_created); `llama_decode` runs without crash; output is garbage-echo as expected (uninitialized slot VRAM). Next: Phase C — real NVMe direct I/O + cache prefill for `n_slots == n_expert` correctness gate (max|Δlogit| < 1e-3). Repack keeps source alignment (32 B); sub-page alignment handled by buffered fallback in Phase C, not direct I/O.

Deliver the full SSD-backed MoE expert offloading MVP for `llama.cpp.offload` on top of the existing M0 control-plane scaffold. Build with CUDA. Target Qwen3.5-35B-A3B Q4_K_M on RTX 5070 Ti 16 GB. Use Layout B (fused weights stay in GGUF, repacker adds per-expert offset table) and Strategy A (per-layer host sync via eval-callback + slot-indexed weights + graph slot_table input + `ggml_get_rows` remap).

## Architecture choices (locked)

- **Repack layout (B)**: leave fused `ffn_*_exps[d_in, d_out, n_expert]` tensors in the GGUF; the repacker pads tensor data to 4 KiB alignment and emits `moe_offload.expert_blob.table` as `u64` array `(file_offset, size)` per (layer, expert). `size` = per-expert slice in bytes within the fused tensor's contiguous expert-axis data.
- **Slot-indexed weights**: when `--moe-offload`, the loader does NOT materialize the fused `ffn_*_exps` tensors as model weights. Instead it allocates a per-layer VRAM tensor `ffn_*_exps_slots[d_in, d_out, n_slots]` of `n_slots < n_expert`. Slot tensor lives in CUDA VRAM, owned by an out-of-band `ggml_backend_buffer` managed by `moe_cache`.
- **Remap mechanism**: replace `selected_experts` in `build_moe_ffn` with `ggml_get_rows(slot_table[il], selected_experts)`. `slot_table[il]` is a graph **input** tensor of shape `[n_expert]` (I32), populated by the scheduler eval-callback right after `argsort_top_k` for that layer.
- **Per-layer host sync**: `ggml_backend_sched_eval_callback` watches for the `argsort_top_k` node of each MoE layer; on `ask=false` (post-eval), it:
  1. `ggml_backend_tensor_get` top-k IDs (D2H, small).
  2. Calls `moe_cache::begin_layer(il, ids)` → cache decides hits/misses, picks slots, fires I/O requests.
  3. Writes the slot_table input tensor (H2D) via `ggml_backend_tensor_set`.
  4. Calls `moe_cache::wait_misses(il)` which `cudaStreamWaitEvent`s the compute stream behind H2D-completion events for missed experts.
- **Storage tier**: NVMe direct I/O. Linux `pread`; Windows `ReadFile` with `FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED`. Pinned host staging via `cudaHostAlloc`. Async H2D on a dedicated `moe_h2d_stream`.
- **Predictors**: keep both `lru` and `eamc` (current scaffold implementations); wire `observe()` from eval-callback and use `score()` for eviction. EAMC sidecar `.eamc` dump/reload.
- **Profiler**: per-layer rows emitted from eval-callback; CUDA `cudaEventElapsedTime` between H2D begin/end and compute begin/end events; CPU `chrono` for SSD read latency.
- **Deferred** (per user): `--moe-oracle` mode; full GoogleTest suite (smoke tests only).

## Phases

### Phase A — Repack: per-expert offset table

Goal: `llama-moe-repack` writes the `moe_offload.expert_blob.table` array so the loader can locate every expert slice.

Edits:
- `tools/moe-repack/main.cpp`: while iterating tensors, for each `ffn_{gate,up,down}_exps[d_in, d_out, n_expert]`, record `(file_offset = dst_data_offset + dst_tensor_offset + e*per_expert_bytes, size = per_expert_bytes)` for every (layer, expert, kind). Concatenate into one flat `u64[]` ordered as `[layer][expert][kind={gate,up,down}]`. Write via `gguf_set_arr_data(dst, "moe_offload.expert_blob.table", GGUF_TYPE_UINT64, ptr, n*2)`. Add `moe_offload.expert_kinds = ["gate","up","down"]` and bump version to 2. Update layout string to `fused-tensors-page-aligned-v1`.
- Manifest JSON gains expert count and per-kind sizes for diagnostics.

Verification:
- `llama-moe-repack` on the real Qwen3.5 GGUF.
- Reload via `gguf_init_from_file` in a smoke check; assert table size == `n_moe_layers * n_experts_per_layer * 3`.
- Stock `llama-cli` (offload OFF) still loads the repacked file and matches vanilla logits on a fixed prompt → confirms file is byte-valid.

### Phase B — Loader: skip expert weights + allocate slot tensors

Goal: when `--moe-offload`, replace original `ffn_*_exps` with VRAM slot tensors and capture the manifest.

Edits:
- `src/llama-model.cpp` `create_tensor` path used by `qwen35moe.cpp` (and other MoE archs): under `LLAMA_MOE_OFFLOAD`, when `llama_moe::runtime_enabled()` is true and tensor name matches `blk.*.ffn_{gate,up,down}_exps`, set `flags |= TENSOR_NOT_REQUIRED` and store the original tensor's shape + dtype + buft on a parallel structure (`moe_layer_binding[il].kind`). Skip the actual load.
- `src/moe-offload/loader.cpp`: after model devices are prepared, allocate one CUDA buffer per (layer, kind) of shape `[d_in, d_out, n_slots]` via `ggml_backend_cuda_buffer_type(device)` + a private `ggml_context`. Compute `n_slots` from VRAM budget (`moe_cache_vram_mb` or `moe_cache_vram_frac * free_mem`) and per-expert byte size. Bind these `ggml_tensor *` back into `layer.ffn_*_exps` so `build_moe_ffn` finds slot-indexed tensors automatically.
- Update `qwen35moe.cpp` only if the binding cannot be done generically; otherwise no change.

Verification:
- Load Qwen3.5 with `--moe-offload --moe-cache-vram-mb 8000`; observe `[moe-offload] slots=NN/layer, expert_size=XX MiB, layers=40`.
- `llama_decode` returns without crashing (data plane is still empty; output will be garbage — expected at this phase).
- Memory snapshot: VRAM usage ≈ non-expert weights + `40 * 3 * n_slots * per_expert_size`.

### Phase C — Platform I/O + cache fill + sync mode ("all resident")

Goal: get correctness first by setting `n_slots == n_expert` and prefetching every expert at load. This validates Phases A+B without the eval-callback complexity.

Edits:
- `src/moe-offload/platform_io.cpp`: real implementation. Windows: `CreateFileW(FILE_FLAG_NO_BUFFERING|FILE_FLAG_OVERLAPPED)`, sector-aligned offset/size handling. Linux: `pread(O_DIRECT)`. Expose `read_at(handle, void* aligned_buf, size_t size, uint64_t offset, completion_event*)`.
- `src/moe-offload/io.cpp` (new): one I/O worker thread (`std::thread`), SPSC queue (`moodycamel` or hand-rolled), pinned host staging pool (`cudaHostAlloc`), dedicated `cudaStream_t moe_h2d_stream`, dedicated `moe_h2d_events` pool. Submit: pread into pinned buf → `cudaMemcpyAsync(slot_ptr, pinned_buf, size, H2D, moe_h2d_stream)` → `cudaEventRecord`.
- `src/moe-offload/cache.cpp` already models slots; extend to call I/O scheduler on miss.
- `src/moe-offload/runtime.cpp::configure_runtime`: if `n_slots >= n_expert`, fire one I/O request per (layer, expert, kind) at startup, wait on all events. Then `remap_selected_experts` returns `selected_experts` unchanged (slot id == expert id by construction).

Verification (correctness gate):
- Golden logits: run vanilla `llama-cli` on `Qwen3.5-35B-A3B-Q4_K_M.gguf`, prompt `"The quick brown fox"`, `-n 32`, `--temp 0`, `--seed 42`. Save logits.
- Run `llama-cli --moe-offload --moe-cache-vram-mb 99999 --moe-predictor lru` on the repacked `.moe.gguf` with the same prompt. `max|Δlogit| < 1e-3`.
- VRAM peak ≈ vanilla (since cache holds everything).

### Phase D — Slot-indexed remap + per-layer eval-callback

Goal: shrink `n_slots < n_expert`, with real cache eviction and on-demand fetch.

Edits:
- `src/llama-graph.cpp build_moe_ffn`: under `LLAMA_MOE_OFFLOAD` and when runtime enabled, after `selected_experts` is produced:
  - Build a per-layer slot_table input tensor: `ggml_tensor * slot_table = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_expert); ggml_set_input(slot_table); ggml_set_name(slot_table, ("moe.slot_table." + il).c_str());`
  - `selected_experts = ggml_get_rows(ctx0, slot_table, selected_experts);` (note: argsort output is `[n_expert_used, n_tokens]` I32; we need a get_rows-equivalent that maps each I32 element through a lookup — verify with existing API, may need a tiny custom op or `ggml_get_rows_back` style — research during Phase D start).
- `src/llama-context.cpp` or `llama_context::decode`: register `ggml_backend_sched_set_eval_callback(sched, moe_eval_callback, ctx)` when offload runtime is enabled.
- New `src/moe-offload/scheduler.cpp`:
  - `bool moe_eval_callback(ggml_tensor * t, bool ask, void * user_data)`.
  - Match nodes by name prefix `ffn_moe_topk.` or `moe.slot_table.`.
  - On `argsort_top_k` post-eval: `ggml_backend_tensor_get_async` (or sync) the I32 top-k, collect unique experts per token batch, call `moe_cache::begin_layer(il, ids)`. For each (kind, miss_expert) enqueue I/O. Compute slot_table[expert]=slot. `ggml_backend_tensor_set(slot_table_tensor, slot_table_host, ...)`. Insert `cudaStreamWaitEvent(compute_stream, ev)` for each pending miss event before returning (or rely on the scheduler's auto-wait if we route slot tensor through ggml event API).
- Connect compute stream wait: easiest is to make `moe_h2d_stream` use the same CUDA context as the backend's stream, then wait via cudaStreamWaitEvent on the backend's compute stream (require exposing the backend stream — add a `ggml_backend_cuda_get_stream(int device)` accessor under `LLAMA_MOE_OFFLOAD` if absent).

Verification:
- Repeat golden-logits test with `--moe-cache-vram-mb 4000` (forces churn). Tolerance still `<1e-3`.
- Stress: 1k prefill + 256 decode at 2 GB cache; no crash, asserts in debug build catch any use-before-event-ready.
- A/B vs `--moe-cache-vram-mb 99999` (Phase C) must agree.

### Phase E — Profiler rows + summary + EAMC wiring

Goal: real CSV/summary output and predictor A/B.

Edits:
- `src/moe-offload/profiler.cpp`: add `record_layer(token_idx, phase, il, stats)` called from eval-callback. Use `cudaEventElapsedTime` between SSD-read-begin / H2D-end / compute-begin / compute-end events.
- `src/moe-offload/runtime.cpp::end_request`: emit summary with cache-hit-rate, I/O breakdown, totals.
- `src/moe-offload/predictor.cpp`: wire `eamc_predictor::observe` from eval-callback; on `end_request` dump EAMC; on `configure_runtime` reload `.eamc` sidecar.
- `tools/moe-bench/main.cpp`: keep current pass-through; ensure summary path is honored.

Verification:
- Run with `--moe-predictor lru` and `--moe-predictor eamc`, same prompt, both produce identical logits (predictor changes eviction, not outputs).
- EAMC run shows hit_rate ≥ LRU on representative prompts after a few warmup runs (informational, not gating).
- CSV columns match plan §4.7(A). Summary keys match §4.7(B).

### Phase F — Bench harness + docs

Goal: `llama-moe-bench` reports the plan §4.7(B) summary; docs updated.

Edits:
- `tools/moe-bench/main.cpp`: confirm `--pp/--tg/--repeat` translation; add summary print after each run; aggregate over repeats.
- `docs/moe-offload/README.md`: update with real numbers, `--moe-predictor eamc` usage, EAMC sidecar location, removal of "still pending" caveats, troubleshooting (CUDA stream wait, direct-I/O alignment).

Verification:
- `llama-moe-bench --model ...moe.gguf --pp 1024 --tg 256 --repeat 3 --moe-offload --moe-cache-vram-mb 8000` produces a stable summary across reps.

## Relevant files

- `tools/moe-repack/main.cpp` — emit expert_blob.table.
- `src/moe-offload/loader.{h,cpp}` — manifest, n_slots calc, slot tensor allocation.
- `src/moe-offload/cache.{h,cpp}` — slot management, miss dispatch, pin/unpin.
- `src/moe-offload/predictor.{h,cpp}` + `eamc` — wire observe/score, sidecar dump.
- `src/moe-offload/platform_io.{h,cpp}` — direct-I/O reads with pinned dest.
- `src/moe-offload/io.{h,cpp}` (new) — worker thread + CUDA H2D stream + events.
- `src/moe-offload/scheduler.{h,cpp}` (new) — `ggml_backend_sched` eval callback.
- `src/moe-offload/profiler.{h,cpp}` — CUDA-event-driven row emission.
- `src/moe-offload/runtime.{h,cpp}` — orchestrate request lifecycle.
- `src/llama-model.cpp` — skip expert tensor materialization under offload; bind slot tensors back.
- `src/llama-graph.cpp` `build_moe_ffn` — insert slot_table input + `ggml_get_rows` remap.
- `src/llama-context.cpp` — register eval callback per request.
- `ggml/src/ggml-cuda/ggml-cuda.cu` — minimal exposure of compute stream and device for offload module (guarded).
- `tools/moe-bench/main.cpp` — summary print.
- `docs/moe-offload/README.md` — finalize.

## Verification (final acceptance)

1. `cmake -B build-moe -DLLAMA_MOE_OFFLOAD=ON -DGGML_CUDA=ON` configures and builds, default-off build remains green.
2. `llama-moe-repack` produces `.moe.gguf` with `moe_offload.version=2` and expert_blob.table sized `n_layers * n_experts * 3`.
3. Vanilla `llama-cli` (offload OFF) on the `.moe.gguf` matches original logits exactly.
4. `llama-cli --moe-offload` with large cache matches vanilla logits within `1e-3` (Phase C gate).
5. `llama-cli --moe-offload --moe-cache-vram-mb 4000` (Phase D) matches Phase C logits within `1e-3`.
6. `--moe-predictor lru` vs `--moe-predictor eamc` produce identical logits.
7. CSV emitted; summary contains TTFT, TPOT, hit_rate, ssd_read/h2d/compute/stall, VRAM/DRAM/SSD-bytes.
8. `llama-moe-bench` runs three reps without OOM at `--ctx 8192` and reports a stable summary.

## Decisions captured

- CUDA build: yes (`-DGGML_CUDA=ON`).
- Model present at `C:\AI\models\qwen\Qwen3.5-35B-A3B-Q4_K_M.gguf`.
- Remap strategy A; repack layout B.
- Deferred: `--moe-oracle`, full GoogleTest suite.
- Kept: per-layer-per-token CSV, EAMC predictor, M6 async overlap.

## Phase A — DONE

- Repacker (`tools/moe-repack/main.cpp`) extended:
  - Records per (layer,expert,kind) `(rel_offset, size)` table; emits `moe_offload.expert_blob.table` (u64 array, layout `(li, e, kind, field) flat`).
  - Emits `moe_offload.layer_ids` (u32), `version=2`, `layout="fused-tensors-page-aligned-v1"`.
  - Bug fixes during validation: `gguf_init_empty` does not populate `ctx->offset` → use `ftell` after metadata write for true data_offset. `"ab"` mode is broken for seek+write on Windows → use `"rb+"`. `std::ftell` is 32-bit on Windows → use `_ftelli64`/`_fseeki64` for files > 2 GB.
- Loader v2 schema (`src/moe-offload/loader.{h,cpp}`):
  - `expert_record { rel_offset, size }`; `manifest` adds `data_offset`, `source_path`, `layer_ids`, flat `experts` vector; methods `at(layer, expert, kind)`, `logical_layer_of(transformer_layer)`.
  - `inspect_manifest(ctx, source_path)` 2-arg signature; caller in `configure_from_params` updated.
- End-to-end verified on Qwen3.5-35B-A3B-Q4_K_M.gguf: ~30 s repack, 40 MoE layers × **256 experts/layer** (note: was assumed 128 — Qwen3.5 actually has 256 experts), expert_blob_size_max = 720896 bytes (~704 KiB) for down (largest per-expert slice), output 20.5 GB.
- Output: `C:\AI\models\qwen\Qwen3.5-35B-A3B-Q4_K_M.moe.gguf` + `.json` manifest.

## Phase A — NOT YET VERIFIED

- Vanilla `llama-cli` (offload OFF) on the `.moe.gguf` must produce same logits as on the original `.gguf`. NOT YET RUN. Build target `llama-cli` (large) and test prompt `"The quick brown fox"`, `-n 32`, `--temp 0 --seed 42`. This validates that the file is byte-readable by llama.cpp's normal loader.

## NEXT ACTIONS

1. Build `llama-cli` in `build-moe`, smoke-load the `.moe.gguf` with offload OFF, verify it generates text without errors. (Phase A correctness gate.)
2. Phase B: replace materialized `ffn_*_exps` with VRAM slot tensors when offload is active.
3. Note: `n_experts_per_layer = 256` (not 128). Update any code that assumes 128.

## Further considerations

1. `ggml_get_rows` may not accept an arbitrary I32 source tensor as the index for a 1-D I32 lookup — if not, we add a minimal `ggml_gather_i32(ctx, table, indices)` op CPU + CUDA implementation, or work around by tile-broadcasting the slot_table to `[n_expert_used, n_tokens]` and using a custom kernel. Confirm in Phase D-0 spike (≤ half day) before writing the rest of Phase D.
2. ggml-cuda doesn't currently expose its compute stream. We will add a guarded `ggml_backend_cuda_get_stream(ggml_backend_t)` accessor used only by the offload module — small surface change in `ggml-cuda.cu`.
3. Windows direct-I/O requires sector-aligned offsets, sizes, AND host pointers. Repacker already 4 KiB-aligns offsets; size and pinned buffer alignment must match `dwBytesPerSector` from `GetDiskFreeSpaceW`. Allocate pinned buffers as `expert_blob_size_max` rounded up to 4 KiB.
4. Eval-callback runs on the compute thread. Long blocking on SSD reads stalls compute. Mitigation: I/O worker thread + cudaStreamWaitEvent so the callback returns immediately after enqueueing wait-on-event; SSD reads overlap with prior-layer GPU compute already on the wire.


---

# Appended from /memories/session/plan.md (2026-05-27 21:11)

# Plan: MoE Offload MVP — Data Plane Delivery

> **Status 2026-05-28 (early)**: Phase D-2 partially debugged. Major fixes landed but the streaming case still crashes in `mmq.cu:206` (mul_mat_q quantize) when `n_uniq` is high even though `n_uniq <= n_slots`. **Verified working:** cache=4000 batch 1 with `n_uniq=16` (all 40 layer callbacks fire, slot writes verified). **Failing:** llama-completion with cache=12000 (n_slots=145, n_uniq=96) crashes at call #0 in first mul_mat_id. Cache=21500 (full residency) generates "Thinking Process: 1.  **" matching Phase C but crashes on a subsequent batch — confirms slot indirection works architecturally.
>
> **NEW FIXES THIS SESSION (committed to source):**
> 1. Added `llama_moe::reset_graph_state()` (slot_pool.{h,cpp}) — clears `topk_to_slot_table`, `topk_to_logical`, `all_slot_tables` maps. Called from `llama-context.cpp` rebuild branch BEFORE `build_graph` so only current graph''s tensor pointers are visible. This made batch 1 work fully for cache=4000.
> 2. Added n_uniq > n_slots `GGML_ABORT` in `moe_eval_callback` (slot_pool.cpp ~line 530) with clear error message instructing user to increase --moe-cache-vram-mb. This catches the architectural constraint: per-batch unique experts must fit in slot pool.
> 3. Added per-call diagnostic log (call #N layer=L n_uniq=NN, limited to 200 calls).
> 4. Added load_expert dump (first 6 loads) showing rec.size/stride/tensor_off/buf_size/data — all sizes match strides (no buffer overflow).
>
> **STILL BROKEN — open hypothesis for next session:**
> - Even with n_uniq=96 ≤ n_slots=145, mul_mat_q crashes at FIRST call (call #0). GPU readback verify confirms slot_table writes are correct (mismatches=0).
> - Crash sig: `CUDA error: an illegal memory access` at `ggml_cuda_mul_mat_q mmq.cu:206 cudaGetLastError()` (with CUDA_LAUNCH_BLOCKING=1).
> - Hypothesis: the per-batch load of MANY experts (96+) might be doing something the simpler n_uniq=16 case doesn''t — maybe `load_expert_into_slot` for slot indices BEYOND ~16 has a bug (e.g., loading via FILE*+fread with offset computation incorrect when slot is far in tensor).
> - **NEXT STEP**: Add diagnostic to `load_expert_into_slot` that prints tensor_off for ALL loads (not just first 6) and verify slots 16-95 have non-zero distinct tensor_off = slot * stride increments correctly. Or test with n_uniq=16 but cache=12000 (force loads to slots 0-15 only) to isolate whether the bug is slot-count-dependent or load-count-dependent.
> - ALTERNATIVE: compute-sanitizer with `--print-limit=100 --show-backtrace=yes` to capture the OOB address pattern.

 Phase C added `prefetch_all_experts()` in src/moe-offload/slot_pool.cpp (buffered FILE* + `_fseeki64` + `ggml_backend_tensor_set` slice-by-slice; 30720 reads, 18.12 GiB in 15.6 s = 1.16 GiB/s) hooked in src/llama-model.cpp after `load_all_data` under `LLAMA_MOE_OFFLOAD`. Greedy A/B match: `--moe-offload .moe.gguf -ngl 99` vs vanilla `.gguf -ngl 20` produced IDENTICAL first-32 tokens for prompt "The quick brown fox" (both: `Thinking Process:\n\n1.  **Analyze the Request:** ...`). Argmax equivalence ⇒ correctness gate satisfied. Next: Phase D — slot_table graph input + eval-callback scheduler. Phase A + Phase B green. Phase B implementation: new `src/moe-offload/slot_pool.{h,cpp}` + new public `llama_model_loader::create_unfiled_tensor` and `mark_tensor_unloaded` + intercept hook in `llama_model_base::create_tensor` (src/llama-model.cpp ~line 1573) + `configure_slot_pool()` called from `configure_from_params`. Verified: `configure_slot_pool: 40 layers x 256 experts -> 256 slots/layer`; intercept fires for all `ffn_{gate,up,down}_exps`; `done_getting_tensors` happy (via `mark_tensor_unloaded` decrementing size_data + bumping n_created); `llama_decode` runs without crash; output is garbage-echo as expected (uninitialized slot VRAM). Next: Phase C — real NVMe direct I/O + cache prefill for `n_slots == n_expert` correctness gate (max|Δlogit| < 1e-3). Repack keeps source alignment (32 B); sub-page alignment handled by buffered fallback in Phase C, not direct I/O.

Deliver the full SSD-backed MoE expert offloading MVP for `llama.cpp.offload` on top of the existing M0 control-plane scaffold. Build with CUDA. Target Qwen3.5-35B-A3B Q4_K_M on RTX 5070 Ti 16 GB. Use Layout B (fused weights stay in GGUF, repacker adds per-expert offset table) and Strategy A (per-layer host sync via eval-callback + slot-indexed weights + graph slot_table input + `ggml_get_rows` remap).

## Architecture choices (locked)

- **Repack layout (B)**: leave fused `ffn_*_exps[d_in, d_out, n_expert]` tensors in the GGUF; the repacker pads tensor data to 4 KiB alignment and emits `moe_offload.expert_blob.table` as `u64` array `(file_offset, size)` per (layer, expert). `size` = per-expert slice in bytes within the fused tensor''s contiguous expert-axis data.
- **Slot-indexed weights**: when `--moe-offload`, the loader does NOT materialize the fused `ffn_*_exps` tensors as model weights. Instead it allocates a per-layer VRAM tensor `ffn_*_exps_slots[d_in, d_out, n_slots]` of `n_slots < n_expert`. Slot tensor lives in CUDA VRAM, owned by an out-of-band `ggml_backend_buffer` managed by `moe_cache`.
- **Remap mechanism**: replace `selected_experts` in `build_moe_ffn` with `ggml_get_rows(slot_table[il], selected_experts)`. `slot_table[il]` is a graph **input** tensor of shape `[n_expert]` (I32), populated by the scheduler eval-callback right after `argsort_top_k` for that layer.
- **Per-layer host sync**: `ggml_backend_sched_eval_callback` watches for the `argsort_top_k` node of each MoE layer; on `ask=false` (post-eval), it:
  1. `ggml_backend_tensor_get` top-k IDs (D2H, small).
  2. Calls `moe_cache::begin_layer(il, ids)` → cache decides hits/misses, picks slots, fires I/O requests.
  3. Writes the slot_table input tensor (H2D) via `ggml_backend_tensor_set`.
  4. Calls `moe_cache::wait_misses(il)` which `cudaStreamWaitEvent`s the compute stream behind H2D-completion events for missed experts.
- **Storage tier**: NVMe direct I/O. Linux `pread`; Windows `ReadFile` with `FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED`. Pinned host staging via `cudaHostAlloc`. Async H2D on a dedicated `moe_h2d_stream`.
- **Predictors**: keep both `lru` and `eamc` (current scaffold implementations); wire `observe()` from eval-callback and use `score()` for eviction. EAMC sidecar `.eamc` dump/reload.
- **Profiler**: per-layer rows emitted from eval-callback; CUDA `cudaEventElapsedTime` between H2D begin/end and compute begin/end events; CPU `chrono` for SSD read latency.
- **Deferred** (per user): `--moe-oracle` mode; full GoogleTest suite (smoke tests only).

## Phases

### Phase A — Repack: per-expert offset table

Goal: `llama-moe-repack` writes the `moe_offload.expert_blob.table` array so the loader can locate every expert slice.

Edits:
- `tools/moe-repack/main.cpp`: while iterating tensors, for each `ffn_{gate,up,down}_exps[d_in, d_out, n_expert]`, record `(file_offset = dst_data_offset + dst_tensor_offset + e*per_expert_bytes, size = per_expert_bytes)` for every (layer, expert, kind). Concatenate into one flat `u64[]` ordered as `[layer][expert][kind={gate,up,down}]`. Write via `gguf_set_arr_data(dst, "moe_offload.expert_blob.table", GGUF_TYPE_UINT64, ptr, n*2)`. Add `moe_offload.expert_kinds = ["gate","up","down"]` and bump version to 2. Update layout string to `fused-tensors-page-aligned-v1`.
- Manifest JSON gains expert count and per-kind sizes for diagnostics.

Verification:
- `llama-moe-repack` on the real Qwen3.5 GGUF.
- Reload via `gguf_init_from_file` in a smoke check; assert table size == `n_moe_layers * n_experts_per_layer * 3`.
- Stock `llama-cli` (offload OFF) still loads the repacked file and matches vanilla logits on a fixed prompt → confirms file is byte-valid.

### Phase B — Loader: skip expert weights + allocate slot tensors

Goal: when `--moe-offload`, replace original `ffn_*_exps` with VRAM slot tensors and capture the manifest.

Edits:
- `src/llama-model.cpp` `create_tensor` path used by `qwen35moe.cpp` (and other MoE archs): under `LLAMA_MOE_OFFLOAD`, when `llama_moe::runtime_enabled()` is true and tensor name matches `blk.*.ffn_{gate,up,down}_exps`, set `flags |= TENSOR_NOT_REQUIRED` and store the original tensor''s shape + dtype + buft on a parallel structure (`moe_layer_binding[il].kind`). Skip the actual load.
- `src/moe-offload/loader.cpp`: after model devices are prepared, allocate one CUDA buffer per (layer, kind) of shape `[d_in, d_out, n_slots]` via `ggml_backend_cuda_buffer_type(device)` + a private `ggml_context`. Compute `n_slots` from VRAM budget (`moe_cache_vram_mb` or `moe_cache_vram_frac * free_mem`) and per-expert byte size. Bind these `ggml_tensor *` back into `layer.ffn_*_exps` so `build_moe_ffn` finds slot-indexed tensors automatically.
- Update `qwen35moe.cpp` only if the binding cannot be done generically; otherwise no change.

Verification:
- Load Qwen3.5 with `--moe-offload --moe-cache-vram-mb 8000`; observe `[moe-offload] slots=NN/layer, expert_size=XX MiB, layers=40`.
- `llama_decode` returns without crashing (data plane is still empty; output will be garbage — expected at this phase).
- Memory snapshot: VRAM usage ≈ non-expert weights + `40 * 3 * n_slots * per_expert_size`.

### Phase C — Platform I/O + cache fill + sync mode ("all resident")

Goal: get correctness first by setting `n_slots == n_expert` and prefetching every expert at load. This validates Phases A+B without the eval-callback complexity.

Edits:
- `src/moe-offload/platform_io.cpp`: real implementation. Windows: `CreateFileW(FILE_FLAG_NO_BUFFERING|FILE_FLAG_OVERLAPPED)`, sector-aligned offset/size handling. Linux: `pread(O_DIRECT)`. Expose `read_at(handle, void* aligned_buf, size_t size, uint64_t offset, completion_event*)`.
- `src/moe-offload/io.cpp` (new): one I/O worker thread (`std::thread`), SPSC queue (`moodycamel` or hand-rolled), pinned host staging pool (`cudaHostAlloc`), dedicated `cudaStream_t moe_h2d_stream`, dedicated `moe_h2d_events` pool. Submit: pread into pinned buf → `cudaMemcpyAsync(slot_ptr, pinned_buf, size, H2D, moe_h2d_stream)` → `cudaEventRecord`.
- `src/moe-offload/cache.cpp` already models slots; extend to call I/O scheduler on miss.
- `src/moe-offload/runtime.cpp::configure_runtime`: if `n_slots >= n_expert`, fire one I/O request per (layer, expert, kind) at startup, wait on all events. Then `remap_selected_experts` returns `selected_experts` unchanged (slot id == expert id by construction).

Verification (correctness gate):
- Golden logits: run vanilla `llama-cli` on `Qwen3.5-35B-A3B-Q4_K_M.gguf`, prompt `"The quick brown fox"`, `-n 32`, `--temp 0`, `--seed 42`. Save logits.
- Run `llama-cli --moe-offload --moe-cache-vram-mb 99999 --moe-predictor lru` on the repacked `.moe.gguf` with the same prompt. `max|Δlogit| < 1e-3`.
- VRAM peak ≈ vanilla (since cache holds everything).

### Phase D — Slot-indexed remap + per-layer eval-callback

Goal: shrink `n_slots < n_expert`, with real cache eviction and on-demand fetch.

Edits:
- `src/llama-graph.cpp build_moe_ffn`: under `LLAMA_MOE_OFFLOAD` and when runtime enabled, after `selected_experts` is produced:
  - Build a per-layer slot_table input tensor: `ggml_tensor * slot_table = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_expert); ggml_set_input(slot_table); ggml_set_name(slot_table, ("moe.slot_table." + il).c_str());`
  - `selected_experts = ggml_get_rows(ctx0, slot_table, selected_experts);` (note: argsort output is `[n_expert_used, n_tokens]` I32; we need a get_rows-equivalent that maps each I32 element through a lookup — verify with existing API, may need a tiny custom op or `ggml_get_rows_back` style — research during Phase D start).
- `src/llama-context.cpp` or `llama_context::decode`: register `ggml_backend_sched_set_eval_callback(sched, moe_eval_callback, ctx)` when offload runtime is enabled.
- New `src/moe-offload/scheduler.cpp`:
  - `bool moe_eval_callback(ggml_tensor * t, bool ask, void * user_data)`.
  - Match nodes by name prefix `ffn_moe_topk.` or `moe.slot_table.`.
  - On `argsort_top_k` post-eval: `ggml_backend_tensor_get_async` (or sync) the I32 top-k, collect unique experts per token batch, call `moe_cache::begin_layer(il, ids)`. For each (kind, miss_expert) enqueue I/O. Compute slot_table[expert]=slot. `ggml_backend_tensor_set(slot_table_tensor, slot_table_host, ...)`. Insert `cudaStreamWaitEvent(compute_stream, ev)` for each pending miss event before returning (or rely on the scheduler''s auto-wait if we route slot tensor through ggml event API).
- Connect compute stream wait: easiest is to make `moe_h2d_stream` use the same CUDA context as the backend''s stream, then wait via cudaStreamWaitEvent on the backend''s compute stream (require exposing the backend stream — add a `ggml_backend_cuda_get_stream(int device)` accessor under `LLAMA_MOE_OFFLOAD` if absent).

Verification:
- Repeat golden-logits test with `--moe-cache-vram-mb 4000` (forces churn). Tolerance still `<1e-3`.
- Stress: 1k prefill + 256 decode at 2 GB cache; no crash, asserts in debug build catch any use-before-event-ready.
- A/B vs `--moe-cache-vram-mb 99999` (Phase C) must agree.

### Phase E — Profiler rows + summary + EAMC wiring

Goal: real CSV/summary output and predictor A/B.

Edits:
- `src/moe-offload/profiler.cpp`: add `record_layer(token_idx, phase, il, stats)` called from eval-callback. Use `cudaEventElapsedTime` between SSD-read-begin / H2D-end / compute-begin / compute-end events.
- `src/moe-offload/runtime.cpp::end_request`: emit summary with cache-hit-rate, I/O breakdown, totals.
- `src/moe-offload/predictor.cpp`: wire `eamc_predictor::observe` from eval-callback; on `end_request` dump EAMC; on `configure_runtime` reload `.eamc` sidecar.
- `tools/moe-bench/main.cpp`: keep current pass-through; ensure summary path is honored.

Verification:
- Run with `--moe-predictor lru` and `--moe-predictor eamc`, same prompt, both produce identical logits (predictor changes eviction, not outputs).
- EAMC run shows hit_rate ≥ LRU on representative prompts after a few warmup runs (informational, not gating).
- CSV columns match plan §4.7(A). Summary keys match §4.7(B).

### Phase F — Bench harness + docs

Goal: `llama-moe-bench` reports the plan §4.7(B) summary; docs updated.

Edits:
- `tools/moe-bench/main.cpp`: confirm `--pp/--tg/--repeat` translation; add summary print after each run; aggregate over repeats.
- `docs/moe-offload/README.md`: update with real numbers, `--moe-predictor eamc` usage, EAMC sidecar location, removal of "still pending" caveats, troubleshooting (CUDA stream wait, direct-I/O alignment).

Verification:
- `llama-moe-bench --model ...moe.gguf --pp 1024 --tg 256 --repeat 3 --moe-offload --moe-cache-vram-mb 8000` produces a stable summary across reps.

## Relevant files

- `tools/moe-repack/main.cpp` — emit expert_blob.table.
- `src/moe-offload/loader.{h,cpp}` — manifest, n_slots calc, slot tensor allocation.
- `src/moe-offload/cache.{h,cpp}` — slot management, miss dispatch, pin/unpin.
- `src/moe-offload/predictor.{h,cpp}` + `eamc` — wire observe/score, sidecar dump.
- `src/moe-offload/platform_io.{h,cpp}` — direct-I/O reads with pinned dest.
- `src/moe-offload/io.{h,cpp}` (new) — worker thread + CUDA H2D stream + events.
- `src/moe-offload/scheduler.{h,cpp}` (new) — `ggml_backend_sched` eval callback.
- `src/moe-offload/profiler.{h,cpp}` — CUDA-event-driven row emission.
- `src/moe-offload/runtime.{h,cpp}` — orchestrate request lifecycle.
- `src/llama-model.cpp` — skip expert tensor materialization under offload; bind slot tensors back.
- `src/llama-graph.cpp` `build_moe_ffn` — insert slot_table input + `ggml_get_rows` remap.
- `src/llama-context.cpp` — register eval callback per request.
- `ggml/src/ggml-cuda/ggml-cuda.cu` — minimal exposure of compute stream and device for offload module (guarded).
- `tools/moe-bench/main.cpp` — summary print.
- `docs/moe-offload/README.md` — finalize.

## Verification (final acceptance)

1. `cmake -B build-moe -DLLAMA_MOE_OFFLOAD=ON -DGGML_CUDA=ON` configures and builds, default-off build remains green.
2. `llama-moe-repack` produces `.moe.gguf` with `moe_offload.version=2` and expert_blob.table sized `n_layers * n_experts * 3`.
3. Vanilla `llama-cli` (offload OFF) on the `.moe.gguf` matches original logits exactly.
4. `llama-cli --moe-offload` with large cache matches vanilla logits within `1e-3` (Phase C gate).
5. `llama-cli --moe-offload --moe-cache-vram-mb 4000` (Phase D) matches Phase C logits within `1e-3`.
6. `--moe-predictor lru` vs `--moe-predictor eamc` produce identical logits.
7. CSV emitted; summary contains TTFT, TPOT, hit_rate, ssd_read/h2d/compute/stall, VRAM/DRAM/SSD-bytes.
8. `llama-moe-bench` runs three reps without OOM at `--ctx 8192` and reports a stable summary.

## Decisions captured

- CUDA build: yes (`-DGGML_CUDA=ON`).
- Model present at `C:\AI\models\qwen\Qwen3.5-35B-A3B-Q4_K_M.gguf`.
- Remap strategy A; repack layout B.
- Deferred: `--moe-oracle`, full GoogleTest suite.
- Kept: per-layer-per-token CSV, EAMC predictor, M6 async overlap.

## Phase A — DONE

- Repacker (`tools/moe-repack/main.cpp`) extended:
  - Records per (layer,expert,kind) `(rel_offset, size)` table; emits `moe_offload.expert_blob.table` (u64 array, layout `(li, e, kind, field) flat`).
  - Emits `moe_offload.layer_ids` (u32), `version=2`, `layout="fused-tensors-page-aligned-v1"`.
  - Bug fixes during validation: `gguf_init_empty` does not populate `ctx->offset` → use `ftell` after metadata write for true data_offset. `"ab"` mode is broken for seek+write on Windows → use `"rb+"`. `std::ftell` is 32-bit on Windows → use `_ftelli64`/`_fseeki64` for files > 2 GB.
- Loader v2 schema (`src/moe-offload/loader.{h,cpp}`):
  - `expert_record { rel_offset, size }`; `manifest` adds `data_offset`, `source_path`, `layer_ids`, flat `experts` vector; methods `at(layer, expert, kind)`, `logical_layer_of(transformer_layer)`.
  - `inspect_manifest(ctx, source_path)` 2-arg signature; caller in `configure_from_params` updated.
- End-to-end verified on Qwen3.5-35B-A3B-Q4_K_M.gguf: ~30 s repack, 40 MoE layers × **256 experts/layer** (note: was assumed 128 — Qwen3.5 actually has 256 experts), expert_blob_size_max = 720896 bytes (~704 KiB) for down (largest per-expert slice), output 20.5 GB.
- Output: `C:\AI\models\qwen\Qwen3.5-35B-A3B-Q4_K_M.moe.gguf` + `.json` manifest.

## Phase A — NOT YET VERIFIED

- Vanilla `llama-cli` (offload OFF) on the `.moe.gguf` must produce same logits as on the original `.gguf`. NOT YET RUN. Build target `llama-cli` (large) and test prompt `"The quick brown fox"`, `-n 32`, `--temp 0 --seed 42`. This validates that the file is byte-readable by llama.cpp''s normal loader.

## NEXT ACTIONS

1. Build `llama-cli` in `build-moe`, smoke-load the `.moe.gguf` with offload OFF, verify it generates text without errors. (Phase A correctness gate.)
2. Phase B: replace materialized `ffn_*_exps` with VRAM slot tensors when offload is active.
3. Note: `n_experts_per_layer = 256` (not 128). Update any code that assumes 128.

## Further considerations

1. `ggml_get_rows` may not accept an arbitrary I32 source tensor as the index for a 1-D I32 lookup — if not, we add a minimal `ggml_gather_i32(ctx, table, indices)` op CPU + CUDA implementation, or work around by tile-broadcasting the slot_table to `[n_expert_used, n_tokens]` and using a custom kernel. Confirm in Phase D-0 spike (≤ half day) before writing the rest of Phase D.
2. ggml-cuda doesn''t currently expose its compute stream. We will add a guarded `ggml_backend_cuda_get_stream(ggml_backend_t)` accessor used only by the offload module — small surface change in `ggml-cuda.cu`.
3. Windows direct-I/O requires sector-aligned offsets, sizes, AND host pointers. Repacker already 4 KiB-aligns offsets; size and pinned buffer alignment must match `dwBytesPerSector` from `GetDiskFreeSpaceW`. Allocate pinned buffers as `expert_blob_size_max` rounded up to 4 KiB.
4. Eval-callback runs on the compute thread. Long blocking on SSD reads stalls compute. Mitigation: I/O worker thread + cudaStreamWaitEvent so the callback returns immediately after enqueueing wait-on-event; SSD reads overlap with prior-layer GPU compute already on the wire.


# Plan: MoE Offload MVP — Finish D, then E + F

## TL;DR
Phases A+B+C are GREEN. Phase D-1 (graph remap + per-build slot_table registration) lands; Phase D-2 (streaming eval-callback) is implemented but crashes in `mmq.cu:206` when per-batch `n_uniq` is high (96) even though `n_uniq <= n_slots`. Finish D by instrumenting first, fixing the crash, swapping the in-callback sync `fread`+`tensor_set` for the planned async I/O pipeline (worker thread + pinned staging + `moe_h2d_stream` + `cudaStreamWaitEvent`), then wire profiler/CSV/summary and the EAMC predictor, then finalize `llama-moe-bench` + docs.

## Phase D — finish streaming (blocking)

### D-3 (debug): isolate the mmq.cu:206 crash
1. In src/moe-offload/slot_pool.cpp `load_expert_into_slot`, drop the `dump<6` gate; emit one log line per load with `L,e,k,slot, rec.size, stride, tensor_off, write_off, buf_name=ggml_backend_buffer_name(slot_tensor->buffer)`. Cap at ~3000 lines via a global counter.
2. In `moe_eval_callback`, after writing slot_table, GPU-read it back for **every** layer (not only logical==0 && topk_calls==0), compare to host buffer, log mismatches and the first 8 slot indices.
3. Run `llama-cli --moe-offload --moe-cache-vram-mb 12000` on the .moe.gguf with the standard golden prompt. Read the trace; verify (a) every slot_tensor buffer is `CUDA0` (not split/CPU), (b) `tensor_off + write_off + rec.size <= buf_size` holds for every load, (c) slot_table readback equals what was written for every layer, (d) all slot indices written are `< n_slots`.
4. If all 4 hold → the crash is NOT in our load/write path; suspect graph-side. Inspect with `CUDA_LAUNCH_BLOCKING=1` + `compute-sanitizer --tool memcheck --print-limit 50` to capture the OOB address, then compare to the slot tensor base/extent for the failing layer.
5. Most likely root cause to verify: the slot_table tensor needs `ggml_set_input` removed (already done per code review) AND it must NOT be subject to graph-reuse aliasing across builds. Confirm by adding a unique-name suffix per build counter so the scheduler can never alias a stale tensor pointer.

### D-4 (fix): correct the root cause
6. Apply the minimal fix from D-3 (one of: load-offset bug, slot_table aliasing, or graph-reuse). Add a one-line assertion in `moe_eval_callback` that `lc.exp2slot[e] < s.n_slots` before writing slot_table. Re-run golden test at `--moe-cache-vram-mb 12000`; assert first-32 tokens match Phase C output.
7. Stress: rerun at `--moe-cache-vram-mb 4000` (forces frequent eviction). First-32 tokens must still match. 1k-prefill / 256-decode run must not crash.

### D-5 (async I/O pipeline) — per user choice "full async"
8. Create new src/moe-offload/io.{h,cpp}:
   - One `std::thread` I/O worker; SPSC ring queue of `io_request { layer, expert, kind, slot, pinned_buf, ev_done }`.
   - Pinned host staging pool: ring of N buffers, each `expert_blob_size_max` rounded up to 4 KiB, allocated via `cudaHostAlloc(cudaHostAllocDefault)`. Size N = max(2 * n_slots, 64) so the callback can issue an entire layer's misses before blocking.
   - One CUDA `moe_h2d_stream` (per device), created lazily by querying device id from a slot_tensor's buffer.
   - Worker loop: pop request → `fread` from buffered FILE* (Phase F may upgrade to `ReadFile(FILE_FLAG_NO_BUFFERING|OVERLAPPED)`; not required now) → `cudaMemcpyAsync(slot.data + slot_idx*stride, pinned, size, H2D, moe_h2d_stream)` → `cudaEventRecord(ev_done, moe_h2d_stream)` → push back to a free-list.
9. Expose `ggml_backend_cuda_get_stream(ggml_backend_t)` in ggml/src/ggml-cuda/ggml-cuda.cu under `#ifdef LLAMA_MOE_OFFLOAD` (or unconditional accessor — minimal, additive). Plumb that backend handle from `llama_context` into `moe_h2d` init.
10. Rewrite `moe_eval_callback` miss-path: enqueue 3 I/O requests per missed expert (gate/up/down), keep their `ev_done` handles in `layer_cache::pending_evs`. After the slot_table write, `cudaStreamWaitEvent(compute_stream, ev, 0)` for each pending event, then clear `pending_evs`. (Compute stream is queried via the new accessor.)
11. Initial residency (Phase C path) still uses the simpler buffered loader so startup behavior is unchanged.

### D verification
- `--moe-cache-vram-mb 99999` (full residency, streaming OFF) matches vanilla logits within 1e-3 (Phase C gate already verified).
- `--moe-cache-vram-mb 12000` and `--moe-cache-vram-mb 4000` (streaming) match Phase C output exactly on the golden prompt (greedy argmax).
- A/B at `--moe-cache-vram-mb 4000` with sync I/O (toggle via env var `MOE_OFFLOAD_SYNC_IO=1`) vs async produces identical logits.
- No CUDA error / OOB through 1k prefill + 256 decode at `--moe-cache-vram-mb 4000`.

## Phase E — Profiler rows + summary + EAMC wiring

### E-1: profiler
12. Per-layer event timing in `moe_eval_callback`:
    - Record CPU-time deltas around the synchronous `fread` (ssd_read_us) and around the eval-callback enqueue (h2d_us_async, end → cudaEventSynchronize before write_off bookkeeping; for fully-async D-5, use `cudaEventElapsedTime` between submit-event and done-event recorded by the I/O worker).
    - For compute_us: record `cudaEventRecord` on `compute_stream` just before returning from the callback (start) and on the topk node of the NEXT layer (end). Approximation: aggregate per-token via a per-layer running deltas.
12b. In src/moe-offload/profiler.cpp wire `record_layer(token_idx, phase, il, stats)` exposed via a new `record_layer` API; called from `moe_eval_callback`. Stats include: `k_required, k_hit, k_miss, ssd_read_us, h2d_us, compute_us, stall_us, cache_resident_experts, predictor`.
13. Open CSV at first record_layer using `runtime_options::profile_csv` path; emit header per plan §4.7(A). Flush per row to survive crashes during early testing.

### E-2: summary
14. In src/moe-offload/runtime.cpp::end_request, aggregate counters from slot_pool + profiler into the §4.7(B) summary (TTFT, TPOT, hit_rate, ssd_read_us_total, h2d_us_total, compute_us_total, stall_us_total, VRAM/DRAM/SSD bytes). Print to stderr and write JSON sidecar.

### E-3: predictor wiring (per user: full)
15. Refactor `slot_pool.cpp` victim selection: replace inline `lc.lru` with a call to `predictor.score(layer, candidates)` exposed via a tiny interface (the existing `predictor.h` already has `lru_predictor` and `eamc_predictor`). Slot_pool calls `predictor.observe(layer, expert_id)` for each unique selected expert in the callback; calls `predictor.score(layer, candidates)` when picking an eviction victim.
16. Hook `predictor.end_request()` from `llama_moe::end_request` — for `eamc`, dumps the `.eamc` sidecar next to the `.moe.gguf`. On `configure_from_params`, if `.eamc` exists, load it.

### E verification
- `--moe-predictor lru` and `--moe-predictor eamc` produce identical logits on the golden prompt (predictor only changes eviction).
- CSV file contains all columns from plan §4.7(A) with at least n_layers rows per generated token.
- Summary contains all keys from §4.7(B); JSON sidecar parses.
- EAMC run produces `.eamc` file; reloading on a second invocation skips re-warmup.

## Phase F — Bench + docs

17. In tools/moe-bench/main.cpp finalize: parse `--pp`, `--tg`, `--repeat`; per repeat call `begin_request → llama_decode(prefill) → loop llama_decode(decode) → end_request`; aggregate summaries across repeats (mean + stddev for headline numbers).
18. Update docs/moe-offload/README.md: remove "still pending" caveats; concrete invocation lines for repack / cli / bench; troubleshooting (slot vs n_uniq budget; sector alignment); EAMC sidecar location.

### F verification
- `llama-moe-bench --model ...moe.gguf --pp 1024 --tg 256 --repeat 3 --moe-offload --moe-cache-vram-mb 8000` runs without OOM at `--ctx 8192` and prints a stable aggregated summary across repeats.
- `cmake -B build-moe -DLLAMA_MOE_OFFLOAD=ON -DGGML_CUDA=ON` and default-off (`-DLLAMA_MOE_OFFLOAD=OFF`) both compile clean.

## Relevant files

- `src/moe-offload/slot_pool.{h,cpp}` — D-3 instrumentation; D-4 fix; D-5 callback refactor to use io.h; E-3 predictor wiring; E-1 record_layer.
- `src/moe-offload/io.{h,cpp}` (NEW, Phase D-5) — worker thread, pinned staging pool, `moe_h2d_stream`, events.
- `src/moe-offload/cache.{h,cpp}` — currently dormant; either delete or fold its LRU into the predictor interface in E-3.
- `src/moe-offload/predictor.{h,cpp}` (+ `eamc_predictor`) — wire `observe/score/end_request`; sidecar dump/reload.
- `src/moe-offload/profiler.{h,cpp}` — `record_layer` API + CSV emission; summary aggregation.
- `src/moe-offload/runtime.{h,cpp}` — `end_request` summary; `configure_from_params` reloads `.eamc`.
- `src/moe-offload/loader.cpp` — none expected after D.
- `src/llama-context.cpp` (≈ line 1285) — currently registers callback only when `streaming_mode()`; keep, but also register a profiler-only callback in non-streaming so E-1 still records rows.
- `ggml/src/ggml-cuda/ggml-cuda.cu` — add `ggml_backend_cuda_get_stream(ggml_backend_t)` (Phase D-5).
- `tools/moe-bench/main.cpp` — Phase F orchestrator.
- `docs/moe-offload/README.md` — Phase F.

## Decisions

- I/O: full async (worker + pinned + `moe_h2d_stream` + `cudaStreamWaitEvent`); keep buffered FILE* read (no Win32 direct I/O yet — deferred to a future polish).
- Predictor: full wiring with EAMC sidecar.
- Debug-first for the mmq crash: instrument all loads + every-layer slot_table readback; only escalate to compute-sanitizer if instrumentation is inconclusive.
- Keep golden-prompt greedy argmax as the regression test (already proven for Phase C); no full GoogleTest suite.
- Direct-I/O (`FILE_FLAG_NO_BUFFERING|OVERLAPPED`) and `--moe-oracle` remain deferred.

## Further considerations
1. If D-3 instrumentation shows the slot_table tensor still ends up in a non-`CUDA0` buffer for some graph builds (split / probe / reuse), the fix is to attach the slot_table to the existing topk tensor's backend via a no-op `ggml_view` so the scheduler places it on the same backend as the consumer.
2. The compute-stream accessor in `ggml-cuda.cu` is a small upstream-touching change; keep it `#ifdef LLAMA_MOE_OFFLOAD`-guarded to avoid surface drift on the default build.
3. `record_layer` timing for "compute_us" can over-count if the next layer's MoE topk runs before this layer's mul_mat_id finishes (CUDA stream pipelining). For MVP, document this as "approximate compute_us" and prefer cudaEvent-based measurement of just the SSD-read + H2D phases for the headline numbers.


# Plan: MoE Offload MVP — Finish D, then E + F

## TL;DR
Phases A+B+C are GREEN. Phase D-1 (graph remap + per-build slot_table registration) lands; Phase D-2 (streaming eval-callback) is implemented but crashes in `mmq.cu:206` when per-batch `n_uniq` is high (96) even though `n_uniq <= n_slots`. Finish D by instrumenting first, fixing the crash, swapping the in-callback sync `fread`+`tensor_set` for the planned async I/O pipeline (worker thread + pinned staging + `moe_h2d_stream` + `cudaStreamWaitEvent`), then wire profiler/CSV/summary and the EAMC predictor, then finalize `llama-moe-bench` + docs.

## Phase D — finish streaming (blocking)

### D-3 STATUS (2026-05-27 late night):
- D-3 instrumentation completed in [src/moe-offload/slot_pool.cpp](c:\code\llama.cpp.offload\src\moe-offload\slot_pool.cpp): fprintf(stderr) for every load + per-layer GPU readback.
- Diagnostic runs at cache=12000 (n_slots=145):
  * default -ub (512): CRASH at mmq.cu:206 after layer 0 readback. n_uniq=96. All loads correct (CUDA0, offsets, strides). slot_table readback mismatches=0.
  * `-ub 1`: WORKS. 40 layers readback OK. Output "Okay,".
  * `-ub 2`: WORKS. Output "Thinking Process".
  * `-ub 4`: WORKS. 40 layers readback OK.
  * `-ub 8 `: TBD (slow with CUDA_LAUNCH_BLOCKING=1).
- Conclusion: crash threshold tied to PER-MICROBATCH n_uniq. ub=4 → 4 tokens × top8 = ≤32 unique expert usage per layer; ub=12+ at default ub → 96 unique. Bug is downstream of our slot table — likely in mul_mat_id kernel when ne02 (slot dim) is 145 and many unique slots are active in one batch.
- LIKELY ROOT CAUSE: mmq's expert_bounds tracking expects `ne02 == n_experts` to match unique counts well. With sparse usage of 96 of 145 slots, some kernel parameter exceeds a limit, or `ne12` (n_tokens) interaction with `n_expert_used` creates >some bound.
- NEXT FIX HYPOTHESIS: try making slot tensor's n_slots dimension snap up to a power-of-2 OR ensure n_slots ≥ n_expert when streaming. Simpler: lower the per-microbatch token budget so n_uniq stays small (via `-ub` cap) — but this kills prefill throughput.
- ALTERNATIVE FIX: make remap_selected_experts also remap THE EXPERT AXIS of slot tensor presentation. Currently slot_tensor->ne[2]=n_slots=145 is passed to mul_mat_id which treats it as the "n_experts" dimension. The graph thinks model has 145 experts, but topk produced ids in [0..255] → remapped to [0..144]. If mul_mat_id's mmq kernel uses ne02 internally in any per-expert allocation sized in a way that interacts with ne12=12 (tokens) × n_expert_used=8 = 96 = n_uniq, that's the cliff.
- Logs at: c:\code\llama.cpp.offload\build-moe\phaseD3-*.log

### D-3 (debug): isolate the mmq.cu:206 crash
1. In src/moe-offload/slot_pool.cpp `load_expert_into_slot`, drop the `dump<6` gate; emit one log line per load with `L,e,k,slot, rec.size, stride, tensor_off, write_off, buf_name=ggml_backend_buffer_name(slot_tensor->buffer)`. Cap at ~3000 lines via a global counter.
2. In `moe_eval_callback`, after writing slot_table, GPU-read it back for **every** layer (not only logical==0 && topk_calls==0), compare to host buffer, log mismatches and the first 8 slot indices.
3. Run `llama-cli --moe-offload --moe-cache-vram-mb 12000` on the .moe.gguf with the standard golden prompt. Read the trace; verify (a) every slot_tensor buffer is `CUDA0` (not split/CPU), (b) `tensor_off + write_off + rec.size <= buf_size` holds for every load, (c) slot_table readback equals what was written for every layer, (d) all slot indices written are `< n_slots`.
4. If all 4 hold → the crash is NOT in our load/write path; suspect graph-side. Inspect with `CUDA_LAUNCH_BLOCKING=1` + `compute-sanitizer --tool memcheck --print-limit 50` to capture the OOB address, then compare to the slot tensor base/extent for the failing layer.
5. Most likely root cause to verify: the slot_table tensor needs `ggml_set_input` removed (already done per code review) AND it must NOT be subject to graph-reuse aliasing across builds. Confirm by adding a unique-name suffix per build counter so the scheduler can never alias a stale tensor pointer.

### D-4 (fix): correct the root cause
6. Apply the minimal fix from D-3 (one of: load-offset bug, slot_table aliasing, or graph-reuse). Add a one-line assertion in `moe_eval_callback` that `lc.exp2slot[e] < s.n_slots` before writing slot_table. Re-run golden test at `--moe-cache-vram-mb 12000`; assert first-32 tokens match Phase C output.
7. Stress: rerun at `--moe-cache-vram-mb 4000` (forces frequent eviction). First-32 tokens must still match. 1k-prefill / 256-decode run must not crash.

### D-5 (async I/O pipeline) — per user choice "full async"
8. Create new src/moe-offload/io.{h,cpp}:
   - One `std::thread` I/O worker; SPSC ring queue of `io_request { layer, expert, kind, slot, pinned_buf, ev_done }`.
   - Pinned host staging pool: ring of N buffers, each `expert_blob_size_max` rounded up to 4 KiB, allocated via `cudaHostAlloc(cudaHostAllocDefault)`. Size N = max(2 * n_slots, 64) so the callback can issue an entire layer's misses before blocking.
   - One CUDA `moe_h2d_stream` (per device), created lazily by querying device id from a slot_tensor's buffer.
   - Worker loop: pop request → `fread` from buffered FILE* (Phase F may upgrade to `ReadFile(FILE_FLAG_NO_BUFFERING|OVERLAPPED)`; not required now) → `cudaMemcpyAsync(slot.data + slot_idx*stride, pinned, size, H2D, moe_h2d_stream)` → `cudaEventRecord(ev_done, moe_h2d_stream)` → push back to a free-list.
9. Expose `ggml_backend_cuda_get_stream(ggml_backend_t)` in ggml/src/ggml-cuda/ggml-cuda.cu under `#ifdef LLAMA_MOE_OFFLOAD` (or unconditional accessor — minimal, additive). Plumb that backend handle from `llama_context` into `moe_h2d` init.
10. Rewrite `moe_eval_callback` miss-path: enqueue 3 I/O requests per missed expert (gate/up/down), keep their `ev_done` handles in `layer_cache::pending_evs`. After the slot_table write, `cudaStreamWaitEvent(compute_stream, ev, 0)` for each pending event, then clear `pending_evs`. (Compute stream is queried via the new accessor.)
11. Initial residency (Phase C path) still uses the simpler buffered loader so startup behavior is unchanged.

### D verification
- `--moe-cache-vram-mb 99999` (full residency, streaming OFF) matches vanilla logits within 1e-3 (Phase C gate already verified).
- `--moe-cache-vram-mb 12000` and `--moe-cache-vram-mb 4000` (streaming) match Phase C output exactly on the golden prompt (greedy argmax).
- A/B at `--moe-cache-vram-mb 4000` with sync I/O (toggle via env var `MOE_OFFLOAD_SYNC_IO=1`) vs async produces identical logits.
- No CUDA error / OOB through 1k prefill + 256 decode at `--moe-cache-vram-mb 4000`.

## Phase E — Profiler rows + summary + EAMC wiring

### E-1: profiler
12. Per-layer event timing in `moe_eval_callback`:
    - Record CPU-time deltas around the synchronous `fread` (ssd_read_us) and around the eval-callback enqueue (h2d_us_async, end → cudaEventSynchronize before write_off bookkeeping; for fully-async D-5, use `cudaEventElapsedTime` between submit-event and done-event recorded by the I/O worker).
    - For compute_us: record `cudaEventRecord` on `compute_stream` just before returning from the callback (start) and on the topk node of the NEXT layer (end). Approximation: aggregate per-token via a per-layer running deltas.
12b. In src/moe-offload/profiler.cpp wire `record_layer(token_idx, phase, il, stats)` exposed via a new `record_layer` API; called from `moe_eval_callback`. Stats include: `k_required, k_hit, k_miss, ssd_read_us, h2d_us, compute_us, stall_us, cache_resident_experts, predictor`.
13. Open CSV at first record_layer using `runtime_options::profile_csv` path; emit header per plan §4.7(A). Flush per row to survive crashes during early testing.

### E-2: summary
14. In src/moe-offload/runtime.cpp::end_request, aggregate counters from slot_pool + profiler into the §4.7(B) summary (TTFT, TPOT, hit_rate, ssd_read_us_total, h2d_us_total, compute_us_total, stall_us_total, VRAM/DRAM/SSD bytes). Print to stderr and write JSON sidecar.

### E-3: predictor wiring (per user: full)
15. Refactor `slot_pool.cpp` victim selection: replace inline `lc.lru` with a call to `predictor.score(layer, candidates)` exposed via a tiny interface (the existing `predictor.h` already has `lru_predictor` and `eamc_predictor`). Slot_pool calls `predictor.observe(layer, expert_id)` for each unique selected expert in the callback; calls `predictor.score(layer, candidates)` when picking an eviction victim.
16. Hook `predictor.end_request()` from `llama_moe::end_request` — for `eamc`, dumps the `.eamc` sidecar next to the `.moe.gguf`. On `configure_from_params`, if `.eamc` exists, load it.

### E verification
- `--moe-predictor lru` and `--moe-predictor eamc` produce identical logits on the golden prompt (predictor only changes eviction).
- CSV file contains all columns from plan §4.7(A) with at least n_layers rows per generated token.
- Summary contains all keys from §4.7(B); JSON sidecar parses.
- EAMC run produces `.eamc` file; reloading on a second invocation skips re-warmup.

## Phase F — Bench + docs

17. In tools/moe-bench/main.cpp finalize: parse `--pp`, `--tg`, `--repeat`; per repeat call `begin_request → llama_decode(prefill) → loop llama_decode(decode) → end_request`; aggregate summaries across repeats (mean + stddev for headline numbers).
18. Update docs/moe-offload/README.md: remove "still pending" caveats; concrete invocation lines for repack / cli / bench; troubleshooting (slot vs n_uniq budget; sector alignment); EAMC sidecar location.

### F verification
- `llama-moe-bench --model ...moe.gguf --pp 1024 --tg 256 --repeat 3 --moe-offload --moe-cache-vram-mb 8000` runs without OOM at `--ctx 8192` and prints a stable aggregated summary across repeats.
- `cmake -B build-moe -DLLAMA_MOE_OFFLOAD=ON -DGGML_CUDA=ON` and default-off (`-DLLAMA_MOE_OFFLOAD=OFF`) both compile clean.

## Relevant files

- `src/moe-offload/slot_pool.{h,cpp}` — D-3 instrumentation; D-4 fix; D-5 callback refactor to use io.h; E-3 predictor wiring; E-1 record_layer.
- `src/moe-offload/io.{h,cpp}` (NEW, Phase D-5) — worker thread, pinned staging pool, `moe_h2d_stream`, events.
- `src/moe-offload/cache.{h,cpp}` — currently dormant; either delete or fold its LRU into the predictor interface in E-3.
- `src/moe-offload/predictor.{h,cpp}` (+ `eamc_predictor`) — wire `observe/score/end_request`; sidecar dump/reload.
- `src/moe-offload/profiler.{h,cpp}` — `record_layer` API + CSV emission; summary aggregation.
- `src/moe-offload/runtime.{h,cpp}` — `end_request` summary; `configure_from_params` reloads `.eamc`.
- `src/moe-offload/loader.cpp` — none expected after D.
- `src/llama-context.cpp` (≈ line 1285) — currently registers callback only when `streaming_mode()`; keep, but also register a profiler-only callback in non-streaming so E-1 still records rows.
- `ggml/src/ggml-cuda/ggml-cuda.cu` — add `ggml_backend_cuda_get_stream(ggml_backend_t)` (Phase D-5).
- `tools/moe-bench/main.cpp` — Phase F orchestrator.
- `docs/moe-offload/README.md` — Phase F.

## Decisions

- I/O: full async (worker + pinned + `moe_h2d_stream` + `cudaStreamWaitEvent`); keep buffered FILE* read (no Win32 direct I/O yet — deferred to a future polish).
- Predictor: full wiring with EAMC sidecar.
- Debug-first for the mmq crash: instrument all loads + every-layer slot_table readback; only escalate to compute-sanitizer if instrumentation is inconclusive.
- Keep golden-prompt greedy argmax as the regression test (already proven for Phase C); no full GoogleTest suite.
- Direct-I/O (`FILE_FLAG_NO_BUFFERING|OVERLAPPED`) and `--moe-oracle` remain deferred.

## Further considerations
1. If D-3 instrumentation shows the slot_table tensor still ends up in a non-`CUDA0` buffer for some graph builds (split / probe / reuse), the fix is to attach the slot_table to the existing topk tensor's backend via a no-op `ggml_view` so the scheduler places it on the same backend as the consumer.
2. The compute-stream accessor in `ggml-cuda.cu` is a small upstream-touching change; keep it `#ifdef LLAMA_MOE_OFFLOAD`-guarded to avoid surface drift on the default build.
3. `record_layer` timing for "compute_us" can over-count if the next layer's MoE topk runs before this layer's mul_mat_id finishes (CUDA stream pipelining). For MVP, document this as "approximate compute_us" and prefer cudaEvent-based measurement of just the SSD-read + H2D phases for the headline numbers.

