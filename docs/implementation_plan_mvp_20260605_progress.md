# MoE Offload MVP Closeout Progress (2026-06-05)

This document summarizes the implementation work performed against
`implementation_plan_mvp_20260605.md`, plus the remaining validation and
implementation gaps. Repository under work: `C:\code\llama.cpp.offload`.

## Summary Status

Implemented and build-tested:

- EAMC sidecar path plumbing and binary save/load.
- `--moe-oracle` post-MVP fail-fast behavior.
- `pred_us` profiler field and summary output.
- Repacker byte-slice verification test.
- Slot tensor allocation honesty by default (`n_slots` expert axis).
- CUDA `.slot` `MUL_MAT_ID` fast-path bypass for correctness-first validation.
- README and known-issues updates.
- CTest coverage for MoE unit tests and oracle fail-fast.

Not yet proven:

- Large-model original-vs-repacked golden logits.
- Large-model repacker byte-slice verifier against Qwen3.5-35B-A3B Q4_K_M.
- Streaming offload `max|Delta logit| < 1e-3`.
- LRU/EAMC large-model logit identity.
- 1K prefill + 256 decode stress run on the target model.

The local build and lightweight CTest suite passed, but the MVP cannot be
called fully closed until the dev-box model gates pass.

## Files Changed

Core public/API plumbing:

- `include/llama.h`
- `src/llama-model.cpp`
- `common/common.h`
- `common/common.cpp`
- `common/arg.cpp`

MoE runtime:

- `src/moe-offload/loader.cpp`
- `src/moe-offload/runtime.h`
- `src/moe-offload/predictor.h`
- `src/moe-offload/predictor.cpp`
- `src/moe-offload/profiler.h`
- `src/moe-offload/profiler.cpp`
- `src/moe-offload/slot_pool.cpp`

CUDA path:

- `ggml/src/ggml-cuda/ggml-cuda.cu`

Tools:

- `tools/moe-bench/main.cpp`
- `tools/llama-bench/llama-bench.cpp`

Tests and test wrapper:

- `tests/CMakeLists.txt`
- `tests/moe-offload/test-eamc-cosine.cpp`
- `tests/moe-offload/test-repack-slices.cpp`
- `cmake/moe-oracle-failfast.cmake`

Docs:

- `docs/moe-offload/README.md`
- `docs/moe-offload/known-issues.md`
- `paper/docs/implementation_plan_mvp_20260605.md`
- `paper/docs/implementation_plan_mvp_20260605_progress.md`

Note: this repository has `tests/.gitignore` rules that cause files under
`tests/moe-offload/` to appear as ignored. The new/rewritten test files exist
on disk and were built by CMake, but they will need force-add or ignore-rule
adjustment if they are to be committed.

## Implemented

### 1. Closeout Plan File

Created:

- `C:\code\paper\docs\implementation_plan_mvp_20260605.md`

The file contains the June 5 MVP closeout plan with `--moe-oracle` excluded
from MVP and the repacker-validation-first ordering.

### 2. EAMC Sidecar Path and Persistence

Added a new model/common parameter:

- `llama_model_params::moe_eamc_path`
- `common_params::moe_eamc_path`

Added CLI/tool support:

- `--moe-eamc-path PATH` in common arg parsing.
- `--moe-eamc-path PATH` in `llama-bench`.
- `--moe-eamc-path PATH` in `llama-moe-bench`.

Runtime behavior:

- If `--moe-predictor eamc` is used and no path is provided, the default sidecar
  path is next to the model with a `.eamc` suffix.
- EAMC attempts to load the sidecar during slot-pool configuration.
- EAMC saves the sidecar at request end.

Binary sidecar format implemented:

- Magic: `EAM1`
- `uint32 n_layers`
- `uint32 n_experts`
- `uint64 capacity`
- `uint64 top_k`
- `uint64 rows`
- Dense float32 rows with shape `rows x (n_layers * n_experts)`

Compatibility behavior:

- Missing sidecar is ignored.
- Incompatible shape/version or truncated data is ignored with a warning.
- Empty corpus is not written.

Test coverage:

- `test-eamc-cosine` now covers save/load round-trip and verifies that an
  A-prefix score ordering survives reload.

### 3. Oracle Mode Deferred and Fail-Fast

Implemented fail-fast behavior in the MoE loader:

- If `params.moe_oracle` is true, model load returns false.
- Error message clearly states that `--moe-oracle` is deferred to post-MVP.

Added CTest coverage:

- `cmake/moe-oracle-failfast.cmake`
- `test-moe-oracle-failfast`

The wrapper test runs `llama-completion` with `--moe-oracle`, requires nonzero
exit, and checks for the expected post-MVP message.

### 4. Profiler `pred_us`

Added `pred_us` to:

- `profile_row`
- `profile_phase_stats`
- CSV output
- runtime aggregate summary
- formatted bench summary
- `llama-moe-bench` summary

Measured predictor time includes:

- `predictor::observe()` per callback.
- Eviction candidate `predictor::score()` scans.

The profiler still treats `stall_us` as an approximation of overlap loss. It is
documented as host wall-clock around miss completion and compute-stream wait
insertion, not a pure CUDA event duration.

### 5. Repacker Byte-Slice Verifier

Added:

- `tests/moe-offload/test-repack-slices.cpp`

Behavior:

- Self-skips unless both env vars are set:
  - `LLAMA_MOE_TEST_ORIGINAL_GGUF`
  - `LLAMA_MOE_TEST_GGUF`
- Opens the original and repacked GGUF metadata.
- Reads `moe_offload.layer_ids`, `n_moe_layers`, `n_experts_per_layer`, and
  `moe_offload.expert_blob.table`.
- For every `(logical_layer, expert, kind)` table record:
  - Finds the corresponding fused expert tensor in the original GGUF.
  - Computes the original expert byte slice.
  - Computes the repacked byte slice from the table entry.
  - Verifies table offset/size consistency.
  - Compares bytes exactly in chunks.

Registered under CTest label `moe-offload`.

What this covers:

- It verifies the repacker table points to exact slices in the fused tensors.
- It is stronger than the existing manifest range test.

What this does not cover until run with real model env vars:

- It did not compare the actual Qwen3.5-35B-A3B Q4_K_M original/repacked files
  during this session.

### 6. Runtime Correctness Changes

Changed slot tensor allocation:

- Slot tensors now default to `ne[2] = n_slots`.
- The previous full expert-axis behavior can be restored with:
  - `LLAMA_MOE_FULL_EXPERT_AXIS=1`

Reason:

- The cache should allocate honestly based on actual slot count.
- The previous full expert-axis workaround hid the cache size and wasted memory.

Changed CUDA `MUL_MAT_ID` dispatch:

- In `ggml/src/ggml-cuda/ggml-cuda.cu`, `.slot` tensors bypass the specialized
  fast-path block for `MUL_MAT_ID`.
- This prevents the remapped MoE slot tensors from using MMVQ/MMQ/MMF paths
  while golden-logit correctness is being validated.
- The generic sorted CUDA path is used instead.

Important limitation:

- I did not implement a new standalone CUDA allocation owner for slot tensors.
  The existing implementation uses loader-created unfiled tensors that are
  persistent model-side tensors. The plan item about "private persistent CUDA
  buffers" is therefore only partially satisfied unless those existing unfiled
  tensors are considered sufficient. A fully separate manual CUDA buffer pool
  remains unimplemented.

Fingerprint diagnostics:

- Existing fingerprint diagnostics remain diagnostic/reload support.
- The implementation does not rely on fingerprints as the primary correctness
  mechanism.

### 7. Documentation Updates

Updated `docs/moe-offload/README.md`:

- Repacker layout now documented as `fused-tensors-page-aligned-v1`.
- Clarified that fused expert tensors remain in the GGUF and the table records
  per-expert byte ranges.
- Added `--moe-eamc-path`.
- Added `pred_us` CSV/summary semantics.
- Documented oracle fail-fast and new CTest coverage.
- Documented slot tensor `n_slots` default and `.slot` CUDA fast-path bypass.

Replaced stale `docs/moe-offload/known-issues.md` content:

- Moved EAMC sidecar persistence to closed.
- Moved layout mismatch and manifest-only validation gap to closed.
- Marked oracle ambiguity closed by fail-fast test.
- Kept streaming golden-logit validation open until large-model gates pass.
- Kept streaming `n_ubatch <= 8` cap open/documented.

## Validation Performed

### Build

Command run:

```powershell
cmake --build build-moe --config Release --target llama-completion llama-moe-bench llama-bench test-eamc-cosine test-lru-eviction test-manifest-roundtrip test-repack-slices
```

Result:

- Build succeeded.
- Existing warnings appeared from CUDA/MSVC/OpenSSL/NCCL configuration; no new
  blocking build errors were observed.

### CTest

Command run:

```powershell
ctest --test-dir build-moe -C Release -L moe-offload --output-on-failure
```

Result:

- 6/6 tests passed.

Passing tests:

- `test-cuda-stream`
- `test-eamc-cosine`
- `test-lru-eviction`
- `test-manifest-roundtrip`
- `test-repack-slices`
- `test-moe-oracle-failfast`

Notes:

- `test-manifest-roundtrip` self-skips without `LLAMA_MOE_TEST_GGUF`.
- `test-repack-slices` self-skips without both large-model env vars.
- The oracle test actually runs `llama-completion` against a tiny bundled vocab
  GGUF and verifies nonzero exit plus expected error text.

## Not Done / Still Required

### 1. Repacker Large-Model Validation

Not run:

- `test-repack-slices` against:
  - original `Qwen3.5-35B-A3B-Q4_K_M.gguf`
  - repacked `Qwen3.5-35B-A3B-Q4_K_M.moe.gguf`

Required command pattern:

```powershell
$env:LLAMA_MOE_TEST_ORIGINAL_GGUF="C:\path\to\Qwen3.5-35B-A3B-Q4_K_M.gguf"
$env:LLAMA_MOE_TEST_GGUF="C:\path\to\Qwen3.5-35B-A3B-Q4_K_M.moe.gguf"
ctest --test-dir build-moe -C Release -R test-repack-slices --output-on-failure
```

Acceptance:

- Every table entry must byte-compare exactly.

### 2. Original vs Repacked Logit Validation With Offload Disabled

Not run:

- Original GGUF vs repacked `.moe.gguf` with MoE offload disabled.

Acceptance:

- Repacked/offload-disabled logits must match original GGUF golden logits.

This is necessary to prove the repacker is not the cause of runtime issues.

### 3. Runtime Golden-Logit Validation

Not run:

- Full-residency offload vs repacked/offload-disabled golden logits.
- Streaming offload forced-eviction cache vs full-residency golden logits.

Required gate from plan:

- Streaming offload at `4000 MiB`, or the smallest safe cache, reports
  `max|Delta logit| < 1e-3`.

Suggested command pattern:

```powershell
powershell -ExecutionPolicy Bypass -File tests\moe-offload\test-golden-logits.ps1 `
    -Tol 1e-3 -NPredict 8 -StreamCacheMb 4000
```

Status:

- The historical known issue was `max|Delta logit| = 4.64e-01`.
- The June 5 changes target the leading suspects, but the result is unknown
  until the large-model harness is rerun.

### 4. LRU vs EAMC Logit Identity

Not run:

- LRU and EAMC comparison at `--temp 0 --seed 42`.

Acceptance:

- LRU and EAMC produce identical logits under deterministic decode settings.

### 5. Stress Run

Not run:

- 1K prefill + 256 decode stress run.

Acceptance:

- Completes without hang.
- No CUDA illegal access.
- No use-after-evict.

### 6. Profiler/Bench Large-Model Gate

Not run:

```powershell
.\build-moe\bin\Release\llama-moe-bench.exe `
  --model <Qwen3.5-35B-A3B-Q4_K_M.moe.gguf> `
  --pp 1024 --tg 256 --repeat 3 `
  --moe-cache-vram-mb 8000 `
  --moe-predictor eamc `
  --moe-profile-csv moe-profile.csv `
  --moe-profile-summary moe-summary.txt
```

Expected summary fields:

- TTFT
- TPOT
- total time
- hit rates
- SSD/H2D/compute/stall timings
- predictor overhead
- VRAM/DRAM/SSD usage

### 7. CPU-Only / Offload-Off Build Gate

Not run in this session:

- A clean non-MoE/offload-off build and smoke test.

Reason:

- Work was verified in the existing `build-moe` tree only.

## Risks and Caveats

### Test Files Under `tests/` Are Ignored

`tests/.gitignore` ignores test files by default. The following source files
exist and were built, but show as ignored:

- `tests/moe-offload/test-eamc-cosine.cpp`
- `tests/moe-offload/test-repack-slices.cpp`

If these changes are committed, force-add them or adjust the ignore rules.

### Runtime Correctness Is Not Proven Yet

The code changes are aimed at the known corruption/drift suspects, but only the
large-model golden-logit harness can prove whether the MVP correctness gap is
closed.

### Private CUDA Buffer Ownership Is Not Fully Implemented

The plan called for moving slot tensors and slot tables into private persistent
CUDA buffers not reused by graph temporaries. The current implementation uses
persistent unfiled tensors created through the model loader path and bypasses
the suspect CUDA fast paths. A fully separate CUDA buffer allocator for MoE slot
storage has not been implemented.

### Performance May Regress

Bypassing specialized `MUL_MAT_ID` fast paths for `.slot` tensors is a
correctness-first choice. It may reduce throughput until a safe specialized
path is restored.

## Current Closeout Checklist

Done:

- Plan file written.
- EAMC sidecar path added.
- EAMC binary load/save implemented.
- EAMC sidecar round-trip test added.
- Oracle fail-fast implemented and tested.
- Repacker byte-slice verifier implemented.
- Profiler `pred_us` added.
- README and known issues updated.
- Slot tensor default shape changed to actual cache slots.
- `.slot` CUDA `MUL_MAT_ID` fast-path bypass added.
- MoE CTest label passes locally.

Pending:

- Run byte-slice verifier with real original/repacked Qwen GGUF files.
- Run original vs repacked/offload-disabled logits.
- Run full-residency offload logits.
- Run streaming forced-eviction logits and verify `< 1e-3`.
- Run LRU vs EAMC deterministic logit identity.
- Run 1K prefill + 256 decode stress.
- Run `llama-moe-bench` profiler/summary gate on the target model.
- Run offload-off build/smoke gate.

## Bottom Line

The implementation surface for the June 5 MVP closeout is largely in place and
the lightweight MoE test suite passes. The MVP is not fully closed until the
large-model correctness gates prove that:

1. The repacked model is byte/logit-equivalent with offload disabled.
2. Full-residency offload matches the repacked/offload-disabled reference.
3. Streaming offload with forced eviction reaches `max|Delta logit| < 1e-3`.
