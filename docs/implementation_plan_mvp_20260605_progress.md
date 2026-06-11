# MoE Offload MVP Closeout Progress (2026-06-05)

Repository under work: `C:\code\llama.cpp.offload`.

This document records the final closeout work against
`implementation_plan_mvp_20260605.md`. The MVP closeout gates are now
implemented and validated on the dev box, with the remaining items explicitly
classified as post-MVP limitations.

## Summary Status

Closed for MVP:

- Repacker validation-first gate.
- Original GGUF vs repacked `.moe.gguf` offload-disabled logit validation.
- Full-residency offload logit validation.
- Forced-eviction streaming offload logit validation.
- LRU vs EAMC deterministic logit identity.
- EAMC sidecar path, binary persistence, compatibility checks, and test
  round-trip.
- Profiler CSV/summary semantics, including `pred_us`.
- `--moe-oracle` post-MVP fail-fast.
- MoE CTest/dev-box tests.
- 1024 prefill + 256 decode `llama-moe-bench` stress/profiler gate.
- Offload-off build gate.

Still post-MVP:

- Removing the streaming `n_ubatch <= 8` cap.
- Re-enabling CUDA top-k MoE fusion safely under `LLAMA_MOE_OFFLOAD`.
- Re-enabling specialized CUDA `.slot` `MUL_MAT_ID` fast paths after targeted
  kernel validation.
- Direct I/O, multi-GPU, CPU DRAM expert tier, KV offload, learned predictors,
  speculative prefetch/decoding, and FineMoE-style splitting.

No performance floor is required for MVP. The acceptance bar is correctness,
stability, and truthful measurement.

## Implementation Completed

### Repacker Validation

Implemented and validated:

- `fused-tensors-page-aligned-v1` documented as the actual repacker layout.
- The layout keeps the original fused expert tensors and adds
  `moe_offload.expert_blob.table` with per-expert byte ranges.
- `test-repack-slices` verifies each `(layer, expert, kind)` table entry points
  to the exact byte slice in the original GGUF.

Validation status:

- Manifest/table size/range test passed.
- Byte-slice verifier passed against the Qwen3.5-35B-A3B Q4_K_M original and
  repacked files.
- Repacked `.moe.gguf` with offload disabled matched original GGUF golden
  logits exactly.

Conclusion: the repacker is not the source of the previous streaming drift.

### Streaming Correctness

Implemented:

- Streaming slot tensors use the actual cache slot count by default.
- `LLAMA_MOE_FULL_EXPERT_AXIS=1` remains only as a guarded fallback.
- Streaming remap no longer depends on a persistent slot table plus
  `ggml_get_rows`.
- `remap_selected_experts()` now creates a graph-private
  `moe.slot_ids.<layer>` I32 tensor, marks it as an output, registers it with
  the top-k tensor, and returns it directly to `MUL_MAT_ID`.
- The eval callback stops after registered MoE top-k tensors, ensures required
  experts are resident, then fills the registered slot-id tensor with the exact
  slot ids needed by that forward pass.
- Same-forward-pass hit/miss eviction protection is preserved: every expert in
  the current top-k set is reserved before eviction candidates are considered.
- Fingerprint diagnostics are debug-only behind `LLAMA_MOE_DEBUG_SLOT_FP` and no
  longer repair cache state.

CUDA correctness guards:

- CUDA `.slot` `MUL_MAT_ID` bypasses specialized MMVQ/MMQ/MMF paths and uses
  the generic sorted path for correctness-first validation.
- CUDA graphs are disabled for `.slot` `MUL_MAT_ID` nodes because the generic
  sorted path performs host/device synchronization.
- CUDA top-k MoE fusion is disabled in `LLAMA_MOE_OFFLOAD` builds. Diagnostics
  isolated this fusion as sufficient to reproduce the previous
  full-vs-streaming logit mismatch.

Implementation note:

- No standalone manual CUDA allocator was added. Slot weight tensors remain
  loader-created persistent unfiled tensors, and the table-based remap path is
  bypassed by callback-filled slot-id tensors. This satisfies the MVP
  correctness requirement without adding another CUDA buffer ownership layer.

### EAMC Persistence and Predictor Cost

Implemented:

- `--moe-eamc-path PATH`.
- Default EAMC sidecar path next to the model with `.eamc` suffix.
- Binary EAMC save/load with magic/version, shape checks, capacity/top-k checks,
  and warnings for incompatible or truncated sidecars.
- `test-eamc-cosine` sidecar round-trip coverage.
- Cached EAMC nearest-neighbor ranking inside a callback so repeated eviction
  `score()` calls do not recompute cosine ranking for each candidate.

The bench gate showed EAMC predictor overhead reduced enough for the 3-repeat
1024/256 run to complete.

### Profiler Semantics

Implemented:

- Event-based `h2d_us` on the CUDA async H2D path.
- Event-based `compute_us`.
- `pred_us` in profile rows, aggregate stats, CSV output, and summary output.
- `stall_us` documented as host wall time around miss completion and
  compute-stream wait insertion. It remains an approximation of overlap loss.

### Oracle Scope

Implemented:

- Parser/parameter field can remain.
- Passing `--moe-oracle` fails fast with a clear post-MVP error.
- `test-moe-oracle-failfast` verifies this behavior.

`--moe-oracle` is not in the MVP acceptance gates.

## Code Files Changed In Final Closeout

Runtime/CUDA changes currently in the worktree:

- `ggml/src/ggml-cuda/ggml-cuda.cu`
- `src/moe-offload/predictor.cpp`
- `src/moe-offload/runtime.cpp`
- `src/moe-offload/slot_pool.cpp`
- `src/moe-offload/slot_pool.h`
- `tests/moe-offload/test-cuda-stream.cpp`

Documentation updated:

- `docs/moe-offload/README.md`
- `docs/moe-offload/known-issues.md`
- `tests/moe-offload/README.md`
- `paper/docs/implementation_plan_mvp_20260605_progress.md`

Earlier closeout work already present in the repository covers the EAMC CLI
plumbing, profiler structs, fail-fast test wrapper, repacker verifier, and
bench/common argument wiring.

Repository note:

- `tests/moe-offload/` is ignored by this checkout's Git rules, so edits under
  that directory do not appear in normal `git status`. The files were still
  rebuilt and used by CTest.

## Validation Performed

### Build Gates

MoE/CUDA build:

```powershell
cmake --build build-moe --config Release --target `
  llama-completion llama-moe-bench llama-bench `
  test-eamc-cosine test-lru-eviction test-manifest-roundtrip test-repack-slices
```

Result: passed.

Offload-off build:

```powershell
cmake --build build-vanilla-check --config Release --target llama-completion
```

Result: passed. `build-vanilla-check` used `LLAMA_MOE_OFFLOAD=OFF` and
`GGML_CUDA=OFF`.

### CTest

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

Final cleanup note:

- `test-cuda-stream` initially failed once after rebuild because the test
  zeroed the device buffer on the default stream and immediately started the
  MoE H2D copy on a non-blocking stream. The test now synchronizes after the
  setup `cudaMemset`, then the full `moe-offload` CTest label passed 6/6.

### Golden Logits

Command:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tests\moe-offload\test-golden-logits.ps1 `
  -Model "C:\AI\models\qwen\Qwen3.5-35B-A3B-Q4_K_M.moe.gguf" `
  -Tol 1e-3 -NPredict 8 -StreamCacheMb 4000 `
  -Prompt "Hello" -Seed 42 -Context 4096 -UBatch 8
```

Result: passed.

Observed:

- `n_steps=8`
- `n_vocab=248320`
- `max|d|=0`
- `mean|d|=0`

### LRU vs EAMC Identity

Validated with two deterministic `llama-completion` runs at:

- `--moe-cache-vram-mb 4000`
- `--temp 0`
- `--seed 42`
- `-n 8`

The resulting logit dumps compared equal:

- `max|d|=0`
- `mean|d|=0`

### Bench and Stress Gate

Command:

```powershell
build-moe\bin\Release\llama-moe-bench.exe `
  --model "C:\AI\models\qwen\Qwen3.5-35B-A3B-Q4_K_M.moe.gguf" `
  --pp 1024 --tg 256 --repeat 3 `
  --moe-cache-vram-mb 8000 `
  --moe-predictor eamc `
  --moe-eamc-path tests\moe-offload\_out\moe-bench-1024x256.eamc `
  --moe-profile-csv tests\moe-offload\_out\moe-bench-1024x256.csv `
  --moe-profile-summary tests\moe-offload\_out\moe-bench-1024x256.summary.txt `
  -ngl 99 -c 4096
```

Result: exit code 0 in about 297.5 seconds.

Summary:

- Repeats: 3.
- TTFT: 68463.7 ms.
- TPOT: 115.54 ms.
- Total: 98041.7 ms.
- Prefill hit rate: 77.8%.
- Decode hit rate: 90.2%.
- Decode SSD reads: 42.63 GB, average 56.85 MB/token.
- Predictor overhead: 40.63 ms/token.
- VRAM peak: 10.15 GB / 15.92 GB.
- DRAM peak: 1.56 GB.
- Profile rows: prefill=15360, decode=30720.
- CSV and summary written under `tests\moe-offload\_out\`.

This also satisfies the 1K prefill + 256 decode stress requirement: it
completed without hang, CUDA illegal access, or use-after-evict symptoms.

## Diagnostics That Drove the Fix

The previous streaming drift was not caused by SSD eviction/loading or by the
repacker:

- Full-cache forced streaming and no-remap diagnostics still failed with
  `max|d|=3.291510e-01` at step 1.
- `GGML_CUDA_DISABLE_FUSION=1` passed exactly.
- Per-fusion isolation showed disabling top-k MoE fusion alone was sufficient.
- Disabling SSM conv fusion alone did not help.

The final runtime/CUDA changes therefore keep the offload-specific path on
correctness-first CUDA behavior until the fused/specialized paths can be
validated separately.

## Current Closeout Checklist

Done:

- Offload-off build gate.
- MoE/CUDA build gate.
- Manifest/table size/range test.
- Byte-slice verifier against Qwen original/repacked GGUF.
- Original vs repacked/offload-disabled logits.
- Full-residency offload logits.
- Streaming forced-eviction logits at 4000 MiB.
- LRU vs EAMC deterministic logit identity.
- 1024 prefill + 256 decode stress/bench run.
- Profiler CSV and summary gate.
- EAMC sidecar persistence and round-trip coverage.
- `--moe-oracle` fail-fast behavior and test.
- README, known-issues, and test docs updated.

Not done because it is post-MVP:

- Remove streaming `n_ubatch <= 8` cap.
- Restore CUDA top-k MoE fusion for MoE offload.
- Restore specialized CUDA `.slot` `MUL_MAT_ID` kernels.
- Implement direct I/O, multi-GPU, CPU DRAM expert tier, KV offload, learned
  predictors, speculative prefetch/decoding, or FineMoE-style splitting.

## Bottom Line

The MoE offload MVP closeout is implemented and passes the requested dev-box
gates for Qwen3.5-35B-A3B Q4_K_M. The remaining work is no longer MVP
correctness closure; it is post-MVP performance and feature hardening.
