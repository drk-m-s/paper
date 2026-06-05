# MoE Offload MVP Closeout Plan (2026-06-05)

## Summary

Close the MoE offload MVP by proving the repacked model is correct first, then
fixing the remaining runtime correctness gap so streaming offload matches
full-residency logits. The MVP must support Qwen3.5-35B-A3B Q4_K_M with LRU and
EAMC predictors, async SSD-to-VRAM expert loading, profiler CSV/summary output,
EAMC sidecar persistence, and dev-box tests. `--moe-oracle` is not part of MVP.

## Key Changes

- Add a repacker validation pass before runtime fixes:
  - Verify original GGUF vs repacked `.moe.gguf` with offload disabled.
  - Add a byte-level table verifier that checks each `(layer, expert, kind)`
    table entry points to the exact slice in the fused source tensor.
  - Fix README/layout naming mismatch (`fused-tensors-page-aligned-v1`) and
    document that the implemented layout keeps fused tensors plus a per-expert
    offset table.
- Fix streaming correctness:
  - Restore cache allocation honesty: slot tensors use the actual cache slot
    count unless a CUDA kernel requires a guarded fallback.
  - Prevent same-forward-pass slot corruption by moving slot tensors and slot
    tables into private persistent CUDA buffers not reused by graph temporaries.
  - For MVP correctness, bypass/disable the suspect MMQ `mul_mat_id` path for
    remapped MoE slot tensors until golden logits pass; use the safer generic
    CUDA path.
  - Keep fingerprint diagnostics as debug detection only, not as the primary
    correctness mechanism.
- Complete EAMC persistence:
  - Add `--moe-eamc-path PATH`; default sidecar path is next to the model with
    `.eamc`.
  - Implement binary EAMC save/load with shape/version checks; ignore
    incompatible sidecars with a warning.
  - Add sidecar round-trip coverage to the EAMC test.
- Tighten profiler semantics:
  - Keep event-based `h2d_us` and `compute_us`.
  - Redefine `stall_us` as compute-stream wait time for miss events; document
    any remaining approximation.
  - Add `pred_us` for EAMC predictor overhead.
- Treat `--moe-oracle` as deferred:
  - Keep the parser field if already present, but fail fast with a clear
    "oracle mode is post-MVP" error when passed.
  - Remove oracle from MVP docs and acceptance gates.

## Test Plan

- Build gates:
  - Offload OFF build passes existing smoke tests.
  - `LLAMA_MOE_OFFLOAD=ON + GGML_CUDA=ON` builds `llama-completion`,
    `llama-moe-bench`, and all MoE tests.
- Repacker gates:
  - Manifest/table size/range test passes.
  - New byte-slice verifier passes against
    `Qwen3.5-35B-A3B-Q4_K_M.moe.gguf`.
  - Repacked model with offload disabled matches original GGUF golden logits.
- Runtime correctness gates:
  - Full-residency offload matches repacked/offload-disabled golden logits.
  - Streaming offload at forced-eviction cache (`4000 MiB`, or smallest safe
    cache) reports `max|Delta logit| < 1e-3`.
  - LRU and EAMC produce identical logits at `--temp 0 --seed 42`.
  - 1K prefill + 256 decode stress run completes without hang, CUDA illegal
    access, or use-after-evict.
- Profiler/bench gates:
  - `llama-moe-bench --pp 1024 --tg 256 --repeat 3 --moe-cache-vram-mb 8000`
    emits CSV and summary.
  - Summary includes TTFT, TPOT, total time, hit rates,
    SSD/H2D/compute/stall timings, predictor overhead, VRAM/DRAM/SSD usage.
- CTest/dev-box tests:
  - Existing `moe-offload` tests pass.
  - Add cache lifecycle, I/O buffer/event lifetime, EAMC sidecar round-trip,
    repacker byte-slice verification, and `--moe-oracle` fail-fast tests.

## Assumptions

- Direct I/O, multi-GPU, CPU DRAM expert tier, KV offload, learned predictors,
  speculative prefetch/decoding, FineMoE-style splitting, and ubatch-cap
  removal remain post-MVP.
- The streaming `n_ubatch <= 8` cap remains acceptable for MVP if correctness
  passes and the limitation is documented.
- No perf floor is required; MVP acceptance is correctness, stability, and
  truthful measurement.
