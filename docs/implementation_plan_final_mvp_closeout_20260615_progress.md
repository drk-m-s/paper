# Final MoE Offload MVP Closeout Progress (2026-06-15)

Plan: `paper/docs/implementation_plan_final_mvp_closeout_20260615.md`.

## Scope Update

This closeout is now `llama-cli` first. The goal is for humans to experience
the MoE offload speedup in interactive chat, not only through
`llama-moe-bench` summaries.

Out of scope for this closeout:

- Post-MVP cache-management research in
  `post_mvp_plan_cache_improve_20260615.md`.

## Initial Plan Changes

- Removed non-CLI phases, validation, deliverables, and closeout criteria from
  the final MVP closeout plan.
- Prioritized `llama-cli` performance parity with `llama-moe-bench`.
- Added explicit README and known-issues documentation requirements for the
  eventual `llama-cli` fix.
- Added a requirement to preserve a conservative ubatch-1 fallback even if the
  fast interactive path is promoted.

## Current Implementation Status

Implemented the CLI-first closeout path.

Code changes in `C:\code\llama.cpp.offload`:

- Added `--moe-fast-paths` for `llama-cli`, with `LLAMA_MOE_FAST_PATHS=1` as
  the environment equivalent.
- `--moe-fast-paths` applies the accepted fast guard stack:
  `LLAMA_MOE_SLOT_MMVQ=1`, `LLAMA_MOE_SLOT_GRAPHS=1`,
  `LLAMA_MOE_SLOT_GLU_FUSION=1`, `LLAMA_MOE_PREFILL_MMVQ=0`, and
  `LLAMA_MOE_TOPK_FUSION_DIAG=0`.
- The default `llama-cli --moe-offload` path still forces `n_ubatch=1` when no
  streaming ubatch override is set.
- The fast profile no longer applies the CLI ubatch-1 clamp, so the runtime
  auto-sizes streaming ubatch from the cache budget.
- `llama-cli` now prints the effective MoE `n_ubatch` after model load.
- `llama-cli --moe-profile-summary` now writes the bench-style MoE summary at
  process exit, using the existing `llama_moe::format_summary()` helper and
  aggregating CLI turn timings.
- The shared summary formatter now reports VRAM as unavailable when a caller
  does not provide memory samples, instead of printing a misleading 0 GB peak.

Documentation changes:

- Updated `docs/moe-offload/README.md` with the human-facing fast
  `llama-cli` command, profiling command, fallback behavior, and fast-profile
  guard explanation.
- Updated `docs/moe-offload/known-issues.md` so the old ubatch-1 caveat now
  reflects the explicit fast profile and conservative fallback.

## Validation

Builds:

```powershell
cmake --build build-moe-static --config Release --target llama-cli -j 8
cmake --build build-moe\tools\cli --config Release --target llama-cli -j 8
```

Both passed.

Focused model-dependent fast CLI run:

```powershell
.\build-moe-static\bin\Release\llama-cli.exe `
  --model C:/AI/models/qwen/Qwen3.5-35B-A3B-Q4_K_M.moe.gguf `
  --moe-offload --moe-cache-vram-mb 12000 --moe-predictor lru `
  --moe-fast-paths `
  --moe-profile-csv tests\moe-offload\_out\cli-closeout-fast.csv `
  --moe-profile-summary tests\moe-offload\_out\cli-closeout-fast.summary.txt `
  --jinja --reasoning off --temp 0 --seed 42 --simple-io --no-warmup `
  -c 4096 -n 32 -sys "You are a helpful assistant."
```

Result:

- Answered `what is the capital of France?` with Paris.
- Reported `llama-cli: MoE effective n_ubatch=16 (fast paths)`.
- Summary reported `ubatch: requested=512 effective=16  slots=145/256`.
- Summary emitted bench-style TTFT/TPOT and prefill/decode I/O breakdowns.

Correctness and smoke gates:

```powershell
ctest --test-dir build-moe-static -C Release -L moe-offload --output-on-failure
```

Passed: 8/8 tests.

```powershell
$env:LLAMA_MOE_FAST_PATHS='1'
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\tests\moe-offload\test-llama-cli-chat.ps1 `
  -Bin "$PWD\build-moe-static\bin\Release\llama-cli.exe" `
  -CacheMb 12000 -Predictor lru -NPredict 64 -StreamingUBatch 0
```

Passed: formatted `llama-cli` chat smoke.

Earlier in this closeout run, the accepted guard stack also passed the raw
golden-logit gate with `max|d| = 0` for the short `Hello` prompt, 4000 MiB
streaming cache, ubatch 8, and 8 predicted tokens.

## Residual Notes

- The fast CLI summary is now comparable in structure to `llama-moe-bench`, but
  CLI prompt shape, chat loop, console I/O, and cache state still differ from
  synthetic bench runs. Use matched cache/predictor/guard/cache-state settings
  when comparing numbers.
- The short one-turn CLI timing is noisy. The important closeout behavior is
  that humans can run one documented `llama-cli` command, see effective
  auto-sized ubatch, get coherent chat output, and collect the same category of
  MoE breakdown as the bench report.
