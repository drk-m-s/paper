# Final MoE Offload MVP Closeout Plan (2026-06-15)

Repository under work: `C:\code\llama.cpp.offload`.

This final MVP closeout plan is focused on `llama-cli` performance and user
experience. Humans need to run `llama-cli` and feel the MoE offload speedup
directly; `llama-moe-bench` numbers are not sufficient for closeout by
themselves.

The human-written `post_mvp_plan_cache_improve_20260615.md` is out of scope;
that document is for later post-MVP cache-management work and should not drive
this closeout.

## Goal

Make `llama-cli --moe-offload` perform close to the validated
`llama-moe-bench` path under comparable settings, while preserving the Phase K
correctness gates and maintaining a clear fallback for conservative chat
behavior.

The final user-facing result should be a documented `llama-cli` command that:

- uses the accepted MoE offload fast-path guard stack,
- uses a cache budget that reaches a useful effective ubatch on the target GPU,
- produces coherent Qwen chat responses,
- reports or can reproduce MoE profile metrics,
- does not require users to infer benchmark-only setup details.

## Inputs To Use

- `docs/moe-offload/README.md`
- `docs/moe-offload/known-issues.md`
- `paper/docs/implementation_plan_mvp_perf_20260611.md`
- `paper/docs/implementation_plan_mvp_perf_20260611_progress.md`
- `tools/cli/cli.cpp`
- `tools/moe-bench/main.cpp`
- `common/arg.cpp`
- `common/common.cpp`
- `src/moe-offload/runtime.*`
- `src/moe-offload/profiler.*`
- `src/moe-offload/slot_pool.*`

Do not use `paper/docs/post_mvp_plan_cache_improve_20260615.md` as an input for
this final MVP closeout plan.

## Current Findings Before Implementation

`tools/cli/cli.cpp` currently forces `params.n_ubatch = 1` whenever
`--moe-offload` is enabled and `LLAMA_MOE_STREAMING_UBATCH` is unset:

```cpp
if (params.moe_offload && std::getenv("LLAMA_MOE_STREAMING_UBATCH") == nullptr) {
    params.n_ubatch = 1;
}
```

That guard was added for correctness-first interactive chat behavior. It is
also the most likely reason `llama-cli` does not show the same prefill speed as
`llama-moe-bench`, because the Phase K recommendation uses 12000 MiB cache and
effective ubatch 16.

`llama-moe-bench` closeout numbers also assume an accepted guard stack that is
not automatically enabled by ordinary `llama-cli` commands:

```powershell
$env:LLAMA_MOE_SLOT_MMVQ='1'
$env:LLAMA_MOE_PREFILL_MMVQ='0'
$env:LLAMA_MOE_SLOT_GRAPHS='1'
$env:LLAMA_MOE_SLOT_GLU_FUSION='1'
$env:LLAMA_MOE_TOPK_FUSION_DIAG='0'
```

Therefore the likely causes of the user-visible gap are:

- CLI forces ubatch 1 by default.
- CLI users do not know they must enable the accepted fast-path guards.
- CLI chat prompts/templates differ from synthetic bench prompt shape.
- CLI is often measured cold while bench may be warm-cache or hot-started.
- CLI wall time includes chat-loop, sampling, and console overhead that
  `llama-moe-bench` isolates more tightly.
- The normal runtime profile summary may not expose the same cold/warm and
  prefill/decode breakdowns that `llama-moe-bench` emits.

## Non-Goals

- Do not make benchmark-only `--moe-hot-start` a CLI default.
- Do not promote `LLAMA_MOE_PREFILL_MMVQ=1`.
- Do not promote `LLAMA_MOE_TOPK_FUSION_DIAG=1`.
- Do not use post-MVP cache-management papers or plans as blockers for this
  closeout.

## Phase A - CLI Versus Bench Baseline

Establish a comparable baseline before changing code.

### Tasks

1. Build the static validation targets:

```powershell
cmake --build build-moe-static --config Release --target `
  llama-cli llama-completion llama-moe-bench `
  test-slot-mmvq test-topk-moe-fusion -j 8
```

2. Record environment and guard stack:

```powershell
$env:LLAMA_MOE_SLOT_MMVQ='1'
$env:LLAMA_MOE_PREFILL_MMVQ='0'
$env:LLAMA_MOE_SLOT_GRAPHS='1'
$env:LLAMA_MOE_SLOT_GLU_FUSION='1'
$env:LLAMA_MOE_TOPK_FUSION_DIAG='0'
```

3. Run the Phase K comparable benchmark:

```powershell
.\build-moe-static\bin\Release\llama-moe-bench.exe `
  --model C:/AI/models/qwen/Qwen3.5-35B-A3B-Q4_K_M.moe.gguf `
  --pp 256 --tg 128 --repeat 3 `
  --moe-cache-vram-mb 12000 --moe-predictor eamc `
  --moe-reset-cache-between-repeats `
  --moe-profile-csv tests\moe-offload\_out\cli-closeout-a-bench.csv `
  --moe-profile-summary tests\moe-offload\_out\cli-closeout-a-bench.summary.txt
```

4. Run `llama-cli` default MoE behavior with profiling:

```powershell
.\build-moe-static\bin\Release\llama-cli.exe `
  --model C:/AI/models/qwen/Qwen3.5-35B-A3B-Q4_K_M.moe.gguf `
  --moe-offload --moe-cache-vram-mb 12000 --moe-predictor eamc `
  --moe-profile-csv tests\moe-offload\_out\cli-closeout-a-default.csv `
  --moe-profile-summary tests\moe-offload\_out\cli-closeout-a-default.summary.txt `
  --jinja --reasoning off --temp 0 --seed 42 --simple-io `
  -sys "You are a helpful assistant."
```

5. Repeat `llama-cli` with `LLAMA_MOE_STREAMING_UBATCH=16` to isolate the
   current CLI ubatch guard.

6. Repeat `llama-cli` with LRU predictor because it is simpler and may be a
   better first user-facing default if EAMC overhead hurts interactivity.

### Metrics

Compare these fields between bench and CLI:

- effective ubatch,
- slots,
- prompt token count,
- generated token count,
- TTFT,
- TPOT,
- prefill and decode hit rates,
- SSD read ms/token,
- H2D ms/token,
- stall ms/token,
- `gpu_compute` ms/token,
- predictor ms/token,
- callback wall ms/token,
- unattributed wall time,
- peak VRAM.

### Acceptance

- The baseline identifies whether the main CLI gap is ubatch, missing fast
  paths, predictor/cache behavior, prompt shape, or unprofiled wall time.
- Baseline artifacts are recorded in the progress report.

## Phase B - Make Fast `llama-cli` Easy To Invoke

Users should not have to remember several environment variables to experience
the validated fast path.

### Preferred Implementation

Add one documented CLI/common option or one documented environment profile for
the accepted fast path.

Candidate option:

```text
--moe-fast-paths
```

Candidate environment profile:

```text
LLAMA_MOE_FAST_PATHS=1
```

The profile may enable only accepted Phase K guards:

- `LLAMA_MOE_SLOT_MMVQ=1`
- `LLAMA_MOE_SLOT_GRAPHS=1`
- `LLAMA_MOE_SLOT_GLU_FUSION=1`

The profile must not enable:

- `LLAMA_MOE_PREFILL_MMVQ=1`
- `LLAMA_MOE_TOPK_FUSION_DIAG=1`

### Tasks

1. Choose the smallest implementation that matches existing common-arg style.
2. Make the fast-path selection visible in logs or profile summary so users can
   confirm it was active.
3. Document the human-facing fast CLI command in
   `docs/moe-offload/README.md`.
4. Keep the raw individual env vars documented for diagnostics.

### Acceptance

- A user can run one documented command and get the accepted fast-path stack.
- The feature is disabled or conservative unless explicitly requested, unless
  Phase C proves it can safely become the default.
- README distinguishes accepted fast paths from experimental prefill MMVQ and
  diagnostic top-k fusion.

## Phase C - Replace Or Relax The CLI UBatch Guard

The current ubatch-1 CLI guard protects old chat behavior but directly blocks
human-visible prefill performance.

### Candidate Policies

1. **Preferred if validation passes:** remove the hard-coded ubatch-1 override
   and let normal MoE auto-sizing choose the effective ubatch from cache slots.
2. **Conservative compromise:** keep default ubatch 1 unless
   `--moe-fast-paths` or `LLAMA_MOE_FAST_PATHS=1` is set.
3. **Explicit user mode:** add `--moe-chat-fast` or similar only if a general
   fast-path knob is too broad.

The fallback must remain simple:

```powershell
$env:LLAMA_MOE_STREAMING_UBATCH='1'
```

or an equivalent documented CLI flag.

### Tasks

1. Implement the chosen policy in `tools/cli/cli.cpp` and/or common args.
2. Re-run formatted chat smoke:
   - default CLI,
   - accepted fast path,
   - forced `LLAMA_MOE_STREAMING_UBATCH=8`,
   - fallback ubatch 1.
3. Re-run golden logits with accepted fast path.
4. Compare CLI profile metrics to `llama-moe-bench` after the policy change.

### Acceptance

- `llama-cli` no longer silently prevents the validated faster prefill path
  when the user asks for the final-MVP fast MoE configuration.
- Chat smoke remains coherent for identity, factual, and arithmetic prompts.
- The fallback to ubatch 1 is documented and works.

## Phase D - CLI Profile Summary Parity

If CLI profiling is weaker than `llama-moe-bench`, improve reporting so CLI
performance can be diagnosed without guessing.

### Tasks

1. Check whether `--moe-profile-summary` from `llama-cli` emits the Phase J
   cold/warm and prefill/decode breakdowns or an older aggregate format.
2. If it emits older output, reuse `llama_moe::format_summary()` for normal
   runtime summaries or add a CLI final-summary path with:
   - effective ubatch,
   - slots,
   - TTFT,
   - TPOT,
   - prefill/decode I/O breakdown,
   - prefill/decode profiler breakdown,
   - VRAM peak if available.
3. Ensure `--moe-profile-csv` still writes per-layer rows and request rows.

### Acceptance

- `llama-cli` profile artifacts are sufficient to compare against
  `llama-moe-bench`.
- README shows how to enable CLI profiling.

## Phase E - Human-Facing CLI Validation

Validate the path a human will actually run.

### Required Commands

Fast interactive command, to be finalized after implementation:

```powershell
.\build-moe-static\bin\Release\llama-cli.exe `
  --model C:/AI/models/qwen/Qwen3.5-35B-A3B-Q4_K_M.moe.gguf `
  --moe-offload `
  --moe-cache-vram-mb 12000 `
  --moe-predictor lru `
  --moe-fast-paths `
  --jinja `
  --reasoning off `
  -sys "You are a helpful assistant."
```

Profiling command:

```powershell
.\build-moe-static\bin\Release\llama-cli.exe `
  --model C:/AI/models/qwen/Qwen3.5-35B-A3B-Q4_K_M.moe.gguf `
  --moe-offload `
  --moe-cache-vram-mb 12000 `
  --moe-predictor lru `
  --moe-fast-paths `
  --moe-profile-csv tests\moe-offload\_out\cli-final.csv `
  --moe-profile-summary tests\moe-offload\_out\cli-final.summary.txt `
  --jinja `
  --reasoning off `
  --temp 0 --seed 42 --simple-io `
  -sys "You are a helpful assistant."
```

Fallback command:

```powershell
$env:LLAMA_MOE_STREAMING_UBATCH='1'
.\build-moe-static\bin\Release\llama-cli.exe `
  --model C:/AI/models/qwen/Qwen3.5-35B-A3B-Q4_K_M.moe.gguf `
  --moe-offload --moe-cache-vram-mb 8000 --moe-predictor lru `
  --jinja --reasoning off `
  -sys "You are a helpful assistant."
```

### Acceptance

- A human can use the fast interactive command without memorizing hidden env
  vars.
- The output is coherent on the existing chat-smoke prompts.
- Prefill and decode speeds are visibly closer to bench results than the
  current default CLI path.
- If CLI wall time still trails bench, the remaining delta is measured and
  documented.

## Phase F - Final Correctness And Performance Gates

Run after CLI changes.

### Correctness

```powershell
ctest --test-dir build-moe-static -C Release -L moe-offload --output-on-failure
```

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tests\moe-offload\test-golden-logits.ps1 `
  -Bin "$PWD\build-moe-static\bin\Release\llama-completion.exe" `
  -Model "C:\AI\models\qwen\Qwen3.5-35B-A3B-Q4_K_M.moe.gguf" `
  -Tol 1e-3 -NPredict 8 -StreamCacheMb 4000 `
  -Prompt "Hello" -Seed 42 -Context 4096 -UBatch 8
```

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tests\moe-offload\test-llama-cli-chat.ps1 `
  -Bin "$PWD\build-moe-static\bin\Release\llama-cli.exe" `
  -Model "C:\AI\models\qwen\Qwen3.5-35B-A3B-Q4_K_M.moe.gguf" `
  -CacheMb 12000 -Predictor lru -NPredict 64
```

Repeat chat smoke with the final fast-path mode and fallback ubatch 1.

### Performance

Run comparable `llama-moe-bench` and `llama-cli` cases:

| Tool | Cache | Predictor | Prompt | Gen | Cache state | Required output |
| --- | ---: | --- | --- | ---: | --- | --- |
| `llama-moe-bench` | 12000 | lru | 256 tokens | 128 | cold | summary + CSV |
| `llama-cli` | 12000 | lru | same chat prompt class | 128 | cold | summary + CSV |
| `llama-moe-bench` | 12000 | eamc | 256 tokens | 128 | cold | summary + CSV |
| `llama-cli` | 12000 | eamc | same chat prompt class | 128 | cold | summary + CSV |

### Acceptance

- `llama-cli` effective ubatch matches the intended final-MVP policy.
- With comparable settings, CLI MoE profile metrics are within 15% of
  `llama-moe-bench` for:
  - prefill `gpu_compute`,
  - decode `gpu_compute`,
  - H2D,
  - stall,
  - predictor.
- Wall TTFT/TPOT are within 20% after excluding documented chat/console
  overhead, or the remaining overhead is measured and documented.
- No correctness regression versus Phase K.

## Documentation Requirements

Update these after fixing `llama-cli`:

- `docs/moe-offload/README.md`
  - preferred human-facing fast `llama-cli` command,
  - conservative fallback command,
  - profiling command,
  - accepted fast-path explanation,
  - explicit warning that benchmark numbers require matching cache, predictor,
    guard stack, and cache state.
- `docs/moe-offload/known-issues.md`
  - update or close the conservative ubatch caveat based on validation,
  - record any residual CLI-vs-bench gap.
- `paper/docs/implementation_plan_final_mvp_closeout_20260615_progress.md`
  - record implementation, commands, artifacts, metrics, and final status.

## Deliverables

- Code changes for CLI performance parity.
- Optional CLI/common fast-path knob if chosen.
- CLI-vs-bench performance artifacts.
- Updated MoE offload README and known issues.
- Progress report:
  `paper/docs/implementation_plan_final_mvp_closeout_20260615_progress.md`.

## Final Closeout Criteria

The final MVP closeout is complete only when:

1. Humans have a documented fast `llama-cli` command that demonstrates MoE
   offload speed in interactive chat.
2. `llama-cli` either matches `llama-moe-bench` within the accepted tolerance
   under comparable settings, or every remaining delta is measured and
   documented.
3. The conservative fallback remains available and documented.
4. Phase K correctness gates still pass.
5. README wording no longer implies benchmark speeds apply to `llama-cli`
   unless the required cache, predictor, fast-path mode, and cache state are
   also used.
