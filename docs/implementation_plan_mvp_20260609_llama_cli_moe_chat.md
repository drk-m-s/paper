# MoE Offload Plan: Enable `llama-cli` Chat Frontend (2026-06-09)

Repository under work: `C:\code\llama.cpp.offload`.

## Summary

The next MVP step is to make `llama-cli` the supported interactive chat
frontend for MoE-offloaded models. `llama-completion` is useful for raw
completion and diagnostic runs, but its conversation mode currently has two
problems for this fork:

- chat-template behavior is easy to misuse, especially for Qwen3.5 where
  `--jinja` matters,
- unconditional `[moe-d4]` debug prints interleave with generated text in an
  interactive terminal.

`llama-cli` already parses common arguments and uses `common_init_from_params`,
so the MoE model-load path and `--moe-offload` options should already be
reachable. This plan is about proving that path, fixing chat-specific rough
edges, and documenting the supported invocation.

## Goal

Enable a user to run:

```powershell
.\build-moe\bin\Release\llama-cli.exe `
  --model C:/AI/models/qwen/Qwen3.5-35B-A3B-Q4_K_M.moe.gguf `
  --moe-offload `
  --moe-cache-vram-mb 8000 `
  --moe-predictor lru `
  --jinja `
  -sys "You are a helpful assistant."
```

and chat interactively with correct answers and a clean terminal transcript.

## Current State

Relevant facts from the codebase:

- `common/arg.cpp` registers `--moe-offload`, `--moe-cache-vram-mb`,
  `--moe-cache-vram-frac`, `--moe-predictor`, `--moe-eamc-path`,
  `--moe-profile-csv`, and `--moe-profile-summary` under
  `LLAMA_MOE_OFFLOAD`.
- `common/common.cpp` forwards those fields into `llama_model_params`.
- `tools/cli/cli.cpp` uses `common_params_parse()` and
  `server_context::load_model(params)`, so it should get the MoE load path
  through common initialization.
- `llama-cli` is always a chat frontend and rejects `--no-conversation`.
- `llama-cli` already uses the common chat-template stack and passes
  `params.use_jinja` into chat formatting.
- The MoE eval callback is installed inside `llama_context`, so it should work
  for `llama-cli` once the model is loaded with MoE offload enabled.

Open risks:

- `llama-cli` has not yet been explicitly built and smoke-tested with
  `--moe-offload`.
- The MoE runtime still emits unconditional `[moe-d4]` `fprintf(stderr, ...)`
  diagnostics from `slot_pool.cpp`.
- Qwen3.5 chat behavior is sensitive to using the Jinja template path.
- The server-style request loop used by `llama-cli` may exercise request
  lifecycle paths differently from `llama-completion`.

## Non-Goals

- Do not rewrite `llama-cli` into a separate MoE-specific chat frontend.
- Do not change raw `llama-completion` behavior except for shared MoE logging
  cleanup.
- Do not re-enable CUDA top-k MoE fusion or specialized `.slot` `MUL_MAT_ID`
  paths in this plan.
- Do not add multi-GPU, CPU DRAM expert tier, direct I/O, or learned
  predictors.

## Implementation Plan

### Phase A - Establish the Baseline

1. Build `llama-cli` in the MoE build:

   ```powershell
   cmake --build build-moe --config Release --target llama-cli
   ```

2. Confirm `llama-cli --help` exposes the MoE options when
   `LLAMA_MOE_OFFLOAD=ON`.

3. Run a load-only or short prompt smoke with:

   ```powershell
   .\build-moe\bin\Release\llama-cli.exe `
     --model C:/AI/models/qwen/Qwen3.5-35B-A3B-Q4_K_M.moe.gguf `
     --moe-offload `
     --moe-cache-vram-mb 8000 `
     --moe-predictor lru `
     --jinja `
     -sys "You are a helpful assistant." `
     -p "what is the capital of France?"
   ```

4. Record whether model load, context creation, streaming ubatch auto-sizing,
   and request shutdown all complete without hangs.

Acceptance:

- `llama-cli` loads the repacked `.moe.gguf`.
- Startup log shows MoE slot pool configuration and streaming ubatch sizing.
- The first answer contains `Paris` for the France prompt.

### Phase B - Make MoE Logs Interactive-Safe

1. Audit unconditional debug output in `src/moe-offload/slot_pool.cpp`.

2. Convert `[moe-d4]` diagnostics to one of:

   - `LLAMA_LOG_DEBUG` / `LLAMA_LOG_INFO` with normal verbosity control, or
   - environment-gated `fprintf(stderr, ...)` behind an explicit debug flag.

3. Keep only true errors and warnings visible by default.

4. Preserve useful debug flags:

   - `LLAMA_MOE_DEBUG_SLOT_IDS`
   - `LLAMA_MOE_DEBUG_EVICT`
   - `LLAMA_MOE_DEBUG_SLOT_HASH`
   - existing targeted diagnostics

Acceptance:

- A normal `llama-cli --moe-offload` chat does not interleave `[moe-d4]`
  counters or tensor-allocation traces into the visible terminal session.
- Debug output can still be recovered with explicit environment flags.

### Phase C - Verify Chat Template Behavior

1. Test `llama-cli` with Qwen3.5 using `--jinja`.

2. Test the same command without `--jinja` and document whether it is unsafe
   for this model.

3. If `--jinja` is required for correct Qwen3.5 chat, update the MoE docs with
   the supported command and rationale.

4. Do not silently force Jinja globally unless upstream common behavior already
   supports a model/template capability check. A global default change could
   alter unrelated models.

Acceptance:

- The documented Qwen3.5 command includes `--jinja`.
- The docs do not recommend `-p "Hello"` as a system prompt.
- The recommended path uses `-sys` for system instructions.

### Phase D - Request Lifecycle and Predictor Hygiene

`llama-cli` uses the server-style request loop, so validate MoE request
boundaries:

1. Confirm `llama_moe::begin_request()` / `end_request()` happen per chat turn
   or at least leave predictor and profile state coherent.

2. Confirm `slot_pool_shutdown_io()` runs on exit.

3. Confirm Ctrl+C, `/exit`, and `/clear` do not leave the I/O worker running.

4. Check EAMC sidecar persistence with:

   ```powershell
   --moe-predictor eamc `
   --moe-eamc-path C:/AI/models/qwen/Qwen3.5-35B-A3B-Q4_K_M.eamc
   ```

Acceptance:

- Repeated chat turns do not accumulate stale request state.
- `/clear` resets chat history without corrupting MoE slot-cache state.
- `/exit` stops the async I/O worker cleanly.

### Phase E - Add Focused Validation

Add or document a lightweight smoke path that can run without manual typing.
Possible approach:

```powershell
"what is the capital of France?`n/exit`n" | .\build-moe\bin\Release\llama-cli.exe `
  --model C:/AI/models/qwen/Qwen3.5-35B-A3B-Q4_K_M.moe.gguf `
  --moe-offload `
  --moe-cache-vram-mb 8000 `
  --moe-predictor lru `
  --jinja `
  -sys "You are a helpful assistant." `
  --no-warmup `
  -c 4096 `
  -n 64
```

If stdin piping does not interact cleanly with the console abstraction, add a
small PowerShell harness under `tests/moe-offload/` that drives a process and
captures stdout/stderr.

Acceptance:

- The smoke captures the assistant response.
- The response contains `Paris`.
- `stderr` has no default `[moe-d4]` spam after Phase B.
- The process exits cleanly.

## Documentation Updates

Update:

- `docs/moe-offload/README.md`
- `docs/moe-offload/known-issues.md`
- `tests/moe-offload/README.md` if a harness is added
- a new progress report in `paper/docs/` after implementation

Recommended README section:

```powershell
.\build-moe\bin\Release\llama-cli.exe `
  --model C:/AI/models/qwen/Qwen3.5-35B-A3B-Q4_K_M.moe.gguf `
  --moe-offload `
  --moe-cache-vram-mb 8000 `
  --moe-predictor eamc `
  --jinja `
  -sys "You are a helpful assistant."
```

Mention:

- use `llama-cli` for chat,
- use `llama-completion -no-cnv` for raw completion diagnostics,
- use `--jinja` for this Qwen3.5 GGUF,
- use `-sys` for system prompts,
- avoid using `-p "Hello"` as a system prompt in conversation mode.

## Validation Matrix

Minimum validation:

| Case | Command shape | Expected result |
| --- | --- | --- |
| Help | `llama-cli --help` | MoE flags are listed |
| Load | `llama-cli --moe-offload --jinja` | model loads, slot pool configured |
| Fact chat | France prompt | answer contains `Paris` |
| Small cache | `--moe-cache-vram-mb 8000` | no `n_uniq exceeds n_slots` |
| Explicit ubatch | `LLAMA_MOE_STREAMING_UBATCH=8` | still answers correctly |
| EAMC | `--moe-predictor eamc` | no sidecar/load regression |
| Exit | `/exit` or Ctrl+C | I/O worker shuts down |

Optional validation:

- Compare `llama-cli` chat prompt logits between full residency and streaming
  for a short deterministic run.
- Run one session with `--moe-profile-csv` and confirm rows are populated.
- Run one session with `--moe-profile-summary` and confirm summary writes on
  clean exit.

## Success Criteria

This plan is complete when:

- `llama-cli` is the documented MoE-offload chat entry point.
- Qwen3.5 chat with `--jinja` produces sane answers in interactive use.
- Default terminal output is clean enough for chat.
- Small-cache streaming mode still auto-sizes ubatch and does not fail during
  chat turns.
- Known issues clearly distinguish remaining chat-template limitations from
  MoE slot-cache correctness issues.
