# MoE Offload Progress: `llama-cli` Chat Frontend (2026-06-09)

Repository under work: `C:\code\llama.cpp.offload`.

## Summary

Implemented the first MVP pass for using `llama-cli` as the supported
interactive chat frontend with MoE offload enabled.

## Changes

- Built and validated `llama-cli` against the MoE-enabled common argument path.
- Gated the old default `[moe-d3]` and `[moe-d4]` slot-pool traces behind:
  - `LLAMA_MOE_DEBUG_D4`
  - `LLAMA_MOE_DEBUG_LOADS`
  - `LLAMA_MOE_DEBUG_TRACE`
- Added a dev-box PowerShell smoke harness:
  `tests\moe-offload\test-llama-cli-chat.ps1`.
- The smoke runs:
  - `llama-cli --moe-offload`
  - `--moe-cache-vram-mb 8000`
  - `--moe-predictor lru`
  - `--jinja`
  - `--reasoning-budget 0`
  - deterministic sampling with `--temp 0 --seed 42`
- The smoke asserts:
  - process exit code is 0,
  - stdout contains `Paris`,
  - stdout/stderr contain no default `[moe-d3]` or `[moe-d4]` trace spam.
- Fixed an I/O worker/callback synchronization bug found by the smoke:
  - failed seek/read requests are now returned as failed completions instead
    of being silently consumed by the worker,
  - the outstanding-request counter now uses release/acquire ordering so the
    callback does not observe `outstanding == 0` before the completion list is
    visible.
- Updated MoE docs:
  - `docs\moe-offload\README.md`
  - `docs\moe-offload\known-issues.md`
  - `tests\moe-offload\README.md`

## Validation

Commands run:

```powershell
cmake --build build-moe\tools\cli --config Release --target llama-cli
cmake --build build-moe --config Release --target llama-completion llama-moe-bench
ctest --test-dir build-moe -C Release -L moe-offload --output-on-failure
powershell -NoProfile -ExecutionPolicy Bypass -File tests\moe-offload\test-llama-cli-chat.ps1 `
  -Model "C:\AI\models\qwen\Qwen3.5-35B-A3B-Q4_K_M.moe.gguf" `
  -CacheMb 8000 -Predictor lru -NPredict 96 -Seed 42
```

Results:

- `llama-cli`: build passed.
- `llama-completion` and `llama-moe-bench`: build passed.
- CTest label `moe-offload`: 6/6 passed.
- `test-llama-cli-chat.ps1`: passed.
- `llama-cli --help`: MoE flags and chat-template/reasoning flags are listed.

The passing smoke output contained:

```text
It seems like there might be a small typo in your question. If you're asking
about the capital of France, the answer is Paris.
```

## Notes

- `llama-cli` is now the documented MoE chat frontend.
- `llama-completion -no-cnv` remains the preferred raw-completion and logit
  diagnostic frontend.
- For this Qwen3.5 GGUF, keep `--jinja` enabled.
- `--reasoning off` is the validated concise-chat path. `--reasoning-budget 0`
  can still expose reasoning text for this Qwen3.5 GGUF.

## Addendum: Chat Gibberish Fix

Follow-up testing found that the previous `--reasoning-budget 0` README command
was not robust: simple prompts could produce visible reasoning text or repeated
role/template fragments. The offload-disabled CPU baseline answered cleanly,
while MoE offload answered cleanly only when streaming prefill was reduced to
single-token ubatches.

Key findings:

- `llama-cli` with the `.moe.gguf` and no `--moe-offload` answered `who are
  you?` cleanly.
- `llama-cli --moe-offload --moe-cache-vram-mb 8000` with default batched
  streaming prefill produced `assistant` / `<think>` fragments.
- `LLAMA_MOE_STREAMING_UBATCH=1` made the offloaded path answer cleanly for:
  - `who are you?`
  - `what is the capital of France?`
  - `what is 1 + 1?`
- `LLAMA_MOE_STREAMING_UBATCH=4` could leak extra `</think>` text.
- `LLAMA_MOE_STREAMING_UBATCH=8` could emit reasoning text instead of the final
  answer.
- `LLAMA_MOE_DEBUG_NO_ASYNC=1`, `LLAMA_MOE_DEBUG_NO_HIT=1`,
  `LLAMA_MOE_DEBUG_ID_FILL=1`, and `LLAMA_MOE_DEBUG_SYNC=1` did not fix the
  corruption.

Implemented mitigation:

- `llama-cli --moe-offload` now sets `params.n_ubatch=1` by default unless
  `LLAMA_MOE_STREAMING_UBATCH` is explicitly set.
- `llama-cli` now honors `params.reasoning_format` instead of forcing
  `COMMON_REASONING_FORMAT_DEEPSEEK`.
- The README command now uses `--reasoning off`.
- The dev-box chat smoke now checks identity, France, and arithmetic prompts
  and fails if thinking text leaks into the transcript.
