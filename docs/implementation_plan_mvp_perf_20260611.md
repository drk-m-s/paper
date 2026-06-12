# MoE Offload Performance Plan (2026-06-11)

Repository under work: `C:\code\llama.cpp.offload`.

This plan follows the MVP state where `llama-cli` and the benchmark path can
run streaming MoE offload. It focuses on improving prefill and decode
performance without relaxing the current correctness gates.

## Inputs Reviewed

- `docs/moe-offload/README.md`
- `docs/moe-offload/known-issues.md`
- `paper/docs/implementation_plan_raw_20260526.md`
- `paper/docs/implementation_plan_mvp_20260609.md`
- `paper/docs/implementation_plan_mvp_20260609_progress.md`
- `paper/docs/implementation_plan_mvp_20260609_llama_cli_moe_chat_progress.md`
- `moe-summary.txt`
- `moe-profile.csv`
- Relevant runtime code in:
  - `src/moe-offload/predictor.cpp`
  - `src/moe-offload/slot_pool.cpp`
  - `src/moe-offload/io.cpp`
  - `src/moe-offload/profiler.cpp`
  - `tools/llama-bench/llama-bench.cpp`
  - `tools/moe-bench/main.cpp`
  - `ggml/src/ggml-cuda/ggml-cuda.cu`

## Current Profile

The current `moe-summary.txt` reports:

```text
model: qwen35moe 35B.A3B Q4_K - Medium
predictor: eamc
cache: 8000 MB
n_prompt: 256
n_gen: 256
repeats: 3
ubatch: requested=512 effective=8
slots=96/256
mode=streaming

prefill: 33646.2 ms, 8 tok/s
decode: 1839318.6 ms, 7184.84 ms/token
```

CSV aggregate across all repeats:

| Phase | Rows | Hit rate | Misses | SSD read | H2D | Compute | Stall | Predictor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prefill | 3840 | 80.1% | 48800 | 33.15 s | 14.89 s | 18.28 s | 27.26 s | 24.40 s |
| decode | 30720 | 88.8% | 27594 | 13.46 s | 33.47 s | 94.45 s | 41.64 s | 102.61 s |

Decode means per generated token across 3 repeats:

| Metric | Mean |
| --- | ---: |
| misses | 35.9 experts/token |
| SSD read | 17.53 ms/token |
| H2D | 43.59 ms/token |
| compute | 122.98 ms/token |
| stall | 54.22 ms/token |
| predictor row time | 133.61 ms/token |
| visible CSV columns total | 371.91 ms/token |
| summary wall TPOT | 7184.84 ms/token |

The wall TPOT is far larger than the CSV columns. That gap is a diagnosis
target, not noise.

## Diagnosis

### 1. EAMC Finalization and Persistence Are the First Decode Bottleneck

The current run used `--moe-predictor eamc` with the default sidecar:

```text
C:\AI\models\qwen\Qwen3.5-35B-A3B-Q4_K_M.eamc
```

The file is already at full EAMC capacity:

```text
size: 41943076 bytes
row bytes: 40 layers * 256 experts * sizeof(float) = 40960
rows: 1024
header: 36 bytes
```

Two implementation details make this extremely expensive:

1. `llama_decode()` calls `llama_moe::begin_request()` and
   `llama_moe::end_request()` for every prefill/decode batch.
2. `slot_pool_end_request()` calls `s.pred->end_request()` and then saves the
   EAMC sidecar on every batch when `opts.eamc_path` is set.

For EAMC, `end_request()` appends the current activation row. Once the corpus
is full, it calls `evict_redundant()`, which is O(capacity^2 * row_size). With
capacity 1024 and row size 10240 floats, this can dominate every token.

The code also saves the 40 MiB sidecar after every token. This write cost and
the EAMC `end_request()` cost are not represented by per-layer CSV `pred_us`.
This explains why the CSV-visible decode work is about 372 ms/token while the
summary wall TPOT is about 7185 ms/token.

There is also a lifecycle bug: `configure_slot_pool()` calls
`s.pred->begin_request()` once, but `llama_moe::begin_request()` does not reset
the slot-pool predictor state before each `llama_decode()`. EAMC `current`
therefore accumulates across batches until process teardown instead of
representing one request window.

### 2. EAMC Scoring Is Also Expensive Inside the CSV

Even before the hidden `end_request()`/save cost, decode spends 102.61 s in
per-layer predictor row time across the run, or 133.61 ms/token.

Rows with no miss have near-zero `pred_us`. Rows with misses average:

```text
avg miss row: 1.81 misses
avg predictor: 6.73 ms/layer
avg stall: 2.73 ms/layer
```

The cause is `eamc_predictor::nearest_neighbors()`. It recomputes dense partial
cosines over up to 1024 corpus rows, scanning `0..current_layer` and all 256
experts per layer. That is repeated once per layer when eviction scoring first
needs a score.

The current cache avoids recomputing nearest-neighbor ranking for every
candidate expert inside one callback, but it still recomputes a dense ranking
for many callbacks.

### 3. H2D and Stall Are Larger Than SSD Read Time

Decode reads about 48.84 GiB from SSD across 768 generated tokens:

```text
65.12 MiB/token
SSD read: 17.53 ms/token
H2D: 43.59 ms/token
stall: 54.22 ms/token
```

The SSD path is not the only problem. H2D throughput is roughly 1.5 GiB/s from
the profile, which is far below the 8 GB/s target link budget. The current
callback does:

1. worker reads into pinned memory,
2. worker queues `cudaMemcpyAsync` on the MoE H2D stream,
3. callback inserts `cudaStreamWaitEvent()` on the compute stream,
4. callback immediately calls `io_event_sync()` before releasing the pinned
   buffer.

Step 4 preserves buffer lifetime but also blocks the host on every H2D event,
reducing overlap and inflating `stall_us`.

### 4. CUDA Compute Is Still Correctness-First

Known issues document two intentional performance guards:

- CUDA top-k MoE fusion is disabled for `LLAMA_MOE_OFFLOAD`.
- CUDA `MUL_MAT_ID` on `.slot` tensors bypasses specialized MMVQ/MMQ/MMF
  paths and uses the generic sorted path.

The profile shows decode `compute_us` at 122.98 ms/token. Restoring the
specialized decode path after targeted validation is a major performance
opportunity, but it must come after the EAMC lifecycle fix and H2D overlap fix
so correctness regressions are easier to isolate.

### 5. Prefill Is Slot-Budget Limited at 8 GiB

The benchmark requested `n_ubatch=512`, but streaming auto-sizing lowered the
effective value to 8:

```text
slots=96
top_k=8
effective ubatch=8
```

At this cache size, requesting 512 cannot improve prefill. Larger effective
ubatches need either:

- more expert-cache VRAM, for example 12000 MiB gives more slots and can reach
  effective ubatch 16 in the current policy, or
- an internal layer-wise sub-batching strategy that can process a larger
  prompt microbatch while keeping each MoE callback's unique expert set within
  the slot budget.

The first prefill callback is also a cold-start worst case:

```text
token_idx=0 prefill group: 2560 misses across 40 layers
```

The benchmark repeats do not reset the MoE slot cache between repeats, so
current summary numbers mix cold-cache and warm-cache behavior. Future
benchmark output should report those separately.

### 6. Batched Chat Prefill Is Still a Correctness Risk

`docs/moe-offload/known-issues.md` records that batched streaming prefill can
corrupt Qwen chat logits. `llama-cli --moe-offload` therefore defaults to
`n_ubatch=1` unless `LLAMA_MOE_STREAMING_UBATCH` is explicitly set.

Performance work can use `llama-moe-bench` and `llama-completion` with larger
ubatches, but any change that affects chat prefill must pass the chat smoke
and golden-logit gates before changing the `llama-cli` default.

## Goals

1. Remove hidden per-token EAMC persistence and expensive full-corpus eviction
   cost from inference while keeping online EAMC updates.
2. Make the profiler account for predictor finalization, sidecar writes, and
   unattributed wall time.
3. Reduce EAMC scoring overhead enough that it can be compared fairly with
   LRU.
4. Improve H2D overlap by preserving pinned-buffer lifetime without host
   synchronizing every copy.
5. Restore safe CUDA `.slot` fast paths behind validation gates.
6. Separate cold-cache and warm-cache prefill measurements.

## Non-Goals

- Do not add CPU DRAM expert tier in this plan.
- Do not add direct I/O in this plan.
- Do not add multi-GPU support in this plan.
- Do not add speculative decoding in this plan.
- Do not replace EAMC with a new learned predictor in this plan.
- Do not change `llama-cli` back to batched prefill until the known correctness
  issue is closed.

## Phase A - Profiler and Benchmark Calibration

Add enough instrumentation to make wall time explainable.

### Changes

1. Add request-level timing fields around `llama_moe::end_request()`:
   - `predictor_end_us`
   - `predictor_save_us`
   - `sidecar_write_bytes`
   - `profile_flush_us`

2. Add callback-level fields:
   - `callback_wall_us`
   - `topk_d2h_us`
   - `slot_ids_h2d_us`
   - `slot_table_h2d_us`
   - split `pred_us` into `pred_observe_us` and `pred_score_us`

3. Add summary gap reporting:

```text
wall_decode_us
profiled_decode_us
unattributed_decode_us
```

4. Add repeat/request identifiers to the CSV:
   - `request_idx`
   - `repeat_idx`
   - `batch_idx`

5. Add benchmark controls:
   - `--moe-reset-cache-between-repeats`
   - `--moe-warm-cache`
   - explicit summary labels for cold vs warm prefill

### Acceptance

- The current EAMC run shows the missing wall time under
  `predictor_end_us`, `predictor_save_us`, or `unattributed_decode_us`.
- A 1-token decode profile can be reconciled to within 10% of wall time.
- CSV rows can be grouped by benchmark repeat without inferring from
  `token_idx`.

## Phase B - Keep EAMC Online, Defer Persistence

Status: implemented and validated on 2026-06-11. The 8000 MiB EAMC benchmark
improved decode TPOT from 7184.84 ms/token to 162.66 ms/token, with
`predictor_save_us=0` and `sidecar_write_bytes=0` on all per-token decode
request rows. One sidecar save occurred at benchmark end.

EAMC should keep updating online during inference so it can preserve any cache
hit-rate benefit from recent expert-activation history. The performance bug is
that online update, corpus eviction, and persistent sidecar save are currently
all tied to the per-token hot path. Phase B separates those concerns:

- update EAMC in DRAM during inference,
- use cheap bounded in-memory corpus management,
- save the sidecar only at logical user request end or session/context end,
- never rewrite the sidecar after every generated token.

### Changes

1. Add a predictor lifecycle call from `llama_moe::begin_request()` into the
   slot pool so EAMC's current activation row is reset at the start of each
   `llama_decode()` request. This fixes the current lifecycle bug where the
   row can accumulate across batches.

2. Keep `--moe-predictor eamc` semantically online:
   - loaded sidecar rows are available for scoring,
   - new request rows are appended to an in-memory corpus,
   - current-request observations can influence later eviction decisions,
   - no user-facing read-only/update mode is added in this phase.

3. Stop saving the EAMC sidecar from every `slot_pool_end_request()`.
   Replace it with an explicit flush point:
   - `llama-moe-bench` flushes once after all measured repeats complete,
   - normal runtime flushes when the context/session is destroyed,
   - server/chat integrations can call the same flush helper at logical user
     request end if they want persistence before session teardown.

   In this phase, "request end" means the end of a logical generation/chat
   request, not the end of each internal `llama_decode()` batch.

4. Add a narrow API:

```cpp
bool llama_moe::flush_predictor();
```

   Behavior:
   - no-op for LRU,
   - for EAMC, save the current in-memory corpus to `opts.eamc_path`,
   - return success/failure so benchmark tools can report save errors.

5. Replace online `evict_redundant()` with a cheap bounded policy:
   - use FIFO/ring replacement when corpus capacity is full,
   - keep the dense sidecar file format initially for compatibility,
   - move expensive redundancy pruning to a future offline compaction tool.

6. Keep Phase A request metrics:
   - `predictor_end_us` should include cheap in-memory row append only,
   - `predictor_save_us` should be zero for per-token decode requests,
   - sidecar bytes should be written only at explicit flush.

7. Add tests:
   - EAMC appends online rows in memory across decode requests.
   - Full-capacity EAMC uses ring/FIFO replacement without O(capacity^2)
     redundancy pruning.
   - A decode request with `--moe-predictor eamc --moe-eamc-path PATH` does
     not change `PATH` mtime.
   - `flush_predictor()` writes `PATH` and updates its mtime.
   - `llama-moe-bench` saves EAMC once at benchmark end when an EAMC path is
     configured.

### Acceptance

- With the same 8000 MiB cache and sidecar, decode TPOT drops from seconds per
  token to the range explained by per-layer CSV timers.
- Sidecar file mtime is unchanged during token generation and changes only at
  explicit flush/session end.
- `predictor_save_us` is zero on per-token decode request rows.
- `predictor_end_us` is bounded and no longer runs O(capacity^2) redundancy
  pruning.
- EAMC hit rate is compared before/after Phase B to verify that online DRAM
  updates preserve or improve cache behavior.

## Phase C - Optimize EAMC Scoring

After Phase B, EAMC scoring still costs about 90.54 ms/token in the 8000 MiB,
256 prompt, 256 decode benchmark. Make it cheap enough to be a real predictor
option without reducing EAMC cache hit rate.

Phase C must be semantics-preserving first. The current EAMC scoring behavior
is slow, but it is also the reason EAMC beats the old run's hit rate. Do not
start by shrinking or approximating the corpus. First make the same scoring
cheaper; only then test approximate row caps as an opt-in experiment.

Current measured cost:

- prefill EAMC score time: about 25.9 s across the 256-token, 3-repeat run,
- decode EAMC score time: about 69.5 s across the 256-token, 3-repeat run,
- decode predictor cost: 90.54 ms/token,
- `pred_observe_us` and `predictor_end_us` are already negligible.

Important prefill detail: one outer prefill `llama_decode()` contains multiple
internal MoE callback waves because effective ubatch is 8. The callback order is
`L0..L39` for token index 0, then `L0..L39` for token index 8, and so on. Any
incremental EAMC state must handle this non-monotonic layer order instead of
assuming layers only advance once from 0 to the final layer.

### Changes

#### C1 - Sparse Representation With Dense-Equivalent Scores

1. Store activation rows sparsely while preserving the existing scoring
   semantics:
   - per layer, only activated expert ids and counts,
   - counts remain callback-observation counts, not per-token frequencies,
   - loaded dense sidecar rows are converted to sparse rows once at load time,
   - saving still writes the existing dense sidecar format for compatibility.

2. Precompute sparse row metadata:
   - full row norm,
   - per-layer squared norm contribution,
   - prefix norm for `0..L` lookup,
   - optional per-layer sorted expert/count vectors for deterministic tests.

3. Keep dense-equivalent cosine behavior for the uncapped path:
   - for a prefix ending at layer `L`, compute dot products only over current
     sparse activations and matching corpus layer entries,
   - use corpus prefix norms equivalent to the dense implementation's
     `cosine_partial()`,
   - preserve zero-weight fallback to LRU recency when EAMC weights are zero.

4. Add tests comparing old dense scoring to new sparse scoring:
   - synthetic deterministic corpora,
   - repeated observations in the same layer,
   - partial-prefix scoring,
   - prefill-like non-monotonic callback order: `L0..L39`, then `L0..L39`
     again in the same request,
   - loaded sidecar -> sparse conversion -> save dense sidecar round trip.

#### C2 - Lazy Per-Callback Score Materialization

5. Do not compute EAMC scores on callbacks that do not evict. In the slot-pool
   flow, `score()` is only needed while selecting a victim for a miss. Keep
   that property.

6. On first `score(layer, expert)` after an `observe()` for that callback,
   lazily compute and cache:
   - nearest-neighbor rows for the current prefix,
   - one `score_for_expert[expert]` vector for the requested layer,
   - metadata counters for rows scanned and materialization cost.

7. Eviction scans then read `score_for_expert[exp]` in O(1):

```text
score_for_expert[expert] = weighted average over top EAMC neighbors
```

8. Cache invalidation rules:
   - `begin_request()` clears current sparse activations and score caches,
   - `observe(layer, experts)` invalidates only prefix/score caches affected by
     the new observation,
   - `end_request()` appends the sparse row and invalidates corpus-dependent
     caches,
   - prefill callback waves must not reuse a score vector computed before a
     later observation changed the request row.

#### C3 - Optional Corpus Row Cap, Hit-Rate Gated

9. Add row caps only after C1/C2 pass:
   - default uncapped behavior should continue using all loaded corpus rows,
   - add `LLAMA_MOE_EAMC_ROWS=N` as a diagnostic override,
   - row selection must use newest ring/FIFO rows, not arbitrary vector order,
   - nearest-neighbor count remains the sidecar/default value unless a
     separate diagnostic override is added.

10. A row cap cannot become the default unless it does not hurt hit rate:
    - compare uncapped EAMC, capped EAMC, and LRU on identical cache/reset
      settings,
    - require decode hit rate within 0.5 percentage points of uncapped EAMC,
    - require prefill hit rate not worse than uncapped EAMC by more than 0.5
      percentage points,
    - if capped EAMC is faster but hit rate regresses, keep it diagnostic only.

11. Add profile counters:
   - `eamc_rows_scored`
   - `eamc_cosine_us`
   - `eamc_score_materialize_us`
   - `eamc_score_cache_hits`
   - `eamc_score_cache_misses`

### Acceptance

- EAMC online predictor row time is below 10 ms/token on the 8000 MiB,
  256 prompt, 256 decode benchmark.
- Uncapped sparse/incremental EAMC hit rates match Phase B uncapped EAMC within
  0.5 percentage points:
  - prefill baseline: 81.9%,
  - decode baseline: 90.3%.
- TPOT improves materially without increasing misses/token; target decode TPOT
  is below 100 ms/token before Phase D/E work.
- New dense-equivalence unit tests pass before row-cap experiments are enabled.
- Row caps remain diagnostic unless they preserve hit rate under the gates
  above.
- EAMC and LRU are compared on identical cache/reset settings after C1/C2. If
  optimized uncapped EAMC still does not beat LRU on TPOT or hit rate, make LRU
  the documented performance default and keep EAMC experimental.

## Phase D - Remove Host Synchronization From H2D Buffer Lifetime

Status: implemented and benchmarked on 2026-06-12. The normal miss-completion
path no longer synchronizes the host on each H2D event before recycling pinned
memory; pinned buffers are released by polling the H2D end event. This reduced
decode `stall_us` from about 16.8 ms/token to 0.41 ms/token on the 8000 MiB
EAMC benchmark. End-to-end TPOT did not improve in that run (57.92 -> 62.13
ms/token versus the latest root summary) because `ssd_read_us`, callback wall,
and predictor time rose. Treat Phase D as an overlap/profiler-correctness fix,
not yet a throughput win. See
`implementation_plan_mvp_perf_20260611_progress.md` for artifacts and metrics.

Current async H2D is ordered correctly but the host waits for every event
before recycling pinned memory. Replace this with asynchronous lifetime
tracking.

### Changes

1. Add an event query helper:

```cpp
bool io_event_query(void * ev);
```

2. Keep completed H2D buffers in an `inflight_h2d` list:
   - pinned buffer pointer,
   - begin/end events,
   - destination metadata for diagnostics.

3. Release a pinned buffer only after its H2D end event has completed.
   Poll completed events at:
   - start of each callback,
   - after draining worker completions,
   - before blocking for a free pinned buffer,
   - request end.

4. Only block the host when:
   - no pinned buffers are free,
   - queue progress is impossible, and
   - no H2D event has completed.

5. Keep `cudaStreamWaitEvent()` on the compute stream. That is the correctness
   ordering requirement; host synchronization is only a buffer lifetime issue.

6. Raise or make configurable:
   - pinned buffer count,
   - event pool soft cap,
   - worker queue capacity.

7. Sort miss blobs by file offset before submission. If adjacent blob ranges
   are contiguous or close, optionally coalesce reads into one pinned buffer
   and issue multiple H2D copies from slices.

### Acceptance

- `stall_us` falls materially on decode miss rows.
- H2D effective throughput improves from about 1.5 GiB/s toward the Oculink
  link budget.
- `test-cuda-stream` covers async buffer lifetime without immediate
  `cudaEventSynchronize()`.
- Golden-logit gates still pass.

## Phase E - Restore Safe CUDA Slot Fast Paths

Once host-side overhead is controlled, attack `compute_us`.

### Changes

1. Add a synthetic `.slot` `MUL_MAT_ID` CUDA test:
   - random slot tensors,
   - random slot id tensors,
   - compare generic sorted path vs specialized path,
   - cover single-token decode and multi-token prefill shapes.

2. Re-enable specialized MMVQ for `.slot` decode first, behind a guard:

```text
LLAMA_MOE_SLOT_MMVQ=1
```

3. If decode passes, test MMQ/MMF and multi-token prefill paths separately.

4. Keep CUDA graph capture disabled for `.slot` until the graph and callback
   ordering is explicitly proven safe.

5. Treat top-k MoE fusion as a separate later step:
   - first restore unfused specialized `MUL_MAT_ID`,
   - then validate GLU/MoE fusion,
   - only then revisit the disabled top-k fusion.

### Acceptance

- Golden-logit matrix passes with `LLAMA_MOE_SLOT_MMVQ=1`.
- Chat smoke remains clean with default `llama-cli` ubatch 1.
- End-to-end speed is the primary gate: prefill TTFT and decode TPOT improve
  versus the generic sorted path on the same model, sidecar, cache budget, and
  benchmark settings.
- Decode `compute_us` drops significantly versus the generic sorted path.
- Secondary metrics do not regress enough to erase the wall-clock gain:
  `h2d_us`, `stall_us`, predictor time, SSD read time, callback wall time, hit
  rate, and misses/token should stay within normal run-to-run noise, with any
  larger regression explicitly justified by a faster prefill/decode result.
- The guard can be disabled quickly if a model or shape regresses.

## Phase F - Prefill-Specific Improvements

At 8000 MiB the current slot budget forces effective ubatch 8. Prefill gains
need either more slots or a different prefill strategy.

### Changes

1. Benchmark cache/ubatch matrix after Phases B-D:

| Cache MiB | Expected slots | Expected effective ubatch |
| ---: | ---: | ---: |
| 8000 | 96 | 8 |
| 12000 | about 145 | 16 |
| 14000 | measure | 16 or 32 |
| 16000 | measure | 32 or higher if memory allows |

2. Add cold/warm prefill reporting:
   - first prefill after empty slot cache,
   - subsequent prefill with warm cache,
   - optional reset between repeats.

3. Add optional hot-start expert loading:
   - load top experts per layer from an offline profile or EAMC sidecar,
   - fill only available slot budget,
   - measure TTFT impact separately from decode.

4. Investigate router-aware internal prefill splitting:
   - accept a larger prompt ubatch,
   - split one MoE layer's selected tokens/experts into chunks whose unique
     expert count fits `n_slots`,
   - avoid changing the user-visible ubatch or context API.

5. Do not change `llama-cli` default prefill above 1 until the known batched
   chat-logit issue is closed.

### Acceptance

- Benchmark docs show which cache budget is needed for effective ubatch 16+
  on the 16 GiB target GPU.
- Cold and warm TTFT are reported separately.
- Any hot-start feature improves cold TTFT without hurting decode correctness.
- Batched prefill changes pass golden logits and chat smoke before becoming
  defaults.

## Phase G - Validation Matrix

Run after each performance phase, not only at the end.

### Correctness

```powershell
ctest --test-dir build-moe -C Release -L moe-offload --output-on-failure
```

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tests\moe-offload\test-golden-logits.ps1 `
  -Model "C:\AI\models\qwen\Qwen3.5-35B-A3B-Q4_K_M.moe.gguf" `
  -Tol 1e-3 -NPredict 8 -StreamCacheMb 4000 `
  -Prompt "Hello" -Seed 42 -Context 4096 -UBatch 8
```

Also rerun the existing ubatch matrix:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tests\moe-offload\test-streaming-ubatch-matrix.ps1 `
  -StreamCacheMb 4000,8000,12000 `
  -UBatch 8,16,32,64
```

Chat smoke:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tests\moe-offload\test-llama-cli-chat.ps1 `
  -Model "C:\AI\models\qwen\Qwen3.5-35B-A3B-Q4_K_M.moe.gguf" `
  -CacheMb 8000 -Predictor lru -NPredict 96
```

### Performance

Run LRU and EAMC online deferred-save with identical settings. The commands
below use
`llama-moe-bench` because it is the most controlled MoE-specific benchmark
surface; the same profiler fields and diagnosis apply to `llama-bench` runs
that emit `moe-profile.csv`.

```powershell
.\build-moe\bin\Release\llama-moe-bench.exe `
  --model C:/AI/models/qwen/Qwen3.5-35B-A3B-Q4_K_M.moe.gguf `
  --pp 256 --tg 256 --repeat 3 `
  --moe-cache-vram-mb 8000 `
  --moe-predictor lru `
  --moe-profile-csv tests\moe-offload\_out\perf-lru-8gb.csv `
  --moe-profile-summary tests\moe-offload\_out\perf-lru-8gb.summary.txt `
  -ngl 99 -c 4096 -ub 512
```

```powershell
.\build-moe\bin\Release\llama-moe-bench.exe `
  --model C:/AI/models/qwen/Qwen3.5-35B-A3B-Q4_K_M.moe.gguf `
  --pp 256 --tg 256 --repeat 3 `
  --moe-cache-vram-mb 8000 `
  --moe-predictor eamc `
  --moe-eamc-path C:/AI/models/qwen/Qwen3.5-35B-A3B-Q4_K_M.eamc `
  --moe-profile-csv tests\moe-offload\_out\perf-eamc-online-8gb.csv `
  --moe-profile-summary tests\moe-offload\_out\perf-eamc-online-8gb.summary.txt `
  -ngl 99 -c 4096 -ub 512
```

Repeat for `--moe-cache-vram-mb 12000` and `-ub 16`/`-ub 32`.

### Metrics to Compare

- TTFT cold
- TTFT warm
- TPOT
- decode hit rate
- prefill hit rate
- SSD bytes/token
- H2D ms/token
- stall ms/token
- compute ms/token
- predictor observe/score/end/save ms/token
- sidecar bytes written
- unattributed wall time
- peak VRAM

## Recommended Order

1. Phase A: add missing profiler buckets.
2. Phase B: keep EAMC online in memory and remove per-token sidecar save.
3. Run an LRU baseline immediately after Phase B.
4. Phase C: optimize EAMC scoring only if EAMC still looks useful versus LRU.
5. Phase D: fix H2D buffer lifetime to recover overlap.
6. Phase E: re-enable CUDA `.slot` fast paths one at a time.
7. Phase F: tune prefill using cache budget, cold/warm reporting, and optional
   hot-start.

The expected biggest immediate win is Phase B. Until the EAMC lifecycle is
fixed, decode wall time is dominated by work that the current CSV does not
attribute to any layer.
