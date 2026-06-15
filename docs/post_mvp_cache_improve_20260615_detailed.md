# Detailed Post-MVP MoE Cache Improvement Plan (2026-06-15)

Implementation repository: `C:\code\llama.cpp.offload`.

Planning inputs:

- `paper/docs/post_mvp_plan_cache_improve_20260615.md`
- `paper/docs/implementation_plan_final_mvp_closeout_20260615.md`
- `paper/docs/implementation_plan_final_mvp_closeout_20260615_progress.md`
- `paper/docs/implementation_plan_raw_20260526.md`
- `llama.cpp.offload/docs/moe-offload/known-issues.md`
- `paper/papers/5-star/MoE Infinity.pdf`
- `paper/papers/5-star/FineMoE.pdf`

This plan is for the private MoE-offload fork. If any part is later proposed
upstream, the human contributor must own the design, implementation, testing,
review discussion, and disclosure requirements described in `AGENTS.md` and
`CONTRIBUTING.md`.

## Goal

Improve prefill and decode speed by increasing useful expert-cache hit rate and
by overlapping SSD read plus H2D transfers with GPU compute.

The work should remain limited to cache-management mechanisms:

- expert-cache prediction,
- expert-cache eviction,
- expert prefetching,
- sidecar and hot-start policy,
- optional CPU DRAM warm tier after the VRAM policy is measured.

Do not use this phase for learned predictors, speculative decoding, KV offload,
model architecture changes, or unrelated kernel fusion work.

## Paper-Derived Design Takeaways

### MoE-Infinity

MoE-Infinity is the lower-risk first validation target because the MVP already
claims an EAMC-style predictor and the code has EAMC sidecar plumbing.

Important mechanisms from the paper:

- The target scenario is batch-size-1 local inference with limited accelerator
  memory. That matches the intended interactive/local use case better than
  server throughput papers.
- Expert reuse is sparse inside a single request, but global reuse across many
  requests can become almost uniform. A global hot-count policy alone is not
  enough.
- The paper defines an Expert Activation Matrix, or EAM, as an `L x E` matrix
  of routed-token counts per layer and expert.
- It distinguishes iteration-level EAM from request-level EAM. The request-level
  EAM accumulates the whole prompt/decode request and is the main trace stored
  in the EAMC.
- EAMC matching flattens EAMs and uses cosine similarity to find historical
  traces similar to the current partial trace.
- The predicted EAM is built by aggregating matched historical EAMs and applying
  layer-proximity weighting so nearer future layers have higher priority.
- The paper uses the predicted EAM for both eviction and prefetching. Eviction
  chooses the resident expert with lowest predicted future reuse.
- Prefetch is conceptually integrated with on-demand fetch. The paper does not
  provide enough low-level scheduling detail for SSD-backed llama.cpp, so the
  fork needs its own guarded implementation.

Limits when applying the paper here:

- MoE-Infinity stores offloaded experts in CPU DRAM, while this fork currently
  uses SSD as the cold tier. Misprediction is more expensive here.
- The paper's cache model is closer to a global expert cache; this fork has
  per-layer slot tensors and per-layer residency. Cross-layer victim selection
  is not directly transferable.
- The paper does not solve SSD admission control, direct I/O, or host DRAM warm
  tiering.

### FineMoE

FineMoE is the stronger long-term cache-controller reference, but it is more
invasive than a MoE-Infinity-fidelity pass.

Important mechanisms from the paper:

- FineMoE argues that request-level EAMs are too coarse for many decoder-only
  MoE LLMs.
- It records iteration-level ExpertMaps: per-layer gate probability
  distributions across experts, not only selected expert IDs.
- It uses two lookup signals:
  - semantic similarity from embedding-layer outputs for early-layer prefetch,
  - trajectory similarity from already-observed expert probability maps for
    later-layer prefetch.
- It dynamically chooses how many experts to prefetch. Lower similarity means
  lower confidence and therefore a larger prefetch set; higher similarity means
  fewer speculative experts.
- Its prefetch set is formed by taking highest-probability experts until their
  cumulative probability reaches a threshold, with at least the model top-k
  experts represented.
- It prioritizes prefetch by probability and layer proximity.
- It combines predicted expert probability with visit frequency for eviction,
  and explicitly avoids LRU as the main policy because layer execution is
  sequential.
- It runs map search and prefetch asynchronously so prediction work does not sit
  on the critical path.
- On a 6x RTX 3090 system with CPU DRAM offload, FineMoE reports lower TTFT,
  lower TPOT, and higher hit rate than MoE-Infinity.

Limits when applying the paper here:

- FineMoE assumes CPU DRAM offload and high host bandwidth, not SSD as the cold
  tier.
- Capturing full gate probability distributions may require changes near the
  router/top-k graph, while the current MVP mostly consumes selected top-k IDs.
- Semantic embedding extraction is more invasive than trajectory-only expert
  maps. It should be deferred until cheaper route-history signals are measured.
- The paper evaluates a multi-GPU server/workstation. The current target is a
  single 16 GB GPU over an 8 GB/s host link, so admission control must be more
  conservative than in the paper.

## Current MVP Interpretation

The current code already has a useful offload foundation:

- `src/moe-offload/slot_pool.cpp` owns per-layer slot tensors, per-layer expert
  residency, demand loading, and eviction.
- `src/moe-offload/predictor.cpp` implements `lru` and `eamc`.
- `src/moe-offload/profiler.*` records per-layer hit/miss, SSD, H2D, compute,
  stall, callback, and predictor timings.
- `tools/moe-bench/main.cpp` supports cache reset, warm cache, hot-start, EAMC
  sidecar, CSV, and summary output.
- The final MVP fast path is validated through `--moe-fast-paths` and keeps the
  conservative fallback available.

However, the current EAMC should be treated as an EAMC-lite implementation, not
as a paper-complete MoE-Infinity implementation:

- It stores rows at the `llama_decode()` batch boundary. In `llama-moe-bench`,
  that means prefill is one row and each decode token can become one row. This
  is closer to iteration-level traces than MoE-Infinity request-level EAMs.
- `predictor::score(layer, expert)` only supports eviction scoring for one
  candidate expert. It does not expose a ranked prefetch set.
- The live cache is per-layer. The paper's layer-proximity/global victim formula
  does not transfer directly.
- The current sidecar format `EAM1` stores dense rows and uses bounded FIFO/ring
  replacement, not a uniqueness or redundancy-aware EAMC store.
- The predictor observes selected experts, not full router probability maps.
- There is no true predictive expert prefetch today. Misses are detected at the
  callback boundary, then loaded before computation can proceed.

This explains why the human 8000 MiB EAMC run can trail LRU without proving the
paper is wrong.

Observed human 8000 MiB results:

| Predictor | Prefill | Decode | Prefill hit | Decode hit | Decode SSD read | Decode predictor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LRU | 10.62 ms/token | 29.12 ms/token | 86.2% | 91.0% | 39.32 GB | 0.12 ms/token |
| EAMC | 15.70 ms/token | 33.04 ms/token | 88.3% | 88.6% | 49.70 GB | 1.48 ms/token |

Interpretation:

- EAMC had slightly better prefill hit rate but much higher prefill predictor
  cost.
- EAMC had worse decode hit rate than LRU, causing more SSD reads and H2D.
- EAMC's predictor cost was also higher in decode.
- Since the current EAMC does not prefetch, any inaccurate eviction decision is
  paid as blocking SSD/H2D work.
- The first action should be to validate and repair the cache controller, not to
  promote EAMC as default.

## Strategy

Validate one paper's approach first.

The recommended order is:

1. Make the current MoE-Infinity-derived path measurable and paper-faithful
   enough to judge it fairly.
2. Add true prediction-guided prefetch using MoE-Infinity signals.
3. Only then introduce FineMoE-style ExpertMaps and dynamic prefetch, because
   they require new trace data and a richer predictor interface.
4. Add CPU DRAM tiering after VRAM eviction and prefetch policy are stable.

Default policy should remain conservative:

- Keep LRU as the user-facing default until another predictor beats it on both
  speed and correctness gates.
- Keep new predictors and prefetchers opt-in until they pass matched
  `llama-moe-bench`, golden-logit, and chat-smoke validation.
- Use benchmark flags before CLI defaults.

## Phase 0 - Baseline And Diagnostics

Purpose: establish a clean baseline and explain why EAMC lost to LRU before
changing algorithms.

### Tasks

1. Reproduce the human 8000 MiB benchmark with the final MVP fast guard stack:

```powershell
$env:LLAMA_MOE_SLOT_MMVQ='1'
$env:LLAMA_MOE_PREFILL_MMVQ='0'
$env:LLAMA_MOE_SLOT_GRAPHS='1'
$env:LLAMA_MOE_SLOT_GLU_FUSION='1'
$env:LLAMA_MOE_TOPK_FUSION_DIAG='0'

.\build-moe-static\bin\Release\llama-moe-bench.exe `
  --model C:/AI/models/qwen/Qwen3.5-35B-A3B-Q4_K_M.moe.gguf `
  --pp 256 --tg 256 --repeat 3 `
  --moe-cache-vram-mb 8000 --moe-predictor lru `
  --moe-warm-cache --moe-hot-start `
  --moe-profile-csv tests\moe-offload\_out\post-cache-p0-lru.csv `
  --moe-profile-summary tests\moe-offload\_out\post-cache-p0-lru.summary.txt
```

Repeat with `--moe-predictor eamc`.

2. Add a small benchmark matrix:

| Cache | Predictor | Cache state | Purpose |
| ---: | --- | --- | --- |
| 8000 MiB | lru | cold/reset | miss penalty baseline |
| 8000 MiB | lru | warm/hot-start | reproduce human run |
| 8000 MiB | eamc | cold/reset | sidecar-independent behavior |
| 8000 MiB | eamc | warm/hot-start | reproduce human run |
| 12000 MiB | lru | cold/reset | final-MVP practical cache |
| 12000 MiB | eamc | cold/reset | final-MVP EAMC compare |

3. Extend diagnostics without changing policy:

- per-layer hit/miss and bytes-read summary,
- per-layer eviction count,
- victim score distribution,
- miss reason: cold slot, eviction, or prefetch-not-ready once prefetch exists,
- sidecar row count and effective EAMC rows scored,
- EAMC score fallback count where prediction falls back to last-use,
- optional `LLAMA_MOE_DEBUG_EVICT=1` structured CSV rather than stderr text.

4. Add an offline CSV analyzer under `tests/moe-offload/` or `scripts/` that
   consumes profiler CSV and emits:

- hit rate by phase/layer,
- misses/token by phase/layer,
- SSD bytes/token by phase/layer,
- predictor overhead by phase/layer,
- top victim layers,
- LRU-vs-EAMC deltas.

### Acceptance

- The reproduced LRU/EAMC numbers match the human trend or the difference is
  explained by cache state/sidecar state.
- The profile shows whether EAMC loses from hit-rate quality, predictor
  overhead, or both.
- No runtime behavior changes are made in this phase.

## Phase 1 - MoE-Infinity Fidelity Audit

Purpose: decide whether the current `eamc` is a faithful enough
MoE-Infinity implementation or should be split into old `eamc-lite` and a new
paper-faithful predictor.

### Tasks

1. Make request/iteration semantics explicit in the predictor API.

Current API:

```cpp
begin_request();
observe(layer, experts_used);
score(layer, expert);
end_request();
```

Needed API direction:

```cpp
begin_sequence_or_turn();
begin_iteration(phase, token_count);
observe(layer, experts_used, optional_weights);
predict(layer_or_future_window);
end_iteration();
end_sequence_or_turn();
```

This can be implemented incrementally, but the plan must distinguish:

- prefill iEAM,
- decode iEAM,
- accumulated request-level rEAM,
- corpus row stored into the sidecar.

2. Add tests that prove the intended semantics.

Extend `tests/moe-offload/test-eamc-cosine.cpp` or add a new test:

- one user request with multiple decode iterations should produce one rEAM row
  when using MoE-Infinity mode,
- separate requests should produce separate rows,
- prefill and decode rows are separated or tagged,
- partial current iEAM can match against historical rEAM rows,
- sidecar round-trip preserves version, phase, row count, and values.

3. Version the sidecar before changing semantics.

Keep `EAM1` load compatibility. Add a new version if row meaning changes:

- `EAM2` for request-level EAM rows,
- include `phase` or separate prefill/decode corpora,
- include capacity, top-k neighbors, and replacement policy,
- include model shape/version checks.

4. Replace FIFO corpus replacement with a measured policy.

Candidate policies:

- `fifo`: keep current behavior for compatibility and tests.
- `dedupe-nearest`: if a new row is highly similar to an existing row, replace
  or merge the redundant row; otherwise evict the oldest or least useful row.
- `reservoir-diverse`: keep rows that improve corpus diversity.

Start with `fifo` and `dedupe-nearest` behind a diagnostic option; do not
change defaults until Phase 2 numbers justify it.

5. Audit scoring against the paper.

Confirm and document:

- cosine similarity uses only observed layers for partial matching,
- matched rows are aggregated before scoring,
- layer-proximity is meaningful with per-layer caches,
- fallback to LRU/last-use is explicit when similarity is zero.

### Acceptance

- Tests clearly state whether `eamc` means MVP EAMC-lite or paper-faithful
  request-level EAMC.
- A new predictor mode can be compared without breaking existing `EAM1`
  sidecars.
- Predictor overhead remains below 1 ms/token in decode for 1024 rows on the
  target model, or row caps/defaults are adjusted before benchmarking.

## Phase 2 - Eviction Policy Experiments

Purpose: improve hit rate before adding prefetch. Eviction-only experiments are
easier to validate and isolate.

### Candidate Predictors

Add predictor names only as needed:

- `lru`: current default and fallback.
- `eamc-lite`: current behavior if semantics are changed.
- `eamc-r`: request-level MoE-Infinity EAMC.
- `lfu`: frequency-only baseline, because FineMoE argues LFU beats LRU in
  sequential expert-cache settings.
- `eamc-lfu`: predicted EAM probability combined with visit frequency.

### Tasks

1. Add per-layer visit frequency to live cache state.

FineMoE's eviction priority combines predicted probability with frequency.
Because the fork has per-layer caches, maintain frequency per `(layer, expert)`.

2. Add a pure LFU baseline.

This is cheap and important because:

- FineMoE's ablation says LRU is weak for expert offloading,
- current human run says LRU beats MVP EAMC,
- LFU separates "recency is wrong" from "EAMC is wrong."

3. Add `eamc-lfu` scoring.

For a resident expert:

```text
keep_score = predicted_probability(layer, expert) * visit_frequency(layer, expert)
victim = lowest keep_score
```

Use epsilon and LRU tie-breaks:

```text
keep_score = max(predicted_probability, eps) * max(freq, 1)
tie-break = older LRU first
```

4. Keep unsafe policies opt-in:

```text
--moe-predictor lfu
--moe-predictor eamc-r
--moe-predictor eamc-lfu
```

5. Benchmark with no prefetch:

| Cache | Predictor | Required outcome |
| ---: | --- | --- |
| 8000 MiB | lru, lfu, eamc-lite, eamc-r, eamc-lfu | identify best tight-cache policy |
| 12000 MiB | lru, lfu, eamc-r, eamc-lfu | ensure final-MVP cache does not regress |

### Acceptance

- A non-LRU predictor must improve decode hit rate and TPOT versus LRU in the
  same cache state before it can become the base for prefetch.
- Predictor overhead must not erase the hit-rate win.
- If LRU still wins, keep LRU as default and proceed to prefetch using the
  best measured predictor only as an experimental option.

## Phase 3 - MoE-Infinity-Style True Prefetch

Purpose: overlap SSD read and H2D with GPU compute instead of only reacting to
misses at the callback boundary.

This is the first phase expected to materially reduce `ssd_read`, `h2d`, and
`stall` time when predictions are accurate.

### Design

Add a guarded prefetcher behind an explicit option:

```text
--moe-prefetch off|next-layer|eamc
--moe-prefetch-distance N
--moe-prefetch-max-per-layer N
```

Start with `off` as default.

The prefetcher should run after a callback observes layer `l`:

1. Ask the predictor for future-layer scores for layers `l+1..l+d`.
2. Select a bounded set of likely experts per future layer.
3. Reserve free or low-priority slots in those future layers.
4. Submit SSD reads and H2D copies to the existing async I/O worker.
5. Mark slots as `loading`.
6. When the future layer arrives:
   - if the expert is `ready`, count a prefetch hit,
   - if it is `loading`, wait and count prefetch-not-ready stall,
   - if absent, pause/cancel low-priority prefetch and do demand load.

### Required Safety Rules

- Never evict an expert reserved by the current callback.
- Never expose a slot mapping to compute until its H2D has completed.
- A loading slot must not be selected as a victim unless the prefetch is
  cancelled and its completion is safely ignored.
- On-demand misses have priority over speculative prefetch.
- Prefetch must be bounded by pinned-buffer pool, I/O queue depth, and per-layer
  free slots.
- If the predictor confidence is low, prefer fewer prefetches on SSD. Wasted SSD
  reads are expensive on the target system.

### New Metrics

Extend profiler rows and summary with:

- `prefetch_submitted`,
- `prefetch_ready_hit`,
- `prefetch_not_ready`,
- `prefetch_cancelled`,
- `prefetch_wasted`,
- `prefetch_bytes`,
- `prefetch_wait_us`,
- `prefetch_overlap_us` if measurable,
- `demand_miss_after_prefetch`.

### Implementation Touch Points

- `src/moe-offload/predictor.h`: expose future-layer score vectors or ranked
  candidates.
- `src/moe-offload/predictor.cpp`: implement ranked prediction for EAMC/LFU
  variants.
- `src/moe-offload/slot_pool.cpp`: add slot state, prefetch reservation,
  completion handling, cancellation, and metrics.
- `src/moe-offload/io.*`: verify queue priority or add demand-before-prefetch
  scheduling if needed.
- `src/moe-offload/profiler.*`: add prefetch metrics.
- `tools/moe-bench/main.cpp`: add prefetch args and print summary context.
- `common/arg.cpp`: add common args only after benchmark validation.

### Acceptance

- Golden logits remain `max|d|=0` for forced-eviction cases.
- Chat smoke still passes with `--moe-fast-paths`.
- At 8000 MiB cache, prefetch improves decode TPOT by at least 10% versus the
  same predictor without prefetch, or reduces decode SSD/H2D/stall enough to
  justify further tuning.
- Wasted prefetch bytes stay below 10% of total decode expert bytes, or the
  policy remains diagnostic-only.
- Demand-load latency does not regress when predictions are poor.

## Phase 4 - Better Hot-Start

Purpose: reduce cold TTFT without making interactive defaults unsafe.

The current benchmark-only hot-start preloads experts ranked from EAMC sidecar
global scores. The closeout notes say it worsened TTFT in at least one run, so
it needs a better source and stricter admission.

### Tasks

1. Keep hot-start benchmark-only until it proves useful.

2. Compare sources:

- global sidecar frequency,
- request-level EAMC nearest rows after observing prompt prefill prefix,
- first-layer route prefix,
- LFU sidecar frequency per layer,
- FineMoE-style semantic/trajectory source later.

3. Add hot-start diagnostics:

- hot-start slots loaded,
- hot-start experts used in first prefill,
- hot-start experts evicted before use,
- hot-start SSD bytes,
- cold TTFT delta.

4. Use a two-step hot-start:

- fast minimal preload for high-confidence experts,
- defer speculative lower-confidence preload to Phase 3 prefetcher.

### Acceptance

- Hot-start reduces cold TTFT on the short-chat and 256-token benchmark prompts
  without worsening decode TPOT.
- Hot-start remains off by default until the benefit reproduces across prompt
  classes.

## Phase 5 - FineMoE ExpertMap Foundation

Purpose: add the data model needed for FineMoE-style prediction without yet
changing cache policy.

### Minimal Transferable Version

Start with trajectory-only ExpertMaps.

Capture per-layer expert probabilities in the least invasive way:

1. First implementation: top-k selected expert IDs plus routing weights,
   normalized over selected experts.
2. Later implementation, only if needed: full router probability distribution
   before top-k.

The top-k-only map is not paper-complete, but it is enough to test whether
iteration-level probability traces beat count-only EAMC on this fork.

### Tasks

1. Add a new sidecar version for ExpertMaps:

```text
MAP1
model shape: n_layers, n_experts, top_k
row fields: phase, iteration index, layer probability vectors
optional fields: prompt feature hash, embedding feature id
```

2. Add a predictor mode:

```text
--moe-predictor expertmap
```

3. Implement trajectory similarity:

- for target layer `l`, compare observed maps from layers `1..l-d`,
- choose nearest historical map,
- use its probability vector for layer `l`.

4. Use fixed prefetch distance first.

Start with:

```text
--moe-prefetch-distance 3
```

Do not tune dynamically until Phase 6.

5. Add tests:

- top-k probability capture stores expected weights,
- trajectory similarity chooses the expected historical row,
- sidecar round-trip preserves probabilities,
- old EAMC sidecars are ignored or converted explicitly.

### Acceptance

- ExpertMap collection overhead is measurable and below 1 ms/token in decode.
- Offline replay shows ExpertMap predictions rank actual future experts above
  LRU/LFU/EAMC on at least one workload.
- No cache behavior changes are enabled by default in this phase.

## Phase 6 - FineMoE Dynamic Prefetch And Eviction

Purpose: test the FineMoE control policy once ExpertMap traces exist.

### Tasks

1. Implement similarity-aware prefetch selection.

For a target layer:

```text
delta = clip(1 - similarity_score, 0, 1)
prefetch experts in probability order until cumulative probability >= delta
always prefetch at least model_top_k candidates if slots allow
```

For SSD, add stricter bounds:

```text
max_prefetch_per_layer
max_prefetch_bytes_per_token
min_similarity_for_extra_prefetch
```

2. Implement FineMoE-style priority:

```text
prefetch_priority = probability / max(1, target_layer - current_layer)
eviction_keep_score = probability * frequency
victim = lowest eviction_keep_score
```

Use LRU only as a tie-breaker.

3. Add asynchronous search/prefetch separation.

The search can run on CPU after each layer observation and enqueue prefetch
work, but the callback must not wait for search unless it is handling a demand
miss.

4. Miss handling rule:

When a demand miss occurs:

- pause or deprioritize speculative prefetch,
- load demand expert immediately,
- record prefetch miss reason,
- resume prefetch only after demand work is drained.

### Acceptance

- `expertmap + dynamic prefetch` beats the best Phase 3 MoE-Infinity-prefetch
  configuration on decode TPOT or hit rate.
- It does not regress prefill TTFT.
- It does not increase wasted SSD reads above the Phase 3 threshold.
- If it only wins with CPU DRAM and not SSD, keep it deferred until Phase 8.

## Phase 7 - Semantic Early-Layer Search

Purpose: address FineMoE's early-layer problem only if trajectory-only maps are
not enough.

This is intentionally deferred because it may require more invasive integration
with llama.cpp embeddings or hidden states.

### Lower-Risk Alternatives First

Before extracting full semantic embeddings, test cheaper early-layer signals:

- first-layer router top-k pattern,
- prompt length bucket,
- prompt hash or token n-gram fingerprint,
- system/chat-template class,
- prefill first microbatch route prefix.

### Full Semantic Search Option

If cheap signals fail, capture embedding-layer output or a reduced projection:

- do not store raw large vectors unless memory and overhead are acceptable,
- use a fixed low-dimensional projection or sampled dimensions,
- store only sidecar metadata needed for cosine lookup,
- keep this predictor opt-in.

### Acceptance

- Early-layer hit rate improves versus trajectory-only ExpertMap.
- TTFT improves, not just decode.
- Capture and lookup overhead is below the benefit from fewer misses.

## Phase 8 - Optional CPU DRAM Warm Tier

Purpose: reduce SSD miss penalty after the VRAM policy is stable.

This was requested in the raw plan, but it should come after VRAM cache policy
and prefetch are measured. The current target has only 32 GB system DRAM, so a
bounded host cache is needed.

### Design

Add:

```text
--moe-cache-ram-mb N
```

Tiers:

```text
GPU VRAM slot cache: hot experts, compute-ready
CPU DRAM blob cache: warm expert blobs, page-aligned host memory
SSD GGUF blob source: cold source of truth
```

Data path:

- GPU hit: no I/O.
- CPU DRAM hit: H2D only.
- SSD miss: SSD read to host buffer, optional insert into CPU DRAM, H2D to GPU.

Policy:

- Reuse the best Phase 2/3 predictor for DRAM admission.
- Keep DRAM cache per expert blob or per whole expert group.
- Track exact memory use because non-expert weights, KV cache, and OS page cache
  compete with this memory.

### Metrics

- DRAM cache hit rate,
- SSD bypassed bytes,
- H2D bytes from DRAM,
- SSD-to-DRAM fill time,
- DRAM memory peak,
- total process RSS.

### Acceptance

- At 8000 MiB VRAM cache, CPU DRAM tier reduces decode SSD bytes/token and
  improves TPOT without causing system memory pressure.
- If Windows file cache already provides the same benefit, document that and
  avoid a custom DRAM tier unless direct I/O or controlled cache state becomes
  necessary.

## Validation Gates

Run these after any phase that changes runtime behavior.

### Build

```powershell
cmake --build build-moe-static --config Release --target `
  llama-cli llama-completion llama-moe-bench `
  test-eamc-cosine test-lru-eviction test-slot-mmvq test-topk-moe-fusion -j 8
```

### CTest

```powershell
ctest --test-dir build-moe-static -C Release -L moe-offload --output-on-failure
```

### Golden Logits

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tests\moe-offload\test-golden-logits.ps1 `
  -Bin "$PWD\build-moe-static\bin\Release\llama-completion.exe" `
  -Model "C:\AI\models\qwen\Qwen3.5-35B-A3B-Q4_K_M.moe.gguf" `
  -Tol 1e-3 -NPredict 8 -StreamCacheMb 4000 `
  -Prompt "Hello" -Seed 42 -Context 4096 -UBatch 8
```

### Chat Smoke

```powershell
$env:LLAMA_MOE_FAST_PATHS='1'
powershell -NoProfile -ExecutionPolicy Bypass -File tests\moe-offload\test-llama-cli-chat.ps1 `
  -Bin "$PWD\build-moe-static\bin\Release\llama-cli.exe" `
  -Model "C:\AI\models\qwen\Qwen3.5-35B-A3B-Q4_K_M.moe.gguf" `
  -CacheMb 12000 -Predictor lru -NPredict 64 -StreamingUBatch 0
```

Repeat chat smoke with any new predictor/prefetcher before promotion.

### Performance Matrix

Minimum matrix for each runtime-changing phase:

| Cache | Predictor | Prefetch | Prompt | Gen | Repeats | Cache state |
| ---: | --- | --- | ---: | ---: | ---: | --- |
| 8000 | lru | off | 256 | 256 | 3 | cold/reset |
| 8000 | candidate | off/on | 256 | 256 | 3 | cold/reset |
| 8000 | lru | off | 256 | 256 | 3 | warm/hot-start |
| 8000 | candidate | off/on | 256 | 256 | 3 | warm/hot-start |
| 12000 | lru | off | 256 | 128 | 3 | cold/reset |
| 12000 | candidate | off/on | 256 | 128 | 3 | cold/reset |

Track:

- TTFT,
- TPOT,
- prefill/decode hit rate,
- misses/token,
- SSD bytes/token,
- SSD read ms/token,
- H2D ms/token,
- stall ms/token,
- GPU compute ms/token,
- predictor ms/token,
- callback wall ms/token,
- prefetch metrics if enabled,
- VRAM and DRAM peak.

## Promotion Rules

A candidate can be promoted from diagnostic to recommended only if:

- it passes correctness gates,
- it improves TPOT or TTFT on matched workloads,
- it does not regress coherent `llama-cli --jinja --reasoning off` chat,
- it improves hit rate or overlap for an understandable reason,
- it does not rely on hot OS page-cache artifacts unless documented,
- its sidecar format has compatibility checks,
- LRU remains available as fallback.

Suggested default path:

- Keep `lru` default for CLI until a candidate wins consistently.
- Keep `--moe-prefetch off` default until Phase 3 or Phase 6 passes.
- Benchmark may expose experimental flags earlier than CLI.

## Expected Milestones

### Milestone A - EAMC Explained

Deliverables:

- baseline comparison report,
- EAMC-lite versus MoE-Infinity fidelity note,
- decision on new sidecar version,
- updated predictor tests.

Outcome:

- Answer whether current EAMC is correct: it is correct against its own tests
  and high-level idea, but not paper-complete MoE-Infinity. Treat it as
  EAMC-lite until request-level traces and prefetch are implemented.

### Milestone B - Best Eviction-Only Policy

Deliverables:

- LRU/LFU/EAMC variant benchmark matrix,
- per-layer miss analysis,
- default/fallback recommendation.

Outcome:

- Either identify a non-LRU predictor worth using for prefetch, or keep LRU as
  the base predictor.

### Milestone C - First True Prefetch Win

Deliverables:

- guarded prefetcher,
- prefetch metrics,
- same-build TPOT/TTFT matrix,
- correctness gates.

Outcome:

- Demonstrate useful overlap of SSD/H2D with GPU compute, or prove the current
  SSD/link/cache geometry makes speculative prefetch too wasteful.

### Milestone D - FineMoE Trajectory Evaluation

Deliverables:

- ExpertMap sidecar,
- trajectory similarity predictor,
- dynamic threshold experiments,
- comparison against MoE-Infinity-prefetch best case.

Outcome:

- Decide whether FineMoE-style control should become the next mainline cache
  policy or remain research-only for this fork.

### Milestone E - CPU DRAM Tier Decision

Deliverables:

- CPU DRAM warm-cache prototype or explicit rejection,
- DRAM/SSD/H2D breakdown,
- memory pressure notes.

Outcome:

- Decide whether `--moe-cache-ram-mb` is worth maintaining on the target
  32 GB host-memory system.

## Immediate Next Step

Start with Phase 0 and Phase 1. Do not implement FineMoE first.

The next concrete implementation plan should be:

1. Add diagnostics and baseline analyzer.
2. Reproduce LRU versus EAMC under matched cache states.
3. Split or rename EAMC-lite if the request-level audit confirms the current
   sidecar rows are not MoE-Infinity rEAM rows.
4. Only then implement eviction variants and true prefetch.
