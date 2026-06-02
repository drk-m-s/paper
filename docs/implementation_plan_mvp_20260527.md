# MVP Implementation Plan — MoE Offloading Inference Engine on llama.cpp

**Date:** 2026-05-27
**Status:** Draft, ready for execution
**Scope:** First implementation only (MVP + Profiling Infrastructure). Excludes second/third/fourth/fifth phases of the raw plan.

---

## 1. Scope & Goals

### 1.1 In scope
- Modify llama.cpp to run **Qwen3.5-35B-A3B (Q4_K_M GGUF)** with **SSD-backed expert offloading**, **all compute on GPU (CUDA)**, on the target dev hardware: AMD HX 470 CPU + NVIDIA RTX 5070 Ti (16 GB VRAM) + NVMe SSD via PCIe 4.0 ×4.
- Repack the existing `Qwen3.5-35B-A3B-Q4_K_M.gguf` (located at `C:\AI\models\qwen\`) into an **MoE-offload-friendly GGUF** where each per-layer per-expert tensor is contiguous and page-aligned, with an offset index.
- Implement an **expert cache** in VRAM with a **pluggable predictor interface**, providing two predictors in the MVP:
  - **LRU-only floor** — demand fetch + LRU eviction, no prediction.
  - **MoE-Infinity-style EAMC** — request-level Expert Activation Matrix collection, cosine-distance matching, activation-aware prefetch + eviction.
- Implement **asynchronous demand fetch**: on a dedicated I/O thread + CUDA stream, dispatch expert reads immediately after the router (gate) result for a layer is known, overlapped with the same layer's attention/router work and with previous-layer compute.
- Implement a **profiling subsystem** that emits CSV (per-token-per-layer + summary) and a llama-bench-style text summary covering: cache hit rate, I/O latency breakdown (SSD read, PCIe H2D, GPU compute, stall), VRAM/DRAM/SSD usage, prefill/decode speed, TTFT, TPOT, total exec time.
- Ship `llama-moe-bench`, a new benchmark binary modeled on `llama-bench`.
- Maintain everything as a **fork-with-overlay** of upstream llama.cpp; new code in new files; hook points minimal; guarded by `LLAMA_MOE_OFFLOAD` CMake option + `--moe-offload` runtime flag (default OFF → build is byte-identical to upstream).

### 1.2 Out of scope (deferred to later phases)
- CPU DRAM expert tier (Phase 2).
- KV-cache SSD/CPU offload (Phase 2+). MVP keeps KV-cache fully in VRAM. Context length capped at whatever fits (target ≥ 8K; 32K stretch).
- 256K context, hybrid CPU+GPU compute, fine-grained sub-expert offloading (FineMoE), pre-attention prediction, speculative decoding, learned predictors.
- Quantization/dequantization changes — we consume the existing Q4_K_M GGUF as-is.
- Multi-GPU, server mode, batch > 1 throughput tuning.

### 1.3 Non-goals / explicit simplifications
- **No speculation in the I/O scheduler** for the MVP. Only demand fetch (current layer's experts, immediately after gate). The EAMC predictor influences *eviction* and *post-fetch warming of likely-soon experts*, but does not break the "fetch only what the router asked for" rule for the critical path.
- **No model modification.** Stock HuggingFace / GGUF weights only.
- **No new quantization formats.**

---

## 2. Hardware target & assumptions

| Item | Value |
|------|-------|
| CPU | AMD HX 470 |
| System DRAM | 32 GB |
| GPU | NVIDIA RTX 5070 Ti, 16 GB VRAM |
| GPU ↔ CPU link | OcuLink, PCIe 4.0 ×4 ≈ 8 GB/s effective |
| SSD | NVMe (PCIe 4.0 ×4 expected, ≈ 6–7 GB/s sequential read) |
| Model | Qwen3.5-35B-A3B, Q4_K_M, ~17–20 GB on disk |

**Memory budget sketch on 16 GB VRAM** (rough, to be measured):
- Non-expert weights (attention, embeddings, norms, router): ~3–5 GB after Q4_K_M.
- KV-cache for 8K context: ~1–2 GB depending on attention layout.
- CUDA workspace + compute buffers: ~1 GB.
- ⇒ **Expert cache budget ≈ 8–10 GB** (configurable).
- One Qwen3.5-A3B expert (gate+up+down at Q4_K_M) is on the order of **15–25 MB** (estimate; will be measured during repacking and recorded by the converter). At 10 GB cache that is **~400–600 experts** resident — substantial fraction of total experts (~128 experts × 48 layers ≈ 6144) but small in absolute terms.

---

## 3. High-level architecture

```
                         ┌────────────────────────────────────────────────┐
                         │              llama-cli / llama-moe-bench       │
                         └───────────────────────┬────────────────────────┘
                                                 │
                         ┌───────────────────────▼────────────────────────┐
                         │  llama.cpp core (mostly unmodified)            │
                         │   ├─ model loader  ──hook──► moe_offload_loader│
                         │   ├─ llm_build_moe_ffn ──hook──► moe_dispatch  │
                         │   └─ ggml-cuda backend  ──hook──► moe_cuda_io  │
                         └───────────────────────┬────────────────────────┘
                                                 │
        ┌────────────────────────────────────────┼────────────────────────────────────────┐
        │                                        │                                        │
┌───────▼────────┐                       ┌───────▼────────┐                       ┌───────▼────────┐
│ Expert Cache   │  miss / prefetch req  │ I/O Scheduler  │   pread + cudaMemcpy  │  Storage Layer │
│ (VRAM-resident)│◄──────────────────────│ (worker thread)│──────────────────────►│  (mmap header  │
│  + LRU         │   completion event    │ + CUDA stream  │                       │   + pread blob)│
│  + EAMC pred.  │──────────────────────►│                │                       │                │
└───────┬────────┘                       └────────────────┘                       └────────────────┘
        │
        │ tensor handle (VRAM ptr)
        ▼
   compute (existing CUDA backend)
        │
        ▼
┌────────────────┐
│ Profiler       │  per-layer per-token timing & counters
│  (CSV + text)  │
└────────────────┘
```

Key invariants:
- Every expert tensor consumed by GPU compute must be **resident in VRAM at use time**. The cache layer guarantees this by blocking the compute stream on the prefetch completion event when the expert was a miss.
- The CPU is **only** an I/O orchestrator and (briefly) a staging host for pinned buffers. No CPU compute on expert weights in the MVP.

---

## 4. Components

### 4.1 Converter tool — `llama-moe-repack`

New binary in `tools/moe-repack/`.

**Input:** A standard GGUF (the user's `Qwen3.5-35B-A3B-Q4_K_M.gguf`).

**Output:** A new GGUF (`*.moe.gguf`) plus a sidecar JSON manifest with statistics. The new GGUF is **a valid GGUF readable by stock llama.cpp** (degraded path: vanilla llama.cpp can still load it as-is if VRAM is sufficient), with the following extra properties:

1. The fused expert tensors (`blk.N.ffn_gate_exps`, `blk.N.ffn_up_exps`, `blk.N.ffn_down_exps`) are **split** into per-expert tensors and renamed:
   - `blk.N.ffn_gate_exps.eXXX`
   - `blk.N.ffn_up_exps.eXXX`
   - `blk.N.ffn_down_exps.eXXX`
   where `XXX` is the expert index. Splitting is the simplest path that keeps tensors discoverable by name and works with GGUF's existing alignment rules.
2. The byte offset of each expert tensor in the file is **page-aligned to 4 KiB** (configurable, default 4096). The three tensors of one expert are **placed contiguously** so a single `pread` can fetch all three. The converter records this contiguous-range layout in the GGUF KV metadata (`moe_offload.expert_blob.layer.N.expert.E = {offset, size}`).
3. KV metadata added:
   - `moe_offload.version = 1`
   - `moe_offload.n_experts_per_layer`
   - `moe_offload.n_moe_layers`
   - `moe_offload.expert_blob_size_max` (worst-case expert blob size, used to size pinned staging buffers)
   - The per-(layer, expert) `{offset, size}` table is stored as a compact array tensor for fast O(1) lookup at load time.

**Why one repacked GGUF rather than many files / a separate blob:** keeps distribution simple, allows vanilla llama.cpp to still consume it, and avoids filesystem overhead of thousands of small files. Page alignment + per-expert offsets give us the same I/O behavior as a split-file layout. (Decision matches user choice in the discussion.)

**Behavior with `mul_mat_id`:** since the loader will reconstruct *fused virtual tensors* in memory (see §4.2), the in-graph `ggml_mul_mat_id` operation is unchanged. The expert-cache layer is responsible for materializing the right slice into the right position of the fused tensor's VRAM region just before dispatch (see §4.4).

### 4.2 Loader extension — `moe_offload_loader`

Files: `src/moe-offload/loader.{h,cpp}`.

Responsibilities at model load:
- Detect `moe_offload.version` in the GGUF; if absent **and** `--moe-offload` is requested, fail with a clear message ("model not repacked, run `llama-moe-repack` first").
- mmap the non-expert region of the file (everything except the expert blobs) as today.
- Do **not** map expert blobs into VRAM. Instead, register them with `moe_offload_loader` which builds an in-memory table:
  ```
  struct expert_record {
      uint64_t file_offset;   // page-aligned
      uint32_t size;          // bytes, three sub-tensors concatenated
      uint16_t layer;
      uint16_t expert;
  };
  ```
- For each MoE layer, allocate a **fused VRAM tensor placeholder** of shape `[d, d_ff, K]` where `K` is the *cache slot count for this layer* (not the total expert count). The router output's logical expert IDs are remapped to physical slots at dispatch time (see §4.4). This is the minimal change that keeps `ggml_mul_mat_id` working.
- Allocate **pinned host staging buffers** sized to `2 × expert_blob_size_max × n_io_workers` (double-buffered per I/O worker).

### 4.3 I/O scheduler — `moe_io`

Files: `src/moe-offload/io.{h,cpp}`, `ggml/src/ggml-cuda/moe-offload-io.cu`.

- One dedicated **I/O worker thread** in the MVP (single thread is sufficient given ~6 GB/s SSD; multi-thread can be added once profiling shows queue depth helps).
- One dedicated **CUDA stream** (`moe_h2d_stream`) for host→device copies. Compute uses the existing main stream. Synchronization between streams uses CUDA events.
- Fetch primitive: `pread(fd, pinned_buf, size, offset)` — bypasses page cache nondeterminism. On Windows: `ReadFile` with `FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED` and aligned buffers. Wrap in a small OS abstraction (`moe_pread`).
- Issue order: requests are pushed onto a lock-free SPSC queue from the compute thread. The I/O worker pops in FIFO order. Each request carries `{layer, expert_id, target_vram_slot, completion_event}`. Completion = `cudaEventRecord(completion_event, moe_h2d_stream)` after `cudaMemcpyAsync`.
- On a cache **miss**, the compute stream waits on `completion_event` (`cudaStreamWaitEvent`) before dispatching `mul_mat_id` for that expert. On a cache **hit**, no wait.

### 4.4 Expert cache — `moe_cache`

Files: `src/moe-offload/cache.{h,cpp}`.

Data structures per MoE layer:
```
class layer_cache {
    int n_slots;                       // physical VRAM slots
    std::vector<int> slot_to_expert;   // -1 = empty
    std::unordered_map<int,int> expert_to_slot;
    intrusive_list lru_list;           // for LRU
    // for EAMC predictor:
    eam_predictor* pred;               // nullptr if predictor == "lru"
};
```

Key operations (called from the patched `llm_build_moe_ffn` and from a small per-step driver):

1. **`begin_layer(layer_id, router_topk_experts)`** — called as soon as router output is known for this layer (CPU-side copy of the indices is required; we add a small `cudaMemcpyAsync` D2H of the top-k indices on a tiny side stream, blocking only the CPU control thread). For each required expert:
   - If resident → mark as `pinned-this-step`, update LRU, hit++.
   - If not resident → choose an eviction victim (LRU floor or EAMC) among the non-pinned slots, enqueue an I/O request to that slot, store the completion event in the slot. Miss++.
2. **`bind_slots(layer_id) -> {expert_id -> slot_id}`** — returned to the patched build code so it can rewrite the `mul_mat_id` index tensor from logical expert IDs into physical slot IDs.
3. **`wait_misses(layer_id)`** — emits `cudaStreamWaitEvent(compute_stream, slot.event)` for each miss-slot used this layer.
4. **`end_layer(layer_id)`** — unpins; calls predictor's `observe(layer_id, experts_used)`.
5. **`begin_request()` / `end_request()`** — bracketing hooks the predictor uses to roll the request-level EAM.

**Eviction policies:**
- `lru`: pick least-recently-used non-pinned slot.
- `eamc`: pick the non-pinned slot whose expert has the **lowest predicted score** (see §4.5).

### 4.5 Predictor interface — `moe_predictor`

Files: `src/moe-offload/predictor.{h,cpp}`, `src/moe-offload/eamc.{h,cpp}`.

```cpp
class predictor {
public:
    virtual ~predictor() = default;
    virtual void begin_request() = 0;
    virtual void observe(int layer, const std::vector<int>& experts_used) = 0;
    virtual float score(int layer, int expert) const = 0;   // higher = more likely needed soon
    virtual void end_request() = 0;
};
```

Two implementations in the MVP:

**(a) `lru_predictor`** — returns scores derived from LRU recency only. Score = (last_use_step). Eviction picks lowest score among non-pinned. (Functionally identical to LRU eviction.)

**(b) `eamc_predictor`** — faithful to MoE-Infinity:
- Maintains `iEAM` (current request, iteration EAM) of shape `[L, E]` storing per-expert activation counts as the request progresses.
- Maintains `rEAM` at end-of-request (aggregated counts over the whole request).
- Maintains `EAMC`: a bounded collection of past `rEAM`s (default capacity 1024; on insert when full, evict the entry with highest mean cosine similarity to the rest — the redundancy-replacement rule from the paper).
- `score(layer, expert)` is computed once per layer transition by:
  1. Compute cosine similarity of the current request's partial iEAM (up to current layer) to each rEAM in EAMC.
  2. Take top-K matches (default K=8). Weighted-average their per-layer rows for layers `> current_layer`.
  3. Apply layer-proximity weighting `(1 - (i - l)/L)`.
  4. Sum into a `predicted EAM` row for "next several layers"; `score(layer, expert)` reads from that row.
- EAMC persistence: on shutdown, the EAMC is dumped to a sidecar file `*.eamc` next to the model GGUF (configurable path). On startup, it is reloaded if present. This lets the system "warm up" over deployment time without an offline profiling phase (consistent with our "no offline profiling in MVP" rule — EAMC is an *online* trace cache, not an offline profile).
- Cost budget: EAMC matching should stay < 1% of decode time. For Qwen3.5-A3B with L=48, E=128, EAMC=1024, one cosine match is ~6K multiply-adds; 1024 matches × 6K = 6M ops/layer ≈ negligible on modern CPU. We will measure and add memoization if needed (the iEAM only changes incrementally — most past similarities can be updated, not recomputed).

### 4.6 llama.cpp hook points (minimal overlay)

Concrete list — these are the **only** lines we expect to edit inside upstream files. Each is a single guarded callout.

| File | Hook | Purpose |
|------|------|---------|
| `src/llama-model-loader.cpp` (or current equivalent) | After GGUF KV parse | If `moe_offload.version` present and `--moe-offload` set, route MoE tensor loads through `moe_offload_loader`. |
| `src/llama.cpp` :: `llm_build_moe_ffn` | Before `ggml_mul_mat_id` for each `gate/up/down` | Call `moe_cache::begin_layer(...)`, rewrite index tensor to slot IDs via `bind_slots(...)`, then `wait_misses(...)`. After post-MoE work, `end_layer(...)`. |
| `src/llama-context.cpp` | On `llama_decode` entry/exit | `moe_predictor::begin_request()` / `end_request()` per logical request (decision needed: per-prompt or per-token; MVP = per-prompt). |
| `ggml/src/ggml-cuda.cu` :: backend init | After CUDA context creation | Initialize `moe_h2d_stream`, expose handle to `moe_io`. |
| `tools/main/main.cpp` and `common/arg.cpp` | CLI parsing | Add `--moe-offload`, `--moe-cache-vram-mb`, `--moe-cache-vram-frac`, `--moe-predictor=lru|eamc`, `--moe-profile-csv=PATH`. |

All hooks are guarded by `#ifdef LLAMA_MOE_OFFLOAD`. With the macro undefined, the build is byte-identical to upstream.

### 4.7 Profiling subsystem — `moe_profiler`

Files: `src/moe-offload/profiler.{h,cpp}`.

Two streams:

**(A) Per-layer-per-token CSV** (`--moe-profile-csv=PATH`, optional):

| Column | Notes |
|--------|-------|
| `token_idx` | 0 = first prefill token, then continues into decode |
| `phase` | `prefill` / `decode` |
| `layer` | MoE layer index |
| `k_required` | Number of distinct experts the router asked for |
| `k_hit` | Number that were cache hits |
| `k_miss` | k_required - k_hit |
| `ssd_read_us` | Total wall μs spent in `pread` for this layer (I/O worker) |
| `h2d_us` | μs for `cudaMemcpyAsync` H2D (GPU timing) |
| `compute_us` | μs of GPU compute attributable to this layer's experts |
| `stall_us` | μs the compute stream waited on `cudaStreamWaitEvent` for misses |
| `cache_resident_experts` | Cache size for this layer |
| `predictor` | `lru` / `eamc` |

CSV writes are buffered (4 KiB lines) and flushed in a background thread to avoid I/O perturbing measurements.

**(B) Summary report** (always emitted; `--moe-profile-summary=PATH`, default stdout):

llama-bench-style aligned table:
```
model: Qwen3.5-35B-A3B Q4_K_M
predictor: eamc      cache: 8192 MB   ssd: /dev/nvme0n1
n_prompt: 1024  n_gen: 256

phase     tokens   total_ms   per_token_ms   tok/s
prefill     1024     1320.4          1.29     774
decode       256     9874.2         38.57      26

cache hit rate (decode): 78.3%
SSD bytes read (decode): 3.42 GB  (avg 13.4 MB/token)
TTFT: 1320.4 ms
TPOT: 38.57 ms
total: 11194.6 ms

I/O breakdown (decode, mean per token):
  ssd_read       21.5 ms
  h2d              7.8 ms
  gpu_compute     14.2 ms
  stall (overlap loss)    9.1 ms

VRAM peak: 14.8 GB  (experts: 8.0 GB, kv: 1.7 GB, non-expert: 4.6 GB, other: 0.5 GB)
DRAM peak (process): 1.4 GB
SSD reads: 18234 (avg 19.4 MB each, avg latency 1.18 ms)
```

Counters are collected via:
- `cudaEventElapsedTime` for GPU timing (events placed around each MoE-layer dispatch).
- `clock_gettime(CLOCK_MONOTONIC)` for I/O timings on the worker thread.
- Process VRAM peak via `cudaMemGetInfo` snapshots, augmented with our own allocator's high-water-mark for the expert cache region.
- Process DRAM peak via OS APIs (GetProcessMemoryInfo / getrusage).
- SSD usage = sum of pread sizes.

### 4.8 `llama-moe-bench` tool

New tool at `tools/moe-bench/`. Mirrors `llama-bench` CLI surface; adds:
```
--pp N            number of prefill tokens (default 1024)
--tg N            number of generate tokens (default 256)
--ctx N           context size (default = pp + tg)
--moe-cache-vram-mb N | --moe-cache-vram-frac F
--moe-predictor lru|eamc
--moe-profile-csv PATH
--moe-profile-summary PATH
--repeat N        number of repeats for stable means
--warmup N        number of warmup runs (not counted)
```

Output: same llama-bench-style summary as §4.7(B) per run, plus an aggregated table over `--repeat`.

---

## 5. Repository layout (overlay on upstream llama.cpp fork)

```
llama.cpp.offload/                       (fork root)
├── CMakeLists.txt               <-- adds option(LLAMA_MOE_OFFLOAD)
├── src/
│   ├── llama.cpp                <-- ≤ 10 lines of hook callouts, all #ifdef'd
│   ├── llama-model-loader.cpp   <-- ≤ 5 lines of hook
│   ├── llama-context.cpp        <-- ≤ 3 lines of hook
│   └── moe-offload/             <-- NEW directory, all new code here
│       ├── loader.{h,cpp}
│       ├── cache.{h,cpp}
│       ├── io.{h,cpp}
│       ├── predictor.{h,cpp}
│       ├── eamc.{h,cpp}
│       ├── profiler.{h,cpp}
│       ├── platform_io.{h,cpp}  <-- pread / Win32 ReadFile abstraction
│       └── CMakeLists.txt
├── ggml/src/ggml-cuda/
│   ├── ggml-cuda.cu             <-- ≤ 5 lines of hook (init h2d stream)
│   └── moe-offload-io.cu        <-- NEW (cudaMemcpyAsync helpers, event mgmt)
├── tools/
│   ├── moe-repack/              <-- NEW (converter binary)
│   └── moe-bench/               <-- NEW (benchmark binary)
├── tests/moe-offload/           <-- NEW
└── docs/moe-offload/            <-- NEW (README, design, internals)
```

CMake:
```
option(LLAMA_MOE_OFFLOAD "Enable MoE expert offloading subsystem" OFF)
if (LLAMA_MOE_OFFLOAD)
    add_compile_definitions(LLAMA_MOE_OFFLOAD)
    add_subdirectory(src/moe-offload)
    add_subdirectory(tools/moe-repack)
    add_subdirectory(tools/moe-bench)
endif()
```

---

## 6. Milestones (within the MVP)

Suggested execution order. Each milestone is independently testable.

| ID | Milestone | Deliverable | Exit criterion |
|----|-----------|-------------|----------------|
| M0 | Fork + CMake skeleton | `LLAMA_MOE_OFFLOAD` flag, empty `src/moe-offload/`, vanilla build still passes upstream tests | `cmake --build` succeeds with flag ON and OFF; upstream unit tests pass |
| M1 | `llama-moe-repack` converter | Repacked `qwen3.5-35b-a3b-q4_k_m.moe.gguf` + sidecar JSON | Repacked file loads in **stock** llama.cpp and produces identical logits (within tolerance) on a fixed prompt vs. original GGUF |
| M2 | Loader + dummy cache (cache big enough to hold all experts) | Model runs end-to-end via the offload path, but with no real eviction | Logits match stock llama.cpp on the same prompt; no crashes |
| M3 | I/O scheduler + real LRU cache + cache size knobs | Model runs with `--moe-cache-vram-mb 8000` and produces correct outputs | (a) Correctness preserved; (b) runs at `--ctx 8192` without OOM |
| M4 | Profiler + `llama-moe-bench` | CSV + summary text outputs | Numbers are sane, reproducible across `--repeat 3`, summary matches per-line CSV totals |
| M5 | EAMC predictor + pluggable interface | `--moe-predictor lru` and `--moe-predictor eamc` both functional | Both predictors yield correct outputs; A/B run shows EAMC hit rate ≥ LRU hit rate on a representative prompt |
| M6 | Async overlap (I/O on dedicated CUDA stream + thread) | Reduced `stall_us` in profiler vs M3 baseline | Decode `stall_us` decreases vs. the serial baseline by a measurable margin |
| M7 | README + repro scripts | `docs/moe-offload/README.md` with build, convert, run, bench instructions | A fresh checkout can produce a working bench run by following the README |

M6's overlap work can be done partially in M3 (the scheduler will already be on a worker thread); M6 is the polish + measurement pass.

---

## 7. Correctness strategy

- **Golden logits test**: run vanilla llama.cpp on the *original* GGUF with the model fully resident (CPU offload if necessary) for one fixed prompt (`"The quick brown fox"`, 32 tokens out, seed=42, temp=0). Save logits at every step.
- **Repack test**: convert to `*.moe.gguf`, run vanilla llama.cpp on it (without offload), expect identical (byte-equal) tensor data → logits should be exactly equal.
- **Offload-correctness test**: run with `--moe-offload --moe-cache-vram-mb <small>` so eviction is forced, compare logits against golden. Allowable delta: `max |Δlogit| < 1e-3` (Q4_K_M is non-deterministic across batches due to reduction order on GPU; we will calibrate the tolerance during M3).
- **Stress test**: 1K-token prefill + 256-token decode at small cache (e.g. 2 GB) to maximize eviction churn; assert no use-after-evict (a debug-build assertion in the cache verifies every dispatched expert slot's last write event has been completed before compute begins).
- **Cross-predictor consistency**: outputs with `--moe-predictor lru` and `--moe-predictor eamc` must match (same logits) — the predictor only affects *which* experts get evicted/prefetched, never *which* experts get used.

---

## 8. Profiling: what we will measure & why

The user explicitly wants to know "what can be improved later". The profiler is designed to answer these specific diagnostic questions:

| Question | Metric(s) |
|----------|-----------|
| Are we PCIe-bound or SSD-bound? | Compare `ssd_read_us` (worker timer) vs. `h2d_us` (CUDA event). |
| Is overlap working? | `stall_us` per layer; ratio `stall_us / (ssd_read_us + h2d_us)` ideally → 0 with predictor. |
| Is the cache too small? | `k_miss / k_required` (overall and per layer); see if it falls off a cliff at small cache sizes. |
| Does EAMC help vs LRU? | Hit rate delta, stall delta, decode TPS delta across two runs. |
| Are some layers worse than others? | Per-layer hit rate distribution. (MoE-Infinity reports early layers are harder.) |
| Is the predictor itself expensive? | Wall-clock overhead of EAMC matching per layer (`pred_us` column added when predictor=eamc). |
| What's the realistic ceiling? | A "oracle" mode where we pretend all needed experts are pre-cached, run the same prompt, and report tok/s — this is the floor of what the I/O is costing us. |

The "oracle" mode is a flag `--moe-oracle` that, before each layer, calls `begin_layer` then immediately blocks until all required experts have been fetched (no overlap, but no stalls during compute either) — effectively measuring "perfect predictor" performance. Small addition, big diagnostic value.

---

## 9. Tests

Under `tests/moe-offload/`:

1. **Unit:** `eamc_test.cpp` — synthetic rEAMs, verify cosine matching + redundancy-eviction policy.
2. **Unit:** `lru_test.cpp` — sequence-based eviction sanity.
3. **Unit:** `cache_test.cpp` — slot allocation, pinning, hit/miss bookkeeping, no leaks.
4. **Unit:** `io_test.cpp` — pread offset/size correctness; pinned-buffer recycling; H2D event ordering.
5. **Integration:** `repack_roundtrip_test.py` — repack the small Qwen3 test (if available) or a stripped 2-layer slice of Qwen3.5-A3B; assert tensor data byte-equality.
6. **Integration:** `offload_logits_test.py` — golden logits comparison (see §7).
7. **Smoke:** `bench_smoke.sh` — run `llama-moe-bench --pp 64 --tg 16 --repeat 2`; assert summary is non-empty and contains expected keys.

CI on a smaller machine is impractical for the full model; run integration tests on the dev box, unit tests on CI.

---

## 10. Open risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Qwen3.5-35B-A3B not yet supported by upstream llama.cpp's MoE path | Medium | High | Verify on day 0 by running stock llama.cpp on the GGUF first. If unsupported, the MVP must include adding the architecture — track as M0.5. |
| `mul_mat_id` with a slot-remapped index tensor breaks numerically due to reduction ordering | Low | Medium | The slot-remap only changes *which physical row* is accessed; the math is identical. Verify via golden-logits test. |
| Repacked GGUF is too "spread out" on SSD, causing slower reads than fused | Low | Medium | Page-aligned + contiguous (gate,up,down) per expert means one big sequential read per expert. Measure in M3. |
| EAMC matching cost dominates predictor benefit at very small cache | Low | Low | Profiler measures it; EAMC has a `max_matches` parameter and incremental cosine updates to keep cost bounded. |
| Pinned buffer pool exhausts host memory if I/O queue depth grows | Low | Medium | Fixed-size pool with backpressure (worker thread blocks when no free buffer). |
| CUDA stream wait turns into full-device sync on driver corner cases | Low | High | Validate with `nsys` (Nsight Systems) trace during M6. |
| OcuLink + Windows storage stack has higher pread latency than expected | Medium | Medium | Profiler reports per-read latency distribution; if too high, fall back to `O_DIRECT`-equivalent path on Windows (`FILE_FLAG_NO_BUFFERING`). |

---

## 11. Deliverables (definition of done for the MVP)

1. ✅ Fork of llama.cpp with `LLAMA_MOE_OFFLOAD` build option, all new code under `src/moe-offload/`, `tools/moe-repack/`, `tools/moe-bench/`.
2. ✅ `llama-moe-repack` converts `Qwen3.5-35B-A3B-Q4_K_M.gguf` into `*.moe.gguf`.
3. ✅ `llama-cli --moe-offload --model qwen3.5-35b-a3b-q4_k_m.moe.gguf --moe-cache-vram-mb 8000 --moe-predictor {lru|eamc} -p "..." -n 256` runs to completion on the 5070 Ti, correctness verified vs. golden logits.
4. ✅ `llama-moe-bench` reproduces the summary report shown in §4.7(B), with both predictors, and emits a per-layer-per-token CSV when requested.
5. ✅ Profiling summary numbers reported for: prefill TPS, decode TPS, TTFT, TPOT, total exec, cache hit rate (per phase), I/O breakdown (ssd / h2d / compute / stall), VRAM/DRAM/SSD usage.
6. ✅ `docs/moe-offload/README.md` covering: build (`cmake -DLLAMA_MOE_OFFLOAD=ON`), convert (`llama-moe-repack ...`), run (`llama-cli ...`), bench (`llama-moe-bench ...`), and the meaning of every column in the CSV and summary.
7. ✅ Tests in §9 passing on the dev box; unit tests on CI.

**Not** in MVP deliverables: comparison to other MoE offloading systems (we'll do that manually with the numbers in deliverable 5), tuning recommendations, or perf claims beyond "it runs and the profiler tells us where the bottlenecks are".

---

## 12. Decisions captured from discussion

| Decision | Choice |
|----------|--------|
| Predictor scope | Both LRU floor **and** MoE-Infinity-style EAMC, switchable via flag |
| Storage layout | Single repacked GGUF with page-aligned per-(layer,expert) contiguous blobs + offset index |
| Target model | Qwen3.5-35B-A3B Q4_K_M at `C:\AI\models\qwen\` only |
| llama.cpp integration | Fork + overlay (Option 2), guarded by `LLAMA_MOE_OFFLOAD` |
| KV-cache offload | Not in MVP; KV stays in VRAM; context capped to what fits |
| Async I/O overlap | Yes — dedicated I/O thread + dedicated H2D CUDA stream; demand only |
| Profile output format | CSV + llama-bench-style summary text |
| Bench tool | New binary `llama-moe-bench` in `tools/moe-bench/` |
| VRAM budget knobs | Two: `--moe-cache-vram-mb` (cap) and `--moe-cache-vram-frac` (auto) |
| Acceptance bar | Runs correctly on Qwen3.5-35B-A3B Q4_K_M with both predictors; emits the profiling report; **no** hard perf floor |

---

## 13. What this MVP intentionally leaves on the table

(All deferred to phases 2+ per the raw plan. Listed so reviewers know they were considered and not forgotten.)

- CPU DRAM tier between VRAM and SSD (Phase 2).
- Offline profiling for warm-start (Fiddler / DALI / PowerInfer style) (Phase 2).
- Fine-grained sub-expert offloading (FineMoE) (Phase 2).
- Hot-expert preloading at init from EAMC dump (trivial extension of M5 — flag for Phase 2).
- Speculative prefetch via Fate / YALIS hidden-state predictors (Phase 3).
- Speculative decoding via MoE-SpAc (Phase 4).
- Learned cache replacement (FlashMoE) (Phase 5).
- KV-cache SSD offload and 256K context (Phase 2+).
- CPU + GPU hybrid compute (Phase 2+).
- NPU port (separate project — this MVP is GPU-only on the dev hardware).
