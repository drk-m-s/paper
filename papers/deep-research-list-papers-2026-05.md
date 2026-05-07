# MoE Offloading Papers – Categorized Reference List

> Last updated: May 2026 | 48 papers across 5 categories

---

## Category 1: Foundational Offloading & Cache Management

| # | Title (Short) | Authors | Venue | Date | arXiv | Key Idea | Importance |
|---|---------------|---------|-------|------|-------|----------|------------|
| 1 | Fast Inference of MoE LMs with Offloading | Eliseev & Mazur | Technical Report | 2023-12 | 2312.17238 | LRU expert cache, CPU→GPU async prefetch, foundational | ⭐⭐⭐⭐⭐ |
| 2 | MoE-Infinity | Xue et al. (Edinburgh) | arXiv | 2024-01 | 2401.14361 | Sparsity-aware cache, activation trace-guided replacement, batch=1 | ⭐⭐⭐⭐⭐ |
| 3 | AdapMoE | Zhong et al. (PKU) | ICCAD 2024 | 2024-08 | 2408.10284 | Sensitivity-based expert gating and cache management | ⭐⭐⭐ |
| 4 | Cache Management for MoE LLMs (ext.) | Angelopoulos et al. | arXiv | 2025-09 | 2509.02408 | Formal analysis, competitive ratio, workload-adaptive replacement | ⭐⭐⭐ |
| 5 | DALI | Zhu et al. (CAS-ICT) | arXiv | 2026-02 | 2602.03495 | Greedy assignment + residual-based prefetch + workload-aware cache | ⭐⭐⭐⭐⭐ |
| 6 | DyMoE | Huang et al. (SYSU) | arXiv | 2026-03 | 2603.19172 | Dynamic orchestration + mixed-precision + I/O overlap on edge | ⭐⭐⭐⭐ |

---

## Category 2: Predictive Prefetching & Scheduling

| # | Title (Short) | Authors | Venue | Date | arXiv | Key Idea | Importance |
|---|---------------|---------|-------|------|-------|----------|------------|
| 7 | EdgeMoE | Yi et al. (BUPT) | arXiv | 2023-08 | 2308.14352 | Expert-wise bitwidth adaptation + expert preloading, mobile edge | ⭐⭐⭐⭐⭐ |
| 8 | ExpertFlow (predictive cache) | He et al. (HKUST) | DAC 2026 | 2024-10 | 2410.17954 | Router prediction 2–4 layers ahead + token scheduling | ⭐⭐⭐⭐ |
| 9 | Read-ME | Cai et al. (UT Austin) | NeurIPS 2024 | 2024-10 | 2410.19123 | Decoupled pre-router, async prefetch, hardware-aware batching | ⭐⭐⭐ |
| 10 | DAOP | Zhang et al. (NUS) | DATE 2025 | 2025-01 | 2501.10375 | Data-aware offloading + predictive pre-calculation | ⭐⭐⭐⭐ |
| 11 | Fate | Fang et al. (SYSU) | arXiv | 2025-02 | 2502.12224 | Cross-layer gate: predicts layer L+k from layer L, 3-step lookahead | ⭐⭐⭐⭐ |
| 12 | Not All Models Suit Offloading | Liang et al. (Fudan) | arXiv | 2025-05 | 2505.16056 | Local Routing Consistency (LRC) analysis; which models benefit | ⭐⭐⭐⭐ |
| 13 | FloE | Zhou et al. (ZJU) | ICML 2025 | 2025-05 | 2505.05950 | On-the-fly JIT streaming without cache, speculative 0.5-layer buffering | ⭐⭐⭐⭐ |
| 14 | LayerScope / PreScope | Yu et al. (NUDT) | ICS 2026 | 2025-09 | 2509.23638 | Prediction-driven cross-layer scheduling window, PCIe utilization | ⭐⭐⭐ |
| 15 | DuoServe-MoE | Zhang et al. (Sydney) | arXiv | 2025-09 | 2509.07379 | Offline static hot cache + online dynamic prefetch, QoS | ⭐⭐⭐ |
| 16 | Pre-Attention Expert Prediction | Zhu et al. (ETH Zurich) | arXiv | 2025-11 | 2511.10676 | Pre-attention predictor overlaps attention+expert prefetch, 85–90% hit | ⭐⭐⭐⭐⭐ |
| 17 | PROBE | Zhu et al. (ByteDance) | arXiv | 2026-01 | 2602.00509 | Online adaptive prefetch window, co-balance compute/PCIe comm | ⭐⭐⭐⭐ |
| 18 | FlashMoE | Kim et al. (KAIST) | arXiv | 2026-01 | 2601.17063 | ML-based (LSTM) cache replacement for SSD-offloaded edge inference | ⭐⭐⭐⭐⭐ |
| 19 | DATE 2026 NDP Scheduling | Wu et al. (SEU) | DATE 2026 | 2026-01 | 2601.03992 | Scheduling framework for edge GPU-NDP, joint placement+exec order | ⭐⭐⭐⭐ |
| 20 | MoE-SpAc | Li et al. (multiple) | arXiv | 2026-03 | 2603.09983 | Speculative Activation Utility (SAU) guides prefetch priority/eviction | ⭐⭐⭐⭐ |
| 21 | Expert Streaming | Ma et al. (HKUST) | arXiv | 2026-03 | 2603.27624 | Multi-chiplet expert trajectory scheduling for low-batch edge MoE | ⭐⭐⭐⭐ |

---

## Category 3: Hybrid CPU-GPU/NPU Compute

| # | Title (Short) | Authors | Venue | Date | arXiv | Key Idea | Importance |
|---|---------------|---------|-------|------|-------|----------|------------|
| 22 | MoNDE | Kim et al. (SNU) | DAC 2024 | 2024-05 | 2405.18832 | Cold experts computed in near-data memory (NDP DIMM), hot to GPU | ⭐⭐⭐⭐ |
| 23 | MoE-Lightning | Cao et al. (UC Berkeley) | arXiv | 2024-11 | 2411.11217 | CGOPipe + Hierarchical Roofline Model, 10.3× throughput, paged weights | ⭐⭐⭐⭐ |
| 24 | HybriMoE | Zhong et al. (PKU) | DAC 2025 | 2025-04 | 2504.05897 | Dynamic CPU+GPU intra-layer + impact-driven prefetch + score cache | ⭐⭐⭐⭐⭐ |
| 25 | Klotski | Fang et al. (SYSU) | arXiv | 2025-02 | 2502.06888 | Expert-aware multi-batch pipeline, optimal token assignment | ⭐⭐⭐ |
| 26 | Fine-Grained Expert Offloading | Yu et al. (multiple) | arXiv | 2025-02 | 2502.05370 | Sub-tensor granularity offloading, Pareto-optimal latency-memory | ⭐⭐⭐ |
| 27 | Efficient CPU-GPU for MoE (ASP-DAC 2026) | Huang et al. (multiple) | ASP-DAC 2026 | 2025-12 | 2512.16473 | CPU-GPU collaborative for memory-limited systems | ⭐⭐⭐⭐ |
| 28 | OD-MoE | Wang et al. (CUHK) | arXiv | 2025-12 | 2512.03927 | On-demand expert loading, cacheless distributed edge inference | ⭐⭐⭐ |
| 29 | Context-Aware MoE on CXL-NDP | Fan et al. (RPI) | arXiv | 2025-12 | 2512.04476 | CXL memory + NDP compute for cold experts, context-aware scheduling | ⭐⭐⭐ |
| 30 | TriMoE | Pan et al. (CAS-ICT) | DAC 2026 | 2026-03 | 2603.01058 | GPU hot + AMX-CPU warm + DIMM-NDP cold; 3-tier hot/warm/cold | ⭐⭐⭐⭐⭐ |
| 31 | NPUMoE (Apple ANE) | Benazir & Lin (UVA) | arXiv | 2026-04 | 2604.18788 | Static capacity tiers + grouped dispatch + load-aware residency on ANE | ⭐⭐⭐⭐⭐ |
| 32 | Pipelined Sharding (Intel, MLSys 2026) | Ukarande et al. (Intel) | MLSys 2026 | 2026-04 | 2604.26334 | Benchmark-guided CPU-GPU hybrid scheduling, layer-to-unit mapping | ⭐⭐⭐⭐ |
| 33 | A3D-MoE | Huang et al. (Georgia Tech) | arXiv | 2025-07 | 2507.19142 | 3D chiplet stacked memory; quantifies HBM vs PCIe speedup potential | ⭐⭐⭐ |
| 34 | Adaptive Expert Split | Yan et al. (USTC) | arXiv | 2025-09 | 2509.08342 | Dynamically splits large experts for finer cache granularity | ⭐⭐⭐ |

---

## Category 4: Compression, Quantization & Expert Merging

| # | Title (Short) | Authors | Venue | Date | arXiv | Key Idea | Importance |
|---|---------------|---------|-------|------|-------|----------|------------|
| 35 | HOBBIT | Tang et al. (SJTU) | arXiv | 2024-11 | 2411.01433 | INT8 hot / INT4 cold mixed-precision + async prefetch | ⭐⭐⭐⭐ |
| 36 | STUN | Lee et al. (POSTECH) | ACL 2025 | 2024-09 | 2409.06211 | Structured→unstructured pruning, removes redundant experts | ⭐⭐⭐ |
| 37 | BuddyMoE | Wang et al. (Microsoft) | arXiv | 2025-11 | 2511.10054 | Expert redundancy exploitation: substitute cached similar expert on miss | ⭐⭐⭐ |
| 38 | Dynamic Expert Quantization | Chu et al. (Alibaba) | arXiv | 2025-11 | 2511.15015 | Per-expert dynamic precision (FP16/INT8/INT2) based on frequency | ⭐⭐⭐ |
| 39 | SliceMoE | Choi et al. (KAIST) | DAC 2026 | 2025-12 | 2512.12990 | Bit-sliced expert storage, continuous precision under miss-rate constraint | ⭐⭐⭐⭐ |
| 40 | Low-Rank Compensation for MoE BW | Liu et al. (multiple) | arXiv | 2025-12 | 2512.17073 | Low-rank in-device + async residual fetch, 30-50% BW reduction | ⭐⭐⭐ |
| 41 | CoMoE | Li et al. (HKUST) | arXiv | 2025-08 | 2508.09208 | Expert aggregation (merging similar experts) + offloading co-design | ⭐⭐⭐ |
| 42 | SSD Offloading Considered Harmful | Kyung et al. (SNU) | IEEE CAL 2025 | 2025-08 | 2508.06978 | Energy analysis: SSD offloading 3-10× more energy than DRAM | ⭐⭐⭐⭐ |
| 43 | MoBiLE | Zhao et al. (THU) | ASP-DAC 2026 | 2025-10 | 2510.12357 | Big+little experts: always-resident small fallback expert | ⭐⭐⭐ |
| 44 | SlimCaching | Chen et al. (HKU) | IEEE TMC 2025 | 2025-07 | 2507.06567 | Edge multi-node expert caching, placement optimization | ⭐⭐⭐ |

---

## Category 5: Speculative Decoding + Offloading

| # | Title (Short) | Authors | Venue | Date | arXiv | Key Idea | Importance |
|---|---------------|---------|-------|------|-------|----------|------------|
| 45 | SpecMoEOff | Wang et al. (NJU) | arXiv | 2025-08 | 2508.21706 | SD hides offloading I/O, CPU chunked-attn kernel, 2.5× decode | ⭐⭐⭐⭐ |
| 46 | SP-MoE | Chen et al. (HKU) | arXiv | 2025-10 | 2510.10302 | Unified draft model for SD + expert prefetch prediction | ⭐⭐⭐⭐ |
| 47 | MoE-SpeQ | Wang et al. (SJTU) | arXiv | 2025-11 | 2511.14102 | Speculative quantized decoding + proactive prefetch, ARM roofline | ⭐⭐⭐⭐ |
| 48 | Speculating Experts | Madan et al. (UMD) | arXiv | 2026-03 | 2603.19289 | Speculative expert execution with rollback on wrong prediction | ⭐⭐⭐⭐ |
| 49 | SpecMoE (self-assisted, DAC 2026) | Bang et al. (POSTECH/KAIST) | DAC 2026 | 2026-04 | 2604.10152 | Self-speculation: shallow layers = draft; no separate model needed | ⭐⭐⭐⭐ |

---

## Architecture Recommendations for the NPU MoE Offloading Project

> Based on background.md: NPU M.2 card, 56 TOPS INT8, 10 GB HBM @ 1200 GB/s, PCIe 4.0 ×4 @ ~8 GB/s.
> Target: Qwen3-35B-A3B Q4, >25 tok/s decode, TTFT <2s @ 1k tokens, batch=1.

### Quick Bandwidth Analysis

| Resource | Bandwidth | Notes |
|----------|-----------|-------|
| NPU HBM | 1200 GB/s | Expert compute, hot weights |
| CPU DRAM (host) | ~68 GB/s | Warm experts computed on CPU |
| PCIe 4.0 ×4 | ~8 GB/s | NPU ↔ CPU transfer |
| NVMe SSD | ~8 GB/s | Cold expert storage (shared PCIe!) |

**Critical insight**: PCIe and SSD share the same 8 GB/s bus. Simultaneous SSD reads + NPU transfers = ~4 GB/s each—severe bottleneck. Maximize cache hit rate to avoid SSD reads during NPU operation.

**Qwen3-35B-A3B memory estimate (Q4_K_M):**
- Total: ~22 GB at 4-bit (ballpark)
- Non-expert (attention, embedding, norm): ~6–7 GB
- Expert weights: ~15 GB
- Active per forward pass: ~1.5 GB (3B active params × 0.5 bytes/param at 4-bit)
- NPU HBM: 10 GB → fits all non-expert + ~3–4 GB of hot experts (~20–25% of expert pool)

### Recommended Three-Tier Architecture

```
┌─────────────────────────────────────────────────────┐
│  Tier 0: NPU HBM (10 GB @ 1200 GB/s)               │
│  • Non-expert weights: ~6 GB (permanent)            │
│  • Hot expert cache: ~3 GB (~20% of expert pool)    │
│  • Expert activation predictor: ~50 MB              │
│  • Buffer for in-flight DMA from CPU: ~0.5 GB       │
├─────────────────────────────────────────────────────┤
│  Tier 1: CPU DRAM (~8–16 GB allocated @ 68 GB/s)   │
│  • Warm expert cache (LRU/frequency): ~8 GB         │
│  • Compute warm experts using CPU SIMD (AVX2/AMX)  │
│  • SSD read buffer: ~1 GB                          │
├─────────────────────────────────────────────────────┤
│  Tier 2: NVMe SSD (full model ~22 GB + OS)         │
│  • Cold expert storage (Q4 quantized)               │
│  • Only accessed on CPU DRAM cache miss             │
│  • **Avoid during active NPU decode phase**         │
└─────────────────────────────────────────────────────┘
```

### Recommended Expert Prediction Pipeline

Based on Pre-Attention Prediction (2511.10676) + Fate (2502.12224) + DALI (2602.03495):

```
Token T enters NPU
│
├── [Tiny predictor] Predict expert set for layers T+1 to T+k
│     • Linear cross-layer gate (Fate) OR residual-based (DALI)
│     • Calibrated offline, runs on NPU at <0.1ms overhead
│
├── [Async DMA] Begin PCIe prefetch of predicted non-resident experts
│     • From CPU DRAM or SSD depending on tier
│     • Overlaps with attention computation
│
├── [NPU] Compute: attention + hot expert FFN layers
│     • Hot experts = those permanently in NPU HBM cache
│     • Use NPU GEMM for INT4/INT8 compute @ 1200 GB/s
│
├── [Sync point] Wait for DMA if predicted experts not yet arrived
│     • If predictor was accurate (85–95%), no stall
│
└── [CPU] Warm expert FFN computation (HybriMoE/TriMoE pattern)
      • CPU handles experts that are in DRAM but not NPU HBM
      • NPU handles rest; exchange activations via PCIe
```

### Recommended Cache Replacement Policy

Based on DALI (2602.03495) + FlashMoE (2601.17063) + Cache Theory (2509.02408):

1. **Offline profiling phase** (optional, improves hit rate):
   - Run 200–500 sample prompts through Qwen3
   - Measure expert activation frequency → build "static hot" set
   - Permanently cache top-20% most frequent experts in NPU HBM

2. **Online frequency+recency scoring** (baseline, always active):
   - Track per-expert: activation count + last-used timestamp
   - Score = α × frequency + (1-α) × recency, α ≈ 0.6
   - Evict lowest-score when cache full

3. **ML-based predictor** (enhancement, add if baseline insufficient):
   - Lightweight LSTM or linear cross-layer predictor (FlashMoE/Fate)
   - Prefetch predicted high-score experts before miss occurs
   - Expected: +20–35% hit rate improvement over pure LRU

### Product Variant Design Differences

| Aspect | Low-Cost Variant (SSD Offload) | Performance Variant (DRAM Offload) |
|--------|-------------------------------|-------------------------------------|
| Expert storage | NVMe SSD (full 22 GB) | CPU DRAM (16–32 GB allocated) |
| CPU role | Minimal (just routing) | Active warm-expert compute (HybriMoE) |
| SSD I/O | Yes (cold expert reads) | No (everything in DRAM) |
| PCIe usage | SSD + NPU compete | NPU exclusive use |
| Expected tok/s | 10–20 tok/s | >25 tok/s (target) |
| TTFT | 2–5s (SSD latency) | <2s (DRAM latency) |
| Main bottleneck | PCIe 4.0 ×4 shared with SSD | PCIe 4.0 ×4 (NPU↔DRAM) |
| Key papers | FlashMoE, EdgeMoE, SSD Harmful | HybriMoE, TriMoE, DALI, NPUMoE |

### Implementation Priority for llama.cpp

**Phase 1 (Core, no prediction):**
- LRU expert cache manager in `ggml_backend_buffer` layer
- Async PCIe DMA for expert weight transfer (CPU→NPU direction)
- Three-tier weight store: NPU HBM / CPU buffer / SSD mmap
- Measure hit rates and profiling baselines

**Phase 2 (Prediction):**
- Cross-layer gate predictor (simple linear, calibrated offline)
- Prefetch queue management (depth = 2–3 layers)
- Overlap attention compute with expert DMA

**Phase 3 (Hybrid CPU compute):**
- CPU-side Q4 expert GEMM (AVX2 / AMX if available)
- Dynamic CPU/NPU assignment per expert (HybriMoE pattern)
- Threshold-based: use CPU if expert not arriving in time for NPU

**Phase 4 (Speculative decode, optional):**
- Self-speculative decoding using first N transformer layers as draft
- Amortize PCIe I/O across multiple speculated tokens (SpecMoE/MoE-SpeQ)

### Key Papers to Study First (Ranked)

1. **arXiv:2604.18788** (NPUMoE) — NPU-specific MoE inference patterns
2. **arXiv:2603.01058** (TriMoE) — Three-tier hot/warm/cold expert classification
3. **arXiv:2504.05897** (HybriMoE) — CPU+GPU hybrid with impact-driven prefetch
4. **arXiv:2602.03495** (DALI) — Residual-based prediction + workload-aware cache
5. **arXiv:2511.10676** (Pre-Attention Prediction) — Overlap attention with expert prefetch
6. **arXiv:2601.17063** (FlashMoE) — ML-based cache for SSD offloading
7. **arXiv:2401.14361** (MoE-Infinity) — Batch=1 sparsity-aware cache baseline
8. **arXiv:2312.17238** (Eliseev & Mazur) — Original offloading design patterns
