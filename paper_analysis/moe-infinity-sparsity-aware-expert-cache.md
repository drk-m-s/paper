# MOE-INFINITY: Efficient MoE Inference on Personal Machines with Sparsity-Aware Expert Cache

## 📋 Metadata

| Field | Value |
|-------|-------|
| **Authors** | Leyang Xue, Yao Fu, Zhan Lu, Chuanhao Sun, Luo Mai, Mahesh Marina |
| **Organization(s)** | University of Edinburgh |
| **Venue** | arXiv preprint (submitted to SOSP 2024) |
| **Publication Date** | January 2024 (arXiv: 2401.14361) |
| **Citations** | ~50+ (estimated from GitHub popularity: 305 stars, 28 forks) |
| **arXiv / DOI** | arXiv:2401.14361 |
| **Open-Source Code** | https://github.com/EfficientMoE/MoE-Infinity (Apache-2.0, 305 ⭐) |

---

## 🔍 Executive Summary

MOE-INFINITY is an MoE inference system targeting **personal machines with single consumer GPUs**. Its key insight: in batch-size-1 (single-user) scenarios, MoE models exhibit high activation sparsity — a small number of experts are frequently reused during decoding within the same request. The paper introduces a **Sparsity-Aware Expert Cache** that traces expert activation patterns at request-level (via Expert Activation Matrices) and uses them to guide cache replacement and prefetching. It achieves **3.1–16.7× latency reduction** over vLLM, Ollama, DeepSpeed, and BrainStorm across diverse MoE models (DeepSeek, Mixtral, Arctic, Switch, NLLB).

---

## 📐 10-Dimension Analysis

### 1. MoE Inference vs. Training

| Aspect | Detail |
|--------|--------|
| **Type** | **Inference only** (serving/deployment) |
| **Explanation** | MOE-INFINITY is purely an inference system. It does not address MoE training (load balancing, distributed training). The entire design — expert cache, prefetching, on-demand fetching — is built around the auto-regressive decoding loop of MoE models during inference. |

### 2. Edge vs. Cloud Scenario

| Aspect | Detail |
|--------|--------|
| **Target** | **Edge / Personal Machines** (batch-size = 1) |
| **Batch Size in Experiments** | 1 (single-user, single-prompt) |
| **Hardware Constraints Mentioned** | Single consumer GPU with 24–48 GB memory; PCIe 4.0 with 24 GB/s bandwidth |
  
The paper explicitly states: *"MoE inference on personal devices typically operates with a batch size of one. Unlike cloud-based environments that batch multiple user requests using continuous batching, personal machines are single-user environments."* This is directly aligned with our edge NPU scenario.

### 3. Hardware Setup

| Aspect | Detail |
|--------|--------|
| **Compute Device** | **GPU + CPU** (GPU compute, CPU for weight storage) |
| **Specific Hardware** | Single **NVIDIA A5000 24GB** via PCIe 4.0 (24 GB/s) |
| **GPU Memory** | 24 GB (consumer-grade) |
| **CPU Memory** | Host DRAM (sufficient to hold full model, e.g., up to 900 GB for Arctic) |

Multi-GPU expert parallelism is also implemented (Appendix B.2), with NUMA-aware expert partitioning. However, the primary target is single-GPU personal machines.

### 4. Offloaded Weight Storage

| Aspect | Detail |
|--------|--------|
| **Storage Location** | **CPU DRAM** (host main memory) |
| **Storage Capacity** | Up to 1 TB (commodity multi-GPU server host memory) |
| **Assumed Bandwidth** | PCIe 4.0: 24 GB/s (single direction), 32 GB/s (bidirectional) |

The paper states: *"The full MoE model resides in host memory, and only the activated experts are fetched into the GPU when needed."* This is a **CPU DRAM → GPU** offloading scheme, not SSD-based. Note: our project also considers SSD offloading for cost-oriented markets, which this paper does not address.

### 5. MoE Models Evaluated

| Model | Total Params | Active Params | # Experts | Notes |
|-------|-------------|---------------|-----------|-------|
| **DeepSeek-V2-Lite** | 16B | 2.4B | 64 experts (8 layers) | Primary evaluation model |
| **DeepSeek-MoE** | 236B | ~21B | 64 experts | Largest DeepSeek variant |
| **Mixtral 8x7B** | 46.7B | 12.9B | 8 experts (32 layers) | Worst-case for caching (few large experts) |
| **Arctic-MoE** | 480B | ~17B | 128 experts | Largest model tested (900 GB on disk) |
| **Switch-MoE** | 1.6T | ~32B | 128 experts | Google Switch Transformer |
| **NLLB-MoE** | 54B | ~7B | 128 experts | Meta NLLB translation model |

### 6. Model Modification / Retraining

| Aspect | Detail |
|--------|--------|
| **Architecture Changes Required?** | **No** |
| **Retraining Required?** | **No** |
| **Plug-and-Play with Off-the-Shelf Weights?** | **Yes** |
| **Details** | MOE-INFINITY works directly with HuggingFace model checkpoints. The GitHub README confirms "HuggingFace model compatible, and HuggingFace programmer friendly." No model architecture changes or fine-tuning needed. The system operates entirely at the inference runtime level by intercepting expert dispatch decisions. |

### 7. Offline Profiling Required?

| Aspect | Detail |
|--------|--------|
| **Profiling Needed?** | **No offline profiling — uses online tracing instead** |
| **What is Profiled?** | Expert Activation Matrix (EAM) — per-request traces of which experts are activated at each layer |
| **Profiling Duration** | N/A (online, adaptive). Only needs ~3% of total requests (120 entries) to capture most activation patterns |
| **Profiling Dataset** | Not applicable — traces are built during live inference from actual user prompts |

The system uses an **Expert Activation Matrix Collection (EAMC)** that accumulates traces during inference. The EAMC has a fixed capacity and uses cosine-distance-based replacement. This is an online warm-up, not an offline profiling phase. This is a significant advantage for our project — no separate profiling step needed.

### 8. Open-Source Code

| Aspect | Detail |
|--------|--------|
| **Available?** | **Yes** |
| **GitHub URL** | https://github.com/EfficientMoE/MoE-Infinity |
| **License** | Apache-2.0 |
| **Activity** | Active (305 stars, 28 forks, latest commits ~2 months ago as of May 2026) |
| **Install** | `pip install moe-infinity` (PyPI available) |
| **Note** | The open-source version has been redesigned for HuggingFace compatibility. It differs from the paper version (which prioritized extreme performance). Multi-GPU expert parallelism and KV-cache offloading are planned/in-progress. |

### 9. Performance Results

| Metric | Value | Baseline | Notes |
|--------|-------|----------|-------|
| **Speedup (DeepSeek-V2-Lite)** | **3.1–16.7×** | vs. vLLM, Ollama, DeepSpeed, BrainStorm | Per-token latency reduction |
| **TPOT Avg (DeepSeek-V2-Lite)** | **155 ms** | vLLM: 485ms, Ollama: 2590ms | On single A5000 24GB |
| **TPOT Tail (DeepSeek-V2-Lite)** | **169 ms** | vLLM: 493ms, DeepSpeed: 803ms | P99 latency |
| **GPU Idle Time** | **51 ms** | vLLM: 254ms, DeepSpeed: 513ms | Time GPU waits for expert fetch |
| **Decode Latency (Switch)** | **155 ms/token** | Comparable to 4-GPU non-offloading setup | Single GPU matching multi-GPU |
| **Decode Latency (NLLB)** | **531 ms/token** | Comparable to 8-GPU non-offloading setup | Single GPU matching multi-GPU |
| **Decode Latency (Mixtral, worst case)** | **836 ms** (total) | 1.4× over BrainStorm/Mixtral-Offloading | Few large experts → harder to cache |
| **Long Context (128K)** | Stable up to 2^15 context | Degrades after 2^16 (KV-cache fills GPU buffer) | Better than vLLM which offloads KV-cache too |
| **EAMC Overhead** | 21 μs/query (1K EAMs), 226 μs (10K) | <1% of inference latency | Negligible CPU overhead |
| **Memory: GPU Expert Cache** | 3 GB (typical, for DeepSeek-V2-Lite) | Model-dependent | Shrinks as context length grows |

**Key finding:** For models with many small experts (Switch, DeepSeek), MOE-INFINITY on a single GPU achieves latency comparable to multi-GPU non-offloading deployment. For models with few large experts (Mixtral), the advantage is smaller (1.4×).

### 10. Techniques & Innovation

**Core innovation:** **Sparsity-Aware Expert Cache** with **request-level Expert Activation Matrix (EAM) tracing**.

**Key mechanisms:**
- **Expert Activation Matrix (EAM):** An L×E matrix tracking how many tokens each expert processes per layer. Built at request-level (rEAM) and iteration-level (iEAM).
- **EAM Collection (EAMC):** Stores historical EAMs. When a new iEAM arrives, it's matched against the collection to find similar activation patterns. Cosine distance used for similarity.
- **Activation-Aware Prediction:** The predicted EAM (pEAM) is computed by aggregating matched historical EAMs and applying layer-proximity weighting `(1 − (i−l)/L)` to prioritize nearer future layers.
- **Activation-Aware Eviction:** When cache is full, the expert with the lowest predicted reuse likelihood is evicted.
- **Prefetching Integration:** Prefetch and cache replacement are unified — the predictor guides both which expert to prefetch and which to evict.

**Builds upon / combines:**
- Expert caching concepts from DeepSpeed-Inference, vLLM, Mixtral-Offloading
- Memory tracing ideas from Sentinel, DeepUM (but at expert-level granularity)
- K-means clustering for offline analysis of activation pattern groups (but NOT used for prediction — paper explicitly rules out learning-based prediction due to low group transition probabilities)

**What makes it novel:** Prior systems use LRU, statistical counting, or dependency-based prediction — all of which fail to capture the sparse, request-specific activation patterns of MoE models in batch-size-1 scenarios. MOE-INFINITY's key insight is that **activation sparsity exists at the per-request level** (not globally), and tracing it per-request enables accurate cache decisions.

---

## 🎯 Relevance Assessment

| Criterion | Rating |
|-----------|--------|
| **Overall Relevance to Edge MoE Inference** | ⭐⭐⭐⭐⭐ (5/5) |

**Key Strengths:**
- **Directly targets batch-size-1 edge scenario** — the paper's entire motivation matches our use case
- **No model modification required** — plug-and-play with HuggingFace weights, aligns with our requirement
- **No offline profiling phase** — online tracing adapts to actual user behavior, simpler deployment
- **Open-source with active development** — Apache-2.0 license, can study/adapt the cache design
- **Demonstrated on diverse MoE models** — DeepSeek, Mixtral, etc. — good coverage for our target qwen3.6-35B-A3B
- **Negligible CPU overhead** — EAM matching costs <1% of inference latency
- **Clean, well-documented architecture** — EAM concept is elegant and could be adapted to llama.cpp

**Key Limitations / Gaps:**
- **Assumes GPU compute, not NPU** — compute model is CUDA-based; adapting to NPU with 56 TOPS INT8 and different memory hierarchy requires significant work
- **CPU DRAM offloading only** — does not address SSD-based offloading, which is our cost-oriented market requirement
- **No CPU compute fallback** — unlike llama.cpp which can run experts on CPU, MOE-INFINITY always computes on GPU. Our NPU-only compute path would need a different design.
- **PCIe bandwidth assumed higher (24 GB/s) than our NPU's 8 GB/s** — the cache hit rate must be higher or prefetching must be more aggressive to compensate
- **Not integrated with llama.cpp** — MOE-INFINITY is a standalone PyTorch-based system

**Actionable for Our Project?**
**Yes — with adaptation.** The EAM-based sparsity-aware caching concept is directly transferable. The key insight that "at batch-size-1, expert activation is sparse at the per-request level" applies equally to our NPU scenario. We could implement a similar EAM tracing mechanism in llama.cpp, adapted for our lower PCIe bandwidth (8 GB/s vs 24 GB/s) and potentially extended to support SSD offloading by adding a two-tier cache (NPU DRAM ↔ CPU DRAM ↔ SSD). The paper's finding that only ~3% of requests (120 entries) are needed for effective prediction means the memory overhead of EAMC is negligible.

---

## 📝 Notes & Discussion Points

- **Mixtral as worst case:** The paper identifies Mixtral's architecture (8 large experts per layer) as the worst case for caching. Qwen3.6-35B-A3B has 128 experts with 3B active — more similar to DeepSeek-V2, which is the paper's best case. This bodes well.

- **Two-tier offloading extension:** For our cost-oriented market (SSD offloading), we could extend the EAM concept to a two-tier cache: hot experts in NPU DRAM, warm in CPU DRAM, cold on SSD. The EAM prediction could guide both tiers.

- **NPU bandwidth consideration:** Our NPU has 8 GB/s PCIe 4.0 bandwidth vs. the paper's 24 GB/s. This means expert fetch latency will be ~3× higher for the same expert size. The cache hit rate becomes even more critical for us — we may need a larger cache or more accurate prediction.

- **llama.cpp integration feasibility:** The EAM tracing is algorithm-level and could be implemented in llama.cpp's C++ codebase. The main challenge would be managing the EAMC data structure and cosine-distance matching efficiently on CPU.

- **The paper explicitly rules out learning-based prediction** (Section 4.4): group transition probabilities are too low for reliable ML-based prediction across different groups. This validates our design choice to avoid complex ML-based predictors.
