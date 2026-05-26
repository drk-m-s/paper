# Fate: Fast Edge Inference of Mixture-of-Experts Models via Cross-Layer Gate

## 📋 Metadata

| Field | Value |
|-------|-------|
| **Authors** | Zhiyuan Fang, Zicong Hong, Yuegui Huang, Yufeng Lyu, Wuhui Chen, Yue Yu, Fan Yu, Zibin Zheng |
| **Organization(s)** | Sun Yat-sen University (SYSU) |
| **Venue** | arXiv preprint (arXiv:2502.12224) |
| **Publication Date** | February 2025 |
| **Citations** | ~0–5 (recent preprint) |
| **arXiv / DOI** | arXiv:2502.12224 |
| **Open-Source Code** | Not found (no GitHub link in paper) |

---

## 🔍 Executive Summary

Fate is a lightweight, elegantly simple MoE inference system for **edge scenarios**. Its key insight: adjacent MoE layers have **high cosine similarity between their gate inputs** (>78%), enabling accurate cross-layer expert prefetching **without any training or complex prediction models**. Fate uses the previous layer's gate input (`Gate_in_i`) to predict which experts will be activated at the next layer. It combines this with a shallow-favoring cache strategy (cache all experts in early layers where prediction is harder), ARC eviction, phase-differentiated optimization (prefill = popularity reordering, decode = accurate prediction), and popularity-aware hybrid quantization (INT4/INT2). Evaluated on high-end (RTX 3090) and low-end (RTX 1080 Ti) edge PCs, it achieves up to **4.5× prefill speedup** and **3.7× decode speedup** over Load on Demand at batch-size-1, with <1% accuracy loss.

---

## 📐 10-Dimension Analysis

### 1. MoE Inference vs. Training

| Aspect | Detail |
|--------|--------|
| **Type** | **Inference only** (serving/deployment) |
| **Explanation** | Fate is purely an inference system targeting edge deployment. It does not address MoE training at all. The entire design focuses on the autoregressive decoding loop and prefill stage of MoE inference under offloading scenarios. |

### 2. Edge vs. Cloud Scenario

| Aspect | Detail |
|--------|--------|
| **Target** | **Edge — explicitly batch-size = 1** |
| **Batch Size in Experiments** | 1 (single-token autoregressive decoding; single-user scenario) |
| **Hardware Constraints Mentioned** | High-end edge PC: RTX 3090 24GB + PCIe 3.0 x16; Low-end edge PC: RTX 1080 Ti 11GB + PCIe 1.0 x16 (4 GB/s!) |

The paper is titled "Fast **Edge** Inference" and explicitly targets edge scenarios. The low-end PC setup with RTX 1080 Ti and PCIe 1.0 x16 (4 GB/s bandwidth) is remarkably similar to our NPU's 8 GB/s PCIe 4.0 x4 — Fate's low-end bandwidth is actually **half** of ours. This makes Fate's performance on low-end hardware directly informative for our project.

### 3. Hardware Setup

| Aspect | Detail |
|--------|--------|
| **Compute Device** | **GPU only** (CPU used only for gate-based prediction, not for expert compute) |
| **Specific Hardware (High-end)** | NVIDIA **RTX 3090 24GB** + Intel i9-9900X + 128GB RAM + PCIe 3.0 x16 (16 GB/s) |
| **Specific Hardware (Low-end)** | NVIDIA **RTX 1080 Ti 11GB** + Intel E5-2650v4 + 64GB RAM + PCIe 1.0 x16 (4 GB/s) |
| **GPU Memory** | 24 GB (high-end), 11 GB (low-end) |

The CPU is used for gate computation and prediction (small overhead since gate/input sizes are tiny), not for expert computation. This is GPU-only compute with CPU-assisted prediction — different from LayerScope's CPU+GPU hybrid compute, but simpler.

### 4. Offloaded Weight Storage

| Aspect | Detail |
|--------|--------|
| **Storage Location** | **CPU DRAM** (host main memory) |
| **Storage Capacity** | 128 GB (high-end), 64 GB (low-end) |
| **Assumed Bandwidth** | PCIe 3.0 x16: 16 GB/s (high-end); PCIe 1.0 x16: 4 GB/s (low-end) |

Experts are stored in CPU memory. The paper also stores **dual-quantized versions** (INT4 + INT2) of all experts in CPU memory to enable dynamic quantization during prefill — the memory overhead of storing both versions is negligible for CPU DRAM.

### 5. MoE Models Evaluated

| Model | Total Params | Active Params | # Experts | Notes |
|-------|-------------|---------------|-----------|-------|
| **DeepSeek-MoE** | 16.4B | ~2.4B | 64 experts (26 layers), top-6 | Fine-grained + shared expert structure |
| **Qwen1.5-MoE** | 14.3B | ~2.7B | 60 experts (24 layers), top-4 | Direct predecessor of our target Qwen3.6-35B-A3B |

Both models use the "shared expert + fine-grained expert" architecture. Qwen1.5-MoE is directly relevant as it's from the same model family as our target.

### 6. Model Modification / Retraining

| Aspect | Detail |
|--------|--------|
| **Architecture Changes Required?** | **No** |
| **Retraining Required?** | **No** |
| **Plug-and-Play with Off-the-Shelf Weights?** | **Yes** |
| **Details** | Fate operates entirely at the inference runtime level. The cross-layer gate prefetch simply reads the gate input hidden state from one layer and uses it to predict experts for the next layer. No model modification, no fine-tuning, no additional prediction network training. This is the simplest approach among all papers analyzed. |

### 7. Offline Profiling Required?

| Aspect | Detail |
|--------|--------|
| **Profiling Needed?** | **Minimal — only lightweight offline statistics** |
| **What is Profiled?** | (1) Average computation times per layer: `t_moe`, `t_attn`, `t_gate`; (2) I/O time per expert: `t_expert`; (3) Shallow/deep layer boundary `L` via offline accuracy analysis; (4) Per-layer prefetch accuracy statistics |
| **Profiling Duration** | Negligible — just a few inference runs to collect timing and accuracy stats |
| **Profiling Dataset** | Not specified (just needs representative inference runs) |

**This is a standout feature.** Unlike LayerScope (30 epochs of LLaPor training) or MoE-Infinity (EAM collection), Fate requires **no training whatsoever**. The prediction is based purely on cosine similarity of gate inputs — a property that emerges from model architecture, not learned behavior. The offline profiling is just collecting timing statistics and determining the shallow/deep boundary `L`. This aligns perfectly with our requirement for lightweight offline profiling.

### 8. Open-Source Code

| Aspect | Detail |
|--------|--------|
| **Available?** | **Not yet** (arXiv preprint, no GitHub link) |
| **GitHub URL** | N/A |
| **License** | N/A |

The paper is an arXiv preprint (Feb 2025). No code repository is mentioned. However, given the simplicity of the approach, reimplementing the core ideas would be straightforward.

### 9. Performance Results

| Metric | Value | Baseline | Notes |
|--------|-------|----------|-------|
| **Prefill Speed (High-end, DeepSeek, 512 tok)** | **804 tok/s** | Load on Demand: ~180 tok/s | 4.5× speedup |
| **Prefill Speed (Low-end, DeepSeek, 512 tok)** | **182 tok/s** | Load on Demand: ~100 tok/s | 1.8× speedup |
| **Decode Speed (High-end, Qwen1.5)** | **14 tok/s** | Load on Demand: ~3.7 tok/s | 3.7× speedup |
| **Decode Speed (Low-end, Qwen1.5)** | **3.7 tok/s** | Load on Demand: ~1.5 tok/s | 2.3× speedup |
| **Decode Speed (High-end, DeepSeek)** | **8.9 tok/s** | EAP: 4.1 tok/s | 2.2× over EAP |
| **Decode Speed (Low-end, DeepSeek)** | **2.6 tok/s** | EAP: 2.2 tok/s | 1.2× over EAP |
| **Memory Budget Scaling (7→24GB)** | Linear improvement | | Fate scales well across budgets |
| **Prefetch Accuracy (Decode)** | **78.79%** (base) | | Improved to >99% with shallow-favoring cache |
| **Shallow Layer Cache Hit Rate** | **99.08%** | | Caching layers 0-3 |
| **Accuracy Loss (High-end PC)** | **<1%** | BF16 baseline | INT4 cache + INT2 prefill for non-popular |
| **Accuracy Loss (Low-end PC)** | **~1.8% avg** | BF16 baseline | INT2 fallback on mispredict |

**Key observations for our NPU:**
- Fate achieves **3.7 tok/s decode on low-end PC with PCIe 1.0 x16 (4 GB/s)** — our NPU has PCIe 4.0 x4 (8 GB/s), double the bandwidth. Even if we only match Fate's efficiency, we'd expect better performance on our hardware.
- The low-end PC's 11GB GPU is similar to our NPU's 10GB DRAM budget. Fate proves that with aggressive caching + quantization, 10-11GB is sufficient.
- Fate's 14 tok/s decode on high-end (PCIe 3.0 x16 = 16 GB/s) suggests our target of 25 tok/s may require the full 8 GB/s PCIe bandwidth plus efficient NPU compute.

### 10. Techniques & Innovation

**Core innovation:** **Training-free cross-layer expert prefetch using gate input cosine similarity.**

**Key mechanisms:**

1. **Cross-Layer Gate Prefetch:** The most elegant idea — use `Gate_in_i` (the hidden state fed into layer i's gate) to predict experts for layer i+1. This works because adjacent gates have >78% cosine similarity. The prediction runs on CPU in parallel with GPU computation, adding zero GPU overhead. The gate input is tiny, so CPU cost is negligible.

2. **Similar Routing Weight Expansion:** When the base prediction misses (the correct expert is #5 instead of top-4), Fate observes that routing weight distributions are nearly identical between predicted and actual. So it expands prefetch to include additional experts with similar routing weights — essentially "prefetch top-k+α" where α is determined by available I/O time budget `n = T / t_expert`.

3. **Phase-Differentiated Optimization:**
   - **Prefill (I/O-intensive):** Popularity-based computation reordering — transfer and compute the most popular experts first, using their longer compute time to hide I/O for less popular experts. Reduces pipeline bubbles.
   - **Decode (latency-sensitive):** Focus on accurate prediction and full I/O-compute overlap. Pre-computes `n` (max prefetchable experts per layer based on timing budget).

4. **Shallow-Favoring Expert Cache:** Key insight: shallow layers have lower prefetch accuracy (routing weights are more uniform), while deep layers have higher accuracy (clear expert preferences). So Fate caches ALL experts in shallow layers (boundary L=3 for Qwen) and distributes remaining cache evenly across deep layers. Uses ARC (Adaptive Replacement Cache) for eviction.

5. **Popularity-Aware Hybrid Quantization:** Stores both INT4 and INT2 versions of all experts in CPU memory. During prefill, popular experts are transferred as INT4, non-popular experts (bottom 25%) as INT2 — reducing I/O with negligible accuracy impact. During decode, all experts are INT4 since each token's experts are equally important.

**Builds upon / combines:**
- Inspired by InfiniGen's hidden state similarity analysis
- HQQ (Half-Quadratic Quantization) for fast quant/dequant
- ARC (Adaptive Replacement Cache) for cache eviction
- Concepts from MoE-Infinity's expert caching but without EAM tracing overhead

**What makes it special:** The simplicity. No ML training, no complex prediction models, no per-model predictor networks. Just exploiting a fundamental property of MoE architectures (gate input cosine similarity) in a clever way. This is the most "llama.cpp-friendly" approach we've seen — it could be implemented as a straightforward modification to the inference loop.

---

## 🎯 Relevance Assessment

| Criterion | Rating |
|-----------|--------|
| **Overall Relevance to Edge MoE Inference** | ⭐⭐⭐⭐⭐ (5/5) |

**Key Strengths:**
- **Explicitly targets edge batch-size-1** — the paper's entire motivation aligns with our use case
- **No model modification, no training, no ML predictors** — the simplest approach analyzed; extremely llama.cpp-friendly
- **Evaluated on low-end hardware with PCIe 1.0 x16 (4 GB/s)** — our NPU has PCIe 4.0 x4 (8 GB/s), double the bandwidth; Fate's low-end results are a lower bound for our expected performance
- **Evaluates Qwen1.5-MoE** — from the same model family as our target Qwen3.6-35B-A3B; the shallow-favoring cache insight (L=3 boundary) may transfer directly
- **Minimal offline profiling** — just timing stats and layer boundary detection, no training needed
- **Phase-differentiated design** — separately optimizes prefill (I/O-heavy, popularity reordering) and decode (latency-sensitive, accurate prediction)
- **INT4/INT2 hybrid quantization** — practical technique for reducing I/O that maintains accuracy
- **Elegant and simple** — the cross-layer gate prefetch concept is the most elegant we've seen; easy to understand and implement

**Key Limitations / Gaps:**
- **GPU-only compute, not NPU** — Fate computes experts on GPU; adapting to NPU with 56 TOPS INT8 requires reimplementing compute kernels
- **No CPU compute fallback** — unlike LayerScope which can run experts on CPU, Fate only uses CPU for lightweight gate prediction. Our hybrid NPU+CPU compute requirement is not addressed.
- **No SSD offloading** — only CPU DRAM → GPU offloading. Our cost-oriented market's SSD tier is not covered.
- **No open-source code** — but the approach is simple enough to reimplement
- **Qwen1.5-MoE is 14.3B, our target is 35B** — the cache/prefetch behavior of a 3× larger model may differ; the shallow-favoring boundary L might change
- **Decode speed on low-end is 3.7 tok/s** — far from our 25 tok/s target. However, our NPU has 2× PCIe bandwidth and 10× memory bandwidth (1200 GB/s vs. GPU GDDR5), so direct comparison is misleading.

**Actionable for Our Project?**
**Yes — this is the most actionable paper we've analyzed.** The cross-layer gate prefetch mechanism is trivially implementable in llama.cpp: after computing the gate at layer i, use its input hidden state to predict experts for layer i+1. The shallow-favoring cache strategy can be directly applied by caching all experts in the first L layers in NPU DRAM. The phase-differentiated approach (popularity reordering for prefill, accurate prediction for decode) maps directly to our requirements. The popularity-aware INT4/INT2 quantization is a practical technique for reducing PCIe traffic. Compared to MoE-Infinity (requires EAM tracing) and LayerScope (requires LLaPor training), Fate's approach is the simplest to implement and the most aligned with our "clean and lean" design philosophy.

---

## 📝 Notes & Discussion Points

- **Comparison with MoE-Infinity:** Both target batch=1 edge. MoE-Infinity uses EAM tracing (online, more complex but potentially more adaptive). Fate uses gate cosine similarity (offline stats, simpler but fixed accuracy). For our project, Fate's simplicity is more attractive — we could start with Fate's approach and add MoE-Infinity's EAM tracing later if prediction accuracy is insufficient.

- **NT4 INT2 for NPU:** Our NPU has 56 TOPS INT8 compute. Fate's INT4/INT2 quantization for I/O reduction is orthogonal to compute precision — we could use INT4/INT2 for PCIe transfer and dequantize to INT8 for NPU compute. This would further reduce I/O without affecting compute throughput.

- **PCIe bandwidth comparison:** Fate on low-end (PCIe 1.0 x16 = 4 GB/s) achieves 3.7 tok/s decode for Qwen1.5-MoE (14.3B). Our NPU has PCIe 4.0 x4 = 8 GB/s. Assuming linear scaling with bandwidth and accounting for the larger model (35B vs 14.3B), we might achieve 3-5 tok/s with Fate's approach. To reach 25 tok/s, we'd need additional optimizations like NPU compute acceleration (56 TOPS vs. GPU TFLOPS) and the 1200 GB/s NPU DRAM bandwidth advantage.

- **Shallow-favoring cache for Qwen3.6-35B-A3B:** Fate identifies L=3 for Qwen1.5-MoE (24 layers). For Qwen3.6-35B-A3B with ~48 layers, the boundary might be different. We should profile this on our target model — it's one of the few profiling steps needed.

- **ARC vs. LRU:** Fate uses ARC (Adaptive Replacement Cache) instead of LRU. ARC balances recency and frequency, which is important because expert activation patterns have both short-term (recency within a sequence) and long-term (frequency across sequences) components. For llama.cpp integration, ARC adds modest complexity over LRU but could improve hit rates under mixed workloads.

- **Prefetch accuracy ceiling:** Fate achieves ~79% base prediction accuracy from gate cosine similarity alone. Improving this with EAM tracing (MoE-Infinity) or learned predictors (LayerScope) could push it higher, but the shallow-favoring cache already compensates for lower accuracy in early layers by simply caching everything there.
