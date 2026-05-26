# LayerScope: Predictive Cross-Layer Scheduling for Efficient Multi-Batch MoE Inference on Legacy Servers

## 📋 Metadata

| Field | Value |
|-------|-------|
| **Authors** | Enda Yu, Dezun Dong, Zhaoning Zhang, Zhe Bai, Weiling Yang, Haojie Wang, Dongsheng Li, Yongwei Wu, Xiangke Liao |
| **Organization(s)** | National University of Defense Technology (NUDT), Tsinghua University |
| **Venue** | **ICS '26** (International Conference on Supercomputing), July 2026, Belfast, UK |
| **Publication Date** | July 2026 |
| **Citations** | ~0 (brand new, just published at ICS 2026) |
| **arXiv / DOI** | DOI: 10.1145/3797905.3807834 |
| **Open-Source Code** | Not found (paper just published; no GitHub link in paper) |

---

## 🔍 Executive Summary

LayerScope is a prediction-driven cross-layer expert scheduling system targeting **multi-batch MoE inference on legacy/commodity servers** with limited GPU memory. It addresses three challenges: inaccurate expert activation prediction, PCIe bandwidth contention between prefetch and on-demand loads, and cross-device scheduling complexity. The system introduces three co-designed components: **LLaPor** (a learnable layer-group-aware predictor achieving >90% Top-4 accuracy), **PreSched** (a cross-layer scheduler that globally optimizes prefetch vs. on-demand tradeoffs), and **AsyncIO** (an asynchronous I/O optimizer). Evaluated on Mixtral, DeepSeek, Qwen3, and Moonlight, it achieves **141% higher throughput** and **74.6% lower decoding latency** vs. state-of-the-art.

---

## 📐 10-Dimension Analysis

### 1. MoE Inference vs. Training

| Aspect | Detail |
|--------|--------|
| **Type** | **Inference only** (serving/deployment) |
| **Explanation** | LayerScope is purely an inference serving system. It does not address MoE training. The entire design focuses on the auto-regressive decoding loop: predicting expert activation for upcoming layers, scheduling prefetch/on-demand loads, and overlapping I/O with compute. |

### 2. Edge vs. Cloud Scenario

| Aspect | Detail |
|--------|--------|
| **Target** | **Multi-batch server** (NOT edge/batch=1) |
| **Batch Size in Experiments** | **4–64** (multi-batch, multi-request serving) |
| **Hardware Constraints Mentioned** | Legacy/commodity servers with 24GB GPU memory limit, PCIe 3.0/4.0 |

The paper explicitly targets "multi-batch MoE inference on legacy servers" and "resource-constrained environments." Batch sizes evaluated are 4–64, which is server-side continuous batching, not edge single-user scenarios. The paper mentions single-batch inference only in passing (referring to kTransformers as suitable for single-batch). **This is fundamentally a cloud/server system, not edge.**

### 3. Hardware Setup

| Aspect | Detail |
|--------|--------|
| **Compute Device** | **CPU + GPU hybrid** (both CPU and GPU compute experts) |
| **Specific Hardware** | NVIDIA **V100** (PCIe 3.0 x16) & **A100** (PCIe 4.0 x16), both capped at **24GB** memory |
| **GPU Memory** | 24 GB (artificially limited to simulate consumer GPUs like RTX 4090) |
| **CPU** | Intel Xeon Gold 6230R |

CPU+GPU hybrid compute is a core feature: LayerScope dynamically decides whether each expert runs on CPU or GPU based on a cross-layer cost model. The system maintains a Hot Expert Table that preloads the hottest experts into GPU memory.

### 4. Offloaded Weight Storage

| Aspect | Detail |
|--------|--------|
| **Storage Location** | **CPU DRAM** (host main memory) |
| **Storage Capacity** | Sufficient to hold full models (up to ~80GB for Mixtral) |
| **Assumed Bandwidth** | PCIe 3.0 x16 (~12 GB/s) and PCIe 4.0 x16 (~24 GB/s) |

Experts are stored in CPU host memory. Hot experts are preloaded into GPU memory at initialization. Cold experts are fetched on-demand or prefetched via AsyncIO's pinned-memory, multi-stream transfer mechanism. SSD offloading is not explored. The paper achieves 70–80% PCIe bandwidth utilization.

### 5. MoE Models Evaluated

| Model | Total Params | Active Params | # Experts | Notes |
|-------|-------------|---------------|-----------|-------|
| **Mixtral-8×7B** | 46.7B | 12.9B | 8 experts (32 layers) | Large-expert model, compute-intensive |
| **DeepSeek-V2-Lite** | 16B | 2.4B | 64 experts (26 layers) | Small-expert model, I/O-bound |
| **Qwen3-30B-A3B** | ~30B | 3B | 128 experts (48 layers) | Small-expert, our target model family |
| **Moonlight-16B-A3B** | 16B | 3B | 64 experts (26 layers) | Small-expert model |

Notably includes Qwen3-30B-A3B — directly relevant to our target qwen3.6-35B-A3B.

### 6. Model Modification / Retraining

| Aspect | Detail |
|--------|--------|
| **Architecture Changes Required?** | **No** |
| **Retraining Required?** | **No** (model weights untouched) |
| **Plug-and-Play with Off-the-Shelf Weights?** | **Yes** |
| **Details** | LayerScope operates at the inference runtime level. It does not modify model architecture or require fine-tuning of the MoE model itself. However, it **does require training LLaPor** — a separate lightweight predictor network — which is a system component, not a model modification. |

### 7. Offline Profiling Required?

| Aspect | Detail |
|--------|--------|
| **Profiling Needed?** | **YES — requires offline training of LLaPor predictor + hot-expert table construction** |
| **What is Profiled?** | (1) Hot-expert ranking by token count; (2) Per-layer records (hidden state `a`, expert routing `e_i, w_i`) for LLaPor training dataset; (3) Per-operation timing (GPU compute, CPU compute, I/O transfer) |
| **Profiling Duration** | 128 batches × 512 tokens each; LLaPor training: 30 epochs × ~1.2s/epoch on A100 = **~36 seconds total** |
| **Profiling Dataset** | ShareGPT (primary); also validated on DPO and LLaVA for generalization |

LLaPor requires significant offline training (30 epochs on 128 batches). The system also supports **online fine-tuning**: a lightweight cycle triggered when prediction error exceeds 10% over a sliding window of 64 tokens, completing in **1.8ms per iteration**. This is a notable feature — adapts to distribution shifts at low cost.

### 8. Open-Source Code

| Aspect | Detail |
|--------|--------|
| **Available?** | **Not yet** (paper published July 2026 — may be released later) |
| **GitHub URL** | N/A |
| **License** | N/A (paper is CC BY 4.0) |

The paper was just published at ICS '26 (July 2026). No code repository link is provided in the paper. Given the authors are from NUDT (military university), open-sourcing may face additional restrictions.

### 9. Performance Results

| Metric | Value | Baseline | Notes |
|--------|-------|----------|-------|
| **Throughput Improvement** | **+141%** over Klotski, **+70.8%** over HybriMoE | Multi-batch baselines | Best overall improvement |
| **Decoding Latency Reduction** | **−74.6%** vs Klotski, **−42.1%** vs HybriMoE | | |
| **DeepSeek-V2 A100 batch=64** | **41.7 tok/s** throughput | Klotski: 26.8, HybriMoE: 30.0 | Small-expert (I/O-bound) |
| **Qwen3-30B A100 batch=64** | **17.4 tok/s** throughput | | Our target model family |
| **Moonlight A100 batch=64** | **59.5 tok/s** throughput | | |
| **Qwen3-30B V100 batch=64** | **3.74 s/token** latency | Klotski: 14.7s, Fate: 7.0s | 74.6% over Klotski |
| **Mixtral V100 batch=64** | **2.79 s** latency | Klotski: 6.05s, Fate: 4.02s | Large-expert (compute-bound) |
| **LLaPor Top-4 Accuracy** | **≥94%** across all models | Fate: ~50-80% | Token coverage error <5% |
| **LLaPor Inference** | **0.12–0.48 ms** per prediction | | Negligible overhead, hidden in I/O gaps |
| **LLaPor Memory** | **0.5–2.8 MB per layer** | | Total: ~15-90 MB across all layers |
| **PCIe Bandwidth Util.** | **70–80%** | | AsyncIO fine-grained chunking |

**Key observation:** Performance is reported in multi-batch settings (batch 4–64). Single-batch (batch=1) numbers are not provided, making direct comparison to our edge scenario difficult.

### 10. Techniques & Innovation

**Core innovation:** **Predictive cross-layer scheduling with layer-group-aware expert activation prediction.**

**Key mechanisms:**

1. **LLaPor (Learnable Layer-Aware Predictor):** A lightweight neural predictor with distinct architectures for three layer groups:
   - **Input/output groups:** Shallow networks with strong PCA compression (4096→256 dim), using hybrid loss (expert-balancing + focal loss) to handle class imbalance. These layers have higher routing correlation and skewed gating weights.
   - **Middle groups:** Deeper networks with milder PCA (4096→512 dim), residual blocks, and a gating unit that modulates based on hidden state `a`. These layers have higher cosine similarity and dominant hot experts.
   - Supports online continuous learning (1.8ms per fine-tuning cycle).

2. **PreSched (Prefetch-Aware Cross-Layer Scheduler):** Instead of greedy per-layer optimization, jointly models the current layer and the prefetch-target layer. Formulates a cost model with 5 premises (serial I/O, computation masking, prefetch non-interruptibility, on-demand monotonicity, etc.) and makes globally optimal decisions about which experts to load on-demand vs. prefetch, and whether to compute on CPU vs. GPU.

3. **AsyncIO (Asynchronous I/O Optimizer):** Decouples expert transfer from computation using pinned memory, async copies, dual buffer rotation, and fine-grained chunking (splitting each expert's 3 weight matrices across parallel CUDA streams). Achieves 70-80% PCIe utilization.

4. **Layer-group decomposition insight:** First to systematically characterize MoE layers as belonging to input, output, and middle groups with distinct activation properties — and design the predictor accordingly.

---

## 🎯 Relevance Assessment

| Criterion | Rating |
|-----------|--------|
| **Overall Relevance to Edge MoE Inference** | ⭐⭐⭐ (3/5) |

**Key Strengths:**
- **Evaluates Qwen3-30B-A3B** — directly relevant to our target qwen3.6-35B-A3B model family
- **CPU+GPU hybrid compute** — aligns with our requirement for hybrid NPU+CPU compute option
- **No model modification** — plug-and-play with off-the-shelf weights
- **Layer-group insight is transferable** — the finding that input/output/middle layers have distinct activation properties applies to any MoE model, including ours
- **Online fine-tuning mechanism** — LLaPor's lightweight continuous learning (1.8ms) is practical and could be adapted to our offline profiling requirement
- **Strong engineering** — AsyncIO's PCIe optimization techniques (pinned memory, multi-stream, fine-grained chunking) are directly applicable to our NPU's PCIe 4.0 x4 (8 GB/s) link

**Key Limitations / Gaps:**
- **Multi-batch focus (batch 4–64)** — the system is designed for server-side continuous batching, not edge batch-size-1. Single-batch performance is not characterized. Many optimizations (cross-layer prefetch, CPU offload of cold experts) may be less effective or unnecessary at batch=1.
- **Requires significant offline training** — LLaPor needs 30 epochs of training on 128 batches (~36s on A100). While the online fine-tuning is lightweight, the initial training adds deployment complexity for edge scenarios.
- **GPU compute assumed, not NPU** — the entire system is CUDA-based. Adapting to NPU with 56 TOPS INT8 would require re-implementing GPU kernels.
- **No SSD offloading** — only CPU DRAM → GPU offloading. Our cost-oriented market requires SSD tier.
- **LLaPor predictor is model-specific** — each MoE model needs its own trained predictor, adding per-model deployment overhead.
- **Not open-source (yet)** — paper is brand new (July 2026), and NUDT affiliation may limit open-sourcing.

**Actionable for Our Project?**
**Partially.** The layer-group insight and AsyncIO techniques are directly transferable. The cross-layer scheduling concept could be simplified for batch=1. However, the multi-batch focus, GPU-only compute, and LLaPor's offline training requirements make direct adoption challenging. The most valuable takeaways are: (1) the characterization of layer-group properties (input/output/middle), (2) the AsyncIO fine-grained transfer techniques for maximizing PCIe utilization, and (3) the hot-expert table preloading strategy.

---

## 📝 Notes & Discussion Points

- **Comparison with MoE-Infinity:** MoE-Infinity targets batch=1 with sparsity-aware caching via EAM tracing (online, no training). LayerScope targets batch 4–64 with learned prediction via LLaPor (offline training required). They solve different problems. MoE-Infinity is more directly applicable to our edge scenario, but LayerScope has better techniques for hybrid CPU-GPU compute and PCIe optimization.

- **Qwen3-specific insights:** The paper evaluates Qwen3-30B-A3B (128 experts, 48 layers). For this model, LayerScope achieves 17.4 tok/s throughput on A100 at batch=64. Single-batch performance would be lower. The layer-group properties observed on Qwen3 could inform our own cache/prefetch design.

- **LLaPor complexity vs. benefit:** The LLaPor predictor requires per-model training (30 epochs) but achieves >94% Top-4 accuracy with only 0.5-2.8MB per layer. For our project, we might consider a simpler predictor (like MoE-Infinity's EAM tracing) to avoid the offline training burden, unless the accuracy gains justify the cost.

- **AsyncIO for NPU:** The fine-grained chunking technique (splitting gate_proj, up_proj, down_proj across parallel streams) could be adapted to our NPU's PCIe 4.0 x4 link. With only 8 GB/s bandwidth, maximizing utilization is critical. The 70-80% utilization achieved by AsyncIO suggests we could reach ~5.6-6.4 GB/s effective bandwidth.

- **Hot Expert Table relevance:** Preloading hot experts into GPU/NPU memory at initialization is a simple, effective technique. For our NPU with 10GB DRAM (5-8GB usable), preloading the most frequently activated experts could significantly reduce PCIe transfers.
