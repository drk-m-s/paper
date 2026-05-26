# PreScope: Unleashing the Power of Prefetching for Resource-Constrained MoE Inference

## 📋 Metadata

| Field | Value |
|-------|-------|
| **Authors** | Enda Yu, Zhaoning Zhang*, Dezun DONG*, Yongwei Wu, Xiangke Liao (*corresponding) |
| **Organization(s)** | National University of Defense Technology (NUDT), Tsinghua University, China |
| **Venue** | Conference submission (labeled "Conference'26", venue unconfirmed — not yet publicly indexed) |
| **Publication Date** | ~Late 2025 / Early 2026 (not yet on arXiv or Semantic Scholar as of May 2026) |
| **Citations** | 0 (not yet indexed) |
| **arXiv / DOI** | Not publicly available (paper is likely under review) |
| **Open-Source Code** | Not found — no GitHub repo discovered; paper states 6k+ lines of Python (PyTorch + HuggingFace) |

---

## 🔍 Executive Summary

PreScope is a prediction-driven, cross-layer expert scheduling system for CPU–GPU hybrid MoE inference on resource-constrained hardware. It addresses three intertwined challenges: inaccurate expert activation prediction, PCIe bandwidth competition between prefetching and on-demand loading, and high cross-device scheduling complexity. PreScope introduces three co-designed components: **LLaPor** (a layer-group-aware lightweight predictor achieving >90% Top-4 accuracy), **PreSched** (a prefetch-aware cross-layer scheduler that globally optimizes expert placement across current and next layers), and **AsyncIO** (an asynchronous I/O optimizer that decouples transfers from computation). Evaluated on Mixtral-8×7B, DeepSeek-V2-Lite, Qwen3-30B-A3B, and Moonlight-16B-A3B with V100/A100 GPUs, PreScope achieves 141% throughput improvement and 74.6% decoding latency reduction over state-of-the-art baselines.

---

## 📐 10-Dimension Analysis

### 1. MoE Inference vs. Training

| Aspect | Detail |
|--------|--------|
| **Type** | Inference |
| **Explanation** | PreScope is a pure MoE **inference** framework for serving/deployment. It focuses on optimizing the expert placement problem (which experts go to GPU vs. CPU, and when to prefetch) during autoregressive generation. It does not address MoE training at all. |

### 2. Edge vs. Cloud Scenario

| Aspect | Detail |
|--------|--------|
| **Target** | Cloud (batch > 1) — but with "resource-constrained" hardware assumptions |
| **Batch Size in Experiments** | 4 to 64 (ShareGPT dataset, 512 input tokens, 64 output tokens) |
| **Hardware Constraints Mentioned** | Single commodity GPU with 32GB VRAM (V100/A100), where full MoE models cannot fit in GPU memory. GPU memory is explicitly described as a bottleneck. However, the system assumes ample CPU DRAM is available to hold all experts. |

The paper targets "resource-constrained environments" but uses server-class GPUs (V100 32GB, A100 32GB) with batch sizes ≥ 4 — this is a **cloud/small-server** scenario, not a single-user edge scenario.

### 3. Hardware Setup

| Aspect | Detail |
|--------|--------|
| **Compute Device** | GPU + CPU (hybrid CPU–GPU co-execution) |
| **Specific Hardware** | NVIDIA V100 (32GB, PCIe 3.0 ×16) and NVIDIA A100 (32GB, PCIe 4.0 ×16); Intel Xeon Gold 6230R CPU |
| **GPU Memory** | 32GB (V100 and A100) |
| **CPU Memory** | Not explicitly stated (assumed sufficient to hold all expert weights, likely 128GB+ given server-class CPU) |

### 4. Offloaded Weight Storage

| Aspect | Detail |
|--------|--------|
| **Storage Location** | CPU DRAM (main memory) |
| **Storage Capacity** | Sufficient to hold full model weights (explicitly stated that all experts reside in CPU memory) |
| **Assumed Bandwidth** | PCIe 3.0 ×16 (~16 GB/s) for V100; PCIe 4.0 ×16 (~32 GB/s) for A100. Paper achieves 79% PCIe bandwidth utilization with pinned memory + async transfers. |

PreScope offloads experts to **CPU DRAM only** (not SSD). The paper does mention SSD offloading in related work (citing FlashMoE, Klotski) but does not use it in its own system.

### 5. MoE Models Evaluated

| Model | Total Params | Active Params | # Experts | # Layers | Notes |
|-------|-------------|---------------|-----------|----------|-------|
| Mixtral-8×7B | 47B | 13B | 8 experts/layer (2 activated) | 32 | Large experts (336MB each in BF16) |
| DeepSeek-V2-Lite | ~16B | ~2.4B | 64 experts/layer (6 activated) | 26 | Small experts (16.5MB each) |
| Qwen3-30B-A3B | 30.3B | 3.3B | 128 experts/layer (8 activated) | 48 | Very small experts (9MB each); directly relevant to our target |
| Moonlight-16B-A3B | 16B | 3B | 64 experts/layer (6 activated) | 26 | Small experts (16.5MB each) |

### 6. Model Modification / Retraining

| Aspect | Detail |
|--------|--------|
| **Architecture Changes Required?** | No (to the MoE model itself) |
| **Retraining Required?** | No (for the base MoE model) |
| **Plug-and-Play with Off-the-Shelf Weights?** | Yes |
| **Details** | PreScope does not modify the MoE model architecture or weights. LLaPor is a separate lightweight predictor per MoE block that is trained offline and loaded alongside the model. The base model uses standard HuggingFace Transformers weights without any fine-tuning. The hot expert table is populated by profiling activation frequencies — no model modification. |

### 7. Offline Profiling Required?

| Aspect | Detail |
|--------|--------|
| **Profiling Needed?** | Yes (for both LLaPor training and hot expert identification) |
| **What is Profiled?** | Hidden states (𝑎), expert activation indices, and routing weights per layer across 128 batches × 512 tokens. Per-operation costs (GPU compute time, CPU compute time, PCIe transfer time) are also recorded. A hot expert table is built by ranking (layer, expert_id) pairs by empirical activation frequency. |
| **Profiling Duration** | Training: 30 epochs × 0.6s/epoch on A100 = ~18 seconds per predictor. Online fine-tuning during inference is lighter (milliseconds). Dataset collection: 128 batches, each 512 tokens. |
| **Profiling Dataset** | Training on ShareGPT; generalization tested on DPO and LLaVA datasets. |

### 8. Open-Source Code

| Aspect | Detail |
|--------|--------|
| **Available?** | No |
| **GitHub URL** | Not found |
| **License** | N/A |

The paper states the implementation comprises "over 6k lines of Python code" using PyTorch 2.1 and HuggingFace Transformers, but no repository has been found online.

### 9. Performance Results

| Metric | Value | Baseline | Notes |
|--------|-------|----------|-------|
| **Throughput Improvement** | +141% over Klotski, +70.8% over HybriMoE | Klotski, HybriMoE, Fate, Fiddler | Across 4 models, V100+A100, batch 4–64 |
| **Decoding Latency Reduction** | −74.6% vs Klotski, −42.1% vs HybriMoE | Klotski, HybriMoE, Fate, Fiddler | Best-case per-token latency reduction |
| **A100 Throughput (batch 64)** | 37.4 tok/s (DeepSeek), 13.3 tok/s (Qwen3), 47.6 tok/s (Moonlight) | — | Absolute numbers at max batch |
| **V100 Decode Latency (batch 4)** | 0.37s/tok (DeepSeek) | HybriMoE: 0.64s, Fiddler: 1.34s | Small batch, small-expert model |
| **V100 Decode Latency (batch 64)** | 5.05s (Qwen3-30B) | Klotski: 19.9s, Fate: 9.48s | 74.6% reduction over Klotski |
| **LLaPor Top-4 Accuracy** | 94–99% across all models | — | With second-order sliding window (Top-4 within true Top-6) |
| **LLaPor Inference Overhead** | 0.12–0.48ms per forward pass | — | Fully hidden within GPU I/O wait gaps |
| **LLaPor Memory** | 0.5–2.8MB per predictor per layer | — | Lightweight; middle layers largest (2.8MB) |

Note: The paper reports throughput as tokens/second during generation (prefill + decode combined), not separated into prefill throughput and decode throughput. TTFT and TPOT are not explicitly broken out.

### 10. Techniques & Innovation

The core innovation is a **prediction-driven cross-layer scheduling framework** for CPU–GPU hybrid MoE inference. Instead of greedy per-layer optimization, PreScope jointly models the current layer's on-demand loading needs and the next layer's prefetch opportunities into a global latency-minimization problem.

**Key mechanisms:**
- **LLaPor (Learnable Layer-Aware Predictor)**: A per-layer lightweight neural predictor that exploits layer-group structure — input/output layers use shallow networks focusing on routing weights, middle layers use deeper residual networks leveraging cosine similarity of gating inputs. Uses PCA-compressed hidden states as features. Achieves >94% Top-4 accuracy.
- **PreSched (Prefetch-Aware Cross-Layer Scheduler)**: Builds a unified cost model that quantifies the trade-off between on-demand GPU loading and CPU compute-offload with prefetching. Uses a cross-layer ordering of experts from current + predicted layers, decides GPU vs. CPU placement per expert, and controls prefetch count to avoid bandwidth conflicts. Runs in O(n) time.
- **AsyncIO (Asynchronous I/O Optimizer)**: Decouples expert transfers from computation using dual-buffer GPU caches (on-demand + prefetch), splits each expert into 3 weight sub-tensors transferred via 3 independent CUDA streams to saturate PCIe bandwidth, and uses pinned host memory with async copies.

**Builds upon / combines:**
- HybriMoE's queue mechanism for hot/cold expert prioritization (retained and enhanced)
- Fate's gate-based prefetching (LLaPor improves upon it with layer-group-aware learning)
- Klotski's I/O masking via larger batches (PreScope achieves better masking without increasing batch size)
- Belady-like optimal replacement philosophy (applied to prefetch scheduling rather than cache eviction)

---

## 🎯 Relevance Assessment

| Criterion | Rating |
|-----------|--------|
| **Overall Relevance to Edge MoE Inference** | ⭐⭐⭐ (3/5) |

**Key Strengths:**
- Evaluated on **Qwen3-30B-A3B**, the exact model family we target, with detailed per-layer analysis
- **No model modification required** — works with off-the-shelf HuggingFace weights, aligning with our requirement
- The **LLaPor predictor** concept is lightweight (0.5–2.8MB per layer), fast (<0.5ms inference), and could potentially be adapted for predicting which experts to cache in our NPU's 3D DRAM
- The **PreSched cross-layer cost model** provides a formal framework for reasoning about the trade-off between on-demand loading and prefetching that could inform our offloading scheduler design
- **AsyncIO design** (decoupling transfers from compute, fine-grained chunking) is directly applicable to our NPU scenario where PCIe transfers need to overlap with NPU compute

**Key Limitations / Gaps:**
- **Cloud/server-oriented, not edge**: Uses server GPUs (V100/A100 32GB), batch sizes 4–64, and assumes all expert weights fit in CPU DRAM. This is fundamentally different from our edge scenario (batch=1, 10GB NPU memory, experts potentially on SSD)
- **No SSD offloading**: PreScope offloads to CPU DRAM only. Our "low-cost" scenario requires SSD offloading. The paper does not address the much higher latency of SSD I/O vs. PCIe DRAM transfers
- **GPU-centric compute**: Assumes NVIDIA CUDA GPU with 32GB VRAM. Our NPU has fundamentally different compute characteristics (56 TOPS INT8, 1200GB/s memory bandwidth, 10GB DRAM)
- **Batch-size dependency**: PreScope's prefetching benefits are most pronounced at batch sizes ≥ 8. At batch=1 (our scenario), the cross-layer prefetch opportunity is significantly reduced since fewer experts are activated per layer and the CPU compute gap for hiding prefetch shrinks
- **No quantization**: Evaluates BF16 models only. Our 4-bit quantized models would have different I/O and compute characteristics
- **No code available**: Cannot benchmark or reuse directly
- **Not indexed**: Paper appears to be a conference submission still under review; quality has not been peer-validated

**Actionable for Our Project?**
**Partially** — PreScope provides valuable design principles (layer-group-aware prediction, cross-layer cost modeling, async I/O pipelining) that could inspire our NPU offloading scheduler. However, the GPU+DRAM-centric architecture, batch-size dependency, and lack of SSD offloading mean we cannot directly adopt it. The LLaPor predictor concept is the most transferable component — a lightweight per-layer predictor could help decide which experts to pre-load into our NPU's 10GB 3D DRAM from CPU DRAM or SSD.

---

## 📝 Notes & Discussion Points

- **Relation to LayerScope**: Enda Yu (first author) also co-authored "LayerScope: Predictive Cross-Layer Scheduling for Efficient Multi-Batch MoE Inference on Legacy Servers" (arXiv:2509.23638). LayerScope appears to be a variant targeting multi-batch legacy servers, while PreScope targets single-GPU resource-constrained scenarios with CPU–GPU hybrid inference. The papers share similar concepts (cross-layer scheduling, prediction) but different hardware targets.
- The paper's observation about **layer-group structure** (input, middle, output groups with distinct routing characteristics) is an important insight that could generalize to any MoE offloading system — even our NPU-based one. Input/output layers have skewed expert activation (few hot experts), while middle layers have more balanced activations.
- PreScope's finding that GPU compute time for a single expert is significantly lower than PCIe transfer time (GPU is I/O-bound, not compute-bound) is also true for our NPU — the NPU's 1200GB/s internal bandwidth far exceeds the 8GB/s PCIe 4.0 ×4 link, making expert prefetch scheduling critical.
- The **hot expert table** (pre-loading frequently activated experts into GPU/NPU memory at init time) is a simple but effective technique that we should adopt — it requires only offline profiling of activation frequencies and has no runtime overhead.
- Key challenge for batch=1 adaptation: PreScope's PreSched algorithm relies on CPU compute gaps created by processing multiple experts concurrently to hide prefetch latency. At batch=1 with only 2–8 experts activated per layer, the CPU compute gap would be much smaller, reducing prefetch opportunity. A different prefetch strategy (e.g., lookahead across more layers) would be needed.
- The paper claims LLaPor can perform online continuous learning during inference to adapt to shifting input distributions — this is an interesting capability for long-context scenarios (our 256K target) where routing patterns may drift over the course of a single generation.
