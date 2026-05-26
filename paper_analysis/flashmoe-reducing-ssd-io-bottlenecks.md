# FlashMoE: Reducing SSD I/O Bottlenecks via ML-Based Cache Replacement for Mixture-of-Experts Inference on Edge Devices

## 📋 Metadata

| Field | Value |
|-------|-------|
| **Authors** | Byeongju Kim, Jungwan Lee, Donghyeon Han, Hoi-Jun Yoo, Sangyeob Kim |
| **Organization(s)** | Likely KAIST (based on Hoi-Jun Yoo affiliation; not explicitly stated in PDF) |
| **Venue** | arXiv preprint (not yet accepted to a conference) |
| **Publication Date** | 2026-01-22 |
| **Citations** | 0 (very recent, as of May 2026) |
| **arXiv / DOI** | arXiv:2601.17063 / 10.48550/arXiv.2601.17063 |
| **Open-Source Code** | GitHub: [EfficientMoE/FlashMoE](https://github.com/EfficientMoE/FlashMoE) — **repo exists but is empty** (no code released yet) |

---

## 🔍 Executive Summary

FlashMoE is a system-level framework for efficient MoE model inference on edge devices with limited DRAM (16–64GB). It offloads inactive expert weights to SSD (NVMe) instead of DRAM, then uses a lightweight ML-based cache replacement policy — a tiny per-layer feed-forward network (~113KB) — that learns to combine recency (LRU) and frequency (LFU) signals to approximate Belady's optimal eviction policy. Evaluated on real desktop hardware (RTX 5070 Ti + PCIe 5.0 SSD) with OLMoE-1B-7B and Qwen3-30B-A3B, FlashMoE achieves up to 51% higher cache hit rates than LFU and up to 2.6× speedup over existing MoE inference systems.

---

## 📐 10-Dimension Analysis

### 1. MoE Inference vs. Training

| Aspect | Detail |
|--------|--------|
| **Type** | Inference |
| **Explanation** | FlashMoE is a pure MoE **inference** framework for serving/deployment. It focuses on reducing SSD I/O during autoregressive decoding on edge devices. It does not address MoE training whatsoever. |

### 2. Edge vs. Cloud Scenario

| Aspect | Detail |
|--------|--------|
| **Target** | Edge (batch=1) |
| **Batch Size in Experiments** | Per-token decoding (effectively batch=1); prefill processes a batch of tokens together but each token independently routes to experts |
| **Hardware Constraints Mentioned** | Desktop with only 16GB DRAM, with ~1GB deliberately left available (rest locked). Target environments with 16–64GB DRAM where the full model cannot reside in memory. |

### 3. Hardware Setup

| Aspect | Detail |
|--------|--------|
| **Compute Device** | GPU + CPU (NVIDIA RTX 5070 Ti + AMD Ryzen 9600X) |
| **Specific Hardware** | AMD Ryzen 9600X (Zen5, 6 cores, PCIe 5.0), NVIDIA RTX 5070 Ti (16GB GDDR7, PCIe 5.0), Gigabyte X870 AORUS motherboard, 16GB DDR5-6000 × 4 (Dual Channel), SK hynix P51 NVMe SSD (PCIe 5.0, 7.4 GB/s read) |
| **GPU Memory** | 16GB GDDR7 |
| **CPU Memory** | 16GB DDR5 (4 × 4GB, dual channel) — deliberately limited to ~1GB available |

### 4. Offloaded Weight Storage

| Aspect | Detail |
|--------|--------|
| **Storage Location** | SSD (NVMe, PCIe 5.0) |
| **Storage Capacity** | >1TB SSD |
| **Assumed Bandwidth** | 7.4 GB/s read (SK hynix P51 PCIe 5.0). Expert loading from SSD takes ~3ms per expert. |

### 5. MoE Models Evaluated

| Model | Total Params | Active Params | # Experts | Notes |
|-------|-------------|---------------|-----------|-------|
| OLMoE-1B-7B | 7B | 1B | 64 experts (per layer) | Smaller model for detailed ablation |
| Qwen3-30B-A3B | 30.3B | 3.3B | Not explicitly stated (Qwen3 series) | Target-scale model for edge deployment |

The paper also references Mixtral 8x7B and DeepSeek models in background discussion but does not evaluate them.

### 6. Model Modification / Retraining

| Aspect | Detail |
|--------|--------|
| **Architecture Changes Required?** | No |
| **Retraining Required?** | No (the MoE model itself is not modified) |
| **Plug-and-Play with Off-the-Shelf Weights?** | Yes |
| **Details** | FlashMoE works by separating expert and non-expert weights into individual `.pt` files at the file level (per-layer, per-expert), loaded with `torch.load`. Non-expert MLP layers are bypassed by setting hidden states to zero in an overridden forward function. No model architecture changes, no fine-tuning of the base model. The ML cache model is a separate tiny FFN trained on routing traces — it does not modify the MoE model. |

### 7. Offline Profiling Required?

| Aspect | Detail |
|--------|--------|
| **Profiling Needed?** | Yes (for training the ML cache policy) |
| **What is Profiled?** | Expert routing traces: recency scores (steps since last access) and frequency scores (total routing count) for each expert, extracted during inference on a calibration dataset |
| **Profiling Duration** | ~20 minutes for routing trace extraction (A100 GPU, batch size 32, 512 samples), plus ~2 hours total for feature construction + FFN training (200 epochs with early stopping). The ML cache model itself is tiny (~113KB per layer). |
| **Profiling Dataset** | TriviaQA (512 samples, up to 512 tokens each). The question set for evaluation was a held-out split not used in training. |

### 8. Open-Source Code

| Aspect | Detail |
|--------|--------|
| **Available?** | No (repo exists but is empty) |
| **GitHub URL** | https://github.com/EfficientMoE/FlashMoE |
| **License** | None specified |

### 9. Performance Results

| Metric | Value | Baseline | Notes |
|--------|-------|----------|-------|
| **Speedup (prefill+load)** | 2.5× over llama.cpp, 4.1× over Fiddler/DAOP | llama.cpp, Fiddler, DAOP | For OLMoE-1B-7B |
| **Cache Hit Rate Improvement** | +51% over LFU, +28% over ARC, +21% over LRU | LRU, LFU, LIFO, ARC, LeCaR | OLMoE-1B-7B; varies by cache size |
| **Decode Throughput Improvement** | ~22% (OLMoE-1B-7B), ~7% (Qwen3-30B-A3B) | LRU (best heuristic baseline) | ML cache vs LRU |
| **Model Loading Speedup** | 4× faster than llama.cpp, 6.8× faster than Fiddler/DAOP | llama.cpp, Fiddler, DAOP | Only loads non-expert weights at startup |
| **Expert SSD Load Latency** | ~3ms per expert | — | SK hynix P51 PCIe 5.0 SSD |
| **FFN Computation Latency** | ~158µs per operation | — | Fully overlapped with expert loading |
| **ML Cache Computation** | ~0.1ms | — | Negligible overhead, async |
| **GPU Memory Usage** | Configurable cache per layer; non-expert layers <2GB | — | Cache controlled by user via VRAM budget |

The paper does not report explicit TTFT, TPOT, or end-to-end latency numbers for specific prompt/output lengths.

### 10. Techniques & Innovation

The core technical innovation is an **ML-based expert cache replacement policy** for SSD-offloaded MoE inference. Instead of using traditional heuristic policies (LRU, LFU), FlashMoE trains a tiny per-layer feed-forward network (~113KB) that takes normalized recency and frequency scores for each cached expert and predicts which expert is least likely to be used next (approximating Belady's optimal replacement). The ML cache runs asynchronously, overlapping its ~0.1ms computation with the ~3ms expert SSD load latency.

**Key mechanisms:**
- **Expert/non-expert file-level separation**: Model weights stored as individual `.pt` files per layer per expert, enabling on-demand SSD loading without full model initialization
- **ML-based cache policy**: Tiny per-layer FFN trained on routing traces to predict optimal eviction, combining recency (LRU) and frequency (LFU) signals
- **Async cache replacement**: Cache eviction decisions computed in parallel with expert I/O loading, hiding the ML computation overhead
- **Prefill optimization**: Required experts loaded exactly once per iteration from SSD, with outputs redistributed via indexed addition

**Builds upon / combines:**
- Belady's optimal replacement algorithm (theoretical target)
- LRU and LFU heuristic policies (used as input features)
- Fiddler and DAOP (prior MoE offloading frameworks, used as baselines)

---

## 🎯 Relevance Assessment

| Criterion | Rating |
|-----------|--------|
| **Overall Relevance to Edge MoE Inference** | ⭐⭐⭐⭐ (4/5) |

**Key Strengths:**
- Directly targets **SSD-offloaded MoE inference** on memory-constrained edge devices (16GB DRAM), which aligns with our "Low-price/Cost-effectiveness" scenario (NPU 3D DRAM + SSD offloading)
- **No model modification required** — works with off-the-shelf HuggingFace model weights (just needs file-level repackaging)
- Evaluated on **Qwen3-30B-A3B**, the exact model family we target (qwen3.6-35B-A3B)
- The ML cache policy is lightweight (~113KB per layer), trainable in ~2 hours, and provides meaningful cache hit rate improvements
- The system design (separating expert/non-expert, async cache replacement) is conceptually clean and could inform our llama.cpp-based implementation

**Key Limitations / Gaps:**
- **No open-source code available** — GitHub repo is empty; cannot directly reuse or benchmark
- **GPU-only compute** — uses NVIDIA RTX 5070 Ti; does not address NPU or hybrid CPU+NPU compute. The compute hardware assumption (GPU with CUDA) is fundamentally different from our NPU scenario
- **No quantized model evaluation** — only evaluates FP16/FP32 models. Our target is 4-bit quantized 35B models, which would change memory footprint and I/O patterns
- **Limited performance metrics** — does not report TTFT, TPOT, or end-to-end latency for specific prompt/output lengths. The reported "2.6× speedup" is aggregated and hard to map to our 25 tok/s decode target
- **Profiling dependency** — requires offline routing trace collection on a calibration dataset, which adds deployment complexity
- **SSD bandwidth assumption (7.4 GB/s PCIe 5.0)** is optimistic — our m.2 card has only PCIe 4.0 ×4 (8 GB/s) shared with SSD, and real-world SSD bandwidth is typically lower

**Actionable for Our Project?**
**Partially** — FlashMoE's SSD-offloading architecture and ML-based cache replacement are conceptually very relevant to our low-cost scenario, but the GPU-centric implementation, lack of quantization support, and missing code mean we would need to re-implement the ideas from scratch in llama.cpp for NPU inference. The paper provides useful design principles (file-level expert separation, async cache replacement, ML-based eviction) but is not directly deployable.

---

## 📝 Notes & Discussion Points

- **Two FlashMoE papers exist**: There is another FlashMoE paper (arXiv:2506.04667, "Fast Distributed MoE in a Single Kernel") targeting distributed training on multi-GPU nodes — completely different scope. This analysis covers the SSD-offloading inference paper only.
- The paper states that LRU's evicted experts are reused 34.2% of the time within the next 5 steps (vs Belady's 0.1%), highlighting significant room for improvement via better cache policies.
- The ML cache model input features (recency + frequency scores) are simple enough that they could potentially be tracked in llama.cpp's inference loop without heavy overhead.
- The async cache replacement design (ML computation overlapped with I/O) is a smart systems optimization that we should consider for our NPU pipeline — our NPU compute time could overlap with SSD expert loading.
- For our NPU scenario: the 10GB of 3D-stacked DRAM could serve as the "expert cache" (analogous to FlashMoE's VRAM cache), with experts evicted to/loaded from SSD. The key question is whether our NPU can manage the cache replacement logic or if the CPU must handle it.
- The paper's SSD bandwidth (7.4 GB/s PCIe 5.0) is ~equivalent to our max PCIe 4.0 ×4 bandwidth (8 GB/s), so the I/O assumptions are comparable.
