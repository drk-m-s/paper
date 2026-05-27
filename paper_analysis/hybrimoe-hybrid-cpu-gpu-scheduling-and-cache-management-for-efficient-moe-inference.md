# HybriMoE: Hybrid CPU-GPU Scheduling and Cache Management for Efficient MoE Inference

## 📋 Metadata

| Field | Value |
|-------|-------|
| **Authors** | Shuzhang Zhong, Yanfan Sun, Ling Liang, Runsheng Wang, Ru Huang, Meng Li |
| **Organization(s)** | Peking University Institute for Artificial Intelligence; Peking University School of Integrated Circuits; Institute of Electronic Design Automation, Peking University; Beijing Advanced Innovation Center for Integrated Circuits; Beihang University |
| **Venue** | DAC 2025 (62nd ACM/IEEE Design Automation Conference) |
| **Publication Date** | 2025-06-22 published at DAC 2025; arXiv submitted 2025-04-08 |
| **Citations** | Google Scholar: 19 citations visible at fetch time; Crossref is-referenced-by count: 2; Semantic Scholar rate-limited during this pass |
| **arXiv / DOI** | arXiv:2504.05897 / 10.1109/DAC63849.2025.11133274 |
| **Open-Source Code** | Yes — official repository: https://github.com/PKU-SEC-Lab/HybriMoE (Apache-2.0) |

---

## 🔍 Executive Summary

HybriMoE is a **hybrid CPU-GPU MoE inference system** that improves over prior hybrid offloading frameworks by dynamically deciding which experts should run on CPU, which should stay on GPU, which should be prefetched, and which should be kept in cache. Its three main ideas are **dynamic intra-layer scheduling**, **impact-driven inter-layer prefetching**, and **score-aware caching**, all built on top of a kTransformers-based runtime with llama.cpp kernels.

For your project, HybriMoE is highly relevant to the **performance-oriented CPU DRAM + accelerator** direction because it explicitly models CPU expert compute as a first-class resource, covers both **prefill and decode**, and has real open-source code. It is much less useful for the **SSD-offload** direction, because the design assumes a CPU-memory-backed hybrid system and does not study SSD latency or bandwidth constraints.

---

## 📐 10-Dimension Analysis

### 1. MoE Inference vs. Training

| Aspect | Detail |
|--------|--------|
| **Type** | Inference |
| **Explanation** | HybriMoE is purely an **inference system**. It optimizes CPU-GPU scheduling, expert prefetching, and cache replacement during MoE serving and does not propose a training framework or a model-training method. |

### 2. Edge vs. Cloud Scenario

| Aspect | Detail |
|--------|--------|
| **Target** | Edge-like / resource-constrained single-node inference, not cloud-scale distributed serving |
| **Batch Size in Experiments** | Not explicitly stated in the paper text retrieved here; evaluation is latency-oriented with TTFT and per-token decode measurements, which suggests low-concurrency serving |
| **Hardware Constraints Mentioned** | Memory-limited deployment, on-demand PCIe transfers, CPU/GPU workload imbalance, varying cache budgets, limited CPU cores |

This paper is closer to your target scenario than cloud-serving papers because it explicitly frames the problem as **resource-constrained MoE offloading** and restricts CPU usage to **10 cores** to simulate practical deployments. However, it still assumes a discrete GPU server/workstation setup rather than an NPU card.

### 3. Hardware Setup

| Aspect | Detail |
|--------|--------|
| **Compute Device** | GPU + CPU |
| **Specific Hardware** | NVIDIA RTX A6000 GPU; Intel Xeon Gold 5220R CPU, with usage restricted to 10 cores during evaluation |
| **GPU Memory** | 48GB on RTX A6000 |
| **CPU Memory** | Not explicitly reported |

The paper also builds on kTransformers and llama.cpp kernels, which improves its engineering relevance for your stack even though the target accelerator differs.

### 4. Offloaded Weight Storage

| Aspect | Detail |
|--------|--------|
| **Storage Location** | CPU DRAM |
| **Storage Capacity** | Not explicitly reported |
| **Assumed Bandwidth** | PCIe-connected host memory; exact bandwidth is not explicitly stated in the evaluation section |

The introduction mentions that experts can in principle be offloaded to CPU memory or SSDs, but **HybriMoE itself is a CPU-memory-backed hybrid compute design**. Its optimization logic assumes that cache misses can be handled by CPU computation rather than by an SSD-backed storage tier.

### 5. MoE Models Evaluated

| Model | Total Params | Active Params | # Experts | Notes |
|-------|-------------|---------------|-----------|-------|
| Mixtral-8x7B-Instruct | 46.7B | 12.9B | 8 routed experts/layer, top-2 active | 32 layers; no shared experts |
| Qwen2-57B-A14B-Instruct | 57B | 14B | 64 routed experts + 1 shared expert; 8 routed experts active | 28 layers |
| DeepSeek-V2-Lite-Chat | 15.7B | 2.4B | 64 routed experts + 2 shared experts; 6 routed experts active | 26 layers |

This is a useful model set for your work because it spans a classic Mixtral-style model and two more recent MoE families with shared experts and larger routing spaces.

### 6. Model Modification / Retraining

| Aspect | Detail |
|--------|--------|
| **Architecture Changes Required?** | No |
| **Retraining Required?** | No |
| **Plug-and-Play with Off-the-Shelf Weights?** | Yes |
| **Details** | HybriMoE is a runtime system for existing MoE checkpoints. It relies on scheduling, prefetching, caching, and quantized kernels rather than modifying the model architecture or training auxiliary predictors. |

This aligns well with your requirement to serve off-the-shelf models from Hugging Face without retraining.

### 7. Offline Profiling Required?

| Aspect | Detail |
|--------|--------|
| **Profiling Needed?** | Yes, but lightweight hardware calibration rather than dataset-driven expert profiling |
| **What is Profiled?** | A warmup phase collects CPU processing speed, GPU processing speed, and data transfer latency to guide scheduling decisions |
| **Profiling Duration** | Not reported |
| **Profiling Dataset** | No dedicated offline dataset is required for expert activation profiling |

This is an attractive property for your project. HybriMoE needs **system calibration**, not a separate learned predictor or historical routing dataset.

### 8. Open-Source Code

| Aspect | Detail |
|--------|--------|
| **Available?** | Yes |
| **GitHub URL** | https://github.com/PKU-SEC-Lab/HybriMoE |
| **License** | Apache-2.0 |

The official repository is public and, at fetch time, showed an actively maintained codebase derived from kTransformers.

### 9. Performance Results

| Metric | Value | Baseline | Notes |
|--------|-------|----------|-------|
| **Speedup** | 1.33x average prefill speedup and 1.70x average decode improvement | Primarily versus kTransformers, the stated SOTA CPU-GPU hybrid MoE baseline | Headline result from abstract and evaluation |
| **Prefill Throughput** | Not reported as throughput | N/A | Prefill is evaluated via TTFT |
| **Decode Throughput** | 1.70x average throughput improvement over kTransformers | kTransformers | The paper also reports decode latency/TBT-style measurements |
| **TTFT** | Average 1.33x lower prefill latency than kTransformers across input lengths and cache configurations | llama.cpp, AdapMoE, kTransformers | Prefill input lengths are around 32, 128, 512, and 1024 tokens |
| **TPOT** | Not reported under that exact name; the paper uses Time Between Tokens (TBT) for decode | N/A | TBT plays the same role as per-token decode latency |
| **GPU Memory Usage** | Evaluated under expert cache ratios of 25%, 50%, and 75% rather than absolute GB totals | Same cache-ratio settings across baselines | The work focuses on cache ratios more than absolute footprint tables |
| **CPU Memory Usage** | Not explicitly reported | N/A | CPU memory is the cold expert tier and CPU compute device |
| **End-to-End Latency** | Not reported as a separate request-level latency table | N/A | Results are stage-specific: TTFT for prefill and TBT/throughput for decode |

Additional quantitative details:
- Under a Qwen2 configuration with 25% cache ratio, the ablation table reports **1.26x** speedup from scheduling alone, **1.06x** from prefetching alone, and **1.31x** for the full prefill pipeline over baseline.
- In decode under the same ablation setup, scheduling gives **1.46x**, prefetching **1.15x**, caching **1.38x**, and the full system **1.86x** over baseline.
- The proposed Minus Recent Score (MRS) caching policy improves cache hit rate over LRU by **6% to 8%** at 25% cache capacity: Mixtral **30.2% to 36.2%**, DeepSeek **47.7% to 52.7%**, and Qwen2 **45.0% to 52.8%**.

### 10. Techniques & Innovation

HybriMoE's core innovation is to treat hybrid MoE serving as a **dynamic workload-balancing problem** rather than a fixed expert placement problem. It does not just keep hot experts on GPU; it continuously decides whether an expert should run on CPU, stay on GPU, be prefetched from later layers, or be retained in cache, based on workload, routing scores, and simulated scheduling impact.

This is a **systems optimization paper**. The main novelty is the combination of simulation-guided intra-layer CPU/GPU scheduling, impact-driven inter-layer prefetching, and score-aware cache replacement specialized for unstable MoE activations.

**Key mechanisms:**
- Dynamic intra-layer hybrid scheduling with GPU-priority, CPU-priority, and transfer-priority rules
- A simulation phase that chooses task allocation to balance CPU compute, GPU compute, and PCIe transfer timelines
- Impact-driven prefetching that estimates the gain from preloading experts from subsequent layers
- Score-aware caching using the proposed Minus Recent Score (MRS) replacement policy
- Runtime integration on top of kTransformers with llama.cpp kernels and Marlin 4-bit quantization

**Builds upon / combines:**
- kTransformers-style CPU/GPU hybrid MoE inference
- llama.cpp kernel infrastructure
- Prior offloading baselines including llama.cpp, AdapMoE, and kTransformers
- MoE-specific routing-score and expert-reuse signals for cache management

---

## 🎯 Relevance Assessment

| Criterion | Rating |
|-----------|--------|
| **Overall Relevance to Edge MoE Inference** | ⭐⭐⭐☆ ☆ (3-5) |

**Key Strengths:**
- Strong fit for the **performance-oriented CPU DRAM + accelerator** path because it explicitly treats CPU expert compute as part of the serving design
- Covers both **prefill and decode**, which is important for your TTFT and token-rate goals
- Requires **no retraining or model modification** and has a public **Apache-2.0** implementation
- Closer to your intended implementation stack than many papers because it builds on **kTransformers and llama.cpp kernels**

**Key Limitations / Gaps:**
- The system is designed around **CPU DRAM offload**, not **SSD offload**, so it does not directly solve your low-cost tier
- It assumes a discrete **RTX A6000 + Xeon** setup rather than an NPU card with an 8GB/s host link
- CPU memory capacity and absolute memory-footprint tables are not emphasized, making budgeting less direct for your hardware plan
- The paper reports stage metrics well, but does not provide a richer end-to-end user-facing latency analysis beyond TTFT and decode metrics

**Actionable for Our Project?**
Partially, but strongly so for the **performance SKU**. HybriMoE is a valuable reference for dynamic CPU-assisted expert execution, score-aware caching, and prefill/decode-aware scheduling, but it is not a direct template for the **SSD-backed** product path and would need substantial adaptation for an NPU-centric runtime.

---

## 📝 Notes & Discussion Points

- HybriMoE sits conceptually between KTransformers and FineMoE: like KTransformers, it treats CPU compute as a real execution resource; like FineMoE, it focuses explicitly on latency-sensitive prefill and decode behavior.
- The paper is especially useful because it compares against **llama.cpp**, **AdapMoE**, and **kTransformers**, which makes the tradeoffs easier to interpret for a practical implementation path.
- Its lightweight warm-up requirement is a good fit for your engineering preferences: it avoids both predictor training and heavy offline workload profiling.
- The design is much more transferable to your **CPU+NPU hybrid** concept than to your **NPU+SSD-only** concept.
- Because the official repo already exists and is public, HybriMoE is a stronger implementation-study target than papers that only describe ideas at the PDF level.