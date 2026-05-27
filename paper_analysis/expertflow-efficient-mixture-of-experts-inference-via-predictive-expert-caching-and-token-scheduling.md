# ExpertFlow: Efficient Mixture-of-Experts Inference via Predictive Expert Caching and Token Scheduling

## 📋 Metadata

| Field | Value |
|-------|-------|
| **Authors** | Xin He, Shunkang Zhang, Kaijie Tang, Shaohuai Shi, Yuxin Wang, Zihao Zeng, Zhenheng Tang, Xiaowen Chu, Haiyan Yin, Ivor W. Tsang, Yew Soon Ong |
| **Organization(s)** | CFAR, A*STAR Singapore; Hong Kong University of Science and Technology; Harbin Institute of Technology, Shenzhen; Hong Kong Baptist University; Nanyang Technological University; Hong Kong University of Science and Technology (Guangzhou) |
| **Venue** | DAC 2026 (63rd ACM/IEEE Design Automation Conference) |
| **Publication Date** | 2026-07-26 to 2026-07-29 conference dates; arXiv v1 posted 2024-10-23, revised 2026-04-02 |
| **Citations** | Semantic Scholar: 1 citation at fetch time; Google Scholar did not return a stable visible count in this pass |
| **arXiv / DOI** | arXiv:2410.17954 / 10.1145/3770743.3804292 |
| **Open-Source Code** | No public implementation repository found during this pass |

---

## 🔍 Executive Summary

ExpertFlow is an MoE inference offloading system for memory-constrained single-GPU deployment. It combines a transformer-based routing-path predictor, cross-batch token rescheduling, and a predictive expert cache so that only likely-needed experts are prefetched from CPU memory to GPU memory, with runtime correction when predictions miss.

For your project, this paper is important because it directly targets **expert offloading under tight accelerator memory** and shows large memory savings. The main caveats are that it is still a **GPU+CPU DRAM** design rather than an NPU+SSD design, and its strongest gains depend on **auxiliary predictor training** plus **multi-sequence batching**, which weakens direct transfer to strict batch-size-1 edge decode.

---

## 📐 10-Dimension Analysis

### 1. MoE Inference vs. Training

| Aspect | Detail |
|--------|--------|
| **Type** | Inference |
| **Explanation** | The paper is purely an **inference system**. It optimizes expert offloading, caching, and execution scheduling for deployed MoE models and does not propose a MoE training framework. It does train an auxiliary routing predictor, but that training supports inference-time scheduling rather than base-model training. |

### 2. Edge vs. Cloud Scenario

| Aspect | Detail |
|--------|--------|
| **Target** | Edge-like resource-constrained single-GPU inference, but not strict batch-size-1 decode |
| **Batch Size in Experiments** | Greater than 1 in reported throughput experiments: examples include BS=4/8/16 for Mixtral and Qwen1.5, and up to BS=32 for Switch models |
| **Hardware Constraints Mentioned** | Single GPU with constrained memory, CPU memory used as the cold tier, dynamic expert routing, transfer overhead between CPU and GPU |

This paper is closer to edge than cloud-serving papers because it focuses on **single-GPU memory-constrained deployment** rather than large distributed serving. However, it is not a perfect match to your target because the reported gains rely on **cross-batch token scheduling**, which is less natural in low-concurrency interactive batch-size-1 decode.

### 3. Hardware Setup

| Aspect | Detail |
|--------|--------|
| **Compute Device** | GPU + CPU |
| **Specific Hardware** | Single NVIDIA A40 GPU with 48GB memory and Intel Xeon Gold 6338 CPU @ 2.00GHz |
| **GPU Memory** | 48GB |
| **CPU Memory** | Not explicitly reported |

The hardware is much closer to a workstation/server GPU setup than to your target NPU card, but the memory-pressure problem is highly similar.

### 4. Offloaded Weight Storage

| Aspect | Detail |
|--------|--------|
| **Storage Location** | CPU DRAM |
| **Storage Capacity** | Not explicitly reported |
| **Assumed Bandwidth** | Not explicitly reported; the system assumes repeated CPU-GPU expert transfers and overlaps them with compute where possible |

ExpertFlow keeps inactive experts in CPU memory and prefetches predicted experts into GPU memory. It does **not** study SSD-backed expert storage.

### 5. MoE Models Evaluated

| Model | Total Params | Active Params | # Experts | Notes |
|-------|-------------|---------------|-----------|-------|
| Switch-32 | 1.98B | 0.22B | 32 experts/layer, 1 active | 12 MoE layers; expert params are 91.58% of total |
| Switch-64 | 3.79B | 0.22B | 64 experts/layer, 1 active | 12 MoE layers; expert params are 95.61% of total |
| Switch-128 | 7.41B | 0.22B | 128 experts/layer, 1 active | 12 MoE layers; expert params are 97.75% of total |
| Mixtral-8x7B | 46.70B | 12.90B | 8 experts/layer, 2 active | 32 layers |
| Qwen1.5-MoE | 14.30B | 2.70B | 60 experts/layer, 4 active | 24 layers |
| DeepSeek-MoE | 16.40B | 2.80B | 64 experts/layer, 6 active | 27 layers |

This is a solid model set for MoE offloading research because it spans both classic Switch-style sparse models and newer LLM-relevant architectures like Mixtral, Qwen1.5-MoE, and DeepSeek-MoE.

### 6. Model Modification / Retraining

| Aspect | Detail |
|--------|--------|
| **Architecture Changes Required?** | No for the base MoE model; yes for the serving stack |
| **Retraining Required?** | Yes, for the auxiliary Routing Path Predictor |
| **Plug-and-Play with Off-the-Shelf Weights?** | Partially |
| **Details** | ExpertFlow does not require modifying or fine-tuning the underlying MoE weights. However, its full design depends on training a separate T5-style Routing Path Predictor (RPP), so it is not plug-and-play in the strongest sense. |

This is one of the most important caveats for your project. The system preserves the base model, but the deployment stack becomes more complex because you need a **separate learned predictor**.

### 7. Offline Profiling Required?

| Aspect | Detail |
|--------|--------|
| **Profiling Needed?** | Yes |
| **What is Profiled?** | Routing-path data used to train the Routing Path Predictor; the paper logs token-level expert selections and encodes each token's routing path as a binary layer-by-expert matrix |
| **Profiling Duration** | Not reported |
| **Profiling Dataset** | For each task-model pair, the paper samples 10,000 input sequences and runs each sequence three times, producing 30,000 input-output-routing-path triples per routing-path dataset |

This is more than lightweight heuristics and less than full model retraining. The offline cost is real, especially if you need model-specific or domain-specific predictors.

### 8. Open-Source Code

| Aspect | Detail |
|--------|--------|
| **Available?** | No public release found during this pass |
| **GitHub URL** | N/A |
| **License** | N/A |

I found no official repository tied to the paper title during this pass. That makes implementation study harder than papers like KTransformers.

### 9. Performance Results

| Metric | Value | Baseline | Notes |
|--------|-------|----------|-------|
| **Speedup** | Up to 10x throughput improvement overall; specific reported gains include up to 9.99x on Switch-128, 1.99x on Mixtral-8x7B, 2.12x on Qwen1.5-MoE, 1.94x on DeepSeek-MoE, and 2.21x in a cross-domain Qwen1.5 setting | Cache-MoE, SE-MoE, Pregated-MoE | Throughput results are the paper's main performance metric |
| **Prefill Throughput** | Not separately reported | N/A | Throughput is reported as end-to-end inference throughput rather than split into prefill/decode |
| **Decode Throughput** | Not separately reported | N/A | TokenScheduler targets decoding, but the paper does not present a clean standalone decode tok/s table |
| **TTFT** | Not reported | N/A | |
| **TPOT** | Not reported | N/A | |
| **GPU Memory Usage** | Up to 93.72% reduction; examples: Switch-128 from 15.26GB to 1.03GB, DeepSeek-MoE from 31.35GB to 6.38GB, Qwen1.5-MoE from 35.21GB to 6.52GB; Mixtral-8x7B OOM under all-in-GPU but runs at 15.99GB with ExpertFlow | All-In-GPU baseline | This is one of the paper's strongest results |
| **CPU Memory Usage** | Not separately quantified | N/A | CPU memory acts as the cold expert store |
| **End-to-End Latency** | Not reported as a latency table | N/A | The paper emphasizes throughput and cache hit rate rather than user-facing latency |

Additional quantitative details:
- The Routing Path Predictor reaches over **90%** accuracy in most in-domain settings and up to **95%** on some models.
- The predictor model itself is small: **7.21MB**.
- The predictive cache reaches up to **91.96%** hit ratio and outperforms LRU by **15-36 percentage points** on Switch-32 configurations.
- TokenScheduler alone improves throughput by **1.03x / 1.15x / 1.17x** on Switch-32 / 64 / 128.

### 10. Techniques & Innovation

ExpertFlow's core innovation is to treat expert offloading as a **prediction-informed scheduling problem** rather than a purely reactive cache problem. Instead of waiting for each MoE layer to expose its routing decisions, it predicts the full routing path ahead of time, uses that signal to reorganize token batches for better expert locality, and proactively loads likely-needed experts into GPU memory.

Technically, this is a **systems optimization plus learned auxiliary model**. The novelty is not a new MoE architecture, but the coordinated combination of a T5-style routing predictor, routing-aware token rebatching, predictive locality-aware caching, and runtime correction for mispredictions.

**Key mechanisms:**
- A T5-style encoder-decoder Routing Path Predictor that predicts expert usage across all MoE layers in one pass
- A TokenScheduler that rebatches tokens from adjacent batches based on routing-path similarity
- A Predictive Locality-aware Expert Caching policy that allocates cache slots across layers based on predicted demand
- A runtime correction path that swaps in missed experts and evicts mispredicted ones while overlapping I/O with compute
- A dual-batch pipeline that hides predictor and scheduling overhead behind inference execution

**Builds upon / combines:**
- Reactive expert offloading baselines such as Cache-MoE
- Predictor-based offloading ideas such as Pre-gated MoE and related expert-prediction work
- Routing-locality concepts and cache-management ideas for sparse MoE execution
- Standard MoE model families including Switch, Mixtral, Qwen-MoE, and DeepSeek-MoE

---

## 🎯 Relevance Assessment

| Criterion | Rating |
|-----------|--------|
| **Overall Relevance to Edge MoE Inference** | ⭐⭐⭐⭐☆ (4-5) |

**Key Strengths:**
- Directly tackles **memory-constrained expert offloading** rather than generic dense-model serving
- Shows very strong **GPU memory reduction** results and meaningful throughput gains on several MoE families
- Keeps the **base MoE weights unchanged**, which is important for compatibility with existing checkpoints
- The predictive cache design is conceptually relevant to your **CPU DRAM offload** path and may inspire a future SSD-aware variant

**Key Limitations / Gaps:**
- The system depends on a **separately trained routing predictor**, which adds offline cost and deployment complexity
- Experiments are **not batch-size-1**; the TokenScheduler's benefits rely on multi-sequence batching and cross-batch token grouping
- It is a **GPU+CPU DRAM** design, not an **NPU+SSD** design, so the storage and bandwidth assumptions are only a partial match
- No public code was found during this pass, which raises implementation risk compared with open-source systems papers

**Actionable for Our Project?**
Partially. ExpertFlow is a strong reference for **predictive expert caching** and for using routing forecasts to reduce cold-expert traffic, especially on the CPU-DRAM performance path. It is less directly actionable for your strict **batch-size-1** and **SSD-offload** goals because its biggest gains depend on an auxiliary predictor and token rebatching across batches.

---

## 📝 Notes & Discussion Points

- The most transferable part of ExpertFlow for your project is probably **predictive cache management**, not the full pipeline.
- The TokenScheduler looks effective in the paper, but it would be much harder to justify in a llama.cpp-style edge stack when there is only one interactive stream.
- The Routing Path Predictor is small enough to be practical, but the real cost is not model size; it is the **need to build routing-path datasets and train per-model or per-domain predictors**.
- A useful design question for your architecture is whether you can get most of the value from **prediction-informed prefetch** alone, without committing to cross-batch token reordering.
- If you later explore an SSD-based tier, ExpertFlow's cache-planning ideas may still transfer, but the latency tolerance for mispredictions will be much tighter than in the paper's CPU-DRAM setting.