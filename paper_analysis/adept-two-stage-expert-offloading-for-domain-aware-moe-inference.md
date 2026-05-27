# Two-Stage Expert Offloading for Domain-Aware MoE Inference

## 📋 Metadata

| Field | Value |
|-------|-------|
| **Authors** | Hangyeol Kim, Honguk Woo, Younghwan Kim |
| **Organization(s)** | Sungkyunkwan University; Korea Electronics Technology Institute (KETI) |
| **Venue** | IEEE Access |
| **Publication Date** | Received 2026-01-19, accepted 2026-02-03, published 2026-02-17; current version dated 2026-03-05 |
| **Citations** | Crossref is-referenced-by count: 0 at fetch time; Google Scholar surfaced the paper but the fetched snippet did not expose a visible citation count |
| **arXiv / DOI** | No public arXiv found during this pass / 10.1109/ACCESS.2026.3665697 |
| **Open-Source Code** | No public ADEPT repository found. The paper explicitly states that, due to software registration processes, the algorithms and simulator configurations are described in the manuscript instead of a full public code release. |

---

## 🔍 Executive Summary

ADEPT is a **two-stage MoE offloading framework** for **latency-sensitive, batch-size-1 inference** under limited GPU memory. It uses **domain-aware expert preloading during prefill** and a **locality-aware Random-Forest expert prefetcher during decoding**, targeting the classic tradeoff between VRAM pressure and repeated CPU-to-GPU expert transfers.

For your project, ADEPT is one of the more relevant papers in this set for the **low-latency edge direction**, because it explicitly optimizes single-request inference and separates prefill from decode. Its main weaknesses are that it relies on substantial **offline activation logging and auxiliary-model preparation**, evaluates only **CPU-memory offloading** rather than SSD offload, and backs its strongest end-to-end result with a **simulation-plus-measurement methodology** rather than a full open implementation.

---

## 📐 10-Dimension Analysis

### 1. MoE Inference vs. Training

| Aspect | Detail |
|--------|--------|
| **Type** | Inference |
| **Explanation** | ADEPT is an MoE **inference/serving** framework. It optimizes expert loading and caching during prefill and decoding, and does not propose a training system for the base MoE model. |

### 2. Edge vs. Cloud Scenario

| Aspect | Detail |
|--------|--------|
| **Target** | Edge-like, latency-sensitive single-request inference |
| **Batch Size in Experiments** | **Batch size = 1** for latency measurements; the paper explicitly states that all latency measurements report single-request inference latency typical of interactive edge applications |
| **Hardware Constraints Mentioned** | Tight GPU memory budgets, repeated CPU-to-GPU expert transfers, latency-sensitive applications, and commodity single-GPU deployment |

This is one of the clearest edge-oriented papers in the collection. Unlike throughput-serving work, ADEPT explicitly frames the workload as interactive inference and optimizes **Time-Per-Output-Token** rather than aggregate throughput.

### 3. Hardware Setup

| Aspect | Detail |
|--------|--------|
| **Compute Device** | GPU + CPU |
| **Specific Hardware** | Single **NVIDIA RTX A6000** GPU; the paper does not name the host CPU model |
| **GPU Memory** | RTX A6000 class, nominally **48GB** VRAM; end-to-end validation also imposes a strict **20GB GPU memory constraint** to force heavy offloading |
| **CPU Memory** | Not explicitly reported |

The hardware is still much stronger than your target NPU card, but the single-device setup and forced-memory-constraint methodology make it more relevant than cluster-scale serving papers.

### 4. Offloaded Weight Storage

| Aspect | Detail |
|--------|--------|
| **Storage Location** | CPU DRAM in the actual evaluated baseline and ADEPT workflow |
| **Storage Capacity** | Not explicitly reported |
| **Assumed Bandwidth** | PCIe-style CPU-to-GPU transfer path is central to the analysis, but the paper does not provide a precise bandwidth number |

The introduction mentions that MoE experts can be offloaded to **CPU memory or NVMe**, but the actual ADEPT experiments and baselines are framed around **CPU offloading**. This makes the paper much more relevant to a **CPU-DRAM offload path** than to your SSD-backed low-cost design.

### 5. MoE Models Evaluated

| Model | Total Params | Active Params | # Experts | Notes |
|-------|-------------|---------------|-----------|-------|
| DeepSeek-MoE-16B-Chat | **16.4B** | Not explicitly restated in ADEPT; the paper specifies **top-6 routing** | **64 experts**, across **27 expert layers** | Main evaluation model |
| Qwen1.5-MoE-A2.7B-Chat | **14.3B** | **2.7B** | **64 experts**, **top-4 routing** | Used in cross-model generalization analysis |

ADEPT is primarily validated on **DeepSeek-MoE-16B**. The Qwen result is useful but secondary.

### 6. Model Modification / Retraining

| Aspect | Detail |
|--------|--------|
| **Architecture Changes Required?** | No |
| **Retraining Required?** | No for the base MoE model itself |
| **Plug-and-Play with Off-the-Shelf Weights?** | Partially |
| **Details** | ADEPT does **not** change the MoE architecture or require retraining/fine-tuning of the target model. However, it does require offline construction of **domain-to-expert heatmaps** and training of an auxiliary **Random Forest predictor** from activation traces. |

So this is not model-modifying, but it is also not purely zero-preparation runtime logic.

### 7. Offline Profiling Required?

| Aspect | Detail |
|--------|--------|
| **Profiling Needed?** | Yes |
| **What is Profiled?** | Domain embeddings/medoids, per-domain expert activation frequencies, and token-level expert activation traces used to train the decoding predictor |
| **Profiling Duration** | Not reported |
| **Profiling Dataset** | Domain corpora spanning **Code, Math, Science, Creative, and Business**; the paper states it collected **over 8.3 million** expert activation logs and also describes per-domain heatmaps built from **10,000 tokens per domain** |

This is a significant practical cost. ADEPT's stage-1 domain ID avoids training a dedicated classifier, but the system still depends on substantial **offline domain analysis and activation logging**.

### 8. Open-Source Code

| Aspect | Detail |
|--------|--------|
| **Available?** | No public release found |
| **GitHub URL** | N/A for ADEPT itself |
| **License** | N/A |

The manuscript explicitly says that, because of ongoing software registration processes, the paper provides algorithmic and simulator details **instead of a full public code release**. GitHub searches by title, acronym, and author names did not find an ADEPT repository during this pass.

### 9. Performance Results

| Metric | Value | Baseline | Notes |
|--------|-------|----------|-------|
| **Speedup** | **3.45x** end-to-end speedup | Measured HF Accelerate offloading baseline under a 20GB GPU limit | This is ADEPT's headline practical result |
| **Prefill Throughput** | Not reported in tokens/s | N/A | Prefill is evaluated through domain accuracy, prefill hit rate, and memory reduction rather than throughput |
| **Decode Throughput** | Not reported in tokens/s | N/A | The paper focuses on TPOT and cache hit rate instead of tokens/s |
| **TTFT** | Not explicitly reported | N/A | No separate first-token metric is given |
| **TPOT** | **1100 ms** simulated for ADEPT vs **3797 ms** measured HF Accelerate baseline; **468 ms** measured full-load lower bound | HF Accelerate, Full Expert Loading, LRU-style baselines, and MoE-Infinity-style lookahead | ADEPT's TPOT is a simulation-backed projection derived from measured compute and I/O components |
| **GPU Memory Usage** | **33% lower peak usage** than full loading per abstract; discussion section says peak cache occupancy is about **67%** of full expert memory, with macro-average usage around **33%** of full-model size | Full Expert Loading | Memory reporting mixes peak and average views, but overall direction is clear: materially lower than full loading |
| **CPU Memory Usage** | Not explicitly reported | N/A | CPU acts as the cold expert tier |
| **End-to-End Latency** | Same as TPOT framing for the batch-size-1 decode path; ADEPT reaches **1100 ms TPOT** versus **3797 ms** for measured HF Accelerate | HF Accelerate | Figure 6 and Table 11 also position ADEPT above MoE-Infinity-style lookahead and LRU baselines |

Additional quantitative details:
- Simulation shows about **50% reduction in I/O wait time**.
- Stage-2 Top-6 prediction reaches about **83.3%** average Top-6 accuracy / effective hit rate across domains, up from about **48.8%** for a naive baseline.
- The paper reports average expert hit rate rising from **0.49 to 0.83**, nearly halving CPU-GPU transfers.
- Domain classification reaches **97.2%** macro Top-1 accuracy and about **95.9%** prefill expert hit rate.

The key caution is that the strongest ADEPT number is **not** a fully measured end-to-end ADEPT runtime on released software. It is a projection built from measured hardware components plus trace-driven hit rates.

### 10. Techniques & Innovation

ADEPT's main innovation is to split MoE offloading into **two different problems** rather than treating the whole generation process with one heuristic. During prefill, it predicts the prompt's domain and loads a domain-specific short list of experts to reduce peak memory. During decoding, it uses **temporal, spatial, and domain locality** to predict the next experts and asynchronously prefetch them into GPU memory.

This is a **systems optimization** with lightweight auxiliary learning components. The novelty is not in changing the MoE model, but in combining **domain-conditioned prefill warm-starting** with **Random-Forest-based decode prefetching** and evaluating their effect on the memory-latency tradeoff.

**Key mechanisms:**
- Stage-1 domain identification via medoid similarity using a lightweight sentence embedding model
- Offline domain-to-expert heatmaps, with short lists that cover at least 95% cumulative activation mass per layer
- Stage-2 expert prediction using temporal locality, cross-layer locality, and domain signals
- Top-6 prefetch policy with asynchronous preloading and LRU eviction
- Event-driven simulation combined with real hardware measurements for latency estimation

**Builds upon / combines:**
- MoE offloading ideas from CPU-memory and hybrid offload systems
- Expert locality and lookahead concepts similar in spirit to MoE-Infinity and proactive caching work
- Standard memory hierarchy heuristics such as LRU and LFU, but augmented with domain and cross-layer signals

---

## 🎯 Relevance Assessment

| Criterion | Rating |
|-----------|--------|
| **Overall Relevance to Edge MoE Inference** | ⭐⭐⭐⭐⭐ (5-5) |

**Key Strengths:**
- Directly targets **batch-size-1, latency-sensitive inference**, which matches your deployment objective much better than throughput-serving papers
- Separates **prefill** and **decode** into different optimization problems, which is highly relevant to your TTFT versus decode-speed split
- Works with **off-the-shelf model weights** and does not require changing the MoE model architecture
- Provides a concrete CPU-memory offloading reference for **single-device** deployment under constrained GPU memory |

**Key Limitations / Gaps:**
- Requires substantial **offline activation logging**, domain heatmap construction, and auxiliary RF predictor training |
- No public ADEPT code release is available, which weakens implementation transferability |
- Strongest end-to-end ADEPT speedup is **simulation-backed**, not a full real-system ADEPT measurement on released code |
- The evaluated cold tier is **CPU memory**, not **SSD**, so the paper is much weaker for your low-cost NPU+SSD path |

**Actionable for Our Project?**
Partially. ADEPT is a strong conceptual reference for the **latency-oriented path**, especially its prefill/decode split and prompt-conditioned warm start, but it is not a drop-in blueprint because it depends on heavy offline preparation and does not validate an SSD-backed or NPU-centric runtime.

---

## 📝 Notes & Discussion Points

- ADEPT is unusually aligned with your **interactive edge** goal compared with many MoE-serving papers, but the alignment comes with a cost: it trades implementation simplicity for better latency by introducing offline data processing and an auxiliary predictor.
- The paper is careful about the distinction between **full-loading**, **on-demand offloading**, and **lookahead/locality-aware caching**. That makes it useful for structuring your own ablation plan later.
- Its prompt-domain logic is interesting for your project because it suggests a lightweight **warm-start policy** for prefill without modifying the model. That may transfer even if you do not adopt the full RF-based decode predictor.
- The absence of SSD evaluation is the biggest gap relative to your low-cost product mode. ADEPT is more naturally a reference for the **CPU-DRAM performance SKU** than for the SSD-backed SKU.
- The paper's memory numbers are reported in multiple ways: abstract-level peak reduction, discussion-level peak cache occupancy, and macro-average resident-memory fractions. The overall conclusion is still clear, but the exact percentage depends on which phase and denominator is used.

---

## 🔄 Comparison with MOE-INFINITY

ADEPT and MOE-INFINITY target the same broad bottleneck: **batch-size-1 MoE inference on memory-limited personal/edge-class hardware where experts cannot all stay resident on the accelerator**. Both papers assume a **GPU + CPU-memory offloading path** and both try to cut the latency caused by repeated expert transfers. The difference is in where they place the intelligence and what part of the generation pipeline they emphasize.

| Dimension | ADEPT | MOE-INFINITY |
|-----------|-------|--------------|
| **Primary problem framing** | Memory-latency bottleneck for **latency-sensitive interactive inference**, with separate prefill and decode challenges | Inefficient MoE inference on **personal machines**, with emphasis on decode-time GPU idle time and poor cache hit rate |
| **Core idea** | **Two-stage expert offloading**: domain-aware prefill loading + locality-aware decode prefetching | **Sparsity-aware expert cache** driven by online expert-activation tracing |
| **Main predictive signal** | Prompt semantic domain, domain-to-expert heatmaps, and a **Random Forest** decode predictor using temporal/spatial/domain locality | **Expert Activation Matrix** traces collected online and matched against prior request patterns |
| **Prefill handling** | Explicitly optimized; prefill is treated as its own stage and uses prompt-domain warm-start loading | Not the main novelty; the paper focuses much more on cache behavior during autoregressive decoding |
| **Decode handling** | Predictive prefetch with learned auxiliary model and asynchronous cache fills | Unified prediction, prefetch, and eviction through online sparsity-aware cache management |
| **Offline preparation** | Heavy: domain corpora, expert heatmaps, and millions of activation logs for predictor training | Light: no offline profiling; the cache warms up from live inference traces |
| **Evaluation style** | Single RTX A6000, DeepSeek-MoE-16B/Qwen1.5-MoE-A2.7B; strongest ADEPT latency number is partly **simulation-backed** | Single NVIDIA A5000 24GB, broader model set; headline TPOT gains are presented as **measured runtime** against serving baselines |
| **Open-source transferability** | No public code release | Public Apache-2.0 implementation |
| **Best takeaway for this project** | Better reference for **prefill-vs-decode separation** and prompt-conditioned warm start | Better reference for a **first practical runtime cache design** with lower deployment complexity |

**Contents-wise, the papers emphasize different evidence.** ADEPT spends much more of its space on prompt-domain identification, domain-to-expert heatmaps, and the rationale for splitting prefill from decode into two separate policies. MOE-INFINITY spends more of its paper on trace analysis showing that expert activation is sparse at the **per-request** level, then turns that observation into a cache design based on online matching of activation patterns.

**Approach-wise, ADEPT is more prior-driven and policy-rich, while MOE-INFINITY is more runtime-adaptive and systems-minimal.** ADEPT injects semantic priors before decoding even begins, which is useful when TTFT and prefill memory spikes matter. MOE-INFINITY avoids offline domain preparation and instead learns from the actual live request stream, which makes it operationally simpler and easier to reproduce.

For your design space, the split is useful: **ADEPT is the stronger conceptual reference for prefill warm-starting and latency-sensitive phase separation, while MOE-INFINITY is the stronger implementation reference for decode-time cache management.** If you ever build a hybrid, the most natural combination would be **ADEPT-style prompt-conditioned preload for prefill** plus **MOE-INFINITY-style online sparse-activation tracing for decode**.