# Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading

## 📋 Metadata

| Field | Value |
|-------|-------|
| **Authors** | Hanfei Yu, Xingqi Cui, Hong Zhang, Hao Wang, Hao Wang |
| **Organization(s)** | Stevens Institute of Technology; Rice University; University of Waterloo; Rutgers University |
| **Venue** | EuroSys 2026 (21st European Conference on Computer Systems) |
| **Publication Date** | 2026-04-26 online publication; conference dates 2026-04-27 to 2026-04-30; arXiv v1 posted 2025-02-07 |
| **Citations** | Semantic Scholar: 11 citations at fetch time; Crossref is-referenced-by count: 0 |
| **arXiv / DOI** | arXiv:2502.05370 / 10.1145/3767295.3769319 |
| **Open-Source Code** | No public FineMoE implementation repository found during this pass |

---

## 🔍 Executive Summary

FineMoE is a **lossless MoE inference offloading system** that tries to tame the latency-memory trade-off by making expert offloading finer-grained than prior cache-based methods. Instead of training a separate predictor, it records iteration-level expert probability maps and combines **semantic similarity** and **expert-trajectory similarity** to guide CPU-to-GPU expert prefetching, caching, and eviction.

For your project, this is one of the more relevant papers for the **performance-oriented CPU DRAM + accelerator** path because it avoids model retraining, measures **TTFT** and **TPOT** directly, and focuses on latency-sensitive serving rather than pure throughput. Its main limitations are that it assumes a **multi-GPU server-style platform**, uses **CPU DRAM rather than SSD** as the cold tier, and is not a direct template for a single NPU card with an 8GB/s host link.

---

## 📐 10-Dimension Analysis

### 1. MoE Inference vs. Training

| Aspect | Detail |
|--------|--------|
| **Type** | Inference |
| **Explanation** | FineMoE is a serving-time MoE offloading system. It optimizes expert prefetching, caching, and eviction for deployed MoE LLMs and does not introduce a training framework for the base model. |

### 2. Edge vs. Cloud Scenario

| Aspect | Detail |
|--------|--------|
| **Target** | Both, but closer to latency-sensitive workstation/server serving than strict edge |
| **Batch Size in Experiments** | Sensitivity analysis explicitly varies inference batch size from 1 to 8; many evaluations focus on request latency rather than large-batch throughput |
| **Hardware Constraints Mentioned** | GPU memory pressure, CPU-GPU transfer overhead, prefill/decode latency, expert cache budget, expert misses |

This paper is not a cloud-scale multi-node serving paper, but it is also not a pure edge paper in your sense. It studies **latency-sensitive serving** with TTFT/TPOT and even checks batch size 1, yet the main platform is a **six-GPU server/workstation** with NVLink and large host DRAM.

### 3. Hardware Setup

| Aspect | Detail |
|--------|--------|
| **Compute Device** | GPU + CPU |
| **Specific Hardware** | Main testbed: six NVIDIA GeForce RTX 3090 GPUs with pairwise NVLinks, AMD Ryzen Threadripper PRO 3955WX CPU with 32 cores; additional high-end testbed: NVIDIA A100 |
| **GPU Memory** | 24GB per RTX 3090; 80GB HBM2e on the A100 testbed |
| **CPU Memory** | 480GB on the main testbed |

The main testbed also uses **PCIe 4.0 with 32GB/s** bandwidth between GPU and CPU memory. FineMoE supports multi-GPU inference with expert parallelism.

### 4. Offloaded Weight Storage

| Aspect | Detail |
|--------|--------|
| **Storage Location** | CPU DRAM |
| **Storage Capacity** | 480GB CPU memory on the main testbed |
| **Assumed Bandwidth** | PCIe 4.0 with 32GB/s bandwidth between GPUs and CPU memory |

FineMoE offloads experts from GPU memory to **host DRAM**. It does not evaluate SSD-backed offloading.

### 5. MoE Models Evaluated

| Model | Total Params | Active Params | # Experts | Notes |
|-------|-------------|---------------|-----------|-------|
| Mixtral-8x7B | 46.7B | 12.9B | 2 active out of 8 experts/layer | 32 layers |
| Qwen1.5-MoE | 14.3B | 2.7B | 4 active out of 60 experts/layer | 24 layers; includes shared experts |
| Phi-3.5-MoE | 42B | 6.6B | 2 active out of 16 experts/layer | 32 layers |

These are all open-source MoE LLMs and are closer to your intended deployment family than older Switch-style benchmarks alone.

### 6. Model Modification / Retraining

| Aspect | Detail |
|--------|--------|
| **Architecture Changes Required?** | No |
| **Retraining Required?** | No |
| **Plug-and-Play with Off-the-Shelf Weights?** | Yes |
| **Details** | FineMoE is explicitly a **lossless** serving approach. It does not compress, prune, or retrain the MoE model and is prototyped on top of HuggingFace Transformers using off-the-shelf models. |

This is a strong match for your requirement of serving stock HuggingFace checkpoints without model modification.

### 7. Offline Profiling Required?

| Aspect | Detail |
|--------|--------|
| **Profiling Needed?** | Yes, but lighter than predictor-training approaches |
| **What is Profiled?** | Historical semantic embeddings and iteration-level expert maps are collected into an ExpertMapStore; model-specific prefetch distance is also tuned |
| **Profiling Duration** | Not reported |
| **Profiling Dataset** | Offline experiments pre-populate the ExpertMapStore using 70% of prompts from LMSYS-Chat-1M and ShareGPT; online experiments can start with an empty store and warm up from serving traces |

This is an important nuance: FineMoE does need historical routing/context data for its best offline results, but it does **not** require expensive per-layer predictor training like ProMoE. It can also operate in an online warm-start mode.

### 8. Open-Source Code

| Aspect | Detail |
|--------|--------|
| **Available?** | No public FineMoE release found |
| **GitHub URL** | N/A |
| **License** | N/A |

The paper states that FineMoE is built on top of the public MoE-Infinity codebase, but I did not find a dedicated FineMoE repository during this pass.

### 9. Performance Results

| Metric | Value | Baseline | Notes |
|--------|-------|----------|-------|
| **Speedup** | 47% lower inference latency overall and 39% higher expert hit rate in the paper's headline summary | State-of-the-art offloading baselines | Headline claim from abstract/conclusion |
| **Prefill Throughput** | Not reported in tok/s | N/A | Prefill is evaluated with TTFT, not throughput |
| **Decode Throughput** | Not reported in tok/s | N/A | Decode is evaluated with TPOT, not throughput |
| **TTFT** | Average TTFT reduced by 74%, 67%, 56%, and 53% versus DeepSpeed-Inference, Mixtral-Offloading, ProMoE, and MoE-Infinity | Same | Exact absolute TTFT values are primarily shown in figures |
| **TPOT** | Average TPOT reduced by 46%, 38%, 27%, and 22% versus DeepSpeed-Inference, Mixtral-Offloading, ProMoE, and MoE-Infinity | Same | FineMoE also has the lowest TTFT and TPOT in most batch-size sensitivity cases |
| **GPU Memory Usage** | Evaluated under expert-cache budgets from 6GB to 96GB; FineMoE consistently gives the best TPOT under varying cache limits | Same cache-budget setting across baselines | At 6GB cache budget, TPOT improves by 36%, 25%, 16%, and 29% over DeepSpeed-Inference, Mixtral-Offloading, ProMoE, and MoE-Infinity |
| **CPU Memory Usage** | Offloaded experts live in 480GB host DRAM; ExpertMapStore stays under 200MB even at 32K capacity, and 1K maps are sufficient in evaluation | N/A | The expert-map metadata overhead is small relative to host memory |
| **End-to-End Latency** | Online-serving CDF shows consistently lower request latency than all baselines | Same | Exact percentile values are not provided in the extracted text |

Additional quantitative details:
- FineMoE improves average expert hit rate by **14%**, **37%**, and **68%** over Mixtral-Offloading, ProMoE, and MoE-Infinity, respectively.
- On the A100 80GB testbed, FineMoE still outperforms all baselines, though gains are smaller than on the 6x3090 setup.
- The latency overhead of FineMoE's non-asynchronous operations is reported as **less than 50ms**, about **1% of one iteration**.

### 10. Techniques & Innovation

FineMoE's core innovation is to replace coarse request-level expert statistics with **iteration-level expert probability maps**. It treats each inference step as a distinct prediction problem and uses both **input semantics** and **partial expert trajectories** to retrieve the most relevant historical expert map, which then guides expert prefetching and cache eviction.

This is primarily a **systems optimization** rather than a learned auxiliary model. The paper's key contribution is a heuristic, training-free control loop that combines expert-map search, dynamic prefetch thresholds, and probability-aware LFU-style caching to reduce expert misses without modifying the underlying model.

**Key mechanisms:**
- A new `ExpertMap` data structure that records per-iteration gate probability distributions across MoE layers
- Semantic-similarity search over embedding-layer outputs for early-layer prefetch guidance
- Trajectory-similarity search over prior-layer expert probabilities for later-layer guidance
- Dynamic expert-prefetch thresholds rather than fixed coarse rules
- Probability-aware LFU-based expert caching and eviction
- Asynchronous expert prefetching and on-demand loading integrated into a MoE-Infinity-style runtime

**Builds upon / combines:**
- Expert offloading and caching baselines such as MoE-Infinity and Mixtral-Offloading
- HuggingFace Transformers serving stacks
- MoE-Infinity's codebase and multi-GPU expert-parallel execution model
- Similarity-search style runtime matching instead of separate neural predictors

---

## 🎯 Relevance Assessment

| Criterion | Rating |
|-----------|--------|
| **Overall Relevance to Edge MoE Inference** | ⭐⭐⭐⭐☆ (4-5) |

**Key Strengths:**
- Strong fit for the **CPU-DRAM performance path** because it is explicitly an expert-offloading paper and reports **TTFT** and **TPOT**
- Requires **no model retraining or architecture modification**, which aligns well with your llama.cpp-style deployment requirement
- More practical than predictor-training papers because it uses a **heuristic historical-map search** instead of a separately trained expert predictor
- The control logic is directly about **latency-memory trade-off**, which is the right systems objective for your project

**Key Limitations / Gaps:**
- Assumes a **six-GPU server/workstation** with NVLink and large host memory, which is far from a single NPU M.2 card
- Offloads experts to **CPU DRAM**, not **SSD**, so it does not directly solve the low-cost storage tier you care about
- No public code was found, increasing implementation risk versus papers with released systems
- The approach depends on a historical **ExpertMapStore**, so it still needs workload traces or warm-up data to reach its best performance

**Actionable for Our Project?**
Partially, but strongly so for the **performance-oriented CPU-DRAM path**. FineMoE is one of the better references for designing a training-free, lossless expert cache controller that reasons about per-iteration routing behavior, but it is not a direct blueprint for your **NPU+SSD** design and its server-class hardware assumptions are materially heavier than your target product.

---

## 📝 Notes & Discussion Points

- FineMoE is a useful counterpoint to ExpertFlow: it gets strong gains **without** a separately trained routing predictor, which lowers integration risk.
- The most transferable ideas for your project are likely **iteration-level expert-state tracking**, **hybrid semantic/trajectory lookup**, and **probability-aware cache eviction**, not the exact multi-GPU runtime.
- Its offline-vs-online distinction matters: the system can warm up from an empty store, which is appealing if you want to avoid a hard offline profiling requirement.
- The paper is especially relevant for your **CPU-assisted high-performance SKU**, where host DRAM and a stronger CPU are available.
- It is much less directly relevant to the **SSD-offload SKU**, because SSD latency makes expert-miss penalties harsher and the paper never studies that storage tier.


## Comparison with MoE-Infinity

FineMoE and MoE-Infinity share the same high-level offloading setting: both keep a GPU-side expert cache and offload cold experts to **CPU DRAM** rather than SSD. The main difference is how they decide which experts to prefetch, retain, and evict. **MoE-Infinity** relies on a request-level **Expert Activation Matrix (EAM)** view of sparse expert reuse: it matches the current iteration trace against historical request-level activation traces in an **EAMC**, derives a predicted EAM, and uses that prediction to guide cache replacement and prefetching. **FineMoE** argues that this signal is still too coarse. Instead, it records iteration-level **ExpertMap** entries containing full gate-probability distributions, uses **semantic similarity** to guide early-layer prefetching, uses **expert-trajectory similarity** to guide later-layer prefetching, dynamically adjusts how many experts to prefetch based on similarity confidence, and decouples map search and prefetching from the critical path with an asynchronous design.

In the paper's direct evaluation, **FineMoE is the stronger offloading method**. Relative to MoE-Infinity, it reports **53% lower average TTFT**, **22% lower average TPOT**, and **68% higher expert hit rate** in the offline comparison, and it still outperforms MoE-Infinity under tighter cache budgets. The improvement comes from five concrete changes over MoE-Infinity: 1) replacing request-level counts with iteration-level probability maps, 2) adding a semantic signal for the early layers where little trajectory is available, 3) using trajectory matching for later layers instead of only coarse request traces, 4) making prefetch aggressiveness depend on confidence rather than a fixed coarse rule, and 5) moving map search and prefetching off the synchronous inference path.

The practical conclusion is that **FineMoE improves upon MoE-Infinity as an offloading controller**, but not because it changes the storage tier or the basic cache/offload architecture. It improves the **decision quality** and the **runtime scheduling** around the same CPU-DRAM expert-cache design. That said, MoE-Infinity is still the cleaner reference for strict single-consumer-GPU local deployment, while FineMoE is validated on a heavier **6x RTX 3090 + 480 GB host DRAM** setup. So if the question is which paper proposes the better expert-offloading algorithm, the answer is **FineMoE**; if the question is which one is closer to a simple personal-machine deployment target, **MoE-Infinity** remains more directly aligned.

