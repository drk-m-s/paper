# DALI: A Workload-Aware Offloading Framework for Efficient MoE Inference on Local PCs

## 📋 Metadata

| Field | Value |
|-------|-------|
| **Authors** | Zeyu Zhu, Gang Li, Peisong Wang, Zitao Mo, Minnan Pei, Zhuoran Song, Xiaoyao Liang, Jian Cheng |
| **Organization(s)** | Institute of Automation, Chinese Academy of Sciences; School of Future Technology, University of Chinese Academy of Sciences; Shanghai Jiao Tong University; AiRiA; Maicro.ai |
| **Venue** | arXiv preprint |
| **Publication Date** | 2026-02-03 (arXiv v1) |
| **Citations** | 0 visible citations on Google Scholar at fetch time (May 2026); Semantic Scholar API returned 429 Too Many Requests |
| **arXiv / DOI** | arXiv:2602.03495 / 10.48550/arXiv.2602.03495 |
| **Open-Source Code** | No official DALI repository found; implementation described as built on open-source KTransformers |

---

## 🔍 Executive Summary

DALI is a local-PC MoE offloading framework that explicitly targets the CPU-DRAM plus GPU hybrid execution path. Its core idea is to treat MoE inference as a workload-aware scheduling problem: at runtime it dynamically assigns activated experts to CPU or GPU via a low-overhead greedy approximation to a 0-1 optimization problem, predicts high-workload next-layer experts using residual-corrected features for prefetching, and updates the GPU expert cache using temporal workload correlations across adjacent tokens. Relative to your project, this is one of the more relevant papers for the performance-oriented CPU+accelerator configuration because it directly studies heterogeneous compute, limited PCIe bandwidth, and local-PC deployment; the main gap is that the experiments are mostly **batched local inference** rather than strict batch-size-1 interactive inference.

---

## 📐 10-Dimension Analysis

### 1. MoE Inference vs. Training

| Aspect | Detail |
|--------|--------|
| **Type** | Inference |
| **Explanation** | DALI is purely an **inference** system. It optimizes expert scheduling, prefetching, and caching for MoE deployment on local PCs. It does not introduce a training framework, balancing loss, or any routing modification during model training. |

### 2. Edge vs. Cloud Scenario

| Aspect | Detail |
|--------|--------|
| **Target** | Edge / local PC, but evaluated in batched serving regimes |
| **Batch Size in Experiments** | Mainly 8, 16, 32, 64, 128 depending on the figure; not batch size 1 |
| **Hardware Constraints Mentioned** | Local PCs with RTX 3090/4090-class GPUs, limited GPU memory, PCIe 4.0 x16 bandwidth bottleneck, and much smaller budgets than H100-class servers |

The paper is explicitly motivated by **local PCs**, not cloud clusters. However, the evaluation focus is not the strict edge case you care most about. Most figures benchmark **batched throughput** rather than single-request interactive latency, so the setting is closer to small-scale local serving than to pure batch-size-1 edge inference.

### 3. Hardware Setup

| Aspect | Detail |
|--------|--------|
| **Compute Device** | GPU + CPU |
| **Specific Hardware** | AMD EPYC 7532 CPU (64 cores), 256GB DDR4 DRAM, NVIDIA RTX 3090 GPU (24GB), PCIe 4.0 x16 |
| **GPU Memory** | 24GB |
| **CPU Memory** | 256GB DDR4 |

The motivation section also contrasts local PCs against H100-based servers, but the actual experiments are run on the single RTX 3090 local-PC platform above.

### 4. Offloaded Weight Storage

| Aspect | Detail |
|--------|--------|
| **Storage Location** | CPU DRAM |
| **Storage Capacity** | 256GB DDR4 system memory in the evaluation platform |
| **Assumed Bandwidth** | PCIe 4.0 x16 at 32GB/s; CPU DRAM path shown around 50GB/s in the framework diagrams |

Although the paper mentions DRAM / SSD / HDD as general offloading tiers in the background discussion, the actual DALI implementation is a **CPU-DRAM offloading** system. During deployment, all expert weights are stored in CPU DRAM, with a subset cached in GPU memory.

### 5. MoE Models Evaluated

| Model | Total Params | Active Params | # Experts | Notes |
|-------|-------------|---------------|-----------|-------|
| DeepSeek-V2-Lite-Chat | Not stated in the paper | Not stated in the paper | 64 routed experts per layer, 2 shared experts per layer, 27 layers, 6 activated experts | Main benchmark model |
| Qwen3-30B-A3B | 30B | 3B | 128 routed experts per layer, 48 layers, 8 activated experts | Strong relevance to your target model family |
| Mixtral-8x7B-Instruct | ~47B | ~13B active | 8 routed experts per layer, 32 layers, 2 activated experts | Classic MoE reference model |

The paper’s Table 3 focuses on architectural structure rather than complete parameter breakdowns. The strongest direct match to your project is **Qwen3-30B-A3B**.

### 6. Model Modification / Retraining

| Aspect | Detail |
|--------|--------|
| **Architecture Changes Required?** | No |
| **Retraining Required?** | No |
| **Plug-and-Play with Off-the-Shelf Weights?** | Yes, with lightweight offline calibration and warm-up profiling |
| **Details** | DALI does not change the underlying MoE model architecture and does not retrain or fine-tune the target model. It adds runtime scheduling logic plus an offline residual vector per MoE layer, which is computed from a small calibration dataset and reused at inference time. |

This is compatible with your requirement to use off-the-shelf model weights. The extra work is in runtime support and calibration, not model modification.

### 7. Offline Profiling Required?

| Aspect | Detail |
|--------|--------|
| **Profiling Needed?** | Yes |
| **What is Profiled?** | Warm-up profiling for hardware-specific timing values (`t_trans_time`, CPU execution time, GPU execution time) and offline residual-vector construction for prefetching |
| **Profiling Duration** | Not explicitly reported |
| **Profiling Dataset** | 1K sequences sampled from Wikitext for residual-vector construction; speed benchmarks use C4 |

DALI has two offline components. First, it profiles hardware timing values before execution so the greedy assignment strategy can estimate CPU / GPU costs. Second, it computes a per-layer residual vector offline from a small Wikitext calibration set. This is a reasonable amount of profiling for your project because it is lightweight and model-specific rather than requiring costly fine-tuning.

### 8. Open-Source Code

| Aspect | Detail |
|--------|--------|
| **Available?** | No official release found |
| **GitHub URL** | N/A |
| **License** | N/A |

The paper states that DALI is implemented **based on the open-source KTransformers framework**, and describes adding over 1,000 lines of C++ and 2,000 lines of Python. However, I did **not** find an official DALI repository or code release by title-based GitHub search. So the implementation idea is described, but the framework itself does not appear to be publicly released.

### 9. Performance Results

| Metric | Value | Baseline | Notes |
|--------|-------|----------|-------|
| **Speedup** | Decode: 3.97× vs llama.cpp, 2.16× vs KTransformers, 1.48× vs MoE-Lightning, 1.32× vs HybriMoE. Prefill: 7.62×, 3.80×, 2.45×, 2.00× respectively | llama.cpp, KTransformers, MoE-Lightning, HybriMoE | Headline average speedups reported by the paper |
| **Prefill Throughput** | Reported in tokens/s; exact values vary by model and batch size | Same baselines | Prefill benchmarks use prompt length 64 by default |
| **Decode Throughput** | Reported in tokens/s; representative values at batch size 32 from Table 9: DeepSeek 1.97 tok/s, Qwen 1.96 tok/s, Mixtral 2.02 tok/s under the best tested DALI settings | HybriMoE baseline in Table 9: 1.25 / 1.75 / 1.65 tok/s respectively | Prompt length and generated length both set to 64 by default |
| **TTFT** | Not explicitly reported | N/A | The paper reports prefill speed rather than TTFT |
| **TPOT** | Not explicitly reported | N/A | Metrics are given in tokens/s, not TPOT |
| **GPU Memory Usage** | DALI uses slightly less GPU memory than HybriMoE. Table 7: Mixtral 12.6GB–15.1GB, Qwen 4.79GB–7.02GB depending on batch size | HybriMoE | Sequence length 64 |
| **CPU Memory Usage** | 256GB DDR4 system memory in the evaluation machine | N/A | CPU DRAM is the main expert storage tier |
| **End-to-End Latency** | Greedy-assignment scheduling overhead is 4.50% of end-to-end inference latency on average vs 3.01% for HybriMoE; despite this overhead, DALI reports 4.42× end-to-end speedup from greedy assignment over naive placement | HybriMoE / naive assignment | End-to-end discussion appears in the appendix / overhead analysis |

Additional details worth carrying forward:
- PCIe transfer accounts for up to **78.1%** of total execution time under hybrid execution before DALI’s optimizations.
- DALI’s greedy assignment alone achieves about **4.42×** speedup over naive unscheduled execution and about **23%** improvement over HybriMoE’s static assignment in the isolated scheduling study.
- The greedy assignment reaches up to **92% of the optimal assignment’s performance** while incurring only about **5%** end-to-end latency overhead versus about **55%** for direct optimization solving.
- Residual-based prefetching improves prefetch accuracy by **6.9%** on DeepSeek and **15.7%** on Qwen on average over HybriMoE in the reported downstream comparison.

### 10. Techniques & Innovation

DALI’s core contribution is a **workload-aware hybrid offloading runtime** that treats MoE inference on a local PC as a joint scheduling, prefetching, and caching problem. Rather than statically deciding that certain experts or layers always run on the CPU or GPU, DALI estimates the workload of currently activated experts at runtime and chooses a CPU/GPU placement that balances heterogeneous execution time. It then uses residual-corrected inter-layer features to prefetch the next layer’s likely high-workload experts and updates the GPU expert cache using temporal workload correlations observed across adjacent tokens.

**Key mechanisms:**
- **Greedy Assignment Strategy**: a low-overhead approximation to a 0-1 integer program that dynamically assigns each activated expert to CPU or GPU based on workload and hardware cost
- **Residual-Based Prefetching**: adds a per-layer offline residual vector to the current gate input so it better approximates the next layer’s gate input, which improves prediction of high-workload next-layer experts
- **Workload-Aware Cache Replacement**: updates the GPU expert cache using accumulated recent workload scores rather than plain LRU or activation-score heuristics
- **Warm-up hardware profiling**: calibrates transfer and execution-time estimates so the runtime can reason about heterogeneous cost more accurately

**Builds upon / combines:**
- Expert-wise CPU/GPU hybrid execution from Fiddler / HybriMoE-style systems
- MoE expert prefetching and caching literature
- Runtime hardware-cost modeling and scheduling
- Temporal locality of expert usage rather than simple recency alone

---

## 🎯 Relevance Assessment

| Criterion | Rating |
|-----------|--------|
| **Overall Relevance to Edge MoE Inference** | ⭐⭐⭐⭐☆ (4-5) |

**Key Strengths:**
- Directly targets **local PCs with limited GPU memory and PCIe constraints**, which is much closer to your deployment setting than cloud GPU clusters
- Evaluates **Qwen3-30B-A3B**, making the paper unusually relevant to your target model family
- Strongly aligns with your **performance-oriented CPU DRAM + accelerator** product mode because it explicitly uses **CPU+GPU hybrid compute** rather than GPU-only offloading
- Does **not require model retraining or architecture changes**, only lightweight calibration and profiling
- The three-part design is technically coherent: scheduling, prefetching, and caching all optimize the same hardware bottleneck from different angles

**Key Limitations / Gaps:**
- The paper does **not** target your strictest scenario of **batch-size-1** inference; most evaluations are at **batch 8–128**, so it is not a direct answer to single-user interactive latency
- It is a **CPU-DRAM** system only and does not address your **SSD-offloading** low-cost product path
- No official DALI code release was found, which makes direct adoption slower than papers like Fiddler or YALIS
- The implementation is built on **KTransformers**, not **llama.cpp**, so integration lessons are less direct for your intended stack
- Performance is reported mainly as tokens/s rather than TTFT / TPOT product-style metrics, which makes direct comparison to your targets less clean

**Actionable for Our Project?**
Partially — and it is a strong reference for the **performance-oriented CPU DRAM + accelerator** architecture. DALI is especially useful for its runtime design patterns: dynamic expert placement, workload-aware prefetching, and cache replacement under limited PCIe bandwidth. However, because the experiments are mainly batched local serving rather than batch-size-1 inference, I would treat DALI as a **high-value systems reference**, not as the final architecture to copy directly for your NPU engine.

---

## 📝 Notes & Discussion Points

- The PDF text extraction misread the arXiv identifier; the correct preprint is **arXiv:2602.03495**.
- DALI is one of the strongest papers so far for the **performance SKU**: it is closer than cloud-serving papers because it is explicitly built for **local PCs**, but it is still less directly aligned with your batch-size-1 target than Fiddler or YALIS.
- Compared with Fiddler, DALI is more sophisticated on **runtime scheduling + cache + prefetch coordination**, but Fiddler is cleaner for strict single-request local inference and has a public implementation.
- Compared with YALIS, DALI is stronger on **dynamic expert placement** and overall systems optimization, while YALIS is conceptually leaner and closer to a low-overhead prefetching module.
- A plausible synthesis for your project is: use **Fiddler/DALI-style CPU-vs-accelerator expert placement** for the performance path, borrow **YALIS-style lightweight prediction** where it is cheaper than DALI’s fuller runtime machinery, and keep SSD-tier logic separate for the low-cost path.
- DALI’s use of lightweight offline profiling and calibration is probably acceptable under your requirements, but its batch-heavy evaluation means you would still need a dedicated batch-size-1 validation pass before treating it as a top candidate.
