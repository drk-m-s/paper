# DAOP: Data-Aware Offloading and Predictive Pre-Calculation for Efficient MoE Inference

## 📋 Metadata

| Field | Value |
|-------|-------|
| **Authors** | Yujie Zhang, Shivam Aggarwal, Tulika Mitra |
| **Organization(s)** | School of Computing, National University of Singapore |
| **Venue** | 2025 Design, Automation & Test in Europe Conference (DATE) |
| **Publication Date** | Published 2025-03-31 at DATE 2025; arXiv v1 submitted 2024-12-16, updated 2025-05-04 |
| **Citations** | Google Scholar: Cited by 19 at fetch time |
| **arXiv / DOI** | arXiv:2501.10375 / 10.23919/DATE64628.2025.10992741 |
| **Open-Source Code** | Yes. Public repo: https://github.com/ecolab-nus/DAOP (Apache-2.0). The README describes it as a proof-of-concept and says the repository was released in 2025-02. |

---

## 🔍 Executive Summary

DAOP is an **on-device MoE inference engine** for **batch-size-1, memory-constrained GPU+CPU systems**. Instead of relying only on expert caching or prefetching, it dynamically decides which experts stay on the GPU, which experts execute directly on the CPU, and which CPU-side experts should be **pre-calculated one layer ahead** to hide transfer latency.

For your project, DAOP is one of the stronger references for the **performance-oriented CPU-DRAM offload path**, because it explicitly embraces **hybrid accelerator + CPU execution**, avoids model retraining, and focuses on low-resource single-request inference. Its main gaps are that it does **not** evaluate SSD-based offloading, assumes a much stronger CPU-side platform and interconnect than your NPU card, and still delivers only modest absolute decode rates on A6000-class hardware.

---

## 📐 10-Dimension Analysis

### 1. MoE Inference vs. Training

| Aspect | Detail |
|--------|--------|
| **Type** | Inference |
| **Explanation** | DAOP is an MoE **inference/serving** system. It optimizes expert placement, offloading, and hybrid CPU/GPU execution during prefill and decode, and does not propose a training framework for MoE models. |

### 2. Edge vs. Cloud Scenario

| Aspect | Detail |
|--------|--------|
| **Target** | Edge-like / on-device single-request inference |
| **Batch Size in Experiments** | **Batch size = 1**, explicitly used to simulate real-time inference |
| **Hardware Constraints Mentioned** | Memory-constrained GPU devices, low-resource/on-device deployment, CPU-GPU transfer bottlenecks, and limited ability to keep all experts resident on the accelerator |

This is not a cloud throughput paper. DAOP is explicitly framed around **on-device** inference and resource-constrained hardware, which is materially closer to your target than multi-tenant serving systems.

### 3. Hardware Setup

| Aspect | Detail |
|--------|--------|
| **Compute Device** | GPU + CPU |
| **Specific Hardware** | Main evaluation platform: **Intel i9-10980XE** CPU + **NVIDIA A6000** GPU. Microbenchmark table also reports an **Intel Xeon Gold 6326 @ 2.9 GHz** with **NVIDIA A100** for operation-cost comparison. |
| **GPU Memory** | **48 GB HBM** on the A6000, with **768 GB/s** memory bandwidth |
| **CPU Memory** | **130 GB** host memory on the evaluation platform |

The paper is clearly optimized for a relatively strong workstation-style edge box rather than a tiny embedded platform, but the single-device setup is still relevant.

### 4. Offloaded Weight Storage

| Aspect | Detail |
|--------|--------|
| **Storage Location** | **CPU DRAM** for the active offload tier |
| **Storage Capacity** | **130 GB** host memory on the reported platform |
| **Assumed Bandwidth** | The paper states **PCIe 4.0** with **64 GB/s** data-transfer capability on the evaluation platform |

Although one figure shows a system with **3 TB SSD** capacity, DAOP's actual inference path is centered on **CPU-memory offloading and CPU execution**, not an SSD-resident expert tier. For your purposes, this is a **CPU-DRAM hybrid-compute** reference, not an SSD-offload reference.

### 5. MoE Models Evaluated

| Model | Total Params | Active Params | # Experts | Notes |
|-------|-------------|---------------|-----------|-------|
| Mixtral 8x7B | **46.6B** | Not explicitly reported in DAOP; commonly described as about **12.9B active** | **8 experts**, **top-2**, **32 blocks** | Main evaluation model for most accuracy and speed plots |
| Phi-3.5-MoE | **41.7B** in the paper, roughly **42B** in the Microsoft model card | **6.6B active** | **16 experts**, **top-2**, **32 blocks** | Secondary evaluation model |

DAOP evaluates two decoder-only top-2 MoE families and does not include DeepSeek or Qwen MoE models.

### 6. Model Modification / Retraining

| Aspect | Detail |
|--------|--------|
| **Architecture Changes Required?** | No |
| **Retraining Required?** | No |
| **Plug-and-Play with Off-the-Shelf Weights?** | Yes |
| **Details** | The paper explicitly says DAOP avoids model modification and fine-tuning. Its changes are entirely at the runtime level: expert placement, one-layer-ahead prediction, CPU pre-calculation, and graceful degradation policies. |

This matches your preference well. DAOP is a **systems/runtime method**, not a model-altering method.

### 7. Offline Profiling Required?

| Aspect | Detail |
|--------|--------|
| **Profiling Needed?** | Yes, but relatively light |
| **What is Profiled?** | A **calibration dataset** is used to initialize dominant experts for the initial GPU cache setup; the runtime then uses per-sequence prefill activation patterns and one-layer-ahead gating predictions online |
| **Profiling Duration** | Not reported |
| **Profiling Dataset** | **ShareGPT** is used for initial expert-cache calibration; the paper also analyzes prediction/similarity behavior on datasets such as **Alpaca, C4, MATH, and GSM8K** |

This is importantly different from heavy offline expert profiling pipelines. DAOP needs **initial calibration**, but its main sequence-specific allocation logic runs online at inference time.

### 8. Open-Source Code

| Aspect | Detail |
|--------|--------|
| **Available?** | Yes |
| **GitHub URL** | https://github.com/ecolab-nus/DAOP |
| **License** | Apache-2.0 |

The repository is public and currently reachable. The README describes it as a **proof-of-concept** and still under construction, which is worth keeping in mind when judging implementation maturity.

### 9. Performance Results

| Metric | Value | Baseline | Notes |
|--------|-------|----------|-------|
| **Speedup** | Up to **8.20x** over expert caching/prefetching methods and **1.35x** over offloading methods | MoE-OnDemand, Mixtral-Offloading, DeepSpeed-MII, Fiddler | Stated in abstract and conclusion |
| **Prefill Throughput** | Not separately reported | N/A | Prefill is mainly used for sequence-specific expert allocation rather than standalone throughput reporting |
| **Decode Throughput** | **4.52 tok/s** for Mixtral 8x7B and **8.21 tok/s** for Phi-3.5-MoE at input/output **[256, 512]** with full GPU-memory utilization; at **25% ECR**, DAOP still reaches **3.23 tok/s** and **5.03 tok/s** respectively | Fiddler and other baselines | The paper reports end-to-end generation rate in tokens/s rather than separate kernel timings |
| **TTFT** | Not explicitly reported | N/A | No dedicated first-token latency metric is given |
| **TPOT** | Not explicitly reported; the reported decode rates imply roughly **221 ms/token** for Mixtral 8x7B and **122 ms/token** for Phi-3.5-MoE in the [256, 512] setting | Fiddler and other baselines | These TPOT values are derived from reported tokens/s, not stated directly in the paper |
| **GPU Memory Usage** | Uses the full **48 GB** A6000 memory budget; a full-memory setting corresponds to about **46.9% Expert Cache Ratio (ECR)** for DAOP/Fiddler/MoE-OnDemand | N/A | Accuracy and speed are also reported at **62.5%, 50.0%, 37.5%, and 25.0% ECR** |
| **CPU Memory Usage** | Host system includes **130 GB** DRAM, but runtime CPU-memory consumption is not broken out separately | N/A | CPU DRAM is the cold expert tier and also supports CPU execution |
| **End-to-End Latency** | Not reported directly in seconds; evaluated through generated **tokens/s** over multiple input/output length pairs | MoE-OnDemand, DeepSpeed-MII, Mixtral-Offloading, Fiddler | For Mixtral 8x7B, several baseline methods remain below **1 tok/s**, while DAOP achieves materially higher decode rates |

Additional quantitative details:
- DAOP outperforms **Fiddler by 40.4%** at the **[256, 512]** configuration.
- Across cache sizes, DAOP reports an average **35.4%** speed improvement over Fiddler.
- In energy efficiency, DAOP reaches **14.37 tok/kJ** vs **10.06 tok/kJ** for Fiddler on Mixtral 8x7B, and **27.07 tok/kJ** vs **17.15 tok/kJ** on Phi-3.5-MoE, for about **1.50x** average improvement.
- Prefill-to-decode expert-activation similarity averages about **90.72%** on Mixtral 8x7B across **C4, MATH, and GSM8K**.
- One-layer-ahead expert prediction accuracy averages **84.11%** across **Alpaca, MATH, and C4**.

### 10. Techniques & Innovation

DAOP's core idea is that when expert migration is too expensive to hide, the system should not insist on moving every missing expert to the accelerator. Instead, it uses **sequence-specific expert allocation** to keep more useful experts on the GPU, executes some experts **directly on the CPU**, and **pre-calculates** likely next-layer CPU experts one layer ahead so CPU and GPU work can overlap.

This is a combined **systems optimization plus lightweight prediction policy**. The novelty is not just expert caching; it is the integration of **online allocation**, **hybrid CPU/GPU expert execution**, and a **graceful degradation** mechanism that substitutes a next-best GPU expert when the predicted CPU-side execution path would be too costly.

**Key mechanisms:**
- Initial dominant-expert cache seeding from a calibration dataset
- Sequence-specific expert allocation using the prefill-stage activation pattern
- One-layer-ahead expert prediction from the next layer's gate applied to current hidden states
- CPU-side selective expert pre-calculation during decode
- Graceful degradation by substituting next-best GPU experts when two predicted experts would otherwise remain on CPU

**Builds upon / combines:**
- Expert caching and prefetching ideas from Mixtral-Offloading, MoE-OnDemand, MoE-Infinity, and related proactive-loading work
- CPU execution ideas similar in spirit to Fiddler
- Simple gating-based prediction rather than auxiliary model retraining

---

## 🎯 Relevance Assessment

| Criterion | Rating |
|-----------|--------|
| **Overall Relevance to Edge MoE Inference** | ⭐⭐⭐⭐☆ (4/5) |

**Key Strengths:**
- Directly targets **on-device, batch-size-1 inference** under constrained accelerator memory
- Explicitly supports **hybrid CPU + accelerator execution**, which aligns well with your performance-oriented CPU+NPU mode
- Requires **no model modification or fine-tuning**, so it fits your off-the-shelf model requirement
- Uses only **light calibration** plus online sequence-specific decisions, making it much more practical than heavy offline profiling schemes
- Has a **public Apache-2.0 implementation**, which makes transfer into llama.cpp-style logic more realistic

**Key Limitations / Gaps:**
- The paper is fundamentally about **CPU-DRAM offloading**, not **SSD-based** offloading, so it is weak for your low-cost NPU+SSD SKU
- It assumes a much stronger host platform and much higher host-link bandwidth than your **PCIe 4.0 x4 8 GB/s** NPU card
- Absolute decode throughput is still modest even on an **A6000**, so the paper is more useful architecturally than as direct evidence that your final performance target will be met
- Evaluation covers only **Mixtral 8x7B** and **Phi-3.5-MoE**, not the newer DeepSeek/Qwen-style MoE models you care about most

**Actionable for Our Project?**
Yes, especially for the **performance-oriented NPU+CPU-DRAM path**. DAOP is one of the most relevant papers here because it provides a clean, implementable reference for when **CPU compute is preferable to expert migration**, but it does not answer the SSD-backed product mode by itself.

---

## 📝 Notes & Discussion Points

- DAOP's most important argument is architectural: **if moving expert weights is slower than just executing the missing expert on the CPU, then hybrid compute can beat pure accelerator-centric caching**. That logic becomes even more relevant for your platform because your host-link bandwidth is much lower than the paper's stated setup.
- The paper separates **initial calibration** from **runtime sequence-specific adaptation**. That is a good fit for your requirement that offline preparation should be limited unless it clearly pays off.
- Compared with MoE-Infinity, DAOP is less purely cache-centric and more explicitly about **compute placement**. It is closer in spirit to a refined, prediction-aware version of Fiddler than to a standalone expert-cache paper.
- The graceful-degradation idea is practically useful: when the ideal expert placement is impossible, DAOP chooses a nearby cheaper alternative instead of forcing expensive data movement. That kind of fallback policy could transfer well to an NPU runtime.
- The biggest missing piece for your broader product plan is still **SSD-tier orchestration**. DAOP is strong for the **CPU-DRAM performance SKU**, but not sufficient for the **NPU-DRAM + SSD** SKU.