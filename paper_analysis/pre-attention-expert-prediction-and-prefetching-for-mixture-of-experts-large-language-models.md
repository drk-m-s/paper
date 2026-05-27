# Pre-Attention Expert Prediction and Prefetching for Mixture-of-Experts Large Language Models

## 📋 Metadata

| Field | Value |
|-------|-------|
| **Authors** | Shien Zhu, Samuel Bohl, Robin Oester, Gustavo Alonso |
| **Organization(s)** | Systems Group, D-INFK, ETH Zurich, Switzerland |
| **Venue** | arXiv preprint; the PDF states it is under review by a journal |
| **Publication Date** | 2025-11-10 (arXiv v1) |
| **Citations** | 0 visible citations on Google Scholar at fetch time (May 2026) |
| **arXiv / DOI** | arXiv:2511.10676 / 10.48550/arXiv.2511.10676 |
| **Open-Source Code** | No official repository found by title-based GitHub search |

---

## 🔍 Executive Summary

This paper proposes a same-layer MoE expert prediction method that predicts the experts of the current layer from the **pre-attention activations** of that same layer, instead of using the previous layer’s activations. The main benefit is higher prediction accuracy, especially for the first layer, while keeping runtime overhead low enough to overlap prediction and prefetch with attention.

Relative to your project, this is one of the more relevant expert-prefetch papers because it explicitly discusses **I/O-constrained edge deployments**, includes a **Qwen3-30B-A3B** evaluation, and analyzes both **disk-to-GPU** and **memory-to-GPU** loading. The main downside is that it is **not plug-and-play**: it requires collecting a large offline dataset and training separate predictor heads for each layer of each target model.

---

## 📐 10-Dimension Analysis

### 1. MoE Inference vs. Training

| Aspect | Detail |
|--------|--------|
| **Type** | Inference |
| **Explanation** | The paper is an **inference-time expert prediction and prefetching** method. It does not change MoE training itself. The only training introduced is for the auxiliary predictor heads used at inference time. |

### 2. Edge vs. Cloud Scenario

| Aspect | Detail |
|--------|--------|
| **Target** | Both, with explicit edge and cloud deployment modes |
| **Batch Size in Experiments** | Not reported as a serving-batch benchmark; the evaluation is framed around per-token autoregressive inference and per-layer expert loading rather than multi-request throughput |
| **Hardware Constraints Mentioned** | Explicitly discusses I/O-constrained edge scenarios where only one expert can be loaded during the attention window, and cloud scenarios where extra I/O allows over-provisioning |

This paper is more edge-aware than most prefetch papers. Section 6.3 explicitly distinguishes **cloud deployments with enough bandwidth** from **edge devices with I/O bandwidth constraints**, and its top-1 prediction metric is introduced specifically for the edge case where only one expert can be loaded in parallel with attention.

### 3. Hardware Setup

| Aspect | Detail |
|--------|--------|
| **Compute Device** | GPU + CPU (GPU for MoE execution and predictor training; CPU can run predictor inference) |
| **Specific Hardware** | Training/evaluation box: NVIDIA TITAN RTX 24GB, 128GB system memory, 8 CPU cores. Timing studies: Tesla V100-SXM2-32GB, NVIDIA A100-PCIE-40GB, NVIDIA A100 80GB PCIe |
| **GPU Memory** | 24GB, 32GB, 40GB, and 80GB across the different experiments |
| **CPU Memory** | 128GB on the main training/evaluation system |

The paper is still GPU-centric, not NPU-centric, but it explicitly notes that **CPU-only inference of the predictors is sufficient** in deployment scenarios where predictor overhead should be minimized.

### 4. Offloaded Weight Storage

| Aspect | Detail |
|--------|--------|
| **Storage Location** | CPU DRAM and disk |
| **Storage Capacity** | Not fully specified as a system budget; the loading study benchmarks 6 experts totaling 99MB, with each expert about 16.5MB |
| **Assumed Bandwidth** | Not stated as a single bandwidth number; instead the paper measures 99MB transfers taking 33.5-49.8ms from disk and 4.0-9.5ms from memory depending on hardware |

This paper is useful because it does not assume only DRAM offload. Its performance analysis explicitly compares **disk-to-GPU** and **memory-to-GPU** expert loading, which makes it relevant to both your SSD-oriented and DRAM-oriented product modes.

### 5. MoE Models Evaluated

| Model | Total Params | Active Params | # Experts | Notes |
|-------|-------------|---------------|-----------|-------|
| DeepSeek-V2-Lite | 16.4B | 2.4B | 64 experts/layer | 6 routed experts plus 2 shared experts, 27 layers |
| Qwen3-30B-A3B | 30.5B | 3.3B | 128 experts/layer | Directly relevant to your target model family, 8 active experts, 48 layers |
| Phi-mini-MoE | 7.6B | 2.4B | 16 experts/layer | Smaller resource-constrained case, 2 active experts, 32 layers |

The inclusion of **Qwen3-30B-A3B** is a major positive for your use case because very few papers in this pipeline test a Qwen3 MoE directly.

### 6. Model Modification / Retraining

| Aspect | Detail |
|--------|--------|
| **Architecture Changes Required?** | Yes |
| **Retraining Required?** | Yes, for auxiliary predictors |
| **Plug-and-Play with Off-the-Shelf Weights?** | No |
| **Details** | The base MoE checkpoint is not fine-tuned, but the method adds a separate predictor for each transformer layer and trains those predictors offline on model-derived activations. In practice this means no base-model retraining, but it is still not a drop-in runtime-only optimization. |

This is the paper’s main mismatch with your preferred design constraints. It avoids modifying the original router weights, but it still requires **extra learned components** and **model-specific offline training**.

### 7. Offline Profiling Required?

| Aspect | Detail |
|--------|--------|
| **Profiling Needed?** | Yes, and it goes beyond profiling into auxiliary training |
| **What is Profiled?** | 10M training samples of pre-attention activations and ground-truth expert selections; runtime timing of attention, norms, gating, expert computation, and disk/memory loading |
| **Profiling Duration** | Not explicitly reported; predictor training uses 30 epochs |
| **Profiling Dataset** | MMLU-derived samples for predictor training |

This is much heavier than the lightweight offline statistics used by papers like Fate. The paper requires collecting training data from the target model and training layer-specific predictors, which is a meaningful operational tax if you want support for arbitrary Hugging Face MoE models.

### 8. Open-Source Code

| Aspect | Detail |
|--------|--------|
| **Available?** | No official release found |
| **GitHub URL** | N/A |
| **License** | N/A |

I did not find an official repository by title-based GitHub search, and the paper itself does not provide a public implementation link.

### 9. Performance Results

| Metric | Value | Baseline | Notes |
|--------|-------|----------|-------|
| **Speedup** | About 15 percentage points higher exact-match prediction accuracy vs FATE on DeepSeek-V2-Lite; expected expert-loading time reduced to 0.27-0.64ms/token vs FATE’s 0.85-2.01ms/token | FATE; DuoServe-MoE; HOBBIT | This is mainly a **prefetch-accuracy / I/O-latency** paper, not an end-to-end throughput benchmark |
| **Prefill Throughput** | Not reported | N/A | The paper does not report tokens/s during prefill |
| **Decode Throughput** | Not reported | N/A | There is no full end-to-end decode tok/s benchmark |
| **TTFT** | Not reported | N/A | Not measured directly |
| **TPOT** | Not reported directly; expert-loading component for 6 experts is 33.5-49.8ms from disk and 4.0-9.5ms from memory | N/A | These are loading latencies, not complete TPOT numbers |
| **GPU Memory Usage** | Not presented as a full serving-memory budget | N/A | Hardware memory capacities are given, but not an explicit runtime memory breakdown |
| **CPU Memory Usage** | 128GB system memory on the main evaluation machine | N/A | No full host-memory budget analysis beyond the hardware description |
| **End-to-End Latency** | Projected savings of 569-1352ms over 1000 generated tokens vs FATE across tested hardware | FATE | This is estimated from loading-time analysis rather than a direct end-to-end serving benchmark |

Additional notable numbers:
- Exact-match prediction accuracy: **93.03%** on DeepSeek-V2-Lite, **94.69%** on Qwen3-30B, **97.62%** on Phi-mini-MoE.
- Top-1 accuracy for edge-style one-expert overlap: **98.85%**, **99.55%**, and **98.95%** on those three models.
- Prediction latency is reported at about **0.15ms**, while self-attention takes about **0.74-1.13ms**, leaving a practical overlap window for prefetching.
- Over-provisioning raises DeepSeek hit rate from **93.03%** to **98.65%** by loading 10 experts instead of 6.

### 10. Techniques & Innovation

The paper’s core idea is to move expert prediction from **cross-layer** to **same-layer pre-attention** activations. The authors argue that the representation immediately before attention is temporally closer to the actual router decision, and that because routing is fundamentally a ranking problem, lightweight linear predictors with ranking-aware loss are sufficient.

The contribution is therefore a combination of **systems optimization** and a **small learned auxiliary model**: the prediction heads are lightweight enough to overlap with attention, but still trained specifically to improve first-layer and general-layer prefetch accuracy.

**Key mechanisms:**
- Same-layer **pre-attention expert prediction** instead of previous-layer prediction
- Lightweight per-layer predictors with **two linear layers** and ranking-aware training losses
- Separate deployment modes for **exact-match**, **over-provisioning**, and **top-1 edge loading**
- Parallel execution of predictor inference with attention to hide predictor overhead
- Disk-to-GPU and memory-to-GPU loading analysis to quantify how accuracy converts into I/O savings

**Builds upon / combines:**
- Prior MoE expert prefetch work such as FATE, DuoServe-MoE, MoE-Infinity, HOBBIT, and SP-MoE
- Ranking-aware loss design from multi-label learning intuition
- Transformer pipeline overlap between pre-attention normalization, attention, and expert loading

---

## 🎯 Relevance Assessment

| Criterion | Rating |
|-----------|--------|
| **Overall Relevance to Edge MoE Inference** | ⭐⭐⭐⭐☆ (4-5) |

**Key Strengths:**
- Explicitly addresses **edge I/O constraints** and introduces a **top-1 prediction metric** for the case where only one expert can be loaded during attention
- Evaluates **Qwen3-30B-A3B**, which is unusually close to your target model family
- Analyzes both **disk** and **memory** expert loading, which is directly relevant to your SSD and DRAM offload paths
- Predictor runtime cost is low enough to overlap with attention, so the online overhead is small once the auxiliary heads are trained

**Key Limitations / Gaps:**
- Requires **training separate predictor heads** for each target model and layer, so it is not a clean plug-and-play runtime optimization
- Does not provide **end-to-end tok/s, TTFT, or TPOT** measurements on an integrated serving system
- No official code release was found
- Evaluation is still GPU-centric and not tied to **llama.cpp** or NPU-oriented runtimes

**Actionable for Our Project?**
Partially. The **same-layer prefetch signal** and especially the **top-1 edge mode** are highly relevant, but the method’s reliance on auxiliary training makes it less attractive than training-free approaches if your priority is broad model compatibility and a lean implementation path.

---

## 📝 Notes & Discussion Points

- This paper is a useful complement to Fate: Fate is much simpler and training-free, while this work shows that **learned same-layer predictors** can significantly improve accuracy when you are willing to pay the offline training cost.
- The top-1 edge scenario is the most relevant part for your hardware. It maps well to an environment where PCIe/NPU overlap may only hide the transfer of **one expert** during the attention window.
- Because the paper measures both **disk-to-GPU** and **memory-to-GPU** loading, it is one of the few papers in the set that can inform both your **low-cost SSD path** and your **performance DRAM path**.
- The strongest caution is architectural cleanliness: if you want end users to load arbitrary Hugging Face MoE checkpoints without any model-specific preprocessing, this paper’s auxiliary-training requirement is a real product burden.