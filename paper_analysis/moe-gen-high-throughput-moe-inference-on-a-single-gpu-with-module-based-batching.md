# MoE-Gen: High-Throughput MoE Inference on a Single GPU with Module-Based Batching

## 📋 Metadata

| Field | Value |
|-------|-------|
| **Authors** | Tairan Xu, Leyang Xue, Zhan Lu, Adrian Jackson, Luo Mai |
| **Organization(s)** | The University of Edinburgh; EPCC, The University of Edinburgh |
| **Venue** | arXiv preprint |
| **Publication Date** | 2025-03-12 submitted to arXiv |
| **Citations** | Google Scholar: 7 citations visible at fetch time; Semantic Scholar title search returned HTTP 429 during this pass |
| **arXiv / DOI** | arXiv:2503.09716 / No DOI found |
| **Open-Source Code** | The paper claims public code at https://github.com/EfficientMoE/MoE-Gen, but the cited repository currently returns "Repository not found" |

---

## 🔍 Executive Summary

MoE-Gen is a **single-GPU, host-memory-offloaded MoE inference system** built for **high-throughput offline workloads**, not latency-sensitive interactive serving. Its main idea is **module-based batching**: instead of pushing a single model-wide batch through the whole network, it accumulates tokens in host memory and launches much larger per-module batches for attention and experts, then searches for a CPU/GPU/offload schedule that maximizes throughput.

For your project, this paper is only **partially relevant**. It is useful as evidence that batching granularity and CPU-assisted bandwidth reduction matter in MoE offloading, but it is a poor direct match for your target because it explicitly trades latency for throughput, assumes very large host DRAM and a much stronger host link, and does not target SSD-backed or batch-size-1 edge inference.

---

## 📐 10-Dimension Analysis

### 1. MoE Inference vs. Training

| Aspect | Detail |
|--------|--------|
| **Type** | Inference |
| **Explanation** | MoE-Gen is strictly an **inference system**. It optimizes batching, CPU/GPU execution, KV-cache handling, and offloading for MoE serving on a single GPU. It is not a training framework. |

### 2. Edge vs. Cloud Scenario

| Aspect | Detail |
|--------|--------|
| **Target** | Offline high-throughput single-node inference, much closer to cloud-style batch serving than edge batch-size-1 serving |
| **Batch Size in Experiments** | Large accumulated module batches; Table 1 reports average expert batch sizes of **8192** in prefill and **75** in decode for DeepSeek-V2 under a 512-token prompt plus 256-token decode setting. The paper also measures dataset-scale runs with **8.5K to 116K sequences** and includes an appendix study at batch sizes **1** and **32**. |
| **Hardware Constraints Mentioned** | Single-GPU memory limits, host-to-device bandwidth bottlenecks, GPU underutilization under small expert batches, and the need for large host DRAM to accumulate tokens and store offloaded weights/KV-cache |

This is not an edge paper in the sense you care about. It targets **offline throughput**, explicitly stating that high-throughput inference trades latency for larger batch sizes. The appendix does show some low-batch behavior, but that is not the paper's main optimization target.

### 3. Hardware Setup

| Aspect | Detail |
|--------|--------|
| **Compute Device** | GPU + CPU |
| **Specific Hardware** | C1: NVIDIA A5000 24GB + AMD 7453 28-Core + 256GB host memory; C2: NVIDIA A5000 24GB + AMD 7453 28-Core + 512GB host memory; C3: NVIDIA A6000 48GB + AMD 7313P 16-Core + 480GB host memory |
| **GPU Memory** | 24GB on A5000 or 48GB on A6000 |
| **CPU Memory** | 256GB, 512GB, or 480GB host memory depending on the testbed |

Figure 3 also explicitly analyzes the offloading setting on an **NVIDIA A5000 over PCIe 4.0 at 32GB/s**, which is materially stronger than your target host link.

### 4. Offloaded Weight Storage

| Aspect | Detail |
|--------|--------|
| **Storage Location** | CPU DRAM / host memory |
| **Storage Capacity** | 256GB to 512GB host memory in the main testbeds; one KV-cache study uses 128GB CPU KV-cache capacity |
| **Assumed Bandwidth** | PCIe 4.0 host-device link; Figure 3 cites **32GB/s** in the analyzed A5000 setting |

MoE-Gen assumes that the full model weights and KV-cache live in **host memory**, with GPU memory used as a constrained execution/cache tier. It does **not** study SSD as the cold tier.

### 5. MoE Models Evaluated

| Model | Total Params | Active Params | # Experts | Notes |
|-------|-------------|---------------|-----------|-------|
| Mixtral-8x7B | ~47B | ~13B active | 8 routed experts, top-2 active | Core evaluation model; HF model card shows 47B-scale model |
| Mixtral-8x22B | ~141B | ~39B active | 8 routed experts, top-2 active | Used for large-model single-GPU offloading study |
| DeepSeek-V2 | 236B | 21B active | 160 routed experts, top-6 active | Paper explicitly describes it as a much sparser model than Mixtral |
| DeepSeek-R1 | 671B | 37B active | Not explicitly stated in the paper text retrieved here | Evaluated as an extreme large-model case |

The appendix also includes **DeepSeek-V2-Lite** in a small-batch decoding study.

### 6. Model Modification / Retraining

| Aspect | Detail |
|--------|--------|
| **Architecture Changes Required?** | No |
| **Retraining Required?** | No |
| **Plug-and-Play with Off-the-Shelf Weights?** | Yes |
| **Details** | MoE-Gen is a runtime-system optimization. It changes batching, scheduling, CPU/GPU task placement, and memory management. It does not require model retraining or router modification. The only implementation caveat is that the CPU-assisted path depends on a custom CPU attention kernel for good performance. |

### 7. Offline Profiling Required?

| Aspect | Detail |
|--------|--------|
| **Profiling Needed?** | Yes |
| **What is Profiled?** | Hardware and kernel characteristics used by the batching scheduler: connection speed, GPU memory capacity, and the performance/memory usage of CPU and GPU kernels under different input batch sizes |
| **Profiling Duration** | Not reported |
| **Profiling Dataset** | No routing-history or expert-hotness dataset is required; this is system calibration rather than data-driven expert profiling |

This is materially different from papers that need offline routing traces or learned expert predictors. MoE-Gen profiles the **system**, not the workload distribution.

### 8. Open-Source Code

| Aspect | Detail |
|--------|--------|
| **Available?** | Claimed by the paper, but not verifiable now |
| **GitHub URL** | https://github.com/EfficientMoE/MoE-Gen |
| **License** | Not verifiable during this pass because the cited repository currently returns "Repository not found" |

The PDF explicitly says the source code is publicly available, but both web fetches and `git ls-remote` against the cited repository failed during this pass.

### 9. Performance Results

| Metric | Value | Baseline | Notes |
|--------|-------|----------|-------|
| **Speedup** | **8-31x** higher throughput over model-based batching systems; **7-13x** higher throughput for long-context generation; **9-63x** lower dataset completion time on some tasks | DeepSpeed, FlexGen, MoE-Lightning; paper also discusses continuous batching systems such as vLLM and Ollama | Biggest gains appear on sparse models and decode-heavy workloads |
| **Prefill Throughput** | On C2 with prompt length 512: **2790 tok/s** on Mixtral-8x7B, **907 tok/s** on Mixtral-8x22B, **787 tok/s** on DeepSeek-V2, **204 tok/s** on DeepSeek-R1 | DeepSpeed: 2621 / 710 / 109 / Fail; FlexGen: 2199 / 655 / 77 / Fail; MoE-Lightning: 2237 / 702 / 98 / Fail | Table 7 |
| **Decode Throughput** | On C2 with prompt length 512, MOE-Gen(H) reaches **469 / 283 tok/s** on Mixtral-8x7B, **91 / 57 tok/s** on Mixtral-8x22B, **31 / 16 tok/s** on DeepSeek-V2, and **17 / 9 tok/s** on DeepSeek-R1 for decode lengths 256 / 1024 | Llama.cpp, vLLM, DeepSpeed, FlexGen, MoE-Lightning | Table 6; the CPU-assisted variant is substantially stronger than GPU-only MoE-Gen(G) |
| **TTFT** | Not reported directly | N/A | The paper optimizes throughput, not first-token latency |
| **TPOT** | Not reported directly | N/A | Decode throughput is reported instead of per-token latency |
| **GPU Memory Usage** | Runs within **24GB or 48GB GPU memory**, using full KV-cache offloading and reduced resident weights to expand executable batch size | Same device budgets across baselines | Absolute runtime memory-footprint tables are not emphasized |
| **CPU Memory Usage** | Uses **256GB / 512GB / 480GB** host memory in the main testbeds | N/A | Host memory holds offloaded weights and KV-cache |
| **End-to-End Latency** | For Mixtral-8x22B on C2, dataset completion time drops to **18h / 8h / 82h** on MMLU / GSM8K / ChatbotArena | DeepSpeed: 23h / 115h / 1710h; MoE-Lightning: 23h / 68h / 5123h; vLLM: 112h / 303h / 5205h | These are dataset-level completion times, not single-request latency |

Additional observations:
- The paper's headline Table 1 reports **841 tok/s** prefill throughput and **31 tok/s** decode throughput for DeepSeek-V2 on a single A5000 with 512GB host memory under 512-prompt/256-decode context.
- In the appendix small-batch study, MoE-Gen still beats baselines at **batch size 1** on DeepSeek-V2-Lite and Mixtral-8x7B, but its advantage shrinks or disappears as workloads get smaller and more of the model can remain resident.

### 10. Techniques & Innovation

MoE-Gen's core technical idea is that **model-wide batching is the wrong batching granularity for MoE throughput**. Instead of sending a single fixed batch through the full network, it accumulates tokens in host memory and launches large per-module batches for attention and experts, then searches over micro-batch sizes, CPU/GPU work splits, and buffer allocations to maximize overlap and GPU utilization.

This is a **systems optimization paper** rather than a new model algorithm. The novelty comes from combining module-based batching, full KV-cache offloading, optional CPU attention execution, and a DAG-based search procedure that chooses a high-throughput offloading schedule under memory and bandwidth constraints.

**Key mechanisms:**
- Module-based batching for attention and expert modules instead of model-based batching
- Full KV-cache offloading to free GPU memory for larger executable batches
- CPU-assisted attention to reduce host-to-device KV-cache traffic in bandwidth-bound settings
- A scheduler that profiles kernels and searches over batch sizes, CPU/GPU split ratio, expert buffer size, and cached-parameter size
- DAG-based runtime estimation to choose the configuration with the shortest completion time

**Builds upon / combines:**
- Single-GPU offloading frameworks such as DeepSpeed-Inference, FlexGen, Mixtral-Offloading, and MoE-Lightning
- Continuous batching baselines such as vLLM and llama.cpp, mainly as negative comparisons for offline throughput
- CPU-assisted offload ideas related to PowerInfer, Fiddler, and MoE-Lightning

---

## 🎯 Relevance Assessment

| Criterion | Rating |
|-----------|--------|
| **Overall Relevance to Edge MoE Inference** | ⭐⭐☆☆☆ (2-5) |

**Key Strengths:**
- Clear evidence that **batching granularity** is a first-order bottleneck in MoE offloading systems
- Uses **off-the-shelf model weights** without retraining or architecture changes
- Gives concrete single-GPU results on several large open MoE models, including **DeepSeek-V2** and **DeepSeek-R1**
- Shows that **CPU-assisted computation** can improve throughput by reducing offload traffic in bandwidth-bound settings

**Key Limitations / Gaps:**
- The paper explicitly targets **offline high-throughput inference**, not low-latency interactive serving
- It assumes **very large host DRAM** and an approximately **32GB/s PCIe 4.0** host link, both much stronger than your product target
- It does not study **SSD offloading**, which limits usefulness for your low-cost NPU + SSD path
- The cited code repository is currently **not reachable**, so implementation study is less direct than the PDF suggests |

**Actionable for Our Project?**
Partially. MoE-Gen is worth mining for ideas about batching granularity, host-memory scheduling, and optional CPU-assisted execution for a performance-oriented SKU, but it is not a primary blueprint for your batch-size-1 edge engine and offers little direct guidance for the SSD-backed low-cost path.

---

## 📝 Notes & Discussion Points

- MoE-Gen is one of the clearest examples in this paper set of a system that is **technically single-node and resource-conscious**, yet still **strategically misaligned** with your target because it optimizes the wrong objective: throughput over latency.
- The appendix matters for your use case. Even though the system is not designed for batch-size-1, it still posts some small-batch gains, which suggests its runtime engineering is not useless for interactive inference. The problem is that the design center remains far from your requirements.
- The paper's use of **full KV-cache offloading** is notable. That may be appealing for freeing accelerator memory, but on your platform the much weaker 8GB/s link makes this far riskier than in MoE-Gen's A5000-based setup.
- If you borrow anything from this paper, it should be the **scheduler mindset** and the separation of module-specific batching decisions, not the full offline accumulation strategy.