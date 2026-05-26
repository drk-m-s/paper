# Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models

## 📋 Metadata

| Field | Value |
|-------|-------|
| **Authors** | Keisuke Kamahori, Tian Tang, Yile Gu, Kan Zhu, Baris Kasikci |
| **Organization(s)** | University of Washington; Tsinghua University |
| **Venue** | ICLR 2025 (Poster) |
| **Publication Date** | 2025-01-23 (ICLR publication); arXiv v1 submitted 2024-02-10 |
| **Citations** | 67 on Google Scholar (May 2026); Semantic Scholar citation page was not accessible due JS / anti-bot gating |
| **arXiv / DOI** | arXiv:2402.07033 / 10.48550/arXiv.2402.07033 |
| **Open-Source Code** | https://github.com/efeslab/fiddler (Apache-2.0) |

---

## 🔍 Executive Summary

Fiddler is a local-inference MoE serving system for resource-constrained machines where the full model does not fit in GPU memory. Its key idea is to use CPU computation as well as CPU memory: when an expert is absent from GPU memory, Fiddler decides whether it is faster to transfer the expert weights to the GPU and run there, or to transfer only activations to the CPU, execute the expert on the CPU, and copy the outputs back. By combining offline hot-expert placement with a lightweight runtime latency model, Fiddler outperforms prior offloading-only and CPU-heavy baselines across single-batch inference, long prefill, and beam search.

---

## 📐 10-Dimension Analysis

### 1. MoE Inference vs. Training

| Aspect | Detail |
|--------|--------|
| **Type** | Inference |
| **Explanation** | Fiddler is purely an **inference** system. It targets local / resource-constrained execution of MoE LLMs and optimizes runtime expert execution across CPU and GPU. It does not introduce any MoE training framework, balancing loss, or training-time routing changes. |

### 2. Edge vs. Cloud Scenario

| Aspect | Detail |
|--------|--------|
| **Target** | Edge (batch=1) / local workstation |
| **Batch Size in Experiments** | Primarily single-request inference; authors explicitly state single-batch inference is the target local setting. They also evaluate long prefill and beam search, and note beam search behaves like multi-batch inference at the token level. |
| **Hardware Constraints Mentioned** | Single-GPU systems with insufficient GPU memory to hold the full model; local / resource-constrained environments where latency matters more than throughput. |

This is one of the few papers that is genuinely targeted at **local / edge-style MoE inference** rather than cloud serving.

### 3. Hardware Setup

| Aspect | Detail |
|--------|--------|
| **Compute Device** | GPU + CPU |
| **Specific Hardware** | Environment 1: NVIDIA Quadro RTX 6000 (24GB), Intel Xeon Gold 6126 (48 cores), PCIe Gen3 x16. Environment 2: NVIDIA RTX 6000 Ada (49GB), Intel Xeon Platinum 8480+ (112 cores), PCIe Gen4 x16. |
| **GPU Memory** | 24GB and 49GB |
| **CPU Memory** | Not explicitly specified |

The implementation also includes a specialized CPU kernel using **AVX512_BF16**, which is important for the CPU-side performance claims.

### 4. Offloaded Weight Storage

| Aspect | Detail |
|--------|--------|
| **Storage Location** | CPU DRAM |
| **Storage Capacity** | Enough to hold expert weights that do not fit in GPU memory; exact DRAM capacity is not specified |
| **Assumed Bandwidth** | PCIe Gen3 x16 (32GB/s) in Environment 1 and PCIe Gen4 x16 (64GB/s) in Environment 2 |

Fiddler is a **CPU-DRAM offloading** system, not an SSD offloading system. Non-expert layers are always kept in GPU memory, while a subset of expert layers are resident on GPU and the remaining experts reside in CPU memory.

### 5. MoE Models Evaluated

| Model | Total Params | Active Params | # Experts | Notes |
|-------|-------------|---------------|-----------|-------|
| Mixtral-8x7B | 47B | ~13B active | 8 experts per MoE layer, 32 layers, top-2 routing | Main model used for all core evaluations; >90GB parameters at 16-bit |
| Phi-3.5-MoE | Not specified in the paper | Not specified in the paper | Not specified in the paper | Appendix E shows additional results to demonstrate model-agnostic applicability |

The paper focuses on **uncompressed 16-bit Mixtral-8x7B** in the main experiments because it is supported by all baselines.

### 6. Model Modification / Retraining

| Aspect | Detail |
|--------|--------|
| **Architecture Changes Required?** | No |
| **Retraining Required?** | No |
| **Plug-and-Play with Off-the-Shelf Weights?** | Yes |
| **Details** | Fiddler does not change the model architecture or require retraining / fine-tuning. It works with the original 16-bit model weights and only changes runtime placement and execution strategy. The paper explicitly positions Fiddler as an alternative to approaches that require ReLU conversion or other model modifications. |

### 7. Offline Profiling Required?

| Aspect | Detail |
|--------|--------|
| **Profiling Needed?** | Yes |
| **What is Profiled?** | Expert popularity / selection frequency, plus CPU / GPU / transfer latency measurements for different input sizes |
| **Profiling Duration** | Not reported |
| **Profiling Dataset** | Calibration data sampled from ShareGPT |

Fiddler uses offline profiling in two places: first, to identify the most popular experts to place on GPU at initialization; second, to build the latency model used at runtime.

### 8. Open-Source Code

| Aspect | Detail |
|--------|--------|
| **Available?** | Yes |
| **GitHub URL** | https://github.com/efeslab/fiddler |
| **License** | Apache-2.0 |

The repository is a research prototype and the README notes that it currently focuses on 16-bit Mixtral-8x7B local inference.

### 9. Performance Results

| Metric | Value | Baseline | Notes |
|--------|-------|----------|-------|
| **Speedup** | 1.26× average in single-batch inference | Best baseline, mainly llama.cpp | Averaged across 15 input/output configurations and 2 environments |
| **Prefill Throughput** | Not separately reported | N/A | Paper reports TTFT for long prefill rather than tokens/s |
| **Decode Throughput** | End-to-end token/s used for single-batch and beam-search scenarios; no isolated decode-throughput table | llama.cpp, Mixtral-Offloading, DeepSpeed-MII | Throughput metric mixes prefill and decode for scenario (a) |
| **TTFT** | 1.30× speedup in long prefill on average vs different baselines; 1.07× vs DeepSpeed-MII and 1.65× vs Mixtral-Offloading | DeepSpeed-MII, Mixtral-Offloading, llama.cpp | Long-context prefill evaluated for input lengths 512 / 1024 / 2048 / 4096 |
| **TPOT** | Not explicitly reported as TPOT; Appendix F reports ITL instead | N/A | ITL is the closest reported metric |
| **GPU Memory Usage** | 56 / 256 experts on GPU (Env 1), 125 / 256 experts on GPU (Env 2) | N/A | Reflects how many experts fit in available GPU memory |
| **CPU Memory Usage** | Not explicitly reported | N/A | CPU DRAM holds remaining expert weights |
| **End-to-End Latency** | 11.57× speedup in beam search inference vs llama.cpp | llama.cpp | Beam widths 4 / 8 / 12 / 16, input length 32, output length 64 |

Additional details from the paper and OpenReview discussion:
- Appendix F breaks out **TTFT** and **ITL** separately.
- The authors report an average **1.13× TTFT speedup** across baselines and environments and **1.43× ITL speedup** across baselines and environments.
- The scheduling overhead is reported by the authors as **less than 3%** of total execution time.

### 10. Techniques & Innovation

Fiddler’s core innovation is a simple but effective **CPU-vs-GPU execution decision for missing experts**. Instead of always transferring a missing expert’s weights from CPU memory to GPU memory, it compares two alternatives at runtime: move weights to the GPU and compute there, or move only activations to the CPU and compute on the CPU. This leverages the asymmetry that GPU execution is faster but weight transfer is expensive, while CPU execution is slower but activation transfer is cheap for small batches.

**Key mechanisms:**
- **Hybrid execution strategy**: when an expert is absent from GPU memory, choose between GPU execution with weight transfer or CPU execution with activation transfer
- **Runtime latency model**: uses microbenchmarks to model GPU latency as roughly constant with input size while CPU latency grows roughly linearly with input size
- **Offline popular-expert placement**: uses calibration data to pre-place the most frequently selected experts in GPU memory to maximize hit rate
- **CPU kernel optimization**: custom AVX512_BF16 expert kernel to improve CPU-side expert execution

**Builds upon / combines:**
- CPU-memory offloading methods such as Mixtral-Offloading and DeepSpeed-MII
- CPU/GPU split execution ideas from llama.cpp-style heterogeneous inference
- Offline calibration / expert popularity profiling
- Cost-based runtime scheduling rather than static placement alone

---

## 🎯 Relevance Assessment

| Criterion | Rating |
|-----------|--------|
| **Overall Relevance to Edge MoE Inference** | ⭐⭐⭐⭐☆ (4-5) |

**Key Strengths:**
- Directly targets **resource-constrained local inference** rather than cloud-scale batching
- Matches your preferred constraint of **no model modification** and works with off-the-shelf weights
- Explicitly studies the **CPU-accelerator hybrid path**, which is highly relevant for your performance-oriented CPU DRAM + accelerator configuration
- Includes an **offline profiling stage** for hot experts, which aligns with your willingness to use profiling if it yields meaningful gains
- Open-source implementation is available, making it much more actionable than most recent papers

**Key Limitations / Gaps:**
- The paper only addresses **CPU DRAM offloading**, not **SSD offloading**, so it does not solve your low-cost SKU directly
- It assumes a **GPU** with CUDA semantics rather than an **edge NPU** with different compute and runtime constraints
- The implementation depends on **AVX512_BF16** for strong CPU-side performance, which may not hold on weaker commodity CPUs
- Main experiments use **16-bit** models; the paper does not evaluate **4-bit quantized** MoE models
- It focuses on single-GPU local systems, not an NPU card connected over PCIe 4.0 x4 with only 10GB on-device DRAM

**Actionable for Our Project?**
Partially — and it is highly actionable for the **performance-oriented CPU DRAM + accelerator** path. Fiddler gives a concrete design pattern for deciding when to execute missing experts on the host CPU versus the accelerator, and its use of hot-expert placement plus a lightweight latency model is directly relevant. However, it does not address the SSD-offload path, and its GPU-specific execution assumptions would need to be reworked for a llama.cpp-based NPU runtime.

---

## 📝 Notes & Discussion Points

- Fiddler is probably the best fit so far for your **performance-oriented** product configuration, where CPU DRAM is available and the host CPU can contribute compute.
- Its central decision rule is easy to reason about and likely implementable in a cleaner form in llama.cpp: if the expert is not on-device, compare the cost of **moving weights to the accelerator** against **moving activations to the CPU**.
- The paper assumes non-expert layers always fit in GPU memory. That maps reasonably well to your requirement that only **5GB–8GB** of NPU DRAM should be used for the offload engine, but it would need careful validation against your exact model partitioning.
- For your low-cost SKU, Fiddler is incomplete because it ignores SSD. A plausible path is to combine **Fiddler’s CPU/NPU execution choice** with **FlashMoE-style SSD expert staging**.
- The open-source repo is a useful implementation reference, but it is a proof-of-concept and currently limited in model and quantization support.
