# KTransformers: Unleashing the Full Potential of CPU/GPU Hybrid Inference for MoE Models

## 📋 Metadata

| Field | Value |
|-------|-------|
| **Authors** | Hongtao Chen, Weiyu Xie, Boxin Zhang, Jingqi Tang, Jiahao Wang, Jianwei Dong, Shaoyuan Chen, Ziwei Yuan, Chen Lin, Chengyu Qiu, Yuening Zhu, Qingliang Ou, Jiaqi Liao, Xianglin Chen, Zhiyuan Ai, Yongwei Wu, Mingxing Zhang |
| **Organization(s)** | Tsinghua University; Approaching.AI; Hangzhou Dianzi University; University of Electronic Science and Technology of China; Beijing University of Posts and Telecommunications; Beijing Institute of Technology |
| **Venue** | SOSP 2025 (Proceedings of the ACM SIGOPS 31st Symposium on Operating Systems Principles) |
| **Publication Date** | 2025-10-12 (online publication); conference dates 2025-10-13 to 2025-10-16 |
| **Citations** | 12 visible citations on Google Scholar at fetch time (May 2026); Crossref is-referenced-by count 3 |
| **arXiv / DOI** | No public arXiv found during this pass / 10.1145/3731569.3764843 |
| **Open-Source Code** | Yes — official repository: https://github.com/kvcache-ai/ktransformers (Apache-2.0) |

---

## 🔍 Executive Summary

KTransformers is a systems paper on **local, low-concurrency CPU/GPU hybrid MoE inference**. It treats routed experts as CPU-resident compute rather than merely CPU-stored weights, then improves end-to-end performance with AMX/AVX-512 specialized kernels, NUMA-aware placement, asynchronous CPU-GPU scheduling, and an optional **Expert Deferral** mechanism that overlaps deferred expert execution with later attention.

Relative to your project, this is one of the strongest references for the **performance-oriented CPU DRAM + accelerator** path because it is explicitly evaluated at **batch size 1**, targets local deployment, compares against both **Fiddler** and **llama.cpp**, and provides real open-source code. Its main drawbacks are that it assumes a **very strong CPU + huge DRAM budget**, focuses on **CPU DRAM rather than SSD offload**, and its most aggressive speedup technique slightly changes model execution semantics.

---

## 📐 10-Dimension Analysis

### 1. MoE Inference vs. Training

| Aspect | Detail |
|--------|--------|
| **Type** | Inference |
| **Explanation** | KTransformers is purely an **inference system**. It optimizes local heterogeneous execution of MoE models and does not introduce a training framework, routing-loss modification, or retraining procedure. |

### 2. Edge vs. Cloud Scenario

| Aspect | Detail |
|--------|--------|
| **Target** | Local / edge-style low-concurrency inference, not cloud-scale high-concurrency serving |
| **Batch Size in Experiments** | 1 |
| **Hardware Constraints Mentioned** | Single or few local requests, constrained VRAM, CPU DRAM as the larger memory tier, PCIe 4.0 bandwidth limits, strong dependence on CPU compute and NUMA effects |

This paper is explicit about the distinction between cloud-scale deployments with massive concurrency and **local deployments with low concurrency**. Its workload section uses **batch size 1** as the representative local setting, which aligns well with your target use case.

### 3. Hardware Setup

| Aspect | Detail |
|--------|--------|
| **Compute Device** | GPU + CPU |
| **Specific Hardware** | Dual-socket Intel Xeon Platinum 8452Y machine; each socket has 36 physical cores. GPUs are NVIDIA A100 40GB and NVIDIA RTX 4080 16GB. |
| **GPU Memory** | 40GB on A100; 16GB on RTX 4080 |
| **CPU Memory** | 1TB DDR5 per socket on the evaluation machine |

Additional hardware details from the paper:
- Intra-socket memory bandwidth: about **220GB/s**
- Cross-socket memory bandwidth: about **125GB/s**
- CPU-GPU interconnect: **PCIe 4.0**, theoretical peak **32GB/s**

This setup is much heavier on host CPU and DRAM than your final M.2 NPU product, but it is directly relevant to your **high-performance SKU** concept.

### 4. Offloaded Weight Storage

| Aspect | Detail |
|--------|--------|
| **Storage Location** | CPU DRAM |
| **Storage Capacity** | Up to multi-terabyte host DRAM in the evaluated system; for DS-3 the paper places 654B parameters on CPU memory |
| **Assumed Bandwidth** | PCIe 4.0 with theoretical 32GB/s CPU-GPU bandwidth; host DRAM bandwidth dominates the CPU-resident expert path |

The paper is about **CPU-DRAM-backed computation offloading**, not SSD offloading. Instead of streaming weights to the GPU layer by layer, KTransformers keeps routed experts in CPU memory and executes them directly on the CPU.

### 5. MoE Models Evaluated

| Model | Total Params | Active Params | # Experts | Notes |
|-------|-------------|---------------|-----------|-------|
| DeepSeek-V3-0324 (DS-3) | 671B | Not explicitly stated; GPU-resident subset is 17B | 256 routed experts/layer | 58 MoE layers, top-8 routing |
| DeepSeek-V2.5-1210 (DS-2) | 236B | Not explicitly stated; GPU-resident subset is 13B | 160 routed experts/layer | 59 MoE layers, top-6 routing |
| Qwen2-57B-A14B (QW-2) | 57B | About 14B active by model name; GPU-resident subset is 8B | 64 routed experts/layer | 28 MoE layers, top-8 routing |

The paper’s Table 1 provides the total parameter count, GPU-resident parameter count, CPU-resident parameter count, number of MoE layers, and routed experts per layer. It does **not** evaluate Qwen3-family models in the paper itself.

### 6. Model Modification / Retraining

| Aspect | Detail |
|--------|--------|
| **Architecture Changes Required?** | No for the core system; optional runtime behavior change for Expert Deferral |
| **Retraining Required?** | No |
| **Plug-and-Play with Off-the-Shelf Weights?** | Yes |
| **Details** | KTransformers works by replacing HuggingFace modules with optimized kernels and placement policies via a YAML-driven injection framework. No model retraining is required. The optional Expert Deferral optimization changes execution order during decode, but it does not require weight changes or fine-tuning. |

This is a strong match to your requirement of using off-the-shelf models. The biggest caveat is that the most aggressive optimization, Expert Deferral, is an **approximate runtime change** rather than a mathematically identical execution path.

### 7. Offline Profiling Required?

| Aspect | Detail |
|--------|--------|
| **Profiling Needed?** | Yes, but lightweight and systems-oriented |
| **What is Profiled?** | Expert popularity or shared-expert structure for placement; decode traces for CPU/GPU overlap; hardware throughput, NUMA effects, and the number of deferred experts needed to saturate CPU utilization |
| **Profiling Duration** | Not explicitly reported |
| **Profiling Dataset** | No dedicated training dataset is required; profiling is based on system traces and model behavior |

This is much lighter than predictor-training papers. KTransformers does require **hardware-aware profiling and tuning**, but not an offline ML training phase.

### 8. Open-Source Code

| Aspect | Detail |
|--------|--------|
| **Available?** | Yes |
| **GitHub URL** | https://github.com/kvcache-ai/ktransformers |
| **License** | Apache-2.0 |

This is one of the most practical papers in the set because the official implementation is public and actively maintained. The repository has already evolved beyond the paper and now exposes a broader inference/fine-tuning framework, but the original heterogeneous-inference ideas remain available.

### 9. Performance Results

| Metric | Value | Baseline | Notes |
|--------|-------|----------|-------|
| **Speedup** | 4.62-19.74x prefill speedup; 1.25-4.09x decode speedup; up to 1.45x extra throughput from Expert Deferral | Fiddler, llama.cpp, PyTorch-based baselines | Headline claim from abstract and evaluation |
| **Prefill Throughput** | Example baseline case: 70.02 tok/s prefill on DeepSeek-V3 under a Fiddler-style A100 + 2x Xeon setup before KTransformers optimizations | Fiddler-style hybrid setup | Full prefill plots are reported over prompt lengths 32-8192; exact per-point values are mostly graphical |
| **Decode Throughput** | Example baseline case: 4.68 tok/s decode on DeepSeek-V3 in the same initial hybrid setup; later text notes decode throughput of 5.87 tok/s before Expert Deferral in an optimized large-model case | Fiddler-style hybrid setup | Decode evaluation uses prompt length 32 and output length up to 512 |
| **TTFT** | Not explicitly reported | N/A | The paper focuses on tokens/s rather than TTFT |
| **TPOT** | Not explicitly reported | N/A | Decode speed is reported in tokens/s, not per-token latency |
| **GPU Memory Usage** | Evaluated on 40GB A100 and 16GB RTX 4080; GPU-resident parameters are 17B / 13B / 8B for DS-3 / DS-2 / QW-2 | Baselines on same hardware | The paper is explicitly designed to reduce VRAM pressure by pushing routed experts to CPU |
| **CPU Memory Usage** | CPU-resident parameters are 654B / 223B / 49B for DS-3 / DS-2 / QW-2; the server has 1TB DDR5 per socket | Baselines on same hardware | Very large host DRAM budgets are central to the system design |
| **End-to-End Latency** | Not reported as a dedicated TTFT/latency table; throughput is the primary metric | N/A | Quality/speed tradeoff for Expert Deferral is analyzed instead |

Additional useful details:
- Workloads use **batch size 1**.
- Prefill prompt lengths range from **32 to 8192 tokens**.
- Decode tests use prompt length **32** and maximum output length **512**.
- For full-precision decode, KTransformers without Expert Deferral gives **2.42-4.09x** speedups over Fiddler and **1.25-1.76x** over llama.cpp.
- On quantized models, KTransformers shows **1.77-1.93x** speedups over llama.cpp.
- Expert Deferral provides up to **45%** additional decode improvement and improves CPU/GPU utilization in the DeepSeek-V3 showcase from **74%/28%** to **100%/37%**.
- The default LiveBench-style accuracy drop under Expert Deferral is about **0.5%**, much smaller than expert skipping.

### 10. Techniques & Innovation

KTransformers is primarily a **systems optimization paper**. Its main insight is that MoE local inference should treat the CPU as a real expert-compute device rather than a passive storage tier, then co-design kernels, memory layout, NUMA placement, and CPU/GPU scheduling around that choice.

The optional **Expert Deferral** mechanism is the most novel part: instead of forcing all routed experts to complete before the next layer’s attention, it delays some lower-priority experts to later layers so CPU expert execution overlaps more effectively with GPU attention. This increases utilization substantially while keeping accuracy loss small.

**Key mechanisms:**
- **Arithmetic-intensity-aware hybrid kernels**: AMX for high-ARI prefill work and AVX-512 for lower-ARI decode work
- **NUMA-aware tensor placement / tensor parallelism** to reduce cross-socket traffic
- **Single-graph decode scheduling** using CUDA Graphs and callback-based submit/sync hiding
- **Expert Deferral** for fine-grained CPU/GPU overlap during decode
- **YAML-driven module injection** for replacing HuggingFace modules with optimized operators and placement policies

**Builds upon / combines:**
- Fiddler-style CPU/GPU hybrid MoE execution
- llama.cpp as a practical local-inference baseline
- CUDA Graphs, FlashInfer, Marlin, quantization, and NUMA-aware systems tuning
- Shared-expert or popularity-aware partitioning between GPU-resident and CPU-resident experts

---

## 🎯 Relevance Assessment

| Criterion | Rating |
|-----------|--------|
| **Overall Relevance to Edge MoE Inference** | ⭐⭐⭐⭐☆ (4-5) |

**Key Strengths:**
- Explicitly targets **local low-concurrency inference** and evaluates **batch size 1**, which is much closer to your deployment setting than cloud-serving papers
- Strong match to your **performance-oriented CPU DRAM + accelerator** path because it makes CPU expert compute a first-class part of the design
- Requires **no retraining** and has an actively maintained **open-source implementation**
- Compares directly against both **Fiddler** and **llama.cpp**, making the engineering tradeoffs easier to interpret for your intended stack

**Key Limitations / Gaps:**
- Assumes a **very strong CPU and huge DDR5 memory pool**, which is much closer to a workstation/server than to the leanest edge product configuration
- Focuses on **CPU DRAM offload**, not **SSD offload**, so it does not solve your low-cost NPU + SSD path
- Hardware assumptions are CPU/GPU-centric; direct transfer to **NPU + CPU** will require nontrivial kernel and scheduling adaptation
- The most aggressive optimization, **Expert Deferral**, slightly perturbs model execution and introduces a small accuracy tradeoff

**Actionable for Our Project?**
Partially, but strongly so for the **performance SKU**. KTransformers is one of the best references for CPU-assisted MoE inference when you are willing to spend CPU DRAM and host compute; however, it is not a direct template for the **SSD-offload SKU** and its heavyweight host assumptions mean you should treat it as a systems pattern library rather than a drop-in architecture.

---

## 📝 Notes & Discussion Points

- KTransformers is best viewed as a more ambitious, more optimized evolution of the same general direction as Fiddler: **CPU+GPU hybrid MoE execution** rather than simple weight streaming.
- For your project, the most transferable ideas are probably **kernel specialization by workload regime**, **fine-grained overlap scheduling**, and **module-injection-style separation between model code and optimization policy**.
- The paper’s comparison against **llama.cpp** is especially relevant because it grounds the benefits of hybrid CPU expert execution in a baseline closer to your intended implementation environment.
- The downside is architectural heaviness: a dual-socket Xeon plus terabyte-scale DDR5 machine is a very different host environment from a small-form-factor Mini-PC paired with an NPU card.
- The official repository has already evolved beyond the paper and now includes features such as broader model support and additional hardware backends. That is useful for implementation study, but those repo-level capabilities should not be conflated with what the SOSP paper itself directly evaluates.