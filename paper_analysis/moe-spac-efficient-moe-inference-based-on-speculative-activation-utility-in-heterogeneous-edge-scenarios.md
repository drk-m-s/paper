# MoE-SpAc: Efficient MoE Inference Based on Speculative Activation Utility in Heterogeneous Edge Scenarios

## 📋 Metadata

| Field | Value |
|-------|-------|
| **Authors** | Shuhuai Li, Jianghao Lin, Dongdong Ge, Yinyu Ye |
| **Organization(s)** | Shanghai University; Shanghai Jiao Tong University; Stanford University |
| **Venue** | arXiv preprint |
| **Publication Date** | 2026-02-12 (arXiv v1) |
| **Citations** | 1 on Google Scholar at fetch time (May 2026); Semantic Scholar API returned 429 Too Many Requests |
| **arXiv / DOI** | arXiv:2603.09983 / 10.48550/arXiv.2603.09983 |
| **Open-Source Code** | https://github.com/lshAlgorithm/MoE-SpAc (MIT) |

---

## 🔍 Executive Summary

MoE-SpAc is an edge-oriented MoE inference system that repurposes speculative decoding as a memory-management signal rather than only a compute accelerator. It uses speculative activation frequencies to estimate expert utility online, then solves a per-layer integer optimization problem to decide which experts should stay on the GPU and which should execute on the CPU, with unified asynchronous prefetch and eviction. Relative to your project, this is one of the closest papers so far because it targets batch-size-1 edge inference, uses Qwen3-30B-A3B in the experiments, and has an open-source implementation built on llama.cpp.

---

## 📐 10-Dimension Analysis

### 1. MoE Inference vs. Training

| Aspect | Detail |
|--------|--------|
| **Type** | Inference |
| **Explanation** | MoE-SpAc is purely an **inference** system. It optimizes expert scheduling, prefetch, eviction, and CPU/GPU execution during serving under memory constraints. It does not introduce a training framework, routing-loss modification, or expert-balancing method for training. |

### 2. Edge vs. Cloud Scenario

| Aspect | Detail |
|--------|--------|
| **Target** | Edge (batch=1) |
| **Batch Size in Experiments** | 1 |
| **Hardware Constraints Mentioned** | Memory-constrained edge deployment, single-request inference, constrained GPU VRAM, and PCIe I/O bottlenecks between CPU memory and GPU memory |

This paper is explicitly framed as an **edge / personal-device** MoE inference system. The authors state that they use **batch size 1 to simulate edge deployment**, where inference is typically single-request and memory-constrained.

### 3. Hardware Setup

| Aspect | Detail |
|--------|--------|
| **Compute Device** | GPU + CPU |
| **Specific Hardware** | Single NVIDIA GeForce RTX 4090 GPU connected to CPU memory via PCIe 4.0 |
| **GPU Memory** | 24GB on the RTX 4090 (confirmed in Appendix Figure 6) |
| **CPU Memory** | Not specified |

The CPU model is not disclosed in the main text, but the system clearly assumes heterogeneous execution across CPU and GPU, with CPU memory serving as the backing store for cold experts.

### 4. Offloaded Weight Storage

| Aspect | Detail |
|--------|--------|
| **Storage Location** | CPU DRAM |
| **Storage Capacity** | Not specified |
| **Assumed Bandwidth** | PCIe 4.0, reported as 32GB/s |

MoE-SpAc is a **CPU-DRAM offloading** system. It does not evaluate SSD or remote storage as an expert tier.

### 5. MoE Models Evaluated

| Model | Total Params | Active Params | # Experts | Notes |
|-------|-------------|---------------|-----------|-------|
| Qwen3-30B-A3B | 30B (from model name) | 3B active (from model name) | Not explicitly tabulated in the paper | Main target / verifier model in the core experiments |
| DeepSeek-V2-Lite | Not stated in the paper | Not stated in the paper | Not stated in the paper | Compatibility analysis against the strongest baselines |
| Qwen3-235B-A22B | 235B (from model name) | 22B active (from model name) | Not explicitly tabulated in the paper | Used in appendix-level theoretical / empirical analysis of speculative reuse |

The paper also relies on a **separate dense draft model** for speculative decoding, specifically **Qwen3-4B-FP8** in the main setup and an **int4-quantized dense draft model** in the DeepSeek-V2-Lite compatibility study.

### 6. Model Modification / Retraining

| Aspect | Detail |
|--------|--------|
| **Architecture Changes Required?** | No |
| **Retraining Required?** | No |
| **Plug-and-Play with Off-the-Shelf Weights?** | No |
| **Details** | The target MoE model itself is not retrained or structurally modified, but deployment is **not plug-and-play** in the strict sense. MoE-SpAc requires a **separate draft model** for speculation plus a custom runtime and weight-conversion path in its llama.cpp-based implementation. |

This distinction matters for your project: the paper respects the “no model retraining” requirement, but it does introduce a system dependency on a compatible draft model.

### 7. Offline Profiling Required?

| Aspect | Detail |
|--------|--------|
| **Profiling Needed?** | No |
| **What is Profiled?** | N/A |
| **Profiling Duration** | N/A |
| **Profiling Dataset** | N/A |

MoE-SpAc is driven by **online utility estimation**, not offline hot-expert profiling. The system tracks speculative activation frequencies during inference and updates expert utility in real time.

### 8. Open-Source Code

| Aspect | Detail |
|--------|--------|
| **Available?** | Yes |
| **GitHub URL** | https://github.com/lshAlgorithm/MoE-SpAc |
| **License** | MIT |

This is unusually valuable for your use case because the released code is presented as an **implementation based on llama.cpp**. The repository includes runtime and model-conversion changes rather than being only a thin artifact dump.

### 9. Performance Results

| Metric | Value | Baseline | Notes |
|--------|-------|----------|-------|
| **Speedup** | 4.04× average TPS speedup over standard baselines; 41.9% average TPS improvement over `llama.cpp-w/SD` | Accelerate, vLLM, llama.cpp, llama.cpp-w/SD, MixtralOffload, MoE-Infinity, SP-MoE, Fate, HybriMoE | Main result across 7 benchmarks |
| **Prefill Throughput** | Not separately reported | N/A | The paper reports end-to-end TPS, not a separate prefill-only throughput metric |
| **Decode Throughput** | Not separately reported | N/A | TPS is reported at system level rather than decode-only token rate |
| **TTFT** | Not separately reported | N/A | The paper uses benchmark-level latency rather than standalone TTFT |
| **TPOT** | Not separately reported as a deployment metric | N/A | Appendix A provides a TPOT-oriented theoretical analysis, but not a standalone production-style TPOT table |
| **GPU Memory Usage** | Expert cache ratio set to 17%; draft model adds about 8% static memory overhead; OOM appears around 21% cache ratio in the reported setup | Baselines vary by cache policy | Memory results are described in cache-ratio terms rather than full VRAM accounting |
| **CPU Memory Usage** | Not explicitly reported | N/A | CPU DRAM stores cold experts, but capacity is not tabulated |
| **End-to-End Latency** | Average latency 16.31 vs 21.62 for `llama.cpp-w/SD` and 32.45 for HybriMoE | Same benchmark set as above | Average latency reduction is 24.6% vs the best SD baseline |

Representative table values from the main Qwen3-30B-A3B evaluation:
- **Average TPS:** 25.06 for MoE-SpAc vs 17.66 for `llama.cpp-w/SD`, 15.23 for `llama.cpp`, and 12.15 for HybriMoE.
- **Average latency:** 16.31 for MoE-SpAc vs 21.62 for `llama.cpp-w/SD` and 32.45 for HybriMoE.
- **Model-compatibility study:** on DeepSeek-V2-Lite, MoE-SpAc reports 53.0% / 55.0% / 51.0% TPS gains over the strongest baselines on MT-bench, HumanEval, and CNN/DM, respectively.

### 10. Techniques & Innovation

MoE-SpAc’s core innovation is to reinterpret **speculative decoding as an online lookahead sensor for expert scheduling**. Instead of using draft tokens only to accelerate verification, the system extracts speculative activation frequencies, compresses them into discrete utility scores, and uses those scores to coordinate expert placement, prefetch, eviction, and CPU/GPU execution. The result is a combined algorithm-and-systems design that turns expert offloading into a utility-guided scheduling problem rather than a pure cache-policy problem.

**Key mechanisms:**
- **Speculative Utility Estimator**: converts speculative activation frequencies into stable, discrete utility scores with inertial transition and adaptive boundary calibration
- **Heterogeneous Workload Balancer**: solves a small online integer optimization problem at each layer to set the threshold that separates GPU-resident hot experts from CPU-executed cold experts
- **Unified Asynchronous Execution Engine**: uses the same utility signal for both prefetch and eviction, reducing cache thrashing and hiding I/O behind drafting / verification work
- **Speculative-decoding reuse**: exploits multi-token verification windows to increase expert reuse and make expert-demand signals more informative than binary autoregressive activations

**Builds upon / combines:**
- Speculative decoding for LLM inference
- CPU-GPU heterogeneous expert execution as in Fiddler / kTransformers / HybriMoE-style systems
- Expert prefetching and expert caching literature
- Online scheduling under I/O and VRAM constraints rather than static placement or offline profiling

---

## 🎯 Relevance Assessment

| Criterion | Rating |
|-----------|--------|
| **Overall Relevance to Edge MoE Inference** | ⭐⭐⭐⭐☆ (4-5) |

**Key Strengths:**
- Explicitly targets **edge, batch-size-1, memory-constrained** MoE inference rather than cloud throughput serving
- Evaluates **Qwen3-30B-A3B**, which is very close to your target model family
- Uses **CPU-DRAM offloading plus heterogeneous CPU/GPU execution**, which maps well to your performance-oriented CPU+NPU path
- Provides **open-source llama.cpp-based code**, making it more actionable than most papers in this space
- Avoids **offline profiling** and instead adapts online to changing expert-demand patterns

**Key Limitations / Gaps:**
- Requires a **separate draft model**, which adds engineering complexity and memory overhead that may conflict with a very lean deployment target
- Does **not address SSD offloading**, so it is weaker for your low-cost NPU+SSD product direction
- Assumes **CUDA GPU semantics** and PCIe-connected CPU memory rather than an edge NPU runtime
- Reports system-level TPS and latency, but does **not provide TTFT / TPOT in the exact product-metric form** you care about

**Actionable for Our Project?**
Partially — and it is one of the stronger references so far for the **performance-oriented CPU DRAM + accelerator** path. The batch-size-1 edge framing, Qwen3 target model, and llama.cpp implementation make it unusually relevant. However, I would treat its **speculative-decoding stack as an optional second-stage optimization**, not as the mandatory core architecture, because the separate draft model and added runtime complexity may be too expensive for the cleanest first implementation.

---

## 📝 Notes & Discussion Points

- The PDF text extraction misread the arXiv identifier as `2603.09938`; the correct record is **arXiv:2603.09983**.
- This paper is one of the rare cases where the implementation angle matters almost as much as the paper itself: the public repo is explicitly **built on llama.cpp**, which is directly aligned with your intended software base.
- MoE-SpAc looks more relevant to your **performance SKU** than to the **low-cost SSD SKU**. It gives a concrete online CPU/accelerator scheduling design, but no SSD-tier design.
- The separate draft model is the main architectural tax. The paper itself notes that **self-speculation** could remove this extra model in future work, which is likely the right direction if you want to preserve simplicity and on-device memory.
- A practical path for your project would be to borrow **MoE-SpAc’s online utility-guided hot/cold scheduling** while initially keeping the rest of the runtime simpler than the full speculative stack.