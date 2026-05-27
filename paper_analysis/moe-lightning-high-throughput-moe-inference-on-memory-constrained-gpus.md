# MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs

## 📋 Metadata

| Field | Value |
|-------|-------|
| **Authors** | Shiyi Cao, Shu Liu, Tyler Griggs, Peter Schafhalter, Xiaoxuan Liu, Ying Sheng, Joseph E. Gonzalez, Matei Zaharia, Ion Stoica |
| **Organization(s)** | UC Berkeley; Stanford University |
| **Venue** | ASPLOS 2025 (Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 1) |
| **Publication Date** | 2025-03-30 (proceedings); arXiv v1 2024-11-18 |
| **Citations** | 78 visible citations on Google Scholar at fetch time (May 2026); Crossref is-referenced-by count 6 |
| **arXiv / DOI** | arXiv:2411.11217 / 10.1145/3669940.3707267 |
| **Open-Source Code** | No official repository found; the paper says the system is implemented on top of PyTorch, vLLM, and SGLang with custom CPU GQA kernels |

---

## 🔍 Executive Summary

MoE-Lightning is a systems paper about running MoE LLMs on memory-constrained GPUs by aggressively optimizing **offline batch throughput**, not interactive single-request latency. Its two main ideas are **CGOPipe**, a CPU-GPU-I/O pipelining schedule with paged weight transfers, and **HRM**, a Hierarchical Roofline Model used to pick hardware-aware execution policies offline.

Relative to your project, this is a meaningful reference for the **performance-oriented CPU-DRAM + accelerator** path because it explicitly studies CPU attention, CPU-resident weights and KV cache, limited GPU memory, and low-cost GPUs like T4/L4. The main mismatch is that the paper is explicitly framed around **offline batch-processing workloads** and evaluates large batched throughput rather than **batch-size-1 edge inference**.

---

## 📐 10-Dimension Analysis

### 1. MoE Inference vs. Training

| Aspect | Detail |
|--------|--------|
| **Type** | Inference |
| **Explanation** | MoE-Lightning is purely an **inference system**. It does not modify MoE training, routing losses, or expert specialization. The work is about runtime scheduling, offloading, and hardware-aware policy selection for existing checkpoints. |

### 2. Edge vs. Cloud Scenario

| Aspect | Detail |
|--------|--------|
| **Target** | Throughput-oriented batch inference on local / memory-constrained GPUs; not batch size 1 edge serving |
| **Batch Size in Experiments** | Not batch size 1. The paper optimizes large offline batches with reported micro-batch sizes such as 32, 36, 64, 100, 128, and 156, plus multiple micro-batches per pipeline. |
| **Hardware Constraints Mentioned** | Single low-cost GPUs such as T4 16GB and L4 24GB, or 2-4 T4 GPUs; limited CPU-GPU bandwidth; GPU-memory-capacity bottlenecks; large host DRAM budgets |

The paper repeatedly states that it focuses on **off-line, batch-processing workloads** such as model evaluation, synthetic data generation, data wrangling, form processing, and relational analytics. That makes it closer to batched local serving or offline throughput optimization than to the strict interactive edge scenario you care about.

### 3. Hardware Setup

| Aspect | Detail |
|--------|--------|
| **Compute Device** | GPU + CPU |
| **Specific Hardware** | Main evaluation uses 1x NVIDIA T4 (16GB), 1x NVIDIA L4 (24GB), 2x T4, and 4x T4 with Intel Xeon hosts; a separate case study explores synthetic 2x A100-80GB configurations |
| **GPU Memory** | 16GB, 24GB, 32GB, or 64GB aggregate in the main evaluation |
| **CPU Memory** | 192GB or 416GB host DRAM in the main evaluation |

This is one of the stronger matches among the reviewed papers if your performance SKU is willing to lean on a large CPU-memory pool. It is much less aligned with the low-memory, no-big-host-RAM edge interpretation.

### 4. Offloaded Weight Storage

| Aspect | Detail |
|--------|--------|
| **Storage Location** | CPU DRAM |
| **Storage Capacity** | 192GB or 416GB CPU DRAM in the evaluated systems |
| **Assumed Bandwidth** | Hardware-aware model parameterizes GPU, CPU, and CPU-GPU bandwidth; the L4 case study uses 100GB/s CPU-memory bandwidth and 32GB/s CPU-GPU bandwidth |

The paper discusses CPU DRAM and disk in the background, but the actual system explicitly **does not consider disk offloading**. In the main design, weights and KV cache are offloaded to CPU memory, not SSD.

### 5. MoE Models Evaluated

| Model | Total Params | Active Params | # Experts | Notes |
|-------|-------------|---------------|-----------|-------|
| Mixtral 8x7B | ~47B | ~13B | 8 | Main single-GPU benchmark model |
| Mixtral 8x22B | ~141B | ~39B | 8 | Multi-GPU T4 benchmark |
| DBRX | 132B | ~36B | 16 | Multi-GPU benchmark on 2x/4x T4 |

These counts reflect standard public model specifications where the paper names the model family but does not restate every parameter figure in detail. There is no evaluation on Qwen-family MoEs.

### 6. Model Modification / Retraining

| Aspect | Detail |
|--------|--------|
| **Architecture Changes Required?** | No |
| **Retraining Required?** | No |
| **Plug-and-Play with Off-the-Shelf Weights?** | Yes |
| **Details** | The paper applies runtime scheduling, CPU attention, paged weight transfer, and policy optimization around existing MoE checkpoints. It states that the implementation supports models compatible with vLLM model classes. |

This is good news for your project constraints. The burden is on the runtime and hardware-policy layer rather than on model surgery.

### 7. Offline Profiling Required?

| Aspect | Detail |
|--------|--------|
| **Profiling Needed?** | Yes, but lightweight |
| **What is Profiled?** | Peak GPU/CPU compute throughput, memory bandwidth, and hardware parameters used by the performance model; offline search over batch size, micro-batch size, and offload ratios |
| **Profiling Duration** | The offline MILP optimizer is reported to take less than a minute; separate hardware profiling duration is not explicitly reported |
| **Profiling Dataset** | No model-behavior calibration dataset is required; the optimizer uses workload statistics such as average prompt length |

Compared with systems that depend on heavy empirical fitting, this is a relatively practical offline step. The bigger issue for your use case is not profiling cost but whether the resulting policy assumptions still hold under batch-size-1 workloads.

### 8. Open-Source Code

| Aspect | Detail |
|--------|--------|
| **Available?** | No official release found |
| **GitHub URL** | N/A |
| **License** | N/A |

I found no official MoE-Lightning repository by title-based GitHub search. The paper is therefore informative as a systems reference, but not especially convenient as an implementation starting point.

### 9. Performance Results

| Metric | Value | Baseline | Notes |
|--------|-------|----------|-------|
| **Speedup** | Up to 10.3x higher generation throughput without request padding; 3.5x with request padding on a single GPU | FlexGen, FlexGen(c), DeepSpeed-Zero-Inference | Headline result for Mixtral 8x7B on single T4 / L4 settings |
| **Prefill Throughput** | Not reported as a separate top-line metric | N/A | The paper reports end-to-end generation throughput rather than a standalone TTFT-prefill number |
| **Decode Throughput** | Representative end-to-end generation throughput: 26.349 tok/s (Synthetic Reasoning, S1), 105.29 tok/s (Synthetic Reasoning, S2), 4.52 tok/s (Summarization, S1), 12.393 tok/s (Summarization, S2) | FlexGen, FlexGen(c), DeepSpeed-Zero-Inference | These are full generation-throughput numbers, not pure decode-kernel timings |
| **TTFT** | Not explicitly reported | N/A | The evaluation target is throughput, not first-token latency |
| **TPOT** | Not explicitly reported | N/A | The paper uses tokens/s throughput rather than TPOT |
| **GPU Memory Usage** | Supports Mixtral 8x7B on 16GB T4 / 24GB L4 and larger MoEs on 2-4 T4s (32-64GB aggregate) | Baselines compared on the same hardware envelopes | The paper frames achievable throughput as often bounded by GPU-memory capacity |
| **CPU Memory Usage** | Can reach the throughput upper bound with 2-3x less CPU memory than prior offloading systems; evaluated hosts provide 192GB or 416GB DRAM | Existing offloading systems | This is one of the strongest practical results in the paper |
| **End-to-End Latency** | Not the optimization target; evaluation is reported in total generation throughput over prefill + decode | N/A | Hard to map directly onto your TTFT and interactive decode targets |

Additional details worth retaining:
- On Mixtral 8x22B, MoE-Lightning reports **2.77-3.38x** higher throughput on **4x T4** than on **2x T4**, showing super-linear scaling under tensor parallelism.
- On DBRX, it reports about **2.1-2.8x** throughput improvement when scaling from **2 GPUs to 4 GPUs** in the tested settings.
- The strongest paper claim is not just raw speedup, but that better CPU-GPU-I/O overlap allows similar throughput with substantially **less CPU memory pressure**.

### 10. Techniques & Innovation

MoE-Lightning combines a systems scheduler with a lightweight analytical optimizer. The main insight is that, for memory-constrained MoE inference, the bottleneck can move between GPU memory capacity, CPU-GPU bandwidth, CPU attention, KV transfer, and MoE FFN execution, so a good runtime must jointly schedule computation and transfers instead of only offloading weights layer by layer.

Its novelty is less about predicting future experts and more about **resource-overlap engineering**: paged weight transfers, CPU attention when that is advantageous, and a roofline-style performance model that chooses a good batch / micro-batch / offload policy offline.

**Key mechanisms:**
- **CGOPipe**: a CPU-GPU-I/O pipeline that overlaps GPU work, CPU work, and multiple transfer types
- **Paged weight transfer**: slices expert weights into pages so different transfers can be interleaved with less pipeline bubble
- **HRM**: a Hierarchical Roofline Model that estimates the bottleneck regime and guides policy search
- **Offline MILP policy search**: picks batch size, micro-batch size, and CPU/GPU placement ratios for a given hardware-workload-model tuple
- **Tensor parallelism and balanced request batching**: extends the method to 2-4 T4 setups and variable-length requests

**Builds upon / combines:**
- FlexGen-style offloading for limited-GPU-memory inference
- FastDecode-style heterogeneous CPU/GPU pipeline thinking
- Roofline-model performance analysis
- vLLM / SGLang-style serving infrastructure rather than llama.cpp-style lightweight runtimes

---

## 🎯 Relevance Assessment

| Criterion | Rating |
|-----------|--------|
| **Overall Relevance to Edge MoE Inference** | ⭐⭐⭐☆☆ (3-5) |

**Key Strengths:**
- Directly studies **limited-GPU-memory MoE inference** on inexpensive hardware such as **T4** and **L4**, which is closer to your hardware budget than H100-centric papers
- Strongly relevant to the **performance-oriented CPU-DRAM + accelerator** path because it explicitly uses **CPU compute plus CPU-memory offloading**, not just storage offload
- Requires **no model retraining or architecture modification**
- The analytical framing is useful: HRM gives a clean way to reason about when CPU attention or different offload ratios actually help
- Shows a concrete systems benefit from reducing CPU-memory demand, which matters if host memory is the real bottleneck in a consumer deployment

**Key Limitations / Gaps:**
- The paper is explicitly about **offline batch throughput**, not **interactive batch-size-1 inference**
- It does **not** address **SSD offloading**, so it is not useful for your low-cost NPU + SSD path
- No official code release was found
- The implementation stack is **PyTorch + vLLM + SGLang**, not **llama.cpp**, so integration lessons are less direct for your target runtime
- It assumes **very large CPU DRAM budgets** relative to what some edge devices can tolerate
- It evaluates Mixtral and DBRX, not Qwen3.6-35B-A3B or Qwen3-30B-A3B

**Actionable for Our Project?**
Partially. MoE-Lightning is a good reference for the **performance SKU** if you want to explore CPU-assisted execution and hardware-aware policy search, but it is not a direct solution for your **batch-size-1 NPU product** and contributes little to the **SSD-offload SKU**.

---

## 📝 Notes & Discussion Points

- The extracted PDF text misread the arXiv identifier as **2411.21171**; the correct preprint is **arXiv:2411.11217**.
- The paper explicitly states that it **does not consider disk offloading**, which sharply limits its relevance to the SSD-backed deployment mode.
- A reusable idea for your project is the **HRM-style offline policy search**: the optimizer reportedly takes **less than a minute**, so this kind of hardware-aware search may still be acceptable if you later expose CPU+NPU hybrid modes.
- One of the more interesting results is that **CPU attention can outperform KV-cache transfer** over some operating regimes, which supports hybrid CPU compute only when throughput batching is large enough; that same argument is much weaker for batch-size-1 interactive decoding.
- Total and active parameter counts in the model table use common public model specifications where the paper itself only names the model family.