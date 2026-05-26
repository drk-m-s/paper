# PROBE: Co-Balancing Computation and Communication in MoE Inference via Real-Time Predictive Prefetching

## 📋 Metadata

| Field | Value |
|-------|-------|
| **Authors** | Qianchao Zhu, Xucheng Ye, Yuliang Liu, Haodong Ouyang, Chengru Song |
| **Organization(s)** | Kling Infra, Kuaishou Technology |
| **Venue** | arXiv preprint |
| **Publication Date** | 2026-01-31 (v1), revised 2026-02-03 (v2) |
| **Citations** | N/A in accessible sources; Google Scholar indexes the paper, but the fetched view did not expose a visible count and Semantic Scholar API was rate-limited |
| **arXiv / DOI** | arXiv:2602.00509 / 10.48550/arXiv.2602.00509 |
| **Open-Source Code** | No official implementation found as of May 2026 |

---

## 🔍 Executive Summary

PROBE is a distributed MoE inference system for multi-GPU serving that proactively predicts next-layer expert hotspots and prefetches replicated experts to reduce expert-parallel stragglers. Its main contribution is a continuous lookahead pipeline that overlaps prediction, planning, and peer-to-peer expert transfer with the current layer’s dispatch, compute, and combine phases so that balancing overhead stays off the critical path. On 8-way Hopper hardware, PROBE reports up to 1.32× lower prefill latency and up to 1.26× higher decoding throughput than strong expert-parallel baselines under volatile continuous-batching workloads.

---

## 📐 10-Dimension Analysis

### 1. MoE Inference vs. Training

| Aspect | Detail |
|--------|--------|
| **Type** | Inference |
| **Explanation** | This is an **MoE inference** paper, not a training framework. It studies latency-critical serving under expert parallelism, continuous batching, and distributed load imbalance. The paper explicitly frames the problem as real-time inference straggler mitigation and evaluates prefill latency and decoding throughput. |

### 2. Edge vs. Cloud Scenario

| Aspect | Detail |
|--------|--------|
| **Target** | Cloud (batch>1) |
| **Batch Size in Experiments** | Large multi-request decoding batches; the paper sweeps per-rank batch size from 512 to 1536 in decoding and evaluates prefill with total input tokens from 265K to 640K across 8 ranks |
| **Hardware Constraints Mentioned** | 8-GPU NVSwitch node, expert parallelism, All-to-All collectives, continuous batching, CUDA Graph compatibility, and network-bandwidth hiding windows |

PROBE is decisively a **cloud-scale distributed serving** system, not an edge or batch-size-1 design. Its core problem is dynamic imbalance across GPUs under expert parallelism and continuous batching.

### 3. Hardware Setup

| Aspect | Detail |
|--------|--------|
| **Compute Device** | GPU only |
| **Specific Hardware** | 8× NVIDIA Hopper 141GB GPUs interconnected via 900 GB/s NVSwitch |
| **GPU Memory** | 141GB per GPU |
| **CPU Memory** | Not addressed |

The software stack is PyTorch 2.9, CUDA 12.9, NCCL 2.27.3, and NVSHMEM 3.3.20. The system is built on top of SGLang with DeepEP as the communication backend.

### 4. Offloaded Weight Storage

| Aspect | Detail |
|--------|--------|
| **Storage Location** | Other |
| **Storage Capacity** | Replicated expert buffer in peer GPU memory; double-buffered replica region supporting up to 3 redundant experts per rank, for 6 expert slots per device |
| **Assumed Bandwidth** | 900 GB/s NVSwitch interconnect; peer-to-peer expert transfer uses NVSHMEM symmetric memory and custom Triton remote-put kernels |

This paper does **not** offload cold experts to CPU DRAM or SSD. Instead, it performs **dynamic expert replication and peer-to-peer prefetching across GPUs**.

### 5. MoE Models Evaluated

| Model | Total Params | Active Params | # Experts | Notes |
|-------|-------------|---------------|-----------|-------|
| GPT-OSS-120B | 120B | N/A | 128 total, Top-4 active | 36 layers, BF16, sparser routing |
| Qwen3-MoE-235B | 235B | N/A | 128 total, Top-8 active | 93 layers, BF16 |

The paper also analyzes activation patterns and imbalance behavior during prefill and decoding for both models. Active-parameter counts are not explicitly reported.

### 6. Model Modification / Retraining

| Aspect | Detail |
|--------|--------|
| **Architecture Changes Required?** | No |
| **Retraining Required?** | No |
| **Plug-and-Play with Off-the-Shelf Weights?** | Yes |
| **Details** | PROBE does not modify the MoE model or require retraining/fine-tuning of the base weights. It clones the target layer’s router as a frozen prior and adds a lightweight residual MLP predictor for lookahead forecasting. Actual execution still follows the ground-truth router outputs, preserving semantic equivalence. |

### 7. Offline Profiling Required?

| Aspect | Detail |
|--------|--------|
| **Profiling Needed?** | No |
| **What is Profiled?** | N/A for a separate offline hot/cold expert profiling phase |
| **Profiling Duration** | N/A |
| **Profiling Dataset** | N/A |

The paper’s predictor is trained through **scale-driven online distillation** from the live inference stream rather than a separate offline profiling stage. This is materially different from offline hot-expert profiling systems.

### 8. Open-Source Code

| Aspect | Detail |
|--------|--------|
| **Available?** | No |
| **GitHub URL** | Not found |
| **License** | N/A |

The paper states that PROBE is implemented atop SGLang and DeepEP, but no official repository for PROBE itself was found.

### 9. Performance Results

| Metric | Value | Baseline | Notes |
|--------|-------|----------|-------|
| **Speedup** | Up to 1.32× lower prefill latency | SGLang | Reported in prefill on 8-way expert-parallel serving |
| **Prefill Throughput** | Not explicitly reported | N/A | Paper reports prefill latency / TTFT-style results instead |
| **Decode Throughput** | Up to 1.26× higher | DeepSeek-EPLB | Same batch size in decoding; reported over initial 500 decoding steps |
| **TTFT** | Not directly tabulated; prefill latency is reported instead | SGLang | Prefill evaluated over 265K to 640K total input tokens across ranks |
| **TPOT** | Not directly tabulated | SGLang / DeepSeek-EPLB | Paper reports throughput-latency Pareto curves rather than a standalone TPOT table |
| **GPU Memory Usage** | Replica overhead bounded to 6 expert slots per device | DeepSeek-EPLB / SGLang | Dynamic slot reuse is a design advantage, but total GB usage is not tabulated |
| **CPU Memory Usage** | Not addressed | N/A | CPU memory is not a focus of the system |
| **End-to-End Latency** | Lower decoding latency at a given throughput; exact end-to-end figures not fully tabulated | SGLang / DeepSeek-EPLB | Robustness experiments show stable throughput across abrupt semantic shifts |

Additional quantitative details reported in the paper:
- Predictor Top-K accuracy improves from roughly 70%–80% (frozen router only) to **87%–94%** after online distillation.
- The average imbalance ratio across 35 layers is reduced from **2.13 to 1.09** in the latency breakdown study.
- Computation latency skew (max/avg) drops from **2.27 to 1.18**.

### 10. Techniques & Innovation

PROBE’s core technical innovation is **Continuous Lookahead Pipelining** for distributed MoE serving. Instead of reacting after hotspot formation, it predicts next-layer expert demand, computes a bounded replication plan under a hardware-specific hiding window, and prefetches experts via split-phase peer-to-peer transfer so that balancing work does not stall the main inference path.

**Key mechanisms:**
- **Gate-Initialized Lookahead Predictor**: clones each target router as a frozen prior and adds a lightweight residual MLP to predict next-layer expert activations from the previous layer’s hidden state
- **Hardware-Aware Balance Planning**: solves dynamic expert replication and token reassignment under strict transfer-hiding constraints so the planner never exposes balancing overhead on the critical path
- **Phase-Locked Co-Scheduling**: aligns predictor, planner, and prefetch stages with complementary main-stream phases; split-phase transmission pauses around All-to-All combine to avoid contention
- **Dynamic GPU-to-GPU expert replication**: uses NVSHMEM symmetric memory plus custom Triton kernels to move expert weights just in time across ranks

**Builds upon / combines:**
- Expert parallel inference and All-to-All communication backends such as DeepEP
- Router distillation / lookahead prediction ideas
- Dynamic expert replication for straggler mitigation
- CUDA-graph-compatible asynchronous scheduling techniques

---

## 🎯 Relevance Assessment

| Criterion | Rating |
|-----------|--------|
| **Overall Relevance to Edge MoE Inference** | ⭐⭐☆☆☆ (2-5) |

**Key Strengths:**
- Strong systems insight into **predictive prefetching** and how to keep control overhead off the critical path
- Does **not require modifying or retraining the base MoE model**, which matches our preference for off-the-shelf weights
- Quantifies the interaction between **compute skew and communication skew**, which is useful when reasoning about any hierarchical MoE serving pipeline
- The lookahead predictor and split-phase scheduling ideas are conceptually portable to other prefetch problems

**Key Limitations / Gaps:**
- This is a **multi-GPU cloud serving** paper, not an edge deployment paper
- It does **not offload experts to CPU DRAM or SSD**; its storage tier is remote GPU HBM over NVSwitch
- It assumes **continuous batching and large per-rank batch sizes**, which is the opposite of our batch-size-1 target
- It relies on **8× Hopper 141GB GPUs and NVSwitch**, making the hardware model fundamentally different from an NPU M.2 card connected over PCIe 4.0 x4
- It focuses on expert-parallel All-to-All and straggler mitigation, not the memory-tier trade-offs that dominate edge MoE offloading

**Actionable for Our Project?**
Partially — only at the idea level. The predictive lookahead and phased prefetch scheduling concepts are worth remembering, but the concrete system design is not directly applicable to our llama.cpp-based edge NPU engine because it assumes distributed GPU expert parallelism, high-bandwidth GPU-to-GPU transfers, and large continuous batches rather than SSD/DRAM offloading for batch-size-1 inference.

---

## 📝 Notes & Discussion Points

- PROBE is best read as a **distributed serving optimization** paper rather than an MoE offloading paper.
- The most transferable concept is the **lookahead predictor**: if future expert demand can be predicted with sufficiently high confidence, prefetch can be scheduled off the critical path even in a different memory hierarchy.
- The paper’s “double penalty” framing is useful: hotspot experts hurt both **compute balance** and **communication balance** simultaneously. In our setting, that same reasoning could map to **compute balance + SSD/PCIe transfer contention**, even though the transport medium is different.
- The absence of CPU/SSD offloading, quantization discussion, and batch-size-1 results makes the paper a weak direct fit for our target architecture despite the solid systems design.
