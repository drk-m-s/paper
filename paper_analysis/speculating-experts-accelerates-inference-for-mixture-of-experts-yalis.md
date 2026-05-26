# Speculating Experts Accelerates Inference for Mixture-of-Experts (YALIS)

## 📋 Metadata

| Field | Value |
|-------|-------|
| **Authors** | Vivan Madan, Prajwal Singhania, Abhinav Bhatele, Tom Goldstein, Ashwinee Panda |
| **Organization(s)** | University of Maryland; Together AI |
| **Venue** | arXiv preprint |
| **Publication Date** | 2026-03-09 (arXiv v1) |
| **Citations** | 0 visible citations on Google Scholar at fetch time (May 2026) |
| **arXiv / DOI** | arXiv:2603.19289 / 10.48550/arXiv.2603.19289 |
| **Open-Source Code** | https://github.com/axonn-ai/yalis/tree/offload_prefetch (Apache-2.0 with LLVM Exceptions) |

---

## 🔍 Executive Summary

This paper introduces a parameter-free expert prefetching scheme for memory-constrained MoE inference. It uses internal model representations — specifically "quasi-hidden states" formed from the current residual stream and precomputed per-expert default vectors — to predict which experts the next layer's router will select, then overlaps CPU→GPU expert transfers with ongoing computation. The key insight is speculative execution: instead of treating mispredictions as cache misses that must be re-fetched, prefetched experts are executed directly, which preserves the compute-memory overlap but can degrade accuracy on some models. For architectures where router-based speculation hurts accuracy (e.g., Qwen3-30B-A3B on math tasks), a lightweight neural estimator is trained via distillation to improve hit rates on high-drift layers.

---

## 📐 10-Dimension Analysis

### 1. MoE Inference vs. Training

| Aspect | Detail |
|--------|--------|
| **Type** | Inference |
| **Explanation** | YALIS is purely an **inference** system. The paper introduces an inference-time expert prefetching scheme that operates on pretrained models during serving. It does not address MoE training, load-balancing losses, or routing optimization during training. The neural estimator is a small auxiliary network trained via distillation from the target model's router, but this is inference-side tooling, not model retraining. |

### 2. Edge vs. Cloud Scenario

| Aspect | Detail |
|--------|--------|
| **Target** | Edge (batch=1) / single-GPU consumer hardware |
| **Batch Size in Experiments** | 1 |
| **Hardware Constraints Mentioned** | Memory-constrained settings where model parameters exceed available GPU HBM; single-user local inference; small batch sizes where expert transfers are minimized |

The paper explicitly states it focuses on "memory-constrained inference settings" where models do not fit in GPU memory, and "B=1, as in single-user local inference." This is one of the more clearly batch-size-1-oriented papers in the literature.

### 3. Hardware Setup

| Aspect | Detail |
|--------|--------|
| **Compute Device** | GPU only (CPU is a memory tier, not a compute tier) |
| **Specific Hardware** | NVIDIA RTX A6000 (48GB), NVIDIA A100 (80GB), NVIDIA GH200 (96GB) |
| **GPU Memory** | 48GB, 80GB, 96GB |
| **CPU Memory** | 128GB (A6000 setup), 256GB (A100 setup), 480GB (GH200 setup) |

The GH200 setup uses NVLink C2C (not PCIe), which has much higher bandwidth than the PCIe 4.0 connections on the A6000 and A100 setups. The A6000 with 48GB + 128GB CPU DRAM + PCIe 4.0 is the closest analogy to your edge NPU configuration, though with CUDA semantics rather than an NPU runtime.

### 4. Offloaded Weight Storage

| Aspect | Detail |
|--------|--------|
| **Storage Location** | CPU DRAM (pinned memory) |
| **Storage Capacity** | 128GB–480GB CPU DRAM |
| **Assumed Bandwidth** | PCIe 4.0 (theoretical 32GB/s uni-directional); NVLink C2C on GH200 |

The paper is a pure **CPU-DRAM offloading** system. Offloaded expert weights are stored in pinned CPU memory to enable faster asynchronous transfers. Non-expert parameters (attention weights, router weights, norms) remain resident on GPU. The paper does not evaluate or discuss SSD offloading.

### 5. MoE Models Evaluated

| Model | Total Params | Active Params | # Experts | Notes |
|-------|-------------|---------------|-----------|-------|
| Qwen3-30B-A3B | 30B | 3B | 128 experts per layer, 48 layers, top-K routing | Expert weights ~54GB, non-expert ~3GB (bf16) |
| GLM-4.7-Flash | Not stated | Not stated | 64 experts per layer, 47 layers | Expert weights ~53GB, non-expert ~3GB (bf16); shared expert discarded due to engine limitation |
| GPT-OSS-120B | 120B | Not stated | 32 experts per layer, 24 layers | Expert weights ~213GB, non-expert ~4GB (bf16) |
| Qwen3-235B-A22B | 235B | 22B | 128 experts per layer, 94 layers | Expert weights ~230GB, non-expert ~14GB (bf16) |

All evaluations use **bf16 precision**. The paper does not evaluate quantized models.

### 6. Model Modification / Retraining

| Aspect | Detail |
|--------|--------|
| **Architecture Changes Required?** | No |
| **Retraining Required?** | No |
| **Plug-and-Play with Off-the-Shelf Weights?** | Mostly — default vectors require a one-time offline computation pass |
| **Details** | The paper operates directly on existing pretrained MoE models without changing the model architecture. However, it requires precomputed **default vectors** (per-expert average activations aggregated offline) and, for accuracy-sensitive layers, an optional lightweight neural estimator trained via distillation from the target router. Neither requires modifying or retraining the target model itself. |

This is a strong match for your "no model modification" requirement — the default vectors are a lightweight offline preprocessing step, not model surgery, and the estimators are small auxiliary networks.

### 7. Offline Profiling Required?

| Aspect | Detail |
|--------|--------|
| **Profiling Needed?** | Yes — one-time offline pass |
| **What is Profiled?** | Default vectors: per-expert average activations aggregated by running inference on representative data |
| **Profiling Duration** | Not reported explicitly; the estimator training requires ~4M–5M tokens (< 1 epoch on a small calibration set) |
| **Profiling Dataset** | Not specified; implied to be a general text corpus or the training data's validation split |

The default vectors are computed once offline and can be reused across inference sessions for the same model. The neural estimator training uses a distillation approach on 4M–5M tokens. Both are lightweight offline steps that do not require per-user or per-deployment calibration cycles.

### 8. Open-Source Code

| Aspect | Detail |
|--------|--------|
| **Available?** | Yes |
| **GitHub URL** | https://github.com/axonn-ai/yalis/tree/offload_prefetch |
| **License** | Apache-2.0 with LLVM Exceptions |

The code is in the **YALIS** inference engine (Yet Another LLM Inference System), which is a Python/C++ framework supporting CPU offloading, torch.compile, CUDA Graphs, and FlashAttention. The `offload_prefetch` branch specifically implements this paper's prefetching scheme. YALIS is **not llama.cpp** — it is a PyTorch-based system that uses CUDA streams for async overlap. This makes it less directly reusable for your llama.cpp-based project, but the algorithmic ideas are implementation-agnostic.

The README includes a `CPUOffloadConfig` API with prefetch modes (`all`, `selective`, `none`), default-vector routing, and a complete working example at `examples/infer_cpu_offload.py`.

### 9. Performance Results

| Metric | Value | Baseline | Notes |
|--------|-------|----------|-------|
| **Speedup** | 9–14% TPOT reduction on A6000; 5–8% on A100/GH200 | On-demand CPU expert loading in YALIS | Gains are larger on weaker GPUs (A6000) where compute time is smaller relative to copy time |
| **Prefill Throughput** | Not separately reported | N/A | Paper notes prefill has all experts active with large prompts, making prefetching trivially "load all experts"; prefill is not the focus |
| **Decode Throughput** | Not reported as tokens/s | N/A | TPOT is the primary metric, not tokens/s |
| **TTFT** | Not reported | N/A | Not addressed |
| **TPOT** | 9–14% lower than on-demand loading on A6000 across context lengths 1024–65536 | On-demand loading | Gains increase with context length because compute time grows while copy time stays constant |
| **GPU Memory Usage** | Non-expert parameters only (3GB–14GB depending on model); expert weights are streamed on-demand | N/A | Expert cache ratio is 0 (no expert caching); all experts except the current layer's are on CPU |
| **CPU Memory Usage** | 54GB–230GB for expert weights (model-dependent) | N/A | Pinned CPU memory; exact capacity is not the bottleneck |
| **End-to-End Latency** | Not reported as an end-to-end latency number | N/A | TPOT is the primary latency metric |

**Representative TPOT values** for Qwen3-30B-A3B on A6000:
- Context 1024: On-demand ~158.9ms → Prefetch ~144.4ms (9.1% reduction)
- Context 4096: On-demand ~159.7ms → Prefetch ~145.7ms (8.8% reduction)
- Context 16384: On-demand ~160.2ms → Prefetch ~145.7ms (9.1% reduction)
- Context 65536: On-demand ~169.4ms → Prefetch ~146.0ms (13.8% reduction)

The maximum theoretical speedup is 2× (when compute and copy times are exactly equal). In practice, copy time dominates (84–88% of TPOT for Qwen3-30B-A3B), so the achievable overlap is limited to the compute fraction (~8–13%).

### 10. Techniques & Innovation

The core innovation is a **parameter-free expert prefetching scheme** that uses internal model representations — specifically "quasi-hidden states" formed by combining the current layer's residual stream with precomputed per-expert default vectors — to predict the next layer's expert selection. Unlike prior prefetching approaches that treat mispredictions as cache misses requiring re-fetch, this paper proposes **speculative execution** where prefetched experts are run directly without verifying against the true router. A companion lightweight neural estimator can replace router-based speculation on layers with high representational drift.

**Key mechanisms:**
- **Quasi-hidden states**: combine the current layer's normalized residual stream with a weighted combination of per-expert default vectors (precomputed offline) to approximate the next layer's router input
- **Default vectors**: per-expert average activations computed once offline; used to bias the quasi-hidden state toward the next-layer routing signal
- **Speculative execution**: prefetched experts are executed directly with their predicted routing weights rather than being treated as cache hints; mispredictions do not trigger re-fetch
- **Neural estimator**: a lightweight feed-forward network trained via distillation to predict router logits; applied selectively on high-drift layers where router-based speculation accuracy is low (e.g., early layers in Qwen3-30B-A3B)
- **Hybrid prefetching**: uses router-based speculation on most layers and the estimator only on low-hit-rate layers, recovering most of the accuracy gap while minimizing overhead

**Builds upon / combines:**
- Expert prefetching literature (MoE-Infinity, SP-MoE, Fate, HybriMoE)
- Default vectors from Panda et al. (2025)
- Async CUDA stream overlap for CPU→GPU transfers
- Distillation-based routing prediction
- Single-GPU resource-constrained inference

---

## 🎯 Relevance Assessment

| Criterion | Rating |
|-----------|--------|
| **Overall Relevance to Edge MoE Inference** | ⭐⭐⭐⭐☆ (4-5) |

**Key Strengths:**
- Explicitly targets **batch-size-1, single-user, memory-constrained** local inference rather than cloud batching
- Evaluates **Qwen3-30B-A3B**, which is virtually the same model family as your target (qwen3.6-35B-A3B)
- **Parameter-free core method** operates on off-the-shelf model weights with no model modification; the approach is clean and conceptually simple
- The quasi-hidden state + default vector mechanism is **lean and implementable** — it does not require a separate draft model, online integer optimization, or complex scheduling infrastructure
- Open-source implementation with a clear CPU offloading API and documented examples

**Key Limitations / Gaps:**
- Only addresses **CPU-DRAM offloading**, not **SSD offloading**, so it does not cover your low-cost product SKU
- The performance gains are **modest** (5–14% TPOT reduction) compared to MoE-SpAc's 42% over SD baselines — this is a simpler but less impactful approach
- For Qwen3-30B-A3B, **router-based speculative execution degrades accuracy** significantly on math tasks (e.g., GSM8K drops from 0.950 to 0.576), requiring the neural estimator or hybrid variant to recover performance
- Code is in **YALIS (PyTorch/CUDA)**, not llama.cpp, reducing direct reusability for your software stack
- Evaluates **bf16 only**; no quantization results are reported, which is critical for your 4-bit deployment target
- The default vectors require a **one-time offline pass** over calibration data, adding a deployment step (though a lightweight one)

**Actionable for Our Project?**
Partially — and it is one of the cleaner algorithmic references for a **simple, lightweight prefetching module** in a CPU-DRAM offloading engine. The quasi-hidden state mechanism is conceptually easy to implement in llama.cpp and does not require a separate draft model. However, the modest 5–14% gain ceiling, the accuracy degradation on Qwen3 models under pure speculative execution, and the YALIS-only codebase mean this is better treated as a **foundational building block** rather than a complete architecture reference. The approach likely needs combination with an expert caching scheme (for hot experts) and quantization-aware evaluation to be practical for your 4-bit NPU target.

---

## 📝 Notes & Discussion Points

- The PDF text extraction misread the arXiv identifier as `2603.09928`; the correct record is **arXiv:2603.19289**.
- This paper and MoE-SpAc are the only two papers analyzed so far that explicitly use **Qwen3-30B-A3B** as a primary evaluation model. For your project, this model-family alignment is a significant filter criterion.
- The paper's own **Future Work** section explicitly mentions "more resource-constrained deployments (e.g., smartphones, robotic platforms, and embedded systems), where prefetching may be applied to offloading involving disk-CPU memory transfers" — this suggests the authors see SSD offloading as a natural extension, though they haven't evaluated it yet.
- Compared to MoE-SpAc, YALIS's prefetching is **simpler and leaner** (no draft model, no online integer optimization, no speculative decoding infrastructure), but also delivers **smaller gains** (5–14% vs. 42%). If simplicity is paramount, YALIS is the more attractive reference; if performance is paramount, MoE-SpAc's speculative-decoding stack is stronger.
- A practical synthesis for your project could be: adopt YALIS's **quasi-hidden state + default vector** mechanism as the low-overhead prefetching layer, combine it with **Fiddler-style CPU vs. accelerator execution decisions**, and optionally add **MoE-SpAc-style utility-guided scheduling** as a second-stage optimization. None of these require model modification.
- The accuracy degradation on Qwen3-30B-A3B under pure speculative execution is a real concern. The paper's Hybrid-PF approach (estimator on early layers only) recovers most accuracy, but this adds a trained component to an otherwise parameter-free system. If you want to stay fully training-free, you may need to limit speculative execution to later layers where accuracy is preserved, or treat mispredictions as cache misses (like traditional prefetching) rather than executing speculatively.
