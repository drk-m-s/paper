# Research Report: MoE Offloading Techniques (2024–2026)

> **Survey Date:** May 2026 | **Papers Covered:** 48 | **Focus:** Edge NPU + llama.cpp MoE offloading

---

## Executive Summary

The explosion of sparse Mixture-of-Experts (MoE) LLMs—Mixtral, DeepSeek-V2/V3, Qwen3, and others—has driven a surge of research into how to run these models on memory-constrained hardware. From 2024 through early 2026, at least 48 significant papers have addressed the core bottleneck: MoE models have enormous total parameter counts (22B–1T+) but only activate a small fraction per token, creating an inherently memory-I/O-bound workload rather than compute-bound. Research has converged on five dominant strategies: (1) software-managed expert caches with sparsity-aware replacement; (2) predictive prefetching using lightweight gating predictors; (3) hybrid CPU-GPU-NPU compute to leverage the CPU's DRAM bandwidth; (4) expert compression, quantization, and merging to reduce transfer volume; and (5) speculative decoding combined with offloading to hide I/O latency.

For the target project—an NPU M.2 card (56 TOPS INT8, 10 GB HBM, PCIe 4.0 ×4 at 8 GB/s) running Qwen3-35B-A3B at >25 tok/s with llama.cpp—the most directly applicable papers are those targeting single-device, batch-1 inference with PCIe-bandwidth awareness. The 2024–2025 wave (EdgeMoE, MoE-Infinity, HOBBIT, MoE-Lightning, HybriMoE) established the baseline patterns. The 2025–2026 wave (DALI, SpecMoE family, TriMoE, NPUMoE, FlashMoE, SliceMoE) has refined these patterns substantially. A critical gap remains: **no published work targets a PCIe-attached edge NPU with HBM, batch=1, llama.cpp integration, and SSD offloading simultaneously**—making this project novel.

The key architectural insight that should drive design is the three-tier memory hierarchy: NPU HBM (hot non-expert weights + hottest experts) → CPU DRAM (warm experts) → NVMe SSD (cold experts). PCIe 4.0 ×4 at 8 GB/s constrains the NPU↔CPU transfer; the CPU's internal DRAM bandwidth (68 GB/s) and the SSD bandwidth (8 GB/s shared with PCIe) are the other key limits. The most performant systems overlap I/O with computation using asynchronous prefetching driven by expert activity prediction.

---

## Paper Details

---

### 1. Fast Inference of Mixture-of-Experts Language Models with Offloading

**Authors:** Artyom Eliseev, Denis Mazur (Independent) | **Venue:** arXiv Technical Report | **Date:** 2023-12-28 | **Citations:** ~350 | **Importance:** ⭐⭐⭐⭐⭐ (5/5)

**Core Technique:** Foundational offloading framework for MoE LLMs on consumer hardware. Stores non-expert weights in GPU VRAM and expert weights in CPU RAM, using LRU caching plus a simple "stay-on-device" heuristic for frequently activated experts. Introduces expert prefetching that overlaps CPU→GPU transfer with GPU computation using CUDA streams. Works without any model modification.

**Hardware:** Single consumer GPU (RTX 3090 24 GB, free-tier Colab T4 16 GB), CPU RAM 32–64 GB | **Models:** Mixtral-8x7B (with mixed INT4/INT8 quantization) | **Benchmarks:** Token generation speed (tok/s), memory footprint

**Key Results:** Enables Mixtral-8x7B inference on a single T4 GPU (16 GB VRAM); achieves ~2 tok/s on consumer hardware that could not otherwise run the model. Provides the conceptual blueprint for all subsequent expert offloading work.

**Pros (for project goal):**
- Requires zero model modification; any HuggingFace checkpoint works directly
- LRU + prefetch pattern is directly implementable in llama.cpp
- Explicitly addresses PCIe bandwidth as the dominant bottleneck—directly applicable to PCIe 4.0 ×4 constraint
- Provides the basic overlapping-async-prefetch-with-compute pattern this project should adopt

**Cons (for project goal):**
- Targets GPU not NPU; transfer protocol is CUDA-specific, not PCIe-generic
- LRU cache replacement is suboptimal; better predictors now exist (see DALI, FlashMoE)
- CPU-side experts idle when GPU/NPU is active; does not leverage CPU compute
- 2 tok/s is far below the project's 25 tok/s target; needs substantial improvement

**URL:** [arXiv:2312.17238](https://arxiv.org/abs/2312.17238)

---

### 2. MoE-Infinity: Efficient MoE Inference on Personal Machines with Sparsity-Aware Expert Cache

**Authors:** Leyang Xue et al. (University of Edinburgh) | **Venue:** arXiv preprint | **Date:** 2024-01-25 (v3: 2025-03-12) | **Citations:** ~80 | **Importance:** ⭐⭐⭐⭐⭐ (5/5)

**Core Technique:** Sparsity-aware expert cache for single-user (batch=1) personal machines. Key insight: at batch=1, expert activation exhibits high temporal locality—the same experts are reused across consecutive tokens. MoE-Infinity traces expert activation patterns during inference, builds a frequency/recency-weighted cache, and uses activation traces to guide both replacement (evict least-likely-needed) and prefetching (preload predicted experts). Implemented as a standalone Python system with public code.

**Hardware:** Personal machine / consumer GPU (RTX 3090, A100, T4) + CPU RAM | **Models:** DeepSeek-MoE 16B, Mixtral 8x7B, Mixtral 8x22B | **Benchmarks:** Per-token latency (ms), throughput (tok/s) vs. vLLM, Ollama, DeepSpeed, BrainStorm

**Key Results:** 3.1–16.7× per-token latency improvement over vLLM, Ollama, DeepSpeed, and BrainStorm on DeepSeek and Mixtral. At batch=1 on a single RTX 3090, achieves practical speeds for personal use. Code: https://github.com/EfficientMoE/MoE-Infinity

**Pros (for project goal):**
- Explicitly designed for batch=1, single-user scenarios—perfectly matching the M.2 card edge use case
- Sparsity-aware trace-based cache is directly implementable in llama.cpp without model modification
- Quantifies the temporal locality of expert activations—useful empirical data for designing the project's cache
- Open-source; can be studied for implementation patterns

**Cons (for project goal):**
- Implemented on CUDA GPUs; porting to a custom NPU backend requires significant engineering
- Does not consider a three-tier hierarchy (NPU HBM / DRAM / SSD); assumes GPU↔CPU only
- Does not leverage CPU computation; NPU HBM + CPU DRAM compute synergy unexplored
- Peak performance still below 25 tok/s target for large MoE models

**URL:** [arXiv:2401.14361](https://arxiv.org/abs/2401.14361)

---

### 3. EdgeMoE: Empowering Sparse Large Language Models on Mobile Devices

**Authors:** Rongjie Yi et al. (Beijing University of Posts and Telecom) | **Venue:** arXiv (originally Aug 2023, v2 Mar 2025) | **Date:** 2023-08-28 | **Citations:** ~120 | **Importance:** ⭐⭐⭐⭐⭐ (5/5)

**Core Technique:** On-device inference engine for edge hardware (mobile/embedded). Two key innovations: (1) **Expert-wise bitwidth adaptation**—offline profiling assigns each expert a precision level (INT8/INT4/INT2) based on sensitivity, reducing transfer volume without retraining; (2) **Expert preloading**—predicts activated experts ahead of time using a lightweight predictor and preloads via compute-I/O overlap. Non-expert weights stay in device memory; expert weights on flash/storage. Requires only an offline calibration pass, no model retraining.

**Hardware:** Mobile edge devices (ARM CPU, limited RAM) | **Models:** Mixtral 8x7B, OPT-Switch-13B | **Benchmarks:** Memory reduction, inference speed (tok/s), accuracy (perplexity)

**Key Results:** Significant memory savings (experts compressed 2–4×); enables models otherwise unrunnable on edge devices; expert preloading reduces stall overhead by ~30–50%.

**Pros (for project goal):**
- Expert-wise bitwidth adaptation directly applicable to reducing NPU↔CPU/SSD transfer volume (critical for 8 GB/s PCIe)
- Offline calibration + prediction approach aligns with background.md requirement for optional profiling phase
- Works without model retraining—just offline analysis, compatible with pulling from HuggingFace
- Compute-I/O pipeline pattern is exactly what the project needs

**Cons (for project goal):**
- Targets ARM CPU cores, not a specialized NPU with HBM; compute patterns differ significantly
- The lightweight predictor requires some offline calibration data—minor but adds deployment complexity
- Does not exploit CPU-side expert compute; pure prefetch approach
- Expert bitwidth adaptation technically modifies quantization parameters (borderline "no model modification")

**URL:** [arXiv:2308.14352](https://arxiv.org/abs/2308.14352)

---

### 4. MoNDE: Mixture of Near-Data Experts for Large-Scale Sparse Models

**Authors:** Taehyun Kim et al. (Seoul National University) | **Venue:** DAC 2024 | **Date:** 2024-05-29 | **Citations:** ~40 | **Importance:** ⭐⭐⭐⭐ (4/5)

**Core Technique:** Near-data processing (NDP) approach that reverses the offloading paradigm: instead of transferring cold expert weights to the GPU, MoNDE **executes cold experts inside the host memory device** (DIMM-NDP or CXL memory with compute). Only hot experts and small activations are moved to GPU. Reduces data movement volume from (large expert weights) to (small activation tensors).

**Hardware:** GPU + DIMM-NDP or CXL-attached memory with compute units | **Models:** Switch-C 1.5T, Mixtral 8x7B | **Benchmarks:** Speedup over CPU offloading baselines, throughput

**Key Results:** 2–4× speedup over parameter offloading frameworks for encoder and decoder operations.

**Pros (for project goal):**
- The conceptual near-data computing insight is directly relevant: instead of pulling experts to the NPU over PCIe, could execute them on CPU (which is near its DRAM at 68 GB/s). This is the "CPU computes cold experts" mode in background.md
- Quantifies the trade-off between transfer volume vs. near-data compute efficiency—useful for the project's hybrid CPU+NPU design
- No model modification required

**Cons (for project goal):**
- Requires specialized NDP hardware (DIMM-NDP); project has a standard CPU+DRAM setup, not NDP-enabled DRAM
- Does not address SSD offloading (third tier)
- GPU-centric; NPU specifics not addressed
- Published in 2024; newer hybrid approaches (TriMoE) have advanced this concept further

**URL:** [arXiv:2405.18832](https://arxiv.org/abs/2405.18832)

---

### 5. AdapMoE: Adaptive Sensitivity-based Expert Gating and Management for Efficient MoE Inference

**Authors:** Shuzhang Zhong et al. (Peking University) | **Venue:** ICCAD 2024 | **Date:** 2024-08-18 | **Citations:** ~15 | **Importance:** ⭐⭐⭐ (3/5)

**Core Technique:** Identifies that not all experts are equally sensitive to removal or replacement. Uses a sensitivity metric (based on activation magnitudes) to (1) classify experts as "critical" vs. "non-critical" and (2) guide expert gating (dropping low-sensitivity activations) and cache management (always retain high-sensitivity experts). Reduces memory footprint without significant accuracy loss.

**Hardware:** Single GPU (A100 80 GB) | **Models:** Switch Transformer, Mixtral 8x7B | **Benchmarks:** Perplexity, throughput, cache hit rate

**Key Results:** Up to 30% memory reduction with <1% perplexity degradation; improved cache hit rates over LRU.

**Pros (for project goal):**
- Sensitivity-based classification of experts aligns with the hot/cold expert concept in background.md
- Could be used for offline profiling to decide which experts to keep in NPU HBM vs. offload
- Provides a methodology for expert importance scoring without model retraining

**Cons (for project goal):**
- Expert gating (dropping activations) is a form of model behavior modification—contradicts the "no modification" requirement
- Targets datacenter GPU; edge/NPU specifics not considered
- Does not address the compute-I/O overlap problem
- Single-GPU only; three-tier hierarchy not considered

**URL:** [arXiv:2408.10284](https://arxiv.org/abs/2408.10284)

---

### 6. HOBBIT: A Mixed Precision Expert Offloading System for Fast MoE Inference

**Authors:** Peng Tang et al. (SJTU) | **Venue:** arXiv preprint | **Date:** 2024-11-03 | **Citations:** ~20 | **Importance:** ⭐⭐⭐⭐ (4/5)

**Core Technique:** Mixed-precision offloading where hot experts are kept in GPU memory at INT8/FP16 and cold experts are stored in CPU RAM at INT4. An expert activation predictor determines which experts will be needed for the next few tokens and asynchronously prefetches them. Combines quantization-based bandwidth reduction with async prefetch to hide I/O latency.

**Hardware:** Single GPU (RTX 3090, A100) + CPU RAM | **Models:** Mixtral 8x7B, DeepSeek-MoE | **Benchmarks:** Tok/s, memory, accuracy (MMLU)

**Key Results:** 1.8–2.5× speedup over DeepSpeed-MoE and Mixtral offloading baselines; expert-cache hit rate >85% with INT4 cold experts; minimal accuracy loss.

**Pros (for project goal):**
- Mixed-precision offloading (INT8 hot, INT4 cold) directly applicable to reduce PCIe transfer volume—critical for the 8 GB/s constraint
- Expert activation predictor pattern is implementable in llama.cpp
- No model modification required—works with standard checkpoints
- Quantized cold experts reduces SSD storage requirement too

**Cons (for project goal):**
- CUDA-based implementation; NPU portability requires re-implementation
- No three-tier hierarchy (GPU/CPU/SSD)
- Batch=1 latency not explicitly optimized; focuses on moderate-batch throughput
- Predictor accuracy varies by model; needs per-model calibration

**URL:** [arXiv:2411.01433](https://arxiv.org/abs/2411.01433)

---

### 7. ExpertFlow: Efficient Mixture-of-Experts Inference via Predictive Expert Caching and Token Scheduling

**Authors:** Xin He et al. (HKUST) | **Venue:** DAC 2026 | **Date:** 2024-10-23 | **Citations:** ~10 | **Importance:** ⭐⭐⭐⭐ (4/5)

**Core Technique:** Two-part system: (1) **Predictive expert cache** using a lightweight router prediction model to forecast which experts will be activated 2–4 layers ahead, enabling proactive prefetching; (2) **Token scheduling** that batches tokens by their predicted expert activation patterns to improve cache hit rates when batch_size > 1. Introduces a "token similarity" metric to group tokens that activate similar experts.

**Hardware:** GPUs (A100, RTX 3090) + CPU RAM | **Models:** Mixtral 8x7B, DeepSeek-MoE 16B | **Benchmarks:** Throughput (tok/s), latency, cache hit rate

**Key Results:** ~2× throughput improvement over MoE-Infinity; cache hit rate improved to 92% with predictive caching.

**Pros (for project goal):**
- Multi-layer lookahead prediction is implementable as an offline-calibrated predictor in llama.cpp
- Quantifies how far ahead one can predict (2–4 layers)—useful for designing the project's prefetch pipeline depth
- No model modification required

**Cons (for project goal):**
- Token batching component assumes batch_size > 1; project is batch=1 (single-user edge device)
- Router prediction requires a secondary small model—adds memory overhead in already-constrained NPU HBM
- GPU-centric; NPU-specific implementation details absent
- DAC 2026 paper may still have limited community validation

**URL:** [arXiv:2410.17954](https://arxiv.org/abs/2410.17954)

---

### 8. MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs

**Authors:** Shiyi Cao et al. (UC Berkeley) | **Venue:** arXiv preprint | **Date:** 2024-11-18 | **Citations:** ~35 | **Importance:** ⭐⭐⭐⭐ (4/5)

**Core Technique:** CGOPipe (CPU-GPU-I/O pipeline) with paged expert weights. Introduces the HRM (Hierarchical Roofline Model) to analytically determine the optimal scheduling policy across CPU compute, GPU compute, and I/O transfers. Uses paged weight management to reduce memory fragmentation. Explicitly models the three-way bottleneck between CPU compute, GPU compute, and PCIe bandwidth.

**Hardware:** Single/multi low-cost GPU (T4 16 GB, multiple T4s) + CPU RAM | **Models:** Mixtral 8x7B, Mixtral 8x22B, DBRX | **Benchmarks:** Throughput (tok/s), memory, comparison to prior offloading systems

**Key Results:** 10.3× higher throughput than state-of-the-art offloading systems for Mixtral 8x7B on a single T4 GPU. Reaches theoretical throughput upper bound. 2–3× less CPU memory than competing systems at same throughput.

**Pros (for project goal):**
- HRM analytical model directly applicable to the NPU/CPU/PCIe system—allows calculating optimal scheduling policy before implementation
- CGOPipe's three-way CPU/GPU/IO pipelining directly maps to NPU/CPU/PCIe pipeline design
- Demonstrates batch inference speedup; useful for context of prefill phase
- Berkeley group; well-validated with strong baselines

**Cons (for project goal):**
- Primarily optimizes batch throughput, not batch=1 latency—decode at tok/s target needs different approach
- T4 GPU ≠ NPU; transfer mechanisms differ (CUDA unified memory vs. PCIe DMA)
- Does not consider SSD as a third storage tier
- CPU compute in CGOPipe is not used; it's CPU RAM → GPU transfer only

**URL:** [arXiv:2411.11217](https://arxiv.org/abs/2411.11217)

---

### 9. Read-ME: Refactorizing LLMs as Router-Decoupled Mixture of Experts with System Co-Design

**Authors:** Ruisi Cai et al. (UT Austin, Qualcomm) | **Venue:** NeurIPS 2024 | **Date:** 2024-10-24 | **Citations:** ~25 | **Importance:** ⭐⭐⭐ (3/5)

**Core Technique:** Decouples the routing decision from the expert execution: a lightweight "pre-router" runs ahead of the main transformer layers to predict expert assignments. This enables asynchronous expert prefetching without waiting for the transformer to complete the routing. System co-design includes a hardware-aware batching strategy and prefetch scheduling.

**Hardware:** GPU cluster / single high-end GPU | **Models:** OPT-MoE, Mixtral 8x7B | **Benchmarks:** Latency, throughput, cache hit rate vs. standard MoE

**Key Results:** Router decoupling achieves >90% cache hit rate; significant latency reduction in constrained scenarios.

**Pros (for project goal):**
- Pre-router concept (predict expert assignment before full layer computation) is very powerful for NPU pipeline—allows async PCIe prefetch during NPU computation
- NeurIPS 2024—well-validated, peer-reviewed
- No full model modification required; pre-router is an add-on predictor

**Cons (for project goal):**
- Pre-router adds parameters and compute overhead—NPU's 10 GB HBM is tight
- Decoupled router changes the inference pipeline significantly; non-trivial to implement in llama.cpp
- Requires calibration data to train/initialize pre-router
- GPU-focused; NPU kernel mapping not addressed

**URL:** [arXiv:2410.19123](https://arxiv.org/abs/2410.19123)

---

### 10. DAOP: Data-Aware Offloading and Predictive Pre-Calculation for Efficient MoE Inference

**Authors:** Yujie Zhang, Shivam Aggarwal, Tulika Mitra (NUS) | **Venue:** DATE 2025 | **Date:** 2025-01-16 | **Citations:** ~15 | **Importance:** ⭐⭐⭐⭐ (4/5)

**Core Technique:** Combines two ideas: (1) **Data-aware offloading**—uses input data characteristics (token type, context pattern) to predict which experts will be hot vs. cold, enabling smarter placement decisions; (2) **Predictive pre-calculation**—starts computing partial expert outputs before expert selection is complete, hiding the routing latency. Targets single-GPU memory-constrained scenarios.

**Hardware:** GPU (A100, RTX 3090) + CPU RAM | **Models:** DeepSeek-MoE 16B, Mixtral 8x7B | **Benchmarks:** Latency, throughput, DATE conference evaluation

**Key Results:** Significant latency reduction vs. naïve offloading; DATE 2025 acceptance validates practical value.

**Pros (for project goal):**
- Input-aware expert prediction (rather than purely historical) could be valuable for longer context inference on edge
- Pre-calculation concept could partially overlap NPU attention computation with expert prefetch
- DATE 2025 venue provides hardware/system credibility
- No model modification required

**Cons (for project goal):**
- Data-aware predictor requires per-input feature extraction—adds overhead at batch=1
- Partial pre-calculation risks wasted computation if prediction is wrong
- NUS targets datacenter GPUs; edge/NPU-specific optimizations absent
- Does not address SSD offloading or three-tier hierarchy

**URL:** [arXiv:2501.10375](https://arxiv.org/abs/2501.10375)

---

### 11. Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading

**Authors:** Hanfei Yu et al. (multiple institutions) | **Venue:** arXiv preprint | **Date:** 2025-02-07 | **Citations:** ~10 | **Importance:** ⭐⭐⭐ (3/5)

**Core Technique:** Fine-grained expert offloading that treats individual expert sub-tensors (e.g., gate/up/down projection matrices) rather than whole experts as the granularity for caching and offloading. Allows partial expert residency—keeping only the "gate" projection in GPU while offloading "up/down" projections—enabling finer trade-off between latency and memory.

**Hardware:** GPU (A100, H100) + CPU RAM | **Models:** Mixtral 8x7B, DeepSeek-V2 | **Benchmarks:** Throughput, latency, memory footprint

**Key Results:** Better latency-memory Pareto curve than whole-expert offloading; achieves up to 1.5× better latency at same memory constraint.

**Pros (for project goal):**
- Sub-expert tensor granularity could be valuable: for Qwen3's fine-grained MoE, individual projection matrices are small enough to partially reside in NPU HBM
- No model modification required
- Useful for the "only 5–8 GB of NPU HBM for experts" constraint

**Cons (for project goal):**
- Fine-grained management increases runtime overhead (more PCIe transactions for sub-tensors)
- Complexity increases significantly for llama.cpp implementation
- Focused on serving (batch > 1); edge batch=1 optimization not primary focus
- Sub-tensor offloading may cause fragmented PCIe transfers—harmful for PCIe bandwidth utilization

**URL:** [arXiv:2502.05370](https://arxiv.org/abs/2502.05370)

---

### 12. Fate: Fast Edge Inference of Mixture-of-Experts Models via Cross-Layer Gate

**Authors:** Zhiyuan Fang et al. (Sun Yat-sen University) | **Venue:** arXiv preprint | **Date:** 2025-02-17 | **Citations:** ~8 | **Importance:** ⭐⭐⭐⭐ (4/5)

**Core Technique:** **Cross-layer gate** prediction: the gating decision for layer L+k is predicted at layer L using the current hidden states, enabling multi-step-ahead prefetching. The cross-layer gate is a lightweight linear predictor trained on calibration data (no retraining of the base model). Combines with a LRU cache and priority queue for expert prefetching on edge devices.

**Hardware:** Edge GPU (RTX 3070 Ti, 8 GB VRAM) + CPU RAM | **Models:** Mixtral 8x7B, DeepSeek-MoE | **Benchmarks:** Tok/s, latency, accuracy retention

**Key Results:** 1.6–2.3× speedup over vanilla offloading; ~90% cache hit rate with 3-layer lookahead; negligible accuracy degradation.

**Pros (for project goal):**
- Cross-layer gate directly enables asynchronous prefetch over PCIe—critical for the 8 GB/s bandwidth constraint
- Lightweight predictor fits within NPU HBM budget (linear layer, tiny parameters)
- No model modification; predictor is a plug-on addition
- Explicitly targets edge GPU with limited memory—closely analogous to NPU scenario
- Calibration-only approach (no retraining) aligns with background.md

**Cons (for project goal):**
- Linear predictor may have lower accuracy on very fine-grained MoE models (many small experts like Qwen3)
- Implemented on NVIDIA edge GPU; NPU adaptation requires kernel rewrite
- Does not consider SSD as a second offload tier
- The predictor requires calibration data collection

**URL:** [arXiv:2502.12224](https://arxiv.org/abs/2502.12224)

---

### 13. Klotski: Efficient Mixture-of-Expert Inference via Expert-Aware Multi-Batch Pipeline

**Authors:** Zhiyuan Fang et al. (Sun Yat-sen University) | **Venue:** arXiv preprint | **Date:** 2025-02-09 | **Citations:** ~5 | **Importance:** ⭐⭐⭐ (3/5)

**Core Technique:** For batch-size > 1, groups tokens by their predicted expert activations ("expert-aware batching"), then pipelines expert loading and computation such that experts needed by the current batch's tokens are fetched while the previous batch's tokens are computed. Introduces a "Klotski scheduler" that solves the assignment problem optimally given memory constraints.

**Hardware:** GPU + CPU RAM (multi-user serving) | **Models:** Mixtral 8x7B | **Benchmarks:** Throughput (tok/s), latency SLO compliance

**Key Results:** 1.5–2× throughput improvement over DeepSpeed and vLLM in batch serving scenarios.

**Pros (for project goal):**
- Expert-aware batching concept partially applicable if the project serves multiple concurrent requests
- Scheduling framework could inspire the prefetch planner design

**Cons (for project goal):**
- Primarily designed for batch > 1 serving; minimal benefit for batch=1 edge case
- Requires knowing all tokens' routing in advance (batch-level), impractical for streaming decode
- CPU compute not utilized—pure cache/transfer optimization
- Edge-device / NPU specifics not addressed

**URL:** [arXiv:2502.06888](https://arxiv.org/abs/2502.06888)

---

### 14. DAOP (see #10) | Skipping duplicate

---

### 14. HybriMoE: Hybrid CPU-GPU Scheduling and Cache Management for Efficient MoE Inference

**Authors:** Shuzhang Zhong et al. (Peking University) | **Venue:** DAC 2025 | **Date:** 2025-04-08 | **Citations:** ~20 | **Importance:** ⭐⭐⭐⭐⭐ (5/5)

**Core Technique:** Hybrid CPU-GPU inference that dynamically assigns each expert to run on CPU or GPU at runtime. Three innovations: (1) **Dynamic intra-layer scheduling**—each inference step solves a lightweight assignment problem to balance workload across CPU and GPU; (2) **Impact-driven inter-layer prefetching**—predicts future expert needs and prioritizes prefetching high-impact experts; (3) **Score-based caching**—a frequency+recency hybrid score determines eviction. Implemented on top of kTransformers framework.

**Hardware:** CPU (multi-core, AVX-512) + GPU (RTX 4090 24 GB, A100 80 GB) | **Models:** DeepSeek-V2, Mixtral 8x7B, Qwen2-MoE | **Benchmarks:** Prefill speedup, decode speedup vs. kTransformers baseline

**Key Results:** 1.33× prefill speedup, 1.70× decode speedup over state-of-the-art hybrid MoE inference. Code at: https://github.com/PKU-SEC-Lab/HybriMoE

**Pros (for project goal):**
- **Directly analogous** to the NPU+CPU hybrid mode in background.md: CPU computes warm experts, NPU computes hot experts—same architectural split
- Dynamic assignment at runtime handles the instability of expert activation patterns
- Score-based cache is implementable in llama.cpp
- Tests on Qwen2-MoE (predecessor to Qwen3)—highly relevant
- DAC 2025 peer-reviewed; open-source implementation available
- Impact-driven prefetching can be adapted for PCIe prefetch in NPU context

**Cons (for project goal):**
- CPU backend uses AVX-512 SIMD; performance depends heavily on CPU capabilities—for low-cost markets with weak CPUs this may not help
- GPU-centric; swapping GPU→NPU requires kernel re-implementation
- Does not address SSD offloading (CPU RAM is the only offload tier)
- Dynamic assignment adds per-step overhead which may hurt latency at batch=1

**URL:** [arXiv:2504.05897](https://arxiv.org/abs/2504.05897)

---

### 15. Not All Models Suit Expert Offloading: On Local Routing Consistency of Mixture-of-Expert Models

**Authors:** Jingcong Liang et al. (Fudan University) | **Venue:** arXiv preprint | **Date:** 2025-05-21 | **Citations:** ~5 | **Importance:** ⭐⭐⭐⭐ (4/5)

**Core Technique:** Empirical study measuring "local routing consistency" (LRC)—the degree to which the same experts are activated for nearby tokens in a sequence. High LRC means expert caches are effective; low LRC means offloading will have poor hit rates. Studies multiple MoE model families and quantifies which ones are suitable for expert offloading strategies.

**Hardware:** Analysis study; evaluated across multiple GPUs | **Models:** Mixtral 8x7B, DeepSeek-V2, DeepSeek-V3, Qwen2-MoE, Phi-MoE | **Benchmarks:** LRC metric, hit rate simulations, perplexity

**Key Results:** Models like DeepSeek-V2/V3 have high LRC (good for offloading); some models have low LRC and would not benefit. Qwen-MoE family shows moderate-to-high LRC.

**Pros (for project goal):**
- **Directly relevant**: tells us whether Qwen3-35B-A3B is suitable for expert caching/offloading strategies—answer appears to be yes for Qwen family
- Provides quantitative guidance on cache size needed for different hit rates
- Helps calibrate performance expectations before building the engine
- No implementation required; pure analysis study

**Cons (for project goal):**
- Qwen3 is not explicitly studied (only Qwen2-MoE); Qwen3's fine-grained architecture may differ
- Purely empirical; does not provide an optimization algorithm
- Does not address the three-tier storage hierarchy
- Findings may change with different quantization levels

**URL:** [arXiv:2505.16056](https://arxiv.org/abs/2505.16056)

---

### 16. FloE: On-the-Fly MoE Inference on Memory-constrained GPU

**Authors:** Yuxin Zhou et al. (Zhejiang University) | **Venue:** ICML 2025 | **Date:** 2025-05-09 | **Citations:** ~10 | **Importance:** ⭐⭐⭐⭐ (4/5)

**Core Technique:** "On-the-fly" offloading that avoids maintaining a persistent expert cache altogether. Instead, FloE uses streaming I/O: as each transformer layer executes, required experts are fetched just-in-time (JIT) from host memory, with careful overlap of DMA transfers and GPU computation. Introduces "speculative expert buffering" to read ahead by 0.5–1.0 layer. Eliminates cache management overhead and cold-start issues.

**Hardware:** Memory-constrained GPU (RTX 3080 10 GB, T4 16 GB) + CPU RAM | **Models:** Mixtral 8x7B, DeepSeek-V2-Lite | **Benchmarks:** Latency (ms), throughput (tok/s) vs. MoE-Infinity and DeepSpeed

**Key Results:** On ICML 2025; achieves competitive latency without cache management overhead; particularly good for first-token latency (TTFT).

**Pros (for project goal):**
- JIT streaming without caching could simplify llama.cpp integration
- Speculative buffering (0.5–1 layer ahead) maps to a small PCIe DMA queue, feasible within NPU design
- Particularly strong TTFT performance—directly addresses the <2s TTFT requirement
- ICML 2025 peer-reviewed

**Cons (for project goal):**
- No cache means every token pays full I/O cost; decode latency may not reach 25 tok/s at 8 GB/s PCIe
- Streaming I/O creates many small DMA transfers vs. larger batched transfers—may not suit PCIe 4.0 efficiency
- GPU-specific CUDA implementation; NPU DMA engine behavior may differ
- Weak on repeated expert activations—no reuse benefit

**URL:** [arXiv:2505.05950](https://arxiv.org/abs/2505.05950)

---

### 17. Cache Management for Mixture-of-Experts LLMs (Extended Version)

**Authors:** Spyros Angelopoulos et al. (multiple European institutions) | **Venue:** arXiv preprint | **Date:** 2025-09-02 | **Citations:** ~5 | **Importance:** ⭐⭐⭐ (3/5)

**Core Technique:** Formal analysis of expert cache management as an online algorithm problem. Derives theoretical bounds on cache hit rates under different replacement policies (OPT, LRU, FIFO, frequency-based). Proposes a workload-adaptive cache replacement algorithm based on online learning of expert activation patterns.

**Hardware:** Theoretical analysis + GPU validation | **Models:** Mixtral 8x7B | **Benchmarks:** Cache hit rate, miss penalty, theoretical competitive ratio

**Key Results:** Shows LRU is within 2× of optimal for typical MoE workloads; frequency-based replacement outperforms LRU; provides theoretical lower bounds on cache size for target hit rates.

**Pros (for project goal):**
- Provides theoretical guidance on minimum NPU HBM allocation needed for target hit rate with Qwen3-size model
- The workload-adaptive algorithm is lightweight and implementable in llama.cpp
- Theoretical bounds help set expectations for what's achievable with 5–8 GB expert cache

**Cons (for project goal):**
- Theoretical focus; limited practical system implementation details
- Does not consider multi-tier hierarchy (NPU/DRAM/SSD)
- Doesn't address prefetching—only replacement policy
- Model studied (Mixtral) differs from Qwen3 in expert count and activation patterns

**URL:** [arXiv:2509.02408](https://arxiv.org/abs/2509.02408)

---

### 18. LayerScope: Predictive Cross-Layer Scheduling for Efficient Multi-Batch MoE Inference on Legacy Servers

**Authors:** Enda Yu et al. (NUDT) | **Venue:** ICS 2026 | **Date:** 2025-09-28 | **Citations:** ~5 | **Importance:** ⭐⭐⭐ (3/5)

**Core Technique:** "PreScope" system that uses a prediction-driven expert scheduling approach addressing three key issues: (1) inaccurate expert prediction at layer boundaries; (2) PCIe transfer latency exceeding GPU computation; (3) multi-batch expert reuse. Introduces cross-layer scheduling windows where prefetch decisions are made 2–3 layers ahead using an online prediction model.

**Hardware:** Legacy server (1–2 GPUs, PCIe 3.0/4.0, CPU RAM) | **Models:** Mixtral 8x7B, DeepSeek-V2 | **Benchmarks:** Throughput, PCIe utilization, latency

**Key Results:** PCIe utilization improved to >85%; significant throughput gains in multi-batch scenarios with legacy PCIe bandwidth.

**Pros (for project goal):**
- Explicitly addresses PCIe transfer latency exceeding compute time—exactly the scenario at 8 GB/s PCIe 4.0 ×4 for Qwen3
- Cross-layer scheduling window concept directly implementable for NPU→CPU prefetch
- PCIe bandwidth utilization metric is directly relevant to the project's bottleneck analysis

**Cons (for project goal):**
- Multi-batch focus; single-request edge decoding is different
- Legacy server GPU target; NPU-specific considerations absent
- Online prediction model has warm-up latency before becoming accurate

**URL:** [arXiv:2509.23638](https://arxiv.org/abs/2509.23638)

---

### 19. DuoServe-MoE: Dual-Phase Expert Prefetch and Caching for LLM Inference QoS Assurance

**Authors:** Yuning Zhang et al. (University of Sydney) | **Venue:** arXiv preprint | **Date:** 2025-09-09 | **Citations:** ~3 | **Importance:** ⭐⭐⭐ (3/5)

**Core Technique:** Two-phase approach: (1) **Offline phase**—profiles workload to identify statically hot experts and permanently caches them; (2) **Online phase**—dynamically prefetches remaining experts based on per-request routing predictions. Designed for multi-tenant serving with SLO constraints.

**Hardware:** GPU serving cluster | **Models:** Mixtral 8x7B, DeepSeek-V2 | **Benchmarks:** SLO compliance, throughput, latency

**Key Results:** Improves SLO compliance under varying workloads; better than pure online-only or static-only caching strategies.

**Pros (for project goal):**
- Two-phase offline+online design aligns with background.md's allowance for offline profiling
- Permanent caching of hot experts in fast memory (NPU HBM) with dynamic prefetch of warm experts is the right architecture for the project
- QoS assurance concepts useful for TTFT guarantees

**Cons (for project goal):**
- Multi-tenant serving; edge single-user case is much simpler
- GPU-centric cluster architecture; NPU M.2 card is very different
- Offline profiling adds deployment complexity

**URL:** [arXiv:2509.07379](https://arxiv.org/abs/2509.07379)

---

### 20. Accelerating Mixture-of-Expert Inference with Adaptive Expert Split Mechanism

**Authors:** Jiaming Yan et al. (USTC) | **Venue:** arXiv preprint | **Date:** 2025-09-10 | **Citations:** ~3 | **Importance:** ⭐⭐⭐ (3/5)

**Core Technique:** Dynamically splits large experts into smaller sub-experts to improve memory efficiency and enable finer-grained caching. The split granularity is adapted based on available memory and access frequency. Allows partial expert residency at a finer granularity than whole experts.

**Hardware:** GPU + CPU RAM | **Models:** Mixtral 8x7B, Switch Transformer | **Benchmarks:** Memory, throughput

**Key Results:** 1.3–1.6× throughput improvement with adaptive splitting vs. fixed-granularity offloading.

**Pros (for project goal):**
- Sub-expert granularity is relevant for Qwen3's fine-grained MoE where experts are already small
- Adaptive granularity based on available memory maps well to NPU HBM (5–8 GB budget)

**Cons (for project goal):**
- Expert splitting modifies the model's runtime structure—borderline model modification
- More complex memory management than whole-expert caching; harder to implement in llama.cpp
- Splitting overhead may be significant at the small expert sizes in Qwen3

**URL:** [arXiv:2509.08342](https://arxiv.org/abs/2509.08342)

---

### 21. SpecMoEOff: Accelerating Mixture-of-Experts Inference by Hiding Offloading Latency with Speculative Decoding

**Authors:** Zhibin Wang et al. (NJU) | **Venue:** arXiv preprint | **Date:** 2025-08-29 | **Citations:** ~5 | **Importance:** ⭐⭐⭐⭐ (4/5)

**Core Technique:** Combines speculative decoding with MoE offloading. The draft model generates multiple candidate tokens; the target MoE model verifies them in a single forward pass, amortizing the I/O cost across multiple tokens. Introduces a CPU chunked-attention verification kernel for PCIe-offloaded scenarios and an automatic hyperparameter optimizer (speculation width, depth) based on roofline analysis.

**Hardware:** GPU (A100, RTX 3090) + CPU RAM, PCIe | **Models:** Mixtral 8x7B, DeepSeek-V2 | **Benchmarks:** Decode throughput (tok/s), latency

**Key Results:** Up to 2.5× decode throughput improvement over state-of-the-art MoE offloading (MoE-Infinity baseline).

**Pros (for project goal):**
- Speculative decoding can dramatically increase expert reuse per PCIe transfer—critical for 8 GB/s bandwidth
- Auto-tuning of speculation depth based on hardware roofline directly applicable to NPU+PCIe configuration
- No model modification required (draft model is a separate smaller model)
- CPU chunked attention kernel concept applicable when using CPU for prefill in performance mode

**Cons (for project goal):**
- Speculative decoding requires a separate draft model—additional memory pressure on NPU HBM
- Draft model acceptance rate varies by task; gains are workload-dependent
- Speculative decoding adds implementation complexity to llama.cpp
- Throughput-focused; per-token latency (which determines decode speed) may not improve proportionally

**URL:** [arXiv:2508.21706](https://arxiv.org/abs/2508.21706)

---

### 22. CoMoE: Collaborative Optimization of Expert Aggregation and Offloading for MoE-based LLMs at Edge

**Authors:** Muqing Li et al. (HKUST) | **Venue:** arXiv preprint | **Date:** 2025-08-10 | **Citations:** ~3 | **Importance:** ⭐⭐⭐ (3/5)

**Core Technique:** Co-optimizes two dimensions: (1) **Expert aggregation**—merging similar experts to reduce total expert count and memory footprint; (2) **Expert offloading**—intelligent placement of merged experts across CPU/GPU memory tiers. The merging is done as post-training compression, creating a smaller but semantically equivalent expert set.

**Hardware:** Edge device (GPU + CPU RAM) | **Models:** Mixtral 8x7B, DeepSeek-MoE | **Benchmarks:** Memory reduction, accuracy, throughput

**Key Results:** 30–40% reduction in expert count with minimal accuracy loss; combined with offloading achieves better latency than offloading alone.

**Pros (for project goal):**
- Expert merging to reduce total parameter count is useful for fitting within NPU HBM + PCIe budget
- Edge-focused—closest to the project's deployment scenario

**Cons (for project goal):**
- Expert merging modifies the model's structure—contradicts the "no model modification" requirement
- Accuracy impact of merging may vary significantly by model (Qwen3 not studied)
- Offline merging adds deployment complexity

**URL:** [arXiv:2508.09208](https://arxiv.org/abs/2508.09208)

---

### 23. SSD Offloading for LLM Mixture-of-Experts Weights Considered Harmful in Energy Efficiency

**Authors:** Kwanhee Kyung, Sungmin Yun, Jung Ho Ahn (Seoul National University) | **Venue:** IEEE Computer Architecture Letters 2025 | **Date:** 2025-08-09 | **Citations:** ~5 | **Importance:** ⭐⭐⭐⭐ (4/5)

**Core Technique:** Critical analysis measuring the energy efficiency of SSD-based expert offloading. Shows that while SSD offloading enables running models that wouldn't fit otherwise, the energy cost per token is dramatically higher than GPU-only inference. Quantifies the energy penalty from repeated SSD read-modify-write cycles and provides analytical models.

**Hardware:** GPU + PCIe SSD | **Models:** Mixtral 8x7B, DeepSeek-V2 | **Benchmarks:** Energy per token (J/tok), latency, performance/watt

**Key Results:** SSD-offloaded MoE inference uses 3–10× more energy per token than GPU-only inference; recommends DRAM-offload as a preferred alternative when possible.

**Pros (for project goal):**
- **Critical design input**: for the cost-effective product variant (SSD offload), this paper quantifies the energy penalty—project should be aware users in this market tier will have higher power consumption
- Directly addresses the SSD offloading tier that the project must support
- IEEE CAL venue; credible and concise

**Cons (for project goal):**
- Primarily an energy analysis, not a performance optimization
- Does not propose solutions—only identifies the problem
- The "harmful" framing may overstate the issue; for a 15W NPU, total system power budget matters more
- Does not consider the SSD → CPU DRAM → NPU pipeline with caching at each tier

**URL:** [arXiv:2508.06978](https://arxiv.org/abs/2508.06978)

---

### 24. SP-MoE: Speculative Decoding and Prefetching for Accelerating MoE-based Model Inference

**Authors:** Liangkun Chen et al. (HKU) | **Venue:** arXiv preprint | **Date:** 2025-10-11 | **Citations:** ~5 | **Importance:** ⭐⭐⭐⭐ (4/5)

**Core Technique:** Combines speculative decoding (SD) with expert prefetching in a unified pipeline. The draft model serves dual purpose: (1) generating speculative tokens for verification; (2) providing expert routing predictions for prefetching of the verifier model's experts. This is more efficient than using separate predictors for SD and prefetching.

**Hardware:** GPU (A100, RTX 3090) + CPU RAM | **Models:** Mixtral 8x7B, DeepSeek-V2 | **Benchmarks:** Speedup, acceptance rate, memory

**Key Results:** 1.8–2.4× speedup over MoE-Infinity; unified SD+prefetch predictor reuses computation.

**Pros (for project goal):**
- Reusing draft model for both SD and expert prediction reduces memory overhead—important for NPU HBM budget
- Orthogonal to quantization and caching improvements
- No model modification required

**Cons (for project goal):**
- Draft model acceptance rate is variable; gains not guaranteed
- Requires a suitable draft model for Qwen3 (a smaller Qwen model exists but adds complexity)
- Speculative decoding adds verification overhead that can hurt latency in fast hardware

**URL:** [arXiv:2510.10302](https://arxiv.org/abs/2510.10302)

---

### 25. MoBiLE: Efficient MoE Inference on Consumer GPU with Mixture of Big Little Experts

**Authors:** Yushu Zhao et al. (Tsinghua University) | **Venue:** ASP-DAC 2026 | **Date:** 2025-10-14 | **Citations:** ~5 | **Importance:** ⭐⭐⭐ (3/5)

**Core Technique:** Trains a two-tier expert system: "big" high-capacity experts (normal size) + "little" low-capacity compressed experts. During inference, if a big expert would need to be offloaded, the system falls back to the always-resident little expert. Eliminates offloading latency for infrequently-accessed big experts.

**Hardware:** Consumer GPU (RTX 3090, RTX 4090) | **Models:** Custom-trained MoE models based on Mixtral | **Benchmarks:** Accuracy, latency, memory

**Key Results:** 1.5–2× latency reduction with minimal accuracy loss; eliminates tail latency from cold expert accesses.

**Pros (for project goal):**
- Concept of always-resident fallback experts could inspire a "shadow expert" approach for Qwen3
- Eliminates worst-case offload latency for rare expert activations

**Cons (for project goal):**
- Requires training the "little" experts—significant model modification and retraining required
- Incompatible with the "no model modification, pull from HuggingFace" requirement
- Custom trained models only; off-the-shelf Qwen3 cannot use this

**URL:** [arXiv:2510.12357](https://arxiv.org/abs/2510.12357)

---

### 26. BuddyMoE: Exploiting Expert Redundancy to Accelerate Memory-Constrained MoE Inference

**Authors:** Yun Wang et al. (Microsoft Research) | **Venue:** arXiv preprint | **Date:** 2025-11-13 | **Citations:** ~5 | **Importance:** ⭐⭐⭐ (3/5)

**Core Technique:** Identifies that many experts in a MoE model produce similar outputs for most inputs ("expert redundancy"). BuddyMoE builds a "buddy expert" mapping: when the needed expert is not in cache, use its most similar cached buddy instead. This reduces cache miss penalties without perfect prediction.

**Hardware:** GPU + CPU RAM | **Models:** Mixtral 8x7B, Phi-MoE | **Benchmarks:** Throughput, accuracy, cache miss recovery

**Key Results:** Expert substitution recovers ~70% of cache miss penalty with <1% accuracy drop; throughput improved over baseline offloading.

**Pros (for project goal):**
- Expert substitution when cache misses occur could reduce worst-case latency spikes
- Works without model modification—only post-hoc similarity analysis needed

**Cons (for project goal):**
- Substituting similar but not identical experts changes model outputs—questionable "no modification" compliance
- Expert redundancy may be model-specific; Qwen3's fine-grained experts may have less redundancy
- The similarity computation requires an additional profiling step

**URL:** [arXiv:2511.10054](https://arxiv.org/abs/2511.10054)

---

### 27. Pre-Attention Expert Prediction and Prefetching for Mixture-of-Experts Large Language Models

**Authors:** Shien Zhu et al. (ETH Zurich) | **Venue:** arXiv preprint | **Date:** 2025-11-10 | **Citations:** ~5 | **Importance:** ⭐⭐⭐⭐⭐ (5/5)

**Core Technique:** Inserts a lightweight "pre-attention predictor" before the multi-head attention computation in each transformer block. This predictor sees the current token's early representation and forecasts which experts will be activated in the MoE sublayer of the same block and subsequent blocks. The prediction is made early enough to begin PCIe prefetch before attention completes, overlapping attention computation with expert loading.

**Hardware:** GPU (A100) + CPU RAM | **Models:** Mixtral 8x7B, DeepSeek-V2 | **Benchmarks:** Hit rate, latency, accuracy

**Key Results:** Pre-attention prediction achieves ~85–90% hit rate; significant latency reduction by overlapping attention with expert prefetch. No accuracy loss with high-accuracy predictor.

**Pros (for project goal):**
- **Highly relevant**: NPU attention compute time (~attention QKV projections) can overlap with PCIe DMA for expert weights—this paper quantifies the gain
- Pre-attention predictor is tiny (linear layer) and fits in NPU HBM with minimal overhead
- No model modification; predictor is an inference-time add-on
- ETH Zurich implementation quality typically high

**Cons (for project goal):**
- Predictor accuracy is ~85–90%; residual misses still pay full PCIe latency
- Requires integration into the transformer's attention-MoE pipeline—non-trivial llama.cpp change
- For Qwen3's very fine-grained MoE (many small experts), prediction accuracy may vary

**URL:** [arXiv:2511.10676](https://arxiv.org/abs/2511.10676)

---

### 28. MoE-SpeQ: Speculative Quantized Decoding with Proactive Expert Prefetching and Offloading

**Authors:** Wenfeng Wang et al. (SJTU) | **Venue:** arXiv preprint | **Date:** 2025-11-18 | **Citations:** ~5 | **Importance:** ⭐⭐⭐⭐ (4/5)

**Core Technique:** Co-design of speculative decoding and expert offloading. Uses a draft model to predict both future tokens AND future expert sequences, enabling proactive prefetching of experts for future tokens. Introduces an "Amortization Roofline Model" to adaptively tune speculation depth based on hardware bandwidth-compute ratio. Specifically targets PCIe-bandwidth-bottlenecked scenarios.

**Hardware:** Memory-constrained device (RTX 3090, T4) + CPU RAM, PCIe | **Models:** Phi-MoE, Mixtral 8x7B | **Benchmarks:** Speedup over state-of-the-art offloading

**Key Results:** Up to 2.34× speedup over best offloading baseline. Amortization Roofline Model provides principled tuning.

**Pros (for project goal):**
- **Directly addresses PCIe I/O bottleneck** with hardware-aware roofline model—exactly the project's constraint
- Proactive expert prefetching using speculation aligns with the "predict future experts" design principle
- Amortization Roofline Model could be calibrated for the NPU+PCIe 4.0 ×4 configuration
- No model modification of base model (draft model is separate)

**Cons (for project goal):**
- Requires a draft model for Qwen3—adds memory pressure on NPU HBM
- Speculation acceptance rate is task-dependent; not guaranteed for all use cases
- Phi-MoE and Mixtral studied, not Qwen3 specifically
- PCIe bottleneck model needs re-calibration for NPU vs. GPU compute throughput differences

**URL:** [arXiv:2511.14102](https://arxiv.org/abs/2511.14102)

---

### 29. Dynamic Expert Quantization for Scalable Mixture-of-Experts Inference

**Authors:** Kexin Chu et al. (Alibaba DAMO) | **Venue:** arXiv preprint | **Date:** 2025-11-18 | **Citations:** ~5 | **Importance:** ⭐⭐⭐ (3/5)

**Core Technique:** Dynamically adjusts the quantization precision of each expert at inference time based on its activation frequency and importance. Hot experts use FP16/INT8; cold experts are compressed to INT2 or INT1. A lightweight runtime controller makes per-expert per-step precision decisions.

**Hardware:** GPU (H100, A100) | **Models:** DeepSeek-V3, Mixtral 8x22B | **Benchmarks:** Accuracy (various NLP tasks), throughput, memory

**Key Results:** 2–4× memory reduction with acceptable accuracy; dynamic quantization outperforms static across workload variations.

**Pros (for project goal):**
- Dynamic quantization reduces transfer volume for cold experts over PCIe—directly reduces 8 GB/s bottleneck
- Frequency-based precision assignment is implementable without model retraining
- INT2 cold experts could reduce SSD storage requirements significantly

**Cons (for project goal):**
- Runtime precision switching adds compute overhead; NPU INT2 kernels may not be well-optimized
- Very low precision (INT2/INT1) may cause unacceptable accuracy loss on Qwen3 reasoning tasks
- GPU-centric kernels; NPU INT2 support needs verification
- High variance in quality depending on task type

**URL:** [arXiv:2511.15015](https://arxiv.org/abs/2511.15015)

---

### 30. Bandwidth-Efficient Adaptive Mixture-of-Experts via Low-Rank Compensation

**Authors:** Zhenyu Liu et al. (multiple US institutions) | **Venue:** arXiv preprint | **Date:** 2025-12-18 | **Citations:** ~5 | **Importance:** ⭐⭐⭐ (3/5)

**Core Technique:** For experts that are offloaded to CPU/SSD, stores only the low-rank compressed version in device memory and the full-rank version off-device. When an expert is accessed, computes the low-rank approximation immediately (cheap) and asynchronously fetches the residual compensation term (full expert minus low-rank). The compensation fetch is smaller, reducing bandwidth requirements.

**Hardware:** GPU + CPU RAM | **Models:** Mixtral 8x7B, DeepSeek-V2 | **Benchmarks:** Accuracy, bandwidth, latency

**Key Results:** 30–50% bandwidth reduction with small accuracy loss; adaptive rank selection per expert.

**Pros (for project goal):**
- Low-rank residual compression could reduce PCIe transfer volume significantly
- The approach is essentially a form of quantization that's expert-aware and content-adaptive
- No model retraining needed (offline SVD decomposition)

**Cons (for project goal):**
- The low-rank approximation introduces accuracy loss that may be unacceptable
- Adds two-stage fetch complexity (synchronous low-rank + async residual)
- SVD decomposition modifies the expert matrices—technically a form of model modification
- Complex implementation in llama.cpp

**URL:** [arXiv:2512.17073](https://arxiv.org/abs/2512.17073)

---

### 31. SliceMoE: Bit-Sliced Expert Caching under Miss-Rate Constraints for Efficient MoE Inference

**Authors:** Yuseon Choi et al. (KAIST) | **Venue:** DAC 2026 | **Date:** 2025-12-15 | **Citations:** ~3 | **Importance:** ⭐⭐⭐⭐ (4/5)

**Core Technique:** Stores expert weights in a "bit-sliced" format—each expert weight matrix is decomposed into multiple 1-bit planes. The device can hold more or fewer bit-planes of each expert in cache, enabling a continuous precision spectrum rather than binary in-cache/out-of-cache decisions. Under a miss-rate constraint, SliceMoE optimizes the bit allocation per expert to maximize accuracy.

**Hardware:** GPU memory-constrained scenario | **Models:** Mixtral 8x7B, DeepSeek-V2 | **Benchmarks:** Accuracy, cache hit rate, throughput

**Key Results:** Better accuracy-miss-rate trade-off than fixed precision; more flexible cache utilization.

**Pros (for project goal):**
- Bit-sliced storage enables partial expert residency in NPU HBM at mixed precision—very flexible for the 5–8 GB HBM budget
- Miss-rate constraint formulation directly maps to "achieve X% hit rate within Y GB of NPU HBM"
- DAC 2026 peer-reviewed hardware system paper

**Cons (for project goal):**
- Bit-sliced format requires custom kernel implementations; NPU may not support bit-level operations efficiently
- Complex memory management; difficult to implement in llama.cpp cleanly
- The precision interpolation between bit-planes adds computational overhead

**URL:** [arXiv:2512.12990](https://arxiv.org/abs/2512.12990)

---

### 32. Context-Aware Mixture-of-Experts Inference on CXL-Enabled GPU-NDP Systems

**Authors:** Zehao Fan et al. (RPI) | **Venue:** arXiv preprint | **Date:** 2025-12-04 | **Citations:** ~3 | **Importance:** ⭐⭐⭐ (3/5)

**Core Technique:** Uses CXL (Compute Express Link) memory expansion to provide a larger memory pool for expert storage, with NDP (Near-Data Processing) units on the CXL device to execute cold experts locally. Context-aware scheduling assigns experts to GPU, NDP-CXL, or CPU based on predicted activation patterns and current hardware utilization.

**Hardware:** GPU + CXL-attached memory device with NDP compute | **Models:** Mixtral 8x7B | **Benchmarks:** Throughput, latency vs. CPU offloading

**Key Results:** 2–3× better performance than CPU offloading by leveraging NDP compute.

**Pros (for project goal):**
- CXL NDP concept maps loosely to the NPU-as-NDP in the target system—NPU could be seen as a smart accelerator near its HBM
- Context-aware scheduling across tiers is the right architecture pattern

**Cons (for project goal):**
- Requires CXL hardware not present in the target M.2 card + mini-PC environment
- The NPU is not connected via CXL but PCIe; bandwidth model differs
- CXL NDP is specialized hardware; mainstream availability limited

**URL:** [arXiv:2512.04476](https://arxiv.org/abs/2512.04476)

---

### 33. OD-MoE: On-Demand Expert Loading for Cacheless Edge-Distributed MoE Inference

**Authors:** Liujianfu Wang et al. (CUHK) | **Venue:** arXiv preprint | **Date:** 2025-12-03 | **Citations:** ~3 | **Importance:** ⭐⭐⭐ (3/5)

**Core Technique:** Distributes expert weights across multiple edge devices (IoT/mobile) and loads them on-demand over wireless networks. Eliminates local expert caching by routing requests to the device holding the required expert. Uses mobility prediction and network topology awareness for prefetching.

**Hardware:** Edge device cluster (mobile phones, IoT devices) with wireless connectivity | **Models:** Mixtral 8x7B (pruned versions) | **Benchmarks:** End-to-end latency, network bandwidth usage

**Key Results:** Enables MoE inference on devices with <2 GB RAM by distributing expert weights across devices; latency competitive with local cached approaches for high-connectivity scenarios.

**Pros (for project goal):**
- Distributed architecture concept could inspire a "NPU cache + SSD + DRAM" view as a distributed local storage system
- Demonstrates edge-scale deployability of MoE

**Cons (for project goal):**
- Wireless distributed inference is completely different from the target PCIe M.2 card scenario
- Network latency (ms) is far worse than local PCIe (μs); not directly applicable
- IoT device class; much weaker than target hardware

**URL:** [arXiv:2512.03927](https://arxiv.org/abs/2512.03927)

---

### 34. FlashMoE: Reducing SSD I/O Bottlenecks via ML-Based Cache Replacement for MoE Inference on Edge Devices

**Authors:** Byeongju Kim et al. (KAIST) | **Venue:** arXiv preprint | **Date:** 2026-01-22 | **Citations:** ~3 | **Importance:** ⭐⭐⭐⭐⭐ (5/5)

**Core Technique:** ML-based cache replacement policy specifically designed for SSD-offloaded MoE inference. Trains a lightweight LSTM/GRU model to predict future expert access patterns from the history of activations, then uses these predictions to make smarter eviction decisions. Explicitly targets NVMe SSD as the cold expert storage tier with an edge device as the main processor.

**Hardware:** Edge device (embedded GPU or CPU) + NVMe SSD | **Models:** Mixtral 8x7B, DeepSeek-MoE | **Benchmarks:** Cache hit rate, SSD I/O reduction, end-to-end latency

**Key Results:** 20–35% higher cache hit rate than LRU for SSD-offloaded workloads; SSD read I/O reduced by 30–40%; significant latency improvement at the edge.

**Pros (for project goal):**
- **Directly addresses SSD offloading on edge devices**—the cost-effective product variant of the project
- ML-based predictor is more accurate than LRU for SSD-heavy access patterns
- No model modification required; purely inference-system optimization
- Edge device target closely matches the M.2 card deployment scenario
- Explicitly analyzes the bandwidth gap between NVMe SSD and device memory (same constraint as PCIe 4.0 ×4)

**Cons (for project goal):**
- ML predictor requires training on inference traces—adds offline calibration complexity
- LSTM model adds memory overhead; needs to fit within NPU HBM budget
- Cold-start problem: predictor must warm up before becoming accurate
- Targets embedded GPU rather than custom NPU; kernel portability unclear

**URL:** [arXiv:2601.17063](https://arxiv.org/abs/2601.17063)

---

### 35. A Scheduling Framework for Efficient MoE Inference on Edge GPU-NDP Systems

**Authors:** Qi Wu et al. (Southeast University) | **Venue:** DATE 2026 | **Date:** 2026-01-07 | **Citations:** ~2 | **Importance:** ⭐⭐⭐⭐ (4/5)

**Core Technique:** Systematic scheduling framework for MoE inference on edge devices with GPU-NDP (Near-Data Processing) architecture. Maps expert placement to memory tiers (on-GPU, GPU-NDP, host) based on activation frequency and data movement cost. Introduces a scheduling algorithm that minimizes end-to-end latency by jointly optimizing expert placement and execution order.

**Hardware:** Edge GPU + NDP unit + host memory | **Models:** Mixtral 8x7B, DeepSeek-V2-Lite | **Benchmarks:** Latency, throughput, DATE 2026 evaluation

**Key Results:** Up to 2.1× latency reduction over non-scheduled baseline; framework generalizes across hardware configurations.

**Pros (for project goal):**
- **Directly targets edge GPU-NDP**—highly analogous to NPU (HBM) + CPU (DRAM) architecture
- Systematic scheduling framework is exactly what the project needs for the two memory tiers
- DATE 2026 (hardware) venue validates practical implementation
- Joint optimization of placement and execution order is the right approach for the project

**Cons (for project goal):**
- NDP hardware differs from standard CPU—NPU-specific compute characteristics not perfectly matched
- Does not address SSD as a third tier
- Scheduling overhead may be too high for real-time batch=1 inference

**URL:** [arXiv:2601.03992](https://arxiv.org/abs/2601.03992)

---

### 36. DALI: A Workload-Aware Offloading Framework for Efficient MoE Inference on Local PCs

**Authors:** Zeyu Zhu et al. (ICT, Chinese Academy of Sciences) | **Venue:** arXiv preprint | **Date:** 2026-02-03 | **Citations:** ~3 | **Importance:** ⭐⭐⭐⭐⭐ (5/5)

**Core Technique:** Three-part framework: (1) **Greedy Dynamic Assignment**—solves expert-to-CPU/GPU assignment as a 0-1 optimization problem at each inference step, adapting to dynamic activation patterns; (2) **Residual-Based Prefetching**—uses inter-layer residual connections to predict high-workload experts more accurately than naive routing prediction; (3) **Workload-Aware Cache Replacement**—exploits temporal correlations in expert activations for better eviction decisions. Targets local PC (GPU + CPU RAM) scenarios.

**Hardware:** Local PC (RTX 4090, A100) + CPU RAM | **Models:** Mixtral 8x7B, DeepSeek-V2, DeepSeek-V3 | **Benchmarks:** Prefill throughput, decode throughput (tok/s), cache hit rate

**Key Results:** Significant speedups in both prefill and decode phases over MoE-Infinity, HybriMoE, and kTransformers baselines. Best-in-class workload-aware cache replacement.

**Pros (for project goal):**
- Residual-based prefetch predictor is novel and more accurate than attention-based predictors—important for Qwen3's many experts
- Dynamic CPU/GPU assignment is the right model for the NPU/CPU hybrid design
- Local PC scenario (single user, consumer hardware) is analogous to M.2 card + mini-PC
- No model modification required
- Tests on DeepSeek-V3—most recently-relevant large MoE model

**Cons (for project goal):**
- GPU-centric (CPU and GPU are both x86-based); NPU kernel porting needed
- 0-1 optimization solver at each step adds latency overhead
- Does not address SSD as a third tier (CPU RAM is the only offload storage)
- DeepSeek models studied; Qwen3's different expert structure may require re-tuning

**URL:** [arXiv:2602.03495](https://arxiv.org/abs/2602.03495)

---

### 37. PROBE: Co-Balancing Computation and Communication in MoE Inference via Real-Time Predictive Prefetching

**Authors:** Qianchao Zhu et al. (ByteDance) | **Venue:** arXiv preprint | **Date:** 2026-01-30 | **Citations:** ~2 | **Importance:** ⭐⭐⭐⭐ (4/5)

**Core Technique:** Real-time predictor that co-balances compute and communication (PCIe transfer) by dynamically adjusting the prefetch window size based on observed hardware utilization. Uses a lightweight online learning algorithm that continuously adapts to workload changes. Introduces "predictive prefetch slack"—the optimal timing gap between prediction and actual expert use.

**Hardware:** Multi-GPU server (for tensor parallelism), PCIe interconnect | **Models:** DeepSeek-V3, Mixtral 8x22B | **Benchmarks:** Throughput, PCIe utilization, latency

**Key Results:** >85% PCIe utilization; throughput improvements by reducing idle periods in the compute-communication pipeline.

**Pros (for project goal):**
- Real-time adaptive prefetch window is directly applicable to the NPU+PCIe scenario
- Co-balancing compute/communication is the exact problem when NPU compute time ≈ PCIe transfer time
- ByteDance production system quality; practical implementation insights

**Cons (for project goal):**
- Multi-GPU tensor parallelism context; single-NPU scenario is different
- Communication here is GPU-to-GPU over NVLink/PCIe; differs from CPU→NPU PCIe DMA
- Online learning has warm-up period before optimal performance

**URL:** [arXiv:2602.00509](https://arxiv.org/abs/2602.00509)

---

### 38. TriMoE: Augmenting GPU with AMX-Enabled CPU and DIMM-NDP for High-Throughput MoE Inference via Offloading

**Authors:** Yudong Pan et al. (ICT, Chinese Academy of Sciences) | **Venue:** DAC 2026 | **Date:** 2026-03-01 | **Citations:** ~2 | **Importance:** ⭐⭐⭐⭐⭐ (5/5)

**Core Technique:** Three-way compute architecture: GPU handles hot experts, AMX-enabled CPU (Intel Advanced Matrix Extensions) handles warm experts, and DIMM-NDP handles cold experts. Introduces a **hot/warm/cold expert classification** scheme based on activation frequency, and a **bottleneck-aware scheduling policy** that maps each tier to its optimal expert set. A prediction-driven **dynamic relayout/rebalancing** scheme redistributes experts across tiers as workload patterns change.

**Hardware:** GPU (A100/H100) + AMX-enabled CPU (Intel Xeon) + DIMM-NDP | **Models:** Mixtral 8x7B, DeepSeek-V2, Llama-MoE | **Benchmarks:** Throughput (tok/s), latency vs. GPU-only and GPU+CPU baselines

**Key Results:** Up to 2.83× speedup over state-of-the-art (GPU+CPU only) solutions. Hot/warm/cold three-tier classification is key enabler.

**Pros (for project goal):**
- **Most directly analogous architecture**: NPU HBM (hot) → CPU DRAM (warm) → SSD (cold) maps exactly to GPU VRAM (hot) → CPU DRAM (warm) → DIMM-NDP (cold). Three-tier classification is the right model.
- Hot/warm/cold expert scheduling is exactly what background.md describes
- Bottleneck-aware scheduling policy avoids over-/under-loading any tier
- DAC 2026 peer-reviewed; CAS institution with high implementation quality
- AMX-enabled CPU execution → directly applicable to performance-oriented variant (strong CPU + NPU)

**Cons (for project goal):**
- DIMM-NDP is specialized hardware not present in the target system (SSD replaces DIMM-NDP)
- AMX-specific CPU instructions; performance degrades on non-AMX CPUs (weak CPU scenario)
- GPU → NPU porting requires re-implementation
- Dynamic relayout adds runtime complexity

**URL:** [arXiv:2603.01058](https://arxiv.org/abs/2603.01058)

---

### 39. DyMoE: Dynamic Expert Orchestration with Mixed-Precision Quantization for Efficient MoE Inference on Edge

**Authors:** Yuegui Huang et al. (Sun Yat-sen University) | **Venue:** arXiv preprint | **Date:** 2026-03-19 | **Citations:** ~1 | **Importance:** ⭐⭐⭐⭐ (4/5)

**Core Technique:** Dynamic orchestration combining: (1) **Mixed-precision quantization** (INT4/INT8/FP16) assigned dynamically per-expert based on activation importance; (2) **I/O-compute overlap** using predicted next-layer expert demands to pipeline transfers; (3) **Edge-optimized kernels** for ARM/embedded processors. The orchestrator runs a lightweight decision model that observes current system state and makes placement/precision decisions in real time.

**Hardware:** Edge device (embedded processor, ARM, GPU-less scenarios) + DRAM/Flash | **Models:** Mixtral 8x7B, DeepSeek-MoE | **Benchmarks:** Edge latency, memory, throughput

**Key Results:** Efficient edge deployment with significantly reduced latency vs. static precision approaches; I/O overlap recovers most of the transfer latency.

**Pros (for project goal):**
- Mixed-precision + I/O overlap is exactly the right combination for the NPU (INT8/INT4 compute) + PCIe pipeline
- Explicitly targets edge devices—most hardware-relevant scenario
- Dynamic orchestration without model modification
- ARM-optimized kernels relevant for the CPU side in mini-PC deployment

**Cons (for project goal):**
- ARM-centric kernels; NPU kernel porting needed
- Very recent paper (Mar 2026); limited citation/validation
- No SSD tier; DRAM/Flash only

**URL:** [arXiv:2603.19172](https://arxiv.org/abs/2603.19172)

---

### 40. Speculating Experts Accelerates Inference for Mixture-of-Experts

**Authors:** Vivan Madan et al. (University of Maryland) | **Venue:** arXiv preprint | **Date:** 2026-03-09 | **Citations:** ~2 | **Importance:** ⭐⭐⭐⭐ (4/5)

**Core Technique:** Formulates expert prediction as a speculative execution problem. A fast "draft oracle" predicts which experts will be activated, initiates their prefetch, and speculatively begins computing with them. The actual router output either confirms (computation accepted) or contradicts (rollback and recompute with correct expert) the speculation. The acceptance rate determines the performance gain.

**Hardware:** Memory-constrained GPU + CPU RAM | **Models:** Mixtral 8x7B, DeepSeek-V2-Lite | **Benchmarks:** Effective throughput (tok/s accounting for rollbacks), acceptance rate, latency

**Key Results:** Acceptance rates of 85–95% on typical NLP tasks; 1.5–2× throughput improvement over non-speculative baselines.

**Pros (for project goal):**
- High acceptance rate means most expert prefetches are useful—good PCIe utilization
- Speculation + rollback is conceptually simpler than training a separate SD draft model
- No model modification required
- Applicable to batch=1 latency scenarios

**Cons (for project goal):**
- Rollback overhead can be significant if acceptance rate drops (e.g., for reasoning tasks with unusual token sequences)
- Rollback requires recomputing with correct expert—doubles worst-case per-token time
- Acceptance rate validation on Qwen3 not available

**URL:** [arXiv:2603.19289](https://arxiv.org/abs/2603.19289)

---

### 41. MoE-SpAc: Efficient MoE Inference Based on Speculative Activation Utility in Heterogeneous Edge Scenarios

**Authors:** Shuhuai Li et al. (multiple institutions) | **Venue:** arXiv preprint | **Date:** 2026-03-11 | **Citations:** ~1 | **Importance:** ⭐⭐⭐⭐ (4/5)

**Core Technique:** Introduces "Speculative Activation Utility" (SAU)—a metric combining predicted activation probability and expert utility to the current computation. SAU guides both prefetch priority and cache eviction decisions. In heterogeneous edge environments (varying CPU+GPU resources), SAU adapts to hardware state.

**Hardware:** Heterogeneous edge (CPU+GPU combinations, mobile SoC) | **Models:** Mixtral 8x7B, DeepSeek-MoE-Lite | **Benchmarks:** Edge latency, SSD hit rate, accuracy

**Key Results:** Better cache efficiency than LRU and frequency-only approaches; heterogeneous adaptation reduces latency variance across different hardware configs.

**Pros (for project goal):**
- SAU metric combines prediction confidence with utility—directly applicable for deciding prefetch priority given limited PCIe bandwidth
- Heterogeneous adaptation across CPU configurations matches the "weak CPU vs. strong CPU" product variants
- Edge-focused; directly relevant scenario

**Cons (for project goal):**
- SAU computation adds overhead per expert decision
- Very new paper; limited validation
- Does not explicitly address NPU scenarios

**URL:** [arXiv:2603.09983](https://arxiv.org/abs/2603.09983)

---

### 42. Expert Streaming: Accelerating Low-Batch MoE Inference via Multi-chiplet Architecture and Dynamic Expert Trajectory Scheduling

**Authors:** Songchen Ma et al. (HKUST) | **Venue:** arXiv preprint | **Date:** 2026-03-29 | **Citations:** ~1 | **Importance:** ⭐⭐⭐⭐ (4/5)

**Core Technique:** Proposes multi-chiplet architecture for edge MoE inference where different chiplets hold different experts (near-memory partitioning). Introduces "expert trajectory scheduling" that predicts the expert activation sequence across layers and schedules chiplet operations to maximize parallelism. Directly targets the "on-device deployments with limited on-chip memory" problem statement.

**Hardware:** Multi-chiplet edge SoC with limited on-chip memory | **Models:** Mixtral 8x7B (representative) | **Benchmarks:** Throughput, latency, memory efficiency

**Key Results:** Significant throughput improvement at low batch sizes by exploiting inter-chiplet expert streaming; workload imbalance addressed by trajectory scheduling.

**Pros (for project goal):**
- **Most architecturally similar to the target system**: multi-chiplet ≈ NPU HBM + CPU DRAM; expert streaming ≈ PCIe DMA prefetch
- Low-batch inference optimization directly matches edge single-user use case
- Dynamic expert trajectory scheduling concept is the right approach for the project's prefetch engine
- Edge AI focused; hardware constraints similar

**Cons (for project goal):**
- Requires custom multi-chiplet architecture; the project uses a standard NPU + CPU PC setup
- chiplet-to-chiplet interconnect ≠ PCIe 4.0 bandwidth model
- No public implementation

**URL:** [arXiv:2603.27624](https://arxiv.org/abs/2603.27624)

---

### 43. SpecMoE: A Fast and Efficient MoE Inference via Self-Assisted Speculative Decoding (DAC 2026)

**Authors:** Jehyeon Bang et al. (POSTECH, KAIST) | **Venue:** DAC 2026 | **Date:** 2026-04-11 | **Citations:** ~1 | **Importance:** ⭐⭐⭐⭐ (4/5)

**Core Technique:** "Self-assisted" speculative decoding: instead of a separate draft model, uses the MoE model's own earlier (shallower) layers as the draft. The first N layers of the transformer provide a cheap prediction of the final output, which is used to generate speculative tokens AND predict expert routing for prefetching deeper layers' experts. This eliminates the need for a separate draft model.

**Hardware:** Memory-constrained GPU + CPU RAM | **Models:** Mixtral 8x7B, DeepSeek-V2 | **Benchmarks:** Throughput, latency, memory vs. separate draft model approaches

**Key Results:** Better performance than separate-draft speculative decoding while using ~50% less additional memory for the draft component.

**Pros (for project goal):**
- No separate draft model—saves NPU HBM; uses the model's own shallow layers for prediction
- Self-speculation is elegant and implementable without additional model weights
- DAC 2026 peer-reviewed; architecture-aware approach
- Expert prediction using shallow layers is implementable in llama.cpp without model modification

**Cons (for project goal):**
- Shallow-layer prediction requires running early layers before prefetching later-layer experts—still adds latency to the pipeline
- Speculative decoding verification overhead at NPU level
- Self-speculation acceptance rate may vary by task

**URL:** [arXiv:2604.10152](https://arxiv.org/abs/2604.10152)

---

### 44. NPUMoE: Efficient Mixture-of-Experts LLM Inference with Apple Silicon NPUs

**Authors:** Afsara Benazir, Felix Xiaozhu Lin (University of Virginia) | **Venue:** arXiv preprint | **Date:** 2026-04-20 | **Citations:** ~1 | **Importance:** ⭐⭐⭐⭐⭐ (5/5)

**Core Technique:** First paper directly addressing NPU-based MoE inference. Three techniques for Apple's ANE (Apple Neural Engine): (1) **Static expert capacity tiers**—offline calibration determines expert popularity and pre-assigns capacity budgets, eliminating dynamic shape issues; (2) **Grouped expert execution**—batches multiple small experts into single NPU dispatch to reduce kernel launch overhead; (3) **Load-aware expert graph residency**—keeps the most critical compute graphs permanently on NPU to avoid CPU-NPU synchronization overhead. Dense, static computation → NPU; dynamic operations (top-k, scatter/gather) → CPU/GPU fallback.

**Hardware:** Apple Silicon M-series (Apple Neural Engine + CPU + GPU) | **Models:** Mixtral 8x7B, Phi-MoE, Llama-MoE (3 representative MoE LLMs) | **Benchmarks:** Latency reduction, energy efficiency improvement, CPU cycle reduction

**Key Results:** 1.32–5.55× latency reduction; 1.81–7.37× energy efficiency improvement; 1.78–5.54× CPU cycle reduction vs. CPU/GPU baselines.

**Pros (for project goal):**
- **Most directly applicable paper in the entire corpus**: addresses exactly the challenge of using an NPU (Apple ANE ≈ project's custom NPU) for MoE inference
- Expert capacity static-tiering = the "hot experts permanently in NPU HBM" approach advocated in background.md
- Grouped expert execution to reduce dispatch overhead is directly needed for the project's NPU with many small Qwen3 experts
- Load-aware residency management is implementable in llama.cpp
- No model modification; works with standard checkpoints

**Cons (for project goal):**
- Apple ANE architecture differs from project NPU (different ISA, memory hierarchy, dispatch model)
- Apple Silicon has unified memory; project has PCIe-separated NPU memory—fundamental bandwidth model difference
- Expert capacity tiers determined by offline calibration; calibration effort needed for each model
- Grouped expert execution assumes enough experts activate together—may not hold for Qwen3's sparse fine-grained routing

**URL:** [arXiv:2604.18788](https://arxiv.org/abs/2604.18788)

---

### 45. Pipelined Sharding for CPU-GPU Hybrid Inference (MLSys 2026 Industry)

**Authors:** Aditya Ukarande et al. (Intel) | **Venue:** MLSys 2026 Industry Track | **Date:** 2026-04-29 | **Citations:** ~1 | **Importance:** ⭐⭐⭐⭐ (4/5)

**Core Technique:** "Pipelined sharding"—a benchmark-profile-guided CPU-GPU hybrid scheduling technique for running large LLMs and VLMs (including MoE) on client systems. Uses a profiling phase to characterize each model layer's compute-memory ratio and assigns layers to CPU or GPU based on bandwidth rooflines. Introduces a pipeline scheduler that overlaps CPU and GPU execution to maximize throughput.

**Hardware:** Intel-based client systems (CPU + integrated/discrete GPU) | **Models:** Mixtral 8x7B, Llama 3 (various sizes), MoE variants | **Benchmarks:** Throughput (tok/s), latency, vs. llama.cpp and Ollama

**Key Results:** Significant throughput improvements over llama.cpp and Ollama for hybrid CPU-GPU scenarios; validated at MLSys 2026 Industry Track.

**Pros (for project goal):**
- **Directly targets client/edge systems with hybrid CPU+accelerator**—most directly applicable hardware scenario
- Benchmark-profile-guided approach aligns with background.md's optional offline profiling
- MLSys 2026 Industry Track validates practical deployability
- Intel paper likely compatible with x86 CPUs used in mini-PCs
- Comparison against llama.cpp baseline directly relevant

**Cons (for project goal):**
- "GPU" in their system is an integrated/discrete GPU, not a custom NPU; kernel mapping needed
- Pipelined sharding may change the forward pass structure in ways that complicate llama.cpp integration
- Does not address SSD as a third offload tier
- Profile-guided approach assumes representative calibration workloads

**URL:** [arXiv:2604.26334](https://arxiv.org/abs/2604.26334)

---

### 46. STUN: Structured-Then-Unstructured Pruning for Scalable MoE Pruning

**Authors:** Jaeseong Lee et al. (POSTECH, Microsoft) | **Venue:** ACL 2025 | **Date:** 2024-09-10 | **Citations:** ~15 | **Importance:** ⭐⭐⭐ (3/5)

**Core Technique:** Two-stage pruning: first, structured expert pruning (removes entire experts based on behavioral similarity clustering), then unstructured weight pruning within remaining experts. Shows that the structured stage actually helps unstructured pruning outperform unstructured-only. Scales to 480B parameter models (Snowflake Arctic).

**Hardware:** GPU (H100 single-GPU for small models) | **Models:** Mixtral 8x7B, Snowflake Arctic 480B | **Benchmarks:** GSM8K, common NLP benchmarks, perplexity

**Key Results:** 40% sparsity with nearly no performance loss on Mixtral; generalizes to very large MoE models.

**Pros (for project goal):**
- Reducing total expert count via structured pruning directly reduces the expert pool the project needs to manage
- Expert similarity clustering provides the "hot vs. cold" expert importance metric needed for the project's cache
- Could reduce Qwen3-35B-A3B's memory footprint to fit better within NPU+DRAM budget

**Cons (for project goal):**
- Removing experts permanently modifies the model—contradicts the "no model modification" requirement
- Accuracy impact may be unacceptable for reasoning-heavy tasks (Qwen3 is a reasoning model)
- ACL 2025 is an NLP venue; hardware/systems aspects less covered
- Would require modifying the HuggingFace checkpoint, preventing direct use of public models

**URL:** [arXiv:2409.06211](https://arxiv.org/abs/2409.06211)

---

### 47. A3D-MoE: Acceleration of Large Language Models with MoE via 3D Heterogeneous Integration

**Authors:** Wei-Hsing Huang et al. (Georgia Tech) | **Venue:** arXiv preprint | **Date:** 2025-07-25 | **Citations:** ~2 | **Importance:** ⭐⭐⭐ (3/5)

**Core Technique:** Proposes a 3D chiplet integration architecture where memory dies (DRAM) are stacked directly on compute dies, and expert weights are partitioned across the 3D stack. The near-stacked memory enables much higher bandwidth for cold expert access than PCIe. Simulation study showing substantial speedup over PCIe-based offloading.

**Hardware:** Simulated 3D heterogeneous SoC | **Models:** Mixtral 8x7B (simulated) | **Benchmarks:** Simulated throughput, latency vs. PCIe offloading

**Key Results:** 3–5× simulated throughput improvement by eliminating PCIe bandwidth bottleneck.

**Pros (for project goal):**
- The target NPU chip has 3D-stacked HBM (10 GB, 1200 GB/s)—directly relevant architecture
- Quantifies the speedup possible from high-bandwidth stacked memory vs. PCIe—motivates using NPU HBM as expert cache
- Confirms that HBM bandwidth (1200 GB/s) is the key advantage of the NPU for hot expert execution

**Cons (for project goal):**
- Simulation study; no real hardware implementation
- Proposes future 3D integration; project uses existing NPU + standard PCIe connection
- Limited validation; GT research group work

**URL:** [arXiv:2507.19142](https://arxiv.org/abs/2507.19142)

---

### 48. SlimCaching: Edge Caching of Mixture-of-Experts for Distributed Inference

**Authors:** Qian Chen et al. (HKU) | **Venue:** IEEE Transactions on Mobile Computing 2025 | **Date:** 2025-07-09 | **Citations:** ~5 | **Importance:** ⭐⭐⭐ (3/5)

**Core Technique:** Edge caching framework for distributed MoE inference across multiple edge nodes. Formulates expert placement as an optimization problem minimizing communication latency across edge-cloud topology. Each edge node maintains a partial expert cache; missing experts are fetched from neighboring nodes or cloud.

**Hardware:** Edge server + cloud (multiple nodes) | **Models:** Mixtral 8x7B | **Benchmarks:** End-to-end latency, communication cost, hit rate

**Key Results:** Optimal expert placement reduces inter-node communication by 40–60%; IEEE Transactions on Mobile Computing (top venue for edge systems).

**Pros (for project goal):**
- Expert placement optimization formulation could inspire the NPU/DRAM/SSD placement problem formulation
- IEEE TMC is a reputable venue; rigorously reviewed

**Cons (for project goal):**
- Distributed multi-node edge topology is completely different from single-chip M.2 card scenario
- Network communication latency model doesn't apply
- Optimization at a different granularity (across devices vs. across memory tiers)

**URL:** [arXiv:2507.06567](https://arxiv.org/abs/2507.06567)

---

## Trend Summary

### Category 1: Software-Based Offloading & Cache Management

The earliest papers in this space (Eliseev & Mazur 2023, MoE-Infinity Jan 2024) established the core insight: at batch=1, MoE models exhibit high activation locality—the same small subset of experts are repeatedly activated across consecutive tokens. This makes LRU-based expert caching surprisingly effective, with typical hit rates of 60–80%. The key design pattern is: (1) keep non-expert weights permanently in accelerator memory; (2) maintain an expert cache using frequency/recency scoring; (3) prefetch predicted next experts asynchronously during computation of current layer.

By 2025–2026, this baseline was refined with more sophisticated replacement policies (workload-aware in DALI, ML-based in FlashMoE, bit-sliced in SliceMoE) and formal theoretical analysis (Cache Management for MoE LLMs). The community largely converged on the observation that LRU is within 2× of optimal but that ML-based prediction can recover much of that gap—at the cost of a small training/calibration overhead. For the project, the recommendation is to implement a frequency+recency combined score initially and add ML-based prediction if needed.

### Category 2: Predictive Prefetching & Scheduling

Research in 2024–2026 progressively shifted from reactive caching (LRU, evict on miss) to proactive prefetching (predict future experts, load before needed). EdgeMoE introduced expert preloading with simple 1-step prediction; Fate's cross-layer gate extended this to 3-step lookahead; Pre-Attention Prediction (ETH Zurich) overlaps prefetch with attention computation. By 2025–2026, multi-layer lookahead prediction achieved 85–95% hit rates on standard benchmarks.

The key technical evolution was improving predictor accuracy: from simple input-token-hash routing predictors → lightweight linear predictors → attention-based cross-layer gates → residual-based predictors (DALI) → self-speculative shallow-layer predictors (SpecMoE DAC 2026). Each step improved hit rate at the cost of slightly more complexity. For the project, the cross-layer gate predictor (Fate) or residual-based predictor (DALI) offers the best accuracy/complexity tradeoff for llama.cpp implementation.

### Category 3: Hybrid CPU-GPU/NPU Compute

The hybrid compute direction emerged strongly in 2024–2025, driven by the observation that for large MoE models, the CPU's aggregate DRAM bandwidth (often 50–100 GB/s) can execute cold experts faster than transferring them to the GPU over PCIe. HybriMoE (DAC 2025) validated this for standard x86 CPUs with AVX-512; TriMoE (DAC 2026) extended this to AMX-enabled CPUs with DIMM-NDP as a third tier.

For the project's NPU, this direction is highly relevant: the CPU (68 GB/s DRAM bandwidth) can execute warm experts while the NPU (1200 GB/s HBM, 56 TOPS) handles hot experts—exactly the hybrid compute model of background.md. The key challenge is dynamic workload balancing across the two compute units, which HybriMoE and DALI have addressed. NPUMoE (Apple ANE, 2026) provides direct NPU-specific insights about expert dispatch latency, kernel grouping, and capacity tiering.

### Category 4: Compression, Quantization & Merging

Compression is primarily a bandwidth reduction strategy for offloading scenarios. HOBBIT showed mixed-precision (INT8 hot, INT4 cold) experts reduce transfer volume while maintaining accuracy. Dynamic Expert Quantization (Alibaba DAMO) extends this to adaptive precision per expert per step. SliceMoE introduces continuous precision via bit-slicing. These approaches directly reduce the effective bandwidth requirement—a critical benefit when PCIe 4.0 ×4 at 8 GB/s is the bottleneck.

Expert pruning (STUN, CoMoE) is less applicable for the project due to the no-model-modification requirement. Compression (quantization) is more suitable as it can be applied at inference time without changing model behavior beyond acceptable accuracy loss.

### Category 5: Speculative Decoding + Offloading

The combination of speculative decoding (SD) with expert offloading is a 2025–2026 innovation. The core idea: SD generates multiple tokens speculatively, amortizing the I/O cost of expert loading across multiple tokens. SpecMoEOff, SP-MoE, MoE-SpeQ, and SpecMoE (DAC 2026) all demonstrate 1.8–2.5× speedup. The innovation by SpecMoE (DAC 2026) of using the model's own shallow layers as the draft eliminates the need for a separate model.

For the project, speculative decoding is a high-value enhancement but should be treated as a secondary optimization after the base offloading engine is solid. The self-speculative approach (SpecMoE, DAC 2026) is particularly appealing for the NPU scenario as it requires no additional model.

---

## Gaps and Future Directions

1. **No work targets PCIe-attached NPU with HBM + SSD**: All existing hybrid compute papers assume GPU+CPU DRAM (two tiers). The project's three-tier NPU HBM / CPU DRAM / SSD architecture is novel.

2. **NPU-specific expert dispatch overhead unquantified**: NPUMoE (Apple ANE) shows dispatch overhead is a major concern, but the project's custom NPU will have different dispatch characteristics. Profiling and optimizing this is critical and unexplored.

3. **Qwen3's fine-grained MoE not studied**: Qwen3-35B-A3B has many small experts per layer (fine-grained MoE). Most papers study Mixtral (8 experts/layer) or DeepSeek-V2 (64 experts/layer). The activation locality of Qwen3's ~128 experts/layer at the 35B scale needs empirical measurement.

4. **SSD bandwidth modeling missing**: Papers addressing SSD offloading (FlashMoE, EdgeMoE, OD-MoE) don't rigorously model the shared SSD-PCIe bandwidth on single-PCIe-bus systems. In the project's scenario, SSD and NPU share 8 GB/s PCIe bandwidth—a critical constraint not addressed in literature.

5. **llama.cpp MoE expert scheduling is a gap**: None of the papers target llama.cpp as the implementation framework. Implementation complexity for llama.cpp differs significantly from vLLM/DeepSpeed/kTransformers.

6. **Batch=1 decode speed target of 25 tok/s**: Most papers optimize throughput (batch > 1) or don't specify exact latency targets. Achieving >25 tok/s at batch=1 for a 35B MoE model over 8 GB/s PCIe requires careful bandwidth analysis; no paper provides this for the exact configuration.

---
