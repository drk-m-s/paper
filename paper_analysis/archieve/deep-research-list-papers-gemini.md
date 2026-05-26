# **Evolutionary Paradigms in Mixture of Experts Offloading and Efficient Inference (2024–2026)**

The architectural transition from dense to sparse Large Language Models (LLMs) has reached a critical juncture with the widespread adoption of the Mixture of Experts (MoE) framework. By decoupling the total parameter count from the per-token computational cost, MoE models such as Mixtral, DeepSeek, and Qwen have enabled the scaling of model capacity into the trillions of parameters while maintaining a relatively constant floating-point operations (FLOPs) budget.1 However, this shift has introduced a fundamental "memory paradox": while the compute requirements remain manageable, the memory footprint required to store all expert weights frequently exceeds the High Bandwidth Memory (HBM) capacity of modern GPU hardware, particularly in resource-constrained or edge deployment scenarios.4 This discrepancy has necessitated a robust research effort spanning January 2024 to June 2026, focusing on expert offloading, predictive prefetching, hybrid execution, and speculative decoding to bridge the gap between massive sparse models and limited hardware resources.7

## **The Structural Mechanics of MoE Inefficiency**

To understand the systems-level innovations of the 2024–2026 period, one must first characterize the inherent bottlenecks in MoE inference. In a standard MoE layer, an input token ![][image1] is processed by a gating function ![][image2], which typically selects a sparse subset of ![][image3] experts from a total of ![][image4] available expert networks.3 The output is a weighted sum:

![][image5]  
where ![][image6] are the indices of the top\-![][image3] experts and ![][image7] are their corresponding routing weights.1 While this mechanism ensures that only a fraction of the parameters are computed, the memory system must still provide access to all ![][image4] experts. For massive models like DeepSeek-V3 with 671 billion parameters, even the active parameters per token (approximately 37 billion) represent a significant fraction of VRAM, but the total model size (over 1.2 terabytes in FP16) far outstrips the 80GB capacity of an NVIDIA H100 GPU.10

| Model Architecture | Total Parameters | Active Parameters (Per Token) | VRAM Required (FP16) | VRAM Required (4-bit) |
| :---- | :---- | :---- | :---- | :---- |
| Mixtral 8x7B | 47B | 13B | \~94 GB | \~28 GB |
| DeepSeek-MoE (16B) | 16B | 2.8B | \~32 GB | \~10 GB |
| Mixtral 8x22B | 141B | 39B | \~280 GB | \~80 GB |
| DeepSeek-V3 | 671B | 37B | \~1,340 GB | \~380 GB |
| Qwen3-30B-A3B | 30B | 4B | \~60 GB | \~18 GB |

The table above illustrates the scaling challenges. Even with 4-bit quantization, state-of-the-art MoE models require memory capacities that necessitate offloading to CPU RAM or secondary storage (SSD).2 The primary performance bottleneck in such offloading systems is the PCIe interconnect. For instance, transferring a single expert of 100 million parameters in FP16 requires 200MB of bandwidth; at PCIe 4.0 speeds (\~31.5 GB/s), this takes approximately 6.3 milliseconds, which can be significantly longer than the few hundred microseconds required for GPU computation.6

## **Taxonomy of Optimization Methodologies (2024–2026)**

Research between 2024 and 2026 can be systematically categorized into six primary methodologies. Each methodology addresses the I/O-compute mismatch through different layers of the software and hardware stack.

### **Software-Based Offloading and Cache Management**

Early 2024 research focused on reactive caching strategies that treated the GPU VRAM as a limited cache for experts stored in host memory.15 The MoE-Infinity system was a seminal contribution in this category, identifying that MoE models exhibit high activation sparsity and temporal locality during single-batch decoding.10 By tracing expert activations, MoE-Infinity implemented a sparsity-aware expert cache that prioritized the residency of frequently reused experts, achieving per-token latency improvements of 3.1x to 16.7x over generic offloading systems.16

Other systems, such as ADEPT, introduced a two-stage approach: analyzing the input's semantic domain during the prefill phase to preload relevant experts and utilizing a locality-aware caching predictor during the decoding phase.7 This decoupling of prefill and decoding optimizations allowed for a 33% reduction in peak memory usage while speeding up inference by 3.45x.7 The Harvest system, appearing in early 2026, extended this by utilizing peer-GPU memory as an intermediate cache tier, serving expert misses from another GPU's unused VRAM via NVLink or PCIe instead of falling back to host DRAM.19

### **Predictive Prefetching and Speculative Activation**

To overcome the PCIe latency ceiling, systems evolved toward prefetching—loading experts into VRAM before they are selected by the router.4 The Pre-gated MoE framework proposed an architectural modification where the router at layer ![][image8] determines the experts for layer ![][image9], allowing the transfer to occur during the attention computation of layer ![][image8].9 However, this approach's requirement for retraining led to the development of "parameter-free" prefetching.9

Research in 2025 and 2026, such as the YALIS engine and the PreScope system, demonstrated that internal model representations (hidden states) contain latent signals that can accurately predict future routing decisions several layers in advance.6 PreScope introduced the Learnable Layer-aware Predictor (LLaPor), which utilized a small neural estimator to score experts by contextual signals, enabling a prefetch-aware cross-layer scheduling strategy.6 CommitMoE refined this by suggesting that in cases of low router certainty, the model is inherently robust to expert selection, allowing the system to "commit" to a predicted expert and skip the fallback latency entirely.15

### **Hybrid CPU-GPU Heterogeneous Computing**

A dominant theme in the 2025–2026 period is the utilization of the CPU as an active compute participant rather than a mere storage reservoir.23 The Fiddler system introduced a cost-model-driven orchestration that dynamically decided whether to execute an expert on the GPU (incurring transfer cost) or on the CPU (incurring slower compute cost) based on the current batch size and PCIe traffic.23

The kTransformers project further optimized this by developing specialized CPU kernels leveraging Intel AMX (Advanced Matrix Extensions) and AVX512\_BF16 instructions, achieving CPU expert computation speeds that narrowed the gap with GPUs.24 This "heterogeneous expert placement" strategy allowed "hot" experts to reside on the GPU while "cold" experts were computed in-place on the CPU, a technique that proved essential for running models like DeepSeek-V3/R1 on consumer hardware with 24GB VRAM.24 TriMoE extended this into a three-tier GPU-CPU-NDP (Near-Data Processing) architecture, mapping experts across the hierarchy based on their activation frequency and the compute-to-bandwidth ratio of each tier.27

### **Model Compression and Expert-Level Quantization**

Quantization research for MoE reached high levels of sophistication, moving beyond uniform bit-widths to mixed-precision expert offloading.29 The PMQ (Pre-Loading Mixed-Precision Quantization) framework formulated bit-width allocation as an Integer Programming problem, balancing quantization loss against activation frequency.29 This ensured that the most critical experts remained at high precision (BF16/FP8) while rarely used experts were compressed to 2-bit or even 1.58-bit representations.30

MoEpic introduced a novel "expert split" mechanism, where expert weights were vertically divided into top and bottom segments.34 By caching only the top segments of hot experts in VRAM, MoEpic effectively doubled the cache hit rate under fixed memory budgets.34 During inference, the system prefetched the bottom segments of predicted experts, significantly improving the transfer-computation overlap.35 This hierarchical quantization was further refined by HOBBIT, which dynamically replaced cache-miss experts with ultra-low precision versions to minimize latency at the cost of negligible accuracy drops.31

### **Speculative Decoding and Self-Assisted Inference**

The most advanced methodological shift occurred in late 2025 with the application of speculative decoding to MoE architectures.5 Speculative decoding typically uses a small "draft" model to guess tokens, but hosting a separate draft model for a massive MoE is memory-prohibitive.5 SpecMoE solved this by introducing "self-assisted" speculative decoding, where parts of the target MoE model itself served as the draft.5

| System | Draft Strategy | Performance Gain | Bandwidth Reduction |
| :---- | :---- | :---- | :---- |
| SpecMoE | Self-assisted (target sub-selection) | 4.30x throughput | 76.7% |
| Heterogeneous SpecDec | Optimized draft for low-end HW | 2.25x \- 10.14x speedup | N/A |
| FloE | Dual predictors (Inter/Intra expert) | 48.7x vs DeepSpeed-MII | 8.5x memory footprint |
| FASER | Fine-grained phase management | 1.92x latency reduction | N/A |

By generating a sequence of tokens in the speculative phase using experts already resident in VRAM, SpecMoE generated multiple tokens per expert parameter retrieval.5 This fundamentally altered the I/O-compute relationship, as the cost of transferring an expert was amortized over several output tokens, effectively breaking the memory-IO bound nature of decoding.5

### **Semantic Parallelism and Serving-Level Optimizations**

In distributed settings, the bottleneck shifted from memory capacity to the "All-to-All" communication overhead of Expert Parallelism (EP).11 Sem-MoE (Semantic Parallelism) proposed a paradigm shift by co-scheduling tokens and experts based on their semantic affinity.11 By proactively modeling activation likelihood, Sem-MoE clustered co-activating experts onto the same GPU and reshuffled tokens to the appropriate device during the attention phase, reducing cross-device communication volume significantly.11

At the serving layer, MoE-Gen introduced module-based batching, which decoupled the batch sizes for attention and expert modules.44 Since experts have lower arithmetic intensity than attention layers, MoE-Gen accumulated multiple attention batches and processed them in one large expert batch, maximizing GPU utilization and increasing decoding throughput by 31x.45

## **Exhaustive Research Compendium: Jan 2024 – June 2026**

The following tables provide an exhaustive list of MoE offloading and efficient inference papers, categorized by methodology and ranked by their technical significance and impact on subsequent research.

### **Table 1: Software-Based Offloading and Cache Management Systems**

| Paper / System Name | Date | Importance | Key Contribution |
| :---- | :---- | :---- | :---- |
| MoE-Infinity 16 | Jan 2024 | 5 | Sequence-level activation tracing and PEC (Sparsity-aware cache). |
| MoE-Offloading 15 | Jan 2024 | 4 | Defined the LRU-based expert swapping baseline for modern MoE. |
| DeepSpeed-MII 46 | 2024 | 2 | Standard industry framework for low-latency sparse inference. |
| ZeRO-Infinity 13 | 2024 | 3 | SSD offloading and NVMe-to-GPU direct transfer foundation. |
| MemAscend 13 | May 2025 | 1 | Adaptive buffer pool to reduce system memory waste in SSD offloads. |
| Harvest 19 | Feb 2026 | 3 | Peer-GPU memory caching tier to mitigate CPU DRAM latency. |
| ADEPT 7 | Feb 2026 | 4 | Domain-aware prefill and locality-aware decoding phase optimization. |

### **Table 2: Predictive Prefetching and Scheduling Frameworks**

| Paper / System Name | Date | Importance | Key Contribution |
| :---- | :---- | :---- | :---- |
| Pre-gated MoE 9 | Mar 2024 | 4 | Router-at-L predicts L+1; requires model retraining. |
| AdapMoE 15 | Apr 2024 | 3 | Adaptive gating and layer-level prefetching management. |
| Fate 6 | Jan 2025 | 2 | Routing prediction via adjacent-layer gating input similarity. |
| Klotski 6 | Feb 2025 | 3 | Large-batch I/O overlap; trade-off between throughput and accuracy. |
| PreScope 6 | Mar 2025 | 5 | Learnable predictor (LLaPor) and cross-layer scheduling. |
| CommitMoE 15 | Jan 2026 | 4 | Fallback-free Commit Router for deterministic expert preloading. |
| YALIS 9 | Mar 2026 | 5 | Internal representation-driven speculation for pre-trained models. |
| DALI 24 | May 2026 | 3 | Residual-based prefetching and 0-1 optimization for assignment. |

### **Table 3: Hybrid Compute and Collaborative Inference Systems**

| Paper / System Name | Date | Importance | Key Contribution |
| :---- | :---- | :---- | :---- |
| Fiddler 23 | Feb 2024 | 5 | CPU-GPU orchestration using cost-modeling for optimal placement. |
| PowerInfer 47 | Oct 2024 | 4 | Hot/Cold neuron partitioning (SOSP 2024). |
| HyGen 46 | Jan 2025 | 1 | Interference-aware serving for co-located online/offline workloads. |
| HybriMoE 48 | Mar 2025 | 4 | Expert-wise hybrid framework with queuing for hot experts. |
| kTransformers 25 | Apr 2025 | 5 | AMX/AVX CPU kernels; support for DeepSeek-V3 on consumer GPUs. |
| OmniServe 49 | May 2025 | 3 | Attention Piggybacking to offload BE service computation to CPUs. |
| TriMoE 27 | May 2026 | 4 | GPU-CPU-NDP architecture; Hot/Warm/Cold expert mapping. |
| MoE-SpAc 17 | Jun 2026 | 2 | Speculative Utility Estimator for heterogeneous edge scenarios. |

### **Table 4: Compression, Quantization, and Merging Papers**

| Paper / System Name | Date | Importance | Key Contribution |
| :---- | :---- | :---- | :---- |
| NAEE 50 | Apr 2024 | 3 | Pruning unimportant experts by minimizing pruning error. |
| PMQ / MC-MoE 30 | Aug 2024 | 4 | Mixed-precision quantization as an Integer Programming problem. |
| MoEpic 34 | Sep 2024 | 5 | Expert vertical split (Top/Bottom) to increase cache capacity. |
| HC-SMoE 51 | 2024 | 2 | Hierarchical clustering for expert merging. |
| Sub-MoE 51 | Jan 2025 | 4 | Subspace expert merging via subspace decomposition. |
| HOBBIT 37 | Jun 2025 | 3 | Mixed-precision offloading with token-level dynamic loading. |
| ReaLB 52 | Apr 2026 | 2 | Real-time load balancing via dynamic precision adjustment. |

### **Table 5: High-Throughput Serving and Speculative Decoding**

| Paper / System Name | Date | Importance | Key Contribution |
| :---- | :---- | :---- | :---- |
| MoE-Lightning 15 | Apr 2024 | 3 | CGOPIPE strategy for computation-communication overlapping. |
| FloE 4 | Jul 2025 | 4 | Dual predictors for inter-expert sparsity; hybrid compression. |
| MoE-Gen 44 | Sep 2025 | 5 | Module-based batching for high-throughput single-GPU inference. |
| SpecMoE 5 | Jan 2026 | 5 | Self-assisted speculative decoding; DAC 2026\. |
| Sem-MoE 11 | Mar 2026 | 5 | Semantic Parallelism; model-data co-scheduling (ICLR 2026). |
| FASER 52 | Apr 2026 | 3 | Fine-grained phase management for speculative decoding. |

## **Second-Order Insights: Trends in Architectural Adaptation**

The research between 2024 and 2026 reveals three profound shifts in the understanding of sparse model systems.

### **From Model-Centric to Workload-Centric Caching**

Early systems like MoE-Offloading (2024) assumed a static "hotness" of experts, but data from 2025 research (e.g., kTransformers, PreScope) showed that expert activation is highly dynamic and depends heavily on the "layer group"—input, middle, or output layers.6 Input and output layers exhibit stronger routing correlation and skewed weights, whereas middle layers show higher cosine similarity between tokens and are dominated by a few "hot" experts.6 This realization led to layer-group-aware scheduling, where prefetching budgets are dynamically allocated: more aggressive prefetching is applied to input/output layers with predictable routing, while caching is prioritized for the middle layers with stable hot experts.6

### **The Amortization of I/O via Speculation**

A critical causal relationship identified in the 2026 papers (SpecMoE, FASER) is that the only way to fundamentally defeat the PCIe bottleneck is to generate more information per byte transferred.5 In standard auto-regressive decoding, each token requires a new set of expert weights to be transferred.5 Speculative decoding transforms this: by using a draft (even a self-assisted one), the system transfers experts once and uses them to verify multiple tokens.5 This shift essentially moves MoE inference from a latency-bound regime to a throughput-bound regime, enabling high-performance inference even on legacy PCIe 3.0 hardware.5

### **Semantic Alignment as the New Parallelism**

The development of Semantic Parallelism (Sem-MoE) suggests that the traditional separation of "model parallelism" and "data parallelism" is insufficient for MoE models.11 Because experts are specialized, tokens are not "randomly" distributed; they follow semantic paths.3 The insight of Sem-MoE is that by aligning the physical placement of experts with these semantic paths, communication can be "pre-cached" by moving tokens into the correct device during the attention synchronization phase, which occurs before the MoE routing decision is even finalized.41 This proactively collapses the all-to-all communication overhead, which was the single largest inhibitor of distributed MoE scaling in early 2024\.17

## **Analysis of Hardware-Software Co-Design for Sparse Kernels**

A significant portion of the efficiency gains in 2025–2026 can be attributed to the development of specialized kernels that handle the unique memory access patterns of MoE.

### **CPU Acceleration via AMX and AVX512**

For hybrid systems like kTransformers and Fiddler, the efficiency of the CPU backend is paramount.25 Intel’s AMX (Advanced Matrix Extensions) proved to be a transformative technology, providing a tile-based matrix multiplication unit that handles the massive expert FFNs more efficiently than traditional vector-based AVX instructions.24 The 2026 TriMoE system demonstrated that by using AMX for "warm" experts, one could maintain 16 tokens/s on DeepSeek-R1-0528 (FP8) even when only a fraction of layers fit on the GPU.25

### **Near-Data Processing (NDP) and Memory-Centric Computing**

The shift toward TriMoE and DIMM-PIM (Processing-in-Memory) architectures in 2025 and 2026 addresses the "memory wall" head-on.27 In these systems, experts are stored in specialized DIMMs that include small compute units.28 This eliminates the need to transfer weights over the memory bus to the CPU or GPU cores; instead, the computation is performed within the memory hierarchy itself.27 This is particularly effective for "cold" experts, where the low frequency of activation does not justify the energy and bandwidth cost of moving the weights to a high-performance processor.27

## **Deployment Constraints and Future Directions**

Despite these advancements, several challenges remain as of June 2026\. The "Dead Expert" problem—where certain experts are never activated during training and thus provide no value during inference—still plagues many massive models, and current systems handle them inefficiently by reserving memory that could be used for hot experts.1 Furthermore, while prefetching has become highly accurate, "inaccurate prefetching" can still degrade performance below on-demand baselines by saturating PCIe bandwidth with useless data.9

Future research is expected to focus on:

1. **Agentic KV Cache Management:** Integrating MoE offloading with KV cache compression for long-context agentic workloads, where reasoning steps are interleaved with external tool calls.47  
2. **Unified Semantic Serving:** Extending the Sem-MoE paradigm to include RAG (Retrieval-Augmented Generation) systems, where retrieved chunks are co-scheduled with the experts that will process them.11  
3. **On-the-Fly Quantization Adaptation:** Systems that dynamically adjust the bit-width of experts in response to real-time network and PCIe congestion.31

In conclusion, the period from January 2024 to June 2026 has seen the Mixture of Experts architecture evolve from a research curiosity into a deployable reality for local and edge environments. The synergy between speculative decoding, predictive prefetching, and heterogeneous CPU-GPU computation has successfully mitigated the memory-latency bottleneck, enabling the current generation of trillion-parameter sparse models to run efficiently on hardware once thought to be orders of magnitude too small.

#### **Works cited**

1. Mixture of Experts (MoE) Architecture Paper Review and Production Scaling 2026 | Chaos and Order, accessed May 7, 2026, [https://www.youngju.dev/blog/ai-papers/2026-03-04-ai-papers-mixture-of-experts-scaling-2026.en](https://www.youngju.dev/blog/ai-papers/2026-03-04-ai-papers-mixture-of-experts-scaling-2026.en)  
2. Mixture of Experts – Scaling Transformers Without Breaking the FLOPS Bank | Sifal Klioui, accessed May 7, 2026, [https://sifal.social/posts/Mixture-of-Experts-Scaling-Transformers-Without-Breaking-the-FLOPS-Bank/](https://sifal.social/posts/Mixture-of-Experts-Scaling-Transformers-Without-Breaking-the-FLOPS-Bank/)  
3. A Comprehensive Survey of Mixture-of-Experts: Algorithms, Theory, and Applications \- arXiv, accessed May 7, 2026, [https://arxiv.org/html/2503.07137v4](https://arxiv.org/html/2503.07137v4)  
4. ICML Poster FloE: On-the-Fly MoE Inference on Memory ..., accessed May 7, 2026, [https://icml.cc/virtual/2025/poster/44378](https://icml.cc/virtual/2025/poster/44378)  
5. SpecMoE: A Fast and Efficient Mixture-of-Experts Inference via Self-Assisted Speculative Decoding \- arXiv, accessed May 7, 2026, [https://arxiv.org/pdf/2604.10152](https://arxiv.org/pdf/2604.10152)  
6. PreScope: Unleashing the Power of Prefetching for Resource-Constrained MoE Inference, accessed May 7, 2026, [https://arxiv.org/html/2509.23638v1](https://arxiv.org/html/2509.23638v1)  
7. Two-Stage Expert Offloading for Domain-Aware MoE Inference \- IEEE Xplore, accessed May 7, 2026, [https://ieeexplore.ieee.org/iel8/6287639/11323511/11397596.pdf](https://ieeexplore.ieee.org/iel8/6287639/11323511/11397596.pdf)  
8. LayerScope: Predictive Cross-Layer Scheduling for Efficient Multi-Batch MoE Inference on Legacy Servers \- arXiv, accessed May 7, 2026, [https://arxiv.org/html/2509.23638v2](https://arxiv.org/html/2509.23638v2)  
9. Speculating Experts Accelerates Inference for Mixture-of-Experts \- arXiv, accessed May 7, 2026, [https://arxiv.org/html/2603.19289v1](https://arxiv.org/html/2603.19289v1)  
10. Efficient MoE Inference on Personal Machines with Sparsity-Aware Expert Cache \- arXiv, accessed May 7, 2026, [https://arxiv.org/pdf/2401.14361](https://arxiv.org/pdf/2401.14361)  
11. Semantic Parallelism: Redefining Efficient MoE Inference via Model-Data Co-Scheduling, accessed May 7, 2026, [https://arxiv.org/html/2503.04398v5](https://arxiv.org/html/2503.04398v5)  
12. DeepSeek's Technical Playbook: From MLA to Conditional Memory \- Suvash Sedhain, accessed May 7, 2026, [https://mesuvash.github.io/blog/2026/deepseek-v3/](https://mesuvash.github.io/blog/2026/deepseek-v3/)  
13. MemAscend: System Memory Optimization for SSD-Offloaded LLM Fine-Tuning \- arXiv, accessed May 7, 2026, [https://arxiv.org/html/2505.23254v4](https://arxiv.org/html/2505.23254v4)  
14. DeepSeek-V3.1: How to Run Locally | Unsloth Documentation, accessed May 7, 2026, [https://unsloth.ai/docs/models/tutorials/deepseek-v3.1-how-to-run-locally](https://unsloth.ai/docs/models/tutorials/deepseek-v3.1-how-to-run-locally)  
15. CommitMoE: Efficient Fallback-Free MoE Inference with Offloading ..., accessed May 7, 2026, [https://ojs.aaai.org/index.php/AAAI/article/view/39454/43415](https://ojs.aaai.org/index.php/AAAI/article/view/39454/43415)  
16. \[2401.14361\] MoE-Infinity: Efficient MoE Inference on Personal Machines with Sparsity-Aware Expert Cache \- arXiv, accessed May 7, 2026, [https://arxiv.org/abs/2401.14361](https://arxiv.org/abs/2401.14361)  
17. ExpertFlow: Optimized Expert Activation and Token Allocation for Efficient Mixture-of-Experts Inference \- Semantic Scholar, accessed May 7, 2026, [https://www.semanticscholar.org/paper/ExpertFlow%3A-Optimized-Expert-Activation-and-Token-He-Zhang/518ea456c740b5eae4e24b43b2b235d890ef7092](https://www.semanticscholar.org/paper/ExpertFlow%3A-Optimized-Expert-Activation-and-Token-He-Zhang/518ea456c740b5eae4e24b43b2b235d890ef7092)  
18. Two-Stage Expert Offloading for Domain-Aware MoE Inference ..., accessed May 7, 2026, [https://ieeexplore.ieee.org/document/11397596](https://ieeexplore.ieee.org/document/11397596)  
19. Harvest: Opportunistic Peer-to-Peer GPU Caching for LLM Inference \- arXiv, accessed May 7, 2026, [https://arxiv.org/html/2602.00328v1](https://arxiv.org/html/2602.00328v1)  
20. Efficient CPU-GPU Collaborative Inference for MoE-based LLMs on Memory-Limited Systems \- arXiv, accessed May 7, 2026, [https://arxiv.org/html/2512.16473v1](https://arxiv.org/html/2512.16473v1)  
21. \[2603.19289\] Speculating Experts Accelerates Inference for Mixture-of-Experts \- arXiv, accessed May 7, 2026, [https://arxiv.org/abs/2603.19289](https://arxiv.org/abs/2603.19289)  
22. CommitMoE: Efficient Fallback-Free MoE Inference with Offloading Under GPU Memory Constraints \- Underline Science, accessed May 7, 2026, [https://underline.io/lecture/142243-commitmoe-efficient-fallback-free-moe-inference-with-offloading-under-gpu-memory-constraints](https://underline.io/lecture/142243-commitmoe-efficient-fallback-free-moe-inference-with-offloading-under-gpu-memory-constraints)  
23. \[2402.07033\] Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models \- arXiv, accessed May 7, 2026, [https://arxiv.org/abs/2402.07033](https://arxiv.org/abs/2402.07033)  
24. KTransformers: Unleashing the Full Potential of CPU/GPU Hybrid Inference for MoE Models | Request PDF \- ResearchGate, accessed May 7, 2026, [https://www.researchgate.net/publication/396443066\_KTransformers\_Unleashing\_the\_Full\_Potential\_of\_CPUGPU\_Hybrid\_Inference\_for\_MoE\_Models](https://www.researchgate.net/publication/396443066_KTransformers_Unleashing_the_Full_Potential_of_CPUGPU_Hybrid_Inference_for_MoE_Models)  
25. kvcache-ai/ktransformers at softmaxdata.com \- GitHub, accessed May 7, 2026, [https://github.com/kvcache-ai/ktransformers?ref=softmaxdata.com](https://github.com/kvcache-ai/ktransformers?ref=softmaxdata.com)  
26. Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models | OpenReview, accessed May 7, 2026, [https://openreview.net/forum?id=N5fVv6PZGz](https://openreview.net/forum?id=N5fVv6PZGz)  
27. Lian Liu \- CatalyzeX, accessed May 7, 2026, [https://www.catalyzex.com/author/Lian%20Liu](https://www.catalyzex.com/author/Lian%20Liu)  
28. L3: DIMM-PIM Integrated Architecture and Coordination for Scalable Long-Context LLM Inference \- Semantic Scholar, accessed May 7, 2026, [https://www.semanticscholar.org/paper/L3%3A-DIMM-PIM-Integrated-Architecture-and-for-LLM-Liu-Chen/53eb4a36c66311e912554aa32959421107698342](https://www.semanticscholar.org/paper/L3%3A-DIMM-PIM-Integrated-Architecture-and-for-LLM-Liu-Chen/53eb4a36c66311e912554aa32959421107698342)  
29. MoE inference cost cuts: 30+ patents analyzed \- PatSnap, accessed May 7, 2026, [https://www.patsnap.com/resources/blog/articles/moe-inference-cost-cuts-30-patents-analyzed/](https://www.patsnap.com/resources/blog/articles/moe-inference-cost-cuts-30-patents-analyzed/)  
30. Mixture Compressor for Mixture-of-Experts LLMs Gains More | OpenReview, accessed May 7, 2026, [https://openreview.net/forum?id=hheFYjOsWO](https://openreview.net/forum?id=hheFYjOsWO)  
31. AdapMoE: Adaptive Sensitivity-Based Expert Gating and Management for Efficient MoE Inference \- Semantic Scholar, accessed May 7, 2026, [https://www.semanticscholar.org/paper/AdapMoE%3A-Adaptive-Sensitivity-Based-Expert-Gating-Zhong-Liang/51c507cdbf0825c558f0792eecd8d0912a38803b](https://www.semanticscholar.org/paper/AdapMoE%3A-Adaptive-Sensitivity-Based-Expert-Gating-Zhong-Liang/51c507cdbf0825c558f0792eecd8d0912a38803b)  
32. Awesome On-Device Large Language Models \- GitHub, accessed May 7, 2026, [https://github.com/LumosJiang/Awesome-On-Device-LLMs](https://github.com/LumosJiang/Awesome-On-Device-LLMs)  
33. Network Edge Inference for Large Language Models: Principles, Techniques, and Opportunities \- arXiv, accessed May 7, 2026, [https://arxiv.org/html/2604.22906v1](https://arxiv.org/html/2604.22906v1)  
34. (PDF) Accelerating Mixture-of-Expert Inference with Adaptive Expert Split Mechanism, accessed May 7, 2026, [https://www.researchgate.net/publication/395402606\_Accelerating\_Mixture-of-Expert\_Inference\_with\_Adaptive\_Expert\_Split\_Mechanism](https://www.researchgate.net/publication/395402606_Accelerating_Mixture-of-Expert_Inference_with_Adaptive_Expert_Split_Mechanism)  
35. Accelerating Mixture-of-Expert Inference with Adaptive Expert Split Mechanism \- arXiv, accessed May 7, 2026, [https://arxiv.org/html/2509.08342v1](https://arxiv.org/html/2509.08342v1)  
36. Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference \- Semantic Scholar, accessed May 7, 2026, [https://www.semanticscholar.org/paper/Pre-gated-MoE%3A-An-Algorithm-System-Co-Design-for-Hwang-Wei/3ed178316be914658a80e561bf00576577f34389](https://www.semanticscholar.org/paper/Pre-gated-MoE%3A-An-Algorithm-System-Co-Design-for-Hwang-Wei/3ed178316be914658a80e561bf00576577f34389)  
37. HOBBIT: A Mixed Precision Expert Offloading System for Fast MoE Inference, accessed May 7, 2026, [https://www.researchgate.net/publication/385529633\_HOBBIT\_A\_Mixed\_Precision\_Expert\_Offloading\_System\_for\_Fast\_MoE\_Inference](https://www.researchgate.net/publication/385529633_HOBBIT_A_Mixed_Precision_Expert_Offloading_System_for_Fast_MoE_Inference)  
38. SpecMoE: A Fast and Efficient Mixture-of-Experts Inference via Self-Assisted Speculative Decoding \- ResearchGate, accessed May 7, 2026, [https://www.researchgate.net/publication/403790594\_SpecMoE\_A\_Fast\_and\_Efficient\_Mixture-of-Experts\_Inference\_via\_Self-Assisted\_Speculative\_Decoding/download](https://www.researchgate.net/publication/403790594_SpecMoE_A_Fast_and_Efficient_Mixture-of-Experts_Inference_via_Self-Assisted_Speculative_Decoding/download)  
39. DICE: Staleness-Centric Optimizations for Parallel Diffusion MoE Inference \- ResearchGate, accessed May 7, 2026, [https://www.researchgate.net/publication/404412518\_DICE\_Staleness-Centric\_Optimizations\_for\_Parallel\_Diffusion\_MoE\_Inference](https://www.researchgate.net/publication/404412518_DICE_Staleness-Centric_Optimizations_for_Parallel_Diffusion_MoE_Inference)  
40. Understanding Mixture of Experts (MoE) Neural Networks \- IntuitionLabs, accessed May 7, 2026, [https://intuitionlabs.ai/articles/mixture-of-experts-moe-models](https://intuitionlabs.ai/articles/mixture-of-experts-moe-models)  
41. \[2503.04398\] Semantic Parallelism: Redefining Efficient MoE Inference via Model-Data Co-Scheduling \- arXiv, accessed May 7, 2026, [https://arxiv.org/abs/2503.04398](https://arxiv.org/abs/2503.04398)  
42. Semantic Parallelism: Redefining Efficient MoE Inference via Model-Data Co-Scheduling, accessed May 7, 2026, [https://arxiv.org/html/2503.04398v4](https://arxiv.org/html/2503.04398v4)  
43. Semantic Parallelism: Redefining Efficient MoE Inference via Model-Data Co-Scheduling, accessed May 7, 2026, [https://openreview.net/forum?id=MSHPrMpIHZ](https://openreview.net/forum?id=MSHPrMpIHZ)  
44. MoE-Gen: High-Throughput MoE Inference on a Single GPU with Module-Based Batching, accessed May 7, 2026, [https://www.researchgate.net/publication/389821744\_MoE-Gen\_High-Throughput\_MoE\_Inference\_on\_a\_Single\_GPU\_with\_Module-Based\_Batching](https://www.researchgate.net/publication/389821744_MoE-Gen_High-Throughput_MoE_Inference_on_a_Single_GPU_with_Module-Based_Batching)  
45. MoE-Gen: High-Throughput MoE Inference on a Single GPU with Module-Based Batching, accessed May 7, 2026, [https://arxiv.org/html/2503.09716v1](https://arxiv.org/html/2503.09716v1)  
46. BlendServe: Optimizing Offline Inference for Auto-regressive Large Models with Resource-aware Batching \- Semantic Scholar, accessed May 7, 2026, [https://www.semanticscholar.org/paper/BlendServe%3A-Optimizing-Offline-Inference-for-Large-Zhao-Yang/8bedbe4bdfb12a93add15db00c691b129d0c2b9f](https://www.semanticscholar.org/paper/BlendServe%3A-Optimizing-Offline-Inference-for-Large-Zhao-Yang/8bedbe4bdfb12a93add15db00c691b129d0c2b9f)  
47. Large Language Model (LLM) \- Awesome Papers, accessed May 7, 2026, [https://paper.lingyunyang.com/paper-list/systems-for-ml/llm](https://paper.lingyunyang.com/paper-list/systems-for-ml/llm)  
48. DALI: A Workload-Aware Offloading Framework for Efficient MoE Inference on Local PCs, accessed May 7, 2026, [https://arxiv.org/html/2602.03495v1](https://arxiv.org/html/2602.03495v1)  
49. Serving Hybrid LLM Loads with SLO Guarantees Using CPU-GPU Attention Piggybacking \- arXiv, accessed May 7, 2026, [https://arxiv.org/html/2603.12831v2](https://arxiv.org/html/2603.12831v2)  
50. Delta Decompression for MoE-based LLMs Compression \- GitHub, accessed May 7, 2026, [https://raw.githubusercontent.com/mlresearch/v267/main/assets/gu25c/gu25c.pdf](https://raw.githubusercontent.com/mlresearch/v267/main/assets/gu25c/gu25c.pdf)  
51. Sub-MoE: Efficient Mixture-of-Expert LLMs Compression via Subspace Expert Merging, accessed May 7, 2026, [https://ojs.aaai.org/index.php/AAAI/article/view/39464/43425](https://ojs.aaai.org/index.php/AAAI/article/view/39464/43425)  
52. zhixin612/awesome-papers-LMsys: Daily Arxiv Papers on LLM Systems \- GitHub, accessed May 7, 2026, [https://github.com/zhixin612/awesome-papers-LMsys](https://github.com/zhixin612/awesome-papers-LMsys)  
53. SAGA: Workflow-Atomic Scheduling for AI Agent Inference on GPU Clusters \- arXiv, accessed May 7, 2026, [https://arxiv.org/html/2605.00528v1](https://arxiv.org/html/2605.00528v1)  
54. SOSP 2025 \- Awesome Papers, accessed May 7, 2026, [https://paper.lingyunyang.com/reading-notes/conference/sosp-2025](https://paper.lingyunyang.com/reading-notes/conference/sosp-2025)  
55. Networking-Aware Energy Efficiency in Agentic AI Inference: A Survey \- arXiv, accessed May 7, 2026, [https://arxiv.org/html/2604.07857v1](https://arxiv.org/html/2604.07857v1)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAYCAYAAAAs7gcTAAAAyklEQVR4Xu3RsQtBURTH8SMpokRSVslgYDAbKMrfIKtZFgMjo1JmyYBRFrNdKYPJ5A+wKJOB73n3vZIno+n96jPczjm3+84T8fLv+FFEDWH4kEUVobc+iWKJDno4YowhplgjqI16wwAla0wkhQtWyOOKHSJajKMv9iQp4IYGAmgiZ9dc0aa7mPf/jD5phj1iHzUr+nETtJDAScyADmp0G1qzUscTI5TxQNeu6UVzZOyzpHHAAhu0cRazsi0qTqMT3URSzI/5dvbiygvC9RzA6VnpHQAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACkAAAAYCAYAAABnRtT+AAAC00lEQVR4Xu2WS8hNURTH/0J5hoiESCKPkmSEkijyDANFBgyY+PJ+ZXBLJgqFIimPiRQDSSRhRshjwIAUEiMToVD4/629791n3fOde9zrK4PvX7/uuXvvs8/aa6+19gY69f+qH+nuGxuoKxlAuviOMupDZpEVZDxsMqk3GR6eUy0gB/H3Rsq4rWRzeC6lCeQu+UwukC3kHLlBJpJrZE51tGkKuU0Gu/ay0sL0jeW+w0sD95LvZBfpme3GTPKJvEPWkxp3maxM2prRJJhzRviOKBl4nPxA+6vpQa4G9Bw1jzwlQ5K2ZtSNnCcV117VBvKL7ENxXJwle5L/GnuaHEnaWtEq8gCWSBmNIe/JSxS4OugUsvE4iDwjy5I2L825MPxKA8l8MrQ6oqbJ5A2Z5jsqMC+W8YYvMVPJ2/DrJS9vgoXRGvKcHIAlyA6YU8ZVR5tkuIzMLFpl5g75ifqMLaNF5AMZ7TuoubBEjOGjUFHSqZwprr+gfnHRnp1pY7T8I+zlVJpcW6MxKf2TMTJS7+dtXRsZG55VW2+Si7AEmU6WoFZ7o6KR+9PGaGTehzSxPHEPVpYUEo/JumRMkZGp5Gl5POOhHEUjFftVxcAv+lB8MW9byxq5mHwjM3yHU+52a0uPwWJS9S5PCgOFg6+Pkj6qyqCsTKV515ITsA8rKV+TYaFfJ9Nh0iv8j1LpUQnSKZfRSPKCPEH9maxMPgnb6kycBGkBMtInnQr7K1iRVwY/gnlIBmsB22E10Uvf13t+vj9SgN8nX8kZ2AQy6iEsAbaR2XFworg96127FqdsvkKuk40wDylxLpHdYYyXdkbhFz1eJ61wFFkKu/noMpE3kVcFtaxNpczVtsYQ8f/zVIEdjX6ulqUTS6GiC0IrUrm7BbvIdIh0HzyE4nO/kVbDEq3M7jUlTXwU7d+gGkm7oXuqkrhDpeKvW5SvpY3UF5akvrJ06p/qN9cMe0Mb8U8FAAAAAElFTkSuQmCC>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAXCAYAAADduLXGAAAA4UlEQVR4XuXSoYpCQRTG8SMquKigGA0iirDNBzBqsrnR4AtYtBjFKJpsxu2iGOyC0bppk0Fs+wIK6v/cOwMy94p1wQ9+cOfMwJk5XJF/mQhyyLgbbsY444a+sxeaFi6ouRthmeGAvFMPJI0dNkg4e4F84g8Ds9bHVtDAhz1k08YVdcQxwgRrCXmwvW8JQ1TFPxSYThZ7/GAu/pU0eo0ekmbt5XFkZfxiJU8e6o7sW/xO2rGJjqlLClssEDM1PaxrncIURVOXAk7o2gL5whFLU9cxetEPbRe1BRPt+PKHet/cAcfeIy832IBiAAAAAElFTkSuQmCC>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAZCAYAAADuWXTMAAAA/klEQVR4XmNgGAWeQPyfSFwE1YMBFgLxbyC2QRNnBGIjIH4IxEFocmAgCMSngfgBEEujSsHBHCB2QRcEAX0g/gTEa4CYBSoGog2AmBXKnwhVhwGiGSB+KkcSA7kAZBs3lA9ysgBCGgFAiv4AcQAQSwKxPBDPBOJWZEXYAMy/f4H4CRA/AuJXUD5WPyIDYyD+yoDqX14gXgXESlA+M1QMA1CkGRZYyAlAHIgnAzEHlB8BxFkIaQgAJYD5DNgTBwzwAPFiIFZEl4AF1l0GiG3YQAwDxBUgi1AANv/CACh+K4D4NRBbIkuAouAZAyLBI0cTCP9CktsBxJwQbaNgCAEA3l87nMQUjqsAAAAASUVORK5CYII=>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmwAAAA4CAYAAABAFaTtAAAEp0lEQVR4Xu3dTahtUxwA8L98RL7ykY/eKx+RRCGhVzIyIJGYEEbyUZgQIupGSuoZ8EqJhEwwIJReBhcDLxMGRKSQMmKgKMrH+rf27uyz3rn37uvd457c36/+7X3W2me3zxn9+6+114oAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACADbetxAclTm07AABYHA+2DQAALI7tJfaUuL7E7hKHTncDALDZLi3xc4mDSxzW9AEAsADeKnFV1Hlst5U4YLobAIDN1g+BHjjVCgAAAAAAAAAAAAAAbBknl/imxN8l9pvu2suJJa4s8WeJH0qcMt0NAMC8XBs1YcvjGEdEvf7OtgMAgPl5L2oSluuvjZHVuHvbxgX1WNsww+slTmobAQAWzV9Rk7aL2o4FsBzTG9FfEXVYdi1nxbih20xAd7WNAACL5tmoCdsXbccC+DGmt8jK89cGn2fpk7C15ub1dsS45A4AYEWXlNi/O88k5KhB30a5MGrSdmPbMUI+Wz5jvzvC4VGf87TuuC/ymXpPRt3j9JpB2yy5H2q+UDGUz9Y/4/kx+T97rzSfAQBGy/08t5VY6j7nvKxZQ4LfrxLvDq5bzatRE6Sc17YeD5X4MCZJz+clji1xX0yGM2/pjuuR3/0j6m/IDenPnO5e0YtR56b1DomaPGai9mvUhO6FmE4mvxucAwCMlhuvP1ziuqjDdumdLuYhJ99nwjZmqY+hTKR+i/qcWf3LxC2/vz1q4pae645DWe3a0zYO5Cb0/YsDea+8d36n3+90JZmwZfQyWUuXl/g96vfzfkM59AoA8K99OzjPtdCyQtTKtdJWiuMG161lZ4mv2sY15LyyrGhlgpnDlRd07ZnApYujJkuznNM2dPKeyzH9wkF6ucSRURPCPH8kagVtqE3YessxXXkbUmEDAPbJL90xK2BZXct5XPOQ9x+7JttQVtH6SlgOq/ZJ1gMlji5xQokburaxsro2nL+Wno7JywGfdse8byaKN3WfUw4j57Bs7+4SN0e9Xy4AnL/zmEF/mlfVEgDYIp4p8XyJL0vc3/RtpJyHtp6h0KGfor69eVeJj6JWuLISljLJ6hOtMW6PydBszl/LOXtZWRwO1X7WHZ/qjkM5V+3rwees8OUw7aNRE7k3B30pE755/q8AwP/ceVHnf+UQ4e7Ye/hvI+S8sFzao3/Lc6NldSurbRvpiaiL/X4SdY5anxz28m3Xsf9VXjuv3w4AbAE5QT4rVlm5GpuArEdWrN6POmw5Rp/cbaYczrws6jBmDp1mwnX61BVVW0mbZanEPW0jAMAiyTlrOadrrFwL7eO28T+WSeNSiTOa9la+vXpQ29i4Nf79MDAAwNzlUOIdsfdbpcM4O+p6am9EnUOW65jlHDEAAOYsq2q5fEc/uX9s5CK88xiaBQAAAAAAAAAAYGvaFXWdt5Xkemdvx+rXAAAwJ1eXOLdtnCG3dJKwAQBsgtyUfak7X24iq2r9WmcSNgCATbIjJts6tWuwHR+TLZskbAAAm2Rn1G2vVlvt//Go22O91HYAADB/maitlqwBAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAsC7/AA+QodXyz5MfAAAAAElFTkSuQmCC>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAYCAYAAAAlBadpAAABKUlEQVR4Xu3SwStEURTH8SMspGkSVnYyWSiUFMlO9mZD2VgxUpbWI8k/MDsLWYj/gJpZKNKUxdjYKNmQZEWxEc33vHPfu9cz7MWvPot73n3nde67Iv/5q2nHNEbQnHr2bXTjKi6wiE3UcINcsO9LmrCGW/S5WhuOcIkuV2uYATyiJNZI04Fz7AW1QcwF6yhFfGAqqA3hGUtBbVlspCTaRbvfozeoz+MVo0GtYXZxgoxba8Md8fPqH1jHNrJuT5ICrtEj9uIs3sXPO4NhHGLcvZOkFVu4wxUO8CZ+Xv1VYyiLHeSP0XlfxC5KnKLzKROu2O3WLdhHVfx8Ovep2NdXxO5AlAoe0C823wKekI83kE6xfRuYDOrRPT7DsdjV1FPWRunoueip//bUATLjL/bxaLQcAAAAAElFTkSuQmCC>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAAXCAYAAAALHW+jAAABMElEQVR4Xu3TvSuFYRjH8UsYFMlLXspCFpNBBuVttRhYiE0yWYmsNkpGiwwKx2IyWIQymP0BFibZzL6/ruvJ8zzn0aljUc6vPp3u6z7nfj9mtdTyd1KPEUyiMVWvQ0t8KoMYTrULowEOsI077Kf6ZvFhPkgnnvGGgdR3yjKNHbTiHqf2vYJD80E0mGobeLUKA65iCGP4xELU2/Bk2Qn6cG0+gaLd6feqZ9KMW1yiIWpz5hOMRlvRZIup9o/pxYv5OSbZjJr6FE20h/5oj+MKM9HOJDnw3WhrKxeWvYAJbJlvvwvLmMe5+Sspi7b4jhJusI4HPOIER+YXp7SjB8dYilphtLJu8zNVtJqOkH972romTI7g11kzX6Fex1Sur6qs4Mz8XJtyfVVHf8vCC/lH+QIkWyoLAb3oEwAAAABJRU5ErkJggg==>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA0AAAAYCAYAAAAh8HdUAAAAtUlEQVR4XmNgGP7AEYhfA/F/JPwLiHcDsTCSOqxgDhD/A2IPdAlcQBCITwPxAyCWRpXCDTSB+C0QrwFiFjQ5nCCaAeKXcnQJfGASEP8GYht0CVwA5p+7QCyOJocTEPIPGxCzogvC/FOELgEEjEDcBMQ66BKg+MHlHxUgngvEnMiC+OIH5KRZDBCXoABjIP7KgOkfSQaIhkdArAgTdAHiZwyItPYXiJ9AMYgNE1/OgD1wRgHtAQAv+ie2Ic8IBwAAAABJRU5ErkJggg==>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC8AAAAYCAYAAABqWKS5AAABUUlEQVR4Xu2WsUoDQRCGJ6CiXUQwiJWQJiBIsEiTRggqErt0eQFrCwVLKwsr7UJSpwn4AJYp8wI2gojYiOnSKEb/yd7h3bC521vMImQ/+CDM7HI/e3NLiDye+WYPvsHviB/wHq5F1rmmDPdlcRptOIaHsuGQCjyDA1KHeB5v61klteEJbsZbVlzDHVk0gMMfwyM4IsPwJfgOe3BB9Gy4hbuymAHeaxy+SRlekwFOw9/AT1iVDUuchQ/n/REWRM8WZ+HT5n0JLspiwDLc0NiBB5o6X725yc5kjMOH834qG6QedAm3ZSOgAVsaH+Cdpn4F85OdyRiH5/t92rwXSZ3iimyk4GRsku53HhU+LX4zWXESPlwk553nk4M/w61I3ZS/Cn8hG0wNvtLvf5kv+BLIv8N6l/QfcRq24U8onosdwj5cj6ybKbbh/wV1UqPn8XjmmR8B7FPVlVg9hQAAAABJRU5ErkJggg==>