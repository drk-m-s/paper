# **Advanced Systems for Mixture of Experts Offloading: A Technical Analysis of High-Performance Inference on Resource-Constrained Hardware (2024–2026)**

The rapid transition from dense transformer architectures to sparse Mixture of Experts (MoE) models represents one of the most significant shifts in large-scale language modeling. By utilizing a gating mechanism to activate only a small subset of parameters—experts—for each input token, MoE architectures such as Mixtral, Qwen, and DeepSeek enable the deployment of models with massive total parameter counts while maintaining a relatively low computational footprint per token.1 However, the primary challenge of MoE deployment is the memory requirement. While the per-token computation is sparse, the entire set of expert weights must typically reside in memory to avoid the prohibitive latency of on-demand loading from secondary storage.3 In resource-constrained environments, such as those utilizing consumer-grade GPUs with 16GB or 24GB of VRAM, the total parameter count of modern MoEs far exceeds the available hardware capacity.4

The research landscape between 2024 and 2026 has focused on overcoming this memory wall through advanced offloading engines. These systems move beyond simple reactive caching to employ predictive prefetching, speculative execution, and heterogeneous CPU-GPU orchestration.6 This report provides a comprehensive evaluation of these systems, categorizing the methodologies that enable efficient inference of large MoE models on restricted hardware.

## **The Landscape of MoE Offloading Research (2024–2026)**

The following table categorizes the seminal research papers in the field of MoE offloading and inference optimization for resource-constrained environments. These papers have been selected based on their technical novelty, impact on system performance, and relevance to the design of high-efficiency offloading engines.

| Paper Name | Publication Time | Citations (Approx.) | Importance (1-5) | Core Technical Contribution |
| :---- | :---- | :---- | :---- | :---- |
| **ExpertFlow: Efficient Mixture-of-Experts Inference via Predictive Expert Caching and Token Scheduling** 7 | July 2026 (DAC) | 32+ | 5 | Global routing path prediction and token re-batching to maximize expert reuse. |
| **CommitMoE: Efficient Fallback-Free MoE Inference with Offloading Under GPU Memory Constraints** 2 | March 2026 (AAAI) | 15+ | 5 | Eliminates fallback penalties through a "Commit Router" that unconditionally adopts predicted experts. |
| **FineMoE: Taming Latency-Memory Trade-Off in MoE Serving** 9 | April 2026 (EuroSys) | 10+ | 5 | Fine-grained expert maps and semantic hint integration for optimized prefetching. |
| **PreScope: Unleashing the Power of Prefetching for Resource-Constrained MoE Inference** 11 | 2026 | 5+ | 5 | Layer-aware predictors (LLaPor) and cross-layer optimal scheduling (PreSched). |
| **Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models** 8 | May 2025 (ICLR) | 93+ | 4 | Utilizes CPU compute power for experts to avoid weight transfer latency for single batches. |
| **Speculating Experts Accelerates Inference for Mixture-of-Experts** 6 | March 2026 | 5+ | 4 | Parameter-free prefetching using internal hidden representations as speculative signals. |
| **SMoE: An Algorithm-System Co-Design for Pushing MoE to the Edge via Expert Substitution** 15 | April 2026 | 5+ | 4 | Substitutes low-importance active experts with functionally similar experts already in GPU cache. |
| **Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference** 16 | June 2024 (ISCA) | 138+ | 4 | Structural lookahead gating to overlap prefetching with current layer computation. |
| **DALI: A Workload-Aware Offloading Framework for Efficient MoE Inference on Local PCs** 18 | 2026 | 5+ | 3 | Dynamic expert assignment using greedy 0-1 integer optimization based on workload. |
| **ADEPT: Adaptive Domain-aware Expert Prefetching Technique** 5 | 2025 | 10+ | 3 | Two-stage optimization for prefill loading and decoding-phase prefetching. |
| **MoE-Infinity: Activation-Aware Expert Offloading for Efficient MoE Serving** 20 | 2024 | 53+ | 3 | Sequence-level activation tracing to identify sparse activation temporal locality. |
| **FloE: On-the-Fly MoE Inference on Consumer GPUs** 22 | 2025 (ICML) | 5+ | 3 | Hybrid compression and dual predictors for inter- and intra-expert sparsity. |
| **SiftMoE: Energy-Efficient Expert Selection for Wireless Distributed MoE** 23 | March 2026 | 5+ | 3 | Theoretical bounds for expert replacement/skipping in distributed edge networks. |

## **The Bottleneck of PCIe Bandwidth in Offloading Engines**

A central theme across all research snippets is the disparity between GPU memory bandwidth and the PCIe interconnect bandwidth. While high-end GPUs like the H100 or RTX 4090 possess memory bandwidths exceeding 1 TB/s, the PCIe 4.0/5.0 x16 interface provides only 32–64 GB/s of bidirectional throughput.11 For a 10-billion parameter MoE model (such as a 10B subset of a larger architecture), the expert weights for a single layer can exceed 1 GB in a quantized 4-bit or 8-bit format.12

The mathematical delay introduced by on-demand loading is often several times larger than the actual computation time on the GPU. If the expert weight transfer time ![][image1] exceeds the GPU computation time ![][image2], the system enters a memory-bound state where the GPU sits idle, creating "bubbles" in the pipeline.24 Existing studies indicate that for models like Mixtral-8x7B, PCIe transfers can account for over 60% of total inference time if not properly optimized.25

## **Predictive Prefetching Strategies**

To mitigate the bubble caused by PCIe latency, prefetching engines attempt to load experts into GPU VRAM before they are officially selected by the gating network.

### **Internal Representation and Speculative Signals**

Research into **Speculating Experts** has identified that the internal hidden states (![][image3]) of a transformer model contain high-fidelity signals that can predict the routing decisions of future layers.6 By leveraging these representations, an inference engine can speculate which experts will be required in layer ![][image4]. Across multiple architectures, including Qwen3-MoE and GLM 4.7, it has been demonstrated that future experts can be reliably predicted without training separate auxiliary models.6

In scenarios where representations drift significantly—a phenomenon observed in deep models or specific layer groups—lightweight neural estimators can be employed. These estimators require only a small number of training tokens to significantly improve expert hit rates over specific layers.14 For example, the YALIS engine achieves up to a 14% reduction in Time Per Output Token (TPOT) by overlapping these speculative transfers with ongoing computation.6

### **Structural Lookahead with Pre-gated MoE**

The **Pre-gated MoE** framework proposes a more radical modification of the MoE block.16 In a standard MoE block, the gating function ![][image5] determines the active experts for the *current* layer based on the *current* input hidden state. This creates a hard sequential dependency: selection must occur before execution.16 Pre-gated MoE modifies the architecture so that the gate function for layer ![][image6] is executed within the ![][image7]\-th MoE block using the input to that block.2

This architectural shift allows the system to identify the set of experts for the next block in advance, effectively prefetching them to the GPU while the ![][image7]\-th block is still executing.16 Experimental results show that this reduces peak GPU memory consumption by 4.2x compared to keeping all experts resident, with only a 23% performance overhead compared to an oracle GPU-only solution.16 However, this method requires model retraining, making it less suitable for users deploying pre-trained weights without access to training infrastructure.2

### **Layer-Aware Predictors: PreScope**

The **PreScope** system builds on the insight that expert activation patterns are not uniform throughout the model. They exhibit distinct "group-wise" patterns: near-input layers, near-output layers, and middle layers each possess unique activation characteristics.11 PreScope introduces the **Layer-Aware Predictor (LLaPor)**, which uses distinct architectures (e.g., shallower networks for noisier near-input layers and deeper residual blocks for middle layers) to capture these patterns.26

LLaPor utilizes both inter-layer routing correlation and adjacent-layer hidden state statistics to achieve a Top-4 prediction accuracy exceeding 90%.12 This high accuracy is critical for the system's scheduling logic, which must balance prefetching against the limited PCIe bandwidth.

## **Speculative Execution and the Commit Paradigm**

A major drawback of prefetching is the penalty of misprediction. If the prefetcher loads expert ![][image8] but the router selects expert ![][image9], the system must initiate an on-demand load of ![][image9]. Because CUDA memory transfers are typically serialized, the incorrect transfer cannot be canceled once started, leading to an additional 13.7%–47.6% latency overhead.2

### **Fallback-Free Inference with CommitMoE**

**CommitMoE** addresses this by introducing the **Commit Router**.2 The research reveals a fundamental system-level insight: MoE router certainty strongly correlates with prediction accuracy. More importantly, in scenarios where the router exhibits low certainty (i.e., multiple experts have similar scores), the final model output is inherently robust to the specific expert selected.2

Leveraging this robustness, CommitMoE adopts a "fallback-free" approach. The system unconditionally uses the experts identified by its lightweight predictor for token computation.28 By committing to the prediction, the engine eliminates the need for any on-demand loading path. This radical design yields inference speeds 1.3x to 9.4x faster than state-of-the-art offloading frameworks like Fiddler or DeepSpeed-Inference while maintaining the quality of the model output.2

### **Expert Substitution in SMoE**

A complementary strategy is **Expert Scheduler with Substitution (SMoE)**. SMoE identifies "low-score" active experts—those selected by the router but contributing marginally to the output—and substitutes them with "functionally similar" experts that are already resident in the GPU cache.15

This substitution method is particularly effective for fine-grained MoE models (e.g., DeepSeek-MoE, Qwen2-57B-A14B), where a larger number of experts are activated per token, but only a few achieve high activation scores.15 By substituting a subset of active experts with those already in VRAM, SMoE eliminates a significant portion of PCIe traffic. In evaluations on the Qwen2 architecture, SMoE reduced decoding latency by 48% and maintained a cache hit rate above 60%.15

## **Heterogeneous Orchestration: Using the CPU for Computation**

When GPU memory is severely constrained (e.g., trying to run a 90GB Mixtral model on a single 24GB GPU), even the most efficient prefetching may fail to keep up with the generation speed if the weights are constantly moved over PCIe.

### **Activation Offloading: Fiddler**

The **Fiddler** engine introduces the concept of **CPU-GPU orchestration**. The key innovation is the realization that for small batch sizes (typical of local, single-user inference), the size of the activation values is significantly smaller than the size of the expert weights.8

For a model like Mixtral-8x7B:

* **Expert Weights**: Three matrices of size ![][image10] (roughly 176 million parameters per expert).  
* **Activations**: A vector of size ![][image11] (for batch size 1).

Fiddler develops a latency model to decide at runtime whether it is faster to:

1. Load the expert weights from CPU memory to GPU (standard offloading).  
2. Copy the activation values from the GPU to the CPU, execute the expert layer on the CPU, and return the output activation to the GPU.13

Despite the CPU's slower computation speed compared to the GPU, avoiding the massive weight transfer over PCIe makes the second option significantly more efficient for single-batch requests.8 Fiddler can run the uncompressed Mixtral-8x7B model at over 3 tokens per second on a single 24GB GPU, outperforming existing methods by an order of magnitude.30

### **Hybrid Scheduling: HybriMoE and DALI**

**HybriMoE** and **DALI** extend the collaborative concept to multi-expert assignment. **HybriMoE** introduces a dynamic intra-layer scheduling strategy that balances workloads between the CPU and GPU.11 It uses a queuing mechanism to prioritize "hot" (frequently used) experts on the GPU and execute "cold" experts on the CPU.11

**DALI** models the assignment problem as a **0-1 integer optimization problem**, solved at runtime using a greedy strategy to maximize hardware utilization.18 DALI also introduces a "workload-aware cache replacement" policy that exploits temporal correlation in expert activations, drastically improving cache hit rates over traditional LRU policies.18

## **Algorithmic Optimizations: Token Scheduling and Batching**

When serving multiple requests or dealing with long sequences, the routing decisions of individual tokens can be coordinated to improve system efficiency.

### **ExpertFlow and Token Re-batching**

**ExpertFlow** is a comprehensive runtime system that integrates global prediction with token-level scheduling.7 It utilizes a transformer-based **Routing Path Predictor (RPP)** to estimate the expert usage for all tokens across all layers in a single forward pass.7

A core innovation in ExpertFlow is the **Token Scheduler (TS)**. Given multiple batches of input tokens, the TS reorders tokens across batches based on their predicted routing paths.7 This "expert-wise co-location" groups tokens that require the same experts into the same batch. This strategy has two primary benefits:

1. It reduces the total number of unique experts activated per batch, directly lowering the memory transfer requirement.  
2. It increases the computational load per expert, allowing the GPU to process tokens more efficiently once weights are loaded.7

Mathematical formulation of the re-batching objective in ExpertFlow: Minimize ![][image12] where ![][image13] indicates if expert ![][image14] at layer ![][image15] is activated in batch ![][image16].7 By approximating this with K-means clustering, ExpertFlow reduces GPU memory usage by up to 93.72% and improves throughput by 10x over strong offloading baselines.7

### **FineMoE and Semantic Integration**

**FineMoE** approaches the problem from a semantic perspective. It observes that expert selection is not random but correlates with the input context.1 FineMoE extracts **semantic hints** from the input prompts and uses them to search a new data structure called the **expert map**.1 The expert map records iteration-level probability distributions from previous runs.1

By combining trajectory-based information with input semantic embeddings, FineMoE can predict expert selection at a finer granularity than traditional systems.9 This semantic-aware prefetching leads to a 39% improvement in the expert cache hit rate and a significant reduction in the time-to-first-token (TTFT).9

## **Quantitative Performance Comparison**

The table below summarizes reported performance metrics for the various MoE offloading engines on common MoE architectures.

| Engine | Target Model | Hardware Context | Key Performance Metric | Reported Gain |
| :---- | :---- | :---- | :---- | :---- |
| **ExpertFlow** 7 | Switch-128 / Mixtral-8x7B | Single 24GB GPU | Throughput / Memory | 10x speedup; 93.72% memory reduction. |
| **Fiddler** 13 | Mixtral-8x7B | Single 24GB GPU | Latency (Single Batch) | 3+ tokens/sec; 8.2x–10.1x faster than baselines. |
| **CommitMoE** 2 | Mixtral / Qwen | GPU constrained | Decoding Latency | 1.3x–9.4x faster than offloading frameworks. |
| **PreScope** 11 | Mixtral / DeepSeek | Single commodity GPU | End-to-end Throughput | 141% higher throughput than Klotski/HybriMoE. |
| **SMoE** 15 | Qwen2-57B-A14B | Edge/Consumer GPU | Decoding Latency | 48% reduction in latency; 60%+ hit rate. |
| **FineMoE** 1 | Mixtral-8x7B | 6-GPU Testbed (6GB/GPU) | TPOT / Hit Rate | 36% TPOT reduction; 39% improvement in hit rate. |
| **MoE-Infinity** 20 | Various MoEs | Memory Constrained | Latency | 4x–20x reduction in latency. |
| **ADEPT** 5 | Mixtral / Phi-3.5 | Single GPU | Speedup / VRAM | 3.45x speedup; 50%+ reduction in peak memory. |

## **Designing a Custom MoE Offloading Engine: Practical Insights**

For a practitioner building a MoE offloading engine on a resource-constrained GPU, the research suggests a multi-layered design approach.

### **1\. Memory and Cache Management**

Static caching (LRU) is insufficient for the dynamic routing patterns of MoE. The system must employ a **workload-aware cache** (as in DALI) or an **expert map** (as in FineMoE) to retain high-probability experts in VRAM.1 Utilizing **activation-aware prefetching** (MoE-Infinity) helps in identifying the temporal locality of activations across subsequent tokens in a sequence.20

### **2\. The Predictive Layer**

Implementing a predictor is non-negotiable for low-latency inference. While structural changes (Pre-gated MoE) are optimal for performance, they are difficult to implement on pre-trained models.16 A more practical solution is to use **internal hidden representations** as speculative signals, which requires no additional training and can be integrated into existing inference engines like YALIS.6

### **3\. Handling PCIe Contention**

The engine must prioritize between on-demand loads (critical path) and prefetches (speculative path). **PreScope’s PreSched** logic is a robust model for this: it uses a cross-layer cost model to decide if a prefetch is worth the risk of delaying a current-layer on-demand load.12

### **4\. Hybrid Execution**

If the GPU memory is too small to even hold the "hot" experts, the system should incorporate **activation offloading** to the CPU.30 This requires optimized CPU kernels (e.g., using AVX-512 or AMX) for the feed-forward network layers that comprise MoE experts.13

## **Advanced Themes: Compression, Quantization, and Edge Deployment**

Beyond offloading, the research explores how compression can make offloading more efficient. **FloE** integrates contextual sparsity with ultra-low-bit quantization (e.g., INT2) to reduce the weight transfer size.22 However, uniform quantization often degrades model performance. FloE addresses this with a hybrid strategy: contextual activation sparsity for gate and down projections, and INT2 quantization for up projections.22

**SiftMoE** explores the theoretical limits of expert selection in wireless edge environments.23 It develops bounds on accuracy degradation when experts are skipped or replaced by similar counterparts, allowing for energy-efficient inference where the cost of data movement is exceptionally high.23

## **Conclusion: The Path Forward for MoE Offloading**

The evolution of MoE offloading engines from 2024 to 2026 marks a shift toward systemic intelligence. We have moved from simple demand-based loading to engines that "understand" the model's internal states, the semantic nature of the prompts, and the hardware's heterogeneous capabilities.1

The most successful systems share a common trait: they do not treat the model and the hardware as separate entities. **CommitMoE** uses model robustness to solve a systems-level latency problem; **ExpertFlow** uses routing statistics to solve a batching problem; and **Fiddler** uses the data-size ratio to solve a communication problem.2 For a developer building a MoE offloading engine today, the integration of these insights—speculative commitment, global token scheduling, and activation-based CPU orchestration—is essential for achieving high-performance inference on the ever-present constraints of consumer hardware.

#### **Works cited**

1. Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading \- arXiv, accessed May 6, 2026, [https://arxiv.org/html/2502.05370v2](https://arxiv.org/html/2502.05370v2)  
2. CommitMoE: Efficient Fallback-Free MoE Inference with Offloading Under GPU Memory Constraints, accessed May 6, 2026, [https://ojs.aaai.org/index.php/AAAI/article/view/39454/43415](https://ojs.aaai.org/index.php/AAAI/article/view/39454/43415)  
3. Accelerating Edge Inference for Distributed MoE Models with Latency-Optimized Expert Placement \- arXiv, accessed May 6, 2026, [https://arxiv.org/html/2508.12851v4](https://arxiv.org/html/2508.12851v4)  
4. Efficient CPU-GPU Collaborative Inference for MoE-based LLMs on Memory-Limited Systems \- arXiv, accessed May 6, 2026, [https://arxiv.org/html/2512.16473v1](https://arxiv.org/html/2512.16473v1)  
5. Two-Stage Expert Offloading for Domain-Aware MoE Inference \- IEEE Xplore, accessed May 6, 2026, [https://ieeexplore.ieee.org/iel8/6287639/11323511/11397596.pdf](https://ieeexplore.ieee.org/iel8/6287639/11323511/11397596.pdf)  
6. Speculating Experts Accelerates Inference for Mixture-of-Experts, accessed May 6, 2026, [https://arxiv.org/abs/2603.19289](https://arxiv.org/abs/2603.19289)  
7. ExpertFlow: Efficient Mixture-of-Experts Inference via Predictive Expert Caching and Token Scheduling \- arXiv, accessed May 6, 2026, [https://arxiv.org/html/2410.17954v2](https://arxiv.org/html/2410.17954v2)  
8. Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models \- arXiv, accessed May 6, 2026, [https://arxiv.org/html/2402.07033v2](https://arxiv.org/html/2402.07033v2)  
9. Taming Latency-Memory Trade-Off in MoE-Based LLM ... \- Hao Wang, accessed May 6, 2026, [https://arxiv.org/abs/2502.05370](https://arxiv.org/abs/2502.05370)  
10. Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading \- IntelliSys Lab, accessed May 6, 2026, [https://intellisys.haow.us/assets/pdf/Hanfei\_FineMoE\_EuroSys26.pdf](https://intellisys.haow.us/assets/pdf/Hanfei_FineMoE_EuroSys26.pdf)  
11. PreScope: Unleashing the Power of Prefetching for Resource-Constrained MoE Inference, accessed May 6, 2026, [https://arxiv.org/html/2509.23638v1](https://arxiv.org/html/2509.23638v1)  
12. PreScope: Unleashing the Power of Prefetching for Resource-Constrained MoE Inference \- arXiv, accessed May 6, 2026, [https://arxiv.org/pdf/2509.23638](https://arxiv.org/pdf/2509.23638)  
13. FIDDLER: CPU-GPU ORCHESTRATION FOR FAST INFERENCE OF MIXTURE-OF-EXPERTS MODELS \- ICLR Proceedings, accessed May 6, 2026, [https://proceedings.iclr.cc/paper\_files/paper/2025/file/8cd1ce03ea58b3d7dfd809e4d42f08ea-Paper-Conference.pdf](https://proceedings.iclr.cc/paper_files/paper/2025/file/8cd1ce03ea58b3d7dfd809e4d42f08ea-Paper-Conference.pdf)  
14. Speculating Experts Accelerates Inference for Mixture-of-Experts \- arXiv, accessed May 6, 2026, [https://arxiv.org/html/2603.19289v1](https://arxiv.org/html/2603.19289v1)  
15. SMoE: An Algorithm-System Co-Design for Pushing MoE to the Edge via Expert Substitution, accessed May 6, 2026, [https://arxiv.org/html/2508.18983v3](https://arxiv.org/html/2508.18983v3)  
16. Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference \- Microsoft, accessed May 6, 2026, [https://www.microsoft.com/en-us/research/wp-content/uploads/2024/05/isca24\_pregated\_moe\_camera\_ready.pdf](https://www.microsoft.com/en-us/research/wp-content/uploads/2024/05/isca24_pregated_moe_camera_ready.pdf)  
17. ‪Ting Cao 曹婷‬ \- ‪Google Scholar‬, accessed May 6, 2026, [https://scholar.google.com/citations?user=HaP6JgIAAAAJ\&hl=en](https://scholar.google.com/citations?user=HaP6JgIAAAAJ&hl=en)  
18. DALI: A Workload-Aware Offloading Framework for Efficient MoE Inference on Local PCs | Request PDF \- ResearchGate, accessed May 6, 2026, [https://www.researchgate.net/publication/400415570\_DALI\_A\_Workload-Aware\_Offloading\_Framework\_for\_Efficient\_MoE\_Inference\_on\_Local\_PCs](https://www.researchgate.net/publication/400415570_DALI_A_Workload-Aware_Offloading_Framework_for_Efficient_MoE_Inference_on_Local_PCs)  
19. A Scheduling Framework for Efficient MoE Inference on Edge GPU-NDP Systems | Request PDF \- ResearchGate, accessed May 6, 2026, [https://www.researchgate.net/publication/399559182\_A\_Scheduling\_Framework\_for\_Efficient\_MoE\_Inference\_on\_Edge\_GPU-NDP\_Systems](https://www.researchgate.net/publication/399559182_A_Scheduling_Framework_for_Efficient_MoE_Inference_on_Edge_GPU-NDP_Systems)  
20. MoE-Infinity: Activation-Aware Expert Offloading for Efficient MoE Serving \- Hugging Face, accessed May 6, 2026, [https://huggingface.co/papers/2401.14361](https://huggingface.co/papers/2401.14361)  
21. MoE-Infinity: Activation-Aware Expert Offloading for Efficient MoE Serving \- ar5iv \- arXiv, accessed May 6, 2026, [https://ar5iv.labs.arxiv.org/html/2401.14361](https://ar5iv.labs.arxiv.org/html/2401.14361)  
22. ICML Poster FloE: On-the-Fly MoE Inference on Memory-constrained GPU, accessed May 6, 2026, [https://icml.cc/virtual/2025/poster/44378](https://icml.cc/virtual/2025/poster/44378)  
23. SiftMoE: Similarity-Aware Energy-Efficient Expert Selection for Wireless Distributed MoE Inference \- arXiv, accessed May 6, 2026, [https://arxiv.org/pdf/2603.23888](https://arxiv.org/pdf/2603.23888)  
24. LayerScope: Predictive Cross-Layer Scheduling for Efficient Multi-Batch MoE Inference on Legacy Servers \- arXiv, accessed May 6, 2026, [https://arxiv.org/html/2509.23638v2](https://arxiv.org/html/2509.23638v2)  
25. DALI: A Workload-Aware Offloading Framework for Efficient MoE Inference on Local PCs, accessed May 6, 2026, [https://arxiv.org/html/2602.03495v1](https://arxiv.org/html/2602.03495v1)  
26. \[Literature Review\] PreScope: Unleashing the Power of Prefetching for Resource-Constrained MoE Inference \- Moonlight | AI Colleague for Research Papers, accessed May 6, 2026, [https://www.themoonlight.io/en/review/prescope-unleashing-the-power-of-prefetching-for-resource-constrained-moe-inference](https://www.themoonlight.io/en/review/prescope-unleashing-the-power-of-prefetching-for-resource-constrained-moe-inference)  
27. Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference \- ResearchGate, accessed May 6, 2026, [https://www.researchgate.net/publication/373333237\_Pre-gated\_MoE\_An\_Algorithm-System\_Co-Design\_for\_Fast\_and\_Scalable\_Mixture-of-Expert\_Inference](https://www.researchgate.net/publication/373333237_Pre-gated_MoE_An_Algorithm-System_Co-Design_for_Fast_and_Scalable_Mixture-of-Expert_Inference)  
28. CommitMoE: Efficient Fallback-Free MoE Inference with Offloading Under GPU Memory Constraints | Request PDF \- ResearchGate, accessed May 6, 2026, [https://www.researchgate.net/publication/402629996\_CommitMoE\_Efficient\_Fallback-Free\_MoE\_Inference\_with\_Offloading\_Under\_GPU\_Memory\_Constraints](https://www.researchgate.net/publication/402629996_CommitMoE_Efficient_Fallback-Free_MoE_Inference_with_Offloading_Under_GPU_Memory_Constraints)  
29. CommitMoE: Efficient Fallback-Free MoE Inference with Offloading Under GPU Memory Constraints | Proceedings of the AAAI Conference on Artificial Intelligence, accessed May 6, 2026, [https://ojs.aaai.org/index.php/AAAI/article/view/39454](https://ojs.aaai.org/index.php/AAAI/article/view/39454)  
30. GitHub \- efeslab/fiddler: \[ICLR'25\] Fast Inference of MoE Models with CPU-GPU Orchestration, accessed May 6, 2026, [https://github.com/efeslab/fiddler](https://github.com/efeslab/fiddler)  
31. FIDDLER: CPU-GPU ORCHESTRATION FOR FAST INFERENCE OF MIXTURE-OF-EXPERTS MODELS \- OpenReview, accessed May 6, 2026, [https://openreview.net/pdf?id=WX7lxohjFe](https://openreview.net/pdf?id=WX7lxohjFe)  
32. HybriMoE: Efficient Hybrid Inference | PDF | Cache (Computing) \- Scribd, accessed May 6, 2026, [https://www.scribd.com/document/924495066/2504-05897v1](https://www.scribd.com/document/924495066/2504-05897v1)  
33. HybriMoE: Hybrid CPU-GPU Scheduling and Cache Management for Efficient MoE Inference | Request PDF \- ResearchGate, accessed May 6, 2026, [https://www.researchgate.net/publication/395627980\_HybriMoE\_Hybrid\_CPU-GPU\_Scheduling\_and\_Cache\_Management\_for\_Efficient\_MoE\_Inference](https://www.researchgate.net/publication/395627980_HybriMoE_Hybrid_CPU-GPU_Scheduling_and_Cache_Management_for_Efficient_MoE_Inference)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAYCAYAAAAYl8YPAAABNElEQVR4Xu3UzysEcRjH8YelFOXXFkopEkpRW/6BpZwclHKycnB2U26Uiws5OEjJgVonlz3ZHCV3TspF+Rtw8X72eaYZ39rMpLTFp17tzDyz3515npkV+U9DZQ3X6A0LWdOGitPtH2UQL9gNC1nSgQEs4QPL6ENr8qS0KeEYz3jDOQ4xkjwpS361X0WXKrN4988wLWI/MhcW6mULrxgOC1kT9auKdrEp7ondej+OsOPHo+hw9nGFFeSiQh4PEvdrERtoxiqmcIshr8+ILdKDJpxi3mu1A9t4Qtm39Sq0V6NYwKXvd4otrMeinGEzsV9Ll0tGF7gQe5A1BTxK3Ntu3MvXxetmAneYxDqmcSPxH4FOWPutV/xtxsS+rAMYF7v9A7H+al9PxIaUOjrt8K3Q91n9xXwCxKstLvWzf9UAAAAASUVORK5CYII=>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAYCAYAAAAlBadpAAABEUlEQVR4XmNgGAUUgSQg3g3EwugShAAHEG+FYhCbJCADxE+AuBVdAh/gAWJJIA4F4t9AHAHE4kDMiqwIF4gH4llAfB+IfwLxUiCeBMTKyIrwAfr7FwZcgPgXlCYZVAHxcyBWQpcgBGD+3QPE3AyQUO5igHgFBBiB2A6IFwNxExA7ADEnVI5BBIivMiD8GwTEBQwQTSAQDMQzGCCGegDxeQaIHjAAKWoE4jtAvBLKhsWxLBBfAGIbKD8aiNcAMQuUDwcCUIwMPIH4CgPCJlD8pyOk8QOQjdsYIKkQlFkOQ8WIArDAmw/EmxnQ/EsKAAUkVv9iAyBnnmOAhDA/AyQ6/VBU4AEgJ4MKhwYoVkeWpBgAAIm9KRpmM5Z+AAAAAElFTkSuQmCC>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABIAAAAYCAYAAAD3Va0xAAABR0lEQVR4XuXUvytHURjH8UcoREiRKCmLhUE2WeSbhSRFmaT8iMlAmZRsFhZ/gU1JYjKY/BXUV8lmNEji/XHu1XNPwln51Gt5nnPv957nnL5m/yYVaEFT3EjJHt4y61EvOVN4wVDcSM0ByuiI6kmpxxWOUVVspaUXj9hEG8YxjGq/6DeZwCsucYhZXOMMtW7dj9F89KIZC9dA0dfdoT1f5FKJhriYz+fEilvZxa2FrfpozRZ20O8bfj55vhu+rskSpi08+xnN59mK92cAT5hzNaXRwtz6sGrR/LSFshXvzzbu0Y1RLGZ1/dgFShjLah/5agu+Vod99GQ9bV+numzR1ejCA9ZcTae2gRsLL5t09SMLV0TPFaJms4XjjKN/AX1dHh33OQZdLTnatrZ1ik4LH6Brkd+55CxgHisYiXpJ0QhaURM3/mjeAdNZMlOXe9OCAAAAAElFTkSuQmCC>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACoAAAAYCAYAAACMcW/9AAAB6ElEQVR4Xu2VPyhFURzHf0Ih5V+JSEiJkkEG/xeFkgyYLAYDZTEYDEpRBgyYRJksyCArb5LZIMVgUAZJKYsS369zz+ud8y73vnu72/3Up/fu+d13z++d8zu/KxITY1AGe2G+HYiYbFgOC+2AGy3wAi7DGSsWFQXwEH7BbzhkhtPhPzmHg7AdzprhyFmEz7DeDthMwitYBKfhiBn2ZB222oM+yRO1SAnx2Homdw1XRdXmHqwz7vBmG7bZgz6phk9wyw7Y9MNP57MHrsAs4w5vwiTKeVmjehd5sPisbud7Eq7kI2yEB7AhNeiTMIly/lfYJGp3d+GSqFJMHi7WRAKewAU4rgMZEjTR1PqsgRuiSoHlZ3QBnjKethtRk+XqwB/wwZUu7sMBl3H25f/KSM9/CXdghTPeBackJR+uwgc8ErXsXoyJ2hrbO3jqMr4Gi39/6Y6uzzd4K6otur5sWMAvouojDEG3PrV/dopKmNtvwAD/xb2obQpDkETt/qnPyzHMgXOwjzdOwGZRPVSvKAuZW1bqXPslSKJVoroNTz3RifKaZci+mixHFvqmMzgPz2CtDmZAkEQ74Luo1zZhLuw8D6JWddQZT8KmyubKiYwGmwFBEuVcJZLeFXj4/n2VhmFYwtd5TExMlPwAcZJT+hXmDOUAAAAASUVORK5CYII=>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACkAAAAYCAYAAABnRtT+AAAC00lEQVR4Xu2WS8hNURTH/0J5hoiESCKPkmSEkijyDANFBgyY+PJ+ZXBLJgqFIimPiRQDSSRhRshjwIAUEiMToVD4/629791n3fOde9zrK4PvX7/uuXvvs8/aa6+19gY69f+qH+nuGxuoKxlAuviOMupDZpEVZDxsMqk3GR6eUy0gB/H3Rsq4rWRzeC6lCeQu+UwukC3kHLlBJpJrZE51tGkKuU0Gu/ay0sL0jeW+w0sD95LvZBfpme3GTPKJvEPWkxp3maxM2prRJJhzRviOKBl4nPxA+6vpQa4G9Bw1jzwlQ5K2ZtSNnCcV117VBvKL7ENxXJwle5L/GnuaHEnaWtEq8gCWSBmNIe/JSxS4OugUsvE4iDwjy5I2L825MPxKA8l8MrQ6oqbJ5A2Z5jsqMC+W8YYvMVPJ2/DrJS9vgoXRGvKcHIAlyA6YU8ZVR5tkuIzMLFpl5g75ifqMLaNF5AMZ7TuoubBEjOGjUFHSqZwprr+gfnHRnp1pY7T8I+zlVJpcW6MxKf2TMTJS7+dtXRsZG55VW2+Si7AEmU6WoFZ7o6KR+9PGaGTehzSxPHEPVpYUEo/JumRMkZGp5Gl5POOhHEUjFftVxcAv+lB8MW9byxq5mHwjM3yHU+52a0uPwWJS9S5PCgOFg6+Pkj6qyqCsTKV515ITsA8rKV+TYaFfJ9Nh0iv8j1LpUQnSKZfRSPKCPEH9maxMPgnb6kycBGkBMtInnQr7K1iRVwY/gnlIBmsB22E10Uvf13t+vj9SgN8nX8kZ2AQy6iEsAbaR2XFworg96127FqdsvkKuk40wDylxLpHdYYyXdkbhFz1eJ61wFFkKu/noMpE3kVcFtaxNpczVtsYQ8f/zVIEdjX6ulqUTS6GiC0IrUrm7BbvIdIh0HzyE4nO/kVbDEq3M7jUlTXwU7d+gGkm7oXuqkrhDpeKvW5SvpY3UF5akvrJ06p/qN9cMe0Mb8U8FAAAAAElFTkSuQmCC>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADMAAAAYCAYAAABXysXfAAABtUlEQVR4Xu2WPyhFURzHf8KA8i+RMjAqZcBAFjKQpCwGko1BkcX6ymQhMbEweYPMsloMZmZlkyyWp/z5ft85l3OPl3fuvZwM51Ofuv1+5/G+957feVckEAhkYQQ+wHftOawx+vXwwujTM1hnrPEBv9MibLTq36iAB/AFFuBgvF1kBp5KPOhf0wxn4RF8gnew3VxQiiZ4DFdF3fl9UQFN1uGcVXNlDU7YRQcYZhr2w7w4humFO6IW3sJ72GX0q+ChXpeGDThlFxPCm+0Uhnd8SV/nRD2dlc+uSIuoP8YnmAavYbZhn77uEbU/r2CDrg3DPX2dBm9honnh3SfcUifwDY7rGp9a2nkh3sJE82IOPEMwDEPx9HKdl2rYJuofmm7ChRL1VlhZ/GR5nMKY8xLB7cVtxu02Ku7zMiTqiLe9FvVbZdd3YSc/6EDZMHwanIUBuwHmRR0EN3DL6iXFyzaz58WE24XHNANlmRfiJQy3EF9Nau2GJgcfYbdVT8pvheHvX4fdGIPP8vWuVYCTsRUKHtN8V3OZl59IG4YHxKXEv+urqFDLxjqvpA3zL+Epl3WrBgKBQJwPBYtfHOP2KLoAAAAASUVORK5CYII=>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABIAAAAYCAYAAAD3Va0xAAABGUlEQVR4Xu2TsWoCQRCGR7BRQRECYmkpCBZiIdgkWATsbJPeRhDyBNfaaKGVVlY2Yi15Ap/AvICdiI1NLEz+310ve3vkvKu9Dz5YdpbZmbk9kZioPMM9/NGuYcqIZ+GnEacrmDHOuCTgFJ7hN2x4w1c6cCneS3zk4Rz2Rd04EZXc5AO+WXs+qnAEi/AL7mDJiCfhTJ8LhDd19doRVVXPjYo8iaqYlQcyhDW9rsAj3MCc3mvCsV7/y20+vJWwjQW8wFe9x2pDz8ccLhMwERPyK0Wezw22xNbY4ouEmA+rYO91OwDeRQ19CwdWzIc9H5OCqKfAZHfnw7L53NN2QOPAAyxb+y4teJK/f4e/RdtzQsGnwH8vcD4xD8kvcTMzNIxbkGYAAAAASUVORK5CYII=>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABYAAAAYCAYAAAD+vg1LAAABb0lEQVR4Xu2UTSsGURSAj1CEsEIWoqyJFSHKxkZio/gBWFr42tsqlBIl7GRjQf6DH6BIIVmwUzY+4jnuve/cmYZ5551Yeepp3nvOnZnTec8dkX/+miH8yNNZe08qdvEVeyLxIuzAGxyN5BKpxTO8xsZwKsc2DkaDSbThEx5iiY3ptR1L7XrV7kvFhJgeznsxrVyrrLBrbUNNkM4PfcAbjmADNuEmLvub0uL6+453eIsPdp26pz6d+Czh/lbhAbbYdbGNpcL115/ROlzHMrsex5kgnYzO6I7Ez6+jEvexOZr4CdffKzFVxjEppnotohpXcAn7xbxwWoKRzBHXX4eO2QI+YpeN6cj14iWOiXnZHg7b/Ne/fS/BN8CfCPXFy51iublNWnEAT8S0SIvRovz5Lxh9yJr9rYfoQjKOpeIqdB8jvR6LqT4TWuE5HuEUbmF9aEeB6Ehqf3U6Uh+Y79AJWMSNaCIr3WJO4Bz2RXK/yyfm+Eq5ndJxZQAAAABJRU5ErkJggg==>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABYAAAAYCAYAAAD+vg1LAAABXUlEQVR4Xu2UzytFQRSAj/yIUKwkixc7RcSKLG3sXmwU9pKVDdnb2CFFKUk2kqUs3h8g/gaFZMFO2fgR3zEz786bLvfNU1bvq6/bnDNz77nnzlyRKv/NBH6W6ZJdE8UBvuFYEK/BIbzFySCXSTte4Q12laaK7OF4GMxiAJ/xBOtsTK+DWG/HG3ZeFDNierjsxbRyrbLZjrUNbUm6PPQG75jHTszhLq75k2Jx/f3Ae7zDRzuO7qnPML5IaX9b8Rh77LjWxqJw/fX3aAduYaMdT+NCks5G9+i+pO9fRwseYneY+A3X32sxVaYxK6Z6LULRtkzhKc7hqCQtLJLWX4dusxV8whEb0znbuCjmQasSrNWv/SDJP8DfEeqrlzvHJrPs+wGXkrzdplT4/wjRA3Qkplrt/Zn8/G2i0Bu709mHF2LeQnfNn+gX09N53MECrmOvP6lS/MOiP6gGL1fF8AUl0ka5LZ9LRwAAAABJRU5ErkJggg==>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAG8AAAAXCAYAAAAFtBHMAAAFK0lEQVR4Xu2Ya6ilUxzGH6EIicEgl5OUkFtukXTGLTIkjERJUSQUU0Tk5JIIuUUuNfkwJZQPrsWHg5IQX8wolyZyCaGEXHJ5fv7vf/baa79779lnh5j3qaeZvdY6a63nf1trvVKHDh06dOiwfmNLc4W5T91h7GHeaj5onmVu2t/9F2ijjzE3mTv1d6/FBuah5t3mfeaRTdu/gQ3NM8yZqr0Nu5qPmttV7cxxokILmo4zN+4bEdjBvFZhn+Xmzv3dazGxfei8yvzBPLDqO81cbe5vbm7eYL6gcHZie/Ml806F05aaa8wjijFgM0WAPGnuae5tvmkeWw76m0GQnWTeZX6qds01cAb7/kjhhAT2eNy8WOHUg8wPzefUbx/We9bcz9xWYcOfzFOLMWBB9sHT32hQCNHxvnl20baV+YZiw2Aj82HzXXNxDlJk32vqiSBAcO7LRdvl5h/mlc3vfwI47wRz1rxOg5rbsMz8VYPOO8b83bxfYQeAbjRd0PzexHzG/NY8oGnbzfzcXGVu07QtyD4MfEiRzrUQnFa3schKc14ReUTI18XvBNGGMAQC5mCuFAXIUsQiZhSIyDIwarCnHRUlbBJglFpfjRmFbZ7WoPNmzd8UmYKTwNUKg2N4QDv9jJtt2tD9iSJLU9fE9kE0i1Aa24RQd+s28Igicpg0F53XoPPKqJlTRC+llHEYIQWPA6XmMUVZqoGGc8171H7WjEKb5hLMd7t5iEJz7TzWXqSeDrL6eQ3OST/jGA+OVwQ2FSszdk4T2odNsTk22SaEDddtdfs457HBLB2Mu9l8QBE0lOSLNOZAbkAUYhj2nJjGcaBNcwmCmn2yTpvzSpD15ynm40gZpgkdlEaOnrzUTWwfyiXlYKb5XQvBEfNVW6J0HvNwtr1iblGMydrP2JwrnZnRtsT8zjy5+T0OpQOndRyoNZfgdslxkufPKOddaH5sfqG4UVLma3ChwUaUy7cUl5d0ykT24Y8uUxzEiVoIG3ixakuUzgNE6FeKiw+YUVxgaueVZyDAEBiEqBtZIgqkA+/VdI4DteYEc96mnh4wynkJyuYTirOMu8AwHGX+qCiVrDWRfbj1ZLlMtAmpnTSsnWBYan6gWIwDmoM3z7xhgZCbG2eUEqxFSSFYDq/6JkWbZkCkZ7lMrIvzANd/dD+l9vcwoEKRhVxieAZMZJ9M85LfKxb9UjExaU7pqycECCH9s2a3AeeVkUQ5qOdq3dwIYEzOEzJuF0WQlGfgpBjmvDs0aJ9fFPb5TPFYx+BHN2NLOzAXc6Ymyu6c4uZeB0N5oZvKPm1CiECio0zlPFzLVL5E4XDOCcAmV6j/nXem+bP6H+6tZWEISsdlxeDjwDQObNM8DHXmledUOgDkRW2V4g2Xv4f9bT4NprIP7xPq8MFF2yLzdUXkJHZXZB2LJRDGwoc1vzEmT4nyoCWT31G/0CVqOZBbgOO4dfGIrc+4aRzIXmrNbWD9leqvNlwqyJb3zH2LcXw9wSnLmzYCgw8gN6q3d2xIBnMvyPkWZB+yilLJgpBMe1W9b3i8rdaYV5inK664t6jfiEy+2jzfvEYh8hwNXnE5qOnjOnxp839E1uNq7GVer0HHJbZWfHvl33Gg3FH2MGhqznJICazBEVOOpXxm2cTwXPv5REZZRBf9pX0y8MhEEoRxbyvuB3xyLLFQ+4wEG+WD6yka/kF1saJEMK7tqpxYl7n+S+B9x82SwEY/1aoNHB/oZhzZOOxr0P/NPh06dOjQoUOH9Q1/AuSNZs9o3/l/AAAAAElFTkSuQmCC>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEgAAAAXCAYAAACoNQllAAADRUlEQVR4Xu2XS+hNQRzHv/KOKEReuYSSRAmRhCxYkESUUCSy8CqJoosshI2Fd2QhkdggJFGSJK+yIYVIKRthQR7f7/3NuOfMOec+0/0vzrc+3TNz5pw7853f/GYOkCtXrlxtSwWyJKxsgdqRjWRteIPqQTaQ42QH6R+/XZKen0QOkcNkmqtLU/R92105plFkHblNfpEz8dstkQb3jWwN6oeQZ2Q16ULmkJdkYqRNR1IkD8kY2Pjukl1ImjSTvCLLYUZvIZdI12gjvWA+mULeo/UG9YRN1h/EDepATpKL7tprL7mO8qAWkB9k9r8WwFTymUyO1I0mH8hiV+5HXpO3SI/KUqVu1mNQL0eW2pMBSM5cltRuM9mPZAQNIx+DOkmGqO14WFRdjZS9/Ni05CRv9gvSx9Xpv5eSle5+Qo0YNIJchoV+KIX6TljY1mqQltYB2EyHBs0iv4M6aS4s2jS47uQOsg3SPbUZClstisbOpC/pjSr9bMQgaRy5gbhJjZijpaVEWYANLjTIG5FlkOqrGaQlpKXkzb5CjsGi9hy5Rwa5ZxJq1CApalIj5qjdJrLIldMM0nU1g6Qi+U4m+AawHPQT5fzin4kaqX5rNdyEGZ1QMwZJMknJ9QTqM0fSLnQQ1kkpzSDNci0GaZK0s22D9UHvVL5Rm9Ag5SvlLS+9Q5GlCEuoWYPUESVBhbESaq3SbB2FLS2vNINCIyrVF8gt2C71hCxDPAfpeKBnwrFmRWlJzRgkc7TzKHK0fWpmojmpknRWeU7eRfgE6+hXV9aM+mUSdt4bpN0sSzrKaJv3u9hY8gXJsf4Xg6Lm+GWlDtVjUqi0CBpI3qA8SK81sMHrPyUN/insbOc1D5aXtDtK2hAeIHmmqmmJnUXt+UPm7IMl2PCZZkxSgtWAdPT30vv3wAamAUr6/wuwHcgP1EdU0ZXVVsvtSKSNpM8Y5arBrpyZpOWWzgT6zNCLfWgr7BX+lTSDrEfSHK/hZDfpFN7IkM4j9xHvi5abn1EN9ho5D8sjp5DcmnX9CGbmKnd9mnSLtJFUVv1jsgJmsj5jGpnQNiWdzrUEF7pflUNpZ5oOW2Y+QtKkiR0Je5c+aP0umitXrly52rr+AgW3y7YWoIyYAAAAAElFTkSuQmCC>

[image12]: <data:image/png;base64,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>

[image13]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACYAAAAYCAYAAACWTY9zAAACJElEQVR4Xu2Wz0tUURzFT5hkmJgklNhGISUIokSQKChIcKNBbopoJeQuKMnox0IUQVwU6KKg/oIIQRQEN2pCaPsoghYtUhDSlS6UyHM48+R53zihMz43HvjAvHvfdb6ee+73DnCoQ6WrG2SR/IuxRNbJXzJP2klRtCBtvScb5GpsTMU8gAvsJkdic6mojMySn+R0MFdFfu0wt+86T/6Qj+RoMNdI1shXUhnM7bva4Gx1hhNUDzzXFYynoiEk81VMOmAnn2SeU9UJMg2fws+Zz99gl96QU9GLaStbvnT6nsGnsTkzFqmafCILpDaYu0h+wAdJByovRfl6HIw3kFW4jYS6SSZht0Mpp4pG3sqWL+keXHB/MC49R/ZxOS7ntTYv5epfKliFPQ3GS8gY7PQZMgAXqqLUTqbgeOykejJIxpGMyZYukBUk+5c+f8D2wl7AW3iWzJHr5C58YrWtpbDrM6TCSxK6TUZJOWxKH7xuS9fgbh7ej3Ihku5HhV8F3ifvyHG4uGXyEnZPRFnLlS+5qSb9Ct7qt6Rl2xu7kLa3FbZcRUly8DVchJzSfy/9L186TN/JuXCiEIrnS3eo2oa+SEUqb8rXFbgxS9qmR3BbUe6+wOsktaTLSF6Be5Ic1JfXwI13As6JoiHnRkgvacq8X0d+wy6rkIfwYZGrw7CLBZH+eDysuqZyPUs6LHdiz8rjydjzgekWuRQOHrSOwdua+g/MgmoTjLlnVrcMIRYAAAAASUVORK5CYII=>

[image14]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAkAAAAYCAYAAAAoG9cuAAAAoklEQVR4XmNgGAXkAnEg9gViOyBmRZNjUAPiE0C8HIgjgLgaiFcDMSdMgT4QvwbiSiBmhIqJAvFWBojJYJWbgfgJECtCFYAkpgJxKQNUE8iUT0D8C4gfMUCs7AJiY5gCEHAB4n9AXA4TwAZMgfgbA8RH6IALiJlBDH4GiBWtKNIQzSCfCsMEDID4AgPEy7MYIJr6gZgbpgAGQMaKQTHYipEBAERmFwESThcqAAAAAElFTkSuQmCC>

[image15]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAYAAAAZCAYAAAASTF8GAAAAd0lEQVR4XmNgoBngAWIxIGaGCWgB8R0g/g/EV4FYBCYBAoJAfBqIlwIxI7KEJhC/BeJ0ZEEQiAbi30Bsgy4xiQGP+WuAmAVZgnTzcUqQb7E6EFfDJDyB+CcDxPwsII6ASYBC8xwQ7wfixUDMD5MAAVaoAhA9iAEADUgZKJL1EpIAAAAASUVORK5CYII=>

[image16]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAXCAYAAADduLXGAAAA4UlEQVR4XuXSoYpCQRTG8SMquKigGA0iirDNBzBqsrnR4AtYtBjFKJpsxu2iGOyC0bppk0Fs+wIK6v/cOwMy94p1wQ9+cOfMwJk5XJF/mQhyyLgbbsY444a+sxeaFi6ouRthmeGAvFMPJI0dNkg4e4F84g8Ds9bHVtDAhz1k08YVdcQxwgRrCXmwvW8JQ1TFPxSYThZ7/GAu/pU0eo0ekmbt5XFkZfxiJU8e6o7sW/xO2rGJjqlLClssEDM1PaxrncIURVOXAk7o2gL5whFLU9cxetEPbRe1BRPt+PKHet/cAcfeIy832IBiAAAAAElFTkSuQmCC>