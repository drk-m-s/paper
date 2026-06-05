## Target Hardware:
The current target hardware is AMD HX 470 CPU + Nvidia 5070ti GPU. The GPU has 16GB of VRAM. The CPU has 32GB of DRAM. The GPU is connected to the host via Oculink with 4x PCIe gen4 with total 8GB/s bandwidth between GPU and CPU. 


## First implementation - MVP verion:
Use "MoE Infinity" as a base for implementation. Also consider FlashMoE for reference in model file cutting. Modify llama.cpp to achieve MoE Infinity similar functionalities. 
- Seperate the model into non-expert weights and expert weights into two different files to prepare for load-on-demand prefetch. Non-expert weights are loaded into VRAM upon launch. Expert weights are randomly loaded into VRAM at start,  given a parameter which controls the size of the expert cache. Then expert weights are pulled on-demand from cold storage (SSD) into VRAM based prefetch algorithm. Consider more fine-grained seperation like in FlashMoE if it has significant benefitss.
- Maintain hot expert cache in GPU VRAM. Cold experts are offloaded to SSD. Include a new parameter to control how large the expert cache in VRAM is, i.e. how much VRAM is used for hot experts storage.
- Use GPU for all calculation. Use GPU VRAM for KV-cache and attention weights. CPU DRAM is not used to store model weights. Offload and store KV-cache to SSD if necessary. 
- Identify which parts of llama.cpp needs to be modified. Strive to make modifications in the form of extensions to llama.cpp or patches to llama.cpp, so that future llama.cpp community version updates can be merged into the local modified version without conflicts.
- The implementation should be compatible to native llama.cpp supported functionalities including but not limited to flash-attention etc.
- Write tests to make sure the modified code is correct and qwen3.5-35B-A3B model can be successfully run.
- After testing is successfully done, deliver to me a readme on how to start model inference using the modified llama.cpp framework.


## Profiling infrastructure (as part of Fisrt implementation)
A profiling code needs to be established alongside the implementation, it needs to profile very detailed stats during inference:
- Prefetch accuracy (expert cache hit rate) stats, by token by layer, and then aggregated results.
- I/O latency including timing break down of SSD read, PCIe transfer(data from CPU to GPU via PCIe), GPU calculatation time.
- GPU VRAM, CPU DRAM, CPU SSD usage.
- Inference performance including prefill speed, decode speed, TTFT, TPOT, total execution time.
- Mimic llama-bench functionalities. Users can specify input tokens and output token then a profiling result is delivered.


## Second implementation area to explore:
Main "MoE Infinity" first implementation architecture but improve the following areas:
- Use GPU VRAM + CPU DRAM + SSD together. Add a new expert cache pool in CPU DRAM to speed up inference, because reading from SSD is much slower than reading from DRAM. Make the CPU DRAM option a new parameter as well to specify how much CPU DRAM should be used.
- Add an offline profiling phase such as in "Fiddler" or "DALI" or "PowerInfer" to identify hot experts to be loaded at initiation. 

Additionally:
- Thoroughly understand the paper "FineMoE". Incorporate granularity, prefetch policy, eviction policy, etc, which has stronger results compared with MoE-infinity.
- Thoroughly understand the paper "ADEPT" and Incoporate improvements made in "ADEPT".


## Third implementation areas to explore:
Consider using model's internal represenation (or hidden states) as prefetch prediction mechanism:
- "Fate" paper, using the previous layer's gate input (Gate_in_i) to predict which experts will be activated at the next layer
- "YALIS" paper, uses internal model representations — specifically "quasi-hidden states" formed from the current residual stream and precomputed per-expert default vectors — to predict which experts the next layer's router will select


## Fourth implementation area to explore:
Use speculative decoding for speed up:
- Download and run MoE-SpAc which is based on Llama.cpp at https://github.com/lshAlgorithm/MoE-SpAc


## Fifth implementation areas to explore:
Train a lightweight prediction model or a small neural network to predict expert activation
- "FlashMoE", offloads inactive expert weights to SSD (NVMe) instead of DRAM, then uses a lightweight ML-based cache replacement policy — a tiny per-layer feed-forward network (~113KB) — that learns to combine recency (LRU) and frequency (LFU) signals to approximate Belady's optimal eviction policy.
- "DALI", adds a per-layer offline residual vector to the current gate input so it better approximates the next layer’s gate input, which improves prediction of high-workload next-layer experts.
- "Pre-Attention Expert Prediction and Prefetching for Mixture-of-Experts Large Language Models" paper, predicts the experts of the current layer from the pre-attention activations of that same layer, instead of using the previous layer’s activations. But requires modifying the model.
- "ExpertFlow", the coordinated combination of a T5-style routing predictor, routing-aware token rebatching, predictive locality-aware caching, and runtime correction for mispredictions.



