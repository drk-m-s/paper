# Background
I am trying to develop an MoE offloading engine based on llama.cpp for an edge NPU. The NPU has 10GB of high-bandwidth DRAM built-in. Without an offloading engine, it can only run small models. The goal is to enable the NPU to run qwen3.6-35B-A3B at a certain performance.

# Chip spec
The NPU has 56 TOPS INT8 AI compute, 10GB of high-bandwidth 3D-stacked DRAM built-in, with 1200GB/s memory bandwidth. The chip package is 14mmx14mmx1mm, with a typical power of 15W.

# Hardware solution spec
The final product will be a m.2 2280 card containing a single NPU chip. The m.2 card connects to the CPU via 4 lanes of PCIe 4.0 with 8GB/s of bandwidth. The SSD to CPU data bandwidth and DRAM to CPU data bandwidth is not controllable by us, it is determined by the SSD and DRAM used. But as a general reference, consider SSD to CPU having 8GB/s bandwidth and DRAM to CPU with 68GB/s. Note that the SSD and the NPU does not share the same PCIe port, they are seperated connected to the CPU.

# Use case scenario 
The m.2 card will be used in Mini-PCs, Mini-Stations, AI NAS products. It will provide offline inference of qwen3.6-35B-A3B or similar sized MoE models. Values to customers include saving API token fees, privacy protection and faster response. Batch size=1.

# Performance requirements
The MoE offloading engine combined with the NPU chip solution should deliver the following performance with 4-bit quantized 35B model:
1. TTFT less than 2 seconds for 1k input.
2. Decoding speed greater than 25 tokens/second. The higher the better.
3. Support a max context length of 256K.

# Other Requirements
The entire hardware + software solution should cover different market needs:
- Low-price/Cost-effectiveness oriented: NPU M.2 card + SSD(Offload MoE experts to SSD) + Less DRAM(use less system DRAM as compared with offloading to DRAM or no offloading). This means we should adopt a NPU 3D DRAM + SSD offloading scheme.
- Performance oriented(not price sensitive):  NPU M.2 card + Very strong CPU + More DRAM (offload experts to DRAM and also use CPU for prefill so model weights have to reside in CPU DRAM). This means we could adopt a NPU 3D DRAM + CPU DRAM offloading scheme.

# MoE offload engine development considerations
We will need to design our own architecture based on other research papers, but here are some requirements:
1. It should be implementatable in llama.cpp. Although I think all of the papers can be implemented in llama.cpp.
2. It shouldn't make modifications to the model. End users should be able to pull models from huggingface and use it directly.
3. It can contain an offline profiling phase to identify "hot" vs "cold" experts. But only if this yields significant gains in performance.
4. It should have an option to only use NPU for compute or use CPU + NPU for hybrid compute. Meaning when paired with a weak CPU in low-price oriented markets, we use NPU for compute, when paried with a strong CPU in performance oriented markets, we leverage CPU + NPU compute.
5. The final architecture should be easy to understand, easy to implement, clean and lean, with good performance gains.
6. The engine should not use more than 5GB-8GB of 3D-stacked high-bandwidth DRAM inside the NPU.