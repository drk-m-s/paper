
First implementation - MVP verion:
Use "MoE Infinity" as a base for implementation. Modify llama.cpp to achieve MoE Infinity functionalities. Also consider FlashMoE for reference.
- Seperate the model into non-expert weights and expert weight into different file. Consider more fine-grained seperation like in FlashMoE if it has significant benefitss.
- Maintain hot expert cache in GPU VRAM. Cold experts are offloaded to SSD. Include a new parameter to control how much experts is offloaded to SSD, i.e. how much VRAM is used.
- Use GPU for all calculation. Use GPU VRAM for KV-cache and attention weights. CPU DRAM is not uesd to store model weights.
- Identify which parts of llama.cpp needs to be modified. Make modifications similar to class extensions, so that future llama.cpp community version updates can be merged into the local modified version without conflicts.
- Write tests to make sure the modified code is correct and qwen3.5-35B-A3B model can be successfully run.
- Profile the running performance of the modified llama.cpp on the target 35B model, write code to mimic llama-bench functionalities.

The MVP version's target hardware is AMD HX 470 CPU + Nvidia 5070ti GPU. The GPU has 16GB of VRAM. The CPU has 32GB of DRAM. The GPU is connected to the host via Oculink with 4x PCIe gen4 8GB/s bandwidth. 

Second implementation area to explore:
Main "MoE Infinity" first implementation architecture but improve the following areas:
- Use GPU VRAM + CPU DRAM + SSD together. Add a new expert cache pool in CPU DRAM to speed up inference. Make the CPU DRAM option a new parameter as well to specify how much CPU DRAM should be used.
- Add an offline profiling phase such as in "Fiddler" to identify hot experts to be loaded at initiation. 


Third implementation areas to explore:
Consider using model's internal represenation as prefetch prediction mechanism:
- "Fate" paper, using the previous layer's gate input (Gate_in_i) to predict which experts will be activated at the next layer


Fourth implementation areas to explore:
Train a lightweight prediction model to predict expert activation
- FlashMoE