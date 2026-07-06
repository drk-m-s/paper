## From previous implementation progress
The entire MVP perf plan is closed. Remaining work is post-closeout:
router-aware internal prefill splitting, default promotion of guarded fast paths,
normal top-k fusion promotion, prefill graph/fusion work, and broader
interactive-chat prompt coverage.


## Human Run MoE-Bench Results for LRU and EAMC 
#### LRU results
model: qwen35moe 35B.A3B Q4_K - Medium
predictor: lru       cache: 8000 MB   ssd: C:
n_prompt: 256  n_gen: 256  repeats: 3
ubatch: requested=512 effective=8  slots=96/256  mode=streaming
calibration: cache_reset_between_repeats=false  warm_cache=true  hot_start=true
phase     tokens   total_ms   per_token_ms   tok/s
prefill      256     2718.2         10.62       94
decode       256     7454.4         29.12       34

cache hit rate (prefill): 86.2%
cache hit rate (decode): 91.0%
SSD bytes read (decode): 39.32 GB  (avg 52.43 MB/token)
TTFT: 2718.2 ms
TTFT cold: 0.0 ms  (n=0)
TTFT warm: 2718.2 ms  (n=3)
TPOT: 29.12 ms
total: 10172.6 ms

I/O breakdown (prefill, mean per token):
  ssd_read           5.20 ms
  h2d                3.84 ms
  gpu_compute        3.71 ms
  stall (overlap loss)     0.06 ms
  predictor          0.02 ms

I/O breakdown (decode, mean per token):
  ssd_read           9.86 ms
  h2d                7.83 ms
  gpu_compute       12.43 ms
  stall (overlap loss)     0.18 ms
  predictor          0.12 ms

Wall/profile reconciliation (decode):
  wall_decode_us        22363243
  profiled_decode_us    23488786
  unattributed_decode_us -1125543

VRAM peak: 10.15 GB / 15.92 GB  (experts budget: 7.81 GB, other model/kv/compute: 2.34 GB)
DRAM peak (process): 1.31 GB
SSD reads: 66642 (avg 0.60 MB each, avg latency 0.11 ms)
profile rows: prefill=3840 decode=30720

#### EAMC results  
model: qwen35moe 35B.A3B Q4_K - Medium  
predictor: eamc      cache: 8000 MB   ssd: C:  
n_prompt: 256  n_gen: 256  repeats: 3  
ubatch: requested=512 effective=8  slots=96/256  mode=streaming  
calibration: cache_reset_between_repeats=false  warm_cache=true  hot_start=true  

phase     tokens   total_ms   per_token_ms   tok/s
prefill      256     4020.1         15.70       64
decode       256     8458.0         33.04       30

cache hit rate (prefill): 88.3%
cache hit rate (decode): 88.6%
SSD bytes read (decode): 49.70 GB  (avg 66.26 MB/token)
TTFT: 4020.1 ms
TTFT cold: 0.0 ms  (n=0)
TTFT warm: 4020.1 ms  (n=3)
TPOT: 33.04 ms
total: 12478.1 ms

I/O breakdown (prefill, mean per token):
  ssd_read           4.76 ms
  h2d                3.28 ms
  gpu_compute        3.96 ms
  stall (overlap loss)     0.05 ms
  predictor          5.23 ms

I/O breakdown (decode, mean per token):
  ssd_read          12.23 ms
  h2d                9.89 ms
  gpu_compute       11.70 ms
  stall (overlap loss)     0.13 ms
  predictor          1.48 ms

VRAM peak: 10.15 GB / 15.92 GB  (experts budget: 7.81 GB, other model/kv/compute: 2.34 GB)
DRAM peak (process): 1.38 GB
SSD reads: 84231 (avg 0.60 MB each, avg latency 0.11 ms)
profile rows: prefill=3840 decode=30720

## Overall Human Verdict
#### Status
The Close-Out MVP achieved the desired results: llama-cli, and llama-moe-bench can now RUN with the moe offloading functions.  
#### Goals
Looking at the performance stats, I think the next step need to look into cache management strategies including expert prefetch mechanism, cache eviction polices, and etc.   
The overall goal is to increase prefill and decode speed by increasing cache hit rates. Having high cache hit rates will decrease ssd_read and h2d time.  

#### Methods
The first way to increase prefill and decode speed is to add expert prefetch, so as to overlap the ssd_read and h2d time with gpu_compute.  

#### Reference papaers  
/docs/papers/5-star/MoE Infinity.pdf  
/docs/papers/5-star/FineMoE.pdf  

#### Questions  
Why are the EAMC results worse than LRU results from the MVP? The MVP was based on the MoE-Infinity paper, is the EAMC implementation correct?  
What further modifications for cache management can be implemented to reach the goals? I think you need to carefully read those papers and form a action plan, ideally we validate one paper's approach first and then focus on others in later stages. You should limit the ideas to cache management related areas, including but limited to expert prefetch, expert cache eviction policy etc.  