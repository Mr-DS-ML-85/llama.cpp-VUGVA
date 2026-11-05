# Configuration reference

Everything VUGVA exposes at runtime is an environment variable; nothing needs a
rebuild except the DRAM pool default and the CUDA backend selection.

## Environment variables

| Variable | Values | Default | Effect |
|---|---|---|---|
| `VUGVA_MODE` | `auto`, `hybrid`, `zerocopy` | `auto` | how warm-tier layers are executed |
| `VUGVA_VRAM_MB` | integer MiB | computed | caps the hot-tier budget. `0`/invalid = auto. Useful to force the DRAM tier on small models |
| `VUGVA_RESERVE_MB` | integer MiB | `max(VRAM/8, 1024)` | VRAM kept free for KV cache, activations, CUDA context |
| `VUGVA_PIN_MB` | integer MiB | 50 % of RAM | maximum pinned warm-tier memory. Set higher only on a dedicated inference box |
| `VUGVA_DRAM_GB` | integer GB | 32 (CMake) | DRAM pool size at runtime |
| `VUGVA_AUTO_MOE` | `0`, `1` | `1` | auto-route sparse experts to CPU/mmap when the model exceeds VRAM |
| `VUGVA_HOT_CACHE_MB` | integer MiB | `0` | VRAM cache for warm-tier weights; promote on use, LRU evict. Requires `zerocopy`; auto-disabled when static placement already filled VRAM |
| `VUGVA_PRELOAD` | `0`, `1` | `1` | background readahead of the GGUF into the page cache (cold tier) |
| `VUGVA_STATS` | `0`, `1` | `0` | print tier-cache counters (hits/misses/promotions/evictions) every 16 decode steps |
| `VUGVA_ZEROCOPY` | `0`, `1` | set by `VUGVA_MODE` | internal: makes `ggml-cuda` accept `VUGVA_DRAM` buffers |
| `GGML_OP_OFFLOAD_MIN_BATCH` | integer | 32 | llama.cpp: minimum batch size to offload an op |
| `GGML_SCHED_DEBUG` | `0`/`1` | `0` | verbose scheduler/graph output |

### Mode selection details

`auto` computes the warm-tier estimate as

```
hot_est  = free_VRAM − max(VRAM/8, 1 GiB)
warm_est = (model_bytes − hot_est) * 1.2 + 64 MiB
fits     = warm_est + max(4 GiB, warm_est/4) <= MemAvailable
           && warm_est <= min(VUGVA_PIN_MB, RAM/2)
```

and selects `zerocopy` iff `fits`, otherwise `hybrid`. With
`VUGVA_MODE=zerocopy` forced, the same numbers are printed as a warning but the
run still tries; the allocator refuses unsafe pins and degrades automatically.

## CMake options

| Option | Default | Effect |
|---|---|---|
| `VUGVA_ENABLE` | `ON` | build the VUGVA backend |
| `VUGVA_DRAM_GB` | `32` | compile-time DRAM pool default |
| `VUGVA_DIR` | `/home/irfan/Desktop/VUGVA/vugva` | path to the VUGVA Rust crate |
| `GGML_CUDA` | `ON` | CUDA compute backend |
| `GGML_CUDA_FA_QUANTS` | `f16-f16` (+ cache) | FlashAttention KV type combinations, e.g. `q4_0-q4_0;q8_0-q8_0;f16-f16` |

`GGML_CUDA_FA_QUANTS` must include the KV types you intend to use. If
`-ctk q4_0 -ctv q4_0` is requested but `q4_0-q4_0` was not compiled, CUDA
FlashAttention has no kernel for that combination.

## Kernel promotion (`VUGVA_HOT_CACHE_MB`)

Static placement fills VRAM at load time. If VRAM is left over afterwards
(conservative reserve, small `VUGVA_VRAM_MB`, room after the KV cache), that
memory is dead weight. The tier cache uses it:

```bash
VUGVA_HOT_CACHE_MB=4096 VUGVA_STATS=1 ./build/llama-cli -m model.gguf -ngl 99 ...
```

* warm-tier weight tensors are promoted to VRAM on first use and stay there,
* eviction is LRU under the byte budget,
* the cache is skipped when fewer than `budget + 512 MiB` of VRAM is free,
  because promoting into a full card thrashes (measured 1.20 → 0.60 t/s on a
  27B) — the counter line below tells you which happened:

```text
[VUGVA] hot cache: 2048 MiB VRAM for warm-tier weights, promote-on-use + LRU eviction
[VUGVA] tier stats: step 16 cache 394/2048 MiB | hit 3210 miss 214 promote 214 evict 0 promoted 394 MiB
[VUGVA] hot cache ignored: only 961 MiB VRAM free, need 1024 MiB (static placement already uses the rest)
```

A cache smaller than the per-token working set shows up as a large `miss`/`evict`
count and slower decode — that is the honest failure mode, not a bug.

## CLI flags (`build/llama-cli`)

```
-m, --model FILE
-n, --n-predict N
-ngl N
-c N
-b N
-t N
-fa on|off|auto
-ctk TYPE          f16 bf16 q8_0 q5_1 q5_0 q4_1 q4_0 iq4_nl
-ctv TYPE
-cmoe, --cpu-moe
-ncmoe N, --n-cpu-moe N
--temp T
--top-k N
--top-p P
--min-p P
-rp, --repeat-penalty P
--seed N
-p, --prompt TEXT
-h, --help
```

A quantized KV cache automatically enables FlashAttention.

## Recommended presets

### Small model that fits in VRAM

```bash
./build/llama-cli -m model-0.8b-Q4_K_M.gguf -ngl 99 "prompt" -n 64
```

### Dense model larger than VRAM

```bash
./build/llama-cli -m model-27b-Q4_K_M.gguf -ngl 99 "prompt" -n 32
# auto selects zerocopy; force with VUGVA_MODE=zerocopy
```

### MoE model larger than VRAM

```bash
./build/llama-cli -m model-35b-moe-Q4_K_M.gguf -ngl 99 \
    -ctk q4_0 -ctv q4_0 --temp 0.7 --repeat-penalty 1.1 "prompt" -n 64
# experts go to CPU/mmap automatically (VUGVA_AUTO_MOE=1)
```

### Force the DRAM tier for testing

```bash
VUGVA_VRAM_MB=100 VUGVA_MODE=zerocopy \
  ./build/llama-cli -m model-0.8b-Q4_K_M.gguf -ngl 99 "prompt" -n 32
```

### Machine dedicated to inference (allow more pinning)

```bash
VUGVA_PIN_MB=24000 ./build/llama-cli -m model-27b-Q4_K_M.gguf -ngl 99 "prompt"
```

## Inspecting placement

Look for these lines in stderr:

```text
[VUGVA] DRAM pool: 32 GB, NUMA=yes (cudaMallocHost=always, DMA-capable)
[VUGVA] auto mode: model ... MiB, warm-tier estimate ... MiB, RAM available ... MiB (cap ...)
[VUGVA] mode=zerocopy VRAM budget ... MiB (...) -> N/M layers + output on GPU
load_tensors:        CUDA0 model buffer size =   ... MiB
load_tensors:   VUGVA_DRAM model buffer size =   ... MiB
load_tensors:   CPU_Mapped model buffer size =   ... MiB   # --cpu-moe / auto-MoE
sched_reserve: graph splits = ...
```
