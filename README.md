# llama-cpp-VUGVA

**llama.cpp with VUGVA tiered memory** — run models larger than VRAM by tiering between DRAM and GPU.

```
╔══════════════════════════════════════════════════════════════╗
║  VRAM (hot)  ←→  DRAM (warm)  ←→  SSD (cold)               ║
║  8 GB GPU       32 GB RAM         unlimited NVMe             ║
╚══════════════════════════════════════════════════════════════╝
```

VUGVA ([paper](https://doi.org/10.5281/zenodo.21549808)) virtualizes GPU memory by splitting tensors across DRAM and VRAM. Hot tensors stay on GPU; cold tensors live in RAM and are promoted via CPU-bypass DMA when accessed.

## What this does

| Feature | Description |
|---------|-------------|
| **Tiered memory** | 32 GB DRAM + 8 GB VRAM = 40 GB unified pool |
| **CPU-bypass DMA** | CPU touches only 72 bytes per promotion (paper §5.1) |
| **Hot/cold placement** | Membrane-driven tiering with decayed access rates |
| **Predictive prefetch** | Look-ahead attention tracking (paper §3.2) |
| **SSD cold tier** | Spill to NVMe for models > DRAM (optional) |

## Quick start

```bash
# 1. Build VUGVA
cd /home/irfan/Desktop/VUGVA/vugva
cargo build --release

# 2. Build llama-cpp-VUGVA
cd /path/to/llama-cpp-VUGVA
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build build -j

# 3. Run
./build/vugva_test model.gguf "Once upon a time"
```

## Running models larger than VRAM

Weight placement follows Algorithm 1 of the paper: `-ngl` requests GPU layers,
VUGVA measures free VRAM and caps the **hot tier** to a **contiguous** block
that actually fits — the remainder stays in the **DRAM tier** instead of
failing with a CUDA OOM. Models larger than the DRAM pool spill to the **cold
tier** (NVMe via page cache) automatically.

Two DRAM-tier execution modes are supported (selected automatically):

| mode | warm-tier weights | warm-tier compute | when |
|---|---|---|---|
| `zerocopy` | mapped pinned host memory (`cudaHostAllocMapped`) | **CUDA** reads over PCIe (UVA) | warm tier fits in RAM and the GPU supports UVA |
| `hybrid` | pageable/pinned DRAM | **CPU** backend | warm tier does not fit in RAM |

In `zerocopy` mode the CPU never touches data-plane traffic: the GPU kernels
dereference the warm-tier pointers directly, so the scheduler sees a single
graph split instead of one H2D copy per DRAM-resident layer.

```bash
# 0.8B — everything fits in VRAM
./build/llama-cli -m Qwen3.5-0.8B.Q4_K_M.gguf -ngl 99 "The capital of France is" -n 32

# 27B Q4_K_M (17.4 GB) on 8 GB VRAM — 24/65 layers hot, rest zero-copy DRAM
./build/llama-cli -m Qwen3.8-27B-Fable-Distill-Q4_K_M.gguf -ngl 99 "Write a story" -n 32

# 35B MoE Q4_K_M (21.7 GB) — keep sparse experts in CPU/mmap, attention hot:
#   -cmoe        experts stay in the (mmap'd) CPU buffer list, no 20 GB copy
#   -ctk q4_0 -ctv q4_0   int4 KV cache (needs flash attention, auto-enabled)
./build/llama-cli -m Ornith-1.5-35B-Q4_K_M.gguf -ngl 99 -cmoe \
    -ctk q4_0 -ctv q4_0 --temp 0.7 --repeat-penalty 1.1 "Explain why the sky is blue" -n 64

# DRAM pool size: cmake -DVUGVA_DRAM_GB=N, or override at runtime
VUGVA_DRAM_GB=16 ./build/llama-cli -m model.gguf -ngl 99 "Hello"
```

Placement and mode are visible in the load log:

```text
load_tensors: [VUGVA] auto mode: model 20697 MiB, warm-tier estimate 17296 MiB, RAM available 23853 MiB (cap 15334 MiB) -> hybrid
load_tensors: [VUGVA] mode=hybrid VRAM budget 6337 MiB (free 7361, reserve 1024) -> 41/41 layers + output on GPU, remainder -> VUGVA_DRAM tier (requested -ngl 99)
load_tensors:   CPU_Mapped model buffer size = 19486.84 MiB   # --cpu-moe experts (mmap, zero copy)
load_tensors:        CUDA0 model buffer size =  1302.90 MiB   # hot tier (attention/SSM)
```

### Tuning knobs

| variable | meaning |
|---|---|
| `VUGVA_MODE=auto\|hybrid\|zerocopy` | tier execution mode (default `auto`) |
| `VUGVA_VRAM_MB=N` | force the hot-tier (VRAM) budget; useful to exercise the DRAM tier on small models |
| `VUGVA_RESERVE_MB=N` | VRAM to keep free for KV cache/activations (default: `max(total/8, 1 GiB)`) |
| `VUGVA_PIN_MB=N` | max pinned host memory for the warm tier (default: 50% of RAM) |
| `VUGVA_DRAM_GB=N` | DRAM pool size override |

### Measured on this machine (RTX 4060 8 GB + 29 GB DDR5, NVMe)

| model | configuration | decode |
|---|---|---|
| Qwen3.5 0.8B Q4_K_M | `-ngl 99` (25/25 hot) | ~150–185 t/s |
| Qwen3 27B dense Q4_K_M | `-ngl 99`, auto → zerocopy (24/65 hot) | ~1.2 t/s |
| Ornith 35B MoE Q4_K_M | `-ngl 99` (hybrid, 12/41 hot) | ~2.1–2.3 t/s |
| Ornith 35B MoE Q4_K_M | `-ngl 99 -cmoe -ctk q4_0 -ctv q4_0`, warm page cache | **~27–36 t/s** |
| Ornith 35B MoE Q4_K_M | same, cold page cache (first run) | ~4 t/s (NVMe-bound) |

For sparse MoE models the winning configuration is attention/SSM hot in VRAM
plus sparse experts left in the mmap'd CPU buffer: only the ~8/256 active
experts are touched per token, so the working set fits in page cache and no
20 GB copy into the DRAM tier is needed.

### Pinning safety

Pinned (`cudaHostAlloc` / `cudaMallocHost`) pages cannot be reclaimed by the
kernel, so over-committing turns the machine into swap soup. VUGVA therefore:

* refuses to pin beyond **50% of physical RAM** and always keeps ≥ 2 GiB free
  (`VUGVA_PIN_MB` overrides the cap on dedicated inference boxes),
* falls back to pageable DRAM when a mapped allocation fails and **disables
  zero-copy for the rest of the run**, so the scheduler degrades to `hybrid`
  instead of dereferencing non-GPU-addressable memory.

## Architecture

```
llama-cpp-VUGVA/
├── ggml-vugva/              # VUGVA ggml backend
│   ├── ggml-backend-vugva.h # Public API
│   └── ggml-backend-vugva.cpp
├── llama.cpp/               # Upstream llama.cpp (submodule)
├── CMakeLists.txt
└── README.md
```

## How it works

1. **Placement**: `-ngl` layers are packed into VRAM as a contiguous hot block
   (`VUGVA_VRAM_MB` / free-VRAM budget); the rest go to the VUGVA DRAM tier
2. **Warm tier**: VUGVA_DRAM buffers are allocated from the DRAM pool
   (NUMA-aware, page-locked when it fits)
3. **Zero-copy mode**: the warm tier is `cudaHostAllocMapped` pinned memory, so
   ggml-cuda kernels read it directly over PCIe through UVA — the scheduler
   treats the warm tier as GPU-addressable and emits no H2D copies
4. **Hybrid mode**: the warm tier is computed by the CPU backend and only
   activations cross PCIe
5. **Compute**: CUDA is always the compute engine; VUGVA only manages memory
   placement and the DRAM pool

```
DRAM pool (32 GB)                    VRAM (8 GB)
┌──────────────────┐               ┌──────────────┐
│ tensor_0 (cold)  │   promote     │ tensor_A     │
│ tensor_1 (cold)  │ ───────────→  │ tensor_B     │
│ tensor_2 (warm)  │ ←───────────  │ tensor_C     │
│ ...              │   evict       │              │
└──────────────────┘               └──────────────┘
```

## Requirements

- CUDA 11.0+ (tested with 13.1)
- NVIDIA GPU with compute capability 6.0+ (tested on RTX 4060, sm_89)
- 32+ GB system RAM
- Linux (NUMA-aware allocation)

## Integration with upstream

This repo tracks llama.cpp as a submodule. To update:

```bash
cd llama.cpp
git pull origin master
cd ..
git add llama.cpp
git commit -m "chore: update llama.cpp"
```

## License

AGPL-3.0 (same as llama.cpp + VUGVA)

## Citation

```bibtex
@software{vugva2026,
  title   = {VUGVA: Virtual Unified GPU VRAM Architecture},
  author  = {Irfan Mahir},
  year    = {2026},
  doi     = {10.5281/zenodo.21549808},
  url     = {https://github.com/Mr-DS-ML-85/VUGVA}
}
```
