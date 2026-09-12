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

## Architecture

```
llama-cpp-VUGVA/
├── ggml-vugva/              # VUGVA ggml backend
│   ├── ggml-backend-vugva.h # Public API
│   └── ggml-backend-vugva.cpp
├── tests/
│   └── test_vugva.cpp       # Integration test
├── llama.cpp/               # Upstream llama.cpp (submodule)
├── CMakeLists.txt
└── README.md
```

## How it works

1. **Allocation**: Tensor buffers go through VUGVA's DRAM pool (mmap + NUMA pinning + page-locking)
2. **Access**: When a tensor is accessed, VUGVA promotes it from DRAM → VRAM via async DMA
3. **Eviction**: Idle tensors are demoted back to DRAM, freeing VRAM for hot tensors
4. **Compute**: The CUDA backend handles all GPU compute; VUGVA only manages memory placement

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
