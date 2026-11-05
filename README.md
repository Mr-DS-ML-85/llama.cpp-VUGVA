# llama-cpp-VUGVA

**Run LLMs larger than your VRAM.** VUGVA is a software-defined tiered-memory
layer for `llama.cpp`: model weights are automatically split across
**VRAM (hot) → DRAM (warm) → SSD (cold)** instead of failing with a CUDA
out-of-memory error.

```
            hot                 warm                 cold
     ┌──────────────┐   ┌──────────────────┐   ┌────────────────┐
     │   VRAM 8 GB  │ ← │    DRAM 32 GB    │ ← │  NVMe (page    │
     │              │   │  (pinned / UVA)  │   │  cache / SSD)  │
     └──────────────┘   └──────────────────┘   └────────────────┘
        CUDA compute       GPU-readable          mmap zero-copy
```

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21549808-blue)](https://doi.org/10.5281/zenodo.21549808)
[![llama.cpp](https://img.shields.io/badge/llama.cpp-b12119%20(3de52ac3)-green)](https://github.com/ggml-org/llama.cpp)
[![release](https://img.shields.io/badge/release-v0.2.0-orange)](CHANGELOG.md)
[![license](https://img.shields.io/badge/license-AGPL--3.0-lightgrey)](LICENSE)

---

## Contents

- [Version matrix](#version-matrix)
- [What this is (and is not)](#what-this-is-and-is-not)
- [Quick start](#quick-start)
- [How tiering works](#how-tiering-works)
- [The MoE fast path](#the-moe-fast-path)
- [CLI flags](#cli-flags)
- [Configuration](#configuration)
- [Measured results](#measured-results)
- [Pinning safety](#pinning-safety)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)

---

## Version matrix

| Component | Version | Exact reference |
|---|---|---|
| **This integration (VUGVA backend for llama.cpp)** | **v0.2.0** | tag `v0.2.0` |
| **llama.cpp upstream** | **b12119** | commit [`3de52ac3`](https://github.com/ggml-org/llama.cpp/commit/3de52ac38818978d6da8f386eda43a856e53714a) · 2026-11-04 |
| **VUGVA core (Rust)** | `898e5e1` | [Mr-DS-ML-85/VUGVA](https://github.com/Mr-DS-ML-85/VUGVA) · `898e5e1bfd0259f2a49c39345fe307c85bc3c22c` |
| **Paper** | July 2026 | **DOI: [10.5281/zenodo.21549808](https://doi.org/10.5281/zenodo.21549808)** |

The upstream llama.cpp source is **vendored** in `llama.cpp/` (tracked as
regular files, not a submodule, and **trimmed to build-only files** — it has
its own `llama.cpp/README.md`), so a checkout of this repo is
self-contained. Git tags:

| Tag | Meaning |
|---|---|
| `v0.2.0` | this release |
| `llama.cpp-b12119-3de52ac3` | exact vendored upstream snapshot |

---

## What this is (and is not)

**It is** a ggml *buffer type* plus a placement policy:

* `VUGVA_DRAM` — a real backend buffer type whose memory comes from the VUGVA
  DRAM pool (NUMA-aware, page-locked when safe, optionally GPU-mapped).
* **Algorithm 1 placement** — `-ngl` requests GPU layers; the hot tier is
  filled with a contiguous prefix that actually fits, and the remainder is
  served from the warm tier instead of failing.
* Two execution modes for warm-tier weights: **zero-copy** (CUDA reads pinned
  DRAM over PCIe via UVA) and **hybrid** (the CPU backend computes warm layers).
* A **MoE fast path**: when a sparse-expert model does not fit VRAM, expert
  tensors stay in mmap'd CPU memory while attention/SSM stays hot on the GPU.

**It is not** a compute backend. CUDA is always the compute engine; VUGVA never
runs graphs. This is deliberate: routing every layer through a separate
scheduler backend produced >1000 graph splits and a `sched_reserve` crash in an
earlier iteration (see `IMPLEMENTATION.md`).

---

## Quick start

```bash
# 1. Build the VUGVA core (Rust)
git clone https://github.com/Mr-DS-ML-85/VUGVA.git
cd VUGVA/vugva && cargo build --release        # -> target/release/libvugva.so

# 2. Build llama.cpp + VUGVA
cd /path/to/llama-cpp-VUGVA
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
      -DVUGVA_DIR=/path/to/VUGVA/vugva
cmake --build build -j

# 3. Smoke test (integration test binary)
./build/vugva_test model.gguf "Once upon a time"

# 4. Generate text
./build/llama-cli -m model.gguf -ngl 99 "Write a short story" -n 64
```

Requirements: Linux, CUDA 11+, an NVIDIA GPU, and system RAM for the warm tier.

---

## How tiering works

Weight placement happens at load time, before any tensor is allocated:

1. **Budget** — free VRAM minus a reserve (`max(total/8, 1 GiB)`,
   `VUGVA_RESERVE_MB`) is the hot-tier budget; `VUGVA_VRAM_MB` can force it.
2. **Hot block** — layers are assigned as a **contiguous prefix** (no holes:
   holes would add graph splits on every token). The output layer is budgeted
   first, including the duplicated embedding of tied models.
3. **Warm tier** — everything else goes to `VUGVA_DRAM`.
4. **Cold tier** — beyond the DRAM pool, VUGVA falls back to pageable memory,
   so the OS spills to NVMe and pages weights back on demand.

### Execution modes

| Mode | Warm-tier storage | Warm-tier compute | When selected by `auto` |
|---|---|---|---|
| `zerocopy` | `cudaHostAllocMapped` pinned DRAM (UVA) | **CUDA** reads over PCIe | warm tier fits in RAM with margin and stays under the pin cap |
| `hybrid` | pinned or pageable DRAM | **CPU** backend | otherwise |

In `zerocopy` mode the scheduler sees the warm tier as GPU-addressable, so
there is **one graph split** for the whole model instead of one H2D copy per
warm layer. The CPU never touches weight data.

```
hybrid  :  layer0..k (CUDA)  ⇄  CPU copies  ⇄  layer k+1..N (CPU)
zerocopy:  layer0..N (all CUDA)  ── reads ──▶  warm tier in DRAM (UVA)
```

Force a mode with `VUGVA_MODE=hybrid|zerocopy` (default `auto`).

---

## The MoE fast path

Sparse Mixture-of-Experts models (Qwen-MoE, DeepSeek, Mixtral, …) do not fit in
consumer VRAM, but only a handful of experts are active per token. Copying the
whole expert stack into the DRAM tier wastes RAM and does not help: those
weights are read once per token anyway.

**Since v0.2.0 this is automatic.** If `n_expert > 0` and the model does not
fit the VRAM budget (and no explicit tensor overrides were given):

* sparse expert tensors (`blk.N.ffn_{up,down,gate,gate_up}_exps`) are routed to
  **plain CPU memory (mmap-backed, zero copy)**,
* attention / SSM / shared experts stay **hot in VRAM**,
* the hot-tier budget only counts the dense part of each layer.

No flags required. Disable with `VUGVA_AUTO_MOE=0`. The manual equivalents are
still available: `-cmoe` / `--cpu-moe` (all experts) and `-ncmoe N`
(first N layers).

Example load log on a 35B MoE with 8 GB VRAM:

```text
load_tensors: [VUGVA] MoE model does not fit VRAM (20424 MiB > 6284 MiB) ->
              auto-routing expert tensors to CPU/mmap (VUGVA_AUTO_MOE=0 to disable)
load_tensors: [VUGVA] MoE fast path: sparse experts -> CPU/mmap, attention/SSM stays hot in VRAM
load_tensors:        CUDA0 model buffer size =  1302.90 MiB   # attention/SSM hot
load_tensors:   CPU_Mapped model buffer size = 19486.84 MiB   # experts, mmap zero-copy
```

---

## CLI flags

The bundled `llama-cli` is a compact CLI built on `libllama`:

| Flag | Description |
|---|---|
| `-m, --model FILE` | GGUF model |
| `-n, --n-predict N` | tokens to generate (default 32) |
| `-ngl N` | layers to offload (default 99) |
| `-c N` | context size (default prompt + n) |
| `-b N` | batch size |
| `-t N` | threads |
| `-fa on\|off\|auto` | flash attention (auto-enabled for quantized KV) |
| `-ctk TYPE` / `-ctv TYPE` | KV cache types: `f16`, `bf16`, `q8_0`, `q4_0`, … |
| `-cmoe, --cpu-moe` | keep all MoE expert weights in CPU/DRAM |
| `-ncmoe N, --n-cpu-moe N` | first N layers' experts in CPU/DRAM |
| `--temp`, `--top-k`, `--top-p`, `--min-p`, `-rp`, `--seed` | sampling |
| `-p PROMPT` | prompt (or pass it positionally) |
| `-h, --help` | usage |

Examples:

```bash
# all weights hot
./build/llama-cli -m qwen0.8b.gguf -ngl 99 "The capital of France is" -n 32

# dense model larger than VRAM: automatic tiering + zero-copy
./build/llama-cli -m qwen27b.gguf -ngl 99 "Write a story" -n 32

# MoE: automatic fast path, int4 KV cache, sampling
./build/llama-cli -m ornith35b.gguf -ngl 99 -ctk q4_0 -ctv q4_0 \
    --temp 0.7 --repeat-penalty 1.1 "Explain why the sky is blue" -n 64
```

---

## Configuration

All knobs are environment variables checked at load time.

| Variable | Default | Description |
|---|---|---|
| `VUGVA_MODE` | `auto` | `auto` \| `hybrid` \| `zerocopy` |
| `VUGVA_VRAM_MB` | auto | force the hot-tier budget (MiB); useful for testing spill |
| `VUGVA_RESERVE_MB` | `max(total/8, 1 GiB)` | VRAM kept free for KV cache/activations |
| `VUGVA_PIN_MB` | 50 % of RAM | cap on pinned warm-tier memory |
| `VUGVA_DRAM_GB` | 32 | DRAM pool size (runtime override of the CMake value) |
| `VUGVA_AUTO_MOE` | `1` | auto-route experts to CPU/mmap when the model exceeds VRAM |
| `VUGVA_ZEROCOPY` | set by mode | internal; makes the CUDA backend accept `VUGVA_DRAM` |
| `GGML_OP_OFFLOAD_MIN_BATCH` | 32 | min batch for op offload (llama.cpp) |
| `GGML_SCHED_DEBUG` | off | scheduler debug output |

CMake options: `-DVUGVA_ENABLE=ON|OFF`, `-DVUGVA_DRAM_GB=N`,
`-DVUGVA_DIR=/path/to/VUGVA/vugva`, `-DGGML_CUDA=ON`.

---

## Measured results

RTX 4060 (8 GB, sm_89), 29 GB DDR5, NVMe; CUDA 13.2; llama.cpp b12119.

| Model | Configuration | Decode |
|---|---|---|
| Qwen3.5 0.8B Q4_K_M | `-ngl 99` (25/25 hot) | **~150–188 t/s** |
| Qwen3 27B dense Q4_K_M (17.4 GB) | `-ngl 99`, auto → zerocopy (23–24/65 hot) | **~1.15 t/s** |
| Qwen3 27B dense, hybrid | `-ngl 99`, `VUGVA_MODE=hybrid` | ~0.85–1.2 t/s |
| Ornith 35B MoE Q4_K_M (21.7 GB) | `-ngl 99` before the MoE fast path | ~2.1–2.3 t/s |
| Ornith 35B MoE | `-ngl 99`, auto MoE, warm page cache | **~20–36 t/s** |
| Ornith 35B MoE | same, cold page cache (first pass) | ~4 t/s (NVMe-bound) |

KV-cache quantization: 27B, 4 K context → KV buffer **256 MiB (f16) → 72 MiB
(q4_0)**, same placement.

Memory behaviour: zero-copy warm tier on the 27B → **graph splits = 1** (vs 70
in hybrid). 0.8B forced tier (`VUGVA_VRAM_MB=100`) → 8/25 layers hot, CUDA0
buffer exactly 99.4 MiB, output identical to the full-GPU run.

Reproduction details and methodology are in [docs/benchmarks.md](docs/benchmarks.md).

---

## Pinning safety

Pinned host pages (`cudaHostAlloc`, `cudaMallocHost`) cannot be reclaimed by
the kernel. Over-committing RAM therefore does not degrade gracefully — the
machine starts swapping and can be OOM-killed (this happened during
development when a 20 GB warm tier was pinned on a 29 GB box).

VUGVA therefore:

* caps pinned memory at **50 % of physical RAM** and keeps **≥ 2 GiB**
  headroom (`VUGVA_PIN_MB` overrides the cap for dedicated boxes),
* refuses the pin and uses **pageable DRAM** when the limit would be exceeded,
* if a mapped allocation fails, **disables zero-copy for the run** and falls
  back to `hybrid`, so the scheduler never dereferences non-GPU memory.

A refused pin is not an error: the run continues in hybrid mode.

---

## Troubleshooting

See [docs/troubleshooting.md](docs/troubleshooting.md). Common cases:

* **`refusing to pin …`** — expected on large models; the run degrades to
  pageable/hybrid. Raise `VUGVA_PIN_MB` only on a dedicated inference machine.
* **Slow first run, fast afterwards** — the mmap'd weights are being faulted in
  from NVMe. The page cache holds them on subsequent runs.
* **Quantized KV fails to load** — the KV types must be compiled into the CUDA
  FlashAttention kernels (`GGML_CUDA_FA_QUANTS`, e.g. `q4_0-q4_0`).
* **`ggml_backend_sched` splits > 100** — the model is in hybrid mode; the CPU
  is computing warm layers. Compare with `VUGVA_MODE=zerocopy`.

---

## Citation

If you use VUGVA or this integration, please cite the paper:

```bibtex
@article{ mahir2026vugva,
  author       = {Mahir, Irfan},
  title        = {VUGVA: A Software-Defined Virtual Unified GPU VRAM Architecture
                  with CPU-Bypass Hybrid Memory for Non-NVLink Multi-GPU Clusters},
  year         = {2026},
  month        = {July},
  doi          = {10.5281/zenodo.21549808},
  url          = {https://doi.org/10.5281/zenodo.21549808},
  note         = {Reference implementation: https://github.com/Mr-DS-ML-85/VUGVA}
}
```

Plain text:

> Irfan Mahir. *VUGVA: A Software-Defined Virtual Unified GPU VRAM Architecture
> with CPU-Bypass Hybrid Memory for Non-NVLink Multi-GPU Clusters.* July 2026.
> DOI: 10.5281/zenodo.21549808.
> https://github.com/Mr-DS-ML-85/VUGVA

The paper source is included as [`VUGVA_Paper.tex`](VUGVA_Paper.tex); a
machine-readable citation is in [`CITATION.cff`](CITATION.cff).

---

## Project layout

```
llama-cpp-VUGVA/
├── ggml-vugva/          # VUGVA ggml buffer type + DRAM pool bridge
├── llama.cpp/           # vendored llama.cpp b12119 + VUGVA patches
├── tests/test_vugva.cpp # integration test (build target: vugva_test)
├── docs/                # architecture, configuration, benchmarks, troubleshooting
├── VUGVA_Paper.tex      # paper source
├── CHANGELOG.md
└── CMakeLists.txt
```

How the vendored tree is maintained: `llama.cpp/` is a plain directory in this
repo (no submodule). The upstream git history used for rebasing lives outside
the tree in `../llama-cpp-upstream.git`; patches are re-applied per upstream
snapshot and the exact commit is recorded in the version matrix above.

## License

AGPL-3.0 (same as VUGVA core). See [LICENSE](LICENSE).
llama.cpp is MIT-licensed; see `llama.cpp/LICENSE`.
