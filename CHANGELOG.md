# Changelog

All notable changes to the VUGVA integration for llama.cpp.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- **Documentation website** on GitHub Pages (`docs/`, source `main:/docs`):
  landing page, generated per-page docs with sticky table of contents, full-text
  search over 59 sections, and keyboard shortcuts (`/`, `Ctrl+K`, `?`, `g`+key
  navigation, `t` theme toggle, `j`/`k` scrolling).
- `tools/build_docs_site.py`: regenerates `docs/*.html` and
  `docs/assets/search-index.json` from the Markdown sources with pandoc.

## [0.2.1] — 2026-11-05

### Added

- **Tier manager: runtime promotion and LRU eviction of warm-tier weights**
  (`VUGVA_HOT_CACHE_MB`, opt-in). Called from `llama_context::graph_compute`;
  promotion is stable across steps so the CUDA-graph cache keeps working, and
  the cache refuses to enable when static placement already filled VRAM.
  Measured on a forced 100 MiB budget: 25.3 t/s → **149.1 t/s** (5.9×) with the
  warm tier promoted; a cache smaller than the working set thrashes (18.5 t/s,
  2912 evictions) and the counters say so.
- **Cold-tier look-ahead readahead** (`VUGVA_PRELOAD`, on by default):
  background `POSIX_FADV_WILLNEED` over the GGUF, paced, memory-safe, with a
  progress report on exit.
- **GDS / IOMMU P2P capability probe** printed at startup, so the cold-tier
  claim is measured rather than assumed (`iommu=no` here ⇒ page cache).
- `VUGVA_STATS=1`: tier-cache hit/miss/promotion/eviction counters.

### Changed

- `IMPLEMENTATION.md` now has a paper-vs-code table with the measured evidence,
  and states explicitly that Algorithm 3's DRAM→VRAM pipelining is excluded by
  llama.cpp's enqueue-whole-graph execution model, and that the paper's
  aggregate figures are not reproduced.

## [0.2.0] — 2026-11-05

Upstream: llama.cpp **b12119** (`3de52ac3`, 2026-11-04) · VUGVA core `898e5e1`.

### Added

- **Zero-copy warm tier**: `VUGVA_DRAM` buffers can be allocated with
  `cudaHostAllocMapped` (UVA). The CUDA backend accepts them, so warm layers are
  computed by the GPU reading pinned DRAM over PCIe. Graph splits drop to 1.
- **Mode selection** — `VUGVA_MODE=auto|hybrid|zerocopy` with RAM-fit and
  pin-cap checks; `auto` is the default.
- **Automatic MoE fast path** — when `n_expert > 0` and the model does not fit
  VRAM, sparse expert tensors are routed to plain CPU memory (mmap, zero copy)
  while attention/SSM stays hot. No flags needed (`VUGVA_AUTO_MOE=0` disables).
  Measured 2.1 → 19–36 tok/s on a 35B MoE (warm page cache).
- **Real CLI flags** in `build/llama-cli`: `-ctk/-ctv`, `-fa`, `-cmoe`,
  `-ncmoe`, `-c/-b/-t`, sampling (`--temp`, `--top-k`, `--top-p`, `--min-p`,
  `-rp`, `--seed`), `-p`, `-h`.
- Env knobs `VUGVA_VRAM_MB`, `VUGVA_RESERVE_MB`, `VUGVA_PIN_MB`,
  `VUGVA_AUTO_MOE`.
- Extensive documentation (`docs/`), `CITATION.cff`, version matrix in the
  README.

### Fixed

- **Pinning can no longer OOM the machine**: capped at 50 % of RAM with a 2 GiB
  headroom check; refusing a pin falls back to pageable memory, and a failed
  mapped allocation disables zero-copy for the run (`hybrid` fallback).
- Hot-tier budget now counts the **duplicated tied embedding**, so the CUDA
  allocation no longer silently exceeds budget.
- Hot layers are a **contiguous prefix** (no holes → no extra graph splits).
- `--cpu-moe`/`-ncmoe` tensors are routed to plain CPU memory instead of being
  copied into the 20 GB DRAM tier, and are excluded from the VRAM budget.
- `ggml_backend_buffer_type_i` updated for the new optional
  `alloc_buffer_n` / `get_alloc_size_n` slots.

### Changed

- `llama.cpp` is **vendored as regular files** (previously a gitlink without a
  `.gitmodules` entry, which GitHub could not resolve).
- Upgraded the vendored llama.cpp from `56381e4` to `3de52ac3` (b12119).

### Removed

- The `make_gpu_buft_list` VUGVA prepend (GPU buffer type must stay first so
  `-ngl` layers land in VRAM).

## [0.1.0] — 2026-09-13

Upstream: llama.cpp `e5aac25` (VUGVA working baseline).

### Added

- `VUGVA_DRAM` ggml buffer type backed by the VUGVA Rust DRAM pool.
- Algorithm-1 placement: measure free VRAM, cap the hot tier, spill the
  remainder to `VUGVA_DRAM` instead of failing with CUDA OOM.
- Pinned (`cudaMallocHost`) warm tier with async H2D DMA in `ggml-cuda`.
- `vugva_test` integration test.

### Verified

- 0.8B and 27B dense models load and generate; 27B runs with 14–19/65 layers
  hot and the rest in the DRAM tier.
