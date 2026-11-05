# VUGVA Integration — Implementation Status

> Version: integration **0.2.0** · vendored llama.cpp **b12119** (`3de52ac3`,
> 2026-11-04) · VUGVA core `898e5e1` · paper DOI 10.5281/zenodo.21549808.
> See `README.md` (version matrix) and `docs/` for user-facing docs and
> `CHANGELOG.md` for the history.

> This file previously described a plan to make VUGVA a *compute* backend in the
> llama.cpp scheduler (`supports_op=true`, VUGVA claims every op, delegates to
> CUDA). That approach was tried and reverted: it produced >1000 graph splits
> and crashed in `sched_reserve`. The architecture below is what actually ships
> and is verified to run.

## Architecture (current)

VUGVA is a **buffer type + memory tier manager**, never a compute backend:

```
GGUF file ──mmap──► VUGVA_DRAM buffer (warm tier, pool-accounted)
                      │
                      ├─ hot: layers capped to free VRAM ─► CUDA0 buffer (VRAM)
                      ├─ warm: everything else          ─► VUGVA_DRAM (page cache / pinned)
                      └─ cold: pool budget exceeded     ─► SSD via page cache (mmap file)

compute: CUDA runs the hot-tier layers; DRAM-tier weights are read directly by
the CPU backend; llama.cpp's op_offload moves small/batched ops to CUDA.
```

Key properties enforced in `ggml-vugva/ggml-backend-vugva.cpp`:

| Hook | Behavior | Why |
|------|----------|-----|
| `dev.supports_op` | `true` | consulted only by `weight_buft_supported()`/`select_buft()` for **weight placement**. VUGVA is skipped as a scheduler backend in `llama-context.cpp` and has `graph_compute = NULL`, so it never executes ops. |
| `dev.get_buffer_type` | singleton per device | llama.cpp compares buffer-type **pointers** (`is_default_buft`); a fresh instance per call broke the mmap zero-copy import and leaked. |
| `dev.get_host_buffer_type` | `NULL` | otherwise llama.cpp's "avoid host buffers when using mmap" rule rewrites every VUGVA selection back to the plain CPU buffer. |
| `buft.alloc_buffer` | pool-accounted (`vugva_alloc`), `cudaMallocHost` storage, anonymous-mmap fallback | page-locked memory ⇒ scheduler H2D copies are true DMA (paper §5.1); allocation must not fail when the pool is merely over budget. |
| `dev.buffer_from_host_ptr` | wraps the GGUF mmap range | default load mode: zero-copy weights, page cache = DRAM tier, file = cold tier. Over-budget imports are accepted with a logged warning. |
| `buft.get_max_size` | `SIZE_MAX` | used by ggml to chunk allocations, not as a capacity gate. |

Placement changes in `llama.cpp/src/llama-model.cpp`:

1. `make_gpu_buft_list()` does **not** prepend VUGVA (an earlier revision did —
   that sent even `-ngl` layers to DRAM and left the GPU idle). VUGVA enters as
   the first entry of `make_cpu_buft_list()` via the generic ACCEL loop, which
   is appended as fallback: GPU layers keep CUDA first, everything else lands in
   VUGVA_DRAM (input layer, non-offloaded layers, KV of those layers).
2. **Algorithm 1 / VRAM budget** in `load_tensors()`: before assigning layers,
   free VRAM minus a reserve (`max(1 GiB, total/8)`) is compared against
   per-layer weight sizes (`blk.N.*` from the GGUF). Requested layers that no
   longer fit fall back to the DRAM tier instead of aborting with
   `cudaMalloc failed: out of memory`. The output layer is budgeted first
   (largest matmul per token). Logged as:

   ```
   load_tensors: [VUGVA] VRAM budget 5973 MiB (free 6997, reserve 1024) -> 10/41 layers + output on GPU, remainder -> VUGVA_DRAM tier (requested -ngl 99)
   ```

## Root causes fixed (the original broken state)

1. `dev.supports_op` returned `false` → every tensor rejected `VUGVA_DRAM` →
   weights fell through to the CUDA buffer type → one contiguous
   **19.9 GB `cudaMalloc` → OOM** on the 8 GB card (the reported 35B failure).
2. `get_buffer_type` allocated a **new instance per call** → `is_default_buft`
   never matched → mmap zero-copy import path was dead, plus a leak per call.
3. `make_gpu_buft_list()` prepended VUGVA **before CUDA** → with (1) fixed,
   all weights (including `-ngl` layers) would have gone to DRAM, GPU idle.
4. `get_host_buffer_type()` returned VUGVA's own buft → llama.cpp's mmap rule
   silently rewrote VUGVA selections to the CPU buffer.
5. `vugva_alloc` failure (pool over budget) hard-failed the import → any model
   larger than the pool could not load at all.
6. `get_max_size()` returned the 32 GB pool size (ggml uses it to chunk).
7. `tests/test_vugva.cpp` built a `std::vector` from the **negated** return of
   `llama_tokenize` → `std::length_error` abort; it also forced `n_gpu_layers=0`.
8. CMake `VUGVA_DRAM_GB` was never wired into the backend (32 GB hardcoded),
   and the `vugva_test` target referenced by README/CI/CONTRIBUTING did not exist.

## Verified runs (RTX 4060 8 GB + 29 GB DDR5, RTX 4060 sm_89, CUDA 13.2)

| Case | Result |
|------|--------|
| `vugva_test` (0.8B) | `ALL TESTS PASSED`, 64 tokens, 215 tok/s |
| 0.8B Q4_K_M, `-ngl 99` | 25/25 layers + output hot, graph splits=1, ~150–185 tok/s |
| 0.8B Q4_K_M, `VUGVA_VRAM_MB=100`, zerocopy | 8/25 hot (99 MiB VRAM), 395 MiB warm, graph splits=1, 25–45 tok/s, output identical to full-GPU |
| 27B Q4_K_M (17.4 GB dense), `-ngl 99` auto→zerocopy | 24/65 layers hot (6.3 GB), 9.5 GB warm, **graph splits=1**, 1.15–1.18 tok/s, coherent |
| 27B Q4_K_M, `-ngl 99` hybrid | 14/65 hot, graph splits=70/114, 0.85–1.2 tok/s |
| 27B Q4_K_M, `-ctk q4_0 -ctv q4_0`, ctx 4096 | KV cache 256 MiB → 72 MiB, same placement |
| 35B MoE Q4_K_M (21.7 GB), `-ngl 99` hybrid | 12/41 hot, 19.8 GB VUGVA_DRAM, graph splits=46–60, 2.1–2.3 tok/s |
| 35B MoE, `-ngl 99 -cmoe -ctk q4_0 -ctv q4_0`, warm cache | 41/41 attention layers hot (1.3 GB VRAM), experts mmap'd (`CPU_Mapped` 19.5 GB, no copy), graph splits=82, **27–36 tok/s** |
| 35B MoE, same, cold page cache | ~4 tok/s (NVMe-bound first pass) |
| 35B MoE, `VUGVA_MODE=zerocopy` forced | pin refused (>50% RAM) → zero-copy auto-disabled → pageable DRAM + hybrid, exit 0 (no OOM) |
| 35B with `VUGVA_DRAM_GB=16` (< model size) | exit 0, cold-tier warning logged, still runs |

### Mode selection (llama-model.cpp, Algorithm-1 block)

* Hot tier is a **contiguous prefix** of the requested layers (no holes, so no
  extra graph splits); the output layer is budgeted first and its tied
  embedding is counted (llama.cpp duplicates `token_embd` into the output
  buffer via `TENSOR_DUPLICATED`).
* Tensors rewritten by `--cpu-moe` / `-ncmoe` (`tensor_buft_overrides`) are
  excluded from the hot-tier budget, so MoE models can keep all attention/SSM
  layers in VRAM.
* `auto` picks `zerocopy` only when the estimated warm tier fits in RAM with a
  25%/4 GiB margin **and** stays below 50% of physical RAM; otherwise `hybrid`.
* Pinned allocations are capped (50% RAM, ≥2 GiB headroom, `VUGVA_PIN_MB`) and
  a failed mapped allocation downgrades the run to `hybrid` instead of crashing.
* CPU-buft overrides (`--cpu-moe`) now use the plain CPU buffer (mmap-backed)
  instead of the VUGVA DRAM tier, avoiding a full copy of the expert stack.

## Not implemented (paper claims without corresponding code)

Be honest about the gap between paper and code:

- **Look-Ahead Attention prefetch (Algorithm 3)** — no background DRAM→VRAM
  weight streaming exists (`vugva`'s `prefetch.rs`/`dma.rs` are not called by
  the ggml backend). It is unnecessary for the current placement: hot-tier
  weights are already resident in VRAM and warm-tier weights are read directly
  by the CPU without promotion.
- **Runtime promotion/eviction of individual tensors** (membrane-driven tiering)
  — placement is static for the lifetime of the run (`-ngl` + VRAM budget).
- **GPUDirect Storage / IOMMU P2P** — the cold tier is the OS page cache over
  the GGUF file on NVMe, not GDS.
- The paper's measured numbers (54.8% throughput, 84.6% CPU reduction, …) are
  not reproduced or asserted by this codebase.

## Pool accounting

The Rust `TieredPool` (`/home/irfan/Desktop/VUGVA/vugva`) provides NUMA-local
registration and bookkeeping (`vugva_alloc`/`vugva_dealloc`/`vugva_pool_info`);
the ggml buffer owns the actual storage (mmap import or `cudaMallocHost`).
Pool size: `cmake -DVUGVA_DRAM_GB=N`, runtime override `VUGVA_DRAM_GB=N`.
Note: the pool attempts a whole-range `cuMemHostRegister` at creation; on a
32 GB machine registering 32 GB fails harmlessly (bookkeeping still works).
