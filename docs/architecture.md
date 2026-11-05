# Architecture

This document describes how the VUGVA integration works inside llama.cpp, what
each layer owns, and why the design looks the way it does.

## 1. The three tiers

```
            ┌──────────────────────────────────────────────┐
   hot      │ VRAM (8 GB)          ggml CUDA buffer         │  cudaMalloc
            ├──────────────────────────────────────────────┤
   warm     │ DRAM (32 GB pool)    VUGVA_DRAM buffer type   │  cudaHostAllocMapped
            │                      (pinned, UVA-mapped)     │  or pageable mmap
            ├──────────────────────────────────────────────┤
   cold     │ NVMe / page cache    fallback host memory     │  mmap, demand paging
            └──────────────────────────────────────────────┘
```

The warm tier is a first-class ggml buffer type (`VUGVA_DRAM`), not a compute
backend. Its memory is allocated from the Rust VUGVA pool
(`ggml-vugva/vugva-bridge.rs` ↔ FFI `vugva_pool_create/alloc/write/access/…`).

The cold tier is ordinary pageable memory. VUGVA does not need an explicit SSD
path: when the DRAM pool cannot hold everything, the allocation becomes
pageable and the Linux page cache moves cold pages to NVMe on demand. This is
why models larger than the pool still load.

## 2. Placement (Algorithm 1)

Implemented in `llama.cpp/src/llama-model.cpp` in `load_tensors()`.

1. Compute `budget = free VRAM − reserve`, where
   `reserve = max(total_VRAM/8, 1 GiB)` and can be overridden with
   `VUGVA_RESERVE_MB`; `VUGVA_VRAM_MB` caps the budget too.
2. Sum per-layer weight bytes from the GGUF metadata (`blk.N.*`), keeping two
   counters per layer:
   * `layer_bytes` — the whole layer,
   * `layer_bytes_dense` — the layer without sparse expert tensors.
3. Compute the output layer cost. Tied models duplicate `token_embd` into the
   output buffer (`TENSOR_DUPLICATED`), so the embedding is counted there.
4. Decide the MoE fast path (below).
5. Fill the hot tier **in order** — output layer first, then layers
   `i_gpu_start…` — stopping at the first layer that does not fit. A contiguous
   block matters: every gap becomes a CPU↔GPU graph split per token.
6. Tensors of layers marked warm are created in the warm buft list; the rest in
   the GPU list. Overrides (see §4) win over both.

Placement is logged:

```text
[VUGVA] mode=zerocopy VRAM budget 6266 MiB (free 7290, reserve 1024)
        -> 23/65 layers + output on GPU, remainder -> VUGVA_DRAM tier
```

## 3. Execution modes

### `zerocopy`

The warm tier is allocated with
`cudaHostAlloc(cudaHostAllocMapped | cudaHostAllocPortable)` and the device
alias is obtained with `cudaHostGetDevicePointer`. Under UVA the device pointer
equals the host pointer, so a single `tensor->data` is valid on both sides.

`ggml-cuda` accepts `VUGVA_DRAM` as a supported buffer type when
`VUGVA_ZEROCOPY=1`, so the scheduler assigns warm-layer nodes to the GPU. The
kernels read the weights over PCIe directly. The CPU never touches the weights.

Result: **one graph split** for the whole model, no H2D copies of weights.

Cost: warm-tier reads are PCIe-bandwidth-bound. On a dense model this is
roughly on par with CPU compute over RAM bandwidth; on a sparse MoE it wins
decisively because only the active experts are read.

### `hybrid`

Warm layers are computed by the CPU backend; only activations cross PCIe.
This is the safe fallback when the warm tier does not fit in RAM. It produces
many graph splits (one per CPU/GPU boundary), so expect lower throughput.

### `auto` (default)

Picks `zerocopy` when all of the following hold:

* the GPU can map host memory (`ggml_backend_vugva_zerocopy_available()`),
* the estimated warm tier (`model − hot`) plus allocator slack fits
  `MemAvailable` with a `max(4 GiB, warm/4)` margin,
* the warm tier is below the pin cap (50 % of RAM by default).

Otherwise it picks `hybrid`. The decision is logged.

## 4. Tensor overrides and the MoE fast path

llama.cpp resolves each tensor's buffer type in
`llama_model_loader::create_tensor()` in this order:

1. explicit `tensor_buft_overrides` (`-ot`, `-cmoe`, `-ncmoe`),
2. VUGVA auto-MoE override (new),
3. the layer's buffer-type list (GPU list for hot layers, warm list otherwise).

### Auto-MoE

If `n_expert > 0`, no explicit overrides were given, and the model does not fit
the VRAM budget, `llama-model.cpp` sets `ml.auto_cpu_experts`. The loader then
routes tensors matching

```text
\.ffn_(up|down|gate|gate_up)_(ch|)exps
```

to `ggml_backend_cpu_buffer_type()`. With mmap enabled, the CPU buffer type is
backed by the GGUF mapping: **zero copies**, zero extra RAM, and only the
active experts are touched per token. Shared experts (`ffn_*_shexp`) are not
matched and stay hot.

Why this is the right metric: a 35B MoE with 256 experts and 8 active touches
~1/32 of the expert stack per token. Streaming the full stack through PCIe or
copying it into the DRAM pool is wasted work; paging in the active experts from
the (usually cached) file is cheaper.

### Explicit overrides

`-cmoe` overrides all expert tensors; `-ncmoe N` overrides the first N layers.
In both cases the loader routes them to plain CPU memory (not the DRAM tier),
avoiding a full copy of the expert stack into the pool. If a user asks for the
DRAM tier explicitly (`-ot '…=VUGVA_DRAM'` in upstream terms), that is honored
instead.

## 5. Why not a compute backend?

An earlier iteration registered VUGVA as a full scheduler backend for warm
layers. The CUDA↔VUGVA boundary produced **>1000 graph splits** and crashed in
`sched_reserve` on real models. Keeping VUGVA as a buffer type (plus UVA in
zero-copy mode) gives the same memory behaviour with none of the scheduling
cost. See `IMPLEMENTATION.md` for the post-mortem.

## 6. Memory safety model

The warm tier can be pinned (`cudaMallocHost` / `cudaHostAllocMapped`) or
pageable. Pinning is the only operation that can take the machine down, so it is
gated in the allocator:

```
pinned_total + request <= min(VUGVA_PIN_MB, 50% of RAM)
request + pinned_total + 2 GiB <= MemAvailable
```

If either check fails, the allocator refuses the pin and returns pageable
memory. If a mapped allocation fails in zero-copy mode,
`VUGVA_ZEROCOPY` is cleared so `ggml-cuda` stops accepting `VUGVA_DRAM`, and
the scheduler falls back to the CPU. This makes over-commitment a performance
problem, never a correctness or stability problem.

## 7. Source map

| Path | Role |
|---|---|
| `ggml-vugva/ggml-backend-vugva.cpp` | buffer type, DRAM pool bridge, zero-copy alloc, pin safety |
| `ggml-vugva/vugva-bridge.rs` | Rust ↔ C ABI for the pool |
| `llama.cpp/src/llama-model.cpp` | mode selection, budget, hot block, `hparams.n_expert` logic |
| `llama.cpp/src/llama-model-loader.cpp/.h` | tensor → buffer type resolution, auto-MoE, overrides |
| `llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu` | `supports_buft` for `VUGVA_DRAM`, async H2D DMA |
| `llama.cpp/examples/simple/simple.cpp` | the bundled `llama-cli` |
| `tests/test_vugva.cpp` | end-to-end integration test |
