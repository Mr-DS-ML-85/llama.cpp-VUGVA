# VUGVA Tiered Memory Implementation Plan

## Problem Statement

VUGVA backend crashes with `std::length_error: cannot create std::vector larger than max_size()` during `sched_reserve`. The scheduler creates 1028 graph splits because:

1. VUGVA registers as ACCEL backend with `supports_op=true` → becomes primary backend (index 1, after CUDA at 0)
2. Model weights are loaded into CUDA buffers (via `make_gpu_buft_list` in llama-model.cpp)
3. Scheduler calls `ggml_backend_sched_backend_from_buffer()` which checks `supports_buft(backend, buffer_type) && supports_op(backend, op)`
4. For weights in CUDA buffers: CUDA supports them (index 0), but VUGVA doesn't (`supports_buft(VUGVA, CUDA_buft)=false`)
5. Every layer transition where a weight is on a different backend triggers `need_new_split=true` in `ggml_backend_sched_split_graph()` (line 1336)
6. 1028 splits → vector allocation overflow → crash

## Correct Architecture: VUGVA as Primary Backend with CUDA Delegation

```
Model Loading:  GGUF file → VUGVA DRAM (page-locked, NUMA-bound)
                         ↓
Scheduler:      VUGVA claims all ops (supports_op=true)
                Weights in VUGVA buffers → supports_buft(VUGVA, VUGVA_buft)=true
                No graph splits! ✓
                         ↓
Graph Compute:  For each node:
                  1. Promote source tensors DRAM → VRAM (cudaMemcpy H2D)
                  2. Delegate to internal CUDA backend
                  3. Copy outputs VRAM → DRAM (cudaMemcpy D2H)
                  4. Free VRAM copies
```

## Files to Modify

### 1. `src/llama-model.cpp` — Inject VUGVA buffer type into weight loading

**Function: `make_gpu_buft_list()` (line 1096)**

Current: Only adds CUDA's buffer types to the list.
Fix: Prepend VUGVA's buffer type so weights are loaded into VUGVA DRAM.

```cpp
static buft_list_t make_gpu_buft_list(ggml_backend_dev_t dev, llama_split_mode split_mode, const float * tensor_split) {
    buft_list_t buft_list;

    // ... existing split buffer type code ...

    // add the device default buffer type
    buft_list.emplace_back(dev, ggml_backend_dev_buffer_type(dev));

    // add the device extra buffer type (if any)
    // ... existing extra bufts code ...

    // === VUGVA: Prepend VUGVA DRAM buffer type ===
    // This makes weights load into page-locked DRAM instead of CUDA VRAM
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t accel_dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(accel_dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            const char * name = ggml_backend_dev_name(accel_dev);
            if (name && strstr(name, "VUGVA")) {
                auto * vugva_buft = ggml_backend_dev_buffer_type(accel_dev);
                if (vugva_buft) {
                    buft_list.emplace(buft_list.begin(), accel_dev, vugva_buft);
                }
                break;
            }
        }
    }

    return buft_list;
}
```

**Why this works:**
- Model loader tries buffer types in order, picks first that allocates successfully
- VUGVA's buffer type (page-locked DRAM) is tried first
- If VUGVA fails (not enough DRAM), CUDA's buffer type is tried as fallback
- Scheduler sees weights in VUGVA buffers → assigns ops to VUGVA → no splits

### 2. `ggml-vugva/ggml-backend-vugva.cpp` — Fix remaining bugs

**Bug in `buffer_from_host_ptr` (line 695-721):**
- Already fixed: `dev_ctx->gpu_id` now uses `(ggml_backend_vugva_device_context *)dev->context`

**`graph_compute` (line 453-544):**
- Already rewritten with per-node DRAM→VRAM staging
- Promotes source tensors, delegates to internal CUDA, copies results back

**`supports_op` (line 725-728):**
- Returns `true` — VUGVA claims all ops (correct for Approach B)

**`supports_buft` (line 730-733):**
- Returns `true` only for VUGVA's own buffer type (correct)

**`offload_op` (line 735-738):**
- Returns `true` — VUGVA offloads all ops to internal CUDA (correct)

### 3. `ggml-vugva/CMakeLists.txt` — Build configuration

Current state is correct:
- Links `ggml-cuda`, `CUDA::cudart`, `CUDA::cuda_driver`, `CUDA::cublas`
- VUGVA static lib linked from `/home/irfan/Desktop/VUGVA/vugva/target/release/libvugva.a`

### 4. `CMakeLists.txt` (top-level) — NO changes needed

Do NOT add `VUGVA_ENABLE` as a global compile definition. Do NOT modify ggml-cuda.cu.

## How Flags Are Preserved

| Flag | How it works | Impact |
|------|-------------|--------|
| `--n-gpu-layers` | Controls how many layers go to GPU vs CPU | Unaffected — GPU layers now use VUGVA DRAM |
| `--n-cpu-moe` | Keeps MoE expert layers on CPU | Unaffected — CPU layers still use CPU buffers |
| `--split-mode` | Controls tensor splitting across GPUs | Unaffected — VUGVA is single-device |
| MTP (multi-token prediction) | Uses CUDA's MTP kernels | Unaffected — CUDA handles all compute via VUGVA delegation |
| Flash Attention | Uses CUDA FA kernels | Unaffected — VUGVA delegates to internal CUDA |
| Pipeline Parallelism | Uses CUDA async compute | May need VUGVA to support events — future work |

## Scheduler Flow After Fix

```
1. Model loads weights into VUGVA DRAM buffers
2. Scheduler pass 1: assigns backends based on buffer location
   - Weights in VUGVA buffers → assigned to VUGVA (index 1)
   - Input tensors → assigned to CPU (index 2)
3. Scheduler pass 2-4: expand assignments
   - All weight-adjacent ops expand to VUGVA
4. Scheduler pass 5: create splits
   - All VUGVA nodes → same backend → no splits ✓
   - CPU nodes → same backend → no splits ✓
5. graph_compute called per split
   - VUGVA split: promote DRAM→VRAM, delegate to CUDA, copy back
   - CPU split: compute on CPU
```

## Build Commands

```bash
cd "/run/media/irfan/models/VUGVA test/llama-cpp-VUGVA/build"
rm -rf *
CUDACXX=/usr/local/cuda/bin/nvcc cmake .. \
    -DGGML_CUDA=ON \
    -DVUGVA_ENABLE=ON \
    -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

## Run Command

```bash
LD_LIBRARY_PATH=./bin:./lib:./ggml_build/src \
./llama-cli -m /path/to/model.gguf -n 32 -ngl 0 "Hello"
```

## Expected Output

```
[VUGVA] Pool created: DRAM=32 GB, NUMA=yes, pinned=no
[VUGVA] buft_alloc_buffer: allocated ggml_buf_0 (16.2 GB)
sched_reserve: graph splits = 1 (with bs=512), 1 (with bs=1)
... inference proceeds ...
```

## Current State

- [x] ggml-cuda.cu reverted (no changes)
- [x] CMakeLists.txt reverted (no VUGVA_ENABLE compile def)
- [x] ggml-backend-vugva.cpp rewritten with bug fixes
- [ ] llama-model.cpp: needs `make_gpu_buft_list` modification
- [ ] Build and test

## Key Code References

- Scheduler split logic: `ggml/src/ggml-backend.cpp:1297-1425`
- Backend selection: `ggml/src/ggml-backend.cpp:888-908`
- Buffer type check: `ggml/src/ggml-backend.cpp:1037-1056`
- Model weight loading: `src/llama-model.cpp:1447-1519`
- GPU buffer type list: `src/llama-model.cpp:1096-1142`
- VUGVA buffer type interface: `ggml-vugva/ggml-backend-vugva.cpp:345-352`
- VUGVA graph compute: `ggml-vugva/ggml-backend-vugva.cpp:453-544`
