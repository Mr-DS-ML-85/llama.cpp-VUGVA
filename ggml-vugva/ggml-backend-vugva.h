/// ggml-backend-vugva — VUGVA tiered memory backend for ggml/llama.cpp
///
/// Implements the ggml_backend interface on top of VUGVA's three-tier
/// hierarchy (VRAM → DRAM → SSD). Tensor data lives in DRAM and is
/// promoted to VRAM on demand via CPU-bypass DMA (paper §3.3, §4).
///
/// Usage:
///   1. Call ggml_backend_vugva_reg() to get the backend registration.
///   2. Pass it to ggml_backend_sched_new() alongside the CUDA backend.
///   3. Tensors allocated through VUGVA's buffer type go through the
///      tiered pool. Promotions happen on access.

#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// VUGVA backend configuration.
typedef struct {
    int    gpu_id;          ///< CUDA device ordinal (default: 0).
    size_t dram_pool_bytes; ///< DRAM pool size in bytes (default: 32 GB).
    int    spill_enabled;   ///< Enable SSD cold tier (default: 0).
    const char * spill_path;///< Spill file path (if spill_enabled).
} ggml_backend_vugva_params;

/// Default parameters (32 GB DRAM, GPU 0, no spill).
GGML_API ggml_backend_vugva_params ggml_backend_vugva_default_params(void);

/// Get the VUGVA backend registration.
///
/// Call once; returns a singleton. The registration manages VUGVA's
/// tiered pool and exposes it as a ggml device.
GGML_API ggml_backend_reg_t ggml_backend_vugva_reg(void);

/// Get a VUGVA buffer type for a specific GPU.
///
/// Buffers allocated through this type live in VUGVA's DRAM pool and
/// are promoted to VRAM on the specified GPU when accessed.
GGML_API ggml_backend_buffer_type_t ggml_backend_vugva_buffer_type(int gpu_id);

/// Initialize the VUGVA backend with custom parameters.
///
/// Must be called before any tensors are allocated. Returns 0 on success.
GGML_API int ggml_backend_vugva_init(ggml_backend_vugva_params params);

/// True when `buft` is the VUGVA_DRAM tier buffer type.
///
/// The CUDA backend uses this to accept warm-tier tensors in zero-copy mode
/// (VUGVA_ZEROCOPY=1): tensors there are GPU-addressable pinned DRAM, so the
/// CUDA kernels can read them directly over PCIe with no CPU involvement.
GGML_API bool ggml_backend_vugva_buft_is_vugva(ggml_backend_buffer_type_t buft);

/// True when this GPU can address mapped host memory via UVA (zero-copy warm
/// tier). Returns 0 when unavailable. Safe to call before backend init.
GGML_API int ggml_backend_vugva_zerocopy_available(void);

#ifdef __cplusplus
}
#endif
