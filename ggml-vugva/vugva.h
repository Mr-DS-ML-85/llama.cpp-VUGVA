/// VUGVA C API — for llama.cpp integration.
///
/// Opaque pool handle, DRAM/VRAM tiered memory with CPU-bypass DMA.
/// Link against libvugva.a (static) or libvugva.so (dynamic).
/// GPU: CUDA 11.0+, sm_60+. Pool: configurable DRAM + VRAM.

#ifndef VUGVA_H
#define VUGVA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque pool handle.
typedef struct VugvaPool VugvaPool;

/// Pool statistics.
typedef struct {
    size_t dram_capacity;   ///< Total DRAM pool size (bytes).
    size_t dram_used;       ///< DRAM currently handed out (bytes).
    int    numa_bound;      ///< 1 if DRAM is NUMA-local to the GPU.
    int    pinned;          ///< 1 if DRAM is page-locked (real DMA).
} VugvaPoolInfo;

/// Create a tiered pool with `dram_bytes` of DRAM on GPU `gpu_id`.
/// Returns NULL on failure.
VugvaPool *vugva_pool_create(int gpu_id, size_t dram_bytes);

/// Destroy a pool. `pool` must not be used after.
void vugva_pool_destroy(VugvaPool *pool);

/// Allocate `total_bytes` in DRAM under the given name.
/// Returns 0 on success, -1 on failure.
int vugva_alloc(VugvaPool *pool, const char *name, size_t total_bytes);

/// Copy `len` bytes from `data` into the named page's DRAM backing.
/// Returns 0 on success, -1 on failure.
int vugva_write(VugvaPool *pool, const char *name,
                const uint8_t *data, size_t len);

/// Promote the named page to VRAM on GPU 0.
/// On success the device pointer is written to `*out_ptr`.
/// Returns 0 on success, -1 on failure.
int vugva_access(VugvaPool *pool, const char *name, uint64_t *out_ptr);

/// Demote the named page from VRAM back to DRAM.
/// Returns 0 on success, -1 on failure.
int vugva_demote(VugvaPool *pool, const char *name);

/// Deallocate the named page, returning DRAM + VRAM to their pools.
/// Returns 0 on success, -1 on failure.
int vugva_dealloc(VugvaPool *pool, const char *name);

/// Fill `out` with current pool statistics.
/// Returns 0 on success, -1 on failure.
int vugva_pool_info(const VugvaPool *pool, VugvaPoolInfo *out);

#ifdef __cplusplus
}
#endif

#endif /* VUGVA_H */
