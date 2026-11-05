/// ggml-backend-vugva — VUGVA DRAM memory layer for ggml/llama.cpp
///
/// Architecture:
///   VUGVA is a BUFFER TYPE ONLY (not a compute backend).
///   Weights live in pinned DRAM (cudaMallocHost → DMA-capable).
///   CUDA is the compute engine. CPU never touches data-plane traffic.
///
/// CPU-Bypass DMA:
///   When CUDA needs a VUGVA_DRAM tensor, the scheduler copies it
///   to VRAM via cudaMemcpyAsync. Since source is pinned (cudaMallocHost),
///   this is TRUE DMA — CPU never touches the data.
///
/// Flow:
///   1. Model weights loaded into VUGVA_DRAM (pinned host memory)
///   2. Scheduler sees is_host=true, creates CUDA buffer, copies via DMA
///   3. CUDA computes on VRAM data
///   4. CPU only manages metadata, never data-plane traffic

#include "ggml-backend-vugva.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <string>

#include <sys/mman.h>

#include <cuda_runtime.h>

extern "C" {
typedef struct VugvaPool VugvaPool;
typedef struct {
    size_t dram_capacity;
    size_t dram_used;
    int    numa_bound;
    int    pinned;
} VugvaPoolInfo;

VugvaPool * vugva_pool_create(int gpu_id, size_t dram_bytes);
void        vugva_pool_destroy(VugvaPool * pool);
int         vugva_alloc(VugvaPool * pool, const char * name, size_t total_bytes);
int         vugva_dealloc(VugvaPool * pool, const char * name);
int         vugva_pool_info(const VugvaPool * pool, VugvaPoolInfo * out);
}

// ============================================================================
// VUGVA context (singleton)
// ============================================================================

struct ggml_backend_vugva_context {
    VugvaPool * pool = nullptr;
    int         gpu_id = 0;
    size_t      dram_bytes = 0;
    bool        initialized = false;

    bool init(ggml_backend_vugva_params p) {
        if (initialized) return true;
        pool = vugva_pool_create(p.gpu_id, p.dram_pool_bytes);
        if (!pool) return false;
        gpu_id = p.gpu_id;
        dram_bytes = p.dram_pool_bytes;
        initialized = true;

        VugvaPoolInfo info = {};
        vugva_pool_info(pool, &info);
        fprintf(stderr, "[VUGVA] DRAM pool: %zu GB, NUMA=%s (cudaMallocHost=always, DMA-capable)\n",
                info.dram_capacity / 1024 / 1024 / 1024,
                info.numa_bound ? "yes" : "no");
        return true;
    }

    ~ggml_backend_vugva_context() {
        if (pool) { vugva_pool_destroy(pool); pool = nullptr; }
    }
};

static ggml_backend_vugva_context & vugva_ctx() {
    static ggml_backend_vugva_context ctx;
    return ctx;
}

#ifndef VUGVA_DRAM_GB
#define VUGVA_DRAM_GB 32
#endif

// ============================================================================
// Zero-copy (unified DRAM+VRAM) mode
//
// Paper §3/§4: the DRAM tier is part of ONE virtual GPU memory space. When
// VUGVA_ZEROCOPY=1 the DRAM buffers are allocated with cudaHostAllocMapped,
// so the CUDA kernels read warm-tier weights directly from DRAM over PCIe —
// the CPU never touches data-plane traffic and there is nothing to promote:
// the GPU addresses host memory through UVA. llama-model.cpp decides the
// mode (auto/hybrid/zerocopy) and exports it through this environment
// variable before tensors are created.
//
// VUGVA_ZEROCOPY=0 (default): hybrid mode. Warm weights live in the DRAM tier
// (pinned or mmap) and their layers are computed by the CPU backend.
// ============================================================================

static bool vugva_zerocopy_enabled(void) {
    const char * env = getenv("VUGVA_ZEROCOPY");
    return env && *env == '1';
}

extern "C" ggml_backend_vugva_params ggml_backend_vugva_default_params(void) {
    ggml_backend_vugva_params p = {};
    p.gpu_id = 0;
    // pool size: cmake -DVUGVA_DRAM_GB=N, overridable at runtime via VUGVA_DRAM_GB=N
    uint64_t dram_gb = (uint64_t) VUGVA_DRAM_GB;
    if (const char * env = getenv("VUGVA_DRAM_GB")) {
        char * end = nullptr;
        unsigned long long v = strtoull(env, &end, 10);
        if (end && *end == '\0' && v > 0) {
            dram_gb = v;
        } else {
            fprintf(stderr, "[VUGVA] ignoring invalid VUGVA_DRAM_GB=%s\n", env);
        }
    }
    p.dram_pool_bytes = dram_gb * 1024 * 1024 * 1024;
    p.spill_enabled = 0;
    p.spill_path = nullptr;
    return p;
}

extern "C" int ggml_backend_vugva_init(ggml_backend_vugva_params params) {
    return vugva_ctx().init(params) ? 0 : -1;
}

// Probe whether the GPU can address mapped host memory (UVA zero-copy), which
// the warm (DRAM) tier depends on. Called before tensor placement is decided.
extern "C" int ggml_backend_vugva_zerocopy_available(void) {
    void * p = nullptr;
    cudaError_t e = cudaHostAlloc(&p, 1 << 20, cudaHostAllocMapped | cudaHostAllocPortable);
    if (e != cudaSuccess) {
        return 0;
    }
    void * d = nullptr;
    e = cudaHostGetDevicePointer(&d, p, 0);
    const bool ok = e == cudaSuccess && d == p;
    cudaFreeHost(p);
    return ok ? 1 : 0;
}

// ============================================================================
// VUGVA_DRAM buffer — pinned host memory (cudaMallocHost)
// ============================================================================

static bool vugva_is_vugva_buffer(ggml_backend_buffer_t buf) {
    if (!buf || !buf->buft) return false;
    const char * name = buf->buft->iface.get_name ? buf->buft->iface.get_name(buf->buft) : "";
    return strcmp(name, "VUGVA_DRAM") == 0;
}

// ── pinned-memory safety ────────────────────────────────────────────────────
// Pinned pages (cudaHostAlloc / cudaMallocHost) cannot be reclaimed by the
// kernel, so over-committing RAM degrades to swapping/OOM instead of a clean
// error. Refuse to pin when the request would leave too little headroom.
static size_t vugva_pinned_total = 0; // single-threaded at load time

// Allow pinning `bytes` only if (a) we keep >= 2 GiB of free RAM headroom and
// (b) the total pinned memory stays under half of physical RAM. Pinned pages
// cannot be reclaimed, so exceeding this turns the desktop into swap soup.
// VUGVA_PIN_MB overrides the cap for machines dedicated to inference.
static bool vugva_pin_ok(size_t bytes) {
    size_t total = 0, avail = 0;
    FILE * f = fopen("/proc/meminfo", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            size_t kb = 0;
            if (sscanf(line, "MemTotal: %zu kB", &kb) == 1)     total = kb * 1024;
            if (sscanf(line, "MemAvailable: %zu kB", &kb) == 1) avail = kb * 1024;
        }
        fclose(f);
    }

    size_t limit = total / 2;
    if (const char * env = getenv("VUGVA_PIN_MB")) {
        char * end = nullptr;
        unsigned long long v = strtoull(env, &end, 10);
        if (end && *end == '\0' && v > 0) {
            limit = (size_t) v * 1024ull * 1024ull;
        }
    }

    const size_t headroom = 2ull << 30;
    if (vugva_pinned_total + bytes <= limit && bytes + vugva_pinned_total + headroom <= avail) {
        vugva_pinned_total += bytes;
        return true;
    }
    fprintf(stderr, "[VUGVA] refusing to pin %zu MiB (pinned %zu/%zu MiB, RAM available %zu MiB) — using pageable DRAM instead\n",
            bytes / 1024 / 1024, vugva_pinned_total / 1024 / 1024, limit / 1024 / 1024,
            avail / 1024 / 1024);
    return false;
}

struct ggml_backend_vugva_buffer_context {
    std::string name;
    size_t      size;
    void *      dram_ptr;
    bool        owns_dram_ptr;
    bool        is_pinned = false;    // true = page-locked (cudaMallocHost / cudaHostAllocMapped / cudaHostRegister)
    bool        device_mapped = false; // true = GPU can address dram_ptr directly via UVA (zero-copy tier)
    bool        is_mapped_alloc = false; // true = allocated via cudaHostAlloc (cudaFreeHost on free)

    ggml_backend_vugva_buffer_context(const std::string & n, size_t sz, void * ptr, bool owns = true)
        : name(n), size(sz), dram_ptr(ptr), owns_dram_ptr(owns) {}
};

static void ggml_backend_vugva_buffer_free(ggml_backend_buffer_t buffer) {
    auto * ctx = (ggml_backend_vugva_buffer_context *)buffer->context;
    if (ctx) {
        if (!ctx->name.empty() && vugva_ctx().pool) {
            vugva_dealloc(vugva_ctx().pool, ctx->name.c_str());
        }
        if (ctx->dram_ptr && ctx->owns_dram_ptr) {
            if (ctx->is_mapped_alloc || ctx->is_pinned) {
                cudaFreeHost(ctx->dram_ptr);
                vugva_pinned_total = vugva_pinned_total > ctx->size ? vugva_pinned_total - ctx->size : 0;
            } else {
                munmap(ctx->dram_ptr, ctx->size);
            }
        }
        delete ctx;
    }
}

static void * ggml_backend_vugva_buffer_get_base(ggml_backend_buffer_t buffer) {
    auto * ctx = (ggml_backend_vugva_buffer_context *)buffer->context;
    return ctx ? ctx->dram_ptr : nullptr;
}

static enum ggml_status ggml_backend_vugva_buffer_init_tensor(
    ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    (void)buffer; (void)tensor;
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_vugva_buffer_memset_tensor(
    ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
    uint8_t value, size_t offset, size_t size) {
    (void)buffer;
    memset((char *)tensor->data + offset, value, size);
}

static void ggml_backend_vugva_buffer_set_tensor(
    ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
    const void * data, size_t offset, size_t size) {
    (void)buffer;
    memcpy((char *)tensor->data + offset, data, size);
}

static void ggml_backend_vugva_buffer_get_tensor(
    ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor,
    void * data, size_t offset, size_t size) {
    (void)buffer;
    memcpy(data, (const char *)tensor->data + offset, size);
}

static void ggml_backend_vugva_buffer_set_tensor_2d(
    ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
    const void * data, size_t offset, size_t size,
    size_t n_copies, size_t stride_tensor, size_t stride_data) {
    (void)buffer;
    char * base = (char *)tensor->data + offset;
    const char * src = (const char *)data;
    for (size_t i = 0; i < n_copies; i++) {
        memcpy(base + i * stride_tensor, src + i * stride_data, size);
    }
}

static void ggml_backend_vugva_buffer_get_tensor_2d(
    ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor,
    void * data, size_t offset, size_t size,
    size_t n_copies, size_t stride_tensor, size_t stride_data) {
    (void)buffer;
    const char * src = (const char *)tensor->data + offset;
    char * dst = (char *)data;
    for (size_t i = 0; i < n_copies; i++) {
        memcpy(dst + i * stride_data, src + i * stride_tensor, size);
    }
}

static bool ggml_backend_vugva_buffer_cpy_tensor(
    ggml_backend_buffer_t buffer,
    const struct ggml_tensor * src, struct ggml_tensor * dst) {
    (void)buffer;
    size_t nbytes = ggml_nbytes(src);
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, nbytes);
    } else {
        // device -> VUGVA DRAM: hand the transfer to the driver (D2H DMA)
        cudaMemcpy(dst->data, src->data, nbytes, cudaMemcpyDeviceToHost);
    }
    return true;
}

static void ggml_backend_vugva_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * ctx = (ggml_backend_vugva_buffer_context *)buffer->context;
    if (!ctx || !ctx->dram_ptr) return;
    memset(ctx->dram_ptr, value, ctx->size);
}

static const ggml_backend_buffer_i ggml_backend_vugva_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_vugva_buffer_free,
    /* .get_base        = */ ggml_backend_vugva_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_vugva_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_vugva_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_vugva_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_vugva_buffer_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_vugva_buffer_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_vugva_buffer_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_vugva_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_vugva_buffer_clear,
    /* .reset           = */ NULL,
};

// ============================================================================
// VUGVA_DRAM buffer type
// ============================================================================

struct ggml_backend_vugva_buffer_type_context {
    int gpu_id;
};

static const char * ggml_backend_vugva_buft_name(ggml_backend_buffer_type_t buft) {
    (void)buft;
    return "VUGVA_DRAM";
}

static ggml_backend_buffer_t ggml_backend_vugva_buft_alloc_buffer(
    ggml_backend_buffer_type_t buft, size_t size) {

    ggml_backend_vugva_context & vctx = vugva_ctx();
    if (!vctx.initialized) return nullptr;

    if (size == 0) {
        auto * ctx = new(std::nothrow) ggml_backend_vugva_buffer_context("", 0, nullptr, false);
        if (!ctx) return nullptr;
        ggml_backend_buffer_t buffer = ggml_backend_buffer_init(
            buft, ggml_backend_vugva_buffer_interface, ctx, 0);
        if (!buffer) { delete ctx; return nullptr; }
        return buffer;
    }

    static std::atomic<uint64_t> counter{0};
    uint64_t id = counter.fetch_add(1);
    std::string name = "ggml_buf_" + std::to_string(id);

    // Register with the VUGVA DRAM pool (bookkeeping + tier accounting).
    // Going over the pool budget is NOT fatal: the excess is served by the OS
    // as the cold tier (page cache / swap on NVMe) — paper §4 T2 semantics.
    std::string alloc_name;
    if (vugva_alloc(vctx.pool, name.c_str(), size) == 0) {
        alloc_name = name;
    } else {
        fprintf(stderr, "[VUGVA] %zu MiB exceeds the DRAM pool budget — tracked as cold tier (SSD/page-cache)\n",
                size / 1024 / 1024);
    }

    // Page-lock the range so scheduler copies are true CPU-bypass DMA
    // (paper §5.1). In zero-copy mode the allocation is additionally made
    // DEVICE-MAPPED (cudaHostAllocMapped + UVA): the GPU reads the warm tier
    // directly over PCIe, so the CPU never touches data-plane traffic at all.
    // Fall back to ordinary anonymous DRAM if pinning fails.
    void * dram_ptr = nullptr;
    bool is_pinned = false;
    bool device_mapped = false;
    bool is_mapped_alloc = false;

    cudaError_t err;
    if (vugva_zerocopy_enabled() && vugva_pin_ok(size)) {
        // mapped + portable host allocation: GPU-addressable warm tier
        err = cudaHostAlloc(&dram_ptr, size, cudaHostAllocMapped | cudaHostAllocPortable);
        if (err == cudaSuccess) {
            void * dev_ptr = nullptr;
            cudaError_t e2 = cudaHostGetDevicePointer(&dev_ptr, dram_ptr, 0);
            // UVA: the device alias is the same address as the host pointer,
            // which is what lets ggml tensors carry a single `data` pointer.
            if (e2 != cudaSuccess || dev_ptr != dram_ptr) {
                fprintf(stderr, "[VUGVA] cudaHostGetDevicePointer failed (%s) — falling back to hybrid DRAM\n",
                        cudaGetErrorString(e2));
                cudaFreeHost(dram_ptr);
                dram_ptr = nullptr;
            } else {
                is_pinned = true;
                device_mapped = true;
                is_mapped_alloc = true;
            }
        } else {
            fprintf(stderr, "[VUGVA] cudaHostAllocMapped(%zu MiB) failed: %s — falling back to hybrid DRAM\n",
                    size / 1024 / 1024, cudaGetErrorString(err));
        }
    }

    if (!dram_ptr && !device_mapped) {
        // Zero-copy was requested but the mapped allocation failed. Disable the
        // mode globally so the CUDA backend stops claiming VUGVA buffers (they
        // would not be GPU-addressable) and the scheduler falls back to the CPU
        // for the DRAM tier instead of dereferencing a bad pointer.
        if (vugva_zerocopy_enabled()) {
            setenv("VUGVA_ZEROCOPY", "0", 1);
            fprintf(stderr, "[VUGVA] zero-copy allocation failed — disabling zero-copy for this run (hybrid DRAM fallback)\n");
        }
        err = vugva_pin_ok(size) ? cudaMallocHost(&dram_ptr, size) : cudaErrorMemoryAllocation;
        if (err == cudaSuccess) {
            is_pinned = true;
            is_mapped_alloc = true;
        } else {
            fprintf(stderr, "[VUGVA] cudaMallocHost(%zu MiB) failed: %s — falling back to pageable DRAM\n",
                    size / 1024 / 1024, cudaGetErrorString(err));
            dram_ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (dram_ptr == MAP_FAILED) {
                dram_ptr = nullptr;
            }
        }
    }

    if (!dram_ptr) {
        if (!alloc_name.empty()) vugva_dealloc(vctx.pool, alloc_name.c_str());
        return nullptr;
    }

    auto free_storage = [&]() {
        if (!dram_ptr) return;
        if (is_mapped_alloc || is_pinned) cudaFreeHost(dram_ptr);
        else munmap(dram_ptr, size);
        dram_ptr = nullptr;
    };

    auto * ctx = new(std::nothrow) ggml_backend_vugva_buffer_context(alloc_name, size, dram_ptr);
    if (!ctx) {
        free_storage();
        if (!alloc_name.empty()) vugva_dealloc(vctx.pool, alloc_name.c_str());
        return nullptr;
    }
    ctx->is_pinned = is_pinned;
    ctx->device_mapped = device_mapped;
    ctx->is_mapped_alloc = is_mapped_alloc;

    ggml_backend_buffer_t buffer = ggml_backend_buffer_init(
        buft, ggml_backend_vugva_buffer_interface, ctx, size);
    if (!buffer) {
        free_storage();
        delete ctx;
        if (!alloc_name.empty()) vugva_dealloc(vctx.pool, alloc_name.c_str());
        return nullptr;
    }

    return buffer;
}

static size_t ggml_backend_vugva_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    (void)buft;
    return 128;
}

static size_t ggml_backend_vugva_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    (void)buft;
    // unbounded: ggml uses this only to split large allocations into chunks.
    // Real limits are enforced by the DRAM pool + allocator fallbacks above.
    return SIZE_MAX;
}

static size_t ggml_backend_vugva_buft_get_alloc_size(
    ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    (void)buft;
    return ggml_nbytes(tensor);
}

// is_host=true: scheduler copies these to CUDA via cudaMemcpyAsync.
// Since source is cudaMallocHost (pinned), this is TRUE DMA. CPU bypass.
static bool ggml_backend_vugva_buft_is_host(ggml_backend_buffer_type_t buft) {
    (void)buft;
    return true;
}

static const ggml_backend_buffer_type_i ggml_backend_vugva_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_vugva_buft_name,
    /* .alloc_buffer     = */ ggml_backend_vugva_buft_alloc_buffer,
    /* .alloc_buffer_n   = */ nullptr, // default: alloc_buffer + linear allocator
    /* .get_alignment    = */ ggml_backend_vugva_buft_get_alignment,
    /* .get_max_size     = */ ggml_backend_vugva_buft_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_vugva_buft_get_alloc_size,
    /* .get_alloc_size_n = */ nullptr, // default: per-tensor get_alloc_size
    /* .is_host          = */ ggml_backend_vugva_buft_is_host,
};

// ============================================================================
// VUGVA backend — MEMORY LAYER ONLY (no compute, no graph_compute)
// ============================================================================

struct ggml_backend_vugva_context_backend {
    int gpu_id;
    ggml_backend_buffer_type_t buft;
    ggml_backend_dev_t dev;
};

static const char * ggml_backend_vugva_get_name(ggml_backend_t backend) {
    (void)backend;
    return "VUGVA";
}

static void ggml_backend_vugva_free(ggml_backend_t backend) {
    auto * ctx = (ggml_backend_vugva_context_backend *)backend->context;
    delete ctx;
    delete backend;
}

static ggml_backend_buffer_type_t ggml_backend_vugva_get_default_buffer_type(ggml_backend_t backend) {
    auto * ctx = (ggml_backend_vugva_context_backend *)backend->context;
    return ctx->buft;
}

static void ggml_backend_vugva_set_tensor_async(ggml_backend_t backend,
    struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    (void)backend;
    if (tensor->buffer) {
        memcpy((char *)tensor->data + offset, data, size);
    }
}

static void ggml_backend_vugva_get_tensor_async(ggml_backend_t backend,
    const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    (void)backend;
    if (tensor->buffer) {
        memcpy(data, (const char *)tensor->data + offset, size);
    }
}

static void ggml_backend_vugva_set_tensor_2d_async(ggml_backend_t backend,
    struct ggml_tensor * tensor, const void * data, size_t offset, size_t size,
    size_t n_copies, size_t stride_tensor, size_t stride_data) {
    (void)backend;
    if (tensor->buffer) {
        char * base = (char *)tensor->data + offset;
        const char * src = (const char *)data;
        for (size_t i = 0; i < n_copies; i++) {
            memcpy(base + i * stride_tensor, src + i * stride_data, size);
        }
    }
}

static void ggml_backend_vugva_get_tensor_2d_async(ggml_backend_t backend,
    const struct ggml_tensor * tensor, void * data, size_t offset, size_t size,
    size_t n_copies, size_t stride_tensor, size_t stride_data) {
    (void)backend;
    if (tensor->buffer) {
        const char * src = (const char *)tensor->data + offset;
        char * dst = (char *)data;
        for (size_t i = 0; i < n_copies; i++) {
            memcpy(dst + i * stride_data, src + i * stride_tensor, size);
        }
    }
}

static bool ggml_backend_vugva_cpy_tensor_async(ggml_backend_t backend_src,
    ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    (void)backend_src; (void)backend_dst;
    if (!src->buffer || !dst->buffer || !src->data || !dst->data) return false;
    const size_t nbytes = ggml_nbytes(src);
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, nbytes);
    } else {
        // device -> VUGVA DRAM: hand the transfer to the driver (D2H DMA).
        // VUGVA has no stream/event of its own (memory-only backend), so this
        // runs on the default stream and is synchronized before returning.
        cudaMemcpy(dst->data, src->data, nbytes, cudaMemcpyDeviceToHost);
    }
    return true;
}

static void ggml_backend_vugva_synchronize(ggml_backend_t backend) {
    (void)backend;
}

// No graph_compute — CUDA handles all compute via scheduler
// No supports_op — VUGVA is buffer-only, scheduler routes ops to CUDA/CPU

static const ggml_backend_i ggml_backend_vugva_interface = {
    /* .get_name          = */ ggml_backend_vugva_get_name,
    /* .free              = */ ggml_backend_vugva_free,
    /* .set_tensor_async  = */ ggml_backend_vugva_set_tensor_async,
    /* .get_tensor_async  = */ ggml_backend_vugva_get_tensor_async,
    /* .set_tensor_2d_async = */ ggml_backend_vugva_set_tensor_2d_async,
    /* .get_tensor_2d_async = */ ggml_backend_vugva_get_tensor_2d_async,
    /* .cpy_tensor_async  = */ ggml_backend_vugva_cpy_tensor_async,
    /* .synchronize       = */ ggml_backend_vugva_synchronize,
    /* .graph_plan_create = */ NULL,
    /* .graph_plan_free   = */ NULL,
    /* .graph_plan_update = */ NULL,
    /* .graph_plan_compute = */ NULL,
    /* .graph_compute     = */ NULL,  // MEMORY LAYER — no compute
    /* .event_record      = */ NULL,
    /* .event_wait        = */ NULL,
    /* .graph_optimize    = */ NULL,
};

// ============================================================================
// GUID
// ============================================================================

static ggml_guid_t ggml_backend_vugva_guid() {
    static ggml_guid guid = {
        0x76, 0x75, 0x67, 0x76, 0x61, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01
    };
    return &guid;
}

// ============================================================================
// Device interface
// ============================================================================

struct ggml_backend_vugva_device_context {
    int gpu_id;
    std::string name;
    std::string description;
};

static const char * ggml_backend_vugva_dev_name(ggml_backend_dev_t dev) {
    auto * ctx = (ggml_backend_vugva_device_context *)dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_vugva_dev_description(ggml_backend_dev_t dev) {
    auto * ctx = (ggml_backend_vugva_device_context *)dev->context;
    return ctx->description.c_str();
}

static void ggml_backend_vugva_dev_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    (void)dev;
    ggml_backend_vugva_context & vctx = vugva_ctx();
    if (!vctx.initialized) {
        if (free) *free = 0;
        if (total) *total = 0;
        return;
    }
    VugvaPoolInfo info = {};
    vugva_pool_info(vctx.pool, &info);
    if (free) *free = info.dram_capacity - info.dram_used;
    if (total) *total = info.dram_capacity;
}

static enum ggml_backend_dev_type ggml_backend_vugva_dev_get_type(ggml_backend_dev_t dev) {
    (void)dev;
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static void ggml_backend_vugva_dev_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    auto * ctx = (ggml_backend_vugva_device_context *)dev->context;
    props->name        = ggml_backend_vugva_dev_name(dev);
    props->description = ggml_backend_vugva_dev_description(dev);
    props->type        = GGML_BACKEND_DEVICE_TYPE_ACCEL;
    props->device_id   = nullptr;
    ggml_backend_vugva_dev_memory(dev, &props->memory_free, &props->memory_total);
    const bool zerocopy = vugva_zerocopy_enabled();
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ true,
        // In zero-copy mode the DRAM tier must be REAL pinned host memory
        // (cudaHostAllocMapped) so the GPU can address it over PCIe — we
        // cannot wrap the read-only GGUF mmap (cudaHostRegister refuses
        // file-backed RO mappings), so the loader copies the weights into
        // our buffers instead of importing the mapping.
        /* .buffer_from_host_ptr  = */ !zerocopy,
        /* .events                = */ false,
        /* .mmap_support          = */ false,
    };
}

// True when `buft` is the VUGVA_DRAM tier. Used by the CUDA backend to accept
// warm-tier tensors in zero-copy mode (see ggml-cuda.cu) and by tests.
extern "C" bool ggml_backend_vugva_buft_is_vugva(ggml_backend_buffer_type_t buft) {
    if (!buft || !buft->iface.get_name) return false;
    return strcmp(buft->iface.get_name(buft), "VUGVA_DRAM") == 0;
}

// ============================================================================
// Backend init
// ============================================================================

static ggml_backend_buffer_type_t ggml_backend_vugva_dev_get_buffer_type(ggml_backend_dev_t dev);

static ggml_backend_t ggml_backend_vugva_dev_init_backend(ggml_backend_dev_t dev, const char * params) {
    (void)params;
    auto * dev_ctx = (ggml_backend_vugva_device_context *)dev->context;

    if (!vugva_ctx().initialized) {
        auto p = ggml_backend_vugva_default_params();
        vugva_ctx().init(p);
    }

    auto * backend_ctx = new(std::nothrow) ggml_backend_vugva_context_backend();
    if (!backend_ctx) return nullptr;

    backend_ctx->gpu_id = dev_ctx->gpu_id;

    // reuse the singleton buffer type — pointer-stable across get_buffer_type calls
    auto * buft = ggml_backend_vugva_dev_get_buffer_type(dev);
    if (!buft) { delete backend_ctx; return nullptr; }

    backend_ctx->buft = buft;
    backend_ctx->dev  = dev;

    auto * backend = new(std::nothrow) ggml_backend();
    if (!backend) { delete backend_ctx; return nullptr; }

    backend->iface    = ggml_backend_vugva_interface;
    backend->guid     = ggml_backend_vugva_guid();
    backend->device   = dev;
    backend->context  = backend_ctx;

    return backend;
}

// ============================================================================
// Device buffer types
// ============================================================================

static ggml_backend_buffer_type_t ggml_backend_vugva_dev_get_buffer_type(ggml_backend_dev_t dev) {
    // A single stable instance per device: llama.cpp compares buffer type
    // POINTERS (is_default_buft, ctx_map keys). Returning a fresh object per
    // call broke the mmap zero-copy fast path and leaked an instance each call.
    static ggml_backend_buffer_type_t cached = nullptr;
    if (cached && cached->device == dev) {
        return cached;
    }
    auto * ctx = (ggml_backend_vugva_device_context *)dev->context;
    auto * buft_ctx = new ggml_backend_vugva_buffer_type_context();
    buft_ctx->gpu_id = ctx->gpu_id;
    auto * buft = new ggml_backend_buffer_type();
    buft->iface   = ggml_backend_vugva_buffer_type_interface;
    buft->device  = dev;
    buft->context = buft_ctx;
    cached = buft;
    return buft;
}

// VUGVA_DRAM *is* host memory — there is no separate staging buffer type.
// Returning NULL keeps llama.cpp's "avoid using a host buffer when using
// mmap" rule from rewriting VUGVA selections back to the plain CPU buffer.
static ggml_backend_buffer_type_t ggml_backend_vugva_dev_get_host_buffer_type(ggml_backend_dev_t dev) {
    (void)dev;
    return nullptr;
}

static ggml_backend_buffer_t ggml_backend_vugva_dev_buffer_from_host_ptr(
    ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    (void)max_tensor_size;
    ggml_backend_vugva_context & vctx = vugva_ctx();
    if (!vctx.initialized) return nullptr;

    static std::atomic<uint64_t> counter{0};
    uint64_t id = counter.fetch_add(1);
    std::string name = "import_buf_" + std::to_string(id);

    // Register with the DRAM pool when it has room; imports larger than the
    // pool budget (e.g. 70B Q4 on a 32 GB pool) still load — the file mapping
    // itself is backed by page cache / NVMe, i.e. the paper's cold tier.
    std::string alloc_name;
    if (vugva_alloc(vctx.pool, name.c_str(), size) == 0) {
        alloc_name = name;
    } else {
        fprintf(stderr, "[VUGVA] import of %zu MiB exceeds the DRAM pool budget — tracked as cold tier (SSD/page-cache)\n",
                size / 1024 / 1024);
    }

    auto * ctx = new(std::nothrow) ggml_backend_vugva_buffer_context(alloc_name, size, ptr, false);
    if (!ctx) { if (!alloc_name.empty()) vugva_dealloc(vctx.pool, alloc_name.c_str()); return nullptr; }

    auto * buft = ggml_backend_vugva_dev_get_buffer_type(dev);
    ggml_backend_buffer_t buffer = ggml_backend_buffer_init(
        buft, ggml_backend_vugva_buffer_interface, ctx, size);
    if (!buffer) { delete ctx; if (!alloc_name.empty()) vugva_dealloc(vctx.pool, alloc_name.c_str()); return nullptr; }
    return buffer;
}

// VUGVA never computes (graph_compute == NULL) and is deliberately NOT a
// scheduler backend (llama-context.cpp skips ACCEL "VUGVA"), so this callback
// is only consulted when llama.cpp SELECTS BUFFER TYPES FOR WEIGHTS
// (weight_buft_supported / select_buft / llama_model::select_buft for KV).
// Returning false there made every tensor reject VUGVA_DRAM — the root cause
// of weights falling back to CUDA (single huge cudaMalloc -> OOM) or CPU.
static bool ggml_backend_vugva_dev_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    (void)dev; (void)op;
    return true;
}

static bool ggml_backend_vugva_dev_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (!buft) return false;
    const char * name = buft->iface.get_name ? buft->iface.get_name(buft) : "";
    return strcmp(name, "VUGVA_DRAM") == 0;
}

// offload_op=false: VUGVA doesn't compute, CUDA does
static bool ggml_backend_vugva_dev_offload_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    (void)dev; (void)op;
    return false;
}

static const ggml_backend_device_i ggml_backend_vugva_device_interface = {
    /* .get_name            = */ ggml_backend_vugva_dev_name,
    /* .get_description     = */ ggml_backend_vugva_dev_description,
    /* .get_memory          = */ ggml_backend_vugva_dev_memory,
    /* .get_type            = */ ggml_backend_vugva_dev_get_type,
    /* .get_props           = */ ggml_backend_vugva_dev_get_props,
    /* .init_backend        = */ ggml_backend_vugva_dev_init_backend,
    /* .get_buffer_type     = */ ggml_backend_vugva_dev_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_vugva_dev_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ ggml_backend_vugva_dev_buffer_from_host_ptr,
    /* .supports_op         = */ ggml_backend_vugva_dev_supports_op,
    /* .supports_buft       = */ ggml_backend_vugva_dev_supports_buft,
    /* .offload_op          = */ ggml_backend_vugva_dev_offload_op,
    /* .event_new           = */ NULL,
    /* .event_free          = */ NULL,
    /* .event_synchronize   = */ NULL,
};

// ============================================================================
// Registration
// ============================================================================

struct ggml_backend_vugva_reg_context {
    ggml_backend_dev_t dev;
    ggml_backend_t backend;
};

static const char * ggml_backend_vugva_reg_get_name(ggml_backend_reg_t reg) {
    (void)reg;
    return "VUGVA";
}

static size_t ggml_backend_vugva_reg_get_device_count(ggml_backend_reg_t reg) {
    (void)reg;
    return 1;
}

static ggml_backend_dev_t ggml_backend_vugva_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    auto * ctx = (ggml_backend_vugva_reg_context *)reg->context;
    if (index == 0) return ctx->dev;
    return nullptr;
}

static void * ggml_backend_vugva_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    (void)reg;
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_vugva_reg_interface = {
    /* .get_name          = */ ggml_backend_vugva_reg_get_name,
    /* .get_device_count  = */ ggml_backend_vugva_reg_get_device_count,
    /* .get_device        = */ ggml_backend_vugva_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_vugva_reg_get_proc_address,
};

extern "C" ggml_backend_reg_t ggml_backend_vugva_reg(void) {
    static ggml_backend_reg reg;
    static std::atomic<bool> initialized{false};

    if (initialized.load(std::memory_order_acquire)) return &reg;

    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    if (initialized.load(std::memory_order_relaxed)) return &reg;

    auto params = ggml_backend_vugva_default_params();
    if (!vugva_ctx().init(params)) {
        fprintf(stderr, "[VUGVA] Failed to initialize\n");
        return nullptr;
    }

    static ggml_backend_vugva_device_context dev_ctx;
    dev_ctx.gpu_id = vugva_ctx().gpu_id;
    dev_ctx.name = "VUGVA";
    dev_ctx.description = "VUGVA DRAM (pinned, DMA-capable, CPU-bypass)";

    static struct ggml_backend_device dev;
    dev.iface   = ggml_backend_vugva_device_interface;
    dev.reg     = &reg;
    dev.context = &dev_ctx;

    static ggml_backend_vugva_reg_context reg_ctx;
    reg_ctx.dev     = &dev;
    reg_ctx.backend = nullptr;

    reg = {};
    reg.api_version = GGML_BACKEND_API_VERSION;
    reg.iface       = ggml_backend_vugva_reg_interface;
    reg.context     = &reg_ctx;

    initialized.store(true, std::memory_order_release);
    return &reg;
}
