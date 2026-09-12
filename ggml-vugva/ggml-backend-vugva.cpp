/// ggml-backend-vugva — VUGVA tiered memory backend for ggml/llama.cpp
///
/// Implements the ggml_backend interface on top of VUGVA's three-tier
/// hierarchy. Tensor data lives in DRAM and is promoted to VRAM on demand.

#include "ggml-backend-vugva.h"

#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <new>
#include <string>
#include <vector>

// Forward-declare VUGVA C API (from bridge.rs)
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
int         vugva_write(VugvaPool * pool, const char * name,
                         const uint8_t * data, size_t len);
int         vugva_access(VugvaPool * pool, const char * name, uint64_t * out_ptr);
int         vugva_demote(VugvaPool * pool, const char * name);
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
        if (!pool) {
            fprintf(stderr, "[VUGVA] vugva_pool_create failed (gpu=%d, dram=%zu)\n",
                    p.gpu_id, p.dram_pool_bytes);
            return false;
        }
        gpu_id = p.gpu_id;
        dram_bytes = p.dram_pool_bytes;
        initialized = true;

        VugvaPoolInfo info = {};
        vugva_pool_info(pool, &info);
        fprintf(stderr, "[VUGVA] Pool created: DRAM=%zu GB, NUMA=%s, pinned=%s\n",
                info.dram_capacity / 1024 / 1024 / 1024,
                info.numa_bound ? "yes" : "no",
                info.pinned ? "yes" : "no");
        return true;
    }

    ~ggml_backend_vugva_context() {
        if (pool) {
            vugva_pool_destroy(pool);
            pool = nullptr;
        }
    }
};

static ggml_backend_vugva_context & vugva_ctx() {
    static ggml_backend_vugva_context ctx;
    return ctx;
}

// ============================================================================
// ggml_backend_vugva_params
// ============================================================================

extern "C" ggml_backend_vugva_params ggml_backend_vugva_default_params(void) {
    ggml_backend_vugva_params p = {};
    p.gpu_id = 0;
    p.dram_pool_bytes = 32ULL * 1024 * 1024 * 1024; // 32 GB
    p.spill_enabled = 0;
    p.spill_path = nullptr;
    return p;
}

extern "C" int ggml_backend_vugva_init(ggml_backend_vugva_params params) {
    return vugva_ctx().init(params) ? 0 : -1;
}

// ============================================================================
// VUGVA buffer — wraps a VUGVA allocation as a ggml_backend_buffer
// ============================================================================

struct ggml_backend_vugva_buffer {
    ggml_backend_buffer_t base;
    std::string name;
    size_t size;
    uint64_t device_ptr; // VRAM pointer after promotion
};

static void ggml_backend_vugva_buffer_free(ggml_backend_buffer_t buffer) {
    auto * buf = (ggml_backend_vugva_buffer *)buffer->context;
    if (buf && !buf->name.empty()) {
        vugva_dealloc(vugva_ctx().pool, buf->name.c_str());
    }
    delete buf;
    delete[] (char *)buffer->context;
}

static void ggml_backend_vugva_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    (void)buffer; (void)value;
    // VUGVA buffers are zeroed on allocation (mmap).
}

static bool ggml_backend_vugva_buffer_init_tensor(
    ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    (void)buffer; (void)tensor;
    return true;
}

static ggml_status ggml_backend_vugva_buffer_memset(
    ggml_backend_buffer_t buffer, size_t offset, size_t size, uint8_t value) {
    (void)buffer; (void)offset; (void)size; (void)value;
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_vugva_buffer_get_cpu(
    ggml_backend_buffer_t buffer, void ** data_ptr) {
    auto * buf = (ggml_backend_vugva_buffer *)buffer->context;
    if (buf) {
        *data_ptr = (void *)buf->device_ptr;
    } else {
        *data_ptr = nullptr;
    }
    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_buffer_i ggml_backend_vugva_buffer_interface = {
    /* .free           = */ ggml_backend_vugva_buffer_free,
    /* .clear          = */ ggml_backend_vugva_buffer_clear,
    /* .init_tensor    = */ ggml_backend_vugva_buffer_init_tensor,
    /* .memset         = */ ggml_backend_vugva_buffer_memset,
    /* .get_base       = */ ggml_backend_vugva_buffer_get_cpu,
    /* .get_host_addr  = */ nullptr,
};

// ============================================================================
// VUGVA buffer type — how ggml allocates buffers through VUGVA
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

    auto * ctx = (ggml_backend_vugva_buffer_type_context *)buft->context;
    auto & vctx = vugva_ctx();
    if (!vctx.initialized) {
        fprintf(stderr, "[VUGVA] buffer_type: pool not initialized\n");
        return nullptr;
    }

    // Generate unique name for this allocation
    static std::atomic<uint64_t> counter{0};
    uint64_t id = counter.fetch_add(1);
    std::string name = "ggml_buf_" + std::to_string(id);

    // Allocate in VUGVA DRAM pool
    int rc = vugva_alloc(vctx.pool, name.c_str(), size);
    if (rc != 0) {
        fprintf(stderr, "[VUGVA] vugva_alloc failed for %s (%zu bytes)\n",
                name.c_str(), size);
        return nullptr;
    }

    // Wrap as ggml_backend_buffer
    auto * buf = new(std::nothrow) ggml_backend_vugva_buffer();
    if (!buf) {
        vugva_dealloc(vctx.pool, name.c_str());
        return nullptr;
    }
    buf->name = name;
    buf->size = size;
    buf->device_ptr = 0;

    ggml_backend_buffer_t buffer = new(std::nothrow) ggml_backend_buffer();
    if (!buffer) {
        delete buf;
        vugva_dealloc(vctx.pool, name.c_str());
        return nullptr;
    }

    buffer->interface = ggml_backend_vugva_buffer_interface;
    buffer->buft = buft;
    buffer->size = size;
    buffer->context = buf;

    return buffer;
}

static size_t ggml_backend_vugva_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    (void)buft;
    return 64; // VUGVA DMA alignment
}

static size_t ggml_backend_vugva_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    (void)buft;
    auto & vctx = vugva_ctx();
    return vctx.initialized ? vctx.dram_bytes : 0;
}

static size_t ggml_backend_vugva_buft_get_alloc_size(
    ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    (void)buft;
    // Round up to 64-byte alignment
    size_t size = ggml_nbytes(tensor);
    return (size + 63) & ~63;
}

static bool ggml_backend_vugva_buft_is_host(ggml_backend_buffer_type_t buft) {
    (void)buft;
    return false; // VUGVA buffers are device-backed
}

static ggml_backend_dev_t ggml_backend_vugva_buft_get_device(ggml_backend_buffer_type_t buft) {
    (void)buft;
    return nullptr; // handled by registration
}

static const ggml_backend_buffer_type_i ggml_backend_vugva_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_vugva_buft_name,
    /* .alloc_buffer     = */ ggml_backend_vugva_buft_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_vugva_buft_get_alignment,
    /* .get_max_size     = */ ggml_backend_vugva_buft_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_vugva_buft_get_alloc_size,
    /* .is_host          = */ ggml_backend_vugva_buft_is_host,
};

// ============================================================================
// VUGVA device — reports VUGVA as a ggml device
// ============================================================================

struct ggml_backend_vugva_device_context {
    int gpu_id;
    ggml_backend_buffer_type_t buft;
};

static const char * ggml_backend_vugva_dev_name(ggml_backend_dev_t dev) {
    (void)dev;
    return "VUGVA";
}

static const char * ggml_backend_vugva_dev_description(ggml_backend_dev_t dev) {
    (void)dev;
    static thread_local char desc[128];
    auto & vctx = vugva_ctx();
    snprintf(desc, sizeof(desc), "VUGVA tiered memory: %zu GB DRAM + GPU %d",
             vctx.dram_bytes / 1024 / 1024 / 1024, vctx.gpu_id);
    return desc;
}

static size_t ggml_backend_vugva_dev_memory(ggml_backend_dev_t dev, size_t * free) {
    (void)dev;
    auto & vctx = vugva_ctx();
    if (!vctx.initialized) { if (free) *free = 0; return 0; }
    VugvaPoolInfo info = {};
    vugva_pool_info(vctx.pool, &info);
    if (free) *free = info.dram_capacity - info.dram_used;
    return info.dram_capacity;
}

static ggml_backend_buffer_type_t ggml_backend_vugva_dev_buffer_type(ggml_backend_dev_t dev) {
    auto * ctx = (ggml_backend_vugva_device_context *)dev->context;
    return ctx->buft;
}

static ggml_backend_buffer_type_t ggml_backend_vugva_dev_host_buffer_type(ggml_backend_dev_t dev) {
    (void)dev;
    return ggml_backend_cpu_buffer_type();
}

static const ggml_backend_dev_i ggml_backend_vugva_device_interface = {
    /* .get_name            = */ ggml_backend_vugva_dev_name,
    /* .get_description     = */ ggml_backend_vugva_dev_description,
    /* .get_memory          = */ ggml_backend_vugva_dev_memory,
    /* .get_proc_address    = */ nullptr,
    /* .buffer_type         = */ ggml_backend_vugva_dev_buffer_type,
    /* .host_buffer_type    = */ ggml_backend_vugva_dev_host_buffer_type,
    /* .supports_op         = */ nullptr,
    /* .supports_buft       = */ nullptr,
    /* .offload_op          = */ nullptr,
};

// ============================================================================
// VUGVA backend — the ggml_backend wrapping VUGVA memory
// ============================================================================

struct ggml_backend_vugva_context_backend {
    int gpu_id;
    ggml_backend_t cuda_backend; // delegate compute to CUDA
    ggml_backend_buffer_type_t buft;
    ggml_backend_dev_t dev;
};

static const char * ggml_backend_vugva_name(ggml_backend_t backend) {
    (void)backend;
    return "VUGVA";
}

static void ggml_backend_vugva_free(ggml_backend_t backend) {
    auto * ctx = (ggml_backend_vugva_context_backend *)backend->context;
    if (ctx) {
        if (ctx->cuda_backend) ggml_backend_free(ctx->cuda_backend);
        delete ctx;
    }
    delete backend;
}

static ggml_backend_buffer_type_t ggml_backend_vugva_get_default_buffer_type(
    ggml_backend_t backend) {
    auto * ctx = (ggml_backend_vugva_context_backend *)backend->context;
    return ctx->buft;
}

static ggml_status ggml_backend_vugva_graph_compute(
    ggml_backend_t backend, struct ggml_cgraph * graph) {
    // Delegate compute to CUDA backend
    auto * ctx = (ggml_backend_vugva_context_backend *)backend->context;
    if (ctx->cuda_backend) {
        return ggml_backend_graph_compute(ctx->cuda_backend, graph);
    }
    return GGML_STATUS_FAILED;
}

static ggml_status ggml_backend_vugva_synchronize(ggml_backend_t backend) {
    auto * ctx = (ggml_backend_vugva_context_backend *)backend->context;
    if (ctx->cuda_backend) {
        return ggml_backend_synchronize(ctx->cuda_backend);
    }
    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_i ggml_backend_vugva_interface = {
    /* .get_name          = */ ggml_backend_vugva_name,
    /* .free              = */ ggml_backend_vugva_free,
    /* .get_default_buffer_type = */ ggml_backend_vugva_get_default_buffer_type,
    /* .set_tensor_async  = */ nullptr,
    /* .get_tensor_async  = */ nullptr,
    /* .cpy_tensor_from   = */ nullptr,
    /* .cpy_tensor_to     = */ nullptr,
    /* .graph_compute     = */ ggml_backend_vugva_graph_compute,
    /* .synchronize       = */ ggml_backend_vugva_synchronize,
};

// ============================================================================
// VUGVA backend registration — registers with ggml's backend system
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
    return 1; // single VUGVA device
}

static ggml_backend_dev_t ggml_backend_vugva_reg_get_device(
    ggml_backend_reg_t reg, size_t index) {
    auto * ctx = (ggml_backend_vugva_reg_context *)reg->context;
    if (index == 0) return ctx->dev;
    return nullptr;
}

static void * ggml_backend_vugva_reg_get_proc_address(
    ggml_backend_reg_t reg, const char * name) {
    (void)reg;
    if (strcmp(name, "ggml_backend_register_host_buffer") == 0) {
        return nullptr; // not supported
    }
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_vugva_reg_interface = {
    /* .get_name          = */ ggml_backend_vugva_reg_get_name,
    /* .get_device_count  = */ ggml_backend_vugva_reg_get_device_count,
    /* .get_device        = */ ggml_backend_vugva_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_vugva_reg_get_proc_address,
};

// ============================================================================
// Public API
// ============================================================================

extern "C" ggml_backend_reg_t ggml_backend_vugva_reg(void) {
    static ggml_backend_reg reg;
    static bool initialized = false;

    if (initialized) return &reg;
    initialized = true;

    // Initialize VUGVA pool
    auto params = ggml_backend_vugva_default_params();
    if (!vugva_ctx().init(params)) {
        fprintf(stderr, "[VUGVA] Failed to initialize — falling back\n");
        return nullptr;
    }

    // Create buffer type
    static ggml_backend_vugva_buffer_type_context buft_ctx = {0};
    buft_ctx.gpu_id = vugva_ctx().gpu_id;

    static ggml_backend_buffer_type buft;
    buft.iface   = ggml_backend_vugva_buffer_type_interface;
    buft.context = &buft_ctx;
    buft.priority = 1; // higher priority than CPU

    // Create device
    static ggml_backend_vugva_device_context dev_ctx;
    dev_ctx.gpu_id = vugva_ctx().gpu_id;
    dev_ctx.buft   = &buft;

    static ggml_backend_dev dev;
    dev.iface    = ggml_backend_vugva_device_interface;
    dev.reg      = &reg;
    dev.context  = &dev_ctx;

    buft.device = &dev;

    // Create backend
    static ggml_backend_vugva_context_backend backend_ctx;
    backend_ctx.gpu_id    = vugva_ctx().gpu_id;
    backend_ctx.cuda_backend = nullptr; // compute delegated via graph_compute
    backend_ctx.buft      = &buft;
    backend_ctx.dev       = &dev;

    static ggml_backend backend;
    backend.iface    = ggml_backend_vugva_interface;
    backend.reg      = &reg;
    backend.priority = 2; // higher than CUDA (1)
    backend.device    = &dev;
    backend.context  = &backend_ctx;

    // Create registration context
    static ggml_backend_vugva_reg_context reg_ctx;
    reg_ctx.dev     = &dev;
    reg_ctx.backend = &backend;

    reg = {};
    reg.iface    = ggml_backend_vugva_reg_interface;
    reg.context  = &reg_ctx;

    return &reg;
}

extern "C" ggml_backend_buffer_type_t ggml_backend_vugva_buffer_type(int gpu_id) {
    static ggml_backend_vugva_buffer_type_context ctx;
    ctx.gpu_id = gpu_id;

    static ggml_backend_buffer_type buft;
    buft.iface   = ggml_backend_vugva_buffer_type_interface;
    buft.context = &ctx;

    return &buft;
}
