/// ggml-backend-vugva — VUGVA DRAM-as-cache-for-VRAM memory layer (CPU-bypass DMA)
///
/// Architecture (per VUGVA paper §3.3 + DeepSpeed ZeRO-Inference):
///   All weights live in pinned DRAM (cudaMallocHost → DMA-capable).
///   VUGVA graph_compute promotes active layers DRAM→VRAM via true DMA
///   (cudaMemcpyAsync H2D from pinned source = zero CPU involvement).
///   CUDA computes each layer on VRAM data.
///   Promoted weights STAY in VRAM across tokens (LRU cache).
///   CPU only touches 64-byte DMA descriptors per promotion.
///
/// Cache key: (buffer_base + tensor_data_offset) — stable across graph rebuilds.
///
/// Dense model: all 32 layers same treatment, no MoE routing.
/// LRU eviction when VRAM budget exhausted.

#include "ggml-backend-vugva.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <new>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

// ============================================================================
// VUGVA pool FFI (Rust library)
// ============================================================================

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
        fprintf(stderr, "[VUGVA] Pool: DRAM=%zu GB, NUMA=%s, pinned=%s (cudaMallocHost=%s)\n",
                info.dram_capacity / 1024 / 1024 / 1024,
                info.numa_bound ? "yes" : "no",
                info.pinned ? "yes" : "no",
                "always");  // actual alloc always uses cudaMallocHost
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

extern "C" ggml_backend_vugva_params ggml_backend_vugva_default_params(void) {
    ggml_backend_vugva_params p = {};
    p.gpu_id = 0;
    p.dram_pool_bytes = 32ULL * 1024 * 1024 * 1024;
    p.spill_enabled = 0;
    p.spill_path = nullptr;
    return p;
}

extern "C" int ggml_backend_vugva_init(ggml_backend_vugva_params params) {
    return vugva_ctx().init(params) ? 0 : -1;
}

// ============================================================================
// VUGVA_DRAM buffer type — pinned host memory via cudaMallocHost
//
// cudaMallocHost allocates page-locked (pinned) memory.
// cudaMemcpyAsync from pinned host → device = true DMA (no CPU staging).
// This IS the CPU bypass: CPU never touches data plane.
// ============================================================================

static bool vugva_is_vugva_buffer(ggml_backend_buffer_t buf) {
    if (!buf || !buf->buft) return false;
    const char * name = buf->buft->iface.get_name ? buf->buft->iface.get_name(buf->buft) : "";
    return strcmp(name, "VUGVA_DRAM") == 0;
}

struct ggml_backend_vugva_buffer_context {
    std::string name;
    size_t      size;
    void *      dram_ptr;  // pinned via cudaMallocHost (DMA-capable)
    bool        owns_dram_ptr;

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
            cudaFreeHost(ctx->dram_ptr);  // unpin
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
    size_t nbytes = ggml_nbytes(src);
    memcpy(dst->data, src->data, nbytes);
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
// VUGVA_DRAM buffer type interface
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

    int rc = vugva_alloc(vctx.pool, name.c_str(), size);
    if (rc != 0) return nullptr;

    // cudaMallocHost = page-locked (pinned) memory.
    // cudaMemcpyAsync H2D from this = true DMA, zero CPU involvement.
    void * dram_ptr = nullptr;
    cudaError_t err = cudaMallocHost(&dram_ptr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "[VUGVA] cudaMallocHost(%zu) failed: %s\n", size, cudaGetErrorString(err));
        vugva_dealloc(vctx.pool, name.c_str());
        return nullptr;
    }

    auto * ctx = new(std::nothrow) ggml_backend_vugva_buffer_context(name, size, dram_ptr);
    if (!ctx) {
        cudaFreeHost(dram_ptr);
        vugva_dealloc(vctx.pool, name.c_str());
        return nullptr;
    }

    ggml_backend_buffer_t buffer = ggml_backend_buffer_init(
        buft, ggml_backend_vugva_buffer_interface, ctx, size);
    if (!buffer) {
        delete ctx;
        cudaFreeHost(dram_ptr);
        vugva_dealloc(vctx.pool, name.c_str());
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
    return vugva_ctx().initialized ? vugva_ctx().dram_bytes : 0;
}

static size_t ggml_backend_vugva_buft_get_alloc_size(
    ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    (void)buft;
    return ggml_nbytes(tensor);
}

// is_host=true: CUDA can DMA from these buffers (pinned memory).
static bool ggml_backend_vugva_buft_is_host(ggml_backend_buffer_type_t buft) {
    (void)buft;
    return true;
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
// VRAM cache — promotes DRAM weights to VRAM via CPU-bypass DMA
//
// Key: (buffer_base + tensor_data_offset) — stable across graph rebuilds.
// Value: CUDA buffer + VRAM pointer.
// Promoted weights STAY in VRAM across tokens (cache hit on next token).
// LRU eviction when VRAM budget exhausted.
// ============================================================================

struct vugva_vram_entry {
    ggml_backend_buffer_t cuda_buf;
    void * vram_ptr;        // VRAM pointer for CUDA compute
    void * dram_src;        // pinned DRAM source for DMA
    size_t nbytes;
    uint64_t last_access;   // token counter for LRU
};

// Cache key: stable identifier for a weight tensor's data
struct vugva_cache_key {
    void * buffer_base;     // ggml_backend_buffer base pointer
    size_t data_offset;     // tensor->data - buffer_base

    bool operator==(const vugva_cache_key & o) const {
        return buffer_base == o.buffer_base && data_offset == o.data_offset;
    }
};

struct vugva_cache_key_hash {
    size_t operator()(const vugva_cache_key & k) const {
        size_t h1 = std::hash<void*>{}(k.buffer_base);
        size_t h2 = std::hash<size_t>{}(k.data_offset);
        return h1 ^ (h2 * 0x9e3779b97f4a7c15ULL);  // better hash mixing
    }
};

struct vugva_vram_cache {
    std::unordered_map<vugva_cache_key, vugva_vram_entry, vugva_cache_key_hash> entries;
    size_t vram_used = 0;
    size_t vram_capacity = 0;
    uint64_t token_counter = 0;
    ggml_backend_buffer_type_t cuda_buft = nullptr;
    size_t promote_count = 0;
    size_t cache_hit_count = 0;
    size_t cache_miss_count = 0;

    void init(size_t capacity, ggml_backend_buffer_type_t buft) {
        vram_capacity = capacity;
        cuda_buft = buft;
    }

    void new_token() {
        token_counter++;
        // Log stats every 100 tokens
        if (token_counter % 100 == 0 && (promote_count + cache_hit_count) > 0) {
            fprintf(stderr, "[VUGVA] VRAM cache: %zu entries, %zu MB/%zu MB, "
                    "promotes=%zu, hits=%zu, misses=%zu\n",
                    entries.size(),
                    vram_used / 1024 / 1024, vram_capacity / 1024 / 1024,
                    promote_count, cache_hit_count, cache_miss_count);
            promote_count = 0;
            cache_hit_count = 0;
            cache_miss_count = 0;
        }
    }

    // Build cache key from tensor
    static vugva_cache_key make_key(struct ggml_tensor * tensor) {
        if (!tensor || !tensor->buffer) return {nullptr, 0};
        void * base = tensor->buffer->iface.get_base(tensor->buffer);
        size_t offset = (char *)tensor->data - (char *)base;
        return {base, offset};
    }

    // Promote a tensor from pinned DRAM → VRAM via CPU-bypass DMA.
    // Returns VRAM pointer. Cache key is (buffer_base + offset).
    void * promote(struct ggml_tensor * tensor) {
        if (!tensor || !tensor->buffer) return nullptr;

        vugva_cache_key key = make_key(tensor);

        // Cache hit? Return existing VRAM pointer
        auto it = entries.find(key);
        if (it != entries.end()) {
            it->second.last_access = token_counter;
            cache_hit_count++;
            return it->second.vram_ptr;
        }

        cache_miss_count++;

        size_t nbytes = ggml_nbytes(tensor);
        if (nbytes == 0) return nullptr;

        // Evict LRU entries until we have space
        while (vram_used + nbytes > vram_capacity && !entries.empty()) {
            evict_lru();
        }

        // If still no space, can't promote
        if (vram_used + nbytes > vram_capacity) {
            fprintf(stderr, "[VUGVA] VRAM full: cannot promote %zu MB (have %zu/%zu MB)\n",
                    nbytes / 1024 / 1024, vram_used / 1024 / 1024, vram_capacity / 1024 / 1024);
            return nullptr;
        }

        // Allocate VRAM buffer
        ggml_backend_buffer_t cuda_buf = cuda_buft->iface.alloc_buffer(cuda_buft, nbytes);
        if (!cuda_buf) return nullptr;

        void * vram_ptr = cuda_buf->iface.get_base(cuda_buf);
        void * dram_ptr = tensor->buffer->iface.get_base(tensor->buffer);
        if (!vram_ptr || !dram_ptr) {
            cuda_buf->iface.free_buffer(cuda_buf);
            return nullptr;
        }

        // CPU-bypass DMA: pinned DRAM → VRAM (zero CPU involvement)
        // cudaMallocHost gives us page-locked memory.
        // cudaMemcpyAsync(H2D) from pinned = true DMA transfer.
        // CPU only touches the 64-byte DMA descriptor, not the data.
        cudaError_t err = cudaMemcpyAsync(
            vram_ptr,                          // dst: VRAM
            (char *)dram_ptr + key.data_offset, // src: pinned DRAM
            nbytes,
            cudaMemcpyHostToDevice,            // H2D = DMA from pinned
            cudaStreamPerThread);
        if (err != cudaSuccess) {
            fprintf(stderr, "[VUGVA] DMA H2D failed: %s\n", cudaGetErrorString(err));
            cuda_buf->iface.free_buffer(cuda_buf);
            return nullptr;
        }

        entries[key] = {cuda_buf, vram_ptr, (char *)dram_ptr + key.data_offset, nbytes, token_counter};
        vram_used += nbytes;
        promote_count++;

        return vram_ptr;
    }

    // Evict least recently used entry
    void evict_lru() {
        if (entries.empty()) return;

        vugva_cache_key lru_key = {nullptr, 0};
        uint64_t lru_time = UINT64_MAX;
        for (auto & [key, entry] : entries) {
            if (entry.last_access < lru_time) {
                lru_time = entry.last_access;
                lru_key = key;
            }
        }

        auto it = entries.find(lru_key);
        if (it != entries.end()) {
            it->second.cuda_buf->iface.free_buffer(it->second.cuda_buf);
            vram_used -= it->second.nbytes;
            entries.erase(it);
        }
    }

    void free_all() {
        for (auto & [key, entry] : entries) {
            entry.cuda_buf->iface.free_buffer(entry.cuda_buf);
        }
        entries.clear();
        vram_used = 0;
    }
};

// ============================================================================
// VUGVA backend context
// ============================================================================

struct ggml_backend_vugva_context_backend {
    int gpu_id;
    ggml_backend_buffer_type_t buft;
    ggml_backend_dev_t dev;
    ggml_backend_t cuda_backend;
    vugva_vram_cache cache;
};

struct vugva_promoted_tensor {
    struct ggml_tensor * tensor;
    void * orig_data;       // original DRAM data pointer
    void * vram_ptr;        // promoted VRAM pointer
    size_t nbytes;
    bool is_weight;         // weights stay in VRAM, intermediates get restored
};

// ============================================================================
// ggml_backend_i implementation
// ============================================================================

static const char * ggml_backend_vugva_get_name(ggml_backend_t backend) {
    (void)backend;
    return "VUGVA";
}

static void ggml_backend_vugva_free(ggml_backend_t backend) {
    auto * ctx = (ggml_backend_vugva_context_backend *)backend->context;
    if (ctx) {
        ctx->cache.free_all();
        if (ctx->cuda_backend) ggml_backend_free(ctx->cuda_backend);
        delete ctx;
    }
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
    if (!src->buffer || !dst->buffer) return false;
    size_t nbytes = ggml_nbytes(src);
    void * src_base = src->buffer->iface.get_base(src->buffer);
    void * dst_base = dst->buffer->iface.get_base(dst->buffer);
    if (!src_base || !dst_base) return false;
    memcpy(dst_base, src_base, nbytes);
    return true;
}

static void ggml_backend_vugva_synchronize(ggml_backend_t backend) {
    (void)backend;
    cudaStreamSynchronize(cudaStreamPerThread);
}

// ============================================================================
// graph_compute — CPU-bypass DMA staging: DRAM → VRAM → CUDA compute
//
// Dense model (all 32 layers same treatment):
//   Phase 1: Promote weight sources DRAM→VRAM via DMA (cache stays warm)
//   Phase 2: Execute each node on CUDA (VRAM data)
//   Phase 3: Restore non-weight tensor pointers
//   Weight tensors stay in VRAM across tokens (cache hit next token)
// ============================================================================

static enum ggml_status ggml_backend_vugva_graph_compute(
    ggml_backend_t backend, struct ggml_cgraph * graph) {

    auto * ctx = (ggml_backend_vugva_context_backend *)backend->context;
    if (!ctx->cuda_backend) return GGML_STATUS_FAILED;

    auto * cuda_buft = ggml_backend_get_default_buffer_type(ctx->cuda_backend);
    if (!cuda_buft) return GGML_STATUS_FAILED;

    // Initialize VRAM cache: 7 GB for weights (leave 1 GB for KV + workspace)
    if (ctx->cache.vram_capacity == 0) {
        size_t vram_budget = 7ULL * 1024 * 1024 * 1024;
        ctx->cache.init(vram_budget, cuda_buft);
        fprintf(stderr, "[VUGVA] VRAM cache: %zu GB budget\n", vram_budget / 1024 / 1024 / 1024);
    }
    ctx->cache.new_token();

    // Track promoted tensors for restore
    std::vector<vugva_promoted_tensor> promoted;
    promoted.reserve(64);  // ~32 layers × 2 tensors per layer

    // Phase 1: Promote weight sources from pinned DRAM → VRAM via CPU-bypass DMA
    // Only promote WEIGHT tensors (persistent). Intermediates are temporary.
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view(node)) continue;

        // Promote weight sources
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (!src || !src->buffer) continue;
            if (!vugva_is_vugva_buffer(src->buffer)) continue;
            if (ggml_nbytes(src) == 0) continue;

            // Skip if already promoted this graph
            vugva_cache_key key = vugva_vram_cache::make_key(src);
            bool already = false;
            for (auto & p : promoted) {
                if (vugva_vram_cache::make_key(p.tensor) == key) { already = true; break; }
            }
            if (already) continue;

            // Promote: pinned DRAM → VRAM via DMA (cache hit if already promoted)
            void * vram_ptr = ctx->cache.promote(src);
            if (vram_ptr) {
                promoted.push_back({src, src->data, vram_ptr, ggml_nbytes(src), true});
                src->data = vram_ptr;  // point tensor to VRAM
            }
            // If promote fails (VRAM full), tensor stays in DRAM (is_host=true)
            // ggml scheduler will handle it via its normal copy mechanism
        }
    }
    cudaStreamSynchronize(cudaStreamPerThread);  // wait for DMA

    // Phase 2: Execute each node on CUDA (VRAM data)
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_cgraph node_graph = ggml_graph_view(graph, i, i + 1);
        enum ggml_status st = ggml_backend_graph_compute(ctx->cuda_backend, &node_graph);
        if (st != GGML_STATUS_SUCCESS) {
            // Restore on failure
            for (auto & p : promoted) {
                p.tensor->data = p.orig_data;
            }
            return st;
        }
    }

    // Phase 3: Restore tensor data pointers for next graph build
    // Weight tensors: restore to DRAM pointer (VRAM cache preserves the data)
    // Next graph: cache hit, no re-copy needed
    for (auto & p : promoted) {
        p.tensor->data = p.orig_data;  // restore original DRAM pointer
    }

    return GGML_STATUS_SUCCESS;
}

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
    /* .graph_compute     = */ ggml_backend_vugva_graph_compute,
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
// ggml_backend_device_i
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
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ true,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
        /* .mmap_support          = */ false,
    };
}

// ============================================================================
// Backend init
// ============================================================================

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

    // Find and init CUDA backend for compute delegation
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            backend_ctx->cuda_backend = ggml_backend_dev_init(d, nullptr);
            break;
        }
    }

    auto * buft_ctx = new(std::nothrow) ggml_backend_vugva_buffer_type_context();
    if (!buft_ctx) { delete backend_ctx; return nullptr; }
    buft_ctx->gpu_id = dev_ctx->gpu_id;

    auto * buft = new(std::nothrow) ggml_backend_buffer_type();
    if (!buft) { delete buft_ctx; delete backend_ctx; return nullptr; }
    buft->iface   = ggml_backend_vugva_buffer_type_interface;
    buft->device  = dev;
    buft->context = buft_ctx;

    backend_ctx->buft = buft;
    backend_ctx->dev  = dev;

    auto * backend = new(std::nothrow) ggml_backend();
    if (!backend) { delete buft; delete buft_ctx; delete backend_ctx; return nullptr; }

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
    auto * ctx = (ggml_backend_vugva_device_context *)dev->context;
    auto * buft_ctx = new ggml_backend_vugva_buffer_type_context();
    buft_ctx->gpu_id = ctx->gpu_id;
    auto * buft = new ggml_backend_buffer_type();
    buft->iface   = ggml_backend_vugva_buffer_type_interface;
    buft->device  = dev;
    buft->context = buft_ctx;
    return buft;
}

static ggml_backend_buffer_type_t ggml_backend_vugva_dev_get_host_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_vugva_dev_get_buffer_type(dev);
}

static ggml_backend_buffer_t ggml_backend_vugva_dev_buffer_from_host_ptr(
    ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    (void)max_tensor_size;
    ggml_backend_vugva_context & vctx = vugva_ctx();
    if (!vctx.initialized) return nullptr;

    static std::atomic<uint64_t> counter{0};
    uint64_t id = counter.fetch_add(1);
    std::string name = "import_buf_" + std::to_string(id);

    int rc = vugva_alloc(vctx.pool, name.c_str(), size);
    if (rc != 0) return nullptr;

    auto * ctx = new(std::nothrow) ggml_backend_vugva_buffer_context(name, size, ptr, false);
    if (!ctx) { vugva_dealloc(vctx.pool, name.c_str()); return nullptr; }

    auto * buft = ggml_backend_vugva_dev_get_buffer_type(dev);
    ggml_backend_buffer_t buffer = ggml_backend_buffer_init(
        buft, ggml_backend_vugva_buffer_interface, ctx, size);
    if (!buffer) { delete ctx; vugva_dealloc(vctx.pool, name.c_str()); return nullptr; }
    return buffer;
}

static bool ggml_backend_vugva_dev_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    (void)dev; (void)op;
    return true;
}

static bool ggml_backend_vugva_dev_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (!buft) return false;
    const char * name = buft->iface.get_name ? buft->iface.get_name(buft) : "";
    return strcmp(name, "VUGVA_DRAM") == 0;
}

static bool ggml_backend_vugva_dev_offload_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    (void)dev; (void)op;
    return true;
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
// ggml_backend_reg_i
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
    dev_ctx.description = "VUGVA DRAM cache for VRAM (CPU-bypass DMA, pinned memory)";

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
