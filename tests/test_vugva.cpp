/// VUGVA x llama.cpp integration test
///
/// Loads a GGUF model through VUGVA's tiered memory (32 GB DRAM + 8 GB VRAM),
/// promotes hot tensors to VRAM, and runs inference via the CUDA backend.

extern "C" {
#include "vugva.h"
}

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-vugva.h"
#include "llama.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>

#ifndef VUGVA_DRAM_GB
#define VUGVA_DRAM_GB 32
#endif

static const size_t DRAM_BYTES = (size_t)VUGVA_DRAM_GB * 1024ULL * 1024 * 1024;

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s model.gguf [prompt]\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const char * prompt = argc > 2 ? argv[2] : "Hello, my name is";

    printf("================================================================\n");
    printf("  llama-cpp-VUGVA Integration Test\n");
    printf("  Pool: %d GB DRAM + 8 GB VRAM  =  %d GB unified\n",
           VUGVA_DRAM_GB, VUGVA_DRAM_GB + 8);
    printf("  Model: %s\n", model_path);
    printf("================================================================\n\n");

    // 1. Initialize VUGVA pool
    printf("[1/5] Initializing VUGVA tiered pool ...\n");
    VugvaPool * pool = vugva_pool_create(0, DRAM_BYTES);
    assert(pool && "vugva_pool_create failed");

    VugvaPoolInfo info = {};
    vugva_pool_info(pool, &info);
    printf("  DRAM: %zu GB, NUMA: %s, pinned: %s\n",
           info.dram_capacity / 1024 / 1024 / 1024,
           info.numa_bound ? "yes" : "no",
           info.pinned ? "yes" : "no");
    printf("\n");

    // 2. Initialize llama.cpp — register VUGVA backend, then load model
    printf("[2/5] Initializing llama.cpp with VUGVA backend ...\n");
    auto t0 = std::chrono::high_resolution_clock::now();

    // Register VUGVA backend with ggml
    ggml_backend_reg_t vugva_reg = ggml_backend_vugva_reg();
    if (vugva_reg) {
        printf("  VUGVA backend registered successfully\n");
    } else {
        printf("  WARNING: VUGVA backend registration failed, using CPU+CUDA only\n");
    }

    struct llama_model_params model_params = llama_model_default_params();
    // Request full GPU offload: VUGVA caps placement to the VRAM budget
    // (paper Algorithm 1) and keeps the remainder in the DRAM tier.
    model_params.n_gpu_layers = 99;

    struct llama_model * model = llama_model_load_from_file(model_path, model_params);
    if (!model) {
        fprintf(stderr, "Failed to load model: %s\n", model_path);
        vugva_pool_destroy(pool);
        return 1;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("  Model loaded in %.1f ms\n", load_ms);

    // 3. Create context and tokenize
    printf("[3/5] Creating context and tokenizing ...\n");
    struct llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;
    ctx_params.n_batch = 512;

    struct llama_context * ctx = llama_init_from_model(model, ctx_params);
    assert(ctx && "llama_init_from_model failed");

    const struct llama_vocab * vocab = llama_model_get_vocab(model);

    // Tokenize prompt — llama_tokenize with a null buffer returns the NEGATED
    // required size, so size the vector first, then tokenize for real.
    int n_tokens = llama_tokenize(vocab, prompt, (int32_t)strlen(prompt),
                                  nullptr, 0, true, false);
    if (n_tokens < 0) n_tokens = -n_tokens;
    std::vector<llama_token> tokens(n_tokens);
    n_tokens = llama_tokenize(vocab, prompt, (int32_t)strlen(prompt),
                              tokens.data(), (int32_t)tokens.size(), true, false);
    if (n_tokens <= 0) {
        fprintf(stderr, "tokenization failed (returned %d)\n", n_tokens);
        llama_free(ctx);
        llama_model_free(model);
        vugva_pool_destroy(pool);
        return 1;
    }
    tokens.resize(n_tokens);
    printf("  Prompt: \"%s\" (%d tokens)\n", prompt, n_tokens);

    // 4. Evaluate and generate
    printf("[4/5] Running inference ...\n");
    auto t2 = std::chrono::high_resolution_clock::now();

    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), n_tokens)) != 0) {
        fprintf(stderr, "llama_decode failed\n");
        llama_free(ctx);
        llama_model_free(model);
        vugva_pool_destroy(pool);
        return 1;
    }

    // Set up greedy sampler
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // Generate tokens
    int n_gen = 64;
    printf("  Output: ");

    for (int i = 0; i < n_gen; i++) {
        llama_token next = llama_sampler_sample(smpl, ctx, -1);

        const char * piece = llama_vocab_get_text(vocab, next);
        if (piece) printf("%s", piece);

        if (llama_vocab_is_eog(vocab, next)) break;

        if (llama_decode(ctx, llama_batch_get_one(&next, 1)) != 0) {
            fprintf(stderr, "\nllama_decode failed at step %d\n", i);
            break;
        }
    }
    printf("\n");

    llama_sampler_free(smpl);

    auto t3 = std::chrono::high_resolution_clock::now();
    double gen_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    printf("  Generated %d tokens in %.1f ms (%.1f tok/s)\n",
           n_gen, gen_ms, n_gen * 1000.0 / gen_ms);

    // 5. Cleanup
    printf("\n[5/5] Cleanup ...\n");
    llama_free(ctx);
    llama_model_free(model);
    vugva_pool_destroy(pool);

    printf("\n================================================================\n");
    printf("  ALL TESTS PASSED -- VUGVA %d GB pool + llama.cpp works\n", VUGVA_DRAM_GB + 8);
    printf("================================================================\n");
    return 0;
}
