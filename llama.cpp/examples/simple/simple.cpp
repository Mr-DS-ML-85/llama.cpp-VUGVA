#include "llama.h"
#include "ggml-backend.h"

#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n    %s -m model.gguf [options] [prompt]\n", argv[0]);
    printf("\n");
    printf("options:\n");
    printf("  -n, --n-predict N     tokens to generate (default: 32)\n");
    printf("  -ngl N                layers to offload to the GPU (default: 99)\n");
    printf("  -c N                  context size (default: prompt + n_predict)\n");
    printf("  -b N                  batch size (default: prompt length)\n");
    printf("  -t N                  threads (default: llama default)\n");
    printf("  -fa on|off|auto       flash attention (default: auto)\n");
    printf("  -ctk TYPE             K cache type: f16, bf16, q8_0, q4_0, ... (default: f16)\n");
    printf("  -ctv TYPE             V cache type (default: f16)\n");
    printf("  -ncmoe, --n-cpu-moe N keep MoE expert weights of the first N layers in CPU/DRAM\n");
    printf("  -cmoe, --cpu-moe      keep all MoE expert weights in CPU/DRAM\n");
    printf("  -ncmoe2 ...           (repeatable: -ncmoe 8)\n");
    printf("  --temp T              sampling temperature (default: 0 = greedy)\n");
    printf("  --top-k N             top-k sampling (default: 40)\n");
    printf("  --top-p P             top-p sampling (default: 0.95)\n");
    printf("  --min-p P             min-p sampling (default: 0.05)\n");
    printf("  -rp, --repeat-penalty P  repetition penalty (default: 1.0)\n");
    printf("  --seed N              sampler seed (default: 1234)\n");
    printf("  -p PROMPT             prompt (same as passing it positionally)\n");
    printf("\n");
}

static ggml_type parse_type(const char * s) {
    if (!strcmp(s, "f16") || !strcmp(s, "fp16")) return GGML_TYPE_F16;
    if (!strcmp(s, "bf16"))                  return GGML_TYPE_BF16;
    if (!strcmp(s, "q8_0"))                  return GGML_TYPE_Q8_0;
    if (!strcmp(s, "q5_1"))                  return GGML_TYPE_Q5_1;
    if (!strcmp(s, "q5_0"))                  return GGML_TYPE_Q5_0;
    if (!strcmp(s, "q4_1"))                  return GGML_TYPE_Q4_1;
    if (!strcmp(s, "q4_0"))                  return GGML_TYPE_Q4_0;
    if (!strcmp(s, "iq4_nl"))                return GGML_TYPE_IQ4_NL;
    return GGML_TYPE_COUNT;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string model_path;
    std::string prompt = "Hello my name is";
    int ngl        = 99;
    int n_predict  = 32;
    int n_ctx      = 0;
    int n_batch    = 0;
    int n_threads  = 0;
    int n_cpu_moe  = 0;
    int cpu_moe    = 0;
    float temp     = 0.0f;
    float top_p    = 0.95f;
    float min_p    = 0.05f;
    float rep_pen  = 1.0f;
    int   top_k    = 40;
    unsigned int seed = 1234;
    ggml_type type_k = GGML_TYPE_F16;
    ggml_type type_v = GGML_TYPE_F16;
    int fa = -1; // auto

    std::vector<std::string> moe_patterns; // keep pattern strings alive
    std::vector<llama_model_tensor_buft_override> overrides;

    auto need_val = [&](int & i, const char * flag) -> const char * {
        if (i + 1 >= argc) {
            fprintf(stderr, "error: %s requires a value\n", flag);
            exit(1);
        }
        return argv[++i];
    };

    int i = 1;
    for (; i < argc; i++) {
        const char * a = argv[i];
        if (!strcmp(a, "-m") || !strcmp(a, "--model")) {
            model_path = need_val(i, a);
        } else if (!strcmp(a, "-p") || !strcmp(a, "--prompt")) {
            prompt = need_val(i, a);
        } else if (!strcmp(a, "-n") || !strcmp(a, "--n-predict")) {
            n_predict = atoi(need_val(i, a));
        } else if (!strcmp(a, "-ngl") || !strcmp(a, "--n-gpu-layers") || !strcmp(a, "--gpu-layers")) {
            ngl = atoi(need_val(i, a));
        } else if (!strcmp(a, "-c") || !strcmp(a, "--ctx-size")) {
            n_ctx = atoi(need_val(i, a));
        } else if (!strcmp(a, "-b") || !strcmp(a, "--batch-size")) {
            n_batch = atoi(need_val(i, a));
        } else if (!strcmp(a, "-t") || !strcmp(a, "--threads")) {
            n_threads = atoi(need_val(i, a));
        } else if (!strcmp(a, "-fa") || !strcmp(a, "--flash-attn")) {
            const char * v = need_val(i, a);
            fa = !strcmp(v, "on") || !strcmp(v, "1") || !strcmp(v, "true") ? 1 :
                 !strcmp(v, "off") || !strcmp(v, "0") || !strcmp(v, "false") ? 0 : -1;
        } else if (!strcmp(a, "-ctk") || !strcmp(a, "--cache-type-k")) {
            const char * v = need_val(i, a);
            type_k = parse_type(v);
            if (type_k == GGML_TYPE_COUNT) { fprintf(stderr, "error: unknown cache type %s\n", v); return 1; }
        } else if (!strcmp(a, "-ctv") || !strcmp(a, "--cache-type-v")) {
            const char * v = need_val(i, a);
            type_v = parse_type(v);
            if (type_v == GGML_TYPE_COUNT) { fprintf(stderr, "error: unknown cache type %s\n", v); return 1; }
        } else if (!strcmp(a, "-ncmoe") || !strcmp(a, "--n-cpu-moe")) {
            n_cpu_moe = atoi(need_val(i, a));
        } else if (!strcmp(a, "-cmoe") || !strcmp(a, "--cpu-moe")) {
            cpu_moe = 1;
        } else if (!strcmp(a, "--temp") || !strcmp(a, "--temperature")) {
            temp = (float) atof(need_val(i, a));
        } else if (!strcmp(a, "--top-k")) {
            top_k = atoi(need_val(i, a));
        } else if (!strcmp(a, "--top-p")) {
            top_p = (float) atof(need_val(i, a));
        } else if (!strcmp(a, "--min-p")) {
            min_p = (float) atof(need_val(i, a));
        } else if (!strcmp(a, "-rp") || !strcmp(a, "--repeat-penalty")) {
            rep_pen = (float) atof(need_val(i, a));
        } else if (!strcmp(a, "--seed")) {
            seed = (unsigned int) strtoul(need_val(i, a), nullptr, 10);
        } else if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
            print_usage(argc, argv);
            return 0;
        } else if (a[0] == '-' && a[1] != '\0') {
            fprintf(stderr, "error: unknown option %s\n", a);
            print_usage(argc, argv);
            return 1;
        } else {
            // positional prompt (may span several argv entries)
            prompt = a;
            for (i++; i < argc; i++) {
                prompt += " ";
                prompt += argv[i];
            }
            break;
        }
    }

    if (model_path.empty()) {
        print_usage(argc, argv);
        return 1;
    }

    // --cpu-moe / --n-cpu-moe: keep expert tensors out of VRAM. They still go
    // through the VUGVA DRAM tier (the CPU buffer list prefers it), so in
    // zero-copy mode CUDA can read them directly instead of the CPU computing.
    if (cpu_moe) {
        moe_patterns.emplace_back("\\.ffn_(up|down|gate|gate_up)_(ch|)exps");
    } else {
        for (int l = 0; l < n_cpu_moe; l++) {
            moe_patterns.push_back("blk\\." + std::to_string(l) + "\\.ffn_(up|down|gate|gate_up)_(ch|)exps");
        }
    }
    for (auto & p : moe_patterns) {
        overrides.push_back({ p.c_str(), ggml_backend_cpu_buffer_type() });
    }
    overrides.push_back({ nullptr, nullptr });

    ggml_backend_load_all();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;
    if (!moe_patterns.empty()) {
        model_params.tensor_buft_overrides = overrides.data();
        fprintf(stderr, "%s: [cli] MoE expert override: %zu pattern(s) -> CPU/DRAM tier\n",
                __func__, moe_patterns.size());
    }
    if (n_threads > 0) {
        // threads are a context option in this API
    }

    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (model == nullptr) {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);
    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        fprintf(stderr, "%s: error: failed to tokenize the prompt\n", __func__);
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx     = n_ctx > 0 ? n_ctx : n_prompt + n_predict - 1;
    ctx_params.n_batch   = n_batch > 0 ? n_batch : n_prompt;
    ctx_params.n_ubatch  = ctx_params.n_batch;
    ctx_params.no_perf   = false;
    ctx_params.type_k    = type_k;
    ctx_params.type_v    = type_v;
    if (n_threads > 0) {
        ctx_params.n_threads       = n_threads;
        ctx_params.n_threads_batch = n_threads;
    }
    const bool quant_kv = type_k != GGML_TYPE_F16 || type_v != GGML_TYPE_F16;
    if (fa >= 0) {
        ctx_params.flash_attn_type = fa ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    } else if (quant_kv) {
        // a quantized KV cache requires flash attention on the GPU
        ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        fprintf(stderr, "%s: [cli] quantized KV cache -> enabling flash attention\n", __func__);
    }
    if (quant_kv) {
        fprintf(stderr, "%s: [cli] KV cache type: K=%s V=%s\n", __func__,
                ggml_type_name(type_k), ggml_type_name(type_v));
    }

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        fprintf(stderr, "%s: error: failed to create the llama_context\n", __func__);
        return 1;
    }

    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);

    if (rep_pen > 1.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_penalties(llama_vocab_n_tokens(vocab), 64, rep_pen, 0.0f, 0.0f));
    }
    if (temp <= 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    } else {
        if (top_k > 0) llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
        if (top_p < 1.0f) llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
        if (min_p > 0.0f) llama_sampler_chain_add(smpl, llama_sampler_init_min_p(min_p, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));
    }

    for (auto id : prompt_tokens) {
        char buf[128];
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
            return 1;
        }
        printf("%s", std::string(buf, n).c_str());
    }
    fflush(stdout);

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    if (llama_model_has_encoder(model)) {
        if (llama_encode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval\n", __func__);
            return 1;
        }
        llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
        if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
            decoder_start_token_id = llama_vocab_bos(vocab);
        }
        batch = llama_batch_get_one(&decoder_start_token_id, 1);
    }

    const auto t_main_start = ggml_time_us();
    int n_decode = 0;
    llama_token new_token_id;

    for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + n_predict; ) {
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval\n", __func__);
            return 1;
        }
        n_pos += batch.n_tokens;

        new_token_id = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, new_token_id)) {
            break;
        }
        if (llama_vocab_is_control(vocab, new_token_id)) {
            // keep control tokens out of stdout, but still feed them back
        } else {
            char buf[128];
            int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            if (n < 0) {
                fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
                return 1;
            }
            printf("%s", std::string(buf, n).c_str());
            fflush(stdout);
        }

        batch = llama_batch_get_one(&new_token_id, 1);
        n_decode += 1;
    }

    printf("\n");

    const auto t_main_end = ggml_time_us();
    fprintf(stderr, "%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
            __func__, n_decode, (t_main_end - t_main_start) / 1000000.0f,
            n_decode / ((t_main_end - t_main_start) / 1000000.0f));
    fprintf(stderr, "\n");
    llama_perf_sampler_print(smpl);
    llama_perf_context_print(ctx);
    fprintf(stderr, "\n");

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
