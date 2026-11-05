# Benchmarks

All numbers were measured on the reference machine below, with the vendored
llama.cpp b12119 and VUGVA v0.2.0. They are decode throughput after the model
is loaded; prompt processing is reported separately where relevant.

## Test machine

| Component | Spec |
|---|---|
| GPU | NVIDIA RTX 4060, 8 GB VRAM (7804 MiB), sm_89 |
| CPU | 16 threads, x86_64 |
| RAM | 29 GiB DDR5 |
| Storage | NVMe, model files on `/run/media` |
| OS / driver | Linux, CUDA 13.2, driver 595.99 |
| Build | `-DGGML_CUDA=ON -DVUGVA_ENABLE=ON -DVUGVA_DRAM_GB=32`, Release |
| KV FA combos | `f16-f16;q8_0-q8_0;q4_0-q4_0;bf16-bf16` |

## Models

| Model | Size | Type |
|---|---|---|
| Qwen3.5 0.8B Q4_K_M | ~530 MiB | dense |
| Qwen3 27B "Fable Distill" Q4_K_M | 17.4 GB | dense |
| Ornith-1.5 35B Q4_K_M | 21.7 GB | `qwen35moe`: 256 experts, 8 active, SSM layers |

## Results

### Qwen3.5 0.8B (fits VRAM)

| Case | Placement | Graph splits | Decode |
|---|---|---|---|
| `-ngl 99` | 25/25 layers + output hot | 1 | 150–188 t/s |
| `VUGVA_VRAM_MB=100`, zerocopy | 8/25 hot (99 MiB) + 395 MiB warm | 1 | 25–45 t/s |
| `VUGVA_VRAM_MB=100`, hybrid | 8/25 hot + 494 MiB warm | 3–35 | 56–82 t/s |

The forced-tier runs produce the **same text** as the full-GPU run, which is the
correctness check for the warm-tier data path.

### Qwen3 27B dense (17.4 GB, exceeds VRAM)

| Case | Placement | Graph splits | Decode |
|---|---|---|---|
| `auto` → zerocopy | 23–24/65 hot (6.3 GB) + 9.5 GB warm | **1** | **1.15–1.18 t/s** |
| hybrid | 14/65 hot + 14.8 GB warm | 70–114 | 0.85–1.2 t/s |
| `-ctk q4_0 -ctv q4_0`, 4 K ctx | same | 1 | KV 256 → 72 MiB |

Zero-copy removes 60–110 graph splits and the CPU never computes a layer, but
throughput is PCIe-bound and lands on par with hybrid for a dense model. This is
expected: a dense model reads every weight per token, and PCIe ~12–25 GB/s is
comparable to dual-channel DRAM bandwidth.

### Ornith 35B MoE (21.7 GB)

| Case | Placement | Graph splits | Decode |
|---|---|---|---|
| `-ngl 99`, whole-layer spill (pre-v0.2.0) | 12/41 hot + 19.8 GB warm | 46–60 | 2.1–2.3 t/s |
| `-cmoe` (manual), hybrid | 41/41 attention hot (1.3 GB) + experts in DRAM | 82 | 2.06 t/s |
| `-cmoe`, experts in mmap, **warm page cache** | 41/41 attention hot + `CPU_Mapped` 19.5 GB | 82 | **19–36 t/s** |
| same, **cold page cache** | same | 82 | ~4 t/s |
| auto (v0.2.0, no flags), warm cache | same | 82 | 24 t/s |

The cold-vs-warm gap is NVMe vs page cache: the first pass reads ~20 GB of
expert weights from disk, later tokens hit RAM. This is the expected behaviour
of a sparse MoE: the active-expert working set (~0.6 GB/token) is small, so the
page cache keeps up once warm.

Sample run (warm cache):

```text
main: decoded 48 tokens in 2.42 s, speed: 19.80 t/s
llama_perf_context_print: prompt eval time =  431.24 ms / 11 tokens (39.20 ms/token)
llama_perf_context_print:       eval time = 1934.53 ms / 47 runs  (41.16 ms/token, 24.30 t/s)
```

## How to reproduce

```bash
# 0.8B baseline and forced tier
./build/llama-cli -m qwen0.8b.gguf -n 32 -ngl 99 "The capital of France is"
VUGVA_VRAM_MB=100 VUGVA_MODE=zerocopy \
  ./build/llama-cli -m qwen0.8b.gguf -n 32 -ngl 99 "The capital of France is"

# dense spill
VUGVA_MODE=zerocopy ./build/llama-cli -m qwen27b.gguf -n 32 -ngl 99 "Write a story"
VUGVA_MODE=hybrid   ./build/llama-cli -m qwen27b.gguf -n 32 -ngl 99 "Write a story"

# MoE fast path (automatic)
./build/llama-cli -m ornith35b.gguf -n 64 -ngl 99 -ctk q4_0 -ctv q4_0 \
    --temp 0.7 --repeat-penalty 1.1 "Explain why the sky is blue"
```

Compare `sched_reserve: graph splits`, the per-buffer `model buffer size` lines,
and `decoded N tokens in T s` in stderr.

## Notes on measurement

* The bundled CLI uses greedy sampling at `--temp 0`, so repeated runs are
  identical; the numbers above use `--temp 0.7` for readable output only.
* `n_batch` defaults to the prompt length; for long prompts set `-b` explicitly
  to keep prompt processing representative.
* The first run after a reboot is always slower for mmap'd weights; run each
  configuration twice and report the second.
