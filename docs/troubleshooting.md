# Troubleshooting

## The run is extremely slow on the first pass, then speeds up

Expected for mmap'd or pageable weights. The first pass faults pages in from
NVMe; the Linux page cache keeps them afterwards. If the model is larger than
RAM minus the desktop, the cache cannot hold it and every pass is disk-bound.
Options:

* free up RAM (close browsers/IDEs),
* use a smaller quant,
* on a dedicated box, raise `VUGVA_PIN_MB` so the warm tier is pinned.

## `[VUGVA] refusing to pin ... — using pageable DRAM instead`

The pin cap (50 % of RAM, ≥ 2 GiB headroom) was hit. This is a **safety
feature**, not a failure: the run continues with pageable memory and, in
zero-copy mode, automatically switches to hybrid. If the machine is dedicated to
inference, raise the cap explicitly:

```bash
VUGVA_PIN_MB=24000 ./build/llama-cli ...
```

Never set it to the full RAM size; leave room for the page cache and the OS.

## `[VUGVA] zero-copy allocation failed — disabling zero-copy`

The GPU could not map a host allocation (driver/OOM). The scheduler degrades to
the CPU backend for warm layers. Check `nvidia-smi` and available RAM. A reboot
clears stale pinned allocations from killed processes.

## KV cache type error / `-ctk q4_0` fails

Quantized KV needs a FlashAttention kernel compiled for that combination. Set
it at configure time, e.g.:

```bash
cmake -B build -DGGML_CUDA=ON \
      -DGGML_CUDA_FA_QUANTS="f16-f16;q8_0-q8_0;q4_0-q4_0"
```

`-fa on` is enabled automatically by the CLI when a non-F16 KV type is
requested.

## `sched_reserve: graph splits` is huge (50–120)

Warm layers are being computed by the CPU (`hybrid`). Every CPU/GPU boundary is
a split. Compare with `VUGVA_MODE=zerocopy`, which normally reports `graph
splits = 1`. A large split count is expected in hybrid mode; it is not an error.

## GPU shows almost no VRAM used on a MoE model

That is the MoE fast path working: attention/SSM stays hot (~1.3 GB for a 35B),
the sparse experts live in mmap'd CPU memory. Check the log:

```text
[VUGVA] MoE fast path: sparse experts -> CPU/mmap, attention/SSM stays hot in VRAM
load_tensors:   CPU_Mapped model buffer size = 19486.84 MiB
```

Disable with `VUGVA_AUTO_MOE=0` to restore whole-layer placement.

## Model loads but output is garbage

* Confirm the same prompt/seed reproduces with `VUGVA_VRAM_MB=100`
  `VUGVA_MODE=zerocopy` (forced tier) vs. full GPU. If the forced tier matches
  the full-GPU text, the tiering data path is correct.
* With greedy sampling (`--temp 0`) small numerical differences can push a model
  into a repetition loop; try `--temp 0.7 --repeat-penalty 1.1`.
* The 0.8B tests in `docs/benchmarks.md` produce identical text in both modes.

## `unable to load model` / OOM at load

* Check free RAM (`free -g`) and `swapon --show`. A previous killed process can
  leave swap full; a reboot or `swapoff -a && swapon -a` (root) clears it.
* The warm tier allocation is visible in the load log; if it is near your RAM
  size, let `auto` choose hybrid or lower `VUGVA_DRAM_GB`.

## Updating llama.cpp

The vendored source is a plain directory. To move to a new upstream snapshot:

```bash
# the upstream git history is kept outside the tree
git --git-dir=../llama-cpp-upstream.git fetch origin master
git --git-dir=../llama-cpp-upstream.git diff <old>..<new> -- \
    src/llama-model.cpp src/llama-model-loader.cpp \
    ggml/src/ggml-cuda/ggml-cuda.cu examples/simple/simple.cpp
```

Re-apply the VUGVA hunks (they are localized: placement block, loader override
hook, CUDA `supports_buft`, CLI), rebuild, and record the new commit in the
README version matrix. Keep the integration tag up to date
(`llama.cpp-b<build>-<shortsha>`).

## Diagnostic commands

```bash
# what placement happened?
VUGVA_MODE=auto ./build/llama-cli ... 2>&1 | grep -E "VUGVA|buffer size|graph splits"

# scheduler detail
GGML_SCHED_DEBUG=1 ./build/llama-cli ... 2>&1 | head -100

# VRAM and RAM
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
free -m; swapon --show
```
