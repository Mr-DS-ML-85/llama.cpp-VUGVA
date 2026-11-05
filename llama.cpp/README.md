# llama.cpp (vendored)

This directory is **not** a git submodule and **not** an editable copy of
llama.cpp. It is a trimmed snapshot of upstream llama.cpp with the VUGVA
integration patches applied in place.

| | |
|---|---|
| Upstream | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) |
| Version | **b12119** |
| Commit | [`3de52ac38818978d6da8f386eda43a856e53714a`](https://github.com/ggml-org/llama.cpp/commit/3de52ac38818978d6da8f386eda43a856e53714a) (2026-11-04) |
| License | MIT (see [`LICENSE`](LICENSE)) |
| Part of | [llama-cpp-VUGVA](..) v0.2.1 |

## Read this first

* **Upstream docs are not here.** They live at <https://github.com/ggml-org/llama.cpp/tree/master/docs>
  and <https://github.com/ggml-org/llama.cpp/tree/master/examples>. Everything
  specific to this project is in the [top-level README](../README.md) and
  [`../docs/`](../docs).
* **Do not edit this tree to add features.** VUGVA lives in `../ggml-vugva/`
  and in a small number of marked hunks inside `src/`, `ggml/src/ggml-cuda/`
  and `examples/simple/` (see below). Anything else is upstream code.
* **The snapshot is trimmed.** Files that the build does not use were removed to
  keep the repository small; see [What was removed](#what-was-removed).

## What is here

```
llama.cpp/
├── CMakeLists.txt      upstream build root (unmodified except where noted)
├── cmake/              build helpers (needed)
├── include/            public llama.h API (needed)
├── src/                llama library **VUGVA-patched**
├── ggml/
│   ├── include/ src/   ggml core (needed)
│   └── src/ggml-cuda/  CUDA backend **VUGVA-patched**
├── common/             common helpers (needed by llama-bench / simple)
├── vendor/             vendored third-party headers (needed)
├── examples/simple/    the bundled CLI **VUGVA-patched** -> build/llama-cli
├── tools/llama-bench/  benchmark harness -> build/llama-bench
├── licenses/           third-party license texts
├── AUTHORS
└── LICENSE
```

## VUGVA patches in this tree

Only these files differ from upstream `3de52ac3`:

| File | Change |
|---|---|
| `src/llama-model.cpp` | tier placement (Algorithm 1): VRAM budget, contiguous hot block, `VUGVA_MODE` selection, auto-MoE trigger |
| `src/llama-model-loader.cpp` / `.h` | tensor → buffer-type resolution, `auto_cpu_experts`, `-cmoe`/`-ncmoe` overrides |
| `src/llama-context.cpp` | skip VUGVA warm-tier devices when building the compute device list |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | `supports_buft` accepts `VUGVA_DRAM` in zero-copy mode; async H2D from warm tier; `GGML_OP_OFFLOAD_MIN_BATCH` |
| `examples/simple/simple.cpp` | full CLI: `-ngl/-c/-b/-t/-fa/-ctk/-ctv/-cmoe/-ncmoe`, sampling, help |

Each hunk is prefixed `VUGVA` or `[VUGVA]` where practical. To find them all:

```bash
grep -rn "VUGVA" src/ ggml/src/ggml-cuda/ examples/simple/ --include=*.cpp --include=*.h -l
```

## Building

Normally you do **not** build this directory directly; the top-level project
does it:

```bash
cd ..
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
      -DVUGVA_DIR=/path/to/VUGVA/vugva
cmake --build build -j
```

Useful options (all default OFF here because llama.cpp is added as a
subdirectory): `LLAMA_BUILD_EXAMPLES`, `LLAMA_BUILD_TOOLS`, `LLAMA_BUILD_TESTS`,
`LLAMA_BUILD_SERVER`, `LLAMA_BUILD_MTMD`.

Standalone experimentation is still possible:

```bash
cmake -B build-upstream -DLLAMA_BUILD_TOOLS=ON   # needs the removed dirs back
```

## What was removed

Removed to keep the vendored snapshot small (42 MB instead of 190 MB). None of
it is used by `llama-cli`, `llama-bench` or `vugva_test`:

* `docs/`, `media/`, `*.md` project files (`CONTRIBUTING.md`, `SECURITY.md`,
  `AGENTS.md`, `CLAUDE.md`, `CODEOWNERS`)
* `examples/` except `simple/`
* `tools/` except `llama-bench/`
* `tests/`, `benches/`, `pocs/`, `app/`, `ci/`, `scripts/`, `skills/`
* Python tooling: `gguf-py/`, `conversion/`, `convert_*.py`, `requirements*`,
  `pyproject.toml`, `mypy.ini`, `pyrightconfig.json`, `ty.toml`
* `grammars/`, `models/` (server/test data)
* CI and editor config: `.github/`, `.devops/`, `.gemini/`, `.pi/`,
  `.clang-format`, `.clang-tidy`, `.editorconfig`, `.flake8`, `.gitignore`,
  `.gitmodules` (empty), `.pre-commit-config.yaml`, `Makefile`, `flake.nix`,
  `CMakePresets.json`, `build-xcframework.sh`

The full upstream tree can be restored from the snapshot repository kept next to
this checkout:

```bash
git --git-dir=../llama-cpp-upstream.git worktree add /tmp/llama-up 3de52ac3
```

## Updating to a new upstream snapshot

1. Fetch the new commit in `../llama-cpp-upstream.git`.
2. Diff the five VUGVA-patched files against the previous snapshot.
3. Re-apply the hunks here, build, and test (`vugva_test` + one small model).
4. Update the version table in [`../README.md`](../README.md), `../VERSION` and
   add a `../CHANGELOG.md` entry. Tag `llama.cpp-b<build>-<shortsha>`.

Full procedure: [`../docs/troubleshooting.md`](../docs/troubleshooting.md#updating-llamacpp).

## Attribution

> Irfan Mahir. *VUGVA: A Software-Defined Virtual Unified GPU VRAM Architecture
> with CPU-Bypass Hybrid Memory for Non-NVLink Multi-GPU Clusters.* July 2026.
> DOI: [10.5281/zenodo.21549808](https://doi.org/10.5281/zenodo.21549808)
> Reference implementation: <https://github.com/Mr-DS-ML-85/VUGVA>

llama.cpp is © the ggml authors and contributors, MIT licensed.
