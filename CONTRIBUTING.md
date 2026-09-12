# Contributing to llama-cpp-VUGVA

Thanks for your interest in contributing!

## Development setup

```bash
# Prerequisites
# - CUDA 11.0+
# - 32+ GB RAM
# - NVIDIA GPU (sm_60+)

# Clone with submodules
git clone --recurse-submodules https://github.com/Mr-DS-ML-85/llama-cpp-VUGVA.git

# Build VUGVA
cd vugva && cargo build --release && cd ..

# Build llama-cpp-VUGVA
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build build -j
```

## Pull requests

1. Fork the repo
2. Create a feature branch (`git checkout -b feat/my-feature`)
3. Make changes
4. Run tests: `./build/vugva_test model.gguf`
5. Submit PR

## Code style

- C++17, 4-space indent
- `clang-format` for formatting
- `[[nodiscard]]` on error-returning functions
- Comments explain *why*, not *what*

## Testing

- All changes must pass `cargo test` (VUGVA) and the integration test
- GPU tests require CUDA and a supported GPU
- Memory tests run without GPU (unit tests only)
