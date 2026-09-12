cmake_minimum_required(VERSION 3.14)
project(llama-cpp-VUGVA LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Options ───────────────────────────────────────────────────────────────
option(VUGVA_ENABLE    "Enable VUGVA tiered memory backend"  ON)
option(LLAMA_CUDA      "Enable CUDA backend"                 ON)
set(VUGVA_DRAM_GB      32  CACHE STRING "DRAM pool size in GB")

# ── Paths ─────────────────────────────────────────────────────────────────
set(LLAMA_DIR  "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp"  CACHE PATH "llama.cpp source")
set(VUGVA_DIR  "/home/irfan/Desktop/VUGVA/vugva"        CACHE PATH "VUGVA source")

# ── Build llama.cpp ───────────────────────────────────────────────────────
if(LLAMA_CUDA)
    set(CMAKE_CUDA_ARCHITECTURES "89")  # RTX 4060
    find_package(CUDAToolkit REQUIRED)
    set(LLAMA_CUDA_FLAGS "-DGGML_CUDA=ON")
else()
    set(LLAMA_CUDA_FLAGS "-DGGML_CUDA=OFF")
endif()

add_subdirectory(${LLAMA_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml_build)
add_subdirectory(${LLAMA_DIR}      ${CMAKE_BINARY_DIR}/llama_build)
add_subdirectory(ggml-vugva)

# ── Test binary ──────────────────────────────────────────────────────────
add_executable(vugva_test tests/test_vugva.cpp)

target_link_libraries(vugva_test PRIVATE
    ggml-vugva
    llama
    ggml
    ggml-base
    ggml-cpu
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::cublas
    pthread
    dl
    m
)

target_compile_definitions(vugva_test PRIVATE
    VUGVA_DRAM_GB=${VUGVA_DRAM_GB}
)
