# ggml-vugva: VUGVA tiered memory backend for ggml
#
# Flow: ggml → VUGVA (DRAM) → CUDA (VRAM compute)

cmake_minimum_required(VERSION 3.14)
project(ggml-vugva LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find VUGVA static library
find_library(VUGVA_LIB vugva
    PATHS "/home/irfan/Desktop/VUGVA/vugva/target/release"
    NO_DEFAULT_PATH
)

if(NOT VUGVA_LIB)
    message(FATAL_ERROR "libvugva.a not found — build VUGVA first:\n"
        "  cargo build --release --manifest-path /home/irfan/Desktop/VUGVA/vugva/Cargo.toml")
endif()

add_library(ggml-vugva STATIC
    ggml-backend-vugva.cpp
)
set_target_properties(ggml-vugva PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(ggml-vugva PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/llama.cpp/ggml/include
    ${CMAKE_SOURCE_DIR}/llama.cpp/ggml/src
)

# Link VUGVA (DRAM pool) + CUDA (delegated compute) + ggml-base
target_link_libraries(ggml-vugva PUBLIC
    ${VUGVA_LIB}
    ggml-base
    pthread
    dl
)

# CUDA comes from the top-level find_package(CUDAToolkit)
if(GGML_CUDA)
    target_link_libraries(ggml-vugva PUBLIC ggml-cuda CUDA::cudart CUDA::cuda_driver CUDA::cublas)
endif()
