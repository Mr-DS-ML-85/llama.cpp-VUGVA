# ggml-vugva: VUGVA tiered memory backend for ggml
#
# Build as a static library and link with llama.cpp.

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

target_include_directories(ggml-vugva PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(ggml-vugva PUBLIC
    ${VUGVA_LIB}
    CUDA::cudart
    CUDA::cuda_driver
    pthread
    dl
)
