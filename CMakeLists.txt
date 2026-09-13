cmake_minimum_required(VERSION 3.18)
project(llama-cpp-VUGVA LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Options ───────────────────────────────────────────────────────────────
option(GGML_CUDA       "Enable CUDA backend"                 ON)
option(VUGVA_ENABLE    "Enable VUGVA tiered memory backend"   ON)
set(VUGVA_DRAM_GB      32  CACHE STRING "DRAM pool size in GB")

# ── Paths ─────────────────────────────────────────────────────────────────
set(LLAMA_DIR  "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp"  CACHE PATH "llama.cpp source")
set(VUGVA_DIR  "/home/irfan/Desktop/VUGVA/vugva"        CACHE PATH "VUGVA source")

if(NOT EXISTS "${LLAMA_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "llama.cpp not found at ${LLAMA_DIR}\n"
        "Clone it: git clone https://github.com/ggml-org/llama.cpp.git ${LLAMA_DIR}")
endif()

# ── CUDA (needed by ggml-cuda AND ggml-vugva) ───────────────────────────
if(GGML_CUDA)
    set(CMAKE_CUDA_ARCHITECTURES "89")  # RTX 4060
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

# ── Build llama.cpp (ggml + llama) ───────────────────────────────────────
# LLAMA_BUILD_COMMON is needed for llama-bench
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
add_subdirectory(${LLAMA_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml_build)
add_subdirectory(${LLAMA_DIR}      ${CMAKE_BINARY_DIR}/llama_build)

# ── VUGVA Backend (ggml → VUGVA → CUDA) ─────────────────────────────────
if(VUGVA_ENABLE)
    add_subdirectory(ggml-vugva)
    # GGML_USE_VUGVA is PUBLIC on ggml — propagates to llama and all consumers
    target_compile_definitions(ggml PUBLIC GGML_USE_VUGVA=1)
    target_include_directories(ggml PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/ggml-vugva)
    # ggml-backend-reg.cpp calls ggml_backend_vugva_reg() — must link into libggml
    target_link_libraries(ggml PUBLIC ggml-vugva)
endif()

# ── CLI binary ──────────────────────────────────────────────────────────
add_executable(llama-cli ${LLAMA_DIR}/examples/simple/simple.cpp)
target_link_libraries(llama-cli PRIVATE llama ggml ggml-base ggml-cpu pthread dl m)
target_include_directories(llama-cli PRIVATE
    ${LLAMA_DIR}/include
    ${LLAMA_DIR}/ggml/include
)
if(GGML_CUDA)
    target_link_libraries(llama-cli PRIVATE ggml-cuda CUDA::cudart CUDA::cuda_driver CUDA::cublas)
endif()
if(VUGVA_ENABLE)
    target_link_libraries(llama-cli PRIVATE ggml-vugva)
endif()

# ── llama-bench ──────────────────────────────────────────────────────────
add_subdirectory(${LLAMA_DIR}/tools/llama-bench ${CMAKE_BINARY_DIR}/llama_bench_build)
target_include_directories(llama-bench-impl PRIVATE
    ${LLAMA_DIR}/common
    ${LLAMA_DIR}/tools/llama-bench
    ${CMAKE_BINARY_DIR}/llama_build/build-info
)
if(VUGVA_ENABLE)
    target_compile_definitions(llama-bench-impl PRIVATE GGML_USE_VUGVA=1)
    target_link_libraries(llama-bench-impl PRIVATE ggml-vugva)
endif()
