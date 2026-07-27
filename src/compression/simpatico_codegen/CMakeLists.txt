cmake_minimum_required(VERSION 3.24)

set(CMAKE_CUDA_ARCHITECTURES
    native
    CACHE STRING "CUDA architectures (native = detect host GPU)")

project(simpatico_codegen LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(DEFINED ENV{CONDA_PREFIX})
  set(CONDA_PREFIX
      "$ENV{CONDA_PREFIX}"
      CACHE PATH "Conda/pixi prefix")
else()
  message(FATAL_ERROR "CONDA_PREFIX must be set (run inside pixi shell)")
endif()

set(CMAKE_CUDA_COMPILER
    "${CONDA_PREFIX}/bin/nvcc"
    CACHE FILEPATH "Conda nvcc")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  set(CUDA_TARGET_TRIPLE "sbsa-linux")
else()
  set(CUDA_TARGET_TRIPLE "x86_64-linux")
endif()
set(CUDA_TOOLKIT_INCLUDE
    "${CONDA_PREFIX}/targets/${CUDA_TARGET_TRIPLE}/include")
set(CUDA_CCCL_INCLUDE "${CUDA_TOOLKIT_INCLUDE}/cccl")

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include "${CONDA_PREFIX}/include"
  "${CONDA_PREFIX}/include/rapids" "${CUDA_TOOLKIT_INCLUDE}")

add_compile_definitions(LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
add_link_options(-Wl,--enable-new-dtags)
link_directories("${CONDA_PREFIX}/lib")

# Embed the project headers the runtime NVRTC JIT #includes as raw-string
# literals, so a binary distribution needs no simpatico source tree on disk.
# Regenerated whenever either header or the generator script changes.
set(EMBED_JIT_HEADERS_OUT
    "${CMAKE_CURRENT_BINARY_DIR}/generated/codegen/jit/embedded_headers.h")
add_custom_command(
  OUTPUT "${EMBED_JIT_HEADERS_OUT}"
  COMMAND
    ${CMAKE_COMMAND}
    "-DIN_STDINT=${CMAKE_CURRENT_SOURCE_DIR}/include/codegen/stdint_shim.hpp"
    "-DIN_RLE=${CMAKE_CURRENT_SOURCE_DIR}/include/codegen/decode/rle_block.cuh"
    "-DOUT=${EMBED_JIT_HEADERS_OUT}" -P
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_jit_headers.cmake"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/codegen/stdint_shim.hpp"
          "${CMAKE_CURRENT_SOURCE_DIR}/include/codegen/decode/rle_block.cuh"
          "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_jit_headers.cmake"
  COMMENT "Embedding JIT project headers into ${EMBED_JIT_HEADERS_OUT}"
  VERBATIM)
add_custom_target(simpatico_embedded_headers DEPENDS "${EMBED_JIT_HEADERS_OUT}")

# Embed the CCCL (<cuda/std/...>, <cub/...>) header closure the runtime NVRTC
# JIT compiles against. Scanned transitively at build time from the kernel
# prelude roots (so it tracks the CCCL version), and fed to NVRTC as in-memory
# headers -- the JIT then needs no CCCL tree on disk, only the driver + the
# nvrtc runtime the extension already links.
set(EMBED_CCCL_HEADERS_OUT
    "${CMAKE_CURRENT_BINARY_DIR}/generated/codegen/jit/cccl_embedded_headers.cpp"
)
add_custom_command(
  OUTPUT "${EMBED_CCCL_HEADERS_OUT}"
  COMMAND
    ${CMAKE_COMMAND} "-DCCCL_DIR=${CUDA_CCCL_INCLUDE}"
    "-DOUT=${EMBED_CCCL_HEADERS_OUT}" -P
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_cccl_headers.cmake"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_cccl_headers.cmake"
  COMMENT
    "Scanning + embedding CCCL JIT header closure into ${EMBED_CCCL_HEADERS_OUT}"
  VERBATIM)

add_library(
  simpatico STATIC
  # Generated: CCCL header closure embedded for the runtime JIT.
  "${EMBED_CCCL_HEADERS_OUT}"
  # CUDA kernels: operators + offsets
  src/bridge/offsets_cumsum.cu
  src/operators/alp_compressor.cu
  src/operators/alp_rd_compressor.cu
  src/operators/ans_compressor.cu
  src/operators/bitcomp_compressor.cu
  src/operators/cascaded_compressor.cu
  src/operators/snappy_compressor.cu
  src/operators/lz4_compressor.cu
  src/operators/deflate_compressor.cu
  src/operators/bitjoin_bitextract.cu
  src/operators/dictionary_compressor.cu
  src/operators/str_split_compressor.cu
  src/operators/nvcomp_batched_codec.cu
  # Plain-CUDA JIT host
  src/jit/nvrtc_compiler.cpp
  src/jit/kernel_cache.cpp
  src/encode/jit/renderer.cpp
  src/decode/jit/renderer.cpp
  # Runtime: plan executor + bridge + API + rep glue
  src/plan/bit_spec.cpp
  src/plan/plan_dsl.cpp
  src/plan/plan_tree.cpp
  src/plan/operator_registry.cpp
  src/plan/representation_factory.cpp
  src/plan/bitjoin_layout.cpp
  src/plan/compress.cpp
  src/plan/decompress.cpp
  src/util/stream_pool.cpp
  src/util/isoc23_compat.c
  src/bridge/fused_tree_build.cpp
  src/bridge/codegen_runtime.cpp
  src/simpatico_codegen.cpp
  src/api/compressed_table_io.cpp
  src/explore/compression_explorer.cpp)
set_target_properties(simpatico PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(simpatico PUBLIC cxx_std_20)
# nvrtc_compiler.cpp #includes the generated embedded_headers.h from here.
add_dependencies(simpatico simpatico_embedded_headers)
target_include_directories(
  simpatico
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/include
          "${CMAKE_CURRENT_BINARY_DIR}/generated" "${CUDA_TOOLKIT_INCLUDE}"
          "${CUDA_CCCL_INCLUDE}")
# In the Sirius vcpkg build the overlay nvrtc port provides a static NVRTC
# (nvrtc::nvrtc_static) so the extension carries no libnvrtc.so runtime dep.
# Standalone / pixi builds fall back to the conda shared 'nvrtc'.
if(TARGET nvrtc::nvrtc_static)
  set(SIMPATICO_NVRTC_LIB nvrtc::nvrtc_static)
else()
  set(SIMPATICO_NVRTC_LIB nvrtc)
endif()

target_link_libraries(
  simpatico
  PUBLIC cudf
         rmm
         cudart
         nvcomp
         ${SIMPATICO_NVRTC_LIB}
         cuda
         dl)

function(add_codegen_host_test name)
  add_executable(${name} tests/${name}.cpp)
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(
    ${name}
    PRIVATE simpatico
            cudf
            rmm
            cudart
            nvcomp
            ${SIMPATICO_NVRTC_LIB}
            cuda
            dl)
  target_link_options(${name} PRIVATE "-Wl,-rpath,${CONDA_PREFIX}/lib")
endfunction()

add_codegen_host_test(test_compress_with_plan_roundtrip)
add_codegen_host_test(test_compressed_table_io)
add_codegen_host_test(test_leaf_describe)
add_codegen_host_test(test_plan_tree)
add_codegen_host_test(test_fused_tree_build)
add_codegen_host_test(test_jit_kernel_cache_plain)
add_codegen_host_test(test_jit_kernel_cache)
add_codegen_host_test(test_fused_operator_sweep)
add_codegen_host_test(test_operator_sweep)
add_codegen_host_test(test_representation_contract)
add_codegen_host_test(test_bitpack_layout_contract)

# Multi-mode CLI. Target is simpatico_cli to avoid clashing with the lib target;
# the installed binary is named 'simpatico'.
add_executable(simpatico_cli cli/simpatico_main.cpp)
set_target_properties(simpatico_cli PROPERTIES OUTPUT_NAME simpatico)
target_include_directories(simpatico_cli
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(
  simpatico_cli
  PRIVATE simpatico
          cudf
          rmm
          cudart
          nvcomp
          ${SIMPATICO_NVRTC_LIB}
          cuda
          dl)
target_link_options(simpatico_cli PRIVATE "-Wl,-rpath,${CONDA_PREFIX}/lib")

enable_testing()
add_test(NAME compress_with_plan_roundtrip
         COMMAND test_compress_with_plan_roundtrip)
add_test(NAME compressed_table_io COMMAND test_compressed_table_io)
add_test(NAME leaf_describe COMMAND test_leaf_describe)
add_test(NAME plan_tree COMMAND test_plan_tree)
add_test(NAME fused_tree_build COMMAND test_fused_tree_build)
add_test(NAME jit_kernel_cache_plain COMMAND test_jit_kernel_cache_plain)
add_test(NAME jit_kernel_cache COMMAND test_jit_kernel_cache)
add_test(NAME fused_operator_sweep COMMAND test_fused_operator_sweep)
add_test(NAME operator_sweep COMMAND test_operator_sweep)
add_test(NAME representation_contract COMMAND test_representation_contract)
add_test(NAME bitpack_layout_contract COMMAND test_bitpack_layout_contract)
