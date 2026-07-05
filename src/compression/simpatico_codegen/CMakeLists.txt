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
  ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src/util
  "${CONDA_PREFIX}/include" "${CONDA_PREFIX}/include/rapids"
  "${CUDA_TOOLKIT_INCLUDE}")

add_compile_definitions(LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
add_link_options(-Wl,--enable-new-dtags)
link_directories("${CONDA_PREFIX}/lib")

add_library(
  simpatico STATIC
  # CUDA kernels: operators + offsets
  src/bridge/offsets_cumsum.cu
  src/bridge/bitpack_compact.cu
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
  src/operators/nvcomp_scratch.cpp
  # Plain-CUDA JIT host
  src/jit/fused_tree.cpp
  src/jit/nvrtc_compiler.cpp
  src/jit/kernel_cache.cpp
  src/encode/jit/plain_compile.cpp
  src/encode/jit/renderer.cpp
  src/decode/jit/renderer.cpp
  # Runtime: plan executor + bridge + API + rep glue
  src/plan/plan_dsl.cpp
  src/plan/plan_tree.cpp
  src/plan/fusion.cpp
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
  src/explore/operator_catalog.cpp
  src/explore/compression_explorer.cpp)
set_target_properties(simpatico PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(simpatico PUBLIC cxx_std_20)
target_include_directories(
  simpatico
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/include
          "${CUDA_TOOLKIT_INCLUDE}" "${CUDA_CCCL_INCLUDE}")
target_compile_definitions(
  simpatico
  PRIVATE "CODEGEN_JIT_CUDA_INCLUDE=\"${CUDA_TOOLKIT_INCLUDE}\""
          "CODEGEN_JIT_CCCL_INCLUDE=\"${CUDA_CCCL_INCLUDE}\""
          "CODEGEN_JIT_PROJECT_INCLUDE=\"${CMAKE_CURRENT_SOURCE_DIR}/include\"")
target_link_libraries(
  simpatico
  PUBLIC cudf
         rmm
         cudart
         nvcomp
         nvrtc
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
            nvrtc
            cuda
            dl)
  target_link_options(${name} PRIVATE -Wl,--defsym=__libc_single_threaded=0
                      "-Wl,-rpath,${CONDA_PREFIX}/lib")
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

# Benchmark harness (not registered as ctest; kept during transition)
add_executable(compress_with_plan_benchmark
               bench/compress_with_plan_benchmark.cpp)
target_include_directories(compress_with_plan_benchmark
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(
  compress_with_plan_benchmark
  PRIVATE simpatico
          cudf
          rmm
          cudart
          nvcomp
          nvrtc
          cuda
          dl)
target_link_options(
  compress_with_plan_benchmark PRIVATE -Wl,--defsym=__libc_single_threaded=0
  "-Wl,-rpath,${CONDA_PREFIX}/lib")

# simpatico multi-mode CLI (benchmark / explore / compress / decompress /
# verify) CMake target is simpatico_cli to avoid collision with the static
# library target; the installed binary is named 'simpatico'.
add_executable(simpatico_cli cli/simpatico_main.cpp)
set_target_properties(simpatico_cli PROPERTIES OUTPUT_NAME simpatico)
target_include_directories(
  simpatico_cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
                        ${CMAKE_CURRENT_SOURCE_DIR}/cli)
target_link_libraries(
  simpatico_cli
  PRIVATE simpatico
          cudf
          rmm
          cudart
          nvcomp
          nvrtc
          cuda
          dl)
target_link_options(simpatico_cli PRIVATE -Wl,--defsym=__libc_single_threaded=0
                    "-Wl,-rpath,${CONDA_PREFIX}/lib")

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
