cmake_minimum_required(VERSION 3.24)

set(CMAKE_CUDA_ARCHITECTURES
    native
    CACHE STRING "CUDA architectures (native = detect host GPU)")

project(simpatico_codegen LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# libcudf exports its CCCL dependency, including include directories and
# namespace/feature definitions. Use the same targets in every build layout.
if(NOT TARGET cudf::cudf)
  find_package(cudf REQUIRED CONFIG)
endif()
find_package(CUDAToolkit REQUIRED)
if(NOT TARGET nvcomp::nvcomp)
  find_package(nvcomp REQUIRED CONFIG)
endif()

add_compile_definitions(LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
add_link_options(-Wl,--enable-new-dtags)

# Embed the project headers the runtime NVRTC JIT #includes as raw-string
# literals, so a binary distribution needs no simpatico source tree on disk.
# Regenerated whenever either header or the generator script changes.
set(EMBED_JIT_HEADERS_OUT
    "${CMAKE_CURRENT_BINARY_DIR}/generated/codegen/jit/embedded_headers.h")
add_custom_command(
  OUTPUT "${EMBED_JIT_HEADERS_OUT}"
  COMMAND
    ${CMAKE_COMMAND}
    "-DIN_STDINT=${CMAKE_CURRENT_SOURCE_DIR}/src/codegen/stdint_shim.hpp"
    "-DIN_RLE=${CMAKE_CURRENT_SOURCE_DIR}/src/codegen/decode/rle_block.cuh"
    "-DOUT=${EMBED_JIT_HEADERS_OUT}" -P
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_jit_headers.cmake"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/codegen/stdint_shim.hpp"
          "${CMAKE_CURRENT_SOURCE_DIR}/src/codegen/decode/rle_block.cuh"
          "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_jit_headers.cmake"
          "${CMAKE_CURRENT_SOURCE_DIR}/cmake/jit_header_manifest.cmake"
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
set(EMBED_CCCL_HEADERS_DEPFILE "${EMBED_CCCL_HEADERS_OUT}.d")
# CCCL::CCCL's transitive include directories are resolved at generate time; the
# cmake -P embedder cannot evaluate their generator expressions itself. Pass the
# ordered list to support both installed packages and source checkouts with
# separate cub/, thrust/, and libcudacxx/ roots.
set(EMBED_CCCL_INCLUDE_DIRS
    "${CMAKE_CURRENT_BINARY_DIR}/generated/cccl_include_dirs-$<CONFIG>.txt")
file(
  GENERATE
  OUTPUT "${EMBED_CCCL_INCLUDE_DIRS}"
  CONTENT
    "$<JOIN:$<REMOVE_DUPLICATES:$<TARGET_PROPERTY:CCCL::CCCL,INTERFACE_INCLUDE_DIRECTORIES>>,\n>\n"
)
add_custom_command(
  OUTPUT "${EMBED_CCCL_HEADERS_OUT}"
  COMMAND
    ${CMAKE_COMMAND} "-DINCLUDE_DIRS_FILE=${EMBED_CCCL_INCLUDE_DIRS}"
    "-DOUT=${EMBED_CCCL_HEADERS_OUT}" "-DDEPFILE=${EMBED_CCCL_HEADERS_DEPFILE}"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_cccl_headers.cmake"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_cccl_headers.cmake"
          "${CMAKE_CURRENT_SOURCE_DIR}/cmake/jit_header_manifest.cmake"
          "${EMBED_CCCL_INCLUDE_DIRS}"
  DEPFILE "${EMBED_CCCL_HEADERS_DEPFILE}"
  COMMENT
    "Scanning + embedding CCCL JIT header closure into ${EMBED_CCCL_HEADERS_OUT}"
  VERBATIM)

add_library(
  simpatico STATIC
  # Generated: CCCL header closure embedded for the runtime JIT.
  "${EMBED_CCCL_HEADERS_OUT}"
  # CUDA kernels: operators + offsets
  src/bridge/offsets_cumsum.cu
  src/selection/selection_wave.cu
  src/selection/chunk_row_set_build.cu
  src/selection/row_id_space.cu
  src/selection/chunk_row_set_convert.cu
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
  src/selection/decompression_pushdown_policy.cpp
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
# Keep libcudf's CCCL includes ahead of the toolkit's implicit headers. Imported
# -isystem paths would otherwise be searched after ordinary -I paths.
set_target_properties(simpatico PROPERTIES POSITION_INDEPENDENT_CODE ON
                                           NO_SYSTEM_FROM_IMPORTED ON)
target_compile_features(simpatico PUBLIC cxx_std_20)
# Name of the NVTX domain simpatico's (and Sirius's) ranges are published into.
# The parent project sets this from its own project name; the fallback only
# applies to the standalone simpatico_codegen build, whose PROJECT_NAME is not
# "sirius". PUBLIC so consumers — sirius_extension and, transitively,
# sirius_unittest and parquet_benchmark — compile their copy of the tag type
# against the same name.
if(NOT DEFINED SIRIUS_NVTX_DOMAIN_NAME)
  set(SIRIUS_NVTX_DOMAIN_NAME "sirius")
endif()
target_compile_definitions(
  simpatico PUBLIC CCCL_IGNORE_DEPRECATED_STREAM_REF_HEADER
                   SIRIUS_NVTX_DOMAIN_NAME="${SIRIUS_NVTX_DOMAIN_NAME}")
# nvrtc_compiler.cpp #includes the generated embedded_headers.h from here.
add_dependencies(simpatico simpatico_embedded_headers)
target_include_directories(
  simpatico PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
                    "${CMAKE_CURRENT_BINARY_DIR}/generated")
# In the Sirius vcpkg build the overlay nvrtc port provides a static NVRTC
# (nvrtc::nvrtc_static) so the extension carries no libnvrtc.so runtime dep.
# Standalone / pixi builds use the toolkit's shared NVRTC target.
if(VCPKG_BUILD AND NOT TARGET nvrtc::nvrtc_static)
  # A standalone vcpkg configure has no parent to discover this target.
  find_package(nvrtc REQUIRED CONFIG)
endif()
if(TARGET nvrtc::nvrtc_static)
  set(SIMPATICO_NVRTC_LIB nvrtc::nvrtc_static)
else()
  set(SIMPATICO_NVRTC_LIB CUDA::nvrtc)
endif()

if(VCPKG_BUILD)
  set(SIMPATICO_CUDART_LIB CUDA::cudart_static)
else()
  set(SIMPATICO_CUDART_LIB CUDA::cudart)
endif()
target_link_libraries(
  simpatico
  PUBLIC cudf::cudf
         rmm::rmm
         nvcomp::nvcomp
         ${SIMPATICO_CUDART_LIB}
         ${SIMPATICO_NVRTC_LIB}
         CUDA::cuda_driver
         ${CMAKE_DL_LIBS})

function(add_codegen_host_test name)
  add_executable(${name} tests/${name}.cpp)
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(${name} PRIVATE simpatico)
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
add_codegen_host_test(test_multi_gpu_stream_affinity)
add_codegen_host_test(test_masked_decode_variants)
add_codegen_host_test(test_selection_wave)
add_codegen_host_test(test_chunk_row_set_build)
add_codegen_host_test(test_row_id_space)
add_codegen_host_test(test_chunk_row_set_convert)
add_codegen_host_test(test_decompress_column_rows)
add_codegen_host_test(test_render_signature_contract)

# Multi-mode CLI. Target is simpatico_cli to avoid clashing with the lib target;
# the installed binary is named 'simpatico'.
add_executable(simpatico_cli cli/simpatico_main.cpp)
set_target_properties(simpatico_cli PROPERTIES OUTPUT_NAME simpatico)
target_include_directories(simpatico_cli
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(simpatico_cli PRIVATE simpatico)

enable_testing()
add_subdirectory(tests/cache)
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
add_test(NAME multi_gpu_stream_affinity COMMAND test_multi_gpu_stream_affinity)
add_test(NAME masked_decode_variants COMMAND test_masked_decode_variants)
add_test(NAME selection_wave COMMAND test_selection_wave)
add_test(NAME chunk_row_set_build COMMAND test_chunk_row_set_build)
add_test(NAME row_id_space COMMAND test_row_id_space)
add_test(NAME chunk_row_set_convert COMMAND test_chunk_row_set_convert)
add_test(NAME decompress_column_rows COMMAND test_decompress_column_rows)
add_test(NAME render_signature_contract COMMAND test_render_signature_contract)
