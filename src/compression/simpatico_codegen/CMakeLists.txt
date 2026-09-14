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
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  set(CUDA_TARGET_TRIPLE "sbsa-linux")
else()
  set(CUDA_TARGET_TRIPLE "x86_64-linux")
endif()

if(VCPKG_BUILD)
  # vcpkg ships CCCL 3.4 next to rmm/cudf; the CUDA toolkit (and any active
  # conda env) ship older ones, so vcpkg's include dir has to be searched first
  # or RMM's "requires CCCL 3.3 or newer" guard fires. Checked before
  # CONDA_PREFIX so a vcpkg build started from a pixi shell still picks vcpkg.
  find_package(CUDAToolkit REQUIRED)
  list(GET CUDAToolkit_INCLUDE_DIRS 0 _cuda_inc0)
  set(CUDA_TOOLKIT_INCLUDE "${_cuda_inc0}")
  set(_SIMPATICO_VCPKG_INC
      "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/include")
  # cccl installs cub/, thrust/ and libcudacxx straight into <prefix>/include.
  set(CUDA_CCCL_INCLUDE "${_SIMPATICO_VCPKG_INC}")
  set(_SIMPATICO_INCLUDE_DIRS "${_SIMPATICO_VCPKG_INC}"
                              "${CUDA_TOOLKIT_INCLUDE}")
  set(_SIMPATICO_LIB_DIR "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/lib")
elseif(CONDA_PREFIX)
  set(CMAKE_CUDA_COMPILER
      "${CONDA_PREFIX}/bin/nvcc"
      CACHE FILEPATH "nvcc from prefix")
  set(CUDA_TOOLKIT_INCLUDE
      "${CONDA_PREFIX}/targets/${CUDA_TARGET_TRIPLE}/include")
  if(EXISTS "${CUDA_TOOLKIT_INCLUDE}/cccl")
    set(CUDA_CCCL_INCLUDE "${CUDA_TOOLKIT_INCLUDE}/cccl")
  else()
    set(CUDA_CCCL_INCLUDE "${CUDA_TOOLKIT_INCLUDE}")
  endif()
  set(_SIMPATICO_INCLUDE_DIRS
      "${CONDA_PREFIX}/include" "${CONDA_PREFIX}/include/rapids"
      "${CUDA_TOOLKIT_INCLUDE}")
  set(_SIMPATICO_LIB_DIR "${CONDA_PREFIX}/lib")
else()
  # Docker / system install: discover the CUDA toolkit via CMake and fall back
  # to standard system paths for cudf/rmm/rapids headers.
  find_package(CUDAToolkit REQUIRED)
  list(GET CUDAToolkit_INCLUDE_DIRS 0 _cuda_inc0)
  set(CUDA_TOOLKIT_INCLUDE "${_cuda_inc0}")
  if(EXISTS "${_cuda_inc0}/cccl")
    set(CUDA_CCCL_INCLUDE "${_cuda_inc0}/cccl")
  else()
    set(CUDA_CCCL_INCLUDE "${_cuda_inc0}")
  endif()
  set(_SIMPATICO_INCLUDE_DIRS "/usr/include" "/usr/include/rapids"
                              "${CUDA_TOOLKIT_INCLUDE}")
  set(_SIMPATICO_LIB_DIR "/usr/lib")
  message(
    STATUS
      "CONDA_PREFIX not set — using system CUDA toolkit and /usr for rapids headers"
  )
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include
                    ${_SIMPATICO_INCLUDE_DIRS})

add_compile_definitions(LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
add_link_options(-Wl,--enable-new-dtags)
link_directories("${_SIMPATICO_LIB_DIR}")

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
  src/selection/selection_wave.cu
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

if(VCPKG_BUILD)
  # In the vcpkg build, use CMake imported targets (link by full path) — no
  # shared .so files exist for cudf/rmm/cudart/cuda, and link_directories() is
  # directory-scoped so -lcudf etc. don't resolve in consuming targets. nvcomp
  # comes in transitively from cudf::cudf; cuda_driver is the driver stub.
  target_link_libraries(
    simpatico PUBLIC cudf::cudf rmm::rmm CUDA::cudart_static
                     ${SIMPATICO_NVRTC_LIB} CUDA::cuda_driver dl)
else()
  target_link_libraries(
    simpatico
    PUBLIC cudf
           rmm
           cudart
           nvcomp
           ${SIMPATICO_NVRTC_LIB}
           cuda
           dl)
endif()

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
  target_link_options(${name} PRIVATE "-Wl,-rpath,${_SIMPATICO_LIB_DIR}")
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
add_codegen_host_test(test_render_signature_contract)

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
target_link_options(simpatico_cli PRIVATE "-Wl,-rpath,${_SIMPATICO_LIB_DIR}")

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
add_test(NAME multi_gpu_stream_affinity COMMAND test_multi_gpu_stream_affinity)
add_test(NAME masked_decode_variants COMMAND test_masked_decode_variants)
add_test(NAME selection_wave COMMAND test_selection_wave)
add_test(NAME render_signature_contract COMMAND test_render_signature_contract)
