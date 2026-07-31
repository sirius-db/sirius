# =============================================================================
# Copyright 2025, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

project(sirius LANGUAGES CXX CUDA)

# Set CUDA architecture
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
  message(STATUS "CMAKE_CUDA_ARCHITECTURES not defined. Setting to 'native'.")
endif()

# Optional features
option(ENABLE_STREAM_CHECK
       "Enable stream check library for debugging default stream usage" OFF)

# The [s3] integration tests launch MinIO via a patched build of upstream
# testcontainers-native, fetched + patched at configure time and built as a Go
# c-archive — so it needs a Go toolchain and network at configure/build time. ON
# by default (the pixi env provides Go); turn OFF for offline / Go-less /
# release / package builds, which then skip the S3 container harness entirely.
option(SIRIUS_BUILD_S3_TESTS
       "Build the testcontainers-backed S3 integration test harness" ON)

# In the vcpkg build, vcpkg headers must take priority over: 1) conda's CCCL
# 3.1.4 (injected via CXXFLAGS as -I.../include/cccl) 2) DuckDB's bundled fmt
# 6.1.2 (uses duckdb_fmt namespace, incompatible with spdlog) Fix: strip conda
# CCCL path; use NO_SYSTEM_FROM_IMPORTED + BEFORE include (applied per-target
# below) to place vcpkg includes first.
if(VCPKG_BUILD)
  set(_VCPKG_INC "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/include")
  string(REGEX REPLACE "-I[^ ]*/include/cccl" "" CMAKE_CXX_FLAGS
                       "${CMAKE_CXX_FLAGS}")
  string(REGEX REPLACE "-I[^ ]*/include/cccl" "" CMAKE_CUDA_FLAGS
                       "${CMAKE_CUDA_FLAGS}")

  # The loadable extension must not depend on libcudart.so in the vcpkg build.
  # Set the CUDA language default to the static runtime before any CUDA targets
  # are created.
  set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

  # Strip conda-injected -Wl,-rpath flags from linker flags and disable CMake's
  # RPATH mechanism so binaries don't carry a hardcoded path to the pixi/conda
  # env. All deps are statically linked, so no RPATH is needed. FORCE is
  # required because DuckDB's CMakeLists.txt sets this cache var first.
  set(CMAKE_SKIP_RPATH
      ON
      CACHE BOOL "" FORCE)
  foreach(_var CMAKE_SHARED_LINKER_FLAGS CMAKE_EXE_LINKER_FLAGS
               CMAKE_MODULE_LINKER_FLAGS)
    string(REGEX REPLACE "-Wl,-rpath-link[=,][^ ]*" "" ${_var} "${${_var}}")
    string(REGEX REPLACE "-Wl,-rpath[=,][^ ]*" "" ${_var} "${${_var}}")
    set(${_var}
        "${${_var}}"
        CACHE STRING "" FORCE)
  endforeach()
endif()

# DuckDB's unity build emits strong symbols for static constexpr members (e.g.
# duckdb::TableCatalogEntry::Name) that also appear as inline definitions in
# headers pulled in by the vendored substrait reader
# (substrait/src/from_substrait.cpp). Tolerate the benign duplicate on every
# binary that links both libsirius_extension and libduckdb_static — the duckdb
# shell, the loadable extension, and the C++ test/bench exes — mirroring the
# per-target flag already on sirius_unittest / parquet_benchmark (and the Rust
# bindings).
foreach(_sirius_link_flags CMAKE_EXE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS
                           CMAKE_MODULE_LINKER_FLAGS)
  string(APPEND ${_sirius_link_flags} " -Wl,--allow-multiple-definition")
  set(${_sirius_link_flags}
      "${${_sirius_link_flags}}"
      CACHE STRING "" FORCE)
endforeach()

# Dependencies
find_package(cudf REQUIRED CONFIG)
find_package(spdlog REQUIRED CONFIG)
find_package(yaml-cpp REQUIRED CONFIG)
find_package(absl REQUIRED CONFIG)
find_package(PkgConfig REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(ZLIB REQUIRED)

# Static NVRTC (vcpkg overlay port). Statically linking the runtime JIT compiler
# keeps libnvrtc.so out of the distributed extension's runtime dependencies.
# Like cuco/nvcomp, this is gated on the vcpkg build: the overlay port provides
# the nvrtc::nvrtc_static target, which simpatico links instead of the bare
# 'nvrtc' library name (see src/compression/simpatico_codegen/CMakeLists.txt).
# The pixi build has no such target and links the conda shared libnvrtc.so.
if(VCPKG_BUILD)
  find_package(nvrtc CONFIG REQUIRED)
endif()

# --- cuCollections (cuco) --- #

# libcudf no longer ships its bundled copy. In the vcpkg build cuco comes from
# the overlay port (vcpkg_ports/cuco); configure-time downloads are disabled
# there. Otherwise (pixi build) fetch the same commit cudf is built against
# (populate sources without add_subdirectory; CCCL comes from cudf). Either way
# cuco is header-only and exposed as the cuco::cuco target, so both paths
# consume it identically below.
if(VCPKG_BUILD)
  find_package(cuco CONFIG REQUIRED)
else()
  include(FetchContent)
  FetchContent_Declare(
    cuco
    URL https://github.com/NVIDIA/cuCollections/archive/f517bbb1277753b1852dfd388993383e401eaa38.tar.gz
    URL_HASH
      SHA256=d17e2f6d58cfb2cd473efcbd741c633e7607717102cedc08173a4a69bb2a8ba9
    SOURCE_SUBDIR do-not-build)
  FetchContent_MakeAvailable(cuco)
  # SOURCE_SUBDIR do-not-build populates headers without running cuco's CMake,
  # so no target is created. Define the same header-only cuco::cuco the vcpkg
  # port exports (CCCL still comes from cudf) so both build paths consume cuco
  # identically below.
  if(NOT TARGET cuco::cuco)
    add_library(cuco::cuco INTERFACE IMPORTED)
    set_target_properties(cuco::cuco PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                                "${cuco_SOURCE_DIR}/include")
  endif()
endif()

pkg_check_modules(NUMA REQUIRED IMPORTED_TARGET numa)
pkg_check_modules(LIBURING REQUIRED IMPORTED_TARGET liburing)
pkg_check_modules(CURL REQUIRED IMPORTED_TARGET libcurl)

# cuCascade - GPU Memory Reservation Library (submodule)
set(BUILD_TESTS
    OFF
    CACHE BOOL "" FORCE)
set(CUCASCADE_BUILD_TESTS
    OFF
    CACHE BOOL "" FORCE)
set(CUCASCADE_BUILD_BENCHMARKS
    OFF
    CACHE BOOL "" FORCE)
set(CUCASCADE_BUILD_SHARED_LIBS
    OFF
    CACHE BOOL "" FORCE)
set(CUCASCADE_BUILD_STATIC_LIBS
    ON
    CACHE BOOL "" FORCE)
# Sirius consumes cucascade's cudf-coupled representations and converters (PR
# #150 split these into the optional cucascade_cudf target), so build it.
set(CUCASCADE_BUILD_CUDF
    ON
    CACHE BOOL "" FORCE)
set(CUCASCADE_WARNINGS_AS_ERRORS
    OFF
    CACHE BOOL "" FORCE)
set(CUCASCADE_BUILD_IO
    OFF
    CACHE BOOL "" FORCE)
add_subdirectory(cucascade "${CMAKE_BINARY_DIR}/cucascade" EXCLUDE_FROM_ALL)

# Simpatico GPU compression engine Tests disabled here — run them via the
# standalone simpatico_codegen build. Benchmark harness
# (compress_with_plan_benchmark) is built by default.
add_subdirectory(src/compression/simpatico_codegen
                 "${CMAKE_BINARY_DIR}/simpatico_codegen" EXCLUDE_FROM_ALL)

if(VCPKG_BUILD AND TARGET CUDA::cudart_static)
  function(sirius_prefer_static_cudart target_name)
    foreach(_prop LINK_LIBRARIES INTERFACE_LINK_LIBRARIES)
      get_target_property(_libs "${target_name}" "${_prop}")
      if(NOT _libs OR _libs STREQUAL "_libs-NOTFOUND")
        continue()
      endif()

      set(_patched_libs "${_libs}")
      list(TRANSFORM _patched_libs REPLACE "^CUDA::cudart$"
                                           "CUDA::cudart_static")
      list(TRANSFORM _patched_libs REPLACE "^\\$<LINK_ONLY:CUDA::cudart>$"
                                           "$<LINK_ONLY:CUDA::cudart_static>")

      if(NOT _patched_libs STREQUAL _libs)
        set_target_properties("${target_name}" PROPERTIES "${_prop}"
                                                          "${_patched_libs}")
      endif()
    endforeach()

    set_target_properties("${target_name}" PROPERTIES CUDA_RUNTIME_LIBRARY
                                                      Static)
  endfunction()

  foreach(_target
          cucascade_objects cucascade_static cucascade_shared
          cucascade_cudf_objects cucascade_cudf_static cucascade_cudf_shared)
    if(TARGET "${_target}")
      sirius_prefer_static_cudart("${_target}")
    endif()
  endforeach()
endif()

# Rust telemetry instrumentation (C++ FFI via Corrosion)
add_subdirectory(rust/crates/telemetry/bridge)

# Upstream testcontainers-native, fetched + patched at configure time (see
# cmake/testcontainers_native.cmake), used by the S3 integration test harness to
# start MinIO containers from the test binary. Builds a Go c-archive, so a Go
# toolchain (provided by pixi) and network access on the first configure/build
# are required — hence gated behind SIRIUS_BUILD_S3_TESTS.
if(SIRIUS_BUILD_S3_TESTS)
  include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/testcontainers_native.cmake")
endif()

# cmake-format: off
set(EXTENSION_SOURCES
    src/compression/compressed_representation.cpp
    src/compression/compression_converters.cpp
    src/compression/plan_register.cpp
    src/config.cpp
    src/helper/type_conversions.cpp
    src/debug_utils.cpp
    src/creator/task_creator.cpp
    src/downgrade/downgrade_executor.cpp
    src/expression/aggregate_id.cpp
    src/expression/ast/node.cpp
    src/expression/ast/utils.cpp
    src/expression/from_duckdb.cpp
    src/expression/function_id.cpp
    src/expression/join_condition.cpp
    src/expression/to_duckdb.cpp
    src/expression/value.cpp
    src/expression_evaluator/expression_evaluator_strategy.cpp
    src/expression_evaluator/ast_op_counter.cpp
    src/expression_evaluator/expression_evaluator.cpp
    src/expression_evaluator/gpu_expression_translator.cpp
    src/expression_evaluator/regex/regex_playground.cpp
    src/expression_evaluator/specializations/between.cpp
    src/expression_evaluator/specializations/case.cpp
    src/expression_evaluator/specializations/cast.cpp
    src/expression_evaluator/specializations/comparison.cpp
    src/expression_evaluator/specializations/conjunction.cpp
    src/expression_evaluator/specializations/constant.cpp
    src/expression_evaluator/specializations/function.cpp
    src/expression_evaluator/specializations/operator.cpp
    src/expression_evaluator/specializations/reference.cpp
    # Execution primitives (slot-based backpressure / completion tracking)
    src/exec/admission_control.cpp
    # I/O adapters (Phase 19 — io_uring + sirius_datasource)
    src/io/datasource_factory.cpp
    src/io/io_context.cpp
    src/io/parquet_helpers.cpp
    src/io/sirius_datasource.cpp
    src/io/uri_parser.cpp
    src/io/cache/metadata_store.cpp
    src/io/cache/prefetching_cache.cpp
    src/io/cache/types.cpp
    src/io/kvikio/kvikio_context.cpp
    src/io/rest/curl_handle.cpp
    src/io/rest/rest_ioctx.cpp
    src/io/rest/rest_reactor.cpp
    src/io/s3/s3_list_parser.cpp
    src/io/s3/sigv4.cpp
    src/io/s3/sirius_httpfs.cpp
    src/io/s3/sirius_sigv4_authorizer.cpp
    src/io/uring/uring_ioctx.cpp
    src/io/uring/uring_reactor.cpp
    src/log/duckdb_sink.cpp
    src/log/level.cpp
    src/log/logging.cpp
    src/log/noop_sink.cpp
    src/log/spdlog_owning_sink.cpp
    src/memory/defragmenter_oom_policy.cpp
    src/memory/sirius_memory_reservation_manager.cpp
    src/op/aggregate/aggregate_op_util.cpp
    src/op/aggregate/gpu_aggregate_impl.cpp
    src/op/merge/gpu_merge_impl.cpp
    src/op/order/gpu_order_impl.cpp
    src/op/partition/gpu_partition_impl.cpp
    src/op/result/host_table_chunk_reader.cpp
    src/op/scan/cached_ranges.cpp
    src/op/scan/duckdb_insert_delta.cpp
    src/op/scan/duckdb_mvcc_visibility.cpp
    src/op/scan/duckdb_native_metadata.cpp
    src/op/scan/duckdb_native_decoder.cpp
    src/op/scan/dynamic_filter_merge.cpp
    src/op/scan/sirius_physical_dynamic_filter.cpp
    src/op/scan/host_keep_mask.cpp
    src/op/scan/owning_table_view.cpp
    src/op/scan/parquet_schema_mapping.cpp
    src/op/scan/scan_plan.cpp
    src/op/scan/scan_utils.cpp
    src/op/scan/sirius_gpu_scan_operator.cpp
    src/op/scan/sirius_gpu_scan_operator_data.cpp
    src/op/scan/gpu_ingestible.cpp
    src/op/scan/parquet_gpu_ingestible.cpp
    src/op/scan/duckdb_native_gpu_ingestible.cpp
    src/op/dynamic_filter_publish_plan.cpp
    src/op/dynamic_filter_publisher.cpp
    src/op/sirius_dynamic_filter.cpp
    src/op/sirius_physical_column_data_scan.cpp
    src/op/sirius_physical_streaming_source.cpp
    src/op/sirius_physical_concat.cpp
    src/op/sirius_physical_cte.cpp
    src/op/sirius_physical_delim_join.cpp
    src/op/sirius_physical_dummy_scan.cpp
    src/op/sirius_physical_empty_result.cpp
    src/op/sirius_physical_filter.cpp
    src/op/sirius_physical_gpu_values.cpp
    src/op/sirius_physical_grouped_aggregate.cpp
    src/op/sirius_physical_grouped_aggregate_merge.cpp
    src/op/sirius_physical_hash_join.cpp
    src/op/sirius_physical_limit.cpp
    src/op/sirius_physical_merge_sort.cpp
    src/op/sirius_physical_nested_loop_join.cpp
    src/op/sirius_physical_operator.cpp
    src/op/sirius_physical_operator_type.cpp
    src/op/sirius_physical_order.cpp
    src/op/sirius_physical_partition.cpp
    src/op/sirius_physical_partition_consumer_operator.cpp
    src/op/sirius_physical_projection.cpp
    src/op/sirius_physical_result_collector.cpp
    src/op/sirius_physical_sort_partition.cpp
    src/op/sirius_physical_sort_sample.cpp
    src/op/sirius_physical_table_scan.cpp
    src/op/sirius_physical_top_n.cpp
    src/op/sirius_physical_ungrouped_aggregate.cpp
    src/parallel/task_executor.cpp
    src/pipeline/downgraded_task_prefetcher.cpp
    src/pipeline/gpu_pipeline_executor.cpp
    src/pipeline/gpu_pipeline_task.cpp
    src/pipeline/sirius_pipeline_itask.cpp
    src/pipeline/task_scheduler.cpp
    src/pipeline/sirius_meta_pipeline.cpp
    src/pipeline/sirius_pipeline.cpp
    src/pipeline/sirius_pipeline_converter.cpp
    src/pipeline/repository_wiring_materializer.cpp
    src/pipeline/sirius_plan_printer.cpp
    src/pipeline/task_request.cpp
    src/planner/duckdb_join_filter_candidate_adapter.cpp
    src/planner/query.cpp
    src/planner/query_index.cpp
    src/planner/sirius_physical_plan_generator.cpp
    src/planner/sirius_plan_aggregate.cpp
    src/planner/sirius_plan_column_data_get.cpp
    src/planner/sirius_plan_comparison_join.cpp
    src/planner/sirius_plan_cte.cpp
    src/planner/sirius_plan_delim_get.cpp
    src/planner/sirius_plan_delim_join.cpp
    src/planner/sirius_plan_dummy_scan.cpp
    src/planner/sirius_plan_empty_result.cpp
    src/planner/sirius_plan_expression_get.cpp
    src/planner/sirius_plan_filter.cpp
    src/planner/sirius_plan_get.cpp
    src/planner/sirius_plan_limit.cpp
    src/planner/sirius_plan_order.cpp
    src/planner/sirius_plan_projection.cpp
    src/planner/sirius_plan_projection_utils.cpp
    src/planner/sirius_plan_recursive_cte.cpp
    src/planner/sirius_plan_top_n.cpp
    src/pin_table.cpp
    src/scan_manager/load_balancing_scan_batch_coalescer.cpp
    src/scan_manager/insert_delta_job.cpp
    src/scan_manager/mvcc_mask_job.cpp
    src/scan_manager/pinned_chunk_stats.cpp
    src/scan_manager/round_robin_strategy.cpp
    src/scan_manager/memory_prefetcher.cpp
    src/scan_manager/sirius_scan_manager.cpp
    src/scan_manager/split_connector.cpp
    src/scan_manager/split_provider.cpp
    src/sirius_config.cpp
    src/sirius_context.cpp
    src/sirius_engine.cpp
    src/sirius_extension.cpp
    src/sirius_ffi.cpp
    src/sirius_interface.cpp
    src/sirius_sql_rewrite.cpp
    src/telemetry/batch_telemetry.cpp
    src/telemetry/telemetry_context.cpp
    src/telemetry/memory_context.cpp
    src/transparent/physical_sirius_execution.cpp
    src/transparent/sirius_optimizer_extension.cpp
    src/util/stream_check_wrapper.cpp
    src/util/segfault_backtrace_handler.cpp)
# cmake-format: on

# Legacy Sirius — all legacy-specific config lives in src/legacy/CMakeLists.txt.

# Deleting src/legacy/ cleanly removes legacy build support.
set(CUDA_SOURCES
    src/op/partition/crc32_partition_hash.cu
    src/cuda/dynamic_filter_replica_transfer.cu
    src/cuda/sirius_dynamic_bloom_filter.cu
    src/cuda/sirius_dynamic_in_list_filter.cu
    src/cuda/sirius_dynamic_small_in_list_filter.cu
    src/cuda/scan/gpu_decode_alp.cu
    src/cuda/scan/gpu_decode_bitpacking.cu
    src/cuda/scan/gpu_decode_rle.cu
    src/cuda/scan/gpu_decode_strings.cu
    src/cuda/scan/gpu_native_decode.cu
    src/cuda/scan/strings/dict_fsst.cu
    src/cuda/scan/strings/dictionary.cu
    src/cuda/scan/strings/fsst.cu
    src/cuda/scan/strings/uncompressed.cu)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/legacy/CMakeLists.txt")
  add_subdirectory(src/legacy)
  list(APPEND EXTENSION_SOURCES ${SIRIUS_LEGACY_SOURCES}
       ${SIRIUS_LEGACY_CONDITIONAL_SOURCES})
  list(APPEND CUDA_SOURCES ${SIRIUS_LEGACY_CUDA_SOURCES})
endif()

# Compile the DuckDB substrait extension's Substrait->DuckDB reader directly
# into the sirius target so the FFI (src/sirius_ffi.cpp) can call
# `duckdb::SubstraitToDuckDB`. Only the from-substrait direction plus its
# bundled protobuf (renamespaced to `duckdb::google::protobuf`, no abseil) are
# compiled in — NOT the substrait SQL functions, to_substrait, or a second
# loadable extension — so there is no symbol collision with conda/system
# protobuf or a duplicate extension.
set(SIRIUS_SUBSTRAIT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/substrait")
file(GLOB_RECURSE SIRIUS_SUBSTRAIT_PROTOBUF_SOURCES
     "${SIRIUS_SUBSTRAIT_DIR}/third_party/google/protobuf/*.cc")
set(SIRIUS_SUBSTRAIT_SOURCES
    ${SIRIUS_SUBSTRAIT_DIR}/src/from_substrait.cpp
    ${SIRIUS_SUBSTRAIT_DIR}/src/custom_extensions.cpp
    ${SIRIUS_SUBSTRAIT_DIR}/src/custom_extensions_generated.cpp
    ${SIRIUS_SUBSTRAIT_DIR}/third_party/substrait/substrait/algebra.pb.cc
    ${SIRIUS_SUBSTRAIT_DIR}/third_party/substrait/substrait/extended_expression.pb.cc
    ${SIRIUS_SUBSTRAIT_DIR}/third_party/substrait/substrait/plan.pb.cc
    ${SIRIUS_SUBSTRAIT_DIR}/third_party/substrait/substrait/type.pb.cc
    ${SIRIUS_SUBSTRAIT_DIR}/third_party/substrait/substrait/extensions/extensions.pb.cc
    ${SIRIUS_SUBSTRAIT_PROTOBUF_SOURCES})
# The vendored/generated protobuf + substrait TUs are not warning-clean; never
# fail the build on them.
set_source_files_properties(${SIRIUS_SUBSTRAIT_SOURCES}
                            PROPERTIES COMPILE_OPTIONS "-w")
list(APPEND EXTENSION_SOURCES ${SIRIUS_SUBSTRAIT_SOURCES})

build_static_extension(sirius ${EXTENSION_SOURCES} ${CUDA_SOURCES})
build_loadable_extension(sirius CPP ${EXTENSION_SOURCES} ${CUDA_SOURCES})

# rapidsai/rmm#826: DuckDB links the loadable extension with
# -Wl,--exclude-libs,ALL, under which mold (but not bfd) hides RMM's GNU_UNIQUE
# current-device-resource registry symbols, so cuDF's internal allocations
# bypass the cuCascade reservation system. Force bfd to keep them exported.
# Harmless for the single-DSO static vcpkg build.
set_target_properties(sirius_loadable_extension PROPERTIES LINKER_TYPE BFD)

# Shared configuration for both extension targets
foreach(_target sirius_extension sirius_loadable_extension)
  set_target_properties(
    ${_target}
    PROPERTIES CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
               CUDA_STANDARD 20
               CUDA_STANDARD_REQUIRED ON
               CUDA_SEPARABLE_COMPILATION ON
               CUDA_RESOLVE_DEVICE_SYMBOLS ON)

  # cuco's device APIs need nvcc's extended device lambda; cuco compiles its own
  # consumers (tests/benchmarks) with --expt-extended-lambda.
  target_compile_options(
    ${_target}
    PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--expt-extended-lambda>)

  if(VCPKG_BUILD)
    set_target_properties(${_target} PROPERTIES CUDA_RUNTIME_LIBRARY Static)
  endif()

  # In the vcpkg build, vcpkg include dir must be searched before DuckDB's
  # bundled fmt (which uses duckdb_fmt namespace, incompatible with spdlog).
  # NO_SYSTEM_FROM_IMPORTED prevents -isystem/-I collapse by GCC; BEFORE ensures
  # vcpkg includes precede DuckDB's in the search order.
  if(VCPKG_BUILD)
    set_target_properties(${_target} PROPERTIES NO_SYSTEM_FROM_IMPORTED ON)
    target_include_directories(${_target} BEFORE PRIVATE ${_VCPKG_INC})
  endif()

  target_include_directories(
    ${_target}
    PRIVATE
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/include>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/compression/simpatico_codegen/include>
      $<$<BOOL:${SIRIUS_LEGACY_INCLUDE_DIR}>:$<BUILD_INTERFACE:${SIRIUS_LEGACY_INCLUDE_DIR}>>
  )

  # Substrait->DuckDB reader headers (from_substrait.hpp) and its bundled
  # protobuf.
  target_include_directories(
    ${_target}
    PRIVATE ${SIRIUS_SUBSTRAIT_DIR}/src/include
            ${SIRIUS_SUBSTRAIT_DIR}/third_party
            ${SIRIUS_SUBSTRAIT_DIR}/third_party/substrait)

  if(SIRIUS_LEGACY_COMPILE_DEFINITIONS)
    target_compile_definitions(${_target}
                               PRIVATE ${SIRIUS_LEGACY_COMPILE_DEFINITIONS})
  endif()

  # cuCascade::cucascade_cudf holds the cudf-coupled representations and
  # converters Sirius uses; it transitively links the cudf-free core
  # (cuCascade::cucascade) and cudf::cudf.

  target_link_libraries(
    ${_target}
    cudf::cudf
    cuco::cuco
    rmm::rmm
    spdlog::spdlog
    cuCascade::cucascade
    cuCascade::cucascade_cudf
    yaml-cpp::yaml-cpp
    telemetry_bridge
    simpatico)

  add_dependencies(${_target} duckdb_static)
endforeach()

# Additional libraries only needed by the static extension
target_link_libraries(sirius_extension PkgConfig::NUMA PkgConfig::LIBURING
                      PkgConfig::CURL OpenSSL::Crypto absl::any_invocable)

target_link_libraries(sirius_loadable_extension PkgConfig::LIBURING
                      PkgConfig::CURL OpenSSL::Crypto)

# The sirius-sys + sirius Rust crates are built by cargo, not CMake (unlike the
# telemetry bridge above, which CMake drives via Corrosion). Their build.rs
# discovers the Sirius headers (repo + conda) and links the libsirius artifact
# this build produces under build/<preset>/; see rust/crates/sirius-sys.

# cucascade upstream PRs #126/#128/#130 moved cucascade to static-CUDA linkage
# and pulled the NVML *static stub* (libnvidia-ml.a) into libcucascade.a via the
# `CUDA::nvml_static` imported target. Upstream's
# `LINKER:--exclude-libs,libnvidia-ml` workaround (PR #130) keeps cucascade's
# own tests happy by hiding the bundled stub symbols from the dynamic export
# table (NVIDIA bug 6174166: libnvidia-ml.so dlsyms into the host process and
# recurses into the stub). For sirius's larger transitive graph, hiding the
# symbols segfaults at runtime — DuckDB/RMM call nvml directly before nvmlInit's
# jump-table patch fires — while leaving them visible deadlocks in stubSpinLock
# when the real driver loads.
#
# The robust fix for sirius: override CUDA::nvml_static's IMPORTED location to
# point at the *shared* stub (libnvidia-ml.so) in the same pixi env. The shared
# stub uses normal dynamic linking via ld.so, which resolves to the real
# libnvidia-ml.so.1 at runtime, matching the working Phase-24 configuration.
# This bypasses both failure modes (no stub symbols bundled, no dynamic-export
# conflict).
if(TARGET CUDA::nvml_static)
  get_target_property(_nvml_static_loc CUDA::nvml_static IMPORTED_LOCATION)
  if(_nvml_static_loc AND _nvml_static_loc MATCHES "libnvidia-ml\\.a$")
    string(REGEX REPLACE "libnvidia-ml\\.a$" "libnvidia-ml.so" _nvml_shared_loc
                         "${_nvml_static_loc}")
    if(EXISTS "${_nvml_shared_loc}")
      set_target_properties(
        CUDA::nvml_static
        PROPERTIES IMPORTED_LOCATION "${_nvml_shared_loc}"
                   IMPORTED_LOCATION_RELEASE "${_nvml_shared_loc}")
      message(
        STATUS
          "Sirius: redirected CUDA::nvml_static from ${_nvml_static_loc} to ${_nvml_shared_loc}"
      )
    endif()
  endif()
endif()

# Required by DuckDB's export set (duckdb_static -> sirius_extension ->
# cucascade_static / telemetry_bridge). cucascade upstream PR #126 split
# topology_discovery into its own static target, and PR #150 split the
# cudf-coupled code into cucascade_cudf_static — list both here so the export
# set covers the full dependency chain.
install(
  TARGETS sirius_extension cucascade_static cucascade_cudf_static
          cucascade_topology_discovery_static telemetry_bridge simpatico
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")

if(ENABLE_STREAM_CHECK)
  message(
    STATUS "Stream check library will be built (loaded dynamically at runtime)")
  add_subdirectory(utils/stream_check)
else()
  message(
    STATUS
      "Stream check library will not be built (runtime loading will gracefully fail)"
  )
endif()

# cmake-format: off
set(TEST_SOURCES
    test/cpp/compression/test_compression.cpp
    test/cpp/config/test_config.cpp
    test/cpp/config/test_context.cpp
    test/cpp/config/test_object_store_config.cpp
    test/cpp/data/test_convertible_data_batch.cpp
    test/cpp/data/test_data_repository_manager_registry.cpp
    test/cpp/data/test_convertible_gpu_pipeline_task.cpp
    test/cpp/downgrade/test_downgrade_disk.cpp
    test/cpp/downgrade/test_downgrade_executor.cpp
    test/cpp/downgrade/test_downgrade_lifecycle.cpp
    test/cpp/exec/test_bounded_thread_pool.cpp
    test/cpp/exec/test_exchange_channel.cpp
    test/cpp/exec/test_inspectable_mpsc.cpp
    test/cpp/exec/test_interruptible_mpmc.cpp
    test/cpp/exec/test_multi_index_priority_queue.cpp
    test/cpp/exec/test_semi_future.cpp
    test/cpp/expression/test_ast_aggregate.cpp
    test/cpp/expression/test_ast_clone.cpp
    test/cpp/expression/test_ast_substitute.cpp
    test/cpp/expression/test_ast_from_duckdb.cpp
    test/cpp/expression/test_ast_scaffold.cpp
    test/cpp/expression/test_ast_to_duckdb.cpp
    test/cpp/expression/test_comparison_type_mapping.cpp
    test/cpp/expression/test_function_id.cpp
    test/cpp/expression/test_value.cpp
    test/cpp/expression_evaluator/test_expression_evaluator.cpp
    test/cpp/expression_evaluator/test_expression_evaluator_ast_equivalence.cpp
    test/cpp/expression_evaluator/test_gpu_expression_translator.cpp
    test/cpp/expression_evaluator/test_nonduckdb_frontend_proof.cpp
    test/cpp/helper/test_cudf_utils.cpp
    test/cpp/helper/test_logical_type.cpp
    test/cpp/helper/test_transaction_utils.cpp
    test/cpp/integration/test_gpu_execution_aggregate_nulls.cpp
    test/cpp/integration/test_gpu_execution_allnull_scan.cpp
    test/cpp/integration/test_gpu_execution_array.cpp
    test/cpp/integration/test_gpu_execution_dynamic_filter_native.cpp
    test/cpp/integration/test_gpu_execution_filter_nulls.cpp
    test/cpp/integration/test_gpu_execution_join_nulls.cpp
    test/cpp/integration/test_gpu_execution_locality.cpp
    test/cpp/integration/test_gpu_execution_multi_format.cpp
    test/cpp/integration/test_gpu_execution_null_safe_join.cpp
    test/cpp/integration/test_gpu_execution_order_nulls.cpp
    test/cpp/integration/test_query_lifecycle_slot.cpp
    test/cpp/integration/test_gpu_execution_tpcds_nulls.cpp
    test/cpp/integration/test_gpu_execution_tpch.cpp
    test/cpp/integration/test_gpu_execution_tpch_mgpu_audit.cpp
    test/cpp/integration/test_pin_table_host_streaming.cpp
    test/cpp/integration/test_pin_table_merge_columns.cpp
    test/cpp/integration/test_pin_table_column_order.cpp
    test/cpp/integration/test_pin_table_mvcc_delete.cpp
    test/cpp/integration/test_pin_table_mvcc_foundation.cpp
    test/cpp/integration/test_pin_table_mvcc_insert.cpp
    test/cpp/integration/test_pin_table_zone_map_pruning.cpp
    test/cpp/integration/test_table_gpu_cache_warm_mgpu.cpp
    test/cpp/integration/test_transparent_runtime_fallback.cpp
    test/cpp/io/s3/test_sigv4.cpp
    test/cpp/io/s3/test_s3_default_visibility_guard.cpp
    test/cpp/io/s3/test_sirius_sigv4_authorizer.cpp
    test/cpp/io/s3/test_static_credentials.cpp
    test/cpp/io/test_parquet_helpers.cpp
    test/cpp/io/test_uri_parser.cpp
    test/cpp/io/uring/test_uring_readv.cpp
    test/cpp/io/rest/test_rest_reactor.cpp
    test/cpp/log/test_duckdb_sink.cpp
    test/cpp/log/test_logging.cpp
    test/cpp/scan_manager/test_s3_routing_cutover.cpp
    test/cpp/scan_manager/test_pin_table_multi_gpu.cpp
    test/cpp/scan_manager/test_cached_serving_hardening.cpp
    test/cpp/scan_manager/test_insert_delta_job.cpp
    test/cpp/scan_manager/test_mvcc_mask_job.cpp
    test/cpp/scan_manager/test_pinned_chunk_stats.cpp
    test/cpp/operator/test_physical_grouped_aggregate_merge_mgpu.cpp
    test/cpp/operator/test_physical_hash_join_mgpu.cpp
    test/cpp/operator/test_physical_order_mgpu.cpp
    test/cpp/operator/test_mgpu_stress.cpp
    test/cpp/operator/test_partition_memspace_mgpu.cpp
    test/cpp/memory/test_multiple_blocks_allocation_accessor.cpp
    test/cpp/memory/test_topology_index.cpp
    test/cpp/operator/test_build_probe_scheduling.cpp
    test/cpp/operator/test_cross_schedule.cpp
    test/cpp/operator/aggregate/test_gpu_merge_impl.cpp
    test/cpp/operator/aggregate/test_physical_grouped_aggregate.cpp
    test/cpp/operator/aggregate/test_physical_grouped_aggregate_count_distinct.cpp
    test/cpp/operator/aggregate/test_physical_grouped_aggregate_merge.cpp
    test/cpp/operator/test_crc32_partition_hash.cpp
    test/cpp/operator/test_gpu_partition_impl.cpp
    test/cpp/operator/test_host_table_chunk_reader.cpp
    test/cpp/operator/test_physical_concat.cpp
    test/cpp/operator/test_physical_sort_sample.cpp
    test/cpp/operator/test_physical_filter.cpp
    test/cpp/operator/test_physical_limit.cpp
    test/cpp/operator/test_physical_mark_join.cpp
    test/cpp/operator/test_physical_merge_sort.cpp
    test/cpp/operator/test_physical_order.cpp
    test/cpp/operator/test_physical_partition.cpp
    test/cpp/operator/test_physical_projection.cpp
    test/cpp/operator/test_physical_result_collector.cpp
    test/cpp/operator/test_physical_streaming_source.cpp
    test/cpp/operator/test_physical_table_scan.cpp
    test/cpp/operator/test_no_history_peak_memory_estimate.cpp
    test/cpp/operator/test_physical_top_n.cpp
    test/cpp/operator/test_physical_ungrouped_aggregate.cpp
    test/cpp/operator/test_sirius_dynamic_filter.cpp
    test/cpp/operator/test_dynamic_filter_publisher.cpp
    test/cpp/operator/test_sirius_dynamic_filter_mgpu.cpp
    test/cpp/parallel/test_task_executor.cpp
    test/cpp/planner/test_distinct_hash_join_detection.cpp
    test/cpp/planner/test_duckdb_join_filter_candidate_adapter.cpp
    test/cpp/planner/test_dynamic_filter_router.cpp
    test/cpp/planner/test_join_expression_key.cpp
    test/cpp/planner/test_plan_tree_shape.cpp
    test/cpp/planner/test_projection_fold.cpp
    test/cpp/planner/test_sirius_read_parquet_scan.cpp
    test/cpp/planner/test_query_id.cpp
    test/cpp/planner/test_query_index.cpp
    test/cpp/pipeline/test_batch_lock_utils.cpp
    test/cpp/pipeline/test_downgraded_task_prefetcher.cpp
    test/cpp/pipeline/test_gpu_pipeline_executor.cpp
    test/cpp/pipeline/test_oom_reschedule.cpp
    test/cpp/pipeline/test_pipeline_memory_history.cpp
    test/cpp/pipeline/test_gpu_pipeline_task_history.cpp
    test/cpp/pipeline/test_gpu_pipeline_disk_readback.cpp
    test/cpp/pipeline/test_get_next_ports_after_sink.cpp
    test/cpp/pipeline/test_merge_pipeline_fusion.cpp
    test/cpp/pipeline/test_partition_build_pipelines.cpp
    test/cpp/pipeline/test_pipeline_dynamic_filter_native_shape.cpp
    test/cpp/pipeline/test_pipeline_schedule_canonical.cpp
    test/cpp/pipeline/test_plan_printer.cpp
    test/cpp/pipeline/test_repository_wiring_materializer.cpp
    test/cpp/pipeline/test_task_scheduler.cpp
    test/cpp/scan/bench_decode_codecs.cpp
    test/cpp/scan/test_can_serve_with_columns.cpp
    test/cpp/scan/test_duckdb_block_layout.cpp
    test/cpp/scan/test_duckdb_insert_delta.cpp
    test/cpp/scan/test_duckdb_mvcc_visibility.cpp
    test/cpp/scan/test_duckdb_native_batch_coalescer.cpp
    test/cpp/scan/test_duckdb_native_dynamic_filter_remap.cpp
    test/cpp/scan/test_duckdb_native_host_backed_decode.cpp
    test/cpp/scan/test_duckdb_native_walker.cpp
    test/cpp/scan/test_dynamic_filter_merge.cpp
    test/cpp/scan/test_gpu_decode_alp.cpp
    test/cpp/scan/test_gpu_decode_bitpacking.cpp
    test/cpp/scan/test_gpu_decode_rle.cpp
    test/cpp/scan/test_gpu_decode_strings.cpp
    test/cpp/scan/test_host_keep_mask.cpp
    test/cpp/scan/test_gpu_native_decode.cpp
    test/cpp/scan/test_owning_table_view.cpp
    test/cpp/scan/test_parquet_schema_mapping.cpp
    test/cpp/scan/test_parquet_scan_sizing.cpp
    test/cpp/sirius_extension/test_sirius_read_parquet_cardinality.cpp
    test/cpp/sql/test_sirius_sql_rewrite.cpp
    test/cpp/telemetry/test_telemetry_context.cpp
    test/cpp/transparent/test_preserve_dynamic_filter_metadata.cpp
    test/cpp/unittest.cpp
    test/cpp/utils/pipeline_conversion_test_utils.cpp
    test/cpp/utils/sirius_test_env.cpp
    test/cpp/utils/test_gpu_execution_comparator.cpp
    test/cpp/utils/utils.cpp)
# cmake-format: on

if(SIRIUS_LEGACY_TEST_SOURCES)
  list(APPEND TEST_SOURCES ${SIRIUS_LEGACY_TEST_SOURCES})
endif()

# The testcontainers-backed S3 harness only compiles when its dependency is
# built; unittest.cpp guards its use behind SIRIUS_HAVE_TESTCONTAINERS.
if(SIRIUS_BUILD_S3_TESTS)
  list(
    APPEND
    TEST_SOURCES
    test/cpp/utils/s3_container.cpp
    test/cpp/io/s3/test_sirius_httpfs.cpp
    test/cpp/io/rest/test_rest_ioctx_integration.cpp
    test/cpp/scan_manager/test_describe_parquet_s3.cpp
    test/cpp/integration/test_s3_sql_surface.cpp
    test/cpp/integration/test_s3_tpch.cpp)
endif()

# Orphaned test files (present in the tree but missing from TEST_SOURCES) are
# caught by the check-orphan-tests pre-commit hook
# (scripts/check_orphan_tests.py) rather than a configure-time glob — see
# test/cpp/orphan_test_allowlist.txt.

add_executable(sirius_unittest ${TEST_SOURCES})

if(VCPKG_BUILD)
  set_target_properties(sirius_unittest PROPERTIES NO_SYSTEM_FROM_IMPORTED ON)
  target_include_directories(sirius_unittest BEFORE PRIVATE ${_VCPKG_INC})
endif()

target_include_directories(
  sirius_unittest
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/duckdb/third_party/catch>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/test/cpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<$<BOOL:${SIRIUS_LEGACY_INCLUDE_DIR}>:$<BUILD_INTERFACE:${SIRIUS_LEGACY_INCLUDE_DIR}>>
)

target_link_libraries(sirius_unittest sirius_extension duckdb_static ZLIB::ZLIB)
link_extension_libraries(sirius_unittest "")

# S3 container harness: the testcontainers-native bridge plus libcurl for
# host-side fixture upload (SigV4 signing comes from sirius_extension). Gated so
# offline/Go-less builds skip it; the harness calls in unittest.cpp are guarded
# by SIRIUS_HAVE_TESTCONTAINERS.
if(SIRIUS_BUILD_S3_TESTS)
  target_link_libraries(sirius_unittest testcontainers_native PkgConfig::CURL)
  target_compile_definitions(sirius_unittest
                             PRIVATE SIRIUS_HAVE_TESTCONTAINERS=1)
endif()
# DuckDB v1.5.0 unity builds emit strong symbols for static constexpr members
# that conflict with inline definitions from headers included in test files.
target_link_options(sirius_unittest PRIVATE
                    "LINKER:--allow-multiple-definition")

set_target_properties(
  sirius_unittest
  PROPERTIES CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test/cpp")

target_compile_definitions(
  sirius_unittest
  PRIVATE
    $<BUILD_INTERFACE:SIRIUS_DEFAULT_LOG_DIR="${CMAKE_BINARY_DIR}/log">
    $<BUILD_INTERFACE:SIRIUS_UNITTEST_LOG_DIR="${CMAKE_CURRENT_BINARY_DIR}/test/cpp/log">
    $<BUILD_INTERFACE:SIRIUS_PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}">)

if(SIRIUS_LEGACY_COMPILE_DEFINITIONS)
  target_compile_definitions(sirius_unittest
                             PRIVATE ${SIRIUS_LEGACY_COMPILE_DEFINITIONS})
endif()

# -----------------------------------------------------------------------------
# test/io/parquet_benchmark — standalone benchmark binary for sirius_datasource
# -----------------------------------------------------------------------------
add_executable(parquet_benchmark test/io/parquet_benchmark.cpp)

if(VCPKG_BUILD)
  set_target_properties(parquet_benchmark PROPERTIES NO_SYSTEM_FROM_IMPORTED ON)
  target_include_directories(parquet_benchmark BEFORE PRIVATE ${_VCPKG_INC})
endif()

target_include_directories(
  parquet_benchmark
  PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/include>)

target_link_libraries(
  parquet_benchmark
  sirius_extension
  duckdb_static
  cudf::cudf
  rmm::rmm
  spdlog::spdlog
  cuCascade::cucascade
  cuCascade::cucascade_cudf
  PkgConfig::LIBURING
  PkgConfig::NUMA)
link_extension_libraries(parquet_benchmark "")

target_link_options(parquet_benchmark PRIVATE
                    "LINKER:--allow-multiple-definition")

set_target_properties(
  parquet_benchmark
  PROPERTIES CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test/io")
