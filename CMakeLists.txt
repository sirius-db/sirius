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

# Dependencies
find_package(cudf REQUIRED CONFIG)
find_package(spdlog REQUIRED CONFIG)
find_package(yaml-cpp REQUIRED CONFIG)
find_package(absl REQUIRED CONFIG)
find_package(PkgConfig REQUIRED)
find_package(OpenSSL REQUIRED)
pkg_check_modules(NUMA REQUIRED IMPORTED_TARGET numa)
pkg_check_modules(LIBURING REQUIRED IMPORTED_TARGET liburing)
pkg_check_modules(CURL REQUIRED IMPORTED_TARGET libcurl)

# cuCascade - GPU Memory Reservation Library (submodule) cucascade links `numa`
# as a bare library name (-lnuma). In the vcpkg build the static libnuma.a lives
# in the vcpkg installed lib dir, which isn't on the default linker search path.
# Create an imported target so CMake resolves the bare name to the full path
# instead of emitting -lnuma.
if(VCPKG_BUILD)
  add_library(numa STATIC IMPORTED)
  set_target_properties(
    numa
    PROPERTIES IMPORTED_LOCATION
               "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/lib/libnuma.a")
endif()
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
set(CUCASCADE_WARNINGS_AS_ERRORS
    OFF
    CACHE BOOL "" FORCE)
add_subdirectory(cucascade "${CMAKE_BINARY_DIR}/cucascade" EXCLUDE_FROM_ALL)

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

  foreach(_target cucascade_objects cucascade_static cucascade_shared)
    if(TARGET "${_target}")
      sirius_prefer_static_cudart("${_target}")
    endif()
  endforeach()
endif()

# Rust telemetry instrumentation (C++ FFI via Corrosion)
add_subdirectory(rust/crates/telemetry/bridge)

# cmake-format: off
set(EXTENSION_SOURCES
    src/config.cpp
    src/helper/type_conversions.cpp
    src/debug_utils.cpp
    src/creator/task_creator.cpp
    src/data/host_parquet_representation.cpp
    src/data/host_parquet_representation_converters.cpp
    src/downgrade/downgrade_executor.cpp
    src/expression/aggregate_id.cpp
    src/expression/ast/clone.cpp
    src/expression/ast/node.cpp
    src/expression/from_duckdb.cpp
    src/expression/function_id.cpp
    src/expression/join_condition.cpp
    src/expression/to_duckdb.cpp
    src/expression/value.cpp
    src/expression_executor/expression_executor_strategy.cpp
    src/expression_executor/ast_op_counter.cpp
    src/expression_executor/gpu_expression_executor.cpp
    src/expression_executor/gpu_expression_translator.cpp
    src/expression_executor/regex/regex_playground.cpp
    src/expression_executor/specializations/gpu_execute_between.cpp
    src/expression_executor/specializations/gpu_execute_case.cpp
    src/expression_executor/specializations/gpu_execute_cast.cpp
    src/expression_executor/specializations/gpu_execute_comparison.cpp
    src/expression_executor/specializations/gpu_execute_conjunction.cpp
    src/expression_executor/specializations/gpu_execute_constant.cpp
    src/expression_executor/specializations/gpu_execute_function.cpp
    src/expression_executor/specializations/gpu_execute_operator.cpp
    src/expression_executor/specializations/gpu_execute_reference.cpp
    # I/O adapters (Phase 19 — io_uring + sirius_datasource)
    src/io/admission_control.cpp
    src/io/datasource_factory.cpp
    src/io/duckdb_io_object.cpp
    src/io/io_context.cpp
    src/io/parquet_helpers.cpp
    src/io/prefetching_cache.cpp
    src/io/s3/s3_blocking_ioctx.cpp
    src/io/s3/s3_reactor.cpp
    src/io/s3/s3_ioctx.cpp
    src/io/s3/sigv4.cpp
    src/io/s3/sirius_sigv4_authorizer.cpp
    src/io/sirius_datasource.cpp
    src/io/uri_parser.cpp
    src/io/uring/uring_ioctx.cpp
    src/io/uring/uring_reactor.cpp
    src/memory/defragmenter_oom_policy.cpp
    src/memory/sirius_memory_reservation_manager.cpp
    src/op/aggregate/aggregate_op_util.cpp
    src/op/aggregate/gpu_aggregate_impl.cpp
    src/op/merge/gpu_merge_impl.cpp
    src/op/order/gpu_order_impl.cpp
    src/op/partition/gpu_partition_impl.cpp
    src/op/result/host_table_chunk_reader.cpp
    src/op/scan/cached_ranges.cpp
    src/op/scan/duckdb_scan_executor.cpp
    src/op/scan/cpu_source_task.cpp
    src/op/scan/duckdb_native_metadata.cpp
    src/op/scan/duckdb_native_decoder.cpp
    src/op/scan/duckdb_scan_task.cpp
    src/op/scan/equality_delete_filter.cpp
    src/op/scan/hive_partition.cpp
    src/op/scan/iceberg_avro_reader.cpp
    src/op/scan/iceberg_delete_pipeline.cpp
    src/op/scan/iceberg_metadata_reader.cpp
    src/op/scan/iceberg_scan_task.cpp
    src/op/scan/positional_delete_filter.cpp
    src/op/scan/puffin_reader.cpp
    src/op/scan/duckdb_native_scan_info.cpp
    src/op/scan/parquet_scan_info.cpp
    src/op/scan/parquet_scan_task.cpp
    src/op/scan/parquet_schema_mapping.cpp
    src/op/scan/prefetched_data_source.cpp
    src/op/scan/scan_plan.cpp
    src/op/scan/scan_utils.cpp
    src/op/scan/sirius_gpu_duckdb_native_scan_operator.cpp
    src/op/scan/sirius_gpu_parquet_scan_operator.cpp
    src/op/sirius_dynamic_filter.cpp
    src/op/sirius_physical_column_data_scan.cpp
    src/op/sirius_physical_cpu_source.cpp
    src/op/sirius_physical_concat.cpp
    src/op/sirius_physical_cte.cpp
    src/op/sirius_physical_delim_join.cpp
    src/op/sirius_physical_duckdb_scan.cpp
    src/op/sirius_physical_dummy_scan.cpp
    src/op/sirius_physical_empty_result.cpp
    src/op/sirius_physical_filter.cpp
    src/op/sirius_physical_grouped_aggregate.cpp
    src/op/sirius_physical_grouped_aggregate_merge.cpp
    src/op/sirius_physical_hash_join.cpp
    src/op/sirius_physical_limit.cpp
    src/op/sirius_physical_merge_sort.cpp
    src/op/sirius_physical_nested_loop_join.cpp
    src/op/sirius_physical_operator.cpp
    src/op/sirius_physical_operator_type.cpp
    src/op/sirius_physical_order.cpp
    src/op/sirius_physical_iceberg_scan.cpp
    src/op/sirius_physical_parquet_scan.cpp
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
    src/pipeline/gpu_pipeline_executor.cpp
    src/pipeline/gpu_pipeline_task.cpp
    src/pipeline/task_scheduler.cpp
    src/pipeline/sirius_meta_pipeline.cpp
    src/pipeline/sirius_pipeline.cpp
    src/pipeline/sirius_pipeline_converter.cpp
    src/pipeline/repository_wiring_materializer.cpp
    src/pipeline/sirius_plan_printer.cpp
    src/pipeline/task_request.cpp
    src/planner/query.cpp
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
    src/planner/sirius_plan_recursive_cte.cpp
    src/planner/sirius_plan_top_n.cpp
    src/scan_manager/cached_split_provider.cpp
    src/scan_manager/duckdb_native_split_provider.cpp
    src/pin_table.cpp
    src/scan_manager/parquet_split_provider.cpp
    src/scan_manager/sirius_scan_manager.cpp
    src/scan_manager/split_connector.cpp
    src/scan_manager/split_provider.cpp
    src/sirius_config.cpp
    src/sirius_context.cpp
    src/sirius_engine.cpp
    src/sirius_extension.cpp
    src/sirius_interface.cpp
    src/sirius_sql_rewrite.cpp
    src/telemetry/telemetry_context.cpp
    src/transparent/physical_sirius_execution.cpp
    src/transparent/sirius_optimizer_extension.cpp
    src/util/stream_check_wrapper.cpp
    src/util/segfault_backtrace_handler.cpp)
# cmake-format: on

# Legacy Sirius — all legacy-specific config lives in src/legacy/CMakeLists.txt.
# Deleting src/legacy/ cleanly removes legacy build support.
set(CUDA_SOURCES
    src/op/scan/equality_delete_mask.cu src/cuda/scan/gpu_decode_alp.cu
    src/cuda/scan/gpu_decode_bitpacking.cu src/cuda/scan/gpu_decode_rle.cu
    src/cuda/scan/gpu_decode_strings.cu src/cuda/scan/gpu_native_decode.cu)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/legacy/CMakeLists.txt")
  add_subdirectory(src/legacy)
  list(APPEND EXTENSION_SOURCES ${SIRIUS_LEGACY_SOURCES}
       ${SIRIUS_LEGACY_CONDITIONAL_SOURCES})
  list(APPEND CUDA_SOURCES ${SIRIUS_LEGACY_CUDA_SOURCES})
endif()

build_static_extension(sirius ${EXTENSION_SOURCES} ${CUDA_SOURCES})
build_loadable_extension(sirius CPP ${EXTENSION_SOURCES} ${CUDA_SOURCES})

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
      $<$<BOOL:${SIRIUS_LEGACY_INCLUDE_DIR}>:$<BUILD_INTERFACE:${SIRIUS_LEGACY_INCLUDE_DIR}>>
  )

  if(SIRIUS_LEGACY_COMPILE_DEFINITIONS)
    target_compile_definitions(${_target}
                               PRIVATE ${SIRIUS_LEGACY_COMPILE_DEFINITIONS})
  endif()

  target_link_libraries(
    ${_target}
    cudf::cudf
    rmm::rmm
    spdlog::spdlog
    cuCascade::cucascade
    yaml-cpp::yaml-cpp
    telemetry_bridge)

  add_dependencies(${_target} duckdb_static)
endforeach()

# Additional libraries only needed by the static extension
target_link_libraries(
  sirius_extension
  PkgConfig::NUMA
  PkgConfig::LIBURING
  PkgConfig::CURL
  OpenSSL::Crypto
  yaml-cpp::yaml-cpp
  absl::any_invocable)

target_link_libraries(sirius_loadable_extension PkgConfig::LIBURING
                      PkgConfig::CURL OpenSSL::Crypto)

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
# topology_discovery into its own static target — list it here so the export set
# covers the dependency chain.
install(
  TARGETS sirius_extension cucascade_static cucascade_topology_discovery_static
          telemetry_bridge
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
    test/cpp/config/test_config.cpp
    test/cpp/config/test_context.cpp
    test/cpp/config/test_object_store_config.cpp
    test/cpp/data/test_convertible_data_batch.cpp
    test/cpp/data/test_convertible_gpu_pipeline_task.cpp
    test/cpp/data/test_host_parquet_representation.cpp
    test/cpp/downgrade/test_downgrade_disk.cpp
    test/cpp/downgrade/test_downgrade_executor.cpp
    test/cpp/downgrade/test_downgrade_lifecycle.cpp
    test/cpp/exec/test_bounded_thread_pool.cpp
    test/cpp/exec/test_inspectable_mpsc.cpp
    test/cpp/exec/test_interruptible_mpmc.cpp
    test/cpp/expression/test_ast_aggregate.cpp
    test/cpp/expression/test_ast_clone.cpp
    test/cpp/expression/test_ast_from_duckdb.cpp
    test/cpp/expression/test_ast_scaffold.cpp
    test/cpp/expression/test_ast_to_duckdb.cpp
    test/cpp/expression/test_comparison_type_mapping.cpp
    test/cpp/expression/test_function_id.cpp
    test/cpp/expression/test_value.cpp
    test/cpp/expression_executor/test_gpu_expression_executor.cpp
    test/cpp/expression_executor/test_gpu_expression_executor_ast_equivalence.cpp
    test/cpp/expression_executor/test_gpu_expression_translator.cpp
    test/cpp/expression_executor/test_nonduckdb_frontend_proof.cpp
    test/cpp/helper/test_logical_type.cpp
    test/cpp/helper/test_transaction_utils.cpp
    test/cpp/integration/test_gpu_execution_locality.cpp
    test/cpp/integration/test_gpu_execution_multi_format.cpp
    test/cpp/integration/test_gpu_execution_order_nulls.cpp
    test/cpp/integration/test_gpu_execution_tpch.cpp
    test/cpp/integration/test_gpu_execution_tpch_mgpu_audit.cpp
    test/cpp/integration/test_table_gpu_cache_warm_mgpu.cpp
    test/cpp/integration/test_transparent_execution.cpp
    test/cpp/io/s3/test_sigv4.cpp
    test/cpp/io/s3/test_s3_default_visibility_guard.cpp
    test/cpp/io/s3/test_s3_blocking_ioctx.cpp
    test/cpp/io/s3/test_s3_ioctx.cpp
    test/cpp/io/s3/test_sirius_sigv4_authorizer.cpp
    test/cpp/io/s3/test_static_credentials.cpp
    test/cpp/io/test_datasource_factory.cpp
    test/cpp/io/test_duckdb_io_object.cpp
    test/cpp/io/test_parquet_helpers.cpp
    test/cpp/io/test_prefetching_cache_telemetry.cpp
    test/cpp/io/test_uri_parser.cpp
    test/cpp/integration/test_s3_sql_surface.cpp
    test/cpp/integration/test_s3_perf_benchmark.cpp
    test/cpp/integration/test_scan_manager_s3_end_to_end.cpp
    test/cpp/operator/test_physical_grouped_aggregate_merge_mgpu.cpp
    test/cpp/operator/test_physical_hash_join_mgpu.cpp
    test/cpp/operator/test_physical_order_mgpu.cpp
    test/cpp/operator/test_mgpu_stress.cpp
    test/cpp/memory/test_host_table_utils.cpp
    test/cpp/memory/test_multiple_blocks_allocation_accessor.cpp
    test/cpp/operator/aggregate/test_gpu_merge_impl.cpp
    test/cpp/operator/aggregate/test_physical_grouped_aggregate.cpp
    test/cpp/operator/aggregate/test_physical_grouped_aggregate_count_distinct.cpp
    test/cpp/operator/aggregate/test_physical_grouped_aggregate_merge.cpp
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
    test/cpp/operator/test_physical_table_scan.cpp
    test/cpp/operator/test_no_history_peak_memory_estimate.cpp
    test/cpp/operator/test_physical_top_n.cpp
    test/cpp/operator/test_physical_ungrouped_aggregate.cpp
    test/cpp/operator/test_sirius_dynamic_filter.cpp
    test/cpp/parallel/test_task_executor.cpp
    test/cpp/planner/test_distinct_hash_join_detection.cpp
    test/cpp/pipeline/test_gpu_pipeline_executor.cpp
    test/cpp/pipeline/test_pipeline_merge_fusion.cpp
    test/cpp/pipeline/test_oom_reschedule.cpp
    test/cpp/pipeline/test_sirius_read_parquet_converter.cpp
    test/cpp/pipeline/test_pipeline_memory_history.cpp
    test/cpp/pipeline/test_gpu_pipeline_task_history.cpp
    test/cpp/pipeline/test_gpu_pipeline_disk_readback.cpp
    test/cpp/pipeline/test_get_next_ports_after_sink.cpp
    test/cpp/pipeline/test_plan_printer.cpp
    test/cpp/pipeline/test_repository_wiring_materializer.cpp
    test/cpp/pipeline/test_task_scheduler.cpp
    test/cpp/scan/bench_decode_codecs.cpp
    test/cpp/scan/test_column_builder.cpp
    test/cpp/scan/test_duckdb_block_layout.cpp
    test/cpp/scan/test_duckdb_native_split_provider.cpp
    test/cpp/scan/test_duckdb_native_walker.cpp
    test/cpp/scan/test_gpu_decode_alp.cpp
    test/cpp/scan/test_gpu_decode_bitpacking.cpp
    test/cpp/scan/test_gpu_decode_rle.cpp
    test/cpp/scan/test_gpu_decode_strings.cpp
    test/cpp/scan/test_gpu_native_decode.cpp
    test/cpp/scan/test_parquet_schema_mapping.cpp
    test/cpp/scan/test_parquet_scan_task.cpp
    test/cpp/scan/test_parquet_split_provider.cpp
    test/cpp/scan/test_parquet_split_provider_s3.cpp
    test/cpp/scan/test_partition_inject.cpp
    test/cpp/scan/test_scan_executor.cpp
    test/cpp/scan/test_split_connector.cpp
    test/cpp/scan_manager/test_sirius_scan_manager_s3.cpp
    test/cpp/scan_manager/test_describe_parquet_s3.cpp
    test/cpp/scan_manager/test_pin_table_multi_gpu.cpp
    test/cpp/sirius_extension/test_sirius_read_parquet_cardinality.cpp
    test/cpp/sql/test_sirius_sql_rewrite.cpp
    test/cpp/unittest.cpp
    test/cpp/utils/sirius_test_env.cpp
    test/cpp/utils/utils.cpp)
# cmake-format: on

if(SIRIUS_LEGACY_TEST_SOURCES)
  list(APPEND TEST_SOURCES ${SIRIUS_LEGACY_TEST_SOURCES})
endif()

add_executable(sirius_unittest ${TEST_SOURCES})

# TPC-DS integration test fixture, generated at build time (opt-in target). The
# database is not committed; run `cmake --build . --target tpcds-fixture` once
# before the TPC-DS integration tests. Requires the duckdb CLI (or python3 with
# the duckdb module) on PATH; both are provided by the pixi environment.
set(TPCDS_FIXTURE_DIR ${CMAKE_BINARY_DIR}/test-fixtures)
set(TPCDS_FIXTURE ${TPCDS_FIXTURE_DIR}/tpcds.duckdb)
set(TPCDS_FIXTURE_SCRIPT
    ${CMAKE_CURRENT_SOURCE_DIR}/test/cpp/integration/data/duckdb/generate_tpcds_duckdb.sh
)
add_custom_command(
  OUTPUT ${TPCDS_FIXTURE}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${TPCDS_FIXTURE_DIR}
  COMMAND bash ${TPCDS_FIXTURE_SCRIPT} ${TPCDS_FIXTURE}
  DEPENDS ${TPCDS_FIXTURE_SCRIPT}
  COMMENT "Generating TPC-DS test fixture (SF 0.01) at ${TPCDS_FIXTURE}"
  VERBATIM)
add_custom_target(tpcds-fixture DEPENDS ${TPCDS_FIXTURE})

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
    $<$<BOOL:${SIRIUS_LEGACY_INCLUDE_DIR}>:$<BUILD_INTERFACE:${SIRIUS_LEGACY_INCLUDE_DIR}>>
)

target_link_libraries(sirius_unittest sirius_extension duckdb_static)
link_extension_libraries(sirius_unittest "")
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
