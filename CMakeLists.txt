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

# Dependencies
find_package(cudf REQUIRED CONFIG)
find_package(spdlog REQUIRED CONFIG)
find_package(libconfig++ REQUIRED CONFIG)
find_package(absl REQUIRED CONFIG)
find_package(PkgConfig REQUIRED)
pkg_check_modules(NUMA REQUIRED IMPORTED_TARGET numa)

# cuCascade - GPU Memory Reservation Library (submodule)
set(BUILD_TESTS
    OFF
    CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS
    OFF
    CACHE BOOL "" FORCE)
set(BUILD_STATIC_LIBS
    ON
    CACHE BOOL "" FORCE)
set(WARNINGS_AS_ERRORS
    OFF
    CACHE BOOL "" FORCE)
add_subdirectory(cucascade "${CMAKE_BINARY_DIR}/cucascade" EXCLUDE_FROM_ALL)

# cmake-format: off
set(EXTENSION_SOURCES
    src/config.cpp
    src/cpu_cache.cpp
    src/creator/task_creator.cpp
    src/data/host_parquet_representation.cpp
    src/data/host_parquet_representation_converters.cpp
    src/downgrade/downgrade_executor.cpp
    src/downgrade/downgrade_queue.cpp
    src/downgrade/downgrade_task.cpp
    src/expression_executor/gpu_expression_executor.cpp
    src/expression_executor/gpu_expression_executor_state.cpp
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
    src/extension_lock.cpp
    src/fallback.cpp
    src/gpu_buffer_manager.cpp
    src/gpu_columns.cpp
    src/gpu_context.cpp
    src/gpu_executor.cpp
    src/gpu_meta_pipeline.cpp
    src/gpu_physical_operator.cpp
    src/gpu_physical_plan_generator.cpp
    src/gpu_pipeline.cpp
    src/gpu_query_result.cpp
    src/memory/sirius_memory_reservation_manager.cpp
    src/op/aggregate/aggregate_op_util.cpp
    src/op/aggregate/gpu_aggregate_impl.cpp
    src/op/merge/gpu_merge_impl.cpp
    src/op/order/gpu_order_impl.cpp
    src/op/partition/gpu_partition_impl.cpp
    src/op/result/host_table_chunk_reader.cpp
    src/op/scan/cached_ranges.cpp
    src/op/scan/duckdb_scan_executor.cpp
    src/op/scan/duckdb_scan_task.cpp
    src/op/scan/parquet_scan_task.cpp
    src/op/scan/prefetched_data_source.cpp
    src/op/sirius_physical_column_data_scan.cpp
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
    src/operator/gpu_materialize.cpp
    src/operator/gpu_physical_column_data_scan.cpp
    src/operator/gpu_physical_concat.cpp
    src/operator/gpu_physical_cte.cpp
    src/operator/gpu_physical_delim_join.cpp
    src/operator/gpu_physical_dummy_scan.cpp
    src/operator/gpu_physical_empty_result.cpp
    src/operator/gpu_physical_filter.cpp
    src/operator/gpu_physical_grouped_aggregate.cpp
    src/operator/gpu_physical_hash_join.cpp
    src/operator/gpu_physical_limit.cpp
    src/operator/gpu_physical_nested_loop_join.cpp
    src/operator/gpu_physical_order.cpp
    src/operator/gpu_physical_partition.cpp
    src/operator/gpu_physical_projection.cpp
    src/operator/gpu_physical_result_collector.cpp
    src/operator/gpu_physical_strings_matching.cpp
    src/operator/gpu_physical_substring.cpp
    src/operator/gpu_physical_table_scan.cpp
    src/operator/gpu_physical_top_n.cpp
    src/operator/gpu_physical_ungrouped_aggregate.cpp
    src/parallel/task_executor.cpp
    src/pipeline/gpu_pipeline_executor.cpp
    src/pipeline/gpu_pipeline_queue.cpp
    src/pipeline/gpu_pipeline_task.cpp
    src/pipeline/pipeline_executor.cpp
    src/pipeline/pipeline_queue.cpp
    src/pipeline/sirius_meta_pipeline.cpp
    src/pipeline/sirius_pipeline.cpp
    src/pipeline/task_request.cpp
    src/plan/gpu_plan_aggregate.cpp
    src/plan/gpu_plan_column_data_get.cpp
    src/plan/gpu_plan_comparison_join.cpp
    src/plan/gpu_plan_cte.cpp
    src/plan/gpu_plan_delim_get.cpp
    src/plan/gpu_plan_delim_join.cpp
    src/plan/gpu_plan_dummy_scan.cpp
    src/plan/gpu_plan_empty_result.cpp
    src/plan/gpu_plan_expression_get.cpp
    src/plan/gpu_plan_filter.cpp
    src/plan/gpu_plan_get.cpp
    src/plan/gpu_plan_limit.cpp
    src/plan/gpu_plan_order.cpp
    src/plan/gpu_plan_projection.cpp
    src/plan/gpu_plan_recursive_cte.cpp
    src/plan/gpu_plan_top_n.cpp
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
    src/sirius_config.cpp
    src/sirius_context.cpp
    src/sirius_engine.cpp
    src/sirius_extension.cpp
    src/sirius_interface.cpp
    src/util/stream_check_wrapper.cpp
    src/util/segfault_backtrace_handler.cpp)

set(CUDA_SOURCES
    src/cuda/allocator.cu
    src/cuda/communication.cu
    src/cuda/cudf/cudf_aggregate.cu
    src/cuda/cudf/cudf_duplicate_elimination.cu
    src/cuda/cudf/cudf_groupby.cu
    src/cuda/cudf/cudf_join.cu
    src/cuda/cudf/cudf_orderby.cu
    src/cuda/cudf/cudf_utils.cu
    src/cuda/expression_executor/gpu_dispatch_materialize.cu
    src/cuda/expression_executor/gpu_dispatch_select.cu
    src/cuda/expression_executor/gpu_dispatch_string.cu
    src/cuda/operator/arbitrary_expression.cu
    src/cuda/operator/comparison_expression.cu
    src/cuda/operator/hash_join_inner.cu
    src/cuda/operator/hash_join_right.cu
    src/cuda/operator/hash_join_single.cu
    src/cuda/operator/materialize.cu
    src/cuda/operator/nested_loop_join.cu
    src/cuda/operator/strings_matching.cu
    src/cuda/operator/substring.cu
    src/cuda/print.cu
    src/cuda/utils.cu)
# cmake-format: on

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

  target_include_directories(
    ${_target}
    PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/include>)

  target_compile_definitions(
    ${_target}
    PRIVATE $<BUILD_INTERFACE:SIRIUS_DEFAULT_LOG_DIR="${CMAKE_BINARY_DIR}/log">)

  target_link_libraries(${_target} cudf::cudf rmm::rmm spdlog::spdlog
                        cuCascade::cucascade)

  add_dependencies(${_target} duckdb_static)
endforeach()

# Precompiled headers for extension targets
target_precompile_headers(
  sirius_extension
  PRIVATE
  # STL headers (60+ combined inclusions)
  <algorithm>
  <memory>
  <optional>
  <string>
  <string_view>
  <vector>
  # Logging (44/139 source files)
  "src/include/log/logging.hpp"
  # DuckDB planner (30+ files)
  <duckdb/planner/expression/bound_reference_expression.hpp>)
target_precompile_headers(sirius_loadable_extension REUSE_FROM sirius_extension)

# CMake PCH doesn't apply to nvcc — skip PCH for CUDA sources
set_source_files_properties(${CUDA_SOURCES} PROPERTIES SKIP_PRECOMPILE_HEADERS
                                                       ON)

# Additional libraries only needed by the static extension
target_link_libraries(sirius_extension PkgConfig::NUMA ${LIBCONFIG++_LIBRARIES}
                      absl::any_invocable)

# Required by DuckDB's export set (duckdb_static -> sirius_extension ->
# cucascade_static)
install(
  TARGETS sirius_extension cucascade_static
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
    test/cpp/data/test_host_parquet_representation.cpp
    test/cpp/downgrade/test_downgrade_executor.cpp
    test/cpp/exec/test_interruptible_mpmc.cpp
    test/cpp/exec/test_kiosk.cpp
    test/cpp/expression_executor/test_gpu_expression_executor.cpp
    test/cpp/expression_executor/test_gpu_expression_translator.cpp
    test/cpp/integration/test_gpu_execution_tpch.cpp
    test/cpp/memory/test_host_table_utils.cpp
    test/cpp/memory/test_multiple_blocks_allocation_accessor.cpp
    test/cpp/memory_management/test_cpu_cache.cpp
    test/cpp/operator/aggregate/test_gpu_merge_impl.cpp
    test/cpp/operator/aggregate/test_physical_grouped_aggregate.cpp
    test/cpp/operator/aggregate/test_physical_grouped_aggregate_count_distinct.cpp
    test/cpp/operator/aggregate/test_physical_grouped_aggregate_merge.cpp
    test/cpp/operator/test_gpu_partition_impl.cpp
    test/cpp/operator/test_host_table_chunk_reader.cpp
    test/cpp/operator/test_physical_concat.cpp
    test/cpp/operator/test_physical_filter.cpp
    test/cpp/operator/test_physical_limit.cpp
    test/cpp/operator/test_physical_mark_join.cpp
    test/cpp/operator/test_physical_merge_sort.cpp
    test/cpp/operator/test_physical_order.cpp
    test/cpp/operator/test_physical_partition.cpp
    test/cpp/operator/test_physical_projection.cpp
    test/cpp/operator/test_physical_result_collector.cpp
    test/cpp/operator/test_physical_table_scan.cpp
    test/cpp/operator/test_physical_top_n.cpp
    test/cpp/operator/test_physical_ungrouped_aggregate.cpp
    test/cpp/parallel/test_task_executor.cpp
    test/cpp/pipeline/test_gpu_pipeline_executor.cpp
    test/cpp/pipeline/test_oom_reschedule.cpp
    test/cpp/pipeline/test_pipeline_executor.cpp
    test/cpp/scan/test_column_builder.cpp
    test/cpp/scan/test_parquet_scan_task.cpp
    test/cpp/scan/test_scan_executor.cpp
    test/cpp/unittest.cpp
    test/cpp/utils/sirius_test_env.cpp
    test/cpp/utils/utils.cpp)
# cmake-format: on

add_executable(sirius_unittest ${TEST_SOURCES})

target_include_directories(
  sirius_unittest
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/duckdb/third_party/catch>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/test/cpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/include>)

target_link_libraries(sirius_unittest sirius_extension)

# Precompile Catch2 header (18K lines) once instead of in every test file
target_precompile_headers(sirius_unittest PRIVATE <catch.hpp>)
# unittest.cpp defines CATCH_CONFIG_RUNNER before including catch.hpp, which is
# incompatible with the PCH (compiled without that define). Exclude it.
set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/test/cpp/unittest.cpp"
                            PROPERTIES SKIP_PRECOMPILE_HEADERS ON)

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
