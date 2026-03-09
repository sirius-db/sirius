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

# Language standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Optional features
option(ENABLE_STREAM_CHECK
       "Enable stream check library for debugging default stream usage" OFF)

# In the vcpkg build, vcpkg headers must take priority over:
#   1) conda's CCCL 3.1.4 (injected via CXXFLAGS as -I.../include/cccl)
#   2) DuckDB's bundled fmt 6.1.2 (uses duckdb_fmt namespace, incompatible with spdlog)
# Fix: strip conda CCCL path; use NO_SYSTEM_FROM_IMPORTED + BEFORE include
# (applied per-target below) to place vcpkg includes first.
if(VCPKG_BUILD)
  set(_VCPKG_INC "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/include")
  string(REGEX REPLACE "-I[^ ]*/include/cccl" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REGEX REPLACE "-I[^ ]*/include/cccl" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")

  # Strip conda-injected -Wl,-rpath flags from linker flags and disable
  # CMake's RPATH mechanism so binaries don't carry a hardcoded path to the
  # pixi/conda env. All deps are statically linked, so no RPATH is needed.
  # FORCE is required because DuckDB's CMakeLists.txt sets this cache var first.
  set(CMAKE_SKIP_RPATH ON CACHE BOOL "" FORCE)
  foreach(_var CMAKE_SHARED_LINKER_FLAGS CMAKE_EXE_LINKER_FLAGS
               CMAKE_MODULE_LINKER_FLAGS)
    string(REGEX REPLACE "-Wl,-rpath-link[=,][^ ]*" "" ${_var} "${${_var}}")
    string(REGEX REPLACE "-Wl,-rpath[=,][^ ]*" "" ${_var} "${${_var}}")
    set(${_var} "${${_var}}" CACHE STRING "" FORCE)
  endforeach()
endif()

# Dependencies
find_package(cudf REQUIRED CONFIG)
find_package(spdlog REQUIRED CONFIG)
find_package(libconfig CONFIG QUIET)
if(NOT libconfig_FOUND)
  find_package(libconfig++ REQUIRED)
endif()
find_package(absl REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(NUMA REQUIRED IMPORTED_TARGET numa)

# cuCascade - GPU Memory Reservation Library (submodule)
# cucascade links `numa` as a bare library name (-lnuma). In the vcpkg build
# the static libnuma.a lives in the vcpkg installed lib dir, which isn't on the
# default linker search path. Create an imported target so CMake resolves the
# bare name to the full path instead of emitting -lnuma.
if(VCPKG_BUILD)
  add_library(numa STATIC IMPORTED)
  set_target_properties(
    numa PROPERTIES IMPORTED_LOCATION
                    "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/lib/libnuma.a")
endif()
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
build_loadable_extension(sirius " " ${EXTENSION_SOURCES} ${CUDA_SOURCES})

# Shared configuration for both extension targets
foreach(_target sirius_extension sirius_loadable_extension)
  set_target_properties(${_target} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON
                                              LINKER_LANGUAGE CUDA)

  # In the vcpkg build, vcpkg include dir must be searched before DuckDB's
  # bundled fmt (which uses duckdb_fmt namespace, incompatible with spdlog).
  # NO_SYSTEM_FROM_IMPORTED prevents -isystem/-I collapse by GCC;
  # BEFORE ensures vcpkg includes precede DuckDB's in the search order.
  if(VCPKG_BUILD)
    set_target_properties(${_target} PROPERTIES NO_SYSTEM_FROM_IMPORTED ON)
    target_include_directories(${_target} BEFORE PRIVATE ${_VCPKG_INC})
  endif()

  target_include_directories(
    ${_target}
    PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/include>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/include/operator>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/include/cudf>)

  target_compile_definitions(
    ${_target}
    PRIVATE $<BUILD_INTERFACE:SIRIUS_DEFAULT_LOG_DIR="${CMAKE_BINARY_DIR}/log">)

  target_link_libraries(${_target} cudf::cudf rmm::rmm spdlog::spdlog
                        cuCascade::cucascade)

  add_dependencies(${_target} duckdb_static)
endforeach()

# The conda-forge GCC wrapper hard-codes -rpath <conda_prefix>/lib into every
# link invocation. Strip it from the loadable extension so the binary only
# depends on system libraries at runtime.
if(VCPKG_BUILD)
  find_program(_PATCHELF patchelf)
  if(_PATCHELF)
    add_custom_command(
      TARGET sirius_loadable_extension
      POST_BUILD
      COMMAND ${_PATCHELF} --remove-rpath $<TARGET_FILE:sirius_loadable_extension>
      COMMENT "Stripping RPATH from sirius loadable extension")
  endif()
endif()

# Additional libraries only needed by the static extension
if(TARGET libconfig::config++)
  set(_LIBCONFIG_TARGET libconfig::config++)
else()
  set(_LIBCONFIG_TARGET ${LIBCONFIG++_LIBRARIES})
endif()
target_link_libraries(sirius_extension PkgConfig::NUMA ${_LIBCONFIG_TARGET}
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

file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/test/cpp/*.cpp")
list(FILTER TEST_SOURCES EXCLUDE REGEX ".*/test_modified_pipeline\\.cpp$")
list(FILTER TEST_SOURCES EXCLUDE REGEX ".*/test_task_creator\\.cpp$")

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
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/include/operator>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/include/cudf>)

target_link_libraries(sirius_unittest sirius_extension)

set_target_properties(
  sirius_unittest PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                             "${CMAKE_CURRENT_BINARY_DIR}/test/cpp")

target_compile_definitions(
  sirius_unittest
  PRIVATE
    $<BUILD_INTERFACE:SIRIUS_DEFAULT_LOG_DIR="${CMAKE_BINARY_DIR}/log">
    $<BUILD_INTERFACE:SIRIUS_UNITTEST_LOG_DIR="${CMAKE_CURRENT_BINARY_DIR}/test/cpp/log">
    $<BUILD_INTERFACE:SIRIUS_PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}">)

enable_testing()
add_test(NAME sirius COMMAND sirius_unittest)
