# =============================================================================
# Legacy Sirius code path (gpu_processing table function).
#
# Deleting this directory removes all legacy build support. The main
# CMakeLists.txt detects its presence via EXISTS and skips it when absent.
# =============================================================================

option(ENABLE_LEGACY_SIRIUS
       "Enable legacy Sirius code path (gpu_processing table function)" OFF)

# Sources that are always compiled (shared types used by expression executor
# and CUDA kernels: GPUColumn, GPUPipeline, GPUExecutor, operators, etc.)
# cmake-format: off
set(SIRIUS_LEGACY_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/gpu_columns.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gpu_executor.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gpu_meta_pipeline.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gpu_physical_operator.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gpu_pipeline.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gpu_query_result.cpp
    ${CMAKE_CURRENT_LIST_DIR}/operator/gpu_materialize.cpp
    ${CMAKE_CURRENT_LIST_DIR}/operator/gpu_physical_column_data_scan.cpp
    ${CMAKE_CURRENT_LIST_DIR}/operator/gpu_physical_concat.cpp
    ${CMAKE_CURRENT_LIST_DIR}/operator/gpu_physical_cte.cpp
    ${CMAKE_CURRENT_LIST_DIR}/operator/gpu_physical_delim_join.cpp
    ${CMAKE_CURRENT_LIST_DIR}/operator/gpu_physical_dummy_scan.cpp
    ${CMAKE_CURRENT_LIST_DIR}/operator/gpu_physical_empty_result.cpp
    ${CMAKE_CURRENT_LIST_DIR}/operator/gpu_physical_filter.cpp
    ${CMAKE_CURRENT_LIST_DIR}/operator/gpu_physical_grouped_aggregate.cpp
    ${CMAKE_CURRENT_LIST_DIR}/operator/gpu_physical_hash_join.cpp
    ${CMAKE_CURRENT_LIST_DIR}/operator/gpu_physical_limit.cpp
    ${CMAKE_CURRENT_LIST_DIR}/operator/gpu_physical_nested_loop_join.cpp
    ${CMAKE_CURRENT_LIST_DIR}/operator/gpu_physical_order.cpp
    ${CMAKE_CURRENT_LIST_DIR}/operator/gpu_physical_partition.cpp
    ${CMAKE_CURRENT_LIST_DIR}/operator/gpu_physical_projection.cpp
    ${CMAKE_CURRENT_LIST_DIR}/operator/gpu_physical_result_collector.cpp
    ${CMAKE_CURRENT_LIST_DIR}/operator/gpu_physical_strings_matching.cpp
    ${CMAKE_CURRENT_LIST_DIR}/operator/gpu_physical_substring.cpp
    ${CMAKE_CURRENT_LIST_DIR}/operator/gpu_physical_table_scan.cpp
    ${CMAKE_CURRENT_LIST_DIR}/operator/gpu_physical_top_n.cpp
    ${CMAKE_CURRENT_LIST_DIR}/operator/gpu_physical_ungrouped_aggregate.cpp
    PARENT_SCOPE)

# Sources only compiled when the legacy entry point is enabled
if(ENABLE_LEGACY_SIRIUS)
  set(SIRIUS_LEGACY_CONDITIONAL_SOURCES
      ${CMAKE_CURRENT_LIST_DIR}/gpu_context.cpp
      ${CMAKE_CURRENT_LIST_DIR}/gpu_physical_plan_generator.cpp
      ${CMAKE_CURRENT_LIST_DIR}/plan/gpu_plan_aggregate.cpp
      ${CMAKE_CURRENT_LIST_DIR}/plan/gpu_plan_column_data_get.cpp
      ${CMAKE_CURRENT_LIST_DIR}/plan/gpu_plan_comparison_join.cpp
      ${CMAKE_CURRENT_LIST_DIR}/plan/gpu_plan_cte.cpp
      ${CMAKE_CURRENT_LIST_DIR}/plan/gpu_plan_delim_get.cpp
      ${CMAKE_CURRENT_LIST_DIR}/plan/gpu_plan_delim_join.cpp
      ${CMAKE_CURRENT_LIST_DIR}/plan/gpu_plan_dummy_scan.cpp
      ${CMAKE_CURRENT_LIST_DIR}/plan/gpu_plan_empty_result.cpp
      ${CMAKE_CURRENT_LIST_DIR}/plan/gpu_plan_expression_get.cpp
      ${CMAKE_CURRENT_LIST_DIR}/plan/gpu_plan_filter.cpp
      ${CMAKE_CURRENT_LIST_DIR}/plan/gpu_plan_get.cpp
      ${CMAKE_CURRENT_LIST_DIR}/plan/gpu_plan_limit.cpp
      ${CMAKE_CURRENT_LIST_DIR}/plan/gpu_plan_order.cpp
      ${CMAKE_CURRENT_LIST_DIR}/plan/gpu_plan_projection.cpp
      ${CMAKE_CURRENT_LIST_DIR}/plan/gpu_plan_recursive_cte.cpp
      ${CMAKE_CURRENT_LIST_DIR}/plan/gpu_plan_top_n.cpp
      PARENT_SCOPE)
  set(SIRIUS_LEGACY_COMPILE_DEFINITIONS SIRIUS_ENABLE_LEGACY PARENT_SCOPE)
  message(STATUS "Legacy Sirius code path (gpu_processing) is ENABLED")
else()
  set(SIRIUS_LEGACY_CONDITIONAL_SOURCES "" PARENT_SCOPE)
  set(SIRIUS_LEGACY_COMPILE_DEFINITIONS "" PARENT_SCOPE)
  message(STATUS "Legacy Sirius code path (gpu_processing) is DISABLED")
endif()
# cmake-format: on

set(SIRIUS_LEGACY_INCLUDE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../include/legacy"
    PARENT_SCOPE)
