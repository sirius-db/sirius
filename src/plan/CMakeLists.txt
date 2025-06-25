# =============================================================================
# Copyright 2025, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

set(EXTENSION_SOURCES
        ${EXTENSION_SOURCES}
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu_plan_aggregate.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu_plan_column_data_get.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu_plan_comparison_join.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu_plan_delim_get.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu_plan_delim_join.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu_plan_distinct.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu_plan_dummy_scan.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu_plan_empty_result.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu_plan_filter.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu_plan_get.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu_plan_limit.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu_plan_order.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu_plan_projection.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu_plan_cte.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu_plan_recursive_cte.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu_plan_top_n.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu_plan_expression_get.cpp
        PARENT_SCOPE
)