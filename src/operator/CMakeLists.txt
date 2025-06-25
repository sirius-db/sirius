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
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu_physical_column_data_scan.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu_physical_dummy_scan.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu_physical_limit.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu_physical_delim_join.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu_physical_empty_result.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu_physical_filter.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu_physical_hash_join.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu_physical_grouped_aggregate.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu_physical_order.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu_physical_projection.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu_physical_ungrouped_aggregate.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu_physical_result_collector.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu_physical_table_scan.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu_physical_nested_loop_join.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu_materialize.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu_physical_strings_matching.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu_physical_substring.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu_physical_cte.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu_physical_top_n.cpp
        PARENT_SCOPE
)