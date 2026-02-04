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

set(TEST_SOURCES
    ${TEST_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_gpu_merge_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_gpu_partition_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_host_table_chunk_reader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_physical_result_collector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_physical_filter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_physical_limit.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_physical_top_n.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_physical_projection.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_physical_ungrouped_aggregate.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_physical_table_scan.cpp
    PARENT_SCOPE)
