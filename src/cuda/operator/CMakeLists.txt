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

set(CUDA_SOURCES
        ${CUDA_SOURCES}
        ${CMAKE_CURRENT_SOURCE_DIR}/arbitrary_expression.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/comparison_expression.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/materialize.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/hash_join_inner.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/hash_join_right.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/hash_join_single.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/nested_loop_join.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/strings_matching.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/substring.cu
        PARENT_SCOPE
)