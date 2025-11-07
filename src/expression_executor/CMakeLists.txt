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
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu_expression_executor_state.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu_expression_executor.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/specializations/gpu_execute_between.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/specializations/gpu_execute_case.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/specializations/gpu_execute_cast.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/specializations/gpu_execute_comparison.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/specializations/gpu_execute_conjunction.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/specializations/gpu_execute_constant.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/specializations/gpu_execute_function.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/specializations/gpu_execute_operator.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/specializations/gpu_execute_reference.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/regex/regex_playground.cpp
        PARENT_SCOPE
)