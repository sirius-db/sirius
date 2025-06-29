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

cmake_minimum_required(VERSION 3.30.4)

# Enable compile_commands.json generation for clang lsp
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_definitions(ENABLE_CUDF)

# Set extension name here
set(TARGET_NAME sirius)

project(${TARGET_NAME} LANGUAGES CXX CUDA)
find_package(cudf REQUIRED CONFIG)
if(DEFINED ENV{LIBCUDF_ENV_PREFIX})
  set(CONDA_PREFIX "$ENV{LIBCUDF_ENV_PREFIX}")
else()
  message(FATAL_ERROR "LIBCUDF_ENV_PREFIX environment variable is not set.")
endif()

# Set CUDA architecture and flags
set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr -rdc=true")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

# SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -lcudart -lcupti -lcuda")
set(ENV{SKIP_SUBSTRAIT_C_TESTS} ON)

set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

include_directories(
    src/include
    src/include/operator
)
include_directories(duckdb/extension_external/substrait/src/include)
include_directories(${TARGET_NAME} duckdb/extension_external/substrait/third_party/substrait)
include_directories(${TARGET_NAME} duckdb/extension_external/substrait/third_party)

set(EXTENSION_SOURCES 
  src/sirius_extension.cpp
  src/gpu_context.cpp
  src/gpu_executor.cpp
  src/gpu_physical_operator.cpp
  src/gpu_pipeline.cpp
  src/gpu_meta_pipeline.cpp
  src/gpu_physical_plan_generator.cpp
  src/gpu_buffer_manager.cpp
  src/gpu_columns.cpp
  src/gpu_query_result.cpp
)
add_subdirectory(src/operator)
add_subdirectory(src/plan)

set(CUDA_SOURCES 
  src/cuda/communication.cu 
  src/cuda/allocator.cu
  src/cuda/print_column.cu
  src/cuda/utils.cu
)
add_subdirectory(src/cuda/operator)
add_subdirectory(src/cuda/cudf)
add_subdirectory(src/cuda/expression_executor)
add_subdirectory(src/expression_executor)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Manually specify CUDA include and library directories
# Include directories

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(src/include/cudf)
include_directories(${CONDA_PREFIX}/include)
include_directories(${CONDA_PREFIX}/include/cudf)
include_directories(${CONDA_PREFIX}/include/rmm)

# add_library(${EXTENSION_NAME} STATIC ${EXTENSION_SOURCES} ${CUDA_SOURCES})

build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES} ${CUDA_SOURCES})
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES} ${CUDA_SOURCES})

add_dependencies(${EXTENSION_NAME} duckdb_static)
add_dependencies(${LOADABLE_EXTENSION_NAME} duckdb_static)

set_target_properties(${EXTENSION_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(${LOADABLE_EXTENSION_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(${EXTENSION_NAME} PROPERTIES LINKER_LANGUAGE CUDA)
set_target_properties(${EXTENSION_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${LOADABLE_EXTENSION_NAME} PROPERTIES LINKER_LANGUAGE CUDA)
set_target_properties(${LOADABLE_EXTENSION_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(${EXTENSION_NAME}
  substrait_extension
  cudf::cudf 
  rmm::rmm
)
target_link_libraries(${LOADABLE_EXTENSION_NAME} 
  substrait_extension
  cudf::cudf 
  rmm::rmm
)

# Third-party
set(DEPS_PREFIX "_deps")
include(third_party/spdlog.cmake)
target_link_libraries(${EXTENSION_NAME} spdlog::spdlog)
target_link_libraries(${LOADABLE_EXTENSION_NAME} spdlog::spdlog)

install(
  TARGETS ${EXTENSION_NAME}
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")