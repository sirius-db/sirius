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

cmake_minimum_required(VERSION 3.30.4)

# Enable compile_commands.json generation for clang lsp
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_definitions(ENABLE_CUDF)

# Set extension name here
set(TARGET_NAME sirius)

# Set CUDA architecture
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
  message(STATUS "CMAKE_CUDA_ARCHITECTURES not defined. Setting to 'native'.")
endif()

project(${TARGET_NAME} LANGUAGES CXX CUDA)
find_package(cudf REQUIRED CONFIG)

# Derive the environment prefix from where cudf was found cudf_DIR is typically:
# <prefix>/lib/cmake/cudf
cmake_path(GET cudf_DIR PARENT_PATH _cmake_dir)
cmake_path(GET _cmake_dir PARENT_PATH _lib_dir)
cmake_path(GET _lib_dir PARENT_PATH CUDF_ENV_PREFIX)
message(STATUS "Derived environment prefix from cudf: ${CUDF_ENV_PREFIX}")

find_library(
  NUMA_LIB
  NAMES numa
  HINTS "${CUDF_ENV_PREFIX}/lib"
  DOC "Path to the libnuma library")

if(NUMA_LIB)
  message(STATUS "Found libnuma: ${NUMA_LIB}")
else()
  message(
    FATAL_ERROR
      "libnuma not found. Checked system paths and ${CUDF_ENV_PREFIX}/lib.")
endif()

# Set CUDA and CXX flags
set(CMAKE_CUDA_FLAGS
    "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr -rdc=true"
)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")

# SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -lcudart -lcupti -lcuda")
set(ENV{SKIP_SUBSTRAIT_C_TESTS} ON)

set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

# Third-party dependencies (must be before include_directories for cuCascade
# headers)
set(DEPS_PREFIX "_deps")
include(third_party/spdlog.cmake)
include(third_party/cucascade.cmake)

# Include directories - cuCascade headers first for memory module
include_directories(${CUCASCADE_INCLUDE_DIR})
include_directories(src/include src/include/operator)
# include_directories(substrait/src/include) include_directories(${TARGET_NAME}
# substrait/third_party/substrait) include_directories(${TARGET_NAME}
# substrait/third_party)

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
    src/cpu_cache.cpp
    src/gpu_query_result.cpp
    src/config.cpp
    src/fallback.cpp
    src/parallel/task_executor.cpp)
# Note: Memory module sources are now provided by cuCascade library
add_subdirectory(src/operator)
add_subdirectory(src/plan)
add_subdirectory(src/data)
add_subdirectory(src/pipeline)
add_subdirectory(src/downgrade)
add_subdirectory(src/creator)
add_subdirectory(src/op)

set(CUDA_SOURCES src/cuda/communication.cu src/cuda/allocator.cu
                 src/cuda/print.cu src/cuda/utils.cu)
add_subdirectory(src/cuda/operator)
add_subdirectory(src/cuda/cudf)
add_subdirectory(src/cuda/expression_executor)
add_subdirectory(src/expression_executor)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Manually specify CUDA include and library directories Include directories
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(src/include/cudf)
# Note: cudf::cudf and rmm::rmm targets (from find_package) already provide the
# necessary include paths. Do not add CONDA_PREFIX includes here as it may point
# to a different environment causing duplicate definitions.

# add_library(${EXTENSION_NAME} STATIC ${EXTENSION_SOURCES} ${CUDA_SOURCES})
build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES} ${CUDA_SOURCES})
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES}
                         ${CUDA_SOURCES})

add_dependencies(${EXTENSION_NAME} duckdb_static)
add_dependencies(${LOADABLE_EXTENSION_NAME} duckdb_static)

set_target_properties(${EXTENSION_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS
                                                   ON)
set_target_properties(${LOADABLE_EXTENSION_NAME}
                      PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(${EXTENSION_NAME} PROPERTIES LINKER_LANGUAGE CUDA)
set_target_properties(${EXTENSION_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION
                                                   ON)
set_target_properties(${LOADABLE_EXTENSION_NAME} PROPERTIES LINKER_LANGUAGE
                                                            CUDA)
set_target_properties(${LOADABLE_EXTENSION_NAME}
                      PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# target_link_libraries(${EXTENSION_NAME} substrait_extension cudf::cudf
# rmm::rmm ${NUMA_LIB}) target_link_libraries(${LOADABLE_EXTENSION_NAME}
# substrait_extension cudf::cudf rmm::rmm)
target_link_libraries(${EXTENSION_NAME} cudf::cudf rmm::rmm ${NUMA_LIB})
target_link_libraries(${LOADABLE_EXTENSION_NAME} cudf::cudf rmm::rmm)

# Link third-party libraries
target_link_libraries(${EXTENSION_NAME} spdlog::spdlog cuCascade::cucascade)
target_link_libraries(${LOADABLE_EXTENSION_NAME} spdlog::spdlog
                      cuCascade::cucascade)

# Ensure third-party libraries are built before extensions Note: cucascade is
# added via add_subdirectory, so dependencies are automatic
add_dependencies(${EXTENSION_NAME} spdlog_ep)
add_dependencies(${LOADABLE_EXTENSION_NAME} spdlog_ep)

# Install cucascade_static to the DuckDB export set (required for
# sirius_extension export)
install(
  TARGETS cucascade_static
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")

install(
  TARGETS ${EXTENSION_NAME}
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")

# Unit tests
set(TEST_NAME ${TARGET_NAME}_unittest)
add_subdirectory(test/cpp)
add_executable(${TEST_NAME} ${TEST_SOURCES})
target_include_directories(${TEST_NAME} PRIVATE duckdb/third_party/catch
                                                test/cpp)
target_link_libraries(${TEST_NAME} ${EXTENSION_NAME})
set(TEST_EXEC_DIR "${CMAKE_CURRENT_BINARY_DIR}/test/cpp")
set(TEST_LOG_DIR "${CMAKE_CURRENT_BINARY_DIR}/test/cpp/log")
set_target_properties(${TEST_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                              "${TEST_EXEC_DIR}")
target_compile_definitions(${TEST_NAME}
                           PRIVATE SIRIUS_UNITTEST_LOG_DIR="${TEST_LOG_DIR}")
