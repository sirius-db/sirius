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

# Dependencies
find_package(cudf REQUIRED CONFIG)
find_package(spdlog REQUIRED CONFIG)
find_package(libconfig++ REQUIRED)
find_package(absl REQUIRED)
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

file(GLOB_RECURSE EXTENSION_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE CUDA_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")
list(FILTER CUDA_SOURCES EXCLUDE REGEX ".*/unused/.*")

build_static_extension(sirius ${EXTENSION_SOURCES} ${CUDA_SOURCES})
build_loadable_extension(sirius " " ${EXTENSION_SOURCES} ${CUDA_SOURCES})

# Target properties
set_target_properties(
  sirius_extension
  PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON
             CUDA_SEPARABLE_COMPILATION ON
             LINKER_LANGUAGE CUDA)

set_target_properties(
  sirius_loadable_extension
  PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON
             CUDA_SEPARABLE_COMPILATION ON
             LINKER_LANGUAGE CUDA)

# Include directories
target_include_directories(
  sirius_extension PRIVATE src/include src/include/operator src/include/cudf)
target_include_directories(
  sirius_loadable_extension PRIVATE src/include src/include/operator
                                    src/include/cudf)

# Compile definitions
target_compile_definitions(
  sirius_extension PRIVATE SIRIUS_DEFAULT_LOG_DIR="${CMAKE_BINARY_DIR}/log")
target_compile_definitions(
  sirius_loadable_extension
  PRIVATE SIRIUS_DEFAULT_LOG_DIR="${CMAKE_BINARY_DIR}/log")

# Link libraries
target_link_libraries(
  sirius_extension
  cudf::cudf
  rmm::rmm
  PkgConfig::NUMA
  ${LIBCONFIG++_LIBRARIES}
  absl::any_invocable
  spdlog::spdlog
  cuCascade::cucascade)
target_link_libraries(sirius_loadable_extension cudf::cudf rmm::rmm
                      spdlog::spdlog cuCascade::cucascade)

add_dependencies(sirius_extension duckdb_static)
add_dependencies(sirius_loadable_extension duckdb_static)

install(
  TARGETS cucascade_static
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")

install(
  TARGETS sirius_extension
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")

if(ENABLE_STREAM_CHECK)
  message(
    STATUS "Stream check library will be built (loaded dynamically at runtime)")
  add_subdirectory(utils/stream_check)

  install(
    TARGETS stream_check
    LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
    ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
else()
  message(
    STATUS
      "Stream check library will not be built (runtime loading will gracefully fail)"
  )
endif()

set(TEST_NAME sirius_unittest)

file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/test/cpp/*.cpp")
list(FILTER TEST_SOURCES EXCLUDE REGEX ".*/test_modified_pipeline\\.cpp$")
list(FILTER TEST_SOURCES EXCLUDE REGEX ".*/test_task_creator\\.cpp$")

add_executable(${TEST_NAME} ${TEST_SOURCES})

target_include_directories(
  ${TEST_NAME} PRIVATE duckdb/third_party/catch test/cpp src/include
                       src/include/operator src/include/cudf)

target_link_libraries(${TEST_NAME} sirius_extension)

set_target_properties(
  ${TEST_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                          "${CMAKE_CURRENT_BINARY_DIR}/test/cpp")

target_compile_definitions(
  ${TEST_NAME}
  PRIVATE SIRIUS_DEFAULT_LOG_DIR="${CMAKE_BINARY_DIR}/log"
          SIRIUS_UNITTEST_LOG_DIR="${CMAKE_CURRENT_BINARY_DIR}/test/cpp/log"
          SIRIUS_PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
