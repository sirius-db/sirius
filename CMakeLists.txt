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

file(GLOB_RECURSE EXTENSION_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE CUDA_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")
list(FILTER CUDA_SOURCES EXCLUDE REGEX ".*/unused/.*")

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
