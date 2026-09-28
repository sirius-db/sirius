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

project(
  sirius
  VERSION 0.0.0
  LANGUAGES CXX CUDA)

include(GNUInstallDirs)

# Set CUDA architecture
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
  message(STATUS "CMAKE_CUDA_ARCHITECTURES not defined. Setting to 'native'.")
endif()

add_compile_definitions(CCCL_DISABLE_WARPSPEED_SCAN)

# Optional features
option(ENABLE_STREAM_CHECK
       "Enable stream check library for debugging default stream usage" OFF)
option(BUILD_WITH_CTRACK "Build with CTrack profiling support" OFF)

# The [s3] integration tests launch MinIO via a patched build of upstream
# testcontainers-native, fetched + patched at configure time and built as a Go
# c-archive — so it needs a Go toolchain and network at configure/build time. ON
# by default (the pixi env provides Go); turn OFF for offline / Go-less /
# release / package builds, which then skip the S3 container harness entirely.
option(SIRIUS_BUILD_S3_TESTS
       "Build the testcontainers-backed S3 integration test harness" ON)

# In the vcpkg build, vcpkg headers must take priority over: 1) conda's CCCL
# 3.1.4 (injected via CXXFLAGS as -I.../include/cccl) 2) DuckDB's bundled fmt
# 6.1.2 (uses duckdb_fmt namespace, incompatible with spdlog) Fix: strip conda
# CCCL path; use NO_SYSTEM_FROM_IMPORTED + BEFORE include (applied per-target
# below) to place vcpkg includes first.
if(VCPKG_BUILD)
  set(_VCPKG_INC "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/include")
  string(REGEX REPLACE "-I[^ ]*/include/cccl" "" CMAKE_CXX_FLAGS
                       "${CMAKE_CXX_FLAGS}")
  string(REGEX REPLACE "-I[^ ]*/include/cccl" "" CMAKE_CUDA_FLAGS
                       "${CMAKE_CUDA_FLAGS}")

  # The loadable extension must not depend on libcudart.so in the vcpkg build.
  # Set the CUDA language default to the static runtime before any CUDA targets
  # are created.
  set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

  # Strip conda-injected -Wl,-rpath flags from linker flags and disable CMake's
  # RPATH mechanism so binaries don't carry a hardcoded path to the pixi/conda
  # env. All deps are statically linked, so no RPATH is needed. FORCE is
  # required because DuckDB's CMakeLists.txt sets this cache var first.
  set(CMAKE_SKIP_RPATH
      ON
      CACHE BOOL "" FORCE)
  foreach(_var CMAKE_SHARED_LINKER_FLAGS CMAKE_EXE_LINKER_FLAGS
               CMAKE_MODULE_LINKER_FLAGS)
    string(REGEX REPLACE "-Wl,-rpath-link[=,][^ ]*" "" ${_var} "${${_var}}")
    string(REGEX REPLACE "-Wl,-rpath[=,][^ ]*" "" ${_var} "${${_var}}")
    set(${_var}
        "${${_var}}"
        CACHE STRING "" FORCE)
  endforeach()
endif()

# DuckDB's unity build emits strong symbols for static constexpr members (e.g.
# duckdb::TableCatalogEntry::Name) that also appear as inline definitions in
# headers pulled in by the vendored substrait reader
# (substrait/src/from_substrait.cpp). Tolerate the benign duplicate on every
# binary that links both libsirius_extension and libduckdb_static — the duckdb
# shell, the loadable extension, and the C++ test/bench exes — mirroring the
# per-target flag already on sirius_unittest / parquet_benchmark (and the Rust
# bindings).
foreach(_sirius_link_flags CMAKE_EXE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS
                           CMAKE_MODULE_LINKER_FLAGS)
  string(APPEND ${_sirius_link_flags} " -Wl,--allow-multiple-definition")
  set(${_sirius_link_flags}
      "${${_sirius_link_flags}}"
      CACHE STRING "" FORCE)
endforeach()

include(cmake/sirius-dependencies.cmake)

include(cmake/sirius-sources.cmake)

include(cmake/sirius-targets.cmake)

include(cmake/sirius-install.cmake)

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

include(cmake/sirius-test-sources.cmake)

include(cmake/sirius-tests.cmake)
