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
  LANGUAGES C CXX CUDA)

include(GNUInstallDirs)
option(SIRIUS_BUILD_SHARED "Build the shared Sirius library" ON)
option(SIRIUS_BUILD_STATIC "Build the static Sirius library" OFF)
if(SIRIUS_BUILD_STATIC AND NOT VCPKG_BUILD)
  message(
    FATAL_ERROR "The installed static package requires vcpkg dependencies")
endif()
if(NOT SIRIUS_BUILD_SHARED AND NOT SIRIUS_BUILD_STATIC)
  message(FATAL_ERROR "Enable at least one Sirius library output")
endif()
option(SIRIUS_BUILD_TESTS "Build Sirius C++ tests" OFF)

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

if(VCPKG_BUILD)
  set(_VCPKG_INC "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/include")
endif()

include(cmake/sirius-duckdb-provider.cmake)

include(cmake/sirius-dependencies.cmake)

add_library(sirius_objects OBJECT)
add_subdirectory(src)

include(cmake/sirius-substrait.cmake)

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

if(SIRIUS_BUILD_TESTS OR NOT PROJECT_IS_TOP_LEVEL)
  include(cmake/sirius-test-sources.cmake)
  include(cmake/sirius-tests.cmake)
endif()
