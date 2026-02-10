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

project(stream_check LANGUAGES CXX)

# Find required packages
find_package(cudf REQUIRED CONFIG)
find_package(absl REQUIRED CONFIG)

# Create shared library for LD_PRELOAD
add_library(stream_check SHARED stream_check.cpp)

# Set C++ standard
set_target_properties(
  stream_check
  PROPERTIES CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON)

# Include directories
target_include_directories(stream_check PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Link against RMM (cudf dependency) and absl for stack traces
target_link_libraries(
  stream_check PUBLIC rmm::rmm absl::stacktrace absl::symbolize
                      absl::failure_signal_handler)

# Install library
install(
  TARGETS stream_check
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

# Install header
install(FILES stream_check.hpp DESTINATION include)
