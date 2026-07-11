/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy of
 * the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

//! @file
//! ROCm compatibility shim for NVIDIA's nvtx3 C++ API.
//!
//! Sirius uses exactly one nvtx3 type — @c nvtx3::scoped_range — as an RAII
//! profiling range around operator execution. ROCm's equivalent is the roctx
//! API (@c roctxRangeStartA / @c roctxRangeStop), shipped in the ROCm
//! developer-tools package (@c rocm-developer-tools / @c roctx64).
//!
//! This header is placed on the include path (via CMake) only when
//! @c ENABLE_ROCM is ON, so it shadows the NVIDIA <nvtx3/nvtx3.hpp> that does
//! not exist on an AMD host. The CUDA build path is unaffected: it still
//! resolves the real system nvtx3 header.
//!
//! The shim is intentionally minimal: it implements only the constructor and
//! destructor signatures Sirius calls. @c scoped_range is non-copyable and
//! non-movable, matching nvtx3's semantics.

#pragma once

// The installed header is <roctracer/roctx.h> (the roctracer package ships it
// under that prefix; see rocBLAS's find_path for roctracer/roctx.h). The bare
// <roctx.h> form does not exist on a standard ROCm install.
#include <roctracer/roctx.h>

#include <string>

namespace nvtx3 {

/// @brief RAII profiling range backed by roctx, drop-in for @c nvtx3::scoped_range.
///
/// Starts a roctx range on construction and stops it on destruction. The label
/// is copied into the roctx internal buffer, so the caller's string need not
/// outlive the range (matching nvtx3's behavior).
class scoped_range {
 public:
  /// Construct from a string literal or @c const char*.
  explicit scoped_range(char const* name) noexcept
    : id_{roctxRangeStartA(name ? name : "")}
  {
  }

  /// Construct from a @c std::string (e.g. @c .c_str() call sites).
  explicit scoped_range(std::string const& name) noexcept
    : id_{roctxRangeStartA(name.c_str())}
  {
  }

  ~scoped_range() noexcept { roctxRangeStop(id_); }

  scoped_range(scoped_range const&)            = delete;
  scoped_range& operator=(scoped_range const&) = delete;
  scoped_range(scoped_range&&)                 = delete;
  scoped_range& operator=(scoped_range&&)      = delete;

 private:
  roctx_range_id_t const id_;
};

}  // namespace nvtx3
