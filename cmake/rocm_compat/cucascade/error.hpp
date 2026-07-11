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
//! cuCascade error types and macros — ROCm stub.
//!
//! Provides @c cucascade::cuda_error, @c cucascade::logic_error, and the
//! @c CUCASCADE_CUDA_TRY / @c CUCASCADE_FAIL macros. The macros wrap
//! @c cudaError_t (provided by the cuda_runtime.h shim) and throw on failure,
//! matching the real cuCascade's behavior.

#pragma once

#include <cuda_runtime.h>  // cudaError_t, cudaSuccess, cudaGetLastError, cudaGetErrorString (shim)
#include <stdexcept>
#include <string>
#include <string_view>

namespace cucascade {

/// Exception thrown when a CUDA Runtime call fails (mirrors rmm::cuda_error).
struct cuda_error : std::runtime_error {
  explicit cuda_error(std::string_view msg) : std::runtime_error(std::string(msg)) {}
};

/// Exception thrown on logic errors (invalid arguments, invariant violations).
struct logic_error : std::logic_error {
  explicit logic_error(std::string_view msg) : std::logic_error(std::string(msg)) {}
};

}  // namespace cucascade

// ---------------------------------------------------------------------------
// CUCASCADE_CUDA_TRY — mirrors RMM_CUDA_TRY: capture cudaError_t, throw on
// non-success.  Uses cudaError_t / cudaSuccess / cudaGetLastError from the
// cuda_runtime.h shim.
// ---------------------------------------------------------------------------
#define CUCASCADE_CUDA_TRY(_call)                                              \
  do {                                                                         \
    cudaError_t const _cucascade_err = (_call);                                \
    if (cudaSuccess != _cucascade_err) {                                       \
      cudaGetLastError();                                                      \
      throw ::cucascade::cuda_error(                                           \
        std::string{"cuCascade stub: CUDA error: "} +                          \
        cudaGetErrorString(_cucascade_err));                                   \
    }                                                                          \
  } while (0)

#define CUCASCADE_CUDA_TRY_ALLOC(_call, _size) CUCASCADE_CUDA_TRY(_call)

#define CUCASCADE_FAIL(_msg) throw ::cucascade::logic_error(_msg)

#define CUCASCADE_ASSERT_CUDA_SUCCESS(_err)                                    \
  do {                                                                         \
    if (cudaSuccess != (_err)) {                                               \
      throw ::cucascade::cuda_error(cudaGetErrorString(_err));                 \
    }                                                                          \
  } while (0)

#define CUCASCADE_STRINGIFY_IMPL(x) #x
#define CUCASCADE_STRINGIFY(x) CUCASCADE_STRINGIFY_IMPL(x)
