/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cudf/utilities/error.hpp>

#include <cuda_runtime_api.h>

namespace sirius::vss {

/**
 * @brief rmm::cuda_set_device_raii for code a DuckDB thread runs, which also makes the device's
 * primary context current.
 *
 * rmm's guard calls cudaSetDevice only when the device differs. A DuckDB worker that has made no
 * CUDA runtime call yet already reports device 0 but has no current context, and the first
 * driver-API call on it -- a cuda::stream_ref sync -- then fails with CUDA_ERROR_INVALID_CONTEXT.
 * Since CUDA 12 cudaSetDevice binds the primary context, so it is always called here.
 */
class device_context_guard {
 public:
  explicit device_context_guard(int device)
  {
    CUDF_CUDA_TRY(cudaGetDevice(&_previous));
    CUDF_CUDA_TRY(cudaSetDevice(device));
    _device = device;
  }
  ~device_context_guard()
  {
    if (_previous != _device) { cudaSetDevice(_previous); }
  }
  device_context_guard(device_context_guard const&)            = delete;
  device_context_guard& operator=(device_context_guard const&) = delete;

 private:
  int _previous{0};
  int _device{0};
};

}  // namespace sirius::vss
