// SPDX-License-Identifier: Apache-2.0
// Shared CUDA runtime and driver error checks: throw a std::runtime_error tagged with `context`
// when a CUDA call fails, so callers can propagate cleanly instead of open-coding the same
// if/throw. nvcomp-status checks stay local to the nvcomp layer.
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace simpatico {

inline void throw_if_cuda_error(cudaError_t err, const char* context)
{
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(err));
  }
}

inline void throw_if_cu_error(CUresult result, const char* context)
{
  if (result == CUDA_SUCCESS) return;
  const char* name        = nullptr;
  const char* description = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &description);
  throw std::runtime_error(std::string(context) + ": " + (name ? name : "<unknown>") + " (" +
                           (description ? description : "?") + ")");
}

}  // namespace simpatico
