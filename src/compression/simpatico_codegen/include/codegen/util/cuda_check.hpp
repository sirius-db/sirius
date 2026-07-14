// SPDX-License-Identifier: Apache-2.0
// Shared CUDA-error check: throw a std::runtime_error tagged with `context` when
// a CUDA call fails, so callers can propagate cleanly instead of open-coding the
// same if/throw. nvcomp-status checks stay local to the nvcomp layer.
#ifndef SIMPATICO_CUDA_CHECK_HPP
#define SIMPATICO_CUDA_CHECK_HPP

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

}  // namespace simpatico

#endif  // SIMPATICO_CUDA_CHECK_HPP
