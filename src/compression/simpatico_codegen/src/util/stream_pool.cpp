// SPDX-License-Identifier: Apache-2.0
#include "codegen/util/stream_pool.hpp"

namespace simpatico {

bool stream_pool::init(size_t n)
{
  if (!streams.empty()) return true;  // Already initialized
  streams.resize(n);
  for (size_t i = 0; i < n; ++i) {
    cudaError_t err = cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    if (err != cudaSuccess) {
      for (size_t j = 0; j < i; ++j) {
        cudaStreamDestroy(streams[j]);
      }
      streams.clear();
      return false;
    }
  }
  return true;
}

void stream_pool::shutdown()
{
  for (auto& stream : streams) {
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
  }
  streams.clear();
}

void stream_pool::sync_all()
{
  for (auto& stream : streams) {
    cudaStreamSynchronize(stream);
  }
}

}  // namespace simpatico
