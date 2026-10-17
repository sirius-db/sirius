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

#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime_api.h>

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace sirius::vss {

/**
 * @brief Copies @p size bytes at @p offset of the open file @p fd to device memory @p dst.
 *
 * Reads go through two pinned staging pieces per thread, reused across calls, so the read of one
 * piece overlaps the asynchronous copy of the other: a spilled batch's column comes back without a
 * pageable bounce or a host-pool allocation. Returns once the last copy has landed.
 */
inline void copy_file_to_device(int fd,
                                std::size_t offset,
                                std::size_t size,
                                void* dst,
                                rmm::cuda_stream_view stream,
                                std::string const& what)
{
  constexpr std::size_t kPiece = std::size_t{8} << 20;
  struct staging {
    std::array<void*, 2> host{nullptr, nullptr};
    std::array<cudaEvent_t, 2> landed{nullptr, nullptr};
    ~staging()
    {
      for (std::size_t i = 0; i < 2; ++i) {
        if (landed[i] != nullptr) { cudaEventDestroy(landed[i]); }
        if (host[i] != nullptr) { cudaFreeHost(host[i]); }
      }
    }
  };
  thread_local staging buffers;
  for (std::size_t i = 0; i < 2; ++i) {
    if (buffers.host[i] == nullptr) {
      CUDF_CUDA_TRY(cudaMallocHost(&buffers.host[i], kPiece));
      CUDF_CUDA_TRY(cudaEventCreateWithFlags(&buffers.landed[i], cudaEventDisableTiming));
    }
  }
  std::array<bool, 2> pending{false, false};
  std::size_t piece = 0;
  for (std::size_t done = 0; done < size; done += kPiece, ++piece) {
    auto const slot = piece % 2;
    if (pending[slot]) { CUDF_CUDA_TRY(cudaEventSynchronize(buffers.landed[slot])); }
    auto const n = std::min(kPiece, size - done);
    auto* host   = static_cast<std::byte*>(buffers.host[slot]);
    for (std::size_t got = 0; got < n;) {
      auto const r = ::pread(fd, host + got, n - got, static_cast<off_t>(offset + done + got));
      if (r <= 0) { throw std::runtime_error("reading spilled batch " + what + " failed"); }
      got += static_cast<std::size_t>(r);
    }
    CUDF_CUDA_TRY(cudaMemcpyAsync(
      static_cast<std::byte*>(dst) + done, host, n, cudaMemcpyHostToDevice, stream.value()));
    CUDF_CUDA_TRY(cudaEventRecord(buffers.landed[slot], stream.value()));
    pending[slot] = true;
  }
  stream.synchronize();
}

}  // namespace sirius::vss
