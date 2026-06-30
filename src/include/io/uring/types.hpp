/*
 * Copyright 2026, Sirius Contributors.
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

#include "io/io_request.hpp"
#include "io/types.hpp"

#include <cudf/io/datasource.hpp>
#include <cudf/io/text/byte_range_info.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime.h>

#include <sys/uio.h>

#include <cstddef>
#include <memory>
#include <vector>

namespace sirius::io::uring {

// request_manager, device_cpy_request, and the rx_request container live in
// io/io_request.hpp (shared across reactors).  Re-export the two data types
// into this namespace so existing fully-qualified references
// (sirius::io::uring::request_manager, ...) keep resolving.
using request_manager    = sirius::io::request_manager;
using device_cpy_request = sirius::io::device_cpy_request;

struct chunked_rx_request {
  int fd;
  // The read to perform.  @c chunk.buffers holds the destination iovec(s):
  // one buffer => a plain read, more than one => a vectored (readv) read whose
  // iovecs cover [chunk.offset, chunk.offset + chunk.size) contiguously in the
  // file.  The worker's EOF/short-read math uses chunk.offset/chunk.size for
  // both modes.
  io_object_segment chunk;
  // Size of the underlying file.  Used by the worker loop to distinguish a
  // genuine short read (must be re-submitted to read the rest) from a partial
  // read that simply reached EOF (already complete — re-submitting would read
  // at offset == file_size, or a non-block-aligned tail under O_DIRECT).
  size_t file_size{0};

  /// @return true iff this request must be submitted via io_uring_prep_readv.
  [[nodiscard]] bool is_vectored() const noexcept { return chunk.is_vectored(); }

  /// Remaining single-buffer range to read after @p offset bytes have landed
  /// (plain-read resume path).
  [[nodiscard]] io_object_segment get_remaining_chunk(size_t offset) const noexcept
  {
    if (offset >= chunk.size) return io_object_segment{0, 0};
    return io_object_segment{chunk.offset + offset, chunk.size - offset, chunk.data() + offset};
  }

  /// Rebuild the iovec list for resuming a short readv after @p skip bytes were
  /// already read (vectored-read resume path).  Fills @p out in place, reusing
  /// its capacity across resubmissions; @c chunk.buffers is untouched.
  void fill_remaining_iovecs(size_t skip, std::vector<iovec>& out) const
  {
    chunk.fill_remaining_buffers(skip, out);
  }

  cudaError_t copy_h2d_async(cudaEvent_t event = nullptr) noexcept
  {
    if (cpy_req) [[likely]] {
      return cpy_req->copy_async(chunk.data(), chunk.size, event);
    } else {
      return cudaSuccess;
    }
  }

  [[nodiscard]] bool needs_event_for_synchronization() const noexcept
  {
    return !chunk.is_buffer_allocated() && cpy_req != nullptr;
  }

  std::unique_ptr<device_cpy_request> cpy_req;
  std::shared_ptr<request_manager> manager;
};

// The per-reactor request container is the shared rx_request_t template
// instantiated for this backend's chunk type.
using rx_request = rx_request_t<chunked_rx_request>;

}  // namespace sirius::io::uring
