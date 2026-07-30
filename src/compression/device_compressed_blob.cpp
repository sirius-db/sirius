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

#include "device_compressed_blob.hpp"

#include <api/compressed_table_io.hpp>
#include <cucascade/error.hpp>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace sirius {

std::shared_ptr<compressed_device_blob> build_device_compressed_blob(
  std::span<const std::uint8_t> header,
  std::span<const simpatico::payload_buffer_ref> buffers,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref payload_mr,
  rmm::device_async_resource_ref scratch_mr)
{
  auto blob = std::make_shared<compressed_device_blob>();

  constexpr std::size_t kLeafAlign = rmm::CUDA_ALLOCATION_ALIGNMENT;  // 256
  auto const align_up = [](std::uint64_t n, std::uint64_t a) { return (n + a - 1) & ~(a - 1); };

  // offsets[k] holds buffers[slot_src[k]] — only buffers the re-read will actually
  // allocate for get a slot (see the header's note on zero-footprint leaves).
  blob->offsets.reserve(buffers.size());
  std::vector<std::size_t> slot_src;
  slot_src.reserve(buffers.size());
  std::uint64_t cur = 0;
  for (std::size_t i = 0; i < buffers.size(); ++i) {
    auto const& b         = buffers[i];
    std::uint64_t const n = std::max(b.size_bytes, b.alloc_bytes);
    if (n == 0) { continue; }
    cur = align_up(cur, kLeafAlign);
    blob->offsets.push_back(cur);
    slot_src.push_back(i);
    cur += n;
  }

  std::size_t const payload_capacity = align_up(cur, kLeafAlign) + kLeafAlign;  // tail slop
  blob->payload                      = rmm::device_buffer(payload_capacity, stream, payload_mr);
  CUCASCADE_CUDA_TRY(cudaMemsetAsync(blob->payload.data(), 0, payload_capacity, stream.value()));

  for (std::size_t k = 0; k < blob->offsets.size(); ++k) {
    auto const& b = buffers[slot_src[k]];
    if (b.size_bytes > 0 && b.device_ptr != nullptr) {
      CUCASCADE_CUDA_TRY(
        cudaMemcpyAsync(static_cast<std::byte*>(blob->payload.data()) + blob->offsets[k],
                        b.device_ptr,
                        static_cast<std::size_t>(b.size_bytes),
                        cudaMemcpyDeviceToDevice,
                        stream.value()));
    }
  }

  blob->slab_mr = slab_memory_resource{
    static_cast<std::byte*>(blob->payload.data()), &blob->offsets, &blob->slab_cursor};

  auto noop_fetch = [](std::uint64_t, std::size_t, void*, rmm::cuda_stream_view) {};
  std::string read_err;
  blob->table = simpatico::read_compressed_table_from_memory(
    header, noop_fetch, stream, scratch_mr, &read_err, /*leaf_mr=*/blob->slab_mr);
  if (!read_err.empty()) { throw std::runtime_error("[build_device_compressed_blob] " + read_err); }

  // The caller may drop the source compressed_table (which owns `buffers`) as soon
  // as this returns, so the copies above must have landed.
  stream.synchronize();
  return blob;
}

}  // namespace sirius
