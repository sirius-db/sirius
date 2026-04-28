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

#include "data/sirius_p2p_converter.hpp"

#include <cudf/contiguous_split.hpp>
#include <cudf/table/table.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda_runtime_api.h>

#include <spdlog/spdlog.h>

#include <cstdint>
#include <stdexcept>

namespace sirius::data {

std::unique_ptr<cucascade::idata_representation> sirius_p2p_converter_factory(
  cucascade::idata_representation& source,
  const cucascade::memory::memory_space* target_memory_space,
  rmm::cuda_stream_view stream)
{
  auto& gpu_source            = source.cast<cucascade::gpu_table_representation>();
  auto const source_device_id = source.get_device_id();
  auto const target_device_id = target_memory_space->get_device_id();

  // Same-device case: delegate to the source's own clone() (no peer copy
  // needed; matches cucascade's fast path at
  // cucascade/src/data/representation_converter.cpp:151).
  if (source_device_id == target_device_id) { return source.clone(stream); }

  // Synchronize the caller's stream so any prior source-side operation
  // (e.g. the cudf::table construction that built the source payload) is
  // guaranteed complete before we read the source buffer. The caller's
  // stream may live on any device; sync is a no-op at the driver level if
  // there's no outstanding work.
  stream.synchronize();

  // --- Pack on the source device context, using a source-bound stream. ---
  // cudf::pack allocates a contiguous device buffer on the CURRENT device.
  // We explicitly construct a source-bound rmm::cuda_stream under
  // source_guard so cudf's internal async copies + allocations all target
  // source_device (avoids cross-device stream-use errors seen when the
  // caller's stream belongs to a different device).
  rmm::cuda_set_device_raii source_guard{rmm::cuda_device_id{source_device_id}};
  rmm::cuda_stream source_stream;  // bound to source_device (current)
  auto packed              = cudf::pack(gpu_source.get_table(), source_stream.view());
  auto const bytes_to_copy = packed.gpu_data->size();
  source_stream.synchronize();  // pack done; packed.gpu_data stable on source_device.

  // --- Allocate destination on the target device context. ---
  // target_stream belongs to target_device; acquire_stream() returns a
  // cuda_stream_view from the target_memory_space's stream pool.
  auto target_stream = target_memory_space->acquire_stream();
  auto mr            = target_memory_space->get_default_allocator();

  // Switch to target device so the device_uvector allocation targets the
  // right resource on the right device context.
  rmm::cuda_set_device_raii target_guard{rmm::cuda_device_id{target_device_id}};
  rmm::device_uvector<uint8_t> dst_uvector(bytes_to_copy, target_stream, mr);

  // --- Peer copy on target_stream (KEY FIX vs cucascade's body). ---
  // cucascade's convert_gpu_to_gpu issues cudaMemcpyPeerAsync on the
  // CALLER's stream and then builds the target cudf::table on target_stream
  // with no cross-stream event. That cross-stream race is the return-leg
  // bug tracked at Phase-4 Plan 04-05 Task 2 (test_downgrade_executor.cpp).
  // We issue the peer copy on target_stream so the subsequent unpack +
  // table construction on the same stream observes in-order completion.
  //
  // cudaMemcpyPeerAsync accepts src/dst pointers from different device
  // contexts; the current device does not need to match src or dst.
  cudaError_t peer_err = cudaMemcpyPeerAsync(dst_uvector.data(),
                                             target_device_id,
                                             static_cast<const uint8_t*>(packed.gpu_data->data()),
                                             source_device_id,
                                             bytes_to_copy,
                                             target_stream.value());
  if (peer_err != cudaSuccess) {
    // Consume any sticky state before reporting.
    (void)cudaGetLastError();
    throw std::runtime_error(
      std::string("sirius_p2p_converter: cudaMemcpyPeerAsync ") + std::to_string(source_device_id) +
      " -> " + std::to_string(target_device_id) + " failed: " + cudaGetErrorString(peer_err) +
      " (MGPU-06; verify driver-level peer access is enabled for this pair)");
  }

  // Sync target_stream so the peer-copied bytes are stable on target_device
  // before unpack reads them. Keeping the copy + unpack on target_stream
  // means this is the ONLY cross-stream synchronization in the function.
  target_stream.synchronize();

  // --- Unpack on the target device context. ---
  // Construct an owning cudf::table on target_stream using the freshly-
  // peer-copied bytes. Move dst_uvector into a device_buffer so the table
  // takes long-lived ownership of the underlying allocation.
  rmm::device_buffer dst_buffer = std::move(dst_uvector).release();
  auto new_metadata             = std::move(packed.metadata);
  auto new_gpu_data             = std::make_unique<rmm::device_buffer>(std::move(dst_buffer));
  auto new_table_view =
    cudf::unpack(new_metadata->data(), static_cast<uint8_t const*>(new_gpu_data->data()));
  auto new_table = std::make_unique<cudf::table>(new_table_view, target_stream, mr);
  target_stream.synchronize();

  return std::make_unique<cucascade::gpu_table_representation>(
    std::move(new_table), *const_cast<cucascade::memory::memory_space*>(target_memory_space));
}

}  // namespace sirius::data
