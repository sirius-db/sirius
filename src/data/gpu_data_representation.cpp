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

#include "data/gpu_data_representation.hpp"
#include <cudf/utilities/traits.hpp>
#include "memory/host_table.hpp"
#include "data/cpu_data_representation.hpp"
#include "rmm/cuda_stream_view.hpp"
#include "cudf/contiguous_split.hpp"
#include <rmm/device_uvector.hpp>
#include <rmm/device_buffer.hpp>
#include <cuda_runtime_api.h>
namespace sirius {

gpu_table_representation::gpu_table_representation(cudf::table table,
                                                   sirius::memory::memory_space& memory_space)
  : idata_representation(memory_space), _table(std::move(table))
{
}

std::size_t gpu_table_representation::get_size_in_bytes() const
{
  // TODO: Implement proper size calculation
  // This should return the total size of all columns in the table
  std::size_t total_size = 0;
  for (auto const& col : _table.view()) {
    // For now, we can calculate a rough estimate based on column size
    // This will need to be refined to account for all buffers (data, validity, offsets, etc.)
    total_size += col.size() * cudf::size_of(col.type());
  }
  return total_size;
}

const cudf::table& gpu_table_representation::get_table() const { return _table; }

sirius::unique_ptr<idata_representation> gpu_table_representation::convert_to_memory_space(
  const sirius::memory::memory_space* target_memory_space, rmm::cuda_stream_view stream)
{
  auto packed_data = cudf::pack(_table, stream);
  if (target_memory_space->get_tier() == memory::Tier::GPU) {
    assert(this->get_device_id() != target_memory_space->get_device_id());
    auto const target_device_id = target_memory_space->get_device_id();
    auto const source_device_id = this->get_device_id();
    auto const bytes_to_copy    = packed_data.gpu_data->size();
    auto mr                     = target_memory_space->get_default_allocator();

    // Acquire a stream that belongs to the target GPU device
    auto target_stream = target_memory_space->acquire_stream();

    cudaSetDevice(target_device_id);
    rmm::device_uvector<uint8_t> dst_uvector(bytes_to_copy, target_stream, mr);
    target_stream.synchronize();
    // Restore previous device before peer copy
    cudaSetDevice(source_device_id);

    // Asynchronously copy device->device across GPUs
    cudaMemcpyPeerAsync(dst_uvector.data(),
                        target_device_id,
                        static_cast<const uint8_t*>(packed_data.gpu_data->data()),
                        source_device_id,
                        bytes_to_copy,
                        stream.value());
    stream.synchronize();
    // Unpack on target device to build a cudf::table that lives on the target GPU
    cudaSetDevice(target_device_id);
    rmm::device_buffer dst_buffer = std::move(dst_uvector).release();
    // Unpack using pointer-based API and construct an owning cudf::table
    auto new_metadata = std::move(packed_data.metadata);
    auto new_gpu_data = sirius::make_unique<rmm::device_buffer>(std::move(dst_buffer));
    auto new_table_view =
      cudf::unpack(new_metadata->data(), static_cast<uint8_t const*>(new_gpu_data->data()));
    auto new_table = cudf::table(new_table_view, target_stream, mr);
    // Restore previous device
    target_stream.synchronize();
    cudaSetDevice(source_device_id);
    target_memory_space->release_stream(target_stream);

    return sirius::make_unique<gpu_table_representation>(
      std::move(new_table), *const_cast<sirius::memory::memory_space*>(target_memory_space));
  } else if (target_memory_space->get_tier() == memory::Tier::HOST) {
    auto mr = target_memory_space
                ->get_default_allocator_as<sirius::memory::fixed_size_host_memory_resource>();
    auto allocation = mr->allocate_multiple_blocks(packed_data.gpu_data->size());

    size_t block_index      = 0;
    size_t block_offset     = 0;
    size_t source_offset    = 0;
    const size_t block_size = allocation.block_size;
    while (source_offset < packed_data.gpu_data->size()) {
      size_t remaining_bytes = packed_data.gpu_data->size() - source_offset;
      size_t bytes_to_copy   = std::min(remaining_bytes, block_size - block_offset);
      void* block_ptr        = allocation[block_index];
      cudaMemcpyAsync(static_cast<uint8_t*>(block_ptr) + block_offset,
                      static_cast<const uint8_t*>(packed_data.gpu_data->data()) + source_offset,
                      bytes_to_copy,
                      cudaMemcpyDeviceToHost,
                      stream.value());
      source_offset += bytes_to_copy;
      block_offset += bytes_to_copy;
      if (block_offset == block_size) {
        block_index++;
        block_offset = 0;
      }
    }
    stream.synchronize();
    auto host_table_allocation = sirius::make_unique<sirius::memory::host_table_allocation>(
      std::move(allocation), std::move(packed_data.metadata), packed_data.gpu_data->size());
    return sirius::make_unique<host_table_representation>(
      std::move(host_table_allocation),
      const_cast<sirius::memory::memory_space*>(target_memory_space));
  } else {
    throw std::runtime_error(
      "Invalid target memory space for "
      "gpu_table_representation::convert_to_memory_space");
  }
}

}  // namespace sirius
