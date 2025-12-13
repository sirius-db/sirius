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

#include "data/cpu_data_representation.hpp"

#include "cudf/contiguous_split.hpp"
#include "data/gpu_data_representation.hpp"

#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstring>

namespace sirius {

host_table_representation::host_table_representation(
  sirius::unique_ptr<sirius::memory::host_table_allocation> host_table,
  sirius::memory::memory_space* memory_space)
  : idata_representation(*memory_space), _host_table(std::move(host_table))
{
}

std::size_t host_table_representation::get_size_in_bytes() const { return _host_table->data_size; }

const sirius::unique_ptr<sirius::memory::host_table_allocation>&
host_table_representation::get_host_table() const
{
  return _host_table;
}

sirius::unique_ptr<idata_representation> host_table_representation::convert_to_memory_space(
  const sirius::memory::memory_space* target_memory_space, rmm::cuda_stream_view stream)
{
  auto const data_size = _host_table->data_size;

  if (target_memory_space->get_tier() == memory::Tier::GPU) {
    auto mr             = target_memory_space->get_default_allocator();
    int previous_device = -1;
    cudaGetDevice(&previous_device);
    cudaSetDevice(target_memory_space->get_device_id());

    rmm::device_buffer dst_buffer(data_size, stream, mr);
    size_t src_block_index      = 0;
    size_t src_block_offset     = 0;
    size_t dst_offset           = 0;
    size_t const src_block_size = _host_table->allocation->block_size;
    while (dst_offset < data_size) {
      size_t remaining_bytes              = data_size - dst_offset;
      size_t bytes_available_in_src_block = src_block_size - src_block_offset;
      size_t bytes_to_copy                = std::min(remaining_bytes, bytes_available_in_src_block);
      auto src_block                      = _host_table->allocation->at(src_block_index);
      cudaMemcpyAsync(static_cast<uint8_t*>(dst_buffer.data()) + dst_offset,
                      src_block.data() + src_block_offset,
                      bytes_to_copy,
                      cudaMemcpyHostToDevice,
                      stream.value());
      dst_offset += bytes_to_copy;
      src_block_offset += bytes_to_copy;
      if (src_block_offset == src_block_size) {
        src_block_index++;
        src_block_offset = 0;
      }
    }

    auto new_metadata = sirius::make_unique<sirius::vector<uint8_t>>(*_host_table->metadata);
    auto new_gpu_data = sirius::make_unique<rmm::device_buffer>(std::move(dst_buffer));
    auto new_table_view =
      cudf::unpack(new_metadata->data(), static_cast<uint8_t const*>(new_gpu_data->data()));
    auto new_table = cudf::table(new_table_view, stream, mr);
    stream.synchronize();

    cudaSetDevice(previous_device);
    return sirius::make_unique<gpu_table_representation>(
      std::move(new_table), *const_cast<sirius::memory::memory_space*>(target_memory_space));
  } else if (target_memory_space->get_tier() == memory::Tier::HOST) {
    assert(this->get_device_id() != target_memory_space->get_device_id());
    auto mr = target_memory_space
                ->get_memory_resource_as<sirius::memory::fixed_size_host_memory_resource>();
    if (mr == nullptr) {
      throw std::runtime_error(
        "Target HOST memory_space does not have a fixed_size_host_memory_resource");
    }
    auto dst_allocation         = mr->allocate_multiple_blocks(data_size);
    size_t src_block_index      = 0;
    size_t src_block_offset     = 0;
    size_t dst_block_index      = 0;
    size_t dst_block_offset     = 0;
    size_t const src_block_size = _host_table->allocation->block_size;
    size_t const dst_block_size = dst_allocation->block_size;
    size_t copied               = 0;
    while (copied < data_size) {
      size_t remaining     = data_size - copied;
      size_t src_avail     = src_block_size - src_block_offset;
      size_t dst_avail     = dst_block_size - dst_block_offset;
      size_t bytes_to_copy = std::min(remaining, std::min(src_avail, dst_avail));
      auto* src_ptr        = _host_table->allocation->at(src_block_index).data() + src_block_offset;
      auto* dst_ptr        = dst_allocation->at(dst_block_index).data() + dst_block_offset;
      std::memcpy(dst_ptr, src_ptr, bytes_to_copy);
      copied += bytes_to_copy;
      src_block_offset += bytes_to_copy;
      dst_block_offset += bytes_to_copy;
      if (src_block_offset == src_block_size) {
        src_block_index++;
        src_block_offset = 0;
      }
      if (dst_block_offset == dst_block_size) {
        dst_block_index++;
        dst_block_offset = 0;
      }
    }
    auto metadata_copy = sirius::make_unique<sirius::vector<uint8_t>>(*_host_table->metadata);
    auto host_table_allocation = sirius::make_unique<sirius::memory::host_table_allocation>(
      std::move(dst_allocation), std::move(metadata_copy), data_size);
    return sirius::make_unique<host_table_representation>(
      std::move(host_table_allocation),
      const_cast<sirius::memory::memory_space*>(target_memory_space));
  } else if (target_memory_space->get_tier() == memory::Tier::DISK) {
    throw std::runtime_error(
      "Conversion to DISK tier not implemented for host_table_representation");
  } else {
    throw std::runtime_error(
      "Invalid target memory space for host_table_representation::convert_to_memory_space");
  }
}

}  // namespace sirius
