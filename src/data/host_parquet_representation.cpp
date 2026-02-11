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

// sirius
#include <data/host_parquet_representation.hpp>

namespace sirius {

std::unique_ptr<cucascade::idata_representation> host_parquet_representation::clone(
  rmm::cuda_stream_view /*stream */)
{
  // Get the host memory resource from the memory space
  auto* host_mr = get_memory_space().get_memory_resource_of<cucascade::memory::Tier::HOST>();
  if (!host_mr) {
    throw std::runtime_error("Cannot clone host_table_representation: no host memory resource");
  }

  // Allocate new blocks for the copy
  auto allocation_copy = host_mr->allocate_multiple_blocks(_size_in_bytes);

  // Copy data block by block
  const auto& src_blocks = _column_chunks->get_blocks();
  auto dst_blocks        = allocation_copy->get_blocks();
  std::size_t remaining  = _size_in_bytes;
  std::size_t block_size = _column_chunks->block_size();

  for (std::size_t i = 0; i < src_blocks.size() && remaining > 0; ++i) {
    std::size_t copy_size = std::min(remaining, block_size);
    std::memcpy(dst_blocks[i], src_blocks[i], copy_size);
    remaining -= copy_size;
  }

  // Clone the reader
  auto cloned_reader =
    std::make_unique<hybrid_scan_reader>(_parquet_reader->parquet_metadata(), _reader_options);
  return std::make_unique<host_parquet_representation>(&get_memory_space(),
                                                       std::move(allocation_copy),
                                                       std::move(cloned_reader),
                                                       _reader_options,
                                                       _row_group_indices,
                                                       _column_chunk_byte_ranges,
                                                       _size_in_bytes,
                                                       _uncompressed_size_in_bytes);
}

}  // namespace sirius
