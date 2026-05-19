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

#include <duckdb/storage/block_manager.hpp>
#include <duckdb/storage/storage_info.hpp>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace sirius::op::scan {

// DuckDB .db file layout (mirrors SingleFileBlockManager::GetBlockLocation,
// duckdb/src/storage/single_file_block_manager.cpp):
//   [0, 4096)     — main DatabaseHeader
//   [4096, 8192)  — mirror DatabaseHeader
//   [8192, 12288) — reserved third header slot
//   then blocks start at BLOCK_START = FILE_HEADER_SIZE * 3 = 12288.
//   For each block_id ∈ [0, num_blocks):
//     [block_start, block_start + block_header_size)        — checksum
//     [block_start + block_header_size, block_start + alloc) — payload
//     (BlockManager::GetBlockSize() bytes)
//   where:
//     alloc        = bm.GetBlockSize() + bm.GetBlockHeaderSize()
//     block_start  = BLOCK_START + block_id * alloc
//
// Matches what BufferManager::Pin(handle).Ptr() returns: the payload pointer,
// skipping the per-block header. Reading raw bytes via sirius_ioctx::host_read
// at the payload offset gives the same view.

/// Precondition: @p block_id must be a real block id (>= 0). DuckDB uses
/// `INVALID_BLOCK = -1` as a sentinel on `block_id_t` (a signed `int64_t`);
/// callers must filter those out (typical pattern: CONSTANT segments hold
/// their value in stats and never reference a block). Passing a negative
/// id throws @c std::invalid_argument rather than underflowing to a huge
/// offset and returning silent garbage.
inline std::size_t duckdb_block_payload_offset(duckdb::BlockManager const& bm,
                                               duckdb::block_id_t block_id)
{
  if (block_id < 0) {
    throw std::invalid_argument("duckdb_block_payload_offset: block_id must be >= 0 (got " +
                                std::to_string(block_id) + ")");
  }
  constexpr std::size_t BLOCK_START =
    static_cast<std::size_t>(duckdb::Storage::FILE_HEADER_SIZE) * 3;
  const std::size_t alloc =
    static_cast<std::size_t>(bm.GetBlockSize()) + static_cast<std::size_t>(bm.GetBlockHeaderSize());
  return BLOCK_START + static_cast<std::size_t>(block_id) * alloc +
         static_cast<std::size_t>(bm.GetBlockHeaderSize());
}

}  // namespace sirius::op::scan
