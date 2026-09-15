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

#include <duckdb/storage/single_file_block_manager.hpp>
#include <duckdb/storage/storage_info.hpp>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace sirius::op::scan {

// .db file layout (mirrors SingleFileBlockManager::GetBlockLocation):
//   [0, 4096)     — DatabaseHeader
//   [4096, 8192)  — mirror DatabaseHeader
//   [8192, 12288) — reserved third header slot
//   blocks start at FILE_HEADER_SIZE * 3 = 12288.
//   Per block: [block_header_size bytes checksum][block_size bytes payload]
//
// SingleFileBlockManager::GetBlockLocation does this math but is private —
// drop this helper when DuckDB exposes it. Until then, format-version drift
// will land here.
//
// SingleFileBlockManager-typed because the math is specific to that layout.
// The only sibling is InMemoryBlockManager (`:memory:`), which has no blocks
// to offset and whose IO methods all throw.
constexpr std::size_t DUCKDB_BLOCK_START =
  static_cast<std::size_t>(duckdb::Storage::FILE_HEADER_SIZE) * 3;

/// Returns the byte offset of @p block_id's payload in the .db file — the
/// pointer @c BufferManager::Pin(handle).Ptr() would return, but without
/// going through DuckDB's buffer cache (we want raw bytes via
/// @c ioctx::host_read).
///
/// @throws std::invalid_argument on negative @p block_id (DuckDB uses
///         @c INVALID_BLOCK = -1 as a sentinel; callers must filter).
inline std::size_t duckdb_block_payload_offset(duckdb::SingleFileBlockManager const& bm,
                                               duckdb::block_id_t block_id)
{
  if (block_id < 0) {
    throw std::invalid_argument("duckdb_block_payload_offset: block_id must be >= 0 (got " +
                                std::to_string(block_id) + ")");
  }
  const std::size_t alloc =
    static_cast<std::size_t>(bm.GetBlockSize()) + static_cast<std::size_t>(bm.GetBlockHeaderSize());
  return DUCKDB_BLOCK_START + static_cast<std::size_t>(block_id) * alloc +
         static_cast<std::size_t>(bm.GetBlockHeaderSize());
}

}  // namespace sirius::op::scan
