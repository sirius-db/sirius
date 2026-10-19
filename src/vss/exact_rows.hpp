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

#include <cstddef>
#include <cstdint>

namespace sirius::vss {

/// Where the device reads FP32 row `row` of a vector column, one float at a time: either a copy
/// held in blocks of `rows_per_block` whole rows (`copy_blocks`), or the pinned column itself.
///
/// In the pin, row `row_map[row]` (the identity when `row_map` is null) lies in chunk c, the last
/// with `chunk_rows[c] <= it`; the chunk's floats start at byte `chunk_offset[c]` of the run of
/// blocks `pin_blocks[chunk_block[c]]`, `pin_blocks[chunk_block[c] + 1]`, ..., each
/// `1 << block_shift` bytes. A row may cross from one block into the next; a float never does,
/// since blocks are a power of two and floats start 4-byte aligned. A GPU-tier chunk is one block
/// that nothing crosses.
struct exact_rows {
  float const* const* copy_blocks{nullptr};
  std::int64_t rows_per_block{0};

  std::int64_t const* row_map{nullptr};
  std::int64_t const* chunk_rows{nullptr};  ///< n_chunks + 1 entries
  std::int64_t const* chunk_block{nullptr};
  std::int64_t const* chunk_offset{nullptr};
  std::byte const* const* pin_blocks{nullptr};
  std::int64_t n_chunks{0};
  std::int64_t n_rows{0};  ///< rows in the pin
  int block_shift{0};

  [[nodiscard]] bool valid() const noexcept
  {
    return copy_blocks != nullptr || pin_blocks != nullptr;
  }
};

}  // namespace sirius::vss
