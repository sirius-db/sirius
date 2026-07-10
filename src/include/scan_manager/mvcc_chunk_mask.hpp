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
#include <memory>
#include <span>
#include <vector>

namespace sirius::scan_manager {

/**
 * @brief Bit-packed MVCC keep-mask for one cached chunk.
 *
 * One bit per row in cuDF's bitmask convention (bit `row % 32` of uint32 word
 * `row / 32`, 1 = keep), so the packed words upload and expand with
 * @c cudf::mask_to_bools without translation; padding bits past @ref
 * row_count are don't-care. @ref words points into storage owned via the
 * type-erased @ref retention (the mask job's pinned {reservation, blocks}
 * bundle; tests may retain a plain vector). Immutable once published.
 */
struct mvcc_chunk_mask {
  std::span<std::uint32_t> words;   ///< ceil(row_count / 32) packed keep bits
  std::size_t row_count{0};         ///< rows covered (== the chunk's row count)
  std::shared_ptr<void> retention;  ///< keeps the words' storage alive
};

/// Per-pinned-entry mask set for one query: slot i masks chunk i (the cached
/// provider's chunk order). A null slot means every row of that chunk is
/// visible — the chunk serves unmasked (no upload, no kernel).
using mvcc_chunk_mask_set = std::vector<std::shared_ptr<mvcc_chunk_mask const>>;

}  // namespace sirius::scan_manager
