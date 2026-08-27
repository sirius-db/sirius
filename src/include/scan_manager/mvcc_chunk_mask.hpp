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

#include <algorithm>
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
 * row_count are don't-care. @ref words aliases the mask job's pinned
 * {reservation, blocks} bundle or its pageable fallback vector — the pointer
 * IS the owner, so copies are a refcount bump and a mask's words cannot
 * dangle. The mask job installs it at carve time and its fill tasks write
 * the bits; nothing writes once run_mvcc_mask_jobs returns (providers take
 * their copies only after). A default-constructed mask means every row is
 * visible.
 */
struct mvcc_chunk_mask {
  /// This chunk's first word; owns the backing storage via the aliasing ctor.
  std::shared_ptr<std::uint32_t[]> words;
  std::size_t row_count{0};  ///< rows covered (== the chunk's row count)

  /// False = every row visible (the chunk serves unmasked: no upload, no kernel).
  [[nodiscard]] bool has_mask() const { return words != nullptr; }

  /// The ceil(row_count / 32) packed keep words. Only meaningful under has_mask().
  [[nodiscard]] std::span<const std::uint32_t> view() const
  {
    return {words.get(), (row_count + 31) / 32};
  }
};

/// Per-pinned-entry mask set for one query: slot i masks chunk i (the cached
/// provider's chunk order). A default-constructed slot means every row of
/// that chunk is visible. Owned by value at each stage — the mask job's
/// request fills it, then each cached provider takes its own copy (words are
/// shared through the masks' owning pointers, never duplicated).
using mvcc_chunk_mask_set = std::vector<mvcc_chunk_mask>;

/// Whether any slot actually masks a row. A duckdb pin's set is sized to the chunk count at
/// request creation and never resized (see mvcc_mask_job.hpp), so `masks.empty()` is false for
/// every duckdb pin even when the table has no deletions — checking size alone would refuse late
/// materialization for every duckdb-native scan unconditionally. This checks the slots
/// themselves instead: an unmodified table's slots are all default-constructed (has_mask()
/// false), so this reports false and the caller may treat the set as if it carried no mask.
[[nodiscard]] inline bool has_any_mask(mvcc_chunk_mask_set const& masks)
{
  return std::any_of(
    masks.begin(), masks.end(), [](mvcc_chunk_mask const& mask) { return mask.has_mask(); });
}

}  // namespace sirius::scan_manager
