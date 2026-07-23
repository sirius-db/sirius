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

#include <cucascade/memory/common.hpp>
#include <duckdb/common/typedefs.hpp>

#include <cstddef>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cudf {
class column;
}  // namespace cudf

namespace cucascade {
class host_data_representation;
namespace memory {
class memory_space;
}  // namespace memory
}  // namespace cucascade

namespace sirius::scan_manager {

/// npos in a decoded->entry column map means "decoded column is not cached; drop it".
inline constexpr std::size_t kDropDecodedColumn = static_cast<std::size_t>(-1);

/// A decoded, unmasked delta slice captured at the scan hook, waiting to be
/// applied into the pinned entry at QueryEnd. Holds whole closed row groups
/// (all rows visible at the promoting snapshot, so the bytes are committed and
/// immutable). GPU-tier slices own their device columns; HOST-tier slices own a
/// converted host chunk. The reservation is the admission claim held until apply
/// commits (then dropped — the allocation persists, tracked like pin memory) or
/// the slice is dropped.
struct promotion_captured_slice {
  std::size_t first_rowid{0};  ///< absolute rowid of the slice's first row (a base prefix end)
  std::size_t row_count{0};
  std::vector<duckdb::idx_t> row_group_indices;  ///< source row groups (logging / dedup)

  /// GPU tier: columns in entry cache order, each owning its device memory.
  std::vector<std::string> column_names;  ///< parallel to columns (entry order)
  std::vector<std::shared_ptr<cudf::column>> columns;
  cucascade::memory::memory_space* space{nullptr};

  /// HOST tier: the converted host chunk (columns empty).
  std::shared_ptr<cucascade::host_data_representation> host_chunk;

  /// Admission reservation, type-erased; released when this slice is applied or dropped.
  std::shared_ptr<void> reservation;
};

/// Thread-safe side channel between the concurrent decode hook (producers) and
/// the single QueryEnd apply (consumer). Entry mutation never happens here — the
/// hook only stashes decoded slices; apply drains them under the lifecycle lock.
class promotion_sink {
 public:
  /// Per-entry drained state.
  struct entry_capture {
    std::vector<promotion_captured_slice> slices;
    std::string last_skip_reason;
  };

  /// First-op-wins dedup: true (and records it) when (entry, first_row_group) is
  /// new this query; false when another operator's split already claimed that row
  /// group (a self-join decodes the same delta twice). Call before capturing.
  bool try_begin_capture(std::string const& entry_name, duckdb::idx_t first_row_group);

  /// Stash a captured slice for @p entry_name.
  void add(std::string const& entry_name, promotion_captured_slice slice);

  /// Record why a promotion was skipped (reservation failure, etc.).
  void record_skip(std::string const& entry_name, std::string reason);

  [[nodiscard]] bool empty() const;

  /// Move out everything captured this query, clearing the sink.
  std::unordered_map<std::string, entry_capture> take_all();

 private:
  mutable std::mutex mutex_;
  std::unordered_map<std::string, entry_capture> captures_;
  std::set<std::pair<std::string, duckdb::idx_t>> seen_;
};

/// Immutable per-(entry, query) promotion plan referenced from every promotable
/// split's ticket. Built at handoff once the carrier op's decode layout is known.
struct promotion_capture_plan {
  std::string entry_name;
  cucascade::memory::Tier tier{cucascade::memory::Tier::GPU};
  /// Cached column names in entry order; a promoted slice stores its columns in
  /// this order so they line up with pinned_entry::data_batches_by_column.
  std::vector<std::string> column_names;
  /// Maps each decoded column (carrier projection order) to its position in
  /// column_names, or kDropDecodedColumn to drop it.
  std::vector<std::size_t> entry_pos_by_decoded_pos;
  /// HOST tier: GPU device id -> NUMA-local host space for the D2H conversion.
  std::unordered_map<int, cucascade::memory::memory_space*> host_space_by_gpu;
  std::shared_ptr<promotion_sink> sink;
};

/// Ticket attached to a promotable delta split's scan info. Rides the existing
/// split -> batch -> operator-input path to the decode hook.
struct promotion_capture {
  std::shared_ptr<promotion_capture_plan const> plan;
  std::size_t first_rowid{0};
  std::size_t row_count{0};
  std::vector<duckdb::idx_t> row_group_indices;  ///< dedup key = front(); logging
};

/// Contiguity ratchet (pure, unit-testable, no GPU). Returns the maximal run of
/// @p slices that extends contiguously from @p n_cache, in rowid order; every
/// other slice is moved into @p dropped (self-corrects on a later query). A gap
/// stops the run even if later slices would qualify — the base must stay a
/// single unbroken rowid prefix.
std::vector<promotion_captured_slice> select_promotion_prefix(
  std::vector<promotion_captured_slice> slices,
  std::size_t n_cache,
  std::vector<promotion_captured_slice>& dropped);

}  // namespace sirius::scan_manager
