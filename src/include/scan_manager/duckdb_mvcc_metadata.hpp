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

#include <duckdb/common/typedefs.hpp>

#include <cstddef>
#include <numeric>
#include <vector>

namespace sirius::scan_manager {

/// MVCC snapshot metadata captured when a duckdb-native table is pinned, composed
/// onto @ref pinned_entry (absent for parquet pins, whose sources are immutable).
///
/// A duckdb-native pin decodes the table's last-checkpointed disk image MVCC-blind:
/// the cached data is the positionally-stable rowid prefix [0, n_cache()), including
/// rows already tombstoned in memory and excluding rows that live only in
/// uncheckpointed transient segments. Query-time delta merge reconciles that
/// snapshot against DuckDB's live MVCC state; these fields are the pin-side facts
/// it needs. Auto-checkpoint is suppressed while the pin exists so the disk image
/// (and therefore the positional base) cannot change underneath the cache; a
/// manual CHECKPOINT while pinned is outside the supported contract.
struct duckdb_mvcc_metadata {
  /// The pin transaction's DuckTransaction::start_time on the pinned table's own
  /// AttachedDatabase (not the default database — each attached database has its
  /// own MVCC counter domain). Serves as the snapshot fence: a query whose
  /// start_time is >= v_base sees every commit the pin saw, so the cached base
  /// filtered through query-time visibility is exact; an older query may still
  /// see rows the last checkpoint compacted out of the disk image, so it cannot
  /// be served from this cache.
  duckdb::transaction_t v_base{0};

  /// Physical row count of each pinned chunk, in materialization order — parallel
  /// to pinned_entry::chunk_memory_spaces (GPU tier) / host_chunks (HOST tier).
  /// Prefix sums map each chunk to its absolute rowid range [start, start + count).
  /// Each chunk is a contiguous run of whole row groups (validated at pin time),
  /// so per-chunk visibility masks can be assembled at row-group granularity.
  std::vector<std::size_t> base_row_count_per_chunk;

  /// Number of cached base rows: the pin covers rowids [0, n_cache()). Rows at
  /// [n_cache(), live table rows) exist only in DuckDB's in-memory transient
  /// segments — the insert delta.
  [[nodiscard]] std::size_t n_cache() const
  {
    return std::accumulate(
      base_row_count_per_chunk.begin(), base_row_count_per_chunk.end(), std::size_t{0});
  }
};

}  // namespace sirius::scan_manager
