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

#include <duckdb/common/shared_ptr.hpp>
#include <duckdb/common/types.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/planner/table_filter.hpp>
#include <duckdb/storage/data_table.hpp>
#include <duckdb/storage/statistics/base_statistics.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace duckdb {
class RowGroup;
}

namespace sirius::op::scan {

/// One column's row-group statistics captured at snapshot time. Entry i belongs
/// to row group i of the owning @ref table_walk_snapshot; null where the row
/// group exposed no statistics.
struct column_stats_snapshot {
  std::vector<duckdb::unique_ptr<duckdb::BaseStatistics>> per_row_group;
};

/// Immutable, query-independent snapshot of a table's row-group structure.
struct table_walk_snapshot {
  std::size_t n_row_groups = 0;
  std::size_t block_size   = 0;
  duckdb::idx_t total_rows = 0;
  /// Per-row-group identity. A weak_ptr only locks back to the object it was
  /// created from, so a freed-and-reallocated RowGroup never validates.
  std::vector<duckdb::weak_ptr<duckdb::RowGroup>> row_group_identity;
  std::vector<duckdb::idx_t> row_group_start;
  std::vector<duckdb::idx_t> row_count;
};

/// The query-dependent output of the prepare walk: filter-statistics pruning
/// decisions and the varchar-overflow verdict. Immutable once published.
struct walk_plan_product {
  bool viable = false;
  std::string viability_failure_reason;
  std::vector<bool> row_group_pruned_by_stats;
  std::vector<std::size_t> pruned_decoded_bytes_by_row_group;
  std::size_t pruned_row_groups    = 0;
  std::size_t pruned_decoded_bytes = 0;
};

/// Owning product-cache key: the projection signature and deep copies of the
/// prunable pushed-down filters, resolved to storage primary indexes and
/// sorted by them.
struct walk_product_key {
  std::string projection_signature;
  std::vector<std::pair<duckdb::idx_t, duckdb::unique_ptr<duckdb::TableFilter>>> prunable_filters;
};

/// Borrowed view of a product key for lookup.
struct walk_product_key_view {
  const std::string* projection_signature = nullptr;
  const std::vector<std::pair<duckdb::idx_t, const duckdb::TableFilter*>>* prunable_filters =
    nullptr;
};

/**
 * @brief Process-wide memoization of the duckdb-native metadata prepare walk.
 *
 * The walk (GetPartitionStats plus per-row-group column statistics) is
 * recomputed on every query of an unpinned table. Its inputs only change when
 * the table's physical row-group structure changes, so it is cached per table
 * in two layers:
 *  - The structural snapshot (geometry + per-column statistics) is
 *    query-independent and revalidated against the live segment tree on
 *    every acquire.
 *  - The walk product (pruning decisions + overflow verdict) is cached per
 *    (projection signature, prunable-filter set), dropped whenever the
 *    snapshot rebuilds, and only served against the snapshot generation it
 *    was assembled from.
 *
 * Validity is an identity probe, not a commit counter: CHECKPOINT rewrites row
 * groups without bumping the last-commit id, so only the probe is sound.
 * DELETE commits do not invalidate (visibility is applied downstream, not by
 * the walk). Transaction-local appends bypass the cache. Because concurrent
 * commits mutate the tree non-atomically, the capture runs once under the
 * segment-tree lock and is rejected as torn unless internally consistent
 * (contiguous starts, sum of counts == total_rows, stable total_rows), with a
 * few retries before falling back to the uncached walk.
 *
 * One mutex guards the registry. Returned snapshots and products are
 * immutable and shared. Env `SIRIUS_DISABLE_NATIVE_METADATA_CACHE` (set and
 * not "0") forces every acquire to bypass.
 */
class duckdb_native_metadata_cache {
 public:
  struct acquired_snapshot {
    std::shared_ptr<const table_walk_snapshot> core;
    /// Statistics for the requested columns, keyed by storage primary index.
    std::unordered_map<duckdb::idx_t, std::shared_ptr<const column_stats_snapshot>> column_stats;
    /// Non-null on a product hit for the key passed to acquire().
    std::shared_ptr<const walk_plan_product> product;
    /// Snapshot generation of `core`; pass back to store_product.
    std::uint64_t generation = 0;
  };

  /// @brief Return a validated snapshot for @p storage with statistics for
  /// @p stats_columns, building or refreshing as needed, and look up a cached
  /// product when @p product_key is non-null. Returns nullopt when the cache
  /// must be bypassed; the caller then runs the uncached walk.
  std::optional<acquired_snapshot> acquire(duckdb::DataTable& storage,
                                           duckdb::ClientContext& context,
                                           const std::vector<duckdb::idx_t>& stats_columns,
                                           const walk_product_key_view* product_key = nullptr);

  /// @brief Install a walk product for @p storage under @p key. Dropped when
  /// the entry's snapshot generation no longer matches @p generation.
  void store_product(duckdb::DataTable& storage,
                     std::uint64_t generation,
                     walk_product_key key,
                     std::shared_ptr<const walk_plan_product> product);

  //===----------Testing / diagnostics----------===//
  void clear();
  /// Test-only override on top of the env kill switch.
  void set_disabled_for_testing(bool disabled) { _disabled_for_testing.store(disabled); }
  [[nodiscard]] std::uint64_t hits() const { return _hits.load(); }
  [[nodiscard]] std::uint64_t rebuilds() const { return _rebuilds.load(); }
  [[nodiscard]] std::uint64_t bypasses() const { return _bypasses.load(); }
  [[nodiscard]] std::uint64_t product_hits() const { return _product_hits.load(); }

  static duckdb_native_metadata_cache& instance();

 private:
  struct product_slot {
    walk_product_key key;
    std::shared_ptr<const walk_plan_product> product;
    std::uint64_t last_used = 0;
  };

  struct cache_entry {
    std::shared_ptr<const table_walk_snapshot> core;
    std::unordered_map<duckdb::idx_t, std::shared_ptr<const column_stats_snapshot>> columns;
    std::vector<product_slot> products;
    std::uint64_t generation = 0;
    std::uint64_t last_used  = 0;
  };

  /// Bounds dead entries left behind by dropped tables.
  static constexpr std::size_t kMaxEntries = 256;
  /// Distinct (projection, filter) shapes cached per table (LRU).
  static constexpr std::size_t kMaxProductsPerEntry = 16;

  std::mutex _mutex;
  std::unordered_map<duckdb::DataTable const*, cache_entry> _entries;
  std::uint64_t _use_clock = 0;

  std::atomic<bool> _disabled_for_testing{false};
  std::atomic<std::uint64_t> _hits{0};
  std::atomic<std::uint64_t> _rebuilds{0};
  std::atomic<std::uint64_t> _bypasses{0};
  std::atomic<std::uint64_t> _product_hits{0};
};

}  // namespace sirius::op::scan
