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
/// group exposed no statistics. `BaseStatistics` copies are self-contained value
/// objects, so the snapshot holds no references into DuckDB storage.
struct column_stats_snapshot {
  std::vector<duckdb::unique_ptr<duckdb::BaseStatistics>> per_row_group;
};

/// Immutable structural snapshot of a table's row groups: everything the serial
/// `prepare_duckdb_native_walk` derives from `GetPartitionStats` that does not
/// depend on the query (projection / filters).
struct table_walk_snapshot {
  std::size_t n_row_groups = 0;
  std::size_t block_size   = 0;
  duckdb::idx_t total_rows = 0;
  /// Per-row-group identity, ABA-safe: a weak_ptr can only lock back to the
  /// object it was created from, so a freed-and-reallocated RowGroup at the
  /// same address never validates. Never dereferenced beyond the identity
  /// compare; holding weak references also never extends storage lifetime
  /// (safe across DETACH).
  std::vector<duckdb::weak_ptr<duckdb::RowGroup>> row_group_identity;
  std::vector<duckdb::idx_t> row_group_start;
  std::vector<duckdb::idx_t> row_count;
};

/// The query-dependent products of the prepare walk: the filter-statistics
/// pruning decisions and the varchar-overflow verdict. Everything ELSE the
/// walk plan carries comes from @ref table_walk_snapshot. Immutable once
/// published (shared as `shared_ptr<const>`).
struct walk_plan_product {
  bool viable = false;
  std::string viability_failure_reason;
  std::vector<bool> row_group_pruned_by_stats;
  std::vector<std::size_t> pruned_decoded_bytes_by_row_group;
  std::size_t pruned_row_groups    = 0;
  std::size_t pruned_decoded_bytes = 0;
};

/// Owning product-cache key: the projected column set (identity + type,
/// serialized to a canonical string) and deep COPIES of the prunable pushed-down
/// filters, resolved to storage primary indexes and sorted by them (the
/// TableFilterSet map iteration order is already deterministic, but resolution
/// through column_ids is not guaranteed order-stable across plans).
struct walk_product_key {
  std::string projection_signature;
  std::vector<std::pair<duckdb::idx_t, duckdb::unique_ptr<duckdb::TableFilter>>> prunable_filters;
};

/// Borrowed view of a product key for lookup (no filter copies made unless the
/// caller decides to store).
struct walk_product_key_view {
  const std::string* projection_signature = nullptr;
  const std::vector<std::pair<duckdb::idx_t, const duckdb::TableFilter*>>* prunable_filters =
    nullptr;
};

/**
 * @brief Process-wide memoization of the duckdb-native metadata prepare walk.
 *
 * The serial row-group walk in `prepare_duckdb_native_walk` (GetPartitionStats
 * + per-row-group column statistics for filter pruning and the varchar overflow
 * refusal) costs 70-190 ms per ~49k-row-group table at SF1000 and is recomputed
 * on every query of an UNPINNED table, on the query thread, while the
 * query-lifecycle slot is held. Its inputs only change when the table's
 * physical row-group structure changes, so this cache keys the result on the
 * table and validates it against the live segment tree on every acquire.
 *
 * Two layers:
 *  - The STRUCTURAL SNAPSHOT (geometry + per-column statistics) is
 *    query-independent and revalidated by a locked identity probe on every
 *    acquire (see below).
 *  - The WALK PRODUCT (pruning decisions + overflow verdict) additionally
 *    depends on the projected column set and the pushed-down filters; products
 *    are cached per (projection signature, prunable-filter set) under the same
 *    entry, dropped whenever the snapshot rebuilds, and only ever served
 *    against the exact snapshot generation they were assembled from. A product
 *    hit converts the whole prepare to O(probe): no statistics are re-checked.
 *
 * Validity model (probe, not version counter): an acquire walks the live
 * `SegmentNodes()` once and compares, per row group, the weak identity, row
 * start and physical row count, plus the collection total. This catches
 * everything that alters the walk output:
 *  - INSERT commits / optimistic appends: counts or node set change.
 *  - CHECKPOINT / vacuum: row groups are rewritten into new objects (weak
 *    identity fails) or merged (node set changes).
 *  - DROP+CREATE / ALTER: new collection, new row groups.
 * DELETE commits change only row-version state, which the walk output does not
 * depend on (visibility is applied by the MVCC keep-mask / delta machinery, or
 * is knowingly not applied on the MVCC-blind disk-native read), so a delete
 * does not invalidate — by design. UPDATE commits widen row-group statistics
 * in place without touching base segments; cached (narrower) statistics remain
 * a superset of the base data actually decoded, so cached pruning stays sound.
 * An O(1) commit-counter check (the mvcc_mask_snapshot_key discipline) was
 * considered and rejected as the sole validity test: CHECKPOINT restructures
 * row groups without bumping DuckTransactionManager::GetLastCommit, so only
 * the identity probe is structurally sound.
 *
 * Transaction-local appends (uncommitted rows in this transaction's
 * LocalStorage) are per-transaction state that must not leak across queries:
 * acquires bypass the cache entirely while any exist for the table.
 *
 * Concurrent commits (torn-capture rejection): DuckDB commits from other
 * connections mutate the probed state non-atomically — FinalizeAppend bumps
 * row-group counts in order without the tree lock, MergeStorage appends nodes
 * one at a time, and total_rows is updated last in every path. The capture
 * therefore runs single-pass under the segment-tree lock and is REJECTED
 * unless it is internally consistent: contiguous row-group starts, Σ
 * row_count == total_rows, and total_rows stable across the walk. A torn
 * capture is retried a few times first (commit mutation windows are
 * microseconds wide, so a retry almost always lands on a settled state); a
 * capture still torn after the retries bypasses to the uncached walk with
 * nothing installed or served, and logs the rejection at INFO so racing a
 * concurrent commit is observable in production logs. Only settled pre-/post-
 * commit states pass, so a snapshot describing geometry that never existed at
 * a commit boundary can never enter the cache. Column statistics are extracted
 * from the SAME validated capture's strong row-group handles (never by
 * re-walking the live tree), so geometry and statistics always describe one
 * point-in-time state; concurrent stats merges can only widen a row group's
 * statistics, which keeps pruning and the varchar overflow refusal
 * conservative.
 *
 * Thread safety: one mutex guards the registry; concurrent scan constructions
 * (7 streams) serialize on it, which also dedupes identical rebuilds. Returned
 * snapshots and products are immutable and shared (`shared_ptr<const>`); the
 * shared BaseStatistics objects are only read through them (TableFilter::
 * CheckStatistics takes a non-const ref but every implementation is
 * read-only). Product stores carry the generation of the snapshot they were
 * assembled from and are dropped, not installed, if the entry has since
 * rebuilt — a product can never describe a different geometry than the
 * snapshot it is served with.
 *
 * Kill switch: env `SIRIUS_DISABLE_NATIVE_METADATA_CACHE` (set and not "0")
 * forces every acquire to bypass, restoring the uncached walk.
 */
class duckdb_native_metadata_cache {
 public:
  struct acquired_snapshot {
    std::shared_ptr<const table_walk_snapshot> core;
    /// Statistics for the requested columns, keyed by storage primary index.
    std::unordered_map<duckdb::idx_t, std::shared_ptr<const column_stats_snapshot>> column_stats;
    /// Non-null on a product hit: the cached pruning/overflow product for the
    /// product key passed to acquire(), assembled from exactly `core`.
    std::shared_ptr<const walk_plan_product> product;
    /// Snapshot generation of `core`; pass back to store_product so a product
    /// assembled from this snapshot is never installed against a newer one.
    std::uint64_t generation = 0;
  };

  /// @brief Return a validated snapshot for @p storage carrying statistics for
  /// @p stats_columns (storage primary indexes), building or refreshing as
  /// needed. When @p product_key is non-null, also look up a cached walk
  /// product for that key. Returns nullopt when the cache must be bypassed
  /// (disabled, transaction-local appends present, or the capture stayed torn
  /// across a concurrent commit) — the caller then runs the uncached walk.
  std::optional<acquired_snapshot> acquire(duckdb::DataTable& storage,
                                           duckdb::ClientContext& context,
                                           const std::vector<duckdb::idx_t>& stats_columns,
                                           const walk_product_key_view* product_key = nullptr);

  /// @brief Install an assembled walk product for @p storage under @p key.
  /// Dropped (not installed) when the entry's snapshot generation no longer
  /// matches @p generation — the snapshot the product describes is gone.
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

  /// Dropped tables leave dead entries behind (keyed by a dead DataTable*);
  /// bound the registry so they cannot accumulate without limit.
  static constexpr std::size_t kMaxEntries = 256;
  /// Distinct (projection, filter) shapes cached per table. TPC-H style
  /// workloads see a handful per table; varied-predicate runs churn the LRU
  /// tail without evicting the snapshot or its statistics.
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
