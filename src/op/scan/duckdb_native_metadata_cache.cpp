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

#include "op/scan/duckdb_native_metadata_cache.hpp"

#include "log/logging.hpp"
#include "op/scan/metadata_walk_parallel.hpp"

#include <duckdb/main/attached_database.hpp>
#include <duckdb/planner/filter/conjunction_filter.hpp>
#include <duckdb/planner/filter/struct_filter.hpp>
#include <duckdb/storage/block_manager.hpp>
#include <duckdb/storage/storage_index.hpp>
#include <duckdb/storage/storage_manager.hpp>
#include <duckdb/storage/table/row_group.hpp>
#include <duckdb/storage/table/row_group_collection.hpp>
#include <duckdb/storage/table/row_group_segment_tree.hpp>
#include <duckdb/storage/table/segment_tree.hpp>
#include <duckdb/transaction/local_storage.hpp>

#include <algorithm>
#include <cstdlib>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace sirius::op::scan {

namespace {

bool cache_env_disabled()
{
  static bool const disabled = [] {
    char const* v = std::getenv("SIRIUS_DISABLE_NATIVE_METADATA_CACHE");
    return v != nullptr && std::string_view{v} != "0";
  }();
  return disabled;
}

/// @brief Key equality for the product cache's prunable filters.
///
/// TableFilter::Equals is NOT parameter-complete for every subclass: the base
/// implementation compares only filter_type, and OptionalFilter (among others)
/// inherits it — two OPTIONAL_FILTERs wrapping different children would compare
/// equal, which for a product key means serving one filter's pruning decisions
/// for another. So key equality trusts Equals only for the types whose
/// overrides are known parameter-complete (constant/value comparisons, IN
/// lists, deep expression equality), recurses through the structural wrappers
/// itself, and refuses everything else (DYNAMIC_FILTER never reaches here —
/// callers key prunable filters only; BLOOM_FILTER's Equals is already
/// always-false because its payload is run-time built). A refusal only costs a
/// product-cache miss: the snapshot and statistics layers still hit.
bool product_filter_key_equal(const duckdb::TableFilter& a, const duckdb::TableFilter& b)
{
  if (a.filter_type != b.filter_type) { return false; }
  switch (a.filter_type) {
    case duckdb::TableFilterType::CONSTANT_COMPARISON:
    case duckdb::TableFilterType::IN_FILTER:
    case duckdb::TableFilterType::EXPRESSION_FILTER: return a.Equals(b);
    case duckdb::TableFilterType::IS_NULL:
    case duckdb::TableFilterType::IS_NOT_NULL: return true;  // parameterless
    case duckdb::TableFilterType::CONJUNCTION_AND: {
      auto const& ca = a.Cast<duckdb::ConjunctionAndFilter>();
      auto const& cb = b.Cast<duckdb::ConjunctionAndFilter>();
      if (ca.child_filters.size() != cb.child_filters.size()) { return false; }
      for (std::size_t i = 0; i < ca.child_filters.size(); ++i) {
        if (!product_filter_key_equal(*ca.child_filters[i], *cb.child_filters[i])) { return false; }
      }
      return true;
    }
    case duckdb::TableFilterType::CONJUNCTION_OR: {
      auto const& ca = a.Cast<duckdb::ConjunctionOrFilter>();
      auto const& cb = b.Cast<duckdb::ConjunctionOrFilter>();
      if (ca.child_filters.size() != cb.child_filters.size()) { return false; }
      for (std::size_t i = 0; i < ca.child_filters.size(); ++i) {
        if (!product_filter_key_equal(*ca.child_filters[i], *cb.child_filters[i])) { return false; }
      }
      return true;
    }
    case duckdb::TableFilterType::STRUCT_EXTRACT: {
      auto const& sa = a.Cast<duckdb::StructFilter>();
      auto const& sb = b.Cast<duckdb::StructFilter>();
      return sa.child_idx == sb.child_idx &&
             product_filter_key_equal(*sa.child_filter, *sb.child_filter);
    }
    default: return false;  // unknown / runtime-mutable payloads: never key-equal
  }
}

bool product_key_matches(const walk_product_key& stored, const walk_product_key_view& query)
{
  if (query.projection_signature == nullptr || query.prunable_filters == nullptr) { return false; }
  if (stored.projection_signature != *query.projection_signature) { return false; }
  if (stored.prunable_filters.size() != query.prunable_filters->size()) { return false; }
  for (std::size_t i = 0; i < stored.prunable_filters.size(); ++i) {
    auto const& [stored_col, stored_filter] = stored.prunable_filters[i];
    auto const& [query_col, query_filter]   = (*query.prunable_filters)[i];
    if (stored_col != query_col) { return false; }
    if (stored_filter == nullptr || query_filter == nullptr) { return false; }
    if (!product_filter_key_equal(*stored_filter, *query_filter)) { return false; }
  }
  return true;
}

/// Extract statistics for @p columns from the strong row-group handles this
/// acquire's VALIDATED capture pinned (parallel to the snapshot's row groups).
/// Reading from the handles instead of re-iterating the live tree removes the
/// probe-to-extraction race window entirely: the handles cannot be freed, and
/// they are exactly the row groups the returned geometry describes.
/// RowGroup::GetStatistics locks internally (per-column stats_lock; lazy column
/// loads under the per-row-group row_group_lock) and returns a self-contained
/// copy, so the extraction parallelizes across row groups; a concurrent commit
/// can only WIDEN stats, which keeps cached pruning and the varchar overflow
/// refusal conservative.
std::vector<std::shared_ptr<column_stats_snapshot>> extract_column_stats(
  std::span<duckdb::shared_ptr<duckdb::RowGroup> const> row_groups,
  std::span<duckdb::idx_t const> columns)
{
  std::vector<std::shared_ptr<column_stats_snapshot>> snaps;
  snaps.reserve(columns.size());
  for (std::size_t c = 0; c < columns.size(); ++c) {
    snaps.push_back(std::make_shared<column_stats_snapshot>());
    snaps.back()->per_row_group.resize(row_groups.size());
  }
  parallel_over_row_groups(row_groups.size(), [&](std::size_t begin, std::size_t end) {
    for (std::size_t rg = begin; rg < end; ++rg) {
      for (std::size_t c = 0; c < columns.size(); ++c) {
        duckdb::StorageIndex const storage_idx(columns[c]);
        snaps[c]->per_row_group[rg] = row_groups[rg]->GetStatistics(storage_idx);
      }
    }
  });
  return snaps;
}

}  // namespace

duckdb_native_metadata_cache& duckdb_native_metadata_cache::instance()
{
  static duckdb_native_metadata_cache cache;
  return cache;
}

void duckdb_native_metadata_cache::clear()
{
  std::lock_guard<std::mutex> guard(_mutex);
  _entries.clear();
  _hits.store(0);
  _rebuilds.store(0);
  _bypasses.store(0);
  _product_hits.store(0);
}

std::optional<duckdb_native_metadata_cache::acquired_snapshot>
duckdb_native_metadata_cache::acquire(duckdb::DataTable& storage,
                                      duckdb::ClientContext& context,
                                      const std::vector<duckdb::idx_t>& stats_columns,
                                      const walk_product_key_view* product_key)
{
  if (cache_env_disabled() || _disabled_for_testing.load()) {
    ++_bypasses;
    return std::nullopt;
  }
  // Transaction-local appends are per-transaction state that GetPartitionStats
  // folds into its result; a snapshot of them must never leak across queries.
  if (duckdb::LocalStorage::Get(context, storage.GetAttached()).GetStorage(storage)) {
    ++_bypasses;
    return std::nullopt;
  }
  auto const& collection = storage.GetRowGroupCollection();
  if (!collection) {
    ++_bypasses;
    return std::nullopt;
  }
  auto tree = collection->GetRowGroups();
  if (!tree) {
    ++_bypasses;
    return std::nullopt;
  }

  // Probe pass: capture the live structure ONCE, under the segment-tree lock,
  // and validate the capture's internal consistency before it can be installed
  // or served. Commits from other connections mutate this state non-atomically
  // (RowGroupCollection::FinalizeAppend bumps each covered row group's `count`
  // in order WITHOUT the tree lock; MergeStorage appends nodes one AppendSegment
  // at a time; both update `total_rows` LAST), so an unvalidated capture racing
  // a commit can observe geometry that never existed at any commit boundary.
  // The tree lock serializes against node insertion/erasure; the checks below
  // reject every remaining mid-commit state:
  //  - contiguity: row_group_start[i] == row_group_start[0] + Σ row_count[<i].
  //    Node starts are assigned final at insertion, so a not-yet-final tail
  //    count (FinalizeAppend mid-loop) breaks the chain to its successor.
  //  - accounting: Σ row_count == total_rows. Counts are bumped BEFORE
  //    total_rows in every append path (and nodes erased before total_rows in
  //    the revert path), so any partially-applied commit fails one side.
  //  - stability: total_rows re-read after the walk must equal the value read
  //    before it (a commit fully landing mid-walk moves it).
  // Consistent captures are exactly the settled pre-/post-commit states, both
  // of which are legitimate to serve (visibility of committed-but-invisible
  // rows is applied downstream by the MVCC keep-mask / insert-delta / plan-gate
  // machinery, identical to the uncached walk's physical counts).
  // A commit's mutation window (count bumps / node splices) is microseconds
  // wide, so a torn capture almost always settles by the next attempt. Retry
  // the locked capture a few times before giving up: every serve during a
  // commit window then still comes from a VALIDATED capture instead of pushing
  // the query onto the uncached GetPartitionStats walk, which reads the same
  // live state with no validation at all.
  constexpr int kTornCaptureAttempts = 4;

  std::shared_ptr<table_walk_snapshot> live;
  std::vector<duckdb::shared_ptr<duckdb::RowGroup>> pinned_row_groups;
  bool consistent            = false;
  duckdb::idx_t total_before = 0;
  duckdb::idx_t running      = 0;
  int attempts_used          = 0;
  for (int attempt = 0; attempt < kTornCaptureAttempts && !consistent; ++attempt) {
    attempts_used    = attempt + 1;
    total_before     = collection->GetTotalRows();
    live             = std::make_shared<table_walk_snapshot>();
    live->total_rows = total_before;
    live->block_size = storage.GetAttached().GetStorageManager().GetBlockManager().GetBlockSize();
    // Strong handles pinned for the duration of this acquire: identity source
    // for the probe AND the stats-extraction source (no second tree
    // iteration). Only weak_ptrs of these enter the cached snapshot, so entry
    // lifetime never extends DuckDB storage lifetime (safe across DETACH).
    pinned_row_groups.clear();
    {
      auto tree_lock = tree->Lock();
      for (auto& node : tree->SegmentNodes(tree_lock)) {
        pinned_row_groups.push_back(node.ReferenceNode());
        live->row_group_identity.emplace_back(pinned_row_groups.back());
        live->row_group_start.push_back(node.GetRowStart());
        live->row_count.push_back(pinned_row_groups.back()->count.load());
      }
    }
    live->n_row_groups = live->row_group_identity.size();

    running    = 0;
    consistent = true;
    for (std::size_t i = 0; i < live->n_row_groups; ++i) {
      if (live->row_group_start[i] != live->row_group_start[0] + running) {
        consistent = false;
        break;
      }
      running += live->row_count[i];
    }
    consistent =
      consistent && running == total_before && collection->GetTotalRows() == total_before;
  }
  if (!consistent) {
    // Mid-commit tear that outlasted every retry: do not install, do not
    // serve — the caller falls through to the uncached walk, and nothing is
    // memoized from this window. INFO (not DEBUG): this is the observable
    // trace that a query raced a staged-refresh commit here, and it is rare
    // enough (sub-Hz even under RF pressure) to be free.
    ++_bypasses;
    SIRIUS_LOG_INFO(
      "[duckdb_native_metadata_cache] torn capture rejected (concurrent commit, {} attempts): "
      "{} row group(s), sum(count)={}, total_rows={} [hits={} rebuilds={} bypasses={}]",
      kTornCaptureAttempts,
      live->n_row_groups,
      running,
      total_before,
      _hits.load(),
      _rebuilds.load(),
      _bypasses.load());
    return std::nullopt;
  }
  if (attempts_used > 1) {
    // A tear was detected AND settled by a retry — the common case. Logged at
    // INFO so guard activity is countable in benchmark logs (rejections alone
    // undercount it: most mid-commit captures settle within one retry).
    SIRIUS_LOG_INFO(
      "[duckdb_native_metadata_cache] torn capture settled on retry {} of {}: "
      "{} row group(s), total_rows={}",
      attempts_used,
      kTornCaptureAttempts,
      live->n_row_groups,
      total_before);
  }

  std::lock_guard<std::mutex> guard(_mutex);

  auto& entry     = _entries[&storage];
  entry.last_used = ++_use_clock;

  bool same = entry.core != nullptr && entry.core->total_rows == live->total_rows &&
              entry.core->n_row_groups == live->n_row_groups &&
              entry.core->block_size == live->block_size &&
              entry.core->row_group_start == live->row_group_start &&
              entry.core->row_count == live->row_count;
  if (same) {
    for (std::size_t i = 0; i < live->n_row_groups; ++i) {
      // ABA-safe identity: an expired weak_ptr (row group freed, address
      // possibly reused) never compares equal to a live node — and this
      // acquire's strong handles keep the live nodes alive for the compare.
      auto cached = entry.core->row_group_identity[i].lock();
      if (!cached || cached.get() != pinned_row_groups[i].get()) {
        same = false;
        break;
      }
    }
  }

  if (!same) {
    entry.core = std::shared_ptr<const table_walk_snapshot>(std::move(live));
    entry.columns.clear();
    entry.products.clear();  // products describe the previous geometry
    ++entry.generation;
    ++_rebuilds;
    SIRIUS_LOG_DEBUG("[duckdb_native_metadata_cache] snapshot rebuilt: {} row group(s), {} row(s)",
                     entry.core->n_row_groups,
                     entry.core->total_rows);
  } else {
    ++_hits;
  }

  acquired_snapshot out;
  out.core       = entry.core;
  out.generation = entry.generation;

  // Missing-column statistics are extracted in ONE parallel pass over the
  // pinned handles (previously one serial pass per column).
  // pinned_row_groups is parallel to entry.core's row groups in both branches:
  // on a rebuild entry.core IS this capture; on a hit the identity loop above
  // verified node-for-node equality. Extracting from the pinned handles (not
  // the live tree) closes the old probe-to-extraction race window.
  std::vector<duckdb::idx_t> missing;
  for (auto const column : stats_columns) {
    if (entry.columns.find(column) == entry.columns.end()) { missing.push_back(column); }
  }
  if (!missing.empty()) {
    auto extracted = extract_column_stats(pinned_row_groups, missing);
    for (std::size_t c = 0; c < missing.size(); ++c) {
      entry.columns.emplace(missing[c], std::move(extracted[c]));
    }
  }
  for (auto const column : stats_columns) {
    out.column_stats.emplace(column, entry.columns.at(column));
  }

  // Product lookup: only against this entry's CURRENT snapshot (products were
  // dropped on rebuild above, so anything found describes out.core exactly).
  if (product_key != nullptr) {
    for (auto& slot : entry.products) {
      if (product_key_matches(slot.key, *product_key)) {
        slot.last_used = ++_use_clock;
        out.product    = slot.product;
        ++_product_hits;
        break;
      }
    }
  }

  if (_entries.size() > kMaxEntries) {
    auto lru = std::min_element(_entries.begin(), _entries.end(), [](auto const& a, auto const& b) {
      return a.second.last_used < b.second.last_used;
    });
    if (lru != _entries.end() && lru->first != &storage) { _entries.erase(lru); }
  }

  return out;
}

void duckdb_native_metadata_cache::store_product(duckdb::DataTable& storage,
                                                 std::uint64_t generation,
                                                 walk_product_key key,
                                                 std::shared_ptr<const walk_plan_product> product)
{
  if (product == nullptr) { return; }
  std::lock_guard<std::mutex> guard(_mutex);
  auto it = _entries.find(&storage);
  if (it == _entries.end()) { return; }
  auto& entry = it->second;
  // The snapshot this product was assembled from is gone (a commit landed
  // between acquire and store): drop the product. Installing it would pair
  // old-geometry pruning decisions with the new snapshot on later hits.
  if (entry.generation != generation) { return; }
  // Idempotent under concurrent same-shape assembles: keep the incumbent.
  walk_product_key_view view;
  view.projection_signature = &key.projection_signature;
  std::vector<std::pair<duckdb::idx_t, const duckdb::TableFilter*>> borrowed;
  borrowed.reserve(key.prunable_filters.size());
  for (auto const& [col, filter] : key.prunable_filters) {
    borrowed.emplace_back(col, filter.get());
  }
  view.prunable_filters = &borrowed;
  for (auto const& slot : entry.products) {
    if (product_key_matches(slot.key, view)) { return; }
  }
  if (entry.products.size() >= kMaxProductsPerEntry) {
    auto lru = std::min_element(
      entry.products.begin(), entry.products.end(), [](auto const& a, auto const& b) {
        return a.last_used < b.last_used;
      });
    if (lru != entry.products.end()) { entry.products.erase(lru); }
  }
  entry.products.push_back(product_slot{std::move(key), std::move(product), ++_use_clock});
}

}  // namespace sirius::op::scan
