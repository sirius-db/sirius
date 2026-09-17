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
/// TableFilter::Equals is not parameter-complete for every subclass (the base
/// compares only filter_type), so only types with known-complete overrides are
/// trusted, structural wrappers are recursed by hand, and anything else is
/// never key-equal. A refusal only costs a product-cache miss.
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

/// Extract statistics for @p columns from the row-group handles pinned by a
/// validated capture, so geometry and statistics describe the same state.
/// RowGroup::GetStatistics locks internally and returns a copy, so this
/// parallelizes across row groups.
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
  // Uncommitted rows in this transaction must never leak into a shared snapshot.
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

  // Capture the live structure once under the segment-tree lock and reject the
  // capture as torn unless it is internally consistent (contiguous starts, sum
  // of counts == total_rows, total_rows stable). Concurrent commits mutate this
  // state non-atomically, and their windows are microseconds wide, so retry a
  // few times before bypassing.
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
    // Strong handles live only for this acquire; the snapshot keeps weak_ptrs.
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
    // Torn across every retry: bypass to the uncached walk, install nothing.
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
      // An expired weak_ptr never compares equal to a live node.
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

  // pinned_row_groups matches entry.core node-for-node in both branches.
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
  // The snapshot this product was assembled from has been replaced.
  if (entry.generation != generation) { return; }
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
