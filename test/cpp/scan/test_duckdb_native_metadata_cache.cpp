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

// GPU-free tests for the duckdb-native metadata prepare cache and the fused,
// parallel statistics pass: keying, invalidation on structural change
// (insert / checkpoint), non-invalidation on deletes, transaction-local
// bypass, cached-vs-uncached walk equivalence, walk-product (layer 2)
// hit/miss semantics, refusal-order equivalence with the old serial passes,
// parallel-walk determinism, and torn-capture rejection under concurrent
// commits.

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/catalog/catalog.hpp>
#include <duckdb/catalog/catalog_entry/duck_table_entry.hpp>
#include <duckdb/common/column_index.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/planner/filter/constant_filter.hpp>
#include <duckdb/planner/table_filter.hpp>
#include <duckdb/storage/data_table.hpp>
#include <op/scan/duckdb_native_metadata.hpp>
#include <op/scan/duckdb_native_metadata_cache.hpp>
#include <utils/utils.hpp>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

using namespace sirius;
using namespace sirius::op::scan;

namespace {

void exec_ok(duckdb::Connection& con, const std::string& q)
{
  auto result = con.Query(q);
  REQUIRE(result);
  if (result->HasError()) {
    INFO("query failed: " << q << "\n  error: " << result->GetError());
    REQUIRE_FALSE(result->HasError());
  }
}

// Catalog access requires an active transaction; the caller owns the
// transaction lifecycle (COMMIT + BEGIN between mutation phases).
duckdb::DataTable& get_storage_in_txn(duckdb::Connection& con, const std::string& table_name)
{
  auto& ctx     = *con.context;
  auto& catalog = duckdb::Catalog::GetCatalog(ctx, "");
  duckdb::CatalogTransaction txn(catalog, ctx);
  auto& schema = catalog.GetSchema(txn, "main");
  auto entry   = schema.GetEntry(txn, duckdb::CatalogType::TABLE_ENTRY, table_name);
  REQUIRE(entry);
  return entry->Cast<duckdb::DuckTableEntry>().GetStorage();
}

projected_column real_col(duckdb::idx_t col_id)
{
  projected_column pc;
  pc.storage_idx = duckdb::StorageIndex(col_id);
  pc.is_rowid    = false;
  return pc;
}

// One-column TableFilterSet keyed by the relative scan-column index `col_key`,
// plus the parallel column_ids mapping the key back to `storage_idx`.
struct filter_ctx {
  duckdb::TableFilterSet filters;
  duckdb::vector<duckdb::ColumnIndex> column_ids;
};

filter_ctx make_constant_filter(duckdb::idx_t col_key,
                                duckdb::idx_t storage_idx,
                                duckdb::ExpressionType cmp,
                                duckdb::Value constant)
{
  filter_ctx ctx;
  ctx.filters.filters[col_key] =
    duckdb::make_uniq<duckdb::ConstantFilter>(cmp, std::move(constant));
  ctx.column_ids.resize(col_key + 1, duckdb::ColumnIndex(storage_idx));
  ctx.column_ids[col_key] = duckdb::ColumnIndex(storage_idx);
  return ctx;
}

struct walk_result {
  std::vector<duckdb_row_group_metadata> row_groups;
  bool viable = false;
  std::string viability_failure_reason;
  std::size_t pruned_row_groups    = 0;
  std::size_t pruned_decoded_bytes = 0;
};

walk_result walk_all(duckdb::DataTable& storage,
                     duckdb::ClientContext& ctx,
                     const std::vector<projected_column>& cols,
                     const std::vector<sirius::logical_type>& types,
                     const duckdb::TableFilterSet* table_filters           = nullptr,
                     const duckdb::vector<duckdb::ColumnIndex>* column_ids = nullptr)
{
  auto plan = prepare_duckdb_native_walk(storage, ctx, cols, types, table_filters, column_ids);
  if (!plan.viable) {
    return {{},
            false,
            std::move(plan.viability_failure_reason),
            plan.pruned_row_groups,
            plan.pruned_decoded_bytes};
  }
  auto range = walk_duckdb_native_row_group_range(plan, 0, plan.n_row_groups);
  return {std::move(range.row_groups),
          range.viable,
          std::move(range.viability_failure_reason),
          range.pruned_row_groups,
          range.pruned_decoded_bytes};
}

// Structural equality of two full walk results, down to segment placement.
void require_same_walk(const walk_result& a, const walk_result& b)
{
  REQUIRE(a.viable == b.viable);
  REQUIRE(a.viability_failure_reason == b.viability_failure_reason);
  REQUIRE(a.pruned_row_groups == b.pruned_row_groups);
  REQUIRE(a.pruned_decoded_bytes == b.pruned_decoded_bytes);
  REQUIRE(a.row_groups.size() == b.row_groups.size());
  for (std::size_t i = 0; i < a.row_groups.size(); ++i) {
    auto const& ra = a.row_groups[i];
    auto const& rb = b.row_groups[i];
    REQUIRE(ra.row_group_index == rb.row_group_index);
    REQUIRE(ra.row_group_start == rb.row_group_start);
    REQUIRE(ra.row_count == rb.row_count);
    REQUIRE(ra.decoded_bytes_budget == rb.decoded_bytes_budget);
    REQUIRE(ra.columns.size() == rb.columns.size());
    for (std::size_t c = 0; c < ra.columns.size(); ++c) {
      auto const& ca = ra.columns[c];
      auto const& cb = rb.columns[c];
      REQUIRE(ca.column_id == cb.column_id);
      REQUIRE(ca.data_segments.size() == cb.data_segments.size());
      REQUIRE(ca.validity_segments.size() == cb.validity_segments.size());
      for (std::size_t s = 0; s < ca.data_segments.size(); ++s) {
        REQUIRE(ca.data_segments[s].block_id == cb.data_segments[s].block_id);
        REQUIRE(ca.data_segments[s].block_offset == cb.data_segments[s].block_offset);
        REQUIRE(ca.data_segments[s].segment_start == cb.data_segments[s].segment_start);
        REQUIRE(ca.data_segments[s].segment_count == cb.data_segments[s].segment_count);
        REQUIRE(ca.data_segments[s].compression == cb.data_segments[s].compression);
      }
    }
  }
}

// Restores the cache's testing-disable flag even when a REQUIRE throws.
struct cache_disable_guard {
  explicit cache_disable_guard(bool disabled)
  {
    duckdb_native_metadata_cache::instance().set_disabled_for_testing(disabled);
  }
  ~cache_disable_guard()
  {
    duckdb_native_metadata_cache::instance().set_disabled_for_testing(false);
  }
};

// Pins SIRIUS_METADATA_WALK_THREADS for a scope (the knob is re-read per walk).
struct walk_threads_guard {
  explicit walk_threads_guard(const char* value)
  {
    if (value == nullptr) {
      unsetenv("SIRIUS_METADATA_WALK_THREADS");
    } else {
      setenv("SIRIUS_METADATA_WALK_THREADS", value, 1);
    }
  }
  ~walk_threads_guard() { unsetenv("SIRIUS_METADATA_WALK_THREADS"); }
};

std::vector<sirius::logical_type> int_types(std::size_t n)
{
  return std::vector<sirius::logical_type>(n, sirius::logical_type::make(sirius::type_id::INTEGER));
}

std::vector<sirius::logical_type> varchar_types(std::size_t n)
{
  return std::vector<sirius::logical_type>(n, sirius::logical_type::make(sirius::type_id::VARCHAR));
}

}  // namespace

//===--------------------------------------------------------------------===//
// Hit / rebuild keying
//===--------------------------------------------------------------------===//

TEST_CASE("metadata cache hits on repeated identical walks", "[scan][duckdb_native_metadata_cache]")
{
  auto& cache = duckdb_native_metadata_cache::instance();
  cache.clear();
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t3c_hits(a INTEGER)");
  exec_ok(con, "INSERT INTO t3c_hits SELECT range FROM range(0, 300000)");
  exec_ok(con, "CHECKPOINT");
  exec_ok(con, "BEGIN TRANSACTION");
  auto& storage = get_storage_in_txn(con, "t3c_hits");

  std::vector<projected_column> cols = {real_col(0)};
  auto first                         = walk_all(storage, *con.context, cols, int_types(1));
  REQUIRE(first.viable);
  REQUIRE(cache.rebuilds() == 1);
  REQUIRE(cache.hits() == 0);

  auto second = walk_all(storage, *con.context, cols, int_types(1));
  REQUIRE(second.viable);
  REQUIRE(cache.rebuilds() == 1);
  REQUIRE(cache.hits() == 1);
  // Identical query shape: the second walk is served from the product cache
  // (no statistics re-checked at all).
  REQUIRE(cache.product_hits() == 1);
  require_same_walk(first, second);
}

TEST_CASE("metadata cache result matches the uncached walk", "[scan][duckdb_native_metadata_cache]")
{
  auto& cache = duckdb_native_metadata_cache::instance();
  cache.clear();
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t3c_equiv(a INTEGER, b INTEGER)");
  exec_ok(con, "INSERT INTO t3c_equiv SELECT range, range * 2 FROM range(0, 300000)");
  exec_ok(con, "CHECKPOINT");
  exec_ok(con, "BEGIN TRANSACTION");
  auto& storage = get_storage_in_txn(con, "t3c_equiv");

  std::vector<projected_column> cols = {real_col(0), real_col(1)};
  auto ctx                           = make_constant_filter(
    0, 0, duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO, duckdb::Value::INTEGER(250000));

  auto cached = walk_all(storage, *con.context, cols, int_types(2), &ctx.filters, &ctx.column_ids);
  REQUIRE(cached.viable);
  REQUIRE(cache.rebuilds() == 1);

  walk_result uncached;
  {
    cache_disable_guard guard(true);
    uncached = walk_all(storage, *con.context, cols, int_types(2), &ctx.filters, &ctx.column_ids);
  }
  require_same_walk(cached, uncached);
  // 300k monotonic rows span 3 row groups; a >= 250000 prunes the low two.
  REQUIRE(cached.pruned_row_groups == 2);
}

TEST_CASE("metadata cache serves varied filters from one snapshot",
          "[scan][duckdb_native_metadata_cache]")
{
  auto& cache = duckdb_native_metadata_cache::instance();
  cache.clear();
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t3c_filters(a INTEGER, b INTEGER)");
  exec_ok(con, "INSERT INTO t3c_filters SELECT range, range FROM range(0, 300000)");
  exec_ok(con, "CHECKPOINT");
  exec_ok(con, "BEGIN TRANSACTION");
  auto& storage = get_storage_in_txn(con, "t3c_filters");

  std::vector<projected_column> cols = {real_col(0), real_col(1)};

  // No filter: nothing pruned.
  auto base = walk_all(storage, *con.context, cols, int_types(2));
  REQUIRE(base.viable);
  REQUIRE(base.pruned_row_groups == 0);

  // a >= 250000: prunes the low row groups; served from the same snapshot.
  auto lower = make_constant_filter(
    0, 0, duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO, duckdb::Value::INTEGER(250000));
  auto pruned =
    walk_all(storage, *con.context, cols, int_types(2), &lower.filters, &lower.column_ids);
  REQUIRE(pruned.viable);
  REQUIRE(pruned.pruned_row_groups >= 1);
  REQUIRE(pruned.row_groups.size() + pruned.pruned_row_groups == base.row_groups.size());

  // A filter on the OTHER column extends the snapshot's stats lazily — still no
  // rebuild.
  auto other = make_constant_filter(
    0, 1, duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO, duckdb::Value::INTEGER(250000));
  auto pruned_b =
    walk_all(storage, *con.context, cols, int_types(2), &other.filters, &other.column_ids);
  REQUIRE(pruned_b.viable);
  REQUIRE(pruned_b.pruned_row_groups == pruned.pruned_row_groups);

  // a >= 1000000 exceeds every value: all row groups pruned; same shape as the
  // uncached walk (viable, coalescer emits the empty split downstream).
  auto over = make_constant_filter(
    0, 0, duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO, duckdb::Value::INTEGER(1000000));
  auto all_pruned =
    walk_all(storage, *con.context, cols, int_types(2), &over.filters, &over.column_ids);
  walk_result all_pruned_uncached;
  {
    cache_disable_guard guard(true);
    all_pruned_uncached =
      walk_all(storage, *con.context, cols, int_types(2), &over.filters, &over.column_ids);
  }
  require_same_walk(all_pruned, all_pruned_uncached);
  REQUIRE(all_pruned.pruned_row_groups == base.row_groups.size());

  REQUIRE(cache.rebuilds() == 1);
  REQUIRE(cache.hits() >= 3);
}

//===--------------------------------------------------------------------===//
// Walk-product (layer 2) semantics
//===--------------------------------------------------------------------===//

TEST_CASE("walk product cache serves repeated query shapes and misses on new predicates",
          "[scan][duckdb_native_metadata_cache]")
{
  auto& cache = duckdb_native_metadata_cache::instance();
  cache.clear();
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t3c_prod(a INTEGER, b INTEGER)");
  exec_ok(con, "INSERT INTO t3c_prod SELECT range, range FROM range(0, 300000)");
  exec_ok(con, "CHECKPOINT");
  exec_ok(con, "BEGIN TRANSACTION");
  auto& storage = get_storage_in_txn(con, "t3c_prod");

  std::vector<projected_column> cols = {real_col(0), real_col(1)};
  auto f1                            = make_constant_filter(
    0, 0, duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO, duckdb::Value::INTEGER(250000));

  auto first = walk_all(storage, *con.context, cols, int_types(2), &f1.filters, &f1.column_ids);
  REQUIRE(first.viable);
  REQUIRE(cache.product_hits() == 0);

  // Same shape, same constants (a fresh, equal filter OBJECT — Equals keying,
  // not pointer identity): product hit, identical output.
  auto f1_again = make_constant_filter(
    0, 0, duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO, duckdb::Value::INTEGER(250000));
  auto again =
    walk_all(storage, *con.context, cols, int_types(2), &f1_again.filters, &f1_again.column_ids);
  REQUIRE(cache.product_hits() == 1);
  require_same_walk(first, again);

  // Same shape, different constant (--vary-predicates): product miss, snapshot
  // hit, and the new predicate's own pruning decisions.
  auto f2 = make_constant_filter(
    0, 0, duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO, duckdb::Value::INTEGER(1));
  auto varied = walk_all(storage, *con.context, cols, int_types(2), &f2.filters, &f2.column_ids);
  REQUIRE(varied.viable);
  REQUIRE(cache.product_hits() == 1);  // unchanged: this walk assembled fresh
  REQUIRE(varied.pruned_row_groups == 0);
  REQUIRE(first.pruned_row_groups == 2);

  // Projection change: product miss too (pruned-byte estimates depend on it).
  std::vector<projected_column> cols_a = {real_col(0)};
  auto f1_proj                         = make_constant_filter(
    0, 0, duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO, duckdb::Value::INTEGER(250000));
  auto projected =
    walk_all(storage, *con.context, cols_a, int_types(1), &f1_proj.filters, &f1_proj.column_ids);
  REQUIRE(projected.viable);
  REQUIRE(cache.product_hits() == 1);
  REQUIRE(projected.pruned_row_groups == 2);
  REQUIRE(projected.pruned_decoded_bytes < first.pruned_decoded_bytes);

  REQUIRE(cache.rebuilds() == 1);  // one snapshot fed every shape
}

TEST_CASE("walk product cache drops products when the snapshot rebuilds",
          "[scan][duckdb_native_metadata_cache]")
{
  auto& cache = duckdb_native_metadata_cache::instance();
  cache.clear();
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t3c_prod_inv(a INTEGER)");
  exec_ok(con, "INSERT INTO t3c_prod_inv SELECT range FROM range(0, 300000)");
  exec_ok(con, "CHECKPOINT");
  exec_ok(con, "BEGIN TRANSACTION");
  auto& storage = get_storage_in_txn(con, "t3c_prod_inv");

  std::vector<projected_column> cols = {real_col(0)};
  auto f                             = make_constant_filter(
    0, 0, duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO, duckdb::Value::INTEGER(250000));

  auto before = walk_all(storage, *con.context, cols, int_types(1), &f.filters, &f.column_ids);
  REQUIRE(before.viable);
  auto served = walk_all(storage, *con.context, cols, int_types(1), &f.filters, &f.column_ids);
  REQUIRE(cache.product_hits() == 1);
  require_same_walk(before, served);

  // Commit an insert whose rows the OLD product would have pruned wrongly.
  exec_ok(con, "COMMIT");
  exec_ok(con, "INSERT INTO t3c_prod_inv SELECT range FROM range(300000, 400000)");
  exec_ok(con, "BEGIN TRANSACTION");

  auto after = walk_all(storage, *con.context, cols, int_types(1), &f.filters, &f.column_ids);
  REQUIRE(after.viable);
  REQUIRE(cache.rebuilds() == 2);
  REQUIRE(cache.product_hits() == 1);  // old product NOT served post-commit

  walk_result uncached;
  {
    cache_disable_guard guard(true);
    uncached = walk_all(storage, *con.context, cols, int_types(1), &f.filters, &f.column_ids);
  }
  require_same_walk(after, uncached);

  // The re-assembled product serves thereafter.
  auto again = walk_all(storage, *con.context, cols, int_types(1), &f.filters, &f.column_ids);
  REQUIRE(cache.product_hits() == 2);
  require_same_walk(after, again);
}

//===--------------------------------------------------------------------===//
// Invalidation
//===--------------------------------------------------------------------===//

TEST_CASE("metadata cache invalidates on committed insert", "[scan][duckdb_native_metadata_cache]")
{
  auto& cache = duckdb_native_metadata_cache::instance();
  cache.clear();
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t3c_insert(a INTEGER)");
  exec_ok(con, "INSERT INTO t3c_insert SELECT range FROM range(0, 300000)");
  exec_ok(con, "CHECKPOINT");
  exec_ok(con, "BEGIN TRANSACTION");
  auto& storage = get_storage_in_txn(con, "t3c_insert");

  std::vector<projected_column> cols = {real_col(0)};
  auto before                        = walk_all(storage, *con.context, cols, int_types(1));
  REQUIRE(before.viable);
  REQUIRE(cache.rebuilds() == 1);

  // RF1-shaped mutation: commit an insert from the same connection.
  exec_ok(con, "COMMIT");
  exec_ok(con, "INSERT INTO t3c_insert SELECT range FROM range(300000, 305000)");
  exec_ok(con, "BEGIN TRANSACTION");

  auto after = walk_all(storage, *con.context, cols, int_types(1));
  REQUIRE(after.viable);
  REQUIRE(cache.rebuilds() == 2);  // recomputed ONCE after the commit

  duckdb::idx_t rows_before = 0;
  duckdb::idx_t rows_after  = 0;
  for (auto const& rg : before.row_groups) {
    rows_before += rg.row_count;
  }
  for (auto const& rg : after.row_groups) {
    rows_after += rg.row_count;
  }
  REQUIRE(rows_before == 300000);
  REQUIRE(rows_after == 305000);

  // ... and reused thereafter.
  auto again = walk_all(storage, *con.context, cols, int_types(1));
  REQUIRE(cache.rebuilds() == 2);
  require_same_walk(after, again);
}

TEST_CASE("metadata cache stays valid across committed deletes",
          "[scan][duckdb_native_metadata_cache]")
{
  // A DELETE commit changes only row-version state; the physical row-group
  // structure (starts, counts, segments) the walk describes is untouched, so
  // the snapshot must keep serving. Visibility of deleted rows is applied by
  // the MVCC keep-mask machinery, never by this walk.
  auto& cache = duckdb_native_metadata_cache::instance();
  cache.clear();
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t3c_delete(a INTEGER)");
  exec_ok(con, "INSERT INTO t3c_delete SELECT range FROM range(0, 300000)");
  exec_ok(con, "CHECKPOINT");
  exec_ok(con, "BEGIN TRANSACTION");
  auto& storage = get_storage_in_txn(con, "t3c_delete");

  std::vector<projected_column> cols = {real_col(0)};
  auto before                        = walk_all(storage, *con.context, cols, int_types(1));
  REQUIRE(before.viable);
  REQUIRE(cache.rebuilds() == 1);

  exec_ok(con, "COMMIT");
  exec_ok(con, "DELETE FROM t3c_delete WHERE a < 1000");
  exec_ok(con, "BEGIN TRANSACTION");

  auto after = walk_all(storage, *con.context, cols, int_types(1));
  REQUIRE(after.viable);
  REQUIRE(cache.rebuilds() == 1);  // no structural change -> no rebuild
  REQUIRE(cache.hits() >= 1);
  require_same_walk(before, after);

  // Uncached agreement: the walk output really is delete-independent.
  walk_result uncached;
  {
    cache_disable_guard guard(true);
    uncached = walk_all(storage, *con.context, cols, int_types(1));
  }
  require_same_walk(after, uncached);
}

TEST_CASE("metadata cache rebuilds after checkpoint compaction",
          "[scan][duckdb_native_metadata_cache]")
{
  auto& cache = duckdb_native_metadata_cache::instance();
  cache.clear();
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t3c_ckpt(a INTEGER)");
  exec_ok(con, "INSERT INTO t3c_ckpt SELECT range FROM range(0, 300000)");
  exec_ok(con, "CHECKPOINT");
  exec_ok(con, "BEGIN TRANSACTION");
  auto& storage = get_storage_in_txn(con, "t3c_ckpt");

  std::vector<projected_column> cols = {real_col(0)};
  auto before                        = walk_all(storage, *con.context, cols, int_types(1));
  REQUIRE(before.viable);
  REQUIRE(cache.rebuilds() == 1);

  // Delete every row and checkpoint: fully-deleted row groups are reclaimed,
  // so the physical structure changes and the identity probe must catch it
  // regardless of how the rewrite happened. (This is also why the probe, not
  // an O(1) last-commit check, is the validity test: CHECKPOINT restructures
  // row groups without bumping the transaction manager's last_commit.)
  exec_ok(con, "COMMIT");
  exec_ok(con, "DELETE FROM t3c_ckpt");
  exec_ok(con, "CHECKPOINT");
  exec_ok(con, "BEGIN TRANSACTION");

  auto after = walk_all(storage, *con.context, cols, int_types(1));
  REQUIRE(after.viable);
  REQUIRE(cache.rebuilds() == 2);

  walk_result uncached;
  {
    cache_disable_guard guard(true);
    uncached = walk_all(storage, *con.context, cols, int_types(1));
  }
  require_same_walk(after, uncached);

  duckdb::idx_t rows_after = 0;
  for (auto const& rg : after.row_groups) {
    rows_after += rg.row_count;
  }
  REQUIRE(rows_after == 0);  // vacuum reclaimed the deleted rows
}

//===--------------------------------------------------------------------===//
// Bypass and isolation
//===--------------------------------------------------------------------===//

TEST_CASE("metadata cache bypasses on transaction-local appends",
          "[scan][duckdb_native_metadata_cache]")
{
  auto& cache = duckdb_native_metadata_cache::instance();
  cache.clear();
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t3c_local(a INTEGER)");
  exec_ok(con, "INSERT INTO t3c_local SELECT range FROM range(0, 300000)");
  exec_ok(con, "CHECKPOINT");
  exec_ok(con, "BEGIN TRANSACTION");
  auto& storage = get_storage_in_txn(con, "t3c_local");

  // Uncommitted rows in THIS transaction live in LocalStorage; the cache must
  // refuse to serve or capture them.
  exec_ok(con, "INSERT INTO t3c_local VALUES (999999)");
  auto acquired = cache.acquire(storage, *con.context, {});
  REQUIRE_FALSE(acquired.has_value());
  REQUIRE(cache.bypasses() == 1);
  REQUIRE(cache.rebuilds() == 0);

  // Rolled back -> local storage gone -> the cache serves again.
  exec_ok(con, "ROLLBACK");
  exec_ok(con, "BEGIN TRANSACTION");
  auto served = cache.acquire(storage, *con.context, {});
  REQUIRE(served.has_value());
  REQUIRE(served->core->total_rows == 300000);
  REQUIRE(cache.rebuilds() == 1);
}

TEST_CASE("metadata cache keys tables independently", "[scan][duckdb_native_metadata_cache]")
{
  auto& cache = duckdb_native_metadata_cache::instance();
  cache.clear();
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t3c_ind_a(a INTEGER)");
  exec_ok(con, "CREATE TABLE t3c_ind_b(a INTEGER)");
  exec_ok(con, "INSERT INTO t3c_ind_a SELECT range FROM range(0, 150000)");
  exec_ok(con, "INSERT INTO t3c_ind_b SELECT range FROM range(0, 150000)");
  exec_ok(con, "CHECKPOINT");
  exec_ok(con, "BEGIN TRANSACTION");
  auto& storage_a = get_storage_in_txn(con, "t3c_ind_a");
  auto& storage_b = get_storage_in_txn(con, "t3c_ind_b");

  std::vector<projected_column> cols = {real_col(0)};
  REQUIRE(walk_all(storage_a, *con.context, cols, int_types(1)).viable);
  REQUIRE(walk_all(storage_b, *con.context, cols, int_types(1)).viable);
  REQUIRE(cache.rebuilds() == 2);

  // Mutating one table invalidates only its own snapshot.
  exec_ok(con, "COMMIT");
  exec_ok(con, "INSERT INTO t3c_ind_a VALUES (7)");
  exec_ok(con, "BEGIN TRANSACTION");
  REQUIRE(walk_all(storage_b, *con.context, cols, int_types(1)).viable);
  REQUIRE(cache.rebuilds() == 2);  // b untouched -> hit
  REQUIRE(walk_all(storage_a, *con.context, cols, int_types(1)).viable);
  REQUIRE(cache.rebuilds() == 3);  // a changed -> rebuild
}

//===--------------------------------------------------------------------===//
// Varchar overflow refusal through the cached path
//===--------------------------------------------------------------------===//

TEST_CASE("metadata cache preserves the varchar overflow refusal",
          "[scan][duckdb_native_metadata_cache]")
{
  auto& cache = duckdb_native_metadata_cache::instance();
  cache.clear();
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t3c_overflow(s VARCHAR)");
  exec_ok(con, "INSERT INTO t3c_overflow VALUES (repeat('x', 5000))");
  exec_ok(con, "INSERT INTO t3c_overflow SELECT 'short' FROM range(0, 100)");
  exec_ok(con, "CHECKPOINT");
  exec_ok(con, "BEGIN TRANSACTION");
  auto& storage = get_storage_in_txn(con, "t3c_overflow");

  std::vector<projected_column> cols = {real_col(0)};
  auto ts                            = varchar_types(1);

  auto cached = walk_all(storage, *con.context, cols, ts);
  REQUIRE_FALSE(cached.viable);
  REQUIRE(cached.viability_failure_reason.find("overflow") != std::string::npos);
  REQUIRE(cache.rebuilds() == 1);

  // The refusal repeats from the cached snapshot (a cached refusing PRODUCT —
  // repeat refusals cost only the probe)...
  auto cached_again = walk_all(storage, *con.context, cols, ts);
  REQUIRE_FALSE(cached_again.viable);
  REQUIRE(cache.hits() == 1);
  REQUIRE(cache.product_hits() == 1);
  REQUIRE(cached_again.viability_failure_reason == cached.viability_failure_reason);

  // ... with the same reason as the uncached walk.
  walk_result uncached;
  {
    cache_disable_guard guard(true);
    uncached = walk_all(storage, *con.context, cols, ts);
  }
  REQUIRE_FALSE(uncached.viable);
  REQUIRE(cached.viability_failure_reason == uncached.viability_failure_reason);
}

TEST_CASE("metadata cache accepts varchar below the overflow limit",
          "[scan][duckdb_native_metadata_cache]")
{
  auto& cache = duckdb_native_metadata_cache::instance();
  cache.clear();
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t3c_varchar(s VARCHAR)");
  exec_ok(con, "INSERT INTO t3c_varchar SELECT repeat('x', 4000) FROM range(0, 100)");
  exec_ok(con, "CHECKPOINT");
  exec_ok(con, "BEGIN TRANSACTION");
  auto& storage = get_storage_in_txn(con, "t3c_varchar");

  std::vector<projected_column> cols = {real_col(0)};
  auto ts                            = varchar_types(1);

  auto cached = walk_all(storage, *con.context, cols, ts);
  REQUIRE(cached.viable);
  REQUIRE_FALSE(cached.row_groups.empty());

  walk_result uncached;
  {
    cache_disable_guard guard(true);
    uncached = walk_all(storage, *con.context, cols, ts);
  }
  require_same_walk(cached, uncached);
}

//===--------------------------------------------------------------------===//
// Fused-pass equivalence with the old serial passes
//===--------------------------------------------------------------------===//

TEST_CASE("fused stats pass reports the column-outer, row-group-inner refusal",
          "[scan][duckdb_native_metadata_cache]")
{
  // Old semantics: one overflow pass PER VARCHAR COLUMN (projected order),
  // row groups inner — the reported refusal is the min-rg refusal of the min
  // refusing column. Here s0 (ci=0) overflows only in row group 2 and s1
  // (ci=1) only in row group 1: the refusal must name column 0 / row group 2,
  // NOT the (row-group-wise earlier) column 1 / row group 1 — under both the
  // cached and uncached paths and under any worker count.
  auto& cache = duckdb_native_metadata_cache::instance();
  cache.clear();
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t3c_order(s0 VARCHAR, s1 VARCHAR)");
  exec_ok(con,
          "INSERT INTO t3c_order SELECT "
          "CASE WHEN range = 250000 THEN repeat('x', 6000) ELSE 'a' END, "
          "CASE WHEN range = 130000 THEN repeat('y', 6000) ELSE 'b' END "
          "FROM range(0, 300000)");
  exec_ok(con, "CHECKPOINT");
  exec_ok(con, "BEGIN TRANSACTION");
  auto& storage = get_storage_in_txn(con, "t3c_order");

  std::vector<projected_column> cols = {real_col(0), real_col(1)};
  auto ts                            = varchar_types(2);

  auto check_reason = [](const walk_result& r) {
    REQUIRE_FALSE(r.viable);
    INFO("refusal: " << r.viability_failure_reason);
    REQUIRE(r.viability_failure_reason.find("row group 2 varchar column 0") != std::string::npos);
    REQUIRE(r.viability_failure_reason.find("overflow-block limit") != std::string::npos);
  };

  for (const char* threads : {"1", "5"}) {
    walk_threads_guard tg(threads);
    cache.clear();
    auto cached = walk_all(storage, *con.context, cols, ts);
    check_reason(cached);
    walk_result uncached;
    {
      cache_disable_guard guard(true);
      uncached = walk_all(storage, *con.context, cols, ts);
    }
    check_reason(uncached);
    REQUIRE(cached.viability_failure_reason == uncached.viability_failure_reason);
  }
}

TEST_CASE("parallel walk is deterministic across worker counts",
          "[scan][duckdb_native_metadata_cache]")
{
  auto& cache          = duckdb_native_metadata_cache::instance();
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t3c_det(a INTEGER, b INTEGER, s VARCHAR)");
  exec_ok(con,
          "INSERT INTO t3c_det SELECT range, range * 2, "
          "repeat('z', 1 + (range % 37)) FROM range(0, 500000)");
  exec_ok(con, "CHECKPOINT");
  exec_ok(con, "BEGIN TRANSACTION");
  auto& storage = get_storage_in_txn(con, "t3c_det");

  std::vector<projected_column> cols = {real_col(0), real_col(1), real_col(2)};
  std::vector<sirius::logical_type> ts{sirius::logical_type::make(sirius::type_id::INTEGER),
                                       sirius::logical_type::make(sirius::type_id::INTEGER),
                                       sirius::logical_type::make(sirius::type_id::VARCHAR)};
  auto f = make_constant_filter(
    0, 0, duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO, duckdb::Value::INTEGER(370000));

  // Reference: serial, uncached (the old walk's exact traversal shape).
  walk_result reference;
  {
    walk_threads_guard tg("1");
    cache_disable_guard guard(true);
    reference = walk_all(storage, *con.context, cols, ts, &f.filters, &f.column_ids);
  }
  REQUIRE(reference.viable);
  REQUIRE(reference.pruned_row_groups >= 1);

  for (const char* threads : {"2", "3", "8"}) {
    walk_threads_guard tg(threads);
    // Uncached parallel == serial reference.
    walk_result uncached;
    {
      cache_disable_guard guard(true);
      uncached = walk_all(storage, *con.context, cols, ts, &f.filters, &f.column_ids);
    }
    require_same_walk(reference, uncached);
    // Cached (fresh snapshot each round) == serial reference.
    cache.clear();
    auto cached = walk_all(storage, *con.context, cols, ts, &f.filters, &f.column_ids);
    require_same_walk(reference, cached);
  }
}

//===--------------------------------------------------------------------===//
// Concurrent-commit consistency (the stack-t1-t2-t3 throughput hang)
//===--------------------------------------------------------------------===//

// A refresh stream committing INSERTs mutates the row-group tree
// non-atomically (counts are bumped before total_rows; MergeStorage appends
// nodes one at a time). Every snapshot the cache serves while that happens
// must still be internally consistent — contiguous starts and
// Σ row_count == total_rows — or be rejected (bypass). A torn capture served
// (or installed) here is exactly the geometry corruption that poisoned the
// SF1000 throughput run.
TEST_CASE("metadata cache never serves a torn snapshot under concurrent commits",
          "[scan][duckdb_native_metadata_cache]")
{
  auto& cache = duckdb_native_metadata_cache::instance();
  cache.clear();
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t3c_race(a INTEGER)");
  exec_ok(con, "INSERT INTO t3c_race SELECT range FROM range(0, 300000)");
  exec_ok(con, "CHECKPOINT");

  duckdb::DataTable* storage = nullptr;
  {
    exec_ok(con, "BEGIN TRANSACTION");
    storage = &get_storage_in_txn(con, "t3c_race");
    exec_ok(con, "COMMIT");
  }
  // Extra connections come off the DatabaseInstance so the test also runs
  // under the shared-env harness (where db_owner is null).
  auto& db_instance = *con.context->db;

  std::atomic<bool> stop{false};
  std::atomic<std::uint64_t> served{0};
  std::atomic<std::uint64_t> inconsistent{0};

  // Writer: commit append batches back-to-back on its own connection. Batches
  // straddle row-group boundaries so commits both grow the tail row group and
  // append new nodes.
  std::atomic<std::uint64_t> committed_batches{0};
  std::thread writer([&db_instance, &stop, &committed_batches] {
    duckdb::Connection wcon(db_instance);
    for (int i = 0; i < 40 && !stop.load(); ++i) {
      auto r = wcon.Query("INSERT INTO t3c_race SELECT range FROM range(0, 100000)");
      if (!r || r->HasError()) { break; }
      ++committed_batches;
    }
    stop.store(true);
  });

  // Readers: acquire fresh snapshots in short transactions and verify every
  // SERVED snapshot's internal consistency (a bypass — nullopt — is a legal
  // answer under a mid-flight commit; a torn serve is not).
  std::vector<std::thread> readers;
  for (int t = 0; t < 3; ++t) {
    readers.emplace_back([&db_instance, &cache, storage, &stop, &served, &inconsistent] {
      duckdb::Connection rcon(db_instance);
      std::vector<duckdb::idx_t> const stats_cols{0};
      while (!stop.load()) {
        if (rcon.Query("BEGIN TRANSACTION")->HasError()) { break; }
        auto snap = cache.acquire(*storage, *rcon.context, stats_cols);
        if (snap) {
          ++served;
          auto const& core      = *snap->core;
          duckdb::idx_t running = 0;
          bool ok               = core.row_group_start.size() == core.n_row_groups &&
                    core.row_count.size() == core.n_row_groups;
          for (std::size_t i = 0; ok && i < core.n_row_groups; ++i) {
            ok = core.row_group_start[i] == core.row_group_start[0] + running;
            running += core.row_count[i];
          }
          ok      = ok && running == core.total_rows;
          auto it = snap->column_stats.find(0);
          ok      = ok && it != snap->column_stats.end() &&
               it->second->per_row_group.size() == core.n_row_groups;
          if (!ok) { ++inconsistent; }
        }
        rcon.Query("COMMIT");
      }
    });
  }

  writer.join();
  stop.store(true);
  for (auto& r : readers) {
    r.join();
  }

  REQUIRE(inconsistent.load() == 0);

  // Settled state: an acquire must serve (not bypass) and reflect every commit.
  exec_ok(con, "BEGIN TRANSACTION");
  auto settled = cache.acquire(*storage, *con.context, {0});
  REQUIRE(settled.has_value());
  duckdb::idx_t total = 0;
  for (auto const c : settled->core->row_count) {
    total += c;
  }
  REQUIRE(total == settled->core->total_rows);
  REQUIRE(committed_batches.load() > 0);  // the harness actually raced commits
  REQUIRE(settled->core->total_rows ==
          static_cast<duckdb::idx_t>(300000 + committed_batches.load() * 100000));
  exec_ok(con, "COMMIT");
}

// End-to-end variant through prepare_duckdb_native_walk: full walks (cache
// acquire + product assemble/store/serve, plus the uncached bypass fallback)
// racing committed appends. During the race the walks must complete without
// throwing (a torn capture bypasses; the bypassed uncached read is the
// pre-existing, unvalidated behavior, so its output is not asserted on); the
// settled state afterwards must serve a product-cached walk identical to a
// fresh assemble and account for every committed row.
TEST_CASE("prepared walks survive concurrent commits and settle exactly",
          "[scan][duckdb_native_metadata_cache]")
{
  auto& cache = duckdb_native_metadata_cache::instance();
  cache.clear();
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t3c_race2(a INTEGER)");
  exec_ok(con, "INSERT INTO t3c_race2 SELECT range FROM range(0, 300000)");
  exec_ok(con, "CHECKPOINT");

  duckdb::DataTable* storage = nullptr;
  {
    exec_ok(con, "BEGIN TRANSACTION");
    storage = &get_storage_in_txn(con, "t3c_race2");
    exec_ok(con, "COMMIT");
  }
  auto& db_instance = *con.context->db;

  std::atomic<bool> stop{false};
  std::atomic<std::uint64_t> walked{0};
  std::atomic<std::uint64_t> failed{0};

  std::atomic<std::uint64_t> committed_batches{0};
  std::thread writer([&db_instance, &stop, &committed_batches] {
    duckdb::Connection wcon(db_instance);
    for (int i = 0; i < 25 && !stop.load(); ++i) {
      auto r = wcon.Query("INSERT INTO t3c_race2 SELECT range FROM range(0, 100000)");
      if (!r || r->HasError()) { break; }
      ++committed_batches;
    }
    stop.store(true);
  });

  std::vector<std::thread> readers;
  for (int t = 0; t < 3; ++t) {
    readers.emplace_back([&db_instance, storage, &stop, &walked, &failed] {
      duckdb::Connection rcon(db_instance);
      std::vector<projected_column> cols = {real_col(0)};
      auto types                         = int_types(1);
      // Alternate a pruning filter in and out so both product-keyed and
      // filterless shapes race the commits.
      auto f = make_constant_filter(
        0, 0, duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO, duckdb::Value::INTEGER(50000));
      std::uint64_t iter = 0;
      while (!stop.load()) {
        if (rcon.Query("BEGIN TRANSACTION")->HasError()) { break; }
        bool const with_filter = (++iter % 2) == 0;
        try {
          auto res = walk_all(*storage,
                              *rcon.context,
                              cols,
                              types,
                              with_filter ? &f.filters : nullptr,
                              with_filter ? &f.column_ids : nullptr);
          if (res.viable) { ++walked; }
        } catch (...) {
          ++failed;
        }
        rcon.Query("COMMIT");
      }
    });
  }

  writer.join();
  stop.store(true);
  for (auto& r : readers) {
    r.join();
  }

  REQUIRE(failed.load() == 0);
  REQUIRE(walked.load() > 0);
  REQUIRE(committed_batches.load() > 0);

  // Settled: a fresh walk assembles from the final geometry, the repeat is
  // product-served, and both agree exactly (incl. contiguity and totals).
  exec_ok(con, "BEGIN TRANSACTION");
  std::vector<projected_column> cols = {real_col(0)};
  auto f                             = make_constant_filter(
    0, 0, duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO, duckdb::Value::INTEGER(50000));
  auto const product_hits_before = cache.product_hits();
  auto settled = walk_all(*storage, *con.context, cols, int_types(1), &f.filters, &f.column_ids);
  auto repeat  = walk_all(*storage, *con.context, cols, int_types(1), &f.filters, &f.column_ids);
  REQUIRE(settled.viable);
  REQUIRE(cache.product_hits() > product_hits_before);
  require_same_walk(settled, repeat);
  for (std::size_t i = 1; i < settled.row_groups.size(); ++i) {
    REQUIRE(settled.row_groups[i].row_group_start ==
            settled.row_groups[i - 1].row_group_start + settled.row_groups[i - 1].row_count);
  }
  // Unfiltered settled walk accounts for every committed row.
  auto full           = walk_all(*storage, *con.context, cols, int_types(1));
  duckdb::idx_t total = 0;
  for (auto const& rg : full.row_groups) {
    total += rg.row_count;
  }
  REQUIRE(total == static_cast<duckdb::idx_t>(300000 + committed_batches.load() * 100000));
  exec_ok(con, "COMMIT");
}
