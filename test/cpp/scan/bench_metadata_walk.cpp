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

// Hidden microbenchmark for the duckdb-native metadata prepare walk on a real
// (unpinned) table — the P21 fix #3/#4 adjudication harness. GPU-free.
//
// Run:
//   SIRIUS_WALK_BENCH_DB=$HOME/tpch_sf1000.duckdb \
//     build/release/extension/sirius/test/cpp/sirius_unittest '[walk_bench]'
//
// Scenarios (all timings are the prepare walk only — the per-query cost a
// plan build pays on the query thread):
//   uncached/serial    — cache killed, 1 worker: the old build's walk cost
//                        (same GetPartitionStats + same per-(row group,
//                        column) statistics reads; the fusion only reorders
//                        them into one pass).
//   uncached/parallel  — cache killed, default workers (fix #4 alone).
//   first (rebuild)    — cache cleared: locked probe + parallel stats
//                        extraction + fused assemble (unpinned first-query
//                        cost with fix #3).
//   repeat (product)   — identical query shape: probe-only product hit.
//   varied (assemble)  — new predicate constant each rep (--vary-predicates
//                        shape): snapshot hit + fused assemble over cached
//                        statistics, no storage reads.

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/catalog/catalog.hpp>
#include <duckdb/catalog/catalog_entry/duck_table_entry.hpp>
#include <duckdb/common/column_index.hpp>
#include <duckdb/common/types/date.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/planner/filter/constant_filter.hpp>
#include <duckdb/planner/table_filter.hpp>
#include <duckdb/storage/data_table.hpp>
#include <op/scan/duckdb_native_metadata.hpp>
#include <op/scan/duckdb_native_metadata_cache.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <string>
#include <vector>

using namespace sirius;
using namespace sirius::op::scan;

namespace {

double time_ms(const std::function<void()>& fn)
{
  auto const t0 = std::chrono::steady_clock::now();
  fn();
  auto const t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

struct threads_env_guard {
  explicit threads_env_guard(const char* value)
  {
    if (value == nullptr) {
      unsetenv("SIRIUS_METADATA_WALK_THREADS");
    } else {
      setenv("SIRIUS_METADATA_WALK_THREADS", value, 1);
    }
  }
  ~threads_env_guard() { unsetenv("SIRIUS_METADATA_WALK_THREADS"); }
};

}  // namespace

TEST_CASE("metadata walk microbenchmark on an unpinned table", "[.][walk_bench]")
{
  const char* db_path = std::getenv("SIRIUS_WALK_BENCH_DB");
  if (db_path == nullptr) {
    WARN("SIRIUS_WALK_BENCH_DB not set; skipping walk benchmark");
    return;
  }
  std::string const table = [] {
    const char* t = std::getenv("SIRIUS_WALK_BENCH_TABLE");
    return std::string(t ? t : "lineitem");
  }();

  duckdb::DBConfig config;
  config.options.access_mode = duckdb::AccessMode::READ_ONLY;
  duckdb::DuckDB db(db_path, &config);
  duckdb::Connection con(db);
  REQUIRE_FALSE(con.Query("BEGIN TRANSACTION")->HasError());

  auto& ctx     = *con.context;
  auto& catalog = duckdb::Catalog::GetCatalog(ctx, "");
  duckdb::CatalogTransaction txn(catalog, ctx);
  auto& schema = catalog.GetSchema(txn, "main");
  auto entry   = schema.GetEntry(txn, duckdb::CatalogType::TABLE_ENTRY, table);
  REQUIRE(entry);
  auto& storage = entry->Cast<duckdb::DuckTableEntry>().GetStorage();

  // q1-shaped lineitem scan: filter on l_shipdate (10), project quantity /
  // extendedprice / discount / tax / returnflag / linestatus / shipdate.
  // (For a non-lineitem table set SIRIUS_WALK_BENCH_TABLE and adjust here.)
  struct col_spec {
    duckdb::idx_t storage_idx;
    sirius::logical_type type;
  };
  std::vector<col_spec> specs = {
    {4, sirius::logical_type::make_decimal(15, 2)},     // l_quantity
    {5, sirius::logical_type::make_decimal(15, 2)},     // l_extendedprice
    {6, sirius::logical_type::make_decimal(15, 2)},     // l_discount
    {7, sirius::logical_type::make_decimal(15, 2)},     // l_tax
    {8, sirius::logical_type::make(type_id::VARCHAR)},  // l_returnflag
    {9, sirius::logical_type::make(type_id::VARCHAR)},  // l_linestatus
    {10, sirius::logical_type::make(type_id::DATE)},    // l_shipdate
  };
  // SIRIUS_WALK_BENCH_HEAVY=1: add every remaining lineitem varchar — the
  // per-varchar-column stats passes dominated the old walk, so this is the
  // shape the P21 report's 70-190 ms band came from.
  if (const char* heavy = std::getenv("SIRIUS_WALK_BENCH_HEAVY"); heavy && heavy[0] == '1') {
    specs.push_back({13, sirius::logical_type::make(type_id::VARCHAR)});  // l_shipinstruct
    specs.push_back({14, sirius::logical_type::make(type_id::VARCHAR)});  // l_shipmode
    specs.push_back({15, sirius::logical_type::make(type_id::VARCHAR)});  // l_comment
  }
  std::vector<projected_column> cols;
  std::vector<sirius::logical_type> types;
  duckdb::vector<duckdb::ColumnIndex> column_ids;
  for (auto const& s : specs) {
    projected_column pc;
    pc.storage_idx = duckdb::StorageIndex(s.storage_idx);
    cols.push_back(pc);
    types.push_back(s.type);
    column_ids.emplace_back(s.storage_idx);
  }
  duckdb::idx_t shipdate_key = 0;
  for (std::size_t i = 0; i < specs.size(); ++i) {
    if (specs[i].storage_idx == 10) { shipdate_key = static_cast<duckdb::idx_t>(i); }
  }

  auto make_filter_set = [&](int day_offset) {
    auto filters                   = std::make_unique<duckdb::TableFilterSet>();
    auto date                      = duckdb::Date::FromDate(1998, 9, 1 + (day_offset % 27));
    filters->filters[shipdate_key] = duckdb::make_uniq<duckdb::ConstantFilter>(
      duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO, duckdb::Value::DATE(date));
    return filters;
  };

  auto& cache = duckdb_native_metadata_cache::instance();

  auto run_walk = [&](const duckdb::TableFilterSet* filters) {
    auto plan = prepare_duckdb_native_walk(storage, ctx, cols, types, filters, &column_ids);
    REQUIRE(plan.viable);
    return plan.n_row_groups;
  };

  auto const filters = make_filter_set(0);
  std::fprintf(stderr, "[walk_bench] table=%s db=%s\n", table.c_str(), db_path);

  //===----------Cold first walk (page cache state unknown)----------===//
  {
    cache.set_disabled_for_testing(true);
    threads_env_guard tg("1");
    std::size_t n = 0;
    auto const ms = time_ms([&] { n = run_walk(filters.get()); });
    std::fprintf(stderr, "[walk_bench] cold uncached/serial: %.2f ms (%zu row groups)\n", ms, n);
  }

  auto bench = [&](const char* label, int reps, const std::function<double()>& one) {
    double best = 1e30;
    double sum  = 0;
    for (int i = 0; i < reps; ++i) {
      auto const ms = one();
      best          = std::min(best, ms);
      sum += ms;
      std::fprintf(stderr, "[walk_bench]   %s rep %d: %.3f ms\n", label, i, ms);
    }
    std::fprintf(stderr,
                 "[walk_bench] %s: best %.3f ms, mean %.3f ms over %d reps\n",
                 label,
                 best,
                 sum / reps,
                 reps);
    return best;
  };

  //===----------Uncached (old-build-equivalent), serial----------===//
  cache.set_disabled_for_testing(true);
  bench("uncached/serial (old-build walk)", 5, [&] {
    threads_env_guard tg("1");
    return time_ms([&] { run_walk(filters.get()); });
  });

  //===----------Uncached, parallel (fix #4 alone)----------===//
  bench("uncached/parallel", 5, [&] {
    threads_env_guard tg(nullptr);  // default worker count
    return time_ms([&] { run_walk(filters.get()); });
  });

  //===----------Cached: first query (rebuild)----------===//
  cache.set_disabled_for_testing(false);
  bench("first query (snapshot rebuild)", 5, [&] {
    cache.clear();
    return time_ms([&] { run_walk(filters.get()); });
  });
  std::fprintf(stderr,
               "[walk_bench] counters after rebuild reps: rebuilds=%llu hits=%llu "
               "product_hits=%llu bypasses=%llu\n",
               static_cast<unsigned long long>(cache.rebuilds()),
               static_cast<unsigned long long>(cache.hits()),
               static_cast<unsigned long long>(cache.product_hits()),
               static_cast<unsigned long long>(cache.bypasses()));

  //===----------Cached: repeat query, same shape (product hit)----------===//
  cache.clear();
  run_walk(filters.get());  // seed snapshot + product
  bench("repeat query (product hit)", 7, [&] { return time_ms([&] { run_walk(filters.get()); }); });
  REQUIRE(cache.product_hits() >= 7);

  //===----------Cached: varied predicate (snapshot hit + assemble)----------===//
  {
    int day                 = 0;
    auto const product_hits = cache.product_hits();
    bench("varied predicate (snapshot hit + assemble)", 7, [&] {
      auto varied = make_filter_set(++day);
      return time_ms([&] { run_walk(varied.get()); });
    });
    // Every varied rep must have missed the product cache (fresh constants).
    REQUIRE(cache.product_hits() == product_hits);
  }

  std::fprintf(stderr,
               "[walk_bench] final counters: rebuilds=%llu hits=%llu product_hits=%llu "
               "bypasses=%llu\n",
               static_cast<unsigned long long>(cache.rebuilds()),
               static_cast<unsigned long long>(cache.hits()),
               static_cast<unsigned long long>(cache.product_hits()),
               static_cast<unsigned long long>(cache.bypasses()));

  con.Query("COMMIT");
}
