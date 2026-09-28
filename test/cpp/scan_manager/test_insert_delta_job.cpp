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

// Tests for the prepare-time insert-delta job: bundle staging and masks,
// finalize (all-invisible bundles dropped, all-visible ones unmasked),
// byte-budget planning, and per-operator cuts that share the staged storage.
// Capture-level tests live in test/cpp/scan/test_duckdb_insert_delta.cpp.

#include "operator/operator_test_utils.hpp"

#include <catch.hpp>
#include <cucascade/memory/topology_discovery.hpp>
#include <duckdb.hpp>
#include <duckdb/catalog/catalog.hpp>
#include <duckdb/catalog/catalog_entry/duck_table_entry.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/storage/data_table.hpp>
#include <exec/scoped_dispatcher.hpp>
#include <exec/thread_pool.hpp>
#include <io/kvikio/kvikio_context.hpp>
#include <io/sirius_datasource.hpp>
#include <memory/topology_index.hpp>
#include <op/scan/duckdb_native_gpu_ingestible.hpp>
#include <op/scan/duckdb_native_metadata.hpp>
#include <scan_manager/insert_delta_job.hpp>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

using sirius::scan_manager::cut_delta_splits_for_op;
using sirius::scan_manager::insert_delta_job_request;
using sirius::scan_manager::run_insert_delta_jobs;

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

struct job_test_db {
  std::string path;
  std::unique_ptr<duckdb::DuckDB> db;
  std::unique_ptr<duckdb::Connection> con;

  job_test_db()
  {
    static int counter = 0;
    path               = "/tmp/sirius_insert_delta_job_test_" + std::to_string(::getpid()) + "_" +
           std::to_string(counter++) + ".db";
    std::remove(path.c_str());
    std::remove((path + ".wal").c_str());
    db  = std::make_unique<duckdb::DuckDB>(path);
    con = std::make_unique<duckdb::Connection>(*db);
    con->Query("SET gpu_execution = false;");
  }

  ~job_test_db()
  {
    con.reset();
    db.reset();
    std::remove(path.c_str());
    std::remove((path + ".wal").c_str());
  }
};

/// Requires an active transaction on @p con.
duckdb::DataTable& resolve_storage(duckdb::Connection& con, const std::string& table_name)
{
  auto& ctx     = *con.context;
  auto& catalog = duckdb::Catalog::GetCatalog(ctx, "");
  duckdb::CatalogTransaction txn(catalog, ctx);
  auto& schema = catalog.GetSchema(txn, "main");
  auto entry   = schema.GetEntry(txn, duckdb::CatalogType::TABLE_ENTRY, table_name);
  REQUIRE(entry);
  return entry->Cast<duckdb::DuckTableEntry>().GetStorage();
}

bool bit_at(std::span<std::uint32_t const> words, std::size_t i)
{
  return ((words[i / 32] >> (i % 32)) & 1u) != 0;
}

struct job_env {
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> mgr;
  sirius::memory::topology_index topology;
  std::vector<int> gpu_ids{0};

  job_env()
    : mgr(sirius::test::operator_utils::initialize_memory_manager()),
      topology(cucascade::memory::system_topology_info{}, std::vector<int>{0})
  {
  }
};

job_env& env()
{
  static job_env e;
  return e;
}

insert_delta_job_request make_request(duckdb::Connection& con,
                                      duckdb::DataTable& storage,
                                      std::size_t n_cache,
                                      std::vector<duckdb::storage_t> cols,
                                      std::vector<sirius::logical_type> types)
{
  insert_delta_job_request request;
  request.storage     = &storage;
  request.context     = con.context.get();
  request.n_cache     = n_cache;
  request.union_cols  = std::move(cols);
  request.union_types = std::move(types);
  request.entry_name  = "t";
  return request;
}

sirius::op::scan::projected_column real_col(duckdb::idx_t col_id)
{
  sirius::op::scan::projected_column pc;
  pc.storage_idx = duckdb::StorageIndex(col_id);
  pc.is_rowid    = false;
  return pc;
}

void run(std::vector<insert_delta_job_request>& requests)
{
  auto& e = env();
  sirius::exec::static_thread_pool pool(4);
  sirius::exec::scoped_dispatcher dispatcher(pool, 4);
  run_insert_delta_jobs(requests, dispatcher, *e.mgr, e.topology, e.gpu_ids);
}

}  // namespace

TEST_CASE("insert-delta job: bundles carry staged bytes and oracle-exact masks",
          "[insert_delta_job][scan_manager]")
{
  job_test_db tdb;
  exec_ok(*tdb.con, "SET threads TO 1");
  exec_ok(*tdb.con,
          "CREATE TABLE t AS SELECT range::INTEGER AS k, (range*10)::BIGINT AS v "
          "FROM range(130000)");
  exec_ok(*tdb.con, "CHECKPOINT");
  exec_ok(*tdb.con,
          "INSERT INTO t SELECT (130000+range)::INTEGER, ((130000+range)*10)::BIGINT "
          "FROM range(5000)");
  exec_ok(*tdb.con, "DELETE FROM t WHERE k IN (130001, 132048, 134999)");

  exec_ok(*tdb.con, "BEGIN TRANSACTION");
  auto& storage = resolve_storage(*tdb.con, "t");
  std::vector<insert_delta_job_request> requests;
  requests.push_back(make_request(*tdb.con,
                                  storage,
                                  130000,
                                  {0, 1},
                                  {sirius::logical_type::make(sirius::type_id::INTEGER),
                                   sirius::logical_type::make(sirius::type_id::BIGINT)}));
  run(requests);

  auto const& request = requests[0];
  REQUIRE(request.bundles.size() == 1);
  auto const& bundle = request.bundles[0];
  REQUIRE(bundle.total_rows == 5000);
  REQUIRE(bundle.slab_base != nullptr);
  REQUIRE(bundle.preferred_device == 0);

  REQUIRE(bundle.mask.has_mask());
  REQUIRE(bundle.mask.row_count == 5000);
  for (std::size_t i = 0; i < 5000; ++i) {
    auto const rowid   = 130000 + i;
    bool const deleted = (rowid == 130001 || rowid == 132048 || rowid == 134999);
    if (bit_at(bundle.mask.view(), i) != !deleted) {
      INFO("delta rowid " << rowid);
      REQUIRE(bit_at(bundle.mask.view(), i) == !deleted);
    }
  }
  exec_ok(*tdb.con, "ROLLBACK");
}

TEST_CASE("insert-delta job: all-visible bundles serve unmasked",
          "[insert_delta_job][scan_manager]")
{
  job_test_db tdb;
  exec_ok(*tdb.con, "CREATE TABLE t AS SELECT range::INTEGER AS k FROM range(20000)");
  exec_ok(*tdb.con, "CHECKPOINT");
  exec_ok(*tdb.con, "INSERT INTO t SELECT (20000+range)::INTEGER FROM range(1000)");

  exec_ok(*tdb.con, "BEGIN TRANSACTION");
  auto& storage = resolve_storage(*tdb.con, "t");
  std::vector<insert_delta_job_request> requests;
  requests.push_back(make_request(
    *tdb.con, storage, 20000, {0}, {sirius::logical_type::make(sirius::type_id::INTEGER)}));
  run(requests);

  REQUIRE(requests[0].bundles.size() == 1);
  REQUIRE(requests[0].bundles[0].total_rows == 1000);
  REQUIRE_FALSE(requests[0].bundles[0].mask.has_mask());
  exec_ok(*tdb.con, "ROLLBACK");
}

TEST_CASE("insert-delta job: all-invisible bundles are dropped", "[insert_delta_job][scan_manager]")
{
  job_test_db tdb;
  exec_ok(*tdb.con, "CREATE TABLE t AS SELECT range::INTEGER AS k FROM range(1000)");
  exec_ok(*tdb.con, "CHECKPOINT");

  // Pin the snapshot before the second connection's insert: the transaction
  // spawns lazily, so touch the table.
  exec_ok(*tdb.con, "BEGIN TRANSACTION");
  exec_ok(*tdb.con, "SELECT count(*) FROM t");
  duckdb::Connection con2(*tdb.db);
  exec_ok(con2, "INSERT INTO t SELECT (1000+range)::INTEGER FROM range(500)");

  auto& storage = resolve_storage(*tdb.con, "t");
  std::vector<insert_delta_job_request> requests;
  requests.push_back(make_request(
    *tdb.con, storage, 1000, {0}, {sirius::logical_type::make(sirius::type_id::INTEGER)}));
  run(requests);

  REQUIRE_FALSE(requests[0].plan.empty());  // the physical rows were captured...
  REQUIRE(requests[0].bundles.empty());     // ...but nothing is visible to serve
  exec_ok(*tdb.con, "ROLLBACK");
}

TEST_CASE("insert-delta job: a tiny byte budget splits bundles per row group",
          "[insert_delta_job][scan_manager]")
{
  job_test_db tdb;
  exec_ok(*tdb.con, "SET threads TO 1");
  exec_ok(*tdb.con, "CREATE TABLE t AS SELECT range::INTEGER AS k FROM range(10000)");
  exec_ok(*tdb.con, "CHECKPOINT");
  // Bulk insert: MergeStorage lands >= 2 persistent delta row groups.
  exec_ok(*tdb.con, "INSERT INTO t SELECT (10000+range)::INTEGER FROM range(130000)");

  exec_ok(*tdb.con, "BEGIN TRANSACTION");
  auto& storage = resolve_storage(*tdb.con, "t");
  std::vector<insert_delta_job_request> requests;
  requests.push_back(make_request(
    *tdb.con, storage, 10000, {0}, {sirius::logical_type::make(sirius::type_id::INTEGER)}));
  requests[0].approximate_batch_size = 1;  // force one bundle per row group
  run(requests);

  auto const& request = requests[0];
  REQUIRE(request.plan.row_groups.size() >= 2);
  REQUIRE(request.bundles.size() == request.plan.row_groups.size());
  for (auto const& bundle : request.bundles) {
    REQUIRE(bundle.rg_indices.size() == 1);
    REQUIRE_FALSE(bundle.mask.has_mask());  // everything visible
  }
  exec_ok(*tdb.con, "ROLLBACK");
}

TEST_CASE("insert-delta job: per-operator cuts subset the union and share storage",
          "[insert_delta_job][scan_manager]")
{
  job_test_db tdb;
  exec_ok(*tdb.con,
          "CREATE TABLE t AS SELECT range::INTEGER AS k, (range*10)::BIGINT AS v "
          "FROM range(20000)");
  exec_ok(*tdb.con, "CHECKPOINT");
  exec_ok(*tdb.con,
          "INSERT INTO t SELECT (20000+range)::INTEGER, ((20000+range)*10)::BIGINT "
          "FROM range(1600)");
  exec_ok(*tdb.con, "DELETE FROM t WHERE k = 20005");

  exec_ok(*tdb.con, "BEGIN TRANSACTION");
  auto& storage = resolve_storage(*tdb.con, "t");
  std::vector<insert_delta_job_request> requests;
  requests.push_back(make_request(*tdb.con,
                                  storage,
                                  20000,
                                  {0, 1},
                                  {sirius::logical_type::make(sirius::type_id::INTEGER),
                                   sirius::logical_type::make(sirius::type_id::BIGINT)}));
  run(requests);
  auto const& request = requests[0];
  REQUIRE(request.bundles.size() == 1);

  // Op A projects only column v (storage index 1).
  std::vector<sirius::op::scan::projected_column> op_a{real_col(1)};
  auto splits_a =
    cut_delta_splits_for_op(request, op_a, /*datasource=*/nullptr, /*block_manager=*/nullptr);
  REQUIRE(splits_a.size() == 1);
  auto const& info_a = *splits_a[0].info;
  REQUIRE(info_a.host_backed_only);
  REQUIRE(info_a.row_groups.size() == 1);
  REQUIRE(info_a.row_groups[0].columns.size() == 1);
  REQUIRE(info_a.row_groups[0].columns[0].column_id == 1);
  REQUIRE(info_a.row_groups[0].row_count == 1600);
  REQUIRE_FALSE(info_a.staging_keepalive.empty());
  // The host-backed descriptor points into the bundle's slab.
  auto const& seg_a = info_a.row_groups[0].columns[0].data_segments.at(0);
  REQUIRE(seg_a.host_ptr != nullptr);
  REQUIRE(seg_a.host_ptr >= request.bundles[0].slab_base);

  // Op B projects both columns; masks share the same words.
  std::vector<sirius::op::scan::projected_column> op_b{real_col(0), real_col(1)};
  auto splits_b = cut_delta_splits_for_op(request, op_b, nullptr, nullptr);
  REQUIRE(splits_b.size() == 1);
  REQUIRE(splits_b[0].info->row_groups[0].columns.size() == 2);
  REQUIRE(splits_a[0].mask.words.get() == splits_b[0].mask.words.get());
  REQUIRE(splits_a[0].mask.has_mask());  // the delete keeps the mask alive

  exec_ok(*tdb.con, "ROLLBACK");
}

TEST_CASE("insert-delta job: file-backed splits each own a duplicated datasource",
          "[insert_delta_job][scan_manager]")
{
  job_test_db tdb;
  exec_ok(*tdb.con, "SET threads TO 1");
  exec_ok(*tdb.con, "CREATE TABLE t AS SELECT range::INTEGER AS k FROM range(10000)");
  exec_ok(*tdb.con, "CHECKPOINT");
  // The bulk commit flushes persistent row groups (file-backed bundles); the
  // separate small commit stays transient (a host-only bundle).
  exec_ok(*tdb.con, "INSERT INTO t SELECT (10000+range)::INTEGER FROM range(250000)");
  exec_ok(*tdb.con, "INSERT INTO t VALUES (260000)");

  exec_ok(*tdb.con, "BEGIN TRANSACTION");
  auto& storage = resolve_storage(*tdb.con, "t");
  std::vector<insert_delta_job_request> requests;
  requests.push_back(make_request(
    *tdb.con, storage, 10000, {0}, {sirius::logical_type::make(sirius::type_id::INTEGER)}));
  requests[0].approximate_batch_size = 1;  // one bundle per row group
  run(requests);
  auto const& request = requests[0];
  REQUIRE(request.bundles.size() >= 2);

  auto ioctx = std::make_shared<sirius::io::kvikio_context>();
  std::shared_ptr<sirius::io::sirius_datasource> datasource = ioctx->open_datasource(tdb.path);

  std::vector<sirius::op::scan::projected_column> op{real_col(0)};
  auto splits = cut_delta_splits_for_op(request, op, datasource, /*block_manager=*/nullptr);
  REQUIRE(splits.size() == request.bundles.size());

  // The datasource's prefetch handle is per-scan state: every file-backed
  // split must own its own duplicate, never the shared original; host-only
  // splits carry none.
  std::unordered_set<sirius::io::sirius_datasource const*> seen;
  bool saw_host_only = false;
  for (auto const& split : splits) {
    if (split.info->host_backed_only) {
      saw_host_only = true;
      REQUIRE(split.info->datasource == nullptr);
      continue;
    }
    REQUIRE(split.info->datasource != nullptr);
    REQUIRE(split.info->datasource.get() != datasource.get());
    REQUIRE(seen.insert(split.info->datasource.get()).second);
  }
  REQUIRE(seen.size() >= 2);
  REQUIRE(saw_host_only);
  exec_ok(*tdb.con, "ROLLBACK");
}

TEST_CASE("insert-delta job: blockless-only splits carry no datasource",
          "[insert_delta_job][scan_manager]")
{
  job_test_db tdb;
  exec_ok(*tdb.con, "SET threads TO 1");
  exec_ok(*tdb.con,
          "CREATE TABLE t AS SELECT range::INTEGER AS k, range::INTEGER AS v FROM range(10000)");
  exec_ok(*tdb.con, "CHECKPOINT");
  // Constant column: the bulk-flushed v segments are blockless CONSTANT.
  exec_ok(*tdb.con, "INSERT INTO t SELECT (10000+range)::INTEGER, 7 FROM range(130000)");

  exec_ok(*tdb.con, "BEGIN TRANSACTION");
  auto& storage = resolve_storage(*tdb.con, "t");
  std::vector<insert_delta_job_request> requests;
  requests.push_back(make_request(*tdb.con,
                                  storage,
                                  10000,
                                  {0, 1},
                                  {sirius::logical_type::make(sirius::type_id::INTEGER),
                                   sirius::logical_type::make(sirius::type_id::INTEGER)}));
  requests[0].approximate_batch_size = 1;  // one bundle per row group
  run(requests);
  auto const& request = requests[0];

  auto ioctx = std::make_shared<sirius::io::kvikio_context>();
  std::shared_ptr<sirius::io::sirius_datasource> datasource = ioctx->open_datasource(tdb.path);

  // An operator projecting only the constant column cuts splits whose
  // persistent descriptors are all blockless: nothing reads the file, so the
  // split is host-backed-only and carries no datasource.
  std::vector<sirius::op::scan::projected_column> op{real_col(1)};
  auto splits             = cut_delta_splits_for_op(request, op, datasource, nullptr);
  bool saw_blockless_only = false;
  for (auto const& split : splits) {
    auto const& segs = split.info->row_groups.at(0).columns.at(0).data_segments;
    bool blockless   = false;
    for (auto const& s : segs) {
      if (s.host_ptr == nullptr && s.block_id < 0) {
        blockless = true;
        REQUIRE(s.compression == duckdb::CompressionType::COMPRESSION_CONSTANT);
        REQUIRE(s.segment_stats != nullptr);
      }
    }
    if (!blockless) { continue; }
    saw_blockless_only = true;
    REQUIRE(split.info->host_backed_only);
    REQUIRE(split.info->datasource == nullptr);
  }
  REQUIRE(saw_blockless_only);
  exec_ok(*tdb.con, "ROLLBACK");
}

TEST_CASE("insert-delta job: a delta spanning multiple capture ranges merges in order",
          "[insert_delta_job][scan_manager]")
{
  // The former environment override must not change the fixed internal
  // scheduling granularity.
  setenv("SIRIUS_METADATA_PARSE_CHUNK", "1", 1);
  struct chunk_env_restore {
    ~chunk_env_restore() { unsetenv("SIRIUS_METADATA_PARSE_CHUNK"); }
  } restore;
  REQUIRE(sirius::op::scan::metadata_parse_chunk() == 8);

  // Use enough row groups to span multiple fixed internal capture ranges.
  constexpr std::size_t delta_rows = 1'250'000;
  job_test_db tdb;
  exec_ok(*tdb.con, "SET threads TO 1");
  exec_ok(*tdb.con, "CREATE TABLE t AS SELECT range::INTEGER AS k FROM range(10000)");
  exec_ok(*tdb.con, "CHECKPOINT");
  exec_ok(
    *tdb.con,
    "INSERT INTO t SELECT (10000+range)::INTEGER FROM range(" + std::to_string(delta_rows) + ")");
  exec_ok(*tdb.con, "INSERT INTO t VALUES (" + std::to_string(10000 + delta_rows) + ")");

  exec_ok(*tdb.con, "BEGIN TRANSACTION");
  auto& storage = resolve_storage(*tdb.con, "t");
  std::vector<insert_delta_job_request> requests;
  requests.push_back(make_request(
    *tdb.con, storage, 10000, {0}, {sirius::logical_type::make(sirius::type_id::INTEGER)}));
  run(requests);

  auto const& request = requests[0];
  REQUIRE(request.plan.row_groups.size() > sirius::op::scan::metadata_parse_chunk());

  // Row-group order, contiguous boundaries, and filled columns survive the
  // parallel merge.
  std::size_t expected_start = 10000;
  std::size_t plan_rows      = 0;
  for (auto const& rg : request.plan.row_groups) {
    REQUIRE(rg.row_group_start == expected_start);
    REQUIRE(rg.k_offset == 0);
    REQUIRE_FALSE(rg.columns.empty());
    REQUIRE(rg.decoded_bytes_budget == rg.row_count * sizeof(int32_t));
    expected_start += rg.row_count;
    plan_rows += rg.row_count;
  }
  REQUIRE(plan_rows == delta_rows + 1);

  std::size_t bundle_rows = 0;
  for (auto const& b : request.bundles) {
    bundle_rows += b.total_rows;
  }
  REQUIRE(bundle_rows == delta_rows + 1);
  exec_ok(*tdb.con, "ROLLBACK");
}

TEST_CASE("insert-delta job: no delta is a no-op", "[insert_delta_job][scan_manager]")
{
  job_test_db tdb;
  exec_ok(*tdb.con, "CREATE TABLE t AS SELECT range::INTEGER AS k FROM range(1000)");
  exec_ok(*tdb.con, "CHECKPOINT");

  exec_ok(*tdb.con, "BEGIN TRANSACTION");
  auto& storage = resolve_storage(*tdb.con, "t");
  std::vector<insert_delta_job_request> requests;
  requests.push_back(make_request(
    *tdb.con, storage, 1000, {0}, {sirius::logical_type::make(sirius::type_id::INTEGER)}));
  run(requests);
  REQUIRE(requests[0].plan.empty());
  REQUIRE(requests[0].bundles.empty());
  exec_ok(*tdb.con, "ROLLBACK");
}
