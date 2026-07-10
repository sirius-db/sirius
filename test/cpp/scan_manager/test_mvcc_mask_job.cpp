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

// Gates for the prepare-time MVCC mask job: fan_out_and_join must
// assemble concurrent disjoint writes exactly once, rethrow the first task
// error, and turn silently-dropped tasks (a stopping dispatcher) into a loud
// error instead of a deadlock or a partial result; run_mvcc_mask_jobs must
// publish pinned, reservation-backed masks only for chunks that actually
// dropped rows, and do nothing at all for version-state-free tables.

#include "operator/operator_test_utils.hpp"

#include <cuda_runtime.h>

#include <catch.hpp>
#include <cucascade/memory/topology_discovery.hpp>
#include <duckdb.hpp>
#include <duckdb/catalog/catalog.hpp>
#include <duckdb/catalog/catalog_entry/duck_table_entry.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/storage/data_table.hpp>
#include <duckdb/transaction/duck_transaction.hpp>
#include <exec/scoped_dispatcher.hpp>
#include <exec/thread_pool.hpp>
#include <memory/topology_index.hpp>
#include <op/scan/duckdb_mvcc_visibility.hpp>
#include <scan_manager/mvcc_mask_job.hpp>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using sirius::scan_manager::fan_out_and_join;
using sirius::scan_manager::mvcc_chunk_mask_set;
using sirius::scan_manager::mvcc_mask_job_request;
using sirius::scan_manager::run_mvcc_mask_jobs;

namespace {

constexpr std::size_t kRowGroupRows = 122880;

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
    path               = "/tmp/sirius_mvcc_mask_job_test_" + std::to_string(::getpid()) + "_" +
           std::to_string(counter++) + ".db";
    std::remove(path.c_str());
    std::remove((path + ".wal").c_str());
    db  = std::make_unique<duckdb::DuckDB>(path);
    con = std::make_unique<duckdb::Connection>(*db);
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

sirius::scan_manager::duckdb_mvcc_metadata make_metadata(duckdb::Connection& con,
                                                         duckdb::DataTable& storage)
{
  sirius::scan_manager::duckdb_mvcc_metadata metadata;
  metadata.v_base  = duckdb::DuckTransaction::Get(*con.context, storage.GetAttached()).start_time;
  auto const total = static_cast<std::size_t>(storage.GetTotalRows());
  for (std::size_t off = 0; off < total;) {
    auto const rows = std::min(kRowGroupRows, total - off);
    metadata.base_row_count_per_chunk.push_back(rows);  // one row group per chunk
    off += rows;
  }
  return metadata;
}

bool bit_at(std::span<std::uint32_t const> words, std::size_t i)
{
  return ((words[i / 32] >> (i % 32)) & 1u) != 0;
}

/// Shared GPU/host memory manager (host spaces back the pinned mask
/// reservations) + a trivial topology (unknown GPUs normalize to node 0).
struct job_env {
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> mgr;
  sirius::memory::topology_index topology;

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

}  // namespace

TEST_CASE("fan_out_and_join assembles concurrent disjoint writes exactly once",
          "[mvcc_mask_job][scan_manager]")
{
  sirius::exec::static_thread_pool pool(4);
  sirius::exec::scoped_dispatcher dispatcher(pool, 4);

  constexpr std::size_t kSlices = 16;
  constexpr std::size_t kSlice  = 4096;
  std::vector<std::uint8_t> buffer(kSlices * kSlice, 0);

  std::vector<absl::AnyInvocable<void()>> tasks;
  for (std::size_t s = 0; s < kSlices; ++s) {
    tasks.push_back([&buffer, s] {
      std::fill_n(buffer.data() + s * kSlice, kSlice, static_cast<std::uint8_t>(s + 1));
    });
  }
  fan_out_and_join(dispatcher, std::move(tasks), "disjoint-write test");

  for (std::size_t s = 0; s < kSlices; ++s) {
    for (std::size_t i = 0; i < kSlice; ++i) {
      if (buffer[s * kSlice + i] != static_cast<std::uint8_t>(s + 1)) {
        REQUIRE(buffer[s * kSlice + i] == static_cast<std::uint8_t>(s + 1));
      }
    }
  }
}

TEST_CASE("fan_out_and_join rethrows the first task error after the join",
          "[mvcc_mask_job][scan_manager]")
{
  sirius::exec::static_thread_pool pool(4);
  sirius::exec::scoped_dispatcher dispatcher(pool, 4);

  std::atomic<int> ran{0};
  std::vector<absl::AnyInvocable<void()>> tasks;
  tasks.push_back([&ran] { ran.fetch_add(1); });
  tasks.push_back([] { throw std::runtime_error("mask task blew up"); });
  tasks.push_back([&ran] { ran.fetch_add(1); });

  REQUIRE_THROWS_WITH(fan_out_and_join(dispatcher, std::move(tasks), "error test"),
                      Catch::Contains("mask task blew up"));
  REQUIRE(ran.load() == 2);  // the join waited for the healthy tasks too
}

TEST_CASE("fan_out_and_join turns dropped tasks into a loud error, not a deadlock",
          "[mvcc_mask_job][scan_manager]")
{
  sirius::exec::static_thread_pool pool(2);
  sirius::exec::scoped_dispatcher dispatcher(pool, 2);
  dispatcher.request_stop();  // enqueue is a silent no-op from here on

  std::atomic<int> ran{0};
  std::vector<absl::AnyInvocable<void()>> tasks;
  for (int i = 0; i < 3; ++i) {
    tasks.push_back([&ran] { ran.fetch_add(1); });
  }
  REQUIRE_THROWS_AS(fan_out_and_join(dispatcher, std::move(tasks), "dropped test"),
                    std::runtime_error);
  REQUIRE(ran.load() == 0);
}

TEST_CASE("fan_out_and_join with zero tasks returns immediately", "[mvcc_mask_job][scan_manager]")
{
  sirius::exec::static_thread_pool pool(1);
  sirius::exec::scoped_dispatcher dispatcher(pool, 1);
  REQUIRE_NOTHROW(fan_out_and_join(dispatcher, {}, "empty test"));
}

TEST_CASE("run_mvcc_mask_jobs publishes pinned masks for dirty chunks only",
          "[mvcc_mask_job][scan_manager]")
{
  auto& e = env();
  job_test_db tdb;
  // Two row groups -> two chunks; deletes only in chunk 0.
  exec_ok(*tdb.con, "CREATE TABLE t AS SELECT range::INTEGER AS k FROM range(200000)");
  exec_ok(*tdb.con, "CHECKPOINT");
  exec_ok(*tdb.con, "DELETE FROM t WHERE rowid IN (0, 31, 32, 2048, 100000)");

  exec_ok(*tdb.con, "BEGIN TRANSACTION");
  auto& storage = resolve_storage(*tdb.con, "t");
  auto metadata = make_metadata(*tdb.con, storage);
  REQUIRE(metadata.base_row_count_per_chunk.size() == 2);

  auto* host_space = e.mgr->get_memory_space(cucascade::memory::Tier::HOST, 0);
  REQUIRE(host_space != nullptr);

  mvcc_mask_job_request request;
  request.masks        = std::make_shared<mvcc_chunk_mask_set>(2);
  request.metadata     = metadata;
  request.storage      = &storage;
  request.context      = tdb.con->context.get();
  request.chunk_spaces = {host_space, host_space};
  request.entry_name   = "t";
  std::vector<mvcc_mask_job_request> requests;
  requests.push_back(std::move(request));

  sirius::exec::static_thread_pool pool(4);
  sirius::exec::scoped_dispatcher dispatcher(pool, 4);
  run_mvcc_mask_jobs(requests, dispatcher, *e.mgr, e.topology);

  auto const& masks = *requests[0].masks;
  REQUIRE(masks[0] != nullptr);  // chunk with deletes: published
  REQUIRE(masks[1] == nullptr);  // untouched chunk: unmasked fast path

  auto const& mask = *masks[0];
  REQUIRE(mask.row_count == metadata.base_row_count_per_chunk[0]);
  REQUIRE(mask.words.size() == (mask.row_count + 31) / 32);
  for (std::size_t r = 0; r < mask.row_count; ++r) {
    bool const deleted = (r == 0 || r == 31 || r == 32 || r == 2048 || r == 100000);
    if (bit_at(mask.words, r) != !deleted) {
      INFO("row " << r);
      REQUIRE(bit_at(mask.words, r) == !deleted);
    }
  }

  // The published words live in registered (pinned) host memory carved from
  // the reservation-backed staging blocks.
  cudaPointerAttributes attrs{};
  REQUIRE(cudaPointerGetAttributes(&attrs, mask.words.data()) == cudaSuccess);
  REQUIRE(attrs.type == cudaMemoryTypeHost);

  exec_ok(*tdb.con, "ROLLBACK");
}

TEST_CASE("run_mvcc_mask_jobs is a no-op for version-state-free tables",
          "[mvcc_mask_job][scan_manager]")
{
  auto& e = env();
  job_test_db tdb;
  exec_ok(*tdb.con, "CREATE TABLE t AS SELECT range::INTEGER AS k FROM range(50000)");
  exec_ok(*tdb.con, "CHECKPOINT");
  // Reopen so the session-written version managers unload: the provably-clean
  // shape (fresh ATTACH-then-pin).
  tdb.con.reset();
  tdb.db.reset();
  tdb.db  = std::make_unique<duckdb::DuckDB>(tdb.path);
  tdb.con = std::make_unique<duckdb::Connection>(*tdb.db);

  exec_ok(*tdb.con, "BEGIN TRANSACTION");
  auto& storage = resolve_storage(*tdb.con, "t");
  auto metadata = make_metadata(*tdb.con, storage);

  auto* host_space = e.mgr->get_memory_space(cucascade::memory::Tier::HOST, 0);
  mvcc_mask_job_request request;
  request.masks        = std::make_shared<mvcc_chunk_mask_set>(1);
  request.metadata     = metadata;
  request.storage      = &storage;
  request.context      = tdb.con->context.get();
  request.chunk_spaces = {host_space};
  request.entry_name   = "t";
  std::vector<mvcc_mask_job_request> requests;
  requests.push_back(std::move(request));

  sirius::exec::static_thread_pool pool(2);
  sirius::exec::scoped_dispatcher dispatcher(pool, 2);
  run_mvcc_mask_jobs(requests, dispatcher, *e.mgr, e.topology);

  REQUIRE((*requests[0].masks)[0] == nullptr);
  exec_ok(*tdb.con, "ROLLBACK");
}

TEST_CASE("run_mvcc_mask_jobs handles two requests over the same table",
          "[mvcc_mask_job][scan_manager]")
{
  auto& e = env();
  job_test_db tdb;
  exec_ok(*tdb.con, "CREATE TABLE t AS SELECT range::INTEGER AS k FROM range(60000)");
  exec_ok(*tdb.con, "CHECKPOINT");
  exec_ok(*tdb.con, "DELETE FROM t WHERE rowid < 10");

  exec_ok(*tdb.con, "BEGIN TRANSACTION");
  auto& storage = resolve_storage(*tdb.con, "t");
  auto metadata = make_metadata(*tdb.con, storage);

  auto* host_space = e.mgr->get_memory_space(cucascade::memory::Tier::HOST, 0);
  std::vector<mvcc_mask_job_request> requests;
  for (int i = 0; i < 2; ++i) {  // self-join analog: one request per scan op
    mvcc_mask_job_request request;
    request.masks        = std::make_shared<mvcc_chunk_mask_set>(1);
    request.metadata     = metadata;
    request.storage      = &storage;
    request.context      = tdb.con->context.get();
    request.chunk_spaces = {host_space};
    request.entry_name   = "t";
    requests.push_back(std::move(request));
  }

  sirius::exec::static_thread_pool pool(4);
  sirius::exec::scoped_dispatcher dispatcher(pool, 4);
  run_mvcc_mask_jobs(requests, dispatcher, *e.mgr, e.topology);

  for (auto const& request : requests) {
    auto const& mask = (*request.masks)[0];
    REQUIRE(mask != nullptr);
    for (std::size_t r = 0; r < 32; ++r) {
      REQUIRE(bit_at(mask->words, r) == (r >= 10));
    }
  }
  exec_ok(*tdb.con, "ROLLBACK");
}
