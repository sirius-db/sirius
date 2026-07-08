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

/**
 * @file test_pin_table_mvcc_foundation.cpp
 * @brief Foundation tests for query-time MVCC delta merge over the duckdb-native
 *        pinned cache (#819): a duckdb-format pin captures MVCC snapshot metadata
 *        (v_base + per-chunk row counts) on its pinned_entry, suppresses WAL
 *        auto-checkpoint so the pinned disk image cannot shift underneath the
 *        cache, and the table keeps taking INSERT/DELETE traffic while pinned.
 *        The serve path is intentionally unchanged at this stage — nothing here
 *        asserts query-time visibility.
 */

#include "op/scan/duckdb_native_gpu_ingestible.hpp"
#include "pin_table.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "sirius_context.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/catalog/catalog.hpp>
#include <duckdb/common/limits.hpp>
#include <duckdb/main/config.hpp>
#include <duckdb/transaction/duck_transaction.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/transparent_execution_test_utils.hpp>

#include <cstddef>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

using PinMvccFixture = sirius::test::GpuExecutionFixture;

namespace {

/// PR1-relevant state of a pinned entry, copied out inside visit_pinned_entries
/// so no reference to the entry escapes the visitor.
struct entry_probe {
  bool found                   = false;
  bool has_mvcc                = false;
  duckdb::transaction_t v_base = 0;
  std::vector<std::size_t> counts;
  std::size_t n_cache     = 0;
  std::size_t num_rows    = 0;
  std::size_t gpu_chunks  = 0;
  std::size_t host_chunks = 0;
  /// The entry's REAL per-chunk row counts, read back from the cached data
  /// itself (GPU tier: any column's chunk sizes; HOST tier: each host chunk's
  /// rows) — what the recorded counts must match elementwise.
  std::vector<std::size_t> actual_chunk_rows;
};

entry_probe probe_entry(duckdb::Connection& con, const std::string& name)
{
  auto sirius_ctx = sirius::test::get_registered_sirius_context(con);
  REQUIRE(sirius_ctx != nullptr);
  entry_probe out;
  sirius_ctx->get_scan_manager().visit_pinned_entries(
    [&](std::string_view entry_name, const sirius::scan_manager::pinned_entry& entry) {
      if (entry_name != name) { return true; }  // keep scanning
      out.found    = true;
      out.has_mvcc = entry.mvcc != nullptr;
      if (entry.mvcc) {
        out.v_base  = entry.mvcc->v_base;
        out.counts  = entry.mvcc->base_row_count_per_chunk;
        out.n_cache = entry.mvcc->n_cache();
      }
      out.num_rows    = entry.num_rows;
      out.gpu_chunks  = entry.chunk_memory_spaces.size();
      out.host_chunks = entry.host_chunks.size();
      if (!entry.data_batches_by_column.empty()) {
        for (auto const& chunk : entry.data_batches_by_column.begin()->second) {
          out.actual_chunk_rows.push_back(static_cast<std::size_t>(chunk->size()));
        }
      } else {
        for (auto const& chunk : entry.host_chunks) {
          auto const& host_table = chunk->get_host_table();
          out.actual_chunk_rows.push_back(
            host_table && !host_table->columns.empty()
              ? static_cast<std::size_t>(host_table->columns.front().num_rows)
              : 0);
        }
      }
      return false;  // stop
    });
  return out;
}

}  // namespace

TEST_CASE_METHOD(PinMvccFixture,
                 "pin_table mvcc - duckdb GPU pin captures v_base and per-chunk counts",
                 "[integration][gpu_execution][pin_table_mvcc]")
{
  // ~2.5 row groups worth of rows so the pin spans several row groups and the
  // whole-row-group chunk validation in materialize_pin_batches gets real work.
  run_ok("CREATE TABLE mvcc_gpu_t AS SELECT range AS a, range * 2 AS b FROM range(300000);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='mvcc_gpu_t', tier='gpu');");

  auto probe = probe_entry(*con, "mvcc_gpu_t");
  REQUIRE(probe.found);
  REQUIRE(probe.has_mvcc);
  REQUIRE(probe.v_base > 0);
  REQUIRE_FALSE(probe.counts.empty());
  // Counts are per materialized chunk, parallel to the entry's chunk placement,
  // and must equal the cached chunks' real row counts elementwise.
  REQUIRE(probe.counts.size() == probe.gpu_chunks);
  REQUIRE(probe.counts == probe.actual_chunk_rows);
  // The pin covers the checkpointed rowid prefix exactly: N_cache == cached rows.
  REQUIRE(probe.n_cache == 300000);
  REQUIRE(probe.n_cache == probe.num_rows);

  // The pin still serves queries (serve path untouched by the foundation).
  compare_gpu_vs_cpu("SELECT count(*), sum(a), sum(b) FROM mvcc_gpu_t;");

  run_ok("CALL unpin_table('mvcc_gpu_t');");
  REQUIRE_FALSE(probe_entry(*con, "mvcc_gpu_t").found);
}

TEST_CASE_METHOD(PinMvccFixture,
                 "pin_table mvcc - duckdb HOST pin captures v_base and per-chunk counts",
                 "[integration][gpu_execution][pin_table_mvcc]")
{
  run_ok("CREATE TABLE mvcc_host_t AS SELECT range AS a FROM range(200000);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='mvcc_host_t', tier='host');");

  auto probe = probe_entry(*con, "mvcc_host_t");
  REQUIRE(probe.found);
  REQUIRE(probe.has_mvcc);
  REQUIRE(probe.v_base > 0);
  REQUIRE_FALSE(probe.counts.empty());
  // Host tier: counts are parallel to the pinned host chunks (one per batch)
  // and must equal each chunk's real row count.
  REQUIRE(probe.counts.size() == probe.host_chunks);
  REQUIRE(probe.counts == probe.actual_chunk_rows);
  REQUIRE(probe.n_cache == 200000);
  REQUIRE(probe.n_cache == probe.num_rows);

  run_ok("CALL unpin_table('mvcc_host_t');");
}

TEST_CASE_METHOD(PinMvccFixture,
                 "pin_table mvcc - v_base equals the pin transaction's start_time on the pinned "
                 "catalog",
                 "[integration][gpu_execution][pin_table_mvcc]")
{
  run_ok("CREATE TABLE mvcc_vbase_t AS SELECT range AS a FROM range(10000);");
  run_ok("CHECKPOINT;");

  // Pin inside an explicit transaction and read the SAME transaction's
  // start_time on the pinned table's own catalog: the captured fence must equal
  // it exactly. This pins down both the timing (the pin transaction, not some
  // later one) and the MVCC domain (the attached database's counter, not the
  // default in-memory catalog's — each AttachedDatabase counts independently).
  run_ok("BEGIN TRANSACTION;");
  run_ok("CALL pin_table(format='duckdb', name='mvcc_vbase_t', tier='gpu');");
  auto& pinned_catalog = duckdb::Catalog::GetCatalog(*con->context, attach_alias);
  auto const expected  = duckdb::DuckTransaction::Get(*con->context, pinned_catalog).start_time;
  run_ok("COMMIT;");

  auto probe = probe_entry(*con, "mvcc_vbase_t");
  REQUIRE(probe.has_mvcc);
  REQUIRE(probe.v_base == expected);

  run_ok("CALL unpin_table('mvcc_vbase_t');");
}

TEST_CASE_METHOD(PinMvccFixture,
                 "pin_table mvcc - multi-chunk pin records the per-chunk rowid partition",
                 "[integration][gpu_execution][pin_table_mvcc]")
{
  // integration.yaml caps scan_task_batch_size at 100 MB; 13M BIGINTs decode to
  // ~104 MB, so the coalescer must emit at least two chunks — exercising the
  // per-chunk map beyond the single-chunk tautology.
  run_ok("CREATE TABLE mvcc_multi_t AS SELECT range AS a FROM range(13000000);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='mvcc_multi_t', tier='gpu');");

  auto probe = probe_entry(*con, "mvcc_multi_t");
  REQUIRE(probe.has_mvcc);
  REQUIRE(probe.counts.size() >= 2);
  REQUIRE(probe.counts == probe.actual_chunk_rows);
  REQUIRE(probe.n_cache == 13000000);
  // Chunks are whole row groups, so every chunk but the table's last (which owns
  // the one partial row group) covers a multiple of ROW_GROUP_SIZE rows.
  for (std::size_t i = 0; i + 1 < probe.counts.size(); ++i) {
    REQUIRE(probe.counts[i] % 122880 == 0);
  }

  run_ok("CALL unpin_table('mvcc_multi_t');");
}

TEST_CASE_METHOD(PinMvccFixture,
                 "pin_table mvcc - parquet pin carries no MVCC metadata",
                 "[integration][gpu_execution][pin_table_mvcc]")
{
  auto parquet_path = temp_db_path + "_mvcc.parquet";
  run_ok("COPY (SELECT range AS a FROM range(1000)) TO '" + parquet_path + "' (FORMAT parquet);");
  run_ok("CALL pin_table('" + parquet_path + "', name='mvcc_pq', tier='gpu');");

  auto probe = probe_entry(*con, "mvcc_pq");
  REQUIRE(probe.found);
  REQUIRE_FALSE(probe.has_mvcc);

  run_ok("CALL unpin_table('mvcc_pq');");
  std::filesystem::remove(parquet_path);
}

TEST_CASE_METHOD(PinMvccFixture,
                 "pin_table mvcc - re-pin refreshes v_base through the merge path",
                 "[integration][gpu_execution][pin_table_mvcc]")
{
  run_ok("CREATE TABLE mvcc_repin_t AS SELECT range AS a FROM range(50000);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='mvcc_repin_t', tier='gpu');");
  auto const first = probe_entry(*con, "mvcc_repin_t");
  REQUIRE(first.has_mvcc);

  // Committed write transactions on the attached database advance its MVCC
  // counter, so the second pin's transaction has a strictly newer start_time.
  run_ok("CREATE TABLE mvcc_repin_bump AS SELECT 1 AS x;");
  run_ok("DROP TABLE mvcc_repin_bump;");

  // No checkpoint ran in between, so the decoded row count is unchanged and this
  // re-pin takes insert_pinned_entry's MERGE path — the metadata must still be
  // refreshed (attach overwrites) with the newer, more conservative fence.
  run_ok("CALL pin_table(format='duckdb', name='mvcc_repin_t', tier='gpu');");
  auto const second = probe_entry(*con, "mvcc_repin_t");
  REQUIRE(second.has_mvcc);
  REQUIRE(second.v_base > first.v_base);
  REQUIRE(second.n_cache == first.n_cache);

  run_ok("CALL unpin_table('mvcc_repin_t');");
}

TEST_CASE_METHOD(PinMvccFixture,
                 "pin_table mvcc - duckdb pin suppresses WAL auto-checkpoint",
                 "[integration][gpu_execution][pin_table_mvcc]")
{
  run_ok("CREATE TABLE mvcc_wal_t (a BIGINT);");
  run_ok("INSERT INTO mvcc_wal_t SELECT range FROM range(100000);");
  run_ok("CHECKPOINT;");
  // Arm the SECOND auto-checkpoint trigger (entry count) before pinning: the pin
  // must disarm it too, or the 40 commits below would checkpoint and truncate
  // the WAL regardless of the size threshold.
  run_ok("SET GLOBAL wal_autocheckpoint_entries = 5;");
  run_ok("CALL pin_table(format='duckdb', name='mvcc_wal_t', tier='gpu');");

  // The pin disables the size-based trigger outright (the DBConfig is shared by
  // every attached database) and zeroes the entry-count trigger.
  auto& config = duckdb::DBConfig::GetConfig(*con->context);
  REQUIRE(config.options.checkpoint_wal_size == duckdb::NumericLimits<duckdb::idx_t>::Maximum());
  auto entries_setting = con->Query("SELECT current_setting('wal_autocheckpoint_entries');");
  REQUIRE_FALSE(entries_setting->HasError());
  REQUIRE(entries_setting->GetValue(0, 0).GetValue<int64_t>() == 0);

  // Write well past the 16 MiB default threshold across many commits. Under the
  // default config the commit crossing it would auto-checkpoint and truncate the
  // WAL; with the pin's suppression the WAL just keeps growing. The commits stay
  // below a row group (122,880 rows) each: DuckDB writes larger appends
  // optimistically — data blocks straight to the db file, only references in the
  // WAL — so bulk inserts would barely grow the WAL at all.
  for (int i = 0; i < 40; ++i) {
    run_ok("INSERT INTO mvcc_wal_t SELECT range FROM range(60000);");
  }
  auto const wal_path = temp_db_path + ".wal";
  REQUIRE(std::filesystem::exists(wal_path));
  REQUIRE(std::filesystem::file_size(wal_path) > (16ULL << 20));

  run_ok("CALL unpin_table('mvcc_wal_t');");
}

TEST_CASE_METHOD(PinMvccFixture,
                 "pin_table mvcc - INSERT and DELETE run while the table is pinned",
                 "[integration][gpu_execution][pin_table_mvcc]")
{
  run_ok("CREATE TABLE mvcc_dml_t AS SELECT range AS a FROM range(100000);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='mvcc_dml_t', tier='gpu');");
  auto const before = probe_entry(*con, "mvcc_dml_t");
  REQUIRE(before.has_mvcc);

  // The pin takes no table lock: normal DML must proceed while the cache exists.
  run_ok("INSERT INTO mvcc_dml_t SELECT range FROM range(1000);");
  run_ok("DELETE FROM mvcc_dml_t WHERE a < 500;");

  // The entry (and its snapshot metadata) is untouched by the concurrent writes.
  auto const after = probe_entry(*con, "mvcc_dml_t");
  REQUIRE(after.found);
  REQUIRE(after.has_mvcc);
  REQUIRE(after.v_base == before.v_base);
  REQUIRE(after.n_cache == before.n_cache);

  run_ok("CALL unpin_table('mvcc_dml_t');");
}

TEST_CASE_METHOD(PinMvccFixture,
                 "pin_table mvcc - attach_mvcc_metadata without an entry throws",
                 "[integration][pin_table_mvcc]")
{
  auto sirius_ctx = sirius::test::get_registered_sirius_context(*con);
  REQUIRE(sirius_ctx != nullptr);
  REQUIRE_THROWS_AS(sirius_ctx->get_scan_manager().attach_mvcc_metadata("mvcc_no_such_entry", {}),
                    std::invalid_argument);
}

//===----------------------------------------------------------------------===//
// validate_duckdb_pin_chunk failure paths (pure unit tests, no GPU / fixture)
//===----------------------------------------------------------------------===//

namespace {

sirius::op::scan::duckdb_row_group_metadata make_rg(duckdb::idx_t index,
                                                    duckdb::idx_t start,
                                                    duckdb::idx_t count)
{
  sirius::op::scan::duckdb_row_group_metadata rg;
  rg.row_group_index = index;
  rg.row_group_start = start;
  rg.row_count       = count;
  return rg;
}

}  // namespace

TEST_CASE("pin_table mvcc - chunk validator accepts contiguous whole row groups",
          "[pin_table_mvcc]")
{
  sirius::op::scan::duckdb_native_scan_info first;
  first.row_groups.push_back(make_rg(0, 0, 122880));
  first.row_groups.push_back(make_rg(1, 122880, 122880));
  REQUIRE_NOTHROW(sirius::validate_duckdb_pin_chunk(first, 245760, 0));

  // A later chunk continues exactly where the previous one ended.
  sirius::op::scan::duckdb_native_scan_info second;
  second.row_groups.push_back(make_rg(2, 245760, 1000));
  REQUIRE_NOTHROW(sirius::validate_duckdb_pin_chunk(second, 1000, 245760));
}

TEST_CASE("pin_table mvcc - chunk validator rejects a rowid gap between chunks", "[pin_table_mvcc]")
{
  // The chunk starts one row group in while the pin has materialized nothing:
  // a skipped/reordered row group must fail the pin.
  sirius::op::scan::duckdb_native_scan_info batch;
  batch.row_groups.push_back(make_rg(1, 122880, 122880));
  REQUIRE_THROWS_AS(sirius::validate_duckdb_pin_chunk(batch, 122880, 0), std::runtime_error);
}

TEST_CASE("pin_table mvcc - chunk validator rejects out-of-order row groups within a chunk",
          "[pin_table_mvcc]")
{
  sirius::op::scan::duckdb_native_scan_info batch;
  batch.row_groups.push_back(make_rg(1, 122880, 122880));
  batch.row_groups.push_back(make_rg(0, 0, 122880));
  REQUIRE_THROWS_AS(sirius::validate_duckdb_pin_chunk(batch, 245760, 0), std::runtime_error);
}

TEST_CASE("pin_table mvcc - chunk validator rejects a decoded/metadata row-count mismatch",
          "[pin_table_mvcc]")
{
  // The decoder produced fewer rows than the row-group metadata covers (e.g. a
  // future coalescer slicing a row group across batches).
  sirius::op::scan::duckdb_native_scan_info batch;
  batch.row_groups.push_back(make_rg(0, 0, 122880));
  REQUIRE_THROWS_AS(sirius::validate_duckdb_pin_chunk(batch, 100000, 0), std::runtime_error);
}

TEST_CASE("pin_table mvcc - chunk validator skips non-duckdb batches", "[pin_table_mvcc]")
{
  // Parquet (or any other format's) batches carry no duckdb row-group metadata;
  // the validator must not constrain them.
  sirius::op::scan::scan_info parquet_like;
  REQUIRE_NOTHROW(sirius::validate_duckdb_pin_chunk(parquet_like, 12345, 0));
}
