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

// `SELECT ... FROM read_simpatico('x.hpln')` end to end: bind, GPU plan, decode, results.
//
// There is no DuckDB reader for .hpln, so the usual GPU-vs-CPU comparison is impossible -- the
// CPU has nothing to compare against. Everything below is therefore checked against the values
// the fixture writer was handed, and every query additionally asserts a real GPU execution with
// no fallback, because a query that quietly fell back to the CPU would fail loudly rather than
// return these rows.
//
// Value assertions are aggregates rather than a row dump on purpose: a wrong payload offset in
// this decode path does not fault, it returns neighbouring bytes, and a sum over 6000 rows
// catches that where a row count does not.

#include "compression/simpatico_file_ingest.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cuda_runtime_api.h>

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/transparent_execution_test_utils.hpp>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

constexpr int kRows = 6000;

std::vector<std::int32_t> ramp(int n, int stride)
{
  std::vector<std::int32_t> v(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    v[static_cast<std::size_t>(i)] = i * stride + (i % 7);
  }
  return v;
}

std::unique_ptr<cudf::column> int32_column(std::vector<std::int32_t> const& values)
{
  auto col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                       static_cast<cudf::size_type>(values.size()),
                                       cudf::mask_state::UNALLOCATED);
  REQUIRE(cudaMemcpy(col->mutable_view().head<std::int32_t>(),
                     values.data(),
                     values.size() * sizeof(std::int32_t),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  return col;
}

/// Writes a three-column INT32 .hpln at @p path and returns the values it was given.
std::vector<std::vector<std::int32_t>> write_fixture(
  std::string const& path, duckdb::vector<duckdb::LogicalType> const& types)
{
  std::vector<std::vector<std::int32_t>> values{ramp(kRows, 3), ramp(kRows, 11), ramp(kRows, 5)};
  std::vector<std::unique_ptr<cudf::column>> cols;
  for (auto const& v : values) {
    cols.push_back(int32_column(v));
  }
  auto table = std::make_unique<cudf::table>(std::move(cols));

  auto const leaf = std::string("input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n");
  auto const err  = sirius::write_table_to_hpln(table->view(),
                                               types,
                                                {"a", "b", "c"},
                                               leaf + "---\n" + leaf + "---\n" + leaf,
                                               /*group_rows=*/1024,
                                               path,
                                               cudf::get_default_stream(),
                                               rmm::mr::get_current_device_resource_ref());
  REQUIRE(err.empty());
  return values;
}

/// A three-INTEGER file: every column's declared type is carried natively by the cuDF type the
/// file decodes to, which is the only shape this reader serves (see the carrier-mismatch case at
/// the bottom of this file).
class ReadSimpaticoFixture : public sirius::test::GpuExecutionFixture {
 public:
  ReadSimpaticoFixture()
  {
    dir = fs::temp_directory_path() /
          ("sirius_read_simpatico_" + std::to_string(::getpid()) + "_" + std::to_string(::rand()));
    fs::create_directories(dir);
    path   = (dir / "t.hpln").string();
    values = write_fixture(path,
                           {duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER),
                            duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER),
                            duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER)});

    run_ok("SET gpu_execution = true;");
    // A GPU failure on a .hpln query would otherwise be replayed on the CPU and surface as the
    // table function's "no CPU reader" error, hiding the real cause. Off, the GPU error is the
    // error.
    run_ok("SET enable_duckdb_fallback = false;");
  }

  ~ReadSimpaticoFixture() { fs::remove_all(dir); }

  /// Run @p sql and assert it was a GPU execution with no fallback of either kind.
  duckdb::unique_ptr<duckdb::MaterializedQueryResult> query_on_gpu(std::string const& sql)
  {
    auto const before = sirius::test::get_transparent_execution_stats(*con);
    auto result       = con->Query(sql);
    auto const after  = sirius::test::get_transparent_execution_stats(*con);
    REQUIRE(result);
    if (result->HasError()) { UNSCOPED_INFO("query error: " << result->GetError()); }
    REQUIRE_FALSE(result->HasError());
    sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1);
    return duckdb::unique_ptr<duckdb::MaterializedQueryResult>(
      static_cast<duckdb::MaterializedQueryResult*>(result.release()));
  }

  std::int64_t sum(std::size_t column) const
  {
    std::int64_t total = 0;
    for (auto v : values[column]) {
      total += v;
    }
    return total;
  }

  std::string from() const { return " FROM read_simpatico('" + path + "')"; }

  fs::path dir;
  std::string path;
  std::vector<std::vector<std::int32_t>> values;
};

constexpr int kMultiChunks       = 5;
constexpr int kMultiRowsPerChunk = 1200;

/// A five-chunk file. `k` encodes the chunk that holds the row (c*1'000'000 + i), so a GROUP BY
/// over k/1'000'000 says exactly which chunks the scan produced and how many rows each
/// contributed -- the questions a total row count cannot answer.
void write_multi_fixture(std::string const& path)
{
  std::vector<std::unique_ptr<cudf::table>> tables;
  std::vector<cudf::table_view> views;
  for (int c = 0; c < kMultiChunks; ++c) {
    std::vector<std::int32_t> keys(kMultiRowsPerChunk), vals(kMultiRowsPerChunk);
    for (int i = 0; i < kMultiRowsPerChunk; ++i) {
      keys[static_cast<std::size_t>(i)] = c * 1000000 + i;
      vals[static_cast<std::size_t>(i)] = i * 3 + c;
    }
    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.push_back(int32_column(keys));
    cols.push_back(int32_column(vals));
    tables.push_back(std::make_unique<cudf::table>(std::move(cols)));
    views.push_back(tables.back()->view());
  }
  duckdb::vector<duckdb::LogicalType> const types{
    duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER),
    duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER)};
  auto const leaf = std::string("input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n");
  auto const err  = sirius::write_tables_to_hpln(views,
                                                types,
                                                 {"k", "v"},
                                                leaf + "---\n" + leaf,
                                                /*group_rows=*/1024,
                                                path,
                                                cudf::get_default_stream(),
                                                rmm::mr::get_current_device_resource_ref());
  REQUIRE(err.empty());
}

class ReadSimpaticoMultiChunkFixture : public sirius::test::GpuExecutionFixture {
 public:
  ReadSimpaticoMultiChunkFixture()
  {
    dir = fs::temp_directory_path() / ("sirius_read_simpatico_multi_" + std::to_string(::getpid()) +
                                       "_" + std::to_string(::rand()));
    fs::create_directories(dir);
    path = (dir / "multi.hpln").string();
    write_multi_fixture(path);
    run_ok("SET gpu_execution = true;");
    run_ok("SET enable_duckdb_fallback = false;");
  }

  ~ReadSimpaticoMultiChunkFixture() { fs::remove_all(dir); }

  duckdb::unique_ptr<duckdb::MaterializedQueryResult> query_on_gpu(std::string const& sql)
  {
    auto const before = sirius::test::get_transparent_execution_stats(*con);
    auto result       = con->Query(sql);
    auto const after  = sirius::test::get_transparent_execution_stats(*con);
    REQUIRE(result);
    if (result->HasError()) { UNSCOPED_INFO("query error: " << result->GetError()); }
    REQUIRE_FALSE(result->HasError());
    sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1);
    return duckdb::unique_ptr<duckdb::MaterializedQueryResult>(
      static_cast<duckdb::MaterializedQueryResult*>(result.release()));
  }

  std::string from() const { return " FROM read_simpatico('" + path + "')"; }

  fs::path dir;
  std::string path;
};

}  // namespace

TEST_CASE_METHOD(ReadSimpaticoMultiChunkFixture,
                 "read_simpatico - a multi-chunk file returns every chunk exactly once",
                 "[integration][read_simpatico][simpatico_multichunk]")
{
  // One row per chunk, so this fails differently for each way the walk can go wrong: a missing
  // chunk drops a group, a duplicated one doubles its count, and a batch that read the wrong
  // chunk shifts a group id.
  auto result = query_on_gpu("SELECT k // 1000000 AS chunk, count(*), min(k), max(k), sum(v)" +
                             from() + " GROUP BY 1 ORDER BY 1;");
  REQUIRE(result->RowCount() == static_cast<duckdb::idx_t>(kMultiChunks));
  for (int c = 0; c < kMultiChunks; ++c) {
    auto const row = static_cast<duckdb::idx_t>(c);
    REQUIRE(result->GetValue(0, row).GetValue<std::int32_t>() == c);
    REQUIRE(result->GetValue(1, row).GetValue<std::int64_t>() == kMultiRowsPerChunk);
    REQUIRE(result->GetValue(2, row).GetValue<std::int32_t>() == c * 1000000);
    REQUIRE(result->GetValue(3, row).GetValue<std::int32_t>() ==
            c * 1000000 + kMultiRowsPerChunk - 1);
    std::int64_t expected_v = 0;
    for (int i = 0; i < kMultiRowsPerChunk; ++i) {
      expected_v += i * 3 + c;
    }
    REQUIRE(result->GetValue(4, row).GetValue<std::int64_t>() == expected_v);
  }
}

TEST_CASE_METHOD(ReadSimpaticoMultiChunkFixture,
                 "read_simpatico - a multi-chunk file's cardinality is the whole file",
                 "[integration][read_simpatico][simpatico_multichunk]")
{
  // The bind sums the chunk row counts; reporting one chunk's would give the optimizer a
  // cardinality five times too small, which is a plan-quality bug that returns right answers.
  auto result = query_on_gpu("SELECT count(*)" + from() + ";");
  REQUIRE(result->GetValue(0, 0).GetValue<std::int64_t>() ==
          static_cast<std::int64_t>(kMultiChunks) * kMultiRowsPerChunk);

  auto filtered = query_on_gpu("SELECT count(*)" + from() + " WHERE k >= 2000000 AND k < 4000000;");
  REQUIRE(filtered->GetValue(0, 0).GetValue<std::int64_t>() == 2 * kMultiRowsPerChunk);
}

TEST_CASE_METHOD(ReadSimpaticoFixture,
                 "read_simpatico - binds the file's names and declared types",
                 "[integration][read_simpatico]")
{
  auto result = query_on_gpu("SELECT *" + from() + " LIMIT 0;");
  REQUIRE(result->ColumnCount() == 3);
  REQUIRE(result->names == duckdb::vector<std::string>{"a", "b", "c"});
  for (auto const& type : result->types) {
    REQUIRE(type.id() == duckdb::LogicalTypeId::INTEGER);
  }
}

TEST_CASE_METHOD(ReadSimpaticoFixture,
                 "read_simpatico - SELECT * returns every row of the file",
                 "[integration][read_simpatico]")
{
  auto result = query_on_gpu("SELECT *" + from() + ";");
  REQUIRE(result->ColumnCount() == 3);
  REQUIRE(result->RowCount() == static_cast<duckdb::idx_t>(kRows));
}

TEST_CASE_METHOD(ReadSimpaticoFixture,
                 "read_simpatico - decoded values are the ones that were written",
                 "[integration][read_simpatico]")
{
  // SUM over the whole file rather than a spot check: it is the aggregate a shifted payload
  // offset cannot survive. Each column is summed separately because the three ramps differ, so a
  // column read in place of another shows up here too.
  auto result =
    query_on_gpu("SELECT count(*), sum(a), sum(b), sum(c), min(b), max(b)" + from() + ";");
  REQUIRE(result->RowCount() == 1);
  REQUIRE(result->GetValue(0, 0).GetValue<std::int64_t>() == kRows);
  REQUIRE(result->GetValue(1, 0).GetValue<std::int64_t>() == sum(0));
  REQUIRE(result->GetValue(2, 0).GetValue<std::int64_t>() == sum(1));
  REQUIRE(result->GetValue(3, 0).GetValue<std::int64_t>() == sum(2));
  REQUIRE(result->GetValue(4, 0).GetValue<std::int32_t>() == values[1].front());
  REQUIRE(result->GetValue(5, 0).GetValue<std::int32_t>() == values[1].back());
}

TEST_CASE_METHOD(ReadSimpaticoFixture,
                 "read_simpatico - a two-of-three projection reads the columns it names",
                 "[integration][read_simpatico]")
{
  // The scan emits only the columns the query touches, so this is where a projection that
  // confused file order with emission order shows up: it would filter on `a` and sum `a` too.
  // Both the filter and the aggregate run above the scan -- the single-chunk ingestible declares
  // no pushdown.
  std::int64_t expected_rows = 0;
  std::int64_t expected_sum  = 0;
  for (int i = 0; i < kRows; ++i) {
    if (values[0][static_cast<std::size_t>(i)] < 300) {
      ++expected_rows;
      expected_sum += values[1][static_cast<std::size_t>(i)];
    }
  }
  REQUIRE(expected_rows > 0);
  REQUIRE(expected_rows < kRows);

  auto result = query_on_gpu("SELECT count(*), sum(b)" + from() + " WHERE a < 300;");
  REQUIRE(result->GetValue(0, 0).GetValue<std::int64_t>() == expected_rows);
  REQUIRE(result->GetValue(1, 0).GetValue<std::int64_t>() == expected_sum);
}

TEST_CASE_METHOD(ReadSimpaticoFixture,
                 "read_simpatico - COUNT(*) needs no column",
                 "[integration][read_simpatico]")
{
  auto result = query_on_gpu("SELECT count(*)" + from() + ";");
  REQUIRE(result->GetValue(0, 0).GetValue<std::int64_t>() == kRows);
}

TEST_CASE_METHOD(ReadSimpaticoFixture,
                 "read_simpatico - without the GPU there is no reader, and it says so",
                 "[integration][read_simpatico]")
{
  // .hpln has no DuckDB CPU reader, so a CPU execution cannot return rows. The contract is that
  // it errors clearly rather than falling back to something that half-works: the table function's
  // own execute callback is the only place that knows why there are no rows.
  run_ok("SET gpu_execution = false;");
  auto result = con->Query("SELECT count(*)" + from() + ";");
  run_ok("SET gpu_execution = true;");
  REQUIRE(result);
  REQUIRE(result->HasError());
  REQUIRE(result->GetError().find("read_simpatico requires GPU execution") != std::string::npos);
}

TEST_CASE_METHOD(ReadSimpaticoFixture,
                 "read_simpatico - an unreadable path fails at bind",
                 "[integration][read_simpatico]")
{
  // Bind-time refusal keeps a bad path from becoming a mid-query error after the plan has
  // committed to the GPU.
  auto missing =
    con->Query("SELECT * FROM read_simpatico('" + (dir / "nope.hpln").string() + "');");
  REQUIRE(missing);
  REQUIRE(missing->HasError());
}

TEST_CASE_METHOD(ReadSimpaticoFixture,
                 "read_simpatico - a column the file cannot carry is refused, not misread",
                 "[integration][read_simpatico]")
{
  // Same INT32 payload, but declared DATE and DECIMAL(12,2) -- types the engine carries as
  // TIMESTAMP_DAYS and DECIMAL64. Reading the raw INT32 columns under those declarations would
  // reinterpret the bytes rather than convert them, so the plan must refuse. Column `b` stays
  // INTEGER to show the refusal is per column and not a blanket rejection of the file.
  auto const mixed = (dir / "mixed.hpln").string();
  write_fixture(mixed,
                {duckdb::LogicalType(duckdb::LogicalTypeId::DATE),
                 duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER),
                 duckdb::LogicalType::DECIMAL(12, 2)});

  auto refused = con->Query("SELECT sum(c) FROM read_simpatico('" + mixed + "');");
  REQUIRE(refused);
  REQUIRE(refused->HasError());
  REQUIRE(refused->GetError().find("does not convert carriers") != std::string::npos);

  auto served = query_on_gpu("SELECT sum(b) FROM read_simpatico('" + mixed + "');");
  REQUIRE(served->GetValue(0, 0).GetValue<std::int64_t>() == sum(1));
}
