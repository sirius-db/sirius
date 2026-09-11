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

// `COPY (SELECT ...) TO 'x.hpln' (FORMAT simpatico)` and back through read_simpatico().
//
// The test that matters is the round trip, VALUE by value: a wrong chunk boundary, a dropped
// chunk or a permuted column in this path does not fault, it returns plausible rows. Row counts
// alone would pass for all three, so every case below compares the rows themselves against the
// source table -- which is also the only available oracle, since DuckDB has no .hpln reader to
// compare against.

#include "compression/simpatico_file_ingest.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/transparent_execution_test_utils.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

/// A table whose values are distinct per row and per column, so any mix-up between rows, chunks
/// or columns changes a value rather than merely reordering equal ones.
constexpr int kRows = 6000;

class SimpaticoCopyFixture : public sirius::test::GpuExecutionFixture {
 public:
  SimpaticoCopyFixture()
  {
    dir = fs::temp_directory_path() /
          ("sirius_hpln_copy_" + std::to_string(::getpid()) + "_" + std::to_string(::rand()));
    fs::create_directories(dir);

    run_ok("SET gpu_execution = true;");
    run_ok("CREATE TABLE src (a INTEGER, b BIGINT, c DOUBLE, s VARCHAR);");
    // Loaded while the CPU fallback is still available: casting an INTEGER to VARCHAR is not a
    // GPU cast, and this fixture is about the writer, not about that.
    run_ok(
      "INSERT INTO src SELECT i * 3 + (i % 7), i * 1000003, i * 0.5, 'row-' || i::VARCHAR "
      "FROM range(" +
      std::to_string(kRows) + ") t(i);");
    run_ok("CHECKPOINT;");

    // A GPU failure over a .hpln would otherwise be replayed on the CPU and surface as the table
    // function's "no CPU reader" error, hiding the real cause.
    run_ok("SET enable_duckdb_fallback = false;");
  }

  ~SimpaticoCopyFixture() { fs::remove_all(dir); }

  [[nodiscard]] std::string path(std::string const& name) const { return (dir / name).string(); }

  /// Run @p sql, requiring success, and return the materialized result.
  duckdb::unique_ptr<duckdb::MaterializedQueryResult> query(std::string const& sql)
  {
    auto result = con->Query(sql);
    REQUIRE(result);
    if (result->HasError()) { UNSCOPED_INFO("query error: " << result->GetError()); }
    REQUIRE_FALSE(result->HasError());
    return duckdb::unique_ptr<duckdb::MaterializedQueryResult>(
      static_cast<duckdb::MaterializedQueryResult*>(result.release()));
  }

  /// Run @p sql and require it to fail, returning the error text.
  std::string query_error(std::string const& sql)
  {
    auto result = con->Query(sql);
    REQUIRE(result);
    REQUIRE(result->HasError());
    return result->GetError();
  }

  fs::path dir;
};

/// Rows of @p sql in emitted order, stringified.
std::vector<std::vector<std::string>> ordered_rows(duckdb::MaterializedQueryResult& result)
{
  return sirius::test::collect_rows(result, /*sort=*/false);
}

}  // namespace

TEST_CASE_METHOD(SimpaticoCopyFixture,
                 "COPY to .hpln round-trips every value through read_simpatico",
                 "[integration][simpatico_copy]")
{
  auto const file = path("one_chunk.hpln");
  query("COPY (SELECT a, b, c FROM src ORDER BY a) TO '" + file + "' (FORMAT simpatico);");
  REQUIRE(fs::exists(file));

  // One chunk: 6000 rows is far under the 1 Mi-row default.
  auto const schema = sirius::read_hpln_schema(file);
  REQUIRE(schema.num_rows == kRows);
  REQUIRE(schema.chunk_rows.size() == 1);
  // Names must round-trip: they are what the reader binds against.
  REQUIRE(schema.names == std::vector<std::string>{"a", "b", "c"});
  REQUIRE(schema.types.size() == 3);
  REQUIRE(schema.types[0] == duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER));
  REQUIRE(schema.types[1] == duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));
  REQUIRE(schema.types[2] == duckdb::LogicalType(duckdb::LogicalTypeId::DOUBLE));

  auto written   = query("SELECT a, b, c FROM src ORDER BY a;");
  auto read_back = query("SELECT a, b, c FROM read_simpatico('" + file + "') ORDER BY a;");
  REQUIRE(read_back->RowCount() == static_cast<duckdb::idx_t>(kRows));
  REQUIRE(ordered_rows(*read_back) == ordered_rows(*written));
}

TEST_CASE_METHOD(SimpaticoCopyFixture,
                 "COPY to .hpln cuts the stream into chunk_rows-sized chunks",
                 "[integration][simpatico_copy]")
{
  auto const file = path("multi_chunk.hpln");
  // 1024 is smaller than DuckDB's 2048-row vector, so this also exercises a DataChunk that has
  // to be split across two file chunks -- the case a "flush when the staging is full enough"
  // policy would get wrong.
  query("COPY (SELECT a, b, c FROM src ORDER BY a) TO '" + file +
        "' (FORMAT simpatico, chunk_rows 1024);");

  auto const schema = sirius::read_hpln_schema(file);
  REQUIRE(schema.num_rows == kRows);
  REQUIRE(schema.chunk_rows.size() == 6);
  for (std::size_t i = 0; i < 5; i++) {
    REQUIRE(schema.chunk_rows[i] == 1024);
  }
  REQUIRE(schema.chunk_rows[5] == kRows - 5 * 1024);

  // Every chunk carries zone maps, not just the first -- a writer that packed only chunk 0's
  // bounds would still produce a readable file, and pruning would then be answered from the
  // wrong chunk's bounds.
  REQUIRE(schema.group_bounds.chunk_count() == 6);
  // Chunks hold the query's rows in the query's order, which is the only reason to write an
  // ORDER BY to a file at all: a clustered file is what the zone maps can prune. Consecutive
  // chunks' bounds on the sort key must therefore not overlap.
  std::int64_t previous_max = std::numeric_limits<std::int64_t>::min();
  for (std::size_t chunk = 0; chunk < 6; chunk++) {
    REQUIRE(schema.group_bounds.groups_in_chunk(chunk) > 0);
    auto const bounds = schema.group_bounds.cell(0, chunk);
    REQUIRE_FALSE(bounds.empty());
    auto const chunk_min = *std::min_element(bounds.mins.begin(), bounds.mins.end());
    auto const chunk_max = *std::max_element(bounds.maxs.begin(), bounds.maxs.end());
    REQUIRE(chunk_min > previous_max);
    previous_max = chunk_max;
  }

  auto written   = query("SELECT a, b, c FROM src ORDER BY a;");
  auto read_back = query("SELECT a, b, c FROM read_simpatico('" + file + "') ORDER BY a;");
  REQUIRE(read_back->RowCount() == static_cast<duckdb::idx_t>(kRows));
  REQUIRE(ordered_rows(*read_back) == ordered_rows(*written));
}

TEST_CASE_METHOD(SimpaticoCopyFixture,
                 "COPY to .hpln writes the query's column order, not the table's",
                 "[integration][simpatico_copy]")
{
  auto const file = path("permuted.hpln");
  query("COPY (SELECT c AS z, a AS x, b AS y FROM src ORDER BY x) TO '" + file +
        "' (FORMAT simpatico, chunk_rows 2048);");

  auto const schema = sirius::read_hpln_schema(file);
  REQUIRE(schema.names == std::vector<std::string>{"z", "x", "y"});
  REQUIRE(schema.types[0] == duckdb::LogicalType(duckdb::LogicalTypeId::DOUBLE));
  REQUIRE(schema.types[1] == duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER));
  REQUIRE(schema.types[2] == duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));

  auto written   = query("SELECT c, a, b FROM src ORDER BY a;");
  auto read_back = query("SELECT z, x, y FROM read_simpatico('" + file + "') ORDER BY x;");
  REQUIRE(ordered_rows(*read_back) == ordered_rows(*written));
}

/// A projection with a NULL in every position the writer supports: scattered over a fixed-width
/// column, in a contiguous run that covers whole zone-map groups, in a VARCHAR, in a column that
/// is entirely null, and -- as the control -- beside a column that has no nulls at all.
///
/// `i` is never null, so it can order the comparison and key the null positions.
constexpr char const* kNullableProjection =
  "SELECT i,"
  "  CASE WHEN i % 100 = 0 THEN NULL ELSE i * 3 END::INTEGER   AS scattered,"
  "  CASE WHEN i BETWEEN 2000 AND 2999 THEN NULL ELSE i END::BIGINT AS run,"
  "  (i * 0.5)::DOUBLE                                         AS never_null,"
  "  CASE WHEN i % 7 = 0 THEN NULL ELSE 'row-' || i::VARCHAR END AS text,"
  "  NULL::INTEGER                                             AS always_null "
  "FROM range(6000) t(i)";

TEST_CASE_METHOD(SimpaticoCopyFixture,
                 "COPY to .hpln round-trips NULLs in every supported position",
                 "[integration][simpatico_copy]")
{
  auto const file = path("nulls.hpln");
  query("CREATE TABLE nsrc AS " + std::string(kNullableProjection) + ";");
  query("CHECKPOINT;");
  // Three chunks, so a null run spans a chunk boundary and `always_null` produces a chunk whose
  // column is entirely null -- the sidecar shape that stores no payload bytes at all.
  query("COPY (SELECT * FROM nsrc ORDER BY i) TO '" + file +
        "' (FORMAT simpatico, chunk_rows 2048);");

  // Values AND null positions: Value::ToString() renders a NULL as "NULL", and no value in the
  // fixture stringifies to that, so a dropped mask (NULL read back as 0 or '') changes a cell.
  auto written   = query("SELECT * FROM nsrc ORDER BY i;");
  auto read_back = query("SELECT * FROM read_simpatico('" + file + "') ORDER BY i;");
  REQUIRE(ordered_rows(*read_back) == ordered_rows(*written));

  // Belt: count(col) counts non-nulls, so this fails on a mask that survives but is misaligned in
  // a way the row comparison above could conceivably tolerate.
  auto counts = query(
    "SELECT count(*), count(scattered), count(run), count(never_null),"
    " count(text), count(always_null) FROM read_simpatico('" +
    file + "');");
  REQUIRE(counts->GetValue(0, 0).GetValue<std::int64_t>() == 6000);
  REQUIRE(counts->GetValue(1, 0).GetValue<std::int64_t>() == 6000 - 60);
  REQUIRE(counts->GetValue(2, 0).GetValue<std::int64_t>() == 6000 - 1000);
  REQUIRE(counts->GetValue(3, 0).GetValue<std::int64_t>() == 6000);
  REQUIRE(counts->GetValue(4, 0).GetValue<std::int64_t>() == 6000 - 858);
  REQUIRE(counts->GetValue(5, 0).GetValue<std::int64_t>() == 0);
}

TEST_CASE_METHOD(SimpaticoCopyFixture,
                 "a .hpln reports which of its columns carry NULLs",
                 "[integration][simpatico_copy][hpln_io]")
{
  auto const file = path("nulls_schema.hpln");
  query("CREATE TABLE nsrc AS " + std::string(kNullableProjection) + ";");
  query("CHECKPOINT;");
  query("COPY (SELECT * FROM nsrc ORDER BY i) TO '" + file +
        "' (FORMAT simpatico, chunk_rows 2048);");

  auto const schema = sirius::read_hpln_schema(file);
  REQUIRE(schema.column_has_nulls.size() == schema.names.size());
  REQUIRE(schema.names ==
          std::vector<std::string>{"i", "scattered", "run", "never_null", "text", "always_null"});
  // `run` is null only in the middle chunk, so reporting chunk 0's answer for the file would call
  // it non-nullable -- which is the reading a caller would act on.
  REQUIRE(schema.column_has_nulls == std::vector<bool>{false, true, true, false, true, true});
}

TEST_CASE_METHOD(SimpaticoCopyFixture,
                 "a group holding a NULL is not pruned by a filter the NULL cannot satisfy",
                 "[integration][simpatico_copy][simpatico_pruning]")
{
  auto const file = path("nulls_pruning.hpln");
  // `v` climbs with i, so a range predicate leaves most groups provably empty -- except that one
  // group's values are NULL. Zone-map bounds are computed over the non-null values, so the bounds
  // of that group say nothing about the NULL, and pruning it would make the row vanish from a
  // count that must still see it as a non-match rather than not see it at all.
  query(
    "CREATE TABLE psrc AS SELECT i,"
    " CASE WHEN i BETWEEN 3000 AND 3063 THEN NULL ELSE i END::INTEGER AS v"
    " FROM range(6000) t(i);");
  query("COPY (SELECT * FROM psrc ORDER BY i) TO '" + file +
        "' (FORMAT simpatico, chunk_rows 2048, group_rows 64);");

  // The NULL group sits inside the filtered range. A pruned group would drop its rows entirely,
  // so `count(*)` over the range would come back short even though none of those rows match `v`.
  auto got = query("SELECT count(*), count(v), sum(v) FROM read_simpatico('" + file +
                   "') WHERE i >= 2900 AND i < 3200;");
  REQUIRE(got->GetValue(0, 0).GetValue<std::int64_t>() == 300);
  REQUIRE(got->GetValue(1, 0).GetValue<std::int64_t>() == 300 - 64);

  // And the same predicate on the nullable column itself: a NULL satisfies neither side, so the
  // two halves must still account for every non-null row.
  auto split = query("SELECT count(*) FROM read_simpatico('" + file + "') WHERE v < 3000;");
  auto rest  = query("SELECT count(*) FROM read_simpatico('" + file + "') WHERE v >= 3000;");
  REQUIRE(split->GetValue(0, 0).GetValue<std::int64_t>() +
            rest->GetValue(0, 0).GetValue<std::int64_t>() ==
          6000 - 64);
}

TEST_CASE_METHOD(SimpaticoCopyFixture,
                 "COPY to .hpln refuses a type it cannot carry",
                 "[integration][simpatico_copy]")
{
  auto const file = path("unsupported.hpln");
  // HUGEINT's cuDF carrier is INT64, so writing it would silently truncate.
  auto const hugeint_error =
    query_error("COPY (SELECT a::HUGEINT AS a FROM src) TO '" + file + "' (FORMAT simpatico);");
  REQUIRE(hugeint_error.find("HUGEINT") != std::string::npos);

  auto const list_error =
    query_error("COPY (SELECT [a] AS a FROM src) TO '" + file + "' (FORMAT simpatico);");
  REQUIRE(list_error.find("simpatico") != std::string::npos);
}

TEST_CASE_METHOD(SimpaticoCopyFixture,
                 "COPY to .hpln reports a plan that does not cover the query's columns",
                 "[integration][simpatico_copy]")
{
  auto const file  = path("bad_plan.hpln");
  auto const error = query_error("COPY (SELECT a, b FROM src) TO '" + file +
                                 "' (FORMAT simpatico, plan 'input -> identity');");
  REQUIRE(error.find("column blocks") != std::string::npos);
}

TEST_CASE_METHOD(SimpaticoCopyFixture,
                 "COPY to .hpln honours an explicit compression plan",
                 "[integration][simpatico_copy]")
{
  auto const file = path("bitpacked.hpln");
  auto const leaf = std::string("input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed");
  query("COPY (SELECT a, b FROM src ORDER BY a) TO '" + file + "' (FORMAT simpatico, plan '" +
        leaf + "\n---\n" + leaf + "', chunk_rows 2048);");

  auto written   = query("SELECT a, b FROM src ORDER BY a;");
  auto read_back = query("SELECT a, b FROM read_simpatico('" + file + "') ORDER BY a;");
  REQUIRE(ordered_rows(*read_back) == ordered_rows(*written));
}

TEST_CASE_METHOD(SimpaticoCopyFixture,
                 "COPY to .hpln round-trips a VARCHAR column",
                 "[integration][simpatico_copy]")
{
  auto const file = path("strings.hpln");
  query("COPY (SELECT a, s FROM src ORDER BY a) TO '" + file +
        "' (FORMAT simpatico, chunk_rows 2048);");

  auto written   = query("SELECT a, s FROM src ORDER BY a;");
  auto read_back = query("SELECT a, s FROM read_simpatico('" + file + "') ORDER BY a;");
  REQUIRE(ordered_rows(*read_back) == ordered_rows(*written));
}

TEST_CASE_METHOD(SimpaticoCopyFixture,
                 "COPY to .hpln writes a bindable file for a query with no rows",
                 "[integration][simpatico_copy]")
{
  auto const file = path("empty.hpln");
  query("COPY (SELECT a, b FROM src WHERE a < 0) TO '" + file + "' (FORMAT simpatico);");

  auto const schema = sirius::read_hpln_schema(file);
  REQUIRE(schema.num_rows == 0);
  REQUIRE(schema.names == std::vector<std::string>{"a", "b"});
  auto read_back = query("SELECT COUNT(*) FROM read_simpatico('" + file + "');");
  REQUIRE(read_back->GetValue(0, 0).GetValue<std::int64_t>() == 0);
}
