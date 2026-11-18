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
 * @file test_sirius_generate.cpp
 * @brief `sirius_generate` end to end: plain SQL through the transparent GPU path.
 *
 * The function has no CPU body (it throws), so any row a query returns was generated on the GPU.
 * `run_gpu` also asserts it from the transparent-execution counters: one GPU execution, no
 * fallback. Value semantics checked here are the ones documented in
 * `src/op/scan/generator/generator_spec.hpp`.
 */

#include "catch.hpp"
#include "op/scan/generator/generator_detail.cuh"

#include <duckdb.hpp>
#include <duckdb/main/appender.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/sirius_test_env.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

using rows_t = std::vector<std::vector<duckdb::Value>>;

/// `sirius_generate(...)` call text. @p columns is the body of the `columns := [...]` list.
std::string generate_call(std::uint64_t num_batches,
                          std::uint64_t rows_per_batch,
                          std::uint64_t seed,
                          std::string const& columns)
{
  return "sirius_generate(num_batches := " + std::to_string(num_batches) +
         ", rows_per_batch := " + std::to_string(rows_per_batch) +
         ", seed := " + std::to_string(seed) + ", columns := [" + columns + "])";
}

/// One sort key of an ORDER BY, by position in the result.
struct sort_key {
  std::size_t column;
  bool descending{false};
  bool nulls_first{false};
};

/// Three-way comparison of two non-null cells of the same column.
int compare_non_null(duckdb::Value const& a, duckdb::Value const& b)
{
  switch (a.type().id()) {
    case duckdb::LogicalTypeId::VARCHAR: {
      // DuckDB orders VARCHAR bytewise, as std::string does.
      auto const cmp = a.GetValue<std::string>().compare(b.GetValue<std::string>());
      return (cmp > 0) - (cmp < 0);
    }
    case duckdb::LogicalTypeId::DOUBLE: {
      auto const x = a.GetValue<double>();
      auto const y = b.GetValue<double>();
      return (x > y) - (x < y);
    }
    default: {
      auto const x = a.GetValue<std::int64_t>();
      auto const y = b.GetValue<std::int64_t>();
      return (x > y) - (x < y);
    }
  }
}

/// Three-way comparison of two rows under @p keys.
int compare_rows(std::vector<duckdb::Value> const& a,
                 std::vector<duckdb::Value> const& b,
                 std::vector<sort_key> const& keys)
{
  for (auto const& key : keys) {
    auto const& x = a[key.column];
    auto const& y = b[key.column];
    if (x.IsNull() || y.IsNull()) {
      if (x.IsNull() && y.IsNull()) { continue; }
      return (x.IsNull() == key.nulls_first) ? -1 : 1;
    }
    auto const cmp = compare_non_null(x, y);
    if (cmp != 0) { return key.descending ? -cmp : cmp; }
  }
  return 0;
}

/// Index of the first row that sorts before its predecessor, or `rows.size()` when ordered.
std::size_t first_out_of_order(rows_t const& rows, std::vector<sort_key> const& keys)
{
  for (std::size_t r = 1; r < rows.size(); ++r) {
    if (compare_rows(rows[r - 1], rows[r], keys) > 0) { return r; }
  }
  return rows.size();
}

class GenerateFixture : public sirius::test::GpuExecutionFixture {
 public:
  /// Runs @p query on the GPU and requires exactly one GPU execution without a fallback.
  duckdb::unique_ptr<duckdb::QueryResult> run_gpu(std::string const& query)
  {
    INFO(query);
    run_ok("SET gpu_execution = true;");
    auto const before = sirius::test::get_transparent_execution_stats(*con);
    auto result       = con->Query(query);
    auto const after  = sirius::test::get_transparent_execution_stats(*con);
    REQUIRE(result);
    if (result->HasError()) { UNSCOPED_INFO("GPU execution error: " << result->GetError()); }
    REQUIRE_FALSE(result->HasError());
    sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1);
    return result;
  }

  /// Runs @p query on DuckDB's CPU execution.
  duckdb::unique_ptr<duckdb::QueryResult> run_cpu(std::string const& query)
  {
    INFO(query);
    run_ok("SET gpu_execution = false;");
    auto result = con->Query(query);
    run_ok("SET gpu_execution = true;");
    REQUIRE(result);
    if (result->HasError()) { UNSCOPED_INFO("CPU execution error: " << result->GetError()); }
    REQUIRE_FALSE(result->HasError());
    return result;
  }

  /// The cells of @p result in emitted order.
  static rows_t values(duckdb::QueryResult& result)
  {
    rows_t rows;
    rows.reserve(result.RowCount());
    auto cells = result.Collection().GetRows();
    for (duckdb::idx_t r = 0; r < result.RowCount(); ++r) {
      std::vector<duckdb::Value> row;
      row.reserve(result.ColumnCount());
      for (duckdb::idx_t c = 0; c < result.ColumnCount(); ++c) {
        row.push_back(cells.GetValue(c, r));
      }
      rows.push_back(std::move(row));
    }
    return rows;
  }

  /// The single row of an aggregate query run on the GPU.
  std::vector<duckdb::Value> gpu_row(std::string const& query)
  {
    auto result = run_gpu(query);
    REQUIRE(result->RowCount() == 1);
    return values(*result).front();
  }

  /// Stringified rows of a GPU query, sorted (a multiset) or in emitted order.
  std::vector<std::vector<std::string>> gpu_rows(std::string const& query, bool sort = true)
  {
    auto result = run_gpu(query);
    return collect_rows(*result, sort);
  }
};

// INTEGER, BIGINT, DOUBLE and VARCHAR, with and without nulls.
const std::string kMixedColumns =
  "{name: 'k', type: 'INTEGER', dist: 'uniform', p1: 0, p2: 1000, null_fraction: 0.1}, "
  "{name: 'b', type: 'BIGINT', dist: 'sequential', p1: 0, p2: 1}, "
  "{name: 'd', type: 'DOUBLE', dist: 'normal', p1: 10, p2: 2, null_fraction: 0.5}, "
  "{name: 's', type: 'VARCHAR', dist: 'uniform', p1: 8, p2: 16, null_fraction: 0.25}";

// Sort keys with many ties and nulls in every key column. For VARCHAR, p1 and p2 are both the
// key range and the length range: `s` is one of seven 3-digit keys plus up to six random letters,
// so it has ties (the bare keys) as well as long runs sharing a prefix.
const std::string kSortColumns =
  "{name: 'k1', type: 'BIGINT', dist: 'few_distinct', p1: 40, null_fraction: 0.1}, "
  "{name: 'k2', type: 'INTEGER', dist: 'uniform', p1: -100, p2: 100, null_fraction: 0.05}, "
  "{name: 's', type: 'VARCHAR', dist: 'uniform', p1: 3, p2: 9, null_fraction: 0.1}";

// Columns whose aggregates have a host reference: `g` is the global row and `f` is
// `counter_hash(..., VALUE_A) % 1000`, so their sums do not depend on which task, stream or GPU
// produced a batch.
const std::string kSummedColumns =
  "{name: 'g', type: 'BIGINT', dist: 'sequential', p1: 0, p2: 1}, "
  "{name: 'f', type: 'BIGINT', dist: 'few_distinct', p1: 1000}, "
  "{name: 's', type: 'VARCHAR', dist: 'uniform', p1: 4, p2: 8, null_fraction: 0.1}";

constexpr std::uint64_t kSummedDistinct = 1000;

std::string summed_query(std::uint64_t num_batches,
                         std::uint64_t rows_per_batch,
                         std::uint64_t seed)
{
  return "SELECT count(*), sum(g), sum(f), count(DISTINCT g) FROM " +
         generate_call(num_batches, rows_per_batch, seed, kSummedColumns);
}

/// What `summed_query` must return, computed on the host from the generator's hash.
std::vector<std::int64_t> summed_reference(std::uint64_t num_batches,
                                           std::uint64_t rows_per_batch,
                                           std::uint64_t seed)
{
  namespace detail = sirius::op::scan::generator::detail;
  auto const rows  = num_batches * rows_per_batch;
  std::uint64_t f  = 0;
  for (std::uint64_t batch = 0; batch < num_batches; ++batch) {
    for (std::uint64_t row = 0; row < rows_per_batch; ++row) {
      f += detail::counter_hash(seed, batch, row, 1, detail::VALUE_A) % kSummedDistinct;
    }
  }
  return {static_cast<std::int64_t>(rows),
          static_cast<std::int64_t>(rows * (rows - 1) / 2),
          static_cast<std::int64_t>(f),
          static_cast<std::int64_t>(rows)};
}

/// The single row of `summed_query` run on @p con, or @p error set. The function has no CPU
/// body, so a result means the GPU produced it.
std::vector<std::int64_t> summed_row(duckdb::Connection& con,
                                     std::string const& query,
                                     std::string& error)
{
  auto result = con.Query(query);
  if (!result) {
    error = "no result";
    return {};
  }
  if (result->HasError()) {
    error = result->GetError();
    return {};
  }
  if (result->RowCount() != 1) {
    error = "expected one row, got " + std::to_string(result->RowCount());
    return {};
  }
  std::vector<std::int64_t> row;
  auto cells = result->Collection().GetRows();
  for (duckdb::idx_t c = 0; c < result->ColumnCount(); ++c) {
    row.push_back(cells.GetValue(c, 0).GetValue<std::int64_t>());
  }
  return row;
}

/// `summed_row`, requiring success.
std::vector<std::int64_t> summed_row(duckdb::Connection& con, std::string const& query)
{
  std::string error;
  auto row = summed_row(con, query, error);
  INFO(error);
  REQUIRE(error.empty());
  return row;
}

constexpr std::uint64_t kSortBatches      = 8;
constexpr std::uint64_t kSortRowsPerBatch = 2500;
constexpr std::uint64_t kSortRows         = kSortBatches * kSortRowsPerBatch;

std::string sort_source()
{
  return generate_call(kSortBatches, kSortRowsPerBatch, 11, kSortColumns);
}

}  // namespace

//===----------------------------------------------------------------------===//
// Schema and row counts
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(GenerateFixture,
                 "sirius_generate returns num_batches x rows_per_batch rows of the declared schema",
                 "[integration][gpu_execution][sirius_generate]")
{
  auto const source = generate_call(4, 10000, 42, kMixedColumns);

  // DESCRIBE binds without executing.
  auto described = con->Query("DESCRIBE SELECT * FROM " + source);
  REQUIRE(described);
  REQUIRE_FALSE(described->HasError());
  auto const schema = collect_rows(*described, false);
  REQUIRE(schema.size() == 4);
  const std::vector<std::pair<std::string, std::string>> expected = {
    {"k", "INTEGER"}, {"b", "BIGINT"}, {"d", "DOUBLE"}, {"s", "VARCHAR"}};
  for (std::size_t c = 0; c < expected.size(); ++c) {
    CHECK(schema[c][0] == expected[c].first);
    CHECK(schema[c][1] == expected[c].second);
  }

  auto const counts =
    gpu_row("SELECT count(*), count(k), count(b), count(d), count(s) FROM " + source);
  CHECK(counts[0].GetValue<std::int64_t>() == 40000);
  CHECK(counts[2].GetValue<std::int64_t>() == 40000);
  // Null counts are binomial(40000, null_fraction); the bounds are more than six sigma wide.
  CHECK(std::abs(counts[1].GetValue<std::int64_t>() - 36000) <= 400);
  CHECK(std::abs(counts[3].GetValue<std::int64_t>() - 20000) <= 650);
  CHECK(std::abs(counts[4].GetValue<std::int64_t>() - 30000) <= 550);

  // The materialized result carries the same types and names, and the row count.
  auto rows = run_gpu("SELECT * FROM " + source);
  CHECK(rows->RowCount() == 40000);
  CHECK(rows->GetTypes() == duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER,
                                                                duckdb::LogicalType::BIGINT,
                                                                duckdb::LogicalType::DOUBLE,
                                                                duckdb::LogicalType::VARCHAR});
  REQUIRE(rows->GetNames().size() == 4);
  CHECK(rows->ColumnName(0).GetIdentifierName() == "k");
  CHECK(rows->ColumnName(3).GetIdentifierName() == "s");

  // VARCHAR lengths stay inside [p1, p2].
  auto const lengths = gpu_row("SELECT min(length(s)), max(length(s)) FROM " + source);
  CHECK(lengths[0].GetValue<std::int64_t>() >= 8);
  CHECK(lengths[1].GetValue<std::int64_t>() <= 16);
}

TEST_CASE_METHOD(GenerateFixture,
                 "sirius_generate produces every batch of a multi-million-row spec",
                 "[integration][gpu_execution][sirius_generate]")
{
  constexpr std::int64_t rows = 8 * 1000000;
  auto const source           = generate_call(
    8,
    1000000,
    3,
    "{name: 'g', type: 'BIGINT', dist: 'sequential', p1: 0, p2: 1}, "
              "{name: 'u', type: 'INTEGER', dist: 'uniform', p1: 1, p2: 100, null_fraction: 0.01}");
  auto const row =
    gpu_row("SELECT count(*), min(g), max(g), count(DISTINCT g), min(u), max(u) FROM " + source);
  CHECK(row[0].GetValue<std::int64_t>() == rows);
  CHECK(row[1].GetValue<std::int64_t>() == 0);
  CHECK(row[2].GetValue<std::int64_t>() == rows - 1);
  // Every global row exactly once: no batch is skipped or produced twice.
  CHECK(row[3].GetValue<std::int64_t>() == rows);
  CHECK(row[4].GetValue<std::int64_t>() == 1);
  CHECK(row[5].GetValue<std::int64_t>() == 100);
}

//===----------------------------------------------------------------------===//
// Concurrency
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(GenerateFixture,
                 "sirius_generate queries on concurrent connections each produce every batch",
                 "[integration][gpu_execution][sirius_generate]")
{
  constexpr std::uint64_t num_batches    = 64;
  constexpr std::uint64_t rows_per_batch = 10000;
  constexpr std::size_t num_connections  = 4;
  auto const query                       = summed_query(num_batches, rows_per_batch, 7);
  auto const expected                    = summed_reference(num_batches, rows_per_batch, 7);

  CHECK(summed_row(*con, query) == expected);

  // Each query plans its own GENERATOR and so its own batch counter: no batch may be lost or
  // produced twice when they run at the same time.
  std::vector<std::unique_ptr<duckdb::Connection>> connections;
  for (std::size_t i = 0; i < num_connections; ++i) {
    connections.push_back(
      sirius::test::g_integration_env && sirius::test::g_integration_env->is_active()
        ? std::make_unique<duckdb::Connection>(sirius::test::g_integration_env->make_connection())
        : std::make_unique<duckdb::Connection>(*db));
  }
  std::vector<std::vector<std::int64_t>> rows(num_connections);
  std::vector<std::string> errors(num_connections);
  {
    std::vector<std::thread> threads;
    for (std::size_t i = 0; i < num_connections; ++i) {
      // An exception that escapes a thread would terminate the whole test binary.
      threads.emplace_back([&, i] {
        try {
          rows[i] = summed_row(*connections[i], query, errors[i]);
        } catch (std::exception const& e) {
          errors[i] = e.what();
        } catch (...) {
          errors[i] = "unknown exception";
        }
      });
    }
    for (auto& thread : threads) {
      thread.join();
    }
  }
  for (std::size_t i = 0; i < num_connections; ++i) {
    CAPTURE(i);
    CAPTURE(errors[i]);
    CHECK(errors[i].empty());
    CHECK(rows[i] == expected);
  }
}

TEST_CASE("sirius_generate produces every batch once when tasks spread over two GPUs",
          "[integration][gpu_execution][sirius_generate][multi_gpu]")
{
  if (!sirius::test::has_gpus(2)) { return; }
  // The `[integration][multi_gpu]` tags make the `shared_env_listener` in `unittest.cpp` activate
  // this environment; resuming it again here would replace its live SiriusContext.
  auto* env = sirius::test::acquire_integration_env_for(2);
  REQUIRE(env != nullptr);
  REQUIRE(env->is_active());

  constexpr std::uint64_t num_batches    = 256;
  constexpr std::uint64_t rows_per_batch = 4000;
  auto con = std::make_unique<duckdb::Connection>(env->make_connection());
  // The reference is the same one the one-GPU test above checks against: a batch does not
  // depend on the device that generated it.
  CHECK(summed_row(*con, summed_query(num_batches, rows_per_batch, 7)) ==
        summed_reference(num_batches, rows_per_batch, 7));
}

//===----------------------------------------------------------------------===//
// ORDER BY
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(GenerateFixture,
                 "ORDER BY over sirius_generate returns the generated rows in order",
                 "[integration][gpu_execution][sirius_generate]")
{
  struct order_case {
    const char* order_by;
    std::vector<sort_key> keys;
  };
  // Result columns are k1, k2, s. DuckDB's default is NULLS LAST, ascending or descending.
  auto const tested = GENERATE(order_case{"k1", {{0}}},
                               order_case{"s", {{2}}},
                               order_case{"k2 DESC", {{1, true}}},
                               order_case{"k1, k2, s", {{0}, {1}, {2}}},
                               order_case{"k1 DESC NULLS FIRST, s", {{0, true, true}, {2}}},
                               order_case{"s DESC NULLS LAST, k1 NULLS FIRST, k2 DESC",
                                          {{2, true, false}, {0, false, true}, {1, true, false}}});
  CAPTURE(tested.order_by);

  auto const unsorted = gpu_rows("SELECT k1, k2, s FROM " + sort_source());
  REQUIRE(unsorted.size() == kSortRows);

  auto sorted = run_gpu("SELECT k1, k2, s FROM " + sort_source() + " ORDER BY " + tested.order_by);
  REQUIRE(sorted->RowCount() == kSortRows);

  // Adjacent rows are in order ...
  auto const rows   = values(*sorted);
  auto const broken = first_out_of_order(rows, tested.keys);
  if (broken != rows.size()) {
    UNSCOPED_INFO("row " << broken << " sorts before row " << broken - 1 << ": ["
                         << rows[broken][0].ToString() << ", " << rows[broken][1].ToString() << ", "
                         << rows[broken][2].ToString() << "] after ["
                         << rows[broken - 1][0].ToString() << ", " << rows[broken - 1][1].ToString()
                         << ", " << rows[broken - 1][2].ToString() << "]");
  }
  CHECK(broken == rows.size());
  // ... nulls are present in the leading key, so their placement was exercised ...
  auto const lead = tested.keys.front().column;
  CHECK(std::any_of(rows.begin(), rows.end(), [&](auto const& row) { return row[lead].IsNull(); }));
  // ... and sorting neither dropped, duplicated nor changed a row.
  CHECK(collect_rows(*sorted, true) == unsorted);
}

TEST_CASE_METHOD(GenerateFixture,
                 "ORDER BY over sirius_generate matches DuckDB sorting the same rows",
                 "[integration][gpu_execution][sirius_generate]")
{
  // Generate the rows on the GPU, store them in a DuckDB table, then let DuckDB sort the table.
  // (CREATE TABLE AS is not a SELECT statement, so it never reaches the GPU path and would run
  // the function's throwing CPU body.)
  {
    auto generated = run_gpu("SELECT * FROM " + sort_source());
    REQUIRE(generated->RowCount() == kSortRows);
    run_ok("CREATE TABLE generated (k1 BIGINT, k2 INTEGER, s VARCHAR);");
    duckdb::Appender appender(*con, duckdb::Identifier(attach_alias), "main", "generated");
    for (auto const& row : values(*generated)) {
      appender.BeginRow();
      for (auto const& cell : row) {
        appender.Append(cell);
      }
      appender.EndRow();
    }
    appender.Close();
  }
  auto const stored = run_cpu("SELECT count(*), count(k1), count(k2), count(s) FROM generated");
  auto const direct =
    run_gpu("SELECT count(*), count(k1), count(k2), count(s) FROM " + sort_source());
  REQUIRE(collect_rows(*stored, false) == collect_rows(*direct, false));

  // Every ORDER BY names all three columns, so tied rows are identical and the order is total.
  auto const order_by = GENERATE(as<std::string>{},
                                 "k1, k2, s",
                                 "s, k1, k2",
                                 "k2 DESC, s DESC, k1 DESC",
                                 "k1 DESC NULLS LAST, s NULLS FIRST, k2");
  CAPTURE(order_by);
  auto cpu_sorted = run_cpu("SELECT k1, k2, s FROM generated ORDER BY " + order_by);
  auto gpu_sorted = run_gpu("SELECT k1, k2, s FROM " + sort_source() + " ORDER BY " + order_by);
  REQUIRE(cpu_sorted->RowCount() == kSortRows);
  REQUIRE(gpu_sorted->GetTypes() == cpu_sorted->GetTypes());
  auto const cpu_rows = collect_rows(*cpu_sorted, false);
  auto const gpu_rows = collect_rows(*gpu_sorted, false);
  REQUIRE(gpu_rows.size() == cpu_rows.size());
  for (std::size_t r = 0; r < cpu_rows.size(); ++r) {
    if (gpu_rows[r] != cpu_rows[r]) {
      FAIL("row " << r << " differs: GPU [" << gpu_rows[r][0] << ", " << gpu_rows[r][1] << ", "
                  << gpu_rows[r][2] << "] CPU [" << cpu_rows[r][0] << ", " << cpu_rows[r][1] << ", "
                  << cpu_rows[r][2] << "]");
    }
  }
}
