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

#include <duckdb.hpp>
#include <duckdb/main/appender.hpp>
#include <utils/gpu_execution_fixture.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
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

// `g` is the global row number, so it tells which batch a row came from.
const std::string kKeyedColumns =
  "{name: 'g', type: 'BIGINT', dist: 'sequential', p1: 0, p2: 1}, "
  "{name: 'k1', type: 'BIGINT', dist: 'uniform', p1: -1000000, p2: 1000000, null_fraction: 0.1}, "
  "{name: 'x', type: 'DOUBLE', dist: 'uniform', p1: 0, p2: 1}, "
  "{name: 's', type: 'VARCHAR', dist: 'zipf', p1: 6, p2: 12, cardinality: 500, "
  "null_fraction: 0.1}";

// Sort keys with many ties and nulls in every key column. For VARCHAR, p1 and p2 are both the
// key range and the length range: `s` is one of seven 3-digit keys plus up to six random letters,
// so it has ties (the bare keys) as well as long runs sharing a prefix.
const std::string kSortColumns =
  "{name: 'k1', type: 'BIGINT', dist: 'few_distinct', p1: 40, null_fraction: 0.1}, "
  "{name: 'k2', type: 'INTEGER', dist: 'uniform', p1: -100, p2: 100, null_fraction: 0.05}, "
  "{name: 's', type: 'VARCHAR', dist: 'uniform', p1: 3, p2: 9, null_fraction: 0.1}";

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
// Reproducibility
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(GenerateFixture,
                 "sirius_generate is reproducible per seed and per batch",
                 "[integration][gpu_execution][sirius_generate]")
{
  constexpr std::uint64_t rows_per_batch = 1000;
  auto const four = gpu_rows("SELECT * FROM " + generate_call(4, rows_per_batch, 7, kKeyedColumns));
  REQUIRE(four.size() == 4 * rows_per_batch);

  SECTION("the same query returns the same rows")
  {
    auto const again =
      gpu_rows("SELECT * FROM " + generate_call(4, rows_per_batch, 7, kKeyedColumns));
    CHECK(again == four);
  }

  SECTION("a different seed returns different rows")
  {
    auto const reseeded =
      gpu_rows("SELECT * FROM " + generate_call(4, rows_per_batch, 8, kKeyedColumns));
    REQUIRE(reseeded.size() == four.size());
    CHECK(reseeded != four);
    // Not merely a few rows: the sequential column matches, the drawn columns do not.
    std::size_t same_k1 = 0;
    for (std::size_t r = 0; r < four.size(); ++r) {
      REQUIRE(reseeded[r][0] == four[r][0]);
      same_k1 += reseeded[r][1] == four[r][1];
    }
    // Both draws are null with probability 0.1, so about 1% of the rows agree by chance.
    CHECK(same_k1 < four.size() / 10);
  }

  SECTION("a batch does not depend on how many batches there are")
  {
    auto const two =
      gpu_rows("SELECT * FROM " + generate_call(2, rows_per_batch, 7, kKeyedColumns));
    REQUIRE(two.size() == 2 * rows_per_batch);
    // The four-batch multiset contains the two-batch one ...
    CHECK(std::includes(four.begin(), four.end(), two.begin(), two.end()));
    // ... and it is exactly the first two batches: the rows whose global row is below 2000.
    std::vector<std::vector<std::string>> first_two;
    for (auto const& row : four) {
      if (std::stoull(row[0]) < 2 * rows_per_batch) { first_two.push_back(row); }
    }
    CHECK(first_two == two);
  }

  SECTION("a batch does not depend on the query around it")
  {
    // The same rows through a projection and a filter above the source.
    auto const filtered =
      gpu_rows("SELECT g, k1, x, s FROM " + generate_call(4, rows_per_batch, 7, kKeyedColumns) +
               " WHERE g >= 3000");
    std::vector<std::vector<std::string>> last_batch;
    for (auto const& row : four) {
      if (std::stoull(row[0]) >= 3 * rows_per_batch) { last_batch.push_back(row); }
    }
    REQUIRE(last_batch.size() == rows_per_batch);
    CHECK(filtered == last_batch);
  }
}

//===----------------------------------------------------------------------===//
// Distribution statistics
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(GenerateFixture,
                 "sirius_generate uniform columns stay in range with the expected mean",
                 "[integration][gpu_execution][sirius_generate]")
{
  auto const source =
    generate_call(4,
                  50000,
                  5,
                  "{name: 'i', type: 'INTEGER', dist: 'uniform', p1: -50, p2: 50}, "
                  "{name: 'b', type: 'BIGINT', dist: 'uniform', p1: 100, p2: 199}, "
                  "{name: 'd', type: 'DOUBLE', dist: 'uniform', p1: 2, p2: 4}");
  auto const row = gpu_row(
    "SELECT min(i), max(i), avg(i), min(b), max(b), avg(b), min(d), max(d), avg(d), "
    "count(DISTINCT b) FROM " +
    source);
  // 200,000 draws over at most 101 values reach both inclusive bounds.
  CHECK(row[0].GetValue<std::int64_t>() == -50);
  CHECK(row[1].GetValue<std::int64_t>() == 50);
  CHECK(std::abs(row[2].GetValue<double>() - 0.0) < 0.5);
  CHECK(row[3].GetValue<std::int64_t>() == 100);
  CHECK(row[4].GetValue<std::int64_t>() == 199);
  CHECK(std::abs(row[5].GetValue<double>() - 149.5) < 0.5);
  CHECK(row[9].GetValue<std::int64_t>() == 100);
  // DOUBLE is uniform in [lo, hi).
  CHECK(row[6].GetValue<double>() >= 2.0);
  CHECK(row[6].GetValue<double>() < 2.01);
  CHECK(row[7].GetValue<double>() < 4.0);
  CHECK(row[7].GetValue<double>() > 3.99);
  CHECK(std::abs(row[8].GetValue<double>() - 3.0) < 0.02);
}

TEST_CASE_METHOD(GenerateFixture,
                 "sirius_generate normal columns have the requested mean and spread",
                 "[integration][gpu_execution][sirius_generate]")
{
  auto const source = generate_call(4,
                                    50000,
                                    6,
                                    "{name: 'd', type: 'DOUBLE', dist: 'normal', p1: 100, p2: 15}, "
                                    "{name: 'b', type: 'BIGINT', dist: 'normal', p1: -20, p2: 5}");
  // avg(d * d) keeps the check to aggregates every engine supports.
  auto const row    = gpu_row("SELECT avg(d), avg(d * d), avg(b), min(b), max(b) FROM " + source);
  auto const mean_d = row[0].GetValue<double>();
  CHECK(std::abs(mean_d - 100.0) < 0.3);
  auto const stddev_d = std::sqrt(row[1].GetValue<double>() - mean_d * mean_d);
  CHECK(std::abs(stddev_d - 15.0) < 0.3);
  CHECK(std::abs(row[2].GetValue<double>() - (-20.0)) < 0.1);
  // Eight sigma either side: far outside what 200,000 draws reach, but both tails are present.
  CHECK(row[3].GetValue<std::int64_t>() > -60);
  CHECK(row[3].GetValue<std::int64_t>() < -30);
  CHECK(row[4].GetValue<std::int64_t>() < 20);
  CHECK(row[4].GetValue<std::int64_t>() > -10);
}

TEST_CASE_METHOD(GenerateFixture,
                 "sirius_generate few_distinct columns have at most D distinct values",
                 "[integration][gpu_execution][sirius_generate]")
{
  auto const source =
    generate_call(4,
                  25000,
                  9,
                  "{name: 'f', type: 'BIGINT', dist: 'few_distinct', p1: 7}, "
                  "{name: 'i', type: 'INTEGER', dist: 'few_distinct', p1: 1}, "
                  "{name: 's', type: 'VARCHAR', dist: 'few_distinct', p1: 13, p2: 13}");
  auto const row = gpu_row(
    "SELECT count(DISTINCT f), min(f), max(f), count(DISTINCT i), min(i), count(DISTINCT s) FROM " +
    source);
  // hash % D: 100,000 draws hit every one of the D values.
  CHECK(row[0].GetValue<std::int64_t>() == 7);
  CHECK(row[1].GetValue<std::int64_t>() == 0);
  CHECK(row[2].GetValue<std::int64_t>() == 6);
  CHECK(row[3].GetValue<std::int64_t>() == 1);
  CHECK(row[4].GetValue<std::int64_t>() == 0);
  // Fixed length 13 and at most two key digits: the string is the zero-padded key alone.
  CHECK(row[5].GetValue<std::int64_t>() == 13);

  // Each value gets about a seventh of the rows.
  auto groups = run_gpu("SELECT f, count(*) FROM " + source + " GROUP BY f");
  REQUIRE(groups->RowCount() == 7);
  for (auto const& group : values(*groups)) {
    CAPTURE(group[0].ToString());
    CHECK(std::abs(group[1].GetValue<std::int64_t>() - 100000 / 7) < 600);
  }
}

TEST_CASE_METHOD(GenerateFixture,
                 "sirius_generate zipf columns rank 1 first with the expected share",
                 "[integration][gpu_execution][sirius_generate]")
{
  constexpr std::int64_t cardinality = 1000;
  constexpr double exponent          = 1.1;
  constexpr std::int64_t rows        = 200000;
  auto const source                  = generate_call(
    4, 50000, 13, "{name: 'z', type: 'BIGINT', dist: 'zipf', p1: 1.1, cardinality: 1000}");

  auto groups = run_gpu("SELECT z, count(*) FROM " + source + " GROUP BY z");
  auto counts = values(*groups);
  REQUIRE_FALSE(counts.empty());
  CHECK(counts.size() <= static_cast<std::size_t>(cardinality));
  std::sort(counts.begin(),
            counts.end(),
            [](std::vector<duckdb::Value> const& a, std::vector<duckdb::Value> const& b) {
              return a[1].GetValue<std::int64_t>() > b[1].GetValue<std::int64_t>();
            });

  double harmonic = 0;
  for (std::int64_t rank = 1; rank <= cardinality; ++rank) {
    harmonic += std::pow(static_cast<double>(rank), -exponent);
  }
  // P(rank r) = r^-s / H(N, s): about 18% for rank 1 here, a standard deviation of about 170 rows.
  auto expected_count = [&](std::int64_t rank) {
    return static_cast<double>(rows) * std::pow(static_cast<double>(rank), -exponent) / harmonic;
  };
  for (std::int64_t rank = 1; rank <= 3; ++rank) {
    auto const& group = counts[static_cast<std::size_t>(rank - 1)];
    CAPTURE(rank, group[0].ToString(), group[1].ToString());
    CHECK(group[0].GetValue<std::int64_t>() == rank);
    auto const expected = expected_count(rank);
    CHECK(std::abs(static_cast<double>(group[1].GetValue<std::int64_t>()) - expected) <
          0.05 * expected);
  }

  auto const bounds = gpu_row("SELECT min(z), max(z), count(*) FROM " + source);
  CHECK(bounds[0].GetValue<std::int64_t>() == 1);
  CHECK(bounds[1].GetValue<std::int64_t>() <= cardinality);
  CHECK(bounds[2].GetValue<std::int64_t>() == rows);
}

TEST_CASE_METHOD(GenerateFixture,
                 "sirius_generate sequential columns count from p1 in steps of p2",
                 "[integration][gpu_execution][sirius_generate]")
{
  constexpr std::int64_t rows = 5 * 20000;
  auto const source =
    generate_call(5,
                  20000,
                  1,
                  "{name: 'up', type: 'BIGINT', dist: 'sequential', p1: 5, p2: 3}, "
                  "{name: 'down', type: 'BIGINT', dist: 'sequential_desc', p1: 1000, p2: 2}, "
                  "{name: 'unit', type: 'INTEGER', dist: 'sequential', p1: -7, p2: 0}, "
                  "{name: 'd', type: 'DOUBLE', dist: 'sequential', p1: 0.5, p2: 0.25}");
  auto const row = gpu_row(
    "SELECT min(up), max(up), count(DISTINCT up), min(down), max(down), min(unit), max(unit), "
    "min(d), max(d) FROM " +
    source);
  CHECK(row[0].GetValue<std::int64_t>() == 5);
  CHECK(row[1].GetValue<std::int64_t>() == 5 + 3 * (rows - 1));
  CHECK(row[2].GetValue<std::int64_t>() == rows);
  CHECK(row[3].GetValue<std::int64_t>() == 1000 - 2 * (rows - 1));
  CHECK(row[4].GetValue<std::int64_t>() == 1000);
  // A step of 0 means 1.
  CHECK(row[5].GetValue<std::int64_t>() == -7);
  CHECK(row[6].GetValue<std::int64_t>() == -7 + (rows - 1));
  CHECK(row[7].GetValue<double>() == 0.5);
  CHECK(row[8].GetValue<double>() == 0.5 + 0.25 * static_cast<double>(rows - 1));

  // The columns move together: every row is the same global row in each of them.
  auto const aligned =
    gpu_row("SELECT count(*) FROM " + source +
            " WHERE down = 1000 - 2 * ((up - 5) / 3) AND unit = (up - 5) / 3 - 7");
  CHECK(aligned[0].GetValue<std::int64_t>() == rows);
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

//===----------------------------------------------------------------------===//
// GPU execution only
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(GenerateFixture,
                 "sirius_generate plans to a GENERATOR source on the GPU",
                 "[integration][gpu_execution][sirius_generate]")
{
  auto const source = generate_call(2, 1000, 42, kMixedColumns);
  // One GPU execution, no plan-time and no runtime fallback (asserted inside run_gpu), for the
  // bare source and under a sort.
  CHECK(run_gpu("SELECT * FROM " + source)->RowCount() == 2000);
  CHECK(run_gpu("SELECT k, s FROM " + source + " ORDER BY k, s")->RowCount() == 2000);

  // With the CPU fallback disabled the query still succeeds: nothing was replayed on DuckDB.
  run_ok("SET enable_duckdb_fallback = false;");
  CHECK(run_gpu("SELECT b FROM " + source + " ORDER BY b DESC")->RowCount() == 2000);
  run_ok("SET enable_duckdb_fallback = true;");
}

TEST_CASE_METHOD(GenerateFixture,
                 "sirius_generate throws without GPU execution",
                 "[integration][gpu_execution][sirius_generate]")
{
  auto const source = generate_call(2, 1000, 42, kMixedColumns);
  run_ok("SET gpu_execution = false;");
  auto const before = sirius::test::get_transparent_execution_stats(*con);
  for (auto const* query : {"SELECT * FROM ", "SELECT count(*) FROM ", "SELECT k FROM "}) {
    CAPTURE(query);
    auto result = con->Query(query + source);
    REQUIRE(result);
    REQUIRE(result->HasError());
    CHECK_THAT(result->GetError(),
               Catch::Matchers::ContainsSubstring("requires Sirius GPU execution"));
  }
  auto const after = sirius::test::get_transparent_execution_stats(*con);
  CHECK(after.executions == before.executions);
  run_ok("SET gpu_execution = true;");

  // The same connection generates again once GPU execution is back on.
  CHECK(run_gpu("SELECT * FROM " + source)->RowCount() == 2000);
}
