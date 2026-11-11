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

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/scoped_sirius_setting.hpp>

#include <algorithm>
#include <optional>
#include <string>
#include <vector>

namespace {

using rows = std::vector<std::vector<std::string>>;

// Sibling correlated EXISTS / NOT EXISTS with an equality and an inequality over the same table,
// which Sirius merges into one scan and a MIN/MAX aggregate per key. The generated rows carry
// NULL values and many rows per key but no answers; the hand-written keys below each give known
// answers, including NULL keys that only null-safe key comparisons match.
class ExistsMinmaxRewriteFixture : public sirius::test::GpuExecutionFixture {
 public:
  ExistsMinmaxRewriteFixture()
  {
    enable_guard.emplace(*con, "enable_exists_minmax_rewrite", true);
    run_ok(
      "CREATE TABLE fact (k BIGINT, v INTEGER, w INTEGER, d DATE, amount DECIMAL(12, 2), "
      "late BOOLEAN);");
    run_ok(
      "INSERT INTO fact SELECT (range * 7) % 61, (range * 13) % 1000 - 500, "
      "CASE WHEN range % 4 = 0 THEN NULL ELSE (range * 31) % 97 END, "
      "DATE '2020-01-01' + ((range * 11) % 30)::INTEGER, "
      "((range * 17) % 10000)::DECIMAL(12, 2) / 100, range % 3 = 0 FROM range(4000);");
    // Keys with exactly one row: EXISTS another row is false, NOT EXISTS true.
    run_ok(
      "INSERT INTO fact VALUES (1000, 1, 1, DATE '2020-01-01', 1.00, true), "
      "(1001, 2, NULL, DATE '2020-01-02', 2.00, true);");
    // Q21 shape: 2000's late row has a different-`v` partner that is not late, so it is an
    // answer; 2001's late rows each have a late partner. Equal w, d and amount per key keep them
    // out of the value queries.
    run_ok(
      "INSERT INTO fact VALUES (2000, 1, 1, DATE '2020-06-01', 1.00, true), "
      "(2000, 2, 1, DATE '2020-06-01', 1.00, false), "
      "(2001, 1, 1, DATE '2020-06-02', 1.00, true), (2001, 2, 1, DATE '2020-06-02', 1.00, true);");
    // Value queries: 3000's rows both have a different partner and none with w > 40, so both are
    // answers; 3001's w=10 row has a w=50 partner, so only the w=50 row is. The NULL keys behave
    // like 3001 when both key comparisons are null-safe; the w=50 row excludes the other two.
    run_ok(
      "INSERT INTO fact VALUES (3000, 1, 10, DATE '2021-01-01', 1.00, false), "
      "(3000, 2, 20, DATE '2021-01-02', 2.00, false), "
      "(3001, 1, 10, DATE '2021-01-03', 3.00, false), "
      "(3001, 2, 50, DATE '2021-01-04', 4.00, false), "
      "(NULL, 1, 10, DATE '2021-02-01', 10.00, false), "
      "(NULL, 2, 20, DATE '2021-02-02', 20.00, false), "
      "(NULL, 3, 50, DATE '2021-02-03', 30.00, false);");
    run_ok("CHECKPOINT;");
  }

  // The CPU answer is @p expected; the rewritten GPU plan, which normal SQL must select, and the
  // unrewritten one both match it, under both of DuckDB's representations of the delim join.
  // With `delim_join_as_cte`, DuckDB keeps some shapes as MARK joins over a CTE, which the rewrite
  // leaves alone; @p rewritten_as_cte requires the rewrite there too.
  void compare_rewritten_and_not(std::string const& query,
                                 rows expected,
                                 bool rewritten_as_cte = false)
  {
    INFO(query);
    run_ok("SET gpu_execution = false;");
    auto cpu = con->Query(query);
    run_ok("SET gpu_execution = true;");
    REQUIRE_FALSE(cpu->HasError());
    std::sort(expected.begin(), expected.end());
    REQUIRE(collect_rows(*cpu) == expected);
    for (bool const as_cte : {false, true}) {
      CAPTURE(as_cte);
      sirius::test::scoped_sirius_setting representation{*con, "delim_join_as_cte", as_cte};
      auto before = sirius::test::get_transparent_execution_stats(*con).exists_minmax_rewrites;
      compare_gpu_vs_cpu(query);
      auto after = sirius::test::get_transparent_execution_stats(*con).exists_minmax_rewrites;
      if (!as_cte || rewritten_as_cte) { CHECK(after > before); }
      sirius::test::scoped_sirius_setting off{*con, "enable_exists_minmax_rewrite", false};
      compare_gpu_vs_cpu(query);
      CHECK(sirius::test::get_transparent_execution_stats(*con).exists_minmax_rewrites == after);
    }
  }

  std::optional<sirius::test::scoped_sirius_setting> enable_guard;
};

}  // namespace

TEST_CASE_METHOD(ExistsMinmaxRewriteFixture,
                 "gpu_execution EXISTS and NOT EXISTS together, in the shape of TPC-H q21",
                 "[integration][gpu_execution][exists_rewrite]")
{
  compare_rewritten_and_not(
    "SELECT f.k, count(*) AS n FROM fact f WHERE f.late "
    "AND EXISTS (SELECT 1 FROM fact g WHERE g.k = f.k AND g.v <> f.v) "
    "AND NOT EXISTS (SELECT 1 FROM fact g WHERE g.k = f.k AND g.v <> f.v AND g.late) "
    "GROUP BY f.k",
    {{"2000", "1"}},
    true);
}

TEST_CASE_METHOD(ExistsMinmaxRewriteFixture,
                 "gpu_execution sibling subqueries over nullable, temporal and decimal values",
                 "[integration][gpu_execution][exists_rewrite]")
{
  // The hand-written rows' values of each column, in insertion order.
  struct column_values {
    char const* name;
    std::vector<std::string> k3000, k3001, null_key;
  };
  std::vector<column_values> const columns{
    {"w", {"10", "20"}, {"10", "50"}, {"10", "20", "50"}},
    {"d",
     {"2021-01-01", "2021-01-02"},
     {"2021-01-03", "2021-01-04"},
     {"2021-02-01", "2021-02-02", "2021-02-03"}},
    {"amount", {"1.00", "2.00"}, {"3.00", "4.00"}, {"10.00", "20.00", "30.00"}}};
  for (auto const& column : columns) {
    CAPTURE(column.name);
    for (char const* value : {"<>", "IS DISTINCT FROM"}) {
      CAPTURE(value);
      for (char const* first_key : {"=", "IS NOT DISTINCT FROM"}) {
        for (char const* second_key : {"=", "IS NOT DISTINCT FROM"}) {
          CAPTURE(first_key, second_key);
          rows expected{
            {"3000", column.k3000[0]}, {"3000", column.k3000[1]}, {"3001", column.k3001[1]}};
          // NULL keys pass the EXISTS only when it matches them; the NOT EXISTS then keeps all
          // three, or, when it matches them too, only the w=50 row.
          if (std::string(first_key) != "=") {
            bool const second_matches = std::string(second_key) != "=";
            for (std::size_t i = second_matches ? 2 : 0; i < 3; ++i) {
              expected.push_back({"NULL", column.null_key[i]});
            }
          }
          auto const differs = std::string(" g.") + column.name + " " + value + " f." + column.name;
          compare_rewritten_and_not(std::string("SELECT f.k, f.") + column.name +
                                      " FROM fact f WHERE f.v > 0 AND EXISTS (SELECT 1 FROM fact "
                                      "g WHERE g.k " +
                                      first_key + " f.k AND" + differs +
                                      ") AND NOT EXISTS (SELECT 1 FROM fact g WHERE g.k " +
                                      second_key + " f.k AND" + differs + " AND g.w > 40)",
                                    std::move(expected));
        }
      }
    }
  }
}
