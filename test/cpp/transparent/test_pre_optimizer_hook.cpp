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
 * @file test_pre_optimizer_hook.cpp
 * @brief The pre-optimizer hook leaves DuckDB's plan unchanged.
 *
 * DuckDB v2 derives the single-table restrictions implied by an OR-ed join condition itself. The
 * hook used to derive them as well, so every pushed restriction appeared twice. On TPC-H q19's
 * shape the plan with GPU execution on must therefore be the plan DuckDB builds on its own.
 */

#include "utils/gpu_execution_fixture.hpp"

#include <catch.hpp>

#include <cstddef>
#include <string>

namespace {

constexpr char const* q19_shape =
  "SELECT sum(l_extendedprice * (1 - l_discount)) AS revenue FROM lineitem, part "
  "WHERE (p_partkey = l_partkey AND p_brand = 'Brand#41' AND p_container IN ('SM CASE', 'SM BOX') "
  "AND l_quantity >= 2 AND l_quantity <= 12 AND p_size BETWEEN 1 AND 5) "
  "OR (p_partkey = l_partkey AND p_brand = 'Brand#13' AND p_container IN ('MED BAG', 'MED BOX') "
  "AND l_quantity >= 14 AND l_quantity <= 24 AND p_size BETWEEN 1 AND 10)";

std::size_t count_occurrences(std::string const& text, std::string const& needle)
{
  std::size_t count = 0;
  for (auto pos = text.find(needle); pos != std::string::npos;
       pos      = text.find(needle, pos + needle.size())) {
    ++count;
  }
  return count;
}

}  // namespace

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "pre-optimizer hook - an OR-ed join condition is not derived twice",
                 "[integration][gpu_execution][transparent][pre_optimizer_hook]")
{
  run_ok(
    "CREATE TABLE part AS SELECT range AS p_partkey, "
    "'Brand#' || CAST(range % 60 AS VARCHAR) AS p_brand, "
    "CASE range % 4 WHEN 0 THEN 'SM CASE' WHEN 1 THEN 'SM BOX' WHEN 2 THEN 'MED BAG' "
    "ELSE 'MED BOX' END AS p_container, (range % 50 + 1)::INTEGER AS p_size FROM range(20000);");
  run_ok(
    "CREATE TABLE lineitem AS SELECT range % 20000 AS l_partkey, "
    "((range % 50) + 1)::DECIMAL(15,2) AS l_quantity, "
    "((range * 7919) % 100000)::DECIMAL(15,2) AS l_extendedprice, "
    "((range % 11) / 100)::DECIMAL(15,2) AS l_discount FROM range(200000);");
  run_ok("CHECKPOINT;");

  auto const plan_text = [&](bool gpu) {
    run_ok(std::string("SET gpu_execution = ") + (gpu ? "true;" : "false;"));
    auto result = con->Query(std::string("EXPLAIN ") + q19_shape);
    REQUIRE(result);
    REQUIRE_FALSE(result->HasError());
    std::string text;
    for (auto const& row : collect_rows(*result, /*sort=*/false)) {
      for (auto const& cell : row) {
        text += cell;
      }
    }
    return text;
  };
  auto const cpu_plan = plan_text(false);
  auto const gpu_plan = plan_text(true);
  run_ok("SET gpu_execution = true;");
  INFO(gpu_plan);
  CHECK(gpu_plan == cpu_plan);
  // DuckDB derives the part-only restriction by itself; the hook adds no second copy.
  CHECK(count_occurrences(gpu_plan, "(p_brand = 'Brand#41') OR (p_brand = 'Brand#13')") ==
        count_occurrences(cpu_plan, "(p_brand = 'Brand#41') OR (p_brand = 'Brand#13')"));
  CHECK(count_occurrences(gpu_plan, "(p_brand = 'Brand#41') OR (p_brand = 'Brand#13')") >= 1);

  compare_gpu_vs_cpu(q19_shape);
}
