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

#include <string>

using RegexpReplaceFixture = sirius::test::GpuExecutionFixture;

namespace {

void create_regexp_replace_table(RegexpReplaceFixture& fx)
{
  fx.run_ok("CREATE TABLE regex_t(id INTEGER, s VARCHAR);");
  // Global replacement collapses aAB/aAC/aA/a into one key; first-match replacement
  // preserves aB, aC, and a. Include duplicates, NULLs, and multi-byte input as well.
  fx.run_ok(
    "INSERT INTO regex_t VALUES"
    " (1, 'aAB'), (2, 'aAC'), (3, 'aA'), (4, 'a'), (5, 'aAB'),"
    " (6, ''), (7, NULL), (8, 'AB'), (9, 'AC'), (10, 'éAB日C'), (11, 'no match');");
  fx.run_ok("CHECKPOINT;");
}

}  // namespace

TEST_CASE_METHOD(RegexpReplaceFixture,
                 "gpu_execution regexp_replace replaces only the first match",
                 "[integration][gpu_execution][projection][regexp_replace]")
{
  create_regexp_replace_table(*this);
  for (auto const* strategy : {"materialize", "ast_interpret", "ast_jit"}) {
    CAPTURE(strategy);
    run_ok(std::string("SET expression_evaluator_strategy = '") + strategy + "';");
    for (auto const* replacement : {"", "-", "XYZ"}) {
      CAPTURE(replacement);
      compare_gpu_vs_cpu(std::string("SELECT id, regexp_replace(s, '[A-Z]', '") + replacement +
                         "') FROM regex_t");
    }
    compare_gpu_vs_cpu("SELECT id, regexp_replace(s, '[A-Z]+', '-') FROM regex_t");
    compare_gpu_vs_cpu("SELECT id FROM regex_t WHERE regexp_replace(s, '[A-Z]', '') = 'aB'");
  }
  run_ok("SET expression_evaluator_strategy = 'ast_interpret';");
}

TEST_CASE_METHOD(RegexpReplaceFixture,
                 "gpu_execution regexp_replace preserves distinct grouping keys",
                 "[integration][gpu_execution][aggregate][regexp_replace]")
{
  create_regexp_replace_table(*this);
  // Regression for fuzz finding 002-mismatch-5de34b19: compare both keys and group counts.
  compare_gpu_vs_cpu(
    "SELECT regexp_replace(s, '[A-Z]', '') AS k, count(*) FROM regex_t GROUP BY k");
  compare_gpu_vs_cpu("SELECT regexp_replace(s, '[A-Z]', '') AS k FROM regex_t GROUP BY k");
  compare_gpu_vs_cpu("SELECT s, count(*) FROM regex_t GROUP BY s");
}

TEST_CASE_METHOD(RegexpReplaceFixture,
                 "regexp_replace options fall back without changing their semantics",
                 "[integration][gpu_execution][regexp_replace]")
{
  create_regexp_replace_table(*this);
  for (auto const* options : {"g", "i", "gi", "c", ""}) {
    CAPTURE(options);
    expect_plan_fallback_matches_cpu(std::string("SELECT id, regexp_replace(s, '[A-Z]', '', '") +
                                     options + "') FROM regex_t");
  }
  expect_plan_fallback_matches_cpu(
    "SELECT regexp_replace(s, '[A-Z]', '', 'g') AS k, count(*) FROM regex_t GROUP BY k");
}
