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
    " (6, ''), (7, NULL), (8, 'AB'), (9, 'AC'), (10, 'éAB日C'), (11, 'no match'),"
    " (12, 'aba aba aba'), (13, 'aa'), (14, 'baaa'), (15, '123-456-789'),"
    " (16, 'éabaéaba'), (17, 'first' || chr(10) || 'aba aba'),"
    " (18, 'https://www.example.com/a/b'), (19, 'http://other.example/x'),"
    " (20, 'APPLE Orange APPLE'), (21, 'a' || chr(10) || 'a');");
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
                 "regexp_replace replaces only the first match on GPU",
                 "[integration][gpu_execution][regexp_replace]")
{
  create_regexp_replace_table(*this);
  for (auto const* expression : {
         "regexp_replace(s, 'a', 'X')",
         "regexp_replace(s, 'aba', 'X')",
         "regexp_replace(s, 'a', '')",
         "regexp_replace(s, '^a', 'X')",
         "regexp_replace(s, 'a$', 'X')",
         "regexp_replace(s, '[0-9]+', '#')",
         "regexp_replace(s, 'a*', 'X')",
         "regexp_replace(s, 'a?', 'X')",
         "regexp_replace(s, 'a{0,1}', 'X')",
         R"(regexp_replace(s, '\|', 'X'))",
       }) {
    INFO(expression);
    compare_gpu_vs_cpu(std::string{"SELECT id, "} + expression + " FROM regex_t");
  }
}

TEST_CASE_METHOD(RegexpReplaceFixture,
                 "regexp_replace backreferences replace only the first match on GPU",
                 "[integration][gpu_execution][regexp_replace]")
{
  create_regexp_replace_table(*this);
  for (auto const* expression : {
         R"(regexp_replace(s, '(a)', '<\1>'))",
         R"(regexp_replace(s, '(a)(b)', '\2-\1-\2'))",
         R"(regexp_replace(s, '(aba)', '\12'))",
         R"(regexp_replace(s, '(?:a)(b)', '<\1>'))",
         R"(regexp_replace(s, '(a)(b)(a)( )(a)(b)(a)( )(aba)', '<\9>'))",
         R"(regexp_replace(s, 'aba', '<\0>'))",
         R"(regexp_replace(s, '[A-Z]', '\0\0'))",
         R"(regexp_replace(s, '[ \t\r\n\f]', '\0\0'))",
         R"(regexp_replace(s, '^(a)', '<\1>'))",
         R"(regexp_replace(s, '(a)$', '<\1>'))",
         R"(regexp_replace(s, '([0-9]+)', '<\1>'))",
         R"(regexp_replace(s, '(a*)', '<\1>'))",
         R"(regexp_replace(s, '(a?)', '<\1>'))",
         R"(regexp_replace(s, '(a)|(b)', '<\1><\2>'))",
         R"(regexp_replace(s, '^https?://(?:www\.)?([^/]+)/.*$', '\1'))",
       }) {
    INFO(expression);
    compare_gpu_vs_cpu(std::string{"SELECT id, "} + expression + " FROM regex_t");
  }
}

TEST_CASE_METHOD(RegexpReplaceFixture,
                 "regexp_replace empty-only patterns fall back during planning",
                 "[integration][gpu_execution][regexp_replace][regexp_replace_empty]")
{
  run_ok("CREATE TABLE regex_t(s VARCHAR);");
  run_ok("INSERT INTO regex_t VALUES ('abc'), (''), (NULL), ('a'), ('aba'), ('éa'), ('|abc');");
  run_ok("CHECKPOINT;");
  for (auto const* pattern : {"(?:)",
                              "a{0}",
                              "",
                              "()",
                              "|",
                              "(?:){2}",
                              "(a){0}",
                              "(?:a{0}|())",
                              "(?:a{0})*",
                              "^",
                              "$",
                              "^$"}) {
    INFO(pattern);
    expect_plan_fallback_matches_cpu(std::string{"SELECT regexp_replace(s, '"} + pattern +
                                     "', 'X') FROM regex_t");
    expect_plan_fallback_matches_cpu(std::string{"SELECT regexp_replace(s, '("} + pattern +
                                     R"()', '<\1>') FROM regex_t)");
  }
  // These can consume text even though they also match empty input. The escaped pipe is
  // a literal character, unlike the empty alternation above, and should stay on the GPU.
  for (auto const* pattern : {"a*", "a?", "a{0,1}", R"(\|)"}) {
    INFO(pattern);
    compare_gpu_vs_cpu(std::string{"SELECT regexp_replace(s, '"} + pattern +
                       "', 'X') FROM regex_t");
  }
}

TEST_CASE_METHOD(RegexpReplaceFixture,
                 "regexp_replace incompatible character classes fall back during planning",
                 "[integration][gpu_execution][regexp_replace][regexp_replace_classes]")
{
  run_ok("CREATE TABLE regex_t(s VARCHAR);");
  run_ok(
    "INSERT INTO regex_t VALUES ('éa'), ('x٣y'), ('a' || chr(160) || 'b'),"
    " ('Ωmega'), ('abc'), ('a b'), (''), (NULL);");
  run_ok("CHECKPOINT;");
  for (auto const* pattern : {R"(\w)",
                              R"(\W)",
                              R"(\d)",
                              R"(\D)",
                              R"(\s)",
                              R"(\S)",
                              R"(\b)",
                              R"(\B)",
                              R"([\w])",
                              R"(\\\w)",
                              "[[:alpha:]]",
                              "[[:digit:]]",
                              "[[:space:]]",
                              "[^[:alpha:]]",
                              "[a[:digit:]]"}) {
    INFO(pattern);
    // RE2 can match \B inside a UTF-8 sequence. A leading space keeps its first match at
    // a character boundary, so the CPU result can be materialized by the comparison helper.
    auto const subject = std::string{pattern} == R"(\B)" ? "' ' || s" : "s";
    expect_plan_fallback_matches_cpu(std::string{"SELECT regexp_replace("} + subject + ", '" +
                                     pattern + "', 'X') FROM regex_t");
    expect_plan_fallback_matches_cpu(std::string{"SELECT regexp_replace("} + subject + ", '(" +
                                     pattern + R"()', '<\1>') FROM regex_t)");
  }
}

TEST_CASE_METHOD(RegexpReplaceFixture,
                 "regexp_replace escaped class names remain literal on GPU",
                 "[integration][gpu_execution][regexp_replace][regexp_replace_classes]")
{
  run_ok("CREATE TABLE regex_t(s VARCHAR);");
  run_ok(R"(INSERT INTO regex_t VALUES ('\w\W\d\D\s\S\b\B'), ('éa'), (NULL);)");
  run_ok("CHECKPOINT;");
  for (auto const* pattern :
       {R"(\\w)", R"(\\W)", R"(\\d)", R"(\\D)", R"(\\s)", R"(\\S)", R"(\\b)", R"(\\B)"}) {
    INFO(pattern);
    compare_gpu_vs_cpu(std::string{"SELECT regexp_replace(s, '"} + pattern +
                       "', 'X') FROM regex_t");
  }
}

TEST_CASE_METHOD(RegexpReplaceFixture,
                 "regexp_replace out-of-range backreferences fall back during planning",
                 "[integration][gpu_execution][regexp_replace]")
{
  create_regexp_replace_table(*this);
  for (auto const* expression : {
         R"(regexp_replace(s, '(a)', '\9'))",
         R"(regexp_replace(s, '(a)', '\2'))",
         R"(regexp_replace(s, 'a', '\1'))",
         R"(regexp_replace(s, '(?:a)', '\1'))",
         R"(regexp_replace(s, '\(a\)', '\1'))",
         R"(regexp_replace(s, '[(]a[)]', '\1'))",
         R"(regexp_replace(s, '(a)', '\0\1\2'))",
         R"(regexp_replace(s, 'a', '\12'))",
       }) {
    INFO(expression);
    // Use a column to prevent constant folding, and assert fallback occurs before GPU execution.
    expect_plan_fallback_matches_cpu(std::string{"SELECT id, "} + expression + " FROM regex_t");
  }
}

TEST_CASE_METHOD(RegexpReplaceFixture,
                 "regexp_replace options and unsupported rewrites fall back during planning",
                 "[integration][gpu_execution][regexp_replace]")
{
  create_regexp_replace_table(*this);
  // Four-argument calls were never supported by the evaluator. All explicit options must
  // retain DuckDB semantics, including global replacement and combinations of flags.
  for (auto const* options : {"", "g", "c", "i", "l", "m", "n", "p", "s", "gi", "gs"}) {
    INFO(options);
    expect_plan_fallback_matches_cpu(std::string{"SELECT id, regexp_replace(s, 'a.', 'X', '"} +
                                     options + "') FROM regex_t");
    expect_plan_fallback_matches_cpu(
      std::string{R"(SELECT id, regexp_replace(s, '(a)', '<\1>', ')"} + options +
      "') FROM regex_t");
  }
  for (auto const* expression : {
         R"(regexp_replace(s, '(a)', '\\1'))",
         R"(regexp_replace(s, '(a)', '\1${1}'))",
         "regexp_replace(s, s, 'X')",
         "regexp_replace(s, 'a', s)",
       }) {
    INFO(expression);
    expect_plan_fallback_matches_cpu(std::string{"SELECT id, "} + expression + " FROM regex_t");
  }
  expect_plan_fallback_matches_cpu(
    "SELECT regexp_replace(s, '[A-Z]', '', 'g') AS k, count(*) FROM regex_t GROUP BY k");
}
