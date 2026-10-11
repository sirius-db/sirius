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
 * @file test_gpu_execution_regexp_replace.cpp
 * @brief GPU regexp_replace matches DuckDB.
 *
 * DuckDB replaces only the first match unless the options contain 'g', and reads patterns as RE2.
 * The GPU used to replace every match and drop the options. These cases compare GPU and CPU
 * results over a parquet file for first-match and global replacement, every option DuckDB
 * accepts, backreferences, anchors, newlines, non-ASCII text, empty strings and NULLs, and check
 * that each query ran on the GPU. Distinct results must stay distinct grouping keys. Column
 * arguments and RE2 syntax that cuDF reads differently fall back at plan time.
 */

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/dynamic_filter_test_utils.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/scoped_sirius_setting.hpp>

#include <string>

using RegexpReplaceFixture = sirius::test::GpuExecutionFixture;

namespace {

void expect_regex_fallback_error_matches_cpu(RegexpReplaceFixture& fx, std::string const& query)
{
  fx.run_ok("SET gpu_execution = true;");
  auto const before = sirius::test::get_transparent_execution_stats(*fx.con);
  auto gpu_result   = fx.con->Query(query);
  auto const after  = sirius::test::get_transparent_execution_stats(*fx.con);
  REQUIRE(gpu_result);
  REQUIRE(gpu_result->HasError());
  REQUIRE(gpu_result->GetError().find("Invalid replacement string") != std::string::npos);
  sirius::test::require_transparent_execution_delta(before, after, 0, 1, 0);
  sirius::test::scoped_sirius_setting cpu_only(*fx.con, "gpu_execution", false);
  auto cpu_result = fx.con->Query(query);
  REQUIRE(cpu_result);
  REQUIRE(cpu_result->HasError());
  REQUIRE(gpu_result->GetErrorType() == cpu_result->GetErrorType());
  REQUIRE(gpu_result->GetError() == cpu_result->GetError());
}

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
    " (20, 'APPLE Orange APPLE'), (21, 'a' || chr(10) || 'a'),"
    " (22, 'ab' || chr(10)), (23, 'a' || chr(10)),"
    " (24, 'https://www.example.com/a/b' || chr(10)), (25, 'https://www.example.com'),"
    " (26, 'https://www./x'), (27, 'HTTPS://Example.com/x'),"
    " (28, 'https://a.example/p' || chr(10) || 'q');");
  fx.run_ok("CHECKPOINT;");
}

// Writes the test strings to parquet and returns the FROM clause that reads them.
std::string create_strings(RegexpReplaceFixture& fx)
{
  auto const path = fx.temp_db_path + ".regexp_replace.parquet";
  fx.run_ok(
    "COPY (SELECT * FROM (VALUES"
    " (1, 'aaa0'), (2, 'abcabc'), (3, ''), (4, NULL), (5, 'no match here'),"
    " (6, 'AaA aAa'), (7, 'line1' || chr(10) || 'line2' || chr(10)), (8, 'a.b*c+d?e'),"
    " (9, 'foo123bar456'), (10, 'héllo wörld ñ'), (11, 'ab' || chr(10)), (12, 'xyzxyz'),"
    " (13, 'tab' || chr(9) || 'v' || chr(11) || 'r' || chr(13) || 'end'), (14, 'aaaa'),"
    " (15, 'Hello World HELLO'), (16, 'back\\slash\\1'), (17, '٣٤٥ digits 789'),"
    " (18, 'café CAFÉ'), (19, 'ab ab'), (20, 'ba'), (21, 'kelvin ' || chr(8490) || ' long ' ||"
    " chr(383) || ' Ks'), (22, 'http://www.example.com/a' || chr(10)), (23, 'a'),"
    " (24, '[x]{y}(z)|^$'), (25, 'http://www.example.com/path')"
    ") v(id, s)) TO '" +
    path + "' (FORMAT PARQUET);");
  return "read_parquet('" + path + "')";
}

}  // namespace

TEST_CASE_METHOD(RegexpReplaceFixture,
                 "gpu_execution regexp_replace replaces the first match like DuckDB",
                 "[integration][gpu_execution][projection][regexp_replace]")
{
  // Exercise regexp_replace before the unsupported v2 rewrite to regexp_extract (#2146).
  sirius::test::disabled_optimizers_guard optimizer_guard(*con, "expression_rewriter");
  auto const strings = create_strings(*this);
  for (auto const* call : {
         R"(regexp_replace(s, 'a', 'b'))",
         R"(regexp_replace(s, '[0-9]', '#'))",
         R"(regexp_replace(s, 'x', 'y'))",
         R"(regexp_replace(s, 'no such text', 'y'))",
         R"(regexp_replace(s, 'a+', '<\0>'))",
         R"(regexp_replace(s, 'a|ab', 'X'))",
         R"(regexp_replace(s, 'ab|a', 'X'))",
         R"(regexp_replace(s, 'a+?', 'X'))",
         R"(regexp_replace(s, '(\w+) (\w+)', '\2 \1'))",
         R"(regexp_replace(s, '(a)(b)?', '[\2\1]'))",
         R"(regexp_replace(s, '(a|ab)(c|bcd)', '[\1|\2]'))",
         R"(regexp_replace(s, 'b', '\\'))",
         R"(regexp_replace(s, 'b', '$1${1}'))",
         R"(regexp_replace(s, 'b', ''))",
       }) {
    CAPTURE(call);
    compare_gpu_vs_cpu(std::string("SELECT id, ") + call + " AS r FROM " + strings);
  }
  compare_gpu_vs_cpu("SELECT id FROM " + strings + " WHERE regexp_replace(s, 'a', 'b') = 'baa0'");
  // The ClickBench Q28 kernel is case-sensitive, so 'i' must take the generic path.
  for (auto const* options : {"", "i", "g", "s"}) {
    CAPTURE(options);
    compare_gpu_vs_cpu(
      std::string(
        R"(SELECT id, regexp_replace(upper(s), '^https?://(?:www\.)?([^/]+)/.*$', '\1', ')") +
      options + "') AS r FROM " + strings);
  }
  compare_gpu_vs_cpu(
    "SELECT id, upper(regexp_replace(regexp_replace(s, 'a', 'b'), 'b', 'c', 'g'))"
    " AS r FROM " +
    strings);
}

TEST_CASE_METHOD(RegexpReplaceFixture,
                 "gpu_execution regexp_replace honours DuckDB's options",
                 "[integration][gpu_execution][projection][regexp_replace]")
{
  auto const strings = create_strings(*this);
  for (auto const* options : {"g",
                              "i",
                              "gi",
                              "c",
                              "gc",
                              "ic",
                              "ci",
                              "s",
                              "gs",
                              "m",
                              "n",
                              "p",
                              "sm",
                              "l",
                              "gl",
                              "li",
                              "g i",
                              " g\t"}) {
    CAPTURE(options);
    for (auto const* pattern_and_replacement : {
           R"('a', 'b')",
           R"('hello', '#')",
           R"('ks', '#')",
           R"('[a-c]', '-')",
           R"('[^a ]', '-')",
           R"('a.', '<\0>')",
           R"('.', '_')",
           R"('([a-z]+) ([a-z]+)', '\2 \1')",
           R"('a.b', '#')",
           R"('[x]', '#')",
         }) {
      CAPTURE(pattern_and_replacement);
      // Under 'l' a backreference to a group the literal pattern lacks falls back.
      if (std::string(options).find('l') != std::string::npos &&
          std::string(pattern_and_replacement).find(R"(\1)") != std::string::npos) {
        continue;
      }
      compare_gpu_vs_cpu(std::string("SELECT id, regexp_replace(s, ") + pattern_and_replacement +
                         ", '" + options + "') AS r FROM " + strings);
    }
  }
}

TEST_CASE_METHOD(RegexpReplaceFixture,
                 "gpu_execution regexp_replace reads RE2 anchors, classes and escapes like DuckDB",
                 "[integration][gpu_execution][projection][regexp_replace]")
{
  auto const strings = create_strings(*this);
  for (auto const* pattern : {R"(^a)",
                              R"(c$)",
                              R"(b$)",
                              R"(2$)",
                              R"(\Aa)",
                              R"(.\z)",
                              R"(^https?://(?:www\.)?([^/]+)/.*$)",
                              R"(a^b)",
                              R"(\d+)",
                              R"(\w+)",
                              R"(\s)",
                              R"(\S+)",
                              R"(\W)",
                              R"(\D)",
                              R"([\d\s])",
                              R"([^\n]+)",
                              R"(\n)",
                              R"(\t|\x0b|\r)",
                              R"(\.|\*|\+|\?)",
                              R"(\[x\]|\{y\}|\(z\)|\||\^|\$)",
                              R"([]x])",
                              R"([\-\]])",
                              R"(é|ñ)",
                              R"([à-ÿ]+)",
                              R"(x{2}|y{1,}|z{1,2}?)",
                              R"((?:ab)+)",
                              R"(a\x41)"}) {
    CAPTURE(pattern);
    for (auto const* options : {"", "g"}) {
      CAPTURE(options);
      compare_gpu_vs_cpu(std::string("SELECT id, regexp_replace(s, '") + pattern + "', '<\\0>', '" +
                         options + "') AS r FROM " + strings);
    }
  }
}

TEST_CASE_METHOD(RegexpReplaceFixture,
                 "gpu_execution regexp_replace falls back at plan time for what the GPU cannot "
                 "reproduce",
                 "[integration][gpu_execution][projection][regexp_replace]")
{
  auto const strings = create_strings(*this);
  // Column arguments: the GPU compiles one regex per call.
  // s itself would not do: DuckDB v2 rejects the backslash in row 16 as a replacement.
  expect_plan_fallback_matches_cpu(
    "SELECT id, regexp_replace(s, 'a', CAST(id AS VARCHAR)) AS r FROM " + strings);
  expect_plan_fallback_matches_cpu(
    "SELECT id, regexp_replace(s, p, 'X') AS r FROM (SELECT id, s,"
    " CASE WHEN id % 2 = 0 THEN 'a' ELSE 'b' END AS p FROM " +
    strings + ")");
  for (auto const* call : {
         // Empty matches after the first, which cuDF places differently.
         R"(regexp_replace(s, '', 'X', 'g'))",
         R"(regexp_replace(s, 'a*', 'X', 'g'))",
         R"(regexp_replace(s, '(a?)', '<\1>', 'g'))",
         // RE2's \b is ASCII; cuDF's counts é as a word character.
         R"(regexp_replace(s, '\bl', 'X', 'g'))",
         // Syntax cuDF rejects.
         R"(regexp_replace(s, '\pL+', '#', 'g'))",
         R"(regexp_replace(s, 'a{1000}', '#'))",
         // Case folding beyond ASCII.
         R"(regexp_replace(s, 'É', '#', 'gi'))",
       }) {
    CAPTURE(call);
    expect_plan_fallback_matches_cpu(std::string("SELECT id, ") + call + " AS r FROM " + strings);
  }
}

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
  compare_gpu_vs_cpu(
    "SELECT regexp_replace(s, '[A-Z]', '', 'g') AS k, count(*) FROM regex_t GROUP BY k");
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
  // Exercise regexp_replace before the unsupported v2 rewrite to regexp_extract (#2146).
  sirius::test::disabled_optimizers_guard optimizer_guard(*con, "expression_rewriter");
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
                 "regexp_replace dollar anchors match only the end of text on GPU",
                 "[integration][gpu_execution][regexp_replace][regexp_replace_anchors]")
{
  // Exercise regexp_replace before the unsupported v2 rewrite to regexp_extract (#2146).
  sirius::test::disabled_optimizers_guard optimizer_guard(*con, "expression_rewriter");
  create_regexp_replace_table(*this);
  // cuDF's $ also matches before a trailing newline; the evaluator rewrites it to \Z.
  // Rows ending in a newline must therefore stay unchanged, exactly as in DuckDB.
  run_ok(R"(INSERT INTO regex_t VALUES (29, 'ab\' || chr(10)), (30, 'ab\'), (31, 'a$b$');)");
  run_ok("CHECKPOINT;");
  for (auto const* expression : {
         "regexp_replace(s, 'b$', 'X')",
         "regexp_replace(s, 'a$', 'X')",
         "regexp_replace(s, '.$', 'X')",
         "regexp_replace(s, '[^$]$', 'X')",
         "regexp_replace(s, 'a|b$', 'X')",
         "regexp_replace(s, '(?:b$)', 'X')",
         "regexp_replace(s, '[$]', 'X')",
         "regexp_replace(s, '[]$]', 'X')",
         "regexp_replace(s, '[^]$]+$', 'X')",
         R"(regexp_replace(s, '(b)$', '<\1>'))",
         R"(regexp_replace(s, '(a)$', '<\1>'))",
         R"(regexp_replace(s, '(a)(b)?$', '<\1\2>'))",
         R"(regexp_replace(s, '([$])', '<\1>'))",
         R"(regexp_replace(s, '^https?://(?:www\.)?([^/]+)/.*$', '\1'))",
         R"(regexp_replace(s, '\\$', 'X'))",
       }) {
    INFO(expression);
    compare_gpu_vs_cpu(std::string{"SELECT id, "} + expression + " FROM regex_t");
  }
}

TEST_CASE_METHOD(RegexpReplaceFixture,
                 "regexp_replace Q28 implementation stays on GPU before rewrites",
                 "[integration][gpu_execution][aggregate][regexp_replace][regexp_replace_anchors]")
{
  // Exercise regexp_replace before the unsupported v2 rewrite to regexp_extract (#2146).
  sirius::test::disabled_optimizers_guard optimizer_guard(*con, "expression_rewriter");
  create_regexp_replace_table(*this);
  // Q28's shape, with and without the dedicated JIT transform for its pattern.
  for (auto const* jit : {"true", "false"}) {
    INFO("enable_regex_jit_impl = " << jit);
    run_ok(std::string{"SET enable_regex_jit_impl = "} + jit + ";");
    compare_gpu_vs_cpu_approx(
      R"(SELECT regexp_replace(s, '^https?://(?:www\.)?([^/]+)/.*$', '\1') AS k,)"
      " avg(length(s)) AS l, count(*) AS c, min(s) FROM regex_t WHERE s <> '' GROUP BY k",
      {1});
  }
  run_ok("SET enable_regex_jit_impl = true;");
}

TEST_CASE_METHOD(RegexpReplaceFixture,
                 "regexp_replace inline flag groups fall back during planning",
                 "[integration][gpu_execution][regexp_replace][regexp_replace_anchors]")
{
  create_regexp_replace_table(*this);
  // cuDF rejects every (?...) form except (?:...). Multiline and dot-all flags would also
  // change what the rewritten $ anchor and . match.
  for (auto const* pattern : {"(?m)a$", "(?i)a$", "(?s)a.$", "(?i:a)", "(?P<x>a)$"}) {
    INFO(pattern);
    expect_plan_fallback_matches_cpu(std::string{"SELECT id, regexp_replace(s, '"} + pattern +
                                     "', 'X') FROM regex_t");
  }
}

TEST_CASE_METHOD(RegexpReplaceFixture,
                 "regexp_replace escaped dollars remain literal on GPU",
                 "[integration][gpu_execution][regexp_replace][regexp_replace_anchors]")
{
  run_ok("CREATE TABLE regex_t(s VARCHAR);");
  run_ok(R"(INSERT INTO regex_t VALUES ('$'), ('a$b$'), ('\$'), ('$' || chr(10)), (''), (NULL);)");
  run_ok("CHECKPOINT;");
  for (auto const* pattern : {R"(\$)", R"(\\\$)"}) {
    INFO(pattern);
    compare_gpu_vs_cpu(std::string{"SELECT regexp_replace(s, '"} + pattern +
                       "', 'X') FROM regex_t");
    compare_gpu_vs_cpu(std::string{"SELECT regexp_replace(s, '("} + pattern +
                       R"()', '<\1>') FROM regex_t)");
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
  // These can consume text even though they also match empty input, and replace only the first
  // match, which is at the start. The escaped pipe is a literal character, unlike the empty
  // alternation above. All stay on the GPU.
  for (auto const* pattern : {"a*", "a?", "a{0,1}", R"(\|)"}) {
    INFO(pattern);
    compare_gpu_vs_cpu(std::string{"SELECT regexp_replace(s, '"} + pattern +
                       "', 'X') FROM regex_t");
  }
}

TEST_CASE_METHOD(RegexpReplaceFixture,
                 "regexp_replace character classes keep RE2's ASCII meaning",
                 "[integration][gpu_execution][regexp_replace][regexp_replace_classes]")
{
  run_ok("CREATE TABLE regex_t(s VARCHAR);");
  run_ok(
    "INSERT INTO regex_t VALUES ('éa'), ('x٣y'), ('a' || chr(160) || 'b'),"
    " ('Ωmega'), ('abc'), ('a b'), (''), (NULL);");
  run_ok("CHECKPOINT;");
  // RE2's Perl classes are ASCII-only; the GPU spells them out as ASCII sets.
  for (auto const* pattern :
       {R"(\w)", R"(\W)", R"(\d)", R"(\D)", R"(\s)", R"(\S)", R"([\w])", R"(\\\w)"}) {
    INFO(pattern);
    compare_gpu_vs_cpu(std::string{"SELECT regexp_replace(s, '"} + pattern +
                       "', 'X') FROM regex_t");
    compare_gpu_vs_cpu(std::string{"SELECT regexp_replace(s, '("} + pattern +
                       R"()', '<\1>') FROM regex_t)");
  }
  // cuDF's word boundaries are Unicode-aware, and it does not implement POSIX classes.
  for (auto const* pattern : {R"(\b)",
                              R"(\B)",
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
                 "regexp_replace out-of-range backreferences preserve CPU errors",
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
    // Sirius declines GPU planning; DuckDB v2 reports the invalid rewrite on CPU.
    expect_regex_fallback_error_matches_cpu(
      *this, std::string{"SELECT id, "} + expression + " FROM regex_t");
  }
}

TEST_CASE_METHOD(RegexpReplaceFixture,
                 "regexp_replace rewrites cuDF would misread and dynamic arguments fall back",
                 "[integration][gpu_execution][regexp_replace]")
{
  create_regexp_replace_table(*this);
  // Literal mode removes the capture group, making this rewrite invalid in v2.
  expect_regex_fallback_error_matches_cpu(
    *this, R"(SELECT id, regexp_replace(s, '(a)', '<\1>', 'l') FROM regex_t)");
  for (auto const* expression : {
         R"(regexp_replace(s, '(a)', '\1${1}'))",
         "regexp_replace(s, s, 'X')",
       }) {
    INFO(expression);
    expect_plan_fallback_matches_cpu(std::string{"SELECT id, "} + expression + " FROM regex_t");
  }
  // An escaped backslash makes the rewrite literal text.
  compare_gpu_vs_cpu(R"(SELECT id, regexp_replace(s, '(a)', '\\1') FROM regex_t)");
}
