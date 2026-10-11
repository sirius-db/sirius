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
 * @file test_gpu_regexp_replace.cpp
 * @brief The plan-time translation of regexp_replace to cuDF calls.
 *
 * translate_gpu_regexp_replace decides at plan time whether the GPU can run a regexp_replace
 * call, and the evaluator runs what it returns. These cases pin the cuDF pattern, flags and
 * replacement it produces, and the inputs it rejects so that they fall back to the CPU.
 * test_gpu_execution_regexp_replace.cpp checks the results against DuckDB.
 */

#include "catch.hpp"
#include "expression/gpu_regexp_replace.hpp"

#include <optional>
#include <string>
#include <string_view>

using sirius::translate_gpu_regexp_replace;

namespace {

std::string pattern_of(std::string_view pattern, std::string_view options = "")
{
  auto const result = translate_gpu_regexp_replace(pattern, "x", options);
  REQUIRE(result.has_value());
  return result->pattern;
}

}  // namespace

TEST_CASE("gpu_regexp_replace - replaces the first match unless the options contain g",
          "[expression][regexp_replace]")
{
  auto const first = translate_gpu_regexp_replace("a", "b", "");
  REQUIRE(first.has_value());
  CHECK(first->pattern == "a");
  CHECK(first->replacement == "b");
  CHECK_FALSE(first->backrefs);
  CHECK(first->max_replace_count == 1);
  CHECK(first->flags == cudf::strings::regex_flags::DEFAULT);

  for (auto const* options : {"g", "gi", "ig", " g ", "cg"}) {
    CAPTURE(options);
    auto const global = translate_gpu_regexp_replace("a", "b", options);
    REQUIRE(global.has_value());
    CHECK_FALSE(global->max_replace_count.has_value());
  }
}

TEST_CASE("gpu_regexp_replace - rewrites RE2 syntax whose cuDF meaning differs",
          "[expression][regexp_replace]")
{
  // cuDF's $ also matches before a final newline; RE2's only at the end of the text.
  CHECK(pattern_of("b$") == R"(b\Z)");
  CHECK(pattern_of(R"(b\z)") == R"(b\Z)");
  CHECK(pattern_of(R"(\Ab)") == R"(\Ab)");
  // RE2's Perl classes are ASCII, and its \s excludes \v.
  CHECK(pattern_of(R"(\d+)") == "[0-9]+");
  CHECK(pattern_of(R"(\D)") == "[^0-9]");
  CHECK(pattern_of(R"(\s)") == "[\t\n\f\r ]");
  CHECK(pattern_of(R"(\S)") == "[^\t\n\f\r ]");
  CHECK(pattern_of(R"(\w)") == "[0-9A-Za-z_]");
  CHECK(pattern_of(R"(\W)") == "[^0-9A-Za-z_]");
  CHECK(pattern_of(R"([\d_x])") == "[0-9_x]");
  // Escapes of literals become the literal, escaped again only where cuDF needs it.
  CHECK(pattern_of(R"(\/\#\.\x41\x2e\t)") == "/#\\.A\\.\t");
  CHECK(pattern_of(R"([\]\-a-c^])") == R"([\]\-a-c\^])");
  CHECK(pattern_of("[]a]") == R"([\]a])");
  CHECK(pattern_of("a]}") == R"(a\]\})");
  CHECK(pattern_of("(?:ab|c)+?d{2,3}e{1,}") == "(?:ab|c)+?d{2,3}e{1,}");
  CHECK(pattern_of("héllo") == "héllo");
}

TEST_CASE("gpu_regexp_replace - expresses DuckDB's options", "[expression][regexp_replace]")
{
  // RE2 folds k with KELVIN SIGN and s with LATIN SMALL LETTER LONG S.
  CHECK(pattern_of("Ks1", "i") == "[KkK][sSſ]1");
  CHECK(pattern_of("[a-c_]", "i") == "[a-cA-C_]");
  CHECK(pattern_of("[^r-t]", "i") == "[^r-tR-Tſ]");
  CHECK(pattern_of(R"(\d\s\S\D.)", "i") == "[0-9][\t\n\f\r ][^\t\n\f\r ][^0-9].");
  // Later options override earlier ones.
  CHECK(pattern_of("a", "ic") == "a");
  CHECK(pattern_of("a", "ci") == "[aA]");

  CHECK(translate_gpu_regexp_replace("a.", "x", "s")->flags == cudf::strings::regex_flags::DOTALL);
  for (auto const* options : {"", "m", "n", "p", "sm", "sn", "sp"}) {
    CAPTURE(options);
    CHECK(translate_gpu_regexp_replace("a.", "x", options)->flags ==
          cudf::strings::regex_flags::DEFAULT);
  }

  CHECK(pattern_of("a.b*(c)", "l") == R"(a\.b\*\(c\))");
  CHECK(pattern_of("a.k", "li") == R"([aA]\.[kK)"
                                   "K]");
}

TEST_CASE("gpu_regexp_replace - follows RE2's replacement syntax", "[expression][regexp_replace]")
{
  // \\ is a backslash; $ and braces are literal.
  auto const literal = translate_gpu_regexp_replace("a", R"(\\$1${1})", "");
  REQUIRE(literal.has_value());
  CHECK_FALSE(literal->backrefs);
  CHECK(literal->replacement == R"(\$1${1})");

  auto const global = translate_gpu_regexp_replace(R"((\w+) (\w+))", R"(\2 \1\0\\)", "g");
  REQUIRE(global.has_value());
  CHECK(global->backrefs);
  CHECK(global->pattern == "([0-9A-Za-z_]+) ([0-9A-Za-z_]+)");
  CHECK(global->replacement == R"(${2} ${1}${0}\)");

  // replace_with_backrefs replaces every match, so the first match is anchored after a lazy
  // prefix that the replacement puts back.
  auto const first = translate_gpu_regexp_replace("(a)(b)", R"(<\2\1\0>)", "");
  REQUIRE(first.has_value());
  CHECK(first->pattern == R"(\A([\s\S]*?)((a)(b)))");
  CHECK(first->replacement == "${1}<${4}${3}${2}>");

  // A pattern anchored at the start matches at most once and needs no prefix.
  auto const anchored =
    translate_gpu_regexp_replace(R"(^https?://(?:www\.)?([^/]+)/.*$)", R"(\1)", "");
  REQUIRE(anchored.has_value());
  CHECK(anchored->pattern == R"(^https?://(?:www\.)?([^/]+)/.*\Z)");
  CHECK(anchored->replacement == "${1}");
  // A top-level alternation is not anchored.
  CHECK(translate_gpu_regexp_replace("^(a)|b", R"(\1)", "")->pattern == R"(\A([\s\S]*?)(^(a)|b))");
}

TEST_CASE("gpu_regexp_replace - rejects what the GPU cannot reproduce",
          "[expression][regexp_replace]")
{
  SECTION("patterns that match only the empty string")
  {
    for (auto const* pattern : {"", "^", "$", "^$", "()", "(?:)", "|", "(?:^|$)"}) {
      CAPTURE(pattern);
      CHECK_FALSE(translate_gpu_regexp_replace(pattern, "x", "g").has_value());
      CHECK_FALSE(translate_gpu_regexp_replace(pattern, "x", "").has_value());
    }
    CHECK_FALSE(translate_gpu_regexp_replace("", "x", "l").has_value());
  }
  SECTION("patterns that can match the empty string, under g")
  {
    for (auto const* pattern : {"a*", "a?", "a{0,2}", "a|", "(a|)", "(a*)b?"}) {
      CAPTURE(pattern);
      CHECK_FALSE(translate_gpu_regexp_replace(pattern, "x", "g").has_value());
      // The first match is at the start of the string, where cuDF finds it too.
      CHECK(translate_gpu_regexp_replace(pattern, "x", "").has_value());
    }
  }
  SECTION("syntax outside the supported subset")
  {
    for (auto const* pattern :
         {R"(\b)",    R"(a\B)",  "[[:digit:]]", R"(\pL)",    R"(\p{Greek})", "(?i)a",
          "(?P<n>a)", "(?<n>a)", R"(\Qa\E)",    R"(\x{e9})", R"(\xe9)",      R"(\Z)",
          R"(\v)",    R"(\1)",   R"([\D])",     R"(\C)",     "a**",          "a*+",
          "*a",       "a{,2}",   "a{2",         "{",         "a{2}{3}",      "^*",
          "[z-a]",    "[a-\\d]", "(a",          "a)",        "[a",           "\\"}) {
      CAPTURE(pattern);
      CHECK_FALSE(translate_gpu_regexp_replace(pattern, "x", "").has_value());
    }
  }
  SECTION("repetition counts RE2 or cuDF reject")
  {
    CHECK_FALSE(translate_gpu_regexp_replace("a{1001}", "x", "").has_value());
    CHECK_FALSE(translate_gpu_regexp_replace("(a{100}){11}", "x", "").has_value());
    // RE2 accepts a{1000}; cuDF does not compile it.
    CHECK_FALSE(translate_gpu_regexp_replace("a{1000}", "x", "").has_value());
  }
  SECTION("case folding beyond ASCII")
  {
    for (auto const* pattern : {"é", R"(\w)", R"(\W)", "[é]", "[a-é]", R"([\w])"}) {
      CAPTURE(pattern);
      CHECK_FALSE(translate_gpu_regexp_replace(pattern, "x", "i").has_value());
    }
  }
  SECTION("replacements RE2 rejects or cuDF would misread")
  {
    for (auto const* replacement : {R"(\x)", "\\", R"(\2)", R"(\1\\2)", R"(\1${0})"}) {
      CAPTURE(replacement);
      CHECK_FALSE(translate_gpu_regexp_replace("(a)", replacement, "g").has_value());
    }
    CHECK_FALSE(translate_gpu_regexp_replace("a", R"(\1)", "").has_value());
  }
  SECTION("options DuckDB does not accept")
  {
    CHECK_FALSE(translate_gpu_regexp_replace("a", "x", "x").has_value());
  }
}
