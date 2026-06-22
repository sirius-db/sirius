/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * See the LICENSE file at the repo root for the full text.
 */

#include "catch.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::string_view kGuardSourceFilename = "test_s3_default_visibility_guard.cpp";

struct test_case_span {
  std::string header;
  std::string body;
  std::size_t line{0};
};

std::filesystem::path project_root()
{
#ifdef SIRIUS_PROJECT_ROOT
  return std::filesystem::path{SIRIUS_PROJECT_ROOT};
#else
  return std::filesystem::current_path();
#endif
}

std::string read_text_file(std::filesystem::path const& path)
{
  std::ifstream in(path);
  REQUIRE(in);
  std::ostringstream out;
  out << in.rdbuf();
  return out.str();
}

std::size_t line_number(std::string_view text, std::size_t offset)
{
  std::size_t line = 1;
  for (std::size_t i = 0; i < offset && i < text.size(); ++i) {
    if (text[i] == '\n') { ++line; }
  }
  return line;
}

bool starts_with_at(std::string_view text, std::size_t pos, std::string_view token)
{
  return pos <= text.size() && token.size() <= text.size() - pos &&
         text.substr(pos, token.size()) == token;
}

bool is_identifier_char(char c)
{
  return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_';
}

bool has_identifier_boundary_before(std::string_view text, std::size_t pos)
{
  return pos == 0 || !is_identifier_char(text[pos - 1]);
}

bool is_digit_separator(std::string_view text, std::size_t pos)
{
  return pos > 0 && pos + 1 < text.size() &&
         std::isalnum(static_cast<unsigned char>(text[pos - 1])) != 0 &&
         std::isalnum(static_cast<unsigned char>(text[pos + 1])) != 0;
}

std::size_t skip_line_comment(std::string_view text, std::size_t pos)
{
  auto const end = text.find('\n', pos + 2);
  return end == std::string_view::npos ? text.size() : end + 1;
}

std::size_t skip_block_comment(std::string_view text, std::size_t pos)
{
  auto const end = text.find("*/", pos + 2);
  return end == std::string_view::npos ? text.size() : end + 2;
}

std::size_t skip_quoted_literal(std::string_view text, std::size_t pos, char quote)
{
  bool escaped = false;
  for (std::size_t i = pos + 1; i < text.size(); ++i) {
    auto const c = text[i];
    if (escaped) {
      escaped = false;
    } else if (c == '\\') {
      escaped = true;
    } else if (c == quote) {
      return i + 1;
    }
  }
  return text.size();
}

std::size_t skip_raw_string_literal(std::string_view text, std::size_t pos)
{
  if (!starts_with_at(text, pos, "R\"")) { return std::string_view::npos; }

  auto const delimiter_start = pos + 2;
  auto const open_paren      = text.find('(', delimiter_start);
  if (open_paren == std::string_view::npos || open_paren - delimiter_start > 16) {
    return std::string_view::npos;
  }

  auto const delimiter = text.substr(delimiter_start, open_paren - delimiter_start);
  if (delimiter.find_first_of(" \t\n\r()\\") != std::string_view::npos) {
    return std::string_view::npos;
  }

  auto const close = ")" + std::string{delimiter} + "\"";
  auto const end   = text.find(close, open_paren + 1);
  return end == std::string_view::npos ? text.size() : end + close.size();
}

std::size_t skip_ignored_region(std::string_view text, std::size_t pos)
{
  if (starts_with_at(text, pos, "//")) { return skip_line_comment(text, pos); }
  if (starts_with_at(text, pos, "/*")) { return skip_block_comment(text, pos); }

  auto const raw_end = skip_raw_string_literal(text, pos);
  if (raw_end != std::string_view::npos) { return raw_end; }

  if (pos < text.size() && text[pos] == '"') { return skip_quoted_literal(text, pos, '"'); }
  if (pos < text.size() && text[pos] == '\'' && !is_digit_separator(text, pos)) {
    return skip_quoted_literal(text, pos, '\'');
  }

  return std::string_view::npos;
}

std::size_t find_token_in_code(std::string_view text, std::string_view token, std::size_t pos = 0)
{
  for (std::size_t i = pos; i < text.size();) {
    auto const skipped = skip_ignored_region(text, i);
    if (skipped != std::string_view::npos) {
      i = skipped;
      continue;
    }
    if (starts_with_at(text, i, token) && has_identifier_boundary_before(text, i)) { return i; }
    ++i;
  }
  return std::string_view::npos;
}

std::size_t find_test_case_macro_in_code(std::string_view text, std::size_t start)
{
  constexpr std::array<std::string_view, 2> kTestCaseMacros = {"TEST_CASE(", "TEMPLATE_TEST_CASE("};

  std::size_t best = std::string_view::npos;
  for (auto const token : kTestCaseMacros) {
    auto const candidate = find_token_in_code(text, token, start);
    if (candidate != std::string_view::npos &&
        (best == std::string_view::npos || candidate < best)) {
      best = candidate;
    }
  }
  return best;
}

std::size_t find_char_in_code(std::string_view text, char token, std::size_t pos = 0)
{
  for (std::size_t i = pos; i < text.size();) {
    auto const skipped = skip_ignored_region(text, i);
    if (skipped != std::string_view::npos) {
      i = skipped;
      continue;
    }
    if (text[i] == token) { return i; }
    ++i;
  }
  return std::string_view::npos;
}

std::size_t find_matching_brace(std::string_view text, std::size_t open_brace)
{
  std::size_t depth = 0;

  for (std::size_t i = open_brace; i < text.size();) {
    auto const skipped = skip_ignored_region(text, i);
    if (skipped != std::string_view::npos) {
      i = skipped;
      continue;
    }

    auto const c = text[i];
    if (c == '{') {
      ++depth;
      ++i;
      continue;
    }
    if (c == '}') {
      if (depth == 0) { return std::string_view::npos; }
      --depth;
      if (depth == 0) { return i; }
    }
    ++i;
  }
  return std::string_view::npos;
}

std::vector<test_case_span> collect_test_cases(std::string const& source)
{
  std::vector<test_case_span> out;
  std::size_t pos = 0;
  while (true) {
    auto const test_pos = find_test_case_macro_in_code(source, pos);
    if (test_pos == std::string::npos) { break; }
    auto const body_open = find_char_in_code(source, '{', test_pos);
    REQUIRE(body_open != std::string::npos);
    auto const body_close = find_matching_brace(source, body_open);
    REQUIRE(body_close != std::string::npos);

    out.push_back(test_case_span{source.substr(test_pos, body_open - test_pos),
                                 source.substr(body_open, body_close - body_open + 1),
                                 line_number(source, test_pos)});
    pos = body_close + 1;
  }
  return out;
}

bool has_tag(std::string_view header, std::string_view tag)
{
  return header.find("[" + std::string{tag} + "]") != std::string_view::npos;
}

bool hidden_by_default(std::string_view header)
{
  return header.find("[.]") != std::string_view::npos ||
         header.find("[!") != std::string_view::npos;
}

bool references_live_s3_guard(std::string_view body)
{
  return find_token_in_code(body, "read_s3_test_env(") != std::string_view::npos ||
         find_token_in_code(body, "skip_if_no_s3_env(") != std::string_view::npos;
}

bool should_scan_file(std::filesystem::path const& path)
{
  return path.filename() != kGuardSourceFilename;
}

std::vector<std::filesystem::path> discover_cpp_test_files(std::filesystem::path const& root)
{
  std::vector<std::filesystem::path> files;
  auto const test_root = root / "test/cpp";
  REQUIRE(std::filesystem::exists(test_root));

  for (auto const& entry : std::filesystem::recursive_directory_iterator(test_root)) {
    if (!entry.is_regular_file() || entry.path().extension() != ".cpp") { continue; }
    auto const relative = std::filesystem::relative(entry.path(), root);
    if (!should_scan_file(relative)) { continue; }
    files.push_back(relative);
  }

  std::sort(files.begin(), files.end());
  return files;
}

std::vector<std::string> collect_visibility_violations(std::filesystem::path const& relative,
                                                       std::string const& source)
{
  INFO("Scanning " << relative.string());
  if (!should_scan_file(relative)) { return {}; }

  std::vector<std::string> violations;
  for (auto const& test_case : collect_test_cases(source)) {
    if (!has_tag(test_case.header, "s3") || hidden_by_default(test_case.header)) { continue; }
    if (!references_live_s3_guard(test_case.body)) { continue; }
    violations.push_back(relative.string() + ":" + std::to_string(test_case.line));
  }
  return violations;
}

std::string join_lines(std::vector<std::string> const& values)
{
  std::ostringstream out;
  for (auto const& value : values) {
    out << "\n  " << value;
  }
  return out.str();
}

}  // namespace

TEST_CASE("default-visible S3 tests do not silently skip live S3 work", "[s3][test_hygiene]")
{
  auto const root  = project_root();
  auto const files = discover_cpp_test_files(root);
  REQUIRE_FALSE(files.empty());

  std::vector<std::string> violations;
  for (auto const& relative : files) {
    auto file_violations = collect_visibility_violations(relative, read_text_file(root / relative));
    violations.insert(violations.end(), file_violations.begin(), file_violations.end());
  }

  INFO(
    "Default-visible [s3] TEST_CASE bodies must not call read_s3_test_env/"
    "skip_if_no_s3_env; use [.][s3][integration] for live S3 tests. Offenders:"
    << join_lines(violations));
  CHECK(violations.empty());
}

TEST_CASE("S3 default-visibility guard classifies synthetic test sources", "[s3][test_hygiene]")
{
  SECTION("excludes the guard source file itself from auto-discovery results")
  {
    auto const source = R"cpp(
      TEST_CASE("self-referential guard", "[s3][test_hygiene]")
      {
        read_s3_test_env();
      }
    )cpp";

    CHECK(
      collect_visibility_violations("test/cpp/io/s3/test_s3_default_visibility_guard.cpp", source)
        .empty());
  }

  SECTION("flags a newly discovered default-visible S3 test that calls the live guard")
  {
    auto const source = R"cpp(
      TEST_CASE("new file with live S3 work", "[s3]")
      {
        auto env = read_s3_test_env();
      }
    )cpp";

    auto const violations =
      collect_visibility_violations("test/cpp/new_area/test_new_s3_file.cpp", source);
    REQUIRE(violations.size() == 1);
    CHECK(violations.front() == "test/cpp/new_area/test_new_s3_file.cpp:2");
  }

  SECTION("handles Catch template test cases without matching TEST_CASE as a substring")
  {
    auto const source = R"cpp(
      TEMPLATE_TEST_CASE("template S3 file with live work", "[s3]", int, long)
      {
        auto env = read_s3_test_env();
      }

      MY_TEST_CASE("not a Catch test", "[s3]") { auto env = read_s3_test_env(); }
    )cpp";

    auto const violations =
      collect_visibility_violations("test/cpp/new_area/test_template_s3_file.cpp", source);
    REQUIRE(violations.size() == 1);
    CHECK(violations.front() == "test/cpp/new_area/test_template_s3_file.cpp:2");
  }

  SECTION("does not flag hidden S3 integration or benchmark tests")
  {
    auto const integration_source = R"cpp(
      TEST_CASE("hidden live S3 test", "[.][s3][integration]")
      {
        read_s3_test_env();
      }
    )cpp";
    auto const benchmark_source   = R"cpp(
      TEST_CASE("hidden S3 benchmark", "[!benchmark][s3]")
      {
        skip_if_no_s3_env();
      }
    )cpp";

    CHECK(
      collect_visibility_violations("test/cpp/integration/test_hidden_s3.cpp", integration_source)
        .empty());
    CHECK(
      collect_visibility_violations("test/cpp/integration/test_benchmark_s3.cpp", benchmark_source)
        .empty());
  }

  SECTION("does not flag clean S3 tests or non-S3 tests that call the live guard")
  {
    auto const clean_s3_source = R"cpp(
      TEST_CASE("pure S3 logic", "[s3]") { CHECK(true); }
    )cpp";
    auto const non_s3_source   = R"cpp(
      TEST_CASE("helper with live guard", "[test_hygiene]")
      {
        read_s3_test_env();
      }
    )cpp";

    CHECK(collect_visibility_violations("test/cpp/io/s3/test_clean.cpp", clean_s3_source).empty());
    CHECK(collect_visibility_violations("test/cpp/helper/test_non_s3.cpp", non_s3_source).empty());
  }

  SECTION("ignores live-guard tokens inside comments and string literals")
  {
    auto const source = R"outer(
TEST_CASE("comments and strings are inert", "[s3]")
{
  // read_s3_test_env();
  /* skip_if_no_s3_env(); */
  auto regular = "read_s3_test_env(";
  auto raw = R"guard(skip_if_no_s3_env({}))guard";
  CHECK(!regular.empty());
}
)outer";

    CHECK(collect_visibility_violations("test/cpp/io/s3/test_comments.cpp", source).empty());
  }

  SECTION("matches braces while skipping comments and raw string literals")
  {
    auto const source = R"outer(
TEST_CASE("brace tokenizer", "[s3]")
{
  // } read_s3_test_env();
  /* { skip_if_no_s3_env(); } */
  auto raw = R"guard({ read_s3_test_env( } )guard";
  auto rows = 100'000;
  int live = 42;
}
int outside = 7;
)outer";

    auto const cases = collect_test_cases(source);
    REQUIRE(cases.size() == 1);
    CHECK(cases.front().body.find("100'000") != std::string::npos);
    CHECK(cases.front().body.find("int live = 42;") != std::string::npos);
    CHECK(cases.front().body.find("int outside = 7;") == std::string::npos);
    CHECK_FALSE(references_live_s3_guard(cases.front().body));
  }
}
