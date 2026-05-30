/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * See the LICENSE file at the repo root for the full text.
 */

#include "catch.hpp"

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

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

std::size_t find_matching_brace(std::string_view text, std::size_t open_brace)
{
  std::size_t depth = 0;
  bool in_string    = false;
  bool in_char      = false;
  bool escaped      = false;

  for (std::size_t i = open_brace; i < text.size(); ++i) {
    char const c = text[i];
    if (in_string || in_char) {
      if (escaped) {
        escaped = false;
      } else if (c == '\\') {
        escaped = true;
      } else if ((in_string && c == '"') || (in_char && c == '\'')) {
        in_string = false;
        in_char   = false;
      }
      continue;
    }
    if (c == '"') {
      in_string = true;
      continue;
    }
    if (c == '\'') {
      in_char = true;
      continue;
    }
    if (c == '{') {
      ++depth;
      continue;
    }
    if (c == '}') {
      if (depth == 0) { return std::string_view::npos; }
      --depth;
      if (depth == 0) { return i; }
    }
  }
  return std::string_view::npos;
}

std::vector<test_case_span> collect_test_cases(std::string const& source)
{
  std::vector<test_case_span> out;
  std::size_t pos = 0;
  while (true) {
    auto const test_pos = source.find("TEST_CASE(", pos);
    if (test_pos == std::string::npos) { break; }
    auto const body_open = source.find('{', test_pos);
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
  return body.find("read_s3_test_env(") != std::string_view::npos ||
         body.find("skip_if_no_s3_env(") != std::string_view::npos;
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

TEST_CASE("default-visible S3 tests do not silently skip live MinIO work", "[s3][test_hygiene]")
{
  auto const root = project_root();
  std::vector<std::filesystem::path> const files{
    "test/cpp/io/s3/test_s3_ioctx.cpp",
    "test/cpp/scan_manager/test_sirius_scan_manager_s3.cpp",
    "test/cpp/scan_manager/test_describe_parquet_s3.cpp",
    "test/cpp/scan/test_parquet_split_provider_s3.cpp",
    "test/cpp/integration/test_scan_manager_s3_end_to_end.cpp",
    "test/cpp/integration/test_s3_sql_surface.cpp",
  };

  std::vector<std::string> violations;
  for (auto const& relative : files) {
    auto const source = read_text_file(root / relative);
    for (auto const& test_case : collect_test_cases(source)) {
      if (!has_tag(test_case.header, "s3") || hidden_by_default(test_case.header)) { continue; }
      if (!references_live_s3_guard(test_case.body)) { continue; }
      violations.push_back(relative.string() + ":" + std::to_string(test_case.line));
    }
  }

  INFO(
    "Default-visible [s3] TEST_CASE bodies must not call read_s3_test_env/"
    "skip_if_no_s3_env; use [.][s3][integration] for live MinIO tests. Offenders:"
    << join_lines(violations));
  CHECK(violations.empty());
}
