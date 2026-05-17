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

#include "sirius_sql_rewrite.hpp"

#include <cctype>
#include <cstddef>
#include <optional>
#include <string_view>
#include <utility>

namespace sirius {

namespace {

constexpr std::string_view NATIVE_READ_PARQUET_FN = "read_parquet";
constexpr std::string_view SIRIUS_READ_PARQUET_FN = "sirius_read_parquet";
constexpr std::string_view S3_URI_PREFIX          = "s3://";

bool is_identifier_char(char c)
{
  auto const ch = static_cast<unsigned char>(c);
  return std::isalnum(ch) != 0 || c == '_';
}

bool iequals_prefix(std::string_view value, std::string_view prefix)
{
  if (value.size() < prefix.size()) { return false; }
  for (std::size_t i = 0; i < prefix.size(); ++i) {
    auto const lhs = static_cast<unsigned char>(value[i]);
    auto const rhs = static_cast<unsigned char>(prefix[i]);
    if (std::tolower(lhs) != std::tolower(rhs)) { return false; }
  }
  return true;
}

std::size_t skip_whitespace(std::string const& text, std::size_t pos)
{
  while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])) != 0) {
    ++pos;
  }
  return pos;
}

// Parse a single-quoted SQL string literal starting at @p pos. Handles the
// doubled-quote escape (`''`). Returns the decoded value and the index just
// past the closing quote, or nullopt if @p pos is not a literal start or the
// literal is unterminated.
std::optional<std::pair<std::string, std::size_t>> parse_single_quoted_literal(
  std::string const& text, std::size_t pos)
{
  if (pos >= text.size() || text[pos] != '\'') { return std::nullopt; }

  std::string literal;
  for (std::size_t i = pos + 1; i < text.size(); ++i) {
    if (text[i] == '\'') {
      if (i + 1 < text.size() && text[i + 1] == '\'') {
        literal.push_back('\'');
        ++i;
        continue;
      }
      return std::pair{std::move(literal), i + 1};
    }
    literal.push_back(text[i]);
  }
  return std::nullopt;
}

// A `read_parquet` call is Sirius-owned (and gets rewritten) only when its
// first argument is a single-quoted string literal whose value carries the
// `s3://` scheme. Local paths and non-literal arguments are left to DuckDB.
bool should_rewrite_read_parquet_call(std::string const& query, std::size_t name_pos)
{
  auto pos = skip_whitespace(query, name_pos + NATIVE_READ_PARQUET_FN.size());
  if (pos >= query.size() || query[pos] != '(') { return false; }

  pos          = skip_whitespace(query, pos + 1);
  auto literal = parse_single_quoted_literal(query, pos);
  if (!literal.has_value()) { return false; }
  return iequals_prefix(literal->first, S3_URI_PREFIX);
}

}  // namespace

std::string rewrite_sirius_owned_remote_parquet_calls(std::string const& query)
{
  std::string rewritten;
  rewritten.reserve(query.size() + 32);

  bool in_single_quote  = false;
  bool in_double_quote  = false;
  bool in_line_comment  = false;
  bool in_block_comment = false;

  for (std::size_t i = 0; i < query.size();) {
    auto const c = query[i];

    if (in_line_comment) {
      rewritten.push_back(c);
      if (c == '\n') { in_line_comment = false; }
      ++i;
      continue;
    }
    if (in_block_comment) {
      rewritten.push_back(c);
      if (c == '*' && i + 1 < query.size() && query[i + 1] == '/') {
        rewritten.push_back('/');
        i += 2;
        in_block_comment = false;
      } else {
        ++i;
      }
      continue;
    }
    if (in_single_quote) {
      rewritten.push_back(c);
      if (c == '\'') {
        if (i + 1 < query.size() && query[i + 1] == '\'') {
          rewritten.push_back('\'');
          i += 2;
          continue;
        }
        in_single_quote = false;
      }
      ++i;
      continue;
    }
    if (in_double_quote) {
      rewritten.push_back(c);
      if (c == '"') {
        if (i + 1 < query.size() && query[i + 1] == '"') {
          rewritten.push_back('"');
          i += 2;
          continue;
        }
        in_double_quote = false;
      }
      ++i;
      continue;
    }

    if (c == '-' && i + 1 < query.size() && query[i + 1] == '-') {
      rewritten.append("--");
      i += 2;
      in_line_comment = true;
      continue;
    }
    if (c == '/' && i + 1 < query.size() && query[i + 1] == '*') {
      rewritten.append("/*");
      i += 2;
      in_block_comment = true;
      continue;
    }
    if (c == '\'') {
      rewritten.push_back(c);
      ++i;
      in_single_quote = true;
      continue;
    }
    if (c == '"') {
      rewritten.push_back(c);
      ++i;
      in_double_quote = true;
      continue;
    }

    auto const remaining = std::string_view(query).substr(i);
    if ((i == 0 || !is_identifier_char(query[i - 1])) &&
        iequals_prefix(remaining, NATIVE_READ_PARQUET_FN) &&
        (i + NATIVE_READ_PARQUET_FN.size() >= query.size() ||
         !is_identifier_char(query[i + NATIVE_READ_PARQUET_FN.size()])) &&
        should_rewrite_read_parquet_call(query, i)) {
      rewritten.append(SIRIUS_READ_PARQUET_FN);
      i += NATIVE_READ_PARQUET_FN.size();
      continue;
    }

    rewritten.push_back(c);
    ++i;
  }

  return rewritten;
}

}  // namespace sirius
