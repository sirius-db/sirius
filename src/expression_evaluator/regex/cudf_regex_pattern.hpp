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

#pragma once

#include <optional>
#include <string>
#include <string_view>

namespace sirius {
namespace regex {

/// Returns the explicit ASCII class for RE2's Perl class escape `\<c>`, or nullptr if `c`
/// does not name one. cuDF's `\d`, `\w` and `\s` are Unicode-aware, and its `\s` also
/// matches `\v`. RE2's `\s` is exactly tab, newline, form feed, carriage return and space.
inline char const* ascii_perl_class(char c)
{
  switch (c) {
    case 'd': return "[0-9]";
    case 'D': return "[^0-9]";
    case 'w': return "[0-9A-Za-z_]";
    case 'W': return "[^0-9A-Za-z_]";
    case 's': return "[\t\n\f\r ]";
    case 'S': return "[^\t\n\f\r ]";
    default: return nullptr;
  }
}

/// Rewrites a RE2 pattern for cuDF's regex engine. RE2's `$` matches only at the end of the
/// text by default, whereas cuDF's `$` also matches before a trailing newline. cuDF's `\Z`
/// has RE2's end-of-text semantics, so every anchoring `$` is rewritten to `\Z`. Escaped
/// dollars and dollars inside bracket classes are literals for both engines and stay as-is.
/// Perl class escapes become explicit ASCII classes. Inside a bracket class they cannot be
/// spliced in that way, so such patterns return nullopt.
inline std::optional<std::string> to_cudf_regex_pattern(std::string_view pattern)
{
  std::string result;
  result.reserve(pattern.size() + 8);
  bool in_class = false;
  for (std::size_t i = 0; i < pattern.size(); ++i) {
    char const c = pattern[i];
    if (c == '\\') {
      if (i + 1 < pattern.size()) {
        if (auto const* ascii_class = ascii_perl_class(pattern[i + 1])) {
          if (in_class) { return std::nullopt; }
          result += ascii_class;
          ++i;
          continue;
        }
      }
      result += c;
      if (i + 1 < pattern.size()) { result += pattern[++i]; }
    } else if (in_class) {
      result += c;
      if (c == ']') { in_class = false; }
    } else if (c == '[') {
      in_class = true;
      result += c;
      // A ']' directly after '[' or '[^' is a literal member of the class in both engines.
      if (i + 1 < pattern.size() && pattern[i + 1] == '^') { result += pattern[++i]; }
      if (i + 1 < pattern.size() && pattern[i + 1] == ']') { result += pattern[++i]; }
    } else if (c == '$') {
      result += "\\Z";
    } else {
      result += c;
    }
  }
  return result;
}

}  // namespace regex
}  // namespace sirius
