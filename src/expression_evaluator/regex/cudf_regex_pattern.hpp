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

#include <string>
#include <string_view>

namespace sirius {
namespace regex {

/// Rewrites a RE2 pattern for cuDF's regex engine. RE2's `$` matches only at the end of the
/// text by default, whereas cuDF's `$` also matches before a trailing newline. cuDF's `\Z`
/// has RE2's end-of-text semantics, so every anchoring `$` is rewritten to `\Z`. Escaped
/// dollars and dollars inside bracket classes are literals for both engines and stay as-is.
inline std::string to_cudf_regex_pattern(std::string_view pattern)
{
  std::string result;
  result.reserve(pattern.size() + 8);
  bool in_class = false;
  for (std::size_t i = 0; i < pattern.size(); ++i) {
    char const c = pattern[i];
    if (c == '\\') {
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
