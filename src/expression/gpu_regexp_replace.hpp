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

#include <cudf/strings/regex/flags.hpp>
#include <cudf/types.hpp>

#include <optional>
#include <string>
#include <string_view>

namespace sirius {

/// One regexp_replace call expressed as cuDF calls that reproduce DuckDB's RE2 semantics.
struct gpu_regexp_replace {
  /// cuDF regex matching exactly what the RE2 pattern matches
  std::string pattern;
  cudf::strings::regex_flags flags{cudf::strings::regex_flags::DEFAULT};
  /// Literal replacement for cudf::strings::replace_re, or a `${n}` template for
  /// cudf::strings::replace_with_backrefs when `backrefs` is set
  std::string replacement;
  bool backrefs{false};
  /// For replace_re. A backreference pattern matches at most once per row unless DuckDB's
  /// options ask for every match, so replace_with_backrefs needs no count.
  std::optional<cudf::size_type> max_replace_count;
};

/// Shared capability check for planning and evaluation. Maps DuckDB's
/// regexp_replace(string, pattern, replacement[, options]) with constant pattern, replacement and
/// options to cuDF calls that produce DuckDB's result on every row: the first match only unless
/// the options contain `g`, the options `c`, `i`, `l`, `m`, `n`, `p` and `s`, and RE2's
/// replacement syntax (`\0` to `\9`, `\\`).
///
/// Returns nullopt for anything cuDF would not reproduce: RE2 syntax outside a subset whose cuDF
/// meaning is known to match (Unicode and POSIX classes, `\b`, inline flags, named groups, ...),
/// patterns that match only the empty string, patterns that can match the empty string under `g`,
/// replacements RE2 rejects or that reference a missing group, and patterns cuDF cannot compile.
[[nodiscard]] std::optional<gpu_regexp_replace> translate_gpu_regexp_replace(
  std::string_view pattern, std::string_view replacement, std::string_view options);

}  // namespace sirius
