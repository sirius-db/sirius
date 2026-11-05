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

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>

namespace sirius {

/// Character bounds for cudf::strings::slice_strings with step 1. Negative positions count from
/// the end of each string; an absent stop means the end of the string.
struct substring_slice {
  std::optional<int32_t> start;
  std::optional<int32_t> stop;
};

/// Length DuckDB applies to two-argument substring(string, offset).
inline constexpr int64_t substring_default_length = std::numeric_limits<uint32_t>::max();

/// Shared capability check for planning and evaluation. Maps DuckDB's
/// substring(string, offset, length) with constant offset and length to one slice that matches
/// DuckDB on every row. Returns nullopt when no such slice exists: arguments DuckDB rejects as out
/// of range, and windows whose result on strings shorter than the offset depends on whether DuckDB
/// picked its ASCII or Unicode kernel (positive offset with negative length, and negative offset
/// with a length shorter than the offset).
[[nodiscard]] constexpr std::optional<substring_slice> gpu_substring_slice(int64_t offset,
                                                                           int64_t length) noexcept
{
  constexpr int64_t upper = std::numeric_limits<uint32_t>::max();
  constexpr int64_t lower = -upper - 1;
  if (offset < lower || offset > upper || length < lower || length > upper) { return std::nullopt; }
  // Every string has at most INT32_MAX characters, so clamping preserves the result.
  auto const to_position = [](int64_t pos) {
    return static_cast<int32_t>(std::clamp<int64_t>(
      pos, std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max()));
  };

  if (length == 0) { return substring_slice{0, 0}; }
  if (offset >= 0) {
    if (length < 0) {
      if (offset == 0) { return substring_slice{0, 0}; }
      return std::nullopt;
    }
    // Offsets are 1-based; offset 0 starts one character before the string.
    return substring_slice{to_position(std::max<int64_t>(offset - 1, 0)),
                           to_position(offset - 1 + length)};
  }
  // Negative offsets count back from the end of the string.
  if (length < 0) { return substring_slice{to_position(offset + length), to_position(offset)}; }
  if (length < -offset) { return std::nullopt; }
  return substring_slice{to_position(offset), std::nullopt};
}

}  // namespace sirius
