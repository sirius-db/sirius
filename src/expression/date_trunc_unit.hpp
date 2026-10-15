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

#include <cstdint>
#include <optional>
#include <string_view>

namespace sirius {

enum class date_trunc_unit : uint8_t { day, hour, minute, second, millisecond, microsecond };

/// Shared capability check for planning and evaluation. Only canonical spellings are supported;
/// DuckDB's case-insensitive names and aliases currently fall back to CPU.
[[nodiscard]] constexpr std::optional<date_trunc_unit> parse_gpu_date_trunc_unit(
  std::string_view unit) noexcept
{
  if (unit == "day") { return date_trunc_unit::day; }
  if (unit == "hour") { return date_trunc_unit::hour; }
  if (unit == "minute") { return date_trunc_unit::minute; }
  if (unit == "second") { return date_trunc_unit::second; }
  if (unit == "millisecond") { return date_trunc_unit::millisecond; }
  if (unit == "microsecond") { return date_trunc_unit::microsecond; }
  return std::nullopt;
}

}  // namespace sirius
