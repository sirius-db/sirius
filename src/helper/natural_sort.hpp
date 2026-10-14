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

#include <cctype>
#include <cstddef>
#include <string_view>

namespace sirius {

/**
 * @brief Orders strings the way people number files: runs of digits compare by numeric value, and
 * everything else compares character by character.
 *
 * So `part.2.parquet` sorts before `part.10.parquet`. Leading zeros do not change a digit run's
 * value. Strings that are equal under that rule (e.g. `part.07` and `part.7`) fall back to plain
 * lexicographic order, so the result is a strict weak ordering and sorting is deterministic.
 */
inline bool natural_less(std::string_view lhs, std::string_view rhs)
{
  auto const is_digit = [](char c) { return std::isdigit(static_cast<unsigned char>(c)) != 0; };
  std::size_t lhs_idx = 0;
  std::size_t rhs_idx = 0;
  while (lhs_idx < lhs.size() && rhs_idx < rhs.size()) {
    if (is_digit(lhs[lhs_idx]) && is_digit(rhs[rhs_idx])) {
      std::size_t lhs_end = lhs_idx;
      std::size_t rhs_end = rhs_idx;
      while (lhs_end < lhs.size() && is_digit(lhs[lhs_end])) {
        ++lhs_end;
      }
      while (rhs_end < rhs.size() && is_digit(rhs[rhs_end])) {
        ++rhs_end;
      }
      // Skip leading zeros, keeping at least one digit, then the shorter run is the smaller value.
      std::size_t lhs_start = lhs_idx;
      std::size_t rhs_start = rhs_idx;
      while (lhs_start + 1 < lhs_end && lhs[lhs_start] == '0') {
        ++lhs_start;
      }
      while (rhs_start + 1 < rhs_end && rhs[rhs_start] == '0') {
        ++rhs_start;
      }
      std::string_view const lhs_digits = lhs.substr(lhs_start, lhs_end - lhs_start);
      std::string_view const rhs_digits = rhs.substr(rhs_start, rhs_end - rhs_start);
      if (lhs_digits.size() != rhs_digits.size()) { return lhs_digits.size() < rhs_digits.size(); }
      if (lhs_digits != rhs_digits) { return lhs_digits < rhs_digits; }
      lhs_idx = lhs_end;
      rhs_idx = rhs_end;
    } else {
      if (lhs[lhs_idx] != rhs[rhs_idx]) { return lhs[lhs_idx] < rhs[rhs_idx]; }
      ++lhs_idx;
      ++rhs_idx;
    }
  }
  bool const lhs_done = lhs_idx == lhs.size();
  bool const rhs_done = rhs_idx == rhs.size();
  if (lhs_done != rhs_done) { return lhs_done; }
  return lhs < rhs;
}

}  // namespace sirius
