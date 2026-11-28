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

#include <cudf/types.hpp>

#include <algorithm>
#include <cstddef>
#include <limits>

namespace sirius::op {

inline constexpr std::size_t max_sort_rows_per_range =
  static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max()) / 2;

inline std::size_t choose_sort_range_count(std::size_t total_bytes,
                                           std::size_t total_rows,
                                           std::size_t target_bytes,
                                           std::size_t sample_rows)
{
  auto const ceil_div = [](std::size_t count, std::size_t divisor) {
    return count == 0 ? std::size_t{0} : 1 + (count - 1) / divisor;
  };
  auto const by_bytes = target_bytes == 0 ? std::size_t{1} : ceil_div(total_bytes, target_bytes);
  auto const by_rows  = ceil_div(total_rows, max_sort_rows_per_range);
  return std::min(std::max<std::size_t>(1, std::max(by_bytes, by_rows)),
                  std::max<std::size_t>(1, sample_rows));
}

}  // namespace sirius::op
