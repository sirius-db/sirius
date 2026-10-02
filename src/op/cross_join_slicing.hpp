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

#include "data/data_batch_utils.hpp"
#include "memory/size_arithmetic.hpp"

#include <cudf/types.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

namespace sirius::op {

//! Bytes of the cross join of @p left and @p right: every left column once per right row and
//! every right column once per left row. Zero when a row count is unknown.
[[nodiscard]] inline std::size_t cross_join_output_bytes(batch_rows_and_bytes const& left,
                                                         batch_rows_and_bytes const& right) noexcept
{
  if (!left.rows || !right.rows) { return 0; }
  return memory::saturating_add(memory::saturating_mul(left.bytes, *right.rows),
                                memory::saturating_mul(right.bytes, *left.rows));
}

//! Number of tasks the cross join of @p left and @p right is split into, over equal row ranges of
//! the left batch. Each task's output stays near @p task_bytes and within the row limit of a cuDF
//! column. One task when a row count is unknown, at most one task per left row.
[[nodiscard]] inline std::size_t cross_join_num_slices(batch_rows_and_bytes const& left,
                                                       batch_rows_and_bytes const& right,
                                                       uint64_t task_bytes) noexcept
{
  if (!left.rows || !right.rows || *left.rows == 0) { return 1; }
  auto const ceil_div = [](std::size_t a, std::size_t b) { return a / b + (a % b != 0 ? 1 : 0); };
  constexpr auto max_rows = static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max());
  // A slice holds whole left rows, so the row limit caps the left rows per slice.
  auto const left_rows_per_slice =
    *right.rows == 0 ? *left.rows : std::max<std::size_t>(max_rows / *right.rows, 1);
  auto num_slices = ceil_div(*left.rows, left_rows_per_slice);
  if (task_bytes > 0) {
    num_slices = std::max(num_slices, ceil_div(cross_join_output_bytes(left, right), task_bytes));
  }
  return std::clamp<std::size_t>(num_slices, 1, *left.rows);
}

}  // namespace sirius::op
