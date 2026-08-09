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

#include <cstddef>
#include <limits>
#include <optional>

namespace sirius::planner {

/**
 * @brief Checked Top-N candidate count `K = limit + offset` for threshold-refinement eligibility
 *
 * Implements the admission arithmetic of `docs/super-sirius/dynamic-filters-top-n.md`, "Limit and
 * offset": disengaged when `limit` is zero, when the sum would wrap, or when K exceeds the
 * `cudf::size_type` row index space -- K is never wrapped or truncated. `sirius_plan_top_n.cpp`
 * consumes this before constructing a `top_n_threshold_coordinator`.
 */
[[nodiscard]] constexpr std::optional<std::size_t> checked_top_n_k(std::size_t limit,
                                                                   std::size_t offset) noexcept
{
  if (limit == 0) { return std::nullopt; }
  if (limit > std::numeric_limits<std::size_t>::max() - offset) { return std::nullopt; }
  auto const k = limit + offset;
  if (k > static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max())) {
    return std::nullopt;
  }
  return k;
}

}  // namespace sirius::planner
