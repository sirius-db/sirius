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

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace sirius::compression {

/**
 * @brief Select the plan blocks for a pinned column subset.
 *
 * A whole-table plan DSL has one "---"-separated block per full-table column, in
 * schema order. When a pin caches only some columns, @p column_indices gives the
 * full-table index of each pinned column (in the pinned/materialized order); this
 * returns a DSL with just those blocks, in that order, so it lines up 1:1 with the
 * pinned table for compress_with_plan. Returns nullopt if any index is out of
 * range (the plan does not cover a pinned column), so the caller pins uncompressed.
 */
[[nodiscard]] std::optional<std::string> select_plan_blocks(
  const std::string& full_plan_dsl, const std::vector<std::size_t>& column_indices);

}  // namespace sirius::compression
