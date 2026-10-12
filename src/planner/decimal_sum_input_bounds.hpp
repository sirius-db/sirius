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
#include <vector>

namespace duckdb {
class ClientContext;
class LogicalAggregate;
}  // namespace duckdb

namespace sirius::planner {

/**
 * @brief Plan-time magnitude bounds for the decimal SUM inputs of @p op, one per aggregate
 *
 * For SUM and AVG over a DECIMAL input the GPU holds as DECIMAL32 or DECIMAL64, the largest
 * |unscaled value| the input can take, proven from base-table statistics; nullopt for every
 * other aggregate and whenever no proof is available. The aggregate operators use a bound to
 * decide widening from the batch row count instead of measuring the batch
 * (see op::decimal_sums_to_widen).
 *
 * @pre Call before the child is planned: planning moves expressions out of the logical subtree.
 */
[[nodiscard]] std::vector<std::optional<std::uint64_t>> resolve_decimal_sum_input_bounds(
  duckdb::ClientContext& context, duckdb::LogicalAggregate const& op);

}  // namespace sirius::planner
