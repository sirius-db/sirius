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
 * @brief Plan-time magnitude bounds for the SUM inputs of @p op, one per aggregate
 *
 * For SUM and AVG over a DECIMAL input, and for a wide integer SUM or AVG over BIGINT
 * (op::is_wide_integer_sum), the largest |unscaled value| the input can take, proven from
 * base-table statistics; nullopt for every other aggregate, for a bound beyond 64 bits, and
 * whenever no proof is available. The aggregate operators use a bound to decide from the batch row
 * count instead of measuring the batch whether a DECIMAL32/64 or BIGINT SUM must be widened
 * (see op::sums_to_widen) or a DECIMAL128 SUM can be narrowed (see op::decimal_sums_to_narrow).
 *
 * @pre Call before the child is planned: planning moves expressions out of the logical subtree.
 */
[[nodiscard]] std::vector<std::optional<std::uint64_t>> resolve_sum_input_bounds(
  duckdb::ClientContext& context, duckdb::LogicalAggregate const& op);

}  // namespace sirius::planner
