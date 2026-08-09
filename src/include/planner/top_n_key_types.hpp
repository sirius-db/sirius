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

/**
 * @file top_n_key_types.hpp
 * @brief The Top-N per-key type allowlist, as a pure function of a DuckDB type
 *
 * Separated from `sirius_plan_top_n.cpp` so the admission rule can be asserted directly, band by
 * band. Reachability differs sharply by scan format and is easy to get wrong: the duckdb-native
 * scan refuses a `DECIMAL128` column before planning, but the **parquet path has no precision
 * gate**, so `p >= 19` reaches producer admission there and its refusal is live in production. A
 * `DECIMAL(4,2)` column is genuinely unreachable -- `sirius::get_cudf_type` cannot map it at all.
 * Asserting the rule here covers both cases without depending on which scan a test happens to
 * build.
 */

#pragma once

#include "helper/logical_type.hpp"

#include <duckdb/common/types.hpp>

#include <cudf/types.hpp>

#include <cstdint>
#include <optional>

namespace sirius::planner {

/**
 * @brief Exact cuDF storage type for an admitted ORDER BY key, or empty when the type is outside
 * the allowlist (main doc, "Range and lexicographic filters")
 *
 * A type is admitted only when DuckDB SQL ordering, cuDF comparison, exact host extraction, device
 * scalar construction, and the comparison kernel all agree on one physical representation.
 *
 * `DECIMAL(p,s)` is admitted as the fixed-point type its precision selects, carrying cuDF's negated
 * scale, for exactly the two bands whose scaled integer the boundary path holds exactly:
 *
 * - `p <= 4` is INT16 in DuckDB and has no cuDF fixed-point counterpart, so it is refused.
 * - `p <= 9` is `DECIMAL32` and `p <= 18` is `DECIMAL64`: the scaled integer is precisely an
 *   `std::int32_t` or `std::int64_t`, which are alternatives `exact_host_scalar` already holds --
 *   no new variant alternative and no rescaling is involved.
 * - `p >= 19` is `DECIMAL128`, whose scaled integer needs 16 bytes. `boundary_filter_params` widens
 *   every component to `std::int64_t` and the kernel reads by width 1/2/4/8, so a 16-byte component
 *   would be read as garbage rather than rejected. It is refused until those params are widened.
 *
 * The scale rides in the returned `cudf::data_type`, and comparing scaled integers is sound only at
 * equal scale -- `boundary_key_matches_site_type` is what refuses a site that does not match.
 */
[[nodiscard]] inline std::optional<cudf::data_type> admitted_top_n_key_storage_type(
  duckdb::LogicalType const& type)
{
  switch (type.id()) {
    case duckdb::LogicalTypeId::TINYINT: return cudf::data_type{cudf::type_id::INT8};
    case duckdb::LogicalTypeId::SMALLINT: return cudf::data_type{cudf::type_id::INT16};
    case duckdb::LogicalTypeId::INTEGER: return cudf::data_type{cudf::type_id::INT32};
    case duckdb::LogicalTypeId::BIGINT: return cudf::data_type{cudf::type_id::INT64};
    case duckdb::LogicalTypeId::DATE: return cudf::data_type{cudf::type_id::TIMESTAMP_DAYS};
    case duckdb::LogicalTypeId::DECIMAL: {
      auto const precision = duckdb::DecimalType::GetWidth(type);
      auto const scale     = -static_cast<std::int32_t>(duckdb::DecimalType::GetScale(type));
      if (precision <= sirius::logical_type::decimal_max_precision_int16) { return std::nullopt; }
      if (precision <= sirius::logical_type::decimal_max_precision_int32) {
        return cudf::data_type{cudf::type_id::DECIMAL32, scale};
      }
      if (precision <= sirius::logical_type::decimal_max_precision_int64) {
        return cudf::data_type{cudf::type_id::DECIMAL64, scale};
      }
      return std::nullopt;
    }
    default: return std::nullopt;
  }
}

}  // namespace sirius::planner
