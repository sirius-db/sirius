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
 * band. Reachability differs by scan format: the duckdb-native scan refuses a `DECIMAL128`
 * *column* at decode viability, but the parquet path has no precision gate, so `p >= 19` columns
 * reach producer admission there and now admit -- and an aggregate-output `DECIMAL128` key (TPC-H
 * Q3/Q10's `revenue`) admits on every scan format, because the sink consumes its own boundary. A
 * `DECIMAL(4,2)` column is genuinely unreachable -- `sirius::get_cudf_type` cannot map it at all.
 * Asserting the rule here covers every band without depending on which scan a test happens to
 * build.
 */

#pragma once

#include "cudf/cudf_utils.hpp"

#include <cudf/types.hpp>

#include <duckdb/common/types.hpp>

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
 * `DECIMAL(p,s)` is admitted as the fixed-point type its precision selects, carrying cuDF's
 * negated scale, through the single banding derivation `sirius::cudf_decimal_type` -- the same one
 * `sirius::get_cudf_type` executes with, so admission and execution cannot drift. The scaled
 * integer at every band is exactly an alternative `exact_host_scalar` holds (`std::int32_t`,
 * `std::int64_t`, or `__int128_t`), with no rescaling anywhere. Only `p <= 4` is refused: it is
 * INT16 in DuckDB and has no cuDF fixed-point counterpart. (`p >= 19` was refused until
 * `exact_host_scalar::widened()` and the kernel's width-16 load landed; a 16-byte component would
 * previously have been read as garbage rather than rejected.)
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
    case duckdb::LogicalTypeId::DECIMAL:
      return sirius::cudf_decimal_type(duckdb::DecimalType::GetWidth(type),
                                       duckdb::DecimalType::GetScale(type));
    default: return std::nullopt;
  }
}

}  // namespace sirius::planner
