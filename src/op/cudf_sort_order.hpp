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

#include "duckdb/common/enums/order_type.hpp"

#include <cudf/types.hpp>

namespace sirius {
namespace op {

/// Map a DuckDB ORDER BY direction to a cuDF sort order.
inline cudf::order to_cudf_order(duckdb::OrderType order_type)
{
  return order_type == duckdb::OrderType::DESCENDING ? cudf::order::DESCENDING
                                                     : cudf::order::ASCENDING;
}

/// Map a DuckDB NULLS FIRST/LAST to a cuDF null_order, honoring SQL semantics for both directions.
///
/// Single source of truth for the cuDF DESC-null contract: cuDF applies null_order in the ascending
/// frame and then *reverses* it for a DESCENDING column, so the naive NULLS_FIRST?BEFORE:AFTER
/// mapping mis-places NULLs for descending keys (e.g. `ORDER BY x DESC NULLS LAST` would put NULLs
/// first). We compensate by flipping BEFORE<->AFTER for a descending key, so NULLS FIRST/LAST is
/// honored as an absolute position matching DuckDB CPU. Used by every sort/top-n/window path that
/// feeds null_precedence to cuDF sorting.
inline cudf::null_order to_cudf_null_order(duckdb::OrderType order_type,
                                           duckdb::OrderByNullType null_type)
{
  cudf::null_order result = (null_type == duckdb::OrderByNullType::NULLS_FIRST)
                              ? cudf::null_order::BEFORE
                              : cudf::null_order::AFTER;
  if (order_type == duckdb::OrderType::DESCENDING) {
    result =
      (result == cudf::null_order::BEFORE) ? cudf::null_order::AFTER : cudf::null_order::BEFORE;
  }
  return result;
}

}  // namespace op
}  // namespace sirius
