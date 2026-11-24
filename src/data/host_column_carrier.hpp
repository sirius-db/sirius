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

#include <cucascade/memory/column_metadata.hpp>

namespace sirius {

/// Actual cuDF carrier of one column of a host or disk table, rebuilt from its column metadata.
/// Keyed on the DECIMAL type ids, not on a nonzero scale: a DECIMAL(p,0) column has cuDF scale 0
/// and must still take the two-argument fixed-point constructor.
[[nodiscard]] inline cudf::data_type host_column_carrier(
  cucascade::memory::column_metadata const& meta)
{
  auto const id         = static_cast<cudf::type_id>(meta.type_id);
  auto const is_decimal = id == cudf::type_id::DECIMAL32 || id == cudf::type_id::DECIMAL64 ||
                          id == cudf::type_id::DECIMAL128;
  return is_decimal ? cudf::data_type{id, meta.scale} : cudf::data_type{id};
}

}  // namespace sirius
