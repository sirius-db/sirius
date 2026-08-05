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

#include "expression/ast/constant_range.hpp"

#include "cudf/cudf_utils.hpp"
#include "expression/ast/node.hpp"
#include "expression/value.hpp"

#include <variant>

namespace sirius::ast {

std::optional<sirius::numeric_range> constant_numeric_range(constant const& expr)
{
  auto const& payload = expr.payload;
  switch (expr.return_type().id()) {
    case type_id::TINYINT:
      if (auto const* v = std::get_if<int8_t>(&payload)) { return signed_integer_range(*v, *v); }
      return std::nullopt;
    case type_id::SMALLINT:
      if (auto const* v = std::get_if<int16_t>(&payload)) { return signed_integer_range(*v, *v); }
      return std::nullopt;
    case type_id::INTEGER:
      if (auto const* v = std::get_if<int32_t>(&payload)) { return signed_integer_range(*v, *v); }
      return std::nullopt;
    case type_id::BIGINT:
      if (auto const* v = std::get_if<int64_t>(&payload)) { return signed_integer_range(*v, *v); }
      return std::nullopt;
    case type_id::UTINYINT:
      if (auto const* v = std::get_if<uint8_t>(&payload)) { return unsigned_integer_range(*v, *v); }
      return std::nullopt;
    case type_id::USMALLINT:
      if (auto const* v = std::get_if<uint16_t>(&payload)) {
        return unsigned_integer_range(*v, *v);
      }
      return std::nullopt;
    case type_id::UINTEGER:
      if (auto const* v = std::get_if<uint32_t>(&payload)) {
        return unsigned_integer_range(*v, *v);
      }
      return std::nullopt;
    case type_id::UBIGINT:
      if (auto const* v = std::get_if<uint64_t>(&payload)) {
        return unsigned_integer_range(*v, *v);
      }
      return std::nullopt;
    case type_id::DECIMAL: {
      auto const scale = expr.return_type().decimal_scale();
      if (auto const* v = std::get_if<sirius::decimal32>(&payload)) {
        return decimal_range(v->value, v->value, scale);
      }
      if (auto const* v = std::get_if<sirius::decimal64>(&payload)) {
        return decimal_range(v->value, v->value, scale);
      }
      if (auto const* v = std::get_if<sirius::decimal128>(&payload)) {
        return decimal_range(v->value, v->value, scale);
      }
      return std::nullopt;
    }
    default: return std::nullopt;
  }
}

bool constant_representable_in_carrier(cudf::data_type carrier, constant const& expr)
{
  // A typed NULL materializes only a validity bit, so it fits any carrier.
  if (std::holds_alternative<sirius::null_value>(expr.payload)) { return true; }
  auto const range = constant_numeric_range(expr);
  return range && numeric_range_fits(carrier, *range);
}

bool narrow_domain_carrier_eligible(sirius::logical_type const& logical,
                                    cudf::data_type carrier,
                                    std::initializer_list<node const*> constant_operands)
{
  if (!sirius::is_narrowable_numeric_type(logical)) { return false; }
  auto const native = sirius::try_get_cudf_type(logical);
  if (!native || !sirius::can_restore_to(carrier, *native)) { return false; }
  for (auto const* operand : constant_operands) {
    if (!operand || !operand->holds<constant>()) { return false; }
    if (!constant_representable_in_carrier(carrier, operand->get<constant>())) { return false; }
  }
  return true;
}

}  // namespace sirius::ast
