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
    // A DATE literal is int32 days from epoch, the same domain a narrowed DATE column carries, so
    // it folds straight into the narrow carrier and the comparison stays at the narrow width.
    case type_id::DATE:
      if (auto const* v = std::get_if<sirius::date_value>(&payload)) {
        return signed_integer_range(v->days, v->days);
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

namespace {

// True when `carrier` is a strict narrowing of `logical`'s native materialization, so the
// comparison can remain at the carrier width.
bool carried_narrow(sirius::logical_type const& logical, cudf::data_type carrier)
{
  if (!sirius::is_narrowable_numeric_type(logical)) { return false; }
  auto const native = sirius::try_get_cudf_type(logical);
  return native && sirius::can_restore_to(carrier, *native);
}

// Epoch days and plain integers share signed-integer carriers, so carrier equality alone would let
// DATE engage against a narrowed integer. Centralizing domain assignment in `narrow_domain_of`
// requires each supported logical type to choose its semantic domain explicitly.
bool same_narrow_domain(sirius::logical_type const& lhs, sirius::logical_type const& rhs)
{
  auto const domain = sirius::narrow_domain_of(lhs);
  return domain != sirius::narrow_domain::NONE && domain == sirius::narrow_domain_of(rhs);
}

// A typed NULL materializes only a validity bit, so nothing of it is ever read at the carrier's
// width and its declared type cannot be misread. Every other literal is materialized in the carrier
// and therefore has to share the column's domain.
bool literal_in_narrow_domain(sirius::logical_type const& logical, constant const& literal)
{
  return std::holds_alternative<sirius::null_value>(literal.payload) ||
         same_narrow_domain(logical, literal.return_type());
}

}  // namespace

bool narrow_domain_reference_pair_eligible(sirius::logical_type const& lhs_logical,
                                           cudf::data_type lhs_carrier,
                                           sirius::logical_type const& rhs_logical,
                                           cudf::data_type rhs_carrier)
{
  return lhs_carrier == rhs_carrier && carried_narrow(lhs_logical, lhs_carrier) &&
         carried_narrow(rhs_logical, rhs_carrier) && same_narrow_domain(lhs_logical, rhs_logical);
}

bool narrow_domain_carrier_eligible(sirius::logical_type const& logical,
                                    cudf::data_type carrier,
                                    std::initializer_list<node const*> constant_operands)
{
  if (!carried_narrow(logical, carrier)) { return false; }
  for (auto const* operand : constant_operands) {
    if (!operand || !operand->holds<constant>()) { return false; }
    auto const& literal = operand->get<constant>();
    // Preserve the domain even for ASTs constructed outside DuckDB's binder.
    if (!literal_in_narrow_domain(logical, literal)) { return false; }
    if (!constant_representable_in_carrier(carrier, literal)) { return false; }
  }
  return true;
}

}  // namespace sirius::ast
