/*
 * Copyright 2025, Sirius Contributors.
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

// sirius
#include <expression/ast/node.hpp>
#include <expression_evaluator/ast_supported_types.hpp>
#include <expression_evaluator/date_cast.hpp>
#include <expression_evaluator/expression_evaluator.hpp>
#include <helper/logical_type.hpp>
#include <helper/numeric_narrowing.hpp>
#include <helper/timestamp_semantics.hpp>
#include <sirius/exception.hpp>

// cudf
#include <cudf/column/column_factories.hpp>
#include <cudf/cudf_utils.hpp>
#include <cudf/strings/convert/convert_fixed_point.hpp>
#include <cudf/strings/convert/convert_integers.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/traits.hpp>

// standard library
#include <algorithm>
#include <limits>

namespace sirius {
using evaluate_result = expression_evaluator::evaluate_result;

namespace {

cudf::ast::ast_operator cast_op_to_ast(sirius::type_id id)
{
  switch (id) {
    case sirius::type_id::UBIGINT: return cudf::ast::ast_operator::CAST_TO_UINT64;
    case sirius::type_id::BIGINT: return cudf::ast::ast_operator::CAST_TO_INT64;
    case sirius::type_id::DOUBLE: return cudf::ast::ast_operator::CAST_TO_FLOAT64;
    default:
      throw invalid_input_exception(
        "[cast_op_to_ast] unsupported CAST target type id={}; cuDF AST supports "
        "UBIGINT, BIGINT, DOUBLE.",
        static_cast<int>(id));
  }
}

// Largest power of ten an int128 holds is 10^38.
constexpr __int128_t power_of_ten(int32_t exponent)
{
  __int128_t result = 1;
  for (int32_t i = 0; i < exponent; ++i) {
    result *= 10;
  }
  return result;
}

// cuDF narrows a DECIMAL128 HUGEINT carrier by wrapping, where DuckDB's CAST raises an error and
// TRY_CAST returns NULL. Throw when any value lies outside @p target's SQL range so the query
// falls back to DuckDB, which gives its exact behavior. Wider targets need no check.
void require_hugeint_cast_in_range(cudf::column_view const& input,
                                   logical_type const& target,
                                   ::cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr)
{
  __int128_t minimum = 0;
  __int128_t maximum = 0;
  switch (target.id()) {
    case type_id::TINYINT:
      minimum = std::numeric_limits<int8_t>::min();
      maximum = std::numeric_limits<int8_t>::max();
      break;
    case type_id::SMALLINT:
      minimum = std::numeric_limits<int16_t>::min();
      maximum = std::numeric_limits<int16_t>::max();
      break;
    case type_id::INTEGER:
      minimum = std::numeric_limits<int32_t>::min();
      maximum = std::numeric_limits<int32_t>::max();
      break;
    case type_id::BIGINT:
      minimum = std::numeric_limits<int64_t>::min();
      maximum = std::numeric_limits<int64_t>::max();
      break;
    case type_id::UTINYINT: maximum = std::numeric_limits<uint8_t>::max(); break;
    case type_id::USMALLINT: maximum = std::numeric_limits<uint16_t>::max(); break;
    case type_id::UINTEGER: maximum = std::numeric_limits<uint32_t>::max(); break;
    // UHUGEINT is carried as UINT64.
    case type_id::UBIGINT:
    case type_id::UHUGEINT: maximum = std::numeric_limits<uint64_t>::max(); break;
    case type_id::DECIMAL: {
      // |v| < 10^(p-s) also keeps the rescaled value within the target's p digits.
      auto const digits = target.decimal_precision() - target.decimal_scale();
      if (digits >= 38) { return; }
      maximum = power_of_ten(digits) - 1;
      minimum = -maximum;
      break;
    }
    default: return;
  }
  auto const range = compute_exact_numeric_range(input, stream, mr);
  // Empty and all-NULL columns have no values to overflow.
  if (!range) { return; }
  if (range->minimum < minimum || range->maximum > maximum) {
    throw not_implemented_exception(
      "[expression_evaluator:cast] HUGEINT value out of range for {}; falling back to DuckDB",
      target.to_string());
  }
}

}  // namespace

evaluate_result expression_evaluator::evaluate(sirius::ast::cast const& alt, evaluation_mode mode)
{
  auto const ast_supported = ast_supports_cast(alt.child->return_type(), alt.target_type);

  auto const ast_op_count = alt.cudf_ast_op_count();

  // Carrier restores must reach the materialized branch, the only path authorized to use the
  // physical representation tunnel. Semantic casts may use the cuDF AST path.
  if (ast_supported && alt.kind == sirius::ast::cast_kind::semantic &&
      _policy.strategy != expression_evaluator_strategy::MATERIALIZE &&
      (mode == evaluation_mode::AST || ast_op_count >= _min_ast_size)) {
    auto child            = evaluate(*alt.child, evaluation_mode::AST);
    auto const& cast_expr = _ast_tree.emplace<cudf::ast::operation>(
      cast_op_to_ast(alt.target_type.id()), child.get_expr());

    if (mode == evaluation_mode::AST) {
      //===----------1: AST Mode----------===//
      return evaluate_result(compose(cast_expr, {&child}));
    }
    //===----------2: MATERIALIZE Mode, evaluate node with AST----------===//
    auto result_column = evaluate_ast(cast_expr);

    release_temporaries({&child});
    return evaluate_result(std::move(result_column));
  }

  //===----------3: MATERIALIZE Mode, evaluate node with unary/binary ops----------===//
  auto const return_type = sirius::require_cudf_type(alt.target_type);
  auto child             = evaluate(*alt.child, evaluation_mode::MATERIALIZE);
  if (child.is_scalar()) {
    child = evaluate_result(
      cudf::make_column_from_scalar(child.get_scalar(), _input_table.num_rows(), _stream, _mr));
  }
  auto const input = child.get_column_view();
  std::unique_ptr<cudf::column> result_column;
  if (alt.kind == sirius::ast::cast_kind::carrier_restore) {
    // Only planner-certified carrier restoration may tunnel through the representation.
    result_column = sirius::cast_through_rep(input, return_type, _stream, _mr);
  } else if (return_type.id() == cudf::type_id::STRING &&
             cudf::is_integral_not_bool(input.type())) {
    result_column = cudf::strings::from_integers(input, _stream, _mr);
  } else if (return_type.id() == cudf::type_id::STRING &&
             alt.child->return_type().id() == type_id::HUGEINT) {
    // HUGEINT's carrier is DECIMAL128 with scale 0, which formats as a plain integer.
    result_column = cudf::strings::from_fixed_point(input, _stream, _mr);
  } else if (input.type().id() == cudf::type_id::TIMESTAMP_DAYS &&
             (return_type.id() == cudf::type_id::TIMESTAMP_SECONDS ||
              return_type.id() == cudf::type_id::TIMESTAMP_MILLISECONDS ||
              return_type.id() == cudf::type_id::TIMESTAMP_MICROSECONDS ||
              return_type.id() == cudf::type_id::TIMESTAMP_NANOSECONDS)) {
    result_column = cast_date_to_timestamp(input, return_type, alt.try_cast, _stream, _mr);
  } else if ((input.type().id() == cudf::type_id::TIMESTAMP_NANOSECONDS ||
              input.type().id() == cudf::type_id::TIMESTAMP_MILLISECONDS ||
              input.type().id() == cudf::type_id::TIMESTAMP_SECONDS) &&
             return_type.id() == cudf::type_id::TIMESTAMP_MICROSECONDS) {
    // Carrier restores were handled above; only semantic casts reach this branch.
    result_column = temporal::cast_to_microseconds_checked(input, _stream, _mr);
  } else {
    if (alt.child->return_type().id() == type_id::HUGEINT) {
      require_hugeint_cast_in_range(input, alt.target_type, _stream, _mr);
    }
    result_column = cudf::cast(input, return_type, _stream, _mr);
  }
  if (mode == evaluation_mode::AST) {
    // The parent is executing in AST mode, so add the materialized result to the AST tree.
    return materialize_as_ast_column(std::move(result_column));
  }
  return evaluate_result(std::move(result_column));
}

}  // namespace sirius
