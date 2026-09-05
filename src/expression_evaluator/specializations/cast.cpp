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
#include <expression_evaluator/expression_evaluator.hpp>
#include <expression_evaluator/round_to_scale.hpp>
#include <helper/logical_type.hpp>
#include <helper/numeric_narrowing.hpp>
#include <sirius/exception.hpp>

// cudf
#include <cudf/cudf_utils.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/traits.hpp>

// standard library
#include <algorithm>

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

/// A semantic cast with DuckDB's conversion rules. cuDF's floating -> fixed_point conversion
/// truncates toward zero, DuckDB's rounds half away from zero after scaling (`1 - 0.07` in FP64
/// is 0.9299999999999999: DuckDB casts it to DECIMAL(16,2) 0.93, cuDF to 0.92, and every
/// discounted TPC-H revenue then differs by up to one cent per row). Round the floating column
/// to the target scale first, with DuckDB's arithmetic, so the cast that follows is exact:
/// cuDF converts a double that is the nearest representation of an s-digit decimal to exactly
/// that decimal.
std::unique_ptr<cudf::column> cast_like_duckdb(cudf::column_view const& input,
                                               cudf::data_type target,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  if (!cudf::is_floating_point(input.type()) || !cudf::is_fixed_point(target)) {
    return cudf::cast(input, target, stream, mr);
  }
  auto rounded = sirius::round_to_scale_like_duckdb(input, -target.scale(), stream, mr);
  return cudf::cast(rounded->view(), target, stream, mr);
}

}  // namespace

evaluate_result expression_evaluator::evaluate(sirius::ast::cast const& alt, evaluation_mode mode)
{
  auto const ast_supported =
    std::find(supported_ast_cast_types_native.begin(),
              supported_ast_cast_types_native.end(),
              alt.target_type.id()) != supported_ast_cast_types_native.end();

  auto const ast_op_count = alt.cudf_ast_op_count();

  // Carrier restores must reach the materialized branch, the only path authorized to use the
  // physical representation tunnel. Semantic casts may use the cuDF AST path.
  if (ast_supported && alt.kind == sirius::ast::cast_kind::semantic &&
      _strategy != expression_evaluator_strategy::MATERIALIZE &&
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
  auto const return_type = sirius::get_cudf_type(alt.target_type);
  auto child             = evaluate(*alt.child, evaluation_mode::MATERIALIZE);
  D_ASSERT(!child.is_scalar());  // CAST should never be called on a scalar
  // Only planner-certified carrier restoration may tunnel through the narrowed representation.
  // A semantic cast delegates to cuDF and is never reinterpreted as a physical DATE restore.
  auto result_column =
    alt.kind == sirius::ast::cast_kind::carrier_restore
      ? sirius::cast_through_rep(child.get_column_view(), return_type, _stream, _mr)
      : cast_like_duckdb(child.get_column_view(), return_type, _stream, _mr);
  if (mode == evaluation_mode::AST) {
    // The parent is executing in AST mode, so add the materialized result to the AST tree.
    return materialize_as_ast_column(std::move(result_column));
  }
  return evaluate_result(std::move(result_column));
}

}  // namespace sirius
