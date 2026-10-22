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

#include "cudf/cudf_utils.hpp"
#include "expression_evaluator/expression_evaluator.hpp"
#include "helper/numeric_narrowing.hpp"
#include "sirius/exception.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/traits.hpp>

#include <memory>
#include <utility>
#include <vector>

namespace sirius {

expression_evaluator::evaluate_result expression_evaluator::evaluate(
  ast::gpu_function_call const& call, evaluation_mode mode)
{
  if (mode == evaluation_mode::AST) {
    auto result = evaluate(call, evaluation_mode::MATERIALIZE);
    return materialize_as_ast_column(result.release_column());
  }
  auto const& implementation = call.implementation();
  if (!implementation || !implementation->execute || call.arguments().empty()) {
    throw internal_exception("[gpu_scalar] invalid native function definition");
  }

  auto const rows = _input_table.num_rows();
  std::vector<evaluate_result> arguments;
  std::vector<cudf::column_view> views;
  arguments.reserve(call.arguments().size());
  views.reserve(call.arguments().size() + 1);
  for (auto const& child : call.arguments()) {
    if (!child) { throw internal_exception("[gpu_scalar] missing argument expression"); }
    auto value = evaluate(*child, evaluation_mode::MATERIALIZE);
    if (value.is_scalar()) {
      value =
        evaluate_result(cudf::make_column_from_scalar(value.get_scalar(), rows, _stream, _mr));
    }
    // Interior builtins may return cuDF's narrower result type (e.g. YEAR is
    // INT16 and LENGTH is INT32 while SQL declares BIGINT). The ordinary
    // evaluator reconciles only top-level outputs; a UDF is itself a typed
    // consumer and needs that reconciliation at each argument boundary too.
    auto const expected_argument_type = sirius::require_cudf_type(child->return_type());
    if (value.get_column_view().type() != expected_argument_type) {
      if (!cudf::is_fixed_width(value.get_column_view().type()) ||
          !cudf::is_fixed_width(expected_argument_type)) {
        throw internal_exception("[gpu_scalar] unsupported physical argument type conversion");
      }
      value = evaluate_result(
        sirius::cast_through_rep(value.get_column_view(), expected_argument_type, _stream, _mr));
    }
    views.push_back(value.get_column_view());
    arguments.push_back(std::move(value));
  }

  auto output              = implementation->execute(views, rows, _stream, _mr);
  auto const expected_type = sirius::require_cudf_type(call.return_type());
  if (!output || output->size() != rows || output->type() != expected_type) {
    throw internal_exception("[gpu_scalar] callback returned an invalid output type or row count");
  }

  if (call.propagate_nulls()) {
    // Preserve both input NULLs and additional NULLs produced by the callback.
    // This also prevents a callback that emits an all-valid column from changing
    // the CPU function's default SQL NULL contract.
    views.push_back(output->view());
    auto [mask, null_count] = cudf::bitmask_and(cudf::table_view(views), _stream, _mr);
    output->set_null_mask(std::move(mask), null_count);
    if (null_count > 0 && !cudf::is_fixed_width(output->type())) {
      output = cudf::purge_nonempty_nulls(output->view(), _stream, _mr);
    }
  }
  return evaluate_result(std::move(output));
}

}  // namespace sirius
