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

// sirius
#include <cudf/cudf_utils.hpp>

#include <expression/ast/node.hpp>
#include <expression/ast/utils.hpp>
#include <expression_evaluator/expression_evaluator.hpp>
#include <helper/numeric_narrowing.hpp>
#include <sirius/exception.hpp>

// cudf
#include <cudf/copying.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

// standard library
#include <algorithm>
#include <functional>
#include <memory>
#include <vector>

namespace sirius {
using evaluate_result = expression_evaluator::evaluate_result;

evaluate_result expression_evaluator::with_selective_retry(
  std::function<evaluate_result()> const& eager,
  std::function<std::unique_ptr<cudf::column>()> const& selective)
{
  auto const first_temp_scalar = _temp_scalars.size();
  auto const first_temp_column = _temp_columns.size();
  try {
    return eager();
  } catch (invalid_input_exception const&) {
    // A zero-column input cannot carry the row count of a selection.
    if (_input_table.num_columns() == 0) { throw; }
  } catch (not_implemented_exception const&) {
    // Unsupported shapes throw this too, and fail again in the selective pass.
    if (_input_table.num_columns() == 0) { throw; }
  }

  // Release what the failed pass left. AST nodes built before it address temporaries by index, so
  // the slots stay.
  for (auto i = first_temp_scalar; i < _temp_scalars.size(); ++i) {
    _temp_scalars[i].reset();
  }
  for (auto i = first_temp_column; i < _temp_columns.size(); ++i) {
    _temp_columns[i].reset();
  }
  std::erase_if(_restored_reference_cache, [first_temp_column](auto const& entry) {
    return entry.temp_column_index >= first_temp_column;
  });

  ++_selective_evaluation_count;
  return evaluate_result(selective());
}

std::unique_ptr<cudf::column> expression_evaluator::evaluate_on_selected_rows(
  sirius::ast::node const& expr, cudf::column_view const* selection, cudf::data_type type)
{
  cudf::table_view input = _input_table;
  std::unique_ptr<cudf::table> compacted;
  if (selection != nullptr) {
    std::vector<cudf::size_type> referenced;
    sirius::ast::visit_references(expr, [&referenced](sirius::ast::reference const& ref) {
      referenced.push_back(static_cast<cudf::size_type>(ref.column_index));
    });
    std::ranges::sort(referenced);
    referenced.erase(std::unique(referenced.begin(), referenced.end()), referenced.end());

    // Compact only the referenced columns, plus the selection itself: its compacted copy has the
    // selected row count and fills the slots @p expr does not read.
    std::vector<cudf::column_view> to_compact;
    to_compact.reserve(referenced.size() + 1);
    for (auto const index : referenced) {
      to_compact.push_back(_input_table.column(index));
    }
    to_compact.push_back(*selection);
    compacted =
      sirius::apply_retention_mask(cudf::table_view{to_compact}, *selection, _stream, _mr);

    std::vector<cudf::column_view> slots(_input_table.num_columns(),
                                         compacted->get_column(referenced.size()).view());
    for (std::size_t i = 0; i < referenced.size(); ++i) {
      slots[referenced[i]] = compacted->get_column(i).view();
    }
    input = cudf::table_view{slots};
  }

  // A child evaluator keeps this evaluator's AST tree, temporaries and caches, which describe the
  // full input, intact. Its evaluate() also restores the declared type of @p expr.
  expression_evaluator child(
    &expr, _mr, _stream, _policy, _min_ast_size, _like_swar_fastpath, _like_cache);
  auto result = std::move(child.evaluate(input)->release()[0]);
  _like_shared_cache_lookup_count += child._like_shared_cache_lookup_count;
  _ast_jit_evaluation_count += child._ast_jit_evaluation_count;
  _selective_evaluation_count += child._selective_evaluation_count;
  _restored_reference_cast_count += child._restored_reference_cast_count;
  _narrow_domain_comparison_count += child._narrow_domain_comparison_count;

  if (result->type() == type) { return result; }
  if (cudf::is_fixed_width(result->type()) && cudf::is_fixed_width(type)) {
    return sirius::cast_through_rep(result->view(), type, _stream, _mr);
  }
  throw internal_exception("[expression_evaluator] Unsupported type conversion: {} to {}",
                           cudf::type_to_name(result->type()),
                           cudf::type_to_name(type));
}

std::unique_ptr<cudf::column> expression_evaluator::scatter_selected(
  cudf::column_view const& value,
  cudf::column_view const& target,
  cudf::column_view const& selection)
{
  auto scattered = cudf::boolean_mask_scatter(
    cudf::table_view{{value}}, cudf::table_view{{target}}, selection, _stream, _mr);
  return std::move(scattered->release()[0]);
}

}  // namespace sirius
