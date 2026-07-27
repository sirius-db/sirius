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
#include <cudf/cudf_utils.hpp>
#include <cudf/unary.hpp>

#include <expression/ast/node.hpp>
#include <expression_evaluator/expression_evaluator.hpp>
#include <helper/numeric_narrowing.hpp>
#include <sirius/exception.hpp>

#include <algorithm>

namespace sirius {
using ast_result      = expression_evaluator::ast_result;
using evaluate_result = expression_evaluator::evaluate_result;

evaluate_result expression_evaluator::get_or_create_restored_reference(std::uint32_t column_index,
                                                                       cudf::data_type target_type,
                                                                       evaluation_mode mode)
{
  auto const cached =
    std::find_if(_restored_reference_cache.begin(),
                 _restored_reference_cache.end(),
                 [column_index, target_type](restored_reference_cache_entry const& entry) {
                   return entry.column_index == column_index && entry.target_type == target_type;
                 });

  std::size_t temp_column_index;
  if (cached == _restored_reference_cache.end()) {
    temp_column_index = _temp_columns.size();
    _temp_columns.push_back(
      cudf::cast(_input_table.column(column_index), target_type, _stream, _mr));
    _restored_reference_cache.push_back(
      restored_reference_cache_entry{column_index, target_type, temp_column_index});
    ++_restored_reference_cast_count;
  } else {
    temp_column_index = cached->temp_column_index;
  }

  auto const& restored = _temp_columns.at(temp_column_index);
  if (!restored) {
    throw internal_exception(
      "[expression_evaluator] cached restored reference {} to {} was released early",
      column_index,
      cudf::type_to_name(target_type));
  }

  if (mode == evaluation_mode::AST) {
    auto const combined_index    = _input_table.num_columns() + temp_column_index;
    auto const& column_reference = _ast_tree.emplace<cudf::ast::column_reference>(combined_index);
    // Cache-owned entries stay alive for the full top-level call, so this AST result intentionally
    // advertises no releasable temporary-column indices.
    return evaluate_result(ast_result(column_reference));
  }
  return evaluate_result(restored->view());
}

evaluate_result expression_evaluator::evaluate(sirius::ast::reference const& alt,
                                               evaluation_mode mode)
{
  auto const source   = _input_table.column(alt.column_index);
  auto const& logical = alt.return_type();
  if (is_narrowable_numeric_type(logical)) {
    auto const native = get_cudf_type(logical);
    if (can_restore_to(source.type(), native)) {
      return get_or_create_restored_reference(alt.column_index, native, mode);
    }
  }

  if (mode == evaluation_mode::AST) {
    auto const& col_expr = _ast_tree.emplace<cudf::ast::column_reference>(alt.column_index);
    return evaluate_result(ast_result(col_expr));
  }
  return evaluate_result(source);
}
}  // namespace sirius
