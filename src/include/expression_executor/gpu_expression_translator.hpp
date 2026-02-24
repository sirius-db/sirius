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

#pragma once

// sirius
#include <cudf_utils.hpp>

// duckdb
#include <duckdb/common/types.hpp>
#include <duckdb/planner/expression/bound_between_expression.hpp>
#include <duckdb/planner/expression/bound_case_expression.hpp>
#include <duckdb/planner/expression/bound_cast_expression.hpp>
#include <duckdb/planner/expression/bound_comparison_expression.hpp>
#include <duckdb/planner/expression/bound_conjunction_expression.hpp>
#include <duckdb/planner/expression/bound_constant_expression.hpp>
#include <duckdb/planner/expression/bound_function_expression.hpp>
#include <duckdb/planner/expression/bound_operator_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/joinside.hpp>

// cudf
#include <cudf/ast/ast_operator.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/types.hpp>

// rmm
#include <rmm/cuda_stream_view.hpp>

// standard library
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace sirius {

/**
 * @brief Class for translating (a subset of) DuckDB expressions and join conditions into cuDF ASTs
 * for GPU execution.
 *
 * The translator tries to translate the DuckDB expression. If it succeeds, it returns a
 * cudf::ast::tree, otherwise returns std::nullopt. Notable exceptions to translation include
 * anything involving conditional logic (CASE, COALESCE, TRY), functions beyond numeric binary
 * functions, and DISTINCT operators.
 */
class gpu_expression_translator {
 public:
  /**
   * @brief Owning result of translation.
   *
   * cuDF AST literals keep references to cudf::scalar objects. This type co-owns those scalars
   * so the AST remains valid after the translator instance is destroyed.
   */
  struct translated_expression {
    cudf::ast::tree tree{};
    std::vector<std::unique_ptr<cudf::scalar>> owned_literals{};

    [[nodiscard]] cudf::ast::expression const& back() const { return tree.back(); }
    [[nodiscard]] cudf::ast::expression const& front() const { return tree.front(); }
    [[nodiscard]] size_t size() const { return tree.size(); }
  };

  /**
   * @brief Construct a translator instance with the given CUDA stream and resource reference.
   *
   * @param stream The CUDA stream to use for any operations performed by the translator (e.g. in
   * constructing literals).
   * @param resource_ref The RMM resource reference to use for any operations performed
   */
  gpu_expression_translator(rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref resource_ref)
    : _stream(stream), _resource_ref(resource_ref)
  {
  }

  /**
   * @brief Try to translate a DuckDB expression into a cuDF AST.
   *
   * @param expr The DuckDB expression to translate.
   * @param table_src The table reference (LEFT, RIGHT, or NONE) to use for any column references in
   * the expression (DEFAULT is LEFT, and the argument is immaterial for single-table expressions,
   * i.e., not join conditions).
   * @return An optional containing the translated expression and its owned scalar literals if
   * translation succeeded, or std::nullopt if translation failed because it encountered an
   * expression that could not be translated to cuDF AST.
   */
  std::optional<translated_expression> translate_expression(
    duckdb::Expression const& expr,
    cudf::ast::table_reference const table_src = cudf::ast::table_reference::LEFT);

  /**
   * @brief Try to translate a DuckDB join condition into a cuDF AST.
   *
   * @param condition The DuckDB join condition to translate.
   * @return An optional containing the translated expression and its owned scalar literals if
   * translation succeeded, or std::nullopt if translation failed because it encountered an
   * expression that could not be translated to cuDF AST.
   */
  std::optional<translated_expression> translate_join_condition(
    duckdb::JoinCondition const& condition);

 private:
  // std::optional cannot wrap a real reference, so we use reference_wrapper instead
  using expr_ref = std::reference_wrapper<cudf::ast::expression const>;

  /// @brief Reset the translator's AST tree and owned literal scalars.
  void reset_tree()
  {
    _ast_tree = cudf::ast::tree{};
    _literal_scalars.clear();
  }

  /// @brief Add the given expression to the AST, returning a reference to the added expression if
  /// translation was successful, or std::nullopt if translation failed.
  std::optional<expr_ref> add_expression(duckdb::Expression const& expr,
                                         cudf::ast::table_reference const table_src);
  /// @brief Specialized add_expression for BETWEEN expressions.
  std::optional<expr_ref> add_expression(duckdb::BoundBetweenExpression const& expr,
                                         cudf::ast::table_reference const table_src);
  /// @brief Specialized add_expression for CAST expressions.
  std::optional<expr_ref> add_expression(duckdb::BoundCastExpression const& expr,
                                         cudf::ast::table_reference const table_src);
  /// @brief Specialized add_expression for COMPARISON expressions.
  std::optional<expr_ref> add_expression(duckdb::BoundComparisonExpression const& expr,
                                         cudf::ast::table_reference const table_src);
  /// @brief Specialized add_expression for CONJUNCTION expressions.
  std::optional<expr_ref> add_expression(duckdb::BoundConjunctionExpression const& expr,
                                         cudf::ast::table_reference const table_src);
  /// @brief Specialized add_expression for CONSTANT expressions.
  std::optional<expr_ref> add_expression(duckdb::BoundConstantExpression const& expr,
                                         cudf::ast::table_reference const table_src);
  /// @brief Specialized add_expression for FUNCTION expressions.
  std::optional<expr_ref> add_expression(duckdb::BoundFunctionExpression const& expr,
                                         cudf::ast::table_reference const table_src);
  /// @brief Specialized add_expression for OPERATOR expressions.
  std::optional<expr_ref> add_expression(duckdb::BoundOperatorExpression const& expr,
                                         cudf::ast::table_reference const table_src);
  /// @brief Specialized add_expression for column REFERENCE expressions.
  std::optional<expr_ref> add_expression(duckdb::BoundReferenceExpression const& expr,
                                         cudf::ast::table_reference const table_src);

  /// @brief Helper function for adding a binary function expression (e.g. addition) to the AST.
  /// OP is the cuDF AST operator corresponding to the function (e.g. cudf::ast::ast_operator::ADD).
  template <cudf::ast::ast_operator OP>
  std::optional<expr_ref> add_function_expression(duckdb::BoundFunctionExpression const& expr,
                                                  cudf::ast::table_reference const table_src)
  {
    // Translate children
    auto left_expr  = add_expression(*expr.children[0], table_src);
    auto right_expr = add_expression(*expr.children[1], table_src);

    // Check for failure in translating children
    if (!left_expr || !right_expr) { return std::nullopt; }

    // Construct the addition expression
    return _ast_tree.emplace<cudf::ast::operation>(OP, *left_expr, *right_expr);
  }

  /// @brief Construct and own a scalar, then add a literal referencing it to the AST.
  template <typename SCALAR_T, typename... ARGS>
  std::optional<expr_ref> add_literal_expression(ARGS&&... args)
  {
    auto scalar      = std::make_unique<SCALAR_T>(std::forward<ARGS>(args)...);
    auto& scalar_ref = *scalar;
    _literal_scalars.push_back(std::move(scalar));
    return _ast_tree.emplace<cudf::ast::literal>(scalar_ref);
  }

  cudf::ast::tree _ast_tree{};  ///< The cuDF AST being constructed by the translator. The final
                                ///< expression will be the back of the tree.
  std::vector<std::unique_ptr<cudf::scalar>>
    _literal_scalars{};           ///< Owning storage for scalar literals referenced by AST nodes.
  rmm::cuda_stream_view _stream;  ///< The CUDA stream to use for any operations performed by the
                                  ///< translator (e.g. in constructing literals).
  rmm::device_async_resource_ref
    _resource_ref;  ///< The RMM resource reference to use for any operations performed by the
                    ///< translator (e.g. in constructing literals).
};

}  // namespace sirius
