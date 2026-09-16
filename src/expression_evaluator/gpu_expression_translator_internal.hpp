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
#include <cudf/cudf_utils.hpp>

#include <expression/ast/node.hpp>
#include <expression/join_condition.hpp>

// duckdb
#include <duckdb/common/types.hpp>
#include <duckdb/planner/joinside.hpp>

// cudf
#include <cudf/ast/ast_operator.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/types.hpp>

// rmm
#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>

// standard library
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace sirius {

/**
 * @brief Convert a cudf::ast::ast_operator to a human-readable string.
 */
std::string ast_operator_to_string(cudf::ast::ast_operator op);

/**
 * @brief Recursively produce a human-readable string for a cuDF AST expression.
 */
std::string expression_to_string(cudf::ast::expression const& expr);

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
   * @brief A function that maps a BoundReferenceExpression column index to a column name.
   *
   * This is needed for translating expressions that reference columns by name (e.g., parquet with
   * filter pushdown), since the default translator uses column index references to a cuDF table in
   * the AST.
   */
  using column_name_resolver_fxn = std::function<std::string(duckdb::idx_t)>;

  /**
   * @brief Owning result of translation.
   *
   * cuDF AST literals keep references to cudf::scalar objects. This type co-owns those scalars
   * so the AST remains valid after the translator instance is destroyed.
   */
  struct translated_expression {
    cudf::ast::tree tree{};
    /// Owning stream whose lifetime covers all async scalar allocations.
    /// DECLARATION ORDER MATTERS: owned_stream is declared BEFORE owned_literals
    /// so its destructor runs AFTER the scalar destructors (C++ destroys members
    /// in reverse declaration order — last declared is destroyed first).
    /// This ensures cudaStreamDestroy fires AFTER all scalars in owned_literals
    /// are freed via cudaFreeAsync — the scalars' cudaFreeAsync calls reference
    /// this stream handle so it must be valid throughout scalar destruction.
    /// Optional: set only by callers that allocate scalars on a temporary
    /// stream; leave empty when scalars use a persistent stream (or none).
    std::optional<rmm::cuda_stream> owned_stream{};
    std::vector<std::unique_ptr<cudf::scalar>> owned_literals{};

    [[nodiscard]] cudf::ast::expression const& back() const { return tree.back(); }
    [[nodiscard]] cudf::ast::expression const& front() const { return tree.front(); }
    [[nodiscard]] std::size_t size() const { return tree.size(); }
    [[nodiscard]] std::string to_string() const;
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
   * @brief Try to translate a Sirius AST expression into a cuDF AST.
   *
   * @param expr The Sirius AST expression to translate.
   * @param table_src The table reference (LEFT, RIGHT, or NONE) to use for any column references in
   * the expression (DEFAULT is LEFT, and the argument is immaterial for single-table expressions,
   * i.e., not join conditions).
   * @return An optional containing the translated expression and its owned scalar literals if
   * translation succeeded, or std::nullopt if translation failed because it encountered an
   * expression that could not be translated to cuDF AST.
   */
  std::optional<translated_expression> translate_expression(
    sirius::ast::node const& expr,
    cudf::ast::table_reference const table_src = cudf::ast::table_reference::LEFT);

  /**
   * @brief Try to translate a Sirius AST expression into a cuDF AST, using column name references
   * instead of column index references.
   *
   * This is needed for cudf's parquet predicate pushdown (e.g. hybrid scan), which expects
   * filter expressions to use cudf::ast::column_name_reference (by name) rather than
   * cudf::ast::column_reference (by index).
   *
   * @param expr The Sirius AST expression to translate.
   * @param column_name_resolver A function mapping a reference column index to a column
   * name string.
   * @return An optional containing the translated expression, or std::nullopt if translation
   * failed.
   */
  std::optional<translated_expression> translate_expression_with_names(
    sirius::ast::node const& expr, column_name_resolver_fxn column_name_resolver);

  /**
   * @brief Try to translate a sirius join condition into a cuDF AST.
   *
   * @param condition The sirius join condition to translate.
   * @return An optional containing the translated expression and its owned scalar literals if
   * translation succeeded, or std::nullopt if translation failed because it encountered an
   * expression that could not be translated to cuDF AST.
   */
  std::optional<translated_expression> translate_join_condition(
    sirius::join_condition const& condition);

  /**
   * @brief Try to translate a contiguous range of sirius join conditions into a single cuDF AST,
   * combining them with logical AND.
   *
   * Intended for translating the inequality conditions of a mixed join into a binary predicate.
   *
   * @param conditions The vector of join conditions.
   * @param start_idx The first condition index to translate (inclusive).
   * @param end_idx One past the last condition index to translate (exclusive).
   * @param swap_sides If true, swap LEFT/RIGHT table references in the produced AST. Used when the
   * caller needs to flip the join sides (e.g. to implement a RIGHT join as a swapped LEFT join).
   * @return An optional containing the combined translated expression, or std::nullopt if any
   * condition could not be translated.
   */
  std::optional<translated_expression> translate_join_conditions(
    duckdb::vector<sirius::join_condition> const& conditions,
    std::size_t start_idx,
    std::size_t end_idx,
    bool swap_sides = false);

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
  std::optional<expr_ref> add_expression(sirius::ast::node const& expr,
                                         cudf::ast::table_reference const table_src);
  /// @brief Specialized add_expression for column REFERENCE expressions.
  std::optional<expr_ref> add_expression(sirius::ast::reference const& alt,
                                         cudf::ast::table_reference const table_src);
  /// @brief Specialized add_expression for CONSTANT expressions.
  std::optional<expr_ref> add_expression(sirius::ast::constant const& alt,
                                         cudf::ast::table_reference const table_src);
  /// @brief Specialized add_expression for COMPARISON expressions.
  std::optional<expr_ref> add_expression(sirius::ast::comparison const& alt,
                                         cudf::ast::table_reference const table_src);
  /// @brief Specialized add_expression for CONJUNCTION expressions.
  std::optional<expr_ref> add_expression(sirius::ast::conjunction const& alt,
                                         cudf::ast::table_reference const table_src);
  /// @brief Specialized add_expression for BETWEEN expressions.
  std::optional<expr_ref> add_expression(sirius::ast::between const& alt,
                                         cudf::ast::table_reference const table_src);
  /// @brief Specialized add_expression for CASE expressions (not translatable to cuDF AST).
  std::optional<expr_ref> add_expression(sirius::ast::case_expr const& alt,
                                         cudf::ast::table_reference const table_src);
  /// @brief Specialized add_expression for CAST expressions.
  std::optional<expr_ref> add_expression(sirius::ast::cast const& alt,
                                         cudf::ast::table_reference const table_src);
  /// @brief Specialized add_expression for unary SQL operators (NOT, IS NULL, IS NOT NULL, TRY).
  std::optional<expr_ref> add_expression(sirius::ast::unary_op const& alt,
                                         cudf::ast::table_reference const table_src);
  /// @brief Specialized add_expression for COALESCE expressions (not translatable to cuDF AST).
  std::optional<expr_ref> add_expression(sirius::ast::coalesce const& alt,
                                         cudf::ast::table_reference const table_src);
  /// @brief Specialized add_expression for IN / NOT IN expressions.
  std::optional<expr_ref> add_expression(sirius::ast::in_list const& alt,
                                         cudf::ast::table_reference const table_src);
  /// @brief Specialized add_expression for FUNCTION expressions.
  std::optional<expr_ref> add_expression(sirius::ast::function_call const& alt,
                                         cudf::ast::table_reference const table_src);
  /// @brief Specialized add_expression for AGGREGATE expressions (not translatable to cuDF AST).
  std::optional<expr_ref> add_expression(sirius::ast::aggregate const& alt,
                                         cudf::ast::table_reference const table_src);

  /// @brief Add a single join condition to the current AST tree without resetting it.
  /// @param condition The join condition to translate.
  /// @param swap_sides If true, swap LEFT/RIGHT table references in the column references.
  std::optional<expr_ref> add_join_condition(sirius::join_condition const& condition,
                                             bool swap_sides = false);

  /// @brief Helper function for adding a binary function expression (e.g. addition) to the AST.
  /// OP is the cuDF AST operator corresponding to the function (e.g. cudf::ast::ast_operator::ADD).
  template <cudf::ast::ast_operator OP>
  std::optional<expr_ref> add_function_expression(sirius::ast::function_call const& alt,
                                                  cudf::ast::table_reference const table_src)
  {
    // This helper builds a binary cuDF operation. DuckDB overloads some
    // operators on arity (e.g. unary negation also binds as function "-"), so a
    // function_call reaching here may carry a single argument; bail out rather
    // than indexing arguments()[1] out of bounds.
    if (alt.arguments().size() != 2) { return std::nullopt; }

    // Translate children
    auto left_expr  = add_expression(*alt.arguments()[0], table_src);
    auto right_expr = add_expression(*alt.arguments()[1], table_src);

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
  column_name_resolver_fxn
    _column_name_resolver{};  ///> Optional resolver for column name references in the AST
};

}  // namespace sirius
