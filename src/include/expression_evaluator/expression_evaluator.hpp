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
#include <config.hpp>
#include <expression/ast/node.hpp>  // sirius::ast::node + 11 alternative types
#include <expression_evaluator/expression_evaluator_strategy.hpp>

// cudf
#include <cudf/ast/expressions.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

// rmm
#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

// standard library
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <span>
#include <variant>
#include <vector>

namespace sirius {

/**
 * @brief Returns the current default expression_evaluator_strategy configured via
 * `duckdb::Config::EXPRESSION_EVALUATOR_STRATEGY`.
 */
inline expression_evaluator_strategy strategy_from_config()
{
  return duckdb::Config::EXPRESSION_EVALUATOR_STRATEGY;
}

/**
 * @brief The expression_evaluator is responsible for evaluating Sirius AST expressions on the GPU
 * using cuDF.
 *
 * It builds a tree of AST trees whose edges are 'AST breakers', i.e., expression
 * operations that don't have cuDF AST equivalents (e.g., LIKE, SUBSTRING, CASE, etc.). Each AST
 * tree is then interpreted or JIT-compiled with cuDF to produce the final result. If the
 * expression_evaluator_strategy is MATERIALIZE, the AST trees are single operators and revert to
 * direct unary/binary operators. To control how many nodes should be in an AST tree before we
 * evaluate in AST mode (rather than MATERIALIZE mode), toggle the `min_ast_size` parameter in the
 * constructor.
 */
class expression_evaluator {
  using expr_ref = std::reference_wrapper<cudf::ast::expression const>;

 public:
  /**
   * @brief The result of adding an expression to the executor's AST tree.
   *
   * It stores the reference to the AST node corresponding to the expression, as well as the indices
   * of any temporary scalars or columns that need to be kept alive for the AST node to be valid.
   * When the tree to which this AST node belongs is evaluated, these scalars and columns are
   * released.
   */
  struct ast_result {
    expr_ref expr;  ///< The reference to the AST node corresponding to the expression.
    std::vector<std::size_t> temp_scalar_indices;  ///< The indices of the temp scalars that need to
                                                   ///< be kept alive for this AST expression.
    std::vector<std::size_t> temp_column_indices;  ///< The indices of the temp columns that need to
                                                   ///< be kept alive for this AST expression.

    /**
     * @brief Construct an ast_result with the given AST node reference and no temporary scalars or
     * columns
     *
     * @param e The reference to the AST node corresponding to the expression
     */
    ast_result(expr_ref e) : expr(e) {}

    /**
     * @brief Construct an ast_result with the given AST node reference and scalar and column
     * indices.
     *
     * @param e The reference to the AST node corresponding to the expression
     * @param scalar_indices The indices of the temp scalars that need to be kept alive for this AST
     * expression.
     * @param column_indices The indices of the temp columns that need to be kept alive for this AST
     * expression.
     */
    ast_result(expr_ref e,
               std::vector<std::size_t> scalar_indices,
               std::vector<std::size_t> column_indices)
      : expr(e),
        temp_scalar_indices(std::move(scalar_indices)),
        temp_column_indices(std::move(column_indices))
    {
    }
  };

  /**
   * @brief The result placeholder for executing an expression.
   *
   * It holds either 1) an ast_result if the expression was added to the AST tree,
   *                 2) a cudf::column_view if the expression is a BOUND_REFERENCE evaluated in
   *                    MATERIALIZE mode,
   *                 3) a std::unique_ptr<cudf::scalar> if the expression is a BOUND_CONSTANT
   *                    evaluated in MATERIALIZE mode.
   *                 4) a std::unique_ptr<cudf::column> if the expression is an interior
   *                    node evaluated in MATERIALIZE mode.
   */
  struct evaluate_result {
    std::variant<ast_result,
                 cudf::column_view,
                 std::unique_ptr<cudf::scalar>,
                 std::unique_ptr<cudf::column>>
      payload;

    evaluate_result() = delete;

    /// @brief Constructs an evaluate_result holding an AST expression reference.
    evaluate_result(ast_result ast_payload) : payload(std::move(ast_payload)) {}

    /// @brief Constructs an evaluate_result holding a non-owning column view (e.g. a bound
    /// reference in MATERIALIZE mode).
    evaluate_result(cudf::column_view column_view_payload) : payload(column_view_payload) {}

    /// @brief Constructs an evaluate_result holding an owning scalar (e.g. a bound constant in
    /// MATERIALIZE mode).
    evaluate_result(std::unique_ptr<cudf::scalar> scalar_payload)
      : payload(std::move(scalar_payload))
    {
    }

    /// @brief Constructs an evaluate_result holding an owning column (e.g. an interior expression
    /// node evaluated in MATERIALIZE mode).
    evaluate_result(std::unique_ptr<cudf::column> column_payload)
      : payload(std::move(column_payload))
    {
    }

    /// @brief Returns true if the payload holds an ast_result.
    [[nodiscard]] bool is_ast() const { return std::holds_alternative<ast_result>(payload); }

    /// @brief Returns true if the payload holds a cudf::scalar.
    [[nodiscard]] bool is_scalar() const
    {
      return std::holds_alternative<std::unique_ptr<cudf::scalar>>(payload);
    }

    /// @brief Returns true if the payload holds a cudf::column_view.
    [[nodiscard]] bool is_column_view() const
    {
      return std::holds_alternative<cudf::column_view>(payload);
    }

    /// @brief Returns true if the payload holds an owned cudf::column.
    [[nodiscard]] bool is_owned_column() const
    {
      return std::holds_alternative<std::unique_ptr<cudf::column>>(payload);
    }

    /**
     * @brief Returns the AST expression reference from the payload.
     * @throws std::runtime_error if the payload does not hold an ast_result.
     */
    [[nodiscard]] expr_ref get_expr() const;

    /**
     * @brief Returns a const reference to the cudf::scalar held by the payload.
     * @throws std::runtime_error if the payload does not hold a cudf::scalar.
     */
    [[nodiscard]] cudf::scalar const& get_scalar() const;

    /**
     * @brief Returns a column_view of the result.
     *
     * If the payload holds a cudf::column_view, it is returned directly. If it holds a
     * std::unique_ptr<cudf::column>, a view of the owned column is returned.
     *
     * @throws std::runtime_error if the payload holds an ast_result or a cudf::scalar.
     */
    [[nodiscard]] cudf::column_view get_column_view() const;

    /**
     * @brief Moves the owned cudf::column out of the payload.
     * @throws std::runtime_error if the payload does not hold a std::unique_ptr<cudf::column>.
     */
    [[nodiscard]] std::unique_ptr<cudf::column> release_column();
  };

  /**
   * @brief The mode in which to evaluate an expression.
   *
   * This is used internally by the expression executor to switch between trying to add expression
   * nodes to the AST tree or materializing them as cudf::columns during execution. The mode is
   * determined by the expression_evaluator_strategy and the min_ast_size parameters of the
   * constructor, but can also be overridden for individual expressions by passing the desired mode
   * to the evaluate method. Note that if an expression is executed in AST mode, it is added to the
   * AST tree and the result is an ast_result; if it is executed in MATERIALIZE mode, it is executed
   * directly and the result is a column or scalar. The evaluation_mode parameter of the evaluate
   * method is just a hint and is not strictly enforced, e.g., if you pass evaluation_mode::AST but
   * the expression node is an AST breaker, it will be executed in MATERIALIZE mode instead.
   */
  enum class evaluation_mode {
    AST,
    MATERIALIZE,
  };

  /**
   * @brief Construct a expression_evaluator with the given set of expressions (for PROJECTION
   * operators).
   *
   * @param expressions The expressions to evaluate.
   * @param resource_ref The rmm::device_async_resource_ref to pass to cuDF APIs for allocations.
   * @param stream The rmm::cuda_stream_view in which to evaluate any cuDF operations.
   * @param strategy The strategy to use for expression execution (AST_INTERPRET, AST_JIT, or
   * MATERIALIZE). Defaults to the value of `duckdb::Config::EXPRESSION_EVALUATOR_STRATEGY`.
   * @param min_ast_size The minimum number of nodes in an AST tree before we switch from
   * MATERIALIZE mode to AST mode. If an expression subtree rooted at a given node produces an AST
   * with N operators and N < min_ast_size, the expression will be evaluated operator-by-operator
   * (in MATERIALIZE mode). Otherwise, the executor will try to evaluate the expression subtree by
   * adding nodes to the AST tree.
   */
  expression_evaluator(
    duckdb::vector<std::unique_ptr<sirius::ast::node>> const& expressions,
    rmm::device_async_resource_ref resource_ref = cudf::get_current_device_resource_ref(),
    rmm::cuda_stream_view stream                = cudf::get_default_stream(),
    expression_evaluator_strategy strategy      = strategy_from_config(),
    std::size_t min_ast_size                    = 2);

  /**
   * @brief Construct a expression_evaluator with the given expression (for FILTER operators).
   *
   * @param expression The expressions to evaluate.
   * @param resource_ref The rmm::device_async_resource_ref to pass to cuDF APIs for allocations.
   * @param stream The rmm::cuda_stream_view in which to evaluate any cuDF operations.
   * @param strategy The strategy to use for expression execution (AST_INTERPRET, AST_JIT, or
   * MATERIALIZE). Defaults to the value of `duckdb::Config::EXPRESSION_EVALUATOR_STRATEGY`.
   * @param min_ast_size The minimum number of nodes in an AST tree before we switch from
   * MATERIALIZE mode to AST mode. If an expression subtree rooted at a given node produces an AST
   * with N operators and N < min_ast_size, the expression will be evaluated operator-by-operator
   * (in MATERIALIZE mode). Otherwise, the executor will try to evaluate the expression subtree by
   * adding nodes to the AST tree.
   */
  expression_evaluator(
    sirius::ast::node const& expression,
    rmm::device_async_resource_ref resource_ref = cudf::get_current_device_resource_ref(),
    rmm::cuda_stream_view stream                = cudf::get_default_stream(),
    expression_evaluator_strategy strategy      = strategy_from_config(),
    std::size_t min_ast_size                    = 2);

  /**
   * @brief Non-owning ctor for call sites that hold a raw sirius::ast::node pointer
   * (e.g., NLJ lambda over cuDF expressions, parquet scan filter pushdown).
   *
   * The caller retains ownership; the executor only reads from the node tree.
   */
  expression_evaluator(
    sirius::ast::node const* expression,
    rmm::device_async_resource_ref resource_ref = cudf::get_current_device_resource_ref(),
    rmm::cuda_stream_view stream                = cudf::get_default_stream(),
    expression_evaluator_strategy strategy      = strategy_from_config(),
    std::size_t min_ast_size                    = 2);

  /**
   * @brief Non-owning ctor for a pre-filtered list of expressions (e.g. the projection operator's
   * subset of select_list entries that actually need evaluation, after pulling out pure
   * BOUND_REF passthroughs).
   *
   * The caller retains ownership of the nodes; the executor only reads from them. The output
   * table produced by evaluate() contains one column per entry in @p expressions, in the same
   * order.
   */
  expression_evaluator(
    std::vector<sirius::ast::node const*> expressions,
    rmm::device_async_resource_ref resource_ref = cudf::get_current_device_resource_ref(),
    rmm::cuda_stream_view stream                = cudf::get_default_stream(),
    expression_evaluator_strategy strategy      = strategy_from_config(),
    std::size_t min_ast_size                    = 2);

  /**
   * @brief Executes the current set of expressions against the given input batch and emits a new
   * output batch with the results.
   *
   * @param input The read-only locked input batch against which to evaluate expressions.
   * @return A new idle batch containing the results of expression evaluation.
   */
  std::unique_ptr<cudf::table> evaluate(cudf::table_view input);

  /**
   * @brief Select the rows passing the predicate, materializing only @p output_indices.
   *
   * The predicate is evaluated over the full @p input, so columns referenced only by the
   * predicate (pure filter columns) are available to it but are never materialized in the
   * result. Output column @c i is the filtered @p input column @c output_indices[i], so the
   * caller controls both projection and column order in a single gather.
   *
   * @param input The read-only locked input batch from which to select rows.
   * @param output_indices Indices into @p input's columns to materialize, in output order.
   *        Must be non-empty: a 0-column result cannot carry a row count, so count(*)-style
   *        filters with no output columns must use the all-columns select() overload.
   * @return A new idle batch containing the selected rows and columns.
   */
  std::unique_ptr<cudf::table> select(cudf::table_view input,
                                      std::span<cudf::size_type const> output_indices);

  /**
   * @brief Selects rows from the input batch based on the executor's (singular) expression.
   *
   * @param input The read-only locked input batch from which to select rows.
   * @return A new idle batch containing the selected rows.
   *
   * @note This method should only be used when there are no pure filter columns in @p input;
   *       otherwise, the intermediate gather step materializes them unnecessarily.
   */
  std::unique_ptr<cudf::table> select(cudf::table_view input);

  /**
   * @brief Evaluate a single Sirius AST node and return its execution result.
   *
   * Dispatches via std::visit over @p expr's variant to the matching private
   * per-alternative overload; every alternative evaluates natively on the
   * Sirius AST.
   *
   * @param expr The Sirius AST node to evaluate.
   * @param mode AST hint vs. MATERIALIZE hint; honored only where the node kind
   *             supports AST mode. AST breakers (case_expr, coalesce, op_try,
   *             etc.) always materialize regardless of the hint.
   */
  evaluate_result evaluate(sirius::ast::node const& expr,
                           evaluation_mode mode = evaluation_mode::AST);

  /// Number of numeric restoration casts issued by the most recent top-level evaluation.
  /// Exposed so tests can verify repeated references share one cached restoration.
  [[nodiscard]] std::size_t restored_reference_cast_count_for_testing() const noexcept
  {
    return _restored_reference_cast_count;
  }

  /// Number of comparison/BETWEEN nodes the most recent top-level evaluation executed directly on
  /// a narrowed carrier (no restoration cast). Exposed so tests can verify the narrow-domain path
  /// engaged instead of merely producing correct results through a restore.
  [[nodiscard]] std::size_t narrow_domain_comparison_count_for_testing() const noexcept
  {
    return _narrow_domain_comparison_count;
  }

 private:
  std::vector<sirius::ast::node const*> _ast_expressions;  ///< The AST expressions to evaluate
  expression_evaluator_strategy _strategy;  ///< The strategy to use for expression evaluation
  rmm::device_async_resource_ref _mr;  ///< The allocator to pass to cudf APIs for any allocations
  rmm::cuda_stream_view _stream;       ///< The stream in which to evaluate any cuDF operations
  std::size_t _min_ast_size;  ///< The minimum number of nodes in an AST tree before we switch from
                              ///< MATERIALIZE mode to AST mode
  cudf::table_view _input_table;  ///< The input table for expression evaluation
  std::vector<std::unique_ptr<cudf::column>>
    _output_columns;  ///< The output columns generated by expression evaluation (one per
                      ///< expression)

  cudf::ast::tree
    _ast_tree;  ///< The AST tree maintaining the set of AST nodes during expression evaluation.
  std::vector<std::unique_ptr<cudf::scalar>>
    _temp_scalars;  ///< The temporary scalars that need to be kept alive for the AST nodes in
                    ///< _ast_tree.
  std::vector<std::unique_ptr<cudf::column>>
    _temp_columns;  ///< The temporary columns that need to be kept alive for the AST nodes in
                    ///< _ast_tree.

  // Numeric reference restorations live in _temp_columns so AST references address them through the
  // combined-table layout. AST results built over these cache entries deliberately advertise no
  // releasable indices to release_temporaries: a restoration stays alive for the whole top-level
  // evaluation.
  struct restored_reference_cache_entry {
    std::uint32_t column_index;     ///< The index of the input column in the original table
    cudf::data_type target_type;    ///< The target type for the numeric restoration
    std::size_t temp_column_index;  ///< The index of the CAST temporary column in _temp_columns
  };
  std::vector<restored_reference_cache_entry> _restored_reference_cache;
  std::size_t _restored_reference_cast_count{0};   ///< For observability/testing
  std::size_t _narrow_domain_comparison_count{0};  ///< For observability/testing

  // Evaluate the executor's single boolean predicate over @p input and return the resulting
  // mask column (the sole column of evaluate()'s output). Shared by both select() overloads.
  std::unique_ptr<cudf::column> compute_mask(cudf::table_view input);

  // Execute the AST tree rooted at the given expression reference and return the result as a
  // column.
  std::unique_ptr<cudf::column> evaluate_ast(expr_ref root_expr);

  /**
   * @brief Stores a materialized column as a temporary and returns an ast_result referencing it.
   *
   * This is used by AST breakers (e.g. CASE, unsupported CAST) that cannot represent their logic
   * as cudf AST nodes. When called with evaluation_mode::AST from a parent expression, they
   * materialize their result, stash it in _temp_columns, and return an ast_result with a
   * column_reference pointing to the temp column's index in the combined table.
   */
  evaluate_result materialize_as_ast_column(std::unique_ptr<cudf::column> column);

  // Return a cached strict numeric restoration for one input reference. MATERIALIZE mode receives
  // a non-owning view; AST mode receives a reference to the cache-owned _temp_columns entry.
  evaluate_result get_or_create_restored_reference(std::uint32_t column_index,
                                                   cudf::data_type target_type,
                                                   evaluation_mode mode);

  // Narrow-domain comparison support (compressed materialization). Narrowing is value-preserving
  // (same values, same family, same DECIMAL scale — no offset), so a comparison between a
  // narrowed reference and constants exactly representable in its carrier yields identical
  // results computed at the narrow width. These helpers let comparison/BETWEEN skip the
  // full-column restoration cast for that shape; every ineligible shape falls back to the
  // restore path.
  //
  // Returns the narrowed input carrier when @p column_operand is a reference whose materialized
  // carrier is a strict narrowing of its declared native type AND every entry of @p
  // constant_operands is a constant exactly representable in that carrier (typed NULLs always are).
  // Returns `std::nullopt` otherwise.
  [[nodiscard]] std::optional<cudf::data_type> narrow_domain_carrier(
    sirius::ast::node const& column_operand,
    std::initializer_list<sirius::ast::node const*> constant_operands) const;

  // Return the shared carrier when both operands are in-range references and
  // narrow_domain_reference_pair_eligible accepts their logical types and materialized carriers.
  // Return std::nullopt otherwise.
  [[nodiscard]] std::optional<cudf::data_type> narrow_domain_reference_pair_carrier(
    sirius::ast::node const& lhs, sirius::ast::node const& rhs) const;

  // Evaluate a numeric constant as a scalar of @p carrier (a narrowed integer or fixed-point
  // carrier). Callers must have validated representability via narrow_domain_carrier.
  evaluate_result evaluate_constant_in_carrier(sirius::ast::constant const& expr,
                                               cudf::data_type carrier,
                                               evaluation_mode mode);

  // Park @p device_scalar (a concrete cudf scalar type — cudf::ast::literal's constructors are
  // typed) and return it as an evaluate_result: AST mode emplaces a literal over the scalar and
  // stores the ownership in _temp_scalars, advertising the index for release; MATERIALIZE mode
  // returns the owning scalar directly.
  template <typename ScalarPtr>
  evaluate_result finish_scalar(ScalarPtr device_scalar, evaluation_mode mode)
  {
    if (mode == evaluation_mode::AST) {
      auto const& literal_ref    = _ast_tree.emplace<cudf::ast::literal>(*device_scalar);
      auto const temp_scalar_idx = _temp_scalars.size();
      _temp_scalars.push_back(std::move(device_scalar));
      return evaluate_result(
        ast_result(literal_ref, {temp_scalar_idx}, std::vector<std::size_t>{}));
    }
    return evaluate_result(std::move(device_scalar));
  }

  // Evaluate one comparison/BETWEEN operand: under an engaged @p narrow_carrier, a reference
  // passes through at its materialized narrow width and a constant materializes in the carrier;
  // without one, the operand takes the ordinary dispatch.
  evaluate_result evaluate_narrow_domain_operand(sirius::ast::node const& operand,
                                                 std::optional<cudf::data_type> narrow_carrier,
                                                 evaluation_mode mode);

  // Flatten the AST children's kept-alive temporary indices into one ast_result for the composed
  // expression @p e. Non-AST children contribute nothing.
  [[nodiscard]] static ast_result compose(expr_ref e,
                                          std::initializer_list<evaluate_result const*> children);

  // Release the temporary scalars and columns kept alive by the AST results in @p children.
  void release_temporaries(std::initializer_list<evaluate_result const*> children);

  // Leaf Sirius-AST nodes — dispatch targets for the std::visit-based
  // evaluate(sirius::ast::node, mode) above.
  evaluate_result evaluate(sirius::ast::reference const& expr, evaluation_mode mode);
  evaluate_result evaluate(sirius::ast::constant const& expr, evaluation_mode mode);

  // Interior Sirius-AST nodes — 11 alternatives total (BOUND_OPERATOR's kinds
  // split across unary_op, coalesce, and in_list).
  evaluate_result evaluate(sirius::ast::between const& expr, evaluation_mode mode);
  evaluate_result evaluate(sirius::ast::case_expr const& expr, evaluation_mode mode);
  evaluate_result evaluate(sirius::ast::cast const& expr, evaluation_mode mode);
  evaluate_result evaluate(sirius::ast::comparison const& expr, evaluation_mode mode);
  evaluate_result evaluate(sirius::ast::conjunction const& expr, evaluation_mode mode);
  evaluate_result evaluate(sirius::ast::function_call const& expr, evaluation_mode mode);
  evaluate_result evaluate(sirius::ast::unary_op const& expr, evaluation_mode mode);
  evaluate_result evaluate(sirius::ast::coalesce const& expr, evaluation_mode mode);
  evaluate_result evaluate(sirius::ast::in_list const& expr, evaluation_mode mode);
  evaluate_result evaluate(sirius::ast::aggregate const& expr, evaluation_mode mode);
};

}  // namespace sirius
