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

#include <expression/ast/aggregate.hpp>  // sirius::ast::aggregate
#include <expression/ast/node.hpp>       // sirius::ast::node alternatives
#include <expression_evaluator/expression_evaluator.hpp>
#include <expression_evaluator/nvrtc/nvrtc_project_sink.hpp>
#include <expression_evaluator/nvrtc/supportability.hpp>
#include <helper/numeric_narrowing.hpp>
#include <sirius/exception.hpp>
#include <telemetry/nvtx.hpp>

// cudf
#include <cudf/column/column_factories.hpp>
#include <cudf/filling.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

// rmm
#include <rmm/resource_ref.hpp>

#include <cuda/stream>

// standard library
#include <array>
#include <memory>
#include <numeric>
#include <unordered_map>
#include <variant>

namespace {
/// Returns true if cudf::cast supports this type (fixed-width only; no STRING/LIST/STRUCT/etc.).
bool IsFixedWidth(cudf::data_type const& type)
{
  auto const id = type.id();
  return id != cudf::type_id::STRING && id != cudf::type_id::LIST && id != cudf::type_id::STRUCT &&
         id != cudf::type_id::DICTIONARY32 && id != cudf::type_id::EMPTY;
}

}  // namespace

namespace sirius {
using evaluate_result = expression_evaluator::evaluate_result;
using ast_result      = expression_evaluator::ast_result;

//===----------------------------------------------------------------------===//
// ast_result
//===----------------------------------------------------------------------===//

ast_result expression_evaluator::compose(expr_ref e,
                                         std::initializer_list<evaluate_result const*> children)
{
  ast_result result{e};
  std::size_t total_scalars = 0;
  std::size_t total_columns = 0;
  for (auto const* child : children) {
    if (!child->is_ast()) { continue; }
    auto const& ast = std::get<ast_result>(child->payload);
    total_scalars += ast.temp_scalar_indices.size();
    total_columns += ast.temp_column_indices.size();
  }
  result.temp_scalar_indices.reserve(total_scalars);
  result.temp_column_indices.reserve(total_columns);
  for (auto const* child : children) {
    if (!child->is_ast()) { continue; }
    auto const& ast = std::get<ast_result>(child->payload);
    result.temp_scalar_indices.insert(result.temp_scalar_indices.end(),
                                      ast.temp_scalar_indices.begin(),
                                      ast.temp_scalar_indices.end());
    result.temp_column_indices.insert(result.temp_column_indices.end(),
                                      ast.temp_column_indices.begin(),
                                      ast.temp_column_indices.end());
  }
  return result;
}

//===----------------------------------------------------------------------===//
// evaluate_result
//===----------------------------------------------------------------------===//

std::reference_wrapper<cudf::ast::expression const>
expression_evaluator::evaluate_result::get_expr() const
{
  if (!is_ast()) {
    throw std::runtime_error("[evaluate_result] Attempted to get expr from materialized result");
  }
  return std::get<ast_result>(payload).expr;
}

cudf::scalar const& expression_evaluator::evaluate_result::get_scalar() const
{
  if (!is_scalar()) {
    throw std::runtime_error(
      "[evaluate_result] Attempted to get scalar from non-scalar evaluate_result");
  }
  auto const& scalar_payload = std::get<std::unique_ptr<cudf::scalar>>(payload);
  return *scalar_payload;
}

cudf::column_view expression_evaluator::evaluate_result::get_column_view() const
{
  if (is_ast() || is_scalar()) {
    throw std::runtime_error(
      "[evaluate_result] Attempted to get column view from non-column evaluate_result");
  }
  if (std::holds_alternative<cudf::column_view>(payload)) {
    return std::get<cudf::column_view>(payload);
  }
  auto const& column_payload = std::get<std::unique_ptr<cudf::column>>(payload);
  return column_payload->view();
}

std::unique_ptr<cudf::column> expression_evaluator::evaluate_result::release_column()
{
  if (std::holds_alternative<std::unique_ptr<cudf::column>>(payload)) {
    return std::move(std::get<std::unique_ptr<cudf::column>>(payload));
  }
  throw std::runtime_error(
    "[evaluate_result] Attempted to get column from evaluate_result that does not hold a column");
}

//===----------------------------------------------------------------------===//
// expression_evaluator
//===----------------------------------------------------------------------===//

expression_evaluator::expression_evaluator(
  duckdb::vector<std::unique_ptr<sirius::ast::node>> const& expressions,
  rmm::device_async_resource_ref resource_ref,
  ::cuda::stream_ref stream,
  expression_query_policy policy,
  std::size_t min_ast_size,
  bool like_swar_fastpath,
  std::shared_ptr<expression_query_caches const> expression_caches)
  : _policy(policy),
    _mr(resource_ref),
    _stream(stream),
    _min_ast_size(min_ast_size),
    _like_swar_fastpath(like_swar_fastpath),
    _expression_caches(query_caches_or_private(std::move(expression_caches)))
{
  _ast_expressions.reserve(expressions.size());
  for (auto const& expr : expressions) {
    _ast_expressions.push_back(expr.get());
  }
}

expression_evaluator::expression_evaluator(
  sirius::ast::node const& expression,
  rmm::device_async_resource_ref resource_ref,
  ::cuda::stream_ref stream,
  expression_query_policy policy,
  std::size_t min_ast_size,
  bool like_swar_fastpath,
  std::shared_ptr<expression_query_caches const> expression_caches)
  : expression_evaluator(&expression,
                         resource_ref,
                         stream,
                         policy,
                         min_ast_size,
                         like_swar_fastpath,
                         std::move(expression_caches))
{
}

expression_evaluator::expression_evaluator(
  sirius::ast::node const* expression,
  rmm::device_async_resource_ref resource_ref,
  ::cuda::stream_ref stream,
  expression_query_policy policy,
  std::size_t min_ast_size,
  bool like_swar_fastpath,
  std::shared_ptr<expression_query_caches const> expression_caches)
  : _policy(policy),
    _mr(resource_ref),
    _stream(stream),
    _min_ast_size(min_ast_size),
    _like_swar_fastpath(like_swar_fastpath),
    _expression_caches(query_caches_or_private(std::move(expression_caches)))
{
  _ast_expressions.push_back(expression);
}

expression_evaluator::expression_evaluator(
  std::vector<sirius::ast::node const*> expressions,
  rmm::device_async_resource_ref resource_ref,
  ::cuda::stream_ref stream,
  expression_query_policy policy,
  std::size_t min_ast_size,
  bool like_swar_fastpath,
  std::shared_ptr<expression_query_caches const> expression_caches)
  : _ast_expressions(std::move(expressions)),
    _policy(policy),
    _mr(resource_ref),
    _stream(stream),
    _min_ast_size(min_ast_size),
    _like_swar_fastpath(like_swar_fastpath),
    _expression_caches(query_caches_or_private(std::move(expression_caches)))
{
}

std::unique_ptr<cudf::column> expression_evaluator::evaluate_ast(expr_ref root_expr)
{
  std::vector<cudf::column_view> combined_column_views;
  combined_column_views.reserve(_input_table.num_columns() + _temp_columns.size());
  for (int column_idx = 0; column_idx < _input_table.num_columns(); ++column_idx) {
    combined_column_views.push_back(_input_table.column(column_idx));
  }

  // We must be careful not to add invalidated temporary columns, as cuDF will throw when
  // constructing the table_view. Since the input table has a nonzero number of columns, we can
  // use the 0th column to produce a dummy column_view to maintain the integrity of the column
  // indices (note that this dummy column will never be referenced).
  for (auto const& temp_column : _temp_columns) {
    if (temp_column) {
      combined_column_views.push_back(temp_column->view());
    } else {
      combined_column_views.push_back(_input_table.column(0));
    }
  }

  cudf::table_view combined_table_view(combined_column_views);
  if (_policy.strategy == expression_evaluator_strategy::AST_JIT) {
    ++_ast_jit_evaluation_count;
    return cudf::compute_column_jit(combined_table_view, root_expr.get(), _stream, _mr);
  }
  // MATERIALIZE uses a small interpreted AST for floating comparison semantics.
  // Building that AST must not opt a query into the JIT backend.
  return cudf::compute_column(combined_table_view, root_expr.get(), _stream, _mr);
}

evaluate_result expression_evaluator::materialize_as_ast_column(
  std::unique_ptr<cudf::column> column)
{
  auto const col_idx         = _input_table.num_columns() + _temp_columns.size();
  auto const temp_column_idx = _temp_columns.size();
  _temp_columns.push_back(std::move(column));
  auto const& col_ref = _ast_tree.emplace<cudf::ast::column_reference>(col_idx);
  return evaluate_result(ast_result(col_ref, std::vector<std::size_t>{}, {temp_column_idx}));
}

evaluate_result expression_evaluator::evaluate_as_column(sirius::ast::node const& expr)
{
  auto result = evaluate(expr, evaluation_mode::MATERIALIZE);
  if (!result.is_scalar()) { return result; }
  return evaluate_result(
    cudf::make_column_from_scalar(result.get_scalar(), _input_table.num_rows(), _stream, _mr));
}

void expression_evaluator::release_temporaries(
  std::initializer_list<evaluate_result const*> children)
{
  for (auto const* child : children) {
    if (!child->is_ast()) { continue; }
    auto const& ast = std::get<ast_result>(child->payload);
    for (auto const idx : ast.temp_scalar_indices) {
      _temp_scalars[idx].reset();
    }
    for (auto const idx : ast.temp_column_indices) {
      _temp_columns[idx].reset();
    }
  }
}

std::unique_ptr<cudf::column> expression_evaluator::materialize_result_column(
  sirius::ast::node const& expr, evaluate_result result)
{
  if (expr.holds<sirius::ast::reference>()) {
    // A narrowed reference restores as a view of a column owned by the per-evaluation restoration
    // cache; an unchanged reference stays a zero-copy view of the input. Copying the view (or
    // releasing an owned column) gives the output table independent columns.
    if (result.is_owned_column()) { return result.release_column(); }
    return std::make_unique<cudf::column>(result.get_column_view(), _stream, _mr);
  }

  // Cast the `result` from libcudf to the node's return type if `result` has a different
  // type. E.g., `extract(year from col)` from libcudf returns int16_t but the SQL type is
  // int64_t. Only use cudf::cast when both types are fixed-width (cast does not support
  // STRING/LIST/STRUCT).
  auto const cudf_return_type = sirius::require_cudf_type(expr.return_type());
  std::unique_ptr<cudf::column> result_column;
  if (result.is_scalar()) {
    result_column =
      cudf::make_column_from_scalar(result.get_scalar(), _input_table.num_rows(), _stream, _mr);
  } else if (result.is_column_view()) {
    result_column = std::make_unique<cudf::column>(result.get_column_view(), _stream, _mr);
  } else {
    result_column = result.release_column();
  }
  if (result_column->type() != cudf_return_type) {
    // Cast is only valid for fixed-width types (no STRING/LIST/STRUCT/etc.).
    if (IsFixedWidth(result_column->type()) && IsFixedWidth(cudf_return_type)) {
      // Final result-type reconciliation is a physical schema restore. The declared output
      // type provides the provenance needed to restore a narrowed DATE representation.
      result_column =
        sirius::cast_through_rep(result_column->view(), cudf_return_type, _stream, _mr);
    } else {
      throw internal_exception("[expression_evaluator] Unsupported type conversion: {} to {}",
                               cudf::type_to_name(result_column->type()),
                               cudf::type_to_name(cudf_return_type));
    }
  }
  return result_column;
}

std::optional<expression_evaluator::materialized_boundary_state>
expression_evaluator::evaluate_materialization_boundaries(
  std::span<sirius::ast::node const* const> boundaries)
{
  materialized_boundary_state state;
  state.boundary_cols.reserve(boundaries.size());
  state.combined_views.reserve(_input_table.num_columns() + boundaries.size());
  for (cudf::size_type c = 0; c < _input_table.num_columns(); ++c) {
    state.combined_views.push_back(_input_table.column(c));
  }

  state.boundary_map.reserve(boundaries.size());
  for (auto const* bnode : boundaries) {
    auto b_res = evaluate(*bnode, evaluation_mode::MATERIALIZE);
    auto b_col = materialize_result_column(*bnode, std::move(b_res));
    ++_materialization_boundary_count;
    if (!nvrtc::is_supported_boundary_column(b_col->view())) { return std::nullopt; }
    auto const syn_idx = static_cast<std::uint32_t>(state.combined_views.size());
    state.combined_views.push_back(b_col->view());
    state.boundary_cols.push_back(std::move(b_col));
    state.boundary_map.emplace(bnode, syn_idx);
  }
  return state;
}

template <typename Run>
std::optional<std::invoke_result_t<Run&, cudf::table_view, nvrtc::boundary_column_map const*>>
expression_evaluator::run_compile_plan(nvrtc::compile_plan const& plan, Run&& run)
{
  if (plan.is_direct()) { return run(_input_table, nullptr); }
  auto const state = evaluate_materialization_boundaries(plan.boundaries);
  if (!state) { return std::nullopt; }
  return run(state->combined_table(), &state->boundary_map);
}

std::optional<expression_evaluator::nvrtc_projection>
expression_evaluator::try_evaluate_nvrtc_with_boundaries(
  std::span<sirius::ast::node const* const> expressions)
{
  auto plan = nvrtc::plan_project_sink(expressions, _input_table);
  if (!plan) { return std::nullopt; }
  std::vector<sirius::ast::node const*> roots;
  roots.reserve(plan->roots.size());
  for (auto const position : plan->roots) {
    roots.push_back(expressions[position]);
  }
  auto table = run_compile_plan(
    *plan, [&](cudf::table_view kernel_input, nvrtc::boundary_column_map const* boundary_map) {
      auto result = nvrtc::evaluate_project_sink(
        roots, kernel_input, _stream, _mr, nvrtc_sink_options(boundary_map));
      _narrow_domain_comparison_count += result.narrow_domain_comparison_count;
      return std::move(result.table);
    });
  if (!table) { return std::nullopt; }
  return nvrtc_projection{std::move(plan->roots), std::move(*table)};
}

nvrtc::sink_options expression_evaluator::nvrtc_sink_options(
  nvrtc::boundary_column_map const* boundary_map)
{
  return nvrtc::sink_options{
    boundary_map,
    _like_swar_fastpath,
    [this](std::string_view pattern) { return get_or_classify_like(pattern); },
    &_expression_caches->string_constants};
}

std::unique_ptr<cudf::table> expression_evaluator::evaluate(cudf::table_view input)
{
  nvtx_scoped_range nvtx_range{"expression_evaluator::evaluate"};
  _output_columns.clear();
  _output_columns.reserve(_ast_expressions.size());

  // Reset AST state from any previous invocation so that the tree, temp scalars, temp columns,
  // the restored-reference cache over them, and the per-call instrumentation do not carry stale
  // state across calls.
  _restored_reference_cache.clear();
  _hugeint_reference_bounds.clear();
  _restored_reference_cast_count  = 0;
  _narrow_domain_comparison_count = 0;
  _ast_tree                       = cudf::ast::tree{};
  _temp_scalars.clear();
  _temp_columns.clear();

  // Get the table_view from the input_batch
  _input_table = std::move(input);

  for (auto const* ast_expr : _ast_expressions) {
    if (!ast_expr) {
      // This is a Sirius bug, not a user error: a null slot means from_duckdb
      // declined an expression as unsupported, but the planner still routed it
      // to the GPU executor.  This hard-fails the query instead of falling back to CPU.
      // TODO fix to ensure the planner never builds a GPU projection containing unsupported
      // expressions.
      throw internal_exception(
        "[expression_evaluator] null expression in select list — "
        "from_duckdb returned nullptr for an unsupported expression; "
        "cannot evaluate on GPU");
    }
  }

  //===----------NVRTC_JIT mode----------===//
  // One fused kernel computes every root it can; a mixed projection list (e.g. top-level VARCHAR
  // or STRUCT expressions beside fixed-width ones) evaluates the remaining roots through the
  // materialized path.
  if (_policy.strategy == expression_evaluator_strategy::NVRTC_JIT) {
    if (auto projected = try_evaluate_nvrtc_with_boundaries(_ast_expressions)) {
      _output_columns.resize(_ast_expressions.size());
      auto compiled_columns = projected->table->release();
      for (std::size_t k = 0; k < projected->roots.size(); ++k) {
        _output_columns[projected->roots[k]] = std::move(compiled_columns[k]);
      }
      for (std::size_t i = 0; i < _ast_expressions.size(); ++i) {
        if (_output_columns[i]) { continue; }
        auto result        = evaluate(*_ast_expressions[i], evaluation_mode::MATERIALIZE);
        _output_columns[i] = materialize_result_column(*_ast_expressions[i], std::move(result));
      }
      _temp_scalars.clear();
      _temp_columns.clear();
      return std::make_unique<cudf::table>(std::move(_output_columns));
    }
  }

  // Iterate _ast_expressions and route through the std::visit dispatcher, then
  // post-process each result using the node's native kind + return type.
  for (auto const* ast_expr : _ast_expressions) {
    auto result = evaluate(*ast_expr, evaluation_mode::MATERIALIZE);
    _output_columns.push_back(materialize_result_column(*ast_expr, std::move(result)));
  }

  _temp_scalars.clear();
  _temp_columns.clear();
  return std::make_unique<cudf::table>(std::move(_output_columns));
}

std::optional<std::unique_ptr<cudf::table>> expression_evaluator::try_select_nvrtc_with_boundaries(
  cudf::table_view input,
  cudf::table_view projected_input,
  std::unique_ptr<cudf::column>* out_survivors)
{
  D_ASSERT(_ast_expressions.size() == 1);
  D_ASSERT(_ast_expressions[0] != nullptr);
  D_ASSERT(_ast_expressions[0]->return_type().id() == sirius::type_id::BOOLEAN);

  _restored_reference_cache.clear();
  _restored_reference_cast_count  = 0;
  _narrow_domain_comparison_count = 0;
  _ast_tree                       = cudf::ast::tree{};
  _temp_scalars.clear();
  _temp_columns.clear();
  _input_table = input;

  auto const* pred  = _ast_expressions[0];
  auto const finish = [&](nvrtc::filter_sink_result result) {
    _narrow_domain_comparison_count += result.narrow_domain_comparison_count;
    if (out_survivors != nullptr) { *out_survivors = std::move(result.survivors); }
    _temp_scalars.clear();
    _temp_columns.clear();
    return std::move(result.table);
  };
  auto const plan = nvrtc::plan_filter_sink(*pred, _input_table);
  if (!plan) { return std::nullopt; }
  return run_compile_plan(
    *plan, [&](cudf::table_view kernel_input, nvrtc::boundary_column_map const* boundary_map) {
      return finish(nvrtc::evaluate_filter_sink(*pred,
                                                kernel_input,
                                                projected_input,
                                                out_survivors != nullptr,
                                                _stream,
                                                _mr,
                                                nvrtc_sink_options(boundary_map)));
    });
}

std::unique_ptr<cudf::column> expression_evaluator::compute_mask(cudf::table_view input)
{
  D_ASSERT(_ast_expressions.size() == 1);
  // A filter predicate must be BOOLEAN. Read the type natively from the Sirius AST.
  D_ASSERT(_ast_expressions[0]->return_type().id() == sirius::type_id::BOOLEAN);

  return std::move(evaluate(input)->release()[0]);
}

std::unique_ptr<cudf::table> expression_evaluator::select(
  cudf::table_view input, std::span<cudf::size_type const> output_indices)
{
  nvtx_scoped_range nvtx_range{"expression_evaluator::select"};
  // A cuDF table with zero columns cannot carry a row count, so a pure-filter result with no
  // output columns (e.g. count(*) over a filtered scan) is unrepresentable here. Such callers
  // must use the all-columns select() overload so the surviving row count is preserved.
  if (output_indices.empty()) {
    throw internal_exception(
      "[expression_evaluator] select(): output_indices must be non-empty; use the "
      "all-columns select() overload for count(*)-style filters with no output columns");
  }
  auto const projected_input = input.select(output_indices.begin(), output_indices.end());
  if (_policy.strategy == expression_evaluator_strategy::NVRTC_JIT) {
    if (auto out = try_select_nvrtc_with_boundaries(input, projected_input, nullptr)) {
      return std::move(*out);
    }
  }
  auto mask = compute_mask(input);
  return sirius::apply_retention_mask(projected_input, mask->view(), _stream, _mr);
}

std::unique_ptr<cudf::table> expression_evaluator::select_with_survivors(
  cudf::table_view input,
  std::span<cudf::size_type const> output_indices,
  std::unique_ptr<cudf::column>& survivors)
{
  if (output_indices.empty()) {
    throw internal_exception(
      "[expression_evaluator] select_with_survivors(): output_indices must be non-empty");
  }
  auto const projected_input = input.select(output_indices.begin(), output_indices.end());
  if (_policy.strategy == expression_evaluator_strategy::NVRTC_JIT) {
    if (auto out = try_select_nvrtc_with_boundaries(input, projected_input, &survivors)) {
      return std::move(*out);
    }
  }
  // ONE mask, used twice: the survivors must be exactly the rows the output
  // holds, so computing the predicate a second time would risk two answers.
  auto mask     = compute_mask(input);
  auto selected = sirius::apply_retention_mask(projected_input, mask->view(), _stream, _mr);

  auto const rows = input.num_rows();
  auto positions  = cudf::sequence(rows,
                                  cudf::numeric_scalar<std::int32_t>(0, true, _stream, _mr),
                                  cudf::numeric_scalar<std::int32_t>(1, true, _stream, _mr),
                                  _stream,
                                  _mr);
  auto surviving =
    sirius::apply_retention_mask(cudf::table_view{{positions->view()}}, mask->view(), _stream, _mr);
  survivors = std::make_unique<cudf::column>(surviving->get_column(0), _stream, _mr);
  return selected;
}

std::unique_ptr<cudf::table> expression_evaluator::select(cudf::table_view input)
{
  nvtx_scoped_range nvtx_range{"expression_evaluator::select"};
  if (_policy.strategy == expression_evaluator_strategy::NVRTC_JIT) {
    if (auto out = try_select_nvrtc_with_boundaries(input, input, nullptr)) {
      return std::move(*out);
    }
  }
  auto mask = compute_mask(input);
  return sirius::apply_retention_mask(input, mask->view(), _stream, _mr);
}

evaluate_result expression_evaluator::evaluate(sirius::ast::aggregate const& /*expr*/,
                                               evaluation_mode /*mode*/)
{
  throw not_implemented_exception(
    "[expression_evaluator] aggregate nodes are not executed via the expression executor "
    "(#863).");
}

evaluate_result expression_evaluator::evaluate(sirius::ast::node const& expr, evaluation_mode mode)
{
  if (_policy.strategy == expression_evaluator_strategy::NVRTC_JIT &&
      mode == evaluation_mode::MATERIALIZE && !expr.holds<sirius::ast::reference>() &&
      !expr.holds<sirius::ast::constant>()) {
    std::array<sirius::ast::node const*, 1> single_expr{&expr};
    if (auto projected = try_evaluate_nvrtc_with_boundaries(single_expr)) {
      return evaluate_result(std::move(projected->table->release()[0]));
    }
  }

  // Every sirius::ast alternative has a matching private evaluate(alt, mode)
  // overload, so overload resolution enforces dispatch completeness at compile
  // time (a new alternative without an overload fails to compile here).
  return std::visit(
    [this, mode](auto const& alt) -> evaluate_result { return this->evaluate(alt, mode); }, expr.v);
}

}  // namespace sirius
