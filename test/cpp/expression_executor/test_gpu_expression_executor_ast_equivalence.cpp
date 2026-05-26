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

// Three-leg byte-equivalence tests for the dual-path gpu_expression_executor.
//
// For each Sirius AST alternative, the test constructs the same expression
// three ways:
//   1. directly as a duckdb::BoundXxxExpression (DuckDB-direct leg)
//   2. as a hand-built sirius::ast::node{<alt>{...}}            (hand-AST leg)
//   3. via sirius::ast::from_duckdb(*duck_expr)                 (translator leg)
//
// All three are executed against the same input table and the output columns
// are asserted byte-equivalent pairwise on host. Confirms the new
// sirius::ast::node-typed executor surface produces identical output to the
// existing duckdb::Expression-typed surface for every alternative.

// test
#include <catch.hpp>
#include <utils/utils.hpp>

// sirius — AST types + translators + executor
#include <cucascade/data/gpu_data_representation.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>
#include <data/data_batch_utils.hpp>
#include <data/sirius_converter_registry.hpp>
#include <expression/ast/between.hpp>
#include <expression/ast/case_expr.hpp>
#include <expression/ast/cast.hpp>
#include <expression/ast/coalesce.hpp>
#include <expression/ast/comparison.hpp>
#include <expression/ast/conjunction.hpp>
#include <expression/ast/constant.hpp>
#include <expression/ast/from_duckdb.hpp>
#include <expression/ast/function_call.hpp>
#include <expression/ast/in_list.hpp>
#include <expression/ast/node.hpp>
#include <expression/ast/reference.hpp>
#include <expression/ast/to_duckdb.hpp>
#include <expression/ast/unary_op.hpp>
#include <expression/function_id.hpp>
#include <expression/join_condition.hpp>
#include <expression/value.hpp>
#include <expression_executor/gpu_expression_executor.hpp>
#include <helper/logical_type.hpp>
#include <memory/sirius_memory_reservation_manager.hpp>

// duckdb
#include <duckdb/common/helper.hpp>
#include <duckdb/planner/expression/bound_between_expression.hpp>
#include <duckdb/planner/expression/bound_case_expression.hpp>
#include <duckdb/planner/expression/bound_cast_expression.hpp>
#include <duckdb/planner/expression/bound_comparison_expression.hpp>
#include <duckdb/planner/expression/bound_conjunction_expression.hpp>
#include <duckdb/planner/expression/bound_constant_expression.hpp>
#include <duckdb/planner/expression/bound_function_expression.hpp>
#include <duckdb/planner/expression/bound_operator_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>

// cudf
#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <cuda_runtime_api.h>

// standard library
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace duckdb;
using namespace cucascade;
using namespace cucascade::memory;

namespace {

using memory_mgr = ::sirius::memory::sirius_memory_reservation_manager;

std::unique_ptr<memory_mgr> initialize_memory_manager()
{
  ::sirius::converter_registry::reset_for_testing();
  reservation_manager_configurator builder;
  auto constexpr gpu_capacity  = 256ull << 20;  // 256MB
  auto constexpr host_capacity = 512ull << 20;  // 512MB
  auto constexpr limit_ratio   = 0.75;
  builder.set_number_of_gpus(1)
    .set_gpu_usage_limit(gpu_capacity)
    .set_reservation_fraction_per_gpu(limit_ratio)
    .set_per_host_capacity(host_capacity)
    .use_host_per_gpu()
    .set_reservation_fraction_per_host(limit_ratio);
  auto configs = builder.build();
  auto manager = std::make_unique<memory_mgr>(std::move(configs));
  ::sirius::converter_registry::initialize();
  return manager;
}

memory_space* get_default_gpu_space()
{
  static auto manager = initialize_memory_manager();
  return const_cast<memory_space*>(manager->get_memory_space(Tier::GPU, 0));
}

rmm::device_async_resource_ref get_resource_ref(memory_space& space)
{
  return space.get_default_allocator();
}

template <typename T>
std::vector<T> copy_column_to_host(const cudf::column_view& col)
{
  std::vector<T> host(col.size());
  if (col.size() > 0) {
    cudaMemcpy(host.data(), col.data<T>(), sizeof(T) * col.size(), cudaMemcpyDeviceToHost);
  }
  return host;
}

std::vector<uint8_t> copy_bool_column_to_host(const cudf::column_view& col)
{
  std::vector<uint8_t> host(col.size());
  if (col.size() > 0) {
    cudaMemcpy(host.data(), col.head(), sizeof(uint8_t) * col.size(), cudaMemcpyDeviceToHost);
  }
  return host;
}

std::vector<std::string> copy_string_column_to_host(const cudf::column_view& col)
{
  std::vector<std::string> host;
  if (col.size() == 0) { return host; }

  cudf::strings_column_view str_col(col);
  std::vector<cudf::size_type> offsets(col.size() + 1);
  cudaMemcpy(offsets.data(),
             str_col.offsets().data<cudf::size_type>(),
             (col.size() + 1) * sizeof(cudf::size_type),
             cudaMemcpyDeviceToHost);

  std::vector<char> chars(offsets.back());
  if (!chars.empty()) {
    cudaMemcpy(chars.data(),
               str_col.chars_begin(cudf::get_default_stream()),
               offsets.back(),
               cudaMemcpyDeviceToHost);
  }

  host.reserve(col.size());
  for (cudf::size_type i = 0; i < col.size(); ++i) {
    auto start = offsets[i];
    auto end   = offsets[i + 1];
    if (chars.empty()) {
      host.emplace_back();
    } else {
      host.emplace_back(chars.data() + start, chars.data() + end);
    }
  }
  return host;
}

std::vector<bool> copy_valids_to_host(const cudf::column_view& col)
{
  std::vector<bool> valids(col.size(), true);
  if (!col.nullable() || col.null_count() == 0) { return valids; }
  auto const num_words = cudf::num_bitmask_words(col.size());
  std::vector<cudf::bitmask_type> host_mask(num_words);
  cudaMemcpy(host_mask.data(),
             col.null_mask(),
             num_words * sizeof(cudf::bitmask_type),
             cudaMemcpyDeviceToHost);
  constexpr auto bits_per_word = sizeof(cudf::bitmask_type) * 8;
  for (cudf::size_type i = 0; i < col.size(); ++i) {
    auto word = host_mask[i / bits_per_word];
    auto bit  = i % bits_per_word;
    valids[i] = ((word >> bit) & 1U) != 0U;
  }
  return valids;
}

std::shared_ptr<data_batch> make_input_batch(
  memory_space& space,
  const std::vector<cudf::data_type>& column_types,
  const std::vector<std::optional<std::pair<int, int>>>& ranges)
{
  auto mr    = get_resource_ref(space);
  auto table = ::sirius::create_cudf_table_with_random_data(
    128, column_types, ranges, cudf::get_default_stream(), mr);
  auto gpu_repr =
    std::make_unique<gpu_table_representation>(std::move(table), space, cudf::get_default_stream());
  auto batch_id = ::sirius::get_next_batch_id();
  return std::make_shared<data_batch>(batch_id, std::move(gpu_repr));
}

using exp_executor      = ::sirius::gpu_expression_executor;
using exp_strategy_enum = ::sirius::expression_executor_strategy;
auto constexpr MAT      = exp_strategy_enum::MATERIALIZE;
auto constexpr AST_I    = exp_strategy_enum::AST_INTERPRET;

// Helper — pull the cudf::table_view out of a data_batch's read-only handle.
cudf::table_view get_table_view(std::shared_ptr<data_batch> const& batch)
{
  auto input_ro = batch->to_read_only();
  return input_ro.get_data()->cast<gpu_table_representation>().get_table_view();
}

// Run a non-owning executor over the given expression pointer and return the
// resulting output table.
template <class ExprPtr>
std::unique_ptr<cudf::table> run_one(
  memory_space& space, ExprPtr expr_ptr, cudf::table_view tv, exp_strategy_enum strategy)
{
  exp_executor executor(expr_ptr, get_resource_ref(space), cudf::get_default_stream(), strategy);
  return executor.execute(tv);
}

// Small-node construction shortcuts.
std::unique_ptr<sirius::ast::node> ref_node(uint32_t idx)
{
  return std::make_unique<sirius::ast::node>(sirius::ast::reference{idx});
}

std::unique_ptr<sirius::ast::node> int_const_node(int32_t v)
{
  return std::make_unique<sirius::ast::node>(sirius::ast::constant{
    /*payload=*/sirius::value{v},
    /*return_type=*/sirius::logical_type::make(sirius::type_id::INTEGER),
  });
}

// DuckDB-side BoundReferenceExpression with INTEGER placeholder.
duckdb::unique_ptr<Expression> duck_int_ref(uint32_t idx)
{
  return duckdb::unique_ptr<Expression>(duckdb::make_uniq<BoundReferenceExpression>(
                                          LogicalType{LogicalTypeId::INTEGER}, idx)
                                          .release());
}

duckdb::unique_ptr<Expression> duck_int_const(int32_t v)
{
  return duckdb::unique_ptr<Expression>(
    duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(v)).release());
}

}  // namespace

// ============================================================================
// reference
// ============================================================================

TEST_CASE("ast_equivalence - reference identity round-trip", "[gpu_expression_executor_ast]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  auto input = make_input_batch(
    *space, {cudf::data_type{cudf::type_id::INT32}}, {std::pair<int, int>{0, 100}});
  auto tv = get_table_view(input);

  auto duck_expr =
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0);
  auto hand_ast = std::make_unique<sirius::ast::node>(sirius::ast::reference{0});
  auto translated_ast = sirius::ast::from_duckdb(*duck_expr);
  REQUIRE(translated_ast);

  auto duck_out       = run_one(*space, duck_expr.get(), tv, MAT);
  auto hand_out       = run_one(*space, hand_ast.get(), tv, MAT);
  auto translated_out = run_one(*space, translated_ast.get(), tv, MAT);
  REQUIRE(duck_out);
  REQUIRE(hand_out);
  REQUIRE(translated_out);

  auto duck_host       = copy_column_to_host<int32_t>(duck_out->view().column(0));
  auto hand_host       = copy_column_to_host<int32_t>(hand_out->view().column(0));
  auto translated_host = copy_column_to_host<int32_t>(translated_out->view().column(0));
  REQUIRE(duck_host == hand_host);
  REQUIRE(hand_host == translated_host);
}

// ============================================================================
// constant — INTEGER and VARCHAR variants.
// ============================================================================

TEST_CASE("ast_equivalence - constant INTEGER", "[gpu_expression_executor_ast]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  auto input = make_input_batch(
    *space, {cudf::data_type{cudf::type_id::INT32}}, {std::pair<int, int>{0, 100}});
  auto tv = get_table_view(input);

  auto duck_expr = duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(42));
  auto hand_ast  = std::make_unique<sirius::ast::node>(sirius::ast::constant{
    sirius::value{int32_t{42}}, sirius::logical_type::make(sirius::type_id::INTEGER)});
  auto translated_ast = sirius::ast::from_duckdb(*duck_expr);
  REQUIRE(translated_ast);

  auto duck_out       = run_one(*space, duck_expr.get(), tv, MAT);
  auto hand_out       = run_one(*space, hand_ast.get(), tv, MAT);
  auto translated_out = run_one(*space, translated_ast.get(), tv, MAT);

  auto duck_host       = copy_column_to_host<int32_t>(duck_out->view().column(0));
  auto hand_host       = copy_column_to_host<int32_t>(hand_out->view().column(0));
  auto translated_host = copy_column_to_host<int32_t>(translated_out->view().column(0));
  REQUIRE(duck_host == hand_host);
  REQUIRE(hand_host == translated_host);
}

TEST_CASE("ast_equivalence - constant VARCHAR", "[gpu_expression_executor_ast]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  auto input = make_input_batch(
    *space, {cudf::data_type{cudf::type_id::INT32}}, {std::pair<int, int>{0, 100}});
  auto tv = get_table_view(input);

  auto duck_expr = duckdb::make_uniq<BoundConstantExpression>(Value("hello"));
  auto hand_ast  = std::make_unique<sirius::ast::node>(sirius::ast::constant{
    sirius::value{std::string{"hello"}}, sirius::logical_type::make(sirius::type_id::VARCHAR)});
  auto translated_ast = sirius::ast::from_duckdb(*duck_expr);
  REQUIRE(translated_ast);

  auto duck_out       = run_one(*space, duck_expr.get(), tv, MAT);
  auto hand_out       = run_one(*space, hand_ast.get(), tv, MAT);
  auto translated_out = run_one(*space, translated_ast.get(), tv, MAT);

  auto duck_host       = copy_string_column_to_host(duck_out->view().column(0));
  auto hand_host       = copy_string_column_to_host(hand_out->view().column(0));
  auto translated_host = copy_string_column_to_host(translated_out->view().column(0));
  REQUIRE(duck_host == hand_host);
  REQUIRE(hand_host == translated_host);
}

// ============================================================================
// comparison — three variants: MATERIALIZE EQUAL, AST_INTERPRET EQUAL,
// MATERIALIZE LESSTHAN.
// ============================================================================

TEST_CASE("ast_equivalence - comparison EQUAL (MATERIALIZE)", "[gpu_expression_executor_ast]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  auto input = make_input_batch(
    *space, {cudf::data_type{cudf::type_id::INT32}}, {std::pair<int, int>{0, 10}});
  auto tv = get_table_view(input);

  auto duck_expr = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_EQUAL, duck_int_ref(0), duck_int_const(5));
  auto hand_ast = std::make_unique<sirius::ast::node>(
    sirius::ast::comparison{sirius::comparison_type::equal, ref_node(0), int_const_node(5)});
  auto translated_ast = sirius::ast::from_duckdb(*duck_expr);
  REQUIRE(translated_ast);

  auto duck_out       = run_one(*space, duck_expr.get(), tv, MAT);
  auto hand_out       = run_one(*space, hand_ast.get(), tv, MAT);
  auto translated_out = run_one(*space, translated_ast.get(), tv, MAT);

  auto duck_host       = copy_bool_column_to_host(duck_out->view().column(0));
  auto hand_host       = copy_bool_column_to_host(hand_out->view().column(0));
  auto translated_host = copy_bool_column_to_host(translated_out->view().column(0));
  REQUIRE(duck_host == hand_host);
  REQUIRE(hand_host == translated_host);
}

TEST_CASE("ast_equivalence - comparison EQUAL (AST_INTERPRET)", "[gpu_expression_executor_ast]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  auto input = make_input_batch(
    *space, {cudf::data_type{cudf::type_id::INT32}}, {std::pair<int, int>{0, 10}});
  auto tv = get_table_view(input);

  auto duck_expr = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_EQUAL, duck_int_ref(0), duck_int_const(5));
  auto hand_ast = std::make_unique<sirius::ast::node>(
    sirius::ast::comparison{sirius::comparison_type::equal, ref_node(0), int_const_node(5)});
  auto translated_ast = sirius::ast::from_duckdb(*duck_expr);
  REQUIRE(translated_ast);

  auto duck_out       = run_one(*space, duck_expr.get(), tv, AST_I);
  auto hand_out       = run_one(*space, hand_ast.get(), tv, AST_I);
  auto translated_out = run_one(*space, translated_ast.get(), tv, AST_I);

  auto duck_host       = copy_bool_column_to_host(duck_out->view().column(0));
  auto hand_host       = copy_bool_column_to_host(hand_out->view().column(0));
  auto translated_host = copy_bool_column_to_host(translated_out->view().column(0));
  REQUIRE(duck_host == hand_host);
  REQUIRE(hand_host == translated_host);
}

TEST_CASE("ast_equivalence - comparison LESSTHAN (MATERIALIZE)", "[gpu_expression_executor_ast]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  auto input = make_input_batch(
    *space, {cudf::data_type{cudf::type_id::INT32}}, {std::pair<int, int>{0, 10}});
  auto tv = get_table_view(input);

  auto duck_expr = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_LESSTHAN, duck_int_ref(0), duck_int_const(5));
  auto hand_ast = std::make_unique<sirius::ast::node>(
    sirius::ast::comparison{sirius::comparison_type::lt, ref_node(0), int_const_node(5)});
  auto translated_ast = sirius::ast::from_duckdb(*duck_expr);
  REQUIRE(translated_ast);

  auto duck_out       = run_one(*space, duck_expr.get(), tv, MAT);
  auto hand_out       = run_one(*space, hand_ast.get(), tv, MAT);
  auto translated_out = run_one(*space, translated_ast.get(), tv, MAT);

  auto duck_host       = copy_bool_column_to_host(duck_out->view().column(0));
  auto hand_host       = copy_bool_column_to_host(hand_out->view().column(0));
  auto translated_host = copy_bool_column_to_host(translated_out->view().column(0));
  REQUIRE(duck_host == hand_host);
  REQUIRE(hand_host == translated_host);
}

// ============================================================================
// conjunction — AND (MATERIALIZE + AST_INTERPRET).
// ============================================================================

TEST_CASE("ast_equivalence - conjunction AND (MATERIALIZE)", "[gpu_expression_executor_ast]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  auto input = make_input_batch(
    *space, {cudf::data_type{cudf::type_id::INT32}}, {std::pair<int, int>{0, 10}});
  auto tv = get_table_view(input);

  auto duck_lhs = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_GREATERTHAN, duck_int_ref(0), duck_int_const(1));
  auto duck_rhs = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_LESSTHAN, duck_int_ref(0), duck_int_const(9));
  auto duck_expr =
    duckdb::make_uniq<BoundConjunctionExpression>(ExpressionType::CONJUNCTION_AND);
  duck_expr->children.push_back(std::move(duck_lhs));
  duck_expr->children.push_back(std::move(duck_rhs));

  std::vector<std::unique_ptr<sirius::ast::node>> children;
  children.push_back(std::make_unique<sirius::ast::node>(
    sirius::ast::comparison{sirius::comparison_type::gt, ref_node(0), int_const_node(1)}));
  children.push_back(std::make_unique<sirius::ast::node>(
    sirius::ast::comparison{sirius::comparison_type::lt, ref_node(0), int_const_node(9)}));
  auto hand_ast = std::make_unique<sirius::ast::node>(
    sirius::ast::conjunction{sirius::ast::conjunction::kind::op_and, std::move(children)});

  auto translated_ast = sirius::ast::from_duckdb(*duck_expr);
  REQUIRE(translated_ast);

  auto duck_out       = run_one(*space, duck_expr.get(), tv, MAT);
  auto hand_out       = run_one(*space, hand_ast.get(), tv, MAT);
  auto translated_out = run_one(*space, translated_ast.get(), tv, MAT);

  auto duck_host       = copy_bool_column_to_host(duck_out->view().column(0));
  auto hand_host       = copy_bool_column_to_host(hand_out->view().column(0));
  auto translated_host = copy_bool_column_to_host(translated_out->view().column(0));
  REQUIRE(duck_host == hand_host);
  REQUIRE(hand_host == translated_host);
}

TEST_CASE("ast_equivalence - conjunction AND (AST_INTERPRET)", "[gpu_expression_executor_ast]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  auto input = make_input_batch(
    *space, {cudf::data_type{cudf::type_id::INT32}}, {std::pair<int, int>{0, 10}});
  auto tv = get_table_view(input);

  auto duck_lhs = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_GREATERTHAN, duck_int_ref(0), duck_int_const(1));
  auto duck_rhs = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_LESSTHAN, duck_int_ref(0), duck_int_const(9));
  auto duck_expr =
    duckdb::make_uniq<BoundConjunctionExpression>(ExpressionType::CONJUNCTION_AND);
  duck_expr->children.push_back(std::move(duck_lhs));
  duck_expr->children.push_back(std::move(duck_rhs));

  std::vector<std::unique_ptr<sirius::ast::node>> children;
  children.push_back(std::make_unique<sirius::ast::node>(
    sirius::ast::comparison{sirius::comparison_type::gt, ref_node(0), int_const_node(1)}));
  children.push_back(std::make_unique<sirius::ast::node>(
    sirius::ast::comparison{sirius::comparison_type::lt, ref_node(0), int_const_node(9)}));
  auto hand_ast = std::make_unique<sirius::ast::node>(
    sirius::ast::conjunction{sirius::ast::conjunction::kind::op_and, std::move(children)});

  auto translated_ast = sirius::ast::from_duckdb(*duck_expr);
  REQUIRE(translated_ast);

  auto duck_out       = run_one(*space, duck_expr.get(), tv, AST_I);
  auto hand_out       = run_one(*space, hand_ast.get(), tv, AST_I);
  auto translated_out = run_one(*space, translated_ast.get(), tv, AST_I);

  auto duck_host       = copy_bool_column_to_host(duck_out->view().column(0));
  auto hand_host       = copy_bool_column_to_host(hand_out->view().column(0));
  auto translated_host = copy_bool_column_to_host(translated_out->view().column(0));
  REQUIRE(duck_host == hand_host);
  REQUIRE(hand_host == translated_host);
}

// ============================================================================
// between — BoundBetweenExpression with INTEGER bounds (MATERIALIZE).
// ============================================================================

TEST_CASE("ast_equivalence - between (MATERIALIZE)", "[gpu_expression_executor_ast]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  auto input = make_input_batch(
    *space, {cudf::data_type{cudf::type_id::INT32}}, {std::pair<int, int>{0, 20}});
  auto tv = get_table_view(input);

  auto duck_expr = duckdb::make_uniq<BoundBetweenExpression>(
    duck_int_ref(0), duck_int_const(5), duck_int_const(15), /*lo=*/true, /*hi=*/true);
  auto hand_ast = std::make_unique<sirius::ast::node>(sirius::ast::between{
    ref_node(0), int_const_node(5), int_const_node(15), true, true});
  auto translated_ast = sirius::ast::from_duckdb(*duck_expr);
  REQUIRE(translated_ast);

  auto duck_out       = run_one(*space, duck_expr.get(), tv, MAT);
  auto hand_out       = run_one(*space, hand_ast.get(), tv, MAT);
  auto translated_out = run_one(*space, translated_ast.get(), tv, MAT);

  auto duck_host       = copy_bool_column_to_host(duck_out->view().column(0));
  auto hand_host       = copy_bool_column_to_host(hand_out->view().column(0));
  auto translated_host = copy_bool_column_to_host(translated_out->view().column(0));
  REQUIRE(duck_host == hand_host);
  REQUIRE(hand_host == translated_host);
}

// ============================================================================
// case_expr — single WHEN/THEN + ELSE (MATERIALIZE — AST breaker).
// ============================================================================

TEST_CASE("ast_equivalence - case_expr WHEN/THEN/ELSE (MATERIALIZE)",
          "[gpu_expression_executor_ast]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  auto input = make_input_batch(
    *space, {cudf::data_type{cudf::type_id::INT32}}, {std::pair<int, int>{0, 10}});
  auto tv = get_table_view(input);

  auto duck_when = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_EQUAL, duck_int_ref(0), duck_int_const(5));
  auto duck_then = duck_int_const(10);
  auto duck_else = duck_int_const(0);
  BoundCaseCheck check;
  check.when_expr = std::move(duck_when);
  check.then_expr = std::move(duck_then);
  auto duck_expr  = duckdb::make_uniq<BoundCaseExpression>(LogicalType{LogicalTypeId::INTEGER});
  duck_expr->case_checks.push_back(std::move(check));
  duck_expr->else_expr = std::move(duck_else);

  std::vector<sirius::ast::case_expr::when_then> cases;
  cases.push_back(sirius::ast::case_expr::when_then{
    std::make_unique<sirius::ast::node>(
      sirius::ast::comparison{sirius::comparison_type::equal, ref_node(0), int_const_node(5)}),
    int_const_node(10),
  });
  auto hand_ast = std::make_unique<sirius::ast::node>(
    sirius::ast::case_expr{std::move(cases), int_const_node(0)});

  auto translated_ast = sirius::ast::from_duckdb(*duck_expr);
  REQUIRE(translated_ast);

  auto duck_out       = run_one(*space, duck_expr.get(), tv, MAT);
  auto hand_out       = run_one(*space, hand_ast.get(), tv, MAT);
  auto translated_out = run_one(*space, translated_ast.get(), tv, MAT);

  auto duck_host       = copy_column_to_host<int32_t>(duck_out->view().column(0));
  auto hand_host       = copy_column_to_host<int32_t>(hand_out->view().column(0));
  auto translated_host = copy_column_to_host<int32_t>(translated_out->view().column(0));
  REQUIRE(duck_host == hand_host);
  REQUIRE(hand_host == translated_host);
}

// ============================================================================
// cast — INTEGER -> BIGINT (MATERIALIZE).
// ============================================================================

TEST_CASE("ast_equivalence - cast INTEGER->BIGINT (MATERIALIZE)", "[gpu_expression_executor_ast]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  auto input = make_input_batch(
    *space, {cudf::data_type{cudf::type_id::INT32}}, {std::pair<int, int>{0, 50}});
  auto tv = get_table_view(input);

  auto duck_expr =
    BoundCastExpression::AddDefaultCastToType(duck_int_ref(0), LogicalType{LogicalTypeId::BIGINT});
  auto hand_ast = std::make_unique<sirius::ast::node>(sirius::ast::cast{
    ref_node(0), sirius::logical_type::make(sirius::type_id::BIGINT), /*try_cast=*/false});
  auto translated_ast = sirius::ast::from_duckdb(*duck_expr);
  REQUIRE(translated_ast);

  auto duck_out       = run_one(*space, duck_expr.get(), tv, MAT);
  auto hand_out       = run_one(*space, hand_ast.get(), tv, MAT);
  auto translated_out = run_one(*space, translated_ast.get(), tv, MAT);

  auto duck_host       = copy_column_to_host<int64_t>(duck_out->view().column(0));
  auto hand_host       = copy_column_to_host<int64_t>(hand_out->view().column(0));
  auto translated_host = copy_column_to_host<int64_t>(translated_out->view().column(0));
  REQUIRE(duck_host == hand_host);
  REQUIRE(hand_host == translated_host);
}

// ============================================================================
// unary_op — one TEST_CASE per kind (MATERIALIZE).
// ============================================================================

TEST_CASE("ast_equivalence - unary_op NOT (MATERIALIZE)", "[gpu_expression_executor_ast]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  // Need a BOOLEAN-producing child; use a comparison.
  auto input = make_input_batch(
    *space, {cudf::data_type{cudf::type_id::INT32}}, {std::pair<int, int>{0, 10}});
  auto tv = get_table_view(input);

  auto duck_inner = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_EQUAL, duck_int_ref(0), duck_int_const(5));
  auto duck_expr = duckdb::make_uniq<BoundOperatorExpression>(ExpressionType::OPERATOR_NOT,
                                                              LogicalType{LogicalTypeId::BOOLEAN});
  duck_expr->children.push_back(std::move(duck_inner));

  auto hand_ast = std::make_unique<sirius::ast::node>(sirius::ast::unary_op{
    sirius::ast::unary_op::kind::op_not,
    std::make_unique<sirius::ast::node>(
      sirius::ast::comparison{sirius::comparison_type::equal, ref_node(0), int_const_node(5)}),
  });

  auto translated_ast = sirius::ast::from_duckdb(*duck_expr);
  REQUIRE(translated_ast);

  auto duck_out       = run_one(*space, duck_expr.get(), tv, MAT);
  auto hand_out       = run_one(*space, hand_ast.get(), tv, MAT);
  auto translated_out = run_one(*space, translated_ast.get(), tv, MAT);

  auto duck_host       = copy_bool_column_to_host(duck_out->view().column(0));
  auto hand_host       = copy_bool_column_to_host(hand_out->view().column(0));
  auto translated_host = copy_bool_column_to_host(translated_out->view().column(0));
  REQUIRE(duck_host == hand_host);
  REQUIRE(hand_host == translated_host);
}

TEST_CASE("ast_equivalence - unary_op IS_NULL (MATERIALIZE)", "[gpu_expression_executor_ast]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  auto input = make_input_batch(
    *space, {cudf::data_type{cudf::type_id::INT32}}, {std::pair<int, int>{0, 50}});
  auto tv = get_table_view(input);

  auto duck_expr = duckdb::make_uniq<BoundOperatorExpression>(ExpressionType::OPERATOR_IS_NULL,
                                                              LogicalType{LogicalTypeId::BOOLEAN});
  duck_expr->children.push_back(duck_int_ref(0));

  auto hand_ast = std::make_unique<sirius::ast::node>(
    sirius::ast::unary_op{sirius::ast::unary_op::kind::op_is_null, ref_node(0)});

  auto translated_ast = sirius::ast::from_duckdb(*duck_expr);
  REQUIRE(translated_ast);

  auto duck_out       = run_one(*space, duck_expr.get(), tv, MAT);
  auto hand_out       = run_one(*space, hand_ast.get(), tv, MAT);
  auto translated_out = run_one(*space, translated_ast.get(), tv, MAT);

  auto duck_host       = copy_bool_column_to_host(duck_out->view().column(0));
  auto hand_host       = copy_bool_column_to_host(hand_out->view().column(0));
  auto translated_host = copy_bool_column_to_host(translated_out->view().column(0));
  REQUIRE(duck_host == hand_host);
  REQUIRE(hand_host == translated_host);
}

TEST_CASE("ast_equivalence - unary_op IS_NOT_NULL (MATERIALIZE)", "[gpu_expression_executor_ast]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  auto input = make_input_batch(
    *space, {cudf::data_type{cudf::type_id::INT32}}, {std::pair<int, int>{0, 50}});
  auto tv = get_table_view(input);

  auto duck_expr = duckdb::make_uniq<BoundOperatorExpression>(
    ExpressionType::OPERATOR_IS_NOT_NULL, LogicalType{LogicalTypeId::BOOLEAN});
  duck_expr->children.push_back(duck_int_ref(0));

  auto hand_ast = std::make_unique<sirius::ast::node>(
    sirius::ast::unary_op{sirius::ast::unary_op::kind::op_is_not_null, ref_node(0)});

  auto translated_ast = sirius::ast::from_duckdb(*duck_expr);
  REQUIRE(translated_ast);

  auto duck_out       = run_one(*space, duck_expr.get(), tv, MAT);
  auto hand_out       = run_one(*space, hand_ast.get(), tv, MAT);
  auto translated_out = run_one(*space, translated_ast.get(), tv, MAT);

  auto duck_host       = copy_bool_column_to_host(duck_out->view().column(0));
  auto hand_host       = copy_bool_column_to_host(hand_out->view().column(0));
  auto translated_host = copy_bool_column_to_host(translated_out->view().column(0));
  REQUIRE(duck_host == hand_host);
  REQUIRE(hand_host == translated_host);
}

TEST_CASE("ast_equivalence - unary_op TRY translation (no exec)",
          "[gpu_expression_executor_ast]")
{
  // OPERATOR_TRY is recognized by the AST surface but not executable by the
  // current gpu_expression_executor (it throws on dispatch through
  // gpu_execute_operator.cpp). Verify only the translation half of the contract
  // for this kind: the three legs translate to AST shapes with matching unary_op
  // kinds. Execution coverage is intentionally omitted until the underlying
  // OPERATOR_TRY specialization lands.
  auto duck_expr = duckdb::make_uniq<BoundOperatorExpression>(ExpressionType::OPERATOR_TRY,
                                                              LogicalType{LogicalTypeId::INTEGER});
  duck_expr->children.push_back(duck_int_ref(0));

  auto translated_ast = sirius::ast::from_duckdb(*duck_expr);
  REQUIRE(translated_ast);
  REQUIRE(translated_ast->holds<sirius::ast::unary_op>());
  REQUIRE(translated_ast->get<sirius::ast::unary_op>().op ==
          sirius::ast::unary_op::kind::op_try);

  auto round_trip = sirius::ast::to_duckdb(*translated_ast);
  REQUIRE(round_trip);
  REQUIRE(round_trip->GetExpressionType() == ExpressionType::OPERATOR_TRY);
}

// ============================================================================
// coalesce — 3-arg INTEGER (MATERIALIZE — AST breaker).
// ============================================================================

TEST_CASE("ast_equivalence - coalesce (MATERIALIZE)", "[gpu_expression_executor_ast]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  auto input = make_input_batch(
    *space, {cudf::data_type{cudf::type_id::INT32}}, {std::pair<int, int>{0, 50}});
  auto tv = get_table_view(input);

  auto duck_expr = duckdb::make_uniq<BoundOperatorExpression>(
    ExpressionType::OPERATOR_COALESCE, LogicalType{LogicalTypeId::INTEGER});
  duck_expr->children.push_back(duck_int_ref(0));
  duck_expr->children.push_back(duck_int_const(0));

  std::vector<std::unique_ptr<sirius::ast::node>> children;
  children.push_back(ref_node(0));
  children.push_back(int_const_node(0));
  auto hand_ast =
    std::make_unique<sirius::ast::node>(sirius::ast::coalesce{std::move(children)});

  auto translated_ast = sirius::ast::from_duckdb(*duck_expr);
  REQUIRE(translated_ast);

  auto duck_out       = run_one(*space, duck_expr.get(), tv, MAT);
  auto hand_out       = run_one(*space, hand_ast.get(), tv, MAT);
  auto translated_out = run_one(*space, translated_ast.get(), tv, MAT);

  auto duck_host       = copy_column_to_host<int32_t>(duck_out->view().column(0));
  auto hand_host       = copy_column_to_host<int32_t>(hand_out->view().column(0));
  auto translated_host = copy_column_to_host<int32_t>(translated_out->view().column(0));
  REQUIRE(duck_host == hand_host);
  REQUIRE(hand_host == translated_host);
}

// ============================================================================
// in_list — IN + NOT IN (MATERIALIZE + AST_INTERPRET).
// ============================================================================

TEST_CASE("ast_equivalence - in_list IN (MATERIALIZE)", "[gpu_expression_executor_ast]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  auto input = make_input_batch(
    *space, {cudf::data_type{cudf::type_id::INT32}}, {std::pair<int, int>{0, 10}});
  auto tv = get_table_view(input);

  auto duck_expr = duckdb::make_uniq<BoundOperatorExpression>(ExpressionType::COMPARE_IN,
                                                              LogicalType{LogicalTypeId::BOOLEAN});
  duck_expr->children.push_back(duck_int_ref(0));
  duck_expr->children.push_back(duck_int_const(2));
  duck_expr->children.push_back(duck_int_const(5));
  duck_expr->children.push_back(duck_int_const(8));

  std::vector<std::unique_ptr<sirius::ast::node>> values;
  values.push_back(int_const_node(2));
  values.push_back(int_const_node(5));
  values.push_back(int_const_node(8));
  auto hand_ast = std::make_unique<sirius::ast::node>(
    sirius::ast::in_list{ref_node(0), std::move(values), /*negated=*/false});

  auto translated_ast = sirius::ast::from_duckdb(*duck_expr);
  REQUIRE(translated_ast);

  auto duck_out       = run_one(*space, duck_expr.get(), tv, MAT);
  auto hand_out       = run_one(*space, hand_ast.get(), tv, MAT);
  auto translated_out = run_one(*space, translated_ast.get(), tv, MAT);

  auto duck_host       = copy_bool_column_to_host(duck_out->view().column(0));
  auto hand_host       = copy_bool_column_to_host(hand_out->view().column(0));
  auto translated_host = copy_bool_column_to_host(translated_out->view().column(0));
  REQUIRE(duck_host == hand_host);
  REQUIRE(hand_host == translated_host);
}

TEST_CASE("ast_equivalence - in_list IN (AST_INTERPRET)", "[gpu_expression_executor_ast]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  auto input = make_input_batch(
    *space, {cudf::data_type{cudf::type_id::INT32}}, {std::pair<int, int>{0, 10}});
  auto tv = get_table_view(input);

  auto duck_expr = duckdb::make_uniq<BoundOperatorExpression>(ExpressionType::COMPARE_IN,
                                                              LogicalType{LogicalTypeId::BOOLEAN});
  duck_expr->children.push_back(duck_int_ref(0));
  duck_expr->children.push_back(duck_int_const(2));
  duck_expr->children.push_back(duck_int_const(5));
  duck_expr->children.push_back(duck_int_const(8));

  std::vector<std::unique_ptr<sirius::ast::node>> values;
  values.push_back(int_const_node(2));
  values.push_back(int_const_node(5));
  values.push_back(int_const_node(8));
  auto hand_ast = std::make_unique<sirius::ast::node>(
    sirius::ast::in_list{ref_node(0), std::move(values), /*negated=*/false});

  auto translated_ast = sirius::ast::from_duckdb(*duck_expr);
  REQUIRE(translated_ast);

  auto duck_out       = run_one(*space, duck_expr.get(), tv, AST_I);
  auto hand_out       = run_one(*space, hand_ast.get(), tv, AST_I);
  auto translated_out = run_one(*space, translated_ast.get(), tv, AST_I);

  auto duck_host       = copy_bool_column_to_host(duck_out->view().column(0));
  auto hand_host       = copy_bool_column_to_host(hand_out->view().column(0));
  auto translated_host = copy_bool_column_to_host(translated_out->view().column(0));
  REQUIRE(duck_host == hand_host);
  REQUIRE(hand_host == translated_host);
}

// ============================================================================
// function_call — add (MATERIALIZE + AST_INTERPRET).
// ============================================================================

TEST_CASE("ast_equivalence - function_call add (MATERIALIZE)", "[gpu_expression_executor_ast]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  auto input = make_input_batch(
    *space, {cudf::data_type{cudf::type_id::INT32}}, {std::pair<int, int>{0, 50}});
  auto tv = get_table_view(input);

  auto duck_expr = duckdb::make_uniq<BoundFunctionExpression>(
    LogicalType{LogicalTypeId::INTEGER},
    ScalarFunction("+",
                   {LogicalType::INTEGER, LogicalType::INTEGER},
                   LogicalType::INTEGER,
                   nullptr),
    duckdb::vector<duckdb::unique_ptr<Expression>>{},
    nullptr);
  duck_expr->children.push_back(duck_int_ref(0));
  duck_expr->children.push_back(duck_int_const(3));

  std::vector<std::unique_ptr<sirius::ast::node>> args;
  args.push_back(ref_node(0));
  args.push_back(int_const_node(3));
  auto hand_ast = std::make_unique<sirius::ast::node>(sirius::ast::function_call{
    sirius::function_id::add, std::move(args), sirius::logical_type::make(sirius::type_id::INTEGER)});

  auto translated_ast = sirius::ast::from_duckdb(*duck_expr);
  REQUIRE(translated_ast);

  auto duck_out       = run_one(*space, duck_expr.get(), tv, MAT);
  auto hand_out       = run_one(*space, hand_ast.get(), tv, MAT);
  auto translated_out = run_one(*space, translated_ast.get(), tv, MAT);

  auto duck_host       = copy_column_to_host<int32_t>(duck_out->view().column(0));
  auto hand_host       = copy_column_to_host<int32_t>(hand_out->view().column(0));
  auto translated_host = copy_column_to_host<int32_t>(translated_out->view().column(0));
  REQUIRE(duck_host == hand_host);
  REQUIRE(hand_host == translated_host);
}

TEST_CASE("ast_equivalence - function_call add (AST_INTERPRET)", "[gpu_expression_executor_ast]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  auto input = make_input_batch(
    *space, {cudf::data_type{cudf::type_id::INT32}}, {std::pair<int, int>{0, 50}});
  auto tv = get_table_view(input);

  auto duck_expr = duckdb::make_uniq<BoundFunctionExpression>(
    LogicalType{LogicalTypeId::INTEGER},
    ScalarFunction("+",
                   {LogicalType::INTEGER, LogicalType::INTEGER},
                   LogicalType::INTEGER,
                   nullptr),
    duckdb::vector<duckdb::unique_ptr<Expression>>{},
    nullptr);
  duck_expr->children.push_back(duck_int_ref(0));
  duck_expr->children.push_back(duck_int_const(3));

  std::vector<std::unique_ptr<sirius::ast::node>> args;
  args.push_back(ref_node(0));
  args.push_back(int_const_node(3));
  auto hand_ast = std::make_unique<sirius::ast::node>(sirius::ast::function_call{
    sirius::function_id::add, std::move(args), sirius::logical_type::make(sirius::type_id::INTEGER)});

  auto translated_ast = sirius::ast::from_duckdb(*duck_expr);
  REQUIRE(translated_ast);

  auto duck_out       = run_one(*space, duck_expr.get(), tv, AST_I);
  auto hand_out       = run_one(*space, hand_ast.get(), tv, AST_I);
  auto translated_out = run_one(*space, translated_ast.get(), tv, AST_I);

  auto duck_host       = copy_column_to_host<int32_t>(duck_out->view().column(0));
  auto hand_host       = copy_column_to_host<int32_t>(hand_out->view().column(0));
  auto translated_host = copy_column_to_host<int32_t>(translated_out->view().column(0));
  REQUIRE(duck_host == hand_host);
  REQUIRE(hand_host == translated_host);
}
