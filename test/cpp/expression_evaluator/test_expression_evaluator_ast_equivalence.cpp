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

// Byte-equivalence tests for the (now single-path) expression_evaluator.
//
// For each Sirius AST alternative, the test constructs the same expression two
// ways:
//   1. as a hand-built sirius::ast::node{<alt>{...}}            (hand-AST leg)
//   2. via sirius::ast::from_duckdb(*duck_expr)                 (translator leg)
//
// Both are executed against the same input table through the surviving
// sirius::ast::node-typed executor surface and the output columns are asserted
// byte-equivalent on host. This confirms sirius::ast::from_duckdb lowers a
// DuckDB expression into a node the executor evaluates identically to the
// hand-built node. The DuckDB-typed executor entry that this file originally
// exercised as a third leg was removed in Phase 9 (#702); the duckdb::BoundXxx
// expression here now serves only as the input to from_duckdb.
//
// The per-case scaffolding (build input, run both legs, copy to host, assert
// equal) is factored into expect_hand_eq_translated(); each TEST_CASE only
// builds the DuckDB expression and the matching hand AST.

// test
#include "ast_test_support.hpp"

#include <catch.hpp>
#include <utils/utils.hpp>

// sirius — AST types + translators + executor
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>
#include <data/data_batch_utils.hpp>
#include <data/sirius_converter_registry.hpp>
#include <expression/ast/from_duckdb.hpp>
#include <expression/ast/node.hpp>
#include <expression/ast/to_duckdb.hpp>
#include <expression/value.hpp>
#include <expression_evaluator/expression_evaluator.hpp>
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
#include <cudf/types.hpp>

#include <cuda_runtime_api.h>

// standard library
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

using namespace duckdb;
using namespace cucascade;
using namespace cucascade::memory;
using namespace sirius::expr_test;

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

using exp_executor      = ::sirius::expression_evaluator;
using exp_strategy_enum = ::sirius::expression_evaluator_strategy;
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
std::unique_ptr<cudf::table> run_one(memory_space& space,
                                     ExprPtr expr_ptr,
                                     cudf::table_view tv,
                                     exp_strategy_enum strategy)
{
  exp_executor executor(expr_ptr, get_resource_ref(space), cudf::get_default_stream(), strategy);
  return executor.evaluate(tv);
}

// DuckDB-side BoundReferenceExpression with INTEGER placeholder.
duckdb::unique_ptr<Expression> duck_int_ref(uint32_t idx)
{
  return duckdb::unique_ptr<Expression>(
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, idx)
      .release());
}

duckdb::unique_ptr<Expression> duck_int_const(int32_t v)
{
  return duckdb::unique_ptr<Expression>(
    duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(v)).release());
}

// Shared scaffolding: build an input table, run the hand-built and the
// translated AST through the executor, and assert their output columns are
// byte-equal on host. `copy` extracts the comparable host representation of the
// single output column (e.g. copy_column_to_host<int32_t>, copy_bool_column_to_host).
template <class HostCopy>
void expect_hand_eq_translated(duckdb::Expression& duck_expr,
                               sirius::ast::node& hand_ast,
                               std::vector<cudf::data_type> column_types,
                               std::vector<std::optional<std::pair<int, int>>> ranges,
                               exp_strategy_enum strategy,
                               HostCopy copy)
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  auto input = make_input_batch(*space, std::move(column_types), std::move(ranges));
  auto tv    = get_table_view(input);

  auto translated_ast = sirius::ast::from_duckdb(duck_expr);
  REQUIRE(translated_ast);

  auto hand_out       = run_one(*space, &hand_ast, tv, strategy);
  auto translated_out = run_one(*space, translated_ast.get(), tv, strategy);
  REQUIRE(hand_out);
  REQUIRE(translated_out);

  auto hand_host       = copy(hand_out->view().column(0));
  auto translated_host = copy(translated_out->view().column(0));
  REQUIRE(hand_host == translated_host);
}

// Convenience: single-INT32-column input with the given value range.
std::vector<cudf::data_type> int32_col() { return {cudf::data_type{cudf::type_id::INT32}}; }
std::vector<std::optional<std::pair<int, int>>> range(int lo, int hi)
{
  return {std::pair<int, int>{lo, hi}};
}

}  // namespace

// ============================================================================
// reference
// ============================================================================

TEST_CASE("ast_equivalence - reference identity round-trip", "[expression_evaluator_ast]")
{
  auto duck_expr =
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0);
  auto hand_ast = make_ref(0);
  expect_hand_eq_translated(
    *duck_expr, *hand_ast, int32_col(), range(0, 100), MAT, copy_column_to_host<int32_t>);
}

// ============================================================================
// constant — INTEGER and VARCHAR variants.
// ============================================================================

TEST_CASE("ast_equivalence - constant INTEGER", "[expression_evaluator_ast]")
{
  auto duck_expr = duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(42));
  auto hand_ast  = make_int_const(42);
  expect_hand_eq_translated(
    *duck_expr, *hand_ast, int32_col(), range(0, 100), MAT, copy_column_to_host<int32_t>);
}

TEST_CASE("ast_equivalence - constant VARCHAR", "[expression_evaluator_ast]")
{
  auto duck_expr = duckdb::make_uniq<BoundConstantExpression>(Value("hello"));
  auto hand_ast  = make_str_const("hello");
  expect_hand_eq_translated(
    *duck_expr, *hand_ast, int32_col(), range(0, 100), MAT, copy_string_column_to_host);
}

// ============================================================================
// comparison — three variants: MATERIALIZE EQUAL, AST_INTERPRET EQUAL,
// MATERIALIZE LESSTHAN.
// ============================================================================

TEST_CASE("ast_equivalence - comparison EQUAL (MATERIALIZE)", "[expression_evaluator_ast]")
{
  auto duck_expr = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_EQUAL, duck_int_ref(0), duck_int_const(5));
  auto hand_ast = make_cmp(sirius::comparison_type::equal, make_ref(0), make_int_const(5));
  expect_hand_eq_translated(
    *duck_expr, *hand_ast, int32_col(), range(0, 10), MAT, copy_bool_column_to_host);
}

TEST_CASE("ast_equivalence - comparison EQUAL (AST_INTERPRET)", "[expression_evaluator_ast]")
{
  auto duck_expr = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_EQUAL, duck_int_ref(0), duck_int_const(5));
  auto hand_ast = make_cmp(sirius::comparison_type::equal, make_ref(0), make_int_const(5));
  expect_hand_eq_translated(
    *duck_expr, *hand_ast, int32_col(), range(0, 10), AST_I, copy_bool_column_to_host);
}

TEST_CASE("ast_equivalence - comparison LESSTHAN (MATERIALIZE)", "[expression_evaluator_ast]")
{
  auto duck_expr = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_LESSTHAN, duck_int_ref(0), duck_int_const(5));
  auto hand_ast = make_cmp(sirius::comparison_type::lt, make_ref(0), make_int_const(5));
  expect_hand_eq_translated(
    *duck_expr, *hand_ast, int32_col(), range(0, 10), MAT, copy_bool_column_to_host);
}

// ============================================================================
// conjunction — AND (MATERIALIZE + AST_INTERPRET).
// ============================================================================

namespace {
duckdb::unique_ptr<Expression> duck_and_1_9()
{
  auto duck_lhs = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_GREATERTHAN, duck_int_ref(0), duck_int_const(1));
  auto duck_rhs = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_LESSTHAN, duck_int_ref(0), duck_int_const(9));
  auto duck_expr = duckdb::make_uniq<BoundConjunctionExpression>(ExpressionType::CONJUNCTION_AND);
  duck_expr->children.push_back(std::move(duck_lhs));
  duck_expr->children.push_back(std::move(duck_rhs));
  return duckdb::unique_ptr<Expression>(duck_expr.release());
}

std::unique_ptr<sirius::ast::node> hand_and_1_9()
{
  std::vector<std::unique_ptr<sirius::ast::node>> children;
  children.push_back(make_cmp(sirius::comparison_type::gt, make_ref(0), make_int_const(1)));
  children.push_back(make_cmp(sirius::comparison_type::lt, make_ref(0), make_int_const(9)));
  return make_conj(sirius::ast::conjunction::kind::op_and, std::move(children));
}
}  // namespace

TEST_CASE("ast_equivalence - conjunction AND (MATERIALIZE)", "[expression_evaluator_ast]")
{
  auto duck_expr = duck_and_1_9();
  auto hand_ast  = hand_and_1_9();
  expect_hand_eq_translated(
    *duck_expr, *hand_ast, int32_col(), range(0, 10), MAT, copy_bool_column_to_host);
}

TEST_CASE("ast_equivalence - conjunction AND (AST_INTERPRET)", "[expression_evaluator_ast]")
{
  auto duck_expr = duck_and_1_9();
  auto hand_ast  = hand_and_1_9();
  expect_hand_eq_translated(
    *duck_expr, *hand_ast, int32_col(), range(0, 10), AST_I, copy_bool_column_to_host);
}

// ============================================================================
// between — BoundBetweenExpression with INTEGER bounds (MATERIALIZE).
// ============================================================================

TEST_CASE("ast_equivalence - between (MATERIALIZE)", "[expression_evaluator_ast]")
{
  auto duck_expr = duckdb::make_uniq<BoundBetweenExpression>(
    duck_int_ref(0), duck_int_const(5), duck_int_const(15), /*lo=*/true, /*hi=*/true);
  auto hand_ast = make_between(make_ref(0), make_int_const(5), make_int_const(15), true, true);
  expect_hand_eq_translated(
    *duck_expr, *hand_ast, int32_col(), range(0, 20), MAT, copy_bool_column_to_host);
}

// ============================================================================
// case_expr — single WHEN/THEN + ELSE (MATERIALIZE — AST breaker).
// ============================================================================

TEST_CASE("ast_equivalence - case_expr WHEN/THEN/ELSE (MATERIALIZE)", "[expression_evaluator_ast]")
{
  auto duck_when = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_EQUAL, duck_int_ref(0), duck_int_const(5));
  BoundCaseCheck check;
  check.when_expr = std::move(duck_when);
  check.then_expr = duck_int_const(10);
  auto duck_expr  = duckdb::make_uniq<BoundCaseExpression>(LogicalType{LogicalTypeId::INTEGER});
  duck_expr->case_checks.push_back(std::move(check));
  duck_expr->else_expr = duck_int_const(0);

  std::vector<sirius::ast::case_expr::when_then> cases;
  cases.push_back(sirius::ast::case_expr::when_then{
    make_cmp(sirius::comparison_type::equal, make_ref(0), make_int_const(5)),
    make_int_const(10),
  });
  auto hand_ast = std::make_unique<sirius::ast::node>(sirius::ast::case_expr{
    std::move(cases), make_int_const(0), sirius::logical_type::make(sirius::type_id::INTEGER)});

  expect_hand_eq_translated(
    *duck_expr, *hand_ast, int32_col(), range(0, 10), MAT, copy_column_to_host<int32_t>);
}

// ============================================================================
// cast — INTEGER -> BIGINT (MATERIALIZE).
// ============================================================================

TEST_CASE("ast_equivalence - cast INTEGER->BIGINT (MATERIALIZE)", "[expression_evaluator_ast]")
{
  auto duck_expr =
    BoundCastExpression::AddDefaultCastToType(duck_int_ref(0), LogicalType{LogicalTypeId::BIGINT});
  auto hand_ast =
    make_cast(make_ref(0), sirius::logical_type::make(sirius::type_id::BIGINT), /*try_cast=*/false);
  expect_hand_eq_translated(
    *duck_expr, *hand_ast, int32_col(), range(0, 50), MAT, copy_column_to_host<int64_t>);
}

// ============================================================================
// unary_op — one TEST_CASE per kind (MATERIALIZE).
// ============================================================================

TEST_CASE("ast_equivalence - unary_op NOT (MATERIALIZE)", "[expression_evaluator_ast]")
{
  // Need a BOOLEAN-producing child; use a comparison.
  auto duck_inner = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_EQUAL, duck_int_ref(0), duck_int_const(5));
  auto duck_expr = duckdb::make_uniq<BoundOperatorExpression>(ExpressionType::OPERATOR_NOT,
                                                              LogicalType{LogicalTypeId::BOOLEAN});
  duck_expr->children.push_back(std::move(duck_inner));

  auto hand_ast =
    make_unary(sirius::ast::unary_op::kind::op_not,
               make_cmp(sirius::comparison_type::equal, make_ref(0), make_int_const(5)));

  expect_hand_eq_translated(
    *duck_expr, *hand_ast, int32_col(), range(0, 10), MAT, copy_bool_column_to_host);
}

TEST_CASE("ast_equivalence - unary_op IS_NULL (MATERIALIZE)", "[expression_evaluator_ast]")
{
  auto duck_expr = duckdb::make_uniq<BoundOperatorExpression>(ExpressionType::OPERATOR_IS_NULL,
                                                              LogicalType{LogicalTypeId::BOOLEAN});
  duck_expr->children.push_back(duck_int_ref(0));

  auto hand_ast = make_unary(sirius::ast::unary_op::kind::op_is_null, make_ref(0));

  expect_hand_eq_translated(
    *duck_expr, *hand_ast, int32_col(), range(0, 50), MAT, copy_bool_column_to_host);
}

TEST_CASE("ast_equivalence - unary_op IS_NOT_NULL (MATERIALIZE)", "[expression_evaluator_ast]")
{
  auto duck_expr = duckdb::make_uniq<BoundOperatorExpression>(ExpressionType::OPERATOR_IS_NOT_NULL,
                                                              LogicalType{LogicalTypeId::BOOLEAN});
  duck_expr->children.push_back(duck_int_ref(0));

  auto hand_ast = make_unary(sirius::ast::unary_op::kind::op_is_not_null, make_ref(0));

  expect_hand_eq_translated(
    *duck_expr, *hand_ast, int32_col(), range(0, 50), MAT, copy_bool_column_to_host);
}

TEST_CASE("ast_equivalence - unary_op TRY translation (no exec)", "[expression_evaluator_ast]")
{
  // OPERATOR_TRY is recognized by the AST surface but not executable by the
  // current expression_evaluator (it throws on dispatch through
  // operator.cpp). Verify only the translation half of the contract
  // for this kind: the translated node has the expected unary_op kind and round-
  // trips back to OPERATOR_TRY. Execution coverage is intentionally omitted until
  // the underlying OPERATOR_TRY specialization lands.
  auto duck_expr = duckdb::make_uniq<BoundOperatorExpression>(ExpressionType::OPERATOR_TRY,
                                                              LogicalType{LogicalTypeId::INTEGER});
  duck_expr->children.push_back(duck_int_ref(0));

  auto translated_ast = sirius::ast::from_duckdb(*duck_expr);
  REQUIRE(translated_ast);
  REQUIRE(translated_ast->holds<sirius::ast::unary_op>());
  REQUIRE(translated_ast->get<sirius::ast::unary_op>().op == sirius::ast::unary_op::kind::op_try);

  auto round_trip = sirius::ast::to_duckdb(*translated_ast);
  REQUIRE(round_trip);
  REQUIRE(round_trip->GetExpressionType() == ExpressionType::OPERATOR_TRY);
}

// ============================================================================
// coalesce — 2-arg INTEGER (MATERIALIZE — AST breaker).
// ============================================================================

TEST_CASE("ast_equivalence - coalesce (MATERIALIZE)", "[expression_evaluator_ast]")
{
  auto duck_expr = duckdb::make_uniq<BoundOperatorExpression>(ExpressionType::OPERATOR_COALESCE,
                                                              LogicalType{LogicalTypeId::INTEGER});
  duck_expr->children.push_back(duck_int_ref(0));
  duck_expr->children.push_back(duck_int_const(0));

  std::vector<std::unique_ptr<sirius::ast::node>> children;
  children.push_back(make_ref(0));
  children.push_back(make_int_const(0));
  auto hand_ast =
    make_coalesce(std::move(children), sirius::logical_type::make(sirius::type_id::INTEGER));

  expect_hand_eq_translated(
    *duck_expr, *hand_ast, int32_col(), range(0, 50), MAT, copy_column_to_host<int32_t>);
}

// ============================================================================
// in_list — IN (MATERIALIZE + AST_INTERPRET).
// ============================================================================

namespace {
duckdb::unique_ptr<Expression> duck_in_2_5_8()
{
  auto duck_expr = duckdb::make_uniq<BoundOperatorExpression>(ExpressionType::COMPARE_IN,
                                                              LogicalType{LogicalTypeId::BOOLEAN});
  duck_expr->children.push_back(duck_int_ref(0));
  duck_expr->children.push_back(duck_int_const(2));
  duck_expr->children.push_back(duck_int_const(5));
  duck_expr->children.push_back(duck_int_const(8));
  return duckdb::unique_ptr<Expression>(duck_expr.release());
}

std::unique_ptr<sirius::ast::node> hand_in_2_5_8()
{
  std::vector<std::unique_ptr<sirius::ast::node>> values;
  values.push_back(make_int_const(2));
  values.push_back(make_int_const(5));
  values.push_back(make_int_const(8));
  return make_in(make_ref(0), std::move(values), /*negated=*/false);
}
}  // namespace

TEST_CASE("ast_equivalence - in_list IN (MATERIALIZE)", "[expression_evaluator_ast]")
{
  auto duck_expr = duck_in_2_5_8();
  auto hand_ast  = hand_in_2_5_8();
  expect_hand_eq_translated(
    *duck_expr, *hand_ast, int32_col(), range(0, 10), MAT, copy_bool_column_to_host);
}

TEST_CASE("ast_equivalence - in_list IN (AST_INTERPRET)", "[expression_evaluator_ast]")
{
  auto duck_expr = duck_in_2_5_8();
  auto hand_ast  = hand_in_2_5_8();
  expect_hand_eq_translated(
    *duck_expr, *hand_ast, int32_col(), range(0, 10), AST_I, copy_bool_column_to_host);
}

// ============================================================================
// function_call — add (MATERIALIZE + AST_INTERPRET).
// ============================================================================

namespace {
duckdb::unique_ptr<Expression> duck_add_3()
{
  auto duck_expr = duckdb::make_uniq<BoundFunctionExpression>(
    LogicalType{LogicalTypeId::INTEGER},
    ScalarFunction(
      "+", {LogicalType::INTEGER, LogicalType::INTEGER}, LogicalType::INTEGER, nullptr),
    duckdb::vector<duckdb::unique_ptr<Expression>>{},
    nullptr);
  duck_expr->children.push_back(duck_int_ref(0));
  duck_expr->children.push_back(duck_int_const(3));
  return duckdb::unique_ptr<Expression>(duck_expr.release());
}

std::unique_ptr<sirius::ast::node> hand_add_3()
{
  std::vector<std::unique_ptr<sirius::ast::node>> args;
  args.push_back(make_ref(0));
  args.push_back(make_int_const(3));
  return make_func(sirius::function_id::add,
                   std::move(args),
                   sirius::logical_type::make(sirius::type_id::INTEGER));
}
}  // namespace

TEST_CASE("ast_equivalence - function_call add (MATERIALIZE)", "[expression_evaluator_ast]")
{
  auto duck_expr = duck_add_3();
  auto hand_ast  = hand_add_3();
  expect_hand_eq_translated(
    *duck_expr, *hand_ast, int32_col(), range(0, 50), MAT, copy_column_to_host<int32_t>);
}

TEST_CASE("ast_equivalence - function_call add (AST_INTERPRET)", "[expression_evaluator_ast]")
{
  auto duck_expr = duck_add_3();
  auto hand_ast  = hand_add_3();
  expect_hand_eq_translated(
    *duck_expr, *hand_ast, int32_col(), range(0, 50), AST_I, copy_column_to_host<int32_t>);
}
