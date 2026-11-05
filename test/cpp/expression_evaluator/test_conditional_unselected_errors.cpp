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

// CASE, COALESCE, and AND and OR in filters must not raise errors for rows a branch or operand
// does not see (#2203, #2052).

#include "ast_test_support.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <catch.hpp>
#include <expression_evaluator/expression_evaluator.hpp>

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

using namespace sirius::expr_test;

namespace {

using sirius::expression_evaluator_strategy;
using sirius::function_id;
using sirius::logical_type;
using sirius::type_id;
using sirius::ast::case_expr;

constexpr int32_t int_max = std::numeric_limits<int32_t>::max();
constexpr int32_t int_min = std::numeric_limits<int32_t>::min();
constexpr int32_t y2000   = 10957;  // 2000-01-01 in days since the epoch

template <typename T>
using rows = std::vector<std::optional<T>>;

logical_type integer() { return logical_type::make(type_id::INTEGER); }
logical_type date() { return logical_type::make(type_id::DATE); }

template <typename T>
std::unique_ptr<cudf::column> column(cudf::type_id type, rows<T> const& values)
{
  auto const stream = cudf::get_default_stream();
  std::vector<T> payload;
  payload.reserve(values.size());
  for (auto const& value : values) {
    payload.push_back(value.value_or(T{}));
  }
  auto result = cudf::make_fixed_width_column(
    cudf::data_type{type}, values.size(), cudf::mask_state::ALL_VALID);
  if (!payload.empty()) {
    REQUIRE(cudaMemcpy(result->mutable_view().template data<T>(),
                       payload.data(),
                       payload.size() * sizeof(T),
                       cudaMemcpyHostToDevice) == cudaSuccess);
  }
  cudf::size_type null_count = 0;
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (values[i]) { continue; }
    auto const row = static_cast<cudf::size_type>(i);
    cudf::set_null_mask(result->mutable_view().null_mask(), row, row + 1, false, stream);
    ++null_count;
  }
  result->set_null_count(null_count);
  return result;
}

template <typename T>
rows<T> host_rows(cudf::column_view const& column)
{
  auto const values = copy_column_to_host<T>(column);
  auto const valids = copy_valids_to_host(column);
  rows<T> result;
  for (std::size_t i = 0; i < values.size(); ++i) {
    result.push_back(valids[i] ? std::optional<T>{values[i]} : std::nullopt);
  }
  return result;
}

struct evaluation {
  std::unique_ptr<cudf::table> table;
  std::size_t selective_evaluations;

  [[nodiscard]] cudf::column_view column() const { return table->view().column(0); }
};

evaluation evaluate(ast_node const& expr,
                    cudf::table_view input,
                    expression_evaluator_strategy strategy)
{
  sirius::expression_evaluator evaluator(expr,
                                         cudf::get_current_device_resource_ref(),
                                         cudf::get_default_stream(),
                                         sirius::expression_query_policy{strategy});
  auto table = evaluator.evaluate(input);
  return {std::move(table), evaluator.selective_evaluation_count_for_testing()};
}

std::unique_ptr<ast_node> call(function_id id, logical_type type, std::unique_ptr<ast_node> arg)
{
  std::vector<std::unique_ptr<ast_node>> args;
  args.push_back(std::move(arg));
  return make_func(id, std::move(args), std::move(type));
}

std::unique_ptr<ast_node> call(function_id id,
                               logical_type type,
                               std::unique_ptr<ast_node> left,
                               std::unique_ptr<ast_node> right)
{
  std::vector<std::unique_ptr<ast_node>> args;
  args.push_back(std::move(left));
  args.push_back(std::move(right));
  return make_func(id, std::move(args), std::move(type));
}

std::unique_ptr<ast_node> bool_const(int condition)
{
  return std::make_unique<ast_node>(
    sirius::ast::constant{condition < 0 ? sirius::value{} : sirius::value{condition == 1},
                          logical_type::make(type_id::BOOLEAN)});
}

// The DATE tests read a table of DATE d (column 0) and INTEGER n (column 1).
struct date_table {
  std::unique_ptr<cudf::column> d;
  std::unique_ptr<cudf::column> n;

  [[nodiscard]] cudf::table_view view() const { return cudf::table_view{{d->view(), n->view()}}; }
};

// Every d is 2000-01-01; adding INT_MAX days overflows.
date_table dates(rows<int32_t> const& n)
{
  return {column<int32_t>(cudf::type_id::TIMESTAMP_DAYS, rows<int32_t>(n.size(), y2000)),
          column<int32_t>(cudf::type_id::INT32, n)};
}

std::unique_ptr<ast_node> d() { return make_ref_typed(0, date()); }
std::unique_ptr<ast_node> n() { return make_ref_typed(1, integer()); }
std::unique_ptr<ast_node> d_plus_n() { return call(function_id::add, date(), d(), n()); }
std::unique_ptr<ast_node> d_minus_n() { return call(function_id::sub, date(), d(), n()); }

std::unique_ptr<ast_node> n_is(sirius::comparison_type op, int32_t value)
{
  return make_cmp(op, n(), make_int_const(value));
}

// The INTEGER tests read INTEGER i from column 0.
std::unique_ptr<ast_node> i() { return make_ref_typed(0, integer()); }
std::unique_ptr<ast_node> minus_i() { return call(function_id::negate, integer(), i()); }
std::unique_ptr<ast_node> i_is_min()
{
  return make_cmp(sirius::comparison_type::equal, i(), make_int_const(int_min));
}

std::unique_ptr<ast_node> error_call()
{
  return call(function_id::error, integer(), make_str_const("more than one row"));
}

std::unique_ptr<ast_node> both(std::unique_ptr<ast_node> first, std::unique_ptr<ast_node> second)
{
  std::vector<std::unique_ptr<ast_node>> children;
  children.push_back(std::move(first));
  children.push_back(std::move(second));
  return make_conj(sirius::ast::conjunction::kind::op_and, std::move(children));
}

std::unique_ptr<ast_node> either(std::unique_ptr<ast_node> first, std::unique_ptr<ast_node> second)
{
  std::vector<std::unique_ptr<ast_node>> children;
  children.push_back(std::move(first));
  children.push_back(std::move(second));
  return make_conj(sirius::ast::conjunction::kind::op_or, std::move(children));
}

// d + n > d, which overflows where d + n does.
std::unique_ptr<ast_node> d_plus_n_after_d()
{
  return make_cmp(sirius::comparison_type::gt, d_plus_n(), d());
}

struct selection {
  rows<int32_t> n;  ///< The n of each selected row
  std::size_t selective_evaluations;
};

// Selects the rows of @p input that pass @p predicate, as a FILTER operator does, or as a scan
// does when @p short_circuit is false.
selection filter_rows(ast_node const& predicate,
                      date_table const& input,
                      expression_evaluator_strategy strategy,
                      bool short_circuit = true)
{
  sirius::expression_evaluator evaluator(predicate,
                                         cudf::get_current_device_resource_ref(),
                                         cudf::get_default_stream(),
                                         sirius::expression_query_policy{strategy});
  if (short_circuit) { evaluator.enable_short_circuit(); }
  auto const table = evaluator.select(input.view());
  return {host_rows<int32_t>(table->view().column(1)),
          evaluator.selective_evaluation_count_for_testing()};
}

}  // namespace

TEST_CASE("CASE skips errors in branches a row does not select",
          "[expression_evaluator][conditional_errors]")
{
  auto const strategy = GENERATE(expression_evaluator_strategy::MATERIALIZE,
                                 expression_evaluator_strategy::AST_INTERPRET,
                                 expression_evaluator_strategy::AST_JIT);
  CAPTURE(strategy);

  SECTION("DATE arithmetic (#2203)")
  {
    auto const input = dates({int_max, 1});
    auto expr        = make_case(n_is(sirius::comparison_type::equal, 1), d_plus_n(), d(), date());
    auto result      = evaluate(*expr, input.view(), strategy);
    CHECK(result.column().type().id() == cudf::type_id::TIMESTAMP_DAYS);
    CHECK(host_rows<int32_t>(result.column()) == rows<int32_t>{y2000, y2000 + 1});
    CHECK(result.selective_evaluations == 1);
  }
  SECTION("negation (#2128)")
  {
    auto const input = column<int32_t>(cudf::type_id::INT32, {int_min, 1});
    auto expr        = make_case(i_is_min(), make_null_const(integer()), minus_i(), integer());
    auto result      = evaluate(*expr, cudf::table_view{{input->view()}}, strategy);
    CHECK(host_rows<int32_t>(result.column()) == rows<int32_t>{std::nullopt, -1});
    CHECK(result.selective_evaluations == 1);
  }
  SECTION("abs, which leaves the error to DuckDB")
  {
    auto const input = column<int32_t>(cudf::type_id::INT32, {int_min, -1});
    auto expr   = make_case(make_cmp(sirius::comparison_type::gt, i(), make_int_const(int_min)),
                          call(function_id::abs, integer(), i()),
                          make_int_const(0),
                          integer());
    auto result = evaluate(*expr, cudf::table_view{{input->view()}}, strategy);
    CHECK(host_rows<int32_t>(result.column()) == rows<int32_t>{0, 1});
    CHECK(result.selective_evaluations == 1);
  }
  SECTION("timestamp precision cast (#2052)")
  {
    auto const overflow  = std::numeric_limits<int64_t>::max() / 1000000 + 1;
    auto const t         = column<int64_t>(cudf::type_id::TIMESTAMP_SECONDS, {overflow, 1});
    auto const k         = column<int32_t>(cudf::type_id::INT32, {0, 1});
    auto const timestamp = logical_type::make(type_id::TIMESTAMP);
    auto expr            = make_case(
      make_cmp(sirius::comparison_type::equal, make_ref_typed(1, integer()), make_int_const(1)),
      make_cast(make_ref_typed(0, logical_type::make(type_id::TIMESTAMP_SEC)), timestamp, false),
      make_null_const(timestamp),
      timestamp);
    auto result = evaluate(*expr, cudf::table_view{{t->view(), k->view()}}, strategy);
    CHECK(host_rows<int64_t>(result.column()) == rows<int64_t>{std::nullopt, 1000000});
    CHECK(result.selective_evaluations == 1);
  }
}

TEST_CASE("COALESCE skips errors on rows an earlier argument decides",
          "[expression_evaluator][conditional_errors]")
{
  auto const strategy = GENERATE(expression_evaluator_strategy::MATERIALIZE,
                                 expression_evaluator_strategy::AST_INTERPRET,
                                 expression_evaluator_strategy::AST_JIT);
  CAPTURE(strategy);

  SECTION("negation (#2128)")
  {
    auto const i = column<int32_t>(cudf::type_id::INT32, {int_min, int_min, 3});
    auto const a = column<int32_t>(cudf::type_id::INT32, {5, std::nullopt, std::nullopt});
    auto const b = column<int32_t>(cudf::type_id::INT32, {std::nullopt, 7, std::nullopt});
    std::vector<std::unique_ptr<ast_node>> children;
    children.push_back(make_ref_typed(1, integer()));
    children.push_back(make_ref_typed(2, integer()));
    children.push_back(minus_i());
    auto expr   = make_coalesce(std::move(children), integer());
    auto result = evaluate(*expr, cudf::table_view{{i->view(), a->view(), b->view()}}, strategy);
    CHECK(host_rows<int32_t>(result.column()) == rows<int32_t>{5, 7, -3});
    CHECK(result.selective_evaluations == 1);
  }
  SECTION("timestamp precision cast (#2052)")
  {
    auto const overflow = std::numeric_limits<int64_t>::max() / 1000000 + 1;
    auto const t        = column<int64_t>(cudf::type_id::TIMESTAMP_SECONDS, {overflow, 1});
    auto const u        = column<int64_t>(cudf::type_id::TIMESTAMP_MICROSECONDS, {5, std::nullopt});
    auto const timestamp = logical_type::make(type_id::TIMESTAMP);
    std::vector<std::unique_ptr<ast_node>> children;
    children.push_back(make_ref_typed(1, timestamp));
    children.push_back(
      make_cast(make_ref_typed(0, logical_type::make(type_id::TIMESTAMP_SEC)), timestamp, false));
    auto expr   = make_coalesce(std::move(children), timestamp);
    auto result = evaluate(*expr, cudf::table_view{{t->view(), u->view()}}, strategy);
    CHECK(host_rows<int64_t>(result.column()) == rows<int64_t>{5, 1000000});
    CHECK(result.selective_evaluations == 1);
  }
}

TEST_CASE("CASE and COALESCE raise errors for the rows a branch selects",
          "[expression_evaluator][conditional_errors]")
{
  auto const strategy = GENERATE(expression_evaluator_strategy::MATERIALIZE,
                                 expression_evaluator_strategy::AST_INTERPRET,
                                 expression_evaluator_strategy::AST_JIT);
  CAPTURE(strategy);
  auto const input             = dates({int_max, 1});
  auto const date_out_of_range = Catch::Matchers::ContainsSubstring("Date out of range");

  SECTION("THEN")
  {
    auto expr = make_case(n_is(sirius::comparison_type::gt, 0), d_plus_n(), d(), date());
    REQUIRE_THROWS_WITH(evaluate(*expr, input.view(), strategy), date_out_of_range);
  }
  SECTION("ELSE")
  {
    auto expr = make_case(n_is(sirius::comparison_type::equal, 1), d(), d_plus_n(), date());
    REQUIRE_THROWS_WITH(evaluate(*expr, input.view(), strategy), date_out_of_range);
  }
  SECTION("the first WHEN, which every row reaches")
  {
    auto expr = make_case(make_cmp(sirius::comparison_type::gt, d_plus_n(), d()),
                          make_int_const(1),
                          make_int_const(0),
                          integer());
    REQUIRE_THROWS_WITH(evaluate(*expr, input.view(), strategy), date_out_of_range);
  }
  SECTION("nested CASE")
  {
    auto inner = make_case(n_is(sirius::comparison_type::gt, 1), d_plus_n(), d(), date());
    auto expr  = make_case(n_is(sirius::comparison_type::gt, 0), std::move(inner), d(), date());
    REQUIRE_THROWS_WITH(evaluate(*expr, input.view(), strategy), date_out_of_range);
  }
  SECTION("COALESCE")
  {
    auto const i = column<int32_t>(cudf::type_id::INT32, {int_min, 1});
    auto const a = column<int32_t>(cudf::type_id::INT32, {std::nullopt, 5});
    std::vector<std::unique_ptr<ast_node>> children;
    children.push_back(make_ref_typed(1, integer()));
    children.push_back(minus_i());
    auto expr = make_coalesce(std::move(children), integer());
    REQUIRE_THROWS_WITH(evaluate(*expr, cudf::table_view{{i->view(), a->view()}}, strategy),
                        Catch::Matchers::ContainsSubstring("Overflow in negation"));
  }
}

TEST_CASE("CASE keeps first-match semantics across WHEN clauses",
          "[expression_evaluator][conditional_errors]")
{
  auto const strategy = GENERATE(expression_evaluator_strategy::MATERIALIZE,
                                 expression_evaluator_strategy::AST_INTERPRET,
                                 expression_evaluator_strategy::AST_JIT);
  CAPTURE(strategy);
  // The INT_MAX row matches the first WHEN; every later branch overflows on it.
  auto const input = dates({int_max, 1, -5, std::nullopt});

  SECTION("THEN and ELSE")
  {
    std::vector<case_expr::when_then> cases;
    cases.push_back({n_is(sirius::comparison_type::equal, int_max), d()});
    cases.push_back({n_is(sirius::comparison_type::gt, 0), d_plus_n()});
    auto expr   = make_case(std::move(cases), d_minus_n(), date());
    auto result = evaluate(*expr, input.view(), strategy);
    CHECK(host_rows<int32_t>(result.column()) ==
          rows<int32_t>{y2000, y2000 + 1, y2000 + 5, std::nullopt});
    CHECK(result.selective_evaluations == 1);
  }
  SECTION("a later WHEN")
  {
    std::vector<case_expr::when_then> cases;
    cases.push_back({n_is(sirius::comparison_type::equal, int_max), make_int_const(0)});
    cases.push_back({make_cmp(sirius::comparison_type::gt, d_plus_n(), d()), make_int_const(1)});
    auto expr   = make_case(std::move(cases), make_int_const(2), integer());
    auto result = evaluate(*expr, input.view(), strategy);
    CHECK(host_rows<int32_t>(result.column()) == rows<int32_t>{0, 1, 2, 2});
    CHECK(result.selective_evaluations == 1);
  }
}

TEST_CASE("nested CASE evaluates the inner CASE only on the outer branch's rows",
          "[expression_evaluator][conditional_errors]")
{
  auto const strategy = GENERATE(expression_evaluator_strategy::MATERIALIZE,
                                 expression_evaluator_strategy::AST_INTERPRET,
                                 expression_evaluator_strategy::AST_JIT);
  CAPTURE(strategy);
  auto const input = dates({int_max, 1, -5, std::nullopt});
  auto inner       = make_case(n_is(sirius::comparison_type::gt, 0), d_plus_n(), d(), date());
  auto expr   = make_case(n_is(sirius::comparison_type::lt, 100), std::move(inner), d(), date());
  auto result = evaluate(*expr, input.view(), strategy);
  CHECK(host_rows<int32_t>(result.column()) == rows<int32_t>{y2000, y2000 + 1, y2000, y2000});
  // The inner CASE fails the outer eager pass on the INT_MAX row, which only the outer WHEN
  // excludes. The outer CASE retries, and the inner CASE evaluates selectively within that pass.
  CHECK(result.selective_evaluations == 2);
}

TEST_CASE("nested conditionals evaluate selectively once per level at any depth",
          "[expression_evaluator][conditional_errors]")
{
  auto const strategy = GENERATE(expression_evaluator_strategy::MATERIALIZE,
                                 expression_evaluator_strategy::AST_INTERPRET,
                                 expression_evaluator_strategy::AST_JIT);
  auto const depth    = GENERATE(1, 2, 8);
  CAPTURE(strategy, depth);
  // Only the outermost level excludes the failing row; every inner level selects it. Retrying at
  // every level made the work double per level.
  SECTION("CASE")
  {
    auto const input = dates({int_max, 1});
    auto expr        = d_plus_n();
    for (int level = 1; level < depth; ++level) {
      expr = make_case(n_is(sirius::comparison_type::gt, 0), std::move(expr), d(), date());
    }
    expr        = make_case(n_is(sirius::comparison_type::lt, 100), std::move(expr), d(), date());
    auto result = evaluate(*expr, input.view(), strategy);
    CHECK(host_rows<int32_t>(result.column()) == rows<int32_t>{y2000, y2000 + 1});
    CHECK(result.selective_evaluations == static_cast<std::size_t>(depth));
  }
  SECTION("COALESCE")
  {
    auto const i  = column<int32_t>(cudf::type_id::INT32, {int_min, 1});
    auto const a  = column<int32_t>(cudf::type_id::INT32, {5, std::nullopt});
    auto const b  = column<int32_t>(cudf::type_id::INT32, {std::nullopt, std::nullopt});
    auto coalesce = [](std::unique_ptr<ast_node> first, std::unique_ptr<ast_node> second) {
      std::vector<std::unique_ptr<ast_node>> children;
      children.push_back(std::move(first));
      children.push_back(std::move(second));
      return make_coalesce(std::move(children), integer());
    };
    auto expr = minus_i();
    for (int level = 1; level < depth; ++level) {
      expr = coalesce(make_ref_typed(2, integer()), std::move(expr));
    }
    expr        = coalesce(make_ref_typed(1, integer()), std::move(expr));
    auto result = evaluate(*expr, cudf::table_view{{i->view(), a->view(), b->view()}}, strategy);
    CHECK(host_rows<int32_t>(result.column()) == rows<int32_t>{5, -1});
    CHECK(result.selective_evaluations == static_cast<std::size_t>(depth));
  }
}

TEST_CASE("CASE skips errors under scalar and NULL conditions",
          "[expression_evaluator][conditional_errors]")
{
  auto const strategy = GENERATE(expression_evaluator_strategy::MATERIALIZE,
                                 expression_evaluator_strategy::AST_INTERPRET,
                                 expression_evaluator_strategy::AST_JIT);
  CAPTURE(strategy);

  SECTION("constant TRUE, FALSE and NULL WHEN")
  {
    auto const condition = GENERATE(-1, 0, 1);
    CAPTURE(condition);
    auto const input = dates({int_max, 1});
    // The branch the constant does not select overflows on the first row.
    auto expr   = condition == 1 ? make_case(bool_const(condition), d(), d_plus_n(), date())
                                 : make_case(bool_const(condition), d_plus_n(), d(), date());
    auto result = evaluate(*expr, input.view(), strategy);
    CHECK(host_rows<int32_t>(result.column()) == rows<int32_t>{y2000, y2000});
  }
  SECTION("NULL WHEN rows select ELSE")
  {
    auto const input = dates({int_max, std::nullopt, 1});
    auto expr =
      make_case(n_is(sirius::comparison_type::not_equal, int_max), d_plus_n(), d(), date());
    auto result = evaluate(*expr, input.view(), strategy);
    CHECK(host_rows<int32_t>(result.column()) == rows<int32_t>{y2000, y2000, y2000 + 1});
  }
  SECTION("scalar ELSE")
  {
    auto const input = dates({int_max, 1});
    auto expr        = make_case(
      n_is(sirius::comparison_type::equal, 1), d_plus_n(), make_null_const(date()), date());
    auto result = evaluate(*expr, input.view(), strategy);
    CHECK(host_rows<int32_t>(result.column()) == rows<int32_t>{std::nullopt, y2000 + 1});
  }
  SECTION("no row selects the failing branch")
  {
    auto const input = dates({int_max, int_max});
    auto expr        = make_case(n_is(sirius::comparison_type::equal, 1), d_plus_n(), d(), date());
    auto result      = evaluate(*expr, input.view(), strategy);
    CHECK(host_rows<int32_t>(result.column()) == rows<int32_t>{y2000, y2000});
  }
}

TEST_CASE("CASE and COALESCE keep result types and shapes when they retry",
          "[expression_evaluator][conditional_errors]")
{
  auto const strategy = GENERATE(expression_evaluator_strategy::MATERIALIZE,
                                 expression_evaluator_strategy::AST_INTERPRET,
                                 expression_evaluator_strategy::AST_JIT);
  CAPTURE(strategy);

  SECTION("VARCHAR")
  {
    auto const input   = column<int32_t>(cudf::type_id::INT32, {int_min, 1});
    auto const varchar = logical_type::make(type_id::VARCHAR);
    auto expr =
      make_case(i_is_min(), make_str_const("min"), make_cast(minus_i(), varchar, false), varchar);
    auto result = evaluate(*expr, cudf::table_view{{input->view()}}, strategy);
    CHECK(copy_string_column_to_host(result.column()) == std::vector<std::string>{"min", "-1"});
    CHECK(result.selective_evaluations == 1);
  }
  SECTION("operand of an AST expression")
  {
    auto const i  = column<int32_t>(cudf::type_id::INT32, {int_min, 1});
    auto const a  = column<int32_t>(cudf::type_id::INT32, {5, std::nullopt});
    auto plus_two = [](std::unique_ptr<ast_node> operand) {
      return call(function_id::add,
                  integer(),
                  call(function_id::add, integer(), std::move(operand), make_int_const(1)),
                  make_int_const(1));
    };
    auto const table = cudf::table_view{{i->view(), a->view()}};

    auto with_case =
      plus_two(make_case(i_is_min(), make_null_const(integer()), minus_i(), integer()));
    auto result = evaluate(*with_case, table, strategy);
    CHECK(host_rows<int32_t>(result.column()) == rows<int32_t>{std::nullopt, 1});
    CHECK(result.selective_evaluations == 1);

    std::vector<std::unique_ptr<ast_node>> children;
    children.push_back(make_ref_typed(1, integer()));
    children.push_back(minus_i());
    auto with_coalesce = plus_two(make_coalesce(std::move(children), integer()));
    result             = evaluate(*with_coalesce, table, strategy);
    CHECK(host_rows<int32_t>(result.column()) == rows<int32_t>{7, 1});
    CHECK(result.selective_evaluations == 1);
  }
  SECTION("sliced input")
  {
    // Compaction and scatter must honor the slice's offset.
    auto const input = dates({1, int_max, 1});
    auto const table = cudf::slice(input.view(), {1, 3})[0];
    auto expr        = make_case(n_is(sirius::comparison_type::equal, 1), d_plus_n(), d(), date());
    auto result      = evaluate(*expr, table, strategy);
    CHECK(host_rows<int32_t>(result.column()) == rows<int32_t>{y2000, y2000 + 1});
  }
  SECTION("empty input")
  {
    auto const input = dates({});
    auto expr        = make_case(n_is(sirius::comparison_type::equal, 1), d_plus_n(), d(), date());
    auto result      = evaluate(*expr, input.view(), strategy);
    CHECK(result.table->num_rows() == 0);
    CHECK(result.selective_evaluations == 0);
  }
}

TEST_CASE("CASE and COALESCE without failing rows do not retry",
          "[expression_evaluator][conditional_errors]")
{
  auto const strategy = GENERATE(expression_evaluator_strategy::MATERIALIZE,
                                 expression_evaluator_strategy::AST_INTERPRET,
                                 expression_evaluator_strategy::AST_JIT);
  CAPTURE(strategy);
  auto const input = dates({1, 2, std::nullopt});

  auto with_case = make_case(n_is(sirius::comparison_type::equal, 1), d_plus_n(), d(), date());
  auto result    = evaluate(*with_case, input.view(), strategy);
  CHECK(host_rows<int32_t>(result.column()) == rows<int32_t>{y2000 + 1, y2000, y2000});
  CHECK(result.selective_evaluations == 0);

  std::vector<std::unique_ptr<ast_node>> children;
  children.push_back(make_null_const(date()));
  children.push_back(d_plus_n());
  auto with_coalesce = make_coalesce(std::move(children), date());
  result             = evaluate(*with_coalesce, input.view(), strategy);
  CHECK(host_rows<int32_t>(result.column()) == rows<int32_t>{y2000 + 1, y2000 + 2, std::nullopt});
  CHECK(result.selective_evaluations == 0);
}

TEST_CASE("CASE raises error() only for rows its WHEN wins",
          "[expression_evaluator][conditional_errors]")
{
  auto const strategy = GENERATE(expression_evaluator_strategy::MATERIALIZE,
                                 expression_evaluator_strategy::AST_INTERPRET,
                                 expression_evaluator_strategy::AST_JIT);
  CAPTURE(strategy);
  auto const more_than_one_row = Catch::Matchers::ContainsSubstring("More than one row");

  SECTION("first WHEN, as DuckDB generates it")
  {
    auto make = [] {
      return make_case(n_is(sirius::comparison_type::gt, 1), error_call(), n(), integer());
    };
    auto const ok = dates({1, 1});
    auto result   = evaluate(*make(), ok.view(), strategy);
    CHECK(host_rows<int32_t>(result.column()) == rows<int32_t>{1, 1});
    CHECK(result.selective_evaluations == 0);
    auto const bad = dates({1, 2});
    REQUIRE_THROWS_WITH(evaluate(*make(), bad.view(), strategy), more_than_one_row);
  }
  SECTION("later WHEN")
  {
    auto make = [](int32_t error_value) {
      std::vector<case_expr::when_then> cases;
      cases.push_back({n_is(sirius::comparison_type::ge, 1), make_int_const(0)});
      cases.push_back({n_is(sirius::comparison_type::equal, error_value), error_call()});
      return make_case(std::move(cases), make_int_const(2), integer());
    };
    // The second WHEN is true on the first row, which the first WHEN wins.
    auto const input = dates({1, 3, -1});
    auto result      = evaluate(*make(1), input.view(), strategy);
    CHECK(host_rows<int32_t>(result.column()) == rows<int32_t>{0, 0, 2});
    REQUIRE_THROWS_WITH(evaluate(*make(-1), input.view(), strategy), more_than_one_row);
  }
}

TEST_CASE("CASE does not evaluate a branch that no row reaches",
          "[expression_evaluator][conditional_errors]")
{
  auto const strategy = GENERATE(expression_evaluator_strategy::MATERIALIZE,
                                 expression_evaluator_strategy::AST_INTERPRET,
                                 expression_evaluator_strategy::AST_JIT);
  // Branches the GPU cannot evaluate, whatever the rows: error() below the top of a branch has no
  // GPU dispatch, and a hand-written error() returns SQLNULL, which has no cuDF type.
  auto const sqlnull_error = GENERATE(false, true);
  CAPTURE(strategy, sqlnull_error);
  auto failing = [sqlnull_error]() -> std::unique_ptr<ast_node> {
    if (sqlnull_error) {
      return make_cast(call(function_id::error,
                            logical_type::make(type_id::SQLNULL),
                            make_str_const("more than one row")),
                       integer(),
                       false);
    }
    return call(function_id::add, integer(), error_call(), make_int_const(1));
  };

  SECTION("THEN")
  {
    auto expr = make_case(n_is(sirius::comparison_type::gt, 5), failing(), n(), integer());
    auto const unselected = dates({1, 2});
    auto result           = evaluate(*expr, unselected.view(), strategy);
    CHECK(host_rows<int32_t>(result.column()) == rows<int32_t>{1, 2});
    CHECK(result.selective_evaluations == 1);
    auto const selected = dates({1, 6});
    REQUIRE_THROWS(evaluate(*expr, selected.view(), strategy));
  }
  SECTION("WHEN after every row has matched")
  {
    std::vector<case_expr::when_then> cases;
    cases.push_back({n_is(sirius::comparison_type::gt, 0), n()});
    cases.push_back(
      {make_cmp(sirius::comparison_type::gt, failing(), make_int_const(0)), make_int_const(0)});
    auto expr        = make_case(std::move(cases), failing(), integer());
    auto const input = dates({1, 2});
    auto result      = evaluate(*expr, input.view(), strategy);
    CHECK(host_rows<int32_t>(result.column()) == rows<int32_t>{1, 2});
  }
}

TEST_CASE("retried CASE evaluates AST branches over compacted columns",
          "[expression_evaluator][conditional_errors]")
{
  auto const strategy = GENERATE(expression_evaluator_strategy::MATERIALIZE,
                                 expression_evaluator_strategy::AST_INTERPRET,
                                 expression_evaluator_strategy::AST_JIT);
  CAPTURE(strategy);
  // The THEN reads only j, so i and the strings column s fill their slots with placeholders while
  // the THEN runs as a cuDF AST.
  auto const i = column<int32_t>(cudf::type_id::INT32, {int_min, 4});
  auto const s = cudf::make_column_from_scalar(cudf::string_scalar("unread"), 2);
  auto const j = column<int32_t>(cudf::type_id::INT32, {10, 20});
  auto j_plus_two =
    call(function_id::add,
         integer(),
         call(function_id::add, integer(), make_ref_typed(2, integer()), make_int_const(1)),
         make_int_const(1));
  auto expr   = make_case(i_is_min(), std::move(j_plus_two), minus_i(), integer());
  auto result = evaluate(*expr, cudf::table_view{{i->view(), s->view(), j->view()}}, strategy);
  CHECK(host_rows<int32_t>(result.column()) == rows<int32_t>{12, -4});
  CHECK(result.selective_evaluations == 1);
}

TEST_CASE("CASE whose THENs are all error() returns ELSE to an AST parent",
          "[expression_evaluator][conditional_errors]")
{
  auto const strategy    = GENERATE(expression_evaluator_strategy::MATERIALIZE,
                                 expression_evaluator_strategy::AST_INTERPRET,
                                 expression_evaluator_strategy::AST_JIT);
  auto const scalar_else = GENERATE(false, true);
  CAPTURE(strategy, scalar_else);
  // DuckDB's scalar subquery shape. Its ELSE stays a view of the input or a scalar, which the AST
  // parent must still reference as a column.
  auto expr        = call(function_id::add,
                   integer(),
                   call(function_id::add,
                        integer(),
                        make_case(n_is(sirius::comparison_type::gt, 5),
                                  error_call(),
                                  scalar_else ? make_int_const(7) : n(),
                                  integer()),
                        make_int_const(1)),
                   make_int_const(1));
  auto const input = dates({1, 2});
  auto result      = evaluate(*expr, input.view(), strategy);
  CHECK(host_rows<int32_t>(result.column()) ==
        (scalar_else ? rows<int32_t>{9, 9} : rows<int32_t>{3, 4}));
}

TEST_CASE("a filter's AND and OR evaluate each operand only on the rows still undecided",
          "[expression_evaluator][conditional_errors]")
{
  auto const strategy = GENERATE(expression_evaluator_strategy::MATERIALIZE,
                                 expression_evaluator_strategy::AST_INTERPRET,
                                 expression_evaluator_strategy::AST_JIT);
  CAPTURE(strategy);
  using sirius::comparison_type;
  auto const date_out_of_range = Catch::Matchers::ContainsSubstring("Date out of range");
  // d + n overflows on the INT_MAX row.
  auto const input = dates({int_max, 1, -5, std::nullopt});

  SECTION("AND")
  {
    auto const result =
      filter_rows(*both(n_is(comparison_type::lt, 100), d_plus_n_after_d()), input, strategy);
    CHECK(result.n == rows<int32_t>{1});
    CHECK(result.selective_evaluations == 1);
  }
  SECTION("OR")
  {
    auto const result =
      filter_rows(*either(n_is(comparison_type::ge, 100), d_plus_n_after_d()), input, strategy);
    CHECK(result.n == rows<int32_t>{int_max, 1});
    CHECK(result.selective_evaluations == 1);
  }
  SECTION("OR within AND")
  {
    auto predicate    = both(n_is(comparison_type::gt, -100),
                          either(n_is(comparison_type::ge, 100), d_plus_n_after_d()));
    auto const result = filter_rows(*predicate, input, strategy);
    CHECK(result.n == rows<int32_t>{int_max, 1});
    CHECK(result.selective_evaluations == 2);
  }
  SECTION("a row that passes the earlier operands still raises")
  {
    REQUIRE_THROWS_WITH(
      filter_rows(*both(n_is(comparison_type::gt, 0), d_plus_n_after_d()), input, strategy),
      date_out_of_range);
  }
  SECTION("a scan evaluates every operand on every row")
  {
    REQUIRE_THROWS_WITH(filter_rows(*both(n_is(comparison_type::lt, 100), d_plus_n_after_d()),
                                    input,
                                    strategy,
                                    /*short_circuit=*/false),
                        date_out_of_range);
  }
  SECTION("an AND in a value evaluates every operand on every row")
  {
    REQUIRE_THROWS_WITH(
      evaluate(*both(n_is(comparison_type::lt, 100), d_plus_n_after_d()), input.view(), strategy),
      date_out_of_range);
  }
  SECTION("a WHEN's AND")
  {
    auto expr   = make_case(both(n_is(comparison_type::lt, 100), d_plus_n_after_d()),
                          make_int_const(1),
                          make_int_const(0),
                          integer());
    auto result = evaluate(*expr, input.view(), strategy);
    CHECK(host_rows<int32_t>(result.column()) == rows<int32_t>{0, 1, 0, 0});
    // The CASE retries, and the AND in its WHEN evaluates selectively within that pass.
    CHECK(result.selective_evaluations == 2);
  }
}

TEST_CASE("an eager CASE raises the errors of every row, as an aggregate FILTER does",
          "[expression_evaluator][conditional_errors]")
{
  auto const strategy = GENERATE(expression_evaluator_strategy::MATERIALIZE,
                                 expression_evaluator_strategy::AST_INTERPRET,
                                 expression_evaluator_strategy::AST_JIT);
  CAPTURE(strategy);
  using sirius::comparison_type;
  auto const date_out_of_range = Catch::Matchers::ContainsSubstring("Date out of range");
  auto const input             = dates({int_max, 1});
  auto eager                   = [](std::unique_ptr<ast_node> expr) {
    expr->get<case_expr>().eager = true;
    return expr;
  };

  SECTION("THEN")
  {
    auto make = [] {
      return make_case(n_is(comparison_type::lt, 100), d_plus_n(), make_null_const(date()), date());
    };
    auto result = evaluate(*make(), input.view(), strategy);
    CHECK(host_rows<int32_t>(result.column()) == rows<int32_t>{std::nullopt, y2000 + 1});
    REQUIRE_THROWS_WITH(evaluate(*eager(make()), input.view(), strategy), date_out_of_range);
  }
  SECTION("an AND in the WHEN")
  {
    auto expr = eager(make_case(both(n_is(comparison_type::lt, 100), d_plus_n_after_d()),
                                d(),
                                make_null_const(date()),
                                date()));
    REQUIRE_THROWS_WITH(evaluate(*expr, input.view(), strategy), date_out_of_range);
  }
  SECTION("a CASE in the THEN still evaluates its branches on their rows")
  {
    auto inner  = make_case(n_is(comparison_type::lt, 100), d_plus_n(), d(), date());
    auto expr   = eager(make_case(
      n_is(comparison_type::gt, -100), std::move(inner), make_null_const(date()), date()));
    auto result = evaluate(*expr, input.view(), strategy);
    CHECK(host_rows<int32_t>(result.column()) == rows<int32_t>{y2000, y2000 + 1});
    CHECK(result.selective_evaluations == 1);
  }
}
