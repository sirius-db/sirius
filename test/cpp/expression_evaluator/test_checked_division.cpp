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

#include "ast_test_support.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <catch.hpp>
#include <expression_evaluator/expression_evaluator.hpp>

#include <limits>
#include <type_traits>
#include <vector>

using namespace sirius::expr_test;

namespace {

using sirius::expression_evaluator_strategy;
using sirius::logical_type;
using sirius::type_id;

template <typename T>
std::unique_ptr<cudf::column> column(cudf::data_type type,
                                     std::vector<T> const& values,
                                     std::vector<cudf::size_type> const& nulls = {})
{
  auto const stream = cudf::get_default_stream();
  auto result = cudf::make_fixed_width_column(type, values.size(), cudf::mask_state::ALL_VALID);
  if (!values.empty()) {
    REQUIRE(cudaMemcpy(result->mutable_view().template data<T>(),
                       values.data(),
                       values.size() * sizeof(T),
                       cudaMemcpyHostToDevice) == cudaSuccess);
  }
  for (auto row : nulls) {
    cudf::set_null_mask(result->mutable_view().null_mask(), row, row + 1, false, stream);
  }
  result->set_null_count(nulls.size());
  return result;
}

// The SQL type of a signed integer and its cuDF carrier; HUGEINT rides DECIMAL128 with scale 0.
template <typename T>
logical_type sql_type()
{
  if constexpr (std::is_same_v<T, int8_t>) {
    return logical_type::make(type_id::TINYINT);
  } else if constexpr (std::is_same_v<T, int16_t>) {
    return logical_type::make(type_id::SMALLINT);
  } else if constexpr (std::is_same_v<T, int32_t>) {
    return logical_type::make(type_id::INTEGER);
  } else if constexpr (std::is_same_v<T, int64_t>) {
    return logical_type::make(type_id::BIGINT);
  } else {
    static_assert(std::is_same_v<T, __int128_t>);
    return logical_type::make(type_id::HUGEINT);
  }
}

template <typename T>
cudf::data_type carrier()
{
  if constexpr (std::is_same_v<T, __int128_t>) {
    return cudf::data_type{cudf::type_id::DECIMAL128, 0};
  } else {
    return cudf::data_type{cudf::type_to_id<T>()};
  }
}

template <typename T>
std::unique_ptr<ast_node> literal(T value)
{
  if constexpr (std::is_same_v<T, __int128_t>) {
    return std::make_unique<ast_node>(
      sirius::ast::constant{sirius::decimal128{value, 0}, sql_type<T>()});
  } else {
    return std::make_unique<ast_node>(sirius::ast::constant{sirius::value{value}, sql_type<T>()});
  }
}

std::unique_ptr<ast_node> division(sirius::function_id function,
                                   logical_type type,
                                   std::unique_ptr<ast_node> right)
{
  std::vector<std::unique_ptr<ast_node>> args;
  args.push_back(make_ref_typed(0, type));
  args.push_back(std::move(right));
  return make_func(function, std::move(args), type);
}

std::unique_ptr<cudf::table> evaluate(ast_node const& expr,
                                      cudf::table_view input,
                                      expression_evaluator_strategy strategy)
{
  sirius::expression_evaluator evaluator(expr,
                                         cudf::get_current_device_resource_ref(),
                                         cudf::get_default_stream(),
                                         sirius::expression_query_policy{strategy});
  return evaluator.evaluate(input);
}

}  // namespace

TEMPLATE_TEST_CASE("checked division - safe integer column ranges preserve slices and NULLs",
                   "[expression_evaluator][checked_division]",
                   int32_t,
                   int64_t,
                   __int128_t)
{
  auto const strategy = GENERATE(expression_evaluator_strategy::MATERIALIZE,
                                 expression_evaluator_strategy::AST_INTERPRET,
                                 expression_evaluator_strategy::AST_JIT);
  auto const function = GENERATE(sirius::function_id::int_div, sirius::function_id::mod);
  auto const divisor  = static_cast<TestType>(GENERATE(3, -3));
  auto const type     = logical_type::make(std::is_same_v<TestType, int32_t>   ? type_id::INTEGER
                                       : std::is_same_v<TestType, int64_t> ? type_id::BIGINT
                                                                           : type_id::HUGEINT);
  auto const physical = std::is_same_v<TestType, int32_t> ? cudf::data_type{cudf::type_id::INT32}
                        : std::is_same_v<TestType, int64_t>
                          ? cudf::data_type{cudf::type_id::INT64}
                          : cudf::data_type{cudf::type_id::DECIMAL128, 0};
  auto const minimum  = std::numeric_limits<TestType>::min();
  auto left  = column<TestType>(physical, {minimum, minimum, 12, -12, 0, minimum, minimum}, {5});
  auto right = column<TestType>(physical, {-1, divisor, divisor, divisor, 0, divisor, -1}, {4});
  // Exclude exceptional pairs outside the slice; NULL divisor payloads do not affect its range.
  auto l      = cudf::slice(left->view(), {1, 6})[0];
  auto r      = cudf::slice(right->view(), {1, 6})[0];
  auto expr   = division(function, type, make_ref_typed(1, type));
  auto result = evaluate(*expr, cudf::table_view{{l, r}}, strategy);
  CHECK(copy_valids_to_host(result->view().column(0)) ==
        std::vector<bool>{true, true, true, false, false});
  auto const values = copy_column_to_host<TestType>(result->view().column(0));
  std::vector<TestType> numerators{minimum, 12, -12};
  for (std::size_t i = 0; i < numerators.size(); ++i) {
    CHECK(bool(values[i] == (function == sirius::function_id::int_div ? numerators[i] / divisor
                                                                      : numerators[i] % divisor)));
  }
}

TEST_CASE("checked division - unsigned column ranges preserve values above INT64_MAX",
          "[expression_evaluator][checked_division]")
{
  auto const strategy = GENERATE(expression_evaluator_strategy::MATERIALIZE,
                                 expression_evaluator_strategy::AST_INTERPRET,
                                 expression_evaluator_strategy::AST_JIT);
  auto const function = GENERATE(sirius::function_id::int_div, sirius::function_id::mod);
  auto const type     = logical_type::make(type_id::UBIGINT);
  auto const physical = cudf::data_type{cudf::type_id::UINT64};
  auto const maximum  = std::numeric_limits<uint64_t>::max();
  auto left           = column<uint64_t>(physical, {maximum, maximum, 12});
  auto right          = column<uint64_t>(physical, {maximum, 3, 0}, {2});
  auto expr           = division(function, type, make_ref_typed(1, type));
  auto result         = evaluate(*expr, cudf::table_view{{left->view(), right->view()}}, strategy);
  CHECK(copy_valids_to_host(result->view().column(0)) == std::vector<bool>{true, true, false});
  auto const values = copy_column_to_host<uint64_t>(result->view().column(0));
  CHECK(values[0] == (function == sirius::function_id::int_div ? 1 : 0));
  CHECK(values[1] == (function == sirius::function_id::int_div ? maximum / 3 : maximum % 3));
}

TEST_CASE("checked division - valid nullable and sliced column pairs",
          "[expression_evaluator][checked_division]")
{
  auto const strategy = GENERATE(expression_evaluator_strategy::MATERIALIZE,
                                 expression_evaluator_strategy::AST_INTERPRET,
                                 expression_evaluator_strategy::AST_JIT);
  auto const function = GENERATE(sirius::function_id::int_div, sirius::function_id::mod);
  auto const type     = logical_type::make(type_id::INTEGER);
  auto left           = column<int32_t>(cudf::data_type{cudf::type_id::INT32},
                                        {0, 12, 0, std::numeric_limits<int32_t>::min(), 9},
                                        {2, 3});
  auto right = column<int32_t>(cudf::data_type{cudf::type_id::INT32}, {0, 5, 0, -1, 0}, {4});
  // Row zero is invalid division, but outside the sliced input. NULL masks keep their offsets.
  auto l      = cudf::slice(left->view(), {1, 5})[0];
  auto r      = cudf::slice(right->view(), {1, 5})[0];
  auto expr   = division(function, type, make_ref_typed(1, type));
  auto result = evaluate(*expr, cudf::table_view{{l, r}}, strategy);
  REQUIRE(result->num_rows() == 4);
  CHECK(result->view().column(0).null_count() == 3);
  CHECK(copy_column_to_host<int32_t>(result->view().column(0))[0] == 2);

  auto empty   = cudf::slice(left->view(), {0, 0})[0];
  auto by_zero = division(function, type, make_int_const(0));
  CHECK(evaluate(*by_zero, cudf::table_view{{empty}}, strategy)->num_rows() == 0);
}

TEST_CASE("checked division - signed integer exceptional pairs reject before cuDF",
          "[expression_evaluator][checked_division]")
{
  auto const strategy = GENERATE(expression_evaluator_strategy::MATERIALIZE,
                                 expression_evaluator_strategy::AST_INTERPRET,
                                 expression_evaluator_strategy::AST_JIT);
  // The remainder of a minimum by -1 is 0, so only // rejects that pair.
  auto const [function, divisor] = GENERATE(table<sirius::function_id, int>({
    {sirius::function_id::int_div, -1},
    {sirius::function_id::int_div, 0},
    {sirius::function_id::mod, 0},
  }));

  auto const type = logical_type::make(type_id::INTEGER);
  auto input      = column<int32_t>(cudf::data_type{cudf::type_id::INT32},
                                    {12, std::numeric_limits<int32_t>::min()});
  auto expr       = division(function, type, make_int_const(divisor));
  REQUIRE_THROWS_WITH(
    evaluate(*expr, cudf::table_view{{input->view()}}, strategy),
    Catch::Matchers::ContainsSubstring("Division by zero or signed division overflow"));
}

TEST_CASE("checked division - HUGEINT minimum and zero reject before cuDF",
          "[expression_evaluator][checked_division][hugeint]")
{
  auto const strategy = GENERATE(expression_evaluator_strategy::MATERIALIZE,
                                 expression_evaluator_strategy::AST_INTERPRET,
                                 expression_evaluator_strategy::AST_JIT);
  // The remainder of a minimum by -1 is 0, so only // rejects that pair.
  auto const [function, divisor] = GENERATE(table<sirius::function_id, int>({
    {sirius::function_id::int_div, -1},
    {sirius::function_id::int_div, 0},
    {sirius::function_id::mod, 0},
  }));

  auto const type = logical_type::make(type_id::HUGEINT);
  auto input      = column<__int128_t>(cudf::data_type{cudf::type_id::DECIMAL128, 0},
                                       {std::numeric_limits<__int128_t>::min()});
  auto right =
    std::make_unique<ast_node>(sirius::ast::constant{sirius::decimal128{divisor, 0}, type});
  auto expr = division(function, type, std::move(right));
  REQUIRE_THROWS_WITH(
    evaluate(*expr, cudf::table_view{{input->view()}}, strategy),
    Catch::Matchers::ContainsSubstring("Division by zero or signed division overflow"));
}

TEMPLATE_TEST_CASE("checked division - the remainder of a signed minimum by minus one is zero",
                   "[expression_evaluator][checked_division]",
                   int8_t,
                   int16_t,
                   int32_t,
                   int64_t,
                   __int128_t)
{
  auto const strategy = GENERATE(expression_evaluator_strategy::MATERIALIZE,
                                 expression_evaluator_strategy::AST_INTERPRET,
                                 expression_evaluator_strategy::AST_JIT);
  auto const function = GENERATE(sirius::function_id::int_div, sirius::function_id::mod);
  auto const type     = sql_type<TestType>();
  auto const physical = carrier<TestType>();
  auto const minimum  = std::numeric_limits<TestType>::min();
  // Row zero divides by zero, but lies outside the sliced input.
  auto left  = column<TestType>(physical, {minimum, minimum, minimum, 12, minimum, 7}, {4});
  auto right = column<TestType>(physical, {0, -1, -1, -1, -1, 3}, {2});
  auto l     = cudf::slice(left->view(), {1, 6})[0];
  auto r     = cudf::slice(right->view(), {1, 6})[0];
  // Each input holds a minimum/-1 pair, which // rejects. % yields 0 on every valid row before
  // the last one.
  auto const check = [&](ast_node const& expr,
                         cudf::table_view input,
                         std::vector<bool> const& valids,
                         TestType last) {
    if (function == sirius::function_id::int_div) {
      REQUIRE_THROWS_WITH(
        evaluate(expr, input, strategy),
        Catch::Matchers::ContainsSubstring("Division by zero or signed division overflow"));
      return;
    }
    auto result = evaluate(expr, input, strategy);
    CHECK(copy_valids_to_host(result->view().column(0)) == valids);
    auto const values = copy_column_to_host<TestType>(result->view().column(0));
    REQUIRE(values.size() == valids.size());
    for (std::size_t row = 0; row + 1 < values.size(); ++row) {
      if (valids[row]) { CHECK(bool(values[row] == 0)); }
    }
    CHECK(bool(values.back() == last));
  };

  // Column operands: minimum and -1, minimum and NULL, 12 and -1, NULL and -1, 7 and 3.
  check(*division(function, type, make_ref_typed(1, type)),
        cudf::table_view{{l, r}},
        {true, false, true, false, true},
        1);
  // Scalar divisor -1 under the numerators minimum, minimum, 12, NULL and 7.
  check(*division(function, type, literal<TestType>(-1)),
        cudf::table_view{{l}},
        {true, true, true, false, true},
        0);
  // Scalar numerator minimum over the divisors -1, NULL, -1, -1 and 3.
  std::vector<std::unique_ptr<ast_node>> args;
  args.push_back(literal(minimum));
  args.push_back(make_ref_typed(0, type));
  check(*make_func(function, std::move(args), type),
        cudf::table_view{{r}},
        {true, false, true, true, true},
        static_cast<TestType>(minimum % 3));
}

TEMPLATE_TEST_CASE("checked division - a zero divisor beside a minimum by minus one rejects",
                   "[expression_evaluator][checked_division]",
                   int8_t,
                   int16_t,
                   int32_t,
                   int64_t,
                   __int128_t)
{
  auto const strategy = GENERATE(expression_evaluator_strategy::MATERIALIZE,
                                 expression_evaluator_strategy::AST_INTERPRET,
                                 expression_evaluator_strategy::AST_JIT);
  auto const function = GENERATE(sirius::function_id::int_div, sirius::function_id::mod);
  auto const type     = sql_type<TestType>();
  auto const physical = carrier<TestType>();
  auto const minimum  = std::numeric_limits<TestType>::min();
  // A valid zero divisor rejects % as it does //, whatever the other rows hold.
  auto const rejects = [&](ast_node const& expr, cudf::table_view input) {
    REQUIRE_THROWS_WITH(
      evaluate(expr, input, strategy),
      Catch::Matchers::ContainsSubstring("Division by zero or signed division overflow"));
  };
  // Column operands: 12 and 0, minimum and -1, 7 and 3.
  auto left  = column<TestType>(physical, {12, minimum, 7});
  auto right = column<TestType>(physical, {0, -1, 3});
  rejects(*division(function, type, make_ref_typed(1, type)),
          cudf::table_view{{left->view(), right->view()}});
  // Scalar numerator minimum over the same divisors.
  std::vector<std::unique_ptr<ast_node>> args;
  args.push_back(literal(minimum));
  args.push_back(make_ref_typed(0, type));
  rejects(*make_func(function, std::move(args), type), cudf::table_view{{right->view()}});

  // Under a NULL numerator the zero divisor is no error, so % yields 0 for the minimum/-1 pair.
  auto nullable    = column<TestType>(physical, {12, minimum, 7}, {0});
  auto const input = cudf::table_view{{nullable->view(), right->view()}};
  auto expr        = division(function, type, make_ref_typed(1, type));
  if (function == sirius::function_id::int_div) {
    rejects(*expr, input);
  } else {
    auto result = evaluate(*expr, input, strategy);
    CHECK(copy_valids_to_host(result->view().column(0)) == std::vector<bool>{false, true, true});
    auto const values = copy_column_to_host<TestType>(result->view().column(0));
    CHECK(bool(values[1] == 0));
    CHECK(bool(values[2] == 1));
  }
}

TEST_CASE("checked division - HUGEINT column division preserves wide values",
          "[expression_evaluator][checked_division][hugeint]")
{
  auto const strategy    = GENERATE(expression_evaluator_strategy::MATERIALIZE,
                                 expression_evaluator_strategy::AST_INTERPRET,
                                 expression_evaluator_strategy::AST_JIT);
  auto const function    = GENERATE(sirius::function_id::int_div, sirius::function_id::mod);
  auto const type        = logical_type::make(type_id::HUGEINT);
  __int128_t const value = __int128_t{4000000000000000000} * 10;
  auto left   = column<__int128_t>(cudf::data_type{cudf::type_id::DECIMAL128, 0}, {value, -value});
  auto right  = column<__int128_t>(cudf::data_type{cudf::type_id::DECIMAL128, 0}, {3, -3});
  auto expr   = division(function, type, make_ref_typed(1, type));
  auto result = evaluate(*expr, cudf::table_view{{left->view(), right->view()}}, strategy);
  auto values = copy_column_to_host<__int128_t>(result->view().column(0));
  CHECK(bool(values[0] == (function == sirius::function_id::int_div ? value / 3 : value % 3)));
  CHECK(bool(values[1] == (function == sirius::function_id::int_div ? -value / -3 : -value % -3)));
}

TEST_CASE("checked division - scalar numerator preserves NULLs and checks column divisors",
          "[expression_evaluator][checked_division]")
{
  auto const strategy  = GENERATE(expression_evaluator_strategy::MATERIALIZE,
                                 expression_evaluator_strategy::AST_INTERPRET,
                                 expression_evaluator_strategy::AST_JIT);
  auto const function  = GENERATE(sirius::function_id::int_div, sirius::function_id::mod);
  auto const numerator = GENERATE(12, std::numeric_limits<int32_t>::min());
  auto const is_null   = GENERATE(false, true);
  auto const type      = logical_type::make(type_id::INTEGER);
  std::vector<std::unique_ptr<ast_node>> args;
  args.push_back(is_null ? make_null_const(type) : make_int_const(numerator));
  args.push_back(make_ref_typed(0, type));
  auto expr = make_func(function, std::move(args), type);

  auto right  = column<int32_t>(cudf::data_type{cudf::type_id::INT32}, {0, 5, -5, 0}, {3});
  auto slice  = cudf::slice(right->view(), {1, 4})[0];
  auto result = evaluate(*expr, cudf::table_view{{slice}}, strategy);
  CHECK(copy_valids_to_host(result->view().column(0)) ==
        std::vector<bool>{!is_null, !is_null, false});
  if (!is_null) {
    auto values = copy_column_to_host<int32_t>(result->view().column(0));
    CHECK(values[0] == (function == sirius::function_id::int_div ? numerator / 5 : numerator % 5));
    CHECK(values[1] ==
          (function == sirius::function_id::int_div ? numerator / -5 : numerator % -5));
  }

  auto zero = column<int32_t>(cudf::data_type{cudf::type_id::INT32}, {0});
  if (is_null) {
    CHECK(
      evaluate(*expr, cudf::table_view{{zero->view()}}, strategy)->view().column(0).null_count() ==
      1);
  } else {
    REQUIRE_THROWS_WITH(
      evaluate(*expr, cudf::table_view{{zero->view()}}, strategy),
      Catch::Matchers::ContainsSubstring("Division by zero or signed division overflow"));
  }
  auto minus_one = column<int32_t>(cudf::data_type{cudf::type_id::INT32}, {-1});
  if (!is_null && numerator == std::numeric_limits<int32_t>::min() &&
      function == sirius::function_id::int_div) {
    REQUIRE_THROWS_WITH(
      evaluate(*expr, cudf::table_view{{minus_one->view()}}, strategy),
      Catch::Matchers::ContainsSubstring("Division by zero or signed division overflow"));
  } else {
    auto by_minus_one = evaluate(*expr, cudf::table_view{{minus_one->view()}}, strategy);
    CHECK(by_minus_one->view().column(0).null_count() == (is_null ? 1 : 0));
    if (!is_null && function == sirius::function_id::mod) {
      CHECK(copy_column_to_host<int32_t>(by_minus_one->view().column(0))[0] == 0);
    }
  }
  auto empty = cudf::slice(right->view(), {0, 0})[0];
  CHECK(evaluate(*expr, cudf::table_view{{empty}}, strategy)->num_rows() == 0);
}

TEST_CASE("checked division - scalar divisor preserves sliced and all-NULL numerators",
          "[expression_evaluator][checked_division]")
{
  auto const strategy = GENERATE(expression_evaluator_strategy::MATERIALIZE,
                                 expression_evaluator_strategy::AST_INTERPRET,
                                 expression_evaluator_strategy::AST_JIT);
  auto const function = GENERATE(sirius::function_id::int_div, sirius::function_id::mod);
  auto const type     = logical_type::make(type_id::INTEGER);
  auto input          = column<int32_t>(
    cudf::data_type{cudf::type_id::INT32},
    {std::numeric_limits<int32_t>::min(), 12, -12, std::numeric_limits<int32_t>::min()},
    {3});
  auto slice  = cudf::slice(input->view(), {1, 4})[0];
  auto expr   = division(function, type, make_int_const(-1));
  auto result = evaluate(*expr, cudf::table_view{{slice}}, strategy);
  CHECK(copy_valids_to_host(result->view().column(0)) == std::vector<bool>{true, true, false});
  auto values = copy_column_to_host<int32_t>(result->view().column(0));
  CHECK(values[0] == (function == sirius::function_id::int_div ? -12 : 0));
  CHECK(values[1] == (function == sirius::function_id::int_div ? 12 : 0));

  auto by_null = division(function, type, make_null_const(type));
  CHECK(evaluate(*by_null, cudf::table_view{{slice}}, strategy)->view().column(0).null_count() ==
        3);
  auto nulls = column<int32_t>(
    cudf::data_type{cudf::type_id::INT32}, {0, std::numeric_limits<int32_t>::min()}, {0, 1});
  auto by_zero = division(function, type, make_int_const(0));
  CHECK(evaluate(*by_zero, cudf::table_view{{nulls->view()}}, strategy)
          ->view()
          .column(0)
          .null_count() == 2);
}
