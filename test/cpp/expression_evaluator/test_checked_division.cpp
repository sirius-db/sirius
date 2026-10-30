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

#include <catch.hpp>
#include <expression_evaluator/expression_evaluator.hpp>
#include <expression_evaluator/gpu_expression_translator_internal.hpp>

#include <limits>
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
  auto const function = GENERATE(sirius::function_id::int_div, sirius::function_id::mod);
  auto const divisor  = GENERATE(-1, 0);
  auto const type     = logical_type::make(type_id::INTEGER);
  auto input          = column<int32_t>(cudf::data_type{cudf::type_id::INT32},
                                        {12, std::numeric_limits<int32_t>::min()});
  auto expr           = division(function, type, make_int_const(divisor));
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
  auto const function = GENERATE(sirius::function_id::int_div, sirius::function_id::mod);
  auto const divisor  = GENERATE(-1, 0);
  auto const type     = logical_type::make(type_id::HUGEINT);
  auto input          = column<__int128_t>(cudf::data_type{cudf::type_id::DECIMAL128, 0},
                                           {std::numeric_limits<__int128_t>::min()});
  auto right =
    std::make_unique<ast_node>(sirius::ast::constant{sirius::decimal128{divisor, 0}, type});
  auto expr = division(function, type, std::move(right));
  REQUIRE_THROWS_WITH(
    evaluate(*expr, cudf::table_view{{input->view()}}, strategy),
    Catch::Matchers::ContainsSubstring("Division by zero or signed division overflow"));
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
  if (!is_null && numerator == std::numeric_limits<int32_t>::min()) {
    REQUIRE_THROWS_WITH(
      evaluate(*expr, cudf::table_view{{minus_one->view()}}, strategy),
      Catch::Matchers::ContainsSubstring("Division by zero or signed division overflow"));
  } else {
    CHECK(evaluate(*expr, cudf::table_view{{minus_one->view()}}, strategy)
            ->view()
            .column(0)
            .null_count() == (is_null ? 1 : 0));
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
