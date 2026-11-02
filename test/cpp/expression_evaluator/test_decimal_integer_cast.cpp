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

#include "ast_test_support.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <catch.hpp>
#include <expression_evaluator/decimal_integer_cast.hpp>
#include <expression_evaluator/expression_evaluator.hpp>
#include <expression_evaluator/gpu_expression_translator_internal.hpp>
#include <sirius/exception.hpp>

#include <cstdint>
#include <limits>
#include <vector>

using namespace sirius::expr_test;
using sirius::logical_type;
using sirius::type_id;
using strategy = sirius::expression_evaluator_strategy;

namespace {

auto const stream = cudf::get_default_stream();
auto const mr     = cudf::get_current_device_resource_ref();

template <typename Rep>
std::unique_ptr<cudf::column> decimals(std::vector<Rep> const& raw,
                                       int32_t scale,
                                       std::vector<bool> const& valid = {})
{
  auto const id   = sizeof(Rep) == 4   ? cudf::type_id::DECIMAL32
                    : sizeof(Rep) == 8 ? cudf::type_id::DECIMAL64
                                       : cudf::type_id::DECIMAL128;
  auto const size = static_cast<cudf::size_type>(raw.size());
  auto mask       = cudf::create_null_mask(size, cudf::mask_state::ALL_VALID, stream, mr);
  cudf::size_type null_count = 0;
  for (cudf::size_type i = 0; i < size; ++i) {
    if (!valid.empty() && !valid[i]) {
      cudf::set_null_mask(static_cast<cudf::bitmask_type*>(mask.data()), i, i + 1, false, stream);
      ++null_count;
    }
  }
  auto result = cudf::make_fixed_point_column(
    cudf::data_type{id, -scale}, size, std::move(mask), null_count, stream, mr);
  if (size > 0) {
    REQUIRE(cudaMemcpyAsync(result->mutable_view().data<Rep>(),
                            raw.data(),
                            sizeof(Rep) * size,
                            cudaMemcpyHostToDevice,
                            stream.get()) == cudaSuccess);
    stream.sync();
  }
  return result;
}

std::vector<__int128_t> integers(cudf::column_view input)
{
  auto widened = cudf::cast(input, cudf::data_type{cudf::type_id::DECIMAL128, 0}, stream, mr);
  return copy_column_to_host<__int128_t>(widened->view());
}

}  // namespace

TEMPLATE_TEST_CASE("decimal integer casts round ties away from zero under every strategy",
                   "[expression_evaluator][decimal_integer_cast]",
                   int32_t,
                   int64_t,
                   __int128_t)
{
  auto const mode = GENERATE(strategy::MATERIALIZE, strategy::AST_INTERPRET, strategy::AST_JIT);
  constexpr uint8_t width = sizeof(TestType) == 4 ? 9 : sizeof(TestType) == 8 ? 18 : 38;
  auto input =
    decimals<TestType>({-151, -150, -149, -50, -49, 0, 49, 50, 149, 150, 151, 0},
                       2,
                       {true, true, true, true, true, true, true, true, true, true, true, false});
  std::vector<__int128_t> const expected{-2, -2, -1, -1, 0, 0, 0, 1, 1, 2, 2, 0};
  for (auto const target :
       {type_id::TINYINT, type_id::SMALLINT, type_id::INTEGER, type_id::BIGINT, type_id::HUGEINT}) {
    CAPTURE(mode, target);
    auto expr = make_cast(
      make_ref_typed(0, logical_type::make_decimal(width, 2)), logical_type::make(target), false);
    sirius::expression_evaluator evaluator(
      *expr, mr, stream, sirius::expression_query_policy{mode}, 1);
    auto result = evaluator.evaluate(cudf::table_view{{input->view()}});
    auto actual = integers(result->view().column(0));
    for (std::size_t i = 0; i + 1 < expected.size(); ++i) {
      REQUIRE(actual[i] == expected[i]);
    }
    REQUIRE(
      copy_valids_to_host(result->view().column(0)) ==
      std::vector<bool>{true, true, true, true, true, true, true, true, true, true, true, false});
  }
}

TEMPLATE_TEST_CASE("decimal integer casts preserve narrow scale limits and raw extrema",
                   "[expression_evaluator][decimal_integer_cast]",
                   int32_t,
                   int64_t)
{
  constexpr int32_t scale   = sizeof(TestType) == 4 ? 9 : 18;
  constexpr int64_t divisor = sizeof(TestType) == 4 ? 1000000000LL : 1000000000000000000LL;
  constexpr auto half       = static_cast<TestType>(divisor / 2);
  auto ties   = decimals<TestType>({-half - 1, -half, -half + 1, half - 1, half, half + 1}, scale);
  auto result = sirius::cast_decimal_to_integer(
    ties->view(), logical_type::make(type_id::BIGINT), false, stream, mr);
  REQUIRE(integers(result->view()) == std::vector<__int128_t>{-1, -1, 0, 0, 1, 1});

  constexpr auto minimum = std::numeric_limits<TestType>::min();
  constexpr auto maximum = std::numeric_limits<TestType>::max();
  auto whole             = decimals<TestType>({minimum, maximum}, 0);
  result                 = sirius::cast_decimal_to_integer(
    whole->view(), logical_type::make(type_id::BIGINT), false, stream, mr);
  REQUIRE(integers(result->view()) == std::vector<__int128_t>{minimum, maximum});

  auto tenths = decimals<TestType>({minimum, maximum}, 1);
  result      = sirius::cast_decimal_to_integer(
    tenths->view(), logical_type::make(type_id::BIGINT), false, stream, mr);
  constexpr int64_t rounded_maximum = sizeof(TestType) == 4 ? 214748365LL : 922337203685477581LL;
  REQUIRE(integers(result->view()) == std::vector<__int128_t>{-rounded_maximum, rounded_maximum});
}

TEMPLATE_TEST_CASE("decimal integer casts preserve high scales in narrow carriers",
                   "[expression_evaluator][decimal_integer_cast]",
                   int32_t,
                   int64_t)
{
  constexpr auto minimum  = std::numeric_limits<TestType>::min();
  constexpr auto maximum  = std::numeric_limits<TestType>::max();
  constexpr int32_t scale = sizeof(TestType) == 4 ? 10 : 19;
  auto input              = decimals<TestType>({minimum, maximum}, scale);
  auto result             = sirius::cast_decimal_to_integer(
    input->view(), logical_type::make(type_id::BIGINT), false, stream, mr);
  constexpr int64_t rounded_maximum = sizeof(TestType) == 4 ? 0 : 1;
  REQUIRE(integers(result->view()) == std::vector<__int128_t>{-rounded_maximum, rounded_maximum});

  if constexpr (sizeof(TestType) == 8) {
    constexpr int64_t half = 5000000000000000000LL;
    auto ties = decimals<int64_t>({-half - 1, -half, -half + 1, half - 1, half, half + 1}, 19);
    result    = sirius::cast_decimal_to_integer(
      ties->view(), logical_type::make(type_id::BIGINT), false, stream, mr);
    REQUIRE(integers(result->view()) == std::vector<__int128_t>{-1, -1, 0, 0, 1, 1});
  }

  auto tiny = decimals<TestType>({minimum, maximum}, 38);
  result    = sirius::cast_decimal_to_integer(
    tiny->view(), logical_type::make(type_id::BIGINT), false, stream, mr);
  REQUIRE(integers(result->view()) == std::vector<__int128_t>{0, 0});

  for (auto mode : {strategy::MATERIALIZE, strategy::AST_INTERPRET, strategy::AST_JIT}) {
    CAPTURE(mode);
    auto expr = make_cast(make_ref_typed(0, logical_type::make_decimal(38, 38)),
                          logical_type::make(type_id::BIGINT),
                          false);
    sirius::expression_evaluator evaluator(
      *expr, mr, stream, sirius::expression_query_policy{mode}, 1);
    auto evaluated = evaluator.evaluate(cudf::table_view{{tiny->view()}});
    REQUIRE(integers(evaluated->view().column(0)) == std::vector<__int128_t>{0, 0});
  }
}

TEST_CASE("decimal integer casts preserve wide integers and scale 38",
          "[expression_evaluator][decimal_integer_cast]")
{
  auto const base = static_cast<__int128_t>(1) << 65;
  auto input =
    decimals<__int128_t>({base * 100 + 49, base * 100 + 50, -base * 100 - 49, -base * 100 - 50}, 2);
  auto result = sirius::cast_decimal_to_integer(
    input->view(), logical_type::make(type_id::HUGEINT), false, stream, mr);
  REQUIRE(integers(result->view()) == std::vector<__int128_t>{base, base + 1, -base, -base - 1});
  __int128_t power = 1;
  for (int i = 0; i < 38; ++i) {
    power *= 10;
  }
  auto fractions = decimals<__int128_t>(
    {power / 2 - 1, power / 2, power - 1, -power / 2 + 1, -power / 2, -power + 1}, 38);
  result = sirius::cast_decimal_to_integer(
    fractions->view(), logical_type::make(type_id::BIGINT), false, stream, mr);
  REQUIRE(integers(result->view()) == std::vector<__int128_t>{0, 1, 1, 0, -1, -1});

  auto whole = decimals<__int128_t>({power - 1, -power + 1}, 0);
  result     = sirius::cast_decimal_to_integer(
    whole->view(), logical_type::make(type_id::HUGEINT), false, stream, mr);
  REQUIRE(integers(result->view()) == std::vector<__int128_t>{power - 1, -power + 1});
}

TEST_CASE("decimal integer casts check rounded signed and unsigned boundaries",
          "[expression_evaluator][decimal_integer_cast]")
{
  auto signed_input =
    decimals<int32_t>({12749, 12750, -12849, -12850, 0}, 2, {true, true, true, true, false});
  auto result = sirius::cast_decimal_to_integer(
    signed_input->view(), logical_type::make(type_id::TINYINT), true, stream, mr);
  REQUIRE(copy_valids_to_host(result->view()) ==
          std::vector<bool>{true, false, true, false, false});
  auto values = integers(result->view());
  CHECK(values[0] == 127);
  CHECK(values[2] == -128);
  REQUIRE_THROWS_AS(
    sirius::cast_decimal_to_integer(
      signed_input->view(), logical_type::make(type_id::TINYINT), false, stream, mr),
    sirius::not_implemented_exception);

  for (auto const target : {type_id::UTINYINT,
                            type_id::USMALLINT,
                            type_id::UINTEGER,
                            type_id::UBIGINT,
                            type_id::UHUGEINT}) {
    auto negative = decimals<int32_t>({-49, -50, 49, 50}, 2);
    result        = sirius::cast_decimal_to_integer(
      negative->view(), logical_type::make(target), true, stream, mr);
    REQUIRE(copy_valids_to_host(result->view()) == std::vector<bool>{true, false, true, true});
    values = integers(result->view());
    CHECK(values[0] == 0);
    CHECK(values[2] == 0);
    CHECK(values[3] == 1);
  }
  auto const maximum  = static_cast<__int128_t>(std::numeric_limits<uint64_t>::max());
  auto unsigned_input = decimals<__int128_t>({maximum * 100 + 49, maximum * 100 + 50}, 2);
  result              = sirius::cast_decimal_to_integer(
    unsigned_input->view(), logical_type::make(type_id::UBIGINT), true, stream, mr);
  REQUIRE(copy_valids_to_host(result->view()) == std::vector<bool>{true, false});
  CHECK(integers(result->view())[0] == maximum);
  // The second row is valid UHUGEINT, so TRY_CAST must replay, never replace it with NULL.
  REQUIRE_THROWS_AS(
    sirius::cast_decimal_to_integer(
      unsigned_input->view(), logical_type::make(type_id::UHUGEINT), true, stream, mr),
    sirius::not_implemented_exception);
}

TEST_CASE("decimal integer casts handle sliced masks empty columns and all NULLs",
          "[expression_evaluator][decimal_integer_cast]")
{
  auto input =
    decimals<int64_t>({999, 150, -150, 0, -49, 999}, 2, {true, true, true, false, true, true});
  auto slice = cudf::slice(input->view(), {1, 5}, stream).front();
  auto result =
    sirius::cast_decimal_to_integer(slice, logical_type::make(type_id::BIGINT), false, stream, mr);
  REQUIRE(copy_valids_to_host(result->view()) == std::vector<bool>{true, true, false, true});
  auto values = integers(result->view());
  CHECK(values[0] == 2);
  CHECK(values[1] == -2);
  CHECK(values[3] == 0);
  auto empty = decimals<int32_t>({}, 2);
  result     = sirius::cast_decimal_to_integer(
    empty->view(), logical_type::make(type_id::HUGEINT), false, stream, mr);
  REQUIRE(result->size() == 0);
  REQUIRE(result->type() == cudf::data_type{cudf::type_id::DECIMAL128, 0});
  auto nulls = decimals<int32_t>({9999999, -9999999}, 2, {false, false});
  result     = sirius::cast_decimal_to_integer(
    nulls->view(), logical_type::make(type_id::TINYINT), false, stream, mr);
  REQUIRE(result->null_count() == 2);
}

TEST_CASE("decimal casts compose in ASTs while standalone translation declines integer casts",
          "[expression_evaluator][expression_translator][decimal_integer_cast]")
{
  auto const mode = GENERATE(strategy::MATERIALIZE, strategy::AST_INTERPRET, strategy::AST_JIT);
  auto input      = decimals<int64_t>({149, 150, -150}, 2);
  for (auto const target : {type_id::BIGINT, type_id::UBIGINT}) {
    auto cast = make_cast(
      make_ref_typed(0, logical_type::make_decimal(18, 2)), logical_type::make(target), true);
    REQUIRE(cast->cudf_ast_op_count() == 0);
    sirius::gpu_expression_translator translator(stream, mr);
    REQUIRE_FALSE(translator.translate_expression(*cast).has_value());
  }
  auto cast = make_cast(make_ref_typed(0, logical_type::make_decimal(18, 2)),
                        logical_type::make(type_id::BIGINT),
                        false);
  std::vector<std::unique_ptr<ast_node>> args;
  args.push_back(std::move(cast));
  args.push_back(make_bigint_const(10));
  auto expr =
    make_func(sirius::function_id::add, std::move(args), logical_type::make(type_id::BIGINT));
  sirius::expression_evaluator evaluator(
    *expr, mr, stream, sirius::expression_query_policy{mode}, 1);
  auto result = evaluator.evaluate(cudf::table_view{{input->view()}});
  REQUIRE(integers(result->view().column(0)) == std::vector<__int128_t>{11, 12, 8});

  auto scalar =
    make_cast(make_dec64_const(-150, 18, 2), logical_type::make(type_id::BIGINT), false);
  sirius::expression_evaluator scalar_evaluator(
    *scalar, mr, stream, sirius::expression_query_policy{mode}, 1);
  result = scalar_evaluator.evaluate(cudf::table_view{{input->view()}});
  REQUIRE(integers(result->view().column(0)) == std::vector<__int128_t>{-2, -2, -2});
}
