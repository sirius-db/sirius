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

// test
#include "ast_test_support.hpp"

#include <catch.hpp>
#include <utils/utils.hpp>

// sirius
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
#include <expression/ast/function_call.hpp>
#include <expression/ast/in_list.hpp>
#include <expression/ast/node.hpp>
#include <expression/ast/reference.hpp>
#include <expression/ast/unary_op.hpp>
#include <expression/function_id.hpp>
#include <expression/join_condition.hpp>
#include <expression/value.hpp>
#include <expression_executor/gpu_expression_executor.hpp>
#include <helper/logical_type.hpp>
#include <memory/sirius_memory_reservation_manager.hpp>

// cudf, etc.
#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <cuda_runtime_api.h>

// standard library
#include <cstdint>
#include <memory>
#include <numeric>

using namespace cucascade;
using namespace cucascade::memory;
using namespace sirius::expr_test;
using memory_mgr = ::sirius::memory::sirius_memory_reservation_manager;

namespace {

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
  return data_batch::make(batch_id, std::move(gpu_repr));
}

std::shared_ptr<data_batch> make_int32_batch_with_nulls(memory_space& space,
                                                        const std::vector<int32_t>& values,
                                                        const std::vector<bool>& valids)
{
  auto mr     = get_resource_ref(space);
  auto stream = cudf::get_default_stream();
  auto size   = static_cast<cudf::size_type>(values.size());

  auto null_mask = cudf::create_null_mask(size, cudf::mask_state::ALL_VALID, stream, mr);
  auto* mask_ptr = static_cast<cudf::bitmask_type*>(null_mask.data());

  cudf::size_type null_count = 0;
  for (cudf::size_type i = 0; i < size; ++i) {
    if (!valids[i]) {
      cudf::set_null_mask(mask_ptr, i, i + 1, false, stream);
      ++null_count;
    }
  }

  auto col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, size, std::move(null_mask), null_count, stream, mr);
  cudaMemcpy(col->mutable_view().data<int32_t>(),
             values.data(),
             sizeof(int32_t) * values.size(),
             cudaMemcpyHostToDevice);

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(col));
  auto table = std::make_unique<cudf::table>(std::move(cols));

  auto gpu_repr =
    std::make_unique<gpu_table_representation>(std::move(table), space, cudf::get_default_stream());
  auto batch_id = ::sirius::get_next_batch_id();
  return data_batch::make(batch_id, std::move(gpu_repr));
}

std::shared_ptr<data_batch> make_two_int32_batch_with_nulls(memory_space& space,
                                                            const std::vector<int32_t>& values_a,
                                                            const std::vector<bool>& valids_a,
                                                            const std::vector<int32_t>& values_b,
                                                            const std::vector<bool>& valids_b)
{
  auto mr     = get_resource_ref(space);
  auto stream = cudf::get_default_stream();
  auto size   = static_cast<cudf::size_type>(values_a.size());

  auto make_col = [&](const std::vector<int32_t>& values, const std::vector<bool>& valids) {
    auto null_mask = cudf::create_null_mask(size, cudf::mask_state::ALL_VALID, stream, mr);
    auto* mask_ptr = static_cast<cudf::bitmask_type*>(null_mask.data());
    cudf::size_type null_count = 0;
    for (cudf::size_type i = 0; i < size; ++i) {
      if (!valids[i]) {
        cudf::set_null_mask(mask_ptr, i, i + 1, false, stream);
        ++null_count;
      }
    }
    auto col = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT32}, size, std::move(null_mask), null_count, stream, mr);
    cudaMemcpy(col->mutable_view().data<int32_t>(),
               values.data(),
               sizeof(int32_t) * values.size(),
               cudaMemcpyHostToDevice);
    return col;
  };

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(make_col(values_a, valids_a));
  cols.push_back(make_col(values_b, valids_b));
  auto table = std::make_unique<cudf::table>(std::move(cols));

  auto gpu_repr =
    std::make_unique<gpu_table_representation>(std::move(table), space, cudf::get_default_stream());
  auto batch_id = ::sirius::get_next_batch_id();
  return data_batch::make(batch_id, std::move(gpu_repr));
}

std::shared_ptr<data_batch> make_decimal64_batch(memory_space& space,
                                                 int8_t scale,
                                                 const std::vector<int64_t>& values)
{
  auto mr     = get_resource_ref(space);
  auto stream = cudf::get_default_stream();
  auto size   = static_cast<cudf::size_type>(values.size());

  auto col = cudf::make_fixed_point_column(cudf::data_type{cudf::type_id::DECIMAL64, -scale},
                                           size,
                                           cudf::mask_state::UNALLOCATED,
                                           stream,
                                           mr);
  cudaMemcpy(col->mutable_view().data<int64_t>(),
             values.data(),
             sizeof(int64_t) * values.size(),
             cudaMemcpyHostToDevice);

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(col));
  auto table = std::make_unique<cudf::table>(std::move(cols));

  auto gpu_repr =
    std::make_unique<gpu_table_representation>(std::move(table), space, cudf::get_default_stream());
  auto batch_id = ::sirius::get_next_batch_id();
  return data_batch::make(batch_id, std::move(gpu_repr));
}

std::shared_ptr<data_batch> make_decimal64_two_col_batch(memory_space& space,
                                                         int8_t scale,
                                                         const std::vector<int64_t>& values_a,
                                                         const std::vector<int64_t>& values_b)
{
  auto mr     = get_resource_ref(space);
  auto stream = cudf::get_default_stream();
  auto size   = static_cast<cudf::size_type>(values_a.size());

  auto make_col = [&](const std::vector<int64_t>& values) {
    auto col = cudf::make_fixed_point_column(cudf::data_type{cudf::type_id::DECIMAL64, -scale},
                                             size,
                                             cudf::mask_state::UNALLOCATED,
                                             stream,
                                             mr);
    cudaMemcpy(col->mutable_view().data<int64_t>(),
               values.data(),
               sizeof(int64_t) * values.size(),
               cudaMemcpyHostToDevice);
    return col;
  };

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(make_col(values_a));
  cols.push_back(make_col(values_b));
  auto table = std::make_unique<cudf::table>(std::move(cols));

  auto gpu_repr =
    std::make_unique<gpu_table_representation>(std::move(table), space, cudf::get_default_stream());
  auto batch_id = ::sirius::get_next_batch_id();
  return data_batch::make(batch_id, std::move(gpu_repr));
}

std::shared_ptr<data_batch> make_decimal32_batch(memory_space& space,
                                                 int8_t scale,
                                                 const std::vector<int32_t>& values)
{
  auto mr     = get_resource_ref(space);
  auto stream = cudf::get_default_stream();
  auto size   = static_cast<cudf::size_type>(values.size());

  auto col = cudf::make_fixed_point_column(cudf::data_type{cudf::type_id::DECIMAL32, -scale},
                                           size,
                                           cudf::mask_state::UNALLOCATED,
                                           stream,
                                           mr);
  cudaMemcpy(col->mutable_view().data<int32_t>(),
             values.data(),
             sizeof(int32_t) * values.size(),
             cudaMemcpyHostToDevice);

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(col));
  auto table = std::make_unique<cudf::table>(std::move(cols));

  auto gpu_repr =
    std::make_unique<gpu_table_representation>(std::move(table), space, cudf::get_default_stream());
  auto batch_id = ::sirius::get_next_batch_id();
  return data_batch::make(batch_id, std::move(gpu_repr));
}

std::shared_ptr<data_batch> make_decimal128_batch(memory_space& space,
                                                  int8_t scale,
                                                  const std::vector<__int128_t>& values)
{
  auto mr     = get_resource_ref(space);
  auto stream = cudf::get_default_stream();
  auto size   = static_cast<cudf::size_type>(values.size());

  auto col = cudf::make_fixed_point_column(cudf::data_type{cudf::type_id::DECIMAL128, -scale},
                                           size,
                                           cudf::mask_state::UNALLOCATED,
                                           stream,
                                           mr);
  cudaMemcpy(col->mutable_view().data<__int128_t>(),
             values.data(),
             sizeof(__int128_t) * values.size(),
             cudaMemcpyHostToDevice);

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(col));
  auto table = std::make_unique<cudf::table>(std::move(cols));

  auto gpu_repr =
    std::make_unique<gpu_table_representation>(std::move(table), space, cudf::get_default_stream());
  auto batch_id = ::sirius::get_next_batch_id();
  return data_batch::make(batch_id, std::move(gpu_repr));
}

using exp_executor      = ::sirius::gpu_expression_executor;
using exp_strategy_enum = ::sirius::expression_executor_strategy;
auto constexpr MAT      = exp_strategy_enum::MATERIALIZE;

// Strategy tag types for TEMPLATE_TEST_CASE: each test instantiates against all three strategies
// so MATERIALIZE, AST_INTERPRET, and AST_JIT all get coverage from the same assertions.
struct mat_strategy {
  static constexpr auto value = exp_strategy_enum::MATERIALIZE;
};
struct ast_interpret_strategy {
  static constexpr auto value = exp_strategy_enum::AST_INTERPRET;
};
struct ast_jit_strategy {
  static constexpr auto value = exp_strategy_enum::AST_JIT;
};

// Native sirius::ast::node construction helpers (make_ref / make_int_const / ...)
// and GPU->host copy helpers live in ast_test_support.hpp, shared across the
// expression_executor test suites.

using sirius::logical_type;
using sirius::type_id;

// Shorthand: build executor, run execute(), return output table view and input table view.
struct exec_result {
  std::shared_ptr<data_batch> input;
  std::shared_ptr<data_batch> output;
  cudf::table_view input_view;
  cudf::table_view output_view;
};

exec_result run_execute(memory_space& space,
                        std::shared_ptr<data_batch> const& input_batch,
                        std::vector<std::unique_ptr<ast_node>> nodes,
                        exp_strategy_enum strategy = MAT)
{
  duckdb::vector<std::unique_ptr<ast_node>> ast_nodes;
  for (auto& n : nodes) {
    ast_nodes.push_back(std::move(n));
  }
  exp_executor executor(ast_nodes, get_resource_ref(space), cudf::get_default_stream(), strategy);
  auto input_ro     = input_batch->get_read_only();
  auto& in_repr     = input_ro->get_data()->cast<gpu_table_representation>();
  auto output_table = executor.execute(in_repr.get_table_view());
  REQUIRE(output_table != nullptr);
  auto output_batch = sirius::make_data_batch(
    std::move(output_table), *input_ro->get_memory_space(), cudf::get_default_stream());
  auto output_ro = output_batch->get_read_only();
  auto& out_repr = output_ro->get_data()->cast<gpu_table_representation>();
  return {input_batch, output_batch, in_repr.get_table_view(), out_repr.get_table_view()};
}

exec_result run_select(memory_space& space,
                       std::shared_ptr<data_batch> const& input_batch,
                       std::unique_ptr<ast_node> node,
                       exp_strategy_enum strategy = MAT)
{
  exp_executor executor(*node, get_resource_ref(space), cudf::get_default_stream(), strategy);
  auto input_ro     = input_batch->get_read_only();
  auto& in_repr     = input_ro->get_data()->cast<gpu_table_representation>();
  auto output_table = executor.select(in_repr.get_table_view());
  REQUIRE(output_table != nullptr);
  auto output_batch = sirius::make_data_batch(
    std::move(output_table), *input_ro->get_memory_space(), cudf::get_default_stream());
  auto output_ro = output_batch->get_read_only();
  auto& out_repr = output_ro->get_data()->cast<gpu_table_representation>();
  return {input_batch, output_batch, in_repr.get_table_view(), out_repr.get_table_view()};
}

// Convenience: pack a single node into the projection driver's vector.
std::vector<std::unique_ptr<ast_node>> one(std::unique_ptr<ast_node> n)
{
  std::vector<std::unique_ptr<ast_node>> v;
  v.push_back(std::move(n));
  return v;
}
}  // namespace

// ---------------------------------------------------------------------------
// execute() — reference, constant, comparison (basic smoke test per type)
// ---------------------------------------------------------------------------

TEMPLATE_TEST_CASE("execute projects references, constants, and comparisons",
                   "[expression_executor]",
                   mat_strategy,
                   ast_interpret_strategy,
                   ast_jit_strategy)
{
  constexpr auto strategy = TestType::value;
  auto* space             = get_default_gpu_space();
  REQUIRE(space != nullptr);

  SECTION("INT32")
  {
    auto input = make_input_batch(
      *space, {cudf::data_type{cudf::type_id::INT32}}, {std::pair<int, int>{0, 100}});

    std::vector<std::unique_ptr<ast_node>> exprs;
    exprs.push_back(make_ref(0));
    exprs.push_back(make_int_const(42));
    exprs.push_back(make_cmp(sirius::comparison_type::gt, make_ref(0), make_int_const(50)));

    auto [in_batch, out_batch, iv, ov] = run_execute(*space, input, std::move(exprs), strategy);
    REQUIRE(ov.num_columns() == 3);
    REQUIRE(ov.num_rows() == iv.num_rows());

    auto in0  = copy_column_to_host<int32_t>(iv.column(0));
    auto out0 = copy_column_to_host<int32_t>(ov.column(0));
    REQUIRE(out0 == in0);

    std::vector<int32_t> expected_const(iv.num_rows(), 42);
    REQUIRE(copy_column_to_host<int32_t>(ov.column(1)) == expected_const);

    std::vector<uint8_t> expected_cmp;
    for (auto v : in0) {
      expected_cmp.push_back(v > 50 ? 1U : 0U);
    }
    REQUIRE(copy_bool_column_to_host(ov.column(2)) == expected_cmp);
  }

  SECTION("DECIMAL64")
  {
    uint8_t const scale = 5;
    std::vector<int64_t> raw(128);
    std::iota(raw.begin(), raw.end(), 0);
    auto input = make_decimal64_batch(*space, static_cast<int8_t>(scale), raw);

    std::vector<std::unique_ptr<ast_node>> exprs;
    exprs.push_back(make_ref(0));
    exprs.push_back(make_dec64_const(42, 18, scale));
    exprs.push_back(
      make_cmp(sirius::comparison_type::lt, make_ref(0), make_dec64_const(64, 18, scale)));

    auto [in_batch, out_batch, iv, ov] = run_execute(*space, input, std::move(exprs), strategy);
    REQUIRE(ov.num_columns() == 3);
    REQUIRE(ov.num_rows() == iv.num_rows());

    auto in0 = copy_column_to_host<int64_t>(iv.column(0));
    REQUIRE(copy_column_to_host<int64_t>(ov.column(0)) == in0);

    std::vector<int64_t> expected_const(iv.num_rows(), 42);
    REQUIRE(copy_column_to_host<int64_t>(ov.column(1)) == expected_const);

    std::vector<uint8_t> expected_cmp;
    for (auto v : in0) {
      expected_cmp.push_back(v < 64 ? 1U : 0U);
    }
    REQUIRE(copy_bool_column_to_host(ov.column(2)) == expected_cmp);
  }

  SECTION("STRING")
  {
    auto input = make_input_batch(
      *space, {cudf::data_type{cudf::type_id::STRING}}, {std::pair<int, int>{1, 5}});

    std::vector<std::unique_ptr<ast_node>> exprs;
    exprs.push_back(make_ref(0));
    exprs.push_back(make_str_const("hello"));
    exprs.push_back(make_cmp(sirius::comparison_type::equal, make_ref(0), make_str_const("str_3")));

    auto [in_batch, out_batch, iv, ov] = run_execute(*space, input, std::move(exprs), strategy);
    REQUIRE(ov.num_columns() == 3);
    REQUIRE(ov.num_rows() == iv.num_rows());

    auto in0 = copy_string_column_to_host(iv.column(0));
    REQUIRE(copy_string_column_to_host(ov.column(0)) == in0);

    std::vector<std::string> expected_const(iv.num_rows(), "hello");
    REQUIRE(copy_string_column_to_host(ov.column(1)) == expected_const);

    std::vector<uint8_t> expected_cmp;
    for (auto const& v : in0) {
      expected_cmp.push_back(v == "str_3" ? 1U : 0U);
    }
    REQUIRE(copy_bool_column_to_host(ov.column(2)) == expected_cmp);
  }
}

// ---------------------------------------------------------------------------
// select() — basic filter + edge cases
// ---------------------------------------------------------------------------

TEMPLATE_TEST_CASE("select filters rows and handles edge cases",
                   "[expression_executor]",
                   mat_strategy,
                   ast_interpret_strategy,
                   ast_jit_strategy)
{
  constexpr auto strategy = TestType::value;
  auto* space             = get_default_gpu_space();
  REQUIRE(space != nullptr);

  SECTION("basic INT32 filter")
  {
    auto input = make_input_batch(
      *space, {cudf::data_type{cudf::type_id::INT32}}, {std::pair<int, int>{0, 9}});

    auto expr = make_cmp(sirius::comparison_type::gt, make_ref(0), make_int_const(5));

    auto [in_batch, out_batch, iv, ov] = run_select(*space, input, std::move(expr), strategy);
    auto in_vals                       = copy_column_to_host<int32_t>(iv.column(0));
    std::vector<int32_t> expected;
    for (auto v : in_vals) {
      if (v > 5) { expected.push_back(v); }
    }
    REQUIRE(copy_column_to_host<int32_t>(ov.column(0)) == expected);
  }

  SECTION("empty result")
  {
    auto input = make_input_batch(
      *space, {cudf::data_type{cudf::type_id::INT32}}, {std::pair<int, int>{0, 9}});

    auto expr = make_cmp(sirius::comparison_type::gt, make_ref(0), make_int_const(1000));

    auto [in_batch, out_batch, iv, ov] = run_select(*space, input, std::move(expr), strategy);
    REQUIRE(ov.num_rows() == 0);
    REQUIRE(ov.num_columns() == iv.num_columns());
  }

  SECTION("DECIMAL64 filter")
  {
    uint8_t const scale = 2;
    std::vector<int64_t> raw(128);
    std::iota(raw.begin(), raw.end(), 0);
    auto input = make_decimal64_batch(*space, static_cast<int8_t>(scale), raw);

    auto expr = make_cmp(sirius::comparison_type::gt, make_ref(0), make_dec64_const(64, 18, scale));

    auto [in_batch, out_batch, iv, ov] = run_select(*space, input, std::move(expr), strategy);
    auto in_vals                       = copy_column_to_host<int64_t>(iv.column(0));
    std::vector<int64_t> expected;
    for (auto v : in_vals) {
      if (v > 64) { expected.push_back(v); }
    }
    REQUIRE(copy_column_to_host<int64_t>(ov.column(0)) == expected);
  }

  SECTION("STRING equality filter")
  {
    auto input = make_input_batch(
      *space, {cudf::data_type{cudf::type_id::STRING}}, {std::pair<int, int>{1, 3}});

    auto expr = make_cmp(sirius::comparison_type::equal, make_ref(0), make_str_const("str_2"));

    auto [in_batch, out_batch, iv, ov] = run_select(*space, input, std::move(expr), strategy);
    auto in_strs                       = copy_string_column_to_host(iv.column(0));
    std::vector<std::string> expected;
    for (auto const& v : in_strs) {
      if (v == "str_2") { expected.push_back(v); }
    }
    REQUIRE(copy_string_column_to_host(ov.column(0)) == expected);
  }
}

// ---------------------------------------------------------------------------
// Arithmetic functions (AST-capable): col + const, col * col
// ---------------------------------------------------------------------------

TEMPLATE_TEST_CASE("execute arithmetic functions",
                   "[expression_executor]",
                   mat_strategy,
                   ast_interpret_strategy,
                   ast_jit_strategy)
{
  constexpr auto strategy = TestType::value;
  auto* space             = get_default_gpu_space();
  REQUIRE(space != nullptr);

  auto input =
    make_input_batch(*space,
                     {cudf::data_type{cudf::type_id::INT32}, cudf::data_type{cudf::type_id::INT32}},
                     {std::pair<int, int>{1, 50}, std::pair<int, int>{1, 10}});

  SECTION("col0 + 10")
  {
    std::vector<std::unique_ptr<ast_node>> children;
    children.push_back(make_ref(0));
    children.push_back(make_int_const(10));
    auto expr = make_func(
      sirius::function_id::add, std::move(children), logical_type::make(type_id::INTEGER));

    auto [in_batch, out_batch, iv, ov] = run_execute(*space, input, one(std::move(expr)), strategy);
    REQUIRE(ov.num_columns() == 1);
    auto in0  = copy_column_to_host<int32_t>(iv.column(0));
    auto out0 = copy_column_to_host<int32_t>(ov.column(0));
    for (size_t i = 0; i < in0.size(); ++i) {
      REQUIRE(out0[i] == in0[i] + 10);
    }
  }

  SECTION("col0 * col1")
  {
    std::vector<std::unique_ptr<ast_node>> children;
    children.push_back(make_ref(0));
    children.push_back(make_ref(1));
    auto expr = make_func(
      sirius::function_id::mul, std::move(children), logical_type::make(type_id::INTEGER));

    auto [in_batch, out_batch, iv, ov] = run_execute(*space, input, one(std::move(expr)), strategy);
    auto in0                           = copy_column_to_host<int32_t>(iv.column(0));
    auto in1                           = copy_column_to_host<int32_t>(iv.column(1));
    auto out0                          = copy_column_to_host<int32_t>(ov.column(0));
    for (size_t i = 0; i < in0.size(); ++i) {
      REQUIRE(out0[i] == in0[i] * in1[i]);
    }
  }

  SECTION("col0 % 3")
  {
    std::vector<std::unique_ptr<ast_node>> children;
    children.push_back(make_ref(0));
    children.push_back(make_int_const(3));
    auto expr = make_func(
      sirius::function_id::mod, std::move(children), logical_type::make(type_id::INTEGER));

    auto [in_batch, out_batch, iv, ov] = run_execute(*space, input, one(std::move(expr)), strategy);
    auto in0                           = copy_column_to_host<int32_t>(iv.column(0));
    auto out0                          = copy_column_to_host<int32_t>(ov.column(0));
    for (size_t i = 0; i < in0.size(); ++i) {
      REQUIRE(out0[i] == in0[i] % 3);
    }
  }
}

// ---------------------------------------------------------------------------
// Decimal arithmetic (MATERIALIZE path — AST is disabled for decimal return
// types pending cudf#21996, so all decimal function evaluation flows through
// cudf::binary_operation on fixed_point columns/scalars).
// ---------------------------------------------------------------------------

TEMPLATE_TEST_CASE("execute decimal arithmetic (DECIMAL64)",
                   "[expression_executor][decimal]",
                   mat_strategy,
                   ast_interpret_strategy,
                   ast_jit_strategy)
{
  constexpr auto strategy = TestType::value;
  auto* space             = get_default_gpu_space();
  REQUIRE(space != nullptr);

  uint8_t const width = 18;
  uint8_t const scale = 2;
  auto const dec_type = logical_type::make_decimal(width, scale);

  // Values: 0.00, 0.01, ... 0.09 (stored as 0, 1, ... 9 at scale 2)
  std::vector<int64_t> raw_a = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<int64_t> raw_b = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};

  SECTION("col + literal (col + 1.25)")
  {
    auto input = make_decimal64_batch(*space, static_cast<int8_t>(scale), raw_a);

    // 1.25 at scale 2 → raw value 125
    std::vector<std::unique_ptr<ast_node>> children;
    children.push_back(make_ref(0));
    children.push_back(make_dec64_const(125, width, scale));
    auto expr = make_func(sirius::function_id::add, std::move(children), dec_type);

    auto [in_batch, out_batch, iv, ov] = run_execute(*space, input, one(std::move(expr)), strategy);
    REQUIRE(ov.num_columns() == 1);
    REQUIRE(ov.column(0).type().id() == cudf::type_id::DECIMAL64);
    REQUIRE(ov.column(0).type().scale() == -static_cast<int32_t>(scale));

    auto out0 = copy_column_to_host<int64_t>(ov.column(0));
    std::vector<int64_t> expected;
    expected.reserve(raw_a.size());
    for (auto v : raw_a) {
      expected.push_back(v + 125);
    }
    REQUIRE(out0 == expected);
  }

  SECTION("col - col (two-column batch)")
  {
    auto input = make_decimal64_two_col_batch(*space, static_cast<int8_t>(scale), raw_b, raw_a);

    std::vector<std::unique_ptr<ast_node>> children;
    children.push_back(make_ref(0));
    children.push_back(make_ref(1));
    auto expr = make_func(sirius::function_id::sub, std::move(children), dec_type);

    auto [in_batch, out_batch, iv, ov] = run_execute(*space, input, one(std::move(expr)), strategy);
    REQUIRE(ov.column(0).type().id() == cudf::type_id::DECIMAL64);
    REQUIRE(ov.column(0).type().scale() == -static_cast<int32_t>(scale));

    auto out0 = copy_column_to_host<int64_t>(ov.column(0));
    std::vector<int64_t> expected;
    expected.reserve(raw_a.size());
    for (size_t i = 0; i < raw_a.size(); ++i) {
      expected.push_back(raw_b[i] - raw_a[i]);
    }
    REQUIRE(out0 == expected);
  }

  SECTION("col * integer literal (decimal with scale 0)")
  {
    auto input = make_decimal64_batch(*space, static_cast<int8_t>(scale), raw_a);

    // 2 as DECIMAL(18, 0) → same cudf type_id (DECIMAL64), scale 0.
    // cudf MUL: output scale = lhs_scale + rhs_scale = -2 + 0 = -2.
    std::vector<std::unique_ptr<ast_node>> children;
    children.push_back(make_ref(0));
    children.push_back(make_dec64_const(2, width, /*scale=*/0));
    auto expr = make_func(sirius::function_id::mul, std::move(children), dec_type);

    auto [in_batch, out_batch, iv, ov] = run_execute(*space, input, one(std::move(expr)), strategy);
    REQUIRE(ov.column(0).type().id() == cudf::type_id::DECIMAL64);
    REQUIRE(ov.column(0).type().scale() == -static_cast<int32_t>(scale));

    auto out0 = copy_column_to_host<int64_t>(ov.column(0));
    std::vector<int64_t> expected;
    expected.reserve(raw_a.size());
    for (auto v : raw_a) {
      expected.push_back(v * 2);
    }
    REQUIRE(out0 == expected);
  }
}

TEMPLATE_TEST_CASE("execute decimal arithmetic (DECIMAL32)",
                   "[expression_executor][decimal]",
                   mat_strategy,
                   ast_interpret_strategy,
                   ast_jit_strategy)
{
  constexpr auto strategy = TestType::value;
  auto* space             = get_default_gpu_space();
  REQUIRE(space != nullptr);

  // Width 8 → INT32 physical (DuckDB) → DECIMAL32 (cudf). Exercises the
  // DECIMAL32 branches in gpu_execute_constant.cpp and GetCudfType.
  uint8_t const width = 8;
  uint8_t const scale = 2;
  auto const dec_type = logical_type::make_decimal(width, scale);

  // 1.00, 2.00, 3.00 ... 10.00 (raw: 100, 200, ... 1000 at scale 2)
  std::vector<int32_t> raw = {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000};
  auto input               = make_decimal32_batch(*space, static_cast<int8_t>(scale), raw);

  // col0 + 0.50 → raw add 50
  std::vector<std::unique_ptr<ast_node>> children;
  children.push_back(make_ref(0));
  children.push_back(make_dec32_const(50, width, scale));
  auto expr = make_func(sirius::function_id::add, std::move(children), dec_type);

  auto [in_batch, out_batch, iv, ov] = run_execute(*space, input, one(std::move(expr)), strategy);
  REQUIRE(ov.num_columns() == 1);
  REQUIRE(ov.column(0).type().id() == cudf::type_id::DECIMAL32);
  REQUIRE(ov.column(0).type().scale() == -static_cast<int32_t>(scale));

  auto out0 = copy_column_to_host<int32_t>(ov.column(0));
  std::vector<int32_t> expected;
  expected.reserve(raw.size());
  for (auto v : raw) {
    expected.push_back(v + 50);
  }
  REQUIRE(out0 == expected);
}

TEMPLATE_TEST_CASE("execute nested decimal arithmetic (col + 1.00) * 2",
                   "[expression_executor][decimal]",
                   mat_strategy,
                   ast_interpret_strategy,
                   ast_jit_strategy)
{
  // Regression coverage: before disabling AST for decimal-returning functions,
  // nested decimal arithmetic would feed intermediate fixed_point results into
  // the cudf AST kernel and trip cudf#21996. With AST disabled for decimal
  // return types, the inner `+` materializes via cudf::binary_operation and the
  // outer `*` consumes the materialized column via another cudf::binary_operation.
  constexpr auto strategy = TestType::value;
  auto* space             = get_default_gpu_space();
  REQUIRE(space != nullptr);

  uint8_t const width = 18;
  uint8_t const scale = 2;
  auto const dec_type = logical_type::make_decimal(width, scale);

  // 0.00, 0.01 ... 0.09
  std::vector<int64_t> raw = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  auto input               = make_decimal64_batch(*space, static_cast<int8_t>(scale), raw);

  // Inner: col + 1.00 (both scale 2) → result scale 2
  std::vector<std::unique_ptr<ast_node>> add_children;
  add_children.push_back(make_ref(0));
  add_children.push_back(make_dec64_const(100, width, scale));
  auto add_expr = make_func(sirius::function_id::add, std::move(add_children), dec_type);

  // Outer: (col + 1.00) * 2 where 2 is DECIMAL(18, 0) so MUL output scale = -2 + 0 = -2
  std::vector<std::unique_ptr<ast_node>> mul_children;
  mul_children.push_back(std::move(add_expr));
  mul_children.push_back(make_dec64_const(2, width, /*scale=*/0));
  auto expr = make_func(sirius::function_id::mul, std::move(mul_children), dec_type);

  auto [in_batch, out_batch, iv, ov] = run_execute(*space, input, one(std::move(expr)), strategy);
  REQUIRE(ov.num_columns() == 1);
  REQUIRE(ov.column(0).type().id() == cudf::type_id::DECIMAL64);
  REQUIRE(ov.column(0).type().scale() == -static_cast<int32_t>(scale));

  auto out0 = copy_column_to_host<int64_t>(ov.column(0));
  std::vector<int64_t> expected;
  expected.reserve(raw.size());
  for (auto v : raw) {
    expected.push_back((v + 100) * 2);
  }
  REQUIRE(out0 == expected);
}

TEMPLATE_TEST_CASE("execute decimal arithmetic (DECIMAL128)",
                   "[expression_executor][decimal]",
                   mat_strategy,
                   ast_interpret_strategy,
                   ast_jit_strategy)
{
  // Width 38 → INT128 physical (DuckDB) → DECIMAL128 (cudf). Exercises the
  // hugeint_t → __int128_t conversion branch in gpu_execute_constant.cpp for
  // decimal constants, plus cudf::binary_operation on DECIMAL128 columns.
  constexpr auto strategy = TestType::value;
  auto* space             = get_default_gpu_space();
  REQUIRE(space != nullptr);

  uint8_t const width = 38;
  uint8_t const scale = 2;
  auto const dec_type = logical_type::make_decimal(width, scale);

  // 1.00, 2.00, 3.00 (raw: 100, 200, 300 at scale 2)
  std::vector<__int128_t> raw = {100, 200, 300};
  auto input                  = make_decimal128_batch(*space, static_cast<int8_t>(scale), raw);

  // col + 1.25 → raw add 125
  std::vector<std::unique_ptr<ast_node>> children;
  children.push_back(make_ref(0));
  children.push_back(make_dec128_const(__int128_t{125}, width, scale));
  auto expr = make_func(sirius::function_id::add, std::move(children), dec_type);

  auto [in_batch, out_batch, iv, ov] = run_execute(*space, input, one(std::move(expr)), strategy);
  REQUIRE(ov.num_columns() == 1);
  REQUIRE(ov.column(0).type().id() == cudf::type_id::DECIMAL128);
  REQUIRE(ov.column(0).type().scale() == -static_cast<int32_t>(scale));

  auto out0 = copy_column_to_host<__int128_t>(ov.column(0));
  REQUIRE(out0.size() == raw.size());
  for (size_t i = 0; i < raw.size(); ++i) {
    // __int128_t has no stream operator, so compare via int64 cast (values fit).
    REQUIRE(static_cast<int64_t>(out0[i]) == static_cast<int64_t>(raw[i] + 125));
  }
}

TEMPLATE_TEST_CASE("execute decimal DIV (DECIMAL64)",
                   "[expression_executor][decimal]",
                   mat_strategy,
                   ast_interpret_strategy,
                   ast_jit_strategy)
{
  // DIV has a distinct output-scale rule from ADD/SUB/MUL:
  //   cudf fixed_point DIV output scale = lhs_scale - rhs_scale
  // Here: col(scale -2) / literal(scale 0) → output scale -2.
  constexpr auto strategy = TestType::value;
  auto* space             = get_default_gpu_space();
  REQUIRE(space != nullptr);

  uint8_t const width = 18;
  uint8_t const scale = 2;
  auto const dec_type = logical_type::make_decimal(width, scale);

  // 4.00, 5.00, 6.00, 7.00 (raw 400, 500, 600, 700 at scale 2)
  std::vector<int64_t> raw = {400, 500, 600, 700};
  auto input               = make_decimal64_batch(*space, static_cast<int8_t>(scale), raw);

  // col / 2 (2 as DECIMAL(18, 0)) → fixed_point div truncates toward zero in
  // the output scale; 500/2 = 250, 700/2 = 350.
  std::vector<std::unique_ptr<ast_node>> children;
  children.push_back(make_ref(0));
  children.push_back(make_dec64_const(2, width, /*scale=*/0));
  auto expr = make_func(sirius::function_id::div, std::move(children), dec_type);

  auto [in_batch, out_batch, iv, ov] = run_execute(*space, input, one(std::move(expr)), strategy);
  REQUIRE(ov.num_columns() == 1);
  REQUIRE(ov.column(0).type().id() == cudf::type_id::DECIMAL64);
  REQUIRE(ov.column(0).type().scale() == -static_cast<int32_t>(scale));

  auto out0 = copy_column_to_host<int64_t>(ov.column(0));
  std::vector<int64_t> expected;
  expected.reserve(raw.size());
  for (auto v : raw) {
    expected.push_back(v / 2);
  }
  REQUIRE(out0 == expected);
}

TEMPLATE_TEST_CASE("execute decimal TPC-H Q1 shape price * (1 - discount)",
                   "[expression_executor][decimal]",
                   mat_strategy,
                   ast_interpret_strategy,
                   ast_jit_strategy)
{
  // Mirrors TPC-H Q1's `l_extendedprice * (1 - l_discount)`:
  //   inner SUB has a scalar on the LEFT (1.00) and a column on the RIGHT
  //   (discount), which hits the left-scalar branch of execute_numeric_binary_func;
  //   outer MUL then consumes the materialized inner result as a column.
  //   MUL output scale = lhs_scale + rhs_scale = -2 + -2 = -4, so the return
  //   type is DECIMAL(18, 4).
  constexpr auto strategy = TestType::value;
  auto* space             = get_default_gpu_space();
  REQUIRE(space != nullptr);

  uint8_t const width     = 18;
  uint8_t const scale2    = 2;
  uint8_t const scale4    = 4;
  auto const price_type   = logical_type::make_decimal(width, scale2);  // DECIMAL(18,2)
  auto const discount_t   = logical_type::make_decimal(width, scale2);  // DECIMAL(18,2)
  auto const mul_ret_type = logical_type::make_decimal(width, scale4);  // DECIMAL(18,4)

  // price: 1000.00, 2000.00, 3000.00
  std::vector<int64_t> raw_price = {100000, 200000, 300000};
  // discount: 0.05, 0.10, 0.15
  std::vector<int64_t> raw_discount = {5, 10, 15};
  auto input =
    make_decimal64_two_col_batch(*space, static_cast<int8_t>(scale2), raw_price, raw_discount);

  // Inner: 1.00 - discount (scalar on left, column on right)
  std::vector<std::unique_ptr<ast_node>> sub_children;
  sub_children.push_back(make_dec64_const(100, width, scale2));
  sub_children.push_back(make_ref(1));
  auto sub_expr = make_func(sirius::function_id::sub, std::move(sub_children), discount_t);

  // Outer: price * (1.00 - discount) → return type DECIMAL(18, 4)
  std::vector<std::unique_ptr<ast_node>> mul_children;
  mul_children.push_back(make_ref(0));
  mul_children.push_back(std::move(sub_expr));
  auto expr = make_func(sirius::function_id::mul, std::move(mul_children), mul_ret_type);

  auto [in_batch, out_batch, iv, ov] = run_execute(*space, input, one(std::move(expr)), strategy);
  REQUIRE(ov.num_columns() == 1);
  REQUIRE(ov.column(0).type().id() == cudf::type_id::DECIMAL64);
  REQUIRE(ov.column(0).type().scale() == -static_cast<int32_t>(scale4));

  // Expected: price * (100 - discount) interpreted at scale -4.
  //   1000.00 * 0.95 = 950.0000 → raw 100000 * 95 = 9500000
  //   2000.00 * 0.90 = 1800.0000 → raw 200000 * 90 = 18000000
  //   3000.00 * 0.85 = 2550.0000 → raw 300000 * 85 = 25500000
  auto out0 = copy_column_to_host<int64_t>(ov.column(0));
  std::vector<int64_t> expected;
  expected.reserve(raw_price.size());
  for (size_t i = 0; i < raw_price.size(); ++i) {
    expected.push_back(raw_price[i] * (100 - raw_discount[i]));
  }
  REQUIRE(out0 == expected);
}

// ---------------------------------------------------------------------------
// LIKE / NOT LIKE (AST breakers — string functions always materialize)
// ---------------------------------------------------------------------------

TEMPLATE_TEST_CASE("select LIKE and NOT LIKE",
                   "[expression_executor]",
                   mat_strategy,
                   ast_interpret_strategy,
                   ast_jit_strategy)
{
  constexpr auto strategy = TestType::value;
  auto* space             = get_default_gpu_space();
  REQUIRE(space != nullptr);

  // Generates strings "str_1", "str_2", "str_3"
  auto input =
    make_input_batch(*space, {cudf::data_type{cudf::type_id::STRING}}, {std::pair<int, int>{1, 3}});

  SECTION("LIKE 'str_2'")
  {
    // col0 LIKE 'str_2' — exact match via LIKE
    std::vector<std::unique_ptr<ast_node>> children;
    children.push_back(make_ref(0));
    children.push_back(make_str_const("str_2"));
    auto expr = make_func(
      sirius::function_id::like, std::move(children), logical_type::make(type_id::BOOLEAN));

    auto [in_batch, out_batch, iv, ov] = run_select(*space, input, std::move(expr), strategy);
    auto in_strs                       = copy_string_column_to_host(iv.column(0));
    std::vector<std::string> expected;
    for (auto const& s : in_strs) {
      if (s == "str_2") { expected.push_back(s); }
    }
    REQUIRE(copy_string_column_to_host(ov.column(0)) == expected);
  }

  SECTION("LIKE 'str_%' — wildcard")
  {
    // col0 LIKE 'str_%' — should match all rows
    std::vector<std::unique_ptr<ast_node>> children;
    children.push_back(make_ref(0));
    children.push_back(make_str_const("str_%"));
    auto expr = make_func(
      sirius::function_id::like, std::move(children), logical_type::make(type_id::BOOLEAN));

    auto [in_batch, out_batch, iv, ov] = run_select(*space, input, std::move(expr), strategy);
    REQUIRE(ov.num_rows() == iv.num_rows());
  }

  SECTION("NOT LIKE 'str_1'")
  {
    // col0 NOT LIKE 'str_1' — should exclude 'str_1' rows
    std::vector<std::unique_ptr<ast_node>> children;
    children.push_back(make_ref(0));
    children.push_back(make_str_const("str_1"));
    auto expr = make_func(
      sirius::function_id::not_like, std::move(children), logical_type::make(type_id::BOOLEAN));

    auto [in_batch, out_batch, iv, ov] = run_select(*space, input, std::move(expr), strategy);
    auto in_strs                       = copy_string_column_to_host(iv.column(0));
    std::vector<std::string> expected;
    for (auto const& s : in_strs) {
      if (s != "str_1") { expected.push_back(s); }
    }
    REQUIRE(copy_string_column_to_host(ov.column(0)) == expected);
  }
}

// ---------------------------------------------------------------------------
// CASE/WHEN (always materializes — AST breaker)
// ---------------------------------------------------------------------------

TEMPLATE_TEST_CASE("execute CASE expression",
                   "[expression_executor]",
                   mat_strategy,
                   ast_interpret_strategy,
                   ast_jit_strategy)
{
  constexpr auto strategy = TestType::value;
  auto* space             = get_default_gpu_space();
  REQUIRE(space != nullptr);

  // CASE WHEN col0 > 50 THEN 1 ELSE 0 END
  auto input = make_input_batch(
    *space, {cudf::data_type{cudf::type_id::INT32}}, {std::pair<int, int>{0, 100}});

  std::vector<sirius::ast::case_expr::when_then> cases;
  cases.push_back(sirius::ast::case_expr::when_then{
    make_cmp(sirius::comparison_type::gt, make_ref(0), make_int_const(50)), make_int_const(1)});
  auto expr = std::make_unique<ast_node>(sirius::ast::case_expr{
    std::move(cases), make_int_const(0), logical_type::make(type_id::INTEGER)});

  auto [in_batch, out_batch, iv, ov] = run_execute(*space, input, one(std::move(expr)), strategy);
  REQUIRE(ov.num_columns() == 1);
  REQUIRE(ov.num_rows() == iv.num_rows());

  auto in_vals  = copy_column_to_host<int32_t>(iv.column(0));
  auto out_vals = copy_column_to_host<int32_t>(ov.column(0));
  for (size_t i = 0; i < in_vals.size(); ++i) {
    REQUIRE(out_vals[i] == (in_vals[i] > 50 ? 1 : 0));
  }
}

TEMPLATE_TEST_CASE("execute CASE with multiple WHEN branches",
                   "[expression_executor]",
                   mat_strategy,
                   ast_interpret_strategy,
                   ast_jit_strategy)
{
  constexpr auto strategy = TestType::value;
  auto* space             = get_default_gpu_space();
  REQUIRE(space != nullptr);

  // CASE WHEN col0 < 30 THEN -1 WHEN col0 > 70 THEN 1 ELSE 0 END
  auto input = make_input_batch(
    *space, {cudf::data_type{cudf::type_id::INT32}}, {std::pair<int, int>{0, 100}});

  std::vector<sirius::ast::case_expr::when_then> cases;
  cases.push_back(sirius::ast::case_expr::when_then{
    make_cmp(sirius::comparison_type::lt, make_ref(0), make_int_const(30)), make_int_const(-1)});
  cases.push_back(sirius::ast::case_expr::when_then{
    make_cmp(sirius::comparison_type::gt, make_ref(0), make_int_const(70)), make_int_const(1)});
  auto expr = std::make_unique<ast_node>(sirius::ast::case_expr{
    std::move(cases), make_int_const(0), logical_type::make(type_id::INTEGER)});

  auto [in_batch, out_batch, iv, ov] = run_execute(*space, input, one(std::move(expr)), strategy);
  auto in_vals                       = copy_column_to_host<int32_t>(iv.column(0));
  auto out_vals                      = copy_column_to_host<int32_t>(ov.column(0));
  for (size_t i = 0; i < in_vals.size(); ++i) {
    int32_t expected = in_vals[i] < 30 ? -1 : (in_vals[i] > 70 ? 1 : 0);
    REQUIRE(out_vals[i] == expected);
  }
}

// ---------------------------------------------------------------------------
// BETWEEN (decomposed into two comparisons + AND)
// ---------------------------------------------------------------------------

TEMPLATE_TEST_CASE(
  "select BETWEEN", "[expression_executor]", mat_strategy, ast_interpret_strategy, ast_jit_strategy)
{
  constexpr auto strategy = TestType::value;
  auto* space             = get_default_gpu_space();
  REQUIRE(space != nullptr);

  auto input = make_input_batch(
    *space, {cudf::data_type{cudf::type_id::INT32}}, {std::pair<int, int>{0, 100}});

  // col0 BETWEEN 20 AND 50 (inclusive)
  auto expr = make_between(make_ref(0),
                           make_int_const(20),
                           make_int_const(50),
                           /*lower_inclusive=*/true,
                           /*upper_inclusive=*/true);

  auto [in_batch, out_batch, iv, ov] = run_select(*space, input, std::move(expr), strategy);
  auto in_vals                       = copy_column_to_host<int32_t>(iv.column(0));
  std::vector<int32_t> expected;
  for (auto v : in_vals) {
    if (v >= 20 && v <= 50) { expected.push_back(v); }
  }
  REQUIRE(copy_column_to_host<int32_t>(ov.column(0)) == expected);
}

// ---------------------------------------------------------------------------
// IN / NOT IN (AST breaker when constant list — uses cudf::contains)
// ---------------------------------------------------------------------------

TEMPLATE_TEST_CASE("select IN and NOT IN",
                   "[expression_executor]",
                   mat_strategy,
                   ast_interpret_strategy,
                   ast_jit_strategy)
{
  constexpr auto strategy = TestType::value;
  auto* space             = get_default_gpu_space();
  REQUIRE(space != nullptr);

  auto input =
    make_input_batch(*space, {cudf::data_type{cudf::type_id::INT32}}, {std::pair<int, int>{0, 9}});

  SECTION("IN (2, 4, 6)")
  {
    std::vector<std::unique_ptr<ast_node>> values;
    values.push_back(make_int_const(2));
    values.push_back(make_int_const(4));
    values.push_back(make_int_const(6));
    auto expr = make_in(make_ref(0), std::move(values), /*negated=*/false);

    auto [in_batch, out_batch, iv, ov] = run_select(*space, input, std::move(expr), strategy);
    auto in_vals                       = copy_column_to_host<int32_t>(iv.column(0));
    std::vector<int32_t> expected;
    for (auto v : in_vals) {
      if (v == 2 || v == 4 || v == 6) { expected.push_back(v); }
    }
    REQUIRE(copy_column_to_host<int32_t>(ov.column(0)) == expected);
  }

  SECTION("NOT IN (0, 1, 2, 3)")
  {
    std::vector<std::unique_ptr<ast_node>> values;
    values.push_back(make_int_const(0));
    values.push_back(make_int_const(1));
    values.push_back(make_int_const(2));
    values.push_back(make_int_const(3));
    auto expr = make_in(make_ref(0), std::move(values), /*negated=*/true);

    auto [in_batch, out_batch, iv, ov] = run_select(*space, input, std::move(expr), strategy);
    auto in_vals                       = copy_column_to_host<int32_t>(iv.column(0));
    std::vector<int32_t> expected;
    for (auto v : in_vals) {
      if (v != 0 && v != 1 && v != 2 && v != 3) { expected.push_back(v); }
    }
    REQUIRE(copy_column_to_host<int32_t>(ov.column(0)) == expected);
  }
}

// ---------------------------------------------------------------------------
// IS NULL / IS NOT NULL / NOT (operator expressions)
// ---------------------------------------------------------------------------

TEMPLATE_TEST_CASE("select IS NULL and IS NOT NULL",
                   "[expression_executor]",
                   mat_strategy,
                   ast_interpret_strategy,
                   ast_jit_strategy)
{
  constexpr auto strategy = TestType::value;
  auto* space             = get_default_gpu_space();
  REQUIRE(space != nullptr);

  std::vector<int32_t> values = {10, 20, 30, 40, 50};
  std::vector<bool> valids    = {true, false, true, false, true};
  auto input                  = make_int32_batch_with_nulls(*space, values, valids);

  SECTION("IS NULL")
  {
    auto expr = make_unary(sirius::ast::unary_op::kind::op_is_null, make_ref(0));

    auto [in_batch, out_batch, iv, ov] = run_select(*space, input, std::move(expr), strategy);
    // Null rows should pass the IS NULL filter
    REQUIRE(ov.num_rows() == 2);
  }

  SECTION("IS NOT NULL")
  {
    auto expr = make_unary(sirius::ast::unary_op::kind::op_is_not_null, make_ref(0));

    auto [in_batch, out_batch, iv, ov] = run_select(*space, input, std::move(expr), strategy);
    // Non-null rows should pass
    REQUIRE(ov.num_rows() == 3);
    auto out_vals = copy_column_to_host<int32_t>(ov.column(0));
    REQUIRE(out_vals == std::vector<int32_t>{10, 30, 50});
  }
}

// ---------------------------------------------------------------------------
// COALESCE — AST breaker, always materialized, exercised across all strategies
// ---------------------------------------------------------------------------

TEMPLATE_TEST_CASE("execute COALESCE",
                   "[expression_executor]",
                   mat_strategy,
                   ast_interpret_strategy,
                   ast_jit_strategy)
{
  constexpr auto strategy = TestType::value;
  auto* space             = get_default_gpu_space();
  REQUIRE(space != nullptr);

  SECTION("col + scalar fallback fills every null")
  {
    std::vector<int32_t> values = {10, 99, 30, 99, 50};
    std::vector<bool> valids    = {true, false, true, false, true};
    auto input                  = make_int32_batch_with_nulls(*space, values, valids);

    std::vector<std::unique_ptr<ast_node>> children;
    children.push_back(make_ref(0));
    children.push_back(make_int_const(-1));
    auto expr = make_coalesce(std::move(children), logical_type::make(type_id::INTEGER));

    auto [in_batch, out_batch, iv, ov] = run_execute(*space, input, one(std::move(expr)), strategy);
    REQUIRE(ov.num_columns() == 1);
    REQUIRE(ov.num_rows() == iv.num_rows());
    REQUIRE(ov.column(0).null_count() == 0);

    std::vector<int32_t> expected;
    for (size_t i = 0; i < values.size(); ++i) {
      expected.push_back(valids[i] ? values[i] : -1);
    }
    REQUIRE(copy_column_to_host<int32_t>(ov.column(0)) == expected);
  }

  SECTION("col + col leaves residual nulls where both are null")
  {
    // Row 0: col_a valid   → 10
    // Row 1: col_b valid   → 200
    // Row 2: col_a valid   → 30
    // Row 3: col_b valid   → 400
    // Row 4: both null     → null (residual)
    std::vector<int32_t> values_a = {10, 99, 30, 99, 99};
    std::vector<bool> valids_a    = {true, false, true, false, false};
    std::vector<int32_t> values_b = {99, 200, 99, 400, 99};
    std::vector<bool> valids_b    = {false, true, false, true, false};
    auto input = make_two_int32_batch_with_nulls(*space, values_a, valids_a, values_b, valids_b);

    std::vector<std::unique_ptr<ast_node>> children;
    children.push_back(make_ref(0));
    children.push_back(make_ref(1));
    auto expr = make_coalesce(std::move(children), logical_type::make(type_id::INTEGER));

    auto [in_batch, out_batch, iv, ov] = run_execute(*space, input, one(std::move(expr)), strategy);
    REQUIRE(ov.num_columns() == 1);
    REQUIRE(ov.num_rows() == iv.num_rows());
    REQUIRE(ov.column(0).null_count() == 1);

    auto out_vals                     = copy_column_to_host<int32_t>(ov.column(0));
    auto out_valids                   = copy_valids_to_host(ov.column(0));
    std::vector<int32_t> expected     = {10, 200, 30, 400, 0};
    std::vector<bool> expected_valids = {true, true, true, true, false};
    REQUIRE(out_valids == expected_valids);
    for (size_t i = 0; i < expected.size(); ++i) {
      if (expected_valids[i]) { REQUIRE(out_vals[i] == expected[i]); }
    }
  }

  SECTION("col + col + scalar chain fills every null")
  {
    std::vector<int32_t> values_a = {10, 99, 99, 99};
    std::vector<bool> valids_a    = {true, false, false, false};
    std::vector<int32_t> values_b = {99, 200, 99, 99};
    std::vector<bool> valids_b    = {false, true, false, false};
    auto input = make_two_int32_batch_with_nulls(*space, values_a, valids_a, values_b, valids_b);

    std::vector<std::unique_ptr<ast_node>> children;
    children.push_back(make_ref(0));
    children.push_back(make_ref(1));
    children.push_back(make_int_const(-7));
    auto expr = make_coalesce(std::move(children), logical_type::make(type_id::INTEGER));

    auto [in_batch, out_batch, iv, ov] = run_execute(*space, input, one(std::move(expr)), strategy);
    REQUIRE(ov.column(0).null_count() == 0);
    std::vector<int32_t> expected = {10, 200, -7, -7};
    REQUIRE(copy_column_to_host<int32_t>(ov.column(0)) == expected);
  }

  SECTION("col + col + col leaves residual nulls where all three are null")
  {
    // Only the last row has no valid value anywhere.
    std::vector<int32_t> values_a = {10, 99, 99, 99};
    std::vector<bool> valids_a    = {true, false, false, false};
    std::vector<int32_t> values_b = {99, 200, 99, 99};
    std::vector<bool> valids_b    = {false, true, false, false};
    auto input = make_two_int32_batch_with_nulls(*space, values_a, valids_a, values_b, valids_b);

    std::vector<std::unique_ptr<ast_node>> children;
    children.push_back(make_ref(0));
    children.push_back(make_ref(1));
    // Third child is col_a again (same column still has the same nulls) — drives the
    // column-replacement branch a second time with residual nulls surviving.
    children.push_back(make_ref(0));
    auto expr = make_coalesce(std::move(children), logical_type::make(type_id::INTEGER));

    auto [in_batch, out_batch, iv, ov] = run_execute(*space, input, one(std::move(expr)), strategy);
    REQUIRE(ov.column(0).null_count() == 2);

    auto out_vals                     = copy_column_to_host<int32_t>(ov.column(0));
    auto out_valids                   = copy_valids_to_host(ov.column(0));
    std::vector<bool> expected_valids = {true, true, false, false};
    REQUIRE(out_valids == expected_valids);
    REQUIRE(out_vals[0] == 10);
    REQUIRE(out_vals[1] == 200);
  }

  // Short-circuit: once the running result has no nulls, later children must not be evaluated.
  // Uses an out-of-range column reference as a "poison" child — if it's ever evaluated,
  // cudf::table_view::column(999) throws std::out_of_range.
  SECTION("short-circuits once result has no nulls")
  {
    std::vector<int32_t> values_all_valid = {10, 20, 30, 40, 50};
    std::vector<bool> valids_all_true     = {true, true, true, true, true};
    auto input = make_int32_batch_with_nulls(*space, values_all_valid, valids_all_true);

    std::vector<std::unique_ptr<ast_node>> children;
    children.push_back(make_ref(0));
    // Poison: out-of-range reference. Only safe if short-circuit skips it.
    children.push_back(make_ref(999));
    auto expr = make_coalesce(std::move(children), logical_type::make(type_id::INTEGER));

    auto [in_batch, out_batch, iv, ov] = run_execute(*space, input, one(std::move(expr)), strategy);
    REQUIRE(ov.num_columns() == 1);
    REQUIRE(ov.column(0).null_count() == 0);
    REQUIRE(copy_column_to_host<int32_t>(ov.column(0)) == values_all_valid);
  }

  // Negative control: if the first child has nulls, short-circuit must NOT trigger, and the
  // poison child is reached — confirming the poison is real (so the positive test above means
  // something).
  SECTION("does not short-circuit when nulls remain — poison child fires")
  {
    std::vector<int32_t> values = {10, 99, 30};
    std::vector<bool> valids    = {true, false, true};
    auto input                  = make_int32_batch_with_nulls(*space, values, valids);

    std::vector<std::unique_ptr<ast_node>> children;
    children.push_back(make_ref(0));
    children.push_back(make_ref(999));
    auto expr = make_coalesce(std::move(children), logical_type::make(type_id::INTEGER));

    REQUIRE_THROWS(run_execute(*space, input, one(std::move(expr)), strategy));
  }
}

TEMPLATE_TEST_CASE("select COALESCE nested in predicate",
                   "[expression_executor]",
                   mat_strategy,
                   ast_interpret_strategy,
                   ast_jit_strategy)
{
  constexpr auto strategy = TestType::value;
  auto* space             = get_default_gpu_space();
  REQUIRE(space != nullptr);

  // WHERE COALESCE(col0, 0) > 20 — exercises COALESCE as an AST-capable parent's child.
  // With col0 = {10, NULL, 30, NULL, 50}, nulls default to 0 and are filtered out; the
  // predicate keeps {30, 50}.
  std::vector<int32_t> values = {10, 99, 30, 99, 50};
  std::vector<bool> valids    = {true, false, true, false, true};
  auto input                  = make_int32_batch_with_nulls(*space, values, valids);

  std::vector<std::unique_ptr<ast_node>> children;
  children.push_back(make_ref(0));
  children.push_back(make_int_const(0));
  auto coalesce = make_coalesce(std::move(children), logical_type::make(type_id::INTEGER));

  auto expr = make_cmp(sirius::comparison_type::gt, std::move(coalesce), make_int_const(20));

  auto [in_batch, out_batch, iv, ov] = run_select(*space, input, std::move(expr), strategy);
  REQUIRE(ov.num_rows() == 2);
  REQUIRE(copy_column_to_host<int32_t>(ov.column(0)) == std::vector<int32_t>{30, 50});
}

TEMPLATE_TEST_CASE("select respects null mask under plain comparison",
                   "[expression_executor]",
                   mat_strategy,
                   ast_interpret_strategy,
                   ast_jit_strategy)
{
  constexpr auto strategy = TestType::value;
  auto* space             = get_default_gpu_space();
  REQUIRE(space != nullptr);

  std::vector<int32_t> values = {1, 2, 3, 4, 5};
  std::vector<bool> valids    = {true, false, true, false, true};
  auto input                  = make_int32_batch_with_nulls(*space, values, valids);

  auto expr = make_cmp(sirius::comparison_type::gt, make_ref(0), make_int_const(2));

  auto [in_batch, out_batch, iv, ov] = run_select(*space, input, std::move(expr), strategy);

  std::vector<int32_t> expected;
  for (size_t i = 0; i < values.size(); ++i) {
    if (valids[i] && values[i] > 2) { expected.push_back(values[i]); }
  }
  REQUIRE(ov.num_columns() == 1);
  REQUIRE(ov.num_rows() == static_cast<cudf::size_type>(expected.size()));
  REQUIRE(copy_column_to_host<int32_t>(ov.column(0)) == expected);
}

TEMPLATE_TEST_CASE("select COMPARE_NOT_DISTINCT_FROM",
                   "[expression_executor]",
                   mat_strategy,
                   ast_interpret_strategy)
{
  constexpr auto strategy = TestType::value;
  auto* space             = get_default_gpu_space();
  REQUIRE(space != nullptr);

  // Input: {10, NULL, 30, NULL, 50}. Under null-safe equality NULL ≠ 30, so only
  // the real 30 matches NOT_DISTINCT_FROM; the two NULLs are filtered out.
  std::vector<int32_t> values = {10, 99, 30, 99, 50};
  std::vector<bool> valids    = {true, false, true, false, true};
  auto input                  = make_int32_batch_with_nulls(*space, values, valids);

  auto expr = make_cmp(sirius::comparison_type::not_distinct_from, make_ref(0), make_int_const(30));

  auto [in_batch, out_batch, iv, ov] = run_select(*space, input, std::move(expr), strategy);
  REQUIRE(ov.num_rows() == 1);
  REQUIRE(ov.column(0).null_count() == 0);
  REQUIRE(copy_column_to_host<int32_t>(ov.column(0)) == std::vector<int32_t>{30});
}

TEMPLATE_TEST_CASE("select COMPARE_DISTINCT_FROM",
                   "[expression_executor]",
                   mat_strategy,
                   ast_interpret_strategy)
{
  constexpr auto strategy = TestType::value;
  auto* space             = get_default_gpu_space();
  REQUIRE(space != nullptr);

  // Input: {10, NULL, 30, NULL, 50}. Under null-safe equality NULL ≠ 30, so
  // everything except the real 30 matches DISTINCT_FROM — both NULLs pass through.
  std::vector<int32_t> values = {10, 99, 30, 99, 50};
  std::vector<bool> valids    = {true, false, true, false, true};
  auto input                  = make_int32_batch_with_nulls(*space, values, valids);

  auto expr = make_cmp(sirius::comparison_type::distinct_from, make_ref(0), make_int_const(30));

  auto [in_batch, out_batch, iv, ov] = run_select(*space, input, std::move(expr), strategy);
  REQUIRE(ov.num_rows() == 4);
  REQUIRE(ov.column(0).null_count() == 2);
}

TEMPLATE_TEST_CASE("select NOT operator",
                   "[expression_executor]",
                   mat_strategy,
                   ast_interpret_strategy,
                   ast_jit_strategy)
{
  constexpr auto strategy = TestType::value;
  auto* space             = get_default_gpu_space();
  REQUIRE(space != nullptr);

  auto input =
    make_input_batch(*space, {cudf::data_type{cudf::type_id::INT32}}, {std::pair<int, int>{0, 9}});

  // NOT (col0 > 5) — should keep rows where col0 <= 5
  auto cmp  = make_cmp(sirius::comparison_type::gt, make_ref(0), make_int_const(5));
  auto expr = make_unary(sirius::ast::unary_op::kind::op_not, std::move(cmp));

  auto [in_batch, out_batch, iv, ov] = run_select(*space, input, std::move(expr), strategy);
  auto in_vals                       = copy_column_to_host<int32_t>(iv.column(0));
  std::vector<int32_t> expected;
  for (auto v : in_vals) {
    if (v <= 5) { expected.push_back(v); }
  }
  REQUIRE(copy_column_to_host<int32_t>(ov.column(0)) == expected);
}

// ---------------------------------------------------------------------------
// Conjunction with AST breaker: AND/OR mixing AST-capable and non-AST nodes
// ---------------------------------------------------------------------------

TEMPLATE_TEST_CASE("select conjunction with AST breaker",
                   "[expression_executor]",
                   mat_strategy,
                   ast_interpret_strategy,
                   ast_jit_strategy)
{
  constexpr auto strategy = TestType::value;
  auto* space             = get_default_gpu_space();
  REQUIRE(space != nullptr);

  // Table: (INT32 col0, STRING col1)
  auto input = make_input_batch(
    *space,
    {cudf::data_type{cudf::type_id::INT32}, cudf::data_type{cudf::type_id::STRING}},
    {std::pair<int, int>{0, 9}, std::pair<int, int>{1, 5}});

  // WHERE col0 > 3 AND col1 LIKE 'str_%'
  // col0 > 3 is AST-capable, LIKE is not — forces mixed materialization
  auto cmp = make_cmp(sirius::comparison_type::gt, make_ref(0), make_int_const(3));

  std::vector<std::unique_ptr<ast_node>> like_children;
  like_children.push_back(make_ref(1));
  like_children.push_back(make_str_const("str_%"));
  auto like = make_func(
    sirius::function_id::like, std::move(like_children), logical_type::make(type_id::BOOLEAN));

  std::vector<std::unique_ptr<ast_node>> conj_children;
  conj_children.push_back(std::move(cmp));
  conj_children.push_back(std::move(like));
  auto expr = make_conj(sirius::ast::conjunction::kind::op_and, std::move(conj_children));

  auto [in_batch, out_batch, iv, ov] = run_select(*space, input, std::move(expr), strategy);
  auto in_col0                       = copy_column_to_host<int32_t>(iv.column(0));
  auto in_col1                       = copy_string_column_to_host(iv.column(1));

  std::vector<int32_t> expected_col0;
  for (auto v : in_col0) {
    // "str_%" matches everything produced by make_input_batch (all start with "str_")
    if (v > 3) { expected_col0.push_back(v); }
  }
  REQUIRE(copy_column_to_host<int32_t>(ov.column(0)) == expected_col0);
}

TEMPLATE_TEST_CASE("select OR conjunction",
                   "[expression_executor]",
                   mat_strategy,
                   ast_interpret_strategy,
                   ast_jit_strategy)
{
  constexpr auto strategy = TestType::value;
  auto* space             = get_default_gpu_space();
  REQUIRE(space != nullptr);

  auto input =
    make_input_batch(*space, {cudf::data_type{cudf::type_id::INT32}}, {std::pair<int, int>{0, 20}});

  // col0 < 3 OR col0 > 17
  auto lt = make_cmp(sirius::comparison_type::lt, make_ref(0), make_int_const(3));
  auto gt = make_cmp(sirius::comparison_type::gt, make_ref(0), make_int_const(17));

  std::vector<std::unique_ptr<ast_node>> children;
  children.push_back(std::move(lt));
  children.push_back(std::move(gt));
  auto expr = make_conj(sirius::ast::conjunction::kind::op_or, std::move(children));

  auto [in_batch, out_batch, iv, ov] = run_select(*space, input, std::move(expr), strategy);
  auto in_vals                       = copy_column_to_host<int32_t>(iv.column(0));
  std::vector<int32_t> expected;
  for (auto v : in_vals) {
    if (v < 3 || v > 17) { expected.push_back(v); }
  }
  REQUIRE(copy_column_to_host<int32_t>(ov.column(0)) == expected);
}

// ---------------------------------------------------------------------------
// Nested compound: CASE inside a comparison inside a conjunction
// Exercises AST breaker (CASE) nested within AST-capable nodes
// ---------------------------------------------------------------------------

TEMPLATE_TEST_CASE("select with nested CASE in predicate",
                   "[expression_executor]",
                   mat_strategy,
                   ast_interpret_strategy,
                   ast_jit_strategy)
{
  constexpr auto strategy = TestType::value;
  auto* space             = get_default_gpu_space();
  REQUIRE(space != nullptr);

  auto input = make_input_batch(
    *space, {cudf::data_type{cudf::type_id::INT32}}, {std::pair<int, int>{0, 100}});

  // WHERE (CASE WHEN col0 > 50 THEN col0 ELSE 0 END) > 75
  std::vector<sirius::ast::case_expr::when_then> cases;
  cases.push_back(sirius::ast::case_expr::when_then{
    make_cmp(sirius::comparison_type::gt, make_ref(0), make_int_const(50)), make_ref(0)});
  auto case_node = std::make_unique<ast_node>(sirius::ast::case_expr{
    std::move(cases), make_int_const(0), logical_type::make(type_id::INTEGER)});

  auto expr = make_cmp(sirius::comparison_type::gt, std::move(case_node), make_int_const(75));

  auto [in_batch, out_batch, iv, ov] = run_select(*space, input, std::move(expr), strategy);
  auto in_vals                       = copy_column_to_host<int32_t>(iv.column(0));
  std::vector<int32_t> expected;
  for (auto v : in_vals) {
    int32_t case_result = v > 50 ? v : 0;
    if (case_result > 75) { expected.push_back(v); }
  }
  REQUIRE(copy_column_to_host<int32_t>(ov.column(0)) == expected);
}

// ---------------------------------------------------------------------------
// IN with conjunction (multi-column filter, mirrors a TPC-H style predicate)
// ---------------------------------------------------------------------------

TEMPLATE_TEST_CASE("select IN with conjunction multi-column",
                   "[expression_executor]",
                   mat_strategy,
                   ast_interpret_strategy,
                   ast_jit_strategy)
{
  constexpr auto strategy = TestType::value;
  auto* space             = get_default_gpu_space();
  REQUIRE(space != nullptr);

  auto input =
    make_input_batch(*space,
                     {cudf::data_type{cudf::type_id::INT32}, cudf::data_type{cudf::type_id::INT64}},
                     {std::pair<int, int>{0, 9}, std::pair<int, int>{0, 50}});

  // WHERE col0 IN (1, 3, 5, 7) AND col1 >= 20
  std::vector<std::unique_ptr<ast_node>> in_values;
  in_values.push_back(make_int_const(1));
  in_values.push_back(make_int_const(3));
  in_values.push_back(make_int_const(5));
  in_values.push_back(make_int_const(7));
  auto in_expr = make_in(make_ref(0), std::move(in_values), /*negated=*/false);

  auto cmp = make_cmp(sirius::comparison_type::ge, make_ref(1), make_bigint_const(20));

  std::vector<std::unique_ptr<ast_node>> conj_children;
  conj_children.push_back(std::move(in_expr));
  conj_children.push_back(std::move(cmp));
  auto expr = make_conj(sirius::ast::conjunction::kind::op_and, std::move(conj_children));

  auto [in_batch, out_batch, iv, ov] = run_select(*space, input, std::move(expr), strategy);
  auto in_col0                       = copy_column_to_host<int32_t>(iv.column(0));
  auto in_col1                       = copy_column_to_host<int64_t>(iv.column(1));
  std::vector<int32_t> expected_col0;
  std::vector<int64_t> expected_col1;
  for (size_t i = 0; i < in_col0.size(); ++i) {
    bool in_set = in_col0[i] == 1 || in_col0[i] == 3 || in_col0[i] == 5 || in_col0[i] == 7;
    if (in_set && in_col1[i] >= 20) {
      expected_col0.push_back(in_col0[i]);
      expected_col1.push_back(in_col1[i]);
    }
  }
  REQUIRE(copy_column_to_host<int32_t>(ov.column(0)) == expected_col0);
  REQUIRE(copy_column_to_host<int64_t>(ov.column(1)) == expected_col1);
}

// ---------------------------------------------------------------------------
// Arithmetic in projection combined with CASE — complex execute() output
// ---------------------------------------------------------------------------

TEMPLATE_TEST_CASE("execute mixed arithmetic and CASE projection",
                   "[expression_executor]",
                   mat_strategy,
                   ast_interpret_strategy,
                   ast_jit_strategy)
{
  constexpr auto strategy = TestType::value;
  auto* space             = get_default_gpu_space();
  REQUIRE(space != nullptr);

  auto input =
    make_input_batch(*space,
                     {cudf::data_type{cudf::type_id::INT32}, cudf::data_type{cudf::type_id::INT32}},
                     {std::pair<int, int>{1, 50}, std::pair<int, int>{1, 10}});

  // Output: [col0 + col1, CASE WHEN col0 > 25 THEN col1 * 2 ELSE col1 END]
  std::vector<std::unique_ptr<ast_node>> exprs;

  // Expression 0: col0 + col1
  {
    std::vector<std::unique_ptr<ast_node>> children;
    children.push_back(make_ref(0));
    children.push_back(make_ref(1));
    exprs.push_back(make_func(
      sirius::function_id::add, std::move(children), logical_type::make(type_id::INTEGER)));
  }

  // Expression 1: CASE WHEN col0 > 25 THEN col1 * 2 ELSE col1 END
  {
    std::vector<std::unique_ptr<ast_node>> mul_children;
    mul_children.push_back(make_ref(1));
    mul_children.push_back(make_int_const(2));
    auto then_expr = make_func(
      sirius::function_id::mul, std::move(mul_children), logical_type::make(type_id::INTEGER));

    std::vector<sirius::ast::case_expr::when_then> cases;
    cases.push_back(sirius::ast::case_expr::when_then{
      make_cmp(sirius::comparison_type::gt, make_ref(0), make_int_const(25)),
      std::move(then_expr)});
    exprs.push_back(std::make_unique<ast_node>(
      sirius::ast::case_expr{std::move(cases), make_ref(1), logical_type::make(type_id::INTEGER)}));
  }

  auto [in_batch, out_batch, iv, ov] = run_execute(*space, input, std::move(exprs), strategy);
  REQUIRE(ov.num_columns() == 2);
  REQUIRE(ov.num_rows() == iv.num_rows());

  auto in0  = copy_column_to_host<int32_t>(iv.column(0));
  auto in1  = copy_column_to_host<int32_t>(iv.column(1));
  auto out0 = copy_column_to_host<int32_t>(ov.column(0));
  auto out1 = copy_column_to_host<int32_t>(ov.column(1));
  for (size_t i = 0; i < in0.size(); ++i) {
    REQUIRE(out0[i] == in0[i] + in1[i]);
    REQUIRE(out1[i] == (in0[i] > 25 ? in1[i] * 2 : in1[i]));
  }
}

// ---------------------------------------------------------------------------
// native_ast — exercise each migrated AST alternative through the non-owning
// AST executor ctor, building the AST by hand (no DuckDB allocation involved).
// ---------------------------------------------------------------------------

namespace {

// Build a single-column INT32 input batch [0..127] and return its table view.
struct int32_batch {
  std::shared_ptr<data_batch> batch;
  cudf::table_view view;
};

int32_batch make_int32_input(memory_space& space)
{
  auto batch =
    make_input_batch(space, {cudf::data_type{cudf::type_id::INT32}}, {std::pair<int, int>{0, 100}});
  auto ro       = batch->get_read_only();
  auto& in_repr = ro->get_data()->cast<gpu_table_representation>();
  return {batch, in_repr.get_table_view()};
}

std::unique_ptr<sirius::ast::node> ref_node_native(uint32_t idx) { return make_ref(idx); }

std::unique_ptr<sirius::ast::node> int_const_node_native(int32_t v) { return make_int_const(v); }

std::unique_ptr<cudf::table> run_native_ast(memory_space& space,
                                            sirius::ast::node const* expr_ptr,
                                            cudf::table_view tv,
                                            exp_strategy_enum strategy = MAT)
{
  exp_executor executor(expr_ptr, get_resource_ref(space), cudf::get_default_stream(), strategy);
  return executor.execute(tv);
}

}  // namespace

TEST_CASE("native_ast - reference identity", "[expression_executor_ast_native][reference]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  auto in = make_int32_input(*space);

  auto hand_ast = ref_node_native(0);
  auto out      = run_native_ast(*space, hand_ast.get(), in.view, MAT);
  REQUIRE(out);
  auto in_host  = copy_column_to_host<int32_t>(in.view.column(0));
  auto out_host = copy_column_to_host<int32_t>(out->view().column(0));
  REQUIRE(out_host == in_host);
}

TEST_CASE("native_ast - constant INTEGER", "[expression_executor_ast_native][constant]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  auto in = make_int32_input(*space);

  auto hand_ast = std::make_unique<sirius::ast::node>(sirius::ast::constant{
    sirius::value{int32_t{42}}, sirius::logical_type::make(sirius::type_id::INTEGER)});

  auto out = run_native_ast(*space, hand_ast.get(), in.view, MAT);
  REQUIRE(out);
  std::vector<int32_t> expected(in.view.num_rows(), 42);
  REQUIRE(copy_column_to_host<int32_t>(out->view().column(0)) == expected);
}

TEST_CASE("native_ast - constant VARCHAR", "[expression_executor_ast_native][constant]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  auto input =
    make_input_batch(*space, {cudf::data_type{cudf::type_id::STRING}}, {std::pair<int, int>{1, 5}});
  auto ro       = input->get_read_only();
  auto& in_repr = ro->get_data()->cast<gpu_table_representation>();
  auto tv       = in_repr.get_table_view();

  auto hand_ast = std::make_unique<sirius::ast::node>(sirius::ast::constant{
    sirius::value{std::string{"hello"}}, sirius::logical_type::make(sirius::type_id::VARCHAR)});

  auto out = run_native_ast(*space, hand_ast.get(), tv, MAT);
  REQUIRE(out);
  std::vector<std::string> expected(tv.num_rows(), "hello");
  REQUIRE(copy_string_column_to_host(out->view().column(0)) == expected);
}

TEST_CASE("native_ast - comparison EQUAL (MATERIALIZE)",
          "[expression_executor_ast_native][comparison]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  auto in = make_int32_input(*space);

  auto hand_ast = std::make_unique<sirius::ast::node>(sirius::ast::comparison{
    sirius::comparison_type::equal, ref_node_native(0), int_const_node_native(42)});

  auto out = run_native_ast(*space, hand_ast.get(), in.view, MAT);
  REQUIRE(out);
  auto in_host  = copy_column_to_host<int32_t>(in.view.column(0));
  auto out_host = copy_bool_column_to_host(out->view().column(0));
  REQUIRE(out_host.size() == in_host.size());
  for (size_t i = 0; i < in_host.size(); ++i) {
    REQUIRE(out_host[i] == (in_host[i] == 42 ? 1U : 0U));
  }
}

TEST_CASE("native_ast - comparison LESS_THAN (AST_INTERPRET)",
          "[expression_executor_ast_native][comparison]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  auto in = make_int32_input(*space);

  auto hand_ast = std::make_unique<sirius::ast::node>(sirius::ast::comparison{
    sirius::comparison_type::lt, ref_node_native(0), int_const_node_native(50)});

  auto out = run_native_ast(*space, hand_ast.get(), in.view, exp_strategy_enum::AST_INTERPRET);
  REQUIRE(out);
  auto in_host  = copy_column_to_host<int32_t>(in.view.column(0));
  auto out_host = copy_bool_column_to_host(out->view().column(0));
  REQUIRE(out_host.size() == in_host.size());
  for (size_t i = 0; i < in_host.size(); ++i) {
    REQUIRE(out_host[i] == (in_host[i] < 50 ? 1U : 0U));
  }
}

TEST_CASE("native_ast - conjunction AND (MATERIALIZE)",
          "[expression_executor_ast_native][conjunction]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  // Build a 2-column INT32 batch with col0 in [0, 100) and col1 in [0, 100).
  auto input =
    make_input_batch(*space,
                     {cudf::data_type{cudf::type_id::INT32}, cudf::data_type{cudf::type_id::INT32}},
                     {std::pair<int, int>{0, 100}, std::pair<int, int>{0, 100}});
  auto ro       = input->get_read_only();
  auto& in_repr = ro->get_data()->cast<gpu_table_representation>();
  auto tv       = in_repr.get_table_view();

  std::vector<std::unique_ptr<sirius::ast::node>> children;
  children.push_back(std::make_unique<sirius::ast::node>(sirius::ast::comparison{
    sirius::comparison_type::gt, ref_node_native(0), int_const_node_native(10)}));
  children.push_back(std::make_unique<sirius::ast::node>(sirius::ast::comparison{
    sirius::comparison_type::lt, ref_node_native(1), int_const_node_native(90)}));
  auto hand_ast = std::make_unique<sirius::ast::node>(
    sirius::ast::conjunction{sirius::ast::conjunction::kind::op_and, std::move(children)});

  auto out = run_native_ast(*space, hand_ast.get(), tv, MAT);
  REQUIRE(out);
  auto c0       = copy_column_to_host<int32_t>(tv.column(0));
  auto c1       = copy_column_to_host<int32_t>(tv.column(1));
  auto out_host = copy_bool_column_to_host(out->view().column(0));
  REQUIRE(out_host.size() == c0.size());
  for (size_t i = 0; i < c0.size(); ++i) {
    REQUIRE(out_host[i] == ((c0[i] > 10 && c1[i] < 90) ? 1U : 0U));
  }
}

TEST_CASE("native_ast - between (MATERIALIZE)", "[expression_executor_ast_native][between]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  auto in = make_int32_input(*space);

  auto hand_ast =
    std::make_unique<sirius::ast::node>(sirius::ast::between{ref_node_native(0),
                                                             int_const_node_native(5),
                                                             int_const_node_native(15),
                                                             /*lower_inclusive=*/true,
                                                             /*upper_inclusive=*/true});

  auto out = run_native_ast(*space, hand_ast.get(), in.view, MAT);
  REQUIRE(out);
  auto in_host  = copy_column_to_host<int32_t>(in.view.column(0));
  auto out_host = copy_bool_column_to_host(out->view().column(0));
  REQUIRE(out_host.size() == in_host.size());
  for (size_t i = 0; i < in_host.size(); ++i) {
    REQUIRE(out_host[i] == ((in_host[i] >= 5 && in_host[i] <= 15) ? 1U : 0U));
  }
}

TEST_CASE("native_ast - unary_op NOT (MATERIALIZE)", "[expression_executor_ast_native][unary_op]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  auto in = make_int32_input(*space);

  auto cmp      = std::make_unique<sirius::ast::node>(sirius::ast::comparison{
    sirius::comparison_type::equal, ref_node_native(0), int_const_node_native(5)});
  auto hand_ast = std::make_unique<sirius::ast::node>(
    sirius::ast::unary_op{sirius::ast::unary_op::kind::op_not, std::move(cmp)});

  auto out = run_native_ast(*space, hand_ast.get(), in.view, MAT);
  REQUIRE(out);
  auto in_host  = copy_column_to_host<int32_t>(in.view.column(0));
  auto out_host = copy_bool_column_to_host(out->view().column(0));
  REQUIRE(out_host.size() == in_host.size());
  for (size_t i = 0; i < in_host.size(); ++i) {
    REQUIRE(out_host[i] == (in_host[i] != 5 ? 1U : 0U));
  }
}

TEST_CASE("native_ast - unary_op IS_NULL (MATERIALIZE)",
          "[expression_executor_ast_native][unary_op]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  std::vector<int32_t> values{1, 2, 3, 4, 5};
  std::vector<bool> valids{true, false, true, false, true};
  auto batch    = make_int32_batch_with_nulls(*space, values, valids);
  auto ro       = batch->get_read_only();
  auto& in_repr = ro->get_data()->cast<gpu_table_representation>();
  auto tv       = in_repr.get_table_view();

  auto hand_ast = std::make_unique<sirius::ast::node>(
    sirius::ast::unary_op{sirius::ast::unary_op::kind::op_is_null, ref_node_native(0)});

  auto out = run_native_ast(*space, hand_ast.get(), tv, MAT);
  REQUIRE(out);
  auto out_host = copy_bool_column_to_host(out->view().column(0));
  REQUIRE(out_host.size() == valids.size());
  for (size_t i = 0; i < valids.size(); ++i) {
    REQUIRE(out_host[i] == (valids[i] ? 0U : 1U));
  }
}

TEST_CASE("native_ast - unary_op IS_NOT_NULL (MATERIALIZE)",
          "[expression_executor_ast_native][unary_op]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  std::vector<int32_t> values{10, 20, 30, 40};
  std::vector<bool> valids{true, false, true, true};
  auto batch    = make_int32_batch_with_nulls(*space, values, valids);
  auto ro       = batch->get_read_only();
  auto& in_repr = ro->get_data()->cast<gpu_table_representation>();
  auto tv       = in_repr.get_table_view();

  auto hand_ast = std::make_unique<sirius::ast::node>(
    sirius::ast::unary_op{sirius::ast::unary_op::kind::op_is_not_null, ref_node_native(0)});

  auto out = run_native_ast(*space, hand_ast.get(), tv, MAT);
  REQUIRE(out);
  auto out_host = copy_bool_column_to_host(out->view().column(0));
  REQUIRE(out_host.size() == valids.size());
  for (size_t i = 0; i < valids.size(); ++i) {
    REQUIRE(out_host[i] == (valids[i] ? 1U : 0U));
  }
}

TEST_CASE("native_ast - in_list IN (MATERIALIZE)", "[expression_executor_ast_native][in_list]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  auto in = make_int32_input(*space);

  std::vector<std::unique_ptr<sirius::ast::node>> values;
  values.push_back(int_const_node_native(1));
  values.push_back(int_const_node_native(3));
  values.push_back(int_const_node_native(5));
  auto hand_ast = std::make_unique<sirius::ast::node>(
    sirius::ast::in_list{ref_node_native(0), std::move(values), /*negated=*/false});

  auto out = run_native_ast(*space, hand_ast.get(), in.view, MAT);
  REQUIRE(out);
  auto in_host  = copy_column_to_host<int32_t>(in.view.column(0));
  auto out_host = copy_bool_column_to_host(out->view().column(0));
  REQUIRE(out_host.size() == in_host.size());
  for (size_t i = 0; i < in_host.size(); ++i) {
    bool const expected = (in_host[i] == 1 || in_host[i] == 3 || in_host[i] == 5);
    REQUIRE(out_host[i] == (expected ? 1U : 0U));
  }
}

TEST_CASE("native_ast - coalesce (MATERIALIZE)", "[expression_executor_ast_native][coalesce]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  std::vector<int32_t> values{7, 8, 9, 10, 11};
  std::vector<bool> valids{true, false, true, false, true};
  auto batch    = make_int32_batch_with_nulls(*space, values, valids);
  auto ro       = batch->get_read_only();
  auto& in_repr = ro->get_data()->cast<gpu_table_representation>();
  auto tv       = in_repr.get_table_view();

  std::vector<std::unique_ptr<sirius::ast::node>> children;
  children.push_back(ref_node_native(0));
  children.push_back(int_const_node_native(99));
  auto hand_ast = std::make_unique<sirius::ast::node>(sirius::ast::coalesce{
    std::move(children), sirius::logical_type::make(sirius::type_id::INTEGER)});

  auto out = run_native_ast(*space, hand_ast.get(), tv, MAT);
  REQUIRE(out);
  auto out_host = copy_column_to_host<int32_t>(out->view().column(0));
  REQUIRE(out_host.size() == values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    REQUIRE(out_host[i] == (valids[i] ? values[i] : 99));
  }
}

TEST_CASE("native_ast - cast INTEGER->BIGINT (MATERIALIZE)",
          "[expression_executor_ast_native][cast]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  auto in = make_int32_input(*space);

  auto hand_ast = std::make_unique<sirius::ast::node>(
    sirius::ast::cast{ref_node_native(0),
                      sirius::logical_type::make(sirius::type_id::BIGINT),
                      /*try_cast=*/false});

  auto out = run_native_ast(*space, hand_ast.get(), in.view, MAT);
  REQUIRE(out);
  auto in_host  = copy_column_to_host<int32_t>(in.view.column(0));
  auto out_host = copy_column_to_host<int64_t>(out->view().column(0));
  REQUIRE(out_host.size() == in_host.size());
  for (size_t i = 0; i < in_host.size(); ++i) {
    REQUIRE(out_host[i] == static_cast<int64_t>(in_host[i]));
  }
}

TEST_CASE("native_ast - function_call add (MATERIALIZE)",
          "[expression_executor_ast_native][function_call]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  // 2-column INT32 batch.
  auto input =
    make_input_batch(*space,
                     {cudf::data_type{cudf::type_id::INT32}, cudf::data_type{cudf::type_id::INT32}},
                     {std::pair<int, int>{0, 50}, std::pair<int, int>{0, 50}});
  auto ro       = input->get_read_only();
  auto& in_repr = ro->get_data()->cast<gpu_table_representation>();
  auto tv       = in_repr.get_table_view();

  std::vector<std::unique_ptr<sirius::ast::node>> args;
  args.push_back(ref_node_native(0));
  args.push_back(ref_node_native(1));
  auto hand_ast = std::make_unique<sirius::ast::node>(
    sirius::ast::function_call{sirius::function_id::add,
                               std::move(args),
                               sirius::logical_type::make(sirius::type_id::INTEGER)});

  auto out = run_native_ast(*space, hand_ast.get(), tv, MAT);
  REQUIRE(out);
  auto c0       = copy_column_to_host<int32_t>(tv.column(0));
  auto c1       = copy_column_to_host<int32_t>(tv.column(1));
  auto out_host = copy_column_to_host<int32_t>(out->view().column(0));
  REQUIRE(out_host.size() == c0.size());
  for (size_t i = 0; i < c0.size(); ++i) {
    REQUIRE(out_host[i] == c0[i] + c1[i]);
  }
}

TEST_CASE("native_ast - function_call add (AST_INTERPRET)",
          "[expression_executor_ast_native][function_call]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  auto input =
    make_input_batch(*space,
                     {cudf::data_type{cudf::type_id::INT32}, cudf::data_type{cudf::type_id::INT32}},
                     {std::pair<int, int>{0, 50}, std::pair<int, int>{0, 50}});
  auto ro       = input->get_read_only();
  auto& in_repr = ro->get_data()->cast<gpu_table_representation>();
  auto tv       = in_repr.get_table_view();

  std::vector<std::unique_ptr<sirius::ast::node>> args;
  args.push_back(ref_node_native(0));
  args.push_back(ref_node_native(1));
  auto hand_ast = std::make_unique<sirius::ast::node>(
    sirius::ast::function_call{sirius::function_id::add,
                               std::move(args),
                               sirius::logical_type::make(sirius::type_id::INTEGER)});

  auto out = run_native_ast(*space, hand_ast.get(), tv, exp_strategy_enum::AST_INTERPRET);
  REQUIRE(out);
  auto c0       = copy_column_to_host<int32_t>(tv.column(0));
  auto c1       = copy_column_to_host<int32_t>(tv.column(1));
  auto out_host = copy_column_to_host<int32_t>(out->view().column(0));
  REQUIRE(out_host.size() == c0.size());
  for (size_t i = 0; i < c0.size(); ++i) {
    REQUIRE(out_host[i] == c0[i] + c1[i]);
  }
}

TEST_CASE("native_ast - case_expr WHEN/THEN/ELSE (MATERIALIZE)",
          "[expression_executor_ast_native][case_expr]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  auto in = make_int32_input(*space);

  std::vector<sirius::ast::case_expr::when_then> cases;
  cases.emplace_back();
  cases.back().when_ = std::make_unique<sirius::ast::node>(sirius::ast::comparison{
    sirius::comparison_type::equal, ref_node_native(0), int_const_node_native(5)});
  cases.back().then_ = int_const_node_native(100);
  auto hand_ast      = std::make_unique<sirius::ast::node>(
    sirius::ast::case_expr{std::move(cases),
                           int_const_node_native(0),
                           sirius::logical_type::make(sirius::type_id::INTEGER)});

  auto out = run_native_ast(*space, hand_ast.get(), in.view, MAT);
  REQUIRE(out);
  auto in_host  = copy_column_to_host<int32_t>(in.view.column(0));
  auto out_host = copy_column_to_host<int32_t>(out->view().column(0));
  REQUIRE(out_host.size() == in_host.size());
  for (size_t i = 0; i < in_host.size(); ++i) {
    REQUIRE(out_host[i] == (in_host[i] == 5 ? 100 : 0));
  }
}

// Native-AST executor coverage for the null-safe comparison kinds
// (COMPARE_DISTINCT_FROM / COMPARE_NOT_DISTINCT_FROM) built directly as
// sirius::ast::comparison nodes — the comparison-kind enum is preserved end to
// end through the executor without any DuckDB Bound*Expression round-trip
// (sirius-db/sirius#699). DuckDB->Sirius lowering of these kinds is covered by
// the dedicated [ast_from_duckdb] suite.
TEST_CASE("native_ast - comparison DISTINCT_FROM / NOT_DISTINCT_FROM executes",
          "[expression_executor_ast_native][comparison]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  // Input: {10, NULL, 30, NULL, 50}.
  std::vector<int32_t> values = {10, 99, 30, 99, 50};
  std::vector<bool> valids    = {true, false, true, false, true};

  {
    auto batch    = make_int32_batch_with_nulls(*space, values, valids);
    auto ro       = batch->get_read_only();
    auto& in_repr = ro->get_data()->cast<gpu_table_representation>();
    auto tv       = in_repr.get_table_view();

    auto node = std::make_unique<sirius::ast::node>(sirius::ast::comparison{
      sirius::comparison_type::not_distinct_from, ref_node_native(0), int_const_node_native(30)});
    REQUIRE(node->holds<sirius::ast::comparison>());
    REQUIRE(node->get<sirius::ast::comparison>().op == sirius::comparison_type::not_distinct_from);

    // Only the real 30 is null-safe-equal to 30; the two NULLs are not.
    auto out = run_native_ast(*space, node.get(), tv, MAT);
    REQUIRE(out);
    auto out_host = copy_bool_column_to_host(out->view().column(0));
    REQUIRE(out_host.size() == values.size());
    for (size_t i = 0; i < values.size(); ++i) {
      REQUIRE(out_host[i] == ((valids[i] && values[i] == 30) ? 1U : 0U));
    }
  }

  {
    auto batch    = make_int32_batch_with_nulls(*space, values, valids);
    auto ro       = batch->get_read_only();
    auto& in_repr = ro->get_data()->cast<gpu_table_representation>();
    auto tv       = in_repr.get_table_view();

    auto node = std::make_unique<sirius::ast::node>(sirius::ast::comparison{
      sirius::comparison_type::distinct_from, ref_node_native(0), int_const_node_native(30)});
    REQUIRE(node->holds<sirius::ast::comparison>());
    REQUIRE(node->get<sirius::ast::comparison>().op == sirius::comparison_type::distinct_from);

    // Everything except the real 30 is null-safe-distinct from 30 (both NULLs included).
    auto out = run_native_ast(*space, node.get(), tv, MAT);
    REQUIRE(out);
    auto out_host = copy_bool_column_to_host(out->view().column(0));
    REQUIRE(out_host.size() == values.size());
    for (size_t i = 0; i < values.size(); ++i) {
      REQUIRE(out_host[i] == ((valids[i] && values[i] == 30) ? 0U : 1U));
    }
  }
}
