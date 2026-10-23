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
#include "codegen/jit/kernel_cache.hpp"

#include <cucascade/cudf/gpu_data_representation.hpp>
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
#include <expression/ast/constant_range.hpp>
#include <expression/ast/function_call.hpp>
#include <expression/ast/in_list.hpp>
#include <expression/ast/node.hpp>
#include <expression/ast/reference.hpp>
#include <expression/ast/unary_op.hpp>
#include <expression/function_id.hpp>
#include <expression/join_condition.hpp>
#include <expression/value.hpp>
#include <expression_evaluator/expression_evaluator.hpp>
#include <helper/logical_type.hpp>
#include <helper/numeric_narrowing.hpp>
#include <memory/sirius_memory_reservation_manager.hpp>
#include <sirius/exception.hpp>

// cudf, etc.
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/unary.hpp>

#include <cuda_runtime_api.h>

// standard library
#include <cstdint>
#include <limits>
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
    .set_per_numa_region_capacity(host_capacity)
    .use_gpu_id_as_host_id()
    .set_reservation_fraction_per_numa_region(limit_ratio);
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
  auto* mask_ptr = reinterpret_cast<cudf::bitmask_type*>(null_mask.data());

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
    auto* mask_ptr = reinterpret_cast<cudf::bitmask_type*>(null_mask.data());
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

// A column of type `carrier` holding `values`. make_input_batch's generator stops at
// INT32, INT64, and STRING; the narrow-domain paths need carriers strictly narrower than the
// reference's declared type, and a reference pair needs two of them side by side.
template <typename T>
std::unique_ptr<cudf::column> make_carrier_column(memory_space& space,
                                                  cudf::data_type carrier,
                                                  const std::vector<T>& values)
{
  auto stream = cudf::get_default_stream();
  auto col    = cudf::make_numeric_column(carrier,
                                       static_cast<cudf::size_type>(values.size()),
                                       cudf::mask_state::UNALLOCATED,
                                       stream,
                                       get_resource_ref(space));
  cudaMemcpy(col->mutable_view().data<T>(),
             values.data(),
             sizeof(T) * values.size(),
             cudaMemcpyHostToDevice);
  return col;
}

std::shared_ptr<data_batch> make_batch_from_columns(
  memory_space& space, std::vector<std::unique_ptr<cudf::column>> columns)
{
  auto table = std::make_unique<cudf::table>(std::move(columns));
  auto gpu_repr =
    std::make_unique<gpu_table_representation>(std::move(table), space, cudf::get_default_stream());
  return data_batch::make(::sirius::get_next_batch_id(), std::move(gpu_repr));
}

using exp_executor      = ::sirius::expression_evaluator;
using exp_strategy_enum = ::sirius::expression_evaluator_strategy;
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
// expression_evaluator test suites.

using sirius::logical_type;
using sirius::type_id;

// Shorthand: build executor, run evaluate(), return output table view and input table view.
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
  exp_executor executor(ast_nodes,
                        get_resource_ref(space),
                        cudf::get_default_stream(),
                        sirius::expression_query_policy{strategy});
  auto input_ro     = input_batch->to_read_only();
  auto& in_repr     = input_ro.get_data()->cast<gpu_table_representation>();
  auto output_table = executor.evaluate(in_repr.get_table_view());
  REQUIRE(output_table != nullptr);
  auto output_batch = sirius::make_data_batch(std::move(output_table),
                                              *input_ro.get_memory_space(),
                                              cudf::get_default_stream(),
                                              sirius::telemetry::batch_telemetry_info{});
  auto output_ro    = output_batch->to_read_only();
  auto& out_repr    = output_ro.get_data()->cast<gpu_table_representation>();
  return {input_batch, output_batch, in_repr.get_table_view(), out_repr.get_table_view()};
}

exec_result run_select(memory_space& space,
                       std::shared_ptr<data_batch> const& input_batch,
                       std::unique_ptr<ast_node> node,
                       exp_strategy_enum strategy = MAT,
                       bool like_swar_fastpath    = true)
{
  exp_executor executor(*node,
                        get_resource_ref(space),
                        cudf::get_default_stream(),
                        sirius::expression_query_policy{strategy},
                        2,
                        like_swar_fastpath);
  auto input_ro     = input_batch->to_read_only();
  auto& in_repr     = input_ro.get_data()->cast<gpu_table_representation>();
  auto output_table = executor.select(in_repr.get_table_view());
  REQUIRE(output_table != nullptr);
  auto output_batch = sirius::make_data_batch(std::move(output_table),
                                              *input_ro.get_memory_space(),
                                              cudf::get_default_stream(),
                                              sirius::telemetry::batch_telemetry_info{});
  auto output_ro    = output_batch->to_read_only();
  auto& out_repr    = output_ro.get_data()->cast<gpu_table_representation>();
  return {input_batch, output_batch, in_repr.get_table_view(), out_repr.get_table_view()};
}

// Convenience: pack a single node into the projection driver's vector.
std::vector<std::unique_ptr<ast_node>> one(std::unique_ptr<ast_node> n)
{
  std::vector<std::unique_ptr<ast_node>> v;
  v.push_back(std::move(n));
  return v;
}

// DATE literals. The shared builders in ast_test_support.hpp stop at the literal kinds its GPU
// suites materialize; only the narrow-domain eligibility tests below need a date.
sirius::ast::constant date_literal(int32_t days)
{
  return sirius::ast::constant{sirius::value{sirius::date_value{days}},
                               logical_type::make(type_id::DATE)};
}

std::unique_ptr<ast_node> make_date_const(int32_t days)
{
  return std::make_unique<ast_node>(date_literal(days));
}
}  // namespace

TEST_CASE("restored numeric references are memoized per evaluate call",
          "[expression_evaluator][compressed_materialization]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  // Both constants sit outside the INT32 carrier so the narrow-domain comparison path declines
  // and both comparisons consume the restored (INT64) reference — exercising the memo.
  constexpr int64_t below_int32 = -5'000'000'000;
  constexpr int64_t above_int32 = 5'000'000'000;
  auto const bigint_type        = logical_type::make(type_id::BIGINT);
  std::vector<std::unique_ptr<ast_node>> comparisons;
  comparisons.push_back(make_cmp(
    sirius::comparison_type::gt, make_ref_typed(0, bigint_type), make_bigint_const(below_int32)));
  comparisons.push_back(make_cmp(
    sirius::comparison_type::lt, make_ref_typed(0, bigint_type), make_bigint_const(above_int32)));
  auto predicate = make_conj(sirius::ast::conjunction::kind::op_and, std::move(comparisons));

  // Deliberately use the active default strategy: the cache must work whether the configured
  // executor is materializing operators or building a cuDF AST.
  exp_executor executor(*predicate, get_resource_ref(*space));

  auto evaluate_and_check = [&](std::pair<int, int> range) {
    auto input = make_input_batch(
      *space, {cudf::data_type{cudf::type_id::INT32}}, {std::optional<std::pair<int, int>>{range}});
    auto input_ro   = input->to_read_only();
    auto input_view = input_ro.get_data()->cast<gpu_table_representation>().get_table_view();

    auto output = executor.evaluate(input_view);
    REQUIRE(output != nullptr);
    REQUIRE(output->num_columns() == 1);
    REQUIRE(executor.restored_reference_cast_count_for_testing() == 1);
    REQUIRE(executor.narrow_domain_comparison_count_for_testing() == 0);

    auto const input_values  = copy_column_to_host<int32_t>(input_view.column(0));
    auto const output_values = copy_bool_column_to_host(output->view().column(0));
    REQUIRE(output_values.size() == input_values.size());
    for (std::size_t i = 0; i < input_values.size(); ++i) {
      auto const value = static_cast<int64_t>(input_values[i]);
      REQUIRE(output_values[i] == ((value > below_int32 && value < above_int32) ? 1U : 0U));
    }
  };

  evaluate_and_check({-10, 30});
  // A second input must trigger one fresh restoration, not reuse a view into the first input.
  evaluate_and_check({10, 50});
}

TEST_CASE("narrow-domain comparisons evaluate on the narrowed carrier without restoration",
          "[expression_evaluator][compressed_materialization][narrow_domain]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  auto const bigint_type = logical_type::make(type_id::BIGINT);
  std::vector<std::unique_ptr<ast_node>> comparisons;
  comparisons.push_back(
    make_cmp(sirius::comparison_type::gt, make_ref_typed(0, bigint_type), make_bigint_const(5)));
  comparisons.push_back(
    make_cmp(sirius::comparison_type::lt, make_ref_typed(0, bigint_type), make_bigint_const(20)));
  auto predicate = make_conj(sirius::ast::conjunction::kind::op_and, std::move(comparisons));

  exp_executor executor(*predicate, get_resource_ref(*space));

  auto evaluate_and_check = [&](std::pair<int, int> range) {
    auto input = make_input_batch(
      *space, {cudf::data_type{cudf::type_id::INT32}}, {std::optional<std::pair<int, int>>{range}});
    auto input_ro   = input->to_read_only();
    auto input_view = input_ro.get_data()->cast<gpu_table_representation>().get_table_view();

    auto output = executor.evaluate(input_view);
    REQUIRE(output != nullptr);
    REQUIRE(output->num_columns() == 1);
    // Both constants fit the INT32 carrier: no restoration cast, both comparisons narrow.
    REQUIRE(executor.restored_reference_cast_count_for_testing() == 0);
    REQUIRE(executor.narrow_domain_comparison_count_for_testing() == 2);

    auto const input_values  = copy_column_to_host<int32_t>(input_view.column(0));
    auto const output_values = copy_bool_column_to_host(output->view().column(0));
    REQUIRE(output_values.size() == input_values.size());
    for (std::size_t i = 0; i < input_values.size(); ++i) {
      auto const value = static_cast<int64_t>(input_values[i]);
      REQUIRE(output_values[i] == ((value > 5 && value < 20) ? 1U : 0U));
    }
  };

  evaluate_and_check({-10, 30});
  evaluate_and_check({10, 50});
}

TEST_CASE("narrow-domain comparison engages on the MATERIALIZE binary-op path",
          "[expression_evaluator][compressed_materialization][narrow_domain]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  // A lone comparison lowers to one cuDF AST op, below the default min_ast_size of two, so the
  // executor takes the binary-op path even under an AST strategy.
  auto const bigint_type = logical_type::make(type_id::BIGINT);
  auto predicate =
    make_cmp(sirius::comparison_type::lt, make_ref_typed(0, bigint_type), make_bigint_const(15));

  exp_executor executor(*predicate, get_resource_ref(*space));

  auto input      = make_input_batch(*space,
                                     {cudf::data_type{cudf::type_id::INT32}},
                                     {std::optional<std::pair<int, int>>{{-10, 30}}});
  auto input_ro   = input->to_read_only();
  auto input_view = input_ro.get_data()->cast<gpu_table_representation>().get_table_view();

  auto output = executor.evaluate(input_view);
  REQUIRE(output != nullptr);
  REQUIRE(executor.restored_reference_cast_count_for_testing() == 0);
  REQUIRE(executor.narrow_domain_comparison_count_for_testing() == 1);

  auto const input_values  = copy_column_to_host<int32_t>(input_view.column(0));
  auto const output_values = copy_bool_column_to_host(output->view().column(0));
  for (std::size_t i = 0; i < input_values.size(); ++i) {
    REQUIRE(output_values[i] == ((static_cast<int64_t>(input_values[i]) < 15) ? 1U : 0U));
  }
}

TEST_CASE("narrow-domain comparison falls back when the constant does not fit the carrier",
          "[expression_evaluator][compressed_materialization][narrow_domain]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  // INT16 carrier, BIGINT declared type, constant beyond int16 range: restore path required.
  auto const bigint_type = logical_type::make(type_id::BIGINT);
  auto predicate         = make_cmp(
    sirius::comparison_type::lt, make_ref_typed(0, bigint_type), make_bigint_const(100'000));

  exp_executor executor(*predicate, get_resource_ref(*space));

  std::vector<int16_t> const values{-100, -1, 0, 1, 100, 32'000};
  auto mr  = get_resource_ref(*space);
  auto col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT16},
                                       static_cast<cudf::size_type>(values.size()),
                                       cudf::mask_state::UNALLOCATED,
                                       cudf::get_default_stream(),
                                       mr);
  cudaMemcpy(col->mutable_view().data<int16_t>(),
             values.data(),
             sizeof(int16_t) * values.size(),
             cudaMemcpyHostToDevice);
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(col));
  auto table    = std::make_unique<cudf::table>(std::move(cols));
  auto gpu_repr = std::make_unique<gpu_table_representation>(
    std::move(table), *space, cudf::get_default_stream());
  auto input      = data_batch::make(::sirius::get_next_batch_id(), std::move(gpu_repr));
  auto input_ro   = input->to_read_only();
  auto input_view = input_ro.get_data()->cast<gpu_table_representation>().get_table_view();

  auto output = executor.evaluate(input_view);
  REQUIRE(output != nullptr);
  REQUIRE(executor.restored_reference_cast_count_for_testing() == 1);
  REQUIRE(executor.narrow_domain_comparison_count_for_testing() == 0);

  auto const output_values = copy_bool_column_to_host(output->view().column(0));
  for (auto const value : output_values) {
    REQUIRE(value == 1U);  // every int16 value is < 100'000
  }
}

TEST_CASE("narrow-domain DECIMAL comparison requires a matching scale",
          "[expression_evaluator][compressed_materialization][narrow_domain]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  auto const decimal_type = logical_type::make_decimal(15, 2);
  std::vector<int32_t> const raw{100, 2399, 2400, 2401, 90'000};

  auto evaluate_with_constant = [&](std::unique_ptr<ast_node> constant) {
    auto predicate =
      make_cmp(sirius::comparison_type::lt, make_ref_typed(0, decimal_type), std::move(constant));
    exp_executor executor(*predicate, get_resource_ref(*space));
    auto input      = make_decimal32_batch(*space, /*scale=*/2, raw);
    auto input_ro   = input->to_read_only();
    auto input_view = input_ro.get_data()->cast<gpu_table_representation>().get_table_view();
    auto output     = executor.evaluate(input_view);
    REQUIRE(output != nullptr);
    return std::pair{executor.restored_reference_cast_count_for_testing(),
                     executor.narrow_domain_comparison_count_for_testing()};
  };

  // Same scale, unscaled value fits the DECIMAL32 carrier: narrow path, no restoration.
  {
    auto const [restored, narrow] = evaluate_with_constant(make_dec64_const(2400, 15, 2));
    REQUIRE(restored == 0);
    REQUIRE(narrow == 1);
  }
  // Different scale: numerically comparable only after restoration — narrow path must decline.
  {
    auto const [restored, narrow] = evaluate_with_constant(make_dec64_const(24000, 15, 3));
    REQUIRE(restored == 1);
    REQUIRE(narrow == 0);
  }
}

TEST_CASE("narrow-domain DECIMAL comparison matches restored results",
          "[expression_evaluator][compressed_materialization][narrow_domain]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  auto const decimal_type = logical_type::make_decimal(15, 2);
  std::vector<int32_t> const raw{100, 2399, 2400, 2401, 90'000};

  auto predicate = make_cmp(
    sirius::comparison_type::lt, make_ref_typed(0, decimal_type), make_dec64_const(2400, 15, 2));
  exp_executor executor(*predicate, get_resource_ref(*space));

  auto input      = make_decimal32_batch(*space, /*scale=*/2, raw);
  auto input_ro   = input->to_read_only();
  auto input_view = input_ro.get_data()->cast<gpu_table_representation>().get_table_view();
  auto output     = executor.evaluate(input_view);
  REQUIRE(output != nullptr);
  REQUIRE(executor.narrow_domain_comparison_count_for_testing() == 1);

  auto const output_values = copy_bool_column_to_host(output->view().column(0));
  REQUIRE(output_values.size() == raw.size());
  for (std::size_t i = 0; i < raw.size(); ++i) {
    REQUIRE(output_values[i] == ((raw[i] < 2400) ? 1U : 0U));
  }
}

TEST_CASE("narrow-domain BETWEEN evaluates on the narrowed carrier",
          "[expression_evaluator][compressed_materialization][narrow_domain]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  auto const bigint_type = logical_type::make(type_id::BIGINT);
  auto predicate = std::make_unique<ast_node>(sirius::ast::between{make_ref_typed(0, bigint_type),
                                                                   make_bigint_const(5),
                                                                   make_bigint_const(20),
                                                                   /*lower_inclusive=*/true,
                                                                   /*upper_inclusive=*/false});

  exp_executor executor(*predicate, get_resource_ref(*space));

  auto input      = make_input_batch(*space,
                                     {cudf::data_type{cudf::type_id::INT32}},
                                     {std::optional<std::pair<int, int>>{{-10, 30}}});
  auto input_ro   = input->to_read_only();
  auto input_view = input_ro.get_data()->cast<gpu_table_representation>().get_table_view();

  auto output = executor.evaluate(input_view);
  REQUIRE(output != nullptr);
  REQUIRE(executor.restored_reference_cast_count_for_testing() == 0);
  REQUIRE(executor.narrow_domain_comparison_count_for_testing() == 1);

  auto const input_values  = copy_column_to_host<int32_t>(input_view.column(0));
  auto const output_values = copy_bool_column_to_host(output->view().column(0));
  for (std::size_t i = 0; i < input_values.size(); ++i) {
    auto const value = static_cast<int64_t>(input_values[i]);
    REQUIRE(output_values[i] == ((value >= 5 && value < 20) ? 1U : 0U));
  }
}

TEST_CASE("narrow-domain reference pairs evaluate on their shared carrier",
          "[expression_evaluator][compressed_materialization][narrow_domain]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  auto const bigint_type = logical_type::make(type_id::BIGINT);
  auto const date_type   = logical_type::make(type_id::DATE);
  auto const int8        = cudf::data_type{cudf::type_id::INT8};
  auto const int16       = cudf::data_type{cudf::type_id::INT16};

  // Epoch days spanning 1969-12-02 to 1998-12-31, so the same two columns serve the DATE sections.
  std::vector<int16_t> const left_values{-30, 0, 209, 8035, 10'591, 10'591};
  std::vector<int16_t> const right_values{-30, 5, 208, 10'591, 8035, 10'591};

  struct evaluation {
    std::size_t restored_casts;
    std::size_t narrow_comparisons;
    std::vector<uint8_t> values;
    std::vector<bool> valids;
  };

  auto evaluate_predicate = [&](std::vector<std::unique_ptr<cudf::column>> columns,
                                std::unique_ptr<ast_node> predicate) {
    exp_executor executor(*predicate, get_resource_ref(*space));
    auto input      = make_batch_from_columns(*space, std::move(columns));
    auto input_ro   = input->to_read_only();
    auto input_view = input_ro.get_data()->cast<gpu_table_representation>().get_table_view();
    auto output     = executor.evaluate(input_view);
    REQUIRE(output != nullptr);
    REQUIRE(output->num_columns() == 1);
    return evaluation{executor.restored_reference_cast_count_for_testing(),
                      executor.narrow_domain_comparison_count_for_testing(),
                      copy_bool_column_to_host(output->view().column(0)),
                      copy_valids_to_host(output->view().column(0))};
  };

  // Two INT16 carriers holding `left_values` and `right_values`.
  auto narrow_pair_columns = [&] {
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(make_carrier_column(*space, int16, left_values));
    columns.push_back(make_carrier_column(*space, int16, right_values));
    return columns;
  };

  auto require_less_than = [&](evaluation const& result) {
    REQUIRE(result.values.size() == left_values.size());
    for (std::size_t i = 0; i < left_values.size(); ++i) {
      REQUIRE(result.valids[i]);
      REQUIRE(result.values[i] == ((left_values[i] < right_values[i]) ? 1U : 0U));
    }
  };

  SECTION("two BIGINT references on one carrier compare narrow, neither side restored")
  {
    auto const result = evaluate_predicate(narrow_pair_columns(),
                                           make_cmp(sirius::comparison_type::lt,
                                                    make_ref_typed(0, bigint_type),
                                                    make_ref_typed(1, bigint_type)));
    REQUIRE(result.narrow_comparisons == 1);
    REQUIRE(result.restored_casts == 0);
    require_less_than(result);
  }

  SECTION("two BIGINT references on different carriers restore both sides")
  {
    // The positive case's twin: only the left carrier changes, and the pair stops engaging because
    // the evaluator will not widen the narrower side to meet the other.
    std::vector<int8_t> const left_int8{-30, 0, 100, 120, -128, 127};
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(make_carrier_column(*space, int8, left_int8));
    columns.push_back(make_carrier_column(*space, int16, right_values));

    auto const result = evaluate_predicate(std::move(columns),
                                           make_cmp(sirius::comparison_type::lt,
                                                    make_ref_typed(0, bigint_type),
                                                    make_ref_typed(1, bigint_type)));
    REQUIRE(result.narrow_comparisons == 0);
    REQUIRE(result.restored_casts == 2);
    for (std::size_t i = 0; i < left_int8.size(); ++i) {
      REQUIRE(result.values[i] == ((left_int8[i] < right_values[i]) ? 1U : 0U));
    }
  }

  SECTION("two DATE references on one carrier compare narrow, neither side restored")
  {
    auto const result = evaluate_predicate(
      narrow_pair_columns(),
      make_cmp(
        sirius::comparison_type::lt, make_ref_typed(0, date_type), make_ref_typed(1, date_type)));
    REQUIRE(result.narrow_comparisons == 1);
    REQUIRE(result.restored_casts == 0);
    require_less_than(result);
  }

  SECTION("a DATE reference and a DATE literal compare narrow")
  {
    constexpr int32_t last_day_1998 = 10'591;
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(make_carrier_column(*space, int16, left_values));

    auto const result = evaluate_predicate(
      std::move(columns),
      make_cmp(
        sirius::comparison_type::lt, make_ref_typed(0, date_type), make_date_const(last_day_1998)));
    REQUIRE(result.narrow_comparisons == 1);
    REQUIRE(result.restored_casts == 0);
    for (std::size_t i = 0; i < left_values.size(); ++i) {
      REQUIRE(result.valids[i]);
      REQUIRE(result.values[i] == ((left_values[i] < last_day_1998) ? 1U : 0U));
    }
  }

  SECTION("a DATE reference against a typed NULL yields NULL at the narrow width")
  {
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(make_carrier_column(*space, int16, left_values));

    auto const result = evaluate_predicate(
      std::move(columns),
      make_cmp(
        sirius::comparison_type::lt, make_ref_typed(0, date_type), make_null_const(date_type)));
    REQUIRE(result.narrow_comparisons == 1);
    REQUIRE(result.restored_casts == 0);
    REQUIRE(result.valids.size() == left_values.size());
    for (auto const valid : result.valids) {
      REQUIRE_FALSE(valid);
    }
  }
}

// ---------------------------------------------------------------------------
// Narrow-domain eligibility predicates (sirius::ast, host-only)
//
// The evaluator reaches these through narrow_domain_carrier and
// narrow_domain_reference_pair_carrier; the narrowing policy reaches the same predicates with the
// planned carrier instead of the batch's. Testing them directly pins the shapes that are awkward
// to build as batches -- mistyped payloads, decimal scale mismatches, unsigned families.
// ---------------------------------------------------------------------------

TEST_CASE("constant_range - a DATE literal folds to its epoch-day signed range",
          "[expression_evaluator][compressed_materialization][narrow_domain]")
{
  auto const last_day_1998 = sirius::ast::constant_numeric_range(date_literal(10'591));
  REQUIRE(last_day_1998.has_value());
  REQUIRE(last_day_1998->domain == sirius::numeric_range_domain::SIGNED_INTEGER);
  REQUIRE(static_cast<int64_t>(last_day_1998->minimum) == 10'591);
  REQUIRE(static_cast<int64_t>(last_day_1998->maximum) == 10'591);

  auto const pre_epoch = sirius::ast::constant_numeric_range(date_literal(-1));
  REQUIRE(pre_epoch.has_value());
  REQUIRE(static_cast<int64_t>(pre_epoch->minimum) == -1);
  REQUIRE(static_cast<int64_t>(pre_epoch->maximum) == -1);

  // A payload alternative disagreeing with the declared type folds to nothing rather than
  // reinterpreting whichever alternative happens to be present.
  auto const mistyped =
    sirius::ast::constant{sirius::value{int32_t{10'591}}, logical_type::make(type_id::DATE)};
  REQUIRE_FALSE(sirius::ast::constant_numeric_range(mistyped).has_value());

  // DATE is the only temporal type admitted; a sub-day literal has no narrow domain to fold into.
  auto const timestamp = sirius::ast::constant{sirius::value{sirius::timestamp_us_value{1}},
                                               logical_type::make(type_id::TIMESTAMP)};
  REQUIRE_FALSE(sirius::ast::constant_numeric_range(timestamp).has_value());

  // A typed NULL carries no value to fold; constant_representable_in_carrier admits it instead.
  auto const null_date = sirius::ast::constant{sirius::value{}, logical_type::make(type_id::DATE)};
  REQUIRE_FALSE(sirius::ast::constant_numeric_range(null_date).has_value());
}

TEST_CASE("constant_range - a DATE literal is representable only inside the carrier",
          "[expression_evaluator][compressed_materialization][narrow_domain]")
{
  auto const int8_carrier   = cudf::data_type{cudf::type_id::INT8};
  auto const int16_carrier  = cudf::data_type{cudf::type_id::INT16};
  auto const uint16_carrier = cudf::data_type{cudf::type_id::UINT16};

  REQUIRE(sirius::ast::constant_representable_in_carrier(int16_carrier, date_literal(10'591)));
  REQUIRE(sirius::ast::constant_representable_in_carrier(int8_carrier, date_literal(100)));

  // Out of the carrier's range the literal is rejected, never truncated into it.
  REQUIRE_FALSE(sirius::ast::constant_representable_in_carrier(int8_carrier, date_literal(10'591)));
  REQUIRE_FALSE(
    sirius::ast::constant_representable_in_carrier(int16_carrier, date_literal(40'000)));

  // Epoch days are signed, so an unsigned carrier is a different family even for a positive day.
  REQUIRE_FALSE(sirius::ast::constant_representable_in_carrier(uint16_carrier, date_literal(100)));

  // Only a validity bit is materialized, so a typed NULL fits even the narrowest carrier.
  auto const null_date = sirius::ast::constant{sirius::value{}, logical_type::make(type_id::DATE)};
  REQUIRE(sirius::ast::constant_representable_in_carrier(int8_carrier, null_date));
}

TEST_CASE("narrow_domain_carrier_eligible - epoch days never pair with a plain integer literal",
          "[expression_evaluator][compressed_materialization][narrow_domain]")
{
  using sirius::ast::narrow_domain_carrier_eligible;

  auto const date_type     = logical_type::make(type_id::DATE);
  auto const integer_type  = logical_type::make(type_id::INTEGER);
  auto const int16_carrier = cudf::data_type{cudf::type_id::INT16};
  auto const date_carrier  = cudf::data_type{cudf::type_id::TIMESTAMP_DAYS};

  auto const date_const    = make_date_const(10'000);
  auto const integer_const = make_int_const(10'000);

  SECTION("a literal of the column's own domain engages")
  {
    REQUIRE(narrow_domain_carrier_eligible(date_type, int16_carrier, {date_const.get()}));
    REQUIRE(narrow_domain_carrier_eligible(integer_type, int16_carrier, {integer_const.get()}));
  }

  SECTION("a literal of the other domain declines in both directions")
  {
    // Both operands land in the same INT16 carrier, where epoch days and plain integers are
    // indistinguishable, so agreement at the narrow width would be a coincidence of the values.
    REQUIRE_FALSE(narrow_domain_carrier_eligible(date_type, int16_carrier, {integer_const.get()}));
    REQUIRE_FALSE(narrow_domain_carrier_eligible(integer_type, int16_carrier, {date_const.get()}));
  }

  SECTION("a typed NULL is admitted whatever domain it declares")
  {
    // Only the validity bit is materialized, so no carrier width can misread the literal and its
    // declared type does not matter. Declining here would cost a DATE column the narrow path an
    // integer column keeps for the very same predicate.
    auto const null_date    = make_null_const(date_type);
    auto const null_integer = make_null_const(integer_type);
    REQUIRE(narrow_domain_carrier_eligible(date_type, int16_carrier, {null_date.get()}));
    REQUIRE(narrow_domain_carrier_eligible(date_type, int16_carrier, {null_integer.get()}));
    REQUIRE(narrow_domain_carrier_eligible(integer_type, int16_carrier, {null_date.get()}));
  }

  SECTION("a DATE literal outside the carrier declines")
  {
    auto const far_date = make_date_const(40'000);
    REQUIRE_FALSE(narrow_domain_carrier_eligible(date_type, int16_carrier, {far_date.get()}));
  }

  SECTION("a natively carried DATE column declines")
  {
    REQUIRE_FALSE(narrow_domain_carrier_eligible(date_type, date_carrier, {date_const.get()}));
  }

  SECTION("a missing or non-constant operand declines")
  {
    auto const reference = make_ref_typed(0, date_type);
    REQUIRE_FALSE(narrow_domain_carrier_eligible(date_type, int16_carrier, {nullptr}));
    REQUIRE_FALSE(narrow_domain_carrier_eligible(date_type, int16_carrier, {reference.get()}));
  }
}

TEST_CASE("narrow_domain_reference_pair_eligible - one shared carrier and one shared domain",
          "[expression_evaluator][compressed_materialization][narrow_domain]")
{
  using sirius::ast::narrow_domain_reference_pair_eligible;

  auto const date_type      = logical_type::make(type_id::DATE);
  auto const smallint_type  = logical_type::make(type_id::SMALLINT);
  auto const integer_type   = logical_type::make(type_id::INTEGER);
  auto const bigint_type    = logical_type::make(type_id::BIGINT);
  auto const uinteger_type  = logical_type::make(type_id::UINTEGER);
  auto const double_type    = logical_type::make(type_id::DOUBLE);
  auto const timestamp_type = logical_type::make(type_id::TIMESTAMP);
  auto const decimal_2      = logical_type::make_decimal(18, 2);
  auto const decimal_3      = logical_type::make_decimal(18, 3);

  auto const int8_carrier   = cudf::data_type{cudf::type_id::INT8};
  auto const int16_carrier  = cudf::data_type{cudf::type_id::INT16};
  auto const int32_carrier  = cudf::data_type{cudf::type_id::INT32};
  auto const uint16_carrier = cudf::data_type{cudf::type_id::UINT16};
  auto const dec32_scale_2  = cudf::data_type{cudf::type_id::DECIMAL32, -2};
  auto const dec32_scale_3  = cudf::data_type{cudf::type_id::DECIMAL32, -3};

  SECTION("both sides narrowed onto one carrier engage")
  {
    REQUIRE(narrow_domain_reference_pair_eligible(
      integer_type, int16_carrier, integer_type, int16_carrier));
    REQUIRE(
      narrow_domain_reference_pair_eligible(date_type, int16_carrier, date_type, int16_carrier));
    REQUIRE(
      narrow_domain_reference_pair_eligible(decimal_2, dec32_scale_2, decimal_2, dec32_scale_2));
    // Unequal logical widths still compare exactly: narrowing preserves values and applies no
    // offset, so the shared carrier is the whole contract.
    REQUIRE(narrow_domain_reference_pair_eligible(
      bigint_type, int16_carrier, integer_type, int16_carrier));
  }

  SECTION("different carriers decline even within one domain")
  {
    REQUIRE_FALSE(narrow_domain_reference_pair_eligible(
      integer_type, int8_carrier, integer_type, int16_carrier));
    // Scale belongs to the carrier, so a scale mismatch never reaches a width comparison.
    REQUIRE_FALSE(
      narrow_domain_reference_pair_eligible(decimal_2, dec32_scale_2, decimal_3, dec32_scale_3));
  }

  SECTION("a side that is not carried narrow declines")
  {
    // SMALLINT natively materializes as INT16, so an INT16 column of it is native, not narrowed.
    REQUIRE_FALSE(narrow_domain_reference_pair_eligible(
      smallint_type, int16_carrier, integer_type, int16_carrier));
    REQUIRE_FALSE(narrow_domain_reference_pair_eligible(
      integer_type, int32_carrier, integer_type, int32_carrier));
    REQUIRE_FALSE(narrow_domain_reference_pair_eligible(
      double_type, int16_carrier, integer_type, int16_carrier));
    REQUIRE_FALSE(narrow_domain_reference_pair_eligible(
      timestamp_type, int16_carrier, integer_type, int16_carrier));
  }

  SECTION("epoch days and plain integers decline at one carrier, in both orders")
  {
    REQUIRE_FALSE(
      narrow_domain_reference_pair_eligible(date_type, int16_carrier, integer_type, int16_carrier));
    REQUIRE_FALSE(
      narrow_domain_reference_pair_eligible(integer_type, int16_carrier, date_type, int16_carrier));
  }

  SECTION("crossing the signedness family declines")
  {
    REQUIRE_FALSE(narrow_domain_reference_pair_eligible(
      uinteger_type, uint16_carrier, integer_type, int16_carrier));
    // Even at one carrier: INT16 is not a narrowing of UINT32, so the unsigned side is not carried.
    REQUIRE_FALSE(narrow_domain_reference_pair_eligible(
      uinteger_type, int16_carrier, integer_type, int16_carrier));
  }
}

// ---------------------------------------------------------------------------
// evaluate() — reference, constant, comparison (basic smoke test per type)
// ---------------------------------------------------------------------------

TEMPLATE_TEST_CASE("evaluate projects references, constants, and comparisons",
                   "[expression_evaluator]",
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

TEMPLATE_TEST_CASE("evaluate casts scalar constants including HUGEINT and NULL",
                   "[expression_evaluator][scalar_cast]",
                   mat_strategy,
                   ast_interpret_strategy,
                   ast_jit_strategy)
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  auto const hugeint = logical_type::make(type_id::HUGEINT);
  auto const bigint  = logical_type::make(type_id::BIGINT);
  auto const integer = logical_type::make(type_id::INTEGER);

  for (auto const row_count : {0, 1, 7}) {
    CAPTURE(row_count);
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                                row_count,
                                                cudf::mask_state::UNALLOCATED,
                                                cudf::get_default_stream(),
                                                get_resource_ref(*space)));
    auto input = std::make_unique<cudf::table>(std::move(columns));

    std::vector<std::unique_ptr<ast_node>> nodes;
    nodes.push_back(make_cast(make_int_const(-42), hugeint, false));
    // The outer AST cast consumes the materialized HUGEINT cast as a temporary column.
    nodes.push_back(make_cast(make_cast(make_int_const(1), hugeint, false), bigint, false));
    nodes.push_back(make_cast(make_null_const(integer), hugeint, false));
    nodes.push_back(make_cast(make_cast(make_null_const(integer), hugeint, false), bigint, false));
    nodes.push_back(make_cast(make_int_const(42), logical_type::make(type_id::DOUBLE), false));
    std::vector<ast_node const*> expressions;
    for (auto const& node : nodes) {
      expressions.push_back(node.get());
    }
    exp_executor executor(expressions,
                          get_resource_ref(*space),
                          cudf::get_default_stream(),
                          sirius::expression_query_policy{TestType::value},
                          1);
    auto output     = executor.evaluate(input->view());
    auto const view = output->view();
    REQUIRE(view.num_rows() == row_count);
    REQUIRE(view.num_columns() == 5);
    for (int i = 0; i < 4; ++i) {
      REQUIRE(view.column(i).type() == cudf::data_type{cudf::type_id::INT64});
    }
    REQUIRE(copy_column_to_host<int64_t>(view.column(0)) == std::vector<int64_t>(row_count, -42));
    REQUIRE(copy_column_to_host<int64_t>(view.column(1)) == std::vector<int64_t>(row_count, 1));
    REQUIRE(view.column(0).null_count() == 0);
    REQUIRE(view.column(1).null_count() == 0);
    REQUIRE(view.column(2).null_count() == row_count);
    REQUIRE(view.column(3).null_count() == row_count);
    REQUIRE(view.column(4).type() == cudf::data_type{cudf::type_id::FLOAT64});
    REQUIRE(copy_column_to_host<double>(view.column(4)) == std::vector<double>(row_count, 42.0));
  }
}

// ---------------------------------------------------------------------------
// select() — basic filter + edge cases
// ---------------------------------------------------------------------------

TEMPLATE_TEST_CASE("select filters rows and handles edge cases",
                   "[expression_evaluator]",
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

TEMPLATE_TEST_CASE("evaluate arithmetic functions",
                   "[expression_evaluator]",
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

TEMPLATE_TEST_CASE("evaluate decimal arithmetic (DECIMAL64)",
                   "[expression_evaluator][decimal]",
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

TEMPLATE_TEST_CASE("evaluate decimal arithmetic (DECIMAL32)",
                   "[expression_evaluator][decimal]",
                   mat_strategy,
                   ast_interpret_strategy,
                   ast_jit_strategy)
{
  constexpr auto strategy = TestType::value;
  auto* space             = get_default_gpu_space();
  REQUIRE(space != nullptr);

  // Width 8 → INT32 physical (DuckDB) → DECIMAL32 (cudf). Exercises the
  // DECIMAL32 branches in constant.cpp and get_cudf_type.
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

TEMPLATE_TEST_CASE("evaluate nested decimal arithmetic (col + 1.00) * 2",
                   "[expression_evaluator][decimal]",
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

TEMPLATE_TEST_CASE("evaluate decimal arithmetic (DECIMAL128)",
                   "[expression_evaluator][decimal]",
                   mat_strategy,
                   ast_interpret_strategy,
                   ast_jit_strategy)
{
  // Width 38 → INT128 physical (DuckDB) → DECIMAL128 (cudf). Exercises the
  // hugeint_t → __int128_t conversion branch in constant.cpp for
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

TEMPLATE_TEST_CASE("evaluate decimal DIV (DECIMAL64)",
                   "[expression_evaluator][decimal]",
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

TEMPLATE_TEST_CASE("evaluate decimal TPC-H Q1 shape price * (1 - discount)",
                   "[expression_evaluator][decimal]",
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
                   "[expression_evaluator]",
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

  SECTION("LIKE '%tr%1%' — multi-literal SWAR fast path")
  {
    // col0 LIKE '%tr%1%' — a `%lit1%lit2%` pattern, dispatched to the SWAR fast path.
    std::vector<std::unique_ptr<ast_node>> children;
    children.push_back(make_ref(0));
    children.push_back(make_str_const("%tr%1%"));
    auto expr = make_func(
      sirius::function_id::like, std::move(children), logical_type::make(type_id::BOOLEAN));

    auto [in_batch, out_batch, iv, ov] = run_select(*space, input, std::move(expr), strategy);
    auto in_strs                       = copy_string_column_to_host(iv.column(0));
    std::vector<std::string> expected;
    for (auto const& s : in_strs) {
      auto const tr = s.find("tr");
      if (tr != std::string::npos && s.find('1', tr + 2) != std::string::npos) {
        expected.push_back(s);
      }
    }
    REQUIRE(copy_string_column_to_host(ov.column(0)) == expected);
  }

  SECTION("LIKE fast-path classification is shared by pattern value across evaluators")
  {
    auto make_like_expression = [] {
      std::vector<std::unique_ptr<ast_node>> children;
      children.push_back(make_ref(0));
      children.push_back(make_str_const("%tr%1%"));
      return make_func(
        sirius::function_id::like, std::move(children), logical_type::make(type_id::BOOLEAN));
    };
    auto first_expr  = make_like_expression();
    auto second_expr = make_like_expression();
    REQUIRE(first_expr.get() != second_expr.get());

    auto input_ro   = input->to_read_only();
    auto& repr      = input_ro.get_data()->cast<gpu_table_representation>();
    auto like_cache = std::make_shared<sirius::like_multiliteral_cache>();
    exp_executor first_executor(*first_expr,
                                get_resource_ref(*space),
                                cudf::get_default_stream(),
                                sirius::expression_query_policy{strategy},
                                exp_executor::default_min_ast_size,
                                /*like_swar_fastpath=*/true,
                                like_cache);
    exp_executor second_executor(*second_expr,
                                 get_resource_ref(*space),
                                 cudf::get_default_stream(),
                                 sirius::expression_query_policy{strategy},
                                 exp_executor::default_min_ast_size,
                                 /*like_swar_fastpath=*/true,
                                 like_cache);

    auto first       = first_executor.select(repr.get_table_view());
    auto first_again = first_executor.select(repr.get_table_view());
    auto second      = second_executor.select(repr.get_table_view());
    REQUIRE(copy_string_column_to_host(first->view().column(0)) ==
            copy_string_column_to_host(first_again->view().column(0)));
    REQUIRE(copy_string_column_to_host(first->view().column(0)) ==
            copy_string_column_to_host(second->view().column(0)));
    REQUIRE(first_executor.like_shared_cache_lookup_count_for_testing() == 1);
    REQUIRE(second_executor.like_shared_cache_lookup_count_for_testing() == 1);
    REQUIRE(like_cache->classification_count_for_testing() == 1);
  }

  SECTION("NOT LIKE fallback classification is memoized across evaluator batches")
  {
    std::vector<std::unique_ptr<ast_node>> children;
    children.push_back(make_ref(0));
    children.push_back(make_str_const("%tr_1%"));
    auto expr = make_func(
      sirius::function_id::not_like, std::move(children), logical_type::make(type_id::BOOLEAN));

    auto input_ro   = input->to_read_only();
    auto& repr      = input_ro.get_data()->cast<gpu_table_representation>();
    auto like_cache = std::make_shared<sirius::like_multiliteral_cache>();
    exp_executor executor(*expr,
                          get_resource_ref(*space),
                          cudf::get_default_stream(),
                          sirius::expression_query_policy{strategy},
                          exp_executor::default_min_ast_size,
                          /*like_swar_fastpath=*/true,
                          like_cache);

    auto first               = executor.select(repr.get_table_view());
    auto second              = executor.select(repr.get_table_view());
    auto const input_strings = copy_string_column_to_host(repr.get_table_view().column(0));
    std::vector<std::string> expected;
    for (auto const& value : input_strings) {
      if (value != "str_1") { expected.push_back(value); }
    }
    REQUIRE(copy_string_column_to_host(first->view().column(0)) == expected);
    REQUIRE(copy_string_column_to_host(second->view().column(0)) == expected);
    REQUIRE(executor.like_shared_cache_lookup_count_for_testing() == 1);
    REQUIRE(like_cache->classification_count_for_testing() == 1);
  }

  SECTION("NOT LIKE '%tr%1%' — fast path agrees with the cudf path (fast path disabled)")
  {
    auto run_not_like = [&](bool like_swar_fastpath) {
      std::vector<std::unique_ptr<ast_node>> children;
      children.push_back(make_ref(0));
      children.push_back(make_str_const("%tr%1%"));
      auto expr = make_func(
        sirius::function_id::not_like, std::move(children), logical_type::make(type_id::BOOLEAN));
      auto [in_batch, out_batch, iv, ov] =
        run_select(*space, input, std::move(expr), strategy, like_swar_fastpath);
      return copy_string_column_to_host(ov.column(0));
    };

    auto const with_fast_path = run_not_like(true);
    auto const with_cudf      = run_not_like(false);
    REQUIRE(with_fast_path == with_cudf);
  }
}

// ---------------------------------------------------------------------------
// CASE/WHEN (always materializes — AST breaker)
// ---------------------------------------------------------------------------

TEMPLATE_TEST_CASE("evaluate CASE expression",
                   "[expression_evaluator]",
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

TEMPLATE_TEST_CASE("evaluate CASE with multiple WHEN branches",
                   "[expression_evaluator]",
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

TEMPLATE_TEST_CASE("select BETWEEN",
                   "[expression_evaluator]",
                   mat_strategy,
                   ast_interpret_strategy,
                   ast_jit_strategy)
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
                   "[expression_evaluator]",
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
                   "[expression_evaluator]",
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

TEMPLATE_TEST_CASE("evaluate COALESCE",
                   "[expression_evaluator]",
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
                   "[expression_evaluator]",
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
                   "[expression_evaluator]",
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
                   "[expression_evaluator]",
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
                   "[expression_evaluator]",
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
                   "[expression_evaluator]",
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
                   "[expression_evaluator]",
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
                   "[expression_evaluator]",
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

TEMPLATE_TEST_CASE("evaluate conjunction with scalar children",
                   "[expression_evaluator]",
                   mat_strategy,
                   ast_interpret_strategy,
                   ast_jit_strategy)
{
  constexpr auto strategy = TestType::value;
  auto* space             = get_default_gpu_space();
  REQUIRE(space != nullptr);

  auto input =
    make_input_batch(*space, {cudf::data_type{cudf::type_id::INT32}}, {std::pair<int, int>{0, 20}});

  auto const boolean = logical_type::make(type_id::BOOLEAN);
  auto bool_const    = [&](bool v) {
    return std::make_unique<ast_node>(sirius::ast::constant{sirius::value{v}, boolean});
  };
  auto null_const = [&] { return make_null_const(boolean); };
  auto cmp  = [] { return make_cmp(sirius::comparison_type::gt, make_ref(0), make_int_const(17)); };
  auto conj = [](sirius::ast::conjunction::kind kind, auto... children) {
    std::vector<std::unique_ptr<ast_node>> v;
    (v.push_back(std::move(children)), ...);
    return make_conj(kind, std::move(v));
  };
  auto constexpr op_or  = sirius::ast::conjunction::kind::op_or;
  auto constexpr op_and = sirius::ast::conjunction::kind::op_and;

  std::vector<std::unique_ptr<ast_node>> exprs;
  exprs.push_back(conj(op_or, cmp(), null_const()));                     // col0 > 17 OR NULL
  exprs.push_back(conj(op_or, null_const(), cmp()));                     // NULL OR col0 > 17
  exprs.push_back(conj(op_and, cmp(), null_const()));                    // col0 > 17 AND NULL
  exprs.push_back(conj(op_or, cmp(), bool_const(true)));                 // col0 > 17 OR TRUE
  exprs.push_back(conj(op_and, bool_const(false), cmp()));               // FALSE AND col0 > 17
  exprs.push_back(conj(op_or, null_const(), bool_const(false), cmp()));  // NULL OR FALSE OR ...

  auto [in_batch, out_batch, iv, ov] = run_execute(*space, input, std::move(exprs), strategy);
  REQUIRE(ov.num_columns() == 6);
  auto const in0 = copy_column_to_host<int32_t>(iv.column(0));

  // Kleene logic: TRUE OR NULL = TRUE, FALSE AND NULL = FALSE, otherwise NULL.
  std::vector<bool> or_null_valid, and_null_valid;
  std::vector<uint8_t> gt;
  for (auto v : in0) {
    gt.push_back(v > 17 ? 1U : 0U);
    or_null_valid.push_back(v > 17);
    and_null_valid.push_back(v <= 17);
  }
  auto values_where = [](std::vector<uint8_t> values, std::vector<bool> const& valid) {
    for (std::size_t i = 0; i < values.size(); ++i) {
      if (!valid[i]) { values[i] = 0; }
    }
    return values;
  };
  auto check = [&](int col, std::vector<uint8_t> const& expected, std::vector<bool> const& valid) {
    CAPTURE(col);
    REQUIRE(ov.column(col).type() == cudf::data_type{cudf::type_id::BOOL8});
    auto const valids = copy_valids_to_host(ov.column(col));
    REQUIRE(valids == valid);
    REQUIRE(values_where(copy_bool_column_to_host(ov.column(col)), valid) ==
            values_where(expected, valid));
  };

  auto const all_valid = std::vector<bool>(in0.size(), true);
  check(0, gt, or_null_valid);
  check(1, gt, or_null_valid);
  check(2, gt, and_null_valid);
  check(3, std::vector<uint8_t>(in0.size(), 1U), all_valid);
  check(4, std::vector<uint8_t>(in0.size(), 0U), all_valid);
  check(5, gt, or_null_valid);
}

// ---------------------------------------------------------------------------
// Nested compound: CASE inside a comparison inside a conjunction
// Exercises AST breaker (CASE) nested within AST-capable nodes
// ---------------------------------------------------------------------------

TEMPLATE_TEST_CASE("select with nested CASE in predicate",
                   "[expression_evaluator]",
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
                   "[expression_evaluator]",
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
// Arithmetic in projection combined with CASE — complex evaluate() output
// ---------------------------------------------------------------------------

TEMPLATE_TEST_CASE("evaluate mixed arithmetic and CASE projection",
                   "[expression_evaluator]",
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
  auto ro       = batch->to_read_only();
  auto& in_repr = ro.get_data()->cast<gpu_table_representation>();
  return {batch, in_repr.get_table_view()};
}

std::unique_ptr<sirius::ast::node> ref_node_native(uint32_t idx) { return make_ref(idx); }

std::unique_ptr<sirius::ast::node> int_const_node_native(int32_t v) { return make_int_const(v); }

std::unique_ptr<cudf::table> run_native_ast(memory_space& space,
                                            sirius::ast::node const* expr_ptr,
                                            cudf::table_view tv,
                                            exp_strategy_enum strategy = MAT)
{
  exp_executor executor(expr_ptr,
                        get_resource_ref(space),
                        cudf::get_default_stream(),
                        sirius::expression_query_policy{strategy});
  return executor.evaluate(tv);
}

}  // namespace

TEST_CASE("native_ast - reference identity", "[expression_evaluator_ast_native][reference]")
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

TEST_CASE("native_ast - constant INTEGER", "[expression_evaluator_ast_native][constant]")
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

TEST_CASE("native_ast - constant VARCHAR", "[expression_evaluator_ast_native][constant]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  auto input =
    make_input_batch(*space, {cudf::data_type{cudf::type_id::STRING}}, {std::pair<int, int>{1, 5}});
  auto ro       = input->to_read_only();
  auto& in_repr = ro.get_data()->cast<gpu_table_representation>();
  auto tv       = in_repr.get_table_view();

  auto hand_ast = std::make_unique<sirius::ast::node>(sirius::ast::constant{
    sirius::value{std::string{"hello"}}, sirius::logical_type::make(sirius::type_id::VARCHAR)});

  auto out = run_native_ast(*space, hand_ast.get(), tv, MAT);
  REQUIRE(out);
  std::vector<std::string> expected(tv.num_rows(), "hello");
  REQUIRE(copy_string_column_to_host(out->view().column(0)) == expected);
}

TEST_CASE("native_ast - comparison EQUAL (MATERIALIZE)",
          "[expression_evaluator_ast_native][comparison]")
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
          "[expression_evaluator_ast_native][comparison]")
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
          "[expression_evaluator_ast_native][conjunction]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  // Build a 2-column INT32 batch with col0 in [0, 100) and col1 in [0, 100).
  auto input =
    make_input_batch(*space,
                     {cudf::data_type{cudf::type_id::INT32}, cudf::data_type{cudf::type_id::INT32}},
                     {std::pair<int, int>{0, 100}, std::pair<int, int>{0, 100}});
  auto ro       = input->to_read_only();
  auto& in_repr = ro.get_data()->cast<gpu_table_representation>();
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

TEST_CASE("native_ast - between (MATERIALIZE)", "[expression_evaluator_ast_native][between]")
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

TEST_CASE("native_ast - unary_op NOT (MATERIALIZE)", "[expression_evaluator_ast_native][unary_op]")
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
          "[expression_evaluator_ast_native][unary_op]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  std::vector<int32_t> values{1, 2, 3, 4, 5};
  std::vector<bool> valids{true, false, true, false, true};
  auto batch    = make_int32_batch_with_nulls(*space, values, valids);
  auto ro       = batch->to_read_only();
  auto& in_repr = ro.get_data()->cast<gpu_table_representation>();
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
          "[expression_evaluator_ast_native][unary_op]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  std::vector<int32_t> values{10, 20, 30, 40};
  std::vector<bool> valids{true, false, true, true};
  auto batch    = make_int32_batch_with_nulls(*space, values, valids);
  auto ro       = batch->to_read_only();
  auto& in_repr = ro.get_data()->cast<gpu_table_representation>();
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

TEST_CASE("native_ast - in_list IN (MATERIALIZE)", "[expression_evaluator_ast_native][in_list]")
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

TEST_CASE("native_ast - coalesce (MATERIALIZE)", "[expression_evaluator_ast_native][coalesce]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  std::vector<int32_t> values{7, 8, 9, 10, 11};
  std::vector<bool> valids{true, false, true, false, true};
  auto batch    = make_int32_batch_with_nulls(*space, values, valids);
  auto ro       = batch->to_read_only();
  auto& in_repr = ro.get_data()->cast<gpu_table_representation>();
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
          "[expression_evaluator_ast_native][cast]")
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
          "[expression_evaluator_ast_native][function_call]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  // 2-column INT32 batch.
  auto input =
    make_input_batch(*space,
                     {cudf::data_type{cudf::type_id::INT32}, cudf::data_type{cudf::type_id::INT32}},
                     {std::pair<int, int>{0, 50}, std::pair<int, int>{0, 50}});
  auto ro       = input->to_read_only();
  auto& in_repr = ro.get_data()->cast<gpu_table_representation>();
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
          "[expression_evaluator_ast_native][function_call]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  auto input =
    make_input_batch(*space,
                     {cudf::data_type{cudf::type_id::INT32}, cudf::data_type{cudf::type_id::INT32}},
                     {std::pair<int, int>{0, 50}, std::pair<int, int>{0, 50}});
  auto ro       = input->to_read_only();
  auto& in_repr = ro.get_data()->cast<gpu_table_representation>();
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
          "[expression_evaluator_ast_native][case_expr]")
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

TEMPLATE_TEST_CASE("CASE accepts scalar WHEN conditions",
                   "[expression_evaluator][case_expr]",
                   mat_strategy,
                   ast_interpret_strategy,
                   ast_jit_strategy)
{
  auto* space          = get_default_gpu_space();
  auto input           = make_int32_input(*space);
  auto const condition = GENERATE(-1, 0, 1);
  auto const nested    = GENERATE(false, true);
  auto const boolean   = logical_type::make(type_id::BOOLEAN);
  std::vector<sirius::ast::case_expr::when_then> cases;
  cases.push_back({std::make_unique<ast_node>(sirius::ast::constant{
                     condition < 0 ? sirius::value{} : sirius::value{condition == 1}, boolean}),
                   int_const_node_native(100)});
  auto expr = std::make_unique<ast_node>(sirius::ast::case_expr{
    std::move(cases), ref_node_native(0), logical_type::make(type_id::INTEGER)});
  if (nested) {
    expr = make_cmp(sirius::comparison_type::equal, std::move(expr), int_const_node_native(100));
  }
  auto output         = run_native_ast(*space, expr.get(), input.view, TestType::value);
  auto const expected = copy_column_to_host<int32_t>(input.view.column(0));
  REQUIRE(output->num_rows() == input.view.num_rows());
  REQUIRE(output->view().column(0).null_count() == 0);
  if (nested) {
    auto actual = copy_column_to_host<int8_t>(output->view().column(0));
    for (size_t i = 0; i < actual.size(); ++i) {
      CHECK(actual[i] == (condition == 1 || expected[i] == 100));
    }
  } else {
    auto actual = copy_column_to_host<int32_t>(output->view().column(0));
    for (size_t i = 0; i < actual.size(); ++i) {
      CHECK(actual[i] == (condition == 1 ? 100 : expected[i]));
    }
  }
}

// Native-AST executor coverage for the null-safe comparison kinds
// (COMPARE_DISTINCT_FROM / COMPARE_NOT_DISTINCT_FROM) built directly as
// sirius::ast::comparison nodes — the comparison-kind enum is preserved end to
// end through the executor without any DuckDB Bound*Expression round-trip
// (sirius-db/sirius#699). DuckDB->Sirius lowering of these kinds is covered by
// the dedicated [ast_from_duckdb] suite.
TEST_CASE("native_ast - comparison DISTINCT_FROM / NOT_DISTINCT_FROM executes",
          "[expression_evaluator_ast_native][comparison]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  // Input: {10, NULL, 30, NULL, 50}.
  std::vector<int32_t> values = {10, 99, 30, 99, 50};
  std::vector<bool> valids    = {true, false, true, false, true};

  {
    auto batch    = make_int32_batch_with_nulls(*space, values, valids);
    auto ro       = batch->to_read_only();
    auto& in_repr = ro.get_data()->cast<gpu_table_representation>();
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
    auto ro       = batch->to_read_only();
    auto& in_repr = ro.get_data()->cast<gpu_table_representation>();
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

TEMPLATE_TEST_CASE("evaluate string case conversion",
                   "[expression_evaluator][string_case]",
                   mat_strategy,
                   ast_interpret_strategy,
                   ast_jit_strategy)
{
  constexpr auto strategy = TestType::value;
  auto* space             = get_default_gpu_space();
  auto mr                 = get_resource_ref(*space);
  auto stream             = cudf::get_default_stream();
  auto string_type        = logical_type::make(type_id::VARCHAR);
  auto const id           = GENERATE(sirius::function_id::upper, sirius::function_id::lower);

  // Include ASCII, empty strings, NULL, Unicode, and mappings that expand into
  // multiple code points. The expectations deliberately follow cuDF semantics.
  std::vector<std::string> const values{"HeLLo 123!", "", "", "é", "ß", "İ"};
  std::vector<std::unique_ptr<cudf::column>> rows;
  std::vector<cudf::column_view> row_views;
  for (std::size_t i = 0; i < values.size(); ++i) {
    cudf::string_scalar scalar(values[i], i != 2, stream, mr);
    rows.push_back(cudf::make_column_from_scalar(scalar, 1, stream, mr));
    row_views.push_back(rows.back()->view());
  }
  auto input_column = cudf::concatenate(row_views, stream, mr);
  cudf::table_view input({input_column->view()});

  auto case_expr = [&](std::unique_ptr<ast_node> child) {
    std::vector<std::unique_ptr<ast_node>> args;
    args.push_back(std::move(child));
    return make_func(id, std::move(args), string_type);
  };
  auto evaluate = [&](ast_node const& expr, cudf::table_view table) {
    sirius::expression_evaluator evaluator(
      expr, mr, stream, sirius::expression_query_policy{strategy});
    return evaluator.evaluate(table);
  };

  SECTION("column inputs preserve NULLs and use cuDF Unicode mappings")
  {
    auto expr           = case_expr(make_ref_typed(0, string_type));
    auto result         = evaluate(*expr, input);
    auto const output   = result->view().column(0);
    auto const expected = id == sirius::function_id::upper
                            ? std::vector<std::string>{"HELLO 123!", "", "", "É", "SS", "İ"}
                            : std::vector<std::string>{"hello 123!", "", "", "é", "ß", "i\u0307"};
    REQUIRE(copy_string_column_to_host(output) == expected);
    REQUIRE(copy_valids_to_host(output) == std::vector<bool>{true, true, false, true, true, true});
  }
  SECTION("case conversion composes inside an AST predicate")
  {
    auto expr =
      make_cmp(sirius::comparison_type::equal,
               case_expr(make_ref_typed(0, string_type)),
               make_str_const(id == sirius::function_id::upper ? "HELLO 123!" : "hello 123!"));
    sirius::expression_evaluator evaluator(
      *expr, mr, stream, sirius::expression_query_policy{strategy}, /*min_ast_size=*/1);
    auto result = evaluator.select(input);
    REQUIRE(copy_string_column_to_host(result->view().column(0)) ==
            std::vector<std::string>{"HeLLo 123!"});
  }
  SECTION("nested case conversion")
  {
    auto expr   = case_expr(case_expr(make_ref_typed(0, string_type)));
    auto result = evaluate(*expr, input);
    REQUIRE(result->num_rows() == input.num_rows());
    REQUIRE(copy_valids_to_host(result->view().column(0)) == copy_valids_to_host(input.column(0)));
    REQUIRE(copy_string_column_to_host(result->view().column(0))[0] ==
            (id == sirius::function_id::upper ? "HELLO 123!" : "hello 123!"));
  }
  SECTION("empty input")
  {
    auto empty_column = cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
    cudf::table_view empty({empty_column->view()});
    auto expr   = case_expr(make_ref_typed(0, string_type));
    auto result = evaluate(*expr, empty);
    REQUIRE(result->num_rows() == 0);
    REQUIRE(result->view().column(0).type().id() == cudf::type_id::STRING);
  }
}
TEST_CASE("timestamp casts check microsecond overflow", "[expression_evaluator][timestamp_bounds]")
{
  auto* space         = get_default_gpu_space();
  auto const strategy = GENERATE(MAT, exp_strategy_enum::AST_INTERPRET, exp_strategy_enum::AST_JIT);
  auto const seconds  = GENERATE(true, false);
  auto const source   = seconds ? sirius::type_id::TIMESTAMP_SEC : sirius::type_id::TIMESTAMP_MS;
  auto const source_type = sirius::logical_type::make(source);
  int64_t const scale    = seconds ? 1000000 : 1000;
  int64_t const max      = std::numeric_limits<int64_t>::max();
  int64_t const min      = std::numeric_limits<int64_t>::min();
  auto expr              = sirius::ast::node{sirius::ast::cast{
    make_ref_typed(0, source_type), sirius::logical_type::make(sirius::type_id::TIMESTAMP)}};
  auto stream            = cudf::get_default_stream();
  auto mr                = get_resource_ref(*space);
  auto run               = [&](std::vector<int64_t> const& ticks, bool valid = true) {
    auto col =
      cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT64},
                                ticks.size(),
                                valid ? cudf::mask_state::UNALLOCATED : cudf::mask_state::ALL_NULL,
                                stream,
                                mr);
    if (!ticks.empty()) {
      REQUIRE(cudaMemcpy(col->mutable_view().data<int64_t>(),
                         ticks.data(),
                         ticks.size() * sizeof(int64_t),
                         cudaMemcpyHostToDevice) == cudaSuccess);
    }
    auto view = cudf::bit_cast(col->view(), sirius::require_cudf_type(source_type));
    return run_native_ast(*space, &expr, cudf::table_view{{view}}, strategy);
  };
  SECTION("inclusive finite bounds and infinity sentinels")
  {
    auto out                      = run({min / scale, max / scale, -max, max, -1, 0, 1});
    auto actual                   = copy_column_to_host<int64_t>(out->view().column(0));
    std::vector<int64_t> expected = {
      (min / scale) * scale, (max / scale) * scale, -max, max, -scale, 0, scale};
    CHECK(actual == expected);
  }
  SECTION("finite values outside either bound fail")
  {
    for (auto ticks : {min / scale - 1, max / scale + 1, min, -max + 1, max - 1}) {
      CAPTURE(ticks);
      REQUIRE_THROWS_AS(run({0, ticks, max}), sirius::out_of_range_exception);
    }
  }
  SECTION("null payloads do not cause overflow")
  {
    auto out = run({min, max - 1}, false);
    CHECK(out->view().column(0).null_count() == 2);
  }
  SECTION("empty input")
  {
    auto out = run({});
    CHECK(out->num_rows() == 0);
  }
}

TEST_CASE("DATE to TIMESTAMP casts check overflow with out_of_range_exception",
          "[expression_evaluator][timestamp_bounds]")
{
  auto* space         = get_default_gpu_space();
  auto const strategy = GENERATE(MAT, exp_strategy_enum::AST_INTERPRET, exp_strategy_enum::AST_JIT);
  auto const target   = GENERATE(sirius::type_id::TIMESTAMP_SEC,
                               sirius::type_id::TIMESTAMP_MS,
                               sirius::type_id::TIMESTAMP,
                               sirius::type_id::TIMESTAMP_NS);
  auto const date_type   = sirius::logical_type::make(sirius::type_id::DATE);
  auto const target_type = sirius::logical_type::make(target);
  auto expr     = sirius::ast::node{sirius::ast::cast{make_ref_typed(0, date_type), target_type}};
  auto try_expr = sirius::ast::node{
    sirius::ast::cast{make_ref_typed(0, date_type), target_type, /*try_cast=*/true}};
  auto stream = cudf::get_default_stream();
  auto mr     = get_resource_ref(*space);

  // Year ~300000 exceeds the microsecond-intermediate day limit (106751991 days) and the
  // nanosecond day limit (106751 days).
  int32_t const out_of_range_day = 110'000'000;
  auto col                       = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, 2, cudf::mask_state::UNALLOCATED, stream, mr);
  std::vector<int32_t> days = {0, out_of_range_day};
  REQUIRE(cudaMemcpy(col->mutable_view().data<int32_t>(),
                     days.data(),
                     days.size() * sizeof(int32_t),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  auto view = cudf::bit_cast(col->view(), cudf::data_type{cudf::type_id::TIMESTAMP_DAYS});

  REQUIRE_THROWS_AS(run_native_ast(*space, &expr, cudf::table_view{{view}}, strategy),
                    sirius::out_of_range_exception);

  auto try_out = run_native_ast(*space, &try_expr, cudf::table_view{{view}}, strategy);
  REQUIRE(try_out->view().column(0).null_count() == 1);
}

TEST_CASE("abs on signed integer minimum throws out_of_range_exception",
          "[expression_evaluator][abs]")
{
  auto* space         = get_default_gpu_space();
  auto const strategy = GENERATE(MAT, exp_strategy_enum::AST_INTERPRET, exp_strategy_enum::AST_JIT);
  auto stream         = cudf::get_default_stream();
  auto mr             = get_resource_ref(*space);

  auto check_signed_min = [&]<typename T>(sirius::type_id logical_id, cudf::type_id physical_id) {
    auto const type = sirius::logical_type::make(logical_id);
    std::vector<std::unique_ptr<sirius::ast::node>> args;
    args.push_back(make_ref_typed(0, type));
    auto expr = sirius::ast::node{
      sirius::ast::function_call{sirius::function_id::abs, std::move(args), type}};

    auto col = cudf::make_numeric_column(
      cudf::data_type{physical_id}, 2, cudf::mask_state::UNALLOCATED, stream, mr);
    std::vector<T> values = {T{-5}, std::numeric_limits<T>::lowest()};
    REQUIRE(cudaMemcpy(col->mutable_view().data<T>(),
                       values.data(),
                       values.size() * sizeof(T),
                       cudaMemcpyHostToDevice) == cudaSuccess);

    REQUIRE_THROWS_AS(run_native_ast(*space, &expr, cudf::table_view{{col->view()}}, strategy),
                      sirius::out_of_range_exception);
  };

  check_signed_min.template operator()<int8_t>(sirius::type_id::TINYINT, cudf::type_id::INT8);
  check_signed_min.template operator()<int16_t>(sirius::type_id::SMALLINT, cudf::type_id::INT16);
  check_signed_min.template operator()<int32_t>(sirius::type_id::INTEGER, cudf::type_id::INT32);
  check_signed_min.template operator()<int64_t>(sirius::type_id::BIGINT, cudf::type_id::INT64);
}

TEST_CASE("subsecond extraction preserves minimum finite ticks",
          "[expression_evaluator][timestamp_bounds]")
{
  auto* space         = get_default_gpu_space();
  auto const strategy = GENERATE(MAT,
                                 exp_strategy_enum::AST_INTERPRET,
                                 exp_strategy_enum::AST_JIT,
                                 exp_strategy_enum::NVRTC_JIT);
  auto const days     = GENERATE(true, false);
  auto const function =
    GENERATE(sirius::function_id::millisecond, sirius::function_id::microsecond);
  auto const type =
    sirius::logical_type::make(days ? sirius::type_id::DATE : sirius::type_id::TIMESTAMP);
  auto stream = cudf::get_default_stream();
  auto mr     = get_resource_ref(*space);
  auto col =
    cudf::make_numeric_column(cudf::data_type{days ? cudf::type_id::INT32 : cudf::type_id::INT64},
                              4,
                              cudf::mask_state::ALL_VALID,
                              stream,
                              mr);
  if (days) {
    auto const max             = std::numeric_limits<int32_t>::max();
    std::vector<int32_t> ticks = {std::numeric_limits<int32_t>::min(), -max, max, 0};
    REQUIRE(cudaMemcpy(col->mutable_view().data<int32_t>(),
                       ticks.data(),
                       ticks.size() * sizeof(int32_t),
                       cudaMemcpyHostToDevice) == cudaSuccess);
  } else {
    auto const max             = std::numeric_limits<int64_t>::max();
    std::vector<int64_t> ticks = {std::numeric_limits<int64_t>::min(), -max, max, 0};
    REQUIRE(cudaMemcpy(col->mutable_view().data<int64_t>(),
                       ticks.data(),
                       ticks.size() * sizeof(int64_t),
                       cudaMemcpyHostToDevice) == cudaSuccess);
  }
  cudf::set_null_mask(col->mutable_view().null_mask(), 3, 4, false, stream);
  col->set_null_count(1);
  auto view = cudf::bit_cast(col->view(), sirius::require_cudf_type(type));
  std::vector<std::unique_ptr<sirius::ast::node>> args;
  args.push_back(make_ref_typed(0, type));
  auto expr         = sirius::ast::node{sirius::ast::function_call{
    function, std::move(args), sirius::logical_type::make(sirius::type_id::BIGINT)}};
  auto out          = run_native_ast(*space, &expr, cudf::table_view{{view}}, strategy);
  auto const result = out->view().column(0);
  std::vector<bool> expected_valids = {true, false, false, false};
  CHECK(copy_valids_to_host(result) == expected_valids);
  // INT64_MIN microseconds lies 5.224192 seconds into its minute.
  CHECK(copy_column_to_host<int64_t>(result)[0] == (days ? 0
                                                    : function == sirius::function_id::microsecond
                                                      ? 5224192
                                                      : 5224));
}

TEST_CASE("expression_evaluator_strategy - nvrtc_jit round-trip",
          "[expression_evaluator][nvrtc_jit]")
{
  sirius::expression_evaluator_strategy parsed{};
  REQUIRE(sirius::string_to_strategy("nvrtc_jit", parsed));
  CHECK(parsed == sirius::expression_evaluator_strategy::NVRTC_JIT);
  CHECK(sirius::strategy_to_string(sirius::expression_evaluator_strategy::NVRTC_JIT) ==
        "nvrtc_jit");
}

TEST_CASE("nvrtc_jit - Carrier Restoration inline in registers without temporary cast columns",
          "[expression_evaluator][nvrtc_jit][compressed_materialization]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  auto stream = cudf::get_default_stream();
  auto mr     = get_resource_ref(*space);

  // Column 0: INT16 physical carrier with BIGINT logical type
  std::vector<int16_t> int16_vals = {-32000, -1, 0, 42, 32000};
  auto col0                       = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT16}, 5, cudf::mask_state::UNALLOCATED, stream, mr);
  REQUIRE(cudaMemcpy(col0->mutable_view().data<int16_t>(),
                     int16_vals.data(),
                     int16_vals.size() * sizeof(int16_t),
                     cudaMemcpyHostToDevice) == cudaSuccess);

  // Column 1: DECIMAL32(scale=-2) physical carrier with DECIMAL(15, 2) (DECIMAL64) logical type
  std::vector<int32_t> dec32_vals = {-99999, 0, 12345, 500, -1};
  auto col1                       = cudf::make_fixed_point_column(
    cudf::data_type{cudf::type_id::DECIMAL32, -2}, 5, cudf::mask_state::ALL_VALID, stream, mr);
  REQUIRE(cudaMemcpy(col1->mutable_view().data<int32_t>(),
                     dec32_vals.data(),
                     dec32_vals.size() * sizeof(int32_t),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  cudf::set_null_mask(col1->mutable_view().null_mask(), 2, 3, false, stream);
  col1->set_null_count(1);

  // Column 2: INT16 physical carrier with DATE (TIMESTAMP_DAYS) logical type
  std::vector<int16_t> date_vals = {-1000, 0, 18000, 19500, 20000};
  auto col2                      = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT16}, 5, cudf::mask_state::UNALLOCATED, stream, mr);
  REQUIRE(cudaMemcpy(col2->mutable_view().data<int16_t>(),
                     date_vals.data(),
                     date_vals.size() * sizeof(int16_t),
                     cudaMemcpyHostToDevice) == cudaSuccess);

  cudf::table_view tv{{col0->view(), col1->view(), col2->view()}};

  auto const bigint_type = logical_type::make(type_id::BIGINT);
  auto const dec64_type  = logical_type::make_decimal(15, 2);
  auto const date_type   = logical_type::make(type_id::DATE);

  duckdb::vector<std::unique_ptr<ast_node>> exprs;
  exprs.push_back(make_ref_typed(0, bigint_type));
  exprs.push_back(make_ref_typed(1, dec64_type));
  exprs.push_back(make_ref_typed(2, date_type));
  // Reference column 0 a second time to verify deduplication in TranslationContext
  exprs.push_back(make_ref_typed(0, bigint_type));

  exp_executor mat_exec(
    exprs, mr, stream, sirius::expression_query_policy{exp_strategy_enum::MATERIALIZE});
  auto mat_out = mat_exec.evaluate(tv);
  REQUIRE(mat_out != nullptr);
  CHECK(mat_exec.restored_reference_cast_count_for_testing() == 3);

  exp_executor jit_exec(
    exprs, mr, stream, sirius::expression_query_policy{exp_strategy_enum::NVRTC_JIT});
  auto jit_out = jit_exec.evaluate(tv);
  REQUIRE(jit_out != nullptr);
  CHECK(jit_exec.restored_reference_cast_count_for_testing() == 0);

  REQUIRE(jit_out->num_columns() == 4);
  CHECK(jit_out->view().column(0).type() == cudf::data_type{cudf::type_id::INT64});
  CHECK(jit_out->view().column(1).type() == cudf::data_type{cudf::type_id::DECIMAL64, -2});
  CHECK(jit_out->view().column(2).type() == cudf::data_type{cudf::type_id::TIMESTAMP_DAYS});
  CHECK(jit_out->view().column(3).type() == cudf::data_type{cudf::type_id::INT64});

  CHECK(copy_column_to_host<int64_t>(jit_out->view().column(0)) ==
        copy_column_to_host<int64_t>(mat_out->view().column(0)));
  CHECK(copy_column_to_host<int64_t>(jit_out->view().column(3)) ==
        copy_column_to_host<int64_t>(mat_out->view().column(3)));
  CHECK(jit_out->view().column(0).null_count() == 0);
  CHECK(!jit_out->view().column(0).nullable());

  CHECK(jit_out->view().column(1).null_count() == 1);
  CHECK(copy_valids_to_host(jit_out->view().column(1)) ==
        copy_valids_to_host(mat_out->view().column(1)));
  auto const jit_dec = copy_column_to_host<int64_t>(jit_out->view().column(1));
  auto const mat_dec = copy_column_to_host<int64_t>(mat_out->view().column(1));
  auto const valids  = copy_valids_to_host(jit_out->view().column(1));
  for (std::size_t i = 0; i < valids.size(); ++i) {
    if (valids[i]) { CHECK(jit_dec[i] == mat_dec[i]); }
  }

  CHECK(copy_column_to_host<int32_t>(jit_out->view().column(2)) ==
        copy_column_to_host<int32_t>(mat_out->view().column(2)));
}

TEST_CASE("nvrtc_jit - ProjectSink evaluates references and constants across all fixed-width types",
          "[expression_evaluator][nvrtc_jit]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  auto stream = cudf::get_default_stream();
  auto mr     = get_resource_ref(*space);

  constexpr cudf::size_type n_rows = 73;  // non-multiple of 32 (3 warps: 32, 32, 9)

  auto make_col = [&]<typename Rep>(cudf::data_type dt,
                                    Rep base_val,
                                    bool with_nulls) -> std::unique_ptr<cudf::column> {
    std::vector<Rep> vals(n_rows);
    for (cudf::size_type i = 0; i < n_rows; ++i) {
      if constexpr (std::is_same_v<Rep, bool>) {
        vals[i] = ((i % 3) != 0);
      } else {
        vals[i] = static_cast<Rep>(base_val + static_cast<Rep>(i));
      }
    }
    rmm::device_buffer mask{0, stream, mr};
    cudf::size_type null_count = 0;
    if (with_nulls) {
      mask         = cudf::create_null_mask(n_rows, cudf::mask_state::ALL_VALID, stream, mr);
      auto* mask_p = static_cast<cudf::bitmask_type*>(mask.data());
      for (cudf::size_type i = 0; i < n_rows; ++i) {
        if ((i % 5) == 2 || i == 0 || i == n_rows - 1) {
          cudf::set_null_mask(mask_p, i, i + 1, false, stream);
          ++null_count;
        }
      }
    }
    std::unique_ptr<cudf::column> col;
    if (cudf::is_numeric(dt) || cudf::is_boolean(dt)) {
      if (with_nulls) {
        col = cudf::make_numeric_column(dt, n_rows, std::move(mask), null_count, stream, mr);
      } else {
        col = cudf::make_numeric_column(dt, n_rows, cudf::mask_state::UNALLOCATED, stream, mr);
      }
    } else if (cudf::is_timestamp(dt)) {
      if (with_nulls) {
        col = cudf::make_timestamp_column(dt, n_rows, std::move(mask), null_count, stream, mr);
      } else {
        col = cudf::make_timestamp_column(dt, n_rows, cudf::mask_state::UNALLOCATED, stream, mr);
      }
    } else {
      if (with_nulls) {
        col = cudf::make_fixed_point_column(dt, n_rows, std::move(mask), null_count, stream, mr);
      } else {
        col = cudf::make_fixed_point_column(dt, n_rows, cudf::mask_state::UNALLOCATED, stream, mr);
      }
    }
    if constexpr (std::is_same_v<Rep, bool>) {
      std::vector<std::uint8_t> bytes(n_rows);
      for (cudf::size_type i = 0; i < n_rows; ++i) {
        bytes[i] = vals[i] ? 1U : 0U;
      }
      REQUIRE(cudaMemcpy(col->mutable_view().data<bool>(),
                         bytes.data(),
                         n_rows * sizeof(std::uint8_t),
                         cudaMemcpyHostToDevice) == cudaSuccess);
    } else {
      REQUIRE(cudaMemcpy(col->mutable_view().data<Rep>(),
                         vals.data(),
                         n_rows * sizeof(Rep),
                         cudaMemcpyHostToDevice) == cudaSuccess);
    }
    return col;
  };

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(
    make_col.template operator()<std::int8_t>(cudf::data_type{cudf::type_id::INT8}, -20, true));
  cols.push_back(make_col.template operator()<std::int16_t>(
    cudf::data_type{cudf::type_id::INT16}, -1000, false));
  cols.push_back(make_col.template operator()<std::int32_t>(
    cudf::data_type{cudf::type_id::INT32}, -50000, true));
  cols.push_back(make_col.template operator()<std::int64_t>(
    cudf::data_type{cudf::type_id::INT64}, -1000000LL, false));
  cols.push_back(
    make_col.template operator()<std::uint8_t>(cudf::data_type{cudf::type_id::UINT8}, 10U, false));
  cols.push_back(make_col.template operator()<std::uint16_t>(
    cudf::data_type{cudf::type_id::UINT16}, 500U, true));
  cols.push_back(make_col.template operator()<std::uint32_t>(
    cudf::data_type{cudf::type_id::UINT32}, 10000U, false));
  cols.push_back(make_col.template operator()<std::uint64_t>(
    cudf::data_type{cudf::type_id::UINT64}, 999999ULL, true));
  cols.push_back(
    make_col.template operator()<float>(cudf::data_type{cudf::type_id::FLOAT32}, -12.5f, true));
  cols.push_back(
    make_col.template operator()<double>(cudf::data_type{cudf::type_id::FLOAT64}, 1024.125, false));
  cols.push_back(
    make_col.template operator()<bool>(cudf::data_type{cudf::type_id::BOOL8}, true, true));
  cols.push_back(make_col.template operator()<std::int32_t>(
    cudf::data_type{cudf::type_id::TIMESTAMP_DAYS}, 18000, true));
  cols.push_back(make_col.template operator()<std::int64_t>(
    cudf::data_type{cudf::type_id::TIMESTAMP_SECONDS}, 1700000000LL, false));
  cols.push_back(make_col.template operator()<std::int64_t>(
    cudf::data_type{cudf::type_id::TIMESTAMP_MILLISECONDS}, 1700000000000LL, true));
  cols.push_back(make_col.template operator()<std::int64_t>(
    cudf::data_type{cudf::type_id::TIMESTAMP_MICROSECONDS}, 1700000000000000LL, false));
  cols.push_back(make_col.template operator()<std::int64_t>(
    cudf::data_type{cudf::type_id::TIMESTAMP_NANOSECONDS}, 1700000000000000000LL, true));
  cols.push_back(make_col.template operator()<std::int32_t>(
    cudf::data_type{cudf::type_id::DECIMAL32, -2}, -12345, true));
  cols.push_back(make_col.template operator()<std::int64_t>(
    cudf::data_type{cudf::type_id::DECIMAL64, -4}, 987654321LL, false));
  cols.push_back(
    make_col.template operator()<__int128_t>(cudf::data_type{cudf::type_id::DECIMAL128, -6},
                                             (static_cast<__int128_t>(123456789LL) << 40) + 42,
                                             true));

  std::vector<cudf::column_view> col_views;
  for (auto const& c : cols) {
    col_views.push_back(c->view());
  }
  cudf::table_view tv{col_views};

  // Build a multi-expression projection covering all 19 references, all 19 non-null constants,
  // and typed NULL constants.
  duckdb::vector<std::unique_ptr<ast_node>> exprs;
  for (std::uint32_t i = 0; i < cols.size(); ++i) {
    exprs.push_back(make_ref(i));
  }

  // Non-null constants for all 19 types:
  exprs.push_back(std::make_unique<ast_node>(
    sirius::ast::constant{sirius::value{std::int8_t{-42}}, logical_type::make(type_id::TINYINT)}));
  exprs.push_back(std::make_unique<ast_node>(sirius::ast::constant{
    sirius::value{std::int16_t{-12345}}, logical_type::make(type_id::SMALLINT)}));
  exprs.push_back(make_int_const(12345678));
  exprs.push_back(make_bigint_const(-9876543210123LL));
  exprs.push_back(std::make_unique<ast_node>(sirius::ast::constant{
    sirius::value{std::uint8_t{200}}, logical_type::make(type_id::UTINYINT)}));
  exprs.push_back(std::make_unique<ast_node>(sirius::ast::constant{
    sirius::value{std::uint16_t{60000}}, logical_type::make(type_id::USMALLINT)}));
  exprs.push_back(std::make_unique<ast_node>(sirius::ast::constant{
    sirius::value{std::uint32_t{3000000000U}}, logical_type::make(type_id::UINTEGER)}));
  exprs.push_back(std::make_unique<ast_node>(sirius::ast::constant{
    sirius::value{std::uint64_t{1234567890123456789ULL}}, logical_type::make(type_id::UBIGINT)}));
  exprs.push_back(std::make_unique<ast_node>(
    sirius::ast::constant{sirius::value{float{3.25f}}, logical_type::make(type_id::FLOAT)}));
  exprs.push_back(make_double_const(-99.875));
  exprs.push_back(std::make_unique<ast_node>(
    sirius::ast::constant{sirius::value{true}, logical_type::make(type_id::BOOLEAN)}));
  exprs.push_back(make_date_const(19723));
  exprs.push_back(std::make_unique<ast_node>(
    sirius::ast::constant{sirius::value{sirius::timestamp_sec_value{1700000123LL}},
                          logical_type::make(type_id::TIMESTAMP_SEC)}));
  exprs.push_back(std::make_unique<ast_node>(
    sirius::ast::constant{sirius::value{sirius::timestamp_ms_value{1700000123456LL}},
                          logical_type::make(type_id::TIMESTAMP_MS)}));
  exprs.push_back(std::make_unique<ast_node>(
    sirius::ast::constant{sirius::value{sirius::timestamp_us_value{1700000123456789LL}},
                          logical_type::make(type_id::TIMESTAMP)}));
  exprs.push_back(std::make_unique<ast_node>(
    sirius::ast::constant{sirius::value{sirius::timestamp_ns_value{1700000123456789123LL}},
                          logical_type::make(type_id::TIMESTAMP_NS)}));
  exprs.push_back(make_dec32_const(-54321, 9, 2));
  exprs.push_back(make_dec64_const(1234567890123LL, 18, 4));
  exprs.push_back(make_dec128_const((static_cast<__int128_t>(-42) << 64) | 999999ULL, 38, 6));

  // Typed NULL constants:
  exprs.push_back(make_null_const(logical_type::make(type_id::INTEGER)));
  exprs.push_back(make_null_const(logical_type::make(type_id::BOOLEAN)));
  exprs.push_back(make_null_const(logical_type::make(type_id::DATE)));
  exprs.push_back(make_null_const(logical_type::make(type_id::TIMESTAMP)));
  exprs.push_back(make_null_const(logical_type::make_decimal(18, 4)));
  exprs.push_back(make_null_const(logical_type::make_decimal(38, 6)));

  exp_executor mat_exec(
    exprs, mr, stream, sirius::expression_query_policy{exp_strategy_enum::MATERIALIZE});
  auto mat_out = mat_exec.evaluate(tv);
  REQUIRE(mat_out != nullptr);

  exp_executor jit_exec(
    exprs, mr, stream, sirius::expression_query_policy{exp_strategy_enum::NVRTC_JIT});
  auto jit_out = jit_exec.evaluate(tv);
  REQUIRE(jit_out != nullptr);

  REQUIRE(jit_out->num_columns() == mat_out->num_columns());
  REQUIRE(jit_out->num_rows() == n_rows);

  for (cudf::size_type c = 0; c < jit_out->num_columns(); ++c) {
    CAPTURE(c);
    auto const jcol = jit_out->view().column(c);
    auto const mcol = mat_out->view().column(c);
    REQUIRE(jcol.type() == mcol.type());
    REQUIRE(jcol.null_count() == mcol.null_count());
    auto const jvalids = copy_valids_to_host(jcol);
    auto const mvalids = copy_valids_to_host(mcol);
    REQUIRE(jvalids == mvalids);

    auto const elem_size = cudf::size_of(jcol.type());
    std::vector<std::uint8_t> jbytes(n_rows * elem_size);
    std::vector<std::uint8_t> mbytes(n_rows * elem_size);
    REQUIRE(cudaMemcpy(jbytes.data(),
                       jcol.head<std::uint8_t>() + jcol.offset() * elem_size,
                       jbytes.size(),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    REQUIRE(cudaMemcpy(mbytes.data(),
                       mcol.head<std::uint8_t>() + mcol.offset() * elem_size,
                       mbytes.size(),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    for (cudf::size_type r = 0; r < n_rows; ++r) {
      if (jvalids[r]) {
        CHECK(std::memcmp(
                jbytes.data() + r * elem_size, mbytes.data() + r * elem_size, elem_size) == 0);
      }
    }
  }
}

TEST_CASE("nvrtc_jit - sliced input column_views with non-zero offset and 0-row batches",
          "[expression_evaluator][nvrtc_jit]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  auto stream = cudf::get_default_stream();
  auto mr     = get_resource_ref(*space);

  constexpr cudf::size_type full_rows = 110;
  std::vector<std::int64_t> values(full_rows);
  std::vector<bool> valids(full_rows);
  for (cudf::size_type i = 0; i < full_rows; ++i) {
    values[i] = static_cast<std::int64_t>(i * 100 - 500);
    valids[i] = ((i % 4) != 1);
  }

  auto mask   = cudf::create_null_mask(full_rows, cudf::mask_state::ALL_VALID, stream, mr);
  auto* maskp = static_cast<cudf::bitmask_type*>(mask.data());
  cudf::size_type null_count = 0;
  for (cudf::size_type i = 0; i < full_rows; ++i) {
    if (!valids[i]) {
      cudf::set_null_mask(maskp, i, i + 1, false, stream);
      ++null_count;
    }
  }
  auto col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT64}, full_rows, std::move(mask), null_count, stream, mr);
  REQUIRE(cudaMemcpy(col->mutable_view().data<std::int64_t>(),
                     values.data(),
                     full_rows * sizeof(std::int64_t),
                     cudaMemcpyHostToDevice) == cudaSuccess);

  duckdb::vector<std::unique_ptr<ast_node>> exprs;
  exprs.push_back(make_ref_typed(0, logical_type::make(type_id::BIGINT)));
  exprs.push_back(make_int_const(77));
  exprs.push_back(make_null_const(logical_type::make(type_id::BIGINT)));

  exp_executor mat_exec(
    exprs, mr, stream, sirius::expression_query_policy{exp_strategy_enum::MATERIALIZE});
  exp_executor jit_exec(
    exprs, mr, stream, sirius::expression_query_policy{exp_strategy_enum::NVRTC_JIT});

  SECTION("non-zero unaligned slice offset (offset 7, length 70)")
  {
    auto slices = cudf::slice(col->view(), {7, 77}, stream);
    REQUIRE(slices.size() == 1);
    REQUIRE(slices[0].offset() == 7);
    REQUIRE(slices[0].size() == 70);

    cudf::table_view sliced_tv{{slices[0]}};
    auto mat_out = mat_exec.evaluate(sliced_tv);
    auto jit_out = jit_exec.evaluate(sliced_tv);
    REQUIRE(mat_out != nullptr);
    REQUIRE(jit_out != nullptr);
    REQUIRE(jit_out->num_rows() == 70);

    for (cudf::size_type c = 0; c < 3; ++c) {
      auto const jcol = jit_out->view().column(c);
      auto const mcol = mat_out->view().column(c);
      CHECK(jcol.null_count() == mcol.null_count());
      CHECK(copy_valids_to_host(jcol) == copy_valids_to_host(mcol));
    }

    auto const jvals  = copy_column_to_host<std::int64_t>(jit_out->view().column(0));
    auto const mvals  = copy_column_to_host<std::int64_t>(mat_out->view().column(0));
    auto const jvalid = copy_valids_to_host(jit_out->view().column(0));
    for (std::size_t i = 0; i < jvals.size(); ++i) {
      if (jvalid[i]) { CHECK(jvals[i] == mvals[i]); }
    }
    CHECK(copy_column_to_host<std::int32_t>(jit_out->view().column(1)) ==
          copy_column_to_host<std::int32_t>(mat_out->view().column(1)));
    CHECK(jit_out->view().column(2).null_count() == 70);
  }

  SECTION("0-row empty batch")
  {
    auto slices = cudf::slice(col->view(), {15, 15}, stream);
    REQUIRE(slices.size() == 1);
    REQUIRE(slices[0].size() == 0);

    cudf::table_view empty_tv{{slices[0]}};
    auto jit_out = jit_exec.evaluate(empty_tv);
    REQUIRE(jit_out != nullptr);
    CHECK(jit_out->num_rows() == 0);
    REQUIRE(jit_out->num_columns() == 3);
    CHECK(jit_out->view().column(0).type() == cudf::data_type{cudf::type_id::INT64});
    CHECK(jit_out->view().column(1).type() == cudf::data_type{cudf::type_id::INT32});
    CHECK(jit_out->view().column(2).type() == cudf::data_type{cudf::type_id::INT64});
  }
}

TEST_CASE("nvrtc_jit - changing scalar constant values reuses cached compiled kernel",
          "[expression_evaluator][nvrtc_jit]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  auto stream = cudf::get_default_stream();
  auto mr     = get_resource_ref(*space);

  auto in = make_int32_input(*space);

  auto eval_with_constants = [&](std::int32_t c1, std::int64_t c2, bool c2_null) {
    duckdb::vector<std::unique_ptr<ast_node>> exprs;
    exprs.push_back(make_ref_typed(0, logical_type::make(type_id::INTEGER)));
    exprs.push_back(make_int_const(c1));
    if (c2_null) {
      exprs.push_back(make_null_const(logical_type::make(type_id::BIGINT)));
    } else {
      exprs.push_back(make_bigint_const(c2));
    }
    exp_executor exec(
      exprs, mr, stream, sirius::expression_query_policy{exp_strategy_enum::NVRTC_JIT});
    return exec.evaluate(in.view);
  };

  auto out1 = eval_with_constants(10, 1000LL, false);
  REQUIRE(out1 != nullptr);
  auto const cache_size_after_first = codegen::jit::KernelCache::instance().size();
  CHECK(cache_size_after_first >= 1);

  auto out2 = eval_with_constants(999, -5555555555LL, false);
  REQUIRE(out2 != nullptr);
  CHECK(codegen::jit::KernelCache::instance().size() == cache_size_after_first);

  auto out3 = eval_with_constants(-42, 0LL, true);
  REQUIRE(out3 != nullptr);
  CHECK(codegen::jit::KernelCache::instance().size() == cache_size_after_first);

  CHECK(copy_column_to_host<std::int32_t>(out1->view().column(1))[0] == 10);
  CHECK(copy_column_to_host<std::int32_t>(out2->view().column(1))[0] == 999);
  CHECK(copy_column_to_host<std::int32_t>(out3->view().column(1))[0] == -42);
  CHECK(out1->view().column(2).null_count() == 0);
  CHECK(out2->view().column(2).null_count() == 0);
  CHECK(out3->view().column(2).null_count() == in.view.num_rows());
}

// ============================================================================
// Ticket #6: Checked integer/float arithmetic, comparisons, BETWEEN, and
// unary_op (NOT, IS [NOT] NULL, TRY)
// ============================================================================

TEST_CASE("nvrtc_jit - binary arithmetic, abs, round, and constant_or_null match MATERIALIZE",
          "[expression_evaluator][nvrtc_jit][ticket6]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  auto stream = cudf::get_default_stream();
  auto mr     = get_resource_ref(*space);

  // Column 0: nullable INT32 (with Carrier Restoration from INT16!)
  // Column 1: non-nullable INT32
  // Column 2: nullable DOUBLE
  std::vector<std::int16_t> col0_vals = {10, -25, 0, 100, -7, 42};
  std::vector<bool> col0_valids       = {true, true, false, true, true, false};
  std::vector<std::int32_t> col1_vals = {3, 4, 5, -10, 2, 7};
  std::vector<double> col2_vals       = {2.675, -2.675, 123.456, 0.0, -98.765, 1.005};
  std::vector<bool> col2_valids       = {true, true, true, false, true, true};
  auto const n_rows                   = static_cast<cudf::size_type>(col0_vals.size());

  auto mask0             = cudf::create_null_mask(n_rows, cudf::mask_state::ALL_VALID, stream, mr);
  auto* mptr0            = static_cast<cudf::bitmask_type*>(mask0.data());
  cudf::size_type nulls0 = 0;
  for (cudf::size_type i = 0; i < n_rows; ++i) {
    if (!col0_valids[i]) {
      cudf::set_null_mask(mptr0, i, i + 1, false, stream);
      ++nulls0;
    }
  }
  auto col0 = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT16}, n_rows, std::move(mask0), nulls0, stream, mr);
  REQUIRE(cudaMemcpy(col0->mutable_view().data<std::int16_t>(),
                     col0_vals.data(),
                     n_rows * sizeof(std::int16_t),
                     cudaMemcpyHostToDevice) == cudaSuccess);

  auto col1 = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, n_rows, cudf::mask_state::UNALLOCATED, stream, mr);
  REQUIRE(cudaMemcpy(col1->mutable_view().data<std::int32_t>(),
                     col1_vals.data(),
                     n_rows * sizeof(std::int32_t),
                     cudaMemcpyHostToDevice) == cudaSuccess);

  auto mask2             = cudf::create_null_mask(n_rows, cudf::mask_state::ALL_VALID, stream, mr);
  auto* mptr2            = static_cast<cudf::bitmask_type*>(mask2.data());
  cudf::size_type nulls2 = 0;
  for (cudf::size_type i = 0; i < n_rows; ++i) {
    if (!col2_valids[i]) {
      cudf::set_null_mask(mptr2, i, i + 1, false, stream);
      ++nulls2;
    }
  }
  auto col2 = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::FLOAT64}, n_rows, std::move(mask2), nulls2, stream, mr);
  REQUIRE(cudaMemcpy(col2->mutable_view().data<double>(),
                     col2_vals.data(),
                     n_rows * sizeof(double),
                     cudaMemcpyHostToDevice) == cudaSuccess);

  cudf::table_view tv{{col0->view(), col1->view(), col2->view()}};

  auto const int_t = logical_type::make(type_id::INTEGER);
  auto const dbl_t = logical_type::make(type_id::DOUBLE);

  auto make_bin = [&](sirius::function_id fid,
                      std::unique_ptr<ast_node> l,
                      std::unique_ptr<ast_node> r,
                      logical_type ret) {
    std::vector<std::unique_ptr<ast_node>> args;
    args.push_back(std::move(l));
    args.push_back(std::move(r));
    return make_func(fid, std::move(args), ret);
  };

  duckdb::vector<std::unique_ptr<ast_node>> exprs;
  // 0: col0 (narrowed INT16->INT32) + col1
  exprs.push_back(
    make_bin(sirius::function_id::add, make_ref_typed(0, int_t), make_ref_typed(1, int_t), int_t));
  // 1: col0 - col1
  exprs.push_back(
    make_bin(sirius::function_id::sub, make_ref_typed(0, int_t), make_ref_typed(1, int_t), int_t));
  // 2: col0 * col1
  exprs.push_back(
    make_bin(sirius::function_id::mul, make_ref_typed(0, int_t), make_ref_typed(1, int_t), int_t));
  // 3: col0 // col1
  exprs.push_back(make_bin(
    sirius::function_id::int_div, make_ref_typed(0, int_t), make_ref_typed(1, int_t), int_t));
  // 4: col0 % col1
  exprs.push_back(
    make_bin(sirius::function_id::mod, make_ref_typed(0, int_t), make_ref_typed(1, int_t), int_t));
  // 5: abs(col0)
  {
    std::vector<std::unique_ptr<ast_node>> args;
    args.push_back(make_ref_typed(0, int_t));
    exprs.push_back(make_func(sirius::function_id::abs, std::move(args), int_t));
  }
  // 6: round(col2, 2)
  {
    std::vector<std::unique_ptr<ast_node>> args;
    args.push_back(make_ref_typed(2, dbl_t));
    args.push_back(make_int_const(2));
    exprs.push_back(make_func(sirius::function_id::round, std::move(args), dbl_t));
  }
  // 7: round(col2, -1)
  {
    std::vector<std::unique_ptr<ast_node>> args;
    args.push_back(make_ref_typed(2, dbl_t));
    args.push_back(make_int_const(-1));
    exprs.push_back(make_func(sirius::function_id::round, std::move(args), dbl_t));
  }
  // 8: constant_or_null(777, col0, col2)
  {
    std::vector<std::unique_ptr<ast_node>> args;
    args.push_back(make_int_const(777));
    args.push_back(make_ref_typed(0, int_t));
    args.push_back(make_ref_typed(2, dbl_t));
    exprs.push_back(make_func(sirius::function_id::constant_or_null, std::move(args), int_t));
  }

  exp_executor mat_exec(
    exprs, mr, stream, sirius::expression_query_policy{exp_strategy_enum::MATERIALIZE});
  auto mat_out = mat_exec.evaluate(tv);
  REQUIRE(mat_out != nullptr);
  CHECK(mat_exec.restored_reference_cast_count_for_testing() == 1);

  exp_executor jit_exec(
    exprs, mr, stream, sirius::expression_query_policy{exp_strategy_enum::NVRTC_JIT});
  auto jit_out = jit_exec.evaluate(tv);
  REQUIRE(jit_out != nullptr);
  // NVRTC_JIT restores col0 inline in registers!
  CHECK(jit_exec.restored_reference_cast_count_for_testing() == 0);

  REQUIRE(jit_out->num_columns() == mat_out->num_columns());
  for (cudf::size_type c = 0; c < jit_out->num_columns(); ++c) {
    CAPTURE(c);
    auto const jcol = jit_out->view().column(c);
    auto const mcol = mat_out->view().column(c);
    REQUIRE(jcol.type() == mcol.type());
    CHECK(jcol.null_count() == mcol.null_count());
    auto const jvalids = copy_valids_to_host(jcol);
    auto const mvalids = copy_valids_to_host(mcol);
    REQUIRE(jvalids == mvalids);
    if (jcol.type().id() == cudf::type_id::INT32) {
      auto const jv = copy_column_to_host<std::int32_t>(jcol);
      auto const mv = copy_column_to_host<std::int32_t>(mcol);
      for (std::size_t r = 0; r < jv.size(); ++r) {
        if (jvalids[r]) { CHECK(jv[r] == mv[r]); }
      }
    } else if (jcol.type().id() == cudf::type_id::FLOAT64) {
      auto const jv = copy_column_to_host<double>(jcol);
      auto const mv = copy_column_to_host<double>(mcol);
      for (std::size_t r = 0; r < jv.size(); ++r) {
        if (jvalids[r]) { CHECK(jv[r] == mv[r]); }
      }
    }
  }
}

TEST_CASE("nvrtc_jit - division and modulo by zero produce NULL without trapping",
          "[expression_evaluator][nvrtc_jit][ticket6]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  auto stream = cudf::get_default_stream();
  auto mr     = get_resource_ref(*space);

  // Row 0: 20 / 4 -> 5
  // Row 1: 20 / 0 (valid divisor 0) -> NULL
  // Row 2: NULL / 0 (NULL dividend with garbage 0 divisor) -> NULL
  // Row 3: 20 / NULL (NULL divisor with garbage 0 in buffer) -> NULL
  // Row 4: -15 / 2 -> -7
  std::vector<std::int32_t> a_vals = {20, 20, 99, 20, -15};
  std::vector<bool> a_valids       = {true, true, false, true, true};
  std::vector<std::int32_t> b_vals = {4, 0, 0, 0, 2};
  std::vector<bool> b_valids       = {true, true, true, false, true};

  auto batch = make_two_int32_batch_with_nulls(*space, a_vals, a_valids, b_vals, b_valids);
  auto ro    = batch->to_read_only();
  auto tv    = ro.get_data()->cast<gpu_table_representation>().get_table_view();

  auto const int_t = logical_type::make(type_id::INTEGER);
  auto make_bin    = [&](sirius::function_id fid) {
    std::vector<std::unique_ptr<ast_node>> args;
    args.push_back(make_ref_typed(0, int_t));
    args.push_back(make_ref_typed(1, int_t));
    return make_func(fid, std::move(args), int_t);
  };

  duckdb::vector<std::unique_ptr<ast_node>> exprs;
  exprs.push_back(make_bin(sirius::function_id::div));
  exprs.push_back(make_bin(sirius::function_id::int_div));
  exprs.push_back(make_bin(sirius::function_id::mod));

  exp_executor jit_exec(
    exprs, mr, stream, sirius::expression_query_policy{exp_strategy_enum::NVRTC_JIT});
  auto jit_out = jit_exec.evaluate(tv);
  REQUIRE(jit_out != nullptr);
  REQUIRE(jit_out->num_columns() == 3);

  std::vector<bool> expected_valids = {true, false, false, false, true};
  for (cudf::size_type c = 0; c < 3; ++c) {
    auto const col = jit_out->view().column(c);
    CHECK(col.null_count() == 3);
    CHECK(copy_valids_to_host(col) == expected_valids);
  }
  auto const div_vals  = copy_column_to_host<std::int32_t>(jit_out->view().column(0));
  auto const idiv_vals = copy_column_to_host<std::int32_t>(jit_out->view().column(1));
  auto const mod_vals  = copy_column_to_host<std::int32_t>(jit_out->view().column(2));
  CHECK(div_vals[0] == 5);
  CHECK(div_vals[4] == -7);
  CHECK(idiv_vals[0] == 5);
  CHECK(idiv_vals[4] == -7);
  CHECK(mod_vals[0] == 0);
  CHECK(mod_vals[4] == -1);
}

TEST_CASE(
  "nvrtc_jit - checked integer overflow throws out_of_range_exception and TRY suppresses it",
  "[expression_evaluator][nvrtc_jit][ticket6]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  auto stream = cudf::get_default_stream();
  auto mr     = get_resource_ref(*space);

  auto const int_t   = logical_type::make(type_id::INTEGER);
  auto const int_min = std::numeric_limits<std::int32_t>::min();
  auto const int_max = std::numeric_limits<std::int32_t>::max();

  auto eval_two_col = [&](std::vector<std::int32_t> const& a_vals,
                          std::vector<bool> const& a_valids,
                          std::vector<std::int32_t> const& b_vals,
                          std::vector<bool> const& b_valids,
                          ast_node const& expr) {
    auto batch = make_two_int32_batch_with_nulls(*space, a_vals, a_valids, b_vals, b_valids);
    auto ro    = batch->to_read_only();
    auto tv    = ro.get_data()->cast<gpu_table_representation>().get_table_view();
    exp_executor exec(
      expr, mr, stream, sirius::expression_query_policy{exp_strategy_enum::NVRTC_JIT});
    return exec.evaluate(tv);
  };

  auto make_bin = [&](sirius::function_id fid) {
    std::vector<std::unique_ptr<ast_node>> args;
    args.push_back(make_ref_typed(0, int_t));
    args.push_back(make_ref_typed(1, int_t));
    return make_func(fid, std::move(args), int_t);
  };

  SECTION("signed add overflow throws on valid row, ignored on NULL row, suppressed by TRY")
  {
    auto add_expr = make_bin(sirius::function_id::add);
    REQUIRE_THROWS_AS(eval_two_col({10, int_max}, {true, true}, {5, 1}, {true, true}, *add_expr),
                      sirius::out_of_range_exception);

    // When the overflowing row is NULL, no exception is thrown!
    auto null_out = eval_two_col({10, int_max}, {true, false}, {5, 1}, {true, true}, *add_expr);
    CHECK(null_out->view().column(0).null_count() == 1);
    CHECK(copy_column_to_host<std::int32_t>(null_out->view().column(0))[0] == 15);

    // When wrapped in TRY, the overflowing row becomes NULL and valid rows are preserved!
    auto try_expr =
      make_unary(sirius::ast::unary_op::kind::op_try, make_bin(sirius::function_id::add));
    auto try_out =
      eval_two_col({10, int_max, -5}, {true, true, true}, {5, 1, 2}, {true, true, true}, *try_expr);
    CHECK(try_out->view().column(0).null_count() == 1);
    CHECK(copy_valids_to_host(try_out->view().column(0)) == std::vector<bool>{true, false, true});
    auto const try_vals = copy_column_to_host<std::int32_t>(try_out->view().column(0));
    CHECK(try_vals[0] == 15);
    CHECK(try_vals[2] == -3);
  }

  SECTION("signed sub, mul, INT_MIN / -1, and abs(INT_MIN) overflow and INT_MIN % -1 == 0")
  {
    auto sub_expr = make_bin(sirius::function_id::sub);
    REQUIRE_THROWS_AS(eval_two_col({int_min}, {true}, {1}, {true}, *sub_expr),
                      sirius::out_of_range_exception);

    auto mul_expr = make_bin(sirius::function_id::mul);
    REQUIRE_THROWS_AS(eval_two_col({int_min}, {true}, {-1}, {true}, *mul_expr),
                      sirius::out_of_range_exception);

    auto div_expr = make_bin(sirius::function_id::div);
    REQUIRE_THROWS_AS(eval_two_col({int_min}, {true}, {-1}, {true}, *div_expr),
                      sirius::out_of_range_exception);

    auto idiv_expr = make_bin(sirius::function_id::int_div);
    REQUIRE_THROWS_AS(eval_two_col({int_min}, {true}, {-1}, {true}, *idiv_expr),
                      sirius::out_of_range_exception);

    // INT_MIN % -1 must yield 0 without trapping hardware %
    auto mod_expr = make_bin(sirius::function_id::mod);
    auto mod_out  = eval_two_col({int_min}, {true}, {-1}, {true}, *mod_expr);
    CHECK(mod_out->view().column(0).null_count() == 0);
    CHECK(copy_column_to_host<std::int32_t>(mod_out->view().column(0))[0] == 0);

    // abs(INT_MIN) throws, TRY(abs(INT_MIN)) yields NULL
    std::vector<std::unique_ptr<ast_node>> abs_args;
    abs_args.push_back(make_ref_typed(0, int_t));
    auto abs_expr = make_func(sirius::function_id::abs, std::move(abs_args), int_t);
    REQUIRE_THROWS_AS(eval_two_col({-42, int_min}, {true, true}, {0, 0}, {true, true}, *abs_expr),
                      sirius::out_of_range_exception);

    std::vector<std::unique_ptr<ast_node>> try_abs_args;
    try_abs_args.push_back(make_ref_typed(0, int_t));
    auto try_abs     = make_unary(sirius::ast::unary_op::kind::op_try,
                              make_func(sirius::function_id::abs, std::move(try_abs_args), int_t));
    auto try_abs_out = eval_two_col({-42, int_min}, {true, true}, {0, 0}, {true, true}, *try_abs);
    CHECK(try_abs_out->view().column(0).null_count() == 1);
    CHECK(copy_valids_to_host(try_abs_out->view().column(0)) == std::vector<bool>{true, false});
    CHECK(copy_column_to_host<std::int32_t>(try_abs_out->view().column(0))[0] == 42);
  }

  SECTION("unsigned integer overflow on add, sub, mul throws and TRY suppresses it")
  {
    auto const u64_t               = logical_type::make(type_id::UBIGINT);
    std::vector<std::uint64_t> u_a = {10ULL, std::numeric_limits<std::uint64_t>::max()};
    std::vector<std::uint64_t> u_b = {20ULL, 1ULL};
    auto col_a                     = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::UINT64}, 2, cudf::mask_state::UNALLOCATED, stream, mr);
    auto col_b = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::UINT64}, 2, cudf::mask_state::UNALLOCATED, stream, mr);
    REQUIRE(cudaMemcpy(col_a->mutable_view().data<std::uint64_t>(),
                       u_a.data(),
                       2 * sizeof(std::uint64_t),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(col_b->mutable_view().data<std::uint64_t>(),
                       u_b.data(),
                       2 * sizeof(std::uint64_t),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    cudf::table_view tv{{col_a->view(), col_b->view()}};

    auto make_u64_bin = [&](sirius::function_id fid) {
      std::vector<std::unique_ptr<ast_node>> args;
      args.push_back(make_ref_typed(0, u64_t));
      args.push_back(make_ref_typed(1, u64_t));
      return make_func(fid, std::move(args), u64_t);
    };

    auto u_add = make_u64_bin(sirius::function_id::add);
    exp_executor exec_add(
      *u_add, mr, stream, sirius::expression_query_policy{exp_strategy_enum::NVRTC_JIT});
    REQUIRE_THROWS_AS(exec_add.evaluate(tv), sirius::out_of_range_exception);

    // Sub underflow: 10 - 20 on unsigned!
    auto u_sub = make_u64_bin(sirius::function_id::sub);
    exp_executor exec_sub(
      *u_sub, mr, stream, sirius::expression_query_policy{exp_strategy_enum::NVRTC_JIT});
    REQUIRE_THROWS_AS(exec_sub.evaluate(tv), sirius::out_of_range_exception);

    auto try_sub =
      make_unary(sirius::ast::unary_op::kind::op_try, make_u64_bin(sirius::function_id::sub));
    exp_executor exec_try_sub(
      *try_sub, mr, stream, sirius::expression_query_policy{exp_strategy_enum::NVRTC_JIT});
    auto out_sub = exec_try_sub.evaluate(tv);
    CHECK(out_sub->view().column(0).null_count() == 1);
    CHECK(copy_valids_to_host(out_sub->view().column(0)) == std::vector<bool>{false, true});
    CHECK(copy_column_to_host<std::uint64_t>(out_sub->view().column(0))[1] ==
          std::numeric_limits<std::uint64_t>::max() - 1ULL);

    // Mul overflow: UINT64_MAX * 2 on unsigned!
    std::vector<std::uint64_t> u_m = {2ULL, 2ULL};
    REQUIRE(cudaMemcpy(col_b->mutable_view().data<std::uint64_t>(),
                       u_m.data(),
                       2 * sizeof(std::uint64_t),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    auto u_mul = make_u64_bin(sirius::function_id::mul);
    exp_executor exec_mul(
      *u_mul, mr, stream, sirius::expression_query_policy{exp_strategy_enum::NVRTC_JIT});
    REQUIRE_THROWS_AS(exec_mul.evaluate(tv), sirius::out_of_range_exception);

    auto try_mul =
      make_unary(sirius::ast::unary_op::kind::op_try, make_u64_bin(sirius::function_id::mul));
    exp_executor exec_try_mul(
      *try_mul, mr, stream, sirius::expression_query_policy{exp_strategy_enum::NVRTC_JIT});
    auto out_mul = exec_try_mul.evaluate(tv);
    CHECK(out_mul->view().column(0).null_count() == 1);
    CHECK(copy_valids_to_host(out_mul->view().column(0)) == std::vector<bool>{true, false});
    CHECK(copy_column_to_host<std::uint64_t>(out_mul->view().column(0))[0] == 20ULL);
  }

  SECTION("signed integer overflow across INT8, INT16, and INT64")
  {
    auto check_int_type = [&]<typename T>(type_id tid, cudf::type_id cid) {
      auto const lt     = logical_type::make(tid);
      std::vector<T> va = {T{5}, std::numeric_limits<T>::max(), std::numeric_limits<T>::min()};
      std::vector<T> vb = {T{3}, T{2}, T{-1}};
      auto ca           = cudf::make_numeric_column(
        cudf::data_type{cid}, 3, cudf::mask_state::UNALLOCATED, stream, mr);
      auto cb = cudf::make_numeric_column(
        cudf::data_type{cid}, 3, cudf::mask_state::UNALLOCATED, stream, mr);
      REQUIRE(cudaMemcpy(
                ca->mutable_view().data<T>(), va.data(), 3 * sizeof(T), cudaMemcpyHostToDevice) ==
              cudaSuccess);
      REQUIRE(cudaMemcpy(
                cb->mutable_view().data<T>(), vb.data(), 3 * sizeof(T), cudaMemcpyHostToDevice) ==
              cudaSuccess);
      cudf::table_view tv_t{{ca->view(), cb->view()}};

      std::vector<std::unique_ptr<ast_node>> add_args;
      add_args.push_back(make_ref_typed(0, lt));
      add_args.push_back(make_ref_typed(1, lt));
      auto add_node = make_func(sirius::function_id::add, std::move(add_args), lt);
      exp_executor exec_a(
        *add_node, mr, stream, sirius::expression_query_policy{exp_strategy_enum::NVRTC_JIT});
      REQUIRE_THROWS_AS(exec_a.evaluate(tv_t), sirius::out_of_range_exception);

      std::vector<std::unique_ptr<ast_node>> try_add_args;
      try_add_args.push_back(make_ref_typed(0, lt));
      try_add_args.push_back(make_ref_typed(1, lt));
      auto try_add_node =
        make_unary(sirius::ast::unary_op::kind::op_try,
                   make_func(sirius::function_id::add, std::move(try_add_args), lt));
      exp_executor exec_ta(
        *try_add_node, mr, stream, sirius::expression_query_policy{exp_strategy_enum::NVRTC_JIT});
      auto out_ta = exec_ta.evaluate(tv_t);
      CHECK(out_ta->view().column(0).null_count() == 2);
      CHECK(copy_valids_to_host(out_ta->view().column(0)) == std::vector<bool>{true, false, false});
      CHECK(copy_column_to_host<T>(out_ta->view().column(0))[0] == T{8});
    };

    check_int_type.template operator()<std::int8_t>(type_id::TINYINT, cudf::type_id::INT8);
    check_int_type.template operator()<std::int16_t>(type_id::SMALLINT, cudf::type_id::INT16);
    check_int_type.template operator()<std::int64_t>(type_id::BIGINT, cudf::type_id::INT64);
  }
}

TEST_CASE("nvrtc_jit - comparisons, IS [NOT] DISTINCT FROM, BETWEEN, NOT, and IS [NOT] NULL",
          "[expression_evaluator][nvrtc_jit][ticket6][narrow_domain]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  auto stream = cudf::get_default_stream();
  auto mr     = get_resource_ref(*space);

  // Nullable columns to test all comparison operators, DISTINCT FROM, BETWEEN, NOT, IS [NOT] NULL
  std::vector<std::int32_t> a_vals = {10, 20, 30, 99, 99, 50};
  std::vector<bool> a_valids       = {true, true, true, false, false, true};
  std::vector<std::int32_t> b_vals = {10, 25, 20, 99, 30, 50};
  std::vector<bool> b_valids       = {true, true, true, false, true, false};

  auto batch = make_two_int32_batch_with_nulls(*space, a_vals, a_valids, b_vals, b_valids);
  auto ro    = batch->to_read_only();
  auto tv    = ro.get_data()->cast<gpu_table_representation>().get_table_view();

  auto const int_t = logical_type::make(type_id::INTEGER);

  duckdb::vector<std::unique_ptr<ast_node>> exprs;
  exprs.push_back(
    make_cmp(sirius::comparison_type::equal, make_ref_typed(0, int_t), make_ref_typed(1, int_t)));
  exprs.push_back(make_cmp(
    sirius::comparison_type::not_equal, make_ref_typed(0, int_t), make_ref_typed(1, int_t)));
  exprs.push_back(
    make_cmp(sirius::comparison_type::lt, make_ref_typed(0, int_t), make_ref_typed(1, int_t)));
  exprs.push_back(
    make_cmp(sirius::comparison_type::le, make_ref_typed(0, int_t), make_ref_typed(1, int_t)));
  exprs.push_back(
    make_cmp(sirius::comparison_type::gt, make_ref_typed(0, int_t), make_ref_typed(1, int_t)));
  exprs.push_back(
    make_cmp(sirius::comparison_type::ge, make_ref_typed(0, int_t), make_ref_typed(1, int_t)));
  exprs.push_back(make_cmp(
    sirius::comparison_type::distinct_from, make_ref_typed(0, int_t), make_ref_typed(1, int_t)));
  exprs.push_back(make_cmp(sirius::comparison_type::not_distinct_from,
                           make_ref_typed(0, int_t),
                           make_ref_typed(1, int_t)));
  exprs.push_back(
    make_between(make_ref_typed(0, int_t), make_int_const(10), make_int_const(30), true, true));
  exprs.push_back(
    make_between(make_ref_typed(0, int_t), make_int_const(10), make_int_const(30), false, false));
  exprs.push_back(make_unary(
    sirius::ast::unary_op::kind::op_not,
    make_cmp(sirius::comparison_type::equal, make_ref_typed(0, int_t), make_int_const(20))));
  exprs.push_back(make_unary(sirius::ast::unary_op::kind::op_is_null, make_ref_typed(0, int_t)));
  exprs.push_back(
    make_unary(sirius::ast::unary_op::kind::op_is_not_null, make_ref_typed(0, int_t)));

  exp_executor mat_exec(
    exprs, mr, stream, sirius::expression_query_policy{exp_strategy_enum::MATERIALIZE});
  auto mat_out = mat_exec.evaluate(tv);
  REQUIRE(mat_out != nullptr);

  exp_executor jit_exec(
    exprs, mr, stream, sirius::expression_query_policy{exp_strategy_enum::NVRTC_JIT});
  auto jit_out = jit_exec.evaluate(tv);
  REQUIRE(jit_out != nullptr);

  REQUIRE(jit_out->num_columns() == mat_out->num_columns());
  for (cudf::size_type c = 0; c < jit_out->num_columns(); ++c) {
    CAPTURE(c);
    auto const jcol = jit_out->view().column(c);
    auto const mcol = mat_out->view().column(c);
    CHECK(jcol.null_count() == mcol.null_count());
    auto const jvalids = copy_valids_to_host(jcol);
    auto const mvalids = copy_valids_to_host(mcol);
    REQUIRE(jvalids == mvalids);
    auto const jv = copy_bool_column_to_host(jcol);
    auto const mv = copy_bool_column_to_host(mcol);
    for (std::size_t r = 0; r < jv.size(); ++r) {
      if (jvalids[r]) { CHECK(jv[r] == mv[r]); }
    }
  }

  SECTION(
    "narrow-domain comparison and BETWEEN preserve narrow_domain_comparison_count under NVRTC_JIT")
  {
    auto const bigint_t                = logical_type::make(type_id::BIGINT);
    auto const date_t                  = logical_type::make(type_id::DATE);
    std::vector<std::int16_t> narrow_a = {-10, 5, 15, 20, 100};
    std::vector<std::int16_t> narrow_b = {-5, 5, 20, 10, 100};
    auto col_na = make_carrier_column(*space, cudf::data_type{cudf::type_id::INT16}, narrow_a);
    auto col_nb = make_carrier_column(*space, cudf::data_type{cudf::type_id::INT16}, narrow_b);
    cudf::table_view narrow_tv{{col_na->view(), col_nb->view()}};

    duckdb::vector<std::unique_ptr<ast_node>> narrow_exprs;
    // 1. narrow ref vs constant
    narrow_exprs.push_back(
      make_cmp(sirius::comparison_type::gt, make_ref_typed(0, bigint_t), make_bigint_const(10)));
    // 2. narrow ref vs narrow ref
    narrow_exprs.push_back(
      make_cmp(sirius::comparison_type::le, make_ref_typed(0, date_t), make_ref_typed(1, date_t)));
    // 3. narrow BETWEEN
    narrow_exprs.push_back(make_between(
      make_ref_typed(0, bigint_t), make_bigint_const(0), make_bigint_const(20), true, false));

    exp_executor narrow_mat(
      narrow_exprs, mr, stream, sirius::expression_query_policy{exp_strategy_enum::MATERIALIZE});
    auto n_mat_out = narrow_mat.evaluate(narrow_tv);

    exp_executor narrow_jit(
      narrow_exprs, mr, stream, sirius::expression_query_policy{exp_strategy_enum::NVRTC_JIT});
    auto n_jit_out = narrow_jit.evaluate(narrow_tv);

    CHECK(narrow_jit.restored_reference_cast_count_for_testing() == 0);
    CHECK(narrow_jit.narrow_domain_comparison_count_for_testing() == 3);
    CHECK(narrow_mat.narrow_domain_comparison_count_for_testing() == 3);

    for (cudf::size_type c = 0; c < 3; ++c) {
      CHECK(copy_bool_column_to_host(n_jit_out->view().column(c)) ==
            copy_bool_column_to_host(n_mat_out->view().column(c)));
    }
  }

  SECTION(
    "select() with NVRTC_JIT evaluates comparison, BETWEEN, NOT, and IS [NOT] NULL predicates")
  {
    auto pred =
      make_between(make_ref_typed(0, int_t), make_int_const(15), make_int_const(50), true, true);
    exp_executor sel_mat(
      *pred, mr, stream, sirius::expression_query_policy{exp_strategy_enum::MATERIALIZE});
    exp_executor sel_jit(
      *pred, mr, stream, sirius::expression_query_policy{exp_strategy_enum::NVRTC_JIT});

    auto sel_mat_out = sel_mat.select(tv);
    auto sel_jit_out = sel_jit.select(tv);
    REQUIRE(sel_jit_out->num_rows() == sel_mat_out->num_rows());
    CHECK(copy_column_to_host<std::int32_t>(sel_jit_out->view().column(0)) ==
          copy_column_to_host<std::int32_t>(sel_mat_out->view().column(0)));
  }
}

// ============================================================================
// Ticket #7: Lazy branching & poison-safe control flow: CASE WHEN, COALESCE,
// Kleene AND/OR, and IN lists
// ============================================================================

TEST_CASE("nvrtc_jit - conjunction, case_expr, coalesce, and in_list match MATERIALIZE",
          "[expression_evaluator][nvrtc_jit][ticket7]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  auto stream = cudf::get_default_stream();
  auto mr     = get_resource_ref(*space);

  // Column 0: nullable INT32 (with Carrier Restoration from INT16!)
  // Column 1: nullable INT32
  std::vector<std::int16_t> col0_vals = {10, 20, 30, 40, 50, 10, 20, 30};
  std::vector<bool> col0_valids       = {true, false, true, false, true, true, false, false};
  std::vector<std::int32_t> col1_vals = {10, 25, 99, 40, 55, 99, 20, 99};
  std::vector<bool> col1_valids       = {true, true, false, false, true, false, true, false};
  auto const n_rows                   = static_cast<cudf::size_type>(col0_vals.size());

  auto mask0             = cudf::create_null_mask(n_rows, cudf::mask_state::ALL_VALID, stream, mr);
  auto* mptr0            = static_cast<cudf::bitmask_type*>(mask0.data());
  cudf::size_type nulls0 = 0;
  for (cudf::size_type i = 0; i < n_rows; ++i) {
    if (!col0_valids[i]) {
      cudf::set_null_mask(mptr0, i, i + 1, false, stream);
      ++nulls0;
    }
  }
  auto col0 = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT16}, n_rows, std::move(mask0), nulls0, stream, mr);
  REQUIRE(cudaMemcpy(col0->mutable_view().data<std::int16_t>(),
                     col0_vals.data(),
                     n_rows * sizeof(std::int16_t),
                     cudaMemcpyHostToDevice) == cudaSuccess);

  auto mask1             = cudf::create_null_mask(n_rows, cudf::mask_state::ALL_VALID, stream, mr);
  auto* mptr1            = static_cast<cudf::bitmask_type*>(mask1.data());
  cudf::size_type nulls1 = 0;
  for (cudf::size_type i = 0; i < n_rows; ++i) {
    if (!col1_valids[i]) {
      cudf::set_null_mask(mptr1, i, i + 1, false, stream);
      ++nulls1;
    }
  }
  auto col1 = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, n_rows, std::move(mask1), nulls1, stream, mr);
  REQUIRE(cudaMemcpy(col1->mutable_view().data<std::int32_t>(),
                     col1_vals.data(),
                     n_rows * sizeof(std::int32_t),
                     cudaMemcpyHostToDevice) == cudaSuccess);

  cudf::table_view tv{{col0->view(), col1->view()}};

  auto const int_t  = logical_type::make(type_id::INTEGER);
  auto const bool_t = logical_type::make(type_id::BOOLEAN);

  duckdb::vector<std::unique_ptr<ast_node>> exprs;

  // 0: Kleene AND: (col0 == 10) AND (col1 == 10)
  {
    std::vector<std::unique_ptr<ast_node>> ch;
    ch.push_back(
      make_cmp(sirius::comparison_type::equal, make_ref_typed(0, int_t), make_int_const(10)));
    ch.push_back(
      make_cmp(sirius::comparison_type::equal, make_ref_typed(1, int_t), make_int_const(10)));
    exprs.push_back(make_conj(sirius::ast::conjunction::kind::op_and, std::move(ch)));
  }

  // 1: Kleene OR: (col0 == 10) OR (col1 == 25) OR NULL
  {
    std::vector<std::unique_ptr<ast_node>> ch;
    ch.push_back(
      make_cmp(sirius::comparison_type::equal, make_ref_typed(0, int_t), make_int_const(10)));
    ch.push_back(
      make_cmp(sirius::comparison_type::equal, make_ref_typed(1, int_t), make_int_const(25)));
    ch.push_back(make_null_const(bool_t));
    exprs.push_back(make_conj(sirius::ast::conjunction::kind::op_or, std::move(ch)));
  }

  // 2: CASE WHEN col0 == 10 THEN col1 WHEN col1 == 25 THEN col0 ELSE -1 END
  {
    std::vector<sirius::ast::case_expr::when_then> cases;
    cases.push_back(
      {make_cmp(sirius::comparison_type::equal, make_ref_typed(0, int_t), make_int_const(10)),
       make_ref_typed(1, int_t)});
    cases.push_back(
      {make_cmp(sirius::comparison_type::equal, make_ref_typed(1, int_t), make_int_const(25)),
       make_ref_typed(0, int_t)});
    exprs.push_back(std::make_unique<ast_node>(
      sirius::ast::case_expr{std::move(cases), make_int_const(-1), int_t}));
  }

  // 3: COALESCE(col0, col1, 999)
  {
    std::vector<std::unique_ptr<ast_node>> ch;
    ch.push_back(make_ref_typed(0, int_t));
    ch.push_back(make_ref_typed(1, int_t));
    ch.push_back(make_int_const(999));
    exprs.push_back(make_coalesce(std::move(ch), int_t));
  }

  // 4: COALESCE(col0, col1) (leaving residual nulls)
  {
    std::vector<std::unique_ptr<ast_node>> ch;
    ch.push_back(make_ref_typed(0, int_t));
    ch.push_back(make_ref_typed(1, int_t));
    exprs.push_back(make_coalesce(std::move(ch), int_t));
  }

  // 5: col0 IN (10, 30, 50) (all constants)
  {
    std::vector<std::unique_ptr<ast_node>> vals;
    vals.push_back(make_int_const(10));
    vals.push_back(make_int_const(30));
    vals.push_back(make_int_const(50));
    exprs.push_back(make_in(make_ref_typed(0, int_t), std::move(vals), /*negated=*/false));
  }

  // 6: col0 NOT IN (10, 30, 50) (all constants)
  {
    std::vector<std::unique_ptr<ast_node>> vals;
    vals.push_back(make_int_const(10));
    vals.push_back(make_int_const(30));
    vals.push_back(make_int_const(50));
    exprs.push_back(make_in(make_ref_typed(0, int_t), std::move(vals), /*negated=*/true));
  }

  // 7: col0 IN (col1, 30) (expression/column list)
  {
    std::vector<std::unique_ptr<ast_node>> vals;
    vals.push_back(make_ref_typed(1, int_t));
    vals.push_back(make_int_const(30));
    exprs.push_back(make_in(make_ref_typed(0, int_t), std::move(vals), /*negated=*/false));
  }

  // 8: col0 NOT IN (col1, 30) (expression/column list)
  {
    std::vector<std::unique_ptr<ast_node>> vals;
    vals.push_back(make_ref_typed(1, int_t));
    vals.push_back(make_int_const(30));
    exprs.push_back(make_in(make_ref_typed(0, int_t), std::move(vals), /*negated=*/true));
  }

  exp_executor mat_exec(
    exprs, mr, stream, sirius::expression_query_policy{exp_strategy_enum::MATERIALIZE});
  auto mat_out = mat_exec.evaluate(tv);
  REQUIRE(mat_out != nullptr);
  CHECK(mat_exec.restored_reference_cast_count_for_testing() == 1);

  exp_executor jit_exec(
    exprs, mr, stream, sirius::expression_query_policy{exp_strategy_enum::NVRTC_JIT});
  auto jit_out = jit_exec.evaluate(tv);
  REQUIRE(jit_out != nullptr);
  // NVRTC_JIT must compile all 9 expressions in a single kernel and restore col0 in registers!
  CHECK(jit_exec.restored_reference_cast_count_for_testing() == 0);

  REQUIRE(jit_out->num_columns() == mat_out->num_columns());
  for (cudf::size_type c = 0; c < jit_out->num_columns(); ++c) {
    CAPTURE(c);
    auto const jcol = jit_out->view().column(c);
    auto const mcol = mat_out->view().column(c);
    REQUIRE(jcol.type() == mcol.type());
    CHECK(jcol.null_count() == mcol.null_count());
    auto const jvalids = copy_valids_to_host(jcol);
    auto const mvalids = copy_valids_to_host(mcol);
    REQUIRE(jvalids == mvalids);
    if (jcol.type().id() == cudf::type_id::BOOL8) {
      auto const jv = copy_bool_column_to_host(jcol);
      auto const mv = copy_bool_column_to_host(mcol);
      for (std::size_t r = 0; r < jv.size(); ++r) {
        if (jvalids[r]) { CHECK(jv[r] == mv[r]); }
      }
    } else {
      auto const jv = copy_column_to_host<std::int32_t>(jcol);
      auto const mv = copy_column_to_host<std::int32_t>(mcol);
      for (std::size_t r = 0; r < jv.size(); ++r) {
        if (jvalids[r]) { CHECK(jv[r] == mv[r]); }
      }
    }
  }
}

TEST_CASE("nvrtc_jit - lazy branching and poison-safe control flow for CASE, COALESCE, AND/OR, IN",
          "[expression_evaluator][nvrtc_jit][ticket7]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  auto stream = cudf::get_default_stream();
  auto mr     = get_resource_ref(*space);

  auto const int_t   = logical_type::make(type_id::INTEGER);
  auto const bool_t  = logical_type::make(type_id::BOOLEAN);
  auto const int_max = std::numeric_limits<std::int32_t>::max();

  // Helper: overflowing sub-expression `col0 + INT_MAX` (overflows whenever col0 > 0)
  auto make_ovf_int = [&]() {
    std::vector<std::unique_ptr<ast_node>> args;
    args.push_back(make_ref_typed(0, int_t));
    args.push_back(make_int_const(int_max));
    return make_func(sirius::function_id::add, std::move(args), int_t);
  };

  // Helper: overflowing boolean sub-expression `(col0 + INT_MAX) > 0`
  auto make_ovf_bool = [&]() {
    return make_cmp(sirius::comparison_type::gt, make_ovf_int(), make_int_const(0));
  };

  auto eval_one_col = [&](std::vector<std::int32_t> const& vals,
                          std::vector<bool> const& valids,
                          ast_node const& expr) {
    auto batch = make_int32_batch_with_nulls(*space, vals, valids);
    auto ro    = batch->to_read_only();
    auto tv    = ro.get_data()->cast<gpu_table_representation>().get_table_view();
    exp_executor exec(
      expr, mr, stream, sirius::expression_query_policy{exp_strategy_enum::NVRTC_JIT});
    return exec.evaluate(tv);
  };

  SECTION("Kleene AND/OR: FALSE AND <overflow> and <overflow> AND FALSE do not throw")
  {
    // col0 = {10, 20}: col0 + INT_MAX overflows on both rows!
    // 1. FALSE AND (col0 + INT_MAX > 0) -> FALSE (no overflow!)
    {
      std::vector<std::unique_ptr<ast_node>> ch;
      ch.push_back(
        make_cmp(sirius::comparison_type::lt, make_ref_typed(0, int_t), make_int_const(0)));
      ch.push_back(make_ovf_bool());
      auto expr = make_conj(sirius::ast::conjunction::kind::op_and, std::move(ch));
      auto out  = eval_one_col({10, 20}, {true, true}, *expr);
      CHECK(out->view().column(0).null_count() == 0);
      CHECK(copy_bool_column_to_host(out->view().column(0)) == std::vector<std::uint8_t>{0, 0});
    }

    // 2. (col0 + INT_MAX > 0) AND FALSE -> FALSE (RHS false dominates LHS overflow!)
    {
      std::vector<std::unique_ptr<ast_node>> ch;
      ch.push_back(make_ovf_bool());
      ch.push_back(
        make_cmp(sirius::comparison_type::lt, make_ref_typed(0, int_t), make_int_const(0)));
      auto expr = make_conj(sirius::ast::conjunction::kind::op_and, std::move(ch));
      auto out  = eval_one_col({10, 20}, {true, true}, *expr);
      CHECK(out->view().column(0).null_count() == 0);
      CHECK(copy_bool_column_to_host(out->view().column(0)) == std::vector<std::uint8_t>{0, 0});
    }

    // 3. TRUE OR (col0 + INT_MAX > 0) -> TRUE (no overflow!)
    {
      std::vector<std::unique_ptr<ast_node>> ch;
      ch.push_back(
        make_cmp(sirius::comparison_type::gt, make_ref_typed(0, int_t), make_int_const(0)));
      ch.push_back(make_ovf_bool());
      auto expr = make_conj(sirius::ast::conjunction::kind::op_or, std::move(ch));
      auto out  = eval_one_col({10, 20}, {true, true}, *expr);
      CHECK(out->view().column(0).null_count() == 0);
      CHECK(copy_bool_column_to_host(out->view().column(0)) == std::vector<std::uint8_t>{1, 1});
    }

    // 4. (col0 + INT_MAX > 0) OR TRUE -> TRUE (RHS true dominates LHS overflow!)
    {
      std::vector<std::unique_ptr<ast_node>> ch;
      ch.push_back(make_ovf_bool());
      ch.push_back(
        make_cmp(sirius::comparison_type::gt, make_ref_typed(0, int_t), make_int_const(0)));
      auto expr = make_conj(sirius::ast::conjunction::kind::op_or, std::move(ch));
      auto out  = eval_one_col({10, 20}, {true, true}, *expr);
      CHECK(out->view().column(0).null_count() == 0);
      CHECK(copy_bool_column_to_host(out->view().column(0)) == std::vector<std::uint8_t>{1, 1});
    }

    // 5. Undominated overflow: TRUE AND (col0 + INT_MAX > 0) throws out_of_range_exception!
    {
      std::vector<std::unique_ptr<ast_node>> ch;
      ch.push_back(
        make_cmp(sirius::comparison_type::gt, make_ref_typed(0, int_t), make_int_const(0)));
      ch.push_back(make_ovf_bool());
      auto expr = make_conj(sirius::ast::conjunction::kind::op_and, std::move(ch));
      REQUIRE_THROWS_AS(eval_one_col({10, 20}, {true, true}, *expr),
                        sirius::out_of_range_exception);
    }

    // 6. Undominated overflow: NULL AND (col0 + INT_MAX > 0) throws out_of_range_exception!
    {
      std::vector<std::unique_ptr<ast_node>> ch;
      ch.push_back(make_null_const(bool_t));
      ch.push_back(make_ovf_bool());
      auto expr = make_conj(sirius::ast::conjunction::kind::op_and, std::move(ch));
      REQUIRE_THROWS_AS(eval_one_col({10, 20}, {true, true}, *expr),
                        sirius::out_of_range_exception);
    }
  }

  SECTION("CASE WHEN: overflow in unselected THEN/ELSE or later WHEN does not throw")
  {
    // col0 = {10, 20}: col0 > 0 is TRUE for all rows.
    // 1. Overflow in ELSE branch when WHEN is TRUE -> does NOT throw!
    {
      std::vector<sirius::ast::case_expr::when_then> cases;
      cases.push_back(
        {make_cmp(sirius::comparison_type::gt, make_ref_typed(0, int_t), make_int_const(0)),
         make_ref_typed(0, int_t)});
      auto expr =
        std::make_unique<ast_node>(sirius::ast::case_expr{std::move(cases), make_ovf_int(), int_t});
      auto out = eval_one_col({10, 20}, {true, true}, *expr);
      CHECK(copy_column_to_host<std::int32_t>(out->view().column(0)) ==
            std::vector<std::int32_t>{10, 20});
    }

    // 2. Overflow in unselected THEN branch when WHEN is FALSE -> does NOT throw!
    {
      std::vector<sirius::ast::case_expr::when_then> cases;
      cases.push_back(
        {make_cmp(sirius::comparison_type::lt, make_ref_typed(0, int_t), make_int_const(0)),
         make_ovf_int()});
      auto expr = std::make_unique<ast_node>(
        sirius::ast::case_expr{std::move(cases), make_ref_typed(0, int_t), int_t});
      auto out = eval_one_col({10, 20}, {true, true}, *expr);
      CHECK(copy_column_to_host<std::int32_t>(out->view().column(0)) ==
            std::vector<std::int32_t>{10, 20});
    }

    // 3. Overflow in second WHEN condition after first WHEN matched -> does NOT throw!
    {
      std::vector<sirius::ast::case_expr::when_then> cases;
      cases.push_back(
        {make_cmp(sirius::comparison_type::gt, make_ref_typed(0, int_t), make_int_const(0)),
         make_int_const(100)});
      cases.push_back({make_ovf_bool(), make_int_const(200)});
      auto expr = std::make_unique<ast_node>(
        sirius::ast::case_expr{std::move(cases), make_int_const(0), int_t});
      auto out = eval_one_col({10, 20}, {true, true}, *expr);
      CHECK(copy_column_to_host<std::int32_t>(out->view().column(0)) ==
            std::vector<std::int32_t>{100, 100});
    }

    // 4. Overflow in SELECTED THEN branch throws out_of_range_exception (unless wrapped in TRY)!
    {
      std::vector<sirius::ast::case_expr::when_then> cases;
      cases.push_back(
        {make_cmp(sirius::comparison_type::gt, make_ref_typed(0, int_t), make_int_const(15)),
         make_ovf_int()});
      auto expr = std::make_unique<ast_node>(
        sirius::ast::case_expr{std::move(cases), make_ref_typed(0, int_t), int_t});
      REQUIRE_THROWS_AS(eval_one_col({10, 20}, {true, true}, *expr),
                        sirius::out_of_range_exception);

      auto try_expr = make_unary(sirius::ast::unary_op::kind::op_try, std::move(expr));
      auto try_out  = eval_one_col({10, 20}, {true, true}, *try_expr);
      CHECK(try_out->view().column(0).null_count() == 1);
      CHECK(copy_valids_to_host(try_out->view().column(0)) == std::vector<bool>{true, false});
      CHECK(copy_column_to_host<std::int32_t>(try_out->view().column(0))[0] == 10);
    }

    // 5. Overflow in an evaluated WHEN condition before any match throws out_of_range_exception!
    {
      std::vector<sirius::ast::case_expr::when_then> cases;
      cases.push_back({make_ovf_bool(), make_int_const(100)});
      auto expr = std::make_unique<ast_node>(
        sirius::ast::case_expr{std::move(cases), make_int_const(0), int_t});
      REQUIRE_THROWS_AS(eval_one_col({10, 20}, {true, true}, *expr),
                        sirius::out_of_range_exception);
    }
  }

  SECTION("COALESCE: overflow in unreached fallback does not throw; selected overflow throws")
  {
    // 1. col0 is non-null on all rows: COALESCE(col0, col0 + INT_MAX) does NOT throw!
    {
      std::vector<std::unique_ptr<ast_node>> ch;
      ch.push_back(make_ref_typed(0, int_t));
      ch.push_back(make_ovf_int());
      auto expr = make_coalesce(std::move(ch), int_t);
      auto out  = eval_one_col({10, 20}, {true, true}, *expr);
      CHECK(copy_column_to_host<std::int32_t>(out->view().column(0)) ==
            std::vector<std::int32_t>{10, 20});
    }

    // 2. When child 0 is NULL on row 1 and child 1 overflows on row 1, throws!
    {
      auto batch = make_two_int32_batch_with_nulls(
        *space, {10, 99}, {true, false}, {0, int_max}, {true, true});
      auto ro = batch->to_read_only();
      auto tv = ro.get_data()->cast<gpu_table_representation>().get_table_view();

      std::vector<std::unique_ptr<ast_node>> add_args;
      add_args.push_back(make_ref_typed(1, int_t));
      add_args.push_back(make_int_const(10));
      auto ovf_col1 = make_func(sirius::function_id::add, std::move(add_args), int_t);

      std::vector<std::unique_ptr<ast_node>> ch;
      ch.push_back(make_ref_typed(0, int_t));
      ch.push_back(std::move(ovf_col1));
      auto expr = make_coalesce(std::move(ch), int_t);

      exp_executor exec(
        *expr, mr, stream, sirius::expression_query_policy{exp_strategy_enum::NVRTC_JIT});
      REQUIRE_THROWS_AS(exec.evaluate(tv), sirius::out_of_range_exception);

      // Wrapped in TRY -> row 0 is 10, row 1 becomes NULL!
      auto try_expr = make_unary(sirius::ast::unary_op::kind::op_try, std::move(expr));
      exp_executor try_exec(
        *try_expr, mr, stream, sirius::expression_query_policy{exp_strategy_enum::NVRTC_JIT});
      auto try_out = try_exec.evaluate(tv);
      CHECK(try_out->view().column(0).null_count() == 1);
      CHECK(copy_valids_to_host(try_out->view().column(0)) == std::vector<bool>{true, false});
      CHECK(copy_column_to_host<std::int32_t>(try_out->view().column(0))[0] == 10);
    }
  }

  SECTION("CASE WHEN with function_id::error throws when WHEN is true and succeeds when false/null")
  {
    auto make_error_case = [&]() {
      std::vector<sirius::ast::case_expr::when_then> cases;
      cases.push_back(
        {make_cmp(sirius::comparison_type::gt, make_ref_typed(0, int_t), make_int_const(50)),
         make_func(sirius::function_id::error, {}, int_t)});
      return std::make_unique<ast_node>(
        sirius::ast::case_expr{std::move(cases), make_ref_typed(0, int_t), int_t});
    };

    // All rows have col0 <= 50 or NULL -> WHEN is never true -> succeeds!
    {
      auto expr = make_error_case();
      auto out  = eval_one_col({10, 50, 999}, {true, true, false}, *expr);
      CHECK(out->view().column(0).null_count() == 1);
      CHECK(copy_valids_to_host(out->view().column(0)) == std::vector<bool>{true, true, false});
      auto const vals = copy_column_to_host<std::int32_t>(out->view().column(0));
      CHECK(vals[0] == 10);
      CHECK(vals[1] == 50);
    }

    // One valid row has col0 = 60 > 50 -> throws internal_exception!
    {
      auto expr = make_error_case();
      REQUIRE_THROWS_AS(eval_one_col({10, 60, 20}, {true, true, true}, *expr),
                        sirius::internal_exception);
    }
  }

  SECTION("IN and NOT IN three-valued SQL semantics with NULL in list and narrow-domain carrier")
  {
    // x IN (10, NULL):
    //   x = 10 -> TRUE
    //   x = 20 -> NULL
    //   x = NULL -> NULL
    {
      std::vector<std::unique_ptr<ast_node>> vals;
      vals.push_back(make_int_const(10));
      vals.push_back(make_null_const(int_t));
      auto in_expr = make_in(make_ref_typed(0, int_t), std::move(vals), /*negated=*/false);
      auto out     = eval_one_col({10, 20, 99}, {true, true, false}, *in_expr);
      CHECK(out->view().column(0).null_count() == 2);
      CHECK(copy_valids_to_host(out->view().column(0)) == std::vector<bool>{true, false, false});
      CHECK(copy_bool_column_to_host(out->view().column(0))[0] == 1U);
    }

    // x NOT IN (10, NULL):
    //   x = 10 -> FALSE
    //   x = 20 -> NULL
    //   x = NULL -> NULL
    {
      std::vector<std::unique_ptr<ast_node>> vals;
      vals.push_back(make_int_const(10));
      vals.push_back(make_null_const(int_t));
      auto not_in_expr = make_in(make_ref_typed(0, int_t), std::move(vals), /*negated=*/true);
      auto out         = eval_one_col({10, 20, 99}, {true, true, false}, *not_in_expr);
      CHECK(out->view().column(0).null_count() == 2);
      CHECK(copy_valids_to_host(out->view().column(0)) == std::vector<bool>{true, false, false});
      CHECK(copy_bool_column_to_host(out->view().column(0))[0] == 0U);
    }

    // x IN (10, col0 + INT_MAX):
    //   when x = 10 on all rows, short-circuits on element 0 and does NOT throw overflow!
    //   even if evaluated in reverse `(col0 + INT_MAX, 10)`, valid match 10 dominates overflow!
    {
      std::vector<std::unique_ptr<ast_node>> vals1;
      vals1.push_back(make_int_const(10));
      vals1.push_back(make_ovf_int());
      auto in1  = make_in(make_ref_typed(0, int_t), std::move(vals1), /*negated=*/false);
      auto out1 = eval_one_col({10, 10}, {true, true}, *in1);
      CHECK(out1->view().column(0).null_count() == 0);
      CHECK(copy_bool_column_to_host(out1->view().column(0)) == std::vector<std::uint8_t>{1, 1});

      std::vector<std::unique_ptr<ast_node>> vals2;
      vals2.push_back(make_ovf_int());
      vals2.push_back(make_int_const(10));
      auto in2  = make_in(make_ref_typed(0, int_t), std::move(vals2), /*negated=*/false);
      auto out2 = eval_one_col({10, 10}, {true, true}, *in2);
      CHECK(out2->view().column(0).null_count() == 0);
      CHECK(copy_bool_column_to_host(out2->view().column(0)) == std::vector<std::uint8_t>{1, 1});

      // When x = 20 (does not match 10), the overflow is undominated and throws!
      REQUIRE_THROWS_AS(eval_one_col({10, 20}, {true, true}, *in2), sirius::out_of_range_exception);
    }

    // IN across DATE, TIMESTAMP, DECIMAL64, DOUBLE, and BOOL8 matching MATERIALIZE
    {
      auto const date_t  = logical_type::make(type_id::DATE);
      auto const ts_t    = logical_type::make(type_id::TIMESTAMP);
      auto const dec64_t = logical_type::make_decimal(18, 2);
      auto const dbl_t   = logical_type::make(type_id::DOUBLE);

      auto col_d = cudf::make_timestamp_column(cudf::data_type{cudf::type_id::TIMESTAMP_DAYS},
                                               3,
                                               cudf::mask_state::UNALLOCATED,
                                               stream,
                                               mr);
      std::vector<std::int32_t> d_v = {18000, 19000, 20000};
      REQUIRE(cudaMemcpy(col_d->mutable_view().data<std::int32_t>(),
                         d_v.data(),
                         3 * sizeof(std::int32_t),
                         cudaMemcpyHostToDevice) == cudaSuccess);

      auto col_ts =
        cudf::make_timestamp_column(cudf::data_type{cudf::type_id::TIMESTAMP_MICROSECONDS},
                                    3,
                                    cudf::mask_state::UNALLOCATED,
                                    stream,
                                    mr);
      std::vector<std::int64_t> ts_v = {1000000LL, 2000000LL, 3000000LL};
      REQUIRE(cudaMemcpy(col_ts->mutable_view().data<std::int64_t>(),
                         ts_v.data(),
                         3 * sizeof(std::int64_t),
                         cudaMemcpyHostToDevice) == cudaSuccess);

      auto col_dec = cudf::make_fixed_point_column(cudf::data_type{cudf::type_id::DECIMAL64, -2},
                                                   3,
                                                   cudf::mask_state::UNALLOCATED,
                                                   stream,
                                                   mr);
      std::vector<std::int64_t> dec_v = {125, 250, 500};
      REQUIRE(cudaMemcpy(col_dec->mutable_view().data<std::int64_t>(),
                         dec_v.data(),
                         3 * sizeof(std::int64_t),
                         cudaMemcpyHostToDevice) == cudaSuccess);

      auto col_dbl = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::FLOAT64}, 3, cudf::mask_state::UNALLOCATED, stream, mr);
      std::vector<double> dbl_v = {1.5, 2.5, 3.5};
      REQUIRE(cudaMemcpy(col_dbl->mutable_view().data<double>(),
                         dbl_v.data(),
                         3 * sizeof(double),
                         cudaMemcpyHostToDevice) == cudaSuccess);

      cudf::table_view multi_tv{{col_d->view(), col_ts->view(), col_dec->view(), col_dbl->view()}};

      duckdb::vector<std::unique_ptr<ast_node>> in_exprs;
      {
        std::vector<std::unique_ptr<ast_node>> v;
        v.push_back(make_date_const(18000));
        v.push_back(make_date_const(20000));
        in_exprs.push_back(make_in(make_ref_typed(0, date_t), std::move(v), false));
      }
      {
        std::vector<std::unique_ptr<ast_node>> v;
        v.push_back(std::make_unique<ast_node>(
          sirius::ast::constant{sirius::value{sirius::timestamp_us_value{2000000LL}}, ts_t}));
        in_exprs.push_back(make_in(make_ref_typed(1, ts_t), std::move(v), true));
      }
      {
        std::vector<std::unique_ptr<ast_node>> v;
        v.push_back(make_dec64_const(125, 18, 2));
        v.push_back(make_dec64_const(999, 18, 2));
        in_exprs.push_back(make_in(make_ref_typed(2, dec64_t), std::move(v), false));
      }
      {
        std::vector<std::unique_ptr<ast_node>> v;
        v.push_back(make_double_const(2.5));
        v.push_back(make_double_const(3.5));
        in_exprs.push_back(make_in(make_ref_typed(3, dbl_t), std::move(v), false));
      }

      exp_executor m_exec(
        in_exprs, mr, stream, sirius::expression_query_policy{exp_strategy_enum::MATERIALIZE});
      exp_executor j_exec(
        in_exprs, mr, stream, sirius::expression_query_policy{exp_strategy_enum::NVRTC_JIT});
      auto m_out = m_exec.evaluate(multi_tv);
      auto j_out = j_exec.evaluate(multi_tv);
      for (cudf::size_type c = 0; c < 4; ++c) {
        CHECK(copy_bool_column_to_host(j_out->view().column(c)) ==
              copy_bool_column_to_host(m_out->view().column(c)));
      }
    }
  }
}

// ============================================================================
// Ticket #8: DECIMAL(32/64/128) arithmetic, comparisons, and checked
// CAST / TRY_CAST (numeric, decimal, and temporal)
// ============================================================================

TEST_CASE("nvrtc_jit - DECIMAL(32/64/128) arithmetic, scale alignment, and comparisons",
          "[expression_evaluator][nvrtc_jit][ticket8][decimal]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  auto stream = cudf::get_default_stream();
  auto mr     = get_resource_ref(*space);

  SECTION("DECIMAL32, DECIMAL64, DECIMAL128 arithmetic (+, -, *, /, %, abs) and TPC-H Q1 shape")
  {
    // Col 0: DECIMAL32(8, 2) narrowed carrier inside a DECIMAL(15, 2) (DECIMAL64) logical column!
    // Col 1: DECIMAL64(18, 2)
    // Col 2: DECIMAL128(38, 2)
    std::vector<std::int32_t> col0_vals = {10000, -2500, 0, 50000};
    std::vector<std::int64_t> col1_vals = {5, 10, 15, 20};
    std::vector<__int128_t> col2_vals   = {100, -200, 300, 400};
    auto const n_rows                   = static_cast<cudf::size_type>(col0_vals.size());

    auto col0 = cudf::make_fixed_point_column(cudf::data_type{cudf::type_id::DECIMAL32, -2},
                                              n_rows,
                                              cudf::mask_state::UNALLOCATED,
                                              stream,
                                              mr);
    REQUIRE(cudaMemcpy(col0->mutable_view().data<std::int32_t>(),
                       col0_vals.data(),
                       n_rows * sizeof(std::int32_t),
                       cudaMemcpyHostToDevice) == cudaSuccess);

    auto col1 = cudf::make_fixed_point_column(cudf::data_type{cudf::type_id::DECIMAL64, -2},
                                              n_rows,
                                              cudf::mask_state::UNALLOCATED,
                                              stream,
                                              mr);
    REQUIRE(cudaMemcpy(col1->mutable_view().data<std::int64_t>(),
                       col1_vals.data(),
                       n_rows * sizeof(std::int64_t),
                       cudaMemcpyHostToDevice) == cudaSuccess);

    auto col2 = cudf::make_fixed_point_column(cudf::data_type{cudf::type_id::DECIMAL128, -2},
                                              n_rows,
                                              cudf::mask_state::UNALLOCATED,
                                              stream,
                                              mr);
    REQUIRE(cudaMemcpy(col2->mutable_view().data<__int128_t>(),
                       col2_vals.data(),
                       n_rows * sizeof(__int128_t),
                       cudaMemcpyHostToDevice) == cudaSuccess);

    cudf::table_view tv{{col0->view(), col1->view(), col2->view()}};

    auto const dec32_t  = logical_type::make_decimal(8, 2);
    auto const dec64_2  = logical_type::make_decimal(18, 2);
    auto const dec64_4  = logical_type::make_decimal(18, 4);
    auto const dec128_2 = logical_type::make_decimal(38, 2);

    auto make_bin = [&](sirius::function_id fid,
                        std::unique_ptr<ast_node> l,
                        std::unique_ptr<ast_node> r,
                        logical_type ret) {
      std::vector<std::unique_ptr<ast_node>> args;
      args.push_back(std::move(l));
      args.push_back(std::move(r));
      return make_func(fid, std::move(args), ret);
    };

    duckdb::vector<std::unique_ptr<ast_node>> exprs;
    // 0: DECIMAL32 add: col0 (as dec32) + 0.50
    exprs.push_back(make_bin(
      sirius::function_id::add, make_ref_typed(0, dec32_t), make_dec32_const(50, 8, 2), dec32_t));
    // 1: DECIMAL64 TPC-H Q1 shape: col0 (restored from dec32 to dec64) * (1.00 - col1) ->
    // DECIMAL(18, 4)
    exprs.push_back(make_bin(sirius::function_id::mul,
                             make_ref_typed(0, dec64_2),
                             make_bin(sirius::function_id::sub,
                                      make_dec64_const(100, 18, 2),
                                      make_ref_typed(1, dec64_2),
                                      dec64_2),
                             dec64_4));
    // 2: DECIMAL64 div: col0 (as dec64_2) / 2 (scale 0) -> dec64_2
    exprs.push_back(make_bin(
      sirius::function_id::div, make_ref_typed(0, dec64_2), make_dec64_const(2, 18, 0), dec64_2));
    // 3: DECIMAL64 mod: col0 (as dec64_2) % 0.07 -> dec64_2
    exprs.push_back(make_bin(
      sirius::function_id::mod, make_ref_typed(0, dec64_2), make_dec64_const(7, 18, 2), dec64_2));
    // 4: DECIMAL64 abs: abs(col0)
    {
      std::vector<std::unique_ptr<ast_node>> args;
      args.push_back(make_ref_typed(0, dec64_2));
      exprs.push_back(make_func(sirius::function_id::abs, std::move(args), dec64_2));
    }
    // 5: DECIMAL128 add: col2 + 1.25
    exprs.push_back(make_bin(sirius::function_id::add,
                             make_ref_typed(2, dec128_2),
                             make_dec128_const(125, 38, 2),
                             dec128_2));
    // 6: DECIMAL128 abs: abs(col2)
    {
      std::vector<std::unique_ptr<ast_node>> args;
      args.push_back(make_ref_typed(2, dec128_2));
      exprs.push_back(make_func(sirius::function_id::abs, std::move(args), dec128_2));
    }
    // 7: Cross-scale DECIMAL64 comparison: col0 (scale 2) > 50.000 (scale 3)
    exprs.push_back(make_cmp(
      sirius::comparison_type::gt, make_ref_typed(0, dec64_2), make_dec64_const(50000, 18, 3)));

    exp_executor mat_exec(
      exprs, mr, stream, sirius::expression_query_policy{exp_strategy_enum::MATERIALIZE});
    auto mat_out = mat_exec.evaluate(tv);
    REQUIRE(mat_out != nullptr);
    // MATERIALIZE allocates a temporary column to restore col0 from DECIMAL32 to DECIMAL64
    CHECK(mat_exec.restored_reference_cast_count_for_testing() == 1);

    exp_executor jit_exec(
      exprs, mr, stream, sirius::expression_query_policy{exp_strategy_enum::NVRTC_JIT});
    auto jit_out = jit_exec.evaluate(tv);
    REQUIRE(jit_out != nullptr);
    // NVRTC_JIT restores col0 inline in registers and compiles all 8 expressions in one kernel!
    CHECK(jit_exec.restored_reference_cast_count_for_testing() == 0);

    REQUIRE(jit_out->num_columns() == mat_out->num_columns());
    for (cudf::size_type c = 0; c < jit_out->num_columns(); ++c) {
      CAPTURE(c);
      auto const jcol = jit_out->view().column(c);
      auto const mcol = mat_out->view().column(c);
      REQUIRE(jcol.type() == mcol.type());
      CHECK(jcol.null_count() == mcol.null_count());
      auto const elem_size = cudf::size_of(jcol.type());
      std::vector<std::uint8_t> jbytes(n_rows * elem_size);
      std::vector<std::uint8_t> mbytes(n_rows * elem_size);
      REQUIRE(cudaMemcpy(
                jbytes.data(), jcol.head<std::uint8_t>(), jbytes.size(), cudaMemcpyDeviceToHost) ==
              cudaSuccess);
      REQUIRE(cudaMemcpy(
                mbytes.data(), mcol.head<std::uint8_t>(), mbytes.size(), cudaMemcpyDeviceToHost) ==
              cudaSuccess);
      CHECK(jbytes == mbytes);
    }
  }

  SECTION("DECIMAL arithmetic overflow throws out_of_range_exception and TRY suppresses it")
  {
    auto const dec32_t = logical_type::make_decimal(9, 2);
    // Carrier max for DECIMAL32 (int32_t) is INT32_MAX (2,147,483,647)
    auto input = make_decimal32_batch(*space, 2, {100, std::numeric_limits<std::int32_t>::max()});
    auto input_ro = input->to_read_only();
    auto tv       = input_ro.get_data()->cast<gpu_table_representation>().get_table_view();

    std::vector<std::unique_ptr<ast_node>> add_args;
    add_args.push_back(make_ref_typed(0, dec32_t));
    add_args.push_back(make_dec32_const(1, 9, 2));
    auto add_expr = make_func(sirius::function_id::add, std::move(add_args), dec32_t);

    exp_executor exec(
      *add_expr, mr, stream, sirius::expression_query_policy{exp_strategy_enum::NVRTC_JIT});
    REQUIRE_THROWS_AS(exec.evaluate(tv), sirius::out_of_range_exception);

    auto try_add = make_unary(sirius::ast::unary_op::kind::op_try, std::move(add_expr));
    exp_executor try_exec(
      *try_add, mr, stream, sirius::expression_query_policy{exp_strategy_enum::NVRTC_JIT});
    auto try_out = try_exec.evaluate(tv);
    CHECK(try_out->view().column(0).null_count() == 1);
    CHECK(copy_valids_to_host(try_out->view().column(0)) == std::vector<bool>{true, false});
    CHECK(copy_column_to_host<std::int32_t>(try_out->view().column(0))[0] == 101);

    // 128-bit multiplication overflow on DECIMAL128 throws and TRY suppresses it!
    auto const dec128_t = logical_type::make_decimal(38, 2);
    __int128_t const max128 =
      (static_cast<__int128_t>(std::numeric_limits<std::int64_t>::max()) << 64) |
      static_cast<__int128_t>(~0ULL);
    auto input128    = make_decimal128_batch(*space, 2, {100, max128});
    auto input128_ro = input128->to_read_only();
    auto tv128       = input128_ro.get_data()->cast<gpu_table_representation>().get_table_view();

    std::vector<std::unique_ptr<ast_node>> mul128_args;
    mul128_args.push_back(make_ref_typed(0, dec128_t));
    mul128_args.push_back(make_dec128_const(2, 38, 0));
    auto mul128_expr = make_func(sirius::function_id::mul, std::move(mul128_args), dec128_t);

    exp_executor exec128(
      *mul128_expr, mr, stream, sirius::expression_query_policy{exp_strategy_enum::NVRTC_JIT});
    REQUIRE_THROWS_AS(exec128.evaluate(tv128), sirius::out_of_range_exception);

    auto try_mul128 = make_unary(sirius::ast::unary_op::kind::op_try, std::move(mul128_expr));
    exp_executor try_exec128(
      *try_mul128, mr, stream, sirius::expression_query_policy{exp_strategy_enum::NVRTC_JIT});
    auto try_out128 = try_exec128.evaluate(tv128);
    CHECK(try_out128->view().column(0).null_count() == 1);
    CHECK(copy_valids_to_host(try_out128->view().column(0)) == std::vector<bool>{true, false});
    CHECK(static_cast<std::int64_t>(
            copy_column_to_host<__int128_t>(try_out128->view().column(0))[0]) == 200);
  }
}

TEST_CASE(
  "nvrtc_jit - checked CAST and TRY_CAST across numeric, boolean, decimal, and temporal types",
  "[expression_evaluator][nvrtc_jit][ticket8][cast]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  auto stream = cudf::get_default_stream();
  auto mr     = get_resource_ref(*space);

  SECTION("carrier_restore and valid semantic casts match MATERIALIZE")
  {
    // Col 0: INT16 carrier for DATE
    // Col 1: INT32
    // Col 2: DOUBLE
    // Col 3: DECIMAL64(18, 2)
    std::vector<std::int16_t> c0 = {0, 18000, -100};
    std::vector<std::int32_t> c1 = {0, 42, -7};
    std::vector<double> c2       = {0.0, 12.5, -3.25};
    std::vector<std::int64_t> c3 = {0, 1250, -375};

    auto col0 = make_carrier_column(*space, cudf::data_type{cudf::type_id::INT16}, c0);
    auto col1 = make_carrier_column(*space, cudf::data_type{cudf::type_id::INT32}, c1);
    auto col2 = make_carrier_column(*space, cudf::data_type{cudf::type_id::FLOAT64}, c2);
    auto col3 = cudf::make_fixed_point_column(
      cudf::data_type{cudf::type_id::DECIMAL64, -2}, 3, cudf::mask_state::UNALLOCATED, stream, mr);
    REQUIRE(cudaMemcpy(col3->mutable_view().data<std::int64_t>(),
                       c3.data(),
                       3 * sizeof(std::int64_t),
                       cudaMemcpyHostToDevice) == cudaSuccess);

    cudf::table_view tv{{col0->view(), col1->view(), col2->view(), col3->view()}};

    duckdb::vector<std::unique_ptr<ast_node>> exprs;
    // 0: carrier_restore INT16 -> DATE
    exprs.push_back(
      std::make_unique<ast_node>(sirius::ast::cast{make_ref(0),
                                                   logical_type::make(type_id::DATE),
                                                   /*try_cast=*/false,
                                                   sirius::ast::cast_kind::carrier_restore}));
    // 1: semantic INT32 -> BIGINT
    exprs.push_back(make_cast(make_ref_typed(1, logical_type::make(type_id::INTEGER)),
                              logical_type::make(type_id::BIGINT),
                              false));
    // 2: semantic INT32 -> DOUBLE
    exprs.push_back(make_cast(make_ref_typed(1, logical_type::make(type_id::INTEGER)),
                              logical_type::make(type_id::DOUBLE),
                              false));
    // 3: semantic INT32 -> BOOLEAN
    exprs.push_back(make_cast(make_ref_typed(1, logical_type::make(type_id::INTEGER)),
                              logical_type::make(type_id::BOOLEAN),
                              false));
    // 4: semantic INT32 -> DECIMAL(9, 2)
    exprs.push_back(make_cast(make_ref_typed(1, logical_type::make(type_id::INTEGER)),
                              logical_type::make_decimal(9, 2),
                              false));
    // 5: semantic DECIMAL(18, 2) -> DECIMAL(18, 4) (scale up)
    exprs.push_back(make_cast(make_ref_typed(3, logical_type::make_decimal(18, 2)),
                              logical_type::make_decimal(18, 4),
                              false));
    // 6: semantic DECIMAL(18, 2) -> DOUBLE
    exprs.push_back(make_cast(make_ref_typed(3, logical_type::make_decimal(18, 2)),
                              logical_type::make(type_id::DOUBLE),
                              false));
    // 7: semantic DOUBLE -> DECIMAL(18, 2)
    exprs.push_back(make_cast(make_ref_typed(2, logical_type::make(type_id::DOUBLE)),
                              logical_type::make_decimal(18, 2),
                              false));

    exp_executor mat_exec(
      exprs, mr, stream, sirius::expression_query_policy{exp_strategy_enum::MATERIALIZE});
    exp_executor jit_exec(
      exprs, mr, stream, sirius::expression_query_policy{exp_strategy_enum::NVRTC_JIT});

    auto mat_out = mat_exec.evaluate(tv);
    auto jit_out = jit_exec.evaluate(tv);
    REQUIRE(jit_out->num_columns() == mat_out->num_columns());
    for (cudf::size_type c = 0; c < jit_out->num_columns(); ++c) {
      CAPTURE(c);
      auto const jcol = jit_out->view().column(c);
      auto const mcol = mat_out->view().column(c);
      REQUIRE(jcol.type() == mcol.type());
      CHECK(jcol.null_count() == mcol.null_count());
      auto const elem_size = cudf::size_of(jcol.type());
      std::vector<std::uint8_t> jbytes(3 * elem_size);
      std::vector<std::uint8_t> mbytes(3 * elem_size);
      REQUIRE(cudaMemcpy(
                jbytes.data(), jcol.head<std::uint8_t>(), jbytes.size(), cudaMemcpyDeviceToHost) ==
              cudaSuccess);
      REQUIRE(cudaMemcpy(
                mbytes.data(), mcol.head<std::uint8_t>(), mbytes.size(), cudaMemcpyDeviceToHost) ==
              cudaSuccess);
      CHECK(jbytes == mbytes);
    }
  }

  SECTION("checked integer, float, and decimal CAST overflow throws and TRY_CAST returns NULL")
  {
    auto const int_t     = logical_type::make(type_id::INTEGER);
    auto const tinyint_t = logical_type::make(type_id::TINYINT);
    auto const uint_t    = logical_type::make(type_id::UINTEGER);
    auto const dec4_2    = logical_type::make_decimal(5, 2);  // max 999.99

    // Input INT32: {42, 200, -5}
    //   - CAST to TINYINT: 200 overflows!
    //   - CAST to UINTEGER: -5 overflows!
    //   - CAST to DECIMAL(5, 2): 42 -> 4200, 200 -> 20000, -5 -> -500 (all fit), but 1000
    //   overflows!
    auto batch = make_int32_batch_with_nulls(*space, {42, 200, -5, 1000}, {true, true, true, true});
    auto ro    = batch->to_read_only();
    auto tv    = ro.get_data()->cast<gpu_table_representation>().get_table_view();

    // 1. Strict CAST to TINYINT throws
    {
      auto expr = make_cast(make_ref_typed(0, int_t), tinyint_t, /*try_cast=*/false);
      exp_executor exec(
        *expr, mr, stream, sirius::expression_query_policy{exp_strategy_enum::NVRTC_JIT});
      REQUIRE_THROWS_AS(exec.evaluate(tv), sirius::out_of_range_exception);
    }

    // 2. Strict CAST to UINTEGER throws on -5
    {
      auto expr = make_cast(make_ref_typed(0, int_t), uint_t, /*try_cast=*/false);
      exp_executor exec(
        *expr, mr, stream, sirius::expression_query_policy{exp_strategy_enum::NVRTC_JIT});
      REQUIRE_THROWS_AS(exec.evaluate(tv), sirius::out_of_range_exception);
    }

    // 3. Strict CAST to DECIMAL(5, 2) throws on 1000
    {
      auto expr = make_cast(make_ref_typed(0, int_t), dec4_2, /*try_cast=*/false);
      exp_executor exec(
        *expr, mr, stream, sirius::expression_query_policy{exp_strategy_enum::NVRTC_JIT});
      REQUIRE_THROWS_AS(exec.evaluate(tv), sirius::out_of_range_exception);
    }

    // 4. TRY_CAST converts out-of-range rows to NULL without throwing!
    {
      duckdb::vector<std::unique_ptr<ast_node>> try_exprs;
      try_exprs.push_back(make_cast(make_ref_typed(0, int_t), tinyint_t, /*try_cast=*/true));
      try_exprs.push_back(make_cast(make_ref_typed(0, int_t), uint_t, /*try_cast=*/true));
      try_exprs.push_back(make_cast(make_ref_typed(0, int_t), dec4_2, /*try_cast=*/true));

      exp_executor exec(
        try_exprs, mr, stream, sirius::expression_query_policy{exp_strategy_enum::NVRTC_JIT});
      auto out = exec.evaluate(tv);
      REQUIRE(out != nullptr);
      REQUIRE(out->num_columns() == 3);

      // Col 0 (TINYINT): {42, NULL, -5, NULL}
      CHECK(out->view().column(0).null_count() == 2);
      CHECK(copy_valids_to_host(out->view().column(0)) ==
            std::vector<bool>{true, false, true, false});
      auto const c0_vals = copy_column_to_host<std::int8_t>(out->view().column(0));
      CHECK(c0_vals[0] == 42);
      CHECK(c0_vals[2] == -5);

      // Col 1 (UINTEGER): {42, 200, NULL, 1000}
      CHECK(out->view().column(1).null_count() == 1);
      CHECK(copy_valids_to_host(out->view().column(1)) ==
            std::vector<bool>{true, true, false, true});
      auto const c1_vals = copy_column_to_host<std::uint32_t>(out->view().column(1));
      CHECK(c1_vals[0] == 42U);
      CHECK(c1_vals[1] == 200U);
      CHECK(c1_vals[3] == 1000U);

      // Col 2 (DECIMAL(5, 2)): {4200, 20000, -500, NULL}
      CHECK(out->view().column(2).null_count() == 1);
      CHECK(copy_valids_to_host(out->view().column(2)) ==
            std::vector<bool>{true, true, true, false});
      auto const c2_vals = copy_column_to_host<std::int32_t>(out->view().column(2));
      CHECK(c2_vals[0] == 4200);
      CHECK(c2_vals[1] == 20000);
      CHECK(c2_vals[2] == -500);
    }

    // 5. Float/Double -> Integer, Float, and Decimal overflow checks and TRY_CAST
    {
      auto const dbl_t           = logical_type::make(type_id::DOUBLE);
      auto const flt_t           = logical_type::make(type_id::FLOAT);
      auto const i64_t           = logical_type::make(type_id::BIGINT);
      auto const u64_t           = logical_type::make(type_id::UBIGINT);
      std::vector<double> d_vals = {
        12.5,
        1e300,                            // overflows FLOAT, BIGINT, DECIMAL(5,2)
        static_cast<double>(1ULL << 63),  // 2^63: exact power of two just above INT64_MAX!
        -1.0                              // negative: overflows UBIGINT!
      };
      auto col_dbl = make_carrier_column(*space, cudf::data_type{cudf::type_id::FLOAT64}, d_vals);
      cudf::table_view d_tv{{col_dbl->view()}};

      // Strict casts throw out_of_range_exception
      {
        auto c_flt = make_cast(make_ref_typed(0, dbl_t), flt_t, false);
        exp_executor ex(
          *c_flt, mr, stream, sirius::expression_query_policy{exp_strategy_enum::NVRTC_JIT});
        REQUIRE_THROWS_AS(ex.evaluate(d_tv), sirius::out_of_range_exception);
      }
      {
        auto c_i64 = make_cast(make_ref_typed(0, dbl_t), i64_t, false);
        exp_executor ex(
          *c_i64, mr, stream, sirius::expression_query_policy{exp_strategy_enum::NVRTC_JIT});
        REQUIRE_THROWS_AS(ex.evaluate(d_tv), sirius::out_of_range_exception);
      }
      {
        auto c_u64 = make_cast(make_ref_typed(0, dbl_t), u64_t, false);
        exp_executor ex(
          *c_u64, mr, stream, sirius::expression_query_policy{exp_strategy_enum::NVRTC_JIT});
        REQUIRE_THROWS_AS(ex.evaluate(d_tv), sirius::out_of_range_exception);
      }

      // TRY_CAST succeeds and turns overflowing elements into NULL
      duckdb::vector<std::unique_ptr<ast_node>> d_try;
      d_try.push_back(make_cast(make_ref_typed(0, dbl_t), flt_t, true));
      d_try.push_back(make_cast(make_ref_typed(0, dbl_t), i64_t, true));
      d_try.push_back(make_cast(make_ref_typed(0, dbl_t), u64_t, true));
      d_try.push_back(make_cast(make_ref_typed(0, dbl_t), dec4_2, true));

      exp_executor d_exec(
        d_try, mr, stream, sirius::expression_query_policy{exp_strategy_enum::NVRTC_JIT});
      auto d_out = d_exec.evaluate(d_tv);
      // Col 0 (FLOAT): row 1 (1e300) is NULL
      CHECK(copy_valids_to_host(d_out->view().column(0)) ==
            std::vector<bool>{true, false, true, true});
      // Col 1 (BIGINT): row 1 (1e300) and row 2 (2^63) are NULL
      CHECK(copy_valids_to_host(d_out->view().column(1)) ==
            std::vector<bool>{true, false, false, true});
      CHECK(copy_column_to_host<std::int64_t>(d_out->view().column(1))[0] == 12);
      CHECK(copy_column_to_host<std::int64_t>(d_out->view().column(1))[3] == -1);
      // Col 2 (UBIGINT): row 1 (1e300) and row 3 (-1.0) are NULL; row 2 (2^63) fits UBIGINT!
      CHECK(copy_valids_to_host(d_out->view().column(2)) ==
            std::vector<bool>{true, false, true, false});
      CHECK(copy_column_to_host<std::uint64_t>(d_out->view().column(2))[0] == 12ULL);
      CHECK(copy_column_to_host<std::uint64_t>(d_out->view().column(2))[2] == (1ULL << 63));
      // Col 3 (DECIMAL(5,2)): rows 1 and 2 are NULL
      CHECK(copy_valids_to_host(d_out->view().column(3)) ==
            std::vector<bool>{true, false, false, true});
      CHECK(copy_column_to_host<std::int32_t>(d_out->view().column(3))[0] == 1250);
      CHECK(copy_column_to_host<std::int32_t>(d_out->view().column(3))[3] == -100);
    }

    // 6. DECIMAL -> DECIMAL scale down and DECIMAL -> INT / BOOL casts
    {
      auto const dec18_3 = logical_type::make_decimal(18, 3);
      auto const dec5_1  = logical_type::make_decimal(5, 1);  // max 9999.9 (unscaled 99999)
      auto const bool_t  = logical_type::make(type_id::BOOLEAN);
      // Values at scale 3: 12.345 (12345), -99.900 (-99900), 0.000 (0), 50000.000 (50000000 ->
      // overflows dec5_1)
      auto dec_in    = make_decimal64_batch(*space, 3, {12345, -99900, 0, 50000000});
      auto dec_in_ro = dec_in->to_read_only();
      auto dec_tv    = dec_in_ro.get_data()->cast<gpu_table_representation>().get_table_view();

      auto strict_dec = make_cast(make_ref_typed(0, dec18_3), dec5_1, false);
      exp_executor ex_strict(
        *strict_dec, mr, stream, sirius::expression_query_policy{exp_strategy_enum::NVRTC_JIT});
      REQUIRE_THROWS_AS(ex_strict.evaluate(dec_tv), sirius::out_of_range_exception);

      duckdb::vector<std::unique_ptr<ast_node>> dec_exprs;
      dec_exprs.push_back(make_cast(make_ref_typed(0, dec18_3), dec5_1, true));
      dec_exprs.push_back(make_cast(make_ref_typed(0, dec18_3), tinyint_t, true));
      dec_exprs.push_back(make_cast(make_ref_typed(0, dec18_3), bool_t, false));

      exp_executor ex_dec(
        dec_exprs, mr, stream, sirius::expression_query_policy{exp_strategy_enum::NVRTC_JIT});
      auto dec_out = ex_dec.evaluate(dec_tv);
      // Col 0 (DECIMAL(5,1)): {123, -999, 0, NULL}
      CHECK(copy_valids_to_host(dec_out->view().column(0)) ==
            std::vector<bool>{true, true, true, false});
      auto const d5_vals = copy_column_to_host<std::int32_t>(dec_out->view().column(0));
      CHECK(d5_vals[0] == 123);
      CHECK(d5_vals[1] == -999);
      CHECK(d5_vals[2] == 0);
      // Col 1 (TINYINT): {12, -99, 0, NULL}
      CHECK(copy_valids_to_host(dec_out->view().column(1)) ==
            std::vector<bool>{true, true, true, false});
      auto const ti_vals = copy_column_to_host<std::int8_t>(dec_out->view().column(1));
      CHECK(ti_vals[0] == 12);
      CHECK(ti_vals[1] == -99);
      CHECK(ti_vals[2] == 0);
      // Col 2 (BOOLEAN): {1, 1, 0, 1}
      CHECK(copy_bool_column_to_host(dec_out->view().column(2)) ==
            std::vector<std::uint8_t>{1, 1, 0, 1});
    }
  }

  SECTION(
    "temporal CASTs (DATE -> TIMESTAMP and TIMESTAMP_S/MS/NS -> TIMESTAMP_US) under NVRTC_JIT")
  {
    auto const date_t  = logical_type::make(type_id::DATE);
    auto const ts_s_t  = logical_type::make(type_id::TIMESTAMP_SEC);
    auto const ts_ms_t = logical_type::make(type_id::TIMESTAMP_MS);
    auto const ts_us_t = logical_type::make(type_id::TIMESTAMP);
    auto const ts_ns_t = logical_type::make(type_id::TIMESTAMP_NS);

    auto const date_inf = std::numeric_limits<std::int32_t>::max();
    auto const ts_inf   = std::numeric_limits<std::int64_t>::max();

    // 1. DATE -> TIMESTAMP_{S, MS, US, NS} with ±infinity and finite values
    {
      std::vector<std::int32_t> days = {-date_inf, -1, 0, 18000, date_inf};
      auto col_d = cudf::make_timestamp_column(cudf::data_type{cudf::type_id::TIMESTAMP_DAYS},
                                               5,
                                               cudf::mask_state::UNALLOCATED,
                                               stream,
                                               mr);
      REQUIRE(cudaMemcpy(col_d->mutable_view().data<std::int32_t>(),
                         days.data(),
                         5 * sizeof(std::int32_t),
                         cudaMemcpyHostToDevice) == cudaSuccess);
      cudf::table_view tv{{col_d->view()}};

      duckdb::vector<std::unique_ptr<ast_node>> exprs;
      exprs.push_back(make_cast(make_ref_typed(0, date_t), ts_s_t, false));
      exprs.push_back(make_cast(make_ref_typed(0, date_t), ts_ms_t, false));
      exprs.push_back(make_cast(make_ref_typed(0, date_t), ts_us_t, false));
      exprs.push_back(make_cast(make_ref_typed(0, date_t), ts_ns_t, false));

      exp_executor mat_exec(
        exprs, mr, stream, sirius::expression_query_policy{exp_strategy_enum::MATERIALIZE});
      exp_executor jit_exec(
        exprs, mr, stream, sirius::expression_query_policy{exp_strategy_enum::NVRTC_JIT});
      auto mat_out = mat_exec.evaluate(tv);
      auto jit_out = jit_exec.evaluate(tv);
      for (cudf::size_type c = 0; c < 4; ++c) {
        CHECK(copy_column_to_host<std::int64_t>(jit_out->view().column(c)) ==
              copy_column_to_host<std::int64_t>(mat_out->view().column(c)));
      }
    }

    // 2. DATE -> TIMESTAMP out-of-range throws on CAST and returns NULL on TRY_CAST
    {
      std::vector<std::int32_t> days = {0, 110'000'000};
      auto col_d = cudf::make_timestamp_column(cudf::data_type{cudf::type_id::TIMESTAMP_DAYS},
                                               2,
                                               cudf::mask_state::UNALLOCATED,
                                               stream,
                                               mr);
      REQUIRE(cudaMemcpy(col_d->mutable_view().data<std::int32_t>(),
                         days.data(),
                         2 * sizeof(std::int32_t),
                         cudaMemcpyHostToDevice) == cudaSuccess);
      cudf::table_view tv{{col_d->view()}};

      auto strict_cast = make_cast(make_ref_typed(0, date_t), ts_us_t, /*try_cast=*/false);
      exp_executor strict_exec(
        *strict_cast, mr, stream, sirius::expression_query_policy{exp_strategy_enum::NVRTC_JIT});
      REQUIRE_THROWS_AS(strict_exec.evaluate(tv), sirius::out_of_range_exception);

      auto try_cast = make_cast(make_ref_typed(0, date_t), ts_us_t, /*try_cast=*/true);
      exp_executor try_exec(
        *try_cast, mr, stream, sirius::expression_query_policy{exp_strategy_enum::NVRTC_JIT});
      auto try_out = try_exec.evaluate(tv);
      CHECK(try_out->view().column(0).null_count() == 1);
      CHECK(copy_valids_to_host(try_out->view().column(0)) == std::vector<bool>{true, false});
    }

    // 3. TIMESTAMP_NS -> TIMESTAMP_US rounds ties away from epoch and preserves ±infinity
    {
      std::vector<std::int64_t> ns_vals = {
        -ts_inf, -1500, -1499, -500, -499, 0, 499, 500, 1499, 1500, ts_inf};
      auto const n_ns = static_cast<cudf::size_type>(ns_vals.size());
      auto col_ns =
        cudf::make_timestamp_column(cudf::data_type{cudf::type_id::TIMESTAMP_NANOSECONDS},
                                    n_ns,
                                    cudf::mask_state::UNALLOCATED,
                                    stream,
                                    mr);
      REQUIRE(cudaMemcpy(col_ns->mutable_view().data<std::int64_t>(),
                         ns_vals.data(),
                         n_ns * sizeof(std::int64_t),
                         cudaMemcpyHostToDevice) == cudaSuccess);
      cudf::table_view tv{{col_ns->view()}};

      auto expr = make_cast(make_ref_typed(0, ts_ns_t), ts_us_t, false);
      exp_executor mat_exec(
        *expr, mr, stream, sirius::expression_query_policy{exp_strategy_enum::MATERIALIZE});
      exp_executor jit_exec(
        *expr, mr, stream, sirius::expression_query_policy{exp_strategy_enum::NVRTC_JIT});
      auto mat_out = mat_exec.evaluate(tv);
      auto jit_out = jit_exec.evaluate(tv);
      CHECK(copy_column_to_host<std::int64_t>(jit_out->view().column(0)) ==
            copy_column_to_host<std::int64_t>(mat_out->view().column(0)));
    }

    // 4. TIMESTAMP_S -> TIMESTAMP_US overflow throws on CAST and returns NULL on TRY_CAST
    {
      std::vector<std::int64_t> s_vals = {-ts_inf, 100, ts_inf / 1000000 + 1, ts_inf};
      auto col_s = cudf::make_timestamp_column(cudf::data_type{cudf::type_id::TIMESTAMP_SECONDS},
                                               4,
                                               cudf::mask_state::UNALLOCATED,
                                               stream,
                                               mr);
      REQUIRE(cudaMemcpy(col_s->mutable_view().data<std::int64_t>(),
                         s_vals.data(),
                         4 * sizeof(std::int64_t),
                         cudaMemcpyHostToDevice) == cudaSuccess);
      cudf::table_view tv{{col_s->view()}};

      auto strict_cast = make_cast(make_ref_typed(0, ts_s_t), ts_us_t, /*try_cast=*/false);
      exp_executor strict_exec(
        *strict_cast, mr, stream, sirius::expression_query_policy{exp_strategy_enum::NVRTC_JIT});
      REQUIRE_THROWS_AS(strict_exec.evaluate(tv), sirius::out_of_range_exception);

      auto try_cast = make_cast(make_ref_typed(0, ts_s_t), ts_us_t, /*try_cast=*/true);
      exp_executor try_exec(
        *try_cast, mr, stream, sirius::expression_query_policy{exp_strategy_enum::NVRTC_JIT});
      auto try_out = try_exec.evaluate(tv);
      CHECK(try_out->view().column(0).null_count() == 1);
      CHECK(copy_valids_to_host(try_out->view().column(0)) ==
            std::vector<bool>{true, true, false, true});
      auto const out_vals = copy_column_to_host<std::int64_t>(try_out->view().column(0));
      CHECK(out_vals[0] == -ts_inf);
      CHECK(out_vals[1] == 100'000'000LL);
      CHECK(out_vals[3] == ts_inf);
    }
  }
}

// ============================================================================
// Ticket #9: In-kernel temporal extraction (year..microsecond) and date_trunc
// ============================================================================

TEST_CASE("nvrtc_jit - temporal extraction and date_trunc match MATERIALIZE and fuse in one kernel",
          "[expression_evaluator][nvrtc_jit][ticket9]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  auto stream = cudf::get_default_stream();
  auto mr     = get_resource_ref(*space);

  auto const date_t   = logical_type::make(type_id::DATE);
  auto const ts_s_t   = logical_type::make(type_id::TIMESTAMP_SEC);
  auto const ts_ms_t  = logical_type::make(type_id::TIMESTAMP_MS);
  auto const ts_us_t  = logical_type::make(type_id::TIMESTAMP);
  auto const ts_ns_t  = logical_type::make(type_id::TIMESTAMP_NS);
  auto const bigint_t = logical_type::make(type_id::BIGINT);
  auto const int_t    = logical_type::make(type_id::INTEGER);
  auto const bool_t   = logical_type::make(type_id::BOOLEAN);

  auto const date_inf = std::numeric_limits<std::int32_t>::max();
  auto const date_min = std::numeric_limits<std::int32_t>::lowest();
  auto const ts_inf   = std::numeric_limits<std::int64_t>::max();
  auto const ts_min   = std::numeric_limits<std::int64_t>::lowest();

  auto make_ts_col = [&]<typename Rep>(cudf::data_type dt,
                                       std::vector<Rep> const& vals,
                                       std::vector<bool> const& valids) {
    auto const n          = static_cast<cudf::size_type>(vals.size());
    auto mask             = cudf::create_null_mask(n, cudf::mask_state::ALL_VALID, stream, mr);
    auto* mptr            = static_cast<cudf::bitmask_type*>(mask.data());
    cudf::size_type nulls = 0;
    for (cudf::size_type i = 0; i < n; ++i) {
      if (!valids[i]) {
        cudf::set_null_mask(mptr, i, i + 1, false, stream);
        ++nulls;
      }
    }
    auto col = cudf::make_timestamp_column(dt, n, std::move(mask), nulls, stream, mr);
    REQUIRE(cudaMemcpy(col->mutable_view().data<Rep>(),
                       vals.data(),
                       n * sizeof(Rep),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    return col;
  };

  auto check_tables_match = [&](cudf::table_view jit_tv, cudf::table_view mat_tv) {
    REQUIRE(jit_tv.num_columns() == mat_tv.num_columns());
    REQUIRE(jit_tv.num_rows() == mat_tv.num_rows());
    auto const n_rows = jit_tv.num_rows();
    for (cudf::size_type c = 0; c < jit_tv.num_columns(); ++c) {
      CAPTURE(c);
      auto const jcol = jit_tv.column(c);
      auto const mcol = mat_tv.column(c);
      REQUIRE(jcol.type() == mcol.type());
      CHECK(jcol.null_count() == mcol.null_count());
      auto const jvalids = copy_valids_to_host(jcol);
      auto const mvalids = copy_valids_to_host(mcol);
      REQUIRE(jvalids == mvalids);
      auto const elem_size = cudf::size_of(jcol.type());
      std::vector<std::uint8_t> jbytes(n_rows * elem_size);
      std::vector<std::uint8_t> mbytes(n_rows * elem_size);
      REQUIRE(cudaMemcpy(jbytes.data(),
                         jcol.head<std::uint8_t>() + jcol.offset() * elem_size,
                         jbytes.size(),
                         cudaMemcpyDeviceToHost) == cudaSuccess);
      REQUIRE(cudaMemcpy(mbytes.data(),
                         mcol.head<std::uint8_t>() + mcol.offset() * elem_size,
                         mbytes.size(),
                         cudaMemcpyDeviceToHost) == cudaSuccess);
      for (cudf::size_type r = 0; r < n_rows; ++r) {
        if (jvalids[r]) {
          CAPTURE(r);
          CHECK(std::memcmp(
                  jbytes.data() + r * elem_size, mbytes.data() + r * elem_size, elem_size) == 0);
        }
      }
    }
  };

  SECTION("datetime extraction (year..microsecond) across DATE and all TIMESTAMP precisions")
  {
    // DATE test values:
    // -25509: 1900-02-28 (non-leap century year)
    // -1:     1969-12-31
    // 0:      1970-01-01
    // 11016:  2000-02-29 (leap century year)
    // 19782:  2024-02-29 (leap year)
    // -365:   1969-01-01
    // plus NULL, ±infinity, and narrowed INT16 DATE carrier
    std::vector<std::int32_t> date_vals = {
      -25509, -365, -1, 0, 11016, 19782, -date_inf, date_inf, 100};
    std::vector<bool> valids = {true, true, true, true, true, true, true, true, false};

    // Sub-day timestamps (9 rows matching valids):
    // Row 0: pre-epoch exact day boundary (-25509 * 86400s = 1900-02-28 00:00:00)
    // Row 1: pre-epoch mid-day (-1 tick before epoch: 1969-12-31 23:59:59...)
    // Row 2: pre-epoch with subseconds (-86400 - 3723.040523s = 1969-12-30 22:57:56.959477)
    // Row 3: epoch (0 = 1970-01-01 00:00:00)
    // Row 4: leap day 2000-02-29 12:34:56.789123456
    // Row 5: leap day 2024-02-29 23:59:59.999999999
    // Row 6: ts_min (minimum finite int64)
    // Row 7: ts_inf (+infinity)
    // Row 8: NULL (-ts_inf payload)
    std::vector<std::int64_t> ts_s_vals  = {-25509LL * 86400LL,
                                            -1LL,
                                            -86400LL - 3723LL,
                                            0LL,
                                            11016LL * 86400LL + 45296LL,
                                            19782LL * 86400LL + 86399LL,
                                            -2000LL * 365LL * 86400LL - 12345LL,
                                            ts_inf,
                                            -ts_inf};
    std::vector<std::int64_t> ts_ms_vals = {-25509LL * 86400'000LL,
                                            -1LL,
                                            -86400'000LL - 3723'040LL,
                                            0LL,
                                            11016LL * 86400'000LL + 45296'789LL,
                                            19782LL * 86400'000LL + 86399'999LL,
                                            -2000LL * 365LL * 86400'000LL - 12345'678LL,
                                            -ts_inf,
                                            ts_inf};
    std::vector<std::int64_t> ts_us_vals = {-25509LL * 86400'000'000LL,
                                            -1LL,
                                            -86400'000'000LL - 3723'040'523LL,
                                            0LL,
                                            11016LL * 86400'000'000LL + 45296'789'123LL,
                                            19782LL * 86400'000'000LL + 86399'999'999LL,
                                            -2000LL * 365LL * 86400'000'000LL - 12345'678'901LL,
                                            ts_inf,
                                            0LL};
    std::vector<std::int64_t> ts_ns_vals = {-25509LL * 86400'000'000'000LL,
                                            -1LL,
                                            -86400'000'000'000LL - 3723'040'523'789LL,
                                            0LL,
                                            11016LL * 86400'000'000'000LL + 45296'789'123'456LL,
                                            19782LL * 86400'000'000'000LL + 86399'999'999'999LL,
                                            ts_min,
                                            -ts_inf,
                                            0LL};

    // Narrowed INT16 DATE carrier (column 5)
    std::vector<std::int16_t> narrow_date_vals = {
      -25509, -365, -1, 0, 11016, 19782, 1000, -1000, 0};
    auto col_narrow_date = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT16}, 9, cudf::mask_state::UNALLOCATED, stream, mr);
    REQUIRE(cudaMemcpy(col_narrow_date->mutable_view().data<std::int16_t>(),
                       narrow_date_vals.data(),
                       9 * sizeof(std::int16_t),
                       cudaMemcpyHostToDevice) == cudaSuccess);

    auto col_d = make_ts_col(cudf::data_type{cudf::type_id::TIMESTAMP_DAYS}, date_vals, valids);
    auto col_s = make_ts_col(cudf::data_type{cudf::type_id::TIMESTAMP_SECONDS}, ts_s_vals, valids);
    auto col_ms =
      make_ts_col(cudf::data_type{cudf::type_id::TIMESTAMP_MILLISECONDS}, ts_ms_vals, valids);
    auto col_us =
      make_ts_col(cudf::data_type{cudf::type_id::TIMESTAMP_MICROSECONDS}, ts_us_vals, valids);
    auto col_ns =
      make_ts_col(cudf::data_type{cudf::type_id::TIMESTAMP_NANOSECONDS}, ts_ns_vals, valids);

    cudf::table_view tv{{col_d->view(),
                         col_s->view(),
                         col_ms->view(),
                         col_us->view(),
                         col_ns->view(),
                         col_narrow_date->view()}};

    std::vector<sirius::function_id> const extract_fids = {sirius::function_id::year,
                                                           sirius::function_id::month,
                                                           sirius::function_id::day,
                                                           sirius::function_id::hour,
                                                           sirius::function_id::minute,
                                                           sirius::function_id::second,
                                                           sirius::function_id::millisecond,
                                                           sirius::function_id::microsecond};

    std::vector<std::pair<std::uint32_t, logical_type>> const col_specs = {
      {0, date_t}, {1, ts_s_t}, {2, ts_ms_t}, {3, ts_us_t}, {4, ts_ns_t}, {5, date_t}};

    duckdb::vector<std::unique_ptr<ast_node>> exprs;
    for (auto const& [col_idx, ltype] : col_specs) {
      for (auto const fid : extract_fids) {
        std::vector<std::unique_ptr<ast_node>> args;
        args.push_back(make_ref_typed(col_idx, ltype));
        exprs.push_back(make_func(fid, std::move(args), bigint_t));
      }
    }

    exp_executor mat_exec(
      exprs, mr, stream, sirius::expression_query_policy{exp_strategy_enum::MATERIALIZE});
    exp_executor jit_exec(
      exprs, mr, stream, sirius::expression_query_policy{exp_strategy_enum::NVRTC_JIT});

    auto mat_out = mat_exec.evaluate(tv);
    auto jit_out = jit_exec.evaluate(tv);
    REQUIRE(mat_out != nullptr);
    REQUIRE(jit_out != nullptr);
    CHECK(jit_exec.restored_reference_cast_count_for_testing() == 0);
    check_tables_match(jit_out->view(), mat_out->view());
  }

  SECTION("date_trunc across all supported units and temporal types matches MATERIALIZE")
  {
    std::vector<std::int32_t> date_vals = {-25509, -1, 0, 11016, 19782, 42};
    std::vector<bool> valids            = {true, true, true, true, true, false};
    std::vector<std::int64_t> ts_s_vals = {
      -86400LL - 3723LL, -1LL, 0LL, 11016LL * 86400LL + 45296LL, 86400LL, 0LL};
    std::vector<std::int64_t> ts_ms_vals = {
      -86400'000LL - 3723'040LL, -1LL, 0LL, 11016LL * 86400'000LL + 45296'789LL, 1234LL, 0LL};
    std::vector<std::int64_t> ts_us_vals = {-86400'000'000LL - 3723'040'523LL,
                                            -1LL,
                                            0LL,
                                            11016LL * 86400'000'000LL + 45296'789'123LL,
                                            -365LL * 86400'000'000LL - 1LL,
                                            0LL};
    std::vector<std::int64_t> ts_ns_vals = {-86400'000'000'000LL - 3723'040'523'789LL,
                                            -1LL,
                                            0LL,
                                            11016LL * 86400'000'000'000LL + 45296'789'123'456LL,
                                            -365LL * 86400'000'000'000LL - 1LL,
                                            0LL};

    auto col_d = make_ts_col(cudf::data_type{cudf::type_id::TIMESTAMP_DAYS}, date_vals, valids);
    auto col_s = make_ts_col(cudf::data_type{cudf::type_id::TIMESTAMP_SECONDS}, ts_s_vals, valids);
    auto col_ms =
      make_ts_col(cudf::data_type{cudf::type_id::TIMESTAMP_MILLISECONDS}, ts_ms_vals, valids);
    auto col_us =
      make_ts_col(cudf::data_type{cudf::type_id::TIMESTAMP_MICROSECONDS}, ts_us_vals, valids);
    auto col_ns =
      make_ts_col(cudf::data_type{cudf::type_id::TIMESTAMP_NANOSECONDS}, ts_ns_vals, valids);

    cudf::table_view tv{
      {col_d->view(), col_s->view(), col_ms->view(), col_us->view(), col_ns->view()}};

    std::vector<std::string> const units = {
      "day", "hour", "minute", "second", "millisecond", "microsecond"};
    std::vector<std::pair<std::uint32_t, logical_type>> const col_specs = {
      {0, date_t}, {1, ts_s_t}, {2, ts_ms_t}, {3, ts_us_t}, {4, ts_ns_t}};

    duckdb::vector<std::unique_ptr<ast_node>> exprs;
    for (auto const& [col_idx, ltype] : col_specs) {
      for (auto const& unit : units) {
        std::vector<std::unique_ptr<ast_node>> args;
        args.push_back(make_str_const(unit));
        args.push_back(make_ref_typed(col_idx, ltype));
        exprs.push_back(make_func(sirius::function_id::date_trunc, std::move(args), ltype));
      }
    }
    // Also test DuckDB's DATE -> TIMESTAMP return type on date_trunc('day', date_col)
    for (auto const& unit : units) {
      std::vector<std::unique_ptr<ast_node>> args;
      args.push_back(make_str_const(unit));
      args.push_back(make_ref_typed(0, date_t));
      exprs.push_back(make_func(sirius::function_id::date_trunc, std::move(args), ts_us_t));
    }

    exp_executor mat_exec(
      exprs, mr, stream, sirius::expression_query_policy{exp_strategy_enum::MATERIALIZE});
    exp_executor jit_exec(
      exprs, mr, stream, sirius::expression_query_policy{exp_strategy_enum::NVRTC_JIT});

    auto mat_out = mat_exec.evaluate(tv);
    auto jit_out = jit_exec.evaluate(tv);
    REQUIRE(mat_out != nullptr);
    REQUIRE(jit_out != nullptr);
    check_tables_match(jit_out->view(), mat_out->view());
  }

  SECTION(
    "compound expressions combining temporal extraction and date_trunc with arithmetic, "
    "comparisons, and CASE")
  {
    std::vector<std::int64_t> ts_us_vals = {
      -86400'000'000LL - 3723'040'523LL,            // 1969-12-30 22:57:56.959477
      0LL,                                          // 1970-01-01 00:00:00.000000
      11016LL * 86400'000'000LL + 45296'789'123LL,  // 2000-02-29 12:34:56.789123
      19782LL * 86400'000'000LL + 86399'999'999LL,  // 2024-02-29 23:59:59.999999
      ts_inf,
      0LL};
    std::vector<bool> valids = {true, true, true, true, true, false};
    auto col_us =
      make_ts_col(cudf::data_type{cudf::type_id::TIMESTAMP_MICROSECONDS}, ts_us_vals, valids);
    cudf::table_view tv{{col_us->view()}};

    auto make_extract = [&](sirius::function_id fid) {
      std::vector<std::unique_ptr<ast_node>> args;
      args.push_back(make_ref_typed(0, ts_us_t));
      return make_func(fid, std::move(args), bigint_t);
    };
    auto make_trunc = [&](std::string const& unit) {
      std::vector<std::unique_ptr<ast_node>> args;
      args.push_back(make_str_const(unit));
      args.push_back(make_ref_typed(0, ts_us_t));
      return make_func(sirius::function_id::date_trunc, std::move(args), ts_us_t);
    };
    auto make_bin_func = [&](sirius::function_id fid,
                             std::unique_ptr<ast_node> lhs,
                             std::unique_ptr<ast_node> rhs,
                             logical_type ret) {
      std::vector<std::unique_ptr<ast_node>> args;
      args.push_back(std::move(lhs));
      args.push_back(std::move(rhs));
      return make_func(fid, std::move(args), ret);
    };

    duckdb::vector<std::unique_ptr<ast_node>> exprs;
    // 1. year(t) * 10000 + month(t) * 100 + day(t)
    auto y_scaled = make_bin_func(sirius::function_id::mul,
                                  make_extract(sirius::function_id::year),
                                  make_bigint_const(10000),
                                  bigint_t);
    auto m_scaled = make_bin_func(sirius::function_id::mul,
                                  make_extract(sirius::function_id::month),
                                  make_bigint_const(100),
                                  bigint_t);
    auto ym =
      make_bin_func(sirius::function_id::add, std::move(y_scaled), std::move(m_scaled), bigint_t);
    exprs.push_back(make_bin_func(
      sirius::function_id::add, std::move(ym), make_extract(sirius::function_id::day), bigint_t));

    // 2. CASE WHEN year(t) >= 2000 THEN millisecond(t) + microsecond(t) ELSE hour(t) * 60 +
    // minute(t) END
    std::vector<sirius::ast::case_expr::when_then> cases;
    cases.push_back({make_cmp(sirius::comparison_type::ge,
                              make_extract(sirius::function_id::year),
                              make_bigint_const(2000)),
                     make_bin_func(sirius::function_id::add,
                                   make_extract(sirius::function_id::millisecond),
                                   make_extract(sirius::function_id::microsecond),
                                   bigint_t)});
    auto hm = make_bin_func(sirius::function_id::add,
                            make_bin_func(sirius::function_id::mul,
                                          make_extract(sirius::function_id::hour),
                                          make_bigint_const(60),
                                          bigint_t),
                            make_extract(sirius::function_id::minute),
                            bigint_t);
    exprs.push_back(std::make_unique<ast_node>(
      sirius::ast::case_expr{std::move(cases), std::move(hm), bigint_t}));

    // 3. date_trunc('hour', t) == date_trunc('day', t)
    exprs.push_back(
      make_cmp(sirius::comparison_type::equal, make_trunc("hour"), make_trunc("day")));

    // 4. hour(date_trunc('day', t)) + minute(date_trunc('hour', t))
    std::vector<std::unique_ptr<ast_node>> h_args;
    h_args.push_back(make_trunc("day"));
    std::vector<std::unique_ptr<ast_node>> min_args;
    min_args.push_back(make_trunc("hour"));
    exprs.push_back(
      make_bin_func(sirius::function_id::add,
                    make_func(sirius::function_id::hour, std::move(h_args), bigint_t),
                    make_func(sirius::function_id::minute, std::move(min_args), bigint_t),
                    bigint_t));

    // Verify NVRTC_JIT compiles and fuses the compound expressions
    std::vector<cudf::column_view> extra_views = {col_us->view()};
    auto narrow_date_col                       = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT16}, 6, cudf::mask_state::UNALLOCATED, stream, mr);
    std::vector<std::int16_t> nd_vals = {-1, 0, 11016, 19782, 100, 0};
    REQUIRE(cudaMemcpy(narrow_date_col->mutable_view().data<std::int16_t>(),
                       nd_vals.data(),
                       6 * sizeof(std::int16_t),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    cudf::table_view tv_with_narrow{{col_us->view(), narrow_date_col->view()}};
    std::vector<std::unique_ptr<ast_node>> y_narrow_args;
    y_narrow_args.push_back(make_ref_typed(1, date_t));
    exprs.push_back(make_func(sirius::function_id::year, std::move(y_narrow_args), bigint_t));

    exp_executor mat_exec(
      exprs, mr, stream, sirius::expression_query_policy{exp_strategy_enum::MATERIALIZE});
    exp_executor jit_exec(
      exprs, mr, stream, sirius::expression_query_policy{exp_strategy_enum::NVRTC_JIT});

    auto mat_out = mat_exec.evaluate(tv_with_narrow);
    auto jit_out = jit_exec.evaluate(tv_with_narrow);
    REQUIRE(mat_out != nullptr);
    REQUIRE(jit_out != nullptr);
    CHECK(jit_exec.restored_reference_cast_count_for_testing() == 0);
    check_tables_match(jit_out->view(), mat_out->view());
  }
}
