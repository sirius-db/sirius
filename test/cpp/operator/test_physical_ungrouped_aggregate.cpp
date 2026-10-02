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

#include "aggregate/aggregate_test_utils.hpp"
#include "helper/type_conversions.hpp"
#include "op/aggregate/stddev.hpp"
#include "operator_test_utils.hpp"
#include "operator_type_traits.hpp"

#include <catch.hpp>
#include <duckdb/planner/expression/bound_aggregate_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <expression/ast/from_duckdb.hpp>
#include <expression/ast/node.hpp>
#include <op/sirius_physical_ungrouped_aggregate.hpp>
#include <op/sirius_physical_ungrouped_aggregate_merge.hpp>

#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>

using namespace duckdb;
using namespace sirius::op;
using namespace cucascade;
using namespace cucascade::memory;
using namespace sirius::test::operator_utils;

namespace {
inline uint64_t int128_low64(__int128_t value)
{
  return static_cast<uint64_t>(static_cast<unsigned __int128>(value));
}

inline int64_t int128_high64(__int128_t value)
{
  return static_cast<int64_t>(static_cast<unsigned __int128>(value) >> 64);
}

// Translate a vector of DuckDB expressions into Sirius AST nodes (size/order
// preserved, null slot for an unsupported shape).
inline duckdb::vector<std::unique_ptr<sirius::ast::node>> translate_expressions(
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> exprs)
{
  duckdb::vector<std::unique_ptr<sirius::ast::node>> out;
  out.reserve(exprs.size());
  for (auto& e : exprs) {
    out.push_back(e ? sirius::ast::from_duckdb(*e) : nullptr);
  }
  return out;
}
}  // namespace

// Helper to create a dummy AggregateFunction since we only need the name and types for the GPU
// operator
AggregateFunction MakeDummyAggregate(const std::string& name,
                                     const duckdb::vector<LogicalType>& args,
                                     const LogicalType& ret_type)
{
  return AggregateFunction(
    name, args, ret_type, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
}

TEMPLATE_TEST_CASE("sirius_physical_ungrouped_aggregate computes SUM/MIN/MAX/COUNT",
                   "[physical_ungrouped_aggregate]",
                   int32_t,
                   int64_t,
                   float,
                   double,
                   decimal64_tag)
{
  using Traits = gpu_type_traits<TestType>;

  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);

  // Create values for batches
  auto vals = Traits::sample_values();
  // Ensure we have at least 4 values to split across 2 batches
  while (vals.size() < 4) {
    vals.insert(vals.end(), vals.begin(), vals.end());
  }
  if (vals.size() > 4) { vals.resize(4); }

  std::vector<typename Traits::type> batch1_vals(vals.begin(), vals.begin() + 2);
  std::vector<typename Traits::type> batch2_vals(vals.begin() + 2, vals.begin() + 4);

  std::shared_ptr<data_batch> b1, b2;

  if constexpr (Traits::is_decimal) {
    b1 = make_decimal64_batch(*space, batch1_vals, Traits::scale);
    b2 = make_decimal64_batch(*space, batch2_vals, Traits::scale);
  } else {
    b1 = make_numeric_batch<typename Traits::type>(*space, batch1_vals, Traits::cudf_type);
    b2 = make_numeric_batch<typename Traits::type>(*space, batch2_vals, Traits::cudf_type);
  }

  // 1. SUM(col0)
  // 2. MIN(col0)
  // 3. MAX(col0)
  // 4. COUNT(col0)
  // 5. COUNT(*)

  auto make_aggregates = [&](duckdb::vector<duckdb::LogicalType>& ret_types) {
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> aggregates;

    // SUM
    {
      duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> children;
      children.push_back(make_uniq<BoundReferenceExpression>(Traits::logical_type(), 0));
      aggregates.push_back(make_uniq<BoundAggregateExpression>(
        MakeDummyAggregate("sum", {Traits::logical_type()}, Traits::logical_type()),
        std::move(children),
        nullptr,
        nullptr,
        AggregateType::NON_DISTINCT));
      ret_types.push_back(Traits::logical_type());
    }

    // MIN
    {
      duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> children;
      children.push_back(make_uniq<BoundReferenceExpression>(Traits::logical_type(), 0));
      aggregates.push_back(make_uniq<BoundAggregateExpression>(
        MakeDummyAggregate("min", {Traits::logical_type()}, Traits::logical_type()),
        std::move(children),
        nullptr,
        nullptr,
        AggregateType::NON_DISTINCT));
      ret_types.push_back(Traits::logical_type());
    }

    // MAX
    {
      duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> children;
      children.push_back(make_uniq<BoundReferenceExpression>(Traits::logical_type(), 0));
      aggregates.push_back(make_uniq<BoundAggregateExpression>(
        MakeDummyAggregate("max", {Traits::logical_type()}, Traits::logical_type()),
        std::move(children),
        nullptr,
        nullptr,
        AggregateType::NON_DISTINCT));
      ret_types.push_back(Traits::logical_type());
    }

    // COUNT
    {
      duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> children;
      children.push_back(make_uniq<BoundReferenceExpression>(Traits::logical_type(), 0));
      aggregates.push_back(make_uniq<BoundAggregateExpression>(
        MakeDummyAggregate(
          "count", {Traits::logical_type()}, LogicalType(duckdb::LogicalTypeId::BIGINT)),
        std::move(children),
        nullptr,
        nullptr,
        AggregateType::NON_DISTINCT));
      ret_types.push_back(LogicalType(duckdb::LogicalTypeId::BIGINT));
    }

    // COUNT_STAR
    {
      aggregates.push_back(make_uniq<BoundAggregateExpression>(
        MakeDummyAggregate("count_star", {}, LogicalType(duckdb::LogicalTypeId::BIGINT)),
        duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>{},
        nullptr,
        nullptr,
        AggregateType::NON_DISTINCT));
      ret_types.push_back(LogicalType(duckdb::LogicalTypeId::BIGINT));
    }

    return aggregates;
  };

  duckdb::vector<duckdb::LogicalType> local_types;
  auto local_aggregates = make_aggregates(local_types);
  duckdb::vector<duckdb::LogicalType> merge_types;
  auto merge_aggregates = make_aggregates(merge_types);

  sirius_physical_ungrouped_aggregate local_op(
    sirius::from_duckdb_vec(local_types),
    translate_expressions(std::move(local_aggregates)),
    0,
    duckdb::TupleDataValidityType::CANNOT_HAVE_NULL_VALUES);
  sirius_physical_ungrouped_aggregate_merge merge_op(
    sirius::from_duckdb_vec(merge_types),
    translate_expressions(std::move(merge_aggregates)),
    0,
    duckdb::TupleDataValidityType::CANNOT_HAVE_NULL_VALUES);

  auto local_out1 = local_op.execute(pipelineable_operator_data({b1}), cudf::get_default_stream());
  auto local_out2 = local_op.execute(pipelineable_operator_data({b2}), cudf::get_default_stream());
  auto local_out1_batches =
    dynamic_cast<const pipelineable_operator_data&>(*local_out1).get_data_batches();
  auto local_out2_batches =
    dynamic_cast<const pipelineable_operator_data&>(*local_out2).get_data_batches();
  std::vector<std::shared_ptr<data_batch>> merge_inputs;
  merge_inputs.insert(merge_inputs.end(),
                      std::make_move_iterator(local_out1_batches.begin()),
                      std::make_move_iterator(local_out1_batches.end()));
  merge_inputs.insert(merge_inputs.end(),
                      std::make_move_iterator(local_out2_batches.begin()),
                      std::make_move_iterator(local_out2_batches.end()));
  auto out = merge_op.execute(pipelineable_operator_data(merge_inputs), cudf::get_default_stream());
  REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*out).get_data_batches().size() == 1);

  auto view = sirius::get_cudf_table_view(
    *dynamic_cast<const pipelineable_operator_data&>(*out).get_data_batches()[0]);

  REQUIRE(view.num_columns() == 5);
  REQUIRE(view.num_rows() == 1);

  // Verify
  // SUM may widen the output type (e.g. DECIMAL64 -> DECIMAL128), so read with agg_output_type.
  // MIN/MAX are not widened for decimals, so read with min_max_output_type (= type for decimals).
  auto sum_out        = copy_column_to_host<typename Traits::agg_output_type>(view.column(0));
  auto min_out        = copy_column_to_host<typename Traits::min_max_output_type>(view.column(1));
  auto max_out        = copy_column_to_host<typename Traits::min_max_output_type>(view.column(2));
  auto count_out      = copy_column_to_host<int64_t>(view.column(3));
  auto count_star_out = copy_column_to_host<int64_t>(view.column(4));

  // Compute expected SUM in agg_output_type (DECIMAL64 SUM upcasts to DECIMAL128).
  // Compute expected MIN/MAX in min_max_output_type (DECIMAL64 stays DECIMAL64).
  typename Traits::agg_output_type expected_sum = 0;
  typename Traits::min_max_output_type expected_min =
    static_cast<typename Traits::min_max_output_type>(vals[0]);
  typename Traits::min_max_output_type expected_max =
    static_cast<typename Traits::min_max_output_type>(vals[0]);

  for (auto v : vals) {
    expected_sum += static_cast<typename Traits::agg_output_type>(v);
    auto v_mm = static_cast<typename Traits::min_max_output_type>(v);
    if (v_mm < expected_min) expected_min = v_mm;
    if (v_mm > expected_max) expected_max = v_mm;
  }

  // Approximate check for floats
  if constexpr (std::is_floating_point_v<typename Traits::type>) {
    REQUIRE(sum_out[0] == Approx(static_cast<typename Traits::type>(expected_sum)));
    REQUIRE(min_out[0] == Approx(static_cast<typename Traits::type>(expected_min)));
    REQUIRE(max_out[0] == Approx(static_cast<typename Traits::type>(expected_max)));
  } else if constexpr (std::is_same_v<typename Traits::agg_output_type, __int128_t>) {
    // SUM output is DECIMAL128 (__int128_t).
    REQUIRE(int128_high64(sum_out[0]) == int128_high64(expected_sum));
    REQUIRE(int128_low64(sum_out[0]) == int128_low64(expected_sum));
    // MIN/MAX output is DECIMAL64 (int64_t) — not widened.
    REQUIRE(min_out[0] == expected_min);
    REQUIRE(max_out[0] == expected_max);
  } else {
    REQUIRE(sum_out[0] == expected_sum);
    REQUIRE(min_out[0] == expected_min);
    REQUIRE(max_out[0] == expected_max);
  }

  REQUIRE(count_out[0] == 4);
  REQUIRE(count_star_out[0] == 4);
}

TEMPLATE_TEST_CASE("sirius_physical_ungrouped_aggregate resolves AVG in merge",
                   "[physical_ungrouped_aggregate]",
                   int32_t,
                   int64_t,
                   float,
                   double,
                   decimal64_tag)
{
  using Traits = gpu_type_traits<TestType>;

  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);

  auto vals = Traits::sample_values();
  while (vals.size() < 4) {
    vals.insert(vals.end(), vals.begin(), vals.end());
  }
  if (vals.size() > 4) { vals.resize(4); }

  std::vector<typename Traits::type> batch1_vals(vals.begin(), vals.begin() + 2);
  std::vector<typename Traits::type> batch2_vals(vals.begin() + 2, vals.begin() + 4);

  std::shared_ptr<data_batch> b1, b2;
  if constexpr (Traits::is_decimal) {
    b1 = make_decimal64_batch(*space, batch1_vals, Traits::scale);
    b2 = make_decimal64_batch(*space, batch2_vals, Traits::scale);
  } else {
    b1 = make_numeric_batch<typename Traits::type>(*space, batch1_vals, Traits::cudf_type);
    b2 = make_numeric_batch<typename Traits::type>(*space, batch2_vals, Traits::cudf_type);
  }

  auto make_avg_aggregates = [&](duckdb::vector<duckdb::LogicalType>& ret_types) {
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> aggregates;
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> children;
    children.push_back(make_uniq<BoundReferenceExpression>(Traits::logical_type(), 0));
    auto return_type =
      Traits::is_decimal ? Traits::logical_type() : LogicalType(duckdb::LogicalTypeId::DOUBLE);
    aggregates.push_back(make_uniq<BoundAggregateExpression>(
      MakeDummyAggregate("avg", {Traits::logical_type()}, return_type),
      std::move(children),
      nullptr,
      nullptr,
      AggregateType::NON_DISTINCT));
    ret_types.push_back(return_type);
    return aggregates;
  };

  duckdb::vector<duckdb::LogicalType> local_types;
  auto local_aggregates = make_avg_aggregates(local_types);
  local_types.push_back(LogicalType(duckdb::LogicalTypeId::BIGINT));
  duckdb::vector<duckdb::LogicalType> merge_types;
  auto merge_aggregates = make_avg_aggregates(merge_types);

  sirius_physical_ungrouped_aggregate local_op(
    sirius::from_duckdb_vec(local_types),
    translate_expressions(std::move(local_aggregates)),
    0,
    duckdb::TupleDataValidityType::CANNOT_HAVE_NULL_VALUES);
  sirius_physical_ungrouped_aggregate_merge merge_op(
    sirius::from_duckdb_vec(merge_types),
    translate_expressions(std::move(merge_aggregates)),
    0,
    duckdb::TupleDataValidityType::CANNOT_HAVE_NULL_VALUES);

  auto local_out1 = local_op.execute(pipelineable_operator_data({b1}), cudf::get_default_stream());
  auto local_out2 = local_op.execute(pipelineable_operator_data({b2}), cudf::get_default_stream());
  auto local_out1_batches =
    dynamic_cast<const pipelineable_operator_data&>(*local_out1).get_data_batches();
  auto local_out2_batches =
    dynamic_cast<const pipelineable_operator_data&>(*local_out2).get_data_batches();
  std::vector<std::shared_ptr<data_batch>> merge_inputs;
  merge_inputs.insert(merge_inputs.end(),
                      std::make_move_iterator(local_out1_batches.begin()),
                      std::make_move_iterator(local_out1_batches.end()));
  merge_inputs.insert(merge_inputs.end(),
                      std::make_move_iterator(local_out2_batches.begin()),
                      std::make_move_iterator(local_out2_batches.end()));

  auto out = merge_op.execute(pipelineable_operator_data(merge_inputs), cudf::get_default_stream());
  REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*out).get_data_batches().size() == 1);

  auto view = sirius::get_cudf_table_view(
    *dynamic_cast<const pipelineable_operator_data&>(*out).get_data_batches()[0]);
  REQUIRE(view.num_columns() == 1);
  REQUIRE(view.num_rows() == 1);

  if constexpr (Traits::is_decimal) {
    REQUIRE(view.column(0).type() == cudf::data_type{cudf::type_id::DECIMAL64, Traits::scale});
    auto avg_out                       = copy_column_to_host<typename Traits::type>(view.column(0));
    typename Traits::type expected_sum = 0;
    for (auto v : vals) {
      expected_sum += v;
    }
    // The sample values divide exactly in their fixed-point representation.
    REQUIRE(avg_out[0] == expected_sum / static_cast<typename Traits::type>(vals.size()));
  } else {
    auto avg_out        = copy_column_to_host<double>(view.column(0));
    double expected_sum = 0.0;
    for (auto v : vals) {
      expected_sum += static_cast<double>(v);
    }
    double expected_avg = expected_sum / static_cast<double>(vals.size());
    REQUIRE(avg_out[0] == Approx(expected_avg));
  }
}

TEST_CASE("stddev_samp ungrouped partials merge before finalization",
          "[stddev_samp][physical_ungrouped_aggregate]")
{
  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space  = manager->get_memory_space(Tier::GPU, 0);
  REQUIRE(space);
  auto defs = sirius::test::create_aggregate_expressions<gpu_type_traits<double>>(
    {}, {"stddev_samp", "avg"}, {0, 0});
  sirius_physical_ungrouped_aggregate local(std::move(defs.output_types),
                                            std::move(defs.aggregates),
                                            1,
                                            duckdb::TupleDataValidityType::CAN_HAVE_NULL_VALUES);
  sirius_physical_ungrouped_aggregate_merge merge(&local);
  REQUIRE(local.get_local_output_types().size() == 3);
  REQUIRE(local.get_local_output_types()[0].id() == sirius::type_id::STRUCT);
  std::vector<std::shared_ptr<data_batch>> batches;
  bool const multiple_values_per_partial = GENERATE(false, true);
  for (int i = 1; i <= 4; ++i) {
    std::vector<double> values{1e12 + i};
    if (multiple_values_per_partial) { values.push_back(1e12 + i + 4); }
    batches.push_back(make_numeric_batch<double>(*space, values, cudf::type_id::FLOAT64));
  }
  batches.push_back(
    make_numeric_batch_with_nulls<double>(*space, {0}, {false}, cudf::type_id::FLOAT64));
  batches.push_back(make_numeric_batch<double>(*space, {}, cudf::type_id::FLOAT64));
  auto partial        = local.execute(pipelineable_operator_data(batches), default_stream());
  auto result         = merge.execute(*partial, default_stream());
  auto const& outputs = dynamic_cast<pipelineable_operator_data&>(*result).get_data_batches();
  REQUIRE(outputs.size() == 1);
  auto ro   = outputs[0]->to_read_only();
  auto view = ro.get_data()->cast<gpu_table_representation>().get_table_view();
  REQUIRE(view.column(0).null_count() == 0);
  REQUIRE(copy_column_to_host<double>(view.column(0))[0] ==
          Approx(std::sqrt(multiple_values_per_partial ? 6.0 : 5.0 / 3.0)).margin(1e-9));
  REQUIRE(copy_column_to_host<double>(view.column(1))[0] ==
          Approx(1e12 + (multiple_values_per_partial ? 4.5 : 2.5)));
}

TEST_CASE("stddev_samp rejects non-finite results after checking sample size", "[stddev_samp]")
{
  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space  = manager->get_memory_space(Tier::GPU, 0);
  REQUIRE(space);
  auto stream = default_stream();
  auto mr     = get_resource_ref(*space);
  for (auto value :
       {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
    for (int count : {1, 2}) {
      auto batch = make_numeric_batch<double>(
        *space, std::vector<double>(count, value), cudf::type_id::FLOAT64);
      auto ro    = batch->to_read_only();
      auto input = ro.get_data()->cast<gpu_table_representation>().get_table_view().column(0);
      auto state = local_stddev_state(input, stream, mr);
      if (count == 1) {
        auto result = finalize_stddev(state->view(), stream, mr);
        REQUIRE(result->null_count() == 1);
      } else {
        REQUIRE_THROWS_WITH(finalize_stddev(state->view(), stream, mr),
                            Catch::Matchers::ContainsSubstring("STDDEV_SAMP is out of range"));
      }
    }
  }
}
