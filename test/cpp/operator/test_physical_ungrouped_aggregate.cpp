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

#include "operator_test_utils.hpp"
#include "operator_type_traits.hpp"

#include <catch.hpp>
#include <duckdb/planner/expression/bound_aggregate_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <op/sirius_physical_ungrouped_aggregate.hpp>

using namespace duckdb;
using namespace sirius::op;
using namespace cucascade;
using namespace cucascade::memory;
using namespace sirius::test::operator_utils;

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

  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  data_repository_mgr repo_mgr;

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
    b1 = make_decimal64_batch(repo_mgr, *space, batch1_vals, Traits::scale);
    b2 = make_decimal64_batch(repo_mgr, *space, batch2_vals, Traits::scale);
  } else {
    b1 =
      make_numeric_batch<typename Traits::type>(repo_mgr, *space, batch1_vals, Traits::cudf_type);
    b2 =
      make_numeric_batch<typename Traits::type>(repo_mgr, *space, batch2_vals, Traits::cudf_type);
  }

  // 1. SUM(col0)
  // 2. MIN(col0)
  // 3. MAX(col0)
  // 4. COUNT(col0)
  // 5. COUNT(*)

  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> aggregates;
  duckdb::vector<duckdb::LogicalType> ret_types;

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
      MakeDummyAggregate("count", {Traits::logical_type()}, LogicalType::BIGINT),
      std::move(children),
      nullptr,
      nullptr,
      AggregateType::NON_DISTINCT));
    ret_types.push_back(LogicalType::BIGINT);
  }

  // COUNT_STAR
  {
    aggregates.push_back(
      make_uniq<BoundAggregateExpression>(MakeDummyAggregate("count_star", {}, LogicalType::BIGINT),
                                          duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>{},
                                          nullptr,
                                          nullptr,
                                          AggregateType::NON_DISTINCT));
    ret_types.push_back(LogicalType::BIGINT);
  }

  sirius_physical_ungrouped_aggregate agg_op(std::move(ret_types),
                                             std::move(aggregates),
                                             0,
                                             duckdb::TupleDataValidityType::CANNOT_HAVE_NULL_VALUES,
                                             &repo_mgr);

  // Execute with multiple batches to test state accumulation
  auto out = agg_op.execute({b1, b2});
  REQUIRE(out.size() == 1);

  auto table = out[0]->get_data()->template cast<gpu_table_representation>().get_table();
  auto view  = table.view();

  REQUIRE(view.num_columns() == 5);
  REQUIRE(view.num_rows() == 1);

  // Verify
  auto sum_out        = copy_column_to_host<typename Traits::type>(view.column(0));
  auto min_out        = copy_column_to_host<typename Traits::type>(view.column(1));
  auto max_out        = copy_column_to_host<typename Traits::type>(view.column(2));
  auto count_out      = copy_column_to_host<int64_t>(view.column(3));
  auto count_star_out = copy_column_to_host<int64_t>(view.column(4));

  typename Traits::type expected_sum = 0;
  typename Traits::type expected_min = vals[0];
  typename Traits::type expected_max = vals[0];

  for (auto v : vals) {
    expected_sum += v;
    if (v < expected_min) expected_min = v;
    if (v > expected_max) expected_max = v;
  }

  // Approximate check for floats
  if constexpr (std::is_floating_point_v<typename Traits::type>) {
    REQUIRE(sum_out[0] == Approx(expected_sum));
    REQUIRE(min_out[0] == Approx(expected_min));
    REQUIRE(max_out[0] == Approx(expected_max));
  } else {
    REQUIRE(sum_out[0] == expected_sum);
    REQUIRE(min_out[0] == expected_min);
    REQUIRE(max_out[0] == expected_max);
  }

  REQUIRE(count_out[0] == 4);
  REQUIRE(count_star_out[0] == 4);
}
