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

#include "memory/sirius_memory_reservation_manager.hpp"
#include "operator_test_utils.hpp"
#include "operator_type_traits.hpp"

#include <catch.hpp>
#include <cucascade/data/gpu_data_representation.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <duckdb/common/types/date.hpp>
#include <duckdb/common/types/timestamp.hpp>
#include <duckdb/planner/expression/bound_comparison_expression.hpp>
#include <duckdb/planner/expression/bound_constant_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <op/sirius_physical_filter.hpp>

#include <variant>

using namespace duckdb;
using namespace sirius::op;
using namespace cucascade;
using namespace cucascade::memory;

namespace {

using namespace sirius::test::operator_utils;
}  // namespace

TEMPLATE_TEST_CASE("sirius_physical_filter executes on data_batch for multiple numeric types",
                   "[physical_filter]",
                   int32_t,
                   int64_t,
                   float,
                   double,
                   int16_t,
                   bool,
                   decimal64_tag,
                   string_tag,
                   timestamp_us_tag,
                   date32_tag)
{
  using Traits = gpu_type_traits<TestType>;

  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);

  // Build filter column (int64) and data column (TestType)
  std::vector<int64_t> filter_vals{1, 2, 3, 5, 7};
  auto data_vals = Traits::sample_values();
  // align lengths
  while (data_vals.size() < filter_vals.size()) {
    data_vals.push_back(data_vals.back());
  }
  data_vals.resize(filter_vals.size());

  std::shared_ptr<data_batch> input_batch;
  if constexpr (Traits::is_string) {
    input_batch = make_two_column_batch<int64_t, typename Traits::type>(
      *space, filter_vals, data_vals, Traits::cudf_type, std::nullopt);
  } else if constexpr (Traits::is_decimal) {
    input_batch = make_two_column_batch<int64_t, typename Traits::type>(
      *space, filter_vals, data_vals, Traits::cudf_type, Traits::scale, cudf::type_id::INT64);
  } else if constexpr (Traits::is_ts) {
    input_batch = make_two_column_batch<int64_t, typename Traits::type>(
      *space, filter_vals, data_vals, Traits::cudf_type, std::nullopt);
  } else {
    input_batch = make_two_column_batch<int64_t, typename Traits::type>(
      *space, filter_vals, data_vals, Traits::cudf_type, std::nullopt);
  }

  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> exprs;
  exprs.push_back(duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_GREATERTHAN,
    duckdb::make_uniq<BoundReferenceExpression>(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT),
                                                0),
    duckdb::make_uniq<BoundConstantExpression>(duckdb::Value::BIGINT(3))));

  duckdb::vector<duckdb::LogicalType> types;
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));  // filter column
  types.push_back(Traits::logical_type());

  sirius_physical_filter filter(std::move(types), std::move(exprs), filter_vals.size());

  std::vector<std::shared_ptr<cucascade::data_batch>> inputs{input_batch};
  auto outputs = filter.execute(inputs, cudf::get_default_stream());
  REQUIRE(outputs.size() == 1);
  auto output_table = outputs[0]->get_data()->cast<gpu_table_representation>().get_table();
  auto out_view     = output_table.view();
  auto host_vals    = copy_column_to_host<typename Traits::type>(out_view.column(1));
  auto host_filter  = copy_column_to_host<int64_t>(out_view.column(0));

  std::vector<typename Traits::type> expected_data;
  std::vector<int64_t> expected_filter;
  for (size_t i = 0; i < filter_vals.size(); ++i) {
    if (filter_vals[i] > 3) {
      expected_data.push_back(data_vals[i]);
      expected_filter.push_back(filter_vals[i]);
    }
  }

  REQUIRE(host_filter == expected_filter);
  REQUIRE(host_vals == expected_data);
}
