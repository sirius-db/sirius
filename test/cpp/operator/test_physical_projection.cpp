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
#include <duckdb/common/types/decimal.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <op/sirius_physical_projection.hpp>

using namespace duckdb;
using namespace sirius::op;
using namespace cucascade;
using namespace cucascade::memory;

namespace {

using namespace sirius::test::operator_utils;
}  // namespace

TEMPLATE_TEST_CASE("sirius_physical_projection executes on data_batch for multiple types",
                   "[physical_projection]",
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

  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  data_repository_mgr repo_mgr;

  std::vector<int64_t> key_vals{10, 20, 30, 40};
  auto data_vals = Traits::sample_values();
  while (data_vals.size() < key_vals.size()) {
    data_vals.push_back(data_vals.back());
  }
  data_vals.resize(key_vals.size());

  std::shared_ptr<data_batch> input_batch;
  if constexpr (Traits::is_string) {
    input_batch = make_two_column_batch<int64_t, typename Traits::type>(
      repo_mgr, *space, key_vals, data_vals, Traits::cudf_type, std::nullopt);
  } else if constexpr (Traits::is_decimal) {
    input_batch = make_two_column_batch<int64_t, typename Traits::type>(
      repo_mgr, *space, key_vals, data_vals, Traits::cudf_type, Traits::scale);
  } else if constexpr (Traits::is_ts) {
    input_batch = make_two_column_batch<int64_t, typename Traits::type>(
      repo_mgr, *space, key_vals, data_vals, Traits::cudf_type, std::nullopt);
  } else {
    input_batch = make_two_column_batch<int64_t, typename Traits::type>(
      repo_mgr, *space, key_vals, data_vals, Traits::cudf_type, std::nullopt);
  }

  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> exprs;
  exprs.push_back(
    duckdb::make_uniq<BoundReferenceExpression>(Traits::logical_type(), 1));  // data column first
  exprs.push_back(duckdb::make_uniq<BoundReferenceExpression>(
    duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT), 0));  // key column second

  duckdb::vector<duckdb::LogicalType> types;
  types.push_back(Traits::logical_type());
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));

  sirius_physical_projection projection(
    std::move(types), std::move(exprs), key_vals.size(), &repo_mgr);

  auto outputs = projection.execute({input_batch});
  REQUIRE(outputs.size() == 1);
  auto output_table = outputs[0]->get_data()->cast<gpu_table_representation>().get_table();
  auto out_view     = output_table.view();

  auto host_data = copy_column_to_host<typename Traits::type>(out_view.column(0));
  auto host_keys = copy_column_to_host<int64_t>(out_view.column(1));

  auto logical          = Traits::logical_type();
  constexpr bool is_dec = Traits::is_decimal || std::is_same_v<TestType, decimal64_tag>;
  if constexpr (is_dec) {
    auto scale  = duckdb::DecimalType::GetScale(logical);
    int64_t mul = 1;
    for (int i = 0; i < scale; ++i) {
      mul *= 10;
    }

    std::vector<int64_t> expected_scaled;
    expected_scaled.reserve(data_vals.size());
    for (auto v : data_vals) {
      expected_scaled.push_back(static_cast<int64_t>(v) * mul);
    }
    REQUIRE(host_data == expected_scaled);
  } else {
    REQUIRE(host_data == data_vals);
  }
  REQUIRE(host_keys == key_vals);
}

TEMPLATE_TEST_CASE("sirius_physical_projection can drop columns",
                   "[physical_projection][subset]",
                   int64_t)
{
  using Traits = gpu_type_traits<TestType>;

  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  data_repository_mgr repo_mgr;

  std::vector<int64_t> key_vals{10, 20, 30};
  auto data_vals = Traits::sample_values();
  while (data_vals.size() < key_vals.size()) {
    data_vals.push_back(data_vals.back());
  }
  data_vals.resize(key_vals.size());

  auto input_batch = make_two_column_batch<int64_t, typename Traits::type>(
    repo_mgr, *space, key_vals, data_vals, Traits::cudf_type, std::nullopt);

  // Select only the data column
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> exprs;
  exprs.push_back(duckdb::make_uniq<BoundReferenceExpression>(Traits::logical_type(), 1));

  duckdb::vector<duckdb::LogicalType> types;
  types.push_back(Traits::logical_type());

  sirius_physical_projection projection(
    std::move(types), std::move(exprs), key_vals.size(), &repo_mgr);

  auto outputs = projection.execute({input_batch});
  REQUIRE(outputs.size() == 1);
  auto output_table = outputs[0]->get_data()->template cast<gpu_table_representation>().get_table();
  auto out_view     = output_table.view();

  auto host_data = copy_column_to_host<typename Traits::type>(out_view.column(0));
  REQUIRE(host_data == data_vals);
}

TEMPLATE_TEST_CASE("sirius_physical_projection can duplicate/reorder columns",
                   "[physical_projection][reorder]",
                   int64_t)
{
  using Traits = gpu_type_traits<TestType>;

  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  data_repository_mgr repo_mgr;

  std::vector<int64_t> key_vals{100, 200, 300};
  auto data_vals = Traits::sample_values();
  while (data_vals.size() < key_vals.size()) {
    data_vals.push_back(data_vals.back());
  }
  data_vals.resize(key_vals.size());

  auto input_batch = make_two_column_batch<int64_t, typename Traits::type>(
    repo_mgr, *space, key_vals, data_vals, Traits::cudf_type, std::nullopt);

  // Output order: key, data, key (duplicate)
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> exprs;
  exprs.push_back(duckdb::make_uniq<BoundReferenceExpression>(
    duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT), 0));
  exprs.push_back(duckdb::make_uniq<BoundReferenceExpression>(Traits::logical_type(), 1));
  exprs.push_back(duckdb::make_uniq<BoundReferenceExpression>(
    duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT), 0));

  duckdb::vector<duckdb::LogicalType> types;
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));
  types.push_back(Traits::logical_type());
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));

  sirius_physical_projection projection(
    std::move(types), std::move(exprs), key_vals.size(), &repo_mgr);

  auto outputs = projection.execute({input_batch});
  REQUIRE(outputs.size() == 1);
  auto output_table = outputs[0]->get_data()->template cast<gpu_table_representation>().get_table();
  auto out_view     = output_table.view();

  auto host_key0 = copy_column_to_host<int64_t>(out_view.column(0));
  auto host_data = copy_column_to_host<typename Traits::type>(out_view.column(1));
  auto host_key1 = copy_column_to_host<int64_t>(out_view.column(2));

  REQUIRE(host_key0 == key_vals);
  REQUIRE(host_key1 == key_vals);
  REQUIRE(host_data == data_vals);
}
