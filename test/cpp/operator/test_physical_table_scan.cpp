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
#include <duckdb/function/table_function.hpp>
#include <duckdb/planner/filter/constant_filter.hpp>
#include <duckdb/planner/table_filter.hpp>
#include <op/sirius_physical_table_scan.hpp>

#include <iostream>
#include <variant>

using namespace duckdb;
using namespace sirius::op;
using namespace cucascade;
using namespace cucascade::memory;

namespace {

using namespace sirius::test::operator_utils;
}  // namespace

TEMPLATE_TEST_CASE(
  "sirius_physical_table_scan applies filters on data_batch for multiple numeric types",
  "[physical_table_scan]",
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

  // Create table filter set with a filter on first column (filter_vals > 3)
  auto table_filters   = duckdb::make_uniq<duckdb::TableFilterSet>();
  auto constant_filter = duckdb::make_uniq<duckdb::ConstantFilter>(
    duckdb::ExpressionType::COMPARE_GREATERTHAN, duckdb::Value::BIGINT(3));
  table_filters->PushFilter(duckdb::ColumnIndex(0), std::move(constant_filter));

  // Setup types for the table scan
  duckdb::vector<duckdb::LogicalType> types;
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));  // filter column
  types.push_back(Traits::logical_type());                              // data column

  duckdb::vector<duckdb::LogicalType> returned_types = types;

  duckdb::vector<duckdb::ColumnIndex> column_ids;
  column_ids.push_back(duckdb::ColumnIndex(0));
  column_ids.push_back(duckdb::ColumnIndex(1));

  duckdb::vector<duckdb::idx_t> projection_ids;
  projection_ids.push_back(0);
  projection_ids.push_back(1);

  duckdb::vector<std::string> names{"filter_col", "data_col"};
  duckdb::vector<duckdb::Value> parameters;
  duckdb::virtual_column_map_t virtual_columns;

  // Create a minimal table function (not used in this test but required by constructor)
  duckdb::TableFunction table_function("test_scan", {}, nullptr, nullptr);

  sirius_physical_table_scan table_scan(std::move(types),
                                        std::move(table_function),
                                        nullptr,  // bind_data
                                        std::move(returned_types),
                                        std::move(column_ids),
                                        std::move(projection_ids),
                                        std::move(names),
                                        std::move(table_filters),
                                        filter_vals.size(),
                                        duckdb::ExtraOperatorInfo(),
                                        std::move(parameters),
                                        std::move(virtual_columns));

  std::vector<std::shared_ptr<cucascade::data_batch>> inputs{input_batch};
  auto outputs = table_scan.execute(operator_data(inputs), cudf::get_default_stream());
  REQUIRE(outputs.get_data_batches().size() == 1);
  auto output_table =
    outputs.get_data_batches()[0]->get_data()->cast<gpu_table_representation>().get_table();
  auto out_view    = output_table.view();
  auto host_vals   = copy_column_to_host<typename Traits::type>(out_view.column(1));
  auto host_filter = copy_column_to_host<int64_t>(out_view.column(0));

  // Expected: filter_vals > 3, so we expect values at indices 3 and 4 (5 and 7)
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

TEST_CASE("sirius_physical_table_scan with no filters passes through data", "[physical_table_scan]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);

  // Create a simple two-column batch
  std::vector<int64_t> col0_vals{1, 2, 3, 4, 5};
  std::vector<int32_t> col1_vals{10, 20, 30, 40, 50};

  auto input_batch = make_two_column_batch<int64_t, int32_t>(
    *space, col0_vals, col1_vals, cudf::type_id::INT32, std::nullopt);

  // Empty table filter set (no filters)
  auto table_filters = duckdb::make_uniq<duckdb::TableFilterSet>();

  // Setup types
  duckdb::vector<duckdb::LogicalType> types;
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER));

  duckdb::vector<duckdb::LogicalType> returned_types = types;

  duckdb::vector<duckdb::ColumnIndex> column_ids;
  column_ids.push_back(duckdb::ColumnIndex(0));
  column_ids.push_back(duckdb::ColumnIndex(1));

  duckdb::vector<duckdb::idx_t> projection_ids{0, 1};
  duckdb::vector<std::string> names{"col0", "col1"};
  duckdb::vector<duckdb::Value> parameters;
  duckdb::virtual_column_map_t virtual_columns;

  duckdb::TableFunction table_function("test_scan", {}, nullptr, nullptr);

  sirius_physical_table_scan table_scan(std::move(types),
                                        std::move(table_function),
                                        nullptr,
                                        std::move(returned_types),
                                        std::move(column_ids),
                                        std::move(projection_ids),
                                        std::move(names),
                                        std::move(table_filters),
                                        col0_vals.size(),
                                        duckdb::ExtraOperatorInfo(),
                                        std::move(parameters),
                                        std::move(virtual_columns));

  std::vector<std::shared_ptr<cucascade::data_batch>> inputs{input_batch};
  auto outputs = table_scan.execute(operator_data(inputs), cudf::get_default_stream());

  REQUIRE(outputs.get_data_batches().size() == 1);
  auto output_table =
    outputs.get_data_batches()[0]->get_data()->cast<gpu_table_representation>().get_table();
  auto out_view = output_table.view();

  // Verify all data passes through unchanged
  auto host_col0 = copy_column_to_host<int64_t>(out_view.column(0));
  auto host_col1 = copy_column_to_host<int32_t>(out_view.column(1));

  REQUIRE(host_col0 == col0_vals);
  REQUIRE(host_col1 == col1_vals);
}

TEST_CASE("sirius_physical_table_scan with multiple filters", "[physical_table_scan]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);

  // Create a two-column batch
  std::vector<int64_t> col0_vals{1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int32_t> col1_vals{5, 10, 15, 20, 25, 30, 35, 40};

  auto input_batch = make_two_column_batch<int64_t, int32_t>(
    *space, col0_vals, col1_vals, cudf::type_id::INT32, std::nullopt);

  // Create table filter set with filters on both columns
  // col0 > 2 AND col1 <= 30
  auto table_filters = duckdb::make_uniq<duckdb::TableFilterSet>();

  auto filter0 = duckdb::make_uniq<duckdb::ConstantFilter>(
    duckdb::ExpressionType::COMPARE_GREATERTHAN, duckdb::Value::BIGINT(2));
  table_filters->PushFilter(duckdb::ColumnIndex(0), std::move(filter0));

  auto filter1 = duckdb::make_uniq<duckdb::ConstantFilter>(
    duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO, duckdb::Value::INTEGER(30));
  table_filters->PushFilter(duckdb::ColumnIndex(1), std::move(filter1));

  // Setup types
  duckdb::vector<duckdb::LogicalType> types;
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER));

  duckdb::vector<duckdb::LogicalType> returned_types = types;

  duckdb::vector<duckdb::ColumnIndex> column_ids;
  column_ids.push_back(duckdb::ColumnIndex(0));
  column_ids.push_back(duckdb::ColumnIndex(1));

  duckdb::vector<duckdb::idx_t> projection_ids{0, 1};
  duckdb::vector<std::string> names{"col0", "col1"};
  duckdb::vector<duckdb::Value> parameters;
  duckdb::virtual_column_map_t virtual_columns;

  duckdb::TableFunction table_function("test_scan", {}, nullptr, nullptr);

  sirius_physical_table_scan table_scan(std::move(types),
                                        std::move(table_function),
                                        nullptr,
                                        std::move(returned_types),
                                        std::move(column_ids),
                                        std::move(projection_ids),
                                        std::move(names),
                                        std::move(table_filters),
                                        col0_vals.size(),
                                        duckdb::ExtraOperatorInfo(),
                                        std::move(parameters),
                                        std::move(virtual_columns));

  std::vector<std::shared_ptr<cucascade::data_batch>> inputs{input_batch};
  auto outputs = table_scan.execute(operator_data(inputs), cudf::get_default_stream());

  REQUIRE(outputs.get_data_batches().size() == 1);
  auto output_table =
    outputs.get_data_batches()[0]->get_data()->cast<gpu_table_representation>().get_table();
  auto out_view = output_table.view();

  auto host_col0 = copy_column_to_host<int64_t>(out_view.column(0));
  auto host_col1 = copy_column_to_host<int32_t>(out_view.column(1));

  // Expected: col0 > 2 AND col1 <= 30
  // Indices: 2(3,15), 3(4,20), 4(5,25), 5(6,30)
  std::vector<int64_t> expected_col0{3, 4, 5, 6};
  std::vector<int32_t> expected_col1{15, 20, 25, 30};

  REQUIRE(host_col0 == expected_col0);
  REQUIRE(host_col1 == expected_col1);
}

TEST_CASE("sirius_physical_table_scan filters all rows", "[physical_table_scan]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);

  // Create a batch
  std::vector<int64_t> col0_vals{1, 2, 3};
  std::vector<int32_t> col1_vals{10, 20, 30};

  auto input_batch = make_two_column_batch<int64_t, int32_t>(
    *space, col0_vals, col1_vals, cudf::type_id::INT32, std::nullopt);

  // Filter that excludes all rows: col0 > 100
  auto table_filters = duckdb::make_uniq<duckdb::TableFilterSet>();
  auto filter        = duckdb::make_uniq<duckdb::ConstantFilter>(
    duckdb::ExpressionType::COMPARE_GREATERTHAN, duckdb::Value::BIGINT(100));
  table_filters->PushFilter(duckdb::ColumnIndex(0), std::move(filter));

  // Setup types
  duckdb::vector<duckdb::LogicalType> types;
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER));

  duckdb::vector<duckdb::LogicalType> returned_types = types;

  duckdb::vector<duckdb::ColumnIndex> column_ids;
  column_ids.push_back(duckdb::ColumnIndex(0));
  column_ids.push_back(duckdb::ColumnIndex(1));

  duckdb::vector<duckdb::idx_t> projection_ids{0, 1};
  duckdb::vector<std::string> names{"col0", "col1"};
  duckdb::vector<duckdb::Value> parameters;
  duckdb::virtual_column_map_t virtual_columns;

  duckdb::TableFunction table_function("test_scan", {}, nullptr, nullptr);

  sirius_physical_table_scan table_scan(std::move(types),
                                        std::move(table_function),
                                        nullptr,
                                        std::move(returned_types),
                                        std::move(column_ids),
                                        std::move(projection_ids),
                                        std::move(names),
                                        std::move(table_filters),
                                        col0_vals.size(),
                                        duckdb::ExtraOperatorInfo(),
                                        std::move(parameters),
                                        std::move(virtual_columns));

  std::vector<std::shared_ptr<cucascade::data_batch>> inputs{input_batch};
  auto outputs = table_scan.execute(operator_data(inputs), cudf::get_default_stream());

  REQUIRE(outputs.get_data_batches().size() == 1);
  auto table =
    outputs.get_data_batches()[0]->get_data()->cast<gpu_table_representation>().get_table();
  auto view = table.view();
  REQUIRE(view.num_columns() == 2);
  REQUIRE(view.num_rows() == 0);
}
