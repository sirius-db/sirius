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

#include "../operator_test_utils.hpp"
#include "../operator_type_traits.hpp"
#include "aggregate_test_utils.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "op/sirius_physical_grouped_aggregate_merge.hpp"
#include "utils/data_utils.hpp"
#include "utils/test_validation_utility.hpp"

#include <cudf/copying.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/unary.hpp>

#include <catch.hpp>

#include <algorithm>
#include <cmath>
#include <utility>

using namespace duckdb;
using namespace sirius::op;
using namespace cucascade;
using namespace cucascade::memory;

namespace {

using namespace sirius::test::operator_utils;
using sirius::test::vector_to_cudf_column;

std::unique_ptr<sirius_physical_grouped_aggregate> make_mixed_stddev_aggregate(bool grouping_sets)
{
  using Traits = gpu_type_traits<double>;
  auto defs    = sirius::test::create_aggregate_expressions<Traits>(
    {0}, {"stddev_samp", "avg", "count"}, {1, 1, 1});
  auto distinct = sirius::test::create_count_distinct_expressions<Traits>({0}, 1);
  defs.aggregates.insert(defs.aggregates.begin() + 1, std::move(distinct.aggregates[0]));
  defs.output_types.insert(defs.output_types.begin() + 2, distinct.output_types[1]);
  duckdb::vector<duckdb::GroupingSet> sets;
  duckdb::vector<duckdb::unsafe_vector<std::size_t>> functions;
  if (grouping_sets) {
    sets = {{0}, {0}, {}};  // Duplicate leaf sets must remain separate through merge.
    functions.push_back({0});
    defs.output_types.insert(defs.output_types.begin() + 1,
                             sirius::logical_type::make(sirius::type_id::INTEGER));
    defs.output_types.insert(defs.output_types.begin() + 2,
                             sirius::logical_type::make(sirius::type_id::BIGINT));
  }
  return std::make_unique<sirius_physical_grouped_aggregate>(
    std::move(defs.output_types),
    std::move(defs.aggregates),
    std::move(defs.groups),
    std::move(sets),
    std::move(functions),
    7,
    duckdb::TupleDataValidityType::CAN_HAVE_NULL_VALUES,
    duckdb::TupleDataValidityType::CAN_HAVE_NULL_VALUES);
}
}  // namespace

TEST_CASE("mixed stddev partial schema includes grouping-set columns",
          "[stddev_samp][grouping_sets][unit]")
{
  bool const grouping_sets = GENERATE(false, true);
  auto local               = make_mixed_stddev_aggregate(grouping_sets);
  auto const prefix        = local->num_output_group_columns();
  REQUIRE(prefix == (grouping_sets ? 3 : 1));
  REQUIRE(local->local_types.size() == prefix + 5);
  REQUIRE(local->local_types[0].id() == sirius::type_id::DOUBLE);
  if (grouping_sets) {
    REQUIRE(local->local_types[1].id() == sirius::type_id::INTEGER);
    REQUIRE(local->local_types[2].id() == sirius::type_id::BIGINT);
  }
  REQUIRE(local->local_types[prefix].id() == sirius::type_id::STRUCT);
  REQUIRE(local->local_types[prefix + 1].id() == sirius::type_id::LIST);
  REQUIRE(local->local_types[prefix + 2].id() == sirius::type_id::DOUBLE);
  REQUIRE(local->local_types[prefix + 3].id() == sirius::type_id::BIGINT);
  REQUIRE(local->local_types[prefix + 4].id() == sirius::type_id::BIGINT);
  sirius_physical_grouped_aggregate_merge merge(local.get());
  REQUIRE(merge.types.size() == prefix + 4);
  REQUIRE(merge.types[prefix].id() == sirius::type_id::DOUBLE);
  REQUIRE(merge.types[prefix + 1].id() == sirius::type_id::BIGINT);
}

TEST_CASE("grouped merge sizing applies the row floor to its strategy and repository",
          "[physical_grouped_aggregate_merge][partition_build][unit]")
{
  constexpr uint64_t kTiB = uint64_t{1} << 40;
  uint64_t total_bytes    = 122ull << 30;
  uint64_t total_rows     = 3'387'732'936ull;
  int expected_count      = 4;

  SECTION("large row count raises a one-partition byte estimate") {}
  SECTION("one partition suffices at the row target")
  {
    total_rows     = uint64_t{1} << 30;
    expected_count = 1;
  }
  SECTION("one row above the target requires a second partition")
  {
    total_rows     = (uint64_t{1} << 30) + 1;
    expected_count = 2;
  }
  SECTION("a larger byte-based count is preserved")
  {
    total_bytes    = 6 * kTiB;
    expected_count = 6;
  }

  // Sizing only: synthetic totals and an empty repository need no GPU data allocations.
  sirius_physical_grouped_aggregate local({}, {}, {}, 0);
  shared_data_repository repo;
  sirius_physical_grouped_aggregate_merge merge(&local, kTiB);
  merge.set_active_gpu_ids({0});
  auto port  = std::make_unique<sirius_physical_operator::port>();
  port->repo = &repo;
  merge.add_port("default", std::move(port));

  auto const strategy =
    merge.get_partition_strategy({total_bytes, false, false, total_bytes, total_rows});
  CHECK(strategy.num_partitions == expected_count);
  CHECK(repo.num_partitions() == static_cast<std::size_t>(expected_count));
  CHECK(strategy.placement.num_partitions() == static_cast<std::size_t>(expected_count));
  CHECK_FALSE(strategy.broadcast);
  CHECK_FALSE(strategy.build_probe);
}

// single batch expects to return the exact same data as the input batch, since it assumes it was
// already aggregated
TEST_CASE("sirius_physical_grouped_aggregate_merge grouped aggregates single data_batch",
          "[physical_grouped_aggregate_merge]")
{
  using Traits = gpu_type_traits<int32_t>;

  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space != nullptr);

  auto mr                = get_resource_ref(*space);
  auto stream            = default_stream();
  std::size_t num_groups = 10;

  // Create test data with single group key column
  // For the merge test, we need the expected (aggregated) table as input
  auto [raw_input_table, expected_table] =
    sirius::test::make_test_data_for_grouped_aggregate<Traits>(num_groups, 1, stream, mr);

  // Create aggregate expressions: GROUP BY column 0, SUM(column 1)
  auto agg_result = sirius::test::create_aggregate_expressions<Traits>(
    {0},                      // group_indexes: GROUP BY column 0
    {"min", "max", "count"},  // aggregations: MIN, MAX, COUNT
    {1, 1, 1}                 // agg_indexes: MIN(column 1), MAX(column 1), COUNT(column 1)
  );

  // Create the grouped aggregate merge operator
  sirius_physical_grouped_aggregate_merge grouped_aggregate_merger(
    std::move(agg_result.output_types),
    std::move(agg_result.aggregates),
    std::move(agg_result.groups),
    num_groups);

  // For merge test, the input is already aggregated data (the expected table)
  // The merge operator should return it unchanged for a single batch
  auto input_table = std::make_unique<cudf::table>(expected_table->view());
  auto input_batch = sirius::make_data_batch(
    std::move(input_table), *space, stream, sirius::telemetry::batch_telemetry_info{});

  auto outputs = grouped_aggregate_merger.execute(
    pipelineable_operator_data({std::move(input_batch)}), default_stream());

  // Verify we got one output batch
  REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches().size() == 1);

  // Compare output with expected using the validation utility
  // Dont Sort both tables before comparison since they should be identical
  bool tables_match = sirius::test::expect_data_batch_equivalent_to_table(
    dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches()[0],
    expected_table->view(),
    false);
  REQUIRE(tables_match);
}

TEMPLATE_TEST_CASE(
  "sirius_physical_grouped_aggregate_merge grouped aggregates data_batch with multiple partition "
  "key, multiple aggregations",
  "[physical_grouped_aggregate_merge]",
  int32_t,
  int64_t,
  float,
  double,
  int16_t,
  decimal64_tag,
  string_tag,
  timestamp_us_tag,
  date32_tag)
{
  using Traits = gpu_type_traits<TestType>;

  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space != nullptr);

  auto mr                = get_resource_ref(*space);
  auto stream            = default_stream();
  std::size_t num_groups = 1000;

  // Create test data with two group key columns
  auto [input_table, expected_table] =
    sirius::test::make_test_data_for_grouped_aggregate<Traits>(num_groups, 2, stream, mr);

  auto input_tables =
    sirius::test::make_random_striped_split(std::move(input_table), 5, stream, mr);

  // Create aggregate expressions for grouped_aggregator
  auto agg_result1 = sirius::test::create_aggregate_expressions<Traits>(
    {0, 1},                   // group_indexes: GROUP BY column 0 and 1
    {"min", "max", "count"},  // aggregations: MIN, MAX, COUNT
    {2, 2, 2}                 // agg_indexes: MIN(column 2), MAX(column 2), COUNT(column 2)
  );

  // Create aggregate expressions for grouped_aggregate_merger
  auto agg_result2 = sirius::test::create_aggregate_expressions<Traits>(
    {0, 1},                   // group_indexes: GROUP BY column 0 and 1
    {"min", "max", "count"},  // aggregations: MIN, MAX, COUNT
    {2, 2, 2}                 // agg_indexes: MIN(column 2), MAX(column 2), COUNT(column 2)
  );

  // Create the grouped aggregate operator
  sirius_physical_grouped_aggregate grouped_aggregator(std::move(agg_result1.output_types),
                                                       std::move(agg_result1.aggregates),
                                                       std::move(agg_result1.groups),
                                                       num_groups);

  // Create the grouped aggregate merge operator
  sirius_physical_grouped_aggregate_merge grouped_aggregate_merger(
    std::move(agg_result2.output_types),
    std::move(agg_result2.aggregates),
    std::move(agg_result2.groups),
    num_groups);

  std::vector<std::shared_ptr<data_batch>> agg_outputs;

  for (auto& input_table : input_tables) {
    std::shared_ptr<data_batch> input_batch = sirius::make_data_batch(
      std::move(input_table), *space, stream, sirius::telemetry::batch_telemetry_info{});

    auto outputs =
      grouped_aggregator.execute(pipelineable_operator_data({input_batch}), default_stream());

    // Verify we got one output batch
    REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches().size() ==
            1);
    agg_outputs.push_back(
      dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches()[0]);
  }

  auto outputs =
    grouped_aggregate_merger.execute(pipelineable_operator_data(agg_outputs), default_stream());
  REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches().size() == 1);

  REQUIRE(expected_table->get_column(4).type().id() == cudf::type_id::INT64);

  // Compare output with expected using the validation utility
  // Sort both tables before comparison since aggregation order is not guaranteed
  bool tables_match = sirius::test::expect_data_batch_equivalent_to_table(
    dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches()[0],
    expected_table->view(),
    true);
  REQUIRE(tables_match);
}

TEMPLATE_TEST_CASE("sirius_physical_grouped_aggregate_merge end-to-end with AVG",
                   "[physical_grouped_aggregate_merge]",
                   int32_t,
                   int64_t,
                   float,
                   double,
                   int16_t)
//  decimal64_tag)  TODO: the unit test with decimal64_tag is failing with a cuda memory alignment
//  error due to https://github.com/rapidsai/cudf/issues/21512

{
  using Traits = gpu_type_traits<TestType>;

  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space != nullptr);

  auto mr                = get_resource_ref(*space);
  auto stream            = default_stream();
  std::size_t num_groups = 100;

  // Create test data with AVG expected values
  auto [input_table, expected_table] =
    sirius::test::make_test_data_for_grouped_aggregate_with_avg<Traits>(num_groups, 1, stream, mr);

  // Split input into 5 batches for distributed aggregation
  auto input_tables =
    sirius::test::make_random_striped_split(std::move(input_table), 5, stream, mr);

  // Create aggregate expressions for local operator
  auto agg_result1 = sirius::test::create_aggregate_expressions<Traits>(
    {0},                             // GROUP BY column 0
    {"min", "max", "count", "avg"},  // aggregations including AVG
    {1, 1, 1, 1}                     // all on column 1
  );

  // Create aggregate expressions for merge operator
  auto agg_result2 = sirius::test::create_aggregate_expressions<Traits>(
    {0}, {"min", "max", "count", "avg"}, {1, 1, 1, 1});

  // Create local and merge operators
  sirius_physical_grouped_aggregate grouped_aggregator(std::move(agg_result1.output_types),
                                                       std::move(agg_result1.aggregates),
                                                       std::move(agg_result1.groups),
                                                       num_groups);

  sirius_physical_grouped_aggregate_merge grouped_aggregate_merger(&grouped_aggregator);

  // Run local aggregation on each split
  std::vector<std::shared_ptr<data_batch>> agg_outputs;
  for (auto& split_table : input_tables) {
    auto input_batch = sirius::make_data_batch(
      std::move(split_table), *space, stream, sirius::telemetry::batch_telemetry_info{});
    auto outputs =
      grouped_aggregator.execute(pipelineable_operator_data({input_batch}), default_stream());
    REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches().size() ==
            1);
    agg_outputs.push_back(
      dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches()[0]);
  }

  // Run merge with AVG projection
  auto outputs =
    grouped_aggregate_merger.execute(pipelineable_operator_data(agg_outputs), default_stream());
  REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches().size() == 1);

  REQUIRE(expected_table->get_column(3).type().id() == cudf::type_id::INT64);

  bool tables_match = sirius::test::expect_data_batch_equivalent_to_table(
    dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches()[0],
    expected_table->view(),
    true);
  REQUIRE(tables_match);
}

TEST_CASE("stddev_samp merges singleton states and preserves NULL groups",
          "[stddev_samp][physical_grouped_aggregate_merge]")
{
  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space  = manager->get_memory_space(Tier::GPU, 0);
  REQUIRE(space);
  auto stream = default_stream();
  auto mr     = get_resource_ref(*space);
  auto defs   = sirius::test::create_aggregate_expressions<gpu_type_traits<double>>(
    {0}, {"stddev_samp", "avg"}, {1, 1});
  sirius_physical_grouped_aggregate local(
    std::move(defs.output_types), std::move(defs.aggregates), std::move(defs.groups), 4);
  sirius_physical_grouped_aggregate_merge merge(&local);
  REQUIRE(local.local_types.size() == 4);  // key, stddev STRUCT, AVG sum, AVG count
  REQUIRE(local.local_types[1].id() == sirius::type_id::STRUCT);
  bool const multiple_values_per_partial = GENERATE(false, true);
  std::vector<std::shared_ptr<data_batch>> batches;
  for (int i = 1; i <= 4; ++i) {
    std::vector<double> raw_values{1e12 + i, 0, 42, 7};
    std::vector<bool> valid{true, false, i == 1, true};
    std::vector<double> raw_keys{0, 1, 2, 3};
    if (multiple_values_per_partial) {
      raw_values.push_back(1e12 + i + 4);
      valid.push_back(true);
      raw_keys.push_back(0);
    }
    auto values =
      make_numeric_batch_with_nulls<double>(*space, raw_values, valid, cudf::type_id::FLOAT64);
    auto ro    = values->to_read_only();
    auto table = ro.get_data()->cast<gpu_table_representation>().get_table_view();
    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.push_back(vector_to_cudf_column<gpu_type_traits<double>>(raw_keys, stream, mr));
    cols.push_back(std::make_unique<cudf::column>(table.column(0), stream, mr));
    batches.push_back(sirius::make_data_batch(std::make_unique<cudf::table>(std::move(cols)),
                                              *space,
                                              stream,
                                              sirius::telemetry::batch_telemetry_info{}));
  }
  auto partial        = local.execute(pipelineable_operator_data(batches), stream);
  auto result         = merge.execute(*partial, stream);
  auto const& outputs = dynamic_cast<pipelineable_operator_data&>(*result).get_data_batches();
  REQUIRE(outputs.size() == 1);
  auto ro   = outputs[0]->to_read_only();
  auto view = ro.get_data()->cast<gpu_table_representation>().get_table_view();
  auto keys = copy_column_to_host<double>(view.column(0));
  REQUIRE(view.num_rows() == 4);
  for (int row = 0; row < 4; ++row) {
    auto sd = cudf::get_element(view.column(1), row, stream, mr);
    if (keys[row] == 1 || keys[row] == 2) {
      REQUIRE_FALSE(sd->is_valid(stream));
    } else {
      REQUIRE(sd->is_valid(stream));
      auto expected =
        keys[row] == 0 ? std::sqrt(multiple_values_per_partial ? 6.0 : 5.0 / 3.0) : 0.0;
      REQUIRE(static_cast<cudf::numeric_scalar<double> const&>(*sd).value(stream) ==
              Approx(expected).margin(1e-9));
    }
  }
}

TEST_CASE("mixed stddev grouping states merge across batches including empty partials",
          "[stddev_samp][grouping_sets][physical_grouped_aggregate_merge]")
{
  bool const grouping_sets = GENERATE(false, true);
  bool const split_batches = GENERATE(false, true);
  bool const empty_input   = GENERATE(false, true);
  auto manager             = sirius::test::operator_utils::initialize_memory_manager();
  auto* space              = manager->get_memory_space(Tier::GPU, 0);
  REQUIRE(space);
  auto stream = default_stream();
  auto mr     = get_resource_ref(*space);
  auto local  = make_mixed_stddev_aggregate(grouping_sets);
  sirius_physical_grouped_aggregate_merge merge(local.get());
  auto const prefix = local->num_output_group_columns();

  std::vector<std::shared_ptr<data_batch>> batches;
  auto add_batch = [&](std::vector<double> const& keys,
                       std::vector<double> const& values,
                       std::vector<bool> const& valid) {
    auto value_batch =
      make_numeric_batch_with_nulls<double>(*space, values, valid, cudf::type_id::FLOAT64);
    auto ro    = value_batch->to_read_only();
    auto table = ro.get_data()->cast<gpu_table_representation>().get_table_view();
    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.push_back(vector_to_cudf_column<gpu_type_traits<double>>(keys, stream, mr));
    cols.push_back(std::make_unique<cudf::column>(table.column(0), stream, mr));
    batches.push_back(sirius::make_data_batch(std::make_unique<cudf::table>(std::move(cols)),
                                              *space,
                                              stream,
                                              sirius::telemetry::batch_telemetry_info{}));
  };
  constexpr double offset = 1e12;
  if (!empty_input) {
    if (split_batches) {
      add_batch({0, 0, 1, 2}, {offset + 1, offset + 1, 0, 42}, {true, true, false, true});
      add_batch({0, 0, 1}, {offset + 3, offset + 3, 0}, {true, true, false});
    } else {
      add_batch({0, 0, 1, 2, 0, 0, 1},
                {offset + 1, offset + 1, 0, 42, offset + 3, offset + 3, 0},
                {true, true, false, true, true, true, false});
    }
  }
  // A real empty batch must emit a neutral grand-total state. Exercise both the single
  // batch clone/finalization path and the multi-batch merge path for wholly empty input.
  if (empty_input || split_batches) { add_batch({}, {}, {}); }
  if (empty_input && split_batches) { add_batch({}, {}, {}); }

  auto partial = local->execute(pipelineable_operator_data(batches), stream);
  for (auto const& batch : dynamic_cast<pipelineable_operator_data&>(*partial).get_data_batches()) {
    auto ro   = batch->to_read_only();
    auto view = ro.get_data()->cast<gpu_table_representation>().get_table_view();
    REQUIRE(view.num_columns() == prefix + 5);
    for (std::size_t i = 0; i < local->local_types.size(); ++i) {
      REQUIRE(view.column(i).type() == sirius::get_cudf_type(local->local_types[i]));
    }
    cudf::structs_column_view state(view.column(prefix));
    REQUIRE(state.num_children() == 3);
    REQUIRE(state.child(0).type().id() == cudf::type_id::INT64);
    REQUIRE(state.child(1).type().id() == cudf::type_id::FLOAT64);
    REQUIRE(state.child(2).type().id() == cudf::type_id::FLOAT64);
    if (empty_input && grouping_sets) {
      REQUIRE(view.num_rows() == 1);
      REQUIRE(view.column(prefix).null_count() == 0);
      auto count = cudf::get_element(state.child(0), 0, stream, mr);
      REQUIRE(count->is_valid(stream));
      REQUIRE(static_cast<cudf::numeric_scalar<int64_t> const&>(*count).value(stream) == 0);
      for (int child = 1; child < 3; ++child) {
        auto value = cudf::get_element(state.child(child), 0, stream, mr);
        REQUIRE(value->is_valid(stream));
        REQUIRE(static_cast<cudf::numeric_scalar<double> const&>(*value).value(stream) == 0);
      }
    }
  }

  auto result         = merge.execute(*partial, stream);
  auto const& outputs = dynamic_cast<pipelineable_operator_data&>(*result).get_data_batches();
  REQUIRE(outputs.size() == 1);
  auto ro   = outputs[0]->to_read_only();
  auto view = ro.get_data()->cast<gpu_table_representation>().get_table_view();
  REQUIRE(view.num_columns() == prefix + 4);
  REQUIRE(view.num_rows() == (empty_input ? (grouping_sets ? 1 : 0) : (grouping_sets ? 7 : 3)));
  for (int row = 0; row < view.num_rows(); ++row) {
    bool total = false;
    if (grouping_sets) {
      auto id       = cudf::get_element(view.column(1), row, stream, mr);
      total         = static_cast<cudf::numeric_scalar<int32_t> const&>(*id).value(stream) == 2;
      auto grouping = cudf::get_element(view.column(2), row, stream, mr);
      REQUIRE(static_cast<cudf::numeric_scalar<int64_t> const&>(*grouping).value(stream) ==
              (total ? 1 : 0));
    }
    auto key = cudf::get_element(view.column(0), row, stream, mr);
    REQUIRE(key->is_valid(stream) == !total);
    std::vector<double> expected_values;
    if (!empty_input) {
      auto const k =
        total ? -1 : static_cast<cudf::numeric_scalar<double> const&>(*key).value(stream);
      if (total || k == 0) { expected_values = {offset + 1, offset + 1, offset + 3, offset + 3}; }
      if (total || k == 2) { expected_values.push_back(42); }
    }
    auto distinct = cudf::get_element(view.column(prefix + 1), row, stream, mr);
    auto count    = cudf::get_element(view.column(prefix + 3), row, stream, mr);
    REQUIRE(distinct->is_valid(stream));
    REQUIRE(count->is_valid(stream));
    auto unique_values = expected_values;
    std::sort(unique_values.begin(), unique_values.end());
    unique_values.erase(std::unique(unique_values.begin(), unique_values.end()),
                        unique_values.end());
    REQUIRE(static_cast<cudf::numeric_scalar<int64_t> const&>(*distinct).value(stream) ==
            unique_values.size());
    REQUIRE(static_cast<cudf::numeric_scalar<int64_t> const&>(*count).value(stream) ==
            expected_values.size());
    auto avg = cudf::get_element(view.column(prefix + 2), row, stream, mr);
    REQUIRE(avg->is_valid(stream) == !expected_values.empty());
    long double mean = 0;
    for (auto v : expected_values) {
      mean += v;
    }
    if (!expected_values.empty()) {
      mean /= expected_values.size();
      REQUIRE(static_cast<cudf::numeric_scalar<double> const&>(*avg).value(stream) ==
              Approx(static_cast<double>(mean)).epsilon(1e-12));
    }
    auto sd = cudf::get_element(view.column(prefix), row, stream, mr);
    REQUIRE(sd->is_valid(stream) == (expected_values.size() > 1));
    if (expected_values.size() > 1) {
      long double m2 = 0;
      for (auto v : expected_values) {
        m2 += (v - mean) * (v - mean);
      }
      REQUIRE(
        static_cast<cudf::numeric_scalar<double> const&>(*sd).value(stream) ==
        Approx(static_cast<double>(std::sqrt(m2 / (expected_values.size() - 1)))).epsilon(1e-12));
    }
  }
}
