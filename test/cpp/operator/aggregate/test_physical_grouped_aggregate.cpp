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
#include "data/data_batch_utils.hpp"
#include "expression/ast/node.hpp"
#include "utils/test_validation_utility.hpp"

#include <catch.hpp>
#include <op/sirius_physical_grouped_aggregate.hpp>

#include <algorithm>
#include <optional>
#include <set>

using namespace duckdb;
using namespace sirius::op;
using namespace cucascade;
using namespace cucascade::memory;

namespace {

using namespace sirius::test::operator_utils;

}  // namespace

TEMPLATE_TEST_CASE(
  "sirius_physical_grouped_aggregate grouped aggregates data_batch with single partition key, "
  "multiple aggregations on numerics",
  "[physical_grouped_aggregate]",
  int32_t,
  int64_t,
  float,
  double,
  int16_t,
  decimal64_tag)
{
  using Traits = gpu_type_traits<TestType>;

  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space != nullptr);

  auto mr                = get_resource_ref(*space);
  auto stream            = default_stream();
  std::size_t num_groups = 1000;

  // Create test data with single group key column
  auto [input_table, expected_table] =
    sirius::test::make_test_data_for_grouped_aggregate<Traits>(num_groups, 1, stream, mr);

  std::shared_ptr<data_batch> input_batch = sirius::make_data_batch(
    std::move(input_table), *space, stream, sirius::telemetry::batch_telemetry_info{});

  // Create aggregate expressions: GROUP BY column 0, SUM(column 1)
  auto agg_result = sirius::test::create_aggregate_expressions<Traits>(
    {0},                      // group_indexes: GROUP BY column 0
    {"min", "max", "count"},  // aggregations: MIN, MAX, COUNT
    {1, 1, 1}                 // agg_indexes: MIN(column 1), MAX(column 1), COUNT(column 1)
  );

  // Create the grouped aggregate merge operator
  sirius_physical_grouped_aggregate grouped_aggregator(std::move(agg_result.output_types),
                                                       std::move(agg_result.aggregates),
                                                       std::move(agg_result.groups),
                                                       num_groups);

  auto outputs =
    grouped_aggregator.execute(pipelineable_operator_data({input_batch}), default_stream());

  // Verify we got one output batch
  REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches().size() == 1);

  // Compare output with expected using the validation utility
  // Sort both tables before comparison since aggregation order is not guaranteed
  bool tables_match = sirius::test::expect_data_batch_equivalent_to_table(
    dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches()[0],
    expected_table->view(),
    true);
  REQUIRE(tables_match);
}

TEMPLATE_TEST_CASE("sirius_physical_grouped_aggregate grouped aggregates with AVG decomposition",
                   "[physical_grouped_aggregate]",
                   int32_t,
                   int64_t,
                   float,
                   double,
                   int16_t,
                   decimal64_tag)
{
  using Traits = gpu_type_traits<TestType>;

  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space != nullptr);

  auto mr                = get_resource_ref(*space);
  auto stream            = default_stream();
  std::size_t num_groups = 100;

  auto [input_table, expected_table] =
    sirius::test::make_test_data_for_grouped_aggregate_with_avg<Traits>(num_groups, 1, stream, mr);

  std::shared_ptr<data_batch> input_batch = sirius::make_data_batch(
    std::move(input_table), *space, stream, sirius::telemetry::batch_telemetry_info{});

  // AVG is decomposed into SUM + COUNT_VALID internally
  auto agg_result = sirius::test::create_aggregate_expressions<Traits>(
    {0},                             // GROUP BY column 0
    {"min", "max", "count", "avg"},  // aggregations including AVG
    {1, 1, 1, 1}                     // all on column 1
  );

  sirius_physical_grouped_aggregate grouped_aggregator(std::move(agg_result.output_types),
                                                       std::move(agg_result.aggregates),
                                                       std::move(agg_result.groups),
                                                       num_groups);

  auto outputs =
    grouped_aggregator.execute(pipelineable_operator_data({input_batch}), default_stream());
  REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches().size() == 1);

  // The local operator outputs expanded columns: group_key, min, max, count, sum, count_valid
  // (AVG decomposed into SUM + COUNT_VALID). Verify column count = 1 group + 5 aggregates.
  auto output_view = sirius::get_cudf_table_view(
    *dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches()[0]);
  REQUIRE(output_view.num_columns() == 6);  // 1 group + 3 non-avg + 2 (sum+count for avg)
}

TEMPLATE_TEST_CASE(
  "sirius_physical_grouped_aggregate grouped aggregates data_batch with multiple partition key, "
  "multiple aggregations",
  "[physical_grouped_aggregate]",
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

  std::shared_ptr<data_batch> input_batch = sirius::make_data_batch(
    std::move(input_table), *space, stream, sirius::telemetry::batch_telemetry_info{});

  // Create aggregate expressions: GROUP BY column 0, SUM(column 1)
  auto agg_result = sirius::test::create_aggregate_expressions<Traits>(
    {0, 1},                   // group_indexes: GROUP BY column 0 and 1
    {"min", "max", "count"},  // aggregations: MIN, MAX, COUNT
    {2, 2, 2}                 // agg_indexes: MIN(column 2), MAX(column 2), COUNT(column 2)
  );

  // Create the grouped aggregate merge operator
  sirius_physical_grouped_aggregate grouped_aggregator(std::move(agg_result.output_types),
                                                       std::move(agg_result.aggregates),
                                                       std::move(agg_result.groups),
                                                       num_groups);

  auto outputs =
    grouped_aggregator.execute(pipelineable_operator_data({input_batch}), default_stream());

  // Verify we got one output batch
  REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches().size() == 1);

  // Compare output with expected using the validation utility
  // Sort both tables before comparison since aggregation order is not guaranteed
  bool tables_match = sirius::test::expect_data_batch_equivalent_to_table(
    dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches()[0],
    expected_table->view(),
    true);
  REQUIRE(tables_match);
}

TEST_CASE("sirius_physical_grouped_aggregate drops whole groups that fail the merge's HAVING",
          "[physical_grouped_aggregate]")
{
  using Traits        = gpu_type_traits<int64_t>;
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space != nullptr);

  // Groups and their COUNT(v): 1 -> 2, 2 -> 1, 3 -> 3, 5 -> 1, NULL -> 2.
  auto keys =
    make_numeric_batch_with_nulls<int64_t>(*space,
                                           {1, 1, 2, 3, 3, 3, 5, 0, 0},
                                           {true, true, true, true, true, true, true, false, false},
                                           cudf::type_id::INT64);
  auto value =
    make_numeric_batch<int64_t>(*space, std::vector<int64_t>(9, 10), cudf::type_id::INT64);
  auto input = concatenate_batches_horizontal({keys, value}, *space);

  auto expressions = sirius::test::create_aggregate_expressions<Traits>({0}, {"count"}, {1});
  sirius_physical_grouped_aggregate aggregate(std::move(expressions.output_types),
                                              std::move(expressions.aggregates),
                                              std::move(expressions.groups),
                                              5);
  // HAVING COUNT(v) > 1, with 1 and 5 the min and max of a row group: a group with either key may
  // continue in another row group, so only group 2 is whole and failing.
  auto const bigint = sirius::logical_type::make(sirius::type_id::BIGINT);
  auto filter       = std::make_shared<complete_group_having>();
  filter->predicate = std::make_unique<sirius::ast::node>(sirius::ast::comparison{
    sirius::comparison_type::gt,
    std::make_unique<sirius::ast::node>(sirius::ast::reference{1, bigint}),
    std::make_unique<sirius::ast::node>(sirius::ast::constant{sirius::value{int64_t{1}}, bigint})});
  filter->key       = 0;
  std::vector<int64_t> const bounds{1, 5};
  filter->row_group_bounds = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT64}, 2, cudf::mask_state::UNALLOCATED, default_stream());
  cudaMemcpy(filter->row_group_bounds->mutable_view().data<int64_t>(),
             bounds.data(),
             sizeof(int64_t) * bounds.size(),
             cudaMemcpyHostToDevice);
  aggregate.complete_groups = filter;

  auto const groups = [&](bool whole_row_groups) {
    pipelineable_operator_data data({input});
    data.set_whole_row_groups(whole_row_groups);
    auto out          = aggregate.execute(data, default_stream());
    auto const& batch = dynamic_cast<pipelineable_operator_data const&>(*out).get_data_batches();
    REQUIRE(batch.size() == 1);
    auto const view  = sirius::get_cudf_table_view(*batch[0]);
    auto const key   = copy_column_to_host<int64_t>(view.column(0));
    auto const valid = copy_validity_to_host(view.column(0));
    std::multiset<std::optional<int64_t>> out_keys;
    for (std::size_t i = 0; i < key.size(); ++i) {
      out_keys.insert(valid[i] ? std::optional<int64_t>{key[i]} : std::nullopt);
    }
    return out_keys;
  };

  // The NULL group gathers the NULLs of every row group, so it is never dropped.
  CHECK(groups(true) == std::multiset<std::optional<int64_t>>{std::nullopt, 1, 3, 5});
  // Without the mark a batch may hold part of a row group: nothing is dropped.
  CHECK(groups(false) == std::multiset<std::optional<int64_t>>{std::nullopt, 1, 2, 3, 5});
  CHECK(filter->groups_seen.load() == 5);
  CHECK(filter->groups_kept.load() == 4);
}
