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

/**
 * @file test_grouped_aggregate_top_n_producer.cpp
 * @brief Stage-5 group-key producer seam in the grouped aggregate sink, deterministically
 *
 * The failure mode this file exists for is not "no pruning" but *quietly wrong aggregate values*:
 * a boundary one comparison too strict drops rows belonging to a group that is in the answer, and
 * that group's `sum` comes out low with no error and no missing row. Driving the operator directly
 * removes the batch-arrival race an end-to-end run has, so the boundary is guaranteed to be in
 * front of the second batch and the corruption is guaranteed to be reachable.
 */

#include "../operator_test_utils.hpp"
#include "../operator_type_traits.hpp"
#include "aggregate_test_utils.hpp"
#include "data/data_batch_utils.hpp"

#include <catch.hpp>
#include <op/cudf_sort_order.hpp>
#include <op/dynamic_filter/dynamic_filter_stats.hpp>
#include <op/dynamic_filter/top_n_group_key_producer.hpp>
#include <op/dynamic_filter/top_n_threshold_coordinator.hpp>
#include <op/sirius_physical_grouped_aggregate.hpp>

#include <cstdint>
#include <map>
#include <memory>
#include <utility>
#include <vector>

using namespace sirius::op;
using namespace cucascade;
using namespace sirius::test::operator_utils;

namespace {

constexpr auto k_int64 = cudf::data_type{cudf::type_id::INT64};

/// One ascending INT64 grouping key, nulls last -- the semantics the planner freezes for
/// `ORDER BY <grouping key>`.
std::vector<top_n_key_semantics> ascending_int64_key()
{
  return {{.storage_type = k_int64,
           .order        = cudf::order::ASCENDING,
           .null_order   = cudf::null_order::AFTER}};
}

/// A grouped aggregate computing `SUM(col 1) GROUP BY col 0` over two INT64 columns.
std::unique_ptr<sirius_physical_grouped_aggregate> make_sum_aggregate()
{
  auto expressions = sirius::test::create_aggregate_expressions<gpu_type_traits<std::int64_t>>(
    {0}, {"sum"}, {1});
  return std::make_unique<sirius_physical_grouped_aggregate>(std::move(expressions.output_types),
                                                             std::move(expressions.aggregates),
                                                             std::move(expressions.groups),
                                                             0);
}

/// Run the operator batch by batch, as the pipeline does, and fold every partial result into one
/// group-to-sum map -- the host-side equivalent of the merge aggregate.
std::map<std::int64_t, std::int64_t> run_and_merge(
  sirius_physical_grouped_aggregate& aggregate,
  std::vector<std::shared_ptr<data_batch>> const& batches)
{
  std::map<std::int64_t, std::int64_t> sums;
  for (auto const& batch : batches) {
    auto outputs =
      aggregate.execute(pipelineable_operator_data({batch}), sirius::test::operator_utils::default_stream());
    for (auto const& produced :
         dynamic_cast<pipelineable_operator_data const&>(*outputs).get_data_batches()) {
      auto const view   = sirius::get_cudf_table_view(*produced);
      auto const keys   = copy_column_to_host<std::int64_t>(view.column(0));
      auto const values = copy_column_to_host<std::int64_t>(view.column(1));
      REQUIRE(keys.size() == values.size());
      for (std::size_t i = 0; i < keys.size(); ++i) {
        sums[keys[i]] += values[i];
      }
    }
  }
  return sums;
}

}  // namespace

TEST_CASE("the group-key producer keeps boundary-tied rows so their group's sum stays exact",
          "[physical_grouped_aggregate][dynamic_filter][top_n]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space != nullptr);

  // Batch one carries thirteen distinct grouping keys, so a K = 5 witness set fills immediately
  // and the boundary becomes the fifth-best key, 4. Batch two then meets that boundary holding
  // rows *tied* with it. Inclusive keeps them and group 4's sum is exact; strict drops all three
  // thousand of their contribution while still returning a row for group 4 -- the silent
  // corruption. Group 9's rows are strictly worse and may legitimately be pruned.
  std::vector<std::shared_ptr<data_batch>> batches;
  batches.push_back(make_two_column_batch<std::int64_t, std::int64_t>(
    *space,
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
    {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22},
    cudf::type_id::INT64));
  batches.push_back(make_two_column_batch<std::int64_t, std::int64_t>(
    *space, {4, 4, 9, 0, 9}, {1000, 2000, 5, 7, 6}, cudf::type_id::INT64));

  auto reference = make_sum_aggregate();
  auto const expected = run_and_merge(*reference, batches);

  dynamic_filter_stats stats;
  auto coordinator = std::make_shared<top_n_threshold_coordinator>(
    5, ascending_int64_key(), true, &stats, top_n_producer_kind::GROUP_KEY);
  auto armed = make_sum_aggregate();
  armed->top_n_producer = std::make_unique<top_n_group_key_producer>(
    coordinator, std::vector<cudf::size_type>{0}, scan::dynamic_filter_gate::k_default_keep_threshold);
  auto const actual = run_and_merge(*armed, batches);

  // The damage assertion comes first, deliberately. A counter assertion proves which rows the
  // predicate selected; only this proves what that did to the answer, and a correct flag
  // mismapped into the kernel's `strict` parameter produces identical counters elsewhere.
  // Every group no worse than the boundary keeps its exact sum. A strict predicate reports 14
  // for group 4 instead of 3014 and fails right here.
  for (auto const& [group, sum] : expected) {
    if (group > 4) { continue; }
    INFO("group " << group);
    REQUIRE(actual.count(group) == 1);
    REQUIRE(actual.at(group) == sum);
  }
  REQUIRE(actual.at(4) == 3014);

  // The boundary is defined and is the fifth-best distinct key.
  REQUIRE(stats.top_n_group_offers.load() == 2);
  REQUIRE(stats.top_n_group_witness_set_full.load() == 1);
  auto const boundary = coordinator->tightest_boundary();
  REQUIRE(boundary.has_value());
  REQUIRE(std::get<std::int64_t>(boundary->component(0)->value()) == 4);

  // The predicate really did face batch two and prune there, so the equality above is evidence
  // about inclusivity rather than about an unexercised path.
  REQUIRE(stats.top_n_group_prefilter_rows_in.load() == 5);
  REQUIRE(stats.top_n_group_prefilter_rows_out.load() == 3);
}

TEST_CASE("an unarmed grouped aggregate is untouched by the group-key seam",
          "[physical_grouped_aggregate][dynamic_filter][top_n]")
{
  // The feature is dark by default: with no producer installed the operator must aggregate every
  // row of every batch and move no counter.
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space != nullptr);

  std::vector<std::shared_ptr<data_batch>> batches;
  batches.push_back(make_two_column_batch<std::int64_t, std::int64_t>(
    *space, {0, 1, 2, 3, 4, 5}, {1, 2, 3, 4, 5, 6}, cudf::type_id::INT64));
  batches.push_back(make_two_column_batch<std::int64_t, std::int64_t>(
    *space, {4, 9, 0}, {100, 200, 300}, cudf::type_id::INT64));

  auto aggregate  = make_sum_aggregate();
  auto const sums = run_and_merge(*aggregate, batches);
  REQUIRE(sums.at(0) == 301);
  REQUIRE(sums.at(4) == 105);
  REQUIRE(sums.at(9) == 200);
}
