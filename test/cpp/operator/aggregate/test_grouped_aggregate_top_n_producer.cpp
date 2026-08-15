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

#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>

#include <catch.hpp>
#include <op/cudf_sort_order.hpp>
#include <op/dynamic_filter/dynamic_filter_stats.hpp>
#include <op/dynamic_filter/top_n_group_key_producer.hpp>
#include <op/dynamic_filter/top_n_threshold_coordinator.hpp>
#include <op/sirius_physical_grouped_aggregate.hpp>

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
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
  auto expressions =
    sirius::test::create_aggregate_expressions<gpu_type_traits<std::int64_t>>({0}, {"sum"}, {1});
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
    auto outputs = aggregate.execute(pipelineable_operator_data({batch}),
                                     sirius::test::operator_utils::default_stream());
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
  // and the boundary becomes the fifth-best key, 4 -- and witness-first means batch one then
  // prunes itself down to the five boundary-or-better rows before its own hash insert. Batch two
  // then meets that boundary holding rows *tied* with it. Inclusive keeps them and group 4's sum
  // is exact; strict drops all three thousand of their contribution while still returning a row
  // for group 4 -- the silent corruption. Group 9's rows are strictly worse and may legitimately
  // be pruned.
  std::vector<std::shared_ptr<data_batch>> batches;
  batches.push_back(make_two_column_batch<std::int64_t, std::int64_t>(
    *space,
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
    {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22},
    cudf::type_id::INT64));
  batches.push_back(make_two_column_batch<std::int64_t, std::int64_t>(
    *space, {4, 4, 9, 0, 9}, {1000, 2000, 5, 7, 6}, cudf::type_id::INT64));

  auto reference      = make_sum_aggregate();
  auto const expected = run_and_merge(*reference, batches);

  dynamic_filter_stats stats;
  auto coordinator = std::make_shared<top_n_threshold_coordinator>(
    5, ascending_int64_key(), true, &stats, top_n_producer_kind::GROUP_KEY);
  auto armed = make_sum_aggregate();
  armed->top_n_producer =
    std::make_unique<top_n_group_key_producer>(coordinator,
                                               std::vector<cudf::size_type>{0},
                                               scan::dynamic_filter_gate::k_default_keep_threshold);
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

  // The predicate really did prune, in both places witness-first makes reachable: batch one
  // self-prunes (13 in, 5 boundary-or-better out) and batch two prunes its two strictly-worse
  // rows (5 in, 3 out) -- so the equality above is evidence about inclusivity rather than about
  // an unexercised path.
  REQUIRE(stats.top_n_group_prefilter_rows_in.load() == 18);
  REQUIRE(stats.top_n_group_prefilter_rows_out.load() == 8);
}

TEST_CASE("group-key prefilter keep-ratio disable stops measuring while witnessing continues",
          "[physical_grouped_aggregate][dynamic_filter][top_n]")
{
  // The group-path twin of the row producer's disable pin (test_physical_top_n_prefilter.cpp):
  // deleting or mis-wiring the producer's record_keep_ratio/record_prefilter_disabled calls
  // leaves the gate undecided forever, which no selective-workload test can see.
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space != nullptr);

  dynamic_filter_stats stats;
  auto coordinator = std::make_shared<top_n_threshold_coordinator>(
    5, ascending_int64_key(), true, &stats, top_n_producer_kind::GROUP_KEY);
  auto armed = make_sum_aggregate();
  // keep_threshold 0.0: any measured batch that keeps a row disables the prefilter.
  armed->top_n_producer = std::make_unique<top_n_group_key_producer>(
    coordinator, std::vector<cudf::size_type>{0}, /*gate_keep_threshold=*/0.0);

  auto const execute = [&](std::vector<std::int64_t> const& keys,
                           std::vector<std::int64_t> const& values) {
    std::vector<std::shared_ptr<data_batch>> batch;
    batch.push_back(make_two_column_batch<std::int64_t, std::int64_t>(
      *space, keys, values, cudf::type_id::INT64));
    (void)run_and_merge(*armed, batch);
  };

  // Batch 1: witness-first fills the K = 5 set from thirteen distinct keys, so the boundary (4)
  // exists when this same batch's prefilter runs -- the one measured batch. Keeping 5 of 13 rows
  // exceeds threshold 0.0, so the gate disables and the shared disable counter fires.
  execute({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
          {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22});
  REQUIRE(stats.top_n_group_offers.load() == 1);
  REQUIRE(stats.top_n_group_witness_set_full.load() == 1);
  REQUIRE(stats.top_n_group_prefilter_rows_in.load() == 13);
  REQUIRE(stats.top_n_group_prefilter_rows_out.load() == 5);
  REQUIRE(stats.top_n_prefilter_disabled.load() == 1);

  // Batch 2: keys no better than the boundary, so witnessing continues (the offer merges) but
  // tightens nothing -- the boundary-update count is unchanged and the disabled prefilter
  // declines. rows_in frozen is the disable made observable.
  execute({5, 9, 11}, {100, 200, 300});
  REQUIRE(stats.top_n_group_offers.load() == 2);
  REQUIRE(stats.top_n_group_prefilter_rows_in.load() == 13);
  REQUIRE(stats.top_n_group_prefilter_rows_out.load() == 5);
  REQUIRE(stats.top_n_prefilter_disabled.load() == 1);
}

TEST_CASE("the group-key producer witnesses DECIMAL128 keys exactly at width 16",
          "[physical_grouped_aggregate][dynamic_filter][top_n]")
{
  // The cudf::distinct + cudf::sort at-DECIMAL128 verification, plus the 16-byte D2H staging and
  // read_element arm: the producer's witness path computes the batch's K best *distinct* keys on
  // device and reads them back as exact host values. Every discriminating rep sits outside
  // int64's range, so a staging stride, load, or host comparison that touched only 8 of the 16
  // bytes would misorder the set and land the wrong boundary.
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space != nullptr);
  auto const stream = sirius::test::operator_utils::default_stream();
  auto const mr     = sirius::test::operator_utils::get_resource_ref(*space);

  constexpr __int128_t k_ten_pow_19 = __int128_t{10} * 1'000'000'000'000'000'000LL;
  constexpr __int128_t k_two_pow_64 = __int128_t{1} << 64;
  auto const k_dec128               = cudf::data_type{cudf::type_id::DECIMAL128, -4};

  auto const make_key_column = [&](std::vector<std::optional<__int128_t>> const& values) {
    auto const size = static_cast<cudf::size_type>(values.size());
    auto null_mask  = cudf::create_null_mask(size, cudf::mask_state::ALL_VALID, stream, mr);
    auto* mask_ptr  = static_cast<cudf::bitmask_type*>(null_mask.data());
    cudf::size_type null_count = 0;
    std::vector<__int128_t> reps(values.size(), 0);
    for (cudf::size_type i = 0; i < size; ++i) {
      if (values[static_cast<std::size_t>(i)]) {
        reps[static_cast<std::size_t>(i)] = *values[static_cast<std::size_t>(i)];
      } else {
        cudf::set_null_mask(mask_ptr, i, i + 1, false, stream);
        ++null_count;
      }
    }
    auto column =
      cudf::make_fixed_point_column(k_dec128, size, std::move(null_mask), null_count, stream, mr);
    cudaMemcpy(column->mutable_view().data<__int128_t>(),
               reps.data(),
               reps.size() * sizeof(__int128_t),
               cudaMemcpyHostToDevice);
    return column;
  };

  dynamic_filter_stats stats;
  auto coordinator = std::make_shared<top_n_threshold_coordinator>(
    2,
    std::vector<top_n_key_semantics>{{.storage_type = k_dec128,
                                      .order        = cudf::order::ASCENDING,
                                      .null_order   = cudf::null_order::AFTER}},
    true,
    &stats,
    top_n_producer_kind::GROUP_KEY);
  top_n_group_key_producer producer{coordinator,
                                    std::vector<cudf::size_type>{0},
                                    scan::dynamic_filter_gate::k_default_keep_threshold};

  // Batch one: duplicates, nulls, and a negative below -int64-max. The distinct keys sort to
  // -10^19, -1, 3 ascending (nulls last), so the K = 2 witness set is {-10^19, -1} and the
  // boundary is the second-best distinct key, -1. The pairing is truncation-inverting: -10^19's
  // low 64 bits read as +8.4e18, which orders *after* -1 -- a host comparison that dropped the
  // high 8 bytes would land -10^19 as the Kth-best instead.
  auto const key1 = make_key_column({-k_ten_pow_19, -1, -k_ten_pow_19, std::nullopt, 3});
  producer.witness(cudf::table_view{{key1->view()}}, stream, mr);
  REQUIRE(stats.top_n_group_offers.load() == 1);
  REQUIRE(stats.top_n_group_witness_set_full.load() == 1);
  auto boundary = coordinator->tightest_boundary();
  REQUIRE(boundary.has_value());
  REQUIRE(std::get<__int128_t>(boundary->component(0)->value()) == -1);

  // Batch two tightens the boundary with a still-better wide key: -(1<<64) < -10^19, so the
  // union's best two are both below -int64-max and the boundary becomes the wide value -10^19 --
  // the exact rep the assertion pins is a 16-byte D2H staging read.
  auto const key2 = make_key_column({-k_two_pow_64, std::nullopt, -k_two_pow_64});
  producer.witness(cudf::table_view{{key2->view()}}, stream, mr);
  REQUIRE(stats.top_n_group_offers.load() == 2);
  boundary = coordinator->tightest_boundary();
  REQUIRE(boundary.has_value());
  REQUIRE(std::get<__int128_t>(boundary->component(0)->value()) == -k_ten_pow_19);
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
