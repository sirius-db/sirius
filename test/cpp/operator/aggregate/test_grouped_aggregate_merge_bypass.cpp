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

//! Consumer-side tests for the group-by memory-aware bypass prototype (issue #1746 point 2).
//!
//! These drive the real sirius_physical_grouped_aggregate_merge::get_partition_strategy, so they
//! cover the operator's own classification — aggregate partial-state kinds, the key/aggregate
//! column split, and the downstream walk — rather than re-checking the pure policy's arithmetic
//! (test_group_by_bypass_policy.cpp does that). They need no GPU: the metadata the PARTITION would
//! collect is supplied directly, which is also how the "unknown metadata" cases are expressed.

#include "../operator_type_traits.hpp"
#include "aggregate_test_utils.hpp"
#include "op/aggregate/group_by_bypass_policy.hpp"
#include "op/sirius_physical_grouped_aggregate_merge.hpp"
#include "planner/sirius_physical_plan_generator.hpp"

#include <cudf/types.hpp>

#include <catch.hpp>

#include <memory>
#include <utility>
#include <vector>

using namespace sirius::op;

namespace {

using namespace sirius::test::operator_utils;

/// AUTO is ceil(total_bytes / hash_partition_bytes); this is comfortably above 1 partition at the
/// merge's default target, so every test starts from an AUTO > 1 the prototype could overturn.
constexpr uint64_t kBigInputBytes = 8ULL * 1024 * 1024 * 1024;

group_by_bypass_metadata ample_metadata(std::vector<bypass_column_meta> columns)
{
  group_by_bypass_metadata meta;
  meta.upstream_complete            = true;
  meta.single_gpu_resident          = true;
  meta.distinct_memory_spaces       = 1;
  meta.columns                      = std::move(columns);
  meta.total_rows                   = 1'000'000;
  meta.admissible_additional_budget = 32ULL * 1024 * 1024 * 1024;
  meta.space_capacity               = 48ULL * 1024 * 1024 * 1024;
  meta.target_device_id             = 3;  // deliberately not 0
  meta.num_batches                  = 8;
  meta.total_bytes                  = kBigInputBytes;
  meta.headroom_fraction            = 0.25;
  return meta;
}

partition_sizing_input sizing_input(const group_by_bypass_metadata* meta)
{
  return partition_sizing_input{kBigInputBytes,
                                /*is_build_side=*/false,
                                /*build_foldable=*/false,
                                /*combined_total_bytes=*/kBigInputBytes,
                                meta};
}

/// A merge with `aggregations` over one INT64 group key, wired under a RESULT_COLLECTOR so the
/// downstream walk finds a bounded collection path.
struct merge_fixture {
  explicit merge_fixture(const std::vector<std::string>& aggregations, bool attach_collector = true)
  {
    std::vector<std::size_t> agg_indexes(aggregations.size(), 1);
    auto agg = sirius::test::create_aggregate_expressions<gpu_type_traits<int64_t>>(
      {0}, aggregations, agg_indexes);
    merge = duckdb::make_uniq<sirius_physical_grouped_aggregate_merge>(
      std::move(agg.output_types), std::move(agg.aggregates), std::move(agg.groups), 1'000'000);
    merge->operator_id = 7;
    if (attach_collector) {
      collector = duckdb::make_uniq<sirius_physical_operator>(
        SiriusPhysicalOperatorType::RESULT_COLLECTOR, duckdb::vector<sirius::logical_type>{}, 0);
      collector->operator_id = 8;
      // Stamps merge->_parent_op without descending into the collector (which would try to cast
      // a plain operator to sirius_physical_result_collector).
      sirius::planner::sirius_physical_plan_generator::set_parent_ops(*merge, collector.get());
    }
  }

  duckdb::unique_ptr<sirius_physical_operator> collector;
  duckdb::unique_ptr<sirius_physical_grouped_aggregate_merge> merge;
};

}  // namespace

TEST_CASE("merge bypass selects one partition and requests memory only when eligible",
          "[physical_grouped_aggregate_merge][group_by_bypass]")
{
  merge_fixture f{{"count"}};
  auto meta     = ample_metadata({{static_cast<int>(cudf::type_id::INT64), 8, false},
                                  {static_cast<int>(cudf::type_id::INT64), 8, false}});
  bool enabled  = true;
  bool selected = false;
  int expected =
    natural_num_partitions(kBigInputBytes, sirius::config::DEFAULT_HASH_PARTITION_BYTES, 1);
  SECTION("enabled with ample budget")
  {
    expected = 1;
    selected = true;
  }
  SECTION("disabled") { enabled = false; }
  SECTION("insufficient budget") { meta.admissible_additional_budget = 1; }
  SECTION("incomplete input") { meta.upstream_complete = false; }
  SECTION("input spans devices") { meta.distinct_memory_spaces = 2; }
  SECTION("input is not GPU resident") { meta.single_gpu_resident = false; }
  SECTION("unknown schema") { meta.columns = std::nullopt; }
  SECTION("schema has the wrong column count") { meta.columns->pop_back(); }
  SECTION("unknown budget") { meta.admissible_additional_budget = std::nullopt; }
  SECTION("multi-GPU")
  {
    f.merge->set_num_gpus(4);
    expected =
      natural_num_partitions(kBigInputBytes, sirius::config::DEFAULT_HASH_PARTITION_BYTES, 4);
  }
  auto const result = f.merge->get_partition_strategy(sizing_input(enabled ? &meta : nullptr));
  CHECK(result.num_partitions == expected);
  CHECK_FALSE(result.broadcast);
  CHECK_FALSE(result.build_probe);
  CHECK((f.merge->mandatory_peak_memory_floor({8, kBigInputBytes}) > 0) == selected);
  CHECK(f.merge->mandatory_peak_memory_floor({0, 0}) == 0);
}

TEST_CASE("merge bypass checks physical state types and aggregate operations",
          "[physical_grouped_aggregate_merge][group_by_bypass]")
{
  std::vector<std::string> aggregations{"sum"};
  std::vector<bypass_column_meta> columns{{static_cast<int>(cudf::type_id::INT64), 8, false},
                                          {static_cast<int>(cudf::type_id::INT64), 8, false}};
  SECTION("string grouping key")
  {
    columns[0] = {static_cast<int>(cudf::type_id::STRING), 0, false};
  }
  SECTION("floating-point partial state")
  {
    columns[1].type_id = static_cast<int>(cudf::type_id::FLOAT64);
  }
  SECTION("AVG needs a post-merge divide")
  {
    aggregations = {"avg"};
    columns.push_back(columns.back());
  }
  merge_fixture f{aggregations};
  auto meta = ample_metadata(columns);
  CHECK(f.merge->get_partition_strategy(sizing_input(&meta)).num_partitions > 1);
  CHECK(f.merge->mandatory_peak_memory_floor({8, kBigInputBytes}) == 0);
}

TEST_CASE("merge bypass checks downstream operations and charges a transformed copy",
          "[physical_grouped_aggregate_merge][group_by_bypass]")
{
  auto const meta = ample_metadata({{static_cast<int>(cudf::type_id::INT64), 8, false},
                                    {static_cast<int>(cudf::type_id::INT64), 8, false}});
  SECTION("unknown downstream")
  {
    merge_fixture f{{"sum"}, false};
    CHECK(f.merge->get_partition_strategy(sizing_input(&meta)).num_partitions > 1);
  }
  SECTION("TOP_N")
  {
    merge_fixture f{{"sum"}, false};
    sirius_physical_operator top_n(SiriusPhysicalOperatorType::TOP_N, {}, 0);
    top_n.operator_id = 9;
    sirius::planner::sirius_physical_plan_generator::set_parent_ops(*f.merge, &top_n);
    CHECK(f.merge->get_partition_strategy(sizing_input(&meta)).num_partitions > 1);
  }
  SECTION("projection retains an extra copy")
  {
    merge_fixture bare{{"sum"}};
    REQUIRE(bare.merge->get_partition_strategy(sizing_input(&meta)).num_partitions == 1);
    auto const bare_floor = bare.merge->mandatory_peak_memory_floor({8, kBigInputBytes});
    merge_fixture projected{{"sum"}};
    auto projection = duckdb::make_uniq<sirius_physical_operator>(
      SiriusPhysicalOperatorType::PROJECTION, duckdb::vector<sirius::logical_type>{}, 0);
    auto* merge             = projected.merge.get();
    projection->operator_id = 9;
    projection->children.push_back(std::move(projected.merge));
    sirius::planner::sirius_physical_plan_generator::set_parent_ops(*projection,
                                                                    projected.collector.get());
    REQUIRE(merge->get_partition_strategy(sizing_input(&meta)).num_partitions == 1);
    CHECK(merge->mandatory_peak_memory_floor({8, kBigInputBytes}) > bare_floor);
  }
}

TEST_CASE("an automatic one-partition merge does not gain a bypass memory floor",
          "[physical_grouped_aggregate_merge][group_by_bypass]")
{
  merge_fixture f{{"sum"}};
  auto meta         = ample_metadata({{static_cast<int>(cudf::type_id::INT64), 8, false},
                                      {static_cast<int>(cudf::type_id::INT64), 8, false}});
  auto const result = f.merge->get_partition_strategy({1024, false, false, 1024, &meta});
  CHECK(result.num_partitions == 1);
  CHECK(f.merge->mandatory_peak_memory_floor({1, 1024}) == 0);
}
