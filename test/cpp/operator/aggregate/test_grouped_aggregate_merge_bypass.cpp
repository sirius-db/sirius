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

//! Consumer-side tests for the group-by memory-aware bypass policy.
//!
//! These drive the real sirius_physical_grouped_aggregate_merge::get_partition_strategy, so they
//! cover the operator's own classification — aggregate partial-state kinds, the key/aggregate
//! column split, and the downstream walk — rather than re-checking the pure policy's arithmetic
//! (test_group_by_bypass_policy.cpp does that). They need no GPU: the metadata the PARTITION would
//! collect is supplied directly, which is also how the "unknown metadata" cases are expressed.

#include "../operator_type_traits.hpp"
#include "aggregate_test_utils.hpp"
#include "expression/ast/cast.hpp"
#include "expression/ast/constant.hpp"
#include "expression/ast/node.hpp"
#include "expression/ast/reference.hpp"
#include "op/aggregate/group_by_bypass_policy.hpp"
#include "op/sirius_physical_filter.hpp"
#include "op/sirius_physical_grouped_aggregate_merge.hpp"
#include "op/sirius_physical_projection.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "utils/telemetry_utils.hpp"

#include <cudf/types.hpp>

#include <catch.hpp>

#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

using namespace sirius::op;

namespace {

using namespace sirius::test::operator_utils;

/// AUTO is ceil(total_bytes / hash_partition_bytes); this is comfortably above 1 partition at the
/// merge's default target, so every test starts from an AUTO > 1 the bypass policy could change.
constexpr uint64_t kBigInputBytes = 8ULL * 1024 * 1024 * 1024;

group_by_bypass_metadata ample_metadata(std::vector<bypass_column_meta> columns)
{
  group_by_bypass_metadata meta;
  meta.upstream_complete            = true;
  meta.single_gpu_resident          = true;
  meta.columns                      = std::move(columns);
  meta.total_rows                   = kBigInputBytes / 16;
  meta.admissible_additional_budget = 128ULL * 1024 * 1024 * 1024;
  meta.target_device_id             = 3;  // deliberately not 0
  return meta;
}

/// The PARTITION's lazy metadata source, backed by @p meta; empty (bypass disabled) when null.
std::function<std::optional<group_by_bypass_metadata>()> metadata_source(
  const group_by_bypass_metadata* meta)
{
  if (meta == nullptr) { return {}; }
  return [meta] { return std::optional<group_by_bypass_metadata>{*meta}; };
}

partition_sizing_input sizing_input(const group_by_bypass_metadata* meta)
{
  return partition_sizing_input{kBigInputBytes,
                                /*is_build_side=*/false,
                                /*build_foldable=*/false,
                                /*combined_total_bytes=*/kBigInputBytes,
                                metadata_source(meta)};
}

sirius::logical_type bigint() { return sirius::logical_type::make(sirius::type_id::BIGINT); }

std::unique_ptr<sirius::ast::node> bigint_ref(uint32_t index)
{
  return std::make_unique<sirius::ast::node>(sirius::ast::reference{index, bigint()});
}

/// An evaluated (non-reference) BIGINT expression over @p child.
std::unique_ptr<sirius::ast::node> bigint_cast(std::unique_ptr<sirius::ast::node> child)
{
  return std::make_unique<sirius::ast::node>(sirius::ast::cast{std::move(child), bigint()});
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

class merge_sized_input : public operator_data {
 public:
  explicit merge_sized_input(std::size_t bytes) : bytes_(bytes) {}
  std::size_t get_estimated_size_in_bytes() const override { return bytes_; }

 private:
  std::size_t bytes_;
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
  CHECK((f.merge->no_history_peak_memory_estimate({8, kBigInputBytes}) > 2 * kBigInputBytes) ==
        selected);
  CHECK(f.merge->no_history_peak_memory_estimate({0, 0}) == 0);
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
  CHECK(f.merge->no_history_peak_memory_estimate({8, kBigInputBytes}) == 2 * kBigInputBytes);
}

TEST_CASE("merge bypass rejects downstream operations it cannot size",
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
}

TEST_CASE("merge bypass sizes the columns downstream steps materialize",
          "[physical_grouped_aggregate_merge][group_by_bypass]")
{
  auto const meta = ample_metadata({{static_cast<int>(cudf::type_id::INT64), 8, false},
                                    {static_cast<int>(cudf::type_id::INT64), 8, false}});
  merge_fixture bare{{"sum"}};
  REQUIRE(bare.merge->get_partition_strategy(sizing_input(&meta)).num_partitions == 1);
  auto const bare_estimate = bare.merge->no_history_peak_memory_estimate({8, kBigInputBytes});

  // Wire `parent` between a fresh merge and its collector; return the merge's estimate when the
  // bypass is selected, or nullopt when the downstream is rejected.
  auto estimate_under =
    [&](duckdb::unique_ptr<sirius_physical_operator> parent) -> std::optional<std::size_t> {
    merge_fixture f{{"sum"}};
    auto* merge         = f.merge.get();
    parent->operator_id = 9;
    parent->children.push_back(std::move(f.merge));
    sirius::planner::sirius_physical_plan_generator::set_parent_ops(*parent, f.collector.get());
    if (merge->get_partition_strategy(sizing_input(&meta)).num_partitions != 1) {
      return std::nullopt;
    }
    return merge->no_history_peak_memory_estimate({8, kBigInputBytes});
  };
  auto projection = [](duckdb::vector<std::unique_ptr<sirius::ast::node>> list,
                       duckdb::vector<sirius::logical_type> types) {
    return duckdb::unique_ptr<sirius_physical_operator>(
      duckdb::make_uniq<sirius_physical_projection>(std::move(types), std::move(list), 0));
  };
  // Key and SUM state, then `extra` computed BIGINT columns.
  auto computed = [&](std::size_t extra) {
    duckdb::vector<std::unique_ptr<sirius::ast::node>> list;
    list.push_back(bigint_ref(0));
    list.push_back(bigint_ref(1));
    for (std::size_t i = 0; i < extra; ++i) {
      list.push_back(bigint_cast(bigint_ref(1)));
    }
    return projection(std::move(list), duckdb::vector<sirius::logical_type>(2 + extra, bigint()));
  };

  SECTION("pure references are zero-copy") { CHECK(estimate_under(computed(0)) == bare_estimate); }
  SECTION("each computed column is charged, so a wide projection costs more")
  {
    auto const one  = estimate_under(computed(1));
    auto const wide = estimate_under(computed(8));
    REQUIRE(one.has_value());
    REQUIRE(wide.has_value());
    CHECK(*one > bare_estimate);
    // Eight extra INT64 columns over kBigInputBytes / 16 rows.
    CHECK(*wide - *one >= 7 * 8 * (kBigInputBytes / 16));
  }
  SECTION("a variable-width computed column cannot be bounded")
  {
    duckdb::vector<std::unique_ptr<sirius::ast::node>> list;
    list.push_back(bigint_ref(0));
    list.push_back(std::make_unique<sirius::ast::node>(
      sirius::ast::cast{bigint_ref(1), sirius::logical_type::make(sirius::type_id::VARCHAR)}));
    CHECK_FALSE(
      estimate_under(projection(std::move(list),
                                {bigint(), sirius::logical_type::make(sirius::type_id::VARCHAR)}))
        .has_value());
  }
  SECTION("a reference outside the merge output is rejected")
  {
    duckdb::vector<std::unique_ptr<sirius::ast::node>> list;
    list.push_back(bigint_ref(5));
    CHECK_FALSE(estimate_under(projection(std::move(list), {bigint()})).has_value());
  }
  SECTION("a limit copies its input row")
  {
    auto const limited = estimate_under(duckdb::make_uniq<sirius_physical_operator>(
      SiriusPhysicalOperatorType::LIMIT, duckdb::vector<sirius::logical_type>{}, 0));
    REQUIRE(limited.has_value());
    CHECK(*limited - bare_estimate >= 16 * (kBigInputBytes / 16));
  }
}

TEST_CASE("merge bypass tracks filter output mappings through projection and limit",
          "[physical_grouped_aggregate_merge][group_by_bypass]")
{
  // Narrow physical key plus wide SUM state. References must follow the filter's mapping,
  // even when the logical key type is wider than its physical carrier.
  auto meta = ample_metadata({{static_cast<int>(cudf::type_id::INT8), 1, false},
                              {static_cast<int>(cudf::type_id::INT64), 8, false}});
  std::vector<cudf::size_type> indices{1};
  std::uint64_t filter_width    = 8;
  std::size_t filter_columns    = 1;
  std::uint32_t reference_index = 0;
  bool supported                = true;
  SECTION("drop the narrow key") {}
  SECTION("reorder the wide state before the key")
  {
    indices        = {1, 0};
    filter_width   = 9;
    filter_columns = 2;
  }
  SECTION("passthrough keeps the original indexes")
  {
    indices.clear();
    filter_width    = 9;
    filter_columns  = 2;
    reference_index = 1;
  }
  SECTION("negative index is rejected")
  {
    indices   = {-1};
    supported = false;
  }
  SECTION("out-of-range index is rejected")
  {
    indices   = {2};
    supported = false;
  }

  merge_fixture f{{"sum"}};
  auto* merge    = f.merge.get();
  auto predicate = std::make_unique<sirius::ast::node>(sirius::ast::constant{
    sirius::value{true}, sirius::logical_type::make(sirius::type_id::BOOLEAN)});
  auto filter    = duckdb::make_uniq<sirius_physical_filter>(
    duckdb::vector<sirius::logical_type>(filter_columns, bigint()),
    std::move(predicate),
    0,
    indices);
  filter->children.push_back(std::move(f.merge));
  duckdb::vector<std::unique_ptr<sirius::ast::node>> list;
  for (int i = 0; i < 16; ++i) {
    list.push_back(bigint_ref(reference_index));
  }
  auto projection = duckdb::make_uniq<sirius_physical_projection>(
    duckdb::vector<sirius::logical_type>(16, bigint()), std::move(list), 0);
  projection->children.push_back(std::move(filter));
  auto limit = duckdb::make_uniq<sirius_physical_operator>(
    SiriusPhysicalOperatorType::LIMIT, duckdb::vector<sirius::logical_type>(16, bigint()), 0);
  limit->children.push_back(std::move(projection));
  sirius::planner::sirius_physical_plan_generator::set_parent_ops(*limit, f.collector.get());

  if (!supported) {
    CHECK(merge->get_partition_strategy(sizing_input(&meta)).num_partitions > 1);
    return;
  }
  group_by_bypass::candidate_input expected;
  expected.total_rows           = meta.total_rows;
  expected.key_width_bytes      = 1;
  expected.agg_width_bytes      = 8;
  expected.key_columns          = 1;
  expected.agg_columns          = 1;
  expected.nullable_key_columns = 0;
  expected.nullable_agg_columns = 0;
  expected.downstream_row_bytes = filter_width + 16 * 8;
  expected.downstream_columns   = filter_columns + 16;
  auto const model              = group_by_bypass::model_additional_bytes(expected);
  REQUIRE(merge->get_partition_strategy(sizing_input(&meta)).num_partitions == 1);
  CHECK(merge->no_history_peak_memory_estimate({8, kBigInputBytes}) == model.additional_needed);
  // A budget just below the corrected requirement must preserve normal partitioning.
  meta.admissible_additional_budget = model.required_bytes - 1;
  CHECK(merge->get_partition_strategy(sizing_input(&meta)).num_partitions > 1);
}

TEST_CASE("an automatic one-partition merge keeps its existing memory estimate",
          "[physical_grouped_aggregate_merge][group_by_bypass]")
{
  merge_fixture f{{"sum"}};
  auto meta = ample_metadata({{static_cast<int>(cudf::type_id::INT64), 8, false},
                              {static_cast<int>(cudf::type_id::INT64), 8, false}});
  auto const result =
    f.merge->get_partition_strategy({1024, false, false, 1024, metadata_source(&meta)});
  CHECK(result.num_partitions == 1);
  CHECK(f.merge->no_history_peak_memory_estimate({1, 1024}) == 2048);
}

TEST_CASE("merge bypass skips metadata collection when a cheap gate already rejects it",
          "[physical_grouped_aggregate_merge][group_by_bypass]")
{
  merge_fixture f{{"sum"}};
  auto const meta = ample_metadata({{static_cast<int>(cudf::type_id::INT64), 8, false},
                                    {static_cast<int>(cudf::type_id::INT64), 8, false}});
  int calls       = 0;
  auto source     = [&] {
    ++calls;
    return std::optional<group_by_bypass_metadata>{meta};
  };
  SECTION("automatic count is already one")
  {
    CHECK(f.merge->get_partition_strategy({1024, false, false, 1024, source}).num_partitions == 1);
  }
  SECTION("multi-GPU")
  {
    f.merge->set_num_gpus(4);
    CHECK(f.merge->get_partition_strategy({kBigInputBytes, false, false, kBigInputBytes, source})
            .num_partitions > 1);
  }
  CHECK(calls == 0);
}

TEST_CASE("bypass uses the query policy snapshot for headroom",
          "[physical_grouped_aggregate_merge][group_by_bypass]")
{
  auto meta = ample_metadata({{static_cast<int>(cudf::type_id::INT64), 8, false},
                              {static_cast<int>(cudf::type_id::INT64), 8, false}});
  merge_fixture baseline{{"sum"}};
  REQUIRE(baseline.merge->get_partition_strategy(sizing_input(&meta)).num_partitions == 1);
  auto const needed = baseline.merge->no_history_peak_memory_estimate({8, kBigInputBytes});
  meta.admissible_additional_budget = needed * 2;

  merge_fixture custom{{"sum"}};
  auto params                               = std::make_shared<sirius::operator_params>();
  params->group_by_bypass_headroom_fraction = 4.0;
  sirius::pipeline::pipeline_build_context ctx{nullptr, true, 1, params};
  custom.merge->set_pipeline(std::make_shared<sirius::pipeline::sirius_pipeline>(ctx));
  CHECK(custom.merge->get_partition_strategy(sizing_input(&meta)).num_partitions > 1);

  // Headroom changes eligibility, not the cold-start allocation estimate. Once the budget
  // includes the configured slack, selection still publishes the same unpadded model.
  meta.admissible_additional_budget = needed * 5;
  REQUIRE(custom.merge->get_partition_strategy(sizing_input(&meta)).num_partitions == 1);
  CHECK(custom.merge->no_history_peak_memory_estimate({8, kBigInputBytes}) == needed);
}

TEST_CASE("bypass reservation uses cold estimation and existing OOM retry state",
          "[physical_grouped_aggregate_merge][group_by_bypass][reservation]")
{
  using namespace sirius::pipeline;
  merge_fixture f{{"sum"}};
  auto const meta = ample_metadata({{static_cast<int>(cudf::type_id::INT64), 8, false},
                                    {static_cast<int>(cudf::type_id::INT64), 8, false}});
  REQUIRE(f.merge->get_partition_strategy(sizing_input(&meta)).num_partitions == 1);
  auto const model = f.merge->no_history_peak_memory_estimate({8, kBigInputBytes});
  REQUIRE(model > 2 * kBigInputBytes);

  pipeline_build_context ctx{nullptr, true};
  auto pipeline = std::make_shared<sirius_pipeline>(ctx);
  sirius_pipeline_build_state build_state;
  build_state.set_pipeline_source(*pipeline, *f.merge);
  build_state.set_pipeline_operators(*pipeline, {*f.merge, *f.collector});
  build_state.set_pipeline_sink(*pipeline, *f.collector, 0);
  auto global = std::make_shared<sirius_pipeline_task_global_state>(
    pipeline, sirius::test::make_test_telemetry_context());

  auto local = std::make_unique<gpu_pipeline_task_local_state>(
    std::make_unique<merge_sized_input>(kBigInputBytes));
  auto* original = local.get();
  gpu_pipeline_task task(1, {}, std::move(local), global);
  auto const first = task.get_estimated_reservation_size_info(nullptr);
  REQUIRE_FALSE(first.had_history);
  CHECK(first.peak_memory_estimate == model);
  CHECK(first.reservation_size == model);

  std::size_t resume_index = 0;
  std::size_t retry_input  = kBigInputBytes;
  SECTION("retry at the merge keeps the original input") {}
  SECTION("retry after a fused merge starts from its output")
  {
    resume_index = 1;
    retry_input  = 1024;
  }
  // Failed tasks populate this query's history, while the existing retry floor grows from the
  // granted reservation and survives the executor's replacement of local state.
  global->get_memory_history().record_on_failure(kBigInputBytes, model / 2);
  original->update_retry_reservation_floor_after_oom(model, model / 2, 1024);
  auto retry_local = std::make_unique<gpu_pipeline_task_local_state>(
    std::make_unique<merge_sized_input>(retry_input), resume_index);
  retry_local->inherit_retry_reservation_floor(*original);
  auto retry      = task.create_rescheduled_task(2, std::move(retry_local));
  auto const info = retry->get_estimated_reservation_size_info(nullptr);
  CHECK(info.had_history);
  CHECK(info.retry_reservation_floor == 2 * model);
  CHECK(info.reservation_size == info.retry_reservation_floor);

  // A separate query owns a fresh pipeline and therefore starts with no learned samples.
  auto next_pipeline = std::make_shared<sirius_pipeline>(ctx);
  CHECK(next_pipeline->get_memory_history().size() == 0);
}
