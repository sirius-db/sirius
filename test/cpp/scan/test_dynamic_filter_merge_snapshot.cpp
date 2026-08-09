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
 * @file test_dynamic_filter_merge_snapshot.cpp
 * @brief Stage-2 coverage beside `test_dynamic_filter_merge.cpp` (which stays byte-unchanged as
 *        the migration safety net): the gate's generation domain, count/generation domain
 *        separation across instances, parity of the snapshot-consuming merge/apply overloads
 *        with their set-based spellings, and generation-driven re-arm through a refinement-slot
 *        replacement.
 */

#include <cudf/ast/expressions.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda_runtime.h>

#include <catch.hpp>
#include <op/dynamic_filter/exact_host_scalar.hpp>
#include <op/dynamic_filter/sirius_dynamic_filter.hpp>
#include <op/scan/dynamic_filter_merge.hpp>
#include <op/scan/scan_plan.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

using sirius::op::dynamic_filter_snapshot;
using sirius::op::refinement_publish_result;
using sirius::op::sirius_dynamic_filter_set;
using sirius::op::sirius_dynamic_zone_map_filter;
using sirius::op::zone_map_entry;
using sirius::op::scan::dynamic_filter_apply_mode;
using sirius::op::scan::dynamic_filter_gate;
using sirius::op::scan::merge_dynamic_filters_into_ast;
using sirius::op::scan::scan_plan;

namespace {

std::unique_ptr<cudf::scalar> make_int32_scalar(int32_t v)
{
  return std::make_unique<cudf::numeric_scalar<int32_t>>(
    v, true, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
}

std::shared_ptr<sirius_dynamic_zone_map_filter> make_zone_map(int32_t lo, int32_t hi)
{
  std::vector<zone_map_entry> zones;
  zones.push_back({make_int32_scalar(lo), make_int32_scalar(hi)});
  return std::make_shared<sirius_dynamic_zone_map_filter>(std::move(zones));
}

scan_plan make_data_only_plan(std::size_t col_idx, std::string name)
{
  scan_plan plan;
  plan.data_columns.push_back({/*primary_idx=*/col_idx, std::move(name)});
  plan.output_layout.resize(col_idx + 1, {scan_plan::output_entry::DATA, 0});
  plan.output_layout[col_idx] = {scan_plan::output_entry::DATA, 0};
  return plan;
}

std::unique_ptr<cudf::table> make_sequence_table(int32_t size, rmm::cuda_stream_view stream)
{
  auto col = cudf::sequence(size,
                            cudf::numeric_scalar<int32_t>(0, true, stream),
                            cudf::numeric_scalar<int32_t>(1, true, stream),
                            stream);
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(col));
  return std::make_unique<cudf::table>(std::move(cols));
}

std::vector<int32_t> to_host_int32(cudf::column_view const& col, rmm::cuda_stream_view stream)
{
  std::vector<int32_t> host(static_cast<std::size_t>(col.size()));
  cudaMemcpyAsync(host.data(),
                  col.data<int32_t>(),
                  host.size() * sizeof(int32_t),
                  cudaMemcpyDeviceToHost,
                  stream.value());
  stream.synchronize();
  return host;
}

/// Column-free snapshot carrying only the two gate-relevant signals.
dynamic_filter_snapshot gate_signal(std::uint64_t generation, std::size_t logical_count)
{
  return {.generation = generation, .logical_filter_count = logical_count, .columns = {}};
}

}  // namespace

TEST_CASE("gate generation domain: a disable decision re-arms exactly once per newer generation",
          "[dynamic_filter][scan_merge]")
{
  dynamic_filter_gate gate{/*keep_threshold=*/0.5};

  REQUIRE_FALSE(gate.applicable(gate_signal(1, 0)));  // no filters -> no work
  REQUIRE(gate.applicable(gate_signal(1, 1)));        // unmeasured -> measure

  gate.record_keep_ratio(10, 9, /*observed_marker=*/1);  // 0.9 > 0.5 -> disabled at generation 1
  REQUIRE_FALSE(gate.applicable(gate_signal(1, 1)));
  REQUIRE(gate.applicable(gate_signal(2, 1)));  // newer generation -> one re-measurement
  gate.record_keep_ratio(10, 8, /*observed_marker=*/2);
  REQUIRE_FALSE(gate.applicable(gate_signal(2, 1)));  // disabled again at generation 2

  // An older completing measurement cannot overwrite the newer-generation decision.
  gate.record_keep_ratio(10, 1, /*observed_marker=*/1);
  REQUIRE_FALSE(gate.applicable(gate_signal(2, 1)));

  // A selective measurement at a newer generation makes ACTIVE, which is terminal.
  gate.record_keep_ratio(10, 1, /*observed_marker=*/3);
  REQUIRE(gate.applicable(gate_signal(3, 1)));
  gate.record_keep_ratio(10, 10, /*observed_marker=*/4);  // cannot demote ACTIVE
  REQUIRE(gate.applicable(gate_signal(4, 1)));
}

TEST_CASE("gate count domain keeps its Stage-1 behavior on its own instance",
          "[dynamic_filter][scan_merge]")
{
  // One marker domain per instance: this gate only ever observes update counts, exactly like the
  // Top-N prefilter gate; the generation-domain instance above never observes counts.
  dynamic_filter_gate gate{/*keep_threshold=*/0.5};

  REQUIRE_FALSE(gate.applicable(std::size_t{0}));
  REQUIRE(gate.applicable(std::size_t{1}));
  gate.record_keep_ratio(10, 9, std::size_t{1});
  REQUIRE_FALSE(gate.applicable(std::size_t{1}));
  REQUIRE(gate.applicable(std::size_t{2}));
  gate.record_keep_ratio(10, 1, std::size_t{2});
  REQUIRE(gate.applicable(std::size_t{2}));  // ACTIVE, terminal
}

TEST_CASE("merge_dynamic_filters_into_ast lowers a snapshot's filters into the tree",
          "[dynamic_filter][scan_merge]")
{
  // Stage 4 retired the set-based spelling, so there is no second form left to compare against;
  // what remains worth pinning is that the snapshot form lowers into the caller's tree and
  // returns its root. The full derivation-level checks live in the filter suites.
  auto filters = std::make_shared<sirius_dynamic_filter_set>();
  filters->push_filter(0, make_zone_map(3, 6));
  auto plan = make_data_only_plan(0, "id");

  cudf::ast::tree tree;
  auto const* root = merge_dynamic_filters_into_ast(tree, nullptr, filters->snapshot(), plan);

  REQUIRE(root != nullptr);
  REQUIRE(root == &tree.back());
  REQUIRE(tree.size() > 1);
}

TEST_CASE("apply_dynamic_filters_to_view drops the rows a snapshot's zone map excludes",
          "[dynamic_filter][scan_merge]")
{
  auto stream = cudf::get_default_stream();
  auto table  = make_sequence_table(10, stream);  // [0..9]

  auto filters = std::make_shared<sirius_dynamic_filter_set>();
  filters->push_filter(0, make_zone_map(3, 6));

  auto filtered =
    sirius::op::scan::apply_dynamic_filters_to_view(table->view(), filters->snapshot(), stream);
  stream.synchronize();

  REQUIRE(filtered != nullptr);
  REQUIRE(to_host_int32(filtered->view().column(0), stream) == std::vector<int32_t>{3, 4, 5, 6});
  REQUIRE(table->num_rows() == 10);  // input untouched
}

TEST_CASE("gated view re-arms once when a refinement replacement advances the generation",
          "[dynamic_filter][scan_merge]")
{
  auto stream = cudf::get_default_stream();
  auto table  = make_sequence_table(10, stream);  // [0..9]

  auto filters   = std::make_shared<sirius_dynamic_filter_set>();
  auto publisher = filters->register_refinement_slot(0);
  REQUIRE(publisher.publish(1, make_zone_map(0, 9)) == refinement_publish_result::ACCEPTED);

  dynamic_filter_gate gate;  // default keep threshold 0.9

  // Measured batch keeps everything (10/10): the gate disables at the snapshot's generation.
  auto first = sirius::op::scan::apply_dynamic_filters_gated_view(
    table->view(), filters->snapshot(), gate, stream, dynamic_filter_apply_mode::include_ast_row_masks);
  REQUIRE(first != nullptr);
  REQUIRE(first->num_rows() == 10);
  auto second = sirius::op::scan::apply_dynamic_filters_gated_view(
    table->view(), filters->snapshot(), gate, stream, dynamic_filter_apply_mode::include_ast_row_masks);
  REQUIRE(second == nullptr);  // disabled; filter_count did not change

  // Replacement bumps only the generation -- exactly the change the count-based rule would miss.
  REQUIRE(publisher.publish(2, make_zone_map(3, 6)) == refinement_publish_result::ACCEPTED);
  REQUIRE(filters->filter_count() == 1);
  auto rearmed = sirius::op::scan::apply_dynamic_filters_gated_view(
    table->view(), filters->snapshot(), gate, stream, dynamic_filter_apply_mode::include_ast_row_masks);
  stream.synchronize();
  REQUIRE(rearmed != nullptr);  // one re-measurement: 4/10 kept -> ACTIVE
  REQUIRE(to_host_int32(rearmed->view().column(0), stream) == std::vector<int32_t>{3, 4, 5, 6});
}

//===----------------------------------------------------------------------===//
// Parquet reader-AST lowering of a multi-column LEX boundary
//===----------------------------------------------------------------------===//

namespace {

/// A scan_plan whose output ordinals map to the given parquet column names, with `partition_at`
/// (when set) marked as a hive-partition column rather than a decoded one.
scan_plan make_named_plan(std::vector<std::string> const& names,
                          std::optional<std::size_t> partition_at = std::nullopt)
{
  scan_plan plan;
  plan.output_layout.resize(names.size(), {scan_plan::output_entry::DATA, 0});
  for (std::size_t i = 0; i < names.size(); ++i) {
    if (partition_at && *partition_at == i) {
      plan.output_layout[i] = {scan_plan::output_entry::PARTITION, 0};
      continue;
    }
    plan.output_layout[i] = {scan_plan::output_entry::DATA, plan.data_columns.size()};
    plan.data_columns.push_back({/*primary_idx=*/i, names[i]});
  }
  return plan;
}

/// Every column name the lowered tree references, in emission order.
std::vector<std::string> referenced_column_names(cudf::ast::tree const& tree)
{
  std::vector<std::string> names;
  for (cudf::size_type i = 0; i < static_cast<cudf::size_type>(tree.size()); ++i) {
    if (auto const* ref = dynamic_cast<cudf::ast::column_name_reference const*>(&tree[i])) {
      names.push_back(ref->get_column_name());
    }
  }
  return names;
}

std::shared_ptr<sirius::op::sirius_dynamic_lex_range_filter> make_lex_filter(
  std::size_t primary_ordinal, std::size_t tail_ordinal)
{
  auto const key = sirius::op::top_n_key_semantics{.storage_type = cudf::data_type{cudf::type_id::INT32},
                                                   .order      = cudf::order::ASCENDING,
                                                   .null_order = cudf::null_order::AFTER};
  std::vector<sirius::op::lex_component_semantics> components{
    {.consumer_ordinal = primary_ordinal, .key = key},
    {.consumer_ordinal = tail_ordinal, .key = key}};
  std::vector<std::optional<sirius::op::exact_host_scalar>> boundary{
    sirius::op::exact_host_scalar{std::int32_t{5}, cudf::data_type{cudf::type_id::INT32}},
    sirius::op::exact_host_scalar{std::int32_t{7}, cudf::data_type{cudf::type_id::INT32}}};
  return std::make_shared<sirius::op::sirius_dynamic_lex_range_filter>(
    sirius::op::exact_host_key_tuple{boundary}, std::move(components), /*inclusive=*/false);
}

}  // namespace

TEST_CASE("a LEX boundary lowers into the parquet reader AST against its own component columns",
          "[dynamic_filter][scan_merge]")
{
  // The consumer ordinals are deliberately neither 0/1 nor adjacent, so a defect that lowered
  // against positional order, the primary column twice, or the wrong plan entry would name
  // different columns than the two asserted here.
  auto const plan = make_named_plan({"a", "b", "key_primary", "c", "key_tail"});
  auto filters    = std::make_shared<sirius_dynamic_filter_set>();
  auto publisher  = filters->register_refinement_slot(2, {4});
  REQUIRE(publisher.publish(1, make_lex_filter(2, 4)) == refinement_publish_result::ACCEPTED);

  cudf::ast::tree tree;
  auto const* root = merge_dynamic_filters_into_ast(tree, nullptr, filters->snapshot(), plan);
  REQUIRE(root != nullptr);

  auto const names = referenced_column_names(tree);
  REQUIRE_FALSE(names.empty());
  // Exactly the two component columns are referenced -- never a third, never the primary alone.
  std::set<std::string> const distinct(names.begin(), names.end());
  REQUIRE(distinct == std::set<std::string>{"key_primary", "key_tail"});
}

TEST_CASE("a LEX boundary whose component is a hive partition is skipped whole",
          "[dynamic_filter][scan_merge]")
{
  // The tail lands on a partition column, whose values never reach the reader, so the filter
  // cannot be lowered at all -- not partially against its primary.
  auto const plan = make_named_plan({"a", "b", "key_primary", "c", "key_tail"}, /*partition_at=*/4);
  auto filters    = std::make_shared<sirius_dynamic_filter_set>();
  auto publisher  = filters->register_refinement_slot(2, {4});
  REQUIRE(publisher.publish(1, make_lex_filter(2, 4)) == refinement_publish_result::ACCEPTED);

  cudf::ast::tree tree;
  auto const* root = merge_dynamic_filters_into_ast(tree, nullptr, filters->snapshot(), plan);
  REQUIRE(root == nullptr);
  REQUIRE(referenced_column_names(tree).empty());
}
