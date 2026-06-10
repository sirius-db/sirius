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
 * @file test_dynamic_filter_merge.cpp
 * @brief Tests for sirius::op::scan::merge_dynamic_filters_into_ast — the helper that AND-merges
 *        AST-capable dynamic filters into a parquet reader's filter tree, resolving consumer
 *        column indices through scan_plan and skipping hive-partition columns.
 */

#include <cudf/ast/expressions.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <catch.hpp>
#include <op/scan/dynamic_filter_merge.hpp>
#include <op/scan/scan_plan.hpp>
#include <op/sirius_dynamic_filter.hpp>

#include <memory>
#include <vector>

using sirius::op::sirius_dynamic_filter;
using sirius::op::sirius_dynamic_filter_kind;
using sirius::op::sirius_dynamic_filter_set;
using sirius::op::sirius_dynamic_zone_map_filter;
using sirius::op::zone_map_entry;
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

/// Build a minimal scan_plan with one DATA column at consumer index @p col_idx named @p name.
scan_plan make_data_only_plan(std::size_t col_idx, std::string name)
{
  scan_plan plan;
  plan.data_columns.push_back({/*primary_idx=*/col_idx, std::move(name)});
  // output_layout must have enough entries to cover col_idx.
  plan.output_layout.resize(col_idx + 1, {scan_plan::output_entry::DATA, 0});
  plan.output_layout[col_idx] = {scan_plan::output_entry::DATA, 0};
  return plan;
}

/// Build a plan where the column at @p col_idx is a hive partition (not in the parquet file).
/// The merge function skips partition columns at the @c output_entry::source check, so we don't
/// need to populate @c partition_columns with a real type.
scan_plan make_partition_plan(std::size_t col_idx)
{
  scan_plan plan;
  plan.output_layout.resize(col_idx + 1, {scan_plan::output_entry::DATA, 0});
  plan.output_layout[col_idx] = {scan_plan::output_entry::PARTITION, 0};
  return plan;
}

/// Filter that inherits the base but NOT the AST mixin — exercises the "lacks capability" skip.
class stub_runtime_only_filter final : public sirius_dynamic_filter {
 public:
  [[nodiscard]] sirius_dynamic_filter_kind kind() const override
  {
    return sirius_dynamic_filter_kind::ZONE_MAP;
  }
};

}  // namespace

TEST_CASE("merge_dynamic_filters_into_ast returns existing_root unchanged for an empty set",
          "[dynamic_filter][scan_merge]")
{
  sirius_dynamic_filter_set filters;  // empty
  auto plan = make_data_only_plan(0, "id");

  cudf::ast::tree tree;
  auto const& base = tree.emplace<cudf::ast::column_name_reference>("static_root_placeholder");

  auto const* root = merge_dynamic_filters_into_ast(tree, &base, filters, plan);

  REQUIRE(root == &base);
  REQUIRE(tree.size() == 1);
}

TEST_CASE("merge_dynamic_filters_into_ast returns nullptr when existing_root is null and set empty",
          "[dynamic_filter][scan_merge]")
{
  sirius_dynamic_filter_set filters;
  auto plan = make_data_only_plan(0, "id");

  cudf::ast::tree tree;
  auto const* root = merge_dynamic_filters_into_ast(tree, nullptr, filters, plan);

  REQUIRE(root == nullptr);
  REQUIRE(tree.size() == 0);
}

TEST_CASE("merge_dynamic_filters_into_ast builds a dynamic-only tree from one filter",
          "[dynamic_filter][scan_merge]")
{
  sirius_dynamic_filter_set filters;
  filters.push_filter(0, make_zone_map(100, 200));
  auto plan = make_data_only_plan(0, "id");

  cudf::ast::tree tree;
  auto const* root = merge_dynamic_filters_into_ast(tree, nullptr, filters, plan);

  REQUIRE(root != nullptr);
  // 1 column_name_reference + 2 literals + 2 comparisons + 1 AND for the single-zone filter.
  REQUIRE(tree.size() == 6);
  REQUIRE(root == &tree.back());
}

TEST_CASE("merge_dynamic_filters_into_ast AND-conjoins dynamic fragment with existing_root",
          "[dynamic_filter][scan_merge]")
{
  sirius_dynamic_filter_set filters;
  filters.push_filter(0, make_zone_map(100, 200));
  auto plan = make_data_only_plan(0, "id");

  cudf::ast::tree tree;
  auto const& base = tree.emplace<cudf::ast::column_name_reference>("static_root_placeholder");
  auto const* root = merge_dynamic_filters_into_ast(tree, &base, filters, plan);

  REQUIRE(root != nullptr);
  REQUIRE(root != &base);
  // 1 base + 1 col_ref + 2 lit + 2 op + 1 AND (filter) + 1 AND (merge with base) = 8.
  REQUIRE(tree.size() == 8);
  REQUIRE(root == &tree.back());
}

TEST_CASE("merge_dynamic_filters_into_ast skips hive-partition columns",
          "[dynamic_filter][scan_merge]")
{
  sirius_dynamic_filter_set filters;
  filters.push_filter(0, make_zone_map(100, 200));
  auto plan = make_partition_plan(0);  // col 0 is a hive partition

  cudf::ast::tree tree;
  auto const* root = merge_dynamic_filters_into_ast(tree, nullptr, filters, plan);

  REQUIRE(root == nullptr);  // nothing contributed
  REQUIRE(tree.size() == 0);
}

TEST_CASE("merge_dynamic_filters_into_ast skips filters lacking the AST capability",
          "[dynamic_filter][scan_merge]")
{
  sirius_dynamic_filter_set filters;
  filters.push_filter(0, std::make_shared<stub_runtime_only_filter>());
  auto plan = make_data_only_plan(0, "id");

  cudf::ast::tree tree;
  auto const* root = merge_dynamic_filters_into_ast(tree, nullptr, filters, plan);

  REQUIRE(root == nullptr);
  REQUIRE(tree.size() == 0);
}

TEST_CASE("merge_dynamic_filters_into_ast AND-conjoins multiple filters across columns",
          "[dynamic_filter][scan_merge]")
{
  sirius_dynamic_filter_set filters;
  filters.push_filter(0, make_zone_map(100, 200));
  filters.push_filter(1, make_zone_map(-5, 5));

  scan_plan plan;
  plan.data_columns.push_back({0, "id"});
  plan.data_columns.push_back({1, "value"});
  plan.output_layout = {{scan_plan::output_entry::DATA, 0}, {scan_plan::output_entry::DATA, 1}};

  cudf::ast::tree tree;
  auto const* root = merge_dynamic_filters_into_ast(tree, nullptr, filters, plan);

  REQUIRE(root != nullptr);
  REQUIRE(root == &tree.back());
  // 2 cols × (1 col_ref + 2 lit + 2 op + 1 AND) + 1 cross-col AND = 13.
  REQUIRE(tree.size() == 13);
}

TEST_CASE("merge_dynamic_filters_into_ast AND-conjoins multiple filters on the same column",
          "[dynamic_filter][scan_merge]")
{
  sirius_dynamic_filter_set filters;
  filters.push_filter(0, make_zone_map(100, 200));
  filters.push_filter(0, make_zone_map(150, 175));
  auto plan = make_data_only_plan(0, "id");

  cudf::ast::tree tree;
  auto const* root = merge_dynamic_filters_into_ast(tree, nullptr, filters, plan);

  REQUIRE(root != nullptr);
  REQUIRE(root == &tree.back());
  // 2 filters × (1 col_ref + 2 lit + 2 op + 1 AND) + 1 AND-merging-the-two = 13.
  REQUIRE(tree.size() == 13);
}

TEST_CASE("merge_dynamic_filters_into_ast ignores out-of-range col_idx defensively",
          "[dynamic_filter][scan_merge]")
{
  sirius_dynamic_filter_set filters;
  filters.push_filter(99, make_zone_map(0, 10));  // col 99 doesn't exist in plan
  auto plan = make_data_only_plan(0, "id");

  cudf::ast::tree tree;
  auto const* root = merge_dynamic_filters_into_ast(tree, nullptr, filters, plan);

  REQUIRE(root == nullptr);
  REQUIRE(tree.size() == 0);
}

//===----------------------------------------------------------------------===//
// apply_dynamic_filters_to_output_table — runtime apply (post-decode / cached)
//===----------------------------------------------------------------------===//

namespace {
/// One INT32 column [0, 1, ..., size-1] wrapped in a single-column table.
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
}  // namespace

TEST_CASE("apply_dynamic_filters_to_output_table drops rows outside the zone",
          "[dynamic_filter][scan_merge]")
{
  auto stream = cudf::get_default_stream();
  auto table  = make_sequence_table(10, stream);  // [0..9]

  auto plan = make_data_only_plan(0, "v");
  sirius_dynamic_filter_set filters;
  filters.push_filter(0, make_zone_map(3, 6));  // inclusive [3,6] keeps 3,4,5,6

  auto out = sirius::op::scan::apply_dynamic_filters_to_output_table(
    std::move(table), filters, plan, stream);
  stream.synchronize();

  REQUIRE(out->num_columns() == 1);
  REQUIRE(out->num_rows() == 4);
}

TEST_CASE("apply_dynamic_filters_to_output_table is a no-op for an empty channel",
          "[dynamic_filter][scan_merge]")
{
  auto stream = cudf::get_default_stream();
  auto table  = make_sequence_table(10, stream);

  auto plan = make_data_only_plan(0, "v");
  sirius_dynamic_filter_set filters;  // empty

  auto out = sirius::op::scan::apply_dynamic_filters_to_output_table(
    std::move(table), filters, plan, stream);
  stream.synchronize();

  REQUIRE(out->num_rows() == 10);  // unchanged
}

TEST_CASE("apply_dynamic_filters_to_output_table skips hive-partition columns",
          "[dynamic_filter][scan_merge]")
{
  auto stream = cudf::get_default_stream();
  auto table  = make_sequence_table(10, stream);

  auto plan = make_partition_plan(0);  // output col 0 is a partition column
  sirius_dynamic_filter_set filters;
  filters.push_filter(0, make_zone_map(3, 6));  // would prune, but col is a partition → skipped

  auto out = sirius::op::scan::apply_dynamic_filters_to_output_table(
    std::move(table), filters, plan, stream);
  stream.synchronize();

  REQUIRE(out->num_rows() == 10);  // unchanged — partition columns are never row-filtered here
}

TEST_CASE("apply_dynamic_filters_to_output_table AND-conjoins multiple zone filters on a column",
          "[dynamic_filter][scan_merge]")
{
  auto stream = cudf::get_default_stream();
  auto table  = make_sequence_table(10, stream);  // [0..9]

  auto plan = make_data_only_plan(0, "v");
  sirius_dynamic_filter_set filters;
  filters.push_filter(0, make_zone_map(2, 8));  // keeps 2..8
  filters.push_filter(0, make_zone_map(5, 9));  // AND keeps 5..9 → intersection 5..8

  auto out = sirius::op::scan::apply_dynamic_filters_to_output_table(
    std::move(table), filters, plan, stream);
  stream.synchronize();

  REQUIRE(out->num_rows() == 4);  // 5,6,7,8
}

//===----------------------------------------------------------------------===//
// Membership filters — IN-list (exact) and Bloom (no false negatives)
//===----------------------------------------------------------------------===//

namespace {
/// One INT64 sequence column [0, 1, ..., size-1] in a single-column table.
std::unique_ptr<cudf::table> make_int64_sequence_table(int64_t size, rmm::cuda_stream_view stream)
{
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(cudf::sequence(static_cast<cudf::size_type>(size),
                                cudf::numeric_scalar<int64_t>(0, true, stream),
                                cudf::numeric_scalar<int64_t>(1, true, stream),
                                stream));
  return std::make_unique<cudf::table>(std::move(cols));
}
}  // namespace

TEST_CASE("sirius_dynamic_in_list_filter keeps exactly the rows whose key is a build key",
          "[dynamic_filter][scan_merge]")
{
  auto stream = cudf::get_default_stream();
  // Build key set {0,1,2,3,4}; probe table [0..9]. Exact membership keeps the first five.
  auto keys = cudf::sequence(5,
                             cudf::numeric_scalar<int32_t>(0, true, stream),
                             cudf::numeric_scalar<int32_t>(1, true, stream),
                             stream);
  sirius_dynamic_filter_set filters;
  filters.push_filter(
    0, std::make_shared<sirius::op::sirius_dynamic_in_list_filter>(std::move(keys)));

  auto plan = make_data_only_plan(0, "v");
  auto out  = sirius::op::scan::apply_dynamic_filters_to_output_table(
    make_sequence_table(10, stream), filters, plan, stream);
  stream.synchronize();
  REQUIRE(out->num_rows() == 5);
}

TEST_CASE("sirius_dynamic_bloom_filter never drops a true match (no false negatives)",
          "[dynamic_filter][scan_merge]")
{
  auto stream = cudf::get_default_stream();
  auto keys   = cudf::sequence(5,
                             cudf::numeric_scalar<int64_t>(0, true, stream),
                             cudf::numeric_scalar<int64_t>(1, true, stream),
                             stream);
  sirius_dynamic_filter_set filters;
  filters.push_filter(0,
                      std::make_shared<sirius::op::sirius_dynamic_bloom_filter>(
                        keys->view(), stream, cudf::get_current_device_resource_ref()));

  auto plan = make_data_only_plan(0, "v");
  auto out  = sirius::op::scan::apply_dynamic_filters_to_output_table(
    make_int64_sequence_table(10, stream), filters, plan, stream);
  stream.synchronize();
  // All five build keys are in the probe, so every one must survive (Bloom has no false negatives).
  // False positives may keep a few extras, so the surviving count is in [5, 10].
  REQUIRE(out->num_rows() >= 5);
  REQUIRE(out->num_rows() <= 10);
}
