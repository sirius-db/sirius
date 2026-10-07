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

// libcudf's AST header uses std::variant without including <variant>.
// clang-format off
#include <variant>
#include <cudf/aggregation.hpp>
#include <cudf/ast/expressions.hpp>
// clang-format on
#include "utils/test_validation_utility.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>

#include <cuda/stream>
#include <cuda_runtime.h>

#include <catch.hpp>
#include <op/dynamic_filter/sirius_dynamic_filter.hpp>
#include <op/scan/dynamic_filter_merge.hpp>
#include <op/scan/scan_plan.hpp>
#include <op/scan/sirius_gpu_scan_operator_data.hpp>

#include <atomic>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <future>
#include <initializer_list>
#include <latch>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <vector>

using sirius::op::sirius_dynamic_filter;
using sirius::op::sirius_dynamic_filter_kind;
using sirius::op::sirius_dynamic_filter_set;
using sirius::op::sirius_dynamic_zone_map_filter;
using sirius::op::zone_map_entry;
using sirius::op::scan::dynamic_filter_apply_mode;
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

/// Zone-map with explicit boundary inclusivity, so the exclusive GREATER / LESS lowering branches
/// (sirius_dynamic_filter.cpp) get exercised — the default helper above only builds inclusive
/// bounds.
std::shared_ptr<sirius_dynamic_zone_map_filter> make_zone_map(int32_t lo,
                                                              int32_t hi,
                                                              bool inclusive_min,
                                                              bool inclusive_max)
{
  std::vector<zone_map_entry> zones;
  zones.push_back({make_int32_scalar(lo), make_int32_scalar(hi)});
  return std::make_shared<sirius_dynamic_zone_map_filter>(
    std::move(zones), inclusive_min, inclusive_max);
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

  auto const snapshot = filters.snapshot();
  auto const* root    = merge_dynamic_filters_into_ast(tree, &base, snapshot, plan);

  REQUIRE(root == &base);
  REQUIRE(tree.size() == 1);
}

TEST_CASE("merge_dynamic_filters_into_ast returns nullptr when existing_root is null and set empty",
          "[dynamic_filter][scan_merge]")
{
  sirius_dynamic_filter_set filters;
  auto plan = make_data_only_plan(0, "id");

  cudf::ast::tree tree;
  auto const snapshot = filters.snapshot();
  auto const* root    = merge_dynamic_filters_into_ast(tree, nullptr, snapshot, plan);

  REQUIRE(root == nullptr);
  REQUIRE(tree.size() == 0);
}

TEST_CASE("merge_dynamic_filters_into_ast builds a dynamic-only tree from one filter",
          "[dynamic_filter][scan_merge]")
{
  sirius_dynamic_filter_set filters;
  auto filters_producer = filters.register_producer({0});
  REQUIRE(filters_producer.push_filter(0, make_zone_map(100, 200)));
  auto plan = make_data_only_plan(0, "id");

  cudf::ast::tree tree;
  auto const snapshot = filters.snapshot();
  auto const* root    = merge_dynamic_filters_into_ast(tree, nullptr, snapshot, plan);

  REQUIRE(root != nullptr);
  // 1 column_name_reference + 2 literals + 2 comparisons + 1 AND for the single-zone filter.
  REQUIRE(tree.size() == 6);
  REQUIRE(root == &tree.back());
}

TEST_CASE("merge_dynamic_filters_into_ast AND-conjoins dynamic fragment with existing_root",
          "[dynamic_filter][scan_merge]")
{
  sirius_dynamic_filter_set filters;
  auto filters_producer = filters.register_producer({0});
  REQUIRE(filters_producer.push_filter(0, make_zone_map(100, 200)));
  auto plan = make_data_only_plan(0, "id");

  cudf::ast::tree tree;
  auto const& base    = tree.emplace<cudf::ast::column_name_reference>("static_root_placeholder");
  auto const snapshot = filters.snapshot();
  auto const* root    = merge_dynamic_filters_into_ast(tree, &base, snapshot, plan);

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
  auto filters_producer = filters.register_producer({0});
  REQUIRE(filters_producer.push_filter(0, make_zone_map(100, 200)));
  auto plan = make_partition_plan(0);  // col 0 is a hive partition

  cudf::ast::tree tree;
  auto const snapshot = filters.snapshot();
  auto const* root    = merge_dynamic_filters_into_ast(tree, nullptr, snapshot, plan);

  REQUIRE(root == nullptr);  // nothing contributed
  REQUIRE(tree.size() == 0);
}

TEST_CASE("merge_dynamic_filters_into_ast skips filters lacking the AST capability",
          "[dynamic_filter][scan_merge]")
{
  sirius_dynamic_filter_set filters;
  auto filters_producer = filters.register_producer({0});
  REQUIRE(filters_producer.push_filter(0, std::make_shared<stub_runtime_only_filter>()));
  auto plan = make_data_only_plan(0, "id");

  cudf::ast::tree tree;
  auto const snapshot = filters.snapshot();
  auto const* root    = merge_dynamic_filters_into_ast(tree, nullptr, snapshot, plan);

  REQUIRE(root == nullptr);
  REQUIRE(tree.size() == 0);
}

TEST_CASE("merge_dynamic_filters_into_ast AND-conjoins multiple filters across columns",
          "[dynamic_filter][scan_merge]")
{
  sirius_dynamic_filter_set filters;
  auto filters_producer = filters.register_producer({0, 1});
  REQUIRE(filters_producer.push_filter(0, make_zone_map(100, 200)));
  REQUIRE(filters_producer.push_filter(1, make_zone_map(-5, 5)));

  scan_plan plan;
  plan.data_columns.push_back({0, "id"});
  plan.data_columns.push_back({1, "value"});
  plan.output_layout = {{scan_plan::output_entry::DATA, 0}, {scan_plan::output_entry::DATA, 1}};

  cudf::ast::tree tree;
  auto const snapshot = filters.snapshot();
  auto const* root    = merge_dynamic_filters_into_ast(tree, nullptr, snapshot, plan);

  REQUIRE(root != nullptr);
  REQUIRE(root == &tree.back());
  // 2 cols × (1 col_ref + 2 lit + 2 op + 1 AND) + 1 cross-col AND = 13.
  REQUIRE(tree.size() == 13);
}

TEST_CASE("merge_dynamic_filters_into_ast AND-conjoins multiple filters on the same column",
          "[dynamic_filter][scan_merge]")
{
  sirius_dynamic_filter_set filters;
  auto filters_producer = filters.register_producer({0});
  REQUIRE(filters_producer.push_filter(0, make_zone_map(100, 200)));
  REQUIRE(filters_producer.push_filter(0, make_zone_map(150, 175)));
  auto plan = make_data_only_plan(0, "id");

  cudf::ast::tree tree;
  auto const snapshot = filters.snapshot();
  auto const* root    = merge_dynamic_filters_into_ast(tree, nullptr, snapshot, plan);

  REQUIRE(root != nullptr);
  REQUIRE(root == &tree.back());
  // 2 filters × (1 col_ref + 2 lit + 2 op + 1 AND) + 1 AND-merging-the-two = 13.
  REQUIRE(tree.size() == 13);
}

TEST_CASE("merge_dynamic_filters_into_ast ignores out-of-range col_idx defensively",
          "[dynamic_filter][scan_merge]")
{
  sirius_dynamic_filter_set filters;
  auto filters_producer = filters.register_producer({99});
  REQUIRE(filters_producer.push_filter(99, make_zone_map(0, 10)));  // col 99 doesn't exist in plan
  auto plan = make_data_only_plan(0, "id");

  cudf::ast::tree tree;
  auto const snapshot = filters.snapshot();
  auto const* root    = merge_dynamic_filters_into_ast(tree, nullptr, snapshot, plan);

  REQUIRE(root == nullptr);
  REQUIRE(tree.size() == 0);
}

//===----------------------------------------------------------------------===//
// apply_dynamic_filters_to_view — runtime apply (post-decode / cached)
//===----------------------------------------------------------------------===//

namespace {
/// Column of @p size values of type @p T: [start, start + step, start + 2 * step, ...].
template <typename T>
std::unique_ptr<cudf::column> make_sequence_column(cudf::size_type size,
                                                   ::cuda::stream_ref stream,
                                                   T start = 0,
                                                   T step  = 1)
{
  return cudf::sequence(size,
                        cudf::numeric_scalar<T>(start, true, stream),
                        cudf::numeric_scalar<T>(step, true, stream),
                        stream);
}

/// First value and increment of one column built by make_sequence_table().
template <typename T>
struct sequence_column_spec {
  T start;
  T step;
};

/// Table holding one make_sequence_column() of @p size rows per entry of @p columns; by default a
/// single INT32 column [0, 1, ..., size-1].
template <typename T = int32_t>
std::unique_ptr<cudf::table> make_sequence_table(
  cudf::size_type size,
  ::cuda::stream_ref stream,
  std::initializer_list<sequence_column_spec<T>> columns = {{0, 1}})
{
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.reserve(columns.size());
  for (auto const& [start, step] : columns) {
    cols.push_back(make_sequence_column<T>(size, stream, start, step));
  }
  return std::make_unique<cudf::table>(std::move(cols));
}

/// Copy an INT32 column's values to host. Applying a retention mask gathers survivors in order with
/// no nulls, so the result is directly comparable to an expected sequence.
std::vector<int32_t> to_host_int32(cudf::column_view const& col, ::cuda::stream_ref stream)
{
  std::vector<int32_t> host(static_cast<std::size_t>(col.size()));
  cudaMemcpyAsync(host.data(),
                  col.data<int32_t>(),
                  host.size() * sizeof(int32_t),
                  cudaMemcpyDeviceToHost,
                  stream.get());
  stream.sync();
  return host;
}

/// Build a zone-map-*only* filter (no membership) the way the hash-join producer does: reduce the
/// build column's min/max into device scalars on `stream`. A large build whose membership structure
/// doesn't fit L2 emits exactly this — the path whose missing build-stream sync produced
/// cross-stream false negatives (Q8/Q9/Q17 with enable_dynamic_zone_map_filter). The producer's
/// drain-before-publish is the structural guard; this exercises the zone-map-only correctness
/// (bounds, column, superset).
std::shared_ptr<sirius_dynamic_zone_map_filter> make_zone_map_from_reduce(
  cudf::column_view const& build_col, ::cuda::stream_ref stream)
{
  auto mr    = cudf::get_current_device_resource_ref();
  auto min_s = cudf::reduce(build_col,
                            *cudf::make_min_aggregation<cudf::reduce_aggregation>(),
                            build_col.type(),
                            stream,
                            mr);
  auto max_s = cudf::reduce(build_col,
                            *cudf::make_max_aggregation<cudf::reduce_aggregation>(),
                            build_col.type(),
                            stream,
                            mr);
  std::vector<zone_map_entry> zones;
  zones.push_back({std::move(min_s), std::move(max_s)});
  return std::make_shared<sirius_dynamic_zone_map_filter>(std::move(zones));
}
}  // namespace

TEST_CASE("apply_dynamic_filters_to_view drops rows outside the zone",
          "[dynamic_filter][scan_merge]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();
  auto table                = make_sequence_table(10, stream);  // [0..9]

  sirius_dynamic_filter_set filters;
  auto filters_producer = filters.register_producer({0});
  REQUIRE(filters_producer.push_filter(0, make_zone_map(3, 6)));  // inclusive [3,6] keeps 3,4,5,6

  auto out =
    sirius::op::scan::apply_dynamic_filters_to_view(table->view(), filters.snapshot(), stream);
  stream.sync();

  REQUIRE(out != nullptr);
  REQUIRE(out->num_columns() == 1);
  REQUIRE(out->num_rows() == 4);
  REQUIRE(to_host_int32(out->view().column(0), stream) == std::vector<int32_t>{3, 4, 5, 6});
  REQUIRE(table->num_rows() == 10);  // input untouched
}

TEST_CASE("apply_dynamic_filters_to_view honors an exclusive upper bound [lo, hi)",
          "[dynamic_filter][scan_merge]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();
  auto table                = make_sequence_table(10, stream);  // [0..9]

  sirius_dynamic_filter_set filters;
  auto filters_producer = filters.register_producer({0});
  // [3,6): inclusive_min, exclusive_max -> GREATER_EQUAL(3) AND LESS(6) -> {3,4,5}
  REQUIRE(filters_producer.push_filter(
    0, make_zone_map(3, 6, /*inclusive_min=*/true, /*inclusive_max=*/false)));

  auto out =
    sirius::op::scan::apply_dynamic_filters_to_view(table->view(), filters.snapshot(), stream);
  stream.sync();

  REQUIRE(out != nullptr);
  REQUIRE(to_host_int32(out->view().column(0), stream) == std::vector<int32_t>{3, 4, 5});
}

TEST_CASE("apply_dynamic_filters_to_view honors an exclusive lower bound (lo, hi]",
          "[dynamic_filter][scan_merge]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();
  auto table                = make_sequence_table(10, stream);  // [0..9]

  sirius_dynamic_filter_set filters;
  auto filters_producer = filters.register_producer({0});
  // (3,6]: exclusive_min, inclusive_max -> GREATER(3) AND LESS_EQUAL(6) -> {4,5,6}
  REQUIRE(filters_producer.push_filter(
    0, make_zone_map(3, 6, /*inclusive_min=*/false, /*inclusive_max=*/true)));

  auto out =
    sirius::op::scan::apply_dynamic_filters_to_view(table->view(), filters.snapshot(), stream);
  stream.sync();

  REQUIRE(out != nullptr);
  REQUIRE(to_host_int32(out->view().column(0), stream) == std::vector<int32_t>{4, 5, 6});
}

TEST_CASE("zone-map-only filter from a device reduce keeps a correct superset (no false negative)",
          "[dynamic_filter][scan_merge]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();

  // Build keys spanning [100, 200] — the producer reduces these to min=100, max=200 and, with no
  // membership filter, publishes a zone-map alone. Probe [0..299]: only [100..200] can possibly
  // join, and crucially every value a build key could equal must survive (no false negative).
  auto build = make_sequence_column<int32_t>(101, stream, 100);
  sirius_dynamic_filter_set filters;
  auto filters_producer = filters.register_producer({0});
  REQUIRE(filters_producer.push_filter(0, make_zone_map_from_reduce(build->view(), stream)));

  auto probe = make_sequence_table(300, stream);  // [0..299]
  auto out =
    sirius::op::scan::apply_dynamic_filters_to_view(probe->view(), filters.snapshot(), stream);
  stream.sync();

  REQUIRE(out != nullptr);
  std::vector<int32_t> expected(101);
  std::iota(expected.begin(), expected.end(), 100);  // {100, 101, ..., 200}, inclusive bounds
  REQUIRE(to_host_int32(out->view().column(0), stream) == expected);
}

TEST_CASE("apply_dynamic_filters_to_view returns nullptr for an empty channel",
          "[dynamic_filter][scan_merge]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();
  auto table                = make_sequence_table(10, stream);

  sirius_dynamic_filter_set filters;  // empty

  auto out =
    sirius::op::scan::apply_dynamic_filters_to_view(table->view(), filters.snapshot(), stream);
  stream.sync();

  REQUIRE(out == nullptr);
  REQUIRE(table->num_rows() == 10);  // input untouched
}

TEST_CASE("sirius_dynamic_filter_set ignore_columns drops filters for ignored columns",
          "[dynamic_filter][scan_merge]")
{
  // Wiring-time partition skip: a consumer marks its hive-partition output columns so the producer
  // never publishes a filter the post-decode apply would have to skip.
  sirius_dynamic_filter_set filters;
  auto filters_producer = filters.register_producer({0, 1});
  filters.ignore_columns({0});  // output col 0 is a hive partition
  // dropped — column 0 is ignored
  REQUIRE_FALSE(filters_producer.push_filter(0, make_zone_map(3, 6)));
  // kept — column 1 is a data column
  REQUIRE(filters_producer.push_filter(1, make_zone_map(3, 6)));

  auto const snapshot         = filters.snapshot();
  std::size_t kept_on_ignored = 0;
  std::size_t kept_on_data    = 0;
  for (auto const& entry : snapshot.entries()) {
    if (entry.column_index == 0) { ++kept_on_ignored; }
    if (entry.column_index == 1) { ++kept_on_data; }
  }
  REQUIRE(kept_on_ignored == 0);
  REQUIRE(kept_on_data == 1);
  REQUIRE(filters.filter_count() == 1);
}

TEST_CASE("apply_dynamic_filters_to_view AND-conjoins multiple zone filters on a column",
          "[dynamic_filter][scan_merge]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();
  auto table                = make_sequence_table(10, stream);  // [0..9]

  sirius_dynamic_filter_set filters;
  auto filters_producer = filters.register_producer({0});
  REQUIRE(filters_producer.push_filter(0, make_zone_map(2, 8)));  // keeps 2..8
  REQUIRE(
    filters_producer.push_filter(0, make_zone_map(5, 9)));  // AND keeps 5..9 → intersection 5..8

  auto out =
    sirius::op::scan::apply_dynamic_filters_to_view(table->view(), filters.snapshot(), stream);
  stream.sync();

  REQUIRE(out != nullptr);
  REQUIRE(out->num_rows() == 4);  // 5,6,7,8
  REQUIRE(table->num_rows() == 10);
}

//===----------------------------------------------------------------------===//
// Membership filters — IN-list (exact) and Bloom (no false negatives)
//===----------------------------------------------------------------------===//

namespace {
/// One INT64 sequence column [0, 1, ..., size-1] in a single-column table.
std::unique_ptr<cudf::table> make_int64_sequence_table(cudf::size_type size,
                                                       ::cuda::stream_ref stream)
{
  return make_sequence_table<int64_t>(size, stream);
}

/// Single-column table from explicit host values — for keys/probes that aren't arithmetic
/// sequences (e.g. ones containing the type-min sentinel).
template <class T>
std::unique_ptr<cudf::table> make_values_table(std::vector<T> const& values,
                                               cudf::data_type dtype,
                                               ::cuda::stream_ref stream)
{
  auto col = cudf::make_numeric_column(
    dtype, static_cast<cudf::size_type>(values.size()), cudf::mask_state::UNALLOCATED, stream);
  cudaMemcpyAsync(col->mutable_view().data<T>(),
                  values.data(),
                  values.size() * sizeof(T),
                  cudaMemcpyHostToDevice,
                  stream.get());
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(col));
  return std::make_unique<cudf::table>(std::move(cols));
}

/// Copy an INT64 column's values to host (companion to to_host_int32).
std::vector<int64_t> to_host_int64(cudf::column_view const& col, ::cuda::stream_ref stream)
{
  std::vector<int64_t> host(static_cast<std::size_t>(col.size()));
  cudaMemcpyAsync(host.data(),
                  col.data<int64_t>(),
                  host.size() * sizeof(int64_t),
                  cudaMemcpyDeviceToHost,
                  stream.get());
  stream.sync();
  return host;
}
}  // namespace

TEST_CASE("sirius_dynamic_in_list_filter keeps exactly the rows whose key is a build key",
          "[dynamic_filter][scan_merge]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();
  // Build key set {0,1,2,3,4}; probe table [0..9]. Exact membership keeps the first five.
  auto keys = make_sequence_column<int64_t>(5, stream);
  sirius_dynamic_filter_set filters;
  auto filters_producer = filters.register_producer({0});
  REQUIRE(
    filters_producer.push_filter(0,
                                 std::make_shared<sirius::op::sirius_dynamic_in_list_filter>(
                                   keys->view(), stream, cudf::get_current_device_resource_ref())));

  auto table = make_int64_sequence_table(10, stream);
  auto out =
    sirius::op::scan::apply_dynamic_filters_to_view(table->view(), filters.snapshot(), stream);
  stream.sync();
  REQUIRE(out != nullptr);
  REQUIRE(out->num_rows() == 5);
  REQUIRE(table->num_rows() == 10);
}

TEST_CASE("sirius_dynamic_in_list_filter INT64 path uses a persistent set and probes repeatedly",
          "[dynamic_filter][scan_merge]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();
  auto keys                 = make_sequence_column<int64_t>(5, stream);
  auto filter               = std::make_shared<sirius::op::sirius_dynamic_in_list_filter>(
    keys->view(), stream, cudf::get_current_device_resource_ref());
  REQUIRE(filter->has_persistent_set());  // INT64, non-null keys → fast path

  sirius_dynamic_filter_set filters;
  auto filters_producer = filters.register_producer({0});
  REQUIRE(filters_producer.push_filter(0, filter));

  // Two applies against the same persistent structure (the per-split pattern).
  for (int i = 0; i < 2; ++i) {
    auto table = make_int64_sequence_table(10, stream);
    auto out =
      sirius::op::scan::apply_dynamic_filters_to_view(table->view(), filters.snapshot(), stream);
    stream.sync();
    REQUIRE(out != nullptr);
    REQUIRE(out->num_rows() == 5);  // exact membership: 0..4
    REQUIRE(table->num_rows() == 10);
  }
}

TEST_CASE("sirius_dynamic_bloom_filter never drops a true match (no false negatives)",
          "[dynamic_filter][scan_merge]")
{
  auto const num_keys       = GENERATE(5, 1024);
  ::cuda::stream_ref stream = cudf::get_default_stream();
  auto keys                 = make_sequence_column<int64_t>(num_keys, stream);
  sirius_dynamic_filter_set filters;
  auto filters_producer = filters.register_producer({0});
  REQUIRE(
    filters_producer.push_filter(0,
                                 std::make_shared<sirius::op::sirius_dynamic_bloom_filter>(
                                   keys->view(), stream, cudf::get_current_device_resource_ref())));

  auto table = make_int64_sequence_table(2 * num_keys, stream);
  auto out =
    sirius::op::scan::apply_dynamic_filters_to_view(table->view(), filters.snapshot(), stream);
  stream.sync();
  REQUIRE(out != nullptr);
  auto const survivors = to_host_int64(out->view().column(0), stream);
  REQUIRE(survivors.size() >= num_keys);
  REQUIRE(survivors.size() <= 2 * num_keys);
  // Build keys precede every possible false positive in the probe sequence.
  for (int64_t key = 0; key < num_keys; ++key) {
    REQUIRE(survivors[key] == key);
  }
  REQUIRE(table->num_rows() == 2 * num_keys);
}

TEST_CASE("sirius_dynamic_in_list_filter supports INT32 keys exactly",
          "[dynamic_filter][scan_merge]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();
  // INT32 build key set {0,1,2,3,4}; INT32 probe [0..9]. Exact membership keeps exactly
  // {0,1,2,3,4}.
  auto keys   = make_sequence_column<int32_t>(5, stream);
  auto filter = std::make_shared<sirius::op::sirius_dynamic_in_list_filter>(
    keys->view(), stream, cudf::get_current_device_resource_ref());
  REQUIRE(filter->has_persistent_set());  // INT32, non-null keys → persistent-set fast path

  sirius_dynamic_filter_set filters;
  auto filters_producer = filters.register_producer({0});
  REQUIRE(filters_producer.push_filter(0, filter));

  auto table = make_sequence_table(10, stream);  // INT32 [0..9]
  auto out =
    sirius::op::scan::apply_dynamic_filters_to_view(table->view(), filters.snapshot(), stream);
  stream.sync();
  REQUIRE(out != nullptr);
  REQUIRE(to_host_int32(out->view().column(0), stream) == std::vector<int32_t>{0, 1, 2, 3, 4});
}

TEST_CASE("sirius_dynamic_bloom_filter supports INT32 keys with no false negatives",
          "[dynamic_filter][scan_merge]")
{
  auto const num_keys       = GENERATE(5, 1024);
  ::cuda::stream_ref stream = cudf::get_default_stream();
  auto keys                 = make_sequence_column<int32_t>(num_keys, stream);
  sirius_dynamic_filter_set filters;
  auto filters_producer = filters.register_producer({0});
  REQUIRE(
    filters_producer.push_filter(0,
                                 std::make_shared<sirius::op::sirius_dynamic_bloom_filter>(
                                   keys->view(), stream, cudf::get_current_device_resource_ref())));

  auto table = make_sequence_table(2 * num_keys, stream);
  auto out =
    sirius::op::scan::apply_dynamic_filters_to_view(table->view(), filters.snapshot(), stream);
  stream.sync();
  REQUIRE(out != nullptr);
  auto const survivors = to_host_int32(out->view().column(0), stream);
  REQUIRE(survivors.size() >= num_keys);
  REQUIRE(survivors.size() <= 2 * num_keys);
  for (int32_t key = 0; key < num_keys; ++key) {
    REQUIRE(survivors[key] == key);
  }
}

TEST_CASE("sirius_dynamic_bloom_filter excludes null build slots from the key set",
          "[dynamic_filter][scan_merge]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();

  // Build keys [0,1,2,3,4,999] with the 999 slot nulled: only {0..4} may enter the set.
  std::vector<int64_t> const key_values{0, 1, 2, 3, 4, 999};
  auto keys = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT64}, 6, cudf::mask_state::ALL_VALID, stream);
  cudaMemcpyAsync(keys->mutable_view().data<int64_t>(),
                  key_values.data(),
                  key_values.size() * sizeof(int64_t),
                  cudaMemcpyHostToDevice,
                  stream.get());
  cudf::set_null_mask(keys->mutable_view().null_mask(), 5, 6, false, stream);
  keys->set_null_count(1);

  // Reference filter over the same valid keys, built without nulls. Compaction is exact and the
  // hash policy deterministic, so the nullable build must produce a bit-identical filter.
  auto clean_keys = make_sequence_column<int64_t>(5, stream);

  sirius_dynamic_filter_set nullable_channel;
  auto nullable_channel_producer = nullable_channel.register_producer({0});
  REQUIRE(nullable_channel_producer.push_filter(
    0,
    std::make_shared<sirius::op::sirius_dynamic_bloom_filter>(
      keys->view(), stream, cudf::get_current_device_resource_ref())));
  sirius_dynamic_filter_set reference_channel;
  auto reference_channel_producer = reference_channel.register_producer({0});
  REQUIRE(reference_channel_producer.push_filter(
    0,
    std::make_shared<sirius::op::sirius_dynamic_bloom_filter>(
      clean_keys->view(), stream, cudf::get_current_device_resource_ref())));

  // Probe [0..9] plus 999 — the value present only at the null build slot.
  auto probe = make_values_table<int64_t>(
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 999}, cudf::data_type{cudf::type_id::INT64}, stream);
  auto out_nullable = sirius::op::scan::apply_dynamic_filters_to_view(
    probe->view(), nullable_channel.snapshot(), stream);
  auto out_reference = sirius::op::scan::apply_dynamic_filters_to_view(
    probe->view(), reference_channel.snapshot(), stream);
  stream.sync();
  REQUIRE(out_nullable != nullptr);
  REQUIRE(out_reference != nullptr);

  auto const survivors = to_host_int64(out_nullable->view().column(0), stream);
  // No false negatives: the five valid keys lead the probe and must all survive, in order.
  REQUIRE(survivors.size() >= 5);
  REQUIRE(std::vector<int64_t>(survivors.begin(), survivors.begin() + 5) ==
          std::vector<int64_t>{0, 1, 2, 3, 4});
  // The nullable build must behave identically to the clean one: a null build slot
  // contributes no payload, so 999 must not survive the nullable filter either.
  REQUIRE(survivors == to_host_int64(out_reference->view().column(0), stream));
}

TEST_CASE("sirius_dynamic_in_list_filter keeps a build key equal to the INT64 sentinel",
          "[dynamic_filter][scan_merge]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();
  auto const dtype          = cudf::data_type{cudf::type_id::INT64};
  // Build keys include INT64_MIN — the cuco empty-slot sentinel that static_set never inserts.
  auto keys =
    make_values_table<int64_t>({std::numeric_limits<int64_t>::min(), 0, 1, 2}, dtype, stream);
  auto filter = std::make_shared<sirius::op::sirius_dynamic_in_list_filter>(
    keys->view().column(0), stream, cudf::get_current_device_resource_ref());
  REQUIRE(filter->has_persistent_set());  // stays on the exact IN-list path (no Bloom downgrade)

  sirius_dynamic_filter_set filters;
  auto filters_producer = filters.register_producer({0});
  REQUIRE(filters_producer.push_filter(0, filter));

  // Probe {INT64_MIN, 2, 7}: INT64_MIN and 2 are build keys and must survive; 7 must be dropped.
  auto probe =
    make_values_table<int64_t>({std::numeric_limits<int64_t>::min(), 2, 7}, dtype, stream);
  auto out =
    sirius::op::scan::apply_dynamic_filters_to_view(probe->view(), filters.snapshot(), stream);
  stream.sync();
  REQUIRE(out != nullptr);
  REQUIRE(to_host_int64(out->view().column(0), stream) ==
          std::vector<int64_t>{std::numeric_limits<int64_t>::min(), 2});
}

TEST_CASE("sirius_dynamic_in_list_filter keeps a build key equal to the INT32 sentinel",
          "[dynamic_filter][scan_merge]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();
  auto const dtype          = cudf::data_type{cudf::type_id::INT32};
  auto keys =
    make_values_table<int32_t>({std::numeric_limits<int32_t>::min(), 0, 1, 2}, dtype, stream);
  auto filter = std::make_shared<sirius::op::sirius_dynamic_in_list_filter>(
    keys->view().column(0), stream, cudf::get_current_device_resource_ref());
  REQUIRE(filter->has_persistent_set());

  sirius_dynamic_filter_set filters;
  auto filters_producer = filters.register_producer({0});
  REQUIRE(filters_producer.push_filter(0, filter));

  auto probe =
    make_values_table<int32_t>({std::numeric_limits<int32_t>::min(), 2, 7}, dtype, stream);
  auto out =
    sirius::op::scan::apply_dynamic_filters_to_view(probe->view(), filters.snapshot(), stream);
  stream.sync();
  REQUIRE(out != nullptr);
  REQUIRE(to_host_int32(out->view().column(0), stream) ==
          std::vector<int32_t>{std::numeric_limits<int32_t>::min(), 2});
}

TEST_CASE("sirius_dynamic_small_in_list_filter keeps exactly the rows whose key is a build key",
          "[dynamic_filter][scan_merge]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();
  // Small build key set {0,1,2,3,4}; probe [0..9]. The brute-force scan keeps the first five.
  auto keys = make_sequence_column<int64_t>(5, stream);
  sirius_dynamic_filter_set filters;
  auto filters_producer = filters.register_producer({0});
  REQUIRE(
    filters_producer.push_filter(0,
                                 std::make_shared<sirius::op::sirius_dynamic_small_in_list_filter>(
                                   keys->view(), stream, cudf::get_current_device_resource_ref())));

  auto table = make_int64_sequence_table(10, stream);
  auto out =
    sirius::op::scan::apply_dynamic_filters_to_view(table->view(), filters.snapshot(), stream);
  stream.sync();
  REQUIRE(out != nullptr);
  REQUIRE(to_host_int64(out->view().column(0), stream) == std::vector<int64_t>{0, 1, 2, 3, 4});
  REQUIRE(table->num_rows() == 10);
}

TEST_CASE("sirius_dynamic_small_in_list_filter supports INT32 keys exactly",
          "[dynamic_filter][scan_merge]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();
  auto keys                 = make_sequence_column<int32_t>(5, stream);
  sirius_dynamic_filter_set filters;
  auto filters_producer = filters.register_producer({0});
  REQUIRE(
    filters_producer.push_filter(0,
                                 std::make_shared<sirius::op::sirius_dynamic_small_in_list_filter>(
                                   keys->view(), stream, cudf::get_current_device_resource_ref())));

  auto table = make_sequence_table(10, stream);  // INT32 [0..9]
  auto out =
    sirius::op::scan::apply_dynamic_filters_to_view(table->view(), filters.snapshot(), stream);
  stream.sync();
  REQUIRE(out != nullptr);
  REQUIRE(to_host_int32(out->view().column(0), stream) == std::vector<int32_t>{0, 1, 2, 3, 4});
}

TEST_CASE("sirius_dynamic_small_in_list_filter matches a key equal to INT32_MIN (cuco's sentinel)",
          "[dynamic_filter][scan_merge]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();
  auto const dtype          = cudf::data_type{cudf::type_id::INT32};
  // Single build key {INT32_MIN} — the value cuco::static_set reserves as its empty slot and never
  // stores. The brute-force scan has no reserved value, so INT32_MIN is a valid needle: this filter
  // prunes non-matches exactly, where sirius_dynamic_in_list_filter would (harmlessly) keep them.
  auto keys = make_values_table<int32_t>({std::numeric_limits<int32_t>::min()}, dtype, stream);
  sirius_dynamic_filter_set filters;
  auto filters_producer = filters.register_producer({0});
  REQUIRE(filters_producer.push_filter(
    0,
    std::make_shared<sirius::op::sirius_dynamic_small_in_list_filter>(
      keys->view().column(0), stream, cudf::get_current_device_resource_ref())));

  // Probe {INT32_MIN, INT32_MIN+1, INT32_MIN+2}; only the first is a build key.
  auto probe = make_values_table<int32_t>({std::numeric_limits<int32_t>::min(),
                                           std::numeric_limits<int32_t>::min() + 1,
                                           std::numeric_limits<int32_t>::min() + 2},
                                          dtype,
                                          stream);
  auto out =
    sirius::op::scan::apply_dynamic_filters_to_view(probe->view(), filters.snapshot(), stream);
  stream.sync();
  REQUIRE(out != nullptr);
  REQUIRE(to_host_int32(out->view().column(0), stream) ==
          std::vector<int32_t>{std::numeric_limits<int32_t>::min()});
}

TEST_CASE("sirius_dynamic_small_in_list_filter: kind, size, capabilities, and supports gate",
          "[dynamic_filter][scan_merge]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();
  auto const mr             = cudf::get_current_device_resource_ref();
  using F                   = sirius::op::sirius_dynamic_small_in_list_filter;

  auto one_i32 = make_sequence_column<int32_t>(1, stream);
  auto max_i32 = make_sequence_column<int32_t>(static_cast<cudf::size_type>(F::k_max_keys), stream);
  auto oversized_i32 =
    make_sequence_column<int32_t>(static_cast<cudf::size_type>(F::k_max_keys + 1), stream);
  auto empty_i32 = cudf::make_empty_column(cudf::data_type{cudf::type_id::INT32});
  auto null_i32  = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, 1, cudf::mask_state::ALL_NULL, stream);
  auto f64 =
    make_values_table<double>({0.0, 1.0, 2.0}, cudf::data_type{cudf::type_id::FLOAT64}, stream);

  // supports() gate: 1..k_max_keys *valid* integer keys; the all-null column has none.
  REQUIRE(F::supports(one_i32->view()));
  REQUIRE(F::supports(max_i32->view()));
  REQUIRE_FALSE(F::supports(empty_i32->view()));
  REQUIRE_FALSE(F::supports(oversized_i32->view()));
  REQUIRE_FALSE(F::supports(f64->view().column(0)));
  REQUIRE_FALSE(F::supports(null_i32->view()));

  F f(max_i32->view(), stream, mr);
  stream.sync();
  REQUIRE(f.kind() == sirius_dynamic_filter_kind::IN_LIST);
  REQUIRE(f.size() == F::k_max_keys);
  REQUIRE(f.replica_count() == 1);  // source-device snapshot built in the constructor
  // Cast through the base pointer, exactly as the consumer-side merge does: the filter advertises
  // the runtime-mask capability but not AST lowering, keeping it out of the parquet row-group path.
  sirius_dynamic_filter const* base = &f;
  REQUIRE(dynamic_cast<sirius::op::sirius_mask_applicable const*>(base) != nullptr);
  REQUIRE(dynamic_cast<sirius::op::sirius_ast_lowerable const*>(base) == nullptr);
}

//===----------------------------------------------------------------------===//
// apply_dynamic_filters_to_view — view-based core (pinned cached path)
//===----------------------------------------------------------------------===//

namespace {
/// IN-list filter keeping keys [0, count) of type @p T.
template <typename T = int64_t>
std::shared_ptr<sirius::op::sirius_dynamic_in_list_filter> make_in_list_prefix(
  int64_t count, ::cuda::stream_ref stream)
{
  auto keys = make_sequence_column<T>(static_cast<cudf::size_type>(count), stream);
  return std::make_shared<sirius::op::sirius_dynamic_in_list_filter>(
    keys->view(), stream, cudf::get_current_device_resource_ref());
}

bool forced_strategies_equal(cudf::table_view const& input,
                             sirius::op::dynamic_filter_snapshot const& snapshot,
                             ::cuda::stream_ref stream,
                             dynamic_filter_apply_mode mode)
{
  using sirius::op::scan::detail::apply_dynamic_filters_to_view_for_testing;
  using sirius::op::scan::detail::compaction_strategy;
  auto const apply = [&](compaction_strategy strategy) {
    auto out = apply_dynamic_filters_to_view_for_testing(input, snapshot, stream, strategy, mode);
    stream.sync();
    return out;
  };
  auto const cascade = apply(compaction_strategy::CASCADE);
  for (auto const strategy :
       {compaction_strategy::DEFERRED_KEYS, compaction_strategy::GATHER_ONCE}) {
    auto const other = apply(strategy);
    if (static_cast<bool>(cascade) != static_cast<bool>(other)) { return false; }
    if (cascade &&
        !sirius::test::expect_tables_equivalent_impl(cascade->view(), other->view(), stream)) {
      return false;
    }
  }
  return true;
}

std::unique_ptr<cudf::column> make_nullable_list_payload(cudf::size_type rows,
                                                         ::cuda::stream_ref stream)
{
  std::vector<int32_t> offsets(static_cast<std::size_t>(rows) + 1);
  std::vector<int32_t> values(static_cast<std::size_t>(rows) * 2);
  for (cudf::size_type row = 0; row <= rows; ++row) {
    offsets[static_cast<std::size_t>(row)] = row * 2;
  }
  std::iota(values.begin(), values.end(), 100);
  auto offsets_column = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, rows + 1, cudf::mask_state::UNALLOCATED, stream);
  auto values_column = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, rows * 2, cudf::mask_state::ALL_VALID, stream);
  cudaMemcpyAsync(offsets_column->mutable_view().data<int32_t>(),
                  offsets.data(),
                  offsets.size() * sizeof(int32_t),
                  cudaMemcpyHostToDevice,
                  stream.get());
  cudaMemcpyAsync(values_column->mutable_view().data<int32_t>(),
                  values.data(),
                  values.size() * sizeof(int32_t),
                  cudaMemcpyHostToDevice,
                  stream.get());
  cudf::set_null_mask(values_column->mutable_view().null_mask(), 4, 5, false, stream);
  values_column->set_null_count(1);
  auto parent_mask = cudf::create_null_mask(rows, cudf::mask_state::ALL_VALID, stream);
  cudf::set_null_mask(
    reinterpret_cast<cudf::bitmask_type*>(parent_mask.data()), 6, 7, false, stream);
  return cudf::make_lists_column(
    rows, std::move(offsets_column), std::move(values_column), 1, std::move(parent_mask));
}

/// Membership filter that counts compute_mask calls and delegates to a wrapped IN-list filter.
/// Makes "the gate did not re-run this filter" directly observable.
class counting_in_list_filter final : public sirius_dynamic_filter,
                                      public sirius::op::sirius_mask_applicable {
 public:
  explicit counting_in_list_filter(std::shared_ptr<sirius::op::sirius_dynamic_in_list_filter> inner)
    : _inner(std::move(inner))
  {
  }

  [[nodiscard]] sirius_dynamic_filter_kind kind() const override { return _inner->kind(); }

  [[nodiscard]] bool is_available_on_device(int device_id) const noexcept override
  {
    return _inner->is_available_on_device(device_id);
  }

  using sirius::op::sirius_mask_applicable::compute_mask;
  [[nodiscard]] std::unique_ptr<cudf::column> compute_mask(
    cudf::column_view const& probe,
    std::uint32_t const* prior_mask_words,
    int device_id,
    ::cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) const override
  {
    ++_mask_calls;
    _probe_rows.push_back(probe.size());
    return _inner->compute_mask(probe, prior_mask_words, device_id, stream, mr);
  }

  [[nodiscard]] int mask_calls() const noexcept { return _mask_calls; }
  [[nodiscard]] std::vector<cudf::size_type> const& probe_rows() const noexcept
  {
    return _probe_rows;
  }

 private:
  std::shared_ptr<sirius::op::sirius_dynamic_in_list_filter> _inner;
  mutable int _mask_calls = 0;
  mutable std::vector<cudf::size_type> _probe_rows;
};

class nullable_declining_filter final : public sirius_dynamic_filter,
                                        public sirius::op::sirius_mask_applicable {
 public:
  [[nodiscard]] sirius_dynamic_filter_kind kind() const override
  {
    return sirius_dynamic_filter_kind::IN_LIST;
  }

  [[nodiscard]] bool is_available_on_device(int) const noexcept override { return true; }

  [[nodiscard]] std::unique_ptr<cudf::column> compute_mask(
    cudf::column_view const& probe,
    std::uint32_t const*,
    int,
    ::cuda::stream_ref,
    rmm::device_async_resource_ref) const override
  {
    ++_mask_calls;
    _saw_nulls = probe.has_nulls();
    return nullptr;
  }

  [[nodiscard]] int mask_calls() const noexcept { return _mask_calls; }
  [[nodiscard]] bool saw_nulls() const noexcept { return _saw_nulls; }

 private:
  mutable int _mask_calls = 0;
  mutable bool _saw_nulls = false;
};

struct blocked_filter_work {
  std::promise<void> started;
  std::latch release{1};
  std::atomic<bool> released{false};
  std::atomic<bool> completed{false};

  void unblock() noexcept
  {
    if (!released.exchange(true)) { release.count_down(); }
  }
};

class throwing_mask_filter final : public sirius_dynamic_filter,
                                   public sirius::op::sirius_mask_applicable {
 public:
  explicit throwing_mask_filter(blocked_filter_work& work) : _work(work) {}

  [[nodiscard]] sirius_dynamic_filter_kind kind() const override
  {
    return sirius_dynamic_filter_kind::IN_LIST;
  }

  [[nodiscard]] bool is_available_on_device(int) const noexcept override { return true; }

  using sirius::op::sirius_mask_applicable::compute_mask;
  [[nodiscard]] std::unique_ptr<cudf::column> compute_mask(
    cudf::column_view const&,
    std::uint32_t const*,
    int,
    ::cuda::stream_ref stream,
    rmm::device_async_resource_ref) const override
  {
    auto const status = cudaLaunchHostFunc(
      stream.get(),
      [](void* state) {
        auto& work = *static_cast<blocked_filter_work*>(state);
        work.started.set_value();
        work.release.wait();
        work.completed.store(true);
      },
      &_work);
    if (status != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(status)); }
    throw std::runtime_error("injected dynamic filter failure after submission");
  }

 private:
  blocked_filter_work& _work;
};
}  // namespace

TEST_CASE("apply_dynamic_filters_to_view returns nullptr when no filter contributes",
          "[dynamic_filter][scan_merge]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();
  auto table                = make_int64_sequence_table(10, stream);

  sirius_dynamic_filter_set filters;  // empty

  auto out =
    sirius::op::scan::apply_dynamic_filters_to_view(table->view(), filters.snapshot(), stream);
  REQUIRE(out == nullptr);
  REQUIRE(table->num_rows() == 10);  // input untouched, still usable
}

TEST_CASE("apply_dynamic_filters_to_view gathers survivors without consuming the input",
          "[dynamic_filter][scan_merge]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();
  auto table                = make_int64_sequence_table(10, stream);  // [0..9]

  sirius_dynamic_filter_set filters;
  auto filters_producer = filters.register_producer({0});
  REQUIRE(filters_producer.push_filter(0, make_in_list_prefix(2, stream)));  // keeps 0,1

  auto out =
    sirius::op::scan::apply_dynamic_filters_to_view(table->view(), filters.snapshot(), stream);
  stream.sync();

  REQUIRE(out != nullptr);
  REQUIRE(out->num_rows() == 2);
  REQUIRE(table->num_rows() == 10);  // the view's backing table is intact
}

//===----------------------------------------------------------------------===//
// dynamic_filter_gate — selectivity gating and re-arm on new publishes
//===----------------------------------------------------------------------===//

TEST_CASE("dynamic_filter_gate is not applicable before any filter publishes",
          "[dynamic_filter][scan_merge]")
{
  sirius::op::scan::dynamic_filter_gate gate;
  sirius_dynamic_filter_set filters;  // empty
  REQUIRE_FALSE(gate.applicable(filters.snapshot()));
}

TEST_CASE("dynamic filter application keeps one captured generation across later publications",
          "[dynamic_filter][scan_merge][snapshot]")
{
  auto const stream = cudf::get_default_stream();
  auto input        = make_int64_sequence_table(10, stream);
  sirius_dynamic_filter_set filters;
  auto producer = filters.register_producer({0});
  REQUIRE(producer.push_filter(0, make_in_list_prefix(7, stream)));
  auto const first = filters.snapshot();
  REQUIRE(producer.push_filter(0, make_in_list_prefix(2, stream)));
  auto const second = filters.snapshot();

  auto first_output = sirius::op::scan::apply_dynamic_filters_to_view(input->view(), first, stream);
  auto second_output =
    sirius::op::scan::apply_dynamic_filters_to_view(input->view(), second, stream);
  REQUIRE(first.generation() == 1);
  REQUIRE(second.generation() == 2);
  REQUIRE(first_output->num_rows() == 7);
  REQUIRE(second_output->num_rows() == 2);

  sirius::op::scan::dynamic_filter_gate gate;
  gate.record_keep_ratio(10, 10, first.generation());
  REQUIRE_FALSE(gate.applicable(first));
  REQUIRE(gate.applicable(second));
}

TEST_CASE("captured dynamic filter owners survive channel destruction through GPU application",
          "[dynamic_filter][scan_merge][snapshot]")
{
  auto const stream = cudf::get_default_stream();
  auto input        = make_int64_sequence_table(10, stream);
  sirius::op::dynamic_filter_snapshot snapshot;
  std::weak_ptr<sirius_dynamic_filter const> filter_lifetime;
  {
    sirius_dynamic_filter_set filters;
    auto producer   = filters.register_producer({0});
    auto filter     = make_in_list_prefix(2, stream);
    filter_lifetime = filter;
    REQUIRE(producer.push_filter(0, std::move(filter)));
    snapshot = filters.snapshot();
  }
  REQUIRE_FALSE(filter_lifetime.expired());
  auto output = sirius::op::scan::apply_dynamic_filters_to_view(input->view(), snapshot, stream);
  REQUIRE(output->num_rows() == 2);
  snapshot = {};
  REQUIRE(filter_lifetime.expired());
}

TEST_CASE("decode probes retain exactly the captured snapshot after channel growth and destruction",
          "[dynamic_filter][scan_merge][snapshot]")
{
  auto const stream = cudf::get_default_stream();
  auto input        = make_int64_sequence_table(10, stream);
  sirius::op::scan::membership_snapshot probes;
  std::weak_ptr<sirius_dynamic_filter const> filter_lifetime;
  {
    sirius_dynamic_filter_set filters;
    auto producer   = filters.register_producer({0});
    auto filter     = make_in_list_prefix(7, stream);
    filter_lifetime = filter;
    REQUIRE(producer.push_filter(0, std::move(filter)));
    probes = sirius::op::scan::snapshot_membership_probes(filters.snapshot(), 2);
    REQUIRE(producer.push_filter(0, make_in_list_prefix(2, stream)));
    REQUIRE(probes.generation == 1);
    REQUIRE(probes.attached_probes == 1);
    REQUIRE(probes.probes[0].size() == 1);
    REQUIRE(probes.probes[1].empty());
  }
  REQUIRE_FALSE(filter_lifetime.expired());
  auto mask = probes.probes[0][0].probe(input->view().column(0),
                                        /*prior_mask_words=*/nullptr,
                                        stream,
                                        cudf::get_current_device_resource_ref());
  REQUIRE(cudaStreamSynchronize(cuda::stream_ref{stream}.get()) == cudaSuccess);
  REQUIRE(mask != nullptr);
  REQUIRE(mask->size() == input->num_rows());
  probes = {};
  REQUIRE(filter_lifetime.expired());
}

TEST_CASE("dynamic filter application retires submitted work before propagating an exception",
          "[dynamic_filter][scan_merge][snapshot][compaction_lifetime]")
{
  auto const strategy = GENERATE(sirius::op::scan::detail::compaction_strategy::CASCADE,
                                 sirius::op::scan::detail::compaction_strategy::DEFERRED_KEYS,
                                 sirius::op::scan::detail::compaction_strategy::GATHER_ONCE);
  rmm::cuda_stream stream;
  auto input    = make_sequence_table<int64_t>(10, stream, {{0, 1}, {100, 1}});
  int device_id = -1;
  REQUIRE(cudaGetDevice(&device_id) == cudaSuccess);
  blocked_filter_work work;
  sirius::op::dynamic_filter_snapshot snapshot;
  std::weak_ptr<sirius_dynamic_filter const> filter_lifetime;
  {
    sirius_dynamic_filter_set filters;
    auto producer = filters.register_producer({0, 1});
    REQUIRE(producer.push_filter(0, make_in_list_prefix(8, stream)));
    auto filter     = std::make_shared<throwing_mask_filter>(work);
    filter_lifetime = filter;
    REQUIRE(producer.push_filter(1, std::move(filter)));
    snapshot = filters.snapshot();
  }
  auto started = work.started.get_future();
  std::promise<void> returned;
  auto result = returned.get_future();
  std::jthread worker([&] {
    try {
      rmm::cuda_set_device_raii device{rmm::cuda_device_id{device_id}};
      (void)sirius::op::scan::detail::apply_dynamic_filters_to_view_for_testing(
        input->view(),
        snapshot,
        stream,
        strategy,
        dynamic_filter_apply_mode::MEMBERSHIP_MASKS_ONLY);
      returned.set_value();
    } catch (...) {
      returned.set_exception(std::current_exception());
    }
  });
  struct release_join_guard {
    blocked_filter_work& work;
    std::jthread& worker;
    cuda::stream_ref stream;

    ~release_join_guard()
    {
      work.unblock();
      if (worker.joinable()) { worker.join(); }
      (void)cudaStreamSynchronize(stream.get());
    }
  } cleanup{work, worker, stream};

  auto const callback_started =
    started.wait_for(std::chrono::seconds{5}) == std::future_status::ready;
  auto const returned_while_blocked =
    result.wait_for(std::chrono::milliseconds{100}) == std::future_status::ready;
  work.unblock();
  worker.join();
  REQUIRE(cudaStreamSynchronize(stream.value()) == cudaSuccess);

  REQUIRE(callback_started);
  REQUIRE_FALSE(returned_while_blocked);
  REQUIRE(work.completed.load());
  REQUIRE_THROWS_WITH(result.get(), "injected dynamic filter failure after submission");
  snapshot = {};
  REQUIRE(filter_lifetime.expired());
}

TEST_CASE("dynamic_filter_gate disables after an unselective first split",
          "[dynamic_filter][scan_merge]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();
  sirius::op::scan::dynamic_filter_gate gate;

  sirius_dynamic_filter_set filters;
  auto filters_producer = filters.register_producer({0});
  REQUIRE(filters_producer.push_filter(
    0, make_in_list_prefix(10, stream)));  // covers [0..9] — keeps 100%
  REQUIRE(gate.applicable(filters.snapshot()));

  auto table = make_int64_sequence_table(10, stream);
  auto out   = sirius::op::scan::apply_dynamic_filters_gated_view(
    table->view(),
    filters.snapshot(),
    gate,
    stream,
    dynamic_filter_apply_mode::INCLUDE_AST_ROW_MASKS);
  stream.sync();

  REQUIRE(out != nullptr);                             // a mask was computed (keeps everything)
  REQUIRE(out->num_rows() == 10);                      // ... so kept ratio is 1.0
  REQUIRE_FALSE(gate.applicable(filters.snapshot()));  // gate disabled for subsequent splits

  auto second = sirius::op::scan::apply_dynamic_filters_gated_view(
    table->view(),
    filters.snapshot(),
    gate,
    stream,
    dynamic_filter_apply_mode::INCLUDE_AST_ROW_MASKS);
  REQUIRE(second == nullptr);  // gated out — no work
}

TEST_CASE("dynamic_filter_gate ignores a device with no local replica",
          "[dynamic_filter][scan_merge]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();
  sirius::op::scan::dynamic_filter_gate gate;

  sirius_dynamic_filter_set filters;
  auto filters_producer = filters.register_producer({0});
  REQUIRE(filters_producer.push_filter(0, make_in_list_prefix(2, stream)));
  auto table = make_int64_sequence_table(10, stream);

  // The filter has only its current-device source replica. A consumer device with no local copy
  // must skip without recording a synthetic 100% keep ratio in the scan-global gate.
  auto unavailable = sirius::op::scan::apply_dynamic_filters_gated_view(
    table->view(),
    filters.snapshot(),
    gate,
    stream,
    dynamic_filter_apply_mode::INCLUDE_AST_ROW_MASKS,
    /*device_id=*/12345);
  REQUIRE(unavailable == nullptr);
  REQUIRE(gate.applicable(filters.snapshot()));

  auto local = sirius::op::scan::apply_dynamic_filters_gated_view(
    table->view(),
    filters.snapshot(),
    gate,
    stream,
    dynamic_filter_apply_mode::INCLUDE_AST_ROW_MASKS);
  stream.sync();
  REQUIRE(local != nullptr);
  REQUIRE(local->num_rows() == 2);
  REQUIRE(gate.applicable(filters.snapshot()));
}

TEST_CASE("dynamic_filter_gate re-arms when a filter publishes after the disable decision",
          "[dynamic_filter][scan_merge]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();
  sirius::op::scan::dynamic_filter_gate gate;

  // Unselective filter publishes first and disables the gate (the Q8 supplier hazard).
  sirius_dynamic_filter_set filters;
  auto filters_producer = filters.register_producer({0});
  REQUIRE(filters_producer.push_filter(0, make_in_list_prefix(10, stream)));
  auto table = make_int64_sequence_table(10, stream);
  (void)sirius::op::scan::apply_dynamic_filters_gated_view(
    table->view(),
    filters.snapshot(),
    gate,
    stream,
    dynamic_filter_apply_mode::INCLUDE_AST_ROW_MASKS);
  stream.sync();
  REQUIRE_FALSE(gate.applicable(filters.snapshot()));

  // A selective filter lands later: the channel grew, so the gate must re-arm...
  REQUIRE(filters_producer.push_filter(0, make_in_list_prefix(2, stream)));
  REQUIRE(gate.applicable(filters.snapshot()));

  // ...and the re-measurement sees the combined mask (AND → keeps 0,1), going ACTIVE.
  auto out = sirius::op::scan::apply_dynamic_filters_gated_view(
    table->view(),
    filters.snapshot(),
    gate,
    stream,
    dynamic_filter_apply_mode::INCLUDE_AST_ROW_MASKS);
  stream.sync();
  REQUIRE(out != nullptr);
  REQUIRE(out->num_rows() == 2);
  REQUIRE(gate.applicable(filters.snapshot()));  // active — stays applicable
}

TEST_CASE("dynamic_filter_gate stays active once a selective split proves the filter useful",
          "[dynamic_filter][scan_merge]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();
  sirius::op::scan::dynamic_filter_gate gate;

  sirius_dynamic_filter_set filters;
  auto filters_producer = filters.register_producer({0});
  REQUIRE(filters_producer.push_filter(0, make_in_list_prefix(2, stream)));  // selective: keeps 20%
  auto table = make_int64_sequence_table(10, stream);
  (void)sirius::op::scan::apply_dynamic_filters_gated_view(
    table->view(),
    filters.snapshot(),
    gate,
    stream,
    dynamic_filter_apply_mode::INCLUDE_AST_ROW_MASKS);
  stream.sync();
  REQUIRE(gate.applicable(filters.snapshot()));

  // A later unselective publish must not demote an active gate.
  REQUIRE(filters_producer.push_filter(0, make_in_list_prefix(10, stream)));
  auto out = sirius::op::scan::apply_dynamic_filters_gated_view(
    table->view(),
    filters.snapshot(),
    gate,
    stream,
    dynamic_filter_apply_mode::INCLUDE_AST_ROW_MASKS);
  stream.sync();
  REQUIRE(out != nullptr);
  REQUIRE(out->num_rows() == 2);                 // cascade of both filters == their conjunction
  REQUIRE(gate.applicable(filters.snapshot()));  // still active
}

TEST_CASE("dynamic_filter_gate serializes concurrent stale and re-armed decisions",
          "[dynamic_filter][scan_merge][concurrent]")
{
  // Model tasks that started on the old one-filter generation and finish alongside selective tasks
  // that observed the newly-published second filter. Once any current-generation task commits
  // ACTIVE, stale completions must not overwrite it with DISABLED.
  sirius_dynamic_filter_set filters;
  auto filters_producer = filters.register_producer({0});
  REQUIRE(filters_producer.push_filter(0, std::make_shared<stub_runtime_only_filter>()));
  REQUIRE(filters_producer.push_filter(0, std::make_shared<stub_runtime_only_filter>()));

  constexpr std::size_t rounds          = 256;
  constexpr std::size_t stale_workers   = 6;
  constexpr std::size_t current_workers = 2;
  constexpr std::size_t total_workers   = stale_workers + current_workers;
  std::vector<std::unique_ptr<sirius::op::scan::dynamic_filter_gate>> gates;
  gates.reserve(rounds);
  for (std::size_t i = 0; i < rounds; ++i) {
    gates.push_back(std::make_unique<sirius::op::scan::dynamic_filter_gate>());
  }

  std::barrier phase{static_cast<std::ptrdiff_t>(total_workers + 1)};
  std::vector<std::thread> workers;
  workers.reserve(total_workers);
  auto record_all = [&](bool current_generation) {
    for (auto const& gate : gates) {
      phase.arrive_and_wait();
      gate->record_keep_ratio(/*rows_before=*/100,
                              /*rows_after=*/current_generation ? 10 : 100,
                              /*observed_filter_count=*/current_generation ? 2 : 1);
      phase.arrive_and_wait();
    }
  };
  for (std::size_t i = 0; i < stale_workers; ++i) {
    workers.emplace_back(record_all, false);
  }
  for (std::size_t i = 0; i < current_workers; ++i) {
    workers.emplace_back(record_all, true);
  }

  bool all_stayed_active = true;
  for (auto const& gate : gates) {
    phase.arrive_and_wait();
    phase.arrive_and_wait();

    // If a stale completion overwrote ACTIVE with the old generation's DISABLED decision, this
    // current-generation unselective result seals DISABLED and makes applicable() false. With the
    // serialized transition, ACTIVE is terminal and this call is a no-op.
    gate->record_keep_ratio(/*rows_before=*/100,
                            /*rows_after=*/100,
                            /*observed_filter_count=*/2);
    all_stayed_active = all_stayed_active && gate->applicable(filters.snapshot());
  }
  for (auto& worker : workers) {
    worker.join();
  }

  REQUIRE(all_stayed_active);
}

TEST_CASE("cascaded membership filters produce the conjunction of all filters",
          "[dynamic_filter][scan_merge]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();

  sirius_dynamic_filter_set filters;
  auto filters_producer = filters.register_producer({0});
  REQUIRE(filters_producer.push_filter(0, make_in_list_prefix(7, stream)));  // keeps 0..6
  REQUIRE(filters_producer.push_filter(0, make_in_list_prefix(4, stream)));  // keeps 0..3

  auto table = make_int64_sequence_table(10, stream);
  auto out =
    sirius::op::scan::apply_dynamic_filters_to_view(table->view(), filters.snapshot(), stream);
  stream.sync();
  REQUIRE(out != nullptr);
  REQUIRE(out->num_rows() == 4);  // 0..3 — intersection regardless of cascade order
}

TEST_CASE("per-filter gate measures marginal keep and skips a useless filter on later splits",
          "[dynamic_filter][scan_merge]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();
  sirius::op::scan::dynamic_filter_gate gate;

  auto useless   = make_in_list_prefix(10, stream);  // covers the whole domain — keep 1.0
  auto selective = make_in_list_prefix(2, stream);   // keeps 20%
  sirius_dynamic_filter_set filters;
  auto filters_producer = filters.register_producer({0});
  REQUIRE(filters_producer.push_filter(0, useless));
  REQUIRE(filters_producer.push_filter(0, selective));

  auto table = make_int64_sequence_table(10, stream);
  auto out   = sirius::op::scan::apply_dynamic_filters_gated_view(
    table->view(),
    filters.snapshot(),
    gate,
    stream,
    dynamic_filter_apply_mode::INCLUDE_AST_ROW_MASKS);
  stream.sync();
  REQUIRE(out != nullptr);
  REQUIRE(out->num_rows() == 2);

  // First split measured both marginals: the domain-covering filter is now skippable, the
  // selective one is not.
  auto useless_kept = gate.filter_keep_ratio(useless.get(), filters.filter_count());
  REQUIRE(useless_kept.has_value());
  REQUIRE(sirius::op::scan::dynamic_filter_gate::filter_skippable(*useless_kept));
  auto selective_kept = gate.filter_keep_ratio(selective.get(), filters.filter_count());
  REQUIRE(selective_kept.has_value());
  REQUIRE_FALSE(sirius::op::scan::dynamic_filter_gate::filter_skippable(*selective_kept));

  // Later splits still produce the right rows with the useless filter dropped from the cascade.
  auto out2 = sirius::op::scan::apply_dynamic_filters_gated_view(
    table->view(),
    filters.snapshot(),
    gate,
    stream,
    dynamic_filter_apply_mode::INCLUDE_AST_ROW_MASKS);
  stream.sync();
  REQUIRE(out2 != nullptr);
  REQUIRE(out2->num_rows() == 2);
}

TEST_CASE("per-filter gate keeps a dead verdict when the channel grows",
          "[dynamic_filter][scan_merge]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();
  sirius::op::scan::dynamic_filter_gate gate;

  auto useless = make_in_list_prefix(10, stream);  // covers the whole domain -- keep 1.0
  sirius_dynamic_filter_set filters;
  auto filters_producer = filters.register_producer({0});
  REQUIRE(filters_producer.push_filter(0, useless));

  auto table = make_int64_sequence_table(10, stream);
  auto out   = sirius::op::scan::apply_dynamic_filters_gated_view(
    table->view(),
    filters.snapshot(),
    gate,
    stream,
    dynamic_filter_apply_mode::INCLUDE_AST_ROW_MASKS);
  stream.sync();
  REQUIRE(out != nullptr);

  auto const measured = gate.filter_keep_ratio(useless.get(), filters.filter_count());
  REQUIRE(measured.has_value());
  REQUIRE(sirius::op::scan::dynamic_filter_gate::filter_skippable(*measured));

  // Growth re-opens the scan-level gate, not this verdict: the dead filter stays dead.
  REQUIRE(filters_producer.push_filter(0, make_in_list_prefix(2, stream)));
  auto const after_growth = gate.filter_keep_ratio(useless.get(), filters.filter_count());
  REQUIRE(after_growth.has_value());
  REQUIRE(sirius::op::scan::dynamic_filter_gate::filter_skippable(*after_growth));
  REQUIRE(after_growth == measured);  // the stored verdict, not a remeasure trigger
}

TEST_CASE("per-filter gate excludes a dead filter from the re-armed apply without re-running it",
          "[dynamic_filter][scan_merge]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();
  sirius::op::scan::dynamic_filter_gate gate;

  // The only filter covers the whole domain: the first split disables the scan-level gate and
  // records the filter's dead marginal verdict.
  auto useless = std::make_shared<counting_in_list_filter>(make_in_list_prefix(10, stream));
  sirius_dynamic_filter_set filters;
  auto filters_producer = filters.register_producer({0});
  REQUIRE(filters_producer.push_filter(0, useless));

  auto table = make_int64_sequence_table(10, stream);
  auto out   = sirius::op::scan::apply_dynamic_filters_gated_view(
    table->view(),
    filters.snapshot(),
    gate,
    stream,
    dynamic_filter_apply_mode::INCLUDE_AST_ROW_MASKS);
  stream.sync();
  REQUIRE(out != nullptr);
  REQUIRE(out->num_rows() == 10);
  REQUIRE(useless->mask_calls() == 1);
  REQUIRE_FALSE(gate.applicable(filters.snapshot()));

  // Growth re-arms the scan-level gate; the dead verdict is permanent, so the re-armed apply
  // runs only the newcomer.
  auto selective = make_in_list_prefix(2, stream);
  REQUIRE(filters_producer.push_filter(0, selective));
  REQUIRE(gate.applicable(filters.snapshot()));

  auto out2 = sirius::op::scan::apply_dynamic_filters_gated_view(
    table->view(),
    filters.snapshot(),
    gate,
    stream,
    dynamic_filter_apply_mode::INCLUDE_AST_ROW_MASKS);
  stream.sync();
  REQUIRE(out2 != nullptr);
  REQUIRE(out2->num_rows() == 2);
  REQUIRE(useless->mask_calls() == 1);

  auto const useless_kept = gate.filter_keep_ratio(useless.get(), filters.filter_count());
  REQUIRE(useless_kept.has_value());
  REQUIRE(sirius::op::scan::dynamic_filter_gate::filter_skippable(*useless_kept));
  auto const selective_kept = gate.filter_keep_ratio(selective.get(), filters.filter_count());
  REQUIRE(selective_kept.has_value());
  REQUIRE_FALSE(sirius::op::scan::dynamic_filter_gate::filter_skippable(*selective_kept));
  REQUIRE(gate.applicable(filters.snapshot()));  // the re-arm measured 0.2 -> ACTIVE
}

TEST_CASE("per-filter gate stales a selective verdict when the channel grows",
          "[dynamic_filter][scan_merge]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();
  sirius::op::scan::dynamic_filter_gate gate;

  auto selective = make_in_list_prefix(2, stream);  // keeps 20%
  sirius_dynamic_filter_set filters;
  auto filters_producer = filters.register_producer({0});
  REQUIRE(filters_producer.push_filter(0, selective));

  auto table = make_int64_sequence_table(10, stream);
  auto out   = sirius::op::scan::apply_dynamic_filters_gated_view(
    table->view(),
    filters.snapshot(),
    gate,
    stream,
    dynamic_filter_apply_mode::INCLUDE_AST_ROW_MASKS);
  stream.sync();
  REQUIRE(out != nullptr);
  REQUIRE(out->num_rows() == 2);

  auto const measured = gate.filter_keep_ratio(selective.get(), filters.filter_count());
  REQUIRE(measured.has_value());
  REQUIRE_FALSE(sirius::op::scan::dynamic_filter_gate::filter_skippable(*measured));

  // Growth stales a selective reading: new arrivals change the rows reaching the filter.
  REQUIRE(filters_producer.push_filter(0, make_in_list_prefix(4, stream)));
  REQUIRE_FALSE(gate.filter_keep_ratio(selective.get(), filters.filter_count()).has_value());

  // The next apply remeasures it against the larger cascade.
  auto out2 = sirius::op::scan::apply_dynamic_filters_gated_view(
    table->view(),
    filters.snapshot(),
    gate,
    stream,
    dynamic_filter_apply_mode::INCLUDE_AST_ROW_MASKS);
  stream.sync();
  REQUIRE(out2 != nullptr);
  REQUIRE(out2->num_rows() == 2);
  REQUIRE(gate.filter_keep_ratio(selective.get(), filters.filter_count()).has_value());
}

TEST_CASE("dynamic filter compaction policy selects the exact production strategy",
          "[dynamic_filter][scan_merge][compaction_policy]")
{
  using sirius::op::scan::detail::choose_compaction_strategy;
  using sirius::op::scan::detail::compaction_strategy;

  auto choose = [](std::size_t rows,
                   std::optional<std::size_t> bytes,
                   std::vector<std::optional<double>> const& membership,
                   bool has_ast_mask = false) {
    return choose_compaction_strategy({rows, bytes, has_ast_mask, membership});
  };

  std::vector<std::optional<double>> const unknown{std::nullopt, std::nullopt};
  CHECK(choose(100, 8000, {}) == compaction_strategy::CASCADE);
  CHECK(choose(100, 8000, {std::nullopt}) == compaction_strategy::CASCADE);
  CHECK(choose(0, 8000, unknown) == compaction_strategy::CASCADE);
  CHECK(choose(100, std::nullopt, unknown) == compaction_strategy::CASCADE);
  CHECK(choose(100, 0, unknown) == compaction_strategy::CASCADE);
  CHECK(choose(100, 99, unknown) == compaction_strategy::CASCADE);
  CHECK(choose(100, std::numeric_limits<std::size_t>::max(), unknown) ==
        compaction_strategy::CASCADE);

  CHECK(choose(100, 6300, unknown) == compaction_strategy::CASCADE);
  CHECK(choose(100, 6399, unknown) == compaction_strategy::CASCADE);
  CHECK(choose(100, 6400, unknown) == compaction_strategy::DEFERRED_KEYS);
  CHECK(choose(100, 6499, unknown) == compaction_strategy::DEFERRED_KEYS);
  CHECK(choose(100, 8000, unknown) == compaction_strategy::DEFERRED_KEYS);
  CHECK(choose(100, 14400, unknown) == compaction_strategy::DEFERRED_KEYS);

  auto const above_wide_boundary = std::nextafter(0.35, 1.0);
  std::vector<std::optional<double>> const wide_boundary{0.35, 0.8};
  std::vector<std::optional<double>> const wide_weak{above_wide_boundary, 0.8};
  CHECK(choose(100, 6400, wide_boundary) == compaction_strategy::DEFERRED_KEYS);
  CHECK(choose(100, 6400, wide_weak) == compaction_strategy::GATHER_ONCE);
  CHECK(choose(100, 8000, {{0.5}, {0.8}, {0.9}}) == compaction_strategy::GATHER_ONCE);

  auto const below_narrow_boundary = std::nextafter(0.40, 0.0);
  CHECK(choose(100, 3200, {{0.40}, {0.8}}) == compaction_strategy::GATHER_ONCE);
  CHECK(choose(100, 3200, {{below_narrow_boundary}, {0.8}}) == compaction_strategy::CASCADE);
  CHECK(choose(100, 3200, {{0.5}, {0.8}, {0.39}}) == compaction_strategy::CASCADE);

  std::vector<std::optional<double>> const invalid_estimates{
    std::nullopt,
    std::numeric_limits<double>::quiet_NaN(),
    std::numeric_limits<double>::infinity(),
    -0.01,
    1.01};
  for (auto const invalid : invalid_estimates) {
    std::vector<std::optional<double>> const estimates{0.8, invalid};
    CHECK(choose(100, 6400, estimates) == compaction_strategy::DEFERRED_KEYS);
    CHECK(choose(100, 6300, estimates) == compaction_strategy::CASCADE);
  }

  CHECK(choose(100, 8000, {}, /*has_ast_mask=*/true) == compaction_strategy::CASCADE);
  CHECK(choose(100, 8000, {{0.5}}, /*has_ast_mask=*/true) == compaction_strategy::GATHER_ONCE);
}

TEST_CASE("deferred keys support arbitrary repeated membership filters",
          "[dynamic_filter][scan_merge][deferred_keys]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();
  auto table                = make_int64_sequence_table(10, stream);
  sirius_dynamic_filter_set filters;
  auto producer = filters.register_producer({0}, 8);
  for (int64_t count = 10; count >= 3; --count) {
    REQUIRE(producer.push_filter(0, make_in_list_prefix(count, stream)));
  }

  auto out = sirius::op::scan::apply_dynamic_filters_to_view(
    table->view(),
    filters.snapshot(),
    stream,
    dynamic_filter_apply_mode::MEMBERSHIP_MASKS_ONLY,
    nullptr,
    -1,
    /*input_bytes=*/800);
  stream.sync();

  REQUIRE(out != nullptr);
  REQUIRE(out->num_columns() == 1);
  REQUIRE(to_host_int64(out->view().column(0), stream) == std::vector<int64_t>{0, 1, 2});
}

TEST_CASE("compaction strategies equal cascade for AST and INT32 membership",
          "[dynamic_filter][scan_merge][deferred_keys][equivalence]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();
  auto table                = make_sequence_table<int32_t>(10, stream, {{0, 1}, {0, 1}, {100, 1}});
  sirius_dynamic_filter_set filters;
  auto producer = filters.register_producer({0, 1});
  REQUIRE(producer.push_filter(0, make_zone_map(2, 8)));
  REQUIRE(producer.push_filter(1, make_in_list_prefix<int32_t>(5, stream)));

  REQUIRE(forced_strategies_equal(
    table->view(), filters.snapshot(), stream, dynamic_filter_apply_mode::INCLUDE_AST_ROW_MASKS));
}

TEST_CASE("forced compaction strategies agree for an AST-only mask",
          "[dynamic_filter][scan_merge][equivalence]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();
  auto table                = make_sequence_table(10, stream);
  sirius_dynamic_filter_set filters;
  auto producer = filters.register_producer({0});
  REQUIRE(producer.push_filter(0, make_zone_map(3, 7)));

  REQUIRE(forced_strategies_equal(
    table->view(), filters.snapshot(), stream, dynamic_filter_apply_mode::INCLUDE_AST_ROW_MASKS));
}

TEST_CASE("compaction strategies equal cascade for sliced nested nullable payloads",
          "[dynamic_filter][scan_merge][deferred_keys][equivalence]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(make_sequence_column<int64_t>(10, stream));
  columns.push_back(make_sequence_column<int64_t>(10, stream));
  columns.push_back(make_nullable_list_payload(10, stream));
  auto table        = std::make_unique<cudf::table>(std::move(columns));
  auto sliced_input = cudf::slice(table->view(), {1, 9}, stream).front();

  sirius_dynamic_filter_set filters;
  auto producer = filters.register_producer({0, 1});
  REQUIRE(producer.push_filter(0, make_in_list_prefix(8, stream)));
  REQUIRE(producer.push_filter(1, make_in_list_prefix(5, stream)));

  REQUIRE(forced_strategies_equal(
    sliced_input, filters.snapshot(), stream, dynamic_filter_apply_mode::MEMBERSHIP_MASKS_ONLY));
}

TEST_CASE("compaction strategies equal cascade for all-true and empty results",
          "[dynamic_filter][scan_merge][deferred_keys][equivalence]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();

  SECTION("all contributing masks retain every row")
  {
    auto table = make_sequence_table<int64_t>(10, stream, {{0, 1}, {0, 1}, {100, 1}});
    sirius_dynamic_filter_set filters;
    auto producer = filters.register_producer({0, 1});
    REQUIRE(producer.push_filter(0, make_in_list_prefix(10, stream)));
    REQUIRE(producer.push_filter(1, make_in_list_prefix(10, stream)));
    REQUIRE(forced_strategies_equal(
      table->view(), filters.snapshot(), stream, dynamic_filter_apply_mode::MEMBERSHIP_MASKS_ONLY));
  }

  SECTION("contributing masks produce an empty table with the original schema")
  {
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(make_sequence_column<int64_t>(10, stream));
    columns.push_back(make_sequence_column<int64_t>(10, stream, 9, -1));
    columns.push_back(make_nullable_list_payload(10, stream));
    auto table = std::make_unique<cudf::table>(std::move(columns));
    sirius_dynamic_filter_set filters;
    auto producer = filters.register_producer({0, 1});
    REQUIRE(producer.push_filter(0, make_in_list_prefix(3, stream)));
    REQUIRE(producer.push_filter(1, make_in_list_prefix(3, stream)));
    REQUIRE(forced_strategies_equal(
      table->view(), filters.snapshot(), stream, dynamic_filter_apply_mode::MEMBERSHIP_MASKS_ONLY));
  }
}

TEST_CASE("compaction strategies equal cascade for Bloom membership filters",
          "[dynamic_filter][scan_merge][deferred_keys][equivalence]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();
  auto build_five           = make_sequence_column<int64_t>(5, stream);
  auto build_three          = make_sequence_column<int64_t>(3, stream);
  sirius_dynamic_filter_set filters;
  auto producer = filters.register_producer({0});
  REQUIRE(
    producer.push_filter(0,
                         std::make_shared<sirius::op::sirius_dynamic_bloom_filter>(
                           build_five->view(), stream, cudf::get_current_device_resource_ref())));
  REQUIRE(
    producer.push_filter(0,
                         std::make_shared<sirius::op::sirius_dynamic_bloom_filter>(
                           build_three->view(), stream, cudf::get_current_device_resource_ref())));
  auto input = make_int64_sequence_table(20, stream);

  REQUIRE(forced_strategies_equal(
    input->view(), filters.snapshot(), stream, dynamic_filter_apply_mode::MEMBERSHIP_MASKS_ONLY));
}

TEST_CASE("deferred keys preserve null pass-through when every mask declines",
          "[dynamic_filter][scan_merge][deferred_keys][equivalence]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();
  auto input                = make_int64_sequence_table(10, stream);
  sirius_dynamic_filter_set filters;
  auto producer = filters.register_producer({0});
  REQUIRE(producer.push_filter(0, std::make_shared<nullable_declining_filter>()));
  REQUIRE(producer.push_filter(0, std::make_shared<nullable_declining_filter>()));

  REQUIRE(forced_strategies_equal(
    input->view(), filters.snapshot(), stream, dynamic_filter_apply_mode::MEMBERSHIP_MASKS_ONLY));
}

TEST_CASE("gather once declines null masks without training their marginals",
          "[dynamic_filter][scan_merge][gather_once][gate]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();
  auto input                = make_int64_sequence_table(10, stream);
  auto declining            = std::make_shared<nullable_declining_filter>();
  sirius_dynamic_filter_set filters;
  auto producer = filters.register_producer({0});
  REQUIRE(producer.push_filter(0, declining));
  sirius::op::scan::dynamic_filter_gate gate;

  auto output = sirius::op::scan::detail::apply_dynamic_filters_to_view_for_testing(
    input->view(),
    filters.snapshot(),
    stream,
    sirius::op::scan::detail::compaction_strategy::GATHER_ONCE,
    dynamic_filter_apply_mode::MEMBERSHIP_MASKS_ONLY,
    &gate);

  REQUIRE(output == nullptr);
  REQUIRE(declining->mask_calls() == 1);
  REQUIRE_FALSE(gate.filter_keep_ratio(declining.get(), filters.filter_count()).has_value());
}

TEST_CASE("production policy uses gather once after weak marginals become current",
          "[dynamic_filter][scan_merge][gather_once][gate][production_policy]")
{
  // 32-byte narrow rows and 80-byte wide rows both reach gather-once once every marginal is
  // current, so the two width branches of the policy are exercised against the same filters.
  auto const input_bytes    = GENERATE(std::size_t{640}, std::size_t{1600});
  ::cuda::stream_ref stream = cudf::get_default_stream();
  auto input                = make_sequence_table<int64_t>(20, stream, {{0, 1}, {100, 1}});

  // The marginal keeps these two filters train are 9/20 and 4/9. Both stay at or below the gate's
  // skippable cutoff, so neither filter is dropped before strategy selection, and both stay above
  // the wide and narrow gather-once cutoffs.
  auto first  = std::make_shared<counting_in_list_filter>(make_in_list_prefix(9, stream));
  auto second = std::make_shared<counting_in_list_filter>(make_in_list_prefix(4, stream));
  sirius_dynamic_filter_set filters;
  auto producer = filters.register_producer({0}, 3);
  REQUIRE(producer.push_filter(0, first));
  REQUIRE(producer.push_filter(0, second));
  sirius::op::scan::dynamic_filter_gate gate;

  auto const apply = [&] {
    auto out = sirius::op::scan::apply_dynamic_filters_gated_view(
      input->view(),
      filters.snapshot(),
      gate,
      stream,
      dynamic_filter_apply_mode::MEMBERSHIP_MASKS_ONLY,
      -1,
      input_bytes);
    stream.sync();
    return out;
  };

  auto trained = apply();
  REQUIRE(trained != nullptr);
  REQUIRE(to_host_int64(trained->view().column(0), stream) == std::vector<int64_t>{0, 1, 2, 3});
  auto const first_keep  = gate.filter_keep_ratio(first.get(), filters.filter_count());
  auto const second_keep = gate.filter_keep_ratio(second.get(), filters.filter_count());
  REQUIRE(first_keep == Approx(0.45));
  REQUIRE(second_keep == Approx(4.0 / 9.0));
  REQUIRE_FALSE(sirius::op::scan::dynamic_filter_gate::filter_skippable(*first_keep));
  REQUIRE_FALSE(sirius::op::scan::dynamic_filter_gate::filter_skippable(*second_keep));

  auto gathered_once = apply();

  REQUIRE(gathered_once != nullptr);
  REQUIRE(to_host_int64(gathered_once->view().column(0), stream) ==
          std::vector<int64_t>{0, 1, 2, 3});
  REQUIRE(to_host_int64(gathered_once->view().column(1), stream) ==
          std::vector<int64_t>{100, 101, 102, 103});
  // The training apply probed the second filter in survivor space; the gather-once apply probed
  // both filters against all 20 original rows and left their measurements untouched.
  REQUIRE(first->probe_rows() == std::vector<cudf::size_type>{20, 20});
  REQUIRE(second->probe_rows() == std::vector<cudf::size_type>{9, 20});
  REQUIRE(gate.filter_keep_ratio(first.get(), filters.filter_count()) == first_keep);
  REQUIRE(gate.filter_keep_ratio(second.get(), filters.filter_count()) == second_keep);

  // Channel growth stales both measurements, so gather-once becomes ineligible and the next apply
  // has to retrain in survivor space before it can be chosen again.
  REQUIRE(producer.push_filter(0, make_in_list_prefix(3, stream)));
  REQUIRE_FALSE(gate.filter_keep_ratio(first.get(), filters.filter_count()).has_value());
  REQUIRE_FALSE(gate.filter_keep_ratio(second.get(), filters.filter_count()).has_value());

  auto retrained = apply();

  REQUIRE(retrained != nullptr);
  REQUIRE(to_host_int64(retrained->view().column(0), stream) == std::vector<int64_t>{0, 1, 2});
  REQUIRE(second->probe_rows() == std::vector<cudf::size_type>{9, 20, 9});
  REQUIRE(gate.filter_keep_ratio(second.get(), filters.filter_count()).has_value());
}

TEST_CASE("deferred keys preserve alignment across repeated and different columns",
          "[dynamic_filter][scan_merge][deferred_keys]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();
  auto table = make_sequence_table<int64_t>(10, stream, {{0, 1}, {9, -1}, {100, 1}, {200, 1}});

  sirius_dynamic_filter_set filters;
  auto producer = filters.register_producer({0, 1});
  REQUIRE(producer.push_filter(0, make_in_list_prefix(8, stream)));
  REQUIRE(producer.push_filter(1, make_in_list_prefix(6, stream)));
  REQUIRE(producer.push_filter(1, make_in_list_prefix(4, stream)));

  auto out = sirius::op::scan::apply_dynamic_filters_to_view(
    table->view(),
    filters.snapshot(),
    stream,
    dynamic_filter_apply_mode::MEMBERSHIP_MASKS_ONLY,
    nullptr,
    -1,
    /*input_bytes=*/800);
  stream.sync();

  REQUIRE(out != nullptr);
  REQUIRE(out->num_columns() == 4);
  REQUIRE(to_host_int64(out->view().column(0), stream) == std::vector<int64_t>{6, 7});
  REQUIRE(to_host_int64(out->view().column(1), stream) == std::vector<int64_t>{3, 2});
  REQUIRE(to_host_int64(out->view().column(2), stream) == std::vector<int64_t>{106, 107});
  REQUIRE(to_host_int64(out->view().column(3), stream) == std::vector<int64_t>{206, 207});
}

TEST_CASE("deferred keys realign after nullable mask decline without training it",
          "[dynamic_filter][scan_merge][deferred_keys]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();
  auto table                = make_sequence_table<int64_t>(10, stream, {{0, 1}, {0, 1}, {100, 1}});
  auto& nullable_key        = table->get_column(0);
  nullable_key.set_null_mask(cudf::create_null_mask(10, cudf::mask_state::ALL_VALID, stream), 1);
  cudf::set_null_mask(nullable_key.mutable_view().null_mask(), 0, 1, false, stream);

  auto declining = std::make_shared<nullable_declining_filter>();
  auto selective = make_in_list_prefix(3, stream);
  sirius_dynamic_filter_set filters;
  auto producer = filters.register_producer({0, 1});
  REQUIRE(producer.push_filter(0, declining));
  REQUIRE(producer.push_filter(1, selective));
  sirius::op::scan::dynamic_filter_gate gate;

  auto out = sirius::op::scan::apply_dynamic_filters_to_view(
    table->view(),
    filters.snapshot(),
    stream,
    dynamic_filter_apply_mode::MEMBERSHIP_MASKS_ONLY,
    &gate,
    -1,
    /*input_bytes=*/800);
  stream.sync();

  REQUIRE(out != nullptr);
  REQUIRE(declining->mask_calls() == 1);
  REQUIRE(declining->saw_nulls());
  REQUIRE_FALSE(gate.filter_keep_ratio(declining.get(), filters.filter_count()).has_value());
  REQUIRE(gate.filter_keep_ratio(selective.get(), filters.filter_count()) == Approx(0.3));
  REQUIRE(out->view().column(0).null_count() == 1);
  REQUIRE(to_host_int64(out->view().column(1), stream) == std::vector<int64_t>{0, 1, 2});
  REQUIRE(to_host_int64(out->view().column(2), stream) == std::vector<int64_t>{100, 101, 102});
}

TEST_CASE("wide rows keep weak measured filters that narrow rows skip",
          "[dynamic_filter][scan_merge][gather_once][gate][production_policy]")
{
  // 20 rows at 80 bytes are wide; 20 rows at 32 bytes are narrow.
  auto const [input_bytes, wide] =
    GENERATE(std::pair{std::size_t{1600}, true}, std::pair{std::size_t{640}, false});
  ::cuda::stream_ref stream = cudf::get_default_stream();
  auto input                = make_sequence_table<int64_t>(20, stream, {{0, 1}, {100, 1}});

  // Marginal keeps of 11/20 and 6/11: both above the gate's 50% cutoff, both at or below 60%.
  auto first  = std::make_shared<counting_in_list_filter>(make_in_list_prefix(11, stream));
  auto second = std::make_shared<counting_in_list_filter>(make_in_list_prefix(6, stream));
  sirius_dynamic_filter_set filters;
  auto producer = filters.register_producer({0});
  REQUIRE(producer.push_filter(0, first));
  REQUIRE(producer.push_filter(0, second));
  sirius::op::scan::dynamic_filter_gate gate;

  auto const apply = [&] {
    auto out = sirius::op::scan::apply_dynamic_filters_gated_view(
      input->view(),
      filters.snapshot(),
      gate,
      stream,
      dynamic_filter_apply_mode::MEMBERSHIP_MASKS_ONLY,
      -1,
      input_bytes);
    stream.sync();
    return out;
  };

  auto trained = apply();
  REQUIRE(trained != nullptr);
  REQUIRE(trained->num_rows() == 6);
  auto const first_keep  = gate.filter_keep_ratio(first.get(), filters.filter_count());
  auto const second_keep = gate.filter_keep_ratio(second.get(), filters.filter_count());
  REQUIRE(first_keep == Approx(0.55));
  REQUIRE(second_keep == Approx(6.0 / 11.0));
  REQUIRE(sirius::op::scan::dynamic_filter_gate::filter_skippable(*first_keep));
  REQUIRE(sirius::op::scan::dynamic_filter_gate::filter_skippable(*second_keep));

  auto again = apply();

  if (wide) {
    // Both weak filters stay, every marginal is known, so gather once probes the original rows.
    REQUIRE(again != nullptr);
    REQUIRE(to_host_int64(again->view().column(0), stream) ==
            std::vector<int64_t>{0, 1, 2, 3, 4, 5});
    REQUIRE(to_host_int64(again->view().column(1), stream) ==
            std::vector<int64_t>{100, 101, 102, 103, 104, 105});
    REQUIRE(first->probe_rows() == std::vector<cudf::size_type>{20, 20});
    REQUIRE(second->probe_rows() == std::vector<cudf::size_type>{11, 20});
  } else {
    // Narrow rows skip both, so nothing is probed and the batch passes through unchanged.
    REQUIRE(again == nullptr);
    REQUIRE(first->probe_rows() == std::vector<cudf::size_type>{20});
    REQUIRE(second->probe_rows() == std::vector<cudf::size_type>{11});
  }
}

TEST_CASE("wide rows skip measured filters that keep more than sixty percent",
          "[dynamic_filter][scan_merge][gate][production_policy]")
{
  auto const [prefix, applied] =
    GENERATE(std::pair{int64_t{12}, true}, std::pair{int64_t{13}, false});
  ::cuda::stream_ref stream = cudf::get_default_stream();
  auto input                = make_int64_sequence_table(20, stream);
  auto filter = std::make_shared<counting_in_list_filter>(make_in_list_prefix(prefix, stream));
  sirius_dynamic_filter_set filters;
  auto producer = filters.register_producer({0});
  REQUIRE(producer.push_filter(0, filter));
  sirius::op::scan::dynamic_filter_gate gate;
  constexpr std::size_t wide_bytes = 1600;

  auto const apply = [&] {
    auto out = sirius::op::scan::apply_dynamic_filters_gated_view(
      input->view(),
      filters.snapshot(),
      gate,
      stream,
      dynamic_filter_apply_mode::MEMBERSHIP_MASKS_ONLY,
      -1,
      wide_bytes);
    stream.sync();
    return out;
  };

  auto trained = apply();
  REQUIRE(trained != nullptr);
  REQUIRE(trained->num_rows() == prefix);

  auto again = apply();

  if (applied) {
    REQUIRE(again != nullptr);
    REQUIRE(again->num_rows() == prefix);
    REQUIRE(filter->mask_calls() == 2);
  } else {
    REQUIRE(again == nullptr);
    REQUIRE(filter->mask_calls() == 1);
  }
}
