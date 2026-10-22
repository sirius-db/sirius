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
#include "operator/operator_test_utils.hpp"
#include "utils/host_allocation_fault.hpp"
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
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/error.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <cuda/stream>
#include <cuda_runtime.h>

#include <catch.hpp>
#include <cucascade/memory/config.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/null_device_memory_resource.hpp>
#include <data/data_batch_utils.hpp>
#include <duckdb/planner/filter/constant_filter.hpp>
#include <op/dynamic_filter/detail/accumulated_bloom_builder.hpp>
#include <op/dynamic_filter/detail/accumulation_failure.hpp>
#include <op/dynamic_filter/dynamic_filter_mask_ops.hpp>
#include <op/dynamic_filter/dynamic_filter_stats.hpp>
#include <op/dynamic_filter/sirius_dynamic_filter.hpp>
#include <op/scan/detail/read_retirement.hpp>
#include <op/scan/dynamic_filter_merge.hpp>
#include <op/scan/parquet_gpu_ingestible.hpp>
#include <op/scan/scan_plan.hpp>
#include <op/scan/sirius_gpu_scan_operator.hpp>
#include <op/scan/sirius_gpu_scan_operator_data.hpp>
#include <sirius/exception.hpp>
#include <utils/table_filter_test_utils.hpp>

#include <algorithm>
#include <array>
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
#include <optional>
#include <span>
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
// dynamic_filter_consumer — runtime apply (post-decode / cached)
//===----------------------------------------------------------------------===//

namespace {
using sirius::op::scan::application_input;
using sirius::op::scan::application_program;
using sirius::op::scan::application_result;
using sirius::op::scan::batch_receipt;
using sirius::op::scan::binding;
using sirius::op::scan::consumer_config;
using sirius::op::scan::dynamic_filter_consumer;
using sirius::op::scan::detail::compaction_strategy;
using sirius::op::scan::detail::consumer_test_access;
using sirius::op::scan::detail::execution_fault;
using sirius::op::scan::detail::gate_test_access;

std::optional<application_program> prepare_identity(
  dynamic_filter_consumer const& consumer,
  cudf::table_view input,
  std::optional<std::size_t> input_bytes = std::nullopt,
  int device_id                          = -1)
{
  auto const bindings =
    sirius::op::scan::identity_bindings(static_cast<std::size_t>(input.num_columns()));
  return consumer.prepare(
    {.source = input, .bindings = bindings, .input_bytes = input_bytes, .device_id = device_id});
}

std::unique_ptr<cudf::table> apply_consumer(
  dynamic_filter_consumer& consumer,
  std::shared_ptr<cudf::table const> input,
  ::cuda::stream_ref stream,
  std::optional<std::size_t> input_bytes     = std::nullopt,
  int device_id                              = -1,
  std::optional<cudf::table_view> input_view = std::nullopt)
{
  auto program =
    prepare_identity(consumer, input_view.value_or(input->view()), input_bytes, device_id);
  if (!program) { return nullptr; }
  auto result = consumer_test_access::apply(consumer,
                                            std::move(*program),
                                            std::move(input),
                                            stream,
                                            cudf::get_current_device_resource_ref(),
                                            execution_fault::NONE,
                                            nullptr,
                                            input_view);
  auto output = result.take_output();
  static_cast<void>(consumer.finish(std::move(result), 0));
  return output;
}

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

TEST_CASE("dynamic_filter_consumer drops rows outside the zone", "[dynamic_filter][scan_merge]")
{
  ::cuda::stream_ref stream          = cudf::get_default_stream();
  std::shared_ptr<cudf::table> table = make_sequence_table(10, stream);  // [0..9]

  auto filters = std::make_shared<sirius_dynamic_filter_set>();
  dynamic_filter_consumer consumer(
    filters, {.mode = dynamic_filter_apply_mode::INCLUDE_AST_ROW_MASKS}, {});
  auto filters_producer = filters->register_producer({0});
  REQUIRE(filters_producer.push_filter(0, make_zone_map(3, 6)));  // inclusive [3,6] keeps 3,4,5,6

  auto out = apply_consumer(consumer, table, stream);
  stream.sync();

  REQUIRE(out != nullptr);
  REQUIRE(out->num_columns() == 1);
  REQUIRE(out->num_rows() == 4);
  REQUIRE(to_host_int32(out->view().column(0), stream) == std::vector<int32_t>{3, 4, 5, 6});
  REQUIRE(table->num_rows() == 10);  // input untouched
}

TEST_CASE("dynamic_filter_consumer honors an exclusive upper bound [lo, hi)",
          "[dynamic_filter][scan_merge]")
{
  ::cuda::stream_ref stream          = cudf::get_default_stream();
  std::shared_ptr<cudf::table> table = make_sequence_table(10, stream);  // [0..9]

  auto filters = std::make_shared<sirius_dynamic_filter_set>();
  dynamic_filter_consumer consumer(
    filters, {.mode = dynamic_filter_apply_mode::INCLUDE_AST_ROW_MASKS}, {});
  auto filters_producer = filters->register_producer({0});
  // [3,6): inclusive_min, exclusive_max -> GREATER_EQUAL(3) AND LESS(6) -> {3,4,5}
  REQUIRE(filters_producer.push_filter(
    0, make_zone_map(3, 6, /*inclusive_min=*/true, /*inclusive_max=*/false)));

  auto out = apply_consumer(consumer, table, stream);
  stream.sync();

  REQUIRE(out != nullptr);
  REQUIRE(to_host_int32(out->view().column(0), stream) == std::vector<int32_t>{3, 4, 5});
}

TEST_CASE("dynamic_filter_consumer honors an exclusive lower bound (lo, hi]",
          "[dynamic_filter][scan_merge]")
{
  ::cuda::stream_ref stream          = cudf::get_default_stream();
  std::shared_ptr<cudf::table> table = make_sequence_table(10, stream);  // [0..9]

  auto filters = std::make_shared<sirius_dynamic_filter_set>();
  dynamic_filter_consumer consumer(
    filters, {.mode = dynamic_filter_apply_mode::INCLUDE_AST_ROW_MASKS}, {});
  auto filters_producer = filters->register_producer({0});
  // (3,6]: exclusive_min, inclusive_max -> GREATER(3) AND LESS_EQUAL(6) -> {4,5,6}
  REQUIRE(filters_producer.push_filter(
    0, make_zone_map(3, 6, /*inclusive_min=*/false, /*inclusive_max=*/true)));

  auto out = apply_consumer(consumer, table, stream);
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
  auto build   = make_sequence_column<int32_t>(101, stream, 100);
  auto filters = std::make_shared<sirius_dynamic_filter_set>();
  dynamic_filter_consumer consumer(
    filters, {.mode = dynamic_filter_apply_mode::INCLUDE_AST_ROW_MASKS}, {});
  auto filters_producer = filters->register_producer({0});
  REQUIRE(filters_producer.push_filter(0, make_zone_map_from_reduce(build->view(), stream)));

  std::shared_ptr<cudf::table> probe = make_sequence_table(300, stream);  // [0..299]
  auto out                           = apply_consumer(consumer, probe, stream);
  stream.sync();

  REQUIRE(out != nullptr);
  std::vector<int32_t> expected(101);
  std::iota(expected.begin(), expected.end(), 100);  // {100, 101, ..., 200}, inclusive bounds
  REQUIRE(to_host_int32(out->view().column(0), stream) == expected);
}

TEST_CASE("dynamic_filter_consumer bypasses an empty channel", "[dynamic_filter][scan_merge]")
{
  ::cuda::stream_ref stream          = cudf::get_default_stream();
  std::shared_ptr<cudf::table> table = make_sequence_table(10, stream);

  auto filters = std::make_shared<sirius_dynamic_filter_set>();  // empty
  dynamic_filter_consumer consumer(
    filters, {.mode = dynamic_filter_apply_mode::INCLUDE_AST_ROW_MASKS}, {});

  auto out = apply_consumer(consumer, table, stream);
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

TEST_CASE("dynamic_filter_consumer AND-conjoins multiple zone filters on a column",
          "[dynamic_filter][scan_merge]")
{
  ::cuda::stream_ref stream          = cudf::get_default_stream();
  std::shared_ptr<cudf::table> table = make_sequence_table(10, stream);  // [0..9]

  auto filters = std::make_shared<sirius_dynamic_filter_set>();
  dynamic_filter_consumer consumer(
    filters, {.mode = dynamic_filter_apply_mode::INCLUDE_AST_ROW_MASKS}, {});
  auto filters_producer = filters->register_producer({0});
  REQUIRE(filters_producer.push_filter(0, make_zone_map(2, 8)));  // keeps 2..8
  REQUIRE(
    filters_producer.push_filter(0, make_zone_map(5, 9)));  // AND keeps 5..9 → intersection 5..8

  auto out = apply_consumer(consumer, table, stream);
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
  auto keys    = make_sequence_column<int64_t>(5, stream);
  auto filters = std::make_shared<sirius_dynamic_filter_set>();
  dynamic_filter_consumer consumer(
    filters, {.mode = dynamic_filter_apply_mode::INCLUDE_AST_ROW_MASKS}, {});
  auto filters_producer = filters->register_producer({0});
  REQUIRE(
    filters_producer.push_filter(0,
                                 std::make_shared<sirius::op::sirius_dynamic_in_list_filter>(
                                   keys->view(), stream, cudf::get_current_device_resource_ref())));

  std::shared_ptr<cudf::table> table = make_int64_sequence_table(10, stream);
  auto out                           = apply_consumer(consumer, table, stream);
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

  auto filters = std::make_shared<sirius_dynamic_filter_set>();
  dynamic_filter_consumer consumer(
    filters, {.mode = dynamic_filter_apply_mode::INCLUDE_AST_ROW_MASKS}, {});
  auto filters_producer = filters->register_producer({0});
  REQUIRE(filters_producer.push_filter(0, filter));

  // Two applies against the same persistent structure (the per-split pattern).
  for (int i = 0; i < 2; ++i) {
    std::shared_ptr<cudf::table> table = make_int64_sequence_table(10, stream);
    auto out                           = apply_consumer(consumer, table, stream);
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
  auto filters              = std::make_shared<sirius_dynamic_filter_set>();
  dynamic_filter_consumer consumer(
    filters, {.mode = dynamic_filter_apply_mode::INCLUDE_AST_ROW_MASKS}, {});
  auto filters_producer = filters->register_producer({0});
  REQUIRE(
    filters_producer.push_filter(0,
                                 std::make_shared<sirius::op::sirius_dynamic_bloom_filter>(
                                   keys->view(), stream, cudf::get_current_device_resource_ref())));

  std::shared_ptr<cudf::table> table = make_int64_sequence_table(2 * num_keys, stream);
  auto out                           = apply_consumer(consumer, table, stream);
  stream.sync();
  REQUIRE(out != nullptr);
  auto const survivors = to_host_int64(out->view().column(0), stream);
  REQUIRE(survivors.size() >= static_cast<std::size_t>(num_keys));
  REQUIRE(survivors.size() <= static_cast<std::size_t>(2 * num_keys));
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

  auto filters = std::make_shared<sirius_dynamic_filter_set>();
  dynamic_filter_consumer consumer(
    filters, {.mode = dynamic_filter_apply_mode::INCLUDE_AST_ROW_MASKS}, {});
  auto filters_producer = filters->register_producer({0});
  REQUIRE(filters_producer.push_filter(0, filter));

  std::shared_ptr<cudf::table> table = make_sequence_table(10, stream);  // INT32 [0..9]
  auto out                           = apply_consumer(consumer, table, stream);
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
  auto filters              = std::make_shared<sirius_dynamic_filter_set>();
  dynamic_filter_consumer consumer(
    filters, {.mode = dynamic_filter_apply_mode::INCLUDE_AST_ROW_MASKS}, {});
  auto filters_producer = filters->register_producer({0});
  REQUIRE(
    filters_producer.push_filter(0,
                                 std::make_shared<sirius::op::sirius_dynamic_bloom_filter>(
                                   keys->view(), stream, cudf::get_current_device_resource_ref())));

  std::shared_ptr<cudf::table> table = make_sequence_table(2 * num_keys, stream);
  auto out                           = apply_consumer(consumer, table, stream);
  stream.sync();
  REQUIRE(out != nullptr);
  auto const survivors = to_host_int32(out->view().column(0), stream);
  REQUIRE(survivors.size() >= static_cast<std::size_t>(num_keys));
  REQUIRE(survivors.size() <= static_cast<std::size_t>(2 * num_keys));
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

  auto nullable_channel = std::make_shared<sirius_dynamic_filter_set>();
  dynamic_filter_consumer nullable_channel_consumer(
    nullable_channel, {.mode = dynamic_filter_apply_mode::INCLUDE_AST_ROW_MASKS}, {});
  auto nullable_channel_producer = nullable_channel->register_producer({0});
  REQUIRE(nullable_channel_producer.push_filter(
    0,
    std::make_shared<sirius::op::sirius_dynamic_bloom_filter>(
      keys->view(), stream, cudf::get_current_device_resource_ref())));
  auto reference_channel = std::make_shared<sirius_dynamic_filter_set>();
  dynamic_filter_consumer reference_channel_consumer(
    reference_channel, {.mode = dynamic_filter_apply_mode::INCLUDE_AST_ROW_MASKS}, {});
  auto reference_channel_producer = reference_channel->register_producer({0});
  REQUIRE(reference_channel_producer.push_filter(
    0,
    std::make_shared<sirius::op::sirius_dynamic_bloom_filter>(
      clean_keys->view(), stream, cudf::get_current_device_resource_ref())));

  // Probe [0..9] plus 999 — the value present only at the null build slot.
  std::shared_ptr<cudf::table> probe = make_values_table<int64_t>(
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 999}, cudf::data_type{cudf::type_id::INT64}, stream);
  auto out_nullable  = apply_consumer(nullable_channel_consumer, probe, stream);
  auto out_reference = apply_consumer(reference_channel_consumer, probe, stream);
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

  auto filters = std::make_shared<sirius_dynamic_filter_set>();
  dynamic_filter_consumer consumer(
    filters, {.mode = dynamic_filter_apply_mode::INCLUDE_AST_ROW_MASKS}, {});
  auto filters_producer = filters->register_producer({0});
  REQUIRE(filters_producer.push_filter(0, filter));

  // Probe {INT64_MIN, 2, 7}: INT64_MIN and 2 are build keys and must survive; 7 must be dropped.
  std::shared_ptr<cudf::table> probe =
    make_values_table<int64_t>({std::numeric_limits<int64_t>::min(), 2, 7}, dtype, stream);
  auto out = apply_consumer(consumer, probe, stream);
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

  auto filters = std::make_shared<sirius_dynamic_filter_set>();
  dynamic_filter_consumer consumer(
    filters, {.mode = dynamic_filter_apply_mode::INCLUDE_AST_ROW_MASKS}, {});
  auto filters_producer = filters->register_producer({0});
  REQUIRE(filters_producer.push_filter(0, filter));

  std::shared_ptr<cudf::table> probe =
    make_values_table<int32_t>({std::numeric_limits<int32_t>::min(), 2, 7}, dtype, stream);
  auto out = apply_consumer(consumer, probe, stream);
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
  auto keys    = make_sequence_column<int64_t>(5, stream);
  auto filters = std::make_shared<sirius_dynamic_filter_set>();
  dynamic_filter_consumer consumer(
    filters, {.mode = dynamic_filter_apply_mode::INCLUDE_AST_ROW_MASKS}, {});
  auto filters_producer = filters->register_producer({0});
  REQUIRE(
    filters_producer.push_filter(0,
                                 std::make_shared<sirius::op::sirius_dynamic_small_in_list_filter>(
                                   keys->view(), stream, cudf::get_current_device_resource_ref())));

  std::shared_ptr<cudf::table> table = make_int64_sequence_table(10, stream);
  auto out                           = apply_consumer(consumer, table, stream);
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
  auto filters              = std::make_shared<sirius_dynamic_filter_set>();
  dynamic_filter_consumer consumer(
    filters, {.mode = dynamic_filter_apply_mode::INCLUDE_AST_ROW_MASKS}, {});
  auto filters_producer = filters->register_producer({0});
  REQUIRE(
    filters_producer.push_filter(0,
                                 std::make_shared<sirius::op::sirius_dynamic_small_in_list_filter>(
                                   keys->view(), stream, cudf::get_current_device_resource_ref())));

  std::shared_ptr<cudf::table> table = make_sequence_table(10, stream);  // INT32 [0..9]
  auto out                           = apply_consumer(consumer, table, stream);
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
  auto keys    = make_values_table<int32_t>({std::numeric_limits<int32_t>::min()}, dtype, stream);
  auto filters = std::make_shared<sirius_dynamic_filter_set>();
  dynamic_filter_consumer consumer(
    filters, {.mode = dynamic_filter_apply_mode::INCLUDE_AST_ROW_MASKS}, {});
  auto filters_producer = filters->register_producer({0});
  REQUIRE(filters_producer.push_filter(
    0,
    std::make_shared<sirius::op::sirius_dynamic_small_in_list_filter>(
      keys->view().column(0), stream, cudf::get_current_device_resource_ref())));

  // Probe {INT32_MIN, INT32_MIN+1, INT32_MIN+2}; only the first is a build key.
  std::shared_ptr<cudf::table> probe =
    make_values_table<int32_t>({std::numeric_limits<int32_t>::min(),
                                std::numeric_limits<int32_t>::min() + 1,
                                std::numeric_limits<int32_t>::min() + 2},
                               dtype,
                               stream);
  auto out = apply_consumer(consumer, probe, stream);
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
// dynamic_filter_consumer — view-based core (pinned cached path)
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

bool forced_strategies_equal(std::shared_ptr<cudf::table const> input,
                             std::shared_ptr<sirius_dynamic_filter_set> const& channel,
                             ::cuda::stream_ref stream,
                             dynamic_filter_apply_mode mode,
                             std::optional<cudf::table_view> input_view = std::nullopt)
{
  auto const apply = [&](compaction_strategy strategy) {
    dynamic_filter_consumer consumer(
      channel,
      {.mode = mode, .strategy_override = strategy, .validate_deferred_selection = true},
      {});
    return apply_consumer(consumer, input, stream, std::nullopt, -1, input_view);
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
  explicit counting_in_list_filter(std::shared_ptr<sirius::op::sirius_dynamic_in_list_filter> inner,
                                   bool fuse = false)
    : _inner(std::move(inner)), _fuse(fuse)
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

  /// Counts the hint and probes through the unhinted default, which compute_mask counts too.
  [[nodiscard]] std::unique_ptr<cudf::column> compute_mask_if(
    cudf::column_view const& probe,
    cudf::column_view const& prior_keep,
    int device_id,
    ::cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) const override
  {
    ++_hinted_calls;
    return sirius_mask_applicable::compute_mask_if(probe, prior_keep, device_id, stream, mr);
  }

  bool prepare_fused_probe(cudf::column_view const& probe,
                           int device,
                           sirius::op::detail::fused_probe& out) const override
  {
    return _fuse && _inner->prepare_fused_probe(probe, device, out);
  }

  [[nodiscard]] int mask_calls() const noexcept { return _mask_calls; }
  [[nodiscard]] int hinted_calls() const noexcept { return _hinted_calls; }
  [[nodiscard]] std::vector<cudf::size_type> const& probe_rows() const noexcept
  {
    return _probe_rows;
  }

 private:
  std::shared_ptr<sirius::op::sirius_dynamic_in_list_filter> _inner;
  bool _fuse;
  mutable int _mask_calls   = 0;
  mutable int _hinted_calls = 0;
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

TEST_CASE("dynamic_filter_consumer leaves input untouched when no filter contributes",
          "[dynamic_filter][scan_merge]")
{
  ::cuda::stream_ref stream          = cudf::get_default_stream();
  std::shared_ptr<cudf::table> table = make_int64_sequence_table(10, stream);

  auto filters = std::make_shared<sirius_dynamic_filter_set>();  // empty
  dynamic_filter_consumer consumer(
    filters, {.mode = dynamic_filter_apply_mode::INCLUDE_AST_ROW_MASKS}, {});

  auto out = apply_consumer(consumer, table, stream);
  REQUIRE(out == nullptr);
  REQUIRE(table->num_rows() == 10);  // input untouched, still usable
}

TEST_CASE("dynamic_filter_consumer gathers survivors without consuming the input",
          "[dynamic_filter][scan_merge]")
{
  ::cuda::stream_ref stream          = cudf::get_default_stream();
  std::shared_ptr<cudf::table> table = make_int64_sequence_table(10, stream);  // [0..9]

  auto filters = std::make_shared<sirius_dynamic_filter_set>();
  dynamic_filter_consumer consumer(
    filters, {.mode = dynamic_filter_apply_mode::INCLUDE_AST_ROW_MASKS}, {});
  auto filters_producer = filters->register_producer({0});
  REQUIRE(filters_producer.push_filter(0, make_in_list_prefix(2, stream)));  // keeps 0,1

  auto out = apply_consumer(consumer, table, stream);
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
  REQUIRE_FALSE(gate_test_access::applicable(gate, filters.snapshot()));
}

TEST_CASE("dynamic filter application keeps one captured generation across later publications",
          "[dynamic_filter][scan_merge][snapshot]")
{
  auto const stream                        = cudf::get_default_stream();
  std::shared_ptr<cudf::table const> input = make_int64_sequence_table(10, stream);
  auto filters                             = std::make_shared<sirius_dynamic_filter_set>();
  dynamic_filter_consumer consumer(filters, {}, {});
  auto producer = filters->register_producer({0});
  REQUIRE(producer.push_filter(0, make_in_list_prefix(7, stream)));
  auto const first   = filters->snapshot();
  auto first_program = prepare_identity(consumer, input->view());
  REQUIRE(first_program.has_value());
  REQUIRE(producer.push_filter(0, make_in_list_prefix(2, stream)));
  auto const second   = filters->snapshot();
  auto second_program = prepare_identity(consumer, input->view());
  REQUIRE(second_program.has_value());

  auto first_result = consumer_test_access::apply(
    consumer, std::move(*first_program), input, stream, cudf::get_current_device_resource_ref());
  auto second_result = consumer_test_access::apply(
    consumer, std::move(*second_program), input, stream, cudf::get_current_device_resource_ref());
  auto first_output  = first_result.take_output();
  auto second_output = second_result.take_output();
  static_cast<void>(consumer.finish(std::move(first_result), 1));
  static_cast<void>(consumer.finish(std::move(second_result), 2));
  REQUIRE(first.generation() == 1);
  REQUIRE(second.generation() == 2);
  REQUIRE(first_output->num_rows() == 7);
  REQUIRE(second_output->num_rows() == 2);

  sirius::op::scan::dynamic_filter_gate gate;
  gate.record_keep_ratio(10, 10, first.generation());
  REQUIRE_FALSE(gate_test_access::applicable(gate, first));
  REQUIRE(gate_test_access::applicable(gate, second));
}

TEST_CASE("prepared dynamic filter programs retain their channel and filter owners",
          "[dynamic_filter][scan_merge][snapshot]")
{
  auto const stream                        = cudf::get_default_stream();
  std::shared_ptr<cudf::table const> input = make_int64_sequence_table(10, stream);
  std::optional<application_program> program;
  std::weak_ptr<sirius_dynamic_filter_set> channel_lifetime;
  std::weak_ptr<sirius_dynamic_filter const> filter_lifetime;
  {
    auto filters     = std::make_shared<sirius_dynamic_filter_set>();
    channel_lifetime = filters;
    dynamic_filter_consumer consumer(filters, {}, {});
    auto producer   = filters->register_producer({0});
    auto filter     = make_in_list_prefix(2, stream);
    filter_lifetime = filter;
    REQUIRE(producer.push_filter(0, std::move(filter)));
    program = prepare_identity(consumer, input->view());
    REQUIRE(program.has_value());
  }
  REQUIRE_FALSE(channel_lifetime.expired());
  REQUIRE_FALSE(filter_lifetime.expired());
  {
    dynamic_filter_consumer consumer(channel_lifetime.lock(), {}, {});
    auto result = consumer_test_access::apply(
      consumer, std::move(*program), input, stream, cudf::get_current_device_resource_ref());
    auto output = result.take_output();
    REQUIRE(output->num_rows() == 2);
    static_cast<void>(consumer.finish(std::move(result), 0));
  }
  program.reset();
  REQUIRE(channel_lifetime.expired());
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
    auto filters = std::make_shared<sirius_dynamic_filter_set>();
    sirius::op::scan::dynamic_filter_consumer consumer(filters, {}, {});
    auto producer   = filters->register_producer({0});
    auto filter     = make_in_list_prefix(7, stream);
    filter_lifetime = filter;
    REQUIRE(producer.push_filter(0, std::move(filter)));
    sirius::op::scan::detail::consumer_test_access::gate(consumer).record_keep_ratio(10, 7, 1);
    std::vector<sirius::op::scan::binding> bindings{{0, 0}};
    probes = sirius::op::scan::snapshot_membership_probes(consumer.select_for_decode(bindings), 2);
    REQUIRE(producer.push_filter(0, make_in_list_prefix(2, stream)));
    REQUIRE(probes.generation == 1);
    REQUIRE(probes.attached.size() == 1);
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
  std::shared_ptr<cudf::table const> input =
    make_sequence_table<int64_t>(10, stream, {{0, 1}, {100, 1}});
  int device_id = -1;
  REQUIRE(cudaGetDevice(&device_id) == cudaSuccess);
  blocked_filter_work work;
  auto filters  = std::make_shared<sirius_dynamic_filter_set>();
  auto consumer = std::make_unique<dynamic_filter_consumer>(
    filters,
    consumer_config{.mode                        = dynamic_filter_apply_mode::MEMBERSHIP_MASKS_ONLY,
                    .strategy_override           = strategy,
                    .validate_deferred_selection = true},
    std::vector<binding>{});
  std::optional<application_program> program;
  std::weak_ptr<sirius_dynamic_filter const> filter_lifetime;
  {
    auto producer = filters->register_producer({0, 1});
    REQUIRE(producer.push_filter(0, make_in_list_prefix(8, stream)));
    auto filter     = std::make_shared<throwing_mask_filter>(work);
    filter_lifetime = filter;
    REQUIRE(producer.push_filter(1, std::move(filter)));
    program = prepare_identity(*consumer, input->view());
    REQUIRE(program.has_value());
  }
  auto started = work.started.get_future();
  std::promise<void> returned;
  auto result = returned.get_future();
  std::jthread worker([&] {
    try {
      rmm::cuda_set_device_raii device{rmm::cuda_device_id{device_id}};
      (void)consumer_test_access::apply(
        *consumer, std::move(*program), input, stream, cudf::get_current_device_resource_ref());
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
  REQUIRE_FALSE(filter_lifetime.expired());
  REQUIRE_FALSE(gate_test_access::filter_keep_ratio(consumer_test_access::gate(*consumer),
                                                    {filter_lifetime.lock().get(), 1},
                                                    filters->filter_count()));
  program.reset();
  consumer.reset();
  filters.reset();
  REQUIRE(filter_lifetime.expired());
}

TEST_CASE("dynamic_filter_gate disables after an unselective first split",
          "[dynamic_filter][scan_merge]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();

  auto filters = std::make_shared<sirius_dynamic_filter_set>();
  dynamic_filter_consumer consumer(
    filters, {.mode = dynamic_filter_apply_mode::INCLUDE_AST_ROW_MASKS}, {});
  auto& gate            = consumer_test_access::gate(consumer);
  auto filters_producer = filters->register_producer({0});
  REQUIRE(filters_producer.push_filter(
    0, make_in_list_prefix(10, stream)));  // covers [0..9] — keeps 100%
  REQUIRE(gate_test_access::applicable(gate, filters->snapshot()));

  std::shared_ptr<cudf::table> table = make_int64_sequence_table(10, stream);
  auto out                           = apply_consumer(consumer, table, stream);
  stream.sync();

  REQUIRE(out != nullptr);         // a mask was computed (keeps everything)
  REQUIRE(out->num_rows() == 10);  // ... so kept ratio is 1.0
  REQUIRE_FALSE(gate_test_access::applicable(
    gate, filters->snapshot()));  // gate disabled for subsequent splits

  auto second = apply_consumer(consumer, table, stream);
  REQUIRE(second == nullptr);  // gated out — no work
}

TEST_CASE("dynamic_filter_gate ignores a device with no local replica",
          "[dynamic_filter][scan_merge]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();

  auto filters = std::make_shared<sirius_dynamic_filter_set>();
  dynamic_filter_consumer consumer(
    filters, {.mode = dynamic_filter_apply_mode::INCLUDE_AST_ROW_MASKS}, {});
  auto& gate            = consumer_test_access::gate(consumer);
  auto filters_producer = filters->register_producer({0});
  REQUIRE(filters_producer.push_filter(0, make_in_list_prefix(2, stream)));
  std::shared_ptr<cudf::table> table = make_int64_sequence_table(10, stream);

  // The filter has only its current-device source replica. A consumer device with no local copy
  // must skip without recording a synthetic 100% keep ratio in the scan-global gate.
  auto unavailable = apply_consumer(consumer, table, stream, std::nullopt, /*device_id=*/12345);
  REQUIRE(unavailable == nullptr);
  REQUIRE(gate_test_access::applicable(gate, filters->snapshot()));

  auto local = apply_consumer(consumer, table, stream);
  stream.sync();
  REQUIRE(local != nullptr);
  REQUIRE(local->num_rows() == 2);
  REQUIRE(gate_test_access::applicable(gate, filters->snapshot()));
}

TEST_CASE("dynamic_filter_gate re-arms when a filter publishes after the disable decision",
          "[dynamic_filter][scan_merge]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();

  // Unselective filter publishes first and disables the gate (the Q8 supplier hazard).
  auto filters = std::make_shared<sirius_dynamic_filter_set>();
  dynamic_filter_consumer consumer(
    filters, {.mode = dynamic_filter_apply_mode::INCLUDE_AST_ROW_MASKS}, {});
  auto& gate            = consumer_test_access::gate(consumer);
  auto filters_producer = filters->register_producer({0});
  REQUIRE(filters_producer.push_filter(0, make_in_list_prefix(10, stream)));
  std::shared_ptr<cudf::table> table = make_int64_sequence_table(10, stream);
  (void)apply_consumer(consumer, table, stream);
  stream.sync();
  REQUIRE_FALSE(gate_test_access::applicable(gate, filters->snapshot()));

  // A selective filter lands later: the channel grew, so the gate must re-arm...
  REQUIRE(filters_producer.push_filter(0, make_in_list_prefix(2, stream)));
  REQUIRE(gate_test_access::applicable(gate, filters->snapshot()));

  // ...and the re-measurement sees the combined mask (AND → keeps 0,1), going ACTIVE.
  auto out = apply_consumer(consumer, table, stream);
  stream.sync();
  REQUIRE(out != nullptr);
  REQUIRE(out->num_rows() == 2);
  REQUIRE(gate_test_access::applicable(gate, filters->snapshot()));  // active — stays applicable
}

TEST_CASE("dynamic_filter_gate stays active once a selective split proves the filter useful",
          "[dynamic_filter][scan_merge]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();

  auto filters = std::make_shared<sirius_dynamic_filter_set>();
  dynamic_filter_consumer consumer(
    filters, {.mode = dynamic_filter_apply_mode::INCLUDE_AST_ROW_MASKS}, {});
  auto& gate            = consumer_test_access::gate(consumer);
  auto filters_producer = filters->register_producer({0});
  REQUIRE(filters_producer.push_filter(0, make_in_list_prefix(2, stream)));  // selective: keeps 20%
  std::shared_ptr<cudf::table> table = make_int64_sequence_table(10, stream);
  (void)apply_consumer(consumer, table, stream);
  stream.sync();
  REQUIRE(gate_test_access::applicable(gate, filters->snapshot()));

  // A later unselective publish must not demote an active gate.
  REQUIRE(filters_producer.push_filter(0, make_in_list_prefix(10, stream)));
  auto out = apply_consumer(consumer, table, stream);
  stream.sync();
  REQUIRE(out != nullptr);
  REQUIRE(out->num_rows() == 2);  // cascade of both filters == their conjunction
  REQUIRE(gate_test_access::applicable(gate, filters->snapshot()));  // still active
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
    // current-generation unselective result seals DISABLED and makes the gate inapplicable. With
    // the serialized transition, ACTIVE is terminal and this call is a no-op.
    gate->record_keep_ratio(/*rows_before=*/100,
                            /*rows_after=*/100,
                            /*observed_filter_count=*/2);
    all_stayed_active =
      all_stayed_active && gate_test_access::applicable(*gate, filters.snapshot());
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

  auto filters = std::make_shared<sirius_dynamic_filter_set>();
  dynamic_filter_consumer consumer(
    filters, {.mode = dynamic_filter_apply_mode::INCLUDE_AST_ROW_MASKS}, {});
  auto filters_producer = filters->register_producer({0});
  REQUIRE(filters_producer.push_filter(0, make_in_list_prefix(7, stream)));  // keeps 0..6
  REQUIRE(filters_producer.push_filter(0, make_in_list_prefix(4, stream)));  // keeps 0..3

  std::shared_ptr<cudf::table> table = make_int64_sequence_table(10, stream);
  auto out                           = apply_consumer(consumer, table, stream);
  stream.sync();
  REQUIRE(out != nullptr);
  REQUIRE(out->num_rows() == 4);  // 0..3 — intersection regardless of cascade order
}

TEST_CASE("per-filter gate measures marginal keep and skips a useless filter on later splits",
          "[dynamic_filter][scan_merge]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();

  auto useless   = make_in_list_prefix(10, stream);  // covers the whole domain — keep 1.0
  auto selective = make_in_list_prefix(2, stream);   // keeps 20%
  auto filters   = std::make_shared<sirius_dynamic_filter_set>();
  dynamic_filter_consumer consumer(
    filters, {.mode = dynamic_filter_apply_mode::INCLUDE_AST_ROW_MASKS}, {});
  auto& gate            = consumer_test_access::gate(consumer);
  auto filters_producer = filters->register_producer({0});
  REQUIRE(filters_producer.push_filter(0, useless));
  REQUIRE(filters_producer.push_filter(0, selective));

  std::shared_ptr<cudf::table> table = make_int64_sequence_table(10, stream);
  auto out                           = apply_consumer(consumer, table, stream);
  stream.sync();
  REQUIRE(out != nullptr);
  REQUIRE(out->num_rows() == 2);

  // First split measured both marginals: the domain-covering filter is now skippable, the
  // selective one is not.
  auto useless_kept =
    gate_test_access::filter_keep_ratio(gate, {useless.get(), 0}, filters->filter_count());
  REQUIRE(useless_kept.has_value());
  REQUIRE(sirius::op::scan::dynamic_filter_gate::filter_skippable(*useless_kept));
  auto selective_kept =
    gate_test_access::filter_keep_ratio(gate, {selective.get(), 0}, filters->filter_count());
  REQUIRE(selective_kept.has_value());
  REQUIRE_FALSE(sirius::op::scan::dynamic_filter_gate::filter_skippable(*selective_kept));

  // Later splits still produce the right rows with the useless filter dropped from the cascade.
  auto out2 = apply_consumer(consumer, table, stream);
  stream.sync();
  REQUIRE(out2 != nullptr);
  REQUIRE(out2->num_rows() == 2);
}

TEST_CASE("per-filter gate keeps a dead verdict when the channel grows",
          "[dynamic_filter][scan_merge]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();

  auto useless = make_in_list_prefix(10, stream);  // covers the whole domain -- keep 1.0
  auto filters = std::make_shared<sirius_dynamic_filter_set>();
  dynamic_filter_consumer consumer(
    filters, {.mode = dynamic_filter_apply_mode::INCLUDE_AST_ROW_MASKS}, {});
  auto& gate            = consumer_test_access::gate(consumer);
  auto filters_producer = filters->register_producer({0});
  REQUIRE(filters_producer.push_filter(0, useless));

  std::shared_ptr<cudf::table> table = make_int64_sequence_table(10, stream);
  auto out                           = apply_consumer(consumer, table, stream);
  stream.sync();
  REQUIRE(out != nullptr);

  auto const measured =
    gate_test_access::filter_keep_ratio(gate, {useless.get(), 0}, filters->filter_count());
  REQUIRE(measured.has_value());
  REQUIRE(sirius::op::scan::dynamic_filter_gate::filter_skippable(*measured));

  // Growth re-opens the scan-level gate, not this verdict: the dead filter stays dead.
  REQUIRE(filters_producer.push_filter(0, make_in_list_prefix(2, stream)));
  auto const after_growth =
    gate_test_access::filter_keep_ratio(gate, {useless.get(), 0}, filters->filter_count());
  REQUIRE(after_growth.has_value());
  REQUIRE(sirius::op::scan::dynamic_filter_gate::filter_skippable(*after_growth));
  REQUIRE(after_growth == measured);  // the stored verdict, not a remeasure trigger
}

TEST_CASE("per-filter gate excludes a dead filter from the re-armed apply without re-running it",
          "[dynamic_filter][scan_merge]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();

  // The only filter covers the whole domain: the first split disables the scan-level gate and
  // records the filter's dead marginal verdict.
  auto useless = std::make_shared<counting_in_list_filter>(make_in_list_prefix(10, stream));
  auto filters = std::make_shared<sirius_dynamic_filter_set>();
  dynamic_filter_consumer consumer(
    filters, {.mode = dynamic_filter_apply_mode::INCLUDE_AST_ROW_MASKS}, {});
  auto& gate            = consumer_test_access::gate(consumer);
  auto filters_producer = filters->register_producer({0});
  REQUIRE(filters_producer.push_filter(0, useless));

  std::shared_ptr<cudf::table> table = make_int64_sequence_table(10, stream);
  auto out                           = apply_consumer(consumer, table, stream);
  stream.sync();
  REQUIRE(out != nullptr);
  REQUIRE(out->num_rows() == 10);
  REQUIRE(useless->mask_calls() == 1);
  REQUIRE_FALSE(gate_test_access::applicable(gate, filters->snapshot()));

  // Growth re-arms the scan-level gate; the dead verdict is permanent, so the re-armed apply
  // runs only the newcomer.
  auto selective = make_in_list_prefix(2, stream);
  REQUIRE(filters_producer.push_filter(0, selective));
  REQUIRE(gate_test_access::applicable(gate, filters->snapshot()));

  auto out2 = apply_consumer(consumer, table, stream);
  stream.sync();
  REQUIRE(out2 != nullptr);
  REQUIRE(out2->num_rows() == 2);
  REQUIRE(useless->mask_calls() == 1);

  auto const useless_kept =
    gate_test_access::filter_keep_ratio(gate, {useless.get(), 0}, filters->filter_count());
  REQUIRE(useless_kept.has_value());
  REQUIRE(sirius::op::scan::dynamic_filter_gate::filter_skippable(*useless_kept));
  auto const selective_kept =
    gate_test_access::filter_keep_ratio(gate, {selective.get(), 0}, filters->filter_count());
  REQUIRE(selective_kept.has_value());
  REQUIRE_FALSE(sirius::op::scan::dynamic_filter_gate::filter_skippable(*selective_kept));
  REQUIRE(
    gate_test_access::applicable(gate, filters->snapshot()));  // the re-arm measured 0.2 -> ACTIVE
}

TEST_CASE("per-filter gate stales a selective verdict when the channel grows",
          "[dynamic_filter][scan_merge]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();

  auto selective = make_in_list_prefix(2, stream);  // keeps 20%
  auto filters   = std::make_shared<sirius_dynamic_filter_set>();
  dynamic_filter_consumer consumer(
    filters, {.mode = dynamic_filter_apply_mode::INCLUDE_AST_ROW_MASKS}, {});
  auto& gate            = consumer_test_access::gate(consumer);
  auto filters_producer = filters->register_producer({0});
  REQUIRE(filters_producer.push_filter(0, selective));

  std::shared_ptr<cudf::table> table = make_int64_sequence_table(10, stream);
  auto out                           = apply_consumer(consumer, table, stream);
  stream.sync();
  REQUIRE(out != nullptr);
  REQUIRE(out->num_rows() == 2);

  auto const measured =
    gate_test_access::filter_keep_ratio(gate, {selective.get(), 0}, filters->filter_count());
  REQUIRE(measured.has_value());
  REQUIRE_FALSE(sirius::op::scan::dynamic_filter_gate::filter_skippable(*measured));

  // Growth stales a selective reading: new arrivals change the rows reaching the filter.
  REQUIRE(filters_producer.push_filter(0, make_in_list_prefix(4, stream)));
  REQUIRE_FALSE(
    gate_test_access::filter_keep_ratio(gate, {selective.get(), 0}, filters->filter_count())
      .has_value());

  // The next apply remeasures it against the larger cascade.
  auto out2 = apply_consumer(consumer, table, stream);
  stream.sync();
  REQUIRE(out2 != nullptr);
  REQUIRE(out2->num_rows() == 2);
  REQUIRE(gate_test_access::filter_keep_ratio(gate, {selective.get(), 0}, filters->filter_count())
            .has_value());
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
  ::cuda::stream_ref stream          = cudf::get_default_stream();
  std::shared_ptr<cudf::table> table = make_int64_sequence_table(10, stream);
  auto filters                       = std::make_shared<sirius_dynamic_filter_set>();
  dynamic_filter_consumer consumer(
    filters, {.mode = dynamic_filter_apply_mode::MEMBERSHIP_MASKS_ONLY}, {});
  auto producer = filters->register_producer({0}, 8);
  for (int64_t count = 10; count >= 3; --count) {
    REQUIRE(producer.push_filter(0, make_in_list_prefix(count, stream)));
  }

  auto out = apply_consumer(consumer, table, stream, /*input_bytes=*/800);
  stream.sync();

  REQUIRE(out != nullptr);
  REQUIRE(out->num_columns() == 1);
  REQUIRE(to_host_int64(out->view().column(0), stream) == std::vector<int64_t>{0, 1, 2});
}

TEST_CASE("compaction strategies equal cascade for AST and INT32 membership",
          "[dynamic_filter][scan_merge][deferred_keys][equivalence]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();
  std::shared_ptr<cudf::table> table =
    make_sequence_table<int32_t>(10, stream, {{0, 1}, {0, 1}, {100, 1}});
  auto filters  = std::make_shared<sirius_dynamic_filter_set>();
  auto producer = filters->register_producer({0, 1});
  REQUIRE(producer.push_filter(0, make_zone_map(2, 8)));
  REQUIRE(producer.push_filter(1, make_in_list_prefix<int32_t>(5, stream)));

  REQUIRE(forced_strategies_equal(
    table, filters, stream, dynamic_filter_apply_mode::INCLUDE_AST_ROW_MASKS));
}

TEST_CASE("forced compaction strategies agree for an AST-only mask",
          "[dynamic_filter][scan_merge][equivalence]")
{
  ::cuda::stream_ref stream          = cudf::get_default_stream();
  std::shared_ptr<cudf::table> table = make_sequence_table(10, stream);
  auto filters                       = std::make_shared<sirius_dynamic_filter_set>();
  auto producer                      = filters->register_producer({0});
  REQUIRE(producer.push_filter(0, make_zone_map(3, 7)));

  REQUIRE(forced_strategies_equal(
    table, filters, stream, dynamic_filter_apply_mode::INCLUDE_AST_ROW_MASKS));
}

TEST_CASE("compaction strategies equal cascade for sliced nested nullable payloads",
          "[dynamic_filter][scan_merge][deferred_keys][equivalence]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(make_sequence_column<int64_t>(10, stream));
  columns.push_back(make_sequence_column<int64_t>(10, stream));
  columns.push_back(make_nullable_list_payload(10, stream));
  std::shared_ptr<cudf::table> table = std::make_unique<cudf::table>(std::move(columns));
  auto sliced_input                  = cudf::slice(table->view(), {1, 9}, stream).front();

  auto filters  = std::make_shared<sirius_dynamic_filter_set>();
  auto producer = filters->register_producer({0, 1});
  REQUIRE(producer.push_filter(0, make_in_list_prefix(8, stream)));
  REQUIRE(producer.push_filter(1, make_in_list_prefix(5, stream)));

  REQUIRE(forced_strategies_equal(
    table, filters, stream, dynamic_filter_apply_mode::MEMBERSHIP_MASKS_ONLY, sliced_input));
}

TEST_CASE("compaction strategies equal cascade for all-true and empty results",
          "[dynamic_filter][scan_merge][deferred_keys][equivalence]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();

  SECTION("all contributing masks retain every row")
  {
    std::shared_ptr<cudf::table> table =
      make_sequence_table<int64_t>(10, stream, {{0, 1}, {0, 1}, {100, 1}});
    auto filters  = std::make_shared<sirius_dynamic_filter_set>();
    auto producer = filters->register_producer({0, 1});
    REQUIRE(producer.push_filter(0, make_in_list_prefix(10, stream)));
    REQUIRE(producer.push_filter(1, make_in_list_prefix(10, stream)));
    REQUIRE(forced_strategies_equal(
      table, filters, stream, dynamic_filter_apply_mode::MEMBERSHIP_MASKS_ONLY));
  }

  SECTION("contributing masks produce an empty table with the original schema")
  {
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(make_sequence_column<int64_t>(10, stream));
    columns.push_back(make_sequence_column<int64_t>(10, stream, 9, -1));
    columns.push_back(make_nullable_list_payload(10, stream));
    std::shared_ptr<cudf::table> table = std::make_unique<cudf::table>(std::move(columns));
    auto filters                       = std::make_shared<sirius_dynamic_filter_set>();
    auto producer                      = filters->register_producer({0, 1});
    REQUIRE(producer.push_filter(0, make_in_list_prefix(3, stream)));
    REQUIRE(producer.push_filter(1, make_in_list_prefix(3, stream)));
    REQUIRE(forced_strategies_equal(
      table, filters, stream, dynamic_filter_apply_mode::MEMBERSHIP_MASKS_ONLY));
  }
}

TEST_CASE("compaction strategies equal cascade for Bloom membership filters",
          "[dynamic_filter][scan_merge][deferred_keys][equivalence]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();
  auto build_five           = make_sequence_column<int64_t>(5, stream);
  auto build_three          = make_sequence_column<int64_t>(3, stream);
  auto filters              = std::make_shared<sirius_dynamic_filter_set>();
  auto producer             = filters->register_producer({0});
  REQUIRE(
    producer.push_filter(0,
                         std::make_shared<sirius::op::sirius_dynamic_bloom_filter>(
                           build_five->view(), stream, cudf::get_current_device_resource_ref())));
  REQUIRE(
    producer.push_filter(0,
                         std::make_shared<sirius::op::sirius_dynamic_bloom_filter>(
                           build_three->view(), stream, cudf::get_current_device_resource_ref())));
  std::shared_ptr<cudf::table> input = make_int64_sequence_table(20, stream);

  REQUIRE(forced_strategies_equal(
    input, filters, stream, dynamic_filter_apply_mode::MEMBERSHIP_MASKS_ONLY));
}

TEST_CASE("deferred keys preserve null pass-through when every mask declines",
          "[dynamic_filter][scan_merge][deferred_keys][equivalence]")
{
  ::cuda::stream_ref stream          = cudf::get_default_stream();
  std::shared_ptr<cudf::table> input = make_int64_sequence_table(10, stream);
  auto filters                       = std::make_shared<sirius_dynamic_filter_set>();
  auto producer                      = filters->register_producer({0});
  REQUIRE(producer.push_filter(0, std::make_shared<nullable_declining_filter>()));
  REQUIRE(producer.push_filter(0, std::make_shared<nullable_declining_filter>()));

  REQUIRE(forced_strategies_equal(
    input, filters, stream, dynamic_filter_apply_mode::MEMBERSHIP_MASKS_ONLY));
}

TEST_CASE("gather once declines null masks without training their marginals",
          "[dynamic_filter][scan_merge][gather_once][gate]")
{
  ::cuda::stream_ref stream          = cudf::get_default_stream();
  std::shared_ptr<cudf::table> input = make_int64_sequence_table(10, stream);
  auto declining                     = std::make_shared<nullable_declining_filter>();
  auto filters                       = std::make_shared<sirius_dynamic_filter_set>();
  dynamic_filter_consumer consumer(
    filters,
    {.mode                        = dynamic_filter_apply_mode::MEMBERSHIP_MASKS_ONLY,
     .strategy_override           = sirius::op::scan::detail::compaction_strategy::GATHER_ONCE,
     .validate_deferred_selection = true},
    {});
  auto& gate    = consumer_test_access::gate(consumer);
  auto producer = filters->register_producer({0});
  REQUIRE(producer.push_filter(0, declining));

  auto output = apply_consumer(consumer, input, stream);

  REQUIRE(output == nullptr);
  REQUIRE(declining->mask_calls() == 1);
  REQUIRE_FALSE(
    gate_test_access::filter_keep_ratio(gate, {declining.get(), 0}, filters->filter_count())
      .has_value());
}

TEST_CASE("production policy uses gather once after weak marginals become current",
          "[dynamic_filter][scan_merge][gather_once][gate][production_policy]")
{
  // 32-byte narrow rows and 80-byte wide rows both reach gather-once once every marginal is
  // current, so the two width branches of the policy are exercised against the same filters.
  auto const input_bytes             = GENERATE(std::size_t{640}, std::size_t{1600});
  ::cuda::stream_ref stream          = cudf::get_default_stream();
  std::shared_ptr<cudf::table> input = make_sequence_table<int64_t>(20, stream, {{0, 1}, {100, 1}});

  // The marginal keeps these two filters train are 9/20 and 4/9. Both stay at or below the gate's
  // skippable cutoff, so neither filter is dropped before strategy selection, and both stay above
  // the wide and narrow gather-once cutoffs.
  auto first   = std::make_shared<counting_in_list_filter>(make_in_list_prefix(9, stream));
  auto second  = std::make_shared<counting_in_list_filter>(make_in_list_prefix(4, stream));
  auto filters = std::make_shared<sirius_dynamic_filter_set>();
  dynamic_filter_consumer consumer(
    filters, {.mode = dynamic_filter_apply_mode::MEMBERSHIP_MASKS_ONLY}, {});
  auto& gate    = consumer_test_access::gate(consumer);
  auto producer = filters->register_producer({0}, 3);
  REQUIRE(producer.push_filter(0, first));
  REQUIRE(producer.push_filter(0, second));

  auto const apply = [&] {
    auto out = apply_consumer(consumer, input, stream, input_bytes);
    stream.sync();
    return out;
  };

  auto trained = apply();
  REQUIRE(trained != nullptr);
  REQUIRE(to_host_int64(trained->view().column(0), stream) == std::vector<int64_t>{0, 1, 2, 3});
  auto const first_keep =
    gate_test_access::filter_keep_ratio(gate, {first.get(), 0}, filters->filter_count());
  auto const second_keep =
    gate_test_access::filter_keep_ratio(gate, {second.get(), 0}, filters->filter_count());
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
  REQUIRE(gate_test_access::filter_keep_ratio(gate, {first.get(), 0}, filters->filter_count()) ==
          first_keep);
  REQUIRE(gate_test_access::filter_keep_ratio(gate, {second.get(), 0}, filters->filter_count()) ==
          second_keep);

  // Channel growth stales both measurements, so gather-once becomes ineligible and the next apply
  // has to retrain in survivor space before it can be chosen again.
  REQUIRE(producer.push_filter(0, make_in_list_prefix(3, stream)));
  REQUIRE_FALSE(gate_test_access::filter_keep_ratio(gate, {first.get(), 0}, filters->filter_count())
                  .has_value());
  REQUIRE_FALSE(
    gate_test_access::filter_keep_ratio(gate, {second.get(), 0}, filters->filter_count())
      .has_value());

  auto retrained = apply();

  REQUIRE(retrained != nullptr);
  REQUIRE(to_host_int64(retrained->view().column(0), stream) == std::vector<int64_t>{0, 1, 2});
  REQUIRE(second->probe_rows() == std::vector<cudf::size_type>{9, 20, 9});
  REQUIRE(gate_test_access::filter_keep_ratio(gate, {second.get(), 0}, filters->filter_count())
            .has_value());
}

TEST_CASE("deferred keys preserve alignment across repeated and different columns",
          "[dynamic_filter][scan_merge][deferred_keys]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();
  std::shared_ptr<cudf::table> table =
    make_sequence_table<int64_t>(10, stream, {{0, 1}, {9, -1}, {100, 1}, {200, 1}});

  auto filters = std::make_shared<sirius_dynamic_filter_set>();
  dynamic_filter_consumer consumer(
    filters, {.mode = dynamic_filter_apply_mode::MEMBERSHIP_MASKS_ONLY}, {});
  auto producer = filters->register_producer({0, 1});
  REQUIRE(producer.push_filter(0, make_in_list_prefix(8, stream)));
  REQUIRE(producer.push_filter(1, make_in_list_prefix(6, stream)));
  REQUIRE(producer.push_filter(1, make_in_list_prefix(4, stream)));

  auto out = apply_consumer(consumer, table, stream, /*input_bytes=*/800);
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
  std::shared_ptr<cudf::table> table =
    make_sequence_table<int64_t>(10, stream, {{0, 1}, {0, 1}, {100, 1}});
  auto& nullable_key = table->get_column(0);
  nullable_key.set_null_mask(cudf::create_null_mask(10, cudf::mask_state::ALL_VALID, stream), 1);
  cudf::set_null_mask(nullable_key.mutable_view().null_mask(), 0, 1, false, stream);

  auto declining = std::make_shared<nullable_declining_filter>();
  auto selective = make_in_list_prefix(3, stream);
  auto filters   = std::make_shared<sirius_dynamic_filter_set>();
  dynamic_filter_consumer consumer(
    filters, {.mode = dynamic_filter_apply_mode::MEMBERSHIP_MASKS_ONLY}, {});
  auto& gate    = consumer_test_access::gate(consumer);
  auto producer = filters->register_producer({0, 1});
  REQUIRE(producer.push_filter(0, declining));
  REQUIRE(producer.push_filter(1, selective));

  auto out = apply_consumer(consumer, table, stream, /*input_bytes=*/800);
  stream.sync();

  REQUIRE(out != nullptr);
  REQUIRE(declining->mask_calls() == 1);
  REQUIRE(declining->saw_nulls());
  REQUIRE_FALSE(
    gate_test_access::filter_keep_ratio(gate, {declining.get(), 0}, filters->filter_count())
      .has_value());
  REQUIRE(gate_test_access::filter_keep_ratio(
            gate, {selective.get(), 1}, filters->filter_count()) == Approx(0.3));
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
  ::cuda::stream_ref stream          = cudf::get_default_stream();
  std::shared_ptr<cudf::table> input = make_sequence_table<int64_t>(20, stream, {{0, 1}, {100, 1}});

  // Marginal keeps of 11/20 and 6/11: both above the gate's 50% cutoff, both at or below 60%.
  auto first   = std::make_shared<counting_in_list_filter>(make_in_list_prefix(11, stream));
  auto second  = std::make_shared<counting_in_list_filter>(make_in_list_prefix(6, stream));
  auto filters = std::make_shared<sirius_dynamic_filter_set>();
  dynamic_filter_consumer consumer(
    filters, {.mode = dynamic_filter_apply_mode::MEMBERSHIP_MASKS_ONLY}, {});
  auto& gate    = consumer_test_access::gate(consumer);
  auto producer = filters->register_producer({0});
  REQUIRE(producer.push_filter(0, first));
  REQUIRE(producer.push_filter(0, second));

  auto const apply = [&] {
    auto out = apply_consumer(consumer, input, stream, input_bytes);
    stream.sync();
    return out;
  };

  auto trained = apply();
  REQUIRE(trained != nullptr);
  REQUIRE(trained->num_rows() == 6);
  auto const first_keep =
    gate_test_access::filter_keep_ratio(gate, {first.get(), 0}, filters->filter_count());
  auto const second_keep =
    gate_test_access::filter_keep_ratio(gate, {second.get(), 0}, filters->filter_count());
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
  ::cuda::stream_ref stream          = cudf::get_default_stream();
  std::shared_ptr<cudf::table> input = make_int64_sequence_table(20, stream);
  auto filter  = std::make_shared<counting_in_list_filter>(make_in_list_prefix(prefix, stream));
  auto filters = std::make_shared<sirius_dynamic_filter_set>();
  dynamic_filter_consumer consumer(
    filters, {.mode = dynamic_filter_apply_mode::MEMBERSHIP_MASKS_ONLY}, {});
  auto& gate    = consumer_test_access::gate(consumer);
  auto producer = filters->register_producer({0});
  REQUIRE(producer.push_filter(0, filter));
  constexpr std::size_t wide_bytes = 1600;

  auto const apply = [&] {
    auto out = apply_consumer(consumer, input, stream, wide_bytes);
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

namespace {
std::vector<cudf::size_type> counts_of(application_result const& result)
{
  return {result.prefix_counts().begin(), result.prefix_counts().end()};
}

application_result run_owned(dynamic_filter_consumer& consumer,
                             std::shared_ptr<cudf::table const> input,
                             std::span<binding const> bindings,
                             ::cuda::stream_ref stream,
                             batch_receipt const* prior = nullptr)
{
  auto program = consumer.prepare({input->view(), bindings, std::nullopt, prior});
  REQUIRE(program.has_value());
  return consumer_test_access::apply(consumer,
                                     std::move(*program),
                                     std::move(input),
                                     stream,
                                     cudf::get_current_device_resource_ref());
}
}  // namespace

TEST_CASE("consumer gather counts measure correlated ordered survivors",
          "[dynamic_filter][scan_merge][consumer_counts]")
{
  auto const strategy       = GENERATE(compaction_strategy::CASCADE,
                                 compaction_strategy::DEFERRED_KEYS,
                                 compaction_strategy::GATHER_ONCE);
  ::cuda::stream_ref stream = cudf::get_default_stream();
  std::shared_ptr<cudf::table const> input =
    make_sequence_table<int64_t>(100, stream, {{0, 1}, {1000, 1}});
  auto channel  = std::make_shared<sirius_dynamic_filter_set>();
  auto producer = channel->register_producer({0});
  auto first    = make_in_list_prefix(40, stream);
  std::vector<int64_t> keys(60);
  std::iota(keys.begin(), keys.begin() + 10, 0);
  std::iota(keys.begin() + 10, keys.end(), 40);
  auto build  = make_values_table(keys, cudf::data_type{cudf::type_id::INT64}, stream);
  auto second = std::make_shared<sirius::op::sirius_dynamic_in_list_filter>(
    build->view().column(0), stream, cudf::get_current_device_resource_ref());
  REQUIRE(producer.push_filter(0, first));
  REQUIRE(producer.push_filter(0, second));
  dynamic_filter_consumer consumer(
    channel, {.strategy_override = strategy, .force_counts = true}, {});
  std::vector<binding> bindings{{0, 0}};
  auto result = run_owned(consumer, input, bindings, stream);
  REQUIRE(counts_of(result) == std::vector<cudf::size_type>{100, 40, 10});
  REQUIRE(result.counts_read_back() == (strategy == compaction_strategy::GATHER_ONCE));
  auto output   = result.take_output();
  auto expected = make_sequence_table<int64_t>(10, stream, {{0, 1}, {1000, 1}});
  REQUIRE(sirius::test::expect_tables_equivalent_impl(output->view(), expected->view(), stream));
  auto& gate = consumer_test_access::gate(consumer);
  REQUIRE_FALSE(gate_test_access::filter_keep_ratio(gate, {second.get(), 0}, 2));
  auto receipt = consumer.finish(std::move(result), 71);
  REQUIRE(receipt.original_batch_id == 71);
  REQUIRE(receipt.applied.size() == 2);
  REQUIRE(gate_test_access::filter_keep_ratio(gate, {first.get(), 0}, 2) == Approx(0.4));
  REQUIRE(gate_test_access::filter_keep_ratio(gate, {second.get(), 0}, 2) == Approx(0.25));
}

TEST_CASE("consumer counts include AST prefix and dead suffix does not gain coverage",
          "[dynamic_filter][scan_merge][consumer_counts][consumer_receipt]")
{
  auto const strategy                      = GENERATE(compaction_strategy::CASCADE,
                                 compaction_strategy::DEFERRED_KEYS,
                                 compaction_strategy::GATHER_ONCE);
  auto const dead                          = GENERATE(false, true);
  ::cuda::stream_ref stream                = cudf::get_default_stream();
  std::shared_ptr<cudf::table const> input = make_sequence_table<int32_t>(100, stream);
  auto channel                             = std::make_shared<sirius_dynamic_filter_set>();
  auto producer                            = channel->register_producer({0}, 3);
  REQUIRE(producer.push_filter(0, make_zone_map(dead ? 200 : 0, dead ? 250 : 49)));
  auto member = make_in_list_prefix<int32_t>(10, stream);
  REQUIRE(producer.push_filter(0, member));
  dynamic_filter_consumer consumer(
    channel, {.strategy_override = strategy, .force_counts = true}, {});
  std::vector<binding> bindings{{0, 0}};
  auto result = run_owned(consumer, input, bindings, stream);
  REQUIRE(counts_of(result) ==
          (dead ? std::vector<cudf::size_type>{0, 0} : std::vector<cudf::size_type>{50, 10}));
  REQUIRE(result.take_output()->num_rows() == (dead ? 0 : 10));
  auto receipt = consumer.finish(std::move(result), 9);
  REQUIRE(receipt.applied.size() == (dead ? 1 : 2));
  auto ratio =
    gate_test_access::filter_keep_ratio(consumer_test_access::gate(consumer), {member.get(), 0}, 2);
  if (dead) {
    REQUIRE_FALSE(ratio);
  } else {
    REQUIRE(ratio == Approx(0.2));
  }
  REQUIRE_FALSE(gate_test_access::needs_combined_sample(consumer_test_access::gate(consumer),
                                                        channel->snapshot()));
}

TEST_CASE("consumer distinguishes decline all keep and zero-prefix suffix",
          "[dynamic_filter][scan_merge][consumer_counts][consumer_receipt]")
{
  auto const strategy                      = GENERATE(compaction_strategy::CASCADE,
                                 compaction_strategy::DEFERRED_KEYS,
                                 compaction_strategy::GATHER_ONCE);
  ::cuda::stream_ref stream                = cudf::get_default_stream();
  std::shared_ptr<cudf::table const> input = make_int64_sequence_table(10, stream);
  auto channel                             = std::make_shared<sirius_dynamic_filter_set>();
  auto producer                            = channel->register_producer({0}, 4);
  auto decline                             = std::make_shared<nullable_declining_filter>();
  auto keep                                = make_in_list_prefix(10, stream);
  auto empty_keys                          = make_sequence_column<int64_t>(1, stream, 50);
  auto reject = std::make_shared<sirius::op::sirius_dynamic_in_list_filter>(
    empty_keys->view(), stream, cudf::get_current_device_resource_ref());
  auto suffix = make_in_list_prefix(5, stream);
  REQUIRE(producer.push_filter(0, decline));
  REQUIRE(producer.push_filter(0, keep));
  REQUIRE(producer.push_filter(0, reject));
  REQUIRE(producer.push_filter(0, suffix));
  dynamic_filter_consumer consumer(
    channel, {.strategy_override = strategy, .force_counts = true}, {});
  std::vector<binding> bindings{{0, 0}};
  auto result = run_owned(consumer, input, bindings, stream);
  REQUIRE(counts_of(result) == std::vector<cudf::size_type>{10, 10, 10, 0, 0});
  REQUIRE(result.take_output()->num_rows() == 0);
  auto receipt = consumer.finish(std::move(result), 1);
  REQUIRE(receipt.applied.size() == 2);
  auto& gate = consumer_test_access::gate(consumer);
  REQUIRE_FALSE(gate_test_access::filter_keep_ratio(gate, {decline.get(), 0}, 4));
  REQUIRE(gate_test_access::filter_keep_ratio(gate, {keep.get(), 0}, 4) == Approx(1.0));
  REQUIRE(gate_test_access::filter_keep_ratio(gate, {reject.get(), 0}, 4) == Approx(0.0));
  REQUIRE_FALSE(gate_test_access::filter_keep_ratio(gate, {suffix.get(), 0}, 4));
}

TEST_CASE("consumer active known gather avoids count readback",
          "[dynamic_filter][scan_merge][consumer_counts]")
{
  auto const empty                         = GENERATE(false, true);
  ::cuda::stream_ref stream                = cudf::get_default_stream();
  std::shared_ptr<cudf::table const> input = make_sequence_table<int64_t>(
    10,
    stream,
    empty ? std::initializer_list<sequence_column_spec<int64_t>>{{100, 1}}
          : std::initializer_list<sequence_column_spec<int64_t>>{{0, 1}});
  auto channel  = std::make_shared<sirius_dynamic_filter_set>();
  auto producer = channel->register_producer({0});
  auto first    = make_in_list_prefix(5, stream);
  auto second   = make_in_list_prefix(3, stream);
  REQUIRE(producer.push_filter(0, first));
  REQUIRE(producer.push_filter(0, second));
  dynamic_filter_consumer consumer(
    channel, {.strategy_override = compaction_strategy::GATHER_ONCE}, {});
  auto& gate = consumer_test_access::gate(consumer);
  gate.record_keep_ratio(10, 3, 2);
  gate.record_filter_keep_ratio({first.get(), 0}, 0.4, 2);
  gate.record_filter_keep_ratio({second.get(), 0}, 0.5, 2);
  std::vector<binding> bindings{{0, 0}};
  auto result = run_owned(consumer, input, bindings, stream);
  REQUIRE_FALSE(result.counts_read_back());
  REQUIRE(result.prefix_counts().empty());
  REQUIRE(result.take_output()->num_rows() == (empty ? 0 : 3));
  auto receipt = consumer.finish(std::move(result), 1);
  REQUIRE(receipt.applied.size() == (empty ? 0 : 2));
}

TEST_CASE("consumer receipt subtracts only completed bindings from a fresh snapshot",
          "[dynamic_filter][scan_merge][consumer_receipt]")
{
  ::cuda::stream_ref stream                = cudf::get_default_stream();
  std::shared_ptr<cudf::table const> input = make_int64_sequence_table(100, stream);
  auto channel                             = std::make_shared<sirius_dynamic_filter_set>();
  auto producer                            = channel->register_producer({0});
  auto first                               = make_in_list_prefix(40, stream);
  REQUIRE(producer.push_filter(0, first));
  dynamic_filter_consumer consumer(channel, {}, {});
  std::vector<binding> bindings{{0, 0}};
  auto frozen = consumer.prepare({input->view(), bindings});
  auto second = make_in_list_prefix(10, stream);
  REQUIRE(producer.push_filter(0, second));
  auto result = consumer_test_access::apply(
    consumer, std::move(*frozen), input, stream, cudf::get_current_device_resource_ref());
  std::shared_ptr<cudf::table const> checkpoint = result.take_output();
  REQUIRE(checkpoint->num_rows() == 40);
  auto receipt = consumer.finish(std::move(result), 11);
  REQUIRE(receipt.applied.size() == 1);
  auto next = run_owned(consumer, checkpoint, bindings, stream, &receipt);
  REQUIRE(counts_of(next) == std::vector<cudf::size_type>{40, 10});
  std::shared_ptr<cudf::table const> final = next.take_output();
  auto completed                           = consumer.finish(std::move(next), 11);
  REQUIRE(completed.applied.size() == 2);
  REQUIRE_FALSE(consumer.prepare({final->view(), bindings, std::nullopt, &completed}));
  auto other         = std::make_shared<sirius_dynamic_filter_set>();
  completed.endpoint = other;
  REQUIRE(consumer.prepare({final->view(), bindings, std::nullopt, &completed}).has_value());
}

TEST_CASE("consumer decode alias history suppresses attached sampling but permits fresh marginal",
          "[dynamic_filter][scan_merge][consumer_receipt][decode_history]")
{
  auto const strategy                      = GENERATE(compaction_strategy::CASCADE,
                                 compaction_strategy::DEFERRED_KEYS,
                                 compaction_strategy::GATHER_ONCE);
  ::cuda::stream_ref stream                = cudf::get_default_stream();
  std::shared_ptr<cudf::table const> input = make_int64_sequence_table(100, stream);
  auto channel                             = std::make_shared<sirius_dynamic_filter_set>();
  auto producer                            = channel->register_producer({0, 1});
  auto attached = std::make_shared<counting_in_list_filter>(make_in_list_prefix(40, stream));
  auto fresh    = make_in_list_prefix(10, stream);
  REQUIRE(producer.push_filter(0, attached));
  REQUIRE(producer.push_filter(1, attached));
  REQUIRE(producer.push_filter(0, fresh));
  dynamic_filter_consumer consumer(
    channel, {.strategy_override = strategy, .force_counts = true}, {});
  std::vector<binding> bindings{{0, 0}, {1, 0}};
  batch_receipt history{channel, 4, {}, {{attached, 1}}};
  auto result = run_owned(consumer, input, bindings, stream, &history);
  REQUIRE(counts_of(result) == std::vector<cudf::size_type>{100, 40, 10});
  REQUIRE(attached->mask_calls() == 1);
  auto receipt = consumer.finish(std::move(result), 4);
  REQUIRE(receipt.applied.size() == 3);
  REQUIRE(receipt.decode_attached == history.decode_attached);
  auto& gate = consumer_test_access::gate(consumer);
  REQUIRE_FALSE(gate_test_access::filter_keep_ratio(gate, {attached.get(), 0}, 3));
  REQUIRE_FALSE(gate_test_access::filter_keep_ratio(gate, {attached.get(), 1}, 3));
  REQUIRE(gate_test_access::filter_keep_ratio(gate, {fresh.get(), 0}, 3) == Approx(0.25));
  REQUIRE(gate_test_access::needs_combined_sample(gate, channel->snapshot()));
  REQUIRE(consumer.select_for_decode(bindings).steps.empty());
}

TEST_CASE("consumer with decode bindings filters but does not train on input of unknown history",
          "[dynamic_filter][consumer_integration][decode_history]")
{
  ::cuda::stream_ref stream                = cudf::get_default_stream();
  std::shared_ptr<cudf::table const> input = make_int64_sequence_table(100, stream);
  auto channel                             = std::make_shared<sirius_dynamic_filter_set>();
  auto producer                            = channel->register_producer({0});
  auto filter                              = make_in_list_prefix(40, stream);
  REQUIRE(producer.push_filter(0, filter));
  std::vector<binding> bindings{{0, 0}};

  SECTION("without decode bindings a missing receipt means empty history")
  {
    dynamic_filter_consumer consumer(channel, {}, {});
    auto& gate  = consumer_test_access::gate(consumer);
    auto result = run_owned(consumer, input, bindings, stream);
    REQUIRE(result.take_output()->num_rows() == 40);
    (void)consumer.finish(std::move(result), 1);
    REQUIRE(gate_test_access::filter_keep_ratio(gate, {filter.get(), 0}, 1) == Approx(0.4));
    REQUIRE_FALSE(gate_test_access::needs_combined_sample(gate, channel->snapshot()));
  }
  SECTION("with decode bindings a missing or foreign receipt means unknown history")
  {
    dynamic_filter_consumer consumer(channel, {}, bindings);
    auto& gate = consumer_test_access::gate(consumer);
    batch_receipt foreign{std::make_shared<sirius_dynamic_filter_set>(), 1, {}, {}};
    for (auto const* prior : std::array<batch_receipt const*, 2>{nullptr, &foreign}) {
      auto result = run_owned(consumer, input, bindings, stream, prior);
      REQUIRE(result.take_output()->num_rows() == 40);
      (void)consumer.finish(std::move(result), 1);
      REQUIRE_FALSE(gate_test_access::filter_keep_ratio(gate, {filter.get(), 0}, 1).has_value());
      REQUIRE(gate_test_access::needs_combined_sample(gate, channel->snapshot()));
    }
    // A receipt from this endpoint proves the history, here empty, so the same rows train.
    batch_receipt clean{channel, 1, {}, {}};
    auto result = run_owned(consumer, input, bindings, stream, &clean);
    (void)consumer.finish(std::move(result), 1);
    REQUIRE(gate_test_access::filter_keep_ratio(gate, {filter.get(), 0}, 1) == Approx(0.4));
    REQUIRE_FALSE(gate_test_access::needs_combined_sample(gate, channel->snapshot()));
  }
}

TEST_CASE("consumer isolates binding ratios and subtracts proven aliases only",
          "[dynamic_filter][scan_merge][consumer_receipt][gate]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();
  std::shared_ptr<cudf::table const> input =
    make_sequence_table<int64_t>(10, stream, {{0, 1}, {9, -1}});
  auto channel  = std::make_shared<sirius_dynamic_filter_set>();
  auto producer = channel->register_producer({0, 1});
  auto filter   = make_in_list_prefix(4, stream);
  REQUIRE(producer.push_filter(0, filter));
  REQUIRE(producer.push_filter(1, filter));
  dynamic_filter_consumer consumer(channel, {}, {});
  std::vector<binding> distinct{{0, 0}, {1, 1}};
  auto result = run_owned(consumer, input, distinct, stream);
  REQUIRE(counts_of(result) == std::vector<cudf::size_type>{10, 4, 0});
  auto receipt             = consumer.finish(std::move(result), 1);
  auto& gate               = consumer_test_access::gate(consumer);
  auto const first_ordinal = channel->snapshot().entries().front().column_index;
  REQUIRE(gate_test_access::filter_keep_ratio(gate, {filter.get(), first_ordinal}, 2) ==
          Approx(0.4));
  REQUIRE(gate_test_access::filter_keep_ratio(gate, {filter.get(), 1 - first_ordinal}, 2) ==
          Approx(0.0));
  receipt.applied.resize(1);
  REQUIRE(consumer.prepare({input->view(), distinct, std::nullopt, &receipt}));
  std::vector<binding> aliases{{0, 0}, {1, 0}};
  REQUIRE_FALSE(consumer.prepare({input->view(), aliases, std::nullopt, &receipt}));
}

TEST_CASE("consumer destruction does not commit and clean decode opportunity persists",
          "[dynamic_filter][scan_merge][consumer_commit]")
{
  ::cuda::stream_ref stream                = cudf::get_default_stream();
  std::shared_ptr<cudf::table const> input = make_int64_sequence_table(10, stream);
  auto channel                             = std::make_shared<sirius_dynamic_filter_set>();
  auto producer                            = channel->register_producer({0});
  auto filter                              = make_in_list_prefix(3, stream);
  REQUIRE(producer.push_filter(0, filter));
  dynamic_filter_consumer consumer(channel, {}, {});
  std::vector<binding> bindings{{0, 0}};
  {
    auto result = run_owned(consumer, input, bindings, stream);
    REQUIRE(result.take_output()->num_rows() == 3);
  }
  auto& gate = consumer_test_access::gate(consumer);
  REQUIRE(gate_test_access::needs_combined_sample(gate, channel->snapshot()));
  REQUIRE_FALSE(gate_test_access::filter_keep_ratio(gate, {filter.get(), 0}, 1));
  auto clean = consumer.select_for_decode(bindings);
  REQUIRE(clean.generation == 1);
  REQUIRE(clean.steps.empty());
  auto result = run_owned(consumer, input, bindings, stream);
  (void)consumer.finish(std::move(result), 1);
  REQUIRE(consumer.select_for_decode(bindings).steps.size() == 1);
  REQUIRE_FALSE(consumer.prepare({input->view(), {}}));
  auto empty = make_int64_sequence_table(0, stream);
  REQUIRE_FALSE(consumer.prepare({empty->view(), bindings}));
}

TEST_CASE("consumer completion failure retains owned source and pending state without commit",
          "[dynamic_filter][scan_merge][consumer_retirement]")
{
  auto const strategy                              = GENERATE(compaction_strategy::CASCADE,
                                 compaction_strategy::DEFERRED_KEYS,
                                 compaction_strategy::GATHER_ONCE);
  ::cuda::stream_ref stream                        = cudf::get_default_stream();
  std::shared_ptr<cudf::table const> input         = make_int64_sequence_table(10, stream);
  std::weak_ptr<cudf::table const> source_lifetime = input;
  auto channel                                     = std::make_shared<sirius_dynamic_filter_set>();
  auto producer                                    = channel->register_producer({0});
  auto filter                                      = make_in_list_prefix(4, stream);
  REQUIRE(producer.push_filter(0, filter));
  REQUIRE(producer.push_filter(0, make_in_list_prefix(2, stream)));
  dynamic_filter_consumer consumer(
    channel, {.strategy_override = strategy, .force_counts = true}, {});
  std::vector<binding> bindings{{0, 0}};
  auto program = consumer.prepare({input->view(), bindings});
  std::optional<application_result> retained;
  REQUIRE_THROWS_AS(consumer_test_access::apply(consumer,
                                                std::move(*program),
                                                std::move(input),
                                                stream,
                                                cudf::get_current_device_resource_ref(),
                                                execution_fault::FAILED_COMPLETION,
                                                &retained),
                    sirius::unjoined_gpu_work);
  REQUIRE(retained.has_value());
  REQUIRE_FALSE(source_lifetime.expired());
  REQUIRE_FALSE(gate_test_access::filter_keep_ratio(
    consumer_test_access::gate(consumer), {filter.get(), 0}, 2));
  REQUIRE(gate_test_access::needs_combined_sample(consumer_test_access::gate(consumer),
                                                  channel->snapshot()));
  retained.reset();
  REQUIRE(source_lifetime.expired());
}

TEST_CASE("consumer count-copy failure joins work and leaves gate unchanged",
          "[dynamic_filter][scan_merge][consumer_retirement][consumer_counts]")
{
  ::cuda::stream_ref stream                 = cudf::get_default_stream();
  std::shared_ptr<cudf::table const> input  = make_int64_sequence_table(10, stream);
  std::weak_ptr<cudf::table const> lifetime = input;
  auto channel                              = std::make_shared<sirius_dynamic_filter_set>();
  auto producer                             = channel->register_producer({0});
  auto filter                               = make_in_list_prefix(4, stream);
  REQUIRE(producer.push_filter(0, filter));
  REQUIRE(producer.push_filter(0, make_in_list_prefix(2, stream)));
  dynamic_filter_consumer consumer(
    channel, {.strategy_override = compaction_strategy::GATHER_ONCE}, {});
  std::vector<binding> bindings{{0, 0}};
  auto program = consumer.prepare({input->view(), bindings});
  REQUIRE_THROWS_AS(consumer_test_access::apply(consumer,
                                                std::move(*program),
                                                std::move(input),
                                                stream,
                                                cudf::get_current_device_resource_ref(),
                                                execution_fault::AFTER_COUNTS_COPY),
                    std::runtime_error);
  REQUIRE(cudaStreamQuery(stream.get()) == cudaSuccess);
  REQUIRE(lifetime.expired());
  REQUIRE_FALSE(gate_test_access::filter_keep_ratio(
    consumer_test_access::gate(consumer), {filter.get(), 0}, 2));
  REQUIRE(gate_test_access::needs_combined_sample(consumer_test_access::gate(consumer),
                                                  channel->snapshot()));
}

TEST_CASE("consumer best effort finish drops host allocation failure after successful output",
          "[.][dynamic_filter][scan_merge][consumer_commit][host_allocation_fault]")
{
  using fault = sirius::test::scoped_host_allocation_fault;
  REQUIRE(fault::available());
  ::cuda::stream_ref stream                = cudf::get_default_stream();
  std::shared_ptr<cudf::table const> input = make_int64_sequence_table(10, stream);
  auto channel                             = std::make_shared<sirius_dynamic_filter_set>();
  auto producer                            = channel->register_producer({0});
  auto filter                              = make_in_list_prefix(4, stream);
  REQUIRE(producer.push_filter(0, filter));
  dynamic_filter_consumer consumer(channel, {}, {});
  std::vector<binding> bindings{{0, 0}};
  auto result = run_owned(consumer, input, bindings, stream);
  auto output = result.take_output();
  fault injection(fault::every_allocation_scope, 1);
  auto receipt = consumer.finish(std::move(result), 17);
  (void)injection.stop();
  REQUIRE(injection.fired());
  REQUIRE(output->num_rows() == 4);
  REQUIRE(receipt.original_batch_id == 17);
  REQUIRE(receipt.applied.size() == 1);
  REQUIRE_FALSE(gate_test_access::filter_keep_ratio(
    consumer_test_access::gate(consumer), {filter.get(), 0}, 1));
  REQUIRE_FALSE(gate_test_access::needs_combined_sample(consumer_test_access::gate(consumer),
                                                        channel->snapshot()));
}

TEST_CASE("consumer preparation host allocation failure cannot train",
          "[.][dynamic_filter][scan_merge][consumer_commit][host_allocation_fault]")
{
  using fault = sirius::test::scoped_host_allocation_fault;
  REQUIRE(fault::available());
  ::cuda::stream_ref stream = cudf::get_default_stream();
  auto input                = make_int64_sequence_table(10, stream);
  auto channel              = std::make_shared<sirius_dynamic_filter_set>();
  auto producer             = channel->register_producer({0});
  REQUIRE(producer.push_filter(0, make_in_list_prefix(4, stream)));
  dynamic_filter_consumer consumer(channel, {}, {});
  std::vector<binding> bindings{{0, 0}};
  bool failed = false;
  fault injection(fault::every_allocation_scope, 1);
  try {
    (void)consumer.prepare({input->view(), bindings});
  } catch (std::bad_alloc const&) {
    failed = true;
  }
  (void)injection.stop();
  REQUIRE(injection.fired());
  REQUIRE(failed);
  REQUIRE(gate_test_access::needs_combined_sample(consumer_test_access::gate(consumer),
                                                  channel->snapshot()));
}

TEST_CASE("dynamic filter mask fold handles sliced nullable masks and odd block tails",
          "[dynamic_filter][scan_merge][consumer_counts][mask_fold]")
{
  ::cuda::stream_ref stream      = cudf::get_default_stream();
  constexpr cudf::size_type rows = 517;
  std::vector<uint8_t> values(rows + 5, 1);
  values[9]  = 0;
  auto input = make_values_table(values, cudf::data_type{cudf::type_id::BOOL8}, stream);
  input->get_column(0).set_null_mask(
    cudf::create_null_mask(rows + 5, cudf::mask_state::ALL_VALID, stream), 1);
  cudf::set_null_mask(input->get_column(0).mutable_view().null_mask(), 8, 9, false, stream);
  auto sliced  = cudf::slice(input->view().column(0), {3, rows + 3}, stream).front();
  auto running = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::BOOL8}, rows, cudf::mask_state::UNALLOCATED, stream);
  rmm::device_uvector<cudf::size_type> counts(2, stream);
  REQUIRE(cudaMemsetAsync(counts.data(), 0, 2 * sizeof(cudf::size_type), stream.get()) ==
          cudaSuccess);
  sirius::op::fold_dynamic_filter_mask(
    running->mutable_view(), sliced, counts.data(), true, stream);
  sirius::op::fold_dynamic_filter_mask(
    running->mutable_view(), std::nullopt, counts.data() + 1, false, stream);
  std::vector<cudf::size_type> host(2);
  REQUIRE(cudaMemcpyAsync(host.data(),
                          counts.data(),
                          2 * sizeof(cudf::size_type),
                          cudaMemcpyDeviceToHost,
                          stream.get()) == cudaSuccess);
  stream.sync();
  REQUIRE(host == std::vector<cudf::size_type>{rows - 2, rows - 2});
  REQUIRE_FALSE(running->nullable());
  std::vector<uint8_t> result(rows);
  REQUIRE(
    cudaMemcpyAsync(
      result.data(), running->view().data<bool>(), rows, cudaMemcpyDeviceToHost, stream.get()) ==
    cudaSuccess);
  stream.sync();
  REQUIRE(result[5] == 0);
  REQUIRE(result[6] == 0);
  REQUIRE(result.back() == 1);
}

TEST_CASE("consumer coalesced clean aliases record only their chosen representative",
          "[dynamic_filter][scan_merge][consumer_receipt][consumer_alias]")
{
  auto const strategy                      = GENERATE(compaction_strategy::CASCADE,
                                 compaction_strategy::DEFERRED_KEYS,
                                 compaction_strategy::GATHER_ONCE);
  ::cuda::stream_ref stream                = cudf::get_default_stream();
  std::shared_ptr<cudf::table const> input = make_int64_sequence_table(10, stream);
  auto channel                             = std::make_shared<sirius_dynamic_filter_set>();
  auto producer                            = channel->register_producer({0, 1});
  auto filter = std::make_shared<counting_in_list_filter>(make_in_list_prefix(4, stream));
  REQUIRE(producer.push_filter(0, filter));
  REQUIRE(producer.push_filter(1, filter));
  auto const representative = channel->snapshot().entries().front().column_index;
  dynamic_filter_consumer consumer(
    channel, {.strategy_override = strategy, .force_counts = true}, {});
  std::vector<binding> aliases{{0, 0}, {1, 0}};
  auto result = run_owned(consumer, input, aliases, stream);
  REQUIRE(filter->mask_calls() == 1);
  REQUIRE(counts_of(result) == std::vector<cudf::size_type>{10, 4});
  auto receipt = consumer.finish(std::move(result), 1);
  REQUIRE(receipt.applied.size() == 2);
  auto& gate = consumer_test_access::gate(consumer);
  REQUIRE(gate_test_access::filter_keep_ratio(gate, {filter.get(), representative}, 2) ==
          Approx(0.4));
  REQUIRE_FALSE(gate_test_access::filter_keep_ratio(gate, {filter.get(), 1 - representative}, 2));
  REQUIRE(consumer.select_for_decode(aliases).steps.size() == 1);
}

TEST_CASE("consumer measured representative does not sample an unknown alias",
          "[dynamic_filter][scan_merge][consumer_receipt][consumer_alias]")
{
  auto const strategy                      = GENERATE(compaction_strategy::CASCADE,
                                 compaction_strategy::DEFERRED_KEYS,
                                 compaction_strategy::GATHER_ONCE);
  ::cuda::stream_ref stream                = cudf::get_default_stream();
  std::shared_ptr<cudf::table const> input = make_int64_sequence_table(10, stream);
  auto channel                             = std::make_shared<sirius_dynamic_filter_set>();
  auto producer                            = channel->register_producer({0, 1});
  auto filter = std::make_shared<counting_in_list_filter>(make_in_list_prefix(4, stream));
  REQUIRE(producer.push_filter(0, filter));
  REQUIRE(producer.push_filter(1, filter));
  auto const unknown = channel->snapshot().entries().front().column_index;
  auto const known   = 1 - unknown;
  dynamic_filter_consumer consumer(
    channel, {.strategy_override = strategy, .force_counts = true}, {});
  auto& gate = consumer_test_access::gate(consumer);
  gate.record_keep_ratio(10, 4, 2);
  gate.record_filter_keep_ratio({filter.get(), known}, 0.4, 2);
  std::vector<binding> aliases{{0, 0}, {1, 0}};
  auto result = run_owned(consumer, input, aliases, stream);
  REQUIRE(filter->mask_calls() == 1);
  auto receipt = consumer.finish(std::move(result), 1);
  REQUIRE(receipt.applied.size() == 2);
  REQUIRE(gate_test_access::filter_keep_ratio(gate, {filter.get(), known}, 2) == Approx(0.4));
  REQUIRE_FALSE(gate_test_access::filter_keep_ratio(gate, {filter.get(), unknown}, 2));
  auto selection = consumer.select_for_decode(aliases);
  REQUIRE(selection.steps.size() == 1);
  REQUIRE(selection.steps.front().representative.output_ordinal == known);
  REQUIRE(selection.steps.front().represented.size() == 2);
}

TEST_CASE("consumer alias tie order follows eligible representative snapshot position",
          "[dynamic_filter][scan_merge][consumer_counts][consumer_alias]")
{
  auto const strategy                      = GENERATE(compaction_strategy::CASCADE,
                                 compaction_strategy::DEFERRED_KEYS,
                                 compaction_strategy::GATHER_ONCE);
  ::cuda::stream_ref stream                = cudf::get_default_stream();
  std::shared_ptr<cudf::table const> input = make_int64_sequence_table(10, stream);
  auto channel                             = std::make_shared<sirius_dynamic_filter_set>();
  auto producer                            = channel->register_producer({0, 1});
  auto aliases_filter = std::make_shared<counting_in_list_filter>(make_in_list_prefix(4, stream));
  auto intervening    = make_in_list_prefix(6, stream);
  REQUIRE(producer.push_filter(0, aliases_filter));
  REQUIRE(producer.push_filter(1, aliases_filter));
  auto const skipped  = channel->snapshot().entries().front().column_index;
  auto const eligible = 1 - skipped;
  REQUIRE(producer.push_filter(skipped, intervening));
  auto snapshot = channel->snapshot();
  REQUIRE(snapshot.entries()[0].filter == aliases_filter);
  REQUIRE(snapshot.entries()[1].filter == intervening);
  REQUIRE(snapshot.entries()[2].column_index == eligible);
  dynamic_filter_consumer consumer(
    channel, {.strategy_override = strategy, .force_counts = true}, {});
  auto& gate = consumer_test_access::gate(consumer);
  gate.record_keep_ratio(10, 4, 3);
  gate.record_filter_keep_ratio({aliases_filter.get(), skipped}, 0.8, 3);
  std::vector<binding> bindings{{0, 0}, {1, 0}};
  auto result = run_owned(consumer, input, bindings, stream);
  REQUIRE(counts_of(result) == std::vector<cudf::size_type>{10, 6, 4});
  REQUIRE(aliases_filter->mask_calls() == 1);
  REQUIRE(result.take_output()->num_rows() == 4);
  auto receipt = consumer.finish(std::move(result), 1);
  REQUIRE(receipt.applied == std::vector<sirius::op::scan::applied_entry>{
                               {intervening, skipped}, {aliases_filter, eligible}});
  REQUIRE(gate_test_access::filter_keep_ratio(gate, {aliases_filter.get(), skipped}, 3) ==
          Approx(0.8));
  REQUIRE(gate_test_access::filter_keep_ratio(gate, {aliases_filter.get(), eligible}, 3) ==
          Approx(4.0 / 6.0));
}

TEST_CASE("consumer exceptional cleanup retains reachable owners and replaces retryable failure",
          "[dynamic_filter][scan_merge][consumer_retirement]")
{
  auto const strategy       = GENERATE(compaction_strategy::CASCADE,
                                 compaction_strategy::DEFERRED_KEYS,
                                 compaction_strategy::GATHER_ONCE);
  ::cuda::stream_ref stream = cudf::get_default_stream();
  std::optional<application_result> retained;
  std::weak_ptr<cudf::table const> source_lifetime;
  std::weak_ptr<sirius_dynamic_filter const> filter_lifetime;
  std::weak_ptr<sirius_dynamic_filter_set> channel_lifetime;
  {
    std::shared_ptr<cudf::table const> input = make_sequence_table<int32_t>(10, stream);
    auto channel                             = std::make_shared<sirius_dynamic_filter_set>();
    auto producer                            = channel->register_producer({0}, 3);
    auto filter                              = make_in_list_prefix<int32_t>(6, stream);
    REQUIRE(producer.push_filter(0, make_zone_map(0, 7)));
    REQUIRE(producer.push_filter(0, filter));
    REQUIRE(producer.push_filter(0, make_in_list_prefix<int32_t>(4, stream)));
    source_lifetime  = input;
    filter_lifetime  = filter;
    channel_lifetime = channel;
    dynamic_filter_consumer consumer(
      channel, {.strategy_override = strategy, .force_counts = true}, {});
    std::vector<binding> bindings{{0, 0}};
    auto program = consumer.prepare({input->view(), bindings});
    REQUIRE(program.has_value());
    std::exception_ptr error;
    try {
      (void)consumer_test_access::apply(consumer,
                                        std::move(*program),
                                        std::move(input),
                                        stream,
                                        cudf::get_current_device_resource_ref(),
                                        execution_fault::FAILED_CLEANUP,
                                        &retained);
    } catch (...) {
      error = std::current_exception();
    }
    REQUIRE(error != nullptr);
    REQUIRE_THROWS_AS(std::rethrow_exception(error), sirius::unjoined_gpu_work);
    auto const classification = sirius::op::detail::classify_failure(error);
    REQUIRE(classification == sirius::op::detail::accumulation_failure::LEAK);
    REQUIRE_FALSE(sirius::op::detail::recoverable(classification));
    REQUIRE(retained.has_value());
    REQUIRE_FALSE(gate_test_access::filter_keep_ratio(
      consumer_test_access::gate(consumer), {filter.get(), 0}, 3));
    REQUIRE(gate_test_access::needs_combined_sample(consumer_test_access::gate(consumer),
                                                    channel->snapshot()));
  }
  REQUIRE_FALSE(source_lifetime.expired());
  REQUIRE_FALSE(filter_lifetime.expired());
  REQUIRE_FALSE(channel_lifetime.expired());
  auto const storage = consumer_test_access::storage(*retained);
  REQUIRE(storage.has_source);
  REQUIRE(storage.filter_owners == 3);
  REQUIRE(storage.has_ast);
  REQUIRE(storage.has_mask);
  REQUIRE(storage.has_output);
  REQUIRE_FALSE(storage.completed);
  REQUIRE(storage.has_row_ids == (strategy == compaction_strategy::DEFERRED_KEYS));
  REQUIRE(storage.has_counters == (strategy == compaction_strategy::GATHER_ONCE));
  if (strategy == compaction_strategy::GATHER_ONCE) {
    REQUIRE(
      std::vector<cudf::size_type>(storage.copied_counts.begin(), storage.copied_counts.end()) ==
      std::vector<cudf::size_type>{8, 6, 4});
  } else {
    REQUIRE(storage.copied_counts.empty());
  }
  REQUIRE(cudaStreamQuery(stream.get()) == cudaSuccess);
  retained.reset();
  REQUIRE(source_lifetime.expired());
  REQUIRE(filter_lifetime.expired());
  REQUIRE(channel_lifetime.expired());
}

TEST_CASE("consumer result releases scratch and source before commit",
          "[dynamic_filter][scan_merge][consumer_retirement]")
{
  auto const strategy                              = GENERATE(compaction_strategy::CASCADE,
                                 compaction_strategy::DEFERRED_KEYS,
                                 compaction_strategy::GATHER_ONCE);
  ::cuda::stream_ref stream                        = cudf::get_default_stream();
  std::shared_ptr<cudf::table const> input         = make_sequence_table<int32_t>(10, stream);
  std::weak_ptr<cudf::table const> source_lifetime = input;
  auto channel                                     = std::make_shared<sirius_dynamic_filter_set>();
  auto producer                                    = channel->register_producer({0}, 3);
  auto wide                                        = make_in_list_prefix<int32_t>(6, stream);
  auto narrow                                      = make_in_list_prefix<int32_t>(4, stream);
  REQUIRE(producer.push_filter(0, make_zone_map(0, 7)));
  REQUIRE(producer.push_filter(0, wide));
  REQUIRE(producer.push_filter(0, narrow));
  dynamic_filter_consumer consumer(
    channel, {.strategy_override = strategy, .force_counts = true}, {});
  std::vector<binding> bindings{{0, 0}};
  auto program = consumer.prepare({input->view(), bindings});
  REQUIRE(program.has_value());
  auto result = consumer_test_access::apply(consumer,
                                            std::move(*program),
                                            std::move(input),
                                            stream,
                                            cudf::get_current_device_resource_ref());
  REQUIRE(source_lifetime.expired());
  auto const storage = consumer_test_access::storage(result);
  REQUIRE(storage.completed);
  REQUIRE(storage.has_output);
  REQUIRE(storage.filter_owners == 3);
  REQUIRE_FALSE(storage.has_source);
  REQUIRE_FALSE(storage.has_ast);
  REQUIRE_FALSE(storage.has_mask);
  REQUIRE_FALSE(storage.has_row_ids);
  REQUIRE_FALSE(storage.has_counters);
  REQUIRE(storage.copied_counts.empty());
  REQUIRE(result.strategy() == strategy);
  REQUIRE(counts_of(result) == std::vector<cudf::size_type>{8, 6, 4});
  REQUIRE(result.take_output()->num_rows() == 4);

  auto moved = std::move(result);
  // NOLINTNEXTLINE(bugprone-use-after-move): a moved-from result reports empty values.
  REQUIRE_FALSE(result.strategy().has_value());
  consumer.commit(std::move(moved));
  // NOLINTNEXTLINE(bugprone-use-after-move): a committed result reports empty values.
  REQUIRE_FALSE(moved.strategy().has_value());
  auto& gate = consumer_test_access::gate(consumer);
  REQUIRE(gate_test_access::filter_keep_ratio(gate, {wide.get(), 0}, 3) == Approx(0.75));
  REQUIRE(gate_test_access::filter_keep_ratio(gate, {narrow.get(), 0}, 3) == Approx(4.0 / 6.0));
  REQUIRE_FALSE(gate_test_access::needs_combined_sample(gate, channel->snapshot()));
}

TEST_CASE("consumer combined sample does not read counts back",
          "[dynamic_filter][scan_merge][consumer_counts]")
{
  ::cuda::stream_ref stream                = cudf::get_default_stream();
  std::shared_ptr<cudf::table const> input = make_int64_sequence_table(10, stream);
  auto channel                             = std::make_shared<sirius_dynamic_filter_set>();
  auto producer                            = channel->register_producer({0});
  auto first                               = make_in_list_prefix(5, stream);
  auto second                              = make_in_list_prefix(3, stream);
  REQUIRE(producer.push_filter(0, first));
  REQUIRE(producer.push_filter(0, second));
  dynamic_filter_consumer consumer(
    channel, {.strategy_override = compaction_strategy::GATHER_ONCE}, {});
  auto& gate = consumer_test_access::gate(consumer);
  gate.record_filter_keep_ratio({first.get(), 0}, 0.5, 2);
  gate.record_filter_keep_ratio({second.get(), 0}, 0.3, 2);
  REQUIRE(gate_test_access::needs_combined_sample(gate, channel->snapshot()));
  std::vector<binding> bindings{{0, 0}};
  auto result = run_owned(consumer, input, bindings, stream);
  REQUIRE_FALSE(result.counts_read_back());
  REQUIRE(result.prefix_counts().empty());
  REQUIRE(result.take_output()->num_rows() == 3);
  auto receipt = consumer.finish(std::move(result), 1);
  // NOLINTNEXTLINE(bugprone-use-after-move): a finished result reports empty values.
  REQUIRE_FALSE(result.strategy().has_value());
  REQUIRE(receipt.applied.size() == 2);
  REQUIRE_FALSE(gate_test_access::needs_combined_sample(gate, channel->snapshot()));
}

namespace {
/// Membership filter that claims a replica on every device and records the device each probe asks
/// for, while probing with its wrapped current-device IN-list.
class any_device_filter final : public sirius_dynamic_filter,
                                public sirius::op::sirius_mask_applicable {
 public:
  explicit any_device_filter(std::shared_ptr<sirius::op::sirius_dynamic_in_list_filter> inner)
    : _inner(std::move(inner))
  {
  }

  [[nodiscard]] sirius_dynamic_filter_kind kind() const override { return _inner->kind(); }

  [[nodiscard]] bool is_available_on_device(int) const noexcept override { return true; }

  using sirius::op::sirius_mask_applicable::compute_mask;
  [[nodiscard]] std::unique_ptr<cudf::column> compute_mask(
    cudf::column_view const& probe,
    std::uint32_t const* prior_mask_words,
    int device_id,
    ::cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) const override
  {
    requested_devices.push_back(device_id);
    return _inner->compute_mask(probe, prior_mask_words, -1, stream, mr);
  }

  mutable std::vector<int> requested_devices;

 private:
  std::shared_ptr<sirius::op::sirius_dynamic_in_list_filter> _inner;
};

/// Host memory space whose ID names @p device_id, so a batch placed in it reports that device while
/// its table stays on the current one. It reserves no memory.
std::unique_ptr<cucascade::memory::memory_space> make_space_naming_device(int device_id)
{
  return std::make_unique<cucascade::memory::memory_space>(
    cucascade::memory::host_memory_space_config{
      .numa_id = device_id, .initial_number_pools = 0, .mr_factory_fn = [](int, std::size_t) {
        return ::cuda::mr::any_resource<::cuda::mr::device_accessible>{
          cucascade::memory::null_device_memory_resource{}};
      }});
}
}  // namespace

TEST_CASE("consumer passes a batch on another device through without filtering or claims",
          "[dynamic_filter][scan_merge][consumer_device]")
{
  ::cuda::stream_ref stream                = cudf::get_default_stream();
  std::shared_ptr<cudf::table const> input = make_int64_sequence_table(10, stream);
  auto* const gpu_space                    = sirius::test::operator_utils::get_default_gpu_space();
  int current_device                       = -1;
  REQUIRE(cudaGetDevice(&current_device) == cudaSuccess);
  REQUIRE(gpu_space->get_device_id() == current_device);
  auto const other_device = current_device + 1000;
  auto channel            = std::make_shared<sirius_dynamic_filter_set>();
  auto producer           = channel->register_producer({0});
  auto filter             = std::make_shared<any_device_filter>(make_in_list_prefix(4, stream));
  REQUIRE(producer.push_filter(0, filter));
  sirius::op::dynamic_filter_stats stats;
  dynamic_filter_consumer consumer(channel, {.stats = &stats}, {});
  auto const mr        = cudf::get_current_device_resource_ref();
  auto const batch_for = [&](cucascade::memory::memory_space& space) {
    return sirius::make_data_batch(
      std::make_unique<cudf::table>(input->view(), stream), space, stream, {});
  };
  auto const require_passed_through = [&](application_result result) {
    REQUIRE_FALSE(result.completed());
    REQUIRE(result.take_output() == nullptr);
    REQUIRE(result.prefix_counts().empty());
    REQUIRE(filter->requested_devices.empty());
    auto const storage = consumer_test_access::storage(result);
    REQUIRE_FALSE(storage.completed);
    REQUIRE_FALSE(storage.has_source);
    auto receipt = consumer.finish(std::move(result), 1);
    REQUIRE(receipt.endpoint == channel);
    REQUIRE(receipt.original_batch_id == 1);
    REQUIRE(receipt.applied.empty());
    auto& gate = consumer_test_access::gate(consumer);
    REQUIRE_FALSE(gate_test_access::filter_keep_ratio(gate, {filter.get(), 0}, 1).has_value());
    REQUIRE(gate_test_access::needs_combined_sample(gate, channel->snapshot()));
    REQUIRE(stats.snapshot().applications_skipped_other_device == 1);
  };

  SECTION("the batch's memory space names another device")
  {
    // The program is prepared for the current device, so only the batch's memory space differs.
    auto const space = make_space_naming_device(other_device);
    // This synthetic space cannot own the current device's writer stream. Keep the real table
    // alive through a view with an unknown writer; the consumer must decline before GPU access.
    auto representation = std::make_unique<cucascade::gpu_table_representation>(
      input->view(), input, input->alloc_size(), *space, ::cuda::stream_ref{cudaStream_t{nullptr}});
    auto batch =
      cucascade::data_batch::make(sirius::get_next_batch_id(), std::move(representation));
    auto program = prepare_identity(consumer, input->view());
    REQUIRE(program.has_value());
    auto source = batch->to_read_only();
    require_passed_through(
      consumer_test_access::apply(consumer, std::move(*program), std::move(source), stream, mr));
  }
  SECTION("the program was prepared for another device")
  {
    // The bare table has no memory space, so only the program names the other device.
    auto program = prepare_identity(consumer, input->view(), std::nullopt, other_device);
    REQUIRE(program.has_value());
    require_passed_through(
      consumer_test_access::apply(consumer, std::move(*program), input, stream, mr));
  }

  // The same batch on the current device filters, so the filter itself is not what declined.
  auto batch = batch_for(*gpu_space);
  auto local = prepare_identity(consumer, input->view());
  REQUIRE(local.has_value());
  auto filtered =
    consumer_test_access::apply(consumer, std::move(*local), batch->to_read_only(), stream, mr);
  REQUIRE(filtered.completed());
  REQUIRE(filtered.take_output()->num_rows() == 4);
  REQUIRE(filter->requested_devices == std::vector<int>{current_device});
  consumer.commit(std::move(filtered));
  REQUIRE(stats.snapshot().applications_skipped_other_device == 1);
}

TEST_CASE("consumer pass-through finishes with the prior receipt's history unchanged",
          "[dynamic_filter][scan_merge][consumer_device][decode_history]")
{
  ::cuda::stream_ref stream                = cudf::get_default_stream();
  std::shared_ptr<cudf::table const> input = make_int64_sequence_table(10, stream);
  int current_device                       = -1;
  REQUIRE(cudaGetDevice(&current_device) == cudaSuccess);
  auto channel  = std::make_shared<sirius_dynamic_filter_set>();
  auto producer = channel->register_producer({0, 1});
  // The program names another device, so the filter it would run must claim that device.
  auto attached = std::make_shared<any_device_filter>(make_in_list_prefix(4, stream));
  auto applied  = make_in_list_prefix(6, stream);
  REQUIRE(producer.push_filter(0, attached));
  REQUIRE(producer.push_filter(1, applied));
  sirius::op::dynamic_filter_stats stats;
  dynamic_filter_consumer consumer(channel, {.stats = &stats}, {});
  std::vector<binding> bindings{{0, 0}, {1, 0}};
  // A scan's receipt: the decoder was offered `attached`, and a checkpoint already ran `applied`.
  batch_receipt const prior{channel, 0, {{applied, 1}}, {{attached, 0}}};
  auto program = consumer.prepare({.source    = input->view(),
                                   .bindings  = bindings,
                                   .prior     = &prior,
                                   .device_id = current_device + 1000});
  REQUIRE(program.has_value());
  auto result = consumer_test_access::apply(
    consumer, std::move(*program), input, stream, cudf::get_current_device_resource_ref());
  REQUIRE(result.take_output() == nullptr);
  REQUIRE(stats.snapshot().applications_skipped_other_device == 1);
  auto const receipt = consumer.finish(std::move(result), 7);
  CHECK(receipt.endpoint == channel);
  CHECK(receipt.original_batch_id == 7);
  CHECK(receipt.applied == prior.applied);
  CHECK(receipt.decode_attached == prior.decode_attached);
}

TEST_CASE("decode selection ranks unmeasured filters by kind and key count after measured ones",
          "[dynamic_filter][scan_merge][consumer_decode_order]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();
  auto const mr             = cudf::get_current_device_resource_ref();
  auto channel              = std::make_shared<sirius_dynamic_filter_set>();
  auto producer             = channel->register_producer({0, 1, 2, 3});
  auto bloom_keys           = make_sequence_column<int64_t>(4, stream);
  auto small_keys           = make_sequence_column<int64_t>(3, stream);
  auto large                = make_in_list_prefix(6, stream);
  auto few                  = make_in_list_prefix(2, stream);

  auto bloom =
    std::make_shared<sirius::op::sirius_dynamic_bloom_filter>(bloom_keys->view(), stream, mr);
  auto small = std::make_shared<sirius::op::sirius_dynamic_small_in_list_filter>(
    small_keys->view(), stream, mr);

  REQUIRE(producer.push_filter(0, bloom));
  REQUIRE(producer.push_filter(1, large));
  REQUIRE(producer.push_filter(2, small));
  REQUIRE(producer.push_filter(3, few));
  dynamic_filter_consumer consumer(channel, {}, {});
  auto& gate = consumer_test_access::gate(consumer);
  gate.record_keep_ratio(10, 3, 4);
  auto const bindings = sirius::op::scan::identity_bindings(4);
  auto const order    = [&] {
    std::vector<sirius_dynamic_filter const*> filters;
    for (auto const& step : consumer.select_for_decode(bindings).steps) {
      filters.push_back(step.representative.filter.get());
    }
    return filters;
  };
  REQUIRE(order() == std::vector<sirius_dynamic_filter const*>{
                       small.get(), few.get(), large.get(), bloom.get()});
  gate.record_filter_keep_ratio({bloom.get(), 0}, 0.1, 4);
  REQUIRE(order() == std::vector<sirius_dynamic_filter const*>{
                       bloom.get(), small.get(), few.get(), large.get()});
}

namespace {
using sirius::op::scan::cached_view_layout;
using sirius::op::scan::owning_table_view;
using sirius::op::scan::detail::read_checkpoint;
using sirius::op::scan::detail::read_retirement_test_hooks;
using sirius::op::scan::detail::scan_operator_test_access;

std::shared_ptr<cudf::table const> make_cached_residual_input(cudf::size_type rows,
                                                              cudf::size_type residual_rows,
                                                              ::cuda::stream_ref stream)
{
  std::vector<uint8_t> keep(static_cast<std::size_t>(rows), 0);
  std::fill_n(keep.begin(), residual_rows, uint8_t{1});
  auto mask    = make_values_table(keep, cudf::data_type{cudf::type_id::BOOL8}, stream);
  auto columns = mask->release();
  columns.insert(columns.begin(), make_sequence_column<int64_t>(rows, stream));
  return std::make_shared<cudf::table>(std::move(columns));
}

cached_view_layout cached_layout(bool residual = false)
{
  cached_view_layout layout;
  layout.output_columns    = {0};
  layout.projected_columns = {0};
  if (residual) {
    layout.residual = std::make_unique<sirius::ast::node>(
      sirius::ast::reference{1, sirius::logical_type::make(sirius::type_id::BOOLEAN)});
  }
  return layout;
}

application_result run_cached(dynamic_filter_consumer& consumer,
                              std::shared_ptr<cudf::table const> const& input,
                              std::span<binding const> bindings,
                              cached_view_layout const& layout,
                              ::cuda::stream_ref stream,
                              batch_receipt const* prior = nullptr,
                              bool positions             = false)
{
  auto program = consumer.prepare({.source             = input->view(),
                                   .bindings           = bindings,
                                   .prior              = prior,
                                   .cached_layout      = &layout,
                                   .want_row_positions = positions});
  REQUIRE(program.has_value());
  return consumer.apply(std::move(*program),
                        owning_table_view{input, input->view()},
                        stream,
                        cudf::get_current_device_resource_ref());
}
}  // namespace

TEST_CASE("cached consumer owns prepared layout and gathers duplicate nested payload once",
          "[dynamic_filter][scan_merge][cached_checkpoint]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();
  auto const mr             = cudf::get_current_device_resource_ref();
  auto columns              = make_sequence_table<int64_t>(100, stream)->release();
  columns.push_back(make_nullable_list_payload(100, stream));
  std::shared_ptr<cudf::table const> input = std::make_shared<cudf::table>(std::move(columns));
  auto channel                             = std::make_shared<sirius_dynamic_filter_set>();
  auto producer                            = channel->register_producer({0, 1});
  auto filter = std::make_shared<counting_in_list_filter>(make_in_list_prefix(40, stream));
  REQUIRE(producer.push_filter(0, filter));
  REQUIRE(producer.push_filter(1, filter));
  dynamic_filter_consumer consumer(channel, {.force_counts = true}, {});
  std::vector<binding> bindings{{0, 0}, {1, 0}};
  auto layout              = cached_layout();
  layout.output_columns    = {0, 0, 1};
  layout.projected_columns = {1, 0, 0};
  auto program             = consumer.prepare({.source             = input->view(),
                                               .bindings           = bindings,
                                               .cached_layout      = &layout,
                                               .want_row_positions = true});
  REQUIRE(program);
  layout.projected_columns.clear();
  bindings.clear();
  auto result =
    consumer.apply(std::move(*program), owning_table_view{input, input->view()}, stream, mr);
  REQUIRE(result.strategy() == compaction_strategy::GATHER_ONCE);
  REQUIRE(filter->probe_rows() == std::vector<cudf::size_type>{100});
  REQUIRE(counts_of(result) == std::vector<cudf::size_type>{100, 40});
  auto output    = result.take_output();
  auto positions = result.take_row_positions();
  REQUIRE(output->num_columns() == 3);
  REQUIRE(output->num_rows() == 40);
  REQUIRE(output->view().column(1).head<int64_t>() != output->view().column(2).head<int64_t>());
  auto expected =
    cudf::table(input->view().select(std::vector<cudf::size_type>{1, 0, 0}), stream, mr);
  auto first = cudf::slice(expected.view(), std::vector<cudf::size_type>{0, 40});
  REQUIRE(sirius::test::expect_tables_equivalent_impl(output->view(), first.front(), stream));
  std::vector<int32_t> expected_positions(40);
  std::iota(expected_positions.begin(), expected_positions.end(), 0);
  REQUIRE(to_host_int32(positions->view(), stream) == expected_positions);
  REQUIRE(input->num_rows() == 100);
  auto receipt = consumer.finish(std::move(result), 19);
  REQUIRE(receipt.applied.size() == 2);
  REQUIRE(receipt.original_batch_id == 19);
}

TEST_CASE("cached residual uses original positions and residual denominator",
          "[dynamic_filter][scan_merge][cached_checkpoint][consumer_counts]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();
  auto input                = make_cached_residual_input(100, 40, stream);
  auto channel              = std::make_shared<sirius_dynamic_filter_set>();
  auto producer             = channel->register_producer({0});
  auto filter = std::make_shared<counting_in_list_filter>(make_in_list_prefix(30, stream));
  REQUIRE(producer.push_filter(0, filter));
  dynamic_filter_consumer consumer(channel, {.keep_threshold = 0.5}, {});
  std::vector<binding> bindings{{0, 0}};
  auto layout  = cached_layout(true);
  auto program = consumer.prepare({.source             = input->view(),
                                   .bindings           = bindings,
                                   .cached_layout      = &layout,
                                   .want_row_positions = true});
  REQUIRE(program);
  layout.residual.reset();
  auto result = consumer.apply(std::move(*program),
                               owning_table_view{input, input->view()},
                               stream,
                               cudf::get_current_device_resource_ref());
  REQUIRE(counts_of(result) == std::vector<cudf::size_type>{40, 30});
  REQUIRE(result.counts_read_back());
  REQUIRE(filter->probe_rows() == std::vector<cudf::size_type>{100});
  // The residual ran first, so its conjunction hints the only membership probe.
  REQUIRE(filter->hinted_calls() == 1);
  REQUIRE(result.take_output()->num_rows() == 30);
  auto positions = result.take_row_positions();
  REQUIRE(to_host_int32(positions->view(), stream).back() == 29);
  (void)consumer.finish(std::move(result), 1);
  auto& gate = consumer_test_access::gate(consumer);
  REQUIRE(gate_test_access::filter_keep_ratio(gate, {filter.get(), 0}, 1) == Approx(0.75));
  REQUIRE_FALSE(gate_test_access::applicable(gate, channel->snapshot()));
  REQUIRE_FALSE(gate_test_access::needs_combined_sample(gate, channel->snapshot()));
}

TEST_CASE("cached masks preserve correlated prefixes across original-row probes",
          "[dynamic_filter][scan_merge][cached_checkpoint][consumer_counts]")
{
  auto const rows                          = GENERATE(0, 100);
  auto const count                         = GENERATE(1, 4, 9);
  ::cuda::stream_ref stream                = cudf::get_default_stream();
  std::shared_ptr<cudf::table const> input = make_int64_sequence_table(rows, stream);
  auto channel                             = std::make_shared<sirius_dynamic_filter_set>();
  std::vector<std::shared_ptr<counting_in_list_filter>> filters;
  for (int i = 0; i < count; ++i) {
    auto producer = channel->register_producer({0});
    auto filter =
      std::make_shared<counting_in_list_filter>(make_in_list_prefix(90 - i * 10, stream));
    REQUIRE(producer.push_filter(0, filter));
    filters.push_back(std::move(filter));
  }
  dynamic_filter_consumer consumer(channel, {.force_counts = true}, {});
  std::vector<binding> bindings{{0, 0}};
  auto layout = cached_layout();
  if (rows == 0) {
    REQUIRE_FALSE(consumer.prepare({.source             = input->view(),
                                    .bindings           = bindings,
                                    .cached_layout      = &layout,
                                    .want_row_positions = true}));
    for (auto const& filter : filters) {
      REQUIRE(filter->mask_calls() == 0);
    }
    return;
  }
  auto result = run_cached(consumer, input, bindings, layout, stream, nullptr, true);
  std::vector<cudf::size_type> expected_counts{rows};
  for (int i = 0; i < count; ++i) {
    expected_counts.push_back(90 - i * 10);
    REQUIRE(filters[i]->probe_rows() == std::vector<cudf::size_type>{rows});
    // Without a residual nothing precedes the first probe, so only later probes take a hint.
    REQUIRE(filters[i]->hinted_calls() == (i == 0 ? 0 : 1));
  }
  REQUIRE(counts_of(result) == expected_counts);
  auto output    = result.take_output();
  auto positions = result.take_row_positions();
  REQUIRE(output->num_rows() == expected_counts.back());
  REQUIRE(positions->size() == output->num_rows());
  REQUIRE(to_host_int64(output->view().column(0), stream).size() ==
          static_cast<std::size_t>(expected_counts.back()));
  auto receipt = consumer.finish(std::move(result), 2);
  REQUIRE(receipt.applied.size() == static_cast<std::size_t>(count));
}

TEST_CASE("cached checkpoint declines without membership and retains runtime declined source",
          "[dynamic_filter][scan_merge][cached_checkpoint]")
{
  auto const residual       = GENERATE(false, true);
  ::cuda::stream_ref stream = cudf::get_default_stream();
  auto input                = make_cached_residual_input(100, 40, stream);
  auto channel              = std::make_shared<sirius_dynamic_filter_set>();
  dynamic_filter_consumer consumer(channel, {}, {});
  std::vector<binding> bindings{{0, 0}};
  auto layout = cached_layout(residual);
  REQUIRE_FALSE(
    consumer.prepare({.source = input->view(), .bindings = bindings, .cached_layout = &layout}));
  auto producer  = channel->register_producer({0});
  auto declining = std::make_shared<nullable_declining_filter>();
  REQUIRE(producer.push_filter(0, declining));
  auto result = run_cached(consumer, input, bindings, layout, stream, nullptr, true);
  auto output = result.take_output();
  if (residual) {
    REQUIRE(output);
    REQUIRE(output->num_rows() == 40);
    REQUIRE(result.take_row_positions()->size() == 40);
  } else {
    REQUIRE_FALSE(output);
    REQUIRE_FALSE(result.take_row_positions());
    auto source = result.take_source();
    REQUIRE(source.view().column(0).head<int64_t>() == input->view().column(0).head<int64_t>());
    REQUIRE(source.num_rows() == 100);
  }
  auto receipt = consumer.finish(std::move(result), 1);
  REQUIRE(receipt.applied.empty());
  REQUIRE(gate_test_access::needs_combined_sample(consumer_test_access::gate(consumer),
                                                  channel->snapshot()));
}

TEST_CASE("cached clean sample remains available after late endpoint publication",
          "[dynamic_filter][scan_merge][cached_checkpoint][consumer_receipt]")
{
  ::cuda::stream_ref stream                = cudf::get_default_stream();
  std::shared_ptr<cudf::table const> input = make_int64_sequence_table(100, stream);
  auto channel                             = std::make_shared<sirius_dynamic_filter_set>();
  auto producer                            = channel->register_producer({0});
  auto first                               = make_in_list_prefix(100, stream);
  auto late                                = make_in_list_prefix(10, stream);
  REQUIRE(producer.push_filter(0, first));
  dynamic_filter_consumer consumer(channel, {}, {});
  auto& gate = consumer_test_access::gate(consumer);
  std::vector<binding> bindings{{0, 0}};
  auto layout = cached_layout();
  auto scan   = run_cached(consumer, input, bindings, layout, stream);
  REQUIRE(scan.take_output()->num_rows() == 100);
  auto receipt = consumer.finish(std::move(scan), 3);
  REQUIRE_FALSE(gate_test_access::applicable(gate, channel->snapshot()));
  REQUIRE(producer.push_filter(0, late));
  auto endpoint = run_owned(consumer, input, bindings, stream, &receipt);
  REQUIRE(endpoint.take_output()->num_rows() == 10);
  (void)consumer.finish(std::move(endpoint), 3);
  REQUIRE(gate_test_access::filter_keep_ratio(gate, {late.get(), 0}, 2) == Approx(0.1));
  REQUIRE(gate_test_access::needs_combined_sample(gate, channel->snapshot()));
  auto clean = run_cached(consumer, input, bindings, layout, stream);
  REQUIRE(clean.take_output()->num_rows() == 10);
  (void)consumer.finish(std::move(clean), 4);
  REQUIRE_FALSE(gate_test_access::needs_combined_sample(gate, channel->snapshot()));
  REQUIRE(gate_test_access::applicable(gate, channel->snapshot()));
}

TEST_CASE("cached alias receipts suppress represented work and attached training",
          "[dynamic_filter][scan_merge][cached_checkpoint][consumer_alias]")
{
  auto const applied                       = GENERATE(false, true);
  ::cuda::stream_ref stream                = cudf::get_default_stream();
  std::shared_ptr<cudf::table const> input = make_int64_sequence_table(100, stream);
  auto channel                             = std::make_shared<sirius_dynamic_filter_set>();
  auto producer                            = channel->register_producer({0, 1});
  auto alias = std::make_shared<counting_in_list_filter>(make_in_list_prefix(40, stream));
  auto fresh = make_in_list_prefix(10, stream);
  REQUIRE(producer.push_filter(0, alias));
  REQUIRE(producer.push_filter(1, alias));
  REQUIRE(producer.push_filter(0, fresh));
  dynamic_filter_consumer consumer(channel, {.force_counts = true}, {});
  std::vector<binding> bindings{{0, 0}, {1, 0}};
  auto layout = cached_layout();
  batch_receipt prior{channel, 2, {}, {}};
  (applied ? prior.applied : prior.decode_attached).push_back({alias, 1});
  auto result = run_cached(consumer, input, bindings, layout, stream, &prior);
  REQUIRE(alias->mask_calls() == (applied ? 0 : 1));
  REQUIRE(result.take_output()->num_rows() == 10);
  auto receipt = consumer.finish(std::move(result), 2);
  REQUIRE(receipt.applied.size() == (applied ? 2 : 3));
  auto& gate = consumer_test_access::gate(consumer);
  REQUIRE_FALSE(gate_test_access::filter_keep_ratio(gate, {alias.get(), 0}, 3));
  REQUIRE_FALSE(gate_test_access::filter_keep_ratio(gate, {alias.get(), 1}, 3));
  REQUIRE(gate_test_access::filter_keep_ratio(gate, {fresh.get(), 0}, 3) ==
          Approx(applied ? 0.1 : 0.25));
  REQUIRE(gate_test_access::needs_combined_sample(gate, channel->snapshot()));
}

TEST_CASE("cached retirement retains all owners on failed completion and permits safe retry",
          "[dynamic_filter][scan_merge][cached_checkpoint][consumer_retirement]")
{
  auto const fused          = GENERATE(false, true);
  auto const point          = GENERATE(read_checkpoint::RESIDUAL,
                              read_checkpoint::MASK_FOLD,
                              read_checkpoint::COUNTS_COPY,
                              read_checkpoint::PAYLOAD_GATHER,
                              read_checkpoint::READER_EVENT);
  auto const fail_join      = GENERATE(false, true);
  ::cuda::stream_ref stream = cudf::get_default_stream();
  std::weak_ptr<cudf::table const> source_lifetime;
  std::weak_ptr<sirius_dynamic_filter const> filter_lifetime;
  std::weak_ptr<sirius_dynamic_filter_set> channel_lifetime;
  {
    read_retirement_test_hooks faults{point, fail_join};
    {
      auto input    = make_cached_residual_input(100, 80, stream);
      auto channel  = std::make_shared<sirius_dynamic_filter_set>();
      auto producer = channel->register_producer({0}, fused ? 9 : 2);
      auto first    = make_in_list_prefix(40, stream);
      auto second   = make_in_list_prefix(10, stream);
      REQUIRE(producer.push_filter(0, first));
      REQUIRE(producer.push_filter(0, second));
      if (fused) {
        for (int i = 0; i < 7; ++i) {
          REQUIRE(producer.push_filter(0, make_in_list_prefix(100, stream)));
        }
      }
      source_lifetime  = input;
      filter_lifetime  = first;
      channel_lifetime = channel;
      dynamic_filter_consumer consumer(
        channel, {.force_counts = true, .fused_membership = fused}, {});
      std::vector<binding> bindings{{0, 0}};
      auto layout = cached_layout(true);
      if (fail_join) {
        REQUIRE_THROWS_AS(run_cached(consumer, input, bindings, layout, stream, nullptr, true),
                          sirius::unjoined_gpu_work);
      } else {
        REQUIRE_THROWS_AS(run_cached(consumer, input, bindings, layout, stream, nullptr, true),
                          rmm::out_of_memory);
      }
      REQUIRE(gate_test_access::needs_combined_sample(consumer_test_access::gate(consumer),
                                                      channel->snapshot()));
      REQUIRE_FALSE(gate_test_access::filter_keep_ratio(
        consumer_test_access::gate(consumer), {first.get(), 0}, 2));
      if (fail_join) {
        REQUIRE(faults.retained.has_value());
        // The retained state answers only as the consumer's own type.
        REQUIRE_FALSE(scan_operator_test_access::retained_assembly(faults));
        auto const retained = consumer_test_access::retained_cached_storage(faults);
        REQUIRE(retained);
        auto const& storage = *retained;
        REQUIRE(storage.has_source);
        REQUIRE(storage.filter_owners == (fused ? 9 : 2));
        REQUIRE(storage.fused_rounds == (fused && point != read_checkpoint::RESIDUAL ? 3 : 0));
        // The residual's evaluator is freed once its result is folded into the conjunction.
        REQUIRE(storage.has_ast == (point == read_checkpoint::RESIDUAL));
        REQUIRE_FALSE(storage.completed);
        // Each membership mask is freed right after its fold into the conjunction.
        REQUIRE_FALSE(storage.has_mask);
        if (point != read_checkpoint::RESIDUAL) { REQUIRE(storage.has_counters); }
        if (point == read_checkpoint::COUNTS_COPY || point == read_checkpoint::PAYLOAD_GATHER ||
            point == read_checkpoint::READER_EVENT) {
          std::vector<cudf::size_type> expected{80, 40, 10};
          if (fused) { expected.resize(10, 10); }
          REQUIRE(std::vector<cudf::size_type>(storage.copied_counts.begin(),
                                               storage.copied_counts.end()) == expected);
        }
        if (point == read_checkpoint::PAYLOAD_GATHER || point == read_checkpoint::READER_EVENT) {
          REQUIRE(storage.has_output);
        }
        if (point == read_checkpoint::PAYLOAD_GATHER || point == read_checkpoint::READER_EVENT) {
          REQUIRE(storage.has_row_ids);
        }
      } else {
        REQUIRE_FALSE(faults.retained.has_value());
        REQUIRE_FALSE(consumer_test_access::retained_cached_storage(faults));
        faults.fail_at.reset();
        auto retry = run_cached(consumer, input, bindings, layout, stream, nullptr, true);
        REQUIRE(retry.take_output()->num_rows() == 10);
        auto receipt = consumer.finish(std::move(retry), 5);
        REQUIRE(receipt.applied.size() == (fused ? 9 : 2));
      }
    }
    REQUIRE(source_lifetime.expired() == !fail_join);
    REQUIRE(filter_lifetime.expired() == !fail_join);
    REQUIRE(channel_lifetime.expired() == !fail_join);
  }
  REQUIRE(source_lifetime.expired());
  REQUIRE(filter_lifetime.expired());
  REQUIRE(channel_lifetime.expired());
}

TEST_CASE("cached single-mask retirement retains its owners on failed completion",
          "[dynamic_filter][scan_merge][cached_checkpoint][consumer_retirement]")
{
  // Without a residual one membership mask selects the payload and the positions directly.
  auto const point     = GENERATE(read_checkpoint::PAYLOAD_GATHER, read_checkpoint::READER_EVENT);
  auto const fail_join = GENERATE(false, true);
  ::cuda::stream_ref stream = cudf::get_default_stream();
  CAPTURE(point, fail_join);
  std::weak_ptr<cudf::table const> source_lifetime;
  std::weak_ptr<sirius_dynamic_filter const> filter_lifetime;
  {
    read_retirement_test_hooks faults{point, fail_join};
    {
      std::shared_ptr<cudf::table const> input = make_int64_sequence_table(100, stream);
      auto channel                             = std::make_shared<sirius_dynamic_filter_set>();
      auto producer                            = channel->register_producer({0});
      auto filter                              = make_in_list_prefix(40, stream);
      REQUIRE(producer.push_filter(0, filter));
      source_lifetime = input;
      filter_lifetime = filter;
      dynamic_filter_consumer consumer(channel, {}, {});
      std::vector<binding> bindings{{0, 0}};
      auto layout = cached_layout();
      if (fail_join) {
        REQUIRE_THROWS_AS(run_cached(consumer, input, bindings, layout, stream, nullptr, true),
                          sirius::unjoined_gpu_work);
      } else {
        REQUIRE_THROWS_AS(run_cached(consumer, input, bindings, layout, stream, nullptr, true),
                          rmm::out_of_memory);
      }
      auto& gate = consumer_test_access::gate(consumer);
      REQUIRE(gate_test_access::needs_combined_sample(gate, channel->snapshot()));
      REQUIRE_FALSE(gate_test_access::filter_keep_ratio(gate, {filter.get(), 0}, 1));
      if (fail_join) {
        auto const retained = consumer_test_access::retained_cached_storage(faults);
        REQUIRE(retained);
        REQUIRE(retained->has_source);
        REQUIRE(retained->filter_owners == 1);
        REQUIRE(retained->has_mask);
        REQUIRE(retained->has_output);
        REQUIRE_FALSE(retained->has_counters);
        REQUIRE_FALSE(retained->completed);
        // The positions ride the payload's gather.
        REQUIRE(retained->has_row_ids);
      } else {
        REQUIRE_FALSE(faults.retained.has_value());
        faults.fail_at.reset();
        auto retry = run_cached(consumer, input, bindings, layout, stream, nullptr, true);
        REQUIRE(retry.take_output()->num_rows() == 40);
        REQUIRE(retry.take_row_positions()->size() == 40);
        auto receipt = consumer.finish(std::move(retry), 7);
        REQUIRE(receipt.applied.size() == 1);
      }
    }
    REQUIRE(source_lifetime.expired() == !fail_join);
    REQUIRE(filter_lifetime.expired() == !fail_join);
  }
  REQUIRE(source_lifetime.expired());
  REQUIRE(filter_lifetime.expired());
}

TEST_CASE("cached checkpoint retains its owners when its count readback join fails",
          "[dynamic_filter][scan_merge][cached_checkpoint][consumer_retirement]")
{
  auto const fused          = GENERATE(false, true);
  ::cuda::stream_ref stream = cudf::get_default_stream();
  std::weak_ptr<cudf::table const> source_lifetime;
  std::weak_ptr<sirius_dynamic_filter const> filter_lifetime;
  {
    // No checkpoint fails, so the readback's success-path join is the one that fails.
    read_retirement_test_hooks faults{std::nullopt, true};
    {
      auto input    = make_cached_residual_input(100, 80, stream);
      auto channel  = std::make_shared<sirius_dynamic_filter_set>();
      auto producer = channel->register_producer({0});
      auto first    = make_in_list_prefix(40, stream);
      REQUIRE(producer.push_filter(0, first));
      REQUIRE(producer.push_filter(0, make_in_list_prefix(10, stream)));
      source_lifetime = input;
      filter_lifetime = first;
      dynamic_filter_consumer consumer(
        channel, {.force_counts = true, .fused_membership = fused}, {});
      std::vector<binding> bindings{{0, 0}};
      auto layout = cached_layout(true);
      REQUIRE_THROWS_AS(run_cached(consumer, input, bindings, layout, stream, nullptr, true),
                        sirius::unjoined_gpu_work);
      auto& gate = consumer_test_access::gate(consumer);
      REQUIRE(gate_test_access::needs_combined_sample(gate, channel->snapshot()));
      REQUIRE_FALSE(gate_test_access::filter_keep_ratio(gate, {first.get(), 0}, 2));
      auto const retained = consumer_test_access::retained_cached_storage(faults);
      REQUIRE(retained);
      REQUIRE(retained->has_source);
      REQUIRE(retained->filter_owners == 2);
      REQUIRE(retained->has_output);
      REQUIRE(retained->has_row_ids);
      REQUIRE(retained->has_counters);
      REQUIRE_FALSE(retained->completed);
      REQUIRE(std::vector<cudf::size_type>(retained->copied_counts.begin(),
                                           retained->copied_counts.end()) ==
              std::vector<cudf::size_type>{80, 40, 10});
    }
    REQUIRE_FALSE(source_lifetime.expired());
    REQUIRE_FALSE(filter_lifetime.expired());
  }
  REQUIRE(source_lifetime.expired());
  REQUIRE(filter_lifetime.expired());
}

TEST_CASE("cached checkpoint allocates no device counters when nothing reads them",
          "[dynamic_filter][scan_merge][cached_checkpoint][consumer_counts]")
{
  auto const fused                         = GENERATE(false, true);
  ::cuda::stream_ref stream                = cudf::get_default_stream();
  std::shared_ptr<cudf::table const> input = make_int64_sequence_table(100, stream);
  auto channel                             = std::make_shared<sirius_dynamic_filter_set>();
  auto producer                            = channel->register_producer({0});
  auto first                               = make_in_list_prefix(40, stream);
  auto second                              = make_in_list_prefix(10, stream);
  REQUIRE(producer.push_filter(0, first));
  REQUIRE(producer.push_filter(0, second));
  dynamic_filter_consumer consumer(channel, {.fused_membership = fused}, {});
  // Every marginal is current and the gate is active, so no step samples.
  auto& gate = consumer_test_access::gate(consumer);
  gate.record_keep_ratio(100, 10, 2);
  gate.record_filter_keep_ratio({first.get(), 0}, 0.4, 2);
  gate.record_filter_keep_ratio({second.get(), 0}, 0.25, 2);
  std::vector<binding> bindings{{0, 0}};
  auto layout = cached_layout();
  {
    read_retirement_test_hooks faults{read_checkpoint::MASK_FOLD, true};
    REQUIRE_THROWS_AS(run_cached(consumer, input, bindings, layout, stream),
                      sirius::unjoined_gpu_work);
    auto const retained = consumer_test_access::retained_cached_storage(faults);
    REQUIRE(retained);
    REQUIRE_FALSE(retained->has_counters);
    REQUIRE(retained->fused_rounds == (fused ? 1 : 0));
  }
  auto result = run_cached(consumer, input, bindings, layout, stream);
  REQUIRE_FALSE(result.counts_read_back());
  REQUIRE(result.prefix_counts().empty());
  REQUIRE(result.take_output()->num_rows() == 10);
  auto receipt = consumer.finish(std::move(result), 1);
  REQUIRE(receipt.applied.size() == 2);
}

TEST_CASE("cached residual checkpoint holds one probe mask beside its conjunction",
          "[dynamic_filter][scan_merge][cached_checkpoint]")
{
  // The residual keeps 1/64 of the rows and two membership filters halve them twice. The residual's
  // column becomes the conjunction and each probe mask is freed after its fold, so the peak holds
  // two row-sized BOOL8 buffers, not three.
  constexpr cudf::size_type kRows = cudf::size_type{1} << 22;
  ::cuda::stream_ref stream       = cudf::get_default_stream();
  auto input                      = make_cached_residual_input(kRows, kRows / 64, stream);
  auto channel                    = std::make_shared<sirius_dynamic_filter_set>();
  auto producer                   = channel->register_producer({0});
  REQUIRE(producer.push_filter(0, make_in_list_prefix(kRows / 128, stream)));
  REQUIRE(producer.push_filter(0, make_in_list_prefix(kRows / 256, stream)));
  dynamic_filter_consumer consumer(channel, {.force_counts = true, .fused_membership = false}, {});
  std::vector<binding> bindings{{0, 0}};
  auto layout = cached_layout(true);
  auto program =
    consumer.prepare({.source = input->view(), .bindings = bindings, .cached_layout = &layout});
  REQUIRE(program);
  rmm::mr::statistics_resource_adaptor allocations{
    cuda::mr::any_resource<cuda::mr::device_accessible>{cudf::get_current_device_resource_ref()}};
  auto result = consumer.apply(std::move(*program),
                               owning_table_view{input, input->view()},
                               stream,
                               rmm::device_async_resource_ref{allocations});
  REQUIRE(counts_of(result) == std::vector<cudf::size_type>{kRows / 64, kRows / 128, kRows / 256});
  auto const output = result.take_output();
  REQUIRE(output->num_rows() == kRows / 256);
  auto const mask_bytes = static_cast<std::size_t>(kRows);
  auto const peak       = static_cast<std::size_t>(allocations.get_bytes_counter().peak);
  CHECK(peak >= 2 * mask_bytes);
  // Counters and the probe's own scratch fit in the slack.
  CHECK(peak <= 2 * mask_bytes + output->alloc_size() + (std::size_t{1} << 20));
  (void)consumer.finish(std::move(result), 1);
}

TEST_CASE("cached sliced nullable residual and key masks preserve nonconsecutive positions",
          "[dynamic_filter][scan_merge][cached_checkpoint]")
{
  auto const fused          = GENERATE(false, true);
  ::cuda::stream_ref stream = cudf::get_default_stream();
  auto const mr             = cudf::get_current_device_resource_ref();
  auto key_table =
    make_values_table<int64_t>({5, 7, 35}, cudf::data_type{cudf::type_id::INT64}, stream);
  auto filter = std::make_shared<sirius::op::sirius_dynamic_in_list_filter>(
    key_table->view().column(0), stream, mr);
  std::vector<uint8_t> keep(100, 0);
  std::fill_n(keep.begin(), 40, uint8_t{1});
  auto mask_table = make_values_table(keep, cudf::data_type{cudf::type_id::BOOL8}, stream);
  auto columns    = mask_table->release();
  auto nulls      = cudf::create_null_mask(100, cudf::mask_state::ALL_VALID, stream, mr);
  cudf::set_null_mask(static_cast<cudf::bitmask_type*>(nulls.data()), 7, 8, false, stream);
  columns.front()->set_null_mask(std::move(nulls), 1);
  columns.insert(columns.begin(), make_sequence_column<int64_t>(100, stream));
  std::shared_ptr<cudf::table const> input = std::make_shared<cudf::table>(std::move(columns));
  auto slices   = cudf::slice(input->view(), std::vector<cudf::size_type>{5, 55});
  auto view     = slices.front();
  auto channel  = std::make_shared<sirius_dynamic_filter_set>();
  auto producer = channel->register_producer({0});
  REQUIRE(producer.push_filter(0, filter));
  dynamic_filter_consumer consumer(channel, {.fused_membership = fused}, {});
  auto layout = cached_layout(true);
  std::vector<binding> bindings{{0, 0}};
  auto program = consumer.prepare(
    {.source = view, .bindings = bindings, .cached_layout = &layout, .want_row_positions = true});
  REQUIRE(program);
  auto result = consumer.apply(std::move(*program), owning_table_view{input, view}, stream, mr);
  REQUIRE(counts_of(result) == std::vector<cudf::size_type>{34, 2});
  auto output    = result.take_output();
  auto positions = result.take_row_positions();
  REQUIRE(to_host_int64(output->view().column(0), stream) == std::vector<int64_t>{5, 35});
  REQUIRE(to_host_int32(positions->view(), stream) == std::vector<int32_t>{0, 30});
  (void)consumer.finish(std::move(result), 1);
}

TEST_CASE("cached dead residual prefix does not claim membership coverage",
          "[dynamic_filter][scan_merge][cached_checkpoint][consumer_receipt]")
{
  auto const fused          = GENERATE(false, true);
  ::cuda::stream_ref stream = cudf::get_default_stream();
  auto input                = make_cached_residual_input(100, 0, stream);
  auto channel              = std::make_shared<sirius_dynamic_filter_set>();
  auto producer             = channel->register_producer({0});
  auto first                = make_in_list_prefix(40, stream);
  auto second               = make_in_list_prefix(10, stream);
  REQUIRE(producer.push_filter(0, first));
  REQUIRE(producer.push_filter(0, second));
  dynamic_filter_consumer consumer(channel, {.fused_membership = fused}, {});
  auto layout = cached_layout(true);
  std::vector<binding> bindings{{0, 0}};
  auto result = run_cached(consumer, input, bindings, layout, stream, nullptr, true);
  REQUIRE(counts_of(result) == std::vector<cudf::size_type>{0, 0, 0});
  REQUIRE(result.take_output()->num_rows() == 0);
  REQUIRE(result.take_row_positions()->size() == 0);
  auto receipt = consumer.finish(std::move(result), 6);
  REQUIRE(receipt.applied.empty());
  auto& gate = consumer_test_access::gate(consumer);
  REQUIRE_FALSE(gate_test_access::filter_keep_ratio(gate, {first.get(), 0}, 2));
  REQUIRE_FALSE(gate_test_access::filter_keep_ratio(gate, {second.get(), 0}, 2));
}

TEST_CASE("ordinary Parquet residual retirement protects source without a consumer",
          "[dynamic_filter][scan_merge][cached_checkpoint][consumer_retirement]")
{
  // The evaluator's selection submits the residual and the payload gather together.
  auto const point     = GENERATE(read_checkpoint::PAYLOAD_GATHER, read_checkpoint::READER_EVENT);
  auto const fail_join = GENERATE(false, true);
  // A deferral asks the selection for survivor positions as well.
  auto const with_survivors = GENERATE(false, true);
  CAPTURE(point, fail_join, with_survivors);
  ::cuda::stream_ref stream = cudf::get_default_stream();
  auto manager              = sirius::test::operator_utils::initialize_memory_manager();
  auto* space               = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  auto info                 = std::make_unique<sirius::op::scan::parquet_ingestible_table_info>();
  info->names               = {"key", "predicate"};
  info->returned_types      = {sirius::logical_type::make(sirius::type_id::BIGINT),
                               sirius::logical_type::make(sirius::type_id::BIGINT)};
  info->column_ids          = {duckdb::ColumnIndex(0), duckdb::ColumnIndex(1)};
  info->projection_ids      = {0, 1};
  info->projected_output_count = 1;
  info->table_filters          = duckdb::make_uniq<duckdb::TableFilterSet>();
  info->table_filters->PushFilter(
    duckdb::ProjectionIndex(1),
    sirius::test::constant_filter(duckdb::ExpressionType::COMPARE_LESSTHAN,
                                  duckdb::Value::BIGINT(40)));
  auto ingestible = sirius::op::scan::make_ingestible(std::move(info));
  std::weak_ptr<cudf::table const> source_lifetime;
  {
    read_retirement_test_hooks faults{point, fail_join};
    {
      std::shared_ptr<cudf::table const> input =
        make_sequence_table<int64_t>(100, stream, {{0, 1}, {0, 1}});
      source_lifetime = input;
      std::unique_ptr<cudf::column> survivors;
      auto apply = [&] {
        sirius::op::scan::filtered_table source{owning_table_view{input, input->view()},
                                                sirius::op::scan::filter_state::UNFILTERED};
        return ingestible->post_filter_and_project(std::move(source),
                                                   *space,
                                                   stream,
                                                   false,
                                                   nullptr,
                                                   with_survivors ? &survivors : nullptr,
                                                   {});
      };
      if (fail_join) {
        REQUIRE_THROWS_AS(apply(), sirius::unjoined_gpu_work);
        REQUIRE(faults.retained.has_value());
      } else {
        REQUIRE_THROWS_AS(apply(), rmm::out_of_memory);
        REQUIRE_FALSE(faults.retained.has_value());
        faults.fail_at.reset();
        auto output = apply();
        REQUIRE(output->num_rows() == 40);
        REQUIRE(output->num_columns() == 1);
        REQUIRE(to_host_int64(output->view().column(0), stream).back() == 39);
        REQUIRE((survivors != nullptr) == with_survivors);
        if (with_survivors) {
          std::vector<int32_t> expected(40);
          std::iota(expected.begin(), expected.end(), 0);
          REQUIRE(to_host_int32(survivors->view(), stream) == expected);
        }
      }
    }
    REQUIRE(source_lifetime.expired() == !fail_join);
  }
  REQUIRE(source_lifetime.expired());
}

namespace {
/// Allocation tally shared by the copies of one counting_resource
struct allocation_tally {
  rmm::device_async_resource_ref upstream;
  std::size_t calls   = 0;
  std::size_t live    = 0;
  std::size_t fail_at = 0;  ///< The call to refuse with rmm::out_of_memory; zero refuses none
};

/// Device resource that counts its allocations and refuses the one `allocation_tally::fail_at`
/// names
struct counting_resource {
  std::shared_ptr<allocation_tally> tally;

  void* allocate_sync(std::size_t bytes, std::size_t alignment)
  {
    return tally->upstream.allocate_sync(bytes, alignment);
  }
  void deallocate_sync(void* pointer, std::size_t bytes, std::size_t alignment) noexcept
  {
    tally->upstream.deallocate_sync(pointer, bytes, alignment);
  }
  void* allocate(::cuda::stream_ref stream, std::size_t bytes, std::size_t alignment)
  {
    if (++tally->calls == tally->fail_at) {
      throw rmm::out_of_memory{"injected allocation failure"};
    }
    auto* pointer = tally->upstream.allocate(stream, bytes, alignment);
    ++tally->live;
    return pointer;
  }
  void deallocate(::cuda::stream_ref stream,
                  void* pointer,
                  std::size_t bytes,
                  std::size_t alignment) noexcept
  {
    --tally->live;
    tally->upstream.deallocate(stream, pointer, bytes, alignment);
  }
  friend void get_property(counting_resource const&, ::cuda::mr::device_accessible) noexcept {}
  bool operator==(counting_resource const& other) const noexcept { return tally == other.tally; }
};
}  // namespace

TEST_CASE("ordinary Parquet projection copies each repeat of a borrowed source before release",
          "[dynamic_filter][scan_merge][cached_checkpoint][consumer_retirement]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();
  auto manager              = sirius::test::operator_utils::initialize_memory_manager();
  auto* const gpu           = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  auto tally = std::make_shared<allocation_tally>(allocation_tally{gpu->get_default_allocator()});
  cucascade::memory::memory_space space(cucascade::memory::gpu_memory_space_config{
    .device_id       = gpu->get_device_id(),
    .memory_capacity = gpu->get_max_memory(),
    .mr_factory_fn   = [&](int, std::size_t) {
      return cuda::mr::any_resource<cuda::mr::device_accessible>{counting_resource{tally}};
    }});
  // One materialized column, output three times: the second and third outputs are repeats.
  auto info            = std::make_unique<sirius::op::scan::parquet_ingestible_table_info>();
  info->names          = {"key"};
  info->returned_types = {sirius::logical_type::make(sirius::type_id::BIGINT)};
  info->column_ids     = {duckdb::ColumnIndex(0)};
  info->projection_ids = {0, 0, 0};
  info->projected_output_count = 3;
  auto ingestible              = sirius::op::scan::make_ingestible(std::move(info));
  std::vector<int64_t> expected(100);
  std::iota(expected.begin(), expected.end(), 0);
  auto project = [&](std::shared_ptr<cudf::table const> const& input) {
    sirius::op::scan::filtered_table source{owning_table_view{input, input->view()},
                                            sirius::op::scan::filter_state::UNFILTERED};
    return ingestible->post_filter_and_project(
      std::move(source), space, stream, false, nullptr, nullptr, {});
  };

  SECTION("every output is an owned copy and the borrowed source is unchanged")
  {
    std::shared_ptr<cudf::table const> input = make_sequence_table<int64_t>(100, stream);
    auto output                              = project(input);
    // Two repeat copies, then the copy that releases the borrowed column itself.
    CHECK(tally->calls == 3);
    REQUIRE(output->num_columns() == 3);
    auto const view = output->view();
    for (cudf::size_type column = 0; column < view.num_columns(); ++column) {
      CAPTURE(column);
      REQUIRE(to_host_int64(view.column(column), stream) == expected);
      CHECK(view.column(column).head() != input->view().column(0).head());
    }
    CHECK(view.column(0).head() != view.column(1).head());
    CHECK(view.column(1).head() != view.column(2).head());
    CHECK(view.column(0).head() != view.column(2).head());
    REQUIRE(to_host_int64(input->view().column(0), stream) == expected);
  }
  SECTION("a failed repeat copy retires the borrowed source it read")
  {
    auto const fail_join = GENERATE(false, true);
    CAPTURE(fail_join);
    tally->fail_at = 2;  // the second repeat, after the first repeat's copy was submitted
    std::weak_ptr<cudf::table const> source_lifetime;
    {
      read_retirement_test_hooks faults{std::nullopt, fail_join};
      {
        std::shared_ptr<cudf::table const> input = make_sequence_table<int64_t>(100, stream);
        source_lifetime                          = input;
        if (fail_join) {
          REQUIRE_THROWS_AS(project(input), sirius::unjoined_gpu_work);
          REQUIRE(faults.retained.has_value());
        } else {
          REQUIRE_THROWS_AS(project(input), rmm::out_of_memory);
          REQUIRE_FALSE(faults.retained.has_value());
        }
        CHECK(tally->calls == 2);
        // The submitted repeat copy is freed in stream order; only the source may stay retained.
        CHECK(tally->live == 0);
      }
      REQUIRE(source_lifetime.expired() == !fail_join);
    }
    REQUIRE(source_lifetime.expired());
  }
}

TEST_CASE(
  "cached alias selection uses the eligible representative without reviving a skipped alias",
  "[dynamic_filter][scan_merge][cached_checkpoint][consumer_alias]")
{
  ::cuda::stream_ref stream                = cudf::get_default_stream();
  std::shared_ptr<cudf::table const> input = make_int64_sequence_table(10, stream);
  auto channel                             = std::make_shared<sirius_dynamic_filter_set>();
  auto producer                            = channel->register_producer({0, 1});
  auto alias       = std::make_shared<counting_in_list_filter>(make_in_list_prefix(4, stream));
  auto intervening = make_in_list_prefix(6, stream);
  REQUIRE(producer.push_filter(0, alias));
  REQUIRE(producer.push_filter(1, alias));
  auto const skipped  = channel->snapshot().entries().front().column_index;
  auto const eligible = 1 - skipped;
  REQUIRE(producer.push_filter(skipped, intervening));
  dynamic_filter_consumer consumer(channel, {.force_counts = true}, {});
  auto& gate = consumer_test_access::gate(consumer);
  gate.record_keep_ratio(10, 4, 3);
  gate.record_filter_keep_ratio({alias.get(), skipped}, 0.8, 3);
  std::vector<binding> bindings{{0, 0}, {1, 0}};
  auto layout = cached_layout();
  auto result = run_cached(consumer, input, bindings, layout, stream);
  REQUIRE(counts_of(result) == std::vector<cudf::size_type>{10, 6, 4});
  REQUIRE(alias->mask_calls() == 1);
  auto receipt = consumer.finish(std::move(result), 8);
  REQUIRE(receipt.applied ==
          std::vector<sirius::op::scan::applied_entry>{{intervening, skipped}, {alias, eligible}});
  REQUIRE(gate_test_access::filter_keep_ratio(gate, {alias.get(), skipped}, 3) == Approx(0.8));
  REQUIRE(gate_test_access::filter_keep_ratio(gate, {alias.get(), eligible}, 3) ==
          Approx(4.0 / 6.0));
}

TEST_CASE("cached projected nullable string and nested payload matches manual survivor rows",
          "[dynamic_filter][scan_merge][cached_checkpoint]")
{
  // A lone filter without a residual gathers through its own mask; a second filter that keeps
  // every row makes the checkpoint form a conjunction, fused or ordinary.
  std::size_t filters = 1;
  bool fused          = true;
  SECTION("lone filter mask") {}
  SECTION("fused conjunction") { filters = 2; }
  SECTION("ordinary conjunction")
  {
    filters = 2;
    fused   = false;
  }
  ::cuda::stream_ref stream = cudf::get_default_stream();
  auto const mr             = cudf::get_current_device_resource_ref();
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(make_sequence_column<int64_t>(10, stream));
  auto strings = sirius::test::operator_utils::make_string_column(
    {"pad", "alpha", "b", "", "delta", "ee", "null", "gggg", "omega", "tail"}, stream, mr);
  auto string_nulls = cudf::create_null_mask(10, cudf::mask_state::ALL_VALID, stream, mr);
  cudf::set_null_mask(static_cast<cudf::bitmask_type*>(string_nulls.data()), 6, 7, false, stream);
  strings->set_null_mask(std::move(string_nulls), 1);
  columns.push_back(std::move(strings));
  columns.push_back(make_nullable_list_payload(10, stream));
  std::shared_ptr<cudf::table const> input = std::make_shared<cudf::table>(std::move(columns));
  auto slices = cudf::slice(input->view(), std::vector<cudf::size_type>{1, 9});
  auto view   = slices.front();
  auto keys =
    make_values_table<int64_t>({2, 3, 6, 8}, cudf::data_type{cudf::type_id::INT64}, stream);
  auto filter =
    std::make_shared<sirius::op::sirius_dynamic_in_list_filter>(keys->view().column(0), stream, mr);
  auto channel  = std::make_shared<sirius_dynamic_filter_set>();
  auto producer = channel->register_producer({0});
  REQUIRE(producer.push_filter(0, filter));
  if (filters == 2) { REQUIRE(producer.push_filter(0, make_in_list_prefix(10, stream))); }
  dynamic_filter_consumer consumer(channel, {.fused_membership = fused}, {});
  std::vector<binding> bindings{{0, 0}};
  auto layout              = cached_layout();
  layout.output_columns    = {0, 1, 2};
  layout.projected_columns = {2, 1, 0, 1};
  auto program             = consumer.prepare(
    {.source = view, .bindings = bindings, .cached_layout = &layout, .want_row_positions = true});
  REQUIRE(program);
  auto result = consumer.apply(std::move(*program), owning_table_view{input, view}, stream, mr);
  REQUIRE(consumer_test_access::storage(result).fused_rounds == (fused && filters == 2 ? 1 : 0));
  auto output    = result.take_output();
  auto positions = result.take_row_positions();
  REQUIRE(output->num_rows() == 4);
  REQUIRE(to_host_int64(output->view().column(2), stream) == std::vector<int64_t>{2, 3, 6, 8});
  REQUIRE(to_host_int32(positions->view(), stream) == std::vector<int32_t>{1, 2, 5, 7});
  REQUIRE(output->view().column(0).null_count() == 1);
  REQUIRE(output->view().column(1).null_count() == 1);
  REQUIRE(output->view().column(3).null_count() == 1);
  auto expected_indices =
    make_values_table<int32_t>({2, 3, 6, 8}, cudf::data_type{cudf::type_id::INT32}, stream);
  auto expected = cudf::gather(input->view().select(std::vector<cudf::size_type>{2, 1, 0, 1}),
                               expected_indices->view().column(0),
                               cudf::out_of_bounds_policy::DONT_CHECK,
                               cudf::negative_index_policy::NOT_ALLOWED,
                               stream,
                               mr);
  REQUIRE(sirius::test::expect_tables_equivalent_impl(output->view(), expected->view(), stream));
  auto receipt = consumer.finish(std::move(result), 11);
  REQUIRE(receipt.applied.size() == filters);
}

namespace {
using sirius::op::fused_membership_input;
using sirius::op::sirius_mask_applicable;

/// Copies @p values into a new device column of @p type without a null mask.
template <class T>
std::unique_ptr<cudf::column> make_device_column(std::vector<T> const& values,
                                                 cudf::data_type type,
                                                 ::cuda::stream_ref stream)
{
  auto column = cudf::make_fixed_width_column(
    type, static_cast<cudf::size_type>(values.size()), cudf::mask_state::UNALLOCATED, stream);
  if (!values.empty()) {
    RMM_CUDA_TRY(cudaMemcpyAsync(column->mutable_view().data<T>(),
                                 values.data(),
                                 values.size() * sizeof(T),
                                 cudaMemcpyHostToDevice,
                                 stream.get()));
    stream.sync();
  }
  return column;
}

template <class T>
std::vector<T> copy_to_host(T const* data, std::size_t count, ::cuda::stream_ref stream)
{
  std::vector<T> result(count);
  if (count != 0) {
    RMM_CUDA_TRY(cudaMemcpyAsync(
      result.data(), data, count * sizeof(T), cudaMemcpyDeviceToHost, stream.get()));
  }
  stream.sync();
  return result;
}

/// Builds a hash-set IN-list (@p kind 0), Bloom (1) or small IN-list (2) filter over @p keys.
std::unique_ptr<sirius::op::sirius_mask_applicable> make_membership_filter(
  int kind, cudf::column_view keys, ::cuda::stream_ref stream)
{
  auto const mr = cudf::get_current_device_resource_ref();
  switch (kind) {
    case 0: return std::make_unique<sirius::op::sirius_dynamic_in_list_filter>(keys, stream, mr);
    case 1: return std::make_unique<sirius::op::sirius_dynamic_bloom_filter>(keys, stream, mr);
    default:
      return std::make_unique<sirius::op::sirius_dynamic_small_in_list_filter>(keys, stream, mr);
  }
}

/// Runs `try_fused_membership_mask` on the current device and returns its submitted round count.
std::size_t fuse_membership(std::span<sirius::op::fused_membership_input const> steps,
                            cudf::mutable_column_view running,
                            std::optional<cudf::column_view> residual,
                            cudf::device_span<cudf::size_type> counts,
                            ::cuda::stream_ref stream)
{
  std::size_t rounds = 0;
  sirius::op::try_fused_membership_mask(
    steps, rmm::get_current_cuda_device().value(), running, residual, counts, rounds, stream);
  return rounds;
}

/// The BOOL8 values `filter.compute_mask` produces for @p probe on the current device.
std::vector<std::uint8_t> lone_mask(sirius::op::sirius_mask_applicable const& filter,
                                    cudf::column_view probe,
                                    ::cuda::stream_ref stream)
{
  auto const mask = filter.compute_mask(
    probe, rmm::get_current_cuda_device().value(), stream, cudf::get_current_device_resource_ref());
  return copy_to_host(
    mask->view().data<std::uint8_t>(), static_cast<std::size_t>(mask->size()), stream);
}

void set_single_null(cudf::column& column, cudf::size_type row, ::cuda::stream_ref stream)
{
  auto mask = cudf::create_null_mask(column.size(), cudf::mask_state::ALL_VALID, stream);
  cudf::set_null_mask(static_cast<cudf::bitmask_type*>(mask.data()), row, row + 1, false, stream);
  column.set_null_mask(std::move(mask), 1);
}

struct membership_result {
  std::vector<uint8_t> mask;
  std::vector<cudf::size_type> counts;
  std::size_t rounds;
};

/// Runs the fused conjunction of @p steps into a fresh running mask and reads back its results.
membership_result run_membership(std::span<fused_membership_input const> steps,
                                 std::optional<cudf::column_view> residual,
                                 bool count,
                                 ::cuda::stream_ref stream)
{
  auto running = cudf::make_numeric_column(cudf::data_type{cudf::type_id::BOOL8},
                                           steps.front().probe.size(),
                                           cudf::mask_state::UNALLOCATED,
                                           stream);
  rmm::device_uvector<cudf::size_type> counters(
    count ? steps.size() + static_cast<std::size_t>(bool(residual)) : 0, stream);
  if (!counters.is_empty()) {
    RMM_CUDA_TRY(
      cudaMemsetAsync(counters.data(), 0, counters.size() * sizeof(cudf::size_type), stream.get()));
  }
  auto const rounds = fuse_membership(
    steps, running->mutable_view(), residual, {counters.data(), counters.size()}, stream);
  return {copy_to_host(running->view().data<uint8_t>(), running->size(), stream),
          copy_to_host(counters.data(), counters.size(), stream),
          rounds};
}

class forwarding_mask_filter final : public sirius_mask_applicable {
 public:
  explicit forwarding_mask_filter(sirius_mask_applicable const& filter) : filter_{filter} {}

  bool prepare_fused_probe(cudf::column_view const& probe,
                           int device,
                           sirius::op::detail::fused_probe& out) const override
  {
    return filter_.prepare_fused_probe(probe, device, out);
  }

  std::unique_ptr<cudf::column> compute_mask(cudf::column_view const& probe,
                                             uint32_t const* prior,
                                             int device,
                                             ::cuda::stream_ref stream,
                                             rmm::device_async_resource_ref mr) const override
  {
    return filter_.compute_mask(probe, prior, device, stream, mr);
  }

 private:
  sirius_mask_applicable const& filter_;
};
}  // namespace

TEST_CASE("fused membership preserves ordered prefixes across rounds",
          "[dynamic_filter][fused_membership]")
{
  auto const filters        = GENERATE(1, 2, 4, 5, 9);
  auto const count          = GENERATE(false, true);
  auto const with_residual  = GENERATE(false, true);
  ::cuda::stream_ref stream = cudf::get_default_stream();
  auto probe                = make_sequence_column<int32_t>(1000, stream);
  std::vector<uint8_t> keep(1000, 0);
  std::fill_n(keep.begin(), 600, 1);
  auto residual = make_device_column(keep, cudf::data_type{cudf::type_id::BOOL8}, stream);
  std::optional<cudf::column_view> residual_view;
  if (with_residual) { residual_view = residual->view(); }
  std::array<int, 9> const limits{300, 120, 60, 30, 6, 0, 8, 2, 1};
  std::vector<std::shared_ptr<sirius::op::sirius_dynamic_in_list_filter>> owners;
  std::vector<fused_membership_input> steps;
  std::vector<cudf::size_type> expected;
  if (with_residual && count) { expected.push_back(600); }
  int prefix = 1000;
  for (int i = 0; i < filters; ++i) {
    owners.push_back(make_in_list_prefix(limits[i], stream));
    steps.push_back({owners.back().get(), probe->view()});
    prefix = std::min(prefix, limits[i]);
    if (count) { expected.push_back(prefix); }
  }
  // Every prefix is below the residual's 600 kept rows, so exactly the first `prefix` rows survive.
  std::vector<uint8_t> expected_mask(1000, 0);
  std::fill_n(expected_mask.begin(), prefix, uint8_t{1});
  auto const fused = run_membership(steps, residual_view, count, stream);
  REQUIRE(fused.rounds == static_cast<std::size_t>((filters + 3) / 4));
  REQUIRE(fused.mask == expected_mask);
  REQUIRE(fused.counts == expected);
}

TEST_CASE("fused membership handles empty and all-null hash-set and Bloom builds",
          "[dynamic_filter][fused_membership]")
{
  auto const kind           = GENERATE(0, 1);
  auto const all_null       = GENERATE(false, true);
  ::cuda::stream_ref stream = cudf::get_default_stream();
  CAPTURE(kind, all_null);
  auto keys =
    make_device_column<int64_t>(all_null ? std::vector<int64_t>{0, 2, 7} : std::vector<int64_t>{},
                                cudf::data_type{cudf::type_id::INT64},
                                stream);
  if (all_null) {
    keys->set_null_mask(cudf::create_null_mask(keys->size(), cudf::mask_state::ALL_NULL, stream),
                        keys->size());
  }
  auto filter = make_membership_filter(kind, keys->view(), stream);
  auto probe =
    make_device_column<int32_t>({-1, 0, 2, 7, 9}, cudf::data_type{cudf::type_id::INT32}, stream);
  auto residual =
    make_device_column<uint8_t>({1, 1, 0, 1, 1}, cudf::data_type{cudf::type_id::BOOL8}, stream);
  std::array steps{fused_membership_input{filter.get(), probe->view()}};
  auto const fused = run_membership(steps, residual->view(), true, stream);
  REQUIRE(fused.rounds == 1);
  REQUIRE(fused.mask == lone_mask(*filter, probe->view(), stream));
  REQUIRE(fused.mask == std::vector<uint8_t>(probe->size(), 0));
  REQUIRE(fused.counts == std::vector<cudf::size_type>{4, 0});
}

TEST_CASE("fused membership shares signed conversion and sentinel semantics",
          "[dynamic_filter][fused_membership]")
{
  auto const kind           = GENERATE(0, 1, 2);
  ::cuda::stream_ref stream = cudf::get_default_stream();
  auto keys   = make_device_column<int32_t>({0, 2}, cudf::data_type{cudf::type_id::INT32}, stream);
  auto filter = make_membership_filter(kind, keys->view(), stream);
  auto probe  = make_device_column<int64_t>({std::numeric_limits<int64_t>::min(),
                                             int64_t{std::numeric_limits<int32_t>::min()} - 1,
                                             std::numeric_limits<int32_t>::min(),
                                             0,
                                             2,
                                             int64_t{std::numeric_limits<int32_t>::max()} + 1},
                                           cudf::data_type{cudf::type_id::INT64},
                                           stream);
  std::array steps{fused_membership_input{filter.get(), probe->view()}};
  auto const fused = run_membership(steps, std::nullopt, true, stream);
  REQUIRE(fused.rounds == 1);
  REQUIRE(fused.mask == lone_mask(*filter, probe->view(), stream));
  REQUIRE(fused.counts == std::vector<cudf::size_type>{static_cast<cudf::size_type>(
                            std::count(fused.mask.begin(), fused.mask.end(), 1))});
  REQUIRE(fused.mask[0] == 0);
  REQUIRE(fused.mask[1] == 0);
  REQUIRE(fused.mask[3] == 1);
  REQUIRE(fused.mask[4] == 1);
  REQUIRE(fused.mask[5] == 0);
  if (kind == 0) { REQUIRE(fused.mask[2] == 1); }
  if (kind == 2) { REQUIRE(fused.mask[2] == 0); }
}

TEST_CASE("fused membership accepts compatible narrow temporal and decimal carriers",
          "[dynamic_filter][fused_membership]")
{
  auto const kind           = GENERATE(0, 1, 2);
  ::cuda::stream_ref stream = cudf::get_default_stream();
  auto const check          = [&]<class Build, class Probe>(
                       Build, Probe, cudf::data_type build_type, cudf::data_type probe_type) {
    auto keys   = make_device_column<Build>({-3, 2, 9}, build_type, stream);
    auto filter = make_membership_filter(kind, keys->view(), stream);
    auto probe  = make_device_column<Probe>({-3, 0, 2, 9, 11}, probe_type, stream);
    forwarding_mask_filter forwarded{*filter};
    std::array steps{fused_membership_input{&forwarded, probe->view()}};
    auto const fused = run_membership(steps, std::nullopt, true, stream);
    REQUIRE(fused.rounds == 1);
    REQUIRE(fused.mask == lone_mask(forwarded, probe->view(), stream));
    REQUIRE(fused.counts == std::vector<cudf::size_type>{static_cast<cudf::size_type>(
                              std::count(fused.mask.begin(), fused.mask.end(), 1))});
    REQUIRE(fused.mask[0] == 1);
    REQUIRE(fused.mask[2] == 1);
    REQUIRE(fused.mask[3] == 1);
  };
  check(int64_t{},
        int8_t{},
        cudf::data_type{cudf::type_id::INT64},
        cudf::data_type{cudf::type_id::INT8});
  check(int64_t{},
        int16_t{},
        cudf::data_type{cudf::type_id::INT64},
        cudf::data_type{cudf::type_id::INT16});
  check(int64_t{},
        int32_t{},
        cudf::data_type{cudf::type_id::INT64},
        cudf::data_type{cudf::type_id::INT32});
  check(int32_t{},
        int16_t{},
        cudf::data_type{cudf::type_id::TIMESTAMP_DAYS},
        cudf::data_type{cudf::type_id::INT16});
  check(int64_t{},
        int64_t{},
        cudf::data_type{cudf::type_id::TIMESTAMP_MILLISECONDS},
        cudf::data_type{cudf::type_id::TIMESTAMP_MILLISECONDS});
  check(int32_t{},
        int64_t{},
        cudf::data_type{cudf::type_id::DECIMAL32, -2},
        cudf::data_type{cudf::type_id::DECIMAL64, -2});
  check(int64_t{},
        int32_t{},
        cudf::data_type{cudf::type_id::DECIMAL64, -2},
        cudf::data_type{cudf::type_id::DECIMAL32, -2});
}

TEST_CASE("fused membership honors sliced validity while aliasing residual values",
          "[dynamic_filter][fused_membership]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();
  auto keys =
    make_device_column<int64_t>({2, 3, 4, 5, 6, 7}, cudf::data_type{cudf::type_id::INT64}, stream);
  auto bloom   = make_membership_filter(1, keys->view(), stream);
  auto needles = make_membership_filter(2, keys->view(), stream);
  auto set     = make_membership_filter(0, keys->view(), stream);
  auto probe   = make_sequence_column<int16_t>(10, stream);
  set_single_null(*probe, 3, stream);
  auto residual = make_device_column<uint8_t>(
    {1, 1, 1, 1, 0, 1, 1, 1, 1, 1}, cudf::data_type{cudf::type_id::BOOL8}, stream);
  set_single_null(*residual, 6, stream);
  auto probe_view    = cudf::slice(probe->view(), {1, 9}, stream).front();
  auto residual_view = cudf::slice(residual->view(), {1, 9}, stream).front();
  std::array steps{fused_membership_input{needles.get(), probe_view},
                   fused_membership_input{bloom.get(), probe_view},
                   fused_membership_input{set.get(), probe_view},
                   fused_membership_input{bloom.get(), probe_view},
                   fused_membership_input{needles.get(), probe_view}};
  cudf::mutable_column_view alias{residual_view.type(),
                                  residual_view.size(),
                                  residual->mutable_view().head<bool>(),
                                  nullptr,
                                  0,
                                  1};
  rmm::device_uvector<cudf::size_type> counts(6, stream);
  RMM_CUDA_TRY(
    cudaMemsetAsync(counts.data(), 0, counts.size() * sizeof(cudf::size_type), stream.get()));
  REQUIRE(fuse_membership(steps, alias, residual_view, {counts.data(), counts.size()}, stream) ==
          2);
  // Sliced rows 1..8: probe row 3 is null, residual row 4 is false and row 6 null, and only
  // probe keys 2..7 are members.
  REQUIRE(copy_to_host(alias.data<uint8_t>(), alias.size(), stream) ==
          std::vector<uint8_t>{0, 1, 0, 0, 1, 0, 1, 0});
  REQUIRE(copy_to_host(counts.data(), counts.size(), stream) ==
          std::vector<cudf::size_type>{6, 3, 3, 3, 3, 3});
  REQUIRE(residual->view().null_count() == 1);
}

TEST_CASE("fused membership preflights late refusals before changing device buffers",
          "[dynamic_filter][fused_membership]")
{
  auto const size           = GENERATE(5, 9);
  ::cuda::stream_ref stream = cudf::get_default_stream();
  auto probe                = make_sequence_column<int32_t>(8, stream);
  auto filter               = make_in_list_prefix(4, stream);
  counting_in_list_filter refusing{filter};
  std::vector<fused_membership_input> steps(size, {filter.get(), probe->view()});
  steps.back().filter = &refusing;
  auto running        = make_device_column<uint8_t>(
    std::vector<uint8_t>(8, 1), cudf::data_type{cudf::type_id::BOOL8}, stream);
  set_single_null(*running, 2, stream);
  auto const before_mask = copy_to_host(running->view().null_mask(), 1, stream);
  cudf::mutable_column_view facade{
    running->type(), running->size(), running->mutable_view().head<bool>(), nullptr, 0};
  rmm::device_uvector<cudf::size_type> counts(size + 1, stream);
  RMM_CUDA_TRY(
    cudaMemsetAsync(counts.data(), 0x2a, counts.size() * sizeof(cudf::size_type), stream.get()));
  auto const before_counts = copy_to_host(counts.data(), counts.size(), stream);
  REQUIRE(fuse_membership(steps, facade, running->view(), {counts.data(), counts.size()}, stream) ==
          0);
  REQUIRE(copy_to_host(facade.data<uint8_t>(), 8, stream) == std::vector<uint8_t>(8, 1));
  REQUIRE(copy_to_host(running->view().null_mask(), 1, stream) == before_mask);
  REQUIRE(copy_to_host(counts.data(), counts.size(), stream) == before_counts);
  REQUIRE(refusing.mask_calls() == 0);
}

TEST_CASE("fused membership rejects incompatible domains through forwarding wrappers",
          "[dynamic_filter][fused_membership]")
{
  auto const kind           = GENERATE(0, 1, 2);
  ::cuda::stream_ref stream = cudf::get_default_stream();
  auto const check          = [&]<class Build, class Probe>(
                       Build, Probe, cudf::data_type build_type, cudf::data_type probe_type) {
    auto keys   = make_device_column<Build>({2, 3}, build_type, stream);
    auto filter = make_membership_filter(kind, keys->view(), stream);
    auto probe  = make_device_column<Probe>({2, 3}, probe_type, stream);
    forwarding_mask_filter forwarded{*filter};
    std::array steps{fused_membership_input{&forwarded, probe->view()}};
    auto running =
      make_device_column<uint8_t>({1, 0}, cudf::data_type{cudf::type_id::BOOL8}, stream);
    REQUIRE(fuse_membership(steps, running->mutable_view(), std::nullopt, {}, stream) == 0);
    REQUIRE(copy_to_host(running->view().data<uint8_t>(), 2, stream) == std::vector<uint8_t>{1, 0});
  };
  check(int32_t{},
        int64_t{},
        cudf::data_type{cudf::type_id::DECIMAL32, -2},
        cudf::data_type{cudf::type_id::DECIMAL64, -3});
  check(int64_t{},
        int64_t{},
        cudf::data_type{cudf::type_id::TIMESTAMP_MILLISECONDS},
        cudf::data_type{cudf::type_id::TIMESTAMP_SECONDS});
  check(int32_t{},
        int32_t{},
        cudf::data_type{cudf::type_id::INT32},
        cudf::data_type{cudf::type_id::TIMESTAMP_DAYS});
  check(uint32_t{},
        uint32_t{},
        cudf::data_type{cudf::type_id::UINT32},
        cudf::data_type{cudf::type_id::UINT32});
  check(int64_t{},
        __int128{},
        cudf::data_type{cudf::type_id::DECIMAL64, -2},
        cudf::data_type{cudf::type_id::DECIMAL128, -2});
}

TEST_CASE("fused membership empty inputs submit no rounds", "[dynamic_filter][fused_membership]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();
  auto probe                = make_sequence_column<int32_t>(0, stream);
  auto filter               = make_in_list_prefix(1, stream);
  std::array steps{fused_membership_input{filter.get(), probe->view()}};
  auto running = make_device_column<uint8_t>({}, cudf::data_type{cudf::type_id::BOOL8}, stream);
  REQUIRE(fuse_membership(steps, running->mutable_view(), std::nullopt, {}, stream) == 0);
  auto nonempty =
    make_device_column<uint8_t>({1, 0}, cudf::data_type{cudf::type_id::BOOL8}, stream);
  REQUIRE(fuse_membership({}, nonempty->mutable_view(), std::nullopt, {}, stream) == 0);
  REQUIRE(copy_to_host(nonempty->view().data<uint8_t>(), 2, stream) == std::vector<uint8_t>{1, 0});
}

namespace {
struct cached_pair_result {
  std::vector<cudf::size_type> counts;
  batch_receipt receipt;
  cudf::size_type rows;
};

cached_pair_result require_cached_pair(std::shared_ptr<cudf::table const> const& input,
                                       cudf::table_view view,
                                       std::shared_ptr<sirius_dynamic_filter_set> const& channel,
                                       std::span<binding const> bindings,
                                       cached_view_layout const& layout,
                                       bool counts,
                                       std::size_t expected_rounds,
                                       ::cuda::stream_ref stream,
                                       batch_receipt const* prior = nullptr)
{
  // Unknown decode history suppresses sampling in the counts-off pair without changing selection.
  std::vector<binding> decode =
    counts ? std::vector<binding>{} : std::vector<binding>{bindings.begin(), bindings.end()};
  dynamic_filter_consumer ordinary(
    channel, {.force_counts = counts, .fused_membership = false}, decode);
  dynamic_filter_consumer fused(
    channel, {.force_counts = counts, .fused_membership = true}, decode);
  sirius::op::scan::application_input request{.source             = view,
                                              .bindings           = bindings,
                                              .prior              = prior,
                                              .cached_layout      = &layout,
                                              .want_row_positions = true};
  auto left  = ordinary.prepare(request);
  auto right = fused.prepare(request);
  REQUIRE(left);
  REQUIRE(right);
  REQUIRE(consumer_test_access::same_selection(*left, *right));
  auto const mr = cudf::get_current_device_resource_ref();
  auto a        = ordinary.apply(std::move(*left), owning_table_view{input, view}, stream, mr);
  auto b        = fused.apply(std::move(*right), owning_table_view{input, view}, stream, mr);
  REQUIRE(consumer_test_access::storage(a).fused_rounds == 0);
  REQUIRE(consumer_test_access::storage(b).fused_rounds == expected_rounds);
  REQUIRE(counts_of(a) == counts_of(b));
  REQUIRE(a.counts_read_back() == b.counts_read_back());
  auto output_a = a.take_output();
  auto output_b = b.take_output();
  REQUIRE(output_a);
  REQUIRE(output_b);
  REQUIRE(sirius::test::expect_tables_equivalent_impl(output_a->view(), output_b->view(), stream));
  auto positions_a = a.take_row_positions();
  auto positions_b = b.take_row_positions();
  REQUIRE(positions_a);
  REQUIRE(positions_b);
  REQUIRE(to_host_int32(positions_a->view(), stream) == to_host_int32(positions_b->view(), stream));
  auto const prefixes   = counts_of(b);
  auto const snapshot   = channel->snapshot();
  auto const generation = snapshot.generation();
  for (auto const& entry : snapshot.entries()) {
    REQUIRE_FALSE(gate_test_access::filter_keep_ratio(
      consumer_test_access::gate(ordinary), {entry.filter.get(), entry.column_index}, generation));
    REQUIRE_FALSE(gate_test_access::filter_keep_ratio(
      consumer_test_access::gate(fused), {entry.filter.get(), entry.column_index}, generation));
  }
  auto receipt_a = ordinary.finish(std::move(a), 31);
  auto receipt_b = fused.finish(std::move(b), 31);
  REQUIRE(receipt_a.applied == receipt_b.applied);
  REQUIRE(receipt_a.decode_attached == receipt_b.decode_attached);
  for (auto const& entry : snapshot.entries()) {
    REQUIRE(gate_test_access::filter_keep_ratio(consumer_test_access::gate(ordinary),
                                                {entry.filter.get(), entry.column_index},
                                                generation) ==
            gate_test_access::filter_keep_ratio(consumer_test_access::gate(fused),
                                                {entry.filter.get(), entry.column_index},
                                                generation));
  }
  return {prefixes, std::move(receipt_b), output_b->num_rows()};
}
}  // namespace

TEST_CASE("cached fusion pairs fixed programs across rounds and dead suffixes",
          "[dynamic_filter][cached_checkpoint][fused_consumer]")
{
  auto const count          = GENERATE(1, 2, 5, 9);
  auto const counts         = GENERATE(false, true);
  ::cuda::stream_ref stream = cudf::get_default_stream();
  auto input                = make_cached_residual_input(1000, 600, stream);
  auto channel              = std::make_shared<sirius_dynamic_filter_set>();
  auto producer             = channel->register_producer({0}, count);
  std::array<int, 9> limits{300, 120, 60, 30, 6, 0, 8, 2, 1};
  for (int i = 0; i < count; ++i) {
    REQUIRE(producer.push_filter(
      0, std::make_shared<counting_in_list_filter>(make_in_list_prefix(limits[i], stream), true)));
  }
  std::array bindings{binding{0, 0}};
  auto layout       = cached_layout(true);
  auto const result = require_cached_pair(
    input, input->view(), channel, bindings, layout, counts, (count + 3) / 4, stream);
  REQUIRE(result.rows == (count == 9 ? 0 : limits[count - 1]));
  if (counts) {
    std::vector<cudf::size_type> expected{600};
    int prefix = 600;
    for (int i = 0; i < count; ++i) {
      prefix = std::min(prefix, limits[i]);
      expected.push_back(prefix);
    }
    REQUIRE(result.counts == expected);
    REQUIRE(result.receipt.applied.size() == static_cast<std::size_t>(std::min(count, 6)));
  } else {
    REQUIRE(result.counts.empty());
    REQUIRE(result.receipt.applied.size() == (count == 9 ? 0 : static_cast<std::size_t>(count)));
  }
}

TEST_CASE("cached fusion keeps alias identities and decode history across distinct columns",
          "[dynamic_filter][cached_checkpoint][fused_consumer][consumer_alias]")
{
  auto const attached       = GENERATE(false, true);
  ::cuda::stream_ref stream = cudf::get_default_stream();
  std::shared_ptr<cudf::table const> input =
    make_sequence_table<int64_t>(100, stream, {{0, 1}, {10, 1}});
  auto filter   = make_in_list_prefix(40, stream);
  auto channel  = std::make_shared<sirius_dynamic_filter_set>();
  auto producer = channel->register_producer({0, 1, 2});
  for (auto ordinal : {0, 1, 2}) {
    REQUIRE(producer.push_filter(ordinal, filter));
  }
  std::array bindings{binding{0, 0}, binding{1, 0}, binding{2, 1}};
  auto layout              = cached_layout();
  layout.projected_columns = {1, 0, 1};
  batch_receipt prior{channel, 31, {}, {}};
  if (attached) { prior.decode_attached.push_back({filter, 1}); }
  auto const result = require_cached_pair(
    input, input->view(), channel, bindings, layout, true, 1, stream, attached ? &prior : nullptr);
  REQUIRE(result.rows == 30);
  REQUIRE(result.receipt.applied.size() == 3);
  REQUIRE(result.receipt.decode_attached.size() == (attached ? 1 : 0));
}

TEST_CASE("cached fusion falls back for a string unsigned or decimal probe among several filters",
          "[dynamic_filter][cached_checkpoint][fused_consumer]")
{
  auto const family         = GENERATE(0, 1, 2);
  auto const step_count     = GENERATE(5, 9);
  ::cuda::stream_ref stream = cudf::get_default_stream();
  auto const mr             = cudf::get_current_device_resource_ref();
  auto columns              = make_sequence_table<int64_t>(8, stream)->release();
  std::shared_ptr<sirius_dynamic_filter> unsupported;
  if (family == 0) {
    columns.push_back(sirius::test::operator_utils::make_string_column(
      {"a", "b", "c", "d", "e", "f", "g", "h"}, stream, mr));
    auto keys = sirius::test::operator_utils::make_string_column({"a", "d", "f"}, stream, mr);
    unsupported =
      std::make_shared<sirius::op::sirius_dynamic_in_list_filter>(keys->view(), stream, mr);
  } else if (family == 1) {
    columns.push_back(make_device_column<uint32_t>(
      {0, 1, 2, 3, 4, 5, 6, 7}, cudf::data_type{cudf::type_id::UINT32}, stream));
    auto keys =
      make_device_column<uint32_t>({0, 3, 5}, cudf::data_type{cudf::type_id::UINT32}, stream);
    unsupported =
      std::make_shared<sirius::op::sirius_dynamic_in_list_filter>(keys->view(), stream, mr);
  } else {
    columns.push_back(make_device_column<__int128>(
      {0, 1, 2, 3, 4, 5, 6, 7}, cudf::data_type{cudf::type_id::DECIMAL128, -2}, stream));
    auto keys =
      make_device_column<int64_t>({0, 3, 5}, cudf::data_type{cudf::type_id::DECIMAL64, -2}, stream);
    unsupported =
      std::make_shared<sirius::op::sirius_dynamic_in_list_filter>(keys->view(), stream, mr);
  }
  std::shared_ptr<cudf::table const> input = std::make_shared<cudf::table>(std::move(columns));
  auto channel                             = std::make_shared<sirius_dynamic_filter_set>();
  auto producer                            = channel->register_producer({0, 1}, step_count);
  std::vector<std::shared_ptr<counting_in_list_filter>> supported;
  for (int i = 1; i < step_count; ++i) {
    supported.push_back(
      std::make_shared<counting_in_list_filter>(make_in_list_prefix(8, stream), true));
    REQUIRE(producer.push_filter(0, supported.back()));
  }
  REQUIRE(producer.push_filter(1, unsupported));
  std::array bindings{binding{0, 0}, binding{1, 1}};
  auto layout = cached_layout();
  auto const result =
    require_cached_pair(input, input->view(), channel, bindings, layout, true, 0, stream);
  REQUIRE(result.rows == 3);
  REQUIRE(result.receipt.applied.size() == static_cast<std::size_t>(step_count));
  for (auto const& filter : supported) {
    REQUIRE(filter->mask_calls() == 2);
  }
}

TEST_CASE("cached fused residual peak excludes the ordinary probe mask",
          "[dynamic_filter][cached_checkpoint][fused_consumer]")
{
  constexpr cudf::size_type rows = cudf::size_type{1} << 22;
  ::cuda::stream_ref stream      = cudf::get_default_stream();
  auto input                     = make_cached_residual_input(rows, rows / 64, stream);
  auto channel                   = std::make_shared<sirius_dynamic_filter_set>();
  auto producer                  = channel->register_producer({0});
  REQUIRE(producer.push_filter(0, make_in_list_prefix(rows / 128, stream)));
  REQUIRE(producer.push_filter(0, make_in_list_prefix(rows / 256, stream)));
  dynamic_filter_consumer consumer(channel, {.force_counts = true, .fused_membership = true}, {});
  std::array bindings{binding{0, 0}};
  auto layout = cached_layout(true);
  auto program =
    consumer.prepare({.source = input->view(), .bindings = bindings, .cached_layout = &layout});
  REQUIRE(program);
  rmm::mr::statistics_resource_adaptor allocations{
    cuda::mr::any_resource<cuda::mr::device_accessible>{cudf::get_current_device_resource_ref()}};
  auto result = consumer.apply(std::move(*program),
                               owning_table_view{input, input->view()},
                               stream,
                               rmm::device_async_resource_ref{allocations});
  stream.sync();
  REQUIRE(consumer_test_access::storage(result).fused_rounds == 1);
  REQUIRE(counts_of(result) == std::vector<cudf::size_type>{rows / 64, rows / 128, rows / 256});
  auto output     = result.take_output();
  auto const peak = static_cast<std::size_t>(allocations.get_bytes_counter().peak);
  REQUIRE(peak >= static_cast<std::size_t>(rows));
  REQUIRE(peak <= static_cast<std::size_t>(rows) + output->alloc_size() + (std::size_t{1} << 20));
}

namespace {
class preflight_allocation_failure final : public sirius_dynamic_filter,
                                           public sirius_mask_applicable {
 public:
  sirius_dynamic_filter_kind kind() const override { return sirius_dynamic_filter_kind::IN_LIST; }
  bool is_available_on_device(int) const noexcept override { return true; }
  bool prepare_fused_probe(cudf::column_view const&,
                           int,
                           sirius::op::detail::fused_probe&) const override
  {
    throw std::bad_alloc{};
  }
  std::unique_ptr<cudf::column> compute_mask(cudf::column_view const&,
                                             uint32_t const*,
                                             int,
                                             ::cuda::stream_ref,
                                             rmm::device_async_resource_ref) const override
  {
    FAIL("preflight exception must not restart ordinary membership");
    return nullptr;
  }
};
}  // namespace

TEST_CASE("cached fusion retires submitted residual reads on host preflight failure",
          "[dynamic_filter][cached_checkpoint][fused_consumer][consumer_retirement]")
{
  auto const failed_join    = GENERATE(false, true);
  ::cuda::stream_ref stream = cudf::get_default_stream();
  auto input                = make_cached_residual_input(1000, 600, stream);
  auto channel              = std::make_shared<sirius_dynamic_filter_set>();
  auto producer             = channel->register_producer({0});
  auto first                = make_in_list_prefix(300, stream);
  REQUIRE(producer.push_filter(0, first));
  REQUIRE(producer.push_filter(0, std::make_shared<preflight_allocation_failure>()));
  dynamic_filter_consumer consumer(channel, {.force_counts = true, .fused_membership = true}, {});
  auto layout = cached_layout(true);
  std::array bindings{binding{0, 0}};
  auto program =
    consumer.prepare({.source = input->view(), .bindings = bindings, .cached_layout = &layout});
  REQUIRE(program);
  rmm::mr::statistics_resource_adaptor allocations{
    cuda::mr::any_resource<cuda::mr::device_accessible>{cudf::get_current_device_resource_ref()}};
  {
    read_retirement_test_hooks faults{std::nullopt, failed_join};
    auto apply = [&] {
      return consumer.apply(std::move(*program),
                            owning_table_view{input, input->view()},
                            stream,
                            rmm::device_async_resource_ref{allocations});
    };
    if (failed_join) {
      REQUIRE_THROWS_AS(apply(), sirius::unjoined_gpu_work);
      auto retained = consumer_test_access::retained_cached_storage(faults);
      REQUIRE(retained);
      REQUIRE(retained->has_source);
      REQUIRE(retained->has_counters);
      REQUIRE(retained->fused_rounds == 0);
      REQUIRE(allocations.get_bytes_counter().value >= 1000);
    } else {
      REQUIRE_THROWS_AS(apply(), std::bad_alloc);
      REQUIRE_FALSE(faults.retained.has_value());
      REQUIRE(allocations.get_bytes_counter().value == 0);
    }
    REQUIRE_FALSE(gate_test_access::filter_keep_ratio(
      consumer_test_access::gate(consumer), {first.get(), 0}, 2));
  }
  REQUIRE(allocations.get_bytes_counter().value == 0);
}
