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
 * @file test_physical_dynamic_filter_mode.cpp
 * @brief Deterministic flip semantics of `sirius_physical_dynamic_filter::effective_mode()`.
 *
 * The pinned-serve fix (main doc, "Pinned-cache-served scans") promotes a plan-time
 * `membership_masks_only` wrapper to `include_ast_row_masks` when its shared
 * `read_time_filter_bypass` is marked. Each case constructs the operator directly, feeds it one
 * device batch, and pins one leg of that promotion: the unmarked latch preserves the fresh-read
 * defect-by-design pass-through, the marked latch applies Top-N boundaries and zone maps, an
 * already-AST plan mode is unaffected, and membership filters work under either latch state. Each
 * section carries a local `dynamic_filter_stats`, so the `post_decode_apply_rows_in/out` counter
 * mechanics are pinned deterministically alongside the mode.
 */

#include "helper/type_conversions.hpp"
#include "operator_test_utils.hpp"

#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>

#include <catch.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <op/dynamic_filter/dynamic_filter_stats.hpp>
#include <op/dynamic_filter/sirius_dynamic_filter.hpp>
#include <op/scan/read_time_filter_bypass.hpp>
#include <op/scan/sirius_physical_dynamic_filter.hpp>

#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>

using sirius::op::dynamic_filter_null_policy;
using sirius::op::dynamic_filter_stats;
using sirius::op::exact_host_scalar;
using sirius::op::range_bound_side;
using sirius::op::sirius_dynamic_filter_set;
using sirius::op::sirius_dynamic_range_filter;
using sirius::op::sirius_dynamic_zone_map_filter;
using sirius::op::zone_map_entry;
using sirius::op::scan::dynamic_filter_apply_mode;
using sirius::op::scan::dynamic_filter_endpoint_provenance;
using sirius::op::scan::read_time_filter_bypass;
using sirius::op::scan::sirius_physical_dynamic_filter;

namespace {

using namespace sirius::test::operator_utils;

constexpr auto k_int32 = cudf::data_type{cudf::type_id::INT32};

/// One-sided Top-N-style boundary: keeps rows with value <= @p bound (UPPER, inclusive).
std::shared_ptr<sirius_dynamic_range_filter> make_upper_bound(std::int32_t bound)
{
  return std::make_shared<sirius_dynamic_range_filter>(exact_host_scalar{bound, k_int32},
                                                       range_bound_side::UPPER,
                                                       /*inclusive=*/true,
                                                       dynamic_filter_null_policy::REJECT);
}

/// Zone map [lo, hi] with device scalars on the executing (current) device.
std::shared_ptr<sirius_dynamic_zone_map_filter> make_zone_map(std::int32_t lo, std::int32_t hi)
{
  auto stream = default_stream();
  auto mr     = cudf::get_current_device_resource_ref();
  std::vector<zone_map_entry> zones;
  zones.push_back({std::make_unique<cudf::numeric_scalar<std::int32_t>>(lo, true, stream, mr),
                   std::make_unique<cudf::numeric_scalar<std::int32_t>>(hi, true, stream, mr)});
  return std::make_shared<sirius_dynamic_zone_map_filter>(std::move(zones));
}

/// IN-list membership filter keeping INT32 keys [0, count).
std::shared_ptr<sirius::op::sirius_dynamic_in_list_filter> make_in_list_prefix(std::int32_t count)
{
  auto stream = default_stream();
  auto keys   = cudf::sequence(count,
                             cudf::numeric_scalar<std::int32_t>(0, true, stream),
                             cudf::numeric_scalar<std::int32_t>(1, true, stream),
                             stream);
  return std::make_shared<sirius::op::sirius_dynamic_in_list_filter>(
    keys->view(), stream, cudf::get_current_device_resource_ref());
}

duckdb::vector<sirius::logical_type> single_int32_type()
{
  duckdb::vector<duckdb::LogicalType> types;
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER));
  return sirius::from_duckdb_vec(types);
}

/// Run one batch of INT32 values [0, rows) through @p op and return the surviving values.
std::vector<std::int32_t> execute_sequence_batch(sirius_physical_dynamic_filter& op,
                                                 cucascade::memory::memory_space& space,
                                                 std::int32_t rows)
{
  std::vector<std::int32_t> values(static_cast<std::size_t>(rows));
  std::iota(values.begin(), values.end(), 0);
  std::vector<std::shared_ptr<cucascade::data_batch>> inputs{
    make_numeric_batch<std::int32_t>(space, values, cudf::type_id::INT32)};

  auto output = op.execute(sirius::op::pipelineable_operator_data(inputs), default_stream());
  auto const& batches =
    dynamic_cast<sirius::op::pipelineable_operator_data const&>(*output).get_data_batches();
  REQUIRE(batches.size() == 1);
  auto const view = sirius::get_cudf_table_view(*batches[0]);
  return copy_column_to_host<std::int32_t>(view.column(0));
}

}  // namespace

TEST_CASE("physical dynamic filter - membership mode leaves a boundary-only channel unapplied",
          "[dynamic_filter]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);

  auto filters = std::make_shared<sirius_dynamic_filter_set>();
  filters->push_filter(0, make_upper_bound(3));

  dynamic_filter_stats stats;
  auto latch = std::make_shared<read_time_filter_bypass>();
  sirius_physical_dynamic_filter op(single_int32_type(),
                                    10,
                                    filters,
                                    sirius::op::scan::dynamic_filter_gate::k_default_keep_threshold,
                                    dynamic_filter_apply_mode::membership_masks_only,
                                    dynamic_filter_endpoint_provenance::scan_route,
                                    &stats,
                                    latch);

  // Fresh-read premise: the boundary is a reader concern, so the membership-only wrapper must
  // pass the batch through untouched -- and a pass-through batch never counts.
  REQUIRE(op.effective_mode() == dynamic_filter_apply_mode::membership_masks_only);
  auto const survivors = execute_sequence_batch(op, *space, 10);
  REQUIRE(survivors.size() == 10);
  REQUIRE(stats.post_decode_apply_rows_in.load() == 0);
  REQUIRE(stats.post_decode_apply_rows_out.load() == 0);
}

TEST_CASE(
  "physical dynamic filter - a marked latch promotes membership to include_ast and "
  "compacts",
  "[dynamic_filter]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);

  auto filters = std::make_shared<sirius_dynamic_filter_set>();
  filters->push_filter(0, make_upper_bound(3));

  dynamic_filter_stats stats;
  auto latch = std::make_shared<read_time_filter_bypass>();
  sirius_physical_dynamic_filter op(single_int32_type(),
                                    10,
                                    filters,
                                    sirius::op::scan::dynamic_filter_gate::k_default_keep_threshold,
                                    dynamic_filter_apply_mode::membership_masks_only,
                                    dynamic_filter_endpoint_provenance::scan_route,
                                    &stats,
                                    latch);

  latch->mark_bypassed();
  REQUIRE(op.effective_mode() == dynamic_filter_apply_mode::include_ast_row_masks);

  auto const survivors = execute_sequence_batch(op, *space, 10);
  REQUIRE(survivors == std::vector<std::int32_t>{0, 1, 2, 3});
  REQUIRE(stats.post_decode_apply_rows_in.load() == 10);
  REQUIRE(stats.post_decode_apply_rows_out.load() == 4);
}

TEST_CASE("physical dynamic filter - the latch is inert when the plan mode is already include_ast",
          "[dynamic_filter]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);

  auto filters = std::make_shared<sirius_dynamic_filter_set>();
  filters->push_filter(0, make_upper_bound(3));

  // Promotion is monotone: marking the latch on an already-AST wrapper is the identity, so the
  // native-scan parity cannot regress. Two operator instances keep the gates independent.
  auto const run = [&](bool marked) {
    dynamic_filter_stats stats;
    auto latch = std::make_shared<read_time_filter_bypass>();
    if (marked) { latch->mark_bypassed(); }
    sirius_physical_dynamic_filter op(
      single_int32_type(),
      10,
      filters,
      sirius::op::scan::dynamic_filter_gate::k_default_keep_threshold,
      dynamic_filter_apply_mode::include_ast_row_masks,
      dynamic_filter_endpoint_provenance::scan_route,
      &stats,
      latch);
    REQUIRE(op.effective_mode() == dynamic_filter_apply_mode::include_ast_row_masks);
    return execute_sequence_batch(op, *space, 10);
  };

  REQUIRE(run(/*marked=*/false) == run(/*marked=*/true));
}

TEST_CASE("physical dynamic filter - a zone map applies through the promoted wrapper",
          "[dynamic_filter]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);

  auto filters = std::make_shared<sirius_dynamic_filter_set>();
  filters->push_filter(0, make_zone_map(3, 6));

  dynamic_filter_stats stats;
  auto latch = std::make_shared<read_time_filter_bypass>();
  sirius_physical_dynamic_filter op(single_int32_type(),
                                    10,
                                    filters,
                                    sirius::op::scan::dynamic_filter_gate::k_default_keep_threshold,
                                    dynamic_filter_apply_mode::membership_masks_only,
                                    dynamic_filter_endpoint_provenance::scan_route,
                                    &stats,
                                    latch);

  // The join-path leg of the fix: a zone map is AST-lowerable only, so it applies exactly when
  // the promoted mode runs the combined AST row mask.
  latch->mark_bypassed();
  auto const survivors = execute_sequence_batch(op, *space, 10);
  REQUIRE(survivors == std::vector<std::int32_t>{3, 4, 5, 6});
  REQUIRE(stats.post_decode_apply_rows_in.load() == 10);
  REQUIRE(stats.post_decode_apply_rows_out.load() == 4);
}

TEST_CASE("physical dynamic filter - membership filters keep working under an unmarked latch",
          "[dynamic_filter]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);

  auto filters = std::make_shared<sirius_dynamic_filter_set>();
  filters->push_filter(0, make_in_list_prefix(4));

  dynamic_filter_stats stats;
  auto latch = std::make_shared<read_time_filter_bypass>();
  sirius_physical_dynamic_filter op(single_int32_type(),
                                    10,
                                    filters,
                                    sirius::op::scan::dynamic_filter_gate::k_default_keep_threshold,
                                    dynamic_filter_apply_mode::membership_masks_only,
                                    dynamic_filter_endpoint_provenance::scan_route,
                                    &stats,
                                    latch);

  // The one capability that already worked on the fresh-read path must survive the promotion
  // logic untouched.
  REQUIRE(op.effective_mode() == dynamic_filter_apply_mode::membership_masks_only);
  auto const survivors = execute_sequence_batch(op, *space, 10);
  REQUIRE(survivors == std::vector<std::int32_t>{0, 1, 2, 3});
  REQUIRE(stats.post_decode_apply_rows_in.load() == 10);
  REQUIRE(stats.post_decode_apply_rows_out.load() == 4);
}
