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
 * @file test_dynamic_filter_publisher.cpp
 * @brief Pins dynamic_filter_publisher::publish's per-key membership-representation choice
 *
 * Sections, in file order:
 *  - Tier precedence: the raw small IN-list wins at <= k_max_keys build rows, and one row above
 *    that gate falls through to the hash IN-list under the default L2 fraction.
 *  - inlist_max_l2_fraction semantics: a vanishing fraction demotes the hash IN-list to the
 *    Bloom filter; 1.0 reproduces the legacy L2-fit rule; 0 always publishes the Bloom for the
 *    hash tier while leaving small-IN-list precedence untouched; and the
 *    set_bytes <= fraction * l2_bytes comparison is inclusive, pinned at exact double equality
 *    against a synthetic injected L2 size.
 *  - No device L2 info: the publisher's l2_bytes_override constructor seam injects l2_bytes == 0,
 *    which fails the legacy fit rule closed and publishes the Bloom before the fraction is ever
 *    consulted.
 *  - Type-coverage canary: the hash-IN-list and Bloom supported key-type sets must coincide,
 *    checked in both directions over candidate key types; divergence prescribes the follow-up
 *    publish-path test.
 *  - Key-ordinal safety: an inequality condition ordinal never receives a membership filter.
 */

#include "op/dynamic_filter_publisher.hpp"
#include "op/sirius_dynamic_filter.hpp"
#include "operator_test_utils.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_device.hpp>

#include <cuda_runtime_api.h>

#include <catch.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

constexpr int kDeviceId                 = 0;
constexpr std::size_t kProbeColumnIndex = 7;

template <typename MemoryManager>
std::vector<sirius::op::dynamic_filter_replica_space> get_replica_spaces(
  MemoryManager& memory_manager)
{
  auto const gpu_spaces  = memory_manager.get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
  auto const host_spaces = memory_manager.get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
  REQUIRE(gpu_spaces.size() == 1);
  REQUIRE_FALSE(host_spaces.empty());

  auto* gpu_space = memory_manager.get_memory_space(cucascade::memory::Tier::GPU,
                                                    gpu_spaces.front()->get_device_id());
  REQUIRE(gpu_space != nullptr);
  auto const local_host =
    std::find_if(host_spaces.begin(), host_spaces.end(), [gpu_space](auto const* host_space) {
      return host_space->get_device_id() == gpu_space->get_device_id();
    });
  auto const* host_space = local_host == host_spaces.end() ? host_spaces.front() : *local_host;
  return {{*gpu_space, *host_space}};
}

template <typename ExpectedFilter>
void require_published_membership(
  std::size_t rows,
  double inlist_max_l2_fraction =
    sirius::op::dynamic_filter_publish_plan::k_default_inlist_max_l2_fraction,
  std::optional<std::size_t> l2_bytes_override = std::nullopt)
{
  rmm::cuda_set_device_raii const device{rmm::cuda_device_id{kDeviceId}};
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager(1);
  auto replica_spaces = get_replica_spaces(*memory_manager);
  auto& source_space  = replica_spaces.front().get_gpu_space();

  auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  std::vector<sirius::op::dynamic_filter_publish_plan::probe_target> targets;
  targets.push_back({channel, {kProbeColumnIndex}, {cudf::data_type{cudf::type_id::INT64}}});
  sirius::op::dynamic_filter_publish_plan plan{
    std::move(targets),
    false,
    {0},
    std::move(replica_spaces),
    /*domain_coverage_threshold=*/
    sirius::op::dynamic_filter_publish_plan::k_default_domain_coverage_threshold,
    inlist_max_l2_fraction};

  duckdb::JoinFilterPushdownInfo pushdown{};
  pushdown.join_condition.push_back(0);
  std::vector<sirius::op::sirius_physical_hash_join::key_cast_info> key_casts(1);
  std::vector<cudf::size_type> right_key_col_indices{0};

  auto const stream = source_space.acquire_stream();
  auto keys         = cudf::sequence(static_cast<cudf::size_type>(rows),
                             cudf::numeric_scalar<std::int64_t>(0, true, stream),
                             cudf::numeric_scalar<std::int64_t>(1, true, stream),
                             stream,
                             source_space.get_default_allocator());

  if constexpr (std::is_same_v<ExpectedFilter, sirius::op::sirius_dynamic_small_in_list_filter>) {
    REQUIRE(sirius::op::sirius_dynamic_small_in_list_filter::supports(keys->view()));
  } else {
    REQUIRE_FALSE(sirius::op::sirius_dynamic_small_in_list_filter::supports(keys->view()));
    // With an injected L2 size the caller constructs the synthetic scenario deliberately
    // (including l2_bytes == 0), so the live-device sanity checks apply only without one.
    if (!l2_bytes_override) {
      int l2_bytes = 0;
      REQUIRE(cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, kDeviceId) == cudaSuccess);
      REQUIRE(l2_bytes > 0);
      REQUIRE(sirius::op::sirius_dynamic_in_list_filter::estimated_set_bytes(
                rows, cudf::data_type{cudf::type_id::INT64}) <= static_cast<std::size_t>(l2_bytes));
    }
  }

  std::vector<cudf::column_view> columns{keys->view()};
  cudf::table_view build_view{columns};
  sirius::op::dynamic_filter_publisher{
    pushdown, plan, key_casts, right_key_col_indices, l2_bytes_override}
    .publish(build_view, stream);

  auto const snapshot = channel->filters_for_column(kProbeColumnIndex);
  REQUIRE(snapshot.size() == 1);
  auto const* selected = dynamic_cast<ExpectedFilter const*>(snapshot.front().get());
  REQUIRE(selected != nullptr);
  REQUIRE(selected->is_available_on_device(kDeviceId));
  if constexpr (requires(ExpectedFilter const& f) { f.size(); }) {
    REQUIRE(selected->size() == rows);
  }
  REQUIRE(selected->replica_count() == 1);
  if constexpr (std::is_same_v<ExpectedFilter, sirius::op::sirius_dynamic_in_list_filter>) {
    REQUIRE(selected->has_persistent_set());
  }
}

}  // namespace

TEST_CASE("dynamic-filter publisher selects the raw small IN-list", "[dynamic_filter][publisher]")
{
  require_published_membership<sirius::op::sirius_dynamic_small_in_list_filter>(3);
}

TEST_CASE("dynamic-filter publisher falls through to the hash IN-list above the small-list gate",
          "[dynamic_filter][publisher]")
{
  require_published_membership<sirius::op::sirius_dynamic_in_list_filter>(
    sirius::op::sirius_dynamic_small_in_list_filter::k_max_keys + 1);
}

TEST_CASE("dynamic-filter publisher demotes the hash IN-list to Bloom above the L2 fraction",
          "[dynamic_filter][publisher]")
{
  // A vanishing fraction makes the residency threshold (fraction x L2) smaller than any real
  // hash set, so the smallest hash-tier build must demote to the Bloom.
  require_published_membership<sirius::op::sirius_dynamic_bloom_filter>(
    sirius::op::sirius_dynamic_small_in_list_filter::k_max_keys + 1, 1e-12);
}

TEST_CASE("dynamic-filter publisher fraction 1.0 reproduces the legacy L2-fit rule",
          "[dynamic_filter][publisher]")
{
  // fraction = 1.0 -> threshold = the full L2, so every L2-fitting set keeps the exact IN-list.
  require_published_membership<sirius::op::sirius_dynamic_in_list_filter>(
    sirius::op::sirius_dynamic_small_in_list_filter::k_max_keys + 1, 1.0);
}

TEST_CASE("dynamic-filter publisher fraction 0 always publishes the Bloom for the hash tier",
          "[dynamic_filter][publisher]")
{
  // 0 x L2 = 0 and a non-empty build's set estimate is positive, so no special case is needed
  // in the publisher: the hash tier can never win and the Bloom is published instead.
  require_published_membership<sirius::op::sirius_dynamic_bloom_filter>(
    sirius::op::sirius_dynamic_small_in_list_filter::k_max_keys + 1, 0.0);
}

TEST_CASE("dynamic-filter publisher fraction 0 leaves small-list precedence untouched",
          "[dynamic_filter][publisher]")
{
  // The knob governs only the hash-set-vs-Bloom trade; the raw-needle small IN-list still wins
  // first at every fraction, including 0.
  require_published_membership<sirius::op::sirius_dynamic_small_in_list_filter>(3, 0.0);
}

TEST_CASE("dynamic-filter publisher keeps the hash IN-list at exact L2-fraction equality",
          "[dynamic_filter][publisher]")
{
  // Inclusivity pin, exact in double arithmetic on any hardware: with a synthetic L2 of
  // 2 x set_bytes and fraction 0.5, the threshold equals set_bytes exactly, so the inclusive
  // set_bytes <= fraction x l2_bytes comparison keeps the IN-list (a strict '<' would demote);
  // an L2 two bytes smaller puts the threshold at set_bytes - 1 and must demote to the Bloom.
  auto const rows      = sirius::op::sirius_dynamic_small_in_list_filter::k_max_keys + 1;
  auto const set_bytes = sirius::op::sirius_dynamic_in_list_filter::estimated_set_bytes(
    rows, cudf::data_type{cudf::type_id::INT64});
  REQUIRE(set_bytes >= 2);
  require_published_membership<sirius::op::sirius_dynamic_in_list_filter>(rows, 0.5, 2 * set_bytes);
  require_published_membership<sirius::op::sirius_dynamic_bloom_filter>(
    rows, 0.5, 2 * set_bytes - 2);
}

TEST_CASE("dynamic-filter publisher publishes the Bloom when no device L2 size is available",
          "[dynamic_filter][publisher]")
{
  // l2_bytes == 0 fails the legacy fit rule closed before the fraction is ever consulted (the
  // documented fallback in sirius_config.hpp), so the hash tier demotes to the Bloom.
  require_published_membership<sirius::op::sirius_dynamic_bloom_filter>(
    sirius::op::sirius_dynamic_small_in_list_filter::k_max_keys + 1,
    sirius::op::dynamic_filter_publish_plan::k_default_inlist_max_l2_fraction,
    /*l2_bytes_override=*/0);
}

TEST_CASE("dynamic-filter Bloom and hash-IN-list supported key types coincide",
          "[dynamic_filter][publisher]")
{
  // The publish rule keeps an L2-fitting hash IN-list at any fraction when the key type has no
  // Bloom fallback; that clause is structurally unreachable while the two supported-type sets
  // coincide. This canary fails on divergence in EITHER direction over the candidate key types
  // below (compared on null-free empty columns: the IN-list signature also checks nulls, but
  // that dimension is orthogonal and invisible to Bloom's type-only signature). If it fires,
  // add a publish-path test asserting the divergent type keeps a fitting IN-list at fraction 0
  // (IN-list-only types) or demotes to the Bloom (Bloom-only types).
  constexpr cudf::type_id candidate_ids[] = {cudf::type_id::BOOL8,
                                             cudf::type_id::INT8,
                                             cudf::type_id::INT16,
                                             cudf::type_id::INT32,
                                             cudf::type_id::INT64,
                                             cudf::type_id::UINT8,
                                             cudf::type_id::UINT16,
                                             cudf::type_id::UINT32,
                                             cudf::type_id::UINT64,
                                             cudf::type_id::FLOAT32,
                                             cudf::type_id::FLOAT64,
                                             cudf::type_id::STRING};
  for (auto const id : candidate_ids) {
    auto const type  = cudf::data_type{id};
    auto const empty = cudf::make_empty_column(type);
    INFO("type_id=" << static_cast<int>(id));
    REQUIRE(sirius::op::sirius_dynamic_in_list_filter::supports(empty->view()) ==
            sirius::op::sirius_dynamic_bloom_filter::supports(type));
  }
  // Anchor the sets as non-empty: today both are exactly {INT32, INT64}.
  REQUIRE(sirius::op::sirius_dynamic_bloom_filter::supports(cudf::data_type{cudf::type_id::INT32}));
  REQUIRE(sirius::op::sirius_dynamic_bloom_filter::supports(cudf::data_type{cudf::type_id::INT64}));
}

TEST_CASE("dynamic-filter publisher never publishes for an inequality condition ordinal",
          "[dynamic_filter][publisher]")
{
  rmm::cuda_set_device_raii const device{rmm::cuda_device_id{kDeviceId}};
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager(1);
  auto replica_spaces = get_replica_spaces(*memory_manager);
  auto& source_space  = replica_spaces.front().get_gpu_space();

  auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  std::vector<sirius::op::dynamic_filter_publish_plan::probe_target> targets;
  targets.push_back(
    {channel,
     {kProbeColumnIndex, kProbeColumnIndex + 1},
     {cudf::data_type{cudf::type_id::INT64}, cudf::data_type{cudf::type_id::INT64}}});
  sirius::op::dynamic_filter_publish_plan plan{
    std::move(targets), false, {0, 0}, std::move(replica_spaces)};

  // A MIXED_JOIN shape: DuckDB records pushdown candidates for the equality
  // (ordinal 0) AND the inequality (ordinal 1) condition, but Sirius extracts
  // build key columns only for equalities, so right_key_col_indices has one
  // entry and ordinal 1 must publish nothing. A membership filter keyed on an
  // inequality column would drop probe rows that satisfy the inequality but
  // miss the build's exact key set.
  duckdb::JoinFilterPushdownInfo pushdown{};
  pushdown.join_condition.push_back(0);
  pushdown.join_condition.push_back(1);
  std::vector<sirius::op::sirius_physical_hash_join::key_cast_info> key_casts(1);
  std::vector<cudf::size_type> right_key_col_indices{0};

  auto const stream = source_space.acquire_stream();
  auto eq_keys      = cudf::sequence(3,
                                cudf::numeric_scalar<std::int64_t>(0, true, stream),
                                cudf::numeric_scalar<std::int64_t>(1, true, stream),
                                stream,
                                source_space.get_default_allocator());
  auto ineq_keys    = cudf::sequence(3,
                                  cudf::numeric_scalar<std::int64_t>(100, true, stream),
                                  cudf::numeric_scalar<std::int64_t>(1, true, stream),
                                  stream,
                                  source_space.get_default_allocator());

  std::vector<cudf::column_view> columns{eq_keys->view(), ineq_keys->view()};
  cudf::table_view build_view{columns};
  sirius::op::dynamic_filter_publisher{pushdown, plan, key_casts, right_key_col_indices}.publish(
    build_view, stream);

  REQUIRE(channel->filters_for_column(kProbeColumnIndex).size() == 1);
  REQUIRE(channel->filters_for_column(kProbeColumnIndex + 1).empty());
}
