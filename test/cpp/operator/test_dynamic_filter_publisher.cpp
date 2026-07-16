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
void require_published_membership(std::size_t rows)
{
  rmm::cuda_set_device_raii const device{rmm::cuda_device_id{kDeviceId}};
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager(1);
  auto replica_spaces = get_replica_spaces(*memory_manager);
  auto& source_space  = replica_spaces.front().get_gpu_space();

  auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  std::vector<sirius::op::dynamic_filter_publish_plan::probe_target> targets;
  targets.push_back({channel, {kProbeColumnIndex}, {cudf::data_type{cudf::type_id::INT64}}});
  sirius::op::dynamic_filter_publish_plan plan{
    std::move(targets), false, {0}, std::move(replica_spaces)};

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
    int l2_bytes = 0;
    REQUIRE(cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, kDeviceId) == cudaSuccess);
    REQUIRE(l2_bytes > 0);
    REQUIRE(sirius::op::sirius_dynamic_in_list_filter::estimated_set_bytes(
              rows, cudf::data_type{cudf::type_id::INT64}) <= static_cast<std::size_t>(l2_bytes));
  }

  std::vector<cudf::column_view> columns{keys->view()};
  cudf::table_view build_view{columns};
  sirius::op::dynamic_filter_publisher{pushdown, plan, key_casts, right_key_col_indices}.publish(
    build_view, stream);

  auto const snapshot = channel->filters_for_column(kProbeColumnIndex);
  REQUIRE(snapshot.size() == 1);
  auto const* selected = dynamic_cast<ExpectedFilter const*>(snapshot.front().get());
  REQUIRE(selected != nullptr);
  REQUIRE(selected->is_available_on_device(kDeviceId));
  REQUIRE(selected->size() == rows);
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
