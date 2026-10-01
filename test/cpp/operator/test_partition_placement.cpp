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

// Unit tests for partition_placement (which GPU each partition of an exchange runs on), the
// partition_strategy that carries it, and the partitioned_operator_data constructor that stamps a
// partition's device as the task's preferred device. GPU-free.

#include "op/partition_placement.hpp"
#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_partition_consumer_operator.hpp"

#include <catch.hpp>

#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

using sirius::op::partition_placement;
using sirius::op::partition_strategy;
using sirius::op::partitioned_operator_data;
using sirius::op::select_gpu_subset;

namespace {

std::vector<std::shared_ptr<cucascade::data_batch>> no_batches() { return {}; }

}  // namespace

TEST_CASE("partition_placement::round_robin cycles through the GPU list",
          "[partition_placement][unit]")
{
  auto const p = partition_placement::round_robin(4, {0, 1});
  REQUIRE(p.num_partitions() == 4);
  REQUIRE(p.device_for(0) == 0);
  REQUIRE(p.device_for(1) == 1);
  REQUIRE(p.device_for(2) == 0);
  REQUIRE(p.device_for(3) == 1);
  REQUIRE(p.any_pinned());
  REQUIRE(p.devices() == std::vector<int>{0, 1});

  // Fewer partitions than GPUs uses a prefix of the list.
  auto const fewer = partition_placement::round_robin(2, {4, 5, 6});
  REQUIRE(fewer.device_for(0) == 4);
  REQUIRE(fewer.device_for(1) == 5);
  REQUIRE(fewer.devices() == std::vector<int>{4, 5});
}

TEST_CASE("partition_placement::round_robin with no GPUs is unpinned",
          "[partition_placement][unit]")
{
  auto const p = partition_placement::round_robin(3, {});
  REQUIRE(p.num_partitions() == 3);
  REQUIRE_FALSE(p.any_pinned());
  REQUIRE(p.devices().empty());
  for (std::size_t i = 0; i < 3; ++i) {
    REQUIRE_FALSE(p.device_for(i).has_value());
  }
  REQUIRE(p == partition_placement::unpinned(3));
}

TEST_CASE("partition_placement::one_per_device maps slot i to gpu_ids[i]",
          "[partition_placement][unit]")
{
  auto const p = partition_placement::one_per_device({3, 5});
  REQUIRE(p.num_partitions() == 2);
  REQUIRE(p.device_for(0) == 3);
  REQUIRE(p.device_for(1) == 5);
  REQUIRE(p.first_partition_for_device(3) == std::optional<std::size_t>{0});
  REQUIRE(p.first_partition_for_device(5) == std::optional<std::size_t>{1});
  REQUIRE_FALSE(p.first_partition_for_device(4).has_value());

  REQUIRE(partition_placement::one_per_device({}) == partition_placement::unpinned(1));
}

TEST_CASE("partition_placement::first_partition_for_device returns the first pinned slot",
          "[partition_placement][unit]")
{
  auto const p = partition_placement::round_robin(4, {2, 7});
  REQUIRE(p.first_partition_for_device(2) == std::optional<std::size_t>{0});
  REQUIRE(p.first_partition_for_device(7) == std::optional<std::size_t>{1});
  REQUIRE_FALSE(partition_placement::unpinned(2).first_partition_for_device(0).has_value());
}

TEST_CASE("partition_placement rejects invalid shapes and indices", "[partition_placement][unit]")
{
  REQUIRE_THROWS_AS(partition_placement::unpinned(0), std::invalid_argument);
  REQUIRE_THROWS_AS(partition_placement::round_robin(0, {0}), std::invalid_argument);
  REQUIRE_THROWS_AS(partition_placement::round_robin(2, {0}).device_for(2), std::out_of_range);
}

TEST_CASE("partition_placement::devices deduplicates in slot order", "[partition_placement][unit]")
{
  auto const p = partition_placement::round_robin(5, {3, 1, 2});
  REQUIRE(p.devices() == std::vector<int>{3, 1, 2});
}

TEST_CASE("partition_placement::to_string lists every slot", "[partition_placement][unit]")
{
  REQUIRE(partition_placement::round_robin(2, {1, 3}).to_string() == "[0->1, 1->3]");
  REQUIRE(partition_placement::unpinned(2).to_string() == "[0->?, 1->?]");
}

TEST_CASE("select_gpu_subset rotates by the seed and clamps k", "[partition_placement][unit]")
{
  REQUIRE(select_gpu_subset({0, 1, 2}, 1, 4) == std::vector<int>{1});
  REQUIRE(select_gpu_subset({0, 1, 2}, 2, 2) == std::vector<int>{2, 0});
  REQUIRE(select_gpu_subset({0, 1, 2}, 5, 0) == std::vector<int>{0, 1, 2});
  REQUIRE(select_gpu_subset({0, 1, 2}, 0, 1).empty());
  REQUIRE(select_gpu_subset({}, 1, 3).empty());
}

TEST_CASE("partition_strategy requires a placement of the same partition count",
          "[partition_placement][unit]")
{
  partition_strategy const ok{2, false, false, partition_placement::round_robin(2, {0, 1})};
  REQUIRE(ok.num_partitions == 2);
  REQUIRE(ok.placement.num_partitions() == 2);

  REQUIRE_THROWS_AS(partition_strategy(2, false, false, partition_placement::unpinned(3)),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(partition_strategy(0, false, false, partition_placement::unpinned(1)),
                    std::invalid_argument);
}

TEST_CASE("partitioned_operator_data stamps the partition's device as its preference",
          "[partition_placement][unit]")
{
  auto const placement = partition_placement::round_robin(3, {4, 6});

  partitioned_operator_data const pinned(no_batches(), 1, placement);
  REQUIRE(pinned.get_partition_idx() == std::optional<std::size_t>{1});
  REQUIRE(pinned.get_preferred_device_id() == std::optional<int>{6});

  partitioned_operator_data const wrapped(no_batches(), 2, placement);
  REQUIRE(wrapped.get_preferred_device_id() == std::optional<int>{4});

  partitioned_operator_data const unpinned(no_batches(), 0, partition_placement::unpinned(1));
  REQUIRE(unpinned.get_partition_idx() == std::optional<std::size_t>{0});
  REQUIRE_FALSE(unpinned.get_preferred_device_id().has_value());

  REQUIRE_THROWS_AS(partitioned_operator_data(no_batches(), 3, placement), std::invalid_argument);

  // Index-less data is placed by locality and carries no preference.
  partitioned_operator_data const locality(no_batches());
  REQUIRE_FALSE(locality.get_partition_idx().has_value());
  REQUIRE_FALSE(locality.get_preferred_device_id().has_value());
}
