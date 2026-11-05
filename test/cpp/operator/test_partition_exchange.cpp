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

#include "op/sirius_physical_partition.hpp"

#include <catch.hpp>

#include <atomic>
#include <future>

using namespace sirius::op;

namespace {
auto one_column()
{
  return duckdb::vector<sirius::logical_type>{sirius::logical_type::make(sirius::type_id::INTEGER)};
}

void attach(sirius_physical_operator& op, cucascade::shared_data_repository& repo)
{
  auto port  = std::make_unique<sirius_physical_operator::port>();
  port->repo = &repo;
  port->type = MemoryBarrierType::FULL;
  op.add_port("default", std::move(port));
}
}  // namespace

TEST_CASE("partition output metadata preserves sparse destinations through reconstruction",
          "[partition_exchange][unit]")
{
  // Metadata only: null placeholders are never prepared or executed as batches.
  routed_operator_data output({nullptr, nullptr}, {1, 4});
  output.set_preferred_device_id(5);
  auto reconstructed = output.with_batches_preserving_rows({nullptr, nullptr});
  auto const* routed = dynamic_cast<routed_operator_data const*>(reconstructed.get());
  REQUIRE(routed);
  CHECK(routed->destinations() == std::vector<std::size_t>{1, 4});
  CHECK(routed->get_preferred_device_id() == 5);
  CHECK_THROWS_AS(output.with_batches_preserving_rows({nullptr}), std::invalid_argument);
}

TEST_CASE("concurrent partition requests share one layout decision", "[partition_exchange][unit]")
{
  sirius_physical_partition left(one_column(), 0, partition_spec{{0}});
  sirius_physical_partition right(one_column(), 0, partition_spec{{0}});
  sirius_physical_operator target(SiriusPhysicalOperatorType::PROJECTION, one_column(), 0);
  cucascade::shared_data_repository left_repo, right_repo, output_repo;
  attach(left, left_repo);
  attach(right, right_repo);
  attach(target, output_repo);
  left.add_next_port_after_sink({&target, "default"});
  right.add_next_port_after_sink({&target, "default"});
  std::atomic<int> decisions{0};
  partition_exchange::bind({&left, &right}, 1, false, [&](auto const&, bool) {
    ++decisions;
    return std::vector<partition_layout>(2,
                                         {partition_mode::hash, partition_placement::unpinned(4)});
  });
  auto first  = std::async(std::launch::async, [&] { return left.get_next_task_input_data(); });
  auto second = std::async(std::launch::async, [&] { return right.get_next_task_input_data(); });
  CHECK_FALSE(first.get());
  CHECK_FALSE(second.get());
  CHECK(decisions == 1);
  REQUIRE(left.layout());
  REQUIRE(right.layout());
  CHECK(*left.layout()->placement == *right.layout()->placement);
  CHECK(output_repo.num_partitions() == 4);
}
