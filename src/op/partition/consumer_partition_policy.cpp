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

#include "op/partition/consumer_partition_policy.hpp"

#include "memory/size_arithmetic.hpp"
#include "op/dynamic_filter/hash_join_partition_observer.hpp"
#include "op/partition/partition_exchange.hpp"
#include "op/sirius_physical_concat.hpp"
#include "op/sirius_physical_dense_count_join.hpp"
#include "op/sirius_physical_grouped_aggregate_merge.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_nested_loop_join.hpp"
#include "op/sirius_physical_partition.hpp"
#include "sirius/exception.hpp"

#include <algorithm>

namespace sirius::op {
namespace {

void install_placement(sirius_physical_partition_consumer_operator& consumer,
                       partition_strategy const& strategy)
{
  auto const& admitted = consumer.active_gpu_ids();
  for (auto device : strategy.placement.devices()) {
    if (std::find(admitted.begin(), admitted.end(), device) == admitted.end()) {
      throw sirius::internal_exception("partition policy placement names an unadmitted GPU {}",
                                       device);
    }
  }
  consumer.set_placement(std::make_shared<partition_placement const>(strategy.placement));
}

std::vector<partition_layout> hash_layouts(
  std::size_t count, sirius_physical_partition_consumer_operator const& consumer)
{
  return std::vector<partition_layout>(count, {partition_mode::hash, consumer.placement()});
}

}  // namespace

void bind_partition_inputs(sirius_physical_hash_join& join,
                           sirius_physical_partition& build,
                           sirius_physical_partition& probe,
                           sirius_physical_concat* build_concat,
                           bool probe_drives)
{
  build.set_input_barrier(MemoryBarrierType::FULL);
  probe.set_input_barrier(probe_drives ? MemoryBarrierType::FULL : MemoryBarrierType::PARTIAL);
  build.set_input_observer(std::make_unique<hash_join_partition_observer>(join, build));
  partition_exchange::bind(
    {&build, &probe},
    probe_drives ? 1 : 0,
    false,
    [&join, build_concat, probe_drives](auto const& stats, bool empty) {
      auto const& sizing = stats[probe_drives ? 1 : 0];
      partition_sizing_input in{sizing.bytes,
                                !probe_drives,
                                build_concat != nullptr,
                                sirius::memory::saturating_add(stats[0].bytes, stats[1].bytes),
                                sizing.rows};
      auto strategy = empty ? partition_strategy{1, false, false, partition_placement::unpinned(1)}
                            : join.get_partition_strategy(in);
      bool build_arrives_whole = false;
      if (strategy.build_probe) {
        if (!build_concat) { throw std::logic_error("BUILD_PROBE requires a bound build CONCAT"); }
        build_concat->set_concat_all(true);
        build_arrives_whole = strategy.num_partitions == 1 || strategy.broadcast;
      } else if (!empty && in.is_build_side && strategy.num_partitions == 1 &&
                 join.publishes_dynamic_filters() &&
                 in.total_bytes < join.max_build_hash_table_bytes()) {
        if (build_concat) { build_concat->set_concat_all(true); }
        build_arrives_whole = build_concat != nullptr;
      }
      join.set_build_arrives_whole(build_arrives_whole);
      install_placement(join, strategy);
      if (strategy.broadcast) {
        return std::vector<partition_layout>{{partition_mode::broadcast, join.placement()},
                                             {partition_mode::local, join.placement()}};
      }
      return hash_layouts(2, join);
    });
}

void bind_partition_inputs(sirius_physical_nested_loop_join& join,
                           sirius_physical_partition& build,
                           sirius_physical_partition& probe)
{
  build.set_input_barrier(MemoryBarrierType::FULL);
  probe.set_input_barrier(MemoryBarrierType::PARTIAL);
  partition_exchange::bind({&build, &probe}, 0, false, [&join](auto const& stats, bool) {
    auto strategy =
      join.get_partition_strategy({stats[0].bytes,
                                   true,
                                   false,
                                   sirius::memory::saturating_add(stats[0].bytes, stats[1].bytes),
                                   stats[0].rows});
    install_placement(join, strategy);
    return std::vector<partition_layout>(2, {partition_mode::single, join.placement()});
  });
}

void bind_partition_inputs(sirius_physical_dense_count_join& join,
                           sirius_physical_partition& preserved,
                           sirius_physical_partition& counted)
{
  partition_exchange::bind({&preserved, &counted}, 0, true, [&join](auto const& stats, bool empty) {
    auto strategy = empty ? partition_strategy{1, false, false, partition_placement::unpinned(1)}
                          : join.get_partition_strategy(
                              {stats[0].bytes,
                               true,
                               false,
                               sirius::memory::saturating_add(stats[0].bytes, stats[1].bytes),
                               stats[0].rows});
    install_placement(join, strategy);
    return hash_layouts(2, join);
  });
}

void bind_partition_input(sirius_physical_grouped_aggregate_merge& merge,
                          sirius_physical_partition& input)
{
  input.set_input_barrier(input.is_size_estimation_enabled() ? MemoryBarrierType::PARTIAL
                                                             : MemoryBarrierType::FULL);
  partition_exchange::bind({&input}, 0, false, [&merge](auto const& stats, bool) {
    auto strategy =
      merge.get_partition_strategy({stats[0].bytes, false, false, stats[0].bytes, stats[0].rows});
    install_placement(merge, strategy);
    return hash_layouts(1, merge);
  });
}

}  // namespace sirius::op
