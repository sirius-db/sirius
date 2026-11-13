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

#include "log/logging.hpp"
#include "memory/size_arithmetic.hpp"
#include "op/dynamic_filter/hash_join_partition_observer.hpp"
#include "op/partition/build_fold.hpp"
#include "op/partition/partition_exchange.hpp"
#include "op/sirius_physical_concat.hpp"
#include "op/sirius_physical_dense_count_join.hpp"
#include "op/sirius_physical_grouped_aggregate_merge.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_nested_loop_join.hpp"
#include "op/sirius_physical_partition.hpp"
#include "sirius/exception.hpp"

#include <algorithm>
#include <string>

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

/// One `[build_fold]` line per decision, so measurements can be attributed to each join's fold.
void log_build_fold(sirius_physical_hash_join const& join,
                    build_fold_inputs const& inputs,
                    bool publishes_filters,
                    bool fold)
{
  SIRIUS_LOG_DEBUG(
    "[build_fold] v=1 sig={:016x} join={} policy={} mode={} rows={} resident_bytes={} "
    "concat_batch_bytes={} publishes_filters={} fold={}",
    join.dynamic_filter_signature,
    join.get_operator_id(),
    to_string(join.build_fold_policy),
    inputs.mode == HASH_JOIN_MODE::MIXED_JOIN ? "MIXED_JOIN" : "STANDARD",
    inputs.rows ? std::to_string(*inputs.rows) : std::string{"-"},
    inputs.resident_bytes,
    inputs.concat_batch_bytes,
    publishes_filters,
    fold);
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
  join.note_build_observer_attached();
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
                                sizing.rows,
                                sizing.measured_rows,
                                sizing.resident_bytes};
      auto strategy = empty ? partition_strategy{1, false, false, partition_placement::unpinned(1)}
                            : join.get_partition_strategy(in);
      if (strategy.build_probe) {
        if (!build_concat) { throw std::logic_error("BUILD_PROBE requires a bound build CONCAT"); }
        build_concat->set_concat_all(true);
      } else if (build_concat != nullptr && !empty && in.is_build_side &&
                 strategy.num_partitions == 1) {
        build_fold_inputs const inputs{.rows           = in.measured_rows,
                                       .resident_bytes = in.resident_bytes.value_or(in.total_bytes),
                                       .concat_batch_bytes = build_concat->concat_batch_bytes(),
                                       .num_gpus           = join.num_gpus(),
                                       .mode               = join.join_mode()};
        bool const publishes_filters = join.publishes_dynamic_filters();
        bool const fold =
          decide_build_fold(inputs,
                            {.max_resident_bytes = join.max_build_probe_resident_bytes},
                            join.build_fold_policy,
                            publishes_filters);
        log_build_fold(join, inputs, publishes_filters, fold);
        if (fold) { build_concat->set_concat_all(true); }
      }
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
