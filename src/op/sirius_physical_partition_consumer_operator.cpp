/*
 * Copyright 2025, Sirius Contributors.
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

#include "op/sirius_physical_partition_consumer_operator.hpp"

#include "sirius/exception.hpp"
#include "telemetry/batch_telemetry.hpp"

namespace sirius {
namespace op {

sirius_physical_partition_consumer_operator::~sirius_physical_partition_consumer_operator() {}

void sirius_physical_partition_consumer_operator::push_data_batch_partitioned(
  std::string_view port_id,
  std::shared_ptr<::cucascade::data_batch> batch,
  std::size_t partition_idx)
{
  auto* p = get_port(port_id);
  if (p && p->repo) {
    telemetry::batch_telemetry_registry::instance().on_published(
      batch, p->repo, telemetry::batch_origin::partition_output);
    p->repo->add_data_batch(batch, partition_idx);
  }
}

partition_strategy sirius_physical_partition_consumer_operator::get_partition_strategy(
  const partition_sizing_input& /*in*/)
{
  throw std::runtime_error(
    "get_partition_strategy called on a non-sizing partition consumer operator " + get_name() +
    " (id " + std::to_string(get_operator_id()) + ")");
}

void sirius_physical_partition_consumer_operator::set_placement(
  std::shared_ptr<const partition_placement> placement)
{
  if (placement == nullptr) {
    throw sirius::internal_exception("set_placement called with a null placement on " + get_name());
  }
  auto expected = _placement.load(std::memory_order_acquire);
  while (expected == nullptr) {
    if (_placement.compare_exchange_weak(expected, placement, std::memory_order_acq_rel)) {
      return;
    }
  }
  if (*expected != *placement) {
    throw sirius::internal_exception("set_placement: " + get_name() + " already has placement " +
                                     expected->to_string() + ", refusing " +
                                     placement->to_string());
  }
}

std::shared_ptr<const partition_placement>
sirius_physical_partition_consumer_operator::require_placement(
  std::size_t fallback_num_partitions) const
{
  if (auto installed = placement()) { return installed; }
  if (_active_gpu_ids.empty()) {
    return std::make_shared<const partition_placement>(
      partition_placement::unpinned(std::max<std::size_t>(1, fallback_num_partitions)));
  }
  throw sirius::internal_exception("require_placement: " + get_name() +
                                   " is about to emit partitioned data but no upstream PARTITION "
                                   "installed a placement");
}

}  // namespace op
}  // namespace sirius
