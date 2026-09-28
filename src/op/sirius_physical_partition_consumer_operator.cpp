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

}  // namespace op
}  // namespace sirius
