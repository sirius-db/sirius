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

#include "scan_manager/split_provider.hpp"

#include "op/sirius_physical_operator.hpp"
#include "scan_manager/balancing_strategy.hpp"
#include "scan_manager/split_connector.hpp"

#include <utility>

namespace sirius::scan_manager {

void split_provider::push_to_connector(split_connector& connector,
                                       std::unique_ptr<op::operator_data> split)
{
  connector.push_split(std::move(split));
}

void split_provider::apply_balancing(op::operator_data& split)
{
  if (!_balancing_strategy) { return; }
  // Resident splits already live on a specific GPU; let downstream data
  // locality decide their placement instead of overriding it here. Also
  // respect a preference an upstream producer already set.
  if (split.is_resident() || split.get_preferred_device_id().has_value()) { return; }
  auto const device_id = _balancing_strategy->get_next_gpu(_pipeline_id, &split);
  if (device_id >= 0) { split.set_preferred_device_id(device_id); }
}

}  // namespace sirius::scan_manager
