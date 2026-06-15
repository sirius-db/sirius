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

#include "scan_manager/round_robin_strategy.hpp"

#include "op/sirius_physical_operator.hpp"

#include <utility>

namespace sirius::scan_manager {

round_robin_strategy::round_robin_strategy(std::vector<int> device_ids)
  : _device_ids(std::move(device_ids))
{
}

int round_robin_strategy::get_next_gpu(std::size_t pipeline_id,
                                       [[maybe_unused]] const op::operator_data* data,
                                       [[maybe_unused]] balancing_strategy::device_id_hint hint)
{
  if (_device_ids.empty()) { return -1; }
  auto const idx   = _cursor.fetch_add(1, std::memory_order_relaxed) % _device_ids.size();
  int const device = _device_ids[idx];
  return device;
}

}  // namespace sirius::scan_manager
