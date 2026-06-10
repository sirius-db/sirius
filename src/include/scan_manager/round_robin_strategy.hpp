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

#pragma once

#include "scan_manager/balancing_strategy.hpp"

#include <atomic>
#include <cstdint>
#include <vector>

namespace sirius::scan_manager {

/**
 * @brief Round-robin placement across a fixed set of GPUs.
 *
 * Hands out devices from @c _device_ids in turn via a single atomic cursor, so
 * splits spread evenly across all GPUs. The cursor is shared by every provider
 * that holds the same strategy instance, making the walk continuous across the
 * whole scan stage rather than restarting per provider. @c pipeline_id is
 * ignored — placement is global.
 */
class round_robin_strategy : public balancing_strategy {
 public:
  /// @param device_ids GPUs to round-robin over, in a stable order. An empty
  ///                   set turns @ref get_next_gpu into a no-op returning -1.
  explicit round_robin_strategy(std::vector<int> device_ids);

  int get_next_gpu(std::size_t pipeline_id,
                   const op::operator_data* data,
                   device_id_hint hint) override;

 private:
  std::vector<int> _device_ids;
  std::atomic<std::uint64_t> _cursor{0};
};

}  // namespace sirius::scan_manager
