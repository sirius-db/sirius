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

#include <cucascade/memory/common.hpp>

#include <cstddef>
#include <variant>

namespace sirius::op {
class operator_data;
}  // namespace sirius::op

namespace sirius::scan_manager {

/**
 * @brief Policy for placing a scan split onto a GPU.
 *
 * Split providers consult a balancing_strategy as they emit splits, so the
 * choice of which GPU a split's task should run on is decoupled from how
 * splits are produced. Implementations pick a device and record it on the
 * split via @c op::operator_data::set_preferred_device_id; the task creator
 * later reads it back and forwards it onto the pipeline task so the scheduler
 * dispatches the task to that GPU.
 */
class balancing_strategy {
 public:
  virtual ~balancing_strategy() = default;

  struct gpu_id_hint {
    explicit gpu_id_hint(int id) : device_id(id) {}
    int device_id;
  };
  struct numa_id_hint {
    explicit numa_id_hint(int id) : numa_id(id) {}
    int numa_id;
  };
  using device_id_hint = std::variant<std::monostate, gpu_id_hint, numa_id_hint>;

  static device_id_hint make_target_hint(cucascade::memory::memory_space_id id)
  {
    if (id.tier == cucascade::memory::Tier::GPU) { return gpu_id_hint{id.device_id}; }
    if (id.tier == cucascade::memory::Tier::HOST) { return numa_id_hint{id.device_id}; }
    return std::monostate{};
  }

  /**
   * @brief Choose a GPU for @p data and record it on @p data.
   *
   * @param pipeline_id  The pipeline the split belongs to. Strategies that
   *                     balance per pipeline can key off it; global strategies
   *                     (e.g. round-robin) may ignore it.
   * @param data         The split's operating data; the chosen device is
   *                     stamped onto it via @c set_preferred_device_id.
   * @param hint         Optional hint for device selection, e.g., a preferred GPU or NUMA node.
   * @return The chosen device id, or -1 when no device could be assigned (e.g.
   *         the strategy has no GPUs to place onto), in which case @p data is
   *         left unchanged.
   */
  virtual int get_next_gpu(std::size_t pipeline_id,
                           const op::operator_data* data = nullptr,
                           device_id_hint hint           = {}) = 0;
};

}  // namespace sirius::scan_manager
