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

#include "op/sirius_physical_operator.hpp"

#include <cucascade/memory/memory_space.hpp>

#include <stdexcept>

namespace sirius::op {

/**
 * @brief Task input for GPU work that owns no upstream batches.
 *
 * @c prepare_for_processing captures the reservation's memory space for output allocation. This
 * descriptor remains distinct from @c pipelineable_operator_data, whose empty batch list means no
 * task can be created. Operators supply their own estimate and output semantics.
 */
class input_free_operator_data final : public operator_data {
 public:
  explicit input_free_operator_data(std::size_t estimated_bytes) : _estimated_bytes(estimated_bytes)
  {
  }

  [[nodiscard]] operator_data_type get_type() const override
  {
    return operator_data_type::INPUT_FREE;
  }

  using operator_data::prepare_for_processing;
  void prepare_for_processing(cucascade::memory::memory_space const* requested_memory_space,
                              ::cuda::stream_ref,
                              const telemetry::batch_telemetry_info& /*clone_telemetry*/) override
  {
    _gpu_memory_space = const_cast<cucascade::memory::memory_space*>(requested_memory_space);
  }

  [[nodiscard]] std::size_t get_estimated_size_in_bytes() const override
  {
    return _estimated_bytes;
  }

  /**
   * @brief Access the GPU space captured during task preparation.
   *
   * @throws std::runtime_error if preparation has not supplied a GPU space.
   */
  [[nodiscard]] cucascade::memory::memory_space& get_gpu_memory_space() const
  {
    if (_gpu_memory_space == nullptr ||
        _gpu_memory_space->get_tier() != cucascade::memory::Tier::GPU) {
      throw std::runtime_error("Input-free task requires preparation in its reserved GPU space");
    }
    return *_gpu_memory_space;
  }

 private:
  cucascade::memory::memory_space* _gpu_memory_space = nullptr;
  std::size_t _estimated_bytes;
};

}  // namespace sirius::op
