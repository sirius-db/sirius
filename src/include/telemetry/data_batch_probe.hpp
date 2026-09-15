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

#pragma once

#include "telemetry/runtime_fsm_handle.hpp"

#include <cucascade/data/data_batch.hpp>

#include <cstdint>
#include <memory>
#include <optional>

namespace sirius::telemetry {

class memory_context;
class telemetry_context;

/// Identifies the telemetry context and producing pipeline for a data batch.
/// A null context disables telemetry for the batch.
struct batch_telemetry_info {
  const telemetry_context* context = nullptr;
  quent::Uuid producer_pipeline_uuid{};
};

/// Emits a data batch's lifecycle through Quent.
class quent_data_batch_probe final : public cucascade::idata_batch_probe {
 public:
  /// Creates a Quent probe, or a no-op probe when telemetry has no context.
  /// Memory and channel attribution is resolved from the batch representation at each callback.
  static std::unique_ptr<cucascade::idata_batch_probe> create(
    const batch_telemetry_info& telemetry_info, uint64_t batch_id);

  ~quent_data_batch_probe() noexcept override;

  void created(uint64_t batch_id, const cucascade::idata_representation& data) noexcept override;
  void conversion_started(
    const cucascade::idata_representation& current_data,
    const cucascade::memory::memory_space* target_memory_space) noexcept override;
  void conversion_completed(const cucascade::idata_representation& data,
                            bool success) noexcept override;
  void data_replaced(const cucascade::idata_representation& new_data) noexcept override;

 private:
  using runtime_handle = runtime_fsm_handle<quent::DataBatch,
                                            quent::data_batch_state::Constructed,
                                            quent::data_batch_state::Stationary,
                                            quent::data_batch_state::InTransit,
                                            quent::data_batch_state::Destructed>;

  quent_data_batch_probe(const telemetry_context& context,
                         uint64_t batch_id,
                         quent::Uuid producer_pipeline_uuid);

  void stationary(const cucascade::idata_representation& data);
  void stationary(std::optional<quent::refs::MemoryUsageRef> memory);
  void in_transit(std::optional<quent::refs::MemoryUsageRef> source_memory,
                  std::optional<quent::refs::MemoryUsageRef> destination_memory,
                  std::optional<quent::refs::ChannelUsageRef> channel);

  uint64_t batch_id_;
  runtime_handle handle_;
  std::shared_ptr<const memory_context> memory_context_;
};

}  // namespace sirius::telemetry
