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

#include "log/logging.hpp"
#include "telemetry-bridge/gen/data_batch.rs.h"
#include "telemetry-bridge/gen/memory.rs.h"
#include "telemetry-bridge/gen/uuid.rs.h"
#include "telemetry/memory_context.hpp"
#include "telemetry/telemetry_context.hpp"

#include <cucascade/data/data_batch.hpp>

#include <functional>
#include <memory>
#include <optional>

namespace sirius::telemetry {

//! Telemetry attribution for a newly-created data_batch: the telemetry context
//! to emit events into and the pipeline that produced the batch. Passed to the
//! data_batch factories / quent_data_batch_probe::create. A null @ref context
//! yields the no-op base probe, so batch construction stays telemetry-free (e.g.
//! in unit tests, or when pipelines are built without an engine).
struct batch_telemetry_info {
  const telemetry_context* context = nullptr;
  uuid::UUID producer_pipeline_uuid{};
};

/**
 * @brief Implementation of idata_batch_probe that forwards data_batch events to a quent
 * telemetry observer.
 *
 * Usage:
 * @code
 *   auto probe = std::make_unique<quent_data_batch_probe>();
 *   auto batch = std::make_shared<cucascade::data_batch>(batch_id, std::move(data),
 * std::move(probe));
 * @endcode
 */
class quent_data_batch_probe : public cucascade::idata_batch_probe {
 public:
  /**
   * @brief Build  a probe that forwards state transitions to a telemetry observer.
   *
   * Creates its own DataBatchObserver from the given telemetry_context.
   * The memory resource UUID for idle events is resolved at runtime based on the
   * data's current memory tier.
   *
   * @param telemetry_info Telemetry attribution (context + producing pipeline).
   * @param batch_id The unique id of the data_batch this probe is attached to.
   *
   * @note When @p telemetry_info.context is non-null, returns a
   * quent_data_batch_probe that forwards the batch's lifecycle to telemetry. When
   * null (e.g. in unit tests that don't set up a telemetry context), returns the
   * no-op base probe so batch construction stays telemetry-free.
   */
  static std::unique_ptr<cucascade::idata_batch_probe> create(
    const batch_telemetry_info& telemetry_info, const uint64_t batch_id)
  {
    if (telemetry_info.context == nullptr) {
      return std::make_unique<cucascade::idata_batch_probe>();
    }
    return std::unique_ptr<quent_data_batch_probe>(new quent_data_batch_probe(
      *(telemetry_info.context), batch_id, telemetry_info.producer_pipeline_uuid));
  }

  ~quent_data_batch_probe() override
  {
    handle_->destructed();
    handle_->exit();
  }

  void created([[maybe_unused]] const uint64_t batch_id,
               [[maybe_unused]] const cucascade::idata_representation& data) noexcept override
  {
    try {
      auto maybe_memory_handle =
        memory_context_->get_memory_handle(data.get_memory_space().get_id());
      if (not maybe_memory_handle) {
        SIRIUS_LOG_WARN("No quent memory handle found for {}", data.get_memory_space().to_string());
        return;
      }

      handle_->stationary({
        .memory_resource_id    = (*maybe_memory_handle).get().uuid(),
        .memory_capacity_bytes = data.get_size_in_bytes(),
      });
    } catch (...) {
    }
  }

  void conversion_started(
    [[maybe_unused]] const cucascade::idata_representation& current_data,
    [[maybe_unused]] const cucascade::memory::memory_space* target_memory_space) noexcept override
  {
    try {
      const size_t data_size = current_data.get_size_in_bytes();

      auto maybe_source_memory_handle =
        memory_context_->get_memory_handle(current_data.get_memory_space().get_id());
      if (not maybe_source_memory_handle) {
        SIRIUS_LOG_WARN("No quent memory handle found for {}",
                        current_data.get_memory_space().to_string());
        return;
      }

      auto maybe_dest_memory_handle =
        memory_context_->get_memory_handle(target_memory_space->get_id());
      if (not maybe_dest_memory_handle) {
        SIRIUS_LOG_WARN("No quent memory handle found for {}", target_memory_space->to_string());
        return;
      }

      auto maybe_channel_handle = memory_context_->get_channel_handle(
        current_data.get_memory_space().get_id(), target_memory_space->get_id());
      if (not maybe_channel_handle) {
        SIRIUS_LOG_WARN("No quent channel handle found for the channel between {} and {}",
                        current_data.get_memory_space().to_string(),
                        target_memory_space->to_string());
        return;
      }

      handle_->in_transit({
        .source_memory_resource_id    = (*maybe_source_memory_handle).get().uuid(),
        .source_memory_capacity_bytes = data_size,
        .dest_memory_resource_id      = (*maybe_dest_memory_handle).get().uuid(),
        .dest_memory_capacity_bytes   = data_size,
        .channel_resource_id          = (*maybe_channel_handle).get().uuid(),
        .channel_capacity_bytes       = data_size,
      });
    } catch (...) {
    }
  }

  void conversion_completed([[maybe_unused]] const cucascade::idata_representation& data,
                            [[maybe_unused]] const bool success) noexcept override
  {
    try {
      auto maybe_memory_handle =
        memory_context_->get_memory_handle(data.get_memory_space().get_id());
      if (not maybe_memory_handle) {
        SIRIUS_LOG_WARN("No quent memory handle found for {}", data.get_memory_space().to_string());
        return;
      }

      handle_->stationary({
        .memory_resource_id    = (*maybe_memory_handle).get().uuid(),
        .memory_capacity_bytes = data.get_size_in_bytes(),
      });
    } catch (...) {
    }
  }

  void data_replaced(
    [[maybe_unused]] const cucascade::idata_representation& new_data) noexcept override
  {
    try {
      auto maybe_memory_handle =
        memory_context_->get_memory_handle(new_data.get_memory_space().get_id());
      if (not maybe_memory_handle) {
        SIRIUS_LOG_WARN("No quent memory handle found for {}",
                        new_data.get_memory_space().to_string());
        return;
      }
      handle_->stationary({
        .memory_resource_id    = (*maybe_memory_handle).get().uuid(),
        .memory_capacity_bytes = new_data.get_size_in_bytes(),
      });
    } catch (...) {
    }
  }

 private:
  /**
   * @brief Construct a probe that forwards state transitions to a telemetry observer.
   *
   * Creates its own DataBatchObserver from the given telemetry_context.
   * The memory resource UUID for idle events is resolved at runtime based on the
   * data's current memory tier.
   *
   * @param ctx The telemetry context providing instrumentation and memory UUIDs.
   * @param producer_pipeline_uuid UUID of the pipeline that produced this batch.
   */
  quent_data_batch_probe(const telemetry_context& ctx,
                         const uint64_t batch_id,
                         uuid::UUID producer_pipeline_uuid)
    : handle_(quent::data_batch::create(ctx.context(),
                                        {
                                          .instance_name          = "batch",
                                          .data_batch_id          = batch_id,
                                          .producer_pipeline_uuid = producer_pipeline_uuid,
                                        })),
      memory_context_(ctx.get_memory_context())
  {
  }
  rust::Box<quent::data_batch::DataBatchHandle> handle_;
  std::shared_ptr<const memory_context> memory_context_;
};

}  // namespace sirius::telemetry
