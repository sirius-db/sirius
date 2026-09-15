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

#include "telemetry/data_batch_probe.hpp"

#include "log/logging.hpp"
#include "telemetry/memory_context.hpp"
#include "telemetry/telemetry_context.hpp"

#include <cucascade/memory/memory_space.hpp>

#include <exception>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace sirius::telemetry {
namespace {

namespace data_batch_state = quent::data_batch_state;

std::optional<quent::refs::MemoryUsageRef> memory_usage(const memory_handle& memory, uint64_t bytes)
{
  return quent::refs::MemoryUsageRef{
    .target = memory.id(),
    .data   = quent::records::MemoryUsage{.bytes = bytes},
  };
}

std::optional<quent::refs::ChannelUsageRef> channel_usage(const channel_handle& channel,
                                                          uint64_t bytes)
{
  return quent::refs::ChannelUsageRef{
    .target = channel.id(),
    .data   = quent::records::ChannelUsage{.bytes = bytes},
  };
}

void log_probe_failure(uint64_t batch_id,
                       std::string_view operation,
                       const char* error = nullptr) noexcept
{
  // The probe callbacks and destructor are noexcept. A logging failure cannot escape while
  // reporting the original telemetry failure.
  try {
    if (error != nullptr) {
      SIRIUS_LOG_WARN(
        "Failed to {} DataBatch telemetry for batch {}: {}", operation, batch_id, error);
    } else {
      SIRIUS_LOG_WARN(
        "Failed to {} DataBatch telemetry for batch {}: unknown exception", operation, batch_id);
    }
  } catch (...) {
  }
}

// Missing-resource diagnostics must not prevent the state transition they qualify.
void log_missing_memory_handle(uint64_t batch_id,
                               const cucascade::memory::memory_space& memory_space) noexcept
{
  try {
    SIRIUS_LOG_WARN("No Quent memory handle found for {} while updating batch {}",
                    memory_space.to_string(),
                    batch_id);
  } catch (...) {
  }
}

void log_missing_channel_handle(uint64_t batch_id,
                                const cucascade::memory::memory_space& source,
                                const cucascade::memory::memory_space& destination) noexcept
{
  try {
    SIRIUS_LOG_WARN("No Quent channel handle found between {} and {} while updating batch {}",
                    source.to_string(),
                    destination.to_string(),
                    batch_id);
  } catch (...) {
  }
}

template <typename Operation>
void run_probe_operation(uint64_t batch_id,
                         std::string_view operation_name,
                         Operation&& operation) noexcept
{
  try {
    std::forward<Operation>(operation)();
  } catch (const std::exception& error) {
    log_probe_failure(batch_id, operation_name, error.what());
  } catch (...) {
    log_probe_failure(batch_id, operation_name);
  }
}

}  // namespace

std::unique_ptr<cucascade::idata_batch_probe> quent_data_batch_probe::create(
  const batch_telemetry_info& telemetry_info, uint64_t batch_id)
{
  if (telemetry_info.context == nullptr) {
    return std::make_unique<cucascade::idata_batch_probe>();
  }
  return std::unique_ptr<quent_data_batch_probe>(new quent_data_batch_probe(
    *telemetry_info.context, batch_id, telemetry_info.producer_pipeline_uuid));
}

quent_data_batch_probe::quent_data_batch_probe(const telemetry_context& context,
                                               uint64_t batch_id,
                                               quent::Uuid producer_pipeline_uuid)
  : batch_id_(batch_id),
    handle_(
      context.context().data_batch_observer()->handle().constructed(quent::data_batch::Constructed{
        .instance_name          = "batch",
        .data_batch_id          = batch_id,
        .producer_pipeline_uuid = quent::operator_::OperatorId(producer_pipeline_uuid),
      })),
    memory_context_(context.get_memory_context())
{
}

quent_data_batch_probe::~quent_data_batch_probe() noexcept
{
  run_probe_operation(batch_id_, "finalize", [this] {
    if (handle_.holds<data_batch_state::Destructed>()) { return; }

    auto stationary_without_memory = [](auto&& current) {
      return std::move(current).stationary(quent::data_batch::Stationary{
        .memory = std::nullopt,
      });
    };
    if (!handle_.holds<data_batch_state::Stationary>() &&
        !handle_.holds<data_batch_state::Destructed>()) {
      if (!handle_.transition<data_batch_state::Constructed, data_batch_state::InTransit>(
            stationary_without_memory)) {
        throw std::logic_error("invalid Quent DataBatch state during finalization");
      }
    }

    if (handle_.holds<data_batch_state::Stationary>() &&
        !handle_.transition<data_batch_state::Stationary>(
          [](auto&& current) { return std::move(current).destructed(); })) {
      throw std::logic_error("invalid Quent DataBatch state during finalization");
    }
  });
}

void quent_data_batch_probe::created(uint64_t, const cucascade::idata_representation& data) noexcept
{
  run_probe_operation(batch_id_, "emit initial stationary", [this, &data] { stationary(data); });
}

void quent_data_batch_probe::conversion_started(
  const cucascade::idata_representation& current_data,
  const cucascade::memory::memory_space* target_memory_space) noexcept
{
  run_probe_operation(batch_id_, "emit in-transit", [this, &current_data, target_memory_space] {
    const auto data_size = current_data.get_size_in_bytes();
    const auto source_id = current_data.get_memory_space().get_id();
    const auto target_id = target_memory_space->get_id();

    auto source_memory = memory_context_->get_memory_handle(source_id);
    if (!source_memory) { log_missing_memory_handle(batch_id_, current_data.get_memory_space()); }

    auto destination_memory = memory_context_->get_memory_handle(target_id);
    if (!destination_memory) { log_missing_memory_handle(batch_id_, *target_memory_space); }

    auto channel = memory_context_->get_channel_handle(source_id, target_id);
    if (!channel) {
      log_missing_channel_handle(batch_id_, current_data.get_memory_space(), *target_memory_space);
    }

    in_transit(
      source_memory ? memory_usage(source_memory->get(), data_size) : std::nullopt,
      destination_memory ? memory_usage(destination_memory->get(), data_size) : std::nullopt,
      channel ? channel_usage(channel->get(), data_size) : std::nullopt);
  });
}

void quent_data_batch_probe::conversion_completed(const cucascade::idata_representation& data,
                                                  bool) noexcept
{
  run_probe_operation(batch_id_, "emit stationary", [this, &data] { stationary(data); });
}

void quent_data_batch_probe::data_replaced(const cucascade::idata_representation& new_data) noexcept
{
  run_probe_operation(batch_id_, "emit replacement", [this, &new_data] { stationary(new_data); });
}

void quent_data_batch_probe::stationary(const cucascade::idata_representation& data)
{
  auto memory = memory_context_->get_memory_handle(data.get_memory_space().get_id());
  if (!memory) {
    log_missing_memory_handle(batch_id_, data.get_memory_space());
    stationary(std::nullopt);
    return;
  }
  stationary(memory_usage(memory->get(), data.get_size_in_bytes()));
}

void quent_data_batch_probe::stationary(std::optional<quent::refs::MemoryUsageRef> memory)
{
  auto transition = [&memory](auto&& current) {
    return std::move(current).stationary(quent::data_batch::Stationary{
      .memory = std::move(memory),
    });
  };
  if (handle_.transition<data_batch_state::Constructed,
                         data_batch_state::Stationary,
                         data_batch_state::InTransit>(transition)) {
    return;
  }
  throw std::logic_error("invalid Quent DataBatch transition to stationary");
}

void quent_data_batch_probe::in_transit(
  std::optional<quent::refs::MemoryUsageRef> source_memory,
  std::optional<quent::refs::MemoryUsageRef> destination_memory,
  std::optional<quent::refs::ChannelUsageRef> channel)
{
  if (!handle_.transition<data_batch_state::Stationary>([&](auto&& current) {
        return std::move(current).in_transit(quent::data_batch::InTransit{
          .source_memory = std::move(source_memory),
          .dest_memory   = std::move(destination_memory),
          .channel       = std::move(channel),
        });
      })) {
    throw std::logic_error("invalid Quent DataBatch transition to in_transit");
  }
}

}  // namespace sirius::telemetry
