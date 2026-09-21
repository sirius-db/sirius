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

#include "telemetry-bridge/gen/uuid.rs.h"

#include <cucascade/memory/common.hpp>

#include <cstdint>
#include <memory>
#include <string_view>

namespace cucascade {
class data_batch;
class data_repository;
using shared_data_repository = data_repository;
}  // namespace cucascade

namespace sirius::memory {
class sirius_memory_reservation_manager;
}  // namespace sirius::memory

namespace sirius::telemetry {

class telemetry_context;

/// How a batch placement entered the port/task model.
enum class batch_origin : uint8_t {
  operator_output,          ///< pushed into a consumer port's repository
  partition_output,         ///< received by a partition consumer operator
  reschedule_intermediate,  ///< lazily registered at task claim
};

constexpr std::string_view to_string_view(batch_origin origin)
{
  switch (origin) {
    case batch_origin::operator_output: return "operator_output";
    case batch_origin::partition_output: return "partition_output";
    case batch_origin::reschedule_intermediate: return "reschedule_intermediate";
  }
  return "unknown";
}

/// Why a batch placement left the model.
enum class batch_consumed_reason : uint8_t {
  processed,    ///< the claiming task computed on the batch
  task_failed,  ///< the claiming task died before computing
  query_end,    ///< drained at query end
};

constexpr std::string_view to_string_view(batch_consumed_reason reason)
{
  switch (reason) {
    case batch_consumed_reason::processed: return "processed";
    case batch_consumed_reason::task_failed: return "task_failed";
    case batch_consumed_reason::query_end: return "query_end";
  }
  return "unknown";
}

/// Process-global registry emitting BatchPlacement telemetry. A placement is
/// one physical data batch published to one consuming pipeline's input port;
/// fan-out yields one placement per consumer.
///
/// Methods taking a batch acquire its shared read lock — never call them with
/// the batch exclusively locked (`on_tier_change` takes plain values for such
/// callers). Every method no-ops when the registry is not installed.
class batch_telemetry_registry {
 public:
  static batch_telemetry_registry& instance();

  /// Enable batch telemetry: create the per-tier MemoryTier resources and
  /// retain the telemetry context.
  void install(std::shared_ptr<const telemetry_context> context,
               sirius::memory::sirius_memory_reservation_manager& memory_manager);

  /// Drain leftover placements, drop the MemoryTier resources, and disable.
  void uninstall();

  /// Associate a consumer port's data repository with its pipeline and port.
  void register_consumer_port(const cucascade::shared_data_repository* repo,
                              uuid::UUID pipeline_uuid,
                              uuid::UUID port_uuid);

  /// A producer published `batch` into `repo`: registered -> queued. Call
  /// before the batch is added to the repository.
  void on_published(const std::shared_ptr<cucascade::data_batch>& batch,
                    const cucascade::shared_data_repository* repo,
                    batch_origin origin);

  /// A task claimed `batch` as input: queued -> packaged (re-claims re-emit
  /// packaged; unseen batches are lazily registered).
  void on_packaged(const std::shared_ptr<cucascade::data_batch>& batch,
                   uuid::UUID consumer_pipeline_uuid,
                   uuid::UUID task_uuid);

  /// The claiming task started computing: packaged -> processing.
  void on_processing(const std::shared_ptr<cucascade::data_batch>& batch, uuid::UUID task_uuid);

  /// on_processing for batches already released by prepare, using the last
  /// recorded tier/bytes.
  void on_processing_by_id(uint64_t batch_id, uuid::UUID task_uuid);

  /// The claiming task is done with the batch: -> consumed + exit.
  void on_consumed(uint64_t batch_id, uuid::UUID task_uuid);

  /// The batch's data moved to another tier: re-emit every live placement's
  /// state with the new tier usage. Callers may hold the batch's lock.
  void on_tier_change(uint64_t batch_id,
                      cucascade::memory::Tier tier,
                      int32_t device_id,
                      uint64_t bytes);

  /// Drain all remaining placements and clear the consumer-port mappings.
  void on_query_end();

  /// The MemoryTier resource for (tier, device); nil when not installed.
  [[nodiscard]] uuid::UUID tier_resource(cucascade::memory::Tier tier, int32_t device_id) const;

  batch_telemetry_registry(const batch_telemetry_registry&)            = delete;
  batch_telemetry_registry& operator=(const batch_telemetry_registry&) = delete;
  batch_telemetry_registry(batch_telemetry_registry&&)                 = delete;
  batch_telemetry_registry& operator=(batch_telemetry_registry&&)      = delete;

 private:
  batch_telemetry_registry();
  ~batch_telemetry_registry();

  struct impl;
  std::unique_ptr<impl> impl_;
};

}  // namespace sirius::telemetry
