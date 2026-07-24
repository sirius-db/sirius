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

/// How a batch placement entered the port/task model, recorded on the
/// placement's registration event.
enum class batch_origin : uint8_t {
  /// A producing operator pushed the batch into a consumer port's repository.
  operator_output,
  /// A partition consumer operator received one partitioned piece.
  partition_output,
  /// First seen when a task claimed it: an intermediate released by an OOM
  /// reschedule (or otherwise outside the port model), registered lazily.
  reschedule_intermediate,
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

/// Why a batch placement left the model, recorded on the consumed event.
enum class batch_consumed_reason : uint8_t {
  /// The claiming task computed on the batch and released it.
  processed,
  /// The claiming task was destroyed before computing (error paths).
  task_failed,
  /// Query-end drain of placements still sitting in repositories.
  query_end,
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

/// Process-global registry emitting Batch placement telemetry.
///
/// A *placement* is one physical data batch published to one consuming
/// pipeline's input port repository. Fan-out (the same batch pushed to several
/// consumer ports) yields one placement per consumer; all placements of a
/// batch share the engine's process-unique batch id. Placements advance
/// through the Batch FSM (registered -> queued -> packaged -> processing ->
/// consumed) with memory-tier usages, and tier changes (downgrade/spill or
/// prepare-time upgrade) re-emit the current state with the new tier.
///
/// Thread safety: placements are sharded by batch id; consumer-port mappings
/// use a shared mutex. Methods that read a batch (`on_published`,
/// `on_packaged`, `on_processing`) take the batch's shared read lock
/// themselves *before* touching registry state — never call them with the
/// same batch exclusively (mutably) locked. `on_tier_change` takes plain
/// values for exactly that reason: it is called from conversion paths that
/// hold the exclusive lock.
///
/// Every method no-ops when the registry is not installed (telemetry or batch
/// events disabled).
class batch_telemetry_registry {
 public:
  static batch_telemetry_registry& instance();

  /// Enable batch telemetry: creates the "GPU"/"HOST"/"DISK" MemoryTier
  /// resources (capacity = total configured bytes per tier) and retains the
  /// telemetry context. Called once at SiriusContext initialization.
  void install(std::shared_ptr<const telemetry_context> context,
               sirius::memory::sirius_memory_reservation_manager& memory_manager);

  /// Drain any leftover placements, drop the MemoryTier resources, release the
  /// telemetry context, and disable. Called at SiriusContext teardown.
  void uninstall();

  /// Associate a consumer port's data repository with its pipeline (== quent
  /// operator id) and receiving-port uuid. Called from emit_plan_telemetry
  /// during query construction; mappings are cleared by on_query_end.
  void register_consumer_port(const cucascade::shared_data_repository* repo,
                              uuid::UUID pipeline_uuid,
                              uuid::UUID port_uuid);

  /// A producer published `batch` into `repo`: emits registered -> queued for
  /// a new placement on the repo's consumer pipeline. Call *before* the batch
  /// is added to the repository so `queued` always precedes `packaged`.
  void on_published(const std::shared_ptr<cucascade::data_batch>& batch,
                    const cucascade::shared_data_repository* repo,
                    batch_origin origin);

  /// A task claimed `batch` as input: queued -> packaged on the matching
  /// placement. Re-claims after an OOM reschedule re-emit packaged with the
  /// new task; batches never seen before (reschedule intermediates) are
  /// lazily registered.
  void on_packaged(const std::shared_ptr<cucascade::data_batch>& batch,
                   uuid::UUID consumer_pipeline_uuid,
                   uuid::UUID task_uuid);

  /// The claiming task finished preparing and started computing:
  /// packaged -> processing (tier re-read to capture prepare-time upgrades).
  void on_processing(const std::shared_ptr<cucascade::data_batch>& batch, uuid::UUID task_uuid);

  /// Same as on_processing for batches already released by prepare (merge/
  /// concat inputs are consumed while materializing): transitions the
  /// placement with its last recorded tier/bytes instead of re-reading.
  void on_processing_by_id(uint64_t batch_id, uuid::UUID task_uuid);

  /// The claiming task is done with the batch: -> consumed + exit. Placements
  /// re-claimed by another task (OOM reschedule) are left untouched.
  void on_consumed(uint64_t batch_id, uuid::UUID task_uuid);

  /// The batch's data moved to another memory tier: re-emit every live
  /// placement's current state with the new tier usage. Takes values, not the
  /// batch — callers hold the batch's exclusive lock. `device_id` selects the
  /// per-device GPU tier resource; it is ignored for HOST/DISK.
  void on_tier_change(uint64_t batch_id,
                      cucascade::memory::Tier tier,
                      int32_t device_id,
                      uint64_t bytes);

  /// Drain all remaining placements as consumed{"query_end"} and clear the
  /// consumer-port mappings. Called from SiriusContext::QueryEnd.
  void on_query_end();

  /// The MemoryTier resource for (tier, device), e.g. for task states that
  /// report their processing-space reservation on the same tier resources
  /// batches use. Nil when the registry is not installed.
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
