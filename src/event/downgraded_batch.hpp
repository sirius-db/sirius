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

#include "exec/queue_priority.hpp"
#include "query_id.hpp"

#include <cucascade/memory/common.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

namespace cucascade {
class data_batch;
}  // namespace cucascade

namespace sirius::event {

/// Where the downgrade executor found the batch it moved.
enum class downgrade_source : std::uint8_t {
  repository,  ///< tier 1: a batch waiting in a data repository
  task_queue   ///< tier 2: an input of a task extracted from the scheduler queue
};

/**
 * @brief Payload of a @c batch_downgraded event: one batch moved to a lower tier.
 *
 * Describes a single successful conversion. A batch spilled GPU->HOST and later
 * HOST->DISK produces two records; @c conversion_seq orders them regardless of
 * the order in which their events are delivered.
 *
 * The batch is held weakly so an event sitting in a subscriber's mailbox never
 * extends its lifetime: a subscriber must expect the pointer to have expired.
 */
struct downgraded_batch {
  /// The batch that was moved. Never extends its lifetime.
  std::weak_ptr<cucascade::data_batch> batch;
  std::uint64_t batch_id{0};
  /// Process-wide sequence number taken while the batch was exclusively locked
  /// for the conversion, so a later move of the same batch always sorts after
  /// an earlier one.
  std::uint64_t conversion_seq{0};
  /// Memory space the batch was in before the conversion.
  cucascade::memory::memory_space_id from{cucascade::memory::Tier::SIZE, -1};
  /// Memory space the batch is in after the conversion.
  cucascade::memory::memory_space_id to{cucascade::memory::Tier::SIZE, -1};
  /// Size of the batch in @c from.
  std::size_t bytes{0};
  downgrade_source source{downgrade_source::repository};
  /// Tier 1: identity of the repository holding the batch. An opaque token for
  /// matching against known repositories; subscribers never dereference it.
  std::uintptr_t repository{0};
  /// Tier 1: partition of @c repository holding the batch.
  std::size_t partition{0};
  /// Tier 2: priority of the task the batch is an input of.
  std::optional<exec::queue_priority> task_priority;
  /// Tier 2: device the task prefers, if it has a preference.
  std::optional<int> preferred_device;
  /// Tier 2: query the task belongs to, taken from its pipeline.
  std::optional<query_id_t> task_query_id;
};

}  // namespace sirius::event
