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

#include <cucascade/memory/common.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace cucascade {
class data_batch;
}  // namespace cucascade

namespace sirius::event {

/// One input of a queued task that is off the GPU: the downgrade executor moved it there, or it
/// was already there when the task was queued.
struct downgraded_input {
  /// The batch. Never extends its lifetime.
  std::weak_ptr<cucascade::data_batch> batch;
  std::uint64_t batch_id{0};
  /// Memory space the batch is in (HOST or DISK).
  cucascade::memory::memory_space_id to{cucascade::memory::Tier::SIZE, -1};
  /// Preload's device-size estimate: the batch's size on the GPU before the move, or, for a task
  /// queued with the batch already off the GPU, its peak_materialization_bytes() there.
  std::size_t gpu_bytes{0};
};

/**
 * @brief Payload of a @c task_downgraded event: a queued task some of whose
 *        inputs are off the GPU.
 *
 * Published by the downgrade executor's tier 2 for the inputs it moved, and by
 * the task scheduler for a task queued with inputs already off the GPU.
 *
 * The batches are held weakly so an event sitting in a subscriber's mailbox
 * never extends their lifetime: a subscriber must expect any of them to have
 * expired.
 */
struct downgraded_task {
  std::uint64_t task_id{0};
  std::size_t pipeline_id{0};
  /// Device the task prefers; empty when any GPU may run it.
  std::optional<int> preferred_device;
  exec::queue_priority priority{};
  /// Only the inputs off the GPU: those a downgrade moved, in the order they
  /// moved, or those a task was queued with, in input order.
  std::vector<downgraded_input> inputs;
};

}  // namespace sirius::event
