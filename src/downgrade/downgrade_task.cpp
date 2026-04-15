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

#include "downgrade/downgrade_task.hpp"

#include "data/sirius_converter_registry.hpp"

#include <rmm/cuda_stream_view.hpp>

#include <cucascade/data/cpu_data_representation.hpp>
#include <cucascade/data/disk_data_representation.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_space.hpp>

namespace sirius {
namespace parallel {

namespace {

/// Non-blocking probe: try HOST spaces, then DISK spaces.
/// Returns a reservation + the memory_space it came from, or nullptrs if nothing is available.
std::pair<std::unique_ptr<cucascade::memory::reservation>, cucascade::memory::memory_space*>
try_reserve_host_or_disk(sirius::memory::sirius_memory_reservation_manager& mgr, size_t size)
{
  // Try HOST tier first
  for (auto* space : mgr.get_memory_spaces_for_tier(cucascade::memory::Tier::HOST)) {
    auto* ms  = const_cast<cucascade::memory::memory_space*>(space);
    auto  res = ms->make_reservation_or_null(size);
    if (res) { return {std::move(res), ms}; }
  }
  // Fall back to DISK tier
  for (auto* space : mgr.get_memory_spaces_for_tier(cucascade::memory::Tier::DISK)) {
    auto* ms  = const_cast<cucascade::memory::memory_space*>(space);
    auto  res = ms->make_reservation_or_null(size);
    if (res) { return {std::move(res), ms}; }
  }
  return {nullptr, nullptr};
}

}  // namespace

bool downgrade_task::execute(rmm::cuda_stream_view stream)
{
  // Check if already on host tier - nothing to do
  auto memory_space = batch->get_memory_space();
  if (memory_space == nullptr || memory_space->get_tier() != cucascade::memory::Tier::GPU) {
    return false;
  }

  // Save the batch state so we can restore it after the in-transit conversion.
  // The batch may be in task_created state if a pipeline task is pending for it;
  // blindly resetting to idle would cause the pipeline task to fail with invalid_state.
  auto prev_state = batch->get_state();

  // Try to acquire an in-transit lock - if batch is being processed, we can't downgrade
  if (!batch->try_to_lock_for_in_transit()) {
    // Batch is currently being processed or moving, skip downgrade for now
    // The scheduler can retry later
    return false;
  }

  try {
    auto data_size = batch->get_data()->get_size_in_bytes();

    // Non-blocking probe: try HOST first, then DISK.
    // We must not call request_reservation() here because it blocks forever
    // when no candidate can satisfy the request, which would park the
    // downgrade worker and deadlock GPU memory-pressure handling.
    auto [reservation, mem_space] = try_reserve_host_or_disk(res_mgr, data_size);
    if (!reservation) {
      batch->try_to_release_in_transit(std::optional<cucascade::batch_state>{prev_state});
      return false;
    }

    // Select representation type based on granted tier
    auto& converter_registry = sirius::converter_registry::get();
    if (reservation->tier() == cucascade::memory::Tier::DISK) {
      SIRIUS_LOG_INFO(
        "[downgrade] disk fallback: batch {} ({} B)", batch->get_batch_id(), data_size);
      batch->convert_to<cucascade::disk_data_representation>(converter_registry, mem_space, stream);
    } else {
      batch->convert_to<cucascade::host_data_representation>(converter_registry, mem_space, stream);
    }

    // Release the in-transit lock, restoring the batch to its previous state
    batch->try_to_release_in_transit(std::optional<cucascade::batch_state>{prev_state});
    return true;
  } catch (...) {
    batch->try_to_release_in_transit(std::optional<cucascade::batch_state>{prev_state});
    throw;
  }
}

}  // namespace parallel
}  // namespace sirius
