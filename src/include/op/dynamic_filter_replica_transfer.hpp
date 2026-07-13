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

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <cstddef>

namespace cucascade::memory {
class memory_space;
}

namespace sirius::op::detail {

/// @brief The route selected for a dynamic-filter replica transfer.
/// @note This is currently only used for testing and logging.
enum class replica_transfer_route {
  none,
  local,        ///< Same-device copy
  peer_dma,     ///< GPU-to-GPU copy via peer DMA
  host_staging  ///< Copy through the target's Sirius HOST memory space
};

/// @brief The transfer policy.
/// @note This is currently only used for testing.
enum class replica_transfer_policy {
  automatic,
  force_host_staging  ///< For deterministic host-fallback testing
};

/// @brief Submit a copy of finalized dynamic-filter storage and return the selected route.
///
/// The caller must make all source writes complete before calling; this function deliberately does
/// not infer or synchronize the source's producer stream. @p source_space must be the Sirius GPU
/// memory space that owns @p source; the staged fallback acquires a source-bound pooled stream from
/// it. The destination allocation and stream must belong to @p destination_device. Following
/// CuCascade's GPU-to-GPU converter, directionally verified peer DMA is preferred. If the probe
/// rejects peer DMA, the source is copied through fixed pinned blocks borrowed from
/// @p host_staging_space.
///
/// Local and peer-DMA copies are only enqueued on @p destination_stream; the caller must keep both
/// allocations alive and synchronize that stream before publishing the replica. HOST staging
/// completes both dependent legs before returning so its borrowed blocks can immediately return to
/// the pool. An enqueue or synchronization failure propagates.
///
/// @return The selected transfer route (currently only used for testing and logging).
replica_transfer_route enqueue_replica_copy(
  void* destination,
  rmm::cuda_device_id destination_device,
  void const* source,
  cucascade::memory::memory_space const& source_space,
  std::size_t bytes,
  rmm::cuda_stream_view destination_stream,
  cucascade::memory::memory_space const& host_staging_space,
  replica_transfer_policy policy = replica_transfer_policy::automatic);

}  // namespace sirius::op::detail
