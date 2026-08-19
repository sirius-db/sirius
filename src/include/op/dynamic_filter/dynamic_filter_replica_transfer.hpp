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

enum class replica_transfer_route { none, local, peer_dma, host_staging };

enum class replica_transfer_policy { automatic, force_host_staging };

/**
 * @brief Enqueues a finalized replica copy and returns its route
 *
 * Source writes must already be complete. Local and peer copies are asynchronous on
 * @p destination_stream; keep both allocations alive and synchronize before publication. Host
 * staging completes before return. @p source_space must own @p source; @p destination and
 * @p destination_stream must belong to @p destination_device.
 *
 * @throw std::invalid_argument if a pointer is null for a non-empty copy, the source is not
 * GPU-tier, or required host staging is not HOST-tier
 * @throw std::runtime_error if required host staging is unavailable
 */
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
