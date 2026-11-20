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

#include "memory/device_transfer.hpp"

#include <rmm/cuda_device.hpp>

#include <cuda/stream>

#include <cstddef>
#include <exception>
#include <string_view>

namespace cucascade::memory {
class memory_space;
}

namespace sirius::op::detail {

/**
 * @brief Selected copy API; peer_copy means cudaMemcpyPeerAsync, not a measured physical route
 */
using replica_transfer_route = sirius::memory::device_transfer_route;

using replica_transfer_policy = sirius::memory::device_transfer_policy;

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
  ::cuda::stream_ref destination_stream,
  cucascade::memory::memory_space const& host_staging_space,
  replica_transfer_policy policy = replica_transfer_policy::automatic);

/**
 * @brief Keeps the first fatal replica failure while remaining submitted work drains
 *
 * Submission may skip recoverable allocation or availability failures. CUDA, programming, and
 * non-standard exceptions are fatal. Every completion failure is fatal regardless of type. Callers
 * drain all pending copies, rethrow the first fatal failure, and only then install completed
 * replicas.
 */
class replica_fanout_failures final {
 public:
  /**
   * @brief Classifies the exception currently being handled; call only from a catch handler
   *
   * @param filter_name Log prefix naming the filter type
   * @param source_device GPU holding the source replica
   * @param destination_device GPU whose copy failed
   * @return True when the failure was recoverable and has been logged, so the caller continues with
   * the next GPU; false when it is fatal and the caller must stop queueing
   */
  [[nodiscard]] bool skip_if_recoverable(std::string_view filter_name,
                                         int source_device,
                                         int destination_device) noexcept;

  /**
   * @brief Records the current exception as fatal; call only from a catch handler
   *
   * Later failures are logged without replacing the first or interrupting cleanup.
   */
  void record_fatal(std::string_view filter_name,
                    int source_device,
                    int destination_device) noexcept;

  /**
   * @brief Rethrows the fatal failure, if any
   */
  void rethrow_if_fatal() const;

 private:
  std::exception_ptr _fatal;
};

}  // namespace sirius::op::detail
