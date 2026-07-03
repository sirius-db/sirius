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

namespace sirius::op::detail {

/// The route selected for a dynamic-filter replica transfer.
enum class replica_transfer_route { none, local, peer_dma, portable_host };

/// Route policy. The forced mode exists for deterministic fallback validation.
enum class replica_transfer_policy { automatic, force_portable_host };

/**
 * Owns completion of one dynamic-filter replica transfer.
 *
 * The transfer may still be queued on the destination stream when this object is returned. Calling
 * wait() makes the destination bytes visible to other streams and releases any portable pinned-host
 * staging storage. Destruction provides a no-throw completion backstop, but publishers should call
 * wait() explicitly so failures prevent publication of that replica.
 */
class replica_transfer final {
 public:
  replica_transfer() noexcept = default;
  ~replica_transfer() noexcept;

  replica_transfer(replica_transfer const&)            = delete;
  replica_transfer& operator=(replica_transfer const&) = delete;
  replica_transfer(replica_transfer&& other) noexcept;
  replica_transfer& operator=(replica_transfer&& other) noexcept;

  /** Wait for completion. Idempotent. Throws if CUDA reports a transfer failure. */
  void wait();

  [[nodiscard]] bool complete() const noexcept { return _complete; }
  [[nodiscard]] replica_transfer_route route() const noexcept { return _route; }

 private:
  friend replica_transfer enqueue_replica_transfer(void*,
                                                   rmm::cuda_device_id,
                                                   void const*,
                                                   rmm::cuda_device_id,
                                                   std::size_t,
                                                   rmm::cuda_stream_view,
                                                   replica_transfer_policy);

  replica_transfer(replica_transfer_route route,
                   rmm::cuda_device_id destination_device,
                   rmm::cuda_stream_view destination_stream,
                   void* portable_staging) noexcept;

  void wait_no_throw() noexcept;

  replica_transfer_route _route{replica_transfer_route::none};
  rmm::cuda_device_id _destination_device{-1};
  rmm::cuda_stream_view _destination_stream{};
  void* _portable_staging{nullptr};
  bool _complete{true};
};

/**
 * Enqueue a byte-for-byte copy of finalized dynamic-filter storage.
 *
 * The caller must make all source writes complete before calling; this function deliberately does
 * not infer or synchronize the source's producer stream. The destination allocation and stream must
 * belong to @p destination_device. Directionally verified peer DMA is preferred. If its probe,
 * source-pool access grant, or individual enqueue fails, the source is copied synchronously into
 * Sirius-owned portable pinned memory and the host-to-device leg is queued on
 * @p destination_stream. In either case the returned token owns the precise completion contract and
 * must outlive the queued copy.
 */
[[nodiscard]] replica_transfer enqueue_replica_transfer(
  void* destination,
  rmm::cuda_device_id destination_device,
  void const* source,
  rmm::cuda_device_id source_device,
  std::size_t bytes,
  rmm::cuda_stream_view destination_stream,
  replica_transfer_policy policy = replica_transfer_policy::automatic);

}  // namespace sirius::op::detail
