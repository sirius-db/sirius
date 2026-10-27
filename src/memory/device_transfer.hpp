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

#include <cuda/stream>

#include <cucascade/cuda/event.hpp>

#include <cstddef>
#include <span>
#include <variant>

namespace cucascade::memory {
class memory_space;
}

namespace sirius::memory {

struct source_already_complete {};
using transfer_readiness = std::variant<source_already_complete, cucascade::cuda::cuda_event_view>;

enum class device_transfer_route { none, local, peer_copy, host_staging };
enum class device_transfer_policy { automatic, force_host_staging };

/**
 * @brief Copies device bytes using a granted source pool or managed pinned HOST staging
 *
 * @p source_space, when supplied, must back @p source; a matching device ID alone does not
 * establish allocation provenance. @p ready must cover every source writer and its event must
 * already be recorded. Destination allocation must precede the copy on @p destination_stream. Local
 * and peer copies are asynchronous; callers retain both GPU allocations until destination work
 * retires or is ordered before their frees. HOST staging allocates blocks for the entire transfer
 * from @p host_staging_space, submits each leg as a batch where supported, and completes both legs
 * before return. An unknown cross-device source allocation domain is staged. The returned route
 * identifies the API selected, not the physical peer-copy route.
 *
 * @param destination The destination buffer
 * @param destination_device The device of the destination buffer
 * @param destination_stream The stream on which to perform the copy
 * @param source The source buffer
 * @param source_device The device of the source buffer
 * @param source_space The memory space of the source buffer
 * @param ready The readiness of the source data
 * @param host_staging_space The memory space for HOST staging
 * @param policy The transfer policy to use
 *
 * @throw std::invalid_argument if byte extents, memory spaces, or destination stream do not match
 * @throw std::runtime_error if required HOST staging is unavailable
 */
[[nodiscard]] device_transfer_route enqueue_device_copy(
  std::span<std::byte> destination,
  rmm::cuda_device_id destination_device,
  ::cuda::stream_ref destination_stream,
  std::span<std::byte const> source,
  rmm::cuda_device_id source_device,
  cucascade::memory::memory_space const* source_space,
  transfer_readiness ready,
  cucascade::memory::memory_space const* host_staging_space,
  device_transfer_policy policy = device_transfer_policy::automatic);

}  // namespace sirius::memory
