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

// CUDA translation units include this header, and nvcc rejects Abseil headers under a clang host
// compiler, so keep Abseil out; declarations that take Abseil types belong in
// `memory/pool_peer_access_scan.hpp`.

#include <optional>
#include <span>
#include <utility>

namespace cucascade::memory {
class memory_space;
}

namespace sirius::memory {

/**
 * @brief Grants an accessor persistent read/write access to a GPU space's allocation pool
 *
 * Only the ordinary RMM async allocation domain qualifies. The borrowed space must outlive
 * transfers using its allocations. No handle or capability is cached; callers must coordinate later
 * permission changes. Manager initialization, replica transfer, and the choice of
 * accumulated-filter publication transport (peer copy or HOST staging) share this policy.
 *
 * @throw std::invalid_argument if the space is not GPU-tier, lacks its adaptor, or a device ID is
 * invalid
 * @throw cucascade::cuda_error if CUDA fails, with owner, accessor, and native error details
 * @param owner_space Space owning the allocations
 * @param accessor_device Visible CUDA device that will read the allocations
 * @return True for an established grant; false for unsupported, unverified, or unknown pool
 * capability
 */
[[nodiscard]] bool try_grant_pool_access(cucascade::memory::memory_space const& owner_space,
                                         int accessor_device);

/**
 * @brief Stops `for_each_pair_without_pool_access` at the first unavailable pair
 *
 * @return First unavailable (allocation owner, accessor) device pair, or no value when every grant
 * is established
 */
[[nodiscard]] std::optional<std::pair<int, int>> find_pair_without_pool_access(
  std::span<cucascade::memory::memory_space const* const> spaces);

}  // namespace sirius::memory
