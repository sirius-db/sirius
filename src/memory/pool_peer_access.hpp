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

#include <absl/functional/function_ref.h>

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
 * permission changes. Manager initialization, replica transfer, and accumulated-filter admission
 * share this policy.
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
 * @brief Calls `try_grant_pool_access` for each ordered pair of @p spaces on distinct devices, in
 * order, and reports each unavailable pair
 *
 * @param spaces GPU spaces whose pools each other space's device must access
 * @param on_unavailable Receives the allocation owner's and the accessor's device IDs; returning
 * false stops the scan
 */
void for_each_pair_without_pool_access(
  std::span<cucascade::memory::memory_space const* const> spaces,
  absl::FunctionRef<bool(int owner_device, int accessor_device)> on_unavailable);

/**
 * @brief Stops `for_each_pair_without_pool_access` at the first unavailable pair
 *
 * @return First unavailable (allocation owner, accessor) device pair, or no value when every grant
 * is established
 */
[[nodiscard]] std::optional<std::pair<int, int>> find_pair_without_pool_access(
  std::span<cucascade::memory::memory_space const* const> spaces);

}  // namespace sirius::memory
