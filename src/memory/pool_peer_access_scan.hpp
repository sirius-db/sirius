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

#include <span>

namespace cucascade::memory {
class memory_space;
}

namespace sirius::memory {

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

}  // namespace sirius::memory
