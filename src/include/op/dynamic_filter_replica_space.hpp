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

#include <functional>

namespace cucascade::memory {
class memory_space;
}

namespace sirius::op {

/**
 * @brief Non-owning placement handle for one device-local dynamic-filter replica.
 *
 * The referenced GPU memory space supplies both the replica allocator and its pooled CUDA stream.
 * The Sirius memory manager that owns the space (and therefore its allocator and stream pool) must
 * outlive every publication plan containing this handle and every filter replica materialized from
 * it. All GPU uses of a filter must finish before that filter is destroyed, and all such filters
 * must be destroyed before the owning Sirius context tears down its memory manager.
 *
 * This lifetime dependency is also required by the device allocations themselves; this handle
 * makes it explicit and non-null without transferring ownership.
 */
class dynamic_filter_replica_space final {
 public:
  explicit dynamic_filter_replica_space(cucascade::memory::memory_space const& space) noexcept
    : _space{space}
  {
  }

  [[nodiscard]] cucascade::memory::memory_space const& get() const noexcept { return _space.get(); }

 private:
  std::reference_wrapper<cucascade::memory::memory_space const> _space;
};

}  // namespace sirius::op
