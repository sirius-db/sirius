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

#include "io/uring/uring_ioctx.hpp"

#include <memory>

namespace sirius::io {

uring_ioctx::uring_ioctx(unsigned host_ring_depth,
                         unsigned ring_entries,
                         size_t n_reactors,
                         size_t bounce_slot_size,
                         int numa_node)
  : templated_ioctx<uring_reactor>(n_reactors, [ring_entries, bounce_slot_size, numa_node] {
      return std::make_unique<uring_reactor>(ring_entries, bounce_slot_size, numa_node);
    })
{
}

}  // namespace sirius::io
