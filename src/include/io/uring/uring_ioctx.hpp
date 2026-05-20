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

#include "io/templated_ioctx.hpp"
#include "io/uring/uring_reactor.hpp"

namespace sirius::io {

// ---------------------------------------------------------------------------
// uring_ioctx
// ---------------------------------------------------------------------------

/**
 * @brief io_uring-backed ioctx. Thin specialisation of
 *        @c templated_ioctx<uring_reactor>.
 */
class uring_ioctx : public templated_ioctx<uring_reactor> {
 public:
  /// @param host_ring_depth   Side-channel host ring pool depth.
  /// @param ring_entries      SQE depth per reactor io_uring.
  /// @param n_reactors        Number of reactor workers (each owns its own ring).
  /// @param bounce_slot_size  Size of each pinned bounce slot.
  /// @param numa_node         Target NUMA node for pinned bounce buffers
  ///                          shared across all reactors. -1 disables NUMA
  ///                          binding (legacy single-node behaviour).
  explicit uring_ioctx(unsigned host_ring_depth = 16,
                       unsigned ring_entries    = 64,
                       size_t n_reactors        = 4,
                       size_t bounce_slot_size  = 1UL * 1024 * 1024,
                       int numa_node            = -1);
};

}  // namespace sirius::io
