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
  /// Each @c uring_reactor in the pool allocates its bounce slots from
  /// @p mr; @p mr must outlive this ioctx.  The bounce-slot size is taken
  /// from @c mr.get_block_size().
  uring_ioctx(size_t n_reactors,
              unsigned ring_entries,
              cucascade::memory::fixed_size_host_memory_resource& mr);
};

}  // namespace sirius::io
