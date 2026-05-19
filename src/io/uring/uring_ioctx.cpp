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

uring_ioctx::uring_ioctx(size_t n_reactors,
                         unsigned ring_entries,
                         cucascade::memory::fixed_size_host_memory_resource& mr)
  : templated_ioctx<uring_reactor>(
      n_reactors, [&mr, ring_entries] { return std::make_unique<uring_reactor>(mr, ring_entries); })
{
}

}  // namespace sirius::io
