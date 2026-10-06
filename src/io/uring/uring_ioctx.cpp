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

#include "io/uring/uring_reactor.hpp"

#include <format>
#include <memory>

namespace sirius::io::uring {

uring_ioctx::uring_ioctx(size_t n_reactors, std::shared_ptr<uring_reactor::reactor_context> ctx)
  : templated_ioctx<uring_reactor>(n_reactors, [ctx = std::move(ctx), i = 0]() mutable {
      return std::make_unique<uring_reactor>(ctx, std::format("reactor-{}", i++));
    })
{
}

}  // namespace sirius::io::uring
