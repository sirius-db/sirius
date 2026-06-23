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

#include <memory>

namespace sirius::io::uring {

uring_ioctx::uring_ioctx(size_t n_reactors,
                         cucascade::memory::fixed_size_host_memory_resource& mr,
                         bool use_odirect)
  : uring_ioctx(std::make_shared<uring_reactor::reactor_context>(
                  uring_reactor::reactor_config_type{.bounce_size = mr.get_block_size(),
                                                     .use_odirect = use_odirect},
                  &mr),
                n_reactors)
{
}

uring_ioctx::uring_ioctx(const std::shared_ptr<uring_reactor::reactor_context>& ctx,
                         size_t n_reactors)
  : templated_ioctx<uring_reactor>(
      n_reactors, ctx->cfg(), [ctx, i = 0](const uring_reactor::reactor_config_type&) mutable {
        return std::make_unique<uring_reactor>(ctx, fmt::format("reactor-{}", i++));
      })
{
}

}  // namespace sirius::io::uring
