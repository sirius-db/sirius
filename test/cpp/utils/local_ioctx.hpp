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

#include <cucascade/io/io_context.hpp>
#include <cucascade/io/uring/uring_ioctx.hpp>
#include <cucascade/io/uring/uring_reactor.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/numa_region_pinned_host_allocator.hpp>

#include <cstddef>
#include <memory>

namespace sirius::test {

namespace detail {

/// Pinned bounce staging private to one test ioctx.
struct local_ioctx_staging {
  static constexpr std::size_t block_size = std::size_t{1} << 20;
  static constexpr std::size_t pool_size  = 16;
  static constexpr std::size_t capacity   = block_size * 256;  // grown on demand

  local_ioctx_staging() : upstream(0), mr(0, upstream, capacity, capacity, block_size, pool_size, 1)
  {
  }

  cucascade::memory::numa_region_pinned_host_memory_resource upstream;
  cucascade::memory::fixed_size_host_memory_resource mr;
};

}  // namespace detail

/// A started local-file ioctx for tests that read their fixture files directly.
///
/// One uring reactor with its own private pinned staging, so no reservation
/// manager is needed.  The staging is owned by the ioctx's deleter, so it is
/// released only after the ioctx (and its reactor thread) is gone, however the
/// pointer is shared -- including through @c ioctx::shared_from_this.  Only
/// existing regular files are supported (see @c uring_ioctx::supports).
inline std::shared_ptr<cucascade::io::ioctx> make_test_local_ioctx()
{
  namespace uring = cucascade::io::uring;
  auto staging    = std::make_shared<detail::local_ioctx_staging>();
  auto ctx        = std::make_shared<uring::uring_reactor::reactor_context>(
    uring::uring_reactor::reactor_config_type{}, &staging->mr);
  std::shared_ptr<uring::uring_ioctx> io(
    new uring::uring_ioctx(1, std::move(ctx)),
    [staging = std::move(staging)](uring::uring_ioctx* p) { delete p; });
  io->start();
  return io;
}

}  // namespace sirius::test
