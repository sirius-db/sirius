/*
 * Copyright 2025, Sirius Contributors.
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

#include "catch.hpp"

#include <liburing.h>

#include <cerrno>
#include <cstring>

// Smoke test for PR0: ensure liburing symbols (io_uring_queue_init /
// io_uring_queue_exit) are linked into sirius_unittest. A successful
// io_uring_queue_init also confirms the runtime kernel supports io_uring;
// on kernels without io_uring support the call fails with -ENOSYS/-EPERM,
// which still proves the symbols resolved at link time.
TEST_CASE("liburing symbols are linked and callable", "[io_link]")
{
  struct io_uring ring{};
  int ret = io_uring_queue_init(8, &ring, 0);

  if (ret < 0) {
    // Kernel does not support io_uring or the process lacks the capability.
    // The link-time goal of this PR is still met: symbols were found.
    WARN("io_uring_queue_init returned " << ret << " (" << std::strerror(-ret)
                                         << "); skipping queue_exit");
    SUCCEED("liburing symbols linked; runtime init unavailable on this host");
    return;
  }

  io_uring_queue_exit(&ring);
  SUCCEED("liburing init + exit round-trip succeeded");
}
