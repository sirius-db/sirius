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

#include <cstddef>

namespace sirius::io::uring {

struct config {
  std::size_t bounce_size{1UL << 20};
  /// When false, every prep path except the BYO-device-buffer read
  /// (prep_device_rx_request) reads through the buffered (page-cache) file
  /// handle instead of the O_DIRECT one.  Defaults to O_DIRECT.
  bool use_odirect{true};

  std::size_t max_n_chunks{16};

  /// io_uring submission-queue depth (entries).  0 means use the reactor's
  /// compiled-in default.  Increasing this allows more I/O requests to be
  /// in flight simultaneously before the kernel must be notified.
  unsigned ring_entries{0};
};

}  // namespace sirius::io::uring