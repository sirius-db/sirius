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
  /// Select the file handle used by local read requests.  When true, aligned
  /// requests use the O_DIRECT handle and incompatible requests fall back to
  /// the buffered (page-cache) handle; false forces every request through the
  /// buffered handle.  This is a request-mode switch, not a deployment-mode
  /// switch: create_io_object still opens both handles, so the filesystem must
  /// support O_DIRECT in either mode.  Defaults to true.
  bool use_odirect{true};

  // max number of contiguous segments to fuse into one readv SQE.  The
  // prep_host_rxv_request and prep_host_to_device_rx_request paths fuse
  // contiguous segments into one readv SQE, capped at this value.  The
  std::size_t max_n_chunks{1};
};

}  // namespace sirius::io::uring
