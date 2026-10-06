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

#include "io/types.hpp"

#include <cstddef>

namespace sirius::io::uring {

struct config {
  /// How many scan tasks the readahead manager may keep in flight against this
  /// backend at once.  Zero disables readahead for it entirely.
  ///
  /// Local NVMe has no round trip to hide, so a readahead competes with the
  /// executor's own reads for the same device and just reorders the queue
  /// rather than adding throughput.  Measured on SF1000 local-parquet, turning
  /// it off is a large net win, so the local backend defaults to 0 (off).  Set
  /// a positive value (or `max_readahead_scans`) to opt the local path back in.
  std::size_t n_max_concurrent_scans{0};

  /// Whether the config named @c n_max_concurrent_scans explicitly. Preserved
  /// for parity with the REST backend (whose budget is still derived from the
  /// pipeline width) so an explicit value is always distinguishable from the
  /// struct default -- including an explicit 0, which opts the local path out
  /// as deliberately as the default does.
  bool n_max_concurrent_scans_explicit{false};

  /// When false, worker-planned operations use the buffered page-cache handle.
  /// Defaults to O_DIRECT when a physical operation satisfies its constraints.
  bool use_odirect{true};

  /// O_DIRECT transfers whole pages, so a read is widened to a page boundary
  /// either way -- naming it lets the caller align once, up front, instead of
  /// every layer rediscovering it.  Reported even when @ref use_odirect is
  /// false: a buffered read of a page-aligned span costs no more than an
  /// unaligned one, and keeping the value constant keeps the two modes
  /// comparable.
  [[nodiscard]] std::size_t min_alignment_requirement() const noexcept { return io::IO_BLOCK_SIZE; }

  /// A local read is a syscall against NVMe, so bridging is only worth it when
  /// the bridged bytes are cheaper than the extra request -- one page.
  [[nodiscard]] std::size_t merge_gap_size() const noexcept { return io::IO_BLOCK_SIZE; }
};

}  // namespace sirius::io::uring
