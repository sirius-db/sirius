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

#include "exec/config.hpp"
#include "io/types.hpp"

#include <cstddef>

namespace sirius::io::uring {

struct config {
  /// How many scan tasks the readahead manager may keep in flight against this
  /// backend at once.  Zero disables readahead for it entirely.
  ///
  /// Local NVMe saturates at modest queue depth and every in-flight scan pins
  /// staging buffers, so one scan per pipeline executor thread is enough to
  /// keep the decoders fed without over-committing the pinned pool.
  std::size_t n_max_concurrent_scans{
    static_cast<std::size_t>(exec::default_gpu_pipeline_num_threads)};

  std::size_t bounce_size{1UL << 20};
  /// When false, every prep path except the BYO-device-buffer read
  /// (prep_device_rx_request) reads through the buffered (page-cache) file
  /// handle instead of the O_DIRECT one.  Defaults to O_DIRECT.
  bool use_odirect{true};

  // max number of contiguous segments to fuse into one readv SQE.  The
  // prep_host_rxv_request and prep_host_to_device_rx_request paths fuse
  // contiguous segments into one readv SQE, capped at this value.  The
  std::size_t max_n_chunks{1};

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
