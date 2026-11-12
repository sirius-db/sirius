/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */

#pragma once

#include <chrono>
#include <cstddef>

namespace sirius::io::gds {

/// Most cuFile reads one process may have in flight (`gds_config::num_threads`).
inline constexpr std::size_t k_max_num_threads = 64;

/// Settings of the GPUDirect Storage read backend (`scan_manager.backend: gds`,
/// YAML block `scan_manager.gds`).
struct gds_config {
  /// Threads that issue cuFile reads; the bound on reads in flight per process.
  std::size_t num_threads{32};
  /// Bytes per cuFile read. A larger device read is cut into pieces of this size.
  std::size_t task_size{16ULL << 20};
  /// Refuse to start unless cuFile reports GDS available with its POSIX fallback
  /// disabled, and fail reads once cuFile serves most reads through POSIX.
  bool require{true};
  /// A cuFile read in flight for longer than this ends the process (exit 70).
  std::chrono::seconds read_timeout{10};
  /// Reads fail once they stay slower than this many milliseconds per MiB for
  /// two seconds; zero turns the check off. Measured on GPFS: 0.8-1.2 ms per
  /// MiB when healthy, 10-62 ms per MiB on a path that was going bad.
  double slow_read_ms_per_mib{5.0};
};

}  // namespace sirius::io::gds
