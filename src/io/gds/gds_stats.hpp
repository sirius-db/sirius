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

#include <cstdint>
#include <istream>
#include <optional>
#include <string_view>

namespace sirius::io::gds {

/// cuFile's own read counters for this process, summed over its GPUs.
struct cufile_counters {
  std::uint64_t read_bytes{0};
  std::uint64_t total_reads{0};
  /// Reads cuFile served through the nvidia-fs driver.
  std::uint64_t nvfs_reads{0};
  /// Reads cuFile served with POSIX calls: its silent fallback.
  std::uint64_t posix_reads{0};
  /// GPU buffer registrations.
  std::uint64_t mmaps{0};

  [[nodiscard]] cufile_counters operator-(cufile_counters const& earlier) const noexcept
  {
    return {read_bytes - earlier.read_bytes,
            total_reads - earlier.total_reads,
            nvfs_reads - earlier.nvfs_reads,
            posix_reads - earlier.posix_reads,
            mmaps - earlier.mmaps};
  }
};

/// Turn on cuFile's per-GPU statistics (level 3). Returns false when the loaded
/// libcufile has no statistics API or refuses.
bool enable_cufile_counters() noexcept;

/// Current cuFile counters, or nullopt when the loaded libcufile cannot report
/// them (no statistics API, level below 3, driver not open).
[[nodiscard]] std::optional<cufile_counters> read_cufile_counters() noexcept;

/// What a span of cuFile activity shows about the read path.
enum class read_path { none, gds, posix, mixed };

[[nodiscard]] read_path classify(cufile_counters const& delta) noexcept;
[[nodiscard]] std::string_view to_string(read_path path) noexcept;

/// The `n=` value of the `Mmap` line of `/proc/driver/nvidia-fs/stats`: how
/// often the driver mapped GPU memory, node-wide. It rises only under GDS.
[[nodiscard]] std::optional<std::uint64_t> parse_nvidia_fs_mmap_count(std::istream& stats);
[[nodiscard]] std::optional<std::uint64_t> read_nvidia_fs_mmap_count();

}  // namespace sirius::io::gds
