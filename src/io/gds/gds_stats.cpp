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

#include "io/gds/gds_stats.hpp"

#include <dlfcn.h>

#include <algorithm>
#include <fstream>
#include <iterator>
#include <string>

#if __has_include(<cufile.h>)
#include <cufile.h>
#endif

namespace sirius::io::gds {

namespace {

// kvikIO loads libcufile with dlopen, so Sirius has no link dependency on it;
// the statistics entry points are looked up in that same library.
void* cufile_symbol(char const* name) noexcept
{
  static void* const library = [] {
    void* handle = dlopen("libcufile.so.0", RTLD_LAZY | RTLD_NOLOAD);
    return handle != nullptr ? handle : dlopen("libcufile.so.0", RTLD_LAZY);
  }();
  return library != nullptr ? dlsym(library, name) : nullptr;
}

}  // namespace

// CUFILE_GPU_UUID_LEN is defined next to the statistics structs, which older
// cufile.h headers lack.
#ifdef CUFILE_GPU_UUID_LEN

bool enable_cufile_counters() noexcept
{
  auto* set_level = reinterpret_cast<CUfileError_t (*)(int)>(cufile_symbol("cuFileSetStatsLevel"));
  auto* start     = reinterpret_cast<CUfileError_t (*)()>(cufile_symbol("cuFileStatsStart"));
  if (set_level == nullptr || start == nullptr) { return false; }
  if (set_level(3).err != CU_FILE_SUCCESS) { return false; }
  return start().err == CU_FILE_SUCCESS;
}

std::optional<cufile_counters> read_cufile_counters() noexcept
{
  auto* get =
    reinterpret_cast<CUfileError_t (*)(CUfileStatsLevel3_t*)>(cufile_symbol("cuFileGetStatsL3"));
  if (get == nullptr) { return std::nullopt; }
  CUfileStatsLevel3_t stats{};
  if (get(&stats).err != CU_FILE_SUCCESS) { return std::nullopt; }
  cufile_counters out;
  auto const gpus = std::min<std::uint32_t>(stats.num_gpus, std::size(stats.per_gpu_stats));
  for (std::uint32_t i = 0; i < gpus; ++i) {
    auto const& gpu = stats.per_gpu_stats[i];
    out.read_bytes += gpu.read_bytes;
    out.total_reads += gpu.n_total_reads;
    out.nvfs_reads += gpu.n_nvfs_reads;
    out.posix_reads += gpu.n_posix_reads;
    out.mmaps += gpu.n_mmap;
  }
  return out;
}

#else

bool enable_cufile_counters() noexcept { return false; }
std::optional<cufile_counters> read_cufile_counters() noexcept { return std::nullopt; }

#endif

read_path classify(cufile_counters const& delta) noexcept
{
  if (delta.total_reads == 0) { return read_path::none; }
  if (delta.posix_reads >= delta.total_reads) { return read_path::posix; }
  // cuFile serves reads of a few KiB through POSIX even under GDS.
  return delta.posix_reads * 2 > delta.total_reads ? read_path::mixed : read_path::gds;
}

std::string_view to_string(read_path path) noexcept
{
  switch (path) {
    case read_path::none: return "none";
    case read_path::gds: return "gds";
    case read_path::posix: return "posix";
    case read_path::mixed: return "mixed";
  }
  return "unknown";
}

std::optional<std::uint64_t> parse_nvidia_fs_mmap_count(std::istream& stats)
{
  std::string line;
  while (std::getline(stats, line)) {
    if (!line.starts_with("Mmap")) { continue; }
    auto const pos = line.find("n=");
    if (pos == std::string::npos) { return std::nullopt; }
    try {
      return std::stoull(line.substr(pos + 2));
    } catch (...) {
      return std::nullopt;
    }
  }
  return std::nullopt;
}

std::optional<std::uint64_t> read_nvidia_fs_mmap_count()
{
  std::ifstream stats("/proc/driver/nvidia-fs/stats");
  if (!stats) { return std::nullopt; }
  return parse_nvidia_fs_mmap_count(stats);
}

}  // namespace sirius::io::gds
