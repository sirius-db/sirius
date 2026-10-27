/*
 * Copyright 2026, Sirius Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <sys/stat.h>

#include <atomic>
#include <cstdint>
#include <string>
#include <utility>

namespace sirius::io {

/// The backend chooses its evidence. Open-local identity deliberately cannot
/// authorize reuse through a separately opened, size-only handle.
enum class identity_kind : uint8_t { opened_handle, local_stat, object_tag };

struct object_identity {
  identity_kind kind = identity_kind::opened_handle;
  uint64_t size      = 0;
  std::string version;
  uint64_t device   = 0;
  uint64_t inode    = 0;
  uint64_t ctime_ns = 0;

  bool operator==(object_identity const& other) const noexcept
  {
    return kind == other.kind && size == other.size && version == other.version &&
           device == other.device && inode == other.inode && ctime_ns == other.ctime_ns;
  }
  bool operator!=(object_identity const& other) const noexcept { return !(*this == other); }

  [[nodiscard]] std::string cache_key(std::string const& locator) const
  {
    // Length prefix makes arbitrary paths/tags unambiguous; never parse an ETag.
    return std::to_string(locator.size()) + ":" + locator + ":" +
           std::to_string(static_cast<unsigned>(kind)) + ":" + std::to_string(size) + ":" +
           std::to_string(device) + ":" + std::to_string(inode) + ":" + std::to_string(ctime_ns) +
           ":" + version;
  }
};

inline uint64_t next_open_generation() noexcept
{
  static std::atomic<uint64_t> generation{0};
  return generation.fetch_add(1, std::memory_order_relaxed) + 1;
}

/// Capture from the actual open fd, not a second path lookup which could stat
/// a replacement. Failure is explicit and leaves callers with open-local reuse.
inline bool capture_local_identity(int fd, object_identity& result)
{
  struct stat info{};
  if (::fstat(fd, &info) != 0 || info.st_size < 0) return false;
  result.kind   = identity_kind::local_stat;
  result.size   = static_cast<uint64_t>(info.st_size);
  result.device = static_cast<uint64_t>(info.st_dev);
  result.inode  = static_cast<uint64_t>(info.st_ino);
#if defined(__APPLE__)
  result.ctime_ns = static_cast<uint64_t>(info.st_ctimespec.tv_sec) * 1'000'000'000ULL +
                    static_cast<uint64_t>(info.st_ctimespec.tv_nsec);
#else
  result.ctime_ns = static_cast<uint64_t>(info.st_ctim.tv_sec) * 1'000'000'000ULL +
                    static_cast<uint64_t>(info.st_ctim.tv_nsec);
#endif
  // ctime_ns is already part of the structured identity; do not duplicate it
  // in the free-form version field used by object-store validators.
  result.version.clear();
  return true;
}
}  // namespace sirius::io
