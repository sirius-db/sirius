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

  bool operator==(object_identity const&) const = default;

  [[nodiscard]] std::string cache_key(std::string const& locator) const
  {
    // Length prefix makes arbitrary paths/tags unambiguous; never parse an ETag.
    return std::to_string(locator.size()) + ":" + locator + ":" +
           std::to_string(static_cast<unsigned>(kind)) + ":" + std::to_string(size) + ":" + version;
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
  result.kind    = identity_kind::local_stat;
  result.size    = static_cast<uint64_t>(info.st_size);
  result.version = std::to_string(info.st_mtim.tv_sec) + ":" + std::to_string(info.st_mtim.tv_nsec);
  return true;
}
}  // namespace sirius::io
