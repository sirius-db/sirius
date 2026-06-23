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

#include <cudf/io/datasource.hpp>
#include <cudf/io/text/byte_range_info.hpp>

#include <cuda_runtime.h>

#include <sys/uio.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace sirius::io {

static constexpr size_t IO_BLOCK_SIZE = 4096;  // O_DIRECT page size

/**
 * @brief RAII wrapper for a POSIX file descriptor.
 *
 * Non-copyable, movable. Closes the underlying fd on destruction.
 */
struct file_descriptor {
  int fd{-1};
  file_descriptor() = default;
  explicit file_descriptor(int f) noexcept : fd(f) {}
  ~file_descriptor() noexcept
  {
    if (fd >= 0) ::close(fd);
  }
  file_descriptor(file_descriptor const&)            = delete;
  file_descriptor& operator=(file_descriptor const&) = delete;
  file_descriptor(file_descriptor&& o) noexcept : fd(std::exchange(o.fd, -1)) {}
  file_descriptor& operator=(file_descriptor&& o) noexcept
  {
    if (this != &o) {
      if (fd >= 0) ::close(fd);
      fd = std::exchange(o.fd, -1);
    }
    return *this;
  }
  [[nodiscard]] int get() const noexcept { return fd; }
  [[nodiscard]] int native_handle() const noexcept { return fd; }
  explicit operator bool() const noexcept { return fd >= 0; }
};

// ---------------------------------------------------------------------------
// sirius_io_object
// ---------------------------------------------------------------------------

/**
 * @brief Abstract per-file handle.  A passive bag of native handles
 * produced by a backend reactor (e.g. file descriptors, CURL easy
 * handles, S3 client state).  Performs no I/O of its own.
 *
 * Inherits from @c std::enable_shared_from_this so the prefetching cache can
 * take a reference to an io_object and safely extend its lifetime via
 * @c shared_from_this() — this enforces at call sites that every io_object
 * passed in is already owned by a @c std::shared_ptr.
 */
class sirius_io_object : public std::enable_shared_from_this<sirius_io_object> {
 public:
  virtual ~sirius_io_object() = default;

  /// Stable identifier used as the prefetching-cache key.  Often equal to
  /// @c object_path() but may differ for backends that need to distinguish
  /// otherwise-equal paths (versioned S3 keys, normalized URLs, …).
  [[nodiscard]] virtual const std::string& raw_file_cache_id() const noexcept = 0;

  /// The path / URL / key the caller used to construct this object.
  [[nodiscard]] virtual const std::string& object_path() const noexcept = 0;

  /// Total size of the underlying object, populated by the reactor at
  /// construction time and stored on the io_object thereafter.
  [[nodiscard]] virtual size_t size() const noexcept = 0;
};

class sirius_io_object_metadata {
 public:
  virtual ~sirius_io_object_metadata() = default;
};

/// A read of @c size bytes starting at file @c offset, scattered into one or
/// more destination buffers (@c buffers, in file order).  A single-buffer
/// segment is a plain read; a multi-buffer segment is a vectored (readv) read
/// whose iovecs cover @c [offset, offset + size) contiguously in the file but
/// may land in discontiguous host allocations.
///
/// Invariant: @c size == Σ buffers[i].iov_len.  The merge step that fuses
/// neighboring segments during request preparation maintains this by routing
/// every growth through @c append.
class io_object_segment {
 public:
  io_object_segment() = default;

  io_object_segment(size_t offset, size_t size)
    : offset(offset), size(size), buffers{iovec{nullptr, size}}
  {
  }

  io_object_segment(size_t offset, size_t size, uint8_t* buffer)
    : offset(offset), size(size), buffers{iovec{static_cast<void*>(buffer), size}}
  {
  }

  /// Set the destination of a single-buffer segment (the bounce-slot path
  /// assigns the reactor's internal buffer late, once a slot is acquired).
  void set_data(uint8_t* buffer)
  {
    assert(buffers.size() == 1 && "set_data is only valid for a single-buffer segment");
    buffers.front().iov_base = static_cast<void*>(buffer);
  }

  [[nodiscard]] uint8_t* data() const noexcept
  {
    return buffers.empty() ? nullptr : static_cast<uint8_t*>(buffers.front().iov_base);
  }

  [[nodiscard]] bool is_buffer_allocated() const noexcept { return data() != nullptr; }

  /// Number of destination buffers (== number of iovecs in a readv).
  [[nodiscard]] size_t n_chunks() const noexcept { return buffers.size(); }

  /// True iff this segment must be submitted via io_uring_prep_readv (rather
  /// than a single io_uring_prep_read).
  [[nodiscard]] bool is_vectored() const noexcept { return buffers.size() > 1; }

  /// O_DIRECT requires the file offset, the total length, and every iovec base
  /// and length to be block-aligned.
  [[nodiscard]] bool is_odirect_compatible() const noexcept
  {
    if (offset % IO_BLOCK_SIZE != 0 || size % IO_BLOCK_SIZE != 0) { return false; }
    for (auto const& b : buffers) {
      if (b.iov_len % IO_BLOCK_SIZE != 0) { return false; }
      if (b.iov_base != nullptr && reinterpret_cast<uintptr_t>(b.iov_base) % IO_BLOCK_SIZE != 0) {
        return false;
      }
    }
    return true;
  }

  /// Append a destination buffer, fusing a contiguous neighbor into this
  /// segment.  Grows @c size by the buffer length to preserve the invariant.
  void append(iovec iov) noexcept
  {
    buffers.push_back(iov);
    size += iov.iov_len;
  }

  /// Rebuild the iovec list for resuming a short read after @p skip bytes were
  /// already read: drops fully-consumed buffers and advances into the
  /// straddling one.  Fills @p out in place (reusing its capacity across
  /// resubmissions); @c buffers is untouched.
  void fill_remaining_buffers(size_t skip, std::vector<iovec>& out) const
  {
    out.clear();
    out.reserve(buffers.size());
    for (auto const& iov : buffers) {
      if (skip >= iov.iov_len) {
        skip -= iov.iov_len;
        continue;
      }
      out.push_back(iovec{static_cast<uint8_t*>(iov.iov_base) + skip, iov.iov_len - skip});
      skip = 0;
    }
  }

  size_t offset{0};
  size_t size{0};
  // Destination buffers in file order.  Owned here so the iovec array stays
  // alive until the SQE referencing it is reaped.
  std::vector<iovec> buffers;
};

/// True iff @p a immediately precedes @p b in the file (no gap, no overlap):
/// a.offset + a.size == b.offset.  Used to decide whether two segments can be
/// fused into a single vectored (readv) submission over one contiguous range.
[[nodiscard]] inline bool contiguous(const io_object_segment& a,
                                     const io_object_segment& b) noexcept
{
  return a.offset + a.size == b.offset;
}

}  // namespace sirius::io
