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

#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime.h>

#include <io/cache/types.hpp>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace sirius::io {

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
// io_object
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
class io_object : public std::enable_shared_from_this<io_object> {
 public:
  virtual ~io_object() = default;

  /// Stable identifier used as the prefetching-cache key.  Often equal to
  /// @c object_path() but may differ for backends that need to distinguish
  /// otherwise-equal paths (versioned S3 keys, normalized URLs, …).
  [[nodiscard]] virtual const std::string& raw_file_cache_id() const noexcept = 0;

  /// The path / URL / key the caller used to construct this object.
  [[nodiscard]] virtual const std::string& object_path() const noexcept = 0;

  /// Total size of the underlying object, populated by the reactor at
  /// construction time and stored on the io_object thereafter.
  [[nodiscard]] virtual size_t size() const noexcept = 0;

  /// Opaque validation tag associated with this open (the HTTP ETag for
  /// object-store backends, quotes preserved); empty when unavailable.
  /// Consumers compare it only for equality against a tag they captured
  /// earlier; an empty tag disables validation-based caching above —
  /// degraded performance, never wrong bytes.
  [[nodiscard]] virtual std::string_view validation_tag() const noexcept { return {}; }
};

class io_object_metadata {
 public:
  virtual ~io_object_metadata() = default;
};

struct range {
  std::size_t offset{0};
  std::size_t size{0};
};

struct slice {
  slice() noexcept = default;

  explicit slice(std::size_t offset, std::size_t size, std::uint8_t* dst) noexcept
    : rng{offset, size}, dst{dst}
  {
    assert(dst != nullptr);
    assert(size > 0);
  }

  [[nodiscard]] size_t size() const noexcept { return rng.size; }

  [[nodiscard]] std::size_t offset() const noexcept { return rng.offset; }

  range rng;
  std::uint8_t* dst{nullptr};
};

struct host_buffer {
  host_buffer() noexcept = default;

  explicit host_buffer(std::uint8_t* dst) noexcept : buffer(dst) { assert(dst != nullptr); }

  explicit host_buffer(std::span<cache::cached_chunk*> cached_chunks) noexcept
    : buffer(std::move(cached_chunks))
  {
    assert(!cached_chunks.empty());
  }

  [[nodiscard]] bool needs_staging() const noexcept
  {
    return std::holds_alternative<std::monostate>(buffer);
  }

  [[nodiscard]] bool is_fragmented() const noexcept
  {
    return std::holds_alternative<std::span<cache::cached_chunk*>>(buffer);
  }

  [[nodiscard]] bool is_contiguous() const noexcept
  {
    return std::holds_alternative<std::uint8_t*>(buffer);
  }

  [[nodiscard]] bool is_staged() const noexcept
  {
    return std::holds_alternative<std::monostate>(buffer);
  }

  [[nodiscard]] bool is_fragmented() const noexcept
  {
    return std::holds_alternative<std::span<cache::cached_chunk*>>(buffer);
  }

  [[nodiscard]] bool is_contiguous() const noexcept
  {
    return std::holds_alternative<std::uint8_t*>(buffer);
  }

  [[nodiscard]] bool needs_staging() const noexcept
  {
    return std::holds_alternative<std::monostate>(buffer);
  }

  [[nodiscard]] bool is_fragmented() const noexcept
  {
    return std::holds_alternative<std::span<cache::cached_chunk*>>(buffer);
  }

  [[nodiscard]] bool is_contiguous() const noexcept
  {
    return std::holds_alternative<std::uint8_t*>(buffer);
  }

  std::variant<std::monostate, std::uint8_t*, std::span<cache::cached_chunk*>> buffer;
};

struct device_buffer {
  device_buffer() noexcept = default;

  explicit device_buffer(std::uint8_t* dst, rmm::cuda_stream_view stream) noexcept
    : data(dst), stream(stream)
  {
    assert(dst != nullptr);
  }

  std::uint8_t* data{nullptr};
  rmm::cuda_stream_view stream;
};

struct prepared_io_slice {
  range rng;
  host_buffer h_buffer;  // monostate if using staged buffers
  device_buffer d_buffer;

  prepared_io_slice() noexcept = default;
  explicit prepared_io_slice(range r, host_buffer h) noexcept : rng(r), h_buffer(std::move(h)) {}
  explicit prepared_io_slice(range r, device_buffer d) noexcept : rng(r), d_buffer(std::move(d)) {}
  explicit prepared_io_slice(range r, host_buffer h, device_buffer d) noexcept
    : rng(r), h_buffer(std::move(h)), d_buffer(std::move(d))
  {
  }

  [[nodiscard]] bool is_staged() const noexcept { return h_buffer.needs_staging(); }

  [[nodiscard]] bool is_fragmented() const noexcept { return h_buffer.is_fragmented(); }

  [[nodiscard]] bool is_contiguous() const noexcept { return h_buffer.is_contiguous(); }

  [[nodiscard]] bool has_host_request() const noexcept { return !h_buffer.needs_staging(); }

  [[nodiscard]] bool has_device_request() const noexcept { return d_buffer.data != nullptr; }

  [[nodiscard]] bool is_host_request() const noexcept
  {
    return !h_buffer.needs_staging() && !has_device_request();
  }

  [[nodiscard]] size_t size() const noexcept { return rng.size; }

  [[nodiscard]] size_t offset() const noexcept { return rng.offset; }
};

}  // namespace sirius::io
