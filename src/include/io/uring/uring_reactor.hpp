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

#include <cuda_runtime.h>

#include <concurrentqueue.h>
#include <liburing.h>

#include <array>
#include <atomic>
#include <memory>
#include <span>
#include <string>
#include <thread>

namespace {
inline void cuda_check(cudaError_t e, char const* file, int line)
{
  if (e != cudaSuccess)
    throw std::runtime_error(std::string("CUDA error ") + file + ":" + std::to_string(line) +
                             " – " + cudaGetErrorString(e));
}
}  // namespace

#define CUDA_CHECK(call) cuda_check((call), __FILE__, __LINE__)

namespace sirius::io {

// ---- RAII resource wrappers ------------------------------------------------

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
  int get() const noexcept { return fd; }
  explicit operator bool() const noexcept { return fd >= 0; }
};

/**
 * @brief Custom deleter for @c unique_ring: calls @c io_uring_queue_exit
 *        before freeing the allocation.
 */
struct ring_deleter {
  void operator()(io_uring* r) const noexcept
  {
    io_uring_queue_exit(r);
    delete r;
  }
};
using unique_ring = std::unique_ptr<io_uring, ring_deleter>;

/**
 * @brief Custom deleter for CUDA pinned (host) memory allocated with
 *        @c cudaHostAlloc.
 */
struct pinned_deleter {
  void operator()(void* p) const noexcept { cudaFreeHost(p); }
};
using unique_pinned_buf = std::unique_ptr<void, pinned_deleter>;

/**
 * @brief Converts a byte count to mebibytes.
 */
inline double to_mb(size_t bytes) noexcept
{
  return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

// ---- bounce_slot -----------------------------------------------------------

/**
 * @brief One pinned-memory staging buffer with a completion flag.
 */
struct bounce_slot {
  unique_pinned_buf buf;
  std::atomic<bool> cuda_done{false};
};

// ---------------------------------------------------------------------------
// uring_io_object
// ---------------------------------------------------------------------------

/**
 * @brief Concrete @c sirius_io_object backed by a filesystem path.
 *
 * Opens a buffered @c O_RDONLY fd (for @c pread / host_read) and an
 * @c O_DIRECT fd (for reactor-driven device reads).  The two file
 * descriptors are the native handles consumed by @c uring_reactor.
 */
class uring_io_object : public sirius_io_object {
 public:
  explicit uring_io_object(std::string path);

  [[nodiscard]] const std::string& raw_file_cache_id() const noexcept override { return _path; }
  [[nodiscard]] size_t size() const noexcept override { return _file_size; }

  [[nodiscard]] int fd() const noexcept { return _fd.get(); }
  [[nodiscard]] int fd_direct() const noexcept { return _fd_direct.get(); }

  // ---- templated_ioctx / io_object_c requirements -----------------------
  [[nodiscard]] int host_handle() const noexcept { return _fd.get(); }
  [[nodiscard]] int device_handle() const noexcept { return _fd_direct.get(); }

 private:
  std::string _path;
  file_descriptor _fd;
  file_descriptor _fd_direct;
  size_t _file_size{0};
};

// ---------------------------------------------------------------------------
// uring_reactor
// ---------------------------------------------------------------------------

/**
 * @brief Single-threaded I/O reactor for O_DIRECT device reads.
 *
 * Owns one @c io_uring (O_DIRECT), one worker thread, @c NUM_CHUNKS pinned
 * bounce slots, and an MPSC request queue.  Models the reactor concept
 * consumed by @c templated_ioctx.
 */
class uring_reactor {
 public:
  using native_handle_type   = int;
  using io_object_type       = uring_io_object;
  using device_read_req_type = device_read_req<native_handle_type>;
  using host_read_req_type   = host_read_req<native_handle_type>;

  explicit uring_reactor(unsigned ring_entries = 64, size_t bounce_slot_size = CHUNK_SIZE);

  ~uring_reactor();

  uring_reactor(uring_reactor const&)            = delete;
  uring_reactor& operator=(uring_reactor const&) = delete;

  void interrupt();
  void shutdown();
  /// Enqueue a whole batch of device read chunks with a single wake
  /// notification.  Preferred over single-op enqueues when a caller
  /// produces several chunks destined for this reactor — amortises the
  /// wait-atomic notify and uses moodycamel's enqueue_bulk path.
  /// The span's elements are moved out; the caller's backing storage
  /// must outlive this call but the contents are left in moved-from state.
  void enqueue_bulk(std::span<device_read_req_type> batch);

  /// Synchronous buffered host read (pread on @p fd).  Blocks the caller.
  size_t host_read(int fd, size_t offset, size_t size, uint8_t* dst);

  /// Async buffered host read.  Request completion fires via
  /// @c req.ctx->chunk_done / chunk_failed.
  void host_read_async(host_read_req_type req);

  /// Bulk counterpart of @c host_read_async — enqueue a batch of host reads
  /// with a single wake notification.  Mirrors @c enqueue_bulk's span /
  /// move-out semantics: caller owns the storage, elements are moved from.
  void host_enqueue_bulk(std::span<host_read_req_type> batch);

  /// O_DIRECT requires 4 KiB alignment of both file offset and length.
  static cudf::io::text::byte_range_info align_to_physical(cudf::io::text::byte_range_info logical,
                                                           size_t file_size);

 private:
  void worker_loop();

  struct cb_arg {
    uring_reactor* self;
    int slot;
  };
  static void cuda_copy_cb(void* p) noexcept;

  std::array<bounce_slot, NUM_CHUNKS> _bounce;
  std::array<cb_arg, NUM_CHUNKS> _cb_args;
  unsigned _ring_entries;
  std::atomic<uint64_t> _wake_seq{0};
  std::atomic<bool> _stop{false};
  std::thread _worker;
  duckdb_moodycamel::ConcurrentQueue<device_read_req_type> _queue;
  duckdb_moodycamel::ConcurrentQueue<host_read_req_type> _host_queue;
  std::atomic<uint64_t> _cuda_seq{0};
};

}  // namespace sirius::io
