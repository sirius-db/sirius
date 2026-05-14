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
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <liburing.h>

#include <array>
#include <atomic>
#include <memory>
#include <span>
#include <string>
#include <string_view>
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

// ---- bounce_slot -----------------------------------------------------------

/**
 * @brief One pinned-memory staging buffer with a completion flag.
 *
 * The buffer is a non-owning pointer into a block owned by the reactor's
 * @c fixed_size_host_memory_resource allocation — the resource frees the
 * memory when the reactor is destroyed.
 */
struct bounce_slot {
  void* buf{nullptr};
  std::atomic<bool> cuda_done{false};
  // Status captured by cuda_copy_cb at callback time.  Written before the
  // release-store on cuda_done, read after the acquire-load — so the
  // happens-before chain piggybacks on cuda_done.  Lets poll_cuda use the
  // stream's state when our copy finished, ignoring later unrelated work
  // the consumer may have queued onto the same stream.
  cudaError_t cuda_status{cudaSuccess};
};

// ---------------------------------------------------------------------------
// uring_io_object
// ---------------------------------------------------------------------------

/**
 * @brief Concrete @c sirius_io_object backed by a filesystem path.
 *
 * Passive bag of native handles.  The buffered @c O_RDONLY fd
 * (for @c pread / host_read) and the @c O_DIRECT fd (for reactor-driven
 * device reads) are produced by @c uring_reactor::create_io_object — this
 * class does no I/O of its own.
 */
class uring_io_object : public sirius_io_object {
 public:
  uring_io_object(std::string path, file_descriptor fd, file_descriptor fd_direct, size_t file_size)
    : _path(std::move(path)),
      _fd(std::move(fd)),
      _fd_direct(std::move(fd_direct)),
      _file_size(file_size)
  {
  }

  [[nodiscard]] const std::string& raw_file_cache_id() const noexcept override { return _path; }
  [[nodiscard]] const std::string& object_path() const noexcept override { return _path; }
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

  /// Bounce slots are allocated from @p mr; their size is taken from
  /// @c mr.get_block_size().  The reactor keeps the @c multiple_blocks_allocation
  /// alive for its lifetime — blocks return to the resource on destruction.
  explicit uring_reactor(cucascade::memory::fixed_size_host_memory_resource& mr,
                         unsigned ring_entries = 64);

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

  /// Whether @p path can be served by this reactor.  Local-disk only:
  /// returns true iff the path refers to an existing, accessible file.
  [[nodiscard]] static bool supports(std::string_view path);

  /// Open the buffered + O_DIRECT fds for @p path and return them
  /// packaged in a @c uring_io_object.  Throws on unsupported paths or
  /// open() failure.
  static std::unique_ptr<uring_io_object> create_io_object(std::string path);

  /// fstat the open fd to get the file's current size.
  static size_t size(int fd);

 private:
  void worker_loop();

  struct cb_arg {
    uring_reactor* self;
    int slot;
  };
  // cudaStreamAddCallback signature (deprecated but used deliberately).
  // Unlike cudaLaunchHostFunc, the callback fires even when the stream is
  // already in an error state — so we never strand a slot in COPYING
  // waiting for a callback that wouldn't otherwise come.
  static void cuda_copy_cb(cudaStream_t stream, cudaError_t status, void* p) noexcept;

  // Keeps the bounce-slot blocks alive for the reactor's lifetime.  The
  // multiple_blocks_allocation destructor returns the blocks to the upstream
  // resource when the reactor is destroyed.
  cucascade::memory::fixed_multiple_blocks_allocation _bounce_storage;
  std::size_t _bounce_slot_size;
  std::array<bounce_slot, NUM_CHUNKS> _bounce;
  std::array<cb_arg, NUM_CHUNKS> _cb_args;
  unsigned _ring_entries;
  std::atomic<uint64_t> _wake_seq{0};
  std::atomic<bool> _stop{false};
  std::thread _worker;
  duckdb_moodycamel::ConcurrentQueue<device_read_req_type> _queue;
  duckdb_moodycamel::ConcurrentQueue<host_read_req_type> _host_queue;
};

}  // namespace sirius::io
