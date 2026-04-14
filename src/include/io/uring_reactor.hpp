/*
 * Copyright 2025, Sirius Contributors.
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

#include "concurrentqueue.h"
#include "io/types.hpp"

#include <cuda_runtime.h>

#include <liburing.h>

#include <array>
#include <atomic>
#include <memory>
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
// uring_reactor
// ---------------------------------------------------------------------------

/**
 * @brief Single-threaded I/O reactor for O_DIRECT device reads.
 *
 * Owns one @c io_uring (O_DIRECT), one worker thread, @c NUM_CHUNKS pinned
 * bounce slots, and an MPSC request queue. Implements @c sirius_io_reactor.
 */
class uring_reactor : public sirius_io_reactor {
 public:
  explicit uring_reactor(unsigned ring_entries = 64, size_t bounce_slot_size = CHUNK_SIZE);

  ~uring_reactor() override;

  uring_reactor(uring_reactor const&)            = delete;
  uring_reactor& operator=(uring_reactor const&) = delete;

  void interrupt() override;
  void shutdown() override;
  void enqueue(device_read_req req) override;
  void enqueue_host(host_read_req req) override;

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
  moodycamel::ConcurrentQueue<device_read_req> _queue;
  moodycamel::ConcurrentQueue<host_read_req> _host_queue;
  std::atomic<uint64_t> _cuda_seq{0};
};

}  // namespace sirius::io
