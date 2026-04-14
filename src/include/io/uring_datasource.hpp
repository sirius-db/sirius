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

#include "io/types.hpp"
#include "io/uring_reactor.hpp"

#include <cudf/io/datasource.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace sirius::io {

// ---------------------------------------------------------------------------
// ring_pool
// ---------------------------------------------------------------------------

/**
 * @brief Pool of @c io_uring instances for host (buffered) reads.
 */
class ring_pool {
 public:
  struct guard {
    guard(ring_pool* p, size_t i) noexcept : _pool(p), _idx(i) {}

    io_uring& ring() const { return _pool->_rings[_idx]; }

    ~guard()
    {
      if (_pool) _pool->release(_idx);
    }

    guard(guard const&)            = delete;
    guard& operator=(guard const&) = delete;
    guard(guard&& o) noexcept : _pool(std::exchange(o._pool, nullptr)), _idx(o._idx) {}

   private:
    ring_pool* _pool{nullptr};
    size_t _idx{0};
  };

  ring_pool()                            = default;
  ring_pool(ring_pool const&)            = delete;
  ring_pool& operator=(ring_pool const&) = delete;

  void init(size_t n_rings, unsigned ring_entries);
  ~ring_pool();
  guard acquire();

 private:
  void release(size_t idx);

  std::unique_ptr<io_uring[]> _rings;
  std::unique_ptr<bool[]> _in_use;
  size_t _n{0};
  std::mutex _mtx;
  std::condition_variable _cv;
};

// ---------------------------------------------------------------------------
// uring_io_object
// ---------------------------------------------------------------------------

/**
 * @brief Concrete @c sirius_io_object backed by filesystem paths.
 *
 * Opens a buffered @c O_RDONLY fd and an @c O_DIRECT fd at construction time.
 */
class uring_io_object : public sirius_io_object {
 public:
  /**
   * @brief Opens the file (both buffered and O_DIRECT paths).
   * @param path Absolute or relative path to the file.
   */
  explicit uring_io_object(std::string path);

  [[nodiscard]] const std::string& raw_file_cache_id() const noexcept override { return _path; }
  [[nodiscard]] size_t size() const noexcept override { return _file_size; }

  [[nodiscard]] int fd() const noexcept { return _fd.get(); }
  [[nodiscard]] int fd_direct() const noexcept { return _fd_direct.get(); }

  [[nodiscard]] uring_reactor* reactor() const noexcept { return _reactor; }
  void set_reactor(uring_reactor* r) noexcept { _reactor = r; }

 private:
  std::string _path;
  file_descriptor _fd;
  file_descriptor _fd_direct;
  size_t _file_size{0};
  uring_reactor* _reactor{nullptr};
};

// ---------------------------------------------------------------------------
// uring_ioctx
// ---------------------------------------------------------------------------

/**
 * @brief Concrete @c sirius_ioctx backed by io_uring.
 *
 * Owns a @c ring_pool for buffered host reads and a vector of @c uring_reactor
 * instances for O_DIRECT device reads.  Implements all read APIs from
 * @c sirius_ioctx, operating on a @c uring_io_object.
 */
class uring_ioctx : public sirius_ioctx {
 public:
  explicit uring_ioctx(unsigned host_ring_depth = 16,
                       unsigned ring_entries    = 64,
                       size_t n_reactors        = 4,
                       size_t bounce_slot_size  = 1UL * 1024 * 1024);

  std::unique_ptr<cudf::io::datasource> make_datasource(
    std::unique_ptr<sirius_io_object> io_object) override;

  void shutdown() override;

  ring_pool::guard acquire_host_ring();
  uring_reactor& assign_reactor();

  // -- Read API implementations -----------------------------------------------

  size_t host_read(sirius_io_object& obj, size_t offset, size_t size, uint8_t* dst) override;
  std::unique_ptr<cudf::io::datasource::buffer> host_read(sirius_io_object& obj,
                                                          size_t offset,
                                                          size_t size) override;

  std::future<size_t> host_read_async(sirius_io_object& obj,
                                      size_t offset,
                                      size_t size,
                                      uint8_t* dst) override;
  std::future<std::unique_ptr<cudf::io::datasource::buffer>> host_read_async(sirius_io_object& obj,
                                                                             size_t offset,
                                                                             size_t size) override;

  std::unique_ptr<cudf::io::datasource::buffer> device_read(sirius_io_object& obj,
                                                            size_t offset,
                                                            size_t size,
                                                            rmm::cuda_stream_view stream) override;
  size_t device_read(sirius_io_object& obj,
                     size_t offset,
                     size_t size,
                     uint8_t* dst,
                     rmm::cuda_stream_view stream) override;
  std::future<size_t> device_read_async(sirius_io_object& obj,
                                        size_t offset,
                                        size_t size,
                                        uint8_t* dst,
                                        rmm::cuda_stream_view stream) override;

  std::future<size_t> host_read_ranges_async(
    sirius_io_object& obj,
    std::vector<cudf::io::text::byte_range_info> const& ranges,
    std::span<cudf::host_span<std::byte>> dst) override;
  size_t host_read_ranges(sirius_io_object& obj,
                          std::vector<cudf::io::text::byte_range_info> const& ranges,
                          std::span<cudf::host_span<std::byte>> dst) override;

 private:
  std::future<size_t> enqueue_device_read(
    uring_io_object& obj, size_t offset, size_t size, uint8_t* dst, cudaStream_t stream);

  static uring_io_object& as_uring(sirius_io_object& obj);

  ring_pool _host_pool;
  std::vector<std::unique_ptr<uring_reactor>> _reactors;
  std::atomic<size_t> _next{0};
  unsigned _ring_entries;
};

// ---------------------------------------------------------------------------
// uring_datasource
// ---------------------------------------------------------------------------

/**
 * @brief Concrete @c sirius_datasource backed by io_uring.
 *
 * Thin delegate: every read method forwards to @c sirius_ioctx, passing the
 * owned @c sirius_io_object by reference.
 */
class uring_datasource : public sirius_datasource {
 public:
  static constexpr size_t NUM_BUFFERS = NUM_CHUNKS;
  static constexpr size_t BUFFER_SIZE = CHUNK_SIZE;

  explicit uring_datasource(std::shared_ptr<sirius_ioctx> io_ctx,
                            std::unique_ptr<sirius_io_object> io_object);

  ~uring_datasource() override = default;

  uring_datasource(uring_datasource const&)            = delete;
  uring_datasource& operator=(uring_datasource const&) = delete;

  // ---- Context accessors ---------------------------------------------------

  [[nodiscard]] std::shared_ptr<sirius_ioctx> io_ctx() const { return _io_ctx; }

  [[nodiscard]] std::shared_ptr<uring_ioctx> uring_ctx() const
  {
    return std::dynamic_pointer_cast<uring_ioctx>(_io_ctx);
  }

  // ---- cudf::io::datasource overrides ---------------------------------------

  [[nodiscard]] size_t size() const override;
  [[nodiscard]] bool supports_device_read() const override;
  [[nodiscard]] bool is_device_read_preferred(size_t) const override;

  size_t host_read(size_t offset, size_t size, uint8_t* dst) override;
  std::unique_ptr<datasource::buffer> host_read(size_t offset, size_t size) override;
  std::future<size_t> host_read_async(size_t offset, size_t size, uint8_t* dst) override;
  std::future<std::unique_ptr<datasource::buffer>> host_read_async(size_t offset,
                                                                   size_t size) override;

  std::unique_ptr<datasource::buffer> device_read(size_t offset,
                                                  size_t size,
                                                  rmm::cuda_stream_view stream) override;
  size_t device_read(size_t offset,
                     size_t size,
                     uint8_t* dst,
                     rmm::cuda_stream_view stream) override;
  std::future<size_t> device_read_async(size_t offset,
                                        size_t size,
                                        uint8_t* dst,
                                        rmm::cuda_stream_view stream) override;

  // ---- sirius_datasource overrides ------------------------------------------

  std::future<size_t> host_read_ranges_async(
    std::vector<cudf::io::text::byte_range_info> const& ranges,
    std::span<cudf::host_span<std::byte>> dst) override;

  size_t host_read_ranges(std::vector<cudf::io::text::byte_range_info> const& ranges,
                          std::span<cudf::host_span<std::byte>> dst) override;

 private:
  std::shared_ptr<sirius_ioctx> _io_ctx;
  std::unique_ptr<sirius_io_object> _io_object;
};

}  // namespace sirius::io
