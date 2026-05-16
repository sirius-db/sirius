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

#include <rmm/cuda_stream_view.hpp>

#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace sirius::io {

class buffer_pool;
class prefetching_cache;

// ---------------------------------------------------------------------------
// sirius_ioctx
// ---------------------------------------------------------------------------

/**
 * @brief Abstract shared context passed to every datasource.
 *
 * Holds resources that are shared across all datasources (cache, reactor
 * threads, ...). Extend this class to provide a concrete I/O backend.
 */
class sirius_ioctx : public std::enable_shared_from_this<sirius_ioctx> {
 public:
  sirius_ioctx();
  virtual ~sirius_ioctx();

  virtual void shutdown() = 0;

  /// Backend-specific factory: open native handles for @p path and
  /// return a populated io_object.  Throws on unsupported / unreachable
  /// paths (callers that want a check-without-open should use
  /// @c supports()).
  virtual std::shared_ptr<sirius_io_object> create_io_object(std::string path) = 0;

  virtual std::unique_ptr<cudf::io::datasource> make_datasource(
    std::shared_ptr<sirius_io_object> io_object) = 0;

  /// Convenience: @c create_io_object + @c make_datasource in one shot.
  std::unique_ptr<cudf::io::datasource> open_datasource(std::string path)
  {
    return make_datasource(create_io_object(std::move(path)));
  }

  /// Whether this backend can serve reads for @p path.  Backends should
  /// validate scheme/protocol support and any backend-specific
  /// preconditions (e.g. file existence for local-disk backends).
  [[nodiscard]] virtual bool supports(std::string_view path) const = 0;

  /// Construct the owned prefetching_cache.  Must be called before any
  /// read that should consult the cache; until then device_read falls
  /// through directly to device_read_io.
  void initialize_cache(buffer_pool& pool, size_t inflight_budget_chunks = 2048);

  [[nodiscard]] prefetching_cache* cache() noexcept { return _cache.get(); }

  // -- Read API ---------------------------------------------------------------

  size_t host_read(sirius_io_object& obj, size_t offset, size_t size, uint8_t* dst);

  std::future<size_t> host_read_async(sirius_io_object& obj,
                                      size_t offset,
                                      size_t size,
                                      uint8_t* dst);

  size_t device_read(
    sirius_io_object& obj, size_t offset, size_t size, uint8_t* dst, rmm::cuda_stream_view stream);

  std::future<size_t> device_read_async(
    sirius_io_object& obj, size_t offset, size_t size, uint8_t* dst, rmm::cuda_stream_view stream);

  ///

  virtual size_t host_read_io(sirius_io_object& obj, size_t offset, size_t size, uint8_t* dst) = 0;

  virtual void host_read_async_io(sirius_io_object& obj,
                                  size_t offset,
                                  size_t size,
                                  uint8_t* dst,
                                  io_completion_handler handler) = 0;

  virtual size_t device_read_io(sirius_io_object& obj,
                                size_t offset,
                                size_t size,
                                uint8_t* dst,
                                rmm::cuda_stream_view stream) = 0;

  virtual void device_read_async_io(sirius_io_object& obj,
                                    size_t offset,
                                    size_t size,
                                    uint8_t* dst,
                                    rmm::cuda_stream_view stream,
                                    io_completion_handler handler) = 0;

  virtual void host_read_ranges_async_io(sirius_io_object& obj,
                                         std::vector<cudf::io::text::byte_range_info> const& ranges,
                                         std::span<cudf::host_span<std::byte>> dst,
                                         io_completion_handler handler) = 0;

  // -- Physical range alignment ------------------------------------------------

  virtual cudf::io::text::byte_range_info compute_physical_range(
    cudf::io::text::byte_range_info logical, size_t file_size) const = 0;

 protected:
  std::unique_ptr<prefetching_cache> _cache;
};

}  // namespace sirius::io
