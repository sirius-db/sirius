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

#include "io/io_context.hpp"

#include <cudf/io/datasource.hpp>

namespace sirius::io {

// ---------------------------------------------------------------------------
// sirius_datasource
// ---------------------------------------------------------------------------

/**
 * @brief Concrete @c io_datasource backed by io_uring.
 *
 * Thin delegate: every read method forwards to @c sirius_ioctx, passing the
 * owned @c sirius_io_object by reference.
 */
class sirius_datasource : public cudf::io::datasource {
 public:
  static constexpr size_t NUM_BUFFERS = NUM_CHUNKS;
  static constexpr size_t BUFFER_SIZE = CHUNK_SIZE;

  explicit sirius_datasource(std::shared_ptr<sirius_ioctx> io_ctx,
                             std::shared_ptr<sirius_io_object> io_object);

  ~sirius_datasource() override;

  sirius_datasource(sirius_datasource const&)            = delete;
  sirius_datasource& operator=(sirius_datasource const&) = delete;

  // ---- Context accessors ---------------------------------------------------

  [[nodiscard]] std::shared_ptr<sirius_ioctx> io_ctx() const { return _io_ctx; }

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

 private:
  std::shared_ptr<sirius_ioctx> _io_ctx;
  std::shared_ptr<sirius_io_object> _io_object;
};

}  // namespace sirius::io
