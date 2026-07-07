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
#include "io/object_store_config.hpp"

namespace sirius::io::s3 {

/**
 * @brief S3-over-RDMA backend, transport-selection stage.
 *
 * Registered for the `s3://` scheme instead of the REST backend when
 * `object_store_config::s3_transport == transport::RDMA`. Routing, the
 * capability profile, and cache gating are final: device reads are supported,
 * the staged host-to-device and vector host-read paths are not (so
 * `can_use_prefetching_cache()` is false and the prefetch cache is never built
 * for this backend). Every data path currently fails with a
 * "RDMA ... not implemented" error — selecting RDMA and touching S3 is a loud
 * error by design, never a silent fallback to another transport. The RDMA
 * reactor (registered landing arena + device-to-device delivery) replaces the
 * failing bodies without changing this surface.
 */
class s3_rdma_ioctx final : public sirius_ioctx {
 public:
  explicit s3_rdma_ioctx(object_store_config cfg);
  ~s3_rdma_ioctx() override;

  [[nodiscard]] io_context_type type() const noexcept override { return io_context_type::rdma; }

  void shutdown() noexcept override {}

  [[nodiscard]] bool supports(std::string_view path) const noexcept override;

  [[nodiscard]] bool supports_device_read() const noexcept override { return true; }
  [[nodiscard]] bool supports_host_to_device_read() const noexcept override { return false; }
  [[nodiscard]] bool supports_vector_host_read() const noexcept override { return false; }

  [[nodiscard]] cache::prefetching_stage preferred_prefetching_stage() const noexcept override
  {
    return cache::prefetching_stage::none;
  }

  [[nodiscard]] std::vector<cudf::io::text::byte_range_info> align_and_coalesce(
    std::span<const cudf::io::text::byte_range_info> ranges,
    std::optional<size_t> alignment) const noexcept override;

  size_t host_read_io(const sirius_io_object& obj,
                      size_t offset,
                      size_t size,
                      uint8_t* dst) override;

  exec::semi_future<size_t> host_read_async_io(const sirius_io_object& obj,
                                               size_t offset,
                                               size_t size,
                                               uint8_t* dst) noexcept override;

  exec::semi_future<size_t> device_read_async_io(const sirius_io_object& obj,
                                                 size_t offset,
                                                 size_t size,
                                                 uint8_t* dst,
                                                 rmm::cuda_stream_view stream) noexcept override;

  exec::semi_future<size_t> host_to_device_read_async_io(
    const sirius_io_object& obj,
    std::span<io_object_segment> slices,
    size_t offset,
    size_t size,
    uint8_t* device_dst,
    rmm::cuda_stream_view stream) noexcept override;

  exec::semi_future<size_t> host_read_ranges_async_io(
    const sirius_io_object& obj, std::span<io_object_segment> segments) noexcept override;

 protected:
  std::shared_ptr<sirius_io_object> create_io_object(std::string path) override;

 private:
  object_store_config _config;
};

}  // namespace sirius::io::s3
