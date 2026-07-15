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

#include "io/object_store_config.hpp"
#include "io/rdma/cuobj_rdma_reactor.hpp"
#include "io/rdma/rdma_client.hpp"
#include "io/templated_ioctx.hpp"

namespace sirius::io::s3 {

/**
 * @brief S3-over-RDMA ioctx. Specialisation of
 *        @c templated_ioctx<rdma::cuobj_rdma_reactor>.
 *
 * Registered for the `s3://` scheme instead of the REST backend when
 * `object_store_config::s3_transport == transport::RDMA`.  One reactor owns the
 * whole worker pool (`s3_rdma_max_inflight` blocking workers = the global
 * in-flight ceiling); GPU affinity lives in the per-device landing arenas, not
 * the reactor count.  The capability profile is structural — device reads
 * supported, the staged host-to-device and vector host-read paths deliberately
 * absent — so the prefetch cache is never built for this backend.
 *
 * Without a configured @c rdma_client every data path fails loudly with a
 * "not implemented" error (the transport-selection contract); with one, reads
 * are served through the client (the mock in tests, cuObject later).
 */
class s3_rdma_ioctx final : public templated_ioctx<rdma::cuobj_rdma_reactor> {
 public:
  /// @p delivery is the CUDA delivery seam (F01) — construction-time only, no
  /// setter; defaults to the real CUDA runtime.  Throws std::invalid_argument
  /// when a member was nulled out.
  explicit s3_rdma_ioctx(object_store_config cfg,
                         std::shared_ptr<rdma::rdma_client> client = nullptr,
                         rdma::cuda_delivery_ops delivery          = {});

  [[nodiscard]] io_context_type type() const noexcept override { return io_context_type::rdma; }

  /// Pool-aggregated transfer counters (single reactor today; summed if the
  /// pool ever grows).
  [[nodiscard]] rdma::rdma_perf_snapshot perf_snapshot() const noexcept;

  /// Fail-fast without a client: these three intercept before the base builds
  /// any request, so no transfer machinery (and no backend-typed io_object
  /// access) is reached until the transport exists.
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

  /// The two staged paths are structurally unsupported for this backend; keep
  /// the transport-selection error shape ("RDMA ... not implemented") instead
  /// of the generic unsupported-operation message.
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
  /// Parse s3://bucket/key, HEAD it through the client for the size, and build
  /// the io_object.  Throws a "not implemented" error when no client is
  /// configured; propagates HEAD failures (missing object) otherwise.
  std::shared_ptr<sirius_io_object> create_io_object(std::string path) override;

 private:
  std::shared_ptr<rdma::rdma_client> _client;
};

}  // namespace sirius::io::s3
