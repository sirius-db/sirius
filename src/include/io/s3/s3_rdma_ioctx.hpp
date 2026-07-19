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
#include "io/rdma/rdma_transport_client.hpp"
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
 * the reactor count.  The capability profile advertises device reads but
 * deliberately omits the staged host-to-device and vector host-read paths, so
 * the prefetch cache is never built for this backend.
 *
 * A missing transport capability fails loudly at @c start() rather than
 * falling back to another transport; host chunks ride the control plane and
 * device chunks the per-worker data sessions (mocks in tests, the
 * curl/cuObject-backed clients in production).
 */
class s3_rdma_ioctx final : public templated_ioctx<rdma::cuobj_rdma_reactor> {
 public:
  /// @p clients is the split transport bundle (control client + data-session
  /// factory + tag predicate); @p delivery is the CUDA delivery seam.  Both
  /// bind at construction only.  Construction never validates capabilities —
  /// @c start() does, so a misconfigured transport fails loudly on the
  /// routing path instead of at build time.
  s3_rdma_ioctx(object_store_config cfg,
                rdma::rdma_transport_clients clients,
                rdma::cuda_delivery_ops delivery = {});

  /// Validates the transport capabilities (control client and data-session
  /// factory present), then starts the reactor pool; a missing capability is
  /// an RDMA initialization error.
  void start() override;

  [[nodiscard]] io_context_type type() const noexcept override { return io_context_type::rdma; }

  /// Pool-aggregated transfer counters (single reactor today; summed if the
  /// pool ever grows).
  [[nodiscard]] rdma::rdma_perf_snapshot perf_snapshot() const noexcept;

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
  /// Parse s3://bucket/key, HEAD it through the control client for the size,
  /// and build the io_object.  A transport failure or a non-200 status
  /// (missing object) throws.
  std::shared_ptr<sirius_io_object> create_io_object(std::string path) override;

 private:
  /// Retained so the ioctx reaches the admission gate (control permits, the
  /// terminal error) and the transport bundle through the same context the
  /// reactor holds.
  explicit s3_rdma_ioctx(std::shared_ptr<rdma::cuobj_rdma_reactor::reactor_context> reactor_ctx);

  std::shared_ptr<rdma::cuobj_rdma_reactor::reactor_context> _reactor_ctx;
};

}  // namespace sirius::io::s3
