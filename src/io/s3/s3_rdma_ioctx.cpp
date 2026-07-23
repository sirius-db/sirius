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

#include "io/s3/s3_rdma_ioctx.hpp"

#include "io/uri_parser.hpp"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace sirius::io::s3 {

namespace {

std::runtime_error not_implemented(std::string_view entry_point)
{
  return std::runtime_error("s3_rdma_ioctx::" + std::string(entry_point) +
                            ": the S3 RDMA transport does not support this path");
}

rdma::cuobj_rdma_reactor::config reactor_config_from(const object_store_config& cfg)
{
  rdma::cuobj_rdma_reactor::config reactor_cfg;
  reactor_cfg.max_inflight    = cfg.s3_rdma_max_inflight;
  reactor_cfg.arena_slot_size = cfg.s3_rdma_arena_slot_size;
  reactor_cfg.queue_cap       = cfg.s3_rdma_queue_cap;
  return reactor_cfg;
}

}  // namespace

s3_rdma_ioctx::s3_rdma_ioctx(object_store_config cfg,
                             rdma::rdma_transport_clients clients,
                             rdma::cuda_delivery_ops delivery)
  : s3_rdma_ioctx(std::make_shared<rdma::cuobj_rdma_reactor::reactor_context>(
      reactor_config_from(cfg), std::move(clients), std::move(delivery)))
{
}

s3_rdma_ioctx::s3_rdma_ioctx(std::shared_ptr<rdma::cuobj_rdma_reactor::reactor_context> reactor_ctx)
  : templated_ioctx<rdma::cuobj_rdma_reactor>(
      1, [reactor_ctx] { return std::make_unique<rdma::cuobj_rdma_reactor>(reactor_ctx); }),
    _reactor_ctx(std::move(reactor_ctx))
{
}

void s3_rdma_ioctx::start()
{
  const auto& clients = _reactor_ctx->clients();
  if (!clients.control || !clients.data_sessions || clients.tag_predicate == nullptr) {
    const char* missing = !clients.control         ? "the control-plane client"
                          : !clients.data_sessions ? "the data-session factory"
                                                   : "the completion-tag predicate";
    throw std::runtime_error(std::string("s3_rdma_ioctx: RDMA transport initialization failed: ") +
                             missing + " capability is missing");
  }
  templated_ioctx<rdma::cuobj_rdma_reactor>::start();
}

rdma::rdma_perf_snapshot s3_rdma_ioctx::perf_snapshot() const noexcept
{
  rdma::rdma_perf_snapshot total;
  for (const auto& reactor : _reactors) {
    auto const s = reactor->perf_snapshot();
    total.bytes_total += s.bytes_total;
    total.requests_total += s.requests_total;
    total.retries_total += s.retries_total;
    total.short_read_total += s.short_read_total;
    total.error_total += s.error_total;
    total.slot_wait_total += s.slot_wait_total;
    total.flush_total += s.flush_total;
    total.inflight_peak = std::max(total.inflight_peak, s.inflight_peak);
    total.envelope_wait_total += s.envelope_wait_total;
    total.envelope_wait_ns_total += s.envelope_wait_ns_total;
    total.envelope_depth_peak = std::max(total.envelope_depth_peak, s.envelope_depth_peak);
    total.slots_in_use_peak   = std::max(total.slots_in_use_peak, s.slots_in_use_peak);
    total.fail_stop_total += s.fail_stop_total;
    total.arena_leak_total += s.arena_leak_total;
  }
  return total;
}

exec::semi_future<size_t> s3_rdma_ioctx::host_to_device_read_async_io(
  const sirius_io_object& /*obj*/,
  std::span<io_object_segment> /*slices*/,
  size_t /*offset*/,
  size_t /*size*/,
  uint8_t* /*device_dst*/,
  rmm::cuda_stream_view /*stream*/) noexcept
{
  return exec::make_semi_future<size_t>(
    std::make_exception_ptr(not_implemented("host_to_device_read_async_io")));
}

exec::semi_future<size_t> s3_rdma_ioctx::host_read_ranges_async_io(
  const sirius_io_object& /*obj*/, std::span<io_object_segment> /*segments*/) noexcept
{
  return exec::make_semi_future<size_t>(
    std::make_exception_ptr(not_implemented("host_read_ranges_async_io")));
}

std::shared_ptr<sirius_io_object> s3_rdma_ioctx::create_io_object(std::string path)
{
  auto parsed = parse(path);
  if (parsed.scheme != "s3") {
    throw std::invalid_argument("s3_rdma_ioctx::create_io_object: unsupported scheme '" +
                                parsed.scheme + "'");
  }
  // The control permit covers the full HEAD: a closed or failed transport
  // refuses new opens with its terminal error instead of touching the wire.
  auto permit = _reactor_ctx->gate().acquire_control();
  auto const result =
    _reactor_ctx->clients().control->head(rdma::rx_route{parsed.host, parsed.path});
  if (!result.outcome.transport_ok()) {
    throw std::runtime_error("s3_rdma_ioctx::create_io_object: " + path + ": " +
                             result.outcome.transport_error);
  }
  if (result.outcome.http_status != 200) {
    throw std::runtime_error("s3_rdma_ioctx::create_io_object: " + path + " -> HTTP " +
                             std::to_string(result.outcome.http_status));
  }
  return std::make_shared<rdma::cuobj_rdma_io_object>(
    std::move(path), std::move(parsed.host), std::move(parsed.path), result.object_size);
}

void s3_rdma_ioctx::on_device_dispatch_failure() noexcept
{
  // The dispatch exception itself carries no cudaError_t; on a poisoned
  // context every CUDA call returns the sticky code, so a cheap probe
  // recovers it.  Sticky => terminate (contract: any phase); anything else
  // keeps the framework's plain error-future behavior.
  int device           = -1;
  const cudaError_t rc = _reactor_ctx->delivery_ops().get_device(&device);
  if (rc != cudaSuccess && rdma::is_context_fatal(rc)) {
    rdma::invoke_fatal(
      _reactor_ctx->delivery_ops(), "device dispatch failed on a poisoned context", rc);
  }
}

}  // namespace sirius::io::s3
