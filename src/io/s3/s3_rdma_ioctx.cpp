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
                            ": the S3 RDMA transport is not implemented yet");
}

rdma::cuobj_rdma_reactor::config reactor_config_from(const object_store_config& cfg)
{
  return rdma::cuobj_rdma_reactor::config{cfg.s3_rdma_max_inflight, cfg.s3_rdma_arena_slot_size};
}

}  // namespace

s3_rdma_ioctx::s3_rdma_ioctx(object_store_config cfg,
                             std::shared_ptr<rdma::rdma_client> client,
                             rdma::cuda_delivery_ops delivery)
  : templated_ioctx<rdma::cuobj_rdma_reactor>(
      1,
      [ctx = std::make_shared<rdma::cuobj_rdma_reactor::reactor_context>(
         reactor_config_from(cfg), client, std::move(delivery))] {
        return std::make_unique<rdma::cuobj_rdma_reactor>(ctx);
      }),
    _client(std::move(client))
{
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
    total.fallback_stream_sync_total += s.fallback_stream_sync_total;
    total.delivery_fatal_total += s.delivery_fatal_total;
    total.arena_leak_total += s.arena_leak_total;
  }
  return total;
}

size_t s3_rdma_ioctx::host_read_io(const sirius_io_object& obj,
                                   size_t offset,
                                   size_t size,
                                   uint8_t* dst)
{
  if (!_client) { throw not_implemented("host_read_io"); }
  return templated_ioctx<rdma::cuobj_rdma_reactor>::host_read_io(obj, offset, size, dst);
}

exec::semi_future<size_t> s3_rdma_ioctx::host_read_async_io(const sirius_io_object& obj,
                                                            size_t offset,
                                                            size_t size,
                                                            uint8_t* dst) noexcept
{
  if (!_client) {
    return exec::make_semi_future<size_t>(
      std::make_exception_ptr(not_implemented("host_read_async_io")));
  }
  return templated_ioctx<rdma::cuobj_rdma_reactor>::host_read_async_io(obj, offset, size, dst);
}

exec::semi_future<size_t> s3_rdma_ioctx::device_read_async_io(const sirius_io_object& obj,
                                                              size_t offset,
                                                              size_t size,
                                                              uint8_t* dst,
                                                              rmm::cuda_stream_view stream) noexcept
{
  if (!_client) {
    return exec::make_semi_future<size_t>(
      std::make_exception_ptr(not_implemented("device_read_async_io")));
  }
  return templated_ioctx<rdma::cuobj_rdma_reactor>::device_read_async_io(
    obj, offset, size, dst, stream);
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
  if (!_client) { throw not_implemented("create_io_object"); }

  auto parsed = parse(path);
  if (parsed.scheme != "s3") {
    throw std::invalid_argument("s3_rdma_ioctx::create_io_object: unsupported scheme '" +
                                parsed.scheme + "'");
  }
  const size_t size = _client->head(parsed.host, parsed.path);
  return std::make_shared<rdma::cuobj_rdma_io_object>(
    std::move(path), std::move(parsed.host), std::move(parsed.path), size);
}

}  // namespace sirius::io::s3
