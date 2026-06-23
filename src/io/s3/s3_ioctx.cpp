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

#include "io/s3/s3_ioctx.hpp"

#include "io/s3/s3_blocking_ioctx.hpp"  // s3_ioctx_config
#include "io/uri_parser.hpp"

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sirius::io::s3 {

s3_ioctx::s3_ioctx(std::shared_ptr<s3_request_authorizer> creds,
                   long request_timeout_s,
                   std::string ca_bundle_path,
                   bool tls_verify,
                   std::size_t max_connections,
                   cucascade::memory::fixed_size_host_memory_resource* host_mr,
                   std::optional<std::size_t> max_retry_attempts,
                   std::optional<std::chrono::milliseconds> retry_backoff_base,
                   std::optional<std::chrono::milliseconds> retry_jitter,
                   std::optional<bool> honor_retry_after)
  : templated_ioctx<s3_reactor>(1,
                                [creds = std::move(creds),
                                 request_timeout_s,
                                 ca_bundle_path = std::move(ca_bundle_path),
                                 tls_verify,
                                 max_connections,
                                 host_mr,
                                 max_retry_attempts,
                                 retry_backoff_base,
                                 retry_jitter,
                                 honor_retry_after]() {
                                  s3_reactor::config cfg;
                                  cfg.creds                = creds;
                                  cfg.request_timeout_s    = request_timeout_s;
                                  cfg.ca_bundle_path       = ca_bundle_path;
                                  cfg.tls_verify           = tls_verify;
                                  cfg.max_connections      = max_connections;
                                  cfg.host_memory_resource = host_mr;
                                  // Only override the reactor's own retry defaults
                                  // when explicitly provided (no duplicated literals).
                                  if (max_retry_attempts)
                                    cfg.max_retry_attempts = *max_retry_attempts;
                                  if (retry_backoff_base)
                                    cfg.retry_backoff_base = *retry_backoff_base;
                                  if (retry_jitter) cfg.retry_jitter = *retry_jitter;
                                  if (honor_retry_after) cfg.honor_retry_after = *honor_retry_after;
                                  return std::make_unique<s3_reactor>(std::move(cfg));
                                })
{
}

s3_ioctx::s3_ioctx(s3_ioctx_config config)
  : templated_ioctx<s3_reactor>(1, [config = std::move(config)]() {
      s3_reactor::config cfg;
      cfg.creds                   = config.creds;
      cfg.request_timeout_s       = config.request_timeout_s;
      cfg.ca_bundle_path          = config.ca_bundle_path;
      cfg.tls_verify              = config.tls_verify;
      cfg.max_connections         = config.max_connections;
      cfg.host_memory_resource    = config.host_memory_resource;
      cfg.max_retry_attempts      = config.max_retry_attempts;
      cfg.retry_backoff_base      = config.retry_backoff_base;
      cfg.retry_jitter            = config.retry_jitter;
      cfg.honor_retry_after       = config.honor_retry_after;
      cfg.s3_perf_instrumentation = config.s3_perf_instrumentation;
      return std::make_unique<s3_reactor>(std::move(cfg));
    })
{
}

std::shared_ptr<sirius_io_object> s3_ioctx::create_io_object(std::string path)
{
  auto parsed = sirius::io::parse(path);
  if (parsed.scheme != "s3") {
    throw std::invalid_argument("s3_ioctx::create_io_object: unsupported scheme '" + parsed.scheme +
                                "'");
  }
  auto size  = reactor().head_object_size(parsed.host, parsed.path);
  auto state = std::make_shared<s3_object_state>(
    s3_object_state{std::move(parsed.host), std::move(parsed.path), size});
  return std::make_shared<s3_async_io_object>(std::move(path), std::move(state));
}

void s3_ioctx::host_read_ranges_async_io(sirius_io_object& obj,
                                         std::vector<cudf::io::text::byte_range_info> const& ranges,
                                         std::span<cudf::host_span<std::byte>> dst,
                                         io_completion_handler handler)
{
  // Hard contract: never sync-throw; all errors via the handler's exception_ptr;
  // handler fires exactly once.
  if (ranges.size() != dst.size()) {
    handler(0,
            std::make_exception_ptr(
              std::invalid_argument("s3_ioctx::host_read_ranges: ranges/dst size mismatch")));
    return;
  }
  if (ranges.empty()) {
    handler(0, nullptr);  // zero-work success may complete inline
    return;
  }

  auto* tobj = dynamic_cast<s3_async_io_object*>(&obj);
  if (tobj == nullptr) {
    handler(0,
            std::make_exception_ptr(
              std::invalid_argument("s3_ioctx::host_read_ranges: io_object is not an "
                                    "s3_async_io_object")));
    return;
  }
  auto const file_size = tobj->size();

  std::vector<s3_reactor::host_read_req_type> reqs;
  reqs.reserve(ranges.size());
  std::size_t total = 0;
  for (std::size_t i = 0; i < ranges.size(); ++i) {
    auto const off    = static_cast<std::size_t>(ranges[i].offset());
    auto const wanted = static_cast<std::size_t>(ranges[i].size());
    auto const sz     = std::min(wanted, file_size > off ? file_size - off : std::size_t{0});
    if (dst[i].size() < sz) {
      handler(0,
              std::make_exception_ptr(
                std::invalid_argument("s3_ioctx::host_read_ranges: dst span too small")));
      return;
    }
    if (sz == 0) continue;
    s3_reactor::host_read_req_type req;
    req.handle = tobj->host_handle();
    req.offset = off;
    req.size   = sz;
    req.dst    = reinterpret_cast<std::uint8_t*>(dst[i].data());
    reqs.push_back(std::move(req));
    total += sz;
  }

  auto ctx = request_context::create(reqs.size(), total, std::move(handler));
  if (!ctx) return;  // zero valid reqs -> create() already fired the handler
  for (auto& r : reqs)
    r.ctx = ctx;
  reactor().host_enqueue_bulk(std::span<s3_reactor::host_read_req_type>(reqs.data(), reqs.size()));
}

std::uint64_t s3_ioctx::bytes_read_total() const noexcept
{
  std::uint64_t total = 0;
  for (auto const& r : _reactors)
    total += r->bytes_read_total();
  return total;
}

std::uint64_t s3_ioctx::fsmr_borrows_total() const noexcept
{
  std::uint64_t total = 0;
  for (auto const& r : _reactors)
    total += r->fsmr_borrows_total();
  return total;
}

std::uint64_t s3_ioctx::device_copies_total() const noexcept
{
  std::uint64_t total = 0;
  for (auto const& r : _reactors)
    total += r->device_copies_total();
  return total;
}

std::uint64_t s3_ioctx::device_stream_sync_total() const noexcept
{
  std::uint64_t total = 0;
  for (auto const& r : _reactors)
    total += r->device_stream_sync_total();
  return total;
}

std::uint64_t s3_ioctx::device_peak_inflight() const noexcept
{
  // sum the per-reactor peaks: a safe upper bound on concurrent staging blocks
  // (exact for the single-reactor S3 backend).
  std::uint64_t total = 0;
  for (auto const& r : _reactors)
    total += r->device_peak_inflight();
  return total;
}

std::size_t s3_ioctx::head_object_size(std::string_view bucket, std::string_view key)
{
  return reactor().head_object_size(bucket, key);
}

// -- Perf micro + retry accessors (sum totals/counts, max the maxima) --------
namespace {
template <typename Reactors, typename Fn>
std::uint64_t sum_reactors(Reactors const& reactors, Fn fn) noexcept
{
  std::uint64_t total = 0;
  for (auto const& r : reactors)
    total += fn(*r);
  return total;
}
template <typename Reactors, typename Fn>
std::uint64_t max_reactors(Reactors const& reactors, Fn fn) noexcept
{
  std::uint64_t hi = 0;
  for (auto const& r : reactors)
    hi = std::max(hi, fn(*r));
  return hi;
}
}  // namespace

std::uint64_t s3_ioctx::chunk_get_ns_total() const noexcept
{
  return sum_reactors(_reactors, [](auto const& r) { return r.chunk_get_ns_total(); });
}
std::uint64_t s3_ioctx::chunk_get_ns_count() const noexcept
{
  return sum_reactors(_reactors, [](auto const& r) { return r.chunk_get_ns_count(); });
}
std::uint64_t s3_ioctx::chunk_get_ns_max() const noexcept
{
  return max_reactors(_reactors, [](auto const& r) { return r.chunk_get_ns_max(); });
}
std::uint64_t s3_ioctx::queue_wait_ns_total() const noexcept
{
  return sum_reactors(_reactors, [](auto const& r) { return r.queue_wait_ns_total(); });
}
std::uint64_t s3_ioctx::queue_wait_ns_count() const noexcept
{
  return sum_reactors(_reactors, [](auto const& r) { return r.queue_wait_ns_count(); });
}
std::uint64_t s3_ioctx::h2d_observed_ns_total() const noexcept
{
  return sum_reactors(_reactors, [](auto const& r) { return r.h2d_observed_ns_total(); });
}
std::uint64_t s3_ioctx::h2d_observed_ns_count() const noexcept
{
  return sum_reactors(_reactors, [](auto const& r) { return r.h2d_observed_ns_count(); });
}
std::uint64_t s3_ioctx::h2d_observed_ns_max() const noexcept
{
  return max_reactors(_reactors, [](auto const& r) { return r.h2d_observed_ns_max(); });
}
std::uint64_t s3_ioctx::ttfb_ns() const noexcept
{
  return max_reactors(_reactors, [](auto const& r) { return r.ttfb_ns(); });
}
std::uint64_t s3_ioctx::retries_total() const noexcept
{
  return sum_reactors(_reactors, [](auto const& r) { return r.retries_total(); });
}
std::uint64_t s3_ioctx::terminal_failures_total() const noexcept
{
  return sum_reactors(_reactors, [](auto const& r) { return r.terminal_failures_total(); });
}

}  // namespace sirius::io::s3
