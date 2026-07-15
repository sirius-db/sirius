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

#include "io/rest/rest_ioctx.hpp"

#include "io/uri_parser.hpp"

#include <algorithm>
#include <cstddef>
#include <format>
#include <memory>
#include <stdexcept>
#include <utility>

namespace sirius::io::rest {

rest_ioctx::rest_ioctx(std::size_t n_reactors, std::shared_ptr<rest_reactor::reactor_context> ctx)
  : templated_ioctx<rest_reactor>(n_reactors, [ctx = std::move(ctx), i = 0]() mutable {
      return std::make_unique<rest_reactor>(ctx, std::format("rest-{}", i++));
    })
{
}

rest_perf_snapshot rest_ioctx::perf_snapshot() const noexcept
{
  rest_perf_snapshot agg;
  for (auto const& r : _reactors) {
    auto const s = r->perf_snapshot();
    agg.chunk_get_ns_total += s.chunk_get_ns_total;
    agg.chunk_get_count += s.chunk_get_count;
    agg.chunk_get_ns_max = std::max(agg.chunk_get_ns_max, s.chunk_get_ns_max);
    agg.queue_wait_ns_total += s.queue_wait_ns_total;
    agg.queue_wait_count += s.queue_wait_count;
    if (s.ttfb_ns != 0 && (agg.ttfb_ns == 0 || s.ttfb_ns < agg.ttfb_ns)) {
      agg.ttfb_ns = s.ttfb_ns;  // earliest (min non-zero) first-byte across the pool
    }
    agg.h2d_observed_ns_total += s.h2d_observed_ns_total;
    agg.h2d_observed_count += s.h2d_observed_count;
    agg.h2d_observed_ns_max = std::max(agg.h2d_observed_ns_max, s.h2d_observed_ns_max);
    agg.retries_total += s.retries_total;
    agg.terminal_failures_total += s.terminal_failures_total;
    agg.device_stream_sync_total += s.device_stream_sync_total;
    agg.payload_bytes_read_total += s.payload_bytes_read_total;
  }
  return agg;
}

std::shared_ptr<sirius_io_object> rest_ioctx::create_io_object(std::string path)
{
  auto parsed = sirius::io::parse(path);
  if (parsed.scheme != "s3") {
    throw std::invalid_argument("rest_ioctx::create_io_object: unsupported scheme '" +
                                parsed.scheme + "'");
  }
  if (_reactors.empty()) { throw std::runtime_error("rest_ioctx::create_io_object: no reactors"); }

  // A blocking HEAD on the caller thread (a one-time metadata round-trip) via
  // any reactor's authorizer — head_object_size uses a local easy handle and
  // does not touch worker state, so any reactor is equivalent.
  size_t const size = _reactors.front()->head_object_size(parsed.host, parsed.path);
  return std::make_shared<rest_io_object>(
    std::move(path), std::move(parsed.host), std::move(parsed.path), size);
}

std::shared_ptr<sirius_io_object> rest_ioctx::create_io_object(std::string path, open_hint hint)
{
  if (hint == open_hint::parquet_footer_probe) {
    return create_footer_probe_object(std::move(path));
  }
  return create_io_object(std::move(path));
}

std::shared_ptr<sirius_io_object> rest_ioctx::create_footer_probe_object(std::string path)
{
  auto parsed = sirius::io::parse(path);
  if (parsed.scheme != "s3") {
    throw std::invalid_argument("rest_ioctx::create_io_object: unsupported scheme '" +
                                parsed.scheme + "'");
  }
  if (_reactors.empty()) { throw std::runtime_error("rest_ioctx::create_io_object: no reactors"); }

  // One suffix-range GET resolves the size and stashes the footer; cuDF's
  // trailer/footer reads are then served from the stash by host_read.
  footer_probe probe = _reactors.front()->fetch_footer_suffix(
    parsed.host, parsed.path, _reactors.front()->get_config().footer_probe_bytes);
  if (!probe.bytes) {
    // Unusable suffix response (200 full body, 416, missing / "*" Content-Range):
    // fall back to a plain HEAD for the size, with no stash.
    size_t const size = _reactors.front()->head_object_size(parsed.host, parsed.path);
    return std::make_shared<rest_io_object>(
      std::move(path), std::move(parsed.host), std::move(parsed.path), size);
  }
  return std::make_shared<rest_io_object>(std::move(path),
                                          std::move(parsed.host),
                                          std::move(parsed.path),
                                          probe.object_size,
                                          probe.window_lo,
                                          probe.bytes);
}

}  // namespace sirius::io::rest
