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

#include "io/rest/s3/sigv4.hpp"
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
    agg.blocking_host_get_count += s.blocking_host_get_count;
    agg.blocking_host_get_wall_ns_total += s.blocking_host_get_wall_ns_total;
    agg.blocking_host_get_wall_ns_max =
      std::max(agg.blocking_host_get_wall_ns_max, s.blocking_host_get_wall_ns_max);
  }
  return agg;
}

void rest_ioctx::list_objects_paged(
  std::string_view bucket,
  std::string_view prefix,
  std::size_t page_size,
  std::function<bool(s3::list_objects_v2_page const&)> const& sink,
  std::optional<std::size_t> max_scanned)
{
  if (_reactors.empty()) { throw std::runtime_error("rest_ioctx::list_objects: no reactors"); }
  std::size_t const clamped = (page_size == 0 || page_size > 1000) ? 1000 : page_size;
  std::size_t const scanned_cap =
    max_scanned.value_or(_reactors.front()->get_config().list_max_scanned);

  std::size_t scanned = 0;
  std::string token;
  bool truncated = false;
  do {
    // SigV4 canonical order = byte order of the encoded keys; for these params
    // that is continuation-token < list-type < max-keys < prefix.
    std::string query;
    if (!token.empty()) {
      query += "continuation-token=";
      query += s3::uri_encode(token, /*encode_slash=*/true);
      query += '&';
    }
    query += "list-type=2&max-keys=";
    query += std::to_string(clamped);
    query += "&prefix=";
    query += s3::uri_encode(prefix, /*encode_slash=*/true);

    auto const page =
      s3::parse_list_objects_v2(_reactors.front()->list_page(bucket, prefix, query));

    scanned += page.entries.size();
    if (scanned > scanned_cap) {
      throw std::runtime_error("rest_ioctx::list_objects: scanned more than " +
                               std::to_string(scanned_cap) + " objects under s3://" +
                               std::string(bucket) + "/" + std::string(prefix) +
                               " — narrow the glob prefix");
    }
    if (page.is_truncated && page.next_continuation_token.empty()) {
      throw std::runtime_error(
        "rest_ioctx::list_objects: truncated ListObjectsV2 page without a continuation token for "
        "s3://" +
        std::string(bucket) + "/" + std::string(prefix));
    }
    // A truncated page must contain entries and advance the token. Together
    // with scanned_cap these bound pagination for non-conforming backends that
    // would otherwise loop on empty or non-advancing pages.
    if (page.is_truncated && page.entries.empty()) {
      throw std::runtime_error(
        "rest_ioctx::list_objects: truncated ListObjectsV2 page with no entries for s3://" +
        std::string(bucket) + "/" + std::string(prefix));
    }
    if (page.is_truncated && page.next_continuation_token == token) {
      throw std::runtime_error(
        "rest_ioctx::list_objects: ListObjectsV2 continuation token did not advance for s3://" +
        std::string(bucket) + "/" + std::string(prefix));
    }
    truncated = page.is_truncated;
    token     = page.next_continuation_token;
    if (!sink(page)) { return; }
  } while (truncated);
}

std::vector<s3::list_entry> rest_ioctx::list_objects(std::string_view bucket,
                                                     std::string_view prefix,
                                                     std::size_t page_size,
                                                     std::optional<std::size_t> max_keys)
{
  std::size_t const keys_cap = max_keys.value_or(list_max_matches());
  std::vector<s3::list_entry> out;
  list_objects_paged(bucket, prefix, page_size, [&](s3::list_objects_v2_page const& page) {
    if (out.size() + page.entries.size() > keys_cap) {
      throw std::runtime_error("rest_ioctx::list_objects: more than " + std::to_string(keys_cap) +
                               " objects under s3://" + std::string(bucket) + "/" +
                               std::string(prefix) + " — narrow the glob prefix");
    }
    out.insert(out.end(), page.entries.begin(), page.entries.end());
    return true;
  });
  return out;
}

std::size_t rest_ioctx::list_max_matches() const
{
  return _reactors.empty() ? s3::default_max_list_objects
                           : _reactors.front()->get_config().list_max_matches;
}

std::shared_ptr<io_object> rest_ioctx::create_io_object(std::string path)
{
  auto parsed = sirius::io::parse(path);
  if (parsed.scheme != "s3") {
    throw std::invalid_argument("rest_ioctx::create_io_object: unsupported scheme '" +
                                parsed.scheme + "'");
  }
  if (_reactors.empty()) { throw std::runtime_error("rest_ioctx::create_io_object: no reactors"); }

  // A blocking HEAD on the caller thread (a one-time metadata round-trip) via
  // any reactor's authorizer — head_object uses a local easy handle and
  // does not touch worker state, so any reactor is equivalent.
  auto head = _reactors.front()->head_object(parsed.host, parsed.path);
  return std::make_shared<rest_io_object>(std::move(path),
                                          std::move(parsed.host),
                                          std::move(parsed.path),
                                          head.object_size,
                                          std::move(head.etag));
}

std::shared_ptr<io_object> rest_ioctx::create_io_object(std::string path, open_hint hint)
{
  if (hint == open_hint::parquet_footer_probe) {
    return create_footer_probe_object(std::move(path));
  }
  return create_io_object(std::move(path));
}

std::shared_ptr<io_object> rest_ioctx::create_io_object(std::string path, std::uint64_t known_size)
{
  auto parsed = sirius::io::parse(path);
  if (parsed.scheme != "s3") {
    throw std::invalid_argument("rest_ioctx::create_io_object: unsupported scheme '" +
                                parsed.scheme + "'");
  }
  // The size came from a ListObjectsV2 response: build the io_object with zero
  // network — no HEAD, no probe.
  return std::make_shared<rest_io_object>(std::move(path),
                                          std::move(parsed.host),
                                          std::move(parsed.path),
                                          static_cast<size_t>(known_size));
}

std::shared_ptr<io_object> rest_ioctx::create_footer_probe_object(std::string path)
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
    auto head = _reactors.front()->head_object(parsed.host, parsed.path);
    return std::make_shared<rest_io_object>(std::move(path),
                                            std::move(parsed.host),
                                            std::move(parsed.path),
                                            head.object_size,
                                            std::move(head.etag));
  }
  return std::make_shared<rest_io_object>(std::move(path),
                                          std::move(parsed.host),
                                          std::move(parsed.path),
                                          probe.object_size,
                                          probe.window_lo,
                                          probe.bytes,
                                          std::move(probe.etag));
}

}  // namespace sirius::io::rest
