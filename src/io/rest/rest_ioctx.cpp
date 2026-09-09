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
#include <cctype>
#include <cstddef>
#include <format>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace sirius::io::rest {

rest_ioctx::rest_ioctx(std::size_t n_reactors, std::shared_ptr<rest_reactor::reactor_context> ctx)
  : templated_ioctx<rest_reactor>(n_reactors, [ctx = std::move(ctx), i = 0]() mutable {
      return std::make_unique<rest_reactor>(ctx, std::format("rest-{}", i++));
    })
{
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

namespace {

/// The bucket out of an @c s3:// URL, whether or not it names an object.
///
/// @c io::parse cannot serve this: it rejects a bucket-only URL with "empty
/// object key", which is exactly the shape a warm-up takes -- warming is about
/// the endpoint, and naming a data file to reach it would be beside the point.
std::optional<std::string> bucket_of(std::string_view url)
{
  constexpr std::string_view k_scheme = "s3://";
  if (url.size() <= k_scheme.size()) { return std::nullopt; }
  for (std::size_t i = 0; i < k_scheme.size(); ++i) {
    if (std::tolower(static_cast<unsigned char>(url[i])) != k_scheme[i]) { return std::nullopt; }
  }
  auto const rest   = url.substr(k_scheme.size());
  auto const bucket = rest.substr(0, rest.find('/'));
  if (bucket.empty()) { return std::nullopt; }
  return std::string{bucket};
}

}  // namespace

void rest_ioctx::warmup(std::string_view bucket_url) noexcept
{
  try {
    // Only the bucket is read: an object key, if the caller passed one, names
    // nothing the connection pool cares about.
    auto const bucket = bucket_of(bucket_url);
    if (!bucket.has_value()) { return; }

    // Connections are per-endpoint, so the bucket -- not the object -- is the
    // identity that decides whether a warm-up is redundant. A thousand-file
    // scan calling this per file collapses to one round.
    {
      std::lock_guard lk{_warm_mtx};
      auto const now = std::chrono::steady_clock::now();
      // conn_max_age of 0 means "no cap" to libcurl, which leaves us no honest
      // staleness horizon; fall back to a minute rather than re-warm forever.
      auto const stale_after =
        _config.conn_max_age.count() > 0
          ? std::chrono::duration_cast<std::chrono::steady_clock::duration>(_config.conn_max_age)
          : std::chrono::duration_cast<std::chrono::steady_clock::duration>(
              std::chrono::seconds{60});
      bool const same_bucket = _warmed_bucket == *bucket;
      if (same_bucket && _warmed_at.has_value() && now - *_warmed_at < stale_after) { return; }
      _warmed_bucket = *bucket;
      _warmed_at     = now;
    }

    for (auto& reactor : _reactors) {
      if (reactor) { reactor->warmup(*bucket); }
    }
  } catch (...) {  // NOLINT(bugprone-empty-catch)
    // Warming is an optimization; a query that would have run without it still
    // runs. Nothing here is worth failing a read over.
  }
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
