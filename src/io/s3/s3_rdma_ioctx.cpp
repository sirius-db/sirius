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
#include "log/logging.hpp"

#include <algorithm>
#include <cctype>
#include <cstddef>
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

/// Normalized endpoint components for the same-address judgment: scheme and
/// host lowercased, the default port expanded from the scheme, ONE trailing
/// slash stripped from the path (paths stay case-sensitive).  Purely
/// syntactic — no DNS resolution.
struct normalized_endpoint {
  std::string scheme;
  std::string host;
  std::string port;
  std::string path;
};

std::string lowered(std::string_view s)
{
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  }
  return out;
}

normalized_endpoint normalize_endpoint(std::string_view ep)
{
  normalized_endpoint out;
  const auto scheme_end = ep.find("://");
  std::string_view rest = ep;
  if (scheme_end != std::string_view::npos) {
    out.scheme = lowered(ep.substr(0, scheme_end));
    rest       = ep.substr(scheme_end + 3);
  }
  const auto path_begin      = rest.find('/');
  std::string_view authority = rest.substr(0, path_begin);
  if (path_begin != std::string_view::npos) {
    std::string_view path = rest.substr(path_begin);
    if (!path.empty() && path.back() == '/') { path.remove_suffix(1); }
    out.path.assign(path);
  }
  // Bracketed IPv6 keeps the whole authority as the host (no port split
  // heuristics); otherwise split on the last ':' when the suffix is digits.
  if (!authority.empty() && authority.front() == '[') {
    out.host = lowered(authority);
  } else {
    const auto colon    = authority.rfind(':');
    const bool has_port = colon != std::string_view::npos && colon + 1 < authority.size() &&
                          std::all_of(authority.begin() + static_cast<std::ptrdiff_t>(colon) + 1,
                                      authority.end(),
                                      [](unsigned char c) { return std::isdigit(c) != 0; });
    if (has_port) {
      out.host = lowered(authority.substr(0, colon));
      out.port.assign(authority.substr(colon + 1));
    } else {
      out.host = lowered(authority);
    }
  }
  if (out.port.empty()) {
    if (out.scheme == "http") { out.port = "80"; }
    if (out.scheme == "https") { out.port = "443"; }
  }
  return out;
}

}  // namespace

endpoint_topology detect_endpoint_topology(std::string_view host_endpoint,
                                           std::string_view data_endpoint)
{
  const auto host_side = normalize_endpoint(host_endpoint);
  const auto data_side = normalize_endpoint(data_endpoint);
  // Only a provably equal pair is same_address; a missing host on either
  // side (including two empty endpoints) can never prove anything.
  if (host_side.host.empty() || data_side.host.empty()) { return endpoint_topology::split; }
  const bool equal = host_side.scheme == data_side.scheme && host_side.host == data_side.host &&
                     host_side.port == data_side.port && host_side.path == data_side.path;
  return equal ? endpoint_topology::same_address : endpoint_topology::split;
}

void s3_rdma_ioctx::list_objects_paged(
  std::string_view bucket,
  std::string_view prefix,
  std::size_t page_size,
  std::function<bool(s3::list_objects_v2_page const&)> const& sink,
  std::optional<std::size_t> max_scanned)
{
  auto const& control = _reactor_ctx->clients().control;
  if (!control) {
    throw std::runtime_error("s3_rdma_ioctx::list_objects_paged: no control-plane client");
  }
  const std::size_t clamped     = (page_size == 0 || page_size > 1000) ? 1000 : page_size;
  const std::size_t scanned_cap = max_scanned.value_or(s3::default_max_scanned_objects);
  const std::string where       = "s3://" + std::string(bucket) + "/" + std::string(prefix);

  std::size_t scanned = 0;
  std::string token;
  bool truncated = false;
  do {
    // One control permit per page, scoped to the client call ONLY (the gate
    // contract: a permit covers one control call).  Validation and the sink
    // run after release — a sink that closes the ioctx must not deadlock
    // against its own page's permit.  A terminal gate refuses the next page
    // with the single terminal error (first_fatal when latched, else the
    // stable closed error).
    rdma::list_page_result page;
    {
      auto permit = _reactor_ctx->gate().acquire_control();
      page        = control->list_page(bucket, prefix, clamped, token);
    }
    if (!page.outcome.transport_ok()) {
      throw std::runtime_error(
        "s3_rdma_ioctx::list_objects_paged: " + page.outcome.transport_error + " for " + where);
    }
    if (page.outcome.http_status != 200) {
      throw std::runtime_error("s3_rdma_ioctx::list_objects_paged: HTTP " +
                               std::to_string(page.outcome.http_status) + " for " + where);
    }
    scanned += page.page.entries.size();
    if (scanned > scanned_cap) {
      throw std::runtime_error("s3_rdma_ioctx::list_objects_paged: scanned more than " +
                               std::to_string(scanned_cap) + " objects under " + where +
                               " — narrow the glob prefix");
    }
    if (page.page.is_truncated && page.page.next_continuation_token.empty()) {
      throw std::runtime_error(
        "s3_rdma_ioctx::list_objects_paged: truncated ListObjectsV2 page without a continuation "
        "token for " +
        where);
    }
    if (page.page.is_truncated && page.page.entries.empty()) {
      throw std::runtime_error(
        "s3_rdma_ioctx::list_objects_paged: truncated ListObjectsV2 page with no entries for " +
        where);
    }
    if (page.page.is_truncated && page.page.next_continuation_token == token) {
      throw std::runtime_error(
        "s3_rdma_ioctx::list_objects_paged: ListObjectsV2 continuation token did not advance "
        "for " +
        where);
    }
    truncated = page.page.is_truncated;
    token     = page.page.next_continuation_token;
    if (!sink(page.page)) { return; }
  } while (truncated);
}

std::size_t s3_rdma_ioctx::list_max_matches() const { return s3::default_max_list_objects; }

s3_rdma_ioctx::s3_rdma_ioctx(object_store_config cfg,
                             rdma::rdma_transport_clients clients,
                             rdma::cuda_delivery_ops delivery)
  : s3_rdma_ioctx(std::make_shared<rdma::cuobj_rdma_reactor::reactor_context>(
      reactor_config_from(cfg), std::move(clients), std::move(delivery)))
{
  _topology = detect_endpoint_topology(cfg.endpoint, cfg.s3_rdma_data.endpoint);
  // An explicit data-plane key that differs from the host plane is judged
  // here (config is not retained); the start() log decides whether it is
  // worth a warning — only under same_address, where two credential sets
  // against one service is usually a misconfiguration.
  _credentials_differ =
    !cfg.s3_rdma_data.access_key.empty() && cfg.s3_rdma_data.access_key != cfg.access_key;
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
  if (_topology == endpoint_topology::same_address) {
    SIRIUS_LOG_INFO(
      "s3_rdma_ioctx: endpoint topology same_address — one service serves both planes; "
      "cross-endpoint checks reduce to its native consistency and immutable keys are the "
      "operative rule");
    if (_credentials_differ) {
      SIRIUS_LOG_WARN(
        "s3_rdma_ioctx: same-address deployment but the data-plane credentials differ from the "
        "host plane; two credential sets against one service is usually a misconfiguration");
    }
  } else {
    SIRIUS_LOG_INFO(
      "s3_rdma_ioctx: endpoint topology split — the full publisher visibility barrier contract "
      "applies");
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
