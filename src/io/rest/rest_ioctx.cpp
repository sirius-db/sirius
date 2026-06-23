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

#include <fmt/format.h>

#include <stdexcept>
#include <utility>

namespace sirius::io::rest {

namespace {

/// Assemble the primitive per-reactor config from the ioctx constructor
/// arguments, applying only the retry overrides the caller actually provided
/// (so the reactor's own defaults stand otherwise).  Shared collaborators (the
/// authorizer, the staging resource) live on the reactor_context, not here; the
/// staging resource's block size is cached as a primitive so the static
/// prep_device_rx_request can size bounce windows without the live resource.
rest_reactor::config build_config(long request_timeout_s,
                                  std::string ca_bundle_path,
                                  bool tls_verify,
                                  std::size_t max_connections,
                                  cucascade::memory::fixed_size_host_memory_resource* host_mr,
                                  std::optional<std::size_t> max_retry_attempts,
                                  std::optional<std::chrono::milliseconds> retry_backoff_base,
                                  std::optional<std::chrono::milliseconds> retry_jitter,
                                  std::optional<bool> honor_retry_after)
{
  rest_reactor::config cfg;
  cfg.request_timeout_s = request_timeout_s;
  cfg.ca_bundle_path    = std::move(ca_bundle_path);
  cfg.tls_verify        = tls_verify;
  cfg.max_connections   = max_connections;
  cfg.bounce_block_size = host_mr != nullptr ? host_mr->get_block_size() : 0;
  if (max_retry_attempts) { cfg.max_retry_attempts = *max_retry_attempts; }
  if (retry_backoff_base) { cfg.retry_backoff_base = *retry_backoff_base; }
  if (retry_jitter) { cfg.retry_jitter = *retry_jitter; }
  if (honor_retry_after) { cfg.honor_retry_after = *honor_retry_after; }
  return cfg;
}

}  // namespace

rest_ioctx::rest_ioctx(std::shared_ptr<s3::s3_request_authorizer> authorizer,
                       std::size_t n_reactors,
                       long request_timeout_s,
                       std::string ca_bundle_path,
                       bool tls_verify,
                       std::size_t max_connections,
                       cucascade::memory::fixed_size_host_memory_resource* host_mr,
                       std::optional<std::size_t> max_retry_attempts,
                       std::optional<std::chrono::milliseconds> retry_backoff_base,
                       std::optional<std::chrono::milliseconds> retry_jitter,
                       std::optional<bool> honor_retry_after)
  : rest_ioctx(
      std::make_shared<rest_reactor::reactor_context>(build_config(request_timeout_s,
                                                                   std::move(ca_bundle_path),
                                                                   tls_verify,
                                                                   max_connections,
                                                                   host_mr,
                                                                   max_retry_attempts,
                                                                   retry_backoff_base,
                                                                   retry_jitter,
                                                                   honor_retry_after),
                                                      std::move(authorizer),
                                                      host_mr),
      n_reactors)
{
}

rest_ioctx::rest_ioctx(const std::shared_ptr<rest_reactor::reactor_context>& ctx,
                       std::size_t n_reactors)
  : templated_ioctx<rest_reactor>(
      n_reactors, ctx->cfg(), [ctx, i = 0](const rest_reactor::reactor_config_type&) mutable {
        return std::make_unique<rest_reactor>(ctx, fmt::format("rest-{}", i++));
      })
{
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

}  // namespace sirius::io::rest
