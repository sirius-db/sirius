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

#include "io/rest/rest_reactor.hpp"
#include "io/s3/s3_request_authorizer.hpp"
#include "io/templated_ioctx.hpp"

#include <cucascade/memory/fixed_size_host_memory_resource.hpp>

#include <chrono>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>

namespace sirius::io::rest {

// ---------------------------------------------------------------------------
// rest_ioctx
// ---------------------------------------------------------------------------

/**
 * @brief RESTful object-store (s3://) ioctx. Specialisation of
 *        @c templated_ioctx<rest_reactor>.
 *
 * Owns a pool of @c rest_reactor workers (round-robined by the base) that share
 * one @p authorizer.  Overrides @c create_io_object to resolve an object's size
 * via a blocking HEAD before constructing the @c rest_io_object — the static
 * reactor factory cannot do this since it needs the authorizer + a round-trip.
 */
class rest_ioctx : public templated_ioctx<rest_reactor> {
 public:
  /**
   * @param authorizer          Presigned-URL / SigV4 source, shared by all
   *                            reactors (must be non-null).
   * @param n_reactors          Number of worker reactors (each its own event
   *                            loop + connection pool).
   * @param request_timeout_s   Per-request whole-transfer timeout (seconds).
   * @param ca_bundle_path      Optional TLS CA bundle.
   * @param tls_verify          Verify peer/host certificates.
   * @param max_connections     Concurrent in-flight connections per reactor.
   * @param host_mr             Pinned host resource for device-read staging
   *                            (null disables reactor-staged device reads).
   * @param max_retry_attempts  Retry-policy overrides; unset keeps the
   * @param retry_backoff_base  reactor's own defaults.
   * @param retry_jitter
   * @param honor_retry_after
   */
  explicit rest_ioctx(std::shared_ptr<s3::s3_request_authorizer> authorizer,
                      std::size_t n_reactors                                      = 2,
                      long request_timeout_s                                      = 30,
                      std::string ca_bundle_path                                  = "",
                      bool tls_verify                                             = true,
                      std::size_t max_connections                                 = 16,
                      cucascade::memory::fixed_size_host_memory_resource* host_mr = nullptr,
                      std::optional<std::size_t> max_retry_attempts               = std::nullopt,
                      std::optional<std::chrono::milliseconds> retry_backoff_base = std::nullopt,
                      std::optional<std::chrono::milliseconds> retry_jitter       = std::nullopt,
                      std::optional<bool> honor_retry_after                       = std::nullopt);

  [[nodiscard]] io_context_type type() const noexcept override { return io_context_type::restful; }

 protected:
  /// Backend hook invoked by @c sirius_ioctx::open_datasource: parse @p path
  /// (s3://bucket/key), HEAD it for the size, and build a @c rest_io_object.
  /// Throws on a non-s3 scheme or a failed HEAD.
  std::shared_ptr<sirius_io_object> create_io_object(std::string path) override;

 private:
  /// Delegated-to target: build the reactor pool from a shared context (one
  /// context shared across all reactors).  The public constructor assembles the
  /// context from its arguments and forwards here.
  rest_ioctx(const std::shared_ptr<rest_reactor::reactor_context>& ctx, std::size_t n_reactors);
};

}  // namespace sirius::io::rest
