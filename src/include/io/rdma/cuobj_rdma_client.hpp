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

#include "io/rdma/rdma_client.hpp"
#include "io/s3/s3_request_authorizer.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace sirius::io::rdma {

/**
 * @brief The production @c rdma_client: SigV4 HTTP control plane + cuObject
 *        data plane.
 *
 * The HTTP half (@c head and host-destination @c get: ranged, SigV4-signed
 * via the shared authorizer, libcurl transport) always compiles and needs no
 * RDMA hardware or SDK; it serves the bind-time footer / metadata path.
 *
 * The device half (@c get into a registered landing-arena slot via cuObjGet,
 * plus @c register_memory / @c deregister_memory) requires building with
 * @c SIRIUS_ENABLE_S3_RDMA (cuObject SDK).  Without it, device-destination
 * gets fail loudly with an error naming the flag; registration is a no-op.
 * Destination kind is detected per call (host vs device pointer).
 */
class cuobj_rdma_client final : public rdma_client {
 public:
  /// @p authorizer carries endpoint / region / credentials (the same SigV4
  /// authorizer the REST backend uses; presigned or header mode).  TLS options
  /// mirror the REST reactor's.
  explicit cuobj_rdma_client(std::shared_ptr<s3::s3_request_authorizer> authorizer,
                             std::string ca_bundle_path = "",
                             bool tls_verify            = true);
  ~cuobj_rdma_client() override;

  size_t head(std::string_view bucket, std::string_view key) override;
  size_t get(
    std::string_view bucket, std::string_view key, size_t offset, size_t size, void* dst) override;

  void register_memory(void* base, size_t bytes) override;
  void deregister_memory(void* base) noexcept override;

  /// Control-plane GET: SigV4-signed, body discarded, @p extra_headers attached
  /// verbatim (the cuObject data-plane callback rides on this with the RDMA
  /// descriptor token; the reply carries status only).  Throws on non-2xx.
  void control_get(std::string_view bucket,
                   std::string_view key,
                   const std::vector<std::pair<std::string, std::string>>& extra_headers);

 private:
  size_t host_get(
    std::string_view bucket, std::string_view key, size_t offset, size_t size, void* dst);
  size_t device_get(
    std::string_view bucket, std::string_view key, size_t offset, size_t size, void* dst);
  void* ensure_cuobj_client();

  std::shared_ptr<s3::s3_request_authorizer> _authorizer;
  std::string _ca_bundle_path;
  bool _tls_verify;
  void* _cuobj{nullptr};  // lazily-created cuObjClient (SDK builds only)
};

}  // namespace sirius::io::rdma
