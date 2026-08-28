/*
 * Copyright 2025, Sirius Contributors.
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

#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

namespace sirius::io {

/// Inert configuration carrier for remote object-store backends.
/// PR1 only exposes the POD fields and enum/string helpers; runtime plumbing and
/// backend consumption live in the integration/backend PRs.
/// Empty strings are valid and mean "no value configured".
struct object_store_config {
  std::string endpoint;
  std::string region;
  std::string access_key;
  std::string secret_key;
  /// STS temporary-credential session token (empty for long-lived keys). When
  /// set, the SigV4 signer adds X-Amz-Security-Token to presigned URLs.
  std::string session_token;

  /// Requested S3 transport. AUTO leaves the concrete backend/integration code
  /// to choose based on URI scheme and endpoint capabilities.
  enum class transport { AUTO, HTTP, RDMA };
  transport s3_transport = transport::AUTO;

  /// RDMA reactor sizing (transport::RDMA only). @c s3_rdma_max_inflight is the
  /// worker count = the global in-flight ceiling (one blocking GET per worker);
  /// the per-device landing arena is @c max_inflight × @c slot_size bytes.
  /// Defaults are sized for a 100G link; scale bytes in flight with line rate.
  size_t s3_rdma_max_inflight    = 8;
  size_t s3_rdma_arena_slot_size = 4UL << 20;

  /// Envelope-queue bound for the RDMA intake (logical requests). Unset
  /// derives 4 x max_inflight after sanitizing; an explicit zero is rejected.
  std::optional<size_t> s3_rdma_queue_cap{};

  /// SigV4 signing form for S3 requests. @c presigned puts auth in the URL query
  /// string (default; works everywhere AWS does). @c header puts auth in the
  /// @c Authorization header (sign_request) — for on-prem / S3-compatible stores
  /// whose gateways prefer header auth over long presigned query strings.
  enum class signing_mode { presigned, header };
  signing_mode s3_signing_mode = signing_mode::presigned;

  /// PEM CA bundle used to verify the S3 endpoint's TLS certificate
  /// (CURLOPT_CAINFO). Empty (default) uses libcurl's system CA bundle —
  /// correct for AWS. Point it at a private / self-signed CA for on-prem or
  /// S3-compatible gateways (and the local-HTTPS test).
  std::string ca_bundle_path;

  /// Verify the S3 endpoint's TLS certificate (peer + host). Default true;
  /// false disables verification — INSECURE, dev/test only.
  bool tls_verify = true;

  /// The RDMA DATA-plane endpoint (transport::RDMA only): where the token
  /// GETs go, distinct from the control endpoint above. Empty credentials
  /// inherit the control endpoint's at use. Presigned signing is rejected on
  /// this plane (the data plane signs extra headers into the request).
  struct s3_data_endpoint {
    std::string endpoint;  ///< empty = data plane not configured
    std::string region;
    std::string access_key;  ///< empty = inherit the control credentials
    std::string secret_key;
    signing_mode s3_signing_mode = signing_mode::header;
    std::string ca_bundle_path;
    bool tls_verify = true;
  };
  s3_data_endpoint s3_rdma_data;
};

inline bool string_to_enum(std::string_view sv, object_store_config::transport& t)
{
  static const std::unordered_map<std::string_view, object_store_config::transport> map = {
    {"auto", object_store_config::transport::AUTO},
    {"http", object_store_config::transport::HTTP},
    {"https", object_store_config::transport::HTTP},
    {"rdma", object_store_config::transport::RDMA},
  };
  auto it = map.find(sv);
  if (it != map.end()) {
    t = it->second;
    return true;
  }
  return false;
}

inline bool enum_to_string(object_store_config::transport t, std::string& s)
{
  switch (t) {
    case object_store_config::transport::AUTO: s = "auto"; return true;
    case object_store_config::transport::HTTP: s = "http"; return true;
    case object_store_config::transport::RDMA: s = "rdma"; return true;
  }
  return false;
}

inline bool string_to_enum(std::string_view sv, object_store_config::signing_mode& m)
{
  static const std::unordered_map<std::string_view, object_store_config::signing_mode> map = {
    {"presigned", object_store_config::signing_mode::presigned},
    {"header", object_store_config::signing_mode::header},
  };
  auto it = map.find(sv);
  if (it != map.end()) {
    m = it->second;
    return true;
  }
  return false;
}

inline bool enum_to_string(object_store_config::signing_mode m, std::string& s)
{
  switch (m) {
    case object_store_config::signing_mode::presigned: s = "presigned"; return true;
    case object_store_config::signing_mode::header: s = "header"; return true;
  }
  return false;
}

}  // namespace sirius::io
