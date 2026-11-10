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

#include "io/rest/s3/endpoint.hpp"

#include <cucascade/io/io_errors.hpp>

#include <cctype>
#include <utility>

namespace sirius::io::rest::s3 {

using cucascade::io::credential_error;

namespace {

/// Lowercase ASCII. RFC 3986 says scheme is case-insensitive and DNS host is
/// case-insensitive; the SigV4 canonical form requires the lowercased version.
std::string to_lower(std::string_view s)
{
  std::string out(s);
  for (auto& c : out)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return out;
}

}  // namespace

/// Split @p endpoint into (scheme, host_with_optional_port). Trailing path /
/// query / fragment is rejected — the endpoint identifies the S3 service, not
/// a specific bucket or object. Declared in io/rest/s3/endpoint.hpp.
std::pair<std::string, std::string> parse_endpoint(std::string_view endpoint)
{
  if (endpoint.empty()) { throw credential_error("sigv4_presigned_authorizer: empty endpoint"); }
  auto sep = endpoint.find("://");
  if (sep == std::string_view::npos) {
    throw credential_error(
      "sigv4_presigned_authorizer: endpoint missing scheme (expected http:// or https://)");
  }
  std::string scheme = to_lower(endpoint.substr(0, sep));
  if (scheme != "http" && scheme != "https") {
    throw credential_error(
      "sigv4_presigned_authorizer: endpoint scheme must be http or https (got '" + scheme + "')");
  }
  std::string remainder{endpoint.substr(sep + 3)};
  // Anything past host[:port] (path, query, fragment) is rejected — endpoint
  // is service-level, not URL.
  auto bad = remainder.find_first_of("/?#");
  if (bad != std::string::npos) {
    throw credential_error(
      "sigv4_presigned_authorizer: endpoint must be scheme://host[:port] only (got "
      "trailing '" +
      remainder.substr(bad) + "')");
  }
  // "scheme://:port" is a port-only authority: no host. User-info is not part of an endpoint.
  if (remainder.empty() || remainder.front() == ':') {
    throw credential_error("sigv4_presigned_authorizer: endpoint missing host");
  }
  if (remainder.find('@') != std::string::npos) {
    throw credential_error("sigv4_presigned_authorizer: endpoint must not contain user-info");
  }
  return {std::move(scheme), to_lower(remainder)};
}

}  // namespace sirius::io::rest::s3
