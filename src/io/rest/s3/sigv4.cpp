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

#include "io/rest/s3/sigv4.hpp"

#include <openssl/evp.h>
#include <openssl/hmac.h>
#include <openssl/sha.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <stdexcept>

namespace sirius::io::rest::s3 {

namespace {

constexpr char kHexLower[] = "0123456789abcdef";

std::string hex_encode(unsigned char const* data, std::size_t len)
{
  std::string out;
  out.resize(len * 2);
  for (std::size_t i = 0; i < len; ++i) {
    out[2 * i]     = kHexLower[(data[i] >> 4) & 0xF];
    out[2 * i + 1] = kHexLower[data[i] & 0xF];
  }
  return out;
}

// Raw HMAC-SHA256. Returns 32 bytes.
std::array<unsigned char, 32> hmac_sha256(std::string_view key, std::string_view msg)
{
  std::array<unsigned char, 32> out{};
  unsigned int len = 0;
  // HMAC returns nullptr on failure; for HMAC-SHA256 on valid inputs that
  // should never happen, but guard anyway so we fail loudly rather than
  // produce a silently-zero signature.
  if (HMAC(EVP_sha256(),
           key.data(),
           static_cast<int>(key.size()),
           reinterpret_cast<unsigned char const*>(msg.data()),
           msg.size(),
           out.data(),
           &len) == nullptr ||
      len != out.size()) {
    throw std::runtime_error("sigv4: HMAC-SHA256 failed");
  }
  return out;
}

// Trim leading/trailing ASCII spaces. SigV4 canonicalization does not
// collapse interior runs for our header set (host/date/content-sha256/range),
// so this simple trim is sufficient.
std::string trim_header_value(std::string_view v)
{
  std::size_t b = 0;
  std::size_t e = v.size();
  while (b < e && (v[b] == ' ' || v[b] == '\t'))
    ++b;
  while (e > b && (v[e - 1] == ' ' || v[e - 1] == '\t'))
    --e;
  return std::string(v.substr(b, e - b));
}

std::string to_lower(std::string_view s)
{
  std::string out(s);
  for (auto& c : out)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// Public helpers
// ---------------------------------------------------------------------------

std::string sha256_hex(std::string_view data)
{
  std::array<unsigned char, SHA256_DIGEST_LENGTH> md{};
  SHA256(reinterpret_cast<unsigned char const*>(data.data()), data.size(), md.data());
  return hex_encode(md.data(), md.size());
}

std::string uri_encode(std::string_view s, bool encode_slash)
{
  static constexpr char kHexUpper[] = "0123456789ABCDEF";
  std::string out;
  out.reserve(s.size());
  for (unsigned char c : s) {
    bool unreserved = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ||
                      c == '-' || c == '_' || c == '.' || c == '~';
    if (unreserved || (c == '/' && !encode_slash)) {
      out.push_back(static_cast<char>(c));
    } else {
      out.push_back('%');
      out.push_back(kHexUpper[(c >> 4) & 0xF]);
      out.push_back(kHexUpper[c & 0xF]);
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// sign_request
// ---------------------------------------------------------------------------

sigv4_signed_request sign_request(
  std::string_view method,
  std::string_view host,
  std::string_view canonical_uri,
  std::string_view canonical_query,
  std::string_view payload_sha256_hex,
  std::vector<std::pair<std::string, std::string>> const& extra_headers,
  sigv4_signer_config const& creds,
  std::time_t timestamp_utc)
{
  if (creds.access_key.empty() || creds.secret_key.empty() || creds.region.empty() ||
      creds.service.empty()) {
    throw std::invalid_argument("sigv4: empty credentials/region/service");
  }

  // ---- 1. Format timestamp. ----
  std::tm tm_utc{};
  ::gmtime_r(&timestamp_utc, &tm_utc);

  char amz_date[17];   // YYYYMMDDTHHMMSSZ + NUL
  char date_stamp[9];  // YYYYMMDD + NUL
  std::strftime(amz_date, sizeof(amz_date), "%Y%m%dT%H%M%SZ", &tm_utc);
  std::strftime(date_stamp, sizeof(date_stamp), "%Y%m%d", &tm_utc);

  // ---- 2. Assemble the full header set that will be signed. ----
  // host, x-amz-date, x-amz-content-sha256 are always signed; x-amz-security-token
  // is auto-injected when creds carry a session token (temporary creds: STS /
  // IMDS / SSO). Caller-supplied extras are merged in. Keys are lowercased and
  // sorted.
  std::vector<std::pair<std::string, std::string>> headers;
  headers.reserve(3 + (creds.session_token.empty() ? 0 : 1) + extra_headers.size());
  headers.emplace_back("host", std::string(host));
  headers.emplace_back("x-amz-content-sha256", std::string(payload_sha256_hex));
  headers.emplace_back("x-amz-date", amz_date);
  if (!creds.session_token.empty()) {
    headers.emplace_back("x-amz-security-token", creds.session_token);
  }
  for (auto const& [k, v] : extra_headers) {
    headers.emplace_back(to_lower(k), trim_header_value(v));
  }
  std::sort(
    headers.begin(), headers.end(), [](auto const& a, auto const& b) { return a.first < b.first; });

  // ---- 3. Canonical headers + signed header list. ----
  std::string canonical_headers;
  std::string signed_headers;
  for (std::size_t i = 0; i < headers.size(); ++i) {
    canonical_headers += headers[i].first;
    canonical_headers += ':';
    canonical_headers += headers[i].second;
    canonical_headers += '\n';
    if (i != 0) signed_headers += ';';
    signed_headers += headers[i].first;
  }

  // ---- 4. Canonical request. ----
  std::string canonical_request;
  canonical_request += method;
  canonical_request += '\n';
  canonical_request += canonical_uri;
  canonical_request += '\n';
  canonical_request += canonical_query;
  canonical_request += '\n';
  canonical_request += canonical_headers;
  canonical_request += '\n';
  canonical_request += signed_headers;
  canonical_request += '\n';
  canonical_request += payload_sha256_hex;

  // ---- 5. String to sign. ----
  std::string credential_scope;
  credential_scope += date_stamp;
  credential_scope += '/';
  credential_scope += creds.region;
  credential_scope += '/';
  credential_scope += creds.service;
  credential_scope += "/aws4_request";

  std::string string_to_sign;
  string_to_sign += "AWS4-HMAC-SHA256\n";
  string_to_sign += amz_date;
  string_to_sign += '\n';
  string_to_sign += credential_scope;
  string_to_sign += '\n';
  string_to_sign += sha256_hex(canonical_request);

  // ---- 6. Signing key chain. ----
  std::string k_secret = "AWS4" + creds.secret_key;
  auto k_date          = hmac_sha256(k_secret, date_stamp);
  auto k_region        = hmac_sha256(
    std::string_view(reinterpret_cast<char const*>(k_date.data()), k_date.size()), creds.region);
  auto k_service =
    hmac_sha256(std::string_view(reinterpret_cast<char const*>(k_region.data()), k_region.size()),
                creds.service);
  auto k_signing =
    hmac_sha256(std::string_view(reinterpret_cast<char const*>(k_service.data()), k_service.size()),
                "aws4_request");
  auto sig =
    hmac_sha256(std::string_view(reinterpret_cast<char const*>(k_signing.data()), k_signing.size()),
                string_to_sign);

  std::string signature_hex = hex_encode(sig.data(), sig.size());

  // ---- 7. Authorization header. ----
  std::string authorization;
  authorization += "AWS4-HMAC-SHA256 Credential=";
  authorization += creds.access_key;
  authorization += '/';
  authorization += credential_scope;
  authorization += ", SignedHeaders=";
  authorization += signed_headers;
  authorization += ", Signature=";
  authorization += signature_hex;

  // ---- 8. Assemble response headers. Order mirrors what a caller would
  //         typically emit: Authorization last so it's easy to spot in logs.
  sigv4_signed_request out;
  out.headers.reserve(3 + (creds.session_token.empty() ? 0 : 1) + extra_headers.size());
  out.headers.emplace_back("Host", std::string(host));
  out.headers.emplace_back("x-amz-date", amz_date);
  out.headers.emplace_back("x-amz-content-sha256", std::string(payload_sha256_hex));
  if (!creds.session_token.empty()) {
    out.headers.emplace_back("x-amz-security-token", creds.session_token);
  }
  for (auto const& [k, v] : extra_headers) {
    out.headers.emplace_back(k, v);
  }
  out.headers.emplace_back("Authorization", std::move(authorization));
  return out;
}

// ---------------------------------------------------------------------------
// presign_url
// ---------------------------------------------------------------------------

std::string presign_url(std::string_view method,
                        std::string_view scheme,
                        std::string_view host,
                        std::string_view canonical_uri,
                        sigv4_signer_config const& creds,
                        std::time_t timestamp_utc,
                        std::chrono::seconds ttl,
                        std::string_view extra_canonical_query)
{
  if (creds.access_key.empty() || creds.secret_key.empty() || creds.region.empty() ||
      creds.service.empty()) {
    throw std::invalid_argument("sigv4: empty credentials/region/service");
  }
  if (method.empty() || scheme.empty() || host.empty()) {
    throw std::invalid_argument("sigv4: empty method/scheme/host");
  }
  if (ttl.count() <= 0) { throw std::invalid_argument("sigv4: ttl must be positive"); }

  // ---- 1. Format timestamp. ----
  std::tm tm_utc{};
  ::gmtime_r(&timestamp_utc, &tm_utc);

  char amz_date[17];   // YYYYMMDDTHHMMSSZ + NUL
  char date_stamp[9];  // YYYYMMDD + NUL
  std::strftime(amz_date, sizeof(amz_date), "%Y%m%dT%H%M%SZ", &tm_utc);
  std::strftime(date_stamp, sizeof(date_stamp), "%Y%m%d", &tm_utc);

  // ---- 2. Credential scope and credential query-param value. ----
  std::string credential_scope;
  credential_scope += date_stamp;
  credential_scope += '/';
  credential_scope += creds.region;
  credential_scope += '/';
  credential_scope += creds.service;
  credential_scope += "/aws4_request";

  std::string credential_qparam = creds.access_key;
  credential_qparam += '/';
  credential_qparam += credential_scope;

  // ---- 3. Canonical query string.
  //         All X-Amz-* parameters that participate in the signature, sorted
  //         by key (alphabetical), each value RFC3986-encoded with
  //         encode_slash=true (per AWS query-encoding rules). The
  //         X-Amz-Signature itself is appended *after* signing — it is not
  //         part of the canonical request. ----
  //         Stored already-encoded so the X-Amz-* parameters and any
  //         caller-supplied request parameters (@p extra_canonical_query, which
  //         arrives pre-encoded) can be merged and sorted together by encoded
  //         key — AWS signs the combined, sorted set.
  std::vector<std::pair<std::string, std::string>> qparams;
  auto add_amz = [&qparams](std::string_view k, std::string_view v) {
    qparams.emplace_back(uri_encode(k, /*encode_slash=*/true),
                         uri_encode(v, /*encode_slash=*/true));
  };
  add_amz("X-Amz-Algorithm", "AWS4-HMAC-SHA256");
  add_amz("X-Amz-Credential", credential_qparam);
  add_amz("X-Amz-Date", amz_date);
  add_amz("X-Amz-Expires", std::to_string(ttl.count()));
  if (!creds.session_token.empty()) { add_amz("X-Amz-Security-Token", creds.session_token); }
  add_amz("X-Amz-SignedHeaders", "host");

  // Merge the caller's pre-encoded request params (e.g. ListObjectsV2's
  // list-type / prefix / max-keys / continuation-token), taken verbatim.
  for (std::size_t b = 0; b < extra_canonical_query.size();) {
    auto const amp  = extra_canonical_query.find('&', b);
    auto const pair = extra_canonical_query.substr(
      b, amp == std::string_view::npos ? std::string_view::npos : amp - b);
    if (!pair.empty()) {
      auto const eq = pair.find('=');
      if (eq == std::string_view::npos) {
        qparams.emplace_back(std::string{pair}, std::string{});
      } else {
        qparams.emplace_back(std::string{pair.substr(0, eq)}, std::string{pair.substr(eq + 1)});
      }
    }
    if (amp == std::string_view::npos) { break; }
    b = amp + 1;
  }

  std::sort(
    qparams.begin(), qparams.end(), [](auto const& a, auto const& b) { return a.first < b.first; });

  std::string canonical_query;
  for (std::size_t i = 0; i < qparams.size(); ++i) {
    if (i != 0) canonical_query += '&';
    canonical_query += qparams[i].first;
    canonical_query += '=';
    canonical_query += qparams[i].second;
  }

  // ---- 4. Canonical headers (host only) and signed headers list. ----
  std::string canonical_headers = "host:";
  canonical_headers += host;
  canonical_headers += '\n';
  std::string const signed_headers = "host";

  // ---- 5. Canonical request. Payload hash field is the literal string
  //         "UNSIGNED-PAYLOAD" — this is the key difference from the
  //         header-based sign_request path. ----
  std::string canonical_request;
  canonical_request += method;
  canonical_request += '\n';
  canonical_request += canonical_uri;
  canonical_request += '\n';
  canonical_request += canonical_query;
  canonical_request += '\n';
  canonical_request += canonical_headers;
  canonical_request += '\n';
  canonical_request += signed_headers;
  canonical_request += '\n';
  canonical_request += "UNSIGNED-PAYLOAD";

  // ---- 6. String to sign. ----
  std::string string_to_sign;
  string_to_sign += "AWS4-HMAC-SHA256\n";
  string_to_sign += amz_date;
  string_to_sign += '\n';
  string_to_sign += credential_scope;
  string_to_sign += '\n';
  string_to_sign += sha256_hex(canonical_request);

  // ---- 7. Signing key chain (identical to sign_request). ----
  std::string k_secret = "AWS4" + creds.secret_key;
  auto k_date          = hmac_sha256(k_secret, date_stamp);
  auto k_region        = hmac_sha256(
    std::string_view(reinterpret_cast<char const*>(k_date.data()), k_date.size()), creds.region);
  auto k_service =
    hmac_sha256(std::string_view(reinterpret_cast<char const*>(k_region.data()), k_region.size()),
                creds.service);
  auto k_signing =
    hmac_sha256(std::string_view(reinterpret_cast<char const*>(k_service.data()), k_service.size()),
                "aws4_request");
  auto sig =
    hmac_sha256(std::string_view(reinterpret_cast<char const*>(k_signing.data()), k_signing.size()),
                string_to_sign);

  std::string signature_hex = hex_encode(sig.data(), sig.size());

  // ---- 8. Assemble the final URL.  S3 accepts the query params in any order
  //         (only the canonical query fed into the signature must be sorted,
  //         and X-Amz-Signature is never part of it), so signature placement is
  //         presentation-only.  Preserve the existing object-request layout —
  //         a pure X-Amz query appends the signature last; merged LIST queries
  //         insert it in sorted position for a fully canonical-ordered URL. ----
  std::string final_query;
  if (extra_canonical_query.empty()) {
    final_query = canonical_query;
    final_query += "&X-Amz-Signature=";
    final_query += signature_hex;
  } else {
    qparams.emplace_back("X-Amz-Signature", std::move(signature_hex));
    std::sort(qparams.begin(), qparams.end(), [](auto const& a, auto const& b) {
      return a.first < b.first;
    });
    for (std::size_t i = 0; i < qparams.size(); ++i) {
      if (i != 0) final_query += '&';
      final_query += qparams[i].first;
      final_query += '=';
      final_query += qparams[i].second;
    }
  }

  std::string out;
  out.reserve(scheme.size() + 3 + host.size() + canonical_uri.size() + 1 + final_query.size());
  out.append(scheme.data(), scheme.size());
  out += "://";
  out.append(host.data(), host.size());
  out.append(canonical_uri.data(), canonical_uri.size());
  out += '?';
  out += final_query;
  return out;
}

}  // namespace sirius::io::rest::s3
