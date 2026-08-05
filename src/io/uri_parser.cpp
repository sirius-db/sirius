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

#include "io/uri_parser.hpp"

#include <cctype>
#include <stdexcept>
#include <string>
#include <string_view>

namespace sirius::io {

namespace {

constexpr std::string_view kSchemeDelim = "://";
constexpr std::string_view kFileScheme  = "file";
constexpr std::string_view kS3Scheme    = "s3";

[[noreturn]] void fail(std::string_view reason, std::string_view uri)
{
  std::string msg = "uri_parser: ";
  msg.append(reason);
  msg.append(" (uri=");
  msg.append(uri);
  msg.append(")");
  throw std::invalid_argument(msg);
}

int hex_value(char c)
{
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
  if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
  return -1;
}

// Percent-decode @p in into @p out. Throws on malformed sequences (`%ZZ`, `%A`,
// `%` at end of string).
std::string percent_decode(std::string_view in, std::string_view uri)
{
  std::string out;
  out.reserve(in.size());
  for (std::size_t i = 0; i < in.size(); ++i) {
    char c = in[i];
    if (c != '%') {
      out.push_back(c);
      continue;
    }
    if (i + 2 >= in.size()) fail("truncated percent-encoding", uri);
    int hi = hex_value(in[i + 1]);
    int lo = hex_value(in[i + 2]);
    if (hi < 0 || lo < 0) fail("malformed percent-encoding", uri);
    out.push_back(static_cast<char>((hi << 4) | lo));
    i += 2;
  }
  return out;
}

// Parse a query string (the part after `?`, before `#`) into key/value pairs.
// Rules: last-wins on duplicate keys; `?flag` -> `{flag:""}`; `?k=` -> `{k:""}`;
// `?=val` throws; empty segments (`?&&`) ignored.
std::unordered_map<std::string, std::string> parse_query(std::string_view query,
                                                         std::string_view uri)
{
  std::unordered_map<std::string, std::string> out;
  std::size_t i = 0;
  while (i < query.size()) {
    std::size_t end = query.find('&', i);
    if (end == std::string_view::npos) end = query.size();
    std::string_view seg = query.substr(i, end - i);
    i                    = end + 1;
    if (seg.empty()) continue;

    auto eq     = seg.find('=');
    auto key_sv = (eq == std::string_view::npos) ? seg : seg.substr(0, eq);
    auto val_sv = (eq == std::string_view::npos) ? std::string_view{} : seg.substr(eq + 1);
    if (key_sv.empty()) fail("empty query key", uri);

    auto key            = percent_decode(key_sv, uri);
    auto val            = percent_decode(val_sv, uri);
    out[std::move(key)] = std::move(val);  // last-wins
  }
  return out;
}

bool is_valid_scheme(std::string_view s)
{
  if (s.empty()) return false;
  // Lenient versus RFC 3986 §3.1 (which forbids `_`): the factory and tests
  // accept project-internal schemes like `rdma_s3`, so only reject obvious
  // structural delimiters and whitespace. Leading char is not restricted.
  for (char c : s) {
    auto uc = static_cast<unsigned char>(c);
    if (std::isspace(uc) || c == '/' || c == '?' || c == '#' || c == '&' || c == '=') {
      return false;
    }
  }
  return true;
}

std::string to_lower(std::string_view s)
{
  std::string out;
  out.reserve(s.size());
  for (char c : s)
    out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  return out;
}

}  // namespace

parsed_uri parse(std::string_view uri)
{
  if (uri.empty()) fail("empty URI", uri);

  // S3 object keys are literal: '%', '?', '#' are ordinary key bytes. Handle s3
  // before the fragment strip / query split / percent-decode below (which would
  // mutate the key) and return the raw key. Non-s3 schemes fall through unchanged.
  if (auto delim = uri.find(kSchemeDelim);
      delim != std::string_view::npos && delim > 0 && to_lower(uri.substr(0, delim)) == kS3Scheme) {
    auto rest    = uri.substr(delim + kSchemeDelim.size());
    auto slash   = rest.find('/');
    auto host_sv = (slash == std::string_view::npos) ? rest : rest.substr(0, slash);
    auto key_sv  = (slash == std::string_view::npos) ? std::string_view{} : rest.substr(slash);
    if (host_sv.empty()) fail("empty host", uri);
    // Strip exactly one bucket/key separator slash (S3 REST semantics, matching
    // the general object-store branch); any further leading slashes are key bytes.
    if (!key_sv.empty() && key_sv.front() == '/') key_sv.remove_prefix(1);
    if (key_sv.empty()) fail("empty object key", uri);
    parsed_uri s3;
    s3.scheme = std::string{kS3Scheme};
    s3.host   = std::string{host_sv};
    s3.path   = std::string{key_sv};  // RAW literal key: no percent-decode.
    return s3;
  }

  // Strip fragment early: fragments have no semantics for object-store URIs.
  // Why: users may paste browser URLs; silent drop avoids noisy errors.
  if (auto hash = uri.find('#'); hash != std::string_view::npos) {
    uri = uri.substr(0, hash);
    if (uri.empty()) fail("empty URI after fragment strip", uri);
  }

  parsed_uri out;

  // Bare absolute path -> file scheme.
  // Relative bare paths are rejected: callers must pass an absolute POSIX path
  // or a scheme-qualified URI.
  if (uri.front() == '/') {
    out.scheme = std::string{kFileScheme};
    // Split off query.
    auto qpos       = uri.find('?');
    auto path_part  = (qpos == std::string_view::npos) ? uri : uri.substr(0, qpos);
    auto query_part = (qpos == std::string_view::npos) ? std::string_view{} : uri.substr(qpos + 1);
    out.path        = percent_decode(path_part, uri);
    if (!query_part.empty()) out.query = parse_query(query_part, uri);
    return out;
  }

  // Must have `://` for a scheme-qualified URI; bare relative paths are rejected.
  auto delim = uri.find(kSchemeDelim);
  if (delim == std::string_view::npos)
    fail("relative path not allowed; use absolute or scheme://", uri);
  if (delim == 0) fail("empty scheme", uri);

  auto scheme_sv = uri.substr(0, delim);
  if (!is_valid_scheme(scheme_sv)) fail("invalid scheme syntax", uri);
  // Why: RFC 3986 §3.1 — scheme is case-insensitive; normalize early so registry
  // lookup can use exact string match.
  out.scheme = to_lower(scheme_sv);

  auto rest = uri.substr(delim + kSchemeDelim.size());

  // Split off query from rest.
  auto qpos       = rest.find('?');
  auto authpath   = (qpos == std::string_view::npos) ? rest : rest.substr(0, qpos);
  auto query_part = (qpos == std::string_view::npos) ? std::string_view{} : rest.substr(qpos + 1);

  if (out.scheme == kFileScheme) {
    // file:///abs/path — authority must be empty, path keeps leading `/`.
    if (authpath.empty() || authpath.front() != '/') {
      fail("file:// URI must have absolute path (file:///abs/...)", uri);
    }
    // Why: do not percent-decode host/scheme (only path/query), so no host work here.
    out.host = "";
    out.path = percent_decode(authpath, uri);
  } else {
    // Object-store scheme: authpath is `host[/key]`.
    auto slash   = authpath.find('/');
    auto host_sv = (slash == std::string_view::npos) ? authpath : authpath.substr(0, slash);
    auto key_sv  = (slash == std::string_view::npos) ? std::string_view{} : authpath.substr(slash);
    if (host_sv.empty()) fail("empty host", uri);
    // Why: do not percent-decode host; decoding could smuggle a '/' via %2F and
    // change the authority/path boundary after parsing.
    out.host = std::string{host_sv};

    // Strip the single bucket/key separator slash. Any further leading
    // slashes are part of the object key — under S3 REST semantics the URL
    // path "/<bucket>//foo" yields key="/foo", distinct from "/<bucket>/foo"
    // (key="foo"). Examples after this step:
    //   s3://bucket/key       -> key
    //   s3://bucket//key      -> /key
    //   s3://bucket///key     -> //key
    //   s3://bucket/a//b      -> a//b   (interior runs already preserved)
    //   s3://bucket/          -> throws (empty key after stripping separator)
    // The broader S3 URL surface (region-specific endpoints, virtual-hosted
    // vs path-style, dual-stack, accelerate, etc.) is intentionally not
    // modeled by this parser; providing a canonical URI is the caller's
    // responsibility.
    if (!key_sv.empty() && key_sv.front() == '/') key_sv.remove_prefix(1);
    if (key_sv.empty()) fail("empty object key", uri);
    out.path = percent_decode(key_sv, uri);
  }

  if (!query_part.empty()) out.query = parse_query(query_part, uri);
  return out;
}

std::string strip_file_scheme(std::string_view path)
{
  // Deliberately NOT implemented via parse(): this runs on every datasource open,
  // must not throw on inputs parse() rejects (relative paths, empty keys), and
  // must return the ORIGINAL bytes for everything it does not strip — no
  // percent-decoding, no normalization of any other scheme.
  constexpr std::string_view kFileSchemePrefix = "file://";
  if (path.size() <= kFileSchemePrefix.size()) { return std::string{path}; }
  for (std::size_t i = 0; i < kFileSchemePrefix.size(); ++i) {
    if (std::tolower(static_cast<unsigned char>(path[i])) !=
        static_cast<unsigned char>(kFileSchemePrefix[i])) {
      return std::string{path};
    }
  }
  return std::string{path.substr(kFileSchemePrefix.size())};
}

}  // namespace sirius::io
