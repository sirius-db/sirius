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

  // Strip fragment early: fragments have no semantics for object-store URIs.
  // Why: users may paste browser URLs; silent drop avoids noisy errors.
  if (auto hash = uri.find('#'); hash != std::string_view::npos) {
    uri = uri.substr(0, hash);
    if (uri.empty()) fail("empty URI after fragment strip", uri);
  }

  parsed_uri out;

  // Bare absolute path -> file scheme.
  // Relative bare paths are rejected: callers must pass an absolute POSIX path
  // or a scheme-qualified URI. (Spec: PR8 normalization rule.)
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

    // Collapse any run of leading slashes to 0 slashes.
    // Why: s3://bucket/key and s3://bucket//key must both yield key="key";
    // object stores treat them as identical.
    std::size_t i = 0;
    while (i < key_sv.size() && key_sv[i] == '/')
      ++i;
    key_sv.remove_prefix(i);
    if (key_sv.empty()) fail("empty object key", uri);
    out.path = percent_decode(key_sv, uri);
  }

  if (!query_part.empty()) out.query = parse_query(query_part, uri);
  return out;
}

}  // namespace sirius::io
