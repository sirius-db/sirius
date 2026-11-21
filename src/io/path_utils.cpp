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

#include "io/path_utils.hpp"

#include <cucascade/cudf/datasource.hpp>
#include <duckdb/common/path.hpp>

#include <cctype>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace sirius::io {

namespace {

[[noreturn]] void fail(std::string_view reason, std::string_view uri)
{
  std::string msg = "strip_file_scheme: ";
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

}  // namespace

std::string strip_file_scheme(std::string_view path)
{
  // Deliberately NOT implemented via a URI parser: this runs on every datasource open,
  // must not throw on inputs a parser rejects (relative paths, empty keys), and
  // must return the ORIGINAL bytes for everything it does not strip — no
  // normalization of any other scheme.
  //
  // Do NOT reduce the `file:` handling to a `file://` prefix test: that strips one of the three
  // forms the URI scheme admits and leaves `file:/abs` unopenable by any local datasource.
  constexpr std::string_view kFileSchemePrefix = "file:";
  if (path.size() <= kFileSchemePrefix.size()) { return std::string{path}; }
  for (std::size_t i = 0; i < kFileSchemePrefix.size(); ++i) {
    if (std::tolower(static_cast<unsigned char>(path[i])) !=
        static_cast<unsigned char>(kFileSchemePrefix[i])) {
      return std::string{path};
    }
  }

  // duckdb::Path dispatches on a case-SENSITIVE "file:/", but the scheme is case-insensitive per
  // RFC 3986 and manifests spell it FILE:// and File://. A missed match pairs a delete file with
  // no data file, which silently returns deleted rows. Scheme bytes only.
  std::string normalized{kFileSchemePrefix};
  normalized.append(path.substr(kFileSchemePrefix.size()));

  // The non-standard "double-slash path" form, which duckdb::Path rejects. This repo's fixtures
  // use it for repo-RELATIVE paths, which committed metadata cannot spell as absolute URIs, so it
  // keeps the plain strip.
  constexpr std::string_view kDoubleSlash = "file://";
  auto const bare_prefix_strip            = [&]() -> std::string {
    return path.size() > kDoubleSlash.size() ? std::string{path.substr(kDoubleSlash.size())}
                                                        : std::string{path};
  };

  std::string local;
  try {
    auto const parsed = duckdb::Path::FromString(normalized);
    if (!parsed.IsLocal()) { return bare_prefix_strip(); }
    local = parsed.GetAnchor() + parsed.GetPath() + parsed.GetTrailingSeparator();
  } catch (...) {
    // Must not throw: this is on every datasource open.
    return bare_prefix_strip();
  }

  // Only what was stripped is a URI, so only there is `%20` a space.
  try {
    return percent_decode(local, path);
  } catch (...) {
    return local;
  }
}

std::unique_ptr<cucascade::io::datasource> open_datasource(
  std::shared_ptr<cucascade::io::ioctx> io_ctx, std::string path)
{
  return cucascade::io::open_datasource(std::move(io_ctx), strip_file_scheme(path));
}

std::unique_ptr<cucascade::io::datasource> open_datasource(
  std::shared_ptr<cucascade::io::ioctx> io_ctx, std::string path, cucascade::io::open_hint hint)
{
  return cucascade::io::open_datasource(std::move(io_ctx), strip_file_scheme(path), hint);
}

}  // namespace sirius::io
