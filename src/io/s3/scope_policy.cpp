// Copyright 2026 Sirius Contributors
// SPDX-License-Identifier: Apache-2.0
#include "io/s3/scope_policy.hpp"

#include <algorithm>
#include <cctype>

namespace sirius::io::s3 {

std::string lowercase_scheme(std::string_view uri)
{
  std::string out{uri};
  auto const sep = out.find("://");
  if (sep == std::string::npos) { return out; }
  for (std::size_t i = 0; i < sep; ++i) {
    out[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(out[i])));
  }
  return out;
}

namespace {

std::string_view object_key(std::string_view uri)
{
  auto const sep = uri.find("://");
  auto const bucket_end =
    sep == std::string_view::npos ? std::string_view::npos : uri.find('/', sep + 3);
  return bucket_end == std::string_view::npos ? std::string_view{} : uri.substr(bucket_end + 1);
}

// RFC 3986 scheme: an ASCII letter, then ASCII letters, digits, `+`, `-` or `.`.
// A prefix outside this grammar cannot be trusted as a leading scheme. Classified by byte value,
// not through <cctype>, so the answer does not depend on the embedding process's locale.
bool is_scheme(std::string_view text)
{
  auto const alpha = [](char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); };
  auto const digit = [](char c) { return c >= '0' && c <= '9'; };
  if (text.empty() || !alpha(text.front())) { return false; }
  return std::all_of(text.begin(), text.end(), [&](char c) {
    return alpha(c) || digit(c) || c == '+' || c == '-' || c == '.';
  });
}

}  // namespace

bool key_has_unstable_segment(std::string_view uri)
{
  auto const sep = uri.find("://");
  if (sep == std::string_view::npos || uri.find('/', sep + 3) == std::string_view::npos) {
    return false;  // no bucket/key separator: nothing to normalize
  }
  auto key = object_key(uri);
  if (key.empty() || key.back() == '/') { return true; }
  while (!key.empty()) {
    auto const slash   = key.find('/');
    auto const segment = key.substr(0, slash);
    if (segment.empty() || segment == "." || segment == "..") { return true; }
    if (slash == std::string_view::npos) { break; }
    key.remove_prefix(slash + 1);
  }
  return false;
}

bool key_defeats_prefix_match(std::string_view uri)
{
  if (key_has_unstable_segment(uri)) { return true; }
  return object_key(uri).find('%') != std::string_view::npos;
}

std::string redacted_uri(std::string_view uri)
{
  constexpr std::string_view kPlaceholder = "<redacted-uri>";
  auto const at                           = uri.rfind('@');
  if (at == std::string_view::npos) { return std::string{uri.substr(0, uri.find_first_of("?#"))}; }
  auto const scheme_end = uri.find("://");
  if (scheme_end == std::string_view::npos || !is_scheme(uri.substr(0, scheme_end))) {
    return std::string{kPlaceholder};
  }
  // The first `?` or `#` after the scheme is the earliest possible suffix boundary.
  // If the last `@` is beyond it, return the placeholder. Otherwise treat all bytes after
  // the scheme through the last `@` as sensitive; this may remove bucket/key bytes too.
  // Copy only the suffix before that boundary, so query and fragment bytes are never copied.
  auto const cut = uri.find_first_of("?#", scheme_end + 3);
  if (cut != std::string_view::npos && at > cut) { return std::string{kPlaceholder}; }
  auto const end = cut == std::string_view::npos ? uri.size() : cut;
  std::string out{uri.substr(0, scheme_end + 3)};
  out.append(uri.substr(at + 1, end - (at + 1)));
  return out;
}

}  // namespace sirius::io::s3
