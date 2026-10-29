// Copyright 2026 Sirius Contributors
// SPDX-License-Identifier: Apache-2.0
#include "io/s3/scope_policy.hpp"

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


}  // namespace sirius::io::s3
