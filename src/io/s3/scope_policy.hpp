// Copyright 2026 Sirius Contributors
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace sirius::io::s3 {

/// What a description shows in place of an S3 location.
inline constexpr std::string_view redacted_s3_location = "s3://<redacted>";

/// Whether @p uri starts with `s3://`, the scheme compared without regard to ASCII case.
constexpr bool has_s3_scheme(std::string_view uri) noexcept
{
  constexpr std::string_view scheme = "s3://";
  if (uri.size() < scheme.size()) { return false; }
  for (std::size_t i = 0; i < scheme.size(); ++i) {
    auto const c = uri[i];
    if ((c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : c) != scheme[i]) {
      return false;
    }
  }
  return true;
}

/// Why a bucket name is refused.
enum class bucket_defect : std::uint8_t { none, empty_or_dot, byte };

/// The bucket rule shared by scan locations and store prefixes: one or more ASCII letters,
/// digits, `.`, `_` or `-`, and not `.` or `..`.
constexpr bucket_defect check_bucket(std::string_view bucket) noexcept
{
  if (bucket.empty() || bucket == "." || bucket == "..") { return bucket_defect::empty_or_dot; }
  for (auto const c : bucket) {
    if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '.' ||
          c == '_' || c == '-')) {
      return bucket_defect::byte;
    }
  }
  return bucket_defect::none;
}

/// Scheme-only canonicalization for prefix matching: a copy of @p uri with the bytes before
/// `://` lower-cased. Bucket and key bytes compare exactly.
std::string lowercase_scheme(std::string_view uri);

/// Classify raw key text after an explicit scheme and bucket/key separator. An empty key,
/// trailing or adjacent slash, or whole `.`/`..` segment returns true. With no scheme or key
/// separator, return false. This does not validate the URI or enforce rejection at callers.
/// Default libcurl path handling removes dot segments; stores may have additional aliasing rules.
bool key_has_unstable_segment(std::string_view uri);

/// True when a configured non-empty `uri_prefix` cannot serve @p uri byte-for-byte: the key
/// has an unstable segment (above) or a `%` byte the store might decode.
bool key_defeats_prefix_match(std::string_view uri);

/// @p uri as an error message may name it, without the authentication material a URI can
/// carry: user-info, query (presigned `X-Amz-*` parameters live there) and fragment. Text
/// without an `@` loses only its query and fragment, so plain paths and plain URIs come back
/// unchanged. Text with an `@` is shown only when it starts with a well-formed `scheme://` and
/// every `@` precedes the first `?` or `#`: then everything between the scheme and the last `@`
/// is dropped, whatever bytes the user-info holds (`/`, `:` and `@` included), and the query and
/// fragment are cut. Everything else (an `@` with no leading scheme, or an `@` after a `?`/`#`,
/// whose role cannot be told from a raw secret) is replaced by a fixed placeholder. A key
/// containing `@` is over-redacted, never under-redacted.
std::string redacted_uri(std::string_view uri);

}  // namespace sirius::io::s3
