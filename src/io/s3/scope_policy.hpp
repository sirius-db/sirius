// Copyright 2026 Sirius Contributors
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>
#include <string_view>

namespace sirius::io::s3 {

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
