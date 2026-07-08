/*
 * Copyright 2026, Sirius Contributors.
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

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace sirius::io::s3 {

/// Default cap on the number of entries a whole-listing call will accumulate.
/// Exceeding it throws (never truncates) — a partial key set would resolve a
/// glob to a silently incomplete table.
inline constexpr std::size_t default_max_list_objects = 100'000;

/// Default cap on the total number of entries a paged listing will scan across
/// all pages (≤ 1000 LIST round-trips at the maximum page size). Bounds time
/// and request count on a prefix whose population dwarfs the matches.
inline constexpr std::size_t default_max_scanned_objects = 1'000'000;

/// One object from a ListObjectsV2 page: full key + object size in bytes.
/// The size rides the LIST response for free, so downstream opens need no
/// size-discovery round-trip.
struct list_entry {
  std::string key;
  std::uint64_t size = 0;
};

/// One parsed page of an S3 ListObjectsV2 response.
struct list_objects_v2_page {
  /// Every `<Contents>` object in document order, XML-entity-unescaped.
  /// Excludes `<CommonPrefixes><Prefix>` entries (directory rollups, not keys).
  std::vector<list_entry> entries;
  /// `<IsTruncated>` — true when another page follows.
  bool is_truncated = false;
  /// `<NextContinuationToken>` — the cursor for the next page; empty when absent.
  std::string next_continuation_token;
};

/**
 * @brief Parse one ListObjectsV2 XML response body.
 *
 * Hand-rolled (no XML dependency): the S3 ListObjectsV2 schema is small and
 * fixed. Tolerant of the `xmlns` attribute on `<ListBucketResult>` and leading /
 * trailing whitespace. XML entities (`&amp; &lt; &gt; &quot; &apos;`) in key /
 * token text are unescaped.
 *
 * The body must be a COMPLETE response — a truncated / partial (but
 * transport-complete) body must not parse as a silently-incomplete listing, so
 * both the root close tag and every `<Contents>` block are required to close.
 *
 * @throw std::runtime_error when @p xml is not a recognizable, complete
 *        ListObjectsV2 response: no `<ListBucketResult>` open tag (e.g. an S3
 *        `<Error>` body or an empty string), a missing `</ListBucketResult>`
 *        close tag, or a `<Contents>` opened without a `</Contents>` close (both
 *        signal a truncated body); or a `<Contents>` whose `<Size>` is missing /
 *        non-numeric / overflows `uint64_t` (a silently-zeroed or wrapped size
 *        would defeat the no-HEAD open path downstream). `<Size>0</Size>` is
 *        legal — S3 allows zero-byte objects.
 */
list_objects_v2_page parse_list_objects_v2(std::string_view xml);

}  // namespace sirius::io::s3
