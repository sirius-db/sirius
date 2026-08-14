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

#pragma once

#include <string>
#include <string_view>
#include <unordered_map>

namespace sirius::io {

/**
 * @brief Result of parsing a datasource URI.
 *
 * Conventions:
 *   - @c scheme is always lowercased (`S3://...` -> `s3`).
 *   - @c host is the authority verbatim (no percent-decoding); may contain
 *     @c ":port" (e.g. `bucket:9000`). Empty for schemes without an authority
 *     (the `file` scheme and bare absolute paths).
 *   - @c path is percent-decoded EXCEPT for the `s3` scheme, whose key is taken
 *     literally (AWS-CLI semantics): `s3://b/a%20b` -> `a%20b`, and `?`/`#` are
 *     ordinary key bytes, not delimiters. For the other object-store schemes
 *     (gs/azure/http/https/rdma_s3) @c path is the percent-decoded object key.
 *     All object-store schemes strip exactly one bucket/key separator slash;
 *     further leading slashes are part of the key per S3 REST semantics (e.g.
 *     `s3://b/k` -> `k`, `s3://b//k` -> `/k`, `s3://b///k` -> `//k`). For the
 *     `file` scheme @c path keeps its leading `/`.
 *   - @c query holds percent-decoded values. Duplicate keys are last-wins
 *     (unordered_map cannot represent multi-values; matches AWS SDK behavior).
 *     The `s3` scheme never populates @c query (its `?...` is literal key bytes).
 */
struct parsed_uri {
  std::string scheme;
  std::string host;
  std::string path;
  std::unordered_map<std::string, std::string> query;
};

/**
 * @brief Parse @p uri into a @c parsed_uri.
 *
 * Supported shapes:
 *   - `s3://bucket/key` — key taken LITERALLY (no percent-decode, no `?`/`#`
 *     split): `s3://b/a%20b` opens the object whose key is `a%20b`
 *   - `gs://bucket/key`, `azure://container/blob` (percent-decoded key + query)
 *   - `file:///abs/path`, bare absolute `/abs/path`
 *   - Uppercase schemes (normalized to lowercase; `S3://` takes the literal path)
 *   - Fragments (`#...`) are silently stripped for non-s3 schemes; for s3 a `#`
 *     is a literal key byte
 *   - Exactly one bucket/key separator slash is consumed; any further
 *     leading slashes survive into the key (`s3://b/k` -> `k`;
 *     `s3://b//k` -> `/k`; `s3://b///k` -> `//k`)
 *
 * @throw std::invalid_argument on:
 *   - empty URI
 *   - empty scheme (`://foo`)
 *   - relative bare path (`relative/x`, `./x`)
 *   - empty object key (`s3://bucket`, `s3://bucket/`)
 *   - empty query key (`?=val`) — non-s3 schemes only
 *   - malformed percent-encoding (`%ZZ`, truncated `%A`) — non-s3 schemes only
 *     (for s3 these are valid literal key bytes)
 */
parsed_uri parse(std::string_view uri);

}  // namespace sirius::io
