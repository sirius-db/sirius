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
#include <utility>

namespace sirius::io::rest::s3 {

/// Split an S3 service endpoint into (lowercase scheme, lowercase host[:port]).
///
/// The endpoint identifies the service, not a bucket or object: the scheme must be `http` or
/// `https` and anything past `host[:port]` (path, query, fragment) is rejected. This is the
/// parser the SigV4 authorizers construct from; callers that validate a configuration before
/// any backend exists use the same rules through this function.
/// @throws sirius::io::credential_error on an empty or malformed endpoint.
[[nodiscard]] std::pair<std::string, std::string> parse_endpoint(std::string_view endpoint);

}  // namespace sirius::io::rest::s3
