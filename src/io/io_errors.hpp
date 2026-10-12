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

#include <stdexcept>
#include <string>
#include <utility>

namespace sirius::io {

/**
 * @brief Raised by @c request_authorizer implementations when credential
 *        acquisition or signing fails.
 *
 * Surfaces from:
 *   - missing or malformed static credentials (caught at provider construction
 *     or first call),
 *   - misconfigured endpoint / region,
 *   - underlying signing-library failure (HMAC / SHA256),
 *   - upstream credential broker errors (in future refresh-aware impls).
 *
 * Backends translate this into the broader IO error path of the caller.
 * Future IO-layer error types share this header.
 */
class credential_error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

/**
 * @brief A read found the object to be a different version from the one its
 *        datasource was opened on.
 *
 * Raised by the REST reactor when a conditional range GET is refused (412) or
 * when an accepted response does not carry the validator the open recorded.
 * Nothing from such a response is published; the error is terminal for the
 * read and is not retried.  @c observed_tag() is the response's ETag as
 * received, or empty when the response carried none or was refused (412).
 */
class object_changed_error : public std::runtime_error {
 public:
  object_changed_error(std::string object_path, std::string expected_tag, std::string observed_tag)
    : std::runtime_error(describe(object_path, observed_tag)),
      _object_path(std::move(object_path)),
      _expected_tag(std::move(expected_tag)),
      _observed_tag(std::move(observed_tag))
  {
  }

  [[nodiscard]] std::string const& object_path() const noexcept { return _object_path; }
  [[nodiscard]] std::string const& expected_tag() const noexcept { return _expected_tag; }
  [[nodiscard]] std::string const& observed_tag() const noexcept { return _observed_tag; }

 private:
  static std::string describe(std::string const& path, std::string const& observed)
  {
    return "object " + path + " changed since it was opened" +
           (observed.empty() ? " (no matching validator in the response)"
                             : " (response validator differs)");
  }

  std::string _object_path;
  std::string _expected_tag;
  std::string _observed_tag;
};

}  // namespace sirius::io
