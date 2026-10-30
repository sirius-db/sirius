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

#include "sirius/ffi.hpp"

#include <array>
#include <memory>
#include <string>
#include <string_view>

namespace sirius::test {

// Synthetic, deliberately recognizable bytes; never real credentials.
inline constexpr std::array<std::string_view, 5> object_store_sentinels{
  "AKIA_SENTINEL_7f3a",
  "SECRET_SENTINEL_9b42",
  "TOKEN_SENTINEL_c681",
  "SECRET/SENTINEL/9b42",
  "QUERY_SIGNATURE_SENTINEL_d806"};

// Mutable test inputs let each negative case alter exactly one field before constructing
// the public immutable aggregate. Tests never modify a published snapshot.
struct object_store_inputs {
  std::string endpoint{"https://192.0.2.1:9"};
  std::string region{"fixture-region"};
  std::string access_key{object_store_sentinels[0]};
  std::string secret_key{object_store_sentinels[1]};
  std::string session_token{object_store_sentinels[2]};
  sirius::ffi::Transport transport{sirius::ffi::Transport::kAuto};
  sirius::ffi::SigningMode signing_mode{sirius::ffi::SigningMode::kPresigned};
  std::string ca_bundle_path;
  std::string uri_prefix{"s3://bucket/prefix/"};

  [[nodiscard]] std::shared_ptr<sirius::ffi::ObjectStoreConfig> snapshot() const
  {
    return std::make_shared<sirius::ffi::ObjectStoreConfig>(
      sirius::ffi::ObjectStoreConfig{.endpoint       = endpoint,
                                     .region         = region,
                                     .access_key     = access_key,
                                     .secret_key     = secret_key,
                                     .session_token  = session_token,
                                     .transport      = transport,
                                     .signing_mode   = signing_mode,
                                     .ca_bundle_path = ca_bundle_path,
                                     .uri_prefix     = uri_prefix});
  }
};

}  // namespace sirius::test
