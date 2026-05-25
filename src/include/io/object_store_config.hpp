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

/// Inert configuration carrier for remote object-store backends.
/// PR1 only exposes the POD fields and enum/string helpers; runtime plumbing and
/// backend consumption live in the integration/backend PRs.
/// Empty strings are valid and mean "no value configured".
struct object_store_config {
  std::string endpoint;
  std::string region;
  std::string access_key;
  std::string secret_key;

  /// Requested S3 transport. AUTO leaves the concrete backend/integration code
  /// to choose based on URI scheme and endpoint capabilities.
  enum class transport { AUTO, HTTP, RDMA };
  transport s3_transport = transport::AUTO;
};

inline bool string_to_enum(std::string_view sv, object_store_config::transport& t)
{
  static const std::unordered_map<std::string_view, object_store_config::transport> map = {
    {"auto", object_store_config::transport::AUTO},
    {"http", object_store_config::transport::HTTP},
    {"https", object_store_config::transport::HTTP},
    {"rdma", object_store_config::transport::RDMA},
  };
  auto it = map.find(sv);
  if (it != map.end()) {
    t = it->second;
    return true;
  }
  return false;
}

inline bool enum_to_string(object_store_config::transport t, std::string& s)
{
  switch (t) {
    case object_store_config::transport::AUTO: s = "auto"; return true;
    case object_store_config::transport::HTTP: s = "http"; return true;
    case object_store_config::transport::RDMA: s = "rdma"; return true;
  }
  return false;
}

}  // namespace sirius::io
