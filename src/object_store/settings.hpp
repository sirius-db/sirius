// Copyright 2026 Sirius Contributors
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace sirius::object_store {

inline constexpr std::uint32_t transport_auto = 0;
inline constexpr std::uint32_t transport_http = 1;
inline constexpr std::uint32_t transport_rdma = 2;

inline constexpr std::uint32_t signing_presigned = 0;
inline constexpr std::uint32_t signing_header    = 1;

/// Object-store (S3 API) settings of one scope, as the engine holds them. A public interface
/// converts its own descriptor into this value. `transport` and `signing_mode` keep the caller's
/// raw value, so an undeclared one reaches `check_settings()` instead of being narrowed away.
struct settings {
  std::string endpoint;
  std::string region;
  std::string access_key;
  std::string secret_key;
  std::string session_token;
  std::uint32_t transport{transport_auto};
  std::uint32_t signing_mode{signing_presigned};
  std::string ca_bundle_path;
  std::string uri_prefix;
};

/// What the configured IO backend honors. An option the backend would ignore is refused by
/// `check_settings()` rather than dropped.
struct capabilities {
  bool rdma_transport{false};
  bool signing_mode_choice{true};
  bool custom_ca_bundle{true};
  /// Names the backend in a refusal; refers to static storage.
  std::string_view backend_name{"native"};
};

}  // namespace sirius::object_store
