// Copyright 2026 Sirius Contributors
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "object_store/settings.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace sirius::object_store {

/// Why a settings value or a scope key was refused.
enum class reason : std::uint8_t {
  required,
  endpoint_missing_scheme,
  endpoint_bad_scheme,
  endpoint_control_bytes,
  endpoint_missing_host,
  endpoint_has_path,
  endpoint_has_user_info,
  endpoint_unterminated_ipv6,
  endpoint_malformed,
  endpoint_bad_port,
  transport_unknown,
  signing_mode_unknown,
  ca_bundle_not_regular_file,
  ca_bundle_unreadable,
  prefix_shape,
  prefix_no_trailing_slash,
  prefix_query_or_fragment,
  prefix_unmatchable_segment,
  transport_unavailable,
  signing_mode_unsupported,
  ca_bundle_unsupported,
};

/// A refused field and the reason. `field` refers to static storage; the offending value is
/// never carried.
struct violation {
  std::string_view field;
  reason why;
};

/// Checks @p value in a fixed order (endpoint, region, access_key, secret_key, transport,
/// signing_mode, ca_bundle_path, uri_prefix, then the options @p backend does not honor) and
/// returns the first violation. No network access and no backend resource is created; CA
/// readability may access the local filesystem.
[[nodiscard]] std::optional<violation> check_settings(const settings& value,
                                                      const capabilities& backend);

/// Shape of a scope key: empty (the default scope), `s3://bucket/` or `s3://bucket/prefix/`.
[[nodiscard]] std::optional<violation> check_uri_prefix(std::string_view uri_prefix);

/// The fixed reason text of @p refused, without the field name.
[[nodiscard]] std::string describe(const violation& refused, const capabilities& backend);

}  // namespace sirius::object_store
