// Copyright 2026 Sirius Contributors
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "object_store_scope/settings.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace sirius::object_store_scope {

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

/// Whether @p value is one of the declared `transport_*` selectors.
[[nodiscard]] bool is_declared_transport(std::uint32_t value) noexcept;

/// Whether @p value is one of the declared `signing_*` selectors.
[[nodiscard]] bool is_declared_signing_mode(std::uint32_t value) noexcept;

/// Checks @p value in a fixed order (endpoint, region, access_key, secret_key, transport,
/// signing_mode, ca_bundle_path, uri_prefix, then the options @p backend does not honor) and
/// returns the first violation. No network access and no backend resource is created; CA
/// readability may access the local filesystem.
[[nodiscard]] std::optional<violation> check_settings(const settings& value,
                                                      const capabilities& backend);

/// The checks of `check_settings()` that depend on @p value alone, in the same relative order:
/// endpoint, region, access_key, secret_key, transport, signing_mode, uri_prefix. The filesystem
/// is not accessed and no backend is consulted, so `ca_bundle_path` is not examined.
[[nodiscard]] std::optional<violation> check_settings_intrinsic(const settings& value);

/// Shape of a scope key: empty (the default scope), `s3://bucket/` or `s3://bucket/prefix/`.
[[nodiscard]] std::optional<violation> check_uri_prefix(std::string_view uri_prefix);

/// The fixed reason text of @p refused, without the field name.
[[nodiscard]] std::string describe(const violation& refused, const capabilities& backend);

}  // namespace sirius::object_store_scope
