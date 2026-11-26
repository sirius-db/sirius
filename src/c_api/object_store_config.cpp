// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include "c_api/error_internal.hpp"
#include "c_api/handle_internal.hpp"
#include "c_api/object_store_config_internal.hpp"
#include "object_store_scope/validation.hpp"

#include <sirius-c/object_store/config_builder.h>

#include <cstring>
#include <new>
#include <string>
#include <utility>

namespace {
namespace scope = sirius::object_store_scope;
using sirius::c_api::fail;
using sirius::c_api::invoke_redacted;
using sirius::c_api::release;
using sirius::c_api::retain;

// A selector is stored as given, so the public constants must be the core's.
static_assert(SIRIUS_OBJECT_STORE_TRANSPORT_AUTO == scope::transport_auto);
static_assert(SIRIUS_OBJECT_STORE_TRANSPORT_HTTP == scope::transport_http);
static_assert(SIRIUS_OBJECT_STORE_TRANSPORT_RDMA == scope::transport_rdma);
static_assert(SIRIUS_OBJECT_STORE_SIGNING_PRESIGNED == scope::signing_presigned);
static_assert(SIRIUS_OBJECT_STORE_SIGNING_HEADER == scope::signing_header);

// A copy of `builder` with one field changed by `change`. `refusal` runs once the output slots
// are cleared and before anything is copied; a text it returns refuses the argument.
template <typename Refusal, typename Change>
sirius_status derive(const sirius_object_store_config_builder* builder,
                     sirius_object_store_config_builder** out,
                     sirius_error** error,
                     Refusal&& refusal,
                     Change&& change) noexcept
{
  if (out) { *out = nullptr; }
  return invoke_redacted(error, [&] {
    if (!builder || !out) {
      return fail(SIRIUS_INVALID_ARGUMENT, "Builder or output pointer is NULL", error);
    }
    if (const char* invalid_argument = refusal()) {
      return fail(SIRIUS_INVALID_ARGUMENT, invalid_argument, error);
    }
    auto next = builder->value;
    change(next);
    *out = new sirius_object_store_config_builder{.value = std::move(next)};
    return sirius_status{SIRIUS_SUCCESS};
  });
}

sirius_status with_text(const sirius_object_store_config_builder* builder,
                        std::string scope::settings::* field,
                        const char* data,
                        size_t size,
                        sirius_object_store_config_builder** out,
                        sirius_error** error) noexcept
{
  // The range is checked and copied with no write to the caller's storage in between, so the
  // bytes stored are the bytes checked even when the range overlaps an output slot.
  return derive(
    builder,
    out,
    error,
    [&]() -> const char* {
      if (!data && size != 0) { return "Text is NULL with a nonzero size"; }
      if (size != 0 && std::memchr(data, '\0', size)) { return "Text contains a NUL byte"; }
      return nullptr;
    },
    [&](scope::settings& next) { (next.*field).assign(data ? data : "", size); });
}

// Expected field refusals from build. Undeclared selectors cannot enter through derivation;
// CA-file, scope-key and backend reasons cannot come from intrinsic validation. Any such reason
// is an internal invariant failure.
bool refuses_build(scope::reason why) noexcept
{
  using scope::reason;
  switch (why) {
    case reason::required:
    case reason::endpoint_missing_scheme:
    case reason::endpoint_bad_scheme:
    case reason::endpoint_control_bytes:
    case reason::endpoint_missing_host:
    case reason::endpoint_has_path:
    case reason::endpoint_has_user_info:
    case reason::endpoint_unterminated_ipv6:
    case reason::endpoint_malformed:
    case reason::endpoint_bad_port: return true;
    case reason::transport_unknown:
    case reason::signing_mode_unknown:
    case reason::ca_bundle_not_regular_file:
    case reason::ca_bundle_unreadable:
    case reason::prefix_shape:
    case reason::prefix_no_trailing_slash:
    case reason::prefix_query_or_fragment:
    case reason::prefix_bucket:
    case reason::prefix_unmatchable_segment:
    case reason::transport_unavailable: return false;
  }
  return false;
}
}  // namespace

void sirius_object_store_config_retain(sirius_object_store_config* config) noexcept
{
  retain(config);
}
void sirius_object_store_config_release(sirius_object_store_config* config) noexcept
{
  release(config);
}
void sirius_object_store_config_builder_retain(sirius_object_store_config_builder* builder) noexcept
{
  retain(builder);
}
void sirius_object_store_config_builder_release(
  sirius_object_store_config_builder* builder) noexcept
{
  release(builder);
}

sirius_status sirius_object_store_config_builder_create(sirius_object_store_config_builder** out,
                                                        sirius_error** error) noexcept
{
  if (out) { *out = nullptr; }
  return invoke_redacted(error, [&] {
    if (!out) { return fail(SIRIUS_INVALID_ARGUMENT, "out_builder is NULL", error); }
    *out = new sirius_object_store_config_builder{};
    return sirius_status{SIRIUS_SUCCESS};
  });
}

sirius_status sirius_object_store_config_builder_with_endpoint(
  const sirius_object_store_config_builder* builder,
  const char* endpoint,
  size_t endpoint_size,
  sirius_object_store_config_builder** out,
  sirius_error** error) noexcept
{
  return with_text(builder, &scope::settings::endpoint, endpoint, endpoint_size, out, error);
}
sirius_status sirius_object_store_config_builder_with_region(
  const sirius_object_store_config_builder* builder,
  const char* region,
  size_t region_size,
  sirius_object_store_config_builder** out,
  sirius_error** error) noexcept
{
  return with_text(builder, &scope::settings::region, region, region_size, out, error);
}
sirius_status sirius_object_store_config_builder_with_access_key(
  const sirius_object_store_config_builder* builder,
  const char* access_key,
  size_t access_key_size,
  sirius_object_store_config_builder** out,
  sirius_error** error) noexcept
{
  return with_text(builder, &scope::settings::access_key, access_key, access_key_size, out, error);
}
sirius_status sirius_object_store_config_builder_with_secret_key(
  const sirius_object_store_config_builder* builder,
  const char* secret_key,
  size_t secret_key_size,
  sirius_object_store_config_builder** out,
  sirius_error** error) noexcept
{
  return with_text(builder, &scope::settings::secret_key, secret_key, secret_key_size, out, error);
}
sirius_status sirius_object_store_config_builder_with_session_token(
  const sirius_object_store_config_builder* builder,
  const char* session_token,
  size_t session_token_size,
  sirius_object_store_config_builder** out,
  sirius_error** error) noexcept
{
  return with_text(
    builder, &scope::settings::session_token, session_token, session_token_size, out, error);
}
sirius_status sirius_object_store_config_builder_with_ca_bundle_path(
  const sirius_object_store_config_builder* builder,
  const char* ca_bundle_path,
  size_t ca_bundle_path_size,
  sirius_object_store_config_builder** out,
  sirius_error** error) noexcept
{
  return with_text(
    builder, &scope::settings::ca_bundle_path, ca_bundle_path, ca_bundle_path_size, out, error);
}

sirius_status sirius_object_store_config_builder_with_transport(
  const sirius_object_store_config_builder* builder,
  sirius_object_store_transport transport,
  sirius_object_store_config_builder** out,
  sirius_error** error) noexcept
{
  // The core decides membership. The text is this unit's: the core's names the C++ enumerators.
  return derive(
    builder,
    out,
    error,
    [&]() -> const char* {
      return scope::is_declared_transport(transport)
               ? nullptr
               : "transport: must be one of SIRIUS_OBJECT_STORE_TRANSPORT_AUTO, _HTTP, _RDMA";
    },
    [&](scope::settings& next) { next.transport = transport; });
}
sirius_status sirius_object_store_config_builder_with_signing_mode(
  const sirius_object_store_config_builder* builder,
  sirius_object_store_signing_mode signing_mode,
  sirius_object_store_config_builder** out,
  sirius_error** error) noexcept
{
  return derive(
    builder,
    out,
    error,
    [&]() -> const char* {
      return scope::is_declared_signing_mode(signing_mode)
               ? nullptr
               : "signing_mode: must be one of SIRIUS_OBJECT_STORE_SIGNING_PRESIGNED, _HEADER";
    },
    [&](scope::settings& next) { next.signing_mode = signing_mode; });
}

sirius_status sirius_object_store_config_builder_build(
  const sirius_object_store_config_builder* builder,
  sirius_object_store_config** out,
  sirius_error** error) noexcept
{
  if (out) { *out = nullptr; }
  return invoke_redacted(error, [&] {
    if (!builder || !out) {
      return fail(SIRIUS_INVALID_ARGUMENT, "Builder or output pointer is NULL", error);
    }
    if (auto const refused = scope::check_settings_intrinsic(builder->value)) {
      if (!refuses_build(refused->why)) {
        return fail(SIRIUS_INTERNAL_ERROR, "Builder holds settings no derivation produces", error);
      }
      // The status is settled. The text is built only when a diagnostic was asked for, and a
      // failure to build it leaves a NULL diagnostic.
      return sirius::c_api::with_diagnostic(SIRIUS_INVALID_CONFIGURATION, error, [&] {
        auto const message       = std::string{refused->field} + ": " + scope::describe(*refused);
        sirius_error* diagnostic = nullptr;
        (void)fail(SIRIUS_INVALID_CONFIGURATION, message.c_str(), &diagnostic);
        return diagnostic;
      });
    }
    *out = new sirius_object_store_config{.value = builder->value};
    return sirius_status{SIRIUS_SUCCESS};
  });
}
