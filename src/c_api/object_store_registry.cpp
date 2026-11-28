// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include "c_api/error_internal.hpp"
#include "c_api/handle_internal.hpp"
#include "c_api/object_store_config_internal.hpp"
#include "c_api/object_store_registry_internal.hpp"
#include "object_store_scope/validation.hpp"

#include <sirius-c/object_store/registry.h>

#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

namespace {
namespace scope = sirius::object_store_scope;
using sirius::c_api::fail;
using sirius::c_api::invoke_redacted;
}  // namespace

void sirius_object_store_registry_retain(sirius_object_store_registry* registry) noexcept
{
  sirius::c_api::retain(registry);
}
void sirius_object_store_registry_release(sirius_object_store_registry* registry) noexcept
{
  sirius::c_api::release(registry);
}

sirius_status sirius_object_store_registry_create(sirius_object_store_registry** out,
                                                  sirius_error** error) noexcept
{
  if (out) { *out = nullptr; }
  return invoke_redacted(error, [&] {
    if (!out) { return fail(SIRIUS_INVALID_ARGUMENT, "out_registry is NULL", error); }
    *out = new sirius_object_store_registry{.stores = std::make_shared<const scope::scope_set>()};
    return sirius_status{SIRIUS_SUCCESS};
  });
}

sirius_status sirius_object_store_registry_with_store(const sirius_object_store_registry* registry,
                                                      const char* uri_prefix,
                                                      size_t uri_prefix_size,
                                                      const sirius_object_store_config* config,
                                                      sirius_object_store_registry** out,
                                                      sirius_error** error) noexcept
{
  if (out) { *out = nullptr; }
  return invoke_redacted(error, [&] {
    if (!registry || !config || !out) {
      return fail(
        SIRIUS_INVALID_ARGUMENT, "Registry, configuration or output pointer is NULL", error);
    }
    if (!uri_prefix && uri_prefix_size != 0) {
      return fail(SIRIUS_INVALID_ARGUMENT, "Text is NULL with a nonzero size", error);
    }
    if (uri_prefix_size != 0 && std::memchr(uri_prefix, '\0', uri_prefix_size)) {
      return fail(SIRIUS_INVALID_ARGUMENT, "Text contains a NUL byte", error);
    }
    auto const key = std::string_view{uri_prefix ? uri_prefix : "", uri_prefix_size};
    if (auto const refused = scope::check_uri_prefix(key)) {
      if (refused->field != "uri_prefix" || !scope::is_prefix_reason(refused->why)) {
        return fail(SIRIUS_INTERNAL_ERROR, "Unexpected prefix validation reason", error);
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
    auto value = config->value;
    value.uri_prefix.assign(key);
    auto store = std::make_shared<const scope::entry>(
      scope::entry{std::make_shared<const scope::settings>(std::move(value)), nullptr});
    *out = new sirius_object_store_registry{
      .stores = scope::with_entry(*registry->stores, std::move(store))};
    return sirius_status{SIRIUS_SUCCESS};
  });
}
