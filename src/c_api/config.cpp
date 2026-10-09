// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include "c_api/config_internal.hpp"
#include "c_api/error_internal.hpp"

#include <sirius/c/context/config_builder.h>
#include <sirius/c/version.h>

#include <string_view>

struct sirius_error {
  std::string message;
};

namespace sirius::c_api {
sirius_status fail(sirius_status status, const char* message, sirius_error** error) noexcept
{
  if (error) {
    try {
      *error = new sirius_error{message};
    } catch (...) {
      *error = nullptr;
    }
  }
  return status;
}
}  // namespace sirius::c_api

namespace {
template <typename T>
void retain(T* value) noexcept
{
  if (value) { value->references.fetch_add(1, std::memory_order_relaxed); }
}
template <typename T>
void release(T* value) noexcept
{
  if (value && value->references.fetch_sub(1, std::memory_order_acq_rel) == 1) { delete value; }
}
}  // namespace

uint32_t sirius_abi_version() noexcept { return SIRIUS_ABI_VERSION; }
const char* sirius_error_message(const sirius_error* error) noexcept
{
  return error ? error->message.c_str() : "";
}
size_t sirius_error_message_size(const sirius_error* error) noexcept
{
  return error ? error->message.size() : 0;
}
void sirius_error_destroy(sirius_error* error) noexcept { delete error; }
void sirius_config_retain(sirius_config* config) noexcept { retain(config); }
void sirius_config_release(sirius_config* config) noexcept { release(config); }
void sirius_config_builder_retain(sirius_config_builder* builder) noexcept { retain(builder); }
void sirius_config_builder_release(sirius_config_builder* builder) noexcept { release(builder); }

sirius_status sirius_config_builder_create(sirius_config_builder** out,
                                           sirius_error** error) noexcept
{
  if (out) { *out = nullptr; }
  return sirius::c_api::invoke(error, [&] {
    if (!out) { return sirius::c_api::fail(SIRIUS_INVALID_ARGUMENT, "out_builder is NULL", error); }
    *out = new sirius_config_builder;
    return sirius_status{SIRIUS_SUCCESS};
  });
}
sirius_status sirius_config_builder_from_yaml(const char* path,
                                              size_t size,
                                              sirius_config_builder** out,
                                              sirius_error** error) noexcept
{
  if (out) { *out = nullptr; }
  return sirius::c_api::invoke(error, [&] {
    if (!out || !path || std::string_view(path, size).find('\0') != std::string_view::npos) {
      return sirius::c_api::fail(SIRIUS_INVALID_ARGUMENT, "Invalid path or output pointer", error);
    }
    *out = new sirius_config_builder(
      sirius::load_configuration(std::filesystem::path(std::string(path, size))));
    return sirius_status{SIRIUS_SUCCESS};
  });
}
sirius_status sirius_config_builder_build(const sirius_config_builder* builder,
                                          sirius_config** out,
                                          sirius_error** error) noexcept
{
  if (out) { *out = nullptr; }
  return sirius::c_api::invoke(error, [&] {
    if (!builder || !out) {
      return sirius::c_api::fail(
        SIRIUS_INVALID_ARGUMENT, "Builder or output pointer is NULL", error);
    }
    *out = new sirius_config(builder->config);
    return sirius_status{SIRIUS_SUCCESS};
  });
}
