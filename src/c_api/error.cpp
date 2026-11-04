// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include "c_api/error_internal.hpp"

#include <string>

struct sirius_error {
  std::string message;
};

namespace sirius::c_api {
sirius_status fail(sirius_status status, const char* message, sirius_error** error) noexcept
{
  return with_diagnostic(status, error, [&] { return new sirius_error{message}; });
}
}  // namespace sirius::c_api

const char* sirius_error_message(const sirius_error* error) noexcept
{
  return error ? error->message.c_str() : "";
}
size_t sirius_error_message_size(const sirius_error* error) noexcept
{
  return error ? error->message.size() : 0;
}
void sirius_error_destroy(sirius_error* error) noexcept { delete error; }
