// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include <sirius/c/error.h>

#include <string>

struct sirius_error {
  std::string message;
};

const char* sirius_error_message(const sirius_error* error) noexcept
{
  return error ? error->message.c_str() : "";
}
size_t sirius_error_message_size(const sirius_error* error) noexcept
{
  return error ? error->message.size() : 0;
}
void sirius_error_destroy(sirius_error* error) noexcept { delete error; }
