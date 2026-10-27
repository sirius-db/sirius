// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include <sirius-c/error.h>
#include <sirius-c/version.h>

// Retain every exported symbol in this compile/link smoke check.
const char* (*volatile error_message)(const sirius_error*) = &sirius_error_message;
size_t (*volatile error_message_size)(const sirius_error*) = &sirius_error_message_size;
void (*volatile error_destroy)(sirius_error*)              = &sirius_error_destroy;
uint32_t (*volatile abi_version)(void)                     = &sirius_abi_version;

int main(void)
{
  return error_message == NULL || error_message_size == NULL || error_destroy == NULL ||
         abi_version == NULL;
}
