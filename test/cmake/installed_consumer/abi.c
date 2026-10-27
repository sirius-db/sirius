// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include <sirius/c/error.h>
#include <sirius/c/version.h>

int main(void)
{
  sirius_status status = SIRIUS_SUCCESS;
  const char* message  = sirius_error_message(NULL);
  if (message == NULL || message[0] != '\0' || sirius_error_message_size(NULL) != 0) { return 1; }
  sirius_error_destroy(NULL);
  return status == 0 && sirius_abi_version() == SIRIUS_ABI_VERSION ? 0 : 1;
}
