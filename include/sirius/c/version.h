// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
/** @file
 * @brief Identify the C ABI revision of the headers and linked library. */
#pragma once
#include <sirius/c/export.h>
#include <stdint.h>
/** Expected ABI revision. Incompatible C ABI changes increment this value. */
#define SIRIUS_ABI_VERSION UINT32_C(1)
#ifdef __cplusplus
extern "C" {
#endif
/** Return the linked library's ABI revision. Matching revisions do not imply
 * that an older library provides functions added by newer headers.
 * @code{.c}
 * if (sirius_abi_version() != SIRIUS_ABI_VERSION) { return 1; }
 * @endcode
 */
SIRIUS_EXPORT uint32_t sirius_abi_version(void) SIRIUS_C_NOEXCEPT;
#ifdef __cplusplus
}
#endif
