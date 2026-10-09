// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
/** @file
 * @brief Identify the C ABI revision of the headers and linked library. */
#pragma once
#include <sirius/c/export.h>
#include <stdint.h>
/** Reserved for future ABI versioning. Zero denotes an unversioned development API;
 * matching values do not guarantee compatibility. */
#define SIRIUS_ABI_VERSION UINT32_C(0)
#ifdef __cplusplus
extern "C" {
#endif
/** Return the linked library's ABI revision, currently zero (unversioned).
 * Use matching headers and library releases until ABI versioning is introduced.
 * @code{.c}
 * if (sirius_abi_version() != SIRIUS_ABI_VERSION) { return 1; }
 * @endcode
 */
SIRIUS_EXPORT uint32_t sirius_abi_version(void) SIRIUS_C_NOEXCEPT;
#ifdef __cplusplus
}
#endif
