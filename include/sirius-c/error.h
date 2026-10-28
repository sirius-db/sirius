// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
/** @file
 * @brief Status codes and owned diagnostics for the public C API. */
#pragma once
#include <sirius-c/export.h>
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/** A fixed-width result code. Zero means success; unknown nonzero codes are failures. */
typedef uint32_t sirius_status;
/** Status values 1–3 are reserved for configuration errors. */
enum {
  SIRIUS_SUCCESS            = 0, /**< The operation succeeded. */
  SIRIUS_ALLOCATION_FAILURE = 4, /**< Allocation failed; no diagnostic is required. */
  SIRIUS_INVALID_ARGUMENT   = 5, /**< A required pointer or argument is invalid. */
  SIRIUS_INTERNAL_ERROR     = 6  /**< An unexpected implementation failure occurred. */
};
/** An owned diagnostic. Release with sirius_error_destroy(), never free().
 * @par Thread safety
 * Ownership may be transferred between threads. Concurrent reads are safe;
 * destruction must not overlap any access to the handle or its borrowed message.
 */
typedef struct sirius_error sirius_error;

/** Borrow a NUL-terminated diagnostic until its handle is destroyed.
 * Returns an empty string for a null handle. The wording is not a stable interface.
 * Messages may contain embedded NUL bytes; sirius_error_message_size() is authoritative.
 * @code{.c}
 * const char *message = sirius_error_message(error);
 * @endcode
 */
SIRIUS_EXPORT const char* sirius_error_message(const sirius_error* error) SIRIUS_C_NOEXCEPT;
/** Return the message length in bytes, excluding its terminator; zero for a null handle.
 * @code{.c}
 * size_t length = sirius_error_message_size(error);
 * @endcode
 */
SIRIUS_EXPORT size_t sirius_error_message_size(const sirius_error* error) SIRIUS_C_NOEXCEPT;
/** Release a diagnostic. A null handle is allowed. A released handle must not be reused.
 * @code{.c}
 * sirius_error_destroy(error);
 * error = NULL;
 * @endcode
 */
SIRIUS_EXPORT void sirius_error_destroy(sirius_error* error) SIRIUS_C_NOEXCEPT;
#ifdef __cplusplus
}
#endif
