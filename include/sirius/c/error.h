// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
/** @file
 * @brief Status codes and owned diagnostics for the public C API. */
#pragma once
#include <sirius/c/export.h>
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/** A fixed-width result code. Zero means success; unknown nonzero codes are failures. */
typedef uint32_t sirius_status;
enum {
  SIRIUS_SUCCESS                = 0, /**< The operation succeeded. */
  SIRIUS_CONFIGURATION_IO       = 1, /**< A configuration file could not be read. */
  SIRIUS_MALFORMED_YAML         = 2, /**< The input is not valid YAML. */
  SIRIUS_INVALID_CONFIGURATION  = 3, /**< A setting is invalid or conflicting. */
  SIRIUS_ALLOCATION_FAILURE     = 4, /**< Allocation failed; no diagnostic is required. */
  SIRIUS_INVALID_ARGUMENT       = 5, /**< A required pointer or argument is invalid. */
  SIRIUS_INTERNAL_ERROR         = 6, /**< An unexpected implementation failure occurred. */
  SIRIUS_CONTEXT_INITIALIZATION = 7  /**< Hardware resolution or engine initialization failed. */
};
/** An owned diagnostic. Release with sirius_error_destroy(), never free(). */
typedef struct sirius_error sirius_error;

/** Borrow a NUL-terminated diagnostic until its handle is destroyed.
 * Returns an empty string for a null handle. The wording is not a stable interface.
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
