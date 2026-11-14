// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
/** @file
 * @brief Own immutable object-store configurations. */
#pragma once
#include <sirius-c/export.h>
#ifdef __cplusplus
extern "C" {
#endif
/** Object-store (S3 API) settings that passed sirius_object_store_config_builder_build():
 * an endpoint, a region, credentials, transport options, a CA bundle path and the URI
 * prefix they apply to. Its values outlive its builder and the buffers they were copied
 * from, and never change. No function returns a field.
 *
 * Building checks the settings themselves. It does not open the CA bundle path, contact
 * the endpoint or consult an I/O backend.
 *
 * Each owning reference must be released exactly once. Borrowed handles must remain
 * alive throughout a call. Do not release a reference concurrently with its use.
 *
 * @par Thread safety
 * Handles may be transferred between threads, including for final release.
 * Retaining and releasing distinct owning references is thread-safe; keep a live
 * reference throughout each call. Never release the same reference concurrently
 * with its use.
 */
typedef struct sirius_object_store_config sirius_object_store_config;
/** Acquire another owning reference without allocating. A null handle is allowed.
 * @code{.c}
 * sirius_object_store_config_retain(config);
 * sirius_object_store_config *copy = config;
 * @endcode
 */
SIRIUS_EXPORT void sirius_object_store_config_retain(sirius_object_store_config* config)
  SIRIUS_C_NOEXCEPT;
/** Release an owning reference; destroy the configuration after its last reference.
 * A null handle is allowed.
 * @code{.c}
 * sirius_object_store_config_release(config);
 * config = NULL;
 * @endcode
 */
SIRIUS_EXPORT void sirius_object_store_config_release(sirius_object_store_config* config)
  SIRIUS_C_NOEXCEPT;
#ifdef __cplusplus
}
#endif
