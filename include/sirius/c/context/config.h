// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
/** @file
 * @brief Own immutable configuration snapshots without accessing hardware. */
#pragma once
#include <sirius/c/export.h>
#ifdef __cplusplus
extern "C" {
#endif
/** A validated configuration snapshot. Its values outlive its builder and YAML file.
 * Hardware availability and capacity are checked when an engine is initialized.
 * Each owning reference must be released exactly once. Borrowed handles must remain
 * alive throughout a call. Do not release a reference concurrently with its use.
 *
 * @par Thread safety
 * Handles may be transferred between threads, including for final release.
 * Immutable settings may be read and built concurrently. Retaining and releasing
 * distinct owning references is thread-safe; keep a live reference throughout
 * each call. Never release the same reference concurrently with its use.
 * Output slots require exclusive access for the duration of a call.
 */
typedef struct sirius_context_config sirius_context_config;
/** Acquire another owning reference without allocating. A null handle is allowed.
 * @code{.c}
 * sirius_context_config_retain(config);
 * sirius_context_config *copy = config;
 * @endcode
 */
SIRIUS_EXPORT void sirius_context_config_retain(sirius_context_config* config) SIRIUS_C_NOEXCEPT;
/** Release an owning reference; destroy the snapshot after its last reference.
 * A null handle is allowed.
 * @code{.c}
 * sirius_context_config_release(config);
 * config = NULL;
 * @endcode
 */
SIRIUS_EXPORT void sirius_context_config_release(sirius_context_config* config) SIRIUS_C_NOEXCEPT;
#ifdef __cplusplus
}
#endif
