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
 */
typedef struct sirius_config sirius_config;
/** Acquire another owning reference without allocating. A null handle is allowed.
 * @code{.c}
 * sirius_config_retain(config);
 * sirius_config *copy = config;
 * @endcode
 */
SIRIUS_EXPORT void sirius_config_retain(sirius_config* config) SIRIUS_C_NOEXCEPT;
/** Release an owning reference; destroy the snapshot after its last reference.
 * A null handle is allowed.
 * @code{.c}
 * sirius_config_release(config);
 * config = NULL;
 * @endcode
 */
SIRIUS_EXPORT void sirius_config_release(sirius_config* config) SIRIUS_C_NOEXCEPT;
#ifdef __cplusplus
}
#endif
