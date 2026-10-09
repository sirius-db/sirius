// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
/** @file
 * @brief Build immutable configurations from defaults or YAML. */
#pragma once
#include <sirius/c/context/config.h>
#include <sirius/c/error.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
/** An immutable set of builder settings. Retained references share settings.
 * Loading and building do not discover hardware or initialize CUDA or telemetry.
 * Release every owning reference exactly once. Borrowed handles must remain alive
 * throughout a call. Output slots must not contain unreleased owning references.
 *
 * Fallible functions clear their output slots before work begins. An optional
 * out_error receives an owned diagnostic on failure, or NULL if none is available.
 * Diagnostic allocation is best effort and does not change the status code.
 * No C++ exception crosses this interface.
 *
 * @par Thread safety
 * Handles may be transferred between threads, including for final release.
 * Immutable settings may be read and built concurrently. Retaining and releasing
 * distinct owning references is thread-safe; keep a live reference throughout
 * each call. Never release the same reference concurrently with its use.
 * Output slots require exclusive access for the duration of a call.
 */
typedef struct sirius_context_config_builder sirius_context_config_builder;
/** Create settings from built-in defaults. out_builder must not be NULL.
 * @code{.c}
 * sirius_context_config_builder *builder = NULL;
 * sirius_status status = sirius_context_config_builder_create(&builder, NULL);
 * if (status == SIRIUS_SUCCESS) { sirius_context_config_builder_release(builder); }
 * @endcode
 */
SIRIUS_EXPORT sirius_status sirius_context_config_builder_create(
  sirius_context_config_builder** out_builder, sirius_error** out_error) SIRIUS_C_NOEXCEPT;
/** Read and validate one YAML file, retaining its settings independently of later file changes.
 * path is a non-NULL array of path_size bytes; embedded NUL bytes are rejected.
 * Unix path bytes are preserved. Relative paths use the working directory.
 * out_builder must not be NULL. No automatic configuration-file search is performed.
 * Uses the [Sirius YAML
 * schema](https://github.com/sirius-db/sirius/blob/main/docs/super-sirius/configuration.md).
 * Returns IO, YAML, configuration, argument, allocation, or unexpected-error status.
 * @code{.c}
 * sirius_context_config_builder *builder = NULL;
 * sirius_error *error = NULL;
 * sirius_status status = sirius_context_config_builder_from_yaml(
 *   "sirius.yaml", sizeof("sirius.yaml") - 1, &builder, &error);
 * sirius_error_destroy(error);
 * sirius_context_config_builder_release(builder);
 * @endcode
 */
SIRIUS_EXPORT sirius_status
sirius_context_config_builder_from_yaml(const char* path,
                                        size_t path_size,
                                        sirius_context_config_builder** out_builder,
                                        sirius_error** out_error) SIRIUS_C_NOEXCEPT;
/** Acquire another owning reference without allocating. A null handle is allowed.
 * @code{.c}
 * sirius_context_config_builder_retain(builder);
 * sirius_context_config_builder *copy = builder;
 * @endcode
 */
SIRIUS_EXPORT void sirius_context_config_builder_retain(sirius_context_config_builder* builder)
  SIRIUS_C_NOEXCEPT;
/** Release an owning reference. A null handle is allowed.
 * @code{.c}
 * sirius_context_config_builder_release(builder);
 * builder = NULL;
 * @endcode
 */
SIRIUS_EXPORT void sirius_context_config_builder_release(sirius_context_config_builder* builder)
  SIRIUS_C_NOEXCEPT;
/** Produce a snapshot without reading files or accessing hardware.
 * builder and out_config must not be NULL. The snapshot outlives the builder.
 * @code{.c}
 * sirius_context_config *config = NULL;
 * sirius_status status = sirius_context_config_builder_build(builder, &config, NULL);
 * if (status == SIRIUS_SUCCESS) { sirius_context_config_release(config); }
 * @endcode
 */
SIRIUS_EXPORT sirius_status
sirius_context_config_builder_build(const sirius_context_config_builder* builder,
                                    sirius_context_config** out_config,
                                    sirius_error** out_error) SIRIUS_C_NOEXCEPT;
#ifdef __cplusplus
}
#endif
