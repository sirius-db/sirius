// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
/** @file
 * @brief Create and own an initialized engine through the C ABI.
 */
#pragma once
#include <sirius/c/context/config.h>
#include <sirius/c/error.h>
#ifdef __cplusplus
extern "C" {
#endif
/** Unique ownership of an initialized engine.
 * Only one active Sirius engine context per process is supported, including
 * other Sirius integrations. This restriction is not enforced: callers must
 * ensure lifetimes do not overlap. Multiple configurations may coexist.
 * Do not destroy a context concurrently with its use. Forking with an active
 * context is unsupported.
 *
 * CUDA/NVTX initialization persists even after failed creation; NVTX injection
 * settings must remain unchanged for the process lifetime. Changing
 * sirius.executor.downgrade.copy_chunk_bytes after successful creation is unsupported.
 */
typedef struct sirius_context sirius_context;
/** Resolve configuration against hardware and create an initialized engine.
 * config and out_context must not be NULL. The configuration need not outlive the context.
 * out_context is cleared first and owns a context only on success. Output slots
 * must not contain unreleased handles. Optional out_error receives an owned,
 * best-effort diagnostic, or NULL. Status remains available without a diagnostic.
 * Returns SIRIUS_CONTEXT_INITIALIZATION, SIRIUS_ALLOCATION_FAILURE, or
 * SIRIUS_INVALID_ARGUMENT on failure. No C++ exception crosses the interface.
 * An unrecoverable failure to stop workers or destroy resources during rollback
 * terminates the process.
 * @code{.c}
 * sirius_context *context = NULL;
 * sirius_error *error = NULL;
 * sirius_status status = sirius_context_create(config, &context, &error);
 * if (status == SIRIUS_SUCCESS) { sirius_context_destroy(context); }
 * sirius_error_destroy(error);
 * @endcode
 */
SIRIUS_EXPORT sirius_status sirius_context_create(const sirius_context_config* config,
                                                  sirius_context** out_context,
                                                  sirius_error** out_error) SIRIUS_C_NOEXCEPT;
/** Destroy an engine and release its resources. A null handle is allowed.
 * An unrecoverable failure to stop workers or destroy resources terminates the process.
 * The handle must not be used or destroyed again after this call.
 * @code{.c}
 * sirius_context_destroy(context);
 * context = NULL;
 * @endcode
 */
SIRIUS_EXPORT void sirius_context_destroy(sirius_context* context) SIRIUS_C_NOEXCEPT;
#ifdef __cplusplus
}
#endif
