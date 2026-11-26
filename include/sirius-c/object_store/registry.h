// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
/** @file
 * @brief Build immutable registries that map URI prefixes to object-store configurations. */
#pragma once
#include <sirius-c/error.h>
#include <sirius-c/object_store/config.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
/** An immutable list of object stores. A store is a URI prefix and the configuration that
 * serves the URIs under it. Every registry is complete: the empty registry means that no
 * store is configured. sirius_object_store_registry_with_store() returns a new registry
 * with one store added or replaced; the input is unchanged and stays usable. No function
 * returns a store, a prefix or a count.
 *
 * A registry reads no file and consults no I/O backend. Whether a CA bundle can be read
 * and whether a transport is available is checked where the registry is used.
 *
 * Release every owning reference exactly once. Borrowed handles must remain alive
 * throughout a call. Output slots must not contain unreleased owning references: a slot
 * is overwritten, not released. Do not pass the address of the variable that holds the
 * input registry; derive into another variable, then release the input. A text range must
 * not overlap an output slot: the slots are cleared before the range is read.
 *
 * Fallible functions clear their output slots before work begins. On failure the slot
 * stays NULL and the input registry is unchanged. An optional out_error receives an owned
 * diagnostic on failure, or NULL if none is available. A diagnostic never contains a
 * prefix or a field's value; a diagnostic for SIRIUS_INVALID_CONFIGURATION names
 * `uri_prefix` and the reason. Diagnostic allocation is best effort and does not change
 * the status code. No C++ exception crosses this interface.
 *
 * @par Thread safety
 * Registries may be shared and derived from concurrently. Retain and release distinct
 * owning references from any thread; keep a reference alive throughout every call. Output
 * slots require exclusive access.
 */
typedef struct sirius_object_store_registry sirius_object_store_registry;
/** Create the empty registry. out_registry must not be NULL.
 * @code{.c}
 * sirius_object_store_registry *registry = NULL;
 * sirius_status status = sirius_object_store_registry_create(&registry, NULL);
 * @endcode
 */
SIRIUS_EXPORT sirius_status sirius_object_store_registry_create(
  sirius_object_store_registry** out_registry, sirius_error** out_error) SIRIUS_C_NOEXCEPT;
/** Derive a registry in which config is the store for uri_prefix.
 * registry, config and out_registry must not be NULL. uri_prefix is uri_prefix_size bytes,
 * copied before the call returns. A size of zero is the empty prefix. A NULL uri_prefix with a
 * nonzero size, or a range that contains a NUL byte, is SIRIUS_INVALID_ARGUMENT.
 *
 * A nonempty prefix must have the form `s3://bucket/` or `s3://bucket/prefix/`. Write the
 * `s3://` scheme in lowercase. The bucket must consist of one or more ASCII letters, digits,
 * `.`, `_` or `-` and must not be `.` or `..`. The prefix must not contain `?` or `#`. Each
 * segment after the bucket must be nonempty and followed by `/`. It must not be `.` or `..` or
 * contain a `%` byte. A prefix that violates these rules gives `SIRIUS_INVALID_CONFIGURATION`.
 *
 * A store the registry already holds for the same prefix bytes is replaced; the other
 * stores are kept. The registry keeps its own copy of the configuration's settings: one
 * configuration can serve several prefixes, and it may be released after the call.
 *
 * With at least one store registered, the matcher refuses an empty key after the bucket/key
 * separator, a trailing slash, an empty, `.` or `..` key segment, or any literal `%` byte in the
 * key. This applies even when the empty prefix is the only entry. No key bytes are decoded or
 * rewritten. Otherwise the longest matching prefix wins; only the lookup URI's scheme is
 * compared without regard to ASCII case. The empty prefix, if registered, is a fallback only for
 * URIs beginning with `s3://` in either ASCII case.
 * @code{.c}
 * sirius_object_store_registry *next = NULL;
 * sirius_status status = sirius_object_store_registry_with_store(
 *   registry, "s3://bucket/data/", sizeof("s3://bucket/data/") - 1, config, &next, NULL);
 * if (status == SIRIUS_SUCCESS) {
 *   sirius_object_store_registry_release(registry);
 *   registry = next;
 * }
 * @endcode
 */
SIRIUS_EXPORT sirius_status
sirius_object_store_registry_with_store(const sirius_object_store_registry* registry,
                                        const char* uri_prefix,
                                        size_t uri_prefix_size,
                                        const sirius_object_store_config* config,
                                        sirius_object_store_registry** out_registry,
                                        sirius_error** out_error) SIRIUS_C_NOEXCEPT;
/** Acquire another owning reference without allocating; NULL is allowed.
 * @code{.c}
 * sirius_object_store_registry_retain(registry);
 * @endcode
 */
SIRIUS_EXPORT void sirius_object_store_registry_retain(sirius_object_store_registry* registry)
  SIRIUS_C_NOEXCEPT;
/** Release one owning reference; NULL is allowed.
 * @code{.c}
 * sirius_object_store_registry_release(registry);
 * registry = NULL;
 * @endcode
 */
SIRIUS_EXPORT void sirius_object_store_registry_release(sirius_object_store_registry* registry)
  SIRIUS_C_NOEXCEPT;
#ifdef __cplusplus
}
#endif
