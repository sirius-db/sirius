// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
/** @file
 * @brief Assemble and validate immutable object-store configurations. */
#pragma once
#include <sirius-c/error.h>
#include <sirius-c/object_store/config.h>
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/** The wire transport of an object store. A fixed-width selector: only the constants
 * below are accepted. */
typedef uint32_t sirius_object_store_transport;
enum {
  SIRIUS_OBJECT_STORE_TRANSPORT_AUTO = 0, /**< No preference. */
  SIRIUS_OBJECT_STORE_TRANSPORT_HTTP = 1, /**< HTTP or HTTPS. */
  SIRIUS_OBJECT_STORE_TRANSPORT_RDMA = 2  /**< S3 over RDMA. */
};
/** Where SigV4 authentication travels. A fixed-width selector: only the constants
 * below are accepted. */
typedef uint32_t sirius_object_store_signing_mode;
enum {
  SIRIUS_OBJECT_STORE_SIGNING_PRESIGNED = 0, /**< In the URL query. */
  SIRIUS_OBJECT_STORE_SIGNING_HEADER    = 1  /**< In the `Authorization` header. */
};
/** An immutable, possibly incomplete set of object-store settings.
 * A new builder has empty text fields, SIRIUS_OBJECT_STORE_TRANSPORT_AUTO and
 * SIRIUS_OBJECT_STORE_SIGNING_PRESIGNED. Each `with` function returns a new builder
 * with one field replaced; the input is unchanged and stays usable. No function
 * returns a field.
 *
 * A text field is given as size bytes at data. The bytes are copied before the call
 * returns and kept as given: no terminator is needed, nothing is trimmed, and embedded
 * NUL bytes are kept. A size of zero means empty, whether or not data is NULL; NULL with
 * a nonzero size is SIRIUS_INVALID_ARGUMENT. Text is checked by
 * sirius_object_store_config_builder_build(), not when it is supplied.
 *
 * Release every owning reference exactly once. Borrowed handles must remain alive
 * throughout a call. Output slots must not contain unreleased owning references: a
 * slot is overwritten, not released. Do not pass the address of the variable that holds
 * the input builder; derive into another variable, then release the input.
 *
 * Fallible functions clear their output slots before work begins. On failure the slot
 * stays NULL and the input builder is unchanged. An optional out_error receives an owned
 * diagnostic on failure, or NULL if none is available. A diagnostic never contains a
 * field's value; a diagnostic for SIRIUS_INVALID_CONFIGURATION names the field and the
 * reason. Diagnostic allocation is best effort and does not change the status code.
 * No C++ exception crosses this interface.
 *
 * @par Thread safety
 * Handles may be transferred between threads, including for final release.
 * Immutable settings may be derived from and built concurrently. Retaining and releasing
 * distinct owning references is thread-safe; keep a live reference throughout
 * each call. Never release the same reference concurrently with its use.
 * Output slots require exclusive access for the duration of a call.
 */
typedef struct sirius_object_store_config_builder sirius_object_store_config_builder;
/** Create a builder with no settings. out_builder must not be NULL.
 * @code{.c}
 * sirius_object_store_config_builder *builder = NULL;
 * sirius_status status = sirius_object_store_config_builder_create(&builder, NULL);
 * if (status == SIRIUS_SUCCESS) { sirius_object_store_config_builder_release(builder); }
 * @endcode
 */
SIRIUS_EXPORT sirius_status sirius_object_store_config_builder_create(
  sirius_object_store_config_builder** out_builder, sirius_error** out_error) SIRIUS_C_NOEXCEPT;
/** Acquire another owning reference without allocating. A null handle is allowed.
 * @code{.c}
 * sirius_object_store_config_builder_retain(builder);
 * sirius_object_store_config_builder *copy = builder;
 * @endcode
 */
SIRIUS_EXPORT void sirius_object_store_config_builder_retain(
  sirius_object_store_config_builder* builder) SIRIUS_C_NOEXCEPT;
/** Release an owning reference. A null handle is allowed.
 * @code{.c}
 * sirius_object_store_config_builder_release(builder);
 * builder = NULL;
 * @endcode
 */
SIRIUS_EXPORT void sirius_object_store_config_builder_release(
  sirius_object_store_config_builder* builder) SIRIUS_C_NOEXCEPT;
/** Derive a builder with the service endpoint replaced.
 * builder and out_builder must not be NULL. Required at build, where it must be
 * `http://host[:port]` or `https://host[:port]`: no path, query, fragment or user-info,
 * no spaces or control bytes, and a port, if given, in 1..65535.
 * @code{.c}
 * sirius_object_store_config_builder *next = NULL;
 * sirius_status status = sirius_object_store_config_builder_with_endpoint(
 *   builder, "https://s3.example.com", sizeof("https://s3.example.com") - 1, &next, NULL);
 * if (status == SIRIUS_SUCCESS) {
 *   sirius_object_store_config_builder_release(builder);
 *   builder = next;
 * }
 * @endcode
 */
SIRIUS_EXPORT sirius_status
sirius_object_store_config_builder_with_endpoint(const sirius_object_store_config_builder* builder,
                                                 const char* data,
                                                 size_t size,
                                                 sirius_object_store_config_builder** out_builder,
                                                 sirius_error** out_error) SIRIUS_C_NOEXCEPT;
/** Derive a builder with the SigV4 signing region replaced.
 * builder and out_builder must not be NULL. Required at build.
 * @code{.c}
 * sirius_object_store_config_builder *next = NULL;
 * sirius_status status = sirius_object_store_config_builder_with_region(
 *   builder, "us-east-1", sizeof("us-east-1") - 1, &next, NULL);
 * @endcode
 */
SIRIUS_EXPORT sirius_status
sirius_object_store_config_builder_with_region(const sirius_object_store_config_builder* builder,
                                               const char* data,
                                               size_t size,
                                               sirius_object_store_config_builder** out_builder,
                                               sirius_error** out_error) SIRIUS_C_NOEXCEPT;
/** Derive a builder with the access key replaced.
 * builder and out_builder must not be NULL. Required at build.
 * @code{.c}
 * sirius_object_store_config_builder *next = NULL;
 * sirius_status status = sirius_object_store_config_builder_with_access_key(
 *   builder, access_key, access_key_size, &next, NULL);
 * @endcode
 */
SIRIUS_EXPORT sirius_status sirius_object_store_config_builder_with_access_key(
  const sirius_object_store_config_builder* builder,
  const char* data,
  size_t size,
  sirius_object_store_config_builder** out_builder,
  sirius_error** out_error) SIRIUS_C_NOEXCEPT;
/** Derive a builder with the secret key replaced.
 * builder and out_builder must not be NULL. Required at build.
 * @code{.c}
 * sirius_object_store_config_builder *next = NULL;
 * sirius_status status = sirius_object_store_config_builder_with_secret_key(
 *   builder, secret_key, secret_key_size, &next, NULL);
 * @endcode
 */
SIRIUS_EXPORT sirius_status sirius_object_store_config_builder_with_secret_key(
  const sirius_object_store_config_builder* builder,
  const char* data,
  size_t size,
  sirius_object_store_config_builder** out_builder,
  sirius_error** out_error) SIRIUS_C_NOEXCEPT;
/** Derive a builder with the session token of temporary credentials replaced.
 * builder and out_builder must not be NULL. Empty for long-lived keys.
 * @code{.c}
 * sirius_object_store_config_builder *next = NULL;
 * sirius_status status = sirius_object_store_config_builder_with_session_token(
 *   builder, session_token, session_token_size, &next, NULL);
 * @endcode
 */
SIRIUS_EXPORT sirius_status sirius_object_store_config_builder_with_session_token(
  const sirius_object_store_config_builder* builder,
  const char* data,
  size_t size,
  sirius_object_store_config_builder** out_builder,
  sirius_error** out_error) SIRIUS_C_NOEXCEPT;
/** Derive a builder with the path of the PEM CA bundle replaced.
 * builder and out_builder must not be NULL. The bundle verifies the endpoint's TLS
 * certificate; empty selects the system trust store. Build does not open the path.
 * @code{.c}
 * sirius_object_store_config_builder *next = NULL;
 * sirius_status status = sirius_object_store_config_builder_with_ca_bundle_path(
 *   builder, "/etc/ssl/store.pem", sizeof("/etc/ssl/store.pem") - 1, &next, NULL);
 * @endcode
 */
SIRIUS_EXPORT sirius_status sirius_object_store_config_builder_with_ca_bundle_path(
  const sirius_object_store_config_builder* builder,
  const char* data,
  size_t size,
  sirius_object_store_config_builder** out_builder,
  sirius_error** out_error) SIRIUS_C_NOEXCEPT;
/** Derive a builder with the URI prefix the settings apply to replaced.
 * builder and out_builder must not be NULL. At build it must be empty, `s3://bucket/` or
 * `s3://bucket/prefix/`: a bucket, a trailing slash, no query or fragment, and no segment
 * an object key cannot match byte for byte (empty, `.`, `..`, or containing `%`).
 * @code{.c}
 * sirius_object_store_config_builder *next = NULL;
 * sirius_status status = sirius_object_store_config_builder_with_uri_prefix(
 *   builder, "s3://bucket/data/", sizeof("s3://bucket/data/") - 1, &next, NULL);
 * @endcode
 */
SIRIUS_EXPORT sirius_status sirius_object_store_config_builder_with_uri_prefix(
  const sirius_object_store_config_builder* builder,
  const char* data,
  size_t size,
  sirius_object_store_config_builder** out_builder,
  sirius_error** out_error) SIRIUS_C_NOEXCEPT;
/** Derive a builder with the transport replaced.
 * builder and out_builder must not be NULL. A value that is not a declared constant
 * is SIRIUS_INVALID_ARGUMENT. Whether an I/O backend serves the transport is not checked.
 * @code{.c}
 * sirius_object_store_config_builder *next = NULL;
 * sirius_status status = sirius_object_store_config_builder_with_transport(
 *   builder, SIRIUS_OBJECT_STORE_TRANSPORT_HTTP, &next, NULL);
 * @endcode
 */
SIRIUS_EXPORT sirius_status
sirius_object_store_config_builder_with_transport(const sirius_object_store_config_builder* builder,
                                                  sirius_object_store_transport transport,
                                                  sirius_object_store_config_builder** out_builder,
                                                  sirius_error** out_error) SIRIUS_C_NOEXCEPT;
/** Derive a builder with the signing mode replaced.
 * builder and out_builder must not be NULL. A value that is not a declared constant
 * is SIRIUS_INVALID_ARGUMENT. Whether an I/O backend honors the mode is not checked.
 * @code{.c}
 * sirius_object_store_config_builder *next = NULL;
 * sirius_status status = sirius_object_store_config_builder_with_signing_mode(
 *   builder, SIRIUS_OBJECT_STORE_SIGNING_HEADER, &next, NULL);
 * @endcode
 */
SIRIUS_EXPORT sirius_status sirius_object_store_config_builder_with_signing_mode(
  const sirius_object_store_config_builder* builder,
  sirius_object_store_signing_mode signing_mode,
  sirius_object_store_config_builder** out_builder,
  sirius_error** out_error) SIRIUS_C_NOEXCEPT;
/** Check the settings and produce a configuration without reading files, contacting
 * the endpoint or consulting an I/O backend. builder and out_config must not be NULL.
 * The endpoint, region, access key, secret key and URI prefix are checked in that order;
 * the first failure is SIRIUS_INVALID_CONFIGURATION. The builder is not consumed, and
 * the configuration outlives it.
 * @code{.c}
 * sirius_object_store_config *config = NULL;
 * sirius_error *error = NULL;
 * sirius_status status = sirius_object_store_config_builder_build(builder, &config, &error);
 * sirius_error_destroy(error);
 * sirius_object_store_config_release(config);
 * @endcode
 */
SIRIUS_EXPORT sirius_status
sirius_object_store_config_builder_build(const sirius_object_store_config_builder* builder,
                                         sirius_object_store_config** out_config,
                                         sirius_error** out_error) SIRIUS_C_NOEXCEPT;
#ifdef __cplusplus
}
#endif
