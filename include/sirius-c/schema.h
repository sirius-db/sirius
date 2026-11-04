// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
/** @file
 * @brief Immutable schemas shared by plan construction and inspection. */
#pragma once
#include <sirius-c/error.h>
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/** Fixed-width primitive type identifier. Unknown identifiers are rejected by construction. */
typedef uint32_t sirius_data_type;
enum { SIRIUS_TYPE_BOOLEAN = 1, SIRIUS_TYPE_INT64 = 2 };
/** Borrowed field descriptor. Names are nonempty UTF-8 without NUL and must be unique
 * within a schema by byte equality (case-sensitive). nullable is exactly 0 or 1. Factories copy
 * names; inspection borrows them.
 * @par Thread safety
 * Concurrent reads are allowed; do not mutate a descriptor or its name during a call.
 */
typedef struct sirius_field {
  const char* name;
  size_t name_size;
  sirius_data_type type;
  uint32_t nullable;
} sirius_field;
/** An immutable ordered schema. All handles in this API are immutable and own their metadata.
 * Construction performs no I/O, hardware discovery, planning, or optimization. Successful
 * construction establishes structural and type correctness, not support by a particular engine or
 * device.
 *
 * Factories borrow arguments, copy or retain their contents, and clear required output slots
 * before work. Inputs survive failure. Output slots must not alias inputs or contain unreleased
 * references. Invalid pointers, descriptors, types, scopes, or limits return
 * SIRIUS_INVALID_ARGUMENT; allocation and unexpected failures use the common error codes.
 * Optional diagnostics follow sirius_error's ownership contract. No exceptions cross the ABI.
 *
 * @par Thread safety
 * Handles may be transferred between threads. Concurrent reads and construction are safe.
 * Retain/release distinct owning references on any thread; keep a live reference throughout
 * every call. Never release a reference concurrently with its use. Output slots require
 * exclusive access. Borrowed inspection results live until their owning handle is released. */
typedef struct sirius_schema sirius_schema;
/** Copy a nonempty array of fields into an owned schema. fields and out must not be NULL.
 * @code{.c}
 * sirius_field field = {"id", 2, SIRIUS_TYPE_INT64, 1};
 * sirius_schema *schema = NULL;
 * sirius_status status = sirius_schema_create(&field, 1, &schema, NULL);
 * @endcode
 */
SIRIUS_EXPORT sirius_status sirius_schema_create(const sirius_field* fields,
                                                 size_t count,
                                                 sirius_schema** out,
                                                 sirius_error** error) SIRIUS_C_NOEXCEPT;
/** Return the number of fields; zero for NULL.
 * @code{.c}
 * size_t columns = sirius_schema_field_count(schema);
 * @endcode
 */
SIRIUS_EXPORT size_t sirius_schema_field_count(const sirius_schema* schema) SIRIUS_C_NOEXCEPT;
/** Borrow a field by zero-based position. out is cleared on failure. The returned name
 * is owned by schema; schema and out must not be NULL.
 * @code{.c}
 * sirius_field field = {0};
 * sirius_status status = sirius_schema_field(schema, 0, &field, NULL);
 * @endcode
 */
SIRIUS_EXPORT sirius_status sirius_schema_field(const sirius_schema* schema,
                                                size_t index,
                                                sirius_field* out,
                                                sirius_error** error) SIRIUS_C_NOEXCEPT;
/** Acquire another owning reference without allocating; NULL is allowed.
 * @code{.c}
 * sirius_schema_retain(schema);
 * @endcode
 */
SIRIUS_EXPORT void sirius_schema_retain(sirius_schema* value) SIRIUS_C_NOEXCEPT;
/** Release one owning reference; NULL is allowed.
 * @code{.c}
 * sirius_schema_release(schema);
 * schema = NULL;
 * @endcode
 */
SIRIUS_EXPORT void sirius_schema_release(sirius_schema* value) SIRIUS_C_NOEXCEPT;
#ifdef __cplusplus
}
#endif
