// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
/** @file
 * @brief Typed immutable physical plan construction. */
#pragma once
#include <sirius-c/schema.h>
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/** An immutable physical operator with an explicit output schema.
 * All handles in this API are immutable and own their metadata. Construction performs no
 * I/O, hardware discovery, planning, or optimization. Successful construction establishes
 * structural and type correctness, not support by a particular engine or device.
 *
 * Factories borrow arguments, copy or retain their contents, and clear required output slots
 * before work. Inputs survive failure. Output slots must not alias inputs or contain unreleased
 * references. Invalid pointers, descriptors, types, scopes, or limits return
 * SIRIUS_INVALID_ARGUMENT; allocation and unexpected failures use the common error codes.
 * Optional diagnostics follow sirius_error's ownership contract. No exceptions cross the ABI.
 *
 * The initial subset supports one local Parquet scan followed by filters. Each node retains
 * its input. Nodes are never fused or reordered. Operator-specific properties and additional
 * operators can be introduced separately. Node chains and expression trees are limited to
 * 256 levels to bound traversal and destruction stack use.
 *
 * @par Thread safety
 * Handles may be transferred between threads. Concurrent reads and construction are safe.
 * Retain/release distinct owning references on any thread; keep a live reference throughout
 * every call. Never release a reference concurrently with its use. Output slots require
 * exclusive access. Borrowed inspection results live until their owning handle is released.
 */
typedef struct sirius_plan_node sirius_plan_node;
/** Immutable typed expression. Retains its input scope and child expressions. Literals have
 * no scope. Ownership, thread safety and failure rules are those of sirius_plan_node.
 */
typedef struct sirius_expression sirius_expression;
/** Immutable single-root physical plan. Owns its root and an inspectable description.
 * Ownership, thread safety and failure rules are those of sirius_plan_node.
 */
typedef struct sirius_plan sirius_plan;
/** Fixed-width physical operator kind. Zero is reserved for null-handle inspection. */
typedef uint32_t sirius_plan_node_kind;
enum { SIRIUS_NODE_PARQUET_SCAN = 1, SIRIUS_NODE_FILTER = 2 };
/** Fixed-width comparison selector. Comparisons do not implicitly coerce types. */
typedef uint32_t sirius_comparison;
enum {
  SIRIUS_COMPARISON_EQUAL         = 0,
  SIRIUS_COMPARISON_NOT_EQUAL     = 1,
  SIRIUS_COMPARISON_LESS          = 2,
  SIRIUS_COMPARISON_LESS_EQUAL    = 3,
  SIRIUS_COMPARISON_GREATER       = 4,
  SIRIUS_COMPARISON_GREATER_EQUAL = 5
};
/** Describe one local Parquet file with an explicit schema. path is a nonempty native-byte
 * path without NUL or a URI scheme. The path is literal: glob characters are never expanded.
 * Relative paths use the working directory at future preparation time.
 * No file is opened or schema inferred; preparation must bind the source
 * and check its schema and engine support. schema and out must not be NULL.
 * @code{.c}
 * sirius_plan_node *scan = NULL;
 * sirius_status status = sirius_plan_scan_parquet("data.parquet", 12, schema, &scan, NULL);
 * @endcode
 */
SIRIUS_EXPORT sirius_status sirius_plan_scan_parquet(const char* path,
                                                     size_t size,
                                                     const sirius_schema* schema,
                                                     sirius_plan_node** out,
                                                     sirius_error** error) SIRIUS_C_NOEXCEPT;
/** Create a reference scoped to input at a zero-based column position. Its type and
 * nullability come from the schema. input and out must not be NULL.
 * @code{.c}
 * sirius_expression *column = NULL;
 * sirius_status status = sirius_expression_column(scan, 0, &column, NULL);
 * @endcode
 */
SIRIUS_EXPORT sirius_status sirius_expression_column(const sirius_plan_node* input,
                                                     size_t column,
                                                     sirius_expression** out,
                                                     sirius_error** error) SIRIUS_C_NOEXCEPT;
/** Create a non-null Int64 literal with no input scope. out must not be NULL.
 * @code{.c}
 * sirius_expression *ten = NULL;
 * sirius_status status = sirius_expression_literal_int64(10, &ten, NULL);
 * @endcode
 */
SIRIUS_EXPORT sirius_status sirius_expression_literal_int64(int64_t value,
                                                            sirius_expression** out,
                                                            sirius_error** error) SIRIUS_C_NOEXCEPT;
/** Create a non-null Boolean literal with no input scope. value is exactly 0 or 1;
 * out must not be NULL.
 * @code{.c}
 * sirius_expression *yes = NULL;
 * sirius_status status = sirius_expression_literal_boolean(1, &yes, NULL);
 * @endcode
 */
SIRIUS_EXPORT sirius_status sirius_expression_literal_boolean(
  uint32_t value, sirius_expression** out, sirius_error** error) SIRIUS_C_NOEXCEPT;
/** Compare two expressions with identical types and compatible scopes. An unscoped
 * expression is compatible with either scope; otherwise both must reference the same input node.
 * Int64 supports all comparisons; Boolean supports equality and inequality. The result is
 * Boolean, nullable if either operand is nullable. SQL NULL comparisons produce NULL.
 * left, right, and out must not be NULL.
 * @code{.c}
 * sirius_expression *predicate = NULL;
 * sirius_status status = sirius_expression_compare(column, SIRIUS_COMPARISON_GREATER, ten,
 * &predicate, NULL);
 * @endcode
 */
SIRIUS_EXPORT sirius_status sirius_expression_compare(const sirius_expression* left,
                                                      sirius_comparison operation,
                                                      const sirius_expression* right,
                                                      sirius_expression** out,
                                                      sirius_error** error) SIRIUS_C_NOEXCEPT;
/** Create an explicit filter preserving the input schema. Only TRUE rows survive;
 * FALSE and NULL rows are discarded. predicate must be Boolean and either unscoped or
 * scoped to input. input, predicate, and out must not be NULL.
 * @code{.c}
 * sirius_plan_node *filtered = NULL;
 * sirius_status status = sirius_plan_filter(scan, predicate, &filtered, NULL);
 * @endcode
 */
SIRIUS_EXPORT sirius_status sirius_plan_filter(const sirius_plan_node* input,
                                               const sirius_expression* predicate,
                                               sirius_plan_node** out,
                                               sirius_error** error) SIRIUS_C_NOEXCEPT;
/** Return the expression result type; zero for NULL.
 * @code{.c}
 * sirius_data_type type = sirius_expression_type(predicate);
 * @endcode
 */
SIRIUS_EXPORT sirius_data_type sirius_expression_type(const sirius_expression* expression)
  SIRIUS_C_NOEXCEPT;
/** Return 1 when the result may be NULL, otherwise 0. Returns 0 for a NULL handle.
 * @code{.c}
 * uint32_t nullable = sirius_expression_nullable(predicate);
 * @endcode
 */
SIRIUS_EXPORT uint32_t sirius_expression_nullable(const sirius_expression* expression)
  SIRIUS_C_NOEXCEPT;
/** Return the physical node kind; zero for NULL.
 * @code{.c}
 * sirius_plan_node_kind kind = sirius_plan_node_get_kind(filtered);
 * @endcode
 */
SIRIUS_EXPORT sirius_plan_node_kind sirius_plan_node_get_kind(const sirius_plan_node* node)
  SIRIUS_C_NOEXCEPT;
/** Borrow the input of a filter; NULL for a scan or NULL node. The returned node is
 * owned by node. Retain it before releasing node if it is needed independently.
 * @code{.c}
 * sirius_plan_node *input = sirius_plan_node_input(filtered);
 * @endcode
 */
SIRIUS_EXPORT sirius_plan_node* sirius_plan_node_input(const sirius_plan_node* node)
  SIRIUS_C_NOEXCEPT;
/** Borrow the immutable output schema; NULL for a NULL node. The returned schema is
 * owned by node. Retain it before releasing node if it is needed independently.
 * @code{.c}
 * sirius_schema *schema = sirius_plan_node_schema(scan);
 * @endcode
 */
SIRIUS_EXPORT sirius_schema* sirius_plan_node_schema(const sirius_plan_node* node)
  SIRIUS_C_NOEXCEPT;
/** Create a single-root physical plan without binding any runtime resources.
 * root and out must not be NULL. The result retains root independently.
 * @code{.c}
 * sirius_plan *plan = NULL;
 * sirius_status status = sirius_plan_from_root(filtered, &plan, NULL);
 * @endcode
 */
SIRIUS_EXPORT sirius_status sirius_plan_from_root(const sirius_plan_node* root,
                                                  sirius_plan** out,
                                                  sirius_error** error) SIRIUS_C_NOEXCEPT;
/** Borrow the immutable root; NULL for a NULL plan. Retain it before releasing plan
 * if it is needed independently.
 * @code{.c}
 * sirius_plan_node *root = sirius_plan_root(plan);
 * @endcode
 */
SIRIUS_EXPORT sirius_plan_node* sirius_plan_root(const sirius_plan* plan) SIRIUS_C_NOEXCEPT;
/** Borrow a NUL-terminated, human-readable description until plan is released.
 * Returns an empty string for NULL. Names and paths are escaped. Formatting is diagnostic,
 * not a serialization format or stable machine-readable interface.
 * @code{.c}
 * const char *description = sirius_plan_describe(plan);
 * @endcode
 */
SIRIUS_EXPORT const char* sirius_plan_describe(const sirius_plan* plan) SIRIUS_C_NOEXCEPT;
/** Acquire another owning reference without allocating; NULL is allowed.
 * @code{.c}
 * sirius_expression_retain(predicate);
 * @endcode
 */
SIRIUS_EXPORT void sirius_expression_retain(sirius_expression* value) SIRIUS_C_NOEXCEPT;
/** Release one owning reference; NULL is allowed.
 * @code{.c}
 * sirius_expression_release(predicate);
 * predicate = NULL;
 * @endcode
 */
SIRIUS_EXPORT void sirius_expression_release(sirius_expression* value) SIRIUS_C_NOEXCEPT;
/** Acquire another owning reference without allocating; NULL is allowed.
 * @code{.c}
 * sirius_plan_node_retain(scan);
 * @endcode
 */
SIRIUS_EXPORT void sirius_plan_node_retain(sirius_plan_node* value) SIRIUS_C_NOEXCEPT;
/** Release one owning reference; NULL is allowed.
 * @code{.c}
 * sirius_plan_node_release(scan);
 * scan = NULL;
 * @endcode
 */
SIRIUS_EXPORT void sirius_plan_node_release(sirius_plan_node* value) SIRIUS_C_NOEXCEPT;
/** Acquire another owning reference without allocating; NULL is allowed.
 * @code{.c}
 * sirius_plan_retain(plan);
 * @endcode
 */
SIRIUS_EXPORT void sirius_plan_retain(sirius_plan* value) SIRIUS_C_NOEXCEPT;
/** Release one owning reference; NULL is allowed.
 * @code{.c}
 * sirius_plan_release(plan);
 * plan = NULL;
 * @endcode
 */
SIRIUS_EXPORT void sirius_plan_release(sirius_plan* value) SIRIUS_C_NOEXCEPT;
#ifdef __cplusplus
}
#endif
