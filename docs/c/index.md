# Sirius C API

The public C API provides a C ABI for embedding Sirius. Include the headers under
`sirius-c/` and link the Sirius library. No C++ types cross this interface.

## Headers and linking

- `<sirius-c/export.h>` defines visibility and C/C++ declaration helpers.
- `<sirius-c/context/config_builder.h>` builds immutable context configurations.
- `<sirius-c/context/context.h>` creates and owns an engine context.
- `<sirius-c/schema.h>` constructs and inspects immutable typed schemas.
- `<sirius-c/plan/plan.h>` constructs and inspects immutable physical plans.
- `<sirius-c/error.h>` defines fixed-width status codes and owned diagnostics.
- `<sirius-c/version.h>` declares the library's ABI revision.

With CMake, link `sirius::sirius`, or `sirius::sirius_static` when the static package
is enabled. Static consumers must enable CXX in their CMake project and link with
the C++ driver to include the required C++ runtime support. The headers use
C99-compatible declarations.

```c
#include <sirius-c/version.h>

int main(void)
{
    return sirius_abi_version() == SIRIUS_ABI_VERSION ? 0 : 1;
}
```

## Configuration

Create immutable configuration snapshots without accessing GPUs:

```c
#include <sirius-c/context/config_builder.h>
#include <stdio.h>

int main(void)
{
    sirius_context_config_builder *builder = NULL;
    sirius_context_config *config = NULL;
    sirius_error *error = NULL;
    sirius_status status = sirius_context_config_builder_create(&builder, &error);
    if (status == SIRIUS_SUCCESS) {
        status = sirius_context_config_builder_build(builder, &config, &error);
    }
    if (status != SIRIUS_SUCCESS) {
        fprintf(stderr, "Sirius error %u: %s\n", (unsigned)status,
                sirius_error_message(error));
    }
    sirius_error_destroy(error);
    sirius_context_config_release(config);
    sirius_context_config_builder_release(builder);
    return status == SIRIUS_SUCCESS ? 0 : 1;
}
```

Use sirius_context_config_builder_from_yaml() to load the
[YAML configuration schema](https://github.com/sirius-db/sirius/blob/main/docs/super-sirius/configuration.md).
The file is read once; builders and built configurations retain its settings independently.
Hardware availability and capacity are checked when initializing an engine.

## Engine context

Use sirius_context_create() with a built configuration and release the result
with sirius_context_destroy(). Hardware discovery and resource allocation happen
at creation. Only one active engine context per process is supported; callers
must enforce that restriction. See sirius_context for process-wide settings and
lifetime constraints.

## Physical plan construction

Construct plans without a context, GPU, or file access. The initial subset supports
Boolean and Int64 fields, a local Parquet scan with an explicit schema, typed column
references and literals, comparisons, and filters. Each factory returns a complete
immutable object or an error. Schema names and column types are checked during
construction. Field names are case-sensitive; file schemas and engine support
belong to future preparation.

```text
ParquetScan(schema, path) → Filter(predicate) → Plan
predicate = comparison(column, literal)
```

Use sirius_schema_create(), sirius_plan_scan_parquet(), sirius_expression_column(),
sirius_expression_literal_int64(), sirius_expression_compare(), sirius_plan_filter(),
and sirius_plan_from_root() to build a scan with an integer predicate. See the
[complete C example](https://github.com/sirius-db/sirius/blob/main/test/cmake/installed_consumer/plan.c)
for error handling and ownership cleanup.

Nodes retain their inputs and expressions. Releasing temporary handles does not
invalidate the finished plan. Column references belong to one exact input node;
even a node with an identical schema has a different scope. Obtain new references
from a filter's output when constructing another filter.

Inspect the tree structure and output schemas through sirius_plan_root(), sirius_plan_node_get_kind(),
sirius_plan_node_input(), sirius_plan_node_schema(), and sirius_schema_field().
These functions expose node kinds, input links, and schemas; construction neither combines
filters nor changes the chosen operators. sirius_plan_describe() provides a readable
explanation with shared expression definitions. Its text is diagnostic, not a
serialization format. Full expression-tree accessors are a separate addition.
Borrowed inspection handles may be retained independently. Relative source paths
are resolved using the working directory at future preparation time; the same
relative description can therefore refer to different files in different directories.
Paths identify a single literal filename; glob characters are never expanded.

Plan nodes and expression trees currently support up to 256 levels. Unknown type
and comparison identifiers, mismatched scopes, invalid schemas, and unsupported
construction operations return SIRIUS_INVALID_ARGUMENT. Plans do not yet have a
public preparation or execution operation. C++ builders and Rust wrappers can use
these same C primitives in subsequent additions.

## Ownership and errors

Status values 1–3 report configuration errors.

- Factories return owning handles. Builders, configurations, schemas, expressions, plan nodes,
  and plans support retain/release: retain acquires another reference without allocating;
  release each owning reference exactly once. Inspection returns borrowed handles unless
  documented otherwise; retain one before using it independently of its owner.
- Destroy engine context handles with sirius_context_destroy().
- Destroy error diagnostics with sirius_error_destroy(). Never use `free()` on Sirius handles.
- Fallible operations return a status, clear their output slots, and optionally provide
  an owned diagnostic. Release previous results before reusing their output slots.
- Diagnostics are best effort. A nonzero status is a failure even if no message is available.
- No C++ exception crosses the C interface. Treat unknown nonzero status codes as failures.
- Keep every borrowed handle alive throughout each call; do not release it concurrently with
  its use. Immutable builders, configurations, schemas, expressions, plan nodes, and plans support
  concurrent reads, construction, and ownership transfer between threads. Give output slots
  exclusive access.

### Diagnostic output convention

Status-returning functions use an optional `sirius_error **out_error` parameter.
Pass `NULL` to omit diagnostics, or the address of a `sirius_error *` output slot.
Functions clear the slot before doing work. It remains `NULL` on success; on failure,
it receives an owned diagnostic if one can be allocated, otherwise `NULL`.
The status code determines success or failure independently of the diagnostic.

Destroy any previous diagnostic before reusing its output slot. Functions overwrite
slots without reading or releasing previous contents; an unreleased diagnostic
would be leaked, not reported as an argument error. Output slots require exclusive
access for the duration of the call.

Messages may contain embedded NUL bytes. Use sirius_error_message_size() as the
complete byte length rather than `strlen()`. The borrowed bytes remain valid until
the diagnostic is destroyed.

## Compatibility

The C ABI assumes a matching platform and architecture; Sirius and its dependencies
must still meet the platform's runtime requirements.

SIRIUS_ABI_VERSION and sirius_abi_version() are reserved for future ABI versioning and
currently return zero. Matching values do not guarantee compatibility during development;
use matching headers and library releases. We will define the versioning policy before
promising ABI stability.
