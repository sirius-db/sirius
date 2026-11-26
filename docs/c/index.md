# Sirius C API

The public C API provides a C ABI for embedding Sirius. Include the headers under
`sirius-c/` and link the Sirius library. No C++ types cross this interface.

## Headers and linking

- `<sirius-c/export.h>` defines visibility and C/C++ declaration helpers.
- `<sirius-c/context/config_builder.h>` builds immutable context configurations.
- `<sirius-c/object_store/config_builder.h>` builds immutable object-store configurations.
- `<sirius-c/object_store/registry.h>` maps URI prefixes to object-store configurations.
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

## Object-store configuration

Describe an S3-compatible object store without contacting it: an endpoint, a region,
credentials, a transport, a signing mode and a CA bundle path. Builders are immutable. Each
`with` function returns a new builder with one field replaced; release the input once the
result replaces it:

```c
#include <sirius-c/object_store/config_builder.h>
#include <string.h>

static sirius_status set_region(sirius_object_store_config_builder **builder,
                                const char *region)
{
    sirius_object_store_config_builder *next = NULL;
    sirius_status status = sirius_object_store_config_builder_with_region(
        *builder, region, strlen(region), &next, NULL);
    if (status == SIRIUS_SUCCESS) {
        sirius_object_store_config_builder_release(*builder);
        *builder = next;
    }
    return status;
}
```

Text fields are byte ranges: they are copied as given, without trimming, and need no
terminator; a size of zero means empty. A range that contains a NUL byte is
SIRIUS_INVALID_ARGUMENT. sirius_object_store_config_builder_build() checks the endpoint, the
region and the keys, and returns SIRIUS_INVALID_CONFIGURATION for the first one it refuses. It
reads no file and consults no I/O backend: the CA bundle path is not opened, and every declared
transport and signing mode is accepted. Diagnostics never contain a field's value; a
diagnostic for SIRIUS_INVALID_CONFIGURATION names the field and the reason. No function
returns a field. See the
[complete C example](https://github.com/sirius-db/sirius/blob/main/test/cmake/installed_consumer/object_store.c).

## Object-store registry

A registry lists the stores a caller may read from: each store is a URI prefix and the
configuration that serves the URIs under it. Registries are immutable.
sirius_object_store_registry_with_store() returns a new registry with one store added, or
replaced when the prefix is already present; release the input once the result replaces it.
One configuration can serve several prefixes.

Register an empty prefix for the S3 fallback, or a prefix such as `s3://bucket/` or
`s3://bucket/data/`. Write the scheme in lowercase. The bucket may contain ASCII letters,
digits, `.`, `_` and `-`; it must not be empty, `.` or `..`. End each prefix component with `/`;
components after the bucket must not be empty, equal `.` or `..`, or contain `%`. A prefix
cannot contain `?` or `#`.

Matching compares the lookup scheme without regard to ASCII case and compares bucket and key
bytes exactly. A nonempty registry refuses an empty key after the bucket/key separator, an
empty, `.` or `..` key segment, a trailing slash, or a literal `%` in the key before selecting
any store, even if only the fallback is registered. For example, plan construction keeps `%2F`
as written; the matcher refuses that key rather than treating it as `/`. Otherwise the longest
matching prefix wins. If no nonempty prefix matches, the fallback can serve only a URI beginning
with `s3://` in either ASCII case.

A NULL prefix pointer with a nonzero size or a NUL byte in the supplied range gives
`SIRIUS_INVALID_ARGUMENT`. A prefix that fails the grammar checks gives
`SIRIUS_INVALID_CONFIGURATION`; an available diagnostic names `uri_prefix` and the reason, not
the supplied value. Building a registry reads no file and consults no I/O backend. No engine API
consumes a registry yet.

## Engine context

Use sirius_context_create() with a built configuration and release the result
with sirius_context_destroy(). Hardware discovery and resource allocation happen
at creation. Only one active engine context per process is supported; callers
must enforce that restriction. See sirius_context for process-wide settings and
lifetime constraints.

## Physical plan construction

Construct plans without a context, a GPU, or any file or network access. The initial subset
supports Boolean and Int64 fields, a Parquet scan of a local file or an S3 object with an
explicit schema, typed column references and literals, comparisons, and filters. Each factory
returns a complete immutable object or an error. Schema names and column types are checked
during construction. Field names are case-sensitive; file schemas and engine support
belong to future preparation.

```text
ParquetScan(schema, location) → Filter(predicate) → Plan
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
Borrowed inspection handles may be retained independently.

A scan source is a nonempty byte string without NUL, denoting a local path or an S3
location. Its bytes are stored as given and identify a single literal file or object; glob
characters are never expanded. Relative local paths are resolved using the working
directory at future preparation time; the same relative description can therefore refer to
different files in different directories.

For example, `s3//b/k`, a name consisting of a space followed by `s3://b/k`, and `"s3://b/k"`
with the quote bytes included are local paths. Their escaped names appear in the description.
By contrast, `s3:/b/k` has a URI scheme but is not a valid S3 location, so construction
refuses it.

An S3 location is `s3://bucket/key`, with the scheme in either ASCII case. The bucket is one
or more ASCII letters, digits, `.`, `_` or `-`, other than `.` or `..`. User-info and a port
cannot be written, and a host name there is a bucket name, not an endpoint. The key is not
empty: `@`, `:` and `%` are ordinary bytes in it and percent sequences are not decoded. `?`
and `#` are rejected anywhere in the location, so a key that contains one cannot be
expressed. Any other URI scheme or malformed S3 location returns SIRIUS_INVALID_ARGUMENT; its
diagnostic does not repeat the input.

A scan takes no credentials or object-store configuration, and construction does not select
a store. Store selection belongs to future preparation and can refuse a location that
construction accepts: object-store prefix matching currently cannot serve a key that contains
`%`, an empty, `.` or `..` segment, or a trailing slash. For the scan source,
`sirius_plan_describe()` shows `s3://<redacted>` only for a location accepted as S3; it does
not print that location's bucket or key.

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
