# Sirius C API

The public C API provides a C ABI for embedding Sirius. Include the headers under
`sirius/c/` and link the Sirius library. No C++ types cross this interface.

## Configuration

Create immutable configuration snapshots without accessing GPUs:

```c
#include <sirius/c/context/config_builder.h>
#include <stdio.h>

int main(void)
{
    sirius_config_builder *builder = NULL;
    sirius_config *config = NULL;
    sirius_error *error = NULL;
    sirius_status status = sirius_config_builder_create(&builder, &error);
    if (status == SIRIUS_SUCCESS) {
        status = sirius_config_builder_build(builder, &config, &error);
    }
    if (status != SIRIUS_SUCCESS) {
        fprintf(stderr, "Sirius error %u: %s\n", (unsigned)status,
                sirius_error_message(error));
    }
    sirius_error_destroy(error);
    sirius_config_release(config);
    sirius_config_builder_release(builder);
    return status == SIRIUS_SUCCESS ? 0 : 1;
}
```

Use sirius_config_builder_from_yaml() to load the
[YAML configuration schema](https://github.com/sirius-db/sirius/blob/main/docs/super-sirius/configuration.md).
The file is read once; builders and built configurations retain its settings independently.
Hardware availability and capacity are checked when initializing an engine.

## Ownership and errors

- Each returned handle owns a reference. Retain a builder or configuration to acquire
  another reference; release each reference exactly once. Retaining does not allocate.
- Destroy error diagnostics with sirius_error_destroy(). Never use `free()` on Sirius handles.
- Fallible operations return a status, clear their output slots, and optionally provide
  an owned diagnostic. Pass empty output slots; release previous results before reusing them.
- Diagnostics are best effort. A nonzero status is a failure even if no message is available.
- No C++ exception crosses the C interface. Treat unknown nonzero status codes as failures.
- Keep borrowed handles alive for each call. Do not release a reference concurrently with its use.

## Compatibility

Headers use C99-compatible declarations. The C ABI assumes a matching platform and architecture;
Sirius and its dependencies must still meet the platform's runtime requirements.

SIRIUS_ABI_VERSION identifies the expected ABI revision. Compare it with sirius_abi_version()
when loading a library. Breaking C ABI changes increment the revision. New functions may be
added within a revision, so applications must also select a library providing the symbols they use.
The API is under active development; this is not yet a promise of a stable release interface.
