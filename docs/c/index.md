# Sirius C API

The public C API provides a C ABI for embedding Sirius. Include the headers under
`sirius-c/` and link the Sirius library. No C++ types cross this interface.

## Headers and linking

- `<sirius-c/export.h>` defines visibility and C/C++ declaration helpers.
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

## Status codes and diagnostics

Zero means success; treat unknown nonzero status codes as failures. Error diagnostics
are optional, so a failure remains a failure even if no message is available.

Borrow diagnostic text with sirius_error_message() and obtain its byte length with
sirius_error_message_size(). Both accept NULL: they return an empty string and zero,
respectively. Release an owned diagnostic with sirius_error_destroy(), never `free()`.
Keep a borrowed handle alive throughout each call; do not destroy it concurrently
with its use. No C++ exception crosses these functions.

### Diagnostic output convention

Status-returning functions use an optional `sirius_error **out_error` parameter.
Pass `NULL` to omit diagnostics, or the address of a pointer initialized to `NULL`.
Functions clear the slot before doing work. It remains `NULL` on success; on failure,
it receives an owned diagnostic if one can be allocated, otherwise `NULL`.
The status code determines success or failure independently of the diagnostic.

Destroy any previous diagnostic before reusing its output slot. Functions overwrite
slots without releasing previous contents. Output slots require exclusive access
for the duration of the call.

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
