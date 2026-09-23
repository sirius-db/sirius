# NVRTC cache identity format

The identity primitives live in `src/jit/cache_identity.hpp`. They have no CUDA
dependency and are tested by the standalone `tests/cache` CMake project.

## Version 2 encoding

All numbers are unsigned 64-bit integers in big-endian byte order. A string is its
byte length followed by that many bytes, without a terminator. Hash input is streamed
into SHA-256, without concatenating the rendered source. Digests are stored as 32
binary bytes in memory and encoded as 64 lowercase hexadecimal digits in paths.

A request record contains, in order:

1. The string `simpatico-request-v2` (record type and schema version).
2. Source, entry symbol, and program name, each encoded as a string.
3. The option count, followed by each effective NVRTC option as a string, in order.
   Architecture is included through the actual `-arch=sm_...` option.

An environment record contains, in order:

1. The string `simpatico-environment-v2`.
2. Provider name, project header identity, CCCL header identity, and compiler
   identity, each encoded as a string.
3. CUDA runtime build version and driver compatibility version, encoded as numbers.

`CompilationIdentity` combines these records into
`v2/<environment-sha256>/<request-sha256>.cubin`. The process-local table can use
only the request digest when its environment is immutable. A format/semantic change
must bump the namespace and record version. Legacy flat cubins must not be read
under this scheme; they lack sufficient identity information. Different environment
namespaces may coexist. Hash collisions remain theoretically possible.

## Embedded header manifests

The production embedders normalize CRLF/CR to LF before both emitting and hashing
header content. Resolve search paths first, then sort the selected logical names
bytewise. The manifest begins with `simpatico-headers-v1` and a newline. Each record
is the decimal byte length of the logical name, a colon, the name, the 64-character
lowercase SHA-256 of its normalized contents, and a newline. The manifest's SHA-256
is emitted beside the same header table used for compilation.

Physical installation paths are absent. Search-path reordering matters only when
it changes the selected contents. Directory dependencies also track previously
unsuccessful searches so a new earlier header invalidates the generated table.

## Running host coverage

```sh
pixi run cmake -S src/compression/simpatico_codegen/tests/cache -B build/cache-host -G Ninja
pixi run cmake --build build/cache-host
pixi run ctest --test-dir build/cache-host -L cache_host --output-on-failure --no-tests=error
```

The Test workflow invokes this project on its CPU runners for both supported CUDA
environments according to the existing matrix. These tests do not load the CUDA
driver. They exercise the production generators, content-sensitive manifests,
relocated prefixes, include ordering, and incremental content/shadowing updates.
