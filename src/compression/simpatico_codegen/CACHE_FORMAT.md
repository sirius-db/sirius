# NVRTC cache identity format

The identity primitives live in `src/jit/cache_identity.hpp`. They have no CUDA
dependency and are tested by the standalone `tests/cache` CMake project.

## Version 2 encoding

All numbers are unsigned 64-bit integers in big-endian byte order. A string is its
byte length followed by that many bytes, without a terminator. Hash input is streamed
into XXH3-128, without concatenating the rendered source. Digests use xxHash's
canonical big-endian representation: 16 binary bytes in memory and 32 lowercase
hexadecimal digits in paths. Runtime hashing uses xxHash's header-only mode, with
no shared-library dependency or CLI invocation.

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
`v2/<environment-xxh3-128>/<request-xxh3-128>.cubin`. The process-local table can use
only the request digest when its environment is immutable. Once released, a
format/semantic change must bump the namespace and record version. Legacy flat cubins must not be read
under this scheme; they lack sufficient identity information. Different environment
namespaces may coexist. Hash collisions remain theoretically possible.

## Embedded header manifests

The production embedders normalize CRLF/CR to LF before both emitting and hashing
header content. Resolve search paths first, then sort the selected logical names
bytewise. The manifest begins with `simpatico-headers-v2` and a newline. Each record
contains the decimal byte length of the logical name, a colon, the name, then the
decimal byte length of the normalized contents, a colon, and the contents. Lengths
delimit fields without additional separators or per-header digests. The packaged
`xxhsum -H128` tool hashes the complete manifest once; its XXH3-128 digest is emitted
beside the same header table used for compilation.

Project headers and the complete installed CCCL header inventory form separate
bundles. The CCCL inventory includes files under `cub/`, `thrust/`, `cuda/`, and
`nv/`, with the first include root winning duplicate logical names.

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

## Production coverage and diagnostic measurements

Build `simpatico_cache_gpu_tests`, then run CTest with `-L cache_gpu` in the
Simpatico build directory. The Test workflow explicitly builds the workers and
runs their Python harness on GPU runners, independently of Sirius's Catch2 suite.
Each case uses a temporary cache and fresh processes; CUDA's own cache is disabled.
Header variants use the production embedders and cache/compiler implementation,
keep kernel source identical, and check both executed results and cache statistics.
Shared-library cases relocate the actual NVRTC and builtins, then change their
bytes without changing their reported version. Static archive identity is also
covered by host generator fixtures.

The optional `simpatico_cache_benchmark` measures 5,000 warm lookups of rendered
Bitpack encode/decode source. It counts C++ `new`/`new[]` calls during lookup,
not allocations inside the CUDA driver or the C allocator. There are no timing
assertions. Render once, compile/load once, then measure repeated cache lookups.

XXH3-128 diagnostic sample, 2026-09-29: RTX PRO 6000 Blackwell, CUDA 13.4,
GCC 15.3, Release build, full embedded CCCL bundle, disk cache disabled.

| Request | Source bytes | Warm median / p95 | C++ allocations/lookup | Cold compile + load |
| --- | ---: | --- | ---: | --- |
| Encode | 5,515 | 232 / 259 ns | 0 | 670 ms |
| Decode | 3,234 | 174 / 184 ns | 0 | 542 ms |

Discovery deliberately performs an NVRTC preprocessor-error probe to load lazy
builtins, plus one-time XXH3-128 artifact hashing. Thus a disk hit avoids kernel
compilation, but does not promise zero NVRTC work on the first process lookup.
These are local diagnostics, not performance bounds.
