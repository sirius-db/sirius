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

## Storage and maintenance

Each `.cubin` file is a checked record, not a raw ELF image: eight ASCII bytes
`SIMPJIT2`, an unsigned 64-bit big-endian payload length, 32 bytes of payload
SHA-256, then that many cubin bytes. A zero-length payload, length/file-size
mismatch, checksum mismatch, non-regular file, or cubin over 256 MiB is a miss.
The cap bounds allocation on corrupt input; larger kernels still compile and
reuse memory normally. CUDA loading remains the final compatibility check.
Raw legacy/intermediate records are never accepted by this reader.

Writers use `mkostemp` in the destination directory, handle short writes and
`EINTR`, check close, and atomically rename the complete record into place.
Each writer owns only its unique temporary file. Failure removes that temporary,
not the destination: another writer may already have repaired it. Redundant
compilation is allowed. No `fsync` or power-failure durability is promised;
damaged records are rebuilt. Checksums are not authentication: the configured
cache directory must be trusted.

There is no automatic eviction, startup cleanup, or retention bound. With all
writers stopped, `clear_jit_disk_cache()` best-effort removes these regular files
under the configured root:

- Legacy `<16 lowercase hex>_a<digits>_c<digits>_d<digits>.cubin`, and their
  `.tmp.<digits>` remnants.
- `v2/<64 lowercase hex>/<64 lowercase hex>.cubin`, and their
  `.tmp.<six alphanumeric characters>` remnants.

It prunes empty recognized directories but never removes the configured root,
follows symlinks, traverses arbitrary nested directories, or removes unknown
versions/unrelated names. Explicit cleanup removes other environments' recognized
entries too; ordinary cache use never does. Cleanup is not a concurrent maintenance
or security boundary against a process actively replacing directories.

The host storage suite links the same implementation as production. Linker-level
syscall fault injection tests partial/zero/interrupted writes, close and rename
failures without adding production test hooks. Threads and child processes stress
same-key publication; cleanup tests operate only in owned temporary directories.

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

Local validation covers shared and static NVRTC on CUDA 12.9/13.3. Static runs
use the real checksum-pinned NVIDIA archives from the repository's vcpkg overlay
and the toolkit PTX compiler (not a complete vcpkg rebuild). A negative control
replaced the private project-header fixture's identity with the baseline identity,
leaving its changed header contents intact: the production regression failed
because it loaded result 1 instead of the required result 2. Restoring the identity
restores the passing regression.

The optional `simpatico_cache_benchmark` measures 5,000 warm lookups of rendered
Bitpack encode/decode source. It counts C++ `new`/`new[]` calls during lookup,
not allocations inside the CUDA driver or the C allocator. There are no timing
assertions. Render once, compile/load once, then measure repeated cache lookups.

Diagnostic sample, 2026-09-23: RTX PRO 6000 Blackwell, driver 615.71.09, CUDA 13.3,
NVRTC 13.3.33, GCC 15.3, Release build, disk cache disabled. The baseline uses the
production cache/compiler from the cherry-picked #1799 commit, with identical
renderers and embedded headers.

| Request | Source bytes | Baseline median / p95 | New median / p95 | C++ allocations/lookup, before → after |
| --- | ---: | --- | --- | --- |
| Encode | 5,515 | 4,056 / 4,282 ns | 1,338 / 1,412 ns | 3 (16,564 bytes) → 0 |
| Decode | 3,234 | 2,437 / 2,572 ns | 907 / 928 ns | 3 (9,721 bytes) → 0 |

The same disk-disabled sample's cold encode/decode calls were 461/344 ms before
and 460/349 ms after; these include kernel compilation and loading.

In a separate disk-enabled sample under build load, compiler discovery took
46–50 ms once per process. A second process performed zero requested-kernel
compiles and two disk hits: its first lookup took 50.6 ms including discovery;
the subsequent decode disk lookup took 128 µs. Discovery deliberately performs
an NVRTC preprocessor-error probe to load lazy builtins, plus artifact hashing.
Thus a disk hit avoids kernel compilation, but does not promise zero NVRTC work
on the first process lookup. These are local diagnostics, not performance bounds.
