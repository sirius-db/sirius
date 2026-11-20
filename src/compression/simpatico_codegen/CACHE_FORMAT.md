# NVRTC cache identity format

The host-only identity API is in [cache_identity.hpp](src/jit/cache_identity.hpp).

## Versions

One cache-format version, `kCacheFormatVersion` in
[cache_identity.hpp](src/jit/cache_identity.hpp), controls both identity domains
and the disk namespace. The record marker's version must match; they always
advance together.

| Component | Current version | Canonical definition |
| --- | --- | --- |
| Request identity | `simpatico-request-v2` | `request_identity()` |
| Environment identity | `simpatico-environment-v2` | `environment_identity()` |
| Disk path layout | `v2/` | `CompilationIdentity::relative_path()` |
| Disk record marker | `SIMPJIT2` | `disk_cache.cpp` (compile-time version check) |

The domains and path are derived from that constant in
[cache_identity.cpp](src/jit/cache_identity.cpp).

### Version bumps

Bump `kCacheFormatVersion` for incompatible changes to identity fields, their
order/encoding or meaning, the hash convention, or the disk layout/record contract.
For example, `v2.1` would use `simpatico-request-v2.1`,
`simpatico-environment-v2.1`, and `v2.1/` together. This is a format label, not
automatic semantic versioning. Update lookup/cleanup consumers, format expectations,
and this document in the same change.

The host identity test uses the production request builder in
[compilation_request.hpp](src/jit/compilation_request.hpp), including its program
name and effective options, and the production environment encoder. Expected
digests stay hard-coded in the test. Changes to these inputs or their encoding
fail the test; review the format version before updating the golden vectors.

New source, entry/program names, options, header contents/inventory, or NVRTC
versions change the digests automatically; they need no format bump. Neither do
Sirius releases. An unreleased format can be revised in place.

## Identity encoding

Request and environment records start with their length-prefixed domain string
from the table.
Numbers are unsigned 64-bit big-endian. Strings are a byte length followed by
those bytes, with no terminator. Remaining fields, in order:

- **Request:** source, entry symbol, program name; option count; each effective
  NVRTC option in order. Strings use the encoding above; the count is a number.
  Target SM is included through `-arch=sm_...`.
- **Environment:** project-header digest string, CCCL-header digest string;
  NVRTC major, then minor, each encoded as a number.

Records are streamed into XXH3-128 without concatenating the source. Runtime
hashing is header-only, with no shared-library dependency or CLI invocation.
Digests are 16 canonical big-endian bytes, represented by 32 lowercase hex digits
in paths: `v2/<environment-digest>/<request-digest>.cubin`.

Memory lookup needs only the request digest while the environment is immutable.
Different environment directories may coexist. Legacy flat cubins lack sufficient
identity information and must not be read. Hash collisions remain possible.

## Compiler and reuse policy

`cache_environment()` in [nvrtc_compiler.cpp](src/jit/nvrtc_compiler.cpp) combines
the two header-bundle digests with `nvrtcVersion()` major/minor. This is the entire
environment identity for both shared and static builds: no NVRTC/builtins-library
or static-archive hashes, host-object epochs, provider/linkage labels, or CUDA
runtime/driver tags.

Same-major/minor compiler patches retain reuse even if their binaries differ.
A compatible CUDA driver is still required to load the CUBIN.

We deliberately hash the source/options and named headers supplied to NVRTC.
Host-object epochs would also invalidate on build or host-code changes with
unchanged compilation inputs. The tradeoff is maintaining the input contract:
include new compilation inputs and bump the shared format version under the rules
above when its contract changes.

The environment is computed once on first disk use. A failed version query
reports its NVRTC error; it never substitutes zero or silently disables disk
caching. Memory hits and disk-disabled compilation skip the query. Replacing
the compiler/runtime within a process is unsupported.

## Header manifests

Project headers and the complete CCCL inventory (`cub/`, `thrust/`, `cuda/`, `nv/`)
are separate bundles. The first include root wins duplicate logical names; names
are then sorted bytewise. CRLF/CR become LF before both embedding and hashing.

The manifest starts with literal `simpatico-headers-v2` followed by one LF.
Each record is `<name-byte-length>:<name><content-byte-length>:<normalized-content>`.
Lengths are decimal; no separators between records or per-header digests are added.
`xxhsum -H128` hashes the complete manifest once at build time, and its digest is
emitted beside the same header table supplied to NVRTC.

This build-time manifest framing is defined by
[embed_jit_headers.cmake](cmake/embed_jit_headers.cmake) and
[embed_cccl_headers.cmake](cmake/embed_cccl_headers.cmake), with shared encoding in
[jit_header_manifest.cmake](cmake/jit_header_manifest.cmake). Its prefix describes
the bundle encoding, not another runtime cache version. A framing change must
update both embedders together; changed bundle digests invalidate the environment.

Physical paths are excluded. Include-root reordering matters only when selected
contents change. Directory dependencies track unsuccessful searches too, so newly
shadowing headers regenerate the table.

## Storage and maintenance

Each `.cubin` file is a checked record, not a raw ELF image: eight ASCII bytes
`SIMPJIT2`, an unsigned 64-bit big-endian payload length, 16 canonical bytes of
payload XXH3-128, then that many cubin bytes. A zero-length payload, length/file-size
mismatch, checksum mismatch, non-regular file, or cubin over 256 MiB is a miss.
The cap bounds allocation on corrupt input; larger kernels still compile and
reuse memory normally. CUDA loading remains the final compatibility check.
Raw legacy/intermediate records are never accepted by this reader.

Writers use `mkostemp` in the destination directory, handle short writes and
`EINTR`, check close, and atomically rename the complete record into place.
After writing, records get mode bits `0666 & ~umask`; the mask is read from Linux
`/proc/self/status` per publication without changing process-wide state. If the
mask is unavailable, the owner-only temporary mode is left unchanged. A failed
permission change aborts publication and removes only that writer's temporary.
Each writer owns only its unique temporary file. Failure removes that temporary,
not the destination: another writer may already have repaired it. Redundant
compilation is allowed. No `fsync` or power-failure durability is promised;
damaged records are rebuilt. Checksums are not authentication: the configured
cache directory must be trusted.

The record marker follows the shared cache-format version. Header
digests and NVRTC major/minor select the directory; payload validation and atomic
publication are unchanged. A version-query error is propagated outside the
best-effort storage fallback, not treated as an unreadable cache file.

The payload checksum does not bind a record to its request/environment filename:
moving a complete valid record to another request path still passes validation.
There is no record-binding or authentication guarantee.

Automatic retention and explicit cleanup recognize only these regular files
under the configured root:

- Legacy `<16 lowercase hex>_a<digits>_c<digits>_d<digits>.cubin`, and their
  `.tmp.<digits>` remnants.
- `v2/<32 lowercase hex>/<32 lowercase hex>.cubin`, and their
  `.tmp.<six alphanumeric characters>` remnants.

The configured root is trusted and may be a symlink to a directory. Both prune
empty recognized directories but never remove that root or its alias, follow
symlinks below it, traverse arbitrary nested directories, or remove unknown
versions/unrelated names. Maintenance assumes a trusted directory, not a security
boundary against a process actively replacing directories.

Automatic pruning runs once per process on first disk-cache use, including a
first-use disk hit. It creates/touches the current environment directory, removes
recognized legacy flat files, and keeps the current environment plus the three
newest other environments by directory modification time. Other environments
also survive while their timestamp is within one hour (or in the future). Only
older environments beyond that retained set have their recognized entries removed.
A candidate's timestamp is checked again before deletion. Failures are best effort
and do not prevent compilation or disk reuse. Disabled caching performs no scan.

This is not a live-process lease or a strict directory/byte cap. Memory hits and
later disk hits do not refresh timestamps or rerun pruning; publication normally
updates directory timestamps through file creation/rename. A long-running older
process can lose disk reuse or a pending publication if another process prunes its
environment. Such races are misses or skipped writes, not partial cache records;
already-loaded kernels remain usable. Grace periods and unrelated files can keep
more than four directories, and there is no per-environment size bound.

With writers stopped, `clear_jit_disk_cache()` explicitly removes recognized
entries across all environments, including the current one, regardless of age.
It does not query NVRTC and can also remove abandoned temporary files left after
automatic maintenance. Unlike automatic retention, this is a quiescent operation.

The host storage suite links the same implementation as production. Linker-level
syscall fault injection tests partial/zero/interrupted writes, permission changes,
close and rename failures without adding production test hooks. Threads and child
processes stress same-key publication. Pruning tests cover retention/grace, future
timestamps, unrelated files/symlinks, and concurrent publication/read/prune races.
All maintenance tests operate only in owned temporary directories.

## Coverage

Host tests need no CUDA driver and exercise the production embedders, relocation,
include ordering, and incremental content/shadowing changes:

```sh
pixi run env CMAKE_BUILD_PARALLEL_LEVEL=16 cmake -S src/compression/simpatico_codegen/tests/cache -B build/cache-host -G Ninja
pixi run cmake --build build/cache-host --parallel 16
pixi run env CMAKE_BUILD_PARALLEL_LEVEL=16 ctest --test-dir build/cache-host -L cache_host --parallel 1 --output-on-failure --no-tests=error
```

For GPU coverage, build `simpatico_cache_gpu_tests` and run CTest with
`-L cache_gpu` in the Simpatico build directory. Workers use fresh processes and
temporary caches with CUDA's own cache disabled. They check results and statistics
for header changes, same-version NVRTC/builtins relocation or byte changes,
version changes/query errors, and disk-disabled compilation. Header variants use
the production embedders with identical kernel source; Linux version/error cases
wrap the query while compilation still uses real NVRTC.

The Test workflow runs host and GPU coverage using its configured CUDA matrix,
independently of Sirius's Catch2 suite. Disk hits perform no compiler-file scan
or builtins-loading probe.

The storage suite also uses real `CompilationIdentity` paths for changed project
headers, CCCL headers and both NVRTC version components. It checks record isolation
and narrow explicit cleanup across those environments. The fresh-process tests
verify pruning on a first-use disk hit, current-directory refresh, one-time
scheduling even after clearing memory, and safe reuse/recompilation in an older
live process whose disk environment another process prunes. Explicit cleanup is
also tested without a successful NVRTC version query.
