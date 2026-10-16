# NVRTC cache identity format

The host-only identity API is in [cache_identity.hpp](src/jit/cache_identity.hpp).

## Versions

One cache-format version, `kCacheFormatVersion` in
[cache_identity.hpp](src/jit/cache_identity.hpp), controls both identity domains
and the disk namespace. They always advance together.

| Component | Current version | Canonical definition |
| --- | --- | --- |
| Request identity | `simpatico-request-v2` | `request_identity()` |
| Environment identity | `simpatico-environment-v2` | `environment_identity()` |
| Disk path layout | `v2/` | `CompilationIdentity::relative_path()` |

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

## Coverage

Host tests need no CUDA driver and exercise the production embedders, relocation,
include ordering, and incremental content/shadowing changes:

```sh
pixi run cmake -S src/compression/simpatico_codegen/tests/cache -B build/cache-host -G Ninja
pixi run cmake --build build/cache-host
pixi run ctest --test-dir build/cache-host -L cache_host --output-on-failure --no-tests=error
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
