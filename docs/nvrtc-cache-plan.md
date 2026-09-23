# Plan for Simpatico NVRTC caching — issue #1863

Status: implemented as an approved draft stack; readiness remains with the author. The single squashed
commit `8a309bfb` from [PR #1799](https://github.com/sirius-db/sirius/pull/1799) is
cherry-picked onto the bottom layer pending that PR's merge into `dev`.

Planning snapshot: 2026-09-23; local `dev` is `d79d4f97`; the squashed prerequisite PR head is
`8a309bfb78789515a9621ccc6a3ed0f1f5786647` and is still open. Its current diff changes
CCCL header sourcing and CMake dependencies. The existing runtime cache remains the
starting point for [issue #1863](https://github.com/sirius-db/sirius/issues/1863).
Recheck the merged version before reconciling the bottom layer with `dev`.

## Outcome

Reuse a compiled GPU kernel only when its compilation inputs match. Header-only
changes must invalidate disk entries even when rendered source stays identical.
Warm in-process lookup should avoid source copies, header hashing, filesystem
access, and repeated compiler/driver discovery. Cache failures should still allow
compilation and execution.

Deliver this in three stacked PRs, each with its own tests, using GitHub's `gh-stack`:
`dev` → `stacked/nvrtc-cache-01-identity` → `stacked/nvrtc-cache-02-integration` →
`stacked/nvrtc-cache-03-storage`. Push to `origin` (`sirius-db/sirius`); the fork is
named `9prady9`. Publish every PR as draft and leave readiness to the author's self
review. Follow the repository's [stacking guidelines](../CONTRIBUTING.md#stacked-prs)
and [reviewability checklist](../CONTRIBUTING.md#pr-reviewability). Reconcile the
cherry-picked prerequisite through `gh stack rebase` after #1799 merges. The issue
closes after all three layers land; eventual merges proceed bottom-up.

## What the code establishes

| Current behavior | Evidence | Consequence for the design |
| --- | --- | --- |
| Memory and disk keys independently combine source, architecture, runtime, and driver information. | [kernel_cache.cpp](../src/compression/simpatico_codegen/src/jit/kernel_cache.cpp#L64), [ShapeKey](../src/compression/simpatico_codegen/src/codegen/jit/kernel_cache.hpp#L14) | Replace the duplicated construction with one identity model. |
| Every cache lookup concatenates the full source and entry symbol. | [get_or_compile_plain](../src/compression/simpatico_codegen/src/jit/kernel_cache.cpp#L186) | Stream fields into a digest without building another source string. |
| NVRTC receives project headers and CCCL headers, but the cache does not identify either collection. | [compile_plain_kernel](../src/compression/simpatico_codegen/src/jit/nvrtc_compiler.cpp#L163) | Generate identities from the actual named header inputs. |
| The external include setting adds a fallback `-I`; compiler options are assembled inside the compiler wrapper. | [nvrtc_compiler.cpp](../src/compression/simpatico_codegen/src/jit/nvrtc_compiler.cpp#L189) | Share preparation of effective inputs between compilation and caching. |
| Temporary filenames contain only a PID; disk-cache location is initialized once per process. | [publication](../src/compression/simpatico_codegen/src/jit/kernel_cache.cpp#L92), [configuration](../src/compression/simpatico_codegen/src/jit/kernel_cache.cpp#L46) | Exercise same-process writers and set test environments before starting workers. |
| Standalone Simpatico tests are excluded from the default build; CI runs `sirius_unittest`. | [top-level CMake](../CMakeLists.txt#L233), [test workflow](../.github/workflows/test.yml) | Add explicit build and execution steps for the new tests. |

These source links describe the local baseline; the prerequisite's four-file diff was
checked separately to establish the expected post-merge starting point.

## Recommended design

### One description of compilation inputs

Prepare an internal compilation request once, and use it both to identify the
request and to supply NVRTC. It includes the source, entry symbol, program name,
target architecture, ordered effective compiler options, selected header provider,
and compiler identity. Avoid a second manually maintained list of key fields in the
disk-cache code. Keep `get_or_compile_plain` as the cache's production entry point;
test injection belongs in private fixtures, not additional public overloads.

Split identity by lifetime, within that single model:

| Part | Contents | Lifetime/use |
| --- | --- | --- |
| Environment identity | Embedded header manifest, provider kind, compiler artifacts, invariant program/compilation settings, and conservative runtime/driver compatibility tags | Generated at build time where possible; completed once per process. |
| Request identity | Rendered source, entry symbol, architecture, and effective request options | Computed for each lookup using views of the existing strings. |
| Persistent identity | Format version plus the environment and request identities | Used only after an in-memory miss. |

The default memory table is bound to one immutable environment, so it can use only
the request identity. It need not copy or hash a binary-constant header fingerprint
on every lookup. If a future provider permits multiple environments in one process,
partition the tables by environment or add a compact environment identifier.

Use SHA-256, with fixed-size binary digests internally and lowercase hexadecimal
only for filenames/diagnostics. Specify tagged, length-prefixed fields, fixed-width
integer encoding, list counts, and list order; do not use delimiter concatenation or
`std::hash` as a persistent encoding. Share one encoding implementation, with known
vectors covering field-boundary ambiguities. Reuse an established digest implementation
through a small internal wrapper; OpenSSL Crypto is already a Sirius dependency, but
the standalone Simpatico target must declare its dependency explicitly. Measure any
digest-context allocation and reuse scratch state if needed.

Proposed disk layout: `<cache-root>/v2/<environment-id>/<request-id>.cubin`.
Both IDs come from the same prepared identity. Existing flat entries are ignored;
there is no fallback to the incomplete old key. Changing the encoding or identity
semantics requires a namespace version bump. A finite digest makes collisions
unlikely, not impossible.

### Identify the headers actually supplied

Extend the existing [project embedder](../src/compression/simpatico_codegen/cmake/embed_jit_headers.cmake)
and [CCCL embedder](../src/compression/simpatico_codegen/cmake/embed_cccl_headers.cmake).
Generate the manifest while emitting the named header contents so it cannot drift
into a separate list of inputs. Include both `stdint_shim.hpp` and `decode/rle_block.cuh`.

Resolve include roots in their effective order first. Then canonicalize the selected
logical names and content identities for hashing; exclude installation prefixes.
Equivalent selected names/bytes agree across prefixes and disjoint-root reorderings.
Reordering conflicting roots changes identity when it selects different bytes.
Reject ambiguous duplicate logical names in the final in-memory header collection.

Track generator dependencies so edits, closure changes, and changes to selected
inputs regenerate the manifest. Test incremental rebuilds as well as direct script
execution. Include newly shadowing files in the dependency audit; a depfile containing
only previously selected files may not detect those additions by itself.

Keep libcudf-aligned embedded headers as the initial provider. Adopting CUDA 13.3
bundled headers is separate work. Reserve an explicit provider identity so such a
change cannot silently reuse entries from the libcudf provider; any future extraction
directory must work independently of whether cubin caching is enabled.

### Identify the actual compiler

`CUDART_VERSION` describes the runtime build headers. `nvrtcVersion` reports only
major/minor, so neither alone identifies a compiler patch. See NVIDIA's
[NVRTC API and library documentation](https://docs.nvidia.com/cuda/archive/13.3.0/nvrtc/index.html#general-information-query).

For static builds, generate a content identity for the actual linked NVRTC,
builtins, and PTX compiler artifacts. The last dependency is explicit in the
[vcpkg overlay](../vcpkg_ports/nvrtc/portfile.cmake#L103). Use logical component names
and artifact contents, not their installation paths. Register the artifacts as build
dependencies so replacement triggers regeneration.

For shared builds, identify the actual runtime-selected compiler components once;
build-time toolkit metadata is insufficient if loading selects another installation.
The first implementation investigation must establish a reliable way to resolve and
identify those components, including lazily loaded builtins, on the supported builds.
Prefer content identity; do not assume a SONAME or filename encodes the patch.
If identity cannot be established, skip persistent reads/writes and report the reason
through existing JIT diagnostics. Ordinary supported pixi builds must still demonstrate
a second-process disk hit before the integration PR is ready.

Initially retain the existing runtime and driver fields as conservative disk
compatibility tags, queried once successfully. Document that they are not substitutes
for compiler identity. Removing unnecessary invalidation is a later evidence-based
refinement. Measure compiler discovery separately from warm lookup.

### Treat external headers conservatively

When nonempty `SIMPATICO_JIT_CCCL_INCLUDE` is active, bypass disk reads and writes.
NVRTC searches supplied in-memory headers before `-I` directories, so the setting
does not replace the embedded tree. This precedence is documented in
[NVRTC's compile options](https://docs.nvidia.com/cuda/archive/13.3.0/nvrtc/index.html#supported-compile-options).

Recommended first implementation: also bypass memory reuse for requests using this
escape hatch, so edits at the same path are effective without requiring a restart.
Retain ownership of each returned kernel until `KernelCache::clear()` or destruction,
preserving the existing pointer-lifetime contract. Accept the extra compilation and
retained kernels for this exceptional mode, and document the tradeoff. Snapshot the
setting once per request and pass that same value to compilation. Changing process
environment concurrently with active requests remains unsupported.

Audit implicit filesystem inputs too. Test `--no-source-include` on CUDA 12/13 to
prevent the current directory from supplying untracked quoted headers in the normal
embedded-only path. Include the effective option in identity. Do not claim complete
header tracking if another uncontrolled input remains.

### Publish safely; clean up explicitly

Keep cache I/O best effort. Use an exclusively created, unique temporary file in the
destination directory, check writing and closing, then atomically rename. Remove
only the writer's own temporary file on failure. Allow redundant compilation under
contention initially; a cross-process compilation lock is not required for safe
publication.

Missing, unreadable, empty, truncated, or unloadable cubins become misses and trigger
compilation. Catch cache-I/O failures without hiding an actual compile or execution
failure. Avoid race-prone deletion of a destination another writer may have repaired.

Choose explicit cleanup for this iteration. Extend/document the existing
`clear_jit_disk_cache()` behavior for recognized old and versioned cache layouts in
the configured root. Cleanup is operator-requested, preferably with writers stopped;
it is never triggered merely because another environment namespace appears old.
Document that retention is otherwise unbounded, and include abandoned temporary-file
handling in the explicit maintenance policy. Tests operate only in their own temporary
directories. No automatic eviction policy is needed to complete this issue.

## Delivery sequence and acceptance

### PR 1 — Identity primitives, generated manifests, and host coverage

Changes: private identity/digest code, both header generators, their CMake dependencies,
host fixtures, and explicit CPU CI execution. Specify the format and compatibility
policy with the implementation. Production cache adoption belongs to PR 2.

Acceptance:

- Changing one byte independently in a CCCL header, `stdint_shim.hpp`, or
  `rle_block.cuh` changes identity while source remains identical.
- Equivalent input trees in different prefixes agree. Root-order tests cover both
  unchanged selection and changed selection. Logical renames/additions/removals count.
- Every defined identity component is tested, including compiler patch/artifact
  changes, provider, source, entry, architecture, and ordered options. Encoding tests
  distinguish ambiguous concatenations.
- Generator tests detect omitted content hashing; incremental rebuild tests detect
  missing dependency wiring. Tests execute without a GPU or driver initialization.
- CI explicitly builds and runs the new targets and fails if no matching tests exist.

### PR 2 — Production cache integration, migration, and cheap lookup

Changes: [cache implementation](../src/compression/simpatico_codegen/src/jit/kernel_cache.cpp),
[compiler wrapper](../src/compression/simpatico_codegen/src/jit/nvrtc_compiler.cpp), private
identity preparation, compiler discovery, versioned paths, and external-header gating.
Update the corresponding headers, cache callers' comments, and README. Add the process
test harness and GPU CI execution in this PR.

Acceptance:

- A worker linked to production Simpatico compiles and launches a kernel using a
  fresh temporary cache. Another process loads it from disk and returns the same
  correct result. Assert compile/disk-hit counters, not elapsed-time ratios.
- Build header variants using the real embedders and production cache/compiler code.
  Keep source, symbol, architecture, and options fixed; make the kernel's result
  depend on the changed header. Variant B recompiles and returns B's result, then
  hits its own cache on another run. A's entry remains available to A.
- The regression fails when production stops incorporating header identity; merely
  checking a helper-produced filename or injecting a fake digest is insufficient.
- Cover external-only fallback headers, embedded-header precedence, edits at the
  same path, switching/unsetting the override, and ordinary reuse afterward. Cover
  both separate processes and sequential requests where settings can change.
- New code never reads an old-format entry. Unknown compiler identity disables only
  persistence. Verify static and shared compiler-discovery paths.
- Benchmark held-constant real encode/decode sources before and after: lookup
  allocations/bytes, median/p95 warm latency, first-process setup, disk-hit latency,
  and cold compilation. Acceptance: no full-source concatenation, no per-hit header
  copies/scans or version queries, and no additional steady-state lookup allocations.
  Investigate a reproducible warm-latency regression rather than promising an
  unmeasured speedup. Keep timing thresholds out of correctness tests.

### PR 3 — Publication, recovery, cleanup, and final documentation

Changes: private disk-storage helpers, cleanup implementation, storage/process tests,
and the complete user/developer documentation for identity and lifecycle.

Acceptance:

- Concurrent threads in one process and independent processes publishing the same
  key produce a usable final file. Readers never accept a partial publication.
- Interrupted writers and leftover temporary files do not prevent subsequent
  compilation or loading. Failed writes/close/rename clean up only owned temporary
  files. Invalid cubins are rebuilt and subsequent runs can hit the repaired entry.
- Disabled, missing, and unwritable cache locations permit correct execution. Use a
  reliably denied location or injected private I/O failure; a permission-bit-only
  fixture may not be unwritable when CI runs as root.
- Explicit cleanup covers recognized namespaces and orphaned temporaries without
  following arbitrary directories or touching unrelated files. Tests never clear a
  developer's cache. A different build's entries survive ordinary cache use.
- README accurately distinguishes memory reuse, disk reuse, external-header behavior,
  migration, unbounded retention, compiler-discovery limits, and measured costs.

## CI and verification contract

Use `pixi run` for local commands and the matching `pixi run -e cuda12` environment
for CUDA 12. Add focused CTest labels/targets and invoke them explicitly; a green
top-level build does not establish standalone test coverage. CPU build jobs run
identity/generator tests, build the required GPU workers, and package them. GPU jobs
run the subprocess and storage suites after downloading those artifacts. Ensure
relocated build artifacts do not depend on the original builder's absolute paths.

Reuse the existing CUDA matrix: CUDA 13 on PRs; CUDA 12 and 13 on merge-group/dev.
Obtain CUDA 12 evidence before declaring the change ready, and verify the static
vcpkg build as well as shared pixi builds. Keep unrelated standalone-test failures
separate; the new targets must not depend on building every excluded test.

Launch each process test with cache settings supplied before initialization. Use
fresh temporary directories, enable `SIMPATICO_JIT_STATS`, and isolate CUDA's own
compiler cache when measuring cold behavior. Existing
[child-process environment support](../test/cpp/utils/child_process_environment.hpp)
is a useful pattern; do not mutate the parent test runner's cached configuration.
Run existing encode/decode and compression roundtrips alongside the new targeted
tests. Finish implementation with the repository's required build/test and
`pixi run pre-commit run -a` checks.

## Decisions to verify during implementation

The shared-compiler identity resolver is the main feasibility question. Prove its
behavior under a patch replacement and alternate runtime library selection before
settling the integration patch. SHA-256 context allocation and build-time artifact
hashing costs also need measurement; neither is a reason to weaken identity.

This plan deliberately leaves NVRTC bundled-header adoption, automatic eviction,
cross-process compile deduplication, and build-time `sccache` changes outside the
implementation. They can be evaluated separately without delaying the cache
correctness and lifecycle work.
