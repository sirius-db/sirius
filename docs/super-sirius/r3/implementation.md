# R3 implementation record

Baseline: `ca294edad88a991ee00ba4cd7f17a4985ccfc5d9` (R2b), branch `r3`.
Execution plan: `/nv/cpp/workshop/sirius-ssw/1796/r3.md`, v2, 2026-10-07.
Original constraints: shared framework v0.3 sections 11.6, 14, 15.1, 18.6 and phase 4 (#1838).

## S0: API and ownership audit

The existing metadata store is an ioctx member: an independently locked map from
path to shared metadata. All three footer producers use resolve_parquet_metadata.
The raw prefetch cache separately keys file entries by raw_file_cache_id; its
asynchronous users retain the opened io_object. Changing only the footer key
would leave stale raw-byte reuse possible.

The preparation unit already retains footer, approval and datasource references.
No new scan scheduler is required. Cache eviction must drop only its own reference.
The global 1 GiB limit requested for R3 covers retained metadata estimates, not
in-flight parsing, raw prefetch, decoded batches or all process RSS. All ioctx
stores must share this limit, while preserving separate access namespaces.

### Backend capabilities on the installed dependency pin

- REST HEAD/footer probe: size and ETag are already captured, without an extra
  validation request. known-size open has size only. Existing range requests do
  not attach If-Match; adding conditional reads is Sirius-owned work.
- io_uring: the buffered fd is owned and already fstat'ed for size; the same stat
  result can supply nanosecond mtime.
- kvikio local: libkvikio 26.08.00 (`5a77056e`) FileHandle::fd(bool=false) is public;
  Sirius can fstat it without changing the dependency.
- kvikio remote: RemoteHandle exposes nbytes but not response ETag or a general
  conditional-request parameter. It cannot claim verified cross-open reuse.
  No extra network request or third-party patch is assumed.

The module-context skill's generated libkvikio docs describe an older 26.02 pin
and incorrectly label the library unused for this branch. The actual installed
26.08 public headers are the API authority. cuDF I/O, cuCascade memory and the
Super Sirius architecture/scan/memory docs were also read.

### Verification environment and unresolved measurements

Local GPU: RTX 2080, 8192 MiB. Existing test binary is dated 2026-09-24 and is not
accepted as evidence for the R2b commit. Pixi is synchronizing the locked CUDA 13
environment before a current build can be attempted. GPU/runtime compatibility
must be checked, not inferred from binary presence.

TPC-H/TPC-DS SF100/SF1000 datasets are unavailable locally. End-to-end comparison
remains outstanding on a provisioned machine. Host unit tests and synthetic
cache workloads can verify correctness and overhead, but cannot close this gate.

### Decisions needed if capability gaps affect integration

Size-only remote handles must not match arbitrary previous opens. Scope cache
reuse to an owned open when no stronger identity exists; record any warm-scan
cost. ETag is an opaque validator, not a sortable version. Local size+mtime is
the user-selected best-effort change detector and does not detect writes that
preserve both. Parser profile and access namespace remain separate dimensions.

S0 code/API audit is recorded; baseline build and performance qualification are
pending. No claim of passed regression or benchmark is made by this commit.

## S1: backend identity

Added the type-specific identity value and per-open generation. REST inherits
ETag identity from its existing validation_tag; absent tags have open-local
identity. Local uring/kvikio capture size and nanosecond mtime from the owned fd.
No third-party source or network request is changed.

Host-only Catch2 verification: 2 cases, 19 assertions passed using `pixi run
--as-is` and the installed compiler. Coverage includes version/kind/size changes,
local same-size mtime changes, identity after unlink while the fd is retained,
and invalid-fd reporting. Full backend compilation is pending environment setup:
the current include tree is missing cuda/std/iterator during synchronization.
