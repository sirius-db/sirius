# Exchange staging arena — onboarding and review

Reviewed tree: `bench/sf500-2-mig-gpus` at `7610840c` (2026-09-09). This is a code-path
review, not a successful GPU-cluster validation. The base allocator is draft
[PR #1693](https://github.com/sirius-db/sirius/pull/1693), whose head is exactly
`1e61c16c` in this checkout; the end-to-end consumers and several hardening commits came later.

## What this mechanism is for

A distributed exchange needs a device address that a transport can register once, while normal
engine allocations use the RMM/cuDF pool. `exchange_staging_arena` owns one stable device slab and
hands out aligned contiguous subranges (leases). It must remain outside RMM: the project measured
UCX `cuda_ipc` silently falling back to host copies for stream-ordered pool allocations. The arena
is an exchange wire buffer; it is not a cache, a scan allocation, or an alternative general-purpose
memory manager.

```mermaid
sequenceDiagram
  participant S as sender CN / engine
  participant SA as sender arena
  participant N as NIXL + UCX
  participant RA as receiver arena
  participant R as receiver inbound store
  participant P as receiver GPU pool

  S->>SA: export_packed(): lease + pack
  S->>R: request_staging_lease(bytes)
  R->>RA: lease (offset, address)
  S->>N: WRITE sender lease to receiver lease
  S->>R: transmit_packed(metadata, seq, offset)
  R->>P: unpack and deep-copy on inbound stream
  R->>RA: release receiver lease
  S->>SA: release sender lease
  R->>S: frame acknowledged by RPC response
  Note over P,R: ticket waits until all senders send EOS;<br/>then push_inbound moves it into the fragment
```

The happy path is deliberately store-and-forward. A receiver is dispatched only when every remote
sender has sent EOS, so inbound frames must be retained somewhere before that point.

## Components and ownership

| Layer | Responsibility | Lifetime / owner |
|---|---|---|
| `exchange_staging_arena` | one `cudaMalloc` or opt-in fabric VMM region; first-fit, coalescing leases | `Context`, shared by the thread-safe FFI handle; transport registers the entire region once |
| `Fragment::export_packed` | pack one parked GPU batch into a local lease; zero-row batch is metadata-only | caller releases local lease after transfer attempt |
| NIXL agent | registers the whole local arena as VRAM; executes WRITE and waits for completion | one dedicated transport thread; registration handle lives with the agent |
| receiver PRPC service | assigns remote lease, claims sequence before payload work, moves a fresh frame into the rendezvous | receiver CN |
| `InboundStore` | unpacks and deep-copies a frame into normal pool-backed `data_batch`; stores it by ticket | receiver `Context`; ticket is consumed by `push_inbound` or dropped on failure/cancel |
| `LocalExchange` | receiver-first rendezvous, sender EOS/sequence state, retirement | CN service; does not own an arena allocation directly |

The ownership rule is simple: an arena lease belongs to the process that allocated it. A local
export lease is released by the sender after the NIXL attempt. A remote lease is released by the
receiver after copy-out, rejection, retirement, or canary handling. A ticket is **not** a lease:
after copy-out it owns pool memory and must be taken or dropped by ticket.

## Invariants that make the current path work

1. Arena allocations are 256-byte aligned, contiguous, and satisfy `free + live == capacity`.
   The allocator coalesces both neighbours on release; capacity limits concurrent live bytes,
   rather than total historical traffic.
2. A packed lease remains valid until NIXL reports its WRITE done. `export_packed` synchronizes the
   packing stream before returning; `write_and_wait` polls transfer completion before the sender
   releases its source lease.
3. `transmit_packed` claims `(receiver exchange, sender, seq)` before staging or releasing the
   payload. A repeated BRPC frame then does not touch an offset that the original delivery may
   already have returned and reused.
4. A receiver accepts a remote sender only after EOS. It validates stable column names and rejects
   sequence gaps; a duplicate is idempotent.
5. Copy-out completes before returning the receiver lease. The arena therefore represents frames
   in flight, while the pool represents queued receiver input.
6. Pre-run translation/fusion errors, receiver retirement, and failed fragment runs route remaining
   data through RAII cleanup: ticketed data is dropped; legacy in-arena data releases its lease.

## RMM, scan cache, host, disk, and MIG boundaries

The arena is allocated after engine bring-up, intentionally outside the GPU pool budget. It is not
reported in RMM/cuCascade pool accounting, cannot be reclaimed by normal downgrade, and cannot be
used by scan-cache pinning. The normal post-arrival `data_batch` *is* pool-backed and follows normal
GPU accounting; in this checked tree it has no inbound-specific spill source, so queued remote
frames do not automatically become HOST/DISK candidates.

Operators must reserve `GPU pool + staging arena + CUDA/context overhead` per CN. This is binding
on the target two-MIG box: the checked-in MIG evidence reports a 40 GiB pool plus 4 GiB arena using
45,530 MiB of each 48 GiB instance. Adding HOST or DISK capacity does not enlarge the arena; it only
helps after data is represented in a downgrade-managed pool batch. The later non-ancestor credit and
spill changes below address precisely this gap.

One CN is meant to see one CUDA device. The NIXL registration always labels the arena device 0, and
bring-up rejects a multi-value `CUDA_VISIBLE_DEVICES`; the launcher pins a normal GPU process with
`--gpu-device` or attempts a MIG UUID process. The current MIG field note says the UUID route fails
inside cuCascade/NVML and uses CUDA ordinals instead. Treat `MIG_DEVICES` in the stock launcher as
an unvalidated/currently failing deployment path, despite its comment claiming it works.

## Review findings

### P1 — a failed NIXL WRITE leaks the remote lease

After `rpc_request_lease` succeeds, `send_fragment` can fail in `write_and_wait` before it sends
`transmit_packed`. Its cleanup releases only the sender's local lease. The receiver knows the lease
only through `transmit_packed`, which never arrived, so neither its rendezvous retirement nor query
cancel can find and release that outstanding arena block. Repeated transfer failures can exhaust a
peer's arena permanently.

**Required change:** add an idempotent receiver-side lease-abort/release operation, keyed by a lease
identity rather than a bare offset, and invoke it on every post-grant failure path (WRITE timeout,
post failure, and uncertain control-plane response). Add fault injection for a WRITE timeout after
remote grant, then assert the peer can re-lease the entire arena.

### P2 — current ingress admission is a production capacity gate

The checked tree grants a receiver lease based solely on arena availability. It then deep-copies every
fresh inbound frame into the default GPU allocator. A receiver cannot execute until EOS, which means
multiple remote senders can fill pool memory before the receiver gets a chance to consume any ticket.
This converts bounded arena pressure into an unreserved pool allocation failure. The local MIG
checklist records this exact failure class for q18/q21.

**Candidate follow-up, not an audited fix:** `91c0370f` and `222b646a` live only in a separate
performance tree and were not reviewed or validated as a solution here. Their receive-credit and
spill/reservation-binding ideas are relevant candidates: credits alone can wait behind retained
EOS-barrier input, while spill would let ticketed batches leave GPU memory. Rebase them into a
reviewable stack and prove bounded progress, cancellation, and correctness before treating ingress
as production-ready.

### P1 — `InboundStore::stage` races Context teardown

`stage()` reads a raw `gpu_space` pointer while holding the store mutex, releases the mutex for
unpack/copy, then dereferences the pointer. `Context::~Context()` calls `InboundStore::State::close()`
under the same mutex but does not wait for active stages before engine-owned memory-space destruction.
The second null check is too late for a stage that passed the first check. A handle intentionally can
outlive Context, so this is a real lifecycle edge, not merely an API misuse.

**Required change:** make `close()` wait for active stage operations (or retain an owning object whose
lifetime covers each stage); test destruction concurrently with a deliberately blocked stage and
assert a defined failure, never a stale dereference.

### P2 — MIG launcher documentation conflicts with observed engine behavior

`cluster8.sh` says `MIG_DEVICES` runs each CN using a `MIG-<uuid>` in `CUDA_VISIBLE_DEVICES`, while
the current branch's own MIG checklist says that exact configuration fails because cuCascade/NVML
counts devices incorrectly. The executable path should reject this mode with the known remedy or the
launcher/documentation should be corrected before it is presented as a benchmark route.

### P2 — fabric allocation has no integration coverage

`fabric` selects VMM `CU_MEM_HANDLE_TYPE_FABRIC` and configures device ordinal 0. It is intentionally
outside unit coverage and needs IMEX/MNNVL. Keep it opt-in until a two-host integration test verifies
registration, WRITE/read correctness, teardown order, and a failure before any partial VMM resources
remain. This is not a same-host/MIG prerequisite; default `cudaMalloc` is the intended local path.

## Test evidence

| Scope | Present in tree | Executed for this review |
|---|---|---|
| allocator | 13 GPU Catch2 `[staging_arena]` cases: alignment, coalescing, overflow, concurrent leasing, byte disjointedness | not run: this worktree has no built `sirius_unittest` and the tests require a GPU |
| FFI / engine | fragment export/push and inbound-store tests; engine tests exercise push-then-release and busy engine lease service | inspected only; not run |
| CN unit path | transmit staging, retired receiver, cancelled ticket, sequence gap, and canary unit tests | attempted `pixi run -e cn cargo test -p sirius-starrocks-cn --no-default-features`; blocked before tests because `experimental/starrocks/starrocks/gensrc/thrift` is absent (submodule not initialized) |
| NIXL | ignored GPU/libnixl cross-agent smoke: registration, WRITE completion, bandwidth floor | not run; explicitly ignored and does not check payload values |
| CI | pure Rust `--no-default-features`; engine/NIXL and C++ GPU tests are excluded | configuration inspected, not rerun |

`git diff --check dev...HEAD` reports three pre-existing trailing-whitespace lines in
`experimental/starrocks/patches/nixl-exchange-proto.patch`; no source change was made by this review.

## Commit / PR map

| Slice | State in checked tree | Suggested reviewable PR |
|---|---|---|
| pure stable allocator and fabric option | draft #1693, commit `1e61c16c` | retain as standalone engine PR; approve only the default allocator contract independently |
| FFI lease/export/push, schema/cardinality | `b7452f67` | FFI PR after #1693; include C++/Rust ownership docs and pack/unpack tests |
| parked output + receiver-first rendezvous | `85da8f6f`, `37b48a46` | CN local-exchange foundation PR |
| NIXL registration, canary, remote exchange, warmup | `36072ef8`, `9e547c89`, `49d23e8a` | CN transport PR; review registration/device and first-contact ordering together |
| cancellation, copy-out and replay-safe receive | `37360a1b`, `698312a1`, `b00334cb` | CN receive-lifecycle PR; must include the remote-lease-abort fix above |
| receive credits | non-ancestor `91c0370f` | separate performance-tree candidate for a CN/FFI admission PR; re-review and validate after rebase |
| spill staged ingress and bind reservation to copying thread | non-ancestor `222b646a` | separate performance-tree candidate; validate alongside admission under the EOS barrier |
| bounded frames, fair transfer window, retry/replay protocol | non-ancestor `7412ef4d` | separate performance-tree candidate; if adopted, split into protocol, FFI, CN pipeline, and telemetry/tests |
| MIG SF500 plans and DISK configuration | `59ab07c3`, `7c18ce8b`, `7092f445`, `7610840c` | docs/bench PR, after correcting the UUID contradiction |

## Sources

- Arena contract/allocation/lease coalescing: `src/include/exec/exchange_staging_arena.hpp:26-63`, `src/exec/exchange_staging_arena.cpp:60-189`, `src/exec/exchange_staging_arena.cpp:213-294`.
- Engine bring-up ordering and inbound store implementation: `src/sirius_ffi.cpp:154-187`, `src/sirius_ffi.cpp:274-284`, `src/sirius_ffi.cpp:418-469`, `src/sirius_ffi.cpp:949-1024`.
- Transport registration, send, and WRITE timeout: `experimental/starrocks/src/nixl_transport.rs:450-522`, `experimental/starrocks/src/nixl_transport.rs:672-810`.
- Receive/replay and cancellation state: `experimental/starrocks/src/compute_node_service.rs:744-915`, `experimental/starrocks/src/local_exchange.rs:397-498`, `experimental/starrocks/src/local_exchange.rs:567-610`.
- Memory/MIG evidence: `experimental/starrocks/benchmarks/cluster8.sh:28-64`, `plan-2mig/CHECKLIST-2mig.md:12-22`, `plan-2mig/DISK-TIER.md:15-27`.
