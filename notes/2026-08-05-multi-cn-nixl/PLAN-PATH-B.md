# PLAN — Path B: nixl-first multi-CN demo (`demo-multi-cn`)

Branch `demo-multi-cn`, cut from `demo-streaming-integration` at `55fd14bd`. Scope: a **working
demo** of one query's fragments on two CN processes (one host, split GPU), with the exchange hop
carried by **nixl** — no Arrow data tier, no merge-grade hardening. Shared facts, gap table
(G1-G10), design decisions (D1-D7) and the nixl feasibility verdict live in
[MULTI-CN-PLAN.md](MULTI-CN-PLAN.md); this doc only records what Path B changes, its own
milestones, and the abort criterion.

## What Path B cuts, and what it keeps

| Kept from Path A | Cut for the demo |
|---|---|
| M0 EngineConfig + `cluster2` (verbatim) | M2's Arrow-IPC data path + `push_arrow` FFI |
| M1 identity/routing/loud-error + rendezvous refactor (verbatim) | brpc **data** tier entirely — transmit is nixl-only |
| PRPC client (control plane still needs it) | M4 hardening (GC, cancel, caps) — restart CNs on failure |
| G9 multi-file `FILES()` fix (determinism + distributed scan) | Arena↔cuCascade budget integration (arena sits outside the budget, documented risk) |
| D4's remote-sender-as-result-fragment shape + spike | Retry-once-then-fallback (no fallback tier exists; failure = loud query error) |
| EOS on brpc → `close_input`; sequence idempotence | |

**The trade, stated honestly:** Path B has no correctness A/B. If a distributed result is wrong,
there is no second transport to bisect against — only the single-CN cluster as the control. And
it is **gated on B0**: if the probes refute same-host nixl on this box, Path B reverts to Path
A's M2 (the plan for which is already written).

## Path-B-specific decisions

- **D-B1 — nixl carries all batch payload.** No `transmit_chunk` data frames. A remote
  destination with no live nixl session is a loud fragment error, end of story.
- **D-B2 (as built: per-batch signaling rides brpc, not nixl notifs** — column names must cross anyway and the rpcs exist; nixl moves only device bytes; the transfer is WRITE-based with receiver-granted leases so lease lifetimes stay process-local**).** Original text: per-batch metadata rides nixl notifications. The `createXferReq` notification blob
  carries `(query_id, stream_id, sender_id, seq, pack_metadata)` — the receiver learns "a batch
  landed in lease L" from `getNotifs`, no brpc round-trip per batch. `cudf::packed_metadata_view`
  validates schema receiver-side before any device work.
- **D-B3 — control stays on brpc:** one new tiny exchange (`exchange_nixl_md`) for agent-metadata
  blobs at first contact, and EOS via an empty `transmit_chunk{eos=true}` frame (reuses generated
  types, zero proto edits). One source of truth for sender-set completion, as in Path A.
- **D-B4 — the arena is outside the cuCascade budget for the demo.** `--gpu-staging-bytes`
  (default small, e.g. 512 MiB/CN) is simply subtracted from the operator's head when choosing
  `--gpu-memory-limit`. Documented as the §8-adjacent risk it is; budget integration is the first
  post-demo task.
- **D-B5 — copy-out-on-arrival** (from the verdict): received batches leave the lease
  immediately into ordinary pool memory — leases stay short-lived, no lease-aware spill needed.

## Milestones

- **B0 — Probes (gate).** M6.0 verbatim: device handle-type attrs; nixl v1.3.2 install; same-host
  two-process VRAM transfer over `cuda_ipc`; bundled-UCX version; NO_IPC vs `cudaMalloc`
  bandwidth. **Abort criterion: the two-process cudaMalloc transfer must work, or Path B stops.**
- **B1 — M0 verbatim** (EngineConfig, derived YAML, `cluster2`) **+ G9 fix** (multi-file FILES())
  **+ multi-file sf1 data prep**.
- **B2 — M1 verbatim** (identity, routing decision, loud error, rendezvous `SenderSource`
  refactor, dispatch task) — with the Remote variant carrying lease refs, not Arrow bytes.
- **B3 — Staging arena** (M6.1 minus budget integration, per D-B4) + send-copy credit.
- **B4 — Packed FFI pair** (M6.2: `export_packed`/`push_packed`, D4 spike first, equivalence test
  vs `relay_from`).
- **B5 — nixl tier** (M6.3: agent `{host}:{brpc_port}`, arena registration, MD over brpc,
  WRITE lease→lease, notif-carried metadata per D-B2, one transport task).
- **B6 — The demo.** `cluster2` + two-file TPC-H Q6 (and Q1-class with `agg_stage=1`):
  `61567694.95019999`, CN-A logs `transmitted batches via nixl stream_id=... bytes=N`, CN-B logs
  receive + its local relay, `nvidia-smi` shows both CNs in budget, single-CN run as the negative
  control. TPC-H coverage = gather-shape subset (option a); broadcast joins are the first
  extension after the demo stands.

## Status (investigations landed 2026-08-05)

| Item | Status |
|---|---|
| **B0 probes** | **GATE PASSED** — two-process cudaMalloc VRAM→VRAM over nixl/UCX `cuda_ipc` verified on the L4: byte-identical sha256, **~85-90 GB/s** steady vs 0.48 GB/s no-IPC control (177×). Probe code in `../tools/transport_probe/` (moved outside the working tree) |
| B1 M0 coexistence | **CONFIRMED empirically** — two 8 GiB CNs: 8388 MiB each, 6242 MiB headroom, both engines ready, clean drain. A third does NOT fit. Implementation-ready spec delivered (struct, derived-YAML template, literal `cluster2` task, guardrail text from the captured OOM). Probe YAMLs in `../tools/m0_probe/` (moved outside the working tree) |
| B1 G9 fix | **Designed** (`parquet_files_schema`: first-file schema + fail-closed agreement validation; tests listed) — **but G9 was not the real blocker, see below** |
| B1 data prep | **DONE and verified** — `lineitem_multi/` two byte-identical 74,139,347-byte ZSTD files; 6,001,215 rows == original; Q6 revenue `61567694.9502` == original |
| B2 routing + rendezvous | **DONE** — `ca774e84` |
| B3 staging arena | **DONE** — `3f7a9756` |
| B4 packed FFI | **DONE** — `83d68072` (GPU equivalence vs relay_from) |
| B5 nixl tier | **DONE** — `dead333b` (+ proto in submodule `04cd3136`) |
| B6 demo | **DONE** — `61567694.95020001` on the two-CN cluster, canary 67.3 GB/s, hop carried 457856 packed bytes; fix `afb8fbb3` (fetch_data long-poll), docs `3473a686`. GROUP BY on 2 CNs fails loudly (shuffle boundary), as scoped |

## Findings that amend the plan

**F1 — The gate passed, but with a silent trap (B0).** `cudaMallocAsync`-pool memory over
`cuda_ipc` does **not error — it silently degrades ~220×** (0.38 GB/s, correct bytes, endpoint
still advertising a `device(cuda_ipc/cuda)` lane). Consequences, now binding on B3/B5:
- The arena must be **`cudaMalloc`-backed** — rmm `pool_memory_resource<cuda_memory_resource>`,
  never `cuda_async_memory_resource`.
- Because nothing in nixl/UCX flags the degradation, B5 ships a **bandwidth canary**: a startup
  self-transfer between the two CNs' arenas asserting ≥ some floor (e.g. 10 GB/s), or at minimum
  an allocator-type assertion at arena construction.
- `UCX_TLS` **must include `cuda_copy`** (it provides VRAM memory-type detection; without it
  `register_memory` fails with `NIXL_ERR_BACKEND`). Recipe pinned:
  `UCX_TLS=cuda_copy,cuda_ipc,tcp,self` + `UCX_MODULE_DIR=<nixl libs>/ucx`.
- The pip wheel bundles **UCX 1.21.0** but drags in torch (~4 GB) via its Python API — fine for
  probes; B5 links the **C++/Rust surface** (`nixl-sys = "1.3"`, needs a libnixl install:
  `NIXL_PREFIX` or the wheel's shared libs — decide at B5 start).

**F2 — A second, previously uncited blocker found and sidestepped (G9 investigation).** The FE
**byte-splits** parquet files across scan instances: `numInstances = clamp(totalBytes/64MiB, 1,
nodes × dop)` with a hard per-instance byte cap, and any file overflowing the cap is split
(`FileScanNode.java:494-618`). The CN's scan path accepts splits **only when they cover the whole
file within one instance** (`scan_paths.rs:124-177`) — so on a 2-CN cluster, splits land on
*different* CNs and ANY ≥128 MiB `FILES()` scan — including today's single 162 MB demo file —
fails loudly. The prepared `lineitem_multi/` layout defeats this arithmetically: two
**byte-identical** files ⇒ `bytesPerInstance` == file size ⇒ no split possible ⇒ one whole file
per location, round-robined across both CNs — deterministic cross-node scan AND a working
single-CN control. First smoke test at B1: the GPU parquet reader must accept ZSTD compression
and part.0's 61 KB pre-footer pad (spec-legal; verified on CPU DuckDB, unverified on cudf).

**F3 — Honest TPC-H surface for B6 (G9 investigation):** **Q6 and Q1** (with `agg_stage=1`).
Every join query is blocked on the 2-CN topology — the FE places the join in the probe-side scan
fragment, so with 2 scan instances the build-side sender gets 2 destinations, refused by the
single-destination guard. Broadcast-destination support remains the first post-demo coverage
extension (MULTI-CN-PLAN option b).

**F4 — M0 spec tightening:** `--sirius-config` now hard-conflicts (clap error) with the derived
memory flags instead of silently winning — an operator cannot believe a limit is in force when
the YAML is authoritative. Guardrail: with no limit configured and another compute process on
the device, refuse to start pre-priming with the captured rmm OOM text quoted in the message.
