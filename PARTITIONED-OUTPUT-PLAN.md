# Partitioned / multi-destination output (roadmap #1) — implementation plan

Goal, staged. **v1 broadcast:** a sender fragment delivers its FULL output to every
destination — unlocks broadcast joins; gate: a broadcast join over FILES() runs on 2 CNs vs
DuckDB. **v2 hash:** partition i → destination i via the engine's partitioned sink — unlocks
shuffle joins AND grouped two-phase aggregation; gate: the GROUP BY that fails today with
"needs partitioned streaming output (#838)" returns correct groups on 2 CNs, `count(*)`
exactly-once, and the grouped-two-phase translator guard is DELETED (FRAG-7 already pins the
merge semantics on GPU). Both stages keep the single-destination path byte-identical.

Base: `demo-multi-cn` on top of the byte-range stack (`..8c2ebea5`). Provenance: 3 research
reports + 2 converged designs (session scratch `po-R*.md`, `po-design-{deep,opus}.md`);
essentials captured here — this doc is self-contained for resumption.

## Decisions (converged)

**(a) Broadcast is an engine sink mode**, not CN fan-out: `partition_spec` gains a broadcast
kind; destination 0 keeps the batch handle (today's zero-copy N=1 push,
`sirius_physical_streaming_sink.cpp:88-103`), destinations 1..N-1 get independent deep GPU
copies built like partition slices (`gpu_partition_impl.cpp:98-104`). CN pull-once/fan-out was
rejected: it cannot serve HASH at all (that fan-out is a GPU kernel), cannot serve two LOCAL
destinations (`relay_from` is destructive, `sirius_ffi.cpp:588-596`), and would widen the
accepted peer-lease-leak surface. Payoff: broadcast and hash present the SAME shape to the CN
(one park, N output streams, stream i ⇒ destination i) — the sender redesign is written once.

**(b) FFI = two incremental setters** (`declare_output_broadcast()`,
`declare_output_hash_key(col_idx, cast?)`), `require_not_built`, assigned into
`spec.partitioning` at `sirius_ffi.cpp:538` — the one unset line that makes ≥2-output
fragments throw today. Integer keys canonicalize to INT64 before hashing
(sink `key_cast_types`, `gpu_partition_impl.cpp:56-64`); key types outside
{integral, boolean, string} refused (survey wire evidence: only SMALLINT/INT/BIGINT/VARCHAR
keys occur, 0 mismatched-type exchanges in 95). Hash = existing cudf MURMUR3.
**No CRC32, no StarRocks parity, no bucket-shuffle escape needed**: SHUFFLE_HASH_BUCKET emits
plain HASH_PARTITIONED sinks (`PlanFragmentBuilder.java:3242-3250`; 0/306 dumps carry the
bucket type). Cross-side parity = all senders share (exprs order, N = destinations.len(),
INT64 canonicalization); pinned by a cross-instance determinism test.

**(c) Receiver: nothing new.** destination[i] ← stream i ← partition i, positional (matches
`ExecutionDAG.java:550-560`); `per_exch_num_senders` already counts sender instances; the same
`sender_id` at every destination is safe (receiver keys include the receiver instance,
`local_exchange.rs:19-22`). New refusal: duplicate destination instance in one list.

**(d) Failure:** first failing destination aborts the fan-out, fails loudly at the exec RPC or
`results.fail`; the LOCAL lease is released exactly once via a scope guard; peer leases stay
one-per-(destination,batch) — the known leak gap is not widened. The park entry is POISONED
once (unclaimed destinations fail loudly at claim time, no hang). The cross-query
`parked.clear()` (`engine.rs:231-240`) is replaced by per-query drop. Send order: local
`push_sender` first, remote second (fast loud failure beats a 600 s rendezvous wait).

**(e) Guards:** `N == 1 ⇒ gather regardless of partition type` (single-dest path
byte-identical). After v1 refuse: HASH+N>1 (v2 message), RANDOM/RANGE/BUCKET/HYBRID sink
types, UNPARTITIONED-with-exprs, duplicate destinations, sink fragments with `output_exprs`
(0/211 on the wire), `is_merge`+N>1. After v2: delete the HASH refusal and the grouped
two-phase guard (`node_translator.rs:535-546`); keep avg/DISTINCT and exotic partition types.

## Commit stack

v1 — broadcast:
- **C1 engine**: sink broadcast mode + `[streaming_sink]` BCAST tests (N copies, zero-batch
  destination behavior — hard gate for "the partitioned sink never ran under a real pipeline").
  OQ: `data_batch::clone()` (`data_batch.hpp:334`) vs cudf-table rebuild for the copy.
- **C2 FFI**: `declare_output_broadcast` + `spec.partitioning` assignment + cxx bridge; Rust
  multi-output fragment test — the first-ever ≥2-output fragment end to end.
- **C3 CN engine.rs**: park-once under `ParkId` with `parked_by_slot: SenderSlot→(ParkId,
  stream)`, `outstanding` refcount, poison-on-failure, per-query drop replacing
  `parked.clear()`; `SENDER_OUTPUT_STREAM` deleted (stream ids 0..N-1).
- **C4 CN transport**: release-one-destination semantics, poison propagation, lease scope
  guard around export→transmit.
- **C5 CN service**: parse `TDataStreamSink.output_partition`, fan out UNPARTITIONED to N
  routes (local first), staged refusals, single-destination regression test.
- **C6 e2e**: broadcast join over FILES() on 2 CNs vs DuckDB (Q11/Q16 shapes) + DEMO.md.

v2 — hash:
- **C7 translator**: partition exprs (bare SLOT_REFs) → output column indices via
  `slot_global_index`; refusals for non-bare exprs/bad key types.
- **C8 FFI hash keys** + the **cross-instance determinism test**: same keys, different batch
  splits, two independently built sinks ⇒ identical destination assignment (this is the
  silent-wrong-groups risk pinned down).
- **C9 CN wires HASH** (delete the HASH refusal).
- **C10 delete the grouped-two-phase translator guard** (one line; FRAG-7 pins semantics).
- **C11 e2e**: the formerly-refused GROUP BY on 2 CNs vs DuckDB + `count(*)` exactly-once +
  a Q12-class shuffle join + Q17 (BUCKET_SHUFFLE(S) runs as plain HASH).

Verify per commit: `pixi run make` + `sirius_unittest "[streaming_sink]"` / `make test`;
`cargo test -p sirius --lib`; translator/CN: `cargo test -p starrocks-plan-translator`,
`cn-test-no-engine`, `cn-test`; e2e on `cluster2` vs DuckDB.

## Top risks
1. **The refcounted park replaces `parked.remove` as the exactly-once invariant** — a missed
   release leaks GPU memory, a double release starves a destination. It sits under every
   currently-working streaming path; the single-destination regression tests are the net.
2. **Broadcast deep copies inflate GPU memory N-fold** when the FE's stats-less estimate calls
   a big side "small" — copies are per-batch and spillable; accepted for the demo.
3. **Cross-sender hash parity is unverifiable at runtime** — silent wrong groups if violated.
   Pinned by INT64 canonicalization + key-type refusals + the C8 determinism test + C11's
   count(*) gate.

## Open questions
| # | Question | Blocks | Resolution |
|---|---|---|---|
| Q1 | clone API for the broadcast copy (`data_batch::clone()` vs rebuild) | C1 | read data_batch.hpp at implementation |
| Q2 | zero-batch destination: does an empty partition's receiver complete? | C1/C2 gate | BCAST test decides C1's shape |
| Q3 | duplicate destinations: refuse vs dedup (BE dedups) | C5 | refuse; revisit if a dump shows one |
| Q4 | does the CN consume global runtime filters that assume StarRocks hash routing? | C11 argument | one grep before C11 |

## Progress log
- 2026-08-06: plan synthesized from the converged designs; implementation starting (C1),
  pre-approved ("do what you recommend... update docs to review later").
- 2026-08-06 **C1 DONE** — `6c7217aa` feat(exec): broadcast mode on the streaming sink
  (`partition_spec.mode`, dest 0 zero-copy handle + dest 1..N-1 `data_batch::clone` deep
  copies; mode last in the struct so `{keys, casts}` initializers stay hash). OQ1 answered:
  `clone(get_next_batch_id(), stream, quent_probe)` is the sanctioned copy (the PARTITION
  operator's pattern). Verified: SINKROOT-5/6 (6 cases, 64 assertions) +
  `[stream_session],[streaming_fragment]` regression (214 assertions).
- 2026-08-06 **C2 DONE** — `a03d0b88` feat(ffi): `declare_output_broadcast` — the first
  ≥2-output fragment ever built through the FFI (`spec.partitioning` finally set; N==1 stays
  the plain gather byte-identically). Verified: Rust
  `broadcast_fragment_feeds_every_destination` — two output streams, each relayed to its own
  receiver, both produce the full result (8 lib tests green). OQ2 (zero-batch destination)
  deferred to C5's service tests — the engine side proved fine with non-empty outputs.
- 2026-08-06 **C3 DONE** — `b77583ed` park-once/N-claims in engine.rs (`ParkedOutput` +
  `parked_slots: SenderSlot→(park id, stream)`, refcounted release, duplicate-slot refusal,
  `SENDER_OUTPUT_STREAM` deleted; `FragmentRun.outputs: Vec<SenderSlot>` + `broadcast`).
  Deviations from the design, documented: per-query drop and poison-on-failure NOT yet
  implemented — the failure path still clears both park maps (same blast radius as before,
  no regression); revisit with C5 where query context lives. Verified: cn-test (98 engine
  tests) + cn-test-no-engine, all green.
- 2026-08-06 **C4+C5 DONE (merged)** — `845d120a` service fan-out: N destinations routed up
  front (duplicate/missing-transport refusals BEFORE GPU work), sender runs once with N
  output streams + broadcast mode, locals rendezvous first then per-destination remote
  sends; `ReadyFragment` chain → `Vec` (the dispatch worker drains a queue — one fan-out can
  ready several receivers at once); HASH+N>1 refuses with the v2 message; RANDOM/BUCKET/RANGE
  refuse. The design's separate transport commit proved unnecessary: DropParked became
  release-one in C3 and send_fragment is already per-slot. Verified: cn-test (98) +
  cn-test-no-engine (95 incl. single-destination regressions).
- 2026-08-06 **C6 v1 GATE PASSED (live)** — broadcast join over the SPLIT 155 MB lineitem on
  2 CNs: `count(*) = 2756236`, `sum(l_extendedprice) = 105611443786.67` — exactly the DuckDB
  oracle. EXPLAIN confirms the full composition: supplier fragment BROADCASTS to both CNs
  (log: one nixl transmit `stream_id=3 bytes=19008` + one local relay of the same stream —
  the mixed local+remote fan-out), each CN joins its byte-range split, two-phase partial →
  gather → merge. Three roadmap features (splits + broadcast output + two-phase agg)
  composing in one query. DEMO.md update pending with the v2 e2e commit.
- 2026-08-06 **C7 DONE** — `93a45956` translator resolves HASH_PARTITIONED sink keys to
  output column indices (`TranslatedPlan.output_partition_columns`; bare-SLOT_REF-only and
  output_exprs refusals). Verified: translator 97+12, cn-test 98+7, cn-test-no-engine.
  Note: two amend rounds were needed — TranslatedPlan literals in fragment_executor.rs and
  engine.rs tests needed the new field; ALWAYS run cn-test before committing translator
  struct changes.
- 2026-08-06 **v2 COMPLETE — STACK DONE (`6c7217aa..5b4cfc7a`, 9 commits).**
  C8 `2e7ce8d9` FFI hash keys + cross-instance determinism test (identical key→stream maps
  across independently built senders, first run). C9 `22afc9ea` CN wiring (HASH refusal
  deleted; hash_keys through FragmentRun). C10 `8795b92f` grouped-two-phase guard deleted
  (test flipped to grouped_two_phase_translates). C11 `5b4cfc7a` live gates + one real bug
  found by the gate: identity STRING "casts" hit cudf::cast (not fixed-width) — fixed with
  the kernel's EMPTY hash-as-is sentinel (BIGINT also skips its no-op cast; only sub-64-bit
  integrals canonicalize to INT64). **LIVE RESULTS on cluster2 over the split lineitem:
  GROUP BY l_returnflag = DuckDB oracle exactly (1478493/3043852/1478870, Σ=6001215
  exactly-once); lineitem⋈orders PARTITIONED shuffle join = 1201581/45969422546.87
  (oracle-exact); broadcast join + Q6 regressions unchanged.** Engine regression: 14 cases /
  204 assertions green. DEMO.md updated (stale "cannot run"/"one destination" claims gone).
  ROADMAP #1 → DONE; REVIEW-GUIDE Part 4 pending (next session). **Full engine suite green
  over the complete stack: 32,513,698 assertions passed / 1 skipped, 0 failures.**
  SUPERSEDED notes below:
  REMAINING v2: **C8** FFI `declare_output_hash_key(col_idx)` (engine derives INT64
  key_cast_types from _sink_types in streaming_fragment::build; key types outside
  {integral, boolean, string} refused) + Rust hash-fragment test + cross-instance
  determinism test (same keys, different batch splits, two sinks ⇒ identical destination
  assignment); **C9** CN service: replace the HASH+N>1 refusal with wiring
  (outputs=slots, broadcast=false, pass output_partition_columns via new FFI calls in
  engine.rs run path); **C10** delete the grouped-two-phase guard in node_translator.rs
  (search "grouped two-phase aggregation needs partitioned streaming output"); **C11** e2e
  on cluster2: the formerly-refused GROUP BY vs DuckDB + count(*) exactly-once + Q12 shuffle
  + docs (DEMO.md broadcast+hash sections, ROADMAP #1 → DONE, REVIEW-GUIDE Part 4,
  TPCH-SURVEY addendum).
  NEXT: **C6 e2e broadcast join on cluster2** (DONE, see above; docs commit pending); then v2
  (C7 translator keys→indices, C8 FFI hash keys + cross-instance determinism test, C9 CN
  HASH wiring, C10 delete grouped guard, C11 e2e GROUP BY + count(*) + Q12).
  ORIGINAL NEXT-notes (superseded): C4 transport (release-one semantics via drop_parked per destination — may already
  suffice since DropParked is now release-one; check nixl_transport call sites + lease scope
  guard); C5 service: parse `output_partition`, fan out UNPARTITIONED to N routes (local
  relay first, then remote sends; each destination gets its own SenderSlot with its stream
  via outputs order), staged refusals (HASH+N>1 → v2 message, RANDOM/BUCKET etc., duplicate
  dest), single-dest regression tests; C6 e2e broadcast join (Q11/Q16 shape) on cluster2 vs
  DuckDB + docs. Then v2: C7 translator keys→indices, C8 FFI `declare_output_hash_key` +
  cross-instance determinism test, C9 CN HASH wiring, C10 delete grouped guard
  (node_translator.rs "grouped two-phase ... #838"), C11 e2e GROUP BY + Q12 + count(*).
