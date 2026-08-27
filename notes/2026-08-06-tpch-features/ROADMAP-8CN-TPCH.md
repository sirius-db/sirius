# Roadmap — from the two-CN Q6 demo to full TPC-H on 8 compute nodes (DGX, 8 GPUs)

What the `demo-multi-cn` stack deliberately does NOT do, what each gap blocks,
what already exists to build on, and what changes (and what doesn't) at 8 CNs on
a DGX-class box. All file:line claims verified against branch `demo-multi-cn`
(`d6cce3ae..3473a686` on `55fd14bd`); anything not verified is marked
UNVERIFIED/ABSENT.

**Baseline (works today):** one query at a time; Q6-class scalar aggregation;
≤1 destination per sender; whole parquet files per CN; GPU-to-GPU exchange over
nixl. Translator surface: FILE/HDFS scan, SELECT, PROJECT, one-phase
AGGREGATION, SORT, HASH_JOIN, NESTLOOP_JOIN, EXCHANGE
(`rust/crates/starrocks-plan-translator/src/node_translator.rs:168-183`).

---

# Part 1 — Feature gaps for TPC-H Q1–Q22

## 1. Partitioned / multi-destination output — THE gating feature

**Guard today:** a sender fragment with >1 destination is refused: *"a data
stream sink with N destinations needs partitioned streaming"*
(`experimental/starrocks/src/compute_node_service.rs:742-747`). The CN never
reads `TDataStreamSink.output_partition` at all.

**Why TPC-H needs it:** every distributed GROUP BY plans one aggregate instance
per CN = hash shuffle = N destinations — *even at `new_planner_agg_stage=1`*;
every shuffle join needs HASH_PARTITIONED senders on both sides; every broadcast
join needs UNPARTITIONED fan-out. That is **Q1–Q5, Q7–Q22** — everything except
pure scalar-agg Q6.

**Key sequencing insight:** at `agg_stage=1` StarRocks plans
shuffle-then-*final*-agg (need_finalize=true, intermediate==output tuple), which
the translator already accepts. So partitioned output alone — *without*
two-phase agg — unlocks distributed GROUP BY and shuffle joins.

**Already exists:**
- Engine: partitioned STREAMING_SINK (#838, commit `98a042d3`) — N repositories,
  one per destination, GPU hash partition per batch
  (`src/include/op/sirius_physical_streaming_sink.hpp:33-89, 153-158`). Hash =
  cudf MURMUR3 (`src/op/partition/gpu_partition_impl.cpp:73-76`).
- Engine: `partition_spec{key_columns, key_cast_types}`
  (`sirius_physical_streaming_sink.hpp:37-44`); `fragment_spec.outputs`
  (positional, one id per destination) + `fragment_spec.partitioning`
  (`src/include/exec/streaming_fragment.hpp:56-64`); validation + construction
  wired (`src/exec/streaming_fragment.cpp:58, 124-126`).
- Engine: StarRocks-parity CRC32 partition-hash GPU kernel (#1052, commit
  `f37daaf1`) incl. the DecimalV2 int64+int32 split — built and unit-tested
  (`src/op/partition/crc32_partition_hash.cu:17-58`) but **wired into no
  operator**.
- FFI: `Fragment::export_packed(stream_id)` is already per-stream
  (`rust/crates/sirius/src/lib.rs:269`); `declare_output` accepts multiple ids
  (`src/sirius_ffi.cpp:463-472`); `relay_from(source_stream_id, ...)` takes the
  source stream.
- Translator: `slot_global_index` for partition_exprs→column-index resolution
  (`descriptor_table.rs:300`); `duckdb_type_name` for key-cast mapping
  (`type_mapper.rs:43`).
- CN transport: the per-batch lease/WRITE/transmit loop is already per-slot
  (`nixl_transport.rs:480-553`); remote frame sequencing is idempotent per
  (exchange, sender) (`local_exchange.rs:182-199`).

**Missing exactly:**
1. FFI: `spec.partitioning` is never set — `Fragment` has no partition-spec
   setter (`src/sirius_ffi.cpp:514-518` sets inputs/outputs only). Needs a
   `declare_output_partitioning(key_columns, key_cast_type_names)` FFI + Rust
   wrapper.
2. CN: read `output_partition.type`
   (UNPARTITIONED/RANDOM/HASH_PARTITIONED/BUCKET_SHUFFLE_HASH_PARTITIONED),
   translate `partition_exprs` (slot refs) → sender-output column indices,
   derive key casts.
3. CN sender loop: `SENDER_OUTPUT_STREAM` is hardcoded 0
   (`engine.rs:34-35`). Needs N streams with per-destination local-park vs
   remote-transmit. **Hard part:** one parked `sirius::Fragment` owns all N
   output streams, but the local relay path does `parked.remove(slot)`
   (`engine.rs:406-414`) — mixed local+remote destination sets need a
   parked-fragment ownership redesign (park once, relay stream i / export
   stream j, drop when all destinations drained).
4. Broadcast (UNPARTITIONED): no partition kernel needed; the remote side can
   reuse one exported lease for N WRITEs before release (loop structure supports
   it); a local destination in the set needs a batch copy (relay moves, doesn't
   clone).
5. Hash parity: MURMUR3 is fine for all-Sirius HASH_PARTITIONED (correctness
   only needs sender-side consistency; all senders run the same code).
   BUCKET_SHUFFLE_HASH_PARTITIONED would need the CRC32 kernel +
   bucket→instance mapping (kernel exists, mapping ABSENT) — but with
   FILES()-only tables the FE likely never plans bucket shuffle (UNVERIFIED —
   see the plan survey, feature #0).

**Size: L.** Dominant risk: parked-fragment ownership across mixed
local/remote destination sets + N× unspillable output backlog (see §7).

## 2. Two-phase aggregation — **scalar path DONE (2026-08-05)**

**Landed** as the 7-commit stack `64977ebb..11625add` on `demo-multi-cn` (see
REVIEW-GUIDE.md): the FE's **default** `agg_stage` plan now runs end to end for
ungrouped `sum`/`count`/`min`/`max` — partial agg per scan fragment, one
partial-state row per CN across the existing packed hop (Q6 payload: **64
bytes** vs 457 KB one-phase), substituted-function merge on the gather
fragment. Verified live on `cluster2` and single-CN.

**How it landed (differs from the original sketch above — two assumptions were
overturned by implementation research):**
- **No FFI marker.** The engine's Substrait consumer ignores
  `AggregateFunction.phase` AND `Measure.output_type`, so phase semantics can
  only be *which function reads which column*. The phase is classified from
  thrift (`need_finalize` × per-measure `is_merge_agg`, `agg_phase.rs`) and
  resolved entirely in the translator: the merge node becomes a plain aggregate
  with substituted functions (sum→sum, count→**sum**, min→min, max→max — the
  engine's own internal merge table), and the auto-inserted MERGE_AGGREGATE
  wrap performs the cross-CN reduce. `phase` is still set on measures, advisory
  only. MERGE_GROUP_BY stayed unreachable and unneeded (it has no ungrouped
  form and needs PARTITION plumbing).
- **The FE intermediate type is wrong, not just opaque.** Decimal sum's wire
  column is FP64 (the translator's decimal lowering), not the FE's
  DECIMAL128(38,s). `partial_state::wire_type` models the engine's binding as a
  pure function of FE-identical thrift fields; both fragments derive their side
  of the exchange from it. The engine now **validates** every hop
  (`push_packed`/`relay_from` schema guards, commit 64977ebb) so model drift is
  a loud error, not a wrong number.

**Still open (in dependency order):**
- **Grouped two-phase** — refused in the translator ("needs partitioned
  streaming output (#838)"); gated by roadmap item #1. Merge *semantics* for
  the grouped path are already pinned on GPU (FRAG-7), so unblocking it is a
  translator + sink change, not an engine question.
- **avg** — refused loudly: its Sirius partial state is sum+count (two columns
  where the FE allocates one VARBINARY slot), a cardinality change a type
  override cannot express. Follow-up: expand partial avg into `sum(x), count(x)`
  measures + a division ProjectRel on the merge side; needs a synthetic-slot /
  output-width mechanism in the descriptor path.
- **3/4-phase DISTINCT plans** (`multi_distinct_*`, merge-serialize stage) —
  refused; still **XL**, LIST pack round-trip still UNVERIFIED.

Queries: **Q6-class scalar aggregation now runs at FE-default settings**; Q1
and the grouped perf win remain gated by #1.

## 3. Merging (order-preserving) exchange — ORDER BY / LIMIT

**Distributed ORDER BY is NOT correctness-blocked today:** `translate_exchange`
already handles `TExchangeNode.sort_info` by wrapping the stream read in a full
`SortRel` — a receiver-side re-sort (`node_translator.rs:403-417`); exchange
offset/limit become a FetchRel. Each sorted sender has one destination
(gather) → passes the destination guard. Q3/Q10/Q18-style final ORDER BY+LIMIT
works once their upstream shuffle works. `TDataStreamSink.is_merge` and
`TExchangeNode.partition_type` are never read.

**What's missing is efficiency:** a receiver-side k-way merge over per-sender
sorted runs instead of an O(n log n) re-sort of the whole fan-in. Engine-side
`sirius_physical_merge_sort` (MERGE_SORT) and `sirius_physical_top_n_merge`
already exist (`src/include/op/sirius_physical_merge_sort.hpp:28-60`); missing
is per-sender run separation at the stream boundary — the input repository
fan-in interleaves senders' batches with no ordering metadata.

**Size: M** (perf, not correctness). Risk: wasted GPU memory on large fan-ins.

## 4. Byte-range parquet scan splits

**Guard:** `validate_complete_files` rejects any per-fragment range set that
doesn't cover a whole file exactly
(`rust/crates/starrocks-plan-translator/src/scan_paths.rs:124-177`) — this is
what forces the byte-identical two-file gymnastics (the FE byte-splits any file
over `totalBytes/numInstances`). **Mandatory at 8 CNs** — equal-size-file
layouts don't scale, and every TPC-H table scan hits it.

**Already exists:** the translator already *collects* `start_offset/size/
file_size` per range (`scan_paths.rs:104-118`); Substrait
`FileOrFiles.partition_index/start/length` fields exist in the vendored proto
(left default at `node_translator.rs:973-991`); the engine GPU reader is already
row-group- and byte-range-aware (`src/op/scan/parquet_gpu_ingestible.cpp:82-90,
246-257`; `src/scan_manager/split_provider.cpp`); the CN already parses parquet
footers host-side (`file_schema.rs`).

**Missing:** (a) accept partial ranges in `scan_paths.rs` and carry them;
(b) emit start/length on the Substrait read; (c) the consumer gap — DuckDB's
`from_substrait.cpp:826-855` lowers local_files to bare `parquet_scan(paths)`
with NO range support, so ranges must be plumbed either through a consumer
extension into the Sirius scan manager's row-group selection, or resolved
CN-side (read the footer, pick row groups by a deterministic rule such as
midpoint-in-range). The rule must be globally consistent so N CNs' disjoint
byte ranges select disjoint, complete row-group sets.

**Size: M.** Dominant risk: split→row-group assignment consistency — a wrong
rule silently duplicates or drops rows, the exact failure class the current
fail-closed rule exists to prevent (`scan_paths.rs:22-28`).

## 5. Join shapes and scalar subqueries

**Translator today** (`node_translator.rs:738-756`): INNER, LEFT/RIGHT/FULL
OUTER, LEFT_SEMI, LEFT_ANTI, RIGHT_ANTI, NULL_AWARE_LEFT_ANTI; equality
conjuncts only; nestloop inner/cross.

- **RIGHT_SEMI_JOIN missing** — StarRocks emits it on build-side swaps
  (possible in Q4/Q18/Q20/Q21 depending on stats); trivial mirror of the
  existing RIGHT_ANTI handling. **Size S.**
- **ASSERT_NUM_ROWS_NODE missing** — StarRocks wraps uncorrelated scalar
  subqueries in it: **Q2, Q11, Q15, Q17, Q20, Q22**. No engine assert operator
  exists; needs a tiny runtime row-count check (a FetchRel-limit silently
  changes semantics — don't). **Size S-M.**
- **Nothing join-specific blocks distribution:** a shuffle-join fragment is
  HASH_JOIN over two EXCHANGE children, and the receiver rendezvous already
  handles multiple exchange nodes × N senders (`compute_node_service.rs:955-989`,
  `local_exchange.rs:105-139`). The blockers are §1 and §4. Shuffle-hash parity
  across all-Sirius CNs: any consistent hash agrees with itself.
- **Runtime filters:** the FE plans them and the CN **ignores them safely** —
  zero references CN-side; probe sides simply don't filter;
  `transmit_runtime_filter` is never called at us; live demo runs prove
  exec+fetch complete without them. Perf-only gap; Sirius has single-node
  dynamic-filter machinery (`src/op/dynamic_filter_publisher.cpp`) a future
  cross-CN RF could feed. **Size M-L, optional.**

## 6. Concurrency — one query at a time per CN

**For a sequential 22-query TPC-H run, per-CN serialization is sufficient**:
the FE issues queries serially; fragments serialize on the engine thread
(`engine.rs:229` "one request at a time") and receivers chain on the single
dispatch worker; `fetch_data` long-polls up to 600 s.

**What breaks with genuinely concurrent queries** (each verified, file:line):
- `parked.clear()` on *any* failed run drops **every** query's parked sender
  outputs, not just the failed one (`engine.rs:233-239`).
- One process-global query lifecycle: `SiriusContext::QueryBeginStandalone`
  mutex (`src/sirius_context.cpp:219`; `src/sirius_ffi.cpp:351-353, 407,
  479-480`) — per-query lifecycle isolation is the named engine blocker.
- Bind catalog is per-connection with one connection: `catalog.clear()` +
  redeclare per fragment build (`sirius_ffi.cpp:163-164, 431-437`) — safe only
  because the engine thread serializes builds; stream views are keyed by
  exchange-node id only (`sirius_ffi.cpp:410-419`), so two queries sharing a
  node id collide.
- The staging arena's bump head resets only when the LAST outstanding lease
  releases — interleaved leases from concurrent queries starve the reset.
- Cross-CN lease requests queue behind the peer's running fragment
  (`engine.rs:229-271`) with a 60 s brpc timeout (`prpc_client.rs:25`) — a
  >60 s fragment on the receiving CN fails an incoming transmit; already
  plausible at large scale factors, worse under concurrency.
- Rendezvous/result keying is fine — `SenderSlot`/`ExchangeKey` carry the
  FE-unique `fragment_instance_id`; the failure-path clear above is the real bug.

**Size: XL** (engine per-query lifecycle isolation) — NOT needed for the TPC-H
milestone if queries stay sequential.

## 7. Ops / correctness hardening

- **`cancel_plan_fragment`: ABSENT.** The proto rpc exists
  (`internal_service.proto:854`) but the CN implements only
  exec/fetch/nixl/schema; unknown methods return "not implemented". Leak
  inventory on cancel/timeout at N nodes: parked GPU fragments (freed only by
  the next failure's blanket clear or process exit), `LocalExchange`
  receivers/sources/remote_seq for never-completing sender sets, staged arena
  leases for frames whose receiver never runs, `ResultStore` entries never
  evicted, descriptor-table cache never evicted, and an intermediate receiver's
  downstream waiting out the full FE timeout by design. **Size M** (cancel
  handler + per-query GC sweep).
- **Spillability of parked/staged batches: ABSENT, by structural conflict.**
  The downgrade sweep only sees manager-registered repositories
  (`src/downgrade/downgrade_executor.cpp:209`); streaming_fragment deliberately
  creates its repos outside the manager so `QueryEnd()` can't destroy them
  (`src/exec/streaming_fragment.cpp:64-74`). `new-exchange-design.md` §8
  documents the conflict and two fixes (registration-with-exemption vs a second
  enumeration root). Grows linearly with fan-in senders × partitioned-output
  destinations — becomes acute exactly when §1 lands. The cudaMalloc arena
  additionally sits outside the cuCascade budget. **Size M-L**; dominant risk:
  OOM at SF>1 with 8-way fan-in.
- **Timeouts:** `wait_ready` caps at a hardcoded 600 s; the sharper limit is
  the 60 s CN↔CN `REPLY_TIMEOUT` on lease/transmit rpcs. **Size S** to make
  both configurable; moving the lease service off the engine-run critical path
  is M (see Part 2 §4b-2).
- **Transmit retry: ABSENT** — any rpc/WRITE error fails the sender's dispatch
  outright. The receiver's duplicate-frame idempotency already makes bounded
  retry safe to add. **Size S-M.**

## Cross-cutting notes

- **Table access:** TPC-H through this stack means one `FILES()` CTE per table
  (8 for Q2/Q8); multi-file schema inference exists (34931de5). FILES() is the
  only real path today — named-table (HDFS) reads have no backing catalog.
- **Expression surface is TPC-H-adequate:** comparisons, and/or/not,
  arithmetic, CASE, IN, casts, date literals, `year/month/day`,
  constant-pattern `like`, constant-positive `substring`, `if`
  (`expr_translator.rs:519-611`). Aggregates: sum/count/min/max/avg +
  `multi_distinct_count`. TPC-H's only distinct agg (Q16) is grouped — OK
  single-phase.
- **Step 0 for everything:** a translate-only survey of all 22 FE plans has
  not been run. The harness exists — `SIRIUS_CN_TRANSLATE_ONLY` +
  `SIRIUS_CN_DUMP_FRAGMENTS` (`compute_node_service.rs:587-594, 650-663`).
  Run it first; this audit reasoned from the translator surface, not a live
  plan dump.

## Dependency-ordered roadmap

| # | Feature | Depends on | Unlocks | Size | Dominant risk |
|---|---------|-----------|---------|------|---------------|
| 0 | Translate-only plan survey of Q1–Q22 (`SIRIUS_CN_TRANSLATE_ONLY`) | — | ground truth for everything below | S | none |
| 1 | **Partitioned/multi-destination output** — **DONE 2026-08-07** (`6c7217aa..5b4cfc7a`: hash-partitioned fan-out wired through the CN + broadcast sink mode; DECIMAL partition keys hash through a FLOAT64 cast, `8c23e7e7`) | 0 | ~~distributed GROUP BY + shuffle/broadcast joins~~ **cleared** — every shuffle/broadcast TPC-H shape executes live | done | residual: N× parked/staged backlog is still unspillable (§7, #6) |
| 2 | **Byte-range parquet splits** — **DONE 2026-08-06** (`a5c25f76..8c2ebea5`, 7 commits; see BYTE-RANGE-SPLITS-PLAN.md): ranges ride `FileOrFiles.start/length`, a per-plan registry closes the consumer gap in-repo, the engine selects row groups by start-offset ownership (StarRocks reader convention) before stats pruning, the CN emits splits with loud refusals (overlap/past-EOF/has_more). Verified exactly-once live: count(*) = 6001215 over the single split 155 MB lineitem on 2 CNs | 0 | ~~gates 18/22 queries~~ **cleared**; byte-identical-file gymnastics retired | done | residual: S3 ranges refused; FE cross-CN tiling remains the trust boundary |
| 3 | **ASSERT_NUM_ROWS_NODE + RIGHT_SEMI join** | 0 | **Q2, Q11, Q15, Q17, Q20, Q22** (scalar subqueries); robustness for Q4/Q18/Q21 side-swaps | **S-M** | assert needs a real runtime check, not a limit |
| 4 | **Two-phase aggregation** — **scalar DONE 2026-08-05** (`64977ebb..11625add`: translator-resolved phases, no FFI marker; wire-type model + engine hop validation; substituted-function merge). Remaining: grouped (gated by #1), avg (sum+count expansion follow-up), DISTINCT (→ #9) | 1 (grouped part) | ~~FE-default `agg_stage` plans~~ done for scalar; Q1 at scale + grouped perf still gated by #1 | remaining: **M** (avg) / **XL** (distinct) | LIST pack round-trip UNVERIFIED (distinct only) |
| 5 | **cancel_plan_fragment + per-query GC** — loud-failure half DONE (`c858e79a`: query-wide failure propagation + cancel stub; no more silent hangs or cluster cascade); real abort/GC still open — a stalled fragment holds the engine thread + GPU until SIGKILL (q02, task #26) | — (parallel) | reliable 22-query runs; no restart-after-failure | remaining: **M-L** (engine-side abort) | GC sweep vs in-flight fragments |
| 6 | **Spillability of parked/staged batches** (new-exchange-design §8, option 1 or 2) | 1 | SF>1 at 8 CNs without OOM | **M-L** | ownership vs `clear_all_repositories()` |
| 7 | **Merging exchange** (k-way merge receiver; engine MERGE_SORT exists) | 1 | perf for ORDER BY-heavy Q3/Q10/Q18 (already correct via SortRel re-sort) | **M** | per-sender run separation at the stream boundary |
| 8 | **Timeout config + transmit retry + lease path off engine thread** — lease path DONE (`a94e8660`, plus SIGTERM→SIGKILL shutdown escalation); timeout config + transmit retry open | — | long fragments (>60 s) don't false-fail transmits | remaining: **S** | none (idempotency in place) |
| 9 | Multi-phase DISTINCT agg | 4 | **Q16** at FE-auto stages | **XL** | 3/4-phase plan shapes |
| 10 | Runtime filters (opt), CRC32 bucket-shuffle parity (opt), concurrent queries per CN | 1, 4 | perf / mixed-vendor / multi-tenant — not needed for a sequential pass | M–XL | engine-wide lifecycle rework |

**Shortest path to "all 22 sequential at N CNs": 0(done) → 2(done) → 1 → 3 + 5**,
with 6 required before scale factors that pressure GPU memory. Status
2026-08-06: step 0 DONE (survey), **#2 DONE** (byte-range splits — the 18/22
scan blocker is cleared; single large files split correctly across CNs), 4's
scalar half DONE 2026-08-05 (Q6-class runs at FE defaults over split files)
and its grouped half folds into 1. **#1 (partitioned output) is now the single
critical path** — it needs BOTH broadcast and hash-partitioned shapes (survey
F2), six queries raise the bucket-shuffle question (survey F3), and the avg
expansion (Q1/Q17/Q22 leaves) queues behind it. Join-type work (RIGHT_SEMI in
#3) stays off the TPC-H path (survey F5); ASSERT_NUM_ROWS remains #3's content.

**Status 2026-08-07 (measured, not projected): 19/22 TPC-H queries run to
completion on the live 2-CN cluster; 17/22 additionally hold every value inside
a 0.25 % tolerance** (per-query table: TPCH-SURVEY.md addendum 3; per-fix
detail: QUERY-TIMEOUT-ANALYSIS.md "Final status"). #1 DONE
(`6c7217aa..5b4cfc7a`, decimal keys `8c23e7e7`); the avg expansion landed
(`bd232c40`) and grouped two-phase landed with #1; #8's lease half DONE
(`a94e8660`); #5's loud-failure half landed (`c858e79a`), its abort/GC half is
now the top engine ask. #3 turned out to be a non-blocker: the FE elided
ASSERT_NUM_ROWS in every captured plan and RIGHT_SEMI never appeared. Between
19 and 22 stand: real cancellation/abort (q02 engine-thread stall, task #26),
the FP64 equality race (q15 empty-result flake, task #29), and the q14
common_slot_map descriptor gap; the #24 decimal deficit additionally gates
value-exactness on q03/q10.

## What integrating Sirius as a StarRocks CN actually required

Every line is something that actually broke on the way from the Q6 demo to the
19/22 sweep, with its fix commit or open task.

**Engine gaps — what Sirius itself still needs (or needed):**

- **Real cancellation/abort + a per-fragment watchdog.** A stalled fragment
  cannot be aborted: q02 wedges the engine thread and its GPU memory until
  SIGKILL, invisibly to the FE. Cancel is stubbed (`c858e79a`), shutdown
  escalates (`a94e8660`); engine-side abort + watchdog remain OPEN (task #26,
  roadmap #5).
- **Single-engine-thread lease coupling.** Staging-lease RPCs funnelled through
  the one `!Send` engine thread, so a peer's lease queued behind a running
  fragment → cross-CN deadlock (the original q02 hang class). FIXED: leases
  served off the engine thread (`a94e8660`).
- **No INT128.** DuckDB-side sums produce HUGEINT with no wire type; partial
  states must leave through a throwing downcast to the FE-declared slot type,
  so overflow fails loudly instead of wrapping (`bb066e90`).
- **Decimal literal/scale path.** Decimals lower to FP64; every
  `sum(x*(1-l_discount))`-shaped value lands deterministically low — 0.1–0.2 %
  on most queries, up to 0.4 % on q03/q10 rows. OPEN (task #24).
- **FP64 distributed-sum determinism.** Two independently reduced FP64 sums are
  not bit-equal, so q15's `total_revenue = max(total_revenue)` equality flakes
  empty (3/6 on a warm cluster). OPEN (task #29).
- **Exchange staging arena sizing.** The bump arena reclaims only at full
  quiescence and sits outside the cuCascade budget; q09 requested 648 MB
  against 512 MB. Grown to 1280 MiB per CN (`1d4428da`); a real sizing/GC story
  is roadmap #5/#6.

**Translation gaps — what StarRocks→Substrait was missing:**

- **Two-phase aggregation semantics.** The FE encodes phase in thrift
  (`need_finalize` × per-measure merge) and its intermediate tuple types don't
  match the engine's partial-state bindings; needed a phase classifier, a
  partial-state wire-type model, and an engine-checked conformance gate
  (`64977ebb..11625add`, gate `830380f4`).
- **avg expansion.** avg's partial state is sum+count — two columns where the
  FE allocates one slot; expanded into `sum(x), count(x)` measures plus a
  finalize division (`bd232c40`).
- **FE-narrowed builtin return types.** `year()` is SMALLINT to the FE, BIGINT
  to the engine — a silent hang at the hop guard; every projection output is
  now cast to its FE-declared slot type (`4beca977`).
- **Materialized-slot order is load-bearing and undocumented.** The FE's list
  order is NOT the wire order: sort tuples (`4323197d`) and aggregation
  grouping keys (`7bdcd312`) must be shipped in materialized-slot order.
- **CLONE_EXPR.** `TExprNodeType(29)` (the FE's common-subexpression reuse
  marker) was refused outright, blocking q14/q16; translated as a pass-through
  of its child (pending commit).
- **Slot-id resolution / stale tuple ids.** The FE references slots through
  tuples absent from `row_tuples`; a unique-slot-id fallback resolves them
  (pending commit). Still OPEN: q14's `common_slot_map` slot is defined in one
  Project node and referenced by a sibling AGGREGATE node — common-expr
  registrations must become fragment-visible.
- **Intermediate-fragment failure reporting.** A failing non-root fragment had
  no FE-visible channel: silent 300 s hangs that cascaded into a 20-query
  wipe-out; a failure now fails its whole query loudly (`c858e79a`).
- **cancel_plan_fragment.** The unimplemented method returned a transport-level
  error that poisoned the FE's shared brpc channel and misattributed errors
  across queries; stubbed to OK (`c858e79a`); real cancellation is the engine
  ask above.

---

# Part 2 — What changes at 8 CNs / 8 GPUs on a DGX

## 1. Transport: cross-device cuda_ipc works as-is ✔

A WRITE from CN-A's arena on GPU0 to CN-B's arena on GPU3 goes over NVLink P2P;
UCX handles the cross-device mapping automatically. **No transport code change
needed**, provided each CN keeps pinning its GPU with `CUDA_VISIBLE_DEVICES`.

- Verified in the vendored UCX 1.21.0 source: the data mover is
  `cuMemcpyDtoDAsync` (`tools/ucx-src/src/uct/cuda/cuda_ipc/cuda_ipc_ep.c:162`);
  legacy (cudaMalloc) handles — what the arena produces — open with
  `cuIpcOpenMemHandle(..., CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS)`
  (`cuda_ipc_cache.c:282-283`), which delegates peer enablement to the driver.
  On a DGX every GPU pair is P2P-capable through NVSwitch. If a pair weren't,
  the open fails → cuda_ipc declared unreachable → UCX falls back to the staged
  tier — which the bandwidth canary refuses loudly. Every failure mode is loud.
- UCX decides peer reachability *empirically* — by actually opening the remote
  handle and caching per (peer-device, local-device) pair
  (`cuda_ipc_md.c:330-397`); the source comments call out exactly the
  "device not visible through CUDA_VISIBLE_DEVICES" configuration. 8 processes
  each seeing only their own GPU is the *designed-for* shape.
- **One demo assumption to keep:** the nixl descriptor hardcodes
  `device_id() = 0` (`nixl_transport.rs:286-294, 565-570`) — correct only while
  every CN is launched with a distinct `--gpu-device i` (which exports
  `CUDA_VISIBLE_DEVICES`, `main.rs:75-78`). An 8-CN launcher **must** pass one
  device per CN; a CN with all 8 GPUs visible would mislabel descriptors.
- CUDA IPC's one-context-per-process-per-device restriction is satisfied (one
  arena per CN). Bandwidth: same-GPU 67–90 GB/s on the L4 becomes cross-GPU
  NVLink (~450 GB/s/dir aggregate on H100); the 2 GB/s canary floor stays valid.

## 2. Fabric handles / IMEX: direct registration of the rmm pool (optional upgrade)

The mechanism exists end-to-end but is an *optimization experiment*, not a
prerequisite — on single-node DGX, legacy cudaMalloc IPC already works, so the
current arena design scales to 8 GPUs unchanged.

- UCX 1.21 compiles with `HAVE_CUDA_FABRIC` (`tools/ucx-src/config.h:97`) and
  ships the mempool-fabric export/import path (`cuda_ipc_md.c:137-256`,
  `cuda_ipc_cache.c:355-430`). rmm's default pool is created with
  `cudaMemHandleTypeNone` → UCX packs NO_IPC → **that branch is exactly the
  silent 220× cliff the canary guards**.
- rmm 26.06 exposes the fix as a ctor arg:
  `cuda_async_memory_resource(initial, release, export_handle_type)` with
  `allocation_handle_type::fabric = 0x8` (pixi env header,
  `rmm/mr/cuda_async_memory_resource.hpp:44-56`). cucascade constructs the
  pool without it (`cucascade/src/memory/memory_space.cpp:120`); the injection
  seams: (a) `gpu_memory_space_config::mr_factory_fn` — no cucascade change,
  but leaves `pool_handle = nullptr` (degrades OOM diagnostics); (b) cleaner: a
  config export-handle-type field threaded into the default ctor branch.
- **It does not delete the pack copy by itself:** `chunked_pack` also
  linearizes multi-buffer tables and produces the unpack metadata. Direct pool
  registration would need scatter-gather `XferDescList`s + a new wire format +
  receiver-side landing buffers with lifetime signaling.
- IMEX operationals (fabric handles): CUDA 12.4+, `nvidia-caps-imex-channels`
  device + at least one channel file (`NVreg_CreateImexChannel0=1`); the
  `nvidia-imex` *daemon* is only for inter-node. UNVERIFIED: whether a
  single-node DGX H100 reports `CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED=1`
  with only channel0 and no daemon — probe on the target box.

## 3. nixl at 8 agents

- Full-mesh MD over the existing idempotent brpc side-channel
  (`exchange_nixl_md`) is fine at 8: 56 directed one-time exchanges of small
  blobs, cached in `peers: HashMap` (`nixl_transport.rs:311, 396-423`). The
  vendored `libnixl.so` was built WITHOUT etcd (optional meson dep absent;
  `nm -D … | grep etcd` empty) — etcd would buy ops convenience (watch-based
  invalidation on CN restart), not correctness.
- Per-pair canary cost: each first contact runs a 1 MiB warmup + 16 MiB timed
  WRITE and needs a 16 MiB lease on BOTH arenas simultaneously — expect a burst
  of ~56 probes on the first mesh-wide query, each cheap but serialized behind
  real sends (§4a).
- No documented per-agent peer limit in v1.3.2. The real 8-agent bottleneck is
  our own one-thread-owns-the-agent design (the Rust binding's documented
  multithreading caveat), not nixl.

## 4. What actually breaks at N>2 — ranked

**4a. Total sender serialization.** All remote sends funnel through ONE
transport thread served one request at a time (`nixl_transport.rs:75-77,
237-263`); `SendFragment` blocks its caller until the whole fragment crossed.
A CN with outputs for 7 peers transmits strictly sequentially, each batch
paying: blocking lease RPC + synchronous poll-to-DONE WRITE + blocking transmit
RPC. Fix direction: sessions can stay on one agent, but sends need per-peer
workers or async xfer posting (nixl posts are async; the serialization is ours).

**4b. Deadlock/starvation ranking for the blocking lease/MD rpcs (60 s
timeout):**

1. **Mutual first-contact transport-thread cross-wait — a real deadlock,
   near-certain at 8.** A's transport thread inside
   `send_fragment → ensure_session → rpc_exchange_md(B)` blocks on brpc; B's
   inbound `exchange_nixl_md` handler must itself go through B's transport
   thread (`compute_node_service.rs:348, 838-861`). If B's transport thread is
   symmetrically blocked toward A, both handlers queue behind both blocked
   threads → mutual 60 s timeout, both queries fail. Never fired at 2 CNs
   because demo sends were one-directional; at 8 CNs with shuffle, A→B and B→A
   sends are concurrent by construction. **Fix before any bidirectional-shuffle
   attempt:** answer `exchange_nixl_md` without the transport thread (MD load
   is importer-local), or pre-establish all 56 sessions at bring-up before
   accepting queries.
2. **Lease-behind-Run starvation (bounded, not a cycle).** The peer's lease
   handler funnels to its single engine thread, FIFO behind a running
   `EngineRequest::Run`. Runs never block on the network (frames are staged
   before the receiver dispatches), so this is starvation bounded by the
   longest fragment run — but at 8 CNs, 7 senders' per-batch lease+transmit
   rpcs contend with the receiver's Run queue, and a >60 s GPU fragment turns
   every waiting sender into a loud failure. **Cheap fix:** serve
   `StagingLease/Release/Info` off the engine thread — the arena is internally
   mutexed (`exchange_staging_arena.hpp:100`) and doesn't need engine-thread
   affinity. (Roadmap #8.)
3. **Arena exhaustion as failure amplifier.** Receive leases are held from
   frame arrival until the receiver runs and pushes; the bump allocator
   reclaims only at full quiescence. At 8 CNs the arena must hold roughly the
   cumulative leased bytes of the busy epoch: 7 inbound senders × full staged
   output + own in-flight sends + canaries. Exhaustion is loud, but there is no
   per-query GC: a sender dying mid-stream leaves staged leases pinned until
   restart, progressively bricking the arena. Sizing rule of thumb: ≥ 7 ×
   (largest per-peer staged fragment output) + 32 MiB canary slack, informed by
   `high_water()`; remember the arena sits OUTSIDE the cuCascade budget — carve
   it out of `--gpu-memory-limit` headroom explicitly. (Roadmap #5/#6.)
4. **Single receiver-dispatch thread** — ready receivers queue behind each
   other in addition to the engine queue. Latency, not deadlock.

**4c. Launcher recipe per CN i (8-CN DGX):**
- `--gpu-device i` — mandatory (see §1); five disjoint ports per CN
  (heartbeat/thrift/brpc/http/starlet — the `cluster2` pixi task shows the
  pattern); per-CN `--engine-dir`; per-CN `SIRIUS_EXCHANGE_STAGING_BYTES`.
- `numactl --cpunodebind=N --membind=N` with N = GPU i's NUMA node (DGX H100:
  GPUs 0–3 on node 0, 4–7 on node 1 — verify with `nvidia-smi topo -m`).
- Host pools: cucascade builds one host memory space per NUMA node
  (`src/sirius_context.cpp:314-376`), but the CN's derived YAML emits only a
  flat `host: capacity_bytes` (`engine_settings.rs:58-64`). Under
  `numactl --membind` a remote-node host pool would violate the binding —
  either rely on membind + single-node host config, or extend the derived YAML
  to pin `host_memory_space_config.numa_id` to the GPU's node. UNVERIFIED:
  whether the YAML host schema accepts a per-space `numa_id` (inspect
  `sirius_config.cpp` host parsing).
- One whole GPU per CN removes the carve-out pressure entirely — the
  `--gpu-memory-limit` math gets *easier* at 8×.

## 5. StarRocks FE with 8 same-IP CNs ✔

- Node identity is (host, heartbeat_port); duplicates checked on both
  coordinates (`SystemInfoService.java:262-270, 968-975`) — 8 CNs on one IP
  with distinct heartbeat ports register as 8 distinct nodes. No count limit or
  same-IP restriction found; docs treat multi-instance-per-machine as supported.
- `toBrpcHost` keys by (host, be_port) (`SystemInfoService.java:1004-1018`) —
  fine at 8. Nothing on the query path keys by host alone.
- The CN already treats same-host peers as remote by comparing hostname AND
  port (`compute_node_service.rs:55-72`) — exchange routing generalizes
  unchanged.
- **FE scan-range distribution caveat:** the byte-splitting vs
  whole-file-reassembly behavior was only validated at 2 instances; until
  roadmap #2 lands, 8 instances need 8 byte-identical files.

## Bottom line

Transport-wise, nothing conceptual changes at 8 GPUs: the same nixl/UCX
cuda_ipc path carries GPU_i→GPU_j over NVLink automatically, and the arena
design survives. What breaks first is our own concurrency shape, in order:
(1) the transport-thread cross-wait on mutual first contact — fix before any
bidirectional shuffle; (2) leases served on the engine thread — move them off;
(3) arena sizing for 7-peer fan-in with reset-at-quiescence semantics plus the
no-GC leak on failures. Fabric/IMEX direct pool registration is a real
experiment on DGX hardware (rmm ctor arg + config seam; UCX shipped the path
in 1.18) but is an optimization on top, not a prerequisite. And none of the
8-CN transport analysis is *reachable* until roadmap #1 (partitioned output)
lifts the "needs partitioned streaming" guard — that feature is the critical
path for both halves of this document.

## Open questions

1. Which of Q1–Q22 translate cleanly today on a single CN — run the
   `SIRIUS_CN_TRANSLATE_ONLY` + `SIRIUS_CN_DUMP_FRAGMENTS` survey (roadmap #0);
   this audit reasoned from the translator surface, not a live plan dump.
2. Does the FE plan broadcast or shuffle joins (and which agg_stage) for
   FILES()-based tables with no statistics — decides whether UNPARTITIONED
   broadcast or HASH_PARTITIONED lands first inside roadmap #1.
3. Does `cudf::chunked_pack` round-trip LIST columns (COLLECT_SET
   count-distinct partial state) across the packed hop — structurally
   supported, never exercised.
4. FE behavior when `cancel_plan_fragment` returns method-not-implemented at
   query timeout (retry / blacklist / leak) — FE code unread.
5. Does StarRocks ever emit BUCKET_SHUFFLE_HASH_PARTITIONED or colocate shapes
   for pure-FILES() plans — if never, CRC32 parity work drops off the TPC-H
   path entirely.
6. Single-node DGX H100 fabric support with only channel0 and no nvidia-imex
   daemon (`CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED`) — probe on the
   target box.
7. Does the Sirius YAML host-memory schema accept pinning a host space to a
   chosen `numa_id` — needed for the numactl-membind-per-CN launcher.
8. Measured bandwidth of UCX's fabric-mempool path vs the legacy cudaMalloc
   arena path on DGX — decides whether direct rmm-pool registration is worth
   the scatter-gather wire-format change.
