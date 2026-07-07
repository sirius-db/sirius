# Follow-ups: from standalone streaming operators to StarRocks integration tests

> **Status: planning doc.** Owner: Alexander. Written 2026-07-01. Assumes the
> [streaming source](streaming-source-plan.md) (#836) and
> [streaming sink](streaming-sink-plan.md) (#837) PRs have **landed** — two unwired,
> unit-tested operators plus the `exchange_channel` primitive. This doc sequences
> everything between that point and the first real StarRocks integration tests, ending at
> the project's first internal goal: **all of TPC-H across multiple compute nodes,
> correctly** (onboarding §8). References as in the two plans: `design §N` =
> [exchange-design.md@c859311](https://github.com/mbrobbel/sirius/blob/c8593116000a5fab2228b25d9d937025e526adbe/experimental/starrocks/docs/exchange-design.md),
> `discoveries §N` = [discoveries.md](discoveries.md), onboarding §N = [onboarding.md](onboarding.md).

## 1. Ownership frame

Per the onboarding agreement (§7): the streaming/exchange design work **#838/#839/#840 is
Alexander's**; Matthijs owns the plumbing — the #1021→#1022→#1024 execution stack, the
nixl agent (in progress on his machine, no PR yet), the #959 FE↔CN test harness, and
"the follow-ups that wire the operators in: **streaming-session FFI plus the
`sirius`-crate wrappers**". The agreed sequencing:

```
operator (#836/#837, done) → expose via FFI → Rust sirius crate → plan translation → execution
```

Each F-task below names an owner accordingly; "confirm" means the split needs a quick
check with Matthijs before filing.

## 2. Task ladder

Ordered by dependency, not calendar. Each is a candidate GitHub issue (file under the
`starrocks` label, linked from tracking issue #826 — part of the board reorg Alexander
volunteered for, onboarding §7.5).

### Phase A — engine-internal wiring (Sirius core, no StarRocks anywhere)

| # | Task | Owner | Notes / done-when |
|---|------|-------|-------------------|
| F1 | **Stream session (#839)** — already filed. Engine-side session object: owns channels + input/output repositories (registered with the context's `data_repository_manager` — closes the source plan §6 accounting gap partially), `push(stream_id, batch)` / `close_input(stream_id)` / `pull()` / `wait()`, stream-id routing to the right source, **edge-triggered re-arm** (channel `on_push` → `schedule(source)`; `on_pop` → `sink.try_flush_pending()` + `schedule(sink)`), query-cancel semantics for channels, config via `operator_params` (discoveries §13.4). | Alexander | Done when IT-0 (§3) passes. Blocked on nothing — #836/#837 landed. |
| F2 | **Plan wiring** — plan generator emits `STREAMING_SOURCE` for exchange-input fragments and `STREAMING_SINK` at the fragment top (replacing `RESULT_COLLECTOR`); sink placed per the agreed boundary shape (sink plan §3/§9.2); **`finalize_operator()` invocation mechanism** for the terminal sink decided and implemented (discoveries §13.2); input-port barrier type chosen for overlap (sink plan §5.3). | Matthijs (confirm) | Done when a hand-built two-fragment Substrait pair lowers to correctly wired pipelines. |
| F3 | **#838 — partitioned sink.** Per-destination channels, GPU hash partition reusing PARTITION's machinery with StarRocks-compatible hashes (fnv/xxh3 by `exchange_hash_function_version`, CRC32 bucket-shuffle — design §4, validated against `test/sql/test_exchange_hash_function`), coalesce/split to min/max batch size. | Alexander | **Not needed for the first integration tests** — see the gather-first strategy (§3). |
| F4 | **#840 — resource management.** Design pass first (design §6 option A: shared cuCascade manager; staging arena; the four credits/floors), then implementation. Also closes the "wrapper-pushed batches unaccounted" v1 gap (discoveries §13.6). | Alexander | Design review with Matthijs before code. |

### Phase B — public surface (FFI + Rust)

| # | Task | Owner | Notes / done-when |
|---|------|-------|-------------------|
| F5 | **Streaming-session FFI** in `sirius_ffi.hpp` (today: only `make_context*` — discoveries §11; #1022 adds one-shot `execute_substrait`): create/destroy session from Substrait bytes, push a batch into a named stream, close-input, pull/wait on output. Open design point: the batch-ingestion type at the FFI boundary (Arrow C device data interface vs a raw device-buffer descriptor) — whatever nixl's receive path can hand over zero-copy. | Matthijs | Done when a C++ caller can run IT-0's scenario through the FFI only. |
| F6 | **Rust `sirius` crate wrappers** — `StreamSession` type over F5, integrated with the `SiriusEngine` actor model from #1024 (context is `!Send`/`!Sync`, one dedicated thread; sessions must respect that). | Matthijs | Done when IT-2 passes. |

### Phase C — StarRocks path

| # | Task | Owner | Notes / done-when |
|---|------|-------|-------------------|
| F7 | **Fragment-plan fixture capture.** Extend the CN (debug flag or test hook) to dump `TExecPlanFragmentParams` for real FE-planned queries; commit fixtures for: two-fragment gather agg (`SELECT COUNT(*)`), shuffle GROUP BY, shuffle join, merging exchange (`ORDER BY`). These drive translator tests **without** a running FE and tell us exactly which exchange types/partition regimes the FE emits. | Alexander (small, high value — do early) | Fixtures land in `experimental/starrocks/` test data. |
| F8 | **Translator support for exchange fragments** in `starrocks-plan-translator`: `EXCHANGE_NODE` (fragment input) and the fragment's `TDataStreamSink` output descriptor → Substrait (likely Substrait's `ExchangeRel` — confirm representation early, it's also the FFI contract for F5), carrying destination/partition metadata and the **sender count** the source's EOS logic needs (design §7 "cross-CN completion"). Follow the crate's "adding a node" checklist. | Alexander/Matthijs (confirm split) | Done when F7's fixtures translate; pure-Rust tests, CI-covered (`cargo test --no-default-features`). |
| F9 | **CN execution over sessions.** Replace the one-shot `execute_substrait` call path for multi-fragment queries: per fragment instance, create a stream session; route the root fragment's output channel into the existing `result_store`/`fetch_data` drain; **same-CN (local) exchange first** — sink channel to source channel is an in-process handle hand-off, no serialization, no nixl (design §5). Cross-CN EOS: forward `close_input` per sender; source finalizes after all senders close (session aggregates — design §7). | Matthijs + Alexander | Done when IT-3 passes on a single CN. |
| F10 | **nixl transport** — wire Matthijs's in-progress nixl agent to the sink/source channels for cross-CN exchange (metadata side-channel, staging leases, `transfer_complete` → wrap+register+push handle; retry → bRPC fallback — design §5/§7). | Matthijs | Done when IT-5 passes. |

## 3. First integration tests — the ladder

**Strategy: gather-first.** An UNPARTITIONED/gather exchange (every sender → one
destination) exercises source + sink + session + EOS **without #838** — no hash
partitioning, single output channel, which is exactly the v1 sink. StarRocks plans
`SELECT COUNT(*) FROM t` as *leaf fragment (scan + partial agg) → gather exchange → root
fragment (final agg → result sink)* — a two-fragment plan that needs nothing but the v1
operators. Hash-partitioned shuffles come after #838.

| ID | Level | Scenario | Needs | Pass criteria |
|----|-------|----------|-------|---------------|
| IT-0 | C++ (`[integration]`-style test in `test/cpp/`) | **Loopback exchange**: session A runs `scan/cpu_source → STREAMING_SINK`; the test (playing the wrapper) moves handles from A's output channel to session B's input; session B runs `STREAMING_SOURCE → ungrouped agg → result collector`. Include an EOS path and a backpressure path (tiny channel capacity). | F1 | B's result equals the single-session result of the fused plan. **This is the first true integration test — no StarRocks, no FFI, no nixl.** |
| IT-1 | C++ | IT-0 under memory pressure: small GPU budget forces the downgrade executor to spill queued exchange batches mid-stream. | F1 | Same results; log shows exchange-repo batches downgraded and self-healed. |
| IT-2 | Rust (`cargo test`, `sirius-engine` feature, GPU box) | IT-0's scenario driven through the FFI + `sirius` crate: two in-process sessions, handles moved by Rust. | F5, F6 | Same results from Rust. |
| IT-3 | SQL, single CN | Real FE + one CN (three-terminal loop, onboarding §5.1): `SELECT COUNT(*) FROM FILES('file:///…parquet')` planned as two fragments; both instances on the one CN ⇒ **local exchange only**. | F2, F7–F9 (+ #1021/#1022/#1024 landed) | Row count matches DuckDB on the same parquet. No `stub` strings (that means the StubExecutor ran, onboarding §5.1). |
| IT-4 | SQL, single CN | `GROUP BY` / shuffle-agg queries. A single CN instance may degenerate the shuffle to one destination — verify with F7 fixtures whether this works pre-#838 or is the first #838 gate. | IT-3 ± F3 | Results match DuckDB. |
| IT-5 | SQL, 2 CNs, one box | Rerun IT-3/IT-4 with two CN processes (needs the multi-CN local-cluster pixi task below); exchange goes cross-process over nixl (or its bRPC fallback first). | F10 | Same results; both CNs execute fragments (check logs). |
| IT-6 | SQL, multi-CN | **TPC-H ladder** at SF1 via `FILES()`: Q6 (filter+agg, gather) → Q1 (group-by agg) → Q3/Q5 (joins, needs #838) → … → all 22. Then correctness at larger SF, then — and only then — performance vs CPU StarRocks (explicitly deferred, onboarding §8). | F3 + all above | Every query matches the DuckDB oracle. This is the first internal goal. |

**Test infrastructure tasks feeding the ladder:**

- **Oracle**: DuckDB over the same parquet files (simplest trustworthy comparator — the
  engine-side `compare_gpu_vs_cpu()` pattern doesn't reach across the FE). Script the
  comparison; TPC-H parquet generated at fixed SF/seed (the repo's dataset tooling).
- **Multi-CN local cluster**: extend the `pixi run -e cn cluster` task family to N CNs
  with distinct ports; document in the three-terminal loop style (onboarding §5.1),
  including the `LD_LIBRARY_PATH` workaround until the `sirius-sys` static/shared link
  fix lands.
- **CI reality**: `.github/workflows/experimental.yml` runs only the no-engine pure-Rust
  path on CPU runners (onboarding §6) — translator tests (F8) are CI-covered; IT-0/IT-1
  ride the main GPU unit-test job (`pixi run make test`) once listed in `TEST_SOURCES`;
  IT-2+ are **manual on a GPU box** until a GPU CI lane exists for the CN. State this in
  each PR's validation section.
- **#959 harness**: once Matthijs's FE↔CN `cargo test` harness lands, fold IT-3+ into it
  so "boot FE, register CN, run SQL, assert rows" is a test, not a manual loop.

## 4. Sequencing at a glance

```
#836 ──► #837 ──► F1 (#839, session) ──► IT-0/IT-1 ──► F5 ──► F6 ──► IT-2
                        │                                            │
                        └── F2 (plan wiring) ── F7 ── F8 ──────► F9 ──► IT-3/IT-4 (single CN, local exchange)
                                                                     │
                                              F10 (nixl) ───────────► IT-5 ──► F3 (#838) ──► IT-6 (TPC-H)
                                                                                └─ F4 (#840) — parallel design track
```

Critical path to the first SQL-level integration test (IT-3): **F1 → F2 → F8 → F9**, with
F7 early because its fixtures de-risk both F2 and F8. F3 (#838) is deliberately off the
critical path (gather-first); F4 (#840) runs as a parallel design track and becomes
blocking only when real workloads hit the shared-budget wall.

## 5. Risks & watch items

- **FE plan shapes are assumptions until F7.** Whether `COUNT(*)` really produces the
  two-fragment gather plan, and what partition types the FE emits per query class, must
  be read from captured fixtures — capture before building F8/F9 logic on guesses.
- **Sender-count EOS** (design §7): the session must know how many upstream instances
  feed each stream id (from the FE plan via F8) — missing this shows up as queries that
  hang (source never finalizes) or truncate (finalizes early). Make it an explicit F8/F9
  acceptance item, tested in IT-3 with >1 leaf instance.
- **Partial-aggregate wire format** (design §7): the distributed GROUP BY ships
  `HASH_GROUP_BY` partial state to `MERGE_AGGREGATE` — pin the representation during F2
  lowering, before IT-4.
- **Merging exchange (`ORDER BY`)** is v1-unsupported (design §4) — IT-6 must route
  around it (or fall back) until a receiver-side merge exists; affects Q1's final sort
  presentation among others.
- **Interim memory accounting gap** (discoveries §13.6): until F4/#840, wrapper-pushed
  batches are unaccounted — IT-1 bounds the blast radius but large streams can still
  overshoot the budget; keep exchange batch sizes conservative in early tests.
- **Environment blocker** (onboarding §5.1): the `sirius-sys` link workaround
  (`LD_LIBRARY_PATH` on `cn-run`) is still needed for every GPU-linked Rust test until
  Matthijs's build fix lands.

## 6. Housekeeping when this phase starts

- File F1–F10 as issues under #826 (`starrocks` label), with the IT ladder as acceptance
  criteria on the relevant ones; update the onboarding.md status board (#836/#837 → ✅).
- Project-board reorg (onboarding §7.5): use the F-ladder as the column/milestone
  structure — "operators (done) → session → public surface → StarRocks path → TPC-H";
  review with Matthijs. Keep it demoable at every phase (onboarding §8): after IT-3, a
  single-node `COUNT(*)` over parquet through a real FE is the demo.
