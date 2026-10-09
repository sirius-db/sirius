# Plan: TPC-H coverage and runtime filters, in one order

*Oct 9. This doc owns the order of the work. The detail of each step stays in its own plan:*
- *[tpch-coverage-plan.md](tpch-coverage-plan.md): what the CN refuses or gets wrong, gaps G1–G5;*
- *[design/distributed-runtime-filters.md](design/distributed-runtime-filters.md): runtime filters, Steps 1–6 ("RF Step N" below);*
- *[design/multi-phase-aggregation.md](design/multi-phase-aggregation.md): two-phase aggregation.*

## Why one order

Coverage and runtime filters were planned separately. That leaves the filter work with almost nothing to show:
- Partitioned-join filters (class D) have only 2 queries that run today: q12 and q14.
- Same-fragment filters (class B) have only 1: q19.

Most of the 42 filters the FE plans sit in queries the CN refuses. So the order below brings each query in before the filter step that needs it. Every filter step then lands on queries that already run, can be shown with filters on and off, and is checked against DuckDB.

## What each query needs

From the recorded FE plans of the SF1000 22-query run (`20261006_214436_sf1000-all22`). The error a query shows today is only the first thing that refuses it; this table lists everything its plan needs.

Filter classes (from the runtime-filter plan):
- **A:** a broadcast join, with the scan in a separate fragment (#2062 applies these);
- **B:** a broadcast join, with the scan in the join's own fragment;
- **C:** another target;
- **D:** a partitioned join.

| query | needs before it can run | filters it carries |
|---|---|---|
| q05, q07, q08, q09 | runs | A (12 applied, 3 skipped) |
| q06 | runs | none |
| q12, q14 | runs | D, D |
| q19 | runs | B |
| q03 | `LIMIT` | none |
| q10 | `LIMIT` | B |
| q13 | two-phase COUNT | none |
| q15 | two-phase MAX | D, C (aggregation target, not applied) |
| q02 | two-phase MIN, `LIMIT` | A ×4, B ×4, C |
| q04 | RIGHT SEMI, two-phase COUNT | none |
| q20 | RIGHT SEMI | D ×2, B |
| q21 | RIGHT SEMI, two-phase COUNT, `LIMIT` | B ×2 |
| q01 | two-phase AVG and COUNT | none |
| q17 | two-phase AVG | D (Starburst's 3× case) |
| q22 | two-phase AVG and COUNT | D |
| q16 | column-layout bug (its `COUNT(DISTINCT)` is planned as two dedupe steps plus a one-phase count, which we already accept) | B |
| q18 | column-layout bug, `LIMIT` | none |
| q11 | its wrong result (0 rows, no error) | B ×4 |

## The order

Each milestone is one or more PRs, stacked on #2062. A milestone ends with:
- the 22-query run at SF1000, checked against DuckDB, with a leak check after every query;
- the scoreboard from M0.

The counts are upper bounds. A query can still hit a gap no one has seen yet, and the run after each milestone gives the real number.

| # | work | new queries running | passing | filters applied in queries that complete | what it proves |
|---|---|---|---|---|---|
| today | | | 8/22 | 12 | |
| **M0** | Scoreboard; `LIMIT` (G3); RF Step 1 | q03, q10 | ≤ 10 | 13 | #2062's filters, measured properly |
| **M1** | Two-phase COUNT/MIN/MAX (G1, part 1) | q13, q15, q02 | ≤ 13 | 17 | q02's 4 broadcast filters work with no filter change |
| **M2** | RIGHT SEMI join (G2) | q20, q04, q21 | ≤ 16 | 17 | test bed for class D and B grows |
| **M3** | RF Step 2: partitioned-join filters | | ≤ 16 | 22 | class D on q12, q14, q15, q20 |
| **M4** | Two-phase AVG (G1, part 2) | q01, q17, q22 | ≤ 19 | 24 | q17 and q22 arrive already filtered |
| **M5** | Column-layout bugs (G4); q11 (G5) | q16, q18, q11 | ≤ 22 | 24 | every class B query now runs |
| **M6** | RF Step 3 (injection into Sirius scans), then RF Step 4 (class B) | | ≤ 22 | 38 | class B on 7 queries; whether it pays off |
| **M7** | RF Step 5 (q02 rf6) | | ≤ 22 | 39 | the last applicable filter |

The other 3 of the 42 filters are skipped on purpose:
- q08 rf1 and q09 rf1: their keys cover every supplier;
- q15 rf0: it targets an aggregation node, which 4.1.3 doesn't filter either.

### M0: scoreboard, `LIMIT`, filter guards

**Scoreboard** (coverage Step 0). Without it, no later milestone can show a gain.
- Stable datasets on local disk (`/scratch` or `/raid`, not `/opt`).
- `harness/bench.sh --queries all --warmup N`. Report:
  - pass/fail with a refusal reason for each query;
  - the median of 3 warm runs;
  - a geomean over a **fixed** query set (today's 8), so that new queries don't move it. Report a second geomean over every query that passes.
- A CPU baseline on the same machine: StarRocks BEs, or DuckDB on the same parquet.
- **Filters on/off:** every query that passes runs again with `SIRIUS_CN_RUNTIME_FILTERS=0`. Both runs must match DuckDB. This is the correctness gate for every filter step: a wrong filter loses rows silently.
- **Per-filter effect:** for each applied filter, the rows its scan read and the rows and bytes its fragment shipped (from `shipping packed exchange hop`), with filters on and off. SF3000 timings vary a lot between runs (q05 has taken 28–41 s), so these row counts are the main evidence. Time comes second.

**`LIMIT`** (G3, [#1963](https://github.com/sirius-db/sirius/issues/1963), still open; `main` hasn't changed `TransformFetchOp`):
- **Stopgap:** the translator also writes the old `FetchRel.count`/`offset` fields (`node_translator.rs:241`), as the POC branch did.
- **Real fix, in #1963:** the consumer reads `count_expr`/`offset_expr` when set and falls back to `count`/`offset`.
- **Test:** `ORDER BY … LIMIT` keeps its limit through the consumer.
- **Unlocks** q03 and q10. If q10 still fails with "input stream declared but not read", that's a separate bug and goes to M5.

**RF Step 1:**
- the guards: null-safe joins, skew joins, targets that aren't scans;
- one log line per filter planned, applied or skipped, with the reason and the rows cut (the scoreboard reads it);
- density judged on distinct keys, which recovers q05 rf1;
- fragment dumps named per CN.

**Start the RF Step 3 engine design now.** M6 depends on it, and it has the longest lead time.

### M1: two-phase COUNT, MIN, MAX

[multi-phase-aggregation.md](design/multi-phase-aggregation.md) Step 1: port the single-column rules from `partial_state.rs`.
- Derive the type sent across the exchange from the function and phase, on both sides. Never take it from the slot type, which can differ from what a node emits (a partial `avg` slot is `VARBINARY`).
- Expect q13, q15 and q02 (q02 also needs M0's `LIMIT`).
- **Filter check:** q02's 4 broadcast filters (rf1, rf2, rf7, rf8) already apply today, but q02 fails later. Check that they apply and that q02 matches with filters on and off.

### M2: RIGHT SEMI join

[tpch-coverage-plan.md](tpch-coverage-plan.md) G2: map `RIGHT_SEMI_JOIN` in `translate_hash_join`, then re-check q21's RIGHT ANTI with extra conditions.
- Expect q20, q04 and q21 (q04 and q21 also need M1's COUNT; q21 also needs `LIMIT`).
- **The `broadcast_row_limit` experiment goes here.** Four queries with partitioned joins now run (q12, q14, q15, q20). Raise the limit and see how many of them become broadcast joins that #2062 already filters. That sizes M3.

### M3: partitioned-join filters

RF Step 2: each CN sends its share of the build keys to every probing CN over the direct exchange, and each prober combines all the shares before its scan runs.
- **Shown on** q12, q14, q15 and q20: 5 filters, against 2 queries if it landed today.
- **Measure:** rows each probing scan ships before and after, filters on vs off. The exact-key size limit is decided here, with q12's `orders` filter (about 30M keys).

### M4: two-phase AVG

[multi-phase-aggregation.md](design/multi-phase-aggregation.md) Step 2: AVG travels as a sum and a count.
- Expect q01, q17 and q22. q17 and q22 arrive with M3's filters already working.
- **Headline:** q17 with filters on vs off, next to Starburst's 3× regression without dynamic filtering.
- If q17 matters more than M3's head start, swap M3 and M4.

### M5: column-layout bugs and q11

[tpch-coverage-plan.md](tpch-coverage-plan.md) G4 and G5.
- **q16:** not a three-phase plan. The recorded plan shows two dedupe aggregations with no functions, then a one-phase `count`, and the classifier already accepts that. Only the descriptor error is left.
- **q18:** a layout bug; it also needs `LIMIT`.
- **q11:** a wrong result. Start from its recorded plan; exact decimals may be the cause.
- **After M5,** all 7 queries with class B filters run: q19, q10, q02, q20, q21, q16 and q11.

### M6: injection, then same-fragment filters

- **RF Step 3:** an FFI call that pushes a filter into a scan's Sirius dynamic-filter set. Switch classes A and D to it, keeping the semi-join as a fallback until the injected path is at least as fast.
- **RF Step 4:** each ready receiver injects its own build's keys into its scan. That covers 14 filters on 7 running queries.
- **The open question gets answered here,** on 7 queries instead of 1: do these filters pay off when the keys aren't clustered in the files? If not, add the selectivity floor.

### M7: q02 rf6

RF Step 5: defer a non-leaf fragment, with a deadlock check over the query's fragment graph.

### After M7

- **CPU fallback** ([design/cpu-fallback-and-deployment.md](design/cpu-fallback-and-deployment.md)): no query fails for coverage.
- **SF3000 memory and cancellation:** q09 in sequence, and freeing memory promptly after a cancel.
- **RF Step 6:** the StarRocks filter protocol, only for clusters that mix CNs and BEs.
- **Three- and four-phase aggregation** ([design/multi-phase-aggregation.md](design/multi-phase-aggregation.md) Step 3): customer SQL only, since TPC-H never needs it. Two things to do there:
  - classify each function as update or merge, not the whole node;
  - try `new_planner_agg_stage = 2`, which limits three- and four-phase plans to the four cases that can't be two-phase.

## Parallel tracks

| track | milestones | who |
|---|---|---|
| Translator coverage | M0 `LIMIT`, M1, M2, M4, M5 | us |
| CN runtime filters | M0 RF Step 1, M3, M7 | us |
| Engine filter injection | RF Step 3 design from M0, code by M6 | with the engine owners |
| Measurement | M0, then rerun at every milestone | us |

The coverage and filter tracks touch different code (`node_translator.rs` / `agg_phase.rs` against `compute_node_service.rs` / `runtime_filters.rs`). They can run side by side, as long as every milestone runs the same 22-query scoreboard.
