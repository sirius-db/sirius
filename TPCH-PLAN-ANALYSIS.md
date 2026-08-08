# TPC-H plan analysis — Sirius GPU CNs behind the StarRocks FE (SF1, 2 CNs)

Plan-level analysis of all 22 TPC-H queries as this FE plans them for Sirius compute
nodes, cross-referenced with the 2026-08-07 A/B sweep. Companion docs:
`ROADMAP-8CN-TPCH.md`, `OPEN-ISSUES.md`, `QUERY-TIMEOUT-ANALYSIS.md`, `TPCH-SURVEY.md`.

## 1. Assumptions & environment

- **FE**: version 4.1.1-14b7e3f (from `SHOW VARIABLES LIKE '%version%'` →
  `version_comment`). `SELECT current_version()` errors on this cluster: the
  constant-only query is routed to a Sirius CN, which rejects it with "unsupported plan
  node TPlanNodeType(19) at node 0: plan node is outside the v1 StarRocks slice" — FE
  version recorded via `version_comment` instead.
- **Sirius**: repo HEAD `4bf24dff3503af6d7c8bde03e45e65e565156889` (branch
  `demo-multi-cn`).
- **Data**: TPC-H SF1 parquet accessed via `FILES()` external tables — the FE has no
  statistics (every plan node reports `cardinality: 1`), which shapes every join-siding
  and cross-join decision below.
- **Topology**: 2 Sirius CNs, one engine thread per CN (fragments execute serially per
  CN); exchanges = packed-cuDF over nixl (cross-CN) or local relay, with a per-CN
  staging arena (1280 MiB); scans = parquet byte-range splits assigned by the FE.
- **Plan capture**: 47 files in
  `experimental/starrocks/benchmarks/tpch/plans/`: `qNN.explain.txt` (22, `EXPLAIN`)
  and `qNN.verbose.txt` (22, `EXPLAIN VERBOSE`) for q01–q22, all non-empty; plus
  `q01.logical.txt` (`EXPLAIN LOGICAL`, 37 lines), `q01.costs.txt` (`EXPLAIN COSTS`,
  128 lines), `q01.analyze.txt` (`EXPLAIN ANALYZE`, 88 lines, query executed) — all
  three extra modes work against this FE with external `FILES()` tables. Sanity
  spot-checks: q01 explain has 3 PLAN FRAGMENTs / 4 EXCHANGE lines; q09 and q21 each
  have 9 fragments / 16 EXCHANGE lines, incl. MERGING-EXCHANGE and AGGREGATE (merge
  finalize) shapes. Cluster torn down after capture: 0 sirius-starrocks-cn/StarRocksFE
  processes, nvidia-smi 0 MiB.
- **Caveat on EXPLAIN ANALYZE**: `q01.analyze.txt` succeeded but its per-operator
  runtime metrics are placeholders (TotalTime 0ns/NaN%, OutputRows ?, with ANSI escape
  codes) because Sirius CNs do not populate the StarRocks runtime profile; the plan
  tree structure in it is still usable.
- **Timings**: warm medians of 3 runs (1 discarded warm-up) from
  `benchmarks/tpch/results/sf1-2026-08-07-A.csv` (engine A = Sirius CNs) and
  `-B.csv` (engine B = stock StarRocks CPU BEs). All 22 pass.

## 2. Workload summary

**Fragment counts** (per `EXPLAIN`): min 2 (q06), max 14 (q02), median 6, total 157.

| fragments | queries |
|---|---|
| 2–4 | q06(2), q01(3), q04, q14, q19(4) |
| 5–7 | q03, q12, q13, q16, q17(5), q10, q22(6), q15, q18(7) |
| 8–11 | q20(8), q09, q11, q21(9), q07(10), q05(11) |
| 13–14 | q08(13), q02(14) |

**Exchange census** — 129 exchange edges across the 22 plans:

| type | count | notes |
|---|---|---|
| SHUFFLE (hash) | 68 | keys mostly orderkey/partkey/suppkey/custkey (uniform); a handful on 2–7-value keys (skew: q02 n_regionkey, q07 name-pair, q08 year) |
| BROADCAST | 36 | dest-0 zero-copy + clone per extra CN; several are mis-sided full-table builds (q04, q07, q09, q16, q22) |
| GATHER (plain) | 7 | q06, q11, q14, q15, q17, q19, q22 scalar/partial gathers |
| MERGING-EXCHANGE (gather) | 18 | every query except q06/q14/q17/q19 ends in one; Sirius lowers it to stream-read + full SortRel re-sort at the gather CN |

**Sorts**: every explicit SORT/TOP-N in the suite is small — the largest is q16's
18,314-row 4-key sort; TOP-Ns (q02, q03, q10, q18, q21) run post-aggregation over
≤114k grouped rows with limits 10–100, and the FE always confines them below the
merging gather. The suite's real sort volume is *hidden inside aggregation*: since
`312e4535`, every FP64-lowered grouped SUM/AVG runs the canonical-order sort-based
cuDF groupby (the atomics-free determinism workaround for #29), i.e. full sorts of
5.9M rows (q01), 6M (q17 avg, q18), ~910k×2 (q20), and four sorts in q15.

**Aggregations**: textbook two-phase partial/merge with a hash exchange on the group
keys, except where the input is already partitioned on the grouping key (q03, q13
first level, q18 final) — then single-phase. `avg` is expanded to sum+count measures
(`bd232c40`). Integer counts (q04, q12, q13, q21) are exact and exempt from #24;
every decimal SUM/AVG is lowered to FP64 (#24) — 0.1–0.4% low-biased, and it is what
triggers the canonical-sort path. Heavy aggregation concentrates on lineitem
group-bys: q01, q15 (×2), q17, q18, q20.

**Joins**: 65 total (62 hash, 3 nestloop — q08 cross join lowered to a
synthetic-constant single-key hash join, q11 and q22 scalar-threshold nestloops).
Because `FILES()` scans are stats-blind, the FE frequently sides builds wrong:
full-lineitem or full-orders builds probed by tiny sides (q04, q05, q07, q08, q09,
q16, q18, q22). Join-heavy fragments concentrate where 3–4 exchange streams
rendezvous (q05 F5/F2, q08 F5, q09 F2, q18 F3, q21 F4). The FE plans runtime filters
on nearly every scan; Sirius's translator executes **none** of them, so probe volumes
that engine B prunes 30–100× cross the exchanges in full (q03, q08, q12, q17, q18,
q20, q21).

**The fixed floor**: q06 (2 fragments, 1 gather, trivial compute) medians 308ms —
that is the per-query floor: FE dispatch round trips, serial fragment setup,
translate, staging-arena setup, gather, `fetch_data`. At SF1 the floor is 25–75% of
nearly every query's wall time; fragment-heavy small-data plans (q02 14 frags, q11 9,
q22 6) pay it multiplied and post the worst A/B ratios (4.1–5.7× slower). Only where
real per-row work amortizes it (q01, q09, q19, and nearly q13/q21) does the GPU tie
or win. Suite geo-mean: B is ~2× faster (0.48x, per OPEN-ISSUES M3); A wins q01, q09, q19.

## 3. Per-query analysis

### Q1 — pricing summary report
- **Shape**: single-table: lineitem (~5.9M rows after `l_shipdate <=`), 4 sums +
  3 avgs + count grouped by (l_returnflag, l_linestatus) → 4 groups, ordered.
- **Plan**: 3 fragments. F00 scan+project+partial AGGREGATE (STREAMING); F01 merge
  finalize + SORT; F02 MERGING-EXCHANGE root. 2 exchanges: SHUFFLE on the two flag
  keys (~4 groups × 8 partial states per CN), MERGING gather of 4 rows.
- **Sorts**: explicit sorts touch ≤4 rows. Real sort: the canonical-order sort-based
  groupby over 5.9M rows inside the partial aggregate (FP64-lowered sums, #24/#29).
- **Aggs**: two-phase; avg via sum+count VARBINARY expansion; FE types exact
  DECIMAL128, lowered to FP64 (#24); count(*) exact.
- **Joins**: none. **Distribution**: byte-range splits of the 155MB lineitem parquet;
  4 groups → skew irrelevant. Group keys typed VARCHAR(1048576) though 1-char flags.
- **Bottleneck**: GPU-compute-bound on the partial aggregation, floor amortized.
  A 418ms vs B 522ms (one of three A wins); q06=308ms ⇒ the grouped canonical-sort
  aggregation + string keys ≈ the ~110ms delta; exchanges move ~8 rows.
- **Opportunities**: (1) #24 decimal-native SUM — named acceptance query; exact
  fixed_point atomics (`device_aggregators.cuh:126`) remove both the 0.096% drift and
  the canonical-sort tax. (2) Dictionary/narrow-type the 1-char group keys to int8
  before the groupby. (3) Floor trimming still ~70% of runtime.
- **Priority**: agg high; sort medium (artifact of agg); join –; other medium.

### Q2 — minimum-cost supplier
- **Shape**: part×supplier×partsupp×nation×region (EUROPE/'%BRASS'/size 15) joined
  against a correlated min(ps_supplycost) subquery; TOP-N 100.
- **Plan**: 14 fragments (suite max), 13 exchanges: 7 SHUFFLE, 5 BROADCAST,
  1 MERGING. Six fragments are bare scan→sink over dwarf tables (nation 25 rows ×2,
  region 5 ×2, supplier ×2); every exchange pays the full pack→staging-lease→nixl
  WRITE→transmit round trip.
- **Sorts**: TOP-N 100 over ~460 rows + a ≤200-row gather re-sort — trivial.
- **Aggs**: two-phase grouped min(ps_supplycost) by ps_partkey over ~160k rows;
  order-insensitive → no canonical tax, no #24 drift (the `ps_supplycost = min`
  equality compares identically-lowered FP64; held 3/3).
- **Joins**: 8 hash joins (3 PARTITIONED, 5 BROADCAST incl. the min-result build).
  Canonical reproducer of the empty-broadcast-build BUILD_PROBE wedge (fixed
  `59ce6662`).
- **Distribution**: real skew — the subquery join partitions on
  n_regionkey/r_regionkey (5 values, 1 carries data after EUROPE) so the subquery
  half lands on one CN. Data volumes tiny (<50MB total); parallelism loss, not
  capacity.
- **Bottleneck**: fixed-overhead-bound, the clearest case. A 1138ms vs B 229ms
  (4.97×): ~20+ serial fragment dispatches + 13 exchange round trips over <50MB.
- **Opportunities**: (1) leaf-fragment fusion / tiny-broadcast inline fast path;
  (2) per-peer async send workers (ROADMAP-8CN 4a) — this plan pays exchange latency
  13 times serially; (3) low-cardinality partition-key skew fallback to broadcast;
  (4) keep the 59ce6662 regression cover.
- **Priority**: other high; join high (count not size); sort/agg low.

### Q3 — shipping priority
- **Shape**: BUILDING customer × orders(<1995-03-15) × lineitem(>1995-03-15),
  sum(revenue) by (l_orderkey, o_orderdate, o_shippriority), TOP-N 10.
- **Plan**: 5 fragments, 4 exchanges: SHUFFLE l_orderkey (~3.2M lineitem rows × 3
  cols, ~77MB packed — the suite's largest single shuffle), BROADCAST BUILDING
  custkeys (~30k), SHUFFLE o_orderkey (~727k), MERGING gather limit 10.
- **Sorts**: per-CN TOP-N 10 over ~114k groups (full sort + fetch); 20-row gather
  re-sort. This sort tuple was the Class-B column-order refusal (fixed `4323197d`).
- **Aggs**: single-phase `update finalize` (input already partitioned on l_orderkey —
  the exchange-free agg shape). FP64-lowered sum ⇒ canonical-sort groupby over ~590k
  joined rows; the worse of the two out-of-band #24 queries (−0.30%/−0.39% rows).
- **Joins**: BROADCAST o_custkey=c_custkey (cheap, well-sided); PARTITIONED
  l_orderkey=o_orderkey (3.2M probe × 727k build). FE plans remote runtime filter
  filter_id=1 (o_orderkey → lineitem scan) — ignored by Sirius; B wins with it.
- **Distribution**: high-cardinality keys, no skew; the lineitem shuffle transits the
  arena and scales linearly with SF.
- **Bottleneck**: exchange-bound over the floor. A 500ms vs B 295ms (1.69×).
- **Opportunities**: (1) remote runtime filters — flagship query, ~5× less exchange
  volume; (2) #24 (correctness + canonical-sort removal); (3) compute the revenue
  multiply sender-side (~25% narrower shuffle); (4) agg shape already optimal.
- **Priority**: other high; agg medium (correctness); join medium; sort low.

### Q4 — order priority checking
- **Shape**: count orders in Q3-1993 with EXISTS(late lineitem), group by
  o_orderpriority (5 groups), ordered.
- **Plan**: 4 fragments, 3 exchanges: BROADCAST of the filtered lineitem l_orderkey
  column (~3.8M BIGINTs, ~30MB — the cost center), SHUFFLE of 5 count rows, MERGING
  gather of 5 rows.
- **Sorts**: ≤5 rows twice — nil.
- **Aggs**: two-phase grouped count(*), exact BIGINT — no #24, no canonical tax;
  verified exact vs DuckDB.
- **Joins**: one LEFT SEMI BROADCAST, sided backwards by the stats-blind FE: each CN
  builds a ~3.8M-key GPU hash table (un-deduplicated, duplicated per CN by the
  broadcast) to probe ~53k orders.
- **Distribution**: broadcast un-parallelizes the build (each CN materializes both
  halves); grows with SF and with (N−1) clones at 8 CNs.
- **Bottleneck**: floor + broadcast-build. A 428ms vs B 252ms (1.70×).
- **Opportunities**: (1) semi-join build-side dedup below the broadcast sink
  (3.8M → ~1.3M distinct keys, CN-side only); (2) engine-side semi-join side swap
  when build ≫ probe; (3) dedup is the mitigation that scales to 8 CNs.
- **Priority**: join high; other medium; sort/agg low.

### Q5 — local supplier volume
- **Shape**: 6-table join (customer⋈orders⋈lineitem⋈supplier⋈nation⋈region), ASIA +
  1994, sum(revenue) by n_name → 5 rows.
- **Plan**: 11 fragments, 10 exchanges (7 SHUFFLE incl. the **full 6M-row lineitem**
  on l_orderkey, 2 BROADCAST, 1 MERGING gather).
- **Sorts**: 5-row final sort; canonical-sort tax on both agg phases (FP64 sums).
- **Aggs**: two-phase sum by n_name; DECIMAL128 planned, FP64 executed (#24,
  0.1–0.2% low).
- **Joins**: 5 hash joins, all FE-cardinality-1: the o_orderkey=l_orderkey PARTITIONED
  join builds on the **full 6M lineitem** to probe ~227k c⋈o rows (inverted); supplier
  10k / nation 25 / region 1-row broadcasts.
- **Distribution**: skew hazard on s_nationkey (5 surviving nations) and n_name
  (5 groups) shuffles with 2 CNs; runtime filters 0–5 planned, none executed.
- **Bottleneck**: orchestration + exchange. A 1026ms vs B 320ms (3.21×): ~300ms floor
  + 10 exchange stagings + 11 serial fragments + a 6M-row shuffle/hash build that RF
  pruning and stats would have avoided.
- **Opportunities**: (1) runtime-filter execution (shrink the lineitem shuffle);
  (2) translator build-side inversion for lineitem-class builds; (3) fragment
  pipelining; (4) #24; (5) tiny-dimension broadcast elision via local relay.
- **Priority**: join high; other high; sort/agg low.

### Q6 — forecast revenue change
- **Shape**: scalar sum over filtered lineitem; 1 row. Simplest plan: 2 fragments,
  1 GATHER carrying one partial-sum row per CN.
- **Sorts**: none (scalar agg — no canonical-sort path). **Joins**: none.
- **Aggs**: two-phase scalar sum; FP64-lowered (#24), ~0.1% low, within tolerance.
- **Distribution**: byte-range splits; reduction to 1 row per CN before the exchange.
- **Bottleneck**: pure fixed floor. A 308ms vs B 220ms (1.40×) — A's 308ms **is** the
  floor; ~114k qualifying rows cost single-digit ms on GPU.
- **Opportunities**: (1) the floor-measurement query — profile and shave the fixed
  path (dispatch → translate → enqueue → gather → fetch_data); (2) #24's named
  "start here" target (SUM(DECIMAL64) end-to-end, bit-exact gate); (3) verify pushed
  predicates reach parquet row-group/page pruning.
- **Priority**: other high; sort/agg/join low.

### Q7 — volume shipping (FRANCE↔GERMANY)
- **Shape**: supplier⋈lineitem⋈orders⋈customer⋈nation×2, group by (supp_nation,
  cust_nation, year), 4 rows.
- **Plan**: 10 fragments, 9 exchanges (5 SHUFFLE incl. ~1.8M shipdate-filtered
  lineitem, 3 BROADCAST incl. the **full 1.5M-row orders**, 1 MERGING).
- **Sorts**: 4-row final; canonical string sort (two VARCHAR(1048576) n_name keys) on
  both agg phases.
- **Aggs**: two-phase, 3-key groupby; `year()` is the historical SMALLINT-vs-BIGINT
  hang (fixed `4beca977`); FP64-lowered sums (#24).
- **Joins**: 5; two badly shaped: build = 1.8M lineitem probed by 10k suppliers
  (inverted), and a 1.5M-row orders broadcast hash build per CN probed by ~145k rows.
  FR/DE filter applies only at the last join.
- **Distribution**: partial-agg shuffle on a 4-combination key → merge can land on one
  CN; broadcast clones scale with CN count; RFs planned, not executed.
- **Bottleneck**: exchange/orchestration + two avoidable heavy builds. A 934ms vs
  B 328ms (2.85×).
- **Opportunities**: (1) build-side inversion (largest arithmetic win); (2) nation
  filter pushdown / runtime-filter execution (lineitem to ~2/25); (3) fragment
  pipelining; (4) #24 removes the string-keyed canonical sorts; (5) share one
  broadcast clone across the three consecutive broadcast joins.
- **Priority**: join high; other high; agg medium; sort low.

### Q8 — national market share
- **Shape**: 8-table join incl. part and two nation roles; conditional/total sum
  ratio by year; 2 rows.
- **Plan**: 13 fragments, 12 exchanges (7 SHUFFLE, 4 BROADCAST, 1 MERGING). The
  pathology: part(~1.3k filtered) × supplier(10k broadcast) as a NESTLOOP CROSS JOIN
  → ~13.3M-row Cartesian intermediate (F11 = half the plan's total cost), shuffled on
  two keys; plus the **full 6M lineitem** shuffled and hash-built un-pruned.
- **Sorts**: 2-row final; canonical tax on both agg phases.
- **Aggs**: two-phase, 2 measures, group by year (2 distinct values → merge pinned to
  ≤2 partitions); mkt_share division post-merge; CASE→if(), 5 common exprs
  (CLONE_EXPR); FP64 (#24).
- **Joins**: 7. The cross join lowers to "equality join on synthetic constants"
  (node_translator.rs:1832) — every row collides on one hash key (maximal skew).
  Runtime filter 0 (l_partkey) would cut lineitem 6M → ~40k; ignored.
- **Bottleneck**: mixed real data movement + orchestration. A 1236ms (A's slowest) vs
  B 472ms (2.62×). B runs the same shape but prunes lineitem at scan via RF.
- **Opportunities**: (1) runtime-filter execution — worth the most here; (2) real
  `cudf::cross_join` instead of the synthetic-constant funnel; (3) better: re-associate
  cross-join-feeding-a-two-key-join in the translator (kills the 13.3M intermediate);
  (4) build-side inversion at the 2-key join; (5) fragment pipelining (13 serial
  fragments); (6) #24.
- **Priority**: join high; other high; sort/agg low.

### Q9 — product type profit
- **Shape**: 6-table join filtered p_name LIKE '%green%', profit sum by
  (n_name, year) → 175 rows.
- **Plan**: 9 fragments, 8 exchanges. F7: part(10.7k) × supplier(10k broadcast)
  NESTLOOP CROSS → **~107M (s_suppkey,p_partkey) pairs**, SHUFFLE'd (~1.3GB — the
  documented single ~648MB staging-arena lease); F2 fans in 4 exchange inputs:
  lineitem 6M shuffle, partsupp 800k shuffle, **full 1.5M orders broadcast** (no
  filter exists), nation broadcast.
- **Sorts**: 175-row sort + order-preserving merge; canonical-sort rider on the
  600k-row groupby input.
- **Aggs**: two-phase (n_name, year); 600k → 175 groups — excellent reduction;
  FP64-lowered (#24).
- **Joins**: 5; dominant operator = the 2-key PARTITIONED join probing 107M pairs into
  a 6M lineitem build (BUILD_PROBE). 6 RFs planned, all ignored — and per-key blooms
  can't prune a pair domain anyway.
- **Distribution**: 2-key composites near-uniform; F2's serial engine thread drains 4
  input exchanges sequentially. Arena-defining query: peak lease scales ~linearly
  with SF (OPEN-ISSUES M2/M3).
- **Bottleneck**: throughput-bound on a plan-inflicted cost — both engines pay the
  stats-blind cross-join plan. A 1104ms vs B 1181ms (0.93× — GPU ties/wins because
  real work finally amortizes the floor); both engines' slowest query.
- **Opportunities**: (1) M2 arena auto-sizing + chunked multi-lease export
  (prerequisite for SF10); (2) a **pair-hash** runtime filter applied inside the
  cross-join output (~15× smaller dominant exchange) — the one RF variant that pays
  here; (3) overlap build ingestion in F2; (4) #24 (minor); (5) FE-side stats/hints
  would beat all of the above.
- **Priority**: join high; other high (arena, 1.3GB shuffle); sort/agg low.

### Q10 — returned item reporting
- **Shape**: customer⋈orders⋈lineitem('R')⋈nation, sum(revenue) by 7 columns,
  TOP-N 20 by revenue.
- **Plan**: 6 fragments, 5 exchanges (3 BROADCAST — broadcast-only join set, 1 SHUFFLE
  on **all seven group columns**, 1 MERGING limit 20). The 7-key shuffle carries ~115k
  partial rows where 5 VARCHAR(1048576) columns dominate the packed bytes.
- **Sorts**: TOP-N 20 executed as a full sort of ~38k merged groups per CN (no top-n
  pushdown); the real sort cost is the canonical-sort tax over 7 mixed keys
  (5 strings), paid in **both** agg phases.
- **Aggs**: two-phase, 7 keys, ~115k → ~38k groups (weak ~3× partial reduction — the
  wide-string payload crosses the wire nearly whole); FP64-lowered (#24); q10 is one
  of the two out-of-band drift queries (custkeys 143347/146149 + 3-rank rotation).
- **Joins**: 3 BROADCAST inner joins, all small builds; RFs (o_orderkey → lineitem
  scan etc.) ignored — the 1.48M probe could have been ~230k.
- **Bottleneck**: overhead + sort. A 634ms vs B 323ms (1.96×): floor + double
  canonical sort on wide strings + wide-string shuffle + a full 38k sort for top-20.
- **Opportunities**: (1) #24 — headline fix (acceptance query; removes drift and the
  canonical-sort path); (2) group-key reduction: c_custkey functionally determines the
  other 6 — partition on custkey alone, aggregate on custkey with first-value payload
  carry; (3) GPU top-k instead of full sort; (4) honor the o_orderkey remote RF.
- **Priority**: agg high; sort medium; other medium; join low.

### Q11 — important stock identification
- **Shape**: two parallel partsupp⋈supplier⋈nation(GERMANY) pipelines (grouped by
  ps_partkey + scalar total), threshold nestloop, ORDER BY value DESC → 1048 rows.
- **Plan**: 9 fragments, 8 exchanges (5 BROADCAST, 1 SHUFFLE, 1 GATHER, 1 MERGING) to
  move **under 2MB total**; partsupp scanned twice (no CTE reuse).
- **Sorts**: 1048-row sort — trivial; FP64 sort key.
- **Aggs**: four AGGREGATE nodes (grouped two-phase with near-zero merge reduction
  ~32k→29.8k, plus scalar two-phase ×0.0001); HAVING compares through explicit
  CAST-to-DOUBLE — tolerant, no q15-style equality race.
- **Joins**: 4 broadcast hash joins (builds 1–400 rows) + 1 nestloop vs a 1-row
  threshold.
- **Bottleneck**: fixed-overhead, **worst ratio of the suite**: A 830ms vs B 147ms
  (5.65×). 9 fragments + 8 exchanges over ~75MB scanned; the floor alone is 2× B's
  total.
- **Opportunities**: (1) MULTI_CAST_DATA_STREAM_SINK (#31.1) — enables
  `cbo_cte_force_reuse`, halving scans/joins/fragments for this textbook shared
  subplan (today that plan shape silently hangs — the let-else `Ok(Vec::new())`
  defect, so it is a correctness item too); (2) per-fragment floor attack + tiny
  broadcast local-relay fast path; (3) leaf-fragment fusion into the consumer's build
  side (removes 4 of 9 fragments); (4) #24 (free once landed).
- **Priority**: other high; agg medium (per-node overhead); sort/join low.

### Q12 — shipping modes
- **Shape**: filtered lineitem (~31k) ⋈ orders on orderkey; two conditional integer
  sums by l_shipmode (2 groups).
- **Plan**: 5 fragments, 4 exchanges — shuffle-only. Dominant move: the **entire 1.5M
  orders table** shuffled on o_orderkey (the plan's whole reduction strategy is remote
  RF 0, which Sirius ignores; B honors it and ships ~31k rows).
- **Sorts**: 2 rows. **Aggs**: two-phase integer sums — exempt from #24 and the
  canonical tax; cheapest agg profile.
- **Joins**: 1 PARTITIONED, build = 31k lineitem, probe = 1.5M orders (well-sided;
  the problem is feeding it, not joining it).
- **Bottleneck**: exchange/floor. A 469ms vs B 394ms (1.19× — near parity because B is
  mostly floor too).
- **Opportunities**: (1) **remote runtime filters** — 50× smaller orders shuffle;
  generalizes to every PARTITIONED-join plan; (2) fallback: ship distinct build keys
  back to the probe-scan fragment (Sirius-internal semi-join pushdown); (3) column
  pruning/late materialization on o_orderpriority; (4) with RFs, q12 becomes a pure
  floor measurement.
- **Priority**: other high; join medium; sort/agg low.

### Q13 — customer distribution
- **Shape**: customer LEFT OUTER orders (NOT LIKE '%special%requests%'), count per
  custkey, histogram by count → 42 rows.
- **Plan**: 5 fragments, 4 exchanges (orders ~1.48M and customer 150k shuffles to a
  colocated RIGHT OUTER join + single-phase count; two-phase histogram above).
- **Sorts**: 42 rows, distributed sort + merging gather.
- **Aggs**: 150k-group count + histogram — all integer, no canonical tax.
- **Joins**: 1 RIGHT OUTER PARTITIONED (1.48M × 150k; must emit ~50k unmatched);
  PARTITIONED forced correct; RF from unfiltered customer has ~1.0 selectivity (dead
  weight if honored).
- **Bottleneck**: fixed floor. A 450ms vs B 349ms (1.29× — A's best non-win ratio).
  Floor decomposition: A ≈ 300 floor + 150 work; B ≈ 100 + 250 — **A's marginal GPU
  work already beats B's CPU work**; the deficit is entirely dispatch/exchange floor.
- **Opportunities**: (1) fragment overlap (F00/F02 scans are independent but
  serialize) — cleanest proof case; (2) adaptive RF disable on ~1.0 selectivity;
  (3) dedicated two-literal substring-scan kernel for the NOT LIKE over 1.5M comments;
  (4) longer-term: sender-side partial count below the shuffle.
- **Priority**: other high; agg/join medium; sort low.

### Q14 — promotion effect
- **Shape**: one-month lineitem ⋈ part, two ungrouped decimal sums + division; 1 row.
- **Plan**: 4 fragments, 3 exchanges: lineitem ~72k SHUFFLE, part 200k SHUFFLE
  (probe 3× smaller than build — FE chose PARTITIONED blind; broadcast of the 72k
  side would beat the dual shuffle), 2-row GATHER.
- **Sorts**: none anywhere. **Aggs**: two-phase ungrouped sums — no canonical tax;
  FP64-lowered, +0.0023% measured (#24 acceptance query). CLONE_EXPR/common_slot_map
  shape that blocked q14 until `fe236e8b`.
- **Joins**: 1 PARTITIONED (200k build / 72k probe); part→lineitem RF ~1.0 selectivity.
- **Bottleneck**: pure overhead. A 428ms vs B 220ms (1.95×): marginal work is equal
  (~130ms each); the gap is A's fixed per-query cost + the full 6M-row parquet scan.
- **Opportunities**: (1) fuse the gather-merge fragment; overlap the two scans;
  (2) adaptive exchange downgrade (HASH→BROADCAST + build flip when a source drains
  tiny); (3) #24 (bit-exactness, removes the 4-cast FP64 chain); (4) parquet row-group
  min/max skipping on l_shipdate; (5) zero-copy views for carried common-expr slots.
- **Priority**: other high; join medium; sort/agg low.

### Q15 — top supplier
- **Shape**: revenue CTE (grouped sum by l_suppkey over 3-month lineitem) evaluated
  **twice** (no CTE reuse), scalar max, equality join back, 1 row.
- **Plan**: 7 fragments, 6 exchanges; a ~5-stage serial dependency chain; both CTE
  copies read every lineitem split.
- **Sorts**: explicit sorts ≤1 row; the hidden canonical sort is paid **four times**
  (two partials over ~214k rows, two merges) — this query's signature tax (the
  `312e4535` workaround exists because of q15's `total_revenue = max(...)` FP64
  equality flake, #29).
- **Aggs**: grouped FP64 sum by l_suppkey ×2 (two-phase each) + ungrouped max chain;
  0.043% drift (#24).
- **Joins**: 2, both cheap; the 1-row max broadcast + the s_suppkey join whose remote
  RF genuinely pays (supplier scan cut to ~1 row) — the one query where RFs already
  matter and are the FE's, not Sirius's.
- **Bottleneck**: redundant work + fragment serialization. A 681ms vs B 250ms (2.72×);
  ~230ms of structural extra work: double scan/agg/shuffle + 4 canonical sorts +
  deepest serial critical path.
- **Opportunities**: (1) MULTI_CAST_DATA_STREAM_SINK (#31.1) — "structurally removes
  q15's double-evaluation shape"; (2) #24 — acceptance criterion "q15 stays 8/8
  WITHOUT the canonical-sort path"; equality join exact by construction; (3) overlap
  independent fragments (F02/F04/F00); (4) compute max as a second reduction over the
  merged CTE.
- **Priority**: agg high; other high; sort medium (hidden ×4); join low.

### Q16 — parts/supplier relationship
- **Shape**: count(DISTINCT ps_suppkey) by (p_brand, p_type, p_size) over
  partsupp⋈part minus complaint suppliers (null-aware anti); 18,314 rows — largest
  result set.
- **Plan**: 5 fragments, 4 exchanges: **BROADCAST of all 800k partsupp rows** (wrong
  side — 25× bigger than the filtered part probe), BROADCAST of ~4 complaint
  suppliers, 3-key SHUFFLE (~119k dedup rows), MERGING gather of 18,314 rows.
- **Sorts**: 4-key sort (2 VARCHAR) over 18,314 rows — real but far from dominant.
- **Aggs**: count(DISTINCT) → three-node cascade; the partial dedup reduces ~120k →
  ~119k (near-zero — wasted 4-key varchar groupby); integer count, byte-identical to
  DuckDB.
- **Joins**: INNER BROADCAST with an 800k build per CN answering 30k probes (inverted
  25×); null-aware LEFT ANTI vs ~4 rows (correct, cheap).
- **Distribution**: worst per-N scaling of the small queries — broadcast bytes and
  build cost multiply by N.
- **Bottleneck**: floor + misplanned broadcast. A 458ms vs B 150ms (3.05×); B is near
  its own floor.
- **Opportunities**: (1) executor-side build/probe flip for symmetric INNER broadcast
  joins; (2) adaptive STREAMING passthrough when partial-dedup reduction ≈1.0 (stock
  StarRocks CPU does exactly this); (3) dictionary-encode p_brand/p_type;
  (4) fold the 4-key anti build into an IN-list/bloom predicate; (5) result-path
  throughput for the 18,314-row sink.
- **Priority**: other high; agg medium; join medium; sort low.

### Q17 — small-quantity-order revenue
- **Shape**: lineitem ⋈ part(Brand#23/MED BOX) where l_quantity < 0.2×avg(l_quantity)
  per part; scalar sum/7.
- **Plan**: 5 fragments, 4 exchanges; **lineitem scanned twice** (raw + for the
  grouped avg); the raw 6M-row shuffle is un-pruned because RF filter_id=0 (~204 part
  keys → ~100× pruning) is ignored.
- **Sorts**: none explicit; hidden canonical sort on both grouped-avg phases (a 6M-row
  sort by ~200k keys — the largest GPU compute item).
- **Aggs**: grouped avg (6M → 200k, sum+count expansion) + scalar sum; FP64-lowered.
- **Joins**: 2 hash joins with small builds (204 rows; 200k avg rows) + a non-equi
  residual (l_quantity < 0.2×avg) evaluated post-match in FP64.
- **Bottleneck**: fixed-overhead-dominated, GPU-compute second. A 469ms vs B 274ms
  (1.71×); ~170ms attributable: 2×6M scans, canonical sort, un-pruned shuffle.
- **Opportunities**: (1) cross-CN runtime filters (roadmap #10; feed the existing
  `src/op/dynamic_filter_publisher.cpp`) — highest-leverage single change here;
  (2) #24 for both avg phases (removes the sort + bias); (3) shared scan / MULTI_CAST
  to halve parquet IO; (4) floor work.
- **Priority**: agg high; other high; join medium; sort low.

### Q18 — large volume customers
- **Shape**: top-100 of customer⋈orders⋈lineitem grouped on 5 keys, restricted to
  orders with sum(l_quantity) > 300 (grouped-HAVING semi join).
- **Plan**: 7 fragments, 6 exchanges (~9M rows moved): orders 1.5M shuffle, lineitem
  sum-partials 1.5M shuffle, **raw 6M lineitem shuffle landing as a join build**,
  plus two small shuffles and a MERGING limit-100 gather. Lineitem scanned twice.
- **Sorts**: TOP-N over ~57 rows + limit-100 merging gather — trivial (verify the
  gather is a k-way merge, not concat+re-sort, before SF10). Hidden: the canonical
  sort on the 6M-row sum-by-l_orderkey partial — the query's largest kernel.
- **Aggs**: HAVING subquery two-phase sum by l_orderkey (6M → 1.5M groups → ~57
  survivors; FP64 canonical tax both phases) + a single-phase 5-key agg over ~400
  rows.
- **Joins**: LEFT SEMI with ~57-key build probing 1.5M orders (RF filter_id=0 = those
  57 keys would prune the orders scan to ~57 rows — ignored); INNER with a
  **6M-row build probed by 57 rows** (pathological inversion); INNER 150k×400 (also
  inverted, small).
- **Bottleneck**: GPU compute + network over the floor. A 621ms vs B 278ms (2.23×).
- **Opportunities**: (1) runtime filters — flagship: a 57-key exact filter collapses
  every downstream exchange; (2) size-based build/probe swap (extends the 59ce6662
  BUILD_PROBE election); (3) #24 kills the 6M canonical sort; (4) limit-aware merge
  audit; (5) RF pruning is the only realistic lever on the second scan.
- **Priority**: agg high; join high; other high; sort low.

### Q19 — discounted revenue
- **Shape**: scalar sum over lineitem ⋈ part with a 3-arm OR of
  brand/container/quantity/size predicates + common shipmode/shipinstruct conjuncts.
- **Plan**: 4 fragments, 3 exchanges (two ~200k-row shuffles + scalar gather) — the
  lightest exchange profile; scan conjuncts pre-prune lineitem 28× before the shuffle.
- **Sorts**: none, and no canonical-sort exposure (scalar aggs only).
- **Aggs**: scalar two-phase sum; FP64-lowered — accuracy, not speed, is the exposure.
- **Joins**: 1 PARTITIONED, balanced sides (~214k × 200k), carrying the heavy 3-arm OR
  residual as data-parallel column ops — the source of the GPU win.
- **Bottleneck**: floor-bound on A, **CPU-bound on B — A wins**: 398ms vs 478ms
  (0.83×). The branch-heavy OR that costs the CPU vectorizer is flat boolean kernels
  on GPU.
- **Opportunities**: (1) #24 bit-exactness gate for the sum(x·(1−d)) family;
  (2) cleanest floor probe after q06 (~90ms real work); (3) dictionary-encode
  p_brand/p_container so the OR compares codes and the shuffle ships codes;
  (4) guard the scan-conjunct pushdown (the load-bearing 28× reducer) against CPU
  fallback at SF10.
- **Priority**: other high; join medium (keep the win healthy); sort/agg low.

### Q20 — potential part promotion
- **Shape**: CANADA suppliers whose 'forest%' partsupp stock exceeds half their 1994
  shipments (nested semi-joins + correlated sum); 186 rows.
- **Plan**: 8 fragments, 7 exchanges, 5-deep serial dependency chain. Partsupp 800k
  shuffled un-pruned into a LEFT SEMI vs ~2k forest parts; grouped sum partials
  (~910k rows, ~600–800k groups — near-zero partial reduction) cross the exchange.
- **Sorts**: 186-row sort + merge; hidden canonical sort ×2 on the FP64 grouped sum.
- **Aggs**: one two-phase grouped sum by (l_partkey, l_suppkey); weak reduction means
  nearly the whole filtered scan crosses the wire; FP64 residual `0.5×sum` compare.
- **Joins**: 4 (nation 1-row broadcast; PARTITIONED semi forcing the 800k shuffle;
  BUCKET_SHUFFLE inner with residual; ~hundreds-row semi broadcast).
- **Distribution**: uniform keys; remote RFs (forest p_partkey → partsupp; ps keys →
  lineitem) planned and dropped — B wins by pruning exactly these.
- **Bottleneck**: fixed overhead + exchange latency, one real compute item. A 782ms
  vs B 242ms (3.23×).
- **Opportunities**: (1) runtime-filter translation + GPU bloom/IN probe (~100×
  pruning); (2) #24 (removes both canonical sorts + makes the threshold compare
  exact); (3) concurrent fragment execution (4 independent leaf scans queue serially);
  (4) inline sub-threshold exchange payloads (4 of 7 hops carry rows-to-hundreds).
- **Priority**: other high; agg medium; join medium; sort low.

### Q21 — suppliers who kept orders waiting
- **Shape**: SAUDI ARABIA suppliers sole-late on multi-supplier 'F' orders
  (EXISTS→LEFT SEMI, NOT EXISTS→RIGHT ANTI self-joins of lineitem); top-100.
- **Plan**: 9 fragments, 8 exchanges; F4 concentrates ~10M inbound rows from 4
  shuffle streams: l1 late (~3M), l3 late (~3M), orders-'F' (~729k), and the **full
  6M-row l2 as a semi-join build** — the largest GPU hash build in the suite.
- **Sorts**: TOP-N 100 of ~400 groups post-merge; lowered to full sort + fetch —
  negligible.
- **Aggs**: two-phase count(*) by s_name over ~4k rows — integer, no tax; a non-cost.
- **Joins**: 5 — three at lineitem scale incl. RIGHT ANTI and LEFT SEMI with residual
  `l_suppkey !=` predicates evaluated per hash match; two small broadcasts.
- **Distribution**: uniform l_orderkey hashing; the 4 inbound streams serialize behind
  one blocking transport thread per sender (ROADMAP 4a). Documented arena canary:
  pre-`7039665c` exhaustion victim, one transient arena refusal in the A6 sweep — the
  1280 MiB arena is closest to its limit here.
- **Bottleneck**: join + exchange bound — genuinely compute-loaded, hence the best
  ratio of the overhead-family: A 987ms vs B 441ms (2.24×). Residual gap: exchange
  serialization, 9 serial fragments, and B's RFs (orders-'F' o_orderkey ~49%
  selective; post-join l_orderkey → l3).
- **Opportunities**: (1) runtime filters (skip the near-useless full-l2 filter 2);
  (2) async/per-peer exchange senders — the top structural fix and the 8-CN shuffle
  precondition; (3) audit the semi/anti residual-predicate path (in-kernel vs
  gather-then-filter at ~7.5 lines/order fan-out); (4) M2 arena auto-sizing before any
  scale-up; (5) distinct (l_orderkey, l_suppkey) pre-aggregation of the 6M semi build.
- **Priority**: join high; other high; sort/agg low.

### Q22 — global sales opportunity
- **Shape**: per country-code count/sum of above-average-balance customers with no
  orders (scalar avg subquery + LEFT ANTI); 7 rows.
- **Plan**: 6 fragments, 5 exchanges, 5-level serial chain; two of the hops move a
  16-byte scalar (avg partial, then its broadcast) through the full
  staging-lease + packed-cuDF + nixl machinery before the main fragment can start;
  plus a **1.5M-row o_custkey broadcast** (~6MB, clone per extra CN) feeding a LEFT
  ANTI build 70× larger than its probe.
- **Sorts**: 7 rows; canonical tax on ~20k partial rows — measurable, small.
- **Aggs**: scalar avg (sum+count expansion) + grouped two-phase count/sum by
  substring (≤7 groups); FP64 compare `c_acctbal > avg` has near-tie drift risk (#24).
- **Joins**: 1-row-build NESTLOOP (pure residual filter) + LEFT ANTI BROADCAST
  (1.5M build / ~21k probe).
- **Bottleneck**: pure fixed overhead — worst tail ratio: A 485ms vs B 118ms (4.11×)
  over <50ms of actual GPU work.
- **Opportunities**: (1) dispatch independent leaves concurrently (the scalar-subquery
  chain is latency, not throughput); (2) inline small-batch exchange (4 of 5 hops);
  (3) lower 1-row-build nestloop to a GPU filter with a bound scalar; (4) #24 exact
  compare; (5) at 8 CN/SF10: distinct/bloom-reduce the anti build before cloning.
- **Priority**: other high; agg medium (critical-path latency); sort/join low.

## 4. Comparison table

A median = warm median (ms) of 3 runs; A/B = A ÷ B (>1 = Sirius slower). Priority =
dominant optimization category from §3.

| Q | major operators | exchanges (types) | sorts | aggregations | joins | A med | A/B | risk | priority |
|---|---|---|---|---|---|---|---|---|---|
| q01 | lineitem scan, 2-phase 8-measure agg | 2 (1 SH, 1 MG) | 4-row ×2 (+canonical 5.9M) | 2-phase, FP64 #24 | 0 | 418 | 0.80 | #24 drift 0.096% | aggregation |
| q02 | 8 joins, min subquery, TOP-N 100 | 13 (7 SH, 5 BC, 1 MG) | TOP-N 100/~460 | 2-phase min | 8 | 1138 | 4.97 | n_regionkey skew; empty-build regression | other (floor×frags) |
| q03 | 3.2M shuffle, part. join, single-phase agg, TOP-N | 4 (2 SH, 1 BC, 1 MG) | TOP-N 10/114k | 1-phase, FP64 #24 | 2 | 500 | 1.69 | out-of-band #24 (−0.39%) | other (RF/exchange) |
| q04 | semi join, 2-phase count | 3 (1 SH, 1 BC, 1 MG) | 5-row ×2 | 2-phase count (exact) | 1 semi | 428 | 1.70 | broadcast clones ×(N−1) | join |
| q05 | 5 joins, 6M lineitem shuffle | 10 (7 SH, 2 BC, 1 MG) | 5-row | 2-phase, FP64 #24 | 5 | 1026 | 3.21 | 5-value key skew | join + other |
| q06 | scan + scalar sum | 1 (GA) | none | 2-phase scalar | 0 | 308 | 1.40 | none (floor probe) | other (floor) |
| q07 | 5 joins, 1.5M orders BC | 9 (5 SH, 3 BC, 1 MG) | 4-row (+canonical str ×2) | 2-phase 3-key | 5 | 934 | 2.85 | 4-combination merge skew | join + other |
| q08 | cross join (13.3M), 7 joins | 12 (7 SH, 4 BC, 1 MG) | 2-row | 2-phase 2-measure | 7 (1 cross) | 1236 | 2.62 | synthetic-key cross join skew | join + other |
| q09 | cross join (107M pairs), 5 joins | 8 (4 SH, 3 BC, 1 MG) | 175-row | 2-phase, 600k→175 | 5 (1 cross) | 1104 | 0.93 | 648MB arena lease; scales with SF | join + other (arena) |
| q10 | 3 BC joins, 7-key agg, TOP-N 20 | 5 (1 SH, 3 BC, 1 MG) | TOP-N 20/38k (+canonical 7-key ×2) | 2-phase 7-key, weak reduction | 3 | 634 | 1.96 | out-of-band #24 + rank rotation | aggregation |
| q11 | dual pipelines, 4 aggs, nestloop | 8 (1 SH, 5 BC, 1 GA, 1 MG) | 1048-row | 2×2-phase (grouped+scalar) | 4 + NL | 830 | 5.65 | MULTI_CAST shape hangs today | other (floor×frags) |
| q12 | 1.5M orders shuffle, 1 join | 4 (3 SH, 1 MG) | 2-row | 2-phase int sums | 1 | 469 | 1.19 | none | other (RF) |
| q13 | right-outer join, count+histogram | 4 (3 SH, 1 MG) | 42-row | 1-phase + 2-phase counts | 1 outer | 450 | 1.29 | none | other (floor) |
| q14 | dual shuffle join, scalar sums | 3 (2 SH, 1 GA) | none | 2-phase scalar, #24 | 1 | 428 | 1.95 | CLONE_EXPR machinery | other (floor) |
| q15 | CTE ×2, 4 grouped aggs, max chain | 6 (3 SH, 1 GA, 1 BC, 1 MG) | 1-row (+canonical ×4) | 2×2-phase grouped + max | 2 | 681 | 2.72 | #29 FP64 equality (guarded) | aggregation + other |
| q16 | 800k BC build, distinct cascade | 4 (1 SH, 2 BC, 1 MG) | 18,314-row 4-key | 3-node count-distinct | 2 (1 anti) | 458 | 3.05 | worst broadcast N-scaling | other + join |
| q17 | 2× lineitem scan, grouped avg | 4 (3 SH, 1 GA) | none (+canonical 6M) | 2-phase avg + scalar | 2 | 469 | 1.71 | FP64 residual compare | aggregation + other |
| q18 | 6M build join, HAVING semi | 6 (5 SH, 1 MG) | TOP-N 100/57 (+canonical 6M) | 2-phase 6M + 1-phase 5-key | 3 (1 semi) | 621 | 2.23 | 6M build vs 57 probe | agg + join + other |
| q19 | balanced join, 3-arm OR residual | 3 (2 SH, 1 GA) | none | 2-phase scalar | 1 | 398 | 0.83 | scan-conjunct fallback would erase win | other (floor) |
| q20 | nested semis, correlated sum | 7 (4 SH, 2 BC, 1 MG) | 186-row (+canonical ×2 ~910k) | 2-phase weak reduction | 4 (2 semi) | 782 | 3.23 | dropped RFs (~100×) | other (RF/floor) |
| q21 | self-join anti/semi, 6M build | 8 (5 SH, 2 BC, 1 MG) | TOP-N 100/400 | 2-phase count (exact) | 5 (anti+semi) | 987 | 2.24 | arena canary; transport serialization | join + other |
| q22 | scalar avg chain, 1.5M anti BC | 5 (1 SH, 2 BC, 1 GA, 1 MG) | 7-row | scalar avg + 2-phase | NL + anti | 485 | 4.11 | near-tie FP64 compare | other (floor) |

SH = SHUFFLE, BC = BROADCAST, GA = GATHER, MG = MERGING-EXCHANGE.

## 5. Prioritized implementation roadmap

### 5.1 Sorting

No explicit SORT/TOP-N in the suite is a cost center at SF1 — the largest is q16's
18,314 rows; every TOP-N sits post-aggregation below a merging gather. The suite's
dominant sort volume is the **canonical-order sort tax** inside FP64-lowered grouped
aggregation, which is an aggregation artifact and is removed by #24 (see 5.2), not by
sort work.

Ranked queries:
- **High**: none purely sort-bound.
- **Medium**: q10 (full 38k-group sort where top-20 selection would do), q15 (hidden
  canonical sort paid 4×), q01 (canonical sort of 5.9M rows — booked under agg).
- **Low**: everything else (≤1048 explicit rows).

Work items:
1. **GPU top-k for TOP-N** (q10, q18, q02, q03, q21): translate TOP-N to a segmented
   top-k select instead of full SortRel + fetch. Benefit: small at SF1, linear at
   SF10+. Complexity: medium (translator + one engine operator). Deps: none.
2. **Limit-aware merge at MERGING-EXCHANGE**: verify/replace the gather-CN
   stream-read + full re-sort lowering (node_translator.rs:746) with a k-way merge
   honoring the limit. Benefit: negligible at SF1, matters with more instances at
   SF10/8-CN. Complexity: low-medium. Deps: none.
3. **Canonical-sort removal**: rides entirely on #24 (5.2 item 1) — do not build a
   separate sort optimization for it.

### 5.2 Aggregation

Ranked queries:
- **High**: q01 (the 5.9M-row partial groupby IS the query), q15 (grouped FP64 sums
  computed twice + max chain), q17 (6M-row two-phase grouped avg), q18 (6M-row sum by
  l_orderkey — largest single kernel), q10 (7-wide-key groupby, canonical tax ×2,
  weak ~3× partial reduction).
- **Medium**: q03 (590k sort-based groupby + out-of-band drift), q16 (a provably
  useless dedup phase), q20 (~910k rows, near-zero partial reduction), q07 (string-key
  canonical sorts), q11/q13/q22 (many agg nodes, tiny data).
- **Low**: q02, q04, q05, q06, q08, q09, q12, q14, q19, q21.

Work items:
1. **#24 decimal-native SUM/AVG** — the correctness capstone and the single
   highest-value aggregation change. cuDF fixed_point SUM uses exact integer atomics
   (`device_aggregators.cuh:126`): removes the 0.1–0.4% low bias (q01, q03, q05, q07,
   q08, q09, q10, q14 measured), the q10 rank rotation, the q15 equality-flake root
   (#29), **and** the canonical-sort tax on every grouped float sum (q01/q10/q15/q17/
   q18/q20 headline). Complexity: large (expr_translator.rs:826-833, type_mapper.rs,
   partial_state.rs, gpu_aggregate_impl/gpu_merge_impl/aggregate_op_util, expression
   path for the (1−d) literal). Deps: update the 76-case `wire_type_parity` model and
   engine together; start SUM(DECIMAL64) end-to-end gated on q06/q01 bit-exactness.
2. **Group-key narrowing / dictionary encoding** (q01 1-char flags → int8; q10
   partition + aggregate on c_custkey with first-value carry of the 6 dependent
   columns; q16/q19 p_brand/p_type/p_container codes). Benefit: removes string
   hashing/sorting from hot groupbys and shrinks key-heavy shuffles. Complexity:
   medium (translator-local for q10's functional dependency; scan/encode pass for
   dictionaries). Deps: none; compounds with #24.
3. **Adaptive STREAMING passthrough** (q16, q20): measure the partial phase's
   reduction ratio on the first batches, bypass when ≈1.0 (StarRocks CPU BEs do
   this). Benefit: saves a full multi-key varchar groupby on q16; avoids wasted
   partials on q20. Complexity: low-medium, engine-side. Deps: none.

### 5.3 Joins

Ranked queries:
- **High**: q18 (6M-row build probed by 57 rows), q08 (cross join lowered to a
  synthetic-constant single-key hash join → 13.3M intermediate; un-pruned 6M build),
  q09 (107M-pair probe into a 6M build), q04 (mis-sided 3.8M broadcast semi build,
  duplicated per CN), q07 (1.5M broadcast build + 1.8M inverted build), q05 (full-6M
  inverted build), q21 (three lineitem-scale joins incl. a 6M semi build + residual
  `!=` predicates), q02 (8 joins where join count × fixed dispatch drives cost).
- **Medium**: q16 (800k inverted broadcast build — worst N-scaling), q12/q13/q14/q17/
  q03/q20 (healthy kernels, cost is feeding them).
- **Low**: q10, q11, q15, q19 (keep q19's residual-OR path healthy — it is the win),
  q22.

Work items:
1. **Engine-side adaptive build/probe election** (size-based swap when build ≫ probe,
   incl. semi-join side swap emitting matched build keys): fixes q04, q05, q07, q08,
   q14, q16, q18 — every stats-blind FE siding — without touching the FE. Complexity:
   medium; extends the existing BUILD_PROBE election (empty-build fix `59ce6662`).
   Deps: none.
2. **Semi/anti build dedup below the sink** (q04 3.8M→1.3M distinct; q21 distinct
   (l_orderkey,l_suppkey); q22 distinct o_custkey): a sender-side GPU distinct
   shrinks broadcast payloads and per-CN builds — the mitigation that scales to
   8 CNs. Complexity: low-medium. Deps: none.
3. **Real cross join**: replace the synthetic-constant lowering
   (node_translator.rs:1832) with `cudf::cross_join`, or better, recognize
   cross-join-feeding-a-multi-key-join and re-associate (q08 kills the 13.3M
   intermediate + one shuffle; q09's 107M-pair shape needs the pair-filter in 5.4
   instead). Complexity: medium (kernel swap) / high (re-association). Deps: none.
4. **Low-cardinality partition-key skew fallback** (q02 n_regionkey, q07 name-pair,
   q08 year): detect ≤k-value partition keys at the sink and fall back to broadcast
   of the smaller side. Complexity: medium. Deps: none.
5. **Residual-predicate path audit** (q21's `!=` on anti/semi matches; q19's 3-arm
   OR): confirm evaluation stays in/near the hash-join kernel rather than a
   gather-then-filter slow path. Complexity: low (audit) + targeted fixes.

### 5.4 Other (exchange, runtime filters, dispatch floor, scan, memory)

This is the suite's dominant category: 16 of 22 queries are floor- or exchange-bound.

Ranked queries:
- **High**: q02, q11, q22 (pure floor×fragments, ratios 4.1–5.7×), q12, q17, q18,
  q20, q03, q08 (dropped runtime filters, 30–100× excess volume), q05, q07 (exchange
  count), q09, q21 (arena + transport serialization), q15 (double evaluation),
  q06/q13/q14/q19 (clean floor probes).
- **Medium**: q01, q04, q10, q16.

Work items, in recommended order:
1. **Remote runtime-filter execution** — the highest-leverage single feature.
   Translate the FE's build_runtime_filters into a GPU bloom/IN probe applied at
   FileScanNode (or before the partition sink); engine scaffolding exists
   (`src/op/dynamic_filter_publisher.cpp`). Gains: q12 orders shuffle 50× (1.5M→31k),
   q17 probe ~100× (204 keys), q18 orders 1.5M→~57, q20 partsupp/lineitem ~100×,
   q03 lineitem 5×, q08 lineitem 6M→40k, q21 l1/l3 several-M rows. Add: adaptive
   disable on ~1.0 selectivity (q13, q14); a **pair-hash** variant for q09 (~15× on
   the dominant exchange). Complexity: large (translator + wire + engine probe).
   Deps: none hard; interacts with build/probe election (5.3.1).
2. **Dispatch-floor and serialization attack**: overlap independent fragments on the
   engine thread (q13's proof case: GPU work already beats B), per-peer async send
   workers (ROADMAP-8CN 4a; q21's 4 inbound streams; precondition for 8-CN shuffle),
   pre-established sessions, and profiling the q06 fixed path — every ms moves all 22
   queries. Complexity: medium-high (threading model constraints, 19d7cca2). Deps:
   engine-abort surface (#31.3) shares the fragment-lifecycle plumbing.
3. **Small-payload inline exchange**: sub-threshold batches ride the transmit RPC
   directly, bypassing arena lease + nixl WRITE (q02's 6 dwarf-table fragments, q11's
   8 exchanges under 2MB, q20's 4 tiny hops, q22's two 16-byte scalar hops).
   Complexity: medium. Deps: none.
4. **MULTI_CAST_DATA_STREAM_SINK (#31.1)**: first make unhandled sinks `Err` (the
   silent-hang fix, ~30 min), then implement multi-cast → enables CTE reuse →
   structurally removes q15's double evaluation and halves q11's duplicated pipeline;
   also helps q17's double scan. Complexity: medium. Deps: none; correctness item
   regardless.
5. **M2 staging-arena auto-sizing** (+ chunked multi-lease export for q09-class
   streams): prerequisite for SF10/M3; q09's 648MB lease and q21's arena-canary
   behavior scale linearly with SF. Complexity: small (sizing) / medium (chunked
   export). Deps: none.
6. **Scan/misc**: sender-side expression pushdown (q03 revenue multiply, ~25%
   narrower shuffle); parquet row-group min/max skipping (q06, q14); a two-literal
   substring-scan kernel for multi-wildcard LIKE (q13, q09 '%green%'); result-path
   batching for large result sets (q16's 18,314 rows). Complexity: low-medium each.

**Single-engine-thread serialization** is the cross-cutting constraint under items
2–4: with one engine thread per CN, fragment count is a latency multiplier (q02: 14,
q08: 13, q05: 11), independent scans queue (q13, q15, q20, q22), and lease/transmit
RPCs contend. Any concurrency work must respect the manager-thread constraints
(19d7cca2) and the lease-off-engine-thread fix (a94e8660).

## 6. Validation benchmarks & metrics

**Harness** (from `experimental/starrocks/benchmarks/tpch/`):
- `bench.sh <out_csv> [runs] [q01 q02 ...]` — per-run timings CSV
  (`query,run,status,ms,rows`), 1 discarded warm-up + 3 timed runs by default.
  Requires `TPCH_DATA`; set `RESTART_CMD` (no real cancel yet — a wedge strands
  fragments) and `MIN_BACKENDS=2`.
- `TPCH_DATA=... ./run-comparison.sh [out_dir] [runs]` — full A-then-B sweep +
  medians table + plot (`analyze.py`); `SKIP_A`/`SKIP_B` to reuse a side.
- Metrics: **warm median per query** (primary), **A/B ratio** and its geo-mean
  (headline; currently 0.48x B-favoring), and **q06 median** as the floor gauge.
  Per-fragment/per-operator times are *not* available from the StarRocks runtime
  profile (Sirius CNs emit placeholders) — use engine-side logs (`log-analyzer`
  skill) and nsys (`profile-analyzer` / `optimization-advisor` skills) for operator
  attribution.

**Which queries gate which optimization**:

| optimization | gating queries | success metric |
|---|---|---|
| #24 decimal-native agg | q01, q06, q14, q19 bit-exact vs DuckDB (no tolerance band); q03, q10 back in-band; q15 8/8 **without** the canonical-sort path | correctness first; then q01/q10/q15/q17/q18/q20 warm medians (canonical-sort removal) |
| runtime filters | q12 (1.5M→31k orders shuffle — cleanest signal), q17, q18, q20, q03, q08; q09 only via the pair variant | warm median + exchanged-bytes/rows from CN logs |
| build/probe election + dedup | q04, q16, q18, q07, q05; q02 regression (empty-build, 59ce6662) | warm median; q02 must stay green |
| cross-join lowering | q08, q09 | warm median; q08's F11 no longer dominates |
| floor/pipelining/inline exchange | q06, q11, q22, q02, q13, q14 | q06 median (floor), q11/q22 A/B ratio |
| MULTI_CAST / CTE reuse | q15, q11 (+ the `cbo_cte_force_reuse_node_count=1` hang repro) | q15 median ~halves; no silent hang; 8/8 stability |
| M2 arena sizing | q09 (648MB lease), q21 (canary) | q09/q21 pass at SF10 without hand-tuned `SIRIUS_EXCHANGE_STAGING_BYTES` |
| top-k / merge audit | q10, q18 | median at SF10 (SF1 signal too small) |

**Correctness gates** (the OPEN-ISSUES verification protocol): translator tests →
`cn-test-no-engine` → `cn-test` incl. the 76-case `wire_type_parity` gate
(`experimental/starrocks/src/wire_type_parity.rs` — update model + engine together
for #24) → C++ suite if `src/**` changed → GPU harness → live: affected queries solo
vs the **DuckDB oracle** → full 22-query sweep → for anything touching
exchange/arena/transport, an **endurance sweep** (2–3 consecutive sweeps, zero
restarts — the q21 arena and q15 flake classes only show up here). Every fix lands
with the regression test that would have caught it. Cluster ops and the sweep ladder
are in the `tpch-bench` skill.

## 7. Limitations

- **SF1 only.** The ~300ms dispatch floor is 25–75% of most queries here; at SF10+
  (M3) data-proportional costs (q09's arena lease, q03/q05/q08/q18/q21 shuffle
  volumes, broadcast clones) grow linearly while the floor does not — category
  priorities will reorder toward joins/aggregation and away from pure floor work.
  Conversely, several "medium" items (top-k, limit-aware merge, dictionary encoding)
  only become measurable at SF10.
- **2 CNs only.** Skew conclusions (q02, q07, q08 low-cardinality keys) and
  broadcast-clone costs (q04, q16, q22: ×(N−1)) are extrapolated to 8 CNs, not
  measured.
- **EXPLAIN coverage**: all modes attempted worked (EXPLAIN, VERBOSE, LOGICAL, COSTS,
  ANALYZE); none failed. But EXPLAIN ANALYZE's per-operator runtime metrics are
  placeholders (Sirius CNs don't populate the runtime profile), and
  `SELECT current_version()` is unroutable (constant-only queries land on a CN that
  rejects TPlanNodeType(19)).
- **What plans cannot show**: runtime skew realized by hashing (vs the key-cardinality
  inference used here), per-operator GPU timings, canonical-sort cost as a measured
  fraction (inferred from q06/q01 deltas and code paths, not profiled), staging-arena
  occupancy over time, and FileScanNode conjuncts (this vendored FE slice does not
  print broker-scan conjuncts — pushdown was inferred from row counts).
- **Stats-blind plans**: every `FILES()` node reports cardinality 1, so join sidings,
  the q08/q09 cross joins, and broadcast choices reflect an uninformed FE. Injecting
  statistics or session hints would change plan shapes — several Sirius-side
  recommendations (build/probe election, adaptive exchange downgrade) exist precisely
  to be robust to that, but re-analysis is needed if FE stats land.
- **Row-count estimates** in §3 are derived from TPC-H selectivities and spot checks,
  not measured operator output (see the runtime-profile gap above).
- **Single-file parquet per table** limits FE byte-range split balance; multi-file
  layout at SF10 (per M3) changes scan distribution.
