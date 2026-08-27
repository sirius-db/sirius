# TPC-H Q1–Q22 plan survey (roadmap step 0) — 2026-08-06

> **Addendum (same day): F1 is RESOLVED.** The byte-range split stack
> (`a5c25f76..8c2ebea5`, BYTE-RANGE-SPLITS-PLAN.md) landed and was verified exactly-once live:
> `count(*) = 6001215` over the single split 155 MB lineitem on 2 CNs, Q6 = `61567694.9502`.
> The 18/22 scan-level blocker below is historical; the next blocker per query is what F2/F4
> list (partitioned output for joins/GROUP BY, avg expansion for Q1/Q17/Q22).
>
> **Addendum 2 (2026-08-07): partitioned output landed too** (`6c7217aa..5b4cfc7a`) — broadcast
> and shuffle joins plus grouped two-phase GROUP BY verified oracle-exact on 2 CNs. A first live
> execution sweep was attempted and produced two findings: (a) Q1 fails at defaults on the avg
> guard as predicted (fix in progress: sum+count expansion); (b) **a query that fails
> mid-execution wedges the engine for every later statement** (no cancel/GC — roadmap #5), so
> sweeps need CN restarts between execution failures; results after a wedge are invalid.
> Also: the 390-dump capture contains ZERO `assert_num_rows_node: Some` instances — the FE may
> elide the assert for provably-scalar subqueries; the Q2-class blocker needs re-diagnosis from
> a clean sweep (see SCALAR-SUBQUERY-PLAN.md).
>
> **Addendum 3 (2026-08-07): measured execution state supersedes this survey's per-query
> guesses** — see the table at the end of this document. 19/22 queries now run to completion
> live on 2 CNs.

Ran against the live two-CN cluster (`cluster2`, post two-phase stack `64977ebb..11625add`),
FE default settings and `new_planner_agg_stage = 1`, data = single-file SF1 parquet per table at
`/home/ubuntu/git/sirius/scratch/tpch_sf1/<table>/part.0.parquet` (lineitem 155 MB, orders 40 MB,
partsupp 37 MB, customer 11 MB, rest ≤ 5 MB).

## How it was run (reproducible)

- Queries: duckdb's pre-parameterized set (`duckdb/extension/tpch/dbgen/queries/qNN.sql`), each
  prefixed with 8 CTEs mapping table names to `FILES("path"=".../<table>/*.parquet")`. Q15's own
  `WITH revenue` merges into the CTE list. One dialect fix: Q22 `substring(x FROM 1 FOR 2)` →
  `substring(x, 1, 2)` (FE parse error otherwise).
- CNs launched with `SIRIUS_CN_TRANSLATE_ONLY=1` (+`SIRIUS_CN_DUMP_FRAGMENTS`): every fragment is
  accepted, translation verdicts logged, queries fail at fetch — no GPU execution.
- Per query × stage: `EXPLAIN` captured (FE plan shape), then a real submit so every fragment is
  dispatched and translated; CN-log deltas grepped for `error=`.

**Harness caveats (affect interpretation):**
1. In translate-only mode exchange inputs are never bound, so any exchange-*receiving* fragment
   fails translation with "requires a bound same-node input stream" — a harness artifact.
   **Leaf-fragment verdicts are reliable; upper-fragment verdicts come from EXPLAIN shape
   analysis instead.** A real guard inside an exchange-fed fragment (e.g. a grouped merge agg)
   is masked in the logs.
2. The mysql handshake plans a `UNION` fragment (`TPlanNodeType(19)`) per connection — filtered.

## Headline findings

**F1 — Byte-range parquet splits are the dominant blocker (roadmap #2, not #1).**
18/22 queries fail leaf translation at BOTH agg stages with `byte-range splits do not cover the
whole parquet file` — every query touching the 155 MB lineitem file. The FE's split threshold
sits between 40 MB (orders — not split) and 155 MB; only lineitem is split at SF1 single-file.
Clean-leaf queries: Q2, Q11, Q13, Q16 (+Q22 at agg1) — exactly the ones not scanning lineitem.
Near-term workaround (demo-grade): shard every large table into whole files below the threshold
(the `lineitem_multi` pattern generalized); real fix is #2. **This blocks even the Q6 shape on
realistic single-file data** — the demo only worked because `lineitem_multi/` dodges it.

**F2 — The FE plans BOTH broadcast and shuffle joins for stats-less FILES() tables.**
Size-derived estimates give small tables broadcast and big⋈big PARTITIONED shuffle. Q12, Q13,
Q14, Q17, Q18, Q19 have *only* PARTITIONED joins → a broadcast-only first cut of #1 unlocks
roughly Q4/Q10/Q11/Q16/Q22 and leaves the rest waiting on hash-partitioned output. **#1 needs
both UNPARTITIONED-broadcast and HASH_PARTITIONED shapes for real coverage.**

**F3 — Bucket shuffle DOES appear**: `BUCKET_SHUFFLE(S)` join distribution in Q9, Q15, Q17, Q18,
Q20, Q21 (1–2 per plan). Open: whether it is avoidable via session variables for v1 or the
partitioned sink must reproduce the bucket function (the CRC32-parity question revives for
exactly these six queries).

**F4 — Default-stage aggregation is two-phase everywhere** (2–6 AGGREGATE nodes per plan).
The scalar two-phase path just landed; every grouped agg here additionally needs #1 (its shuffle)
+ the grouped-guard removal. avg blocks Q1, Q17, Q22 leaf fragments at default stage (the
sum+count expansion follow-up); at `agg_stage=1` those leaves are clean.

**F5 — Join-type coverage is NOT a blocker for these plans.** The 22 plans use INNER, LEFT SEMI
(Q4, Q20), LEFT ANTI (Q22), NULL-AWARE LEFT ANTI (Q16), RIGHT OUTER (Q13) — all already mapped in
`translate_hash_join` (`node_translator.rs:897-906`). RIGHT_SEMI never appears. (Shape/conjunct
guards may still fire per query; unverifiable for exchange-fed fragments in this harness.)

**F6 — FE dialect:** everything parses except the one Q22 substring form. `extract(year …)`
(Q7-Q9), `GROUP BY <alias>` (Q15), CTE-shadowing of table names — all fine.

## Per-query matrix (default stage)

| Q | leaf verdict (default) | joins (distribution) | agg nodes | bucket-shuffle | notes |
|---|---|---|---|---|---|
| 1 | splits + **avg** | — | 2 | | avg partial refused |
| 2 | clean | 5 BROADCAST, 3 PARTITIONED | 2 | | no lineitem |
| 3 | splits | 1 BC, 1 PART | 1 | | |
| 4 | splits | 1 LEFT SEMI (BC) | 2 | | |
| 5 | splits | 2 BC, 3 PART | 2 | | |
| 6 | splits | — | 2 | | scalar two-phase = supported shape once splits land |
| 7 | splits | 3 BC, 2 PART | 2 | | |
| 8 | splits | 3 BC, 3 PART | 2 | | |
| 9 | splits | 2 BC, 1 PART | 2 | ✓ | |
| 10 | splits | 3 BC | 2 | | broadcast-only join set |
| 11 | clean | 4 BC | 4 | | no lineitem; broadcast-only |
| 12 | splits | 1 PART | 2 | | shuffle-only |
| 13 | clean | 1 RIGHT OUTER (PART) | 3 | | no lineitem; shuffle-only |
| 14 | splits | 1 PART | 2 | | shuffle-only |
| 15 | splits | 1 BC | 6 | ✓ | view→CTE works |
| 16 | clean | 1 BC, 1 NULL-AWARE ANTI (BC) | 3 | | distinct agg (multi-phase) |
| 17 | splits + **avg** | 1 PART | 4 | ✓ | |
| 18 | splits | 1 PART | 3 | ✓ | |
| 19 | splits | 1 PART | 2 | | shuffle-only |
| 20 | splits | 2 BC, 1 LEFT SEMI (PART) | 2 | ✓ | |
| 21 | splits ×3 | 2 BC, 1 PART | 2 | ✓ | 3 lineitem scans |
| 22 | **avg** (clean at agg1) | 1 LEFT ANTI (BC) | 4 | | substring dialect fix applied |

At `agg_stage=1` the picture is identical minus the avg errors (Q1/Q17/Q22 leaves clean).

## What this re-sequences

1. **#2 (byte-range splits) is promoted to co-critical with #1** — it gates 18/22 queries at any
   agg stage and any CN count > 1 on realistic files. It is also the smaller item (M vs L).
   Interim unblock for development: shard the SF1 tables into sub-threshold whole files.
2. **#1 (partitioned output) scope confirmed**: both broadcast and hash shapes needed; grouped
   two-phase guard removal rides on it; six queries additionally raise the bucket-shuffle
   question (F3) — investigate the session-variable escape before building CRC32 parity.
3. **avg expansion** (post-#4 follow-up) is worth scheduling: it is the only leaf blocker beyond
   splits at default settings (Q1, Q17, Q22).
4. Join-type work (RIGHT_SEMI etc.) drops off the TPC-H path entirely (F5); ASSERT_NUM_ROWS
   remains untested by this harness (scalar-subquery shapes are exchange-fed — recheck once
   splits land and fragments execute for real).

## Addendum 3 (2026-08-07) — measured execution state, Q1–Q22

Full live sweep on `cluster2` (SF1, 2 CNs, 3 timed runs per query, 30 s ceiling, harness
`90750142`; CSV `/tmp/sirius-tpch-bench/bench/A4/timings.csv`). Every non-pass row was
re-run solo on a healthy cluster and reproduces with the cause listed — no harness artifacts,
no cascade. This table replaces the translate-only guesses above with execution truth.

| Q | state | ms (3 runs) | rows | cause / notes |
|---|---|---|---|---|
| 1 | pass | 347/337/368 | 4 | max value drift 0.096 % (#24, in band) |
| 2 | wedge | 30004 (run0) | 0 | engine-thread stall, deterministic (#26) — NOT the predicted ASSERT_NUM_ROWS blocker |
| 3 | pass | 596/569/498 | 10 | keys/counts/order exact; 2 revenue values low beyond 0.25 % (#24) |
| 4 | pass | 357/580/519 | 5 | exact |
| 5 | pass | 1145/843/884 | 5 | max drift 0.111 % (#24, in band) |
| 6 | pass | 257/288/296 | 1 | exact |
| 7 | pass | 913/923/913 | 4 | |
| 8 | pass | 1095/1056/1056 | 2 | |
| 9 | pass | 1043/954/1012 | 175 | needed the 1280 MiB staging arena (1d4428da) |
| 10 | pass | 581/580/580 | 20 | keys/counts exact; 2 values low beyond 0.25 % + a 3-rank rotation (#24) |
| 11 | pass | 965/959/879 | 1048 | |
| 12 | pass | 499/499/500 | 2 | |
| 13 | pass | 478/459/459 | 42 | |
| 14 | refused | 250 (run0) | — | `slot 35 (tuple 5) not in row_tuples [5]`: a `common_slot_map` expr from Project node 6 referenced by the sibling AGGREGATE node — cross-node common-expr registration missing (loud, deterministic) |
| 15 | wedge | 1712 (run0) | 0 | empty-result flake (#29, FP64 equality race): 3/6 correct on a warm cluster, correct value drift 0.043 % |
| 16 | pass | 518/488/489 | 18314 | byte-identical to DuckDB; needed CLONE_EXPR + slot-id fallback (pending commit) |
| 17 | pass | 953/953/1064 | 1 | |
| 18 | pass | 752/701/682 | 57 | exact |
| 19 | pass | 408/409/397 | 1 | |
| 20 | pass | 742/812/832 | 186 | |
| 21 | pass | 944/934/912 | 100 | |
| 22 | pass | 523/524/545 | 7 | |

Net: **19/22 complete (exact key sets, counts, self-consistent ordering); 17/22 also inside
the 0.25 % value tolerance.** The survey-era blockers (splits, avg, partitioned output,
bucket shuffle, CLONE_EXPR/TExprNodeType 29) are all cleared; what remains is one engine
stall (#26), one FP64 determinism race (#29), one decimal-path value deficit (#24), and one
translator descriptor gap (q14). Detail: QUERY-TIMEOUT-ANALYSIS.md "Final status".
