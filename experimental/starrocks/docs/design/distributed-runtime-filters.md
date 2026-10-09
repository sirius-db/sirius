# Plan: apply every runtime filter the FE plans

*Oct 9. How the StarRocks FE and BE (4.1.3, `8a8e186`) build, merge and apply runtime filters, and what that means for the CN.*

## Goal

Apply every runtime filter the FE plans for TPC-H that can be applied correctly: **39 of 42** at SF1000. The other 3 are skipped on purpose: 2 whose keys really do cover their whole range, and 1 that targets an aggregation node, which StarRocks 4.1.3 doesn't apply either.

Why it matters:
- At SF3000, q09 runs out of GPU memory without its filter. With it, `lineitem` shrinks about 18× before the shuffle.
- Starburst saw Q17 run 3× slower without dynamic filtering. q17's filter is on a partitioned join, which #2062 doesn't cover.

## How StarRocks handles filters, and what it changes

| StarRocks 4.1.3 | what it changes in this plan |
|---|---|
| The FE never reads a filter's payload. In an all-Sirius cluster, the CNs can send filters in any format, min/max-only included. | Partitioned-join filters don't need StarRocks' Bloom encoding. We pick our own format and transport between CNs (Step 2). |
| The FE sends `runtime_filter_builder_number` and `id_to_prober_params` to the root fragment's instances. | Confirmed in our dumps: builder count 4 and 4 prober addresses per filter. A CN knows from the plan how many shares to wait for and who probes. |
| A merged filter is one slice per build instance. A prober picks its slice with the exchange hash. | We don't slice. A union of all shares is a superset of every slice, so it's always safe on a scan. |
| Skew joins (`enable_optimize_skew_join_v2`, off by default): the merged filter must also carry the broadcast branch's keys. Without them it drops rows. | Refuse filters in `skew_join_runtime_filters`, and any filter with `is_broad_cast_join_in_skew` (Step 1). |
| In 4.1.3 the FE can attach a filter to a join node. A sliced filter there drops rows that match. StarRocks' BE ignores it. | Only scans apply filters. Ignore join, exchange and aggregation targets, and test that we do (Step 1). |
| A missing filter costs speed, never correctness. A filter checked against the wrong keys loses rows. | Every failure path must run unfiltered, never partially filtered. #2062 already does this; keep it a rule. |
| Local filters (`has_remote_targets = false`) skip the FE's cost check, and they never leave the fragment instance. | All 14 same-fragment filters are local. Applying them is free on the plan side. Whether it pays off is a separate question (Step 4). |
| StarRocks scans wait 20 ms for a filter, then start, and install a late filter for the rest of the scan. | #2062 waits for the build (up to 30 s) and never installs late. Keep it: the join can't run before its build is done anyway, and q09 shows the trade pays. Revisit if a profile shows scans idle behind slow builds. |
| `EMPTY_FILTER` (min/max only) must have its min/max filled in. A StarRocks BE reads an unfilled one as an empty range and drops rows. | A Step 6 rule, for mixed clusters only. |
| A null-safe join (`<=>`) gets filters too (`JoinNode.java:213`), but `equalForNull` isn't sent in `TRuntimeFilterDescription`. | Detect `<=>` from the join's `eq_join_conjuncts` opcode (`EQ_FOR_NULL`) instead (Step 1). |

## What the FE plans (TPC-H, SF1000, 4 CNs)

From the recorded FE fragments of the 22-query run (`20261006_214436_sf1000-all22`, `docs/scripts/rf_tally.py` and `rf_logs.py`). Every filter is a join filter with a bare-slot key and an exchange as the join's build child, and every join uses `=`.

| class | filters | queries | today (#2062, `6b527513`) |
|---|---|---|---|
| **A.** Broadcast join; probing scan in a separate leaf fragment on the same CN | 19 | q02, q05, q07, q08, q09 | **16 applied.** 3 skipped by the density check: q08 rf1 and q09 rf1 really are dense (every supplier). q05 rf1 is a false positive: 2M rows but only 5 distinct nation keys; the check counts duplicate rows (`runtime_filters.rs:146`). It costs nothing in q05 because rf2 applies the same keys. |
| **B.** Broadcast join; probing scan in the join's own fragment | 14 | q02, q10, q11, q16, q19, q20, q21 | Dropped. All 14 are local filters. In 12 of them the scan sits directly under the join, or under one projection. |
| **C.** Broadcast join, other targets | 2 | q02 rf6 (a scan in another non-leaf fragment), q15 rf0 (an aggregation node, DECIMAL128 key) | Dropped. |
| **D.** Partitioned join: each CN builds part of the key set | 7 | q12, q14, q15, q17, q20 (2), q22 | Dropped: "no runtime filter of this scan is built on this CN". |
| **Total** | **42** | | **16 applied** |

These counts come from an Oct 6 work-in-progress build. It uses #2062's rules, but rerun the tally on the PR head as part of Step 1.

**Only 8 of 22 queries complete today** ([tpch-coverage-plan.md](../tpch-coverage-plan.md)). That limits what each step can show right now:

| class | queries that complete today | queries that fail for other reasons |
|---|---|---|
| A | q05, q07, q08, q09 | q02 (two-phase aggregation) |
| B | q19 | q02, q10, q16, q20, q21 (translator refusals); q11 returns 0 rows with no error, which needs its own look |
| C | none | q02, q15 |
| D | q12, q14 | q15, q17, q22 (two-phase aggregation), q20 (join type) |

## How #2062 applies a filter

- **Accepted** (`runtime_filter.rs`):
  - a join filter with `build_join_mode == BROADCAST`;
  - the join's build child is an exchange;
  - the key is a bare slot;
  - the target is a FILE/HDFS scan in a leaf fragment;
  - a receiver on the same CN builds it.
- **Applied:**
  - The scan's fragment is deferred until the build exchange is complete on its CN.
  - The engine copies the key column out of the received batches in place (`copy_column`).
  - The translator wraps the scan in an exact LEFT SEMI join against those keys.
  - The FE's own filter payload is never used.
- **Limits:**
  - Signed-integer keys only (`exchange_direct.cpp`, `add_key_stats`).
  - The density skip (`SIRIUS_CN_RUNTIME_FILTER_MAX_DENSITY`, default 0.5, always applied above 2^20 keys).
  - A 30 s wait limit (`SIRIUS_CN_RUNTIME_FILTER_WAIT_MS`).
  - `SIRIUS_CN_RUNTIME_FILTERS=0` turns it off.
- **Two facts that make the steps below possible:**
  - A receiver runs only once every one of its exchanges is complete (`register_receiver` / `drain_ready`). So inside a ready receiver, a same-fragment build exchange is already complete and its keys can be read. This is the moment Step 4 uses.
  - `streaming_fragment::run()` needs every input closed first. So a filter pushed into a fragment between `build()` and `run()` can't race its scan. This is the moment Step 3 uses.

## Steps

Ordered by filters gained per unit of work. Each step ends with:
- the 22-query run at SF1000, checked against DuckDB, with a leak check after every query;
- a per-query count of filters planned, applied and skipped, by reason (from the logs added in Step 1).

The SF3000 runs of 7 queries follow for Steps 2 and 3.

### Step 1: guards, accounting and the density fix (CN only, small)

**Correctness guards.** Each one makes the filter skip, with a logged reason:
- a join whose `eq_join_conjuncts` contain `EQ_FOR_NULL` (`<=>`). The semi-join uses `=`, and it would drop NULL probe keys that a null-safe join keeps.
- a filter whose id is in the join's `skew_join_runtime_filters`, or that has `is_broad_cast_join_in_skew`.
- a target that isn't a scan (a join, exchange or aggregation node), per 4.1.3 behaviour. Already true in the code; make it an explicit, tested rule.
- `filter_type` other than `JOIN_FILTER` (TopN, aggregation filters). Already skipped; same.

**Density by distinct keys.** `add_key_stats` counts duplicates, so a skewed key set looks dense.
- Add a distinct count to `KeyStats` (`cudf::distinct_count` on the key column), and judge density on it.
- This recovers q05 rf1. It also makes the check right for any build that isn't unique on its key.

**Accounting:**
- One line per query and CN: filters planned, applied, and skipped with the reason. The reasons are partitioned, same fragment, other target, key type, density, timeout, not built on this CN, null-safe, skew. Today classes B–D drop with no log.
- Name fragment dumps by CN as well as sequence number (`compute_node_service.rs:772`). Today the 4 CNs write `fragment-NNNN.txt` into one directory and overwrite each other.

**Tests:**
- A CN unit test per skip reason.
- A translator test that each guard reads the right field.

**Gain:** +1 filter (q05 rf1), so **17**. No skipped filter is silent any more.

### Step 2: partitioned-join filters between CNs (class D, 7 filters; CN only)

Each CN builds only its share of the keys, so a probing scan needs the union of every share. The FE never reads the payload. In an all-CN cluster, the format and transport are ours to choose.

- **Topology from the plan:** `runtime_filter_params` on the root fragment gives:
  - `runtime_filter_builder_number`: how many shares make a filter;
  - `id_to_prober_params`: the instance and address of each prober.

  The FE sends these only to the root fragment's instances. The root's CN forwards each filter's prober list and share count to the build CNs, or each build CN derives them from its own fragment. Decide this in the design review.
- **Share:** when a partitioned join's build exchange completes on a CN, the CN computes its share:
  - min, max, distinct count;
  - the exact keys, when they're under a size limit (start at 2^22 keys per share).
- **Transport:** send each share straight to every probing CN as a small broadcast over the existing direct exchange (NIXL). No merge node is needed: every prober unions N shares itself. `transmit_runtime_filter` stays unimplemented until Step 6.
- **Apply:** the probing scan is deferred as #2062 does today, but it waits for N shares instead of a local exchange. Then:
  - if every share carried exact keys, the semi-join on the union of keys;
  - otherwise a range predicate `key BETWEEN min AND max` on the scan, until Step 3 provides Bloom filters.
- **Failure:** a missing share or a timeout runs the scan unfiltered. A purged query drops its waiting shares.
- **By-product:** a broadcast filter whose scan runs on a CN without a build receiver can travel the same way. TPC-H at 4 CNs has none today, but other plans and CN counts will.
- **Tests:**
  - a CN unit test: the union is installed only after all N shares arrive; a missing one times out unfiltered; a purge leaks nothing;
  - end to end: q12 and q14 read fewer probe rows and still match.
- **Gain:** +7 filters, so **24**. Visible today in q12 and q14; q15, q17, q20 and q22 once they run. The size limit decides whether q12's `orders` filter (about 30M keys at SF1000) gets exact keys or only min/max. Measure both.
- **Quick experiment first** (an hour): raise `broadcast_row_limit` (default 15,000,000) for the session. Some partitioned joins then become broadcast joins, which #2062 already covers. This tells us how much of class D a session setting can avoid. It doesn't replace this step.

### Step 3: inject filters into Sirius's own scan filters (engine + CN)

The semi-join rewrite works, but it costs a hash join over the whole scan. It only takes exact keys of a signed integer type, and it saves no I/O. Sirius already has a better place for this: the scan's dynamic-filter channel (`sirius_dynamic_filter_set`, `sirius_physical_table_scan.hpp`). That channel supports:
- IN-list, Bloom, and min/max zone maps (the zone maps prune parquet row groups);
- integer, DATE, TIMESTAMP, DECIMAL32/64 and STRING keys.

What it lacks is an entry point from outside.

- **New FFI call:**
  - `Fragment::add_scan_filter(scan, column, keys | bloom | min-max)`;
  - it registers a producer on that scan's filter set during `build()`;
  - it pushes the filter before `run()`;
  - the producer must be registered before `make_gpu_scan_leaf`, or the channel is dropped.
- **Delivery:** pass the producers in the way byte ranges are, as a `ClientContextState` registered by `lower_substrait`.
- **Switch classes A and D to it.** Keep the semi-join as a fallback until the injected path is at least as fast on q05, q07, q08 and q09.
- **Gains:**
  - row-group pruning, plus filtering during decode;
  - Bloom filters for large partitioned shares (Step 2's min/max fallback goes away);
  - every key type Sirius supports.
- **Tests:**
  - an FFI test: an injected filter prunes row groups and rows;
  - each key type;
  - q05, q07, q08 and q09 no slower than with the semi-join.

### Step 4: same-fragment filters (class B, 14 filters; CN only once Step 3 exists)

All 14 are local filters. In 12 of them the scan is directly under the join, or under one projection. A semi-join there repeats the join's own work and gains nothing. The value is at the scan: row groups not read, rows not decoded. So this step waits for Step 3.

- **No engine flag needed.** When a receiver is ready, its build exchange is complete (see "Two facts" above). The CN reads the keys from the ready inputs, as #2062 reads them for class A, and injects them into the fragment's scan with Step 3's call before `run()`.
- This replaces the earlier idea of a "trusted stream build" flag in Sirius's own filter planner. It needs no change to `sirius_plan_comparison_join.cpp`.
- **Exception:** q02 rf4 and rf5 have joins between the scan and the join that builds the filter. There a filter cuts intermediate work too, so they gain even without row-group pruning.
- **Tests:**
  - a CN test: a ready receiver injects its own build's keys into its scan;
  - q19 reads fewer `lineitem` rows and still matches.
- **Gain:** +14 filters, so **38**. Visible today only in q19 (and q11 once its 0-row result is explained). The others need coverage fixes first.
- **Measure before keeping it on by default.** `l_partkey` and `ps_suppkey` aren't clustered in the files, so row-group pruning may skip little. If a filter costs more than it saves, add a selectivity floor (distinct keys / range) below which it's skipped.

### Step 5: class C targets (1 filter)

- **q02 rf6:** the scan is in a non-leaf fragment that isn't the join's.
  - Generalize #2062's deferral to any fragment whose scan probes a filter built on its CN.
  - Guard against deadlock first: defer only when the build exchange doesn't depend on the probing fragment's output, checked over the query's fragment graph.
  - Gain: +1, so **39**.
- **q15 rf0:** its target is an aggregation node. It isn't applied, which matches 4.1.3 (see Step 1).

### Step 6: the StarRocks wire protocol (only for clusters that mix CNs and BEs)

Needed only once CNs and StarRocks BEs share a query ([cpu-fallback-and-deployment.md](cpu-fallback-and-deployment.md) Step 3). The contract (4.1.3, `8a8e186`):

- **RPC:** `PInternalService.transmit_runtime_filter` (`PTransmitRuntimeFilterParams`: `query_id`, `filter_id`, `finst_id`, `data`, `is_partial`).
- **Merge node:** the worker running instance 0 of the root fragment, fixed by the FE. It waits for `runtime_filter_builder_number` partials and has no timeout. One lost partial means no filter at all.
- **Forwarding:** the merge node sends to some targets, and each target forwards to its share of `forward_targets`. A CN that receives a filter may have to pass it on.
- **Broadcast filters** skip the merge node. Up to 3 senders send straight to the probers, and a prober keeps the first copy. Above 128 KiB (`deliver_broadcast_rf_passthrough_bytes_limit`) only the lowest instance sends, through the forwarding tree.
- **Encodings a BE produces:**
  - min/max plus Bloom;
  - `BITSET_FILTER` for a dense small integer broadcast build (`enable_join_runtime_bitset_filter`, on by default);
  - `EMPTY_FILTER` (min/max only) when a build exceeds `global_runtime_filter_build_max_size` (64M rows), or when any partial of a merged filter was min/max only;
  - `IN_FILTER` for aggregation filters.
- **Plan:**
  - **Send** only `EMPTY_FILTER`, with min/max always filled in. A BE merge node then degrades the whole merged filter to min/max, so no Bloom compatibility is needed.
  - **Receive:** accept `EMPTY_FILTER` and `IN_FILTER` first. Then Bloom and bitset, but only with byte-level tests against StarRocks' C++. A mismatched hash silently drops rows.
  - **Merged filters:** use the slice layout (`layout`, exchange hash XXH3 with seed `0x9E3779B1`, `(h*N)>>32`) only for scans. Never apply a sliced filter to a join node.

## Expected effect (SF1000)

| after step | filters applied | queries completing today that get filtered |
|---|---|---|
| today (#2062) | 16 | q05, q07, q08, q09 |
| 1 | 17 | same |
| 2 | 24 | + q12, q14 |
| 3 | 24, with pruning and every key type | same, faster |
| 4 | 38 | + q19 |
| 5 | 39 | same (q02 doesn't complete yet) |

The rest of the gain arrives as coverage lands: q02, q10, q11, q15, q16, q17, q20, q21 and q22 ([tpch-coverage-plan.md](../tpch-coverage-plan.md)).

## How this lands as PRs

#2062 is ready for review. Adding these steps to it would make it much harder to review. Each step is its own PR, stacked on #2062.

The order of these steps relative to the TPC-H coverage work is in [tpch-coverage-and-filters-plan.md](../tpch-coverage-and-filters-plan.md):
- Step 1 lands at M0;
- Step 2 at M3, once q15 and q20 run;
- Steps 3 and 4 at M6, once every class B query runs;
- Step 5 at M7.

## Open questions

- **For the Sirius engine:** is a producer registered from the FFI (`add_scan_filter`) the right seam for Step 3, or would he rather the publish plan be built from outside?
- **For StarRocks:**
  - whether a later release will stop attaching filters to join nodes, or start evaluating them;
  - whether `runtime_filter_params` can also reach the build fragments' instances, so a build CN doesn't need it forwarded from the root's CN (Step 2).
- **Ours:**
  - Step 2's exact-key size limit;
  - whether Step 4's filters pay off on unclustered keys.

## Out of scope

- Filters built by TopN and aggregation nodes (StarRocks never waits for them).
- Installing a filter mid-scan after a timeout, as StarRocks does.
- Reusing filters across queries.
