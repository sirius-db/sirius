# PLAN-07 — TPC-H q15 returns 0 rows intermittently (exact float equality against a GPU aggregate)

**Status:** analysis + plan only. No code written, no benchmark run by this plan's author.
**Written for a fresh session with zero prior context.** Everything you need is here or cited by
`file:line`.

Repo: `/home/ubuntu/sirius`, branch `demo-multi-cn` (default branch is `dev`).
Box: 2× RTX PRO 6000 Blackwell, 2 CNs (one per GPU), Sirius running as a StarRocks compute node.

---

## 0. Read this first — three corrections to the folklore

Before you act on anything, three things that the informal hand-off gets wrong. All three are
verified against files in this tree.

### 0.1 The "make GPU reductions deterministic" fix is ALREADY IMPLEMENTED, and q15 still flakes

Commit `5d149277` — *"fix(op): bit-stable float grouped sums via the sort-based groupby path"*,
2026-08-07, **an ancestor of the current HEAD `7af763c0`** (`git merge-base --is-ancestor
5d149277 HEAD` → true).

It canonicalises row order before every float SUM, at **both** aggregation stages:

- local/partial aggregate — `src/op/aggregate/gpu_aggregate_impl.cpp:145-170` and `:341-350`
- merge aggregate — `src/op/merge/gpu_merge_impl.cpp:200-221` and `:285`
- ungrouped merge reduce — `src/op/merge/gpu_merge_impl.cpp:125-134` (sorts the partials
  before `cudf::reduce`)
- helpers — `is_order_sensitive_sum()` at `src/op/aggregate/aggregate_op_util.cpp:225-229`,
  `canonicalize_row_order()` at `src/op/aggregate/aggregate_op_util.cpp:231-249`

The header comment at `src/include/op/aggregate/aggregate_op_util.hpp:129-138` **names TPC-H q15
explicitly** as the motivating case. So option (b) in the original framing is not an unexplored
option — it is a shipped fix that did not close the hole. **Any plan that proposes "make
reductions deterministic" as new work is proposing a rewrite of existing code, not a greenfield
change.** Section 3 explains precisely why the shipped fix is insufficient.

### 0.2 The 30-run repro ran against the DECIMAL dataset, not the float64 twin

The hand-off says the datasets are "SF100 float64 at `/home/ubuntu/tpch_parquet_sf100_f64`".
**That path does not exist.** Verified:

```
/home/ubuntu/tpch_parquet_sf100 -> /opt/dlami/nvme/tpch/tpch_parquet_sf100      # DECIMAL
/opt/dlami/nvme/tpch/tpch_parquet_sf100_f64                                     # the f64 twin
```

and the repro's own generated SQL, still on disk, pins the dataset it used:

```
$ grep -m1 file:// /opt/dlami/nvme/sirius-build/q15repro/A.sql
customer AS (SELECT * FROM FILES("path"="file:///home/ubuntu/tpch_parquet_sf100/customer/*.parquet",...
```

`/opt/dlami/nvme/sirius-build/q15-repro.sh:17` defaults `D=/home/ubuntu/tpch_parquet_sf100`, and
the artifacts show it was taken. **The 13/30 measurement is a DECIMAL-dataset measurement.**

This matters because it kills the tempting "just use the float64 dataset" non-fix, and because it
identifies a third defect (§0.3). q15 flakes on *both* datasets — see the corpus in §1.3, where
`sf100-float64-dataset.csv` and `sf100-float64-q10fixed.csv` both contain q15 wedges.

### 0.3 On the DECIMAL dataset the sums are floats anyway — the translator lowers them

`experimental/starrocks/crates/starrocks-plan-translator/src/type_mapper.rs:220-245`: any
StarRocks DECIMAL with **precision > 18** is mapped to Substrait `Fp64`. Only precision ≤ 18
survives as a real decimal.

q15's verbose plan (`experimental/starrocks/benchmarks/tpch/plans/q15.verbose.txt`) shows the
relevant slots are all above that line:

| plan site | declared type | maps to |
|---|---|---|
| `q15.verbose.txt:144` `l_extendedprice * (1 - l_discount)` | `DECIMAL128(31,4)` | **FP64** |
| `q15.verbose.txt:137,160` partial `sum(...)` | `DECIMAL128(38,4)` | **FP64** |
| `q15.verbose.txt:55` join conjunct `43: sum = 62: max` | `DECIMAL128(38,4)` | **FP64 equality** |

So the FE plans exact decimal arithmetic and the CN executes IEEE-754 doubles. This is the same
defect already documented as the open decimal-aggregation work item
(`experimental/starrocks/benchmarks/tpch/REPRODUCE.md:81-83`) and as the ~0.147 % q09 / ≤0.39 %
q03/q10 drift (`bench/rtxpro6000-2gpu/TPCH-STATUS.md:92,111`). **q15's flake and the decimal
drift are two symptoms of one lowering decision.** The hand-off asked not to conflate them; the
correct statement is stronger than "related": one candidate fix (§5, option E) kills both, which
is why it deserves a place on the ballot the original framing did not give it.

---

## 1. Problem statement

### 1.1 The query

`experimental/starrocks/benchmarks/tpch/queries/q15.sql` (stock TPC-H text, no deviation). A CTE
computes per-supplier revenue over a one-quarter window, and the outer query selects the supplier
whose revenue equals the maximum:

```sql
revenue AS (
    SELECT l_suppkey AS supplier_no,
           sum(l_extendedprice * (1 - l_discount)) AS total_revenue
    FROM lineitem
    WHERE l_shipdate >= DATE '1996-01-01' AND l_shipdate < DATE '1996-04-01'
    GROUP BY supplier_no
)
SELECT s_suppkey, s_name, s_address, s_phone, total_revenue
FROM supplier, revenue
WHERE s_suppkey = supplier_no
  AND total_revenue = (SELECT max(total_revenue) FROM revenue)   -- q15.sql:34-37
ORDER BY s_suppkey;
```

The final predicate is an **exact `=` between two floating-point values that the engine computes
by two separate reductions**. Nothing in the query text tolerates a ULP.

### 1.2 Measured evidence (the 30-run, 3-arm repro)

Harness: `/opt/dlami/nvme/sirius-build/q15-repro.sh` (**read it — it is the reference structure
for this whole class of investigation**). Three arms, same cluster, same data, N=30:

| Arm | Query | Result |
|---|---|---|
| **A** | q15 verbatim | **13/30 returned a row** (17 runs silently returned 0 rows) |
| **B** | `total_revenue >= 0.9999999 * (SELECT max(...))` | **30/30** |
| **C** | probe: `SELECT count(*), max(total_revenue), sum(total_revenue) FROM revenue` at full precision | **two distinct max values across the 30 runs** |

Arm C's two values, still on disk at `/opt/dlami/nvme/sirius-build/q15repro/maxvals.txt` (30
lines, 2 distinct):

```
2383449.2756
2383449.2756000003
```

These are **adjacent doubles** — a 1-ULP difference, relative 1.3e-16.

**Arm C is the most important arm and its significance is easy to miss.** Arm C's query
references `revenue` **exactly once**. It therefore contains only ONE evaluation of the CTE — and
it *still* produced two different answers across runs. So the defect is **not merely** "the CTE is
evaluated twice and the two evaluations disagree". It is:

> **A single evaluation of `sum(l_extendedprice * (1 - l_discount)) GROUP BY l_suppkey` is not
> bit-reproducible run to run on this engine.**

Two evaluations disagreeing is then a corollary. Arm B's 30/30 confirms the mechanism is a tiny
numeric gap, not a missing row or a join bug.

Corroborating independent observation: `bench/rtxpro6000-2gpu/TPCH-STATUS.md:63-69` records that
in a byte-for-byte A/B of two builds, **q01 and q15 differed only in the last float digit**, and
that a control diff of two *pre-change* builds reproduced it — explicitly attributed to "GPU
reduction-order non-determinism (≈1 ULP), pre-existing". q01 has no equality, so there it is
cosmetic. q15 turns the same 1 ULP into a silently empty result.

### 1.3 The failure is a *correctness-shaped* failure

The query does not error, does not time out, does not log anything. It returns **0 rows in
~1.9 s** — a plausible-looking fast success. `bench.sh` records it as a `wedge`
(`bench.sh:54-57` — the header states outright that the script "times and counts rows only — it
does not check answers"). A human reading the CSV sees `q15,1,warm,wedge,1937,0` and reasonably
guesses a hang or an OOM. It is neither.

### 1.4 Historical pass-rate corpus (already on disk — no new runs needed to establish the rate)

`grep -H '^q15,' bench/rtxpro6000-2gpu/results/*.csv`:

| CSV | dataset | q15 runs | passes |
|---|---|---|---|
| `sf100-armA-40g16g.csv` | SF100 decimal | 3 | 1 |
| `sf100-armB-60g32g.csv` | SF100 decimal | 2 | 1 |
| `sf100-freelist-40g16g.csv` | SF100 decimal | 3 | 1 |
| `sf100-regression-c1f73993.csv` | SF100 decimal | 3 | 1 |
| `sf100-q08q09-fixed.csv` | SF100 decimal | 2 | 0 |
| `sf100-q08q09-verified-21of22.csv` | SF100 decimal | 3 | 1 |
| `sf100-decimal-final.csv` | SF100 decimal | 2 | 1 |
| `sf100-float64-dataset.csv` | SF100 **f64** | 3 | 2 |
| `sf100-float64-q10fixed.csv` | SF100 **f64** | 2 | 0 |
| **SF100 subtotal** | | **23** | **8 (34.8 %)** |
| `sf300-float64.csv` | SF300 f64 | 3 | 3 |
| `sf500-float64.csv` | SF500 f64 | 3 | 3 |
| `sf500xcold.csv` | SF500 f64 | 2 | 2 (14154 ms / 13341 ms) |
| **SF300+SF500 subtotal** | | **8** | **8 (100 %)** |

Pooling SF100 with the repro's arm A: **21/53 = 39.6 %** pass, 95 % Wilson CI **[27.6 %,
53.1 %]**.

The SF300/SF500 8/8 has 95 % Wilson CI **[67.6 %, 100 %]** — it is *consistent* with a much lower
flake rate at scale, but with n=8 it is also consistent with 70 %. **Do not conclude "fixed at
scale".** The mechanism (§3) has no scale threshold in it. Treat the larger scales as
*undersampled*, not as clean.

---

## 2. The plan, proven from the plan file: `revenue` IS evaluated twice

This needed no cluster to establish. `experimental/starrocks/benchmarks/tpch/plans/q15.explain.txt`
is already in the tree. Read it. It contains **two independent scans of `lineitem`, each with its
own project + partial aggregate + merge aggregate**:

| | branch feeding the outer join | branch feeding `max(...)` |
|---|---|---|
| scan | `PLAN FRAGMENT 5`, `2:FileScanNode` | `PLAN FRAGMENT 4`, `7:FileScanNode` |
| project | `3:Project`, slot 42 | `8:Project`, slot 60 |
| partial agg | `4:AGGREGATE (update serialize) STREAMING`, `sum(42)` group by `28: l_suppkey` | `9:AGGREGATE (update serialize) STREAMING`, `sum(60)` group by `46: l_suppkey` |
| exchange | `05` HASH_PARTITIONED by `28: l_suppkey` | `10` HASH_PARTITIONED by `46: l_suppkey` |
| merge agg | `6:AGGREGATE (merge finalize)` | `11:AGGREGATE (merge finalize)` |
| then | — | `13:AGGREGATE (update serialize) max(61)`, `15:AGGREGATE (merge finalize) max(62)` |

The two branches carry **different slot ids for the same expression** (28/42/43 vs 46/60/61) —
the unmistakable signature of an inlined, duplicated CTE. There is no `CTEProduce`/`CTEConsume`
/ multi-cast node anywhere in the plan.

The comparison itself is `17:HASH JOIN / join op: INNER JOIN (BROADCAST) / equal join conjunct:
43: sum = 62: max` (`q15.explain.txt`, and `q15.verbose.txt:55`). Two consequences worth stating
plainly:

1. The equality is executed as a **hash-join key match on a double**, not as a scalar comparison.
   Bitwise-unequal keys land in different buckets; there is no "close enough" anywhere on that
   path.
2. `q15.verbose.txt:57` shows a **runtime filter** is also built on it
   (`filter_id = 0, build_expr = (62: max)`). A 1-ULP mismatch therefore kills the row twice —
   once at the runtime filter, once at the join.

**Caveat on the plan file:** `q15.explain.txt` / `q15.verbose.txt` were captured against the
DECIMAL dataset (they show `DECIMAL64(15,2)` source columns). On the float64 twin the same shape
appears with DOUBLE slots. V2 in §4 re-captures both to confirm; the structural conclusion (two
scans, two aggregates) is dataset-independent because it is a CBO decision about the CTE, not
about types.

---

## 3. Root cause: what the shipped determinism fix does and does not guarantee

Three defects stack. Naming them separately is the point of this section, because different
options in §5 address different ones.

### D1 — a single evaluation is not bit-reproducible (this is the live defect)

`canonicalize_row_order()` (`src/op/aggregate/aggregate_op_util.cpp:231-249`) sorts rows by
(group keys, float value) and the caller then declares the keys presorted, routing cuDF onto its
sort-based, atomics-free groupby (`gpu_aggregate_impl.cpp:341-350`). Its documented guarantee
(`aggregate_op_util.hpp:129-138`) is that the result becomes *"a pure function of the row
multiset"*.

**That guarantee is per-operator-invocation, and the invocation is per batch.**
`src/op/sirius_physical_grouped_aggregate.cpp:80-95` loops over `input.get_read_only_batches()`
and calls `local_grouped_aggregate` **once per batch**:

```cpp
for (auto const& input_batch : input_batches) {
  auto result = gpu_aggregate_impl::local_grouped_aggregate(input_batch, group_idx, ...);
  results.push_back(std::move(result));
}
```

So the end-to-end chain is:

```
rows --(batching)--> batches --(canonical per-batch sum)--> partials --(canonical merge)--> total
```

Each arrow after `batching` is bit-stable. **`batching` is not part of the guarantee.** If the
same input rows are cut into batches differently between two runs, the *partial values* differ,
the multiset arriving at the merge differs, and the canonical merge faithfully produces a
different (equally canonical) total. Floating-point addition is non-associative, so
`(a+b)+c ≠ a+(b+c)`; canonicalising the order of a *different decomposition* does not help.

Row→batch decomposition can vary run to run through: FE scan-range splitting and assignment
(`experimental/starrocks/crates/starrocks-plan-translator/src/scan_paths.rs:27-30,59-80` — ranges
arrive per node or per driver sequence, and `:112` notes ranges can also be delivered
*incrementally* via `deliver_scan_ranges`), pipeline DOP, `scan_task_batch_size` /
`concat_batch_bytes` interacting with memory pressure, and arrival-order-dependent concatenation.
**V4 in §4 is the experiment that confirms or refutes this.** It is the load-bearing hypothesis
of this plan and it is currently **UNVERIFIED**.

Consistency check in favour of D1: it predicts exactly what arm C measured (single evaluation,
two distinct results) and what `TPCH-STATUS.md:63` measured on q01 (a query with a single
aggregate and no equality, differing in the last digit between builds *and* between two runs of
the same build).

### D2 — the CTE is evaluated twice (§2)

Even if D1 were fully fixed, D2 would remain a latent hazard: two *structurally different*
evaluations (fragment 5's branch feeds a join; fragment 4's feeds a `max`) can be assigned
different DOP and different scan splits *deterministically*, and still produce two different
decompositions and therefore two different sums. D1 and D2 are independent; either alone is
sufficient to empty the query.

Conversely — and this is why option C is attractive — **fixing D2 alone masks D1 for q15**: with
one materialised `revenue`, both references read the *same* values, so the equality holds
whatever those values are. The answer would still wobble by a ULP run to run, but it would be a
non-empty, oracle-matching answer.

### D3 — the enabling condition: DECIMAL > 18 digits lowers to FP64

`type_mapper.rs:233-245`. If q15's `sum` stayed DECIMAL128 all the way, cuDF's decimal sum is
exact and order-independent, `is_order_sensitive_sum()` would correctly return `false`
(`aggregate_op_util.cpp:225-229` — it is `true` only for FLOAT32/FLOAT64), the equality would be
exact by construction, and D1 and D2 would both be *numerically irrelevant for this query*. D3 is
why a decimal-typed benchmark still exhibits a float defect.

---

## 4. Verification tasks (do these before choosing an option)

Each task states what it settles and what a negative result implies. V1–V3 and V7 are read-only
or planner-only; V4–V6 need a cluster.

**Cluster bring-up (needed from V4 on).** From `bench/rtxpro6000-2gpu/STATUS.md:78-83`:

```bash
GPU_MEM=60GiB STAGING=32GiB HOST_MEM=200GiB HPB=1GiB MBHT=2GiB STB=1GiB CBB=1GiB \
  /opt/dlami/nvme/sirius-build/restart-sf500x.sh
```

For SF100 work use the SF100 restart/sweep scripts in `/opt/dlami/nvme/sirius-build/`
(`restart-A.sh`, `restart-B.sh`, `sweep-f64.sh`, …) and point `TPCH_DATA` at the dataset you
mean (see §0.2 for the real paths).

---

### V1 — confirm the CTE is evaluated twice, on BOTH datasets, at the FE (no GPU)

```bash
SR=/home/ubuntu/sirius/experimental/starrocks
export PATH=$SR/.pixi/envs/default/bin:$PATH
for D in /opt/dlami/nvme/tpch/tpch_parquet_sf100 /opt/dlami/nvme/tpch/tpch_parquet_sf100_f64; do
  sed "s|__TPCH_DATA__|$D|g" $SR/benchmarks/tpch/queries/q15.sql > /tmp/q15.sql
  { echo -n "EXPLAIN VERBOSE "; cat /tmp/q15.sql; } | mysql -h127.0.0.1 -P9030 -uroot --batch
done
```

(The FE must be up to plan, but nothing executes; `EXPLAIN` does not dispatch fragments.)

**Expect:** two `FileScanNode`s over `lineitem`, two `AGGREGATE (update serialize) STREAMING`
nodes with different slot ids — i.e. what `plans/q15.explain.txt` already shows.
**If instead you see a `MULTI_CAST` / CTE node:** the CBO already materialises it, D2 is absent,
and the whole diagnosis collapses onto D1 — go straight to V4 and re-read §5 option B′.

### V2 — confirm the Sirius-side types are FP64, not decimal

Grep the FE plan for the slot types (`q15.verbose.txt:137,144,160` in the checked-in copy) and
confirm precision > 18 for the sum. Then confirm the CN's mapping at
`type_mapper.rs:220-245`. Optionally add a temporary trace in the translator, but the static read
is conclusive: `DECIMAL128(38,4)` → `Fp64`.

**Settles:** whether the equality is a float equality at all. If the slot were ≤ 18 digits the
whole defect would be impossible and something else is going on.

### V3 — establish what cuDF guarantees about reduction/groupby order for floats (docs, no GPU)

Two questions, and they have different answers:

1. **Hash groupby SUM** — accumulates via `atomicAdd`; combine order varies with kernel
   scheduling. Sirius's own code comment asserts this at `gpu_aggregate_impl.cpp:145-151` and
   `gpu_merge_impl.cpp:200-205`, and the whole canonicalisation fix exists because of it. Treat
   as established *within this repo*; if you want it from upstream, check the cuDF docs for
   `cudf::groupby` / `sort_groupby` and any statement about float associativity.
2. **Sorted groupby SUM and `cudf::reduce`** — deterministic *given a fixed input order*, which
   is exactly why the fix sorts first. What cuDF does **not** promise is invariance under
   re-partitioning of the input, which is D1.

Run `/module-context "cudf groupby reduction determinism floating point"` to load accurate cuDF
API docs rather than guessing. Record the citation in this file when you have it — currently
**UNVERIFIED against upstream cuDF documentation**; the in-repo assertions are the only source.

### V4 — the decisive experiment: is a single evaluation stable under a FIXED batching?

This is what separates D1 from everything else. Run arm C (the probe from `q15-repro.sh`) N≥30
times **while holding the decomposition fixed**, and compare against N≥30 runs with it free.

Levers to pin, in increasing order of intrusiveness:

- FE session: `SET pipeline_dop = 1; SET parallel_fragment_exec_instance_num = 1;`
- FE session: `SET GLOBAL query_timeout = 1800;` (see §6 gotcha)
- CN config: fix `scan_task_batch_size` and `concat_batch_bytes` (they are already pinned at
  1 GiB in the working config, `STATUS.md:22-27`) and run a single CN (`NUM_CNS=1`) so no
  cross-node exchange participates.

**Reading the result:**

| free-batching distinct values | pinned-batching distinct values | conclusion |
|---|---|---|
| ≥2 | **1** | **D1 confirmed** — the sum is a function of the decomposition. Options B′/C/E are all viable; A and D become "papering over a real nondeterminism". |
| ≥2 | ≥2 | D1 confirmed *and* something below the batching layer is also nondeterministic (e.g. the canonicalisation is not actually firing — check `is_order_sensitive_sum` sees FLOAT64, not a decimal). Investigate before choosing an option. |
| **1** | 1 | D1 refuted at this scale; the flake is then pure D2 (two structurally different evaluations). **Option C becomes the obvious and sufficient fix.** |

Instrument with `SIRIUS_LOG_BACKEND=spdlog` (§6 gotcha) and look for the per-operator batch counts
in the engine log; the `log-analyzer` skill reads these.

### V5 — confirm which Sirius operator actually runs each aggregate node

Establish that q15's aggregates go through the canonicalising path at all. The chain to confirm:
StarRocks `AGGREGATION_NODE` → Substrait aggregate rel
(`crates/starrocks-plan-translator/src/node_translator.rs:237,765-815`) → Sirius
`HASH_GROUP_BY` (`src/op/sirius_physical_grouped_aggregate.cpp:63`) → `local_grouped_aggregate`
(`:85`), and the merge stage → `merge_grouped_aggregate`
(`src/op/sirius_physical_grouped_aggregate_merge.cpp:227`).

Method: run q15 with `SIRIUS_LOG_BACKEND=spdlog SIRIUS_LOG_LEVEL=debug` and read the operator
trace, or use the `log-analyzer` skill. **If some other operator (a streaming aggregate path)
handles node 4/9, the canonicalisation may never fire and D1 has a much more mundane cause.**
Currently **UNVERIFIED**.

### V6 — quantify the baseline properly (this is also the "before" arm of §7)

Run `q15-repro.sh 30 <dataset>` on the **decimal** dataset and again on the **f64** dataset.
Record arm A pass counts and the arm C distinct-value set for each. This gives two baselines with
CIs and confirms §0.2's claim that the dataset is not the variable.

### V7 — sweep the other 21 queries for the same latent hazard (no GPU) — see §8

---

## 5. The options, honestly costed

The genuine tension: **this is not obviously a bug to fix.** TPC-H q15's exact float equality is
a flaw in the *query* when run against any engine with nondeterministic reduction order — and
that includes essentially every parallel engine, on CPU too. But a benchmark that is
intermittently red for a non-defect is corrosive: it trains everyone to ignore red.

Five options, not four. The fifth (E) is not in the original framing and, in this author's
view, is the one that pays for itself.

---

### Option A — do nothing; document it

**What it costs.** Nothing to build. The recurring cost is real and compounding: `q15` shows as
`wedge` in every sweep CSV, `README.md:43` and `TPCH-STATUS.md:105` already carry "1 flaky —
q15" as a permanent asterisk, and every future sweep needs a human to remember that this
particular red is fine. That is precisely the habit that let the q08/q09 "no parked sender
output" masking survive (`QUERY-DEVIATIONS.md`, "Why" section).

**What it risks.** A *different* q15 regression — a real one — would be invisible, because q15 is
already expected to fail sometimes. It also leaves D1 undiagnosed, and D1 is a general
"aggregate results are not reproducible" property that will surface again (it already has, on
q01, as a byte-diff).

**Effort.** ~1 hour: add a q15 stanza to `QUERY-DEVIATIONS.md` and a note in the sweep runbook.

**How to measure.** N/A — that is the problem.

**Verdict:** acceptable only as a *stopgap* while a real fix lands, and only if paired with a
harness change so the run is recorded as `flake-known` rather than `wedge` (that harness work is
PLAN-05's territory).

---

### Option B — make GPU reductions deterministic

**Already done.** See §0.1. Re-proposing it is proposing to *strengthen* an existing fix, which
is option B′.

---

### Option B′ — close the remaining determinism gap: make the sum invariant under re-batching

The only way to make a float sum independent of the decomposition is to stop relying on float
addition's order. Three sub-approaches:

- **B′-1: fixed-point / integer accumulation.** Scale each value into a 128-bit integer
  accumulator (the values *are* decimals with known scale — see D3), sum exactly, convert back.
  Integer addition is associative, so the result is a pure function of the multiset, full stop.
  This is essentially option E arrived at from the other direction.
- **B′-2: reproducible summation** (Kahan/Neumaier compensated, or Demmel–Nguyen
  pre-rounding/binning). Gives bit-reproducibility without changing types.
- **B′-3: force a global canonical order.** Materialise all input rows of the aggregation into
  one canonical order before any partial aggregation. This is the naive reading of the existing
  fix, and it destroys the streaming partial-aggregate design.

**What it costs.** B′-3 is a non-starter on the hottest code path — it converts a streaming
partial aggregate into a full materialisation + global sort. B′-1/B′-2 cost extra work *per
element* in the reduction: a compensated sum is ~2-4× the FLOPs of a naive sum, though these
kernels are memory-bound at SF100+, so the wall-clock cost may be small. **Currently unmeasured —
`UNVERIFIED`.** The existing canonicalisation already pays a sort; B′-1 could plausibly *remove*
that sort (an exact accumulator needs no canonical order), making it net-neutral or faster.

**What it risks.** Touching the aggregate hot path for every query, not just q15. Requires
regression across the full 22-query sweep at SF100/SF300/SF500 with timing deltas.

**Effort.** B′-1: moderate — a custom cuDF aggregation or a pre/post-scaling pass around an
integer SUM, plus type plumbing to know the scale. B′-2: harder inside cuDF's groupby (needs a
custom aggregation or a hand-written kernel).

**How to measure.** V4's pinned-vs-free experiment, plus §7's arms, plus the full sweep for
timing regression (compare against `results/sf500-float64.csv` and `sf100-decimal-final.csv`).

---

### Option C — CTE reuse: materialise `revenue` once

**The intellectual case is strong.** The query says "the max of *this relation*". Evaluating a
relation twice and getting two different relations is the actual anomaly; everything downstream
is a consequence. Fixing C makes the equality trivially true *by construction*, which is a
qualitatively better outcome than making it true *by numerical luck*.

**And StarRocks already has the machinery.** Verified in the vendored FE:

- `cbo_cte_reuse`, default **`true`** —
  `experimental/starrocks/starrocks/fe/fe-core/src/main/java/com/starrocks/qe/SessionVariable.java:499,1505-1506`
  and `starrocks/docs/en/sql-reference/System_variable.md:279-284`. Note the doc's caveat: the
  effective value is `cbo_cte_reuse AND enablePipelineEngine`.
- `cbo_cte_reuse_rate` (alias of the invisible `cbo_cte_reuse_rate_v2`), default **`1.15`** —
  `SessionVariable.java:500-502,1509-1511`. The code comment at `:1508` is the key:
  > `// -1 (< 0): disable cte, force inline. 0: force cte; other (> 0): compute by costs * ratio`
- The decision logic — `starrocks/fe/fe-core/src/main/java/com/starrocks/sql/optimizer/CTEContext.java:157-189`
  (`needInline`) and `:206-235` (`isForceCTE`). With `inlineCTERatio == 0`, `needInline` returns
  `false` and `isForceCTE` returns `true` — **materialisation is forced**, no cost model
  involved. With the default 1.15 the choice falls to the CBO
  (`sql/optimizer/cost/CostModel.java:557-569`, `visitPhysicalCTEAnchor`).
- `cbo_cte_max_limit` = 10, `cbo_cte_force_reuse_node_count` = 2000 —
  `SessionVariable.java:1513-1519`.

**Why it inlines today (hypothesis, consistent with the known q08/q09 finding).**
`visitPhysicalCTEAnchor`'s cost is proportional to `cteStatistics.getOutputSize(...)`. The FE has
**no statistics for `FILES()` external scans** — every node reports `cardinality: 1`
(documented at `experimental/starrocks/benchmarks/tpch/QUERY-DEVIATIONS.md`, "Why" section, which
cites `plans/q08.verbose.txt`). With a size of ~0, materialising looks free *and* inlining looks
free, and the tie resolves to inline. This is the **same root cause** as the q08/q09 cross-join:
missing external-scan statistics. **UNVERIFIED** — V1 plus a `SET cbo_cte_reuse_rate=0` A/B
settles it in minutes.

**What it costs — and here is the blocker.** StarRocks implements CTE reuse with a
**`MULTI_CAST_DATA_STREAM_SINK`**. The Sirius CN **refuses every sink type except
`DATA_STREAM_SINK`**:

`experimental/starrocks/src/compute_node_service.rs:873-881`
```rust
// Accepting an unhandled sink discards the fragment's whole output: its consumers wait
// forever, the FE's fetch_data long-poll times out, and its serial channel wedges
// cluster-wide. Refuse by name so the FE error says which plan shape is unsupported.
if sink.type_ != TDataSinkType::DATA_STREAM_SINK {
    return Err(format!("{} carries a {} output sink, which this CN does not support", ...));
}
```

`MULTI_CAST_DATA_STREAM_SINK` appears in this file **only** in the error-name helper
(`:1317`). A repo-wide grep for `multi_cast|multicast|MULTI_CAST` across
`experimental/starrocks/crates/**/*.rs` returns **zero hits** — the translator has no concept of
it either.

So option C is: *enable a session variable* (5 minutes) **and then implement multi-cast sink
support in the CN** (a genuine feature — one producer fragment fanning its output to N distinct
consumer fragments, with lifetime/refcount semantics for the staged batches, on a codebase whose
`cancel_plan_fragment` is still a stub per `STATUS.md:42` and which already leaks parked sender
outputs per `STATUS.md:41-42`).

**Effort.** Session-variable experiment: **1 hour**. Multi-cast sink in the CN: **weeks**, and it
lands squarely on the same exchange/parking machinery that PLAN-01 and PLAN-02 are already
rewriting. Sequencing it before those is asking for a merge conflict with the memory model.

**What it risks.** A new sink type on a CN that cannot yet cancel fragments; a fan-out producer
whose output must be retained until *all* consumers drain, which is exactly the retention pattern
that PLAN-02 identifies as the 11.3 GiB leak.

**How to measure.** V1 with and without `cbo_cte_reuse_rate=0`; then §7's arms.

---

### Option D — change the query text to a tolerance

Replace `= (SELECT max(...))` with `>= 0.9999999 * (SELECT max(...))` — arm B, already measured
at 30/30.

**What it costs.** Minutes. There is precedent:
`experimental/starrocks/benchmarks/tpch/QUERY-DEVIATIONS.md` already records hand-reordered
`FROM` clauses for q08 and q09, with an explicit rule that *both engines in an A/B must use the
same text*.

**What it risks — and this is the part that must be stated out loud.** **The correctness gate
cannot see it.** `oracle.py` reads the *same* `queries/*.sql` files the benchmark runs
(`/opt/dlami/nvme/sirius-build/oracle.py:4-6,19,23-24,53` — "This runs the **SAME** SQL through DuckDB
over the same parquet"), rewriting only the `FILES(...)` → `read_parquet(...)` path. So editing
`q15.sql` changes the oracle's query *identically*, and `compare.py` will report MATCH. **A query
text change is structurally invisible to the only correctness check we have.** That is a blind
spot in the harness, independent of q15, and worth its own line in PLAN-05.

Also: `0.9999999` is a relative tolerance of 1e-7, about **nine orders of magnitude looser** than
the 1e-16 defect it papers over. It would mask a genuine 1e-8 arithmetic regression. If this
route is taken at all, the tolerance should be near the ULP (e.g. `>= (1 - 1e-13) * (...)`), and
the deviation recorded in `QUERY-DEVIATIONS.md` with the measured evidence.

**How to measure.** §7 arms A vs D.

---

### Option E — stop lowering DECIMAL(19..38) to FP64 (fixes q15 *and* the documented decimal drift)

`type_mapper.rs:233-245` maps precision > 18 to `Fp64`. cuDF has `DECIMAL128` (up to 38 digits)
and Substrait's `Decimal` type also allows precision ≤ 38 — the translator's own guard rejects
only **> 38** as unmappable (`:226-231`). So the 19..38 window appears to be lowered by *choice*,
not by an expressed limitation; **no comment in the file explains why**, and the only
documentation of it is as an open work item (`REPRODUCE.md:81-83`).

If that window is mapped to a real decimal:

- `is_order_sensitive_sum()` returns `false` for decimal (`aggregate_op_util.cpp:225-229`), the
  sum accumulates exactly, and it is order-independent **and** decomposition-independent. **D1
  and D2 both become numerically irrelevant for q15** — the equality is exact by construction, on
  the decimal dataset.
- The ~0.147 % q09 drift and the ≤0.39 % q03/q10/q01/q05/q07/q14/q19 drift
  (`TPCH-STATUS.md:92,111`, `TPCH-SWEEP-RUNBOOK.md:338`) go away. That is **seven-plus queries
  currently reported as "drifting low"** turning exact.

**What it costs.** Real work: DECIMAL128 arithmetic support through the expression path
(multiply with scale promotion is the hard part — cuDF decimal binary ops have strict
scale rules), the merge/partial intermediate types (see the already-handled "the FE declares the
intermediate sum slot as DECIMAL128 — the lie" case at
`crates/starrocks-plan-translator/src/partial_state.rs:5-10,160-170`), and wire type parity
(`experimental/starrocks/src/wire_type_parity.rs:176` explicitly enumerates the >18-digit decimal
whose mapping lowers to FP64 — that test will need updating). DECIMAL128 arithmetic is also
slower than FP64 on GPU.

**What it risks.** The largest blast radius of any option — it changes the numeric type of the
hottest column in TPC-H. Needs the full sweep at all three scales with timing deltas.

**What it does NOT fix.** The **float64 twin datasets**. There `l_extendedprice` really is a
DOUBLE and no type mapping can make the sum exact. q15 on `*_f64` would still need B′ or C. But
the f64 twins exist *because of* D3 (they were built to isolate the decimal drift — 7.71e-15 vs
9.57e-04 on the same expression); if D3 is fixed, their reason for existing largely evaporates
and the primary benchmark returns to the stock decimal schema, which is what TPC-H specifies
anyway.

**Effort.** Weeks, but it is work already on the roadmap under a different name
("the open decimal-aggregation work item").

**How to measure.** Full 22-query sweep vs `compare.py` at SF100/300/500, watching both the
drift columns and q15's pass rate; §7's arms for q15 specifically.

---

## 6. RECOMMENDATION

**Do all three of these, in this order:**

### Now (this week) — Option D-minus, plus honest bookkeeping. ~half a day.

1. Do **not** change `q15.sql`. Instead make the harness able to say "known numeric flake"
   rather than "wedge", so the sweep stops being ambiguously red. (Coordinate with PLAN-05,
   which owns `bench.sh`'s missing correctness gate.)
2. Record the q15 finding in `QUERY-DEVIATIONS.md` — **as a non-deviation**, with the measured
   evidence and the explicit note that the oracle runs the same file so a text change would be
   invisible to `compare.py`. That blind spot deserves to be written down whether or not anyone
   ever exploits it.

Rationale for *not* editing the query: the tolerance edit buys a green cell and buys nothing
else, it is invisible to the only correctness gate we have, and 1e-7 is nine orders of magnitude
too loose to be a safe permanent guard. If schedule pressure forces it, use `>= (1 - 1e-13) *`
and document it.

### Next (1 day of cluster time) — settle C's cheap half and V4.

3. Run **V1 with `SET cbo_cte_reuse_rate = 0`**. If the plan collapses to one `lineitem` scan
   with a `CTEProduce`/multi-cast, we learn (a) the FE will do the right thing on request and
   (b) exactly which CN feature is missing. If the CN then refuses the fragment with
   *"carries a MULTI_CAST_DATA_STREAM_SINK output sink, which this CN does not support"*, that
   error message **is** the deliverable: it converts option C from "investigate" into a
   scoped feature request with a one-line acceptance test.
4. Run **V4**. Its outcome decides whether D1 is real. This is one afternoon and it is the
   highest-information experiment in this plan.

### The real fix — Option E, with B′-1 as its implementation core.

5. **Recommended fix: stop lowering DECIMAL(19..38) to FP64** (option E), implemented with exact
   integer/fixed-point accumulation where cuDF's decimal support falls short (option B′-1).

**Reasoning.** Judge the options by how many separate reported problems each one closes:

| Option | q15 (decimal ds) | q15 (f64 ds) | q09/q03/q10 drift | q01 byte-instability | general "aggregates reproducible" |
|---|---|---|---|---|---|
| A do nothing | no | no | no | no | no |
| D tolerance | masks | masks | no | no | no |
| C CTE reuse | **yes** | **yes** | no | no | no |
| B′ reproducible sum | **yes** | **yes** | no¹ | **yes** | **yes** |
| **E decimal fidelity** | **yes** | no² | **yes** | **yes**³ | partially³ |

¹ B′ makes the drift *reproducible*, not *correct* — it is still ~0.15 % low.
² needs B′ as well for genuinely-double data.
³ for the decimal-typed columns, which is every drifting column in TPC-H.

E is the only option that turns a *wrong* answer into a *right* one rather than turning an
intermittently-wrong answer into a consistently-slightly-wrong one. It also retires an item
already on the roadmap. C is the most intellectually satisfying framing of the q15 bug
specifically, and if the multi-cast sink were free it would win — but it costs a new sink type on
a CN that cannot cancel fragments and already leaks parked outputs, and it fixes exactly one
query.

**If V4 shows D1 is real** (free-batching gives ≥2 distinct values, pinned gives 1), then B′-1's
exact accumulator is not optional even inside option E — it is what makes E's guarantee hold
under re-batching. The two collapse into one piece of work: *accumulate revenue-shaped sums
exactly*. That framing is the recommendation.

**If V4 shows D1 is NOT real** (both pinned and free give one value), then the flake is pure D2,
option C is sufficient and minimal, and the recommendation flips to C — accept the multi-cast
sink cost, because then a *single* well-scoped feature makes the query correct by construction.

---

## 7. Validation procedure

Reuse `q15-repro.sh`'s structure verbatim — three arms, N runs, one summary line each. Copy it to
a new script rather than editing it in place; the original is the reference artifact for the
30-run baseline.

### Arms

| Arm | Purpose |
|---|---|
| **A** | q15 verbatim — the metric that matters. |
| **B** | q15 with tolerance — the positive control. Must stay 30/30; if it drops, something *else* broke. |
| **C** | the single-reference max probe at full precision — measures D1 directly by counting distinct values. |
| **D** | *(new)* q15 verbatim with the candidate fix enabled (`SET cbo_cte_reuse_rate=0`, or the E/B′ build). |

Run the before-arms and after-arms **on the same cluster boot** where possible; if the fix needs
a rebuild, restart identically and re-run all arms.

### N and confidence

- **N = 30** per arm for the go/no-go decision. Cost: SF100 q15 is ~1.9 s (decimal) / ~3.1 s
  (f64) per run, so 30 runs × 3 arms ≈ 5 minutes of query time.
- Report a **Wilson score 95 % interval**, not a bare fraction. Reference points:
  - 13/30 → 43.3 %, CI **[27.4 %, 60.8 %]**
  - 30/30 → 100 %, CI **[88.6 %, 100 %]**
- **State the ceiling honestly: N=30 with zero failures can only support "≥88.6 % at 95 %
  confidence".** It cannot support "fixed". By the rule of three, demonstrating ≥99 % needs
  **≥300 consecutive passes**. At ~2 s/run that is ~10 minutes — so for the *final* claim, run
  **N=300** on arm A/D. There is no excuse for stopping at 30 for the acceptance run.
- Do the same at **SF300 and SF500** with N≥30 each, because the existing SF300/SF500 evidence is
  only n=8 and cannot distinguish 100 % from 70 % (§1.4).

Wilson interval, for the script:

```python
def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 1.0)
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    m = (z/d) * ((p*(1-p)/n + z*z/(4*n*n)) ** 0.5)
    return (max(0.0, c-m), min(1.0, c+m))
```

### Gotchas that will silently waste a day

1. **`SIRIUS_LOG_BACKEND` accepts only `duckdb`, `spdlog`, `noop` — and on the CN path an unknown
   value is SILENTLY DROPPED.** `src/sirius_context.cpp:1550-1578`: the `else if (db)` branch is
   the only one that throws `InvalidInputException("Unknown sirius_log_backend '%s' (expected:
   duckdb, spdlog, noop)")` (`:1576-1577`), and the CN reaches this function with `db == nullptr`
   from `src/sirius_ffi.cpp:170-178`. A typo — or a plausible-looking value like `file`,
   `stderr`, `console` — therefore produces **no error and no logs**, and you will conclude the
   engine is silent when it is merely unconfigured. Use exactly `SIRIUS_LOG_BACKEND=spdlog`, and
   set `SIRIUS_LOG_DIR` (the CN honours all three `SIRIUS_LOG_*` vars only if at least one is
   set — `sirius_ffi.cpp:173`). This is one of the two measurement defects that invalidated
   earlier data (`STATUS.md:43-44`).
2. **The FE's `query_timeout` defaults to 300 s** (confirmed in the vendored session-variable
   dump, `starrocks/docs/ja/faq/Dump_query.md:81` → `"query_timeout":300`). `bench.sh` never
   raised it (`STATUS.md:43-44`). For SF500 work `SET GLOBAL query_timeout = 1800;`
   (`STATUS.md:26`). Note `q15-repro.sh` also wraps each `mysql` call in `timeout 300`
   (`q15-repro.sh:45,47,49`) — raise that too, or your SF500 arm will report false wedges at
   exactly 300 s.
3. **A restart between arms changes the batching.** If you restart mid-experiment, note it; a
   pass-rate difference across a restart boundary is confounded.
4. **`bench.sh` restarts the cluster after a failing warm run** (`bench.sh:19-22,44-50`) — for
   pass-rate work drive `mysql` directly as `q15-repro.sh` does, not through `bench.sh`.
5. **Row-count is not correctness.** Arm A "passes" at `rows=1`. Also diff the returned
   `total_revenue` against the oracle with `tools/compare.py` — otherwise a fix that returns the
   *wrong* supplier scores as green.

### Acceptance

A fix is accepted when, on the same cluster:

- arm D ≥ **300/300** at SF100 (both datasets), and ≥ **30/30** at SF300 and SF500;
- arm C reports **exactly 1 distinct max value** over ≥30 runs (this is the real determinism
  claim; arm A can pass by luck, arm C cannot);
- `tools/compare.py` reports MATCH for q15 against the DuckDB oracle;
- the full 22-query sweep shows no status regression and no timing regression beyond ±3 %
  (the band used for the last no-regression verdict, `TPCH-STATUS.md:59`).

---

## 8. Broader sweep — which other TPC-H queries are latent instances of this class?

Read all 22 files. The mechanical part:

```bash
cd /home/ubuntu/sirius/experimental/starrocks/benchmarks/tpch/queries
grep -n -E '(=|<|>|<=|>=) *\($' *.sql        # comparisons against a subquery
grep -n -E '= *[0-9]+\.[0-9]' *.sql          # equality against a float literal  -> no hits
```

**Every comparison against a subquery in the 22-query set — the complete list (6 sites):**

| Query | Site | Operator | RHS | Order-dependent? | Risk |
|---|---|---|---|---|---|
| **q15** | `q15.sql:34-37` | `=` | `max(sum(l_extendedprice*(1-l_discount)))` | **YES — sum-derived** | **THE DEFECT.** 0 rows. |
| **q02** | `q02.sql:33-46` | `=` | `min(ps_supplycost)` (correlated) | **No** | **Safe.** `min` *selects* an existing input value bit-for-bit; it performs no arithmetic, so it is order-independent for every type. The LHS is the same base column. The equality is between two copies of the same stored value. The only residual risk is a NaN, which TPC-H data does not contain. |
| **q11** | `q11.sql:24-34` | `>` | `sum(ps_supplycost*ps_availqty) * 0.0001` | YES — sum-derived | **Low.** Strict inequality; a 1-ULP threshold shift only matters for a group within 1 ULP of the cut. Would show as a ±1 row-count difference, never 0 rows. Note q11 is *already* correct-empty at SF500 and the harness misreads it (`TPCH-STATUS.md:95`) — do not confuse the two. |
| **q17** | `q17.sql:19-25` | `<` | `0.2 * avg(l_quantity)` (correlated) | YES — `avg` is sum/count | **Low.** `l_quantity` is integral and small; the *sum* is exact in FP64 (values ≤ 50, group sizes tiny — well inside 2^53), so the avg is exact too. Order-independence follows from exactness, not from luck. |
| **q18** | `q18.sql:29-30` | `>` | `sum(l_quantity) > 300` (literal) | No | **Safe.** Integral sum, exact in FP64. |
| **q20** | `q20.sql:30-39` | `>` | `0.5 * sum(l_quantity)` (correlated) | No | **Safe.** Same reasoning as q18: exact integral sum; `0.5 *` is exact in binary. |
| **q22** | `q22.sql:22-29` | `>` | `avg(c_acctbal)` | **YES — sum-derived over a non-integral column** | **Low-moderate.** `c_acctbal` is a real decimal (lowered to FP64 by D3 if precision > 18 — check the slot type). A 1-ULP threshold wobble flips only a customer whose balance sits within 1 ULP of the mean. Manifests as a ±1 row difference in the grouped output, not an empty result. **Worth one N=30 arm-C-style probe** on `avg(c_acctbal)` to see if it varies. |

**Additional exposure, not in the grep:** any float value that reaches a **hash-join key** or a
**hash-exchange partition key**. `src/exec/streaming_fragment.cpp:129-166` documents that
`DOUBLE` keys hash as-is (via cuDF's `MurmurHash3_x86_32<double>` with
`normalize_nans_and_zeros()`) and `FLOAT`/decimal keys widen to `FLOAT64`. So a float key is
matched *bitwise-equivalently* — a 1-ULP difference does not "almost match", it lands in a
different bucket. q15 is the only query in the set that joins on a computed float
(`q15.verbose.txt:55`), but any future plan that does inherits the same cliff.

**And the cosmetic class:** q01's output already varies in its last float digit run to run
(`TPCH-STATUS.md:63,66-69`). Every `sum(l_extendedprice * (1 - l_discount))` query — q01, q03,
q05, q07, q10, q14, q19 — has the same underlying instability; they are only benign because
nothing compares the result for equality. `compare.py`'s default `rel_tol=1e-6`
(`tools/compare.py:16`) absorbs it. **If B′/E lands, the correct expectation is that these
become byte-stable too** — a good secondary acceptance signal.

**Conclusion of the sweep: q15 is the only query that can return 0 rows from this class.**
q22 is the only other one worth probing. Everything else is either exact-by-integrality or
protected by a strict inequality whose blast radius is one row.

---

## 9. Success criteria

**For this plan (the investigation):**

1. V1 answers, with a pasted plan fragment, whether `revenue` is evaluated twice at
   `cbo_cte_reuse_rate` default **and** at `0`.
2. V4 answers, with a distinct-value count from ≥30 runs each, whether a single evaluation is
   stable under pinned batching. **This is the deliverable that decides the recommendation.**
3. V5 names the exact Sirius operator handling q15's partial and merge aggregates, so it is known
   whether the shipped canonicalisation fires at all.
4. If option C is pursued, the CN's refusal message for a multi-cast fragment is captured
   verbatim — turning "investigate CTE reuse" into a scoped feature with an acceptance test.
5. This document is updated in place with the answers and the `UNVERIFIED` markers cleared.

**For the fix (whichever is chosen):**

6. Arm C: **exactly one distinct value** for `max(total_revenue)` over ≥30 runs, at SF100
   (both datasets), SF300, SF500.
7. Arm A/D: **300/300** at SF100, **30/30** at SF300 and SF500 — reported with Wilson CIs, and
   with the explicit statement of what the interval does and does not support.
8. `tools/compare.py` MATCH for q15 vs the DuckDB oracle.
9. Full 22-query sweep: no status regression, timing within ±3 %.
10. `q15.sql` is **unmodified** from the stock TPC-H text — or, if it is modified, the deviation
    is recorded in `QUERY-DEVIATIONS.md` together with the note that the oracle reads the same
    file and therefore cannot catch it.
11. If option E landed: the drift columns for q01/q03/q05/q07/q09/q10/q14/q19 in the next sweep
    read **exact**, not "≤0.39 % low".

---

## Appendix — file:line index

Everything cited above, in one place.

**The query and the plan**
- `experimental/starrocks/benchmarks/tpch/queries/q15.sql:34-37` — the exact float equality
- `experimental/starrocks/benchmarks/tpch/plans/q15.explain.txt` — two `FileScanNode`s over
  `lineitem` (fragments 4 and 5); `17:HASH JOIN ... 43: sum = 62: max`
- `experimental/starrocks/benchmarks/tpch/plans/q15.verbose.txt:55` — join conjunct types
  `DECIMAL128(38,4)`; `:57` runtime filter on it; `:137,144,160` the sum/expr slot types

**The repro**
- `/opt/dlami/nvme/sirius-build/q15-repro.sh` — the 3-arm harness (dataset default `:17`, arm B's
  `sed` at `:26`, arm C's generator at `:28-40`, the `timeout 300` calls at `:45,47,49`)
- `/opt/dlami/nvme/sirius-build/q15repro/{A,B,C}.sql`, `maxvals.txt` — the artifacts, incl. the
  two adjacent doubles

**The determinism fix that is already in**
- `src/op/aggregate/gpu_aggregate_impl.cpp:145-170` — the canonicalisation gate (names q15)
- `src/op/aggregate/gpu_aggregate_impl.cpp:341-350` — presorted-keys groupby
- `src/op/merge/gpu_merge_impl.cpp:125-134` — sorted partials before `cudf::reduce`
- `src/op/merge/gpu_merge_impl.cpp:200-221` — canonicalisation at the merge
- `src/op/aggregate/aggregate_op_util.cpp:225-229` — `is_order_sensitive_sum` (FLOAT32/64 only)
- `src/op/aggregate/aggregate_op_util.cpp:231-249` — `canonicalize_row_order`
- `src/include/op/aggregate/aggregate_op_util.hpp:119-138` — the stated guarantee
- `src/op/sirius_physical_grouped_aggregate.cpp:80-95` — **per-batch** invocation (the gap)
- `src/op/sirius_physical_grouped_aggregate_merge.cpp:227` — the merge call
- commit `5d149277` (2026-08-07), ancestor of HEAD `7af763c0`

**The type lowering**
- `experimental/starrocks/crates/starrocks-plan-translator/src/type_mapper.rs:220-245` —
  precision > 18 → `Fp64`; > 38 → refuse
- `experimental/starrocks/crates/starrocks-plan-translator/src/partial_state.rs:5-10,160-170` —
  the FE's DECIMAL128 intermediate "lie" vs the DOUBLE wire column
- `experimental/starrocks/src/wire_type_parity.rs:176` — enumerates the >18-digit lowering
- `experimental/starrocks/benchmarks/tpch/REPRODUCE.md:81-83` — it as the open work item

**CTE reuse in the FE**
- `.../qe/SessionVariable.java:499-506` — the five CTE variables
- `.../qe/SessionVariable.java:1505-1519` — defaults: `cbo_cte_reuse=true`,
  `cboCTERuseRatio=1.15`, `cboCTEMaxLimit=10`, `cboCTEForceReuseNodeCount=2000`;
  the `-1 / 0 / >0` comment at `:1508`
- `.../sql/optimizer/CTEContext.java:157-189` (`needInline`), `:206-235` (`isForceCTE`)
- `.../sql/optimizer/cost/CostModel.java:557-569` — `visitPhysicalCTEAnchor`
- `starrocks/docs/en/sql-reference/System_variable.md:279-284` — `cbo_cte_reuse` doc + the
  `AND enablePipelineEngine` caveat

**The CN's sink restriction (option C's blocker)**
- `experimental/starrocks/src/compute_node_service.rs:873-881` — refuses any non-
  `DATA_STREAM_SINK`
- `experimental/starrocks/src/compute_node_service.rs:1317` — `MULTI_CAST_DATA_STREAM_SINK`
  appears only as an error name
- zero hits for `multi_cast|multicast` in `experimental/starrocks/crates/**/*.rs`

**Scan splitting (D1's suspected source)**
- `experimental/starrocks/crates/starrocks-plan-translator/src/scan_paths.rs:27-30,44,59-80,112,154`

**Float hash keys**
- `src/exec/streaming_fragment.cpp:129-166`

**Logging and timeouts**
- `src/sirius_context.cpp:1550-1578` — the backend switch; `:1576` the throw that the CN path
  never reaches
- `src/sirius_context.cpp:1583-1589` — env read + `install_configured_log_sink(nullptr)`
- `src/sirius_ffi.cpp:165-178` — the CN/FFI path, `db == nullptr`
- `starrocks/docs/ja/faq/Dump_query.md:81` — `"query_timeout":300` in the default dump

**Harness and gate**
- `experimental/starrocks/benchmarks/tpch/bench.sh:54-57` — "times and counts rows only — it does
  not check answers"; `:44-50` the restart-on-failure behaviour
- `/opt/dlami/nvme/sirius-build/oracle.py:4-6,19,23-24,53` — runs the **same** SQL files (the blind
  spot)
- `bench/rtxpro6000-2gpu/tools/compare.py:16` — default `rel_tol = 1e-6`
- `bench/rtxpro6000-2gpu/tools/drift.py:7-10` — q15's key column for key-matched drift

**Prior findings referenced**
- `bench/rtxpro6000-2gpu/STATUS.md:22-27` (working config), `:41-44` (leak + the two measurement
  defects), `:66` (this plan's entry), `:78-83` (restart command)
- `bench/rtxpro6000-2gpu/TPCH-STATUS.md:35,63-69,92,95,105,111,114`
- `bench/rtxpro6000-2gpu/TPCH-SWEEP-RUNBOOK.md:326,338,379`
- `bench/rtxpro6000-2gpu/SIRIUS-TUNING-RUNBOOK.md:318-319`
- `experimental/starrocks/benchmarks/tpch/QUERY-DEVIATIONS.md` — the q08/q09 precedent, the
  `cardinality: 1` external-scan statistics gap, and the rule that both engines must run the
  same text

**Open / UNVERIFIED at the time of writing**
- D1's mechanism (batch decomposition varies run to run) — V4 settles it
- Whether the canonicalising operator actually handles q15's aggregates — V5
- Whether `cbo_cte_reuse_rate=0` changes q15's plan — V1
- Upstream cuDF's documented position on float reduction order — V3
- B′/E performance cost — unmeasured
- `STATUS.md:85-86` lists the arena freelist rework, FFI `outstanding()`, `streaming_fragment`
  DOUBLE/FLOAT hash keys and the q08/q09 reorders as *uncommitted*; as of HEAD `7af763c0` they
  are committed (`87c77808`, `7af763c0`) and only docs/CSVs remain staged. Minor, but do not
  trust that paragraph.
