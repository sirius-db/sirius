# Issue #1010 — Dynamic-Filter SIP Benchmark Protocol and Rollout Gates

**Status: pre-registered.** This document is checked in **before** any R2 performance data exists,
per the R1 acceptance criterion *"numeric rollout thresholds are recorded in the checked-in benchmark
specification before R2 performance data is examined."*

**R4 evaluates these gates. R4 does not tune them.** No threshold in §6 may be changed after any
measurement covered by this document has been examined, except under §10.

**Related** (delivered by issue #1010's documentation units and design-review notes; these
files are not all in-tree yet): [SIP v2 design](issue-1010-dynamic-filter-sip-design-v2.md),
[v2 delivery plan](issue-1010-github-delivery-plan-v2.md),
[R1b coverage-gate design](issue-1010-r1b-coverage-gate-design.md).

## 1. Purpose

Decide, from predeclared numbers alone, whether the direct-endpoint dynamic filter ships **default-on**,
**limited** to a narrowed planner allowlist, or **not at all**.

A gate is usable only if it can be evaluated with instrumentation that exists when it is evaluated.
§5 records, per gate, what must exist and which unit owns it. A gate whose instrumentation is missing
at R4 is a **fail**, not a waiver.

## 2. Hardware and software

### 2.1 Topology matrix

| Topology | Status | Gates applied |
|---|---|---|
| 1 GPU (NVIDIA GB10) | **mandatory** | all of §6 |
| 2 GPU | **where available** | all of §6; §6.5 per device |
| 4 GPU | **where available** | all of §6; §6.5 per device |

**The "where available" clause.** A topology not run is recorded as **not evaluated**, which is not a
pass. R4's allowlist may not include a topology that was not evaluated. Concretely: if only the
single-GPU row exists at R4, the only outcomes available are single-GPU default-on, single-GPU
limited ship, or no-ship — the delivery plan's multi-GPU R3 criteria remain unmet and multi-GPU stays
behind the flag.

### 2.2 Software versions

Recorded per run, from `pixi list`, not from this table. The pins below are the *constraint*; the run
record carries the resolved versions.

| Component | Pin / observed |
|---|---|
| CUDA toolkit | `cuda-version = "13.2.*"` (default env); `"12.*"` (`duckdb-python` env) — `pixi.toml:70,82` |
| libcudf | `libcudf = "26.06.*"` — `pixi.toml:71,83` |
| NVIDIA driver | 580.126.09 (GB10 development box, 2026-07) |
| DuckDB submodule | recorded as the pinned commit SHA |
| Sirius | recorded as the merge-base SHA of the measured branch |

**The GB10 memory configuration is part of the record.** The development box requires a
`~/.sirius/sirius.yaml` capping the GPU reservation or it OOMs on unified memory. The §6.5 memory
ceiling is meaningless without it, so the run record embeds that file verbatim.

## 3. Corpus

| Axis | Coverage |
|---|---|
| Primary | TPC-H SF30 Parquet (`test_datasets/tpch_parquet_sf30/`), all 22 queries |
| Native scan | TPC-H SF30 DuckDB-native (`test_datasets/tpch_sf30.duckdb`), all 22 queries |
| Nested / bushy | TPC-H Q2, Q5, Q7, Q8, Q9, Q17, Q20, Q21 |
| Narrow / wide payload | Synthetic probe table, 2 columns vs. 40 columns, identical key distribution |
| Clustered / scattered | Synthetic build keys drawn from a contiguous range vs. uniformly over the domain |
| Selective / non-selective | Synthetic build predicate at **0.1%, 1%, 10%, 50%, 95%** retention |

**The 95% retention point is mandatory and load-bearing.** It is the only point in the corpus that
exercises the domain-coverage gate R1b makes live. Without it, the gate ships unmeasured. The
synthetic tables are DuckDB-native with a declared `PRIMARY KEY` on the build key: the gate arms
only for native scans with proven-unique keys (R1b design §B.4, §D.3), so any other construction
would measure nothing.

Synthetic generation is scripted and checked in alongside the run record, so a re-run reproduces the
same data.

## 4. Protocol

### 4.1 Warm runs (gated)

- **Interleaving:** ON and OFF measured **in the same process/session**, in `ABBA` order per query,
  to cancel session drift.
- **Repetitions:** 1 discarded warm-up + **7 measured** iterations per (query, configuration).
- **Statistic:** the **median** of the 7.
- **Variance rule:** a measurement is admissible when `IQR ≤ 5% of the median`. Otherwise re-run with
  15 iterations and report both. If the 15-iteration IQR still exceeds 5%, the query is marked
  **unstable**.
- **Instability attribution — the exclusion is not cause-blind.** A query unstable in the
  **baseline** (OFF-vs-OFF, §11's calibration set) is measurement noise: it is excluded from §6.2
  and §6.3 and recorded as such, and **more than 3 baseline-unstable queries invalidates the entire
  run**. A query *stable in the baseline* that becomes unstable only under ON is
  **feature-induced instability** and **fails G2 for that query and its §9 class** — it may not be
  excluded, because excluding it would let the feature convert a regression into a discarded
  measurement.

*Why 7 and the median.* Seven is odd (an unambiguous median), sufficient for a stable median and an
IQR, and cheap enough for 22 queries × 2 formats × 2 configurations. The median is used rather than
the mean because GPU wall times carry occasional allocation- and driver-induced long tails, and
because a pre-registered gate must not depend on an outlier-trimming choice made after seeing the
data.

### 4.2 Cold runs (recorded, not gated)

Fresh process, page cache dropped, one iteration per query per configuration.

**Cold runs carry no gate.** On GB10 cold wall time is process-initialization-dominated, so a cold
delta measures CUDA context creation and pool warm-up rather than the feature. Cold numbers are
recorded for I/O-path context and for the §8 profiling decision, and they may not be cited for or
against any §6 gate.

### 4.3 Configurations

`OFF` = `enable_dynamic_filter_sip = false` (the Phase-1 topology).
`ON` = `enable_dynamic_filter_sip = true`, all other settings at their defaults from the delivery
plan's flag table. Any deviation is recorded per run.

## 5. Metrics and the instrumentation ledger

| Metric | Source | Exists as of | Owner |
|---|---|---|---|
| Query wall time | `test/tpch_performance/` harness | today | — |
| Result equivalence | `compare_gpu_vs_cpu` | today | — |
| `producers_enabled`, `publication_attempts`, `publications_finished/failed`, `publications_skipped_source_not_resident` | `SiriusContext::dynamic_filter_stats` | **R1b** | R1b |
| `keys_considered`, `keys_with_known_domain`, `keys_skipped_domain_gate`, `keys_skipped_type_mismatch`, `keys_build_exceeded_domain` | same | **R1b** | R1b |
| `membership_filters_built`, `zone_map_filters_built` | same | **R1b** | R1b |
| `publications_skipped_targets_drained`, `filters_pushed` *(delivery observability; never equality anchors)* | same | **R1b** | R1b |
| Route class per key (scan / direct / none) | telemetry "Opportunity and routing" row | **not yet** | **R2** |
| Producer considered/admitted/rejected with stable reason | same | **not yet** | **R2** |
| Direct-route keys with a scan below them | same | **not yet** | **R2** |
| Batches/rows with a visible filter; attempted / kept / removed | telemetry "Application" row | **not yet** | **R2** |
| Endpoint gate decision and observation count | same | **not yet** | **R2** |
| Mask / gather / apply time | same | **not yet** | **R2** |
| Transient & resident bytes, admitted bound, denial/failure | telemetry "Resources and rollout" row | **not yet** | **R3** |
| Replica bytes; construction/replication latency | telemetry "Publication" row | **not yet** | **R2/R3** |

**Consequence, stated so it is not discovered at R4:** §6.4 (opportunity coverage) is unevaluable
until R2 lands its Opportunity/Application telemetry rows, and §6.5 (memory ceiling) until R3 lands
its Resources row. Both are pre-registered now and become checkable then. §6.1, §6.2, §6.3 and §6.7
are evaluable with what exists after R1b.

## 6. Gates

### 6.1 G1 — Correctness (absolute)

Exact result-set equality between GPU-ON, GPU-OFF, and CPU, for **every** corpus query in **every**
configuration and topology, using the corpus's existing comparison rule (documented float tolerance
for aggregate columns only).

**Any failure is an immediate no-ship and an immediate rollback (§7).** No trade-off applies. G1 is
not weighed against any other gate.

### 6.2 G2 — Maximum regression

- **Per query:** `median_ON ≤ median_OFF × 1.05`, **or** `median_ON − median_OFF ≤ 20 ms`, whichever
  admits the query.
- **Corpus:** `geomean over queries of (median_ON / median_OFF) ≤ 1.00`.

*Why 5%.* §4.1 admits a measurement only when `IQR ≤ 5% of the median`; the median's own uncertainty
is then roughly `IQR / (1.35·√7) ≈ 1.4%` of the median. A 5% threshold is therefore ≈ 3.5σ, so a G2
failure is a signal rather than noise.

*Why the 20 ms floor.* Short queries are dominated by fixed per-query costs, where a pure percentage
gate trips on jitter. The floor is **calibrated from null runs before any ON data is examined**:
§11 records the 95th percentile of `|median_A − median_B|` over 10 OFF-vs-OFF pairs on the target
hardware. **Calibration may only lower the floor, never raise it** — so the procedure cannot be used
to rescue a failing result, and the pre-registered 20 ms is a ceiling on the floor.

*Why a corpus geomean of 1.00.* A feature that is net-neutral across the corpus while regressing
individual queries has no case for default-on; it may still qualify for a limited ship under §9.

### 6.3 G3 — Minimum benefit

- **Default-on:** `geomean over queries of (median_OFF / median_ON) ≥ 1.05`.
- **Limited ship:** `≥ 1.15` on at least **5** corpus queries inside the candidate allowlist, and
  `geomean ≥ 1.00` over that allowlist. The allowlist is not proposed after results are seen: it is
  produced by §9's deterministic rule over the planner classes predeclared there.

*Why 5%.* The benefit bar must exceed the per-query regression tolerance (§6.2), or "the feature
helps" and "the feature is neutral" occupy the same band. It is also the smallest return that
justifies the permanent complexity this feature adds — a direct endpoint operator, an endpoint
channel, device replicas, a source-memory reservation, and a publication state machine. Below it,
the delivery plan's **no-ship** outcome is the correct one, and exists for this case.

*Why 15% on 5 queries for a limited ship.* A limited ship narrows the allowlist to the classes that
passed, so those classes must individually repay the machinery rather than being carried by a corpus
average. Three times the default-on bar is the standard form of "if it only helps a few, it must help
them a lot." Five queries is roughly a quarter of TPC-H — enough that the result is about a *class*
of plans rather than one lucky shape.

*Calibration against a known reference.* A parked in-tree optimization (native late materialization
of strings) triggered on 3 of 22 TPC-H queries and was judged net-zero. G3 and G4 are set above that
reference deliberately: this feature must touch materially more of the corpus than the one that was
parked, or it is the same decision.

### 6.4 G4 — Minimum opportunity coverage

- `≥ 60%` of producing joins **eligible under the R2/R3 supported scope** reach publication with at
  least one filter pushed.
- `≥ 8 of the 22` TPC-H queries apply at least one filter to at least one probe batch.

*Why 60%.* The denominator is already narrowed to *eligible* producers — INNER/left-SEMI, equality,
direct INT32/INT64 keys, complete build, at least one live target. Failing to publish for more than
40% of producers that pass all of that means the gap is a defect or an unmodelled skip, not a scope
boundary, and it must be diagnosed before rollout rather than shipped as a silent loss.

*Why 8 of 22.* See the calibration note in §6.3.

**Instrumentation dependency:** R2's Opportunity/Application telemetry rows (§5).

### 6.5 G5 — Memory ceiling

- Peak device bytes attributable to dynamic filtering `≤ 15%` of the query's measured peak device
  footprint with the feature OFF.
- Transient adder at the endpoint `≤ 2.2 ×` the largest single probe batch's byte size.
- `measured peak ≤ admitted bound ≤ 1.15 × measured peak`.

*Why 2.2×.* Application can co-hold the input batch, a BOOL mask, and a near-input-sized gathered
output on a keep-nearly-all batch — the shape the delivery plan's R2 reservation criterion names
explicitly. That is `1 (input) + 1 (output) + mask`, and 2.2× covers the mask plus allocator
rounding. This is derived from the data contract, not chosen.

*Why 15%.* It bounds the aggregate — source structures, per-device replicas, and endpoint buffers —
at a level where the feature cannot be the cause of an OOM on a workload that fits without it, which
is the operative safety property on a unified-memory GB10.

*Why the two-sided estimator band.* `measured ≤ admitted` is the delivery plan's own R3 criterion: an
optimistic estimator makes admission meaningless. The `admitted ≤ 1.15 × measured` side is the
complement: an estimator wasteful by more than 15% denies work that would have fit, converting a
memory gate into a throughput regression.

**Instrumentation dependency:** R3's Resources telemetry row (§5).

### 6.6 G6 — Routing hygiene (decision gate, not pass/fail)

Direct-route keys with a scan below them in the probe subtree `≤ 20%` of all direct-route keys.

Above 20%, the unified-routing follow-up is filed and triaged **before** the default-on decision:
the delivery plan already names this exact number as that follow-up's trigger condition
("route-class telemetry shows direct-route keys with a scan below them"). Exceeding it does not
block a ship, but shipping without recording it means the trigger was never checked.

**Instrumentation dependency:** R2's route-class telemetry (§5).

### 6.7 G7 — Coverage-gate liveness and cost (R1b-local; evaluable now)

- **Live:** `keys_with_known_domain > 0` on at least **5** TPC-H queries on the **DuckDB-native
  corpus**. The evidence source answers only for native table scans (R1b design §D.3), so the
  Parquet corpus is out of the gate's declared scope and is recorded for G7 as **not evaluated** —
  which is not a pass. Below the bar, the domain trace resolves too rarely to be worth its
  complexity and R1b is reverted independently of the rest of #1010.
- **Bounded:** the per-query deterministic counters (the policy-decision family of
  `dynamic_filter_stats`, R1b design §T0) are recorded per run. Delivery figures
  (`filters_pushed`, attempts) are recorded alongside as observability only — they race with
  probe-side draining and are not reproducible run-to-run.
- **Non-regressive:** no query with a non-zero `keys_skipped_domain_gate` may regress beyond G2.
- **Reversible:** at `dynamic_filter_domain_coverage_threshold = 2.0`,
  `keys_skipped_domain_gate == 0` on every query — §B.3's explicit disabled state, exercised at
  benchmark scale (the T4 regression asserts the same contract in CI).

*Why 5 queries.* The gate exists to avoid filter *construction* cost on filters an existing runtime
backstop would disable anyway (see the R1b design §B.2). A construction-cost optimization that fires
on fewer than 5 of 22 queries does not repay a plan-tree walk.

## 7. Rollback

**Trigger — any one of the following, evaluated from the recorded numbers, not from judgement:**

1. Any G1 failure, in any configuration. *(Immediate; no discussion.)*
2. Any query regressing beyond G2 that is not attributed to a fixed and re-measured cause within one
   iteration.
3. Measured peak device memory exceeding the admitted bound in any configuration (G5).
4. Any `publications_failed > 0`, or any `publications_skipped_source_not_resident > 0`, on a
   supported topology during the gate runs.
5. Corpus geomean `median_OFF / median_ON < 1.00` (G3 floor).

**Ladder, in order:**

1. `SET enable_dynamic_filter_sip = false` — restores the exact Phase-1 topology.
2. For a coverage-gate-specific regression only: `SET dynamic_filter_domain_coverage_threshold = 2.0`
   — restores pre-R1b publication behavior without touching the endpoint. *(No new flag exists for
   this; the threshold is the lever, per R4's constraint that the rollout decision introduces no new
   policy knobs.)*
3. Revert R3, then R2, then R1.

Rollback must be **exercised** before R4 records its decision, per the delivery plan's R4 criterion.

## 8. Apply-cascade profiling and the alternative-kernel authorization

Per the delivery plan: **the existing apply cascade is benchmarked first.** R2/R3 profiling records,
per applied batch, the split of dynamic-filter-attributable GPU time into mask / gather / other.

**A prototype of an alternative apply kernel is authorized only when both hold, on at least 3 corpus
queries:**

1. `mask + gather ≥ 40%` of dynamic-filter-attributable GPU time — it dominates the feature's own
   budget; **and**
2. dynamic-filter-attributable GPU time `≥ 5%` of query wall time — the feature's cost is material at
   all.

*Why these two together, and why 5%.* The second condition is calibrated directly against G3: if
eliminating the feature's entire GPU cost cannot move a query past the 5% minimum-benefit bar, then
optimizing a fraction of that cost cannot either, and prototyping a kernel is provably wasted work.
The first condition ensures the prototype targets the dominant term rather than the convenient one.

## 9. Outcome definitions and the predeclared planner-class partition

**The class partition is closed and declared here, before any measurement.** A planner class is one
cell of the cross product:

| Dimension | Values |
|---|---|
| Join type | `INNER`, left-`SEMI` |
| Key arity | single, composite |
| Route class | scan endpoint, direct endpoint |
| Scan format | Parquet, DuckDB-native |
| Topology | 1 GPU, 2 GPU, 4 GPU |

No class outside this product may appear in an allowlist; no dimension may be added, removed, or
re-bucketed after any governed measurement has been examined (§10 applies).

**Minimum sample.** A class is evaluated only when at least **3** corpus queries fall in it. A class
with fewer is recorded as **not evaluated** — which, as everywhere in this document, is not a pass
and cannot enter an allowlist.

**Evaluation unit.** Gates are evaluated per `(query, scan format, topology)` cell, never pooled
across formats or topologies. A class passes a gate only when every one of its evaluated cells
passes.

**The deterministic allowlist rule.** The limited-ship allowlist is *exactly* the set of predeclared
classes for which **all applicable gates hold on every evaluated topology**. No class may be added
by judgement, removed by judgement, or split after results are visible. The rule is a function of
the recorded numbers; two people applying it to the same run record must produce the same allowlist.

| Outcome | Condition |
|---|---|
| **Default-on** | G1 ∧ G2 ∧ G3(default-on) ∧ G4 ∧ G5 ∧ G7, on every **evaluated** topology; G6 recorded |
| **Limited ship** | G1 ∧ G5 ∧ G7, and G2 ∧ G3(limited) ∧ G4 hold on a *subset* of the predeclared classes above; the allowlist is the deterministic rule's output, **with no new policy knobs** |
| **No-ship** | any other result; the flag stays off and the workload or topology classes that missed each gate are recorded for the next investigation |

A limited ship narrows the *planner allowlist*. It does not introduce a threshold, a heuristic, or a
new setting.

## 10. Change control

This document may be amended only:

- **before** any measurement it governs has been examined; or
- to **tighten** a threshold; or
- to record a §11 calibration, which may only lower the §6.2 floor; or
- to add a gate, which may not weaken an existing one.

Any other amendment invalidates pre-registration, and the affected gates must be re-measured from a
clean run on an unexamined corpus.

## 11. Calibration record

*(Filled before any ON data is examined; empty at check-in.)*

| Quantity | Procedure | Value |
|---|---|---|
| §6.2 absolute floor | 95th percentile of `\|median_A − median_B\|` over 10 OFF-vs-OFF pairs, warm, same session | _pending_ (pre-registered ceiling: 20 ms) |
| Baseline unstable-query set | Queries failing §4.1's IQR rule under OFF-vs-OFF; the only set §4.1's exclusion and >3 invalidation apply to | _pending_ |
| Evidence coverage | On the DuckDB-native corpus: count of TPC-H queries with `keys_with_known_domain > 0`, recorded before G7 is read | _pending_ |
| GB10 `sirius.yaml` | Verbatim | _pending_ |
| Resolved CUDA / libcudf / driver / DuckDB SHA / Sirius SHA | `pixi list`, `git rev-parse` | _pending_ |
