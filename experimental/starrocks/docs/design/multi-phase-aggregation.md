# Plan: multi-phase aggregation

## Why

- **StarRocks:** `new_planner_agg_stage = 1` means "single-stage where possible", not always. The FE still forces multi-stage plans for grouping sets, rollup, cube, multi-argument DISTINCT, and `group_concat` or `avg` with DISTINCT. Customer SQL will hit these.
- **TPC-H:** six queries fail today because only SUM can run in two phases: q01, q02, q13, q15, q17 and q22.
- **What StarRocks 4.1.3 plans:**
  - With default settings, three- and four-phase plans come only from `DISTINCT`.
  - Four cases can never be two-phase. Past those, a `DISTINCT` without a `LIMIT` generally isn't either.
  - In AUTO mode, a `DISTINCT` with no `GROUP BY` gets four phases.
  - `new_planner_agg_stage = 2` keeps `DISTINCT` at two phases outside the four cases.
- **TPC-H never needs three or four phases.** The recorded SF1000 plans are all one- or two-phase, and no node mixes update and merge functions. q16's `COUNT(DISTINCT)` is two dedupe aggregations with no functions, then a one-phase `count`; the classifier already accepts that.

## Today

Line numbers refer to `f4780e74`.

- **`agg_phase.rs`** classifies a node as one-phase, partial or merge, from `need_finalize` and the measures' `is_merge_agg`. Three- and four-phase plans ("merge serialize") are refused.
- **Two-phase plans** accept only SUM ("two-phase aggregation supports SUM only").
- **The non-finalizing phase's intermediate tuple** isn't handled: separate intermediate and output tuples are refused (`node_translator.rs:594`).
- **The POC branch** (`feat/pin-table-cn`, `d24f02c4`) already has two-phase SUM/COUNT/MIN/MAX/AVG in `partial_state.rs`.

## Steps

### Step 1: two-phase COUNT, MIN, MAX

Port the single-column rules from `partial_state.rs`. Both sides of the exchange derive the type sent between the partial and merge steps from the same rule, so they agree:

| function | sent between steps | merge |
|---|---|---|
| COUNT | BIGINT | SUM of the counts |
| MIN / MAX | the input type | the same function |

- Don't read the type from the slot. A node's slot type isn't always what it emits: a partial `avg` slot is `VARBINARY` in q01, q17 and q22.
- **Tests:** translator tests for each function's partial and merge fragments. Then expect q13 and q15 in the 22-query run, and q02 once `LIMIT` is fixed. q04 and q21 need COUNT too, once RIGHT SEMI join lands.

### Step 2: two-phase AVG

- AVG is sent as two columns: an FP64 sum and a BIGINT count. The merge step computes `SUM(sum) / SUM(count)`.
- The FE allocates one opaque slot for AVG's partial result, so the exchange row gains a column right after the sum.
- This needs the intermediate-tuple column order from [fe-contract-and-versions.md](fe-contract-and-versions.md) Step 3.
- **Tests:**
  - translator tests that the partial and merge fragments agree on the column list;
  - q01, q17 and q22 in the 22-query run.

### Step 3: three- and four-phase plans

Customer SQL only; TPC-H doesn't reach it.

- **Classify each function, not the node.** One node can mix update and merge functions, so each function's `is_merge_agg` decides how it's translated, and `need_finalize` only decides whether the node's output is final or a partial state. `agg_phase.rs` refuses mixed nodes today.
- **Try `new_planner_agg_stage = 2`** for CN sessions, once its side effects are confirmed. It limits three- and four-phase plans to the four cases that can't be two-phase, which matters most for `COUNT(DISTINCT x)` with no `GROUP BY`.
- Classify "merge serialize" nodes.
- Translate DISTINCT aggregation, which the FE plans as a local distinct step followed by a global aggregation.
- Then grouping sets, rollup and cube.
- Start from recorded FE plans for a `COUNT(DISTINCT)` with no `GROUP BY` and one `ROLLUP` query.
- **Tests:** translator fixtures for each plan shape.

## Open questions for StarRocks

- The exact four cases that can't be two-phase, and whether `new_planner_agg_stage = 2` has side effects.

## Out of scope

- `group_concat` and other aggregates without a GPU implementation.
