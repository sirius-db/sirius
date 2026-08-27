# Scalar subqueries (ASSERT_NUM_ROWS) — Tier 1 + Tier 2 plan

Affects (per TPCH-SURVEY / roadmap #3): Q2, Q11, Q15, Q17, Q20, Q22 — queries with scalar
subqueries (`x = (SELECT min(...))`, `> (SELECT avg(...) ...)`). The FE decorrelates them into
a join against the subquery's result; SQL semantics require the query to ERROR if a scalar
subquery yields more than one row, so the FE inserts an `ASSERT_NUM_ROWS_NODE` above the
subquery side. The CN refuses the node type today ("plan node is outside the v1 slice").

**Thrift shape**: `TAssertNumRowsNode { desired_num_rows: Option<i64>, subquery_string:
Option<String>, assertion: Option<TAssertion> }` (assertion ∈ EQ/NE/LT/LE/GT/GE; the scalar
contract is `LE 1` or `EQ 1`), one child.

**Evidence caveat (2026-08-06)**: the 390-fragment live capture contains ZERO
`assert_num_rows_node: Some` instances — the FE appears to elide the assert when the subquery
is provably scalar (ungrouped agg), at least for these 22 plans at both agg stages. So the
node may only appear in plan shapes we have not captured. **Before implementing either tier,
reproduce a real failing query and dump its fragment** (`SIRIUS_CN_DUMP_FRAGMENTS`) — the
blocker for Q2-class queries may be something else entirely (the wedged live sweep must be
re-run post-avg-fix with restart-on-failure to get clean per-query errors).

## Tier 1 — statically-proven no-op (S)

Translate `ASSERT_NUM_ROWS_NODE` as a checked pass-through when the ≤1-row contract holds
structurally; refuse loudly otherwise. Proof rules, strictest first:
1. Child is an UNGROUPED `AGGREGATION_NODE` (no grouping exprs, any phase that reaches here):
   emits exactly one row by construction ⇒ `LE 1`/`EQ 1` provably holds ⇒ drop the assert
   (translate to the child, `row_tuples`/`output_width` unchanged).
2. Child is a grouped aggregation: rows = #groups, NOT provably ≤1 ⇒ refuse
   ("assertion over a grouped aggregation needs the runtime check (Tier 2)").
3. Any other child, or assertion kind other than LE/EQ with desired_num_rows == 1 ⇒ refuse,
   naming the subquery_string in the message.
Implementation: one arm in `translate_plan_node` (node_translator.rs) + a
`translate_assert_num_rows` fn + translate.rs tests (ungrouped passthrough, grouped refusal,
non-agg-child refusal, GE-assertion refusal). No engine, FFI, or service change.
The WRONG fixes, for the record: unconditional passthrough (silently joins multiples) and
LIMIT 1 (silently truncates) — both turn a required error into wrong results.

## Tier 2 — runtime check (M), only if a non-provable shape appears

A real row-count enforcer for shapes Tier 1 cannot prove:
1. Engine: a tiny pass-through operator (`sirius_physical_assert_num_rows`): counts rows
   across all batches, throws past the bound at pipeline finalize (must count to END —
   per-batch checks miss multi-batch overflow). Mirror the streaming sink's finalize hook.
2. Plan carriage: the Substrait consumer is a submodule with no assert relation — reuse the
   out-of-band per-plan `ClientContextState` pattern proven by the byte-range registry
   (`substrait_scan_ranges.hpp`): the translator records "wrap relation X with assert(N)",
   the plan generator applies it. The hard sub-problem is IDENTIFYING X after lowering
   (node ids do not survive); candidate anchors: the stream view name (exchange-fed child) or
   the aggregate's position. Decide from the real dumped shape when one appears.
3. Failure semantics: the throw must fail the query loudly at run() (the existing failed-
   fragment path), and the error must carry `subquery_string` so the user sees WHICH subquery
   returned multiple rows.
Blocked by: a captured real plan (see caveat). Do not build speculatively.

## Status

- 2026-08-07: doc created. Tier 1 pending a real reproduced failure (see caveat); avg
  expansion implementation runs first — it may be the ONLY blocker for several of the six
  queries if the FE indeed elides the assert.
