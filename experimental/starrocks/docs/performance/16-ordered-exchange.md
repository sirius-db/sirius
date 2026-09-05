**16 · Ordered exchange merge and top-K**

[All paths](README.md) · [Source review](../../../../starrocks-plan-improvement.md)

Status: proposed implementation plan. Baseline: `281b13bc`. Objective: avoid re-sorting all received rows when the producer can provide a proven sorted-run contract, and reduce work for bounded ordered results. Prerequisite: [00 · Trustworthy measurements and benchmark coverage](00-measurement-and-benchmarks.md). Incremental remote consumption additionally uses paths 04/06/12, but a materialized-run merge can be evaluated first.

**Current behavior and code map**

An exchange carrying sort metadata becomes a `SortRel` over its stream read. Sender frame sequence proves delivery order, not that producer output is globally sorted across batches.

| Source | Responsibility |
|---|---|
| [node_translator.rs](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/crates/starrocks-plan-translator/src/node_translator.rs#L783) | Recognize an explicit ordered-exchange contract and preserve sort/offset/limit semantics. |
| [fusion.rs](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/crates/starrocks-plan-translator/src/fusion.rs) | Keep sorted-exchange refusal until equivalent semantics are implemented. |
| [sirius_physical_merge_sort.cpp](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/src/op/sirius_physical_merge_sort.cpp) and [sirius_physical_merge_sort.hpp](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/src/include/op/sirius_physical_merge_sort.hpp) | Assess existing merge machinery for reuse. |
| [sirius_plan_order.cpp](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/src/planner/sirius_plan_order.cpp) and [sirius_plan_top_n.cpp](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/src/planner/sirius_plan_top_n.cpp) | Preserve optimizer lowering and top-N opportunities. |
| [fragment_executor.rs](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/src/fragment_executor.rs) and [local_exchange.rs](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/src/local_exchange.rs) | Retain run/sender identity instead of flattening away ordering boundaries. |
| [test_physical_merge_sort.cpp](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/test/cpp/operator/test_physical_merge_sort.cpp) | Existing operator validation to extend. |

**Proposed contract**

A sorted-run descriptor identifies query/edge, sender, run ID, schema, sort key positions/expressions, direction, null ordering, collation, and run termination. Prove that the producing operator emits each run monotonically, including batch boundaries and concurrent sink scheduling. Merely observing sorted samples is not sufficient.

A k-way merge needs a readable head or known EOS from every active run before emitting the globally next row. Waiting for an unavailable run head is a semantic dependency, unlike accidental whole-fragment barriers. Keep runs distinct through local relay and remote ingress. Start with complete materialized runs under today's EOS gate to isolate sort-work reduction.

For top-K/offset, retain enough rows to satisfy offset plus count and any tie semantics actually requested. Push a bound into each producer only when the FE plan and comparator make that transformation valid. Early global completion may cancel upstream work, but lease and reader cleanup must still complete safely.

**Implementation slices**

1. **Plan/ordering audit:** capture actual FE/CN ordered plans and determine whether current DuckDB lowering already chooses top-N. Define a run contract and reject unsupported collation/type/order shapes.
2. **Materialized merge proof:** add a specialized ordered input/operator path over complete proven runs, reusing current merge APIs where possible. Keep full `SortRel` as the pre-execution fallback.
3. **Bounded top-K:** implement only semantically valid bound propagation and receiver termination. Include offset/ties/NULL policies and cancellation ownership.
4. **Incremental merge:** after the nonblocking graph exists, schedule when all needed heads are available and keep bounded per-run buffers. Track slow-run head wait separately from transport queue time.

**Tests**

Compare full-sort and merge outputs for ascending/descending multicolumn keys, NULL placement, strings/collation where supported, equal-key ties, empty runs, varying batch boundaries, nonzero offsets, K=0/K>N, and one delayed run. Supply a deliberately nonmonotonic run and ensure the contract cannot be silently accepted; use producer-side proof plus debug validation.

Test cancellation after top-K completion, sender failure before its first head, and EOS/replay. Results with non-total ORDER BY should be checked under the correct tie equivalence rather than an invented stable order.

Acceptance: full value/order equivalence, bounded buffering, no rows emitted without the required run-head information, and correct finalization/cleanup. Do not remove the existing fusion guard as a shortcut.

**Benchmark and rollout**

Sweep total rows, number of runs, key width, K/offset, and skew. Measure sort/merge GPU time, temporary bytes, receiver latency, head waits, and bytes avoided by valid top-K pushdown. Require measured reduced receiver work and a query benefit at equal memory budgets.

Enable only for explicitly proven producer/receiver contract combinations. Decide fallback before results are emitted; late discovery of unsorted input requires a loud failure or a designed buffered restart. If sender output order cannot be guaranteed cheaply, retain full sort.
