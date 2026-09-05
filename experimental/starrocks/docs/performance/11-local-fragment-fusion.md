**11 · Measure and expand local fragment fusion**

[All paths](README.md) · [Source review](../../../../starrocks-plan-improvement.md)

Status: proposed implementation plan. Baseline: `281b13bc`. Objective: remove avoidable local materialization and planning barriers while retaining FE semantics and good join choices. Prerequisite: [00 · Trustworthy measurements and benchmark coverage](00-measurement-and-benchmarks.md). Broad runtime concurrency is a separate path: [12 · Nonblocking query-scoped fragment execution](12-nonblocking-fragment-graph.md).

**Current behavior and code map**

Default `leaf` fuses eligible local hash leaves. `leaf-any` considers other partition labels but still requires a leaf, one local destination, and a registered exchange expecting one sender. Limits, sorting, output projections, carried common slots, partial aggregation, and other structural conditions can refuse fusion. Two independent leaves can already be folded into a join receiver.

| Source | Responsibility |
|---|---|
| [tunable.rs](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/src/tunable.rs#L287) | Existing `off`, `leaf`, and `leaf-any` modes. |
| [compute_node_service.rs](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/src/compute_node_service.rs#L981) | Policy, offering/defer logic, and fused execution. |
| [local_exchange.rs](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/src/local_exchange.rs#L282) | Receiver-first eligibility and plan ownership. |
| [fusion.rs](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/crates/starrocks-plan-translator/src/fusion.rs) | Pure structural checks and splice. |
| [agg_phase.rs](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/crates/starrocks-plan-translator/src/agg_phase.rs) and [partial_state.rs](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/crates/starrocks-plan-translator/src/partial_state.rs) | Aggregation/state constraints that fusion must preserve. |

**Selection policy**

First measure existing modes without new code. A fused edge eliminates a materialized input whose exact cardinality was available before receiver planning. For selective dimension branches that can change the join build side, compare actual physical plans and peak build memory, not only fragment count.

Log one stable reason for every offered/declined edge: remote/fan-out, not a leaf, no receiver, multiple senders, structural refusal, mode policy, or retired query. Distinguish policy eligibility from a successfully spliced and executed plan. Count both removed runs and avoided parked bytes.

**Implementation slices**

1. **Baseline and observability:** run `off/leaf/leaf-any` on saved equivalent FE plans. Add missing per-query eligibility and avoided-materialization summaries, retaining existing per-edge logs.
2. **Arrival-order coverage:** if `NoPendingReceiver` is material, add bounded deferred-plan registration or a predeclaration phase. Budget plan memory, avoid duplicate execution, and define a deadline/fallback before the sender begins GPU work. A late receiver must not resurrect a sender already executed or canceled.
3. **Intermediate local fusion:** consider chains with one destination only after inputs can transfer ownership safely. Preserve remaining stream IDs and query/fragment identity; plan splicing must not reset or lose nonfused input sources.
4. **One refusal class at a time:** lower and preserve explicit output projection/filter/limit semantics before allowing that class. Ordered exchange and partial aggregate state fusion require their own semantic implementation and cannot be enabled by deleting guards.

**Tests**

Extend translator fusion tests and compute-node service fixtures for sender-first/receiver-first arrival, duplicate dispatch, two leaves into one join, mixed deferred/parked/remote input, cancellation at every phase, and refusal after an earlier successful splice. Keep source cleanup correct if a fold fails.

For each newly admitted shape, compare values and plans against unfused execution: duplicate and NULL join keys, empty build/probe, selective small build versus large fact table, outer/semi/anti joins where supported, and partial/final aggregate distinctions. Unsupported shapes should remain explicit refusals.

Acceptance: one execution per original logical sender, no lost scan ranges or stream inputs, preserved result semantics, and bounded deferred-plan memory. A performance win must be associated with avoided materialization or planning overhead without a worse cardinality-sensitive join.

**Benchmark**

Start at one CN, where local edges are common, then repeat on mixed local/remote plans. Measure fused edge coverage, fragment build/run counts, parked bytes, scan/predicate work, join build bytes, runtime, and spill. A lower fragment count alone is not success. Retain `leaf` as the default unless `leaf-any` or an added policy wins across its targeted workload set.

**Rollout and alternatives**

Use existing modes for baseline; any new mode is proposed and must be added to the tunable registry and docs. Select policy before dispatch, with existing bounded fallback for refused shapes. Never undo a splice after execution begins.

If exact materialized statistics are essential to a workload, leave that edge unfused or use validated estimates/adaptive behavior in path 12. Fusion is a targeted local optimization and should not grow into an unreviewable substitute for query-scoped scheduling.
