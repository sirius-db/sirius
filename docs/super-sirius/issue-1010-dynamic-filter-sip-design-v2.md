# General Dynamic Filters (v2): Sideways Information Passing at Hash-Join Probe Inputs

**Targets:** [#1010](https://github.com/sirius-db/sirius/issues/1010),
[#1014](https://github.com/sirius-db/sirius/issues/1014) (complete). **Status:** design.
**Supersedes:** [the v1 SIP design](issue-1010-dynamic-filter-sip-design.md).
**Delivery:** [issue-1010-github-delivery-plan-v2.md](issue-1010-github-delivery-plan-v2.md).

## Context and goal

Phase-1 dynamic filtering (#794) lets an eligible <code>BUILD_PROBE</code> hash join publish a
membership filter to a GPU scan. It already provides the filter kinds, publication channel,
multi-GPU replicas, and adaptive cost gate described in
[dynamic-filters.md](dynamic-filters.md) and
[dynamic-filters-multi-gpu.md](dynamic-filters-multi-gpu.md).

Issue #1010 extends that capability to a probe key that cannot be filtered at a scan. The design
reuses the Phase-1 runtime and adds:

- one application endpoint at the producing join's probe boundary;
- Sirius-owned key metadata so publication does not depend on an optional DuckDB scan-pushdown hint.

In this document, **P** is the hash join whose completed build produces the filter, and an
**endpoint** is the operator that applies that filter to probe rows.

## Architectural summary

| Concern | Decision |
|---|---|
| Producer | An eligible <code>BUILD_PROBE</code> hash join P. Its build subtree may itself contain joins. |
| Endpoint selection | One application strategy per key: retain the existing scan endpoint(s) when a legal Phase-1 route exists; otherwise use one endpoint at P's immediate probe boundary. |
| Direct-endpoint placement | Before probe partitioning and concatenation. P already owns the key's local ordinal, so no lineage walk is required. |
| Safety | No false negatives; P remains authoritative. |
| Coordination | Waiter-free. Consumers apply a visible filter or pass rows through. |
| Publication | One-shot, immutable-before-visible, and backed by Sirius-owned admitted-key metadata. |
| Scope | Exclude only producer-to-endpoint paths whose CTE, DELIM, or shared/multi-parent topology prevents proving publication dominance. |

The mode of an inner join inside P's build subtree is irrelevant. What must be
<code>BUILD_PROBE</code> at publication time is P itself, because P publishes from its complete,
materialized build input.

## Safety and supported scope

The central invariant is:

> A dynamic filter must accept every probe row that could match P. False positives are allowed;
> false negatives are not. P's hash probe remains authoritative.

Stated generally, so later producer kinds slot in without re-litigating the architecture: a visible
filter must accept, at every observation instant, every row its authoritative producer could keep.
A producer can satisfy that in two ways. **Complete-at-once** producers — a finished hash-join
build, a completed aggregation's distinct set, post-sort min/max — publish once because their input
is complete. **Monotone-tightening** producers — a Top-N threshold — may publish repeatedly because
every published snapshot stays valid forever. Only a filter that could loosen would need retraction
or versioning, and no such producer is planned. This design implements one member of the first
class — the hash-join build; the substrate (append-only channel, per-batch consumer snapshots,
count-aware gate) already tolerates the second.

The initial producer/key scope is:

- P is INNER or left-SEMI;
- P is <code>BUILD_PROBE</code> when publication is claimed;
- the comparison is equality;
- build and probe keys are direct bound references with compatible INT32 or INT64 representations;
- P's build input is complete rather than partial, spilled, or restored;
- at least one live endpoint exists.

LEFT, FULL/OUTER, ANTI, MARK, and right-family joins are excluded because their preserved or negated
probe semantics make early row removal unsafe. Null-equal comparison, casts, computed keys, and
additional key types remain outside the initial scope.

Normal optimization misses—no route, no useful filter, empty build, policy skip, or an intentionally
unavailable target—pass through. Invalid key mapping, partial visibility, or a corrupt/wrong-device
representation must suppress the affected publication or fail loudly; they are not successful
no-ops.

## Routing and placement

For each eligible key, planning selects exactly one application strategy:

~~~mermaid
flowchart TD
    K[Admitted key] --> R{Legal Phase-1 scan route?}
    R -->|yes| S[Use the existing scan endpoint(s)]
    R -->|no| E[Use P's immediate probe edge]
~~~

The scan route remains the preferred site because it can apply the filter earlier and avoid
downstream work. Scan filtering is best-effort; a missed or unavailable filter affects only
optimization benefit. Adding a second edge endpoint for a scan-reachable key is not part of the
initial design and requires separate evidence that its benefit exceeds the redundant application
cost.

When no scan route exists, the direct endpoint is placed as follows:

~~~mermaid
flowchart LR
    B[Build subtree] -->|complete build| P[Producing hash join P]
    Q[Probe subtree] --> E["Dynamic-filter endpoint<br/>(uncovered keys)"]
    E --> X[Partition and concatenate] -->|probe input| P
    P -. "publish after complete build" .-> F[(Immutable filter replicas)]
    F -. "apply" .-> E
~~~

P's build publication attempt is ordered before its immediate probe producer runs. Any filter the
endpoint sees is complete; if none is published, it passes through without waiting. The endpoint is
placed before partitioning because that fits the current pipeline contract and avoids moving rows
that the filter will discard.

The endpoint uses P's local probe-key ordinal; it does not trace the key through the probe subtree.
Deeper placement and unified scan routing require lineage and are deliberately separate work.

## Publication, ownership, and execution

The planner records eligible build keys and their endpoint-local probe coordinates in immutable,
Sirius-owned metadata. DuckDB may continue to supply existing scan targets, but the direct route and
runtime publisher do not depend on optional DuckDB scan-pushdown metadata.

That boundary is the ownership end-state: DuckDB's join-filter metadata is a planning-time
discovery input for scan endpoints only, and every structure consumed at runtime is Sirius-owned.
The DuckDB coupling that remains is exactly three things — the scan-target discovery walk, the
<code>DynamicTableFilterSet*</code> channel route key, and the transparent-path
metadata-preservation shim that keeps both alive across plan copies. They retire as one unit in the
unified-routing follow-up; nothing in this design grows a new DuckDB dependency in the meantime.

After P's build completes, it constructs and finalizes every planned device representation, then
publishes the immutable filter or records a terminal no-op/failure. Consumer channel and gate state
is endpoint-local; P's publication state is producer-local. Both are fresh for each query execution.
Immutable filter objects and device replicas may be shared across endpoints without cloning.

The memory manager must outlive all publication work, endpoint state, and replicas. The delivery plan
owns the concrete freshness and teardown verification.

Multi-GPU behavior remains the Phase-1 model: build once, create a finalized representation on every
planned usable device, publish only after those representations are ready, and apply the
device-local representation at the endpoint.

## Policy and resource constraints

Existing Phase-1 policy remains in force:

- choose exact membership or Bloom representation using the L2-fit policy;
- skip domain-covering filters when coverage is known;
- let the endpoint-local adaptive gate bypass filtering that does not repay its application cost.

Non-scan endpoints apply membership filters only. They do not perform scan-specific row-group or
zone-map pruning.

The gate is a local cost heuristic, not coordination between endpoints. Resource admission must
cover all concurrently live application buffers and replica work. The delivery plan defines the
concrete estimate, failure tests, telemetry, and benchmark gates.

## Complexity budget

The design keeps the Phase-1 runtime and adds only the owned key metadata and direct endpoint
required by the admitted use case.

The initial implementation does not add:

- changes inside the hash join's probe execution path;
- a general lineage framework or scan-router replacement;
- new routing, identity, subscription, or lifecycle frameworks;
- ordered activation or filter-readiness waiting;
- STANDARD/partitioned producers or recurring filter generations;
- alternative GPU kernels without a measured bottleneck.

This is intentionally minimal but complete: publication visibility, key legality, endpoint
coordinates, lifetime, memory admission, and multi-GPU correctness remain load-bearing contracts.

## Acceptance and rollout

The feature remains off by default until result equivalence, safe scope enforcement,
multi-GPU/lifetime/memory correctness, and measured end-to-end value are established. The detailed
PR boundaries, tests, telemetry, thresholds, and rollback procedure belong to the
[delivery plan](issue-1010-github-delivery-plan-v2.md).

## Deferred work

- **Scan-reachable edge backstop:** consider only if measurements show that late scan publication
  leaves substantial selective work and the recovered benefit exceeds redundant application cost.
- **Unified Sirius scan routing:** replace the DuckDB scan-target dependency when route-class
  telemetry shows admitted keys taking the direct route while a scan sits below them in the probe
  subtree, or when preservation-shim bail-outs are observed in practice. When built, run discovery
  before DuckDB's <code>ColumnBindingResolver</code> on the logical plan — in
  <code>ColumnBinding</code> space, keyed by stable <code>LogicalGet</code> identity — rather than
  tracing the post-fold positional tree. Retiring the dependency removes the
  <code>DynamicTableFilterSet*</code> route key and the preservation shim with it.
- **Alternative endpoint placement:** deeper placement in the probe subtree would let intervening
  operators also skip doomed rows but requires tracing the key's coordinates through those
  operators; post-partition placement requires a partition-preserving operator-data contract.
  Introduce either only when measurements show material work or transfer savings.
- **STANDARD/partitioned producers:** requires a complete global or correctly routed partition
  predicate before visibility.
- **CTE, DELIM, and shared-DAG paths:** these create pipeline-level fan-out (one repository edge
  per CTE/DELIM scan consumer, delim cycles broken heuristically), which is exactly where the
  build-before-probe ordering guarantee is unproven. Supporting them requires an explicit
  publication-dominance design — a prove-it-later fence, not an impossibility.
- **Recurring producers (Top-N threshold):** deferred as an independent producer-axis feature, not
  for architectural reasons. The substrate already tolerates growing filter sets, and a
  monotonically tightening snapshot never becomes wrong. The missing pieces are publication cadence
  and re-execution reset on the producer side, plus a consumer-side supersession rule so endpoints
  apply only the tightest generation (today a superseded filter keeps running — per-filter keep
  ratios are recorded once, and zone maps bypass per-filter gating). Post-sort min/max is a
  one-shot producer of the existing shape and needs none of this.
- **Wider keys, casts, and strings:** expand only with representation and correctness support.
- **Late scan re-pruning or alternative apply kernels:** driven by measured I/O or application
  bottlenecks.

Ordered activation is not planned: the direct endpoint is already ordered, and all other misses are
safe pass-throughs. A Top-N-style producer confirms the boundary — its consumers sit in its own
input subtree, so producer and consumer are concurrent by construction and opportunistic
consumption is the terminal mode there.

## Prior art

DataFusion and Velox both apply join-derived dynamic filters at a safe scan consumer and retain the
join as the correctness authority. Their explicit routing and semantic gates shape this design.
Sirius differs by supporting a direct probe-boundary endpoint for keys without a scan route and by
owning multi-GPU replica publication.
