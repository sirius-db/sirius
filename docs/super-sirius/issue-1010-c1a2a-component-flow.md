# C1a-2a component flow guide

This guide explains the production components introduced or changed by the C1a-2a PR, how data
moves through them, and which component owns each piece of state. It is intended to let an agent
with no prior conversation context follow the implementation from the optimized DuckDB logical
plan to the Sirius physical hash join.

The most important boundary is:

> C1a-2a creates a validated, read-only planning sidecar. It does not replace the legacy dynamic
> filter publication plan, and no runtime code consumes the sidecar in this PR.

The guide deliberately excludes tests. It also distinguishes C1a-2a behavior from later issue-1010
work such as freezing the canonical plan, lifecycle state, compact target telemetry, and SIP
consumers.

## The 30-second model

For each optimized query plan, <code>sirius::planner::sirius_physical_plan_generator</code>:

1. Records the identity of every logical comparison or delim join before DuckDB resolves column
   bindings.
2. Runs the central whole-tree DuckDB type-resolution pass and exactly one whole-tree
   column-binding pass before recursion.
3. Extracts an immutable Sirius-owned candidate for each recorded join.
4. Recursively plans the children. Scan planning creates the existing Sirius runtime channel
   objects.
5. At each comparison join, combines the cached DuckDB candidate with those already-created scan
   channels and with static key evidence to construct a
   <code>sirius::op::dynamic_filter_publish_plan_builder</code>.
6. Moves that builder into <code>sirius::op::sirius_physical_hash_join</code>, whose constructor
   resolves the keys against the actual physical build-input schema.
7. Exposes only a channel-free <code>sirius::op::dynamic_filter_planning_view</code> to later
   planning readers.

In parallel, the pre-existing legacy branch still constructs
<code>sirius::op::dynamic_filter_publish_plan</code> and registers producers. It remains the only
publication-plan model consumed by <code>sirius::op::dynamic_filter_publisher</code> at runtime;
the publisher also borrows its pre-existing DuckDB and physical-key metadata.

~~~mermaid
flowchart LR
    O[Optimized DuckDB logical tree]
    C[Two-pass candidate cache]
    S[Recursive scan planning]
    J[Comparison-join planning]
    H[Physical hash join]
    V[Read-only planning view]
    L[Legacy runtime plan]
    P[Runtime publisher]

    O --> C
    C --> J
    S -->|existing scan-created channels| J
    J -->|sidecar builder| H
    H -->|resolved channel-free values| V
    J -->|legacy plan and sole producer registration| L
    L --> P
~~~

## Vocabulary

### Plan and join terms

**Logical plan**<br>
A DuckDB tree whose operators describe relational work before Sirius creates executable physical
operators. The relevant join class is <code>duckdb::LogicalComparisonJoin</code>, declared in
[logical_comparison_join.hpp](duckdb/src/include/duckdb/planner/operator/logical_comparison_join.hpp).

**Physical plan**<br>
The executable Sirius operator tree produced recursively from the logical plan. The relevant
physical join is <code>sirius::op::sirius_physical_hash_join</code>, declared in
[sirius_physical_hash_join.hpp](src/include/op/sirius_physical_hash_join.hpp) and implemented in
[sirius_physical_hash_join.cpp](src/op/sirius_physical_hash_join.cpp).

**Probe side / left child**<br>
The rows tested against a built hash table. In this code path it is the left child of the hash
join. A dynamic filter is applied to rows flowing on this side.

**Build side / right child**<br>
The input used to build the hash table. In this code path it is the right child. Admitted dynamic
filter keys are resolved to column positions in this child's runtime batch schema.

**Producer**<br>
A comparison join that may construct a filter from a build-side key and publish it to one or more
consumers. C1a-2a records a planning model for eligible INNER and SEMI joins. The legacy path still
performs the actual runtime producer registration.

**Scan consumer**<br>
A base-table <code>duckdb::LogicalGet</code>, declared in
[logical_get.hpp](duckdb/src/include/duckdb/planner/operator/logical_get.hpp), on the producer's
probe subtree that DuckDB selected to receive a filter. Scan consumers are the only target kind
modeled in C1a-2a.

**SIP consumer**<br>
A future filtering checkpoint inside an intermediate join rather than at a base scan. SIP means
sideways information passing. No SIP endpoint or SIP runtime consumption is introduced by
C1a-2a; <code>sirius::op::probe_schema_ordinal</code> is only vocabulary reserved for that later
work.

**Delim join**<br>
A DuckDB comparison-join form used while planning decorrelated subqueries. It follows the same
Sirius comparison-join planning path, so the candidate cache captures
<code>LOGICAL_DELIM_JOIN</code> nodes as well as ordinary comparison joins.

**Binding resolution**<br>
DuckDB's in-place conversion from logical table-and-column references to positions in physical
operator inputs. The relevant class is <code>duckdb::ColumnBindingResolver</code>, declared in
[column_binding_resolver.hpp](duckdb/src/include/duckdb/execution/column_binding_resolver.hpp).

**Key materialization**<br>
Sirius's rewrite of a complex equality expression into a computed temporary column below the join,
followed by a synthetic bound reference to that appended column.

**BUILD_PROBE**<br>
A hash-join execution mode selected later at runtime when the right/build input can be folded to a
single suitable batch. The right side is built once and left/probe batches are streamed through
it. The sidecar requires eligibility for this claim; it neither selects nor proves the eventual
runtime mode.

**MIXED_JOIN**<br>
A hash join containing at least one equality condition and at least one inequality condition.
Such a join cannot satisfy the C1a-2a BUILD_PROBE publication claim.

### Candidate, key, and target terms

**Dynamic filter**<br>
An immutable membership summary built from one build-side key and applied to the corresponding
probe-side values to reject rows that cannot match. The runtime value class is
<code>sirius::op::sirius_dynamic_filter</code>, declared in
[sirius_dynamic_filter.hpp](src/include/op/sirius_dynamic_filter.hpp). C1a-2a plans and validates
the future route; it does not construct this runtime value.

**Publication**<br>
The runtime act of constructing a filter snapshot from the completed build input, creating the
configured replicas, and offering that same snapshot to every accepting target channel.

**Route**<br>
The mapping from one producer, through a channel, to one target endpoint, with an ordinal-aligned
probe column for every recorded key.

**Arity**<br>
The number of recorded keys in the route. Full target arity means every target has exactly one
probe column for every DuckDB filter ordinal.

**DuckDB filter metadata**<br>
<code>duckdb::JoinFilterPushdownInfo</code>, declared in
[join_filter_pushdown.hpp](duckdb/src/include/duckdb/execution/operator/join/join_filter_pushdown.hpp).
It records which join conditions DuckDB selected and where those keys appear at each target scan.

**Candidate**<br>
An immutable Sirius-owned structural snapshot of one logical join's DuckDB filter metadata:
<code>sirius::planner::duckdb_join_filter_candidate</code>. Candidate means "available for Sirius
validation," not "guaranteed to become a runtime filter." Its structural classification can be
<code>absent</code>, <code>statistics_only</code>, <code>admitted</code>, or
<code>malformed</code>.

**Key candidate**<br>
One DuckDB-recorded key plus the static evidence Sirius needs to make a fail-closed decision. It is
represented by <code>sirius::op::dynamic_filter_key_candidate</code>. The vector exists because
DuckDB can record several independent key filters for one join. C1a-2a does not construct one
tuple-valued composite membership filter; each admitted ordinal describes an independent scalar
filter.

**Key decision**<br>
The physical hash join's final verdict for one key candidate:
<code>admitted</code>, <code>non_equality</code>, <code>not_direct_uncast</code>,
<code>unsupported_membership_type</code>, or <code>unresolved</code>. The enum is
<code>sirius::op::dynamic_filter_key_decision</code>. The first three rejection values preserve
specific static evidence failures; several later physical mismatches deliberately collapse to
<code>unresolved</code>.

**Key plan**<br>
The compact, physically resolved description of one admitted key:
<code>sirius::op::dynamic_filter_key_plan</code>. It includes the Sirius key ordinal and the
column index and type in the physical build-side input batch.

**Target**<br>
One producer-to-consumer endpoint edge. In C1a-2a a target is one selected scan. A producer can
have multiple targets when its probe subtree reaches multiple scans. Separate UNION branches are
an example the DuckDB metadata model can represent, although Sirius set-operation planning is not
currently enabled. Each target carries the full DuckDB key arity so ordinal <code>j</code> has a probe
column at every target; partial per-target key tuples are not accepted.

**Compatible target**<br>
A structurally valid target whose column types match every statically admissible key. The temporary
<code>compatible_target_indexes</code> vector contains indexes into
<code>duckdb_join_filter_candidate::targets()</code>; its elements are not target IDs. A bad target
is removed independently so a valid sibling target can survive.

**Live target**<br>
A compatible target for which recursive scan planning has already created a matching Sirius
runtime channel object. "Live" here proves that the endpoint object exists. It does not mean that a
producer is registered, a publication is open, or a filter has been built or delivered.

**Canonical**<br>
The representation intended to become the source of truth at the later C1a-2b freeze. In C1a-2a,
"canonical target" or "canonical sidecar" does not mean current runtime authority or frozen
immutability across execution.

### The three meanings of channel

The implementation correlates three related but non-interchangeable things:

1. **DuckDB channel identity** — the preserved
   <code>duckdb::DynamicTableFilterSet const*</code> object identity. The class is declared in
   [table_filter.hpp](duckdb/src/include/duckdb/planner/table_filter.hpp). Sirius holds it through
   a read-only DuckDB shared pointer and uses its address only as an opaque planning correlation
   key.
2. **Sirius channel object** — <code>sirius::op::sirius_dynamic_filter_set</code>, declared in
   [sirius_dynamic_filter.hpp](src/include/op/sirius_dynamic_filter.hpp) and implemented in
   [sirius_dynamic_filter.cpp](src/op/sirius_dynamic_filter.cpp). This is the shared runtime
   producer/consumer object.
3. **Strong channel ID** — <code>sirius::op::dynamic_filter_channel_id</code>, a typed,
   nonzero planning identity minted by one generator-owned allocator.

The DuckDB pointer correlates discovery; the Sirius object owns runtime channel behavior; the
strong ID gives later planning and telemetry a stable category-safe name. Equal-looking pointer,
object, and ID values must never be treated as interchangeable.

### Policy and state terms

**Build-subtree filter hint**<br>
An observation copied from DuckDB saying that the build side is already filtered or narrowed.
The canonical sidecar records it but does not use it for route admission. The legacy path treats a
false value as an unfiltered build domain and uses that as a production gate, so hint-false plans
can have an enabled planning view while the legacy runtime path publishes nothing.

**Replica space**<br>
A planning description of a GPU or host placement that could hold a filter replica, represented by
<code>sirius::op::dynamic_filter_replica_space</code> in
[dynamic_filter_replica_space.hpp](src/include/op/dynamic_filter_replica_space.hpp). The builder
validates and retains these descriptions for the future freeze. It does not allocate, place, or
publish a filter in C1a-2a, and an empty replica-space list is valid.

**Publication claim**<br>
The execution-mode requirement under which the v1 producer is allowed to publish. The sidecar
currently declares only <code>dynamic_filter_publication_claim::build_probe</code>. A required
claim is a requirement, not proof that the current execution has acquired or satisfied it.

**Sidecar**<br>
Planning metadata owned next to the physical join but not read by runtime publication code.

**Planning view**<br>
The sanctioned read-only, channel-free value surface returned after successful key resolution.
Its spans borrow storage owned by the builder; copying the view does not extend the builder's
lifetime.

**Enabled planning view**<br>
A view with at least one live scan target and at least one admitted key. It does not assert that
runtime mode, replica availability, producer registration, lifecycle state, or publication is
ready.

## Component and file map

| Component | Defined in | Implemented in | Responsibility in C1a-2a |
|---|---|---|---|
| <code>sirius::op::dynamic_filter_entity_id</code>, <code>dynamic_filter_ordinal</code>, aliases, and <code>dynamic_filter_identity_allocator</code> | [dynamic_filter_identity.hpp](src/include/op/dynamic_filter_identity.hpp) | Header-only | Strong identity and index-space vocabulary; one planning-time minting authority |
| <code>sirius::planner::duckdb_probe_target_candidate</code> and <code>duckdb_join_filter_candidate</code> | [duckdb_join_filter_candidate_adapter.hpp](src/include/planner/duckdb_join_filter_candidate_adapter.hpp) | [duckdb_join_filter_candidate_adapter.cpp](src/planner/duckdb_join_filter_candidate_adapter.cpp) | Sirius-owned snapshot values for DuckDB metadata |
| <code>sirius::planner::duckdb_join_filter_candidate_adapter</code> namespace | [duckdb_join_filter_candidate_adapter.hpp](src/include/planner/duckdb_join_filter_candidate_adapter.hpp) | [duckdb_join_filter_candidate_adapter.cpp](src/planner/duckdb_join_filter_candidate_adapter.cpp) | Sole canonical adapter for structural reads of DuckDB filter metadata |
| <code>sirius::planner::dynamic_filter_candidate_cache</code> | [dynamic_filter_candidate_cache.hpp](src/include/planner/dynamic_filter_candidate_cache.hpp) | [dynamic_filter_candidate_cache.cpp](src/planner/dynamic_filter_candidate_cache.cpp) | Two-pass, exact-node, immutable candidate cache for one logical tree |
| <code>sirius::planner::sirius_physical_plan_generator</code> | [sirius_physical_plan_generator.hpp](src/include/planner/sirius_physical_plan_generator.hpp) | [sirius_physical_plan_generator.cpp](src/planner/sirius_physical_plan_generator.cpp) | Owns the planning transaction, cache, channel maps, allocator, ID memos, and recursion |
| <code>sirius_physical_plan_generator::plan_comparison_join</code> | [sirius_physical_plan_generator.hpp](src/include/planner/sirius_physical_plan_generator.hpp) | [sirius_plan_comparison_join.cpp](src/planner/sirius_plan_comparison_join.cpp) | Captures key evidence, finds live targets, builds the sidecar descriptor, and preserves the legacy path |
| <code>sirius_physical_plan_generator::create_plan(duckdb::LogicalGet&amp;)</code> | [sirius_physical_plan_generator.hpp](src/include/planner/sirius_physical_plan_generator.hpp) | [sirius_plan_get.cpp](src/planner/sirius_plan_get.cpp) | Creates or reuses the Sirius scan-consumer channel object |
| <code>sirius::op::dynamic_filter_key_candidate</code>, <code>dynamic_filter_key_plan</code>, planning value types, and <code>dynamic_filter_publish_plan_builder</code> | [dynamic_filter_publish_plan_builder.hpp](src/include/op/dynamic_filter_publish_plan_builder.hpp) | [dynamic_filter_publish_plan.cpp](src/op/dynamic_filter_publish_plan.cpp) | Validates and owns the canonical planning sidecar |
| <code>sirius::op::sirius_physical_hash_join</code> | [sirius_physical_hash_join.hpp](src/include/op/sirius_physical_hash_join.hpp) | [sirius_physical_hash_join.cpp](src/op/sirius_physical_hash_join.cpp) | Owns both models and resolves sidecar keys once against the physical build schema |
| <code>sirius::op::dynamic_filter_publish_plan</code> | [dynamic_filter_publish_plan.hpp](src/include/op/dynamic_filter_publish_plan.hpp) | [dynamic_filter_publish_plan.cpp](src/op/dynamic_filter_publish_plan.cpp) | Pre-existing legacy runtime publication plan; still production authority |
| <code>sirius::op::dynamic_filter_publisher</code> | [dynamic_filter_publisher.hpp](src/include/op/dynamic_filter_publisher.hpp) | [dynamic_filter_publisher.cpp](src/op/dynamic_filter_publisher.cpp) | Runtime filter construction and publication; consumes the legacy publication-plan model plus existing DuckDB and physical-key metadata, never the sidecar |
| <code>sirius::op::sirius_dynamic_filter_set</code> | [sirius_dynamic_filter.hpp](src/include/op/sirius_dynamic_filter.hpp) | [sirius_dynamic_filter.cpp](src/op/sirius_dynamic_filter.cpp) | Existing shared runtime channel object |

The new cache translation unit is included in the build by
[CMakeLists.txt](CMakeLists.txt). The production entrypoints in
[sirius_extension.cpp](src/sirius_extension.cpp) and [sirius_ffi.cpp](src/sirius_ffi.cpp) now hand
the unresolved optimized tree to the generator; resolution belongs to
<code>sirius_physical_plan_generator::create_plan</code>.

The transparent path's preservation call lives in
[sirius_optimizer_extension.cpp](src/transparent/sirius_optimizer_extension.cpp). That preservation
is a C1a-1 prerequisite: it ensures a copied logical tree retains the exact DuckDB route identity
that C1a-2a later correlates.

## Class and ownership model

~~~mermaid
classDiagram
    class sirius_physical_plan_generator {
        -candidate_cache
        -dynamic_filter_channels
        -dynamic_filter_id_allocator
        -dynamic_filter_publication_ids
        -dynamic_filter_channel_ids
        +create_plan(logical_tree)
        +plan_comparison_join(join)
        +create_plan(LogicalGet)
    }

    class dynamic_filter_candidate_cache {
        -_captured_joins
        -_entries
        +capture_pre_resolver(root)
        +extract_post_resolver(root)
        +candidate_for(join)
    }

    class duckdb_join_filter_candidate {
        +kind()
        +build_subtree_has_filter_hint()
        +condition_indexes()
        +condition_comparisons()
        +targets()
    }

    class duckdb_probe_target_candidate {
        +channel_identity()
        +columns()
    }

    class dynamic_filter_identity_allocator {
        +mint_publication_plan_id()
        +mint_target_id()
        +mint_channel_id()
    }

    class dynamic_filter_publish_plan_builder {
        -_scan_targets
        -_scan_target_channels
        -_key_candidates
        -_ordinal_records
        +resolve_keys(decisions, plans, build_width)
        +planning_view()
    }

    class sirius_physical_hash_join {
        -_dynamic_filter_plan
        -_dynamic_filter_builder
        +has_dynamic_filter_planning_view()
        +planning_view()
    }

    class dynamic_filter_publish_plan {
        -_probe_targets
        -_replica_spaces
        +enabled()
    }

    class sirius_dynamic_filter_set {
        +register_producer()
    }

    sirius_physical_plan_generator *-- dynamic_filter_candidate_cache
    sirius_physical_plan_generator *-- dynamic_filter_identity_allocator
    dynamic_filter_candidate_cache *-- duckdb_join_filter_candidate
    duckdb_join_filter_candidate *-- duckdb_probe_target_candidate
    sirius_physical_plan_generator ..> dynamic_filter_publish_plan_builder : constructs
    sirius_physical_plan_generator --> sirius_dynamic_filter_set : creates or reuses
    sirius_physical_hash_join *-- dynamic_filter_publish_plan_builder : unique ownership
    sirius_physical_hash_join *-- dynamic_filter_publish_plan : value ownership
    dynamic_filter_publish_plan_builder --> sirius_dynamic_filter_set : private shared ownership
    dynamic_filter_publish_plan --> sirius_dynamic_filter_set : shared target
~~~

The diagram uses unqualified class names for Mermaid compatibility. The component table above gives
the complete namespace and source location for every class. Member and method names are the actual
implementation names; container and parameter types are abbreviated to keep the diagram readable.

## End-to-end planning sequence

~~~mermaid
sequenceDiagram
    participant E as Production entrypoint
    participant G as sirius_physical_plan_generator
    participant C as dynamic_filter_candidate_cache
    participant R as DuckDB type and binding resolution
    participant A as candidate adapter
    participant S as GET / scan planning
    participant J as comparison-join planning
    participant H as sirius_physical_hash_join

    E->>G: create_plan(owned optimized logical tree)
    G->>C: capture_pre_resolver(root)
    G->>R: root.ResolveOperatorTypes()
    G->>R: ColumnBindingResolver::Verify and VisitOperator once
    G->>C: extract_post_resolver(root)
    loop every captured comparison or delim join
        C->>A: extract(join)
        A-->>C: immutable Sirius candidate value
    end
    G->>J: recursively plan logical operators
    J->>J: snapshot static key evidence before key materialization
    J->>S: plan probe and build children
    S->>G: create or reuse scan channel by DuckDB identity
    J->>J: match compatible candidates to existing scan channels
    J->>J: mint or reuse publication, target, and channel IDs
    J->>H: move legacy plan plus unique sidecar builder
    H->>H: resolve each candidate against physical build columns
    H->>H: builder.resolve_keys exactly once
    H-->>G: physical join with legacy runtime plan and resolved sidecar
~~~

### Stage 1: bracket DuckDB's in-place resolution

<code>sirius_physical_plan_generator::create_plan</code> owns the logical tree and the ordering:

~~~mermaid
flowchart LR
    U[Unresolved optimized tree]
    P[Pre-resolver capture]
    R[Resolve types and column bindings]
    X[Post-resolver extraction]
    Q[Recursive physical planning]

    U --> P --> R --> X --> Q
~~~

Before resolution, a bound column expression carries logical table-and-column identity. After
<code>duckdb::ColumnBindingResolver</code>, it carries a position in a physical input. The cache
creates a stable seam around that mutation:

- <code>capture_pre_resolver</code> records the exact addresses of every
  <code>LOGICAL_COMPARISON_JOIN</code> and <code>LOGICAL_DELIM_JOIN</code>, including joins with no
  filter metadata. C1a-2a records node identity only; later work can add evidence that requires
  unresolved logical bindings.
- The generator runs the central whole-tree operator-type pass, calls
  <code>duckdb::ColumnBindingResolver::Verify</code>, and runs one whole-tree
  <code>VisitOperator</code> binding pass.
- During later recursive scan planning, the LogicalGet overload can rerun
  <code>ResolveOperatorTypes()</code> locally after it changes scan projections.
- <code>extract_post_resolver</code> walks the resolved tree, calls the adapter once per captured
  join, verifies exact pre/post node-set equality, and only then swaps in the complete temporary
  result map.

The cache is single-use and single-threaded. Its raw node-address keys are safe only because the
generator owns the same tree-node objects at stable addresses for this one <code>create_plan</code>
call. The entries themselves are immutable after successful extraction. Metadata absence is an
explicit <code>duckdb_candidate_kind::absent</code> value, not a null entry.

If extraction fails, no partial map is published. The cache remains captured but unextracted and
can retry against the correct tree.

### Stage 2: adapt DuckDB metadata into Sirius values

<code>duckdb_join_filter_candidate_adapter::extract</code> is the canonical structural boundary
for the sidecar. It reads <code>duckdb::JoinFilterPushdownInfo</code> and produces a
<code>duckdb_join_filter_candidate</code> containing:

- The structural <code>duckdb_candidate_kind</code>.
- The observational build-subtree hint.
- The join-condition index and comparison for each DuckDB filter ordinal.
- Zero or more <code>duckdb_probe_target_candidate</code> values.
- For each target, one probe column per DuckDB ordinal.
- A read-only shared handle preserving the exact
  <code>duckdb::DynamicTableFilterSet</code> object identity.

"Adapter admitted" means the recorded DuckDB structure is usable. It is not the final Sirius key
admission decision. Equality shape, direct uncast references, supported membership types, physical
build-column provenance, and target type agreement are checked later.

A target with corrupt arity is rejected independently; structurally valid sibling targets remain.
If no valid target survives, the candidate is malformed. Metadata with deliberately no targets is
<code>statistics_only</code>, regardless of the hint value.

### Stage 3: capture key evidence before Sirius rewrites expressions

At the start of <code>sirius_physical_plan_generator::plan_comparison_join</code>, the function
retrieves the cached candidate and calls its local key-snapshot logic before
<code>materialize_expression_join_keys</code>.

This ordering matters. A complex equality expression may later be computed into a temporary
column, after which the join sees a synthetic bound reference. Looking only after materialization
could make that temporary column look like an originally direct table column. The snapshot
preserves the original evidence needed to reject such a key.

For every DuckDB filter ordinal, <code>dynamic_filter_key_candidate</code> records:

- Its strong DuckDB filter ordinal.
- Its strong join-condition index.
- Whether the condition is exact equality.
- Whether both original key expressions are direct, uncast references.
- Whether both sides have the same supported membership type, currently INTEGER or BIGINT.

All evidence flags default to false, so missing evidence fails closed. Zero remains valid for the
ordinal fields; absence is never encoded as ordinal zero.

The same snapshot finds target candidates whose full arity is present and whose types match all
statically admissible keys. This is where a mixed-provenance composite route would be rejected as a
whole target: every target must account for every recorded ordinal. C1a-2a currently publishes
independent scalar filters rather than one composite tuple filter, so no partially routed
multi-column tuple is constructed.

### Stage 4: recursively plan scans and establish live channels

Comparison-join planning recursively plans its children before it creates canonical target drafts.
When <code>sirius_physical_plan_generator::create_plan(duckdb::LogicalGet&amp;)</code> reaches a
scan with DuckDB dynamic-filter metadata, it calls
<code>sirius_physical_plan_generator::get_or_create_dynamic_filter_channel</code> with the opaque
DuckDB filter-set address and attaches the returned
<code>std::shared_ptr&lt;sirius_dynamic_filter_set&gt;</code> to the physical scan.
The helper returns null when the global dynamic-filter-pushdown setting is disabled, preventing
both the scan channel and any sidecar target that would depend on it.

The generator's <code>dynamic_filter_channels</code> map therefore answers:

> Has the already-planned scan consumer created the Sirius runtime channel corresponding to this
> preserved DuckDB target identity?

After child planning, the canonical target loop performs a find-only lookup in that map. It does
not create a missing channel and does not register a producer. A successful lookup proves a live
scan endpoint and lets planning construct
<code>dynamic_filter_publish_plan_builder::scan_target_draft</code> with:

- A newly minted target ID for this producer-to-consumer edge.
- A channel ID minted once per DuckDB channel identity and reused from the generator's memo.
- Shared ownership of the existing Sirius channel object.
- The target's ordinal-aligned probe column positions.
- The target's ordinal-aligned probe storage types.

Canonical drafts are created only when the candidate is structurally admitted; the join is an
eligible producer and structurally eligible for the BUILD_PROBE claim; at least one key is
statically admissible; and a compatible scan-created channel exists.

### Stage 5: construct the sidecar descriptor

If at least one canonical target draft survived, comparison-join planning creates a
<code>dynamic_filter_publish_plan_builder::descriptor</code>. It supplies:

- One publication-plan ID, minted once for this logical producer and reused from a generator memo.
- The observational build-subtree hint.
- The complete join-condition count.
- The canonical target drafts.
- Zone-map and domain-coverage policy values.
- Replica-space descriptions.
- The pre-materialization key candidates.

Replica-space collection has two inputs but one meaning:

- When the build hint is true, the legacy branch has already inspected placements, so the sidecar
  receives a copy of that read-only result.
- When the hint is false, the legacy branch skips placement inspection. The sidecar takes the
  equivalent read-only snapshot directly so its canonical planning evidence does not depend on the
  legacy runtime gate.

The builder does not "place the filter on the build side." It retains the planner's possible
replica placements (one per GPU device) for the future freeze. This observation-only snapshot is not fed back into the
legacy plan and does not change registration, logging, allocation, or runtime behavior.

The builder constructor's preconditions are established upstream: planner construction supplies
identities, candidates, targets, and one replica per GPU device, and the configuration
boundaries (SET option and YAML load) validate the policy threshold. The preconditions are:

- Nonzero publication, target, and channel IDs.
- Positive policy threshold and nonzero join-condition count.
- Candidate ordinal alignment, condition bounds, and uniqueness.
- Non-null, distinct target channels and complete target arity.
- Unique target IDs.
- GPU/HOST replica tiers with one placement per GPU device.

It then splits every target draft:

~~~mermaid
flowchart LR
    D[scan_target_draft]
    T[dynamic_filter_planning_scan_target<br/>IDs, probe indexes, probe types]
    C[private shared_ptr to<br/>sirius_dynamic_filter_set]
    V[planning_view]
    F[future C1a-2b freeze]

    D --> T
    D --> C
    T --> V
    C -. not exposed .-> V
    C -. retained for .-> F
~~~

This split prevents planning-view readers from reaching the mutable runtime channel while keeping
the live object available for the later freeze.

### Stage 6: resolve keys against the physical build schema

The builder is moved behind a <code>std::unique_ptr</code> into
<code>sirius_physical_hash_join</code>. The stable, noncopyable, nonmovable builder owns all
storage backing future view spans.

Only the physical hash-join constructor has the evidence that static planning lacked:

- The join type and current join-mode classification, including whether it is mixed.
- The actual join comparisons.
- The resolved right-key column indexes and casts.
- The actual <code>children[1]-&gt;get_types()</code> build-input schema.
- The relationship between each join condition and its physical build-batch column.

For each key candidate, the constructor either creates a compact
<code>dynamic_filter_key_plan</code> or records a decision. Static evidence failures retain their
specific reason; later physical mismatches use <code>unresolved</code>. Admission requires:

1. An eligible INNER or SEMI producer.
2. Eligibility to claim BUILD_PROBE and a non-mixed join. Actual BUILD_PROBE selection occurs
   later in <code>sirius_physical_hash_join::update_join_exec_mode()</code>; sidecar admission does
   not prove that the runtime selects it.
3. Exact equality.
4. Direct, uncast key evidence.
5. In-range condition and right-key indexes.
6. A build-batch column index inside the right child's schema.
7. Supported INTEGER or BIGINT membership type.
8. Type agreement with the corresponding column at every scan target.

The compact <code>sirius_key_ordinal</code> counts admitted keys only. The original DuckDB ordinal
and join-condition index are retained so later readers can correlate the compact plan without
guessing across index spaces.

The constructor calls <code>dynamic_filter_publish_plan_builder::resolve_keys</code> exactly once
(the method is private; the hash join is a friend). resolve_keys constructs one
<code>dynamic_filter_planning_ordinal_view</code> per original DuckDB ordinal into local storage
and commits only after every allocation succeeds; its alignment preconditions are documented on
the declaration. After one successful resolution, any second call throws and leaves the resolved
state unchanged.

### Stage 7: expose the sanctioned planning view

After resolution, callers can use
<code>sirius_physical_hash_join::has_dynamic_filter_planning_view()</code> and
<code>sirius_physical_hash_join::planning_view()</code>. The returned
<code>dynamic_filter_planning_view</code> exposes:

- The strong publication-plan ID.
- The observational build-subtree hint.
- The derived <code>enabled</code> flag.
- A span of channel-free <code>dynamic_filter_planning_scan_target</code> values.
- A span of per-DuckDB-ordinal decisions and optional admitted key plans.
- The required BUILD_PROBE publication claim as a type-level constant.

The view does not expose the builder, mutable channels, replica-space storage, or policy storage.
Its spans remain valid only while the owning physical hash join and builder remain alive and
unchanged.

Calling <code>planning_view()</code> on a join without a builder, or before successful builder
resolution, is an internal contract violation.

## Identity and ordinal spaces

The strong types are declared in
[dynamic_filter_identity.hpp](src/include/op/dynamic_filter_identity.hpp). They prevent numbers
from different domains from being mixed accidentally.

### Entity IDs: zero is invalid

| Strong type | Entity named | Current minting point |
|---|---|---|
| <code>dynamic_filter_publication_plan_id</code> | One producing join and its admitted-key planning model | Generator, once per logical producer sidecar |
| <code>dynamic_filter_target_id</code> | One producer-to-consumer endpoint edge | Generator, once per canonical target draft |
| <code>dynamic_filter_channel_id</code> | One logical delivery channel | Generator, memoized by DuckDB channel identity |
| <code>dynamic_filter_id</code> | One immutable constructed filter | Vocabulary only in C1a-2a; execution-scoped minting comes later |

<code>dynamic_filter_entity_id</code> defaults to invalid zero and exposes
<code>is_valid()</code>. Only <code>dynamic_filter_identity_allocator</code> can construct a
nonzero value. The publication, target, and channel counters are independent and begin at one, so
equal numeric values across categories are normal but type-incompatible.

The allocator is owned once per <code>sirius_physical_plan_generator</code> planning transaction.
It is deliberately noncopyable, nonmovable, and not thread-safe.

### Ordinals and indexes: zero is valid

| Coordinate | Meaning |
|---|---|
| <code>duckdb_filter_ordinal</code> | Position <code>j</code> in DuckDB's aligned <code>join_condition[j]</code>, target <code>columns[j]</code>, key-candidate, and planning-view records |
| <code>join_condition_index</code> | Value stored at <code>join_condition[j]</code>; indexes the equality-first reordered Sirius join-condition vector |
| <code>sirius_key_ordinal</code> | Compact position among admitted keys only |
| Raw <code>probe_col_idx</code> / future <code>duckdb_column_ids_index</code> | Position in one target scan's DuckDB <code>column_ids</code> vector, not the base-table column number |
| Raw <code>build_batch_column_index</code> | Position in the physical right-child data batch entering the hash join |
| <code>probe_schema_ordinal</code> | Future C3 position in an intermediate SIP consumer's probe-side input batch; unused in C1a-2a |
| <code>compatible_target_indexes</code> | Ephemeral indexes into the cached candidate's target vector; not durable identities |

Ordinal zero means "first position," not "missing." Strong ordinal types intentionally have no
<code>is_valid()</code>; optionality must be represented explicitly.

## Worked example

Assume the optimized plan keeps <code>orders</code> on the probe/left side and
<code>customers</code> on the build/right side:

~~~text
orders.customer_id = customers.customer_id
~~~

Suppose DuckDB records one filter:

- DuckDB filter ordinal: 0.
- Join-condition index stored at ordinal 0: 0.
- The key is at position 2 in the orders scan's <code>column_ids</code> vector.
- The resolved customers build batch places <code>customer_id</code> at column 0.
- Both physical storage types map to cuDF INT32.

The data develops as follows:

~~~mermaid
flowchart LR
    D[DuckDB ordinal 0<br/>condition index 0]
    P[Scan target<br/>probe_col_idx 0 = 2<br/>probe type INT32]
    K[Key candidate 0<br/>equality<br/>direct and uncast<br/>supported type]
    B[Physical resolution<br/>build_batch_column_index = 0<br/>build type INT32]
    V[Planning view ordinal 0<br/>decision = admitted<br/>Sirius key ordinal = 0]

    D --> P
    D --> K
    K --> B
    P --> V
    B --> V
~~~

In value form:

~~~text
scan_targets[t].probe_col_idx[0] = 2
by_duckdb_ordinal[0].admitted_key->build_batch_column_index = 0
~~~

Probe position 2 and build position 0 describe different schemas. DuckDB ordinal 0 is the common
coordinate that associates them.

For a two-key join, this mapping repeats independently for ordinal 0 and ordinal 1. If a target
cannot supply the full two-ordinal route, its arity is incompatible and that entire target is
excluded. C1a-2a does not construct a tuple-membership filter spanning columns with mixed
provenance.

## Sidecar versus runtime authority

The physical hash join deliberately owns both models during this transition:

~~~mermaid
flowchart TB
    J[plan_comparison_join]

    subgraph H[One sirius_physical_hash_join]
        L[_dynamic_filter_plan<br/>legacy runtime plan]
        B[_dynamic_filter_builder<br/>C1a-2a sidecar]
    end

    J -->|legacy hint and placement gates<br/>register_producer| L
    J -->|lookup-only scan channels<br/>no producer registration| B
    L -->|only publication-plan model| P[dynamic_filter_publisher]
    P --> C[Runtime channel fan-out]
    B -->|planning access only| V[channel-free planning_view]
    B -. future C1a-2b freeze .-> F[Frozen canonical plan]
~~~

The rules for C1a-2a are:

- The legacy branch is the only code in comparison-join planning that calls
  <code>sirius_dynamic_filter_set::register_producer()</code>.
- <code>dynamic_filter_publisher</code> reads the legacy
  <code>dynamic_filter_publish_plan</code> together with existing DuckDB filter-pushdown and
  physical-key metadata; it never reads the sidecar.
- The canonical target loop performs lookup only. It does not create a channel, register a
  producer, emit a filter, or alter legacy logging.
- Direct reads of DuckDB filter metadata still exist in the intentionally preserved legacy branch.
- A sidecar can be present and its view can be enabled even when the legacy hint gate suppresses
  runtime publication.

This duplication is intentional scaffolding for the C1a-2b cutover, not two competing runtime
publication paths.

## Ownership and lifetime

~~~mermaid
flowchart LR
    D[DuckDB DynamicTableFilterSet object]

    subgraph T[One generator-owned planning transaction]
        G[sirius_physical_plan_generator]
        C[candidate cache<br/>keyed by logical-node address]
        M[channel map<br/>keyed by DuckDB filter-set address]
        I[one identity allocator<br/>plus ID memos]
        G -->|owns| C
        G -->|owns| M
        G -->|owns| I
    end

    D -->|read-only shared identity handle| C
    D -. address is correlation key .-> M
    M -->|shared ownership| S[sirius_dynamic_filter_set]
    Q[Physical scan] -->|shared ownership| S

    subgraph H[Physical hash join]
        L[legacy plan value]
        B[unique sidecar builder]
    end

    L -->|shared target channel| S
    B -->|private shared target channel| S
    B -->|owns backing vectors| V[planning view]
    V -. spans borrow builder storage .-> B
~~~

Key lifetime rules:

- The generator owns the logical tree, cache, channel map, allocator, and ID memos for one planning
  transaction.
- Candidate-cache node-address keys are valid only while that owned logical tree remains alive.
- A candidate keeps the exact DuckDB filter-set identity alive through a const shared handle.
- Physical scans, the legacy plan, and the builder can share ownership of the same Sirius channel
  object.
- The hash join owns the legacy plan by value and the builder through a unique pointer.
- The builder owns the vectors referenced by the planning view and privately retains live channel
  objects.
- A copied planning view is non-owning because its spans do not extend the builder's lifetime.

## Validation and state transitions

~~~mermaid
stateDiagram-v2
    [*] --> CacheFresh
    CacheFresh --> CacheCaptured: capture exact join-node set
    CacheCaptured --> CacheExtracted: exact post-resolver set and complete adapter success
    CacheCaptured --> CacheCaptured: extraction failure publishes nothing
    CacheExtracted --> [*]: recursive planning completes

    [*] --> BuilderConstructing
    BuilderConstructing --> BuilderUnresolved: descriptor taken over
    BuilderUnresolved --> BuilderResolved: resolve_keys commits
    BuilderUnresolved --> BuilderUnresolved: allocation failure commits nothing
    BuilderResolved --> BuilderResolved: second resolve_keys throws and leaves state unchanged
    BuilderResolved --> BuilderResolved: planning_view returns borrowed values
~~~

The cache and builder both use transaction-like publication:

- The cache builds a complete temporary candidate map, checks exact tree correlation, then swaps.
- The builder resolves into local vectors, then commits the resolved state.

This prevents later readers from observing partially constructed canonical planning data.

## Admission summary

A canonical planning sidecar exists only if the pipeline can prove all of the following at the
appropriate stage:

| Stage | Evidence required |
|---|---|
| Adapter | DuckDB metadata structure, condition indexes, target arity, and channel identity are coherent |
| Static join snapshot | Eligible producer shape, at least one equality key with direct uncast references and supported matching type |
| Post-child target discovery | Dynamic-filter pushdown is enabled and a compatible target maps to an already-created Sirius scan channel |
| Builder construction | IDs, ordinals, target/channel correlation, policy, and replica descriptions satisfy invariants |
| Physical hash-join construction | Each admitted key maps to an in-range build-batch column with supported type and agrees with every target |

Rejection is fail-closed. A rejected ordinal remains visible in
<code>by_duckdb_ordinal</code> with its decision, possibly <code>unresolved</code>; only admitted
ordinals receive compact
<code>dynamic_filter_key_plan</code> values.

## Deliberate C1a-2a limitations

The following are not implemented by this PR:

- Freezing the builder into the sole immutable publication plan or removing the legacy direct-plan
  branch. That is the C1a-2b cutover.
- Runtime publication lifecycle state, reasoned terminal outcomes, execution generation, reset
  rules, leases, or minting <code>dynamic_filter_id</code>. Those belong to later C1a-2c and
  C1a-2d units.
- Compact target telemetry or shadow policy. That belongs to C1b.
- SIP route discovery, <code>probe_schema_ordinal</code> use, intermediate-join consumers, or
  waiting. That belongs to C3 and later runtime work.
- Tuple-valued composite membership filters. Multiple admitted keys are independent scalar
  filters.
- Channel-owned stable IDs. Today the generator mints a channel ID only when a producer finds a
  scan-created channel. A scan with no producer or no surviving filter cannot obtain an ID from
  the channel object itself; persistent topology ownership is future work.
- A guarantee that <code>planning_view.enabled</code> implies runtime publication. It does not.

## Recommended production-code reading order

For a fresh context, read these files in this order:

1. [dynamic_filter_identity.hpp](src/include/op/dynamic_filter_identity.hpp) — identity and index
   vocabulary.
2. [duckdb_join_filter_candidate_adapter.hpp](src/include/planner/duckdb_join_filter_candidate_adapter.hpp)
   and [duckdb_join_filter_candidate_adapter.cpp](src/planner/duckdb_join_filter_candidate_adapter.cpp)
   — the DuckDB-to-Sirius structural boundary.
3. [dynamic_filter_candidate_cache.hpp](src/include/planner/dynamic_filter_candidate_cache.hpp)
   and [dynamic_filter_candidate_cache.cpp](src/planner/dynamic_filter_candidate_cache.cpp) — the
   two-pass seam around binding resolution.
4. [sirius_physical_plan_generator.hpp](src/include/planner/sirius_physical_plan_generator.hpp)
   and [sirius_physical_plan_generator.cpp](src/planner/sirius_physical_plan_generator.cpp) — the
   planning transaction and ownership.
5. [sirius_plan_get.cpp](src/planner/sirius_plan_get.cpp) — scan-consumer channel creation.
6. [sirius_plan_comparison_join.cpp](src/planner/sirius_plan_comparison_join.cpp) — static evidence,
   target discovery, descriptor assembly, and the parallel legacy branch.
7. [dynamic_filter_publish_plan_builder.hpp](src/include/op/dynamic_filter_publish_plan_builder.hpp)
   and [dynamic_filter_publish_plan.cpp](src/op/dynamic_filter_publish_plan.cpp) — builder
   invariants, resolution commit, and the read-only view.
8. [sirius_physical_hash_join.hpp](src/include/op/sirius_physical_hash_join.hpp) and
   [sirius_physical_hash_join.cpp](src/op/sirius_physical_hash_join.cpp) — physical key resolution
   and dual-model ownership.
9. [dynamic_filter_publisher.hpp](src/include/op/dynamic_filter_publisher.hpp) and
   [dynamic_filter_publisher.cpp](src/op/dynamic_filter_publisher.cpp) — confirmation that runtime
   still excludes the sidecar and consumes the legacy publication-plan model.

For the intended rollout boundary, read
[issue-1010-dynamic-filter-sip-design.md](docs/super-sirius/issue-1010-dynamic-filter-sip-design.md)
and
[issue-1010-github-delivery-plan.md](docs/super-sirius/issue-1010-github-delivery-plan.md).
