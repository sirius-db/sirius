# Dynamic Filters

The language of Sirius's dynamic filters: predicates built from a hash join's build side at runtime
and applied on its probe side, from planning through construction, publication, and application.
These terms are binding in code, comments, docs, specs, PRs, and issues; an _Avoid_ word is a review
defect.

## Language

### Filters

**Dynamic Filter**:
A predicate built at runtime from a hash join's build keys and applied on the probe side to drop
rows before the join.
_Avoid_: runtime filter, SIP, join filter

**Zone Map**:
A Dynamic Filter holding the minimum and maximum of a key's valid build values.
_Avoid_: bounds, global min/max filter

**Small IN-list**:
An IN-list of at most 12 valid build values.
_Avoid_: raw IN-list, needles

**Hash IN-list**:
An IN-list held as a hash set.
_Avoid_: set

**IN-list**:
A Dynamic Filter holding the exact set of a key's valid build values; either a Small IN-list or a
Hash IN-list.

**Bloom Filter**:
A Dynamic Filter holding an approximate membership set that may pass values absent from the build.

**Membership Filter**:
An IN-list or a Bloom Filter, as opposed to a Zone Map.

**Filter Strength**:
The ranking no filter < Bloom Filter < IN-list, used to compare a key's outcome across builds.

**Filtered Decode**:
Applying Membership Filters while a compressed column is being decoded.
_Avoid_: fused decode, decode attachment, decompression pushdown

**Domain Gate**:
The check that drops a Membership Filter when the build covers too much of the key's possible
values.
_Avoid_: pinned-domain usefulness, selectivity-aware Bloom skip, gate

**Usefulness Gate**:
A Consumer's learned verdict (unknown, active, or disabled) on whether applying its filters pays
off, judged from the fraction of rows they keep.
_Avoid_: keep-ratio gate, consumer gate, gate

### Keys

**Admitted Key**:
A join key the planner accepted for dynamic filtering.
_Avoid_: bound key, key admission (for anything at runtime)

**Key Binding**:
An Admitted Key's attachment to one Channel at one probe column.

**Recorded Type**:
An Admitted Key's build type as planned.
_Avoid_: storage type

**Carrier Type**:
The type a batch actually carries an Admitted Key in, which compressed materialization may narrow.

**Construction Type**:
The type a Membership Filter is built at: the Carrier Type when it stays in the Recorded Type's
family, otherwise the Recorded Type.
_Avoid_: native

### Routes

**Scan Route**:
Applying a Channel's filters at a scan, or by the filter operator directly after it; Zone Maps
are allowed.

**Join-Edge Route**:
Applying a Channel's filters by a filter operator inside a join subtree where no scan can safely
consume the key; only Membership Filters are allowed.
_Avoid_: direct route

**Consumer**:
The state that applies one Channel's filters on one route.
_Avoid_: endpoint

### Publication

**Publication Plan**:
The planner's fixed record of a Producing Join's Admitted Keys and Key Bindings.
_Avoid_: publish plan

**Channel**:
The append-only set of Dynamic Filters that one application point reads; several joins may
publish to it.
_Avoid_: filter set, target, endpoint

**Producer**:
One join's right to publish into one Channel.
_Avoid_: producer handle, completion right, publication right

**Producing Join**:
A hash join of a type that may drop unmatched probe rows (INNER, SEMI, or Right-Family) that builds
Dynamic Filters and holds Producers.
_Avoid_: producer (for the join)

**Terminal (Producer)**:
A Producer that can publish nothing further, whatever its outcome (published, skipped, failed, or
cancelled). Terminal says nothing about whether a filter exists or is useful.
_Avoid_: done, complete, finished

**Terminal Channel**:
A Channel all of whose Producers are Terminal.

**Relevant Producers**:
Every Producer registered on a Channel that a given Producing Join publishes to, including
Producers held by other joins.
_Avoid_: target producers, endpoint producers

**Replica**:
A copy of a published Dynamic Filter on one GPU. Every Replica is ready before any Channel can see
the filter.
_Avoid_: target (for a GPU)

**Key Outcome**:
The final result for one Admitted Key: which filter, if any, reached its Channels, and why not
when none did.
_Avoid_: key line, key result

**Publication Session**:
One Producing Join's single lifecycle of building its Dynamic Filters and publishing them through
all of its Producers. A Publication Session is Terminal when all of its Producers are.
_Avoid_: publisher, one-shot session

### Construction

**Right-Family Join**:
A RIGHT, RIGHT SEMI, or RIGHT ANTI join, whose partitioning the probe side drives.
_Avoid_: reverse semi/anti, probe-driven join

**Construction Site**:
The one operator where a join's Dynamic Filters are built: its build-side PARTITION, or the
BUILD_FILTER stage of a right semi or right anti join.
_Avoid_: publication boundary, observer, build-filter construction site

**Certification**:
A Construction Site's establishing that its build input has finished and every batch is recorded,
which makes the record the Complete Build Inventory.
_Avoid_: validation, readiness

**Complete Build Inventory**:
The frozen, exact record of every batch, row count, and per-column carrier types and null counts
entering a Construction Site, certified once the build input has finished. It is the single proof
that the build is complete, and the sole source of the facts that choose and size the join's
Dynamic Filters.
_Avoid_: whole build, build snapshot

**Construction**:
Building a Publication Session's Dynamic Filters from the batches in its Complete Build Inventory.
_Avoid_: accumulation (for the process), one-shot, whole build

**Construction Plan**:
The choice, from the Complete Build Inventory, of each Admitted Key's filter kinds and sizes.

**Contribution**:
One task adding its certified batch to Construction.

**Decline**:
A Construction Site's refusal of its build input, which ends Construction with no filters. A
probe or decode that cannot apply a filter rejects it; it does not decline.

**Publishing Task**:
The Pipeline Task at a Construction Site whose Contribution completes the Complete Build Inventory,
and which therefore finishes Construction and publishes. It is a role of an ordinary task, not a
separate kind.
_Avoid_: root, elected task, final contributor

**Memory Admission**:
Granting the memory a step needs before it runs; always written in full, never as "admission".
_Avoid_: admission

### Probe scheduling

**Build Layout Completion**:
The point at which a hash join's build-side PARTITION has routed every build row into its final
partitions. Distinct from the earlier choice of the partition layout itself.
_Avoid_: layout stable, layout resolved

**Early Probe Activation**:
Starting a hash join's probe-side work after Build Layout Completion, once all Relevant Producers
are Terminal and its Memory Admission is granted, but before the build data is deposited. Hash
probing itself still waits for each partition's built hash table.
_Avoid_: early probe start, early production, subtrees-b

**Ordinary Probe Activation**:
Starting a hash join's probe-side work only after its build data has been deposited; the rule that
always remains available alongside Early Probe Activation.
_Avoid_: normal activation, normal build-deposit readiness
