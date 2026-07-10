# C1 Current-Code Audit

**Audit date:** 2026-07-10
**Branch inspected:** `issue-1010-track-c`
**Scope:** the version-pinned adapter, candidate cache, strong identities, publication-plan
builder, hash-key resolution, freeze seam, publisher decoupling, scan-channel handoff, and their
tests.

This document evaluates the code that exists in the working tree. It is not a replacement for the
[C1 implementation plan](C1ab-adapter-foundation.md). The implementation plan describes the target
contract. This audit says which parts of that contract exist now, which parts are incomplete, and
which implemented parts need correction.

No production or test code was changed as part of this audit.

## Executive Result

The current code is a substantial C1a-2 foundation, not a complete C1a-2 implementation.

The following path is real and mostly coherent:

```text
DuckDB metadata
      |
      v
version-pinned adapter
      |
      v
candidate cache
      |
      v
publication-plan builder
      |
      v
hash-join key decisions
      |
      v
fallible preparation
      |
      v
noexcept slot commit
      |
      v
immutable scan publication plan
      |
      v
runtime publisher using Sirius values only
```

The following required C1a-2 path does not exist yet:

```text
prepared topology lease
      |
      v
begin one execution generation
      |
      +--> clear and reopen every channel exactly once
      +--> reset every publication attempt exactly once
      +--> reset the execution-wide filter-ID counter
      +--> capture the event-clock epoch
      |
      v
run publishers and consumers
      |
      v
quiesce tasks and publishers
      |
      +--> emit normal or partial summaries
      +--> close every channel exactly once
      +--> cancel unfinished attempts
      +--> release the prepared-topology lease
```

Because the second path is missing, the code must not be described or delivered as complete
C1a-2.

## Status By Planned Unit

| Planned unit | Current status | Audit conclusion |
|---|---|---|
| C1a-1 adapter and preservation | Mostly implemented | Core helper behavior is strong; one direct metadata read remains outside the adapter and production-copy parity is not tested end to end. |
| C1a-2 strong ID and ordinal types | Partly implemented | Planning IDs and ordinal types exist. Execution identity exists only as an unused type. |
| C1a-2 candidate cache | Implemented with contract gaps | It is now consumed by physical planning, but pre/post node-set equality is not enforced. |
| C1a-2 builder and key decisions | Implemented with validation gaps | Normal planner inputs are coherent. The public planning view can expose invalid data before final validation. |
| C1a-2 prepare/commit freeze | Implemented with boundary gaps | The happy path prepares all values before noexcept commit. Topology comparison and API encapsulation are incomplete. |
| C1a-2 publisher decoupling | Implemented | Runtime publisher input is a frozen Sirius plan rather than raw DuckDB metadata. |
| C1a-2 reasoned publication lifecycle | Not implemented | The code still uses the old five-state claim and a `void` publisher result. |
| C1a-2 fresh execution/reset | Not implemented | Channels, attempts, and filter IDs are not reset for reuse. |
| C1b domains, compact targets, telemetry | Not implemented | Only placeholders and handoff seams exist. |
| C2 join-probe consumer | Not implemented | Existing `dynamic_filter_router` tests are Phase-1 scan-channel tests, not C2 or C3 tests. |
| C3 discovery and runtime routes | Not implemented | `join_probe_publish_target` is still an empty placeholder. |

## Blocking Findings

### 1. Cached physical-plan reuse can retain stale filters

**Severity:** blocker for C1a-2 completion and for any reused physical topology.

The freeze helper has an explicit cached-reuse branch. It verifies the existing frozen plan and
reuses it. The mutable execution state is not reset.

The current sequence can therefore be:

```text
execution 1
    join state:     OPEN -> PUBLISHING -> FINISHED
    channel state:  filters appended
    scan teardown:  channel closed

execution 2
    freeze:         verifies and reuses the frozen plan
    join state:     still FINISHED
    channel state:  still closed and still contains execution-1 filters
    publication:    cannot claim OPEN, so no new filters are produced
    scan:           can still observe stale execution-1 filters
```

Applying an old membership filter to new data can remove rows that should survive. This is a
correctness risk, not only a telemetry problem.

Relevant code:

- `src/op/dynamic_filter_publish_plan.cpp:443-464`: cached freeze verification and reuse.
- `src/op/sirius_physical_hash_join.cpp:1397-1422`: old five-state publication attempt.
- `src/op/sirius_physical_hash_join.cpp:1437-1446`: publication requires state `OPEN`.
- `src/op/sirius_dynamic_filter.cpp:454-496`: append-only channel and one-way close.
- `src/pipeline/sirius_pipeline_converter.cpp:274-299`: the same channel can be attached again.

Transparent execution currently rebuilds a fresh Sirius physical plan for each execution, which
masks this problem on that path. The freeze API nevertheless claims cached topology reuse, and the
C1a-2 contract explicitly requires reused prepared state to start clean.

**Required before C1a-2 completion:** implement the central execution coordinator, topology lease,
strong generation, channel clear/reopen, attempt reset, filter-ID reset, and canonical success/abort
teardown described in the C1 plan. Add a two-execution test whose second execution uses different
parameters or source data.

### 2. The planned publication lifecycle is still missing

**Severity:** blocker for C1a-2 completion.

The current join states are:

```text
OPEN -> PUBLISHING -> FINISHED
                    -> FAILED
OPEN ----------------> CLOSED
```

The required lifecycle distinguishes why no filter was produced:

```text
OPEN -> PUBLISHING -> PUBLISHED
                    -> NO_MATERIALIZATION(reason)
                    -> FAILED
OPEN ---------------> NO_MATERIALIZATION(reason)
                    -> FAILED
OPEN or PUBLISHING -> CANCELLED during quiescent teardown
```

The current `dynamic_filter_publisher::publish` returns `void`. Empty build, unsupported mode,
unavailable source, policy skip, and closed consumers do not become structured outcomes with
timestamps and per-target results. `dynamic_filter_execution_identity` is defined, but no accepted
plan owns it and no code calls `begin_execution()` or mints a filter ID.

This missing work is part of C1a-2, not C1b.

### 3. The current branch is not split into the planned delivery units

**Severity:** delivery blocker if this branch is proposed for review as C1a-1 or C1a-2.

The plan requires C1a-1, C1a-2, and C1b to be independently reviewable and revertible. Commit
`82d6de36` combines the adapter with later Track C work and documentation. Some files required by
the current build are still only working-tree or untracked files.

This does not make the implementation logic wrong. It does make the planned review, rollback, and
bisect boundaries unavailable.

**Required before delivery:** reconstruct the intended commit/PR boundaries and ensure each one is
green on its own.

## Major Findings

### 4. Cache extraction does not require the tree that was captured

Capture records joins with `try_emplace`. Extraction writes with `operator[]`.

This sequence currently succeeds:

```text
capture_pre_resolver(tree_A)
extract_post_resolver(tree_B)
```

The result contains post-resolver candidates for tree B even though tree B had no pre-resolver
capture. Entries from tree A remain null. C1b would then attach missing or incorrect pre-resolver
evidence.

Relevant code:

- `src/planner/dynamic_filter_candidate_cache.cpp:89-94`
- `src/planner/dynamic_filter_candidate_cache.cpp:120-124`

**Required invariant:** the two walks must see exactly the same logical join node set. Extraction
must reject an unseen node, reject a missing captured node, and fill every captured entry exactly
once.

```text
capture phase                      extraction phase

join A* -> pending evidence        join A* -> adapter candidate
join B* -> pending evidence        join B* -> adapter candidate
          |                                  |
          +--------- exact same set ---------+
                                             |
                                      validate and merge
                                             |
                                             v
                                      immutable entries
```

### 5. One pinned-DuckDB metadata read remains outside the adapter

`src/planner/dynamic_filter_candidate_cache.cpp:92` reads `join.filter_pushdown` directly to decide
whether the resolver fence applies.

The documented architecture says the adapter is the only module that reads DuckDB's internal
dynamic-filter metadata. The current direct read is small and does not cause a known query-result
bug, but it breaks the pin-audit boundary.

**Required before the adapter boundary is called complete:** route the metadata-presence query
through the adapter, or explicitly weaken the architecture rule. The stronger single-owner rule is
preferable.

### 6. The pre-freeze planning view can expose unvalidated values

`resolve_keys()` records whatever decision and key vectors it receives. It deliberately waits for
`finalize()` to report count and identity errors. `planning_view()` can be called before
`finalize()`.

That means the view can contain:

- `decision == admitted` with no `admitted_key`;
- a candidate vector that is a permutation rather than strict DuckDB-ordinal order;
- `enabled == true` even if `wired == false` in a malformed builder;
- an admitted non-equality candidate if an upstream caller supplies inconsistent decisions.

C3 is supposed to trust this view before freeze. The plan promises that an admitted key is present
if and only if the decision is admitted and that the array is in strict DuckDB-ordinal order.

**Required before C3 reads the view:** validate the complete view contract before exposing it.
Final validation should also require `candidate.duckdb_ordinal.value == vector_position` and require
every admitted candidate to be an equality candidate.

### 7. Cached topology verification is not full value equality

The current frozen descriptor compares:

- publication-plan ID;
- enabled state;
- decision bytes;
- target IDs; and
- channel IDs.

It does not compare several values that change runtime behavior:

- DuckDB ordinal and condition index;
- admitted Sirius key ordinal;
- build column index and build type;
- probe column indexes and probe storage types;
- channel-object association;
- zone-map policy;
- domain threshold and domain evidence;
- GPU/HOST replica placement; or
- the preserved `wired` decision.

The implementation-plan appendix says cached verification is exact builder-versus-frozen value
comparison. The current descriptor is only a partial shape check.

**Required:** define one canonical descriptor containing every runtime-relevant value, or compare
the full immutable plans directly through a canonical value representation. A digest may reject
quickly, but full equality is authoritative.

### 8. Plan-wide identity invariants are not validated at freeze

The generator currently constructs valid IDs by using one allocator and memoizing channel IDs.
The fallible freeze nevertheless does not validate the plan-wide identity contract.

It accepts these invalid shapes:

- two producers with the same publication-plan ID;
- two targets on different producers with the same target ID;
- one channel ID naming two different channel objects; and
- one channel object carrying two different channel IDs.

The last two cases matter because the future execution coordinator will deduplicate reset and close
operations by strong channel ID.

One current builder test gives two different `sirius_dynamic_filter_set` objects the same channel
ID, so the test currently encodes a shape the final contract should reject.

**Required:** global preparation must validate publication and target uniqueness plus the two-way
channel-ID/channel-object bijection. Invalid preparation must leave every slot unassigned.

### 9. The freeze boundary is public and can be bypassed

The intended architecture has one legal freeze path. Current public APIs expose:

- `sirius_physical_hash_join::dynamic_filter_builder()`;
- builder candidates, scan drafts, channels, and `finalize()`;
- `prepare_dynamic_filter_plan_assignment()`; and
- `commit_dynamic_filter_plan_assignment()`.

The full `dynamic_filter_publish_plan` constructor is also public. An internal caller can therefore
construct and commit a plan without the generic validation path.

The builder accessor is const, so this is not an immediate mutation escape. It is still broader
than the documented "planning view only" and makes accidental C3 coupling possible.

**Required:** narrow these APIs to the freeze implementation with friendship or an internal owner,
or document and test a precise internal-only exception.

### 10. `single_assignment` silently accepts a foreign or consumed token

Tokens from two slots with the same payload type have the same C++ type. This is expressible:

```text
token = slot_A.prepare(value)
slot_B.commit(token)
```

`commit_assignment()` clears the token's owner pointer before it checks the owner. It then returns
because the token belongs to slot A. Slot A remains permanently `pending`, and token destruction can
no longer roll it back.

Committing an already-consumed token also silently returns, although the plan says a direct second
commit is an internal error.

The current aggregate commit code pairs each token correctly, so this is a forward-safety defect in
the transaction seam rather than a demonstrated happy-path failure.

**Required:** make foreign-token misuse impossible or fail loudly without stranding the source
slot. Align the plan's second-commit contract with the actual API and add focused tests.

## C1b Handoff Findings

### 11. C1b must not populate the current zero-sentinel vector

The current C1a-2 plan stores domain evidence as `vector<size_t>`, with zero meaning unknown. It
fills the vector with zeros, which preserves today's effective behavior.

The current publisher contains active suppression code:

```text
known row coverage >= threshold  -> continue; skip the key
known numeric range >= threshold -> do not publish the zone map
```

If C1b only starts filling that vector with real values, it will activate enforcement immediately.
That would ship the later C1d behavior inside C1b and violate the behavior-preserving promise.

C1b must make these changes together:

1. Replace zero-as-unknown with `std::optional<size_t>` and `nullopt`.
2. Attach one optional value to the correct DuckDB filter ordinal.
3. Replace producer suppression with `unknown | would_publish | would_suppress` observation.
4. Keep actual exact-set/Bloom/zone-map materialization independent from that shadow decision.
5. Leave actual membership enforcement to C1d.
6. Keep zone-map suppression shadow-only until its separate evidence gate is satisfied.

### 12. The C1b cache merge type and algorithm are not specified

The cache currently stores only:

```text
join pointer -> shared_ptr<const duckdb_join_filter_candidate>
```

C1b needs pending pre-resolver evidence and one final immutable enriched entry. The plan does not
yet choose how the pre-pass learns which conditions are candidates without violating the adapter's
single-owner rule.

Two coherent choices exist:

```text
choice A
    adapter exposes a narrow pre-resolver candidate outline
    cache traces only recorded candidate conditions
    post extraction verifies the same condition indexes

choice B
    cache traces every join condition by join-condition index
    post adapter extraction selects candidate conditions
    evidence is reordered into DuckDB-filter-ordinal order
```

Choice B keeps one full candidate extraction but may do extra lineage/statistics work. Choice A does
less work but gives the adapter a separate pre-resolver read surface. The plan must choose one,
define the enriched entry type, and specify arity and mismatch handling before C1b coding begins.

### 13. A scan cannot currently report its channel ID when no filter arrives

Channel IDs are minted while producer target drafts are built. A physical scan retains only the
`sirius_dynamic_filter_set` pointer.

C1b requires one `scan_consume_summary` per channel even if zero filters become visible. In that
case there is no filter entry from which the scan can learn the ID.

**Required design decision:** the stable channel ID must belong to persistent channel state or to a
persistent generator-created channel record shared by producer and consumer. It should be minted
when the channel is first created, regardless of whether the scan or producer is planned first.

The invariant should be:

```text
one Sirius channel object <-> one channel_id

all producers targeting that channel reuse both
the scan consumer can always read the same channel_id
```

## Test Assessment

### Focused tests that passed during this audit

| Selection | Result |
|---|---|
| Candidate cache | 14 cases, 28 assertions |
| Adapter | 16 cases, 76 assertions |
| Transparent preservation | 7 cases, 23 assertions |
| Adapter/router/builder combined selection | 39 cases, 149 assertions |
| Publication-plan builder | 16 cases, 55 assertions |
| Freeze seam | 11 cases, 44 assertions |
| `single_assignment` | 8 cases, 20 assertions |
| Scoped `git diff --check` | Passed |

The release tests above passed. The debug cache binary could not be launched in the managed shell
because the sandbox failed before process start with `bwrap: loopback: Failed RTM_NEWADDR`. This is
not a test failure, but it leaves the debug-only resolver-fence branch unverified by this audit.

### Important missing tests

The current tests do not prove:

- two executions of one prepared graph start with empty channels and open attempts;
- all reasoned lifecycle outcomes and success/abort teardown;
- global publication, target, and channel identity invariants;
- planner-generated ID reuse across two producers and one scan;
- equality-plus-range key alignment through the real hash-join constructor;
- equality after a skipped `NOT DISTINCT` condition;
- cast and unresolved decisions through the real constructor;
- engine pipeline enumeration freezes every retained hash join, including composite shapes;
- full frozen-descriptor equality over policy, columns, types, and replica placement;
- allocation fault injection at every preparation step;
- foreign-token and consumed-token behavior;
- production `LogicalOperator::Copy` plus metadata restoration;
- CPU fallback result parity; or
- both release and debug resolver-fence semantics in automated CI.

The repeated-`find()` cache test proves that callers share the stored pointer. It does not directly
count adapter calls, so it does not prove extraction happened exactly once internally.

## Work That Is Strong

The following work is aligned with the target design and should be retained:

- Adapter extraction produces Sirius-owned values and validates local structural corruption.
- Preservation preflights the whole copied tree before attaching either endpoint.
- Exact shared scan-channel identity survives the logical-plan copy.
- The candidate cache is generator-local, immutable after extraction, and now has a real physical
  planning consumer.
- Strong types separate publication, target, channel, filter, generation, and ordinal spaces.
- The generator owns the planning allocator and memoizes shared scan-channel IDs.
- Raw `JoinFilterPushdownInfo` no longer reaches the hash join or publisher.
- The hash join resolves candidate keys once and the publisher replays frozen decisions.
- Preparation finalizes all producer plans before any slot commit begins.
- Uncommitted tokens roll back pending slots when preparation throws.
- Runtime reads before freeze fail loudly rather than silently acting disabled.
- Builder-less joins receive a canonical disabled immutable plan.
- The current publisher consumes only the frozen Sirius plan plus the runtime build table.

## Documentation Decisions Required Before More C1 Work

1. Define the exact current completion boundary for C1a-2. Builder/freeze/runtime decoupling alone
   is not completion; lifecycle and fresh execution are mandatory.
2. Define the complete canonical topology descriptor used by cached verification.
3. Choose the C1b pre-evidence collection algorithm and final enriched cache-entry type.
4. Assign ownership of `channel_id` so zero-filter scan summaries can name their channel.
5. State clearly that C1a-2 may mint filter IDs internally, while C1b adds ID-carrying channel
   entries and machine-parsed telemetry, or move the whole filter-ID requirement consistently to
   C1b. The current plan assigns parts to both phases.
6. Define plan-wide identity validation before the execution coordinator deduplicates by ID.
7. Decide whether cached C1a-2 verification stores a prepared descriptor or recomputes full
   builder-versus-frozen values. C3b may add a separate persistent routing descriptor later, but
   the phase ownership must be explicit.
8. Mark historical proposal documents as historical so pasted code is not mistaken for the source
   of truth.

## Recommended Delivery Decision

Do not present the current tree as complete C1a-2 and do not begin C1b by merely filling the current
domain vector.

The safest sequence is:

1. Finish the current C1a-2 builder/freeze review findings.
2. Add plan-wide identity validation and full topology comparison.
3. Implement the reasoned publication result and exactly-once attempt state.
4. Implement prepared-topology ownership, central begin/end, generation, and clean reset.
5. Prove two executions of one prepared graph are independent.
6. Reconstruct independent C1a-1 and C1a-2 delivery units.
7. Only then implement C1b's optional domain evidence, shadow-only policy, compact targets, and
   telemetry.
