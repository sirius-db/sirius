# Issue #1010 GitHub Delivery Plan

> **Status (2026-07-09):** based on `dev` `fac81e87`, which contains merged #1134
> (`1eecaf97`). Track A shipped delete-only; A1/A2/A3 never merged. Track B is deferred and
> blocks nothing. The [Track C re-evaluation](issue-1010-implementation-plan.md#status-reconciliation-and-track-c-re-evaluation-2026-07-09)
> is authoritative for delivery boundaries.

**Scope:** Delivery structure for the implementation described in
[General Dynamic Filters: Sideways Information Passing at Hash-Join Probe Inputs](issue-1010-dynamic-filter-sip-design.md).

**Parent issues:** [#1010](https://github.com/sirius-db/sirius/issues/1010) for general dynamic
filters and [#1014](https://github.com/sirius-db/sirius/issues/1014) for build-task priority
removal.

This document defines how to divide the design into GitHub issues and pull requests. The design
remains authoritative for architecture; the re-evaluated implementation and cluster plans are
authoritative for current phasing, dependencies, gates, and acceptance evidence.

## Decision

Use GitHub parent issues and native sub-issues as the umbrella. Do not create an umbrella pull
request.

- `#1010` owns the Phase 0 and Track C delivery needed for dynamic-filter SIP v1.
- `#1014` owned Track A and is complete through merged delete-only PR #1134.
- Create one sub-issue for each active, independently reviewable and mergeable #1010 unit below.
- Normally create one pull request per sub-issue.
- Express real ordering constraints with GitHub's `blocked by`/`blocking` relationships, rather
  than relying on list order or prose alone.
- Use a milestone or GitHub Project for release planning and visualization, not as a replacement
  for the parent issues.

GitHub's native
[sub-issues](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/adding-sub-issues)
provide hierarchy and rolled-up progress. Native
[issue dependencies](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/creating-issue-dependencies)
represent merge and enablement constraints.

## Proposed issue hierarchy

### #1010 — General dynamic filters: join-probe SIP

Create these as native sub-issues of `#1010`:

| ID | Suggested sub-issue title | Delivery boundary |
|---|---|---|
| B1 | Phase 0: adopt the released DuckDB candidate-walk fix | Deferred pin-bump playbook for duckdb/duckdb#22963 plus explicit-oracle regressions; blocks no Track C issue. |
| C1a-1 | Dynamic filters: add the version-pinned DuckDB candidate adapter | Move preservation/extraction behind the adapter with CPU parity tests; no runtime consumer. |
| C1a-2 | Dynamic filters: add canonical plan values, lifecycle, and the freeze seam | Add one candidate cache/identity allocator, builder validation, claim decoupling, fallible preparation + noexcept one-shot runtime-plan commit, the exactly-once waiter-free FSM, and strong-generation fresh-execution lifecycle. |
| C1b | Dynamic filters: add typed targets, scan coverage, telemetry, and shadow policy | Absorb former C1c target compaction/parity; add strong compact targets, ID-keyed scan coverage, analyzer support, and shadow selectivity without enforcing it. |
| C1d | Dynamic filters: enforce the repaired selectivity decision | Ship the suppression-policy change behind an independently reversible A/B control. |
| C1e | Dynamic filters: remove the blanket build-filter exclusion | Expand candidates behind a default-off, independently reversible A/B control. |
| C2a | Dynamic filters: add the reusable join-probe consumer | Rebase after C1b; add the mask operation, probe handle, immutable capability/proof token with preallocated local state, noexcept reset hooks, and behavior-parity tests without routes while preserving C1b scan hooks. |
| C2b | Dynamic filters: add safe join/filter reservation accounting | Add the history-aware reservation floor, current `input_stats` semantics, join multiplicity model, and OOM/history-transition tests. |
| C3a | Dynamic filters: discover and validate join-probe SIP descriptors | Add discovery, endpoint identity resolution, immutable planning-descriptor validation, flag, and planning telemetry; install no runtime producer or consumer endpoint. |
| C3b | Dynamic filters: install and run opportunistic join-probe SIP | Group and validate both route ends, prepare all producer plans/consumer proof tokens, commit with noexcept-only operations, then add fan-out, C2 consumption, coverage tooling, and the default-off experiment. |
| C4 | Dynamic filters: make the SIP rollout decision | Turn SIP on by default only if correctness, coverage, wall-time, and memory gates pass. |
| D | Dynamic filters: add route-local ordered activation if required | Conditional recovery for route classes whose opportunistic coverage misses the C3 gate. |

Track E is vNext scope. Give it a separate follow-up parent issue instead of making it part of the
v1 completion percentage.

### #1014 — Remove dynamic-filter build-task prioritization (historical decomposition)

Do not create these A1–A4 sub-issues for current delivery. They record the staged decomposition
that was considered before the measured delete-only outcome:

| ID | Suggested sub-issue title | Delivery boundary |
|---|---|---|
| A1 | Instrument dynamic-filter publication, coverage, and join residency | Instrumentation only; establish the comparison baseline. |
| A2 | Add a legacy/off switch for dynamic-filter build priority | Keep `legacy` as the default and retain the implementation. |
| A3 | Default dynamic-filter build priority to off | Merge only if the measured acceptance gate passes; preserve one-release rollback. |
| A4 | Delete scheduler-level dynamic-filter prioritization | Remove the pass and scheduler knowledge after the default-off release is validated. |

**Actual and complete delivery (2026-07-09):** Track A shipped as one **delete-only** PR #1134
(`1eecaf97` on `dev`; development commit `51da72ac`). The pass was measured on fork #1124
(PASS) and removed outright. A1/A2/A3 never merged, and no A1–A4 sub-issue is required for
current delivery.

Link `#1010` and `#1014` as related issues. #1014 is complete and does not block Track C.

## Dependency graph

Represent the following edges with native `blocked by` relationships:

```text
C1a-1 ──► C1a-2 ──► C1b ──► C2a ──► C2b ───────────┐
                     ├──► C1d                       │
                     ├──► C1e                       ├──► C3b ──► C4
                     └──► C3a ──────────────────────┘
                                                               ▲
                                                               │
                                                      D if C3 misses

B1: deferred, no blocking edges
A: complete through merged #1134
```

C3b therefore has two hard `blocked by` edges: C3a for the validated planning descriptor and C2b
for the corrected consumer/reservation seam. C3a does not install runtime endpoints.

Do not create D merely to complete the issue tree. Create it when C3 telemetry identifies a
specific route class that requires ordering, then mark C4 as blocked by D. If C3 passes without
ordered activation, close C4 without creating D or close a pre-created D issue as not required.

## Pull-request policy

Each pull request should:

1. Target `dev`, the repository's default branch.
2. Close exactly one delivery sub-issue whenever practical.
3. Reference, but not close, its parent issue.
4. Identify blocking issues or pull requests explicitly.
5. State its behavior boundary: instrumentation-only, behavior-preserving, default-off,
   A/B-controlled, default change, or code removal.
6. Include validation, rollout, and rollback conditions from the design.

Use this header in pull-request descriptions:

```markdown
Parent: #1010
Closes: #<sub-issue>
Design section: <B1 | C1a-1 | C1a-2 | C1b | C1d | C1e | C2a | C2b | C3a | C3b | C4 | D>
Depends on: #<issue-or-PR>, or none
Behavior: <instrumentation-only | preserving | default-off | A/B | default change | removal>
Rollback: <flag, revert boundary, or not applicable>
```

For Track A, use `Parent: #1014` instead.

Do not put `Closes #1010` or `Closes #1014` on implementation pull requests. GitHub applies
[closing keywords](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/linking-a-pull-request-to-an-issue)
when a pull request merges into the default branch; using one on a parent would close the umbrella
before all acceptance criteria are met. Use `Refs #1010` or the `Parent:` line and close the parent
manually after its exit criteria pass.

## Branching and stacking

Prefer sibling pull requests that all target `dev` and merge in dependency order. Structural
foundations and default-off behavior are intentionally separated so that most reviews can remain
independent.

Use a stacked pull request only when a later unit cannot be reviewed or tested without unmerged
prerequisite code:

- base the dependent pull request temporarily on its immediate prerequisite branch;
- open it as a draft and state `Depends on #<PR>` prominently;
- describe which commits belong only to the dependent unit; and
- after the prerequisite merges, rebase or retarget the dependent pull request to `dev` and
  verify its resulting diff.

Avoid a single deep stack across C1a-1 through C4. It obscures rollback boundaries, makes reviews
depend on unrelated unfinished work, and prevents independent experiments from merging or being
reverted cleanly. B1 is a separate deferred pin-bump playbook with no Track C stacking edge.

## Labels, milestone, and project view

Suggested shared labels:

- `area: dynamic-filters`
- `track: scheduler` for completed #1134; A1–A4 are historical only
- `track: duckdb-contract` for B1 and C1a-1
- `track: producer` for C1a-2, C1b, C1d, and C1e
- `track: consumer` for C2a and C2b
- `track: sip` for C3a, C3b, C4, and D
- `behavior-change`
- `default-off`
- `conditional`

Use a milestone such as `Dynamic Filters SIP v1` for C1a-1 through C4. Keep deferred B1 visible
as a related pin-bump issue rather than counting it toward Track C completion. Track A is complete.

If a GitHub Project is useful, add both parent issues, all active sub-issues, and their pull
requests. Useful fields are `Track`, `Design ID`, `Behavior`, `Status`, and `Target release`.
GitHub Projects can expose the
[linked pull-request field](https://docs.github.com/en/issues/planning-and-tracking-with-projects/understanding-fields/about-pull-request-fields),
so the project should derive PR status instead of maintaining a second manual checklist.

## Parent-issue content

Keep each parent issue short and operational:

```markdown
## Design

<link to the design document at a stable GitHub revision>

## Scope

<one-paragraph v1 scope>

## Sub-issues

<native GitHub sub-issue list>

## Dependency and rollout rules

<short link to this delivery plan and any currently active blockers>

## Exit criteria

- [ ] Required sub-issues are complete.
- [ ] Correctness tests pass against explicit oracles.
- [ ] Performance and memory gates pass.
- [ ] Default and rollback decisions are recorded.
- [ ] Architecture and operator documentation is updated.
```

The native sub-issue list is the progress record. Avoid duplicating it with a second hand-edited
checkbox list in the issue body; retain only exit criteria and information that GitHub cannot
derive.

## Completion rules

Close a delivery sub-issue when its pull request has merged into `dev` and its stated acceptance
criteria pass. Closing a code pull request without merging does not complete the sub-issue unless
the issue is explicitly resolved as unnecessary.

Close `#1010` only after C4 records the v1 default-on or explicit no-ship decision and the required
documentation is current. `#1014` is complete: merged PR #1134 removed the scheduler policy after
the measured fork passed, and no A1–A4 closure work remains.
