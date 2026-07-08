# Issue #1010 GitHub Delivery Plan

**Scope:** Delivery structure for the implementation described in
[General Dynamic Filters: Sideways Information Passing at Hash-Join Probe Inputs](issue-1010-dynamic-filter-sip-design.md).

**Parent issues:** [#1010](https://github.com/sirius-db/sirius/issues/1010) for general dynamic
filters and [#1014](https://github.com/sirius-db/sirius/issues/1014) for build-task priority
removal.

This document defines how to divide the design into GitHub issues and pull requests. The design
document remains authoritative for architecture, correctness, gates, and acceptance criteria.

## Decision

Use GitHub parent issues and native sub-issues as the umbrella. Do not create an umbrella pull
request.

- `#1010` owns the Phase 0 and Track C delivery needed for dynamic-filter SIP v1.
- `#1014` separately owns Track A because scheduler cleanup is independently measurable,
  releasable, and reversible.
- Create one sub-issue for each independently reviewable and mergeable unit below.
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
| B1 | Phase 0: repair the pinned DuckDB join-filter candidate walk | Backport or advance past duckdb/duckdb#22963 and add explicit-oracle regressions. |
| C1a | Dynamic filters: add a version-pinned DuckDB candidate adapter | Preserve target identity, extract Sirius value snapshots, and decouple publisher claims without behavior changes. |
| C1b | Dynamic filters: introduce typed target keys and decision telemetry | Add strong key/index types, materialization telemetry, and the shadow selectivity signal without changing policy. |
| C1c | Dynamic filters: enable per-key target fan-out | Ship behind an independently reversible A/B control. |
| C1d | Dynamic filters: enforce the repaired selectivity decision | Ship the suppression-policy change behind an independently reversible A/B control. |
| C1e | Dynamic filters: remove the blanket build-filter exclusion | Expand candidates behind a default-off, independently reversible A/B control. |
| C2 | Dynamic filters: add the reusable mask operation and join-probe consumer | Add the composed probe checkpoint, endpoint component, reservation model, and focused tests without planning routes to it. |
| C3 | Dynamic filters: plan and run opportunistic join-probe SIP | Add discovery, resolution, topology freeze, layered scan/probe targets, telemetry, and a default-off SIP experiment. |
| C4 | Dynamic filters: make the SIP rollout decision | Turn SIP on by default only if correctness, coverage, wall-time, and memory gates pass. |
| D | Dynamic filters: add route-local ordered activation if required | Conditional recovery for route classes whose opportunistic coverage misses the C3 gate. |

Track E is vNext scope. Give it a separate follow-up parent issue instead of making it part of the
v1 completion percentage.

### #1014 — Remove dynamic-filter build-task prioritization

Treat `#1014` as a separate parent and create these native sub-issues:

| ID | Suggested sub-issue title | Delivery boundary |
|---|---|---|
| A1 | Instrument dynamic-filter publication, coverage, and join residency | Instrumentation only; establish the comparison baseline. |
| A2 | Add a legacy/off switch for dynamic-filter build priority | Keep `legacy` as the default and retain the implementation. |
| A3 | Default dynamic-filter build priority to off | Merge only if the measured acceptance gate passes; preserve one-release rollback. |
| A4 | Delete scheduler-level dynamic-filter prioritization | Remove the pass and scheduler knowledge after the default-off release is validated. |

Link `#1010` and `#1014` as related issues. Do not make either entire parent issue block the other:
Tracks A and C are an experimental matrix, and neither is assumed to repair the other's
regressions.

## Dependency graph

Represent the following edges with native `blocked by` relationships:

```text
C1a ──► C1b ──► C1c
              ├─► C1d
              └─► C1e

B1 ───────────► C1c enablement
 ├────────────► C1e enablement
 └────────────► C3

C1a + C1b + C1c + C1d + C1e + C2 ──► C3 ──► C4
                                               ▲
                                               │
                                      D, only if C3
                                      misses its gate

A1 ──► A2 ──► A3 ──► A4
```

The B1 constraint is specifically an enablement constraint. Structural work may proceed in
parallel, but vulnerable behavior must not be enabled against the unpatched DuckDB pin.

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
Design section: <B1 | C1a | C1b | C1c | C1d | C1e | C2 | C3 | C4 | D>
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

Avoid a single deep stack spanning B1 through C4. It obscures rollback boundaries, makes reviews
depend on unrelated unfinished work, and prevents independent experiments from merging or being
reverted cleanly.

## Labels, milestone, and project view

Suggested shared labels:

- `area: dynamic-filters`
- `track: scheduler` for A1–A4
- `track: duckdb-contract` for B1 and C1a
- `track: producer` for C1b–C1e
- `track: consumer` for C2
- `track: sip` for C3, C4, and D
- `behavior-change`
- `default-off`
- `conditional`

Use a milestone such as `Dynamic Filters SIP v1` for B1 and C1a–C4. Track A may use the same
milestone if it shares a release target, but it remains owned by `#1014`.

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
documentation is current. Close `#1014` only after A4 removes the scheduler policy, or after a
recorded decision changes the issue's target outcome based on the measured gate.
