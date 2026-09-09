# Sirius StarRocks onboarding and landing guide

Open [index.html](index.html) in a browser. It is a standalone, offline web
document: no server, package installation, CDN, or network request is needed to
read it. Source and PR links open GitHub. To share the detailed research and PR
packages as well, share this whole directory.

The guide contains an interactive query-path diagram, a seven-step staging-memory
ownership diagram, a MIG memory-envelope calculator, searchable PR and commit
inventories, dependency lanes, review findings, and an onboarding reading path.

## Reviewed scope

- Repository/worktree: `/home/ubuntu/sirius-wt/s500-mig`.
- Branch: `bench/sf500-2-mig-gpus`.
- Immutable source: [`7610840c`](https://github.com/aocsa/sirius/tree/7610840c03f9086edfa072be72a0eb4c96e03d60).
- Upstream `dev` snapshot: [`ea1c2783`](https://github.com/sirius-db/sirius/commit/ea1c2783191c0a5a2480665f8a18217dc9d9cba3).
- Branch-only history: 80 commits, comprising 59 non-merge commits and 21 merges.
  The branch also lacks six current `dev` commits; the comparison is a merge-base
  diff, not a proposal to revert those upstream changes.
- Original inventory: all 19 open PRs by `aocsa`, all Draft and initially unlabeled.
  Their observed non-skipped checks succeeded, but every PR still required review.
- All 19 draft head commits are in the branch. The remaining 40 source commits
  are assigned to two newly opened benchmark/docs drafts (nine commits) and 20 core
  review packages (31 unique commits). P20 is now draft #1739; the other 19
  packages remain proposals. Mixed repairs `b00334cb` and `98b49df9` are
  explicitly split by file/function across packages.

The C++ streaming foundations #1094, #1320, #1479, #1480 and #1481 are already
merged. Merging a draft into an integration branch does not mean it has landed in
`dev`. The web guide preserves that distinction throughout.

## Drafts created by this review

- [#1737: SF500 two-MIG memory and reproduction runbook](https://github.com/sirius-db/sirius/pull/1737).
- [#1738: TPC-H harness and GPU/MIG placement](https://github.com/sirius-db/sirius/pull/1738).
- [#1739: Concurrent PRPC requests on one connection](https://github.com/sirius-db/sirius/pull/1739).

All three target `dev` from the contributor fork and remain Draft. They preserve the
source commits as cherry-picks; the first two add explicit runtime/deployment
prerequisite notes.
No existing PR was merged, marked ready, relabeled or rebased by this review.
The remaining core packages have concrete PR descriptions and file/commit lists
in [pr-packages/](pr-packages/); P01–P19 are proposals; P20 links to the opened draft. Cross-stack
prerequisites and the findings below must be addressed before those packages are
ready to land. The intended landing workflow is bottom-up via individual queue
entries, following [CONTRIBUTING.md](../../../CONTRIBUTING.md).

## Findings that matter for landing

1. **P1 — Remote staging lease leak:** after a receiver lease grant, a NIXL WRITE
   failure before `transmit_packed` leaves that receiver lease unannounced and
   unreclaimed. Add an idempotent abort path and fault-injection coverage (P06).
2. **P1 — InboundStore lifetime race:** an active `stage()` can dereference its raw
   GPU memory-space pointer after concurrent Context teardown. Quiesce active
   stages or retain ownership through the operation (P16).
3. **P1 — Oracle comparator false pass:** a NaN result is reported as MATCH against
   a finite oracle. Reproduced using the real comparator; explicit non-finite
   handling is needed before #1738 is a reliable correctness gate.
4. **P2 — Unadmitted receive backlog:** inbound tickets allocate directly in the
   GPU pool while receivers wait for EOS. Arena copy-out does not make this
   backlog bounded or spillable. This is a documented capacity/production gate,
   with historical q18/q21 staging OOM evidence, not a new benchmark result.
5. **P2 — Malformed byte-range overflow:** unchecked `start + length` can wrap and
   silently select no row groups. Reject overflowing input (#1696/P01); ordinary
   FE-generated splits were not shown to trigger it.

Exact source anchors, evidence and recommended gates are in the web guide and
[staging.md](research/staging.md), [engine.md](research/engine.md), and
[benchmark-validation.md](research/benchmark-validation.md).

## Staging-specific boundaries

The arena is a stable raw-CUDA allocation **outside** the RMM pool. NIXL registers
that region, frames borrow leases, and received payloads are deep-copied into pool
memory before their arena lease is returned. The CN's receiver runs only after
its required senders finish. The C++ stream primitives deliberately have no
channel-level backpressure; ordinary repository batches use downgrade machinery,
but the current inbound store is not bound to those reservations.

`plan-2mig/` includes plans and evidence from other worktrees. Receive credits,
reservation-bound ingress spilling, bounded frames, HOST-first ingress and the
`CN_SIRIUS_CONFIG_DIR` launcher described there are not in `7610840c`. The recorded
engine also rejects UUID-based `MIG_DEVICES`; its documented working route is
`GPU_DEVICES=0,1`. Both new draft READMEs call out this distinction.

## Verification and limitations

- Authenticated GitHub API inventory, exact PR-head ancestry, full non-merge
  commit assignment, and current `dev` comparison were checked.
- New benchmark draft: shell syntax and Python compilation passed through
  `pixi run`; wrong-cold/correct-warm results correctly fail the comparator.
- The real comparator's NaN false-MATCH was reproduced in a temporary fixture.
  An oracle-only query is intentionally ignored for subset runs; certifying all
  22 requires an explicit expected-query manifest.
- Translator and pure-Rust CN tests were attempted but blocked before execution:
  this worktree has no StarRocks `gensrc/thrift` sources initialized. No fresh
  Rust, C++ GPU, NIXL, or SF500 campaign pass is claimed.
- The lease leak and lifetime race are independently confirmed source-level
  findings, not fault-injection or sanitizer reproductions.
- Browser validation results are recorded in [web-validation.json](research/web-validation.json).

Research scripts create only temporary fixtures. The comparator probe is a
review reproduction, not an assertion that the current source is fixed:

```bash
pixi run python docs/onboarding/starrocks-stack/research/benchmark-validation.py
```

## Maintain the document

Edit [guide-data.json](guide-data.json) for reviewed facts and curated text; edit
[index.template.html](index.template.html) for presentation. Rebuild without
external dependencies:

```bash
pixi run python docs/onboarding/starrocks-stack/build.py
```

The build checks the original snapshot's coverage totals, package IDs and files.
For a new source revision, collect fresh PR metadata and git history, update the
source links and scope explicitly, and revise those snapshot assertions. Never
silently turn the original 19-PR inventory into a claim about current live state.

[github-pr-snapshot.json](research/github-pr-snapshot.json) preserves the original
19 PRs' raw checks, bases, labels and file lists. The lighter
[pr-inventory.json](research/pr-inventory.json) and
[commit-map.json](research/commit-map.json) record the initial investigation;
[guide-data.json](guide-data.json) contains the final, more granular package
boundaries and newly created PR assignments.
