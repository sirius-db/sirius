## Description

The StarRocks integration spans C++ streaming, Rust FFI, scan ownership, plan
translation, compute-node dispatch and GPU exchange memory, but its draft PRs
and integration commits have no single onboarding or landing map. This adds an
offline interactive guide with source links, PR dependencies, a staging-ownership
stepper, a MIG memory calculator, searchable PR/commit inventories, and a reading
path for new team members.

The reviewed snapshot is `bench/sf500-2-mig-gpus` at `7610840c`, compared with
`dev` at `ea1c2783`: 19 original drafts, 59 non-merge commits and 21 integration
merges. All 40 commits outside the original drafts are accounted for. Nine are
in the benchmark/runbook extractions, and 31 are mapped into 20 core review
packages; P20 is now #1739 and P01-P19 remain proposed. Mixed repair commits are
split by file/function in the prepared PR descriptions.

The guide distinguishes upstream-merged streaming foundations, unmerged drafts,
integration-only work and capabilities described in other performance worktrees.
It highlights source-confirmed remote-lease and InboundStore lifetime findings,
the reproduced NaN comparator false pass, and the unadmitted-ingress capacity
limit. The page runs without a server or external JavaScript/CSS dependencies.
Its checked-in JSON snapshot and template rebuild with the included Python script.

Validation: all applicable pre-commit hooks pass on the new documentation files.
Playwright Chromium passed navigation, filters, search, ownership steps and memory
calculation at 360/736/1024/1440 px in light and dark themes, with no JavaScript
errors or horizontal overflow. The generated page is deterministic; principal
source anchors and all commit assignments were checked. A temporary comparator
probe reproduces the NaN false-MATCH and verifies wrong-cold/correct-warm rejection.

This is a source review and documentation change. Rust/CN test attempts stopped
before execution because StarRocks Thrift submodule sources were uninitialized;
no new GPU, NIXL or SF500 run is claimed. Reported historical benchmark results
retain their provenance. Keep this documentation PR Draft for factual review by
the component owners; no existing PR is marked ready or merged by this change.

## Checklist

- [x] Read CONTRIBUTING.md and use the self-contained fork-to-dev path.
- [x] Include motivation, scope, reproducible document build and validation.
- [x] Preserve immutable source links and snapshot semantics.
- [x] Distinguish confirmed findings from capacity limits and unexecuted tests.
- [ ] Component owners review the technical map and proposed package boundaries.

## References

- Source branch: https://github.com/aocsa/sirius/tree/7610840c03f9086edfa072be72a0eb4c96e03d60
- Staging arena: https://github.com/sirius-db/sirius/pull/1693
- Runbook extraction: https://github.com/sirius-db/sirius/pull/1737
- Benchmark extraction: https://github.com/sirius-db/sirius/pull/1738
- Concurrent PRPC extraction: https://github.com/sirius-db/sirius/pull/1739
