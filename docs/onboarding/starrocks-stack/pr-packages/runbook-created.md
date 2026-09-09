## Description

The SF500 two-MIG reproduction and memory-sizing instructions currently live only
on `bench/sf500-2-mig-gpus`. This PR extracts the runbook into `plan-2mig/`, with the
pool/arena/host budgets, DISK-tier configuration, benchmark gates, original plan
provenance, and reproduce prompts kept together.

Source commits: `7c18ce8b`, `7092f445`, `7610840c`. A README clarification distinguishes
recorded results from new verification and explicitly names features supplied by
other runtime trees. It also documents that the recorded engine uses
`GPU_DEVICES=0,1` on MIG; UUID-based `MIG_DEVICES` is not yet supported there.

This is a documentation-only extraction against `dev`, with no source-file
overlap with the 19 existing StarRocks draft PRs. The plan is kept as a unit because
its sizing assumptions, execution gates, and reproducibility instructions refer
to one another; `original/` preserves the source plan for those references.

Validation: `git diff --check ea1c2783...HEAD` passes. Reviewed every referenced
local file within the extracted directory and verified all twelve files are
documentation. GPU runs, DISK smoke tests and runtime rebuilds were not performed
for this extraction. Historical measurements are not a fresh validation result.

Runtime prerequisites remain in the corresponding engine/CN/benchmark work. In
particular, receive credits, HOST-first ingress, bounded frames and
`CN_SIRIUS_CONFIG_DIR` are described for separate branches and are not implemented
by this PR. Keep this PR Draft while those dependencies and runbook portability
are reviewed.

## Checklist

- [x] Read CONTRIBUTING.md and use the self-contained fork-to-dev path.
- [x] Describe scope, motivation, validation and runtime dependencies.
- [x] Document configuration and the MIG UUID limitation.
- [ ] Validate reproduction on the named runtime revisions and hardware.

## References

- Source: https://github.com/aocsa/sirius/tree/7610840c03f9086edfa072be72a0eb4c96e03d60/plan-2mig
- Staging allocator: https://github.com/sirius-db/sirius/pull/1693
- Cluster bring-up: https://github.com/sirius-db/sirius/pull/1714
