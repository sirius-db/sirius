# Phase 22.1 Deferred Items

## Out-of-scope discoveries (not auto-fixed by 22.1-02)

### REQUIREMENTS.md hygiene — IO-MGPU-03 not yet registered
- **Found during:** 22.1-02 final state-update pass.
- **Issue:** Plan 22.1-02's frontmatter declares `requirements: [IO-MGPU-03]`, but `IO-MGPU-03` is not present in `.planning/REQUIREMENTS.md` (gsd-tools `requirements mark-complete IO-MGPU-03` returned `not_found`).
- **Why deferred:** Plan correctness is independent of the REQUIREMENTS.md registration — the plan delivers the policy flip per its own `<must_haves>` block. The missing requirement entry is a project-hygiene gap that the phase planner should backfill (likely during `/gsd:complete-phase` for 22.1).
- **Resolution path:** Whoever runs `/gsd:complete-phase 22.1` should add IO-MGPU-03 to REQUIREMENTS.md (description: "datasource_factory strict policy — registry resolves all schemes or throws kvikio-rejection text") and then mark it complete.
