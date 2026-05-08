# Phase 22.1 Deferred Items

## Out-of-scope discoveries (not auto-fixed by 22.1-02)

### REQUIREMENTS.md hygiene — IO-MGPU-03 not yet registered
- **Found during:** 22.1-02 final state-update pass.
- **Issue:** Plan 22.1-02's frontmatter declares `requirements: [IO-MGPU-03]`, but `IO-MGPU-03` is not present in `.planning/REQUIREMENTS.md` (gsd-tools `requirements mark-complete IO-MGPU-03` returned `not_found`).
- **Why deferred:** Plan correctness is independent of the REQUIREMENTS.md registration — the plan delivers the policy flip per its own `<must_haves>` block. The missing requirement entry is a project-hygiene gap that the phase planner should backfill (likely during `/gsd:complete-phase` for 22.1).
- **Resolution path:** Whoever runs `/gsd:complete-phase 22.1` should add IO-MGPU-03 to REQUIREMENTS.md (description: "datasource_factory strict policy — registry resolves all schemes or throws kvikio-rejection text") and then mark it complete.

### Wave-2 in-flight collision RESOLVED: sirius_engine.cpp:385 caller wiring
- **Found during:** 22.1-04 Task 2 (mcp build gate, first run).
- **Issue (now resolved):** Sibling Plan 22.1-05's commit `5c3522b` added a `metadata_ioctx` parameter to `read_iceberg_delete_data` ahead of wiring the caller at `src/sirius_engine.cpp:385`. First mcp build during Plan 22.1-04 verification failed at this site.
- **Resolution:** Plan 22.1-05's commit `9ea53e9 feat(22.1-05): forward metadata_ioctx through public API to materialize step` (landed during my plan execution) wired the caller. Re-run of mcp build during Plan 22.1-04 Task 2 PASSED at exit 0 / 9.9s; `[pin_mgpu]` 2/2 PASS / 46 assertions / 7.1s.
- **Note:** This is documented for posterity as a Wave-2 parallel-execution coordination event — temporary header-vs-caller commit ordering inversion across sibling plans, self-resolved by Plan 22.1-05's own Task 4.
