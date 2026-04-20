# Phase 4: cuCascade Bump + v1.0 Re-integration - Context

**Gathered:** 2026-04-20
**Status:** Ready for planning
**Mode:** Infrastructure phase — smart_discuss skipped

<domain>
## Phase Boundary

Replay the 23 v1.0 multi-GPU commits from `refs/remotes/felipe-ssh/feature/multi-gpu-execution` onto current `dev`, bumping the cuCascade submodule to `origin/main` (f47de0b) so PR #96 headers (`idisk_io_backend`, `io_backend_registry`) are available. The port must adapt to `dev`'s 47 intervening commits: sirius-native types (`logical_type`/`type_id`, PR #643), YAML config replacing libconfig++ (PR #565), DuckDB vocabulary removal (PRs #564/#626/#628), AST expression executor (#531), hive partitioning (#570), row-group pruning (#363).

Phase 4 ends with a green build + the v1.0 multi-GPU test suites (`[multi_gpu_foundation]`, `[test_gpu_execution_locality]`, downgrade NUMA tests) passing. No kvikio removal yet (that's Phase 5). No new features (MGPU-* closures are Phase 6). No P2P / adaptive scan (Phase 7).

</domain>

<decisions>
## Implementation Decisions

### Claude's Discretion
All implementation choices are at Claude's discretion — pure infrastructure phase. Guidance:
- **Conflict strategy** — user chose "Attempt auto-resolution, pause on hard conflicts." Resolve trivial conflicts (formatting, clear renames from `LogicalType` → `logical_type`, libconfig→YAML keyname substitutions, namespace moves). Pause on: semantic behavior changes, ambiguous type replacements, any conflict where both sides wrote substantive logic, changes to test expectations.
- **Commit strategy** — cherry-pick commits one-by-one in v1.0 order, preserving each commit's authorship and subject. Rewrite commit message only when conflict resolution substantively changes the diff (prefix with `(rebased)`).
- **Build command** — use `mcp__project-commands__run_command`, never `pixi run`/`make` directly (user rule).
- **Stream discipline** — any code being ported that still uses `rmm::cuda_stream_default` gets fixed now if trivial; otherwise flagged for HYG-02 in Phase 5.
- **Test gating** — multi-GPU tests that require N>1 GPUs may `WARN+return` on single-GPU hosts per v1.0 Catch2-v2 convention.

</decisions>

<code_context>
## Existing Code Insights

### Source branches
- `refs/remotes/felipe-ssh/feature/multi-gpu-execution` — 23 commits to replay (HEAD before 39ea7bd "docs: map existing codebase" up to tip)
- `dev` (current base) — HEAD at 484db35 "Add extension-ci-tools distribution workflow (#621)"
- Current branch `feature/single-node-multi-gpu2` is based on `dev`

### Key dev-drift areas the port must adapt to
- `src/include/**/*types*.hpp`, `src/type/` — sirius-native `logical_type` / `type_id` (PR #643). v1.0 code used DuckDB `LogicalType`.
- `src/config/`, `src/include/config.hpp` — YAML-based config (PR #565). v1.0 may use libconfig++ for multi-GPU settings.
- `src/pipeline/`, `src/include/pipeline/` — refactors from PRs #564/#626/#628 removed DuckDB vocabulary types; `sirius_physical_operator` base class changes (PR #626).
- `src/expression_executor/` — new AST-based executor (PR #531) replacing old translation path.
- `src/op/scan/` — metadata scan task added (#571), hive partition columns (#570), row group pruning (#363).

### cuCascade dev-drift
- Current pin: 942c0bf (`Implement get_uncompressed_data_size_in_bytes()`)
- Target pin: origin/main = f47de0b (`Drop hard NVML link from cucascade (#104)`)
- Brings in: PR #96 (file-downgrade / `idisk_io_backend` / `io_backend_registry` / `disk_data_representation`), PR #100 (memory_space underflow), PR #103 (stream-sync on GPU representation destroy), PR #104 (NVML link drop)

### Preserved v1.0 patterns (must survive the port)
- Push-model task dispatch: pop task first, route by `preferred_device_id` (v1.0 Phase 02-01)
- `preferred_device_id` on both local_state + global_state (local wins)
- NUMA→GPU mapping via first GPU per NUMA node
- NUMA-aware downgrade via cucascade `any_memory_space_in_tier_with_preference`

</code_context>

<specifics>
## Specific Ideas

- **Cherry-pick order** should match v1.0 commit order (oldest first) to minimize intra-port conflicts. Start with `39ea7bd docs: map existing codebase` if it's not just planning docs, then proceed chronologically.
- **Submodule bump first** (as its own commit) before any code-carrying cherry-picks so the replayed commits compile against the new cucascade API.
- **Docs-only v1.0 commits** (`docs: …`, `docs(NN): …`) can be skipped or squashed — they reference v1.0 planning state that's already superseded by v1.1's `.planning/`.
- **Rebuild cadence** — build after every 3–5 code-carrying commits during the port, not after each; faster feedback than per-commit but catches breakage before it compounds.

</specifics>

<deferred>
## Deferred Ideas

- `cucascade_datasource` + kvikio removal → Phase 5
- FOUND-01 topology discovery, FOUND-04 single-GPU regression, FOUND-06 device-guard audit, CUCS-01 GPU↔GPU converter, CUCS-02 per-NUMA host allocator → Phase 6 (MGPU-01..05)
- P2P `cudaMemcpyPeerAsync`, adaptive scan partitioning → Phase 7 (MGPU-06..07)

</deferred>
