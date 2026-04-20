# Phase 4 Research: cuCascade Bump + v1.0 Re-integration

**Researched:** 2026-04-20
**Domain:** Git cherry-pick / rebase workflow over a 23-commit divergence with active API drift (sirius-native types, YAML config, DuckDB vocabulary removal) and a cucascade submodule bump (942c0bf → f47de0b).
**Overall confidence:** HIGH — every claim below is grounded in `git show`/`git log`/`git diff` output on the actual repository state. No training-data speculation.

## User Constraints (from CONTEXT.md)

### Locked Decisions

All implementation choices are at Claude's discretion — pure infrastructure phase. Guidance from CONTEXT.md:
- **Conflict strategy** — "Attempt auto-resolution, pause on hard conflicts." Resolve trivial conflicts (formatting, clear renames from `LogicalType` → `logical_type`, libconfig→YAML keyname substitutions, namespace moves). Pause on: semantic behavior changes, ambiguous type replacements, any conflict where both sides wrote substantive logic, changes to test expectations.
- **Commit strategy** — cherry-pick commits one-by-one in v1.0 order, preserving each commit's authorship and subject. Rewrite commit message only when conflict resolution substantively changes the diff (prefix with `(rebased)`).
- **Build command** — use `mcp__project-commands__run_command`, never `pixi run`/`make` directly (user rule).
- **Stream discipline** — any code being ported that still uses `rmm::cuda_stream_default` gets fixed now if trivial; otherwise flagged for HYG-02 in Phase 5.
- **Test gating** — multi-GPU tests that require N>1 GPUs may `WARN+return` on single-GPU hosts per v1.0 Catch2-v2 convention.

### Claude's Discretion

All Phase 4 implementation choices except the five constraints above. In practice: task decomposition, batch vs one-by-one cherry-pick calls within a commit group, and whether to squash docs-only v1.0 commits.

### Deferred Ideas (OUT OF SCOPE)

- `cucascade_datasource` + kvikio removal → Phase 5
- FOUND-01 topology discovery, FOUND-04 single-GPU regression, FOUND-06 device-guard audit, CUCS-01 GPU↔GPU converter, CUCS-02 per-NUMA host allocator → Phase 6 (MGPU-01..05)
- P2P `cudaMemcpyPeerAsync`, adaptive scan partitioning → Phase 7 (MGPU-06..07)

## Project Constraints (from CLAUDE.md)

- **Build command:** `mcp__project-commands__run_command` exclusively — never invoke `pixi run`/`make` directly.
- **No `rmm::cuda_stream_default`:** always plumb an explicit stream. If the v1.0 port reintroduces one, either fix it during cherry-pick or file it for Phase 5 HYG-02.
- **Feature branch:** already on `feature/single-node-multi-gpu2`; do not push directly to `dev`.
- **Super Sirius only:** multi-GPU work is `namespace sirius`; the legacy `namespace duckdb` path is not touched.
- **Pre-commit hooks:** `clang-format`, `black`, `cmake-format`, `codespell` will run on every commit — expect whitespace fixups on conflict resolution.
- **Loading library context:** before any significant code change, run `/module-context <task>`. For this phase the relevant modules are cucascade (memory, data, disk_io) and duckdb (pipeline integration).

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| **BUMP-01** | cucascade submodule pointer updated 942c0bf → f47de0b | §3 — one-commit submodule bump; `git -C cucascade checkout f47de0b && cd .. && git add cucascade`. |
| **BUMP-02** | Sirius builds against new cucascade surface (PR #96 headers available, PR #100/#103/#104 absorbed) | §3 — audit shows no Sirius-side call-site break from any of the four cucascade PRs; changes are additive (PR #96), internal (PR #100), behavior-preserving (PR #103), and link-only (PR #104, and Sirius never linked NVML). |
| **BUMP-03** | Pre-existing cucascade-integration tests (`[downgrade]`, `[reservation]`, `[converter]`) pass post-bump with no new flakes | §6 — validation runs the existing test_downgrade_executor.cpp and test_context.cpp suites 5× via mcp unit-tests command; flake threshold = 0 new failures. |
| **PORT-01** | 23 v1.0 commits applied (or effect thereof) on top of current `dev` with clean compilation | §1 + §5 — 10 code-carrying commits + 9 docs-only + 5 merges; docs-only are skipped; merges are squashed; code-carrying are cherry-picked in chronological order. |
| **PORT-02** | Re-integrated code compiles against sirius-native types (`logical_type`/`type_id`, PR #643) — no residual DuckDB vocabulary types | §2 — v1.0 touched files have all been moved to sirius-native types on dev; any `LogicalType::INTEGER` etc. in the cherry-picks must be rewritten. Grep-gate: `grep -rnE 'LogicalType::(INTEGER\|BIGINT\|VARCHAR)' src/` = 0 hits in ported files. |
| **PORT-03** | Multi-GPU runtime settings read from YAML (PR #565), not libconfig++ | §2 — v1.0 did NOT add new libconfig++ keys; it reused existing hw_topology + memory_space_config paths that are already YAML on dev. Expected to be a near-no-op; grep-gate: `grep -rn 'libconfig' src/` = 0 hits. |
| **PORT-04** | Push-model task dispatch + `preferred_device_id` plumbing preserved | §1 (commits 59bc284, dd9264b) + §2 (conflict analysis on `pipeline_executor.cpp`, `gpu_pipeline_task.hpp`, `sirius_pipeline_task_states.hpp`, `task_creator.cpp`). |
| **PORT-05** | Existing multi-GPU test suites pass: `multi_gpu_foundation`, downgrade NUMA tests, data-locality tests | §6 — exact Catch2 tags enumerated; WARN+return convention for `[.]`-prefixed hidden tests on single-GPU hosts. |

---

## 1. The 23 v1.0 Commits to Replay

Source branch: `refs/remotes/felipe-ssh/feature/multi-gpu-execution` (tip = `0d99cde`).
Classification derived from `git show --name-only` on each commit.

| # | SHA | Subject | Type | Files | Code | Planning-only? | Replay? |
|---|-----|---------|------|-------|------|----------------|---------|
| 1 | `39ea7bd` | docs: map existing codebase | docs | 7 | 0 | YES (codebase docs, superseded by v1.1) | **SKIP** |
| 2 | `3777645` | test(01-02): multi-GPU foundation validation tests | code (test) | 1 | 1 | no | **REPLAY** — adds `[multi_gpu_foundation]` tests in `test/cpp/config/test_context.cpp` |
| 3 | `dd86dd0` | feat(01-01): NUMA-aware downgrade, multi-device sync, P2P enablement | code | 5 | 5 | no | **REPLAY** — core behavior: `downgrade_executor`/`downgrade_task` NUMA locality + `sirius_context` P2P loop |
| 4 | `85fe6f5` | Merge branch 'worktree-agent-a866ffbf' | merge | 0 | 0 | no | **SKIP** (empty merge) |
| 5 | `cf5624f` | Merge branch 'worktree-agent-aa0fe554' | merge | 0 | 0 | no | **SKIP** (empty merge) |
| 6 | `c5a3d8e` | test(01-03): NUMA-aware downgrade and GPU-to-GPU transfer tests | code (test) | 2 | 2 | no | **REPLAY** — adds `[downgrade][numa_aware_downgrade]` + `[.][multi_gpu_transfer]` tests |
| 7 | `de091c8` | docs(01-03): complete NUMA-aware downgrade tests plan | docs | 4 | 0 | YES | **SKIP** |
| 8 | `5b16c8e` | docs(02): create phase 2 plans | docs | 3 | 0 | YES | **SKIP** |
| 9 | `2752849` | fix(02): revise plans based on checker feedback | docs | 2 | 0 | YES (under `.claude/worktrees/`) | **SKIP** |
| 10 | `59bc284` | feat(02-01): add preferred_device_id + compute locality score | code | 5 | 5 | no | **REPLAY** — core behavior: `preferred_device_id` on local+global state, locality score in `task_creator` |
| 11 | `dd9264b` | feat(02-01): management_eventloop routes tasks by preferred_device_id | code | 2 | 2 | no | **REPLAY** — push-model dispatch in `pipeline_executor.cpp` + `gpu_pipeline_executor.cpp` |
| 12 | `81d4c75` | docs(02-01): complete plan | docs | 4 | 0 | YES | **SKIP** |
| 13 | `e1dab76` | Merge branch 'worktree-agent-a6bfe8c4' | merge | 1 | 1 | no (touches `src/sirius_context.cpp`) | **INSPECT** — merge recorded a planning conflict resolution; if the `src/sirius_context.cpp` change is already in #10, skip; otherwise include its delta in the cherry-pick for #10. |
| 14 | `6827d4d` | resolve merge conflicts after plan 02-01 | docs+fixup | 4 | 0 | YES (touches `.planning/`, `.gitignore`, `CLAUDE.md` only) | **SKIP** — CLAUDE.md changes are v1.0-planning-only |
| 15 | `7f18e66` | feat(02-02): distribute scan batches across GPUs | code | 2 | 2 | no | **REPLAY** — core behavior: `select_target_gpu` in `duckdb_scan_executor` |
| 16 | `2e6ba26` | test(02-02): integration tests for data-locality | code (test) | 2 | 2 | no | **REPLAY** — adds `[data_locality]` + `[.][data_locality][multi_gpu]` tests in `test/cpp/integration/test_gpu_execution_locality.cpp` + CMakeLists wiring |
| 17 | `83fb5c4` | docs(02-02): complete plan | docs | 1 | 0 | YES | **SKIP** |
| 18 | `3b3cd89` | Merge branch 'worktree-agent-ac2a25ae' | merge | 0 | 0 | no | **SKIP** (empty merge) |
| 19 | `98d94a2` | docs(03): create phase plan | docs | 3 | 0 | YES | **SKIP** |
| 20 | `ec2399e` | test(03-01): NUMA downgrade ordering tests | code (test) | 1 | 1 | no | **REPLAY** — extends `test_downgrade_executor.cpp` with NUMA-ordering tests |
| 21 | `5585a0c` | docs(03-01): complete plan | docs | 3 | 0 | YES | **SKIP** |
| 22 | `9d53259` | Merge branch 'worktree-agent-a6384282' | merge | 0 | 0 | no | **SKIP** (empty merge) |
| 23 | `0d99cde` | test(03-01): MEM-04 P2P + MEM-05 scan distribution tests | code (test) | 1 | 1 | no | **REPLAY** — further extends `test_downgrade_executor.cpp` with P2P+scan-distribution tests |

**Summary:**
- **10 code-carrying commits to replay** (all `test/` or `src/` with real code deltas): `3777645`, `dd86dd0`, `c5a3d8e`, `59bc284`, `dd9264b`, `7f18e66`, `2e6ba26`, `ec2399e`, `0d99cde`. Plus `e1dab76` which needs inspection (likely a 1-file merge-resolution fixup already covered by `59bc284`).
- **9 docs-only commits to skip** — all `.planning/`, `.claude/worktrees/`, `CLAUDE.md` changes. Zero value to replay; content is superseded by v1.1 planning.
- **4 empty merge commits to skip** — they add no code.
- **Net replay surface: 17 files touched** (see §2 table).

**Union of code files touched by the 10 replay commits** (the conflict surface):
```
CMakeLists.txt
src/creator/task_creator.cpp
src/downgrade/downgrade_executor.cpp
src/downgrade/downgrade_task.cpp
src/include/creator/task_creator.hpp
src/include/downgrade/downgrade_executor.hpp
src/include/downgrade/downgrade_task.hpp
src/include/op/scan/duckdb_scan_executor.hpp
src/include/pipeline/gpu_pipeline_task.hpp
src/include/pipeline/sirius_pipeline_task_states.hpp
src/op/scan/duckdb_scan_executor.cpp
src/pipeline/gpu_pipeline_executor.cpp
src/pipeline/pipeline_executor.cpp
src/sirius_context.cpp
test/cpp/config/test_context.cpp
test/cpp/downgrade/test_downgrade_executor.cpp
test/cpp/integration/test_gpu_execution_locality.cpp (NEW file — no conflict possible)
```

---

## 2. Dev Drift Hotspots (the 47 commits)

`git log --oneline refs/remotes/felipe-ssh/feature/multi-gpu-execution..dev` returned 47 commits. For each v1.0-touched file, `git log --oneline <branch-point>..dev -- <file>` enumerates the dev commits that will collide.

### Per-file conflict classification

| v1.0 file | dev commits that touch it | Conflict difficulty | Reason |
|-----------|---------------------------|---------------------|--------|
| `src/sirius_context.cpp` | #565 (YAML config), #579 (Downgrade request) | **HARD (structural)** | v1.0 adds 56 lines (P2P loop, gpu_numa_node pass-through). dev reshapes 137 lines (+90 / −47) via PRs #565 + #579, including *changing the `downgrade_executor` constructor signature that v1.0 is passing `gpu_numa_node` into*. Conflicts on downgrade_executor construction block + on config-topology plumbing. Requires **manual composition**: keep v1.0's gpu_numa_node passthrough but route it through dev's new `downgrade_executor_config` struct. |
| `src/include/downgrade/downgrade_executor.hpp` | #579 (Downgrade request) | **HARD (structural)** | v1.0 adds `std::optional<int> gpu_numa_node` as last ctor param and inherits from `itask_executor`. Dev (PR #579) removes the `itask_executor` base class, switches first param from `exec::thread_pool_config` to `exec::downgrade_executor_config`, and adds a `downgrade_request` queue architecture. v1.0's diff **applies to a class shape that no longer exists on dev**. Must be re-authored: add `gpu_numa_node` as a field on `downgrade_executor_config` (dev's new config struct), then plumb into the downgrade-task dispatch path (which also changed). |
| `src/include/downgrade/downgrade_task.hpp` | #579 (Downgrade request) | **CRITICAL (rewrite)** | v1.0 preserves a `class downgrade_task : public itask` with `_global_state`/`_local_state`. Dev (PR #579) **replaced the entire class with a POD `struct downgrade_task { ... }`**. v1.0's diff is against a type that does not exist on dev. Must re-author: port the NUMA preference (`any_memory_space_in_tier_with_preference`) logic to whatever call-site now invokes the downgrade — likely `downgrade_executor::run_downgrade_pass`. |
| `src/downgrade/downgrade_executor.cpp` | #579 | **HARD (structural)** | Matches the `.hpp` story — dev rewrote this file (385 lines, +promise/future/request-queue). The 8-line v1.0 diff (NUMA param passthrough) must be hand-translated to dev's new shape. |
| `src/downgrade/downgrade_task.cpp` | #579 | **CRITICAL (rewrite)** | Same as `.hpp`. Target may not exist — the 11-line v1.0 diff targets a class that was deleted. Translate the `any_memory_space_in_tier_with_preference` strategy usage to dev's new downgrade-executor flow. |
| `src/creator/task_creator.cpp` | #537 (cpu_source_task fix), #573 (operator_data refactor), #363 (row group pruning), #590 (Q21 deadlock) | **MODERATE (line-level + 1 semantic)** | v1.0 adds 77 lines (locality score computation, NUMA-to-GPU mapping). dev adds 44 lines across 4 PRs. Files overlap but at different regions. Likely text-level conflict resolvable by hunk-level merging, *except* for #573 (operator_data refactor) which may have moved the locality-score insertion point. |
| `src/include/creator/task_creator.hpp` | (same set as .cpp) | **TRIVIAL (line-level)** | Header changes are small (+11 lines v1.0, +~10 dev). Constructor signature compatible (both take `const system_topology_info*` param that already existed). |
| `src/pipeline/pipeline_executor.cpp` | #537, #579 | **MODERATE** | v1.0 rewrites the management_eventloop routing (30 lines +19, -11). dev changes task-request plumbing via #579. Both touch the push/pull loop. Resolution: keep v1.0's `preferred_device_id` routing logic but anchor it to dev's new task-request shape. |
| `src/pipeline/gpu_pipeline_executor.cpp` | #579, #573 | **MODERATE** | v1.0 removes a `task_request_publisher.send()` call (7 lines, -5). dev refactored the surrounding region in #573 (operator_data). Surface conflict; semantic intent (don't send pull signal for GPU tasks) survives. |
| `src/include/pipeline/gpu_pipeline_task.hpp` | #564, #628, #626, #643 (sirius-native types), #561 (Sirius exceptions), #540, #590, #531, #4e44ebe, #596, #092bcf5 | **MODERATE (many tiny conflicts)** | v1.0 adds 35 lines (`get_preferred_device_id()` methods). dev has had 11 distinct refactors on this file. Each dev refactor likely moved the header around but not the slots v1.0 adds to. Expect whitespace-only conflicts from clang-format + minor location drift. |
| `src/include/pipeline/sirius_pipeline_task_states.hpp` | (same set as gpu_pipeline_task.hpp) | **MODERATE** | v1.0 adds 19 lines (preferred_device_id getter/setter on both local and global state). Similar to the task header — many dev touches, most orthogonal. |
| `src/op/scan/duckdb_scan_executor.cpp` | #537, #570 (hive partitions), #643 (sirius-native types), #619, #571 (metadata scan), #564, #573, #363 (row group pruning) | **MODERATE** | v1.0 adds 62 lines (`select_target_gpu`, weighted distribution). dev has had 8 refactors. Most touch different methods (scan creation vs. hive-partition filtering vs. row-group pruning). v1.0's new method likely lands cleanly; existing method modifications may drift. |
| `src/include/op/scan/duckdb_scan_executor.hpp` | (same set as .cpp, fewer) | **TRIVIAL** | +15 lines header (the `select_target_gpu` declaration + a memory-space vector). Low-collision. |
| `test/cpp/config/test_context.cpp` | #565 (YAML config) | **MODERATE (test-rewrite)** | v1.0 adds `[multi_gpu_foundation]` tests. dev's #565 rewrote this file for YAML config loading. File still exists but test fixtures (env var setup, config file paths) changed. v1.0's new tests must be adapted to use the YAML test-config pattern. |
| `test/cpp/downgrade/test_downgrade_executor.cpp` | #579 (Downgrade request) | **HARD** | v1.0 adds ~3 separate sets of tests (01-03, 03-01 ordering, 03-01 P2P+MEM-05) to this file. dev's #579 rewrote the downgrade framework tests. New v1.0 tests target the **old** `downgrade_task_global_state`/`itask` class that no longer exists — these tests will not compile. Must be ported to the new `downgrade_request` + POD `downgrade_task` shape. |
| `test/cpp/integration/test_gpu_execution_locality.cpp` | **NONE** (new file in v1.0) | **NONE** | File does not exist on dev. Cherry-pick is a pure add. Only verification: does the test compile against sirius-native types? (The test includes `pipeline/gpu_pipeline_task.hpp` and `sirius_pipeline_task_states.hpp` — if those headers on dev use `logical_type` / `type_id`, the test body probably still works because it only sets `preferred_device_id`, which is primitive-typed.) |
| `CMakeLists.txt` | 14 dev commits | **MODERATE** | v1.0's `2e6ba26` adds `test/cpp/integration/test_gpu_execution_locality.cpp` to `TEST_SOURCES`. dev has had 14 CMakeLists touches. Most likely conflict: the `TEST_SOURCES` list grew in dev; insert line may drift. Trivial manual resolution. |

### Legend

- **TRIVIAL** — line-level conflict from whitespace, formatting, or location drift. Auto-resolvable.
- **MODERATE** — surface conflicts that require human-legible hunk selection. One or two files per commit.
- **HARD (structural)** — API shape changed on one side; v1.0 patch targets a type/signature that has evolved. Requires re-authoring the diff against the new shape.
- **CRITICAL (rewrite)** — target type/class was deleted or fundamentally rewritten; v1.0 diff has no valid anchor. Requires translating *intent* rather than cherry-picking.

### Dev PR landmines

The four dev PRs that matter most:
1. **PR #579 "Downgrade request"** — rewrote `downgrade_executor`, `downgrade_task`, and the downgrade test file. Touches 4 of the v1.0-port's 17 files, 3 of them CRITICAL. **This is the single largest risk in Phase 4.**
2. **PR #565 "libconfig++ → YAML"** — rewrote `src/sirius_context.cpp` and `test/cpp/config/test_context.cpp`. v1.0 added to both. Moderate conflict; not structural.
3. **PR #573 "operator_data refactor"** — touched `gpu_pipeline_executor.cpp` and `src/op/scan/duckdb_scan_executor.cpp`. Moderate.
4. **PR #626 / #628 / #564 "DuckDB vocabulary removal"** — touched all the `src/pipeline/` and `src/include/pipeline/` headers. Mostly whitespace/namespace drift.

### Surprise-audit: PRs that look scary but don't actually collide

| Dev PR | Touches v1.0 files? | Impact |
|--------|---------------------|--------|
| #643 sirius-native types (`logical_type`/`type_id`) | Yes (pipeline headers, scan) | **Low** — v1.0 code in these files doesn't construct `LogicalType` directly; it uses `preferred_device_id` (int) and `memory_space*`. Transitively safe. |
| #531 AST expression executor | Pipeline headers only | **Low** — v1.0 doesn't touch expression evaluation. |
| #570 hive partition columns | `duckdb_scan_executor.cpp` | **Low** — v1.0's `select_target_gpu` is orthogonal to partition filtering. |
| #363 row group pruning | `duckdb_scan_executor.cpp`, `task_creator.cpp` | **Low-Moderate** — row-group pruning and GPU selection are parallel concerns; may have whitespace conflicts around the materialization loop. |
| #594 libcudf 26.04 stable | (transitive) | **Nil** — happens at cudf header level; v1.0 doesn't hit changed cudf APIs. |

---

## 3. cuCascade Bump Impact

Current pin: `942c0bf` ("Implement get_uncompressed_data_size_in_bytes()").
Target pin: `f47de0b` ("Drop hard NVML link from cucascade (#104)") = `origin/main`.

Seven commits in range:

| SHA | PR | Sirius impact |
|-----|----|---------------|
| `d4f9050` | #95 — Fast path for same-device GPU-to-GPU conversion | **Nil (silent speedup)**. Optimization inside `representation_converter`. Sirius uses the converter via `register_parquet_converters` → unchanged. |
| `2f4d4f0` | #101 — Bump libcudf stable dependency to 26.04 | **Nil**. Matches dev's own #594 libcudf bump. |
| `0b3f49e` | #100 — Fix underflow in 3 `memory_space` APIs | **Nil (bugfix)**. Internal to `memory_space::reserve` path. |
| `9833849` | #96 — File downgrade (adds `idisk_io_backend`, `io_backend_registry`, `disk_data_representation`) | **Strictly additive**. Sirius does not call these in Phase 4. This is the *reason* for the bump: Phase 5 needs these headers available. Verified no Sirius file includes them today (`grep -rn idisk_io_backend src/` = 0 hits). |
| `8f51996` | #102 — Bump `benchmark-action/github-action-benchmark` | **Nil** — CI-only. |
| `5e3a637` | #103 — Sync stream before destroying GPU representation in `data_batch::convert_to` | **Latent behavior change**. Adds a `stream.synchronize()` inside `convert_to` when GPU is involved. Sirius calls `data_batch::convert_to` (found 30+ sites in `src/op/*.cpp`). The sync is **load-bearing for correctness** (prevents UAF on stream-ordered GPU work) but adds a per-conversion stream sync on the caller's stream. **Possible latent throughput delta but no API break.** Mitigation: none needed; if a perf regression shows up, file upstream. |
| `f47de0b` | #104 — Drop hard NVML link | **Nil for Sirius**. Verified Sirius does NOT link NVML explicitly (`grep -nE 'nvml\|NVML\|NVIDIA_ML' CMakeLists.txt` = 0 hits). The drop changes cucascade's `CUCASCADE_PUBLIC_LINK_LIBS` from `{rmm, cudf, cudart, nvml, threads, numa}` to `{rmm, cudf, cudart, threads, numa}`. Sirius transitively linked NVML through cucascade — that transitive link goes away — but nothing in Sirius actually called NVML. |

### `memory_space.hpp` new surface

PR #96 added (additive only):
- New constructor `memory_space(const disk_memory_space_config&, shared_ptr<idisk_io_backend>)`.
- Two getters `get_disk_mount_path()` and `get_io_backend()` that throw on non-DISK tiers.
- Internal `_io_backend` member (null on GPU/HOST).

**Verified Sirius impact:** zero. Sirius constructs `memory_space` only via `cucascade::memory::reservation_manager_configurator::build()` which uses the pre-existing single-arg constructor. Sirius never calls `get_disk_mount_path` or `get_io_backend`. No rebuild warnings expected.

### `representation_converter.hpp` new surface

Only a doc-comment change ("Registers converters between all supported representation types (GPU, HOST, DISK)"). The function signature of `register_builtin_converters` is unchanged. **Zero Sirius impact.**

### `data_batch.hpp` behavior change

`data_batch::convert_to` inline implementation now:
1. Moves old `_data` into a local before reassigning.
2. Computes `needs_sync` if either source or target tier is GPU.
3. Releases the lock early (via `unique_lock`).
4. Calls `stream.synchronize()` before `old_representation` destructs.

**Sirius impact:** Sirius calls `data_batch::convert_to` across many operators (aggregation, order, grouped-aggregate-merge, top-n). Each call now synchronizes the caller's stream. This is **correct** behavior (prevents use-after-free on async GPU reads inside custom converters). It may introduce a latent perf delta on tight converter loops, but measuring that is out of Phase 4 scope — Phase 5 benchmarks will catch it.

### NVML link drop — second-order risk

If any test or build script references `CUDA::nvml` assuming it was in cucascade's public link set, it will break. `grep -nE 'CUDA::nvml\|NVIDIA_ML' CMakeLists.txt test/*.cmake 2>/dev/null` returns nothing. **Safe.**

### CMakeLists.txt — will the bump require Sirius CMake changes?

No. Sirius links `cuCascade::cucascade` (line 321 of CMakeLists.txt) which re-exports whatever cucascade's public link set is. The set got smaller; the link still resolves.

---

## 4. Build + Test Invocation

### MCP commands (read from `.ai-helper/commands.yaml`)

| Command | What it runs | Use for |
|---------|--------------|---------|
| `mcp__project-commands__run_command("build")` | `make -j$(nproc)` | Incremental build. Default during port. |
| `mcp__project-commands__run_command("unit-tests")` | `./build/release/extension/sirius/test/cpp/sirius_unittest --abort` | Full C++ unit suite with abort-on-first-fail. |
| `mcp__project-commands__run_command("clean")` | `rm -rf build` | Only if the build system gets into a confused state (e.g., CMakeLists conflict that requires reconfigure). |
| `mcp__project-commands__run_command("cmake-release")` | `cmake --preset release` | Reconfigure after CMakeLists.txt changes in cherry-pick #15 (`2e6ba26`). |
| `mcp__project-commands__run_command("pre-commit")` | `pre-commit run --all-files` | Before each commit to catch clang-format, codespell, black issues introduced by conflict resolution. |

### Running specific test tags

The MCP wrapper doesn't expose tag-filtered test runs directly. Two options:
1. **Run the whole suite** — simplest; finishes in ~5 minutes; `--abort` stops at first failure.
2. **Run the binary directly inside a separate bash invocation** via `mcp__project-commands__run_command` for a custom command — NOT supported by current `.ai-helper/commands.yaml` without extension.

Phase 4 recommendation: run the full suite after each batch of cherry-picks. For iterative debugging during conflict resolution, the **build** command is the fast feedback loop; the **unit-tests** command is the gate.

### Test tags that matter for PORT-05

Derived from `git show <v1.0-sha>:<test-file>`:

| Tag | From commit | Test file | Notes |
|-----|-------------|-----------|-------|
| `[multi_gpu_foundation]` | `3777645` | `test/cpp/config/test_context.cpp` | topology_discovery, reservation_manager_configurator, memory_manager, converter_registry-has-GPU-GPU |
| `[.][multi_gpu_foundation]` | `3777645` | `test/cpp/config/test_context.cpp` | **Hidden** (requires N≥2 GPUs); `multi_gpu_config_two_gpus` TEST_CASE |
| `[downgrade_executor]` | `c5a3d8e` | `test/cpp/downgrade/test_downgrade_executor.cpp` | Basic downgrade-executor lifecycle |
| `[downgrade][numa_aware_downgrade]` | `c5a3d8e`, `ec2399e` | `test/cpp/downgrade/test_downgrade_executor.cpp` | `numa_aware_downgrade_executor_passes_numa_node`, `downgrade_executor_default_numa_node_is_nullopt` |
| `[.][multi_gpu_transfer]` | `c5a3d8e` | `test/cpp/downgrade/test_downgrade_executor.cpp` | **Hidden** (N≥2 GPUs); `gpu_to_gpu_transfer_via_converter` |
| `[data_locality]` | `2e6ba26` | `test/cpp/integration/test_gpu_execution_locality.cpp` | preferred_device_id defaults, precedence, NUMA-to-GPU mapping, locality-score computation, proportional distribution |
| `[.][data_locality][multi_gpu]` | `2e6ba26` | `test/cpp/integration/test_gpu_execution_locality.cpp` | **Hidden** (N≥2 GPUs); `scan batches distributed across multiple GPUs` |

**Correction of ROADMAP.md §Phase 4 Success Criteria:** the roadmap cites `[test_gpu_execution_locality]` as the Catch2 tag — this is **wrong**. The actual tag is `[data_locality]` (the test file is named `test_gpu_execution_locality.cpp` but its TEST_CASEs all use `[data_locality]`). Planner should use `[data_locality]` in phase gate checks.

### Catch2 version (for `WARN+return` convention)

Sirius links DuckDB's bundled `duckdb/third_party/catch` (line 411 of CMakeLists.txt). This is **Catch2 v2** — it does NOT have a native `SKIP()` macro. The v1.0 convention for GPU-count-gated tests is:

```cpp
TEST_CASE("requires_two_gpus", "[.][multi_gpu_foundation]")
{
  int count = 0;
  cudaGetDeviceCount(&count);
  if (count < 2) {
    WARN("skipping: requires >=2 GPUs");
    return;
  }
  // … real test body …
}
```

The `[.]` tag-prefix means "hidden by default" — Catch2 v2 does not run these unless explicitly selected with `./sirius_unittest "[multi_gpu_foundation]"` (hidden tags require explicit inclusion). On a single-GPU dev host the test can be manually triggered and will `WARN+return` cleanly.

### Incremental rebuild strategy during cherry-pick

- **After each code-carrying cherry-pick** — run `build`. ~30-90s incremental.
- **After a full group of 3-5 code-carrying picks** — run `unit-tests`. ~5min.
- **After submodule bump (first commit)** — full rebuild expected because cucascade is a build dependency. ~10-20min depending on cold/hot caches.
- **After touching CMakeLists.txt (pick #15 `2e6ba26`)** — CMake may auto-reconfigure; if not, run `cmake-release`.

---

## 5. Recommended Replay Strategy

### Overall shape

1. **First commit of the phase = submodule bump alone.** Cherry-pick nothing else. This gives us a green baseline and isolates BUMP-01/BUMP-02/BUMP-03 from the port conflicts.
2. **Full build + full unit tests after the bump**, before touching any code. This proves the bump by itself works. If it doesn't, rollback is trivial (`git reset --hard HEAD~1`).
3. **Then replay the 10 code-carrying commits in chronological order**, using `git cherry-pick` with conflict resolution per commit. Docs-only and empty-merge commits are **skipped** (not cherry-picked at all).
4. **Checkpoint builds every 2-3 cherry-picks**; full unit tests after each group of 3-5.
5. **Final gate:** full build + full unit tests + PORT-02/03 grep checks.

### Why cherry-pick one-by-one, not rebase or batched cherry-pick -n

Three options evaluated:

| Strategy | Pros | Cons | Verdict |
|----------|------|------|---------|
| `git rebase dev refs/…/feature/multi-gpu-execution` on a copy branch | Attempts to reuse commits with their authorship | Fails on first conflict; restart-from-fail is awkward; can't easily skip docs-only picks mid-rebase | **Rejected** — 47-commit drift + structural PR #579 conflicts will blow up immediately |
| `git cherry-pick -n A B C …` then squash | Lets us fold related commits into one "integration" commit | Loses commit-level authorship; harder to isolate which commit caused a build break | **Rejected** — user preference is to preserve per-commit attribution |
| `git cherry-pick A; resolve; commit; cherry-pick B; …` (one-by-one) | Per-commit isolation; preserves authorship; easy rollback; clear audit trail | Slowest; 10 manual conflict-resolution rounds | **Accepted** — matches user's "pause on hard conflicts" preference and preserves attribution |

### Handling HARD / CRITICAL conflicts

For `downgrade_executor.hpp/cpp` and `downgrade_task.hpp/cpp` (the PR #579 collision), `git cherry-pick` will likely produce conflict markers against a file shape that cannot be mechanically resolved. **Protocol:**

1. Accept the cherry-pick's conflict markers so the commit is started.
2. Resolve by **authoring a new diff that expresses the v1.0 intent (NUMA locality preference in downgrade)** against dev's new shape (`downgrade_request`, POD `downgrade_task`). Specifically:
   - Add a `std::optional<int> preferred_numa_node` field on `exec::downgrade_executor_config` (dev's new config struct).
   - In `downgrade_executor::process_requests()` (or wherever dev's new file dispatches downgrade work), route through cucascade's `any_memory_space_in_tier_with_preference` strategy, populating preference from that config.
   - In `SiriusContext::initialize()`, populate the `preferred_numa_node` from `config_.get_hw_topology().gpus[i].numa_node` — matching v1.0's intent.
3. Commit with a message like `feat(01-01): NUMA-aware downgrade (rebased onto dev #579)` — the `(rebased)` suffix signals that the diff was hand-authored to fit dev's shape while preserving v1.0 intent.

**Pause point for human confirmation:** before committing these re-authored changes, surface them via the structured return (or via `/gsd:verify-work` checkpoint) so the user can confirm the behavior translation.

### Handling MODERATE conflicts

Standard `git cherry-pick` resolution via `git mergetool` or hand-editing the conflict hunks. Common pattern:
- Keep dev's surrounding code (sirius-native types, updated method signatures).
- Insert v1.0's new logic (new methods, new fields) in the appropriate location.
- Run `pre-commit run` on the touched files to catch formatting before committing.

### Handling TRIVIAL conflicts

Resolve by selecting both sides where they don't overlap, or keeping dev's formatting over v1.0's when the only conflict is whitespace. No special protocol — `git add` + `git cherry-pick --continue`.

### Commit ordering

Strict chronological (v1.0 commit-date order): `3777645 → dd86dd0 → c5a3d8e → 59bc284 → dd9264b → 7f18e66 → 2e6ba26 → ec2399e → 0d99cde`. This ordering:
1. Matches the logical dependency chain (test adds types before feat adds uses).
2. Minimizes intra-port conflicts (later picks build on earlier picks' introduced fields).
3. Preserves v1.0's commit-date ordering so the log reads sensibly.

Note: `e1dab76` (merge-fixup) is **inspected**, not auto-cherry-picked. If its `src/sirius_context.cpp` delta is already present in `59bc284` (the preceding feat), skip it; otherwise add its 1-file delta as an amendment to `59bc284` or a new small commit.

### Squashing the 10 code-carrying commits into 3 thematic commits — rejected

Evaluated: fold into "Foundation" (dd86dd0), "Scheduling" (59bc284, dd9264b, 7f18e66), "Tests" (3777645, c5a3d8e, 2e6ba26, ec2399e, 0d99cde). **Rejected** because:
1. The user's CONTEXT.md decision says "cherry-pick commits one-by-one in v1.0 order, preserving each commit's authorship".
2. Thematic squashing would collapse 10 small, reversible diffs into 3 large, harder-to-bisect diffs.
3. The audit trail of "which v1.0 commit did what" is lost.

---

## 6. Validation Strategy

### Per-requirement success check

| Req | Validation command | Pass criterion |
|-----|-------------------|----------------|
| **BUMP-01** | `git -C cucascade rev-parse HEAD` | Output = `f47de0bb7bcaddd55081a9c4bc584627532d1ef9` |
| **BUMP-02** | `mcp__project-commands__run_command("build")` | Exit 0, no new compiler warnings in `src/sirius_extension.cpp`, `src/sirius_context.cpp`, `src/data/host_parquet_representation_converters.cpp` (these are the main cucascade-using files) |
| **BUMP-03** | `mcp__project-commands__run_command("unit-tests")` × 5 | All runs pass; zero new flakes; specifically `[downgrade_executor]`, `[reservation]`, `[converter]` tags green each time |
| **PORT-01** | `git log --oneline dev..HEAD` | Shows 10 code-carrying cherry-picks + 1 submodule-bump commit (plus `e1dab76` fixup if included); build is green |
| **PORT-02** | `grep -rnE 'LogicalType::(INTEGER\|BIGINT\|VARCHAR)' src/` applied to the 17 touched files | 0 hits |
| **PORT-03** | `grep -rn 'libconfig' src/` | 0 hits |
| **PORT-04** | `grep -n 'preferred_device_id' src/include/pipeline/gpu_pipeline_task.hpp src/include/pipeline/sirius_pipeline_task_states.hpp src/creator/task_creator.cpp src/pipeline/pipeline_executor.cpp` | Each file contains the accessor/routing code (exact line count per-file: gpu_pipeline_task.hpp ~35 lines; sirius_pipeline_task_states.hpp ~19 lines; task_creator.cpp has `compute_data_locality_score`; pipeline_executor.cpp has the preferred-device routing loop) |
| **PORT-05** | Full unit-test suite pass on single-GPU host | `[multi_gpu_foundation]` tests green (non-`[.]` ones); `[.]` hidden tests `WARN+return` when invoked; `[data_locality]` green; `[downgrade][numa_aware_downgrade]` green |

### Checkpoint cadence

- **After submodule bump** (first commit of the phase): build + unit-tests. Must be green before proceeding.
- **After each code-carrying cherry-pick**: build only. Must compile; unit-test deferred.
- **After every 3 code-carrying cherry-picks**: full unit-tests with `--abort`. First-failure diagnosis while the change is fresh.
- **After the 10th code-carrying cherry-pick**: full unit-tests + all grep checks from the table above + `git log` audit.

### Phase-level exit gate

```bash
# 1. Build
mcp__project-commands__run_command("build")   # must succeed, no new warnings

# 2. Full unit tests
mcp__project-commands__run_command("unit-tests")   # must succeed

# 3. Structural grep checks
grep -rnE 'LogicalType::(INTEGER|BIGINT|VARCHAR)' src/   # must be empty
grep -rn 'libconfig' src/                                # must be empty
git -C cucascade rev-parse HEAD                          # must equal f47de0b
git log --oneline dev..HEAD | wc -l                      # must be >= 11 (1 bump + ≥10 ports)

# 4. Pre-commit clean
mcp__project-commands__run_command("pre-commit")        # must succeed
```

### Test-gating note (Catch2 v2 hidden tests)

On a single-GPU host (likely the dev box running this port), tests tagged `[.][multi_gpu_foundation]`, `[.][multi_gpu_transfer]`, `[.][data_locality][multi_gpu]` are **hidden by default** — the full `unit-tests` command will not run them. They must be invoked explicitly:

```
./build/release/extension/sirius/test/cpp/sirius_unittest "[multi_gpu_foundation]"
```

If invoked on a single-GPU host, they `WARN+return`, which Catch2 v2 counts as a pass. On a multi-GPU host they execute real logic. For Phase 4, hidden-test execution is **optional** — their mere presence + compile-passing is sufficient for PORT-05. Actual multi-GPU validation requires a 2+ GPU host and is done at final phase sign-off if such hardware is available.

---

## 7. Risk Register

| Risk | Likelihood | Impact | Rollback / Mitigation |
|------|-----------|--------|----------------------|
| **R1: Downgrade rewrite (PR #579) conflict is unresolvable by cherry-pick alone** | HIGH (certain) | HIGH | Re-author v1.0 NUMA intent as a new diff against dev's new shape. Commit with `(rebased)` marker. Pause for human review before committing. |
| **R2: Submodule bump breaks Sirius build in an unexpected way** | LOW | MEDIUM | Rollback = `git reset --hard HEAD~1` (the bump is a single commit). Inspect build log; if the break is in a cucascade header we missed (highly unlikely given §3 audit), file upstream bug and hold phase. |
| **R3: PR #565 YAML config does not expose `gpu_numa_node` / multi-GPU settings that v1.0 expects** | LOW | MEDIUM | Verified `numa_id` already exists on `gpu_memory_space_config`; `get_hw_topology()` already returns GPU/NUMA mapping. PORT-03 is likely a near-no-op because v1.0 never added libconfig-specific new keys — it used the existing topology discovery path. Mitigation: verify `sirius_config.cpp:34-65` keys cover what v1.0 consumes; if any key is missing, add it to the YAML reader. |
| **R4: test_gpu_execution_locality.cpp compiles but depends on symbols that the port hasn't yet added** | MEDIUM | LOW | Cherry-pick order guarantees `59bc284` (which adds `preferred_device_id` to `gpu_pipeline_task_local_state`) lands before `2e6ba26` (the test file). If the test still won't compile, the fix is a small include-path or namespace adjustment, not a design change. |
| **R5: PR #565's YAML `test_context.cpp` rewrite makes v1.0 `[multi_gpu_foundation]` tests use wrong fixtures** | MEDIUM | LOW | Translate v1.0 test env-var / config-file fixtures to dev's `SIRIUS_DISABLE` + YAML env pattern. Straightforward; the tests themselves are about topology and memory spaces, not config loading. |
| **R6: Test flake in `[downgrade]` due to PR #103's added `stream.synchronize()` inside `data_batch::convert_to`** | LOW | LOW | Re-run 5× as per BUMP-03. If still flaky, file upstream; sync is a correctness fix so removing it is not an option. |
| **R7: `pre-commit` (clang-format) aggressively rewrites cherry-picked code, making the diff unrecognizable** | MEDIUM | LOW | Accept it — project rule says clang-format on commit. The *semantic* diff is what matters for attribution; formatting is project policy. |
| **R8: CMakeLists.txt conflict on `2e6ba26` (adding test file) due to dev's many CMakeLists touches** | LOW | LOW | The v1.0 add is a single line in a TEST_SOURCES list. Trivial to hand-place. |
| **R9: `e1dab76` merge-fixup contains important NUMA logic that's not in `59bc284`** | LOW | MEDIUM | Inspect `git show e1dab76 -- src/sirius_context.cpp` — if non-empty delta, include as amendment to commit #10 or as its own small commit. |
| **R10: Full unit-tests timeout or deadlock due to interaction between v1.0 push-model dispatch and dev's new downgrade-request queue** | MEDIUM | MEDIUM | `--abort` on unit-tests aborts the process on first failure; deadlock would show as timeout. If observed, the fix is in the re-authored `downgrade_executor` (R1 work). |
| **R11: New compile warnings surface in unrelated Sirius files due to `-Werror` + cucascade header additions** | LOW | LOW | cucascade's `-Werror` config is separate from Sirius. Sirius warnings typically only appear from direct includes, which the audit shows are unchanged. If they do appear, suppress narrowly (not globally). |

### Highest-risk moment

The conflict-resolution for `downgrade_executor.{hpp,cpp}` + `downgrade_task.{hpp,cpp}` at cherry-pick #3 (`dd86dd0`). This is where the "pause on hard conflicts" user preference gets exercised. Expected time cost: 30-90 min of careful re-authoring + review. If this goes sideways, the whole phase is blocked on one resolution.

---

## 8. Open Questions

### OQ-1: Is `e1dab76` (merge-fixup) delta already in `59bc284`, or is it new?

`e1dab76` is a merge commit that touches `src/sirius_context.cpp`. `git show e1dab76 -- src/sirius_context.cpp` shows only the merge-resolution markers. Whether the resulting merge state is *new* content vs. the union of the two parents is unclear without actually running `git show` against both parents. **Recommendation:** during cherry-pick, run `git diff 59bc284 81d4c75 -- src/sirius_context.cpp` to see if that merge introduced any novel content; if yes, cherry-pick `e1dab76` as commit #10.5; if no, skip.

### OQ-2: Does PR #565's YAML config cover `gpu_numa_node` → multi-GPU NUMA mapping?

§3 confirms `gpu_memory_space_config` has a `numa_id` field read via `r.optional("numa_id", opt.numa_id)`. What's unclear: does `sirius_config` persist a runtime `gpu_numa_node` → used-by-downgrade-executor, or is the NUMA info always pulled fresh from `_hw_topology`? v1.0's `sirius_context.cpp` (lines 214-229) populates `gpu_numa_node` from `topo.gpus[dev_id].numa_node` — suggesting it's pulled fresh from topology, not read from the YAML file. If that's the pattern, PORT-03's YAML requirement is **already satisfied** by dev (PR #565 added no barrier). **Recommendation:** verify by reading `src/sirius_config.cpp:267-284` and `src/sirius_context.cpp:219` — if both use `hw_topology` directly, PORT-03 is a no-op modulo confirmation.

### OQ-3: Is PR #103's `stream.synchronize()` insertion going to show up as a measurable perf regression?

§3 flags this as a latent behavior change. No Sirius benchmarks are run in Phase 4. Phase 5 adds TPC-H SF10 regression tests (IO-10). **Recommendation:** defer to Phase 5; note the potential contribution if regression is observed.

### OQ-4: Should hidden `[.]`-prefixed tests be explicitly invoked during Phase 4 verification?

On single-GPU dev hosts they `WARN+return` (safe). On multi-GPU hosts they actually exercise multi-device code paths. Current hardware availability is unknown. **Recommendation:** planner should include an optional verification task "if N≥2 GPUs, run hidden multi_gpu tests explicitly; otherwise only compile-verify." This makes Phase 4 pass on any host but captures the extra signal where possible.

### OQ-5: Does the `catch.hpp` that Sirius uses have a `SKIP()` macro available via any override?

Sirius uses `duckdb/third_party/catch` which is Catch2 v2. Confirmed v1.0 convention is `WARN+return`. **Not an open question so much as a confirmed constraint** — documented here to flag: do NOT write `SKIP()` in ported tests; it won't compile. All v1.0 tests already follow this convention.

---

## Sources

### Primary (HIGH confidence — direct git/file inspection)

- `git log --oneline --reverse dev..refs/remotes/felipe-ssh/feature/multi-gpu-execution` — enumerated all 23 v1.0 commits.
- `git log --oneline refs/…/feature/multi-gpu-execution..dev` — 47 dev commits.
- `git show --name-only --format='' <sha>` — classified each v1.0 commit's file-touch surface.
- `git show --stat <sha>` — measured v1.0 diff sizes.
- `git log --oneline <branch-point>..dev -- <path>` — identified dev commits that touch v1.0 hotspot files.
- `git -C cucascade log --oneline 942c0bf..f47de0b` — 7 cucascade commits in bump range.
- `git -C cucascade diff --stat 942c0bf..f47de0b` — cucascade diff shape.
- `git -C cucascade diff 942c0bf..f47de0b -- <header>` — exact API surface changes.
- `git show dev:<path>` vs `git show <v1.0-sha>:<path>` — side-by-side file state comparison.
- Working-tree: `.mcp.json`, `.ai-helper/commands.yaml`, `.planning/config.json` — MCP/GSD config.
- Working-tree: `CMakeLists.txt` — link libraries, test wiring.
- `.planning/research/CUCASCADE-IO.md` — loaded via files_to_read; provides the independent f47de0b API reference.

### Secondary (MEDIUM confidence — inferred from cross-references)

- Current hw topology API stability: inferred from presence of identical `system_topology_info` references in both `dev` and `refs/…/feature/multi-gpu-execution` — so v1.0's usage translates directly.
- Catch2 v2 behavior (no `SKIP()`): inferred from DuckDB's bundled third_party catch path + CONTEXT.md user convention note.

### Tertiary (LOW confidence — nothing in this research)

None — all claims are grounded in HIGH-confidence direct evidence.

---

## Metadata

**Confidence breakdown:**
- 23-commit classification: **HIGH** — directly enumerated from git.
- Dev-drift hotspot map: **HIGH** for file lists; **MEDIUM** for difficulty ratings (rated by reading actual diffs + current file state, but actual conflict difficulty won't be fully known until cherry-pick attempts).
- cuCascade bump impact: **HIGH** — every claim backed by diff of cucascade headers + Sirius source grep.
- Test tags / gating: **HIGH** — read directly from v1.0 test files.
- Risk register: **MEDIUM** — R1 (downgrade rewrite) is essentially certain to be HARD; R3/R4 likelihoods are extrapolated from file evidence but not empirically validated.

**Research date:** 2026-04-20
**Valid until:** 2026-05-20 (30 days) — dev branch may add more commits in this window, re-validate if Phase 4 start slips significantly.
