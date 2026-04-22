---
phase: 08-multi-gpu-sql-pipeline-fix
plan: 04
subsystem: test-infra

tags: [catch2, test-fixture, multi-gpu, tpch-integration, shared-test-env, generate]

# Dependency graph
requires:
  - phase: 08-multi-gpu-sql-pipeline-fix
    plan: 02
    provides: "Sirius-side host->gpu converter override so TPC-H under num_gpus=2 can even reach the compare_gpu_vs_cpu oracle on the DuckDB ATTACH path (probe FAIL on host_parquet path still expected, handed off to 08-06)"
provides:
  - "integration-2gpu.yaml fixture (mirror of integration.yaml with num_gpus: 2) living alongside the 1-GPU default (which is NOT flipped)"
  - "g_integration_env_2gpu non-owning pointer + acquire_integration_env_for(int num_gpus) helper exposed from test/cpp/utils/sirius_test_env.{hpp,cpp}; returns nullptr on single-GPU hosts when num_gpus==2 so callers can WARN+return per Catch2 v2 convention"
  - "Second shared_test_env instance constructed (paused) in unittest.cpp main() so the TPC-H fixture can switch Sirius contexts between num_gpus=1 and num_gpus=2 per GENERATE iteration without editing the default integration.yaml"
  - "GPUExecutionFixtureBase::bind_env(num_gpus) / release_env() / compare_gpu_vs_cpu_for(num_gpus, query, tol) helpers + virtual setup_schema() hook so DuckDB and Parquet subclasses can re-establish views/attaches after each env swap"
  - "RUN_TPCH_MGPU(...) convenience macro at test_gpu_execution_tpch.cpp:3296 that expands to GENERATE(1,2) + CAPTURE(num_gpus) + compare_gpu_vs_cpu_for(...) with single-GPU WARN+return; applied mechanically to all 44 TPC-H TEST_CASEs (Q1..Q22 DuckDB + parquet)"
affects:
  - 08-05  # AUDIT-01/02/03 TEST_CASE can reuse acquire_integration_env_for(2) + the bind_env/release_env pattern
  - 08-06  # ship-gate will observe 2-GPU variant failures for Q-queries that still hit the open host_parquet converter bug; gate must treat those as expected until 08-06 lands

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Pattern: add a second shared_test_env for an alternate Sirius config (topology) alongside the primary one — both paused at startup; the TEST_CASE body resume()s the one it needs and pause()s it on destruction. At most one Sirius context is live at a time (the extension-lock invariant from the existing [shared_context]/[integration] listener). The testCaseStarting listener also pauses the 2-GPU env between tests as a safety net, even though TEST_CASE bodies own its lifecycle."
    - "Pattern: Catch2 v2 GENERATE(1, 2) inside a TEST_CASE body expands each TEST_CASE into two sections (1-GPU + 2-GPU). Combined with a fixture-side bind_env(num_gpus) + virtual setup_schema() hook, a single TEST_CASE can exercise multiple topology configurations without duplicating the query string or the TEST_CASE_METHOD boilerplate. CAPTURE(num_gpus) makes failure messages report which variant failed."
    - "Pattern: mechanical RUN_TPCH_MGPU(...) one-line macro substitution is the lowest-churn way to parameterize a large TEST_CASE cohort when every body has the identical shape compare_gpu_vs_cpu(QUERY[, TOL]); edit only the call-site token, leave the TEST_CASE_METHOD header + query string verbatim so review diff is minimal and git blame stays useful."
    - "Pattern: virtual setup_schema() hook on the fixture base re-runs subclass view/attach DDL on a fresh connection after each env swap — avoids hardcoding 'run schema only in ctor' which would leak the 1-GPU schema into the 2-GPU connection when bind_env(2) reassigns con."

key-files:
  created:
    - "test/cpp/integration/integration-2gpu.yaml (30 lines — identical to integration.yaml except topology.num_gpus: 2 and the leading comment flagging the sync discipline)"
    - ".planning/phases/08-multi-gpu-sql-pipeline-fix/08-04-SUMMARY.md (this file)"
  modified:
    - "test/cpp/utils/sirius_test_env.hpp (extern shared_test_env* g_integration_env_2gpu; + acquire_integration_env_for(int num_gpus) declaration)"
    - "test/cpp/utils/sirius_test_env.cpp (add <cuda_runtime.h> for cudaGetDeviceCount; define g_integration_env_2gpu and acquire_integration_env_for)"
    - "test/cpp/unittest.cpp (construct shared_test_env for integration-2gpu.yaml; paused at startup; listener pauses it between tests; cleared at shutdown)"
    - "test/cpp/integration/test_gpu_execution_tpch.cpp (fixture base: add bind_env/release_env/compare_gpu_vs_cpu_for/setup_schema; subclasses: hoist schema setup into setup_schema override; TPC-H section: add RUN_TPCH_MGPU macro and convert all 44 TEST_CASEs to it)"

key-decisions:
  - "Chose research-recommended Option A (Catch2 GENERATE inside a single TEST_CASE body) over Option B (TEMPLATE_TEST_CASE_METHOD) and Option C (duplicated TEST_CASE flavors). Rationale: Option A requires zero TEST_CASE_METHOD header edits, preserves test names + tags exactly, and the per-TEST_CASE body edit is a single mechanical substitution. Option B would require re-tagging all 358 TEST_CASEs in the file; Option C would double the code. Research 08-RESEARCH.md 'TEST-01' row explicitly recommends Option A."
  - "Introduced a convenience macro RUN_TPCH_MGPU(...) rather than expanding the three-line GENERATE+CAPTURE+if body inline at every TPC-H TEST_CASE. Rationale: 44 call sites, identical body shape, macro is a net code-shrinker. The macro is defined at the TPC-H section header with a 10-line doc comment explaining exactly what it expands to so reviewers don't need to chase the definition. Plan explicitly permitted macro-based approach (Task 3 step 5)."
  - "Added a virtual setup_schema() hook to the fixture base. Rationale: the DuckDB-subclass runs `ATTACH tpch... USE tpch` and the parquet-subclass runs 8 CREATE VIEW statements on `con`; when bind_env(num_gpus) reassigns `con` from a different env's make_connection(), the views/attach are tied to the OLD connection and must be re-established. Hoisting schema setup into an overridable method is cleaner than either (a) leaving it in the ctor and losing it on rebind, or (b) duplicating the setup code in bind_env per subclass."
  - "Did NOT hide the 2-GPU variant behind a [.] Catch2 tag. Rationale: AUDIT-03 (Plan 08-05) requires the 2-GPU variant to execute in the default unit-tests run (not opt-in). Accepting the doubling of TPC-H test count (44 -> 88 runs) is the explicit cost of the v1.2 coverage promise. Users on single-GPU hosts get WARN+return for free via cudaGetDeviceCount<2 check."
  - "Kept integration.yaml UNCHANGED. Rationale: user constraint in RESEARCH.md Locked Decisions explicitly locks 'no permanent flip of num_gpus'. integration-2gpu.yaml is a separate file mirroring every other key value (memory limits, thread counts, batch sizes, cache params) so a 1-GPU run and a 2-GPU run differ ONLY in topology.num_gpus — keeping other variables constant is a prerequisite for TEST-03's bitwise-identical result comparison across configs."
  - "Listener in unittest.cpp also pauses g_integration_env_2gpu between tests (belt-and-suspenders). Rationale: the 2-GPU env's active/paused lifecycle is primarily owned by the TEST_CASE body via bind_env()/release_env(), but if a TEST_CASE body throws before dtor runs, or if a future TEST_CASE forgets to call release_env(), the listener's unconditional pause() guarantees the extension lock isn't held when the next test starts. Cost: one extra is_active()==false check per testCaseStarting."

patterns-established:
  - "Adding a new Sirius config variant (different topology, different memory limits, etc.) to the test suite is: (1) copy the YAML, bump the one field, add a sync-discipline comment at top; (2) declare a new shared_test_env* global in sirius_test_env.hpp and define it in .cpp; (3) construct it paused in unittest.cpp main(); (4) listener pauses it between tests; (5) TEST_CASE bodies resume()/pause() via acquire_*-style helper. Roughly 30 lines of infra for a new config variant."
  - "To parameterize a large TEST_CASE cohort on a runtime config, add a convenience macro at the section header that expands to GENERATE + CAPTURE + compare_*_for(num_gpus, ...); do a one-line mechanical substitution at each call site; let Catch2's sections machinery + CAPTURE handle the per-variant reporting. Matches the research's Option A from 08-RESEARCH.md."

requirements-completed: [TEST-01, TEST-02]

# Metrics
duration: 20min
completed: 2026-04-22
---

# Phase 08 Plan 04: TPC-H 2-GPU test parameterization Summary

Closes TEST-01 and TEST-02 by adding a 2-GPU integration fixture
(`integration-2gpu.yaml`), wiring a second `shared_test_env` through
`test/cpp/utils/sirius_test_env.{hpp,cpp}` and `test/cpp/unittest.cpp`, and
parameterizing all 44 TPC-H TEST_CASEs in `test_gpu_execution_tpch.cpp` on
`num_gpus ∈ {1, 2}` via a Catch2 `GENERATE(1, 2)` macro. The default
`integration.yaml` is unchanged; a single `mcp__project-commands__run_command
unit-tests` invocation now executes every TPC-H query under both topologies.

## What Shipped

**New fixture YAML** (`test/cpp/integration/integration-2gpu.yaml`, 30 lines):
byte-for-byte identical to `integration.yaml` except `topology.num_gpus: 2` and
a leading sync-discipline comment. Validated by
`diff <(grep -v '^#' ... | grep -v num_gpus) <(...)` returning zero output.

**Second shared_test_env** (`test/cpp/unittest.cpp:106-116`): constructs
`integration_env_2gpu` against the new YAML, pauses it at startup, and wires it
into `sirius::test::g_integration_env_2gpu`. The existing `shared_env_listener`
was extended to also pause the 2-GPU env between tests so the extension lock is
never unexpectedly held across TEST_CASE boundaries. Shutdown clears the pointer
alongside the two existing ones.

**Helper** (`test/cpp/utils/sirius_test_env.{hpp,cpp}`):
`shared_test_env* acquire_integration_env_for(int num_gpus)` returns
`g_integration_env` for `num_gpus==1`, `g_integration_env_2gpu` for `num_gpus==2`
on hosts where `cudaGetDeviceCount(&n)` reports `n>=2`, and `nullptr` otherwise
(drives the single-GPU-host WARN+return path).

**Fixture parameterization** (`test/cpp/integration/test_gpu_execution_tpch.cpp`):
`GPUExecutionFixtureBase` gains
- `bool bind_env(int num_gpus)` — releases any previously-bound env, acquires
  the matching env, resumes it, and creates a fresh `duckdb::Connection` on it;
  returns `false` when the requested env is unavailable;
- `void release_env()` — drops the connection and pauses the currently-bound
  env; safe to call multiple times;
- `bool compare_gpu_vs_cpu_for(int num_gpus, const std::string& query, optional<float> tol)`
  — `bind_env` + `compare_gpu_vs_cpu`; returns `false` when `bind_env` did;
- `virtual void setup_schema()` — subclass hook rerun after each env swap so
  the DuckDB-subclass can reattach the tpch database and the Parquet-subclass
  can recreate the 8 `read_parquet` views on the fresh connection;
- destructor now calls `release_env()` so any acquired 2-GPU env is paused at
  the close of each TEST_CASE.

A section-local convenience macro is defined at line 3296:

    #define RUN_TPCH_MGPU(...)                                        \
      do {                                                            \
        auto const num_gpus = GENERATE(1, 2);                         \
        CAPTURE(num_gpus);                                            \
        if (!compare_gpu_vs_cpu_for(num_gpus, __VA_ARGS__)) return;   \
      } while (0)

All 44 TPC-H TEST_CASEs (Q1..Q22 × {DuckDB, parquet}) were converted by a
single-line mechanical substitution `compare_gpu_vs_cpu(` → `RUN_TPCH_MGPU(`;
query strings and TEST_CASE headers are byte-identical to before. Verified
via `grep -c RUN_TPCH_MGPU = 46` (1 doc-comment reference + 1 `#define` +
44 call sites) and `grep -cE 'GENERATE\s*\(\s*1\s*,\s*2\s*\)' = 1` (inside
the macro body).

## Deviations from Plan

None. Plan executed exactly as written. No Rule 1/2/3 auto-fixes needed.

The plan's "CRITICAL PROBE CONTEXT" anticipated that num_gpus=2 TPC-H runs may
fail on the known open host_parquet converter bug (handed off to 08-06). Since
this host lacks GPU driver access, no smoke run could be performed — the build
pass is the definitive gate for this plan, and the anticipated 08-06 failures
are for later waves to document.

## Authentication Gates

None.

## Test Strategy Validation

- **Build gate:** `mcp__project-commands__run_command build` — exit 0 (35.9s cold,
  10.3s incremental after Task 3). Both the utils/sirius_test_env.cpp change
  (introducing `<cuda_runtime.h>`) and the test_gpu_execution_tpch.cpp macro
  expansion compiled cleanly across all TUs.
- **Smoke run not possible:** the worktree host has no NVIDIA driver loaded
  (`nvidia-smi -L` fails with "couldn't communicate with the NVIDIA driver"),
  so `sirius_unittest` refuses to initialize (MGPU-01 fail-hard). The test
  infra is built and statically invariant; Plan 08-05 will be the first plan
  that exercises it at runtime on the 2×RTX 6000 Ada verification host.
- **Regression invariants preserved:** integration.yaml is byte-unchanged
  (`git diff test/cpp/integration/integration.yaml` is empty); the TPC-H
  TEST_CASE names and tags are byte-unchanged so existing
  filter/include arguments keep working; non-TPC-H TEST_CASEs in the same file
  (scan, projection, join, aggregate, top_n, cpu_source_task — 314 TEST_CASEs)
  were not touched.

## Handoff

**To Plan 08-05 (AUDIT TEST_CASE):** the audit TEST_CASE for AUDIT-01/02/03 can
use the same `sirius::test::acquire_integration_env_for(2)` pattern to bind a
2-GPU connection, plus the same `bind_env`/`release_env` lifecycle. No changes
to sirius_test_env.{hpp,cpp} are required beyond what this plan shipped.

**To Plan 08-06 (ship gate):** the host_parquet_representation converter fix
will make the num_gpus=2 variant of TPC-H parquet TEST_CASEs (Q1-Q22 parquet)
green. The infra this plan ships means ship-gate verification is literally
`mcp__project-commands__run_command unit-tests` without any additional flags;
pass = 2×44 TPC-H variants green on 2×RTX 6000 Ada.

## Commits

| Task | Name                                            | Commit  |
| ---- | ----------------------------------------------- | ------- |
| 1    | Create integration-2gpu.yaml fixture            | ec3f2e9 |
| 2    | Wire 2-GPU shared_test_env + helper             | dbfa082 |
| 3    | Parameterize TPC-H TEST_CASEs on num_gpus       | 25d9aea |

## Self-Check: PASSED

Verified via:
- `test -f test/cpp/integration/integration-2gpu.yaml` → exit 0
- `git log --oneline` shows ec3f2e9, dbfa082, 25d9aea in order
- `grep -c 'g_integration_env_2gpu' test/cpp/utils/sirius_test_env.hpp test/cpp/utils/sirius_test_env.cpp test/cpp/unittest.cpp` → 3, 2, 5 (sum 10 ≥ 4 required)
- `grep -cE 'GENERATE\s*\(\s*1\s*,\s*2\s*\)' test/cpp/integration/test_gpu_execution_tpch.cpp` → 1 (inside macro body; expands to 44 call sites)
- `grep -c RUN_TPCH_MGPU test/cpp/integration/test_gpu_execution_tpch.cpp` → 46 (1 doc ref + 1 define + 44 sites)
- `mcp__project-commands__run_command build` → exit 0
- `git diff test/cpp/integration/integration.yaml` → empty (default 1-GPU config preserved)
