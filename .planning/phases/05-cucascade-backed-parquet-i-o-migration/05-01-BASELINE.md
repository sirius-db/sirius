# TPC-H SF1 Pre-Migration Baseline

**Captured:** 2026-04-21T00:58:44Z
**Purpose:** Correctness baseline for Phase 5 IO-09 — post-migration `tpch-sirius.test` must produce identical pass/fail status for every query on the same host under the same environmental conditions. Any new query-level regression introduced by the cucascade-backed parquet I/O migration is a blocker.

## Environment

- Sirius HEAD: `64d565fa31f1c3dd963bd9fe1f39cf2205003ff5` (Phase-4-HEAD descendant on `feature/single-node-multi-gpu2`; follows 13e4322 "docs(04): complete Phase 4")
- Cucascade HEAD: `f47de0bb7bcaddd55081a9c4bc584627532d1ef9` (matches BUMP-01 pin from Phase 4; includes PR #96 `disk_io_backend` + `io_backend_registry`)
- Hostname: `6f7e4c9-lcedt`
- Kernel: `Linux 6.17.0-1014-nvidia #14-Ubuntu SMP PREEMPT_DYNAMIC Tue Mar 17 19:04:24 UTC 2026 x86_64`
- GPUs: `nvidia-smi` unavailable on this host — NVIDIA driver not loaded (`nvidia-smi -L` exits 9: "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver"). This is a planning/CI host without an attached GPU driver; multi-GPU validation for IO-02 (per-backend CUDA-context isolation) must be re-run on the 2+ GPU validation host identified for Phase 4 (`N=2` host used in plan 04-05).

## Command

```
build/release/test/unittest --test-dir . test/sql/tpch-sirius.test
```

## Result

- Exit code: `1`
- Test cases: `1 | 1 failed`
- Assertions: `1 | 1 failed`
- Queries PASS: `0/22` (the test harness never reached per-query execution)
- Queries FAIL: `n/a` (see Failure Mode below — the extension load itself failed)
- Q4 flake retry: retried once; identical deterministic failure (same exit code, same error message). No Q4-specific behavior observed because the test aborts before any TPC-H query runs.

## Failure Mode — Deterministic (environmental, not a code regression)

The test fails at `test/sql/tpch-sirius.test:20` during `LOAD 'sirius.duckdb_extension'`:

```
test/sql/tpch-sirius.test:20: extension 'sirius' load threw an exception:
Invalid Error: Requested number of GPUs exceeds available GPUs
```

Root cause: the host has no NVIDIA driver available (NVML returns `Driver Not Loaded`). Sirius's extension load path queries available GPUs during context initialization; when the runtime GPU count is zero and the configured GPU count is >= 1, initialization throws the `Invalid Error: Requested number of GPUs exceeds available GPUs` above.

This failure is:

1. **Deterministic** — observed identically on both the baseline run and the retry run.
2. **Environmental, not a code regression** — Phase 4 completed with this HEAD on a 2+ GPU validation host (per plan 04-05 SUMMARY, 966 test cases PASS including the `tpch-sirius.test` suite on GPU hardware).
3. **Reproducible** — the failure will be the baseline comparison on *this* planning host. On a host with an attached GPU driver, the baseline would exercise all 22 TPC-H SF1 queries.

## Validation Rule for Phase 5 Sign-off

Two-tier validation is required:

### Tier A — This host (planning / CI without GPU driver)

After the Phase 5 migration, re-run the same command on this host. **Every query** that would have been exercised here MUST retain the same pass/fail status. Because the extension load fails before any query runs, the post-migration expectation on this host is:

- Exit code: `1`
- Same error: `Invalid Error: Requested number of GPUs exceeds available GPUs`
- No new earlier failure modes (e.g., compile-time include errors, link errors, crash during extension load that isn't the GPU-count check).

Any change in the failure mode on this host signals the migration has introduced an earlier-than-expected failure and MUST be investigated.

### Tier B — 2+ GPU validation host (per plan 04-05)

On a host with ≥1 GPU driver loaded (the same host class used to validate plan 04-05's 966 test cases), the full TPC-H SF1 suite MUST produce per-query pass/fail status identical to Phase-4-HEAD. This is the load-bearing correctness gate for IO-09. The 2+ GPU host is also where the compute-sanitizer memcheck validation (IO-02) executes.

Phase 5 sign-off requires both tiers to pass.

## Raw Output (for diff)

```
Filters: test/sql/tpch-sirius.test

[0/1] (0%): test/sql/tpch-sirius.testFailed to initialize NVML: Driver Not Loaded

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
unittest is a Catch v2.13.7 host application.
Run with -? for options

-------------------------------------------------------------------------------
test/sql/tpch-sirius.test
-------------------------------------------------------------------------------
/home/felipe/sirius/.worktrees/ws-9aa781df-6d8c-4395-9329-737a67e8e272/duckdb/test/sqlite/test_sqllogictest.cpp:212
...............................................................................

/home/felipe/sirius/.worktrees/ws-9aa781df-6d8c-4395-9329-737a67e8e272/duckdb/test/sqlite/sqllogic_parser.cpp:119: FAILED:
explicitly with message:
  test/sql/tpch-sirius.test:20: extension 'sirius' load threw an exception:
  Invalid Error: Requested number of GPUs exceeds available GPUs


[1/1] (100%): test/sql/tpch-sirius.test
===============================================================================
test cases: 1 | 1 failed
assertions: 1 | 1 failed
```

## Cross-Reference

- Phase 4 plan 04-05 SUMMARY recorded 966 Catch2 test cases PASS + all 4 PORT-05 visible tags on the N=2 validation host.
- This baseline is the deterministic local-host companion record; both records together establish "what Phase-4-HEAD does on these two environments".
- When Plan 06 executes the post-migration diff, both records MUST be re-captured on the same respective hosts and compared byte-for-byte at the query-status level (Tier B) and failure-mode level (Tier A).
