# Sirius StarRocks CN: docs

Plans for the Sirius StarRocks compute node (CN), and the order to do them in. Status was checked against the top of the PR stack: #2037 → … → #2043 → #2061 → #2062 (gh stack #2044). The pinned StarRocks is 4.1.3 (`8a8e186`).

## Results

| doc | what it holds |
|---|---|
| [benchmark-results-summary.md](benchmark-results-summary.md) | TPC-H join results on 4× GB200 at SF1000 and SF3000, and which PR each result supports. |

## Plans

| doc | scope |
|---|---|
| [tpch-coverage-and-filters-plan.md](tpch-coverage-and-filters-plan.md) | **The order of work:** TPC-H coverage and runtime filters as one sequence of milestones (M0–M7), each proven on queries that already run. |
| [tpch-coverage-plan.md](tpch-coverage-plan.md) | Get TPC-H from 8/22 to 22/22: the five plan-shape gaps and the order to close them. |
| [design/failure-reporting-and-cancellation.md](design/failure-reporting-and-cancellation.md) | Report failures to the FE and peer CNs, cancel the way the FE expects, interrupt running fragments. |
| [design/fe-contract-and-versions.md](design/fe-contract-and-versions.md) | Batch dispatch fix, version reporting and policy, column-order contract tests. |
| [design/multi-phase-aggregation.md](design/multi-phase-aggregation.md) | Two-phase COUNT/MIN/MAX/AVG, then three- and four-phase plans (DISTINCT, grouping sets). |
| [design/distributed-runtime-filters.md](design/distributed-runtime-filters.md) | Apply 39 of the 42 runtime filters the FE plans: guards, partitioned joins, injection into Sirius scan filters, same-fragment filters, the StarRocks protocol. |
| [design/exact-decimal.md](design/exact-decimal.md) | Replace the FP64 decimal lowering with exact DECIMAL. |
| [design/cpu-fallback-and-deployment.md](design/cpu-fallback-and-deployment.md) | What happens to a query the GPU can't run. |
| [sf3000-memory-plan.md](sf3000-memory-plan.md) | SF3000 memory: spill, streaming, runtime filters (mostly landed in #2061 and #2062). |
| [failed-query-gpu-leak-plan.md](failed-query-gpu-leak-plan.md) | Freeing a failed query's GPU memory (landed in #2041). |

## Recommended order

1. **TPC-H coverage and runtime filters, M0–M4** ([tpch-coverage-and-filters-plan.md](tpch-coverage-and-filters-plan.md)):
   - the scoreboard;
   - `LIMIT` (#1963);
   - two-phase COUNT/MIN/MAX;
   - RIGHT SEMI join;
   - partitioned-join filters;
   - two-phase AVG.

   That's up to 8 → 19 queries, with 24 filters applied in queries that complete.
2. **Failure reporting Steps 1–2**: a failure frame between CNs, and `reportExecStatus`. These turn hangs into fast errors, and streaming receivers need them.
3. **FE contract Steps 1–2**: the batch dispatch fix and the version string. Small and independent.
4. **Cancellation and interruption** (failure reporting Steps 3–5). Fixes the SF3000 sequence fallout (q12 after q09).
5. **Exact decimal investigation.** It decides whether an audited TPC-H run is possible.
6. **M5–M7 of the combined plan** (layout bugs, q11, filter injection into Sirius scans, same-fragment filters), then **CPU fallback**.

## Scripts

- `scripts/rf_tally.py <run_dir>`: tallies the FE's runtime filters in a run's recorded fragments (`SIRIUS_CN_DUMP_FRAGMENTS`) and judges each against the CN's rules.
- `scripts/rf_logs.py <run_dir>`: attributes the CN's runtime-filter log lines to queries.
