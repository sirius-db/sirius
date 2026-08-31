# TPC-H SF100 sweep — stream-fragment-execution @ 4891c6bf, 2 CNs (one per GPU), NIXL exchange

**Setup:** FE-default two-phase plans (`new_planner_agg_stage=0`), `cbo_cte_reuse=false`,
per-CN 60 GiB pool + 8 GiB staging arena (q08/q09 retried at 24 GiB), multi-file `FILES()` over
`/opt/dlami/nvme/tpch/tpch_parquet_sf100` (DECIMAL dataset). Oracle: DuckDB 1.5.5 CPU wheel over
the same parquet, cell-wise numeric comparison. Failed queries restart the cluster before the
next measurement (stranded-lease protocol).

## Result: 12 of 22 correct; 5 more blocked only by the DECIMAL-dataset FP64 drift; 5 with named blockers

| Tier | Queries | Notes |
|---|---|---|
| **PASS — bit-exact** | q04 q12 q13 q16 q20 q21 | incl. semi/anti joins (q21), `multi_distinct_count` (q16), correlated subqueries (q20) |
| **PASS — value-equal** | q02 q11 q18 | differences are decimal rendering only (`9978.1` vs `9978.10`) |
| **PASS — last-ULP** | q06, q22¹ | FP summation order, max rel dev ≤ 2.6e-15 |
| **PASS — via `agg_stage=1`** | q17 | two-phase `avg` refused at stage 0 by design; stage-1 fallback exact |
| **Known dataset drift²** | q03 (0.336%) q05 (0.097%) q19 (0.098%) q10 (LIMIT-20 boundary) q15 (exact-equality on drifting value) | DECIMAL→FP64 lowering; playbook-documented per-query figures reproduced to the digit; exact on a FLOAT64/V1 dataset |
| **FAIL — named blockers** | q01 q07 q08 q09 q14 | see below |

¹ q22 ran via the stage-1 `avg` fallback; value-equal to the oracle.
² identical behavior and magnitudes to the demo branch on the same dataset.

## The five failures, root-caused

| Query | Root cause | Class |
|---|---|---|
| q01 | two-phase `avg` unsupported (loud refusal, by design); stage-1 fallback ships raw lineitem and exhausts any realistic arena | M2 item: avg → sum+count expansion |
| q07 | 300 s timeout; matches the demo's documented parked-sender leak / scheduler stall (its own STATUS lists q07 as flaky) | pre-existing demo defect, not a port regression |
| q08 | `relay into stream 35 column 0 is declared SMALLINT but the source sink produces BIGINT` (`o_year`) — surfaced after raising the arena to 24 GiB cleared `exchange staging arena exhausted` (56 leases / 8.29 GiB held) | wire-type parity for expression-computed exchange columns; demo's `wire_type_parity.rs` (723 ln) was scoped out of the port |
| q09 | same as q08 (`stream 25 column 1`, `o_year`) | same single fix buys both |
| q14 | `unsupported expression node TExprNodeType(29): outside the v1 StarRocks slice` | scoped translator expression gap |

## Reference points
- Demo branch baseline (same box, FLOAT64 dataset, `agg_stage=1`, tuned budgets): 21/22 with q07 flaky.
- The 8 GiB arena default is too small for q08/q09-class shuffles at SF100: receiver-held leases
  peak >8 GiB because copy-out-on-arrival (demo PLAN-01) is not implemented; 24 GiB clears it.
- Raw outputs: /opt/dlami/nvme/mcn-logs/tpch/{results,oracle-out}; harness: mcn-logs/tpch/sweep.sh.
