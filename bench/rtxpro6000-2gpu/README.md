# Sirius + StarRocks on 2× RTX PRO 6000 Blackwell

Build, benchmark and tuning record for TPC-H on this machine class, plus the portable runbooks
produced from it. Measured 2026-08-19–20.

**Box**: 2× NVIDIA RTX PRO 6000 Blackwell Server Edition (97887 MiB each, cc 12.0), driver
580.126.09, GPUs linked `PIX` (PCIe switch, **not** NVLink), `topo -p2p r` = OK.
Intel Xeon 8559C, 48 vCPU, 499 GB RAM, 1 NUMA node. Ubuntu 24.04.4, no root.

## Start here

| Document | Use it when |
|---|---|
| [STATUS.md](STATUS.md) | **Current answer**: SF100 22/22, SF500 21/22, working config, ranked open work |
| [TPCH-STATUS.md](TPCH-STATUS.md) | Per-query pass/fail, timings, and correctness vs DuckDB (SF100 snapshot, 2026-08-19) |
| [BUILD-SIRIUS-STARROCKS.md](BUILD-SIRIUS-STARROCKS.md) | Standing the stack up on a **fresh box** (engine, FE, CN, UCX/nixl) |
| [TPCH-SWEEP-RUNBOOK.md](TPCH-SWEEP-RUNBOOK.md) | Running the 22-query sweep — **any data path, scale factor, or GPU count** |
| [SIRIUS-TUNING-RUNBOOK.md](SIRIUS-TUNING-RUNBOOK.md) | Choosing `GPU_MEM` / `STAGING` / operator knobs for a machine |

The three runbooks are **portable** — parameterised by data path, scale factor, GPU count and
card size. Agent skills wrap them: `build-sirius-starrocks`, `tpch-cn-sweep`, `cn-tuning`.
Open plans live in [`notes/2026-08-20-rtx-sf500/`](../../notes/2026-08-20-rtx-sf500/); the living
queue is [`notes/OPEN.md`](../../notes/OPEN.md).

## Contents

```
results/   SF100 / SF300 / SF500 CSVs (schema: query,run,phase,status,ms,rows)
tools/     oracle.py, compare.py, drift.py, regress.py, baseline.sh
```

`phase=cold` is run 0 (first contact), `phase=warm` are the timed runs.

## Headline — this box

| Scale | Result | Config |
|---|---|---|
| SF100 | **22/22 correct** | 60/32 |
| SF300 | 21/22 | 60/32 |
| SF500 | **21/22 correct** | 60/32 + 1 GiB operator budgets |

q09 is the only real SF500 failure. q08/q09 use a hand-reordered `FROM` (see
[`QUERY-DEVIATIONS.md`](../../experimental/starrocks/benchmarks/tpch/QUERY-DEVIATIONS.md)) pending
real `FILES()` statistics. q11 empty at SF≥100 matches the DuckDB oracle — the harness misreads
that as a wedge. q15 is flaky (float equality). q21 passes but intermittently stalls.

The harness scores `pass` on exit code + non-empty file + no `ERROR` on line 1 — **it never
compares a value**. Every quoted number needs the oracle (§5 of the sweep runbook).

## Reproducing

```bash
# 1. build (fresh box only)          -- BUILD-SIRIUS-STARROCKS.md
# 2. bring up 2 CNs and sweep        -- TPCH-SWEEP-RUNBOOK.md §3-4
# 3. correctness gate (never skip)   -- TPCH-SWEEP-RUNBOOK.md §5
python3 tools/oracle.py  <repo>/experimental/starrocks/benchmarks/tpch/queries $DATA ./oracle
python3 tools/compare.py <sweep-out-dir> ./oracle 1e-3
```

Use a **pip** `duckdb` for the oracle, not `build/release/duckdb` — the repo binary auto-loads
the Sirius extension and fights the running CN for GPU memory.
