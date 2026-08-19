# Sirius + StarRocks on 2× RTX PRO 6000 Blackwell

Build, benchmark and tuning record for TPC-H on this machine class, plus the portable runbooks
produced from it. Measured 2026-08-19.

**Box**: 2× NVIDIA RTX PRO 6000 Blackwell Server Edition (97887 MiB each, cc 12.0), driver
580.126.09, GPUs linked `PIX` (PCIe switch, **not** NVLink), `topo -p2p r` = OK.
Intel Xeon 8559C, 48 vCPU, 499 GB RAM, 1 NUMA node. Ubuntu 24.04.4, no root.

## Start here

| Document | Use it when |
|---|---|
| [TPCH-STATUS.md](TPCH-STATUS.md) | **You want the answer**: per-query pass/fail, timings, and correctness vs DuckDB |
| [BUILD-SIRIUS-STARROCKS.md](BUILD-SIRIUS-STARROCKS.md) | Standing the stack up on a **fresh box** (engine, FE, CN, UCX/nixl) |
| [TPCH-SWEEP-RUNBOOK.md](TPCH-SWEEP-RUNBOOK.md) | Running the 22-query sweep — **any data path, scale factor, or GPU count** |
| [SIRIUS-TUNING-RUNBOOK.md](SIRIUS-TUNING-RUNBOOK.md) | Choosing `GPU_MEM` / `STAGING` / operator knobs for a machine |

The three runbooks are **portable** — parameterised by data path, scale factor, GPU count and
card size. Only `TPCH-STATUS.md` is specific to this box.

## Contents

```
results/   sf100-armA-40g16g.csv    locked config: GPU_MEM=40GiB STAGING=16GiB  -> 19/22 pass
           sf100-armB-60g32g.csv    resized:       GPU_MEM=60GiB STAGING=32GiB  -> 18/22 pass
tools/     oracle.py                run all 22 queries through DuckDB over the same parquet
           compare.py               diff Sirius output against the oracle, with a tolerance
           drift.py                 key-matched drift analysis (survives row reordering)
```

CSV schema: `query,run,phase,status,ms,rows`. `phase=cold` is run 0 (first contact),
`phase=warm` are the timed runs.

## Headline result — SF100, 2 CNs, one per GPU

19/22 recorded `pass` on the locked config. But the harness scores `pass` on exit code +
non-empty file + no `ERROR` on line 1 — **it never compares a value**. Against a DuckDB oracle:

- **11 byte-exact**, 6 more correct within 0.103 %
- **2 wrong in a way that matters** — q03/q10 drift low enough to permute an `ORDER BY … LIMIT N`
- **2 fail** — q08 (exchange deadlock), q09 (`OOM at HASH_JOIN`)
- **1 flaky** — q15
- **q11 is correct**, not a defect: DuckDB also returns 0 rows (unscaled threshold in the query
  text); the harness misreads an empty result as a wedge

## The two arms, and what they proved

Arm A used 56 GiB of a 95.6 GiB card (**59 %**); every validated config in the repo runs at
85–97 %. Arm B re-sized to 92.6 GiB (**96.8 %**), everything else identical:

| | Arm A | Arm B | Conclusion |
|---|---|---|---|
| q08 | `arena exhausted, 14 leases` | `no parked sender output` | sizing fixed one defect, **exposing a second underneath** |
| q09 | `100 retries → OOM at HASH_JOIN` | identical, both CNs | **not sizing** — 50 % more pool changed nothing |
| 18 common queries | 24195 ms | 24105 ms (−0.4 %) | more HBM bought **no speed** |

Correctness was **bit-identical** across both arms, which rules out memory pressure as the cause
of the decimal drift.

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
