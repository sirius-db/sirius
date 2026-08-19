# Running the TPC-H sweep on Sirius + StarRocks

Self-contained procedure to run the 22-query TPC-H benchmark against the Sirius GPU compute-node
cluster — **any data path, any scale factor, any number of GPUs**. Assumes the stack is already
built (see `BUILD-SIRIUS-STARROCKS.md`); this document is only about running it and getting
numbers you can defend.

Everything below is parameterised by four inputs:

| Input | Symbol | Example |
|---|---|---|
| TPC-H parquet root, holding `<table>/*.parquet` for all 8 tables | `$DATA` | `/home/ubuntu/tpch_parquet_sf100` |
| Scale factor | `$SF` | `100` |
| Number of CNs = number of GPUs to use | `$N` | `2` |
| Per-GPU HBM as the card **reports** it | `$CARD` | `95.6 GiB` (97887 MiB) |

---

## 0. The one thing to understand before you start

**The harness has no correctness gate.** `bench.sh` scores a run `pass` on exactly three
conditions (`experimental/starrocks/benchmarks/tpch/bench.sh:175`):

```bash
if [ $rc -eq 0 ] && [ -s "$f" ] && ! head -1 "$f" | grep -q ERROR; then
```

exit code 0, non-empty output file, no `ERROR` on line 1. It never compares a value against
anything. `analyze.py` reads only `status` and `ms`; the `rows` column is written and never
diffed between engines. **A query returning wrong numbers, or 100 rows instead of 100000, is
recorded as a fast WIN.**

Two direct consequences, both of which bit on the reference run in §7:

- Every number you quote must be backed by a DuckDB-oracle check (§5). Non-negotiable.
- A **correct empty result is recorded as `wedge`**, because `mysql --batch` prints nothing at
  all — not even a header — for a zero-row result set, so `[ -s "$f" ]` fails. Before calling an
  empty result a bug, check the oracle: TPC-H q11 legitimately returns 0 rows at SF≥100 (§7.3).

---

## 1. Preflight

All three of these must print nothing, or you will corrupt a live cluster rather than fail to
bind — the FE keys a node by `(advertise_host, heartbeat_port)` and the nixl agent by
`advertise_host:brpc_port`, so a second launch silently rewrites both registries.

```bash
pgrep -fa '[s]irius-starrocks-cn|[S]tarRocksFE'
ss -ltn | grep -E ':(8030|9010|9020|9030|91[0-9][0-9])\b'
nvidia-smi --query-compute-apps=pid --format=csv,noheader
```

Verify the dataset has all 8 tables — `run-abc.sh` requires at least one `*.parquet` in each,
and a missing table surfaces as a `refused` on every query that touches it, not as a setup error:

```bash
for t in customer lineitem nation orders part partsupp region supplier; do
  printf "%-10s %s\n" $t "$(ls $DATA/$t/*.parquet 2>/dev/null | wc -l)"
done
```

---

## 2. Size the cluster

Three numbers, and they interact. Get these wrong and queries fail for reasons that look like
engine bugs but are not.

### The device budget

The exchange staging arena is a bare `cudaMalloc` **outside** the RMM pool;
`usage_limit_fraction` knows nothing about it. So:

```
device occupancy = GPU_MEM (--gpu-memory-limit) + STAGING (SIRIUS_EXCHANGE_STAGING_BYTES) + ~2 GiB
                                                                        (CUDA context, cudf, fragmentation)
```

Every GiB of arena costs a GiB of pool. The committed, validated configs all run the card at
**85–97 % of nameplate**:

| Machine | GPUs | HBM/GPU | N | GPU_MEM | STAGING | HOST_MEM | % of HBM |
|---|---|---|---|---|---|---|---|
| A100x8, 8-CN (validated, SF500) | 8 | 80.0 GiB | 8 | 66 GiB | 12 GiB | 100 GiB | 97.5 % |
| A100x8, 4-CN (validated, SF500) | 8 | 80.0 GiB | 4 | 54 GiB | 24 GiB | 200 GiB | 97.5 % |
| GB200 4-GPU (validated, SF100) | 4 | 184.0 GiB | 4 | 140 GiB | 16 GiB | 160 GiB | 84.8 % |
| GB200 4-GPU (SF1000) | 4 | 184.0 GiB | 4 | 128 GiB | 32 GiB | 112 GiB | 84.9 % |
| `cluster8.sh` defaults | — | — | 8 | 64 GiB | 8 GiB | 128 GiB | 90 % of 80 GiB |

**Sizing rule:** pick `STAGING` first (below), then `GPU_MEM = CARD − STAGING − 2 GiB`.

### STAGING — the arena

This is the knob that decides whether the big shuffle queries (q08, q09) run at all. It scales
**inversely with CN count** and **directly with scale factor**:

```
STAGING(N, SF) ≈ 96 GiB × (SF / 500) / N     rounded up to the next 4 GiB
```

Measured points on 8× A100 at SF500: 8 CN needs 12 GiB (fails at 8); 4 CN needs 24 GiB (fails at
16); 2 CN has no working split on an 80 GiB card, because 48 GiB of arena plus a usable pool does
not fit.

**Treat the formula as a floor, not a prediction** — it underestimated on the §7 reference run
(SF100, N=2: formula says 12 GiB, 16 GiB was measured *exhausted*). The arena must hold the
**sum of concurrent leases**, not the largest one; 9–82 simultaneous leases have been observed.
Fewer CNs is strictly worse: each carries more of the fan-out.

Exhaustion is loud and self-naming, and tells you exactly what to do:

```
exchange staging arena exhausted: requested 1242515456 bytes, 778297088 free of
17179869184 capacity with 14 leases outstanding (raise SIRIUS_EXCHANGE_STAGING_BYTES)
```

**`SIRIUS_EXCHANGE_STAGING_BYTES` has no engine default.** Unset means *no arena at all*: the CN
boots healthy, registers, answers local queries, and every remote exchange destination fails. The
"default" is a launcher-script value and differs per script (8 GiB `cluster8.sh`, 16 GiB
`script-box.sh`, 2 GiB `nixl-echo-2node.sh`).

### HOST_MEM

`memory.host.capacity_bytes` is **per NUMA region**, not machine-wide. On a 1-NUMA box, `N` CNs
each take their full `HOST_MEM` from the same pool, so `N × HOST_MEM` must fit RAM with room for
page cache. It is routinely over-provisioned — measured high-water on the reference run was
631 MiB against a 200 GiB ceiling — so prefer leaving RAM to the page cache for the parquet reads.

---

## 3. Bring the cluster up

`cluster8.sh` launches 1 FE + `N` CNs, one per GPU, GPU ordinal `i` → CN `i` (hardcoded identity
map; there is no way to select a GPU subset). It **blocks** on `wait -n`, and its `EXIT/INT` trap
tears the cluster down with the shell — give it its own terminal or background task, never chain
it behind `&` inside another command.

```bash
cat > up.sh <<'EOF'
#!/usr/bin/env bash
set -uo pipefail
cd /path/to/sirius/experimental/starrocks
unset CUDA_VISIBLE_DEVICES              # MANDATORY -- see below
export SIRIUS_QUERY_WATCHDOG_SECS=90
export NUM_CNS=2 GPU_MEM=60GiB STAGING=32GiB HOST_MEM=128GiB
exec ./benchmarks/cluster8.sh
EOF
chmod +x up.sh && ./up.sh 2>&1 | tee -a /tmp/cluster.log
```

**`unset CUDA_VISIBLE_DEVICES` is mandatory.** An already-exported value **beats `--gpu-device`
and is only `warn!`ed about**, collapsing every CN onto one GPU — a cluster that still answers
queries, so the harness happily records numbers from it. `cluster8.sh` does not clear it.

`cluster8.sh` preflights only three things: CN binary exists, FE script exists, and visible GPU
count ≥ `NUM_CNS`. **No port preflight and no GPU-claim preflight** — that is §1's job.

Wait for the cluster, then assert distinct GPUs:

```bash
M=/path/to/experimental/starrocks/.pixi/envs/default/bin/mysql
until [ "$($M -h127.0.0.1 -P9030 -uroot -N -e 'SHOW COMPUTE NODES;' 2>/dev/null \
        | awk -F'\t' '$9=="true"' | wc -l)" -ge $N ]; do sleep 5; done
nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv   # must be N DISTINCT uuids
```

**Count the `Alive` column, never `grep -c true`.** `SHOW COMPUTE NODES` also emits
`SystemDecommissioned`, `ClusterDecommissioned` and `HasStoragePath`, so a *booting* node with
`Alive=false, HasStoragePath=true` matches a whole-row grep. `Alive` is column 9.

Ports: base 9100, stride 10 — CN `i` gets heartbeat `9100+10i`, thrift `+1`, brpc `+2`, http `+3`,
starlet `+4`. FE uses 8030/9010/9020/9030.

Logs: the CN's Rust/transport output goes to **stdout only**, so FE and all CNs interleave into
whatever you teed. The C++ engine log is separate, at `.cn<i>/log/sirius_<date>.log` — that is
where OOM and scheduler errors land, and it is the file to read when a query wedges.

---

## 4. Run the sweep

`RESTART_CMD` is **mandatory**. The CN has no `cancel_plan_fragment`, so a hung or failed query
strands its fragments and eventually starves the cluster; without a restart every later
measurement is invalid.

```bash
cat > restart.sh <<'EOF'
#!/usr/bin/env bash
set -uo pipefail
M=/path/to/experimental/starrocks/.pixi/envs/default/bin/mysql
pkill -f '[s]irius-starrocks-cn'; pkill -f '[S]tarRocksFE'; sleep 15
nohup /path/to/up.sh >>/tmp/cluster.log 2>&1 &          # APPEND, never truncate
for _ in $(seq 1 90); do
  n=$("$M" -h127.0.0.1 -P9030 -uroot -N -e 'SHOW COMPUTE NODES;' 2>/dev/null \
      | awk -F'\t' '$9=="true"' | wc -l)
  if [ "${n:-0}" -ge 2 ]; then
    [ "$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u | wc -l)" -ge 2 ] \
      || { echo "CNs collapsed onto one GPU" >&2; exit 1; }
    exit 0
  fi
  sleep 5
done
exit 1
EOF
```

Append to the cluster log, never truncate — a restart is fired *by* a failure whose only evidence
is in that log.

```bash
export PATH=/path/to/experimental/starrocks/.pixi/envs/default/bin:$PATH   # bench.sh calls bare `mysql`
TPCH_DATA=$DATA QUERY_TIMEOUT=180 COLD_TIMEOUT=240 MIN_BACKENDS=$N \
RESTART_CMD=/path/to/restart.sh \
  ./benchmarks/tpch/bench.sh --cold /path/to/out/timings.csv 3
```

Environment that matters:

| Var | Default | Note |
|---|---|---|
| `TPCH_DATA` | **required** | textually substituted for `__TPCH_DATA__`; never validated as a directory |
| `QUERY_TIMEOUT` | 30 | warm runs only. **Raise to 180** — at 30 s every failure class collapses into an indistinguishable `wedge` |
| `COLD_TIMEOUT` | 180 | run 0 only, and it applies **even when `COLD=0`** and the row will be discarded |
| `MIN_BACKENDS` | 2 | set to `$N`. It is an **exact-topology gate**, not a floor: *more* alive nodes than this is a hard abort (override with `ALLOW_EXTRA_BACKENDS=1`). It sums CNs + BEs, so it cannot express "N compute nodes" |
| `RESTART_CMD` | empty | mandatory here |
| `FE_PORT` | 9030 | host and user are hardcoded `127.0.0.1`/`root` — you cannot point at a remote FE without editing the script |

`--cold` records run 0 as `phase=cold` instead of discarding it, so first-contact cost (lazy nixl
session setup, plan-cache misses, first-touch allocation) is visible without polluting warm
medians. In `--cold` mode a run-0 failure does **not** restart or break — the warm runs continue
on the same cluster.

**Argument-shifting trap:** a query subset must be preceded by an explicit runs count.
`bench.sh out.csv q05` sets `RUNS=q05` and sweeps all 22. Correct form: `bench.sh out.csv 3 q05`.

CSV schema is `query,run,phase,status,ms,rows`. Failing rows are always written; a run-0 **pass**
is only written with `--cold`. Copy the CSV somewhere durable — nothing is committed under
`results/`.

---

## 5. The correctness gate (do not skip this)

Run the same SQL through DuckDB over the same parquet and diff. This is the only check that
exists.

```python
# oracle.py -- FILES("path"="file:///d/t/*.parquet","format"="parquet") -> read_parquet('/d/t/*.parquet')
import duckdb, glob, os, re, sys
FILES = re.compile(r'FILES\(\s*"path"\s*=\s*"file://([^"]+)"\s*,\s*"format"\s*=\s*"parquet"\s*\)', re.I)
qdir, data, out = sys.argv[1:4]
con = duckdb.connect(); con.execute("PRAGMA threads=48")
for p in sorted(glob.glob(os.path.join(qdir, "q*.sql"))):
    n = os.path.basename(p)[:-4]
    sql = FILES.sub(lambda m: f"read_parquet('{m.group(1)}')",
                    open(p).read().replace("__TPCH_DATA__", data)).strip().rstrip(";")
    rel = con.sql(sql); rows = rel.fetchall()
    with open(f"{out}/{n}.tsv", "w") as f:
        f.write("\t".join(rel.columns) + "\n")
        for r in rows: f.write("\t".join("NULL" if v is None else str(v) for v in r) + "\n")
```

Use a **pip** `duckdb`, not `build/release/duckdb` — the repo binary auto-loads the Sirius
extension and will fight the running CN for GPU memory (`cudaErrorMemoryAllocation`).

At SF100 the whole 22-query oracle takes about 20 s on 48 cores. Compare **key-matched**, not
positionally: if values drift, the `ORDER BY … LIMIT N` reorders rows, and a positional diff
reports a meaningless 300 % difference instead of the real 0.3 %. Join on the query's key column
(`l_orderkey` for q03, `c_custkey` for q10, …), then compare cells.

Expect a **systematic low bias** on `sum(l_extendedprice * (1 - l_discount))` — the open
decimal-lowering defect. Establish the band from the oracle before accepting any mismatch; see
§7.2 for the measured band.

---

## 6. Triage

Read `.cn<i>/log/sirius_<date>.log` — the classification below comes from the engine log, not the
CSV.

| Symptom | Meaning | Move |
|---|---|---|
| `wedge`, ms ≈ `QUERY_TIMEOUT`×1000 | real stall | check the engine log. `exceeded 100 retries … OOM at operator HASH_JOIN` is memory pressure, **not** a hang — but raising `GPU_MEM` does not always fix it (§7.4: a 50 % larger pool did nothing for q09). Try more CNs to shrink the per-node build side |
| `wedge`, ms far **below** the timeout | rc=0 with an empty result | **check the oracle first.** A correct 0-row answer looks identical (§0). Only if the oracle disagrees is it the empty-result defect |
| `refused` + `exchange staging arena exhausted` | arena too small for the concurrent lease set | raise `STAGING` per §2; the error prints capacity, free bytes and lease count |
| `refused` + `no parked sender output to export for SenderSlot` | head-of-line deadlock in the exchange | a known regression class; usually appears alongside an OOM that it masks |
| `refused` at ~60 s | the CN→CN PRPC reply timeout, default 60 s | not this query's plan — read the **peer** CN's stdout, matching `peer=<host:brpc_port>`. Tunable via `SIRIUS_CN_RPC_TIMEOUT_SECS`; see `experimental/starrocks/docs/TUNABLES.md` |
| `pass` but values wrong | decimal drift, or a real bug | §5. Drift is **low-biased** and hits only `sum(x*(1-discount))` expressions |
| `pass` with a plausible row count | **verified by nothing** | §5 |

After any wedge or refusal, restart before trusting the next row.

Teardown, and the check that proves it:

```bash
pkill -f '[s]irius-starrocks-cn'; pkill -f '[S]tarRocksFE'
nvidia-smi --query-compute-apps=pid --format=csv,noheader     # must be EMPTY
```

Check compute-apps, **not** `memory.used` — its idle floor is tens of MiB and never reaches 0.

---

## 7. Reference run — SF100, 2× RTX PRO 6000 Blackwell, 2 CNs

Recorded 2026-08-19 so a new run can be told apart from a known-good deviation. Box: Ubuntu
24.04, Xeon 8559C 48 vCPU, 499 GB RAM, 1 NUMA node; 2× RTX PRO 6000 Blackwell 97887 MiB, cc 12.0,
driver 580.126.09; GPUs linked `PIX` (PCIe switch, **not** NVLink) with `topo -p2p r` = OK.
Data: `/home/ubuntu/tpch_parquet_sf100`, 26 GB, 8 tables, lineitem in 6 parquet files.

Config: `NUM_CNS=2 GPU_MEM=40GiB STAGING=16GiB HOST_MEM=128GiB SIRIUS_QUERY_WATCHDOG_SECS=90`,
`QUERY_TIMEOUT=180 COLD_TIMEOUT=240 MIN_BACKENDS=2`, 1 cold + 3 warm runs.

### 7.1 Timings — 19/22 recorded `pass`, warm medians (ms)

| q01 | q02 | q03 | q04 | q05 | q06 | q07 | q10 | q12 | q13 | q14 |
|---|---|---|---|---|---|---|---|---|---|---|
| 3998 | 512 | 1064 | 728 | 1961 | 791 | 1415 | 1271 | 808 | 842 | 1052 |

| q15 | q16 | q17 | q18 | q19 | q20 | q21 | q22 |
|---|---|---|---|---|---|---|---|
| 1949¹ | 281 | 2249 | 1605 | 1046 | 1464 | 2710 | 398 |

¹ q15 passed 1 of 3 warm runs; the other two returned empty (the known q15 flake).

Not passing: **q08** refused, **q09** wedged (180 s), **q11** recorded `wedge` at 620 ms.

### 7.2 Correctness vs the DuckDB oracle — the part the harness does not do

Of the 20 queries that returned data, **all 20 have the correct row set**. Values:

| Class | Queries | Worst relative error |
|---|---|---|
| Byte-exact | q02 q04 q06 q12 q13 q16 q17 q18 q20 q21 | 0 |
| IEEE noise only | q22 | 1.7e-16 |
| Low-biased decimal drift | q01 q05 q07 q14 q19 q15 | 0.103 % |
| Drift **that reorders the top-N** | **q03, q10** | **0.336 %** |

Every difference is **low** (Sirius under the oracle) and confined to
`sum(l_extendedprice * (1 - l_discount))`; keys, counts and dates are exact. This matches the
documented drift band (≤0.39 % at SF1) — SF100 does not widen it.

q03 and q10 matter more than the percentage suggests: both are `ORDER BY revenue DESC LIMIT N`,
so a 0.34 % error **changes the ranking**. q03 row `165214338` should be 2nd and lands 3rd; q10
rows at positions 12–14 permute. The answer is observably wrong, not merely imprecise.

### 7.3 q11 is not a bug — correct the record

q11 was recorded `wedge` at 620 ms with an empty result. **The DuckDB oracle also returns 0
rows**, and the arithmetic says that is right: the query hardcodes the SF1 threshold
`0.0001`, while the TPC-H spec scales it as `0.0001/SF`. At SF100 that makes the bar
**801,681,490** against a largest single part value of **23,649,655** — 34× too high, so no part
qualifies.

So q11 is (a) a query text that is not scale-adjusted, and (b) a harness misclassification of a
correct empty result. It is **not** the "empty-result defect" that existing repo notes list it
as. Sirius and DuckDB agree exactly.

### 7.4 q08 and q09 — one is sizing, one is not

Both failures name their own cause in the engine log:

- **q09** — `.cn*/log/sirius_*.log`: `task 115 … exceeded 100 retries at operator index 0 —
  terminating query: OOM at operator HASH_JOIN (index 0)`.
- **q08** — `exchange staging arena exhausted: requested 1242515456 bytes, 778297088 free of
  17179869184 capacity with **14 leases outstanding**`.

Arm A used 40 + 16 = 56 GiB of a 95.6 GiB card — **59 %**, where every validated config in §2 runs
at 85–97 %. It was frozen from a 4-CN topology; at 2 CNs each CN carries twice the fan-out. So a
second arm was run at `GPU_MEM=60GiB STAGING=32GiB` (92.6 GiB, **96.8 %** of card, measured
94788 MiB) with everything else identical. **The result separates the two failures:**

| Query | Arm A (40+16) | Arm B (60+32) | Conclusion |
|---|---|---|---|
| q08 | `arena exhausted, 14 leases` | `no parked sender output to export for SenderSlot` | **sizing fixed the arena defect**, exposing a distinct exchange head-of-line deadlock underneath |
| q09 | `100 retries → OOM at HASH_JOIN` | `100 retries → OOM at HASH_JOIN` (both CNs) | **not sizing** — a genuine defect at this topology |
| q15 | cold wedge, 1/3 warm pass | cold pass, 0/3 warm pass | **flaky either way** — not config-sensitive |

Full arm-B warm medians (ms), 18/22 pass. Common-query total **24105 ms vs A's 24195 ms — −0.4 %**:

| q01 | q02 | q03 | q04 | q05 | q06 | q07 | q10 | q12 | q13 |
|---|---|---|---|---|---|---|---|---|---|
| 4093 | 573 | 1058 | 691 | 1949 | 793 | 1420 | 1231 | 791 | 833 |

| q14 | q16 | q17 | q18 | q19 | q20 | q21 | q22 |
|---|---|---|---|---|---|---|---|
| 1043 | 260 | 2264 | 1550 | 1012 | 1459 | 2691 | 394 |

**Correctness was bit-identical between the two arms** — same drift values on the same queries.
The decimal drift is deterministic and independent of memory configuration, which rules out
memory pressure or a downgrade path as its cause.

**Do not assume an `OOM at HASH_JOIN` is undersizing.** A 50 % larger pool did not move q09 at
all: it still exhausts all 100 retries on both CNs. q09 is the widest join in TPC-H
(lineitem+orders+partsupp+part+supplier+nation); at 2 CNs each node's build side exceeds what
60 GiB can hold, and the fix is more CNs (to shrink the per-node build) or engine-side work — not
more HBM per CN.

The methodological point generalises: **an undersized resource can mask a second defect.** Only
after the arena was large enough did q08's real failure become visible.

Timings across the two arms were flat (q01–q07 within ±2 % except q02 +12 %, q04 −5 %), so at
SF100 the extra HBM buys failure-avoidance on the big shuffles, not speed.

---

## 8. Parameterising for a different box or scale

```bash
DATA=/path/to/tpch_parquet_sf<SF>
SF=<scale factor>
N=<number of GPUs to use>
CARD=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)  # MiB

STAGING_GiB=$(python3 -c "import math;print(max(4,4*math.ceil(96*($SF/500)/$N/4)))")
GPU_MEM_GiB=$(python3 -c "print(int($CARD/1024) - $STAGING_GiB - 2)")
echo "NUM_CNS=$N GPU_MEM=${GPU_MEM_GiB}GiB STAGING=${STAGING_GiB}GiB"
```

Then sanity-check against §2: `GPU_MEM + STAGING` should land at 85–97 % of `CARD`. If the
formula puts you below 80 %, you are leaving memory unused and will hit avoidable OOMs; if above
97 %, bring-up will fail on allocation and you should lower `GPU_MEM` first.

Treat the result as a **starting point that must be validated**, not a prediction: the arena
formula underestimated at SF100/N=2 (§7.4). Run the sweep, read the two self-naming errors in §6,
and adjust the one they name.
