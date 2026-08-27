# PLAN-08 — Close four open measurement gaps

**Status: not started. This is a plan, not a result.**
Written to be executed in a **fresh session with zero prior context** — everything needed is below.

---

## 0. Orientation (read this first)

### The box and the tree

| Thing | Value |
|---|---|
| Repo | `/home/ubuntu/sirius`, branch `demo-multi-cn` (default branch is `dev`) |
| GPUs | 2× RTX PRO 6000 Blackwell. `nvidia-smi` reports 97887 MiB/card but **638 MiB is driver-reserved — only 94.97 GiB is allocatable** |
| Topology | 2 StarRocks compute nodes (CNs), one per GPU; 1 FE on MySQL port 9030 |
| Host | 48 cores, ~1.1 TiB RAM |
| StarRocks FE version | 4.1.1 (vendored at `experimental/starrocks/starrocks/`) |

### Datasets (verified to exist)

| Scale | Path |
|---|---|
| SF100 f64 | `/opt/dlami/nvme/tpch/tpch_parquet_sf100_f64` |
| SF300 f64 | `/opt/dlami/nvme/tpch/tpch_parquet_sf300_f64` |
| SF500 f64 | `/opt/dlami/nvme/tpch/tpch_parquet_sf500_f64` |

> **Correction to earlier notes:** the SF100 f64 dataset is **not** at `/home/ubuntu/tpch_parquet_sf100_f64` —
> that path does not exist. `/home/ubuntu/tpch_parquet_sf100` exists but is the **DECIMAL** build, not f64.
> Use the `/opt/dlami/nvme/tpch/` paths above, which is what `sweep-f64b.sh:7` already does.

### DuckDB oracles (verified: 22 `.tsv` each, correct scale)

| Scale | Oracle dir |
|---|---|
| SF100 | `/opt/dlami/nvme/sirius-build/oracle-f64/` |
| SF300 | `/opt/dlami/nvme/sirius-build/oracle-sf300f64/` |
| SF500 | `/opt/dlami/nvme/sirius-build/oracle-sf500f64/` |

### Tooling

| Path | What it is |
|---|---|
| `/opt/dlami/nvme/sirius-build/up-sf500-x.sh` | The **only** bring-up that can express operator budgets, the disk tier and downgrade fractions (it writes a full `--sirius-config` YAML). Every knob is an env var. Despite the name it is **dataset-agnostic** — it never sees `TPCH_DATA`, so the same cluster serves SF100/SF300/SF500. |
| `/opt/dlami/nvme/sirius-build/restart-sf500x.sh` | pkill + re-bring-up + `SET GLOBAL query_timeout`. Used as `RESTART_CMD`. |
| `/opt/dlami/nvme/sirius-build/sweep-sf500x.sh` | `bench.sh --cold` wrapper. **`TPCH_DATA` is hardcoded to SF500** (`sweep-sf500x.sh:7`), so it cannot be repointed by env — call `bench.sh` directly for other scales. |
| `/opt/dlami/nvme/sirius-build/sweep-sf500x-cold.sh` | `bench.sh --cold-restart` wrapper (fresh cluster per query). Same hardcoded `TPCH_DATA`. |
| `/home/ubuntu/sirius/experimental/starrocks/benchmarks/tpch/bench.sh` | The harness. Times and counts rows; **`bench.sh:54` states outright that it does not check answers.** |
| `/opt/dlami/nvme/sirius-build/compare.py` | The correctness gate. `compare.py <sirius_out_dir> <oracle_dir> [rel_tol]`. |
| `/opt/dlami/nvme/sirius-build/oracle.py` | Regenerates an oracle via DuckDB. Only needed if a query text changes. |

### The known-good configuration

```bash
GPU_MEM=60GiB  STAGING=32GiB  HOST_MEM=200GiB  NUM_CNS=2
HPB=1GiB   # hash_partition_bytes        (engine default here: 2.39 GiB)
MBHT=2GiB  # max_build_hash_table_bytes  (engine default here: 4.78 GiB)
STB=1GiB   # scan_task_batch_size
CBB=1GiB   # concat_batch_bytes
```

Current results with it: **SF100 22/22 correct, SF500 21/22 correct** (q09 is the only genuine failure).
SF300 is recorded as 21/22 but **predates the operator budgets** — that is Gap 1.

---

## 0.1 Gotchas that have already invalidated experiments — obey all of these

1. **Engine logs require `SIRIUS_LOG_BACKEND=spdlog`.** Only `duckdb`, `spdlog`, `noop` are accepted
   (`src/sirius_context.cpp:1549-1579`). On the CN path `install_configured_log_sink` is called with
   `db == nullptr` (`src/sirius_context.cpp:1587`), and the `throw` for an unknown backend is guarded by
   `else if (db)` (`:1575-1578`) — so **an unknown value is silently discarded and you get no logs at all**.
   Also set `SIRIUS_LOG_DIR` and `SIRIUS_LOG_LEVEL` (`:1583-1585`). `up-sf500-x.sh:30-32` already defaults
   these to `spdlog` / `.../siriuslog` / `info`.
2. **Both CNs append to ONE log file.** The sink is a daily-rotated `<SIRIUS_LOG_DIR>/sirius.log`
   (`src/log/spdlog_owning_sink.cpp:52-53`), i.e. `sirius_YYYY-MM-DD.log`, opened by both processes.
   Distinguish CNs by the `instance=0x...` field carried on `[window]`, `[gpu_pool]`, `[host_pool]` lines
   (`src/sirius_context.cpp:503`, `:254`, `:239`). **The arena teardown line has no `instance=` field** —
   see Gap 2.
   Because the filename is date-based, two runs on the same day **append to the same file**. Give every run
   its own `SIRIUS_LOG_DIR` (it propagates through `restart-sf500x.sh` into `up-sf500-x.sh:31`).
3. **`grep` fails silently on these logs** (invalid multibyte sequences). Use `grep -a` or `python3` with
   `open(p,'rb').read().decode('utf-8','replace')`.
4. **`restart-sf500x.sh:12` redirects with `>`, truncating `/tmp/cluster-sf500x.log` on every restart.**
   In `--cold-restart` mode that is 22 truncations — only the last bring-up's echoed header survives.
   Archive it per run, or assert the config another way (§0.2).
5. **A restart script that does not re-export `GPU_MEM`/`STAGING`/`HPB`/... silently runs the bring-up
   defaults.** `restart-sf500x.sh:10-11` re-exports them with `up-sf500-x.sh`'s defaults as fallbacks, so a
   sweep that forgot to export them keeps reporting under the *wrong* config with no error. This invalidated
   an experiment already. **Always assert the config actually used (§0.2).**
6. **The FE's `query_timeout` defaults to 300 s** and aborts server-side regardless of the client timeout.
   `restart-sf500x.sh:17-19` sets `SET GLOBAL query_timeout=1800`; if you bring the cluster up any other
   way, set it yourself.
7. **`bench.sh` has no correctness gate** (`bench.sh:54`). Always run `compare.py`.
8. **`bench.sh:175` files a correct-but-empty answer as a `wedge`** (`[ -s "$f" ]`). q11 at SF300/SF500 is
   *correct empty* — the oracle also returns 0 rows, because `queries/q11.sql` hardcodes the SF1 threshold
   `0.0001` where the spec scales it by `1/SF`. **Do not count q11 as a failure**; let `compare.py` decide.
9. **A cluster restart costs ~70 s** (`restart-sf500x.sh` = pkill + `sleep 8` + bring-up + `sleep 60`).
10. **The CN's query watchdog is 280 s** by default (`up-sf500-x.sh:27`). Raise
    `SIRIUS_QUERY_WATCHDOG_SECS` if a procedure expects a longer legitimate query.
11. **A cluster may already be live.** Check before doing anything:
    `pgrep -af 'sirius-starrocks-cn|StarRocksFE'`. Do not kill a cluster you did not bring up without
    confirming with the operator.

### 0.2 Asserting the config a run actually used

Three independent checks; do at least (a) and (b) after each cluster bring-up.

```bash
SR=/home/ubuntu/sirius/experimental/starrocks

# (a) The YAML the bring-up generated and handed to --sirius-config (regenerated every bring-up,
#     up-sf500-x.sh:59-90). This is ground truth for pool size, disk tier and operator budgets.
cat "$SR/.cn0-x.yaml"; cat "$SR/.cn1-x.yaml"

# (b) The arena capacity as the ENGINE saw it (survives log truncation, one line per CN per start).
grep -a 'exchange staging arena:' "$SIRIUS_LOG_DIR"/sirius_*.log | tail -4

# (c) The echoed header of the LAST bring-up only (truncated by every restart).
head -20 /tmp/cluster-sf500x.log
```

Expected in (a) for the known-good config: `usage_limit_bytes: "60GiB"`, a `disk:` block with
`downgrade_root_dirs`, `downgrade_trigger_fraction: 0.8`, `downgrade_stop_fraction: 0.6`, and an
`operator_params:` block with the four budgets. Expected in (b): `34359738368 bytes` (= 32 GiB).

---

# Gap 1 — SF300 was never re-run with the new operator budgets

## The question

SF300's 21/22 (`bench/rtxpro6000-2gpu/results/sf300-float64.csv`) was produced **before** the discovery
that `derived_default_batch_size()` (`src/sirius_config.cpp:38`) sizes every operator budget from
`prop.totalGlobalMem` — the *physical card* — rather than the configured pool, making every budget 2.4×
oversized at a 60 GiB pool. Correcting that took SF500's q08 from `bad_alloc` to a 17.6 s pass
(`results/sf500e5.csv`).

Also, that SF300 run was brought up through `up-sf300.sh` → `benchmarks/cluster8.sh`, which passes
`--gpu-memory-limit` (`cluster8.sh:74`) and **cannot express operator budgets at all**, has **no disk tier**,
and **does not set `SIRIUS_LOG_*`** — so that run was additionally unlogged.

> **Q1a.** With `HPB=1GiB MBHT=2GiB STB=1GiB CBB=1GiB`, is SF300 now 22/22?
> **Q1b.** Does SF100 regress under the same budgets? (It is 22/22 today; the budgets are a global change.)

The single sample of SF300's failure is `q11`, recorded as a `wedge` — which per gotcha 8 is most likely a
**correct empty result**, not a failure. So the honest prior is that SF300 may already be effectively 22/22
and the gap is partly a bookkeeping artifact. `compare.py` settles it.

### Scope note — stay on the f64 datasets; do not add a decimal arm

It is tempting to also re-run the DECIMAL datasets (`/opt/dlami/nvme/tpch/tpch_parquet_sf{100,300}`, and
`/home/ubuntu/tpch_parquet_sf{100,300,500}`) as an "exact vs inexact" contrast. **That contrast is not
clean.** In the plan translator, a DECIMAL with precision ≤ 18 maps to a Substrait `Decimal`, but
**precision 19–38 is lowered to `Fp64`** —
`experimental/starrocks/crates/starrocks-plan-translator/src/type_mapper.rs:233-245`. TPC-H base columns are
`DECIMAL(15,2)` and stay decimal, but the FE widens arithmetic like
`l_extendedprice * (1 - l_discount)` past precision 18, so **the derived values the queries actually
aggregate become doubles in the engine either way**. This is the same lowering behind q09's known ~0.147 %
low bias. Comparing a decimal run against an f64 run therefore measures the *widening threshold*, not
exact-vs-inexact arithmetic, and would produce a misleading result. **Gap 1 runs f64 only.** If a decimal
arm is wanted later it needs its own plan with the lowering threshold as the explicit subject.

## Procedure

One cluster serves both scales (the bring-up never sees `TPCH_DATA`).

```bash
set -a; source /opt/dlami/nvme/sirius-build/env.sh; set +a
SR=/home/ubuntu/sirius/experimental/starrocks
export PATH="$SR/.pixi/envs/default/bin:$PATH"
RUNID=$(date +%m%d-%H%M)
export SIRIUS_LOG_DIR=/opt/dlami/nvme/sirius-build/siriuslog/g1-$RUNID
mkdir -p "$SIRIUS_LOG_DIR"

# The config under test. Exported so restart-sf500x.sh (RESTART_CMD) propagates it — see gotcha 5.
export GPU_MEM=60GiB STAGING=32GiB HOST_MEM=200GiB NUM_CNS=2
export HPB=1GiB MBHT=2GiB STB=1GiB CBB=1GiB
export SIRIUS_LOG_BACKEND=spdlog SIRIUS_LOG_LEVEL=info

# Step 1 — bring up (~70 s), then ASSERT the config (§0.2). Do not skip the assert.
nohup /opt/dlami/nvme/sirius-build/up-sf500-x.sh > /tmp/cluster-g1-$RUNID.log 2>&1 &
sleep 70
head -20 /tmp/cluster-g1-$RUNID.log
cat "$SR/.cn0-x.yaml"
mysql -h127.0.0.1 -P9030 -uroot -e "SET GLOBAL query_timeout=1800; SHOW COMPUTE NODES\G" | head -30

# Step 2 — SF100 regression sweep (cold + 2 warm, one cluster, restart only on failure)
export QUERY_TIMEOUT=180 COLD_TIMEOUT=240 MIN_BACKENDS=2
export RESTART_CMD=/opt/dlami/nvme/sirius-build/restart-sf500x.sh
export TPCH_DATA=/opt/dlami/nvme/tpch/tpch_parquet_sf100_f64
"$SR/benchmarks/tpch/bench.sh" --cold /opt/dlami/nvme/sirius-build/bench/G1-SF100/timings.csv 2
python3 /opt/dlami/nvme/sirius-build/compare.py \
  /opt/dlami/nvme/sirius-build/bench/G1-SF100 /opt/dlami/nvme/sirius-build/oracle-f64

# Step 3 — SF300 sweep
export QUERY_TIMEOUT=300 COLD_TIMEOUT=420
export TPCH_DATA=/opt/dlami/nvme/tpch/tpch_parquet_sf300_f64
"$SR/benchmarks/tpch/bench.sh" --cold /opt/dlami/nvme/sirius-build/bench/G1-SF300/timings.csv 2
python3 /opt/dlami/nvme/sirius-build/compare.py \
  /opt/dlami/nvme/sirius-build/bench/G1-SF300 /opt/dlami/nvme/sirius-build/oracle-sf300f64
```

**Escalation (only if a query fails in step 3).** A single-cluster sweep leaves pool retention as a
confounder — measured: one CN sat at 48 GiB of a 60 GiB pool for six consecutive query windows. Re-run only
the failing queries with a fresh cluster per query:

```bash
export QUERY_TIMEOUT=900 COLD_TIMEOUT=900 FE_QUERY_TIMEOUT=1800
"$SR/benchmarks/tpch/bench.sh" --cold-restart \
  /opt/dlami/nvme/sirius-build/bench/G1-SF300-COLD/timings.csv 1 q11 q21   # <- the failing set
```

## What to record

- `results/g1-sf100-budgets.csv` and `results/g1-sf300-budgets.csv` (copies of the two `timings.csv`).
- The full `compare.py` table for each, verbatim, including `maxreldiff` per query.
- The `.cn0-x.yaml` used, pasted into the results note (proof of config, gotcha 5).
- A per-query delta table vs the pre-budget baselines: `results/sf100-float64-q10fixed.csv` and
  `results/sf300-float64.csv`. Warm median per query, and % change.
- For any query still failing: its `q*.r*.out` first line and the matching engine-log window.

## Success criterion

1. **Answered, either way** — the plan succeeds if Q1a and Q1b have measured answers, not if they are green.
2. **SF100 no regression:** `compare.py` reports **22/22 MATCH**, and no warm-median query is >15 % slower
   than `results/sf100-float64-q10fixed.csv`. Any single-query regression >15 % is a finding to record, not
   a failure of the plan.
3. **SF300:** `compare.py` reports **22/22 MATCH**, or the residual failures are enumerated with their
   `compare.py` verdict (`NO-RESULT`/`ERROR`/`ROWS-DIFFER`) and the engine-log evidence. If q11 comes back
   `MATCH` with 0 rows, SF300 is **22/22** and the old "21/22" was gotcha 8.

## Cost

| Item | Time |
|---|---|
| Bring-up + config assert | ~2 min |
| SF100 sweep (22 q × 3 runs) | ~8 min |
| SF300 sweep (22 q × 3 runs) | ~12 min |
| `compare.py` ×2 + write-up | ~10 min |
| **Subtotal** | **~35 min** |
| Escalation to `--cold-restart` for k queries | + k × ~80 s |

---

# Gap 2 — Arena high-water is largely unmeasured

## The question

`exchange_staging_arena` exposes `peak_live_bytes()`, `live_bytes()`, `total_free()`, `largest_free()`
(`src/include/exec/exchange_staging_arena.hpp:110-120`; impl `src/exec/exchange_staging_arena.cpp:294-336`;
`peak_live_bytes_` is maintained on every successful lease at `:237-238`). But the **only** place any of them
is printed is the destructor (`src/exec/exchange_staging_arena.cpp:159-176`, emitted line at `:168`):

```
exchange staging arena: peak live 28750023168 of 34359738368 bytes (0 leases outstanding, 1 free blocks, largest 34359738368)
```

**That line only fires on a clean C++ teardown.** Measured on the existing corpus
(`/opt/dlami/nvme/sirius-build/siriuslog/sirius_2026-08-19.log`):

| Metric | Count |
|---|---|
| Arena constructions (`exchange_staging_arena.cpp:79`) | **96** |
| Arena teardown lines (`:168`) | **48** |
| **Lifetimes that measured nothing** | **48 / 96 = 50 %** |

(Earlier notes quote 61 %; the exact figure depends on which logs are in the corpus. Either way **at least
half** of all process lifetimes produced no number, and it is not a random half — it is biased toward the
runs that died, i.e. exactly the pathological ones. This survivorship bias already produced one wrong
conclusion this session.)

### Why the teardown line is missed — root cause, verified

The CN installs a SIGTERM handler (`experimental/starrocks/src/main.rs:638-649`) and then spawns an
**escalation task**: after `SHUTDOWN_GRACE = 15 s` (`main.rs:31-34`) it calls `std::process::exit(1)`
(`main.rs:672-686`), whose own comment says *"GPU memory, the staging arena, and UCX resources are reclaimed
by the driver at process death"*. `std::process::exit` does **not** run the arena's destructor. And
`restart-sf500x.sh:4-6` pkills then sleeps only **8 s** — shorter than the 15 s grace — so the next bring-up
starts while the old CNs may still be in teardown.

So: any CN whose engine thread is wedged inside a fragment (i.e. exactly the runs that failed) force-exits
and loses the measurement.

### Known numbers (the entire corpus)

| Case | Arena peak |
|---|---|
| SF100 full sweep | **6.51 GiB** of 32 |
| SF500 healthy max | **26.78 GiB** (q18 18.68, q17 16.06) |
| SF500 pathological (starved 45 GiB pool) | **47.40 GiB** of 48 |
| **SF300** | **no measurement at all** |

The 1.8× spread between the last two is why `STAGING ≈ 96 GiB × SF/500 / N` was retired: the arena is a
**pressure gauge for the pool**, not independent demand — `Fragment::push_packed` deep-copies arena→pool
(`src/sirius_ffi.cpp:849`) and only then releases the lease (`experimental/starrocks/src/engine.rs:563`), so
a pool at its ceiling stalls the drain and the arena ratchets to capacity.

> **Q2a.** What is SF300's arena high-water under the known-good config?
> **Q2b.** What is the per-query arena high-water distribution, rather than one number per process lifetime?
> **Q2c.** Can the number be made to survive a SIGKILL / force-exit?

## Procedure A — clean shutdown (no code change; rides on Gap 1's cluster)

The fix is to give the CN longer than `SHUTDOWN_GRACE` before the next bring-up starts. Write a graceful
restart wrapper (a **new** file — do not edit the shared `restart-sf500x.sh`):

```bash
cat > /opt/dlami/nvme/sirius-build/restart-sf500x-graceful.sh <<'EOF'
#!/usr/bin/env bash
# Same as restart-sf500x.sh but waits past the CN's SHUTDOWN_GRACE (15 s, src/main.rs:34) so the
# exchange_staging_arena destructor runs and its "peak live" line reaches the log
# (src/exec/exchange_staging_arena.cpp:159-176). Without this, a SIGTERMed-then-force-exited CN
# measures nothing.
set -uo pipefail
pkill -f 'target/release/sirius-starrocks-cn' 2>/dev/null
pkill -f 'com.starrocks.StarRocksFE' 2>/dev/null
sleep 22
export GPU_MEM=${GPU_MEM:-60GiB} STAGING=${STAGING:-32GiB} HOST_MEM=${HOST_MEM:-200GiB}
export HPB=${HPB:-} MBHT=${MBHT:-} STB=${STB:-} CBB=${CBB:-} MSPB=${MSPB:-}
nohup /opt/dlami/nvme/sirius-build/up-sf500-x.sh >> "${CLUSTER_LOG:-/tmp/cluster-sf500x.log}" 2>&1 &
sleep 60
export PATH=/home/ubuntu/sirius/experimental/starrocks/.pixi/envs/default/bin:$PATH
mysql --host 127.0.0.1 --port 9030 --user root --connect-timeout=5 \
  -e "SET GLOBAL query_timeout = ${FE_QUERY_TIMEOUT:-1800};" 2>/dev/null \
  && echo "FE query_timeout set to ${FE_QUERY_TIMEOUT:-1800}s"
EOF
chmod +x /opt/dlami/nvme/sirius-build/restart-sf500x-graceful.sh
```

Note it also uses `>>` and an overridable `$CLUSTER_LOG`, fixing gotcha 4 for this run.

**A1 — one number per sweep (free; do this at the end of Gap 1).** After each Gap 1 sweep finishes, shut the
cluster down gracefully and read the two teardown lines:

```bash
pkill -f 'target/release/sirius-starrocks-cn'; sleep 25
grep -a 'peak live' "$SIRIUS_LOG_DIR"/sirius_*.log | tail -4
```

**A2 — one number per query (SF300, `--cold-restart`).** Each query gets a fresh cluster, so each query
contributes one teardown line per CN:

```bash
export RESTART_CMD=/opt/dlami/nvme/sirius-build/restart-sf500x-graceful.sh
export CLUSTER_LOG=/tmp/cluster-g2-$RUNID.log
export TPCH_DATA=/opt/dlami/nvme/tpch/tpch_parquet_sf300_f64
export QUERY_TIMEOUT=900 COLD_TIMEOUT=900 FE_QUERY_TIMEOUT=1800 MIN_BACKENDS=2
"$SR/benchmarks/tpch/bench.sh" --cold-restart \
  /opt/dlami/nvme/sirius-build/bench/G2-SF300COLD/timings.csv 1
```

Attribution: the teardown lines carry **no `instance=`**, so pair them to queries by timestamp against the
`[window]` lines (`src/sirius_context.cpp:503`), which do carry `instance=`, `window=` and `query=`.
The two lines emitted within ~50 ms of each other are the two CNs of one restart, and they belong to the
**query that ran just before** that restart.

```python
# Correlator — run with python3 (grep is unsafe here, gotcha 3).
import re, glob
lines = []
for p in glob.glob('/opt/dlami/nvme/sirius-build/siriuslog/g2-*/sirius_*.log'):
    lines += open(p,'rb').read().decode('utf-8','replace').split('\n')
lines.sort()                       # timestamps are the line prefix, so this is chronological
last_sql = None
for ln in lines:
    m = re.search(r'peak live (\d+) of (\d+) bytes .*?(\d+) free blocks, largest (\d+)', ln)
    if m:
        peak, cap = int(m.group(1)), int(m.group(2))
        print(f"{ln[1:20]}  peak={peak/2**30:7.2f} GiB / {cap/2**30:5.1f} GiB "
              f"({100*peak/cap:5.1f}%)  after: {last_sql}")
    elif '[window] begin' in ln:
        last_sql = ln.strip()[-90:]
```

## Procedure B — make the number survive a kill (code change)

**Design (preferred, smallest blast radius that still lands the number in the shared log):**

The arena is owned by the FFI context, not by `SiriusContext`:
`detail::context_state::staging_arena` (`src/sirius_ffi.cpp:161`), created in `bring_up()` at
`src/sirius_ffi.cpp:228`. `SiriusContext::log_pool_stats` (`src/sirius_context.cpp:229-261`) — which already
emits `[host_pool]` (`:239`) and `[gpu_pool]` (`:254`) — has no handle on it. Give it one:

1. Add to `SiriusContext` a `std::weak_ptr<sirius::exec::exchange_staging_arena> staging_arena_` plus a
   setter.
2. Call the setter from `src/sirius_ffi.cpp` immediately after line 228 (`from_env()`), i.e. after the arena
   exists.
3. In `log_pool_stats`, after the `[gpu_pool]` loop, add:
   ```
   SIRIUS_LOG_INFO("[arena] {} peak_live={} live={} capacity={} total_free={} largest_free={} outstanding={}",
                   tag, a->peak_live_bytes(), a->live_bytes(), a->capacity(),
                   a->total_free(), a->largest_free(), a->outstanding());
   ```
   guarded by `if (auto a = staging_arena_.lock())`.

`log_pool_stats` is already called at window begin (`src/sirius_context.cpp:371`) and window end (`:448`),
and the `tag` it receives already contains `instance=%p connection=… window=… query=…` (built at
`src/sirius_context.cpp:528-533`). So this **simultaneously** fixes three things: the number survives a kill,
it is per-window instead of per-process, and it becomes attributable to a CN and a query — which the
teardown line at `:168` is not.

**Alternative (if touching `SiriusContext` is unwanted):** add `peak_live_bytes()/live_bytes()/total_free()/
largest_free()` to the FFI `StagingArena` wrapper next to the existing `outstanding()`
(`src/include/sirius_ffi.hpp:141-172`, impl `src/sirius_ffi.cpp:343-358`) and log them from the Rust CN at
fragment end. More moving parts (cxx bridge + a Rust call site) for the same information, and it does not
share the `[gpu_pool]` line's key.

**Do not** rely on an `atexit`/signal handler: `main.rs:672-686` deliberately calls `std::process::exit(1)`,
and a SIGKILL bypasses everything regardless.

**Cost warning:** this touches `src/`, so it needs a full engine rebuild (`pixi run make`) and a CN relink
before any measurement — budget 1–2 h of build alone. Volume added: the existing corpus has ~970 windows in
~1.7 h, so one extra `[arena]` line per window begin/end is ~1940 lines — negligible.

## What to record

- `results/g2-arena-highwater.md`: a table of `(scale, query, CN, peak GiB, capacity GiB, % full,
  free blocks, largest free)` from the correlator above.
- Explicitly: the **SF300 arena high-water**, which currently does not exist at any value.
- The **coverage ratio** for the run: teardown lines ÷ arena constructions. State it. If it is not 1.0, say
  which queries are missing and why (force-exit vs pkill-too-early).
- Re-evaluate SF500's `STAGING=32GiB` against the diagnostic rule below and state whether 32 GiB is right,
  oversized, or masking a pool problem.

**Diagnostic rule for `arena exhausted` (from the exhaustion message at
`src/exec/exchange_staging_arena.cpp:243-256`, which reports both totals for exactly this reason):**

| Reading | Meaning | Action |
|---|---|---|
| arena >90 % full **and** pool peak == cap | pathological | **do not raise the arena — fix the pool** |
| arena >90 % full, pool under cap | real demand | raise the arena |
| arena <70 %, `largest_free < request ≤ total_free` | external fragmentation | a bigger arena may not help |

## Success criterion

1. **A number exists for SF300** where none did.
2. **Coverage ≥ 90 %** of arena lifetimes in the Gap 2 run emit a high-water (vs the measured 50 % baseline).
   Procedure A alone should reach this; if it does not, that is itself the finding and Procedure B is
   justified.
3. Per-query attribution for at least the SF300 sweep: every query maps to a peak, or is listed as unmapped
   with the reason.
4. If Procedure B is executed: a `[arena]` line appears at window begin and end, carries `instance=`, and
   a **deliberately SIGKILLed** CN still leaves a high-water in the log from its last completed window.

## Cost

| Item | Time |
|---|---|
| A1 (rides on Gap 1, graceful shutdown + parse) | ~10 min |
| A2 (SF300 `--cold-restart`, 22 × ~92 s restart + queries) | ~40 min |
| Correlation + write-up | ~20 min |
| **Subtotal, procedure A** | **~70 min** |
| Procedure B: edit + `pixi run make` + CN rebuild + re-verify | **+1.5–3 h** (build-dominated) |

---

# Gap 3 — A disk spill has never been confirmed to occur

## The question

A disk tier **is** configured and accepted. `up-sf500-x.sh:74-82` emits a `disk:` block with
`disk_id: 0`, `capacity_bytes: 2000GB`, `downgrade_root_dirs: "$DISK/cn$i"` (default
`/opt/dlami/nvme/sirius_spill`), and the bring-up warning that fires when no DISK memory space exists
(`src/sirius_context.cpp:664-671`) is **absent from the logs** — verified: 0 occurrences in
`sirius_2026-08-19.log`. Downgrade fractions default to trigger 0.8 / stop 0.6
(`src/sirius_config.cpp:308-309`) and `up-sf500-x.sh:69-70` sets them explicitly from `$DGT`/`$DGS`.

But **successful downgrades log at DEBUG.** `src/downgrade/downgrade_executor.cpp` has exactly one
per-request summary, at DEBUG (`:377-394`):

```
[downgrade] [{}] request {}done: {} batches, {} bytes in {:.2f} ms ({:.1f} MB/s) |
repos: {}/{} batches/bytes, pipeline_queue: {}/{} | to_host: {}/{} batches/bytes, to_disk: {}/{} batches/bytes
```

That line is the **only** place `to_host` vs `to_disk` is broken out. Every run so far was at INFO, so the
absence of "downgrade" activity in the logs proves nothing.

Worse, **with a disk tier configured both WARN paths in the executor are unreachable by construction:**

- `:356-362` requires `disk_not_configured` (`:197-198`) to be true.
- `:477-484` ("memory pressure but no viable downgrade target") requires
  `has_viable_downgrade_target()` to be false — but that function returns `true` immediately when
  `has_disk_tier()` (`:405-411`, with `has_disk_tier()` at `:400-403`).

So at INFO with a disk tier, the downgrade subsystem is **completely silent whether it works or not**.

### What *is* observable at INFO, and what it says

The pipeline executor's caller-side view is a WARN and it fired. `gpu_pipeline_executor.cpp:226` calls
`request_downgrade(...).get()`; `:263-271` warns when the post-downgrade reservation is still short:

```
GPU Pipeline Executor: after downgrade (0 bytes freed), reservation still partial (N/M bytes) for pipeline P task T -- proceeding with partial reservation
```

Measured over `sirius_2026-08-19.log`, 20:25:22 → 21:29:28:

| Samples | Nonzero `bytes freed` | Max |
|---|---|---|
| **356** | **0** | **0** |

**Every single on-demand downgrade freed exactly zero bytes.** That is consistent with the known cause: the
downgrade sweep enumerates only the per-query data-repository registry
(`src/downgrade/downgrade_executor.cpp:200-240`) plus the pipeline task queue, and parked exchange sender
outputs are by construction outside both.

### Correcting one claim in the source

`src/sirius_context.cpp:807` says *"HOST->DISK downgrade is not yet implemented, so we skip HOST tier for
now."* **That comment is stale.** Two lines below, `create_executors_for_tier` is called for **both** tiers
(`:838-839`), and in `downgrade_executor::processing_loop` the target list for a **HOST** source is built as
"no HOST targets, then all DISK spaces" (`:174-198`) — i.e. HOST→DISK *is* wired. Whether it *works* is the
open question; the comment is not evidence that it does not.

Also note the spill root is empty: `/opt/dlami/nvme/sirius_spill/{cn0,cn1}` contain 0 files, 12 K total.
That is suggestive but not conclusive — a spill file could be created and unlinked within a query.

> **Q3a.** Does GPU→HOST downgrade ever move bytes? (`to_host` nonzero in the DEBUG line.)
> **Q3b.** Does GPU→DISK ever move bytes? (`to_disk` nonzero, and/or bytes appear under the spill root.)
> **Q3c.** Does HOST→DISK ever fire, given the HOST executor exists (`:839`)?
> **Q3d.** Is the disk tier reachable in practice, or only configurable?

## Procedure

**Step 0 — pick the target queries (free, no cluster).** Find which queries produced the 356 zero-byte
downgrades, so DEBUG is scoped to a handful instead of a whole sweep:

```python
import re, glob
lines=[]
for p in glob.glob('/opt/dlami/nvme/sirius-build/siriuslog/sirius_*.log'):
    lines += open(p,'rb').read().decode('utf-8','replace').split('\n')
lines.sort()
cur=None; hits={}
for ln in lines:
    if '[window] begin' in ln: cur=ln.strip()[-90:]
    if 'after downgrade' in ln: hits[cur]=hits.get(cur,0)+1
for k,v in sorted(hits.items(), key=lambda x:-x[1])[:10]: print(v,k)
```

Cross-reference the window ids against the SQL lines the engine logs at QueryBegin — the query text is
whitespace-normalized at `src/sirius_context.cpp:288-306` and logged in full (no truncation) at
`src/sirius_context.cpp:312-318` as `QueryBegin: instance=… connection=… query=… SQL: …`. Expect the SF500
pressure cases (q05, q07, q17, q18, q21) to dominate.

**Step 1 — dedicated cluster at DEBUG.** DEBUG is verbose (177 non-legacy `SIRIUS_LOG_DEBUG` call sites in
`src/`, several on per-batch paths). Scope it to the 3–4 queries from step 0 and watch the log size.

```bash
set -a; source /opt/dlami/nvme/sirius-build/env.sh; set +a
SR=/home/ubuntu/sirius/experimental/starrocks
export PATH="$SR/.pixi/envs/default/bin:$PATH"
RUNID=$(date +%m%d-%H%M)
export SIRIUS_LOG_DIR=/opt/dlami/nvme/sirius-build/siriuslog/g3-$RUNID
mkdir -p "$SIRIUS_LOG_DIR"
export SIRIUS_LOG_BACKEND=spdlog SIRIUS_LOG_LEVEL=debug     # <-- the point of this gap
export GPU_MEM=60GiB STAGING=32GiB HOST_MEM=200GiB NUM_CNS=2
export HPB=1GiB MBHT=2GiB STB=1GiB CBB=1GiB
export DISK=/opt/dlami/nvme/sirius_spill
nohup /opt/dlami/nvme/sirius-build/up-sf500-x.sh > /tmp/cluster-g3-$RUNID.log 2>&1 &
sleep 70
head -20 /tmp/cluster-g3-$RUNID.log; cat "$SR/.cn0-x.yaml"          # assert disk: block present
grep -ac 'disk memory space is not configured' "$SIRIUS_LOG_DIR"/sirius_*.log   # MUST print 0
mysql -h127.0.0.1 -P9030 -uroot -e "SET GLOBAL query_timeout=1800;"
```

**Step 2 — sample the spill root while queries run** (read-only; start before step 3):

```bash
( while true; do
    printf '%s %s\n' "$(date +%T)" \
      "$(du -sb /opt/dlami/nvme/sirius_spill/cn0 /opt/dlami/nvme/sirius_spill/cn1 | tr '\n' ' ')"
    ls -laR /opt/dlami/nvme/sirius_spill >> /tmp/g3-spill-listing.txt
    sleep 2
  done ) > /tmp/g3-spill-du.txt 2>&1 &
SPILL_WATCH=$!
```

**Step 3 — run the pressure queries at SF500.**

```bash
export TPCH_DATA=/opt/dlami/nvme/tpch/tpch_parquet_sf500_f64
export QUERY_TIMEOUT=900 COLD_TIMEOUT=900 MIN_BACKENDS=2
export RESTART_CMD=/opt/dlami/nvme/sirius-build/restart-sf500x.sh
"$SR/benchmarks/tpch/bench.sh" --cold \
  /opt/dlami/nvme/sirius-build/bench/G3-DEBUG/timings.csv 1 q18 q17 q21 q07
kill $SPILL_WATCH
du -sh "$SIRIUS_LOG_DIR"                     # sanity: how much DEBUG cost in bytes
```

**Step 4 — the actual measurement.**

```python
import re, glob, collections
lines=[]
for p in glob.glob('/opt/dlami/nvme/sirius-build/siriuslog/g3-*/sirius_*.log'):
    lines += open(p,'rb').read().decode('utf-8','replace').split('\n')
pat = re.compile(r'\[downgrade\] \[([^\]]+)\] request (monitor )?done: (\d+) batches, (\d+) bytes'
                 r'.*?to_host: (\d+)/(\d+) batches/bytes, to_disk: (\d+)/(\d+) batches/bytes')
agg=collections.Counter(); n=0
for ln in lines:
    m=pat.search(ln)
    if not m: continue
    n+=1
    agg['batches']+=int(m.group(3)); agg['bytes']+=int(m.group(4))
    agg['to_host_batches']+=int(m.group(5)); agg['to_host_bytes']+=int(m.group(6))
    agg['to_disk_batches']+=int(m.group(7)); agg['to_disk_bytes']+=int(m.group(8))
    agg['monitor' if m.group(2) else 'ondemand']+=1
print("downgrade request summaries:", n); print(agg)
print("caller-side zero-freed WARNs:",
      sum(1 for l in lines if 'after downgrade' in l))
```

## What to record

- `results/g3-downgrade-evidence.md` containing:
  - the exact count of `[downgrade] ... request done` summaries seen (if **0**, the executor never ran a
    request — a completely different finding from "ran and freed nothing");
  - totals for `to_host` and `to_disk` batches/bytes, split monitor vs on-demand;
  - the caller-side `after downgrade (N bytes freed)` distribution (baseline: 356 samples, all 0);
  - max observed size under `/opt/dlami/nvme/sirius_spill` from `/tmp/g3-spill-du.txt`, and whether any file
    was ever seen in `/tmp/g3-spill-listing.txt`;
  - the DEBUG log size, so the next person can budget it.
- A one-line verdict on each of Q3a–Q3d.
- If `to_disk` is 0 everywhere: whether HOST reservation succeeded (making DISK never reached, since DISK
  spaces are appended **after** HOST in the target list, `downgrade_executor.cpp:174-198`) or HOST itself
  freed nothing.
- A note that `src/sirius_context.cpp:807`'s "HOST→DISK ... not yet implemented" comment is stale
  (contradicted by `:838-839`), so it can be corrected under a separate change.

## Success criterion

Each of Q3a–Q3d has a **yes/no backed by a log line or a byte count**, and the evidence is filed. Concretely:

1. At least one `[downgrade] ... request done` summary is captured at DEBUG — proving the instrumentation
   path works and the run is not silently blind again (this is the pass/fail gate for the *procedure*).
2. `to_host_bytes` and `to_disk_bytes` totals are reported, even if zero.
3. A definite answer on the disk tier: either bytes appeared under the spill root / `to_disk_bytes > 0`
   (**disk is reachable**), or both are zero across queries that provably applied memory pressure
   (**disk is configurable but not reachable in practice** — which is a finding worth its own follow-up).

## Cost

| Item | Time |
|---|---|
| Step 0, log mining (no cluster) | ~15 min |
| Bring-up + asserts | ~3 min |
| 4 SF500 queries × (cold + 1 warm), incl. q07 which can run ~290 s | ~20 min |
| Parse + write-up | ~20 min |
| **Subtotal** | **~60 min** |

Needs **its own cluster** (different `SIRIUS_LOG_LEVEL`, own `SIRIUS_LOG_DIR`).

---

# Gap 4 — q08/q09 still rely on hand-edited query text

## The question

`experimental/starrocks/benchmarks/tpch/QUERY-DEVIATIONS.md` records the only deviation from stock TPC-H
text: q08 and q09 have their `FROM` clause reordered so `lineitem` sits between `part` and `supplier`.
Current text: `benchmarks/tpch/queries/q08.sql` (`part, lineitem, supplier, orders, customer, nation n1,
nation n2, region`) and `q09.sql` (`part, lineitem, supplier, partsupp, orders, nation`).

Root cause per that doc: `FILES()` scans carry no statistics, the FE estimates `cardinality: 1` everywhere
(see `benchmarks/tpch/plans/q08.verbose.txt`), and the CBO emitted a `part × supplier` **CROSS JOIN** —
q08: 4 × 82,851 × 1,000,000 = 331,404,000,000 rows; q09: 4 × 673,651 × 1,000,000 = 2,694,604,000,000.
The reorder is semantically identical (inner joins commute) and it works, but it is **not TPC-H-faithful**,
and any A/B against stock StarRocks must use the same text or the comparison is invalid.

> **Q4.** Can real statistics be supplied for parquet read through `FILES()` — via `ANALYZE`, an external
> table with statistics, or a manual injection — such that the CBO picks a sane join order from the **stock**
> query text, unaided?

**This is a scoping/feasibility investigation, not a benchmark run.** No GPU work is required; `EXPLAIN` and
`EXPLAIN COSTS` are FE-only and do not execute.

## What the FE source already says (verified — start from here, do not re-derive)

The answer is largely determined by one function. `StatisticsCalculator.visitLogicalTableFunctionTableScan`
(`experimental/starrocks/starrocks/fe/fe-core/src/main/java/com/starrocks/sql/optimizer/statistics/StatisticsCalculator.java:373-376`)
delegates to `computeFileScanNode` (**same file, `:664-676`**), which is, verbatim:

```java
// Use default statistics for now.
Statistics.Builder builder = Statistics.builder();
for (ColumnRefOperator columnRefOperator : columnRefOperatorColumnMap.keySet()) {
    builder.addColumnStatistic(columnRefOperator, ColumnStatistic.unknown());
}
// cause we don't know the real schema in file，just use the default Row Count now
builder.setOutputRowCount(1);
```

**Row count 1 and `unknown()` column stats are hardcoded. No statistics storage is consulted on this path at
all.** Contrast `computePaimonScanNode` (same file, `:690-704`), which *does* call
`MetadataMgr.getTableStatistics`.

Consequences, each verified:

- **`ANALYZE` over `FILES()` is not expressible.** `AnalyzeStmt` carries a `TableRef`
  (`fe/fe-parser/.../sql/ast/AnalyzeStmt.java:26-46`) and the analyzer resolves it with
  `MetaUtils.getSessionAwareTable(session, db, tableName)`
  (`fe-core/.../sql/analyzer/AnalyzeStmtAnalyzer.java:123, :252, :298`) — a catalog-qualified name. A table
  function has no name. And even if it did, `computeFileScanNode` would ignore the result.
- **Injected stats cannot reach this path** for the same reason: nothing reads them.
- **Analyzable external tables are an enumerated set:** `HIVE, ICEBERG, HUDI, ODPS, DELTALAKE, PAIMON`
  (`fe-core/.../catalog/Table.java:157-165`, gate at `:340-342`). A plain parquet directory is none of them.
- **`FILES()` has no statistics-bearing property.** The full property list is at
  `fe-core/.../catalog/TableFunctionTable.java:118-147` — `path`, `format`, `compression`,
  `auto_detect_sample_files/rows/types`, CSV/parquet writer options, `list_files_only`, `list_recursively`.
  There is no `row_count`, `cardinality` or `statistics` key.
- **But the FE already holds per-file sizes:** `private List<TBrokerFileStatus> fileStatuses`
  (`TableFunctionTable.java:188`, populated during file listing around `:300-335`, each with a `size`).
  A byte-size-derived row-count estimate is therefore available **at plan time with no new I/O**.

Session variables worth checking but unlikely to help: `disable_join_reorder`
(`fe-core/.../qe/SessionVariable.java:430, :1781`), `cbo_enable_greedy_join_reorder` (`:486`),
`cbo_max_reorder_node_use_exhaustive` (`:483`). Per QUERY-DEVIATIONS.md the CBO with `cardinality: 1`
everywhere already *follows the written order* — so disabling reorder is expected to change nothing. Confirm
rather than assume.

## Procedure

Steps 1–3 need no cluster. Step 4 needs only a live FE (reuse Gap 1's or Gap 3's cluster).

1. **Confirm the source findings above by reading**, and record any that differ (this doc's citations were
   taken from the vendored 4.1.1 tree; verify against the tree in front of you).
2. **Enumerate the option space** and mark each Feasible / Infeasible / Requires-FE-patch, with the
   file:line that decides it:
   - (a) `ANALYZE TABLE FILES(...)` → expected **Infeasible** (no table name; and `computeFileScanNode`
     ignores stats).
   - (b) `ANALYZE` on an external catalog table over the same parquet → the **only** route the existing
     statistics machinery supports. Requires the parquet to be an `IS_ANALYZABLE_EXTERNAL_TABLE` type. For
     Hive that needs a metastore; for **Iceberg a filesystem/hadoop catalog needs no metastore**, but
     registering existing parquet as Iceberg requires metadata generation. **Judge explicitly whether that
     breaks the benchmark's "read parquet directly, no load" property** — that property is the reason
     `FILES()` is used at all.
   - (c) Manual injection into the FE's statistics tables → expected **Infeasible** on this path
     (`computeFileScanNode` reads nothing).
   - (d) **FE patch: derive `setOutputRowCount` from `fileStatuses` byte sizes** (or from parquet footer
     `num_rows`) instead of the literal `1`, at
     `StatisticsCalculator.java:664-676`. Estimate the size of this change and whether the FE build is
     already reproducible on this box (`starrocks/output/fe/` is prebuilt — check whether the FE can be
     rebuilt here at all before recommending this).
   - (e) Query hints / session variables → verify with step 4, expected to be a no-op.
   - (f) Status quo: keep the reorder, keep it documented. This is the fallback and it is legitimate.
3. **Write the recommendation** with an effort estimate for the chosen option.
4. **Empirical check (FE only, no GPU, safe on a live cluster):** run the **stock** q08/q09 text through the
   planner without executing and confirm the cross join reappears and the estimates are `cardinality: 1`.

```bash
SR=/home/ubuntu/sirius/experimental/starrocks
export PATH="$SR/.pixi/envs/default/bin:$PATH"
D=/opt/dlami/nvme/tpch/tpch_parquet_sf100_f64
mkdir -p /tmp/g4 && cd /tmp/g4

# Stock FROM order for q08: part, supplier, lineitem, orders, customer, nation n1, nation n2, region.
# Build it by swapping lines back in a COPY of the query; never edit the file in benchmarks/.
sed 's|__TPCH_DATA__|'"$D"'|g' "$SR/benchmarks/tpch/queries/q08.sql" > q08.reordered.sql
python3 - <<'EOF'
s=open('/tmp/g4/q08.reordered.sql').read()
open('/tmp/g4/q08.stock.sql','w').write(s.replace("        part,\n        lineitem,\n        supplier,\n",
                                                  "        part,\n        supplier,\n        lineitem,\n"))
EOF

# Prefix the EXPLAIN mode and feed the whole thing on stdin. Do NOT use `mysql -e "$(cat ...)"`
# with these queries: bench.sh collapses newlines with -e, which is exactly why QUERY-DEVIATIONS.md
# forbids `--` comments in the .sql files. Reading from stdin preserves the line structure.
for v in stock reordered; do
  for mode in "EXPLAIN" "EXPLAIN COSTS" "EXPLAIN VERBOSE"; do
    tag=${mode// /_}
    { printf '%s\n' "$mode"; cat "q08.$v.sql"; } > "q08.$v.$tag.in"
    mysql -h127.0.0.1 -P9030 -uroot -B < "q08.$v.$tag.in" > "q08.$v.$tag.txt" 2>&1
  done
done
grep -ac 'CROSS JOIN' q08.stock.EXPLAIN.txt q08.reordered.EXPLAIN.txt
grep -a 'cardinality' q08.stock.EXPLAIN_COSTS.txt | sort | uniq -c | head
```

Sanity-check first that `q08.stock.sql` actually differs from `q08.reordered.sql` in the expected place —
if the `python3` replace found no match (e.g. the query text has since changed) it silently writes an
identical file and the whole comparison is vacuous:

```bash
diff q08.reordered.sql q08.stock.sql   # MUST show the 3-line FROM swap, not empty
```

Repeat for q09 (stock order: `part, supplier, lineitem, partsupp, orders, nation`). Then re-run the stock
variant with `SET disable_join_reorder=true;` and with `SET cbo_enable_greedy_join_reorder=false;` prefixed,
to settle option (e).

**Safety note:** `EXPLAIN` does not execute the query, so this cannot OOM a GPU. Do **not** run the stock
text without `EXPLAIN` — that is the 331-billion-row cross join.

## What to record

- `results/g4-files-statistics-feasibility.md`: the option table from step 2, each row with a verdict and a
  deciding `file:line`.
- The `EXPLAIN`/`EXPLAIN COSTS` artifacts under `benchmarks/tpch/plans/` naming convention
  (`q08.stock.explain.txt`, `q08.stock.costs.txt`, …) so they sit beside the existing
  `plans/q08.verbose.txt` that QUERY-DEVIATIONS.md already cites.
- Whether `disable_join_reorder` / `cbo_enable_greedy_join_reorder` change the stock plan (expected: no).
- A one-paragraph amendment for QUERY-DEVIATIONS.md's "The principled fix" section, which currently lists
  ANALYZE / external catalog / injected stats as *"options not yet evaluated"* — after this gap they are
  evaluated, and at least two of the three are closed at
  `StatisticsCalculator.java:664-676`. **Draft the amendment in the results note; do not edit
  QUERY-DEVIATIONS.md as part of this plan** unless the executing session is explicitly asked to.

## Success criterion

1. Each of options (a)–(f) has a **Feasible / Infeasible / Requires-FE-patch** verdict with a deciding
   `file:line`.
2. The empirical check reproduces the cross join on stock text at SF100 and confirms `cardinality: 1`
   estimates — i.e. the documented root cause is re-verified on today's tree, not taken on faith.
3. A single recommendation with an effort estimate, and an explicit statement of whether the benchmark's
   "read parquet directly, no load" property survives it.

## Cost

| Item | Time |
|---|---|
| Source reading + option table | ~45 min |
| `EXPLAIN` checks (needs only an FE) | ~15 min |
| Write-up + QUERY-DEVIATIONS amendment draft | ~20 min |
| **Subtotal** | **~80 min, zero GPU time** |

---

# Suggested execution order and budget

| # | Step | Cluster | Config | Wall clock |
|---|---|---|---|---|
| 1 | **Gap 4 steps 1–3** (source reading, option table) | none | — | ~45 min |
| 2 | **Gap 1** SF100 + SF300 sweeps + `compare.py` | **Cluster A** | 60/32 + budgets, `LOG_LEVEL=info` | ~35 min |
| 3 | **Gap 2 procedure A1** — graceful shutdown of Cluster A, read the two teardown lines | Cluster A (teardown) | — | ~10 min |
| 4 | **Gap 4 step 4** (`EXPLAIN` only) — can run against Cluster A **before** step 3 | Cluster A | — | ~15 min |
| 5 | **Gap 2 procedure A2** — SF300 `--cold-restart` with the graceful restart wrapper | Cluster A pattern, 22 fresh clusters | same | ~40 min |
| 6 | **Gap 3** — DEBUG run on the SF500 pressure queries | **Cluster B** | 60/32 + budgets, **`LOG_LEVEL=debug`**, own `SIRIUS_LOG_DIR` | ~60 min |
| 7 | Write-ups for all four gaps | none | — | ~40 min |
| | **Total (no code change)** | | | **~3 h 45 min** |
| 8 | *Optional* **Gap 2 procedure B** — the `[arena]` per-window log line + engine rebuild + re-verify | Cluster C | rebuilt engine | **+1.5–3 h** |

## Which gaps share a cluster, and which cannot

- **Gaps 1, 2(A1) and 4(step 4) share one cluster.** The bring-up is dataset-agnostic, so SF100 and SF300
  run back-to-back on the same CNs; the Gap 2 A1 measurement is simply *how you shut that cluster down*; and
  `EXPLAIN` is FE-only and free to run alongside.
- **Gap 2 A2 needs 22 fresh clusters** by construction (`--cold-restart`), but they are the *same config* —
  it is a continuation of Cluster A's pattern, not a different one. Run it after step 3 so the A1 number is
  banked first.
- **Gap 3 needs its own cluster**: `SIRIUS_LOG_LEVEL=debug` is a bring-up-time env var and DEBUG output must
  land in its own `SIRIUS_LOG_DIR` so the volume does not contaminate the Gap 1/2 corpus.
- **Gap 4 needs no cluster of its own** — steps 1–3 are pure reading; step 4 needs any live FE.
- **Gap 2 procedure B needs its own cluster** because it needs a rebuilt engine; do it last, and re-run a
  short Gap 1 subset afterwards to confirm the added logging changed nothing.

## Ordering rationale

Gap 4's reading comes first because it is the only item that can be **completed without touching hardware**,
and it may close itself outright at `StatisticsCalculator.java:664-676` — in which case the remaining budget
goes to the measurement gaps. Gap 1 comes before Gap 2 A2 because Gap 1's fast single-cluster sweep tells you
whether the expensive per-query cold-restart run is even needed. Gap 3 goes last among the cluster items
because it is the only one whose logging configuration is incompatible with the others.

## Definition of done for PLAN-08

Four result notes exist under `bench/rtxpro6000-2gpu/results/`:
`g1-*.csv` + verdicts, `g2-arena-highwater.md`, `g3-downgrade-evidence.md`,
`g4-files-statistics-feasibility.md` — and `bench/rtxpro6000-2gpu/STATUS.md`'s row 8 is updated from
"four open gaps" to the four measured answers.
