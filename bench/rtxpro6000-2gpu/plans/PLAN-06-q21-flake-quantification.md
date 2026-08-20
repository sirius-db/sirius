# PLAN-06 — q21 at SF500: measure the flake rate, capture the pass/hang corpus

**Type: measurement.** This plan produces two artifacts and no code change:

1. a hang rate for TPC-H q21 at SF500 on the winning config, **with a confidence interval**;
2. a paired corpus of archived **pass** and **hang** logs that [PLAN-04](PLAN-04-scheduler-stall.md)
   consumes to diagnose the underlying defect.

**PLAN-04 owns the diagnosis** ("Defect B": an un-cancellable fragment head-of-line-blocking the CN's
single engine thread until the watchdog). This plan does not analyse the defect and does not
duplicate PLAN-04's content — it produces the evidence PLAN-04 needs and stops.

**q15 is a different flake.** q15 returns 0 rows intermittently (13/30 at SF100) because it compares
a GPU aggregate with exact float equality; it is already diagnosed and lives in
[PLAN-07](PLAN-07-q15-float-determinism.md), with its own repro at
`/opt/dlami/nvme/sirius-build/q15-repro.sh`. Do not conflate the two: q15 fails **correctness in
milliseconds**, q21 fails **liveness for ten minutes**. This plan borrows q15-repro.sh's *shape*
(N runs, arms, per-run table, summary) and nothing else.

---

## 1. The question

> On 2× RTX PRO 6000 with 2 CNs (one per GPU), at TPC-H **SF500 float64**
> (`/opt/dlami/nvme/tpch/tpch_parquet_sf500_f64`), on the config
> `GPU_MEM=60GiB STAGING=32GiB HOST_MEM=200GiB HPB=1GiB MBHT=2GiB STB=1GiB CBB=1GiB`,
> what fraction of q21 executions stall until the CN's 600 s fetch timeout instead of returning
> the correct 100 rows in ~20 s — and does that fraction depend on cluster freshness?

Two rates are wanted, not one:

| Arm | Regime | Why |
|---|---|---|
| **A** | **fresh cluster per run** (bring-up → single q21 → tear down) | the headline number; every run is an independent draw from a clean process state |
| **B** | **warm cluster** (one cluster, runs back-to-back; restart only after a hang) | tests whether freshness is the variable at all |

**Freshness is probably not the variable, and the plan must not assume it is.** The three existing
samples say the opposite of the naive story: the hang happened on a **fresh** cluster
(`--cold-restart`, run 0) and the two passes include a **warm** run
(`bench/SF500X/timings.csv:15`, `q21,1,warm,pass,17387,100`). Arm B is therefore a genuine
comparison, not a control expected to be clean.

## 2. What is already measured (the entire evidence base)

Three q21 executions on the target config, one per cluster generation:

| Source | Row | Outcome |
|---|---|---|
| `/opt/dlami/nvme/sirius-build/bench/SF500X/timings.csv:14` | `q21,0,cold,pass,17734,100` | pass, 17.7 s |
| `/opt/dlami/nvme/sirius-build/bench/SF500XCOLD/timings.csv:42` | `q21,0,cold,refused,617523,0` | **hang**, 617.5 s |
| `/opt/dlami/nvme/sirius-build/bench/SF500E3/timings.csv:6` | `q21,0,cold,pass,20596,100` | pass, 20.6 s |

The fourth observation is collateral, not an independent sample:
`bench/SF500XCOLD/timings.csv:43` → `q21,1,warm,refused,120559,0`, whose output file
`bench/SF500XCOLD/q21.r1.out` reads verbatim

```
ERROR 1064 (HY000) at line 1: rpc failed with 127.0.0.1: exec rpc error. backend [id=11001] [host=127.0.0.1]
```

i.e. the *next* query hitting the CN the previous run had wedged. `bench.sh:190-196` deliberately
keeps the warm runs on the same cluster after a cold failure, which is how this got recorded.

The hang's own error, from `bench/SF500XCOLD/q21.r0.out`:

```
ERROR 1064 (HY000) at line 1: fragment instance 01a01be2-... failed: timed out after 600s
waiting for fragment instance 01a01be2-... to produce rows (its exchange senders may have stalled)
```

That 600 s is `compute_node_service.rs:421` (`wait_ready(id, Duration::from_secs(600))`), whose
message is formatted at `result_store.rs:229-232`. It is a **CN-side** timeout, so it fires
regardless of the client-side `timeout` and regardless of the FE's `query_timeout` once that is
raised above 600 s.

**The hang is not a memory failure.** The GPU pool ledger sat at ~35 GiB of the 60 GiB cap. The
captured engine-log signature (one occurrence):

```
21:17:26.922  CN-B w6 QueryBegin  26.751 GiB
21:22:08.031  task_scheduler: no scheduling progress for 280s
21:22:08.033  query 6 operator  9 port 'default' still had  8 un-consumed data batch(es)
21:22:08.033  query 6 operator 13 port 'default' still had 24 un-consumed data batch(es)
21:22:08.033  [window] end ... outcome=unwind
```

**Three samples cannot support a rate.** The exact (Clopper–Pearson) 95% interval for 1 hang in 3
trials is **[0.008, 0.906]** — the current evidence is compatible with a 1% flake and with a 90%
flake. That is the whole reason this plan exists.

Note also that in that signature the 280 s watchdog fired ~281 s after `QueryBegin`, yet the client
only errored at ~617 s: the watchdog's failure did **not** shorten the FE's wait. Whether that holds
in every hang is a question for PLAN-04; this plan just has to record it faithfully — which is why
§6 captures the watchdog timestamp and the client's `ms` for every run.

## 3. Experimental design

### 3.1 Held constant (changing any of these invalidates the rate)

| Constant | Value | Where it comes from |
|---|---|---|
| Query | `experimental/starrocks/benchmarks/tpch/queries/q21.sql`, `__TPCH_DATA__` substituted | `bench.sh:163` |
| Data | `/opt/dlami/nvme/tpch/tpch_parquet_sf500_f64` | `sweep-sf500x-cold.sh:12` |
| Pool / arena | `GPU_MEM=60GiB STAGING=32GiB` | `up-sf500-x.sh:17,19` |
| Host pool | `HOST_MEM=200GiB` | `up-sf500-x.sh:18` |
| Operator budgets | `HPB=1GiB MBHT=2GiB STB=1GiB CBB=1GiB` | `up-sf500-x.sh:42-57` |
| CNs | `NUM_CNS=2`, one GPU each | `up-sf500-x.sh:20,105-112` |
| Watchdog | `SIRIUS_QUERY_WATCHDOG_SECS=280` (default) | `up-sf500-x.sh:27` |
| Downgrade fractions / disk | `DGT=0.8 DGS=0.6`, `DISK=/opt/dlami/nvme/sirius_spill` | `up-sf500-x.sh:21-23` |
| FE timeout | `SET GLOBAL query_timeout = 1800` after every bring-up | `restart-sf500x.sh:17-19` |
| Logging | `SIRIUS_LOG_BACKEND=spdlog`, `SIRIUS_LOG_LEVEL=info` | `up-sf500-x.sh:30-32` |
| Binary | whatever is at `experimental/starrocks/target/release/sirius-starrocks-cn` — **record `git rev-parse HEAD` and the binary's mtime in the run log; a rebuild mid-experiment voids it** | `up-sf500-x.sh:37` |

The **only** variable is the arm (fresh vs warm cluster).

### 3.2 N, and why

Binomial, Wilson 95% intervals, assuming the point estimate lands near the p ≈ 1/3 the three samples
hint at:

| N | hangs at p̂≈1/3 | Wilson 95% CI | width |
|---|---|---|---|
| 10 | 3 | [0.108, 0.603] | 0.50 |
| 20 | 7 | [0.181, 0.567] | 0.39 |
| **30** | **10** | **[0.192, 0.512]** | **0.32** |
| 50 | 17 | [0.224, 0.478] | 0.25 |
| 100 | 33 | [0.246, 0.427] | 0.18 |

**N = 30 for arm A.** It is the smallest N that answers the question actually being asked — *is this
a frequent flake or a rare one* — by excluding both "rare" (<19%) and "usually broken" (>51%).
Pushing to 50 buys 7 percentage points of width for another ~1.7 h; that trade is only worth making
if PLAN-04 later needs a tighter number to judge a fix. N=30 also gives, at p≈1/3, an expected
**10 hangs** — comfortably above the corpus goal of 3.

If **zero** hangs occur in 30 runs the run is still informative: the rule-of-three 95% upper bound is
**9.5%** (exactly `1 − 0.05^(1/30)`), which would itself contradict the 1-in-3 prior and become the
headline finding. But the corpus goal fails, so see the stop rules (§8.2).

**N = 20 for arm B**, and the arm comparison is **exploratory only**. Two-proportion power at
α = 0.05, arm A at 0.33:

| Arm B true rate | power at n=20/arm | at n=30/arm | at n=50/arm |
|---|---|---|---|
| 0.00 | 0.83 | 0.95 | 1.00 |
| 0.05 | 0.62 | 0.81 | 0.96 |
| 0.10 | 0.42 | 0.59 | 0.81 |
| 0.20 | 0.15 | 0.20 | 0.31 |

So the design can detect "warm never hangs" and is blind to anything subtler. **State this in the
write-up**; do not report a non-significant arm difference as "freshness does not matter".

### 3.3 Classification (exact, mechanical)

Per run, from the client output plus the run's engine-log slice:

| Verdict | Rule | Counts toward |
|---|---|---|
| `pass` | rc=0, first line ≠ `ERROR`, 100 data rows, and byte-identical to the oracle | denominator |
| `pass-stalled` | as `pass`, but the slice contains `no scheduling progress` | denominator, **and flagged** — a recovered stall is a distinct phenomenon |
| `hang` | first line contains `timed out after 600s waiting for fragment instance` | numerator |
| `client-cut` | rc=124 (client `timeout` fired at 700 s before the CN's 600 s error surfaced) | numerator, flagged |
| `wrong` | rc=0 but rows ≠ 100 or bytes differ from the oracle | **neither** — a new correctness defect; stop and report |
| `collateral` | first line contains `rpc failed with` | **excluded from the denominator**, counted separately: it is the previous run's wedge, not an independent draw |
| `other` | anything else; message recorded verbatim | excluded, investigate |

The oracle is `/opt/dlami/nvme/sirius-build/oracle-sf500f64/q21.tsv`. A passing q21 output is
**byte-identical** to it (verified: `diff` of `bench/SF500X/q21.r0.out` against the oracle is empty,
101 lines each; columns are `s_name` + an integer `numwait`, so no float tolerance is needed). Plain
`cmp` is therefore the correctness gate; `compare.py` is unnecessary here, but exists at
`/opt/dlami/nvme/sirius-build/compare.py` if a tolerant diff is ever wanted. **This gate is not
optional**: `bench.sh` has none at all ("this script times and counts rows only — it does not check
answers", `bench.sh:53-56`), and a hang investigation that silently accepted a wrong answer would be
worthless.

### 3.4 Time budget — state it before starting, not after

Per-run cost = cluster bring-up + query. Bring-up is ~70–90 s (kill + settle, the FE needs ~60 s,
then `SHOW COMPUTE NODES` must report 2 alive twice in a row). Query is ~20 s on a pass and ~620 s
on a hang.

| Arm | N | at p=0.10 | at p=0.33 | at p=0.50 |
|---|---|---|---|---|
| A (restart every run) | 30 | 1.3 h | **2.5 h** | 3.4 h |
| A | 50 | 2.2 h | 4.2 h | 5.6 h |
| B (restart only after a hang) | 30 | 0.7 h | 2.1 h | 3.0 h |

Arm A bounds: **0.8 h** if all 30 pass, **5.9 h** if all 30 hang. Budget **a full working day** for
arm A (30) + arm B (20) and run arm A first — it is the deliverable; arm B is the refinement.

## 4. Preconditions — read before touching anything

1. **This experiment takes exclusive ownership of both GPUs.** It runs `pkill -f
   'target/release/sirius-starrocks-cn'` and `pkill -f 'com.starrocks.StarRocksFE'` up to 30 times.
   A cluster may already be live and owned by someone else — as of the last status note the box had a
   cluster up at 65 GiB/27 GiB, *not* the config under test. Confirm the box is free, then run. The
   script refuses to start if a CN is already running unless `FORCE=1`.
2. Do not write to `/tmp/cluster-sf500x.log`; `restart-sf500x.sh:12` truncates it with `>` and
   another session may be reading it. This plan's loop redirects to its own per-run path.
3. Do not edit `up-sf500-x.sh`, `restart-sf500x.sh` or `bench.sh`. Everything here is env vars.
4. Record `git -C /home/ubuntu/sirius rev-parse HEAD`, `git status --porcelain`, and
   `ls -l experimental/starrocks/target/release/sirius-starrocks-cn` into `$OUT/provenance.txt`
   before the first run. The tree is uncommitted work in progress; a rebuild mid-experiment silently
   changes the system under test.

## 5. The script

Write these three files, then run. Nothing here modifies an existing script.

### 5.1 `/opt/dlami/nvme/sirius-build/q21flake/slice-log.py`

Extracts only the bytes a single run appended to the (shared, append-only) engine log.

```python
#!/usr/bin/env python3
"""Slice the bytes appended to a Sirius engine log since the last call.

The spdlog sink writes ONE daily-rotated file per log dir, opened in append mode
(src/log/spdlog_owning_sink.cpp:52-53, `daily_file_sink_mt(..., truncate=false)`), and BOTH CNs
write to it. A restart therefore APPENDS rather than truncating, so per-run attribution needs byte
offsets. Handles midnight rollover: a newly appeared sirius_<date>.log starts at offset 0.

usage: slice-log.py <logdir> <state.json> <out.slice>
"""
import json
import pathlib
import sys

logdir, state_path, out = (pathlib.Path(a) for a in sys.argv[1:4])
state = json.loads(state_path.read_text()) if state_path.exists() else {}
chunks = []
for p in sorted(logdir.glob("sirius*.log")):
    off = state.get(p.name, 0)
    with p.open("rb") as f:
        f.seek(off)
        data = f.read()
    state[p.name] = off + len(data)
    if data:
        chunks.append(data)
out.write_bytes(b"".join(chunks))
state_path.write_text(json.dumps(state))
```

### 5.2 `/opt/dlami/nvme/sirius-build/q21flake/summarize-run.py`

Parses one run's slice. **Deliberately does not shell out to `grep`** — see §9.

```python
#!/usr/bin/env python3
"""Summarize one run's engine-log slice; print the CSV fields the loop records.

Parsing is done in Python because grep silently reports NOTHING on these logs: verified on
/opt/dlami/nvme/sirius-build/siriuslog/sirius_2026-08-19.log, `grep -c gpu_pool` exits 1 with no
output while `grep -ac gpu_pool` prints 1939. A grep-based classifier would score every run "clean".

usage: summarize-run.py <slice.log> <run_dir>
"""
import json
import pathlib
import re
import sys

slice_path, run_dir = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
lines = slice_path.read_text(errors="replace").splitlines()


def sel(pat):
    return [ln for ln in lines if re.search(pat, ln)]


def ints(pat, src):
    return [int(m.group(1)) for ln in src for m in [re.search(pat, ln)] if m]


pool = sel(r"\[gpu_pool\]")
outcomes = {}
for ln in sel(r"\[window\] end"):
    # A torn write can sever `outcome=` from its own line; never let that abort the parse.
    m = re.search(r"outcome=(\S+)", ln)
    key = m.group(1) if m else "<torn>"
    outcomes[key] = outcomes.get(key, 0) + 1
begins = len(sel(r"\[window\] begin"))
# One line per stall EVENT. The bare "no scheduling progress" matches twice per event, because
# sirius_engine.cpp:230 rethrows the same text; keep that wider set only for the dump.
watchdog = sel(r"task_scheduler: no scheduling progress")
watchdog_ctx = sel(r"no scheduling progress")
unconsumed = sel(r"un-consumed data batch")
arena_teardown = sel(r"arena: peak live")
arena_exhausted = sel(r"arena exhausted")
downgrade = sel(r"downgrade")
# Two CN processes appending to one file tear each other's writes; a torn line can swallow a
# record. Measured at ~0.3% in the existing shared log. Report it so a suspiciously clean slice
# can be distrusted rather than believed.
torn = [ln for ln in lines if "�" in ln or len(re.findall(r"\] \[\w+\] \[", ln)) > 1]

summary = {
    "engine_log_lines": len(lines),
    "windows_begin": begins,
    "windows_end": sum(outcomes.values()),
    # A window that begins and never ends is a fragment the process died inside.
    "windows_never_ended": begins - sum(outcomes.values()),
    "window_outcomes": outcomes,
    "watchdog_lines": len(watchdog),
    "unconsumed_lines": len(unconsumed),
    "unconsumed_batches_total": sum(ints(r"still had (\d+) un-consumed", unconsumed)),
    "max_gpu_allocated_bytes": max(ints(r"allocated=(\d+) bytes", pool), default=0),
    "max_gpu_peak_bytes": max(ints(r"peak=(\d+) bytes", pool), default=0),
    "arena_teardown_lines": len(arena_teardown),
    "arena_peak_live_bytes": max(ints(r"peak live (\d+) of", arena_teardown), default=0),
    "arena_exhausted_lines": len(arena_exhausted),
    "downgrade_lines": len(downgrade),
    "torn_lines": len(torn),
    # The ONLY per-CN discriminator in a shared log; two values == the two CNs of this generation.
    # Pointers are 12 hex digits here; the length floor drops fragments left by torn writes.
    "instances": sorted(set(re.findall(r"instance=0x[0-9a-f]{12,}", "\n".join(lines)))),
    "pci": sorted(set(re.findall(r"pci=[0-9a-fA-F:.]+", "\n".join(lines)))),
}
(run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
with (run_dir / "summary.txt").open("w") as f:
    for k, v in summary.items():
        print(f"{k}: {v}", file=f)
    for title, block in (
        ("watchdog", watchdog_ctx),
        ("un-consumed", unconsumed),
        ("arena", arena_teardown + arena_exhausted),
        ("torn", torn),
        ("last 60 engine lines", lines[-60:]),
    ):
        print(f"--- {title} ---", file=f)
        for ln in block:
            print(ln, file=f)

print(
    f'{summary["watchdog_lines"]},{summary["unconsumed_lines"]},'
    f'{summary["unconsumed_batches_total"]},{summary["max_gpu_allocated_bytes"]},'
    f'{summary["arena_teardown_lines"]},{summary["arena_peak_live_bytes"]},'
    f'{summary["windows_never_ended"]},{summary["torn_lines"]}'
)
```

### 5.3 `/opt/dlami/nvme/sirius-build/q21-flake.sh`

```bash
#!/usr/bin/env bash
# q21 SF500 flake-rate measurement + diagnostic corpus (PLAN-06).
#
# Arm A (default): fresh cluster per run  -- the headline rate.
# Arm B:           warm cluster; restart ONLY after a hang, because a wedged CN poisons the next
#                  run ("rpc failed with 127.0.0.1: exec rpc error", measured).
#
# Per run it archives the engine-log slice, the cluster stdout, the query output and a parsed
# summary; over all runs it writes results.csv and prints a table + a Wilson 95% interval.
#
#   ARM=A N=30 /opt/dlami/nvme/sirius-build/q21-flake.sh
#   ARM=B N=20 /opt/dlami/nvme/sirius-build/q21-flake.sh
set -uo pipefail

ARM=${ARM:-A}
N=${N:-30}
TAG=${TAG:-arm$ARM-$(date +%Y%m%d-%H%M%S)}
OUT=${OUT:-/opt/dlami/nvme/sirius-build/q21flake/$TAG}
HELP=/opt/dlami/nvme/sirius-build/q21flake

# ---- system under test: the winning SF500 config. Any change here invalidates the rate. -------
export GPU_MEM=${GPU_MEM:-60GiB} STAGING=${STAGING:-32GiB} HOST_MEM=${HOST_MEM:-200GiB}
export HPB=${HPB:-1GiB} MBHT=${MBHT:-2GiB} STB=${STB:-1GiB} CBB=${CBB:-1GiB}
export NUM_CNS=${NUM_CNS:-2}
# Only duckdb/spdlog/noop are accepted, and an unknown value is SILENTLY dropped on the CN's FFI
# path (src/sirius_context.cpp:1573-1578). Without spdlog this experiment collects no engine logs
# at all and the corpus half of the plan produces nothing.
export SIRIUS_LOG_BACKEND=spdlog
export SIRIUS_LOG_LEVEL=${SIRIUS_LOG_LEVEL:-info}
# SIRIUS_QUERY_WATCHDOG_SECS is left at up-sf500-x.sh:27's 280s: it is part of the SUT.

SR=/home/ubuntu/sirius/experimental/starrocks
UP=/opt/dlami/nvme/sirius-build/up-sf500-x.sh
DATA=${DATA:-/opt/dlami/nvme/tpch/tpch_parquet_sf500_f64}
ORACLE=${ORACLE:-/opt/dlami/nvme/sirius-build/oracle-sf500f64/q21.tsv}
# > the CN's own 600s wait_ready (compute_node_service.rs:421) so the CN's error wins the race and
# the run is classified from its message rather than from an opaque client cut.
CLIENT_TIMEOUT=${CLIENT_TIMEOUT:-700}
FE_QUERY_TIMEOUT=${FE_QUERY_TIMEOUT:-1800}
export PATH=$SR/.pixi/envs/default/bin:$PATH
MYSQL="mysql --host 127.0.0.1 --port 9030 --user root --batch --connect-timeout=5"

for f in "$UP" "$ORACLE" "$HELP/slice-log.py" "$HELP/summarize-run.py" \
         "$SR/benchmarks/tpch/queries/q21.sql"; do
  [ -r "$f" ] || { echo "missing: $f" >&2; exit 1; }
done
mkdir -p "$OUT"
SQL=$(sed "s|__TPCH_DATA__|$DATA|g" "$SR/benchmarks/tpch/queries/q21.sql")

# ---- exclusivity guard: this kills BOTH CNs and the FE up to N times. -------------------------
if pgrep -f 'target/release/sirius-starrocks-cn' >/dev/null && [ "${FORCE:-0}" != 1 ]; then
  echo "A CN is already running; this experiment would kill it." >&2
  pgrep -af 'target/release/sirius-starrocks-cn' >&2
  echo "Confirm the box is yours, then re-run with FORCE=1." >&2
  exit 1
fi

{
  echo "arm=$ARM N=$N tag=$TAG"
  echo "GPU_MEM=$GPU_MEM STAGING=$STAGING HOST_MEM=$HOST_MEM NUM_CNS=$NUM_CNS"
  echo "HPB=$HPB MBHT=$MBHT STB=$STB CBB=$CBB"
  echo "data=$DATA oracle=$ORACLE client_timeout=${CLIENT_TIMEOUT}s"
  echo "sirius HEAD: $(git -C /home/ubuntu/sirius rev-parse HEAD)"
  echo "dirty files: $(git -C /home/ubuntu/sirius status --porcelain | wc -l)"
  ls -l "$SR/target/release/sirius-starrocks-cn"
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
} | tee "$OUT/provenance.txt"

# ---- cluster control --------------------------------------------------------------------------
alive_count() {  # Alive column resolved from the header, as bench.sh:108-113 does
  $MYSQL -e "SHOW COMPUTE NODES;" 2>/dev/null | awk -F'\t' '
    NR == 1 { for (i = 1; i <= NF; i++) if ($i == "Alive") c = i; next }
    c && $c == "true" { n++ }
    END { print n + 0 }'
}

wait_alive() {  # settled count, two consecutive polls (bench.sh:119-131)
  local n prev=-1
  for _ in $(seq 1 150); do
    n=$(alive_count)
    [ "$n" -ge "$NUM_CNS" ] && [ "$n" = "$prev" ] && return 0
    prev=$n
    sleep 2
  done
  return 1
}

cluster_down() {
  pkill -f 'target/release/sirius-starrocks-cn' 2>/dev/null
  pkill -f 'com.starrocks.StarRocksFE' 2>/dev/null
  # SHUTDOWN_GRACE is 15s (experimental/starrocks/src/main.rs:34) and a WEDGED CN only force-exits
  # at the END of it (main.rs:671-685), so restart-sf500x.sh:6's fixed 8s sleep can relaunch while
  # the old CN still owns 9100-9104. Poll instead, then escalate.
  local killed9=0
  for _ in $(seq 1 30); do
    pgrep -f 'target/release/sirius-starrocks-cn' >/dev/null || break
    sleep 1
  done
  if pgrep -f 'target/release/sirius-starrocks-cn' >/dev/null; then
    echo "  CN still alive 30s after SIGTERM -> SIGKILL (no arena teardown line for this run)"
    pkill -9 -f 'target/release/sirius-starrocks-cn'
    killed9=1
    sleep 3
  fi
  pkill -9 -f 'com.starrocks.StarRocksFE' 2>/dev/null
  sleep 2
  # The driver reclaims device memory at process death; confirm it actually happened before the
  # next bring-up tries to reserve 60+32 GiB per card.
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
    awk -F', ' '$2 > 2000 { printf "  WARNING: GPU %s still holds %s MiB\n", $1, $2 }'
  SIGKILLED=$killed9
}

GEN=0
cluster_up() {  # $1 = run dir that owns this generation's logs
  GEN=$((GEN + 1))
  GENDIR=$1/gen$GEN
  mkdir -p "$GENDIR/enginelog"
  LOGSTATE=$GENDIR/logstate.json
  CLUSTER_LOG=$GENDIR/cluster.log
  SIRIUS_LOG_DIR="$GENDIR/enginelog" nohup "$UP" > "$CLUSTER_LOG" 2>&1 &
  wait_alive || return 1
  # The FE aborts server-side at query_timeout (default 300s) no matter what the client does
  # (restart-sf500x.sh:14-19); at 300s a hang would be misreported before the CN's 600s error.
  $MYSQL -e "SET GLOBAL query_timeout = $FE_QUERY_TIMEOUT;" >/dev/null 2>&1
}

# ---- run loop ----------------------------------------------------------------------------------
CSV=$OUT/results.csv
echo "run,arm,gen,verdict,ms,rows,oracle_match,watchdog,unconsumed_lines,unconsumed_batches,max_gpu_alloc_bytes,arena_teardown,arena_peak_bytes,windows_never_ended,torn_lines,sigkilled,note" > "$CSV"
printf '%-4s %-14s %-9s %-6s %-9s %s\n' run verdict ms rows watchdog note

pass=0; hang=0; other=0; collateral=0; wrong=0
for i in $(seq 1 "$N"); do
  RUN=$(printf '%s/run%02d' "$OUT" "$i")
  mkdir -p "$RUN"
  NEED_RESTART=0
  [ "$ARM" = A ] && NEED_RESTART=1
  [ "$i" = 1 ] && NEED_RESTART=1
  [ "${RESTART_NEXT:-0}" = 1 ] && NEED_RESTART=1
  RESTART_NEXT=0
  SIGKILLED=0
  if [ "$NEED_RESTART" = 1 ]; then
    cluster_down
    cluster_up "$RUN" || { echo "run $i: cluster did not come up; aborting"; break; }
  fi

  t0=$(date +%s%3N)
  timeout "$CLIENT_TIMEOUT" $MYSQL -e "$SQL" > "$RUN/q21.out" 2>&1
  rc=$?
  t1=$(date +%s%3N)
  ms=$((t1 - t0))

  # Close out this run's engine-log slice BEFORE any restart: the next bring-up appends to the same
  # daily file when the generation is reused, and truncates nothing.
  python3 "$HELP/slice-log.py" "$GENDIR/enginelog" "$LOGSTATE" "$RUN/engine.slice.log"
  fields=$(python3 "$HELP/summarize-run.py" "$RUN/engine.slice.log" "$RUN")
  IFS=, read -r wd unc uncb gpualloc arenat arenap neverend torn <<< "$fields"
  cp "$CLUSTER_LOG" "$RUN/cluster.log" 2>/dev/null

  head1=$(head -1 "$RUN/q21.out" 2>/dev/null)
  rows=$(( $(wc -l < "$RUN/q21.out") - 1 )); [ "$rows" -lt 0 ] && rows=0
  match=na; note=
  if [ $rc -eq 124 ]; then
    verdict=client-cut; note="client cut at ${CLIENT_TIMEOUT}s"
  elif printf '%s' "$head1" | grep -q 'timed out after 600s waiting for fragment instance'; then
    verdict=hang; note="CN wait_ready timeout"
  elif printf '%s' "$head1" | grep -q 'rpc failed with'; then
    verdict=collateral; note="previous run's wedge"
  elif printf '%s' "$head1" | grep -q '^ERROR'; then
    verdict=other; note=$(printf '%s' "$head1" | cut -c1-100)
  elif [ "$rows" -ne 100 ]; then
    verdict=wrong; note="rows=$rows expected 100"
  elif cmp -s "$RUN/q21.out" "$ORACLE"; then
    match=yes
    if [ "$wd" -gt 0 ]; then verdict=pass-stalled; note="watchdog fired but query completed";
    else verdict=pass; fi
  else
    verdict=wrong; match=no; note="100 rows but differs from oracle"
  fi

  case $verdict in
    pass|pass-stalled) pass=$((pass + 1)) ;;
    hang|client-cut)   hang=$((hang + 1)); RESTART_NEXT=1 ;;
    collateral)        collateral=$((collateral + 1)); RESTART_NEXT=1 ;;
    wrong)             wrong=$((wrong + 1)); RESTART_NEXT=1 ;;
    *)                 other=$((other + 1)); RESTART_NEXT=1 ;;
  esac

  echo "$i,$ARM,$GEN,$verdict,$ms,$rows,$match,$wd,$unc,$uncb,$gpualloc,$arenat,$arenap,$neverend,$torn,$SIGKILLED,\"$note\"" >> "$CSV"
  printf '%-4s %-14s %-9s %-6s %-9s %s\n' "$i" "$verdict" "$ms" "$rows" "wd=$wd" "$note"

  if [ "$verdict" = wrong ]; then
    echo "STOPPING: q21 returned a WRONG answer -- that is a different and more serious defect."
    break
  fi
done
cluster_down

# ---- summary -----------------------------------------------------------------------------------
echo
python3 - "$CSV" <<'PY'
import csv, math, sys
rows = list(csv.DictReader(open(sys.argv[1])))
counts = {}
for r in rows:
    counts[r['verdict']] = counts.get(r['verdict'], 0) + 1
valid = [r for r in rows if r['verdict'] in ('pass', 'pass-stalled', 'hang', 'client-cut')]
k = sum(1 for r in valid if r['verdict'] in ('hang', 'client-cut'))
n = len(valid)
print(f"=== {sys.argv[1]} ===")
for v, c in sorted(counts.items()):
    print(f"  {v:14s} {c}")
print(f"  denominator (independent draws): {n}   hangs: {k}")
if n:
    p, z = k / n, 1.959964
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    print(f"  hang rate = {p:.3f}   Wilson 95% CI = [{max(0,c-h):.3f}, {min(1,c+h):.3f}]")
    if k == 0:
        print(f"  zero hangs: 95% upper bound = {1 - 0.05 ** (1 / n):.3f}")
ok = [int(r['ms']) for r in rows if r['verdict'].startswith('pass')]
if ok:
    ok.sort()
    print(f"  pass latency ms: min={ok[0]} median={ok[len(ok)//2]} max={ok[-1]}")
PY
echo "corpus: $OUT   (per-run: q21.out, engine.slice.log, cluster.log, summary.txt/json)"
```

### 5.4 Recommended: one log directory per CN

Both CNs write to **one** file with independent file handles, and the writes tear. Measured in
`/opt/dlami/nvme/sirius-build/siriuslog/sirius_2026-08-19.log`: 1939 lines contain `gpu_pool` but
only 1934 match `[gpu_pool] GPU:0`; the other five are spliced records, e.g. a
`sirius_context.cpp:312 QueryBegin:` line landing *inside* a `[gpu_pool] GP…` line, and one with a
mojibake `GP\xef\xbf\xbd`. That corruption is also what makes plain `grep` treat the file as binary
(§9.2). ~0.3% of records, but a torn record could be the watchdog line of the one hang that matters.

Fix it without touching the shared script — **copy, then edit the copy**:

```bash
cp /opt/dlami/nvme/sirius-build/up-sf500-x.sh /opt/dlami/nvme/sirius-build/up-sf500-x-perlog.sh
```

In the copy, inside the CN launch loop (`up-sf500-x.sh:105-112`), give each child its own dir:

```bash
  mkdir -p "$SIRIUS_LOG_DIR/cn$i"
  SIRIUS_LOG_DIR="$SIRIUS_LOG_DIR/cn$i" "$CN_BIN" --gpu-device "$i" \
```

(`Config::LOG_DIR` is read from the environment once per CN process at
`src/sirius_context.cpp:1583-1585`, so a per-child override is all that is needed.) Then set
`UP=/opt/dlami/nvme/sirius-build/up-sf500-x-perlog.sh` in the loop script and have `slice-log.py`
run once per `cn0`/`cn1` subdirectory, producing `engine.cn0.slice.log` / `engine.cn1.slice.log`.
Verify on the first run that **two** files appear and that each contains exactly one `pci=` value.

### 5.5 Smoke-test the parsers before burning hours

Run the two helpers against an **existing** log first — no cluster involved:

```bash
mkdir -p /tmp/p6/logdir /tmp/p6/run01
cp /opt/dlami/nvme/sirius-build/siriuslog/sirius_*.log /tmp/p6/logdir/
python3 /opt/dlami/nvme/sirius-build/q21flake/slice-log.py \
        /tmp/p6/logdir /tmp/p6/state.json /tmp/p6/run01/engine.slice.log
python3 /opt/dlami/nvme/sirius-build/q21flake/summarize-run.py \
        /tmp/p6/run01/engine.slice.log /tmp/p6/run01
cat /tmp/p6/run01/summary.txt | head -20
```

Against `sirius_2026-08-19.log` (48 cluster generations) this yields, and these are the numbers to
sanity-check against: `windows_begin: 973`, `window_outcomes: {'ok': 929, 'unwind': 35,
'<torn>': 1}`, `watchdog_lines: 4`, `unconsumed_lines: 6`, `arena_teardown_lines: 48`,
`torn_lines: 157`, `pci: ['pci=00000000:32:00.0', 'pci=00000000:33:00.0']`. A second call reuses
`state.json` and must produce an **empty** slice — that is the offset mechanism working.

## 6. What each run captures, and why PLAN-04 needs it

Every `run<NN>/` directory holds:

| Artifact | Content | What PLAN-04 does with it |
|---|---|---|
| `q21.out` | client output verbatim | classification; the exact CN error string |
| `engine.slice.log` | **only** this run's bytes of the spdlog engine log | the primary diff subject |
| `cluster.log` | the CN/FE stdout for this cluster generation | Rust-side tracing, nixl/UCX errors, and the `forcing process exit: graceful shutdown did not finish` line (`main.rs:677-683`) that proves the engine thread was wedged |
| `summary.txt` / `summary.json` | parsed counters (below) | machine-diffable pass-vs-hang comparison |

Counters extracted per run, each traceable to its emitter:

| Field | Log line | Source |
|---|---|---|
| `max_gpu_allocated_bytes`, `max_gpu_peak_bytes` | `[gpu_pool] GPU:0 QueryBegin\|QueryEnd instance=0x… allocated=… peak=… reserved=…` | `src/sirius_context.cpp:254` |
| `watchdog_lines` | `task_scheduler: no scheduling progress for 280s …` | `src/pipeline/task_scheduler.cpp:248-251` |
| `unconsumed_lines`, `unconsumed_batches_total` | `run_mandatory_cleanup: query N operator M port 'default' still had K un-consumed data batch(es)` | `src/sirius_context.cpp:421-427` |
| `windows_begin/end`, `window_outcomes`, `windows_never_ended` | `[window] begin\|end instance=… window=… outcome=-\|ok\|unwind\|begin_failed\|cleanup_failed` | `src/sirius_context.cpp:503`, outcomes set at `:544,556,565,597,601,612` |
| `arena_teardown_lines`, `arena_peak_live_bytes` | `exchange staging arena: peak live X of Y bytes (…)` | `src/exec/exchange_staging_arena.cpp:159-176` |
| `arena_exhausted_lines` | `exchange staging arena exhausted: requested …` | `src/exec/exchange_staging_arena.cpp:246` |
| `instances`, `pci` | `instance=0x…` and the startup line `GPU 0: NVIDIA RTX PRO 6000 … pci=00000000:3X:00.0` | `src/sirius_context.cpp:637` |
| `torn_lines` | records spliced by the two CNs writing one file (§5.4) | measurement artifact, not engine behaviour — a high count means the slice is unreliable |

**Per-CN attribution.** Both CNs append to the *same* file and every `[gpu_pool]` line says `GPU:0`
(verified: all 1934 intact `[gpu_pool] GPU:n` records in the existing log read `GPU:0`, because each
CN is launched with `--gpu-device i` and sees its card as index 0). The only discriminators are:

* `instance=0x…` — one value per CN *process*, so exactly two values inside one generation's slice.
  Label them CN-X / CN-Y; that is enough to say "one CN stalled while the other did not".
* `pci=00000000:32:00.0` vs `:33:00.0` on the startup topology line — the physical cards. These lines
  carry **no** `instance=`, so mapping instance → card is by startup ordering only, and is not
  reliable. Do not claim a specific card without corroboration from `cluster.log`
  (`up-sf500-x.sh:111` prints `CN0 gpu=0 heartbeat=9100 pid=…`) plus
  `nvidia-smi --query-compute-apps=pid,used_gpu_memory`.
* The watchdog line and the un-consumed-batch warnings carry **no** `instance=` either. Attribute
  them by the nearest preceding `[window] begin` from the same log region and say so explicitly when
  reporting.

Both problems — attribution and the torn writes of §5.4 — disappear if each CN gets its own log
directory. **Do that (§5.4) unless there is a reason not to.**

## 7. Analysis procedure

Run it in this order; stop as soon as a step contradicts the expectation, and hand that to PLAN-04.

1. **Pick the pair.** Take the hang with the cleanest slice and the *temporally nearest* pass in the
   same arm — ideally adjacent runs, so binary, data and box state are identical.
2. **Confirm the classification is real**, not harness noise: the hang's `q21.out` must carry the
   `timed out after 600s` message (not `rpc failed`), and `ms` must be ≈ 600 000–630 000. A run near
   700 000 ms was cut by the client, and its CN-side error is missing — treat it as lower quality
   evidence. Check `torn_lines` on both slices before trusting any *absence* of a log line.
3. **Rule memory in or out first**, because it is the cheapest discriminator and the one existing
   sample says it is not the cause: compare `max_gpu_allocated_bytes` and `max_gpu_peak_bytes`
   against the 60 GiB cap (64 424 509 440 B). Expected: both well under (~35 GiB was measured). If a
   hang ever shows ≥ 99% of the cap, it is a *different* failure from the captured signature and
   belongs with the q09 memory-wall analysis, not PLAN-04.
4. **Count windows.** `windows_begin` vs `windows_end` and the `window_outcomes` histogram. A hang
   should show `outcome=unwind` (the watchdog's exception unwinding the scope,
   `sirius_context.cpp:604-612`) where the pass shows `ok`, and may show a window that never ended.
   A *different* begin count between pass and hang means the two runs did not even plan the same
   fragment set — investigate that before anything else.
5. **Read the un-consumed-batch lines.** In the captured signature these were `operator 9 port
   'default'` (8 batches) and `operator 13 port 'default'` (24). Across ≥3 hangs, ask: are the
   operator ids and port names **stable**? A stable pair names the exact exchange edge that stalled
   and is the single most valuable output of this plan. A passing run should have **zero** such
   lines; if passes also leak batches, the signature is not diagnostic and say so.
6. **Locate the stall in time.** In the hang slice, find the last line before the ~280 s silence,
   and the first line after the watchdog. The gap's boundaries bracket the stalled stage. Compare
   with the pass's line at the same phase.
7. **Check `pass-stalled`.** Any run that passed *with* a watchdog line is a stall that resolved —
   the strongest available evidence that the stall is a livelock/ordering problem rather than a
   deadlock. Count them; they belong in the write-up even though they score as passes.
8. **Check the arena** where available (`arena_peak_live_bytes`). Absent for SIGKILLed CNs — see §9.
9. **Check `cluster.log`** of the *hanging* run for `forcing process exit: graceful shutdown did not
   finish` at its tail: that line, emitted 15 s after the SIGTERM, is independent confirmation that
   the engine thread could not be joined, i.e. that the fragment was genuinely un-cancellable.

**What this corpus cannot answer**: no code today logs *why* a pipeline is blocked — which port is
not EOS, whether a reservation is outstanding. `task_scheduler.cpp:244-255` reports the stall and
nothing else. Adding that logging is PLAN-04's first step; if the corpus cannot localise the stall,
the honest conclusion is "instrument, then re-run this plan", not a guess.

*(The repo's `log-analyzer` skill, `.claude/skills/log-analyzer/`, reads exactly this log format and
supports two-run comparison; use it for step 6 if preferred.)*

## 8. Success criteria and stop rules

### 8.1 Done means

1. **≥30 valid arm-A draws** (`pass` + `pass-stalled` + `hang` + `client-cut`; `collateral` and
   `other` excluded from the denominator and reported separately), with the hang rate quoted **with
   its Wilson 95% interval** — never a bare fraction.
2. **≥3 archived hang runs and ≥3 archived pass runs**, each complete (`q21.out`,
   `engine.slice.log`, `cluster.log`, `summary.json`).
3. **Every pass byte-identical to the oracle.** A single `wrong` verdict outranks this entire plan:
   stop, and report it as a new correctness defect.
4. **Arm B (N≥20) reported alongside**, with the power table from §3.2 quoted so no one reads a
   non-significant difference as evidence of no difference.
5. `results.csv` committed under `bench/rtxpro6000-2gpu/results/` as `q21-flake-armA.csv` /
   `-armB.csv`; the corpus stays on `/opt/dlami/nvme` (too large for the repo) with its absolute path
   recorded in the write-up.
6. Two one-line documentation updates: `SF500-CONFIG-AND-ARCHITECTURE.md:44` ("3 samples: 17.7s pass,
   600s hang, 20.6s pass") replaced with the measured rate + CI, and the PLAN-06 row of
   `STATUS.md:65` marked done with the number.

### 8.2 Stop rules

* **0 hangs in 30 arm-A runs** → report the ≤9.5% upper bound as the finding, and note that the
  corpus goal failed. Do not extend to N=100 hoping for a hang; instead get the corpus from q07,
  whose warm run spent 207 s of 289 s in the same stall, which PLAN-04 already treats as the same
  defect — that is a cheaper hang source.
* **≥3 hangs collected before run 30** → keep going anyway; the rate needs the full N.
* **Two consecutive `collateral`/`other` verdicts, or a bring-up failure** → the box is in a bad
  state. Stop, capture `nvidia-smi`, the cluster log and `pgrep -af`, and restore manually. Do not
  let the loop keep burning hours against a broken cluster.
* **A `wrong` verdict** → stop immediately (the script does this).
* **Any rebuild of `sirius-starrocks-cn` mid-experiment** → discard and restart; the SUT changed.

## 9. Gotchas — all verified, all load-bearing

1. **`SIRIUS_LOG_BACKEND` must be exactly `spdlog`.** Only `duckdb`, `spdlog`, `noop` are accepted,
   and on the CN's FFI path an unknown value is **silently discarded**: the `throw` in
   `install_configured_log_sink` is guarded by `else if (db)`
   (`src/sirius_context.cpp:1570-1578`) and the CN passes `nullptr` (`src/sirius_ffi.cpp:177`,
   `src/sirius_context.cpp:1589`). Set `SIRIUS_LOG_DIR` and `SIRIUS_LOG_LEVEL` too
   (`src/sirius_context.cpp:1583-1585`). With this wrong there are **no engine logs at all** and the
   corpus half of this plan silently produces nothing.
2. **`grep` reports NOTHING on these logs.** Verified on
   `/opt/dlami/nvme/sirius-build/siriuslog/sirius_2026-08-19.log`: `grep -c "gpu_pool"` exits 1 with
   no output, while `grep -ac "gpu_pool"` prints 1939. The cause is the torn writes of §5.4, which
   put invalid bytes in the file and make grep treat it as binary. Parse with `python3` (as the
   helpers above do) or always pass `-a`. A grep-based classifier scores every run clean.
3. **One shared log file, two CNs, and the writes tear.**
   `daily_file_sink_mt(log_dir/"sirius.log", 0, 0, truncate=false)`
   (`src/log/spdlog_owning_sink.cpp:52-53`) — the file is `sirius_<YYYY-MM-DD>.log`, **appended**,
   not truncated, and both CN processes write to it through independent handles. So (a) a restart
   does **not** rotate it, and per-run attribution needs **byte offsets** (`slice-log.py`); (b) per-CN
   attribution is by `instance=0x…` only (see §6); (c) ~0.3% of records are spliced or corrupted
   (§5.4). Giving each *run* its own `SIRIUS_LOG_DIR` — which `up-sf500-x.sh:31` honours via
   `${SIRIUS_LOG_DIR:-…}` — makes arm A trivially separable; arm B still needs the offsets; and only
   per-CN dirs (§5.4) stop the tearing.
4. **The cluster stdout *is* truncated by a restart**: `restart-sf500x.sh:12` redirects with
   `> /tmp/cluster-sf500x.log`. This plan's loop writes per-generation stdout instead, and copies it
   into the run directory after every run.
5. **The arena teardown line only prints on a clean shutdown** (`exchange_staging_arena.cpp:159-176`
   — it is in the destructor). The CN installs a SIGTERM handler and shuts down gracefully, but a CN
   whose engine thread is **wedged inside a fragment run** cannot join it and force-exits after
   `SHUTDOWN_GRACE = 15s` (`experimental/starrocks/src/main.rs:34,671-685`) — precisely the case
   under study. In the existing shared log, **48 of 96** arena lifetimes emitted a teardown line.
   Accept the loss on hangs: send SIGTERM, wait up to 30 s, escalate to SIGKILL, and record
   `sigkilled` in the CSV so a missing arena number is explained rather than mysterious.
6. **`restart-sf500x.sh`'s fixed 8 s sleep is shorter than that 15 s grace** (`restart-sf500x.sh:6`),
   so after a hang it can relaunch while the old CN still owns ports 9100–9104 — the new cluster then
   comes up with one CN and `wait_alive` burns its full 300 s before failing. This plan's
   `cluster_down` polls for process death instead. (`restart-sf500x.sh` is otherwise correct and its
   env propagation at lines 10-11 is essential: a version without those exports silently reverts to
   `up-sf500-x.sh`'s defaults and runs the wrong config — that happened once already.)
7. **The FE's `query_timeout` defaults to 300 s** and cuts healthy slow queries server-side no matter
   what the client does. `SET GLOBAL query_timeout = 1800` after **every** bring-up
   (`restart-sf500x.sh:14-19`). At 300 s a q21 hang would be misclassified before the CN's own 600 s
   error ever appears.
8. **The client timeout must exceed 600 s.** The CN's `wait_ready` deadline is 600 s
   (`compute_node_service.rs:421`), so a hang self-reports at ~600–620 s. Cutting the client earlier
   throws away the diagnostic message; 700 s is the value used here.
9. **`bench.sh` has no correctness gate** ("times and counts rows only", `bench.sh:53-56`), which is
   why this loop drives `mysql` directly and `cmp`s against the oracle on every pass.
10. **Each cluster restart costs ~70–90 s.** An N=30 fresh-cluster arm is `N × (restart + query)` —
    see the budget in §3.4, and do not start one without the hours to finish it.
11. **`bench.sh`'s `MIN_BACKENDS` is an expected size, not a floor** (`bench.sh:36-42`): satisfied by
    a half-booted cluster if set too low. This loop keeps the same semantics — 2 CNs, settled across
    two consecutive polls.
