# PLAN-05 — Fix the TPC-H benchmark harness (`bench.sh`)

**Status:** plan only. Nothing in this document has been implemented.
**Target file:** `experimental/starrocks/benchmarks/tpch/bench.sh` (201 lines at the time of writing).
**Repo:** `/home/ubuntu/sirius`, branch `demo-multi-cn`, HEAD `7af763c0` (default branch is `dev`).

---

## 0. Read this first if you have no context

`bench.sh` is the sweep harness that produced **every TPC-H number this project has published**
— `bench/rtxpro6000-2gpu/results/*.csv`, `bench/a100x8/results/*.csv`, and
`experimental/starrocks/benchmarks/tpch/results/*.csv`. It talks to a StarRocks FE on port 9030
over `mysql --batch`, times each query with `date +%s%3N`, and appends one CSV row per run.

It has five measured defects. Two of them (no correctness gate; 0-row answers filed as hangs)
mean the headline **"N/22 pass"** figure has never been a correctness figure. One of them (the FE's
own 300 s server-side timeout is never raised) has demonstrably converted at least one healthy
20 s query into a recorded failure.

There is substantial **prior art in this repo**: `experimental/starrocks/benchmarks/tpch/run-abc.sh`
(1500+ lines) already implements a correct result classifier, a provenance writer, SF-scaled
timeouts, a staged query set, and a no-cluster `--self-test`. It was written specifically because
`bench.sh` was wrong — see `run-abc.sh:40-44`. **This plan is mostly a port from `run-abc.sh` into
`bench.sh`, plus the one thing neither script has: a correctness gate.**

Everything below cites `file:line`. Claims that could not be checked from the filesystem in this
session are tagged **UNVERIFIED** and carry the command that would settle them.

---

## 1. Problem statement — the five defects

### D1. There is no correctness gate. At all.

`bench.sh:175` is the entire success test:

```bash
if [ $rc -eq 0 ] && [ -s "$f" ] && ! head -1 "$f" | grep -q ERROR; then
```

That is: the client exited 0, the output file is non-empty, and line 1 does not contain `ERROR`.
The answer is never compared to anything. The script says so itself at `bench.sh:54-56`:

> `NOTE: this script times and counts rows only -- it does not check answers.`

`analyze.py` compares the two engines' **row counts** to each other (`analyze.py:102-115`), which
catches an A-vs-B disagreement but cannot catch both engines being wrong, and does nothing at all
for a single-engine sweep. `experimental/starrocks/benchmarks/2NODE-REPLICATE.md:263-265` states
the consequence plainly: *"A query returning 1 row instead of 100,000 registers as a fast win."*

The correctness tooling exists but is invoked **by hand, out of band**, and lives in two places:

| Tool | Tracked copy | Scratch copy (identical, byte-for-byte checked this session) |
|---|---|---|
| `compare.py` | `bench/rtxpro6000-2gpu/tools/compare.py` | `/opt/dlami/nvme/sirius-build/compare.py` |
| `oracle.py` | `bench/rtxpro6000-2gpu/tools/oracle.py` | `/opt/dlami/nvme/sirius-build/oracle.py` |
| `drift.py` | `bench/rtxpro6000-2gpu/tools/drift.py` | — |
| `regress.py` | `bench/rtxpro6000-2gpu/tools/regress.py` | — |

* `compare.py:15-16` — `compare.py <sirius_out_dir> <oracle_dir> [rel_tol]`, default tolerance `1e-6`.
  It diffs `<q>.r<N>.out` (the files `bench.sh:169` writes) against `<q>.tsv`.
* `oracle.py:47-49` generates the oracle through the **pip** `duckdb` module, not
  `build/release/duckdb`. That matters: the repo's own `duckdb` binary auto-loads the Sirius
  extension and would contend with the CN for the GPU, so the oracle would not be an independent
  check. The venv on this box is `/opt/dlami/nvme/tpch/venv/bin/python` (duckdb **1.5.5**, verified).
* `oracle.py:24-27,52` rewrites `FILES("path"="file://…")` to `read_parquet('…')` and substitutes
  `__TPCH_DATA__` — i.e. **the oracle runs the same `.sql` files as the sweep**. Remember this for §5.

### D2. A correct 0-row answer is recorded as a wedge

`bench.sh:175`'s `[ -s "$f" ]` requires a non-empty file. `mysql --batch` prints **nothing at all —
not even the column header —** for a zero-row result set. So `{rc==0, empty file, no ERROR}`, which
is a query that ran to completion and correctly returned no rows, falls through to `bench.sh:186-188`
and is written as `wedge`.

`run-abc.sh:41-43` and `run-abc.sh:47-49` already document this exact bug.

**TPC-H q11 at SF ≥ 100 is this case, every time.** `queries/q11.sql:26` hardcodes:

```sql
            sum(ps_supplycost * ps_availqty) * 0.0001000000
```

The TPC-H spec scales that fraction as `0.0001/SF` (clause 2.11.2; the same constant is computed as
`0.0001/SF` at `run-abc.sh:292`). At SF500 the unscaled literal sets the bar 500× too high, so no
part clears it. Verified facts:

* `/opt/dlami/nvme/sirius-build/oracle-sf500f64/q11.tsv` is **17 bytes / 1 line** — the header
  `ps_partkey\tvalue` and nothing else. `oracle.py:60` always writes the header, so 1 line = 0 rows.
  Same for `oracle/`, `oracle-f64/`, `oracle-sf300f64/` — all 1 line.
* `/opt/dlami/nvme/sirius-build/bench/SF500XCOLD/q11.r0.out` and `q11.r1.out` are **0 bytes**.
* Sirius answered in **5280 ms** (cold) and **4626 ms** (warm) and was recorded
  `q11,0,cold,wedge,5280,0` / `q11,1,warm,wedge,4626,0`
  (`bench/rtxpro6000-2gpu/results/sf500xcold.csv`).
* **DuckDB and Sirius agree exactly.** The answer was right and the harness called it a hang.

Scope of the damage — **every SF ≥ 100 sweep in the repo**:

```
18 CSVs contain a `q11,…,wedge` row:
  bench/rtxpro6000-2gpu/results/{sf100-armA-40g16g, sf100-armB-60g32g, sf100-q08q09-fixed,
    sf100-q08q09-verified-21of22, sf100-decimal-final, sf100-float64-dataset,
    sf100-float64-q10fixed, sf300-float64, sf500-float64, sf500xcold}.csv
  bench/a100x8/results/{sf500-2cn, sf500-4cn, sf500-8cn, study1-2cn-32-46, study1-4cn,
    study1-8cn, study3-sf500, study3-sf1000}-timings.csv
```

At SF1 the literal is correct and q11 passes normally —
`experimental/starrocks/benchmarks/tpch/results/sf1-2026-08-07-A.csv` records
`q11,1,pass,830,1048`. So the failure is purely a function of scale factor.

### D3. The FE's server-side `query_timeout` is never set

`bench.sh:171` applies only a **client-side** cut:

```bash
    timeout "$tmo" $MYSQL -e "${Q}" > "$f" 2>&1
```

`$tmo` is `$QUERY_TIMEOUT` (default 30 s, `bench.sh:80`) or `$COLD_TIMEOUT` (default 180 s,
`bench.sh:81`). StarRocks has its **own** limit: `query_timeout`, default **300 seconds** —
verified in the vendored FE at
`experimental/starrocks/starrocks/fe/fe-core/src/main/java/com/starrocks/qe/SessionVariable.java:1316-1317`:

```java
    @VariableMgr.VarAttr(name = QUERY_TIMEOUT)
    private int queryTimeoutS = 300;
```

(the ceiling is `MAX_QUERY_TIMEOUT = 259200`, `SessionVariable.java:1196`). When it fires the FE
returns `ERROR 5024 (53400) … Query reached its timeout of 300 seconds`, which `bench.sh:183`
classifies as `refused` — indistinguishable in the CSV from an engine that refused the query.

**Measured, this session:**

| Run | Row | Evidence |
|---|---|---|
| `sweep-sf500x.sh` (no FE timeout set) | `q05,1,warm,refused,300104,0` in `results/sf500x.csv` | `/opt/dlami/nvme/sirius-build/bench/SF500X/q05.r1.out` contains verbatim `ERROR 5024 (53400) at line 1: Query reached its timeout of 300 seconds, please increase the 'query_timeout' session variable and retry` |
| `sweep-sf500x-cold.sh` (`FE_QUERY_TIMEOUT=1800`) | `q05,1,warm,pass,20423,5` in `results/sf500xcold.csv` | same query, same data, **20.4 s** |

The same query. A 300 s "failure" that is a 20 s success.

**Other scripts in this repo already fixed this; `bench.sh` never got the fix:**

* `benchmarks/nixl-nvlink/study1-run.sh:86-94` — `SET GLOBAL query_timeout = ${FE_QUERY_TIMEOUT:-900};`
  with a comment naming ERROR 5024 and the exact failure mode.
* `benchmarks/nixl-nvlink/study3-cost.sh:92-102` — same, **plus** the warning that
  `SET GLOBAL` is persisted through the FE edit log and is therefore *wiped* by a restart that does
  `rm -rf …/fe/meta` (as `study3-cost.sh:112` and `study1-run.sh:102` both do), so it must be
  re-applied inside `RESTART_CMD`.
* `/opt/dlami/nvme/sirius-build/restart-sf500x.sh:14-19` — re-applies it after each restart.

The persistence claim is confirmed in the FE source:
`experimental/starrocks/starrocks/fe/fe-core/src/main/java/com/starrocks/qe/VariableMgr.java:416-417`
writes a `GlobalVarPersistInfo` to the edit log on `SET GLOBAL`.

Note that `sweep-sf500x.sh` (the wrapper that produced `sf500x.csv`) sets it **nowhere** — not even
`FE_QUERY_TIMEOUT` — and `restart-sf500x.sh` only runs *after a failure*, so the first bring-up ran
the whole sweep at the stock 300 s. That is why q05 died there and lived in the `--cold-restart` run.

`bench/a100x8/` hit the same wall: `bench/a100x8/results/study1-8cn-raw.log:53` and
`study3-raw.log:120` both show `ERROR 5024` on q02.

**`run-abc.sh` does not fix this either** — grepping it for `query_timeout` returns nothing;
its only `SET GLOBAL` is `enable_pipeline_engine` at `run-abc.sh:791`. So this defect is live in
both harnesses.

### D4. The wedge message reports the configured cut, not the elapsed time

`bench.sh:188`:

```bash
      echo "$q r$r $phase WEDGE/TIMEOUT (rc=$rc, cut at ${tmo}s)"
```

`$tmo` is the *limit*, not the elapsed time, and the line prints it unconditionally. For q11 that
produced `WEDGE/TIMEOUT (rc=0, cut at 900s)` for a query that returned successfully in **5.3 s with
rc=0**. This actively misdirected analysis in this session — a 5 s correct answer read as a 900 s
hang. `$ms` is already computed at `bench.sh:174` and is simply not printed on the failure paths.

Two smaller defects on the same lines:

* `bench.sh:175` and `bench.sh:183` use `grep -q ERROR` **unanchored**, so a data row containing the
  substring `ERROR` anywhere on line 1 would be read as a refusal. `run-abc.sh:322` uses `'^ERROR'`.
* `bench.sh:183` re-runs the `head -1 | grep` test that `bench.sh:175` already ran — two independent
  copies of the same predicate, which is how they drift.

### D5. (Adjacent, out of scope for the code fix) `SIRIUS_LOG_BACKEND` is silently discarded on the CN path

`src/sirius_context.cpp:1550-1578` installs the log sink. An unrecognised backend throws — but only
on the `db != nullptr` branch:

```cpp
  } else if (db) {                                    // src/sirius_context.cpp:1573
    // Only report a bad backend on the db path; the db-less call is best-effort
    // and must not throw.
    throw InvalidInputException("Unknown sirius_log_backend '%s' (expected: duckdb, spdlog, noop)",
                                backend);             // :1576
  }
```

The CN calls it with `nullptr` — `src/sirius_ffi.cpp:170-177`:

```cpp
    const char* log_backend = std::getenv("SIRIUS_LOG_BACKEND");   // :170
    …
      duckdb::install_configured_log_sink(nullptr);                // :177
```

So `SIRIUS_LOG_BACKEND=console` (or any typo) on a CN silently produces **no logs**, and the run is
undiagnosable after the fact. The engine-side fix belongs in its own item. **What is in scope here**
is that the harness should validate the value and record it, so a sweep cannot silently run blind.
`/opt/dlami/nvme/sirius-build/up-sf500-x.sh:30-33` sets `spdlog` correctly, but nothing checks it.

### D6. (bonus, found while verifying) Config provenance is not recorded anywhere

`bench.sh` records `query,run,phase,status,ms,rows` (`bench.sh:160`) and nothing else. It does not
record which dataset, which memory split, which git SHA, or which cluster produced the row.
`restart-sf500x.sh:7-11` documents the specific trap:

> Propagate the caller's sizing: without this the restart silently falls back to `up-sf500-x.sh`'s
> defaults (60GiB/32GiB), which SILENTLY invalidates any experiment that set a different split.

That trap was fixed *in one restart script*. Any other restart script re-introduces it, and the CSV
cannot tell you which config it actually measured. `run-abc.sh:36-38` names this as a first-class
requirement and `run-abc.sh:868-889` implements it.

---

## 2. Proposed changes — diff level

All line numbers below refer to `experimental/starrocks/benchmarks/tpch/bench.sh` as of `7af763c0`.

### C1 — Replace the inline classifier with a function (fixes D2, D4, and the two `grep` nits)

**Remove** `bench.sh:175-189` (the `if`/`if`/`else` block) and **add**, next to the other helpers
(after `restart_cluster()`, i.e. after `bench.sh:138`):

```bash
# Classify one run. Sets $ST (status), $NROWS, $DETAIL. Kept as a function precisely so that
# --self-test can drive it with no cluster, no GPU and no query. Mirrors run-abc.sh:309-330,
# which was written against this script's bug.
#
#   ERROR on line 1            -> refused   The FE or the engine said no. rc is irrelevant here:
#                                           `mysql` exits 1 on a server error, but a query the FE
#                                           aborted at its own `query_timeout` ALSO writes an ERROR
#                                           line -- and that is a CONFIGURATION failure, not an
#                                           engine verdict, so it is tagged separately in $DETAIL.
#   rc != 0, no ERROR line     -> wedge     timeout(1) exits 124; anything else died mute.
#   rc == 0, empty file        -> pass, rows=0
#                                           `mysql --batch` prints NOTHING for a zero-row result
#                                           set -- not even the header. The old `[ -s "$f" ]` test
#                                           filed every correct empty answer as a wedge; that is how
#                                           q11 has been reported as a hang in 18 CSVs.
#   rc == 0, non-empty file    -> pass, rows = lines - 1 (the header)
classify_result() {                       # $1 = rc, $2 = output file
  local rc=$1 f=$2
  ST=; NROWS=0; DETAIL=

  # Anchored: an unanchored `grep -q ERROR` matches a DATA row that merely contains the word.
  if [ -s "$f" ] && head -1 "$f" 2>/dev/null | grep -q '^ERROR'; then
    ST=refused
    if head -1 "$f" | grep -q 'ERROR 5024'; then
      DETAIL=fe_query_timeout           # the FE's own clock ran out. NOT an engine result.
    else
      DETAIL=engine_error
    fi
    return
  fi

  if [ "$rc" -ne 0 ]; then
    ST=wedge
    if [ "$rc" -eq 124 ]; then DETAIL=client_cut; else DETAIL=died_mute_rc$rc; fi
    return
  fi

  ST=pass
  if [ ! -s "$f" ]; then
    # THE FIX. rc==0 + empty + no ERROR == the query completed and returned no rows.
    # rows is 0, NOT `wc -l` minus one -- that arithmetic on an empty file yields -1.
    NROWS=0
    DETAIL=zero_rows
    return
  fi
  NROWS=$(( $(wc -l < "$f") - 1 ))
  # Defensive: a one-line file with no trailing newline would otherwise report -1.
  [ "$NROWS" -lt 0 ] && NROWS=0
}
```

**The exact replacement for `[ -s "$f" ]`**, stated on its own because it is the crux:

```diff
-    if [ $rc -eq 0 ] && [ -s "$f" ] && ! head -1 "$f" | grep -q ERROR; then
-      rows=$(($(wc -l < "$f") - 1))
+    classify_result "$rc" "$f"
+    # `pass` now covers BOTH the non-empty and the correct-empty case; the two are told apart by
+    # $DETAIL (empty string vs `zero_rows`), not by the status.
```

and the new body of the loop replacing `bench.sh:175-198`:

```bash
    classify_result "$rc" "$f"
    if [ "$r" -gt 0 ] || [ "$COLD" = 1 ]; then
      echo "$q,$r,$phase,$ST,$ms,$NROWS,$DETAIL" >> "$OUT_CSV"
    fi
    case $ST in
      pass)
        if [ "$DETAIL" = zero_rows ]; then
          echo "$q r$r $phase pass ${ms}ms rows=0   (completed with no rows -- a RESULT, not a failure)"
        else
          echo "$q r$r $phase pass ${ms}ms rows=$NROWS"
        fi
        continue ;;
      refused)
        echo "$q r$r $phase REFUSED after ${ms}ms [$DETAIL]: $(head -c 160 "$f" | tr '\n' ' ')" ;;
      wedge)
        # D4: print the ELAPSED time. The cut is only meaningful when the client actually made it.
        if [ "$DETAIL" = client_cut ]; then
          echo "$q r$r $phase WEDGE: client killed it at ${ms}ms (the ${tmo}s cut)"
        else
          echo "$q r$r $phase WEDGE: died after ${ms}ms with rc=$rc, no output and no ERROR line"\
               "(the ${tmo}s cut was NOT reached)"
        fi ;;
    esac
    if [ "$COLD" = 1 ] && [ "$phase" = cold ]; then
      echo "  (cold failure recorded; continuing to warm runs on the same cluster)"
      continue
    fi
    restart_cluster || { echo "cluster did not recover"; exit 1; }
    break
```

Note the `case … pass) … continue` keeps `bench.sh:177-181`'s "run 0 is discarded unless `--cold`"
semantics intact, and `zero_rows` runs now continue to the warm runs instead of triggering
`restart_cluster` — restarting the whole cluster because a query correctly returned nothing is the
second-order cost of D2 and it disappears with the fix.

**Decision: `pass` + `DETAIL=zero_rows`, not a new `empty` status.** `run-abc.sh:320-330` uses a
distinct `empty` status. Do **not** copy that into `bench.sh`, because `analyze.py:51` and
`regress.py:24` both gate on `status == "pass"` — an `empty` status would make q11 vanish from every
comparison entirely, which is *worse* than today's visible-but-wrong `wedge`. Record the mapping
`run-abc "empty" == bench.sh "pass" + detail=zero_rows` in `README.md` so the two schemas can be
merged mechanically later.

### C2 — Set the FE's `query_timeout` (fixes D3)

Add to the env block after `bench.sh:84`:

```bash
# The FE aborts server-side at its OWN `query_timeout` (default 300 s --
# SessionVariable.java:1316) no matter what client-side cut this script uses, and reports
# ERROR 5024, which lands in the CSV as `refused` and reads exactly like an engine failure.
# Measured: q05 at SF500 was recorded `refused, 300104 ms`; with the FE timeout raised, the same
# query finished in 20423 ms.
#
# Scope:
#   session (DEFAULT) -- carried on the connection via --init-command, so it applies to this run
#                        and NOTHING ELSE. Survives a `rm -rf fe/meta` restart, because there is
#                        no FE state to survive. This is why it is the default.
#   global            -- one `SET GLOBAL` before the sweep, restored on exit. `SET GLOBAL` is
#                        persisted through the FE edit log (VariableMgr.java:416-417), so it
#                        outlives this script AND is silently lost by any RESTART_CMD that wipes
#                        the FE metadata (study3-cost.sh:96-100 documents exactly that bite).
#                        Use only if the FE rejects a session-scoped set.
#   none              -- opt out; historical behaviour.
FE_TIMEOUT_SCOPE=${FE_TIMEOUT_SCOPE:-session}
# Empty -> derived per phase from the client cut. The FE must give up BEFORE the client does: a
# server-side abort writes ERROR 5024, which names the cause; a client-side kill writes nothing at
# all and lands as an anonymous wedge.
FE_QUERY_TIMEOUT=${FE_QUERY_TIMEOUT:-}

if [ -n "$FE_QUERY_TIMEOUT" ]; then
  for pair in "warm:$QUERY_TIMEOUT" "cold:$COLD_TIMEOUT"; do
    if [ "$FE_QUERY_TIMEOUT" -ge "${pair#*:}" ]; then
      echo "FE_QUERY_TIMEOUT=$FE_QUERY_TIMEOUT >= the ${pair%%:*} client cut of ${pair#*:}s." >&2
      echo "  The client would kill first, so a slow query would be recorded as an anonymous" >&2
      echo "  wedge instead of a self-describing ERROR 5024. Raise the client cut instead." >&2
      exit 2
    fi
  done
fi
```

and inside the run loop, replacing `bench.sh:168-171`:

```bash
    if [ "$r" -eq 0 ]; then phase=cold; tmo=$COLD_TIMEOUT; else phase=warm; tmo=$QUERY_TIMEOUT; fi
    fe_tmo=${FE_QUERY_TIMEOUT:-$(( tmo > 20 ? tmo - 10 : tmo ))}
    init_opt=()
    [ "$FE_TIMEOUT_SCOPE" = session ] && init_opt=(--init-command="SET query_timeout = $fe_tmo")
    f=$OUT/$q.r$r.out
    t0=$(date +%s%3N)
    timeout "$tmo" $MYSQL "${init_opt[@]}" -e "${Q}" > "$f" 2>&1
```

For `FE_TIMEOUT_SCOPE=global`, add before the query loop (`bench.sh:162`):

```bash
if [ "$FE_TIMEOUT_SCOPE" = global ]; then
  FE_PREV_TIMEOUT=$($MYSQL -N -e "SHOW GLOBAL VARIABLES LIKE 'query_timeout';" 2>/dev/null | awk '{print $2}')
  $MYSQL -e "SET GLOBAL query_timeout = ${FE_QUERY_TIMEOUT:-$COLD_TIMEOUT};" >/dev/null 2>&1 \
    && echo "FE global query_timeout: $FE_PREV_TIMEOUT -> ${FE_QUERY_TIMEOUT:-$COLD_TIMEOUT}s (restored on exit)"
  # `SET GLOBAL` is written to the FE edit log, so it MUST be put back or it silently changes
  # every later sweep on this box.
  trap '[ -n "${FE_PREV_TIMEOUT:-}" ] && $MYSQL -e "SET GLOBAL query_timeout = $FE_PREV_TIMEOUT;" >/dev/null 2>&1' EXIT
  # A restart that wipes fe/meta drops it. Re-apply after every restart_cluster().
fi
```

and append the same `SET GLOBAL` to the tail of `restart_cluster()` (`bench.sh:133-138`) when the
scope is `global`.

**UNVERIFIED (2 items) — settle these before writing the code:**

1. Does the pixi `mysql` client accept `--init-command`?
   `pixi run --manifest-path experimental/starrocks/pixi.toml mysql --help | grep -i init-command`
2. Does a session-scoped `SET query_timeout` actually bound a StarRocks query? (It is a
   `VariableMgr.VarAttr` session variable, so it should — `SessionVariable.java:1316` — but the
   error text at `ERROR 5024` says *"please increase the 'query_timeout' session variable"*, which
   is good supporting evidence.) Empirical check against a live FE:
   `mysql … --init-command="SET query_timeout = 5" -e "SELECT sleep(30);"` → expect ERROR 5024
   naming 5 seconds.
   Fallback if `--init-command` is unavailable: prepend to the same `-e` string —
   `-e "SET query_timeout = $fe_tmo; ${Q}"`. The `mysql` client splits on `;` and runs both over one
   connection; `SET` emits no result set, so the output file is unchanged and the classifier is
   unaffected. This is strictly worse than `--init-command` because the TPC-H text is then no longer
   byte-identical to what the oracle runs.

### C3 — Emit a provenance manifest (fixes D6, and is what the gate consumes)

Add after the `wait_alive` gate (`bench.sh:156`), so `$ALIVE_CN` / `$ALIVE_BE` are populated:

```bash
MANIFEST=${OUT_CSV%.csv}.manifest.txt
snapshot_cn_config() {                  # $1 = destination dir
  # THE AUTHORITATIVE config is the one the CN processes were LAUNCHED with, not the one this
  # shell happens to have exported: a RESTART_CMD that forgets to re-export GPU_MEM/STAGING falls
  # back to the bring-up script's defaults and the sweep then reports under a DIFFERENT config than
  # the one requested (restart-sf500x.sh:7-11 documents exactly this).
  #
  # `pgrep -f sirius-starrocks-cn` also matches its own caller and any shell that merely mentions
  # the path (run-abc.sh:34-35), so resolve /proc/<pid>/exe instead.
  mkdir -p "$1"
  local pid exe cfg n=0
  for pid in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
    exe=$(readlink -f "/proc/$pid/exe" 2>/dev/null) || continue
    case $exe in *sirius-starrocks-cn) ;; *) continue ;; esac
    tr '\0' ' ' < "/proc/$pid/cmdline" > "$1/cn.$pid.cmdline" 2>/dev/null
    cfg=$(tr '\0' '\n' < "/proc/$pid/cmdline" 2>/dev/null | grep -A1 -x -- '--sirius-config' | tail -1)
    [ -n "$cfg" ] && [ -f "$cfg" ] && cp "$cfg" "$1/cn.$pid.$(basename "$cfg")"
    n=$((n + 1))
  done
  echo "$n"
}
{
  echo "bench_sh_schema=2"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host=$(hostname)"
  echo "out_csv=$OUT_CSV"
  echo "out_dir=$OUT"                 # where the .rN.out files the gate diffs are written
  echo "tpch_data=$TPCH_DATA"
  echo "tpch_data_bytes=$(du -sb "$TPCH_DATA" 2>/dev/null | cut -f1)"
  echo "tpch_data_fstype=$(df -PT "$TPCH_DATA" 2>/dev/null | awk 'NR==2 {print $2" on "$1}')"
  echo "queries_dir=$QUERIES_DIR"
  echo "queries=${QUERIES[*]}"
  # The single fact that binds a run to an oracle. See §3.
  echo "queryset_sha256=$(cat "$QUERIES_DIR"/q*.sql | sha256sum | cut -d' ' -f1)"
  echo "runs=$RUNS  cold=$COLD  cold_restart=$COLD_RESTART"
  echo "query_timeout_s=$QUERY_TIMEOUT"
  echo "cold_timeout_s=$COLD_TIMEOUT"
  echo "fe_timeout_scope=$FE_TIMEOUT_SCOPE"
  echo "fe_query_timeout=${FE_QUERY_TIMEOUT:-derived-from-client-cut}"
  echo "min_backends=$MIN_BACKENDS  alive_cn=$ALIVE_CN  alive_be=$ALIVE_BE"
  echo "restart_cmd=$RESTART_CMD"
  echo "fe_version=$($MYSQL -N -e 'SELECT current_version();' 2>/dev/null | head -1)"
  echo "cn_versions=$($MYSQL -e 'SHOW COMPUTE NODES;' 2>/dev/null |
        awk -F'\t' 'NR==1{for(i=1;i<=NF;i++) if($i=="Version") c=i; next} c{print $c}' |
        sort -u | paste -sd';')"
  echo "sirius_git_head=$(git -C "$HERE" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "sirius_git_dirty_files=$(git -C "$HERE" status --porcelain 2>/dev/null | wc -l)"
  echo "nvidia_driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
  echo "gpus_visible=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)"
  echo "env_sirius_log_backend=${SIRIUS_LOG_BACKEND:-<unset>}"
  echo "env_sirius_log_dir=${SIRIUS_LOG_DIR:-<unset>}"
  echo "env_sirius_log_level=${SIRIUS_LOG_LEVEL:-<unset>}"
  echo "env_gpu_mem=${GPU_MEM:-<unset>}  env_staging=${STAGING:-<unset>}  env_host_mem=${HOST_MEM:-<unset>}"
  echo "cn_config_snapshot=$OUT/config-start  cn_processes=$(snapshot_cn_config "$OUT/config-start")"
} > "$MANIFEST"
echo "manifest: $MANIFEST"
```

`SHOW COMPUTE NODES` really does carry a `Version` column — it is column 13 of
`ComputeNodeProcDir.TITLE_NAMES`
(`experimental/starrocks/starrocks/fe/fe-core/src/main/java/com/starrocks/common/proc/ComputeNodeProcDir.java:49-54`).
`current_version()` is a real FE function (`FunctionSet.java:295`); its exact output string is
**UNVERIFIED** (no cluster was running this session) — check with
`mysql … -N -e 'SELECT current_version();'`.

At the very end of the sweep (replacing `bench.sh:201`), snapshot again and diff:

```bash
snapshot_cn_config "$OUT/config-end" > /dev/null
if ! diff -rq "$OUT/config-start" "$OUT/config-end" >/dev/null 2>&1; then
  echo "config_changed_midsweep=1" >> "$MANIFEST"
  echo "!! WARNING: the CN configuration CHANGED during this sweep." >&2
  echo "   Rows before and after the change were measured under DIFFERENT configs." >&2
  diff -rq "$OUT/config-start" "$OUT/config-end" >&2
else
  echo "config_changed_midsweep=0" >> "$MANIFEST"
fi
echo "== bench complete: $OUT_CSV (manifest: $MANIFEST) =="
```

This is the direct fix for the trap in the task brief: a restart that silently changes the memory
split now shows up as `config_changed_midsweep=1` in the artifact rather than as a mystery in the
timings.

### C4 — Validate the logging env (partially addresses D5)

Add next to the other gates (after `bench.sh:155`):

```bash
# The CN installs its log sink through the FFI path, which passes a null DatabaseInstance
# (src/sirius_ffi.cpp:177). The "unknown backend" throw in install_configured_log_sink is guarded
# by `else if (db)` (src/sirius_context.cpp:1573-1577), so on a CN a typo'd value is SILENTLY
# discarded and the run produces no engine logs at all. Fail here instead, loudly, before an
# 8-hour sweep turns out to be undiagnosable.
case ${SIRIUS_LOG_BACKEND:-<unset>} in
  duckdb|spdlog|noop) ;;
  '<unset>')
    echo "note: SIRIUS_LOG_BACKEND is unset -- the CNs will emit no engine logs, so a stall or a" >&2
    echo "      downgrade in this sweep will not be diagnosable. Set spdlog + SIRIUS_LOG_DIR." >&2 ;;
  *)
    echo "SIRIUS_LOG_BACKEND='$SIRIUS_LOG_BACKEND' is not one of duckdb|spdlog|noop." >&2
    echo "  The CN accepts it silently and then logs NOTHING (sirius_context.cpp:1573)." >&2
    exit 2 ;;
esac
```

The engine-side fix (make `install_configured_log_sink` reject a bad backend even with `db ==
nullptr`, or have `sirius_ffi.cpp:173-177` validate before assigning) is **out of scope for this
plan** — it is a one-line change in `src/`, and belongs with the config-validation work.

### C5 — Record queries the sweep never reached

Port `run-abc.sh:365`. When `restart_cluster` fails and the script exits (`bench.sh:197`), or when
a query's runs are abandoned, the remaining queries are simply **absent** from the CSV — and
`analyze.py` cannot distinguish "never ran" from "not in the requested subset", so a sweep that died
at q09 reads as a 9-query suite that passed. Emit a row per unreached query:

```bash
record_unrun() { for q in "$@"; do echo "$q,0,cold,unrun,0,-1," >> "$OUT_CSV"; done; }
```

`rows=-1` marks it unknown (`analyze.py:48` reads it as an int; `-1` will never equal a real count,
so it can never be mistaken for agreement). `status=unrun` is a **new status value** — see §6.

### C6 — `--self-test` and `QUERIES_DIR`

* Add `QUERIES_DIR=${QUERIES_DIR:-$HERE/queries}` and use it at `bench.sh:76` and `bench.sh:163` in
  place of `$HERE/queries`. This is what makes the live fixture tests in §7 possible without
  polluting the 22-query default.
* Add `--self-test` (see §7A), modelled on `run-abc.sh:371-542`.

---

## 3. Correctness gate — design and recommendation

### Recommendation: **a manifest + a separate gate**, with an opt-in `--verify` convenience flag.

`bench.sh` writes `<out>.manifest.txt` (C3) and a new `verify.sh` next to it consumes that manifest.
`bench.sh --verify` simply execs `verify.sh "$MANIFEST"` as its last act, so the common path stays
one command.

**Why not have `bench.sh` call `compare.py` inline:**

1. **The gate must be re-runnable without the cluster.** That is not hypothetical — it is exactly
   what happened this session: `/opt/dlami/nvme/sirius-build/bench/SF500XCOLD/*.out` were diffed
   against `oracle-sf500f64/` *after* the sweep finished, and that is where the q11 finding came
   from. Coupling the gate to the sweep would have destroyed that workflow. An 8-hour SF500 sweep
   must never have to be re-run because the tolerance was wrong.
2. **Different dependency footprints.** `bench.sh` today needs only `bash`, `mysql`, `timeout`,
   `awk`. `compare.py` needs `python3`. `oracle.py` needs a pip `duckdb` in a venv
   (`/opt/dlami/nvme/tpch/venv`, duckdb 1.5.5) that is deliberately **not** the repo's
   `build/release/duckdb`, because that binary auto-loads Sirius and would contend with the CN for
   the GPU. Folding all three into the timing path makes a sweep fail on a box that has only the
   first set.
3. **The oracle is a separate, slow, cluster-free artifact.** It is generated once per dataset and
   reused across many sweeps. Its lifecycle does not belong inside a per-sweep script.
4. **Separation of duties is the point.** A timing harness that also grades itself is a harness that
   can grade itself leniently. `analyze.py` and `regress.py` are already separate consumers of the
   same CSV; the gate is the third.
5. `run-abc.sh` already chose the manifest shape (`run-abc.sh:36-38`, `:868-889`) for the same
   reasons. Two harnesses with one artifact format is strictly better than two bespoke gates.

### `verify.sh` — new file, `experimental/starrocks/benchmarks/tpch/verify.sh`

```
usage: verify.sh <run.manifest.txt> [rel_tol]
env:   ORACLE_DIR   explicit oracle directory; otherwise resolved from the manifest (see below)
```

Steps:

1. Parse the manifest into shell vars (`key=value`, one per line — sourceable *and* trivially
   parsed by `dict(l.split('=', 1) for l in open(p))`).
2. Resolve the oracle directory: `$ORACLE_DIR`, else a `oracle_dir=` key in the manifest, else
   **fail loudly**. Never guess — silently comparing against the wrong-SF oracle is the failure mode
   the whole gate exists to prevent.
3. Read `$ORACLE_DIR/ORACLE.txt` (new; see below) and assert the binding.
4. `compare.py "$out_dir" "$ORACLE_DIR" "$tol"` → tee to `$out_dir/verdicts.txt`.
5. Write `${manifest%.manifest.txt}.verdicts.csv` with `query,verdict,detail,sirius_rows,oracle_rows`.
6. Exit 1 unless **every query named in `manifest.queries`** has verdict `MATCH`.

### Binding a run to its oracle — how it is expressed and checked

Oracles are per-dataset. On this box:

| Oracle dir | Dataset |
|---|---|
| `/opt/dlami/nvme/sirius-build/oracle` | `/opt/dlami/nvme/tpch/tpch_parquet_sf100` (decimal) |
| `/opt/dlami/nvme/sirius-build/oracle-f64` | `…/tpch_parquet_sf100_f64` |
| `/opt/dlami/nvme/sirius-build/oracle-sf300f64` | `…/tpch_parquet_sf300_f64` |
| `/opt/dlami/nvme/sirius-build/oracle-sf500f64` | `…/tpch_parquet_sf500_f64` |

**Nothing currently records that mapping** — verified: none of the four oracle dirs contains any
file other than `q*.tsv`. The mapping lives only in the wrapper scripts and in someone's head.
Comparing an SF500 run against `oracle-f64` would produce 22 confident `ROWS-DIFFER` verdicts and
no clue why.

**Fix: `oracle.py` writes `ORACLE.txt` into its output directory**, and `verify.sh` refuses to
proceed unless it matches. Add to `oracle.py` after `os.makedirs(out, ...)` (`oracle.py:21`):

```python
with open(os.path.join(out, "ORACLE.txt"), "w") as f:
    f.write(f"dataset={data}\n")
    f.write(f"dataset_bytes={sum(os.path.getsize(p) for p in glob.glob(data + '/*/*.parquet'))}\n")
    f.write(f"queries_dir={qdir}\n")
    f.write("queryset_sha256=" + hashlib.sha256(
        b"".join(open(os.path.join(qdir, f"{n}.sql"), "rb").read() for n in names)).hexdigest() + "\n")
    f.write(f"duckdb_version={duckdb.__version__}\n")
    f.write(f"generated_utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
```

`verify.sh` then asserts three things, each with its own error message:

| Assertion | Catches |
|---|---|
| `ORACLE.dataset == manifest.tpch_data` | the SF500-run-vs-SF100-oracle mistake |
| `ORACLE.dataset_bytes == manifest.tpch_data_bytes` | the dataset was regenerated/moved under the same path |
| **`ORACLE.queryset_sha256 == manifest.queryset_sha256`** | the oracle was generated from *different query text* |

The third is the important one and it is not obvious. `oracle.py:52` reads the same `.sql` files the
sweep does, so **any edit to `queries/` invalidates every existing oracle**. Commit `7af763c0`
("resolve q08 and q09 failures by reordering FROM clause") did exactly that. The hash makes the
staleness a hard failure instead of a silent comparison against the old plan's answers.

Note the sha covers the whole query set, so editing q09 invalidates the oracle for q01 too. That is
deliberately conservative: it is one `oracle.py` re-run (minutes) versus a wrong verdict.

### One more hole to fix in `compare.py`

`compare.py:35` iterates `glob(sdir + "/q*.out")` — **the files that exist**. A query the sweep never
reached has no `.out` file, so it is simply absent from the report, and the summary line
`compare.py:84-85` reads `19/19 match` instead of `19/22`. `verify.sh` must iterate
`manifest.queries` and treat a missing `.out` as `NO-RESULT`, not as a query that was not asked for.
(Alternatively push a `--expect "q01 q02 …"` argument into `compare.py`; the tracked copy is
`bench/rtxpro6000-2gpu/tools/compare.py`, which **is** in `black`'s scope — see §3, "Canonical location".)

### Canonical location for the tools

`compare.py` and `oracle.py` exist twice, byte-identical: `bench/rtxpro6000-2gpu/tools/` and
`/opt/dlami/nvme/sirius-build/` (untracked scratch). They will drift. **Recommendation:** make
`experimental/starrocks/benchmarks/tpch/` canonical — the gate should live beside the harness it
gates — and document `bench/rtxpro6000-2gpu/tools/` as a frozen snapshot of the versions that
produced the archived numbers. Tradeoff to accept knowingly: `.pre-commit-config.yaml:18` excludes
`^experimental/.*` from all hooks, so moving them there removes them from `black`.

---

## 4. Provenance recorded in the output

Covered by C3. Summary of what lands where:

| Artifact | Contents |
|---|---|
| `<out>.csv` | unchanged schema + two appended columns (§6) |
| `<out>.manifest.txt` | dataset path + bytes + filesystem, queryset sha256, git HEAD + dirty count, FE version, per-CN versions, alive CN/BE counts, both timeouts and their scope, `RESTART_CMD`, driver version, GPU count, all `SIRIUS_LOG_*`, all `GPU_MEM`/`STAGING`/`HOST_MEM`, UTC start |
| `<out_dir>/config-start/` | the `--sirius-config` YAML each CN was **actually launched with**, plus its `/proc/<pid>/cmdline` |
| `<out_dir>/config-end/` | same, at sweep end; `config_changed_midsweep=0\|1` appended to the manifest |
| `<out_dir>/q*.rN.out` | unchanged (already written by `bench.sh:169`); this is what the gate diffs |
| `<out>.verdicts.csv` | written by `verify.sh` |

**Keeping the CSV and its manifest together.** The results in `bench/*/results/*.csv` are copies of
`timings.csv` detached from their output directories — which is why nothing about those runs is
recoverable today. Two mitigations, both cheap:

* Name the sidecar off the same stem (`sf500xcold.csv` → `sf500xcold.manifest.txt`) so a copy step
  that globs the stem takes both.
* Add a `run_id` column to the CSV whose value is `sha256(manifest)[:12]`, repeated on every row.
  An orphaned CSV can then still be matched back to its manifest, and two concatenated sweeps stay
  separable. Optional but recommended.

---

## 5. q11 specifically

### 5.1 What to do now

1. **C1 makes q11 record as `pass, ms, 0` with `detail=zero_rows`.** No query-specific code — the
   general fix is sufficient and q11 is simply its most visible instance.
2. **Add a `QUERY-DEVIATIONS.md` entry** at `experimental/starrocks/benchmarks/tpch/QUERY-DEVIATIONS.md`,
   in the style of the existing q08/q09 entry (`QUERY-DEVIATIONS.md:11-64`: heading with date, a
   `diff` fence, "Semantically identical / Why / Measured / The principled fix"). Note the file's own
   rule at `QUERY-DEVIATIONS.md:5-7` — **the note goes in the `.md`, never as a `--` comment in the
   `.sql`**, because `mysql -e` collapses newlines and a leading `--` swallows the whole statement.
   Draft:

   > ## q11 — FRACTION left at the SF1 value `0.0001` (2026-08-20)
   >
   > The stock text is `0.0001/SF` (spec clause 2.11.2). `queries/q11.sql:26` hardcodes
   > `0.0001000000`, which is correct **only at SF1**.
   >
   > ### Consequence
   >
   > At SF ≥ 100 the HAVING bar is SF× too high and **the correct answer is zero rows**. Verified:
   > `oracle-sf500f64/q11.tsv` is 17 bytes — the header and nothing else; DuckDB returns 0 rows;
   > Sirius returns 0 rows in 5.3 s; they MATCH. At SF1,
   > `results/sf1-2026-08-07-A.csv` records `q11,1,pass,830,1048` — the query is only meaningful
   > there.
   >
   > This is **not an engine defect and not a hang.** Until `7af763c0`+, `bench.sh:175`'s
   > `[ -s "$f" ]` test filed the empty answer as `wedge`, which is how q11 appears as a
   > multi-second "hang" in 18 archived CSVs across `bench/rtxpro6000-2gpu/results/` and
   > `bench/a100x8/results/`. Any pre-fix "N/22" at SF ≥ 100 **undercounts by one**.
   >
   > ### Why the literal is not simply corrected
   >
   > See PLAN-05 §5.2. Short version: one `q11.sql` serves SF1/100/300/500, `bench.sh` substitutes
   > only `__TPCH_DATA__`, and the oracle runs the *same file* — so a hardcoded scaled constant
   > would be invisible to the correctness gate.

3. **Do not rewrite the 18 archived CSVs.** See §6.

### 5.2 Explicit recommendation AGAINST hardcoding a scaled constant into `queries/q11.sql`

Do **not** change `queries/q11.sql:26` from `0.0001000000` to `0.0000002` (or any other
SF-specific value). Three independent reasons:

1. **One file serves four scale factors.** `queries/q11.sql` is used unmodified against
   `tpch_parquet_sf100`, `…_sf100_f64`, `…_sf300_f64`, `…_sf500_f64` and the SF1 dataset. The only
   substitution `bench.sh` performs is `__TPCH_DATA__` (`bench.sh:163`:
   `sed "s|__TPCH_DATA__|$TPCH_DATA|g"`). A constant tuned for SF500 silently breaks SF1 — where
   q11 currently returns 1048 rows and is a real query.
2. **The correctness gate would be structurally blind to it.** `oracle.py:52` reads the *same*
   `queries/qNN.sql` and applies the *same* `__TPCH_DATA__` substitution. Whatever constant is in
   the file, both sides use it. The oracle and the engine would move together, `compare.py` would
   report `MATCH`, and a wrong constant would be indistinguishable from a right one. A gate that
   cannot detect the error it is supposed to guard is worse than no gate, because it manufactures
   confidence.
3. **It changes what is being benchmarked without changing the CSV.** Historical A/B timings for
   q11 would silently become non-comparable to new ones.

### 5.3 If spec compliance is wanted later — the parameterised design

1. Replace the literal in `queries/q11.sql:26` with a placeholder:
   `sum(ps_supplycost * ps_availqty) * __TPCH_Q11_FRACTION__`.
2. Add a **second `sed`** to `bench.sh:163`:
   ```bash
   Q=$(sed -e "s|__TPCH_DATA__|$TPCH_DATA|g" -e "s|__TPCH_Q11_FRACTION__|$Q11_FRACTION|g" \
        "$QUERIES_DIR/$q.sql")
   ```
   with `Q11_FRACTION` computed as `awk -v sf="$SF" 'BEGIN { printf "%.12f", 0.0001 / sf }'` —
   `run-abc.sh:292` already has exactly this helper, and 12 decimals is enough down to SF10000
   (`run-abc.sh:291`).
3. Add the **same** substitution to `oracle.py:52`, and record the value in `ORACLE.txt`.
4. Record `q11_fraction=` in the manifest, and make `verify.sh` assert
   `manifest.q11_fraction == ORACLE.q11_fraction`. **This assertion is what makes the
   parameterised form safe where the hardcoded form is not** — the value is now data flowing
   through both paths and checkable, rather than a constant baked into both.
5. **Introduce `SF` as a required input to `bench.sh`.** It does not have one today: the scale
   factor is implicit in `$TPCH_DATA`. `run-abc.sh` takes `--sf` and derives the timeouts from it
   (`run-abc.sh:282-290`), which is a second good reason to add it.
6. **Regenerate all four oracle directories** (`oracle`, `oracle-f64`, `oracle-sf300f64`,
   `oracle-sf500f64`) — the queryset sha changes, so C3/§3's binding check will (correctly) reject
   every existing oracle until they are rebuilt.
7. Alternatively adopt `run-abc.sh`'s **staged-queryset** approach (`run-abc.sh:1443-1453`): copy
   `queries/*.sql` into `$OUT/queryset/`, rewrite the literal there, and never touch the repo file.
   `bench.sh` would need `QUERIES_DIR` (C6) and `oracle.py` would have to be pointed at the same
   staged directory — otherwise the two diverge and reason 2 above reappears.

**Flag before doing any of this:** parameterising turns q11 from a 5-second no-op into a real
query. At SF100 the spec fraction yields **92,698 rows** (`run-abc.sh:52`, measured against engine
C / cudf-polars). Extrapolating linearly to SF500 gives roughly **5×10⁵ rows** — an aggregate over
the full 400M-row `partsupp` with a large ordered output. **Its SF500 cost is entirely unmeasured
(UNVERIFIED — the row-count extrapolation is linear-scaling arithmetic, not a measurement).** It is
a plausible new OOM candidate on a box where q09 already fails and q21 intermittently stalls. Budget
a probe run before committing to it, and expect the 21/22 headline to be at risk.

---

## 6. Backward compatibility

### CSV schema

Current header, written at `bench.sh:160`:

```
query,run,phase,status,ms,rows
```

Proposed:

```
query,run,phase,status,ms,rows,detail[,run_id]
```

**Append only. Never reorder, never rename.** Every in-repo consumer uses `csv.DictReader`, so extra
trailing columns are ignored and missing ones are handled with `.get()`:

| Consumer | Line | Reads |
|---|---|---|
| `analyze.py` | `:41` `csv.DictReader(f)` | `query, run, status, ms, rows`, `.get("phase")` |
| `regress.py` | `:22` `csv.DictReader(open(path))` | `query, status, ms`, `.get("phase","warm")` |
| `compare.py` | — | does not read the CSV at all; it reads `<q>.r<N>.out` and `<q>.tsv` |
| `drift.py` | — | does not read the CSV |
| `run-comparison.sh` | `:53,:68,:73` | passes paths through to `analyze.py` |

The oldest CSVs have a **5-column** header with no `phase`
(`experimental/starrocks/benchmarks/tpch/results/sf1-2026-08-07-A.csv`:
`query,run,status,ms,rows`). `analyze.py:42-44` already handles that:

```python
            # Older CSVs have no phase column. bench.sh only ever wrote a run-0 row
            # when run 0 failed, so run==0 there means the same thing as phase=cold.
            phase = row.get("phase") or ("cold" if row["run"] == "0" else "warm")
```

That path is untouched.

### Status vocabulary

| Value | Status |
|---|---|
| `pass`, `refused`, `wedge` | unchanged, same meaning |
| `pass` with `rows=0` | **new situation, existing value.** Previously written as `wedge`. `analyze.py:51-54` will now count q11 as passing and add `0` to `entry["rows"]`; A-vs-B agreement still works because both engines return 0. |
| `unrun` (C5) | **new value.** `analyze.py:55-56` falls into `elif entry["status"] != "pass": entry["status"] = status`, so it displays as `unrun` in the table and is excluded from the geomean. `regress.py:23-25` only records it in the status set. Both degrade correctly; no change required, but verify with the fixture in §7B4. |

### The 18 archived CSVs with `q11,…,wedge`

**Do not rewrite them.** They are the record of what the harness actually reported; editing them
destroys the audit trail and makes the defect unprovable. Instead:

* Add the §5.1 `QUERY-DEVIATIONS.md` entry stating that any `q11` `wedge` row in an SF ≥ 100 sweep
  is a correct 0-row answer and that pre-fix "N/22" figures undercount by one.
* Add one line to `bench/rtxpro6000-2gpu/README.md` and `experimental/starrocks/benchmarks/tpch/README.md`
  pointing at it.
* Optionally ship `tools/relabel-q11.py --dry-run` that *prints* the corrected counts from a CSV
  without modifying it. **Do not** add a q11 special case to `analyze.py` or `regress.py` — a
  hardcoded per-query exception in the analysis tool is how you get an analysis tool that lies.

### Documentation that must be updated with the code

* `experimental/starrocks/benchmarks/tpch/README.md:12-20` — describes the classifier and the
  timeouts.
* `experimental/starrocks/benchmarks/tpch/REPRODUCE.md:35-40` — "hangs cut at 30 s".
* `experimental/starrocks/benchmarks/2NODE-REPLICATE.md:263-265` — "The harness has no correctness
  gate", which becomes false.
* `experimental/starrocks/benchmarks/tpch/run-abc.sh:40-44` — the "does not reuse bench.sh because
  bench.sh:175 is wrong" rationale; the line numbers there will drift and the reason changes.
* `bench.sh:51-56` — the header comment that says the script does not check answers.

---

## 7. Tests

### A. `bench.sh --self-test` — no cluster, no GPU, no query

Port the scaffold from `run-abc.sh:371-542` (`ok`/`bad`/`check` helpers, `mktemp -d` + `trap`,
exit 1 on any failure). Assertions:

| # | Input | Expect |
|---|---|---|
| 1 | file starting `ERROR 1064 …`, rc=1 | `refused / 0 / engine_error` |
| 2 | same file, rc=0 | `refused / 0 / engine_error` |
| 3 | file starting `ERROR 5024 (53400) … timeout of 300 seconds`, rc=1 | `refused / 0 / fe_query_timeout` |
| 4 | **empty file, rc=0** | **`pass / 0 / zero_rows`** ← D2 |
| 5 | empty file, rc=124 | `wedge / 0 / client_cut` |
| 6 | empty file, rc=1 | `wedge / 0 / died_mute_rc1` |
| 7 | header + 2 data rows, rc=0 | `pass / 2 / (empty detail)` |
| 8 | header only, no trailing newline, rc=0 | `pass / 0` (the `< 0` clamp) |
| 9 | one data row containing the word `ERROR` in a column, rc=0 | `pass / 1` ← the anchoring fix |
| 10 | `FE_QUERY_TIMEOUT=40 QUERY_TIMEOUT=30` | exit 2 with the "client would kill first" message |
| 11 | `QUERY_TIMEOUT=300` unset `FE_QUERY_TIMEOUT` | derived warm `fe_tmo` = 290 |
| 12 | `SIRIUS_LOG_BACKEND=console` | exit 2 |
| 13 | manifest writer over a fake env | every documented key present, exactly once |

`run-abc.sh:447-457` is a working template for rows 1–7 — including the comment that names this
exact bug.

### B. Live fixtures — needs a running FE

Put them in a **new** `experimental/starrocks/benchmarks/tpch/test-fixtures/` (not `queries/`, so
the default 22-query set at `bench.sh:76` is unchanged) and drive them via `QUERIES_DIR` (C6).

**B1 — a query that correctly returns 0 rows** (`test-fixtures/zero.sql`):

```sql
nation AS (SELECT * FROM FILES("path"="file://__TPCH_DATA__/nation/*.parquet","format"="parquet"))
```
…wrapped as `WITH nation AS (…) SELECT n_nationkey FROM nation WHERE n_nationkey < 0;`
— guaranteed empty, sub-second, and it exercises the same `FILES()` scan path as the real queries.

Expect: `zero,1,warm,pass,<ms>,0,zero_rows`, the console line reading
`pass … rows=0   (completed with no rows -- a RESULT, not a failure)`, and **no cluster restart**.
Pre-fix this run produced `wedge` **and** triggered `restart_cluster` (`bench.sh:197`).

The real-world instance is q11 itself at SF ≥ 100 — assert it too if a SF500 dataset is up:
`QUERIES_DIR=… TPCH_DATA=/opt/dlami/nvme/tpch/tpch_parquet_sf500_f64 bench.sh --cold /tmp/t.csv 1 q11`
must yield `q11,0,cold,pass,~5000,0,zero_rows`.

**B2 — a healthy-but-slow query** (`test-fixtures/slow.sql`): `SELECT sleep(45);`
(`sleep(INT) -> BOOLEAN` exists at
`experimental/starrocks/starrocks/gensrc/script/functions.py:840` and
`FunctionSet.java:301`. **UNVERIFIED** that the Sirius CN implements it — it may downgrade to
DuckDB or refuse. Fallback: q07 at SF500, measured at **289,294 ms** warm in
`results/sf500xcold.csv`.)

Run twice, and the pair is the direct reproduction of the q05 300104 → 20423 finding:

| Setting | Expect |
|---|---|
| `QUERY_TIMEOUT=120 FE_QUERY_TIMEOUT=20` | `refused` with `detail=fe_query_timeout`, `ms≈20000`, `.out` containing `ERROR 5024` |
| `QUERY_TIMEOUT=120 FE_QUERY_TIMEOUT=90` | `pass`, `ms≈45000` |

**B3 — a genuine wedge.** Two levels, because a real engine hang is not reproducible on demand:

* *Client-cut wedge, no hung engine:* `QUERY_TIMEOUT=1 bench.sh /tmp/t.csv 1 q01`. `timeout` fires,
  rc=124 → `wedge / client_cut`, and the console line must read
  `client killed it at ~1000ms (the 1s cut)` — **not** a number the run never reached (D4).
* *Genuine engine wedge:* `kill -STOP` one CN process mid-query, then run a query that needs it.
  The FE's `query_timeout` fires first (by construction, C2) → `refused / fe_query_timeout`; with
  `FE_TIMEOUT_SCOPE=none` the client fires → `wedge / client_cut`. Both are correct and the pair
  demonstrates that the two are now distinguishable. Clean up with `kill -CONT` **and** a full
  cluster restart — a stopped CN strands fragments (`README.md:42-46`).

**B4 — `unrun` rows (C5):** point `RESTART_CMD` at `false`, force a failure on the second of four
queries, and assert the last two appear as `…,unrun,0,-1,` and that `analyze.py` renders them
without crashing.

### C. Gate regression against a known-answer corpus — no cluster needed

The archived outputs from this session are a ready-made fixture:

```bash
verify.sh /opt/dlami/nvme/sirius-build/bench/SF500XCOLD/timings.manifest.txt
#   (the manifest has to be back-filled by hand for this one archived run)
# or directly:
compare.py /opt/dlami/nvme/sirius-build/bench/SF500XCOLD \
           /opt/dlami/nvme/sirius-build/oracle-sf500f64
```

Expected, per `bench/rtxpro6000-2gpu/SF500-CONFIG-AND-ARCHITECTURE.md:33-42`: **19 MATCH**, q11
MATCH at 0 rows, q08/q09 non-MATCH (`ERROR`), exit 1. This validates the gate itself against an
answer that is already known, with no cluster and no GPU.

Also assert the **binding** rejects a mismatch:
`ORACLE_DIR=/opt/dlami/nvme/sirius-build/oracle-f64 verify.sh <sf500 manifest>` must fail with
a dataset-mismatch message, not with 22 confusing `ROWS-DIFFER` lines.

### D. Lint

`.pre-commit-config.yaml:18` excludes `^experimental/.*` from **all** hooks, so `bench.sh`,
`verify.sh`, `analyze.py` and `run-abc.sh` are not linted. `bench/rtxpro6000-2gpu/tools/*.py` **is**
in scope and must pass `black` (`.pre-commit-config.yaml:51-54`):

```bash
pixi run pre-commit run -a
```

`shellcheck` is not configured anywhere in the repo — run it manually on `bench.sh` and `verify.sh`
before opening a PR.

---

## 8. Success criteria

1. `bash bench.sh --self-test` exits 0 with every assertion in §7A passing, on a box with no GPU,
   no cluster and no dataset.
2. A fresh SF500 sweep records **`q11,*,*,pass,*,0,zero_rows`**, and does not restart the cluster
   for it.
3. Every `refused` row carries `detail=fe_query_timeout` or `detail=engine_error`, so a
   configuration limit can never again be read as an engine verdict. The B2 fixture pair
   (`refused@20s` → `pass@45s`) reproduces on demand.
4. Every `wedge` console line prints the **elapsed** ms, and mentions the configured cut only when
   the client actually reached it.
5. No sweep can produce a CSV without a sibling `*.manifest.txt`; the manifest names the dataset,
   the queryset sha256, the git HEAD, the FE and CN versions, and snapshots the `--sirius-config`
   YAML the CNs were **actually launched with**. A restart that changes the config sets
   `config_changed_midsweep=1` and prints a warning.
6. `verify.sh` refuses to compare a run against an oracle whose `dataset`, `dataset_bytes`, or
   `queryset_sha256` disagrees with the manifest, with a message naming which one.
7. `verify.sh` on `/opt/dlami/nvme/sirius-build/bench/SF500XCOLD` + `oracle-sf500f64` reproduces
   19 MATCH + q11 MATCH(0 rows) + q08/q09 non-MATCH and exits 1.
8. `verify.sh` reports `N/22`, not `N/N`, when the sweep did not reach every query.
9. `analyze.py`, `regress.py` and `run-comparison.sh` run **unmodified** against both a new CSV and
   `experimental/starrocks/benchmarks/tpch/results/sf1-2026-08-07-A.csv` (5 columns, no `phase`).
10. `SIRIUS_LOG_BACKEND` set to anything outside `{duckdb, spdlog, noop}` aborts the sweep before
    the first query.
11. The 18 archived CSVs are **byte-identical** after the change; the mislabel is documented in
    `QUERY-DEVIATIONS.md`, not patched out of the data.
12. `pixi run pre-commit run -a` is clean.

---

## 9. Explicitly out of scope

* The `SIRIUS_LOG_BACKEND` engine-side fix (`src/sirius_context.cpp:1573-1577`,
  `src/sirius_ffi.cpp:170-177`) — D5's code change. This plan only adds harness-side validation.
* Merging `bench.sh` and `run-abc.sh` into one harness. They have diverged deliberately
  (`run-abc.sh:40-44`, `:58-60`). Aligning the CSV schema (§6) is the prerequisite for that; doing
  it is a separate item.
* Actually parameterising q11 (§5.3) — this plan recommends **against** it for now and describes
  what it would take.
* SF-derived timeouts (`run-abc.sh:282-290`). Worth porting; not required by any defect above.
* Regenerating the oracles.

## 10. Verification log

Everything asserted in §1 was checked against the filesystem during planning:

| Claim | Checked against |
|---|---|
| `[ -s "$f" ]` at line 175, `rows` at 176, wedge message at 188 | `bench.sh` read in full |
| `mysql --batch` emits nothing for 0 rows | `bench/SF500XCOLD/q11.r{0,1}.out` are 0 bytes with rc=0; corroborated by `run-abc.sh:47-49` |
| q11 oracle is header-only at SF100/300/500 | `wc -l` on all four `oracle*/q11.tsv` → 1 each |
| q11 recorded as wedge in 18 CSVs | `grep -rl '^q11,.*,wedge,'` over `bench/*/results/*.csv` |
| q11 passes with 1048 rows at SF1 | `results/sf1-2026-08-07-{A,B}.csv` |
| FE `query_timeout` default 300 | `SessionVariable.java:1316-1317` |
| `SET GLOBAL` persists via edit log | `VariableMgr.java:416-417` |
| q05 300104 ms `refused` was ERROR 5024 | `bench/SF500X/q05.r1.out` verbatim |
| the same q05 finished in 20423 ms | `results/sf500xcold.csv` |
| other scripts already set it | `study1-run.sh:86-94`, `study3-cost.sh:92-102`, `restart-sf500x.sh:14-19` |
| `run-abc.sh` does **not** set it | `grep -ni query_timeout run-abc.sh` → only a comment at `:272` |
| `run-abc.sh` has **no** correctness gate | `grep -n 'oracle\|compare' run-abc.sh` → no hits |
| `SIRIUS_LOG_BACKEND` throw is `db`-guarded | `sirius_context.cpp:1573-1577`; CN passes `nullptr` at `sirius_ffi.cpp:177` |
| `SHOW COMPUTE NODES` has a `Version` column | `ComputeNodeProcDir.java:49-54` |
| `current_version()` exists | `FunctionSet.java:295` |
| `sleep(INT)` exists | `functions.py:840`, `FunctionSet.java:301` |
| oracle venv is pip duckdb 1.5.5 | `/opt/dlami/nvme/tpch/venv/bin/python -c "import duckdb; print(duckdb.__version__)"` |
| tracked vs scratch `compare.py`/`oracle.py` identical | `diff` → no output |
| no oracle dir records its dataset | `ls` on all four → only `q*.tsv` |
| all CSV consumers use `DictReader` | `analyze.py:41`, `regress.py:22` |
| `experimental/` is excluded from pre-commit | `.pre-commit-config.yaml:18` |

**UNVERIFIED, with the command that would settle each:**

| Claim | Command |
|---|---|
| pixi `mysql` accepts `--init-command` | `mysql --help \| grep -i init-command` |
| session-scoped `SET query_timeout` bounds a StarRocks query | `mysql --init-command="SET query_timeout=5" -e "SELECT sleep(30);"` → expect ERROR 5024 naming 5 s |
| `SELECT current_version()` output format | `mysql -N -e 'SELECT current_version();'` |
| the Sirius CN supports `sleep()` | run B2 and read the `.out` |
| q11 at SF500 under the spec fraction returns ~5×10⁵ rows | linear extrapolation from the 92,698 rows at SF100 (`run-abc.sh:52`); measure with `oracle.py` against a `__TPCH_Q11_FRACTION__`-substituted query |
| q11's SF500 cost under the spec fraction | never run; probe before adopting §5.3 |
