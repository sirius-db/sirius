# Query set — 17 of 22, headline over 8

**Hardware-independent.** This file applies to any box; it is a property of Engine A's current
state, not of the machine. Shared by all three studies.

---

## The set

```
q01 q02 q03 q04 q06 q07 q11 q12 q14 q15 q16 q19 q20 q22 q13 q17 q21
```

Order matters — `run-abc.sh` preserves the order you pass. Clean queries run first so that a
restart triggered by a late failure cannot contaminate the numbers that matter.

## Tiers

| Tier | Queries | Count | Status | Use |
|---|---|---|---|---|
| **1 — anchor** | `q04 q06` | 2 | Value-verified at SF500 on Engine A | Safe for any claim today |
| **2 — expansion** | `q02 q11 q12 q16 q20 q22` | 6 | Complete at SF100, values **byte-identical** to the DuckDB oracle; none touches `(1 − l_discount)` | **HEADLINE AGGREGATE** (Tier 1 + 2 = **8 queries**) |
| **3 — timing-only** | `q01 q03 q14 q15 q19` | 5 | Complete, **values WRONG** — decimal defect | Timings + defect magnitude only. **Never** in a correctness claim or an aggregate |
| **4 — probe** | `q13 q17 q21 q07` | 4 | Each has a named, quantified failure projection at scale. `q07` is **also** numerically wrong | Run once, report outcome, exclude from aggregates |
| **5 — excluded** | `q05 q08 q09 q10 q18` | 5 | Measured hard failures | **Do not run in the measurement sweep** |

> **q13 / q17 caveat.** Both are value-correct at SF100, so on correctness alone they belong in
> Tier 2. They sit in Tier 4 purely on **memory risk at scale**. If they survive the first SF500
> run, promote them and the headline grows from 8 to 10.

## Why the five are excluded

Engine A completes **17 of 22** at SF100 — and beat stock StarRocks on **every one of them**.

> `ENGINE-CONFIGS-AND-EQUIVALENCE.md:290` claims 18/22. It is **wrong**: q18's 4th run wedged.
> Verified by re-reading all 88 Engine A rows of `tpch-sf100-abc/results.csv`.

| query | Behaviour | Mechanism |
|---|---|---|
| q05 | 1.8 s → **wedge 180 s** | Task enters `Computing` and never leaves (undiagnosed) |
| q08 | **refused ~51 s**, ×2 | `HASH_JOIN(4)` requests **142.2 GiB from a 140 GiB pool** off a **12.5 MB** input — 101 identical retries. A sizing bug |
| q09 | **refused 64 s / 131 s** | **1.13 TiB single request**; cross-join, O(SF²) intermediates; staging-arena leases leaked |
| q10 | **refused 121 s** → wedge | Same freeze family as q05 |
| q18 | 1.1 → 1.0 → **61.5 s** → wedge | Monotonic degradation = **a leak** |

**None is a query-shape problem.** The dominant pattern is *state accumulating across runs on the
same cluster* — q18's 1.0 s → 61.5 s → wedge curve is monotonic, not bimodal. A query that is too
hard fails on run 1, not run 3. **More GPUs cannot fix any of these**, and every mechanism worsens
with scale.

**Cost of running them anyway:** at SF500 the harness derives warm=900 s / cold=3000 s. Five
wedging queries is **~5.5 hours** of wall clock re-confirming known failures — and with
`SIRIUS_QUERY_WATCHDOG_SECS=0` a wedged handler blocks the FE↔CN connection and **poisons the
queries after it**. Run them in a separate pass with a fresh cluster per query, or not at all.

---

## The decimal defect — not fixed, and it is one function

Verified at HEAD `4e6439c8`:

`experimental/starrocks/crates/starrocks-plan-translator/src/expr_translator.rs:459-481`
(`translate_arithmetic`) casts **both operands of every decimal `+ − * / %` to FP64 and declares
FP64 output**. So `l_extendedprice * (1 − l_discount)` is already FP64 *before `SUM` ever sees it*.

- `git diff --stat 1d2bbae2..HEAD -- experimental/starrocks/crates/` is **empty** — untouched since the audit.
- `OPEN-ISSUES.md` **#24** is 🔴 OPEN and warns that fixing the SUM/AVG lowering at `:826-833`
  — *what the doc tells you to do* — **would change nothing**. Start at `translate_arithmetic`.
- Post-audit reruns (2026-08-10, `cn2-vs-cn4/`, `nfs-a-vs-c/`) reproduce the wrong values
  bit-identically across two CN topologies and two storage backends.

**Blast radius is 7 queries: `q01 q03 q05 q07 q14 q15 q19`.** q05 is a seventh victim the audit
missed — its cold run passed with 5 correctly-ordered rows, all low by ~0.096%; the audit diffed
only `r1` files and `q05.r1.out` is 0 bytes from the warm wedge.

> **Highest-leverage fix available.** One function, localized to the CN path (standalone Sirius is
> exact on the same files). Fixing it takes the headline set from **8 queries to 14**.

> ⚠️ **Two traps that look like fixes — cite neither.**
> `engineA-fixed-q01-q04-q06-q14.png` is dated **2026-08-08, the day *before* the audit**;
> "fixed"/"httpfix" refer to the CN `http_port` advertisement fix (`1d2bbae2`) — an *availability*
> fix. Its subtitle claims *"results bit-identical to the DuckDB oracle"*, which is **false** for
> the q01 and q14 bars it draws. And `REVIEW-benchmark-findings.md:187` localizes the defect to the
> CN wrapper — a diagnosis, not a repair.

---

## Mandatory: the harness does not check values

`run-abc.sh` defines `status=pass` as **exit code 0 and at least one row**. It **never compares
values**. Every result must be diffed against a pure-CPU DuckDB oracle (`SET gpu_execution = false`)
on the same files.

**Use a relative tolerance of `1e-12` — never string equality.**

> q06 legitimately returns three adjacent doubles across runs
> (`61662234676.307495` / `.3075` / `.30751` — exactly ±1 ULP). An exact comparator flags it as a
> failure about a third of the time. Without a relative-tolerance oracle diff this sweep produces
> another table of liveness checks and we learn nothing about correctness.

## q11's `FRACTION`

Must be `0.0001/SF` — `0.0000002` at SF500, `0.0000001` at SF1000. The literal-vs-spec choice
silently changes the answer by the scale factor, and historical CSVs used the wrong one.

---

## Chart labelling

Engines B and C complete **more** queries than Engine A. Every aggregate must be restricted to the
Engine A set, and that restriction named:

> `17 of 22 queries · q05/q08/q09/q10/q18 excluded (Engine A failures) · aggregates over the 8-query Tier 1+2 set`

**Never** label a chart "all 22 queries complete".
