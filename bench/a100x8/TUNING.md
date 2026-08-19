# Tuning the Sirius CN config — by GPU count and by scale factor

How the numbers in [`CONFIGURATIONS.md`](CONFIGURATIONS.md) were arrived at on 8× A100-SXM4-80GB,
and the method to redo it on other hardware or other scale factors. Everything here is measured;
where a rule is fitted rather than measured, it says so.

Companion docs: [`CONFIGURATIONS.md`](CONFIGURATIONS.md) (the resulting settings) ·
[`SCALE-OUT-SUMMARY.md`](SCALE-OUT-SUMMARY.md) (what the settings produced) ·
[`../common/RETARGETING.md`](../common/RETARGETING.md) (moving the whole folder to a new box) ·
[`../../experimental/starrocks/docs/TUNABLES.md`](../../experimental/starrocks/docs/TUNABLES.md)
(CN environment variables).

> The CN now logs the arena size it got (`exchange staging arena: N bytes`) and, at teardown,
> the high-water mark a workload actually reached
> (`exchange staging arena: high water N of M bytes`). That second line is the measurement §2's
> grid search had to infer from pass/fail — read it before running another sweep.

---

## 1. What you are actually tuning

Three numbers, one of which is not where you would look for it.

| Knob | Set by | Lands in the derived config as |
|---|---|---|
| RMM pool | `--gpu-memory-limit` | `memory.gpu.usage_limit_bytes` |
| Host arena | `--host-memory-limit` | `memory.host.capacity_bytes` |
| **Exchange staging arena** | **`SIRIUS_EXCHANGE_STAGING_BYTES`** (env) | **nowhere — it is not in the YAML** |

The CN writes `<engine-dir>/derived-sirius-config.yaml` from the CLI flags at every start
(`src/main.rs`, `EngineConfig::resolve`). A 4-CN SF1000 node looks like:

```yaml
sirius:
  topology:
    num_gpus: 1                       # always 1 — one CN owns one GPU
  memory:
    gpu:
      usage_limit_bytes: "54GiB"      # the RMM pool
      reservation_limit_fraction: 1.0
    host:
      capacity_bytes: "100GiB"
  telemetry:
    output_directory: ".cn3/telemetry"
```

**Read that file after every config change.** It is the ground truth for what the engine got;
the launcher's echo is only what you asked for. Note also that `--sirius-config` and the memory
flags are mutually exclusive (clap rejects the combination), so the two config sources never
race — but it also means a hand-written YAML silently ignores your `--gpu-memory-limit`.

### The one thing that makes this non-obvious

**The staging arena is a bare `cudaMalloc` OUTSIDE the RMM pool.** It does not appear in the
YAML, `usage_limit_bytes` knows nothing about it, and it is reserved up front. So the real
occupancy of a card is:

```
device  =  usage_limit_bytes  +  SIRIUS_EXCHANGE_STAGING_BYTES  +  ~2 GiB (CUDA context, cudf, fragmentation)
```

which on an 80 GiB A100 gives the budget line every config below obeys:

```
GPU_MEM  =  80 GiB  −  arena  −  2 GiB
```

**Every GiB of arena costs a GiB of pool.** That trade is the whole tuning problem: the two
compete for one card, and — see §3 — their requirements move in the same direction.

---

## 2. Tuning by GPU count (fixed scale factor, SF500)

### Measured

| CNs | Arena | Pool | q03 | q07 | q17 | Verdict |
|---|---|---|---|---|---|---|
| 8 | 8 GiB | 70 GiB | ✅ | ✅ | ❌ arena | too small |
| **8** | **12 GiB** | **66 GiB** | ✅ 7.1s | ✅ 6.7s | ✅ 8.8s | **use** |
| 8 | 16 GiB | 62 GiB | ✅ | ✅ | ✅ | works, wastes 4 GiB of pool |
| 4 | 16 GiB | 62 GiB | ✅ | ✅ | ❌ arena | too small |
| **4** | **24 GiB** | **54 GiB** | ✅ 12.4s | ✅ 10.1s | ✅ 18.4s | **use** |
| 4 | 32 GiB | 46 GiB | ✅ | ✅ | ✅ | works, wastes 8 GiB of pool |
| 2 | 16 GiB | 62 GiB | ❌ arena | ❌ arena | ❌ arena | arena starved |
| 2 | 32 GiB | 46 GiB | ✅ 48s | ❌ rpc | ❌ rpc | pool starved |
| 2 | 48 GiB | 30 GiB | ✅ 37.8s | ❌ rpc | ❌ livelock | pool starved |
| 2 | 64 GiB | 14 GiB | ❌ rpc | ❌ bad_alloc | — | pool starved |

### The counter-intuitive part

**Fewer GPUs need a BIGGER arena.** A query's exchange volume is a property of the query and
the scale factor, not of the cluster. Spread over `N` nodes each carries `1/N` of it, so halving
the node count doubles the per-node requirement. Removing GPUs is the harder case.

That is why the 2-CN column is not a tuning failure but a wall: at `N=2` the arena wants ~48 GiB
*and* each node holds half the working set, and 80 GiB does not cover both. Read the q03 column
top to bottom — **refused → passes → refused**. Below ~48 GiB the arena starves it, above it the
pool does, and the window that satisfies q03 satisfies neither q07 nor q17.

### The rule, and how much to trust it

```
arena(N)  ≈  96 GiB / N        at SF500, rounded up to the next 4 GiB
```

Reproduces 12 GiB at N=8 and 24 GiB at N=4. **It was fitted to exactly those two points, so it
has zero degrees of freedom**, and its N=2 prediction (48 GiB) is precisely a config that fails.
Treat it as a starting guess, then measure.

Two reasons not to promote it to a law:

- **The search grid was circular.** The two grids were `8,12,16` and `16,24,32` — the second
  built as 2× the first, so a 2× answer was baked in. The brackets only bound
  `min(4CN)/min(8CN)` within (1.33, 3.0).
- **The mechanism does not imply it.** The arena must hold the *sum of concurrent leases*, so
  per-lease `1/N` growth only gives `1/N` total demand if the lease count is N-independent.
  It is not: q21 held **82 / 50 / 15** leases at 8 / 4 / 2 CNs.

---

## 3. Tuning by scale factor (fixed at 8 GPUs)

| Scale | Data | Arena | Pool | Host/CN | Pass | Matched-set time |
|---|---|---|---|---|---|---|
| SF500 | 132 GB | 12 GiB | 66 GiB | 100 GiB | 12/15 | 71.4 s |
| SF1000 | 265 GB | **24 GiB** | 54 GiB | 100 GiB | 11/15 | 123.9 s |

Exchange volume tracks the data, so the arena tracks the scale factor:

```
arena(N, SF)  ≈  96 GiB × (SF / 500) / N
```

The SF1000 arena was a **scaled guess, not a measured minimum** — 2× the SF500 value because the
data is 2×. It worked (no `arena exhausted` at SF1000), so the true minimum may be lower and
some of that 24 GiB may be buying nothing. Bracketing it downward is unfinished work.

**Host memory does not scale the same way.** It stayed at 100 GiB/CN for both, sized so the page
cache keeps holding the dataset: 8 × 100 GiB = 800 GiB of 1771 GiB leaves ~597 GiB of cache,
which is 2.3× the SF1000 dataset. The parquet is re-read every query, so that cache is
load-bearing — size host memory from `RAM / CN_count` leaving ≥1.5× the dataset, not from the
scale factor.

---

## 4. The procedure

1. **Write the budget line first.** `GPU_MEM = card − arena − 2 GiB`. Every candidate config is
   a point on it; you are choosing where to sit, not choosing two free numbers.
2. **Pick the probe queries.** Not the whole suite — the arena-sensitive ones. Here: **q03, q07,
   q17**, plus q21 as a known-unfixable control. q17 sets the minimum at every CN count.
3. **Bracket by escalation, one variable at a time.** Start below where you expect to fail and
   double: `8,12,16` at N=8. Use
   [`arena-sweep.sh`](../../experimental/starrocks/benchmarks/nixl-nvlink/arena-sweep.sh),
   which records the *untruncated* diagnostic — `bench.sh` cuts errors at 160 characters,
   exactly where the deciding numbers start.
4. **Classify the failure before changing anything** (§5). Arena, pool and lifecycle failures
   look similar in the sweep log and have opposite remedies.
5. **Take the minimum that passes, not the maximum that fits.** Surplus arena is pool you no
   longer have, and at low CN counts the pool is what fails first.
6. **Re-read the derived YAML** to confirm the engine got what you meant.

### Grid choice

Avoid making each grid a multiple of the previous one — that is how the circularity in §2 got
in. Use an independent grid per arm (e.g. `10,14,18` at one N and `18,22,26,30` at another) if
you want the ratio between arms to be a measurement rather than an assumption.

---

## 5. Failure signatures — read the error, not the label

The single highest-value table here. Five failures look alike and mean different things.

| Signature | Means | Fix |
|---|---|---|
| `exchange staging arena exhausted: requested N bytes, M free of C capacity with L leases outstanding` | Arena too small | Raise arena, lower pool |
| `rpc failed … exec rpc error, backend [id=…]` | **CN process died** — pool too small | Lower arena, raise pool |
| `reschedule (retry 1/100) … OOM at operator GPU_SCAN` (CN log) | Pool starvation **livelock**, not a hang | Raise pool |
| `timed out after 600s waiting for fragment instance` | Exchange stall | Neither — investigate |
| `ERROR 5024 … timeout of 300 seconds … 'query_timeout'` | **FE config**, not the engine | `SET GLOBAL query_timeout` |

### Two traps that cost me hours

**`WEDGE/TIMEOUT (rc=0, cut at 420s)` is usually not a timeout.** `bench.sh:188` prints the
*configured budget*, not elapsed time, and reaches that branch for any zero-byte result —
including exit code 0. **`rc` is the discriminator: `rc=124` is a real kill, `rc=0` is a fast
empty.** q11 "wedged at 420s" actually returned empty in **0.9 s**.

**`SET GLOBAL query_timeout` does not survive a cluster restart.** It lives in the FE metadata,
and a clean-bootstrap restart wipes that metadata, so the setting is silently lost the first
time a query fails. Re-apply it inside `RESTART_CMD` — otherwise a late query hits the stock
300 s ceiling and its `ERROR 5024` reads exactly like an engine failure.

### Sizing vs lifecycle

A failure that scales with `N` is sizing; one that does not is a bug.

- **q17 — sizing.** Monotone in arena within a fixed CN count: at 8 CNs it fails at 8 GiB *with
  the largest pool (70 GiB)* and passes at 12 and 16. A bigger arena fixes it.
- **q21 — lifecycle.** Fails at *every* CN count including the one where its lease is smallest,
  holding **82 concurrent leases** with the arena 98.6% full. Its 8→4 ratio (1.25) is the one
  value that breaks the doubling the others follow. No arena size fixes it.

---

## 6. Traps when moving between boxes

- **Never copy an arena size from other hardware.** A 16 GiB arena is 8.6% of a GB200 and **20%
  of an 80 GB A100**. The GB200's 32 GiB-at-4-CN result was the right *starting point* here and
  the wrong *value* — it works at 4 CNs but wastes 8 GiB of pool.
- **`usage_limit_fraction` is a fraction of total, not free, device memory** — and it still does
  not know about the arena.
- **More arena is not monotonically better.** q17 passes at 12 GiB and the extra pool that buys
  matters more at low CN counts than the extra arena does.
- **One config cannot serve a scale-out study.** The requirement is a function of CN count, so a
  single setting is mis-sized for most arms. Size per arm — then note that per-arm sizing makes
  the arms differ in more than GPU count, which is a methodological cost of its own
  (see [`SCALE-OUT-SUMMARY.md`](SCALE-OUT-SUMMARY.md)).

---

## 7. Measure the box before trusting any of it

The tuning above is only as good as the measurement, and on this box the measurement drifted:
the same 12 queries at the same 8-GPU config ran **3.55× slower** across three campaigns in one
evening (2.04× then a further 1.74×), monotonically, with no exception. GPUs were unthrottled at
max clock and 85 W of a 400 W cap; memory was unfragmented; no disk accumulation. Cause unknown.

Practical consequences:

- **A config difference under ~2× is not resolvable** on a box behaving like this. Within-cell
  spreads reached 2.5× on n=3.
- **Compare configs back to back**, never across sessions. That is why
  [`study3-cost.sh`](../../experimental/starrocks/benchmarks/nixl-nvlink/study3-cost.sh) runs
  both scale factors in one invocation: the SF1000/SF500 *ratio* survives drift even though the
  absolute numbers do not.
- **Re-measure the baseline** whenever a result surprises you, before believing the surprise.
