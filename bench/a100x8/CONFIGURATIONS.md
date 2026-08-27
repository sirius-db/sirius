# Per-arm configurations — Study 1 scale-out, SF500 on 8× A100-SXM4-80GB

Measured on this box, 2026-08-12. Every number below came from
[`../../experimental/starrocks/benchmarks/nixl-nvlink/arena-sweep.sh`](../../experimental/starrocks/benchmarks/nixl-nvlink/arena-sweep.sh)
run against `/home/ubuntu/tpch_parquet_sf500`. Nothing here is inherited from another box.

**Use these, not one global setting.** A single `SIRIUS_EXCHANGE_STAGING_BYTES` cannot serve
every arm of a scale-out study — the requirement is a function of the CN count, and it moves
in the *opposite* direction to intuition.

---

## The configurations

| Arm | `NUM_CNS` | `STAGING` | `GPU_MEM` | `HOST_MEM` | q03 | q07 | q17 | Verdict |
|---|---|---|---|---|---|---|---|---|
| **8-GPU** | 8 | **12GiB** | **66GiB** | 100GiB | ✅ 7.1s | ✅ 6.7s | ✅ 8.8s | **use this** |
| **4-GPU** | 4 | **24GiB** | **54GiB** | 200GiB | ✅ 12.4s | ✅ 10.1s | ✅ 18.4s | **use this** |
| **2-GPU** | 2 | 32GiB† | 46GiB† | 400GiB | — | — | — | **no config passes all three** |
| 1-GPU | 1 | 8GiB | 70GiB | 400GiB | ✅ 16.5s | ✅ 21.3s (n=1) | — | partial arm, **q01 refused** (rpc backend death); not measured by arena-sweep.sh |

† Tested and **rejected**: at 32/46 the pool starves — q01 refused and q04 took 211 s cold
against 6.5 s under the uniform config. See [2-CN did not work at any split tried](#2-cn-did-not-work-at-any-split-tried).

Launch:

```bash
cd experimental/starrocks
NUM_CNS=8 GPU_MEM=66GiB STAGING=12GiB HOST_MEM=100GiB ./benchmarks/nixl-nvlink/script-box.sh
NUM_CNS=4 GPU_MEM=54GiB STAGING=24GiB HOST_MEM=200GiB ./benchmarks/nixl-nvlink/script-box.sh
```

`script-box.sh` exports `SIRIUS_EXCHANGE_STAGING_BYTES` from `STAGING`, so setting `STAGING`
is sufficient — do not set both to different values.

---

## Why the arena grows as you *remove* GPUs

The exchange staging arena is a bare `cudaMalloc` region that sits **outside** the RMM pool.
Send-side packed batches are gathered into it and receive-side transfers land in it, so the
transport can register one fixed region instead of arbitrary engine buffers.

A query's total exchange volume `D` is a property of the query and the scale factor, not of
the cluster. Spread over `N` compute nodes, each node handles `D/N`. **Halving the node count
therefore doubles the per-node arena requirement.** Fewer GPUs is the harder case, not the
easier one.

Failing-lease sizes are consistent with per-lease doubling. **Each is a single failed
allocation, so three digits is spurious precision** — and one ratio contradicts the pattern:

| Query | 8-CN | 4-CN | 2-CN | 4→2 ratio | 8→4 ratio |
|---|---|---|---|---|---|
| q17 | 256.5 MiB (@8GiB arena) | 497.0 MiB | 994.4 MiB | **2.00** | **1.94** |
| q21 | 248.3 MiB | 311.1 MiB | 621.9 MiB | **2.00** | **1.25** ⚠ |
| q03 | (passed) | (passed) | 459.9 MiB | — | — |
| q07 | (passed) | (passed) | 2023.1 MiB | — | — |

The measured minima follow the same doubling — but **the search grids were `8,12,16` and
`16,24,32`, the second built as 2× the first, so a 2× answer was baked in.** The brackets
only bound min(4CN)/min(8CN) within (1.33, 3.0):

| Arm | fails at | passes at | minimum lies in |
|---|---|---|---|
| 8-CN | 8 GiB | 12 GiB | **(8, 12]** |
| 4-CN | 16 GiB | 24 GiB | **(16, 24]** — 2× the 8-CN band |
| 2-CN | 16 GiB | 48 GiB (q03 only) | **(16, 48]** — 2× again |

### Sizing formula

```
staging(N)  ≈  96 GiB / N        (SF500; round up to the next 4 GiB)
```

This reproduces 12 GiB at N=8 and 24 GiB at N=4 — but it was **fitted to exactly those two
points (zero degrees of freedom)**, and its N=2 prediction (48 GiB) is precisely the config
that fails q07 and q17. Treat it as a rule of thumb, not a law.

The stated mechanism does not even imply the rule: the arena holds the **sum of concurrent
leases**, so per-lease 1/N growth gives 1/N total demand only if the lease count is
N-independent. It is not — q21 held **82 / 50 / 15** leases at 8 / 4 / 2 CN. Scale with the
scale factor as a first guess, then measure.

### The arena is not free

It is carved out of the same 80 GiB card as the RMM pool:

```
GPU_MEM  =  80 GiB  −  STAGING  −  ~2 GiB   (CUDA context + cudf overhead + fragmentation)
```

Buy the **minimum that passes**, not the maximum that fits — every GiB of arena is a GiB of
pool. (At 4-CN, raising 24 → 32 GiB moved q17 18.4 → 19.6 s and q03 12.4 → 8.4 s: n=1 each,
in opposite directions, in a system with within-cell spreads to 3.2×. No effect is resolvable
here; the argument for the minimum is the pool it buys back, not a measured slowdown.)

---

## 2-CN did not work at any split tried

Arena and pool demand both rise as the node count falls, and on a fixed 80 GiB card they
compete for the same bytes. **Four splits were tested; none runs q03, q07 and q17 together.**

| Arena | Pool | q03 | q07 | q17 |
|---|---|---|---|---|
| 16 GiB | 62 GiB | ❌ arena exhausted (459.9 MiB lease) | ❌ arena (2023.1 MiB) | ❌ arena (994.4 MiB) |
| 48 GiB | 30 GiB | ✅ **37.8 s** | ❌ backend died after 131.9 s | ❌ wedged at 420 s |
| 64 GiB | 14 GiB | ❌ backend died after 123.5 s | ❌ refused after 257.9 s (`std::bad_alloc`) | (not run) |
| 32 GiB | 46 GiB | ✅ 48.0 s | (not reached) | (not reached) — q01 refused, q04 cold 211 s then refused |

Read the q03 row: it **fails, passes, then fails again**. Below ~48 GiB the arena starves it;
above ~48 GiB the pool does. The window that satisfies q03 satisfies neither q07 nor q17.

**This is inference, not measurement.** Splits between 16 and 48 GiB were never tried — seven
untried points at 4 GiB granularity, and the pass/fail boundary lies inside that gap. The
honest claim is "no split tried works", not "no split works".

`rpc failed … exec rpc error, backend [id=…]` is the pool-starvation signature — the CN
process died. The CN logs show the mechanism directly: during the 48/30 config both nodes
stream `reschedule (retry 1/100) … OOM at operator GPU_SCAN`, so **q17 was livelocked in the
retry loop, not hung**, and the 64/14 q07 refusal ends with `task 814 … exceeded 100 retries
… terminating query`. The arena signature, by contrast, names itself:

```
exchange staging arena exhausted: requested 260369088 bytes (260369152 aligned),
242867200 free of 17179869184 capacity with 82 leases outstanding
(raise SIRIUS_EXCHANGE_STAGING_BYTES)
```

Note `82 leases outstanding`: the arena must hold the **sum of concurrent leases**, not the
largest one. Observed counts at refusal run 9–82 with the arena 97.7–99.4% full, which makes
the requirement **tens of times** any single lease — not hundreds.

**Consequence for Study 1.** A 2-GPU point measured with a per-arm-tuned config is not
available: the best candidate (32/46) starves the pool badly enough to be pathological — q04
took **211 s cold** there against 6.5 s under the uniform 16/62 config. The published 2→4→8
curve therefore uses the **uniform** campaign, where all three arms share one config and are at
least comparable to each other, at the cost of dropping q03/q07/q17 from the 2-GPU point.
Do not splice a 2-GPU point from one campaign into a 4/8 pair from another: repeat measurement
of identical cells differs by **2.0–2.2×**, which is larger than most effects being reported.

---

## Sizing vs lifecycle: how to tell which failure you have

Two failures look identical in the sweep log and have opposite remedies.

**Sizing** — a larger arena fixes it, and within a fixed CN count the query is monotone in
arena size. **q17** is the clean example and it sets the minimum at every arm: at 8 CN it
fails at 8 GiB — *with the largest pool, 70 GiB* — and passes at 12 and 16; at 4 CN it fails
at 16 and passes at 24 and 32. Its ratios (4→2 = 2.00, 8→4 = 1.94) sit on the doubling.

**Lifecycle** — a floor sits under the requirement that no arena size clears. **q21** is the
example: it fails at *every* CN count including the one where its lease is smallest, and it
held **82 concurrent leases** with the arena 98.6% full at refusal. Its 8→4 ratio (1.25) is
the one value that breaks the doubling pattern the other three follow. More arena will not fix
q21 — the lease-release path needs fixing.

> The strongest available evidence is the **occupancy at refusal** (97.7–99.4% full, 9–82
> leases), not the ratio anomaly. Note also that q21 was never retested with a larger arena,
> so "no arena size clears it" rests on the failure pattern rather than on a direct test.

This independently reproduces the lease-lifecycle diagnosis from the 4×GB200 SF500 run, via a
different signature (ratio anomaly rather than lease-count growth).

---

## Defects that no configuration fixes

| Query | Behaviour | Cause | Fix |
|---|---|---|---|
| **q21** | refused at 8/4/2 CN | lease-lifecycle bug (see above) | engine — release leases |
| **q11** | returns **empty in 0.9–14.9 s** at every arm | `FRACTION` is the SF1 literal `0.0001`; SF500 needs `0.0000002`, so the threshold is 500× too high and nothing qualifies | one-line query fix |
| **q15** | non-deterministic **empty result** (4.3 s / 20.9 s — it does not hang) | FP64 decimal lowering makes a float-equality predicate unstable | `translate_arithmetic` |
| q01 q03 q07 q14 q19 | complete, **values wrong** | same FP64 decimal lowering | `translate_arithmetic` |

The last row is why those five carry ⚠ in the query set: they are timing-valid and
correctness-invalid. Never put them in a correctness claim.

---

## Reproducing

```bash
cd experimental/starrocks
# bracket the minimum arena for an arm (records the FULL diagnostic the sweep truncates)
OUT_CSV=/tmp/arena.csv ./benchmarks/nixl-nvlink/arena-sweep.sh 8 8,12,16 q03 q07 q17
OUT_CSV=/tmp/arena.csv ./benchmarks/nixl-nvlink/arena-sweep.sh 4 16,24,32 q03 q07 q17
```

The driver wipes `starrocks/output/fe/meta` between configs. This is **load-bearing**: every
config reuses the same ports, so the previous config's registrations survive a restart and
still read as `Alive` until a heartbeat timeout longer than bring-up. A readiness check that
counts `Alive` rows would return the stale count, the sweep would query a still-booting
cluster, and the resulting `No alive backend …` / `Unable to validate object` errors would be
misread as arena failures. Wiping meta is safe — engine A creates no persistent tables; every
query reads parquet through `FILES()`.
