# Staging Arena Free List — Results (WORKING DOC)

**Temporary.** Written 2026-08-19 to carry the state of the
[STAGING-ARENA-FREELIST-PLAN.md](STAGING-ARENA-FREELIST-PLAN.md) implementation through review.
Delete or fold into the plan once the changes are committed.

**Status: implemented, tested, UNCOMMITTED and UNPUSHED.** Phases 0 and 1 only; Phase 2
deliberately not implemented (§6).

Box: 2× RTX PRO 6000 Blackwell (97887 MiB), 2 CNs, TPC-H SF100, `/home/ubuntu/tpch_parquet_sf100`.

---

## 1. Verdict in one line

The free list does exactly what the plan promised — it recovers **all** the allocator drift — but
it **fixes no failing query**, because it removed a masking layer and exposed the real causes
underneath. That exposure is the actual value delivered.

---

## 2. Phase 0 — the measurement that justified the change

Instrumented the **old** allocator and reproduced q08's exhaustion twice at `STAGING=16GiB`:

| | Run 1 | Run 2 |
|---|---|---|
| Leases outstanding | 15 | 17 |
| Live bytes | 8.41 GiB | 8.64 GiB |
| **Unreachable drift** | **7.07 GiB** | **7.30 GiB** |
| Drift ratio (head / live) | **1.82×** | 1.85× |

**46 % of the arena was unreachable gaps.** Live (8.41) + the failing request (1.16) = 9.57 GiB,
well inside 16 GiB — so a non-drifting allocator would have served that lease. That is what made
Phase 1 worth doing, and it sits in the plan's "1.2×–2×" gate band.

---

## 3. Phase 1 — what shipped

Address-ordered first-fit free list, forward + backward coalescing on release.
`high_water()` removed (one call site) and replaced by `peak_live_bytes()`, `live_bytes()`,
`total_free()`, `largest_free()`. Free list seeded once after `capacity_` is final on **both**
constructor paths — the fabric path rounds capacity up to the VMM granularity, so seeding
per-branch would describe a region that is not mapped.

Also: `outstanding()` bridged through all four FFI layers (C++ → `sirius_ffi` → `sirius-sys` →
`sirius`), as `Result<usize>` rather than the plan's bare `usize`, because unlike
`base()`/`capacity()` the C++ side takes the arena mutex and is not `noexcept`.

**Files** (8, +398 / −74):

```
src/exec/exchange_staging_arena.cpp          src/include/exec/exchange_staging_arena.hpp
test/cpp/exec/test_exchange_staging_arena.cpp
src/sirius_ffi.cpp                           src/include/sirius_ffi.hpp
rust/crates/sirius-sys/src/lib.rs            rust/crates/sirius/src/lib.rs
experimental/starrocks/src/engine.rs         (stale comment only)
```

### Does it work? Yes — measured at the same 16 GiB arena

| | Bump allocator | Free list |
|---|---|---|
| Live bytes when q08 exhausted | 8.41 GiB | **15.53 GiB** |
| Unreachable drift | 7.07 GiB | **~0** |
| Conservation (`free + live == capacity`) | n/a | holds in production |

At `STAGING=32GiB`: **zero arena exhaustions**. Effective capacity is roughly doubled at a given
`STAGING`, and that memory can go back to `GPU_MEM`.

`ARENA-10` makes the defect executable: the old allocator dies at **cycle 248 with 3.1 % live**;
the free list completes all 10,000 cycles.

---

## 4. Regression results — clean

| Check | Result |
|---|---|
| `[staging_arena]` | **13/13**, 20,291 assertions |
| Full C++ suite (`~[s3]~[rest]`) | 2354 cases, **17 failed — exactly the pre-existing 17** |
| `cn-test-no-engine` | green |
| 19-query TPC-H sweep | **19/19 pass**, total 26144 → 26031 ms (**−0.4 %**) |
| Correctness vs DuckDB oracle | **16/19** — identical to baseline (the 3 are the known q03/q10/q15 drift) |

Attribution of the 17 is settled **by construction, not inference**:
`SIRIUS_EXCHANGE_STAGING_BYTES` is unset in the unit-test environment, so `from_env()` returns
`nullptr` and the arena is **never constructed** there. The failures are `gpu_execution` operator
semantics (nested-loop joins, top-n, casting, group-by, LIMIT, empty filter) plus one
load-sensitive thread-pool test.

Results: `bench/rtxpro6000-2gpu/results/sf100-freelist-40g16g.csv`.

---

## 5. The failing queries, ranked by solvability

This is the payoff: each failure now has exactly one cause.

| # | Query | Real cause | Solvable? |
|---|---|---|---|
| 1 | **q11** | **Not a bug.** Query hardcodes the SF1 threshold `0.0001`; the spec scales it `0.0001/SF`. At SF100 the bar is 801,681,490 vs a largest part value of 23,649,655. DuckDB returns 0 rows too. `bench.sh` misfiles the empty output as a wedge. | **Yes, trivially.** Fix the query text (scale by SF) or stop treating an empty file as failure. Free +1. |
| 2 | **q08** | Head-of-line deadlock: `no parked sender output to export for SenderSlot`. With the arena no longer binding, this is now the *only* fault. | **Best real target.** `QUERY-TIMEOUT-ANALYSIS.md` names commit `05d3c7f4` and proposes reverting `nixl_transport.rs` + `compute_node_service.rs` to `c3bfe660`. Treat as a **lead, not fact** — §8 of that doc contradicts itself with a "suspected, not proven" line. |
| 3 | **q15** | Intermittent 0-row result, ~1 pass in 3. Identical across every arm, so config-independent. | **Maybe.** Real but intermittent; needs a repro loop before it is diagnosable. |
| 4 | **q09** | `OOM at operator HASH_JOIN, requested **2,694,604,000,000 bytes** (2.69 TB)`. | **Not by configuration.** A plan/cardinality estimation bug. No pool or arena can satisfy 2.69 TB — which retroactively explains why a 50 % larger pool changed nothing. Needs engine work; the 2.69 TB figure is a precise starting point (likely overflow or a bad estimate). |

**Recommended order: q11 → q08 → q09 → q15.**

---

## 6. Deviations from the plan (review these)

1. **ARENA-3 in the plan contradicts the plan's own `lease()`.** It asserts the *exhaustion*
   message for `lease(8192)` on a 4096 arena, but the new oversize guard fires first and throws
   "exceeds capacity". Rewritten to cover the two refusal paths as genuinely distinct: a
   permanent sizing error vs. a transient fragmentation one.

2. **Phase 2 (RAII) NOT implemented — it would regress the metric being optimised.**
   `ExecuteRequest` owns `remote_inputs`, but `run_fragment` *borrows* it and releases each lease
   **eagerly right after its push**. Adding `Drop` with by-reference iteration defers every
   release to request-drop, which **raises** peak live bytes. It needs by-value iteration — a
   wider change than the plan's "each becomes a `drop(batch)`".

3. **`outstanding()` bridged as `Result<usize>`**, not the plan's bare `usize` (mutex, not
   `noexcept`).

---

## 7. Acceptance criteria not met

| # | Criterion | State |
|---|---|---|
| A7 | Teardown log reports 0 leases outstanding | **Unverifiable as-is.** The log lives in the destructor, and `pkill`/SIGTERM never runs it. `outstanding()` is now bridged so a leak *can* be observed, but nothing calls it yet. |
| A8 | 5-sweep soak, stable `peak live` | **Not run.** This is the plan's "one that matters" for drift, though the mechanism now makes drift structurally impossible. |
| A1–A6, A9 | | Met (§3, §4). A9: fragmentation is now visible — at 16 GiB q08 requested 190,735,616 with a largest free block of 190,635,520, missing by **100 KB**. |

---

## 8. Next steps

- [ ] Review the diff (§3) — nothing is committed.
- [ ] Decide on Phase 2: by-value iteration, or drop it.
- [ ] Wire `outstanding()` into CN quiesce to close A7.
- [ ] q11 fix (cheapest win), then q08 with the `05d3c7f4` lead.
- [ ] Once trusted, lower `STAGING` in the launchers and give the memory back to `GPU_MEM`
      (plan C7 — `bench/a100x8/TUNING.md` §2).
