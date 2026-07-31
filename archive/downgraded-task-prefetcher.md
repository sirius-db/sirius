---
name: downgraded-task-prefetcher
description: "HOST/DISK->GPU prefetcher for queued tasks' downgraded inputs: branch, design, why it's perf-neutral at SF1K, bounce/wedge lesson"
metadata: 
  node_type: memory
  type: project
  originSessionId: 515e9640-e993-433a-b0d1-15cbf9949e23
  modified: 2026-07-22T16:43:31.168Z
---

Branch `claude/downgraded-tasks-prefetcher-919a40` (worktree of same name, 2026-07-22,
uncommitted): `src/pipeline/downgraded_task_prefetcher.{hpp,cpp}` + `inspectable_mpsc::
for_each_mutable` (early-stop visitor) + wiring in `task_scheduler` (env `SIRIUS_TASK_PREFETCH=1`,
`_THREADS`, `_MIN_FREE_FRACTION`, `_LOOKAHEAD` default 4, `_QUIET_MS` default 250).
Walks the task queue front-to-back (= dispatch order: the matcher pops with
`pop_if(front_to_back=true)`; downgrade Tier-2 evicts from the back) and upgrades queued
tasks' HOST/DISK inputs via `convertible_data_batch::convert` (non-blocking, reservation-backed).
6-case Catch2 test `test/cpp/pipeline/test_downgraded_task_prefetcher.cpp`.

**Key findings (2026-07-22, GB300 SF1000 host-pinned, 471GB-host/0.8-0.6 config):**
- **q9/q21 do NOT downgrade at all on current dev at the normal 0.95 GPU budget** (zero
  `[downgrade]` lines at debug level) — #1089 decode accounting + dynamic filters killed the
  pressure; confirms [[sirius-tune-gb300-campaign]] engine-79a3839f observation. The prefetcher
  is a verified no-op there (0 bytes).
- **Unbounded version wedged the engine** under a constrained budget (usage 0.4 = 102GB):
  filled GPU with 58GB of queued-task inputs, downgrade evicted 10 batches and the prefetcher
  re-upgraded them within 5ms (bounce), then a hard hang at the budget limit (all threads
  futex-parked; no ptrace on box — yama scope 1, no sudo). An instantaneous
  headroom/should_downgrade gate is NOT enough.
- **Hardened version** (250ms pressure-quiet hysteresis after any pressure observation +
  lookahead cap of 4 tasks + available() capped at max_memory since usage limit > reservation
  limit makes available exceed max): safe — no hang, byte-identical results, downgrade count
  unchanged (16/16) — but **perf-neutral** (q9 -0.0%, q21 +0.2% hot): the calm windows it may
  act in don't overlap the downgrade traffic it would need to hide.
- **Why:** the scan prefetcher wins because pinned splits sit ready from query start with a huge
  idle-copy-engine window ([[scan-prefetch-overlap-design]]); downgraded task inputs only exist
  *during* pressure, exactly when upgrading feeds the eviction path. Worth revisiting only for
  genuinely oversized workloads (disk-spill readback phases, where the serialized
  pipeline_io_backend makes latency-hiding valuable — see [[downgrade-bounce-fix]] residual wall).

**Gotchas hit:** (1) `~/.sirius/sirius.yaml` on the box still has removed key
`default_scan_task_varchar_size` → every fresh dev extension fails to LOAD without
SIRIUS_CONFIG_FILE pointing elsewhere. (2) performance_test.py `split_sirius_log` crashes on
logs ending mid-UTF-8-char — patched with errors="replace" in the scan-prefetcher-perf worktree,
needs upstreaming. (3) A killed `pixi run make` leaves stale objects → later "successful" link
with undefined symbols (`sirius_dynamic_zone_map_filter` ctor); `make clean` fixed it.

**2026-07-31 real-downgrade proving ground (combo build: dev c4e8a10b + #1181/#1349/#177 + GPU-compressed pins):** q9 GENUINELY downgrades here — 41.8 GB (18×~5GB CONCAT(14)/CONCAT(17) intermediates) GPU→HOST at ~+479ms, re-uploaded ~+925ms, every iteration (quent data_batch InTransit; INFO logs show NOTHING — downgrade activity is debug-level, never grep info logs for it). Port note: dev replaced inspectable_mpsc with exec::multi_index_priority_queue — added for_each_mutable(visitor, front_to_back) walking the priority spine (combo worktree combo-prefetch-comp). VERDICT after full aggressiveness sweep: (1) DESIGN FLAW — default min_free_fraction 0.4 deadlocks against downgrade_stop_fraction 0.6 (eviction stops at exactly 40% free → prefetcher sees floor, upgrades 0 bytes forever); (2) tuned arms staged 2→53 GB total (floor 0.15-0.10, lookahead 16-32, quiet 50ms, 4 threads; saturates ~53GB) and q9 stayed FLAT 1.86-1.87s at every point — the re-upload is already fully overlapped by 8 pipeline threads at task-prepare. (3) What actually recovered the cost: downgrade_trigger 0.8→0.9 (no eviction at all) = q9 1.872→1.813 (-3.2%), but suite-neutral (+0.2%, q4/q6/q21 offset it) — a per-workload knob, not a default. Fourth confirmation of the campaign law: prepare-time uploads don't stall; only removing bytes (or removing the eviction) moves runtime.

**Eviction-ordering experiment (2026-07-31):** user's design idea "prefetch front-of-queue, evict back/last-consumed" — Tier-2 (task queue) ALREADY evicts back-to-front (front_to_back=false in downgrade_executor.cpp Tier 2); Tier-1 (repos, the q9 evictees) walks ASCENDING operator id = earliest-consumed first. Flipped Tier-1 to descending and measured q9 at default 0.8/0.6: **1.871 vs 1.872 = neutral**. Refined root cause: q9's ~60ms downgrade cost is the eviction WORK itself (42 GB D2H + batch locks during execution), order-independent; the re-upload side was already proven overlapped. Victim choice and recovery prefetch both can't help — only not evicting (trigger 0.9/stop 0.85) recovers the time. Flip reverted (kept out of combo branch).

**Concurrent prefetch-during-downgrade (2026-07-31, final experiment):** added `SIRIUS_TASK_PREFETCH_DURING_PRESSURE=1` (skips pressure gate, keeps floor as OOM guard) + `SIRIUS_EVICT_REPOS_DESC=1` (order-disjoint victims) on combo branch. q9 @ default 0.8/0.6: concurrent+orig 1.880, concurrent+desc 1.881 vs baseline 1.872, ceiling 1.817 — SAFE (no wedge, ~50GB staged through episodes; disjointness prevents the bounce) but slightly NEGATIVE. Mechanism cornered by 4 experiments: the eviction cost is the D2H work + locks contending with execution; concurrent prefetch ADDS transfer contention during the episode. Only not-evicting returns the time. The env knobs remain on branch claude/combo-prefetch-compression for future spill-heavy regimes.
