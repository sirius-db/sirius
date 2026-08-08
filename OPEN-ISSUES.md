# Open issues — the work queue for the multi-GPU box

State of the branch this doc ships with: **22/22 TPC-H pass at SF1**, unbounded back-to-back
execution (the arena leak is fixed), every result DuckDB-validated. The journey and evidence
live in `QUERY-TIMEOUT-ANALYSIS.md`; the integration learnings in `ROADMAP-8CN-TPCH.md`; the
benchmark protocol in `experimental/starrocks/benchmarks/tpch/REPRODUCE.md`; day-to-day
operations in the `tpch-bench` skill (`.claude/skills/tpch-bench/SKILL.md`). Reference
numbers: `experimental/starrocks/benchmarks/tpch/results/sf1-2026-08-07.md`.

Recommended order on the GB200 box: **M0 → M1 → M2 → #24 → #31**, with M3 as soon as the
SF100/SF1000 datasets land. M0–M2 unblock everything else on the new hardware. The scale
math behind M1–M3 is `TPCH-PLAN-ANALYSIS.md` §8.

---

## M0. aarch64 port (the GB200 box is Grace — the x86 prebuilts won't load)

**What**: the target node's Grace CPUs are aarch64; the demo's transport stack ships
x86-64 prebuilts. Verified `ELF 64-bit ... x86-64` with `file`: `tools/ucx-install/lib/*.so`
and `tools/nvda_nixl/lib/x86_64-linux-gnu/*` (libnixl.so, plugins) in the tree beside the
worktrees. The CN process cannot start until these are rebuilt.

**Do**:
1. Rebuild UCX 1.21 from the vendored `tools/ucx-1.21.0.tar.gz` / `tools/ucx-src`
   (`./configure --with-cuda=$CUDA_HOME --enable-mt`); UCX 1.21 supports aarch64 + CUDA 13.
2. Meson-rebuild nixl 1.3 from `tools/nixl-src` against that UCX (keep the no-etcd
   configuration). The install dir becomes `lib/aarch64-linux-gnu/` — the path *name*
   changes, not just the contents.
3. Patch `experimental/starrocks/pixi.toml`: `LD_LIBRARY_PATH` and `NIXL_PLUGIN_DIR` embed
   `.../nvda_nixl/lib/x86_64-linux-gnu` AND absolute `/home/ubuntu/...` prefixes; add
   `CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER` (the X86_64 variant is inert on Grace).
4. `pixi install` — both pixi.toml files already declare `linux-aarch64-cuda13` and the lock
   carries a solved linux-aarch64 graph — then a full engine + CN rebuild
   (`build/release/**` and `.pixi/**` are x86 by construction).
5. Nothing needed for engine B: `starrocks/artifacts-ubuntu:3.5.20` is multi-arch (arm64
   manifest verified); point JAVA_HOME at an aarch64 JDK 17. The vendored FE is jars-only;
   Sirius C++/CUDA has no x86 intrinsics.

**Verify**: CN starts, `file` on the loaded .so's says aarch64, q06 passes; then M1.

## M1. Multi-GPU cluster bring-up (config, no code)

**What**: the demo's cluster task is hardcoded to 2 CNs sharing one GPU. The GB200 box
wants a `cluster4`: one CN per GPU, NUMA-pinned.

**Where**: `experimental/starrocks/pixi.toml` — the `cluster2` task (~line 60-81) launches
the FE and two `sirius-starrocks-cn` processes with `--gpu-memory-limit 8GiB
--host-memory-limit 12GiB --engine-dir .cnN` and per-CN port offsets (+2 per CN:
heartbeat 9050/9052/…, thrift 9060/9062/…, brpc 8060/8062/…, http 8040/8042/…,
starlet 9070/9072/…).

**Do**:
1. Copy the `cluster2` task to a `cluster4` variant (or a small launcher script): one CN
   per GPU via `--gpu-device <i>` (mandatory — the nixl descriptor hardcodes device 0, so
   exactly one GPU may be visible per CN), distinct `--engine-dir .cn<i>`, the +2 port
   ladder, `MIN_BACKENDS=4` in bench.sh.
2. NUMA-pin per GPU affinity: CN0/CN1 `numactl --physcpubind=0-71 --membind=0`, CN2/CN3
   `--physcpubind=72-143 --membind=1`. Never membind the HBM NUMA nodes (2/10/18/26). The
   CN's derived YAML emits only a flat host `capacity_bytes` (no `numa_id`) — rely on
   membind (ROADMAP-8CN 4c).
3. Raise `--gpu-memory-limit` to ~140–150 GiB (SF100) / ~128 GiB (SF1000) — the 8 GiB
   figure exists only because two CNs shared a 23 GiB L4. Leave headroom for the staging
   arena + CUDA context: the arena sits OUTSIDE the limit (see the pixi.toml comment,
   PLAN-PATH-B D-B4). `--host-memory-limit` ~160–200 GiB/CN, leaving ≥400 GB of LPDDR for
   page cache; verify total host RAM with `free -g` (expect 2× 480 GB = 960 GB).
4. Raise `SIRIUS_QUERY_WATCHDOG_SECS` (≥180 at SF1000) and make the 60 s CN↔CN
   REPLY_TIMEOUT configurable — a >60 s receiving fragment fails every waiting sender.
5. Pre-establish all 12 directed nixl sessions before the first query: the first-contact
   transport-thread MD deadlock (ROADMAP-8CN 4b-1) never fired at 2 CNs but is
   near-certain under 4-way bidirectional shuffle.
6. The FE needs no changes — CNs self-register via heartbeat. Stale registrations from old
   topologies: `ALTER SYSTEM DROP COMPUTE NODE '<host:port>'`.

**Verify**: `SHOW COMPUTE NODES` = 4 alive; the full sweep
(`benchmarks/tpch/bench.sh`) 22/22; then the endurance shape (2-3 back-to-back sweeps, no
restarts) since exchange traffic patterns change with N.

**Watch out**: `UCX_TLS=cuda_copy,cuda_ipc,tcp,self` suffices — cuda_ipc rides NVLink P2P
automatically on the all-to-all NV18 box; keep the 2 GB/s bandwidth canary to catch silent
fallback. The nixl plugin dir env is absolute in pixi.toml — M0 fixes the paths. Multi-NODE
(not just multi-GPU) additionally needs the nixl RDMA tier, which this demo has never
exercised — the 8 idle mlx5 NICs stay a separate project, not a config change.

## M2. Staging arena: sizing gates SF100, reclaim semantics gate SF1000

**What**: `SIRIUS_EXCHANGE_STAGING_BYTES` is a hand-tuned 1280 MiB, bump-reset only at full
quiescence. Demand scales with fan-in (a receiver holds staging from up to N−1 senders) and
with data volume (TPC-H q09 needs a single ~648 MB lease at SF1 with 2 CNs). At SF100 on
4 CNs, staged inbound per CN is 1.9–6 GB (q03/q05/q17/q18/q21) and single leases exceed the
whole slab (q16 broadcast payload 1.28 GB, q04 3.0 GB) — ≥7 queries fail on day one. At
SF1000 the bump-reset semantics themselves are the limit: demand is cumulative per query
epoch, ~105–125 GB/CN (q05/q18/q21), and no fixed slab coexists with 80–150 GB operator
peaks in 185 GiB HBM (`TPCH-PLAN-ANALYSIS.md` §8.3).

**Where**: the env var is read at CN bring-up (`experimental/starrocks/src/engine_settings.rs`
derives the engine config; the arena itself is `src/exec/exchange_staging_arena.{hpp,cpp}`).

**Do**, in two phases:
1. **Sizing + chunked leases (gates SF100)**: derive a default when the env var is unset —
   proportional to free GPU memory after carve-out (16–32 GiB is trivial on 185 GiB) — and
   FAIL LOUDLY at bring-up if arena + carve-out + context exceed device memory. Cap any
   single lease (~256 MB) with chunked multi-lease export for q09/q16/q04-class payloads.
   Keep the env var as the override; log the chosen size and high_water.
2. **Eager per-lease reclamation (gates SF1000)**: replace bump-at-quiescence with a
   free-list or ring — release on receiver-push / post-WRITE — so demand becomes in-flight
   (senders × chunk × pipeline depth ≈ low GB) instead of cumulative-per-epoch. Add the
   broadcast lease-reuse loop (one lease, N−1 WRITEs). Longer term, peer-direct UCX
   registration retires the arena entirely — an optimization, not a prerequisite (§8.3).

**Verify**: bring-up on 1-GPU and N-GPU boxes without setting the var; q16 at SF100 (the
single-lease unit test) and q09 pass; q05/q18/q21 at SF1000 with high_water ≪ cumulative
epoch bytes; the oversized-request error still names request/free/capacity; endurance
sweeps.

## M3. A fair benchmark at scale (SF100 / SF1000)

**What**: at SF1 (388 MB), fixed per-query overheads made stock StarRocks ~2× faster
overall (geo-mean 0.48x; Sirius won only q01/q09/q19). The GPU's economics need data. This
is the demo's headline experiment on the GB200 box. Targets: **SF100** (600M lineitem rows;
measured parquet 35.7 GB total, lineitem 22.8 GB, snappy) and **SF1000** (6.0B lineitem
rows; ~355–390 GB total, lineitem ~228 GB). Per-CN share at 4 CNs: 150M / 1.5B lineitem
rows — 1.5B is 70% of cuDF's 2^31 row cap, so 4 CNs is the floor topology for SF1000.

**Do**: generate with the `dataset-manager` skill (or DuckDB
`CALL dbgen(sf=...); EXPORT DATABASE '<dir>' (FORMAT PARQUET);`) as **multiple files per
table, a multiple of 4**: SF100 8–16 files/table; SF1000 lineitem 32–64 files (~2.5–5 GB
each), row groups 128–256 MB — whole-file-contiguous ranges balance the FE's byte-range
splits. Then rerun `benchmarks/tpch/run-comparison.sh` with `TPCH_DATA` pointed at it.

**Storage**: local NVMe, not network — cold SF1000 reads ~90–97 GB/CN. One warm pass
page-caches the entire dataset in the ~960 GB of LPDDR (verify at bring-up, M1), after
which scans come from RAM over C2C — publish cold and warm medians separately.

**Watch out**: arena sizing (M2 phase 1) gates SF100 day one; q08/q09 stay excluded until
the cross-join re-association lands (their intermediates are O(SF²): 2.1 TB / 13 TB at
SF100 — `TPCH-PLAN-ANALYSIS.md` §8.2); q04/q10/q13 need the 2^31 guards at SF1000. Engine B
(stock StarRocks BEs on the 144 Grace cores) needs a re-baselined `mem_limit` in
`setup-engine-b.sh` — don't reuse the L4-host numbers.

---

## #24. Decimal-native aggregation — the correctness capstone (largest, highest value)

**What**: the FE plans money arithmetic as exact DECIMAL64/128; the translator lowers
decimal SUM/AVG to FP64. Consequences, all measured:
- Every `sum(x*(1-l_discount))` lands 0.1–0.4 % LOW vs DuckDB (q01 sum_disc_price/charge,
  q03/q05/q07/q09/q10 revenues, q08 mkt_share, q14 promo_revenue) — deterministic,
  low-biased, occasionally reorders near-tie ORDER BY rows (q10 rank rotation).
- FP64 hash-groupby accumulates via cudf atomicAdd (bit-nondeterministic), which forced the
  q15 workaround: canonical-order + `sorted::YES` (atomics-free path) on every float-SUM
  groupby — correct, but pays a sort per aggregation.

**Where** (all mapped during the q15 investigation — wf journals + QUERY-TIMEOUT-ANALYSIS.md):
- Translator: `expr_translator.rs:826-833` (decimal SUM/AVG → `cast_to_fp64`, :999-1003),
  `type_mapper.rs` (DECIMAL(p>18) → Fp64 lowering), the partial-state wire-type model
  (`partial_state.rs` — decimal sums modeled as FP64 on the wire).
- Engine: cuDF supports `fixed_point` (DECIMAL32/64/128) natively; fixed-point SUM uses
  exact integer atomics (`device_aggregators.cuh:126`) — deterministic and exact. The scan
  already reads decimal parquet columns; the gaps are the expression path
  (`src/expression/`, arithmetic over fixed_point incl. the (1 − DECIMAL) literal shape),
  aggregate/merge output types (`gpu_aggregate_impl.cpp`, `gpu_merge_impl.cpp`,
  `aggregate_op_util.cpp`), and the exchange schema spelling (`type_mapper.rs` ↔
  `get_cudf_type`, `cudf_utils.hpp:158-210` — DECIMAL(p,s) strings already round-trip).
- Mind the 76-case `wire_type_parity` gate (`experimental/starrocks/src/wire_type_parity.rs`)
  — it will enforce whatever the model says; update model + engine together.

**Approach hint**: start with SUM(DECIMAL64) end-to-end (scan → partial sum → wire → merge
→ finalize), gate q06/q01 on bit-exactness vs DuckDB, then widen (AVG expansion states,
DECIMAL128, the (1−d) literal). The avg `__count` expansion machinery is orthogonal and
stays.

**Acceptance**: q01/q05/q06/q14 values bit-exact vs DuckDB (no tolerance band); q15 stays
8/8 WITHOUT the canonical-sort path on decimal inputs (keep it for genuine float columns);
the overflow guard (`throw_if_int64_sum_could_overflow`) still covers 64-bit integer sums;
full suite + sweep green.

## #31. The engine-abort surface — kill the silent-failure family (medium)

Three catalogued gaps, one root: the engine cannot abort a running fragment.

1. **Unhandled sink types are silently swallowed**:
   `experimental/starrocks/src/compute_node_service.rs` ~:752-760 — a fragment whose output
   sink is not DATA_STREAM_SINK (e.g. MULTI_CAST_DATA_STREAM_SINK, emitted under
   `cbo_cte_force_reuse_node_count=1`) falls into a let-else returning `Ok(Vec::new())`:
   output discarded, consumers hang, the FE's serial channel wedges cluster-wide.
   Fix: `Err` on any unhandled sink (do this first — 30 minutes); optionally implement
   MULTI_CAST (run the fragment once, park/transmit per nested sink destination) — that also
   enables single-evaluation CTE reuse, structurally removing q15's double-evaluation shape.
2. **Parked-sender bookkeeping defect**: `"no parked sender output to export for
   SenderSlot { node_id: 3, sender_id: 1 }"`, reproducible by adding `CAST(sum AS VARCHAR)`
   to a grouped-CTE probe. Uninvestigated; likely the park/claim accounting for a plan shape
   where an expected sender output was never parked.
3. **Mid-query export failure wedges the CN**: when a sender-side export fails mid-run, the
   client sees a silent timeout (not the loud error) and the wedged statement head-of-line
   blocks the next until restart. The failure propagation (c858e79a) covers pre-run
   failures; mid-run needs the engine to actually abort the fragment. Scaffolding exists:
   the watchdog + `terminate_query` + SIGTERM escalation (a94e8660) and `cancel_plan_fragment`
   stubs (PRPC + thrift) — the missing piece is engine-side fragment abort (interrupt the
   stream waits / task loop; respect the 19d7cca2 manager-thread constraints).

**Acceptance**: no plan shape can hang silently — every unsupported/failed path yields a
loud FE-visible error within seconds AND the next statement runs without restarting; a
real cancel frees GPU memory mid-query.

---

## Verification protocol (any of the above)

The suite ladder and live-gate protocol are in the `tpch-bench` skill; in short:
translator → cn-test-no-engine → cn-test (incl. the wire-type parity gate) → C++ suite if
`src/**` changed → GPU harness → live: the affected queries solo vs the DuckDB oracle →
a full sweep, and for anything touching exchange/arena, an endurance shape (2-3 sweeps,
zero restarts). Every fix lands with the regression test that would have caught it.
