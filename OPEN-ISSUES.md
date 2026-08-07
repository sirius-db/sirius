# Open issues — the work queue for the multi-GPU box

State of the branch this doc ships with: **22/22 TPC-H pass at SF1**, unbounded back-to-back
execution (the arena leak is fixed), every result DuckDB-validated. The journey and evidence
live in `QUERY-TIMEOUT-ANALYSIS.md`; the integration learnings in `ROADMAP-8CN-TPCH.md`; the
benchmark protocol in `experimental/starrocks/benchmarks/tpch/REPRODUCE.md`; day-to-day
operations in the `tpch-bench` skill (`.claude/skills/tpch-bench/SKILL.md`). Reference
numbers: `experimental/starrocks/benchmarks/tpch/results/sf1-2026-08-07.md`.

Recommended order on a fresh multi-GPU box: **M1 → M2 → #24 → #31**, with M3 whenever a
bigger dataset is available. M1/M2 unblock everything else on the new hardware.

---

## M1. Multi-GPU cluster bring-up (config, no code)

**What**: the demo's cluster task is hardcoded to 2 CNs sharing one GPU. A multi-GPU box
wants one CN per GPU.

**Where**: `experimental/starrocks/pixi.toml` — the `cluster2` task (~line 60-81) launches
the FE and two `sirius-starrocks-cn` processes with `--gpu-memory-limit 8GiB
--host-memory-limit 12GiB --engine-dir .cnN` and per-CN port offsets (+2 per CN:
heartbeat 9050/9052/…, thrift 9060/9062/…, brpc 8060/8062/…, http 8040/8042/…,
starlet 9070/9072/…).

**Do**:
1. Copy the `cluster2` task to a `clusterN` variant (or a small launcher script) that starts
   one CN per GPU with `CUDA_VISIBLE_DEVICES=<i>` per process, distinct `--engine-dir .cn<i>`,
   and the +2 port ladder.
2. With a dedicated GPU per CN, raise `--gpu-memory-limit` (the 8 GiB figure exists only
   because two CNs shared a 23 GiB L4). Leave headroom for the staging arena + CUDA context:
   the arena sits OUTSIDE the limit (see the pixi.toml comment, PLAN-PATH-B D-B4).
3. The FE needs no changes — CNs self-register via heartbeat. Stale registrations from old
   topologies: `ALTER SYSTEM DROP COMPUTE NODE '<host:port>'`.

**Verify**: `SHOW COMPUTE NODES` = N alive; the full sweep
(`benchmarks/tpch/bench.sh`) 22/22; then the endurance shape (2-3 back-to-back sweeps, no
restarts) since exchange traffic patterns change with N.

**Watch out**: `UCX_TLS` must include `cuda_ipc` + `cuda_copy` (cross-GPU same-host peer
copies ride cuda_ipc); the nixl plugin dir env is absolute in pixi.toml — fix paths for the
new box. Multi-NODE (not just multi-GPU) additionally needs the nixl RDMA tier, which this
demo has never exercised — treat that as a separate project, not a config change.

## M2. Staging-arena auto-sizing (small code)

**What**: `SIRIUS_EXCHANGE_STAGING_BYTES` is a hand-tuned 1280 MiB. Demand scales with
fan-in (a receiver holds staging from up to N−1 senders) and with data volume (TPC-H q09
needs a single ~648 MB lease at SF1 with 2 CNs).

**Where**: the env var is read at CN bring-up (`experimental/starrocks/src/engine_settings.rs`
derives the engine config; the arena itself is `src/exec/exchange_staging_arena.{hpp,cpp}`).

**Do**: derive a default when the env var is unset: something like
`min(free_gpu_after_carveout × 0.5, base × (N_backends − 1))` — and FAIL LOUDLY at bring-up
if the derived arena + carve-out + context exceed device memory. Keep the env var as the
override. Log the chosen size at startup.

**Verify**: bring-up on 1-GPU and N-GPU boxes without setting the var; q09 passes; the
oversized-request error still names request/free/capacity.

## M3. A fair benchmark at scale (SF10+)

**What**: at SF1 (388 MB), fixed per-query overheads made stock StarRocks ~2× faster
overall (geo-mean 0.48x; Sirius won only q01/q09/q19). The GPU's economics need data. This
is the demo's headline experiment on real hardware.

**Do**: generate TPC-H SF10 (or SF30) parquet — DuckDB one-liner:
`CALL dbgen(sf=10); EXPORT DATABASE '<dir>' (FORMAT PARQUET);` then arrange
`<table>/*.parquet` — and rerun `benchmarks/tpch/run-comparison.sh` with `TPCH_DATA`
pointed at it. Multiple files per table parallelize scans across CNs (the FE byte-splits,
but per-file splits distribute better).

**Watch out**: arena sizing (M2) matters immediately at SF10 — q09-class single leases grow
~linearly with SF. Engine B (stock StarRocks BEs) needs enough host memory
(`mem_limit` in `setup-engine-b.sh` configs).

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
