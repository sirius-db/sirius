---
name: tpch-2host
description: >
  Bring up Sirius StarRocks CNs across two physical GB200 hosts with nixl over cross-host NVLink,
  and the matching stock StarRocks (engine B) two-machine baseline. Use when replicating the
  two-host experiment, setting SIRIUS_EXCHANGE_STAGING_ARENA=fabric, debugging a 0.4 GB/s
  cuda_ipc fallback, or running engine B as 1 FE + 1 BE per host.
---

**Working procedure (engine A):** read [`experimental/starrocks/benchmarks/2NODE-REPLICATE.md`](../../../experimental/starrocks/benchmarks/2NODE-REPLICATE.md) fully. Measured 2026-08-11/12 on `presto-gb200-gcn-17` + `gcn-18`: 1 CN per host, **98 GB/s** canary.

**Engine B tutorial:** [`2NODE-ENGINE-B-TUTORIAL.md`](../../../experimental/starrocks/benchmarks/2NODE-ENGINE-B-TUTORIAL.md). Results snapshot: [`2NODE-ENGINE-B-RESULTS.md`](../../../experimental/starrocks/benchmarks/2NODE-ENGINE-B-RESULTS.md).

Do **not** follow [`notes/2026-08-11-2node-gb200/`](../../../notes/2026-08-11-2node-gb200/) as the procedure. Those files are historical: PHASE1-RESULTS is the pre-fabric refusal (~0.4 GB/s); RUNBOOK commands are corrected in GAPS and then superseded by REPLICATE. Living leftovers (8 CN / 2 hosts) are in [`notes/OPEN.md`](../../../notes/OPEN.md) Phase 2.

You need a shell on **each** host.

## The load-bearing knob

```bash
export SIRIUS_EXCHANGE_STAGING_ARENA=fabric
```

`cudaMalloc`'s IPC handle is node-local. A peer on another host cannot map it; UCX silently falls back to a host bounce at ~0.4 GB/s, which is below the 2.0 GB/s canary floor, so the peer is refused and **no distributed query runs**. Fabric arena is what makes cross-host NVLink work.

## Traps

- **Do not put `rc_mlx5` in `UCX_TLS`.** The CN fails to start. `nvidia_peermem` is not loaded; GPUDirect RDMA is not the path on these hosts.
- Engine A `run_mode = shared_data`. Engine B must stay **`shared_nothing`** (default) — `FILES()` `FileScanNode` looks up BEs, not CNs. Copying A's `run_mode` into B yields "No available backends".
- Engine B registration is manual `ALTER SYSTEM ADD BACKEND`. A's CN self-registers.
- **A and B cannot run at once** on either host (port 9030, all CPUs). `pgrep -af sirius-starrocks-cn` before starting B.
- Verify canary in the cluster log: expect ~98 GB/s, not 0.4. `nvidia-smi nvlink -gt d` deltas must move.
- Phase 1 is 1 CN per host. Phase 2 (8 CNs / 2 hosts) is **not** in REPLICATE and is still open.

## Engine B shape

1 FE on gcn-18 + 1 unpinned BE per host, all 144 cores. No GPU. `mem_limit` is unsettled — choose deliberately, record it, never let `setup-engine-b.sh` rewrite `16G` on every run (OPEN M0.5).
