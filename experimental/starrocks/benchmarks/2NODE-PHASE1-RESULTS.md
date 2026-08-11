# Phase 1 (engine A, 2 CNs across 2 hosts) — execution record

**2026-08-11, gcn-17 + gcn-18.** One Sirius CN per host, GPU 0 each, per
`2NODE-BRINGUP-PLAN.md` Phase 1. Everything below is MEASURED unless marked otherwise.

## Verdict

**The two-machine cluster FORMS and StarRocks SCHEDULES across both hosts. No query can run,
because the cross-host GPU transport is refused.**

Clean run 2026-08-11 23:26-23:31 UTC, after the engine bug below was fixed. The causal chain is
now established end to end, with each link measured separately:

| # | Link | Result |
|---|---|---|
| 1 | Both CNs register and go alive | **PASS** — `10001`@.53, `10004`@.52, 72 cores each |
| 2 | Engine plans and executes | **PASS** — single-CN `nation` → 25, `lineitem` → 600,037,902 |
| 3 | FE schedules fragments on both hosts | **PASS** — F00 and F01 each have instances on `10001` *and* `10004` |
| 4 | Cross-host exchange carries data | **FAIL** — canary 0.32-0.43 GB/s vs a 2.0 GB/s floor, peer refused |

Step 4 is the whole of the remaining gap.

### Every query fails, not just shuffles

With **one CN per host there is no same-host peer**, so every exchange in every plan is remote. A
`count(*)` over `nation` — 2,250 bytes, one file — returned 25 while only one CN was up, and fails
once the second CN joins:

```
ERROR 1064: nixl link to 10.87.140.53:9102 measured 0.43 GB/s, below the 2 GB/s floor
            — Refusing the transport tier    backend [id=10004]
```

The refusal is **bidirectional**: gcn-18 measures the link to gcn-17 at 0.32-0.43 GB/s and gcn-17
measures the link to gcn-18 at 0.43 GB/s. Both refuse.

This is worth stating plainly because it inverts the usual intuition: adding the second node makes
the cluster *strictly less* capable than a single node, until the transport works.

### Blocker 2 (engine) — FIXED, commit `f8360593`

`TransactionContext::ActiveTransaction called without active transaction` on every `FILES()` plan.
Root cause: `Fragment::build()` commits its view-creation transaction (`sirius_ffi.cpp:565`) before
opening the `StandaloneQueryScope`, so `lower_substrait()` bound the plan with no transaction open.
Every catalog lookup goes through `TransactionContext::ActiveTransaction()`, which DuckDB 1.5.5
throws from and 1.5.4 tolerated — so the regression arrived with the submodule bump in `a3c99f4a`,
not with any change to `src/` (unchanged since 08-07).

Fixed by opening a transaction in `lower_substrait()` only when the caller has not, using
`ClientContext::transaction` rather than `Connection::BeginTransaction()` (the latter runs
`Query("BEGIN TRANSACTION")`, an ordinary statement that would take the lifecycle mutex the
enclosing scope holds — `duckdb/src/main/connection.cpp:341`).

Verified: `nation` → 25 rows, `lineitem` → 600,037,902 rows / `sum(l_quantity)` 15,300,829,209 —
byte-identical to stock StarRocks on the same files.

### Unrelated gotcha found while writing the validation query

The plan's shuffle query used `l_orderkey % 4096` as the grouping key. Sirius refuses it:

```
Unsupported expression in projection (falling back to CPU): mod(l_orderkey, 4096)
```

Use a plain column (`GROUP BY l_suppkey` — 1M groups at SF100) instead. The engine-B tutorial's
copy of that query is fine, because stock StarRocks supports `mod()`; only the engine-A variant
needs changing.

## What works

Both CNs registered themselves and went alive. No StarRocks patch was needed anywhere in the
FE↔CN control path.

```
ComputeNodeId: 10002   IP: 10.87.140.52 (gcn-17)   Alive: true
ComputeNodeId: 10001   IP: 10.87.140.53 (gcn-18)   Alive: true
```

* Registration is automatic — each CN dials the FE over MySQL and runs `ALTER SYSTEM ADD
  COMPUTE NODE`; no manual step, no FE/CN ordering requirement.
* `--advertise-host` works: the two nodes are distinct rather than colliding on loopback.
* The FE identity blocker (plan Task 1.1 Step 3b) is resolved by a fresh node-local `meta_dir`
  (`/raid/prestouser/sr-eng-a-2node/fe/meta`); the FE logged
  `Use IP init local addr, IP: /10.87.140.53` and proceeded.
* `cn-2host.sh` works on first real use, including `NUM_CNS_PER_HOST=1 CN_CPUS="0-71"` and the
  host-suffixed `engine-dir=.cn0-53` that stops the two hosts racing on the same NFS config.

**Deviation from the plan:** the FE runs on **gcn-18**, not gcn-17 (the operator could only start
processes there). `priority_networks = 10.87.140.32/27` covers both, so this is functionally
equivalent — and it matches where engine B's FE ran, which removes the coordinator host as a free
variable in the A/B.

## Blocker 1 — no cross-host GPU transport (MEASURED)

| Path | Result |
|---|---|
| `rc_mlx5` (GPUDirect RDMA) | **CN fails to start** |
| `tcp` host-staged | **0.41–0.42 GB/s**, peer refused |
| MNNVL fabric | unreachable — arena is plain `cudaMalloc` |

**GPUDirect RDMA is unavailable on these hosts.** `lsmod | grep nvidia_peermem` → not loaded, and
`/sys/kernel/mm/memory_peers` does not exist. Without a peer-memory client the mlx5 HCAs cannot pin
VRAM, so registering the 16 GiB device-memory staging arena is rejected outright:

```
ib_md.c:307 UCX ERROR ibv_reg_mr(address=0xfff680000000, length=2147483648,
                                 access=0x10000f) failed: Bad address
ucp_mm.c:81 UCX ERROR failed to register address 0xfff5c0000000 (cuda) length 17179869184
                      on md[4]=mlx5_0: Input/output error (md supports: host|cuda-managed)
nixl_agent.cpp:538 registerMem: registration failed for the specified or all potential backends
Error: failed to bring up the nixl exchange transport: failed to register the
       17179869184-byte staging arena with nixl
```

Note `md supports: host|cuda-managed` — the HCA offers host and managed memory, not device memory.
With `rc_mlx5` in `UCX_TLS` this is a **hard startup failure**, not a slow path.

Dropping `rc_mlx5` (`UCX_TLS=cuda_copy,cuda_ipc,tcp,self`) lets the arena register:

```
nixl transport ready; staging arena registered
  agent=10.87.140.53:9102  staging_capacity=17179869184
```

…and then the mandatory canary refuses the peer:

```
nixl bandwidth canary peer=10.87.140.52:9102 gbps="0.4" bytes=16777216
nixl link to 10.87.140.52:9102 measured 0.41 GB/s, below the 2 GB/s floor
  — Refusing the transport tier
```

The floor is a compile-time `const` (`src/nixl_transport.rs:318`) with no env override, and there
is **no fallback tier**. With one CN per host every exchange is remote, so a refused peer means the
cluster cannot run a single distributed query.

### This corrects `2NODE-RUNBOOK.md` §2

| Runbook row | Status |
|---|---|
| `cudaMalloc` + default TLS → **0.37 GB/s**, refused | **REPRODUCED** (0.41 GB/s here) |
| `+ rc_mlx5`, 1 NIC → 48.7 GB/s | **NOT REPRODUCIBLE** — CN cannot start |
| `+ rc_mlx5`, 4 NICs → 97 GB/s | **NOT REPRODUCIBLE** — CN cannot start |
| VMM fabric handle → 765 GB/s | untested here; harness-only, unreachable from the CN |

The two RDMA rows are exactly the ones `2NODE-RUNBOOK-GAPS.md` §A flagged as having **no surviving
on-disk artifact**. §2's "what to do today" prescribes that configuration; on this host it prevents
startup. **Do not follow it without first loading `nvidia_peermem`.**

## Blocker 2 — engine cannot plan a fragment (MEASURED, not two-node)

Every `FILES()` query fails at plan time, including a single-CN scan of a 2,250-byte table with no
exchange. `SELECT 1` (FE-only) succeeds.

```
ERROR 1064 (HY000): failed to plan fragment:
  {"exception_type":"INTERNAL",
   "exception_message":"TransactionContext::ActiveTransaction called without active transaction"}
```

Ruled out:

* **Not CN/engine version skew.** Rebuilt the CN against the current engine; identical error.
* **Not uncommitted source.** `git status src/` clean; `src/` unchanged since 08-07.
* **Not a stray submodule.** `duckdb` is at the committed pointer `d8cdaa33` (v1.5.5), bumped
  08-03 in `a3c99f4a`.

So the committed combination (src @ 08-07 + DuckDB v1.5.5) does not plan a fragment. **Unresolved
— needs its own debugging session.** It reproduces on a single host, so a two-node cluster is not
required to investigate it.

Context that muddies the history: the pre-existing `build/release` tree was internally
inconsistent — `core_functions`/`parquet` were v1.5.5 (08-11 00:34) while
`sirius.duckdb_extension` was v1.5.4 (08-09 04:17), so the CN refused to start at all with
`built for DuckDB version 'v1.5.5' ... this version is 'v1.5.4'`. Rebuilding the engine was
necessary and made the tree consistent; it did not cause this error, but the last state in which
engine A demonstrably ran is not recoverable from the current tree.

## Fixed along the way (committed `d271522a`)

Two latent bugs in `scripts/cn-env.sh`, both of which block *any* CN rebuild, one host or two:

1. **`LIBRARY_PATH` omitted the CUDA driver libs.** `libsirius.so` declares `libcuda.so.1` and
   `libnvidia-ml.so.1` in `DT_NEEDED`, but `LIBRARY_PATH` was set to only UCX's lib dir, so the
   link failed on `cuLaunchKernel` / `nvmlDeviceGetMemoryAffinity`. `LD_LIBRARY_PATH` is run time;
   the link needs `LIBRARY_PATH`.
2. **The conda `ld` was used against the system `gcc`.** `CC=/usr/bin/gcc`, but gcc resolves `ld`
   from `PATH`, which under `pixi run` is conda's — linking the conda sysroot's `libpthread.so.0`
   against system libc and failing with 39 `GLIBC_PRIVATE` undefined references.

Do **not** fix (2) with `RUSTFLAGS`: that invalidates cargo's fingerprint cache, re-runs the
`nixl-sys` build script, and hits a separate latent failure
(`/usr/include/features-time64.h: fatal error: bits/timesize.h: No such file or directory`).

Also note `pixi run cn-build` still cannot be used: it `depends-on = engine-build`, and the full
`make` fails on `test/cpp/exec/test_streaming_fragment.cpp`, which uses a `streaming_fragment::build()`
signature and a `query_lifecycle` symbol that no longer exist. The engine extension itself links
fine — only the unit-test target fails. Build the CN directly:

```bash
cd experimental/starrocks
NIXL_NO_STUBS_FALLBACK=1 pixi run bash -lc \
  'source scripts/cn-env.sh; cargo build --release -p sirius-starrocks-cn'
```

## To unblock Phase 1

Both are required; neither is sufficient alone.

1. **Transport.** Either `modprobe nvidia_peermem` (root, one command — then re-test `rc_mlx5`), or
   implement the fabric-handle staging arena (`cuMemCreate` + `CU_MEM_HANDLE_TYPE_FABRIC`) to reach
   the MNNVL path. `two_node_harness::cuda_vmm` is a working reference and the IMEX domain is
   verified up (`Domain State: UP`, ClusterUUID `3482beb4-a3cd-48a4-9b6c-a6ba43bc59a4` on gcn-18).
2. **Engine.** Root-cause `TransactionContext::ActiveTransaction`.

## Reproduce

```bash
# gcn-18 (FE + CN0)
cd ~/aocsa/sirius/experimental/starrocks
UCX_TLS=cuda_copy,cuda_ipc,tcp,self NUM_CNS_PER_HOST=1 CN_NODE="0" CN_CPUS="0-71" \
  ./benchmarks/cn-2host.sh 10.87.140.53 10.87.140.53

# gcn-17 (CN0 only)
UCX_TLS=cuda_copy,cuda_ipc,tcp,self NUM_CNS_PER_HOST=1 CN_NODE="0" CN_CPUS="0-71" \
  ./benchmarks/cn-2host.sh 10.87.140.52 10.87.140.53 --no-fe

# then
grep 'nixl bandwidth canary' /tmp/cn-5*.log
```
