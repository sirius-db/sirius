# Engine A config set — 4× GB200, 4 Sirius CNs

Configuration for running Sirius as 4 StarRocks compute nodes, one per GB200 GPU, on this
specific box. Engine A is Sirius; engine B is stock StarRocks (`../gb200-4gpu/engine-b`, if
present, and `../../benchmarks/tpch/setup-engine-b.sh`).

| file | what it is |
|---|---|
| `engine-a.env` | every tunable, with the arithmetic behind each value. Bash (defines arrays). |
| `cluster4-numa.sh` | NUMA-pinned launcher: 1 FE + 4 CNs. Preflights, then `wait -n`. |
| `README.md` | this file |

These are **new files**. They do not modify `benchmarks/cluster8.sh`, `scripts/cn-env.sh`, or
`pixi.toml`; `cluster4-numa.sh` sources `scripts/cn-env.sh` unchanged and reuses the same port
ladder, so it is a drop-in alternative to `cluster8.sh`, not a replacement for it.

---

## 1. The box

### GPUs and interconnect

`nvidia-smi topo -m`:

```
        GPU0    GPU1    GPU2    GPU3    CPU Affinity   NUMA Affinity    GPU NUMA ID
GPU0     X      NV18    NV18    NV18    0-71           0,3-9,11-17       2
GPU1    NV18     X      NV18    NV18    0-71           0,3-9,11-17      10
GPU2    NV18    NV18     X      NV18    72-143         1,19-25,27-33    18
GPU3    NV18    NV18    NV18     X      72-143         1,19-25,27-33    26
```

- 4× NVIDIA GB200, compute capability 10.0, **185 GB HBM each**.
- **NV18 between every GPU pair** — an all-to-all full mesh of 18 bonded NVLinks. There is no
  PCIe/SYS hop between any two GPUs, so all four CNs are equidistant peers and no CN placement is
  preferable to another.
- 8× mlx5 NICs (`mlx5_0..7`). **Every NIC is `SYS` from every GPU** — no PCIe locality to any GPU.
  NIC2/NIC3 are `PIX` to each other, as are NIC6/NIC7.

**The NICs are idle and must stay that way.** This box is single-node, so the nixl RDMA tier is
never used. Do not configure RDMA, and do not add `rc`/`ud`/`ib` to `UCX_TLS` — the only thing that
can achieve is to let UCX pick a fabric hop over a 350 GB/s NVLink hop.

### NUMA — 34 nodes, and four of them are GPU memory

| node | CPUs | memory | what it is |
|---|---|---|---|
| 0 | 0–71 | 489,960 MB (478.48 GiB) | **LPDDR, socket 0** |
| 1 | 72–143 | 489,823 MB (478.34 GiB) | **LPDDR, socket 1** |
| 2, 10, 18, 26 | *none* | 188,416 MB each | **GPU HBM** (GPU0, GPU1, GPU2, GPU3) |
| all others | none | 0 MB | — |

- **CPU-addressable LPDDR is ~980 GB (979,783 MiB = 956.82 GiB), not 1.7 TB.**
  `free -g` reports ~1692 GiB because the kernel counts the four 184 GiB HBM nodes as system
  memory. `956.82 + 736.0 = 1692.8` — that is the whole discrepancy. **All memory budgeting uses
  956.82 GiB.** Never size anything from `free`, `/proc/meminfo`, or a percentage of them.
- CPU: 144 Grace cores, aarch64, 72 per socket.
- CUDA 13.0, driver 580.105.08, `CUDA_HOME=/usr/local/cuda`.
- `numactl` is at `/usr/bin/numactl`.
- **Swap is 0.** An over-commit is an OOM-kill, not a slowdown. This is why every budget below
  carries an explicit margin.
- `$HOME` is NFS: slow, and its clock runs ~0.2 s ahead of local, which breaks meson-style
  clock-skew checks. Do not put build dirs there.
- No Docker.

---

## 2. ⚠️ DO NOT membind NUMA nodes 2, 10, 18 or 26

**Those four nodes are the GPUs' HBM.** They have zero CPUs and 188,416 MB each. Binding a CN's
host memory to one of them consumes the HBM of a GPU that a CN is computing on — you will lose
GPU memory with no message saying so.

Unpinned CNs — and an unpinned FE — run today with:

```
Mems_allowed_list:  0-2,10,18,26        <-- GPU0's HBM is in the allowed set
```

### What the `--membind` is and is not buying

An earlier draft of this config set justified the membind with a claim that **does not survive
measurement**, and it is retracted here rather than quietly deleted:

> ~~"node 2's free memory tracks `nvidia-smi`'s free memory for GPU0 exactly."~~

Measured with 4 CNs live, `numactl --hardware` reported node 2 free = **188,386 MB** while
`nvidia-smi` reported GPU0 **used = 160,496 MiB / free = 27,921 MiB** — a ~160 GiB disagreement,
exactly the RMM pool. The two counters coincide *only at idle* (both 188,389 MiB with the GPUs
idle), because the kernel's NUMA free counter does not account for driver device allocations at
all. It can therefore never be evidence that host pages are landing in HBM.

Pointing the other way: **the engine's own NUMA-explicit host allocator already targets a CPU
node without any help from `numactl`.** `cucascade/src/memory/numa_region_pinned_host_allocator.cpp:60`
calls `numa_alloc_onnode(bytes, _numa_node)`, where `_numa_node` comes from `use_host_per_numa()`
→ `bind_cpu_to_gpu_numa` → `gpu.numa_id`, derived from `nvmlDeviceGetMemoryAffinity(...,
NVML_AFFINITY_SCOPE_NODE)` (`cucascade/src/memory/topology_discovery.cpp:269-277`). The live CN log
confirms it: `GPU 0: NVIDIA GB200 (numa=0, pci=…)`. So `CN_NODE="0 0 1 1"` **agrees with what the
engine derives for itself** — the membind is not overriding the engine.

**So why membind at all?** Defence in depth over everything that does *not* go through that
allocator and today runs with the HBM nodes allowed: the glibc heap, ~480 thread stacks,
Rust/jemalloc arenas, the JNI/JVM side, and any `mmap` the CN or its dependencies take. None of
those are NUMA-aware. It is a cheap mechanical interlock — **not** a tuning knob, and **not** a fix
for an observed leak into HBM.

- **Only `--membind=0` and `--membind=1` are ever valid.**
- Never `--interleave=all` — it spreads the CN heap across all four HBM nodes.
- Never `--preferred=N` — it is *soft* and leaves the HBM nodes in `Mems_allowed_list`, i.e. it
  does not exclude the exact thing you are trying to exclude. Only a hard `--membind` works.

`cluster4-numa.sh` enforces this mechanically. Rather than hardcoding a forbidden list (which
would rot), it asserts the positive property that matters — **a valid membind target is a node
that has CPUs** — and refuses to start otherwise. Every HBM node fails that test by construction:

```
$ numactl --hardware | grep -E '^node (0|1|2|10) cpus:'
node 0 cpus: 0 1 2 3 ... 71
node 1 cpus: 72 73 74 ... 143
node 2 cpus:                      <-- empty: HBM, launcher dies here
node 10 cpus:                     <-- empty: HBM, launcher dies here
```

The launcher additionally verifies:

- each CN's `--physcpubind` list is a **subset** of its `--membind` node's CPUs, so cpubind and
  membind can never silently disagree (which would put a CN's threads on one socket and all its
  memory on the other); and
- that pair is the socket the **GPU is actually attached to**, read from the `CPU Affinity` column
  of `nvidia-smi topo -m`. Without this, a transposed `CN_NODE="1 1 0 0"` with correspondingly
  transposed `CN_CPUS` passes every other check and silently runs all four CNs on the socket
  *farthest* from their GPU. A parse failure here only warns — a topology parser that refuses to
  launch the cluster would be worse than the gap it closes — but a parsed mismatch is fatal.

### The FE is membound too

The FE is **not** cpu-pinned (see §3), but it *is* launched under
`numactl --membind=<every CPU-bearing node>` with **no** `--physcpubind`. That keeps its float
across both sockets completely intact while excluding the HBM nodes. A bare FE would sit at
`Mems_allowed_list: 0-2,10,18,26` — a ~10–20 GiB JVM able to allocate into GPU0's 27.24 GiB of
headroom, i.e. exactly the exposure the CN membind exists to close. The node list is derived from
the hardware (`numactl --hardware`, nodes with `NF > 3` CPUs), not hardcoded as `0,1`.

**Verify after launch:**

```bash
grep Mems_allowed_list /proc/<cn-pid>/status     # must read "0" or "1"
grep Mems_allowed_list /proc/<fe-pid>/status     # must read the CPU-node list, e.g. "0-1"
                                                 # neither may read "0-2,10,18,26"
```

---

## 3. The map: CN → GPU → cpubind → membind → ports

Default (`CPU_SPLIT=disjoint`), port ladder `base = PORT_BASE + i*PORT_STRIDE` = `9100 + i*10`:

| CN | GPU | cpubind | membind | heartbeat | thrift | brpc | http | starlet |
|---|---|---|---|---|---|---|---|---|
| CN0 | 0 | `0-35` | **0** | 9100 | 9101 | 9102 | 9103 | 9104 |
| CN1 | 1 | `36-71` | **0** | 9110 | 9111 | 9112 | 9113 | 9114 |
| CN2 | 2 | `72-107` | **1** | 9120 | 9121 | 9122 | 9123 | 9124 |
| CN3 | 3 | `108-143` | **1** | 9130 | 9131 | 9132 | 9133 | 9134 |

FE: 8030 (http), 9010 (edit log), 9020 (rpc), 9030 (query), **6090 (`cloud_native_meta_port`,
StarMgr)** — engine A's packaged FE conf sets `run_mode = shared_data`
(`starrocks/output/fe/conf/fe.conf:80`), so StarMgrServer binds 6090 too and the preflight checks
it. The FE is deliberately **not cpu-pinned** — it is the only process that can still float across
both sockets once all four CNs are hard-membound, and that float is what absorbs error in the
fixed-tenant budget. It *is* membound to the CPU-bearing nodes (§2).

The membind column is just the GPU's own CPU-affinity column from `nvidia-smi topo -m`: GPU0/GPU1
are on socket 0, GPU2/GPU3 on socket 1.

**Why contiguous 10-port blocks.** All CNs advertise `127.0.0.1` (there is no `--advertise-host`
knob in the launcher). The FE keys a node by `(advertise_host, heartbeat_port)`, and the nixl agent
is named `{advertise_host}:{brpc_port}`. The port block is therefore the *only* uniqueness lever
between CNs — an overlap is an **identity** collision in two registries, not merely a bind failure.
The blocks are clear of the FE's ports and of the CN compiled-in defaults
(9050/9060/8040/8060/9070). 5 of each 10 ports are used; 5 are spare.

### `CPU_SPLIT` presets

| preset | cpubind | Σ cores the FE is told about | notes |
|---|---|---|---|
| `socket` *(default)* | `0-71 / 0-71 / 72-143 / 72-143` | 288 (2× over) | what the 2026-08-09 GB200 audit (M1) specified; two CNs per socket may borrow each other's idle cores |
| `disjoint` | `0-35 / 36-71 / 72-107 / 108-143` | **144 (correct)** | see below |
| `none` | — | 576 (4× over) | no numactl; reproduces the exactly-validated unpinned run. Use `HOST_MEM=200GiB` with it. |

`experimental/starrocks/src/lib.rs:357` sends `std::thread::available_parallelism()` to the FE as
`num_hardware_cores`, and Rust's `available_parallelism()` honours `sched_getaffinity` — so
**`--physcpubind` directly changes the FE's planned parallelism.** Unpinned, four CNs each report
144 and the FE believes this box has 576 cores. Only `disjoint` sums to the true 144.

Over-reporting inflates fragment-instance count → more, *smaller* batches → more exchange RPCs, and
the single blocking transport thread per CN is already the dominant cost at scale, so over-reporting
pushes on exactly the wrong lever. CPU is not the bottleneck here (parquet decode happens on-GPU in
cuDF; CN RSS at rest is 1.0–1.5 GiB across ~480 mostly-blocked threads), so a 36-core mask does not
throttle scan fan-out. The counter-argument for `socket` — "CN0 borrows CN1's idle cores" — is weak,
because all four CNs execute the same pipeline stage of the same fragment concurrently, so their CPU
demand peaks are correlated, not anti-correlated.

`socket` is kept as the default because it is what M1 specifies and what is closest to the validated
run. **Try `disjoint` when you want the FE's parallelism model to be physically truthful.**

Fallback rule: if `vmstat 1` shows a sustained run-queue > 36 per socket, prefer `socket` **and**
pin `SET GLOBAL pipeline_dop` explicitly so the core-count over-report stops mattering.

---

## 4. Memory arithmetic

### Measured constants

| quantity | value | how |
|---|---|---|
| GPU nameplate | 189,471 MiB | `nvidia-smi --query-gpu=memory.total` |
| **GPU usable** | **188,416 MiB = 184.00 GiB** | `used+free`; also the node 2/10/18/26 size |
| driver/ECC reserve | 1,055 MiB | nameplate − usable; **not available to you** |
| CUDA ctx + cuDF/RMM | **779 MiB** | `device_used − pool − arena`, confirmed on two CN generations 2 min apart |
| CPU-addressable LPDDR | **956.82 GiB** | node 0 + node 1 |
| `mmfsd` (GPFS daemon) | **50.68 GiB** permanent RSS | `/proc/<pid>/status` |
| FE JVM | `-Xmx8192m`, ~10 GiB with metaspace/threads/direct | `ps` |
| CN host RSS at rest | **1.0–1.5 GiB** against a 200 GiB limit | `/proc/<pid>/status` |
| swap | **0** | `free` |
| SF100 dataset | **26 GiB** | `du -sh` |

### GPU, per CN (SF100 defaults)

```
  usable device                     188,416 MiB   184.00 GiB
- RMM pool          (GPU_MEM)       143,360 MiB   140.00 GiB   pre-reserved at startup
- staging arena     (STAGING)        16,384 MiB    16.00 GiB   OUTSIDE the pool
- CUDA ctx + cuDF/RMM                   779 MiB     0.76 GiB   measured
  ---------------------------------------------------------
  occupied                          160,523 MiB   156.76 GiB
  headroom                           27,893 MiB    27.24 GiB   = 14.8% of usable
```

**The staging arena does not come out of `--gpu-memory-limit`.** It is a bare `cudaMalloc`
(`src/exec/exchange_staging_arena.cpp:44`), not pool memory, so RMM knows nothing about it. Real
per-GPU footprint is `GPU_MEM + STAGING + 779 MiB` — **always budget the sum, never `GPU_MEM`
alone.** It has to stay a plain `cudaMalloc` by contract; pool memory silently loses the `cuda_ipc`
fast path.

The 27 GiB of headroom is not slack to spend: everything outside RMM's view grows under load
(per-kernel local-memory backing across 148 SMs, nvcomp scratch, `cudaIpcOpenMemHandle` page tables
for 3 peers' arenas). **Do not raise `GPU_MEM + STAGING` above 159,744 MiB.**

**Footprint-neutral rebalance available** (commented out in `engine-a.env`, not yet validated):
`132 GiB + 24 GiB = 159,744 MiB` — *identical* total, *identical* 27,893 MiB headroom. It moves
8 GiB from the over-provisioned side to the tight side: the pool is ~7× over-provisioned at SF100
(measured per-CN peaks 15–20 GiB), while the arena has only 1.43× margin (10–12 GB/CN cumulative
demand per query epoch, because the allocator is a monotonic bump arena that only resets at
quiescence). The arena fails **hard** — `"exchange staging arena exhausted … raise
SIRIUS_EXCHANGE_STAGING_BYTES"` — it does not degrade. This is the leading hypothesis for the q05
wedge.

### Host — and the M1 item 3 conflict, resolved

**The conflict.** M1 item 3 asks for 160–200 GiB/CN *and* for ≥400 GB of LPDDR left for page cache.
At 4 CNs those are stated as scale-independent constants, and at 200 GiB/CN they are arithmetically
impossible:

```
4 × 200 GiB = 800 + 72 (fixed tenants) = 872 of 956.82 GiB  ->  only 84.8 GiB left for cache.
```

**The resolution: page-cache demand is a function of dataset size, so the host limit is the
residual, not a constant.** The ≥400 GB ask is *vacuous* at SF100 (the dataset is 26 GiB — measured)
and *binding* at SF1000 (~363 GiB). M1's flat "160–200 for both scales" is an SF100 number stated
scale-independently, and it is wrong at SF1000.

```
HOST_MEM = (956.82 − 72 fixed − 1.05 × dataset − 32 margin) / NUM_CNS
```

Fixed tenants ≈ 72 GiB (mmfsd 50.68 + FE JVM ~10 + kernel/slab/daemons ~10), all measured.
The 32 GiB margin is not padding — swap is 0, so an over-commit is an OOM-kill mid-sweep.

| | **SF100 → `HOST_MEM=160GiB`** | **SF1000 → `HOST_MEM=120GiB`** |
|---|---|---|
| global | 4×160 = 640; +72 = 712; **244.82 GiB free** | 4×120 = 480; +72 = 552; **404.82 GiB free** |
| page cache left | 244.82 GiB vs a 26 GiB dataset ≈ **9×** | 372.82 GiB = **400.3 GB(dec) ≥ 400 GB** ✓ and ≥ the full 363 GiB dataset, so a warm pass caches all of it |
| per socket (hard membind) | 2×160 = 320; +36 = 356; **122.34 GiB free/node** | 2×120 = 240; +36 = 276; **202.34 GiB free/node** (dataset/node 181.5) |

**Why 160 and not the validated 200:** because this launcher adds a hard `--membind`, which the
validated run did not have. Under a hard membind the binding constraint becomes **per-node**, and a
CN exceeding its node's 478.34 GiB is OOM-killed rather than falling back to the other socket. The
per-node ceiling is `(478.34 − 36 − 20 − 16)/2 = 203.2 GiB`, so 200 GiB/CN would leave **3.2 GiB of
slack against a hard cap with zero swap**. 200 GiB is safe today *only* because the live CNs are
unpinned and can spill across sockets.

Giving up the 40 GiB costs nothing: the host limit is a lazily-grown **ceiling**, not a reservation
— measured CN host RSS at rest is 1.0–1.5 GiB against a 200 GiB limit, and host spillability is not
implemented yet, so the host pool is barely used. 160 GiB remains ~8–10× any plausible SF100 demand.

At SF1000 the host limit yields and the cache does not, for the same reason: the limit is an unused
ceiling, while page cache is the difference between a cold and a warm scan of a 363 GiB dataset — a
first-order effect on the headline number.

> **Units note.** `--host-memory-limit` lands on `memory.host.capacity_bytes`, which the config
> schema documents as *per NUMA node*. On this box that reading does not bite — the CN log says
> `1 host memory space(s) created for 34 NUMA node(s)` — so `HOST_MEM` is a **per-CN total**.
> Re-check that log line if the engine's host-space policy ever changes.

### Summary

| knob | SF100 (default) | SF1000 (commented in `engine-a.env`) |
|---|---|---|
| `GPU_MEM` | `140GiB` (`132GiB` rebalance available) | `128GiB` |
| `STAGING` | `16GiB` (`24GiB` rebalance available) | `32GiB` — **requires eager per-lease arena reclaim (M2 phase 2)** |
| `HOST_MEM` | `160GiB` | `120GiB` |
| `SIRIUS_QUERY_WATCHDOG_SECS` | `0` (recommend `120` for unattended sweeps) | `300` |
| GPU headroom | 27.24 GiB (14.8%) | 22.0 GiB (12.0%) |
| page cache left | 244.8 GiB (≈9× dataset) | 372.8 GiB = 400.3 GB ✓ |

**SF1000 honest caveat:** you cannot buy SF1000 with these knobs. Projected per-CN peaks put q21 at
100–150 GB and SF1000 arena demand at 105–125 GB/CN under the current bump-reset policy — no fixed
slab coexists with that. Eager per-lease arena reclaim is a hard prerequisite, plus the operator-side
fixes to bring q17/q18/q21 under a 128 GiB pool.

---

## 5. How to launch

```bash
cd /home/prestouser/aocsa/sirius/experimental/starrocks

./configs/gb200-4gpu/cluster4-numa.sh                      # SF100 defaults
CPU_SPLIT=disjoint ./configs/gb200-4gpu/cluster4-numa.sh    # truthful core accounting
SIRIUS_QUERY_WATCHDOG_SECS=120 ./configs/gb200-4gpu/cluster4-numa.sh   # unattended sweep

# SF1000
GPU_MEM=128GiB STAGING=32GiB HOST_MEM=120GiB SIRIUS_QUERY_WATCHDOG_SECS=300 \
  ./configs/gb200-4gpu/cluster4-numa.sh
```

Run it in its own terminal or as its own background task — **never** chained behind `&` inside
another shell command, or the cluster dies with that shell. Any single child exiting triggers the
cleanup trap and tears the whole cluster down, rather than leaving a half-cluster that the benchmark
would silently measure.

### Preflight — it refuses to start rather than starting degraded

| check | why |
|---|---|
| CN binary present (`target/release/sirius-starrocks-cn`) | `pixi run cn-build` |
| packaged FE present (`starrocks/output/fe/bin/start_fe.sh`) | `pixi run fe-check` |
| `numactl` present | required unless `CPU_SPLIT=none` |
| ≥ `NUM_CNS` GPUs visible | via `nvidia-smi` |
| **every membind node has CPUs** | the HBM interlock (§2) |
| **cpubind ⊆ membind node's CPUs** | otherwise threads and memory land on different sockets |
| **no required port already listening** | FE 8030/9010/9020/9030 + all 4 CN blocks |
| no GPU already has compute processes | override with `ALLOW_SHARED_GPUS=1` |

The port scan reads `/proc/net/tcp{,6}` directly (`st == 0A` is `TCP_LISTEN`) rather than shelling
out to `ss`: no iproute2 dependency, no output-format drift, and it sees sockets owned by every user
— which matters, since the process you are trying not to collide with may not be yours. It is a pure
read and never binds anything.

The GPU-claimed check exists because the CN's own `ensure_gpu_unclaimed` preflight is **skipped**
whenever `--gpu-memory-limit` is set, which it always is here. Since the RMM pool is reserved in
full at startup, sharing a GPU is not a slowdown — it is an allocation failure or a zero-headroom
cluster.

`cluster4-numa.sh` also **unsets any inherited `CUDA_VISIBLE_DEVICES`**: an already-exported value
wins over `--gpu-device` (the CN only logs a warning), which would silently collapse all four CNs
onto one GPU.

### ⚠️ Engine A and engine B cannot run at the same time

Stock StarRocks' FE uses the same ports (9030/9020/8030/9010). The preflight will refuse — that is
the intended behaviour, not a bug. Shut one down before starting the other. The A/B comparison is
**sequential**, which is also why the ≥400 GB page-cache constraint only has to hold per engine,
one engine at a time.

---

## 6. Known-good baseline (measured)

A 4-CN cluster ran with `NUM_CNS=4 GPU_MEM=140GiB HOST_MEM=200GiB STAGING=16GiB`,
`UCX_TLS=cuda_copy,cuda_ipc,tcp,self`, **unpinned** (no numactl), all 4 CNs `Alive=true`.

**Transport — nixl/NVLink confirmed.** The first-contact bandwidth canary measured **322–399 GB/s
per peer** against a 2.0 GB/s floor. That is `cuda_ipc` over the NV18 mesh. Below the floor the tier
is *refused*, not degraded, so a misconfigured `UCX_TLS` fails the cluster loudly at bring-up rather
than silently taking a ~200× slower host bounce.

**TPC-H SF100:**

| query | result |
|---|---|
| q01 | 6.1 s ✅ |
| q02 | 2.4 s ✅ |
| q03 | 6.3 s ✅ |
| q04 | 3.7 s ✅ |
| q06 | 5.3 s ✅ |
| q07 | 8.9 s ✅ |
| q05 | ❌ wedges at the 180 s harness timeout |
| q09 | ❌ wedges at the 180 s harness timeout |
| q08 | ❌ refused at **60758 ms** |

q08's 60758 ms is the hardcoded **60 s CN↔CN `REPLY_TIMEOUT`**
(`experimental/starrocks/src/prpc_client.rs:25`), applied as connect/read/write timeout on the
pRPC socket. **There is no flag and no env var for it** — a config set cannot move it; only a code
change can. Related hardcoded constants a config must design around: `XFER_TIMEOUT = 30s` per posted
nixl WRITE, and the canary's `CANARY_FLOOR_GBPS = 2.0`.

### Deviations of this config set from that baseline, and why

| | baseline | here | why |
|---|---|---|---|
| numactl | none | `--physcpubind` + `--membind` | HBM nodes are in the unpinned CNs' `Mems_allowed_list` (§2) |
| `HOST_MEM` | `200GiB` | `160GiB` | **forced by the membind above** — the per-node ceiling is 203.2 GiB and swap is 0 (§4) |
| `GPU_MEM` / `STAGING` | `140/16` | `140/16` | unchanged — validated values kept; the `132/24` rebalance is available but commented out |
| watchdog | unset (0) | `0` | unchanged — 0 is the compiled default and the validated behaviour |

The guiding rule: **reproduce the validated baseline exactly, and deviate only where the new NUMA
pinning forces it.** `HOST_MEM` is the one forced change. To reproduce the baseline bit-for-bit, use
`CPU_SPLIT=none HOST_MEM=200GiB` — and accept that the HBM nodes stay in `Mems_allowed_list`.

### Open items this config set does not fix

- **q05 / q09 wedge.** Not a config problem as far as the arithmetic goes, though the `132/24`
  arena rebalance is the leading hypothesis for q05. `SIRIUS_QUERY_WATCHDOG_SECS=120` converts a
  wedge into a clean query failure so one wedge stops poisoning the rest of a sweep — it does not
  fix the wedge.
- **q08 refusal.** Hardcoded 60 s `REPLY_TIMEOUT`; needs a code change.
- **SF1000.** Needs eager per-lease arena reclaim before the memory shape closes at all (§4).
