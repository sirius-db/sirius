# Hardware — `presto-gb200-gcn-17` (4× GB200)

Everything below was **probed directly on the box**, 2026-08-12. This is the substrate every
other file in this folder is tuned against. To re-target the folder, replace this file and follow
[`RETARGETING.md`](RETARGETING.md).

---

## The box

| | |
|---|---|
| Hostname | `presto-gb200-gcn-17` |
| GPUs | **4 × NVIDIA GB200**, 189,471 MiB = **185.03 GiB (198.7 GB)** HBM each · 740 GB aggregate |
| Driver | 580.105.08 |
| CPU | **aarch64 Neoverse-V2**, 144 cores |
| CPU NUMA | **2 nodes**: node0 = cpus `0-71` + 489,960 MB · node1 = cpus `72-143` + 489,823 MB |
| Total CPU RAM | 979,783 MB = **956.8 GiB** |
| NUMA nodes total | **34** (0–33). Nodes 2–33 are GPU HBM or empty — **zero CPUs** |
| Storage | `/raid` — ext4 on `/dev/md0`, local NVMe RAID0, **11.4 TB free** |

## Interconnect — `nvidia-smi topo -m`

```
      GPU0  GPU1  GPU2  GPU3   CPU Affinity  NUMA Affinity    GPU NUMA ID
GPU0   X    NV18  NV18  NV18   0-71          0,3-9,11-17      2
GPU1  NV18   X    NV18  NV18   0-71          0,3-9,11-17      10
GPU2  NV18  NV18   X    NV18   72-143        1,19-25,27-33    18
GPU3  NV18  NV18  NV18   X     72-143        1,19-25,27-33    26
```

All 8 NICs (`mlx5_0`–`mlx5_7`) are **`SYS`** from every GPU. `NIC2↔NIC3` and `NIC6↔NIC7` are `PIX`.

### The three facts that drive every decision here

**1. GPU↔GPU bandwidth is uniform.** Every pair is `NV18` — 18 bonded NVLinks, a full all-to-all
mesh. There is no "near" and "far" GPU. **Therefore any performance difference between GPU
selections is entirely host-side NUMA**, which makes this box unusually clean for isolating that
effect.

**2. GPUs split across the two CPU sockets, 2 and 2.**

| GPU | CPU affinity | CPU NUMA node | GPU HBM NUMA ID |
|---|---|---|---|
| GPU0 | `0-71` | **0** | 2 |
| GPU1 | `0-71` | **0** | 10 |
| GPU2 | `72-143` | **1** | 18 |
| GPU3 | `72-143` | **1** | 26 |

**3. The NICs are irrelevant to this folder.** All studies here are single-box; nixl moves data
GPU→GPU over NVLink via `cuda_ipc`. No NIC has locality to any GPU (`SYS` everywhere), so nothing
is lost by ignoring them. `UCX_TLS` must still include `cuda_ipc` — without it nixl falls off a
**1349× cliff**.

---

## CN → GPU → CPU → NUMA mapping

Derived from the affinity table above. **A CN must never own a GPU on the other socket.**

### 4 CNs — the full-box configuration

| CN | GPU | CPUs | membind |
|---|---|---|---|
| cn0 | 0 | `0-35` | 0 |
| cn1 | 1 | `36-71` | 0 |
| cn2 | 2 | `72-107` | 1 |
| cn3 | 3 | `108-143` | 1 |

36 cores and ~239 GiB per CN. This is the mapping the existing `cluster4-numa.sh` uses, and it is
correct for this topology.

### 2 CNs — use GPU0 + GPU2, **not** GPU0 + GPU1

| Option | CN0 | CN1 | Cores used | RAM reachable |
|---|---|---|---|---|
| **GPU0 + GPU2 ✅ default** | GPU0, cpus `0-71`, membind 0 | GPU2, cpus `72-143`, membind 1 | **144** | **958 GiB** |
| GPU0 + GPU1 ⚠️ variant | GPU0, cpus `0-35`, membind 0 | GPU1, cpus `36-71`, membind 0 | 72 | 479 GiB |

> **This is the non-obvious one.** The instinct is to pick adjacent GPUs on the same socket, but
> GPU0+GPU1 confines **both** CNs to socket 0 — half the cores, half the memory bandwidth, and
> socket 1 sitting idle. Because all pairs are `NV18`, choosing GPU0+GPU2 costs **nothing** in
> GPU↔GPU bandwidth while doubling host resources and keeping each GPU local to its own CN.
>
> Run GPU0+GPU1 only as a deliberate variant to measure the host-NUMA penalty in isolation — it
> is the cleanest such experiment this box supports, since the GPU fabric is held constant.

### 1 CN

| Option | GPU | CPUs | membind | Note |
|---|---|---|---|---|
| **Socket-local ✅ default** | 0 | `0-71` | 0 | Matches GPU0's affinity; comparable to one CN of the 2-CN arm |
| All-cores variant | 0 | `0-143` | `0,1` | More scan threads, but half of them are cross-socket from GPU0 |

Use socket-local as the scale-out baseline so that 1 → 2 → 4 CNs varies **only GPU count**.

---

## Pinned-tier feasibility

185.03 GiB (198.7 GB) per GPU, before working memory:

| GPUs | SF500 (132 GB) | SF1000 (283 GB) |
|---|---|---|
| 1 | 132 GB/GPU — **66%**, fits with 67 GB headroom | **impossible** (283 > 198.7) |
| 2 | 66 GB/GPU — 33%, comfortable | 141.5 GB/GPU — **71%, tight**, expect spill |
| 4 | 33 GB/GPU — 17%, very comfortable | 70.8 GB/GPU — 36%, comfortable |

**Consequence:** SF500 can be pinned at every topology, so the SF500 scale-out study can run
pinned end-to-end. SF1000 cannot be pinned at 1 GPU and is marginal at 2 — run **SF1000 scale-out
entirely cold** for internal consistency across arms.

---

## Memory budget arithmetic

**Per GPU.** The exchange staging arena is a bare `cudaMalloc` **outside** the RMM pool, so
`usage_limit_fraction` does not know about it:

```
185.03 GiB total
  −  16.00 GiB  SIRIUS_EXCHANGE_STAGING_BYTES (out-of-pool)
  −   ~1 GiB    CUDA context + fragmentation headroom
  = 168 GiB available to the pool  →  fraction ≤ 0.90
```

A 16 GiB arena is **8.6%** of a GB200 — tolerable, unlike on an 80 GB card where it would be 20%.
Use **0.86** for margin; raise toward 0.90 only after confirming the arena is accounted for.

**Host, across CNs.** Leave real page cache — the dataset is re-read every query unless pinned:

| CNs | host capacity per CN | total | page cache left | × SF500 dataset |
|---|---|---|---|---|
| 1 | 160 GiB | 160 GiB | ~797 GiB | 6.5× |
| 2 | 240 GiB | 480 GiB | ~477 GiB | 3.9× |
| 4 | 160 GiB | 640 GiB | ~317 GiB | 2.6× |

All three leave more page cache than the 132 GB SF500 dataset, so first-touch warms fully. At
SF1000 (283 GB) the 4-CN row leaves 317 GiB — still 1.1×, but tight. Drop to 128 GiB/CN at SF1000.

---

## Traps specific to this box

**`numactl --interleave=all` is harmful here.** It resolves to `{0,1,2,10,18,26}` — nodes 2/10/18/26
are **GPU HBM with zero CPUs**, so ~2/3 of host pages land inside the HBM of the GPUs you are
computing on. Measured **11.4% slower on 22/22 queries**. Use `--membind=<socket>` per CN, or
`--interleave=0,1` if interleaving is genuinely wanted.

**`/proc/<pid>/status` `Mems_allowed_list` is the wrong NUMA probe.** It reports the **cpuset**
restriction, which `numactl --membind` does not change — it will read `0-2,10,18,26` even when the
binding is correct, producing a false alarm every time. The authoritative probe:

```bash
grep -m1 mempolicy /proc/<pid>/numa_maps     # expect bind:0 or bind:1
grep -o 'N[0-9]*=' /proc/<pid>/numa_maps | sort -u   # expect only N0= or only N1=
```

**`/proc/meminfo` over-reports total memory.** It counts GPU HBM: `MemTotal` ≈ 1692 GB against
956.8 GiB of real LPDDR. **Any engine configured with a percentage memory limit will size itself
against the wrong number.** This box has `Swap: 0`, so that is an OOM-kill, not a slowdown. Always
use absolute byte limits.

**GPUs must be verified released between engines.** Clean floor is **~28 MiB** per GPU.

```bash
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
```

**Nightly CI takes all 4 GPUs ~02:00–03:50 UTC.** Do not measure in that window. (Documented for
the GB200 boxes; verify it applies to gcn-17 before scheduling an overnight sweep.)
