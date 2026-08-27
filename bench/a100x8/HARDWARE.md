# Hardware — 8× A100-SXM4-80GB (`massedcompute_A100_sxm4_80Gx8`)

**Source:** `nvidia-smi topo -m` + `nvidia-smi` supplied 2026-08-12, plus the instance spec on
record. **Not probed by me** — items marked 🔍 must be confirmed on first login.

To re-target this folder, see [`../common/RETARGETING.md`](../common/RETARGETING.md).

---

## The box

| | |
|---|---|
| Instance | `massedcompute_A100_sxm4_80Gx8` · **$13.25/hr** |
| GPUs | **8 × NVIDIA A100-SXM4-80GB**, 81,920 MiB = **80.0 GiB (85.9 GB)** each · **640 GiB (687 GB)** aggregate |
| Driver / CUDA | **570.148.08 / CUDA 12.8** |
| Power cap | 400 W per GPU · Persistence-M **On** · MIG **Disabled** |
| PCIe | `0000:06:00.0` … `0000:0D:00.0` — eight consecutive slots, one complex |
| CPU | **240 cores**, 2 NUMA nodes (`0-1`) |
| Host RAM | **1500 GB** (spec on record) |
| Architecture | 🔍 **near-certainly `x86_64`** — confirm. A100-SXM4 platforms are x86 |
| Storage | 🔍 probe — layout, capacity, and whether local NVMe or network |
| Dataset paths | 🔍 **must be staged** — SF500 (132 GB) and SF1000 (283 GB) are not on this box |

### Probe on arrival

```bash
hostname; uname -m
nvidia-smi topo -m
numactl -H                                    # RAM per node, confirm 2 nodes
lscpu | grep -E '^CPU\(s\)|Model name|Architecture|NUMA node[0-9]+ CPU'
free -g; swapon --show                        # Swap:0 makes % memory limits fatal
df -h; lsblk                                  # where to stage 415 GB of parquet
findmnt -T <data path>                        # local vs network — changes the I/O config
```

---

## Interconnect

```
      GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7  NIC0  CPU Affinity  NUMA Affinity  GPU NUMA ID
GPU0   X    NV12  NV12  NV12  NV12  NV12  NV12  NV12  PHB   0-239         0-1            N/A
 ...  (every pair NV12; every GPU identical)
NIC0  PHB   PHB   PHB   PHB   PHB   PHB   PHB   PHB    X
```

### Three facts that drive every decision here

**1. Uniform all-to-all NVLink — `NV12`.** Every GPU pair is 12 bonded NVLinks. As on the GB200
box, there is no near/far GPU. Note this is **NV12, not NV18**: fewer links and an older NVLink
generation, so cross-GPU exchange is materially cheaper on GB200 than here. Expect communication-
bound queries to scale *worse* on this box.

**2. No GPU→socket affinity, and no GPU HBM NUMA nodes.** Every GPU reports `CPU Affinity 0-239`,
`NUMA Affinity 0-1`, `GPU NUMA ID N/A`. Two consequences, one good and one bad:

- ✅ **The GB200 `--interleave=all` trap does not exist here.** On that box, nodes 2/10/18/26 are
  GPU HBM with zero CPUs, so interleaving dumped host pages into the HBM you were computing on —
  11.4% slower on 22/22 queries. Here `GPU NUMA ID` is `N/A`: there are no GPU memory nodes to fall
  into. **Still verify with `numactl -H`** before relying on it.
- ❌ **You cannot use GPU selection to control NUMA placement.** On the GB200 box GPU0/1 belong to
  socket 0 and GPU2/3 to socket 1, which makes the correct CN placement obvious. Here the platform
  expresses no preference, so **CN→socket assignment is a free choice you must make deliberately** —
  balance CNs across the two nodes rather than letting them pile onto node 0.

**3. One NIC (`mlx5_0`), `PHB` to every GPU.** Irrelevant for single-box studies — nixl moves data
GPU→GPU over NVLink via `cuda_ipc`. `UCX_TLS` must still include `cuda_ipc`; without it nixl falls
off a **1349× cliff**.

---

## CN → GPU → CPU → NUMA mapping

240 cores over 2 nodes = **120 cores per node**. Since no GPU has a socket preference, the goal is
simply to **balance CNs across both nodes** and give each a contiguous core range.

### 8 CNs — full box

| CN | GPU | CPUs | membind |
|---|---|---|---|
| cn0–cn3 | 0,1,2,3 | `0-29`, `30-59`, `60-89`, `90-119` | 0 |
| cn4–cn7 | 4,5,6,7 | `120-149`, `150-179`, `180-209`, `210-239` | 1 |

**30 cores and ~87 GiB host per CN.**

### 4 CNs

| CN | GPU | CPUs | membind |
|---|---|---|---|
| cn0, cn1 | 0, 1 | `0-59`, `60-119` | 0 |
| cn2, cn3 | 4, 5 | `120-179`, `180-239` | 1 |

### 2 CNs

| CN | GPU | CPUs | membind |
|---|---|---|---|
| cn0 | 0 | `0-119` | 0 |
| cn1 | 4 | `120-239` | 1 |

### 1 CN — baseline

| CN | GPU | CPUs | membind |
|---|---|---|---|
| cn0 | 0 | `0-119` | 0 |

> Unlike the GB200 box, the 2-CN GPU pairing carries **no structural NUMA consequence** — all pairs
> are `NV12` and no GPU prefers a socket. Pick GPU0+GPU4 for symmetry with the CN→node split. The
> GB200 folder's "GPU0+GPU2 vs GPU0+GPU1" experiment **has no analogue here**.

---

## Pinned-tier feasibility

80.0 GiB (85.9 GB) per GPU, **before** working memory:

| GPUs | SF500 (132 GB) | SF1000 (283 GB) |
|---|---|---|
| 1 | 132 GB — **impossible** | **impossible** |
| 2 | 66.0 GB/GPU — **77%, too tight** | **impossible** |
| 4 | 33.0 GB/GPU — 38%, OK | 70.8 GB/GPU — **82%, too tight** |
| 8 | 16.5 GB/GPU — 19%, comfortable | 35.4 GB/GPU — 41%, OK |

**Consequence:** pinning is far more constrained than on the GB200 box (which pins SF500 at every
topology). Here:

- **SF500** — pin only at **4 and 8** GPUs. Run 1 and 2 cold.
- **SF1000** — pin only at **8**. Run everything else cold.
- **Scale-out studies should therefore run cold on this box** for internal consistency across arms.
  A curve whose arms use different regimes is not a scaling curve.

---

## Memory budget arithmetic

### Per GPU — the binding constraint on this box

The exchange staging arena is a bare `cudaMalloc` **outside** the RMM pool, so
`usage_limit_fraction` does not account for it. At 80 GiB it is no longer a rounding error:

```
 80.0 GiB total
 −  8.0 GiB   SIRIUS_EXCHANGE_STAGING_BYTES   (10% — see the tension below)
 −  ~1.0 GiB  CUDA context + fragmentation
 = 71.0 GiB   available  →  usage_limit_fraction ≤ 0.88
```

**Use `0.85` (68.0 GiB pool)**, leaving ~3 GiB of real headroom.

> **The tension, stated plainly.** A 16 GiB arena — the GB200 default — is **20% of an A100**. But
> shrinking it hurts: q17/q21 staged 1.9–6 GB/CN at SF100 on 4 CNs; at SF500 on 8 CNs that is
> ~4.75–15 GB/CN, which **exceeds an 8 GiB arena at the top end**. There is no setting that is
> comfortable for both. 8 GiB is the compromise; **expect q17 and q21 to be more likely to fail
> here than on the GB200 box**, and treat that as a measurement, not a surprise.

For comparison, the GB200 box gets a **159 GiB pool** against this box's **68 GiB** — 2.3× more per
GPU. Eight A100s have more aggregate VRAM (640 vs 740 GiB → actually *less*), but far less per-GPU
working room, which is what single-operator peaks care about.

### Host, across CNs

1500 GB = ~1397 GiB. Leave real page cache — data is re-read every query unless pinned.

| CNs | per CN | total | page cache left | × SF500 | × SF1000 |
|---|---|---|---|---|---|
| 8 (SF500) | 120 GiB | 960 GiB | ~437 GiB | 3.6× | 1.7× |
| 8 (SF1000) | **100 GiB** | 800 GiB | ~597 GiB | 4.9× | **2.3×** |
| 4 | 200 GiB | 800 GiB | ~597 GiB | 4.9× | 2.3× |
| 2 | 300 GiB | 600 GiB | ~797 GiB | 6.5× | 3.0× |

---

## Differences from the GB200 box, at a glance

| | GB200 ×4 (`gcn-17`) | **A100 ×8 (this box)** |
|---|---|---|
| GPUs | 4 | **8** |
| HBM/GPU | 185.03 GiB | **80.0 GiB** (43%) |
| Aggregate HBM | 740 GiB | 640 GiB |
| Usable pool/GPU | ~159 GiB | **~68 GiB** |
| NVLink | NV18 | **NV12** (slower) |
| Cores | 144 aarch64 Neoverse-V2 | **240**, 🔍 x86_64 |
| GPU→socket affinity | **Yes** — GPU0/1→n0, GPU2/3→n1 | **None** — all GPUs `0-239` |
| GPU HBM NUMA nodes | **Yes** (2,10,18,26) — `interleave=all` is a trap | **None** (`N/A`) — trap absent |
| Driver / CUDA | 580.105.08 / 13.x | **570.148.08 / 12.8** |
| NICs | 8 (`SYS`) | 1 (`PHB`) |
| Data | **Local, staged** | 🔍 **must be staged** |
| Build | aarch64 | 🔍 **x86_64 rebuild required** |
| Price | on-prem, no $/hr | **$13.25/hr** — real cost study possible |

> **Two consequences worth planning around.** (1) Sirius must be **rebuilt for x86_64**, against
> **CUDA 12.8** rather than the 13.x toolchain the GB200 build uses — check the pixi environment
> resolves for that combination before booking box time. (2) This box has a **real hourly price**,
> so unlike the on-prem GB200 it supports a genuine measured cost-per-run study rather than a
> modeled one.

## Hygiene

```bash
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader   # clean floor ≈ 0 MiB here
```

All 8 GPUs read **0 MiB used** in the supplied snapshot — the box was idle. Verify this before
every engine switch, and confirm no stray CN/FE/BE/ray processes survive a teardown.
