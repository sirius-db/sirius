# Runbook — Sirius TPC-H on a 4×H100 NVLink box

Step-by-step bring-up of the Sirius GPU compute-node cluster on a fresh multi-GPU machine, and
how to run the TPC-H A-vs-B benchmark on it.

Nothing in the setup is tied to a machine or path: the repo clones anywhere, every
machine-specific variable is derived at runtime by `scripts/cn-env.sh` (§3.2), the launcher
defaults to one CN per visible GPU (§6.2), and the StarRocks proto patches re-apply themselves
(§4). The absolute paths and counts below describe *this* box; treat them as worked examples,
not requirements.

Target box, as measured on the host this runbook was written against:

| | |
|---|---|
| GPU | 4× NVIDIA H100 80 GB HBM3 (81559 MiB each, 320 GB total) |
| Compute capability | 9.0 (Hopper GH100) |
| Driver / toolkit | 570.148.08 / CUDA 12.8 (`nvcc` 12.8.93) |
| vCPU / RAM | 104 / 885 GB |
| Disk | 11 TB on `/dev/vda1` (mounted `/`) |
| Interconnect | NVLink 4 — `NV18` between **every** pair, 18 links × 26.562 GB/s ≈ 478 GB/s per direction |
| NUMA | single node; all four GPUs on CPU affinity `0-103`, NUMA node 0 |
| NIC | `mlx5_0` (ConnectX, RoCE — `link_layer: Ethernet`, `PORT_ACTIVE`), `PHB` to every GPU |

Because all four GPUs are directly NVLink-connected peers on one host, the exchange tier runs
entirely over `cuda_ipc`; the `mlx5_0` NIC is not on the intra-node data path.

Paths used throughout — export these first:

```bash
export REPO=/home/ubuntu/sirius     # engine + CN source tree
export TOOLS=/home/ubuntu/tools     # ucx-install, nvda_nixl, gds-install (already present)
export NUM_CNS=4                    # one CN per GPU on this box
```

`/usr/local/cuda` may point at a 13.x tree even though the host `nvcc` is 12.8 — set
`CUDA_HOME` explicitly before building UCX/nixl (§3).

**Provenance note.** The NIXL/UCX bring-up (§3), the NVLink verification (§8) and the
throughput numbers quoted there were run on this box. The engine configuration itself
descends from the 2-CN setup (`pixi run cluster2`) that produced the committed 20/22 TPC-H
result on a single 23 GiB L4; the per-GPU memory sizing in §12 is scaled from that, not
independently tuned on H100. Scaling past 4 CNs (the port plan in §6.1 runs to 8) has not been
exercised here.

---

## 1. Check the box

```bash
# GPUs present and their memory
nvidia-smi --query-gpu=index,name,memory.total,compute_cap,driver_version --format=csv

# NVLink topology — every GPU pair should read NV# (not PIX/PHB/SYS)
nvidia-smi topo -m

# NVLink links up
nvidia-smi nvlink --status

# Peer access between every pair must be OK, or cuda_ipc cannot use NVLink
nvidia-smi topo -p2p r
```

On this box `topo -m` reads `NV18` in every off-diagonal GPU cell:

```
        GPU0    GPU1    GPU2    GPU3    NIC0    CPU Affinity  NUMA Affinity
GPU0     X      NV18    NV18    NV18    PHB     0-103         0
GPU1    NV18     X      NV18    NV18    PHB     0-103         0
GPU2    NV18    NV18     X      NV18    PHB     0-103         0
GPU3    NV18    NV18    NV18     X      PHB     0-103         0
NIC0    PHB     PHB     PHB     PHB      X
```

`NV18` is a bonded set of 18 NVLinks. `nvidia-smi nvlink --status -i 0` should show links 0–17
each at 26.562 GB/s — 18 × 26.562 ≈ 478 GB/s per direction per GPU. `PHB` in the `NIC0` row is
expected and harmless: it describes the NIC's PCIe path to the CPU, not a GPU-to-GPU hop.
If you ever see `SYS` or `PHB` **between two GPUs**, that pair talks over PCIe/host and NVLink
is not available to it — the exchange tier still works, just far slower.

`nvidia-smi topo -p2p r` must print `OK` for every pair (it does here, all 4×4). `CNS`
(chipset not supported) means peer-to-peer is off and the whole point of §8 is moot.

**Driver vs engine CUDA version — read this before launching.** The engine is built against
**CUDA 13** (both `pixi.toml`s pin `__cuda = "13"`; the env carries `libcudart.so.13` and
`cuda13`-tagged `librmm`/`libcudf`), which normally requires an **r580+** driver. This box runs
**570.148.08**, which caps at CUDA 12.8. rmm's first device call therefore fails every CN with:

```
CUDA error at .../rmm/work/cpp/src/cuda_device.cpp:35: cudaErrorInsufficientDriver
```

The fix is CUDA **forward compatibility**: `/usr/local/cuda/compat/` ships a newer user-mode
driver (`libcuda.so.580.178.04`) designed to pair a CUDA 13 runtime with an older kernel
driver, and it is supported here because these are data-center GPUs. Putting that directory
**first** on `LD_LIBRARY_PATH` is sufficient — `script-box.sh` does this automatically and skips
it when the directory is absent (i.e. on a box already at r580+). Verified A/B on this box:
without it, `cudaErrorInsufficientDriver`; with it, `Sirius engine context created`.

Also confirm the driver's own version. This box: driver `570.148.08`, host `nvcc` `12.8.93`.

```bash
nvcc --version
nproc; free -g          # 104 vCPU, 885 GB — sizes --host-memory-limit in §12
df -h /                 # 11 TB; the SF100 + SF1000 parquet in §5 occupies ~291 GB of it
```

---

## 2. Clone the repo

Any path works — nothing derives from the clone location being `/home/ubuntu/sirius`:

```bash
git clone <repo-url> /home/ubuntu/sirius     # or /any/path/sirius
cd /home/ubuntu/sirius
git submodule update --init --recursive
```

Worktrees do **not** auto-initialize submodules; if you are on a worktree rather than a fresh
clone, run the `git submodule update` line explicitly from inside it.

Install `pixi`, which drives every build and test in this repo:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
exec $SHELL -l
pixi --version
```

Work from the Sirius checkout for the rest of this runbook:

```bash
export REPO=/home/ubuntu/sirius
export TOOLS=/home/ubuntu/tools
cd "$REPO"
```

`REPO` is the engine/CN source tree. `TOOLS` holds the out-of-tree UCX/nixl/GDS
installs (`ucx-install`, `nvda_nixl`, `gds-install`). They are siblings under
`/home/ubuntu/`, not nested inside the repo.

`REPO`/`TOOLS` are shorthand **for this document only** — the build reads neither. The tasks
find the repo from their own location, and find the tools through `TOOLS_DIR`, which defaults
to a `tools/` directory next to the repo root (the sibling layout above). Only a non-sibling
tools location needs anything exported: `export TOOLS_DIR=/that/path`.

---

## 3. Install NIXL and UCX

The compute node's cross-node exchange rides **nixl** (NVIDIA Inference Xfer Library) over a
**UCX** backend. On a same-host multi-GPU box, UCX selects its `cuda_ipc` transport for
GPU→GPU transfers, and on NVLink hardware `cuda_ipc` moves bytes over NVLink. That chain —
nixl → UCX → cuda_ipc → NVLink — is what §8 verifies.

Build **GDS** (cuFile) plugins too if you need GPUDirect Storage; the exchange path itself
only requires the UCX plugin.

Both libraries live under `$TOOLS` (`/home/ubuntu/tools/`), outside the source tree, and
are referenced by absolute path from `pixi.toml`.

> **On this box these are already built and installed — skip to §3.2 and just verify.**
> `$TOOLS` currently holds `ucx-install` (UCX 1.21.0, `--enable-mt`, rev `b6a9d47`),
> `nvda_nixl`, `gds-install`, and the `ucx-1.21.0/` + `nixl-src/` source trees
> they were built from. The two subsections below are for reproducing that on a fresh host.

### 3.1 Getting the libraries (fresh host only)

**Option A — copy a known-good install** (fastest, if you have one):

```bash
# brev shell sirius-multicn
rsync -av <devbox>:/home/ubuntu/tools/nvda_nixl/   "$TOOLS/nvda_nixl/"
rsync -av <devbox>:/home/ubuntu/tools/ucx-install/ "$TOOLS/ucx-install/"
rsync -av <devbox>:/home/ubuntu/tools/gds-install/ "$TOOLS/gds-install/"
```

**Option B — build from source:**

```bash
mkdir -p "$TOOLS" && cd "$TOOLS"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}

# Build deps (Ubuntu/Debian). Skip any already installed.
sudo apt-get update
sudo apt-get install -y \
  meson ninja-build pkg-config \
  libhwloc-dev libnuma-dev pybind11-dev \
  libcufile-dev

# Ubuntu's distro libfabric is often too old for nixl (needs >= 1.21, and fi_ext.h).
# If pkg-config reports libfabric < 1.21, remove it so meson skips that plugin:
#   sudo apt-get remove -y libfabric-dev libfabric-bin libfabric1

# UCX first — nixl's UCX plugin links it. --enable-mt is required (agent touched
# from a dedicated thread).
wget -nc https://github.com/openucx/ucx/releases/download/v1.21.0/ucx-1.21.0.tar.gz
tar xf ucx-1.21.0.tar.gz && cd ucx-1.21.0
./configure --prefix="$TOOLS/ucx-install" --with-cuda="$CUDA_HOME" --enable-mt
make -j$(nproc) install
cd "$TOOLS"

# Meson rejects include paths that pass through the source tree via `..`.
# Always feed realpath'd absolutes. Default gds_path is /usr/local/cuda, but
# libcufile-dev installs headers under /usr/include — stage a tiny prefix:
mkdir -p "$TOOLS/gds-install/include" "$TOOLS/gds-install/lib64"
ln -sfn /usr/include/cufile.h "$TOOLS/gds-install/include/cufile.h"
ln -sfn /usr/lib/x86_64-linux-gnu/libcufile.so "$TOOLS/gds-install/lib64/libcufile.so"
ln -sfn /usr/lib/x86_64-linux-gnu/libcufile.so.0 "$TOOLS/gds-install/lib64/libcufile.so.0"

git clone https://github.com/ai-dynamo/nixl nixl-src && cd nixl-src
rm -rf build
meson setup build \
  --prefix="$(realpath "$TOOLS/nvda_nixl")" \
  -Ducx_path="$(realpath "$TOOLS/ucx-install")" \
  -Dgds_path="$(realpath "$TOOLS/gds-install")"
meson compile -C build
meson install -C build
```

Confirm UCX + GDS plugins landed:

```bash
ls "$TOOLS/nvda_nixl/lib/x86_64-linux-gnu/plugins/"
# expect: libplugin_UCX.so, libplugin_GDS.so, libplugin_GDS_MT.so, ...
```

Do **not** enable GDS and GDS_MT together in one nixl agent — that combination is rejected
at runtime. Pick one.

### 3.2 The environment — derived, not written by hand

Nothing to author on a new box: `scripts/cn-env.sh` (in the `experimental/starrocks` tree)
derives every build-time and run-time variable from its own location in the repo plus one
input, `TOOLS_DIR` — defaulting to a `tools/` directory **next to the repo root**, exactly the
layout above. `pixi run cn-build` / `cn-test` / `cn-run`, the `cluster*` tasks and
`script-box.sh` all source it, so they work from any clone path with no environment file.
If your nixl/UCX installs live elsewhere, export `TOOLS_DIR` (or `NIXL_PREFIX` /
`NIXL_PLUGIN_DIR` individually — an already-exported value always wins).

It derives: `NIXL_PREFIX`, `NIXL_PLUGIN_DIR`, `LD_LIBRARY_PATH` (engine .so + repo pixi env
lib + nixl + UCX), `PKG_CONFIG_PATH`, `UCX_TLS`, `NIXL_NO_STUBS_FALLBACK=1`, and the
`nixl-sys` build environment (`LIBCLANG_PATH`/`BINDGEN_EXTRA_CLANG_ARGS` from the repo pixi
env, system `CC`/`CXX`/linker with conda's `CXXFLAGS` cleared — pixi's conda toolchain leaking
into the `nixl-sys` build is the usual cause of a `cn-build` failure).

Using it by hand (for a manual `cargo build`, an `ldd`, or launching a CN directly):
**source it, never execute it** — run as a child process it configures a shell that
immediately exits. It fails loudly instead of half-configuring: no nixl under `$TOOLS_DIR`
points here; no clang headers in the repo pixi env points at `pixi install`. Note it also
*unsets* `CXXFLAGS`/`CFLAGS`/`CPATH` (to keep conda out of the nixl-sys build), so if your
shell needs those for other work, source it in a subshell:
`( source scripts/cn-env.sh && <command> )`.

Verify — a missing UCX plugin is the single most common bring-up failure:

```bash
source "$REPO/experimental/starrocks/scripts/cn-env.sh"
ls "$NIXL_PLUGIN_DIR/"          # must contain libplugin_UCX.so
```

On this box that prints `libplugin_UCX.so`, `libplugin_GDS.so`, `libplugin_GDS_MT.so`,
`libplugin_POSIX.so`, and the two Prometheus telemetry exporters.

---

## 4. Build the engine and the compute node

```bash
cd $REPO
pixi shell
pixi run make                 # builds libsirius (long: CUDA + cudf)
```

Then the CN, which links both the engine and nixl:

```bash
cd $REPO/experimental/starrocks
pixi run cn-build
```

`cn-build` sources `scripts/cn-env.sh` (§3.2), which sets, among others:

- `NIXL_PREFIX` — where to find libnixl
- **`NIXL_NO_STUBS_FALLBACK=1`** — mandatory. Without it, a broken nixl link does not fail the
  build; `nixl-sys` silently compiles a dlopen stub and you discover the problem at runtime as
  an agent-creation error, or worse, as a mysteriously slow transport.
- `CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=/usr/bin/gcc` — the CN crate needs the system
  linker, not the pixi one.

Run `cn-build` from `$REPO/experimental/starrocks` (it depends on `engine-build`, so a cold
tree builds libsirius first and takes a while; an already-built tree finishes in seconds).

### 4.1 The StarRocks proto patches

The three Sirius-only exchange RPCs (`exchange_nixl_md`, `request_staging_lease`,
`transmit_packed`) live in `patches/*.patch`, applied onto the stock StarRocks submodule by
`scripts/apply-starrocks-patches.sh` — never committed into the submodule itself, so the
gitlink always points at an upstream commit. **Both builds consume the patched proto**: the
FE's Maven build generates Java stubs from it, and the CN's `build.rs` runs prost over the
same `gensrc/proto/internal_service.proto`. `cn-build`, `cn-test`, `cn-run` and `fe-check`
therefore all depend on the `apply-starrocks-patches` task; a clean
`git submodule update` reverts the submodule to stock and the next build heals it
automatically. The script is idempotent — it detects an already-applied patch via
`git apply --reverse --check` and skips it.

Consequence: `git status` in the superproject permanently shows the submodule as
`Subproject commit <hash>-dirty`. That is the applied patch sitting as an uncommitted
working-tree change, **by design** — do not "fix" it by committing inside the submodule.
`git add experimental/starrocks/starrocks` records only the commit hash, never the dirt.
To silence the marker locally: `git config diff.ignoreSubmodules dirty`.

Confirm the binary exists and is nixl-linked:

```bash
readelf -d target/release/sirius-starrocks-cn | grep -Ei 'nixl|sirius'
```

Verified on this box — the `NEEDED` entries are `libnixl.so`, `libnixl_build.so` and
`sirius.duckdb_extension`. If nixl is absent from that list the stub path was taken — recheck
`NIXL_NO_STUBS_FALLBACK` and rebuild. (`ldd` shows the same thing plus the resolved paths, but
needs `LD_LIBRARY_PATH` from `scripts/cn-env.sh` to resolve them rather than printing
"not found".)

The front end is shipped pre-packaged, so you do not need the multi-hour Maven build:

```bash
pixi run fe-check             # asserts starrocks/output/fe/bin/start_fe.sh exists
```

`fe-check` first re-applies the Sirius-only StarRocks patches (`patches/*.patch`, the nixl
exchange RPCs) onto the submodule; idempotent, so a clean checkout and an already-patched one
both pass. If the FE package is missing, `git submodule update --init --recursive
experimental/starrocks/starrocks` then `pixi run fe-build` (long) — or copy `starrocks/output/fe`
from a box that has it.

---

## 5. The TPC-H data

The benchmark reads external parquet through `FILES()` CTEs — there is no load step, and both
engines read the same files.

**On this box the data is already generated.** Two scale factors are present:

| Path | Scale | Size |
|---|---|---|
| `/home/ubuntu/tpch_parquet_sf100` | SF100 | 26 GB |
| `/home/ubuntu/tpch_parquet_sf1000` | SF1000 | 265 GB |

```bash
export DATA=/home/ubuntu/tpch_parquet_sf100     # first end-to-end pass
export DATA=/home/ubuntu/tpch_parquet_sf1000    # the real run, once the cluster is known good
```

Every later section reads `$DATA`, so export it once and the §7 smoke queries and the §9/§10
sweeps all follow.

Confirm the layout before the first sweep — `bench.sh` substitutes `$TPCH_DATA` into
`FILES()` paths shaped `<dir>/<table>/*.parquet`, so a flat dump of `lineitem.parquet` at the
top level will not resolve:

```bash
ls "$DATA"                      # expect one directory per table: lineitem/ orders/ part/ ...
ls "$DATA"/lineitem/*.parquet | head
du -sh "$DATA"
```

Multiple parquet files per table beats one giant file: the FE byte-range-splits large files
across backends, but a per-file split is cheaper and parallelizes scan setup across all 4 CNs.

### Which scale to run

SF100 first. It is large enough that per-query fixed overheads — fragment dispatch,
first-touch allocation, plan translation — no longer dominate, and small enough that a wedged
query costs you a minute rather than an hour. SF1 says nothing at all about a 4×H100 box.

SF1000 is the headline number but expect it to be the harder run. 265 GB of *compressed*
parquet against 4 × 40 GiB engine carve-outs (§12) means the big joins will not hold their
working set in GPU memory, so it leans hard on the spill/downgrade path rather than measuring
clean GPU execution. Raise `QUERY_TIMEOUT` well past the SF100 setting, and read §14 first —
with no `cancel_plan_fragment`, every wedge costs a full cluster restart.

### Regenerating on a fresh box

```bash
export DATA=/home/ubuntu/tpch_parquet_sf100
mkdir -p "$DATA"
duckdb <<EOF
INSTALL tpch; LOAD tpch;
CALL dbgen(sf=100);
EOF
```

then export each table to its own directory under `$DATA`. Budget roughly the sizes in the
table above; the root filesystem here is 11 TB, and the two datasets together occupy ~291 GB.

---

## 6. Launch the cluster: 1 FE + 4 CNs, one per GPU

### 6.1 The port plan

The `cluster2` task offsets its second CN's ports by `+2`. That stride does not generalize:
past a handful of CNs the heartbeat range (9050, 9052, …) collides with the thrift range
(9060, 9062, …). Give each CN a contiguous block of 10 ports instead, based at 9100 — clear of
the FE's ports (8030 http, 9010 edit-log, 9020 rpc, 9030 query) and of the CN defaults (9050
heartbeat, 9060 thrift, 8040 http, 8060 brpc, 9070 starlet):

| CN | GPU | base | heartbeat | thrift | brpc | http | starlet |
|----|-----|------|-----------|--------|------|------|---------|
| 0 | 0 | 9100 | 9100 | 9101 | 9102 | 9103 | 9104 |
| 1 | 1 | 9110 | 9110 | 9111 | 9112 | 9113 | 9114 |
| 2 | 2 | 9120 | 9120 | 9121 | 9122 | 9123 | 9124 |
| 3 | 3 | 9130 | 9130 | 9131 | 9132 | 9133 | 9134 |

The scheme keeps going at the same stride (9140, 9150, …) if the box ever has more GPUs; only
rows 0–3 are used here.

Two identities matter:

- The **FE** identifies a CN by `(advertise_host, heartbeat_port)`. Those must be unique.
- The **nixl agent** is named `{advertise_host}:{brpc_port}`. Those must be unique too, or two
  CNs will collide when exchanging agent metadata.

Both hold under this plan.

### 6.2 The launch script

Use `script-box.sh` in this directory (it is the `cluster2` pixi task generalized to a loop).
It launches **one CN per visible GPU** by default (4 here), with H100-80GB-sized memory
defaults — 40 GiB engine + 32 GiB staging per GPU (see §12 for why the arena is that large):

```bash
cd $REPO/experimental/starrocks
./benchmarks/nixl-nvlink/script-box.sh                 # one CN per GPU — the default
GPU_MEM=24GiB ./benchmarks/nixl-nvlink/script-box.sh   # smaller carve-out
NUM_CNS=2 ./benchmarks/nixl-nvlink/script-box.sh       # first 2 GPUs only
```

On a box with different GPUs, size `GPU_MEM` and `STAGING` so their sum plus ~3 GiB of CUDA
context fits the device, and `HOST_MEM` to roughly host RAM / CN count. The port plan (§6.1)
extends at the same +10 stride for however many GPUs are present.

Run it in its own terminal, or as its own background task. **Never chain it behind `&` inside
another shell command** — the cluster dies with that shell.

The CN **registers itself** with the FE at startup (`ALTER SYSTEM ADD COMPUTE NODE`, retried up
to `--registration-max-attempts`, default 120). You do not run `ALTER SYSTEM` by hand.

### 6.3 What the flags mean

```
--gpu-device <i>            CUDA ordinal; exported as CUDA_VISIBLE_DEVICES before engine
                            bring-up (an already-exported value wins, so don't set both)
--gpu-memory-limit 40GiB    engine memory carve-out on that device (of the 79.6 GiB an H100
                            80 GB reports)
--host-memory-limit 192GiB  engine host-memory capacity (885 GB / 4 CNs leaves room)
--engine-dir .cn<i>         derived config, logs, telemetry — must be unique per CN
--heartbeat-port / --thrift-port / --brpc-port / --http-port / --starlet-port
```

Environment, set once for all CNs:

```
NIXL_PLUGIN_DIR             derived by scripts/cn-env.sh from $TOOLS_DIR
LD_LIBRARY_PATH             engine .so + pixi env lib + nixl + UCX (also from cn-env.sh)
UCX_TLS=cuda_copy,cuda_ipc,tcp,self
SIRIUS_EXCHANGE_STAGING_BYTES=32GiB
```

`UCX_TLS` is load-bearing in both halves:

- **`cuda_copy`** — without it, UCX cannot detect that a pointer is VRAM and nixl memory
  registration fails outright with `NIXL_ERR_BACKEND`.
- **`cuda_ipc`** — the fast same-host GPU→GPU path. Without it the transfer still *succeeds
  with correct bytes*, just ~200× slower through a host bounce. See §8.

`--gpu-device` is how one-process-one-GPU isolation happens: the CN calls
`setenv("CUDA_VISIBLE_DEVICES", <i>)` before engine bring-up
(`src/engine.rs`, `configure_engine_environment`), so the CUDA runtime in that process sees
exactly one device and every allocation and kernel lands there. Because it is set *after*
exec, `/proc/<pid>/environ` will not show it — verify the mapping through `nvidia-smi`
instead (§7).

`SIRIUS_EXCHANGE_STAGING_BYTES` allocates each CN's exchange staging arena
(`src/exec/exchange_staging_arena.cpp`): a single region that send-side packed batches are
gathered into and receive-side transfers land in, so the transport registers one fixed region
instead of arbitrary engine buffers. It is deliberately plain `cudaMalloc`, never RMM
pool/stream-ordered memory — UCX's `cuda_ipc` cannot export `cudaMallocAsync` allocations and
silently degrades ~200× to host bounces (correct bytes, no error). That is why it sits
**outside** `--gpu-memory-limit`: a CN really occupies `gpu-memory-limit + staging + CUDA
context`, all reserved up front. Leases are bump-allocated and short-lived; exhaustion fails
loudly naming requested/free/capacity (see §12 for sizing). If the variable is unset there is
no arena and the CN refuses cross-fragment exchange.

---

## 7. Verify the cluster

```bash
mysql --host 127.0.0.1 --port 9030 --user root -e "SHOW COMPUTE NODES\G" \
  | grep -E 'IP|HeartbeatPort|Alive'
```

All 4 must show `Alive: true`. A CN that registered but is not alive usually means the
heartbeat port is wrong or the process died during engine bring-up — check `.cn<i>/` logs.

Confirm the one-process-one-GPU mapping and the expected footprint:

```bash
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
```

Expect exactly one `sirius-starrocks-cn` PID per GPU UUID (a PID appearing under two GPUs
means the device pinning failed), each holding `GPU_MEM + STAGING + ~0.6 GiB context` — e.g.
74286 MiB per CN under the old 64+8 GiB sizing on this box. The number is an up-front
reservation (RMM pool + `cudaMalloc` arena): it reads the same idle or mid-query, so a "full"
GPU in `nvidia-smi` is normal, not a leak.

Smoke test a single-fragment query, then a multi-fragment one (the second exercises the nixl
exchange path across CNs, which is what NVLink accelerates):

```bash
mysql --host 127.0.0.1 --port 9030 --user root <<EOF
WITH lineitem AS (SELECT * FROM FILES(
  "path"="file://$DATA/lineitem/*.parquet","format"="parquet"))
SELECT count(*) FROM lineitem;

WITH lineitem AS (SELECT * FROM FILES(
  "path"="file://$DATA/lineitem/*.parquet","format"="parquet"))
SELECT l_returnflag, l_linestatus, sum(l_quantity), avg(l_extendedprice), count(*)
FROM lineitem WHERE l_shipdate <= date '1998-09-02'
GROUP BY 1,2 ORDER BY 1,2;
EOF
```

The second query is the Q1 shape: a partial aggregation per CN, a hash fan-out over the
grouping keys, and a merge — i.e. it moves real data between CNs.

Stale registrations survive FE restarts. If `SHOW COMPUTE NODES` lists nodes that no longer
exist, drop them:

```sql
ALTER SYSTEM DROP COMPUTE NODE "127.0.0.1:9130";
```

---

## 9. Run the benchmark — engine A (Sirius)

```bash
cd $REPO/experimental/starrocks/benchmarks/tpch
: "${DATA:=/home/ubuntu/tpch_parquet_sf100}"    # §5; swap to _sf1000 for the headline run

TPCH_DATA=$DATA \
QUERY_TIMEOUT=120 \
MIN_BACKENDS=4 \
RESTART_CMD='pkill -f "[s]irius-starrocks-cn"; pkill -f "[S]tarRocksFE"; sleep 10;
  (cd '"$REPO"'/experimental/starrocks && nohup ./benchmarks/nixl-nvlink/script-box.sh >/tmp/cluster.log 2>&1 &)' \
  ./bench.sh /tmp/bench/A/timings.csv 3

```

`MIN_BACKENDS` and the launcher's `NUM_CNS` must agree — 4 here. If you cut the cluster down
(`NUM_CNS=2`), lower `MIN_BACKENDS` to match, or the sweep waits forever for a backend that
will never register.

Arguments and environment:

| | |
|---|---|
| `bench.sh <out_csv> [runs] [qNN…]` | 1 discarded warm-up + `runs` timed repetitions |
| `TPCH_DATA` | directory holding `<table>/*.parquet`; substituted into the `FILES()` paths |
| `QUERY_TIMEOUT` | per-run client timeout, seconds. 120 suits SF100; raise well past it for SF1000, where the big joins spill |
| `MIN_BACKENDS` | alive backends required before the sweep starts — match the CN count (**4** here), or a sweep begun while the cluster is still booting records phantom wedges |
| `RESTART_CMD` | full cluster restart after a wedge |
| `FE_PORT` | default 9030 |

`RESTART_CMD` is **mandatory for engine A.** The CN does not implement `cancel_plan_fragment`,
so a hung or mid-execution-failed query strands its fragments; the stranded fragments starve
the CNs and the FE then answers "No available backends" for everything after. Without the
restart, every measurement following the first failure is invalid.

Note the `[s]irius-starrocks-cn` bracket pattern — it stops `pkill` matching its own command
line and killing your shell. The CN binary is `sirius-starrocks-cn`, not
`starrocks-compute-node`.

To sweep a subset:

```bash
TPCH_DATA=$DATA ./bench.sh /tmp/bench/A/timings.csv 3 q01 q06 q14
```

### Expected result quality

The last committed sweep (2-CN L4, SF1) was **20/22 passing**, 17 of them within 0.25 % of the
DuckDB oracle. Known open items you may still hit:

- **q02** hangs hard — an engine-thread wedge with no abort path.
- **q15** returns an empty result on roughly 1 run in 4.
- Up to −0.40 % arithmetic deficit on q03/q10 rows in the multi-fragment
  `sum(x*(1-l_discount))` path; passes are counted within a 0.5 % band.

---

## 10. Run the baseline — engine B (stock StarRocks)

```bash
JAVA_HOME=/usr/lib/jvm/java-21-amazon-corretto ./setup-engine-b.sh
# the script prints the exact commands to start the FE + BEs and register them
TPCH_DATA=$DATA ./bench.sh /tmp/bench/B/timings.csv 3
```

Stock StarRocks is laid out as a shared-nothing FE plus **BEs** (`start_be.sh` +
`ALTER SYSTEM ADD BACKEND`), not CNs — CN mode fails `FILES()` with "No alive backends".
Engine B needs no `RESTART_CMD`; it cleans up after itself.

Point B at the **same** `$DATA` directory A used. Comparing an SF100 A run against an SF1000 B
run is the easiest way to produce a meaningless table.

**Run one engine at a time.** A and B share the FE port 9030, the backend port ranges, and the
host CPUs. Take A fully down before measuring B and vice versa, or both sets of numbers are
meaningless.

---

## 11. Compare

```bash
./analyze.py /tmp/bench/A/timings.csv /tmp/bench/B/timings.csv results.md tpch_a_vs_b.png
```

Emits a markdown table (median ms per query, geometric-mean speedup over the set both engines
completed) and a log-scale bar plot. Paste the table into `BENCHMARK-A-VS-B.md`.

---

## 12. Tuning for H100-80GB

The committed configuration targets a 23 GiB L4 shared by two CNs. On an H100 80 GB with one
CN per GPU the constraint is different — these are the launcher's defaults, and where to start:

| Knob | L4 (2 CNs/GPU) | H100 80 GB (1 CN/GPU) | Why |
|---|---|---|---|
| `--gpu-memory-limit` | 8GiB | 40GiB | leave headroom for the staging arena + CUDA context; the arena is *not* counted inside this limit |
| `SIRIUS_EXCHANGE_STAGING_BYTES` | 1280MiB | 32GiB | scale with the scale factor: 512 MiB starved TPC-H q09 at SF1 (one 648 MB packed-table lease), and 8 GiB starved q05 at SF100 (21 concurrent leases). Fewer CNs makes it worse — each carries more of the fan-out |
| `--host-memory-limit` | 12GiB | 192GiB | 885 GB / 4 CNs = 221 GB each; 192 GiB leaves the page cache room for the parquet reads |

Budget per GPU: `40 GiB limit + 32 GiB arena + ~2–3 GiB CUDA context/cudf overhead ≈ 75 GiB` of
the **79.6 GiB** an H100 80 GB actually reports (81559 MiB). That is a ~5 GiB margin — thinner
than it looks once cudf's own pools warm up. Raising `STAGING` therefore *requires* lowering
`GPU_MEM` by the same amount. If a CN dies at bring-up with an allocation failure, lower
`--gpu-memory-limit` first (`GPU_MEM=24GiB ./benchmarks/nixl-nvlink/script-box.sh`).

`--gpu-memory-fraction <f>` is available as an alternative to the absolute limit; it is a
fraction of **total** device memory, not free memory.

If q09-style queries fail with a staging-arena error, raise
`SIRIUS_EXCHANGE_STAGING_BYTES` — that is the arena, not the engine limit.

---

## 13. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Every CN dies with `cudaErrorInsufficientDriver` | CUDA-13 engine on an r570 driver. Put `/usr/local/cuda/compat` first on `LD_LIBRARY_PATH` (the launcher does this); see §1 |
| CN dies with `libstdc++.so.6: version GLIBCXX_3.4.31 not found` | The pixi env's `lib/` is missing from `LD_LIBRARY_PATH`. Only bites when the CN is started outside `pixi run`; the launcher adds it |
| `no packaged FE at .../output/fe/bin/start_fe.sh` | The FE was never packaged. `pixi run fe-build`, or drop in the FE from the matching stock StarRocks release (the submodule tag — 4.1.1 here). The Sirius patch only adds CN↔CN RPCs, so a stock FE of the same version is correct |
| Agent creation fails at CN startup | libnixl not found or the stub was linked. Check `TOOLS_DIR` points at the installs (`source scripts/cn-env.sh` to see the derived paths); rebuild with `NIXL_NO_STUBS_FALLBACK=1`; check `NIXL_PLUGIN_DIR` really contains the UCX plugin |
| nixl registration fails, `NIXL_ERR_BACKEND` | `UCX_TLS` is missing `cuda_copy` — UCX cannot detect VRAM pointers |
| Everything works but is ~200× slow | `UCX_TLS` is missing `cuda_ipc`, or the arena is not `cudaMalloc`-backed. The bandwidth canary should have refused the tier; see §8 |
| "No available backends" for every query | A wedged query stranded its fragments. Restart the cluster — this is why `RESTART_CMD` exists |
| `SHOW COMPUTE NODES` lists dead nodes | FE metadata persists across restarts: `ALTER SYSTEM DROP COMPUTE NODE "host:port"` |
| Sweep records wedges from the first query on | The sweep started before the cluster was up. Set `MIN_BACKENDS` to the CN count (4 here) |
| `pkill` killed your shell | Use the bracket pattern: `pkill -f '[s]irius-starrocks-cn'` |
| CN exits immediately, no log | Port collision. Check the §6.1 plan against `ss -ltnp` |
| `cargo fmt --check` fails on untouched files | Pre-existing CN files fail fmt. Format only the crates you touched; never run it workspace-wide |
| Submodule always shows `-dirty` in `git status`/diffs | The applied proto patch (§4.1) sitting as an uncommitted working-tree change — expected, by design. Hide it with `git config diff.ignoreSubmodules dirty`; never commit inside the submodule |
| Disk filling under `.cn*/telemetry/` | The CN writes one ndjson session dir per bring-up *and per query*, and never garbage-collects. `./benchmarks/nixl-nvlink/clean-telemetry.sh` removes them (`--dry-run` to preview, `--all` to ignore the keep-recent-sessions guard; guard is skipped automatically when no CN is running) |
| CN build fails with missing `transmit_packed`/`PTransmitPackedParams` | The submodule is stock — the proto patch was reverted (e.g. by `git submodule update`). `pixi run apply-starrocks-patches`, or just re-run `cn-build`, which depends on it |

---

## 14. Known limitations of engine A

- No `cancel_plan_fragment` — hung queries strand fragments; `RESTART_CMD` is mandatory.
  Partial mitigation: `SIRIUS_QUERY_WATCHDOG_SECS=<n>` turns an engine-side scheduling stall
  into a loud query failure after `n` seconds of zero progress (the `cluster`/`cluster2` tasks
  set 60). Opt-in, because a single kernel legitimately running longer than the threshold
  would be misread as a stall — leave it unset for SF1000-scale sweeps or raise it well past
  the longest expected operator.
- No cancel/GC path generally.
- `DISTINCT` aggregation is refused.
- q02 hangs (engine-thread wedge, needs an engine-side abort/watchdog).
- q15 flakes to an empty result ~1 run in 4.
- Merge functions that are not per-column reduces (stddev, median, distinct) are out of reach
  of the current two-phase aggregation design; the stream ABI also cannot spell LIST or STRUCT
  types.

---

## 15. Fairness caveats

Engine A executes on GPUs; engine B is a mature vectorized CPU engine. At small scale factors
fixed per-query overheads — fragment dispatch, first-touch allocation, plan translation —
matter as much as scan throughput, which is why the SF100 and SF1000 sets in §5 say far more
than SF1 about a 4×H100 box. The two scales also flatter different engines: SF100 largely fits
the GPU carve-outs and shows engine A at its best, while SF1000 pushes past them into the
spill path, so quote which one a number came from. Note also that engine B gets all 104 vCPUs
while engine A gets 4 GPUs plus whatever
CPU its scan threads use — the comparison is box-vs-box, not core-vs-core. Both engines read
the same parquet files through `FILES()` with no load step, and the FE byte-splits large files
across backends in both cases. Results are indicative, not a TPC-compliant benchmark.
