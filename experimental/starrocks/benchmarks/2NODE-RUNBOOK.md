# Runbook — two machines, one FE, CNs on both

How to run the Sirius compute-node cluster across two physical hosts so the FE schedules plan
fragments on GPUs in both, and what the cross-host data plane actually costs.

Target pair assumed throughout (a GB200 NVL72 rack, IMEX domain spanning 18 nodes):

| | |
|---|---|
| Hosts | `presto-gb200-gcn-17`, `presto-gb200-gcn-18` |
| GPU | 4× GB200 per host |
| Control LAN | `bond0` — 10.87.140.52 / 10.87.140.53, same /27, routed |
| Data fabric | 4× 400G RoCE (`enp3s0np0`, `enP2p3s0np0`, `enP16p3s0np0`, `enP18p3s0np0`), point-to-point /31s, routed host-to-host |
| Cross-host GPU path | NVLink via the rack IMEX domain, or GPUDirect RDMA over the RoCE NICs |

**Provenance.** The transport numbers in §2 are measured on this pair by
`scripts/nixl-echo-2node.sh` (`nixl_transport::nixl_echo::two_node_gpu_echo`), which sends a GPU
buffer host-to-host over nixl and back, byte-verifying both legs. The cluster bring-up in §4 is
composed from the CN's CLI surface — it is not yet wrapped in a launcher script, and the
committed launchers (`cluster8.sh`, `configs/gb200-4gpu/cluster4-numa.sh`) are single-host only.

---

## 1. Is multi-machine possible?

Yes, and it needs no StarRocks patches. The FE keys a compute node by
`(advertise_host, heartbeat_port)` and dispatches fragments to `advertise_host:brpc_port`; nothing
in that path is host-local. The single-machine setup is the special case: `--advertise-host`
defaults to `127.0.0.1`, so every CN so far has claimed to live on loopback, and none of the
launchers pass the flag.

Three things do have to change, and one of them is a real blocker:

1. The FE must stop advertising loopback (`priority_networks` in `conf/fe.conf`, §3).
2. Each CN must advertise its own routable address and point `--fe-host` at the FE (§4).
3. **The cross-host GPU data plane must clear the transport's 2.0 GB/s admission floor.** Left at
   the single-host defaults it measures 0.37 GB/s and the CN refuses the peer outright (§2).

---

## 2. The data plane, measured

The nixl tier runs a mandatory 16 MiB bandwidth canary on first contact with each peer and
*refuses* the session below a floor, because a wrongly-allocated staging arena still transfers
correct bytes while running ~220× slow — there is no error to catch, only a speed:

```316:318:experimental/starrocks/src/nixl_transport.rs
    /// Floor under which the link is declared degraded. The healthy same-host cuda_ipc path
    /// measured ~85-90 GB/s; the degraded staged-copy path ~0.4 GB/s.
    pub(super) const CANARY_FLOOR_GBPS: f64 = 2.0;
```

Cross-host, one GPU per host, 1 GiB payloads, bytes verified in both directions:

| Staging arena | `UCX_TLS` | Throughput | Path taken | Clears the 2.0 GB/s floor |
|---|---|---|---|---|
| `cudaMalloc` (what Sirius does today) | `cuda_copy,cuda_ipc,tcp,self` | **0.37 GB/s** | host-staged TCP | no — peer refused |
| `cudaMalloc` | `+ rc_mlx5`, 1 NIC | 48.7 GB/s | GPUDirect RDMA | yes |
| `cudaMalloc` | `+ rc_mlx5`, 4 NICs, `UCX_MAX_RNDV_RAILS=4` | 97 GB/s | multi-rail GPUDirect RDMA | yes |
| VMM, `CU_MEM_HANDLE_TYPE_FABRIC` | `cuda_copy,cuda_ipc,tcp,self` | **765 GB/s** | cross-host NVLink (MNNVL) | yes |

Same-host `cuda_ipc` for reference: 85–90 GB/s. So the fastest cross-host path on this rack is
*faster than the same-host path*, by 8×.

### Why `cudaMalloc` collapses across hosts

`cuda_ipc` spans hosts on an NVL72 only if the memory can be exported as a **fabric handle**,
which is a property of how it was allocated. `cuMemCreate` with
`CU_MEM_HANDLE_TYPE_FABRIC` can; `cudaMalloc` cannot — its handle
(`cuIpcGetMemHandle`) is node-local by construction. UCX then sees a device pointer it cannot
map remotely and silently emulates the write by staging through host memory over TCP.
`UCX_PROTO_INFO=y` shows the tell: a `ucp_put` sourced from *"host memory"* with no device id.

Raw `ucx_perftest` reached 693 GB/s `cuda_ipc` between these hosts, which is what made the
0.37 GB/s nixl result look like a nixl bug. It was not — `ucx_perftest` allocates its own
buffers, and the arena is the whole difference.

Sirius allocates the arena with plain `cudaMalloc`, deliberately, to dodge a *different*
degradation (pool/async memory loses the fast path too):

```42:44:src/exec/exchange_staging_arena.cpp
  // Plain cudaMalloc, by contract (see the class comment): pool memory silently loses the
  // transport's GPU-to-GPU fast path.
  if (auto err = cudaMalloc(&base_, capacity_bytes); err != cudaSuccess) {
```

That contract is correct for one host and wrong for two. Reaching 765 GB/s cross-host means
teaching the arena to allocate via the VMM API with a fabric handle — a Sirius-side change, not a
configuration knob. `scripts/nixl-echo-2node.sh` with `NIXL_ECHO_ARENA=fabric` already proves the
allocation works and the transport picks it up (`two_node_harness::cuda_vmm`).

### What to do today

Until the arena changes, run the cross-host data plane on **GPUDirect RDMA**. It needs no code
change, works with `cudaMalloc` memory, and 97 GB/s clears the floor with room to spare:

```bash
export UCX_TLS=cuda_copy,rc_mlx5,tcp,self
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1,enp3s0np0
export UCX_MAX_RNDV_RAILS=4
```

Note this contradicts the single-host guidance in `configs/gb200-4gpu/README.md` ("The NICs are
idle and must stay that way… do not add `rc`/`ud`/`ib` to `UCX_TLS`"). That advice is scoped to a
single box, where RDMA can only be slower than local NVLink. It does not apply here.

`UCX_NET_DEVICES` is not optional either way. Left to itself UCX picks the DPU interface
(`100.127.x`), which is not routable between these hosts, and every connection stalls until the
TCP timeout.

---

## 3. FE configuration

One FE serves both hosts. Put it wherever you like — below it runs on gcn-17. The only required
change is to stop pinning it to loopback:

```diff
- # Keep FE on loopback so it advertises 127.0.0.1 to the CN (matches its --fe-host default).
- priority_networks = 127.0.0.1/32
+ # Advertise the control LAN so CNs on other hosts can register and be heartbeated.
+ priority_networks = 10.87.140.32/27
```

Leave the rest of `conf/fe.conf` alone. In particular `run_mode = shared_data` is required —
shared-nothing schedules only on BEs, and a storage-less CN would never receive a fragment.

FE ports, unchanged: `query_port 9030` (MySQL, and where CNs register), `rpc_port 9020` (where
CNs send inventory reports), `http_port 8030`, `edit_log_port 9010`. Shared-data mode also uses
StarMgr on `6090`.

Start it as usual:

```bash
starrocks/output/fe/bin/start_fe.sh --logconsole
```

---

## 4. CN configuration

Registration is automatic — each CN dials the FE over MySQL and runs
`ALTER SYSTEM ADD COMPUTE NODE "{advertise_host}:{heartbeat_port}"`, retrying with backoff — so
there is no manual `ALTER SYSTEM` step and no ordering requirement between FE and CNs.

The three flags that matter for multi-host, all of which the launchers currently omit:

| Flag | Default | Set it to |
|---|---|---|
| `--advertise-host` | `127.0.0.1` | this host's control-LAN IP |
| `--fe-host` | `127.0.0.1` | the FE's control-LAN IP |
| `--bind-host` | `0.0.0.0` | leave alone — already correct |

`--advertise-host` is load-bearing twice over: it is half the FE's node identity, and it is half
the nixl agent name (`{advertise_host}:{brpc_port}`). Two CNs that both claim `127.0.0.1` with the
same heartbeat port are the *same node* to the FE, so loopback across two hosts silently collides.

It only names the **control** plane. The nixl data plane picks its own interface via
`UCX_NET_DEVICES`, so brpc and heartbeats ride `bond0` while payloads ride the RoCE fabric or
NVLink. That separation is why the /31 point-to-point data planes are fine.

On gcn-17, one CN per GPU:

```bash
cd ~/aocsa/sirius/experimental/starrocks
source scripts/cn-env.sh

export UCX_TLS=cuda_copy,rc_mlx5,tcp,self
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1,enp3s0np0
export UCX_MAX_RNDV_RAILS=4
export SIRIUS_EXCHANGE_STAGING_BYTES=1280MiB

SELF=10.87.140.52   # bond0 on gcn-17; use 10.87.140.53 on gcn-18
FE=10.87.140.52

for i in 0 1 2 3; do
    base=$((9100 + i * 10))
    target/release/sirius-starrocks-cn \
        --fe-host        "$FE" \
        --advertise-host "$SELF" \
        --gpu-device     "$i" \
        --heartbeat-port "$base" \
        --thrift-port    "$((base + 1))" \
        --brpc-port      "$((base + 2))" \
        --http-port      "$((base + 3))" \
        --starlet-port   "$((base + 4))" \
        --gpu-memory-limit  140GiB \
        --host-memory-limit 160GiB \
        --engine-dir ".cn$i" > "/tmp/cn$i.log" 2>&1 &
done
```

The CN logs through `tracing` to stdout only — there is no log file under `--engine-dir` (that
holds the engine's own logs and telemetry). Redirect it, as above, or §6 has nothing to read.

Because the two hosts now advertise different IPs, the port blocks no longer have to be globally
unique — gcn-18 can reuse `9100+i*10`. Keeping them identical across hosts makes the logs easier
to read.

### Ports that must be open

| Direction | Port | Purpose |
|---|---|---|
| CN → FE | 9030 | self-registration SQL |
| CN → FE | 9020 | periodic inventory report |
| FE → CN | heartbeat | liveness; carries the FE's own address back |
| FE → CN | http | FE's blacklist-eviction TCP probe |
| FE → CN | brpc | fragment dispatch, nixl metadata exchange |
| CN → CN | brpc (peer) | cross-host exchange |
| client → FE | 9030 | queries |

### Peer discovery

Each CN's warmup thread polls `SHOW PROC '/compute_nodes'` and pre-establishes a session to every
peer off the query path, so FE registration is enough — no static peer list needed. Do not disable
it: warmup is what avoids a reproducible cold-start deadlock, where two CNs each block their
transport thread on the other's metadata exchange and only recover when the 60 s RPC timeout fires,
long after the FE has given up on the fragment.

Two knobs earn their keep with two hosts, because the hosts are launched by hand and so boot
further apart than a single-box loop:

```bash
export SIRIUS_CN_NIXL_WARMUP_EXPECT_PEERS=7        # NUM_CNS - 1, ends the poll early
export SIRIUS_CN_NIXL_WARMUP_TIMEOUT_SECS=300      # default 180 assumes a single-box launch
```

Warmup failures are logged but never fail bring-up; a cold peer still gets a session on first use.
The canary runs on *either* path, so a degraded link surfaces as a failed query rather than a
failed bring-up.

---

## 5. Data locality — the other gotcha

There is no remote/object-store scan path. Queries read parquet through StarRocks `FILES()` with
`file://` paths, and the plan translator rejects anything that is not local:

```372:373:experimental/starrocks/crates/starrocks-plan-translator/src/scan_paths.rs
    /// Rejects non-local URI schemes and glob metacharacters in a scan path.
    fn check_local_path(node_id: i32, path: &str) -> Result<()> {
```

The FE splits a scan by byte range across CN instances without knowing which host holds the file.
So **every dataset must exist at the same absolute path on both hosts**, or a CN will be handed a
split for a file it cannot open.

**This is already satisfied.** Both hosts carry the same trees at the same absolute path, on
node-local ext4 (`/raid` = `/dev/md0`, 13 TB free):

| Path (identical on gcn-17 and gcn-18) | Size |
|---|---|
| `/raid/prestouser/aocsa/tpch_parquet_sf100` | 26 GB |
| `/raid/prestouser/aocsa/tpch_parquet_sf500` | 132 GB |

Measured SF100 layout: `lineitem` 17,187,602,838 B over 6 files (`part.0.parquet` …
`part.5.parquet`), `orders` 5,051,383,146 B, `nation` 2,250 B. Verify rather than copy — run
`find <path> -type f -printf '%P %s\n' | sort | md5sum` on both hosts and compare.

`/opt/sirius-ci/datasets/` and `/raid/prestouser/kkristensen/` exist on **gcn-18 only** and must not
be used for a two-host run; **there is no SF1 on either host**. Do not "solve" a path problem by
pointing both hosts at `$HOME` — that is NFS here, and the scan becomes the benchmark.

---

## 6. Verify

```bash
mysql -h 10.87.140.52 -P 9030 -uroot --vertical -e "SHOW COMPUTE NODES"
```

Every row should show a real `10.87.140.x` IP and `Alive: true`, with eight rows for four GPUs on
each of two hosts. Loopback in that output means a CN did not get `--advertise-host`.

Then confirm the data plane before trusting any query timing, since the canary logs the real
number per peer:

```bash
grep 'nixl bandwidth canary' /tmp/cn*.log
```

Cross-host pairs should read tens of GB/s on RDMA (or ~700+ once the arena is fabric-allocated).
A `below the 2 GB/s floor` refusal means `UCX_TLS`/`UCX_NET_DEVICES` did not take effect in that
CN's environment.

To exercise the transport on its own, without FE or CNs:

```bash
./scripts/nixl-echo-2node.sh                        # cudaMalloc arena, default TLS
NIXL_ECHO_ARENA=fabric ./scripts/nixl-echo-2node.sh # cross-host NVLink
```

---

## 7. Open work

- **Fabric-handle staging arena.** `exchange_staging_arena` needs a VMM allocation path
  (`cuMemCreate` + `CU_MEM_HANDLE_TYPE_FABRIC`) to reach 765 GB/s cross-host; the harness in
  `two_node_harness::cuda_vmm` is a working reference. Until then cross-host runs on RDMA at 97
  GB/s, and the same-host `cudaMalloc` contract must keep working.
- **A two-host launcher.** No script composes "FE here, CNs on hosts A and B"; §4 is hand-run.
- **`priority_networks` is committed as loopback**, so a two-host run means editing tracked
  config. Worth an override mechanism.
- **First-contact timeout.** `prpc_client` hardcodes a 60 s reply timeout with no env override,
  which is tighter than it looks when a cross-host peer is cold.
- **FE failover.** The CN validates the FE's advertised report address against `--fe-host`, so a
  leader moving to a different host is not supported.
