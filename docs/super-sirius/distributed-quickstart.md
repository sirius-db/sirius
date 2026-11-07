# Distributed quick start: Sirius on several machines

Three commands put the GPUs of several machines behind one SQL endpoint: a gateway, one node
per GPU machine, and a client. The client runs plain SQL. The gateway sends each statement to
every rank (one process per GPU), and rank 0 returns the result. How the gateway works is in
[Distributed Gateway](distributed-gateway.md); how a query runs across ranks is in
[Distributed Execution](distributed-sirius.md).

One machine does not need any of this, even with several GPUs: `pixi run duckdb` uses every
visible GPU in one process.

Through the gateway, CPU fallback is off on every rank: a query that Sirius cannot run on the GPU
fails instead of running on the CPU.

```bash
pixi run sirius-gateway           # on one machine; it needs no GPU
pixi run sirius-node              # on each GPU machine
pixi run sirius-client            # a DuckDB shell whose statements go to the gateway
```

## Contents

- [Prerequisites](#prerequisites)
- [Configure](#configure)
- [Run](#run)
- [What the gateway keeps](#what-the-gateway-keeps)
- [Recovery](#recovery)
- [Slurm](#slurm)
- [Troubleshooting](#troubleshooting)

## Prerequisites

| What | Why |
|---|---|
| The same Sirius checkout at the same path on every machine, built there with `pixi run build-distributed` | Every rank runs the same build. A checkout on a shared file system is built once. The Slurm script starts every node from the submit directory |
| The data at the same path on every machine | Every rank runs the same SQL, so `read_parquet('/data/...')` must resolve on each |
| A shared `$HOME`, or `pixi run sirius-node --gateway HOST:PORT --token-file FILE` on a node without it (or export `SIRIUS_GATEWAY_TOKEN` instead of `--token-file`) | Nodes and clients read the gateway's address, token and TLS fingerprint from `~/.sirius/gateways/<cluster>/gateway.conf` |
| Free TCP ports, open between the machines | The gateway listens on 9494. The rank on a node's i-th GPU listens on 9500 + i (Quack) and 27000 + i (exchange) |

`pixi run build-distributed` runs `scripts/install_nixl.sh`, which builds NIXL into
`~/.sirius/nixl/<commit>` unless a working install is there (`NIXL_PREFIX` sets another prefix).
It then reconfigures Sirius with `-DSIRIUS_WITH_NIXL=ON`, builds it, and builds the gateway
(`make -C gateway release`). The build directory keeps the NIXL prefix it found: after changing
`NIXL_PREFIX`, run `pixi run build-distributed` again. See [Settings and build](distributed-sirius.md#settings-and-build).

## Configure

One file, `~/.sirius/sirius.yaml`, configures every machine when `$HOME` is shared. The launch
scripts read its `sirius.distributed` section; Sirius ignores that section, and the rest of the
file configures Sirius on each rank as usual. The scripts look for the file where Sirius does:
`$SIRIUS_CONFIG_FILE`, then `./sirius.yaml`, then `~/.sirius/sirius.yaml`. Every key is optional.

```yaml
sirius:
  distributed:
    cluster: tpch               # the gateway's name
    gateway:
      port: 9494
      min_ranks: 8              # the first statement waits for this many ranks
      join_wait: 600            # for at most this many seconds
      networks: 10.87.140.0/24  # listen on this host's first address in these CIDRs
      # host: 10.87.140.52      # or on this address; wins over networks
    node:
      gpus: all                 # all, a list such as [0, 1], or slurm
      networks: 10.87.140.0/24  # each rank's listen and exchange address
      quack_port: 9500          # rank i: 9500 + i
      exchange_port: 27000      # rank i: 27000 + i
    exchange:                   # the gateway gives every rank these
      host_staging: auto        # auto: TCP lanes to ranks on other hosts
      tcp_lanes: 0              # 0: 16 shared by the ranks of a host
      mnnvl: auto               # auto | require | off: NVLink between nodes (fabric memory)
      frame_bytes: 16777216
      send_frames: 32
      recv_frames_per_peer: 16  # over TCP lanes, one frame in flight per lane
    ucx:                        # default: every transport on the GPU rail NICs, if any
      # tls: cuda_copy,cuda_ipc,tcp,self
      # net_devices: bond0      # without rail NICs: the interface that holds the rank's address
```

Each key sets an environment variable that the scripts read. A variable you export wins over
the file. Any error in the section, such as an unknown key or a value that is not a string,
number or list, stops the scripts before they apply any of it. `gateway.min_ranks`,
`gateway.join_wait` and `gateway.statement_timeout` must be whole numbers.

| Key | Variable | Default | What |
|---|---|---|---|
| `cluster` | `SIRIUS_CLUSTER` | `default` | Names the gateway and its state directory. One gateway per name runs at a time |
| `gateway.port` | `SIRIUS_GATEWAY_PORT` | 9494 | The gateway's client port. The gateway script's second argument wins over it |
| `gateway.min_ranks` | `SIRIUS_GATEWAY_MIN_RANKS` | 1 | `sirius_gateway_min_ranks`: registered ranks the first statement waits for |
| `gateway.join_wait` | `SIRIUS_GATEWAY_JOIN_WAIT` | 600 | `sirius_gateway_join_wait`: seconds a statement waits for them before it fails, at most a year |
| `gateway.statement_timeout` | `SIRIUS_GATEWAY_STATEMENT_TIMEOUT` | 3600 | `sirius_gateway_statement_timeout`: seconds a statement runs before the gateway cancels it on every rank; 0 is none. Clients cannot set it |
| `gateway.host` | `SIRIUS_GATEWAY_HOST` | | The address the gateway listens on |
| `gateway.networks` | `SIRIUS_GATEWAY_PRIORITY_NETWORKS` | | Or this host's first address in these CIDRs (`;` or `,` between them). Without either, the address of the default route's interface; with ranks spawned on this host, 127.0.0.1 |
| `node.gpus` | `SIRIUS_NODE_GPUS` | `all` | The GPUs that get a rank: `all`, a list, or `slurm` (the GPU Slurm gave the task) |
| `node.networks` | `SIRIUS_NODE_NETWORKS` | the gateway's subnet | Each rank listens on this node's first address in these CIDRs, and UCX uses that address's interface |
| `node.quack_port` | `SIRIUS_NODE_QUACK_PORT` | 9500 | First rank's Quack port |
| `node.exchange_port` | `SIRIUS_NODE_EXCHANGE_PORT` | 27000 | First rank's exchange port |
| `exchange.host_staging` | `SIRIUS_GATEWAY_EXCHANGE_HOST_STAGING` | `auto` | `sirius_gateway_exchange_host_staging`, which the gateway sets as `sirius_exchange_host_staging` on every rank. `auto` reaches ranks on other hosts over TCP lanes |
| `exchange.tcp_lanes` | `SIRIUS_GATEWAY_EXCHANGE_TCP_LANES` | 0 | `sirius_gateway_exchange_tcp_lanes`: TCP lanes per rank, at most 64. 0 shares 16 among the ranks of a host, at least 2 each |
| `exchange.mnnvl` | `SIRIUS_GATEWAY_EXCHANGE_MNNVL` | `auto` | `sirius_gateway_exchange_mnnvl`, which the gateway sets as `sirius_exchange_mnnvl` on every rank: ranks on other nodes of the NVLink domain exchange over NVLink |
| `exchange.frame_bytes` | `SIRIUS_GATEWAY_EXCHANGE_FRAME_BYTES` | 16777216 | `sirius_gateway_exchange_frame_bytes`: the exchange's frame size |
| `exchange.send_frames` | `SIRIUS_GATEWAY_EXCHANGE_SEND_FRAMES` | 32 | `sirius_gateway_exchange_send_frames`: send frames shared by a rank's peers |
| `exchange.recv_frames_per_peer` | `SIRIUS_GATEWAY_EXCHANGE_RECV_FRAMES_PER_PEER` | 16 | `sirius_gateway_exchange_recv_frames_per_peer`: receive frames each peer may fill; over TCP lanes, one frame in flight per lane |
| `ucx.tls` | `UCX_TLS` | `all` on a host with GPU rail NICs, else `cuda_copy,cuda_ipc,tcp,self` | UCX transports; the default is `scripts/nixl_env.sh`'s |
| `ucx.net_devices` | `UCX_NET_DEVICES` | the GPU rail NICs, else the interface of the rank's address | UCX devices, for example `mlx5_0:1`; `rails` asks for the rail NICs. The rail NICs are the active RDMA ports in no bond at the highest link speed (on GB200 nodes `mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1`). Notifications over RC on them let the exchange over NVLink reach full speed. Never list bond0 with them: UCX then hangs reaching a peer's bond0 address from a rail NIC |

A gateway on an address other than loopback serves TLS. It writes the subnet of its address to
`gateway.conf`, and nodes without `node.networks` pick their address in that subnet.

## Run

**1. Gateway**, on a machine with the build that shares `$HOME` with the others:

```bash
pixi run sirius-gateway
```

It prints its address and what to run next, then opens its console, a DuckDB shell where
`FROM sirius_gateway_status()` shows the ranks. It serves until the console's stdin closes
(Ctrl-D). `pixi run sirius-gateway --serve` has no console and serves until it is killed.
`pixi run sirius-gateway N` instead starts N ranks on this machine, rank i on GPU i; it also
replays the saved DDL log when it restarts.

**2. Nodes.** On each GPU machine, check it, then start its ranks:

```bash
pixi run sirius-node --check                      # the build, the GPUs, the gateway
pixi run sirius-node --check --data /data/tpch    # also a path every rank reads
pixi run sirius-node
```

`--check` prints one `ok` or `FAIL` line per check, each failure with a fix, and ends with
`node ready for gateway <cluster>`. `sirius-node` starts one rank per GPU, each under
`scripts/sirius_rank_node.sh`, which starts the rank again whenever it exits and registers it
with the gateway. Ranks started before the gateway wait for its `gateway.conf`. Logs are in
`/tmp/sirius_node_<user>_<cluster>/rank_<port>/rank.log` and `register.log`. Ctrl-C stops the
node's ranks. If one rank's script exits on an error it cannot retry, the node stops all its
ranks.

**3. Client**, on any machine with the build and the shared `$HOME`:

```bash
pixi run sirius-client
```

```sql
CREATE VIEW lineitem AS FROM read_parquet('/data/tpch/lineitem/*.parquet');
SELECT l_shipmode, count(*) FROM lineitem GROUP BY ALL;   -- every rank runs it; rank 0 answers
FROM sirius_gateway_status();                             -- one row per rank, on the gateway
```

The first statement waits until `gateway.min_ranks` ranks are registered, for at most
`gateway.join_wait` seconds, then starts them. While it waits, `sirius_gateway_status()` says
`waiting for N ranks (sirius_gateway_min_ranks); M are registered`. Other arguments go to
DuckDB: `pixi run sirius-client -c 'FROM sirius_gateway_status()'` runs one statement and exits.
A pixi task parses its extra arguments again and drops quotes inside them, so for SQL with
string literals call the script itself, `pixi run scripts/sirius_client.sh -c "..."`, or pass a
file: `pixi run sirius-client < queries.sql`.
The client keeps the token off its command line: it writes the `ATTACH` to `client.init.sql`
next to `gateway.conf` and passes that file with `-init`. That file reads `~/.duckdbrc` first,
which `-init` would otherwise skip.

To run another cluster, set `SIRIUS_CLUSTER=<name>` for the gateway, and pass
`--cluster <name>` to `sirius-node` and `sirius-client` (for the client, as the first argument).

## What the gateway keeps

`scripts/sirius_gateway.sh` keeps its state in `~/.sirius/gateways/<cluster>/`, mode 700:

| File | What |
|---|---|
| `gateway.conf` | `gateway=<host>:<port>`, `gateway_token`, `gateway_fingerprint` (with TLS), `gateway_id` (new at each start), `networks` (the subnet of the gateway's address). Rewritten at each start; nodes read it again every 10 s |
| `token` | The client token, made once and kept, so it stays valid across restarts. `SIRIUS_GATEWAY_TOKEN` overrides it |
| `tls/` | The gateway's TLS key pair, kept, so its fingerprint stays valid |
| `ddl.sql` | The DDL log, mode 600, replayed on every start of the ranks. Statements that carry credentials are not saved |
| `client.init.sql` | Mode 600: the `ATTACH` with the token that `sirius-client` passes to DuckDB with `-init` |
| `lock` | Held by the running gateway, so a second one with the same name refuses to start |

Rank init files and the logs of spawned ranks are in `/tmp/sirius_gateway_<user>_<cluster>`,
or the gateway script's third argument. Each node keeps its rank configs and state in
`/tmp/sirius_node_<user>_<cluster>/`, mode 700. Each rank's directory, `rank_<port>/`, holds
`rank.log`, `register.log`, the rank's token, and `register.sql` (mode 600): the
`sirius_gateway_add_rank` call that registers the rank, which `sirius_rank_node.sh` feeds to
DuckDB on stdin so that no token is on a command line.

## Recovery

| Event | What happens |
|---|---|
| A rank dies between statements | Its node starts it again, after 1 s, doubling up to 30 s. Before the next statement the gateway probes each registered rank's Quack port for 2 s. If the rank is still down, the gateway restarts every rank, replays the DDL log, then runs the statement. The probe does not catch a rank that is already back ([Limits](distributed-gateway.md#limits)) |
| A rank dies during a statement | The statement fails and the cluster is `failed`. The next statement restarts every rank and replays the DDL log |
| The gateway restarts | Start it again with `pixi run sirius-gateway`. Its new `gateway_id` tells the nodes to register their ranks again, within 10 s. The first statement waits for `gateway.min_ranks`, restarts every rank (each rank process that joined the old gateway's ranks exits, and its node starts it again), and replays `ddl.sql`. A saved entry that now fails on every rank is skipped and dropped from `ddl.sql`; the status `error` column says `skipped saved DDL entry N: ...`. Statements with credentials, such as `CREATE SECRET`, are not saved: run them again, or put them in `sirius_gateway_rank_init` (`SET GLOBAL` on the gateway's console). A saved view that needs a missing secret is skipped, so create it again after the secret |
| `ddl.sql` cannot be written | The statement still succeeds. The status `error` column says `cannot save the DDL log: ...`, while the cluster stays `ready` |
| A node is added | Start `sirius-node` on it. Its ranks register, and every rank restarts at the next statement |
| A node is removed | Stopping it leaves its ranks registered, and every restart fails until they are dropped: `CALL sirius_gateway_drop_rank('<host>:<port>')` on the gateway's console |

A node started with `--gateway` has no `gateway.conf` to watch: after the gateway restarts,
restart `sirius-node` there.

## Slurm

```bash
sbatch -N2 scripts/sirius_slurm.sbatch               # from the Sirius checkout
SIRIUS_CLUSTER=<job id> pixi run sirius-client
```

The job runs the gateway on its first node, without a GPU, named after the job id, with
`min_ranks` equal to the job's task count. Each task is one rank on one GPU
(`--gpus-per-task=1`): `sirius_node.sh --gpus slurm`, with Quack port 9500 + `SLURM_LOCALID` and
exchange port 27000 + `SLURM_LOCALID`. The script asks for 4 tasks per node;
`sbatch -N2 --ntasks-per-node=2 scripts/sirius_slurm.sbatch` uses 2 GPUs per node. The gateway
serves until the job ends. If either the gateway's step or the ranks' step exits, the job stops
the other and ends with exit code 1. At exit the script removes `~/.sirius/gateways/<job id>/`,
so the job's token, keys and DDL log go with it. It needs a shared `$HOME`.

Tested on two GB200 nodes with 2 GPUs each (`sbatch -N2 --ntasks-per-node=2`): a count at SF100
and TPC-H Q12 were correct, and cancelling the gateway's step ended the job with exit code 1.

## Troubleshooting

| Message | Cause and fix |
|---|---|
| `FAIL no Sirius build at ...`, `FAIL Sirius is built without NIXL` | Run `pixi run build-distributed` on that machine |
| `FAIL the Sirius build does not find libnixl` | The NIXL install moved or is gone: `pixi run scripts/install_nixl.sh`, then `pixi run build-distributed` (or export `NIXL_PREFIX`) |
| `SIRIUS_WITH_NIXL: no NIXL in <prefix>` (CMake) | `pixi run scripts/install_nixl.sh`, or set `NIXL_PREFIX` to an install |
| `.../libplugin_UCX.so does not resolve its libraries` (CMake) | The NIXL install was built against a pixi environment that is gone. Run `pixi run scripts/install_nixl.sh` again: it removes the files it installed there (listed in `.sirius-install-nixl`), never the whole prefix, and installs again |
| `<prefix> holds something other than a working NIXL` | `install_nixl.sh` does not install over files it did not write: give an empty or new prefix |
| The build still uses the old NIXL after `NIXL_PREFIX` changed | The build directory keeps the prefix it found. Run `pixi run build-distributed`, which looks again, or configure with `-DSIRIUS_NIXL_PREFIX=<prefix>` |
| `FAIL no GPU visible to nvidia-smi -L`, `FAIL no GPU N: this node has M` | Run on a GPU machine, and give `--gpus` (or `node.gpus`) GPUs that exist |
| `FAIL --gpus slurm, but Slurm gave this task no GPU` | Ask Slurm for `--gpus-per-task=1` |
| `FAIL no ~/.sirius/gateways/<cluster>/gateway.conf`, `no ... gateway.conf: start the gateway first` | Start the gateway on a host that shares `$HOME`, with the same cluster name; or pass `--gateway` and `--token-file` to `sirius-node` |
| `--token would show the token to every user in ps` | `sirius-node` has no `--token`: use `--token-file FILE`, or export `SIRIUS_GATEWAY_TOKEN` |
| `bad --gateway HOST:PORT, token ...`, `bad gateway token`, `bad rank token` | Tokens are letters, digits, `_` and `-` only |
| `SIRIUS_GATEWAY_MIN_RANKS, SIRIUS_GATEWAY_JOIN_WAIT and SIRIUS_GATEWAY_STATEMENT_TIMEOUT take a number` | Give whole numbers for `gateway.min_ranks`, `gateway.join_wait` and `gateway.statement_timeout` |
| `FAIL cannot reach the gateway at <host>:<port>` | Start the gateway, or open its TCP port between the two machines |
| `FAIL no address of this node is in <CIDRs>`, `no address of this host is in priority_networks <CIDRs>` | The node has no address in the gateway's subnet, or in `node.networks`. Set `node.networks` (or `--networks`) to the CIDR of the address the other machines reach |
| `gateway <cluster> already runs (lock ...)` | One gateway per name: stop the other one, or set another `SIRIUS_CLUSTER` |
| `no gateway build at ...` | `pixi run build-distributed`, or `pixi run make -C gateway release` |
| `the gateway starts once N ranks are registered (sirius_gateway_min_ranks); M are` | Fewer ranks registered within `gateway.join_wait`. Check the nodes' `register.log` and `rank.log`, or lower `gateway.min_ranks` |
| `registering with the gateway at ... failed (see .../register.log)` | The rank cannot reach the gateway, or holds an old token or fingerprint. The script retries; `register.log` has the gateway's answer |
| `sirius_gateway_add_rank can only be called on the gateway's console` in `register.log` | The gateway does not let ranks register: it was started with ranks of its own. Set `SIRIUS_GATEWAY_ALLOW_JOIN=1` |
| `rank N (<name>) did not answer at quack:...` | The gateway cannot reach that rank's Quack port, or the rank did not come up within 10 minutes. See its `rank.log`; drop it if its node is gone |
| `rank N (<name>) still runs the process the gateway asked to exit` | The rank was asked to exit to join the new set of ranks, and did not within 10 minutes. See its `rank.log` |
| `rank <name>'s exchange address 127.x.x.x is a loopback address, which the other ranks cannot reach` | The node's name resolves to a loopback address, which happens with `--gateway` and no networks. Pass `--networks` (or set `node.networks`) to pick a routable address |
| `rank N failed its session setup: ...` | Usually the exchange: the ranks could not connect to each other. Check that the exchange ports are open, and `ucx.tls` and `ucx.net_devices` |
| `Endpoint timeout` in a rank's log | UCX picked a port that the ranks cannot connect through. Keep the default `ucx.tls` and `ucx.net_devices`, or set `ucx.net_devices` to ports that reach every node |
| `unknown key sirius.distributed.<key>`, `sirius.distributed.<key> must be a string, a number or a list of them` | Fix that key in `sirius.yaml`; the scripts apply none of the section until it is right. The first message lists the known keys |
| `topology.num_gpus must be all or a positive count` | `num_gpus: 0` is an error; use `num_gpus: all` or leave the key out |

`sirius_gateway_status()` has a `last_error` column with each rank's last failure to join, to
answer, or to keep running.
