# Distributed Sirius: running one query across GPUs

Sirius can run one SQL query across several processes, one per GPU, on one host or several.
Every rank runs the same SQL. Each scans its own share of each table, and the plan moves rows
between ranks with two operators, `EXCHANGE_SINK` and `EXCHANGE_SOURCE`, over
[NIXL](https://github.com/ai-dynamo/nixl). Rank 0 returns the query's result.

```sql
LOAD 'sirius-duckdb/build/release/extension/sirius/sirius.duckdb_extension';
SET enable_duckdb_fallback = false;
SET sirius_exchange_peers = '10.0.0.1:27000,10.0.0.2:27000,10.0.0.3:27000,10.0.0.4:27000';
SET sirius_exchange_rank = 2;            -- this process's position in the list
CREATE VIEW lineitem AS SELECT * FROM read_parquet('/data/tpch_sf1000/lineitem/*.parquet');
SELECT ...;                               -- the same statement on every rank
```

Without `sirius_exchange_peers`, nothing here applies and queries run in one process as before.

To set up the ranks, the [gateway](distributed-gateway.md) and a client on several machines, see
the [distributed quick start](distributed-quickstart.md).

## Contents

- [How a query runs](#how-a-query-runs)
- [Layers](#layers)
- [The plan rewrite](#the-plan-rewrite)
- [Dynamic filters across ranks](#dynamic-filters-across-ranks)
- [Settings and build](#settings-and-build)
- [Running TPC-H Q12 on 4 GPUs](#running-tpc-h-q12-on-4-gpus)
- [Limits](#limits)
- [Tests](#tests)
- [Files](#files)

## How a query runs

TPC-H Q12 on 4 ranks: a join of `orders` and `lineitem`, a group-by on `l_shipmode`, and an
`ORDER BY`.

```mermaid
flowchart BT
    LS["GPU_SCAN lineitem<br/>(rank's files) → FILTER"] --> LX["EXCHANGE hash(l_orderkey)"]
    LX --> LB["PARTITION → CONCAT (build)"]
    OS["GPU_SCAN orders<br/>(rank's files) → DYNAMIC_FILTER"] --> OX["EXCHANGE hash(o_orderkey)"]
    OX --> OP["PARTITION → CONCAT (probe)"]
    LB --> J["HASH_JOIN → PROJECTION → HASH_GROUP_BY"]
    OP --> J
    LB -. "Bloom, ORed across ranks" .-> OS
    J --> GX["EXCHANGE hash(l_shipmode)"]
    GX --> M["PARTITION → MERGE_GROUP_BY → ORDER_BY"]
    M --> SS["SORT_SAMPLE<br/>(samples gathered: same boundaries everywhere)"]
    SS --> RX["EXCHANGE range"]
    RX --> SP["SORT_PARTITION → MERGE_SORT"]
    SP --> FX["EXCHANGE gather → rank 0, ordered by rank"]
    FX --> RC["RESULT_COLLECTOR (rank 0)"]
```

Each `EXCHANGE` box is an `EXCHANGE_SOURCE` whose child is an `EXCHANGE_SINK` of the same stream.
The sink ends its producer's pipeline; the source starts the consumer's.

## Layers

From the bottom up. Each layer is tested without the ones above it.

### Wire format (`src/exchange/wire.*`, `batch_io.*`)

A batch travels as one logical byte stream, cut into frames. Each column's buffers (null mask,
data, and for `STRING` the offsets child) sit at the offsets cuCascade's `column_metadata` gives,
256-byte aligned. That is the layout the host tier keeps its tables in, so one set of frames can
land on the GPU or in host blocks. The header (rows, stream bytes, column metadata) travels in
the first frame's notification and is checked before anything is allocated. Columns are
fixed-width or `STRING`.

- `outgoing_batch` holds a GPU batch read-only, so it cannot spill mid-send. It copies a sliced
  table first, and copies any stream range into a frame with one batched copy.
- `incoming_batch` lands a batch without waiting: on the GPU if a reservation is free now, else
  in host blocks, else it refuses. It then seals the frames into a GPU or host `data_batch`.

### Transport (`src/exchange/transport.*`, `messages.*`, `fabric.hpp`)

Each rank has a send ring and a receive ring. Both are fixed-size frames in one device region,
and they are the only memory NIXL registers. The region comes from `cuMemCreate` with a fabric
handle where the GPU and driver allow it (see [NVLink between nodes](#nvlink-between-nodes-mnnvl)),
else from `cudaMalloc`. It is never pool memory: UCX `cuda_ipc` cannot export stream-ordered
allocations, and a pool block could be reused while a peer's write into it is still landing.

- **Writes.** One write covers as many consecutive frames as the sender holds consecutive send
  frames and credits for, up to half a peer's credits, so two writes to a peer can be in flight.
  NVLink between nodes needs large writes: one write at a time moves about 450 GB/s at 16 MB,
  700 at 128 MB and 770 at 1 GB, because each write costs about 15 µs.

- **Credits.** The receive ring is split evenly among the peers at start. A sender writes only
  into the frames it holds, and gets each one back once the receiver has copied it out. There is
  no lane set-up and no grant.
- **Landing.** With no room on the GPU or host, a batch's frames stay in the ring, holding their
  credits. That stops the peer until room appears.
- **End of stream** carries the batch count, so frames, ends and credits may arrive in any order.
  A frame that arrives before its batch's header waits for it.
- **Fail-stop.** A failed write, a malformed notification, a peer's error or a stall past the
  timeout calls `on_error` once and tells every peer.
- **Threading.** One engine thread owns the fabric, the rings' free lists and every landing. It
  polls CUDA events instead of synchronizing. A batch sent to the rank itself is handed over
  without a copy.

A `fabric` is what the transport needs from NIXL: one-sided writes into a peer's receive ring,
whose notification arrives only once the bytes have landed, plus plain notifications. Tests use
an in-process loopback fabric that can deliver notifications late and shuffled, and can fail
writes.

### NIXL fabric (`src/exchange/nixl_fabric.*`)

Each rank's NIXL agent serves its metadata from a listener thread on the rank's port. At start a
rank:

1. registers both rings with the UCX backend;
2. fetches every peer's metadata;
3. sends each peer a hello naming its receive ring's address, size and device.

It returns once every peer's metadata covers that ring. The listener serves metadata before the
rings are registered, so a peer fetched too early is fetched again. After the hello, everything
is a NIXL write or a notification. There is no TCP or RPC channel of our own. Between GPUs on one
host, UCX picks `cuda_ipc` zero-copy.

**Host staging.** `cuda_ipc` needs peer access between the two GPUs. Where there is none (GPUs
under different PCIe switches, which `nvidia-smi topo -p2p r` shows as `NS`), UCX falls back to
staging through the host over TCP, at about 1 GB/s. So a rank also keeps a pinned mirror of its
receive ring in POSIX shared memory (`/dev/shm/sirius-exchange-<pid>-<rank>`) when NVML reports a
GPU on its host without peer access to its own, and its hello carries its PID (0 without a
mirror) and its GPU's PCI bus id. If `auto` cannot create the mirror (a small `/dev/shm`), the
rank logs a warning and goes on without one; a peer that must stage to it then fails to connect,
saying so. For a peer whose GPU NVML reports without peer read and
write access, a write is one device-to-host copy into that peer's mirror, then a notification
naming the offset and size; the peer copies the bytes host-to-device into its receive ring and
only then passes the write's own notification to the transport. Credits are unchanged: a mirror
frame is reused only after the transport has copied its receive frame out.
`SET sirius_exchange_host_staging = 'always'` (or `'never'`) overrides the choice (default `auto`). Each rank logs
its path to every peer at start. Staging maps the peer's mirror from this host's `/dev/shm`, so it
works only between ranks on one host that share `/dev/shm`; ranks on several hosts, or in separate
IPC namespaces, need `never` until #2372. A rank that crashes leaves its mirror in `/dev/shm`;
the launchers in `scripts/` remove mirrors whose process is gone before they start.

### NVLink between nodes (MNNVL)

On GB200 NVL systems, GPUs on different nodes of one NVLink domain reach each other's memory over
NVLink, at up to about 780 GB/s each way per GPU pair, against about 98 GB/s over the RDMA rails.
UCX `cuda_ipc` does this through the IMEX daemon, but only for memory with a fabric handle
(`CU_MEM_HANDLE_TYPE_FABRIC`). Plain `cudaMalloc` memory has only a legacy IPC handle, and UCX then
moves it over the network without a word. So with `sirius_exchange_mnnvl` at `auto` (the default)
or `require`, a rank allocates both rings with `cuMemCreate` and a fabric handle
(`src/exchange/fabric_memory.*`), rounded up to the allocation granularity. Without fabric
handles (no support, or no IMEX channel open), `auto` logs why and uses `cudaMalloc`; `require`
fails.

The hello also carries whether the receive ring is in fabric memory and the GPU's NVLink domain:
the cluster UUID and clique id from NVML's fabric info (`nvidia-smi -q`, section Fabric). A peer
whose GPU is in this GPU's domain, both rings in fabric memory, gets the path "NIXL over the
NVLink fabric" (`choose_path()` in `src/exchange/path_choice.*`); every other peer gets the path
it had before. `require` fails at connect for a peer that cannot be reached that way.

UCX picks its transport per endpoint and says nothing when `cuda_ipc` cannot map a peer's memory.
So after connecting, a rank times one write, as large as the transport's largest, to each peer on
the fabric path, into its own credit frames there, and logs the rate. Below 150 GB/s it warns
(`require`: fails) that UCX did not take `cuda_ipc`. `UCX_PROTO_INFO=y` prints the transport of
each UCX protocol, for example `cuda_ipc/cuda` for a put from `cuda/GPU0` to `cuda/dev[0]`.

The ring sizes are settings, the same on every rank: `sirius_exchange_frame_bytes` (16 MiB),
`sirius_exchange_send_frames` (32, shared by the peers) and `sirius_exchange_recv_frames_per_peer`
(16: two 128 MiB writes to a peer in flight). A rank's ring memory is
`frame × (send frames + receive frames × peers)`: 2.25 GiB with 8 ranks, 18.3 GiB with 72
(credits on demand, #2371, would bound it). With 8 receive frames per peer, one GPU pair moved
626 GB/s instead of 706.

`scripts/distributed_test_nixl_pairs.sh <node1>,<node2>` runs the hidden `[nixl_pairs]` test on
two nodes of a Slurm partition, one rank per GPU paired with the same GPU on the other node: one
and four pairs, one and both directions, 1 GB and 128 MB batches, then smaller batches, then a
check of every value. It writes `summary.md` with GB/s per phase next to raw NIXL's.

### Operators (`src/op/sirius_physical_exchange_{sink,source}.*`)

`EXCHANGE_SINK` routes by an `exchange_spec`:

| Mode | Routes | Notes |
|---|---|---|
| `hash` | by a hash of the key columns | PARTITION's key casts; its own seed, so PARTITION's split of each rank's share stays even. With `null_keys_to_every_rank`, a row with a NULL in a key column goes to every rank |
| `range` | by the key columns against world − 1 boundaries | Input batches sorted on the keys; each rank's rows are one sorted run, cut at `lower_bound` as `SORT_PARTITION` cuts |
| `broadcast` | every row to every rank | |
| `gather` | every row to one rank | |

Finalizing the sink ends the stream to every rank that may receive it. A batch with no rows is
neither routed nor sent, so a range exchange needs no boundaries when no rank has rows.

`EXCHANGE_SOURCE` starts a pipeline with what every expected sender sent:

- **Hint chaining.** Its child sink's pipeline wires a port into it that carries no data. While
  the source waits, its hint names that pipeline's head, so the task creator's walk reaches this
  rank's own scans.
- **Unordered,** each batch is a task on arrival.
- **Ordered,** nothing is released until every sender has ended. Then everything goes out as one
  task, by rank and then send order.
- **Under `MERGE_AGGREGATE` or `MERGE_TOP_N`** the source ends its own pipeline, as a partial
  aggregate would, so the merge runs once over every rank's partials.
- **A stream that brings nothing** still hands on one empty batch, shaped like the stream's
  batches, as a local producer would: a hash join, for one, waits for a batch on each side. A rank
  that a gather does not target hands on none, since an aggregate there would make a row of
  nothing.
- A failed exchange is the exception of the source's next task.

`exchange_service` is one process's end of every exchange. It holds what arrives for a stream
before the stream's source is planned, and replays it when the source attaches.

### Collectives (`src/exchange/all_gather.*`)

`all_gather()` sends one small table from every rank to every rank and blocks until all have
arrived. It is used three ways:

- **`SORT_SAMPLE` in distributed mode.** Each rank shares up to 1024 evenly spaced rows of its
  sorted sample, and every rank merges them all in rank order. From the same global sample each
  rank derives the same world − 1 range boundaries for the exchange after it, and its own
  `SORT_PARTITION` boundaries from inside its range. A rank with no input still runs its boundary
  task (`runs_without_input()`), since the others wait for it. `distribute_plan()` sets a
  `sample_all_gather_boundaries` on the plan's `SORT_SAMPLE` (`set_cross_rank_boundaries()`),
  which the operator asks for its boundaries instead of cutting its own sample. It cannot be set
  again once the boundary task is claimed. It waits up to 10 minutes for the other ranks' samples,
  keeps what it gathered for a retry of the task, and frees it once the boundaries are made.
- **The plan-shape check** (below).
- **The Bloom merge** (below).

`leave_all_gather()` takes part in an all-gather with nothing and returns at once, dropping what
the others send. A rank that would ignore the gathered tables uses it, so it never waits on a peer
that may be waiting on it.

## The plan rewrite

`distribute_plan()` (`src/planner/distributed_plan.cpp`) runs after
`insert_gpu_pipeline_operators()`. It walks the wrapped plan, tracks whether each subtree's rows
are spread across ranks, range-partitioned, or on rank 0, and puts an exchange where the
single-process plan already partitions:

| Plan shape | Exchange |
|---|---|
| `HASH_JOIN`, each side `CONCAT → PARTITION → input` | `hash` on that `PARTITION`'s join keys and casts, below it |
| the same, with a small build side | `broadcast` of the build side; the probe side stays where it is |
| `NESTED_LOOP_JOIN`, each side `CONCAT → PARTITION → input` | `broadcast` of the build side; the probe side stays where it is |
| `DENSE_COUNT_JOIN`, each side `PARTITION → input` | `hash` on the join key, both sides |
| `LEFT/RIGHT_DELIM_JOIN` | `hash` on the delim columns, before the input fans out to the inner join and the `DISTINCT`; `hash` on the inner join's other side |
| `MERGE_GROUP_BY → PARTITION → HASH_GROUP_BY` | `hash` on the group keys, between `PARTITION` and `HASH_GROUP_BY` |
| the same, with `COUNT(DISTINCT)` | `hash` on the group keys, below `HASH_GROUP_BY`: partial distinct sets are LIST columns, which the wire does not carry |
| `MERGE_AGGREGATE → UNGROUPED_AGGREGATE` | `gather` to rank 0, between them |
| `MERGE_TOP_N → TOP_N` | `gather` to rank 0, between them; a row outside its rank's top N is outside the overall top N |
| `MERGE_SORT → SORT_PARTITION → SORT_SAMPLE → ORDER_BY` | `range` between `SORT_PARTITION` and a distributed `SORT_SAMPLE` |
| the result | `gather` to rank 0, ordered after a distributed sort |

`GPU_SCAN`, `FILTER`, `PROJECTION` and `DYNAMIC_FILTER` keep their input's placement. A `CTE`
places nothing itself: each rank materializes its share, and the `CTE_SCAN`s under the consumer side
take the materialization's placement. A hash exchange may start from an input on rank 0 alone (a
gathered scalar): rank 0 sends its rows by key, and the other ranks' copies of it hold none.
A `GPU_VALUES` (a `VALUES` list, or a `count(*)` DuckDB answered from Parquet metadata) is on
rank 0 the same way: every rank planned the same rows, so the other ranks' copies emit none.

A hash join's build side is broadcast when its estimated size is at most 1 GiB and at most the
probe side's divided by ranks − 1, and when its join type emits build rows only with a probe match.
A hashed `MARK` join (`IN`, `NOT IN`, `ANY`) sends build rows with a NULL key to every rank: one
NULL build key makes an unmatched probe row's mark NULL rather than false, on whichever rank that
row is. Each rank also sends one row of its build input to every rank, so a rank's share is empty
only when the whole build is: a NULL probe key's mark is NULL against any non-empty build. That
row matches nothing off its home rank, since every probe row with its key hashes there.
A nested-loop join refuses the types that emit unmatched build rows (`RIGHT`, `OUTER`,
`RIGHT_SEMI`, `RIGHT_ANTI`), since every rank holds the whole build.

Refused with `not_implemented_exception`: a delim join whose join keys are not its delim columns,
`COUNT(DISTINCT)` with grouping sets, `ORDER BY` on an expression, a DuckDB table (#2374), a
`read_parquet` of one remote URI (#2377), `filename` or `file_index` of a split table, byte
ranges carried by the plan itself, a nested column crossing an exchange (grouped `STDDEV_SAMP`
and ungrouped `COUNT(DISTINCT)` partials), and any other operator. Only Parquet tables read through
`read_parquet`'s file list are split across ranks.
Running such an operator on each rank's share would return a wrong answer.

- **Scan shares.** A table's local parquet files are laid end to end and cut into equal byte
  spans, one per rank (`byte_ranges_for_rank`). A rank reads the row groups whose start offset
  falls in its span ([`parquet_byte_range`](../../src/op/scan/parquet_byte_range.hpp)), so every
  row group is read once, and a one-file table still splits. A table of one row group lands on
  one rank. Remote files take no byte ranges: each rank keeps files `rank, rank + world, …`.
  Every rank binds every file, so DuckDB plans every rank from the same statistics.
- **Stream ids** come from the connection's distributed-query count, the same on every rank
  because every rank runs the same statements.
- **Plan-shape check.** Before running, every rank all-gathers a hash of its plan's shape, and
  every rank fails the query if any differs.

### Adding a distributed behaviour

A distributed plan holds the single-process operators and the two exchange operators, and nothing
else. Where an operator needs the other ranks in the middle of its work, it asks an interface:

| Operator | Interface (no `exchange/` include) | Across ranks (`src/op/distributed/`) |
|---|---|---|
| `SORT_SAMPLE` | `cross_rank_sort_boundaries` | `sample_all_gather_boundaries` |
| `HASH_JOIN`'s filter publisher | `cross_rank_filter_merge` | `bloom_all_gather_merge` |

- The interface sits beside its operator and speaks of samples, filters and processes, not ranks
  or streams. With none set, the operator runs as in one process.
- The implementation holds the `exchange_service` and its stream ids.
- Only `distribute_plan()` builds one and sets it, through one call per hook
  (`install_sort_boundaries()`, `install_filter_merge()`), and it adds to the plan shape there.
- Two hooks on every operator cover the rest: `add_finalize_listener()` (act when an operator's
  pipeline ends) and `runs_without_input()` (a task owed with no input).

A new behaviour adds an implementation, an interface if its operator has none that fits, and a
case in `distribute_plan()`: not an operator class.

## Dynamic filters across ranks

After a join's exchange, a rank's build holds only its share of the keys, so a filter from it
alone would drop rows another rank's build matches. In distributed mode a publishing join
carries a `cross_rank_filter_merge` (`bloom_all_gather_merge` in `src/op/distributed/`) and a
publication policy every rank shares (`merged_bloom_policy`). Every rank then:

1. builds a Bloom of one size for each key, from the plan's build estimate clamped to
   [2^16, 2^30] keys;
2. all-gathers it with a header (key rep, family, scale, size);
3. ORs the others into its own, and publishes it.

A key whose filters cannot all be merged publishes nothing: a rank sent none, a header differs,
or a rank missed the 2-minute wait. IN-lists and zone maps are not published in distributed mode.

A build split into several partitions accumulates its Bloom from every batch instead, and merges
it the same way, at the same size, so a rank on either path merges with the others.
`max_dynamic_filter_bloom_bytes_per_gpu` does not cut a distributed Bloom: the merged filter holds
every rank's keys, and one cut far below them passes almost every probe row. Right semi and anti joins plan no
BUILD_FILTER stage in distributed mode (`distribute_plan()` removes it), so they publish nothing.

A rank with an empty build still merges, contributing an empty Bloom. A rank
that never saw a whole build joins each key's all-gather without a filter, so the others drop
that key at once instead of waiting out the timeout. It does so as soon as its build side ends
(a finalize listener on the build `CONCAT` closes the join's publication), before its probe side runs, and without
waiting (`leave_all_gather`): a rank holding build rows holds its probe side until the merge
completes, and that probe side may be what the other rank's probe waits on.

These rules are the publication policy that `merged_bloom_policy()` gives a merged join:
`bloom_only` (no IN-lists), `fixed_bloom_keys` (every rank's Bloom has the same size, uncapped) and
`publish_when_empty_or_drained` (an empty build or drained targets still take part). With one
rank, all three keep their defaults and the publisher behaves as on one GPU.

A join whose build is broadcast publishes from its whole build, as on one GPU, with nothing to
merge. A join that hashes both sides and whose estimated build is under 2^14 rows publishes no
filters.
Filters stay optional and the join authoritative, so a missing filter costs speed, never
correctness.

## Settings and build

The settings are in [Distributed Execution](configuration.md#distributed-execution). The exchange
belongs to the process: with `sirius_exchange_peers` already set, `SET sirius_exchange_rank`
connects it and waits for every peer; set the other way round, the first distributed query
connects it. Once it is connected, none of the settings can change, on any connection. Set
`enable_duckdb_fallback = false` on every rank: one rank falling back to the CPU alone would
return a wrong answer.

Every Sirius build includes the NIXL fabric (NIXL with its UCX backend, built with Sirius), so a
distributed query needs no extra build option or runtime path. One command builds Sirius and the
[gateway](distributed-gateway.md):

```bash
pixi run build-distributed
```

`scripts/launcher_env.sh` holds the defaults every launcher sources. A value you export first wins.

| Variable | Default | Why |
|---|---|---|
| `UCX_TLS`, `UCX_NET_DEVICES` | `all` on the GPU rail NICs, if the host has them; else `cuda_copy,cuda_ipc,tcp,self` | Rail NICs (`sirius_rail_devices` in `scripts/sirius_net.sh`: active RDMA ports in no bond, at the highest link speed) carry UCX's notifications over RC, which the exchange over NVLink needs at full speed: 4 GPU pairs both ways moved 5.9 TB/s, against 5.2 with tcp on bond0. `UCX_NET_DEVICES=rails` asks for them. Without rail NICs, the transports tested on one host and on two. Left to itself, UCX also picks the host's InfiniBand ports, and on multi-rail GB200 hosts the ranks then fail to connect ("Endpoint timeout") |
| `DUCKDB_AGENT_MODE` | 0 | DuckDB changes its output when an AI coding tool runs it; the ranks and the scripts parse that output |

## Running TPC-H Q12 on 4 GPUs

`scripts/distributed_run_sql.sh <ranks> <file.sql> [base port] [log dir]` runs a SQL file on N
local GPUs, rank i on GPU i, and prints rank 0's output. It prepends loading the extension,
`enable_duckdb_fallback = false` and the two exchange settings. It runs on one host only and
refuses more ranks than the host has GPUs.

```bash
cat > /tmp/q12.sql <<'SQL'
create view lineitem as select * from read_parquet('/opt/sirius-ci/datasets/tpch_sf1000/lineitem/*.parquet');
create view orders as select * from read_parquet('/opt/sirius-ci/datasets/tpch_sf1000/orders/*.parquet');
.timer on
select l_shipmode,
  sum(case when o_orderpriority = '1-URGENT' or o_orderpriority = '2-HIGH' then 1 else 0 end) as high_line_count,
  sum(case when o_orderpriority <> '1-URGENT' and o_orderpriority <> '2-HIGH' then 1 else 0 end) as low_line_count
from orders, lineitem
where o_orderkey = l_orderkey and l_shipmode in ('MAIL', 'SHIP')
  and l_commitdate < l_receiptdate and l_shipdate < l_commitdate
  and l_receiptdate >= date '1994-01-01' and l_receiptdate < date '1995-01-01'
group by l_shipmode order by l_shipmode;
SQL
pixi run scripts/distributed_run_sql.sh 4 /tmp/q12.sql
```

Expected at SF1000: `MAIL 6233248 9344850` and `SHIP 6235412 9349394`, DuckDB's answer.

| 4 × GB200, Q12 SF1000 | First run | Warm run |
|---|---|---|
| Without the merged Bloom | 10.2 s | 6.55 s |
| With it (keeps 2.0% of `orders` before its exchange) | 10.0 s | 6.38 s |
| One GB200, one process | | 6.4 s |

The first run includes connecting NIXL to every peer. Q12 here is bound by the `lineitem` scan,
which every rank reads from the same shared storage, so 4 GPUs are no faster than one.

## Limits

- **Joins:** a nested-loop join that emits unmatched build rows, and a delim join whose join
  keys are not its delim columns, are refused rather than run per rank.
- **Remote files:** an S3 table needs at least as many files as there are ranks; local files
  split by byte range have no such limit (#2377).
- **Query start:** every rank must run the same statements in the same order, which is what
  numbers the exchanges' streams. So one connection per process runs distributed queries; another
  connection's distributed query is refused until that one closes. With `max_concurrent_queries`
  above 1, other connections can run local queries alongside it (#2378).
- **Unspillable buffers:** an unordered `EXCHANGE_SOURCE` queues arrivals in its port's
  repository, where they spill like any port's batches. Arrivals before the port is wired, an
  ordered gather's batches, and batches held for a stream not yet attached stay outside any
  repository and cannot spill. Sends block once a rank's queued sends exceed 1 GiB
  (`transport::options::max_queued_bytes`; #2376).
- **Blocking collectives:** `SORT_SAMPLE` and Bloom merges block a GPU worker thread until every
  rank's contribution arrives (#2375).
- **Failures:** a query that fails on one rank (an error, an exhausted retry) is aborted on every
  rank: the rank tells its peers (`exchange_service::abort`), and each fails that query's sources
  and all-gathers at once instead of waiting; the next query runs normally. A query whose
  execution window fails in `begin_execution_window()` is aborted the same way and retired on
  that rank (`distributed_session::fail_window_begin`), and so is one whose `distribute_plan()`
  throws after the query was numbered. A failure before the query is numbered tells no peer.
  A statement that fails after it was planned and before its window is tied to its query is
  not aborted either, and its peers wait out their timeouts: one cancelled
  or refused while it waits for admission, or a prepared statement whose window fails to begin.
  A lost peer or a failed transport is fail-stop: every later distributed query fails, saying
  the transport failed earlier, and nothing reconnects. To recover, restart every rank: a
  restarted rank alone cannot reconnect to peers whose transport failed, and its
  `SET sirius_exchange_rank` fails after the 2-minute connect timeout (#2373). The
  [gateway](distributed-gateway.md) does this restart.
- **Hosts:** two hosts, one GPU each, ran TPC-H Q12 at SF100 through the gateway, with UCX on
  TCP (`UCX_TLS` with `tcp`, `UCX_NET_DEVICES` set to the hosts' shared interface). The launch
  scripts set both by default: `UCX_TLS` in `launcher_env.sh`, and `UCX_NET_DEVICES` from the
  rank's address ([quick start](distributed-quickstart.md)).
  More than two hosts, and RDMA between hosts, are untested (#2372).
- **Credits:** each peer gets a fixed share of the receive ring (8 frames of 8 MiB by default).
  Clusters much larger than 8 ranks would need credits granted on demand (#2371).
- **Ring memory:** the rings take (16 + 8 x (ranks - 1)) x 8 MiB per GPU (192 MiB at 2 ranks,
  576 MiB at 8), allocated outside the GPU memory space's budget, which does not shrink for
  them. Reserving them from the space instead stalls any task whose request is clamped to the
  space's whole budget, so charging them waits for the budget to shrink with them (#2371).
- **CI:** the one-rank-per-process tests (`[nixl_multiprocess]`) need several GPUs and run only on
  developer machines (#2480).

## Tests

| Tag | Covers |
|---|---|
| `[exchange][wire]`, `[exchange][batch_io]` | Headers and refusals; batches through 64 KiB to 1 MiB frames to the GPU and host |
| `[exchange][transport]` | All-to-all on 3 and 4 ranks over the loopback fabric; one credit per peer; writes of several frames; shuffled notifications; no room; failed writes |
| `[exchange][operators]` | Hash, range, broadcast, gather, ordered gather, early arrivals, hints, failure; NULL keys to every rank; `VALUES` on rank 0 |
| `[exchange][byte_range]` | Byte spans tile every file exactly once across ranks; one file split in two; balance by bytes |
| `[exchange][collectives]` | All-gather; distributed `SORT_SAMPLE` on skewed and empty inputs; `SORT_SAMPLE` feeding a range exchange |
| `[exchange][planner]` | The plan rewrite: where exchanges go, refusals, the plan-shape check, a rank whose planning fails |
| `[exchange][distributed_session]` | The session: query numbering, windows tied to queries, connecting, a window that fails to begin |
| `[planner][one_rank_parity]` | One configured peer plans and runs like none |
| `[pipeline][finalize_listener]` | The callbacks run when an operator's pipeline ends |
| `[exchange][nixl]` | Two ranks in one process over NIXL |
| `[nixl_multiprocess]` (hidden) | One rank per process and GPU: `pixi run scripts/distributed_test_nixl.sh 4` |
| `[exchange][path]` | The path rule (NVLink domain, fabric rings, staging, `require`); a frame ring in fabric memory |
| `[nixl_bandwidth]` (hidden) | All-to-all bandwidth, one rank per process: `TEST='[nixl_bandwidth]' scripts/distributed_test_nixl.sh 4` |
| `[nixl_pairs]` (hidden) | GPU pairs on two nodes, 1 GB to 64 KB batches, both directions, every value checked: `scripts/distributed_test_nixl_pairs.sh <node1>,<node2>` |
| `[dynamic_filter][probe]` | Merging two Blooms of one capacity |

## Files

| File | Role |
|---|---|
| `src/exchange/wire.*`, `batch_io.*` | Batch layout on the wire; sending and landing a batch |
| `src/exchange/transport.*`, `messages.*` | Credits, landing, end of stream, failure |
| `src/exchange/fabric.*`, `nixl_fabric.*` | Rings; the NIXL fabric and its start-up |
| `src/exchange/exchange_service.*`, `exchange_runtime.*`, `all_gather.*` | A process's exchange; collectives |
| `src/exchange/distributed_session.*` | A process's distributed state: its exchange, query numbering, each window's query, abort and retire. `SiriusContext` creates it at the first distributed use; a process that runs none has none |
| `src/op/sirius_physical_exchange_{sink,source}.*` | The operators |
| `src/op/cross_rank_sort_boundaries.hpp`, `src/op/distributed/sample_all_gather_boundaries.*` | `SORT_SAMPLE`'s boundary interface; boundaries from every rank's sample |
| `src/planner/distributed_plan.*` | The plan rewrite and plan-shape check; each rank's share of a table's files, and the scans no rank can split |
| `src/op/dynamic_filter/cross_rank_filter_merge.hpp`, `src/op/distributed/bloom_all_gather_merge.*` | The publisher's merge interface; the Bloom merge across ranks |
| `scripts/distributed_run_sql.sh`, `distributed_test_nixl.sh` | Launchers on one host |
| `scripts/launcher_env.sh` | Defaults every launcher sources: `UCX_TLS`, `DUCKDB_AGENT_MODE` |
