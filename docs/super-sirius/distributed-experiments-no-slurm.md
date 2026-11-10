# Distributed experiments without Slurm

How to run a distributed Sirius benchmark on a machine Slurm does not schedule, such as a
standalone NVL4 node (4 GPUs), from an empty account to checked numbers. On a node Slurm
schedules, follow [Distributed experiments](distributed-experiments.md) instead: a process
started outside Slurm there can get the node drained.

One script starts a gateway and one rank per GPU, on one machine or on several, sends the SQL
through the client, and stops everything. CPU fallback is off: a query Sirius cannot run on the
GPU fails. Every step below was run from an empty account on 2026-11-10, on one machine and on
two (8 ranks).

## 1. Requirements

- Git, [Pixi](https://pixi.sh/latest/installation/) 0.78 or newer, `nvidia-smi` and `ss` on
  `PATH`. What the machine needs is in [docs/README.md](../README.md#requirements).
- About 8 GB free in `$HOME` (Pixi's package cache; `PIXI_CACHE_DIR` moves it) and 14 GB for
  the checkout.
- The dataset as Parquet, one folder per table: `lineitem/*.parquet`, `orders/*.parquet`, ...

## 2. Build

```bash
export CHECKOUT=<where the checkout goes> DATA=<the dataset> OUT=<a new run folder>
git clone --branch experimental/distributed-sirius https://github.com/sirius-db/sirius.git "$CHECKOUT"
cd "$CHECKOUT" && git submodule update --init --depth=1 --jobs 5 \
  duckdb substrait cucascade sirius-duckdb/duckdb sirius-duckdb/extension-ci-tools
pixi run build-distributed            # once: NIXL, Sirius with NIXL, the gateway
```

The three paths must be absolute. The build takes 35 to 40 minutes on 144 cores. `pixi run` may
warn that its cache is on a network file system: ignore it, here and in every output file.

## 3. Check the machine

```bash
ls -d "$CHECKOUT/build/release/libsirius.so" "$DATA/lineitem"       # prints both paths
nvidia-smi -q | grep -A4 "^ *Fabric$"                               # State: Completed, per GPU
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader   # prints nothing
ss -ltn | grep -E ':(9494|950[0-9]|2700[0-9]) '                     # prints nothing
```

The last two show that no process holds a GPU and that the ports are free: the gateway takes
9494, and rank i takes 9500 + i and 27000 + i. If they print something, see
[When it goes wrong](#when-it-goes-wrong). With several machines, run the four checks on each.

## 4. Config and queries

Use a new `$OUT` for every run: the rank logs append, the other files are overwritten.

```bash
mkdir -p "$OUT/spill"
cat > "$OUT/config.yaml" << EOF
sirius:
  memory:
    gpu:
      usage_limit_fraction: 0.8841
      downgrade_trigger_fraction: 0.9089
      downgrade_stop_fraction: 0.6786
    host: { capacity_bytes: 200Gi }
    disk: { disk_id: 0, capacity_bytes: 1000000000000, downgrade_root_dirs: "$OUT/spill" }
  executor:
    pipeline: { num_threads: 7 }
    task_creator: { num_threads: 1 }
    downgrade: { num_threads: 2 }
    scan_manager: { num_threads: 9, uring_n_reactors: 1, cache: { mode: none } }
  operator_params:
    scan_task_batch_size: 901Mi
    hash_partition_bytes: 1067Mi
    concat_batch_bytes: 2552Mi
    sort_sample_bytes: 201Mi
    max_build_hash_table_bytes: 6358Mi
    max_build_probe_resident_bytes: 16Gi
    max_sort_partition_memory_fraction: 0.4330
    mark_join_build_switch_ratio: 10.1054
    dynamic_filter_domain_coverage_threshold: 0.6257
    dynamic_filter_keep_threshold: 0.8144
    max_dynamic_filter_bloom_bytes_per_gpu: 2577Mi
EOF
```

This is the SF10000 config of [Config](distributed-experiments.md#config), which also explains
the keys and has the one for larger scales. It runs SF100 unchanged.

`run.sql` creates a view per table and runs each query twice, here five queries:

```bash
frac=0.00000001                                         # Q11's fraction, 0.0001 / SF: SF100 0.000001
q=$CHECKOUT/test/tpch_performance/tpch_queries/orig
{
  printf '.mode csv\n.timer on\n'
  for t in customer lineitem nation orders part partsupp region supplier; do
    echo "CREATE OR REPLACE VIEW $t AS FROM read_parquet('$DATA/$t/*.parquet');"
  done
  for n in 1 3 6 11 12; do                              # $(seq 1 22) for all
    for run in 1 2; do
      echo "SELECT 'Q$n run $run' AS marker;"
      sed "s/\* 0\.00001$/* $frac/" "$q/q$n.sql"        # only Q11 has that line
    done
  done
  echo "FROM sirius_gateway_status();"                  # every rank up, no last_error
} > "$OUT/run.sql"
```

Start narrow: the few queries the question is about. "Cold" is a query's first run, "warm" the
second; the first query of a script is slower still, so keep the query order when comparing
runs. On a machine other people use, repeat a run before quoting it: the same query took 6.4 to
9.5 s across runs here.

## 5. Run

Save this script as `$OUT/run.sh`. One run of it is one arm, with fresh ranks.

```bash
: "${CHECKOUT:?} ${DATA:?} ${OUT:?}"                      # exported, absolute paths
NODES=""                                                  # other machines, as "node2 node3": see More machines
export SIRIUS_CLUSTER=bench SIRIUS_GATEWAY_MIN_RANKS=4    # the ranks of all machines together
export SIRIUS_CONFIG_FILE=$OUT/config.yaml SIRIUS_RANK_LOG_ROOT=$OUT/rank-logs
export SIRIUS_PRIORITY_DOWNSTREAM_FIRST=1 SIRIUS_GATEWAY_STATEMENT_TIMEOUT=300
export SIRIUS_GATEWAY_JOIN_WAIT=120                       # seconds the first statement waits for the ranks
state=$HOME/.sirius/gateways/$SIRIUS_CLUSTER
if ss -ltn | grep -E ':(9494|950[0-9]|2700[0-9]) '; then   # an old gateway or rank, or another user's
  echo "ports in use"; exit 1
fi
cd "$CHECKOUT" && git log -1 --oneline > "$OUT/env.txt"
rm -rf "$state"                                           # or the gateway replays the last run's views
on_node="cd $CHECKOUT && env $(env | grep '^SIRIUS_' | tr '\n' ' ') $(command -v pixi) run scripts/sirius_node.sh"
stop()
{
  for n in $NODES; do ssh "$n" "pkill -u \$USER -f '$CHECKOUT/script[s]/'"; done
  kill $(jobs -p) 2> /dev/null; wait; rm -rf "$state"
}
trap 'exit 1' INT TERM
trap stop EXIT                                            # stops everything, also on Ctrl-C
scripts/sirius_gateway.sh --serve > "$OUT/gateway.out" 2>&1 &
gateway=$!
until [ -s "$state/gateway.conf" ]; do
  kill -0 $gateway 2> /dev/null || { echo "no gateway: see $OUT/gateway.out"; exit 1; }
  sleep 1
done
scripts/sirius_node.sh --check --data "$DATA" || exit 1
for n in $NODES; do ssh "$n" "$on_node --check --data $DATA" || exit 1; done
scripts/sirius_node.sh > "$OUT/node.out" 2>&1 &           # one rank per GPU
for n in $NODES; do ssh "$n" "$on_node" > "$OUT/node-$n.out" 2>&1 & done
scripts/sirius_client.sh -bail < "$OUT/run.sql" > "$OUT/client.txt" 2>&1 &
wait $! || echo "the client failed: see $OUT/client.txt"  # waited for, so that Ctrl-C stops the script at once
sleep 5                                                   # the ranks flush their logs every 3 s
```

Run it in a shell that has `CHECKOUT`, `DATA` and `OUT` exported:

```bash
cd "$CHECKOUT" && pixi run bash "$OUT/run.sh"
```

The script prints the node check (`ok` lines, then `node ready for gateway bench`) and returns
when the client is done; the ranks need about 10 s to start. `-bail` stops the client at the
first failed statement. Ctrl-C, or `kill` of the script, stops the gateway and every rank within
seconds.

| File in `$OUT` | What |
|---|---|
| `client.txt` | For each run: the marker, the answer, then the query's `Run Time` (the second after the marker). Lines end in CR. Large when a query returns many rows: 423 MB with Q11 at SF10000 |
| `rank-logs/<host>-<port>/` | One Sirius log per rank. They contain the gateway's token: remove it before sharing them |
| `node.out`, `node-<machine>.out`, `gateway.out` | Launch output: `registered with the gateway`, `rank exited` |
| `env.txt` | The commit |

Launch files also stay in `/tmp/sirius_node_<user>_bench` and `/tmp/sirius_gateway_<user>_bench`
(each rank's `rank.log`, and the token): delete them when done.

## 6. Check the result

```bash
grep -c "Error" "$OUT/client.txt"                                   # 0
grep -c ",up,ready," "$OUT/client.txt"                              # the number of ranks
nvidia-smi --query-compute-apps=pid --format=csv,noheader           # nothing: every rank is gone
grep -h "exchange: rank .* -> rank" "$OUT"/rank-logs/*/sirius_*.log |
  grep -c "NIXL over the NVLink fabric"                             # ranks x (ranks - 1): 12 with 4
tr -d '\r' < "$OUT/client.txt" |                                    # seconds per run
  awk '/^Q[0-9]+ run [12]$/ { q = $0; n = 0 } /^Run Time/ && q != "" && ++n == 2 { print q, $5; q = "" }'
```

Fewer `NIXL over the NVLink fabric` lines than that means a pair of ranks exchanges over the
network.

Then compare every answer with an independent engine on the same data. The build's shell,
started without Sirius, is a plain DuckDB on the CPU:

```bash
sed '$d' "$OUT/run.sql" > "$OUT/ref.sql"                            # without the gateway status
cd "$CHECKOUT" && pixi run sirius-duckdb/build/release/duckdb -no-agent < "$OUT/ref.sql" > "$OUT/ref.txt" 2>&1
python3 scripts/compare_answers.py "$OUT/client.txt" --ref-client "$OUT/ref.txt" \
  --ties 3:revenue,o_orderdate --ties 18:o_totalprice,o_orderdate
```

It prints one line per run and exits 1 unless every run is `match` or `match up to a LIMIT
tie`. Rows compare in any order, numbers within `1e-6` relative or `0.01` absolute; a text
`diff` does not work, because float digits and the order of equal rows differ. Q3 and Q18 end
in `ORDER BY ... LIMIT` and may return different rows that tie at the cut: `--ties` names their
order columns. At SF10000 the plain DuckDB needs about 15 minutes for 22 queries and 1 TB of
memory; with answers kept as one CSV file per query, use `--ref-dir DIR --ref-pattern
"q{n:02d}.csv"` instead of `--ref-client`.

Report the dataset's layout with every number
([Dataset layout](distributed-experiments.md#check-the-run); run its query in the plain DuckDB),
and what [Report](distributed-experiments.md#report) lists.

## Variations

- **Fewer GPUs:** in `run.sh`, set `SIRIUS_GATEWAY_MIN_RANKS=2` and add `--gpus 0,1` to both
  `scripts/sirius_node.sh` lines.
- **Other ports**, when someone else holds the defaults: add
  `export SIRIUS_GATEWAY_PORT=19494 SIRIUS_NODE_QUACK_PORT=19500 SIRIUS_NODE_EXCHANGE_PORT=37000`
  to the top of `run.sh` (rank i adds i to the last two), and use
  `':(19494|1950[0-9]|3700[0-9]) '` in its `ss` line and in step 3.
- **More machines:** in `run.sh`, set `NODES` to the other machines and
  `SIRIUS_GATEWAY_MIN_RANKS` to the ranks of all machines together (8 for two machines with 4
  GPUs each). The script then checks each machine, starts its ranks over `ssh`, and stops them
  at the end. It needs `ssh` without a password, and on every machine the same `$HOME`, and the
  checkout, the data, `$OUT` and Pixi at the same paths. A machine with another `$HOME` takes
  `--gateway` and `--token-file` instead ([Quick Start](distributed-quickstart.md#prerequisites)).

## When it goes wrong

| Symptom | Cause and fix |
|---|---|
| `ports in use`, after `LISTEN` lines | A gateway or rank of an earlier run that was killed, or another user. Yours: [stop them](#stop-what-a-killed-run-left). Another user's: other ports |
| `no gateway: see .../gateway.out` | That file says why: `config.yaml` is not valid YAML, `gateway bench already runs`, or `port 9494 on ... is in use` (another gateway holds it) |
| A `FAIL` line from the node check | The line under it names the fix |
| `client.txt`: `the gateway starts once 4 ranks are registered ...; 3 are` | Fewer ranks than `SIRIUS_GATEWAY_MIN_RANKS` joined within `SIRIUS_GATEWAY_JOIN_WAIT`. `rank exited` in `node.out` means a rank dies at start (usually the config): read `/tmp/sirius_node_<user>_bench/rank_<port>/rank.log` |
| The script was killed with `kill -9` | Its gateway and ranks still run and hold the GPUs: [stop them](#stop-what-a-killed-run-left) |
| A statement hangs | The statement timeout (300 s here) cancels it and the next statement restarts the ranks |

Failures of the queries themselves (ranks killed mid-query, Q3 and Q18 ties, Q11's fraction) are
in [Known failures](distributed-experiments.md#known-failures).

### Stop what a killed run left

```bash
pkill -u "$USER" -f "$CHECKOUT/script[s]/"; sleep 5      # the gateway, the client, every rank
pkill -9 -u "$USER" -f "$CHECKOUT/script[s]/"            # whatever did not stop
```

Then repeat the last two checks of step 3; with several machines, do both on each. The `[s]`
keeps the command from matching, and killing, the shell that runs it.
