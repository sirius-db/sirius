#!/usr/bin/env bash
# One benchmark run of distributed Sirius on machines that do not share $HOME, such as AWS
# instances: a fresh gateway and one rank per GPU on this machine and on each of --nodes, every
# query twice (cold, warm), then everything is stopped. The launch is the script of
# docs/super-sirius/distributed-experiments-no-slurm.md; this adds what separate machines need.
# See docs/super-sirius/distributed-experiments-aws.md.

usage()
{
  cat <<'USAGE'
usage: sirius_aws_run.sh --name NAME --data DIR --frac FRACTION --config FILE
                         [--nodes "HOST ..."] [--ref-dir DIR] [QUERY ...]

Run it on the machine that hosts the gateway, in the Pixi environment:
  cd $WORK/sirius && pixi run bash scripts/sirius_aws_run.sh --name sf100-2m ...

  --name NAME      the run's folder is $WORK/runs/NAME, on every machine. Must be new
  --data DIR       the TPC-H dataset, one folder per table, at this path on every machine
  --frac FRACTION  Q11's fraction, 0.0001 / SF: 0.000001 at SF100, 0.0000001 at SF1000
  --config FILE    the Sirius config of every rank, copied to every machine
  --nodes HOSTS    the other machines, by private address. Needs ssh without a password from
                   here, and the checkout, the build, the data and $WORK at the same paths
  --ref-dir DIR    reference answers as q<n>.csv: compares every answer with them at the end
  QUERY ...        TPC-H query numbers (default: 1 6 12)

  WORK             as in sirius_aws_setup.sh (default /opt/dlami/nvme/work)
  STMT_TIMEOUT     seconds before the gateway cancels a statement (default 600)

Writes to $WORK/runs/NAME: client.txt (answers and times), gateway.out, node*.out,
rank-logs*/ (they contain the gateway's token: do not share them), and sampler*.txt, one line
per machine every 5 s: epoch, sectors read from all NVMe devices, bytes received, bytes sent,
used MiB,utilisation % per GPU, available GiB of RAM, bytes in $WORK/spill.
USAGE
}

sampler()
{
  while :; do
    disk=$(awk '$3 ~ /^nvme[0-9]+n[0-9]+$/ { s += $6 } END { print s + 0 }' /proc/diskstats)
    net=$(awk 'NR > 2 && $1 != "lo:" { rx += $2; tx += $10 } END { print rx + 0, tx + 0 }' /proc/net/dev)
    gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ' | paste -sd/ -)
    mem=$(awk '/MemAvailable/ { print int($2 / 1048576) }' /proc/meminfo)
    echo "$(date +%s) $disk $net $gpu $mem $(du -sb "$1" 2> /dev/null | cut -f1)"
    sleep 5
  done
}
# The other machines run the sampler from their own checkout: `<this script> --sampler DIR`.
[ "${1:-}" = --sampler ] && sampler "$2"

name='' DATA='' frac='' config='' NODES='' ref=''
while [ $# -gt 0 ]; do
  case $1 in
    -h | --help) usage && exit 0 ;;
    --name) name=${2:-} ;;
    --data) DATA=${2:-} ;;
    --frac) frac=${2:-} ;;
    --config) config=${2:-} ;;
    --nodes) NODES=${2:-} ;;
    --ref-dir) ref=${2:-} ;;
    -*) usage >&2 && exit 2 ;;
    *) break ;;
  esac
  shift 2
done
[ -n "$name" ] && [ -n "$DATA" ] && [ -n "$frac" ] && [ -r "$config" ] || {
  usage >&2
  exit 2
}
[ $# -gt 0 ] || set -- 1 6 12

WORK=${WORK:-/opt/dlami/nvme/work}
. "$WORK/env.sh" || exit 1
export CHECKOUT=$WORK/sirius DATA OUT=$WORK/runs/$name
self=$CHECKOUT/scripts/sirius_aws_run.sh
[ -e "$OUT" ] && echo "$OUT exists: use a new --name" && exit 1
if ss -ltn | grep -E ':(9494|950[0-9]|2700[0-9]) '; then
  echo "ports in use"
  exit 1
fi
ranks=$(nvidia-smi -L | wc -l)
for n in $NODES; do
  remote=$(ssh "$n" "nvidia-smi -L | wc -l") || exit 1
  ranks=$((ranks + remote))
done
mkdir -p "$OUT" && cp "$config" "$OUT/config.yaml" || exit 1
q=$CHECKOUT/test/tpch_performance/tpch_queries/orig
{
  printf '.mode csv\n.timer on\n'
  for t in customer lineitem nation orders part partsupp region supplier; do
    echo "CREATE OR REPLACE VIEW $t AS FROM read_parquet('$DATA/$t/*.parquet');"
  done
  for n in "$@"; do
    for run in 1 2; do
      echo "SELECT 'Q$n run $run' AS marker;"
      sed "s/\* 0\.00001$/* $frac/" "$q/q$n.sql"
    done
  done
  echo "FROM sirius_gateway_status();"
} > "$OUT/run.sql"
for n in $NODES; do
  ssh "$n" "mkdir -p $OUT" && scp -q "$OUT/config.yaml" "$n:$OUT/" || exit 1
done

bash "$self" --sampler "$WORK/spill" > "$OUT/sampler.txt" 2> /dev/null &
samp=$!
for n in $NODES; do
  ssh "$n" "nohup bash $self --sampler $WORK/spill > $OUT/sampler.txt 2> /dev/null &"
done

export SIRIUS_CLUSTER=bench SIRIUS_GATEWAY_MIN_RANKS=$ranks
export SIRIUS_CONFIG_FILE=$OUT/config.yaml SIRIUS_RANK_LOG_ROOT=$OUT/rank-logs
export SIRIUS_PRIORITY_DOWNSTREAM_FIRST=1 SIRIUS_GATEWAY_STATEMENT_TIMEOUT=${STMT_TIMEOUT:-600}
export SIRIUS_GATEWAY_JOIN_WAIT=120
state=$HOME/.sirius/gateways/$SIRIUS_CLUSTER
# The remote pkill pattern also matches, and stops, the remote sampler.
stop()
{
  for n in $NODES; do ssh "$n" "pkill -u \$USER -f '$CHECKOUT/script[s]/'; rm -rf $state"; done
  kill $samp $(jobs -p) 2> /dev/null
  wait
  rm -rf "$state"
}
trap 'exit 1' INT TERM
trap stop EXIT
cd "$CHECKOUT" && git log -1 --oneline > "$OUT/env.txt"
rm -rf "$state"
# Non-interactive ssh does not read ~/.bashrc, hence env.sh; the SIRIUS_ variables travel too.
on_node="cd $CHECKOUT && . $WORK/env.sh && env $(env | grep '^SIRIUS_' | tr '\n' ' ') $(command -v pixi) run scripts/sirius_node.sh"
scripts/sirius_gateway.sh --serve > "$OUT/gateway.out" 2>&1 &
gateway=$!
until [ -s "$state/gateway.conf" ]; do
  kill -0 $gateway 2> /dev/null || {
    echo "no gateway: see $OUT/gateway.out"
    exit 1
  }
  sleep 1
done
# Each machine has its own $HOME: its ranks read the gateway's address and token from this copy.
for n in $NODES; do
  ssh "$n" "mkdir -p $state && chmod 700 $state" && scp -q -p "$state/gateway.conf" "$n:$state/" || exit 1
done
scripts/sirius_node.sh --check --data "$DATA" | grep -v '^ok' || exit 1
for n in $NODES; do ssh "$n" "$on_node --check --data $DATA" | grep -v '^ok' || exit 1; done
scripts/sirius_node.sh > "$OUT/node.out" 2>&1 &
for n in $NODES; do ssh "$n" "$on_node" > "$OUT/node-$n.out" 2>&1 & done
# no -bail: a failed query must not hide the ones after it
scripts/sirius_client.sh < "$OUT/run.sql" > "$OUT/client.txt" 2>&1 &
wait $! || echo "the client failed: see $OUT/client.txt"
sleep 5
for n in $NODES; do
  scp -q -r "$n:$OUT/rank-logs" "$OUT/rank-logs-$n" 2> /dev/null
  scp -q "$n:$OUT/sampler.txt" "$OUT/sampler-$n.txt"
done
echo "--- $name: errors $(grep -c Error "$OUT/client.txt"), ranks up $(grep -c ',up,ready,' "$OUT/client.txt") of $ranks"
tr -d '\r' < "$OUT/client.txt" |
  awk '/^Q[0-9]+ run [12]$/ { q = $0; r = $3; n = 0 }
       /^Run Time/ && q != "" && ++n == 2 { printf "%s %s\n", q, $5; t[r] += $5; q = "" }
       END { print "TOTAL cold", t[1], "warm", t[2] }' | paste - - | column -t
grep -m8 'Error' "$OUT/client.txt" | cut -c1-300
[ -z "$ref" ] || python3 scripts/compare_answers.py "$OUT/client.txt" --ref-dir "$ref" --ref-pattern "q{n}.csv" \
  --ties 3:revenue,o_orderdate --ties 18:o_totalprice,o_orderdate | tee "$OUT/compare.txt"
