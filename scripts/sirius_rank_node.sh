#!/usr/bin/env bash
# Runs one Sirius rank for a distributed gateway, and starts it again whenever it exits: when the
# gateway restarts its ranks (sirius_rank_exit), or after a crash. Start one per GPU on each node,
# outside the gateway (systemd, srun, tmux):
#   pixi run scripts/sirius_rank_node.sh <gpu> <quack port> <exchange port> \
#     [listen host=$(hostname -f)] [dir=/tmp/sirius_rank_<quack port>]
# It prints the sirius_gateway_add_rank call that registers the rank. SIRIUS_RANK_TOKEN sets the
# token; otherwise a random one is kept in <dir>/token. On a listen host other than this machine's
# loopback, Quack serves TLS with a key pair kept in <dir>/tls, whose fingerprint the gateway pins.
# With every rank on one host, set UCX_TLS=cuda_copy,cuda_ipc,tcp,self, as distributed_run_sql.sh
# does: left to itself, UCX picks InfiniBand ports that GB200 hosts cannot connect through.
# Needs a Sirius build with -DSIRIUS_WITH_NIXL=ON.
set -euo pipefail
gpu=$1
port=$2
exchange_port=$3
host=${4:-$(hostname -f)}
dir=${5:-/tmp/sirius_rank_$port}
here=$(cd "$(dirname "$0")" && pwd)
source "$here/nixl_env.sh"
build="$here/../sirius-duckdb/build/release"
duckdb="$build/duckdb"
export CUDA_VISIBLE_DEVICES=$gpu
mkdir -p -m 700 "$dir"

token=${SIRIUS_RANK_TOKEN:-}
if [ -z "$token" ]; then
  [ -s "$dir/token" ] || (umask 077 && head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n' > "$dir/token")
  token=$(cat "$dir/token")
fi
serve="token := '$token'"
pin=
case $host in
  localhost | 127.0.0.1 | ::1) ;;
  *)
    # One key pair for every restart, so the fingerprint the gateway pins stays valid.
    fingerprint=$(env -u AI_AGENT "$duckdb" -noheader -list \
      -c "SELECT fingerprint FROM quack_generate_keys(directory := '$dir/tls')")
    serve+=", allow_other_hostname := true, ssl_cert_file := '$dir/tls/server.pem'"
    serve+=", ssl_key_file := '$dir/tls/private_key.pem'"
    pin=", fingerprint := '$fingerprint'"
    ;;
esac
(
  umask 077
  cat > "$dir/init.sql" <<SQL
LOAD '$build/extension/sirius/sirius.duckdb_extension';
CALL quack_serve('quack:$host:$port', $serve);
SQL
)
echo "CALL sirius_gateway_add_rank('$host:$port', exchange_port := $exchange_port, token := '$token'$pin);" >&2

# A rank reads its stdin until EOF; a FIFO opened read-write never reaches it.
[ -p "$dir/stdin" ] || mkfifo -m 600 "$dir/stdin"
child=
trap '[ -n "$child" ] && kill "$child" 2> /dev/null; exit 0' TERM INT
backoff=1
while :; do
  "$duckdb" -unsigned -init "$dir/init.sql" <> "$dir/stdin" >> "$dir/rank.log" 2>&1 &
  child=$!
  rc=0
  wait "$child" || rc=$?
  # A rank that crashed leaves its shared-memory mirrors behind.
  rm -f /dev/shm/sirius-exchange-"$child"-*
  if [ "$rc" -eq 0 ]; then
    backoff=1
    continue
  fi
  echo "rank exited with $rc; starting it again in ${backoff}s" >&2
  sleep "$backoff"
  backoff=$((backoff < 30 ? backoff * 2 : 30))
done
