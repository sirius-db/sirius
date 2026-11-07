#!/usr/bin/env bash
# Opens a DuckDB shell whose every statement goes to a distributed gateway:
#   sirius_client.sh [--cluster NAME] [duckdb arguments...]
# The gateway's address, token and TLS fingerprint come from ~/.sirius/gateways/<cluster>/
# gateway.conf (cluster: --cluster, else $SIRIUS_CLUSTER, else default). Other arguments go to
# duckdb, e.g. -c 'FROM sirius_gateway_status()' to run one statement and exit.
set -euo pipefail
# Defaults from the distributed: section of sirius.yaml; exported variables win. Captured first:
# eval would ignore the script's failure and run with half the file applied.
defaults=$(python3 "$(dirname "$0")/sirius_distributed_config.py") || exit 1
eval "$defaults"
cluster=${SIRIUS_CLUSTER:-default}
case ${1:-} in
  -h | --help)
    sed -n '2,6s/^# \{0,1\}//p' "$0"
    exit 0
    ;;
  --cluster)
    cluster=$2
    shift 2
    ;;
esac
conf=$HOME/.sirius/gateways/$cluster/gateway.conf
[ -r "$conf" ] || {
  echo "no $conf: start the gateway first (pixi run sirius-gateway)" >&2
  exit 1
}
gateway=
gateway_token=
# shellcheck source=/dev/null
source "$conf"
here=$(cd "$(dirname "$0")" && pwd)
tls=
[ -n "${gateway_fingerprint:-}" ] && tls=", SSL_FINGERPRINT '$gateway_fingerprint'"
# The ATTACH holds the token, so it goes in a file only this user reads, next to gateway.conf
# (which holds the same token), never in duckdb's argv where ps shows it to every user. Every
# client of the gateway writes the same text, and the rename keeps it whole for one starting now.
# -init replaces ~/.duckdbrc, so the file reads that first.
init=${conf%/*}/client.init.sql
(
  umask 077
  {
    [ ! -r "$HOME/.duckdbrc" ] || echo ".read '$HOME/.duckdbrc'"
    # Quack's TLS client stops every call at http_timeout, 30 s by default. Only CONNECT, not
    # USE, sends statements through the gateway's router.
    echo "SET GLOBAL http_timeout = 86400;"
    echo "ATTACH 'quack:$gateway' AS sirius (TOKEN '$gateway_token'$tls);"
    echo "CONNECT sirius;"
  } > "$init.$$"
  mv -f "$init.$$" "$init"
)
exec "$here/../gateway/build/release/duckdb" -unsigned -no-agent -init "$init" "$@"
