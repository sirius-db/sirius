#!/usr/bin/env bash
# Starts one gateway rank on another host over SSH. The gateway runs it for every rank whose host
# is not its own (SET GLOBAL sirius_gateway_rank_launcher = '<this script>'):
#   sirius_gateway_ssh_rank.sh <host> <gpu> <quack port> <duckdb> <init.sql>
# The rank serves Quack on its host's 127.0.0.1, and the SSH session forwards the gateway's
# 127.0.0.1:<quack port> there, so Quack stays plain TCP with no certificate to pin. SSH carries the
# rank's stdin, so the rank still exits when the gateway closes it. Needs passwordless SSH, pixi on
# each host, and this checkout and the DuckDB binary at the same paths on every host. The gateway's
# UCX_*, NIXL_* and SIRIUS_* variables are passed on; set UCX_NET_DEVICES to the interface the
# hosts reach each other on.
set -euo pipefail
host=$1
gpu=$2
port=$3
duckdb=$4
init=$5
root=$(cd "$(dirname "$0")/.." && pwd)

exports=""
for name in $(compgen -e); do
  case $name in
    SIRIUS_GATEWAY_*) ;;
    UCX_* | NIXL_* | SIRIUS_*) exports+=" $name=$(printf %q "${!name}")" ;;
  esac
done

# The init file holds the rank's token: copy it over SSH instead of needing a shared directory.
# If the SSH session dies without closing stdin (the gateway killed it), the watchdog ends the rank
# so it does not keep its GPU.
remote=$(cat <<EOF
set -e
${exports:+export$exports}
export PATH="\$HOME/.pixi/bin:\$PATH" CUDA_VISIBLE_DEVICES=$(printf %q "$gpu")
cd $(printf %q "$root")
source scripts/nixl_env.sh
init=\$(mktemp /tmp/sirius_gateway_rank.XXXXXX)
printf '%s\n' $(printf %q "$(cat "$init")") > "\$init"
session=\$PPID
(
  sleep 60; rm -f "\$init"
  while kill -0 "\$session" && kill -0 \$\$; do sleep 5; done 2>/dev/null
  rm -f "\$init"
  if kill -0 \$\$ 2>/dev/null; then kill -KILL \$(pgrep -P \$\$) \$\$; fi
) </dev/null >/dev/null 2>&1 &
exec pixi run $(printf %q "$duckdb") -unsigned -init "\$init"
EOF
)

exec ssh -T -o BatchMode=yes -o ExitOnForwardFailure=yes \
  -L "127.0.0.1:$port:127.0.0.1:$port" "$host" "bash -c $(printf %q "$remote")"
