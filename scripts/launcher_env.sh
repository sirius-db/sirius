# Defaults for every Sirius launcher. Source it (under set -euo pipefail too). Each value you
# export first wins.
# The transports tested on one host and on two. Left to itself, UCX also picks the host's
# InfiniBand ports, and on multi-rail GB200 hosts the ranks then fail to connect ("Endpoint
# timeout").
export UCX_TLS=${UCX_TLS:-cuda_copy,cuda_ipc,tcp,self}
# DuckDB changes its output when an AI coding tool runs it; the ranks and the scripts parse it.
export DUCKDB_AGENT_MODE=${DUCKDB_AGENT_MODE:-0}
