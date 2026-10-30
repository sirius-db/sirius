#!/bin/sh
# A rank launcher for tests: starts the "remote" rank on this host.
#   local_rank_launcher.sh <host> <gpu> <quack port> <duckdb> <init.sql>
echo "launcher: host $1 gpu $2 port $3"
exec "$4" -unsigned -init "$5"
