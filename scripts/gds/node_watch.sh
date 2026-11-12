#!/usr/bin/env bash
# node_watch.sh <probe file> <report file> <stop file>: watch one node's storage path while a GDS
# job runs on it, without a GPU, and say stop when it degrades.
#
# Every 5 s it times a 256 MiB direct read of <probe file> (a large file on the filesystem the job
# reads) and appends one line to <report file>: the read time, the nvidia-fs error counters, and
# for each storage NIC the RDMA bytes received and the error counters. It creates <stop file> and
# exits when
#   - a direct read has not returned after 10 s, or two in a row take longer than $WATCH_SLOW_S
#     (default 2 s; a healthy node needs about 0.15 s),
#   - an nvidia-fs error counter (Reads err, Mmap err) rises, or
#   - a NIC error counter (ack timeouts, retries exceeded, link down) rises.
# The caller stops the job when the stop file appears. Run it from a node-local copy with the
# report and stop files on a filesystem other than the one under test.
set -u
probe=${1:?usage: node_watch.sh <probe file> <report file> <stop file>}
report=${2:?}
stop=${3:?}
slow=${WATCH_SLOW_S:-2}
node=$(hostname -s)
nics=$(for d in /sys/class/infiniband/mlx5_*; do [ -r "$d/ports/1/counters/port_rcv_data" ] && basename "$d"; done)
say() { echo "$(date -u +%T) $node $*" >> "$report"; }
nvfs_err() { awk '/^Reads/ {r=$0} /^Mmap/ {m=$0} END {print r " | " m}' /proc/driver/nvidia-fs/stats 2> /dev/null | tr -s '\t ' ' '; }
nvfs_bad() { awk -F'err=' '/^Reads/ {split($2,a," "); r=a[1]} /^Mmap/ {split($2,b," "); m=b[1]} END {print r+m}' /proc/driver/nvidia-fs/stats 2> /dev/null; }
nic_line() {
  for n in $nics; do
    c=/sys/class/infiniband/$n/ports/1
    echo -n "$n rcv=$(cat $c/counters/port_rcv_data) xmit=$(cat $c/counters/port_xmit_data) "
  done
}
nic_bad() {
  local t=0 v f
  for n in $nics; do
    for f in local_ack_timeout_err req_transport_retries_exceeded roce_adp_retrans_to link_downed; do
      v=$(cat /sys/class/infiniband/$n/ports/1/hw_counters/$f /sys/class/infiniband/$n/ports/1/counters/$f 2> /dev/null | head -n 1)
      t=$((t + ${v:-0}))
    done
  done
  echo $t
}
trip() { say "STOP: $*"; echo "$(date -u +%T) $node $*" >> "$stop"; exit 1; }
base_nvfs=$(nvfs_bad); base_nic=$(nic_bad); slow_runs=0
say "watch started: probe=$probe slow=${slow}s nvfs_err=$base_nvfs nic_err=$base_nic | $(nvfs_err)"
while :; do
  [ -e "$stop" ] && exit 0
  t0=$(date +%s.%N)
  dd if="$probe" of=/dev/null bs=16M count=16 iflag=direct 2> /dev/null &
  dd_pid=$!
  # Polled rather than `timeout`: a read stuck in the kernel cannot be killed and would block it.
  for _ in $(seq 100); do kill -0 "$dd_pid" 2> /dev/null || break; sleep 0.1; done
  if kill -0 "$dd_pid" 2> /dev/null; then trip "a 256 MiB direct read did not return in 10 s"; fi
  took=$(awk -v a="$t0" -v b="$(date +%s.%N)" 'BEGIN {printf "%.2f", b - a}')
  # Queue pairs and memory registrations the engine holds on the RDMA devices (UCX, cuFile).
  say "read=${took}s nvfs_err=$(nvfs_bad) nic_err=$(nic_bad) duckdb_qp=$(rdma res show qp 2> /dev/null | grep -c 'comm duckdb') duckdb_mr=$(rdma res show mr 2> /dev/null | grep -c 'comm duckdb') $(nic_line)"
  if awk -v t="$took" -v s="$slow" 'BEGIN {exit !(t > s)}'; then
    slow_runs=$((slow_runs + 1))
    [ "$slow_runs" -ge 2 ] && trip "two direct reads in a row took over ${slow}s (last ${took}s)"
  else
    slow_runs=0
  fi
  [ "$(nvfs_bad)" -gt "$base_nvfs" ] && trip "an nvidia-fs error counter rose: $(nvfs_err)"
  [ "$(nic_bad)" -gt "$base_nic" ] && trip "a storage NIC error counter rose"
  sleep 5
done
