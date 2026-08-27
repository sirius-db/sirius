#!/usr/bin/env bash
# Collect every host fact that 2NODE-BRINGUP-PLAN.md Task 0.1 needs, into a file under the
# NFS-shared repo so it can be read from either host.
#
#   ./benchmarks/collect-host-facts.sh
#
# Read-only: nothing is started, nothing is installed, no GPU is claimed. The single write is
# a touch/rm writability probe under /raid/prestouser, which is undone immediately.
#
# Run it on BOTH hosts. Output: benchmarks/host-facts-<hostname>.txt

SELF=$(hostname -s)
OUT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/host-facts-$SELF.txt

exec 3>&2            # keep the real stderr so the final line reaches the terminal
exec > "$OUT" 2>&1

say() { printf '\n===== %s =====\n' "$1"; }

echo "collected: $(date -u '+%Y-%m-%d %H:%M:%S UTC')  on: $(hostname -f 2>/dev/null || hostname)"

say "1. IDENTITY"
uname -m
uname -r
hostname -f 2>/dev/null

say "2. NETWORK (bond0 must be 10.87.140.52/27 on gcn-17, .53 on gcn-18)"
ip -br addr show bond0 2>/dev/null || echo "!! no bond0"
echo "-- all interfaces --"
ip -br addr 2>/dev/null | grep -v '^lo'
echo "-- default route --"
ip route show default 2>/dev/null

say "3. NUMA (node 0 must be cpus 0-71, node 1 cpus 72-143; HBM nodes have NO cpus)"
numactl -H 2>/dev/null | head -12 || echo "!! numactl absent"
echo "-- node sizes for 0 and 1 (LPDDR total; decides mem_limit) --"
numactl -H 2>/dev/null | grep -E '^node [01] size'

say "4. GPUS"
nvidia-smi -L 2>/dev/null || echo "!! nvidia-smi absent"
echo "-- current usage (must be idle before any bring-up) --"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null
echo "-- GPU <-> CPU affinity --"
nvidia-smi topo -m 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | head -10

say "5. ROCE HCAs (need mlx5_0/1/4/5 ACTIVE at 400 Gb/sec, link_layer Ethernet)"
ibv_devinfo -l 2>/dev/null || echo "!! ibv_devinfo absent"
echo "-- hca -> netdev mapping --"
for d in /sys/class/infiniband/*/device/net/*; do
    [ -e "$d" ] || continue
    hca=$(echo "$d" | cut -d/ -f5); net=$(basename "$d")
    printf '%s -> %s\n' "$hca" "$net"
done
echo "-- port state --"
for h in mlx5_0 mlx5_1 mlx5_4 mlx5_5; do
    p=/sys/class/infiniband/$h/ports/1
    [ -d "$p" ] || { echo "$h: ABSENT"; continue; }
    printf '%s: %s | %s | %s\n' "$h" \
        "$(cat $p/state 2>/dev/null)" "$(cat $p/rate 2>/dev/null)" \
        "$(cat $p/link_layer 2>/dev/null)"
done

say "6. MNNVL / IMEX (both nodes READY + SAME ClusterUUID => 765 GB/s path exists)"
nvidia-imex-ctl -N 2>/dev/null | tail -8 || echo "(nvidia-imex-ctl absent or not permitted)"
nvidia-smi -q 2>/dev/null | grep -A4 -i 'fabric' | head -12
echo "-- imex node map (live vs pending) --"
ls -l /etc/nvidia-imex/nodes_config.cfg* 2>/dev/null
diff -q /etc/nvidia-imex/nodes_config.cfg /etc/nvidia-imex/nodes_config.cfg.pending 2>/dev/null \
    && echo "node map: IDENTICAL (no pending reload hazard)" \
    || echo "node map: DIFFERS or unreadable -- investigate before trusting MNNVL"

say "7. STORAGE (/raid must exist, be local, be writable, have room)"
df -PT /raid 2>/dev/null || echo "!! no /raid"
df -PT /home 2>/dev/null
df -PT /opt 2>/dev/null
echo "-- writability probe under /raid/prestouser --"
if mkdir -p /raid/prestouser 2>/dev/null && touch /raid/prestouser/.wtest 2>/dev/null; then
    rm -f /raid/prestouser/.wtest; echo "RAID WRITABLE"
else
    echo "!! RAID NOT WRITABLE"
fi
echo "-- existing tpch datasets --"
ls -d /opt/sirius-ci/datasets/tpch_* 2>/dev/null || echo "(none under /opt/sirius-ci/datasets)"
ls -d /raid/prestouser/tpch/* 2>/dev/null || echo "(none under /raid/prestouser/tpch)"

say "8. LIMITS / KERNEL (BE and CN prerequisites)"
echo "ulimit -n: $(ulimit -n)"
echo "ulimit -u: $(ulimit -u)"
echo "vm.max_map_count: $(cat /proc/sys/vm/max_map_count 2>/dev/null)"
echo "somaxconn: $(cat /proc/sys/net/core/somaxconn 2>/dev/null)"
grep -E 'SwapTotal|MemTotal' /proc/meminfo 2>/dev/null

say "9. TOOLCHAIN"
echo "jdk21: $(ls -d /usr/lib/jvm/java-21-openjdk-arm64 2>/dev/null || echo MISSING)"
echo "numactl: $(command -v numactl || echo MISSING)"
echo "rsync: $(command -v rsync || echo MISSING)"
U=/home/prestouser/aocsa/tools/ucx-install/bin/ucx_info
if [ -x "$U" ]; then
    echo "ucx: $($U -v 2>/dev/null | head -1)"
    echo "UCX_MAX_RNDV_RAILS default: $($U -c 2>/dev/null | grep -i rndv_rails)"
    echo "rc_mlx5 present: $($U -d 2>/dev/null | grep -c rc_mlx5)"
else
    echo "ucx: MISSING at $U"
fi

say "10. RUNNING ENGINES (must be empty before any bring-up)"
pgrep -af 'sirius-starrocks-cn|starrocks_be|StarRocksFE' || echo "(clear)"

say "DONE"
echo "wrote: $OUT" >&3
