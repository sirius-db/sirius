#!/bin/bash
# Follow-up measurements after the phase-6 integration, run on the workstation from the mirror
# root through `.ai-helper/workstation.sh bash test/sort_performance/results/followup/driver.sh`.
# One harness invocation per step; a step that fails does not stop the next one.
set -uo pipefail

OUT=/mnt/disk_2/sort_merge_bench/followup
HERE=test/sort_performance/results/followup
BENCH="python -B test/sort_performance/sort_merge_bench.py --device 0 --spill-dir /mnt/disk_2/sirius_spill"
mkdir -p "$OUT"

step() { # step <name> <harness args...>
  local name=$1
  shift
  rm -rf /mnt/disk_2/sirius_spill/sort_bench_*
  echo "=== $name: $(date -Is) free: $(df -h --output=avail /mnt/disk_2 | tail -1)"
  $BENCH --out-dir "$OUT/$name" "$@" > "$OUT/$name.console.txt" 2>&1
  echo "=== $name: rc=$? $(date -Is)"
}

# 1. The smoke matrix again, on the integrated code: order check on, 12 GiB per dataset.
step smoke-post --gpu-limit-bytes 4294967296 --scale 3 --validate

# 2. Order check at a size that reaches the disk tier and intermediate passes:
#    40 GiB per dataset, 4 GiB GPU limit, 4 GiB host tier.
step validate-disk --gpu-limit-bytes 4294967296 --scale 10 --validate \
  --base-config $HERE/host_cap_4g.yml --modes windowed \
  --datasets uniform-1key,zipf-1key,normal-multikey --timeout-s 3600

# 3. Data that fits the GPU: 32 GiB under an 80 GiB limit (the device keeps about 15 GiB of
#    headroom), every mode, three repeats.
step fits-gpu --gpu-limit-bytes 85899345920 --scale 0.4 --modes one_shot,windowed,auto \
  --compression false --repeat 3 --base-config $HERE/host_cap_96g.yml --timeout-s 900

# 4. A small multiple of the GPU: 160 GiB under the same 80 GiB limit, every mode, one repeat.
step twice-gpu --gpu-limit-bytes 85899345920 --scale 2 --modes windowed,auto,one_shot \
  --compression false --datasets uniform-1key,zipf-1key \
  --base-config $HERE/host_cap_96g.yml --timeout-s 1500

rm -rf /mnt/disk_2/sirius_spill/sort_bench_*
echo "=== done: $(date -Is) free: $(df -h --output=avail /mnt/disk_2 | tail -1)"
