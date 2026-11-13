#!/bin/bash
# Second pass of driver.sh, after the std::bad_cast fix in src/pipeline/gpu_pipeline_task.cpp:
# the one_shot cells that ended with that error, run again on the fixed code. Same settings as
# the `fits-gpu` and `twice-gpu` steps of driver.sh.
set -uo pipefail

OUT=/mnt/disk_2/sort_merge_bench/followup
HERE=test/sort_performance/results/followup
BENCH="python -B test/sort_performance/sort_merge_bench.py --device 0 --spill-dir /mnt/disk_2/sirius_spill"

step() { # step <name> <harness args...>
  local name=$1
  shift
  rm -rf /mnt/disk_2/sirius_spill/sort_bench_*
  echo "=== $name: $(date -Is) free: $(df -h --output=avail /mnt/disk_2 | tail -1)"
  $BENCH --out-dir "$OUT/$name" "$@" > "$OUT/$name.console.txt" 2>&1
  echo "=== $name: rc=$? $(date -Is)"
}

step fits-gpu-rerun --gpu-limit-bytes 85899345920 --scale 0.4 --modes one_shot \
  --datasets normal-multikey,sequential --compression false --repeat 3 \
  --base-config $HERE/host_cap_96g.yml --timeout-s 900

step twice-gpu-rerun --gpu-limit-bytes 85899345920 --scale 2 --modes one_shot \
  --compression false --datasets uniform-1key,zipf-1key \
  --base-config $HERE/host_cap_96g.yml --timeout-s 1500

rm -rf /mnt/disk_2/sirius_spill/sort_bench_*
echo "=== done: $(date -Is) free: $(df -h --output=avail /mnt/disk_2 | tail -1)"
