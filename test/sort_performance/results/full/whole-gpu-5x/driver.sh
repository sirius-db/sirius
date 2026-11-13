set -u
B=/mnt/disk_2/sort_merge_bench/whole-gpu-5x; L=100931731456; SP=/mnt/disk_2/sirius_spill
mkdir -p $B/cells
printf 'sirius:\n  memory:\n    host:\n      capacity_bytes: 103079215104\n' > $B/base_config.yml
run() { # dataset comp mode timeout
  local out=$B/cells/$1__comp-$2__$3
  if [ -f $out/results.csv ]; then echo "### skip (already done) $out"; return; fi
  rm -rf $SP/sort_bench_* 2>/dev/null
  echo "### $(date +%F\ %T) run $1 comp=$2 mode=$3 timeout=$4 | $(df -BG --output=avail /mnt/disk_2 | tail -1) avail | mem avail $(free -g | awk 'NR==2{print $7}')G | gpu0 $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 0)"
  python test/sort_performance/sort_merge_bench.py --device 0 --spill-dir $SP --base-config $B/base_config.yml \
    --gpu-limit-bytes $L --scale 5 --datasets $1 --compression $2 --modes $3 --timeout-s $4 --log-tail-lines 400 --out-dir $out 2>&1 | grep -v '^\[.*\]   \['
  echo "### rc=${PIPESTATUS[0]} $(awk -F, 'NR==2{print "status="$10" wall_s="$11" peak_gpu_mib="$14" spill_host="$15" spill_disk="$16" oom_retries="$20}' $out/results.csv)"
}
for ds in ${DATASETS:-uniform-1key zipf-1key sequential few_distinct-strings normal-multikey}; do
  for comp in false true; do
    [ -f $B/STOP ] && { echo "### STOP file found"; exit 0; }
    grep -qx "$ds $comp" $B/SKIP 2>/dev/null && { echo "### skipping cell $ds $comp (SKIP file)"; continue; }
    run $ds $comp windowed 10800
    w=$(awk -F, 'NR==2{printf "%d", $11+0}' $B/cells/${ds}__comp-${comp}__windowed/results.csv)
    t=$(( w * 4 )); [ $t -lt 1800 ] && t=1800
    run $ds $comp one_shot $t
  done
done
rm -rf $SP/sort_bench_* 2>/dev/null
echo "### $(date +%F\ %T) all cells done"; df -h /mnt/disk_2 | tail -1
