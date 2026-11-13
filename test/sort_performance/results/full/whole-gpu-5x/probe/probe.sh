B=/mnt/disk_2/sort_merge_bench/whole-gpu-5x; mkdir -p $B/probe
printf 'sirius:\n  memory:\n    host:\n      capacity_bytes: 103079215104\n' > $B/base_config.yml
for L in 103079215104 102642155520 102005473280 100931731456; do
  echo "### probe limit $L"
  python test/sort_performance/sort_merge_bench.py --device 0 --spill-dir /mnt/disk_2/sirius_spill --base-config $B/base_config.yml --gpu-limit-bytes $L --scale 0.02 --datasets uniform-1key --modes windowed --compression false --out-dir $B/probe/$L 2>&1 | grep -v "^\[.*\]   \["
  awk -F, 'NR==2{print "status="$10" wall="$11" peak="$14}' $B/probe/$L/results.csv
  tail -c 600 $B/probe/$L/runs/*/child_output.txt; echo
  grep -h "exceeds\|capacity" $B/probe/$L/runs/*/sirius_log_tail.txt | head -5 | cut -c1-300
  if awk -F, 'NR==2{exit !($10=="ok")}' $B/probe/$L/results.csv; then echo "PROBE_OK $L"; break; fi
done
