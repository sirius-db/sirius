#!/bin/bash
# gds_probe.sbatch: the first GDS reads on a node, smallest first, outside the engine.
#
# Steps, each only if the one before passed and the filesystem still answers:
#   a  one 16 MiB read, compared byte for byte with a POSIX read
#   b  1 GiB on 1 thread
#   c  32 GiB on 8 threads
#   d  60 s on 32 threads
#   e  60 s on 32 threads on each of the node's 4 GPUs at once
# FILES="f0 f1 f2 f3" gives each GPU its own file (all read FILE otherwise).
# Stop after a step with LAST=a..e (default e). A step passes when gds_probe exits 0: the reads
# returned, cuFile counted no POSIX read, and for (a) the bytes matched.
#
# Tell the admins before running this, and use one exclusive node. Submit from a folder that is
# not on the filesystem under test, so the output survives if that filesystem stops:
#   cd /home/$USER/gds-t1 && FILE=/scratch/.../lineitem/part.0.parquet \
#     sbatch -w <node> -o gds-probe-%j.out <checkout>/scripts/gds/gds_probe.sbatch
#SBATCH -p nvidia-part
#SBATCH --job-name=gds-probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=144
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=20
set -u
FILE=${FILE:?set FILE to a data file of at least 1 GiB on the filesystem to test}
CHECKOUT=${CHECKOUT:?set CHECKOUT to the built Sirius checkout}
LAST=${LAST:-e}
work=/tmp/gds-probe-$SLURM_JOB_ID
out=$PWD
report=$out/gds-probe-$SLURM_JOB_ID.watchdog
mkdir -p "$work"
# Node-local copies: the probe and the watchdog must not need the filesystem under test.
cp "$CHECKOUT/build/release/test/io/gds_probe" "$CHECKOUT/scripts/gds/fs_watchdog.sh" "$work/" || exit 1
CUFILE_ENV_PATH_JSON=$("$CHECKOUT/scripts/gds/cufile_json.sh" "$work/cufile") || exit 1
export CUFILE_ENV_PATH_JSON
export LD_LIBRARY_PATH=$CHECKOUT/.pixi/envs/default/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
cd "$work"
echo "job $SLURM_JOB_ID node $(hostname -s) file $FILE checkout $(git -C "$CHECKOUT" log -1 --oneline)"
grep -E '^(GDS Version|NVFS Driver|Reads|Mmap)' /proc/driver/nvidia-fs/stats

healthy() {
  timeout 5 stat "$FILE" > /dev/null 2>&1 &&
    timeout 5 dd if="$FILE" of=/dev/null bs=1M count=1 iflag=direct 2> /dev/null
}
# step <name> <gpus> <gds_probe arguments...>: one probe per GPU, watched.
step() {
  local name=$1 gpus=$2 rc=0 pids=() gpu file
  shift 2
  echo "== step $name: gds_probe $* (GPUs $gpus) $(date -u +%T)"
  for gpu in $gpus; do
    file=$FILE
    [ -n "${FILES:-}" ] && file=$(echo $FILES | cut -d' ' -f$((gpu + 1)))
    CUDA_VISIBLE_DEVICES=$gpu ./gds_probe --file "$file" "$@" > "step-$name-gpu$gpu.txt" 2>&1 &
    pids+=($!)
  done
  bash ./fs_watchdog.sh "$FILE" "$report" "${pids[@]}" &
  local watchdog=$!
  for pid in "${pids[@]}"; do wait "$pid" || rc=$?; done
  wait "$watchdog" || rc=1
  for gpu in $gpus; do sed "s/^/gpu$gpu: /" "step-$name-gpu$gpu.txt"; done
  grep -E '^(Reads|Mmap)' /proc/driver/nvidia-fs/stats
  [ "$rc" = 0 ] || { echo "STEP $name FAILED (exit $rc)"; return 1; }
  healthy || { echo "STEP $name: filesystem does not answer afterwards"; return 1; }
  echo "step $name passed"
}
finish() {
  mkdir -p "$out/gds-probe-$SLURM_JOB_ID.cufile"
  cp -r "$work"/cufile/* "$work"/*.log "$out/gds-probe-$SLURM_JOB_ID.cufile/" 2> /dev/null
  rm -rf "$work"
  exit "$1"
}
healthy || { echo "filesystem does not answer before the first read; not starting"; finish 1; }
step a 0 --bytes $((16 << 20)) --verify || finish 1
[ "$LAST" = a ] && finish 0
step b 0 --bytes $((1 << 30)) || finish 1
[ "$LAST" = b ] && finish 0
step c 0 --bytes $((32 << 30)) --threads 8 || finish 1
[ "$LAST" = c ] && finish 0
step d 0 --seconds 60 --threads 32 || finish 1
[ "$LAST" = d ] && finish 0
step e "0 1 2 3" --seconds 60 --threads 32 || finish 1
echo "all steps passed"
finish 0
