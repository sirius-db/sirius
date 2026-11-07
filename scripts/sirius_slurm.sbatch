#!/usr/bin/env bash
# Starts a distributed gateway and one Sirius rank per GPU on every node of a Slurm job:
#   sbatch -N2 scripts/sirius_slurm.sbatch      (from the Sirius checkout)
#   SIRIUS_CLUSTER=<job id> pixi run sirius-client
# Each task is one rank on one GPU: --ntasks-per-node is the number of GPUs used on each node.
# The gateway is named after the job, so jobs never share one. It serves until the job ends.
# Needs $HOME shared by the nodes (the gateway writes ~/.sirius/gateways/<job id>/gateway.conf).
#SBATCH --job-name=sirius
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"
cd "$SLURM_SUBMIT_DIR"
[ -f scripts/sirius_node.sh ] || {
  echo "submit from the Sirius checkout: sbatch scripts/sirius_slurm.sbatch" >&2
  exit 1
}
export SIRIUS_CLUSTER=$SLURM_JOB_ID SIRIUS_GATEWAY_MIN_RANKS=$SLURM_NTASKS
state=$HOME/.sirius/gateways/$SIRIUS_CLUSTER
conf=$state/gateway.conf
rm -f "$conf"

# No GPU, and --overlap so the rank step can still take every CPU and GPU of its node.
srun -N1 -n1 --gpus=0 --overlap pixi run scripts/sirius_gateway.sh --serve &
gateway=$!
nodes=
# The gateway lives as long as the job: its state (token, keys, DDL log) goes with it, and a
# gateway.conf left behind would point clients at a gateway that is gone.
trap 'kill "$gateway" $nodes 2> /dev/null || true; rm -rf "$state"' EXIT
until [ -s "$conf" ]; do
  kill -0 "$gateway" 2> /dev/null || {
    echo "the gateway exited before it wrote $conf" >&2
    exit 1
  }
  sleep 1
done
echo "gateway $SIRIUS_CLUSTER is up; clients: SIRIUS_CLUSTER=$SIRIUS_CLUSTER pixi run sirius-client" >&2

# One shell per task, so SLURM_LOCALID is the task's own; --gpus slurm keeps the task's GPU.
# shellcheck disable=SC2016 # SLURM_LOCALID expands in each task's shell
srun --kill-on-bad-exit=1 bash -c 'exec pixi run scripts/sirius_node.sh --gpus slurm \
  --quack-port $((9500 + SLURM_LOCALID)) --exchange-port $((27000 + SLURM_LOCALID))' &
nodes=$!
# Neither step ends on its own: the ranks retry a dead gateway forever. When one step exits, end
# the job (the EXIT trap stops the other), so that a failed gateway, for example one whose port
# another job holds, or a crashed one, does not keep every GPU until the time limit.
rc=0
wait -n "$gateway" "$nodes" || rc=$?
if kill -0 "$gateway" 2> /dev/null; then
  echo "the ranks' step exited with $rc; stopping the gateway" >&2
else
  echo "the gateway exited with $rc; stopping the ranks" >&2
fi
exit 1
