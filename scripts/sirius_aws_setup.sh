#!/usr/bin/env bash
# Prepares one AWS GPU instance for distributed Sirius: work folders on the instance store, Pixi,
# Rust, the checkout, the build, and optionally a TPC-H dataset. Run it on every machine of a
# cluster: each needs the same checkout, build and data at the same paths.
# See docs/super-sirius/distributed-experiments-aws.md.
set -uo pipefail

usage()
{
  cat <<'USAGE'
usage: sirius_aws_setup.sh [SCALE_FACTOR]

Builds distributed Sirius in $WORK/sirius and, with a scale factor, generates TPC-H as Parquet
in $WORK/tpch_sf<SCALE_FACTOR>. Safe to run again: it skips what is already installed or cloned.
Takes tens of minutes: run it in a detached tmux session.

  WORK    where everything goes (default /opt/dlami/nvme/work, the instance store of the
          Deep Learning AMI). The instance store is erased when the instance stops.
  BRANCH  the branch to clone (default experimental/distributed-sirius)

Progress is in $WORK/logs/steps.log; build.log, datagen.log and their exit codes (build.rc,
datagen.rc) are next to it. $WORK/env.sh is the environment every later command sources.
USAGE
}

case ${1:-} in
  -h | --help) usage && exit 0 ;;
esac
sf=${1:-}
[[ $sf =~ ^[0-9]*$ ]] || {
  usage >&2
  exit 2
}
export WORK=${WORK:-/opt/dlami/nvme/work}
branch=${BRANCH:-experimental/distributed-sirius}

sudo mkdir -p "$WORK" && sudo chown "$USER" "$WORK"
mkdir -p "$WORK"/{cache/pixi,cache/rattler,logs,runs,spill,duckdb_tmp,ref}
step() { echo "$(date -u +%T) $*" >> "$WORK/logs/steps.log"; }
step start
command -v pixi > /dev/null || [ -x "$HOME/.pixi/bin/pixi" ] || curl -fsSL https://pixi.sh/install.sh | bash
cat > "$WORK/env.sh" <<EOF
export WORK=$WORK PIXI_CACHE_DIR=$WORK/cache/pixi RATTLER_CACHE_DIR=$WORK/cache/rattler
export CARGO_HOME=$WORK/cargo RUSTUP_HOME=$WORK/rustup
export PATH=\$HOME/.pixi/bin:\$CARGO_HOME/bin:\$PATH
EOF
. "$WORK/env.sh"
# The dataset generator needs cargo.
[ -x "$CARGO_HOME/bin/cargo" ] || curl -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path
cd "$WORK" || exit 1
[ -d sirius ] || git clone --branch "$branch" https://github.com/sirius-db/sirius.git || {
  step "clone FAILED"
  exit 1
}
cd sirius && git submodule update --init --recursive || {
  step "submodules FAILED"
  exit 1
}
step "build begin $(git log -1 --format=%h)"
pixi run build-distributed > "$WORK/logs/build.log" 2>&1
rc=$?
echo $rc > "$WORK/logs/build.rc"
step "build rc=$rc"
[ $rc -eq 0 ] || exit $rc
[ -n "$sf" ] || exit 0
step "datagen begin sf=$sf"
pixi run bash test/tpch_performance/generate_tpch_data.sh "$sf" --format parquet \
  --output "$WORK/tpch_sf$sf" > "$WORK/logs/datagen.log" 2>&1
rc=$?
echo $rc > "$WORK/logs/datagen.rc"
step "datagen rc=$rc"
exit $rc
