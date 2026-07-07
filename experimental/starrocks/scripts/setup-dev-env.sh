#!/usr/bin/env bash
set -euo pipefail

# Bootstraps a dev environment for the Sirius x StarRocks compute-node project
# (experimental/starrocks/). Gets a fresh checkout to the point where the pixi
# tasks in experimental/starrocks/pixi.toml work, then tells you which one to
# run next -- it does not reimplement those tasks.
#
# See experimental/starrocks/docs/onboarding.md for the architecture this
# script is the companion to.

usage() {
  cat <<'EOF'
Usage: setup-dev-env.sh [--with-fe] [--with-engine] [--full] [-h|--help]

Default (no flags): fastest, CPU-only path -- the one CI runs.
  - init the starrocks + brpc submodules
  - `pixi install -e cn`
  - fmt + clippy + `cargo test --workspace --no-default-features` as a smoke test

  --with-fe       also build the StarRocks Java frontend (`pixi run -e fe fe-build`).
                  Needs JDK/Maven via pixi; slow, no GPU required.
  --with-engine   also init the repo-root submodules needed to build libsirius
                  (duckdb, substrait, cucascade, vcpkg), then build the GPU-linked
                  engine and CN (`engine-build`, `cn-build`, `cn-test`).
                  Needs a CUDA-capable GPU + toolchain; slow.
  --full          --with-fe --with-engine (what you need to run a real FE+CN cluster).
EOF
}

with_fe=0
with_engine=0
for arg in "$@"; do
  case "${arg}" in
    --with-fe) with_fe=1 ;;
    --with-engine) with_engine=1 ;;
    --full) with_fe=1; with_engine=1 ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown option: ${arg}" >&2
      usage >&2
      exit 1
      ;;
  esac
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
starrocks_dir="$(cd "${script_dir}/.." && pwd)"    # experimental/starrocks
repo_root="$(cd "${starrocks_dir}/../.." && pwd)"

# `pixi run --manifest-path` locates the manifest/lockfile but does not change
# the task's working directory, and several pixi tasks here (e.g. `cluster`,
# `engine-build`) use paths relative to experimental/starrocks. `cd` there so
# every pixi/cargo invocation below runs with the right cwd.
cd "${starrocks_dir}"

log() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "error: '$1' is required but not found on PATH." >&2
    echo "       $2" >&2
    exit 1
  fi
}

log "Checking prerequisites"
require_cmd git "Install git: https://git-scm.com/downloads"
require_cmd pixi "Install pixi: https://pixi.sh"

log "Initializing StarRocks + BRPC submodules"
git -C "${repo_root}" submodule update --init --recursive --depth=1 \
  experimental/starrocks/starrocks experimental/starrocks/brpc

if [[ "${with_engine}" -eq 1 ]]; then
  log "Initializing engine-build submodules (duckdb, substrait, cucascade, vcpkg)"
  git -C "${repo_root}" submodule update --init --recursive --depth=1 \
    duckdb substrait cucascade vcpkg
fi

log "Installing the pixi 'cn' environment"
pixi install -e cn

log "Formatting, linting, and smoke-testing the pure-Rust CN (no GPU/engine)"
pixi run -e cn -- \
  cargo fmt --package sirius-starrocks-cn --package starrocks-plan-translator --package starrocks-thrift -- --check
pixi run -e cn -- \
  cargo clippy --all-targets --no-default-features -- -D warnings
pixi run -e cn -- \
  cargo test --workspace --no-default-features

if [[ "${with_fe}" -eq 1 ]]; then
  log "Building the StarRocks Java frontend (pixi -e fe fe-build; slow)"
  pixi run -e fe fe-build
fi

if [[ "${with_engine}" -eq 1 ]]; then
  log "Building the Sirius engine + engine-linked CN (needs a GPU; slow)"
  pixi run -e cn engine-build
  pixi run -e cn cn-build
  pixi run -e cn cn-test
fi

log "Done"
echo "Next steps (from ${starrocks_dir}):"
echo "  pixi run -e cn cargo test --workspace --no-default-features   # fast CPU loop, matches CI"
if [[ "${with_fe}" -eq 1 && "${with_engine}" -eq 1 ]]; then
  echo "  pixi run -e cn cluster                                        # run FE + CN together"
  echo "  pixi run -e client client                                     # (in another terminal) mysql CLI on the FE"
elif [[ "${with_fe}" -eq 0 && "${with_engine}" -eq 0 ]]; then
  echo ""
  echo "For a real local cluster (FE + GPU-linked CN), re-run with --full."
fi
echo "See docs/onboarding.md for the architecture and contribution guide."
