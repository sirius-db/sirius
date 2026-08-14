# =============================================================================
# Copyright 2025, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================
#
# Source this file to route Sirius builds through the RAPIDS sccache
# distributed build farm (see docs/sccache-dist.md):
#
#   source scripts/sccache_dist_env.sh
#   pixi run make
#
# Run scripts/sccache_dist_setup.sh once first. Everything here is scoped to
# the current shell — open a fresh shell to go back to normal local builds.

if [[ "${BASH_SOURCE[0]:-}" == "$0" ]]; then
  echo "this file must be sourced, not executed: source ${0}" >&2
  exit 1
fi

_sirius_dist_home="${SIRIUS_SCCACHE_DIST_HOME:-$HOME/.local/share/sirius/sccache-dist}"
_sirius_dist_bin="$_sirius_dist_home/bin/sccache"
_sirius_dist_creds="$_sirius_dist_home/aws-credentials"

if [[ ! -x "$_sirius_dist_bin" || ! -f "$_sirius_dist_creds" ]]; then
  echo "sccache dist farm is not set up yet — run: scripts/sccache_dist_setup.sh" >&2
  return 1
fi

if [[ -n "$(find "$_sirius_dist_creds" -mmin +720 2>/dev/null)" ]]; then
  echo "WARNING: AWS credentials are older than 12h and have likely expired." >&2
  echo "         Refresh with: scripts/sccache_dist_setup.sh --creds-only" >&2
fi

if ! _sirius_dist_token="$(gh auth token 2>/dev/null)"; then
  echo "gh is not authenticated — run scripts/sccache_dist_setup.sh" >&2
  return 1
fi

case "$(uname -m)" in
  x86_64) _sirius_dist_arch=amd64 ;;
  aarch64) _sirius_dist_arch=arm64 ;;
  *)
    echo "unsupported architecture for the sccache dist farm: $(uname -m)" >&2
    return 1
    ;;
esac

# Consumed by the Makefile: switches the CMake compiler launchers to the
# RAPIDS sccache fork binary below.
export SIRIUS_SCCACHE_DIST=1
export SIRIUS_SCCACHE_DIST_BIN="$_sirius_dist_bin"

# Canonical pixi env (see setup script): compiling through compilers at this
# fixed path — instead of the checkout-local .pixi env — is what makes cache
# keys match across checkouts, worktrees, and machines.
_sirius_canon_env="$_sirius_dist_home/pixi/.pixi/envs/default"
_sirius_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -x "$_sirius_canon_env/bin/nvcc" ]]; then
  export SIRIUS_SCCACHE_DIST_CANON_ENV="$_sirius_canon_env"
  if ! cmp -s "$_sirius_dist_home/pixi/pixi.lock" "$_sirius_repo_root/pixi.lock"; then
    echo "WARNING: canonical env's pixi.lock differs from this checkout's — builds may use" >&2
    echo "         stale toolchains. Refresh with: scripts/sccache_dist_setup.sh" >&2
  fi
else
  echo "WARNING: canonical pixi env not found — cache keys will be checkout-specific and" >&2
  echo "         cross-machine cache hits will not occur. Fix with: scripts/sccache_dist_setup.sh" >&2
fi
unset _sirius_canon_env _sirius_repo_root

# Dedicated server port so the fork's local server never collides with the
# pixi-provided sccache 0.15.0 server (default port 4226) used by normal builds.
export SCCACHE_SERVER_PORT="${SIRIUS_SCCACHE_DIST_PORT:-4227}"

# Shared RAPIDS S3 compilation cache.
export SCCACHE_BUCKET="rapids-sccache-devs"
export SCCACHE_REGION="us-east-2"
export SCCACHE_S3_KEY_PREFIX="sirius-${_sirius_dist_arch}"
export SCCACHE_S3_USE_PREPROCESSOR_CACHE_MODE=true
export SCCACHE_S3_PREPROCESSOR_CACHE_KEY_PREFIX="sirius-${_sirius_dist_arch}-preprocessor-cache"
export AWS_SHARED_CREDENTIALS_FILE="$_sirius_dist_creds"
export AWS_PROFILE=default

# Distributed compilation cluster (arch-specific scheduler).
export SCCACHE_DIST_SCHEDULER_URL="https://${_sirius_dist_arch}.linux.sccache.rapids.nvidia.com"
export SCCACHE_DIST_AUTH_TYPE=token
export SCCACHE_DIST_AUTH_TOKEN="$_sirius_dist_token"
# Compile locally instead of failing when the farm is unreachable.
export SCCACHE_DIST_FALLBACK_TO_LOCAL_COMPILE=true

# Make __DATE__/__TIME__ expand deterministically (they otherwise bake the
# preprocessing wall-clock second into cache keys via headers like duckdb's
# pcg_extras.hpp, permanently missing across machines). Built binaries report
# a 1970 build date in dist mode.
export SOURCE_DATE_EPOCH=0

# Strip checkout-local prefixes (sources, .pixi env) from hashed paths and
# preprocessor output. pixi activation exports the same value inside
# `pixi run`, but sccache reads it in the *server* process at startup — so
# export it here too and restart the server below to make it stick.
_sirius_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SCCACHE_BASEDIRS="$_sirius_repo_root:$_sirius_repo_root/.pixi/envs/default"
unset _sirius_repo_root

# The dist-mode server (port 4227) caches its config at startup; restart it so
# the exports above (basedirs, S3 creds) apply to the next build. The normal
# build's sccache server (port 4226) is unaffected.
"$_sirius_dist_bin" --stop-server >/dev/null 2>&1 || true

unset _sirius_dist_home _sirius_dist_creds _sirius_dist_token _sirius_dist_arch

echo "RAPIDS sccache dist farm enabled for this shell (launcher: $_sirius_dist_bin)"
echo "Build with: pixi run make    Check farm health with: pixi run make sccache-dist-status"
unset _sirius_dist_bin
