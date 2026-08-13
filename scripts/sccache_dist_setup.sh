#!/usr/bin/env bash
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
# One-time (and credential-refresh) setup for building Sirius against the
# RAPIDS sccache distributed build farm. See docs/sccache-dist.md.
#
# Everything is installed under a dedicated directory
# (default: ~/.local/share/sirius/sccache-dist, override with
# SIRIUS_SCCACHE_DIST_HOME). This script deliberately does NOT touch
# ~/.aws/credentials, ~/.config/sccache, the pixi environment, or anything
# else the normal (non-distributed) build relies on.
#
# Usage:
#   scripts/sccache_dist_setup.sh               # full setup (idempotent)
#   scripts/sccache_dist_setup.sh --creds-only  # only refresh the AWS creds (they expire after 12h)
#   scripts/sccache_dist_setup.sh --force       # re-download the sccache fork binary

set -euo pipefail

DIST_HOME="${SIRIUS_SCCACHE_DIST_HOME:-$HOME/.local/share/sirius/sccache-dist}"
DIST_BIN="$DIST_HOME/bin/sccache"
CREDS_FILE="$DIST_HOME/aws-credentials"
DIST_PORT="${SIRIUS_SCCACHE_DIST_PORT:-4227}"
CREDS_DURATION=43200 # 12 hours (maximum allowed)

FORCE=0
CREDS_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    --creds-only) CREDS_ONLY=1 ;;
    -h|--help)
      sed -n '16,29p' "$0"
      exit 0
      ;;
    *)
      echo "unknown argument: $arg (try --help)" >&2
      exit 1
      ;;
  esac
done

die() { echo "sccache_dist_setup: $*" >&2; exit 1; }

for tool in curl tar gh; do
  command -v "$tool" >/dev/null || die "'$tool' is required but not on PATH"
done

mkdir -p "$DIST_HOME/bin"

# -----------------------------------------------------------------------------
# 1) Install the RAPIDS sccache fork (not upstream Mozilla sccache — the fork
#    carries the distributed-compilation features the farm requires). Kept
#    separate from the pixi-provided sccache 0.15.0 used by normal builds.
# -----------------------------------------------------------------------------
if [[ "$CREDS_ONLY" == 0 ]]; then
  if [[ -x "$DIST_BIN" && "$FORCE" == 0 ]]; then
    echo "==> RAPIDS sccache fork already installed: $DIST_BIN ($("$DIST_BIN" --version))"
  else
    arch="$(uname -m)"
    [[ "$arch" == "x86_64" || "$arch" == "aarch64" ]] || die "unsupported architecture: $arch"
    url="https://github.com/rapidsai/sccache/releases/latest/download/sccache-${arch}-unknown-linux-musl.tar.gz"
    echo "==> Downloading RAPIDS sccache fork: $url"
    tmpdir="$(mktemp -d)"
    trap 'rm -rf "$tmpdir"' EXIT
    curl -fsSL "$url" | tar -xz -C "$tmpdir"
    unpacked="$(find "$tmpdir" -type f -name sccache | head -n 1)"
    [[ -n "$unpacked" ]] || die "no 'sccache' binary found in the release tarball"
    install -m 0755 "$unpacked" "$DIST_BIN"
    echo "==> Installed $DIST_BIN ($("$DIST_BIN" --version))"
  fi
fi

# -----------------------------------------------------------------------------
# 2) GitHub auth — the farm authenticates with a gh token that needs the
#    gist, repo, read:org and read:enterprise scopes, and requires membership
#    in the NVIDIA GitHub org (see https://github-onboarding.nvidia.com/).
# -----------------------------------------------------------------------------
gh auth status >/dev/null 2>&1 || die "gh is not authenticated. Run:
  gh auth login --web --scopes gist --scopes repo --scopes read:org --scopes read:enterprise"

scopes="$(gh auth status 2>&1 | grep -i 'token scopes' || true)"
for scope in gist repo read:org read:enterprise; do
  if [[ -n "$scopes" && "$scopes" != *"'$scope'"* && "$scopes" != *" $scope"* ]]; then
    echo "WARNING: gh token may be missing the '$scope' scope. If auth to the farm fails, re-run:" >&2
    echo "  gh auth refresh --scopes gist --scopes repo --scopes read:org --scopes read:enterprise" >&2
    break
  fi
done

# -----------------------------------------------------------------------------
# 3) gh-nv-gha-aws extension (mints the temporary AWS creds for the shared
#    S3 cache bucket).
# -----------------------------------------------------------------------------
if ! gh extension list 2>/dev/null | grep -q "nv-gha-aws"; then
  echo "==> Installing gh extension nv-gha-runners/gh-nv-gha-aws"
  gh extension install nv-gha-runners/gh-nv-gha-aws
fi

# -----------------------------------------------------------------------------
# 4) Temporary AWS credentials for the shared RAPIDS sccache S3 bucket.
#    Written to a dedicated file (NOT ~/.aws/credentials);
#    scripts/sccache_dist_env.sh points AWS_SHARED_CREDENTIALS_FILE at it.
# -----------------------------------------------------------------------------
echo "==> Fetching temporary AWS credentials (valid for 12h)"
gh nv-gha-aws org nvidia \
  --profile default \
  --output creds-file \
  --duration "$CREDS_DURATION" \
  --aud sts.amazonaws.com \
  --idp-url https://token.gha-runners.nvidia.com \
  --role-arn arn:aws:iam::279114543810:role/nv-gha-token-sccache-devs \
  > "$CREDS_FILE.tmp"
chmod 600 "$CREDS_FILE.tmp"
mv "$CREDS_FILE.tmp" "$CREDS_FILE"
echo "==> Wrote $CREDS_FILE"

# Stop any running dist-mode sccache server so the next build picks up the
# fresh credentials. Normal builds use a different port (4226) and are
# unaffected.
if [[ -x "$DIST_BIN" ]]; then
  SCCACHE_SERVER_PORT="$DIST_PORT" "$DIST_BIN" --stop-server >/dev/null 2>&1 || true
fi

cat <<EOF

Setup complete. To build through the farm (per shell):

  source scripts/sccache_dist_env.sh
  pixi run make

The AWS credentials expire after 12 hours; refresh them with:

  scripts/sccache_dist_setup.sh --creds-only

EOF
