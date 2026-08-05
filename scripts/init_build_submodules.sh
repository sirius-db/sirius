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
# Initialize the submodules a build needs, and give the DuckDB submodule its version tag.
#
# DuckDB derives its version from `git describe --tags` at configure time. A shallow submodule
# carries no tags, so the describe fails and DuckDB falls back to the dummy version v0.0.1
# (duckdb/CMakeLists.txt: "likely due to shallow clone ... Continuing with dummy version v0.0.1").
#
# That dummy version is not cosmetic. It becomes the extension directory name, so a build made
# this way looks for extensions in ~/.duckdb/extensions/v0.0.1/, and there is no upstream
# extension repository at that version for INSTALL to resolve against. Anything that needs a
# DuckDB extension — avro and iceberg, which the iceberg tests LOAD — cannot work. It passes
# locally, where the submodule was cloned with tags, and fails only in CI.
#
# Fetching the one tag that names the pinned commit costs a single ref and keeps the clone
# shallow. The tag is derived from the pin rather than hardcoded, so a DuckDB bump needs no
# edit here.

set -euo pipefail

git submodule update --init --depth=1 --jobs 3 duckdb substrait cucascade

pinned_sha="$(git -C duckdb rev-parse HEAD)"

# `ls-remote --tags` lists an annotated tag twice: the tag object, and the commit it peels to as
# `<tag>^{}`. Match on either and strip the peel suffix.
tag="$(git -C duckdb ls-remote --tags origin |
  awk -v sha="$pinned_sha" '$1 == sha { print $2 }' |
  sed 's#^refs/tags/##; s#\^{}$##' |
  sort -u | head -1)"

if [[ -z "$tag" ]]; then
  echo "::error::The duckdb submodule is pinned to ${pinned_sha}, which no tag on" \
    "https://github.com/duckdb/duckdb points at." >&2
  echo "DuckDB would build as the dummy version v0.0.1, whose extension directory has no" \
    "upstream repository, so avro and iceberg could be neither installed nor loaded." >&2
  echo "Pin the submodule to a release tag, or set OVERRIDE_GIT_DESCRIBE for the build." >&2
  exit 1
fi

git -C duckdb fetch --depth=1 origin "refs/tags/${tag}:refs/tags/${tag}"

echo "duckdb submodule pinned at ${tag} (describe: $(git -C duckdb describe --tags --long))"
