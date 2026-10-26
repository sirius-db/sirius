# Extension updating
When cloning this template, the target version of DuckDB should be the latest stable release of DuckDB. However, there
will inevitably come a time when a new DuckDB is released and the extension repository needs updating. This process goes
as follows:

## Steps

1. Bump submodules
    - `./duckdb` and `./sirius-duckdb/duckdb` should be set to latest tagged release and match (see verify below)
    - Keep related submodules such as `./substrait` and `./sirius-duckdb/vcpkg` aligned when the release requires it
2. Update the DuckDB package in `pixi.toml` to the matching release for Python users
    - Check [conda-forge channel](https://anaconda.org/channels/conda-forge/packages/python-duckdb/overview) for packages first
    - Open PRs if needed for missing/new versions [conda-forge/python-duckdb-feedstock](https://github.com/conda-forge/python-duckdb-feedstock/pulls)
3. Bump versions in `.github/workflows`
    - `duckdb_version` input in `distribution.yml` should be set to latest tagged release (e.g. `v2.0.0`)
    - The reusable `sirius-db/extension-ci-tools` workflow ref and `ci_tools_version` input should be updated only when the remote CI tooling branch changes
4. Check for use of the current DuckDB version in the repo to determine if they should be updated with the current PR or documented as a gap in an issue for follow up later

## Verify

Test `check.yml`'s DuckDB consistency gates locally before pushing:

```bash
engine=$(git rev-parse HEAD:duckdb)
wrapper=$(git rev-parse HEAD:sirius-duckdb/duckdb)
[[ "$engine" == "$wrapper" ]] && echo "PASS: engine == wrapper" || echo "FAIL: mismatch"

declared=$(grep -m1 'duckdb_version:' .github/workflows/distribution.yml | sed -E 's/.*duckdb_version:[[:space:]]*//' | tr -d '"[:space:]')
if [[ "$declared" == v* ]]; then
  git -C duckdb fetch --depth=1 origin "refs/tags/${declared}:refs/tags/${declared}"
  declared=$(git -C duckdb rev-parse "${declared}^{commit}")
fi
[[ "$declared" == "$engine" ]] && echo "PASS: distribution.yml matches" || echo "FAIL: distribution.yml mismatch (declared=$declared, engine=$engine)"
```

**NOTE:** This can fail locally if your `./duckdb` submodule's `origin` remote points somewhere
other than `.gitmodules`'s declared URL (e.g. a personal fork that doesn't mirror every upstream
tag) — `fatal: couldn't find remote ref refs/tags/vX.Y.Z`. Fix with:
```
git -C duckdb remote set-url origin https://github.com/duckdb/duckdb.git
```
