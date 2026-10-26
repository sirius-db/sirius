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
    - `duckdb_version` input in `distribution.yml` should be set to latest tagged release
    - The reusable `sirius-db/extension-ci-tools` workflow ref and `ci_tools_version` input should be updated only when the remote CI tooling branch changes
4. Check for use of the current DuckDB version in the repo to determine if they should be updated with the current PR or documented as a gap in an issue for follow up later

## Verify

Reproduce `check.yml`'s DuckDB consistency gates locally before pushing:

```bash
engine=$(git rev-parse HEAD:duckdb)
wrapper=$(git rev-parse HEAD:sirius-duckdb/duckdb)
[[ "$engine" == "$wrapper" ]] && echo "pass: engine == wrapper" || echo "FAIL: mismatch"
grep "duckdb_version: $engine" .github/workflows/distribution.yml
```
