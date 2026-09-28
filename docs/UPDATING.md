# Extension updating
When cloning this template, the target version of DuckDB should be the latest stable release of DuckDB. However, there
will inevitably come a time when a new DuckDB is released and the extension repository needs updating. This process goes
as follows:

- Bump submodules
  - `./duckdb` should be set to latest tagged release
  - Keep related submodules such as `./duckdb-python`, `./substrait`, and `./vcpkg` aligned when the release requires it
- Bump versions in `.github/workflows`
  - `duckdb_version` input in `distribution.yml` should be set to latest tagged release
  - The reusable `sirius-db/extension-ci-tools` workflow ref and `ci_tools_version` input should be updated only when the remote CI tooling branch changes
- Set `OVERRIDE_GIT_DESCRIBE` in `cmake/CMakePresets.json` to the current DuckDB version `vX.Y.Z`
  - This should match the tag at the base of the branch for the submodule
