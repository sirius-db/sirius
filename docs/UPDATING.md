# Extension updating 
When cloning this template, the target version of DuckDB should be the latest stable release of DuckDB. However, there 
will inevitably come a time when a new DuckDB is released and the extension repository needs updating. This process goes
as follows:

- Bump submodules
  - `./duckdb` should be set to latest tagged release
  - `./extension-ci-tools` should be set to updated branch corresponding to latest DuckDB release
- Bump versions in `./github/workflows`
  - `duckdb_version` input in `MainDistributionPipeline.yml` should be set to latest tagged release
  - reusable workflow `_extension_distribution.yml` should be set to updated branch corresponding to latest DuckDB release

