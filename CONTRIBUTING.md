# Contributing to Sirius

## Getting started

**Prerequisites:** [pixi](https://prefix.dev/) for environment management.

```bash
git clone --recurse-submodules <repo>
cd sirius
pixi shell                              # activate environment
CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) make
```

If you create a new git worktree, initialize submodules manually:
```bash
git submodule update --init --recursive
```

## Running tests

```bash
# C++ unit tests
build/release/extension/sirius/test/cpp/sirius_unittest

# Run a specific tag or test name
build/release/extension/sirius/test/cpp/sirius_unittest "[cpu_cache]"
build/release/extension/sirius/test/cpp/sirius_unittest "test_cpu_cache_basic_string_single_col"

# SQL logic tests (end-to-end)
make test
```

Test logs are written to `build/release/extension/sirius/test/cpp/log/`.

## Code style

Sirius uses pre-commit hooks for formatting and linting. Install them once after cloning:

```bash
pre-commit install
```

To run all checks manually:

```bash
pre-commit run -a
```

Tools enforced: `clang-format` (C++/CUDA), `black` (Python), `cmake-format`, `codespell`.

## Submodules

The `duckdb/`, `duckdb-python/`, and `vcpkg/` directories are third-party submodules. Their `CONTRIBUTING.md` files apply to contributing to those upstream projects, not to Sirius. Do not modify submodule contents directly.

## Troubleshooting CI failures

When the C++ unit test job fails or times out, the workflow automatically uploads log artifacts to GitHub Actions. Download them from the **Summary** tab of the failed run.

### Available artifacts

| Artifact | Contents | Retained |
|---|---|---|
| `unittest-logs-<run_id>` | Sirius log files from the failing test run | 14 days |
| `tpch-run-<run_id>` | TPC-H benchmark comparison and validation CSV | 14 days |

### How to download

1. Open the failed GitHub Actions run
2. Scroll to the **Artifacts** section at the bottom of the Summary tab
3. Click the artifact name to download a zip
4. The unit test log is at `sirius_<YYYY-MM-DD>.log` inside the zip

### What to look for in `sirius_<date>.log`

- The last test number logged before output stops — this is the test that hung or crashed
- Any CUDA error messages preceding the hang
- Stack traces if the binary aborted

## Pull requests

### Commit and title convention

Sirius squash-merges PRs, the PR titles become the commit message on merge. Commits and PR titles follow [Conventional Commits](https://www.conventionalcommits.org/) format for readability.

Example of Conventional Commits:
```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Scope is optional — use it when the change is clearly localized to a subsystem (e.g. `fix(join):`, `ci(distribution):`), omit it for cross-cutting changes.

New contributors: put your best title — reviewers will help refine it before merge.

### Configuration changes

If your PR changes any Sirius configuration option:
1. Document the change inline where the config lives (code comments, settings files)
2. Summarize the change in the PR description for reviewers and the changelog
