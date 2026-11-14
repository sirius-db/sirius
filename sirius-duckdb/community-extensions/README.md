# Community extension descriptors

Drafts of the `description.yml` files that publish the Sirius backends in
[duckdb/community-extensions](https://github.com/duckdb/community-extensions):

| Draft | Upstream |
| --- | --- |
| `sirius_cuda12/description.yml` | `extensions/sirius_cuda12/description.yml` |
| `sirius_cuda13/description.yml` | `extensions/sirius_cuda13/description.yml` |

Upstream is authoritative. These copies stage documentation changes; they are
not read by any build.

## What renders on duckdb.org

The page at `duckdb.org/community_extensions/extensions/<name>` shows only
`extension.description`, `extension.maintainers`, `docs.hello_world` and
`docs.extended_description` (Markdown) from the descriptor. It also shows a generic
`INSTALL <name> FROM community; LOAD <name>;` block, which is why
`extended_description` tells users to `INSTALL sirius; LOAD sirius;` instead.
`docs.hardware_requirements`, `license` and `version` are not rendered.

The function and setting tables come from a `LOAD` on a runner without a GPU, so
they cannot list Sirius's functions and settings; `extended_description` covers
the important ones.

## Syncing

Refresh the drafts from upstream before editing, so build fields (`repo.ref`,
`cuda_archs`, `vcpkg_commit`, `extra_extension_config`, ...) stay current:

```sh
for ext in sirius_cuda12 sirius_cuda13; do
  gh api "repos/duckdb/community-extensions/contents/extensions/$ext/description.yml" \
    --jq .content | base64 -d > "/tmp/$ext.yml"
  diff "/tmp/$ext.yml" "sirius-duckdb/community-extensions/$ext/description.yml"
done
```

Keep the two drafts identical apart from the backend name, CUDA version, GPU
architectures and driver requirements (`diff` them). Check every name, default and
example against the code: the loader (`sirius-duckdb/loader/src/sirius_loader.cpp`),
the backend's settings and functions (`src/sirius_extension.cpp`) and the build
architectures (`CUDAARCHS` in `pixi.toml`, `cuda_archs` in
`.github/workflows/distribution.yml`).

The drafts document the code on `main`, but the community build compiles `repo.ref`.
Before publishing, bump `repo.ref` to a commit that contains everything the drafts
describe, and make sure DuckDB's pin of the core `sirius` loader
(`.github/config/extensions/sirius.cmake` in duckdb/duckdb) includes the loader
behavior they describe.

To publish, copy the files into a fork of duckdb/community-extensions and open a pull
request there.
