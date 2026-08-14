# Distributed builds with the RAPIDS sccache build farm

Sirius builds already run every compilation through [sccache](https://github.com/mozilla/sccache)
(the pixi-provided Mozilla sccache 0.15.0, with a local disk cache). This page describes an
**optional, opt-in** mode that instead routes compilation through the
[RAPIDS sccache fork](https://github.com/rapidsai/sccache) and NVIDIA's RAPIDS build
infrastructure, which adds:

- a **shared S3 compilation cache** (`rapids-sccache-devs`), so object files compiled by one
  machine (or teammate) are reused by others, and
- a **distributed build farm** (`sccache-dist`), which offloads compile jobs to a cluster of
  remote build servers.

This is useful when running experiments across machines: the first build populates the shared
cache, and every subsequent build of the same sources — anywhere — mostly hits it.

Nothing changes unless you explicitly enable the mode: normal `pixi run make` builds, CI
(`ci-release`/ccache and the GitHub-hosted sccache backends), and the vcpkg presets are
untouched. The mode is enabled per shell and gated on the `SIRIUS_SCCACHE_DIST=1` environment
variable.

## Prerequisites

- Membership in the **NVIDIA GitHub org** (required for the build cluster; see
  [github-onboarding.nvidia.com](https://github-onboarding.nvidia.com/)).
- `gh` (GitHub CLI), `curl`, and `tar` on the host.
- `gh` authenticated with the `gist`, `repo`, `read:org`, and `read:enterprise` scopes:

```bash
gh auth login --web --scopes gist --scopes repo --scopes read:org --scopes read:enterprise
```

## One-time setup

```bash
scripts/sccache_dist_setup.sh
```

This is idempotent and:

1. installs the RAPIDS sccache fork into `~/.local/share/sirius/sccache-dist/bin/sccache`
   (override the location with `SIRIUS_SCCACHE_DIST_HOME`) — it does **not** replace the
   pixi-provided sccache used by normal builds;
2. checks `gh` auth and installs the `gh-nv-gha-aws` extension;
3. mints temporary AWS credentials (12h) for the shared S3 cache into a **dedicated**
   credentials file — your `~/.aws/credentials` and `~/.config/sccache` are never touched;
4. installs a **canonical pixi env** under `~/.local/share/sirius/sccache-dist/pixi/`
   (hardlinked from the shared pixi package cache, so it is cheap). Dist-mode builds compile
   through this env's compilers instead of the checkout-local `.pixi` env — see
   [Why cache keys need the canonical env](#why-cache-keys-need-the-canonical-env).

The AWS credentials expire after 12 hours. Refresh them with:

```bash
scripts/sccache_dist_setup.sh --creds-only
```

## Building through the farm

Enable the mode in the current shell, then build as usual:

```bash
source scripts/sccache_dist_env.sh
pixi run make
```

`sccache_dist_env.sh` exports `SIRIUS_SCCACHE_DIST=1` plus the `SCCACHE_*` configuration
(S3 bucket, arch-specific scheduler URL, `gh` auth token, dedicated server port 4227). The
Makefile picks up `SIRIUS_SCCACHE_DIST=1` and passes
`-DCMAKE_{C,CXX,CUDA}_COMPILER_LAUNCHER=<fork binary>` on the CMake configure command line,
which overrides the launchers baked into the presets. Toggling the mode on or off is detected
automatically and re-runs the configure step for the presets you build.

`SCCACHE_DIST_FALLBACK_TO_LOCAL_COMPILE=true` is set, so if the farm is unreachable the build
still succeeds by compiling locally (the shared S3 cache is still used).

To go back to normal local builds, open a fresh shell (the mode is scoped to the shell that
sourced the env file).

## Why cache keys need the canonical env

sccache's cache key includes a digest of the compiler binary *and its extra files* — for
conda/pixi gcc that includes the `specs` file, into which conda writes the env's **absolute
path** at install time (a link-stage `-rpath <env>/lib`). Since every checkout carries its own
`.pixi` env, compiling with the checkout-local compilers gives every checkout (and every
worktree, and every machine with a different checkout path) disjoint cache keys — 0% sharing,
even for identical sources.

Everything else is already path-independent: source paths, include paths, and the working
directory are normalized out of the key (`SCCACHE_BASEDIRS`, which the env script exports and
which requires the dist server restart the env script performs — sccache reads it in the
server process at startup), and keys are stable across sccache fork versions and dist/local
compilation. The env script also sets `SOURCE_DATE_EPOCH=0` so `__DATE__`/`__TIME__` expand
deterministically instead of baking the preprocessing wall-clock second into the key (duckdb's
`pcg_extras.hpp` expands them in every TU that includes it); dist-mode binaries therefore
report a 1970 build date.

The canonical env fixes the one remaining input: all dist-mode builds use compilers at the same
absolute path (`~/.local/share/sirius/sccache-dist/pixi/.pixi/envs/default`), so the specs file
content — and therefore the key — is identical across checkouts, worktrees, and machines
(usernames must match for `~` to expand identically). The Makefile passes the canonical
compilers via `-DCMAKE_{C,CXX,CUDA}_COMPILER` / `-DCMAKE_CUDA_HOST_COMPILER` only when dist
mode is on; normal builds keep using the checkout's own env.

Two things must still match for cache hits — both legitimate key inputs:

- the **sources** (same commit / same file contents; unchanged files like the DuckDB submodule
  hit regardless), and
- the **toolchain** (same `pixi.lock`; the env script warns when the canonical env's lock has
  drifted from the checkout's — re-run `scripts/sccache_dist_setup.sh` after lock bumps).

Note: binaries built in dist mode embed an rpath to the canonical env, so keep
`~/.local/share/sirius/sccache-dist/` around while using them.

## Health checks

```bash
pixi run make sccache-dist-status
```

This prints the farm status (`--dist-status`: number of reachable build servers and cores) and
the cache statistics (`--show-stats`: cache hit rates, S3 backend info).

## Caveats

- The env file exports `AWS_SHARED_CREDENTIALS_FILE`/`AWS_PROFILE` for sccache's S3 access.
  If you need different AWS credentials in the same shell (e.g. `make s3-test-aws`), use a
  separate shell.
- Dist mode also applies to the `ci-release` preset if you run it locally with the mode on
  (its ccache launcher gets overridden). CI itself never sets `SIRIUS_SCCACHE_DIST` and is
  unaffected.
- Compile results are cached per architecture (`x86_64` → `amd64` scheduler, `aarch64` →
  `arm64`), with Sirius-specific S3 key prefixes so they don't collide with RAPIDS builds.

## Troubleshooting

- **Auth errors / farm unreachable**: refresh the AWS creds
  (`scripts/sccache_dist_setup.sh --creds-only`) and check `gh auth status`. If your token is
  missing scopes:
  `gh auth refresh --scopes gist --scopes repo --scopes read:org --scopes read:enterprise`.
- **`gh nv-gha-aws org nvidia` fails**: NVIDIA GitHub org membership is required for the build
  cluster — see the prerequisites above.
- **Corrupted state**: remove the install directory and re-run setup:

```bash
rm -rf ~/.local/share/sirius/sccache-dist
scripts/sccache_dist_setup.sh
```
