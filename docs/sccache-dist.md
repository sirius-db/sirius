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
   credentials file — your `~/.aws/credentials` and `~/.config/sccache` are never touched.

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
