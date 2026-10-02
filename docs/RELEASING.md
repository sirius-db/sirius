# Releasing Sirius

Sirius publishes two rolling GitHub Releases, `stable` and `latest`, via
[`.github/workflows/release.yml`](../.github/workflows/release.yml). Neither is a versioned
release; both are mutable snapshots that get replaced in place.

- **`stable`**: a maintainer-selected build, promoted manually.
- **`latest`**: tracks the newest successful build on `main` automatically, no maintainer action
  needed in normal operation.

## Promoting `stable`

1. Pick a commit SHA. The usual criterion is the most recent commit whose build passed the
   nightly benchmark suite (external to this repo, not something `release.yml` checks).
2. Confirm it already has a successful, `push`-triggered
   [`Distribution` run on `main`](https://github.com/sirius-db/sirius/actions/workflows/distribution.yml?query=branch%3Amain).
   The release workflow only ever consumes an existing build, it never triggers one. If none
   exists for that SHA, re-run the existing Distribution run, or pick a different commit.
3. Go to **Actions → Release → Run workflow**, select `mode: stable`, paste the full 40-character
   commit SHA into `sha`, and dispatch.
4. The workflow validates the SHA (format, ancestor-of-`main`), stages the new assets on a
   scratch release, verifies checksums, then swaps them onto the live `stable` release without
   ever taking it offline.

## `latest`

Publishes automatically on every successful, `push`-triggered `Distribution` completion on
`main`, no dispatch needed. To manually recover (e.g. after a failed automatic publish), dispatch
`mode: latest` with no `sha` input, it always resolves to the current `main` HEAD.

A built-in staleness check skips the publish (not an error) if the target commit isn't newer
than, or the same as, whatever `latest` currently points at, this is what makes it safe for a
manually re-run old `Distribution` build to never regress `latest` backward.

## Troubleshooting

- **"No successful push-triggered Distribution run found for `<sha>`"**: the target commit's
  `Distribution` build hasn't finished yet, or never ran as a `push` build on `main`. Check
  [recent `Distribution` runs](https://github.com/sirius-db/sirius/actions/workflows/distribution.yml?query=branch%3Amain) for that commit. CUDA
  builds can take over an hour, this is expected if you dispatch right after a merge.
- **A dispatch 403s with "Resource not accessible by integration"**: the `SIRIUS_RELEASE_TOKEN`
  repo secret (a fine-grained PAT, `Contents: Read and write` only) is missing, expired, or scoped
  incorrectly. The default `GITHUB_TOKEN` doesn't have write access to Releases in this repo, so
  `SIRIUS_RELEASE_TOKEN` is required for the release-mutating steps. The token for this secret is
  managed by the `siriusdbbot` GitHub user.
- **A dispatch or the `pull_request` dry-run skips/fails with no clear reason**: check the job's
  `if:` condition in the Actions log, for `workflow_run` events specifically, only a successful,
  `push`-triggered `Distribution` run on `main` actually publishes, everything else is a
  deliberate no-op.

## Changelog fallback behavior

The `latest` changelog uses `stable..<sha>` when a `stable` tag exists, and falls back to the
last 50 commits of the entire repo history when it doesn't. If `stable` is ever deleted, promote
it again before the next `latest` publish so the changelog stays meaningful.
