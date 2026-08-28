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

Tools enforced: `clang-format` (C++/CUDA), `black` (Python), `cmake-format`, `codespell`, and
`rumdl` (Markdown links and anchors).

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

### Reviewer assignment

Sirius uses [CODEOWNERS](.github/CODEOWNERS) to automatically route PRs to the right reviewers based on the files changed. Reviewers are assigned from component teams (e.g. `sirius-core`, `sirius-io`) — one member per team via load balancing. You do not need to manually request reviewers.

Approval from any maintainer is sufficient to merge — it does not have to be the auto-assigned reviewer.

The component teams can also be mentioned directly in PR comments and issues to bring the right people into a discussion without a Slack ping — useful when you need expert input on a specific area:

| Team | Components |
|------|------------|
| `@sirius-db/sirius-core` | Planner, Executors, Operators & Expressions |
| `@sirius-db/sirius-io` | Scan & I/O, Memory & Compression |
| `@sirius-db/sirius-integrations` | Integrations |
| `@sirius-db/sirius-telemetry` | Telemetry & Observability |
| `@sirius-db/sirius-cmake` | CMake |
| `@sirius-db/sirius-build` | CI & Build |
| `@sirius-db/sirius-docs` | Documentation |
| `@sirius-db/sirius-rust` | Rust |
| `@sirius-db/sirius-python` | Python |

When you open or update a PR, you will also be automatically assigned as the author. This helps maintainers track ownership and is not a request for you to review your own work.

### PR branching strategy

Sirius uses ***Self-contained*** PRs submitted from contributor forks as our **primary** method of
contributing to Sirius. There are two approved exceptions, both restricted to [Maintainers](MAINTAINERS.md):
Stacked PRs as a **secondary** contribution method, and **break-glass** CI & Critical changes.

| PR Type | PR Author | Push To | Reasoning |
| --- | --- | --- | --- |
| ***Primary*** - Self-contained | Any Contributor | Personal fork | This allows all contributors to submit PRs from their forks, keeping the main repo focused on development, releases, and Stacked PRs |
| ***Secondary*** - Stacked PRs (dependent changes)  | Only Maintainers | `sirius-db/sirius` repo  with `stacked/` prefix for branch names | Stacked PRs help developers contribute larger changes to the code base in smaller pieces that are easier to review and have fewer conflicts compared to a single large PR. See [Stacked PRs](#stacked-prs) below for more information |
| ***Break-glass*** - CI & Critical | Only Maintainers | `sirius-db/sirius` repo | By exception there may be CI and Critical PRs that must be pushed to this repo if there are CI tests or other triggers necessary before merging; examples: `workflow_dispatch` and certain workflows using secrets and tokens not available in forks |

**NOTE:** Branches of PRs pushed to the main repo and personal forks are cleaned up automatically; this
repo has "Automatically delete head branches" enabled, so merged branches don't need manual deletion.

### Stacked PRs

:warning: Stacked PRs require push permissions to the `sirius-db/sirius` repo in order to work. Currently
this development option is restricted to [Maintainers](MAINTAINERS.md). If you are interested in becoming a
maintainer, see [GOVERNANCE.md](GOVERNANCE.md#becoming-a-maintainer) for details.

For a chain of dependent changes, Sirius uses the [`gh-stack`](https://github.com/github/gh-stack)
`gh` CLI extension — it's the only supported tool for stacked PRs. Don't use Graphite,
`git-spice`, `ghstack`, or similar; a mix of tools operating on the same branches will conflict.

#### Install it once

If needed, follow these [instructions](https://cli.github.com/) to install GitHub CLI.
```bash
# Install Stacked PRs extension
gh extension install github/gh-stack
```

#### Conventions

- Every branch in a stack must be prefixed `stacked/` (e.g. `stacked/docs-contributing-gh-stacks`)
  so stacked work is identifiable and groupable, separate from regular single-branch PRs.
- `stacked/`-prefixed branches may be pushed directly to `origin` (this repo)
  - If you are working on a local copy that is your fork, you will need to ensure `origin` points
    to this repo and not your fork.
  - This is a workaround for [gh-stack#381](https://github.com/github/gh-stack/issues/381), where
    `gh stack`'s `--remote` flag doesn't actually control which repo it operates against, only
    `origin` does. Once that's fixed upstream, this remote-renaming step should no longer be
    necessary.
  - Run the following command replacing `$USER` with your name; otherwise it will use your local username:
      ```bash
      # Rename current origin to local user's name and set origin to main Sirius repo
      git remote rename origin $USER
      git remote add origin "git@github.com:sirius-db/sirius.git"
      ```

#### Basic commands

See `gh stack --help` for the full set:

```bash
gh stack init stacked/<branch1> stacked/<branch2> ...   # create a new stack
gh stack add stacked/<branch>                           # add a branch on top of the current stack
gh stack submit                                         # push all branches and create/update PRs
gh stack sync                                           # rebase and sync the stack with GitHub
```

#### Navigating a stack

Move between layers without typing branch names:

```bash
gh stack top       # check out the top branch (furthest from dev)
gh stack bottom    # check out the bottom branch (closest to dev)
gh stack up        # check out one branch up (further from dev)
gh stack down      # check out one branch down (closer to dev)
gh stack trunk     # check out dev
gh stack switch    # interactively pick a branch in the stack
```

`gh stack checkout <stack-number|PR-number|PR-URL|branch>` resumes a stack you don't currently
have checked out, for example one someone else started or one you left a while ago.

`gh stack modify` opens an interactive editor to reorder, insert, drop, or fold branches in a
stack, useful if a stack's layering needs to change after the fact.

#### Managing a stack

**Adding layers to a stack**

`gh stack` CLI can manage the stack, adding layers as needed with `gh stack add stacked/<branch>`.
This can also be done in the Web UI by clicking on the stack icon (next to the PR status button,
in the top left of the PR page). There you can see the full stack of PRs, and an option to add
to the stack.

**Collaborating with existing stacks**

Any Maintainer can use `gh stack checkout <PR-number>` to check out an existing stack to contribute
code or to manage the stack during the "bottom-up" merge process.

**Managing conflicts**

During active development of the stack, always use `gh stack rebase` to manage conflicts with `dev`
or between your stacked layers. If you have submitted your stack, run `gh stack submit` to push
these changes to the repo and open PRs.

Conflicts can also occur while merging "bottom-up" as described below. The two-step process above
can be shortened to `gh stack sync --prune` during merging to handle the merge conflicts between
the stack layers as they are merged.

**Rebase stack notifications**

GitHub's support for stacked PRs includes a "Rebase stack" option in the Web UI when the target base
(usually `dev`) has moved forward; it is not recommended to use this method.

**NOTE:** CI and the merge queue both test the combined merge commit of the PR and the latest
`dev` before merging. Rebasing is not necessary unless there is a conflict. If you have a conflict,
follow the instructions in **Managing conflicts**.

**Unstacking PRs**

Sometimes it may be necessary to unstack PRs. This can be done in the CLI with
`gh stack unstack` or in the Web UI clicking on the stack icon and selecting the stack icon with
an "x" labeled "Unstack pull requests." Both methods remove the stack and convert the PRs to
self-contained PRs targeting `dev`.

#### Merging a stack

**Note:** each PR's page always shows "wants to merge into `<branch>`" pointing at the layer
directly below it; that's just its git base for diffing, not the actual merge order. Merges
always land bottom-up regardless of what that header displays.

:warning: **Do not use "Enqueue stack" (Web UI) or `gh stack merge` (CLI).** We tested both
against this repo's actual merge queue settings (1 PR built and merged at a time) and confirmed
they do not reliably merge the whole stack through. This is a tested limitation of GitHub's
support for merge queues using squash commits. Until this is fixed, we can use the following
bottom-up merging method in this repo.

**Merging bottom-up**

Steps 1-4 and 7 only need GitHub Web UI, so any maintainer with merge permission can do them,
not just the stack's author. Steps 5-6 need a local checkout with the stack tracked to manage
(see **Managing a stack** above).

1. ***Web UI*** - Open the bottom-most unmerged PR of the stack (the stack panel on any PR in it,
   click the `N/M` badge next to the PR title, shows the whole stack in order).
2. ***Web UI*** - Confirm it's approved and its required checks have passed.
3. ***Web UI*** - Click **"Enqueue pull request"** on that PR's own page to
   merge just that one PR. Do not click "Enqueue stack" (this is only on the top-most layer of
   the stack) and don't use `gh stack merge`.
4. ***Web UI*** - Wait for it to clear the merge queue and merge to `dev`. GitHub automatically
   retargets the next layer's base to `dev` once the branch it was targeting is deleted; no
   manual action is needed for the retargeting.
5. ***GH CLI*** - If the next layer's PR now shows "This branch has conflicts that must be
   resolved" / "Unable to merge", run `gh stack sync --prune` to rebase the remaining branches
   onto the updated `dev` and clean up local branches for the merged PR. Always resolve stack
   conflicts via the CLI.
6. ***GH CLI*** - Confirm it worked via `gh stack view`: the merged layer should show a `✓` under
   a `╌╌╌ merged ╌╌╌` marker, and the remaining branches shouldn't show a `⚠` warning icon. If
   they still do, repeat **step 5**.
7. ***Web UI*** - Open the new bottom-most unmerged PR and repeat from **step 2**, until the whole
   stack is merged.

Our goal is to automate this process in the future; however, this is the current merge method
that works with our repo's merge queue settings.

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
