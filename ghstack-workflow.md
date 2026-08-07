# Stacked PRs with ghstack

One commit = one PR. Your local stack becomes a chain of reviewable diffs.

---

## One-time setup

```bash
# Install
uv tool install ghstack

# Configure (borrows token from gh CLI — no interactive prompts)
cat > ~/.ghstackrc <<EOF
[ghstack]
github_url = github.com
github_oauth = $(gh auth token)
github_username = $(gh api user --jq .login)
EOF
```

> Never run `ghstack auth` non-interactively — it reads stdin and aborts.

---

## Daily workflow

```bash
# 1. Branch from upstream default
git fetch origin
git checkout -b feature/my-stack origin/main

# 2. Make commits — one logical change each
git add <files> && git commit -m "feat(scope): change A"
git add <files> && git commit -m "feat(scope): change B"
git add <files> && git commit -m "test: coverage for A and B"

# 3. Submit the stack
ghstack --draft
```

ghstack opens one PR per commit, stacked bottom-up. The top PR is the entry point for reviewers.

---

## Update after review feedback

```bash
# Top commit
git commit --amend && ghstack

# Middle commit
git rebase -i origin/main   # mark as "edit", amend, then rebase --continue
ghstack

# Message-only change (tree unchanged — ghstack skips by default)
ghstack --no-skip

# Overwrite PR title/body from local commit message
ghstack --update-fields
```

---

## Rebase onto latest main

```bash
git fetch origin
git rebase origin/main
ghstack
```

Never `git merge` — it breaks the linear commit-to-PR mapping.

---

## Land

```bash
# The GitHub merge button does NOT work for ghstack PRs
ghstack land https://github.com/org/repo/pull/N
# lands the entire stack up to and including PR #N
```

---

## PR descriptions

Every PR in the stack gets the same stack header at the top so any reviewer who lands on any PR immediately sees the full picture. Below that, each PR has its own focused body.

### Stack header (paste into every PR)

```
Stack from ghstack (oldest at bottom):

* #4 feat: test coverage for A and B
* #3 feat(scope): change B
* --> #2 feat(scope): change A   ← mark the current PR with -->

Closes #<issue>
```

`-->` marks the current PR. Copy the same list into each PR, just move the arrow.

### Full body template

```bash
gh pr edit <N> --body "$(cat <<'EOF'
Stack from ghstack (oldest at bottom):

* [#4](url) feat: test coverage for A and B
* [#3](url) feat(scope): change B
* **--> [#2](url) feat(scope): change A**

Closes #<issue>

---

## What

One paragraph: what this diff does and why.

## How it works

Key decision or algorithm, only if non-obvious.

## Files changed

| File | Change |
|---|---|
| `path/to/file` | what changed |
EOF
)"
```

### Real example (from this repo)

PR #2 body:
```
Stack from ghstack (oldest at bottom):

* [#4](https://github.com/aocsa/btree_project/pull/4) test: add find() coverage for memory and disk btrees
* [#3](https://github.com/aocsa/btree_project/pull/3) feat(disk): implement find() for disk-backed btree
* **--> [#2](https://github.com/aocsa/btree_project/pull/2) feat(memory): implement find() for in-memory btree**

---

## What

Implements `find()` for `utec::memory::btree`. Previously returned `false` unconditionally.

## How it works

Binary search at each node, recurse into the matching child pointer; return `false` at a null leaf.
```

PR #4 body (top of stack — entry point for reviewers):
```
Stack from ghstack (oldest at bottom):

* **--> [#4](https://github.com/aocsa/btree_project/pull/4) test: add find() coverage for memory and disk btrees**
* [#3](https://github.com/aocsa/btree_project/pull/3) feat(disk): implement find() for disk-backed btree
* [#2](https://github.com/aocsa/btree_project/pull/2) feat(memory): implement find() for in-memory btree

---

## What

Adds `FindExistingKeys` and `FindMissingKeys` test cases to both btree variants.
No tests existed for the `find()` path before this stack.
```

---

## Pitfalls

| Problem | Fix |
|---|---|
| `ghstack auth` hangs in scripts | Set `~/.ghstackrc` manually instead |
| Message-only edit not pushed | Add `--no-skip` |
| `git merge` broke the stack | Rebase only — always |
| Stale default branch cached | Delete `.git/ghstack-repo-info.json` and rerun |
| GitHub merge button fails | Use `ghstack land <PR-url>` |
