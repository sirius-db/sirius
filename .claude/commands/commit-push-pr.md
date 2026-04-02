---
description: Commit all changes, push to remote, and create a GitHub PR against dev.
---

# Commit, Push, and Create PR

Automate the full flow from local changes to an open pull request.

## Steps

1. Run `git status` and `git diff` to understand all changes (staged + unstaged + untracked).
2. Run `git log --oneline -5` to match the repository's commit message style.
3. Stage relevant files — prefer explicit filenames over `git add -A`. Never stage `.env`, credentials, or large binaries.
4. Draft a concise commit message (1-2 sentences) that focuses on **why** not **what**. End with the co-author trailer.
5. Commit the changes.
6. Run `pixi run pre-commit run -a` to check all files. If it fails, fix the issues, re-stage, and create a NEW commit (do not amend).
7. Determine the current branch. If on `dev`, create a new feature branch first (`git checkout -b <descriptive-name>`).
8. Push to remote with `git push -u origin <branch>`.
9. Create a PR against `dev` using `gh pr create` with:
   - A short title (under 70 chars)
   - A body with `## Summary` (bullet points) and `## Test plan` (checklist)
   - Footer: `Generated with [Claude Code](https://claude.com/claude-code)`
10. Return the PR URL.

## Important

- Never force push.
- Never push directly to `dev` or `main`.
- If pre-commit hooks fail, fix the issues and create a NEW commit (do not amend).
- Ask the user before proceeding if anything looks ambiguous.
