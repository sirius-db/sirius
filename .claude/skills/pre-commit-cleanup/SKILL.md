---
name: pre-commit-cleanup
description: Clean the current Git changes before committing. Use when preparing to commit or when the pre-commit cleanliness hook blocks a commit.
allowed-tools:
  - Read
  - Grep
  - Glob
  - Edit
  - Write
  - Bash(git status *)
  - Bash(git diff *)
---

Review only the current Git changes using `git status`, `git diff`, and
`git diff --cached`.

Clean up artifacts that should not be committed:

- AI running notes, progress logs, temporary plans, and implementation diaries
- prompt fragments or commentary addressed to the user
- statements such as "I added", "we should now", or "the AI generated"
- debugging output and temporary diagnostic code
- redundant comments that merely restate the code
- excessive documentation added for obvious implementation details
- speculative documentation for behavior that does not exist
- obsolete TODOs introduced during the current task
- temporary files created while investigating the task

Preserve:

- documentation explicitly requested by the user
- public API documentation
- architectural decisions and explanations of non-obvious constraints
- comments explaining why something is necessary
- existing documentation unrelated to the current change

Do not change behavior, stage files, commit, or modify unrelated user changes.

After cleanup, inspect the diff again and summarize what was removed.
