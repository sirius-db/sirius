---
name: build-errors
description: >
  Analyze Sirius compilation, linker, CUDA, and other build failures. Diagnose the
  cause and, when requested, fix it and verify the build.
argument-hint: "[preset]"
---

# Build Error Analyzer

Use the supplied build output or reproduce the failure with the requested preset
(default `release`). See `.claude/skills/_shared/build-and-query.md` for build and
sanitizer details. Run commands through `pixi run`:

```bash
pixi run make <preset>
```

For wrapper failures, use `pixi run make` for the release build or the shared
reference's install-and-wrapper steps for other presets. `make <preset>` alone
builds only Sirius and its C++ tests.

Read the relevant diagnostics and source context, including earlier errors that
may cause later failures. Consult `error-patterns.md` for known Sirius failures.
Check submodule state or sccache when the error suggests a problem there. Preserve
local submodule changes; do not reset them to resolve a mismatch without permission.
Reduce parallelism on an out-of-memory build, for example:

```bash
CMAKE_BUILD_PARALLEL_LEVEL=4 pixi run make <preset>
```

Explain the diagnosis and proposed fix. If the request includes fixing the build,
apply the fix and rebuild; otherwise report the findings. Report remaining errors
and verification results. Confirm before a destructive clean that could remove
user data or build artifacts.
