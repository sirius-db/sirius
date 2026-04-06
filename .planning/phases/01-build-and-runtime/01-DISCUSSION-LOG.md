# Phase 1: Build and Runtime - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-06
**Phase:** 01-Build and Runtime
**Areas discussed:** CUDA target paths, LD_LIBRARY_PATH strategy, nixl submodule scope, Verification approach

---

## CUDA Target Directory Detection

| Option | Description | Selected |
|--------|-------------|----------|
| Inline uname -m detection | Add CUDA_TARGET variable at top of bash script using uname -m. Simple, self-contained, matches project's stated approach. | ✓ |
| Pixi activation env var | Define SIRIUS_CUDA_TARGET in feature activation env with platform override. Cleaner separation but adds TOML complexity. | |
| You decide | Claude picks the best approach. | |

**User's choice:** Inline uname -m detection
**Notes:** Consistent with project-level decision to use `uname -m` in shell scripts.

---

## LD_LIBRARY_PATH Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Inline uname -m detection | Wrap cmd in bash, compute LIB_ARCH triplet via uname -m. Consistent with CUDA path approach. | ✓ |
| Pixi platform-conditional env | Use activation env with platform override for arch-specific lib path. Keeps task definitions clean. | |
| Drop the multiarch path | Remove x86_64-linux-gnu component entirely if conda packages install everything under $CONDA_PREFIX/lib/. | |

**User's choice:** Inline uname -m detection
**Notes:** User preferred consistency — same detection pattern across all arch-specific paths.

---

## nixl Submodule Scope

| Option | Description | Selected |
|--------|-------------|----------|
| No nixl changes needed | All hardcoded paths are in pixi.toml and nixl-test/build.rs. nixl's meson.build already supports aarch64. | ✓ |
| Minimal nixl fix may be needed | If testing reveals issues, a one-line fix might be needed. Decide after testing. | |
| Check nixl build first | Read nixl's meson.build to verify it handles sbsa-linux correctly before committing. | |

**User's choice:** No nixl changes needed
**Notes:** Codebase scout confirmed nixl meson.build already detects host_machine.cpu_family() for x86_64/aarch64.

---

## Verification Approach

| Option | Description | Selected |
|--------|-------------|----------|
| x86_64 regression test only | Build and run on x86_64 to confirm no regressions. Code review arch-detection logic. Defer aarch64 to first deploy. | ✓ |
| Dry-run path verification | Add temporary validation task that prints resolved paths and checks they exist. | |
| Manual aarch64 test | Actually test on aarch64 hardware before marking complete. | |

**User's choice:** x86_64 regression test only
**Notes:** No aarch64 hardware available; arch-detection logic is simple enough for code review.

---

## Claude's Discretion

- Error messages for failed architecture detection
- Comments explaining sbsa-linux convention
- Implementation order within the plan

## Deferred Ideas

None — discussion stayed within phase scope.
