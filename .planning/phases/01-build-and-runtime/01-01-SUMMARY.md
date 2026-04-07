---
phase: 01-build-and-runtime
plan: 01
subsystem: infra
tags: [pixi, conda, aarch64, arm, cuda, nixl, sysroot, ld-library-path]

# Dependency graph
requires: []
provides:
  - pixi doris environment resolves on aarch64 (sysroot_linux-aarch64 platform-conditional dep)
  - nixl-build meson CUDA paths use CUDA_TARGET variable (sbsa-linux on aarch64)
  - sirius-be and sirius-be-2 tasks compute LIB_ARCH dynamically at runtime
affects:
  - 01-02 (Rust build scripts and nixl-test build.rs changes in plan 02)
  - 02-documentation (build guide correctness depends on these pixi.toml task definitions)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Pixi platform-conditional deps: [feature.X.target.PLATFORM.dependencies] for arch-specific package names"
    - "Inline uname -m detection in pixi bash tasks: CUDA_TARGET=$([ $(uname -m) = aarch64 ] && echo sbsa-linux || echo x86_64-linux)"
    - "exec env pattern for dynamic LD_LIBRARY_PATH inside bash -c tasks"

key-files:
  created: []
  modified:
    - pixi.toml

key-decisions:
  - "sbsa-linux is the correct CUDA target dir for aarch64 data center (Grace Hopper, etc.); Tegra uses aarch64-linux which would be wrong"
  - "Remove env = { LD_LIBRARY_PATH } blocks and move LD_LIBRARY_PATH inside bash -c using exec env — pixi env blocks do not evaluate $() subshells"
  - "Platform-conditional sysroot in [feature.doris.target.linux-aarch64.dependencies] rather than runtime workaround"

patterns-established:
  - "Pattern: arch detection in pixi tasks uses [ $(uname -m) = aarch64 ] && echo ARCH_VALUE || echo x86_64_VALUE"
  - "Pattern: CUDA toolkit target dir maps as uname aarch64 -> sbsa-linux, x86_64 -> x86_64-linux"

requirements-completed: [BUILD-01, BUILD-02, RUNTIME-01]

# Metrics
duration: 12min
completed: 2026-04-07
---

# Phase 01 Plan 01: pixi.toml aarch64 Architecture Fixes Summary

**Platform-conditional sysroot and dynamic arch detection in pixi.toml so the doris environment resolves, nixl builds with sbsa-linux CUDA paths, and sirius-be starts with correct LD_LIBRARY_PATH on aarch64**

## Performance

- **Duration:** ~12 min
- **Started:** 2026-04-07T00:15:00Z
- **Completed:** 2026-04-07T00:27:45Z
- **Tasks:** 3 (2 with code changes, 1 verification)
- **Files modified:** 1

## Accomplishments

- Removed `sysroot_linux-64` from global `[feature.doris.dependencies]` and added platform-conditional `[feature.doris.target.linux-64.dependencies]` and `[feature.doris.target.linux-aarch64.dependencies]` sections (BUILD-01)
- Added `CUDA_TARGET` arch detection in nixl-build task using `uname -m`; replaced all 3 hardcoded `targets/x86_64-linux/` meson flags with `targets/$CUDA_TARGET/` (BUILD-02)
- Replaced both sirius-be and sirius-be-2 `env = { LD_LIBRARY_PATH }` blocks with `bash -c` + `LIB_ARCH` detection + `exec env LD_LIBRARY_PATH` pattern (RUNTIME-01)
- Validated TOML syntax and confirmed all x86_64 fallback branches produce identical values to originals

## Task Commits

Each task was committed atomically:

1. **Task 1: Platform-conditional sysroot and arch-aware nixl-build CUDA paths** - `4a7d461` (feat)
2. **Task 2: Fix sirius-be and sirius-be-2 LD_LIBRARY_PATH for aarch64** - `068c091` (feat)
3. **Task 3: Verify pixi.toml is valid TOML and x86_64 regression** - no commit (verification only, no code changes)

## Files Created/Modified

- `pixi.toml` - Platform-conditional sysroot, arch-aware nixl-build CUDA paths, arch-aware sirius-be LD_LIBRARY_PATH

## Decisions Made

- Used `sbsa-linux` (not `aarch64-linux`) for aarch64 CUDA target directory — confirmed by conda activation script and existing `.pixi/envs/default/targets/sbsa-linux/` directory on this machine; `aarch64-linux` is for Tegra/Jetson, not data center GPUs
- Removed `env = { LD_LIBRARY_PATH }` pixi blocks entirely and moved LD_LIBRARY_PATH setup inside `bash -c` using `exec env` — pixi's env block expands `$VAR` references but does NOT evaluate `$()` subshell commands, making the old approach unusable for dynamic arch detection
- Added platform-conditional deps using pixi's native `[feature.X.target.PLATFORM.dependencies]` syntax rather than any runtime workaround

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- Worktree branch was created from an older base commit; `git reset --soft` to the correct base was required, and the working tree's `pixi.toml` had to be restored from the committed HEAD version before applying changes.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- pixi.toml is ready: `pixi install -e doris` should now succeed on aarch64 (unblocks all subsequent build steps)
- Plan 02 can proceed: nixl-test/build.rs CUDA stubs path fix and nixl-sys crate patch remain as the next set of changes

## Self-Check: PASSED

- FOUND: .planning/phases/01-build-and-runtime/01-01-SUMMARY.md
- FOUND: 4a7d461 (Task 1 commit — feat: platform-conditional sysroot + nixl CUDA paths)
- FOUND: 068c091 (Task 2 commit — feat: arch-aware LD_LIBRARY_PATH for sirius-be tasks)
- FOUND: 3c42d99 (SUMMARY commit)

---
*Phase: 01-build-and-runtime*
*Completed: 2026-04-07*
