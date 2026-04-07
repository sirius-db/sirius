---
phase: 01-build-and-runtime
plan: 02
subsystem: rust-build-scripts
tags: [aarch64, cuda, nixl, build-scripts, cargo]
dependency_graph:
  requires: []
  provides: [BUILD-03, BUILD-04]
  affects: [doris/crates/nixl-test, doris/Cargo.toml, doris/thirdparty/nixl]
tech_stack:
  added: []
  patterns:
    - "cfg!(target_arch) compile-time arch detection in Rust build.rs"
    - "[patch.crates-io] Cargo workspace override for local crate substitution"
key_files:
  created: []
  modified:
    - doris/crates/nixl-test/build.rs
    - doris/Cargo.toml
    - doris/thirdparty/nixl/src/bindings/rust/build.rs
decisions:
  - "Use cfg!(target_arch = \"aarch64\") to select sbsa-linux vs x86_64-linux CUDA target at compile time (D-03)"
  - "Override crates.io nixl-sys 0.10.0 with local submodule via [patch.crates-io] to apply one-line fix (D-07 extended)"
  - "One-line fix to nixl submodule build.rs line 114: use arch variable instead of hardcoded x86_64-linux-gnu"
metrics:
  duration: "~5 minutes"
  completed_date: "2026-04-07"
  tasks_completed: 2
  tasks_total: 2
  files_modified: 3
---

# Phase 01 Plan 02: Rust Build Script aarch64 Fixes Summary

Arch-aware CUDA stubs and nixl lib path in Rust build scripts using `cfg!(target_arch)` compile-time detection and `[patch.crates-io]` to redirect the published nixl-sys crate to the local submodule.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Fix nixl-test/build.rs CUDA stubs path for aarch64 | 22b0d16 | doris/crates/nixl-test/build.rs |
| 2 | Patch nixl-sys to use local submodule and fix hardcoded lib path | 3086a3a | doris/Cargo.toml, doris/thirdparty/nixl/src/bindings/rust/build.rs |

## What Was Built

**Task 1 — nixl-test/build.rs:**
Replaced the hardcoded `targets/x86_64-linux/lib/stubs` path on line 9 with a `cfg!(target_arch = "aarch64")` conditional that selects `sbsa-linux` (aarch64 SBSA data center ARM) or `x86_64-linux` (x86_64). The x86_64 path is identical to the original, preserving backward compatibility. Comments explain the sbsa-linux convention.

**Task 2 — doris/Cargo.toml + nixl submodule build.rs:**
- Added `[patch.crates-io]` section to `doris/Cargo.toml` pointing `nixl-sys` to `thirdparty/nixl/src/bindings/rust` (the local submodule). This overrides the published crates.io version (which has the same hardcoded bug) with our patched copy.
- Fixed line 114 of the nixl submodule's `build.rs`: replaced `x86_64-linux-gnu` literal with `{}-linux-gnu", nixl_root_path, arch` using the `arch` variable already defined on line 103 via `get_arch()`. On x86_64 this produces `lib/x86_64-linux-gnu` (identical to original); on aarch64 it produces `lib/aarch64-linux-gnu` (correct for ARM).
- Committed the one-line fix inside the submodule first (detached HEAD), then committed the submodule pointer update together with `Cargo.toml` in the parent repo.

## Verification Results

All 5 plan verification checks passed:
1. `cfg!(target_arch = "aarch64")` present in nixl-test/build.rs
2. `sbsa-linux` string present in nixl-test/build.rs
3. `[patch.crates-io]` section present in doris/Cargo.toml
4. `{}-linux-gnu", nixl_root_path, arch` pattern found in nixl submodule build.rs
5. `x86_64-linux-gnu.*nixl_root_path` (hardcoded) NOT present in nixl submodule build.rs

## Deviations from Plan

### Auto-fixed Issues

None — plan executed exactly as written.

### Procedural Note

The nixl submodule (`doris/thirdparty/nixl`) is a nested git repository. Staging `doris/thirdparty/nixl/src/bindings/rust/build.rs` directly from the parent repo fails with "Pathspec is in submodule" error. The correct procedure (applied here):
1. `cd` into submodule, `git add`, `git commit` to capture the one-line fix
2. Return to parent repo, `git add doris/thirdparty/nixl` to update the submodule pointer
3. Commit parent repo with `doris/Cargo.toml` and the updated submodule pointer together

This is standard git submodule workflow. Not a deviation from plan intent — the plan correctly called this a "one-line submodule fix".

## Known Stubs

None — no stub patterns, placeholder text, or unwired data sources exist in the modified files.

## Threat Flags

None — modified files are Rust build scripts (linker search path configuration). No network endpoints, auth paths, file access patterns at trust boundaries, or schema changes introduced.

## Self-Check: PASSED

Files exist:
- FOUND: doris/crates/nixl-test/build.rs
- FOUND: doris/Cargo.toml
- FOUND: doris/thirdparty/nixl/src/bindings/rust/build.rs

Commits exist:
- FOUND: 22b0d16 (Task 1 — feat(01-02): arch-aware CUDA stubs path)
- FOUND: 3086a3a (Task 2 — feat(01-02): patch nixl-sys and fix arch-aware lib path)
