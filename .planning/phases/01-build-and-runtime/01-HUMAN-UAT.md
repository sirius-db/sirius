---
status: partial
phase: 01-build-and-runtime
source: [01-VERIFICATION.md]
started: 2026-04-07T00:30:00Z
updated: 2026-04-07T00:30:00Z
---

## Current Test

[awaiting human testing]

## Tests

### 1. pixi install on aarch64
expected: pixi install -e doris completes without errors; sysroot_linux-aarch64 resolves from conda-forge
result: [pending]

### 2. cargo build on aarch64 (BUILD-03, BUILD-04)
expected: pixi run -e doris doris-build completes; nixl-test links against CUDA stubs at sbsa-linux path; nixl Rust bindings link against aarch64-linux-gnu
result: [pending]

### 3. sirius-be startup on aarch64 (RUNTIME-01)
expected: pixi run -e doris sirius-be starts without missing library errors; LD_LIBRARY_PATH includes aarch64-linux-gnu
result: [pending]

## Summary

total: 3
passed: 0
issues: 0
pending: 3
skipped: 0
blocked: 0

## Gaps
