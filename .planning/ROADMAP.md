# Roadmap: Sirius-Doris aarch64 Support

## Overview

Port the Doris integration layer to aarch64 by fixing all hardcoded x86_64 paths in the build chain (pixi, meson, Rust build scripts, runtime library paths), then documenting aarch64 as a supported platform. The fixes follow a strict dependency chain -- each fix unblocks the next -- so Phase 1 delivers the entire working build-and-run capability, and Phase 2 captures the final state in documentation.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Build and Runtime** - Fix all hardcoded x86_64 paths so the Sirius BE compiles and starts on aarch64
- [ ] **Phase 2: Documentation** - Update build guide to reflect aarch64 as a supported platform

## Phase Details

### Phase 1: Build and Runtime
**Goal**: The Sirius Doris BE binary compiles from source and starts successfully on an aarch64 NVIDIA platform
**Depends on**: Nothing (first phase)
**Requirements**: BUILD-01, BUILD-02, BUILD-03, BUILD-04, RUNTIME-01
**Success Criteria** (what must be TRUE):
  1. Running `pixi install` on an aarch64 machine resolves all dependencies without errors (no x86_64-only sysroot failure)
  2. The nixl C++ library builds via meson on aarch64, finding CUDA libraries under `targets/sbsa-linux/` instead of failing on x86_64 paths
  3. `cargo build` for the Sirius BE (including nixl-test and nixl Rust bindings) completes on aarch64 without link errors
  4. The `sirius-be` pixi task launches the Sirius BE process on aarch64 with correct LD_LIBRARY_PATH (process starts, does not crash on missing libraries)
  5. All changes preserve existing x86_64 build and runtime behavior (no regressions on x86_64)
**Plans**: TBD

### Phase 2: Documentation
**Goal**: A developer on aarch64 can follow the existing build guide to build and deploy Sirius without consulting anyone
**Depends on**: Phase 1
**Requirements**: DOCS-01
**Success Criteria** (what must be TRUE):
  1. BUILD_DEPLOY_TEST_GUIDE.md lists aarch64 as a supported platform with any architecture-specific notes (e.g., SBSA requirement, supported GPU families)
  2. A developer reading only the build guide has enough information to build and run Sirius on aarch64 -- no undocumented steps
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Build and Runtime | 0/0 | Not started | - |
| 2. Documentation | 0/0 | Not started | - |
