# Phase 1: Build and Runtime - Context

**Gathered:** 2026-04-06
**Status:** Ready for planning

<domain>
## Phase Boundary

Fix all hardcoded x86_64 paths in the Doris integration build chain so the Sirius BE binary compiles from source and starts successfully on an aarch64 NVIDIA platform. All changes must preserve existing x86_64 build and runtime behavior.

</domain>

<decisions>
## Implementation Decisions

### CUDA Target Directory Detection
- **D-01:** Use inline `uname -m` detection in pixi.toml bash scripts: `CUDA_TARGET=$([ $(uname -m) = aarch64 ] && echo sbsa-linux || echo x86_64-linux)`
- **D-02:** Replace all 3 hardcoded `targets/x86_64-linux/` references in the nixl-build task with `$CUDA_TARGET` variable
- **D-03:** Apply same `uname -m` pattern in `doris/crates/nixl-test/build.rs` using `cfg!(target_arch = "aarch64")` to select `sbsa-linux` vs `x86_64-linux`

### LD_LIBRARY_PATH Strategy
- **D-04:** Use inline `uname -m` detection for library arch triplet: `LIB_ARCH=$([ $(uname -m) = aarch64 ] && echo aarch64-linux-gnu || echo x86_64-linux-gnu)`
- **D-05:** Wrap sirius-be task commands in `bash -c` to compute LIB_ARCH and set LD_LIBRARY_PATH dynamically (both sirius-be and sirius-be-2 tasks)

### Pixi Sysroot Dependency
- **D-06:** Replace `sysroot_linux-64` with platform-conditional dependency: `sysroot_linux-64` for x86_64, `sysroot_linux-aarch64` for aarch64 (using pixi's `[feature.doris.target.linux-aarch64.dependencies]`)

### nixl Submodule
- **D-07:** No changes to the nixl submodule — all hardcoded x86_64 paths are in pixi.toml and nixl-test/build.rs, not in nixl's own meson.build (which already supports aarch64)

### Verification Strategy
- **D-08:** Verify via x86_64 regression test only — build nixl, build Rust, start BE on x86_64 to confirm no regressions; defer aarch64 testing to first deploy on real hardware
- **D-09:** Code review of arch-detection logic (uname -m conditionals) for correctness

### Claude's Discretion
- Exact error messages if architecture detection fails
- Whether to add a comment explaining the sbsa-linux convention in pixi.toml
- Order of changes within the implementation plan

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Build system
- `pixi.toml` lines 64-110 — Doris feature dependencies, nixl-build task (CUDA paths), sirius-be tasks (LD_LIBRARY_PATH)
- `doris/crates/nixl-test/build.rs` — Rust build script with CUDA stubs link path

### Project constraints
- `.planning/PROJECT.md` — Key Decisions table (sbsa-linux, uname -m, platform-conditional sysroot)
- `.planning/REQUIREMENTS.md` — BUILD-01 through BUILD-04, RUNTIME-01 acceptance criteria

### Architecture reference
- `.pixi/envs/default/etc/conda/activate.d/~cuda-nvcc_activate.sh` — Conda's own aarch64 CUDA target detection (proves sbsa-linux pattern)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- nixl meson.build already detects `host_machine.cpu_family()` for x86_64/aarch64 — no changes needed there
- Conda activation script (`~cuda-nvcc_activate.sh`) demonstrates the sbsa-linux mapping pattern
- Pixi supports `[feature.X.target.linux-aarch64.dependencies]` for platform-conditional deps

### Established Patterns
- pixi.toml tasks use inline bash (`bash -c '...'`) for multi-step operations — arch detection fits this pattern
- Rust build scripts use `cfg!(target_arch)` for compile-time architecture detection
- Project already declares `platforms = ["linux-64", "linux-aarch64"]` in pixi.toml

### Integration Points
- nixl-build task: meson `-Dcudapath_*` flags receive the CUDA target directory
- sirius-be tasks: `env = { LD_LIBRARY_PATH = "..." }` sets runtime library search path
- doris-build task: depends on nixl-build, inherits CONDA_PREFIX environment
- pixi.toml `[feature.doris.dependencies]`: sysroot package must match target platform

</code_context>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-build-and-runtime*
*Context gathered: 2026-04-06*
