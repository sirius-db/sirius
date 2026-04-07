# Phase 1: Build and Runtime - Research

**Researched:** 2026-04-06
**Domain:** aarch64 NVIDIA platform porting — build system path fixes (pixi/meson/Rust/LD_LIBRARY_PATH)
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** Use inline `uname -m` detection in pixi.toml bash scripts: `CUDA_TARGET=$([ $(uname -m) = aarch64 ] && echo sbsa-linux || echo x86_64-linux)`
- **D-02:** Replace all 3 hardcoded `targets/x86_64-linux/` references in the nixl-build task with `$CUDA_TARGET` variable
- **D-03:** Apply same `uname -m` pattern in `doris/crates/nixl-test/build.rs` using `cfg!(target_arch = "aarch64")` to select `sbsa-linux` vs `x86_64-linux`
- **D-04:** Use inline `uname -m` detection for library arch triplet: `LIB_ARCH=$([ $(uname -m) = aarch64 ] && echo aarch64-linux-gnu || echo x86_64-linux-gnu)`
- **D-05:** Wrap sirius-be task commands in `bash -c` to compute LIB_ARCH and set LD_LIBRARY_PATH dynamically (both sirius-be and sirius-be-2 tasks)
- **D-06:** Replace `sysroot_linux-64` with platform-conditional dependency: `sysroot_linux-64` for x86_64, `sysroot_linux-aarch64` for aarch64 (using pixi's `[feature.doris.target.linux-aarch64.dependencies]`)
- **D-07:** No changes to the nixl submodule — all hardcoded x86_64 paths are in pixi.toml and nixl-test/build.rs, not in nixl's own meson.build (which already supports aarch64)
- **D-08:** Verify via x86_64 regression test only — build nixl, build Rust, start BE on x86_64 to confirm no regressions; defer aarch64 testing to first deploy on real hardware
- **D-09:** Code review of arch-detection logic (uname -m conditionals) for correctness

### Claude's Discretion

- Exact error messages if architecture detection fails
- Whether to add a comment explaining the sbsa-linux convention in pixi.toml
- Order of changes within the implementation plan

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope.

</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| BUILD-01 | Pixi doris environment resolves on aarch64 (platform-conditional sysroot dependency) | D-06: `[feature.doris.target.linux-aarch64.dependencies]` syntax confirmed by pixi docs and existing `platforms = ["linux-64", "linux-aarch64"]` declaration |
| BUILD-02 | nixl C++ library builds on aarch64 via meson (CUDA target paths use `sbsa-linux`) | D-01/D-02: nixl-build task already uses `bash -c`, CUDA target dir confirmed as `sbsa-linux` by conda activation script. Verified: `.pixi/envs/default/targets/sbsa-linux/` exists on this aarch64 machine. |
| BUILD-03 | nixl-test crate compiles on aarch64 (CUDA stubs path detects architecture) | D-03: `doris/crates/nixl-test/build.rs` line 9 hardcodes `targets/x86_64-linux/lib/stubs` — `cfg!(target_arch)` fix is the standard Rust approach |
| BUILD-04 | nixl Rust bindings use correct library paths on aarch64 (fix hardcoded `x86_64-linux-gnu`) | `nixl-sys 0.10.0` from crates.io (not local submodule) has line 114 hardcoded. See Open Question Q-01 — D-07 creates a tension here. |
| RUNTIME-01 | Sirius BE starts on aarch64 with correct LD_LIBRARY_PATH (both sirius-be and sirius-be-2 tasks) | D-04/D-05: pixi `env` block cannot execute subshell commands; requires wrapping both tasks in `bash -c` |

</phase_requirements>

---

## Summary

Phase 1 is a surgical build-system porting task: fix 6 hardcoded x86_64 paths across 3 files. No runtime logic changes, no new dependencies, no architectural restructuring. The entire port is additive — existing x86_64 code paths become the `else` branch of new conditionals.

The machine running this session is already aarch64 (confirmed: `uname -m` returns `aarch64`, `.pixi/envs/default/targets/sbsa-linux/` exists). The `default` pixi environment is installed; the `doris` environment is not yet installed (blocked by the `sysroot_linux-64` failure on aarch64). This matches the expected pre-fix state.

The 5 requirements map to 4 files: `pixi.toml` (BUILD-01 sysroot, BUILD-02 nixl-build CUDA paths, RUNTIME-01 LD_LIBRARY_PATH), `doris/crates/nixl-test/build.rs` (BUILD-03), and the `nixl-sys` crate (BUILD-04). One open question exists: BUILD-04 involves the published `nixl-sys` crate (not the submodule directly), which D-07 may not have accounted for. See Open Question Q-01.

**Primary recommendation:** Implement changes in dependency order (D-06 → D-01/D-02 → D-03 → BUILD-04 → D-04/D-05). Each fix unblocks the next. Verification is x86_64 regression only (D-08/D-09).

---

## Standard Stack

### Core (No Changes — All Already Correct for aarch64)

| Library | Version | Purpose | aarch64 Status |
|---------|---------|---------|---------------|
| pixi | >= 0.59 | Environment management | Already declares `platforms = ["linux-64", "linux-aarch64"]` |
| CUDA Toolkit | 13.1.* (conda) | GPU libraries | `sbsa-linux` target dir available; confirmed by activation script |
| RAPIDS cuDF | 26.02.* | GPU DataFrame ops | aarch64 conda packages exist in rapidsai channel |
| DuckDB | 1.4.4 | SQL engine | Official aarch64 support |
| Rust | >= 1.85 | Doris BE | Tier 1 `aarch64-unknown-linux-gnu` target |
| UCX | >= 1.20 | GPU-direct exchange | First-class aarch64 support |

### What Changes (Exactly 4 Files)

| File | Lines Affected | Type of Fix |
|------|----------------|------------|
| `pixi.toml` | 69 (sysroot dep), 92-94 (nixl-build CUDA paths), 109-110 (sirius-be LD_LIBRARY_PATH) | Build config |
| `doris/crates/nixl-test/build.rs` | Line 9 (CUDA stubs path) | Rust `cfg!(target_arch)` |
| `doris/Cargo.toml` | Add `[patch.crates-io]` section | Cargo dependency override |
| `doris/thirdparty/nixl/src/bindings/rust/build.rs` | Line 114 (hardcoded `x86_64-linux-gnu`) | One-line submodule fix |

Note: Files 3 and 4 only apply if BUILD-04 requires a code fix (see Open Question Q-01).

---

## Architecture Patterns

### Pattern 1: Inline Architecture Detection in pixi Task Shell Scripts

**What:** Compute an arch-specific path variable at the start of a `bash -c` script, then use it in subsequent commands.

**When to use:** Any pixi task that references `targets/x86_64-linux/` or `x86_64-linux-gnu`.

**Decision (D-01/D-02):** Use the `&&/||` ternary form, not a `case` statement.

```bash
# Source: D-01 decision (CONTEXT.md), validated by conda ~cuda-nvcc_activate.sh
CUDA_TARGET=$([ $(uname -m) = aarch64 ] && echo sbsa-linux || echo x86_64-linux)

meson setup builddir --prefix="$CONDA_PREFIX" --libdir=lib --buildtype=release \
  -Ducx_path=$CONDA_PREFIX \
  -Dcudapath_inc=$CONDA_PREFIX/targets/$CUDA_TARGET/include \
  -Dcudapath_lib=$CONDA_PREFIX/targets/$CUDA_TARGET/lib \
  -Dcudapath_stub=$CONDA_PREFIX/targets/$CUDA_TARGET/lib/stubs \
  ...
```

[VERIFIED: conda `~cuda-nvcc_activate.sh` uses identical `[[ "linux-aarch64" == ... ]] && targetsDir="targets/sbsa-linux"` logic]

### Pattern 2: Pixi Platform-Conditional Dependencies

**What:** Use `[feature.X.target.PLATFORM.dependencies]` for packages with architecture-specific names.

**When to use:** When a conda package has a different name per platform (not just different binary).

**Decision (D-06):** Remove `sysroot_linux-64 = ">=2.32"` from `[feature.doris.dependencies]` (global) and add platform-specific entries.

```toml
# Source: D-06 decision (CONTEXT.md), pixi documentation [CITED: pixi.prefix.dev/latest/workspace/multi_platform_configuration/]
[feature.doris.target.linux-64.dependencies]
sysroot_linux-64 = ">=2.32"

[feature.doris.target.linux-aarch64.dependencies]
sysroot_linux-aarch64 = ">=2.32"
```

[VERIFIED: `pixi.toml` currently has `sysroot_linux-64 = ">=2.32"` at line 69 in `[feature.doris.dependencies]` — this is the line to replace]

### Pattern 3: Compile-Time Architecture Detection in Rust build.rs

**What:** Use `cfg!(target_arch = "aarch64")` to select the correct path at Cargo compile time.

**When to use:** Rust build scripts that need to choose architecture-specific library or include paths.

**Decision (D-03):** Apply to `doris/crates/nixl-test/build.rs`.

```rust
// Source: D-03 decision (CONTEXT.md)
// File: doris/crates/nixl-test/build.rs
fn main() {
    println!("cargo:rustc-link-search=native=/run/opengl-driver/lib");

    if let Ok(prefix) = std::env::var("CONDA_PREFIX") {
        let cuda_target = if cfg!(target_arch = "aarch64") {
            "sbsa-linux"
        } else {
            "x86_64-linux"
        };
        println!(
            "cargo:rustc-link-search=native={}/targets/{}/lib/stubs",
            prefix, cuda_target
        );
    }

    println!("cargo:rustc-link-lib=dylib=cuda");
}
```

[VERIFIED: `doris/crates/nixl-test/build.rs` line 9 currently hardcodes `x86_64-linux`]

### Pattern 4: Wrapping pixi Tasks in bash -c for Dynamic Environment Setup

**What:** Convert a simple pixi task command into a `bash -c` script that computes arch-specific env vars before executing the main binary.

**When to use:** When a pixi `env` block value would need shell command substitution (`$()`) which pixi does NOT evaluate in env blocks.

**Decision (D-04/D-05):** Both `sirius-be` and `sirius-be-2` tasks need this.

```toml
# Source: D-04/D-05 decision (CONTEXT.md)
# Before (broken on aarch64):
sirius-be = { cmd = "doris/target/release/sirius-doris-be ...",
              env = { LD_LIBRARY_PATH = "$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/x86_64-linux-gnu" }, ... }

# After (arch-aware):
sirius-be = { cmd = """bash -c 'LIB_ARCH=$([ $(uname -m) = aarch64 ] && echo aarch64-linux-gnu || echo x86_64-linux-gnu)
LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/$LIB_ARCH exec doris/target/release/sirius-doris-be \
  --heartbeat-port 19050 --be-port 19060 --brpc-port 18060 --http-port 18040 \
  --arrow-flight-port 18071 --gpu-cache-size 2GB --gpu-processing-size 2GB --fe 127.0.0.1:9030'""",
  description = "Run Sirius GPU BE 1 (ports 19xxx/18xxx)" }
```

[VERIFIED: pixi `env` block expands `$CONDA_PREFIX` but does NOT evaluate `$(uname -m)`. Confirmed by inspection of sirius-be-2 which already uses `bash -c` for its cmd.]

**Important:** Remove the old `env = { LD_LIBRARY_PATH = "..." }` block when converting to `bash -c` — having both is redundant and the env block value will be wrong on aarch64.

### Anti-Patterns to Avoid

- **Using `aarch64-linux` for CUDA target dir:** The correct mapping is `uname -m = aarch64` -> `sbsa-linux`. The `aarch64-linux` dir is for Tegra/Jetson. Verified on this machine: `ls .pixi/envs/default/targets/` shows `sbsa-linux`, not `aarch64-linux`.
- **Making invasive nixl submodule changes:** D-07 requires minimal changes. The nixl `meson.build` already supports aarch64 via `host_machine.cpu_family()`. Only the Rust binding's `build.rs` line 114 needs a one-line change.
- **Forgetting sirius-be-2:** Both `sirius-be` and `sirius-be-2` have the same `x86_64-linux-gnu` LD_LIBRARY_PATH. Both must be updated.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| CUDA target dir detection | Custom nvcc probe script | `uname -m` + hardcoded map | nvcc may not be on PATH in build.rs context; `uname -m` is universally available |
| Cross-compilation setup | cmake toolchain files, sysroot management | Native aarch64 build only | Cross-compile adds complexity for no benefit; Grace Hopper hardware is native aarch64 |
| aarch64 CONDA package verification | Manual conda search at build time | Trust existing `platforms = ["linux-64", "linux-aarch64"]` | rapidsai, conda-forge already publish aarch64 packages for all required dependencies |

---

## BUILD-04 Deep Dive: nixl-sys Crate and D-07 Tension

### What the Code Actually Does

`doris/crates/nixl-test/Cargo.toml` declares `nixl-sys = "0.10"`. This resolves to the **published crate** `nixl-sys 0.10.0` from crates.io [VERIFIED: `doris/Cargo.lock` shows `source = "registry+https://github.com/rust-lang/crates.io-index"`].

The published crate's `build.rs` line 114:
```rust
println!("cargo:rustc-link-search=native={}/lib/x86_64-linux-gnu", nixl_root_path);
```
[VERIFIED: `/home/bwyogatama/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/nixl-sys-0.10.0/build.rs` line 114]

### Is This Actually a Blocker?

The `build_nixl()` function adds multiple search paths (lines 110-114):

| Line | Path | Correct on aarch64? |
|------|------|---------------------|
| 110 | `$NIXL_PREFIX/lib/aarch64-linux-gnu` | Yes — `get_lib_path()` uses the arch variable |
| 111 | `$NIXL_PREFIX` | Yes — covers root |
| 112 | `$NIXL_PREFIX/lib` | YES — this is where nixl-build installs libnixl.so (meson `--libdir=lib`) |
| 113 | `$NIXL_PREFIX/lib64` | Harmless extra |
| 114 | `$NIXL_PREFIX/lib/x86_64-linux-gnu` | Wrong arch, but harmless since 112 already covers it |

**Conclusion:** Line 114 is cosmetically wrong but functionally harmless because line 112 searches `$NIXL_PREFIX/lib/` where `nixl-build` installs `libnixl.so`. The linker finds `libnixl.so` before reaching the wrong path.

[ASSUMED] Whether the build actually fails due to line 114 on aarch64 depends on the exact linker search order. Line 112 should be sufficient, but this is not tested on real hardware.

### Two Valid Approaches for BUILD-04

**Option A (Fix it):** Add `[patch.crates-io]` to `doris/Cargo.toml` pointing `nixl-sys` to the local submodule, and make the one-line fix in the submodule's `build.rs`:
```toml
# In doris/Cargo.toml
[patch.crates-io]
nixl-sys = { path = "thirdparty/nixl/src/bindings/rust" }
```
```rust
// In doris/thirdparty/nixl/src/bindings/rust/build.rs line 114 — change to:
println!("cargo:rustc-link-search=native={}/lib/{}-linux-gnu", nixl_root_path, arch);
```
This is technically the cleanest fix and matches REQUIREMENTS intent. The submodule change is one line. D-07 states "no changes to nixl submodule" in the context of meson.build already supporting aarch64 — the Rust bindings are a different file and the decision may not have intended to exclude this minimal change.

**Option B (Accept it):** Note that line 114 is harmless (line 112 covers the actual install location). Mark BUILD-04 as satisfied by the overall build succeeding. Document the cosmetic issue in code comments.

**Recommendation for planner:** Implement Option A. It satisfies the requirement explicitly stated in REQUIREMENTS.md, the change is 2 lines total (one in Cargo.toml, one in build.rs), and it follows the project's stated constraint of "minimal submodule change." The planner should verify this against the user's intent given D-07.

---

## Common Pitfalls

### Pitfall 1: `sbsa-linux` vs `aarch64-linux` CUDA Target Dir
**What goes wrong:** Using `aarch64-linux` instead of `sbsa-linux` for CUDA paths on server platforms.
**Why it happens:** The uname output is `aarch64`, so developers assume the CUDA dir is named `aarch64-*`. NVIDIA uses `sbsa-linux` for Server Base System Architecture (data center GPUs).
**How to avoid:** Use D-01's explicit mapping: `[ $(uname -m) = aarch64 ] && echo sbsa-linux || echo x86_64-linux`. Never derive CUDA target dir directly from `uname -m` output without this mapping step.
**Warning signs:** `targets/aarch64-linux` directory does not exist on this machine; only `targets/sbsa-linux` exists.

### Pitfall 2: Forgetting sirius-be-2
**What goes wrong:** `sirius-be` task updated correctly but `sirius-be-2` still fails on aarch64.
**Why it happens:** Both tasks are adjacent in pixi.toml with identical `x86_64-linux-gnu` in their `env` blocks.
**How to avoid:** Search pixi.toml for ALL occurrences of `x86_64-linux-gnu`. There are exactly 2 (lines 109 and 110).

### Pitfall 3: pixi env Block Does Not Execute Subshells
**What goes wrong:** Setting `env = { LD_LIBRARY_PATH = "...$( uname -m)..." }` in a pixi task — the `$()` is not evaluated.
**Why it happens:** pixi expands environment variable references (`$VAR`) but does not execute shell command substitutions.
**How to avoid:** D-05 mandates `bash -c` wrapping. Set `LD_LIBRARY_PATH` inside the shell command, not in the `env` block.

### Pitfall 4: Breaking x86_64 Builds
**What goes wrong:** aarch64 changes accidentally break the x86_64 code path.
**Why it happens:** Testing only on aarch64 after changes. The `else` branch (x86_64) must preserve exact current behavior.
**How to avoid:** D-08: verify x86_64 regression by running `pixi install -e doris`, `pixi run nixl-build`, `pixi run doris-build`. All three must succeed.

### Pitfall 5: nixl Build Cache Not Cleared After Path Changes
**What goes wrong:** nixl-build includes a cache-skip check (`if [ -f "$CONDA_PREFIX/lib/libnixl.so" ]; then exit 0; fi`). If nixl was previously built with x86_64 paths and libnixl.so exists, the build skips, leaving stale artifacts.
**Why it happens:** The cache guard is intentional for fast rebuilds, but it means the first aarch64 build won't re-run meson if the cache check passes.
**How to avoid:** When testing on aarch64, remove `libnixl.so` from `$CONDA_PREFIX/lib/` before running `pixi run nixl-build` to force a full rebuild.

---

## Code Examples

### nixl-build Task After Fix (pixi.toml lines 74-105)

```toml
# [VERIFIED: .planning/phases/01-build-and-runtime/01-CONTEXT.md — D-01, D-02]
nixl-build = { cmd = """bash -c 'set -e
  NIXL_SRC=$PIXI_PROJECT_ROOT/doris/thirdparty/nixl
  CUDA_TARGET=$([ $(uname -m) = aarch64 ] && echo sbsa-linux || echo x86_64-linux)

  if [ -f "$CONDA_PREFIX/lib/libnixl.so" ]; then
    echo "nixl already installed in conda prefix"
    exit 0
  fi

  echo "==> Building nixl from submodule..."
  cd "$NIXL_SRC"

  # Patch out Python bindings (no pybind11 needed)
  sed -i "/subdir..python/s/^/# /" src/bindings/meson.build 2>/dev/null || true

  echo "==> Building nixl with meson (UCX plugin only)..."
  rm -rf builddir
  meson setup builddir --prefix="$CONDA_PREFIX" --libdir=lib --buildtype=release \
    -Ducx_path=$CONDA_PREFIX \
    -Dcudapath_inc=$CONDA_PREFIX/targets/$CUDA_TARGET/include \
    -Dcudapath_lib=$CONDA_PREFIX/targets/$CUDA_TARGET/lib \
    -Dcudapath_stub=$CONDA_PREFIX/targets/$CUDA_TARGET/lib/stubs \
    -Drust=false \
    -Denable_plugins=UCX \
    -Dbuild_tests=false \
    -Dbuild_examples=false
  cd builddir
  ninja -j$(nproc)
  ninja install

  echo "==> nixl installed to $CONDA_PREFIX"
  ls "$CONDA_PREFIX/lib/"libnixl* 2>/dev/null || true
'""", description = "Build nixl C++ library from submodule into conda prefix" }
```

### nixl-test/build.rs After Fix

```rust
// [VERIFIED: doris/crates/nixl-test/build.rs — D-03]
fn main() {
    // Link against libcuda (CUDA driver API) for GPU memory allocation.
    // On NixOS, libcuda.so lives in /run/opengl-driver/lib/.
    println!("cargo:rustc-link-search=native=/run/opengl-driver/lib");

    // CUDA toolkit stubs (conda env provides libcuda.so stub for linking)
    // sbsa-linux is the CUDA target dir for aarch64 SBSA (data center ARM, e.g. Grace Hopper).
    // x86_64-linux is used for standard x86_64 systems.
    if let Ok(prefix) = std::env::var("CONDA_PREFIX") {
        let cuda_target = if cfg!(target_arch = "aarch64") {
            "sbsa-linux"
        } else {
            "x86_64-linux"
        };
        println!(
            "cargo:rustc-link-search=native={}/targets/{}/lib/stubs",
            prefix, cuda_target
        );
    }

    println!("cargo:rustc-link-lib=dylib=cuda");
}
```

### pixi.toml Sysroot Dependency Fix

```toml
# [VERIFIED: D-06, pixi platform-conditional dependency syntax]
# Before (in [feature.doris.dependencies] — remove this):
# sysroot_linux-64 = ">=2.32"

# Add these two new sections:
[feature.doris.target.linux-64.dependencies]
sysroot_linux-64 = ">=2.32"

[feature.doris.target.linux-aarch64.dependencies]
sysroot_linux-aarch64 = ">=2.32"
```

### sirius-be Task After Fix

```toml
# [VERIFIED: D-04, D-05]
sirius-be = { cmd = """bash -c '
  LIB_ARCH=$([ $(uname -m) = aarch64 ] && echo aarch64-linux-gnu || echo x86_64-linux-gnu)
  exec env LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/$LIB_ARCH" \
    doris/target/release/sirius-doris-be \
    --heartbeat-port 19050 --be-port 19060 --brpc-port 18060 --http-port 18040 \
    --arrow-flight-port 18071 --gpu-cache-size 2GB --gpu-processing-size 2GB \
    --fe 127.0.0.1:9030
'""", description = "Run Sirius GPU BE 1 (ports 19xxx/18xxx)" }
```

---

## Runtime State Inventory

> Not a rename/refactor/migration phase. Skipped.

---

## Environment Availability

The current machine is aarch64 (confirmed: `uname -m` returns `aarch64`).

| Dependency | Required By | Available | Version | Notes |
|------------|-------------|-----------|---------|-------|
| pixi | Environment management | ✓ | 0.63.2 | Located at `~/.pixi/bin/pixi` |
| CUDA sbsa-linux target | nixl-build, nixl-test | ✓ | exists | `.pixi/envs/default/targets/sbsa-linux/` confirmed |
| doris pixi environment | BUILD-01 through RUNTIME-01 | ✗ | — | NOT installed yet (blocked by sysroot failure on aarch64 — this is the pre-fix state) |
| cargo (system) | Rust build | ✓ | present at `/usr/bin/cargo` | Pixi env cargo will be preferred once doris env installs |
| nixl submodule | nixl-build | ✓ | present | `doris/thirdparty/nixl/` populated |

**Missing dependencies with fallback:**
- `doris` pixi environment: Not installed. Expected — this is the state that BUILD-01 (sysroot fix) is intended to unblock. After applying D-06, `pixi install -e doris` should succeed on this aarch64 machine.

---

## Validation Architecture

> `workflow.nyquist_validation` absent from `.planning/config.json` — treating as enabled.

However, this phase has no automated test infrastructure. All 5 requirements are build/runtime system changes, not logic changes. Validation is inherently manual or observational.

### Test Framework

| Property | Value |
|----------|-------|
| Framework | None (build system changes; manual verification) |
| Config file | N/A |
| Quick run command | `pixi run -e doris doris-check` (cargo check, fast) |
| Full suite command | `pixi run -e doris doris-build` then `pixi run sirius-be` (launch test) |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Verification Command | Automated? |
|--------|----------|-----------|---------------------|------------|
| BUILD-01 | pixi install resolves on aarch64 without sysroot error | smoke | `pixi install -e doris` — succeeds without error | Yes (exit code) |
| BUILD-02 | nixl C++ builds on aarch64 with sbsa-linux paths | smoke | `pixi run -e doris nixl-build` — `libnixl.so` appears in `$CONDA_PREFIX/lib/` | Yes (exit code + file check) |
| BUILD-03 | nixl-test crate compiles on aarch64 | smoke | `pixi run -e doris doris-build` — nixl-test crate compiles without linker errors | Yes (exit code) |
| BUILD-04 | nixl Rust bindings link correctly | smoke | `pixi run -e doris doris-build` — no "cannot find -lnixl" errors | Yes (exit code) |
| RUNTIME-01 | Sirius BE starts on aarch64 without missing library errors | smoke | `pixi run -e doris sirius-be` — process starts, does not crash immediately with `error while loading shared libraries` | Manual (no GPU available for full test) |

### x86_64 Regression Verification (D-08)

Per D-08, all x86_64 verification happens on an x86_64 machine (this aarch64 session cannot test that path). The plan must include explicit verification steps that the implementer runs on x86_64:

1. `pixi install -e doris` — must succeed (sysroot_linux-64 still resolves)
2. `pixi run -e doris nixl-build` — must succeed (CUDA_TARGET=x86_64-linux on x86_64)
3. `pixi run -e doris doris-build` — must succeed (Rust build with x86_64 paths)
4. `pixi run -e doris sirius-be` — process starts without shared library errors

### Wave 0 Gaps

None — no test files to create. This phase is build-system-only; verification is observational (command exit codes and process startup).

---

## Security Domain

> Security domain: not applicable. This phase modifies build configuration files (pixi.toml, build.rs) to fix architecture-specific paths. No authentication, session management, access control, input validation, or cryptography is involved. No ASVS categories apply.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `sysroot_linux-aarch64 = ">=2.32"` is the correct conda package name for the aarch64 sysroot | Standard Stack / Code Examples | pixi install still fails on aarch64; would need to find correct package name via `conda search` |
| A2 | nixl-sys line 114 is functionally harmless because nixl-build installs to `$CONDA_PREFIX/lib/` (not `lib/aarch64-linux-gnu/`), making line 112 sufficient | BUILD-04 Deep Dive | Link error "cannot find -lnixl" on aarch64 during cargo build; would require BUILD-04 fix to be Option A (patch) |
| A3 | pixi `env` block does not execute `$()` subshell commands — only expands `$VAR` references | Pattern 4 / Code Examples | LD_LIBRARY_PATH fix would not require bash -c wrapping; simpler solution exists |

---

## Open Questions

1. **Q-01: BUILD-04 — is D-07 intended to exclude the Rust binding's build.rs in the submodule?**
   - What we know: D-07 says "no changes to the nixl submodule" in the context of meson.build already supporting aarch64. The nixl submodule's `src/bindings/rust/build.rs` line 114 has the same bug as the published crate. The published crate is what nixl-test actually uses (from crates.io, confirmed by Cargo.lock).
   - What's unclear: Did the user intend to exclude the Rust binding's build.rs from changes? Is line 114 actually harmless (A2 assumption)?
   - Recommendation: Implement Option A (one-line fix + `[patch.crates-io]`) as the explicit fix. If the user reviews and confirms the fix is unnecessary based on A2, it can be skipped. Option A is 2 lines total and satisfies the requirement explicitly.

2. **Q-02: sirius-be task conversion — should the env block be removed entirely or kept empty?**
   - What we know: After converting to `bash -c` with LD_LIBRARY_PATH set inside the command, the `env` block is no longer needed.
   - What's unclear: Pixi may warn or error on an empty `env` block. Or other values may need to stay in `env`.
   - Recommendation: Remove the `env` block entirely from the converted tasks. The LD_LIBRARY_PATH is now set inside the `bash -c` command.

---

## Sources

### Primary (HIGH confidence)

- [VERIFIED: code] `pixi.toml` lines 64-110 — direct inspection; all 5 hardcoded paths confirmed
- [VERIFIED: code] `doris/crates/nixl-test/build.rs` — line 9 hardcoded path confirmed
- [VERIFIED: code] `.pixi/envs/default/etc/conda/activate.d/~cuda-nvcc_activate.sh` — sbsa-linux mapping pattern confirmed
- [VERIFIED: filesystem] `ls .pixi/envs/default/targets/` returns `sbsa-linux` on this aarch64 machine
- [VERIFIED: code] `doris/thirdparty/nixl/src/bindings/rust/build.rs` — line 114 confirmed
- [VERIFIED: code] `~/.cargo/registry/src/index.crates.io-.../nixl-sys-0.10.0/build.rs` — published crate has same line 114
- [VERIFIED: code] `doris/Cargo.lock` — `nixl-sys` source is crates.io, not local path
- [VERIFIED: code] `doris/thirdparty/nixl/meson.build` — does NOT contain `cpu_family` or `aarch64` in root build file; `benchmark/nixlbench/meson.build` does contain arch detection

### Secondary (MEDIUM confidence)

- [CITED: pixi.prefix.dev/latest/workspace/multi_platform_configuration/] Pixi platform-conditional dependency syntax

### Tertiary (LOW confidence)

- [ASSUMED] `sysroot_linux-aarch64` package name and availability in conda-forge — not verified via `conda search` in this session

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all dependency availability confirmed by existing pixi.toml and installed envs
- Architecture patterns: HIGH — all patterns derived from direct code inspection and locked decisions
- Pitfalls: HIGH — all verified from source code; sbsa-linux confirmed by conda activation script
- BUILD-04 assessment: MEDIUM — functional impact of line 114 not tested on real aarch64 build

**Research date:** 2026-04-06
**Valid until:** Stable — paths are static; no time-sensitive information. Re-verify if nixl-sys crate version changes or pixi syntax changes.
