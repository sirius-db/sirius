# Sirius-Doris aarch64 Support

## What This Is

Porting the Doris integration layer (`doris/`) of the Sirius GPU SQL engine to work on aarch64 (ARM) platforms. The core Sirius C++ engine already supports aarch64, but the Doris build system, Rust build scripts, Docker deployment, and documentation have hardcoded x86_64 assumptions that prevent building and running on ARM.

## Core Value

The Doris integration layer builds and deploys on aarch64 NVIDIA platforms (Grace Hopper, Grace Blackwell, Vera Rubin) using the same build guide as x86_64.

## Requirements

### Validated

- ✓ Sirius C++ GPU engine builds on aarch64 — existing
- ✓ pixi.toml declares `platforms = ["linux-64", "linux-aarch64"]` — existing
- ✓ CUDA toolkit conda packages available for aarch64 — existing
- ✓ nixl Rust bindings have `get_arch()` detecting aarch64 — existing
- ✓ aarch64 Dockerfile exists at `docker/aarch64/stable/Dockerfile` — existing

### Active

- [ ] pixi doris environment resolves on aarch64 (sysroot dependency)
- [ ] nixl C++ library builds on aarch64 (CUDA paths in meson build)
- [ ] Rust BE binary compiles on aarch64 (CUDA stubs path in build.rs)
- [ ] nixl Rust bindings use correct library paths on aarch64 (hardcoded x86_64-linux-gnu)
- [ ] Sirius BE runs on aarch64 (LD_LIBRARY_PATH in pixi tasks)
- [ ] Docker deployment works on aarch64 (glibc loader in docker-compose)
- [ ] Documentation reflects aarch64 support

### Out of Scope

- Tegra/Jetson support — different CUDA target dir (`aarch64-linux` vs `sbsa-linux`), different use case
- Cross-compilation (building aarch64 binaries on x86_64) — native builds only
- Performance optimization on aarch64 — correctness first
- Doris C++ BE build on aarch64 — only the Sirius GPU BE (Rust)

## Context

- On aarch64, CUDA toolkit uses `targets/sbsa-linux/` (not `targets/aarch64-linux/`). Confirmed by conda's `~cuda-nvcc_activate.sh` and `.pixi/envs/default/targets/` directory.
- SBSA = Server Base System Architecture — NVIDIA's ARM spec for data center GPUs (Grace Hopper, etc.)
- The nixl thirdparty submodule is an external NVIDIA repo — changes there should be minimal and upstreamable.
- `sysroot_linux-64` is an x86_64-only conda package; the aarch64 equivalent is `sysroot_linux-aarch64`.
- Pixi supports platform-conditional dependencies via `[feature.X.target.linux-aarch64.dependencies]`.
- Docker compose files use NixOS-specific glibc loader workaround (`ld-linux-x86-64.so.2`).

## Constraints

- **Submodule**: nixl changes must be minimal (one-line fix) since it's an external NVIDIA repo
- **Backwards compatible**: All changes must preserve x86_64 functionality — runtime detection, not replacement
- **Build guide**: The same 4-step build process from `BUILD_DEPLOY_TEST_GUIDE.md` must work on both architectures

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Use `sbsa-linux` for aarch64 CUDA target dir | Confirmed by conda activation script and pixi env; Tegra uses `aarch64-linux` but we target data center GPUs | — Pending |
| Runtime arch detection via `uname -m` in shell, `cfg!(target_arch)` in Rust | Simplest cross-platform approach, no new dependencies | — Pending |
| Platform-conditional sysroot in pixi.toml | Pixi native feature, cleaner than runtime workaround | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-06 after initialization*
