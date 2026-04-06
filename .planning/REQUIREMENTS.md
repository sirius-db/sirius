# Requirements: Sirius-Doris aarch64 Support

**Defined:** 2026-04-06
**Core Value:** The Doris integration layer builds and deploys on aarch64 NVIDIA platforms using the same build guide as x86_64.

## v1 Requirements

### Build System

- [ ] **BUILD-01**: Pixi doris environment resolves on aarch64 (platform-conditional sysroot dependency)
- [ ] **BUILD-02**: nixl C++ library builds on aarch64 via meson (CUDA target paths use `sbsa-linux`)
- [ ] **BUILD-03**: nixl-test crate compiles on aarch64 (CUDA stubs path detects architecture)
- [ ] **BUILD-04**: nixl Rust bindings use correct library paths on aarch64 (fix hardcoded `x86_64-linux-gnu`)

### Runtime

- [ ] **RUNTIME-01**: Sirius BE starts on aarch64 with correct LD_LIBRARY_PATH (both sirius-be and sirius-be-2 tasks)

### Documentation

- [ ] **DOCS-01**: BUILD_DEPLOY_TEST_GUIDE.md documents aarch64 as a supported platform

## v2 Requirements

### Performance

- **PERF-01**: UCX transport tuning for Grace Hopper coherent memory (NVLink-C2C)
- **PERF-02**: Performance benchmark comparison aarch64 vs x86_64

### CI/CD

- **CI-01**: Automated aarch64 build in CI pipeline
- **CI-02**: Multi-arch Docker image builds (buildx)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Tegra/Jetson support | Different CUDA target dir (`aarch64-linux` vs `sbsa-linux`), different use case |
| Cross-compilation | Native aarch64 builds only; cross-compile adds complexity for minimal benefit |
| aarch64 performance optimization | Correctness first; optimization is v2 |
| Doris C++ BE on aarch64 | Only the Sirius GPU BE (Rust) — standard Doris BE is a separate project |
| CUDA architecture flag changes | Existing CUDAARCHS in pixi.toml already covers aarch64-relevant GPUs |
| Docker compose aarch64 fix | NixOS-specific deployment, not used in standard build flow |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| BUILD-01 | Phase 1 | Pending |
| BUILD-02 | Phase 1 | Pending |
| BUILD-03 | Phase 1 | Pending |
| BUILD-04 | Phase 1 | Pending |
| RUNTIME-01 | Phase 1 | Pending |
| DOCS-01 | Phase 2 | Pending |

**Coverage:**
- v1 requirements: 6 total
- Mapped to phases: 6
- Unmapped: 0 ✓

---
*Requirements defined: 2026-04-06*
*Last updated: 2026-04-06 after initial definition*
