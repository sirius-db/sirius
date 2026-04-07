---
phase: 1
slug: build-and-runtime
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-06
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Manual verification (build system changes — no unit test framework applicable) |
| **Config file** | pixi.toml (build system configuration) |
| **Quick run command** | `pixi run -e doris doris-build 2>&1 | tail -5` |
| **Full suite command** | `pixi run -e doris doris-build && pixi run -e doris sirius-be --help` |
| **Estimated runtime** | ~300 seconds (first build), ~30 seconds (rebuild) |

---

## Sampling Rate

- **After every task commit:** Run `pixi install -e doris --dry-run` to verify dependency resolution
- **After every plan wave:** Run full build + start test
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 300 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 1-01-01 | 01 | 1 | BUILD-01 | — | N/A | manual | `pixi install -e doris` on aarch64 | ✅ | ⬜ pending |
| 1-01-02 | 01 | 1 | BUILD-02 | — | N/A | manual | `pixi run -e doris nixl-build` on aarch64 | ✅ | ⬜ pending |
| 1-01-03 | 01 | 1 | BUILD-03 | — | N/A | manual | `cargo build -p nixl-test` on aarch64 | ✅ | ⬜ pending |
| 1-01-04 | 01 | 1 | BUILD-04 | — | N/A | manual | `cargo build -p nixl-sys` on aarch64 | ✅ | ⬜ pending |
| 1-01-05 | 01 | 1 | RUNTIME-01 | — | N/A | manual | `pixi run -e doris sirius-be --help` on aarch64 | ✅ | ⬜ pending |
| 1-01-06 | 01 | 1 | ALL | — | N/A | manual | Full x86_64 build regression test | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Existing infrastructure covers all phase requirements. Build system changes are validated by running the build itself — no additional test framework needed.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| pixi resolves aarch64 deps | BUILD-01 | Requires aarch64 hardware | Run `pixi install -e doris` on aarch64 machine |
| nixl builds on aarch64 | BUILD-02 | Requires aarch64 + CUDA | Run `pixi run -e doris nixl-build` on aarch64 |
| nixl-test compiles on aarch64 | BUILD-03 | Requires aarch64 + CUDA | Run `cargo build -p nixl-test` on aarch64 |
| nixl-sys links correctly | BUILD-04 | Requires aarch64 libs | Run `cargo build -p nixl-sys` on aarch64 |
| BE starts on aarch64 | RUNTIME-01 | Requires aarch64 + GPU | Run `pixi run -e doris sirius-be` on aarch64 |
| x86_64 regression | ALL | Must verify no regressions | Full build + start on x86_64 |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 300s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
