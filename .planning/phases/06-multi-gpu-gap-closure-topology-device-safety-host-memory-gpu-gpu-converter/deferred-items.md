# Phase 06 — Deferred Items

Out-of-scope discoveries logged during plan execution. These are NOT fixed inline (per executor scope boundary rule); they are observations for future phases or parallel plans.

## Plan 06-02 (2026-04-21)

### Unit-test environmental failure on dev host (no NVIDIA driver)

- **Symptom:** `mcp__project-commands__run_command unit-tests` exits 255 after 294 test cases pass, when the iceberg test `gpu_execution iceberg - V1 basic scan` attempts a 25.4 GB GPU allocation.
- **Root cause:** `nvidia-smi` reports "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver." The current dev host has no working NVIDIA driver; RMM can't allocate GPU memory and throws `std::bad_alloc: out_of_memory` from `cuda_async_view_memory_resource.hpp:86`.
- **Scope:** Unrelated to Plan 06-02 edits (which only modify two `noexcept` lambdas in per-thread init callbacks). The 294 tests that ran before the crash include `Downgrade executor starts and stops cleanly` (test 76) and the full `bounded_thread_pool` suite (tests 99-111), all PASS — proving the edited callbacks work correctly.
- **Why not fixed here:** Plan 06-02 scope is "two-line cudaSetDevice wrap." Environment setup (GPU driver installation) is orthogonal to code changes.
- **Resolution path:** The MGPU-03 compute-sanitizer validation gate runs in Plan 06-04 on the N=2 verification host (`6f7e4c9-lcedt`, 2 × RTX 6000 Ada, driver 595.58.03) where the driver works. That host is the authoritative test bed for these callback paths.
