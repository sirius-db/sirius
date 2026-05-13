---
phase: 23-update-cucascade-and-sirius-from-upstream
plan: 23-07
type: build-evidence
date: 2026-05-13
---

# Phase 23 Plan 23-07 Build Evidence

## Build 1: cucascade gitlink bump to 37df815 (Plan 23-06 fix only)

**Commit:** 15c47f5 submodule: bump cucascade to 37df815 (p23 dst_guard fix)
**Build result:** 128/128 PASS — exit 0
**Wall-clock:** incremental build, only cucascade representation_converter.cpp recompiled
**New warnings:** spdlog ACTIVE_LEVEL warning (baseline; not new)
**Smoke test [multi_gpu_foundation]:** 6/7 FAIL (cudaErrorInvalidResourceHandle at gpu_data_representation.cpp:106)

## Deviation: Rule 1 auto-fix — run_p2p_probe_locked device-restore bug

The Plan 23-06 dst_guard fix (37df815) closed the cudaErrorInvalidValue at line 628.
However, this exposed a second bug in cucascade's run_p2p_probe_locked: the function
ends with a hardcoded `cudaSetDevice(0)` that clobbers any caller-held RAII device
guard. This left device=0 after probe, causing cudaEventRecord to fail with
cudaErrorInvalidResourceHandle when target_stream (from gpu1's pool) didn't match
the current device (0).

Fix: Save current device at function entry with cudaGetDevice; restore with
cudaSetDevice(saved_device) at exit.

**Cucascade commit:** 9da4047 fix(p23): run_p2p_probe_locked must restore device context on exit
**Sirius gitlink bump:** 5c554d1 submodule: bump cucascade to 9da4047 (p23 probe-device-restore fix)

## Build 2: cucascade gitlink bump to 9da4047 (both fixes)

**Commit:** 5c554d1 submodule: bump cucascade to 9da4047 (p23 probe-device-restore fix)
**Build result:** 128/128 PASS — exit 0
**Wall-clock:** incremental build
**New warnings:** spdlog ACTIVE_LEVEL warning (baseline; not new)

Build output tail:
```
[125/128] Building CXX object extension/sirius/CMakeFiles/sirius_unittest.dir/test/cpp/scan_manager/test_pin_table_multi_gpu.cpp.o
[126/128] Linking CXX static library extension/sirius/CMakeFiles/sirius_extension.dir/cmake_device_link.o
[127/128] Linking CXX static library extension/sirius/libsirius_extension.a
[128/128] Linking CXX executable extension/sirius/test/cpp/sirius_unittest
```

## Smoke test: [multi_gpu_foundation] 7/7 PASS

```
All tests passed (38 assertions in 7 test cases)
```

**stderr:** `[cucascade] direct GPU↔GPU peer DMA broken on 2 direction(s); cudaMemcpyPeer* will host-stage automatically.`

This is the expected message on 2 x RTX 6000 Ada hardware (peer DMA broken; host-staging path active).

## Binary check

```
-rwxrwxr-x 1 felipe felipe 90946056 May 12 19:19 build/release/extension/sirius/test/cpp/sirius_unittest
```
(Rebuilt fresh after both fixes)

## Cucascade state

- cucascade HEAD: 9da4047 (8 commits ahead of bcddb89 origin/main)
- Sirius gitlink: 9da404756a8354d84d1dcd6bf3f3b46c29abfb3e (clean, no leading +)
- Cucascade commits ahead of upstream: 8 (was 6 pre-23-06, now 8 with two gap-closure fixes)

## Note on deviation

Plan 23-07 anticipated 1 additional cucascade commit (37df815 from Plan 23-06) bringing
the total to 7 commits ahead. The second bug (probe device restore) required an additional
cucascade commit (9da4047), bringing the total to 8 commits ahead. The ROADMAP and
23-CUCASCADE-DIFF.md will be updated to reflect 8 commits ahead.
