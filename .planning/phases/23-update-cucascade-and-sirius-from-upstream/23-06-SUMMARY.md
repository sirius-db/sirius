---
plan: 23-06
phase: 23-update-cucascade-and-sirius-from-upstream
status: complete
gap_closure: true
created: 2026-05-13T00:16:20Z
tasks: 2/2
requirements: [MERGE-CC-23, GAUNTLET-23]
key_files:
  modified:
    - cucascade/src/data/representation_converter.cpp
  created:
    - .planning/phases/23-update-cucascade-and-sirius-from-upstream/23-06-CUCASCADE-CTEST.md
    - .planning/phases/23-update-cucascade-and-sirius-from-upstream/23-06-SUMMARY.md
commits:
  - repo: cucascade
    sha: 37df8153bf8330203954da99d341a139fcedd18c
    short: 37df815
    subject: "fix(p23): cuda_set_device_raii guard for HtoD in alloc_and_peer_copy_async"
hand_off:
  cucascade_head_for_next_plan: 37df8153bf8330203954da99d341a139fcedd18c
  helper_file: /tmp/claude/p23_06_new_cucascade_head.txt
  sirius_gitlink_state: unchanged (still 1e889d7)  # Plan 23-07 bumps it
---

# Plan 23-06 — Cucascade `alloc_and_peer_copy_async` dst_guard fix

## Outcome

Surgical one-scope fix to `cucascade/src/data/representation_converter.cpp`'s `alloc_and_peer_copy_async`: the host-staging HtoD `cudaMemcpyAsync` is now wrapped in `rmm::cuda_set_device_raii dst_guard{rmm::cuda_device_id{dst_device}}` so the destination device's CUDA context is active. This closes the root cause of VERIFICATION.md gaps #1 (REG-05 [mgpu_stress]) and #2 (REG-06 Leg 1 [multi_gpu_foundation]). Plan 23-07 will bump the sirius gitlink to pick up the fix and re-run the gauntlet.

## Tasks

### Task 1 — Edit `alloc_and_peer_copy_async` host-staging branch

Per plan: wrap the HtoD `cudaMemcpyAsync` + `cudaStreamSynchronize` pair at line ~628 in a new scope that opens with `rmm::cuda_set_device_raii dst_guard{rmm::cuda_device_id{dst_device}}`. The exact diff is captured in `23-06-CUCASCADE-CTEST.md`.

Done. The diff matches the plan's verbatim "Replace this / With this" block.

### Task 2 — Commit on cucascade fork (no push)

Committed as `37df815` on `fix/pinned-portable-flags`. The branch is now 7 commits ahead of `origin/main` (was 6). No `git push origin` — local-fork-only per CC-UPSTREAM-01.

Helper file `/tmp/claude/p23_06_new_cucascade_head.txt` written with the SHA `37df8153bf8330203954da99d341a139fcedd18c` for Plan 23-07 to read.

## Smoke build

MCP `build` returned `[128/128]` — sirius extension and `sirius_unittest` linked. Note: sirius gitlink is still pinned to `1e889d7` at this point, so the smoke build verifies sirius still builds, not that the fix is wired in. The wiring happens in Plan 23-07 Task 1.

## Invariant grep checks

| Invariant | Pre | Post |
|-----------|-----|------|
| `rmm::cuda_stream_default` in modified file | 0 | 0 |
| Peer-DMA path (`cudaMemcpyPeerAsync`) byte-identical to pre-fix | yes | yes |
| `src_guard` scope at line ~619 byte-identical to pre-fix | yes | yes |
| `dst_guard` scope at line ~646 added exactly once | n/a | yes |
| `target_stream` used for both DtoH and HtoD | yes | yes (Cluster B invariant preserved) |

## Deviations

None — the diff matches the plan's verbatim "Replace this / With this" block exactly.

One operational note: the initial executor agent for this plan hit an `Internal server error` mid-run after applying the edit but before committing or writing evidence. The orchestrator (Claude) finished the plan in-place — verified the edit matched the plan, ran the smoke build via MCP, committed cucascade, wrote the helper file, the CUCASCADE-CTEST evidence file, and this SUMMARY. Cleaner than respawning the agent and re-doing the same edit.

## Next plan

`/gsd:execute-phase 23 --gaps-only` continues to Plan 23-07:
- Bump sirius cucascade gitlink to `37df815`
- MCP rebuild
- Rerun REG-05, REG-06 Leg 1, REG-06 Leg 2 (previously SKIPped)
- Fix `test/scripts/sanitizer_gate_22.sh` cluster_B false-positive
- Flip 23-VERDICT.md PARTIAL → PASS
- Update REQUIREMENTS / STATE / ROADMAP
