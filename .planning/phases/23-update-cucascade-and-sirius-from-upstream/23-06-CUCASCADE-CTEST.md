---
plan: 23-06
type: cucascade-evidence
created: 2026-05-13T00:16:20Z
cucascade_head_before: 1e889d7e67070de7dc88860c373622182afe35df
cucascade_head_after: 37df8153bf8330203954da99d341a139fcedd18c
fork_branch: fix/pinned-portable-flags
fork_commits_ahead_of_origin: 7
sirius_gitlink_after_this_plan: 1e889d7e67070de7dc88860c373622182afe35df  # UNCHANGED — bumped in Plan 23-07
git_push_origin: NONE  # CC-UPSTREAM-01
---

# Plan 23-06 — Cucascade `alloc_and_peer_copy_async` dst_guard fix — Evidence

## Commit

`37df8153bf8330203954da99d341a139fcedd18c` on `fix/pinned-portable-flags`:

```
fix(p23): cuda_set_device_raii guard for HtoD in alloc_and_peer_copy_async
```

`1 file changed, 23 insertions(+), 3 deletions(-)` — `cucascade/src/data/representation_converter.cpp` only.

## Diff (semantic)

The HtoD `cudaMemcpyAsync` at the post-host-staging tail of `alloc_and_peer_copy_async` was bare; it is now wrapped in a `rmm::cuda_set_device_raii dst_guard{rmm::cuda_device_id{dst_device}}` scope so the destination device's CUDA context is active for the duration of the copy + sync.

The peer-DMA-works branch (around line 603–608, `cudaMemcpyPeerAsync`) and the `src_guard` scope (around line 619) are unchanged. `target_stream` is used for both the DtoH and HtoD copies — the new guard sets the CUDA *context*, not the stream, so the Phase 22 Cluster B same-stream invariant is preserved.

## Invariant grep checks

| Check | Result |
|-------|--------|
| `grep -c rmm::cuda_stream_default cucascade/src/data/representation_converter.cpp` | 0 |
| Peer-DMA path `cudaMemcpyPeerAsync(buf.data(), dst_device, src_ptr, src_device, size, target_stream.value())` exists verbatim | yes |
| `src_guard` at line ~619 still `rmm::cuda_set_device_raii src_guard{rmm::cuda_device_id{src_device}}` | yes |
| New `dst_guard` line: `rmm::cuda_set_device_raii dst_guard{rmm::cuda_device_id{dst_device}}` exists exactly once | yes (line 646) |

## Smoke build (MCP)

```
mcp__project-commands__run_command name=build
```

Result: `[128/128]` — extension and `sirius_unittest` both linked. Only pre-existing benign warnings (SPDLOG_ACTIVE_LEVEL, telemetry-bridge `operator` keyword). No new errors or warnings introduced by this commit.

(Sirius parent's cucascade gitlink is still pinned to `1e889d7e67070de7dc88860c373622182afe35df` at this point, so the smoke build was against the PRE-fix cucascade tree — it verifies sirius still builds, not that the fix is wired in. The fix is wired in by the gitlink bump in Plan 23-07.)

## Sirius parent state at end of Plan 23-06

- Branch: `feature/single-node-multi-gpu2`
- HEAD: `ef81cf8` (unchanged from start of Plan 23-06 — this plan did not commit anything to sirius)
- Cucascade submodule index pointer: `1e889d7` (unchanged from start of Plan 23-06)
- Cucascade submodule working-tree HEAD: `37df815` (one commit ahead of index — Plan 23-07 will bump)

`git status` in sirius shows ` m cucascade` (lowercase m: submodule working tree differs from index). This is intentional and expected — Plan 23-07's first commit is `git add cucascade && git commit` to bump the gitlink to `37df815`.

## Hand-off to Plan 23-07

- New cucascade HEAD SHA: `37df8153bf8330203954da99d341a139fcedd18c`
- Helper file: `/tmp/claude/p23_06_new_cucascade_head.txt`
- Expected first action of Plan 23-07 Task 1: bump sirius gitlink to this SHA, commit as standalone "submodule: bump cucascade to 37df815 (Phase 23 gap-closure)" — atomic per D-12.

## CC-UPSTREAM-01

This fix lives on the local cucascade fork branch `fix/pinned-portable-flags`. No `git push` was issued. The fork is now 7 commits ahead of `origin/main` (was 6). 23-07 Task 5 will update `23-CUCASCADE-DIFF.md` to reflect the new commit count.
