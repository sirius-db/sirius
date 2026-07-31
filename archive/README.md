# SF1000 compression campaign archive — 2026-07-31 (DO NOT MERGE)

Raw artifacts from the GB300 TPC-H SF1000 compressed-pinning campaign, preserved
ahead of a node wipe. Headline: dev baseline 20.74 s hot → 15.99 s (−22.9%) via
GPU-pinned Simpatico-compressed fact tables + downgrade trigger/stop tuning +
zero-copy scan materialize (PR #1361).

Contents:
- `simpatico-compressed-pinning.md`, `downgraded-task-prefetcher.md`,
  `scan-prefetch-overlap-design.md` — full campaign notes (results, dead ends,
  lessons; written as session memory, dates absolute).
- `plans_gpu_facts/` — the GPU-tier plan variants used by the 15.99 s recipe
  (lineitem with l_quantity→bitpack; orders with o_comment→identity).
- `plans_370_strict/` — strict-370 plan variant (below-bar columns → identity).
- `percol/` — per-column simpatico compress/decompress measurements on GB300
  (ratio + GB/s per codec, verified roundtrips), all 6 TPC-H tables.
- `hotspots/` — nsys hotspot reports for q1/q9/q13/q18/q21 (pre-#1361 build):
  operator efficiency, occupancy (cudf single_pass_shmem_aggs 6.3% occ,
  mixed_join 43.8%, ans decode 6.3%), sync attribution, memcpy hotspots.
- `run_logs/` — per-iteration timings for every experiment arm (A/B/C/D/E/M/G/H,
  trigger sweep, task-prefetcher sweep, concurrent-prefetch arms, heavy-5
  config experiments, final Y1/Y2 combination runs).

Related PRs: #1181/#1349 (prefetcher stack), #1351 (quent labels), #1352/#1356
(docs), #1353 (downgraded-task prefetcher RFC + experimental record), #1354
(plan fixes), #1355 (harness hooks), #1361 (zero-copy scan materialize),
NVIDIA/cuCascade#177 (NVTX + mask-skip).
Reproducibility branch (full experiment tree): felipeblazing/sirius
`claude/combo-prefetch-compression` (+ felipeblazing/cuCascade `combo-mask-skip`
for its submodule pointer).
