---
phase: 24-update-cucascade-and-sirius-from-upstream-round-2
type: phase-verdict
status: PASS
verdict_date: 2026-05-13
requirements_satisfied: [MERGE-CC-24, MERGE-DEV-24, GAUNTLET-24]
branch: feature/single-node-multi-gpu2
head_commit: a1909c8
head_commit_at_close: a1909c8
prior_phase_head: 3520db7
cucascade_pin: 5203de5a028ccb57402a4105e35282c567c3ee5a
cucascade_pin_short: 5203de5
prior_cucascade_pin: 9da4047
upstream_base: 9ceebaa
upstream_base_subject: "Fix for: Invalid Error: reconstruct_column STRING column metadata must have at least one child (offsets) (#124)"
hardware: "2 x NVIDIA RTX 6000 Ada Generation (49 GB each), CUDA 13.0"
hyg_02: 40
cucascade_commits_ahead_of_9ceebaa: 9
build_at_close: PASS
---

# Phase 24 Verdict: Update cucascade + sirius from upstream (round 2) — PASS

## Overview

Phase 24 goal: pull 2 new upstream cucascade commits (`96bfea1` "feat: adding the ability to slice host
table" #122, and `9ceebaa` "Fix for: Invalid Error: reconstruct_column STRING #124") plus 2 new sirius
commits (`ba5ed27` "refactor: split wire_data_repositories Phase 2" #770, `2e197c6` "feat(pin_table):
support tier='host'" #774) into our forks, repeating the Phase 23 pattern with D-01
upstream-as-source-of-truth triage for all conflicts.

**Result: PASS.** All 18 invariant gates PASS on first attempt — no gap-closure plans needed. Two
improvements over the Phase 23 baseline: REG-06 Leg 1 memcheck advanced from PARTIAL (6/7) to PASS
(7/7); the new D-07 gate (pin_table tier='host' smoke) passed 1/1 via the upstream test supplied by
`2e197c6`. The branch is bisectable via D-04 atomic commits (A through F); no `git push origin`
executed (D-06); cucascade gitlink stays on the local fork (D-05).

| Gate | Result | Note |
|------|--------|------|
| REG-01 `[mgpu]` | **PASS** | 16/16, 79091 assertions, 127.9s |
| REG-02 `[TPC-H][parquet]` | **PASS** | 22/22, 36256 assertions, 109.6s |
| REG-03 `[integration][TPC-H]` | **PASS** | 49/49, 71623 assertions, 211.1s |
| REG-04 SF100 Q1 num_gpus=2 | **PASS** | 4 rows byte-identical; ~7.0s cold-shell |
| REG-05 `[mgpu_stress]` | **PASS** | 1/1, 77053 assertions, 82.4s |
| REG-06 Leg 1 functional `[multi_gpu_foundation]` | **PASS** | 7/7, 38 assertions, 5.7s |
| REG-06 Leg 1 memcheck | **PASS (improved)** | 7/7 — cudf lib violations absent; was 6/7 PARTIAL in Phase 23 |
| REG-06 Leg 2 memcheck | **PASS** | 42/42, 1,922,202 assertions, 0 new violations |
| `[datasource_factory]` | **PASS** | 11/11, 38 assertions, 4.8s |
| `[tpch_sf10]` (K.7 NO-REPRO) | **PASS** | 4/4, 64 assertions, 6.5s |
| `[mgpu-audit]` | **PASS** | 6/6, 103 assertions, 12.0s |
| HYG-02 `rmm::cuda_stream_default` | **PASS** | 40 (≤40 limit; all in src/legacy/) |
| GATE-22.1-A kvikio bypass-grep | **PASS** | 0 hits |
| GATE-22.1-B sanitizer Cluster A | **PASS** | cluster_A=0 |
| GATE-22.1-C SF1 Q11 num_gpus=2 | **PASS** | 1/1, 9011 assertions, 9.8s |
| K.6 NO-REPRO SF100 Q11 | **PASS** | exit 0, 0 rows, ~3.8s |
| K.7 NO-REPRO | **PASS** | Covered by `[tpch_sf10]` 4/4 |
| Phase 22 Cluster B same-stream | **PASS** | cluster_B=0 via sanitizer_gate_22.sh |
| **D-07 NEW: `[pin_table_host]` smoke** | **PASS (new gate)** | 1/1, 51 assertions, 6.6s — upstream test from 2e197c6 |
| PIN-MGPU-01 coexistence `[pin_mgpu]` | **PASS** | 2/2, 46 assertions, 9.5s |

**18/18 gates PASS.** Zero regressions. Two improvements. Phase 24 is complete.

---

## Section A: REG-01 [mgpu]

**Gate:** 16/16 PASS, ≥79091 assertions, ≤130s wall-clock, exit 0
**Status: PASS**

Evidence (`/tmp/claude/p24_04_reg01_mgpu.log`):
- Test count: 16/16
- Assertion count: 79091
- Wall-clock: 127.9s
- Exit code: 0

No regression vs Phase 23 baseline (16/16, 79091, 125.2s). The 2.7s timing delta is within normal
measurement variance (test-suite load conditions, GPU thermal state). The origin/dev merge (ba5ed27
wire_data_repositories + 2e197c6 host-tier pin_table) did not regress multi-GPU correctness.

---

## Section B: REG-02 [TPC-H][parquet]

**Gate:** 22/22 PASS, ≥36256 assertions, exit 0
**Status: PASS**

Evidence (`/tmp/claude/p24_04_reg02_parquet.log`):
- Test count: 22/22
- Assertion count: 36256
- Wall-clock: 109.6s
- Exit code: 0

Exact match to Phase 23 baseline (22/22, 36256 assertions). The 0.2s delta is noise.

---

## Section C: REG-03 [integration][TPC-H]

**Gate:** 49/49 PASS, ≥71623 assertions, exit 0
**Status: PASS**

Evidence (`/tmp/claude/p24_04_reg03_integration.log`):
- Test count: 49/49
- Assertion count: 71623
- Wall-clock: 211.1s
- Exit code: 0

Phase 23 baseline was 49/49, 71623 assertions — same as Phase 24. The Phase 23 merge had added
+1 test / +16 assertions vs the prior baseline; those persist. The 24-03 merge did not add further
integration tests (ba5ed27's new wire_data_repositories materializer is a refactor, not a test
addition; 2e197c6's new `[pin_table_host]` test is under Section O).

---

## Section D: REG-04 SF100 TPC-H Q1 num_gpus=2

**Gate:** Result byte-identical to 1-GPU baseline; 4 rows correct; exit 0
**Status: PASS**

Evidence (`/tmp/claude/p24_04_reg04_sf100_q1.log`):
- Result: 4 rows (byte-identical to Phase 23 and Phase 22 baselines)
- Wall-clock: ~7.0s (cold shell invocation, includes ~3.5s DuckDB process startup)
- Exit code: 0

**Timing note:** Phase 23 measured 3.048s for a warm in-process benchmark iteration. Phase 24
uses cold shell invocations (~7.0s including DuckDB init + GPU context setup). The actual GPU query
execution time (subtracting ~3.5s shell overhead) is consistent with the 3.0s Phase 23 in-process
baseline. Query correctness (4 rows, byte-identical) is the authoritative gate criterion.

---

## Section E: REG-05 [mgpu_stress]

**Gate:** 1/1 PASS, ≥77053 assertions, exit 0
**Status: PASS**

Evidence (`/tmp/claude/p24_04_reg05_mgpu_stress.log`):
- Test count: 1/1 PASS
- Assertion count: 77053
- Wall-clock: 82.4s
- Exit code: 0

Phase 23 baseline: 1/1, 77053, 83.7s. Exact assertion match; 1.3s timing delta is noise. The
Phase 23 REG-05 FAIL was caused by the cudaErrorInvalidValue in `alloc_and_peer_copy_async` before
the `dst_guard` / probe-restore fixes (Plans 23-06/23-07). Those fixes carried through Phase 24's
cucascade rebase as commits 7 (4319726) and 8 (1522e0b). The Phase 24 500-iter stress test PASSES
on first attempt — no gap-closure needed.

---

## Section F: REG-06 HYG-02 + Leg 1 + Leg 2

### F.1 HYG-02

**Gate:** ≤40 (Phase 22.x baseline); ≤43 (D-30 budget)
**Status: PASS**

Evidence (`/tmp/claude/p24_04_grep_gates.txt`):
- `grep -rn 'rmm::cuda_stream_default' src/ | wc -l` = **40**
- All 40 in `src/legacy/` (unchanged from Phase 22 baseline)

The 24-03 origin/dev merge (ba5ed27 + 2e197c6) did not introduce new `rmm::cuda_stream_default`
usages in active Super Sirius code. The D-30 budget (≤43) and Phase 22.x baseline (≤40) both hold.

### F.2 REG-06 Leg 1 — functional [multi_gpu_foundation]

**Gate:** 7/7 PASS, 38 assertions, exit 0
**Status: PASS**

Evidence (`/tmp/claude/p24_04_reg06_leg1_functional.log`):
- Test count: 7/7 PASS
- Assertions: 38
- Wall-clock: 5.7s
- Exit code: 0

Phase 23 baseline: 7/7, 38, 5.7s — exact match. The cucascade `dst_guard` (commit 4319726,
re-derived from `37df815`) and probe-device-restore (commit 1522e0b, re-derived from `9da4047`)
remain intact on the rebased fork. The `gpu_to_gpu round-trip preserves bytes` test passes
without gap-closure.

### F.3 REG-06 Leg 1 — memcheck [multi_gpu_foundation] — IMPROVED

**Gate:** 7/7 PASS, 38 assertions, 0 NEW violations
**Status: PASS (improved from Phase 23 PARTIAL)**

Evidence (`/tmp/claude/p24_04_reg06_leg1_memcheck.log`):
- Test count: 7/7 PASS
- Assertions: 38
- ERROR SUMMARY: 7 errors (all `cudaErrorPeerAccessAlreadyEnabled` error 704 API-error backtraces
  from `probe_peer_dma_works` during GPU init — confirmed pre-existing; same class as Leg 2's
  6 API-error backtraces)
- No `Invalid __global__ read` / `Use-before-alloc` race findings detected
- Exit code: 0

**Phase 23 PARTIAL context:** Phase 23 Section F.2 noted 6/7 under compute-sanitizer due to
pre-existing `cudf::detail::contiguous_split Invalid __global__ read` violations in `libcudf.so`
from the `compute_batch_checksum_fnv1a64` path. These 94 violations were in the cudf library, not
in sirius or cucascade code.

**Phase 24 improvement:** Those cudf library violations are absent in this run. All 7 sanitizer
errors are the `cudaErrorPeerAccessAlreadyEnabled` (error 704) API-error backtraces from
`probe_peer_dma_works` — the same class as the Leg 2 pre-existing baseline (6 API-error
backtraces). This is a PASS: 7/7 tests pass, 0 new violations. The improvement is likely a
cudf version or workload-path difference, not a Phase 24 fix.

### F.4 REG-06 Leg 2 — memcheck [integration][gpu_execution][parquet][join]

**Gate:** 42/42 PASS, ≥1.92M assertions, 0 NEW violations
**Status: PASS**

Evidence (`/tmp/claude/p24_04_reg06_leg2_memcheck.log`):
- Test count: 42/42 PASS
- Assertions: 1,922,202
- Bytes leaked: 0
- ERROR SUMMARY: 6 errors (all `cudaErrorPeerAccessAlreadyEnabled` error 704 — pre-existing baseline)
- Exit code: 0

Exact match to Phase 23 Leg 2 baseline (42/42, 1,922,202, 6 pre-existing API-error backtraces).
No new violations introduced by the Phase 24 cucascade rebase or sirius merge.

---

## Section G: GATE-22.1-A bypass-grep

**Gate:** 0 hits (`grep -rn 'cudf::io::datasource::create\|cudf::io::source_info{' src/ | grep -v
data_source.get()\|datasource.get() | grep -v ^[^:]*:.*//'`)
**Status: PASS**

Evidence (`/tmp/claude/p24_04_grep_gates.txt`):
- Line count: **0**

The kvikio-free invariant introduced in Phase 22.1 is fully preserved through the 24-03
origin/dev merge (ba5ed27 + 2e197c6). The sirius `2e197c6` commit adds host-tier pin_table
reading via `sirius_ioctx::make_datasource` (the standard Phase 22.1 bypass pattern); it does NOT
introduce new `cudf::io::datasource::create(path)` calls. GATE-22.1-A holds.

---

## Section H: GATE-22.1-B sanitizer Cluster A

**Gate:** cluster_A = 0 (Phase 22.1 K.1 Cluster A closed)
**Status: PASS**

Evidence (`/tmp/claude/p24_04_sanitizer_gate_full.log`):
- `cluster_A=0`

Cluster A (cudf+kvikio `read_column_chunks_async` races) remains 0. The Phase 22.1 kvikio removal
holds through the Phase 24 merge. No new cudf+kvikio-adjacent code paths were introduced.

---

## Section I: GATE-22.1-C SF1 Q11 num_gpus=2 functional

**Gate:** PASS; result correct; 0 `cudaSetDevice(-1)` errors
**Status: PASS**

Evidence (`/tmp/claude/p24_04_gate22_1c.log`):
- MCP unit-test `[integration][gpu_execution][parquet][TPC-H][Q11]`: 1/1 PASS, 9011 assertions,
  9.8s
- 0 `cudaSetDevice(-1)` errors (Phase 22.2 K.6 downgrade_executor fix preserved)
- Exit code: 0

Exact match to Phase 23 baseline (1/1, 9011 assertions, 9.8s). The K.6 fix (Plan 22.2) was in
`src/downgrade/downgrade_executor.cpp` — not touched by either ba5ed27 or 2e197c6.

---

## Section J: Phase 22 Cluster B same-stream invariant

**Gate:** sanitizer_gate_22.sh returns cluster_B=0; P22_SELFTEST PASS
**Status: PASS**

Evidence (`/tmp/claude/p24_04_sanitizer_gate_full.log`, `/tmp/claude/p24_04_sanitizer_gate_selftest.log`):
- `P22_SELFTEST=1` exit code: 0 (SELFTEST PASS)
- Full run: `cluster_B=0`, `cluster_A=0`, `total_races=0`
- Exit code: 0

The windowed-awk `sanitizer_gate_22.sh` script (introduced in Plan 23-07, commit `0a3e2a7`) functions
correctly on the Phase 24 tree. The `ba5ed27` repository_wiring split did NOT introduce new symbols
that triggered false positives in the script — no extension of the API-error filter was needed.

The Phase 22 same-stream invariant (`alloc_and_peer_copy_async` uses a single `target_stream`
for all legs — committed as cucascade `b21bd97`, re-derived from `1e889d7`) holds on the rebased
cucascade fork. `total_races=0` confirms no stream-ordered race findings.

---

## Section K: K.6 NO-REPRO (SF100 Q11 num_gpus=2)

**Gate:** 0 `cudaSetDevice(-1)` errors; exit 0; correct result
**Status: PASS**

Evidence (`/tmp/claude/p24_04_k6_sf100_q11.log`):
- SF100 Q11 num_gpus=2: exit 0, 0 rows (spec-compliant — fraction 0.0001 at SF100 yields 0 rows)
- 0 CUDA errors in output
- Wall-clock: ~3.8s
- Exit code: 0

Phase 22.2 `downgrade_executor` fix (gate `_space_id.tier == cucascade::memory::Tier::GPU` for stream
pool creation and per-thread init) preserved through Phase 24. K.6 is confirmed NO-REPRO for the
third consecutive phase.

---

## Section L: K.7 NO-REPRO (SF10 Q11 spec-compliant test)

**Gate:** `[tpch_sf10]` 4/4 PASS including Phase 22.3 spec-compliant `tpch_q11_sf10_2gpu` test
**Status: PASS**

Evidence (`/tmp/claude/p24_04_tpch_sf10.log`):
- `[tpch_sf10]` suite: 4/4 PASS, 64 assertions, 6.5s, exit 0
- Test `tpch_q11_sf10_2gpu` (test_gpu_execution_tpch.cpp:4415) is present and passes

K.7 was reclassified NO-REPRO in Phase 22.3 (SQL fixture used constant `0.0001` instead of
spec-compliant `0.0001/SF`; at SF10+ the threshold exceeds max single-partkey value, so 0 rows
is correct — DuckDB CPU agrees). The spec-compliant test from Phase 22.3 gates against regression
and passes in Phase 24.

---

## Section M: Phase 22.x invariant grep gates

All grep gates confirmed present post-Phase-24-merge:

| Invariant | Phase 24 Result | Log Path |
|-----------|-----------------|----------|
| HYG-02 `rmm::cuda_stream_default` | 40 (≤40 limit; all in src/legacy/) | p24_04_grep_gates.txt |
| kvikio bypass-grep | 0 hits (no `cudf::io::datasource::create`) | p24_04_grep_gates.txt |
| Phase 22.2 `drain_after_error` presence | 4 sites (task_scheduler.cpp:203; sirius_engine.cpp:161,165,183) | p24_04_grep_gates.txt |
| Phase 14 SCHED-RR presence | 4 hits (task_scheduler.cpp:156,160,253,261) | p24_04_grep_gates.txt |
| Phase 22.3 CTE `producer_types` | 2 hits (sirius_plan_cte.cpp:52,56) | p24_04_grep_gates.txt |
| Phase 22.2 downgrade tier gate | 3 hits (downgrade_executor.cpp:79,89,182) | p24_04_grep_gates.txt |
| PIN-MGPU-01 `chunk_memory_spaces` | 42 hits (see note below) | p24_04_grep_gates.txt |
| Phase 22.3 SF10 Q11 regression test | 2 hits (test_gpu_execution_tpch.cpp:4415,4425) | p24_04_grep_gates.txt |
| cucascade gitlink | `5203de5` (Phase 24 rebased fork HEAD) | git submodule status |

**`chunk_memory_spaces` count note:** Count dropped from 60 (Phase 23) to 42 (Phase 24). This is a
non-regression: the 24-03 merge integrated upstream `2e197c6`'s host-tier path which uses
`host_chunks`, `tier`, and `memory_space` fields for the HOST branch in `cached_split_provider`
and `sirius_scan_manager`, not `chunk_memory_spaces`. The PIN-MGPU-01 GPU-tier round-robin path
still uses `chunk_memory_spaces`; functional coexistence verified by `[pin_mgpu]` PASS 2/2,
`[mgpu-audit]` PASS 6/6, and `[pin_table_host]` PASS 1/1. The count drop reflects the
"integrate-both" merge strategy, not a PIN-MGPU-01 regression. Accepted at approval checkpoint.

---

## Section N: Cucascade pin verification

**Gate:** `git submodule status cucascade` SHA matches Plan 24-02 post-rebase HEAD
**Status: PASS**

```
 5203de5a028ccb57402a4105e35282c567c3ee5a cucascade (heads/fix/pinned-portable-flags)
```

Matches Plan 24-02 post-rebase HEAD (`5203de5`). The cucascade submodule is pinned to the correct
commit: 9 commits ahead of `9ceebaa` (upstream `origin/main` PR #124 base). D-05 ours-wins
enforcement verified: `2e197c6`'s gitlink conflict (upstream proposed `96bfea1`) was resolved to
our fork HEAD `5203de5` per D-05 policy.

---

## Section O: D-07 NEW gate — pin_table tier='host' smoke

**Gate:** 1/1 PASS, >0 assertions, exit 0 (D-07 new gate for Phase 24)
**Status: PASS (new gate)**

Evidence (`/tmp/claude/p24_04_pin_table_host.log`):
- Test count: 1/1 PASS
- Assertions: 51
- Wall-clock: 6.6s
- Exit code: 0

**Gate source:** Upstream commit `2e197c6` "feat(pin_table): support tier='host' for host-tier
caching" added `[pin_table_host]` Catch2 integration test at
`test/cpp/integration/test_gpu_execution_tpch.cpp:4556` (tag string:
`[integration][gpu_execution][parquet][pin_table_host]`, test name: "gpu_execution — pin_table
host tier scan and aggregate").

**D-04 Commit E disposition: Branch A — upstream test exists.** No new test file needed;
the upstream test is the durable, bisectable artifact per D-04 intent. Detection via source-level
grep (binary `--list-tags` fails on this host when GPUs absent at tag-listing time).

**D-01 application:** The upstream test exercises the new `tier='host'` path added by `2e197c6`.
Our integration both gates: (a) PIN-MGPU-01 GPU-tier round-robin doesn't break host-tier, and
(b) host-tier doesn't break GPU-tier. The `[pin_mgpu]` 2/2 PASS and `[mgpu-audit]` 6/6 PASS
provide additional coexistence coverage.

---

## Section P: Sanitizer baseline diff vs Phase 23

**Gate:** Zero NEW sirius-frame or cucascade-frame races vs Phase 23 baseline
**Status: PASS** (no new actual races; Cluster B 0; Cluster A 0)

From `sanitizer_gate_22.sh` and individual sanitizer runs:
- `total_races=0` — zero stream-ordered race findings from racecheck
- `cluster_A=0` — no Cluster A (kvikio) races
- `cluster_B=0` — Cluster B same-stream invariant holds

The Phase 24 tree shows no new sirius-frame or cucascade-frame races vs Phase 23. The `ba5ed27`
repository_wiring split and `2e197c6` host-tier path do not introduce stream-ordering hazards.
The `alloc_and_peer_copy_async` same-stream invariant (cucascade commit `b21bd97`) continues
to hold; `total_races=0` confirms this.

REG-06 Leg 1 memcheck improvement vs Phase 23: the `cudf::detail::contiguous_split Invalid
__global__ read` violations from libcudf.so that were present in Phase 23's Leg 1 memcheck run
are absent in Phase 24. All 7 sanitizer errors are pre-existing `cudaErrorPeerAccessAlreadyEnabled`
API-error backtraces from `probe_peer_dma_works` — not race findings.

---

## Section Q: Conflict resolution audit

**Gate:** All conflict files resolved correctly; resolution logic documented
**Status: PASS — see `24-CONFLICT-LOG.md`**

Phase 24 resolved conflicts in two stages:

**Part 1 — Cucascade rebase (Plan 24-02):**
- 1 RE-DERIVE conflict: `cucascade/src/data/representation_converter.cpp`
  - Conflict in `convert_host_fast_to_gpu()` at `fast_table->allocation` vs `*fast_table->allocation`
  - Resolution: take upstream's `*` dereference (shared_ptr API from `96bfea1`) AND keep our
    `target_stream` (multi-GPU correctness; avoids cudaErrorInvalidValue when caller stream belongs
    to different device context)
  - Result: `reconstruct_column(col_meta, *fast_table->allocation, target_stream, mr, batch)`
- 8 commits CLEAN (applied without conflict; see 24-CONFLICT-LOG.md Part 1 for per-commit detail)
- 1 additional test-fix commit (5203de5): upstream `96bfea1` added slice-roundtrip test using old
  2-arg `gpu_table_representation` constructor; our commit `c15cb01` requires 3 args (writer_stream);
  fixed by adding `stream.view()` as 3rd arg

**Part 2 — Sirius origin/dev merge (Plan 24-03):**
All 9 conflict files driven by `2e197c6`'s cucascade API changes colliding with our `ff06fac`
pre-adaptation (D-04 Commit B — API adapter for 96bfea1 private constructor):

| File | Strategy | Rationale |
|------|----------|-----------|
| `cucascade` gitlink | OURS-WINS | D-05: our fork `5203de5` descends from `96bfea1` |
| `multiple_blocks_allocation_accessor.hpp` | UPSTREAM | D-01: 3-line comment before template |
| `host_table_chunk_reader.hpp` | INTEGRATE | Upstream method sigs + our value-type private field |
| `host_table_chunk_reader.cpp` | INTEGRATE | Upstream shared_ptr params + our flexible template |
| `cached_split_provider.hpp` | INTEGRATE BOTH | `_chunk_memory_spaces` (PIN-MGPU-01) + upstream HOST-tier fields |
| `cached_split_provider.cpp` | INTEGRATE BOTH | Our GPU ctor + upstream's new HOST ctor |
| `sirius_scan_manager.hpp` | INTEGRATE BOTH | Our `chunk_memory_spaces` + upstream `host_chunks`/`tier`/`memory_space` |
| `sirius_scan_manager.cpp` | INTEGRATE BOTH | Our `chunk_memory_spaces` move + upstream `tier=GPU` assignment |
| `sirius_pipeline_converter.cpp` | PARTIAL OURS | Keep `configure_partition_min_partitions()`, drop `log_pipeline_debug_info()` |
| `sirius_extension.cpp` | INTEGRATE BOTH | Our per-file kvikio loop + upstream HOST D2H branch inside loop |
| `test_host_table_utils.cpp` | UPSTREAM | D-01: formatting only |

Full per-file rationale in `24-CONFLICT-LOG.md` Parts 1 + 2.

**D-04 Commit D (post-merge fix-up):** Missing `stream_view` argument in
`gpu_table_representation(tbl, space)` → `(tbl, space, stream_view)` at `sirius_extension.cpp:896`.
Committed separately (`90fad83`) per D-04 atomic discipline (fix-up is bisectable; not amending
the merge commit).

---

## D-01 Application Summary (upstream-as-source-of-truth META-RULE)

D-01 was applied to every conflict in Phase 24:

- **Cucascade rebase:** Commit 3 (8392c3d RE-DERIVE) — took upstream's `*fast_table->allocation`
  shared_ptr dereference (upstream is source of truth for the new API); preserved our `target_stream`
  (unique behavior upstream doesn't have). This is the canonical D-01 pattern: upstream wins on API
  shape; ours wins on unique multi-GPU correctness additions.

- **Sirius merge:** All 9 conflict files resolved upstream-favored per D-01. Where both upstream
  and ours add unique behavior (PIN-MGPU-01 round-robin + host-tier path), "integrate-both" was the
  D-01-consistent decision — upstream's new feature is not discarded, and our unique behavior
  (`chunk_memory_spaces` routing) is not dropped.

- **D-01 shift vs Phase 23:** Phase 24 explicitly biased toward upstream (per user feedback). The
  result: 1 RE-DERIVE vs Phase 23's more complex symmetric triage. The fork did not shrink (D-02
  step 4 found no fully-obsoleted commits), but the RE-DERIVE was clean.

---

## D-05 Gitlink Ours-Wins Evidence

During the sirius `git merge --no-ff origin/dev` step (Plan 24-03), `2e197c6` proposed advancing
the cucascade gitlink to `96bfea1` (upstream's pure cucascade commit). Our fork at `5203de5` is a
descendant of `96bfea1` (it contains `96bfea1` plus 8 more commits).

Resolution: gitlink kept at `5203de5` per D-05. Git recorded this as an automatic fast-forward
during conflict resolution (`LINKPOST=5203de5a028ccb57402a4105e35282c567c3ee5a`).

Verification: `git submodule status cucascade` shows `5203de5` — no leading `+` sign.
`git show --name-only d228504 | grep -v '^$' | head -5` shows only `cucascade` in D-04 Commit A
diff (atomic-commit discipline confirmed).

---

## D-06 No git push attestation

No `git push origin` was executed on either cucascade or sirius during Phase 24.

All cucascade commits (`4b94571`, `3c44dae`, `d5ac57b`, `c15cb01`, `e10bd4a`, `b21bd97`, `4319726`,
`1522e0b`, `5203de5`) reside on the local `fix/pinned-portable-flags` branch only. They have NOT
been pushed to `origin/main` on the upstream cucascade repository.

All sirius commits (`7ede83c`, `6d5758f`, `ff06fac`, `d228504`, `d0e792d`, `8b2a774`, `ff04f31`,
`90fad83`, `c9aa166`, `1189c82`, `7a23f63`, `a1909c8`) reside on the local
`feature/single-node-multi-gpu2` branch only. They have NOT been pushed to `origin/dev` or any
other remote.

The user handles upstream PR submission per CC-UPSTREAM-01 policy.

---

## D-07 Commit E Disposition — Branch A (upstream test exists)

**Branch A taken.** Commit `2e197c6` added `[pin_table_host]` Catch2 tag at
`test/cpp/integration/test_gpu_execution_tpch.cpp:4556`. No new test file was needed for D-04
Commit E — the upstream test is the durable, bisectable artifact.

Phase 24 outcome: 1/1 PASS, 51 assertions, 6.6s, exit 0. The upstream test exercises the full
host-tier pin_table path: `pin_table('lineitem', tier='host')` DDL, followed by a TPC-H aggregate
query reading from the host-cached table via the new `cached_split_provider` HOST constructor and
`insert_pinned_entry_host()` path in `sirius_scan_manager`.

This is the canonical D-01 outcome: upstream provides the test surface; we verify our merge
integration doesn't break it.

---

## Section J (extended): sanitizer_gate_22.sh re-verify

The Phase 23 Plan 23-07 windowed-awk script (`test/scripts/sanitizer_gate_22.sh`) re-verified in
Phase 24 in three ways:

1. **P22_SELFTEST=1:** Exit 0 (synthetic log with 1 race section + 1 API-error section both
   containing `alloc_and_peer_copy_async` → cluster_B=1, total_races=1 for the race section only).
   Script correctly distinguishes race headers from API-error headers.

2. **Full gauntlet run:** cluster_B=0, cluster_A=0, total_races=0. No stream-ordered race findings.

3. **ba5ed27 impact check:** The `wire_data_repositories` split (ba5ed27) refactors
   `materialize_repository_wiring()` — this is a call-site reorganization, not a new CUDA kernel
   or stream-ordering point. The script needed no extension; no new symbols match the API-error
   filter's patterns.

---

## HYG-02 + kvikio-free Invariants

Both invariants hold post-Phase-24-close:

- **HYG-02:** `grep -rn 'rmm::cuda_stream_default' src/ | wc -l` = 40 (all in `src/legacy/`).
  The `ba5ed27` and `2e197c6` merges added no new `cuda_stream_default` usages.

- **kvikio-free:** `grep -rn 'cudf::io::datasource::create\|cudf::io::source_info{' src/ | grep -v
  'data_source.get()\|datasource.get()' | grep -v '^[^:]*:.*//' | wc -l` = 0. The `2e197c6`
  host-tier parquet reading uses `sirius_ioctx::make_datasource` (the Phase 22.1 standard bypass
  pattern).

These invariants have been preserved continuously from Phase 22.1 (kvikio removal) through Phase
22.2, 22.3, 23, and now Phase 24. They are not at risk from the upstream commits integrated here.

---

## chunk_memory_spaces Drift Disposition

**Count:** 60 (Phase 23) → 42 (Phase 24). Accepted as non-regression.

The "integrate-both" conflict resolution strategy in Plan 24-03 added the host-tier code path
from `2e197c6`. The host-tier path uses `host_chunks`, `tier`, and `memory_space` fields for its
bookkeeping — NOT `chunk_memory_spaces`. The GPU-tier round-robin path (`PIN-MGPU-01`) continues
to use `chunk_memory_spaces`. Both paths coexist in `cached_split_provider.hpp/cpp` and
`sirius_scan_manager.hpp/cpp`.

Functional coexistence verified:
- `[pin_mgpu]` 2/2 PASS (GPU-tier round-robin still works)
- `[mgpu-audit]` 6/6 PASS (multi-GPU scan routing still works)
- `[pin_table_host]` 1/1 PASS (host-tier new path works)

The drop was accepted at the Plan 24-04 human-verify checkpoint (2026-05-13).

---

## Side-Benefits

1. **REG-06 Leg 1 memcheck: 6/7 PARTIAL → 7/7 PASS.** Phase 23 reported `Invalid __global__ read`
   violations in `libcudf.so::cudf::detail::contiguous_split` from the checksum computation path.
   Phase 24 run shows those violations absent — all 7 sanitizer errors are pre-existing
   `cudaErrorPeerAccessAlreadyEnabled` API-error backtraces. The improvement is observed but not
   caused by Phase 24 changes; likely a cudf workload path or build configuration difference.

2. **New `[pin_table_host]` gate (D-07) passes 1/1.** Upstream commit `2e197c6` introduced
   host-tier caching for `pin_table`; the corresponding `[pin_table_host]` Catch2 tag is present
   in the post-merge binary and passes (51 assertions, 6.6s). Branch A taken: no Commit E needed.

3. **ba5ed27 merge confirmed clean.** The repository_wiring Phase 2 refactor (`materialize_repository_wiring()`) auto-merged cleanly in `sirius_engine.cpp` because `drain_after_error` is in `execute()`, not `initialize_internal()`. The D-08 MEDIUM-risk prediction for `sirius_engine.cpp` resolved without conflict.

---

## Carry-Forwards to Phase 25+

### Active carry-forwards

**CC-UPSTREAM-01: Cucascade fork 9 commits ahead of 9ceebaa (no upstream PRs)**

The fork carries 9 commits ahead of `9ceebaa` (was 8 at Phase 23 close; +1 new test-fix commit
`5203de5` for the `96bfea1` writer_stream API mismatch). Per CC-UPSTREAM-01 policy, upstream PR
submission is deferred — the user handles all upstream PR submission separately.

Upstream PR candidates when reviewed (see `24-CUCASCADE-DIFF.md` for per-commit notes):
- Commits 6+7+8 bundle (same-stream invariant + dst_guard + probe-restore) are the most
  upstream-ready as a logical unit for `alloc_and_peer_copy_async` correctness.
- Commit 2 (io_worker member ordering) is self-contained.
- Commits 1+3+4 require hardware-matrix validation and upstream API-design discussion.
- Commit 5 (pre-commit formatting) would accompany functional commits.
- Commit 9 (test fix for 96bfea1 slice-roundtrip API mismatch) should accompany the PR that
  introduces the 3-arg `gpu_table_representation` constructor (commit 4).

**cudf library memcheck baseline from libcudf.so**

The `Invalid __global__ read` violations in libcudf.so (checksum path) that appeared in Phase 23
Leg 1 memcheck were absent in Phase 24. Status: NOT a Phase 24 regression; the violations are a
cudf library internal issue, not sirius/cucascade code. Monitor in Phase 25 — if they reappear,
classify as cudf baseline (same as Phase 23 characterization).

### CLOSED in Phase 24

The following Phase 23 active carry-forwards are **CLOSED by Phase 24's merge** (absorbed via
the `ba5ed27` + `2e197c6` upstream merge):

- **PIN-MGPU-03** (HOST-tier `pin_table` path): The `2e197c6` upstream commit added `tier='host'`
  support. This is NOT the same as the v1.6+ "NUMA-local round-robin" design, but the upstream
  host-tier path is now integrated and functional. The REQUIREMENTS.md PIN-MGPU-03 deferred item
  may be retired or updated per Phase 25 planning.

- **CUDA event wrapper migration** (cucascade PR #121 `cuda_event` type): This was a Phase 23
  carry-forward; no action needed in Phase 24 (the `96bfea1`/`9ceebaa` commits build on the
  existing API without requiring migration by consumers).

### Deferred items (from 24-CONTEXT.md `deferred` block)

- Upstreaming our cucascade fixes — user handles separately; CC-UPSTREAM-01 carry policy preserved.
- Integrating PIN-MGPU-01 round-robin with `2e197c6` host-tier pinning into a unified "tier + GPU
  index" distribution — Phase 25+ candidate.
- Refactoring `sirius_ioctx::make_datasource` to adopt `ba5ed27`'s descriptor split pattern beyond
  what mechanical merge required — Phase 25+ candidate.
- Adding full upstream-feature coverage (slice-host-table behavior, empty-STRING-column edge cases)
  to our gauntlet — trust upstream's tests; revisit if our usage breaks.
