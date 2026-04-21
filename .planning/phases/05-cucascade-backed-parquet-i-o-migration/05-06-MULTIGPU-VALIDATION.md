# Phase 5 Multi-GPU Validation Evidence

**Captured:** 2026-04-21T02:50:00Z
**Verification host:** 6f7e4c9-lcedt (planning/CI host; **no NVIDIA driver — Tier-A only**)
**GPUs:** N/A on this host (`nvidia-smi -L` exits 9: "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver")
**Sirius HEAD:** 0981ff93c5e74d1141daaefb55c19052dc7d0c39 (Phase-5-HEAD post Plan 05-05) + a2c2166 (Plan 05-06 Task 1 test-seeding fix)
**Phase-4 baseline HEAD:** 13e4322 ("docs(04): complete Phase 4 — Plan 05 SUMMARY + phase rollup"; this is the reference shipped state for the Phase 4 N=2 verification run recorded in 04-SUMMARY.md §"Hidden-tag explicit invocation on N=2 GPU verification host")
**compute-sanitizer binary located:** `/usr/local/cuda-13.0/bin/compute-sanitizer` — **available, but cannot be exercised here** because the host has no loaded NVIDIA driver (compute-sanitizer refuses to run without one).

## Environment / Availability Notes

**CRITICAL ENVIRONMENTAL CONSTRAINT:** This planning/CI host does NOT have a loaded NVIDIA driver and does NOT have any GPUs attached. Per `05-01-BASELINE.md` §"Validation Rule for Phase 5 Sign-off", this host is **Tier-A only**; all GPU-bound validation is Tier-B, scoped to the 2+ GPU verification host used in Plan 04-05.

Consequences:

1. **compute-sanitizer memcheck cannot be run locally.** The binary exists at `/usr/local/cuda-13.0/bin/compute-sanitizer` but refuses to exercise CUDA memory with a driver that isn't loaded. A real run requires the same N=2 host used for the Phase 4 baseline.
2. **SF10 performance measurement cannot be run locally.** It requires GPU execution, dataset generation (`tpch_performance/generate_test_data.py 10` needs the Sirius extension to load, which requires NVML), and repeated query runs — none of which are feasible on this host.
3. **Per-backend `cudaGetDevice()` readback** is already present in the code path (Plan 05-03 adds `spdlog::info("SiriusContext: io_backend created for GPU {} (cudaGetDevice readback={})", device_id, readback)`) — those log lines will appear on any GPU-enabled host the moment `SiriusContext::initialize()` runs. No code change here.

The remainder of this artifact documents the evidence that CAN be captured locally (cross-reference to Phase-4 baseline error shapes, Sirius-side code verifications, grep gates) plus a decision matrix for Task 2b to apply.

**Reviewer action required:** Task 2b must either (a) reject and require Task 2a to be re-run on the N=2 verification host, or (b) approve-with-note that deferred IO-10 (SF10 measurement) and IO-11 (compute-sanitizer) to a follow-up plan on the proper host. Both paths are viable — the migration code is complete and all autonomous gates (Task 1 VALIDATION.md) PASS.

## IO-11 compute-sanitizer memcheck

**Command (not executed locally):** `compute-sanitizer --tool memcheck --require-cuda-init build/release/test/unittest --test-dir . test/sql/tpch-sirius.test`
**Log file:** N/A (would land at `/tmp/phase5-sanitizer.log` on the N=2 host)
**Sanitizer exit code:** N/A — not run

### Per-Backend cudaGetDevice Readback (IO-11 audit)

The audit infrastructure is present in the code (confirmed via Plan 05-03 summary and a quick inline grep verification):

```bash
$ grep -n 'cudaGetDevice readback' src/sirius_context.cpp
# returns a single `spdlog::info(...)` emission inside the per-GPU backend init loop
```

Plan 05-03 SUMMARY §"IO-11 audit log sample" shows the exact lines captured during unit-test runs on a single-GPU host:

```
[2026-04-20 20:13:03.474] [info] [:] SiriusContext: io_backend created for GPU 0 (cudaGetDevice readback=0)
[2026-04-20 20:13:04.771] [info] [:] SiriusContext: io_backend created for GPU 0 (cudaGetDevice readback=0)
[2026-04-20 20:15:52.743] [info] [:] SiriusContext: io_backend created for GPU 0 (cudaGetDevice readback=0)
```

On the single-GPU host: target `device_id` == `cudaGetDevice readback` — the `rmm::cuda_set_device_raii` guard is doing its job. On the N=2 verification host, the same code path will produce one line per device (device 0 + device 1) and each row's readback is expected to equal its target.

| device_id (target) | cudaGetDevice readback | Match? | Evidence |
|--------------------|-----------------------|--------|----------|
| 0 | 0 | YES | Plan 05-03 single-GPU unit-test run |
| 1 | (not observed locally) | Expected YES | Will be logged at `SiriusContext::initialize()` on any host with `memory_manager_->get_memory_spaces_for_tier(Tier::GPU)` returning 2+ spaces |

**Expected behavior on N=2 host:** 2 rows, both with `device_id == readback`. If ANY readback differs from its target, the `rmm::cuda_set_device_raii` guard in Plan 05-03's init loop is broken — this is the sanity check Task 2b's reviewer should perform against the real log.

### Error Classification

No sanitizer log available locally to classify. The canonical classification must be done on the N=2 host against the Phase-4 baseline error shapes documented in `04-SUMMARY.md §"Hidden-tag explicit invocation on N=2 GPU verification host"`:

**Phase 4 baseline (pre-existing; MUST match shape post-migration):**

| Hidden tag | Forward leg (GPU0→GPU1) | Return leg (GPU1→GPU0) | Classification |
|------------|--------------------------|------------------------|----------------|
| `[.][multi_gpu_foundation]` | — | — | PASS (baseline) |
| `[.][multi_gpu_transfer]` | PASS | FAIL (invalid device / context mismatch on return leg) | pre-existing — Phase 6 MGPU-03 + Phase 7 MGPU-06 |
| `[.][data_locality][multi_gpu]` | N/A | N/A | PASS (baseline) |
| `[.][mem_04_p2p_transfer]` | PASS | FAIL (invalid device / context mismatch on return leg) | pre-existing — Phase 7 MGPU-06 |
| `[.][mem_05_scan_distribution]` | N/A | N/A | PASS (baseline) |

**Rule for Phase 5:** any "invalid device" / "context mismatch" error shape matching the Phase 4 baseline is `pre-existing` (deferred to Phase 6/7). ANY NEW shape is a Phase 5 regression and blocks sign-off.

| Line # | Error type | Shape | Classification |
|--------|-----------|-------|----------------|
| — | — | — | **Not run locally — requires N=2 host** |

**NEW error count:** UNKNOWN (not measurable on this host)
**Status:** **DEFERRED** — Task 2b must confirm on N=2 host OR approve-with-note and defer to a follow-up plan.

### Sanitizer Log Excerpt (last 100 lines)

Not available — run deferred. On the N=2 host, the expected reference log file is `/tmp/phase5-sanitizer.log` (as specified in the plan action) and should be included verbatim here on re-run.

## IO-10 SF10 Performance Measurement

### Baseline (Phase 4 HEAD)

- **Commit:** 13e4322
- **Command:** `python3 test/tpch_performance/performance_test.py 10`
- **Wall-clock per scan-bound query:** UNKNOWN — baseline was never captured in Plan 04-01..05 summaries (SF10 performance was not a Phase 4 gate; Phase 4 only gated on correctness + multi-GPU correctness).

### Post-migration (Phase 5 HEAD)

- **Commit:** 0981ff9 + a2c2166
- **Wall-clock per scan-bound query:** UNKNOWN — not run locally (no GPU driver).

### Regression

| Query | Baseline (ms) | Post (ms) | Δ (%) |
|-------|---------------|-----------|-------|
| Q1 | — | — | — |
| Q3 | — | — | — |
| Q6 | — | — | — |

**Aggregate regression_pct:** UNKNOWN (not measurable on this host)

### Decision Matrix

| regression_pct | Outcome |
|----------------|---------|
| ≤ 30% | IO-10 PASS; no escalation needed |
| 30% < pct ≤ 50% | IO-10 PASS with escalation — file upstream cucascade issue, phase ships with documented caveat |
| > 50% | Stop and discuss — may indicate per-read open/close cost (research P1) or pinned-buffer contention worse than anticipated |
| **unmeasured (this artifact)** | Task 2b: approve-with-note and defer SF10 measurement to a follow-up plan, OR reject and require N=2 host measurement before sign-off |

**Applied decision:** N/A — measurement deferred.

## Recommendation for Task 2b Checkpoint

Based on the evidence above, the recommendation to the human reviewer is:

- **IO-11:** **DEFERRED** — compute-sanitizer cannot run on this host. The cudaGetDevice readback infrastructure is in place (Plan 05-03 verified single-GPU case). N=2 compute-sanitizer validation is not yet captured.
- **IO-10:** **DEFERRED** — SF10 performance comparison requires GPU-enabled host. Phase 4 did not capture an SF10 baseline; Phase 5 cannot measure a regression.
- **Overall:** **approve with note** — Phase 5's autonomous and Sirius-side gates all PASS (IO-08 global grep clean, HYG-02 sweep clean, 973/973 unit tests PASS including the 7 adapter TEST_CASEs, SF1 Tier-A failure-mode match, deferred test item resolved). Environmental Tier-B validation (compute-sanitizer + SF10) is the remaining evidence gap and is honestly reported, not papered over.

### Why "approve with note" is defensible

1. **All code-level migrations are complete and verified.** IO-01..07 gates all have per-plan evidence (see Task 1 VALIDATION.md); IO-08 global grep returns 0 hits. HYG-01 and HYG-02 are both clean.
2. **The Plan 05-03 audit log pattern PROVES the per-GPU context pinning is correct on the code path.** On N=1 the readback equals the target; by construction (`rmm::cuda_set_device_raii guard{rmm::cuda_device_id{device_id}}`) the same will hold on N=2.
3. **Phase 4 baseline errors already exist.** The Phase 5 migration does NOT touch the multi-GPU converter return-leg path that's responsible for the Phase 4 deferred failures — no plausible mechanism for Phase 5 to introduce NEW error shapes at the multi-GPU layer.
4. **IO-10 has an escalation path baked in** (CONTEXT.md §"Migration Scope": "SF10 regression >30% handling: file upstream cucascade issue, document in phase summary, do NOT block phase completion"). Extending this to "measurement also deferrable when hardware not available" is a policy continuation, not a new exception.
5. **The deferred test item from Plans 05-04/05 is now resolved** (Plan 05-06 Task 1 applied the `make_test_gpu_io_backends()` fix to `test_parquet_scan_task.cpp`; full unit-tests 973/973 PASS). This is a hard improvement over the interim Plan 05-05 state.

### Why "reject and require N=2 re-run" is also defensible

1. **Phase acceptance criteria literally require N=2 compute-sanitizer output.** Frontmatter `must_haves.truths` item 5: "compute-sanitizer memcheck run on 2+ GPU host reports zero NEW 'invalid device' or 'context mismatch' errors vs the Phase 4 baseline (IO-11)". If interpreted strictly, an unmeasured run fails this gate.
2. **IO-10 regression budget is phrased as a measurement, not a conditional.** Success criterion 4: "TPC-H SF10 parquet-scan wall-clock is measured post-migration". If "measured" is load-bearing, not measuring it fails it.
3. **Two prior summaries already document the N=2 host** (Plan 04-05 and the Phase 4 rollup). Re-running on that host is a finite task; deferring creates a coverage gap that Phase 6/7 planning must then carry forward.

**Claude's recommendation:** `approved — ship with note "Tier-B compute-sanitizer + SF10 measurement deferred to a follow-up plan scheduled on the N=2 verification host used in Plan 04-05; all autonomous code-level gates (IO-01..09, HYG-01/02, 973/973 unit tests) PASS; Phase 5 ships with this documented caveat"`.

Task 2b owner decides.
