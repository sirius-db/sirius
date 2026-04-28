# Project Retrospective

*A living document updated after each milestone. Lessons feed forward into future planning.*

## Milestone: v1.2 — Multi-GPU SQL Pipeline Fix

**Shipped:** 2026-04-28
**Phases:** 3 (Phase 8, 9, 10) | **Plans:** 18 | **Tasks:** 39

### What Was Built
- Per-GPU stream pool replacing the GPU-0-bound singleton in `duckdb_scan_executor` (FIX-01)
- Pattern 2 idiom (target-bound stream + target-device RAII) applied at 4 cross-device call sites (FIX-02)
- Per-GPU filter translation at plan time — one `translated_expression` per configured GPU, scalars allocated on the correct device resource (Phase 8 residual closure 93fea6f)
- `_batch_gpu_affinity` map in `duckdb_scan_executor` recording batch→GPU ownership atomically with `[mgpu-audit]` log emission, reset per query
- `preferred_device_id` plumbing in `parquet_scan_task_local_state` with two-tier local-wins-over-global lookup mirroring `gpu_pipeline_task.hpp`
- `std::set_intersection` cross-GPU disjointedness REQUIRE in AUDIT TEST_CASE (substantive regression gate)
- `translated_expression::owned_stream` declared before `owned_literals` so reverse-destruction order keeps the stream alive past `cudf::scalar` `cudaFreeAsync` calls (Phase 10-03 stream use-after-destroy fix, 36 LOC)
- `integration-2gpu.yaml` fixture + `RUN_TPCH_MGPU(GENERATE(1, 2))` Catch2 macro across 45 TEST_CASE call sites
- SF100 TPC-H Q1 num_gpus=2 ship-gate: 5.70s wall-clock, byte-identical to 1-GPU baseline, 71 scan batches GPU0=42 / GPU1=29 with cross-GPU intersection=0

### What Worked
- **Probe-then-fix discipline.** Phase 8-07 added grep-stable `[mgpu-probe]` breadcrumbs at frame boundaries; Phase 8-08 ran a clean MCP reproduction whose payload discriminated hypotheses A/B/C/D *and* surfaced the unforeseen pattern E (cross-GPU batch double-dispatch). Without the breadcrumbs, the Phase 9 scope would have been guesswork.
- **HALT-and-rescope when evidence breaks the plan.** Plan 08-09 was pre-authored against H-A/B/C/D; when 08-08 rejected all four, halting + scoping Phase 9 instead of forcing a fix saved tens of LOC of throwaway work. The `08-09-HALT.md` document was the right primitive.
- **Goal-backward verdict honesty.** Phase 9's verifier marked PARTIAL despite SF100 ship-gate PASS — distributor goal achieved, but ship-gate criterion 2 (TABLE_FUNCTION-form unit tests) still failing. That honesty is what scoped Phase 10 cleanly with a deterministic 4-plan structure (bisect → gdb → fix → validate).
- **Bisect that returned NONE was a real signal.** Plan 10-01's `regressing_commit: NONE` finding wasn't a wasted plan — it told us the SIGSEGV was test-ordering dependent, which immediately ruled out "blame a single commit" theories and reshaped 10-02's GDB approach.
- **MCP-only host capability discovery (2026-04-24).** Confirming 2× RTX 6000 Ada visible to MCP via `nvidia-smi` collapsed the original "user-delegated SF100 run" checkpoint into an autonomous MCP run, removing one human handoff per phase.

### What Was Inefficient
- **Initial Phase 9 plan over-scoped the user-delegated ship-gate.** Plan 09-04 was authored as a `checkpoint:human-action` requiring the user to run SF100 manually — this was based on a stale memory ("MCP has no GPUs"). The autonomous re-run via MCP later in the session would have shipped Phase 9's verdict the same day if the plan had assumed MCP autonomy upfront.
- **`mcp unit-tests filter='gpu_execution - TPC-H Query*'` produced a fixture-init-order SIGSEGV** that looked like a regression but was a test-ordering artifact — wasted ~15 minutes investigating before realizing the full-suite run (which initialized fixtures in catalog order) was the canonical test mode.
- **Two memory entries needed correcting mid-session.** "MCP for unit tests, ask user for integration" (stale assumption) and the AUDIT TEST_CASE "wait for human runbook" pattern both burned cycles before being updated to reflect actual MCP capability on this host.
- **Phase 9-02 planner explicitly flagged "minimum viable" affinity-map recording** without dispatch-time consultation. That was correct scoping but it left the hive-partition path failing with a separate cross-GPU lookup pattern that's now Phase 11-candidate territory. A clearer dispatch-time-affinity plan in Phase 9 might have closed both paths.

### Patterns Established
- **Probe-first when a fix doesn't reproduce.** Add `[mgpu-probe]` (or equivalent grep-stable INFO) breadcrumbs at frame boundaries before authoring a fix when the failure mode is intermittent or has multiple plausible call chains.
- **HALT plans authored against rejected hypotheses.** When evidence rejects the plan's assumed cause, write a `XX-YY-HALT.md` superseded-by record and rescope rather than forcing the original plan through.
- **3-phase debug pattern: bisect → gdb → fix.** Phase 10's structure (10-01 bisect, 10-02 GDB, 10-03 targeted fix, 10-04 validate) is reusable for any "regression of unknown origin" scenario.
- **C++ destruction-order discipline for CUDA stream-allocated objects.** When a `cudf::scalar` (or any RMM-allocated object) holds a stream handle, the stream object must outlive it. Declare the stream BEFORE the allocations in the same struct/class to leverage reverse-destruction order. This is now a documented Sirius pattern via Phase 10's `translated_expression::owned_stream`.
- **Disjointedness REQUIRE as canonical multi-GPU regression gate.** `std::set_intersection(counts[0].scan_ids, counts[1].scan_ids).empty()` is a one-line assertion that catches every cross-GPU double-dispatch shape going forward — keep it in any future multi-GPU AUDIT TEST_CASE.

### Key Lessons
1. **Goal-backward honesty unblocks faster than "passing" verdicts.** Phase 9 marking PARTIAL (not PASS) is what made Phase 10's deterministic 4-plan structure possible. A "PASS with caveats" verdict would have hidden the work needed.
2. **Memory entries about host capability go stale fast.** Re-confirm `mcp__project-commands__run_command nvidia-smi` once per session before making "delegate to user" decisions about GPU-dependent tests.
3. **When a bisect returns NONE, the bug is environmental (test order, build state, env var, GPU contention) — not commit-specific.** That finding alone reshapes the next plan.
4. **Pre-existing failures should be confirmed via `git stash; run; git stash pop` before being attributed to your fix.** Phase 10-03 used this pattern to definitively show `[mgpu-audit]` SIGSEGV was orthogonal — saved a cascade of follow-up "did I break this?" cycles.
5. **Synthetic IDs (CRIT-1/2/4/6) for ROADMAP success criteria are a useful intermediate.** They let phase plans address ship-gate criteria as first-class requirements without polluting REQUIREMENTS.md with non-feature IDs. Document the convention in PROJECT.md if it persists into v1.3.

### Cost Observations
- Sessions: ~5 (paused/resumed 2026-04-24 over 3 days while Phase 10 plans were authored off-session)
- Notable: Phase 10's executor agents averaged ~50 minutes per plan; the longest (10-04 ship-gate validation) took 115 minutes due to SF100 dataset I/O + 1-GPU baseline diff + audit log capture
- Token efficiency win: spawning gsd-executor agents in serialized sequence within Wave 1 (rather than parallel) eliminated merge-conflict risk on `duckdb_scan_executor.cpp` at the cost of ~30 minutes of wall-clock — clear win

---

## Cross-Milestone Trends

### Process Evolution

| Milestone | Sessions | Phases | Plans | Key Change |
|-----------|----------|--------|-------|------------|
| v1.1 | ~6 | 4 | 19 | Established push-model task dispatch + cucascade I/O migration; introduced `[mgpu-audit]` logging |
| v1.2 | ~5 | 3 | 18 | Established probe-first + HALT discipline; introduced bisect→gdb→fix pattern for unknown-origin regressions; MCP autonomy for full ship-gate runs |

### Cumulative Quality

| Milestone | Tests | Hardware Validation | HYG-02 (`rmm::cuda_stream_default`) |
|-----------|-------|---------------------|-------------------------------------|
| v1.1 | 979/979 | N=2 hardware (2× RTX 6000 Ada) | 41 (baseline) |
| v1.2 | 665/666 (1 pre-existing SIGSEGV pre-dating v1.2) | N=2 hardware + SF100 | 40 (improved by 1) |

### Top Lessons (Verified Across Milestones)

1. **Probe-first beats fix-first when failure modes are unclear.** v1.1 used `[mgpu-audit]` for end-to-end traceability; v1.2 added `[mgpu-probe]` for intermediate frame boundaries. Both unblocked otherwise-stuck investigations.
2. **PARTIAL verdicts ship safer than forced-PASS.** v1.1's "approved with deferrals" sign-off pattern and v1.2's PARTIAL verdict on Phase 9/10 both produced shippable milestones with documented residuals — the residuals became scoped follow-up phases rather than hidden tech debt.
3. **Static invariants caught regressions automatically.** HYG-02 grep gate (`rmm::cuda_stream_default` count) prevented 2+ separate stream-default re-introductions across v1.1 and v1.2. Cheap, deterministic, high-leverage.
