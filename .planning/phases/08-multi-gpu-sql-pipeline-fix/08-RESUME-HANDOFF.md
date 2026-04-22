---
phase: 08-multi-gpu-sql-pipeline-fix
type: resume-handoff
recorded: 2026-04-22T22:00:00Z
branch: feature/single-node-multi-gpu2
base_commit: 6f64840
purpose: Resume Phase 8 follow-ups #1 and #5 after /clear
---

# Resume handoff — Phase 8 follow-ups #1 and #5

The core Phase 8 MGPU fix landed in commits `93fea6f` (sirius) +
`ecb96c1` (cucascade bump). Diagnosis in
`.planning/phases/08-multi-gpu-sql-pipeline-fix/08-11-DIAGNOSIS.md`.
Follow-ups catalogued in
`.planning/phases/08-multi-gpu-sql-pipeline-fix/08-FOLLOWUPS.md`.

Two follow-ups are tagged for the next session.

---

## Follow-up #1 — Iceberg metadata-scan has same eager-translate pattern

**What:** `src/op/scan/sirius_parquet_metadata_scan_operator.cpp:214`
builds a `gpu_expression_translator` with
`cudf::get_current_device_resource_ref()` at task time, stores the
result on `partitioned_parquet_metadata::reader_options` via
`set_filter(ast_filter->back())`, and downstream
`sirius_gpu_parquet_scan_operator::execute()` reads those opts and
calls `cudf::io::read_parquet(opts, stream)` potentially on another
GPU. Same hazard as the one 93fea6f fixed in the parquet-scan path.

**Shape of fix (mirror 93fea6f):**

1. In `sirius_parquet_metadata_scan_operator`, replace the single
   `std::shared_ptr<translated_expression>` with
   `std::shared_ptr<std::unordered_map<int, translated_expression>>`.
2. Loop over configured GPU device ids (get from SiriusContext like
   sirius_engine did), wrap each translation in
   `rmm::cuda_set_device_raii` + `get_per_device_resource_ref(id)`.
3. Change `partitioned_parquet_metadata` to carry the per-device map
   instead of a single tree + `reader_options` with filter pre-set.
   Consumers (`sirius_gpu_parquet_scan_operator::execute()`) pick the
   entry matching the current device and call `set_filter` on a local
   opts copy.
4. `parquet_scan_data::filter_expression` currently holds either
   `shared_ptr<translated_expression>` or `shared_ptr<duckdb::Expression>`
   (std::variant) — if it holds a pre-translated one, switch to the
   per-device map shape.

**Files to edit (estimated):**
- `src/include/op/scan/sirius_parquet_metadata_scan_operator.hpp`
- `src/op/scan/sirius_parquet_metadata_scan_operator.cpp`
- `src/op/scan/sirius_gpu_parquet_scan_operator.cpp`
- Possibly `src/include/op/scan/parquet_scan_data.hpp` (if filter_expression variant needs updating)
- Need to find how the metadata scan op gets the configured device IDs — it likely needs a new param or access to SiriusContext similar to `sirius_engine::construct_sirius_specific_operator`.

**Testing:** The iceberg path is exercised by
`test/cpp/integration/test_gpu_execution_iceberg*.cpp` tests. No
current failing test captures the num_gpus>1 iceberg path, so you'd
need to either flip `integration.yaml`'s num_gpus to 2 for an iceberg
test OR add a test that explicitly runs iceberg_scan with num_gpus=2.

**Est:** ~60 LOC across 3-4 files + a test. Maybe 1 hour.

---

## Follow-up #5 — mgpu-audit distribution assertion fails

**Background:** `test/cpp/integration/test_gpu_execution_tpch_mgpu_audit.cpp:138`
is the audit TEST_CASE that Phase 8 authored but never ran green
(blocked by the residual crash 93fea6f fixed). Now it runs and fails
in two stages:

### Stage 1 (pool-prime OOM) — workaround found, NOT applied

Lowering `usage_limit_fraction` from 0.5 to 0.4 in
`test/cpp/integration/integration-2gpu.yaml` bypasses the OOM. Not
applied in this session because exposing stage 2 below is more
important than hiding it.

### Stage 2 (distribution assertion) — real work to do

After the stage-1 workaround, test fails at line 243 with:
```
  REQUIRE(counts[0].pipeline_ids.size() >= min_count)
  0 >= 1
  per-GPU audit counts: GPU0{pipeline=0, scan=4} GPU1{pipeline=3, scan=0}
```

All scans on GPU 0, all pipeline tasks on GPU 1 — no co-distribution.
The audit invariant is that both kinds run on both GPUs.

### Investigation steps (fresh session)

1. Apply the stage-1 workaround (or fix OOM properly by auditing the
   `cuda_async_memory_resource` pool-prime path):
   ```
   sed -i 's/usage_limit_fraction: 0.5/usage_limit_fraction: 0.4/' \
     test/cpp/integration/integration-2gpu.yaml
   ```
2. Build: `mcp__project-commands__run_command build`
3. Run the audit test directly (GPU required, use
   `dangerouslyDisableSandbox: true`):
   ```
   build/release/extension/sirius/test/cpp/sirius_unittest '*per-GPU distribution*'
   ```
4. Examine Sirius log from the run (`ls -td /tmp/sirius-mgpu-audit-* | head -1`):
   - `[mgpu-audit] scan_batch assigned to GPU N batch_id=M` lines —
     do they actually hit both GPUs?
   - `[mgpu-audit] pipeline_task dispatched to GPU N task_id=M` lines
     — same.
5. Questions to classify:
   - Does the scan_executor's device selection
     (`duckdb_scan_executor::select_target_gpu`,
     `src/op/scan/duckdb_scan_executor.cpp:180`-ish) use weighted
     round-robin? Does it return the same GPU on consecutive calls
     for a 6-batch workload?
   - Does the pipeline_executor round-robin across GPUs? Or does it
     prefer the GPU holding the scan output (which would explain
     pipeline-only on one GPU if scans all landed on the other)?
6. Classify:
   - **A. Dispatch is broken:** fix the scan or pipeline dispatch
     policy so a 6-batch SF1 workload splits across GPUs. The audit
     test asserts the correct invariant.
   - **B. Audit is over-specified:** small SF1 workloads can't
     guarantee both kinds on both GPUs; relax the threshold or gate
     the assertion on `SIRIUS_TEST_SF10_PATH` (see line 235 —
     already gates the >=5 version, maybe the >=1 version should too).

**Recovery:** if stage-1 workaround is needed for tests to run at
all on this host, apply it as a committed change with a comment
pointing to the stage-2 investigation.

**Est:** 1-2 hours.

---

## How to resume

After `/clear`, the first prompt should be:

```
Resume Phase 8 follow-ups from
.planning/phases/08-multi-gpu-sql-pipeline-fix/08-RESUME-HANDOFF.md.
Start with follow-up #<1 or 5> as described there.
```

That loads the plumbing doc and the model picks up with the exact
investigation steps and file paths. Current branch is
`feature/single-node-multi-gpu2` at commit `6f64840`. Cucascade is on
its own branch `fix/pinned-portable-flags` at `abdeaf9`.

The original Phase 8 fix (`93fea6f`) is validated and committed —
these follow-ups are scope extensions, not repairs.

---

*Phase: 08-multi-gpu-sql-pipeline-fix*
*Handoff authored: 2026-04-22T22:00:00Z*
*Parent commit: 6f64840*
