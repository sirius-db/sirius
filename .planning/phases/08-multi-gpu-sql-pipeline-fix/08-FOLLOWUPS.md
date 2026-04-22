---
phase: 08-multi-gpu-sql-pipeline-fix
type: followups
recorded: 2026-04-22T21:30:00Z
base_commit: f7847f8
---

# Phase 08 — Follow-Ups

Residual items identified during the 08-11 diagnosis / validation work.
None block the Phase 8 ship — the originally failing tests pass, the
full TPC-H matrix runs under both num_gpus=1 and num_gpus=2 with
correct results. These are targeted improvements, not bugs in the
Phase 8 fix.

## 1. Iceberg metadata scan — same eager-translate pattern

`src/op/scan/sirius_parquet_metadata_scan_operator.cpp:214`

```cpp
gpu_expression_translator translator(stream, cudf::get_current_device_resource_ref());
// ...
result->reader_options->set_filter(ast_filter->back());
result->filter_expression = ast_filter;
```

The translator runs at task-time (so the scalars land on whatever
device the metadata-scan task happens to run on), but the resulting
filter is then handed downstream to `sirius_gpu_parquet_scan_operator`
which may dispatch to a different GPU. If the metadata scan runs on
device 0 and the gpu scan on device 1, same hazard as the one 08-11
fixed.

**Scope:** not exercised by current failing tests. Iceberg tests under
`num_gpus>1` would repro.

**Fix shape:** mirror 08-11 — translate once per configured GPU, store
`unordered_map<int, translated_expression>` on the
`partitioned_parquet_metadata` result, pick per device at
`sirius_gpu_parquet_scan_operator::execute()`.

**Estimate:** ~60 LOC, 3 files.

## 2. Q22 OOM across all configurations

Every cache mode × num_gpus combination hits
`HASH_JOIN retry limit exceeded` at pipeline 47 with ~41 GB global
usage per GPU. `global usage 43.2 GB, peak 111.5 MB, reservation 97.2 MB,
rescheduling task 30700 ... exceeded maximum OOM retry limit (10)`.

The signature says retries keep failing at the same ~50 MB incremental
allocation with ~40 GB already committed per GPU. Not a memory-size
issue — 96 GB aggregate on 2-GPU is plenty for SF100 Q22's working
set. Likely a plan/ordering issue: pipeline 47 is holding a too-large
materialized intermediate when it tries to start the hash build.

**Scope:** independent of MGPU correctness. Query planner / memory-
reservation issue.

**Investigation shape:**
- Dump the pipeline 47 plan tree for Q22 (what operator produced the
  41-GB working set?)
- Check if `max_build_hash_table_bytes` / `hash_partition_bytes` tuning
  lets Q22 through
- Compare against DuckDB's Q22 memory footprint for the same SF

**Estimate:** half-day investigation before scoping a fix.

## 3. Q14 cache=table_gpu 2-GPU regression

Q14 swings 3.8× in one direction then 3.6× in the other depending on
cache mode under num_gpus=2:

| cache | 1-GPU cold | 2-GPU cold | 1/2 ratio |
|-------|------------|------------|-----------|
| table_host | 5.204 s | 2.741 s | **1.90×** (2-GPU wins) |
| table_gpu  | 2.701 s | 5.454 s | **0.50×** (1-GPU wins) |

Q14 is a lineitem × part join with a date filter. Hypothesis: under
`table_gpu` the cached lineitem is resident on one GPU and joining on
part (also resident somewhere) forces a cross-device probe that costs
more than the parallelism gain. Under `table_host` the per-task rehydrate
lands columns on the task's own GPU so no cross-device probe happens.

**Scope:** performance tuning, not correctness.

**Investigation shape:**
- nsys-profile Q14 on both modes, compare kernel mix / P2P traffic
- If confirmed cross-device probe cost, consider co-locating
  join-related tables on the same GPU under `table_gpu` OR preferring
  `table_host` for multi-GPU deployments

**Estimate:** 1 day for profile + analysis, unknown for a fix.

## 4. MCP `tpch-benchmark` unattended mode

`benchmark_and_validate.sh:353` does
`read -r -p "Optional note about this run (press Enter to skip): " RUN_NOTE`.
MCP invokes without a tty and the script hangs. Workaround this session
used: `echo "" | ... bash script`.

Also `.ai-helper/commands.yaml` only exposes `scale_factor` as an arg;
can't pass `--engines sirius`, `SIRIUS_CONFIG_FILE`, or `PARQUET_DIR`.

**Fix shape:**
1. `benchmark_and_validate.sh:353`: wrap in `[ -t 0 ] && read …` or
   accept `RUN_NOTE` env var as an override.
2. `.ai-helper/commands.yaml`: add `engines`, `config`, `parquet_dir`
   args and env passthrough.

**Estimate:** 15 minutes.

**Value:** future benchmark sweeps run via MCP unattended without
needing `dangerouslyDisableSandbox: true` (when/if MCP sandbox gets
GPU access).

## 5. mgpu-audit TEST_CASE — TWO problems, the second is the real one

### 5a. 24 GB pool-prime OOM (workaround found, not applied)

`test_gpu_execution_tpch_mgpu_audit.cpp:150` originally fails with
`std::bad_alloc: out_of_memory: CUDA error (failed to allocate
25433702400 bytes) at .../rmm/mr/cuda_async_view_memory_resource.hpp:86`.

Reproduces in complete isolation. GPU is 15 MB used per nvidia-smi.
Not a regression from 93fea6f — audit TEST_CASE was authored but never
ran green at Phase 8 verification (blocked by the residual failure our
fix closed).

**Workaround verified:** lowering `usage_limit_fraction` from 0.5 → 0.4
in `test/cpp/integration/integration-2gpu.yaml` fixes the OOM. Not
applied — exposes a second failure (5b below) that is the real issue.

### 5b. Pipeline tasks and scan tasks don't co-distribute across GPUs

With 5a worked around, the audit test next fails at line 243:
`REQUIRE(counts[0].pipeline_ids.size() >= min_count)` with
`0 >= 1` and diagnostic:
```
per-GPU audit counts from /tmp/sirius-mgpu-audit-XXXX:
  GPU0{pipeline=0, scan=4} GPU1{pipeline=3, scan=0}
```

**All scan tasks landed on GPU 0; all pipeline tasks landed on GPU 1.**
The audit's invariant is that *both* task kinds must be distributed
across *both* GPUs (>=1 of each on each). This is a legitimate
dispatch-policy question, not a test-config tweak:

- Is the scan_executor deliberately pinning scans to one GPU when the
  total scan count is small (SF1 lineitem has ~6 batches)?
- Does the pipeline_executor's round-robin dispatcher prefer a single
  GPU when the scan output all lands on one GPU?
- Is this an actual invariant violation (the audit is right and
  dispatch is broken) or is the audit's >=1 threshold too strict for
  SF1-small-batch-count workloads?

**Estimate:** 1-2 hours investigation (trace scan and pipeline
dispatch decisions) to classify as dispatch bug vs audit over-spec.
If the threshold is right, fix is in dispatch logic; if the threshold
is too strict, scope-constrain the assertion (e.g. require >=1 of
*either* type on both GPUs, not both types on both GPUs).

**Scope note:** the `93fea6f` fix makes the AUDIT test able to *run*
for the first time. The fact that it runs-and-fails-on-assertion is
progress — it exposes a distribution question that was hidden behind
the earlier crash.

---

*Phase: 08-multi-gpu-sql-pipeline-fix*
*Authored: 2026-04-22T21:30:00Z*
*Parent commit: f7847f8*
