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

## 5. mgpu-audit TEST_CASE cannot prime 24 GB RMM pool

`test/cpp/integration/test_gpu_execution_tpch_mgpu_audit.cpp:150` fails
on this host (2× RTX 6000 Ada 48 GB) with
`std::bad_alloc: out_of_memory: CUDA error (failed to allocate
25433702400 bytes) at .../rmm/mr/cuda_async_view_memory_resource.hpp:86`
when `acquire_integration_env_for(2)` → `env->resume()` →
SiriusContext → `cuda_async_memory_resource(capacity=24GB)` tries to
prime the pool via an initial alloc/dealloc on device 0.

Reproduces in complete isolation (1 test, fresh process). GPU is 15 MB
used per nvidia-smi, so 24 GB should fit. Likely a CUDA/driver/rmm
interaction specific to this host — not introduced by 93fea6f. Audit
TEST_CASE was authored but never ran green at Phase 8 verification
(blocked by the earlier residual failure our fix closes).

Full suite run (no --abort): **983 tests, 981 passed, 2 failed** on
feature/single-node-multi-gpu2. One of the 2 is this audit case; the
other is likely a sibling that shares the 2-GPU env.

**Scope:** not a regression from 93fea6f. Investigation needs to look at
pool prime timing vs pre-existing CUDA context state, or lower
`usage_limit_fraction` in `integration-2gpu.yaml` to see if the pool
primes cleanly at 0.4 or 0.3.

**Estimate:** 30 min investigation; fix size unknown.

---

*Phase: 08-multi-gpu-sql-pipeline-fix*
*Authored: 2026-04-22T21:30:00Z*
*Parent commit: f7847f8*
