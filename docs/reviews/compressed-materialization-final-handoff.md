# Compressed Materialization — Final Engineering Handoff

Date: 2026-07-21  
Repository: `/home/kkristensen/Code/sirius_1`  
Branch: `dev`  
Base revision reported by the benchmark harness: `3ee47f8b`  
Status: implemented, reviewed, built, tested, and benchmarked; changes are not committed or staged  
Scope: Super Sirius only

## Executive summary

Compressed materialization is now implemented as a general numeric optimization, not a key-only
special case. It can retain bounded signed integers, unsigned integers, and fixed-point DECIMAL
columns in narrower cuDF physical carriers while preserving their original SQL logical types.

The optimization can narrow data after a fresh scan or while a table is pinned. Narrow carriers flow
through compatible pass-through operators and are restored to the native carrier when an expression
or operator needs native semantics. The first implementation always restores to the native type; it
does not yet co-narrow join equivalence classes or choose a common intermediate type.

The feature is opt-in and defaults to `false`.

The final SF50 TPC-H A/B result was:

| Mode | Feature off | Feature on | Change |
|---|---:|---:|---:|
| Unpinned | 29.714 s | 30.604 s | +3.00% slower |
| HOST-pinned | 10.465 s | 9.404 s | **10.14% faster** |

These totals are sums of the per-query medians from warm iterations 2–6. All four runs validated
22/22 queries against DuckDB. The host-pinned result demonstrates the intended benefit; the unpinned
result shows that exact range reductions and casts currently cost more than they save on that path.

## Original requirements captured

The implementation was designed around these requirements:

1. Use min/max statistics to select narrower physical carriers for integer columns.
2. Support narrowing during a scan and during `pin_table` materialization.
3. Keep data narrow until an operator needs the native representation.
4. Insert restoration casts through the expression executor.
5. Restore to the native SQL carrier in the initial version.
6. Apply the optimization to general columns and payloads, not only keys.
7. Support fixed-point DECIMAL columns without changing precision or scale semantics.
8. Permit batch/chunk-granular narrowing where exact runtime data makes it safe.
9. Run a full TPC-H suite and measure the effect.
10. Review and respond to `docs/reviews/compressed-materialization-review.md`.

## Architecture

### Logical and physical schemas

`sirius_physical_operator::types` remains the authoritative SQL schema. A complete optional
`physical_types` sidecar describes actual cuDF output carriers when at least one column differs from
its native mapping. An empty sidecar means all output columns use native carriers.

This separation is essential for DECIMAL. For example, a SQL `DECIMAL(18,2)` may be physically held
in `DECIMAL32(scale=-2)`, but its precision, scale, aggregate behavior, return type, and host result
remain those of `DECIMAL(18,2)`.

The sidecar is all-or-none when present: it has one physical type for every logical output column.
Planner preflight clears it conservatively if a complete, safe mapping cannot be produced.

### Eligible carriers

Narrowing preserves numeric family, signedness, and DECIMAL scale.

| Native/logical family | Candidate narrower carriers |
|---|---|
| Signed `SMALLINT`/`INTEGER`/`BIGINT` | narrowest fitting `INT8`, `INT16`, or `INT32` |
| Unsigned `USMALLINT`/`UINTEGER`/`UBIGINT` | narrowest fitting `UINT8`, `UINT16`, or `UINT32` |
| `DECIMAL64(scale)` | `DECIMAL32` with the same scale |
| `DECIMAL128(scale)` | narrowest fitting `DECIMAL32` or `DECIMAL64`, same scale |

Already-minimal types, booleans, floating-point values, temporal values, strings, and 128-bit integer
types are not narrowed. DECIMAL precision/scale remains logical metadata; bounds are compared as raw
unscaled signed values.

### Correctness invariants

1. SQL logical types never change.
2. DECIMAL scale never changes during a physical carrier conversion.
3. Signed and unsigned integer families never cross.
4. A downcast is allowed only when the target is strictly narrower and every materialized value fits.
5. A restore is allowed only when it is a same-family widening with matching DECIMAL scale.
6. Planner/source statistics are advisory; exact materialized min/max verifies each scan downcast.
7. Pin-time narrowing is selected from exact per-chunk cuDF min/max results.
8. Join keys, dynamic-filter keys, unsupported boundaries, and final results are native.
9. A nonempty physical sidecar describes the complete output schema.
10. Feature-off fresh scans without a sidecar retain their legacy path.

### Scan-time flow

1. Planning asks the scan source for compatible min/max statistics.
2. Statistics propose a narrower physical target for eligible projected columns.
3. The GPU scan decodes at the source/native width and performs normal filtering/projection.
4. Before any downcast, the scan computes exact min/max over each non-null materialized column.
5. A whole-batch preflight verifies every conversion before any source columns are released.
6. Verified columns are cast to their planned physical targets.
7. Empty and all-null columns are safe without a numeric bound.

Incorrect or incomplete writer metadata therefore cannot silently truncate values. A planned target
whose real materialized values do not fit is treated as an invariant failure instead of performing a
lossy cast.

A fresh scan with no physical sidecar bypasses normalization entirely. A resident cached scan with no
sidecar may still restore a previously pinned narrow carrier to the native schema, but only when the
conversion is a strict compatible numeric widening. Unrelated legacy carrier mismatches retain their
previous pass-through behavior.

### Pin-time flow

For each cache chunk, `pin_table`:

1. Decodes/materializes the chunk at native width.
2. Captures zone-map statistics from the native data where configured.
3. Computes exact min/max for every eligible numeric column.
4. Chooses and casts to the narrowest exact carrier.
5. Stores the resulting GPU table or converts it to pinned host representation.
6. Records a per-chunk/per-column `narrowed_columns` marker matrix.

Different chunks may use different widths. Each cached chunk is served as an individual resident
split and normalized before downstream concatenation, so heterogeneous carrier widths do not enter a
single `cudf::concatenate` call.

The marker matrix is propagated through materialization, cache insertion, replacement/merge, host and
GPU serving, projection/reordering, the batch coalescer, scan operator data, and `input_stats`.
Malformed matrix shapes are rejected.

### Expression restoration

A typed reference compares the actual input carrier with its declared logical/native target. A strict
same-family widening is materialized as a cuDF cast before arithmetic, comparison, `IN`, or another
semantic consumer. Incompatible default-path mismatches remain pass-through to preserve the legacy
contract identified by Claude review finding B3.

Restored references are memoized per top-level evaluation by `(column_index, exact target type)`, so a
predicate such as `a > 1 AND a < 10` performs one restore rather than two. The memo is reset before each
new input table, preventing a cached view from outliving its source batch.

Pure-reference projection fast paths can forward narrow columns without copying.

### Operator boundaries

The propagation pass keeps narrow payloads through compatible identity-preserving operators and
restores columns conservatively where native representation is required.

- Filters, pure-reference projections, limits, and hash-join payload outputs may remain narrow.
- Every hash-join predicate/key is native before partitioning and hashing.
- Aggregates, ordering, unsupported joins, and other unsupported boundaries receive native inputs.
- Dynamic-filter producer/consumer paths remain native.
- Query roots are restored before DuckDB result materialization.
- DELIM_JOIN out-of-band roots participate in whole-plan mapping preflight.

### Memory accounting

Resident scan reservation is based on actual narrowed-cache markers or an explicit sidecar, not on the
current feature flag. This handles pin-on/query-off and pin-off/query-on setting changes safely.

`kMaxNumericCarrierExpansion` is 8. A converting unfiltered resident input accounts for both source and
destination: one stored-width working set plus up to eight stored-width units for the restored output,
for a conservative 9x peak in the `INT8` to `INT64` case. Arithmetic saturates instead of wrapping.
Ordinary resident inputs retain the legacy estimate, and fresh-read/filter accounting remains separate.

### Configuration and observability

YAML:

```yaml
sirius:
  operator_params:
    enable_compressed_materialization: true
```

SQL connection setting:

```sql
SET enable_compressed_materialization = true;
```

The setting defaults to `false`.

`SiriusContext::compressed_materialization_stats` exposes relaxed-atomic counters for:

- `scan_columns_narrowed`
- `scan_columns_restored`
- `pin_columns_narrowed`

Actual scan and pin casts also emit debug logs. Integration tests assert counter deltas so a test
cannot pass merely because feature-on silently selected no narrower carrier.

## Implementation map

| Area | Primary files | Responsibility |
|---|---|---|
| Architecture | `docs/super-sirius/compressed-materialization.md` | Design, invariants, operation, measurement |
| Build wiring | `CMakeLists.txt` | Adds helper implementation and tests |
| Numeric policy | `src/include/helper/numeric_narrowing.hpp`, `src/helper/numeric_narrowing.cpp` | Family checks, direction-safe conversions, carrier selection, exact min/max |
| Configuration | `src/include/sirius_config.hpp`, `src/sirius_config.cpp`, `src/sirius_extension.cpp` | Default-off YAML and `SET` setting |
| Observability | `src/include/sirius_context.hpp`, `src/sirius_context.cpp` | Narrow/restore counters |
| Scan execution | `src/include/op/scan/sirius_gpu_scan_operator.hpp`, `src/op/scan/sirius_gpu_scan_operator.cpp` | Preflight, exact verification, narrow/restore casts, memory estimate |
| Scan data | `src/include/op/scan/sirius_gpu_scan_operator_data.hpp` | Resident narrowed-column state |
| Physical schema | `src/include/op/sirius_physical_operator.hpp` | Logical schema plus optional physical sidecar |
| Scan planning | `src/planner/sirius_plan_get.cpp` | Statistics-to-sidecar candidate mapping |
| Plan propagation | `src/planner/sirius_physical_plan_generator.cpp` | Pass-through propagation and native boundaries |
| Pinning | `src/include/pin_table.hpp`, `src/pin_table.cpp` | Exact per-chunk narrowing and markers |
| Cache serving | `src/include/scan_manager/sirius_scan_manager.hpp`, `src/scan_manager/sirius_scan_manager.cpp` | Marker validation, projection, merge, serving |
| Coalescing | `src/include/scan_manager/load_balancing_scan_batch_coalescer.hpp`, `src/scan_manager/load_balancing_scan_batch_coalescer.cpp` | Carries resident marker state |
| Task accounting | `src/pipeline/gpu_pipeline_task.cpp` | Physical output checks and `input_stats` wiring |
| Expression casts | `src/include/expression_evaluator/expression_evaluator.hpp`, `src/expression_evaluator/expression_evaluator.cpp`, `src/expression_evaluator/specializations/reference.cpp` | Typed restoration and memoization |
| AST reference metadata | `src/include/expression/ast/reference.hpp` | Logical target information for references |
| A/B config | `test/cpp/integration/integration-gb10-compressed-materialization.yaml` | GB10 treatment config; differs from control by one flag |

## Claude review disposition

The source review is `docs/reviews/compressed-materialization-review.md`.

| Finding | Disposition |
|---|---|
| B1: unconditional normalization changes feature-off behavior | Fixed. Fresh no-sidecar scans bypass normalization; compatible resident restore is explicit. |
| B2: direction check did not distinguish narrow from restore | Fixed. Central `can_narrow_to` and `can_restore_to` helpers are direction-explicit and tested. |
| B3: typed-reference throw breaks a documented legacy contract | Fixed. Only valid numeric restores cast; incompatible default-path references pass through. |
| S4: Parquet writer statistics are advisory | Fixed. Exact materialized min/max is checked before every scan downcast. |
| S5: memory estimate keys on the flag and omits live source | Fixed. Actual marker/sidecar state drives a saturating source-plus-destination estimate. |
| S6: tests cannot tell whether narrowing ran | Fixed. Runtime counters, debug logs, positive counter assertions, and real device min/max tests were added. |
| S7: repeated references allocate repeated restores | Fixed. Restores are memoized per evaluate call and tested across two inputs. |
| S8: pinned filters can restore all rows before selection | Correctness is preserved; retained as a known performance limitation for this version. |
| S9: native schema mapping can throw on unsupported logical types | Fixed conservatively through whole-plan mappability preflight and native fallback/clear behavior. |
| U10: full `cudf::data_type` equality | Intentional decision: retain full equality because DECIMAL scale is part of the carrier contract. |
| Q1: duplicated/diverged type predicates | Fixed by centralizing carrier-family, width, direction, and fit logic in `numeric_narrowing`. |
| Q2: raw `std` exceptions | Nonblocking cleanup remains; current invariant failures still include raw standard exceptions. |
| Q3: zero observability | Fixed with context counters and debug logs. |
| Q4: sidecar/API encapsulation | Partially deferred. Setter invariants are used everywhere, but `physical_types` remains publicly stored and some adjacent booleans remain positional. |
| Q5: dead/unreachable code | Genuinely unreachable dynamic-filter sidecar copy and latent install logic were removed. The claim that PARTITION/CONCAT propagation was dead was refuted by pass ordering. |
| Q6: non-projection-pushdown column order | Removed/skipped rather than relying on a width-only invariant. |
| Q7: build/include/format hygiene | Fixed; final clang-format and whitespace checks pass. |
| D1: superseded root notes | User-owned file preserved untracked; authoritative design explicitly labels and cross-links it as historical. |
| D2: documentation claims stronger than implementation | Corrected for advisory statistics, exact verification, filtering paths, cache mechanism, and validation evidence. |
| D3: missing configuration/optimization/scan docs | Added to `configuration.md`, `optimizations.md`, `scan.md`, and the Super Sirius README. |
| D4: orphaned/inconsistent benchmark config | Config now matches the GB10 control exactly except for the feature flag and is documented in the A/B procedure. |

Two independent final read-only audits found no production blocker in observer lifetime, scan range
verification, restoration memo lifetime, pinned-marker alignment/merge/indexing, or default-off
behavior. The only blocker found by the final code reviewer was an INT8 test-fixture helper mismatch;
the test was changed to exercise `INT32` to `BIGINT` restoration and then passed.

## Test and build evidence

### Build and static checks

- `pixi run make`: successful final release build.
- `pixi run clang-format --dry-run --Werror ...`: clean across every changed C++ source/header.
- `git diff --check`: clean.

### Focused tests

| Test selection | Result |
|---|---:|
| `[numeric_narrowing]`, including non-key DECIMAL integration | 9 cases, 605 assertions, passed |
| `restored numeric references are memoized per evaluate call` | 1 case, 265 assertions, passed |
| `[no_history_peak_memory_estimate][gpu_scan]` | 3 cases, 14 assertions, passed |
| `gpu_execution - empty parquet count identity` | 1 case, 49 assertions, passed |
| `[cached_serving][scan_manager]` | 9 cases, 70 assertions, passed |
| Pin-table same-row-count merge integration | 1 case, 31 assertions, passed |

Coverage includes signed/unsigned boundary selection, conversion direction, exact fit checks,
DECIMAL precision eligibility, DECIMAL64/128 raw unscaled min/max, null/empty/all-null device columns,
memo lifetime, cache-marker shape/selection/merge, source-plus-destination memory estimates,
saturation, a non-key DECIMAL payload used directly and in arithmetic, and feature-off counters.

### Complete test suite

The complete 1,970-case executable was run in four GPU-isolated slices:

| Slice | Cases | Assertions | Result | Log |
|---|---:|---:|---|---|
| 0–499 | 500 | 6,620 | passed | `/tmp/sirius-cm-final-0-499.log` |
| 500–999 | 500 | 31,977,041 | passed | `/tmp/sirius-cm-final-500-999.log` |
| 1000–1499 | 500 | 98,472 | passed | `/tmp/sirius-cm-final-1000-1499.log` |
| 1500–1969 | 470 | 162,833 | passed | `/tmp/sirius-cm-final-1500-1969.log` |
| **Total** | **1,970** | **32,244,966** | **passed** | |

The host has one GPU, so existing tests requiring two or more GPUs self-skipped. Three optional SF10
variants also self-skipped because `SIRIUS_TEST_SF10_PATH` was not set. These were environment gates,
not failures. The requested SF50 TPC-H suite ran separately in full.

## TPC-H SF50 measurement

### Method

- Dataset: `test_datasets/tpch_parquet_sf50`
- Queries: all TPC-H Q1–Q22
- Iterations: six per query
- Primary metric: median of warm iterations 2–6 for each query, summed across Q1–Q22
- Binary: the same final release binary for feature-off and feature-on runs
- Config: `integration-gb10.yaml` versus the otherwise-identical treatment YAML
- Modes: unpinned and `SIRIUS_PIN_TIER=host` per-query pinning
- Validation: all results compared to 22 stored DuckDB result sets from
  `runs/2026-07-20_21-35-46_sf50_6iter/duckdb`
- Hardware reported by the harness: NVIDIA GB10, 16 CPUs, approximately 125 GB RAM

The harness's built-in “warm” column reports the best warm iteration. This handoff instead uses the
requested five-run median. For reference, warm-best totals were 28.12 s to 29.39 s unpinned and
10.31 s to 9.27 s host-pinned, consistent with the median direction.

### Summary

| Mode | Off median total | On median total | On versus off | Validation |
|---|---:|---:|---:|---:|
| Unpinned | 29.714 s | 30.604 s | +3.00% slower | 22/22 in both runs |
| HOST-pinned | 10.465 s | 9.404 s | **10.14% faster** | 22/22 in both runs |

On the host-pinned path, 20 queries improved, Q22 was unchanged at the reported precision, and Q1
regressed by 3.11%. On the unpinned path, three queries improved, Q2 was unchanged, and eighteen
regressed.

### Per-query warm medians

Positive change is slower; negative change is faster.

| Query | Unpinned off | Unpinned on | Change | HOST off | HOST on | Change |
|---|---:|---:|---:|---:|---:|---:|
| Q1 | 1.541 | 1.602 | +3.96% | 0.675 | 0.696 | +3.11% |
| Q2 | 0.786 | 0.786 | +0.00% | 0.152 | 0.131 | -13.82% |
| Q3 | 1.372 | 1.371 | -0.07% | 0.533 | 0.454 | -14.82% |
| Q4 | 0.595 | 0.617 | +3.70% | 0.273 | 0.251 | -8.06% |
| Q5 | 1.191 | 1.290 | +8.31% | 0.393 | 0.323 | -17.81% |
| Q6 | 0.816 | 0.817 | +0.12% | 0.262 | 0.222 | -15.27% |
| Q7 | 1.424 | 1.443 | +1.33% | 0.414 | 0.323 | -21.98% |
| Q8 | 1.557 | 1.573 | +1.03% | 0.476 | 0.382 | -19.75% |
| Q9 | 2.591 | 2.596 | +0.19% | 1.018 | 0.908 | -10.81% |
| Q10 | 2.164 | 2.229 | +3.00% | 0.534 | 0.473 | -11.42% |
| Q11 | 0.313 | 0.373 | +19.17% | 0.141 | 0.131 | -7.09% |
| Q12 | 0.787 | 0.807 | +2.54% | 0.383 | 0.352 | -8.09% |
| Q13 | 2.016 | 1.907 | -5.41% | 0.584 | 0.574 | -1.71% |
| Q14 | 0.959 | 1.038 | +8.24% | 0.252 | 0.181 | -28.17% |
| Q15 | 1.149 | 1.210 | +5.31% | 0.261 | 0.201 | -22.99% |
| Q16 | 0.719 | 0.710 | -1.25% | 0.223 | 0.222 | -0.45% |
| Q17 | 1.392 | 1.533 | +10.13% | 0.493 | 0.444 | -9.94% |
| Q18 | 1.744 | 1.919 | +10.03% | 0.999 | 0.978 | -2.10% |
| Q19 | 1.421 | 1.462 | +2.89% | 0.625 | 0.504 | -19.36% |
| Q20 | 1.604 | 1.616 | +0.75% | 0.292 | 0.232 | -20.55% |
| Q21 | 2.745 | 2.847 | +3.72% | 1.341 | 1.281 | -4.47% |
| Q22 | 0.828 | 0.858 | +3.62% | 0.141 | 0.141 | +0.00% |
| **Total** | **29.714** | **30.604** | **+3.00%** | **10.465** | **9.404** | **-10.14%** |

### Run artifacts

| Mode | Setting | Run directory |
|---|---|---|
| Unpinned | off | `runs/2026-07-21_20-39-00_sf50_6iter` |
| Unpinned | on | `runs/2026-07-21_20-44-05_sf50_6iter` |
| HOST-pinned | off | `runs/2026-07-21_20-51-53_sf50_6iter` |
| HOST-pinned | on | `runs/2026-07-21_20-59-53_sf50_6iter` |

Each directory contains `timings.csv`, `validation.csv`, `comparison.txt`, `run_info.txt`, the exact
Sirius config, per-query results, and per-query logs. Every `validation.csv` contains 22 successes and
zero failures.

### Interpretation

The host-pinned result is the strongest evidence for the design: narrower cached host transfers and
GPU-resident carriers reduce bandwidth and downstream work enough to improve the full suite by about
10%. The benefit is broad rather than isolated to keys.

The unpinned path currently pays for exact `cudf::minmax` verification and conversion kernels on
freshly decoded batches. At SF50 those costs exceed the bandwidth savings by about 3%. The feature
therefore remains default-off, and a production rollout should initially favor pinned workloads or add
a cost/eligibility policy for scan-time narrowing.

The numbers in repo-root `compressed-materialization-notes.md` describe a superseded offset-based
prototype and must not be compared directly with these carrier-width results.

## Known limitations and recommended follow-ups

1. **Pinned filter restoration (Claude S8).** A cached narrow predicate column is restored over the
   full resident chunk before the filter selects survivors. Explore comparisons in the narrow domain,
   compatible literal conversion, or filter-aware late restoration.
2. **Unpinned cost model.** Avoid exact reductions where projected bytes or expected reuse cannot repay
   the kernels. Consider source-trusted exact metadata, sampling, or a query/batch cost threshold.
3. **More batch-granular scan selection.** Pinning chooses independently per chunk; scan planning still
   proposes a plan-wide target and verifies each materialized batch. A later version could select the
   narrowest carrier directly from each batch's exact range.
4. **Join equivalence classes.** The current version restores join keys. A more sophisticated version
   could choose a common narrow carrier for compatible equality classes.
5. **Dynamic filters.** Dynamic-filter scans are deliberately native. Coordinated narrow key types are
   a future optimization.
6. **API cleanup (Claude Q2/Q4).** Make the physical sidecar private, use checked access consistently,
   replace positional feature booleans with options structs, and use project-specific invariant
   exceptions.
7. **Additional planner matrix tests.** Claude proposed broader pure plan-tree tests for every join
   kind, delim joins, root restoration, aggregate/order boundaries, and dynamic filters. Existing
   integration/full-suite coverage passed, but a dedicated planner test file would make the contract
   easier to maintain.
8. **Multi-GPU validation.** Repeat the complete suite on a 2+ GPU host; this host had one GB10.
9. **Benchmark scope.** Repeat on another scale factor/hardware and record pin setup cost separately if
   deciding a default or rollout policy.

## Reproduction commands

All repository commands should be run through `pixi run` per `CLAUDE.md`.

### Build

```bash
pixi run make
```

### Focused tests

```bash
pixi run ./build/release/extension/sirius/test/cpp/sirius_unittest \
  '[numeric_narrowing]' -r compact -a

pixi run ./build/release/extension/sirius/test/cpp/sirius_unittest \
  'restored numeric references are memoized per evaluate call' -r compact -a

pixi run ./build/release/extension/sirius/test/cpp/sirius_unittest \
  '[no_history_peak_memory_estimate][gpu_scan]' -r compact -a

pixi run ./build/release/extension/sirius/test/cpp/sirius_unittest \
  '[cached_serving][scan_manager]' -r compact -a
```

### Complete test suite in reliable slices

```bash
pixi run ./build/release/extension/sirius/test/cpp/sirius_unittest \
  --start-offset 0 --end-offset 500 -r compact -a -o /tmp/sirius-cm-final-0-499.log

pixi run ./build/release/extension/sirius/test/cpp/sirius_unittest \
  --start-offset 500 --end-offset 1000 -r compact -a -o /tmp/sirius-cm-final-500-999.log

pixi run ./build/release/extension/sirius/test/cpp/sirius_unittest \
  --start-offset 1000 --end-offset 1500 -r compact -a -o /tmp/sirius-cm-final-1000-1499.log

pixi run ./build/release/extension/sirius/test/cpp/sirius_unittest \
  --start-offset 1500 --end-offset 1970 -r compact -a -o /tmp/sirius-cm-final-1500-1969.log
```

Run slices sequentially; concurrent GPU tests distort memory behavior and can cause resource failures.

### TPC-H A/B using the existing DuckDB results

Unpinned control:

```bash
pixi run bash test/tpch_performance/benchmark_and_validate.sh \
  --config "$PWD/test/cpp/integration/integration-gb10.yaml" \
  --parquet-dir "$PWD/test_datasets/tpch_parquet_sf50" \
  --engines "sirius" --iterations 6 --timeout 3600 \
  --duckdb-results "$PWD/runs/2026-07-20_21-35-46_sf50_6iter" \
  --pinning-mode none 50 < /dev/null
```

Unpinned treatment:

```bash
pixi run bash test/tpch_performance/benchmark_and_validate.sh \
  --config "$PWD/test/cpp/integration/integration-gb10-compressed-materialization.yaml" \
  --parquet-dir "$PWD/test_datasets/tpch_parquet_sf50" \
  --engines "sirius" --iterations 6 --timeout 3600 \
  --duckdb-results "$PWD/runs/2026-07-20_21-35-46_sf50_6iter" \
  --pinning-mode none 50 < /dev/null
```

HOST-pinned control:

```bash
SIRIUS_PIN_TIER=host pixi run bash test/tpch_performance/benchmark_and_validate.sh \
  --config "$PWD/test/cpp/integration/integration-gb10.yaml" \
  --parquet-dir "$PWD/test_datasets/tpch_parquet_sf50" \
  --engines "sirius" --iterations 6 --timeout 3600 \
  --duckdb-results "$PWD/runs/2026-07-20_21-35-46_sf50_6iter" \
  --pinning-mode per-query 50 < /dev/null
```

HOST-pinned treatment:

```bash
SIRIUS_PIN_TIER=host pixi run bash test/tpch_performance/benchmark_and_validate.sh \
  --config "$PWD/test/cpp/integration/integration-gb10-compressed-materialization.yaml" \
  --parquet-dir "$PWD/test_datasets/tpch_parquet_sf50" \
  --engines "sirius" --iterations 6 --timeout 3600 \
  --duckdb-results "$PWD/runs/2026-07-20_21-35-46_sf50_6iter" \
  --pinning-mode per-query 50 < /dev/null
```

## Workspace state and preservation requirements

The implementation is intentionally left uncommitted and unstaged. Generated benchmark run
directories are present under `runs/` and are ignored by normal source status.

Do not delete or overwrite these user-owned/unrelated untracked paths:

- `compressed-materialization-notes.md`
- `docs/reviews/compressed-materialization-review.md`
- `test_datasets/tpch_parquet_sf30/`
- `test_datasets/tpch_parquet_sf50/`
- `test_datasets/tpchgen-rs/`

New implementation files that must be included if preparing a commit are:

- `docs/reviews/compressed-materialization-final-handoff.md`
- `docs/super-sirius/compressed-materialization.md`
- `src/include/helper/numeric_narrowing.hpp`
- `src/helper/numeric_narrowing.cpp`
- `test/cpp/helper/test_numeric_narrowing.cpp`
- `test/cpp/integration/integration-gb10-compressed-materialization.yaml`

The current environment intermittently fails the normal patch helper or sandbox startup with
`bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`. The established fallback was an
explicit, narrowly scoped `git apply` patch outside the failing sandbox. Do not use destructive Git
commands to work around this.

## Recommended next-agent checklist

1. Read `AGENTS.md`, `CLAUDE.md`, this handoff, the authoritative architecture document, and Claude's
   original review.
2. Inspect `git status --short` and preserve all unrelated/user-owned files listed above.
3. Review the final diff, especially `numeric_narrowing`, scan preflight, cache marker propagation,
   and plan boundary insertion.
4. Decide rollout policy from the data: host-pinned workloads benefit; unpinned SF50 regresses.
5. If changing production code, rerun the focused tests, all four full-suite slices, and the matched
   TPC-H A/B before updating the recorded result.
6. If preparing a PR, include this handoff and the five new implementation/docs/test files; keep the
   feature default
   off, and do not present the historical offset-prototype numbers as evidence for this design.
7. Consider addressing S8 and the Q2/Q4 API cleanup before enabling the feature broadly.

## Final status

The general integer-and-DECIMAL compressed-materialization implementation is complete and has no
known correctness blocker. It remains default-off, produces a strong measured gain for HOST-pinned
TPC-H SF50 workloads, has a measurable unpinned regression, and carries explicit follow-ups for filter
restoration, cost selection, API cleanup, and broader multi-GPU benchmarking.

---

## Addendum — performance follow-ups implemented (2026-07-21, evening)

This addendum records a second engineering pass that implemented recommended follow-ups #1
(pinned filter restoration / review S8) and #2 (unpinned cost model), re-validated, and
re-measured. Both changes strictly reduce work; neither alters results or feature-off behavior.

### Change A — narrow-domain comparisons (evaluator)

A comparison or `BETWEEN` whose column operand is a narrowed reference and whose constant
operands are exactly representable in that carrier (typed NULLs always are; decimal constants
must match the carrier scale) now evaluates directly on the narrow carrier: the raw column plus
constants converted host-side, on both the cuDF-AST and binary-operator paths. Narrowing is
value-preserving — same values, family, and decimal scale, no offset — so every comparison
outcome including NULL handling is identical at either width. Ineligible shapes fall back to the
restore path unchanged. Filter masks over narrow resident chunks are now computed at narrow
width and survivors gathered narrow; restoration applies to survivors at their consumers instead
of to the full chunk before selection, resolving the S8 limitation for the dominant filter
shapes. `IN`-list narrow evaluation is a remaining follow-up (TPC-H `IN` lists are
predominantly strings, which never narrow).

Files: `src/expression_evaluator/specializations/narrow_domain.cpp` (new),
`comparison.cpp`, `between.cpp`, `expression_evaluator.{hpp,cpp}`, `CMakeLists.txt`.
A `narrow_domain_comparison_count_for_testing()` accessor makes engagement observable to tests.

### Change B — zero-benefit scan-narrowing pruning (planner)

After propagation inserts restoration boundaries, `prune_immediate_scan_restores` removes
scan-time narrowing that a restore projection undoes before any batch is materialized narrow:
the restore sits directly above the scan (join keys, aggregate/ordering inputs, root restores)
or is separated from it only by zero-copy pure-reference projections. Such columns paid exact
range verification plus a narrowing cast and a widening cast with zero narrow batch writes in
between. Pruned columns become native in the scan sidecar, their restore casts collapse to
passthrough references, and a restore projection reduced to a positional identity is removed
(returning its pipeline stage). Pin-time narrowing is unaffected — a pruned sidecar restores
resident narrow chunks during scan normalization instead of one operator later.

Files: `src/planner/sirius_physical_plan_generator.cpp` (new pass + call site in `create_plan`).

### Validation

- Build clean; `clang-format --dry-run --Werror` clean on every changed file.
- New unit tests: 6 narrow-domain evaluator cases (AST and binary-op paths, INT and DECIMAL
  carriers, non-representable and scale-mismatch fallbacks, BETWEEN); the restoration-memo test
  was updated to use constants outside the carrier so it still exercises the restore path.
- New integration tests: aggregate-only and join-key-only scans assert
  `scan_columns_narrowed` does NOT move (pruning proof); the existing payload test still
  asserts it does.
- Full suite: 1,978/1,978 cases passed (~32.2M assertions) in sequential GPU-isolated slices
  (`/tmp/sirius-cm-perf-*.log`).

### Measured SF50 result (same method as above; 6 iterations, medians of warm 2–6)

| Mode | Feature off | Feature on | Change | Prior change |
|---|---:|---:|---:|---:|
| Unpinned | 29.536 s | 29.914 s | **+1.28%** | +3.00% |
| HOST-pinned | 10.501 s | 9.304 s | **−11.40%** | −10.14% |

All four runs validated 22/22 against the stored DuckDB results. Feature-off totals drifted
±0.6% versus the morning baseline (same-day noise), so cross-day per-query deltas should be read
cautiously; the off/on pairs above are same-binary, same-session comparisons.

Notable per-query movements versus the prior treatment: unpinned Q5 +8.31% → −9.20%,
Q12 +2.54% → −8.70%, Q14 +8.24% → −3.10%, Q11 +19.17% → +7.28%; host-pinned Q6 −15.27% → −20.24%,
Q20 −20.55% → −26.73%, Q17 −9.94% → −14.09%. The residual unpinned cost is concentrated in
queries whose narrowed columns feed arithmetic-only consumers directly above the scan (Q1, Q17):
those restore inside the expression evaluator, which pruning deliberately does not cover — a
per-column expression-use cost model is the next refinement.

Run directories:

| Mode | Setting | Run directory |
|---|---|---|
| Unpinned | off | `runs/2026-07-21_22-25-46_sf50_6iter` |
| Unpinned | on | `runs/2026-07-21_22-28-50_sf50_6iter` |
| HOST-pinned | off | `runs/2026-07-21_22-31-57_sf50_6iter` |
| HOST-pinned | on | `runs/2026-07-21_22-34-01_sf50_6iter` |

### Remaining follow-ups (delta to the list above)

Follow-up #1 is resolved for comparison/BETWEEN shapes (`IN` remains). Follow-up #2 is
substantially mitigated; the full per-column cost model (expression-use analysis through mixed
projections, e.g. Q1's arithmetic-only columns) remains open. Follow-ups #3–#9 are unchanged.

