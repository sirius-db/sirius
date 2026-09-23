# Memory-aware group-by bypass

An opt-in implementation of point 2 of [issue #1746](https://github.com/sirius-db/sirius/issues/1746).
It is independent of projected partition sizing (#1765). With runtime size estimation disabled,
the existing full-input barrier is preserved. If estimation fixes the partition count before the
producer finishes, bypass is declined and that count remains fixed.

## Behavior

The merge first computes the normal partition count. If that count exceeds one, the bypass
may choose one instead, reusing the existing path that forwards batches without redistributing
rows. Otherwise the normal count remains unchanged.

The bypass requires complete input resident in one GPU memory space, one admitted GPU, and
known physical column metadata. Keys and partial aggregate states must be fixed-width integers
or Booleans; accepted aggregate operations are SUM, COUNT, MIN and MAX. AVG, COUNT(DISTINCT),
floating/decimal states and variable-width columns are excluded. Later operations may only be
unary projection, filter or limit steps ending at a result collector, and every column a
projection computes must have a fixed-width result type. Sorting, joins, variable-width
computed columns and other plan shapes retain normal partitioning.

## Memory accounting

The model charges additional allocations for concatenation, the grouping hash table, a row-index
map, intermediate aggregates and final output. Output is sized for the worst case of one group
per partial row. Every column is charged its own allocator padding and, where it may be
nullable, a validity mask. Downstream steps are charged the columns they materialize: a
projection pays for each computed column at its result width (pure column references are
zero-copy), a filter pays for its selected output columns, and a limit pays for a copy of its
input row. Filter column drops and reordering update the widths used by later steps. Charges are
summed across the chain. Temporaries inside nested projection expressions are not sized and fall under
the headroom. The model rejects arithmetic overflow and inputs beyond cuDF's row-index limit.

The budget is the memory space's **reservation cap minus charged bytes**, clamped at zero.
Charged bytes already include resident input and outstanding reservations, so they are not
subtracted again or added to the model. Physical free VRAM is not the reservation budget.

Selection requires the model plus a configurable margin to fit. A selected merge reuses its
additional-allocation estimate, before that margin, through `no_history_peak_memory_estimate()`.
The cold-start request is at least the existing 2× input estimate. The executor adds input
materialization costs and records the request through its existing reservation telemetry.
Headroom is used as selection slack rather than part of the predicted allocation peak, keeping
the cold-start estimate on the same basis as subsequent measured history. That slack is not
reserved, and the selection check does not guarantee a full reservation grant.

The first bypassed merge has no prior partitioned-merge history. Subsequent attempts reuse the
query's existing pipeline memory history and task-local OOM retry floor, including resumes after
a fused merge. No separate reservation or retry mechanism is introduced.

Both settings come from the immutable query policy. Input metadata is collected lazily, only
when the automatic count exceeds one on a single admitted GPU. The summary is transient:
batch placement and the reservation budget can change after inspection.

The model is an estimate, not a guaranteed allocation bound. Selection does not reserve memory,
and the executor can grant a partial reservation. Automatic repartitioning after OOM is not
implemented; the chosen count stays fixed.

## Settings and validation

`enable_group_by_memory_aware_bypass` is **false by default**. Internal settings require
`SIRIUS_ENABLE_TEST_OPTIONS=1`. `group_by_bypass_headroom_fraction` defaults to 0.25 and accepts
finite values from 0 to 4. Selections are logged at INFO and evaluated rejections at DEBUG,
including counts, estimated bytes and budget. Required bytes include the configured margin.

Run focused tests with:

```bash
pixi run build/release/extension/sirius/test/cpp/sirius_unittest '[group_by_bypass]'
```

## Measured results

On one GB300, six selected custom group-bys over TPC-H SF1000 data ran 1.08–1.25× faster
(engine time) with the bypass on, with automatic counts of 2–7 reduced to 1. Both arms matched
CPU references. These are selected eligible shapes, not standard TPC-H queries or an expected
workload-wide speedup. Allocation traces for two of them peaked below the model (model-to-peak
about 1.3–1.4×). These measurements predate the downstream-accounting changes and are not
validation of the latest revision.

## Before default-on

Before default-on, verify the model, secure memory before committing to bypass, and add bounded
OOM recovery that repartitions preserved input without duplicate output. Validate recovery
under injected allocation failures and concurrent memory pressure as a separate change.
