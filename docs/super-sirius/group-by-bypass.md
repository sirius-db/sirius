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
unary projection, filter or limit steps ending at a result collector. Sorting, joins and other
plan shapes retain normal partitioning.

## Memory accounting

The model charges additional allocations for concatenation, the grouping hash table, a row-index
map, intermediate aggregates and final output. Output is sized for the worst case of one group
per partial row. Null masks and a possible downstream copy are included. The model rejects
arithmetic overflow and inputs beyond cuDF's row-index limit.

The budget is the memory space's **reservation cap minus charged bytes**, clamped at zero.
Charged bytes already include resident input and outstanding reservations, so they are not
subtracted again or added to the model. Physical free VRAM is not the reservation budget.

Selection requires the model plus a configurable margin to fit. A selected merge reuses its
additional-allocation estimate, before that margin, through `no_history_peak_memory_estimate()`.
The cold-start request is at least the existing 2× input estimate. The executor adds input
materialization costs and records the request through its existing reservation telemetry.

Memory history belongs to the current query's pipeline. The partition count is chosen before
merge tasks run, and a bypassed single partition produces one merge task; its first attempt has
no prior partitioned-merge history. After an OOM, the existing history and task-local retry floor
size subsequent requests, including resumes after the merge in a fused pipeline. No additional
executor-wide floor or separate history is introduced.

The partition reads the enable flag and the merge reads headroom from the existing immutable
query policy snapshot. A transient input summary carries only the observed rows, column layout,
residency, target device, and reservation-budget snapshot. It is not a repository cache: batch
placement can change after each read handle is released. No persistent row metrics are needed
for this complete-input decision.

This remains an estimate: the allocator assumptions need verification against the pinned cuDF
implementation. Selection does not reserve memory, and the executor can grant a partial
reservation. Existing executor diagnostics report reservation shortfalls; automatic
repartitioning after OOM is not implemented. The partition count stays fixed once chosen.

## Settings and validation

`enable_group_by_memory_aware_bypass` is **false by default**. Internal settings require
`SIRIUS_ENABLE_TEST_OPTIONS=1`. `group_by_bypass_headroom_fraction` defaults to 0.25 and accepts
values from 0 to 4. Decisions log the reason, automatic/chosen counts, estimated bytes and budget.

Run focused tests with:

```bash
pixi run build/release/extension/sirius/test/cpp/sirius_unittest '[group_by_bypass]'
```

The full experiment harness and detailed design are preserved on branch
`backup/groupby-bypass-full-20260921`. Earlier GB300 measurements at `2a64af22` showed 1.15× and
1.35× engine speedups for two supported SF1000 group-bys. All 54 off/on pairs at SF100/SF1000
matched the CPU reference, but TPC-H never activated the bypass. Those measurements predate the
standalone port and this cleanup; they do not validate this revision's GPU integration.

Before default-on, verify the model, secure memory before committing to bypass, and add bounded
OOM recovery that repartitions preserved input without duplicate output. Validate recovery
under injected allocation failures and concurrent memory pressure as a separate change.
