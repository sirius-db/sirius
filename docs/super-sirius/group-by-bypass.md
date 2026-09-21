# Memory-aware group-by bypass

An opt-in implementation of point 2 of [issue #1746](https://github.com/sirius-db/sirius/issues/1746).
It is independent of projected partition sizing (#1765) and preserves the full-input barrier.

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

Selection requires the model plus a configurable margin to fit. A selected merge supplies its
additional-allocation estimate, before that margin, as a minimum memory request. That floor
applies even when previous executions supplied a smaller learned estimate. Larger historical
estimates and OOM retry requests still win.

This remains an estimate: the allocator assumptions need verification against the pinned cuDF
implementation. Selection does not reserve memory, and the executor can grant a partial
reservation. A grant below the floor is logged; automatic repartitioning after OOM is not
implemented. The partition count stays fixed once chosen.

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
