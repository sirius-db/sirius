# Late materialization: carrying a rowid instead of the values

A query that selects wide columns usually carries them from the scan all the way to whatever
finally looks at them — through joins that copy them beside their keys, partitions that write
them to a repository and read them back, and aggregates that group on them. Nothing in that
stretch reads what is IN them.

Late materialization replaces those columns at the scan with an 8-byte (or 4-byte) **pin-order
rowid**, carries that instead, and puts the values back at the far end by gathering them out of
the pinned table. On TPC-H q10 at SF1000 that is 0.4510 s → 0.2359 s; the five `customer`
columns are 158 B/row and the query carries them across eleven port crossings before anything
needs them.

It works only for a **pinned** table — the values have to still exist somewhere addressable when
the far end asks for them, and a pin is what guarantees that.

## Turning it on (experimental)

```bash
SIRIUS_EXP_LATE_MAT=1                          # the gate; off = nothing below exists
SIRIUS_EXP_LATE_MAT_PIN_UNIQUE_COLS=all        # or a comma-separated name list
```

Off by default and inert when off: no origins are stamped, no pin handles are published, and
the plan pass reports lifetimes nobody reads. The census goes to the log at `info` under the
`[late-mat]` tag — every refusal says which refusal it was, because a deferral that silently
did not happen looks exactly like one that did nothing.

```
operator 13: deferring 5 column(s) to RESULT_COLLECTOR (id=28) — 96 B/row over 11 boundaries
operator 10: riding along with the bundle at RESULT_COLLECTOR (id=28) — 1 column(s), unique key 'n_name'
operator 3:  not deferring — the scan restricts rows and its pin is compressed
```

### All gates

| Variable | Default | Effect |
|---|---|---|
| `SIRIUS_EXP_LATE_MAT` | off | The master gate. |
| `SIRIUS_EXP_LATE_MAT_PIN_UNIQUE_COLS` | off | Columns the pin-time uniqueness probe observes: `all`, a comma-separated name list, or `none`/`off`/`0`. Without a proof there is no group-by-rowid ride and no riders. |
| `SIRIUS_EXP_LATE_MAT_EXACT_MAX_ROWS` | 300000000 | Row cap on the exact uniqueness stage, so a fact table does not buy a multi-second sort per column at pin time. |
| `SIRIUS_EXP_LATE_MAT_MIN_BOUNDARIES` | 4 | Port crossings a ride must save to be worth taking. |
| `SIRIUS_EXP_LATE_MAT_GBR_MIN_GROUP_ROWS` | 0 (inert) | Floor on the first ridden aggregate's input rows for a group-by-rowid ride. Never calibrated. |
| `SIRIUS_EXP_LATE_MAT_COUNT_DEFER` | off | Count-on-deferred (below). Dark: the shapes it fires on save ~4 B/row. |
| `SIRIUS_EXP_LATE_MAT_MIN_VALUE_COMPRESSED` | = the ordinary floor | Separate value floor for compressed origins. Inert until the decode-skip exists to measure. |

## What a deferral is

**A pair of instructions, installed together or not at all.** The scan stops emitting the
values; the consumer puts them back. Either half alone is a wrong answer — the scan half alone
throws data away, the consumer half alone corrupts a batch that was already correct — so
`planner::install_deferral` writes both or neither.

Between the two ends the deferred columns ride as ordinary data: a UINT64/UINT32 rowid at the
FIRST deferred position and 1-byte placeholders at the rest. Arity and positions are preserved,
so every operator in between sees a table of the shape it expected and needs to know nothing
about any of this.

**The two ends speak different coordinate systems.** A join widens and reorders the table, so a
column deferred at scan position 1 may arrive at the consumer as column 7 of a table twice as
wide. Both halves therefore carry their own schema and their own positions, and the consumer
matches a batch by its WHOLE schema — a matcher that only checked "is there a UINT64 here"
would fire on an unrelated batch, and materializing against the wrong batch reads arbitrary
rows of the pinned table.

## Which columns, and how far

`planner::analyze_column_lifetimes` walks up from the scan and asks, per column: which operator
first reads its CONTENT, and how many PORT crossings did it cross to get there? A crossing is
counted when the operator being left is a pipeline sink — a filter or projection hands its
columns on inside one pipeline and a wide column rides past it for free.

**The walk fails closed.** Any operator shape it does not model reads everything, which can only
end a ride early. Modelled: partitions and concats (positionally transparent), hash joins,
filters, projections, dynamic filters, both group-by shapes, and both top-n shapes.

`late_mat::defer_policy` then decides whether the ride is worth taking. Both floors come from
measurement, not intuition: a bundle of 11–25 B dimension columns COST +61 ms on an 800M-row
port, while a 154.6 B bundle and a 50 B pair both won. One bundle installs per scan (one rowid
rides), and arbitration is widest-wins.

## Riding through a group-by

If the deferred columns are the GROUP BY keys, the sound stop is the aggregate's input — which
is the join output, one row per match. Grouping by the rowid instead moves the materialization
to the aggregate's OUTPUT, one row per group. On q10 that is the difference between tens of
millions of gathered rows and ~150k.

**Why the answer does not change:** the ride is admitted only when a column that rides REAL is
proven distinct over the whole pinned table AND is a group key at every ridden aggregate. Then
each group is exactly one row of that pinned table, so every row of a group carries one and the
same rowid — grouping by the id yields exactly the groups grouping by the values did. A
deferred column that is itself unique proves it just as well, since the rowid is then a
bijective relabelling of that column.

The same argument is why a partition ABOVE a local aggregate may hash a riding key (all rows of
a group share the rowid, so the group lands where its values would have), while a partition
below a JOIN may not — equal keys must land in one partition, and a rowid does not preserve
that. That narrow exception is the difference between a fast query and a wrong one.

Two more admission checks: every pipeline hop from the aggregate to the far consumer must feed
exactly ONE port (another producer's batch reaching the port would be materialized against
origins that do not describe it), and the far consumer must exist.

A top-n on the way is transparent — it reads its sort keys and gathers whole rows — so the ride
usually ends at the result collector, materializing the rows the query actually returns.

## Riders: a second table on the same ride

Two pinned tables can materialize at one consumer. On q10 `customer` rides five wide columns
and `nation` rides `n_name` beside it. Each rider carries its OWN rowid, origins and width,
because a rowid means nothing outside the table it indexes.

A rider is admitted two ways:

1. **Its own columns prove it** — one of them is distinct over its pinned table.
2. **The ride determines it** — the rider and the ride's scan meet on the two sides of ONE
   equality condition of one join, with the rider's side proven distinct over its pin. Then at
   most one rider row exists per row of the other side, and the other side's row IS the group.

A rider is weighed at "any saving at all" rather than the bundle floor: the ride it joins has
already paid the crossing, so what a rider must repay is only its own rowid.

## Proving a column distinct, at pin time

Everything above rests on "this column is distinct over the pinned table", which is established
once, at pin time, in two stages (`late_mat::unique_probe`):

- **Per chunk:** no nulls, distinct count == row count, and the chunks' `[min, max]` ranges
  pairwise DISJOINT. Disjoint rather than strictly increasing, because our `part.0 … part.14`
  files glob lexicographically and `part.10` is read before `part.2`. The count comes off
  SORTEDNESS, never a hash set — a hash `distinct_count` over a gigabyte-sized pinned chunk
  overruns cuco's representable extent and fails the whole pin.
- **Exactly, for what the ranges leave undecided:** concatenate the chunks, sort, count
  consecutive runs. Any type cuDF can sort, strings included. This stage is load-bearing, not a
  fallback: the coalescer interleaves files' row groups rather than partitioning the key space,
  so chunk ranges usually overlap. `c_custkey` at SF1000 proves in 24 ms.

**Absence of a fact means UNKNOWN, never "not unique."** A false positive would collapse
distinct groups into one — wrong answers, not slow ones. Verdicts are a tri-state so that
"repeats a value" is not re-checked exactly while "ranges overlap" is, and facts are attached to
the pinned entry BY NAME, because a re-pin that merges columns appends them and a positional
attach would mark the wrong one.

## Filtered scans

A filtered batch is no longer the chunk's rows in order, so the rowid comes from the surviving
row positions: `expression_evaluator::select_with_survivors` returns them from the same mask it
filters with, and the scan emits `origin.start + survivor` instead of a sequence. Using one mask
is the point — a second evaluation could describe different rows than the output holds.

Refused where the survivors are decided somewhere this path cannot see them: a COMPRESSED pin
filters inside the fused decode (see [Compressed Pinning](compressed-pinning.md)), and a rowid
guessed there would address rows the batch does not hold.

## Count-on-deferred (dark)

A column every reader only COUNTs needs no far end at all — the aggregate counts the rowid and
gets the same answer. `install_count_deferral` is the one deferral with a single half, and it is
admitted only over pinned columns with NO nulls, since counting a rowid that is never null would
count rows `COUNT(col)` skips. Off by default: the shapes that fire it on TPC-H save ~4 B/row,
which no A/B run could separate from noise.

## Results (GB300, TPC-H SF1000, GPU-pinned, narrowing off)

| | Suite | q10 |
|---|---|---|
| Gate off | 7.1878 s | 0.5202 s |
| Gate on | **6.8801 s** | **0.2359 s** |

The suite gain is q10; no other query moves outside noise. The hand decomposition bounds the
prize: q10 verbatim 531 ms, payload as `min()` 462 ms, payload absent 215 ms — carrying the
payload through the joins costs 247 ms, and having the columns as GROUP BY keys a further 69 ms.

**`enable_compressed_materialization = false` is required to see this.** It defaults ON, and
narrowing a deferred column suppresses the deferral: the bundle stops at the hash join instead
of reaching the aggregate. Disabling it costs ~0.17% on its own, so it is close to a pure
unlock — but it is the reason these numbers are quoted with narrowing off.

## Known limits

- A column an outer join could null is withheld: a null rowid must materialize a null, which
  the materializer does not do yet.
- A compressed origin cannot skip its decode. The scan substitutes on the FINISHED output, so a
  deferred column from a compressed pin is decompressed and then discarded;
  `SIRIUS_EXP_LATE_MAT_MIN_VALUE_COMPRESSED` exists to price that case once it can be skipped.
- Filtered scans of compressed pins are refused (above).
- Deferred-value widths are ESTIMATED at 24 B for variable-width columns rather than measured,
  which can only refuse a bundle that would have qualified.

## Where the code is

| Piece | File |
|---|---|
| What a deferral is (the pair, the substituted schemas) | `src/include/late_mat/defer_directive.hpp` |
| Whether a bundle is worth deferring, and the floors | `src/include/late_mat/defer_policy.hpp` |
| How long each scanned column's values are needed | `src/planner/late_mat_plan_pass.cpp` |
| Admission (uniqueness proof, pipelines, riders) | `src/scan_manager/sirius_scan_manager.cpp` |
| Pin-time distinctness proof | `src/late_mat/pin_uniqueness.cpp` |
| Putting the values back | `src/late_mat/port_materialize.cpp`, `materialize.cpp` |
| Reading a pinned entry the way the materializer needs it | `src/scan_manager/late_mat_resolver.cpp` |
