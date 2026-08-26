# Late materialization: carrying a rowid instead of the values

A query that selects wide columns usually carries them from the scan to whatever finally looks
at them — through joins that copy them beside their keys, partitions that write them to a
repository and read them back, and aggregates that group on them. Nothing in that stretch reads
what is IN them.

Late materialization replaces those columns at the scan with a pin-order **rowid**, carries that
instead, and puts the values back at the far end by gathering them out of the pinned table. It
works only for a **GPU-tier pinned** table — the values have to still exist somewhere
addressable when the far end asks for them, and that is what the pin guarantees.

## Turning it on (experimental)

```bash
SIRIUS_EXP_LATE_MAT=1                               # the gate; off = nothing below exists
SIRIUS_EXP_LATE_MAT_PIN_UNIQUE_COLS=c_custkey,...   # or `all`
```

For the TPC-H SF1000 q10 ride specifically, the column list needs to cover BOTH the group-by-rowid
key and every rider's key: `c_custkey` (customer, the ride itself) plus `n_name,n_nationkey`
(nation, riding alongside on the join to customer):

```bash
SIRIUS_EXP_LATE_MAT_PIN_UNIQUE_COLS=c_custkey,n_name,n_nationkey   # needed for q10's full ride
```

Off by default and inert when off. The census goes to the log under the `[late-mat]` tag, and
every refusal says which refusal it was — a deferral that silently did not happen looks exactly
like one that did nothing.

The uniqueness probe costs pin-time work, so name the columns a ride actually needs rather than
using `all`; without a proof there is no group-by-rowid ride and no riders.

| Variable | Default | Effect |
|---|---|---|
| `SIRIUS_EXP_LATE_MAT` | off | The master gate. |
| `SIRIUS_EXP_LATE_MAT_PIN_UNIQUE_COLS` | off | Columns the pin-time uniqueness probe observes: `all`, a name list, or `none`. |
| `SIRIUS_EXP_LATE_MAT_EXACT_MAX_ROWS` | 300000000 | Row cap on the exact uniqueness stage, so a fact table does not buy a sort per column at pin time. |
| `SIRIUS_EXP_LATE_MAT_MIN_BOUNDARIES` | 4 | Port crossings a ride must save. |
| `SIRIUS_EXP_LATE_MAT_MIN_VALUE_X_BOUNDARIES` | 128 | Net bytes/row TIMES crossings saved — value and crossings trade off, so this is the floor that matters. |
| `SIRIUS_EXP_LATE_MAT_COUNT_DEFER` | off | Count-on-deferred (below). |
| `SIRIUS_EXP_LATE_MAT_MIN_VALUE_COMPRESSED` | = the ordinary floor | Separate floor for compressed origins. Inert until the decode-skip exists to measure. |

## What a deferral is

**A pair of instructions, installed together or not at all.** The scan stops emitting the
values; the consumer puts them back. Either half alone is a wrong answer, so
`planner::install_deferral` writes both or neither. (Count-on-deferred, below, is the one
exception: when NOTHING downstream reads the values — every reader only counts — there is no
"putting them back" to do, so that deferral installs a single half by design, not by omission.)

Between the two ends the deferred columns ride as ordinary data: a rowid at the FIRST deferred
position and 1-byte placeholders at the rest. Arity and positions are preserved, so every
operator in between sees the shape it expected and needs to know nothing about any of this.

**The two ends speak different coordinate systems.** A join widens and reorders the table, so
both halves carry their own schema and positions, and the consumer matches a batch by its WHOLE
schema — materializing against the wrong batch would read arbitrary rows of the pinned table.
`port_materialize_directive` (`defer_directive.hpp`) carries only that schema and the origins it
was installed for — no producer or operator id. Schema equality is the entire identity check.

That sounds thinner than it is. Two structurally identical batches from different producers can
only collide at a PORT that has more than one producer feeding it — and `trace_through`'s switch
(`late_mat_plan_pass.cpp`) gives ride-preserving handling to only PARTITION, CONCAT, HASH_JOIN,
DYNAMIC_FILTER, the two group-by shapes, TOP_N/MERGE_TOP_N, FILTER, and PROJECTION. Every other
operator type — UNION, CROSS_PRODUCT, and every join variant except HASH_JOIN (NESTED_LOOP_JOIN,
BLOCKWISE_NL_JOIN, PIECEWISE_MERGE_JOIN, IE_JOIN, the delim joins, POSITIONAL_JOIN, ASOF_JOIN) —
falls to the walk's default (`return step::reads()`), which ends any ride there: no rowid crosses
a UNION or a non-hash join, so neither can converge two independently-deferred batches on one
port. HASH_JOIN is therefore the only multi-producer operator a ride can cross — which is exactly
the self-join / two-scans-of-one-table case — and it is handled by the RIDER mechanism, not by
hoping schemas differ: `install_rider` (`late_mat_plan_pass.cpp`) requires the merged directive's
`expected_schema` to match the existing one at every position except the new rider's own, and
`port_materialize_directive::valid()` re-checks self-consistency and position collisions before
the merge is accepted. Two scans converging at one port get ONE directive with each side's rowid
at its own fixed position — disambiguated structurally, not by a runtime schema coincidence.

## Which columns, and how far

`planner::analyze_column_lifetimes` walks up from the scan and asks, per column: which operator
first reads its CONTENT, and how many PORT crossings did it cross to get there? A crossing is
counted when the operator being left is a pipeline sink; a filter or projection hands its
columns on inside one pipeline and a wide column rides past it for free.

**The walk fails closed.** Any operator shape it does not model reads everything, which can only
end a ride early.

Two things then decide whether the ride is taken:

- **It has to repay.** Value per row and crossings saved TRADE OFF — a thin ride over many
  crossings can pay where a fat one over few does not — so what is weighed is their product.
- **It must not sit above a fan-out.** A join emits one scan row once per match, so a port above
  one gathers a row set larger than the scan produced. The walk tracks whether a join below a
  column multiplied the rows and whether a reduction has undone it, and a port that is still
  fanned out is refused. This is the difference between a ride that wins and one that costs
  more than an order of magnitude.

One bundle installs per scan (one rowid rides), and arbitration is widest-wins.

## Riding through a group-by, and riders

If the deferred columns are the GROUP BY keys, the sound stop is the aggregate's input — which
is the join output, one row per match. Grouping by the rowid instead moves the materialization
to the aggregate's OUTPUT, one row per group. That reduction is also what clears the fan-out,
which is why it is the shape that pays.

**Why the answer does not change:** the ride is admitted only when a column that rides REAL is
proven distinct over the whole pinned table AND is a group key at every ridden aggregate. Each
group is then exactly one row of that table, so every row of a group carries the same rowid. A
deferred column that is itself unique proves it just as well.

The same argument is why a partition ABOVE a local aggregate may hash a riding key while a
partition below a JOIN may not. A rowid is a bijective relabelling of a proven-unique column, so
equal values on THIS side really do still imply equal rowids — that alone would not break a
partition hashing only this side. What breaks it is the OTHER side of the join: it still hashes
the real value, since it is not the side the deferral was proven against, so a row pair that
matches on value can land in different partitions once one side is rehashed by rowid and the
other is not. Equal keys must land in one partition, and a rowid does not preserve that agreement
across the two sides. That narrow exception is the difference between a fast query and a wrong
one.

Two pinned tables can materialize at one consumer: each **rider** carries its own rowid and
origins, because a rowid means nothing outside the table it indexes. A rider is admitted either
by its own columns being distinct, or by meeting the ride's scan on the two sides of one
equality condition with its side proven distinct. It is weighed at "any saving at all" — the
ride it joins has already paid the crossing, so a rider repays only its own rowid.

## Proving a column distinct, at pin time

Established once, at pin time, in two stages (`late_mat::unique_probe`): per chunk, no nulls
with distinct count equal to row count and pairwise disjoint ranges; then, exactly, for what the
ranges leave undecided, by sorting the concatenated chunks and counting runs. The exact stage is
load-bearing rather than a fallback, because chunk ranges usually overlap.

`SIRIUS_EXP_LATE_MAT_EXACT_MAX_ROWS` is a ROW cap only — there is no separate memory guard on the
exact stage. It concatenates and sorts, so a wide or variable-width column just under the cap is
a multi-GB GPU allocation at pin time, on top of the pin itself. This is the sharpest edge of
`PIN_UNIQUE_COLS=all`: it is not just "extra pin-time work" (above) but, for a wide column near
the cap, a pin-time OOM risk rather than a slow pin. Name the columns a ride actually needs.

**Absence of a fact means UNKNOWN, never "not unique."** A false positive would collapse
distinct groups into one — wrong answers, not slow ones. Facts attach to the pinned entry BY
NAME, since a re-pin that merges columns would make a positional attach mark the wrong one.

## Filtered scans, and count-on-deferred

A filtered batch is no longer the chunk's rows in order, so the rowid comes from the surviving
row positions, taken from the same mask the scan filters with.

**A batch off a compressed pin can be restricted twice.** The fused decode drops rows while
decoding (see [Compressed Pinning](compressed-pinning.md)), and whatever conjuncts it could not
carry are evaluated again on what it kept. Each stage reports positions in ITS OWN input, so the
second stage's positions index the first stage's output rather than the chunk, and the pin-order
rowid is only right once the two are composed — `compose_survivors`
(`sirius_gpu_scan_operator.cpp`) gathers the decode's list by the residual's.

The decode reports its half because it is asked to: a split carrying a deferral sets
`late_mat_wants_survivors`, which `prepare_for_processing` turns into
`decompression_pushdown_scan::with_survivor_reporting()`, and the decode then expands the
selection mask it already balloted into an ascending INT32 index list
(`pushdown_outcome::survivor_rows`). Off a deferring scan the decode does neither, so the cost is
one index-list expansion plus 4 bytes per surviving row, paid only where a ride uses it.

Admitting the shape is not the same as it paying. On TPC-H SF1000 with
`PIN_UNIQUE_COLS=all` it turns 39 structural refusals into ordinary policy decisions and installs
no additional deferral: the newly reachable candidates are `lineitem` and `orders` filtered scans,
and they are withheld because a partition hashes them or refused on the value-times-crossings
floor. Suite time is unchanged (6.312 s before, 6.304 s / 6.326 s after, same build and machine),
which is what a change that only removes a refusal should look like where the refusal was not the
binding constraint on value.

Both directions fail closed. A decode that compacted without accounting for its survivors throws
rather than emit a rowid, at `prepare_for_processing` and again at substitution; a survivor list
whose length does not match the rows handed on throws too. A scan whose INGESTIBLE cannot report
survivors (the duckdb-native one filters with a plain select) is still refused at install, since
the residual half would have nothing to say.

A column every reader only COUNTs needs no far end at all: `install_count_deferral` is the one
deferral with a single half, admitted only over pinned columns with no nulls. Off by default.

## Known limits

- A column an outer join could null is withheld: a null rowid must materialize a null, which the
  materializer does not do yet.
- A nullable PINNED SOURCE column is admitted for any origin whose gather path propagates
  validity, which is now every one of them. The uncompressed shapes (single-batch, multi-batch
  fixed-width, multi-batch variable-width) do it natively. A COMPRESSED origin does it through
  the validity sidecar compression stores beside each column's plan tree: the count is readable
  from the `.hpln` header without decoding (`simpatico::column_null_count`), so the install gate
  can tell "no nulls" apart from "unknown", and the decode routes reattach the mask
  (`materialize.cpp`'s `attach_selected_validity`). The two COMPACTING routes gather the stored
  bitmask by the same rows they selected the values by — the mask describes the whole chunk while
  a compacted output holds only the selection, so copying it verbatim would pair each value with
  another row's validity. A chunk carrying no blob still answers nothing and is refused.
- A compressed origin cannot skip its decode — the scan substitutes on the FINISHED output, so a
  deferred column from a compressed pin is decompressed and then discarded.
- Deferred-value widths are ESTIMATED for variable-width columns, not measured, and the error runs
  both ways: underestimating a wide column can refuse a bundle that would have qualified, but
  overestimating a short one can just as easily admit a bundle that does not actually repay (a
  handful of short strings can clear the floor on the estimate while saving fewer bytes/row than
  the measured value would show). The real fix is measuring the width — the scan already has the
  offsets for it.
- **Host-tier pins are refused**, not just unpinned scans: `install_late_materialization`
  declines on tier, and `resolve_pinned_layout`/`resolve_pinned_column` refuse independently at
  the far end. Supporting them would mean staging a chunk back to the device per gather, so the
  ride would have to repay a host round trip rather than a device read.
- **A non-pinned scan cannot defer at all.** A fresh parquet or DuckDB-native read consumes its
  decoded batch, so there is nothing for the port to gather from. Deferring against a file would
  mean a rowid addressing a file offset and a re-read per gather — the shape classic disk-based
  late materialization takes, and a different cost model from this one, whose floors are
  calibrated against a device-memory gather.
- **A pin's lifecycle is `pin_table`/`unpin_table`, not the downgrade executor** — pinned entries
  are not spilled to reclaim memory the way ordinary data batches are, so a chunk backing an
  installed deferral does not move tiers mid-query on its own. What DOES change it — an explicit
  `unpin_table`, a re-pin that replaces the entry, or an in-place column merge — bumps or
  invalidates the entry's generation (`pin_entry_handle`, `column_origin.hpp`). A consumer
  resolving an origin against a generation that no longer matches gets `nullopt`, never a stale or
  dangling pointer. Fail-closed here means the PORT THROWS, not that the query quietly re-reads:
  by that point the scan has already emitted rowids in place of the values, so there is nothing to
  fall back to locally. A re-read happens only if an outer layer catches the error and replays the
  query. The guarantee is that changed data is never materialized against — not that the query
  survives it unaided.
- **Multi-GPU pins are refused.** `resolve_pinned_column`/`materialize()` pass a pin's raw column
  views and compressed-table pointers straight to a gather on the consumer's current GPU, with no
  per-device tag and no P2P check, clone, or host-staging fallback — a chunk pinned on a different
  GPU than the consumer could otherwise be dereferenced directly. Installation refuses outright
  whenever more than one GPU memory space is active, matching the memory prefetcher's own
  single-GPU prototype scope.

## Results

GB300, TPC-H SF1000, unpatched libcudf, GPU-pinned, `SIRIUS_EXP_FUSED_SCAN_FILTER=1` held on in
both arms, best-of-3, 22/22 byte-identical validated vs DuckDB CPU. The suite gain is q9 and
q10; no other query moves outside noise.

| | Suite | q9 | q10 |
|---|---|---|---|
| Gate off | 7.3911 s | 0.8481 s | 0.4991 s |
| Gate on | **7.0031 s** | **0.7611 s** | **0.2250 s** |

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
