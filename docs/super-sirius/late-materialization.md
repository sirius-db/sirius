# Late materialization: carrying a rowid instead of the values

A query that selects wide columns usually carries them from the scan to whatever finally looks
at them — through joins that copy them beside their keys, partitions that write them to a
repository and read them back, and aggregates that group on them. Nothing in that stretch reads
what is IN them.

Late materialization replaces those columns at the scan with a pin-order **rowid**, carries that
instead, and puts the values back at the far end by gathering them out of the pinned table. It
works only for a **pinned** table — the values have to still exist somewhere addressable when the
far end asks for them, and that is what the pin guarantees. A GPU-tier pin is gathered out of
device memory; a HOST-tier pin is gathered out of pinned host memory in place, over unified
virtual addressing, with nothing staged back to the device (see [Host-tier pins](#host-tier-pins)).

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
using `all`; without a proof there is no group-by-rowid ride and no riders. On TPC-H SF1000 that
cost is the ONLY difference `all` makes (see Results): it proves eight more columns distinct and
not one of them changes an admission, so the choice between `all` and the narrow list is a pin-time
question, not a query-time one.

| Variable | Default | Effect |
|---|---|---|
| `SIRIUS_EXP_LATE_MAT` | off | The master gate. |
| `SIRIUS_EXP_LATE_MAT_PIN_UNIQUE_COLS` | off | Columns the pin-time uniqueness probe observes: `all`, a name list, or `none`. |
| `SIRIUS_EXP_LATE_MAT_EXACT_MAX_ROWS` | 300000000 | Row cap on the exact uniqueness stage, so a fact table does not buy a sort per column at pin time. |
| `SIRIUS_EXP_LATE_MAT_MIN_BOUNDARIES` | 4 | Port crossings a ride must save. |
| `SIRIUS_EXP_LATE_MAT_MIN_VALUE_X_BOUNDARIES` | 128 | Net bytes/row TIMES crossings saved — value and crossings trade off, so this is the floor that matters. |
| `SIRIUS_EXP_LATE_MAT_COUNT_DEFER` | off | Count-on-deferred (below). |
| `SIRIUS_EXP_LATE_MAT_MIN_VALUE_COMPRESSED` | = the ordinary floor | Separate floor for compressed origins. Inert until the decode-skip exists to measure. |
| `SIRIUS_EXP_LATE_MAT_HOST_COST_MULTIPLIER` | 12 | What the value x crossings floor is multiplied by for a HOST-tier pin. |

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
at its own fixed position — disambiguated structurally, not by a runtime schema coincidence. No
code change identified as necessary; recorded here because the reasoning isn't obvious from the
"schema equality is the entire identity check" statement alone.

## Which columns, and how far

`planner::analyze_column_lifetimes` walks up from the scan and asks, per column: which operator
first reads its CONTENT, and how many PORT crossings did it cross to get there? A crossing is
counted when the operator being left is a pipeline sink; a filter or projection hands its
columns on inside one pipeline and a wide column rides past it for free.

**The walk fails closed.** Any operator shape it does not model reads everything, which can only
end a ride early.

Two things then decide whether the ride is taken:

- **It has to repay.** Value per row and crossings saved TRADE OFF — a thin ride over many
  crossings can pay where a fat one over few does not — so what is weighed is their product. What
  the ride saves is the values less the CARRIER: the rowid plus a 1-byte placeholder for every
  column past the first, so an n-column bundle costs `rowid + (n - 1)` bytes per row to carry.
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
row positions, taken from the same mask the scan filters with. It is refused where the survivors
are decided somewhere this path cannot see them — a compressed pin filters inside the fused
decode (see [Compressed Pinning](compressed-pinning.md)).

A column every reader only COUNTs needs no far end at all: `install_count_deferral` is the one
deferral with a single half, admitted only over pinned columns with no nulls. Off by default.

## Known limits

- A column an outer join could null is withheld: a null rowid must materialize a null, which the
  materializer does not do yet.
- A nullable PINNED SOURCE column is admitted for any uncompressed origin (single-batch,
  multi-batch fixed-width, or multi-batch variable-width — every such gather path propagates
  validity). A compressed origin is refused UNCONDITIONALLY, whether or not it actually has any
  nulls: per-column nullability inside a Simpatico-compressed blob is opaque (see
  [Compressed Materialization](compressed-materialization.md)), so nothing upstream of the
  install-time gate can tell "no nulls" apart from "unknown," and the gate treats both as unsafe.
  `materialize_compressed`'s decode routes in `materialize.cpp` do write values only, with no
  output validity buffer — but that is a secondary reason and moot in practice, since a compressed
  origin never reaches them today.
- A compressed origin cannot skip its decode — the scan substitutes on the FINISHED output, so a
  deferred column from a compressed pin is decompressed and then discarded.
- Filtered scans of compressed pins are refused (above).
- Deferred-value widths are ESTIMATED for variable-width columns, not measured, and the error runs
  both ways: underestimating a wide column can refuse a bundle that would have qualified, but
  overestimating a short one can just as easily admit a bundle that does not actually repay (a
  handful of short strings can clear the floor on the estimate while saving fewer bytes/row than
  the measured value would show). The real fix is measuring the width — the scan already has the
  offsets for it.
- **A host-tier pin rides only FIXED-WIDTH, uncompressed columns**; see
  [Host-tier pins](#host-tier-pins).
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

## Host-tier pins

A host-tier pin stores its data the other way round from a GPU one: `entry.host_chunks` holds one
representation PER EMITTED BATCH, each carrying every pinned column, where a GPU pin holds a chunk
vector per column. The resolver is what bridges the two orientations, and there is no per-column
merge path on the host side because of it.

**The gather reads the rows where they lie**, rather than staging the chunk back to the device per
materialization. Measured on GB300 (Grace-Blackwell, C2C rather than PCIe), a 150M x 8 B pinned
column stages H2D at 163 GB/s, so a full stage costs 7.4 ms whatever the selection, while a blocked
zero-copy gather of the same column costs 0.012 ms at 0.01% selectivity, 0.42 ms at 1%, and 5.1 ms
even when the selection covers every row. Cheaper at every selectivity, and it allocates only its
output rather than putting the whole column back in device memory, which is what pinning on the
host was avoiding.

**The deferred columns are never served.** A pinned scan carrying a deferral withholds them: the
provider drops their entry positions from what it projects, so a HOST-tier chunk never stages them
across the link and a GPU-tier chunk never copies them. The batch then arrives NARROWER than the
scan's arity, which two things downstream absorb — `assemble_scan_output` renumbers the output
layout past the missing columns (`renumber_output_for_withheld`), and `substitute_deferred_columns`
INSERTS the rowid and its placeholders at their positions instead of overwriting values that are
there. Withholding and the older elision are alternatives, never both: elision drops columns the
batch DID carry, and paid to read them.

Measured, 500k customers x 40 deferred BIGINTs over two joins and two group-by stages, host pin:

| | H2D bytes | of which the pinned scan | GPU kernel time |
|---|---|---|---|
| Gate off | 196.231 MB | 55.983 + 22.0 + 4.0 MB | 27.6 ms |
| Gate on, serving the deferred columns | 196.259 MB | 55.983 + 22.0 + 4.0 MB | 33.0 ms |
| Gate on, withholding them | **116.259 MB** | **2.0 MB** | 33.5 ms |

The middle row is what the ride cost before withholding existed: per-copy sizes identical to gate
off, not merely the totals, so the payload crossed in full whether or not it rode. The last row
lands on the 116.228 MB a control that projects only the key measures, which is what identifies
the bytes that disappeared as exactly the deferred payload.

**Withholding is refused rather than guessed at.** It requires a parquet plan (the duckdb-native
reader has no output layout to renumber), no row filter (a post-decode filter evaluates over the
reader's D-order batch, so a missing column shifts every filter reference past it, and a deferred
column can itself be a filter column), no partition columns (synthesized rather than read, so
nothing could have withheld one), and at least one column left over. Where any of those fails the
scan serves the columns and elides them from the projection, exactly as before.

Against a device gather the same column costs 2.0x, 8.1x, 10.6x, 12.2x, 11.0x and 6.6x at those
selectivities. That worst case is `SIRIUS_EXP_LATE_MAT_HOST_COST_MULTIPLIER`: the value x crossings
floor is calibrated against a device gather, and both the floor and the gather scale with the same
row count, so a host-tier bundle is weighed against `128 x 12` instead of `128`. It bounds one
operation rather than calibrating a query, which is why it is a knob. It does NOT price the
scan-time staging above, which no floor currently accounts for.

**How a chunk-major host chunk becomes a column-major view.** A host chunk's storage is a list of
equally sized pinned blocks that are NOT contiguous with one another, so a column's buffer has no
single base pointer and cannot be described by a `cudf::column_view` at all. What it does have is a
byte offset into the logical concatenation of those blocks (`cucascade::memory::column_metadata`),
and `batch_source::host` carries exactly that: the block pointers, the block size, and the data and
null-mask offsets. `multi_source_gather_fixed_host` turns a global rowid into a batch, a row, a byte
offset, a block index and an offset within it — one divide on top of the batch search the GPU-tier
gather already does. Validity comes along one mask word at a time, in the same coordinates.

The block pointers are used as device addresses directly. That is legitimate only where the device
reports both `cudaDevAttrUnifiedAddressing` and `cudaDevAttrCanUseHostPointerForRegisteredMem`; the
resolver asks once and refuses host-tier deferral outright when either is false, rather than
translating or staging.

**What a host-tier pin refuses**, beyond every refusal the GPU tier makes:

- **A variable-width or nested column.** There is no element width to multiply a row by, and its
  offsets would have to be rebuilt against buffers that are not contiguous. Refused per COLUMN, not
  per pin: the fixed-width columns of the same chunks stay eligible.
- **A compressed host chunk** (`compressed_host_representation`). A Simpatico blob has no
  addressable per-row layout to read in place, and its per-column nullability is opaque besides —
  the same reason a compressed GPU chunk is refused.
- **A translation that does not land on a boundary.** The block size must divide by the element
  width, the data offset must start on an element, and the mask offset must be word-aligned. A pin
  that fails any of these is refused rather than read crookedly.
- **A filtered scan**, which was already refused for host pins and stays refused.

`install_late_materialization` and the rider pass check exactly what `resolve_pinned_column` checks,
for the same reason the nullability gate does: by the time a port resolves an origin the scan has
already emitted rowids in place of the values, so a refusal there is a thrown query rather than a
slower one.

## Results

GB300, TPC-H SF1000, unpatched libcudf, GPU-pinned, `SIRIUS_EXP_FUSED_SCAN_FILTER=1` held on in
both arms, best-of-3, 22/22 byte-identical validated vs DuckDB CPU. The suite gain is q9 and
q10; no other query moves outside noise.

| | Suite | q9 | q10 |
|---|---|---|---|
| Gate off | 7.3911 s | 0.8481 s | 0.4991 s |
| Gate on | **7.0031 s** | **0.7611 s** | **0.2250 s** |

### `PIN_UNIQUE_COLS`: the narrow list versus `all`

Same machine, ONE build (upstream `dev` at `8c88f2f3`), three arms back to back, best-of-3, 22/22
results byte-identical across all three:

| `PIN_UNIQUE_COLS` | Suite | q9 | q10 | Pin time, 72 `pin_table` calls |
|---|---|---|---|---|
| `none` | 6.6852 s | 0.8523 s | 0.4995 s | 77.6 s |
| `c_custkey,n_name,n_nationkey` | 6.3065 s | 0.7596 s | 0.2264 s | 77.7 s |
| `all` | **6.2992 s** | 0.7648 s | 0.2257 s | 91.5 s |

`all` does not regress: no query moves by more than 0.7%, and the per-operator census is
IDENTICAL between the narrow list and `all` — same installs, same refusals, same byte counts. The
eight extra columns `all` proves distinct (`c_name`, `c_address`, `p_partkey`, `s_suppkey`,
`s_name`, `s_address`, `r_name`, `r_regionkey`) do not unlock a single admission, because every
other candidate is stopped by a floor, the fan-out guard, or the compressed-filtered-scan refusal
— none of which uniqueness affects. The measured 0.7% spread is the run-to-run noise floor: a
fourth arm executing an identical plan set moved q9 by the same amount.

What `all` does cost is **+13.8 s of pin time (+17.8%)**, which grouped-mode query timings do not
include. That is the price of not making the user name columns.

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
| Gathering by global rowid, device and host tiers | `src/late_mat/multi_source_gather.cu` |
