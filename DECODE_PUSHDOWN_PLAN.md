# Filter-into-decompression: API and renderer plan

Target: `exp/fused-scan-filter` (PR #1391) and follow-ups.

## 0. What this is about

Two PRs push filtering into decompression:

- **#1380** (closed) — answer a string equality/IN off a dictionary's key set instead of
  gathering the decoded chars. Column comes back BOOL8.
- **#1391** (open) — decode-time selection masks: range/pair/dict/membership conjuncts
  produce a mask during decode, output columns are decoded survivor-compacted.

**Note: #1380's work is now on `dev`.** It arrived via #1371 (`84c49bee`, "TPC-H SF1000
campaign integration") — `decode_predicate`, `decode_equality_pushdown` and
`extract_string_equality_pushdown` are all in mainline; #1391's fused scan-filter machinery
(`decompress_scan_filter`, `selection_mask`, `range_predicate`) is not, yet.

So this plan covers refactoring APIs that are **already shipped**, not only unmerged ones. In
particular P5 — the BOOL8 type sniff, and the hazard that a genuine BOOL8 column becomes
indistinguishable from a substituted one once the pushdown widens past VARCHAR — is a fix to
`dev` rather than a cleanup of a PR. Part A's design is unchanged; its starting point is
`dev`'s shipped surface plus #1391's additions.

The mechanism works and is measured (SF1000 8.180 s → 6.918 s). The problem is the boundary:
~40 new public names across five layers, four encodings of one internal fact, and two
out-of-band signalling channels. This plan keeps the mechanism and replaces the boundary.

## 1. Problems, with evidence

**P1 — Four encodings of "how does this column decode." CLOSED (`e1442655`, `52cc1b5a`).**
The sirius-side mirror went with the facade; the rest collapsed into
`codegen::decode_route` (`full, bitpack_mask, delta_mask, dict_codes, str_split`), which is now
the only encoding — `request.routes`, `result.routes` and `decode_selection::route` are the same
value. `full` IS "not compactable", so the parallel capability vector has nothing left to say.

**P2 — Five capability probes. CLOSED (`52cc1b5a`).** Six, counting
`column_supports_predicate_decode`. All are now `simpatico::probe_column(tree)` →
`column_decode_caps{compact_route, can_answer_equality}`, with `can_produce_mask()` derived. The
invariant that was comment-enforced is unstateable: one probe cannot disagree with itself.

**P3 — Four parallel conjunct vectors. CLOSED (`ec38c489`).** Assembly now builds one
`selection_source` list; its size is the count and the cap, one comparator states the ordering
(exact conjuncts, which cost no probe launch, ahead of join filters by kind then key count), and
one switch emits it. `scan_filter_request` still carries four vectors — that is the kernels'
dispatch shape, not the caller's problem.

**P4 — Ragged two-call protocol. CLOSED (`7aaf3da8`).** `decompress_scan_filter` returns a
`cudf::table` that is already uniformly survivor-sized; `compact_scan_filter_output` is internal.
A refused assembly falls back to the unfiltered decode and reports `status = failed`, so no caller
can hold a half-filtered batch either.

**P5 — Type sniffing. CLOSED (`11cef624`).** `parquet_gpu_ingestible.cpp:886` infers the BOOL8 substitution from
`batch.column(pos).type().id() == BOOL8`, then rebuilds the filter expression and throws if the
rebuild degenerates (`:908-912`). Unambiguous today only because candidates are VARCHAR-only;
extend the pushdown to numeric or boolean equality and a genuine BOOL8 column becomes
indistinguishable.

**P6 — RTTI as a signalling channel. CLOSED (`2b085d81`).** `row_filtered_gpu_table_representation` and
`rule2_bailed_gpu_table_representation` each carry one bit via dynamic type;
`clone()` "intentionally degrades". Safe only because two call sites are adjacent on one thread.

**P7 — Pushdown carriers duplicated. CLOSED (`e1442655`).** Five setter/getter pairs on both
representation classes, with `clone()` obliged to copy each (#1380 already had to patch `clone()`
for one field). Both classes now carry one `shared_ptr<const compressed_scan>`.

**P8 — Policy in three places. CLOSED (`e1442655`, `660ce427`).** The env gate and the selectivity
ceiling were copied per TU and had *drifted* — two readers accepted only `"1"` where the decoder
accepts anything but `"0"` — so `SIRIUS_EXP_FUSED_SCAN_FILTER=true` turned the feature on in one
layer and off in another. All six knobs now live once, with the decode that acts on them
(`codegen/selection/decode_policy.hpp`); the sirius-side header re-exports the two the scan uses.
No `fusion_policy` object was needed — free functions with one parse helper each say the same
thing without a lifetime to thread.

**P9 — Emitter duplication.** See §5.

## 2. Part A — top-level API (intent only)

The caller has three things to say: which columns, why each, and what must be true of the rows.

```cpp
namespace sirius {

enum class column_use : std::uint8_t {
  value,        // I will read this column's values
  filter_only,  // exists only to evaluate `where` — drop it if you handle that part
};

struct requested_column { std::size_t column; column_use use = column_use::value; };

struct scan_read {
  std::span<const requested_column> columns;
  ast::node const* where = nullptr;   // the caller's WHOLE filter, undigested
};

struct scan_read_result {
  std::unique_ptr<cudf::table> table;
  std::vector<std::optional<std::size_t>> position;  // requested i -> table col; nullopt = elided
  std::int64_t rows = 0;
  ast::owned_node unapplied;   // what you must still evaluate; null => none
};

/// One logical scan. Shape analysis once; each read decides afresh per chunk.
class compressed_scan {
 public:
  explicit compressed_scan(scan_read plan);
  scan_read_result            read (compressed_chunk const&, rmm::cuda_stream_view, mr);
  std::optional<std::int64_t> count(compressed_chunk const&, rmm::cuda_stream_view, mr);
};

}
```

Rules:

- **`where` goes down whole.** The engine decomposes it — that is the "decide the how" step.
  The three scan-side `extract_*` functions become one internal analysis pass.
- **Dynamic join filters ride inside `where`** as one AST node kind (`x IN <filter-handle>`).
  Shape is stable (analysed once), contents snapshotted per `read`. No `generation` in the API.
- **`unapplied` is a predicate, not indices.** Rewrite rule: drop the top-level AND conjuncts
  applied; if the predicate is not a top-level AND, the residual is the whole thing. Replaces
  `covers_whole_filter`, `scan_filter_status`, the `applied` bool, both marker representation
  classes, and `build_filter_expression_for`.
- **Elision invariant:** a column is elided **iff** its use is `filter_only` and `unapplied`
  does not reference it. "Elided" therefore means provably unreachable.
- **The session owns adaptive state:** RULE-2 bail latch, JIT cache keying, per-scan policy.
  `decode_hint` is unnecessary — the engine simply stops trying.

Caller code, complete, covering both PRs' optimizations:

```cpp
// bind time (roles come from _plan->pure_filter_batch_positions(), already computed)
_scan = std::make_unique<compressed_scan>(scan_read{ .columns = cols, .where = _filter_ast.get() });

// per batch
auto r   = _scan->read(chunk, stream, mr);
auto out = r.unapplied ? evaluate(*r.unapplied, r.table->view()) : std::move(r.table);
```

### Deletes from the public surface

`scan_filter_request` / `_result` / `_status`, `selection_mask`, `range_predicate`,
`pair_predicate`, `pair_compare_op`, `output_tier`, `decode_output_tier`,
`column_decode_directive`, `make_scan_filter_request`, all four `*_filter_directive` types,
`decode_selection`, `fused_scan_directives`, `build_fused_scan_directives`, all five plan
probes, `compact_scan_filter_output`, `probe_fused_scan_reservation`,
`decode_equality_pushdown`, `decode_range_pushdown`, `decode_membership_pushdown`,
`decode_pair_pushdown`, `numeric_range_extraction`, `pair_conjunct`, and the three scan-side
`extract_*` functions.

Retained as plumbing: an estimate call for cuCascade pre-reservation
(`decode_estimate estimate_scan_read(...)`), replacing `probe_fused_scan_reservation`.

## 3. Part B — the BOOL8 substitution becomes partial evaluation

Three cases; the caller writes the same two lines for all of them.

| Case | Engine | Result |
|---|---|---|
| A — equality folded into the row selection | dictionary route, ANDed into mask | conjunct absent from `unapplied`; column **elided**; no BOOL8 created at all |
| B — cannot answer it | not dict-rooted / uncompressed split / gate off | column materialized as strings; conjunct in `unapplied` |
| C — can answer cheaply, cannot fold into a selection | computes BOOL8 | `unapplied` = bare reference to it |

Case C is #1380 standing alone, and it forces one refinement:

> Every column reachable through `position` carries its **declared type**. The engine may append
> additional columns past those; they are reachable only from `unapplied`, which is the engine's
> own expression over its own output.

So the BOOL8 is an **appended intermediate**, not a substitution. Case A is strictly better than
#1380, which materialized the BOOL8, gathered it to survivors and re-ANDed it.

Plumbing through cuCascade: one `shared_ptr<compressed_scan>` attached to the representation
(replacing five pushdown vectors × two classes), and one result type carrying a value:

```cpp
class decoded_batch_representation final : public cucascade::gpu_table_representation {
 public:
  scan_read_result const& result() const noexcept;   // position, rows, unapplied
};
```

One `dynamic_cast` to fetch a struct, replacing two that each mean a boolean; `clone()` copies
a value instead of degrading.

## 4. Part C — internal layering

1. **Intent** — §2. Sirius-side, speaks `sirius::ast`.
2. **Analysis** — predicate → internal plan, against this chunk's compression plans, under one
   policy object. The three `extract_*` functions and `build_fused_scan_directives` merge here;
   `decode_route` lives here as an internal enum (P1, P2, P3 resolved). Sirius-side.
3. **Mechanism** — simpatico: waves, masks, CNT, compaction. Roughly today's
   `decompress_scan_filter`, but with exactly one caller, so it can be shaped for the kernels.
4. **Renderer / JIT** — §5.

One `fusion_policy` object owns the env gate and every threshold currently read in three places
(P8): `SIRIUS_EXP_FUSED_SCAN_FILTER`, `MAX_SEL` (0.35), `TIERB_MAX_SEL` (0.10),
`K4_MAX_SEL` (0.15), `MAX_MEMBER` (1), `DIAG`.

Internal route enum, replacing the four encodings:

```cpp
enum class decode_route : uint8_t { full, bitpack_mask, delta_mask, dict_codes, str_split };
struct column_decode_caps { decode_route compact_route; bool can_produce_mask, can_answer_equality; };
column_decode_caps probe_column(compressed_table const&, std::size_t column);
```

`compact_route == full` *is* "not compactable", so P2's invariant becomes unstateable.

## 5. Part D — JIT emitter and launcher refactor (done)

Renderer and launcher had described the same taxonomy twice — a flat
`DecodeVariant` enum plus a separate pair entry point on one side, seven flat
launcher functions on the other — so the two enumerations had to be kept in step
by hand. They are now one two-axis product.

```
Enumerator: all_rows | mask_bits | index_list          (how rows are walked)
Consumer:   write_column | ballot_range | ballot_pair
            | dict_gather | offsets_meta                (what happens per row)
```

|              | write_column | ballot_range | ballot_pair | dict_gather | offsets_meta |
|--------------|--------------|--------------|-------------|-------------|--------------|
| `all_rows`   | plain        | K1           | K1m2        | —           | —            |
| `mask_bits`  | K3           | —            | —           | K5          | K6·1         |
| `index_list` | K4           | —            | —           | *unbuilt*   | *unbuilt*    |

Everything derives from the pair: trailing parameters are
`enumerator_params ++ consumer_params`, the `out` slot's type follows from the
consumer alone (ballot consumers repurpose it for mask words), the launcher's
precondition contract is two switches over the axes, and `Walker::build`
dispatches on them. `shape_is_supported()` rejects invalid points at render.

What this replaced:

- **F1** — `emit_bitpack_mask_out` / `_mask_consume` were byte-identical to their
  `_generic` counterparts (`value_source` already routes a Bitpack leaf to
  `bitpack_value_source`). Deleted; their `RenderError` throws were unreachable.
- **F2** — the K3 / K5 / K6 emitters open-coded one survivor loop, differing only
  in a 1–4 line sink. Now `emit_mask_survivor_loop(sink)`.
- **F3** — `emit_delta_mask_consume` duplicated ~100 lines of
  `emit_delta_producer`'s striped path. Now a `DeltaStore` policy on the
  producer; delta is no longer a special case.
- **F4** — the pair launcher open-coded the whole bind/launch sequence and
  **omitted the per-chunk metadata bounds guard**, risking an out-of-bounds read
  that faults the CUDA context. Both paths now share `launch_rendered_spec`.
- **F5** — the renderer emitted trailing parameter DECLARATIONS from one switch
  while the launcher pushed ARGUMENTS from a second switch in another file, with
  `cuLaunchKernel`'s untyped `void**` between them: a mismatch was silent
  argument misalignment, not a compile error. `DecodeKernelSpec::trailing` now
  carries the list, emitted from the same table as the declaration text.
- **F6** — five bespoke precondition blocks became `VariantContract` plus one
  checker; diagnostics now name the specific missing field.

Method worth reusing: `renderer.cpp` depends only on `fused_tree.hpp` /
`render_util.hpp` / stdlib — no CUDA, cudf or rmm — so a dump-and-diff of
`render()` output compiles standalone with `g++ -std=c++20 -I include` in
seconds and needs no GPU. Every step above was verified byte-identical that way
before it was committed, which is what made threading a store policy through the
*plain* decode path safe to attempt.

### Safety net

`tests/test_render_signature_contract.cpp` (CTest: `render_signature_contract`)
asserts `declared params == buffers + 2 + trailing` for every shape × dtype ×
tree shape and the pair path, plus tag validity and uniqueness. It parses the
rendered signature rather than diffing a golden string, so kernel-*body* changes
do not churn it — only a break in the signature contract does. Mutation-tested:
injecting one undeclared parameter fails it across every shape. No GPU, no
NVRTC; milliseconds.

Entry-symbol suffixes live in `shape_symbol_suffix()`. They key the JIT cache,
so changing one forces a recompile of that kernel everywhere.

## 6. Sequencing

**Phase 0 — land #1391 on its measurements.** ~17k lines with real numbers; do not block it on
an API refactor.

**Phase D — codegen dedup. DONE** (§5). Independent of Parts A–C: it changed no public API and
no emitted code, so it neither helps nor blocks them. It does make every later kernel-shape
change cheap, which is the point.

**Phase 1 — value-carrying results. DONE** (`2b085d81`, `11cef624`).
Both marker representations collapsed into one `decoded_batch_representation` carrying
`decode_outcome{row_filtered, rule2_bailed, predicate_columns}`; the scan does one
`dynamic_cast` to fetch a struct instead of two to test identity, and the BOOL8 type sniff in
`build_filter_expression_for` is gone — the converter reports the substituted positions, which
reach `post_filter_and_project` through `filtered_table::predicate_columns`.

Not yet delivered from §2's design: `position` (column elision) and `unapplied` (the residual as
a predicate). Those need the facade, so they land in Phase 2.

**Caveat carried into Phase 2:** `predicate_columns` is populated only on the compression
converter path. A source that substitutes by some other route reports nothing and simply gets no
rewrite — safe, but it makes the field a *converter* fact rather than a universal one. Phase 2
generalises it: under `compressed_scan` the outcome is the engine's answer for every source, not
one converter's.

**Phase 2 — the facade. DONE** (`e1442655`, `5798182d`).

Delivered:

- `sirius::scan_decode_request` — one `column_entry` per column (equality set, bounds, join
  filters) replacing four parallel vectors; `sirius::compressed_scan` holds it, immutable, and the
  per-batch adjustments (`for_chunk`, `with_membership_probes`, `without_row_selection`) return a
  new one. One `shared_ptr<const compressed_scan>` on each representation (P7).
- `decode_compressed_chunk()` — the single entry point. It owns request assembly, the two-call
  compaction protocol and the fallback, so both converters are one call each and the ~190-line
  `try_decompress_scan_filter` is gone from the converter (P3/P4 hidden, not yet simplified).
- `op::analyze_scan_filters` — the three `extract_*` functions merged into one pass, run once per
  scan by the ingestible (`gpu_ingestible::filter_analysis()`) and only *mapped onto slots* by the
  scan manager, which drops the manager's duplicate range extraction. `scan_utils.hpp` no longer
  includes `codegen/selection/selection.hpp`.
- `probe_fused_scan_reservation` → `compressed_scan::forecast_compaction`.
- Deleted with no callers left: `column_decode_directive`, `make_scan_filter_request`,
  `decode_output_tier` (the converter's mirror of `codegen::output_tier`), `decode_pushdown.hpp`.

Two behaviour changes, both deliberate:

- whole-filter coverage now also requires every range to reach a decoded slot. A range mapping to
  no served column was silently dropped while the batch could still be tagged as needing no
  further filtering.
- the parquet ingestible's candidate extraction ran **twice** — a merge had duplicated the block
  (risk 6 below, manifested) — appending every candidate to the position list twice.

Not delivered from §2's design: `position` (column elision) and `unapplied` as an AST residual.
The residual is still expressed by rebuilding the DuckDB filter expression in the ingestible.
Both want the analysis to speak `sirius::ast`, which is Phase 3 work, not facade work.

**Open question 1 answered by construction.** The `range_predicate` / `pair_compare_op`
respelling is no longer a header-isolation problem: the carrier types are the facade's own
(`sirius::decode_range`, `sirius::column_compare_op`) and the conversion to `codegen::` happens at
exactly one point, inside `compressed_scan.cpp`, where the mechanism is invoked. No shared header,
no inverted dependency.

**Phase 3 — internals. DONE** (`52cc1b5a`, `ec38c489`, `7aaf3da8`, `660ce427`).
P1–P4, P7 and P8 are all closed. The merged analysis pass landed in Phase 2 with the facade.

*Not done, deliberately:* the §2 API's `position` (column elision) and `unapplied` (the residual
as a predicate). Both need the analysis to speak `sirius::ast` rather than rebuilding a DuckDB
expression, which is a change to the scan's filter representation, not to the decode's internals.

**No measurement.** Phase 2 and 3 were verified for correctness only — `[compression]` with the
gate on and off, plus all 14 simpatico tests. This box has no GPU budget for the SF1000 run, so
the "no measurable delta" clause on both phases is UNVERIFIED. Re-run before merging: the
mechanism's whole justification is 8.180 s → 6.918 s, and every one of these commits sits on the
per-batch path.

Phase 2 preceded Phase 3 for a reason: the facade is what buys freedom to change internals
without touching the scan. That freedom now exists — `decode_compressed_chunk` has exactly one
caller shape, and nothing outside `src/compression/` names a tier, a plan probe or a wave.

**Phase N — naming. DONE** (`5798182d`). The branch's shorthand (K1/K1m2/K3/K4/K5/K6, RULE 1 and
RULE 2, W1-W4, "iteration N", "STATUS-W2", "track A/B", "bail", "classic") is spelled out
everywhere it appeared: the range/pair ballot, the mask/index walk, the dictionary gather, the
masked str_split route, the static output-shape check, the selectivity ceiling, giving compaction
up, the ordinary decode. Identifiers followed (`tier_dict_k5` → `tier_dict_gather`, `tier_str_k6`
→ `tier_str_split`, `bailed_high_selectivity` → `declined_unselective`, `rule2_bailed` →
`selection_unprofitable`).

## 6b. Where things stand

Every problem in §1 is closed. What is left is not cleanup:

1. **Measure.** Nothing since Phase D has been timed. This is the blocker for merging.
2. **`unapplied` / `position`** (§2) — the residual as a predicate and column elision. Needs the
   analysis to produce `sirius::ast`; `src/include/op/scan/scan_filter_analysis.hpp` is where it
   would land.
3. **§7's capability work**, now cheap: a new consumer (`ballot_membership` first — membership is
   expensive today *because* it cannot work on packed bits) or the two unbuilt points of the
   enumerator × consumer product, each an emitter plus a `shape_is_supported` entry.

**Entry points, cold:** `src/compression/compressed_scan.hpp` is the whole scan↔decoder boundary;
`compressed_scan.cpp` holds the per-chunk narrowing (`plan_decode`) and request assembly;
`simpatico_codegen.cpp` holds the wave orchestration; `codegen/selection/decode_policy.hpp` holds
every threshold.

**Verification that works here:** `sirius_unittest [compression]` (36 cases — cases 9–11 run SQL
through the equality answer and check the aggregate, so they catch a broken substitution rather
than just a broken build) plus simpatico's own 14 ctest targets, which must be built explicitly:

```bash
pixi run ninja -C build/release test_compress_with_plan_roundtrip test_compressed_table_io \
  test_leaf_describe test_plan_tree test_fused_tree_build test_jit_kernel_cache_plain \
  test_jit_kernel_cache test_fused_operator_sweep test_operator_sweep \
  test_representation_contract test_bitpack_layout_contract test_multi_gpu_stream_affinity \
  test_masked_decode_variants test_render_signature_contract
cd build/release/simpatico_codegen && pixi run --manifest-path ../../../pixi.toml ctest
```

`ctest` without that build step reports "Not Run", not a failure.

## 7. Future directions this unlocks

**Missing kernel combinations become constructible.** With the op expressed as the enumerator ×
consumer product (§5), the 15-point space contains 7 implemented kernels; the buildable
remainder needs an emitter and a `shape_is_supported` entry, not a new variant tag, launcher,
capability probe and tier:

- **K4 × dict_gather**, **K4 × offsets_meta** (i.e. `index_list` × `dict_gather` / `offsets_meta`)
  — below the ~15% K4 crossover, dictionary and `str_split` strings are stuck on the mask walk
  purely because the emitter wasn't written. q12-class scans sit near 0.5% selectivity.
- **mask_bits × delta for K5/K6** — dict codes or string offsets behind a delta are `tier_b`
  today (`plan_supports_dict_selection_decode` requires bitpack codes;
  `plan_supports_str_selection_decode` requires a plain bitpack offsets child). `l_comment`'s
  `delta -> ans` offsets and `c_phone`'s `delta -> rle -> bitpack` are the concrete cases.

### Can the axes grow?

New *points* on the two axes, derived from what each axis can express rather than listed ad hoc.
Growth is additive: a consumer works under every enumerator it composes with, so one emitter
buys several kernels.

**Consumer — bounded by what it emits per row.**

| category | shipped | candidates |
|---|---|---|
| store the value | `write_column` | **`transform_store`** — store *f(v)*, not *v* |
| store a derived payload | `dict_gather`, `offsets_meta` | `lookup_gather` (generalised side-table expansion) |
| reduce to one bit | `ballot_range`, `ballot_pair` | **`ballot_membership`**, `ballot_expression`, `ballot_null` |
| reduce across rows | — | `count`, `aggregate` (SUM/MIN/MAX), `zone_map` (per-chunk min/max) |
| derive a value for elsewhere | — | `hash` (join hashes at decode time) |

- **`ballot_membership`** is the highest-value: membership is the most expensive conjunct today
  *because* it cannot work on packed bits, so wave 1 decodes the key column full width and the
  source cap is 1 (q8: 3 probes = +134 ms probe-side vs −48 ms compaction). A consumer sees the
  value in-register and never materialises the column.
- **`transform_store`** is the sleeper: projection pushdown into decode. q1/q6's
  `l_extendedprice * (1 - l_discount)` materialises two full columns to compute one; the
  arithmetic is free where the values already sit in registers.
- **`ballot_expression`** would make `ballot_range` and `ballot_pair` special cases rather than
  siblings, and is how predicate coverage widens (`LIKE 'x%'`, `year(d) = 1994`, disjunctions)
  without a new consumer per shape.

**Enumerator — bounded by how the row set is described.**

| | description | output slot | status |
|---|---|---|---|
| `all_rows` | implicit, all | identity | shipped |
| `mask_bits` | bitmap | survivor rank | shipped |
| `index_list` | **ascending** ids | position in list | shipped |
| `run_list` | `(start, len)` runs | running offset | candidate |
| `range_slice` | one `[lo, hi)` | offset from lo | candidate |
| `stride` | `(offset, step)` | index / step | candidate |
| `gather_list` | **arbitrary** ids | position in list | candidate |

- **`run_list`** beats `index_list` whenever survivors are clustered — what a range predicate on
  sorted or clustered data produces. `index_list` pays random access per row; a run coalesces.
- **`gather_list`** is a capability, not a tuning win: dropping `index_list`'s ascending
  requirement lets an arbitrary gather fuse into decode — a join's probe-side output or sort
  order materialised straight from compressed bytes, with no full-width intermediate. Same class
  of win as the fused scan-filter, applied to a different operator. **It is not free** — see
  below.
- **`range_slice`** is what LIMIT/OFFSET pushdown and splits that cut mid-chunk would need; a
  chunk is all-or-nothing today.

#### Why `gather_list` costs more than dropping a sort

K4's launch maps **one block per chunk** (`chunk_id = blockIdx.x`), and block *c* handles exactly
`row_indices[chunk_offsets[c] .. chunk_offsets[c+1])`. Two invariants make that work, both
supplied by the mask→indices wave that produces the list:

1. **Ids are partitioned by chunk.** The kernel computes the in-chunk position as
   `idxs[k] - chunk_start`. If an id in slice *c* belonged to another chunk, that subtraction
   lands outside `[0, 1024)` and the bitpack read addresses outside the chunk's packed region.
2. **Per-chunk scalars are loaded once per block.** `bitpack_value_source` emits the
   `chunk_min[c]` / `chunk_bits[c]` / `bp_offsets[c]` prelude in the block header, amortised over
   every element the block decodes. Bit-unpacking is parameterised *per chunk*, so this only
   works if a block's elements all share a chunk.

The output slot itself is fine — `(out + out_base)[k]` is the position in the list, which is
exactly gather semantics. It is the *partitioning* that breaks. So an arbitrary permutation needs
one of:

- a stable pre-pass partitioning the gather list by chunk, each entry carrying its destination
  index so the caller's ordering survives — an extra pass, and a scatter instead of a contiguous
  store; or
- a launch over the list instead of over chunks, with each element looking up its own chunk's
  scalars — which gives up both the per-block scalar amortisation and coalescing.

Neither is fatal, but it means `gather_list` is a new launch strategy, not just a new way to
enumerate — the one candidate here that does not drop into the existing frame unchanged.

**Count-only and existence.** Wave 1 + CNT already computes the exact survivor count and
`run_selection_cnt` already returns it; everything after is waste for `COUNT(*)`. Reads
compressed bytes, writes 1 bit/row, allocates no output column. Contract differs enough to
justify a separate entry point (`compressed_scan::count`, `optional<int64_t>`), because every
soundness escape hatch inverts:

- no conjunct dropping (today a dropped conjunct just clears `covers_whole_filter`)
- no membership cap truncation (`MAX_MEMBER` drops probes — sound for masking, fatal for counting)
- RULE 2 does not apply (no decode to stop paying for)
- **no inexact filters**: a Bloom probe over-keeps by construction. This argues for
  `bool exact` on the membership conjunct, turning `kind_rank`'s comment ("the set forms are
  exact; Bloom over-keeps") into a checkable field. Existence (`LIMIT 1`, anti-join probes) is
  the same wave with an early exit.

**Cross-kind conjunct ordering.** With one conjunct list and one `expected_keep` comparator, the
scarce mask slots go to the most selective conjunct regardless of kind — today the ordering
sort only runs *within* the membership vector, so a highly selective range cannot outrank a weak
Bloom.

**Selectivity feedback in the session.** `compressed_scan` sees every batch of a scan, so the
RULE-2 decision can become measured rather than latched-on-first-bail, and the K3/K4 pick can
adapt per scan instead of using a fixed 0.15.

**Wider predicate coverage without API churn.** Once `where` goes down whole and routes are
internal, adding `LIKE 'prefix%'` on dictionary keys, `IS NULL` masks, or disjunction support is
an analysis-layer change plus a consumer — no new public type, no new tier, no new probe, no
scan edit.

**Nullability.** Iteration 1 refuses null-masked columns throughout. A null model is one more
`Consumer` concern (ballot with validity, compacted validity output) rather than a new variant
per shape.

## 8. Risks / open questions

1. ~~**Where do `range_pred` / comparison ops live?**~~ **ANSWERED (Phase 2).** The carrier types
   are the facade's own and the conversion to `codegen::` happens at one point inside
   `compressed_scan.cpp`. `scan_utils.hpp` no longer includes the codegen header, so the isolation
   is real rather than aspirational — no shared header, no inverted dependency.
2. `unapplied` requires an AST node referencing a column by table position (Case C). Confirm
   `sirius::ast` bound references cover this.
3. cuCascade reservation happens before conversion, so `estimate_scan_read` still leaks *that*
   filtering may shrink the batch. Unavoidable; keep it a forecast, not a route list.
4. The membership probe closure must pin its device structure (`shared_ptr` capture) for the
   call's duration — preserve that contract when conjuncts move into the AST.
5. F4's fix changed the pair path's behaviour in one respect: it now enforces the metadata
   bounds guard. If any plan launches a pair kernel with under-sized per-chunk channels it now
   refuses instead of silently reading OOB. That is the intended outcome, but expect it to
   surface rather than stay quiet.
6. Git's auto-merge silently duplicated four definitions when merging `dev` (§0), because both
   sides had added the same construct at different offsets in one file — one of them in a file
   with no conflict at all. Only the compiler caught it. Expect this anywhere #1380's content
   exists on both sides of a future merge. **A fifth case was found in Phase 2 that the compiler
   could NOT catch:** the parquet ingestible's candidate-extraction block existed twice, so every
   candidate was appended to `_pushdown_primary_by_batch_position` twice. It compiled and behaved
   (the lookup breaks on the first match). Grep for repeated blocks, do not rely on the build.
