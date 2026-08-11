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

**P1 — Four encodings of "how does this column decode."**
`codegen::output_tier` (5 values, `selection.hpp:112`) → `sirius::decode_output_tier` (5 values,
`compression_converters.hpp:346`) via `to_shared_tier`, overridden by a parallel
`vector<uint8_t> compact_capable`, re-expressed downstream as `decode_selection`'s four bools
(`plan_interpreter.hpp:40-79`). The converter comment at `compression_converters.cpp:319-322`
("not via `make_scan_filter_request`, whose `column_decode_directive` collapses tiers to a
boolean") is this fighting itself.

**P2 — Five capability probes** walking the same tree
(`plan_supports_predicate_decode`, `…selection_decode`, `…dict_selection_decode`,
`…str_selection_decode`, `plan_selection_tier`), with the invariant "umbrella true ⇔ classifier
!= tier_b" enforced only by comment.

**P3 — Four parallel conjunct vectors** (`scan_filter_request`, `selection.hpp:199-210`), each
with its own cap arithmetic, ordering rule and coverage rule, all hand-written in
`try_decompress_scan_filter` (`compression_converters.cpp:252-439`, ~190 lines).

**P4 — Ragged two-call protocol.** `decompress_scan_filter` returns
`vector<unique_ptr<column>>` where TierA columns are survivor-sized and TierB are full-width;
only `compact_scan_filter_output` reconciles them. Nothing in the types enforces the second call.

**P5 — Type sniffing.** `parquet_gpu_ingestible.cpp:886` infers the BOOL8 substitution from
`batch.column(pos).type().id() == BOOL8`, then rebuilds the filter expression and throws if the
rebuild degenerates (`:908-912`). Unambiguous today only because candidates are VARCHAR-only;
extend the pushdown to numeric or boolean equality and a genuine BOOL8 column becomes
indistinguishable.

**P6 — RTTI as a signalling channel.** `row_filtered_gpu_table_representation` and
`rule2_bailed_gpu_table_representation` each carry one bit via dynamic type;
`clone()` "intentionally degrades". Safe only because two call sites are adjacent on one thread.

**P7 — Pushdown carriers duplicated.** Five setter/getter pairs on both representation classes,
with `clone()` obliged to copy each (#1380 already had to patch `clone()` for one field).

**P8 — Policy in three places.** RULE 1 in the converter, RULE 2 inside simpatico
(`simpatico_codegen.cpp:348-430`), re-derived a third time in `probe_fused_scan_reservation`.

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

**Phase 1 — value-carrying results.** Replace P5's type sniff and P6's RTTI tags with one
result type carrying `position` / `rows` / `unapplied`.
*Accept:* `build_filter_expression_for`, `boolean_substituted_primary_indices`, both marker
classes and the two `dynamic_cast`s are gone; TPC-H results unchanged.
**Now the highest-value item**: since #1371 put #1380's pushdown on `dev`, P5's hazard is in
shipped mainline code, not an unmerged PR.

**Phase 2 — the facade.** Introduce `compressed_scan` over the *existing*
`build_fused_scan_directives` + `decompress_scan_filter`, moving the three `extract_*` functions
behind it. No kernel changes.
*Accept:* every name in §2's delete list is gone from `src/include/` and
`src/compression/*.hpp`; the scan-side call site is the two lines in §2; no measurable delta.

**Phase 3 — internals.** One route enum, one `probe_column`, merged analysis pass, one
`fusion_policy`, `compact_scan_filter_output` folded into the entry point.
*Accept:* P1–P4, P7, P8 closed; `try_decompress_scan_filter`'s ~190 lines substantially reduced.

Phase 2 must precede Phase 3: the facade is what buys freedom to change internals without
touching the scan. Today every internal change is scan-visible.

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

Yes, and that is the main reason to hold the shape open rather than freeze the seven points.
Growth on either axis is **additive**: a new consumer works under every enumerator that composes
with it, so one emitter buys several kernels.

**Enumerator — how rows are walked.** Candidates, roughly by expected value:

- **`run_list`** — survivors as `(start, length)` runs rather than individual ids. Sits between
  `mask_bits` and `index_list`: strictly better than `index_list` when survivors are *clustered*,
  which is exactly what a range predicate on clustered or sorted data produces. `index_list`
  pays random access per row; a run keeps coalescing.
- **`range_slice`** — a contiguous `[lo, hi)` sub-range of the chunk. A chunk is all-or-nothing
  today, so this is what LIMIT/OFFSET pushdown and split boundaries that cut mid-chunk would
  need.
- **`stride`** — every Nth row, for `TABLESAMPLE` / approximate aggregation.

**Consumer — what happens per row.** This axis has more headroom, because the decode already
holds the value in a register and currently throws that away:

- **`ballot_membership`** — probe a device hash set / Bloom per row and ballot. This directly
  fixes the asymmetry that makes membership the most expensive conjunct today: it is expensive
  *because* it cannot work on packed bits, so wave 1 decodes the key column full width and the
  cap is 1 (q8: 3 probes = +134 ms probe-side vs −48 ms compaction). A consumer sees decoded
  values in-register and never materialises the column. Probably the highest-value future
  consumer.
- **`aggregate`** — accumulate SUM / MIN / MAX during decode instead of writing a column.
  `SELECT SUM(l_extendedprice) WHERE …` currently materialises a column purely to reduce it.
- **`count`** — popcount the ballot per block with no mask buffer at all. §7's count-only work is
  `all_rows × count` once this exists; the mask words become unnecessary rather than merely
  unused.
- **`hash`** — compute join hash values at decode time.
- **`null_ballot`** — `IS [NOT] NULL` masks, once a null model exists.

**Constraints worth recording**, so the product is not mistaken for fully orthogonal:

- **Ballot consumers require a non-compacting enumerator.** The mask layout depends on ballot
  lanes lining up with mask bits (one `uint32` per 32 *consecutive* rows). Under `mask_bits` or
  `index_list` the surviving rows are no longer contiguous, so the ballot cannot address the
  output word. That is why `mask_bits × ballot_*` is unsupported rather than merely unbuilt —
  refining a mask in a second pass would need a different output layout, not just an emitter.
- **Arity currently hides inside the consumer** (`ballot_range` = 1 tree, `ballot_pair` = 2).
  That is fine at 1–2. If a 3-ary consumer ever appears, arity should become explicit rather
  than spawning `ballot_triple` — treating arity as a fork rather than a parameter is precisely
  what produced F4's 165-line duplicate.
- **Some consumers constrain the plan shape, not the axis** — `dict_gather` needs a bitpack code
  leaf, `offsets_meta` needs a terminal chars channel. Those stay per-consumer render checks.

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

1. **Where do `range_pred` / comparison ops live?** `decode_pushdown.hpp:50-62` respells `lo`/`hi`
   to stay simpatico-free, but `src/include/op/scan/scan_utils.hpp` already includes
   `codegen/selection/selection.hpp` for `range_predicate` and `pair_compare_op`. The isolation
   is already broken and the duplication buys nothing. Either formalize a single shared header,
   or invert the dependency with a small `sirius/decode_types.hpp` that simpatico includes.
   **Needs a decision before Phase 2.**
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
   exists on both sides of a future merge.
