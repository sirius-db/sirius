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

**Still open, deliberately deferred:** they are environment variables, not Sirius parameters, so
they bypass the ~62 typed `SET` options in `sirius_extension.cpp` — invisible to
`duckdb_settings()`, not settable per session, and cached on first read, which is why some decode
paths are only reachable by a unit test rather than end to end. The obstacle is layering:
simpatico has no DuckDB dependency, so the values cannot be pulled from the setting registry where
they are read; they would have to be pushed down per query or passed into the decode call as a
policy value (the latter preferred — it makes policy a value like everything else, and removes the
cache-on-first-read testing problem). Left as env while the feature is experimental; revisit
before it ships.

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

**`unapplied` — DONE** (`a2b38ca3`, `175ae6a4`). The residual is a Sirius AST predicate now:
`decompose_table_filters` splits the filter into conjuncts at bind (and
`convert_table_filters_to_expression` is reimplemented on top of it, so the two cannot diverge),
each conjunct is lowered once, and `residual_filter::against` picks a form per conjunct per batch.
Nothing is rebuilt or converted on the batch path.

Three outcomes per conjunct, which is the rewrite rule §2 asked for:

| the decode | the residual |
|---|---|
| did nothing with it | keeps the comparison |
| ANSWERED it (BOOL8 delivered, no row dropped) | references the answer |
| APPLIED it (folded into the row selection) | drops it |

The third is new and needed `decode_outcome::predicates_enforced` — an answered equality is ANDed
into the mask before wave 2, so on an *applied* decode the surviving rows already satisfy it, and
the scan was re-ANDing a condition that could no longer be false. A scan whose whole filter is one
dictionary equality now skips the post-decode pass entirely; an empty residual means "already
filtered", never "no filter".

*Watch out:* the enforced path is unreachable at default thresholds for the sizes the end-to-end
tests use — they decline at 0.25 selectivity against the 0.10 full-route ceiling — so it is pinned
by `test/cpp/scan/test_residual_filter.cpp` at the decision, not end to end. Raising
`SIRIUS_EXP_FUSED_SCAN_TIERB_MAX_SEL` to 0.9 reaches it in the two dictionary pushdown tests and
their answers are unchanged.

*Not done:* `position` — true column ELISION, where an answered filter-only column is never
delivered at all. The decode still gathers its BOOL8 to the slot (1 B/row); skipping it means a
variable output arity, which every downstream position mapping would have to follow. The conjunct
side of elision — the column being unreferenced by the residual — is what landed here.

**Measurement — DONE.** Re-run on GB300 after Phases 2 and 3 and the reuse review: the numbers
match the original branch's. The refactor cost nothing, which is what "no measurable delta" asked
for. The merge blocker is cleared.

Still unverified on hardware: the multi-GPU stream-pool fix (`ed568684`) — that needs a
two-GPU box to run `test_multi_gpu_stream_affinity`'s device-switching half.

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

1. ~~**Measure.**~~ DONE on GB300; the numbers match.
2. **`unapplied` / `position`** (§2) — the residual as a predicate and column elision. Needs the
   analysis to produce `sirius::ast`; `src/include/op/scan/scan_filter_analysis.hpp` is where it
   would land.
3. **§7's capability work**, now cheap: a new consumer (`ballot_membership` first — membership is
   expensive today *because* it cannot work on packed bits) or the two unbuilt points of the
   enumerator × consumer product — but see §9: both were built and measured off-branch, and the
   axis turned out to be saturated.

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

## 6c. Reuse review

A sweep for functionality the branch reimplemented. Findings, and what was done:

- **Numeric range extraction — DONE** (`0f894f89`). The decode-side constant lowering had its own
  int128 floor/ceil, accumulator and payload switch alongside `sirius::numeric_range` and
  `ast::constant_numeric_range`, which are the same thing and older. Folded onto them; what stays
  local is restating a constant at the COLUMN's scale, which narrowing has no notion of.
- **Decode policy knobs — DEFERRED** (see P8). They are env vars, not Sirius parameters.
- **The four mask kernels — EVALUATED, NOT SWAPPED.** cudf has an apparent equivalent for each and
  none fits: these are in-place, chunk-segmented operations over a caller-owned arena with a
  padding invariant, and cudf's bitmask API is allocating and whole-column. The reasoning is
  recorded at the top of `selection_wave.cu` so it is not re-litigated. The evaluation did surface
  the real gap — none of the four had a direct test, and `combine_masks_and` was almost certainly
  never executed anywhere, since the decode calls it only with two or more sources and no test
  built such a request. Now covered (`160657e0`).
- **Stream pools — DONE** (`ed568684`), and it was a bug, not tidying. The two thread_locals were
  device-oblivious: `stream_pool::init` creates streams on whatever device is current at first use
  and never re-creates them, so a thread that decoded on device 0 and later worked on device 1
  would submit device-1 work onto device-0 streams. Both now go through
  `simpatico::thread_device_stream_pool`, keyed per (thread, device). **Unverified on hardware** —
  this box has one GPU, so `test_multi_gpu_stream_affinity` still self-skips.
- **Two TableFilterSet walkers — DONE, but not by merging** (`0171b1c7`). They are not duplicates:
  `collect_null_prune_predicates` exists BECAUSE `decompose_table_filters` drops IS_NOT_NULL, so
  one keeps what the other skips. What they did duplicate is the column bookkeeping underneath,
  which had drifted (an out-of-range index skipped in one and threw from `.at()` in the other).
  `resolve_filtered_column` states it once, with three outcomes — the third being where the callers
  legitimately differ: a conjunct that must be EVALUATED cannot reference an unmaterialized column,
  while one used only to PRUNE drops silently.
- **The orchestrator's "duplicate" validation — EXAMINED, KEPT.** The premise did not survive
  counting. Of the 29 refusal points in `try_decompress_fused`'s preconditions, 25 are structural
  (bounds, arity, chunk geometry, the source cap) and protect the kernels; only 4 re-derive plan
  capability, about 20 lines. Deleting those would remove the last check before a WRONG MASK — a
  range ballot rendered over a non-bitpack root reads the wrong bits and drops the wrong rows —
  in exchange for 20 lines, on the path with the thinnest coverage. They now say in the code that
  they are boundary assertions rather than a second opinion: since Phase 3 both sides call the same
  `probe_column`, they cannot disagree, and the caller's guarantee holds only through a chain of
  reasoning across several files.
- **Zone-map filter bounds — NOT a duplicate, a missed connection.** A
  `sirius_dynamic_zone_map_filter` holds `[min, max]`; a decode range holds decoded-domain bounds.
  A zone-map filter could feed the range ballot instead of a full-width membership probe, which
  would use a kernel that already exists.

## 7. What is done, and what is open

### Done

| | |
|---|---|
| **Phase D** — codegen dedup | `enumerator × consumer` product; 7 of 15 points built |
| **Phase 1** — value-carrying results | `decode_outcome` replaced two marker representation classes |
| **Phase 2** — the facade | `compressed_scan` / `scan_decode_request` / `decode_compressed_chunk`; one carrier replaced five setter pairs; three `extract_*` merged into `analyze_scan_filters` |
| **Phase 3** — internals | one `decode_route`, one `probe_column`, one source list, one finished table, one definition per policy knob |
| **`unapplied`** | the residual is a Sirius AST predicate built once at bind; three outcomes per conjunct (keep / reference / drop) |
| **Naming** | K#/RULE/W#/iteration-N shorthand replaced with what the code does |
| **Reuse review** | §6c — one fold, one deferral, one evaluated-and-declined, one bug found |
| **Measurement** | GB300, matches the original branch's numbers |

P1–P8 are all closed. What is NOT done from §2's design is `position` — true column elision, where an
answered filter-only column is never delivered. The conjunct side landed; skipping the 1 B/row
gather needs a variable output arity that every downstream position mapping would have to follow.

**Unverified:** the multi-GPU stream-pool fix (`ed568684`) needs a two-GPU box.

**Testing note carried out of this work:** five of the six reuse tasks touched code with no
coverage at all, found by checking rather than by a failure — the numeric-range lowering, the four
selection-wave kernels (`combine_masks_and` was never executed anywhere), the residual decision,
the filter-column resolution. Anything below that adds *policy* rather than a kernel lands where
coverage is still thin; write the test first.

### What the SF1000 diag log says

One instrumented run (22 queries x 2 iterations, gate on) settled more than the code reading did.
Anything proposed below should be checked against it first.

| | |
|---|---|
| filtered decodes planned | 928 — of which **920 APPLIED**, 8 not |
| given up as unselective | **6**. The selectivity ceilings are not a live constraint |
| source cap reached | **0**. `kMaxSelectionSources` is not a live constraint either |
| selectivity distribution | 0.004% – 15% for nearly every batch; one cluster at 52.6% (the q1 shape, which proceeds only because a dictionary output exempts it from the ceiling) |
| row-filtered tag | 254 true / 674 false, and coverage was the sole gate |

Two things follow directly. **Almost every batch runs below the 0.15 index-walk crossover**, so a
dictionary or `str_split` column on the mask walk is on the wrong enumeration nearly always — that
is direction 5's justification, and it is a measured one. And **the 674 untagged batches were
untagged only because of `has_external_selection`** — `covers_whole_filter=false` never once
occurred without an external source. That is a REAL DEFECT, still present on this branch: the
tag is cleared by a term that has nothing to do with coverage. It was fixed and measured
off-branch — see §9.1, which is where the fix should be re-applied from.

Method note: three directions (1, 3, 4) were proposed from reading the code and all three turned
out to have no reachable customer. The ones that survived came from a profile or from this log.
Check first.

**What this log CANNOT say.** It reports the orchestrator's per-batch enumeration REQUEST, and
prints a column's route as a bare enum (`decode_route` has no name function), so no run of this
branch can answer "did this column actually take the index walk" — every index-walking site may
silently keep the mask walk, and one batch mixes routes. Anyone measuring the enumeration axis
needs the per-column instrumentation described in §9.4 first; without it, a claim about which
enumerator ran is an inference, and one such inference was wrong (§9.3).

### Check what actually reaches the scan

Three capabilities have now died to one mechanism: **DuckDB rewrites the predicate before the scan
sees it.** A prefix LIKE arrives as two ConstantFilters (direction 3). A column-vs-column pair —
q12's `l_commitdate < l_receiptdate` — is folded by the FilterCombiner into constant hulls, which
is why the pair ballot was landed dark-but-tested and then DELETED here: the kernel, its launcher,
the directives, the harvester and the plumbing were ~450 lines that no query could reach, and
wiring the harvester up would have harvested nothing. (Recoverable from history if a workload
turns up whose predicate the FilterCombiner cannot fold.)

So before proposing anything predicate-shaped: EXPLAIN the query and look at what the scan is
handed, not at what the SQL says. Two of these were proposed from query text and one from code
structure; all three were dark.

### Open, cheapest first

Each entry: where it lands in the code, and which TPC-H queries it touches.

**1. Zone-map join filters → the range ballot. TRIED, MEASURED, NOT KEPT** (§9.2).
Measured on GB300: inert by default and a pessimization when forced on. Publication is off by
default (`enable_dynamic_zone_map_filter`) because the filter only helps when build keys correlate
with the filter column and TPC-H keys are scattered — 296 decode-time attaches across q3/4/5/9/12/21
logged `ranges=0`. Forced on, consuming the bounds cost **+131 ms**, concentrated in the
DYNAMIC_FILTER-heavy queries (q17 +30, q8 +29, q20 +27, q7 +17, q21 +13, q19 +9).

Two mechanisms, and the obvious one is NOT among them. A range cannot displace a probe:
`order_key()` sorts every non-membership source ahead of all probes, and the membership cap
truncates only the membership vector. What actually happens is

  (a) exceeding `kMaxSelectionSources` DECLINES the whole filtered decode rather than truncating,
      so one added source can cost a batch its entire compaction — which fits the source-heavy
      queries; and
  (b) below the cap, a zone map over scattered keys spans nearly the whole used domain, so the
      range costs a full-width ballot pass and rejects almost nothing.

Admitting it profitably needs a selectivity signal the decode does not have. The one exact test —
does the published `[min, max]` exclude any of THIS chunk's values — needs device-resident
per-chunk metadata, and reading it to decide whether to launch a kernel is its own cost. **The
publisher is better placed:** it already gates on coverage, and Sirius has probe-side statistics
(`pinned_chunk_stats.hpp`), so it could publish an estimate of the fraction a zone map excludes and
let the decode admit only ranges above a threshold. Revisit there, not here.

**2. Disjunction → a range hull. TRIED, MEASURED FLAT, NOT KEPT** (§9.2). Analysis only.
`fold_numeric_conjunct` returns no bounds for
`CONJUNCTION_OR`; a hull (min of los, max of his) over-approximates and so cannot drop a surviving
row. Coverage stays false.
*TPC-H:* q19 — a three-way OR that today yields no range at all; a hull gives `l_quantity` 1..30 and
`p_size` 1..15. Also q16.

**3. `LIKE 'prefix%'` off dictionary keys. INVESTIGATED, NOT BUILT — no customer.**
DuckDB already rewrites a prefix LIKE before the scan sees it: `p_type LIKE 'PROMO%'` arrives as
`p_type >= 'PROMO' AND p_type < 'PROMP'`, two ordinary ConstantFilters (verified by EXPLAIN). So
there is no LIKE shape to recognise, and the real design would be larger than described — let a
dictionary answer a string RANGE rather than only an equality set, which means extending
`simpatico::decode_predicate` and its key comparison, not just the analysis.

The TPC-H customers claimed here were wrong, from reading query text rather than checking where the
predicate sits:
  - q14's `p_type like 'PROMO%'` is inside the SELECT's CASE, not the WHERE — never pushed.
  - q16's `p_type not like 'MEDIUM PLATED%'` is negated, so not a range.
  - q20's `p_name like 'antique%'` is genuine and is the ONLY one — on a high-cardinality column
    that suits a dictionary poorly, in a table `bench/sf1000-repro/plans/` does not even carry a
    plan for.

Note the general fact for anything similar: a VARCHAR column's comparison filters DO reach the
filter set, and nothing extracts them today. If a workload with real string-range predicates turns
up, the dictionary can answer them the same way it answers equality.

**4. Truncate at the source cap instead of declining. MEASURED, DARK.**
`declined on shape` fired **0 times** across TPC-H SF1000 (22 queries x 2 iterations, gate on):
928 filtered decodes were planned and none exceeded `kMaxSelectionSources`. The cap is not a live
constraint on this workload, so keeping a prefix instead of giving up has nothing to improve.
Justifying it needs a query shape with more concurrent row-selecting sources than TPC-H produces.

It would start firing in a configuration with zone-map publication enabled, since those add
sources — but that configuration is itself a pessimization (see 1), so this stays dark either way.
Settled in one run by a decline line that reported the cap and the competing kinds — that
instrumentation is not on this branch (§9.4); re-add it before re-opening this.

**5. `index_list × dict_gather` and `index_list × offsets_meta`. BUILT, MEASURED, NOT KEPT** —
and the measurement says **do not rebuild it for its own sake**. Full account in §9.3; the short
version is that the axis was already saturated by `bitpack_mask`, these two points served 3.6% of
column walks, and the largest remaining group (delta) cannot use them at all.

**6. `ballot_membership`. NOT one direction — three, and only one of them is cheap.**
Today `membership_probe_fn` is a `std::function` returning a full-width BOOL8 column, so a probe
costs a full key decode + a full BOOL8 + the mask adapter. In-register testing needs the filter's
device structure as kernel parameters, and what that means depends entirely on which filter kind
published it — the rendered kernel is plain CUDA compiled by NVRTC against simpatico's own headers
only:

| kind | structure | renderable? |
|---|---|---|
| `sirius_dynamic_small_in_list_filter` (rank 0) | raw snapshot of **<= 12** INT32/INT64 needles | **Yes** — a pointer, a count and a branchless 12-iteration compare. No external header, no layout coupling. This is a contained emitter, direction-5 sized |
| `sirius_dynamic_in_list_filter` (rank 1) | `cuco::static_set`, PIMPL'd into a `.cu` | Only by reimplementing cuco's bucket layout, probing scheme, hasher and sentinel in EMITTED source — coupling a rendered kernel to a dependency's internals |
| `sirius_dynamic_bloom_filter` (rank 2) | `cuco::bloom_filter` + policy | Same, plus the policy must match the build side exactly — and `bench/sf1000-repro` documents that policy silently changing (the Arrow 128 MiB cap, the fast-range block index) |

So the doc's old framing — "one new Consumer, emitter, launcher and a directive carrying the view"
— holds for rank 0 and understates ranks 1-2 by a lot. `sirius_device_replicable` publishes
replicas, but a replica is not a device VIEW; two of the three kinds keep their probe behind a
PIMPL precisely so cuco device code stays in one `.cu`.

**Check first, and it needs no new code:** the `[decode-filter] wave-1 sources` line already prints
`join(cN, rank R)` for every membership source carried into a decode. Rank 0 dominant ⇒ build the
small-in-list ballot and stop. Ranks 1-2 dominant ⇒ rank 0 is another direction-4 (measured dark),
and the real work is a device-view abstraction over cuco structures, which is a project, not an
emitter. Three directions have already died from being proposed on structure rather than counted.

If it is built, `decode_max_membership_sources` (currently 1) should rise with it — that cap exists
only because probes run full width.
*TPC-H:* q8 (measured: 3 probes = +134 ms probe-side vs −48 ms compaction), q21 (the suppkey
in-list vs orders Bloom cap-ordering case), q3, q5, q9, q10, q17, q20 — but which of those the
rank-0 emitter can serve is exactly what the rank distribution decides.

**7. Count-only / existence.** `compressed_scan::count()` is the easy half — wave 1 + CNT already
computes the exact count. The hard half is upstream and is a PLAN-shape decision: something must
recognise "this aggregate is COUNT(*) and no column's values are used" and push it to the scan
(`sirius_plan_aggregate` / the physical plan generator, not `src/compression/`). Every soundness
hatch inverts: no conjunct dropping, no cap truncation, and no inexact filters — a Bloom over-keeps
by construction, which turns `bool exact` on the membership conjunct from a comment into a
checkable field.
*TPC-H:* weak — q4, q13, q21, q22 all still need other columns for their group-by. Do it for the
capability, not for these numbers.

**8. `transform_store`.** Highest ceiling, lowest readiness. Needs the projection expression to
reach the scan (projections sit above it today) and a lowering from `sirius::ast` arithmetic into
JIT CUDA source — a new path parallel to the cuDF-AST translator, since the renderer emits text and
nothing turns a Sirius expression into a kernel body. Plus decimal overflow reasoning in-register.
*TPC-H:* q6 is the clean case. q1 and q14 have the shape, but q1 needs both columns for other
aggregates anyway, so nothing is saved there.

**9. `run_list` enumerator.** A third `Enumerator` point plus a wave to build the run list. Drops
into the product cleanly, unlike `gather_list` — see below for why that one is a separate project.
*TPC-H:* clustered survivors — q3, q4, q5, q12, q21, where date filters correlate with orderkey
order. Speculative until measured: run-length statistics over the existing mask are a diagnostic
away.

**Not worth it for TPC-H:** `IS NULL` masks (no nulls in the dataset), `ballot_expression` beyond
the LIKE case, and `gather_list`.

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

## 9. The follow-up campaign: what was tried after the refactor, and what it was worth

Everything in this section was built on top of this branch, measured on TPC-H SF1000, and then
**reset off it**. The branch is deliberately the refactor plus its measurement; these were
optimization attempts on the open directions, and mixing them in was hiding what the refactor
itself costs. Nothing here is in the tree — no commit ids, because the reset orphaned them.
Re-apply from the descriptions, and re-measure: two of the six are worth taking, three are
measured dead, and one is instrumentation you want before touching this area at all.

### 9.0 The suite numbers everything below is measured against

**Base caveat:** every number in this section was taken with the branch based on the dev commit it
was developed against. The branch has since been rebuilt onto a later dev, so these describe the
work, not this exact tree — re-measure before quoting them, especially the gate-off regression,
which is the one number that could plausibly be dev's rather than ours.

| arm | suite | vs dev |
|---|---|---|
| `dev` baseline | 8.704 s | — |
| this branch, **gate off** | 8.900 / 8.911 s | **+2.2 / +2.4%** |
| this branch, gate on | 7.626 s | −12.4% |
| + coverage fix and index walk (§9.1, §9.3) | 7.610 / 7.615 s | −12.6% |
| + the same with per-column diag enabled | 7.596 s | −12.7% |

Two readings, and the second matters more than anything else in this section:

- The gate-on win is real and reproducible, and the follow-ups added ~13-30 ms to it.
- **Gate OFF, the branch is 2.2-2.4% SLOWER than `dev`** — ~200 ms, measured twice (8.911 before the
  follow-ups, 8.900 after), so not noise. **EXPLAINED, and not a defect in the fused path**, which
  is inert when the gate is off exactly as designed. The cause is a PLAN-SELECTION consequence:
  this branch lifted the dictionary compressor's historic `1 << 28` row cap, replacing it with the
  HyperLogLog distinct-fraction gate that addresses the real hazard. Columns in very wide pins
  (q12's ~276M-row lineitem pin, the 600-900M-row orders pins) used to be forced to raw by that cap
  and now dictionary-encode instead. With the gate ON that is a large win — the dictionary route is
  2.1-2.6x at ALL selectivities, which is why it is exempt from the selectivity ceiling. With the
  gate OFF nothing exploits the codes, so those columns just pay a dictionary decode where identity
  would have passed the bytes through.

  So the ~200 ms is the price of making the dictionary route available, charged on the arm that
  cannot use it. Three ways to settle it, in increasing order of how much they claim to know:
  ship with the gate on and the question disappears; teach plan selection that a dictionary's value
  is conditional on a consumer, so a wide pin with no consumer keeps identity; or leave the cap
  lifted and accept the default-path cost while the feature is experimental. What should NOT happen
  is restoring `1 << 28` as a "sanity bound" — it never was one (see the comment in
  `dictionary_compressor.cu`), and the cardinality hazard it was credited with is handled by the
  HLL gate.

### 9.1 Row-filtered tag cleared by an unrelated term — REAL FIX, RE-APPLY

The defect is described in §8's log table: `row_filtered` was ANDed with `!has_external_selection`,
so a batch whose decode carried the WHOLE filter was still tagged "needs post-decode filtering"
merely because an extra mask source existed. An extra source only removes more rows; it cannot make
a surviving row stop satisfying the filter. Coverage is the only correct gate.

Measured: tagged batches **254 → 404** (+150 of 928), each of which then skips a post-decode filter
evaluation and regains the zero-copy steal, at roughly 0.1 ms/batch. The corrected trace shows
404 `coverage=true/tagged=true` and 524 `coverage=false/tagged=false`, with **no line where
coverage is true and the tag is false** — the term is gone.

Worth knowing before re-applying: the analysis that motivated it implied all 928 batches would
become tagged. They do not. The remaining 524 are untagged for a legitimate and different reason —
genuine partial coverage (a range that reaches no decoded slot, a dropped conjunct or pair) — so
the fix is complete even though the number is not what the reasoning predicted.

This is the single highest value/cost item recovered from the campaign: a few lines, and on its own
arithmetic (150 × 0.1 ms) it accounts for essentially the whole 13 ms that §9.3 was measured
alongside.

### 9.2 Two analysis-side directions that measured flat or negative — DO NOT REBUILD

**Zone-map join filters into the range ballot (direction 1).** Inert by default and a pessimization
when forced on: **+131 ms**, concentrated in the DYNAMIC_FILTER-heavy queries (q17 +30, q8 +29,
q20 +27, q7 +17, q21 +13, q19 +9). Publication is off by default because a zone map only helps when
build keys correlate with the filter column, and TPC-H keys are scattered — 296 decode-time attaches
logged `ranges=0`. Direction 1 above keeps the full mechanism analysis, including the correction
that a range never displaces a probe and that the profitable version belongs on the PUBLISHER side.

**Disjunction → a range hull (direction 2).** Built (a hull over `CONJUNCTION_OR` bounds, coverage
left false, plus its own test) and measured **flat** — no query moved outside noise. The analysis
is sound and cannot drop a surviving row; there is simply nothing to win, because the queries it
targets (q19, q16) spend their time elsewhere. Rebuild only if a workload shows a disjunction over
a column whose decode is the bottleneck.

### 9.3 The index walk for the dictionary and string decodes — BUILT, SATURATED, LOW VALUE

Both unbuilt points of the enumerator × consumer product were built: `index_list × dict_gather` and
`index_list × offsets_meta`, taking the product to 9 of 15. The design held up well — the survivor
walk splits into a prologue and a loop, so each consumer states its per-row sink ONCE and runs under
either compacting enumerator, and the existing `index_consume` folded into the same seam (it had
been a third hand-written copy of the walk). Both new points require a Bitpack leaf root; a staged
root (Delta, RLE cascade) reconstructs the chunk regardless, so render rejects it, the launcher
returns false and the caller retries the mask walk. That retry matters for `str_split`, whose
decline path otherwise re-runs the whole batch unfiltered.

Then the attribution, over 5,850 column walks, and it is the reason this is not worth rebuilding
on its own:

| route | enumerator | walks | why |
|---|---|---|---|
| `bitpack_mask` | index list | 4,053 | below the crossover; ALREADY SHIPPED before this work |
| `delta_mask` | mask bits | 828 | **structural** — a prefix sum cannot row-skip, so `delta_mask × index_list` is not a gap in the product, it is impossible |
| `bitpack_mask` | mask bits | 540 | above the crossover; the mask walk is the right pick |
| `dict_codes` | mask bits | 216 | above the crossover — a dictionary output exempts its batch from the selectivity ceiling, so these are the high-selectivity q1-shaped batches |
| `str_split` | index list | 186 | NEW |
| `dict_codes` | index list | 27 | NEW |

Zero decline lines, so no requested index walk was ever refused: every mask-bits row is either the
crossover deciding correctly or delta's prefix sum. **The two new points serve 213 of 5,850 walks
(3.6%)**, and they were measured together with §9.1, whose mechanism accounts for the delta on its
own arithmetic — so this work's separate contribution was never isolated and sits inside noise.

The general dictionary route was a genuine gap found while reading the trace, and is worth
recovering independently of the two new kernels: dual delivery, variable-width and compressed keys
all decode their CODES through the value path, where the enumeration pick was hard-coded to
`bitpack_mask`. That region is a bitpack leaf (the site's own op check proves it), so it should
consult a route predicate like every other pick. On SF1000 that is the path most dictionary columns
take.

Two lessons worth more than the code:

- **NVRTC caught what review did not.** The index loop's variable collided with a local the
  dictionary sink declares. A sink is written against `i`/`rank`/`out_base` and declares its own
  names, so an enumerator's loop variable must stay out of the way — one-letter names in emitted
  source are a hazard the C++ side never sees.
- **The in-repo tests cannot cover either new point end to end.** The `[compression]` cases have one
  dictionary batch and it ANSWERS A PREDICATE, so it never decodes codes compacted; no case produces
  a `str_split` route at all. The kernels were covered by `test_masked_decode_variants` (index walk
  vs the mask walk's own output, plus the staged-root rejection) and `render_signature_contract`.
  Closing the gap needs a case that projects a dictionary or split-string column while filtering on
  another — the KERNELS are testable in-repo, the PICK is not.

### 9.4 The instrumentation — RE-APPLY FIRST, it is what made the rest checkable

Three diagnostic gaps were closed off-branch. None is on this branch, and each one is the reason a
question above could be answered at all:

- **`decode_route` has no name function**, so the plan trace prints a column's route as a bare enum
  — which reads as no information. A `decode_route_name` next to the enum, used by every line that
  mentions a route, makes the logs greppable and joinable.
- **No line reports the enumerator a column ACTUALLY used.** The batch line reports the
  orchestrator's request; every index-walking site may silently keep the mask walk. A per-column
  line (route, enumerator, and which of the three refusal reasons applied when a requested index
  walk was declined) is what produced §9.3's table — and it immediately falsified a claim I had made
  from the batch line, that the `[compression]` cases exercised the dictionary index walk end to
  end. They do not.
- **The source-cap decline line** (which cap, which competing source kinds) is what settled
  direction 4 as dark in a single run.

Cost is trivial and they pay for themselves the first time someone measures this area. The
per-column line goes to stderr with the other simpatico diag; the route-name change touches the
sirius-side log lines, which is where the coverage counts already live.

### 9.5 Direction 6 was scoped but not started

No code. The finding is in direction 6 above: `ballot_membership` is three problems wearing one
name, only the small-in-list one (≤ 12 raw needles) is renderable in an NVRTC plain-CUDA kernel, and
the rank distribution in an existing log line decides whether that one has a customer. Check before
building.
