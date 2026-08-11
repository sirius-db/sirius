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

## 5. Part D — JIT emitter and launcher refactor

Renderer and launcher are covered together because they turn out to be **one taxonomy expressed
twice** — see "The two refactors are one refactor" below. Read that first if you only read one
part of this section.

`renderer.cpp` is 1700 lines (+649 in this PR) with seven variant emitters plus a separate pair
builder. Findings:

**F1 — two emitters are exact duplicates.**
`value_source` (`renderer.cpp:1587-1590`) routes a Bitpack leaf straight to
`bitpack_value_source`, so for a Bitpack root the "tuned" and "generic" emitters produce
character-identical bodies:

- `emit_bitpack_mask_out` (554-578) ≡ `emit_generic_mask_out` (939-959)
- `emit_bitpack_mask_consume` (614-639) ≡ `emit_generic_mask_consume` (835-856)

Removing them costs 53 lines and no behaviour. Their `RenderError` throws are **unreachable**:
`emit_bitpack_mask_out` is only called when `tree.op == Bitpack` yet throws if
`node.op != Bitpack`. The one reachable case (Bitpack *with* children) is rejected by
`value_source`'s own "Bitpack must be a leaf" anyway; only the message text differs.

The emitted comment text does differ (`Bitpack`→`BITPACK` from `op_kind_name`,
`(K1)`→`(K1-generic)`), which alters the JIT cache key — a one-time NVRTC recompile per
(shape, dtype, variant). Making the diff empty would mean hardcoding `Bitpack` in an emitter
that serves every shape, which would be a lie; the blip is accepted.

The "tuned closed-form path" comments at `build()` lines 150 and 161 are wrong and should go
with the emitters.

**F2 — one skeleton, four copies.** The mask-consuming emitters share
`emit_selection_stage()` + the same strided loop + the same rank expression, differing only in a
2–4 line sink: K3 stores, K5 copies `key_width` key bytes, K6 writes `{src_offset, length}`.
K4 (`emit_bitpack_index_consume`, 801-824) is the *same sink* as K3 with a different row
enumerator.

**F3 — the delta emitter duplicates ~100 lines.** `emit_delta_mask_consume` (651-750) copies
`emit_delta_producer`'s striped branch (1086-1147) — the same striped load, `cub::BlockScan` /
`BlockExchange` union, three transposes, unsigned-counterpart reconstruction — changing only the
final store. The header comment states why: *"so the plain rendered source stays byte-identical."*

### Refactor: two axes instead of seven variants

```cpp
struct Enumerator { /* all_rows | mask_bits | index_list */ };
struct Consumer   { /* write_column | ballot(pred) | dict_gather | offsets_meta */ };
```

Two axes, not three: the predicate is a **parameter of the `ballot` consumer**, and its arity
(1 for a range, 2 for a pair) is what fixes `DecodeInputs.trees.size()`. Treating arity as an
axis — or worse, as a fork — is exactly what produced F4's 165-line duplicate.

| Kernel | Enumerator | Consumer |
|---|---|---|
| plain | `all_rows` | `write_column` |
| K1 / K1-generic | `all_rows` | `ballot(range)` |
| K1m2 (pair) | `all_rows` | `ballot(pair)` |
| K3 / K3-delta / K3-generic | `mask_bits` | `write_column` |
| K4 | `index_list` | `write_column` |
| K5 | `mask_bits` | `dict_gather` |
| K6 | `mask_bits` | `offsets_meta` |

Seven small pieces replace seven emitters plus `build_pair`/`finalize_pair`. Estimated ~300
lines of emitter code → ~120.

Cascades:

- The trailing-parameter switch (343-387) becomes `enumerator.params() + consumer.params()`.
- The entry-symbol suffix chain (330-333) becomes the same concatenation.
- **Delta stops being special**: thread `Consumer` through `emit_delta_producer`; plain passes
  `write_column`, K3-delta passes `write_column` behind `mask_bits`. F3's copy disappears and
  future delta work is done once.

### Launcher findings (`codegen_runtime.cpp`, 1903 lines)

`masked_launch.hpp` declares seven entry points; an eighth (`launch_decode_fused_tree`, plain
decode) lives in `codegen_bridge.hpp`.

| Launcher | Kernel | Does | Body |
|---|---|---|---|
| `…_mask_out` (729) | K1 | decode + range compare → mask words, no column out | 34 |
| `…_mask_consume` (765) | K3 | decode consuming mask+offsets → compacted column | 27 |
| `…_index_consume` (794) | K4 | decode driven by survivor row-id list → compacted | 28 |
| `…_mask_dict_gather` (1099) | K5 | decode dict codes, gather fixed-width key bytes → compacted chars | 33 |
| `…_str_split_meta` (824) | K6·1 | masked offsets → survivor `{src_offset, length}` | 33 |
| `launch_masked_char_copy` (884) | K6·2 | fixed kernel, byte-gather survivor chars | 46 |
| `…_pair_mask_out` (932) | K1m2 | two columns, `a OP b` + optional ranges → mask words | 165 |

The launcher layer is **already** factored around a shared core
(`launch_decode_fused_tree_impl` → `run_rendered_decode`, driven by `VariantLaunchArgs:459-469`).
Five entry points are thin: of their 27-34 lines, ~12-22 is the precondition block and its
`fprintf`, ~5-8 fills `VariantLaunchArgs`, one line calls the core. The duplication is elsewhere.

**F4 — the pair launcher reimplements the core.** `launch_decode_fused_tree_pair_mask_out`
(932-1096) is 165 lines, ~120 of which duplicate the core: the 4-arm try/catch (945, 1083-1095
vs 699-712), `dtype_to_cxx` + report (962-970 vs 657-663), the elem_size ternary (997-1002 vs
664-667, a third copy), transients + alloc lambda (990-996 vs 671-677),
`synthesize_decode_transients` + error (1003-1011 vs 680-685), render + `RenderError` catch
(1019-1026 vs 490-496), `compile_rendered` (1027-1032 vs 500-505), the buffer bind loop
(1034-1045 vs 509-519), args assembly (1052-1062 vs 552-586), and
`maybe_raise_smem` + launch + sync (1064-1081 vs 588-612). It forked because the core assumes
**one** tree/dtype/`LabeledBuffers`; the pair needs two plus a merge (1014-1017).

> **Safety consequence:** the pair path omits the per-chunk metadata bounds guard at
> `run_rendered_decode:520-547`, whose comment states it prevents an out-of-bounds read that
> "would fault the CUDA context". This is a missing check, not just redundancy.

**F5 — two switches over one enum, unchecked.** `run_rendered_decode:568-586` pushes trailing
kernel *arguments* per variant; `renderer.cpp:343-387` emits the matching trailing
*parameters*. They must agree exactly across two files, and `cuLaunchKernel` takes `void**` —
**a mismatch is silent argument misalignment, not a compile error.** Highest-risk duplication
in this area.

**F6 — five bespoke precondition blocks.** (737-745, 773-785, 803-815, 833-846,
1109-1123), ~60 lines differing only in which `VariantLaunchArgs` slots must be non-null — a
property of the variant.


### Launcher refactor

**1. Spec carries its trailing parameters** (fixes F5 by construction):

```cpp
enum class TrailingParam { pred_lo, pred_hi, sel_mask, chunk_offsets,
                           keys_chars, key_width, row_indices, len_out };
struct DecodeKernelSpec { ...; std::vector<TrailingParam> trailing; };  // kernel param order
```

`finalize()` emits the C++ parameter text *and* `trailing` from one list; the launcher becomes
`for (auto p : spec.trailing) args.push_back(va.slot(p));`. Both switches disappear. Under the
enumerator × consumer factoring, each piece owns its emitted text *and* its `TrailingParam`
tags — a new variant is one object, not two coordinated switch arms in two files.

**2. Core takes N trees** (fixes F4):

```cpp
struct DecodeInputs {
  std::span<const jit::FusedTree* const> trees;    // 1, or 2 for pair
  std::span<const char* const>           dtypes;
  std::span<jit::LabeledBuffers* const>  labeled;  // merged + re-keyed internally
};
```

`"0.*" → "k.*"` re-keying moves into the core for `k > 0`. The pair launcher drops to a thin
wrapper **and inherits the metadata bounds guard it currently lacks.**

**3. Table-driven preconditions** (fixes F6):

```cpp
struct VariantContract {
  bool needs_mask_words, needs_chunk_offsets, needs_row_indices,
       needs_keys, needs_len_out, needs_out;
  bool rejects_float;                       // ballot variants: integer-domain compare only
  std::int64_t (*domain)(std::int64_t);     // identity; rows+1 for str_split_meta
};
```

One `check_contract(...)` with one uniform diagnostic replaces five blocks; `str_split_meta`'s
`num_string_rows + 1` domain shift (854-855) becomes declared rather than buried; the thrice-
written elem_size ternary folds into one helper.

**4. One entry point over per-op structs.** Once F6 removes the validation bodies, the wrappers
are one line each — so why keep them? They do buy something real: each gives one variant a
**total signature over a partial struct**. `VariantLaunchArgs` has eight optional, defaulted
fields; each variant needs a different subset, so calling the core directly and forgetting
`va.keys_chars` compiles and fails at runtime. But they buy it *badly* — they funnel into the
same optional bag and the guarantee is then re-established by hand (F6). Keep the totality,
drop the bag:

```cpp
namespace decode_op {
  struct Plain        { void* out; };
  struct MaskOut      { range_predicate pred;  selection_mask& mask; };
  struct MaskConsume  { selection_mask const& mask; void* out; };
  struct IndexConsume { selection_mask const& mask; std::int32_t const* row_indices; void* out; };
  struct DictGather   { selection_mask const& mask; void const* keys; std::int32_t key_width; void* out; };
  struct StrMeta      { selection_mask const& mask; std::int64_t* src_offsets; std::int32_t* lengths; };
  struct PairMaskOut  { pair_cmp op; range_predicate a, b; selection_mask& mask; };
}
bool launch_decode(DecodeInputs, std::int64_t num_rows,
                   std::variant<decode_op::…> const&, rmm::cuda_stream_view);
```

No defaults ⇒ a missing field is a compile error (what the wrappers were reaching for). The
runtime table then checks only what types cannot: pointer non-null, `mask.num_rows == num_rows`,
non-float dtype for ballot ops, chunk-geometry match for pairs.

Optionally keep the seven named functions as trivial inline forwarders — discoverability and
grep-ability at zero cost, once F6 has emptied their bodies.

Net: the decode section (636-1133, ~500 lines) should land near 250.

**Leave `launch_masked_char_copy` (884-929) alone** — fixed non-JIT source, no tree, no spec;
forcing it into the framework adds coupling for nothing.

### The two refactors are one refactor

The launcher ops and the renderer variants are **the same taxonomy, enumerated flatly in two
places with two spellings**. Every existing op is a point in the enumerator × consumer product:

| Launcher op | Enumerator | Consumer |
|---|---|---|
| `Plain` | `all_rows` | `write_column` |
| `MaskOut` | `all_rows` | `ballot(range)` |
| `PairMaskOut` | `all_rows` | `ballot(pair)` — 2 trees |
| `MaskConsume` | `mask_bits` | `write_column` |
| `IndexConsume` | `index_list` | `write_column` |
| `DictGather` | `mask_bits` | `dict_gather` |
| `StrMeta` | `mask_bits` | `offsets_meta` |

So the op should be the **product**, not a flat list of its currently-implemented points:

```cpp
struct decode_op { Enumerator enumerator; Consumer consumer; };
```

Each axis piece then owns, in one place:

1. its emitted CUDA text (renderer),
2. its `TrailingParam` tags **and** the matching runtime pointers/scalars (launcher args — F5
   dissolves, because emission and binding come from the same object),
3. its precondition contract (F6's table),
4. its capability requirement (which plan roots support it — §4's `probe_column`).

That is the endgame: **adding a kernel shape touches one descriptor instead of five coordinated
sites** (`DecodeVariant` enum, emitted params, pushed args, a new wrapper, a `plan_supports_*`
probe).

Two consequences worth stating:

- **Arity stops being special.** `PairMaskOut` is not a separate path — it is `ballot` with a
  2-ary predicate, and `DecodeInputs.trees.size()` must match the predicate's arity. F4's
  165-line fork was the cost of treating arity as a fork rather than a parameter.
- **The product is bigger than the implemented set** (3 × 4 = 12 points, 7 implemented). Missing
  combinations become constructible the moment both pieces exist — see §7. Not all 12 are
  meaningful (`index_list` × `ballot` would re-ballot only survivors), so the product needs a
  small validity predicate; that predicate is one function, not twelve emitters.

### Safety net

Pin the rendered source for the plain variant of each shipped SF1000 plan shape in a golden
test **first**. Then any perturbation of the plain path is a string diff, not a performance
mystery — this is what makes threading sinks through the plain producers safe, and it removes
the constraint that forced F3.

Caveats: deduping F1 changes emitted *comment* text, hence the source string, hence the JIT
cache key — expect a one-time cold-cache blip. And F1 is verified as textual equivalence, not
compiled equivalence; diff the rendered source for the shipped plans before deleting.

## 6. Sequencing

**Phase 0 — land #1391 on its measurements.** ~17k lines with real numbers; do not block it on
an API refactor.

**Phase 1 — value-carrying results** (small, independently reviewable).
Replace P5's type sniff and P6's RTTI tags with `decoded_batch_representation` + `scan_read_result`.
*Accept:* `build_filter_expression_for`, `boolean_substituted_primary_indices`, both marker
classes and the two `dynamic_cast`s are gone; TPC-H results unchanged.

**Phase 2 — the facade.** Introduce `compressed_scan` over the *existing*
`build_fused_scan_directives` + `decompress_scan_filter`. No kernel changes. Move the three
`extract_*` functions behind it.
*Accept:* every name in §2's delete list is gone from `src/include/` and `src/compression/*.hpp`;
scan-side call site is the two lines above; no measurable delta.

**Phase 3 — internals.** One route enum, one `probe_column`, merged analysis pass, one
`fusion_policy`, `compact_scan_filter_output` folded into `decompress_scan`.
*Accept:* P1–P4, P7, P8 closed; `try_decompress_scan_filter`'s ~190 lines substantially reduced.

**Phase 4 — golden-source test**, then the §5 refactor.
*Accept:* plain rendered source byte-identical for all shipped plans; `test_masked_decode_variants`
reparameterized over (enumerator, consumer); emitter LOC roughly halved.

**Phase 4a — F5 first, independently of everything else.** Spec-carried `trailing` removes a
silent-argument-misalignment hazard and is a small, self-contained change; it does not need the
enumerator/consumer factoring to land. **Phase 4b** — F4 (N-tree core, which also closes the
pair path's missing bounds guard) and F6.

**Phase 4c — unify the taxonomies.** Only after 4a/4b and the emitter factoring: collapse
`decode_op` and the renderer variants into the single enumerator × consumer product (§5, "The
two refactors are one refactor"). Doing this earlier means refactoring both sides at once with
no golden test in between.
*Accept:* `DecodeVariant` is gone; adding a kernel shape touches one descriptor; the
`plan_supports_*` probes are derived from consumer requirements rather than hand-written.

Phase 2 must precede Phase 3: the facade is what buys freedom to change internals without
touching the scan. Today every internal change is scan-visible.

## 7. Future directions this unlocks

**Missing kernel combinations become constructible.** Once the op *is* the enumerator × consumer
product (§5), the 12-point space contains 7 implemented kernels; the rest need no new emitter,
launcher or probe — only a validity check and a reason to want them:

- **K4 × dict_gather**, **K4 × offsets_meta** (i.e. `index_list` × `dict_gather` / `offsets_meta`)
  — below the ~15% K4 crossover, dictionary and `str_split` strings are stuck on the mask walk
  purely because the emitter wasn't written. q12-class scans sit near 0.5% selectivity.
- **mask_bits × delta for K5/K6** — dict codes or string offsets behind a delta are `tier_b`
  today (`plan_supports_dict_selection_decode` requires bitpack codes;
  `plan_supports_str_selection_decode` requires a plain bitpack offsets child). `l_comment`'s
  `delta -> ans` offsets and `c_phone`'s `delta -> rle -> bitpack` are the concrete cases.

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
5. F5 (two unchecked switches over one enum) is a live correctness hazard — a mismatch is
   silent argument misalignment through `void**`, not a compile error — and should not wait for
   the rest of the plan.
6. F4's fix changes the pair path's behaviour in one respect: it starts enforcing the metadata
   bounds guard. If any shipped plan currently launches a pair kernel with under-sized per-chunk
   channels, it will now refuse instead of silently reading OOB. That is the intended outcome,
   but expect it to surface rather than stay quiet.
