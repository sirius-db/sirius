# Intra-pipeline kernel fusion: matcher and TPC-H coverage

Question: every `sirius_pipeline` is an ordered operator list executed sequentially by
`compute_task()`, bounded by pipeline breakers (HASH_JOIN, HASH_GROUP_BY, PARTITION). Adjacent
operators in that list are separate GPU kernel launches with an intermediate materialization
between them. Is it worth collapsing straight-line runs of them into a single fused kernel?

`src/pipeline/fusion_matcher.{hpp,cpp}` answers that by measurement rather than design. It is
static analysis only: it reads the plan and reports which runs of adjacent FILTER/PROJECTION
operators *could* collapse into one expression evaluation, and logs the report at DEBUG next to
the plan print in `sirius_engine::initialize_internal`. Nothing executes differently.

**Not to be confused with the existing "pipeline fusion."** `physical-plan-generation.md`'s
"Merge fusion" (`fuse_merge_pipelines`) folds a MERGE_GROUP_BY/MERGE_TOP_N stage into an adjacent
pipeline as one more *operator*, saving a task launch and a repository round trip. That is
task-level fusion. This is about *kernel* launches within a single pipeline's operator list.

## Why the fusability predicate is a separate walk

`cudf_ast_op_count()` looks like the "does this tree lower into a cuDF AST" predicate but cannot
serve as one: it returns 0 for `reference` and `constant`, which *are* AST-expressible, and it
counts only the supported prefix of a tree, so a breaker nested under a supported parent is
invisible in the count. `aggregate::cudf_ast_op_count()` additionally throws. `is_ast_fusable()` /
`find_ast_breaker()` therefore walk the tree separately, mirroring the AST-mode branch conditions
in `src/expression_evaluator/specializations/`, and report the first breaker by kind.

## TPC-H coverage

Measured over the 22 canonical TPC-H queries:

| | count |
|---|---|
| FILTER + PROJECTION operators in all plans | 17 + 42 = **59** |
| ...adjacent to another one (i.e. inside a fusable chain) | **12** (6 chains) |
| ...where the whole chain lowers into one cuDF AST (`ast_clean`) | **4** (2 chains) |
| queries containing any chain at all | **6 / 22** |

Every chain is exactly length 2, and every one sits at operator index `[1..3)` — directly after
the source and directly before a pipeline breaker (HASH_GROUP_BY ×3, UNGROUPED_AGGREGATE, TOP_N).
A chain-level fuser would therefore remove 6 intermediate materializations across the entire
TPC-H suite, only 2 of which would collapse into a single JIT kernel.

Reproduce with:

```bash
pixi run build/release/extension/sirius/test/cpp/sirius_unittest "[fusion_matcher]"
```

## Why the adjacency rate is so low

Roughly 20% of these operators have a fusable neighbour, because the two cases a chain fuser
would have caught are already handled elsewhere:

- `sirius_physical_filter`'s `output_mask` (`src/include/op/sirius_physical_filter.hpp`) already
  folds a trailing pure-passthrough projection into the filter's own gather, so the easiest
  FILTER+PROJECTION pairs never reach the operator list as two operators.
- Decode-time filter pushdown already fuses scan-side filtering into decompression:
  `simpatico::decompress_scan_filter(...)` is called from `src/compression/compressed_scan.cpp`,
  and `DECODE_PUSHDOWN_PLAN.md` records that work.

## Where the leverage actually is

The 4 non-`ast_clean` operators split 2/2 between two breaker kinds:

- `decimal_function` (q1, q19) — DECIMAL-returning arithmetic such as
  `l_extendedprice * (1 - l_discount)`. `function_call::cudf_ast_op_count()` returns 0 for these
  pending [rapidsai/cudf#21996](https://github.com/rapidsai/cudf/pull/21996).
- `unsupported_function` (q7, q8) — a function outside `supported_ast_functions`, which today
  holds only `{add, sub, mul, div, int_div, mod}`.

Lifting either restriction widens what the *existing* per-operator `AST_JIT` path fuses in every
query touching such an expression — a far larger surface than the 6 chains a chain-level fuser
would ever see, and it needs no plan-level machinery.

## Conclusion

A chain-level fuser is not worth building against this workload. The matcher is worth keeping: it
is the cheap instrument for re-testing that conclusion against a different workload, and its
breaker histogram is independently useful.

One further finding, for anyone considering a runtime kernel generator here: cuDF already
maintains a durable on-disk JIT kernel cache at `~/.cudf/<cudf-version>/<sm-arch>/`, with entries
keyed by `{source digest, sm arch, nvrtc version, nvjitlink version, cache format version}`. Any
approach built on `cudf::compute_column_jit` inherits cross-process kernel caching for free and
does not need a separate cache layer such as `simpatico_codegen`'s.

Not measured: the absolute kernel-launch and materialization overhead on TPC-H at realistic scale
factors. It would size the two levers above, but it is not a gate on the fusion question — however
large that overhead is, there are only 6 operator pairs in all of TPC-H for a chain fuser to act
on.
