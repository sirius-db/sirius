# Byte-range parquet splits (roadmap #2) — implementation plan

Goal: with 2+ CNs and a single large parquet file, the FE's split scan ranges translate and
execute; N splits of one file read every row **exactly once** across CNs (no duplication, no
loss, deterministic under any placement). Unblocks the 18/22 survey queries failing on
"byte-range splits do not cover the whole parquet file" (TPCH-SURVEY.md F1) — for most of them
up to the next guard (#1 partitioned output); full e2e gate is the Q6 class.

Base branch: `demo-multi-cn` (on top of the two-phase stack `64977ebb..11625add`).
Provenance: 3 research reports + 2 independent designs (converged); full designs in session
scratch (`brs-design-{deep,opus}.md`), essentials captured here.

## Decisions

**(a) Ranges ride the Substrait plan.** The CN sets `FileOrFiles.start/length` (fields already
exist in both protos: Rust prost u64s; C++ `algebra.pb.h:31788-31814`). The consumer gap is
closed **inside Sirius, not the duckdb-substrait submodule**: `from_substrait.cpp` is a git
submodule and DuckDB's `parquet_scan` has no byte-range parameter, so teaching it would still
need a side channel. Instead `lower_substrait` (`src/sirius_ffi.cpp:82-109`, the only
`SubstraitToDuckDB` caller) re-parses the plan bytes it already holds, walks
`ReadRel.LocalFiles`, and parks a `path → ranges` registry on a `ClientContextState`
(precedent: the stream bind catalog, `sirius_ffi.cpp:165-166`), consumed by
`build_parquet_table_info` (`sirius_physical_plan_generator.cpp:110-147`).
Encoding: `start==0 && length==0` ⇒ whole file (today's plans byte-identical);
canonical empty split ⇒ `(start=file_size, length=0)`.

**(b) Split→row-group mapping happens in the engine**, in `build_file_scan_info` between
`all_row_groups` (`parquet_gpu_ingestible.cpp:567`) and `filter_row_groups_with_stats` (`:570`)
— the parquet footer is already fetched and cached there (`:446-466`); selection flows into
`set_row_groups` (`:721`). Not in the CN (second footer read, TOCTOU). cudf's
`filter_row_groups_with_byte_range` is NOT the source of truth (its row-group "start"
definition is unverifiable here) — it becomes a cross-check test only.

**(c) The deterministic rule = StarRocks BE start-offset containment**
(`be/src/formats/parquet/utils.cpp:121-140`): a row group's start =
`min(col0 data_page_offset, index_page_offset if >0, dictionary_page_offset if >0,
rg.file_offset if set)`; a split owns the row group iff `start <= rg_start < start+length`.
cudf's footer structs have no thrift `__isset` — treat 0 as unset or every row group starts
at 0. Edge cases: `length==0` selects nothing; a row group straddling a boundary belongs to
the split holding its start (the read runs past the range — size bounds ownership, not I/O);
a split entirely inside one row group selects zero row groups ⇒ valid EMPTY scan via the
existing empty-split path (`parquet_gpu_ingestible.cpp:104-116`, `set_num_rows(0)` `:713-719`)
— never an error, never a whole-file read; one huge row group ⇒ its owner reads it all, other
splits are empty; an undeterminable start ⇒ loud throw.

**(d) CN gate: `validate_complete_files` → `resolve_ranges`** (`scan_paths.rs:124-177`):
normalize → refuse → coalesce. A complete tiling collapses to one whole-file item (zero
regression risk; existing tests stay green); partial coverage emits one `FileOrFiles` per range
with explicit start/length. Refusals (all loud): missing/disagreeing file size (existing),
negative start, `end > file_size`, **overlapping ranges** (today silently tolerated —
would duplicate rows), whole+partial mix for one file, two scan nodes over the same file set
with different ranges (the engine registry is path-set-keyed), and
`TScanRangeParams.has_more == Some(true)` (incremental delivery — silently ignored today, and
partial delivery is exactly what the removed coverage check made loud).

## Commits (engine before CN — a CN emitting ranges into an engine that ignores them is silent N× duplication)

- **C1 engine — the rule, wired to nothing.** New `src/include/op/scan/parquet_byte_range.hpp`
  (+ .cpp): `row_group_start_offset(FileMetaData, i)` and
  `row_groups_in_byte_range(FileMetaData, start, length)`. Catch2: every-boundary sweep over
  the multi-row-group test lineitem (union over any tiling == all row groups, pairwise
  disjoint), straddle, empty-split, single-row-group file, and a cudf
  `filter_row_groups_with_byte_range` cross-check (informational). Verify: `make test`.
- **C2 engine — the ingestible honors per-file ranges.** `resolved_file_ranges` beside the
  paths (`parquet_gpu_ingestible.hpp:69`), threaded via the split provider (`:404-423`),
  applied before stats pruning; pinned-cache identity includes the range so a whole-file pin
  can't serve a ranged scan (`sirius_scan_manager.cpp:697-731`). Catch2: two splits of one
  file — disjoint row counts summing to the whole; ranged vs whole cache miss. Verify: `make test`.
- **C3 FFI/planner — plan-carried ranges.** Range registry `ClientContextState`; populate in
  `lower_substrait` (hand-written walk of the plan proto, no reflection); consume in
  `build_parquet_table_info`; `assert_all_consumed()` before plan-gen returns
  (`sirius_physical_plan_generator.cpp:903/931`) so an unapplied range throws instead of
  degrading to whole-file; S3 path refuses ranges. Rust test in `rust/crates/sirius`:
  3-row-group fixture, split0+split1 results == whole-file results. Verify: `make test` +
  `cargo test -p sirius --lib`.
- **C4 CN — emit ranges.** `resolve_ranges` replaces `validate_complete_files`; `for_node`
  returns `{path, range}`; `local_files_rel` sets start/length (`node_translator.rs:1131-1149`);
  refusals from (d) incl. `has_more`. Translate tests with the REAL dump shapes
  (lineitem split offsets 0/81070259, lengths 81070259/81070518, file 162140777-ish — use the
  exact numbers from the R1 report). Verify: `cn-test-no-engine`.
- **C5 CN — hardening refusals.** Non-default `compression_type`,
  `num_of_columns_from_file` ≠ source-slot count, explicit empty-range skip. Verify:
  `cn-test-no-engine`.
- **C6 e2e + docs.** `cluster2` over the SINGLE-file 155 MB lineitem:
  `count(*) == 6001215`, `sum` checks vs DuckDB, Q6 → `61567694.9502…` (tolerance), CN logs
  show disjoint row-group sets; translate-only survey re-sweep → zero "do not cover" errors;
  update DEMO.md + TPCH-SURVEY.md + ROADMAP + REVIEW-GUIDE. Verify: mysql transcript +
  `pre-commit run -a`.

## Non-goals

1. Q12-class full e2e (its PARTITIONED-only joins still need roadmap #1) — its gate here is
   leaf translation only.
2. A guard inside the duckdb-substrait submodule (cross-repo; the in-repo
   `assert_all_consumed` covers the failure mode).
3. Cross-CN tiling verification (no single CN can see the whole tiling — the FE is the
   documented trust boundary; per-CN refusals of overlap/past-EOF/has_more are the net).
4. File-rewritten-between-plan-and-execute detection (optional file-size check deferred).
5. S3/remote byte ranges (refused loudly).

## Risks (ranked)

1. **Ranges emitted but not applied → every row read N times, silently.** Commit order,
   `assert_all_consumed`, and the C6 `count(*)` gate.
2. **Row-group ownership drift (ours vs BE vs cudf) → lost/duplicated row groups.** One
   implementation of the BE formula, the boundary-sweep test, the cudf cross-check.
3. **FE behaviours outside the tiling assumption** (`has_more` incremental delivery, dop>1
   spreading ranges) → refused loudly rather than absorbed.

## Open questions

| # | Question | Blocks | Resolution path |
|---|---|---|---|
| Q1 | cudf's own byte-range "start" definition | nothing (cross-check only) | C1 test answers it |
| Q2 | does the FE ever co-locate two ranges of one file on one instance | e2e-vs-unit coverage of the multi-range path | inspect R1 dumps during C4 |
| Q3 | DuckDB MultiFileList behaviour on duplicate literal paths | C3 test-first | write the two-range single-file test first |

## Progress log

- 2026-08-06: plan approved in advance ("implement the plan; update docs per phase"). Implementation starting with C1.
- 2026-08-06 **C1 DONE** — `a5c25f76` feat(scan): deterministic byte-range → row-group ownership
  rule (`parquet_byte_range.{hpp,cpp}`), wired to nothing. Verified: `[parquet_byte_range]`
  4 cases / 58 assertions — exact-tiling sweep (k=1..16), straddle/empty/boundary edges,
  convention corners, real-footer check on the test lineitem. **cudf's
  `filter_row_groups_with_byte_range` agrees with the StarRocks rule on the real footer**
  (open question Q1 answered: cudf also uses start-containment; kept informational).
- 2026-08-06 **C2 DONE** — `ad2400b7` feat(scan): the ingestible honors per-file ranges
  (threaded via the split provider, applied before stats pruning; empty selection = valid
  empty split via the all-pruned fallback; pinned cache refuses ranged identities). Verified:
  6 cases / 84 assertions on a generated 10-row-group file + full suite 2187 passed / 1 skipped.
- 2026-08-06 **C3 DONE** — `bf36716e` feat(ffi): ranges ride the plan (lower_substrait
  extraction → per-plan ClientContextState → claimed in build_parquet_table_info;
  single-shot claims, assert_all_consumed, throw-on-unknown-rel, S3 refusal). Verified: Rust
  GPU test `byte_range_splits_read_every_row_exactly_once` — two half-file splits partition
  30k rows exactly; **two-splits-in-one-plan == whole file (open question Q3 answered: the
  duplicate-path case works end to end)**; empty split = valid empty result.
- 2026-08-06 **C4 DONE** — `d1e5d969` feat(starrocks): the CN emits splits instead of refusing
  them (`resolve_ranges` replaces `validate_complete_files`; whole-tiling collapse keeps old
  plans byte-identical; refusals: overlap, past-EOF, negative, zero-owned-bytes, `has_more`).
  Verified: 96 translator tests (incl. real FE dump shapes: 162140518 split at 81070259),
  cn-test-no-engine, clippy clean.
- 2026-08-06 **C5 DONE** — `02e90ef6` refuse compressed-container ranges (97 translator tests).
  Deviation: no separate `num_of_columns_from_file` check — it only diverges when path-derived
  columns exist, already refused via `columns_from_path`.
- 2026-08-06 **C6 DONE** — e2e + docs. First live run tripped the unclaimed-range guard
  (loud, as designed) and exposed a real bug: the FE spells `file:/…` in the plan while
  DuckDB's bind reports the plain path → registry never matched. Fixed by canonicalizing at
  insert AND lookup (`e0180970`). Then the full gate passed on cluster2 over the SINGLE
  155 MB lineitem split across both CNs: **count(*) = 6001215 (exactly-once), sum/min/max
  match, Q6 = 61567694.9502**; `agg_stage=1` over the split file returns the same value;
  GROUP BY / avg guards unchanged. DEMO.md updated (`8c2ebea5`).

**STACK COMPLETE: `a5c25f76..8c2ebea5` (7 commits).** Full engine suite green over the
complete stack (2187 passed / 1 skipped). The byte-identical-file workaround is
retired; the next blocker for the survey's join/GROUP-BY queries is partitioned output
(roadmap #1), with avg expansion behind it.
