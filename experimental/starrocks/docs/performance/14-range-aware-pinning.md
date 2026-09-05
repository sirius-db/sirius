**14 · Range-aware pinning and scan balance**

[All paths](README.md) · [Source review](../../../../starrocks-plan-improvement.md)

Status: proposed implementation plan. Baseline: `281b13bc`. Objective: reuse pinned data for FE byte-range scans without reading unassigned rows, and distinguish partition skew from exchange cost. Prerequisite: [00 · Trustworthy measurements and benchmark coverage](00-measurement-and-benchmarks.md). This is independent of remote transfer concurrency.

**Current behavior and code map**

Partial file ranges already select row groups by start-offset ownership. Pin matching deliberately rejects byte-range scans, because current whole-file provenance cannot safely select their rows. Whole-file subset serving exists and should remain correct.

| Source | Responsibility |
|---|---|
| [sirius_scan_manager.cpp](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/src/scan_manager/sirius_scan_manager.cpp#L1918) and [sirius_scan_manager.hpp](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/src/include/scan_manager/sirius_scan_manager.hpp) | Range-aware pin identity, serviceability, and selected chunks. |
| [parquet_gpu_ingestible.cpp](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/src/op/scan/parquet_gpu_ingestible.cpp#L823) and [parquet_byte_range.cpp](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/src/op/scan/parquet_byte_range.cpp) | Canonical row-group ownership rules. |
| [sirius_extension.cpp](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/src/sirius_extension.cpp) | Pin construction/provenance retention. |
| [load_balancing_scan_batch_coalescer.cpp](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/src/scan_manager/load_balancing_scan_batch_coalescer.cpp) | Avoid losing provenance when combining row groups. |
| [scan_paths.rs](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/crates/starrocks-plan-translator/src/scan_paths.rs) | FE range normalization and exact tiling rules. |
| [test_pin_table_file_subset.cpp](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/test/cpp/integration/test_pin_table_file_subset.cpp) and [test_parquet_byte_range.cpp](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/test/cpp/scan/test_parquet_byte_range.cpp) | Existing correctness foundations. |

**Proposed cache contract**

Key pinned data by canonical file identity/version, schema/projection, row-group identity, and row intervals represented by each chunk. Preserve provenance through pin coalescing, compression, column merging, and reload. A range selects exactly the row groups whose canonical start offset is in its interval; it does not clip I/O or ownership at an arbitrary row inside a straddling row group.

If a chunk contains both selected and unselected row groups, use precise row slices or materialize only selected rows. File names alone cannot decide serviceability. Start with row-group-aligned pin chunks if that yields a simpler proof, then measure the batching/compression cost before allowing finer slicing.

Cache miss or incomplete provenance falls back to the existing disk path. Removed/replaced files invalidate relevant entries. Never treat a pin as valid solely because a path and size match if the data version changed.

**Implementation slices**

1. **Scan accounting:** report assigned byte ranges, selected row groups/rows, pin-hit/miss reasons, decoded bytes, and per-CN elapsed scan work. Record these alongside FE plans.
2. **Provenance schema:** retain file version, row-group ID, and row spans in pinned metadata; ensure whole-file and subset behavior still works.
3. **Range-aware serving:** resolve normalized ranges to pinned row groups and project/slice safely. Keep the current rejection when coverage or type agreement cannot be proven.
4. **Compression/coalescing integration:** preserve the mapping through merged chunks and selected-column pins; compare row-group pinning against existing batch-size choices.
5. **Balance experiments:** vary FE split layout and source row-group sizes offline on copied benchmark datasets. Do not change production files or override FE ownership ad hoc.

**Tests**

Partition the same file into adjacent, uneven, empty, and row-group-straddling ranges across simulated CNs. The union must equal a whole-file read exactly once, with no duplicated/missing rows. Test multiple ranges of one file, strict file subsets, projection reorder, missing columns, nulls, compressed pins, mixed chunks, and replaced files.

Acceptance: a serviceable byte-range scan uses only assigned rows; unsupported provenance retains the disk fallback; pin hits and row counts are auditable per CN. Preserve fail-closed overlap validation in the translator.

**Performance experiment**

Compare disk, warm filesystem cache, whole-file pin where eligible, and new range-aware pinning as separate modes. Hold total data and CN count constant, then sweep row-group size and assignment skew. Report pin memory, preparation cost, decode avoided, scan completion skew, and total query latency.

The largest indivisible row group can limit parallelism; smaller FE byte intervals alone do not split its row ownership. Rewriting benchmark data into smaller row groups may improve balance but costs metadata and compression efficiency, so evaluate it separately from the code change.

**Rollout and decisions**

Enable range-aware hits only for the new provenance version; old entries continue through current matching/fallback. Select behavior before scan admission. Keep pin creation and query timing separate and report amortization across repeated queries.

Resolve a reliable local file version contract and how compressed/coalesced chunks retain exact row spans. Do not remove the `has_byte_ranges` safeguard until the range serviceability proof and tests are in place.
