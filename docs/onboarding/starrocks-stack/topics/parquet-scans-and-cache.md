# Parquet scans and the pin cache

**Module / language:** Super Sirius scan planner and operators, C++ / cuDF
**Scope:** This guide describes the reviewed `7610840c` snapshot. It is not a
description of later development work or a claim that the code has been run.

[Back to the guide](../README.md)

## Why this scan path exists

Parquet is a columnar file format. Each file is divided into **row groups**:
independent blocks of rows that can be read and filtered separately. A
distributed StarRocks plan can assign different byte ranges of the same Parquet
file to different fragments. Sirius must then make the same ownership decision
for every fragment, or a row group could be read twice or not at all.

The relevant feature sequence in the original draft inventory was #1696,
**byte-range ownership**; #1700, **range-aware ingest**; and #1717, **pinned
tables serving file subsets**. Together, they make an ordinary file scan aware
of distributed splits while preserving the whole-file cache's correctness.

## Byte ranges own row groups by their start

A range is written as `(start, length)`. It owns a row group when that row
group's starting byte offset is at least `start` and strictly less than
`start + length`. The end is excluded. A row
group that crosses a range's end still belongs to the range containing its
start. This matches the StarRocks back-end convention and makes adjacent ranges
partition the file without needing to split a row group.

For example, with row groups at offsets 16, 100, 200, and 220, a range starting at
100 with length 100 owns the group at 100, even if that group extends past byte
200. The group starting exactly at 200 belongs to the next range, not this one.
The pure ownership function is in
[parquet_byte_range.cpp](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/src/op/scan/parquet_byte_range.cpp#L25).

Later branch work, proposed as [P01](../pr-packages/p01.md), carries the file path
and range through the Substrait `LocalFiles` plan. Per-plan state
claims a range for a file only once and checks that all declared ranges were
used. That prevents two physical scan operators from consuming the same split
by accident ([range state interface](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/src/include/planner/substrait_scan_ranges.hpp#L32)).

The Parquet ingestible applies range ownership before statistics pruning. A
statistics prune can later discard a row group whose min/max values prove it
cannot match a filter, but it cannot make a ranged scan fall back to reading the
whole file. An empty owned set is still an intentional empty split
([selection order](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/src/op/scan/parquet_gpu_ingestible.cpp#L821)).

## What the pin cache can and cannot serve

`pin_table` keeps a prepared table in a chosen memory tier so later compatible
scans avoid rereading it. A whole-file pin represents complete files. A ranged
scan represents only parts of a file, so it must bypass the pin cache; treating
it as a whole-file hit could return rows outside the assigned range. This is an
intentional correctness boundary, not a missed optimization.

#1717 addresses a different case: a pin made from files A, B, and C can safely
serve a later complete-file scan of A and C. At pin time, each stored chunk
records its source-file provenance—where that chunk came from. At serve time,
the cache canonicalizes file paths and classifies the request as an exact match,
a subset, or a miss. A request including D is a miss because the pin cannot
provide D. A subset hit selects only chunks from the requested files, preserving
the caller's requested file set.

"A and C" asks for two complete files and may use the subset-serving path.
"Bytes 100 through 200 of A" asks for part of a file and bypasses the cache. The
scan-manager interface documents the exact/subset distinction
([cache interface](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/src/include/scan_manager/sirius_scan_manager.hpp#L120)).

## Important limitation from review

The reviewed ownership arithmetic adds unsigned `start + length` without first
rejecting overflow. A malformed input such as `start = 4` and an extremely
large length can wrap the calculated end to a smaller number and exclude a row
group that the unwrapped range would include. Normal StarRocks-generated splits
were not shown to produce this input, but it is still a validation gap. A future
fix should reject a length greater than `UINT64_MAX - start` and add a regression
test. The code path is [the interval calculation](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/src/op/scan/parquet_byte_range.cpp#L56).

## Where to continue

Read `substrait_scan_ranges.hpp` and `substrait_scan_ranges.cpp` for plan-level
range state, then `parquet_byte_range.cpp`, then the selection block in
`parquet_gpu_ingestible.cpp`. The end-to-end file-subset scenarios are recorded
in [the integration test](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/test/cpp/integration/test_pin_table_file_subset.cpp#L196);
that file is a map of intended coverage, not a fresh test result in this review.

The output of a successful scan can feed the exchange path described in
[C++ streaming primitives](cpp-streaming.md). For the API that builds and moves
those fragments, read [Rust FFI and DuckDB lowering](rust-ffi-duckdb.md).
