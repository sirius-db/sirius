**09 · Export partition views without slice copies**

[All paths](README.md) · [Source review](../../../../starrocks-plan-improvement.md)

Status: proposed implementation plan. Baseline: `281b13bc`. Objective: eliminate deep materialization of every remote hash-partition slice before packing. Prerequisites: [03 · Spillable exchange repositories and reload](03-exchange-spill-and-reload.md) and [07 · Independent packing and CUDA completion](07-independent-gpu-packing.md); transport experiments use [00 · Trustworthy measurements and benchmark coverage](00-measurement-and-benchmarks.md).

**Current behavior and code map**

Hash partitioning builds normalized key columns when needed, partitions the effective table, slices the partitioned result, and deep-copies the original columns of each slice into destination batches. Remote export then packs those copies.

| Source | Responsibility |
|---|---|
| [gpu_partition_impl.cpp](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/src/op/partition/gpu_partition_impl.cpp#L45) | Key normalization, cuDF partition, slice materialization. |
| [sirius_physical_streaming_sink.cpp](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/src/op/sirius_physical_streaming_sink.cpp#L168) | Emit transport-oriented destination views. |
| [owning_table_view.hpp](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/src/include/op/scan/owning_table_view.hpp) | Existing ownership/view concepts to inspect for reuse; not assumed compatible. |
| [sirius_ffi.cpp](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/src/sirius_ffi.cpp#L747) | Pack a supported sliced view with an explicit parent owner. |
| [test_gpu_partition_impl.cpp](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/test/cpp/operator/test_gpu_partition_impl.cpp) and [test_physical_streaming_sink.cpp](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/test/cpp/operator/test_physical_streaming_sink.cpp) | Partition semantics and stream emission tests. |

**Proposed representation**

Retain the partitioned parent table in immutable, spill-aware ownership. A destination descriptor carries the parent handle, row interval, selected original columns, exact rows, destination ID, and writer completion. Packing consumes a view while holding the parent's residency guard. Drop temporary hash-only columns as early as allowed by ownership and GPU completion.

A child view does not own only its apparent bytes: the slowest remaining destination may retain the whole parent. Track physical parent bytes once and logical child bytes separately. Release the parent after every child is packed or independently materialized; do not wait for remote consumption when a packed child already owns separate source bytes.

Keep this change exchange-specific initially. The shared partition kernel is used outside StarRocks; callers expecting independently mutable destination batches must retain their current behavior.

**Implementation slices**

1. **Representation and accounting:** add an exchange result type that retains parent ownership and destination ranges. Verify view slicing/packing behavior against the pinned cuDF APIs, especially string offsets and null masks.
2. **Remote export prototype:** emit parent-backed descriptors only for remote hash output; materialize local children through the existing path where native consumers require independent residency.
3. **Pressure policy:** select eager child materialization or parent spill when a slow child would retain an excessive parent allocation. Use path 03 for reload and hold guards through pack completion.
4. **Hash-only temporary reduction:** separately evaluate partitioning from normalized-key-derived row indices or other supported APIs so temporary key columns need not be gathered as payload. This is a second optimization requiring its own parity and bandwidth measurements.

**Correctness and tests**

Compare all destination multisets against the current hash path for mixed-width integer keys, supported decimal normalization, duplicate keys, strings, nulls, zero-row inputs, empty partitions, and skew. Compare the mapping between independently planned senders; changing hash/seed/cast representation on one side alone is forbidden.

Test pack from a nonzero row offset, parent destruction after child creation, one child spilled while another packs, cancellation after some children complete, and delayed writer events. Validate counts and values, not just partition sizes.

Acceptance: destination results and routing are unchanged, intermediate D2D slice copies disappear for eligible remote output, and parent retained bytes remain bounded under a delayed child. Any zero-copy claim must include the partition reorder and subsequent pack, which still move data.

**Benchmark**

Use narrow and wide tables, variable strings/nulls, 2/4/8 destinations, uniform and concentrated key distributions, and several batch sizes. Measure partition time, temporary cast bytes, slice-copy time/bytes, pack time, peak parent residency, and end-to-end shuffle time. Report the cost of pressure fallback as well as the fast path.

**Rollout and decisions**

Gate the representation at the StarRocks streaming sink and retain existing materialized partitions elsewhere. Rollback applies to new output batches; active views retain parents until done. If parent retention costs more than the eliminated copies, retain eager copies for that size/fan-out regime.

Resolve whether an existing owning-view type meets immutable residency and spill needs before introducing a new one. Do not expose raw cuDF views across FFI without an owning handle and an explicit completion contract.
