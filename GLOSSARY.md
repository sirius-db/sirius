# Sirius Query Compilation

GPU-native SQL execution and query compilation in Sirius, translating relational plans and expressions into compiled GPU kernels.

## Language

**Expression Tree**:
A hierarchical representation of a scalar SQL computation (references, constants, operators, conditionals, and functions) that yields a single typed, potentially nullable value per input row.
_Avoid_: AST (when referring to cuDF's `cudf::ast`), formula

**Pipeline**:
A maximal sequence of relational operators in which a tuple can be processed in GPU registers from source to sink without materializing intermediate columns in device memory.
_Avoid_: Stage, fragment, wave

**Pipeline Breaker**:
An operator boundary that requires materializing tuples out of GPU registers into device memory before downstream processing can continue.
_Avoid_: AST breaker, barrier, materialization point

**Register Value**:
A per-row intermediate value held in GPU registers during compiled evaluation, consisting of a typed payload, a boolean validity flag, and an overflow poison flag.
_Avoid_: Slot, variable, optional

**Translation Context**:
The compile-time state passed through `produce`/`consume` that tracks which input attributes and sub-expressions are already loaded in GPU registers for the current row.
_Avoid_: Symbol table, scope, environment

**Materialization Boundary**:
A sub-expression within an Expression Tree that cannot be evaluated inline in GPU registers and must be pre-evaluated into a temporary device column before the compiled kernel runs.
_Avoid_: AST breaker, fallback node, cogwheel

**Sink Consumer**:
The terminal consumer at the root of a compiled `produce`/`consume` tree that defines how surviving Register Values exit GPU registers--such as writing 1-to-1 projected columns (`ProjectSink`), compacting filtered rows (`FilterSink`), or accumulating into tiered state (`AggregateSink`).
_Avoid_: Root operator, output writer, epilogue

**Carrier Restoration**:
Widening a narrowed physical storage type (such as a narrowed integer, decimal, or date produced by compressed materialization) back to its declared SQL logical type in GPU registers upon loading.
_Avoid_: Schema cast, type widening, temporary cast column

**Tiered Memory Schedule**:
A mapping of an operator's working state across GPU Thread-Private Registers (TPR), Block-Shared Memory (BSM), and Global High-Bandwidth Memory (HBM) to minimize off-chip memory traffic and atomic contention.
_Avoid_: Cache hierarchy, memory placement

**Sparse-to-Dense Compaction**:
The filter write schedule that writes variable-cardinality surviving tuples contiguously into HBM in ascending row order: warp-level ballot/ranking and a block-level shared-memory prefix sum within a multi-row tile. A one-tile batch scatters directly from the predicate kernel; a larger batch publishes per-tile survivor counts and per-warp ballot words, scans the counts in one block, and scatters with a third kernel.
_Avoid_: Stream compaction, two-pass filter, densification pass
