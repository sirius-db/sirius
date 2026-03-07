# Porting Optimizations to dev-fused-kernel

Branch: `cudf-25.12-optimization` → `dev-fused-kernel` (main project)

---

## Status of Each Optimization

| Optimization | cudf-25.12-optimization | dev-fused-kernel | Action |
|---|---|---|---|
| P1: COUNT DISTINCT two-phase | committed (d65ab00) | missing — uses `make_nunique_aggregation` | Port |
| P1b: Mixed aggregate split | committed (863c90e), **NO guard — causes Q09 regression** | missing | Port WITH cardinality guard |
| P3: STRLEN offset arithmetic | committed (67a12c8) | missing — materializes full string column | Port |
| P4: String GROUP BY hash fingerprint | committed (2634ce9), env var toggle | missing | Port (optional, only if testing shows benefit at 100M+ rows) |
| P5: Empty-string filter via offsets | not yet implemented | missing | Implement directly on dev-fused-kernel |

---

## Pipeline Execution Model: cuCascade

**CRITICAL**: dev-fused-kernel uses cuCascade for batched pipeline execution. Before porting
any optimization, verify that the groupby operator still receives the **full input** in a single
call, not batched chunks. If cuCascade batches data before groupby:
- P1/P1b's `cudf::distinct()` call would only deduplicate within a batch, producing wrong results
- P3/P5 are per-row transforms and work correctly with batching

Check `GPUPhysicalGroupedAggregate::Execute()` on dev-fused-kernel to confirm groupby
receives the complete input table.

---

## Key API Differences: cudf 25.12 → cudf 26.x

All cudf calls in `dev-fused-kernel` use two new parameters that replace the old globals:

| Old (cudf 25.12) | New (dev-fused-kernel) |
|---|---|
| `rmm::cuda_stream_default` | `executor.execution_stream` |
| `GPUBufferManager::GetInstance().mr` | `executor.resource_ref` |

In cudf_groupby.cu specifically, you don't have access to `executor` directly — use
`rmm::cuda_stream_default` and `gpuBufferManager->mr` as in the old branch, then verify
they still work with cudf 26.x (they should — these are valid cudf overloads).

Check cudf 26.x release notes for any signature changes to:
- `cudf::distinct` — verify `duplicate_keep_option` enum values unchanged
- `cudf::left_join` — verify return type (should still be pair of device_uvector)
- `cudf::scatter` — verify overload still accepts column_view scatter map
- `cudf::gather` — verify table_view + column_view overload exists
- `cudf::hashing::xxhash_64` — verify table_view overload exists

---

## Porting P1: COUNT DISTINCT Two-Phase

**What it does:** replaces `make_nunique_aggregation` with `distinct(group_key + cd_col)` →
`COUNT_STAR groupby`. Avoids cudf's sort-based nunique for pure COUNT DISTINCT queries.

**File:** `src/cuda/cudf/cudf_groupby.cu`

**Where to insert:** before the `cudf::groupby::groupby grpby_obj(...)` call.

**Condition to trigger:**
```cpp
bool all_count_distinct = (num_aggregates > 0);
for (int agg = 0; agg < num_aggregates; agg++) {
    if (agg_mode[agg] != AggregationType::COUNT_DISTINCT) {
        all_count_distinct = false; break;
    }
}
if (all_count_distinct) { /* P1 path — handles 1 or more CD columns */ }
```

Note: do NOT add `&& num_aggregates == 1`. P1 already handles multiple COUNT DISTINCT
columns via a loop over each `agg` (see existing code lines 210–270).

**P1 path logic:**
1. For each COUNT DISTINCT aggregate:
   a. Build `table_view` of {group_key_cols..., cd_col}
   b. Call `cudf::distinct(table_view, all_key_indices, KEEP_ANY, ...)`
   c. Build COUNT_STAR groupby on the deduplicated table (keys = first `num_keys` columns)
   d. Extract group keys (only on first aggregate) and count results
   e. Convert INT32 count result to INT64 (Sirius uses INT64 for counts)

**Copy from:** `cudf-25.12-optimization:src/cuda/cudf/cudf_groupby.cu`, lines 195–277
(the `// --- Two-phase COUNT DISTINCT ---` block)

**Required includes to add:**
```cpp
#include <cudf/stream_compaction.hpp>
#include <cudf/copying.hpp>
```

---

## Porting P1b: Mixed Aggregate Split (with Cardinality Guard)

**What it does:** for queries with COUNT DISTINCT + other aggregates (SUM/AVG/COUNT),
splits into two separate groupbys and joins results. Replaces expensive sort-based nunique
with hash-based distinct + count.

**WARNING:** P1b caused a 3x regression on Q09 (100M rows, no filter, UserID nearly unique).
The cardinality guard below is MANDATORY — do not port P1b without it.

### Why Q09 regresses without guard

Q09: `SELECT RegionID, SUM(AdvEngineID), COUNT(*), AVG(ResolutionWidth), COUNT(DISTINCT UserID) FROM hits GROUP BY RegionID`

At 100M rows with no WHERE filter:
- `distinct(RegionID, UserID)` produces ~100M rows (UserID is nearly unique → zero dedup)
- P1b does: regular groupby + distinct + count groupby + left_join + gather + scatter
- That's 2× the work of a single groupby with nunique, for zero benefit

Q22 works because its LIKE filter reduces input to ~50K rows, where P1b overhead is negligible.

### Cardinality guard

```cpp
// Guard: P1b only helps when either:
//   (a) input is small (overhead always negligible), OR
//   (b) input is large but expected deduplication is significant
//
// size_threshold = K * estimated_output_groups
//   K is measured via benchmark_results/calibrate_p1b.py
//   Override: SIRIUS_P1B_K env var (default 100)
//
// dedup_threshold: if expected_distinct_pairs / input_rows > threshold, skip P1b
//   Override: SIRIUS_P1B_DEDUP_THRESHOLD env var (0.0–1.0, default 0.8)

static idx_t p1b_K = []() -> idx_t {
    const char* env = std::getenv("SIRIUS_P1B_K");
    return env ? std::stoull(env) : 100ULL;
}();
static double p1b_dedup_threshold = []() -> double {
    const char* env = std::getenv("SIRIUS_P1B_DEDUP_THRESHOLD");
    return env ? std::stod(env) : 0.8;
}();

bool skip_p1b = false;
idx_t size_threshold = p1b_K * estimated_output_groups;
if (size > size_threshold) {
    // Large input: check expected dedup ratio
    idx_t group_ndv = estimated_output_groups;
    idx_t cd_ndv = cd_col_ndv;  // from HLL stats, or size as conservative fallback
    // Overflow-safe: check if group_ndv * cd_ndv would overflow or exceed size
    double dedup_ratio;
    if (group_ndv > 0 && cd_ndv > size / group_ndv) {
        dedup_ratio = 1.0;  // product exceeds size — no dedup expected
    } else {
        dedup_ratio = static_cast<double>(group_ndv * cd_ndv) /
                      static_cast<double>(size);
    }
    if (dedup_ratio > p1b_dedup_threshold) {
        skip_p1b = true;  // not enough dedup to justify overhead
        SIRIUS_LOG_DEBUG("P1b guard: skip (dedup_ratio={:.3f} > threshold={:.2f})",
                         dedup_ratio, p1b_dedup_threshold);
    }
}

if (!skip_p1b) {
    // Apply P1b mixed aggregate split...
}
// else: fall through to regular groupby with make_nunique_aggregation
```

### Getting `cd_col_ndv` (COUNT DISTINCT input column's distinct count)

**Option B (recommended):** Add `agg_input_stats` to `LogicalAggregate` in
`propagate_aggregate.cpp`, parallel to the existing `group_stats`. This is ~20 lines:

In `duckdb/src/optimizer/statistics/operator/propagate_aggregate.cpp`, after line 85
where `group_stats` is filled:
```cpp
// Fill agg_input_stats for each aggregate expression
for (auto &expr : aggr.expressions) {
    auto &bound_aggr = expr->Cast<BoundAggregateExpression>();
    if (!bound_aggr.children.empty()) {
        auto child_stats = PropagateExpression(bound_aggr.children[0]);
        aggr.agg_input_stats.push_back(child_stats ? child_stats->ToUnique() : nullptr);
    } else {
        aggr.agg_input_stats.push_back(nullptr);
    }
}
```

Then in `gpu_plan_aggregate.cpp`, access `op.agg_input_stats[i]->GetDistinctCount()`.

**Why not Option A:** `PropagateExpression` is a method on `StatisticsPropagator` which
runs during the optimization phase. By the time `gpu_plan_aggregate.cpp` runs (plan
generation phase), the `StatisticsPropagator` is no longer accessible. Only Option B works.

**Fallback without plumbing:** Use `size` as `cd_ndv` (worst case — assumes all values
unique). This makes the guard conservative: it will skip P1b more often than necessary,
but never cause a regression.

### Getting `estimated_output_groups`

Pass from `GPUPhysicalGroupedAggregate` into `cudf_groupby()` as a new parameter.
In `gpu_plan_aggregate.cpp`, `op.estimated_cardinality` is already available and accounts
for filter selectivity.

**P1b logic after the guard:**
1. Split aggregates: non-COUNT-DISTINCT aggs → regular groupby
2. COUNT DISTINCT aggs → `distinct(group_keys + cd_cols)` → `COUNT_STAR groupby`
3. Join results by group keys using `cudf::left_join` + `cudf::gather` + `cudf::scatter`

**Copy from:** `cudf-25.12-optimization:src/cuda/cudf/cudf_groupby.cu`,
lines 279–499 (the `// --- Mixed aggregate optimization ---` block).

**Required includes:**
```cpp
#include <cudf/stream_compaction.hpp>
#include <cudf/join/join.hpp>
#include <cudf/copying.hpp>
#include <cstdlib>
```

---

## Porting P3: STRLEN Offset Arithmetic

**What it does:** for `STRLEN(varchar_col)` where the input is a direct column reference
(not a computed expression), compute string lengths from the offsets array
(`offset[i+1] - offset[i]`) WITHOUT materializing the string column.

**Why it's faster:** The bottleneck is NOT `cudf::strings::count_bytes` (which internally
already uses offset arithmetic). The bottleneck is the string column **materialization**
that happens when `executor.Execute(*expr.children[0], ...)` builds a cudf column from the
raw chars buffer. For a 100M-row URL column, this reads ~10.5 GB of character data into a
cudf string column. P3 bypasses materialization entirely by reading the offsets array
directly from `GPUColumn::data_wrapper.offset` (only ~800 MB).

**Current dev-fused-kernel code** (`gpu_execute_function.cpp`, line ~566):
```cpp
case UnaryFunctionType::STRLEN:
    return cudf::strings::count_bytes(
        input->view(), executor.execution_stream, executor.resource_ref);
```

This materializes the string column via the UnaryFunctionDispatcher before calling
count_bytes. Replace with the fast path.

**Port these files from cudf-25.12-optimization:**
- `src/cuda/operator/strlen_from_offsets.cu` — CUDA kernel: `result[i] = offsets[i+1] - offsets[i]`
  (handles both direct access and row_id-based access for filtered inputs)
- `src/include/operator/strlen_from_offsets.cuh` — kernel declaration

**Modify** `gpu_execute_function.cpp` STRLEN handler. The existing code on
cudf-25.12-optimization (lines 776–794) is the correct pattern:
```cpp
else if (func_str == STRLEN_FUNC_STR) {
    // P3: fast path — compute STRLEN from offsets without materializing string column
    if (expr.children[0]->type == ExpressionType::BOUND_REF) {
        auto& ref = expr.children[0]->Cast<BoundReferenceExpression>();
        auto& col = input_columns[ref.index];
        if (col->data_wrapper.is_string_data && col->data_wrapper.offset != nullptr) {
            size_t num_rows = col->row_ids != nullptr ? col->row_id_count : col->column_length;
            return StrlenFromOffsets(col->data_wrapper.offset,
                                    col->row_ids,
                                    num_rows,
                                    execution_stream,
                                    resource_ref);
        }
    }
    // Fallback: materialize string column then count bytes
    UnaryFunctionDispatcher<UnaryFunctionType::STRLEN> dispatcher(*this);
    return dispatcher(expr, state);
}
```

**Key difference from Sonnet's original suggestion:** Do NOT call
`executor.Execute(*expr.children[0], ...)` to get a cudf column and then read its offsets.
That defeats the purpose — it materializes the string data. Instead, access
`input_columns[ref.index]->data_wrapper.offset` directly, which is the raw GPU pointer
to the offsets array without any materialization.

**Note on INT32 vs INT64 offsets:** cudf 25.12 used INT64 offsets; cudf 26.x may use
INT32 offsets for smaller strings. The `StrlenFromOffsets` kernel should be templated on
offset type. Check `data_wrapper.offset` pointer type at runtime.

---

## Implementing P5: Empty-String Filter via Offsets (New)

**What it does:** rewrite `varchar_col <> ''` to check `offset[i+1] - offset[i] > 0`,
avoiding full string column materialization for the non-emptiness check.

**No flag needed** — always correct (Arrow format invariant: empty string ↔ equal
consecutive offsets), only triggers on this specific pattern, always faster.

**Where to implement:** in `Execute(BoundComparisonExpression)` in
`gpu_execute_comparison.cpp`, BEFORE the switch statement that dispatches to
`ComparisonDispatcher`. This is critical — `ComparisonDispatcher::operator()` at line 140
calls `executor.Execute(*expr.left, ...)` which materializes the string column. P5 must
intercept before that happens.

**New files needed:**
- `src/cuda/operator/empty_str_check.cu` — kernel: `result[i] = (offsets[idx+1] - offsets[idx]) > 0`
  (handles both direct and row_id-based access, outputs BOOL8 column)
- `src/include/operator/empty_str_check.cuh`

**Pattern detection and implementation:**

Insert at `gpu_execute_comparison.cpp` line 219, before the switch:
```cpp
// P5: empty-string filter via offsets — avoid materializing string column
if (expr.GetExpressionType() == ExpressionType::COMPARE_NOTEQUAL &&
    expr.left->type == ExpressionType::BOUND_REF &&
    expr.right->GetExpressionClass() == ExpressionClass::BOUND_CONSTANT) {
    auto& right_val = expr.right->Cast<BoundConstantExpression>().value;
    if (right_val.type().id() == LogicalTypeId::VARCHAR &&
        !right_val.IsNull() &&
        right_val.GetValue<std::string>().empty()) {
        auto& ref = expr.left->Cast<BoundReferenceExpression>();
        auto& col = input_columns[ref.index];
        if (col->data_wrapper.is_string_data && col->data_wrapper.offset != nullptr) {
            size_t num_rows = col->row_ids != nullptr ? col->row_id_count : col->column_length;
            SIRIUS_LOG_DEBUG("P5: empty-string check via offsets ({} rows)", num_rows);
            return EmptyStrCheck(col->data_wrapper.offset,
                                col->row_ids,
                                num_rows,
                                execution_stream,
                                resource_ref);
        }
    }
}
```

**Also handle the symmetric case:** `'' <> varchar_col`. DuckDB usually normalizes
constants to the right, but check if this holds for all query patterns. If not, add the
mirror check for `expr.left` being the empty string constant.

---

## Re-Optimization Layer Design

### Problem

Sirius takes a physical plan from DuckDB via Substrait and executes it on GPU. The plan
was optimized by DuckDB's CPU-side optimizer which has no knowledge of:
- GPU memory bandwidth vs compute tradeoffs
- cudf algorithm constants (sort vs hash break-even)
- Actual post-filter cardinalities at runtime

This means some operator strategy decisions (e.g., P1b: two-phase COUNT DISTINCT vs
nunique) can only be made correctly with information DuckDB's optimizer doesn't have.

### Proposed Abstraction: `GPUOptimizerHints`

A struct that flows alongside the physical plan, populated at plan time from already-computed
DuckDB statistics (zero runtime overhead):

```cpp
struct GPUColumnHints {
    idx_t estimated_ndv;      // HLL-based distinct count from DuckDB stats
    Value min_val, max_val;   // NumericStats range if available
    bool ndv_is_post_filter;  // true if estimated_cardinality accounts for filter
};

struct GPUOptimizerHints {
    // Populated at plan time (zero runtime overhead)
    idx_t estimated_input_rows;    // op.estimated_cardinality of child
    idx_t estimated_output_groups; // op.estimated_cardinality of aggregate
    vector<GPUColumnHints> group_key_hints;  // from op.group_stats
    vector<GPUColumnHints> agg_input_hints;  // from agg_input_stats (Option B)

    // Populated at runtime (optional, Tier 2)
    optional<idx_t> actual_input_rows;  // set after filter, before groupby
};
```

### HLL Statistics Pipeline

DuckDB's statistics propagator uses HyperLogLog for distinct count estimation:

```
DistinctStatistics (HLL, sample-based)
  → BaseStatistics::distinct_count (via StatisticsPropagator::PropagateExpression)
  → op.group_stats[i]->GetDistinctCount()  ← available NOW for group key columns
```

`GetDistinctCount()` returns an HLL estimate — it IS the full HyperLogLog result, not
just a flag. The estimate has ~2-5% error typical of HLL with DuckDB's default precision.

For COUNT DISTINCT input columns (cd_col), the HLL ndv IS computed during statistics
propagation (`propagate_aggregate.cpp` lines 99–106) but stored in the internal
`statistics_map`, not on `LogicalAggregate`. Option B plumbing exposes it.

### Where It Lives

```
gpu_plan_aggregate.cpp
  → fills GPUOptimizerHints from op.group_stats + op.estimated_cardinality
  → stores on GPUPhysicalGroupedAggregate

GPUPhysicalGroupedAggregate::Execute()
  → passes hints to cudf_groupby()

cudf_groupby()
  → uses hints.estimated_output_groups + hints.agg_input_hints[i].estimated_ndv
  → makes P1b guard decision without any extra GPU pass
```

### Implementation Priority

1. `GPUOptimizerHints` struct definition (1 hour)
2. Fill `estimated_output_groups` + `group_key_hints` from existing `group_stats` (2 hours)
3. Use in P1b guard — replaces the current env-var-based approach (1 hour)
4. Fill `agg_input_hints` via Option B plumbing in `propagate_aggregate.cpp` (2 hours)
5. Tier 2 runtime collection at port boundaries (future, complex)

---

## Porting Checklist

```
[ ] cudf API audit: verify distinct/join/scatter/gather/hashing signatures in cudf 26.x
[ ] Verify cuCascade pipeline model: does groupby receive full input or batched chunks?
[ ] P1: port cudf_groupby.cu two-phase block + required includes
[ ] P1b: port mixed aggregate split + ADD cardinality guard + pass estimated_cardinality
[ ] P1b guard: pass estimated_output_groups from gpu_plan_aggregate.cpp to cudf_groupby
[ ] P1b guard: implement Option B (agg_input_stats) in propagate_aggregate.cpp (~20 lines)
[ ] P3: copy strlen_from_offsets.cu/.cuh + modify STRLEN dispatch in gpu_execute_function.cpp
[ ] P4: port string hash groupby block (optional, test at 100M rows first)
[ ] P5: implement empty_str_check.cu/.cuh + add pattern detection in gpu_execute_comparison.cpp
[ ] GPUOptimizerHints: define struct, fill from group_stats, wire through to cudf_groupby
[ ] Run ClickBench on GH200 100M rows after each step to verify no regression
[ ] Run TPC-H 22 queries to verify correctness (benchmark_results/run_tpch.sh)
```
