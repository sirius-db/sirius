# Porting Optimizations to dev-fused-kernel

Branch: `cudf-25.12-optimization` → `dev-fused-kernel` (main project)

---

## Status of Each Optimization

| Optimization | cudf-25.12-optimization | dev-fused-kernel | Action |
|---|---|---|---|
| P1: COUNT DISTINCT two-phase | ✅ committed (d65ab00) | ❌ missing — uses `make_nunique_aggregation` | Port |
| P1b: Mixed aggregate split | ✅ committed (863c90e) | ❌ missing | Port (with cardinality guard — see below) |
| P3: STRLEN offset arithmetic | ✅ committed (67a12c8) | ❌ missing — uses `cudf::strings::count_bytes` which reads full chars | Port |
| P4: String GROUP BY hash fingerprint | ✅ committed (2634ce9), env var toggle | ❌ missing | Port (optional, only if testing shows benefit at 100M+ rows) |
| P5: Empty-string filter via offsets | 🔲 not yet implemented | ❌ missing | Implement directly on dev-fused-kernel |

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
- `cudf::hashing::xxhash_64` — verify table_view overload exists

---

## Porting P1: COUNT DISTINCT Two-Phase

**What it does:** replaces `make_nunique_aggregation` with `distinct(group_key + cd_col)` → `COUNT_STAR groupby`. Avoids cudf's sort-based nunique for pure COUNT DISTINCT queries.

**File:** `src/cuda/cudf/cudf_groupby.cu`

**Where to insert:** before the `cudf::groupby::groupby grpby_obj(...)` call.

**Condition to trigger:**
```cpp
bool all_count_distinct = true;
for (int agg = 0; agg < num_aggregates; agg++) {
    if (agg_mode[agg] != AggregationType::COUNT_DISTINCT) {
        all_count_distinct = false; break;
    }
}
if (all_count_distinct && num_aggregates == 1) { /* P1 path */ }
```

**P1 path logic:**
1. Build `table_view` of {group_key_cols..., cd_col}
2. Call `cudf::distinct(table_view, keep_first, null_equal, ...)`
3. Build new keys_table from first `num_keys` columns of distinct result
4. Build COUNT_STAR groupby on the deduplicated table
5. Convert INT32 count result to INT64 (Sirius uses INT64 for counts)

**Copy from:** `cudf-25.12-optimization:src/cuda/cudf/cudf_groupby.cu`, lines ~180–390
(the `// --- P1: COUNT DISTINCT two-phase ---` block)

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

**WARNING:** P1b caused a regression on Q09 (100M rows, no filter). The guard below is
mandatory — do not port P1b without it.

**Cardinality guard — skip P1b when the distinct step won't reduce data:**

```cpp
// Guard: P1b only helps when either:
//   (a) input is small (overhead always negligible), OR
//   (b) input is large but expected deduplication is high
//
// size_threshold = K * estimated_cardinality, where K is measured via
// benchmark_results/calibrate_p1b.py. Default K=100 (conservative).
// Override: SIRIUS_P1B_K env var.
//
// dedup_threshold: if expected_distinct_pairs / size > threshold, skip P1b.
// Override: SIRIUS_P1B_DEDUP_THRESHOLD env var (0.0–1.0, default 0.8).

static idx_t p1b_K = []() -> idx_t {
    const char* env = std::getenv("SIRIUS_P1B_K");
    return env ? std::stoull(env) : 100ULL;
}();
static double p1b_dedup_threshold = []() -> double {
    const char* env = std::getenv("SIRIUS_P1B_DEDUP_THRESHOLD");
    return env ? std::stod(env) : 0.8;
}();

idx_t size_threshold = p1b_K * estimated_cardinality;
if (size > size_threshold) {
    // Large input: check expected dedup ratio
    idx_t group_ndv = estimated_cardinality;
    idx_t cd_ndv = cd_col_stats_distinct_count;  // from planner, or type max as fallback
    double dedup_ratio = static_cast<double>(std::min(group_ndv * cd_ndv, size)) /
                         static_cast<double>(size);
    if (dedup_ratio > p1b_dedup_threshold) {
        // Skip P1b — not enough deduplication to justify overhead
        goto regular_path;
    }
}
// Apply P1b ...
```

**For `cd_ndv` without extra plumbing:** use `NumericStats::HasMinMax` to get range
as an upper bound. If no stats, use size as worst case (most conservative).

**For `estimated_cardinality`:** pass it from `GPUPhysicalGroupedAggregate` into
`cudf_groupby` as a new parameter. In `gpu_plan_aggregate.cpp`, `op.estimated_cardinality`
is available at plan time.

**P1b logic after the guard:**
1. Split aggregates: non-COUNT-DISTINCT aggs → regular groupby
2. COUNT DISTINCT aggs → `distinct(group_keys + cd_cols)` → `COUNT_STAR groupby`
3. Join results by group keys using `cudf::left_join` + `cudf::scatter`

**Copy from:** `cudf-25.12-optimization:src/cuda/cudf/cudf_groupby.cu`,
lines ~390–500 (the `// --- P1b mixed aggregate optimization ---` block).

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
(not a computed expression), compute `offset[i+1] - offset[i]` instead of materializing
the string column. Saves reading the full char buffer (10.5 GB → 800 MB at 100M rows).

**Current dev-fused-kernel code** (`gpu_execute_function.cpp`, line ~566):
```cpp
case UnaryFunctionType::STRLEN:
    return cudf::strings::count_bytes(
        input->view(), executor.execution_stream, executor.resource_ref);
```

`count_bytes` reads the full string chars buffer. Replace with offset arithmetic when
input is a direct column reference.

**Port these files from cudf-25.12-optimization:**
- `src/cuda/operator/strlen_from_offsets.cu` — CUDA kernel: `result[i] = offsets[i+1] - offsets[i]`
- `src/include/operator/strlen_from_offsets.cuh` — kernel declaration

**Modify** `gpu_execute_function.cpp` STRLEN case:
```cpp
case UnaryFunctionType::STRLEN: {
    // P3: if input is a direct VARCHAR column (not a computed temp), use offset arithmetic
    // to avoid materializing the full chars buffer.
    auto input_col = executor.Execute(*expr.children[0], state->child_states[0].get());
    if (input_col->type().id() == cudf::type_id::STRING) {
        auto col_view = input_col->view();
        auto offsets_child = col_view.child(0);  // INT64 offsets
        if (offsets_child.type().id() == cudf::type_id::INT64) {
            return strlen_from_offsets(
                offsets_child.data<int64_t>(), col_view.size(),
                executor.execution_stream, executor.resource_ref);
        }
    }
    return cudf::strings::count_bytes(
        input->view(), executor.execution_stream, executor.resource_ref);
}
```

**Note on INT32 vs INT64 offsets:** cudf 25.12 used INT64 offsets; cudf 26.x may use
INT32 offsets for smaller strings. Check `offsets_child.type().id()` at runtime and
handle both. The kernel template already handles this if parameterized.

---

## Implementing P5: Empty-String Filter via Offsets (New)

**What it does:** rewrite `varchar_col <> ''` to `offset[i+1] - offset[i] > 0`,
avoiding the full string scan for non-emptiness check.

**No flag needed** — always correct (Arrow format invariant), only triggers on this
specific pattern, always faster.

**Where to implement:** in the comparison expression dispatcher in
`gpu_execute_function.cpp`, detect `(VARCHAR column) != (empty string constant)`.

**New files needed:**
- `src/cuda/operator/empty_str_check.cu` — kernel: `result[i] = (offsets[i+1] - offsets[i]) > 0`
- `src/include/operator/empty_str_check.cuh`

**Pattern detection:**
```cpp
// In comparison expression handler:
if (op == ExpressionType::COMPARE_NOTEQUAL) {
    bool left_is_varchar = left->type().id() == cudf::type_id::STRING;
    bool right_is_empty_const = /* check right child is BoundConstantExpression("")  */;
    if (left_is_varchar && right_is_empty_const) {
        // use offset-based kernel
    }
}
```

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

A struct that flows alongside the physical plan, populated at two points:

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
    vector<GPUColumnHints> agg_input_hints;  // from propagated agg expr stats

    // Populated at runtime (optional, Tier 2)
    optional<idx_t> actual_input_rows;  // set after filter, before groupby
};
```

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

### Getting agg_input_hints (minor plumbing needed)

The HLL ndv for COUNT DISTINCT input columns IS computed by the statistics propagator
(`propagate_aggregate.cpp:99-106`) but stored in `statistics_map`, not on
`LogicalAggregate`. Two options:

**Option A (recommended):** in `gpu_plan_aggregate.cpp`, before calling `CreatePlan` on
the child, extract statistics for each aggregate input column by calling
`PropagateExpression` on `bound_aggr.children[0]` and storing in `agg_input_hints`.
This requires a `StatisticsPropagator` reference in the GPU plan generator — check if
one is available or can be passed in.

**Option B (simpler):** Add `agg_input_stats` to `LogicalAggregate` (parallel to
`group_stats`) in `propagate_aggregate.cpp`. This is ~20 lines and gives the cleanest
interface.

### Would the hints layer slow down other queries?

**No** — `GPUOptimizerHints` is populated at plan time from already-computed DuckDB
statistics. It adds zero runtime overhead. It is purely a struct that carries existing
information (that was being discarded) through to the execution layer.

The optional Tier 2 (`actual_input_rows`) would require one extra GPU atomic at pipeline
port boundaries, but only when enabled and only for operators that flagged uncertainty.

### Implementation Priority

1. `GPUOptimizerHints` struct definition (1 hour)
2. Fill `estimated_output_groups` + `group_key_hints` from existing `group_stats` (2 hours)
3. Use in P1b guard — replaces the current env-var-based approach (1 hour)
4. Fill `agg_input_hints` via Option B plumbing (2 hours) — enables accurate cd_ndv
5. Tier 2 runtime collection at port boundaries (future, complex)

---

## Porting Checklist

```
[ ] cudf API audit: verify distinct/join/scatter/hashing signatures in cudf 26.x
[ ] P1: port cudf_groupby.cu two-phase block + required includes
[ ] P1b: port mixed aggregate split + add cardinality guard + pass estimated_cardinality
[ ] P3: copy strlen_from_offsets.cu/.cuh + modify STRLEN dispatch in gpu_execute_function.cpp
[ ] P4: port string hash groupby block (optional, test at 100M rows first)
[ ] P5: implement empty_str_check.cu/.cuh + add pattern detection in comparison dispatch
[ ] GPUOptimizerHints: define struct, fill from group_stats, wire through to cudf_groupby
[ ] agg_input_hints: Option B plumbing in propagate_aggregate.cpp
[ ] Run ClickBench on GH200 100M rows after each step to verify no regression
[ ] Run TPC-H 22 queries to verify correctness (benchmark_results/run_tpch.sh)
```
