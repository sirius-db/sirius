# Correctness Baseline: CPU vs GPU (pre-optimization)

## Test: DuckDB CPU vs Sirius GPU on 10M rows, cudf-25.12-optimization branch

### Matching queries (31 of 43):
Q00-Q02, Q04-Q16, Q19-Q20, Q23, Q25-Q30, Q33-Q37, Q41

### Known pre-existing differences (12 queries):

| Query | Category | Notes |
|-------|----------|-------|
| Q03 | **Precision** | FP precision diff: `2.5131007489380997e+18` vs `2.5131007489381e+18`. Acceptable. |
| Q17 | **Tie-breaking** | `LIMIT 10` no `ORDER BY` — any 10 rows valid. Different row selection. Acceptable. |
| Q18 | **Tie-breaking** | `ORDER BY COUNT(*) DESC LIMIT 10` — tied counts produce different row selection. Acceptable. |
| Q21 | **Tie-breaking** | `ORDER BY c DESC LIMIT 10` — tied counts produce different row selection. Acceptable. |
| Q22 | **Data diff** | `COUNT(DISTINCT UserID)` values differ. Needs investigation. |
| Q24 | **Data diff** | `SELECT *` result differs. Needs investigation. |
| Q31 | **Tie-breaking** | `ORDER BY c DESC LIMIT 10` on WatchID (unique), all c=1. Any 10 rows valid. Acceptable. |
| Q32 | **Tie-breaking** | Same as Q31 but without WHERE filter. Acceptable. |
| Q38 | **GPU returns 0 rows** | CPU returns data, GPU returns 0 rows. `OFFSET 1000` query. Known GPU OFFSET bug? |
| Q39 | **GPU returns fewer rows** | `OFFSET 1000` query. Same issue as Q38. |
| Q40 | **GPU returns 0 rows** | `OFFSET 100` query. Same issue. |
| Q42 | **GPU returns 0 rows** | `OFFSET 1000` query. Same issue. |

### Summary of categories:
- **FP precision (1)**: Q03 — acceptable
- **Tie-breaking in ORDER BY LIMIT (5)**: Q17, Q18, Q21, Q31, Q32 — acceptable (no deterministic order)
- **OFFSET bug (4)**: Q38, Q39, Q40, Q42 — GPU returns 0/fewer rows for queries with OFFSET. Pre-existing bug.
- **Data differences (2)**: Q22, Q24 — need investigation but are pre-existing

### Validation rule for optimizations:
After each optimization, compare against **DuckDB CPU golden output** (`duckdb_cpu_golden_10m.txt`).
- The 31 matching queries MUST still match exactly
- The 12 pre-existing differences should not get worse (same category of mismatch)
- Any NEW mismatch = optimization introduced a bug → reject
