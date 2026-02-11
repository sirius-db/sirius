# Limitations

Current limitations and known issues in Sirius. This document helps set appropriate expectations and guides workaround strategies.

## Table of Contents
- [SQL Feature Limitations](#sql-feature-limitations)
- [Data Type Limitations](#data-type-limitations)
- [Operator Limitations](#operator-limitations)
- [Scale Limitations](#scale-limitations)
- [Performance Limitations](#performance-limitations)
- [Hardware Limitations](#hardware-limitations)

---

## SQL Feature Limitations

### 1. Window Functions

**Status**: ⚠️ Partial Support (New Mode Only)

**Supported**:
```sql
SELECT * FROM gpu_execution('
    SELECT customer_id, order_date,
           ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date) as rn
    FROM orders
');
```

**Not Supported**:
- Complex frame specifications (ROWS BETWEEN)
- RANGE windows
- Multiple window specifications

**Workaround**: Fallback to DuckDB for unsupported window functions

### 2. Common Table Expressions (CTEs)

**Status**: ⚠️ Partial Support

**Supported**: Simple CTEs
```sql
SELECT * FROM gpu_execution('
    WITH sales_summary AS (
        SELECT customer_id, SUM(amount) as total
        FROM sales
        GROUP BY customer_id
    )
    SELECT * FROM sales_summary WHERE total > 1000
');
```

**Not Supported**:
- Recursive CTEs
- Multiple CTEs with complex dependencies
- CTEs with nested subqueries

**Workaround**: Materialize intermediate results as temp tables

### 3. Subqueries

**Status**: ⚠️ Limited Support

**Supported**: Correlated scalar subqueries (simple cases)
```sql
SELECT * FROM gpu_execution('
    SELECT order_id, amount,
           (SELECT AVG(amount) FROM orders) as avg_amount
    FROM orders
');
```

**Not Supported**:
- Complex correlated subqueries
- Subqueries in WHERE EXISTS
- Multiple levels of nesting

**Workaround**: Rewrite as joins or materialize subqueries

### 4. Set Operations

**Status**: ❌ Not Supported

**Not Available**:
- `UNION` / `UNION ALL`
- `INTERSECT`
- `EXCEPT`

**Workaround**: Use DuckDB for set operations, then process result

### 5. INSERT/UPDATE/DELETE

**Status**: ❌ Not Supported

Sirius is **read-only** (OLAP workloads only).

**Not Available**:
- `INSERT INTO`
- `UPDATE`
- `DELETE`
- `MERGE`

**Design**: Sirius is optimized for analytical queries, not transactions

---

## Data Type Limitations

### 1. Integer Types

**Status**: ✅ Fully Supported

All integer types supported:
- `TINYINT` (INT8)
- `SMALLINT` (INT16)
- `INTEGER` (INT32)
- `BIGINT` (INT64)

**Exception**: `HUGEINT` (INT128) → **Downcasted to BIGINT**

**Impact**: Precision loss for values > 2^63-1

```sql
-- HUGEINT aggregate might overflow
SELECT SUM(very_large_column) FROM huge_table;
-- Result might be incorrect if sum > BIGINT_MAX
```

**Workaround**: Use DECIMAL for large sums

### 2. Floating Point

**Status**: ✅ Fully Supported

- `FLOAT` (32-bit)
- `DOUBLE` (64-bit)

**Caveat**: IEEE 754 floating point precision limits apply

### 3. Decimal Types

**Status**: ⚠️ Limited Support

**Supported**:
- `DECIMAL(p, s)` where p ≤ 18 (DECIMAL64)
- `DECIMAL(p, s)` where 19 ≤ p ≤ 38 (DECIMAL128)

**Not Supported**:
- Very high precision decimals (p > 38)
- Some complex decimal operations

**Performance**: Decimals are slower than native types (BIGINT, DOUBLE)

### 4. String Types

**Status**: ✅ Supported with Caveats

**Supported**:
- `VARCHAR` / `STRING`
- Basic string operations (comparison, concatenation)

**Performance Considerations**:
- Strings are slower than numeric types
- Very long strings (> 1MB) may cause memory pressure
- String operations not as optimized as cuDF numerics

**Best Practice**: Use fixed-width types when possible

### 5. Date/Time Types

**Status**: ✅ Mostly Supported

**Supported**:
- `DATE`
- `TIMESTAMP`
- Basic date arithmetic

**Not Supported**:
- `TIME` (without date)
- `INTERVAL` types
- Complex timezone handling

**Workaround**: Convert to TIMESTAMP or use epoch seconds

### 6. Complex Types

**Status**: ❌ Not Supported

**Not Available**:
- `LIST` / `ARRAY`
- `STRUCT`
- `MAP`
- `JSON`
- `BLOB`

**Workaround**: Flatten complex types before querying

```sql
-- Bad: Won't work
SELECT * FROM gpu_execution('
    SELECT list_col[1] FROM table_with_lists
');

-- Good: Flatten first
CREATE TABLE flattened AS
    SELECT UNNEST(list_col) as value FROM table_with_lists;

SELECT * FROM gpu_execution('
    SELECT value FROM flattened
');
```

---

## Operator Limitations

### 1. Join Types

**Status**: ⚠️ Partial Support

**Supported**:
- `INNER JOIN`
- `LEFT JOIN`
- `RIGHT JOIN` (converted to LEFT JOIN)
- `CROSS JOIN` (as nested loop)

**Not Supported**:
- `FULL OUTER JOIN`
- `SEMI JOIN`
- `ANTI JOIN`

**Workaround**: Use DuckDB or rewrite with supported joins

### 2. Join Conditions

**Supported**:
- Equi-joins: `ON a.id = b.id`
- Multi-column equi-joins: `ON a.id = b.id AND a.type = b.type`

**Limited Support**:
- Non-equi joins: `ON a.value < b.value`
  - Fallback to nested loop (slow for large tables)

**Not Supported**:
- Complex expressions in join condition
- OR conditions in joins

### 3. Aggregate Functions

**Supported**:
- `COUNT`, `COUNT(DISTINCT)`
- `SUM`, `AVG`, `MIN`, `MAX`
- `STDDEV`, `VARIANCE`

**Not Supported**:
- `MEDIAN`
- `MODE`
- `PERCENTILE_CONT`, `PERCENTILE_DISC`
- Custom aggregate functions

**Workaround**: Use APPROX functions or fallback to DuckDB

### 4. String Functions

**Supported**:
- Basic: `LOWER`, `UPPER`, `SUBSTRING`, `LENGTH`
- Comparison: `=`, `!=`, `<`, `>`, `LIKE`

**Limited Support**:
- Regular expressions (slow)
- Complex string manipulation

**Workaround**: Process strings on CPU when possible

---

## Scale Limitations

### 1. GPU Memory Constraints

**Hard Limit**: GPU memory size (8-80GB typical)

**Impact**:
- Single data batch cannot exceed GPU memory
- Hash tables limited by GPU memory

**Mitigation**: New Mode spills to HOST/DISK automatically

**Example**:
```
16GB GPU:
- Can process ~12GB data at once (with overhead)
- Larger datasets spill to host memory
- Very large datasets (> 100GB) spill to disk
```

### 2. Maximum Table Size

**Practical Limit**: ~1-10TB depending on:
- GPU memory
- Host memory
- Disk space
- Query complexity

**Tested**: TPC-H SF1000 (1TB) works with spilling enabled

**Workaround for Larger**:
- Partition data
- Process in chunks
- Use distributed system (future)

### 3. String Length Limits

**Soft Limit**: 1MB per string

**Impact**:
- Very long strings consume significant GPU memory
- String operations slower with long strings

**Recommendation**: Keep strings < 1KB when possible

### 4. Number of Columns

**Practical Limit**: ~1000 columns

**Impact**:
- Wide tables increase memory overhead
- Column pruning becomes critical

**Best Practice**: Use column pruning aggressively

---

## Performance Limitations

### 1. Small Query Overhead

**Issue**: GPU kernel launch overhead (~5-10μs per kernel)

**Impact**: Queries on small datasets (< 100K rows) may be slower than CPU

**Guideline**:
- **< 10K rows**: Always use CPU
- **10K-100K rows**: Depends on complexity
- **> 100K rows**: GPU likely beneficial
- **> 1M rows**: GPU strongly recommended

### 2. Cold Start Time

**Issue**: First GPU query initializes CUDA context (~100-500ms)

**Impact**: First query slower than subsequent queries

**Mitigation**:
- Warm up with dummy query
- Reuse connections

**Example**:
```python
# Warm up
conn.execute("SELECT * FROM gpu_execution('SELECT 1')").fetchall()

# Now real queries are fast
for query in queries:
    conn.execute(f"SELECT * FROM gpu_execution('{query}')").fetchall()
```

### 3. Data Transfer Bottleneck

**Issue**: PCIe bandwidth limits CPU ↔ GPU transfers

**Bandwidth**:
- PCIe 3.0 x16: ~16 GB/s
- PCIe 4.0 x16: ~32 GB/s

**Impact**: Large data transfers take time

**Mitigation**:
- Minimize transfers (keep intermediate results on GPU)
- Aggregate before transferring
- Use pipelined transfers (New Mode does this)

### 4. String Operations

**Issue**: String operations slower than numeric operations

**Impact**: Queries with heavy string processing may not benefit much

**Best Practice**:
- Use numeric keys when possible
- Pre-process strings on CPU if needed
- Avoid complex string functions in tight loops

---

## Hardware Limitations

### 1. NVIDIA GPUs Only

**Limitation**: Requires NVIDIA GPU with CUDA support

**Not Supported**:
- AMD GPUs
- Intel GPUs
- Apple M-series GPUs
- Integrated GPUs

**Reason**: Built on CUDA/cuDF which requires NVIDIA

### 2. Compute Capability 7.0+

**Requirement**: Volta architecture or newer (2017+)

**Supported GPUs**:
- ✅ Volta (V100)
- ✅ Turing (RTX 20xx, T4)
- ✅ Ampere (A100, RTX 30xx)
- ✅ Ada Lovelace (RTX 40xx)
- ✅ Hopper (H100)

**Not Supported**:
- ❌ Pascal (GTX 10xx) - older than CC 7.0
- ❌ Maxwell, Kepler - too old

### 3. Linux Only

**Supported**:
- ✅ Ubuntu 20.04+
- ✅ RHEL 8+
- ✅ CentOS 8+
- ✅ Debian 11+

**Not Supported**:
- ❌ Windows
- ❌ macOS

**Reason**: DuckDB extension build system and CUDA toolkit requirements

### 4. Single GPU

**Current Limitation**: Sirius uses only one GPU

**Impact**: Cannot scale beyond single GPU memory/compute

**Future**: Multi-GPU support planned (see [Roadmap](roadmap.md))

---

## Known Issues

### 1. Memory Estimation

**Issue**: Some operators underestimate memory requirements

**Impact**: Occasional OOM even with spilling enabled

**Workaround**: Reduce `gpu_memory_limit` to trigger earlier spilling

### 2. Null Handling

**Issue**: Some operators have edge cases with NULL values

**Status**: Being addressed

**Workaround**: Filter NULLs explicitly when possible

### 3. Type Coercion

**Issue**: Automatic type coercion may fail in some cases

**Example**:
```sql
-- May fail if types don't match exactly
SELECT * FROM gpu_execution('
    SELECT CASE WHEN x > 0 THEN x ELSE 0.5 END
    FROM table
');
```

**Workaround**: Explicit CAST

### 4. Error Messages

**Issue**: Some error messages not as clear as they could be

**Status**: Improving error reporting in progress

**Workaround**: Check logs for more details

---

## Unsupported SQL Features

Quick reference of SQL features **not supported**:

| Feature | Status | Alternative |
|---------|--------|-------------|
| INSERT/UPDATE/DELETE | ❌ Not supported | Use DuckDB |
| FULL OUTER JOIN | ❌ Not supported | Split into LEFT + RIGHT |
| Recursive CTEs | ❌ Not supported | Materialize intermediate |
| UNION/INTERSECT/EXCEPT | ❌ Not supported | Use DuckDB |
| Window functions (complex) | ⚠️ Limited | Simple windows only |
| LIST/STRUCT/JSON types | ❌ Not supported | Flatten first |
| User-defined functions | ❌ Not supported | Use built-ins |
| Stored procedures | ❌ Not supported | N/A |
| Triggers | ❌ Not supported | N/A |
| Transactions | ❌ Not supported | Read-only |

---

## Fallback Behavior

When Sirius encounters unsupported features:

**With Fallback Enabled** (`ENABLE_FALLBACK=true`):
```sql
SET enable_gpu_fallback = true;

SELECT * FROM gpu_execution('
    SELECT * FROM t1 UNION SELECT * FROM t2
');
-- Automatically falls back to DuckDB CPU execution
```

**With Fallback Disabled** (`ENABLE_FALLBACK=false`):
```sql
SET enable_gpu_fallback = false;

SELECT * FROM gpu_execution('
    SELECT * FROM t1 UNION SELECT * FROM t2
');
-- Error: UNION not supported
```

**Recommendation**: Enable fallback during development, disable in production to catch unsupported queries early.

---

## Workaround Strategies

### Strategy 1: Hybrid Execution

Process on GPU where beneficial, use CPU otherwise:

```sql
-- Heavy lifting on GPU
CREATE TEMP TABLE gpu_result AS
    SELECT * FROM gpu_execution('
        SELECT customer_id, SUM(amount) as total
        FROM large_orders
        GROUP BY customer_id
    ');

-- Complex operations on CPU
SELECT *
FROM gpu_result
WHERE customer_id IN (
    SELECT DISTINCT customer_id FROM small_customers
);
```

### Strategy 2: Preprocessing

Transform data to GPU-friendly format:

```python
# Flatten nested structures
df = df.explode('list_column')

# Cast complex types
df['json_col'] = df['json_col'].astype(str)

# Write to Parquet
df.to_parquet('preprocessed.parquet')

# Now query with Sirius
conn.execute("""
    SELECT * FROM gpu_execution('
        SELECT * FROM read_parquet(''preprocessed.parquet'')
    ')
""")
```

### Strategy 3: Incremental Processing

For very large datasets:

```sql
-- Process in date ranges
FOR date_range IN date_ranges:
    CREATE TEMP TABLE chunk AS
        SELECT * FROM gpu_execution('
            SELECT * FROM huge_table
            WHERE date BETWEEN ''' || date_range.start || '''
              AND ''' || date_range.end || '''
        ');

    -- Accumulate results
    INSERT INTO final_results SELECT * FROM chunk;
```

---

## Reporting Issues

If you encounter a limitation not listed here:

1. **Check logs**: `SET sirius_log_level = 'DEBUG'`
2. **Verify version**: Ensure using latest Sirius
3. **Test fallback**: Does DuckDB handle it?
4. **Report**: File issue with:
   - Query that fails
   - Error message
   - Sirius version
   - GPU model

---

## See Also

- [Roadmap](roadmap.md) - Planned features addressing some limitations
- [Performance Tips](performance-tips.md) - Optimize within current limitations
- [API Reference](../08-reference/api-reference.md) - Supported APIs
- [Debugging Guide](../07-development/debugging.md) - Troubleshoot issues
