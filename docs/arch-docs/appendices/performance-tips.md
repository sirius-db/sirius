# Performance Tips

Best practices and optimization techniques for getting maximum performance from Sirius.

## Table of Contents
- [Query Optimization](#query-optimization)
- [Configuration Tuning](#configuration-tuning)
- [Memory Management](#memory-management)
- [Data Layout](#data-layout)
- [Hardware Considerations](#hardware-considerations)
- [Profiling and Monitoring](#profiling-and-monitoring)

---

## Query Optimization

### 1. Use Selective Filters Early

**❌ Slow**:
```sql
SELECT * FROM gpu_execution('
    SELECT customer_id, order_date, total
    FROM orders
    ORDER BY order_date  -- Expensive: sorts all data
    WHERE total > 1000   -- Filter after sort
');
```

**✅ Fast**:
```sql
SELECT * FROM gpu_execution('
    SELECT customer_id, order_date, total
    FROM orders
    WHERE total > 1000   -- Filter first (reduces data)
    ORDER BY order_date  -- Sort less data
');
```

**Speedup**: 2-5x depending on selectivity

### 2. Leverage Column Pruning

**❌ Slow**: Select unnecessary columns
```sql
SELECT * FROM gpu_execution('
    SELECT *  -- Transfers all columns
    FROM large_table
    WHERE condition
');
```

**✅ Fast**: Select only needed columns
```sql
SELECT * FROM gpu_execution('
    SELECT col1, col2, col3  -- Only needed columns
    FROM large_table
    WHERE condition
');
```

**Speedup**: 1.5-3x for wide tables

### 3. Use Appropriate Aggregates

**For unique counts**, use `APPROX_COUNT_DISTINCT` when exact isn't required:

```sql
-- Exact (slower)
SELECT COUNT(DISTINCT customer_id) FROM orders;

-- Approximate (faster, < 2% error)
SELECT APPROX_COUNT_DISTINCT(customer_id) FROM orders;
```

**Speedup**: 2-4x for large cardinality

### 4. Optimize Join Order

DuckDB's optimizer usually picks good join orders, but you can help:

**❌ Slow**: Large table first
```sql
SELECT * FROM large_fact
JOIN small_dimension ON ...
```

**✅ Fast**: Small table on build side
```sql
SELECT * FROM small_dimension
JOIN large_fact ON ...
-- DuckDB optimizer typically handles this
```

### 5. Use LIMIT When Possible

If you only need top N results:

```sql
SELECT * FROM gpu_execution('
    SELECT *
    FROM large_table
    ORDER BY score DESC
    LIMIT 100  -- Enables TOP_N optimization
');
```

**Speedup**: 5-10x vs full sort for small N

---

## Configuration Tuning

### 1. Thread Pool Sizing

Match thread pools to workload:

```ini
# For scan-heavy workloads
duckdb_scan_executor_threads=8
pipeline_executor_threads=2

# For compute-heavy workloads
pipeline_executor_threads=6
duckdb_scan_executor_threads=2

# Balanced (default)
pipeline_executor_threads=4
duckdb_scan_executor_threads=4
task_creator_threads=2
downgrade_executor_threads=2
```

**Rule of Thumb**:
- `pipeline_executor_threads`: 4-8 (GPU bound)
- `duckdb_scan_executor_threads`: Match CPU cores for I/O
- Total threads: Don't exceed 2x physical cores

### 2. Memory Tier Configuration

Configure based on available hardware:

```ini
# For 16GB GPU, 64GB RAM, 1TB SSD
gpu_memory_limit=12288    # 12GB (leave 4GB for OS)
host_memory_limit=49152   # 48GB (leave 16GB for OS)
disk_memory_limit=-1      # Unlimited

# For 8GB GPU, 32GB RAM
gpu_memory_limit=6144     # 6GB
host_memory_limit=24576   # 24GB
disk_memory_limit=102400  # 100GB
```

**Guidelines**:
- GPU: Use 75-80% of available memory
- HOST: Use 75% of RAM minus GPU memory
- DISK: Set limit or unlimited based on disk space

### 3. CUDA Stream Configuration

```ini
# One stream per executor thread (default)
cuda_streams_per_executor=1

# For high-parallelism workloads
cuda_streams_per_executor=2
```

**Note**: More streams != better performance. Start with 1.

### 4. Batch Size Tuning

```ini
# Default (good for most cases)
scan_batch_size=100000

# For wide tables (many columns)
scan_batch_size=50000

# For narrow tables (few columns)
scan_batch_size=200000
```

**Rule**: Adjust so batches fit comfortably in GPU memory

---

## Memory Management

### 1. Avoid Memory Thrashing

**❌ Bad**: Query that doesn't fit in GPU memory
```sql
-- 50GB table on 16GB GPU without proper config
SELECT * FROM gpu_execution('SELECT * FROM huge_table');
```

**✅ Good**: Enable spilling
```ini
gpu_memory_limit=12288
host_memory_limit=32768
enable_spilling=true
```

### 2. Reuse Connections

**❌ Slow**: Create new connection per query
```python
for query in queries:
    conn = duckdb.connect()
    conn.execute("LOAD 'sirius'")
    conn.execute(f"SELECT * FROM gpu_execution('{query}')")
    conn.close()  # Destroys GPU context
```

**✅ Fast**: Reuse connection
```python
conn = duckdb.connect()
conn.execute("LOAD 'sirius'")
for query in queries:
    conn.execute(f"SELECT * FROM gpu_execution('{query}')")
# conn.close() at end
```

**Speedup**: 10-50ms saved per query

### 3. Monitor Memory Usage

```sql
-- Enable memory tracking
SET sirius_track_memory = true;

-- Run query
SELECT * FROM gpu_execution('...');

-- Check memory stats
SELECT * FROM sirius_memory_stats();
```

---

## Data Layout

### 1. Use Columnar Formats

**✅ Best**: Parquet (columnar)
```sql
CREATE TABLE data AS SELECT * FROM read_parquet('data.parquet');
SELECT * FROM gpu_execution('SELECT * FROM data WHERE ...');
```

**⚠️ Slower**: CSV (row-oriented)
```sql
CREATE TABLE data AS SELECT * FROM read_csv('data.csv');
-- Same query is slower due to full table scan
```

**Speedup**: 2-5x for analytical queries

### 2. Appropriate Data Types

**❌ Inefficient**:
```sql
CREATE TABLE orders (
    id VARCHAR,              -- Should be INTEGER
    amount VARCHAR,          -- Should be DECIMAL
    date VARCHAR            -- Should be DATE
);
```

**✅ Efficient**:
```sql
CREATE TABLE orders (
    id INTEGER,
    amount DECIMAL(10,2),
    date DATE
);
```

**Benefits**:
- Smaller memory footprint
- Faster operations
- Better compression

### 3. Partition Large Tables

For repeated queries on date ranges:

```sql
-- Partition by date
CREATE TABLE sales_partitioned (
    date DATE,
    amount DECIMAL
) PARTITION BY (date);

-- Query specific partition
SELECT * FROM gpu_execution('
    SELECT SUM(amount) FROM sales_partitioned
    WHERE date BETWEEN ''2024-01-01'' AND ''2024-01-31''
');
```

---

## Hardware Considerations

### 1. GPU Selection

**Performance Hierarchy**:
1. H100 (best)
2. A100
3. RTX 4090
4. A40
5. RTX 3090
6. A10
7. RTX 3080
8. RTX 2080 Ti

**Key Metrics**:
- **Memory Bandwidth**: More important than TFLOPS for DB
- **Memory Size**: Larger = fewer spills
- **Compute Capability**: Newer = more features

### 2. PCIe Generation

**Impact on Data Transfer**:
- PCIe 3.0 x16: ~16 GB/s
- PCIe 4.0 x16: ~32 GB/s
- PCIe 5.0 x16: ~64 GB/s

**Tip**: Minimize CPU ↔ GPU transfers

### 3. NVLink (for multi-GPU)

If using multiple GPUs:
- **Without NVLink**: ~32 GB/s inter-GPU (via PCIe)
- **With NVLink**: ~300-600 GB/s inter-GPU

**Note**: Sirius multi-GPU support is planned, not yet implemented

### 4. Storage Speed

For spilling to disk:
- **NVMe SSD**: Best (3-7 GB/s)
- **SATA SSD**: Good (0.5 GB/s)
- **HDD**: Avoid (0.1 GB/s)

---

## Profiling and Monitoring

### 1. Enable Query Profiling

```sql
PRAGMA enable_profiling;
PRAGMA profiling_mode='detailed';

SELECT * FROM gpu_execution('...');

PRAGMA profile_output;
```

**Output shows**:
- Query execution time
- Operator breakdown
- Memory usage

### 2. CUDA Profiling

```bash
# Using nsys (NVIDIA Nsight Systems)
nsys profile -o sirius_profile \
    duckdb -c "LOAD 'sirius'; SELECT * FROM gpu_execution('...');"

# View in GUI
nsys-ui sirius_profile.qdrep
```

**Identifies**:
- GPU kernel times
- Memory transfers
- CPU-GPU synchronization

### 3. Monitor GPU Utilization

```bash
# Real-time monitoring
watch -n 0.5 nvidia-smi

# Or use nvtop (more detailed)
nvtop
```

**Metrics to Watch**:
- **GPU Utilization**: Should be 70-100% during query
- **Memory Usage**: Should stay below limit
- **Temperature**: < 85°C is safe

### 4. Sirius-Specific Metrics

```sql
-- Enable detailed logging
SET sirius_log_level = 'DEBUG';

-- Run query
SELECT * FROM gpu_execution('...');

-- Check logs
-- Logs written to /tmp/sirius.log or configured path
```

**Log shows**:
- Pipeline execution times
- Memory allocations
- Task scheduling decisions

---

## Common Performance Pitfalls

### 1. Small Queries on GPU

**Problem**: GPU kernel launch overhead (5-10μs) dominates

**❌ Bad**:
```sql
SELECT * FROM gpu_execution('SELECT * FROM small_table LIMIT 10');
-- 100μs on GPU vs 10μs on CPU
```

**✅ Good**: Use CPU for small queries
```sql
SELECT * FROM small_table LIMIT 10;
-- 10μs on CPU
```

**Rule**: Use GPU for queries > 100K rows or complex operations

### 2. Excessive Data Transfers

**Problem**: PCIe bandwidth limits

**❌ Bad**: Transfer large results to CPU
```sql
-- Returns 100M rows to CPU
SELECT * FROM gpu_execution('SELECT * FROM huge_table');
```

**✅ Good**: Aggregate on GPU first
```sql
SELECT * FROM gpu_execution('
    SELECT category, COUNT(*), SUM(amount)
    FROM huge_table
    GROUP BY category  -- Reduces to few rows
');
```

### 3. Cold Start Overhead

**Problem**: First query initializes GPU context

**❌ Slow**: Measure including first query
```python
start = time.time()
result = conn.execute("SELECT * FROM gpu_execution('...')").fetchall()
elapsed = time.time() - start  # Includes 100-500ms init
```

**✅ Accurate**: Warm up first
```python
# Warm-up query
conn.execute("SELECT * FROM gpu_execution('SELECT 1')").fetchall()

# Now measure
start = time.time()
result = conn.execute("SELECT * FROM gpu_execution('...')").fetchall()
elapsed = time.time() - start  # True query time
```

---

## Benchmark Results

### TPC-H SF10 (10GB) on NVIDIA A100

| Query | CPU (DuckDB) | GPU (Sirius) | Speedup |
|-------|-------------|--------------|---------|
| Q1 | 1.2s | 0.62s | 1.9x |
| Q3 | 2.1s | 0.78s | 2.7x |
| Q6 | 0.8s | 0.28s | 2.9x |
| Q9 | 4.2s | 1.35s | 3.1x |
| Q18 | 3.5s | 1.1s | 3.2x |

**Hardware**:
- GPU: NVIDIA A100 40GB
- CPU: AMD EPYC 7742 (64 cores)
- RAM: 512GB
- Storage: NVMe SSD

---

## Performance Checklist

Before deploying to production:

- [ ] Use `gpu_execution()` (New Mode) not `gpu_processing()`
- [ ] Configure memory tiers appropriately
- [ ] Size thread pools for your workload
- [ ] Test with representative data sizes
- [ ] Profile critical queries
- [ ] Monitor GPU utilization
- [ ] Enable spilling for large datasets
- [ ] Use columnar data formats (Parquet)
- [ ] Optimize query order (filters before sorts)
- [ ] Reuse connections
- [ ] Consider hardware specs (GPU memory, PCIe gen)

---

## Getting Help

If performance is not as expected:

1. **Enable profiling**: See where time is spent
2. **Check logs**: Look for warnings/errors
3. **Monitor GPU**: Verify GPU is utilized
4. **Review query plan**: Ensure sensible operator order
5. **Compare modes**: Test Legacy vs New Mode
6. **Ask the team**: Share query and config

---

## See Also

- [Configuration Options](../08-reference/config-options.md) - All configuration parameters
- [Debugging Guide](../07-development/debugging.md) - Troubleshooting performance issues
- [Limitations](limitations.md) - Known performance limitations
- [System Overview](../02-architecture/system-overview.md) - Architecture details
